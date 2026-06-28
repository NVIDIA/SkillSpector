# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Behavioral taint-tracking analyzer (SADD B.2.2): sources -> sinks data-flow analysis.

Parses Python AST to identify data sources (env vars, file reads, network input)
and sinks (network output, exec, file writes), then tracks flows between them
to flag potential credential/data exfiltration chains.
"""

from __future__ import annotations

import ast
from typing import NamedTuple

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Finding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from .common import (
    apply_import_aliases,
    build_import_aliases,
    build_type_map,
    get_context_from_lines,
    get_source_segment,
    resolve_call_name_typed,
    resolve_dotted_name,
    resolve_dynamic_import_call,
)
from .static_runner import MAX_FILE_BYTES, analyzer_finding_to_finding

ANALYZER_ID = "behavioral_taint_tracking"
logger = get_logger(__name__)

_CREDENTIAL_SOURCES = frozenset(
    {
        "os.environ.get",
        "os.environ",
        "os.getenv",
    }
)

_FILE_READ_SOURCES = frozenset(
    {
        "open",
        "pathlib.Path.read_text",
        "pathlib.Path.read_bytes",
    }
)

_NETWORK_INPUT_SOURCES = frozenset(
    {
        "requests.get",
        "requests.post",
        "requests.put",
        "requests.patch",
        "requests.delete",
        "httpx.get",
        "httpx.post",
        "httpx.put",
        "httpx.patch",
        "httpx.delete",
        "urllib.request.urlopen",
        "urllib.request.urlretrieve",
        "socket.socket.recv",
        "socket.socket.recvfrom",
    }
)

_USER_INPUT_SOURCES = frozenset(
    {
        "input",
        "sys.stdin.read",
        "sys.stdin.readline",
    }
)

_ALL_SOURCES = (
    _CREDENTIAL_SOURCES | _FILE_READ_SOURCES | _NETWORK_INPUT_SOURCES | _USER_INPUT_SOURCES
)

_NETWORK_OUTPUT_SINKS = frozenset(
    {
        "requests.post",
        "requests.put",
        "requests.patch",
        "requests.get",
        "httpx.post",
        "httpx.put",
        "httpx.patch",
        "httpx.get",
        "urllib.request.urlopen",
        "socket.socket.send",
        "socket.socket.sendall",
        "socket.socket.sendto",
    }
)

_EXEC_SINKS = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "os.system",
        "os.popen",
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_output",
        "subprocess.check_call",
        "subprocess.Popen",
    }
)

_FILE_WRITE_SINKS = frozenset(
    {
        "open",
        "pathlib.Path.write_text",
        "pathlib.Path.write_bytes",
        "shutil.copy",
        "shutil.copy2",
        "shutil.copyfile",
    }
)

_ALL_SINKS = _NETWORK_OUTPUT_SINKS | _EXEC_SINKS | _FILE_WRITE_SINKS

# Pre-computed for _pick_rule — avoids rebuilding the union on every call.
_EXTERNAL_INPUT_SOURCES = _NETWORK_INPUT_SOURCES | _USER_INPUT_SOURCES

_RULE_SEVERITIES: dict[str, Severity] = {
    "TT1": Severity.HIGH,
    "TT2": Severity.MEDIUM,
    "TT3": Severity.CRITICAL,
    "TT4": Severity.HIGH,
    "TT5": Severity.CRITICAL,
}

_RULE_CONFIDENCES: dict[str, float] = {
    "TT1": 0.80,
    "TT2": 0.65,
    "TT3": 0.90,
    "TT4": 0.80,
    "TT5": 0.90,
}

_TAG = "Data Flow"

_SOURCE_CATEGORIES: list[tuple[frozenset[str], str]] = [
    (_CREDENTIAL_SOURCES, "credential/environment"),
    (_FILE_READ_SOURCES, "file read"),
    (_NETWORK_INPUT_SOURCES, "network input"),
    (_USER_INPUT_SOURCES, "user input"),
]

_SINK_CATEGORIES: list[tuple[frozenset[str], str]] = [
    (_NETWORK_OUTPUT_SINKS, "network output"),
    (_EXEC_SINKS, "code execution"),
    (_FILE_WRITE_SINKS, "file write"),
]


def _resolve_sink_name(
    node: ast.Call,
    type_map: dict[str, str] | None = None,
    aliases: dict[str, str] | None = None,
) -> str | None:
    """Resolve a call to its canonical sink name, including dynamic-import chains.

    Wraps :func:`resolve_call_name_typed` (type-/alias-aware resolution) and falls back
    to :func:`resolve_dynamic_import_call` so that
    ``importlib.import_module('subprocess').run(...)`` resolves to ``'subprocess.run'``
    and re-enters ``_EXEC_SINKS`` like the statically-imported form would.
    """
    name = resolve_call_name_typed(node, type_map, aliases)
    if name is None:
        name = resolve_dynamic_import_call(node, aliases)
    return name


def _classify(name: str, categories: list[tuple[frozenset[str], str]], default: str) -> str:
    for names, label in categories:
        if name in names:
            return label
    return default


def _pick_rule(source_name: str, sink_name: str, is_direct: bool) -> str:
    """Choose the most specific rule ID for a source->sink pair."""
    if source_name in _CREDENTIAL_SOURCES and sink_name in _NETWORK_OUTPUT_SINKS:
        return "TT3"
    if source_name in _FILE_READ_SOURCES and sink_name in _NETWORK_OUTPUT_SINKS:
        return "TT4"
    if source_name in _EXTERNAL_INPUT_SOURCES and sink_name in _EXEC_SINKS:
        return "TT5"
    return "TT1" if is_direct else "TT2"


class _TaintedVar(NamedTuple):
    name: str
    source_call: str
    lineno: int


def _is_open_for_write(node: ast.Call) -> bool:
    """Heuristic: open() is a write sink if mode arg contains 'w' or 'a'."""
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        mode = str(node.args[1].value)
        return any(c in mode for c in "wa")
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            mode = str(kw.value.value)
            return any(c in mode for c in "wa")
    return False


def _find_source_in_expr(
    node: ast.expr,
    type_map: dict[str, str] | None = None,
    aliases: dict[str, str] | None = None,
) -> str | None:
    """Find a source call anywhere in an expression tree (handles chained calls).

    Handles patterns like ``open("f").read()``, ``requests.get(url).text``,
    and plain ``os.environ.get("K")``.
    """
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        name = resolve_call_name_typed(child, type_map, aliases)
        if name is None or name not in _ALL_SOURCES:
            continue
        if name == "open" and _is_open_for_write(child):
            continue
        return name
    return None


def _find_nested_sources(
    node: ast.Call,
    type_map: dict[str, str] | None = None,
    aliases: dict[str, str] | None = None,
) -> list[tuple[str, ast.Call]]:
    """Walk children to find source calls nested inside a sink call."""
    results: list[tuple[str, ast.Call]] = []
    for child in ast.walk(node):
        if child is node:
            continue
        if not isinstance(child, ast.Call):
            continue
        name = resolve_call_name_typed(child, type_map, aliases)
        if name and name in _ALL_SOURCES:
            results.append((name, child))
    return results


def _find_tainted_names_in_args(
    node: ast.Call, tainted: dict[str, _TaintedVar]
) -> list[_TaintedVar]:
    """Find references to tainted variables in a call's arguments and keywords."""
    seen: set[str] = set()
    hits: list[_TaintedVar] = []
    for child in ast.walk(node):
        if child is node:
            continue
        var_name: str | None = None
        if isinstance(child, ast.Name):
            var_name = child.id
        elif isinstance(child, ast.Subscript):
            var_name = resolve_dotted_name(child.value)
        if var_name and var_name not in seen:
            tv = tainted.get(var_name)
            if tv:
                seen.add(var_name)
                hits.append(tv)
    return hits


def _mark_targets(
    targets: list[ast.expr],
    tainted: dict[str, _TaintedVar],
    src_name: str,
    lineno: int,
) -> None:
    for target in targets:
        if isinstance(target, ast.Name):
            tainted[target.id] = _TaintedVar(target.id, src_name, lineno)
        elif isinstance(target, ast.Tuple):
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    tainted[elt.id] = _TaintedVar(elt.id, src_name, lineno)


def _find_tainted_in_expr(node: ast.expr, tainted: dict[str, _TaintedVar]) -> _TaintedVar | None:
    """Return the first tainted variable referenced in *node*, or None.

    Handles Name references, container literals (dict, list, tuple, set),
    and f-strings so that taint propagates through re-assignment and
    data packaging (e.g. ``payload = {"key": secret}``).
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            tv = tainted.get(child.id)
            if tv:
                return tv
    return None


def _analyze_python(content: str, file_path: str) -> list[AnalyzerFinding]:
    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        logger.debug("SyntaxError parsing %s, skipping", file_path)
        return []

    type_map = build_type_map(tree)
    aliases = build_import_aliases(tree)
    lines = content.splitlines()
    findings: list[AnalyzerFinding] = []
    tainted: dict[str, _TaintedVar] = {}
    seen: set[tuple[str, int]] = set()

    def _emit(
        rule_id: str,
        lineno: int,
        end_lineno: int | None,
        msg: str,
    ) -> None:
        key = (rule_id, lineno)
        if key in seen:
            return
        seen.add(key)
        findings.append(
            AnalyzerFinding(
                rule_id=rule_id,
                message=msg,
                severity=_RULE_SEVERITIES[rule_id],
                location=Location(file=file_path, start_line=lineno, end_line=end_lineno),
                confidence=_RULE_CONFIDENCES[rule_id],
                tags=[_TAG],
                context=get_context_from_lines(lines, lineno),
                matched_text=get_source_segment(lines, lineno, end_lineno),
            )
        )

    for ast_node in ast.walk(tree):
        # Record tainted assignments.
        if isinstance(ast_node, ast.Assign):
            src_name = _find_source_in_expr(ast_node.value, type_map, aliases)

            # Subscript sources like os.environ["KEY"] (also os aliased as `o`)
            if src_name is None and isinstance(ast_node.value, ast.Subscript):
                base = resolve_dotted_name(ast_node.value.value)
                if base is not None:
                    base = apply_import_aliases(base, aliases)
                if base and base in _CREDENTIAL_SOURCES:
                    src_name = base

            # Propagate taint through re-assignment and container construction:
            # data = secret, payload = {"k": secret}, items = [secret], msg = f"{secret}"
            if src_name is None:
                tv = _find_tainted_in_expr(ast_node.value, tainted)
                if tv:
                    src_name = tv.source_call

            if src_name:
                _mark_targets(ast_node.targets, tainted, src_name, ast_node.lineno)
            continue

        # Detect flows at sink call sites.
        if not isinstance(ast_node, ast.Call):
            continue

        sink_name = _resolve_sink_name(ast_node, type_map, aliases)
        if not sink_name or sink_name not in _ALL_SINKS:
            continue

        if sink_name == "open" and not _is_open_for_write(ast_node):
            continue

        lineno = getattr(ast_node, "lineno", 1)
        end_lineno = getattr(ast_node, "end_lineno", None)

        for src_name, src_node in _find_nested_sources(ast_node, type_map, aliases):
            if src_name == "open" and _is_open_for_write(src_node):
                continue
            rule = _pick_rule(src_name, sink_name, is_direct=True)
            src_cat = _classify(src_name, _SOURCE_CATEGORIES, "data source")
            sink_cat = _classify(sink_name, _SINK_CATEGORIES, "data sink")
            _emit(
                rule,
                lineno,
                end_lineno,
                f"Direct flow: {src_name} ({src_cat}) \u2192 {sink_name} ({sink_cat})",
            )

        for tv in _find_tainted_names_in_args(ast_node, tainted):
            rule = _pick_rule(tv.source_call, sink_name, is_direct=False)
            src_cat = _classify(tv.source_call, _SOURCE_CATEGORIES, "data source")
            sink_cat = _classify(sink_name, _SINK_CATEGORIES, "data sink")
            _emit(
                rule,
                lineno,
                end_lineno,
                f"Tainted flow: '{tv.name}' from {tv.source_call} (line {tv.lineno}, "
                f"{src_cat}) \u2192 {sink_name} ({sink_cat})",
            )

    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Parse Python files and detect source\u2192sink data flows."""
    components: list[str] = state.get("components") or []
    file_cache: dict[str, str] = state.get("file_cache") or {}
    all_findings: list[Finding] = []

    for path in components:
        if not path.endswith(".py"):
            continue
        content = file_cache.get(path)
        if content is None or len(content) > MAX_FILE_BYTES:
            continue
        raw = _analyze_python(content, path)
        all_findings.extend(analyzer_finding_to_finding(af) for af in raw)

    logger.info("%s: %d findings", ANALYZER_ID, len(all_findings))
    return {"findings": all_findings}
