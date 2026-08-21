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

"""Behavioral taint-tracking analyzer (TT1–TT5): sources -> sinks data-flow analysis.

Parses Python AST to identify data sources (env vars, file reads, network input)
and sinks (network output, exec, file writes), then tracks flows between them
to flag potential credential/data exfiltration chains.
"""

from __future__ import annotations

import ast
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

from skillspector.inspection_ledger import (
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    analyzer_status_event,
    analyzer_status_for_events,
    ledger_event,
)
from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Finding, Location, Severity
from skillspector.python_ast import ParsedPythonFile, get_python_ast
from skillspector.state import (
    AnalyzerNodeResponse,
    SkillspectorState,
    transitive_remaining_seconds,
)

from .common import (
    apply_import_aliases,
    build_type_map,
    get_complete_source_segment,
    get_context_from_lines,
    resolve_call_name_typed,
    resolve_dotted_name,
    resolve_dynamic_import_call,
)
from .static_runner import (
    MAX_FILE_CHARS,
    MAX_FINDINGS_PER_ANALYZER,
    MAX_FINDINGS_PER_ARTIFACT,
    analyzer_finding_to_finding,
)

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

# Deserializers that reconstruct arbitrary objects / execute code on their input.
# When untrusted data (network, user, or a bundled/downloaded file) reaches one of
# these, it is an RCE-class flow — the deserialization analogue of _EXEC_SINKS.
# Only unconditionally-unsafe names are listed; argument-dependent forms
# (yaml.load / torch.load / numpy.load) are handled by behavioral_ast (AST10) where
# keyword arguments can be inspected without false positives on the hardened forms.
_DESERIALIZATION_SINKS = frozenset(
    {
        "pickle.load",
        "pickle.loads",
        "cPickle.load",
        "cPickle.loads",
        "_pickle.load",
        "_pickle.loads",
        "marshal.load",
        "marshal.loads",
        "dill.load",
        "dill.loads",
        "jsonpickle.decode",
        "pandas.read_pickle",
        "joblib.load",
        "yaml.unsafe_load",
    }
)

_ALL_SINKS = _NETWORK_OUTPUT_SINKS | _EXEC_SINKS | _FILE_WRITE_SINKS | _DESERIALIZATION_SINKS

# Pre-computed for _pick_rule — avoids rebuilding the union on every call.
_EXTERNAL_INPUT_SOURCES = _NETWORK_INPUT_SOURCES | _USER_INPUT_SOURCES

_RULE_SEVERITIES: dict[str, Severity] = {
    "TT1": Severity.HIGH,
    "TT2": Severity.MEDIUM,
    "TT3": Severity.CRITICAL,
    "TT4": Severity.HIGH,
    "TT5": Severity.CRITICAL,
    "TT6": Severity.HIGH,
}

_RULE_CONFIDENCES: dict[str, float] = {
    "TT1": 0.80,
    "TT2": 0.65,
    "TT3": 0.90,
    "TT4": 0.80,
    "TT5": 0.90,
    "TT6": 0.85,
}

_TAG = "Data Flow"


class _BehavioralResourceLimitError(RuntimeError):
    """Internal signal that retains findings constructed before a hard limit."""

    def __init__(self, reason: LedgerReason, metrics: dict[str, int | float]) -> None:
        super().__init__(reason.value)
        self.reason = reason
        self.metrics = metrics


@dataclass
class _BehavioralBudget:
    """Bound taint work while findings are being constructed, not afterwards."""

    state: SkillspectorState
    started_at: float = field(default_factory=time.monotonic)
    initial_allowance: float | None = None
    total_findings: int = 0
    current_findings: list[AnalyzerFinding] = field(default_factory=list)

    def begin_artifact(self) -> None:
        self.current_findings = []
        self.check_runtime()

    def check_runtime(self) -> None:
        remaining = transitive_remaining_seconds(self.state)
        if remaining is None:
            return
        if self.initial_allowance is None:
            self.initial_allowance = max(0.0, remaining)
        if remaining <= 0:
            raise _BehavioralResourceLimitError(
                LedgerReason.RUNTIME_LIMIT,
                {
                    "observed_seconds": max(0.0, time.monotonic() - self.started_at),
                    "limit_seconds": self.initial_allowance,
                },
            )

    def emit(self, finding: AnalyzerFinding) -> None:
        self.check_runtime()
        artifact_observed = len(self.current_findings) + 1
        analyzer_observed = self.total_findings + 1
        if artifact_observed > MAX_FINDINGS_PER_ARTIFACT:
            raise _BehavioralResourceLimitError(
                LedgerReason.OUTPUT_LIMIT,
                {
                    "observed_findings": artifact_observed,
                    "limit_findings": MAX_FINDINGS_PER_ARTIFACT,
                },
            )
        if analyzer_observed > MAX_FINDINGS_PER_ANALYZER:
            raise _BehavioralResourceLimitError(
                LedgerReason.OUTPUT_LIMIT,
                {
                    "observed_findings": analyzer_observed,
                    "limit_findings": MAX_FINDINGS_PER_ANALYZER,
                },
            )
        self.current_findings.append(finding)
        self.total_findings = analyzer_observed

    def analyzer_exhausted(self) -> bool:
        return self.total_findings >= MAX_FINDINGS_PER_ANALYZER


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
    (_DESERIALIZATION_SINKS, "deserialization"),
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
    if sink_name in _DESERIALIZATION_SINKS and (
        source_name in _EXTERNAL_INPUT_SOURCES or source_name in _FILE_READ_SOURCES
    ):
        return "TT6"
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
    check_runtime: Callable[[], None] | None = None,
) -> str | None:
    """Find a source call anywhere in an expression tree (handles chained calls).

    Handles patterns like ``open("f").read()``, ``requests.get(url).text``,
    and plain ``os.environ.get("K")``.
    """
    for child in ast.walk(node):
        if check_runtime is not None:
            check_runtime()
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
    check_runtime: Callable[[], None] | None = None,
) -> list[tuple[str, ast.Call]]:
    """Walk children to find source calls nested inside a sink call."""
    results: list[tuple[str, ast.Call]] = []
    for child in ast.walk(node):
        if check_runtime is not None:
            check_runtime()
        if child is node:
            continue
        if not isinstance(child, ast.Call):
            continue
        name = resolve_call_name_typed(child, type_map, aliases)
        if name and name in _ALL_SOURCES:
            results.append((name, child))
    return results


def _find_tainted_names_in_args(
    node: ast.Call,
    tainted: dict[str, _TaintedVar],
    check_runtime: Callable[[], None] | None = None,
) -> list[_TaintedVar]:
    """Find references to tainted variables in a call's arguments and keywords."""
    seen: set[str] = set()
    hits: list[_TaintedVar] = []
    for child in ast.walk(node):
        if check_runtime is not None:
            check_runtime()
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


def _find_tainted_in_expr(
    node: ast.expr,
    tainted: dict[str, _TaintedVar],
    check_runtime: Callable[[], None] | None = None,
) -> _TaintedVar | None:
    """Return the first tainted variable referenced in *node*, or None.

    Handles Name references, container literals (dict, list, tuple, set),
    and f-strings so that taint propagates through re-assignment and
    data packaging (e.g. ``payload = {"key": secret}``).
    """
    for child in ast.walk(node):
        if check_runtime is not None:
            check_runtime()
        if isinstance(child, ast.Name):
            tv = tainted.get(child.id)
            if tv:
                return tv
    return None


def _analyze_python(
    python_ast: ParsedPythonFile,
    file_path: str,
    budget: _BehavioralBudget | None = None,
) -> list[AnalyzerFinding]:
    tree = python_ast.tree
    if tree is None:
        return []

    aliases = python_ast.import_aliases
    type_map = build_type_map(tree, aliases)
    lines = python_ast.lines
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
        complete_match = get_complete_source_segment(lines, lineno, end_lineno)
        finding = AnalyzerFinding(
            rule_id=rule_id,
            message=msg,
            severity=_RULE_SEVERITIES[rule_id],
            location=Location(file=file_path, start_line=lineno, end_line=end_lineno),
            confidence=_RULE_CONFIDENCES[rule_id],
            tags=[_TAG],
            context=get_context_from_lines(lines, lineno),
            matched_text=complete_match[:200],
            complete_match=complete_match,
        )
        if budget is None:
            findings.append(finding)
        else:
            budget.emit(finding)

    for ast_node in ast.walk(tree):
        if budget is not None:
            budget.check_runtime()
        # Record tainted assignments.
        if isinstance(ast_node, ast.Assign):
            src_name = _find_source_in_expr(
                ast_node.value,
                type_map,
                aliases,
                budget.check_runtime if budget is not None else None,
            )

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
                tv = _find_tainted_in_expr(
                    ast_node.value,
                    tainted,
                    budget.check_runtime if budget is not None else None,
                )
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

        for src_name, src_node in _find_nested_sources(
            ast_node,
            type_map,
            aliases,
            budget.check_runtime if budget is not None else None,
        ):
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

        for tv in _find_tainted_names_in_args(
            ast_node,
            tainted,
            budget.check_runtime if budget is not None else None,
        ):
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

    return findings if budget is None else list(budget.current_findings)


def _partial_limit_event(
    path: str,
    limit: _BehavioralResourceLimitError,
    *,
    emitted_finding_ids: list[str] | None = None,
) -> InspectionLedgerEvent:
    """Account one current or unstarted Python work item as explicitly partial."""
    return ledger_event(
        outcome=LedgerOutcome.PARTIAL,
        phase="behavioral",
        analyzer_id=ANALYZER_ID,
        path=path,
        reason=limit.reason,
        emitted_finding_ids=emitted_finding_ids or (),
        observed_findings=(
            int(limit.metrics["observed_findings"])
            if limit.reason is LedgerReason.OUTPUT_LIMIT
            else None
        ),
        limit_findings=(
            int(limit.metrics["limit_findings"])
            if limit.reason is LedgerReason.OUTPUT_LIMIT
            else None
        ),
        observed_seconds=(
            float(limit.metrics["observed_seconds"])
            if limit.reason is LedgerReason.RUNTIME_LIMIT
            else None
        ),
        limit_seconds=(
            float(limit.metrics["limit_seconds"])
            if limit.reason is LedgerReason.RUNTIME_LIMIT
            else None
        ),
    )


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Parse Python files and detect source\u2192sink data flows."""
    components: list[str] = state.get("components") or []
    file_cache: dict[str, str] = state.get("local_file_cache") or state.get("file_cache") or {}
    python_ast_cache_key = state.get("python_ast_cache_key")
    all_findings: list[Finding] = []
    ledger_events: list[InspectionLedgerEvent] = []
    budget = _BehavioralBudget(state)
    terminal_limit: _BehavioralResourceLimitError | None = None

    for path in components:
        if not path.endswith(".py"):
            continue
        if terminal_limit is None and budget.analyzer_exhausted():
            terminal_limit = _BehavioralResourceLimitError(
                LedgerReason.OUTPUT_LIMIT,
                {
                    "observed_findings": budget.total_findings + 1,
                    "limit_findings": MAX_FINDINGS_PER_ANALYZER,
                },
            )
        if terminal_limit is not None:
            event = _partial_limit_event(path, terminal_limit)
            ledger_events.append(event)
            continue
        content = file_cache.get(path)
        if content is None:
            event = ledger_event(
                outcome=LedgerOutcome.FAILED,
                phase="behavioral",
                analyzer_id=ANALYZER_ID,
                path=path,
                reason=LedgerReason.MISSING_FILE_CACHE,
            )
        elif len(content) > MAX_FILE_CHARS:
            event = ledger_event(
                outcome=LedgerOutcome.PARTIAL,
                phase="behavioral",
                analyzer_id=ANALYZER_ID,
                path=path,
                reason=LedgerReason.SIZE_LIMIT,
                observed_characters=len(content),
                limit_characters=MAX_FILE_CHARS,
                observed_bytes=len(content.encode("utf-8")),
            )
        else:
            budget.current_findings = []
            resource_limit: _BehavioralResourceLimitError | None = None
            python_ast: ParsedPythonFile | None = None
            try:
                budget.begin_artifact()
                python_ast = get_python_ast(python_ast_cache_key, content, path)
                budget.check_runtime()
                if python_ast.is_parseable:
                    _analyze_python(python_ast, path, budget)
            except _BehavioralResourceLimitError as exc:
                resource_limit = exc

            path_findings = [analyzer_finding_to_finding(af) for af in budget.current_findings]
            all_findings.extend(path_findings)
            if resource_limit is not None:
                event = _partial_limit_event(
                    path,
                    resource_limit,
                    emitted_finding_ids=[finding.finding_id for finding in path_findings],
                )
                if (
                    resource_limit.reason is LedgerReason.RUNTIME_LIMIT
                    or budget.analyzer_exhausted()
                ):
                    terminal_limit = resource_limit
            elif python_ast is None or not python_ast.is_parseable:
                event = ledger_event(
                    outcome=LedgerOutcome.SKIPPED,
                    phase="behavioral",
                    analyzer_id=ANALYZER_ID,
                    path=path,
                    reason=LedgerReason.SYNTAX_ERROR,
                )
            else:
                event = ledger_event(
                    outcome=LedgerOutcome.COMPLETED,
                    phase="behavioral",
                    analyzer_id=ANALYZER_ID,
                    path=path,
                    emitted_finding_ids=[finding.finding_id for finding in path_findings],
                )
        ledger_events.append(event)

    logger.info("%s: %d findings", ANALYZER_ID, len(all_findings))
    if not ledger_events:
        status = analyzer_status_event(
            analyzer_id=ANALYZER_ID,
            status="not_applicable",
            reason=LedgerReason.NO_APPLICABLE_FILES,
        )
    else:
        status = analyzer_status_for_events(ANALYZER_ID, ledger_events)
    return {
        "findings": all_findings,
        "inspection_ledger": ledger_events,
        "analyzer_status_events": [status],
    }
