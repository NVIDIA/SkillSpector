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

"""Behavioral AST analyzer: detect dangerous execution patterns in Python code."""

from __future__ import annotations

import ast

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Finding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from .canonical_sink import (
    canonical_sibling_method,
    canonical_sibling_sink,
    resolve_to_canonical_sink,
)
from .common import (
    build_import_aliases,
    build_type_map,
    get_context_from_lines,
    get_source_segment,
    resolve_call_name,
    resolve_dynamic_import_call,
)
from .static_runner import MAX_FILE_CHARS, analyzer_finding_to_finding

ANALYZER_ID = "behavioral_ast"
logger = get_logger(__name__)

_DANGEROUS_BUILTINS = frozenset({"exec", "eval", "compile", "__import__"})

# Names that turn ``getattr(obj, "<name>")`` into a reflective handle on a code- or
# command-execution sink. ``getattr(os, "system")(cmd)`` and
# ``getattr(builtins, "exec")(src)`` are functionally identical to ``os.system(cmd)``
# / ``exec(src)`` but evade AST1/AST5: the inner ``getattr`` has a *constant* second
# argument (so AST7 is intentionally skipped), and the outer call's ``func`` is an
# ``ast.Call`` whose name does not resolve, so AST1/AST5 never fire. The set is kept
# deliberately small — only names with essentially no legitimate ``getattr`` use — so
# benign reflection such as ``getattr(obj, "name")`` stays unflagged.
_DANGEROUS_GETATTR_NAMES = frozenset({"exec", "eval", "system", "popen", "__import__"})

_SUBPROCESS_CALLS = frozenset(
    {
        "call",
        "run",
        "Popen",
        "check_output",
        "check_call",
        "getoutput",
        "getstatusoutput",
    }
)

_OS_EXEC_CALLS = frozenset(
    {
        "system",
        "popen",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "posix_spawn",
        "posix_spawnp",
    }
)

_RULE_MESSAGES: dict[str, str] = {
    "AST1": "exec() call detected",
    "AST2": "eval() call detected",
    "AST3": "Dynamic import via __import__()",
    "AST4": "subprocess module call",
    "AST5": "os.system() or os exec-family call",
    "AST6": "compile() call detected",
    "AST7": "Dynamic attribute access via getattr()",
    "AST8": "Dangerous execution chain",
    "AST9": "Reflective dangerous call via getattr() with a literal sink name",
}

_RULE_SEVERITIES: dict[str, Severity] = {
    "AST1": Severity.HIGH,
    "AST2": Severity.HIGH,
    "AST3": Severity.MEDIUM,
    "AST4": Severity.MEDIUM,
    "AST5": Severity.HIGH,
    "AST6": Severity.MEDIUM,
    "AST7": Severity.LOW,
    "AST8": Severity.CRITICAL,
    "AST9": Severity.HIGH,
}

_RULE_CONFIDENCES: dict[str, float] = {
    "AST1": 0.85,
    "AST2": 0.85,
    "AST3": 0.75,
    "AST4": 0.70,
    "AST5": 0.85,
    "AST6": 0.65,
    "AST7": 0.50,
    "AST8": 0.95,
    "AST9": 0.85,
}

_TAG = "Dangerous Code Execution"


def _covered_by_ast9(node: ast.Call) -> bool:
    """True when *node* invokes a literal ``getattr`` already flagged by AST9.

    AST9 catches ``getattr(obj, "<name>")(...)`` for ``<name>`` in
    :data:`_DANGEROUS_GETATTR_NAMES`. The canonical-sink fallback skips exactly those so
    the two rules stay complementary rather than double-firing: the canonical layer then
    owns only the spellings AST9 does not (subscript ``__builtins__["exec"]`` /
    ``vars(builtins)["exec"]`` and getattr targets outside the allowlist such as
    ``getattr(subprocess, "Popen")``).
    """
    callee = node.func
    if not (isinstance(callee, ast.Call) and isinstance(callee.func, ast.Name)):
        return False
    if callee.func.id != "getattr" or len(callee.args) < 2:
        return False
    attr = callee.args[1]
    return (
        isinstance(attr, ast.Constant)
        and isinstance(attr.value, str)
        and attr.value in _DANGEROUS_GETATTR_NAMES
    )


def _is_chain_sink(node: ast.Call, aliases: dict[str, str] | None = None) -> bool:
    """True if this call is exec(), eval(), or compile() — the outer dangerous call."""
    name = resolve_call_name(node, aliases)
    return name in ("exec", "eval", "compile")


def _contains_dangerous_source(node: ast.AST, aliases: dict[str, str] | None = None) -> str | None:
    """Walk children to find a nested dangerous call that forms a chain."""
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        name = resolve_call_name(child, aliases)
        if name is None:
            continue
        if name in ("compile", "__import__"):
            return name
        if name.startswith("subprocess.") or name.startswith("os."):
            return name
        if any(
            part in name for part in ("base64", "codecs", "marshal", "urllib", "requests", "httpx")
        ):
            return name
    return None


def _analyze_python(content: str, file_path: str) -> list[AnalyzerFinding]:
    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        logger.debug("SyntaxError parsing %s, skipping", file_path)
        return []

    aliases = build_import_aliases(tree)
    type_map = build_type_map(tree)
    lines = content.splitlines()
    findings: list[AnalyzerFinding] = []

    def _emit(
        rule_id: str,
        lineno: int,
        end_lineno: int | None,
        msg_override: str | None = None,
    ) -> None:
        findings.append(
            AnalyzerFinding(
                rule_id=rule_id,
                message=msg_override or _RULE_MESSAGES[rule_id],
                severity=_RULE_SEVERITIES[rule_id],
                location=Location(file=file_path, start_line=lineno, end_line=end_lineno),
                confidence=_RULE_CONFIDENCES[rule_id],
                tags=[_TAG],
                context=get_context_from_lines(lines, lineno),
                matched_text=get_source_segment(lines, lineno, end_lineno),
            )
        )

    for ast_node in ast.walk(tree):
        if not isinstance(ast_node, ast.Call):
            continue

        call_name = resolve_call_name(ast_node, aliases)
        if call_name is not None:
            # Dynamic-import / code-exec sibling machinery (importlib.__import__,
            # importlib.util.find_spec, runpy.run_module, code.interact, and the
            # instance-method tails spec.loader.exec_module / runsource) resolves by name
            # but matches no ladder; remap it to the primitive it equals so it re-enters
            # the __import__/exec arms below.
            call_name = (
                canonical_sibling_sink(call_name)
                or canonical_sibling_method(ast_node, type_map, aliases)
                or call_name
            )
        if call_name is None:
            # Dynamic-import chain: importlib.import_module('os').system(...) →
            # 'os.system', so it re-enters the os./subprocess. sink ladders below.
            call_name = resolve_dynamic_import_call(ast_node, aliases)
        reflective = False
        if call_name is None and not _covered_by_ast9(ast_node):
            # Reflective invocation whose callee does not resolve by name:
            # getattr(subprocess, "Popen")(cmd) (outside AST9's allowlist),
            # __builtins__["exec"](src), vars(builtins)["exec"](src). Canonicalize to
            # the bare/dotted sink id so it re-enters the ladders below with the correct
            # rule + severity. Literal getattr() on an AST9-allowlisted name is left to
            # AST9 (see _covered_by_ast9) so the two rules stay complementary, not
            # duplicate.
            call_name = resolve_to_canonical_sink(ast_node, aliases)
            reflective = call_name is not None
        if call_name is None:
            continue

        lineno = getattr(ast_node, "lineno", 1)
        end_lineno = getattr(ast_node, "end_lineno", None)

        if call_name == "exec":
            if _is_chain_sink(ast_node, aliases) and ast_node.args:
                source = _contains_dangerous_source(ast_node.args[0], aliases)
                if source:
                    _emit("AST8", lineno, end_lineno, f"Dangerous chain: exec() wrapping {source}")
            _emit("AST1", lineno, end_lineno)

        elif call_name == "eval":
            if _is_chain_sink(ast_node, aliases) and ast_node.args:
                source = _contains_dangerous_source(ast_node.args[0], aliases)
                if source:
                    _emit("AST8", lineno, end_lineno, f"Dangerous chain: eval() wrapping {source}")
            _emit("AST2", lineno, end_lineno)

        elif call_name == "__import__":
            _emit("AST3", lineno, end_lineno)

        elif call_name == "compile":
            _emit("AST6", lineno, end_lineno)

        elif call_name.startswith("subprocess."):
            attr = call_name.split(".", 1)[1]
            if attr in _SUBPROCESS_CALLS:
                # A *reflective* subprocess invocation (getattr/subscript) signals the
                # same evasion intent as reflective os.system, so it grades AST9-HIGH to
                # stay consistent with that case; a direct subprocess.* call keeps its
                # baseline AST4-MEDIUM.
                _emit("AST9" if reflective else "AST4", lineno, end_lineno)

        elif call_name.startswith("os."):
            attr = call_name.split(".", 1)[1]
            if attr in _OS_EXEC_CALLS:
                _emit("AST5", lineno, end_lineno)

        elif call_name == "getattr" and len(ast_node.args) >= 2:
            second_arg = ast_node.args[1]
            if not isinstance(second_arg, ast.Constant):
                _emit("AST7", lineno, end_lineno)
            elif isinstance(second_arg.value, str) and second_arg.value in _DANGEROUS_GETATTR_NAMES:
                _emit("AST9", lineno, end_lineno)

    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Parse Python files via AST and detect dangerous execution patterns."""
    components: list[str] = state.get("components") or []
    file_cache: dict[str, str] = state.get("file_cache") or {}
    all_findings: list[Finding] = []

    for path in components:
        if not path.endswith(".py"):
            continue
        content = file_cache.get(path)
        if content is None or len(content) > MAX_FILE_CHARS:
            continue
        raw = _analyze_python(content, path)
        all_findings.extend(analyzer_finding_to_finding(af) for af in raw)

    logger.info("%s: %d findings", ANALYZER_ID, len(all_findings))
    return {"findings": all_findings}
