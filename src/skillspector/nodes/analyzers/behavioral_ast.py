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
import time
from collections.abc import Callable
from dataclasses import dataclass, field

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
    get_complete_source_segment,
    get_context_from_lines,
    resolve_call_name,
    resolve_dynamic_import_call,
)
from .static_runner import (
    MAX_FILE_CHARS,
    MAX_FINDINGS_PER_ANALYZER,
    MAX_FINDINGS_PER_ARTIFACT,
    analyzer_finding_to_finding,
)

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

# Deserializers that reconstruct arbitrary objects (or execute code) from their
# input, regardless of arguments. Feeding attacker-controlled bytes to any of
# these is equivalent to code execution: pickle invokes ``__reduce__`` during
# unpickling, ``yaml.unsafe_load`` constructs arbitrary Python objects, etc.
# ``yaml.load``/``torch.load``/``numpy.load`` are handled separately because
# their safety depends on arguments (see ``_deserialization_message``).
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

# Loader classes that make ``yaml.load`` safe (no arbitrary object construction).
_SAFE_YAML_LOADERS = frozenset({"SafeLoader", "CSafeLoader", "BaseLoader"})

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
    "AST10": "Insecure deserialization of untrusted data",
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
    "AST10": Severity.MEDIUM,
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
    "AST10": 0.70,
}

_TAG = "Dangerous Code Execution"


class _BehavioralResourceLimitError(RuntimeError):
    """Internal signal that retains findings constructed before a hard limit."""

    def __init__(self, reason: LedgerReason, metrics: dict[str, int | float]) -> None:
        super().__init__(reason.value)
        self.reason = reason
        self.metrics = metrics


@dataclass
class _BehavioralBudget:
    """Bound AST work while findings are being constructed, not after return."""

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


def _is_chain_sink(node: ast.Call, aliases: dict[str, str] | None = None) -> bool:
    """True if this call is exec(), eval(), or compile() — the outer dangerous call."""
    name = resolve_call_name(node, aliases)
    return name in ("exec", "eval", "compile")


def _contains_dangerous_source(
    node: ast.AST,
    aliases: dict[str, str] | None = None,
    check_runtime: Callable[[], None] | None = None,
) -> str | None:
    """Walk children to find a nested dangerous call that forms a chain."""
    for child in ast.walk(node):
        if check_runtime is not None:
            check_runtime()
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


def _loader_arg_name(node: ast.expr) -> str | None:
    """Return the trailing name of a yaml ``Loader`` argument (``yaml.SafeLoader`` → 'SafeLoader')."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _is_true_constant(node: ast.expr) -> bool:
    """True if *node* is the literal ``True``."""
    return isinstance(node, ast.Constant) and node.value is True


def _kwarg_is_true(node: ast.Call, name: str) -> bool:
    """True if keyword *name* is passed as a literal ``True``."""
    return any(kw.arg == name and _is_true_constant(kw.value) for kw in node.keywords)


def _deserialization_message(call_name: str, node: ast.Call) -> str | None:
    """Return an AST10 message if *node* is an unsafe deserialization call, else None.

    ``_DESERIALIZATION_SINKS`` are unconditionally unsafe. ``yaml.load``, ``torch.load``,
    and ``numpy.load`` are argument-dependent: an explicit safe ``Loader``,
    ``weights_only=True``, or the default ``allow_pickle=False`` respectively make them
    safe and must not be flagged (avoids false positives on the hardened forms).
    """
    if call_name in _DESERIALIZATION_SINKS:
        return f"Insecure deserialization: {call_name}()"
    if call_name == "yaml.load":
        for kw in node.keywords:
            if kw.arg == "Loader":
                if _loader_arg_name(kw.value) in _SAFE_YAML_LOADERS:
                    return None
                return "Insecure deserialization: yaml.load() with an unsafe Loader"
        if len(node.args) >= 2 and _loader_arg_name(node.args[1]) in _SAFE_YAML_LOADERS:
            return None
        return "Insecure deserialization: yaml.load() without SafeLoader"
    if call_name == "torch.load":
        return (
            None
            if _kwarg_is_true(node, "weights_only")
            else ("Insecure deserialization: torch.load() without weights_only=True")
        )
    if call_name == "numpy.load":
        # ``allow_pickle`` is the third positional parameter (file, mmap_mode,
        # allow_pickle, ...), so ``numpy.load(f, None, True)`` enables pickle
        # loading without ever naming the keyword.
        if _kwarg_is_true(node, "allow_pickle") or (
            len(node.args) >= 3 and _is_true_constant(node.args[2])
        ):
            return "Insecure deserialization: numpy.load(allow_pickle=True)"
        return None
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
    lines = python_ast.lines
    findings: list[AnalyzerFinding] = []

    def _emit(
        rule_id: str,
        ast_node: ast.Call,
        msg_override: str | None = None,
    ) -> None:
        lineno = getattr(ast_node, "lineno", 1)
        end_lineno = getattr(ast_node, "end_lineno", None)
        complete_match = ast.get_source_segment(python_ast.content, ast_node)
        if complete_match is None:
            complete_match = get_complete_source_segment(lines, lineno, end_lineno)
        start_column = getattr(ast_node, "col_offset", 0)
        end_column = getattr(ast_node, "end_col_offset", start_column)
        complete_identity = f"{complete_match}\x1f{start_column}:{end_column}"
        finding = AnalyzerFinding(
            rule_id=rule_id,
            message=msg_override or _RULE_MESSAGES[rule_id],
            severity=_RULE_SEVERITIES[rule_id],
            location=Location(file=file_path, start_line=lineno, end_line=end_lineno),
            confidence=_RULE_CONFIDENCES[rule_id],
            tags=[_TAG],
            context=get_context_from_lines(lines, lineno),
            matched_text=complete_match[:200],
            complete_match=complete_identity,
        )
        if budget is None:
            findings.append(finding)
        else:
            budget.emit(finding)

    for ast_node in ast.walk(tree):
        if budget is not None:
            budget.check_runtime()
        if not isinstance(ast_node, ast.Call):
            continue

        call_name = resolve_call_name(ast_node, aliases)
        if call_name is None:
            # Dynamic-import chain: importlib.import_module('os').system(...) →
            # 'os.system', so it re-enters the os./subprocess. sink ladders below.
            call_name = resolve_dynamic_import_call(ast_node, aliases)
        if call_name is None:
            continue

        if call_name == "exec":
            if _is_chain_sink(ast_node, aliases) and ast_node.args:
                source = _contains_dangerous_source(
                    ast_node.args[0],
                    aliases,
                    budget.check_runtime if budget is not None else None,
                )
                if source:
                    _emit("AST8", ast_node, f"Dangerous chain: exec() wrapping {source}")
            _emit("AST1", ast_node)

        elif call_name == "eval":
            if _is_chain_sink(ast_node, aliases) and ast_node.args:
                source = _contains_dangerous_source(
                    ast_node.args[0],
                    aliases,
                    budget.check_runtime if budget is not None else None,
                )
                if source:
                    _emit("AST8", ast_node, f"Dangerous chain: eval() wrapping {source}")
            _emit("AST2", ast_node)

        elif call_name == "__import__":
            _emit("AST3", ast_node)

        elif call_name == "compile":
            _emit("AST6", ast_node)

        elif call_name.startswith("subprocess."):
            attr = call_name.split(".", 1)[1]
            if attr in _SUBPROCESS_CALLS:
                _emit("AST4", ast_node)

        elif call_name.startswith("os."):
            attr = call_name.split(".", 1)[1]
            if attr in _OS_EXEC_CALLS:
                _emit("AST5", ast_node)

        elif (deser_msg := _deserialization_message(call_name, ast_node)) is not None:
            _emit("AST10", ast_node, deser_msg)

        elif call_name == "getattr" and len(ast_node.args) >= 2:
            second_arg = ast_node.args[1]
            if not isinstance(second_arg, ast.Constant):
                _emit("AST7", ast_node)
            elif isinstance(second_arg.value, str) and second_arg.value in _DANGEROUS_GETATTR_NAMES:
                _emit("AST9", ast_node)

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
    """Parse Python files via AST and detect dangerous execution patterns."""
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
