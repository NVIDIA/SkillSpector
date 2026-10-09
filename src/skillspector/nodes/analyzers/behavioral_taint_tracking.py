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

"""Behavioral taint-tracking analyzer (TT1–TT6): sources -> sinks data-flow analysis.

Parses Python AST to identify data sources (env vars, file reads, network input)
and sinks (network output, exec, file writes), then tracks flows between them
to flag potential credential/data exfiltration chains. Taint follows Python's
lexical scopes and crosses direct calls to functions and methods defined in the
same file (arguments into parameters, return values back to callers).
"""

from __future__ import annotations

import ast
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
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
from skillspector.python_ast import (
    ParsedPythonFile,
    PythonSourceClassification,
    get_python_ast,
    resolve_python_source_classification,
)
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
    """One reported taint fact: data from ``source_call`` reaches ``name``.

    ``lineno`` is the line of the statement that last tainted ``name`` (the
    assignment, or the call that bound a parameter), the wording earlier
    releases used, so messages and message-based baselines keep matching.
    """

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


def _call_source(
    node: ast.Call,
    type_map: dict[str, str] | None,
    aliases: dict[str, str] | None,
) -> str | None:
    """Return the source a call reads (``os.getenv``, ``open`` for reading, ...)."""
    name = resolve_call_name_typed(node, type_map, aliases)
    if name is None or name not in _ALL_SOURCES:
        return None
    if name == "open" and _is_open_for_write(node):
        return None
    return name


def _credential_subscript(node: ast.AST, aliases: dict[str, str]) -> str | None:
    """``os.environ["K"]`` (also with ``os`` aliased) as a whole value is a source."""
    if not isinstance(node, ast.Subscript):
        return None
    base = resolve_dotted_name(node.value)
    if base is not None:
        base = apply_import_aliases(base, aliases)
    if base and base in _CREDENTIAL_SOURCES:
        return base
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


# ── Lexical scopes and direct calls ─────────────────────────────────────
#
# Taint is keyed by lexical scope rather than by bare name, so a tainted local
# in one function cannot taint an unrelated same-named variable or parameter
# in another. Keys are small integers; a key is built from a scope id plus a
# reference to the identifier string already in the tree, never from a
# qualified name, so memory does not grow with identifier length.
#
# Interprocedural flow is deliberately narrow. Only calls whose callee is
# known from syntax alone, in O(1), bind arguments and read return values:
#
# * ``f(...)`` where ``f`` resolves lexically to functions (or a lambda
#   assigned to the name) defined in this file; every definition bound to the
#   name is one callee, so redefinitions cost nothing per call;
# * ``self.m(...)`` / ``cls.m(...)`` in a method, ``Class.m(...)``,
#   ``super().m(...)`` and ``Class().m(...)`` / ``cls().m(...)``, through the
#   class's own methods or those of its nearest in-file base (a bounded,
#   precomputed lookup in C3 order);
# * ``Class(...)`` / ``cls(...)``, which bind to ``__init__`` and ``__new__``.
#
# Receivers of any other shape (``obj.m(...)``) are not resolved and bind
# nothing, exactly as before. ``self.x`` / ``cls.x`` attributes are per class:
# a read sees stores in the class and its in-file bases, never in siblings.

_MODULE = 0
_FUNCTION = 1
_LAMBDA = 2
_COMPREHENSION = 3
_CLASS = 4

_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)

# Call-site positional arguments at or beyond this parameter index share one
# wildcard slot, and an ``*args`` unpacking fills a wildcard slot keyed by the
# parameter index it starts at (at most this one), so binding work per
# definition is bounded by a constant.
_MAX_POSITIONAL_SLOTS = 16
# The in-file base lookup visits at most this many classes (the class itself
# included), merging at most this many bases of each, so it is a constant per
# class.
_MAX_ANCESTORS = 8
_MAX_BASES = 8

# How a flow changes the fact it carries (see ``_mark_targets``).
_ASSIGN = 0  # cites the flow's own line
_PASS = 1  # keeps the line of the fact it forwards
_RETURN = 2  # into a function's return summary: never carries bound facts
_BIND = 3  # from a call-site argument into a callee's parameter slot

# A key keeps one fact per lane; when the lanes agree (almost always) they
# share one fact, which propagates once:
#
# * lane 0, the most severe source for network output: credential (TT3) >
#   file read (TT4) > external input (TT2), so a helper's network or file
#   source never hides a credential;
# * lane 1, the most severe source for code execution and deserialization:
#   external input (TT5/TT6) > file read (TT6) > credential (TT2), so a
#   helper's credential never hides user or network input reaching ``exec``
#   or ``pickle.loads``;
# * lane 2, the first source to arrive, unranked, in the order earlier
#   releases propagated (the source a ``main`` message cites).
#
# Lanes 0 and 1 decide the rule at a sink (file writes, TT2 for every
# source, read lane 0). The message cites the lane 2 fact whenever it gives
# that same rule, so a more severe source replaces the cited one only when it
# changes the rule (see ``_TaintGraph.cited``).
_SOURCE_NAMES: tuple[str, ...] = tuple(sorted(_ALL_SOURCES))
_SOURCE_INDEX: dict[str, int] = {name: index for index, name in enumerate(_SOURCE_NAMES)}
_LANE_RANKS: tuple[tuple[int, int, int], ...] = tuple(
    (2, 0, 0)
    if name in _CREDENTIAL_SOURCES
    else (0, 2, 0)
    if name in _EXTERNAL_INPUT_SOURCES
    else (1, 1, 0)
    for name in _SOURCE_NAMES
)
_LANES = (0, 1, 2)
_FIRST_LANE = 2
_ALL_LANES = 7

# A fact is (source index, line).
_Fact = tuple[int, int]


def _noop() -> None:
    return None


class _Scope:
    """One Python namespace: module, function, lambda, comprehension or class body."""

    __slots__ = (
        "kind",
        "parent",
        "id",
        "name",
        "in_function",
        "bound",
        "declared_global",
        "declared_nonlocal",
        "children",
        "names",
        "definitions",
        "function",
        "class_def",
    )

    def __init__(self, kind: int, parent: _Scope | None, scope_id: int, name: str) -> None:
        self.kind = kind
        self.parent = parent
        self.id = scope_id
        self.name = name
        # Whether facts bound from a call site stay bound in this scope's keys.
        self.in_function = kind in (_FUNCTION, _LAMBDA) or (
            kind == _COMPREHENSION and parent is not None and parent.in_function
        )
        self.bound: set[str] | None = None
        self.declared_global: set[str] | None = None
        self.declared_nonlocal: set[str] | None = None
        self.children: list[_Scope] | None = None
        # Name nodes, and def/class statements, whose names resolve in this scope.
        self.names: list[ast.Name] | None = None
        self.definitions: list[tuple[str, _Function | _ClassDef]] | None = None
        self.function: _Function | None = None
        self.class_def: _ClassDef | None = None

    def bind(self, name: str) -> None:
        if self.bound is None:
            self.bound = set()
        self.bound.add(name)

    def request(self, node: ast.Name) -> None:
        if self.names is None:
            self.names = []
        self.names.append(node)

    def define(self, name: str, definition: _Function | _ClassDef) -> None:
        self.bind(name)
        if self.definitions is None:
            self.definitions = []
        self.definitions.append((name, definition))


class _Function:
    """A ``def``/``async def`` or lambda, with the callee group it belongs to."""

    __slots__ = ("node", "scope", "kind", "key", "group")

    def __init__(self, node: _FunctionNode | ast.Lambda, scope: _Scope) -> None:
        self.node = node
        self.scope = scope
        self.kind = "function"  # function | instance | class | static
        self.key = -1  # key of the name the definition binds
        self.group: _Group | None = None


class _ClassDef:
    __slots__ = ("node", "scope", "key", "cls")

    def __init__(self, node: ast.ClassDef, scope: _Scope) -> None:
        self.node = node
        self.scope = scope
        self.key = -1
        self.cls: _Class | None = None


class _Class:
    """Every ``class`` statement bound to one name (redefinitions merge)."""

    __slots__ = ("id", "defs", "bases", "methods", "ancestors")

    def __init__(self, class_id: int) -> None:
        self.id = class_id
        self.defs: list[_ClassDef] = []
        self.bases: list[_Class] = []
        self.methods: dict[str, dict[str, _Group]] = {}
        self.ancestors: list[_Class] | None = None


class _Group:
    """One callee: every function bound to a name, or every same-kind method
    of a class with one name. Call sites bind into its argument slots, which
    flow into each definition's parameters once.

    Each function belongs to exactly one group. A lambda bound to several
    names (``a = b = lambda ...``) is one group that every such name shares,
    so its parameters are linked once however many names it has.
    """

    __slots__ = (
        "id",
        "kind",
        "functions",
        "merged",
        "slots",
        "named",
        "kwarg_named",
        "kw_names",
        "kw_open",
    )

    def __init__(self, group_id: int, kind: str) -> None:
        self.id = group_id
        self.kind = kind
        self.functions: list[_Function] = []
        # The group this one was merged into, if any (union-find parent).
        self.merged: _Group | None = None
        self.slots: dict[object, int] = {}
        # Computed once, on the first call bound into the group (see
        # ``signature``): names a keyword argument can bind to, and the names
        # every ``**kwargs`` definition of the group binds by name instead.
        self.named: set[str] | None = None
        self.kwarg_named: set[str] | None = None
        # Keyword names call sites pass into the group's ``**kwargs``, so a
        # forwarded ``**kwargs`` binds only those (see ``forwarded_names``);
        # ``kw_open`` once a call passes ``**mapping`` or too many names.
        self.kw_names: set[str] = set()
        self.kw_open = False

    def add(self, function: _Function) -> None:
        self.functions.append(function)
        function.group = self

    def root(self) -> _Group:
        group = self
        while group.merged is not None:
            group = group.merged
        node = self
        while node.merged is not None and node.merged is not group:
            following = node.merged
            node.merged = group  # path compression
            node = following
        return group

    def signature(self) -> tuple[set[str], set[str] | None]:
        """(names a keyword can bind, names every ``**kwargs`` definition takes
        by name or ``None`` when no definition has ``**kwargs``).

        A keyword naming a parameter binds only that parameter; it reaches a
        ``**kwargs`` only when some definition with ``**kwargs`` lacks it
        (positional-only names do reach ``**kwargs``).
        """
        if self.named is None:
            named: set[str] = set()
            kwarg_named: set[str] | None = None
            for function in self.functions:
                args = function.node.args
                names = {arg.arg for arg in args.args}
                names.update(arg.arg for arg in args.kwonlyargs)
                named |= names
                if args.kwarg is not None:
                    kwarg_named = names if kwarg_named is None else kwarg_named & names
            self.named = named
            self.kwarg_named = kwarg_named
        return self.named, self.kwarg_named


class _CallTargets(NamedTuple):
    # (callee, receiver offset) pairs the call binds its arguments into.
    members: tuple[tuple[_Group, int], ...]
    # Callees whose return summary the call's value reads.
    results: tuple[_Group, ...]


_NO_TARGETS = _CallTargets((), ())


def _parameters(args: ast.arguments) -> list[ast.arg]:
    """Every parameter a function binds, in signature order."""
    params = [*args.posonlyargs, *args.args]
    if args.vararg is not None:
        params.append(args.vararg)
    params.extend(args.kwonlyargs)
    if args.kwarg is not None:
        params.append(args.kwarg)
    return params


def _decorator_name(decorator: ast.expr) -> str | None:
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    return None


def _method_kind(node: _FunctionNode) -> str:
    names = {_decorator_name(decorator) for decorator in node.decorator_list}
    if "staticmethod" in names or node.name == "__new__":
        return "static"
    if "classmethod" in names or node.name in ("__init_subclass__", "__class_getitem__"):
        return "class"
    return "instance"


def _receiver_offset(kind: str, class_object: bool) -> int:
    """Leading parameters a call binds implicitly (``self`` / ``cls``).

    ``self.m(x)`` binds ``x`` to an instance method's second parameter;
    ``Class.m(obj, x)`` and ``cls.m(obj, x)`` pass ``obj`` explicitly.
    """
    if kind == "class":
        return 1
    if kind == "instance":
        return 0 if class_object else 1
    return 0


def _c3(cls: _Class, bases: list[_Class]) -> list[_Class]:
    """C3 linearization of *cls* from its bases' linearizations, cut at ``_MAX_ANCESTORS``.

    Bases are already linearized (and cut). A base still being linearized
    (a cycle) stands for itself. An inconsistent hierarchy, which Python
    rejects, continues with the leftmost remaining head.
    """
    result = [cls]
    if not bases:
        return result
    sequences = [
        [c for c in (base.ancestors if base.ancestors is not None else [base]) if c is not cls]
        for base in bases
    ]
    if len(sequences) == 1:  # single inheritance: the base's order
        result.extend(sequences[0][: _MAX_ANCESTORS - 1])
        return result
    sequences.append([base for base in bases if base is not cls])
    # How many sequences hold each class after their head.
    tails: dict[int, int] = {}
    for sequence in sequences:
        for c in sequence[1:]:
            tails[c.id] = tails.get(c.id, 0) + 1
    positions = [0] * len(sequences)
    placed = {cls.id}
    while len(result) < _MAX_ANCESTORS:
        candidate: _Class | None = None
        leftmost: _Class | None = None
        for sequence, position in zip(sequences, positions, strict=True):
            if position < len(sequence):
                head = sequence[position]
                if leftmost is None:
                    leftmost = head
                if not tails.get(head.id):
                    candidate = head
                    break
        candidate = candidate or leftmost
        if candidate is None:
            break
        if candidate.id not in placed:
            placed.add(candidate.id)
            result.append(candidate)
        for i, sequence in enumerate(sequences):
            position = positions[i]
            if position < len(sequence) and sequence[position] is candidate:
                position += 1
                positions[i] = position
                if position < len(sequence):
                    tails[sequence[position].id] -= 1
    return result


class _ScopeIndex:
    """Lexical scopes of one module plus the in-file callees calls can bind to."""

    def __init__(self) -> None:
        self.scopes: list[_Scope] = []
        # Statements and expressions that create flows, with their scope, in
        # ``ast.walk`` (breadth-first) order: Assign, Return and Call.
        self.statements: list[tuple[ast.AST, _Scope]] = []
        self.functions: list[_Function] = []
        self.lambdas: dict[ast.Lambda, _Function] = {}
        self.lambda_bindings: list[tuple[ast.Name, ast.Lambda]] = []
        self.class_defs: list[_ClassDef] = []
        self.classes: list[_Class] = []
        self.attributes: list[ast.Attribute] = []
        self.super_scopes: dict[ast.Call, _Scope] = {}
        # Keys: an id per (scope id, name) or per synthetic descriptor.
        self._ids: dict[tuple[object, ...], int] = {}
        self.key_scope: list[_Scope | None] = []
        self.key_name: list[str] = []
        # 1 when a fact bound from a call site stays bound in the key (function
        # locals); 0 for module, class and attribute state, where it becomes
        # an ordinary fact.
        self.local = bytearray()
        self.name_key: dict[ast.Name, int] = {}
        # (class key, module key) for each name a class body reads and also
        # binds: such a read sees the class's value or, before the class
        # binds it, the module global (Python's LOAD_NAME).
        self.class_reads: dict[tuple[int, int], None] = {}
        # Keys some statement assigns, and each ``**kwargs`` parameter's
        # key -> its function.
        self.stored: set[int] = set()
        self.kwarg_params: dict[int, _Function] = {}
        self.class_of_key: dict[int, _Class] = {}
        self.function_groups: dict[int, _Group] = {}
        self.groups: list[_Group] = []
        # Key of each method's ``self``/``cls`` parameter -> (class, is ``cls``).
        self.receivers: dict[int, tuple[_Class, bool]] = {}
        # Attribute read keys (``self.x`` / ``cls.x``) by node and by (class, attr).
        self.views: dict[ast.Attribute, int] = {}
        self.view_list: list[tuple[_Class, str, int]] = []
        self._calls: dict[ast.Call, _CallTargets] = {}
        self._lookups: dict[tuple[int, str, int], tuple[_Group, ...]] = {}
        self._lambda_groups: dict[ast.Lambda, _Group] = {}

    @property
    def module(self) -> _Scope:
        return self.scopes[0]

    def new_scope(self, kind: int, parent: _Scope | None, name: str) -> _Scope:
        scope = _Scope(kind, parent, len(self.scopes), name)
        self.scopes.append(scope)
        if parent is not None:
            if parent.children is None:
                parent.children = []
            parent.children.append(scope)
        return scope

    # ── keys ──

    def _new_key(self, scope: _Scope | None, name: str, local: bool) -> int:
        key = len(self.key_scope)
        self.key_scope.append(scope)
        self.key_name.append(name)
        self.local.append(1 if local else 0)
        return key

    def key(self, scope: _Scope, name: str) -> int:
        """Key of *name* bound in *scope* itself."""
        ident = (scope.id, name)
        key = self._ids.get(ident)
        if key is None:
            key = self._new_key(scope, name, scope.in_function)
            self._ids[ident] = key
        return key

    def synthetic(self, ident: tuple[object, ...], local: bool, name: str = "") -> int:
        key = self._ids.get(ident)
        if key is None:
            key = self._new_key(None, name, local)
            self._ids[ident] = key
        return key

    def visible(self, key: int, root: _Scope) -> bool:
        """Whether a read evaluated from *root* carries *key*'s data.

        A lambda's parameters and a comprehension's targets live only inside
        it: an expression that merely contains the lambda or comprehension
        does not hold their values (a function object does not carry its
        arguments), so they are skipped unless the walk starts in that scope.
        """
        owner = self.key_scope[key]
        return owner is None or owner is root or owner.kind not in (_LAMBDA, _COMPREHENSION)

    # ── callees ──

    def function_group(self, key: int) -> _Group:
        group = self.function_groups.get(key)
        if group is None:
            group = self._new_group("function")
            self.function_groups[key] = group
        return group

    def _new_group(self, kind: str) -> _Group:
        group = _Group(len(self.groups), kind)
        self.groups.append(group)
        return group

    def method_group(self, cls: _Class, name: str, kind: str) -> _Group:
        kinds = cls.methods.setdefault(name, {})
        group = kinds.get(kind)
        if group is None:
            group = self._new_group(kind)
            kinds[kind] = group
        return group

    def lambda_group(self, node: ast.Lambda, function: _Function) -> _Group:
        """The one callee of a lambda, shared by every name it is bound to."""
        group = self._lambda_groups.get(node)
        if group is None:
            group = self._new_group("function")
            group.add(function)
            self._lambda_groups[node] = group
        return group.root()

    def bind_name(self, key: int, group: _Group) -> None:
        """Make *group* (a lambda's) a callee of the name *key*.

        The first binding shares the group; a name with another definition
        merges the two groups (redefinitions are one callee), so a lambda
        bound to many names is never copied per name.
        """
        current = self.function_groups.get(key)
        if current is None:
            self.function_groups[key] = group
            return
        current, group = current.root(), group.root()
        if current is group:
            return
        if len(current.functions) < len(group.functions):
            current, group = group, current
        group.merged = current
        for function in group.functions:
            function.group = current
        current.functions.extend(group.functions)
        group.functions = []

    def finish_groups(self) -> None:
        for key, group in self.function_groups.items():
            if group.merged is not None:
                self.function_groups[key] = group.root()

    def ancestors(self, cls: _Class, tick: Callable[[], None]) -> list[_Class]:
        """*cls* then its in-file bases in C3 (method resolution) order, bounded.

        Each class merges the (bounded) linearizations of at most
        ``_MAX_BASES`` bases and keeps the first ``_MAX_ANCESTORS`` classes,
        so the work per class is a constant. Classes are linearized bases
        first, iteratively, so a long inheritance chain cannot exhaust the
        stack; a cycle (possible when redefinitions merge) is cut.
        """
        if cls.ancestors is not None:
            return cls.ancestors
        visiting: set[int] = set()
        stack = [cls]
        while stack:
            tick()
            current = stack[-1]
            if current.ancestors is not None:
                stack.pop()
                continue
            bases = current.bases[:_MAX_BASES]
            if current.id not in visiting:
                visiting.add(current.id)
                pending = [
                    base for base in bases if base.ancestors is None and base.id not in visiting
                ]
                if pending:
                    stack.extend(reversed(pending))
                    continue
            stack.pop()
            current.ancestors = _c3(current, bases)
        return cls.ancestors

    def lookup(
        self, cls: _Class, name: str, skip: int, tick: Callable[[], None]
    ) -> tuple[_Group, ...]:
        """Method groups named *name* on the nearest class in *cls*'s lookup chain.

        *skip* is 1 for ``super()``, which starts after the class itself.
        """
        memo = (cls.id, name, skip)
        found = self._lookups.get(memo)
        if found is None:
            found = ()
            for candidate in self.ancestors(cls, tick)[skip:]:
                kinds = candidate.methods.get(name)
                if kinds:
                    found = tuple(kinds.values())
                    break
            self._lookups[memo] = found
        return found

    def super_class(self, call: ast.Call) -> _Class | None:
        """The class whose bases ``super()`` / ``super(C, obj)`` starts from."""
        func = call.func
        if not isinstance(func, ast.Name) or func.id != "super":
            return None
        key = self.name_key.get(func)
        if key is None or self.key_scope[key] is not self.module:
            return None
        if self.module.bound is not None and "super" in self.module.bound:
            return None  # shadowed builtin
        if call.args:
            first = call.args[0]
            if isinstance(first, ast.Name):
                return self.class_of_key.get(self.name_key.get(first, -1))
            return None
        scope = self.super_scopes.get(call)
        while scope is not None and scope.kind in (_LAMBDA, _COMPREHENSION):
            scope = scope.parent
        if scope is None or scope.kind != _FUNCTION:
            return None
        parent = scope.parent
        if parent is None or parent.class_def is None:
            return None
        return parent.class_def.cls

    def constructed_class(self, key: int) -> _Class | None:
        """The class ``Name(...)`` constructs: an in-file class name, or ``cls``."""
        receiver = self.receivers.get(key)
        if receiver is not None:
            return receiver[0] if receiver[1] else None  # cls(...) in a classmethod
        return self.class_of_key.get(key)

    def call_targets(self, call: ast.Call, tick: Callable[[], None]) -> _CallTargets:
        """The in-file callees a call reaches, resolved from syntax alone."""
        cached = self._calls.get(call)
        if cached is not None:
            return cached
        members: list[tuple[_Group, int]] = []
        results: list[_Group] = []
        func = call.func
        if isinstance(func, ast.Name):
            key = self.name_key.get(func, -1)
            group = self.function_groups.get(key)
            if group is not None:
                members.append((group, 0))
                results.append(group)
            constructed = self.constructed_class(key)
            if constructed is not None:
                for group in self.lookup(constructed, "__init__", 0, tick):
                    members.append((group, 0 if group.kind == "static" else 1))
                # ``C(x)`` also calls ``C.__new__(C, x)``: ``cls`` is implicit.
                for group in self.lookup(constructed, "__new__", 0, tick):
                    members.append((group, 1))
        elif isinstance(func, ast.Lambda):
            function = self.lambdas.get(func)
            if function is not None:
                members.append((self.lambda_group(func, function), 0))
        elif isinstance(func, ast.Attribute):
            value = func.value
            cls: _Class | None = None
            class_object = False
            skip = 0
            if isinstance(value, ast.Name):
                key = self.name_key.get(value, -1)
                receiver = self.receivers.get(key)
                if receiver is not None:
                    cls, class_object = receiver
                else:
                    cls = self.class_of_key.get(key)
                    class_object = True
            elif isinstance(value, ast.Call):
                cls = self.super_class(value)
                if cls is not None:
                    skip = 1
                elif isinstance(value.func, ast.Name):
                    # ``C().m(...)`` / ``cls().m(...)``: an instance of an in-file class.
                    cls = self.constructed_class(self.name_key.get(value.func, -1))
            if cls is not None:
                for group in self.lookup(cls, func.attr, skip, tick):
                    members.append((group, _receiver_offset(group.kind, class_object)))
                    results.append(group)
        found = _CallTargets(tuple(members), tuple(results)) if members else _NO_TARGETS
        self._calls[call] = found
        return found

    # ── attributes ──

    def receiver_class(self, node: ast.Attribute) -> _Class | None:
        """The class of ``self`` / ``cls`` in ``self.x`` / ``cls.x``, if it is one."""
        if not isinstance(node.value, ast.Name):
            return None
        receiver = self.receivers.get(self.name_key.get(node.value, -1))
        return receiver[0] if receiver is not None else None

    def attribute_store(self, node: ast.Attribute) -> int | None:
        cls = self.receiver_class(node)
        if cls is None:
            return None
        return self.synthetic(("store", cls.id, node.attr), False)

    def class_body_store(self, cls: _Class, name: str) -> int:
        """Key of a name a class body assigns, read as ``self.name`` / ``cls.name``.

        Kept apart from ``self.x`` stores: a subclass that rebinds the name in
        its own body shadows a base's class-body value.
        """
        return self.synthetic(("class", cls.id, name), False)

    def attribute_view(self, node: ast.Attribute) -> int | None:
        """Key read by ``self.x`` / ``cls.x``: stores in the class and its bases."""
        key = self.views.get(node)
        if key is not None:
            return key
        if not isinstance(node.ctx, ast.Load):
            return None
        cls = self.receiver_class(node)
        if cls is None:
            return None
        ident = ("view", cls.id, node.attr)
        key = self._ids.get(ident)
        if key is None:
            key = self.synthetic(ident, False)
            self.view_list.append((cls, node.attr, key))
        self.views[node] = key
        return key

    # ── debugging / tests ──

    def qualname(self, scope: _Scope) -> str:
        parts: list[str] = []
        current: _Scope | None = scope
        while current is not None and current.kind != _MODULE:
            parent = current.parent
            if parent is not None and parent.kind in (_FUNCTION, _LAMBDA, _COMPREHENSION):
                parts.append(f"<locals>.{current.name}")
            else:
                parts.append(current.name)
            current = parent
        return ".".join(reversed(parts))


def _build_scope_index(
    tree: ast.AST,
    check_runtime: Callable[[], None] | None = None,
) -> _ScopeIndex:
    """Assign every name to its lexical scope and index in-file callees.

    Three linear passes: a breadth-first visit (the same order as
    ``ast.walk``) that records scopes, bindings and flow statements; a
    depth-first pass over the scope tree that resolves every name with one
    stack per identifier, O(1) per name however deep the nesting; and an
    indexing pass over class and function definitions.
    """
    tick = check_runtime or _noop
    index = _ScopeIndex()
    module = index.new_scope(_MODULE, None, "")
    # Nodes evaluated in a different scope than their parent (bodies of
    # functions, classes and lambdas; comprehension elements).
    switch: dict[ast.AST, _Scope] = {}
    # Walrus targets bind in the nearest enclosing non-comprehension scope.
    walrus: dict[ast.Name, _Scope] = {}

    queue: deque[tuple[ast.AST, _Scope]] = deque([(tree, module)])
    while queue:
        tick()
        node, scope = queue.popleft()
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                scope.request(node)
            else:
                binding = walrus.pop(node, scope)
                binding.bind(node.id)
                binding.request(node)
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inner = index.new_scope(_FUNCTION, scope, node.name)
            function = _Function(node, inner)
            inner.function = function
            index.functions.append(function)
            scope.define(node.name, function)
            for arg in _parameters(node.args):
                inner.bind(arg.arg)
            # Only the body runs in the new scope; decorators, defaults and
            # annotations are evaluated where the def stands.
            for statement in node.body:
                switch[statement] = inner
        elif isinstance(node, ast.ClassDef):
            inner = index.new_scope(_CLASS, scope, node.name)
            class_def = _ClassDef(node, inner)
            inner.class_def = class_def
            index.class_defs.append(class_def)
            scope.define(node.name, class_def)
            for statement in node.body:
                switch[statement] = inner
        elif isinstance(node, ast.Lambda):
            inner = index.new_scope(_LAMBDA, scope, "<lambda>")
            function = _Function(node, inner)
            inner.function = function
            index.functions.append(function)
            index.lambdas[node] = function
            for arg in _parameters(node.args):
                inner.bind(arg.arg)
            switch[node.body] = inner
        elif isinstance(node, _COMPREHENSIONS):
            # Python 3 comprehensions have their own scope; only the first
            # iterable is evaluated in the enclosing one.
            inner = index.new_scope(_COMPREHENSION, scope, f"<{type(node).__name__.lower()}>")
            if isinstance(node, ast.DictComp):
                switch[node.key] = inner
                switch[node.value] = inner
            else:
                switch[node.elt] = inner
            for position, generator in enumerate(node.generators):
                switch[generator.target] = inner
                for condition in generator.ifs:
                    switch[condition] = inner
                if position:
                    switch[generator.iter] = inner
        elif isinstance(node, ast.NamedExpr):
            binding = scope
            while binding.kind == _COMPREHENSION and binding.parent is not None:
                binding = binding.parent
            if isinstance(node.target, ast.Name):
                walrus[node.target] = binding
        elif isinstance(node, ast.Global):
            if scope.declared_global is None:
                scope.declared_global = set()
            scope.declared_global.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            if scope.declared_nonlocal is None:
                scope.declared_nonlocal = set()
            scope.declared_nonlocal.update(node.names)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name != "*":
                    scope.bind(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                scope.bind(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)):
            if node.name:
                scope.bind(node.name)
        elif isinstance(node, ast.MatchMapping):
            if node.rest:
                scope.bind(node.rest)
        elif isinstance(node, ast.Assign):
            index.statements.append((node, scope))
            if isinstance(node.value, ast.Lambda):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        index.lambda_bindings.append((target, node.value))
        elif isinstance(node, ast.Return):
            index.statements.append((node, scope))
        elif isinstance(node, ast.Call):
            index.statements.append((node, scope))
            if isinstance(node.func, ast.Name) and node.func.id == "super":
                index.super_scopes[node] = scope
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                index.attributes.append(node)
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.expr_context):
                queue.append((child, switch.pop(child, scope)))

    _resolve_names(index, tick)
    _index_definitions(index, tick)
    return index


def _resolve_names(index: _ScopeIndex, tick: Callable[[], None]) -> None:
    """Resolve every recorded name by Python's rules, O(1) per name.

    A depth-first walk of the scope tree keeps, per identifier, a stack of
    the function-like scopes that bind it (``global`` pushes the module).
    A name read in a function, lambda or comprehension is local if bound
    there unless declared ``global``/``nonlocal``, and otherwise resolves to
    the nearest enclosing binder, then the module. Class bodies see their own
    names but are invisible to the functions and comprehensions inside them.
    A class-body read of a name the class binds sees the class's value or,
    before the class binds it, the module global (CPython's LOAD_NAME; an
    enclosing function's local is not consulted), so it reads both.
    """
    module = index.module
    stacks: dict[str, list[_Scope]] = {}

    def binding_scope(scope: _Scope, name: str) -> _Scope:
        if scope.declared_global is not None and name in scope.declared_global:
            return module
        if scope.kind == _MODULE:
            return module
        if scope.kind == _CLASS:
            nonlocal_names = scope.declared_nonlocal
            if (
                scope.bound is not None
                and name in scope.bound
                and (nonlocal_names is None or name not in nonlocal_names)
            ):
                return scope
        stack = stacks.get(name)
        return stack[-1] if stack else module

    work: list[tuple[_Scope, list[str] | None]] = [(module, None)]
    while work:
        scope, pushed = work.pop()
        if pushed is not None:
            for name in pushed:
                stacks[name].pop()
            continue
        tick()
        names: list[str] = []
        if scope.kind not in (_MODULE, _CLASS):
            declared_global = scope.declared_global or set()
            declared_nonlocal = scope.declared_nonlocal or set()
            for name in declared_global:
                stacks.setdefault(name, []).append(module)
                names.append(name)
            for name in scope.bound or ():
                if name not in declared_global and name not in declared_nonlocal:
                    stacks.setdefault(name, []).append(scope)
                    names.append(name)
        for node in scope.names or ():
            tick()
            binder = binding_scope(scope, node.id)
            key = index.key(binder, node.id)
            index.name_key[node] = key
            if not isinstance(node.ctx, ast.Load):
                index.stored.add(key)
            elif binder is scope and scope.kind == _CLASS:
                index.class_reads[(key, index.key(module, node.id))] = None
        for name, definition in scope.definitions or ():
            tick()
            definition.key = index.key(binding_scope(scope, name), name)
        work.append((scope, names))
        if scope.children:
            work.extend((child, None) for child in reversed(scope.children))


def _index_definitions(index: _ScopeIndex, tick: Callable[[], None]) -> None:
    """Merge class redefinitions, link in-file bases, and group callees."""
    for class_def in index.class_defs:
        tick()
        cls = index.class_of_key.get(class_def.key)
        if cls is None:
            cls = _Class(len(index.classes))
            index.classes.append(cls)
            index.class_of_key[class_def.key] = cls
        cls.defs.append(class_def)
        class_def.cls = cls
    for cls in index.classes:
        linked = {cls.id}
        for class_def in cls.defs:
            for base in class_def.node.bases:
                tick()
                if isinstance(base, ast.Subscript):  # Base[T]
                    base = base.value
                if not isinstance(base, ast.Name):
                    continue
                parent = index.class_of_key.get(index.name_key.get(base, -1))
                if parent is not None and parent.id not in linked:
                    linked.add(parent.id)
                    cls.bases.append(parent)
    for function in index.functions:
        tick()
        node = function.node
        if node.args.kwarg is not None:
            index.kwarg_params[index.key(function.scope, node.args.kwarg.arg)] = function
        if isinstance(node, ast.Lambda):
            continue
        parent = function.scope.parent
        if parent is None or parent.class_def is None or parent.class_def.cls is None:
            index.function_group(function.key).add(function)
            continue
        # A method is reached through its class, never by its bare name.
        cls = parent.class_def.cls
        kind = _method_kind(node)
        function.kind = kind
        index.method_group(cls, node.name, kind).add(function)
        positional = [*node.args.posonlyargs, *node.args.args]
        if kind != "static" and positional:
            receiver_key = index.key(function.scope, positional[0].arg)
            index.receivers[receiver_key] = (cls, kind == "class")
    for target, lambda_node in index.lambda_bindings:
        tick()
        function = index.lambdas.get(lambda_node)
        key = index.name_key.get(target)
        if function is not None and key is not None:
            index.bind_name(key, index.lambda_group(lambda_node, function))
    index.finish_groups()


def _mark_targets(
    graph: _TaintGraph,
    targets: Sequence[int],
    kind: int,
    line: int,
    fact: _Fact,
    bound: int,
    lanes: int = _ALL_LANES,
) -> list[tuple[int, int, int]]:
    """Give each target *fact*, in each of *lanes*, where it outranks what it holds.

    Monotone: a target's fact in a lane is only ever replaced by a strictly
    more severe source for that lane (never in lane 2, where every source
    ranks the same and the first to arrive stays), so each key changes at
    most three times per lane and slot, and the worklist terminates. A key
    has two slots: ordinary facts, and facts bound from a call site into a
    function's parameters (``bound``). A bound fact stays bound in function
    locals; module, class and attribute state turns it into an ordinary
    fact. Returns the (key, slot, changed lanes) triples to propagate from.
    """
    source, fact_line = fact
    if kind != _PASS:
        fact_line = line
    if kind == _BIND:
        bound = 1
    new_fact = (source, fact_line)
    ranks = _LANE_RANKS[source]
    local = graph.index.local
    facts = graph.facts
    ordinary = facts[0]
    tick = graph.tick
    changed: list[tuple[int, int, int]] = []
    for position, target in enumerate(targets):
        if position and not position & 1023:
            tick()
        slot = 1 if bound and local[target] else 0
        held = facts[slot]
        mask = 0
        for lane in _LANES:
            if not lanes >> lane & 1:
                continue
            rank = ranks[lane]
            current = ordinary[lane].get(target)
            if current is not None and _LANE_RANKS[current[0]][lane] >= rank:
                continue
            if slot:
                current = held[lane].get(target)
                if current is not None and _LANE_RANKS[current[0]][lane] >= rank:
                    continue
            held[lane][target] = new_fact
            mask |= 1 << lane
        if mask:
            changed.append((target, slot, mask))
    return changed


class _TaintGraph:
    """Scope-aware taint, independent of AST visit order.

    Taint is reachability in a flow graph over keys, computed with a monotone
    worklist that fires each flow at most once per (lane, source rank, slot):
    at most 14 times whatever the number of sources (three ranks in lanes 0
    and 1, one in lane 2, two slots each).

    * Phase A replays exactly the assignment flows earlier releases used
      (``x = <expr>`` reads every name in ``<expr>``), in the same order, plus
      a class body's read of the module global under a name the class also
      binds (earlier releases had one name for both). In lane 2, which keeps
      the first source to arrive, a key that earlier releases tainted the
      same way holds the source and line they cite. Their fact differs only
      where it came from a same-named variable in another scope, a name clash
      this scoping removes.
    * Phase B adds what scoping makes precise: arguments of direct in-file
      calls bind into parameters (as bound facts), parameter defaults,
      return summaries read by callers, and ``self.x`` / ``cls.x`` attributes.
      Phase B never replaces a fact by one of equal rank, so it only fills
      keys phase A left clean or upgrades lane 0 or 1 to a more severe
      source; lane 2 is never replaced.

    At a sink, the sink's ranked lane decides the rule, and the message cites
    the lane 2 fact whenever that gives the same rule (``cited``). A more
    severe source therefore changes a message only where it changes the rule.

    A callee's return summary carries only facts that do not depend on its
    parameters: bound facts never enter it. What a call returns from its
    own arguments is read at that call site, from those arguments (every
    call's value includes its arguments, as before), so one caller's secret
    cannot taint another caller's result, also through closures and lambdas.
    """

    def __init__(
        self,
        index: _ScopeIndex,
        type_map: dict[str, str],
        aliases: dict[str, str],
        check_runtime: Callable[[], None] | None = None,
    ) -> None:
        self.index = index
        self.type_map = type_map
        self.aliases = aliases
        self.tick = check_runtime or _noop
        # facts[slot][lane]: key -> fact; slot 1 holds facts bound from a call site.
        self.facts: tuple[tuple[dict[int, _Fact], ...], ...] = (
            ({}, {}, {}),
            ({}, {}, {}),
        )
        # Each flow's (targets, kind, line), keyed by its index (a stable id).
        # ``prop_a`` / ``prop_b`` map a key read by a phase A / B flow to its ids.
        self.flows: list[tuple[tuple[int, ...], int, int]] = []
        self.prop_a: dict[int, list[int]] = {}
        self.prop_b: dict[int, list[int]] = {}
        self.worklist: deque[tuple[int, int, int]] = deque()
        self.seeds: list[tuple[tuple[int, ...], int, int, _Fact, int]] = []
        # Per flow, one bit per (lane, source rank, slot) it has already propagated.
        self.fired: list[int] = []
        self.returns: dict[int, int] = {}  # group id -> return summary key
        self._values: dict[ast.Call, int] = {}
        self._class_links: set[tuple[int, str]] = set()

    # ── building ──

    def add_flow(
        self, reads: Iterable[int], targets: Sequence[int], kind: int, line: int, phase_a: bool
    ) -> None:
        if not targets:
            return
        keys = list(dict.fromkeys(reads))
        if not keys:
            return
        flow_id = len(self.flows)
        self.flows.append((tuple(targets), kind, line))
        table = self.prop_a if phase_a else self.prop_b
        for key in keys:
            table.setdefault(key, []).append(flow_id)

    def seed(
        self, targets: Sequence[int], kind: int, line: int, sources: list[tuple[str, int]]
    ) -> None:
        """Queue phase B sources for *targets* (applied after phase A)."""
        if targets:
            for source, lanes in sources:
                fact = (_SOURCE_INDEX[source], line)
                self.seeds.append((tuple(targets), kind, line, fact, lanes))

    def ret_key(self, group: _Group) -> int:
        key = self.returns.get(group.id)
        if key is None:
            key = self.index.synthetic(("return", group.id), False)
            self.returns[group.id] = key
        return key

    def value_key(self, call: ast.Call, root: _Scope) -> int | None:
        """Key of an in-file call's value, for an enclosing call's arguments.

        Reading a nested in-file call through its own value key walks each
        argument once, however deeply such calls nest.
        """
        if self.index.call_targets(call, self.tick) is _NO_TARGETS:
            return None
        key = self._values.get(call)
        if key is None:
            key = self.index.synthetic(("value", id(call)), root.in_function)
            self._values[call] = key
        return key

    def scan(
        self, node: ast.AST, root: _Scope, nested_values: bool, opaque_lambdas: bool = False
    ) -> tuple[list[int], list[int], list[tuple[str, int]]]:
        """Walk an expression once: (names read, other keys read, sources).

        Names are the reads earlier releases used. Other keys are the return
        summaries of in-file calls and ``self.x`` attributes. With
        *nested_values*, an in-file call is read through its value key and not
        walked. With *opaque_lambdas* a lambda is not walked: a default holds
        the function object, and each lambda's own defaults are scanned once,
        for that lambda. Sources, each with the lanes it is for: in lanes 0
        and 1 the most severe source call (the first in walk order among
        equals), in lane 2 the first source call in walk order (the one
        earlier releases took); or an ``os.environ[...]`` subscript as the
        whole expression.
        """
        index = self.index
        name_key = index.name_key
        tick = self.tick
        names: dict[int, None] = {}
        others: dict[int, None] = {}
        best: list[str | None] = [None, None, None]
        best_rank = [-1, -1, -1]
        queue: deque[ast.AST] = deque([node])
        while queue:
            tick()
            child = queue.popleft()
            if isinstance(child, ast.Name):
                key = name_key.get(child)
                if key is not None and index.visible(key, root):
                    names[key] = None
                continue
            if isinstance(child, ast.Call):
                if nested_values:
                    value = self.value_key(child, root)
                    if value is not None:
                        others[value] = None
                        continue
                source = _call_source(child, self.type_map, self.aliases)
                if source is not None:
                    ranks = _LANE_RANKS[_SOURCE_INDEX[source]]
                    for lane in _LANES:
                        if ranks[lane] > best_rank[lane]:
                            best[lane], best_rank[lane] = source, ranks[lane]
                for group in index.call_targets(child, tick).results:
                    others[self.ret_key(group)] = None
            elif isinstance(child, ast.Attribute):
                view = index.attribute_view(child)
                if view is not None:
                    others[view] = None
            elif opaque_lambdas and isinstance(child, ast.Lambda):
                continue
            for grandchild in ast.iter_child_nodes(child):
                if not isinstance(grandchild, ast.expr_context):
                    queue.append(grandchild)
        if best[0] is None:
            best = [_credential_subscript(node, self.aliases)] * len(_LANES)
        lanes_of: dict[str, int] = {}
        for lane, source in enumerate(best):
            if source is not None:
                lanes_of[source] = lanes_of.get(source, 0) | 1 << lane
        return list(names), list(others), list(lanes_of.items())

    def assign(self, node: ast.Assign, scope: _Scope) -> None:
        index = self.index
        names, others, sources = self.scan(node.value, scope, False)
        # Phase A targets are the ones earlier releases tainted: names and
        # tuples of names. ``self.x`` / ``cls.x`` targets are phase B.
        plain: list[int] = []
        attributes: list[int] = []
        for target in node.targets:
            elements = target.elts if isinstance(target, ast.Tuple) else [target]
            for element in elements:
                if isinstance(element, ast.Name):
                    plain.append(index.name_key[element])
                    if scope.kind == _CLASS:
                        self.class_attribute(scope, element.id, index.name_key[element])
                elif isinstance(element, ast.Attribute):
                    store = index.attribute_store(element)
                    if store is not None:
                        attributes.append(store)
        line = node.lineno
        if sources:
            # A direct source seeds its targets in phase A, as before; a more
            # severe source among the names it reads may still upgrade them.
            for source, lanes in sources:
                fact = (_SOURCE_INDEX[source], line)
                self.worklist.extend(_mark_targets(self, plain, _ASSIGN, line, fact, 0, lanes))
            self.seed(attributes, _ASSIGN, line, sources)
            self.add_flow([*names, *others], [*plain, *attributes], _ASSIGN, line, False)
        else:
            self.add_flow(names, plain, _ASSIGN, line, True)
            self.add_flow(others, plain, _ASSIGN, line, False)
            self.add_flow([*names, *others], attributes, _ASSIGN, line, False)

    def class_attribute(self, scope: _Scope, name: str, key: int) -> None:
        """A name assigned in a class body is also the class's attribute."""
        class_def = scope.class_def
        if class_def is None or class_def.cls is None:
            return
        link = (scope.id, name)
        if link in self._class_links:
            return
        self._class_links.add(link)
        store = self.index.class_body_store(class_def.cls, name)
        self.add_flow((key,), (store,), _PASS, 0, False)

    def returned(self, value: ast.expr, scope: _Scope, line: int) -> None:
        function = scope.function
        if function is None or function.group is None:
            return
        targets = (self.ret_key(function.group),)
        names, others, sources = self.scan(value, scope, False)
        self.seed(targets, _RETURN, line, sources)
        self.add_flow([*names, *others], targets, _RETURN, line, False)

    def call(self, node: ast.Call, scope: _Scope) -> None:
        targets = self.index.call_targets(node, self.tick)
        value = self._values.get(node)
        if not targets.members and value is None:
            return
        parts = [arg.value if isinstance(arg, ast.Starred) else arg for arg in node.args]
        parts.extend(keyword.value for keyword in node.keywords)
        scans = [self.scan(part, scope, True) for part in parts]
        line = node.lineno
        if value is not None:
            reads: list[int] = []
            for names, others, sources in scans:
                reads.extend(names)
                reads.extend(others)
                self.seed((value,), _ASSIGN, line, sources)
            reads.extend(self.ret_key(group) for group in targets.results)
            self.add_flow(reads, (value,), _ASSIGN, line, False)
        if targets.members:
            self.bind_arguments(node, targets.members, scans, line)

    def bind_arguments(
        self,
        node: ast.Call,
        members: tuple[tuple[_Group, int], ...],
        scans: list[tuple[list[int], list[int], list[tuple[str, int]]]],
        line: int,
    ) -> None:
        """Bind a call's arguments into each callee's slots.

        Positional arguments bind by parameter index (the receiver *offset*
        added); from the first ``*args`` on they share a wildcard slot keyed
        by the parameter index it starts at, and arguments at index
        ``_MAX_POSITIONAL_SLOTS`` or beyond share the last one. ``**mapping``
        binds a wildcard keyed by the offset, except a forwarded ``**kwargs``
        whose keyword names are known, which binds as those keywords. So an
        implicit ``self`` / ``cls`` is never filled by an unpacking.
        """
        slots: list[list[int]] = []
        starred_at = -1
        for position, arg in enumerate(node.args):
            if starred_at < 0 and isinstance(arg, ast.Starred):
                starred_at = position
            keys: list[int] = []
            for group, offset in members:
                slot: object
                if starred_at >= 0:
                    slot = ("*", min(starred_at + offset, _MAX_POSITIONAL_SLOTS))
                elif position + offset >= _MAX_POSITIONAL_SLOTS:
                    slot = ("*", _MAX_POSITIONAL_SLOTS)
                else:
                    slot = position + offset
                keys.append(self.slot(group, slot))
            slots.append(keys)
        for keyword in node.keywords:
            names = (
                [keyword.arg] if keyword.arg is not None else self.forwarded_names(keyword.value)
            )
            keys = []
            for group, offset in members:
                if names is None:
                    keys.append(self.slot(group, ("**", offset)))
                    continue
                named, kwarg_named = group.signature()
                for name in names:
                    if name in named:
                        keys.append(self.slot(group, ("kw", name)))
                    if kwarg_named is not None and name not in kwarg_named:
                        keys.append(self.slot(group, "kw*"))
            slots.append(list(dict.fromkeys(keys)))
        for keys, (names, others, sources) in zip(slots, scans, strict=True):
            self.seed(keys, _BIND, line, sources)
            self.add_flow([*names, *others], keys, _BIND, line, False)

    def forwarded_names(self, value: ast.expr) -> list[str] | None:
        """The keywords a forwarded ``**kwargs`` can hold, or ``None`` if unknown.

        ``**kwargs`` of an in-file function holds only the keyword names its
        call sites pass into it (``collect_keywords``), unless some call site
        passes a ``**mapping`` or the parameter is reassigned.
        """
        if not isinstance(value, ast.Name):
            return None
        index = self.index
        key = index.name_key.get(value, -1)
        function = index.kwarg_params.get(key)
        if function is None or function.group is None or key in index.stored:
            return None
        group = function.group
        return None if group.kw_open else sorted(group.kw_names)

    def collect_keywords(self) -> None:
        """Record the keyword names each callee's ``**kwargs`` receives."""
        index = self.index
        for node, _scope in index.statements:
            if not isinstance(node, ast.Call) or not node.keywords:
                continue
            self.tick()
            for group, _offset in index.call_targets(node, self.tick).members:
                _named, kwarg_named = group.signature()
                if kwarg_named is None or group.kw_open:
                    continue
                for keyword in node.keywords:
                    if keyword.arg is None:
                        group.kw_open = True
                        break
                    if keyword.arg not in kwarg_named:
                        group.kw_names.add(keyword.arg)
                if len(group.kw_names) > _MAX_POSITIONAL_SLOTS:
                    group.kw_open = True

    def slot(self, group: _Group, slot: object) -> int:
        key = group.slots.get(slot)
        if key is None:
            key = self.index.synthetic(("slot", group.id, slot), True)
            group.slots[slot] = key
        return key

    def defaults(self, function: _Function) -> None:
        args = function.node.args
        defining = function.scope.parent or self.index.module
        positional = [*args.posonlyargs, *args.args]
        defaulted = positional[len(positional) - len(args.defaults) :]
        pairs: list[tuple[ast.arg, ast.expr]] = list(zip(defaulted, args.defaults, strict=True))
        for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
            if default is not None:
                pairs.append((arg, default))
        for arg, default in pairs:
            self.tick()
            target = (self.index.key(function.scope, arg.arg),)
            names, others, sources = self.scan(default, defining, False, opaque_lambdas=True)
            self.seed(target, _ASSIGN, default.lineno, sources)
            self.add_flow([*names, *others], target, _ASSIGN, default.lineno, False)

    def link_callees(self) -> None:
        """Flow each callee's argument slots into the parameters they bind.

        Parameter *i* reads positional slot *i*, the ``*args`` wildcards that
        start at or before *i*, its keyword slot and the ``**mapping``
        wildcards whose receiver offset is at most *i* (keywords and mappings
        never reach positional-only parameters).
        """
        index = self.index
        for group in index.groups:
            if not group.slots:
                continue
            slots = group.slots
            stars = [
                (start, slots[("*", start)])
                for start in range(_MAX_POSITIONAL_SLOTS + 1)
                if ("*", start) in slots
            ]
            mappings = [
                (offset, slots[("**", offset)]) for offset in (0, 1) if ("**", offset) in slots
            ]
            rest = slots.get("kw*")
            for function in group.functions:
                self.tick()
                args = function.node.args
                scope = function.scope
                positional = [*args.posonlyargs, *args.args]
                posonly = len(args.posonlyargs)
                for position, arg in enumerate(positional):
                    reads = [slots.get(position)]
                    reads.extend(key for start, key in stars if start <= position)
                    if position >= posonly:
                        reads.append(slots.get(("kw", arg.arg)))
                        reads.extend(key for offset, key in mappings if offset <= position)
                    self.bind_parameter(reads, index.key(scope, arg.arg))
                for arg in args.kwonlyargs:
                    reads = [slots.get(("kw", arg.arg)), *(key for _, key in mappings)]
                    self.bind_parameter(reads, index.key(scope, arg.arg))
                if args.vararg is not None:
                    reads = [key for _, key in stars]
                    reads.extend(
                        slots.get(i) for i in range(len(positional), _MAX_POSITIONAL_SLOTS)
                    )
                    self.bind_parameter(reads, index.key(scope, args.vararg.arg))
                if args.kwarg is not None:
                    reads = [rest, *(key for _, key in mappings)]
                    self.bind_parameter(reads, index.key(scope, args.kwarg.arg))

    def bind_parameter(self, reads: list[int | None], target: int) -> None:
        self.add_flow([key for key in reads if key is not None], (target,), _PASS, 0, False)

    def link_attributes(self) -> None:
        """A ``self.x`` read sees ``self.x`` stores in its class and in-file
        bases, and the class-body ``x`` of the nearest of them that binds it."""
        index = self.index
        ids = index._ids
        for node in index.attributes:
            self.tick()
            index.attribute_view(node)
        for cls, attr, view in index.view_list:
            self.tick()
            reads = []
            shadowed = False
            for owner in index.ancestors(cls, self.tick):
                store = ids.get(("store", owner.id, attr))
                if store is not None:
                    reads.append(store)
                if not shadowed:
                    body = ids.get(("class", owner.id, attr))
                    if body is not None:
                        reads.append(body)
                        shadowed = True
                    elif attr in owner.methods:
                        shadowed = True
            self.add_flow(reads, (view,), _PASS, 0, False)

    def build(self) -> None:
        index = self.index
        for class_key, module_key in index.class_reads:
            self.tick()
            self.add_flow((module_key,), (class_key,), _PASS, 0, True)
        self.collect_keywords()
        for node, scope in index.statements:
            self.tick()
            if isinstance(node, ast.Assign):
                self.assign(node, scope)
            elif isinstance(node, ast.Return):
                if node.value is not None:
                    self.returned(node.value, scope, node.lineno)
            elif isinstance(node, ast.Call):
                self.call(node, scope)
        for function in index.functions:
            self.tick()
            self.defaults(function)
        self.link_attributes()
        self.link_callees()

    # ── fixpoint ──

    def run(self) -> None:
        self.build()
        self.propagate()

    def propagate(self) -> None:
        # Phase A: the assignment flows earlier releases propagated, in order.
        self.drain(self.worklist, (self.prop_a,))
        # Phase B: every flow. Phase A facts are read again through the new
        # flows (each phase A flow already fired for them and is skipped).
        worklist: deque[tuple[int, int, int]] = deque(
            (key, 0, _ALL_LANES) for key in self.facts[0][_FIRST_LANE]
        )
        for targets, kind, line, fact, lanes in self.seeds:
            self.tick()
            worklist.extend(_mark_targets(self, targets, kind, line, fact, 0, lanes))
        self.drain(worklist, (self.prop_a, self.prop_b))

    def drain(
        self, worklist: deque[tuple[int, int, int]], tables: tuple[dict[int, list[int]], ...]
    ) -> None:
        flows = self.flows
        fired = self.fired
        if len(fired) < len(flows):
            fired.extend([0] * (len(flows) - len(fired)))
        facts = self.facts
        while worklist:
            self.tick()
            key, slot, lanes = worklist.popleft()
            held = facts[slot]
            # Lanes holding the same fact propagate it together.
            carried: list[list] = []
            for lane in _LANES:
                if lanes >> lane & 1:
                    fact = held[lane].get(key)
                    if fact is None:
                        continue
                    for entry in carried:
                        if entry[0] is fact:
                            entry[1] |= 1 << lane
                            break
                    else:
                        carried.append([fact, 1 << lane])
            for fact, fact_lanes in carried:
                ranks = _LANE_RANKS[fact[0]]
                # One bit per (lane, rank, slot): 14 bits per flow.
                want = 0
                for lane in _LANES:
                    if fact_lanes >> lane & 1:
                        want |= 1 << (6 * lane + 2 * ranks[lane] + slot)
                for table in tables:
                    for flow_id in table.get(key, ()):
                        mark = fired[flow_id]
                        todo_bits = want & ~mark
                        if not todo_bits:
                            # Already propagated a source of this rank: its targets
                            # hold one at least as severe. Skip to stay linear.
                            continue
                        targets, kind, line = flows[flow_id]
                        if slot and kind == _RETURN:
                            continue
                        fired[flow_id] = mark | todo_bits
                        todo = (
                            (1 if todo_bits & 0x3F else 0)
                            | (2 if todo_bits & 0xFC0 else 0)
                            | (4 if todo_bits & 0x3000 else 0)
                        )
                        worklist.extend(_mark_targets(self, targets, kind, line, fact, slot, todo))

    # ── results ──

    def fact(self, key: int, lane: int = 0) -> _Fact | None:
        """The most severe fact at *key* for *lane* (an ordinary one on ties)."""
        ordinary = self.facts[0][lane].get(key)
        bound = self.facts[1][lane].get(key)
        if bound is not None and (
            ordinary is None or _LANE_RANKS[bound[0]][lane] > _LANE_RANKS[ordinary[0]][lane]
        ):
            return bound
        return ordinary

    def cited(self, key: int, sink_name: str, lane: int) -> _Fact | None:
        """The fact a sink of *lane* reports for *key*.

        The ranked lane gives the most severe rule. The first source to
        arrive (lane 2, what earlier releases cite) is reported instead
        whenever it gives that same rule, so a more severe source replaces
        the cited one only when it changes the rule.
        """
        ranked = self.fact(key, lane)
        if ranked is None:
            return None
        first = self.fact(key, _FIRST_LANE)
        if first is not None and first is not ranked:
            rule = _pick_rule(_SOURCE_NAMES[ranked[0]], sink_name, is_direct=False)
            if _pick_rule(_SOURCE_NAMES[first[0]], sink_name, is_direct=False) == rule:
                return first
        return ranked

    def tainted_names(self) -> dict[str, _TaintedVar]:
        """Tainted variables by readable name (module globals keep their bare name)."""
        index = self.index
        result: dict[str, _TaintedVar] = {}
        for key in (*self.facts[0][0], *self.facts[1][0]):
            scope = index.key_scope[key]
            fact = self.fact(key)
            if scope is None or fact is None:
                continue
            name = index.key_name[key]
            if scope.kind == _MODULE:
                label = name
            elif scope.kind == _CLASS:
                label = f"{index.qualname(scope)}.{name}"
            else:
                label = f"{index.qualname(scope)}.<locals>.{name}"
            result.setdefault(label, _TaintedVar(name, _SOURCE_NAMES[fact[0]], fact[1]))
        return result


def _collect_tainted(
    tree: ast.AST,
    type_map: dict[str, str],
    aliases: dict[str, str],
    check_runtime: Callable[[], None] | None = None,
) -> dict[str, _TaintedVar]:
    """Compute taint for *tree*: readable variable name -> its fact.

    Module globals keep their bare name; function locals read
    ``qualname.<locals>.name``. See ``_TaintGraph`` for the propagation rules.
    """
    index = _build_scope_index(tree, check_runtime)
    graph = _TaintGraph(index, type_map, aliases, check_runtime)
    graph.run()
    return graph.tainted_names()


# Identifiers longer than this are shortened in messages (``prefix...``), so
# message bytes stay bounded per finding instead of growing with identifier
# length times the number of sinks. Real identifiers are far shorter, so their
# messages are unchanged.
_MAX_NAME_CHARS = 120


def _shorten(name: str) -> str:
    return name if len(name) <= _MAX_NAME_CHARS else name[: _MAX_NAME_CHARS - 3] + "..."


def _spelling(node: ast.AST) -> str:
    """How a sink spells what it reads: ``name``, ``self.x``, ``helper()``, ``self.m()``.

    Built from the sink's own nodes, so a message never copies an identifier
    spelled elsewhere in the file, and shortened to a bounded length.
    """
    if isinstance(node, ast.Name):
        return _shorten(node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return _shorten(f"{_shorten(node.value.id)}.{_shorten(node.attr)}")
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return _shorten(f"{_shorten(func.id)}()")
        if isinstance(func, ast.Attribute):
            value = func.value
            if isinstance(value, ast.Name):
                return _shorten(f"{_shorten(value.id)}.{_shorten(func.attr)}()")
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                return _shorten(f"{_shorten(value.func.id)}().{_shorten(func.attr)}()")
    return "<lambda>()"


def _find_tainted_names_in_args(
    node: ast.Call,
    graph: _TaintGraph,
    sink_name: str,
    check_runtime: Callable[[], None] | None = None,
) -> list[_TaintedVar]:
    """Find tainted variables, ``self.x`` attributes and helper results a sink reads.

    Plain variable reads come first, in ``ast.walk`` order, so a flow earlier
    releases reported keeps its message; attributes and in-file call results
    follow. Each is named as the sink spells it, and reported once.
    """
    tick = check_runtime or _noop
    index = graph.index
    lane = 1 if sink_name in _EXEC_SINKS or sink_name in _DESERIALIZATION_SINKS else 0
    seen: set[int] = set()
    names: list[_TaintedVar] = []
    others: list[_TaintedVar] = []
    for child in ast.walk(node):
        tick()
        if child is node:
            continue
        reads: list[tuple[int | None, ast.AST, list[_TaintedVar]]]
        if isinstance(child, ast.Name):
            reads = [(index.name_key.get(child), child, names)]
        elif isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name):
            reads = [(index.name_key.get(child.value), child.value, names)]
        elif isinstance(child, ast.Attribute):
            reads = [(index.views.get(child), child, others)]
        elif isinstance(child, ast.Call):
            reads = [
                (graph.returns.get(group.id), child, others)
                for group in index.call_targets(child, tick).results
            ]
        else:
            continue
        for key, spelled, out in reads:
            if key is None or key in seen:
                continue
            fact = graph.cited(key, sink_name, lane)
            if fact is None:
                continue
            seen.add(key)
            out.append(_TaintedVar(_spelling(spelled), _SOURCE_NAMES[fact[0]], fact[1]))
    return names + others


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
    check_runtime = budget.check_runtime if budget is not None else None
    index = _build_scope_index(tree, check_runtime)
    graph = _TaintGraph(index, type_map, aliases, check_runtime)
    graph.run()
    seen: set[tuple[str, ast.Call]] = set()
    contexts: dict[int, str] = {}

    def context_for(lineno: int) -> str:
        context = contexts.get(lineno)
        if context is None:
            context = get_context_from_lines(lines, lineno)
            contexts[lineno] = context
        return context

    def _emit(
        node_index: int,
        rule_id: str,
        ast_node: ast.Call,
        msg: str,
    ) -> None:
        lineno = getattr(ast_node, "lineno", 1)
        end_lineno = getattr(ast_node, "end_lineno", None)
        start_byte_column = getattr(ast_node, "col_offset", None)
        end_byte_column = getattr(ast_node, "end_col_offset", None)
        # Deduplicate flows into the same sink without merging distinct nodes
        # whose optional source columns are unavailable.
        key = (rule_id, ast_node)
        if key in seen:
            return
        seen.add(key)
        complete_match = python_ast.source_segment(ast_node)
        if complete_match is None:
            complete_match = get_complete_source_segment(lines, lineno, end_lineno)
        start_column = python_ast.character_column(lineno, start_byte_column)
        end_column = python_ast.character_column(end_lineno or lineno, end_byte_column)
        finding = AnalyzerFinding(
            rule_id=rule_id,
            message=msg,
            severity=_RULE_SEVERITIES[rule_id],
            location=Location(
                file=file_path,
                start_line=lineno,
                end_line=end_lineno,
                start_column=start_column,
                end_column=end_column,
            ),
            confidence=_RULE_CONFIDENCES[rule_id],
            tags=[_TAG],
            context=context_for(lineno),
            matched_text=complete_match[:200],
            complete_match=complete_match,
            # Evidence participates in report compaction. Keep distinct syntax
            # nodes distinguishable when their source spans are incomplete.
            evidence=(
                {"python_ast_node_index": node_index}
                if start_column is None or end_column is None
                else {}
            ),
        )
        if budget is None:
            findings.append(finding)
        else:
            budget.emit(finding)

    # Taint is fully computed above, independent of traversal order, so this
    # pass only needs to check sink call sites against it.
    for node_index, ast_node in enumerate(ast.walk(tree)):
        if budget is not None:
            budget.check_runtime()

        if not isinstance(ast_node, ast.Call):
            continue

        sink_name = _resolve_sink_name(ast_node, type_map, aliases)
        if not sink_name or sink_name not in _ALL_SINKS:
            continue

        if sink_name == "open" and not _is_open_for_write(ast_node):
            continue

        for src_name, src_node in _find_nested_sources(
            ast_node,
            type_map,
            aliases,
            check_runtime,
        ):
            if src_name == "open" and _is_open_for_write(src_node):
                continue
            rule = _pick_rule(src_name, sink_name, is_direct=True)
            src_cat = _classify(src_name, _SOURCE_CATEGORIES, "data source")
            sink_cat = _classify(sink_name, _SINK_CATEGORIES, "data sink")
            _emit(
                node_index,
                rule,
                ast_node,
                f"Direct flow: {src_name} ({src_cat}) \u2192 {sink_name} ({sink_cat})",
            )

        for tv in _find_tainted_names_in_args(ast_node, graph, sink_name, check_runtime):
            rule = _pick_rule(tv.source_call, sink_name, is_direct=False)
            src_cat = _classify(tv.source_call, _SOURCE_CATEGORIES, "data source")
            sink_cat = _classify(sink_name, _SINK_CATEGORIES, "data sink")
            _emit(
                node_index,
                rule,
                ast_node,
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
    raw_file_cache = state.get("raw_file_cache")
    source_classifications = (
        state.get("python_source_classifications")
        if "python_source_classifications" in state
        else None
    )
    source_classification_limitations = state.get("python_source_classification_limitations") or {}
    source_decode_failures = state.get("python_source_decode_failures") or {}
    python_ast_cache_key = state.get("python_ast_cache_key")
    all_findings: list[Finding] = []
    ledger_events: list[InspectionLedgerEvent] = []
    budget = _BehavioralBudget(state)
    terminal_limit: _BehavioralResourceLimitError | None = None

    for path in components:
        content = file_cache.get(path)
        source_classification: PythonSourceClassification | None = None
        if source_classifications is not None and path in source_classifications:
            source_classification = resolve_python_source_classification(
                path,
                content,
                source_classifications=source_classifications,
                raw_file_cache=raw_file_cache,
            )
            if source_classification is PythonSourceClassification.NON_PYTHON:
                continue
        if path in source_classification_limitations:
            ledger_events.append(
                ledger_event(
                    outcome=LedgerOutcome.PARTIAL,
                    phase="behavioral",
                    analyzer_id=ANALYZER_ID,
                    path=path,
                    reason=LedgerReason.RUNTIME_LIMIT,
                )
            )
            continue
        if terminal_limit is None and budget.analyzer_exhausted():
            terminal_limit = _BehavioralResourceLimitError(
                LedgerReason.OUTPUT_LIMIT,
                {
                    "observed_findings": budget.total_findings + 1,
                    "limit_findings": MAX_FINDINGS_PER_ANALYZER,
                },
            )
        if terminal_limit is None:
            try:
                budget.check_runtime()
            except _BehavioralResourceLimitError as exc:
                terminal_limit = exc
        if terminal_limit is not None:
            event = _partial_limit_event(path, terminal_limit)
            ledger_events.append(event)
            continue
        if path in source_decode_failures:
            ledger_events.append(
                ledger_event(
                    outcome=LedgerOutcome.PARTIAL,
                    phase="behavioral",
                    analyzer_id=ANALYZER_ID,
                    path=path,
                    reason=LedgerReason.PYTHON_SOURCE_DECODE_ERROR,
                )
            )
            continue
        if source_classification is None:
            source_classification = resolve_python_source_classification(
                path,
                content,
                source_classifications=source_classifications,
                raw_file_cache=raw_file_cache,
            )
            try:
                budget.check_runtime()
            except _BehavioralResourceLimitError as exc:
                terminal_limit = exc
                ledger_events.append(_partial_limit_event(path, exc))
                continue
        if source_classification is PythonSourceClassification.NON_PYTHON:
            continue
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
                    outcome=(
                        LedgerOutcome.PARTIAL
                        if source_classification is PythonSourceClassification.AMBIGUOUS
                        else LedgerOutcome.COMPLETED
                    ),
                    phase="behavioral",
                    analyzer_id=ANALYZER_ID,
                    path=path,
                    reason=(
                        LedgerReason.PYTHON_SOURCE_AMBIGUOUS
                        if source_classification is PythonSourceClassification.AMBIGUOUS
                        else None
                    ),
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
