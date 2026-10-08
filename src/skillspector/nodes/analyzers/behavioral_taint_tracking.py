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
to flag potential credential/data exfiltration chains. Taint follows Python's
lexical scopes and crosses calls to functions defined in the same file
(arguments into parameters, return values back to callers).
"""

from __future__ import annotations

import ast
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
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


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


class _TaintedVar(NamedTuple):
    """One reported taint fact: data from ``source_call`` reaches ``name``.

    ``lineno`` is the line of the statement that last tainted ``name`` (the
    assignment, the call that bound a parameter), the wording earlier releases
    used, so messages, exact baselines and message-glob rules keep matching.
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


def _direct_sources(
    node: ast.expr,
    type_map: dict[str, str] | None,
    aliases: dict[str, str],
    check_runtime: Callable[[], None] | None = None,
    stop: Callable[[ast.AST], bool] | None = None,
) -> list[tuple[str, ast.AST]]:
    """Every distinct source a value reads directly, in ``ast.walk`` order.

    Finds source calls anywhere in the expression (``open("f").read()``,
    ``requests.get(url).text``, ``os.environ.get("K")``) and an
    ``os.environ["K"]`` subscript (also ``os`` aliased) that is the whole
    value. Returns ``(source name, source node)`` for the first occurrence of
    each source. *stop* prunes subtrees whose sources are accounted elsewhere.
    """
    found: dict[str, ast.AST] = {}
    if stop is not None and stop(node):
        return []
    queue: deque[ast.AST] = deque([node])
    while queue:
        if check_runtime is not None:
            check_runtime()
        child = queue.popleft()
        if isinstance(child, ast.Call):
            name = _call_source(child, type_map, aliases)
            if name is not None and name not in found:
                found[name] = child
        for grandchild in ast.iter_child_nodes(child):
            if isinstance(grandchild, ast.expr_context):
                continue
            if stop is not None and stop(grandchild):
                continue
            queue.append(grandchild)
    if isinstance(node, ast.Subscript):
        base = resolve_dotted_name(node.value)
        if base is not None:
            base = apply_import_aliases(base, aliases)
        if base and base in _CREDENTIAL_SOURCES and base not in found:
            found[base] = node
    return list(found.items())


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


# ── Lexical scopes ──────────────────────────────────────────────────────
#
# Taint is keyed by lexical scope rather than by bare name, so a tainted local
# in one function cannot taint an unrelated same-named variable or parameter
# in another. Keys are small integers (see ``_Keys``) built from a scope or
# class id plus the identifier the source spells; no key ever embeds a
# qualified name, so memory stays linear in the file even for very long names.
#
# * Names follow Python's rules: parameters and names bound in a function,
#   lambda or comprehension are local to it unless declared ``global`` /
#   ``nonlocal``; free names resolve through enclosing functions to the
#   module; class bodies are visible only to their own statements.
# * ``self.x`` / ``cls.x`` / ``Class.x`` and class-body names are attributes of
#   the class. A store in class C reaches a read in class E when C and E share
#   a descendant (E is C, a subclass, a base, or a mixin combined with C), so
#   sibling subclasses stay apart. Very large hierarchies share one namespace.
# * Calls to functions and methods defined in the file bind their arguments to
#   the callee's parameters through per-callee argument slots, and read the
#   callee's return value. Taint that reached a function only through its own
#   parameters never enters its return value: each call site already reads its
#   own arguments, so one caller's secret cannot taint other callers' results.

_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
_Comprehension = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
# Values that can never be an instance of a class defined in this file.
_LITERAL_VALUES = (
    ast.Constant,
    ast.JoinedStr,
    ast.Dict,
    ast.List,
    ast.Set,
    ast.Tuple,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)

# Call-site positional arguments at or beyond this parameter index (and
# anything after a ``*args`` unpacking) share one wildcard slot, keeping the
# per-definition binding work bounded by a constant.
_MAX_POSITIONAL_SLOTS = 16

# Inheritance components up to this many classes track attributes per class;
# larger ones share one attribute namespace, keeping the work linear.
_PRECISE_HIERARCHY_LIMIT = 32

# Higher-order calls that run a callback with the remaining arguments, and the
# position of the callback among their positional arguments.
_CALLBACK_POSITIONS: dict[str, int] = {
    "Thread": 1,
    "Process": 1,
    "Timer": 1,
    "submit": 0,
    "map": 0,
    "imap": 0,
    "imap_unordered": 0,
    "starmap": 0,
    "apply": 0,
    "apply_async": 0,
    "map_async": 0,
    "starmap_async": 0,
    "run_in_executor": 1,
    "to_thread": 0,
    "call_soon": 0,
    "call_soon_threadsafe": 0,
    "call_later": 1,
    "call_at": 1,
    "partial": 0,
    "start_new_thread": 0,
}
_CALLBACK_KEYWORDS = frozenset({"target", "function", "func", "fn", "callback"})
_PROPERTY_DECORATORS = frozenset({"property", "cached_property", "abstractproperty"})

_NO_OWNER = -1

# How a flow changes the facts it carries (see ``_mark_targets``).
_ASSIGN = 0  # cites the flow's own line
_PASS = 1  # keeps the line of the fact it forwards
_RETURN = 2  # into a function's return value
_BIND = 3  # into a parameter, from a call site

_SOURCE_NAMES = tuple(sorted(_ALL_SOURCES))
_SOURCE_INDEX = {name: index for index, name in enumerate(_SOURCE_NAMES)}
# A fact label is ``2 * source index + from_parameter``.
_LABELS = 2 * len(_SOURCE_NAMES)


class _Keys:
    """Dense integer ids for taint keys, with their owner and display name.

    A key's owner is the function whose locals it belongs to (or
    ``_NO_OWNER``); its display is the spelling used in messages, kept as a
    reference to identifiers already in the tree and joined only on output.
    """

    def __init__(self) -> None:
        self.ids: dict[tuple[object, ...], int] = {}
        self.owner: list[int] = []
        self.display: list[str | tuple[str, ...]] = []
        self.describe: list[tuple[object, ...]] = []

    def get(
        self, descriptor: tuple[object, ...], owner: int, display: str | tuple[str, ...]
    ) -> int:
        key = self.ids.get(descriptor)
        if key is None:
            key = self.fresh(owner, display, descriptor)
            self.ids[descriptor] = key
        return key

    def fresh(
        self, owner: int, display: str | tuple[str, ...], descriptor: tuple[object, ...]
    ) -> int:
        key = len(self.owner)
        self.owner.append(owner)
        self.display.append(display)
        self.describe.append(descriptor)
        return key

    def name(self, key: int) -> str:
        display = self.display[key]
        return display if isinstance(display, str) else "".join(display)


@dataclass(eq=False)
class _Scope:
    """One Python namespace: module, function, lambda, comprehension or class body."""

    kind: str
    name: str
    parent: _Scope | None
    id: int
    # Function whose locals live here (comprehensions belong to theirs).
    owner: int
    bound: set[str] = field(default_factory=set)
    declared_global: set[str] = field(default_factory=set)
    declared_nonlocal: set[str] = field(default_factory=set)
    class_info: _ClassInfo | None = None
    function: _FunctionInfo | None = None


@dataclass(eq=False)
class _ClassInfo:
    node: ast.ClassDef
    scope: _Scope
    parent: _Scope
    logical: _LogicalClass | None = None


@dataclass(eq=False)
class _LogicalClass:
    """Every ``class`` statement bound to one name in one scope (redefinitions merge)."""

    id: int
    name: str
    infos: list[_ClassInfo] = field(default_factory=list)
    bases: list[_LogicalClass] = field(default_factory=list)
    subclasses: list[_LogicalClass] = field(default_factory=list)
    methods: set[str] = field(default_factory=set)
    properties: set[str] = field(default_factory=set)
    component: int = -1
    # Attribute namespace: the class itself, or its whole component when large.
    unit: int = -1
    precise: bool = True
    mro: list[_LogicalClass] = field(default_factory=list)
    ancestors: set[int] = field(default_factory=set)
    descendants: set[int] = field(default_factory=set)
    # Classes sharing a descendant with this one (including itself).
    sharing: list[_LogicalClass] = field(default_factory=list)
    sharing_ids: set[int] = field(default_factory=set)


@dataclass(eq=False)
class _FunctionInfo:
    node: _FunctionNode | ast.Lambda
    scope: _Scope
    kind: str = "function"  # function | instance | class | static
    # Groups whose return value this function produces.
    groups: list[_Group] = field(default_factory=list)


@dataclass(eq=False)
class _Group:
    """Functions reached through one callable name; they share argument slots."""

    id: int
    kind: str
    label: str
    functions: list[_FunctionInfo] = field(default_factory=list)
    keyword_params: set[str] = field(default_factory=set)
    has_kwarg: bool = False
    slots: dict[object, int] = field(default_factory=dict)
    returns: int | None = None

    def add(self, function: _FunctionInfo) -> None:
        self.functions.append(function)
        args = function.node.args
        self.keyword_params.update(arg.arg for arg in args.args)
        self.keyword_params.update(arg.arg for arg in args.kwonlyargs)
        self.has_kwarg = self.has_kwarg or args.kwarg is not None


@dataclass(eq=False)
class _BindingSet:
    """The callees one call site binds into, each with its receiver offset."""

    id: int
    members: tuple[tuple[_Group, int], ...]
    targets: dict[object, tuple[int, ...]] = field(default_factory=dict)


@dataclass(eq=False)
class _CallInfo:
    binding: _BindingSet | None
    results: tuple[_Group, ...]
    # Key of the call's value (return plus arguments), for in-file callees.
    value: int | None


@dataclass(eq=False)
class _TypeInfo:
    """What a variable or attribute is assigned, as far as call binding cares.

    A recorded binding that is neither an in-file instance nor unknown is a
    literal, library object, function or module: no in-file method.
    """

    classes: list[_LogicalClass] = field(default_factory=list)
    dynamic: bool = False  # instance of a class or of any subclass (``cls()``)
    unknown: bool = False  # some value of unknown type


class _Receiver(NamedTuple):
    classes: list[_LogicalClass]
    dynamic: bool  # may be any subclass instance (``self``, ``cls()``)
    class_object: bool  # the class itself rather than an instance
    fallback: bool  # unknown: may be an instance of any class in the file


_UNKNOWN_VALUE = object()
_KNOWN_VALUE = object()


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


def _receiver_offset(class_object: bool, kind: str) -> int:
    """Leading parameters a call binds implicitly (``self`` / ``cls``)."""
    if kind == "class":
        return 1
    if kind == "instance":
        return 0 if class_object else 1
    return 0


def _root_name(node: ast.expr) -> str | None:
    """The name at the root of ``a.b().c[0]``, if any."""
    while True:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        else:
            return None


def _is_super_call(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "super"
    )


def _callee_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _noop() -> None:
    return None


def _linearize(cls: _LogicalClass) -> list[_LogicalClass]:
    """C3 method resolution order over in-file bases; DFS order if inconsistent."""
    order: list[_LogicalClass] = []
    pending: list[tuple[_LogicalClass, bool]] = [(cls, False)]
    linearized: dict[int, list[_LogicalClass]] = {}
    visiting: set[int] = set()
    while pending:
        current, expanded = pending.pop()
        if current.id in linearized:
            continue
        if not expanded:
            if current.id in visiting:  # inheritance cycle through redefinitions
                linearized[current.id] = [current]
                continue
            visiting.add(current.id)
            pending.append((current, True))
            pending.extend((base, False) for base in reversed(current.bases))
            continue
        visiting.discard(current.id)
        sequences = [list(linearized.get(base.id, [base])) for base in current.bases]
        sequences.append(list(current.bases))
        merged = [current]
        while True:
            sequences = [sequence for sequence in sequences if sequence]
            if not sequences:
                break
            for sequence in sequences:
                head = sequence[0]
                if not any(head in other[1:] for other in sequences):
                    break
            else:
                merged = []
                break
            merged.append(head)
            for sequence in sequences:
                if sequence[0] is head:
                    del sequence[0]
        if not merged:
            seen: set[int] = set()
            stack = [current]
            while stack:
                item = stack.pop()
                if item.id in seen:
                    continue
                seen.add(item.id)
                merged.append(item)
                stack.extend(reversed(item.bases))
        linearized[current.id] = merged
    order = linearized.get(cls.id, [cls])
    return order


@dataclass(eq=False)
class _ScopeIndex:
    """Lexical scopes of one module plus what is needed to bind call sites."""

    aliases: dict[str, str]
    type_map: dict[str, str]
    keys: _Keys = field(default_factory=_Keys)
    scopes: list[_Scope] = field(default_factory=list)
    # Assign / Return / Call / Lambda / comprehension nodes with their scope,
    # in ``ast.walk`` order.
    statements: list[tuple[ast.AST, _Scope]] = field(default_factory=list)
    call_scope: dict[ast.Call, _Scope] = field(default_factory=dict)
    # Nodes evaluated in a different scope than their parent (lambda bodies,
    # comprehension elements, targets and conditions).
    scope_switch: dict[ast.AST, _Scope] = field(default_factory=dict)
    functions: list[_FunctionInfo] = field(default_factory=list)
    lambdas: dict[ast.Lambda, _FunctionInfo] = field(default_factory=dict)
    classes: list[_ClassInfo] = field(default_factory=list)
    logical: list[_LogicalClass] = field(default_factory=list)
    groups: dict[tuple[object, ...], _Group] = field(default_factory=dict)
    function_groups: dict[int, list[_Group]] = field(default_factory=dict)
    unit_groups: dict[tuple[int, str], list[_Group]] = field(default_factory=dict)
    fallback_groups: dict[str, list[_Group]] = field(default_factory=dict)
    definers: dict[tuple[int, str], list[_LogicalClass]] = field(default_factory=dict)
    classes_by_key: dict[int, list[_LogicalClass]] = field(default_factory=dict)
    # Key of each method's receiver parameter -> (class, whether it is ``cls``).
    self_keys: dict[int, tuple[_LogicalClass, bool]] = field(default_factory=dict)
    var_types: dict[int, _TypeInfo] = field(default_factory=dict)
    # Raw bindings recorded while scoping, typed once classes are known.
    name_stores: list[tuple[_Scope, str, object]] = field(default_factory=list)
    attribute_stores: list[tuple[_Scope, ast.Attribute, object]] = field(default_factory=list)
    lambda_bindings: list[tuple[_Scope, str, ast.Lambda]] = field(default_factory=list)
    attribute_loads: list[tuple[ast.Attribute, _Scope]] = field(default_factory=list)
    attribute_reads: dict[tuple[int, str], int] = field(default_factory=dict)
    binding_sets: dict[tuple[tuple[int, int], ...], _BindingSet] = field(default_factory=dict)
    alias_roots: set[str] = field(default_factory=set)
    _resolved: dict[tuple[int, str], int] = field(default_factory=dict)
    _calls: dict[ast.Call, _CallInfo] = field(default_factory=dict)
    _methods: dict[tuple[int, str, bool], list[_Group]] = field(default_factory=dict)

    @property
    def module(self) -> _Scope:
        return self.scopes[0]

    def new_scope(self, kind: str, name: str, parent: _Scope | None) -> _Scope:
        scope_id = len(self.scopes)
        if kind in ("function", "lambda"):
            owner = scope_id
        elif kind == "comprehension" and parent is not None:
            owner = parent.owner
        else:
            owner = _NO_OWNER
        scope = _Scope(kind, name, parent, scope_id, owner)
        self.scopes.append(scope)
        return scope

    # ── names ──

    def binding_scope(self, scope: _Scope, name: str) -> _Scope:
        """The scope *name* is bound in when read or written in *scope*.

        Follows Python's rules: a name bound anywhere in a function body is
        local to it unless declared ``global``/``nonlocal``; free names
        resolve through enclosing function scopes (closures) to the module.
        Class bodies are visible only to their own statements, not to methods.
        """
        current: _Scope | None = scope
        while current is not None and current.kind != "module":
            if current is scope or current.kind != "class":
                if name in current.declared_global:
                    break
                if name in current.bound and name not in current.declared_nonlocal:
                    return current
            current = current.parent
        return self.module

    def binding_key(self, scope: _Scope, name: str) -> int:
        """Key of *name* bound in *scope* itself."""
        if scope.kind == "class" and scope.class_info is not None:
            logical = scope.class_info.logical
            if logical is not None:
                return self.keys.get((-1, logical.unit, name), _NO_OWNER, name)
        return self.keys.get((scope.id, name), scope.owner, name)

    def resolve(self, scope: _Scope, name: str) -> int:
        """Return the key *name* refers to when read or written in *scope*."""
        cache_key = (scope.id, name)
        key = self._resolved.get(cache_key)
        if key is None:
            key = self.binding_key(self.binding_scope(scope, name), name)
            self._resolved[cache_key] = key
        return key

    def is_module_import(self, scope: _Scope, name: str) -> bool:
        """Whether *name* is an imported module/object or a builtin, not a local."""
        bound = self.binding_scope(scope, name)
        if bound.kind != "module":
            return False
        return name in self.alias_roots or name not in bound.bound

    # ── classes and attributes ──

    def receiver_classes(self, scope: _Scope, name: str) -> list[_LogicalClass]:
        """Classes whose attribute namespace ``name.x`` refers to.

        ``self``/``cls`` in a method, a class itself, or a variable assigned an
        instance of an in-file class in its own scope (``cfg = Config()``).
        """
        key = self.resolve(scope, name)
        entry = self.self_keys.get(key)
        if entry is not None:
            return [entry[0]]
        classes = self.classes_by_key.get(key)
        if classes:
            return classes
        info = self.var_types.get(key)
        return info.classes if info is not None else []

    def attribute_store(self, node: ast.Attribute, scope: _Scope) -> list[int]:
        """Keys written by ``self.x = ...`` / ``cls.x = ...`` / ``Class.x = ...``."""
        if not isinstance(node.value, ast.Name):
            return []
        display = (node.value.id, ".", node.attr)
        return [
            self.keys.get((-1, cls.unit, node.attr), _NO_OWNER, display)
            for cls in self.receiver_classes(scope, node.value.id)
        ]

    def attribute_load(self, node: ast.Attribute, scope: _Scope) -> list[int]:
        """Keys read by ``self.x`` / ``cls.x`` / ``Class.x``."""
        if not isinstance(node.value, ast.Name):
            return []
        classes = self.receiver_classes(scope, node.value.id)
        if not classes:
            return []
        display = (node.value.id, ".", node.attr)
        keys: list[int] = []
        for cls in classes:
            if not cls.precise:
                keys.append(self.keys.get((-1, cls.unit, node.attr), _NO_OWNER, display))
                continue
            read_key = (cls.id, node.attr)
            key = self.attribute_reads.get(read_key)
            if key is None:
                key = self.keys.get((-2, cls.id, node.attr), _NO_OWNER, display)
                self.attribute_reads[read_key] = key
            keys.append(key)
        return keys

    def attribute_type(self, cls: _LogicalClass, attr: str) -> _TypeInfo:
        """Union of what ``self.<attr>`` is assigned across the classes that see it."""
        result = _TypeInfo()
        recorded = False
        for other in cls.sharing if cls.precise else [cls]:
            key = self.keys.ids.get((-1, other.unit, attr))
            info = self.var_types.get(key) if key is not None else None
            if info is None:
                continue
            recorded = True
            result.classes.extend(c for c in info.classes if c not in result.classes)
            result.dynamic = result.dynamic or info.dynamic
            result.unknown = result.unknown or info.unknown
        if not recorded:
            result.unknown = True
        return result

    # ── call resolution ──

    def group(self, identity: tuple[object, ...], kind: str, label: str) -> _Group:
        group = self.groups.get(identity)
        if group is None:
            group = _Group(len(self.groups), kind, label)
            self.groups[identity] = group
        return group

    def method_groups(self, cls: _LogicalClass, name: str, *, exact: bool) -> list[_Group]:
        """Groups ``obj.<name>`` may dispatch to when ``obj`` is a *cls* instance.

        *exact* means the receiver is exactly *cls* (``Cls()``, ``obj = Cls()``);
        otherwise it may be any subclass (``self``, ``cls``), so overrides in
        subclasses and in mixins combined with *cls* are included too.
        """
        cache_key = (cls.id, name, exact)
        cached = self._methods.get(cache_key)
        if cached is not None:
            return cached
        result: list[_Group] = []
        if not cls.precise:
            result.extend(self.unit_groups.get((cls.unit, name), ()))
        else:
            first = next((c for c in cls.mro if name in c.methods), None)
            if first is not None:
                result.extend(self.unit_groups.get((first.unit, name), ()))
            if not exact:
                for other in self.definers.get((cls.component, name), ()):
                    if (
                        other is not first
                        and other.id in cls.sharing_ids
                        and other.id not in cls.ancestors
                    ):
                        result.extend(self.unit_groups.get((other.unit, name), ()))
        self._methods[cache_key] = result
        return result

    def super_groups(self, cls: _LogicalClass, name: str) -> list[_Group]:
        """Groups ``super().<name>`` may reach from a method of *cls*."""
        if not cls.precise:
            return list(self.unit_groups.get((cls.unit, name), ()))
        result: list[_Group] = []
        first = next((c for c in cls.mro[1:] if name in c.methods), None)
        if first is not None:
            result.extend(self.unit_groups.get((first.unit, name), ()))
        for other in self.definers.get((cls.component, name), ()):
            if (
                other is not first
                and other.id in cls.sharing_ids
                and other.id not in cls.ancestors
                and other.id not in cls.descendants
            ):
                result.extend(self.unit_groups.get((other.unit, name), ()))
        return result

    def super_class(self, call: ast.Call, scope: _Scope) -> _LogicalClass | None:
        if call.args:
            first = call.args[0]
            if isinstance(first, ast.Name):
                classes = self.classes_by_key.get(self.resolve(scope, first.id))
                return classes[0] if classes else None
            return None
        current: _Scope | None = scope
        while current is not None and current.kind in ("comprehension", "lambda"):
            current = current.parent
        if current is None or current.kind != "function":
            return None
        parent = current.parent
        if parent is None or parent.class_info is None:
            return None
        return parent.class_info.logical

    def value_type(self, value: ast.expr, scope: _Scope) -> object:
        """Classes *value* is an instance of: a ``(classes, dynamic)`` pair,
        ``_KNOWN_VALUE`` (literal, library object, function) or ``_UNKNOWN_VALUE``."""
        if isinstance(value, _LITERAL_VALUES) or isinstance(value, ast.Lambda):
            return _KNOWN_VALUE
        if not isinstance(value, ast.Call):
            return _UNKNOWN_VALUE
        func = value.func
        if isinstance(func, ast.Name):
            key = self.resolve(scope, func.id)
            classes = self.classes_by_key.get(key)
            if classes:
                return classes, False
            entry = self.self_keys.get(key)
            if entry is not None and entry[1]:
                return [entry[0]], True
            if key in self.function_groups:
                return _UNKNOWN_VALUE
            return _KNOWN_VALUE if self.is_module_import(scope, func.id) else _UNKNOWN_VALUE
        root = _root_name(func)
        if root is not None and root in self.alias_roots and self.is_module_import(scope, root):
            return _KNOWN_VALUE
        return _UNKNOWN_VALUE

    def receiver(self, node: ast.expr, scope: _Scope) -> _Receiver:
        """What ``node.m(...)`` may dispatch on."""
        none = _Receiver([], False, False, False)
        if isinstance(node, ast.Name):
            key = self.resolve(scope, node.id)
            entry = self.self_keys.get(key)
            if entry is not None:
                return _Receiver([entry[0]], True, entry[1], False)
            classes = self.classes_by_key.get(key)
            if classes:
                return _Receiver(classes, False, True, False)
            info = self.var_types.get(key)
            if info is None:
                # Never bound in this file: a builtin or a star import.
                unbound_local = not self.is_module_import(scope, node.id)
                return _Receiver([], False, False, unbound_local)
            if info.classes:
                return _Receiver(info.classes, info.dynamic, False, False)
            return _Receiver([], False, False, info.unknown)
        if isinstance(node, _LITERAL_VALUES):
            return none
        if isinstance(node, ast.Call):
            found = self.value_type(node, scope)
            if isinstance(found, tuple):
                classes, dynamic = found
                return _Receiver(classes, dynamic, False, False)
            return _Receiver([], False, False, found is _UNKNOWN_VALUE)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            classes = self.receiver_classes(scope, node.value.id)
            if classes:
                result = _TypeInfo()
                for cls in classes:
                    info = self.attribute_type(cls, node.attr)
                    result.classes.extend(c for c in info.classes if c not in result.classes)
                    result.dynamic = result.dynamic or info.dynamic
                    result.unknown = result.unknown or info.unknown
                if result.classes:
                    return _Receiver(result.classes, result.dynamic, False, False)
                return _Receiver([], False, False, result.unknown)
        root = _root_name(node)
        if root is not None and root in self.alias_roots and self.is_module_import(scope, root):
            return none
        return _Receiver([], False, False, True)

    def resolve_callee(
        self, func: ast.expr, scope: _Scope, *, fallback: bool
    ) -> tuple[list[tuple[_Group, int]], list[_Group]]:
        """Callees ``func(...)`` may run: ``(group, receiver offset)`` pairs to
        bind arguments into, and the groups whose return value the call yields.

        A plain name resolves lexically to the functions bound to it, or to the
        ``__init__`` of a class it names (``cls(...)`` in a classmethod too).
        ``obj.m`` resolves through ``obj``'s class: ``self``/``cls``, a class,
        an instance built in this scope or stored on ``self``, or
        ``super()``. With *fallback*, a receiver of unknown type binds into
        every method named ``m`` in the file, but its return value is not read.
        """
        members: list[tuple[_Group, int]] = []
        results: list[_Group] = []
        if isinstance(func, ast.Name):
            key = self.resolve(scope, func.id)
            for group in self.function_groups.get(key, ()):
                members.append((group, 0))
                results.append(group)
            constructed = list(self.classes_by_key.get(key, ()))
            exact = True
            entry = self.self_keys.get(key)
            if entry is not None and entry[1]:
                constructed.append(entry[0])
                exact = False
            for cls in constructed:
                for group in self.method_groups(cls, "__init__", exact=exact):
                    members.append((group, _receiver_offset(False, group.kind)))
            return members, results
        if not isinstance(func, ast.Attribute):
            return members, results
        if _is_super_call(func.value):
            assert isinstance(func.value, ast.Call)
            cls = self.super_class(func.value, scope)
            if cls is not None:
                for group in self.super_groups(cls, func.attr):
                    members.append((group, _receiver_offset(False, group.kind)))
                    results.append(group)
            return members, results
        receiver = self.receiver(func.value, scope)
        if receiver.classes:
            for cls in receiver.classes:
                for group in self.method_groups(cls, func.attr, exact=not receiver.dynamic):
                    members.append((group, _receiver_offset(receiver.class_object, group.kind)))
                    results.append(group)
        elif receiver.fallback and fallback and not func.attr.startswith("__"):
            # Dunder methods run through syntax (``Cls(...)``, ``super()``),
            # not through explicit calls on unknown objects.
            for group in self.fallback_groups.get(func.attr, ()):
                members.append((group, _receiver_offset(False, group.kind)))
        return members, results

    def binding_set(self, members: list[tuple[_Group, int]]) -> _BindingSet:
        identity = tuple(dict.fromkeys((group.id, offset) for group, offset in members))
        bset = self.binding_sets.get(identity)
        if bset is None:
            by_id = {group.id: group for group, _ in members}
            bset = _BindingSet(
                len(self.binding_sets),
                tuple((by_id[group_id], offset) for group_id, offset in identity),
            )
            self.binding_sets[identity] = bset
        return bset

    def call_info(self, call: ast.Call, scope: _Scope) -> _CallInfo:
        info = self._calls.get(call)
        if info is None:
            members, results = self.resolve_callee(call.func, scope, fallback=True)
            binding = self.binding_set(members) if members else None
            value = None
            if members:
                name = _callee_name(call.func) or "<call>"
                value = self.keys.fresh(scope.owner, (name, "()"), ("call", call))
            info = _CallInfo(binding, tuple(dict.fromkeys(results)), value)
            self._calls[call] = info
        return info

    def is_value_call(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        scope = self.call_scope.get(node)
        return scope is not None and self.call_info(node, scope).value is not None

    # ── reads ──

    def flow_reads(
        self, node: ast.AST, scope: _Scope, check_runtime: Callable[[], None]
    ) -> Iterator[int]:
        """Keys an expression's value depends on, for building flows.

        Names resolve through their scope; ``self.x`` resolves to the class
        attribute; a call to a function defined in this file is read through
        its value key and not re-walked, keeping nested calls linear.
        """
        switch = self.scope_switch
        queue: deque[tuple[ast.AST, _Scope]] = deque([(node, scope)])
        while queue:
            check_runtime()
            child, current = queue.popleft()
            if isinstance(child, ast.Name):
                yield self.resolve(current, child.id)
                continue
            if isinstance(child, ast.Call):
                info = self.call_info(child, current)
                if info.value is not None:
                    yield info.value
                    continue
            elif isinstance(child, ast.Attribute):
                yield from self.attribute_load(child, current)
            for grandchild in ast.iter_child_nodes(child):
                if not isinstance(grandchild, ast.expr_context):
                    queue.append((grandchild, switch.get(grandchild, current)))

    def references(
        self,
        node: ast.AST,
        scope: _Scope,
        check_runtime: Callable[[], None],
        *,
        skip_root: bool = False,
    ) -> Iterator[tuple[int, bool]]:
        """Yield ``(key, plain_name)`` for every key an expression reads.

        Walks in ``ast.walk`` order. ``plain_name`` marks keys read through a
        variable name (or ``name[...]``), the only reads earlier releases saw;
        ``self.x`` and calls to functions defined in this file are the rest.
        """
        switch = self.scope_switch
        queue: deque[tuple[ast.AST, _Scope]] = deque([(node, scope)])
        while queue:
            check_runtime()
            child, current = queue.popleft()
            if not (skip_root and child is node):
                if isinstance(child, ast.Name):
                    yield self.resolve(current, child.id), True
                elif isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name):
                    yield self.resolve(current, child.value.id), True
                elif isinstance(child, ast.Attribute):
                    for key in self.attribute_load(child, current):
                        yield key, False
                elif isinstance(child, ast.Call):
                    info = self.call_info(child, current)
                    if info.value is not None:
                        yield info.value, False
            for grandchild in ast.iter_child_nodes(child):
                if not isinstance(grandchild, ast.expr_context):
                    queue.append((grandchild, switch.get(grandchild, current)))

    def assign_targets(self, target: ast.expr, scope: _Scope) -> list[int]:
        """Keys written by one assignment target (names, ``self.x``, tuples of them)."""
        elements = target.elts if isinstance(target, ast.Tuple) else [target]
        targets: list[int] = []
        for element in elements:
            if isinstance(element, ast.Name):
                targets.append(self.resolve(scope, element.id))
            elif isinstance(element, ast.Attribute):
                targets.extend(self.attribute_store(element, scope))
        return targets

    # ── debugging / tests ──

    def qualname(self, scope: _Scope) -> str:
        parts: list[str] = []
        current: _Scope | None = scope
        while current is not None and current.kind != "module":
            parent = current.parent
            if parent is not None and parent.kind in ("function", "lambda", "comprehension"):
                parts.append(f"<locals>.{current.name}")
            else:
                parts.append(current.name)
            current = parent
        return ".".join(reversed(parts))

    def debug_name(self, key: int) -> str:
        descriptor = self.keys.describe[key]
        tag = descriptor[0]
        if tag == "call":
            call = descriptor[1]
            assert isinstance(call, ast.Call)
            return f"{self.keys.name(key)}@{call.lineno}:{call.col_offset}"
        if isinstance(tag, int) and tag >= 0:
            scope = self.scopes[tag]
            name = str(descriptor[1])
            if scope.kind == "module":
                return name
            return f"{self.qualname(scope)}.<locals>.{name}"
        if tag in (-1, -2):
            unit = descriptor[1]
            assert isinstance(unit, int)
            owner = self.logical[unit] if unit < len(self.logical) else None
            label = self.qualname(owner.infos[0].scope) if owner is not None else f"<{unit}>"
            prefix = "" if tag == -1 else "<read>"
            return f"{prefix}{label}.{descriptor[2]}"
        if tag in (-3, -4, -5, -6):
            return f"<{tag}>{descriptor[1:]}"
        return repr(descriptor)


def _build_scope_index(
    tree: ast.AST,
    aliases: dict[str, str],
    type_map: dict[str, str],
    check_runtime: Callable[[], None] | None = None,
) -> _ScopeIndex:
    """Assign every node to its lexical scope and index callable definitions.

    Visits nodes in ``ast.walk`` (breadth-first) order so statements are
    recorded in the same order the order-independent taint pass always used.
    """
    tick = check_runtime or _noop
    index = _ScopeIndex(aliases, type_map)
    index.alias_roots = {name.split(".")[0] for name in aliases}
    module = index.new_scope("module", "", None)
    switch = index.scope_switch
    # Walrus targets bind in the nearest enclosing non-comprehension scope.
    walrus_scopes: dict[ast.Name, _Scope] = {}
    # Values assigned to name / attribute targets, for receiver typing.
    stored_values: dict[ast.AST, object] = {}

    queue: deque[tuple[ast.AST, _Scope]] = deque(
        (child, module) for child in ast.iter_child_nodes(tree)
    )
    while queue:
        tick()
        node, scope = queue.popleft()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope.bound.add(node.name)
            index.name_stores.append((scope, node.name, _KNOWN_VALUE))
            if isinstance(node, ast.ClassDef):
                inner = index.new_scope("class", node.name, scope)
                inner.class_info = _ClassInfo(node, inner, scope)
                index.classes.append(inner.class_info)
            else:
                inner = index.new_scope("function", node.name, scope)
                inner.function = _FunctionInfo(node, inner)
                index.functions.append(inner.function)
                _bind_parameters(index, inner, node.args)
            # Only the body runs in the new scope; decorators, defaults,
            # annotations and bases are evaluated where the def/class stands.
            for statement in node.body:
                switch[statement] = inner
        elif isinstance(node, ast.Lambda):
            inner = index.new_scope("lambda", "<lambda>", scope)
            inner.function = _FunctionInfo(node, inner)
            index.functions.append(inner.function)
            index.lambdas[node] = inner.function
            _bind_parameters(index, inner, node.args)
            switch[node.body] = inner
            index.statements.append((node, inner))
        elif isinstance(node, _Comprehension):
            # Python 3 comprehensions have their own scope; only the first
            # iterable is evaluated in the enclosing one.
            inner = index.new_scope("comprehension", f"<{type(node).__name__.lower()}>", scope)
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
                index.statements.append((generator, inner))
        elif isinstance(node, ast.NamedExpr):
            binding = scope
            while binding.kind == "comprehension" and binding.parent is not None:
                binding = binding.parent
            walrus_scopes[node.target] = binding
            stored_values[node.target] = node.value
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                binding = walrus_scopes.pop(node, scope)
                binding.bound.add(node.id)
                if isinstance(node.ctx, ast.Store):
                    value = stored_values.pop(node, _UNKNOWN_VALUE)
                    index.name_stores.append((binding, node.id, value))
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if isinstance(node.ctx, ast.Load):
                    index.attribute_loads.append((node, scope))
                elif isinstance(node.ctx, ast.Store):
                    value = stored_values.pop(node, _UNKNOWN_VALUE)
                    index.attribute_stores.append((scope, node, value))
        elif isinstance(node, ast.Global):
            scope.declared_global.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            scope.declared_nonlocal.update(node.names)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name != "*":
                    local = alias.asname or alias.name.split(".")[0]
                    scope.bound.add(local)
                    index.name_stores.append((scope, local, _KNOWN_VALUE))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            scope.bound.add(node.name)
            index.name_stores.append((scope, node.name, _UNKNOWN_VALUE))
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            scope.bound.add(node.name)
            index.name_stores.append((scope, node.name, _UNKNOWN_VALUE))
        elif isinstance(node, ast.MatchMapping) and node.rest:
            scope.bound.add(node.rest)
            index.name_stores.append((scope, node.rest, _UNKNOWN_VALUE))
        elif isinstance(node, ast.Assign):
            index.statements.append((node, scope))
            for target in node.targets:
                if isinstance(target, (ast.Name, ast.Attribute)):
                    stored_values[target] = node.value
                    if isinstance(target, ast.Name) and isinstance(node.value, ast.Lambda):
                        index.lambda_bindings.append((scope, target.id, node.value))
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None and isinstance(node.target, (ast.Name, ast.Attribute)):
                stored_values[node.target] = node.value
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                target = item.optional_vars
                if isinstance(target, (ast.Name, ast.Attribute)):
                    stored_values[target] = item.context_expr
        elif isinstance(node, ast.Return):
            index.statements.append((node, scope))
        elif isinstance(node, ast.Call):
            index.call_scope[node] = scope
            index.statements.append((node, scope))
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.expr_context):
                queue.append((child, switch.get(child, scope)))

    _index_classes(index, tick)
    _index_callables(index, tick)
    _index_types(index, tick)
    return index


def _bind_parameters(index: _ScopeIndex, scope: _Scope, args: ast.arguments) -> None:
    """Parameters are local, and hold whatever callers pass (untyped)."""
    for arg in _parameters(args):
        scope.bound.add(arg.arg)
        index.name_stores.append((scope, arg.arg, _UNKNOWN_VALUE))


def _index_classes(index: _ScopeIndex, tick: Callable[[], None]) -> None:
    """Merge redefinitions, link in-file bases lexically, and size each hierarchy.

    Attributes are tracked per class in hierarchies of at most
    ``_PRECISE_HIERARCHY_LIMIT`` classes and per hierarchy above that.
    """
    by_binding: dict[tuple[int, str], _LogicalClass] = {}
    for info in index.classes:
        tick()
        binding = index.binding_scope(info.parent, info.node.name)
        identity = (binding.id, info.node.name)
        cls = by_binding.get(identity)
        if cls is None:
            cls = _LogicalClass(len(index.logical), info.node.name)
            index.logical.append(cls)
            by_binding[identity] = cls
        cls.infos.append(info)
        info.logical = cls
    for cls in index.logical:
        linked = {cls.id}
        for info in cls.infos:
            for base in info.node.bases:
                tick()
                if not isinstance(base, ast.Name):
                    continue
                binding = index.binding_scope(info.parent, base.id)
                parent = by_binding.get((binding.id, base.id))
                if parent is not None and parent.id not in linked:
                    linked.add(parent.id)
                    cls.bases.append(parent)
                    parent.subclasses.append(cls)

    components: list[list[_LogicalClass]] = []
    for cls in index.logical:
        tick()
        if cls.component >= 0:
            continue
        cls.component = len(components)
        members = [cls]
        stack = [cls]
        while stack:
            current = stack.pop()
            for other in (*current.bases, *current.subclasses):
                if other.component < 0:
                    other.component = cls.component
                    members.append(other)
                    stack.append(other)
        components.append(members)

    for members in components:
        precise = len(members) <= _PRECISE_HIERARCHY_LIMIT
        merged_unit = len(index.logical) + members[0].component
        for cls in members:
            tick()
            cls.precise = precise
            cls.unit = cls.id if precise else merged_unit
            if not precise:
                continue
            cls.ancestors = _closure(cls, lambda c: c.bases)
            cls.descendants = _closure(cls, lambda c: c.subclasses)
            cls.mro = _linearize(cls)
        if not precise:
            continue
        by_id = {cls.id: cls for cls in members}
        for cls in members:
            tick()
            sharing: set[int] = set()
            for descendant in cls.descendants:
                sharing |= by_id[descendant].ancestors
            cls.sharing_ids = sharing
            cls.sharing = [other for other in members if other.id in sharing]


def _closure(cls: _LogicalClass, edges: Callable[[_LogicalClass], list[_LogicalClass]]) -> set[int]:
    seen = {cls.id}
    stack = [cls]
    while stack:
        for other in edges(stack.pop()):
            if other.id not in seen:
                seen.add(other.id)
                stack.append(other)
    return seen


def _index_callables(index: _ScopeIndex, tick: Callable[[], None]) -> None:
    """Group functions by the name calls reach them through."""
    for function in index.functions:
        tick()
        node = function.node
        if isinstance(node, ast.Lambda):
            continue
        parent = function.scope.parent
        assert parent is not None
        if parent.kind == "class" and parent.class_info is not None:
            cls = parent.class_info.logical
            assert cls is not None
            kind = _method_kind(node)
            function.kind = kind
            cls.methods.add(node.name)
            group = index.group(("method", cls.unit, node.name, kind), kind, node.name)
            if not group.functions:
                index.unit_groups.setdefault((cls.unit, node.name), []).append(group)
            group.add(function)
            function.groups.append(group)
            fallback = index.group(("any", node.name, kind), kind, node.name)
            if not fallback.functions:
                index.fallback_groups.setdefault(node.name, []).append(fallback)
            fallback.add(function)
            if any(_decorator_name(d) in _PROPERTY_DECORATORS for d in node.decorator_list):
                cls.properties.add(node.name)
            positional = [*node.args.posonlyargs, *node.args.args]
            if kind != "static" and positional:
                key = index.binding_key(function.scope, positional[0].arg)
                index.self_keys[key] = (cls, kind == "class")
        else:
            key = index.resolve(parent, node.name)
            _add_function_group(index, key, node.name, function)
    for scope, name, lambda_node in index.lambda_bindings:
        tick()
        function = index.lambdas.get(lambda_node)
        if function is not None:
            _add_function_group(index, index.resolve(scope, name), name, function)
    for cls in index.logical:
        tick()
        if cls.precise:
            for name in cls.methods:
                index.definers.setdefault((cls.component, name), []).append(cls)
        # Redefinitions share one binding, so this runs once per class.
        key = index.resolve(cls.infos[0].parent, cls.name)
        index.classes_by_key.setdefault(key, []).append(cls)


def _add_function_group(index: _ScopeIndex, key: int, name: str, function: _FunctionInfo) -> None:
    group = index.group(("function", key), "function", name)
    if not group.functions:
        index.function_groups.setdefault(key, []).append(group)
    group.add(function)
    function.groups.append(group)


def _index_types(index: _ScopeIndex, tick: Callable[[], None]) -> None:
    """Record which in-file classes each variable and attribute may hold."""

    def record(key: int, scope: _Scope, value: object) -> None:
        info = index.var_types.get(key)
        if info is None:
            info = _TypeInfo()
            index.var_types[key] = info
        if value is _UNKNOWN_VALUE or value is _KNOWN_VALUE:
            found = value
        else:
            assert isinstance(value, ast.expr)
            found = index.value_type(value, scope)
        if isinstance(found, tuple):
            classes, dynamic = found
            info.classes.extend(cls for cls in classes if cls not in info.classes)
            info.dynamic = info.dynamic or dynamic
        elif found is _UNKNOWN_VALUE:
            info.unknown = True

    for scope, name, value in index.name_stores:
        tick()
        record(index.resolve(scope, name), scope, value)
    for scope, attribute, value in index.attribute_stores:
        tick()
        for key in index.attribute_store(attribute, scope):
            record(key, scope, value)


def _mark_targets(
    graph: _TaintGraph,
    targets: Sequence[int],
    kind: int,
    arg: int,
    source_owner: int,
    label: int,
    line: int,
    node: ast.AST,
) -> list[tuple[int, int]]:
    """Add one source fact to each target that lacks it.

    Add-only: an existing fact is never overwritten, so taint can only grow.
    A fact that reached a function through one of its own parameters keeps
    that mark while it stays in the function's locals and is dropped at the
    function's own ``return``. Returns the facts newly added, for the
    worklist to propagate from.
    """
    newly_tainted: list[tuple[int, int]] = []
    source_label = label & ~1
    from_parameter = label & 1
    if kind == _ASSIGN:
        line = arg
    owners = graph.keys.owner
    facts = graph.facts
    for target in targets:
        if kind == _BIND:
            out = source_label | 1
        elif kind == _RETURN:
            if from_parameter and source_owner == arg:
                continue
            out = source_label
        elif from_parameter and owners[target] == source_owner:
            out = label
        else:
            out = source_label
        entry = facts.get(target)
        if entry is None:
            facts[target] = {out: (line, node)}
        elif out in entry:
            continue
        else:
            entry[out] = (line, node)
        newly_tainted.append((target, out))
    return newly_tainted


class _TaintGraph:
    """Scope-aware taint, independent of AST visit order.

    Any single ordered pass over the tree misses flows where a sink and the
    assignment that taints it are visited in the "wrong" relative order: a
    function body defined before the module-level assignment it reads only
    runs after that assignment. Taint is therefore computed as reachability in
    a flow graph whose nodes are keys (see ``_Keys``):

    * every ``Assign`` flows from the keys its value reads to its targets;
    * every argument of a call to a function defined in this file flows to
      that callee's argument slot, and each slot flows to the parameter it
      binds (positionally, by keyword, or into ``*args``/``**kwargs``); a
      function passed as a callback (``Thread(target=f, args=(x,))``,
      ``executor.submit(f, x)``) receives the arguments that follow it;
    * every ``return`` flows into the function's return value, which each
      call to it reads; parameter defaults flow into their parameter; a
      comprehension's iterable flows into its targets.

    A flow whose value contains source calls (or ``os.environ[...]``) seeds
    its targets with each of them. Each key keeps one fact per source, so a
    later, stronger source is never hidden by an earlier one. Flows are
    recorded once each, keyed by a stable id and indexed by every key they
    read. A monotone worklist then drains new facts, firing each flow AT MOST
    ONCE per source: its targets already hold that source after the first
    firing, so a later firing could add nothing.

    Taint is add-only and bounded by keys x sources, so the loop cannot
    oscillate and terminates; the work is linear in the flows plus their
    references.

    Before scoping, keys were bare names shared by the whole file, so a
    tainted ``headers`` local in one function tainted an unrelated
    ``headers`` parameter elsewhere — reported as TT3 once 26bc7d6 (#611)
    made propagation order-independent.
    """

    def __init__(
        self,
        index: _ScopeIndex,
        type_map: dict[str, str],
        aliases: dict[str, str],
        check_runtime: Callable[[], None] | None = None,
    ) -> None:
        self.index = index
        self.keys = index.keys
        self.type_map = type_map
        self.aliases = aliases
        self.tick = check_runtime or _noop
        self.check_runtime = check_runtime
        # key -> {label: (line, source node)}
        self.facts: dict[int, dict[int, tuple[int, ast.AST]]] = {}
        # Each flow's (targets, kind, kind argument), keyed by its index here
        # (a stable id). ``propagators`` maps a key read by a flow to its ids.
        self.flows: list[tuple[tuple[int, ...], int, int]] = []
        self.propagators: dict[int, list[int]] = {}
        self.worklist: deque[tuple[int, int]] = deque()
        self._result_sets: dict[tuple[int, ...], int] = {}
        # Reads and direct sources of a call's arguments, walked once for both
        # the call's value and its argument binding.
        self._arguments: dict[ast.AST, tuple[list[int], list[tuple[str, ast.AST]]]] = {}

    # ── building ──

    def add_edge(self, reads: Iterable[int], targets: tuple[int, ...], kind: int, arg: int) -> None:
        if not targets:
            return
        flow_id = len(self.flows)
        self.flows.append((targets, kind, arg))
        propagators = self.propagators
        for key in dict.fromkeys(reads):
            propagators.setdefault(key, []).append(flow_id)

    def seed(
        self,
        targets: tuple[int, ...],
        kind: int,
        arg: int,
        sources: list[tuple[str, ast.AST]],
        line: int,
    ) -> None:
        for name, node in sources:
            label = 2 * _SOURCE_INDEX[name]
            self.worklist.extend(
                _mark_targets(self, targets, kind, arg, _NO_OWNER, label, line, node)
            )

    def flow(
        self,
        value: ast.expr,
        scope: _Scope,
        targets: tuple[int, ...],
        kind: int,
        arg: int,
        line: int,
    ) -> None:
        """Flow *value* into *targets*: seed its direct sources, edge its reads."""
        if not targets:
            return
        sources = _direct_sources(value, self.type_map, self.aliases, self.check_runtime)
        if sources:
            self.seed(targets, kind, arg, sources, line)
        self.add_edge(self.index.flow_reads(value, scope, self.tick), targets, kind, arg)

    def argument(
        self, value: ast.expr, scope: _Scope
    ) -> tuple[list[int], list[tuple[str, ast.AST]]]:
        """Reads and direct sources of one call argument, walked once.

        Stops at calls to functions defined in this file: their own value key
        already carries the sources in their arguments, so nested calls are
        walked once each rather than once per enclosing call.
        """
        cached = self._arguments.pop(value, None)
        if cached is None:
            cached = (
                list(dict.fromkeys(self.index.flow_reads(value, scope, self.tick))),
                _direct_sources(
                    value, self.type_map, self.aliases, self.check_runtime, self.index.is_value_call
                ),
            )
        return cached

    def argument_flow(
        self, value: ast.expr, scope: _Scope, targets: tuple[int, ...], line: int
    ) -> None:
        if not targets:
            return
        reads, sources = self.argument(value, scope)
        if sources:
            self.seed(targets, _ASSIGN, line, sources, line)
        self.add_edge(reads, targets, _ASSIGN, line)

    def group_returns(self, group: _Group) -> int:
        if group.returns is None:
            group.returns = self.keys.get((-4, group.id), _NO_OWNER, (group.label, "()"))
        return group.returns

    def result_key(self, groups: tuple[_Group, ...]) -> int:
        if len(groups) == 1:
            return self.group_returns(groups[0])
        identity = tuple(group.id for group in groups)
        key = self._result_sets.get(identity)
        if key is None:
            key = self.keys.get((-5, identity), _NO_OWNER, (groups[0].label, "()"))
            self._result_sets[identity] = key
            self.add_edge([self.group_returns(group) for group in groups], (key,), _PASS, 0)
        return key

    def slot(self, group: _Group, slot: object) -> int:
        key = group.slots.get(slot)
        if key is None:
            key = self.keys.get((-3, group.id, slot), _NO_OWNER, "")
            group.slots[slot] = key
        return key

    def member_slots(self, group: _Group, offset: int, slot: object) -> list[int]:
        if isinstance(slot, int):
            position = slot + offset
            return [self.slot(group, position if position < _MAX_POSITIONAL_SLOTS else "*")]
        if slot in ("*", "**"):
            return [self.slot(group, slot)]
        assert isinstance(slot, str)
        keys: list[int] = []
        if slot in group.keyword_params:
            keys.append(self.slot(group, slot))
        if group.has_kwarg:
            keys.append(self.slot(group, "kw*"))
        return keys

    def binding_targets(self, bset: _BindingSet, slot: object) -> tuple[int, ...]:
        """Slot keys an argument in call-relative *slot* flows into.

        Several callees share one intermediate key per slot, so each argument
        is one edge however many methods a receiver may dispatch to.
        """
        cached = bset.targets.get(slot)
        if cached is None:
            member_keys: list[int] = []
            for group, offset in bset.members:
                member_keys.extend(self.member_slots(group, offset, slot))
            members = tuple(dict.fromkeys(member_keys))
            if len(members) > 1:
                key = self.keys.get((-6, bset.id, slot), _NO_OWNER, "")
                self.add_edge((key,), members, _PASS, 0)
                cached = (key,)
            else:
                cached = members
            bset.targets[slot] = cached
        return cached

    def bind_arguments(
        self,
        bset: _BindingSet,
        args: Sequence[ast.expr],
        keywords: Sequence[tuple[str | None, ast.expr]],
        scope: _Scope,
        line: int,
    ) -> None:
        unpacked = False
        for position, arg in enumerate(args):
            if isinstance(arg, ast.Starred):
                unpacked, arg = True, arg.value
            slot: object = "*" if unpacked or position >= _MAX_POSITIONAL_SLOTS else position
            self.argument_flow(arg, scope, self.binding_targets(bset, slot), line)
        for name, value in keywords:
            slot = "**" if name is None else name
            self.argument_flow(value, scope, self.binding_targets(bset, slot), line)

    def call(self, call: ast.Call, scope: _Scope) -> None:
        info = self.index.call_info(call, scope)
        if info.value is not None:
            # The call's value: the callees' return values plus everything its
            # receiver and arguments read (an over-approximation earlier
            # releases also made for every call).
            func = call.func
            receiver = func.value if isinstance(func, ast.Attribute) else func
            arguments = [arg.value if isinstance(arg, ast.Starred) else arg for arg in call.args]
            arguments.extend(keyword.value for keyword in call.keywords)
            reads: list[int] = []
            for part in (receiver, *arguments):
                part_reads, sources = self.argument(part, scope)
                if part is not receiver:
                    self._arguments[part] = (part_reads, sources)
                reads.extend(part_reads)
                if sources:
                    self.seed((info.value,), _ASSIGN, call.lineno, sources, call.lineno)
            if info.results:
                reads.append(self.result_key(info.results))
            self.add_edge(reads, (info.value,), _PASS, 0)
        if info.binding is not None:
            self.bind_arguments(
                info.binding,
                call.args,
                [(keyword.arg, keyword.value) for keyword in call.keywords],
                scope,
                call.lineno,
            )
        self.callbacks(call, scope)

    def callbacks(self, call: ast.Call, scope: _Scope) -> None:
        """Bind the arguments a higher-order call forwards to its callback."""
        index = self.index
        members: list[tuple[_Group, int]] = []
        position: int | None = None
        for keyword in call.keywords:
            if keyword.arg in _CALLBACK_KEYWORDS:
                members, _ = index.resolve_callee(keyword.value, scope, fallback=True)
                if members:
                    break
        if not members:
            spawner = _CALLBACK_POSITIONS.get(_callee_name(call.func) or "")
            if spawner is not None and spawner < len(call.args):
                candidate = call.args[spawner]
                if not any(isinstance(arg, ast.Starred) for arg in call.args[: spawner + 1]):
                    members, _ = index.resolve_callee(candidate, scope, fallback=True)
                    position = spawner
        if not members:
            position = None
            for at, arg in enumerate(call.args):
                if isinstance(arg, ast.Starred):
                    break
                if isinstance(arg, (ast.Name, ast.Attribute)):
                    members, _ = index.resolve_callee(arg, scope, fallback=False)
                    if members:
                        position = at
                        break
        if not members:
            return
        forwarded = list(call.args[position + 1 :]) if position is not None else []
        forwarded_keywords: list[tuple[str | None, ast.expr]] = []
        for keyword in call.keywords:
            if keyword.arg == "args" and isinstance(keyword.value, (ast.Tuple, ast.List)):
                forwarded.extend(keyword.value.elts)
            elif keyword.arg == "kwargs" and isinstance(keyword.value, ast.Dict):
                for key, value in zip(keyword.value.keys, keyword.value.values, strict=True):
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        forwarded_keywords.append((key.value, value))
                    elif key is None:
                        forwarded_keywords.append((None, value))
        self.bind_arguments(
            index.binding_set(members), forwarded, forwarded_keywords, scope, call.lineno
        )

    def returns(self, node: ast.Return, scope: _Scope) -> None:
        function = scope.function
        if node.value is None or function is None:
            return
        targets = tuple(self.group_returns(group) for group in function.groups)
        self.flow(node.value, scope, targets, _RETURN, scope.owner, node.lineno)

    def lambda_returns(self, node: ast.Lambda, scope: _Scope) -> None:
        function = scope.function
        if function is None or not function.groups:
            return
        targets = tuple(self.group_returns(group) for group in function.groups)
        self.flow(node.body, scope, targets, _RETURN, scope.owner, node.lineno)

    def iteration(self, generator: ast.comprehension, scope: _Scope) -> None:
        targets = tuple(self.index.assign_targets(generator.target, scope))
        iter_scope = self.index.scope_switch.get(generator.iter, scope.parent or scope)
        line = generator.target.lineno
        self.flow(generator.iter, iter_scope, targets, _ASSIGN, line, line)

    def defaults(self, function: _FunctionInfo) -> None:
        args = function.node.args
        defining = function.scope.parent or self.index.module
        positional = [*args.posonlyargs, *args.args]
        defaulted = positional[len(positional) - len(args.defaults) :]
        pairs = [
            *zip(defaulted, args.defaults, strict=True),
            *((arg, d) for arg, d in zip(args.kwonlyargs, args.kw_defaults, strict=True) if d),
        ]
        for arg, default in pairs:
            assert default is not None
            target = (self.index.binding_key(function.scope, arg.arg),)
            self.flow(default, defining, target, _ASSIGN, default.lineno, default.lineno)

    def attribute_edges(self) -> None:
        """Connect attribute stores to the reads that may see them.

        A read of ``x`` through class E sees stores of ``x`` in every class
        sharing a descendant with E, plus ``@property`` getters named ``x``.
        """
        index = self.index
        for node, scope in index.attribute_loads:
            self.tick()
            index.attribute_load(node, scope)
        for (class_id, attr), key in index.attribute_reads.items():
            self.tick()
            cls = index.logical[class_id]
            reads: list[int] = []
            for other in cls.sharing:
                store = self.keys.ids.get((-1, other.unit, attr))
                if store is not None:
                    reads.append(store)
                if attr in other.properties:
                    for group in index.unit_groups.get((other.unit, attr), ()):
                        reads.append(self.group_returns(group))
            self.add_edge(reads, (key,), _PASS, 0)
        for cls in index.logical:
            if cls.precise or not cls.properties:
                continue
            self.tick()
            for attr in cls.properties:
                store = self.keys.get((-1, cls.unit, attr), _NO_OWNER, attr)
                for group in index.unit_groups.get((cls.unit, attr), ()):
                    self.add_edge((self.group_returns(group),), (store,), _PASS, 0)

    def parameter_edges(self) -> None:
        """Flow each callee's argument slots into the parameters they bind."""
        for group in self.index.groups.values():
            if not group.slots:
                continue
            slots = group.slots
            for function in group.functions:
                self.tick()
                args = function.node.args
                key = self.index.binding_key
                scope = function.scope
                positional = [*args.posonlyargs, *args.args]
                for position, arg in enumerate(positional):
                    reads = [
                        slots.get(position) if position < _MAX_POSITIONAL_SLOTS else None,
                        slots.get("*"),
                        None if position < len(args.posonlyargs) else slots.get(arg.arg),
                        slots.get("**"),
                    ]
                    self.bind(reads, key(scope, arg.arg))
                for arg in args.kwonlyargs:
                    self.bind([slots.get(arg.arg), slots.get("**")], key(scope, arg.arg))
                if args.vararg is not None:
                    extra = range(len(positional), _MAX_POSITIONAL_SLOTS)
                    reads = [slots.get("*"), *(slots.get(i) for i in extra)]
                    self.bind(reads, key(scope, args.vararg.arg))
                if args.kwarg is not None:
                    self.bind([slots.get("kw*"), slots.get("**")], key(scope, args.kwarg.arg))

    def bind(self, reads: list[int | None], target: int) -> None:
        present = [key for key in reads if key is not None]
        if present:
            self.add_edge(present, (target,), _BIND, 0)

    def build(self) -> None:
        for node, scope in self.index.statements:
            self.tick()
            if isinstance(node, ast.Assign):
                targets = tuple(
                    key
                    for target in node.targets
                    for key in self.index.assign_targets(target, scope)
                )
                self.flow(node.value, scope, targets, _ASSIGN, node.lineno, node.lineno)
            elif isinstance(node, ast.Return):
                self.returns(node, scope)
            elif isinstance(node, ast.Call):
                self.call(node, scope)
            elif isinstance(node, ast.Lambda):
                self.lambda_returns(node, scope)
            elif isinstance(node, ast.comprehension):
                self.iteration(node, scope)
        for function in self.index.functions:
            self.tick()
            self.defaults(function)
        self.attribute_edges()
        self.parameter_edges()

    # ── fixpoint ──

    def run(self) -> None:
        self.build()
        self.propagate()

    def propagate(self) -> None:
        """Drain the worklist until no flow adds a new fact."""
        facts = self.facts
        flows = self.flows
        propagators = self.propagators
        owners = self.keys.owner
        worklist = self.worklist
        fired: set[object] = set()
        while worklist:
            self.tick()
            key, label = worklist.popleft()
            line, node = facts[key][label]
            owner = owners[key]
            for flow_id in propagators.get(key, ()):
                # A fact from a function's own parameter propagates differently
                # depending on which function's key carries it.
                mark: object = (flow_id, label, owner) if label & 1 else flow_id * _LABELS + label
                if mark in fired:
                    # Already propagated this source: its targets hold it, so
                    # firing again marks nothing new. Skip to stay linear.
                    continue
                fired.add(mark)
                targets, kind, arg = flows[flow_id]
                worklist.extend(_mark_targets(self, targets, kind, arg, owner, label, line, node))

    # ── results ──

    def strongest(self, key: int, sink_name: str, excluded: set[ast.AST]) -> _TaintedVar | None:
        """The fact at *key* giving the most severe rule for *sink_name*.

        Facts whose source is one of *excluded* (sources nested directly in
        the sink, already reported as direct flows) are skipped.
        """
        best: _TaintedVar | None = None
        best_rank = -1
        for label, (line, node) in self.facts.get(key, {}).items():
            if node in excluded:
                continue
            source = _SOURCE_NAMES[label >> 1]
            rank = _SEVERITY_RANK[_RULE_SEVERITIES[_pick_rule(source, sink_name, False)]]
            if rank > best_rank:
                best = _TaintedVar(self.keys.name(key), source, line)
                best_rank = rank
        return best

    def view(self) -> dict[str, dict[str, _TaintedVar]]:
        """Tainted keys by readable name, each with one fact per source."""
        view: dict[str, dict[str, _TaintedVar]] = {}
        for key, entry in self.facts.items():
            sources = view.setdefault(self.index.debug_name(key), {})
            for label, (line, _node) in entry.items():
                source = _SOURCE_NAMES[label >> 1]
                if source not in sources:
                    sources[source] = _TaintedVar(self.keys.name(key), source, line)
        return view


def _collect_tainted(
    tree: ast.AST,
    type_map: dict[str, str],
    aliases: dict[str, str],
    check_runtime: Callable[[], None] | None = None,
) -> dict[str, dict[str, _TaintedVar]]:
    """Compute taint for *tree*: readable key -> {source call: fact}.

    Module globals keep their bare name; function locals read
    ``qualname.<locals>.name``. See ``_TaintGraph`` for the propagation rules.
    """
    index = _build_scope_index(tree, aliases, type_map, check_runtime)
    graph = _TaintGraph(index, type_map, aliases, check_runtime)
    graph.run()
    return graph.view()


def _find_tainted_names_in_args(
    node: ast.Call,
    graph: _TaintGraph,
    scope: _Scope,
    sink_name: str,
    direct_sources: set[ast.AST],
    check_runtime: Callable[[], None] | None = None,
) -> list[_TaintedVar]:
    """Find tainted variables, attributes and helper results a sink call reads.

    Plain variable reads come first, in ``ast.walk`` order, so a flow earlier
    releases reported keeps its message; attributes and in-file call results
    follow. Each key reports its most severe source.
    """
    seen: set[int] = set()
    names: list[int] = []
    others: list[int] = []
    facts = graph.facts
    for key, plain_name in graph.index.references(
        node, scope, check_runtime or _noop, skip_root=True
    ):
        if key in seen or key not in facts:
            continue
        seen.add(key)
        (names if plain_name else others).append(key)
    hits: list[_TaintedVar] = []
    for key in (*names, *others):
        tv = graph.strongest(key, sink_name, direct_sources)
        if tv is not None:
            hits.append(tv)
    return hits


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
    scopes = _build_scope_index(tree, aliases, type_map, check_runtime)
    graph = _TaintGraph(scopes, type_map, aliases, check_runtime)
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

        direct: set[ast.AST] = set()
        for src_name, src_node in _find_nested_sources(
            ast_node,
            type_map,
            aliases,
            check_runtime,
        ):
            if src_name == "open" and _is_open_for_write(src_node):
                continue
            direct.add(src_node)
            rule = _pick_rule(src_name, sink_name, is_direct=True)
            src_cat = _classify(src_name, _SOURCE_CATEGORIES, "data source")
            sink_cat = _classify(sink_name, _SINK_CATEGORIES, "data sink")
            _emit(
                node_index,
                rule,
                ast_node,
                f"Direct flow: {src_name} ({src_cat}) → {sink_name} ({sink_cat})",
            )

        # A source nested in this sink, even inside a helper call, is already
        # reported as a direct flow above; report each source once.
        for tv in _find_tainted_names_in_args(
            ast_node,
            graph,
            scopes.call_scope.get(ast_node, scopes.module),
            sink_name,
            direct,
            check_runtime,
        ):
            rule = _pick_rule(tv.source_call, sink_name, is_direct=False)
            src_cat = _classify(tv.source_call, _SOURCE_CATEGORIES, "data source")
            sink_cat = _classify(sink_name, _SINK_CATEGORIES, "data sink")
            _emit(
                node_index,
                rule,
                ast_node,
                f"Tainted flow: '{tv.name}' from {tv.source_call} (line {tv.lineno}, "
                f"{src_cat}) → {sink_name} ({sink_cat})",
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
