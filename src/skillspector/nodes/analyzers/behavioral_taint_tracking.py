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


class _TaintedVar(NamedTuple):
    """One taint fact: data from ``source_call`` reaches the variable ``name``.

    ``lineno`` is the line of the original source call (``os.getenv(...)``,
    ``input()``, ...), carried unchanged through every propagation step, so a
    finding names where the data entered the program rather than the last
    assignment that copied it.
    """

    name: str
    source_call: str
    lineno: int


class _Target(NamedTuple):
    """A taint destination: a scope-qualified ``key`` and its display ``name``."""

    key: str
    name: str


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
) -> tuple[str, int] | None:
    """Find a source call anywhere in an expression tree (handles chained calls).

    Handles patterns like ``open("f").read()``, ``requests.get(url).text``,
    and plain ``os.environ.get("K")``. Returns the source name and the line of
    the source call itself.
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
        return name, child.lineno
    return None


def _direct_source(
    node: ast.expr,
    type_map: dict[str, str],
    aliases: dict[str, str],
    check_runtime: Callable[[], None] | None = None,
) -> tuple[str, int] | None:
    """Return the source (name, line) a value reads directly, if any."""
    found = _find_source_in_expr(node, type_map, aliases, check_runtime)
    if found is not None:
        return found
    # Subscript sources like os.environ["KEY"] (also os aliased as `o`)
    if isinstance(node, ast.Subscript):
        base = resolve_dotted_name(node.value)
        if base is not None:
            base = apply_import_aliases(base, aliases)
        if base and base in _CREDENTIAL_SOURCES:
            return base, node.lineno
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


# ── Lexical scopes ──────────────────────────────────────────────────────
#
# Taint is keyed by scope-qualified variable keys rather than bare names, so a
# tainted local in one function cannot taint an unrelated same-named variable
# or parameter in another. Keys are plain strings:
#
# * module globals keep their bare name (``TOKEN``);
# * function locals and parameters are prefixed with the function's qualified
#   name (``upload.<locals>.payload``, ``Client.send.<locals>.body``);
# * a class body and ``self.<attr>`` / ``cls.<attr>`` in its methods share one
#   attribute namespace per in-module inheritance family (``Client.token``);
# * every callable group (see ``_ScopeIndex.callee_groups``) has argument slots
#   (``upload()#0``, ``upload()#kw:payload``) and a return value
#   (``upload()<return>``).
#
# Lambdas and comprehensions are folded into their enclosing scope, and match
# captures do not bind: both resolve names more widely than Python does, which
# can only add taint, never drop it.

_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef

# Call-site positional arguments at or beyond this index (and anything after a
# ``*args`` unpacking) share one wildcard slot, keeping the per-definition
# binding work bounded by a constant.
_MAX_POSITIONAL_SLOTS = 16


@dataclass(eq=False)
class _ClassInfo:
    node: ast.ClassDef
    qualname: str
    # Attribute-namespace prefix shared by the class's in-module inheritance
    # family; assigned once every class in the module is known.
    prefix: str = ""


@dataclass(eq=False)
class _FunctionInfo:
    node: _FunctionNode
    scope: _Scope
    # (callable group key, number of leading parameters bound by the receiver)
    groups: list[tuple[str, int]] = field(default_factory=list)


@dataclass(eq=False)
class _Scope:
    """One Python namespace: the module, a function body, or a class body."""

    kind: str  # "module" | "function" | "class"
    qualname: str
    parent: _Scope | None
    bound: set[str] = field(default_factory=set)
    declared_global: set[str] = field(default_factory=set)
    declared_nonlocal: set[str] = field(default_factory=set)
    class_info: _ClassInfo | None = None
    function: _FunctionInfo | None = None

    @property
    def prefix(self) -> str:
        if self.class_info is not None:
            return self.class_info.prefix
        if self.kind == "function":
            return f"{self.qualname}.<locals>."
        return ""


@dataclass(eq=False)
class _ScopeIndex:
    """Lexical scopes of one module plus what is needed to bind call sites."""

    module: _Scope
    aliases: dict[str, str]
    type_map: dict[str, str]
    # Assign / Return / Call nodes with their scope, in ``ast.walk`` order.
    statements: list[tuple[ast.Assign | ast.Return | ast.Call, _Scope]] = field(
        default_factory=list
    )
    call_scope: dict[ast.Call, _Scope] = field(default_factory=dict)
    functions: list[_FunctionInfo] = field(default_factory=list)
    classes: list[tuple[_ClassInfo, _Scope]] = field(default_factory=list)
    # Callable group keys with at least one definition in this module.
    groups: set[str] = field(default_factory=set)
    class_bindings: dict[str, list[_ClassInfo]] = field(default_factory=dict)
    classes_by_name: dict[str, list[_ClassInfo]] = field(default_factory=dict)
    # Key of each method's receiver parameter (``self``/``cls``) -> its class.
    self_keys: dict[str, _ClassInfo] = field(default_factory=dict)
    _resolved: dict[tuple[_Scope, str], str] = field(default_factory=dict)

    def resolve(self, scope: _Scope, name: str) -> str:
        """Return the key *name* refers to when read or written in *scope*.

        Follows Python's rules: a name bound anywhere in a function body is
        local to it unless declared ``global``/``nonlocal``; free names resolve
        through enclosing function scopes (closures) to the module. Class
        bodies are visible only to their own statements, not to methods.
        """
        cache_key = (scope, name)
        key = self._resolved.get(cache_key)
        if key is None:
            key = name
            current: _Scope | None = scope
            while current is not None and current.kind != "module":
                if current is scope or current.kind != "class":
                    if name in current.declared_global:
                        break
                    if name in current.bound and name not in current.declared_nonlocal:
                        key = current.prefix + name
                        break
                current = current.parent
            self._resolved[cache_key] = key
        return key

    def attribute(self, node: ast.Attribute, scope: _Scope) -> _Target | None:
        """Map ``self.x`` / ``cls.x`` / ``Class.x`` to its class attribute key."""
        if not isinstance(node.value, ast.Name):
            return None
        key = self.resolve(scope, node.value.id)
        cls = self.self_keys.get(key)
        if cls is None:
            classes = self.class_bindings.get(key)
            cls = classes[0] if classes else None
        if cls is None:
            return None
        return _Target(cls.prefix + node.attr, f"{node.value.id}.{node.attr}")

    def callee_groups(
        self, func: ast.expr, scope: _Scope, *, any_method: bool = False
    ) -> list[str]:
        """Callable groups in this module that ``func`` may refer to.

        A plain name resolves lexically to the functions bound to it (several
        when redefined), or to the ``__init__`` methods of a class it names.
        ``self.m``/``cls.m``/``Class.m``/``obj.m`` with ``obj = Class(...)``
        resolve to ``m`` in that class family. With *any_method*, any other
        ``obj.m`` not rooted at an imported name resolves to every method named
        ``m`` in the module. That fallback is used only to bind arguments,
        which can taint nothing but parameters of this module's own methods;
        reading return values through it would taint every ``x.get(...)`` in a
        file that defines a ``get`` method.
        """
        groups: list[str] = []
        if isinstance(func, ast.Name):
            key = self.resolve(scope, func.id)
            if f"{key}()" in self.groups:
                groups.append(f"{key}()")
            for cls in self.class_bindings.get(key, ()):
                group = f"{cls.prefix}__init__()"
                if group in self.groups and group not in groups:
                    groups.append(group)
            return groups
        if not isinstance(func, ast.Attribute):
            return groups
        if isinstance(func.value, ast.Name):
            key = self.resolve(scope, func.value.id)
            receivers = list(self.class_bindings.get(key, ()))
            if key in self.self_keys:
                receivers.append(self.self_keys[key])
            constructed = self.type_map.get(func.value.id)
            if constructed is not None:
                if constructed.split(".")[0] in self.aliases:
                    return groups  # an instance of an imported (library) class
                receivers.extend(self.classes_by_name.get(constructed, ()))
            for cls in receivers:
                group = f"{cls.prefix}{func.attr}()"
                if group in self.groups and group not in groups:
                    groups.append(group)
            if groups:
                return groups
        root = func.value
        while isinstance(root, ast.Attribute):
            root = root.value
        if not any_method or (isinstance(root, ast.Name) and root.id in self.aliases):
            return groups
        if f"<method>.{func.attr}()" in self.groups:
            groups.append(f"<method>.{func.attr}()")
        return groups

    def references(
        self,
        node: ast.AST,
        scope: _Scope,
        check_runtime: Callable[[], None] | None = None,
        *,
        skip_root: bool = False,
    ) -> Iterator[str]:
        """Yield every taint key an expression reads, in ``ast.walk`` order.

        Names resolve through *scope*; ``self.x`` resolves to the class
        attribute key; a call to a function defined in this module also reads
        that function's return value. Handles container literals and f-strings
        because ``ast.walk`` reaches the names nested inside them.
        """
        for child in ast.walk(node):
            if check_runtime is not None:
                check_runtime()
            if skip_root and child is node:
                continue
            if isinstance(child, ast.Name):
                yield self.resolve(scope, child.id)
            elif isinstance(child, ast.Attribute):
                target = self.attribute(child, scope)
                if target is not None:
                    yield target.key
            elif isinstance(child, ast.Call):
                for group in self.callee_groups(child.func, scope):
                    yield f"{group}<return>"

    def assign_targets(self, target: ast.expr, scope: _Scope) -> list[_Target]:
        """Keys written by one assignment target (names, ``self.x``, tuples of them)."""
        elements = target.elts if isinstance(target, ast.Tuple) else [target]
        targets: list[_Target] = []
        for element in elements:
            if isinstance(element, ast.Name):
                targets.append(_Target(self.resolve(scope, element.id), element.id))
            elif isinstance(element, ast.Attribute):
                attribute = self.attribute(element, scope)
                if attribute is not None:
                    targets.append(attribute)
        return targets


def _parameters(args: ast.arguments) -> list[ast.arg]:
    """Every parameter a function binds, in signature order."""
    params = [*args.posonlyargs, *args.args]
    if args.vararg is not None:
        params.append(args.vararg)
    params.extend(args.kwonlyargs)
    if args.kwarg is not None:
        params.append(args.kwarg)
    return params


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
    module = _Scope("module", "", None)
    index = _ScopeIndex(module, aliases, type_map)
    qualname_counts: dict[str, int] = {}

    def child_scope(kind: str, name: str, parent: _Scope) -> _Scope:
        if parent.kind == "module":
            qualname = name
        elif parent.kind == "class":
            qualname = f"{parent.qualname}.{name}"
        else:
            qualname = f"{parent.qualname}.<locals>.{name}"
        # Redefinitions get distinct namespaces; calls still reach all of them.
        count = qualname_counts.get(qualname, 0) + 1
        qualname_counts[qualname] = count
        return _Scope(kind, qualname if count == 1 else f"{qualname}#{count}", parent)

    # (node, scope it belongs to, whether a Store name binds in that scope)
    queue: deque[tuple[ast.AST, _Scope, bool]] = deque(
        (child, module, True) for child in ast.iter_child_nodes(tree)
    )
    while queue:
        if check_runtime is not None:
            check_runtime()
        node, scope, binds = queue.popleft()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope.bound.add(node.name)
            if isinstance(node, ast.ClassDef):
                inner = child_scope("class", node.name, scope)
                inner.class_info = _ClassInfo(node, inner.qualname)
                index.classes.append((inner.class_info, scope))
            else:
                inner = child_scope("function", node.name, scope)
                inner.function = _FunctionInfo(node, inner)
                index.functions.append(inner.function)
                inner.bound.update(arg.arg for arg in _parameters(node.args))
            # Only the body runs in the new scope; decorators, defaults,
            # annotations and bases are evaluated where the def/class stands.
            body = {id(statement) for statement in node.body}
            for child in ast.iter_child_nodes(node):
                queue.append((child, inner if id(child) in body else scope, True))
            continue
        if isinstance(node, ast.comprehension):
            # Comprehension targets bind in the comprehension's own scope in
            # Python 3; not binding them here keeps outer names visible.
            queue.append((node.target, scope, False))
            queue.extend((child, scope, binds) for child in (node.iter, *node.ifs))
            continue
        if isinstance(node, ast.Name):
            if binds and isinstance(node.ctx, (ast.Store, ast.Del)):
                scope.bound.add(node.id)
        elif isinstance(node, ast.Global):
            scope.declared_global.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            scope.declared_nonlocal.update(node.names)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name != "*":
                    scope.bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            scope.bound.add(node.name)
        elif isinstance(node, (ast.Assign, ast.Return)):
            index.statements.append((node, scope))
        elif isinstance(node, ast.Call):
            index.call_scope[node] = scope
            index.statements.append((node, scope))
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.expr_context):
                queue.append((child, scope, binds))

    _assign_class_families(index)

    for function in index.functions:
        node = function.node
        parent = function.scope.parent or module
        if parent.class_info is not None:
            static = any(
                isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
                for decorator in node.decorator_list
            )
            offset = 0 if static else 1
            function.groups.append((f"{parent.class_info.prefix}{node.name}()", offset))
            function.groups.append((f"<method>.{node.name}()", offset))
            positional = [*node.args.posonlyargs, *node.args.args]
            if offset and positional:
                index.self_keys[function.scope.prefix + positional[0].arg] = parent.class_info
        else:
            function.groups.append((f"{index.resolve(parent, node.name)}()", 0))
        index.groups.update(group for group, _ in function.groups)
    for cls, parent in index.classes:
        index.class_bindings.setdefault(index.resolve(parent, cls.node.name), []).append(cls)
    return index


def _assign_class_families(index: _ScopeIndex) -> None:
    """Give classes linked by in-module inheritance one attribute namespace.

    ``self.token`` set in a base class and read in a subclass (or the other way
    round) must meet. Same-named classes are merged as well, so the work stays
    linear: union-find over (class, first class with the base's name) pairs.
    """
    parent: dict[_ClassInfo, _ClassInfo] = {}
    order = {cls: position for position, (cls, _) in enumerate(index.classes)}

    def find(cls: _ClassInfo) -> _ClassInfo:
        root = cls
        while parent.get(root, root) is not root:
            root = parent[root]
        while cls is not root:
            following = parent[cls]
            parent[cls] = root
            cls = following
        return root

    def union(first: _ClassInfo, second: _ClassInfo) -> None:
        first, second = find(first), find(second)
        if first is not second:
            # The earliest-defined class names the family, for stable keys.
            if order[second] < order[first]:
                first, second = second, first
            parent[second] = first

    by_name = index.classes_by_name
    for cls, _ in index.classes:
        same_name = by_name.setdefault(cls.node.name, [])
        same_name.append(cls)
        union(same_name[0], cls)
    for cls, _ in index.classes:
        for base in cls.node.bases:
            if isinstance(base, ast.Name) and base.id in by_name:
                union(cls, by_name[base.id][0])
    for cls, _ in index.classes:
        cls.prefix = f"{find(cls).qualname}."


def _find_tainted_names_in_args(
    node: ast.Call,
    tainted: dict[str, _TaintedVar],
    scopes: _ScopeIndex,
    scope: _Scope,
    check_runtime: Callable[[], None] | None = None,
) -> list[_TaintedVar]:
    """Find tainted variables, attributes and helper results a call reads."""
    seen: set[str] = set()
    hits: list[_TaintedVar] = []
    for key in scopes.references(node, scope, check_runtime, skip_root=True):
        if key in seen:
            continue
        tv = tainted.get(key)
        if tv:
            seen.add(key)
            hits.append(tv)
    return hits


def _mark_targets(
    targets: Sequence[_Target],
    tainted: dict[str, _TaintedVar],
    src_name: str,
    lineno: int,
) -> list[str]:
    """Taint each target key that is not tainted yet.

    Add-only: an existing entry is never overwritten, so taint can only grow.
    Returns the keys newly added, for the worklist to propagate from.
    """
    newly_tainted: list[str] = []
    for key, name in targets:
        if key not in tainted:
            tainted[key] = _TaintedVar(name, src_name, lineno)
            newly_tainted.append(key)
    return newly_tainted


def _collect_tainted(
    tree: ast.AST,
    type_map: dict[str, str],
    aliases: dict[str, str],
    check_runtime: Callable[[], None] | None = None,
    scopes: _ScopeIndex | None = None,
) -> dict[str, _TaintedVar]:
    """Compute scope-aware taint, independent of AST visit order.

    Any single ordered pass over the tree misses flows where a sink and the
    assignment that taints it are visited in the "wrong" relative order: a
    function body defined before the module-level assignment it reads only
    runs after that assignment. Taint is therefore computed as reachability in
    a flow graph whose nodes are scope-qualified keys (see ``_ScopeIndex``):

    * every ``Assign`` flows from the keys its value reads to its targets;
    * every argument of a call to a function defined in this module flows to
      that callee's argument slot, and each slot flows to the parameter it
      binds (positionally, by keyword, or into ``*args``/``**kwargs``); a
      function passed as an argument (``Thread(target=f, args=(x,))``,
      ``executor.submit(f, x)``) receives the arguments that follow it;
    * every ``return`` flows into the function's return key, which any call
      to that function reads, so ``key = get_key()`` carries the helper's
      taint into the caller; parameter defaults flow into their parameter.

    A flow whose value contains a source call (or ``os.environ[...]``) seeds
    its targets directly. The remaining flows are recorded once each, keyed by
    a stable id and indexed by every key they read. A monotone worklist then
    drains newly tainted keys, firing each flow AT MOST ONCE: its targets are
    all tainted after the first firing, so a later firing could add nothing.

    Taint is add-only and bounded by the number of keys, so the loop cannot
    oscillate and terminates; each key is dequeued once and each flow fires at
    most once, so the work is linear in the flows plus their references. A
    fact keeps the original source call and its line through every hop.

    Before scoping, keys were bare names shared by the whole file, so a
    tainted ``headers`` local in one function tainted an unrelated
    ``headers`` parameter elsewhere — reported as TT3 once 26bc7d6 (#611)
    made propagation order-independent.
    """
    if scopes is None:
        scopes = _build_scope_index(tree, aliases, type_map, check_runtime)
    tainted: dict[str, _TaintedVar] = {}
    worklist: deque[str] = deque()
    # Each propagating flow's targets, stored once and keyed by its index here
    # (a stable id). ``propagators`` maps a key read by a flow to its ids.
    propagating: list[list[_Target]] = []
    propagators: dict[str, list[int]] = {}

    def add_edge(reads: Iterable[str], targets: list[_Target]) -> None:
        flow_id = len(propagating)
        propagating.append(targets)
        for key in dict.fromkeys(reads):
            propagators.setdefault(key, []).append(flow_id)

    def add_flow(value: ast.expr, scope: _Scope, targets: list[_Target]) -> None:
        if not targets:
            return
        source = _direct_source(value, type_map, aliases, check_runtime)
        if source is not None:
            # Direct source: seed taint for this flow's targets.
            worklist.extend(_mark_targets(targets, tainted, *source))
        else:
            add_edge(scopes.references(value, scope, check_runtime), targets)

    def bind_arguments(
        groups: list[str],
        args: Sequence[ast.expr],
        keywords: Sequence[tuple[str | None, ast.expr]],
        scope: _Scope,
    ) -> None:
        unpacked = False
        for position, arg in enumerate(args):
            if isinstance(arg, ast.Starred):
                unpacked, arg = True, arg.value
            slot = "#*" if unpacked or position >= _MAX_POSITIONAL_SLOTS else f"#{position}"
            add_flow(arg, scope, [_Target(group + slot, group + slot) for group in groups])
        for name, value in keywords:
            slots = ("#**",) if name is None else (f"#kw:{name}", "#kw*")
            add_flow(
                value, scope, [_Target(group + s, group + s) for group in groups for s in slots]
            )

    def bind_call(call: ast.Call, scope: _Scope) -> None:
        groups = scopes.callee_groups(call.func, scope, any_method=True)
        if groups:
            bind_arguments(groups, call.args, [(k.arg, k.value) for k in call.keywords], scope)
        # A module function passed as an argument receives the positional
        # arguments after it, plus ``args=(...)`` / ``kwargs={...}``.
        for position, arg in enumerate([*call.args, *(k.value for k in call.keywords)]):
            if not isinstance(arg, (ast.Name, ast.Attribute)):
                continue
            callback = scopes.callee_groups(arg, scope)
            if not callback:
                continue
            forwarded = list(call.args[position + 1 :])
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
            bind_arguments(callback, forwarded, forwarded_keywords, scope)
            break  # only the first function argument, keeping the work linear

    for node, scope in scopes.statements:
        if isinstance(node, ast.Assign):
            targets = [t for target in node.targets for t in scopes.assign_targets(target, scope)]
            add_flow(node.value, scope, targets)
        elif isinstance(node, ast.Return):
            function = scope.function
            if node.value is not None and function is not None:
                label = f"{function.node.name}()"
                returns = [_Target(f"{group}<return>", label) for group, _ in function.groups]
                add_flow(node.value, scope, returns)
        else:
            bind_call(node, scope)

    for function in scopes.functions:
        args = function.node.args
        defining = function.scope.parent or scopes.module
        prefix = function.scope.prefix
        param = {arg.arg: [_Target(prefix + arg.arg, arg.arg)] for arg in _parameters(args)}
        positional = [*args.posonlyargs, *args.args]
        defaulted = positional[len(positional) - len(args.defaults) :]
        for arg, default in zip(defaulted, args.defaults, strict=True):
            add_flow(default, defining, param[arg.arg])
        for arg, kw_default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
            if kw_default is not None:
                add_flow(kw_default, defining, param[arg.arg])
        for group, offset in function.groups:
            bound = positional[offset:]
            for position, arg in enumerate(bound):
                reads = [f"{group}#*"]
                if position < _MAX_POSITIONAL_SLOTS:
                    reads.append(f"{group}#{position}")
                if arg not in args.posonlyargs:
                    reads += [f"{group}#kw:{arg.arg}", f"{group}#**"]
                add_edge(reads, param[arg.arg])
            for arg in args.kwonlyargs:
                add_edge([f"{group}#kw:{arg.arg}", f"{group}#**"], param[arg.arg])
            if args.vararg is not None:
                extra = range(len(bound), _MAX_POSITIONAL_SLOTS)
                add_edge([f"{group}#*", *(f"{group}#{i}" for i in extra)], param[args.vararg.arg])
            if args.kwarg is not None:
                add_edge([f"{group}#kw*", f"{group}#**"], param[args.kwarg.arg])

    fired: set[int] = set()
    while worklist:
        if check_runtime is not None:
            check_runtime()
        key = worklist.popleft()
        origin = tainted[key]
        for flow_id in propagators.get(key, ()):
            if flow_id in fired:
                # Already propagated once: its targets are all tainted, so
                # firing again marks nothing new. Skip to stay linear.
                continue
            fired.add(flow_id)
            worklist.extend(
                _mark_targets(propagating[flow_id], tainted, origin.source_call, origin.lineno)
            )

    return tainted


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
    tainted = _collect_tainted(tree, type_map, aliases, check_runtime, scopes)
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

    # `tainted` is fully populated above, independent of traversal order, so
    # this pass only needs to check sink call sites against it.
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
                f"Direct flow: {src_name} ({src_cat}) → {sink_name} ({sink_cat})",
            )

        for tv in _find_tainted_names_in_args(
            ast_node,
            tainted,
            scopes,
            scopes.call_scope.get(ast_node, scopes.module),
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
