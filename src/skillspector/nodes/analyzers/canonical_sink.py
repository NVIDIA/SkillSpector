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

"""Single canonicalization chokepoint for dangerous-callable resolution.

Every dangerous primitive can be spelled many equivalent ways — bare name, import
alias, ``from`` / ``as`` rebinding, ``builtins.*`` qualification, dynamic import via
``importlib.import_module``, reflective ``getattr(obj, "lit")`` and subscript
reflection ``__builtins__["exec"]``. Detectors that enumerate one idiom at a time
(``import os as o`` here, ``getattr`` there, ``builtins.exec`` elsewhere) leak a new
blind spot with every missed spelling — the recurring regressions behind #114/#115,
#166 and the builtins/importlib gap.

:func:`resolve_to_canonical_sink` collapses *all* of those spellings to one canonical
sink id (the bare/dotted name the sink sets already match: ``"exec"``, ``"os.system"``,
``"subprocess.run"``). Resolving once, here, lets the sink ladders stay simple and lets
a shared evasion corpus assert the invariant for every new sink.
"""

from __future__ import annotations

import ast

from .common import (
    apply_import_aliases,
    resolve_call_name_typed,
    resolve_dotted_name,
    resolve_dynamic_import_call,
)


def _canonical_from_getattr(node: ast.Call, aliases: dict[str, str] | None) -> str | None:
    """Resolve ``getattr(obj, "attr")(...)`` to the dotted sink ``"<obj>.attr"``.

    The dangerous invocation is the *outer* call, so this inspects ``node.func``: when
    the callee is itself ``getattr(os, "system")`` the result is ``"os.system"``, and
    ``getattr(builtins, "exec")`` collapses to ``"exec"`` (the ``builtins`` prefix is
    stripped by :func:`apply_import_aliases`). Only a *constant string* attribute is
    resolved; a non-literal attribute is dynamic and left to the caller's
    reflective-access rule. Returns ``None`` when ``node.func`` is not a literal
    ``getattr`` call.
    """
    callee = node.func
    if not isinstance(callee, ast.Call):
        return None
    func_name = resolve_dotted_name(callee.func)
    if func_name is not None and aliases:
        func_name = apply_import_aliases(func_name, aliases)
    if func_name != "getattr" or len(callee.args) < 2:
        return None
    obj, attr = callee.args[0], callee.args[1]
    if not (isinstance(attr, ast.Constant) and isinstance(attr.value, str)):
        return None
    base = resolve_dotted_name(obj)
    if base is None:
        return None
    if aliases:
        base = apply_import_aliases(base, aliases)
    if base == "builtins":
        return attr.value
    return f"{base}.{attr.value}"


def _namespace_module(base: ast.expr, aliases: dict[str, str] | None) -> str | None:
    """Resolve a subscript base to the module whose namespace dict it exposes.

    Handles the three equivalent namespace handles used to reach a module's callables
    by string key:

    - ``__builtins__`` / ``builtins`` → ``"builtins"``
    - ``vars(builtins)`` → ``"builtins"`` (and ``vars(os)`` → ``"os"``)
    - ``os.__dict__`` / ``builtins.__dict__`` → ``"os"`` / ``"builtins"``

    Returns the resolved module name, or ``None`` when the base is not a recognized
    namespace handle.
    """
    base_name = resolve_dotted_name(base)
    if base_name is None and isinstance(base, ast.Call):
        # vars(<module>) namespace handle.
        inner = resolve_dotted_name(base.func)
        if inner == "vars" and base.args:
            arg = resolve_dotted_name(base.args[0])
            if aliases and arg is not None:
                arg = apply_import_aliases(arg, aliases)
            return arg
        return None
    if base_name is None:
        return None
    if base_name.endswith(".__dict__"):
        # ``<module>.__dict__`` exposes the module namespace by string key. Strip the
        # trailing ``.__dict__`` *before* alias normalization so the builtins-collapse
        # in ``apply_import_aliases`` (which turns ``builtins.__dict__`` into
        # ``__dict__``) does not erase the module name.
        module = base_name[: -len(".__dict__")]
        if aliases:
            module = apply_import_aliases(module, aliases)
        return "builtins" if module in ("__builtins__", "builtins") else module
    # Bare ``__builtins__`` / ``builtins`` are the only other namespace handles; any
    # other plain name (a user dict such as ``handlers["exec"]``) is NOT a namespace.
    if aliases:
        base_name = apply_import_aliases(base_name, aliases)
    if base_name in ("__builtins__", "builtins"):
        return "builtins"
    return None


def _canonical_from_subscript(node: ast.Call, aliases: dict[str, str] | None) -> str | None:
    """Resolve namespace-dict reflection like ``os.__dict__["system"](...)`` to its sink.

    Recognizes the dict-reflection idiom where ``node.func`` is a subscript whose base is
    a module namespace handle (``__builtins__`` / ``vars(builtins)`` / ``os.__dict__``)
    and whose index is a string literal: ``__builtins__["exec"]`` → ``"exec"``,
    ``os.__dict__["system"]`` → ``"os.system"``. Returns the canonical id, or ``None``
    for any other subscript so a plain user dict (``handlers["exec"]``) is never a sink.
    """
    func = node.func
    if not isinstance(func, ast.Subscript):
        return None
    key = func.slice
    if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
        return None
    module = _namespace_module(func.value, aliases)
    if module is None:
        return None
    if module == "builtins":
        return key.value
    return f"{module}.{key.value}"


# Sibling machinery whose only purpose is dynamic import / code execution. Each maps to
# the canonical sink id of the existing primitive it is equivalent to, so it re-enters
# the established sink ladders rather than introducing a parallel severity scheme:
#
# - ``importlib.__import__`` is literally the ``__import__`` builtin (dynamic import).
# - ``importlib.util.{find_spec,module_from_spec}`` + ``runpy.run_module/run_path``
#   load/execute a module by name → dynamic-import class (``"__import__"``).
# - ``code.interact`` executes arbitrary source → ``exec`` class.
_SIBLING_SINKS: dict[str, str] = {
    "importlib.__import__": "__import__",
    "importlib.util.find_spec": "__import__",
    "importlib.util.module_from_spec": "__import__",
    "importlib.util.spec_from_loader": "__import__",
    "runpy.run_module": "__import__",
    "runpy.run_path": "__import__",
    "code.interact": "exec",
}

# Instance-method tails matched on their canonical code-exec sink, but only when the
# *receiver* statically resolves to the relevant machinery (see :func:`canonical_sibling_method`):
# a loader/interpreter method is typically called on an instance whose class does not
# resolve, so the receiver must be tied to importlib / the ``code`` module to avoid
# flagging an unrelated user method that merely shares the name.
_SIBLING_METHOD_TAILS: dict[str, str] = {
    "exec_module": "__import__",
    "runsource": "exec",
    "runcode": "exec",
}

# Module roots whose namespace owns the gated instance methods.
_IMPORTLIB_ROOTS: tuple[str, ...] = ("importlib", "importlib.util", "importlib.machinery")
_CODE_INTERPRETER_TYPES: tuple[str, ...] = (
    "code.InteractiveInterpreter",
    "code.InteractiveConsole",
)


def canonical_sibling_sink(name: str | None) -> str | None:
    """Map a resolved sibling-machinery name to the canonical sink it is equivalent to.

    Matches the fully-qualified spelling (``importlib.util.find_spec``); the bare
    instance-method tail (``exec_module``, ``runsource``) is resolved separately, with a
    receiver check, via :func:`canonical_sibling_method`. Returns the canonical id
    (``"__import__"`` / ``"exec"``) or ``None`` when *name* is not a known sibling.
    """
    if name is None:
        return None
    return _SIBLING_SINKS.get(name)


def _receiver_is_importlib_loader(receiver: ast.expr, type_map: dict[str, str] | None) -> bool:
    """True when *receiver* statically resolves to an importlib spec loader.

    Recognizes the documented ``importlib`` spec-loader chain: ``<spec>.loader`` (the
    ``ModuleSpec.loader`` attribute, including ``importlib.util.module_from_spec(...)``
    results carried through *type_map*), and bare names whose inferred type is rooted in
    ``importlib``. A user object's ``.exec_module`` (receiver not tied to importlib) does
    not match.
    """
    if isinstance(receiver, ast.Attribute):
        # ``<x>.loader`` — the ModuleSpec.loader protocol attribute.
        if receiver.attr == "loader":
            return True
    inferred = None
    if isinstance(receiver, ast.Name) and type_map is not None:
        inferred = type_map.get(receiver.id)
    if inferred is None:
        inferred = resolve_dotted_name(receiver)
    return bool(inferred and inferred.split(".", 1)[0] == "importlib")


def _receiver_is_code_interpreter(receiver: ast.expr, type_map: dict[str, str] | None) -> bool:
    """True when *receiver* resolves to ``code.Interactive{Interpreter,Console}``.

    Matches a direct construction (``code.InteractiveInterpreter().runsource(...)``) and a
    variable whose inferred constructor type is one of those classes (carried through
    *type_map*). A user object defining ``runsource`` / ``runcode`` does not match.
    """
    inferred = None
    if isinstance(receiver, ast.Call):
        inferred = resolve_dotted_name(receiver.func)
    elif isinstance(receiver, ast.Name) and type_map is not None:
        inferred = type_map.get(receiver.id)
    return inferred in _CODE_INTERPRETER_TYPES


def canonical_sibling_method(node: ast.Call, type_map: dict[str, str] | None = None) -> str | None:
    """Map a gated instance-method call to its canonical code-exec sink id.

    Fires only when *node* is ``<recv>.exec_module(...)`` / ``.runsource(...)`` /
    ``.runcode(...)`` **and** *recv* statically resolves to the matching machinery —
    importlib spec loaders for ``exec_module`` (:func:`_receiver_is_importlib_loader`),
    ``code.Interactive{Interpreter,Console}`` for ``runsource`` / ``runcode``
    (:func:`_receiver_is_code_interpreter`). ``exec_module`` → ``"__import__"``,
    ``runsource`` / ``runcode`` → ``"exec"``. Returns ``None`` otherwise, so an unrelated
    user method sharing the name stays unflagged (documented def-use residual).
    """
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    canonical = _SIBLING_METHOD_TAILS.get(func.attr)
    if canonical is None:
        return None
    if func.attr == "exec_module":
        return canonical if _receiver_is_importlib_loader(func.value, type_map) else None
    return canonical if _receiver_is_code_interpreter(func.value, type_map) else None


def resolve_to_canonical_sink(
    node: ast.Call,
    aliases: dict[str, str] | None = None,
    type_map: dict[str, str] | None = None,
) -> str | None:
    """Reduce any spelling of a call to its canonical sink id.

    Tries, in order of precedence:

    1. Direct/alias/``from``/``as``/``builtins.*`` resolution via
       :func:`resolve_call_name_typed` (also consults *type_map* for instance methods),
       then maps dynamic-import / code-exec sibling machinery to the primitive it equals.
    2. Dynamic import chains ``importlib.import_module("os").system`` via
       :func:`resolve_dynamic_import_call`.
    3. Reflective ``getattr(obj, "lit")`` via :func:`_canonical_from_getattr`.
    4. Subscript reflection ``__builtins__["exec"]`` / ``os.__dict__["system"]`` via
       :func:`_canonical_from_subscript`.

    Returns the canonical id (``"exec"``, ``"os.system"``, …) or ``None`` when the call
    cannot be reduced statically (e.g. a non-literal ``getattr`` attribute), which the
    caller may still flag through its generic reflective-access rule.
    """
    name = resolve_call_name_typed(node, type_map, aliases)
    if name is not None:
        sibling = canonical_sibling_sink(name) or canonical_sibling_method(node, type_map)
        return sibling or name
    dynamic = resolve_dynamic_import_call(node, aliases)
    if dynamic is not None:
        return dynamic
    reflective = _canonical_from_getattr(node, aliases)
    if reflective is not None:
        return reflective
    subscript = _canonical_from_subscript(node, aliases)
    if subscript is not None:
        return subscript
    # Instance-method machinery whose receiver does not resolve to a name, e.g.
    # ``code.InteractiveInterpreter().runsource(src)`` — match the gated method (the
    # receiver must resolve to the interpreter/loader machinery) so the sibling is still
    # recognized as a code-exec sink without flagging unrelated same-named user methods.
    return canonical_sibling_method(node, type_map)
