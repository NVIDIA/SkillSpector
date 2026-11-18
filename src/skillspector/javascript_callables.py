# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded callable provenance and diagnostic witnesses in complete JavaScript syntax trees."""

from __future__ import annotations

from collections.abc import Callable
from html.parser import HTMLParser

from tree_sitter import Node

from skillspector.javascript_ast import JavaScriptProgram

_MAX_BINDING_VISITS = 400_000


_SHELL_EXECUTION_CALLS = frozenset(
    {
        "exec",
        "execfile",
        "execfilesync",
        "execsync",
        "eval",
        "function",
        "execa",
        "spawn",
        "spawnfile",
        "spawnfilesync",
        "spawnsync",
        "system",
        "run",
    }
)


_SHELL_EXECUTION_MODULES = frozenset({"child_process", "execa", "shelljs", "zx"})


def _is_shell_execution_module(module: str) -> bool:
    if "\\" in module:
        return True
    normalized = module.strip("\"'").removeprefix("node:")
    return any(
        normalized == name or normalized.startswith(f"{name}/") for name in _SHELL_EXECUTION_MODULES
    )


def _pattern_binding_names(
    program: JavaScriptProgram, pattern: Node, check_runtime: Callable[[], None]
) -> set[str]:
    """Collect the identifiers bound by a parameter or destructuring pattern.

    Only binding positions count: the left side of assignment patterns, the
    identifiers inside object and array patterns, and rest targets.
    Default-value expressions, computed keys, and property names are
    references, not bindings, so they are never descended into.
    """
    names: set[str] = set()
    pending = [pattern]
    while pending:
        check_runtime()
        node = pending.pop()
        if node.type in {"identifier", "shorthand_property_identifier_pattern"}:
            names.add(program.text(node))
        elif node.type in {"assignment_pattern", "object_assignment_pattern"}:
            left = node.child_by_field_name("left")
            if left is not None:
                pending.append(left)
        elif node.type in {"required_parameter", "optional_parameter"}:
            # Type annotations and default values do not bind runtime names.
            bound = node.child_by_field_name("pattern")
            if bound is not None:
                pending.append(bound)
        elif node.type == "pair_pattern":
            value = node.child_by_field_name("value")
            if value is not None:
                pending.append(value)
        elif node.type == "rest_pattern":
            pending.extend(node.named_children)
        elif node.type in {"formal_parameters", "object_pattern", "array_pattern"}:
            pending.extend(node.named_children)
    return names


def _callable_member(program: JavaScriptProgram, node: Node) -> str | None:
    member = node.child_by_field_name("property") or node.child_by_field_name("index")
    if member is None or member.type not in {"property_identifier", "string"}:
        return None
    return program.text(member).strip("\"'")


def _direct_js_call(ref: Node) -> Node | None:
    call = ref.parent
    if call is None or call.type not in {"call_expression", "new_expression"}:
        return None
    return (
        call
        if (call.child_by_field_name("function") or call.child_by_field_name("constructor")) == ref
        else None
    )


def _written_script_field(
    program: JavaScriptProgram, call: Node, check_runtime: Callable[[], None]
) -> Node | None:
    """Locate the dynamic body of an executable script written in a native fs call."""
    callee, arguments = call.child_by_field_name("function"), call.child_by_field_name("arguments")
    if (
        callee is None
        or arguments is None
        or _callable_member(program, callee) not in {"writeFile", "writeFileSync"}
    ):
        return None
    receiver = callee.child_by_field_name("object")
    if receiver is None or receiver.type != "identifier":
        return None
    fs = program.text(receiver)
    args = arguments.named_children
    if len(args) < 2 or args[1].type != "identifier":
        return None
    data = program.text(args[1])
    if data == fs:
        return None
    template = None
    imported = None
    receiver_refs = []
    for node in program.nodes:
        check_runtime()
        if node.type == "import_statement":
            source = node.child_by_field_name("source")
            if source is None or _is_shell_execution_module(program.text(source)):
                return None
            if program.text(source)[1:-1] in {"node:fs", "node:fs/promises", "fs", "fs/promises"}:
                for clause in node.named_children:
                    if clause.type != "import_clause":
                        continue
                    for binding in clause.named_children:
                        if binding.type == "identifier" and program.text(binding) == fs:
                            if imported is not None:
                                return None
                            imported = binding
        elif node.type in {
            "identifier",
            "shorthand_property_identifier",
            "shorthand_property_identifier_pattern",
        }:
            name = program.text(node)
            if name == fs:
                receiver_refs.append(node)
            elif name == data:
                parent = node.parent
                if (
                    parent.type == "variable_declarator"
                    and parent.child_by_field_name("name") == node
                ):
                    continue
                # Only ordinary argument/return values can consume the immutable
                # string here. A second lexical binding or a write stays unknown.
                if parent.type not in {"arguments", "return_statement"}:
                    return None
        elif node.type == "variable_declarator":
            name, value = node.child_by_field_name("name"), node.child_by_field_name("value")
            if name is not None and name.type == "identifier" and program.text(name) == data:
                if (
                    template is not None
                    or value is None
                    or value.type != "template_string"
                    or node.parent.children[0].type != "const"
                ):
                    return None
                template = value
        elif node.type in {"call_expression", "new_expression"}:
            selector = node.child_by_field_name("function") or node.child_by_field_name(
                "constructor"
            )
            if selector is not None and (
                selector.type == "import"
                or program.text(selector).casefold() in _SHELL_EXECUTION_CALLS
                or (_callable_member(program, selector) or "").casefold() in _SHELL_EXECUTION_CALLS
            ):
                return None
    if imported is None or template is None or template.end_byte - template.start_byte > 32768:
        return None
    for ref in receiver_refs:
        check_runtime()
        if ref == imported:
            continue
        member = ref.parent
        if (
            member.type != "member_expression"
            or member.child_by_field_name("object") != ref
            or (
                _callable_member(program, member)
                not in {"readFile", "readFileSync", "writeFile", "writeFileSync"}
            )
            or _direct_js_call(member) is None
        ):
            return None
    scope = template.parent.parent.parent
    if not scope.start_byte <= call.start_byte < scope.end_byte:
        return None
    prefix = "__skillspector_script_body_"
    chunks: list[str] = []
    fields: dict[str, Node] = {}
    for part in template.named_children:
        check_runtime()
        if part.type == "template_substitution":
            key = f"{prefix}{len(fields)}__"
            fields[key] = part
            chunks.append(key)
        elif part.type == "string_fragment" and prefix not in program.text(part):
            chunks.append(program.text(part))
        else:
            return None

    class Scripts(HTMLParser):
        executable = False
        field: Node | None = None

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            check_runtime()
            self.executable = tag == "script" and all(
                key == "type" and value in {"module", "text/javascript", "application/javascript"}
                for key, value in attrs
            )

        def handle_endtag(self, tag: str) -> None:
            self.executable = False

        def handle_data(self, data: str) -> None:
            if self.executable and self.field is None:
                self.field = fields.get(data.strip())

    parser = Scripts()
    try:
        parser.feed("".join(chunks))
        parser.close()
    except (ValueError, AssertionError):
        return None
    return parser.field


def _literal_module_initializers(
    program: JavaScriptProgram, callee: Node, check_runtime: Callable[[], None]
) -> bool:
    """Verify that each reference to this lexical factory supplies a source-defined module."""
    function_types = {
        "function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
    }
    current = callee.parent
    while current is not None and current.type != "variable_declarator":
        check_runtime()
        current = current.parent
    if current is None:
        return False
    binding, function = current.child_by_field_name("name"), current.child_by_field_name("value")
    if (
        binding is None
        or binding.type != "identifier"
        or function is None
        or function.type != "arrow_function"
    ):
        return False
    params = function.child_by_field_name("parameters")
    if (
        params is None
        or not params.named_children
        or params.named_children[0].type != "identifier"
        or program.text(params.named_children[0]) != program.text(callee)
    ):
        return False
    initializer = params.named_children[0]
    refs = [function]
    while refs:
        check_runtime()
        ref = refs.pop()
        if (
            ref.type
            in {
                "identifier",
                "shorthand_property_identifier",
                "shorthand_property_identifier_pattern",
            }
            and program.text(ref) == program.text(callee)
            and ref not in {initializer, callee}
        ):
            assignment = ref.parent
            value = assignment.child_by_field_name("right")
            if (
                assignment.type != "assignment_expression"
                or assignment.child_by_field_name("left") != ref
                or value is None
                or value.type != "null"
            ):
                return False
        refs.extend(ref.named_children)
    scope = current.parent.parent
    if scope.type != "statement_block" or scope.parent.type not in function_types:
        return False
    name = program.text(binding)
    shadows: set[int] = set()
    budget = _MAX_BINDING_VISITS
    for node in program.nodes:
        check_runtime()
        if node.type != "variable_declarator" or node == current:
            continue
        local = node.child_by_field_name("name")
        if local is None or name not in _pattern_binding_names(program, local, check_runtime):
            continue
        declaration = node.parent
        scopes = (
            function_types | {"program"}
            if declaration.type == "variable_declaration"
            else {"statement_block", "for_statement", "for_in_statement", "program"}
        )
        block = declaration.parent
        while block is not None and block.type not in scopes:
            check_runtime()
            budget -= 1
            if budget < 0:
                return False
            block = block.parent
        if block == scope or block == scope.parent:
            return False
        if block is not None:
            shadows.add(block.id)
    stack = [scope]
    found = False
    while stack:
        check_runtime()
        part = stack.pop()
        if part == function or part.id in shadows:
            continue
        if part.type in function_types:
            parameters = part.child_by_field_name("parameters") or part.child_by_field_name(
                "parameter"
            )
            if parameters is not None and name in _pattern_binding_names(
                program, parameters, check_runtime
            ):
                continue
        if (
            part.type
            in {
                "identifier",
                "shorthand_property_identifier",
                "shorthand_property_identifier_pattern",
            }
            and program.text(part) == name
            and part != binding
        ):
            call = _direct_js_call(part)
            args = call.child_by_field_name("arguments") if call is not None else None
            if (
                call is None
                or call.type != "call_expression"
                or args is None
                or len(args.named_children) != 1
            ):
                return False
            value = args.named_children[0]
            while value.type == "parenthesized_expression" and len(value.named_children) == 1:
                value = value.named_children[0]
            if value.type not in {"function_expression", "arrow_function"}:
                return False
            found = True
        stack.extend(part.named_children)
    return found


def _module_wrapper_markup(
    program: JavaScriptProgram, witness: Node, check_runtime: Callable[[], None]
) -> Node | None:
    """Prefer a native dynamic DOM write to a CommonJS module initialization call."""
    callee, args = witness.child_by_field_name("function"), witness.child_by_field_name("arguments")
    if callee is None or callee.type != "identifier" or args is None:
        return None
    values = args.named_children
    if (
        len(values) != 2
        or values[1].type != "identifier"
        or _callable_member(program, values[0]) != "exports"
    ):
        return None
    receiver = values[0].child_by_field_name("object")
    if (
        receiver is None
        or receiver.type != "parenthesized_expression"
        or len(receiver.named_children) != 1
    ):
        return None
    assign = receiver.named_children[0]
    target, obj = assign.child_by_field_name("left"), assign.child_by_field_name("right")
    if (
        assign.type != "assignment_expression"
        or target is None
        or program.text(target) != program.text(values[1])
        or obj is None
        or obj.type != "object"
    ):
        return None
    pairs = obj.named_children
    if (
        len(pairs) != 1
        or (key := pairs[0].child_by_field_name("key")) is None
        or program.text(key) != "exports"
        or (exports := pairs[0].child_by_field_name("value")) is None
        or exports.type != "object"
        or exports.named_children
    ):
        return None
    if not _literal_module_initializers(program, callee, check_runtime):
        return None
    budget = _MAX_BINDING_VISITS
    for node in program.nodes:
        check_runtime()
        if node.type != "assignment_expression":
            continue
        left, right = node.child_by_field_name("left"), node.child_by_field_name("right")
        if (
            left is None
            or right is None
            or right.type != "identifier"
            or _callable_member(program, left) != "innerHTML"
        ):
            continue
        element = left.child_by_field_name("object")
        if element is None or element.type != "identifier":
            continue
        function = node.parent
        while function is not None and function.type not in {
            "function_expression",
            "function_declaration",
            "arrow_function",
        }:
            check_runtime()
            function = function.parent
        if function is None or (params := function.child_by_field_name("parameters")) is None:
            continue
        names = {program.text(arg) for arg in params.named_children if arg.type == "identifier"}
        if program.text(element) in names and program.text(right) in names:
            budget -= function.descendant_count
            if budget < 0:
                return None
            body = [function.child_by_field_name("body")]
            while body:
                check_runtime()
                part = body.pop()
                if part is None or part.type in {
                    "function_declaration",
                    "function_expression",
                    "arrow_function",
                }:
                    continue
                obj = (
                    part.child_by_field_name("object") if part.type == "member_expression" else None
                )
                if (
                    obj is not None
                    and obj.type == "identifier"
                    and program.text(obj) == program.text(element)
                    and _callable_member(program, part)
                    in {"namespaceURI", "appendChild", "removeChild"}
                ):
                    return node
                body.extend(part.named_children)
    return None


def _private_callable_uses(
    program: JavaScriptProgram, check_runtime: Callable[[], None]
) -> tuple[set[int], set[str]]:
    """Resolve direct calls to fixed namespace members and private function arguments.

    A private helper must have an exhaustive set of direct callers. Its formal
    stays unknown if the helper escapes, the argument is spread or reassigned,
    or any caller can supply an arbitrary function. Only the proved call-site
    identifiers are owned; every callback body remains in the normal traversal.
    """
    if (
        "function" not in program.content
        and "=>" not in program.content
        and "import" not in program.content
    ):
        return set(), set()
    refs: dict[str, list[Node]] = {}
    functions: list[tuple[Node, Node]] = []
    namespaces: dict[str, Node] = {}
    aliases: list[tuple[Node, Node]] = []
    module = False
    for node in program.nodes:
        check_runtime()
        if node.type in {"import_statement", "export_statement"}:
            module = True
        if node.type == "with_statement":
            return set(), set()
        if node.type in {
            "identifier",
            "shorthand_property_identifier",
            "shorthand_property_identifier_pattern",
        }:
            name = program.text(node)
            if "\\" in name or name == "eval":
                return set(), set()
            refs.setdefault(name, []).append(node)
        if node.type == "namespace_import":
            source = node.parent.parent.child_by_field_name("source")
            if (
                source is not None
                and source.type == "string"
                and program.text(source)[1:-1].startswith(("./", "../"))
            ):
                for child in node.named_children:
                    if child.type == "identifier":
                        namespaces[program.text(child)] = child
        if node.type == "function_declaration":
            name = node.child_by_field_name("name")
            if name is not None and name.type == "identifier":
                functions.append((name, node))
        if node.type == "variable_declarator":
            name = node.child_by_field_name("name")
            value = node.child_by_field_name("value")
            if (
                name is not None
                and name.type == "identifier"
                and value is not None
                and node.parent.children[0].type == "const"
            ):
                if value.type == "arrow_function":
                    functions.append((name, value))
                elif value.type == "member_expression":
                    receiver = value.child_by_field_name("object")
                    if receiver is not None and receiver.type == "identifier":
                        aliases.append((name, receiver))
    owned: set[int] = set()
    imported: set[str] = set()
    budget = _MAX_BINDING_VISITS
    for name, receiver in aliases:
        spelling = program.text(receiver)
        if spelling not in namespaces:
            continue
        targets = refs[spelling]
        uses = refs[program.text(name)]
        budget -= len(targets) + len(uses)
        if budget < 0:
            return owned, imported
        block = name.parent.parent.parent
        if any(
            ref != name and not block.start_byte <= ref.start_byte < block.end_byte for ref in uses
        ):
            continue
        if any(
            ref != namespaces[spelling]
            and (
                ref.parent.type != "member_expression"
                or ref.parent.child_by_field_name("object") != ref
            )
            for ref in targets
        ) or any(ref != name and _direct_js_call(ref) is None for ref in uses):
            continue
        member = _callable_member(program, receiver.parent)
        if member is None or member.casefold() in _SHELL_EXECUTION_CALLS:
            continue
        imported.add(program.text(name))
        owned.update(ref.id for ref in uses if ref != name)
    for name, fn in functions:
        check_runtime()
        declaration = fn if fn.type == "function_declaration" else fn.parent.parent
        if declaration.parent.type != "statement_block" and not (
            module and declaration.parent.type == "program"
        ):
            continue
        formals = fn.child_by_field_name("parameters")
        if formals is None or any(arg.type != "identifier" for arg in formals.named_children):
            continue
        bindings = formals.named_children
        if len({program.text(binding) for binding in bindings}) != len(bindings):
            continue
        uses = refs[program.text(name)]
        cases: list[list[Node]] = []
        budget -= len(uses)
        if budget < 0:
            break
        for ref in uses:
            if ref == name:
                continue
            if not declaration.parent.start_byte <= ref.start_byte < declaration.parent.end_byte:
                break
            call = _direct_js_call(ref)
            args = call.child_by_field_name("arguments") if call is not None else None
            if args is None or call.type != "call_expression" or args.type != "arguments":
                break
            actual = [arg for arg in args.named_children if arg.type != "comment"]
            if any(arg.type == "spread_element" for arg in actual):
                break
            cases.append(actual)
        else:
            if not cases or any(
                fn.start_byte < ref.start_byte < fn.end_byte for ref in refs.get("arguments", ())
            ):
                continue
            for index, binding in enumerate(bindings):
                related = refs[program.text(binding)]
                budget -= len(related) + len(cases)
                if budget < 0:
                    return owned, imported
                local = [
                    ref
                    for ref in related
                    if fn.start_byte <= ref.start_byte < fn.end_byte and ref != binding
                ]
                calls = [ref for ref in local if _direct_js_call(ref) is not None]
                if not calls:
                    continue
                # A matching name in a nested binding or a write/escape prevents ownership.
                others = [ref for ref in local if _direct_js_call(ref) is None]
                target = [args[index] if index < len(args) else None for args in cases]
                if all(arg is None for arg in target):
                    guards = {
                        ref.parent.parent.id
                        for ref in others
                        if ref.parent.type == "parenthesized_expression"
                        and ref.parent.parent.type == "if_statement"
                        and ref.parent.parent.child_by_field_name("condition") == ref.parent
                        and ref.parent.named_children == [ref]
                    }
                    if len(guards) != len(others) or any(
                        (statement := ref.parent.parent).type != "expression_statement"
                        or statement.parent.id not in guards
                        or statement.parent.child_by_field_name("consequence") != statement
                        for ref in calls
                    ):
                        continue
                elif others or any(arg is None for arg in target):
                    continue
                elif all(arg.type == "arrow_function" for arg in target):
                    if any(ref.parent.type != "call_expression" for ref in calls):
                        continue
                    if any(
                        (formal := arg.child_by_field_name("parameters")) is not None
                        and any(child.type != "identifier" for child in formal.named_children)
                        for arg in target
                    ):
                        continue
                    pending = [arg.child_by_field_name("body") for arg in target]
                    while pending:
                        check_runtime()
                        part = pending.pop()
                        budget -= 1
                        if budget < 0:
                            return owned, imported
                        if part is None or part.type not in {
                            "identifier",
                            "number",
                            "subscript_expression",
                            "binary_expression",
                            "parenthesized_expression",
                            "array",
                        }:
                            break
                        pending.extend(part.named_children)
                    else:
                        owned.update(ref.id for ref in calls)
                    continue
                elif all(
                    arg.type == "identifier"
                    and program.text(arg)
                    in {
                        "Int8Array",
                        "Uint8Array",
                        "Uint8ClampedArray",
                        "Int16Array",
                        "Uint16Array",
                        "Int32Array",
                        "Uint32Array",
                        "Float32Array",
                        "Float64Array",
                    }
                    for arg in target
                ):
                    if any(ref.parent.type != "new_expression" for ref in calls):
                        continue
                    supplied = {arg.id for arg in target}
                    native_names = {program.text(arg) for arg in target}
                    natives = [ref for native in native_names for ref in refs[native]]
                    budget -= len(natives)
                    if budget < 0:
                        return owned, imported
                    if any(
                        ref.id not in supplied
                        and not (
                            ref.parent.type == "new_expression"
                            and ref.parent.child_by_field_name("constructor") == ref
                        )
                        and not (
                            ref.parent.type == "binary_expression"
                            and ref.parent.child_by_field_name("right") == ref
                            and any(child.type == "instanceof" for child in ref.parent.children)
                        )
                        and not (
                            ref.parent.type == "member_expression"
                            and ref.parent.child_by_field_name("object") == ref
                            and _callable_member(program, ref.parent) in {"from", "of"}
                            and _direct_js_call(ref.parent) is not None
                        )
                        for ref in natives
                    ):
                        continue
                    # Without a trusted global binding, retain the constructor barrier.
                    if any(
                        key in refs
                        for key in ("globalThis", "window", "self", "global", "Reflect", "Function")
                    ):
                        continue
                else:
                    continue
                owned.update(ref.id for ref in calls)
    return owned, imported


def _external_option_callback(
    program: JavaScriptProgram, check_runtime: Callable[[], None]
) -> Node | None:
    """Locate a callback read directly from the options of an exported function."""
    exported: set[str] = set()
    for statement in program.root_node.named_children:
        check_runtime()
        if statement.type != "export_statement":
            continue
        declaration = statement.child_by_field_name("declaration")
        if declaration is None or declaration.type != "lexical_declaration":
            continue
        for binding in declaration.named_children:
            value = binding.child_by_field_name("value")
            if value is not None and value.type == "object":
                for pair in value.named_children:
                    check_runtime()
                    target = pair.child_by_field_name("value")
                    if pair.type == "pair" and target is not None and target.type == "identifier":
                        exported.add(program.text(target))
    candidates: dict[str, tuple[Node, Node, Node | None]] = {}
    calls: dict[str, list[Node]] = {}

    def public(fn: Node) -> bool:
        name = fn.child_by_field_name("name")
        return fn.type == "function_declaration" and (
            fn.parent.type == "export_statement"
            or fn.parent.type == "program"
            and name is not None
            and program.text(name) in exported
        )

    for node in program.nodes:
        check_runtime()
        if node.type == "object_pattern":
            owner = node.parent
            value = None
            if owner.type == "variable_declarator":
                value = owner.child_by_field_name("value")
                function = owner.parent.parent.parent
                if (
                    function is None
                    or not public(function)
                    or value is None
                    or value.type != "identifier"
                ):
                    continue
                params = function.child_by_field_name("parameters")
                if params is None or program.text(value) not in _pattern_binding_names(
                    program, params, check_runtime
                ):
                    continue
            else:
                params = owner.parent if owner.type == "assignment_pattern" else owner
                if params.type != "formal_parameters" or not public(params.parent):
                    continue
                function = params.parent
            for binding in node.named_children:
                field = (
                    binding.child_by_field_name("left")
                    if binding.type == "object_assignment_pattern"
                    else binding
                )
                if field is not None and field.type == "shorthand_property_identifier_pattern":
                    candidates[program.text(field)] = (field, function, value)
        elif node.type in {
            "identifier",
            "shorthand_property_identifier_pattern",
            "shorthand_property_identifier",
        }:
            calls.setdefault(program.text(node), []).append(node)
    budget = _MAX_BINDING_VISITS
    for name, (field, function, options) in candidates.items():
        if options is not None:
            inputs = calls.get(program.text(options), ())
            budget -= len(inputs)
            if budget < 0:
                return None
            params = function.child_by_field_name("parameters")
            # Any additional reference can rebind, mutate, or expose the options
            # before destructuring. Direct parameter patterns require no alias.
            if any(
                ref != options and params.end_byte <= ref.start_byte < function.end_byte
                for ref in inputs
            ):
                continue
        actual = None
        for ref in calls.get(name, ()):
            check_runtime()
            budget -= 1
            if budget < 0:
                return None
            if ref == field or not function.start_byte <= ref.start_byte < function.end_byte:
                continue
            direct = _direct_js_call(ref)
            if direct is not None:
                actual = direct
            elif ref.parent.type not in {"binary_expression", "parenthesized_expression"}:
                break
        else:
            if actual is not None:
                return actual
    return None


def _forwarded_stage_call(
    program: JavaScriptProgram, check_runtime: Callable[[], None]
) -> Node | None:
    """Find an awaited call to a stage supplied to an exposed pipeline function."""
    if "await" not in program.content or " of " not in program.content:
        return None
    exposed = {
        program.text(node)
        for node in program.nodes
        if node.type == "shorthand_property_identifier"
        and node.parent.type == "object"
        and node.parent.parent.type == "arguments"
    }
    if not exposed:
        return None
    process_calls = set(_SHELL_EXECUTION_CALLS)
    for statement in program.root_node.named_children:
        check_runtime()
        source = statement.child_by_field_name("source")
        if (
            statement.type == "import_statement"
            and source is not None
            and _is_shell_execution_module(program.text(source))
        ):
            pending = list(statement.named_children)
            while pending:
                check_runtime()
                part = pending.pop()
                if part.type == "import_specifier":
                    local = part.child_by_field_name("alias") or part.child_by_field_name("name")
                    if local is not None:
                        process_calls.add(program.text(local).casefold())
                pending.extend(part.named_children)
    stage_uses: dict[str, list[Node]] = {}
    for item in program.nodes:
        check_runtime()
        if item.type == "call_expression":
            callee, args = (
                item.child_by_field_name("function"),
                item.child_by_field_name("arguments"),
            )
            if (
                callee is not None
                and callee.type == "identifier"
                and program.text(callee).casefold() in process_calls
                and args is not None
                and (
                    args.type != "arguments"
                    or not args.named_children
                    or args.named_children[0].type != "string"
                )
            ):
                return None
        if item.type in {
            "identifier",
            "shorthand_property_identifier",
            "shorthand_property_identifier_pattern",
        }:
            stage_uses.setdefault(program.text(item), []).append(item)
    budget = _MAX_BINDING_VISITS
    for loop in program.nodes:
        check_runtime()
        if loop.type != "for_in_statement" or not any(
            child.type == "of" for child in loop.children
        ):
            continue
        stage, stages = loop.child_by_field_name("left"), loop.child_by_field_name("right")
        if (
            stage is None
            or stage.type != "identifier"
            or stages is None
            or stages.type != "identifier"
        ):
            continue
        if not any(child.type == "const" for child in loop.children):
            continue
        owner = loop.parent
        while owner is not None:
            check_runtime()
            if owner.type in {
                "function_declaration",
                "arrow_function",
                "function_expression",
                "method_definition",
            }:
                params = owner.child_by_field_name("parameters") or owner.child_by_field_name(
                    "parameter"
                )
                if params is not None and program.text(stages) in _pattern_binding_names(
                    program, params, check_runtime
                ):
                    break
            owner = owner.parent
        if owner is None or owner.type != "function_declaration":
            continue
        name = owner.child_by_field_name("name")
        if name is None or program.text(name) not in exposed:
            continue
        uses = stage_uses.get(program.text(stages), ())
        budget -= len(uses)
        if budget < 0:
            break
        if any(
            ref != stages and params.end_byte <= ref.start_byte < owner.end_byte for ref in uses
        ):
            continue
        references = stage_uses.get(program.text(stage), ())
        budget -= len(references)
        if budget < 0:
            return None
        selected = None
        for ref in references:
            check_runtime()
            if ref == stage or not loop.start_byte <= ref.start_byte < loop.end_byte:
                continue
            direct = _direct_js_call(ref)
            if direct is None:
                # A nested declaration, escape or reassignment cannot establish
                # that an invocation selected the original iteration binding.
                break
            if direct.type == "call_expression" and direct.parent.type == "await_expression":
                selected = direct
        else:
            if selected is not None:
                return selected
    return None
