# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Node argv and RegExp ownership for bounded JavaScript inspection."""

from __future__ import annotations

import re
from collections.abc import Callable

from tree_sitter import Node

from skillspector.javascript_ast import JavaScriptProgram
from skillspector.javascript_callables import _callable_member

_PATH_FUNCTIONS = frozenset(
    {
        "basename",
        "dirname",
        "extname",
        "format",
        "isAbsolute",
        "join",
        "normalize",
        "parse",
        "relative",
        "resolve",
        "toNamespacedPath",
    }
)


# Only a filesystem path argument has native filename semantics. In particular,
# generated file contents can carry executable text and still need inspection.
_FS_PATH_METHODS = frozenset(
    {
        "access",
        "accessSync",
        "appendFile",
        "appendFileSync",
        "existsSync",
        "lstat",
        "lstatSync",
        "mkdir",
        "mkdirSync",
        "open",
        "openSync",
        "readFile",
        "readFileSync",
        "readdir",
        "readdirSync",
        "realpath",
        "realpathSync",
        "rm",
        "rmSync",
        "stat",
        "statSync",
        "truncate",
        "truncateSync",
        "unlink",
        "unlinkSync",
        "writeFile",
        "writeFileSync",
    }
)
_FS_FD_METHODS = frozenset(
    {"close", "closeSync", "fsync", "fsyncSync", "read", "readSync", "write", "writeSync"}
)


def _regex_receiver_call(program: JavaScriptProgram, value: Node) -> Node | None:
    """Return the .exec/.test member for a direct regex receiver, else None.

    The call must take no template-string argument: a template argument
    keeps the conservative reading, same as for a literal receiver.
    """
    member = value.parent
    call = None if member is None else member.parent
    arguments = None if call is None else call.child_by_field_name("arguments")
    if (
        member is not None
        and member.type == "member_expression"
        and member.child_by_field_name("object") == value
        and _callable_member(program, member) in {"exec", "test"}
        and call is not None
        and call.type == "call_expression"
        and call.child_by_field_name("function") == member
        and arguments is not None
        and arguments.type != "template_string"
    ):
        return member
    return None


def _local_regex_arguments(
    program: JavaScriptProgram,
    references: dict[str, list[Node]],
    literals: list[Node],
    constructors: list[Node],
    exec_members: set[int],
    check_runtime: Callable[[], None],
) -> bool:
    """Bound the uses of other RegExp literals before trusting a shared prototype."""
    if not literals and not constructors:
        return True
    if len(literals) + len(constructors) > 256:
        return False
    budget = 8192
    factories: dict[str, bool] = {}
    matches: dict[str, bool] = {}
    local_constructors: set[int] = set()

    def uses(name: str) -> list[Node]:
        nonlocal budget
        check_runtime()
        seen = references.get(name, [])
        budget -= len(seen) + 1
        return seen if budget >= 0 else []

    def field(node: Node | None, name: str) -> Node | None:
        return node.child_by_field_name(name) if node is not None else None

    def const_name(value: Node) -> Node | None:
        declarator = value.parent
        declaration = declarator.parent if declarator else None
        if (
            declarator is not None
            and field(declarator, "value") == value
            and declaration is not None
            and declaration.type == "lexical_declaration"
            and declaration.children[0].type == "const"
            and declaration.parent is not None
            and declaration.parent.type in {"statement_block", "program"}
        ):
            return field(declarator, "name")
        return None

    def direct_method(member: Node | None, names: set[str]) -> bool:
        return (
            member is not None
            and member.type == "member_expression"
            and _callable_member(program, member) in names
            and member.parent is not None
            and member.parent.type == "call_expression"
            and field(member.parent, "function") == member
            and (args := field(member.parent, "arguments")) is not None
            and args.type == "arguments"
        )

    def lexical_scope(binding: Node) -> Node | None:
        declarator = binding.parent
        declaration = None if declarator is None else declarator.parent
        if declaration is None or declaration.type != "lexical_declaration":
            return None
        scope = declaration.parent
        return scope if scope is not None and scope.type in {"statement_block", "program"} else None

    def in_scope(ref: Node, scope: Node | None) -> bool:
        return scope is not None and scope.start_byte <= ref.start_byte < scope.end_byte

    def match_values(name: str) -> bool:
        # A RegExp match's numbered captures are strings (or absent). Check
        # every reference so a reassigned or escaped match cannot lend a type.
        refs = uses(name)
        declarations = [
            ref
            for ref in refs
            if ref.parent is not None
            and ref.parent.type == "variable_declarator"
            and field(ref.parent, "name") == ref
        ]
        if len(declarations) != 1:
            return False
        binding = declarations[0]
        scope = lexical_scope(binding)
        assigned = False
        for ref in refs:
            owner = ref.parent
            if owner is None or not in_scope(ref, scope):
                return False
            if ref == binding:
                if field(owner, "value") is not None:
                    return False
            elif owner.type == "assignment_expression" and field(owner, "left") == ref:
                rhs = field(owner, "right")
                callee = field(rhs, "function")
                if (
                    rhs is None
                    or rhs.type != "call_expression"
                    or callee is None
                    or callee.id not in exec_members
                ):
                    return False
                assigned = True
            elif owner.type == "subscript_expression" and field(owner, "object") == ref:
                index = field(owner, "index")
                use = owner.parent
                if index is None or not re.fullmatch(r"0|[1-9][0-9]{0,2}", program.text(index)):
                    return False
                if (
                    use is not None
                    and direct_method(use, {"trim"})
                    and field(use, "object") == owner
                ):
                    continue
                if use is None or use.type != "arguments" or use.parent is None:
                    return False
                method = field(use.parent, "function")
                regex = field(method, "object")
                if not direct_method(method, {"test"}) or regex is None or regex.type != "regex":
                    return False
            else:
                return False
        return assigned

    def string_receiver(receiver: Node | None) -> bool:
        if receiver is None:
            return False
        if receiver.type in {"string", "template_string"}:
            return True
        if receiver.type == "call_expression":
            trim = field(receiver, "function")
            args = field(receiver, "arguments")
            if not direct_method(trim, {"trim"}) or args is None or args.named_child_count:
                return False
            receiver = field(trim, "object")
        if receiver is None or receiver.type != "subscript_expression":
            return False
        obj = field(receiver, "object")
        if obj is None or obj.type != "identifier":
            return False
        name = program.text(obj)
        if name not in matches:
            matches[name] = match_values(name)
        return matches[name]

    string_names: dict[str, bool] = {}

    def literal_string_sum(value: Node) -> bool:
        if value.type in {"regex", "string"}:
            return True
        left, operator, right = (
            field(value, "left"),
            field(value, "operator"),
            field(value, "right"),
        )
        if (
            value.type != "binary_expression"
            or left is None
            or right is None
            or right.type != "string"
            or operator is None
            or operator.type != "+"
        ):
            return False
        if left.type == "string":
            return True
        if left.type != "identifier":
            return False
        name = program.text(left)
        if name not in string_names:
            refs = uses(name)
            owners = [
                ref
                for ref in refs
                if ref.parent is not None
                and (v := field(ref.parent, "value")) is not None
                and v.type == "string"
                and const_name(v) == ref
            ]
            # All references must name this constant, without reassignment or
            # another declaration that could substitute a value with hooks.
            string_names[name] = len(owners) == 1 and all(
                in_scope(ref, lexical_scope(owners[0]))
                and (
                    ref == owners[0]
                    or ref.parent is not None
                    and ref.parent.type == "binary_expression"
                    and field(ref.parent, "left") == ref
                )
                for ref in refs
            )
        return string_names[name]

    def read_array(array: Node, factory_name: str, field_name: str, metadata: set[str]) -> bool:
        binding = const_name(array)
        if binding is None or binding.type != "identifier" or array.child_count > 1026:
            return False
        for element in array.named_children:
            check_runtime()
            callee, args = field(element, "function"), field(element, "arguments")
            if (
                element.type != "call_expression"
                or callee is None
                or callee.type != "identifier"
                or program.text(callee) != factory_name
                or args is None
                or args.named_child_count != 2
            ):
                return False
            value, description = args.named_children
            if not literal_string_sum(value) or description.type != "string":
                return False
        scope = lexical_scope(binding)
        for ref in uses(program.text(binding)):
            if not in_scope(ref, scope):
                return False
            if ref == binding:
                continue
            loop = ref.parent
            row = field(loop, "left")
            body = field(loop, "body")
            if (
                loop is None
                or loop.type != "for_in_statement"
                or field(loop, "right") != ref
                or row is None
                or row.type != "identifier"
                or body is None
                or not {"const", "of"} <= {child.type for child in loop.children}
            ):
                return False
            for row_ref in uses(program.text(row)):
                if not body.start_byte <= row_ref.start_byte < body.end_byte:
                    continue
                member = row_ref.parent
                if (
                    member is None
                    or member.type != "member_expression"
                    or field(member, "object") != row_ref
                ):
                    return False
                key = _callable_member(program, member)
                if key in metadata:
                    continue
                method = member.parent
                if (
                    key != field_name
                    or not direct_method(method, {"test", "exec"})
                    or field(method, "object") != member
                ):
                    return False
        return budget >= 0

    def factory(name: str) -> bool:
        # A local const may pack a regex and independent data into a record.
        # Only an immutable array iterated for those exact fields owns it.
        definitions = []
        arrays: dict[int, Node] = {}
        refs = uses(name)
        for ref in refs:
            parent = ref.parent
            if parent is None:
                return False
            fn = field(parent, "value")
            if fn is not None and fn.type == "arrow_function" and const_name(fn) == ref:
                definitions.append(fn)
            elif parent.type == "call_expression" and field(parent, "function") == ref:
                array = parent.parent
                if array is None or array.type != "array":
                    return False
                arrays[array.id] = array
            else:
                return False
        if len(definitions) != 1 or not arrays:
            return False
        fn = definitions[0]
        binding = const_name(fn)
        if binding is None or not all(in_scope(ref, lexical_scope(binding)) for ref in refs):
            return False
        params = field(fn, "parameters")
        body = field(fn, "body")
        if (
            params is None
            or params.named_child_count != 2
            or body is None
            or body.type != "parenthesized_expression"
            or body.named_child_count != 1
        ):
            return False
        if any(p.type != "identifier" for p in params.named_children):
            return False
        if any(p.end_byte - p.start_byte > 128 for p in params.named_children):
            return False
        pattern, meta = (program.text(p) for p in params.named_children)
        obj = body.named_children[0]
        if pattern == meta or obj.type != "object" or obj.named_child_count != 2:
            return False
        pattern_key = None
        metadata = set()
        for prop in obj.named_children:
            if prop.type == "shorthand_property_identifier" and program.text(prop) == meta:
                metadata.add(meta)
                continue
            key, value = field(prop, "key"), field(prop, "value")
            if (
                prop.type != "pair"
                or key is None
                or key.type != "property_identifier"
                or value is None
            ):
                return False
            prop_name = program.text(key)
            if value.type == "identifier" and program.text(value) == meta:
                metadata.add(prop_name)
                continue
            # The conditional either retains the supplied RegExp or constructs
            # one from the same argument. Other code cannot run in the factory.
            if value.end_byte - value.start_byte > 256:
                return False
            arg = re.escape(pattern)
            if not re.fullmatch(
                rf"{arg}(?:\s+instanceof\s+RegExp\s*\?\s*{arg}\s*:\s*new\s+RegExp\s*\(\s*{arg}\s*\))?",
                program.text(value),
            ):
                return False
            alternative = field(value, "alternative")
            if alternative is not None and alternative.type == "new_expression":
                local_constructors.add(alternative.id)
            pattern_key = prop_name
        if (
            pattern_key is None
            or len(metadata) != 1
            or pattern_key in metadata
            or "__proto__" in metadata | {pattern_key}
        ):
            return False
        return all(read_array(array, name, pattern_key, metadata) for array in arrays.values())

    for literal in literals:
        check_runtime()
        args = literal.parent
        call = None if args is None else args.parent
        callee = field(call, "function")
        if (
            args is None
            or call is None
            or call.type != "call_expression"
            or callee is None
            or args.named_child(0) != literal
        ):
            return False
        if callee.type == "identifier":
            name = program.text(callee)
            if name not in factories:
                factories[name] = factory(name)
            if not factories[name]:
                return False
        elif (
            args.named_child_count != 1
            or not direct_method(callee, {"split"})
            or not string_receiver(field(callee, "object"))
        ):
            return False
    return budget >= 0 and all(node.id in local_constructors for node in constructors)


def _ordinary_native_calls(
    program: JavaScriptProgram, check_runtime: Callable[[], None]
) -> tuple[set[int], set[int], set[int]]:
    """Prove direct native calls without granting ownership to escaped methods.

    As with the existing browser/assertion proofs, ordinary built-ins are
    assumed. Explicit overrides, reflection, shadowing and escaping references
    cannot borrow that assumption. Raw finding inspection remains unchanged.
    """
    references: dict[str, list[Node]] = {}
    bindings: dict[str, tuple[Node, str, str, bool]] = {}
    regex_calls: list[Node] = []
    regex_bindings: dict[str, Node] = {}
    intrinsic_override = False
    module_unproven = False
    imports: dict[str, int] = {}
    process_import_ids: set[int] = set()
    regexp_prototype_unproven = False
    process_options_unproven = False
    local_regex_calls: list[Node] = []
    regex_arguments: list[Node] = []
    regex_constructors: list[Node] = []
    override_names = {
        "RegExp",
        "Reflect",
        "constructor",
        "__proto__",
        "defineProperty",
        "defineProperties",
        "getPrototypeOf",
        "setPrototypeOf",
        "getOwnPropertyDescriptor",
        "getOwnPropertyDescriptors",
        "__defineGetter__",
        "__defineSetter__",
    }
    for node in program.nodes:
        check_runtime()
        if node.type == "subscript_expression":
            index = node.child_by_field_name("index")
            if index is None or index.type != "number":
                regexp_prototype_unproven = True
        if node.type == "method_definition" and any(
            t.type in {"get", "set"} for t in node.children
        ):
            process_options_unproven = True
        if node.type in {"identifier", "property_identifier"} and "\\" in program.text(node):
            # Escaped names can denote the same binding or mutation primitive.
            regexp_prototype_unproven = True
        if node.type == "with_statement":
            return set(), set(), set()
        if node.type in {
            "identifier",
            "shorthand_property_identifier",
            "shorthand_property_identifier_pattern",
        }:
            references.setdefault(program.text(node), []).append(node)
        if node.type in {"identifier", "property_identifier", "string"}:
            spelling = program.text(node).strip("\"'")
            if spelling in {"Proxy", "__defineGetter__", "__defineSetter__"}:
                process_options_unproven = True
            if spelling in {
                "__proto__",
                "getPrototypeOf",
                "setPrototypeOf",
                "defineProperty",
                "defineProperties",
                "prototype",
                "Reflect",
            }:
                process_options_unproven = True
            if spelling in {"assign", "defineGetter", "defineSetter", "globalThis", "global"}:
                regexp_prototype_unproven = True
            if spelling in override_names:
                intrinsic_override = True
                parent = node.parent
                if not (
                    spelling == "RegExp"
                    and parent is not None
                    and (
                        parent.type == "new_expression"
                        and parent.child_by_field_name("constructor") == node
                        or parent.type == "binary_expression"
                        and parent.child_by_field_name("right") == node
                        and (operator := parent.child_by_field_name("operator")) is not None
                        and program.text(operator) == "instanceof"
                    )
                ):
                    regexp_prototype_unproven = True
            if spelling in {"node:process", "node:module", "globalThis", "global"} or (
                spelling == "process" and not _ordinary_process_read(program, node)
            ):
                process_options_unproven = True
            if spelling in {"syncBuiltinESMExports", "require", "getBuiltinModule"}:
                module_unproven = True
            if spelling == "prototype":
                # A direct hasOwnProperty.call read does not expose a prototype
                # to an uploaded mutator. Other prototype uses remain unproved.
                owner = node.parent
                method = None if owner is None else owner.parent
                helper = None if method is None else method.parent
                call = None if helper is None else helper.parent
                if not (
                    owner is not None
                    and owner.type == "member_expression"
                    and (object_node := owner.child_by_field_name("object")) is not None
                    and program.text(object_node) == "Object"
                    and method is not None
                    and method.type == "member_expression"
                    and _callable_member(program, method) == "hasOwnProperty"
                    and helper is not None
                    and helper.type == "member_expression"
                    and _callable_member(program, helper) == "call"
                    and call is not None
                    and call.type == "call_expression"
                    and call.child_by_field_name("function") == helper
                ):
                    intrinsic_override = True
                    regexp_prototype_unproven = True
        if node.type in {
            "assignment_expression",
            "augmented_assignment_expression",
            "update_expression",
        }:
            target = node.child_by_field_name("left") or node.child_by_field_name("argument")
            if target is not None and (
                target.type == "subscript_expression"
                or (
                    target.type == "member_expression"
                    and _callable_member(program, target) in {"exec", "test"}
                )
            ):
                regexp_prototype_unproven = True
        if node.type == "this":
            intrinsic_override = True
        if (
            node.type == "new_expression"
            and (constructor := node.child_by_field_name("constructor")) is not None
            and constructor.type == "identifier"
            and program.text(constructor) == "RegExp"
        ):
            # A direct .exec/.test call on a constructed regex is owned
            # like the literal form, and is not an escaping constructor.
            # It rides local_regex_calls, not regex_calls: the RegExp
            # constructor name itself sets intrinsic_override, which
            # clears regex_calls, so a direct append would be wiped.
            member = _regex_receiver_call(program, node)
            if member is not None:
                if _callable_member(program, member) == "exec":
                    local_regex_calls.append(member)
                continue
            owner = node.parent
            binding = None if owner is None else owner.child_by_field_name("name")
            declaration = None if owner is None else owner.parent
            if (
                owner is not None
                and owner.type == "variable_declarator"
                and owner.child_by_field_name("value") == node
                and binding is not None
                and binding.type == "identifier"
                and declaration is not None
                and declaration.type == "lexical_declaration"
                and declaration.children[0].type == "const"
                and declaration.parent is not None
                and declaration.parent.type in {"program", "statement_block"}
            ):
                regex_bindings[program.text(binding)] = binding
            else:
                regex_constructors.append(node)
        if node.type == "regex":
            member = node.parent
            call = None if member is None else member.parent
            if (
                member is not None
                and member.type == "member_expression"
                and member.child_by_field_name("object") == node
                and _callable_member(program, member) in {"exec", "test"}
                and call is not None
                and call.type == "call_expression"
                and call.child_by_field_name("function") == member
                and (arguments := call.child_by_field_name("arguments")) is not None
                and arguments.type != "template_string"
            ):
                if _callable_member(program, member) == "exec":
                    regex_calls.append(member)
            elif member is not None and member.type == "variable_declarator":
                binding = member.child_by_field_name("name")
                declaration = member.parent
                if (
                    binding is not None
                    and binding.type == "identifier"
                    and member.child_by_field_name("value") == node
                    and declaration is not None
                    and declaration.type == "lexical_declaration"
                    and declaration.children[0].type == "const"
                    and declaration.parent is not None
                    and declaration.parent.type in {"program", "statement_block"}
                ):
                    regex_bindings[program.text(binding)] = binding
                else:
                    intrinsic_override = True
                    regexp_prototype_unproven = True
            else:
                # A literal passed to opaque code can expose the shared
                # prototype, including to code outside this module.
                intrinsic_override = True
                if member is not None and member.type == "arguments":
                    regex_arguments.append(node)
                else:
                    regexp_prototype_unproven = True
        if node.type == "call_expression":
            callee = node.child_by_field_name("function")
            arguments = node.child_by_field_name("arguments")
            if callee is not None and program.text(callee) in {"require", "import"}:
                args = [] if arguments is None else arguments.named_children
                if (
                    len(args) != 1
                    or args[0].type != "string"
                    or program.text(args[0]).strip("\"'").removeprefix("node:")
                    in {"path", "fs", "fs/promises", "child_process"}
                ):
                    module_unproven = True
        if node.type != "import_statement":
            continue
        source = node.child_by_field_name("source")
        module = "" if source is None else program.text(source).strip("\"'")
        if module.removeprefix("node:").startswith("path/"):
            module_unproven = True
        module = module.removeprefix("node:")
        if module not in {"path", "fs", "fs/promises", "child_process"}:
            continue
        family = module.split("/")[0]
        imports[family] = imports.get(family, 0) + 1
        methods = _PATH_FUNCTIONS if module == "path" else _FS_PATH_METHODS | _FS_FD_METHODS
        if family == "child_process":
            process_import_ids.add(node.id)
            methods = {"spawn", "spawnSync", "execFile", "execFileSync"}
        for clause in node.named_children:
            if clause.type != "import_clause":
                continue
            for selector in clause.named_children:
                if selector.type == "named_imports":
                    continue
                if family == "child_process":
                    module_unproven = True
                    continue
                binding = selector
                if selector.type == "namespace_import" and len(selector.named_children) == 1:
                    binding = selector.named_children[0]
                if binding.type != "identifier" or program.text(binding).casefold() in {
                    "eval",
                    "function",
                }:
                    module_unproven = True
                    continue
                bindings[program.text(binding)] = (binding, family, "", True)
        pending = list(node.named_children)
        while pending:
            check_runtime()
            specifier = pending.pop()
            if specifier.type == "import_specifier":
                original = specifier.child_by_field_name("name")
                binding = specifier.child_by_field_name("alias") or original
                if original is not None and binding is not None:
                    name = program.text(original)
                    if program.text(binding).casefold() in {"eval", "function"}:
                        module_unproven = True
                        continue
                    path_namespace = module == "path" and name in {"posix", "win32"}
                    if name in methods or path_namespace:
                        if family == "child_process":
                            process_import_ids.add(original.id)
                        bindings[program.text(binding)] = (binding, family, name, path_namespace)
                    else:
                        module_unproven = True
            pending.extend(specifier.named_children)
    for name, binding in regex_bindings.items():
        named_calls: list[Node] = []
        declarator = binding.parent
        declaration = None if declarator is None else declarator.parent
        scope = None if declaration is None else declaration.parent
        for reference in references[name]:
            check_runtime()
            if scope is None or not scope.start_byte <= reference.start_byte < scope.end_byte:
                regexp_prototype_unproven = True
                break
            if reference == binding:
                continue
            member = reference.parent
            call = None if member is None else member.parent
            method = None if member is None else _callable_member(program, member)
            if (
                member is not None
                and member.type == "member_expression"
                and member.child_by_field_name("object") == reference
                and method == "lastIndex"
                and call is not None
                and call.type == "assignment_expression"
                and call.child_by_field_name("left") == member
                and (value := call.child_by_field_name("right")) is not None
                and value.type == "number"
            ):
                continue
            if not (
                member is not None
                and member.type == "member_expression"
                and member.child_by_field_name("object") == reference
                and method in {"test", "exec"}
                and call is not None
                and call.type == "call_expression"
                and call.child_by_field_name("function") == member
                and (arguments := call.child_by_field_name("arguments")) is not None
                and arguments.type != "template_string"
            ):
                intrinsic_override = True
                regexp_prototype_unproven = True
                break
            if method == "exec":
                named_calls.append(member)
        else:
            local_regex_calls.extend(named_calls)
    for reference in references.get("Object", ()):
        parent = reference.parent
        if (
            parent is None
            or parent.type != "member_expression"
            or parent.child_by_field_name("object") != reference
        ):
            intrinsic_override = True
            regexp_prototype_unproven = True
    owned: set[int] = set()
    native_calls: set[int] = set()
    if intrinsic_override:
        regex_calls.clear()
    if (
        not regexp_prototype_unproven
        and local_regex_calls
        and _local_regex_arguments(
            program,
            references,
            regex_arguments,
            regex_constructors,
            {member.id for member in local_regex_calls},
            check_runtime,
        )
    ):
        regex_calls.extend(local_regex_calls)
    for member in regex_calls:
        owned.add(member.id)
        property_node = member.child_by_field_name("property")
        if property_node is not None:
            owned.add(property_node.id)
    rejected: set[str] = set()
    module_calls: dict[str, set[int]] = {}
    module_identifiers: dict[str, set[int]] = {}
    for name, (binding, family, imported_name, namespace) in bindings.items():
        calls: set[int] = set()
        identifiers = {binding.id}
        methods = _PATH_FUNCTIONS if family == "path" else _FS_PATH_METHODS | _FS_FD_METHODS
        for reference in references[name]:
            check_runtime()
            if reference == binding:
                continue
            callee = reference.parent if namespace else reference
            if namespace and (
                callee is None
                or callee.type != "member_expression"
                or callee.child_by_field_name("object") != reference
                or _callable_member(program, callee) not in methods
            ):
                break
            call = None if callee is None else callee.parent
            arguments = None if call is None else call.child_by_field_name("arguments")
            if (
                callee is None
                or call is None
                or call.type != "call_expression"
                or call.child_by_field_name("function") != callee
                or arguments is None
                or arguments.type != "arguments"
            ):
                break
            selected_method = _callable_member(program, callee) if namespace else imported_name
            if family == "child_process":
                if process_options_unproven or not _fixed_revision_query(program, arguments):
                    break
                calls.add(call.id)
            elif family == "path":
                calls.add(call.id)
            elif selected_method in _FS_PATH_METHODS:
                # Exempt only the first argument when checking imported callables.
                # Nested calls and all other arguments keep their own analysis.
                args = [arg for arg in arguments.named_children if arg.type != "comment"]
                if args and args[0].type != "spread_element":
                    calls.add(args[0].id)
            identifiers.add(reference.id)
        else:
            if family == "child_process" and not calls:
                rejected.add(family)
            module_calls.setdefault(family, set()).update(calls)
            module_identifiers.setdefault(family, set()).update(identifiers)
            continue
        rejected.add(family)
    native_paths: set[int] = set()
    for family, calls in module_calls.items():
        if not module_unproven and family not in rejected and imports[family] == 1:
            if family == "child_process":
                owned.update(process_import_ids)
            if family in {"path", "child_process"}:
                native_calls.update(calls)
            else:
                native_paths.update(calls)
            owned.update(module_identifiers[family])
    return owned, native_calls, native_paths


def _dynamic_process_call(
    program: JavaScriptProgram, statement: Node, check_runtime: Callable[[], None]
) -> Node | None:
    """Locate a direct Node dispatch with a runtime-selected executable or argv."""
    names: dict[str, Node] = {}
    pending = [statement]
    while pending:
        check_runtime()
        part = pending.pop()
        if part.type == "import_specifier":
            origin = part.child_by_field_name("name")
            binding = part.child_by_field_name("alias") or origin
            if (
                origin is not None
                and binding is not None
                and program.text(origin)
                in {
                    "spawn",
                    "spawnSync",
                    "exec",
                    "execSync",
                    "execFile",
                    "execFileSync",
                }
            ):
                names[program.text(binding)] = binding
        pending.extend(part.named_children)
    if not names:
        return None
    selected: dict[str, Node] = {}
    uncertain: set[str] = set()
    for node in program.nodes:
        check_runtime()
        if node.type not in {
            "identifier",
            "shorthand_property_identifier",
            "shorthand_property_identifier_pattern",
        }:
            continue
        name = program.text(node)
        if name not in names or node == names[name]:
            continue
        call = node.parent
        if (
            call is None
            or call.type != "call_expression"
            or call.child_by_field_name("function") != node
        ):
            uncertain.add(name)
            continue
        args = call.child_by_field_name("arguments")
        words = [] if args is None else [a for a in args.named_children if a.type != "comment"]
        if words and (
            words[0].type != "string"
            or len(words) > 1
            and (
                words[1].type == "identifier"
                or words[1].type == "array"
                and any(a.type == "spread_element" for a in words[1].named_children)
            )
        ):
            selected.setdefault(name, call)
    return next((call for name, call in selected.items() if name not in uncertain), None)


def _ordinary_process_read(program: JavaScriptProgram, identifier: Node) -> bool:
    """The global process and its environment must not escape or select another executable."""
    parent = identifier.parent
    if (
        identifier.type != "identifier"
        or parent is None
        or parent.type != "member_expression"
        or parent.child_by_field_name("object") != identifier
    ):
        return False
    name = _callable_member(program, parent)
    if name != "env":
        return name in {"argv", "cwd", "exit", "on", "pid", "stderr", "stdin", "stdout", "version"}
    value = parent.parent
    if (
        value is None
        or value.type != "member_expression"
        or value.child_by_field_name("object") != parent
    ):
        return False
    # Only an ordinary read of one environment string. Walk assignment
    # patterns as well; destructuring can write the same property as '='.
    usage = value.parent
    while usage is not None and usage.type in {
        "array_pattern",
        "object_pattern",
        "pair_pattern",
        "parenthesized_expression",
        "rest_pattern",
    }:
        value, usage = usage, usage.parent
    return usage is not None and not (
        usage.child_by_field_name("left") == value
        or usage.type == "update_expression"
        or (
            usage.type == "unary_expression"
            and (op := usage.child_by_field_name("operator")) is not None
            and program.text(op) == "delete"
        )
    )


def _fixed_revision_query(program: JavaScriptProgram, arguments: Node) -> bool:
    """Recognize a Git metadata query with an explicit argv and no command interpreter."""
    args = [n for n in arguments.named_children if n.type != "comment"]

    def literal(n: Node) -> str | None:
        if n.type != "string":
            return None
        return program.text(n)[1:-1]

    def dense_array(node: Node) -> list[Node] | None:
        if node.type != "array" or node.child_count > 128:
            return None
        entries = [n for n in node.children if n.type not in {"[", "]", "comment"}]
        if any(n.type in {",", "spread_element"} for n in entries[::2]) or any(
            n.type != "," for n in entries[1::2]
        ):
            return None
        return entries[::2]

    if len(args) not in {2, 3} or literal(args[0]) != "git":
        return False
    # An elision, spread, option-injecting operand or dynamic command cannot
    # borrow the fixed argument layout. A directory after -C is one value.
    words = dense_array(args[1])
    if words is None:
        return False
    command_index = 0
    while command_index + 1 < len(words) and literal(words[command_index]) == "-C":
        path = words[command_index + 1]
        while path.type == "member_expression":
            base = path.child_by_field_name("object")
            if base is None:
                return False
            path = base
        if path.type not in {"identifier", "string"}:
            return False
        command_index += 2
    if command_index + 1 >= len(words) or literal(words[command_index]) != "rev-parse":
        return False
    operands = {literal(n) for n in words[command_index + 1 :]}
    if not operands <= {
        "HEAD",
        "--abbrev-ref",
        "--git-dir",
        "--absolute-git-dir",
        "--git-common-dir",
        "--show-toplevel",
        "--show-prefix",
        "--is-inside-work-tree",
        "--is-bare-repository",
        "--short",
        "--verify",
    }:
        return False
    if len(args) == 2:
        return True
    if args[2].type != "object" or args[2].child_count > 32:
        return False
    options: set[str] = set()
    for pair in args[2].named_children:
        if pair.type == "comment":
            continue
        key, value = pair.child_by_field_name("key"), pair.child_by_field_name("value")
        if pair.type != "pair" or key is None or key.type != "property_identifier" or value is None:
            return False
        name = program.text(key)
        if name in options:
            return False
        options.add(name)
        if name == "shell" and value.type == "false":
            continue
        if name == "encoding" and value.type == "string":
            continue
        if name in {"timeout", "maxBuffer"} and value.type == "number":
            continue
        if name == "stdio":
            values = dense_array(value) if value.type == "array" else [value]
            if values is not None and all(
                literal(v) in {"ignore", "pipe", "inherit"} for v in values
            ):
                continue
        return False
    return True
