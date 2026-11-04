# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reached imported functions, not all definitions, own source-write effects."""

import pytest

from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.shell_source_effects import (
    SHELL_PID_MARKER,
    ShellEffectResult,
    source_effects_collect,
    source_effects_preserve,
)
from skillspector.shell_sources import ShellSourceContext, ShellSourceResolver


def _resolver(caller: str, library: str) -> ShellSourceResolver:
    return ShellSourceResolver(
        ShellSourceContext(
            "scripts/caller.sh",
            {
                "scripts/caller.sh": caller,
                "scripts/library.sh": library,
            },
        ),
        tm._next_shell_invocation_word,
        lambda text, start, end: tm._skip_command_substitution(text, start, end, lambda: None),
        tm._static_heredoc_header,
        tm._skip_static_heredoc_bodies,
        lambda: None,
    )


def _preserves(caller: str, library: str, *, use_offset: int | None = None, **kwargs) -> bool:
    resolver = _resolver(caller, library)
    return source_effects_preserve(
        resolver, "scripts/library.sh", len(caller) if use_offset is None else use_offset, **kwargs
    )


def _pure_preserve_effect(command):
    if command.child_inspected:
        return ShellEffectResult(command.stdout, command.child_status)
    if command.argv[0] in {":", "true", "false", "printf"}:
        return ShellEffectResult(
            command.stdout if command.argv[0] == "printf" else "",
            command.argv[0] != "false",
            native_identity=True,
        )
    return None


def _preserves_joined(caller: str, **kwargs) -> bool:
    resolver = _resolver(caller, _LIBRARY)
    kwargs.setdefault("read_source", resolver.context.sources.get)
    kwargs.setdefault("preserve_effect", _pure_preserve_effect)
    kwargs.setdefault("preserve_state", lambda: ())
    return source_effects_preserve(resolver, "scripts/library.sh", len(caller), **kwargs)


_LIBRARY = """set -euo pipefail
write_file() { local target="$1"; printf '%s' changed > "$target"; }
main() { write_file scripts/library.sh; }
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
"""


def test_preservation_mode_is_explicit_and_cannot_collect_generated_facts() -> None:
    caller = 'ignored="${!#}"\n'
    resolver = _resolver(caller, _LIBRARY)
    assert not _preserves(caller, _LIBRARY)
    assert not _preserves(caller, _LIBRARY, native_effect=_pure_preserve_effect)
    assert not _preserves(caller, _LIBRARY, preserve_effect=_pure_preserve_effect)
    assert not _preserves(":\n", _LIBRARY, preserve_state=lambda: ())
    assert not _preserves_joined(caller, native_effect=_pure_preserve_effect)
    assert not source_effects_collect(
        resolver,
        len(caller),
        read_source=resolver.context.sources.get,
        native_effect=_pure_preserve_effect,
    )
    assert not source_effects_preserve(
        resolver,
        "scripts/generated.sh",
        len(caller),
        read_source=resolver.context.sources.get,
        preserve_effect=_pure_preserve_effect,
    )
    assert _preserves_joined(caller)


@pytest.mark.parametrize(
    "consumer",
    [
        '"$value"',
        'printf changed > "$value"',
        'source "$value"',
        'bash -c "$value"',
    ],
)
def test_unknown_last_positional_data_cannot_supply_executable_or_path(consumer: str) -> None:
    assert not _preserves_joined('value="${!#}"\n' + consumer + "\n")


@pytest.mark.parametrize(
    "expansion", ["${!unknown}", "${!items[$(printf changed > scripts/library.sh)]}"]
)
def test_preservation_mode_does_not_generalize_indirect_parameter_expansion(expansion: str) -> None:
    assert not _preserves_joined('value="' + expansion + '"\n')


def test_preservation_joins_inert_branches_without_expanding_exact_collection() -> None:
    caller = 'if [[ "$unknown" == yes ]]; then value=one; else value=two; fi\n' * 9
    assert not _preserves(caller, _LIBRARY, native_effect=_pure_preserve_effect)
    assert _preserves_joined(caller)
    assert not _preserves_joined(caller + "printf changed > scripts/library.sh\n")


@pytest.mark.parametrize(
    "left, right, preserved",
    [
        ("/tmp/other", "/tmp/other", True),
        ("scripts/library.sh", "/tmp/other", False),
        ("/tmp/other", "scripts/library.sh", False),
        ("scripts/library.sh", "scripts/library.sh", False),
    ],
)
def test_preservation_join_retains_equal_paths_and_refuses_unknown_destinations(
    left: str, right: str, preserved: bool
) -> None:
    caller = (
        'if [[ "$unknown" == yes ]]; then path='
        + left
        + "; else path="
        + right
        + '; fi\nprintf changed > "$path"\n'
    )
    assert _preserves_joined(caller) is preserved


@pytest.mark.parametrize(
    "branch",
    [
        "writer() { :; }",
        'trap ":" EXIT',
        "export data",
        "shift",
    ],
)
def test_preservation_join_refuses_conflicting_code_or_scope_identity(branch: str) -> None:
    caller = 'check() { if [[ "$unknown" == yes ]]; then ' + branch + "; fi; }\ncheck one two\n"
    assert not _preserves_joined(caller)
    assert not _preserves_joined(
        'check() { if [[ "$unknown" == yes ]]; then local data=value; fi; }\ncheck\n'
    )


def test_preservation_join_keeps_exited_and_continuing_effects_separate() -> None:
    caller = 'if [[ "$unknown" == yes ]]; then exit 0; else path=scripts/library.sh; fi\nprintf changed > "$path"\n'
    assert not _preserves_joined(caller)
    assert _preserves_joined(caller.replace("path=scripts/library.sh", "path=/tmp/other"))


def test_preservation_keeps_distinct_exited_callable_states_without_guessing() -> None:
    caller = """bash -c 'if [[ "$unknown" == yes ]]; then action() { printf one; }; exit 0; else action() { printf two; }; exit 0; fi'
"""
    assert _preserves_joined(caller)


@pytest.mark.parametrize("trap", ['trap "action" EXIT', 'trap "printf safe" EXIT'])
def test_distinct_exited_states_keep_each_late_exit_trap_binding(trap: str) -> None:
    caller = (
        """bash -c 'action() { printf safe; }; trap "action" EXIT; if [[ "$unknown" == yes ]]; then action() { printf changed > scripts/library.sh; }; exit 0; else """
        + trap
        + """; exit 0; fi'
"""
    )
    assert not _preserves_joined(caller)
    assert _preserves_joined(caller.replace("printf changed > scripts/library.sh", "printf other"))


@pytest.mark.parametrize("branches, preserved", [(127, True), (128, False)])
def test_retained_exit_alternatives_keep_the_shared_state_bound(
    branches: int, preserved: bool
) -> None:
    caller = 'if [[ "$unknown" == yes ]]; then exit 0; fi\n' * branches
    assert _preserves_joined(caller) is preserved


def test_preservation_mode_requires_current_source_dependency_bytes() -> None:
    assert not _preserves_joined("source scripts/library.sh\n", read_source=lambda path: None)
    assert not _preserves_joined("unknown_helper /tmp/other\n")

    class UnavailableSource:
        def __bool__(self):
            return False

        def __call__(self, path):
            return None

    assert not _preserves_joined("source scripts/library.sh\n", read_source=UnavailableSource())


def test_preservation_callbacks_see_every_optional_write_before_join() -> None:
    writes = set()

    def callback(command):
        writes.update(value for operator, value in command.redirects if ">" in operator)
        return _pure_preserve_effect(command)

    caller = 'if [[ "$unknown" == yes ]]; then printf one > /tmp/left; else printf two > /tmp/right; fi\n'
    assert _preserves_joined(caller, preserve_effect=callback)
    assert writes == {"/tmp/left", "/tmp/right"}


def test_unknown_array_loop_is_preservation_only_and_audits_its_body() -> None:
    caller = 'for item in "${items[@]}"; do printf "%s" "$item"; done\n'
    assert not _preserves(caller, _LIBRARY, native_effect=_pure_preserve_effect)
    assert not _preserves_joined(caller, preserve_state=None)
    assert _preserves_joined(caller)
    assert not _preserves_joined(
        caller.replace('printf "%s" "$item"', "printf changed > scripts/library.sh")
    )


@pytest.mark.parametrize(
    "body",
    [
        'printf changed > "$item"',
        '"$item" /tmp/other',
        'source "$item"',
        'bash -c "$item"',
        'unknown_helper "$item"',
    ],
)
def test_unknown_array_items_cannot_supply_paths_or_code(body: str) -> None:
    assert not _preserves_joined('for item in "${items[@]}"; do ' + body + "; done\n")


@pytest.mark.parametrize("flow", ["break", "continue", "return", "exit"])
def test_unknown_array_loop_does_not_erase_zero_iterations(flow: str) -> None:
    caller = (
        'check() { path=scripts/library.sh; for item in "${items[@]}"; do path=/tmp/other; '
        + flow
        + '; done; printf changed > "$path"; }\ncheck\n'
    )
    assert not _preserves_joined(caller)


@pytest.mark.parametrize("flow", ["break", "continue", "return", "exit"])
def test_unknown_array_loop_audits_writes_before_control_transfer(flow: str) -> None:
    caller = (
        'check() { for item in "${items[@]}"; do printf changed > scripts/library.sh; '
        + flow
        + "; done; }\ncheck\n"
    )
    assert not _preserves_joined(caller)
    assert _preserves_joined(caller.replace("printf changed > scripts/library.sh", "printf safe"))


def test_unknown_array_loop_widens_before_a_possible_second_iteration() -> None:
    caller = 'path=/tmp/other\nfor item in "${items[@]}"; do printf changed > "$path"; path=scripts/library.sh; done\n'
    assert not _preserves_joined(caller)
    assert _preserves_joined(caller.replace("path=scripts/library.sh", "path=/tmp/other"))


def test_unknown_array_loop_keeps_break_paths_out_of_later_iterations() -> None:
    caller = 'for item in "${items[@]}"; do break; printf changed > scripts/library.sh; done\n'
    assert _preserves_joined(caller)
    assert not _preserves_joined(caller + "printf changed > scripts/library.sh\n")


@pytest.mark.parametrize(
    "body", ["callee() { :; }", 'trap ":" EXIT', "export changed", "local changed=value", "shift"]
)
def test_unknown_array_loop_refuses_changing_callable_or_scope_identity(body: str) -> None:
    caller = 'check() { for item in "${items[@]}"; do ' + body + "; done; }\ncheck one two\n"
    assert not _preserves_joined(caller)


def test_unknown_array_loop_callbacks_cannot_publish_one_iterations_output() -> None:
    requests = []

    def callback(command):
        requests.append(command)
        return _pure_preserve_effect(command)

    assert _preserves_joined(
        'for item in "${items[@]}"; do printf safe > /tmp/output; done\n', preserve_effect=callback
    )
    assert requests and all(command.uncertain for command in requests)


def test_known_array_loop_stays_exact_in_preservation_mode() -> None:
    caller = 'items=()\nfor item in "${items[@]}"; do printf changed > scripts/library.sh; done\n'
    assert _preserves_joined(caller, preserve_state=None)
    assert not _preserves_joined(caller.replace("items=()", "items=(one)"))


def test_unknown_loop_rechecks_a_file_predicate_after_late_mutation() -> None:
    touched = set()

    def callback(command):
        if command.argv == ("test", "-e", "/tmp/marker"):
            return ShellEffectResult(status=bool(touched))
        if command.argv == ("touch", "/tmp/marker"):
            touched.add("/tmp/marker")
            return ShellEffectResult(writes=("/tmp/marker",))
        return _pure_preserve_effect(command)

    caller = 'for item in "${items[@]}"; do if [[ -e /tmp/marker ]]; then printf changed > scripts/library.sh; fi; touch /tmp/marker; done\n'
    assert not _preserves_joined(
        caller, preserve_effect=callback, preserve_state=lambda: frozenset(touched)
    )
    assert touched == {"/tmp/marker"}


def test_unknown_loop_rechecks_current_helper_bytes_after_late_rewrite() -> None:
    rewritten = set()

    def callback(command):
        if ("1>", "/tmp/helper.sh") in command.redirects:
            rewritten.add("/tmp/helper.sh")
        return _pure_preserve_effect(command)

    def read_source(path):
        return ":\n" if path == "/tmp/helper.sh" and not rewritten else None

    caller = 'for item in "${items[@]}"; do source /tmp/helper.sh; printf changed > /tmp/helper.sh; done\n'
    assert not _preserves_joined(
        caller,
        read_source=read_source,
        preserve_effect=callback,
        preserve_state=lambda: frozenset(rewritten),
    )
    assert rewritten == {"/tmp/helper.sh"}


def test_unknown_loop_repeated_disjoint_write_can_reach_semantic_fixed_point() -> None:
    writes = set()

    def callback(command):
        writes.update(value for operator, value in command.redirects if ">" in operator)
        return _pure_preserve_effect(command)

    caller = 'for item in "${items[@]}"; do printf safe > /tmp/log; done\n'
    assert _preserves_joined(
        caller, preserve_effect=callback, preserve_state=lambda: frozenset(writes)
    )
    assert writes == {"/tmp/log"}


def test_unknown_loop_refuses_unbounded_callback_state_changes() -> None:
    calls = []

    def callback(command):
        calls.append(command)
        return _pure_preserve_effect(command)

    assert not _preserves_joined(
        'for item in "${items[@]}"; do printf safe; done\n',
        preserve_effect=callback,
        preserve_state=lambda: len(calls),
    )
    assert len(calls) <= 16


def test_import_installs_definitions_without_running_main_or_uncalled_writer() -> None:
    assert _preserves("source scripts/library.sh\n", _LIBRARY)


@pytest.mark.parametrize(
    "diagnostic", ["--board-config requires a value", "$(printf changed > scripts/library.sh)"]
)
def test_present_positional_parameter_skips_its_unused_diagnostic(diagnostic: str) -> None:
    caller = 'parse() { value="${2:?' + diagnostic + '}"; }\nparse --board-config known\n'
    assert _preserves(caller, _LIBRARY)
    assert not _preserves(caller + "printf changed > scripts/library.sh\n", _LIBRARY)


@pytest.mark.parametrize(
    "arguments", ["--board-config", '--board-config ""', '--board-config "$unknown"']
)
def test_missing_or_unknown_positional_diagnostic_cannot_hide_effects(arguments: str) -> None:
    caller = (
        'parse() { value="${2:?$(printf changed > scripts/library.sh)}"; }\nparse '
        + arguments
        + "\n"
    )
    assert not _preserves(caller, _LIBRARY)


@pytest.mark.parametrize(
    "initial, modifier", [("value=known", "-"), ("value=known", "?"), ('value=""', "+")]
)
def test_known_parameter_value_skips_only_unused_fallback(initial: str, modifier: str) -> None:
    caller = (
        initial + '\nresult="${value:' + modifier + '$(printf changed > scripts/library.sh)}"\n'
    )
    assert _preserves(caller, _LIBRARY)


@pytest.mark.parametrize("zero", ["0", "00", "000:?missing"])
def test_zero_parameter_never_selects_the_last_function_argument(zero: str) -> None:
    condition = '[[ "${' + zero + '}" == "$0" ]]'
    safe = (
        "check() { if "
        + condition
        + "; then :; else printf changed > scripts/library.sh; fi; }\ncheck other\n"
    )
    unsafe = (
        "check() { if "
        + condition
        + "; then printf changed > scripts/library.sh; fi; }\ncheck other\n"
    )
    assert _preserves(safe, _LIBRARY)
    assert not _preserves(unsafe, _LIBRARY)


@pytest.mark.parametrize("name", ["0" * 17, "١"])
def test_unproved_numeric_parameter_spelling_refuses(name: str) -> None:
    assert not _preserves('value="${' + name + '}"\n', _LIBRARY)


@pytest.mark.parametrize(
    "condition, outcome",
    [
        ("! x == x || y == y", True),
        ("! x == x && y == y", False),
        ("! x == y || y == x", True),
        ("! x == y && y == x", False),
        ("x == y && y == x || x == x", True),
        ("! ! x == x && y == y", True),
    ],
)
def test_conditional_negation_binds_before_logical_operators(condition: str, outcome: bool) -> None:
    caller = "if [[ " + condition + " ]]; then printf changed > scripts/library.sh; fi\n"
    opposite = "if [[ " + condition + " ]]; then :; else printf changed > scripts/library.sh; fi\n"
    assert _preserves(caller, _LIBRARY) is not outcome
    assert _preserves(opposite, _LIBRARY) is outcome


@pytest.mark.parametrize(
    "condition", ["( x == x || y == y )", "! ( x == y )", "! " * 18 + "x == x"]
)
def test_unsupported_condition_grouping_or_depth_refuses(condition: str) -> None:
    assert not _preserves("if [[ " + condition + " ]]; then :; fi\n", _LIBRARY)


@pytest.mark.parametrize("left, operator", [("x == x", "||"), ("x == y", "&&")])
def test_skipped_conditional_operand_never_runs_substitution_effects(
    left: str, operator: str
) -> None:
    caller = (
        "if [[ "
        + left
        + " "
        + operator
        + ' "$(set -e; printf changed > scripts/library.sh)" ]]; then :; fi\n'
    )
    assert _preserves(caller, _LIBRARY)


@pytest.mark.parametrize("operator", ["||", "&&"])
def test_optional_conditional_operand_cannot_publish_certain_effects(operator: str) -> None:
    caller = '[[ "$unknown" == yes ' + operator + ' "$(set -e; printf value)" ]]\n'
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult(stdout="value", status=True)

    assert _preserves(caller, _LIBRARY, native_effect=callback)
    assert requests and all(command.uncertain for command in requests)


@pytest.mark.parametrize(
    "index, preserved", [("8", False), ("10", True), ("010", False), ("08", False)]
)
def test_array_indices_never_misread_bash_octal_as_decimal(index: str, preserved: bool) -> None:
    items = [f"/tmp/output{number}" for number in range(11)]
    items[8] = "scripts/library.sh"
    caller = "items=(" + " ".join(items) + ')\nprintf changed > "${items[' + index + ']}"\n'
    assert _preserves(caller, _LIBRARY) is preserved


@pytest.mark.parametrize("comparison", ["010 -eq 8", "8 -eq 010", "08 -eq 8"])
def test_nondecimal_conditional_numbers_cannot_skip_source_writes(comparison: str) -> None:
    assert not _preserves(
        "if [[ " + comparison + " ]]; then printf changed > scripts/library.sh; fi\n", _LIBRARY
    )


@pytest.mark.parametrize("number, preserved", [("8", True), ("010", False), ("08", False)])
def test_arithmetic_condition_requires_canonical_decimal(number: str, preserved: bool) -> None:
    caller = "if (( " + number + " )); then :; else printf changed > scripts/library.sh; fi\n"
    assert _preserves(caller, _LIBRARY) is preserved


def test_array_element_length_never_becomes_element_contents() -> None:
    caller = 'items=(safe)\nif [[ "${#items[0]}" == safe ]]; then :; else printf changed > scripts/library.sh; fi\n'
    assert not _preserves(caller, _LIBRARY)
    assert _preserves(
        'items=(safe)\nif [[ "${#items[@]}" == 1 ]]; then :; else printf changed > scripts/library.sh; fi\n',
        _LIBRARY,
    )


@pytest.mark.parametrize(
    "operand, arguments",
    [
        ('"$@x"', "scripts/library.sh /tmp/other"),
        ('"prefix$@"', "/tmp/other scripts/library.sh"),
        ('"${@}x"', "scripts/library.sh /tmp/other"),
    ],
)
def test_embedded_positional_list_never_flattens_distinct_write_paths(
    operand: str, arguments: str
) -> None:
    caller = "remove() { rm " + operand + "; }\nremove " + arguments + "\n"
    assert not _preserves(caller, _LIBRARY)


def test_exact_positional_list_still_preserves_each_argument() -> None:
    caller = 'remove() { rm "$@"; }\n'
    assert _preserves(caller + "remove /tmp/one /tmp/two\n", _LIBRARY)
    assert not _preserves(caller + "remove scripts/library.sh /tmp/two\n", _LIBRARY)


@pytest.mark.parametrize("opening, closing", [("[", " ]"), ("test", "")])
def test_ordinary_test_refuses_unproved_splitting_and_globbing(opening: str, closing: str) -> None:
    caller = (
        "value='x y'\nif "
        + opening
        + " $value != z"
        + closing
        + "; then :; else printf changed > scripts/library.sh; fi\n"
    )
    assert not _preserves(caller, _LIBRARY)
    assert _preserves(caller.replace("$value", '"$value"'), _LIBRARY)
    pattern = (
        "if "
        + opening
        + " safe == s*"
        + closing
        + "; then :; else printf changed > scripts/library.sh; fi\n"
    )
    assert not _preserves(pattern, _LIBRARY)


def test_extended_condition_keeps_its_distinct_no_splitting_and_pattern_rules() -> None:
    assert _preserves(
        "value='x y'\nif [[ $value != z ]]; then :; else printf changed > scripts/library.sh; fi\n",
        _LIBRARY,
    )
    assert _preserves(
        "if [[ safe == s* ]]; then :; else printf changed > scripts/library.sh; fi\n", _LIBRARY
    )


def test_ordinary_test_substitution_output_cannot_skip_field_splitting() -> None:
    value = "$(printf '%s' 'x y')"
    caller = "if [ " + value + " != z ]; then :; else printf changed > scripts/library.sh; fi\n"
    assert not _preserves(caller, _LIBRARY)
    assert _preserves(caller.replace(value, '"' + value + '"'), _LIBRARY)
    assert not _preserves("rm $(printf '%s' 'scripts/library.sh /tmp/other')\n", _LIBRARY)


def test_ordinary_test_brace_expansion_cannot_skip_argument_count_failure() -> None:
    caller = "if [ x != {x,y} ]; then :; else printf changed > scripts/library.sh; fi\n"
    assert not _preserves(caller, _LIBRARY)
    assert _preserves(caller.replace("{x,y}", "'{x,y}'"), _LIBRARY)


@pytest.mark.parametrize("prefix, operand", [("value=~\n", '"$value"'), ("", "~")])
def test_unquoted_tilde_never_becomes_literal_in_assignment_or_condition(
    prefix: str, operand: str
) -> None:
    caller = (
        prefix
        + "if [[ "
        + operand
        + ' == "~" ]]; then :; else printf changed > scripts/library.sh; fi\n'
    )
    assert not _preserves(caller, _LIBRARY)
    quoted = (
        prefix.replace("=~", "='~'")
        + "if [[ "
        + ("'~'" if operand == "~" else operand)
        + ' == "~" ]]; then :; else printf changed > scripts/library.sh; fi\n'
    )
    assert _preserves(quoted, _LIBRARY)


@pytest.mark.parametrize("value", ["end$", '"end$"'])
def test_terminal_dollar_is_literal_without_a_parameter_name(value: str) -> None:
    caller = (
        "value="
        + value
        + "\nif [[ \"$value\" == 'end$' ]]; then :; else printf changed > scripts/library.sh; fi\n"
    )
    assert _preserves(caller, _LIBRARY)


def test_regex_anchor_stays_literal_without_claiming_regex_truth() -> None:
    condition = '[[ "$BRANCH" =~ ^[A-Za-z0-9._-]+$ ]]'
    assert _preserves("BRANCH=dev\n" + condition + "\n", _LIBRARY)
    assert not _preserves(
        "BRANCH=dev\nif " + condition + "; then printf changed > scripts/library.sh; fi\n", _LIBRARY
    )


@pytest.mark.parametrize(
    "left, right",
    [
        ('"$(printf changed > scripts/library.sh)"', "literal"),
        ("literal", '"$(printf changed > scripts/library.sh)"'),
    ],
)
def test_regex_operands_still_audit_executable_substitutions(left: str, right: str) -> None:
    assert not _preserves("[[ " + left + " =~ " + right + " ]]\n", _LIBRARY)


@pytest.mark.parametrize("value", ["$missing", '"$missing"', "$[1+1]"])
def test_real_or_unsupported_dollar_expansion_does_not_become_literal(value: str) -> None:
    assert not _preserves("value=" + value + '\nprintf changed > "$value"\n', _LIBRARY)


@pytest.mark.parametrize("executable", ["df", "/bin/df", "/usr/bin/df"])
def test_proven_disk_metadata_query_can_run_beside_active_log(executable: str) -> None:
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult(native_identity=command.identity_only)

    assert _preserves(
        "( " + executable + " -B1 --output=avail /tmp ) > /tmp/log\n",
        _LIBRARY,
        native_effect=callback,
    )
    probes = [command for command in requests if command.identity_only]
    assert len(probes) == 1 and probes[0].redirects == ()


@pytest.mark.parametrize(
    "proof",
    [
        None,
        ShellEffectResult(),
        ShellEffectResult(native_identity=True, script_path="scripts/library.sh"),
        ShellEffectResult(native_identity=True, writes=("/tmp/log",)),
    ],
)
def test_unknown_or_generated_disk_query_cannot_skip_active_log_reads(proof) -> None:
    def callback(command):
        return proof if command.identity_only else ShellEffectResult()

    assert not _preserves(
        "( df -B1 --output=avail /tmp ) > /tmp/log\n", _LIBRARY, native_effect=callback
    )


@pytest.mark.parametrize(
    "caller",
    [
        'df() { cat "$3/log"; }\n( df -B1 --output=avail /tmp ) > /tmp/log\n',
        "( df -B1 --output=avail /tmp > /tmp/log ) > /tmp/log\n",
        "( df /tmp ) > /tmp/log\n",
    ],
)
def test_disk_metadata_exception_keeps_shadow_and_write_guards(caller: str) -> None:
    assert not _preserves(
        caller, _LIBRARY, native_effect=lambda command: ShellEffectResult(native_identity=True)
    )


@pytest.mark.parametrize("interpreter", ["bash", "/bin/bash", "/usr/bin/bash"])
def test_explicit_bash_retains_script_zero_arguments_environment_and_cwd(interpreter: str) -> None:
    requests = []

    def callback(command):
        requests.append(command)
        if command.interpreter is not None and not command.child_inspected:
            return ShellEffectResult(script_path="scripts/library.sh")
        return ShellEffectResult(
            command.stdout,
            command.child_status if command.child_inspected else True,
            native_identity=command.interpreter is None,
        )

    caller = (
        "set -e\nexport CARRY=kept\ncd scripts\nPREFIX=value "
        + interpreter
        + " ./library.sh argument\n"
    )
    condition = (
        '[[ "$0" == ./library.sh && "$1" == argument && "$CARRY" == kept && "$PREFIX" == value ]]'
    )
    library = "if " + condition + "; then printf ready; else printf changed > library.sh; fi\n"
    assert _preserves(caller, library, native_effect=callback)
    invocations = [command for command in requests if command.interpreter is not None]
    assert len(invocations) == 2
    assert all(
        command.argv == ("./library.sh", "argument")
        and command.cwd == "scripts"
        and command.interpreter == interpreter
        for command in invocations
    )
    assert invocations[-1].child_inspected
    assert not _preserves(
        caller,
        library.replace(
            "then printf ready; else printf changed > library.sh",
            "then printf changed > library.sh; else printf ready",
        ),
        native_effect=callback,
    )


@pytest.mark.parametrize(
    "call", ["sh scripts/library.sh", "bash -e scripts/library.sh", "bash -", 'bash "$missing"']
)
def test_explicit_script_bridge_refuses_other_interpreters_options_and_unknown_paths(
    call: str,
) -> None:
    assert not _preserves(call + "\n", ":\n", native_effect=lambda command: None)


def test_explicit_script_requires_an_authoritative_callback() -> None:
    assert not _preserves("bash scripts/library.sh\n", ":\n")
    assert not _preserves("bash scripts/library.sh\n", ":\n", native_effect=lambda command: None)


@pytest.mark.parametrize("name, argument", [("cd", "scripts"), ("set", "-e")])
def test_interpreted_script_name_never_changes_parent_native_state(
    name: str, argument: str
) -> None:
    requests = []

    def callback(command):
        requests.append(command)
        if command.interpreter is not None and not command.child_inspected:
            return ShellEffectResult(script_path="scripts/library.sh")
        return ShellEffectResult(
            command.stdout,
            command.child_status if command.child_inspected else True,
            native_identity=command.interpreter is None,
        )

    assert _preserves(
        "bash " + name + " " + argument + "\nprintf ready\n", ":\n", native_effect=callback
    )
    final = requests[-1]
    assert final.argv == ("printf", "ready") and final.cwd == "." and not final.fail_stop


@pytest.mark.parametrize(
    "caller",
    [
        "( bash scripts/library.sh ) > scripts/library.sh\n",
        "( bash df -B1 --output=avail /tmp ) > /tmp/log\n",
    ],
)
def test_interpreted_scripts_keep_active_output_and_metadata_identity_guards(caller: str) -> None:
    assert not _preserves(
        caller, ":\n", native_effect=lambda command: ShellEffectResult(native_identity=True)
    )


@pytest.mark.parametrize(
    "caller",
    [
        "check() { printf ready; }\n( check /tmp ) > /tmp/log\n",
        '( bash -c "printf ready" child /tmp ) > /tmp/log\n',
    ],
)
def test_audited_child_completion_does_not_reinterpret_arguments_as_file_reads(caller: str) -> None:
    assert _preserves(caller, _LIBRARY)


@pytest.mark.parametrize("body", ['cat "$1/log"', 'printf changed > "$1/log"'])
def test_audited_function_body_still_refuses_active_output_reads_and_writes(body: str) -> None:
    caller = "check() { " + body + "; }\n( check /tmp ) > /tmp/log\n"
    assert not _preserves(caller, _LIBRARY)


def test_child_completion_still_checks_callback_writes_and_nested_redirects() -> None:
    caller = "check() { :; }\n( check /tmp ) > /tmp/log\n"

    def callback(command):
        writes = ("/tmp/log",) if command.argv[0] == "check" and command.child_inspected else ()
        return ShellEffectResult(command.stdout, True, writes)

    assert not _preserves(caller, _LIBRARY, native_effect=callback)
    assert not _preserves(caller.replace("check /tmp )", "check /tmp > /tmp/log )"), _LIBRARY)


@pytest.mark.parametrize(
    "consumer", ["IFS='|' read -r first second", "read -r value", "mapfile -t values"]
)
def test_bounded_process_input_audits_benign_producer_without_inventing_values(
    consumer: str,
) -> None:
    assert _preserves(consumer + " < <(printf ready)\n", _LIBRARY)
    assert not _preserves(consumer + " < <(printf changed > scripts/library.sh)\n", _LIBRARY)


def test_process_input_assignments_do_not_leak_back_into_parent() -> None:
    caller = "kept=original\nread -r value < <(kept=changed; printf ready)\n"
    caller += 'if [[ "$kept" == original ]]; then :; else printf changed > scripts/library.sh; fi\n'
    assert _preserves(caller, _LIBRARY)
    assert not _preserves(
        'target=/tmp/safe\nread -r target < <(printf ready)\nprintf changed > "$target"\n', _LIBRARY
    )


def test_process_input_refuses_file_writes_even_with_explicit_errexit() -> None:
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult(command.stdout, True, ("/tmp/generated",))

    assert not _preserves(
        "mapfile -t lines < <(set -e; printf generated > /tmp/generated)\n",
        _LIBRARY,
        native_effect=callback,
    )
    assert not requests


def test_process_input_cannot_prune_file_branch_before_parent_creates_marker() -> None:
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult(command.stdout, False if command.argv[0] == "test" else True)

    caller = "read -r value < <(printf ready; if [[ -e /tmp/marker ]]; then printf changed > scripts/library.sh; fi)\n"
    caller += "printf created > /tmp/marker\n"
    assert not _preserves(caller, _LIBRARY, native_effect=callback)
    assert not requests


@pytest.mark.parametrize(
    "read",
    [
        "source scripts/helper.sh",
        ". scripts/helper.sh",
        "bash scripts/helper.sh",
        "scripts/helper.sh",
        "cat scripts/helper.sh",
        "command -v helper",
    ],
)
def test_process_input_refuses_source_and_command_identity_snapshot_races(read: str) -> None:
    caller = "read -r value < <(printf ready; " + read + ")\nprintf changed > scripts/helper.sh\n"
    assert not _preserves(caller, _LIBRARY, read_source={"scripts/helper.sh": ":\n"}.get)


def test_process_input_can_use_copied_pure_function_without_artifact_callbacks() -> None:
    caller = 'emit() { if [[ "$1" == ready ]]; then printf "%s" "$1"; fi; }\nread -r value < <(emit ready)\n'

    def callback(command):
        pytest.fail("pure process producer must not consult mutable artifact state")

    assert _preserves(caller, _LIBRARY, native_effect=callback)


def test_process_input_async_restrictions_follow_function_and_substitution_copies() -> None:
    caller = 'emit() { value="$(cat scripts/helper.sh)"; }\nread -r value < <(emit)\n'
    assert not _preserves(caller, _LIBRARY, read_source={"scripts/helper.sh": ":\n"}.get)


@pytest.mark.parametrize("caller", ["printf ready &\n", "printf ready | :\n"])
def test_background_and_pipeline_allow_only_proved_pure_effects(caller: str) -> None:
    assert _preserves(caller, _LIBRARY)


@pytest.mark.parametrize("prefix", ["", "/bin/", "/usr/bin/"])
@pytest.mark.parametrize(
    "command", ["cat", "df -B1 --output=avail /tmp", "awk 'NR == 2 {print $1}'"]
)
def test_concurrent_stable_native_stream_readers_do_not_consult_artifact_state(
    prefix: str, command: str
) -> None:
    def callback(request):
        pytest.fail("stable native stream readers must not inspect mutable artifact identity")

    assert _preserves("printf input | " + prefix + command + "\n", _LIBRARY, native_effect=callback)


def test_concurrent_disk_metadata_does_not_read_active_log_contents() -> None:
    caller = "( df -B1 --output=avail /tmp | awk 'NR == 2 {print $1}' ) > /tmp/log\n"
    assert _preserves(caller, _LIBRARY)


@pytest.mark.parametrize(
    "command", ["cat", "df -B1 --output=avail /tmp", "awk 'NR == 2 {print $1}'"]
)
def test_concurrent_stream_reader_requires_stable_native_path(command: str) -> None:
    prefix = 'PATH="/tmp/replacement:$PATH" '
    assert not _preserves("printf input | " + prefix + command + "\n", _LIBRARY)
    assert _preserves("printf input | " + prefix + "/bin/" + command + "\n", _LIBRARY)


@pytest.mark.parametrize(
    "command",
    [
        "cat scripts/library.sh",
        "/tmp/cat",
        'df -B1 --output=avail "$missing"',
        "awk 'NR == 2 {print $1}' scripts/library.sh",
        "awk 'NR == 2 {system(\"touch scripts/library.sh\")}'",
        "awk 'NR == 2 {print $1 > \"scripts/library.sh\"}'",
        "awk 'NR == 2 {print $1} END {print 1}'",
        "awk -f scripts/library.sh",
    ],
)
def test_concurrent_reader_exception_refuses_file_operands_and_awk_effects(command: str) -> None:
    assert not _preserves("printf input | " + command + "\n", _LIBRARY)


@pytest.mark.parametrize(
    "name, args", [("cat", ""), ("df", "-B1 --output=avail /tmp"), ("awk", "'NR == 2 {print $1}'")]
)
def test_concurrent_native_reader_name_does_not_bypass_function_body(name: str, args: str) -> None:
    caller = (
        name
        + "() { printf changed > scripts/library.sh; }\nprintf input | "
        + name
        + " "
        + args
        + "\n"
    )
    assert not _preserves(caller, _LIBRARY)


@pytest.mark.parametrize("suffix", [" &\n", " | printf ready\n"])
@pytest.mark.parametrize(
    "body",
    [
        "if [[ -e /tmp/marker ]]; then printf changed > scripts/library.sh; fi",
        "source scripts/helper.sh",
        "printf changed > /tmp/generated",
    ],
)
def test_background_and_pipeline_cannot_reuse_mutable_file_snapshots(
    body: str, suffix: str
) -> None:
    caller = "( " + body + " )" + suffix + "printf changed > scripts/helper.sh\n"
    assert not _preserves(caller, _LIBRARY, read_source={"scripts/helper.sh": ":\n"}.get)


@pytest.mark.parametrize(
    "producer",
    [
        'printf changed > "$unknown"',
        "trap 'printf changed > scripts/library.sh' EXIT; printf ready",
        "cat /tmp/log",
        "printf changed > /tmp/log",
    ],
)
def test_process_input_keeps_unknown_path_trap_and_active_output_guards(producer: str) -> None:
    assert not _preserves("( read -r value < <(" + producer + ") ) > /tmp/log\n", _LIBRARY)


def test_process_input_does_not_inherit_parent_exit_trap() -> None:
    caller = "trap 'printf changed > scripts/library.sh' EXIT\nread -r value < <(printf ready)\n"
    assert _preserves(caller, _LIBRARY)


@pytest.mark.parametrize(
    "caller",
    [
        "read -d x value < <(printf ready)\n",
        "mapfile -n 1 values < <(printf ready)\n",
        "read() { :; }\nread -r value < <(printf ready)\n",
        'IFS="$unknown" read -r value < <(printf ready)\n',
        'IFS=" " read -r value < <(printf ready)\n',
        "read -r PATH < <(printf ready)\n",
        "read -r value < < (printf ready)\n",
        "read -r value < <(printf ready) > /tmp/output\n",
        "read -r value < <(printf ready) < <(printf other)\n",
    ],
)
def test_unsupported_process_input_forms_refuse(caller: str) -> None:
    assert not _preserves(caller, _LIBRARY)


@pytest.mark.parametrize("consumer", ["read -r value", "mapfile -t values"])
@pytest.mark.parametrize(
    "branch",
    [
        "then printf changed > scripts/library.sh; else :",
        "then :; else printf changed > scripts/library.sh",
    ],
)
def test_process_input_never_uses_producer_status_as_consumer_success(
    consumer: str, branch: str
) -> None:
    assert not _preserves("if " + consumer + " < <(false); " + branch + "; fi\n", _LIBRARY)


@pytest.mark.parametrize("operator", ["#", "##", "%", "%%"])
def test_uncalled_quoted_pattern_removal_keeps_its_function_boundary(operator: str) -> None:
    library = 'unused() {\nprintf "%s\\n" "${path' + operator + '"$ROOT"/}"\n}\n'
    assert _preserves("source scripts/library.sh\n", library)
    assert not _preserves(
        "source scripts/library.sh\n", library + "printf changed > scripts/library.sh\n"
    )
    assert not _preserves("source scripts/library.sh\nunused\n", library)


def test_sourced_main_guard_without_else_returns_success() -> None:
    assert not _preserves(
        "if source scripts/library.sh; then printf changed > scripts/library.sh; fi\n", _LIBRARY
    )


@pytest.mark.parametrize(
    "statement",
    [
        "if false; then :; fi",
        "false; case value in other) : ;; esac",
        "false; for item in; do :; done",
        "false; while false; do :; done",
        "false; until true; do :; done",
    ],
)
def test_empty_compound_commands_do_not_retain_false_condition_status(statement: str) -> None:
    caller = statement + '\nif [[ "$?" == 0 ]]; then printf changed > scripts/library.sh; fi\n'
    assert not _preserves(caller, _LIBRARY)


@pytest.mark.parametrize("opener, closer", [("$(", ")"), ('"$(', ')"')])
def test_first_assignment_substitution_audits_only_prior_effects(opener: str, closer: str) -> None:
    caller = (
        "source scripts/library.sh\nhelper="
        + opener
        + "sed -n 'p' scripts/library.sh"
        + closer
        + "\nmain\n"
    )
    assert _preserves(caller, _LIBRARY, use_offset=caller.index("sed -n"))


@pytest.mark.parametrize(
    "prefix, closer",
    [
        ("source scripts/library.sh\nmain\nhelper=$(", ")"),
        ('source scripts/library.sh\nhelper="$(printf changed > scripts/library.sh; ', ')"'),
        ('source scripts/library.sh\nhelper="$(main)$( ', ')"'),
    ],
)
def test_earlier_writes_or_executable_word_pieces_cannot_disappear(
    prefix: str, closer: str
) -> None:
    caller = prefix + "sed -n 'p' scripts/library.sh" + closer + "\n"
    assert not _preserves(caller, _LIBRARY, use_offset=caller.index("sed -n"))


def test_collection_has_no_fake_unchanged_source_target() -> None:
    caller = "printf changed > scripts/library.sh\n"
    resolver = _resolver(caller, _LIBRARY)

    def callback(command):
        return ShellEffectResult(command.stdout, True, ("scripts/library.sh",))

    assert source_effects_collect(
        resolver, len(caller), read_source=resolver.context.sources.get, native_effect=callback
    )
    assert not _preserves(caller, _LIBRARY, native_effect=callback)


@pytest.mark.parametrize(
    "call",
    [
        "write_file scripts/library.sh",
        'target=scripts/library.sh; write_file "$target"',
        'wrapper() { write_file "$@"; }; wrapper scripts/library.sh',
        "write_file() { printf changed > scripts/library.sh; }; write_file harmless",
    ],
)
def test_actual_arguments_and_latest_overrides_retain_writes(call: str) -> None:
    assert not _preserves("source scripts/library.sh\n" + call + "\n", _LIBRARY)


@pytest.mark.parametrize(
    "caller",
    [
        "source scripts/library.sh\nwrite_file /tmp/effects-output.txt\n",
        "source scripts/library.sh\nwrite_file() { :; }\nwrite_file scripts/library.sh\n",
        "source scripts/library.sh\nif false; then main; else :; fi\n",
        "source scripts/library.sh\nwrite_file() { return 0; main; }; write_file anything\n",
    ],
)
def test_disjoint_writes_and_reached_control_flow_preserve_source(caller: str) -> None:
    assert _preserves(caller, _LIBRARY)


def test_import_replaces_an_earlier_function_override() -> None:
    assert not _preserves("write_file() { :; }\nsource scripts/library.sh\nmain\n", _LIBRARY)


def test_unknown_condition_audits_both_function_definitions() -> None:
    caller = """source scripts/library.sh
if test -e /runtime/unknown; then write_file() { :; }; fi
write_file scripts/library.sh
"""
    assert not _preserves(caller, _LIBRARY)


@pytest.mark.parametrize(
    "condition",
    [
        'safe == "safe"*',
        'safe == "s"afe',
        'safex != safe"*"',
        r"safex != safe\*",
        "safe == [[:alpha:]]*",
    ],
)
def test_mixed_quoted_or_complex_patterns_cannot_skip_source_mutation(condition: str) -> None:
    caller = f"if [[ {condition} ]]; then printf changed > scripts/library.sh; fi\n"
    assert not _preserves(caller, _LIBRARY)


def test_wholly_quoted_pattern_is_literal_and_simple_unquoted_pattern_stays_supported() -> None:
    assert _preserves(
        'if [[ safe == "safe*" ]]; then printf changed > scripts/library.sh; fi\n', _LIBRARY
    )
    assert not _preserves(
        "if [[ safe == safe* ]]; then printf changed > scripts/library.sh; fi\n", _LIBRARY
    )


def test_expanded_extglob_pattern_cannot_skip_source_mutation() -> None:
    assert not _preserves(
        'pattern="@(safe|other)"\nif [[ safe == $pattern ]]; then printf changed > scripts/library.sh; fi\n',
        _LIBRARY,
    )


@pytest.mark.parametrize("condition", ['[[ "$$" == 123 ]]', '[[ "$$" != 123 ]]'])
def test_pid_identity_remains_unknown_in_conditional_effects(condition: str) -> None:
    assert not _preserves(
        f"if {condition}; then printf changed > scripts/library.sh; fi\n", _LIBRARY
    )


@pytest.mark.parametrize("mode", ["", "-d "])
@pytest.mark.parametrize(
    "condition",
    [
        '[[ "$tmp" == /tmp/example.ABCDEF ]]',
        '[[ /tmp/example.ABCDEF == "$tmp" ]]',
        '[ "$tmp" = /tmp/example.ABCDEF ]',
        '[[ "$tmp" == /tmp/example.A* ]]',
        '[[ "$tmp" != /tmp/example.ABCDEF ]]',
    ],
)
def test_fresh_runtime_name_cannot_prune_conditional_writes(mode: str, condition: str) -> None:
    prefix = 'set -e\ntmp="$(mktemp ' + mode + '/tmp/example.XXXXXX)"\n'
    assert not _preserves(
        prefix + f"if {condition}; then printf changed > scripts/library.sh; fi\n", _LIBRARY
    )
    assert not _preserves(
        prefix + f"if {condition}; then :; else printf changed > scripts/library.sh; fi\n", _LIBRARY
    )
    assert _preserves(
        prefix + f"if {condition}; then printf safe; else printf other; fi\n", _LIBRARY
    )


@pytest.mark.parametrize("mode", ["", "-d "])
def test_fresh_runtime_name_cannot_prune_case_writes(mode: str) -> None:
    prefix = 'set -e\ntmp="$(mktemp ' + mode + '/tmp/example.XXXXXX)"\n'
    assert not _preserves(
        prefix + 'case "$tmp" in /tmp/example.A*) printf changed > scripts/library.sh ;; esac\n',
        _LIBRARY,
    )
    # The shared case grammar still refuses slash-containing pattern labels.
    assert not _preserves(
        prefix + 'case "$tmp" in /tmp/example.A*) printf safe ;; esac\n', _LIBRARY
    )
    assert not _preserves(
        prefix + 'case "$tmp" in *) printf changed > scripts/library.sh ;; esac\n', _LIBRARY
    )
    assert _preserves(prefix + 'case "$tmp" in *) printf safe ;; esac\n', _LIBRARY)


@pytest.mark.parametrize("expansion", ["${tmp:0:6}", "${tmp#/tmp/}", "${#tmp}"])
def test_fresh_runtime_name_transforms_remain_unproved(expansion: str) -> None:
    prefix = 'set -e\ntmp="$(mktemp -d /tmp/example.XXXXXX)"\n'
    assert not _preserves(prefix + 'value="' + expansion + '"\n', _LIBRARY)


def test_pid_case_selection_and_output_are_not_exact_facts() -> None:
    assert not _preserves(
        'case "$$" in 123) printf changed > scripts/library.sh ;; esac\n', _LIBRARY
    )
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult(command.stdout, True, native_identity=True)

    assert _preserves(
        'pid="$(printf "%s" "$$")"\nprintf "%s" "$pid"\n', _LIBRARY, native_effect=callback
    )
    assert requests[-1].stdout is None


def test_pid_marker_cannot_be_injected_as_uploaded_literal() -> None:
    assert not _preserves(f'printf "%s" "{SHELL_PID_MARKER}"\n', _LIBRARY)
    resolver = _resolver("printf safe\n", _LIBRARY)
    resolver.context = ShellSourceContext(
        resolver.context.path,
        {**resolver.context.sources, "scripts/" + SHELL_PID_MARKER: "printf fake\n"},
    )
    assert not source_effects_preserve(resolver, "scripts/library.sh", len(resolver.content))


def test_pid_redirect_retains_a_domain_and_conflicting_numeric_reads_refuse() -> None:
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult(command.stdout, True, native_identity=True)

    assert _preserves('(printf ready) > "/tmp/log.$$"\n', _LIBRARY, native_effect=callback)
    assert any(
        command.redirects == (("1>", "/tmp/log." + SHELL_PID_MARKER),) for command in requests
    )
    assert not _preserves('(cat /tmp/log.123) > "/tmp/log.$$"\n', _LIBRARY, native_effect=callback)


def test_sourced_top_level_write_is_not_treated_as_a_definition() -> None:
    assert not _preserves("source scripts/library.sh\n", "printf changed > scripts/library.sh\n")


def test_child_scope_cannot_hide_filesystem_effects() -> None:
    caller = """bash -c 'source scripts/library.sh; write_file scripts/library.sh'
"""
    assert not _preserves(caller, _LIBRARY)


def test_child_does_not_borrow_unexported_parent_binding() -> None:
    assert not _preserves(
        "target=/tmp/harmless\nbash -c 'printf changed > \"$target\"'\n", _LIBRARY
    )


def test_sequential_prefix_bindings_cannot_use_stale_parent_value() -> None:
    caller = """first=/tmp/harmless
first=scripts/library.sh target="$first" bash -c 'printf changed > "$target"'
"""
    assert not _preserves(caller, _LIBRARY)


def test_prefix_bindings_respect_dynamic_local_scope() -> None:
    caller = """wrapper() {
local first=/tmp/harmless
first=scripts/library.sh target="$first" bash -c 'printf changed > "$target"'
}
wrapper
"""
    assert not _preserves(caller, _LIBRARY)


@pytest.mark.parametrize("modifier, initial", [("-", ""), ("+", "present")])
def test_parameter_fallback_cannot_hide_executable_expansion(modifier: str, initial: str) -> None:
    caller = f'x="{initial}"\nignored="${{x:{modifier}$(printf changed > scripts/library.sh)}}"\n'
    assert not _preserves(caller, _LIBRARY)


def test_failed_status_does_not_invent_exact_exit_code() -> None:
    caller = """failure() { return 2; }
failure
rc="$?"
if [[ "$rc" -eq 2 ]]; then printf changed > scripts/library.sh; fi
"""
    assert not _preserves(caller, _LIBRARY)


def test_unknown_cd_failure_cannot_change_the_assumed_write_directory() -> None:
    assert not _preserves("cd /tmp/may-not-exist\nrm -rf .\n", _LIBRARY)


@pytest.mark.parametrize(
    "caller",
    [
        "false\nif plain=value; then printf changed > scripts/library.sh; fi\n",
        'if value="$(false)"; then :; else printf changed > scripts/library.sh; fi\n',
        "printf value > /tmp/may-fail || printf changed > scripts/library.sh\n",
    ],
)
def test_assignment_and_io_status_cannot_skip_reachable_writes(caller: str) -> None:
    assert not _preserves(caller, _LIBRARY)


@pytest.mark.parametrize(
    "statement, expected",
    [
        ("set -e\nprintf ready\n", True),
        ("set -e\nprintf ready || :\n", False),
        ("set -e\nif printf ready; then :; fi\n", False),
        ("set -e\nf() { printf ready; }; if f; then :; fi\n", False),
        ("set -e\nset +e\nprintf ready\n", False),
    ],
)
def test_fail_stop_context_reaches_native_artifact_callback(statement: str, expected: bool) -> None:
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult(command.stdout, True, native_identity=not command.child_inspected)

    assert _preserves(statement, _LIBRARY, native_effect=callback)
    printf = [command for command in requests if command.argv[0] == "printf"]
    assert printf and all(command.fail_stop is expected for command in printf)


@pytest.mark.parametrize(
    "statement, expected",
    [
        ('set -e\ntmp="$(mktemp -d /tmp/example.XXXXXX)"\n', True),
        ('set -e\ntmp="$(mktemp -d /tmp/example.XXXXXX; true)"\n', False),
        ('set -e\ntmp="$(mktemp -d /tmp/example.XXXXXX)" true\n', False),
    ],
)
def test_only_final_native_mktemp_borrows_assignment_failure_guard(
    statement: str, expected: bool
) -> None:
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult(
            "/tmp/example\n" if command.argv[0] == "mktemp" else "", True, native_identity=True
        )

    assert _preserves(statement, _LIBRARY, native_effect=callback)
    mktemp = [command for command in requests if command.argv[0] == "mktemp"]
    assert mktemp and all(command.fail_stop is expected for command in mktemp)


def test_unknown_parent_directory_computation_does_not_invent_file_contents() -> None:
    assert _preserves('set -e\nrepo="$(cd ./.. && pwd)"\nprintf ready\n', _LIBRARY)
    assert not _preserves('set -e\nrepo="$(cd ./.. && pwd)"\nsource "$repo/library.sh"\n', _LIBRARY)


def test_unknown_directory_success_cannot_hide_later_relative_writes() -> None:
    assert not _preserves("cd /runtime/unknown\nprintf changed > scripts/library.sh\n", _LIBRARY)


def test_checked_known_directory_does_not_skip_possible_failure_branch() -> None:
    assert not _preserves(
        "if cd scripts; then :; else printf changed > scripts/library.sh; fi\n", _LIBRARY
    )


def test_dirname_option_cannot_invent_success_and_skip_mutation() -> None:
    assert not _preserves(
        "if dirname -z; then :; else printf changed > scripts/library.sh; fi\n", _LIBRARY
    )


@pytest.mark.parametrize(
    "statement, expected",
    [
        ('set -e\nroot="$(cd scripts && pwd)"\n', True),
        ('set -e\nroot="$(cd scripts && pwd; true)"\n', False),
        ('set -e\nroot="$(cd scripts && pwd)" true\n', False),
        ('root="$(cd scripts && pwd)"\n', False),
    ],
)
def test_only_final_native_path_probe_requires_success(statement: str, expected: bool) -> None:
    requests = []

    def callback(command):
        requests.append(command)
        output = None if command.cwd is None else command.cwd + "\n"
        return ShellEffectResult(
            output if command.argv[0] == "pwd" else "", True, native_identity=True
        )

    assert _preserves(statement, _LIBRARY, native_effect=callback)
    cd = [command for command in requests if command.argv[0] == "cd"]
    assert cd and all(command.fail_stop is expected for command in cd)


def test_local_argument_binding_restores_parent_value() -> None:
    caller = """source scripts/library.sh
target=/tmp/parent-output
write_file /tmp/child-output
printf safe > "$target"
"""
    assert _preserves(caller, _LIBRARY)


@pytest.mark.parametrize(
    "statement",
    [
        "[[ x == x ]] > scripts/library.sh",
        "printf safe > scripts/library.sh",
        '(trap "printf changed > scripts/library.sh" EXIT; :)',
        'result="$(trap "printf changed > scripts/library.sh" EXIT; :)"',
        "cp --target-directory=scripts/library.sh harmless",
    ],
)
def test_redirects_traps_and_writer_options_cannot_escape_effect_audit(statement: str) -> None:
    assert not _preserves(statement + "\n", _LIBRARY)


def test_callback_refusal_is_authoritative() -> None:
    assert not _preserves("cat harmless\n", _LIBRARY, native_effect=lambda command: None)


def test_unknown_branches_are_marked_for_artifact_callback() -> None:
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult(status=None if command.argv[0] == "test" else True)

    assert _preserves(
        "if test -e /runtime/unknown; then echo first; else echo second; fi\n",
        _LIBRARY,
        native_effect=callback,
    )
    branches = [command for command in requests if command.argv[0] == "echo"]
    assert len(branches) == 2
    assert all(command.uncertain for command in branches)


@pytest.mark.parametrize("caller", ["echo background &\n", "echo producer | cat\n"])
def test_concurrent_effects_cannot_publish_ordered_artifact_facts(caller: str) -> None:
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult(status=True)

    assert _preserves(caller, _LIBRARY, native_effect=callback)
    # Pure concurrent commands now bypass mutable artifact callbacks entirely.
    assert not requests


def test_function_redirect_opens_in_caller_directory_before_body() -> None:
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult(command.stdout, True, native_identity=True)

    caller = "emit() { cd /tmp/other; printf code; }; emit > /tmp/result\n"
    assert _preserves(caller, _LIBRARY, native_effect=callback)
    opening = next(command for command in requests if command.redirect_open)
    completion = [
        command for command in requests if command.child_inspected and not command.redirect_open
    ][-1]
    assert opening.cwd == completion.cwd == "."
    assert requests.index(opening) < next(
        index for index, command in enumerate(requests) if command.argv[0] == "cd"
    )


@pytest.mark.parametrize("opening, closing", [("(", ")"), ("{", "; }")])
def test_compound_output_redirect_preserves_body_effects(opening: str, closing: str) -> None:
    assert _preserves(f"{opening} printf safe {closing} > /tmp/log 2>&1\n", _LIBRARY)
    assert not _preserves(f"{opening} printf safe {closing} > scripts/library.sh 2>&1\n", _LIBRARY)
    assert not _preserves(
        f"{opening} printf changed > scripts/library.sh {closing} > /tmp/log 2>&1\n", _LIBRARY
    )


def test_subshell_redirect_opens_before_body_and_retains_caller_directory() -> None:
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult(command.stdout, True, native_identity=True)

    assert _preserves(
        "set -e\n(cd scripts; printf safe) > /tmp/log 2>&1\n", _LIBRARY, native_effect=callback
    )
    opening = next(command for command in requests if command.redirect_open)
    completion = [
        command for command in requests if command.child_inspected and not command.redirect_open
    ][-1]
    assert opening.cwd == completion.cwd == "."
    assert requests.index(opening) < next(
        index for index, command in enumerate(requests) if command.argv[0] == "cd"
    )


def test_subshell_cannot_read_its_still_open_destination() -> None:
    assert not _preserves("(printf safe; source /tmp/log) > /tmp/log\n", _LIBRARY)


def test_exit_trap_output_is_not_dropped_from_generated_source_proof() -> None:
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult(
            command.stdout,
            command.child_status if command.child_inspected else True,
            native_identity=True,
        )

    caller = 'set -e\n(trap "printf extra" EXIT; printf body) > /tmp/log\n'
    assert _preserves(caller, _LIBRARY, native_effect=callback)
    completion = [
        command for command in requests if command.child_inspected and not command.redirect_open
    ][-1]
    assert completion.stdout is None and completion.child_status is None


@pytest.mark.parametrize(
    "statement", ['value="$(printf safe)"', "(printf safe)", "printf safe &", "printf safe | cat"]
)
def test_child_shell_does_not_run_parent_exit_trap(statement: str) -> None:
    caller = 'trap "printf changed > scripts/library.sh" EXIT\n' + statement + "\n"
    assert _preserves(caller, _LIBRARY)


def test_pipeline_member_retains_its_own_exit_trap_effects() -> None:
    caller = 'emit() { trap "printf changed > scripts/library.sh" EXIT; }; emit | cat\n'
    assert not _preserves(caller, _LIBRARY)


def test_child_cannot_borrow_bytes_from_its_active_output_file() -> None:
    def callback(command):
        return ShellEffectResult(command.stdout, True, native_identity=True)

    caller = "emit() { printf prefix; cat /tmp/result; }; emit > /tmp/result\n"
    assert not _preserves(caller, _LIBRARY, native_effect=callback)


def test_path_prefix_needs_native_shell_identity() -> None:
    assert not _preserves('PATH="/tmp/replacement:$PATH" bash -c ":"\n', _LIBRARY)


def test_recursive_function_hits_shared_depth_bound() -> None:
    assert not _preserves("again() { again; }; again\n", _LIBRARY)


def test_while_argument_loop_is_finite_and_binds_each_call() -> None:
    library = """write_files() {
while (($#)); do printf changed > "$1"; shift; done
}
"""
    assert _preserves("source scripts/library.sh\nwrite_files /tmp/a /tmp/b\n", library)
    assert not _preserves(
        "source scripts/library.sh\nwrite_files /tmp/a scripts/library.sh\n", library
    )


def test_quoted_heredoc_in_uncalled_function_is_literal_data() -> None:
    library = """unused() {
cat <<'TEXT'
} $(printf changed > scripts/library.sh)
TEXT
}
"""
    assert _preserves("source scripts/library.sh\n", library)
