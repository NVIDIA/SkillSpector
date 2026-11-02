# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reached imported functions, not all definitions, own source-write effects."""

import pytest

from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.shell_source_effects import SHELL_PID_MARKER, ShellEffectResult, source_effects_collect, source_effects_preserve
from skillspector.shell_sources import ShellSourceContext, ShellSourceResolver


def _resolver(caller: str, library: str) -> ShellSourceResolver:
    return ShellSourceResolver(
        ShellSourceContext("scripts/caller.sh", {
            "scripts/caller.sh": caller,
            "scripts/library.sh": library,
        }),
        tm._next_shell_invocation_word,
        lambda text, start, end: tm._skip_command_substitution(text, start, end, lambda: None),
        tm._static_heredoc_header,
        tm._skip_static_heredoc_bodies,
        lambda: None,
    )


def _preserves(caller: str, library: str, *, use_offset: int | None = None, **kwargs) -> bool:
    resolver = _resolver(caller, library)
    return source_effects_preserve(resolver, "scripts/library.sh", len(caller) if use_offset is None else use_offset, **kwargs)


_LIBRARY = '''set -euo pipefail
write_file() { local target="$1"; printf '%s' changed > "$target"; }
main() { write_file scripts/library.sh; }
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
'''


def test_import_installs_definitions_without_running_main_or_uncalled_writer() -> None:
    assert _preserves('source scripts/library.sh\n', _LIBRARY)


@pytest.mark.parametrize("diagnostic", ["--board-config requires a value", "$(printf changed > scripts/library.sh)"])
def test_present_positional_parameter_skips_its_unused_diagnostic(diagnostic: str) -> None:
    caller = 'parse() { value="${2:?' + diagnostic + '}"; }\nparse --board-config known\n'
    assert _preserves(caller, _LIBRARY)
    assert not _preserves(caller + 'printf changed > scripts/library.sh\n', _LIBRARY)


@pytest.mark.parametrize("arguments", ["--board-config", '--board-config ""', '--board-config "$unknown"'])
def test_missing_or_unknown_positional_diagnostic_cannot_hide_effects(arguments: str) -> None:
    caller = 'parse() { value="${2:?$(printf changed > scripts/library.sh)}"; }\nparse ' + arguments + '\n'
    assert not _preserves(caller, _LIBRARY)


@pytest.mark.parametrize("initial, modifier", [('value=known', '-'), ('value=known', '?'), ('value=""', '+')])
def test_known_parameter_value_skips_only_unused_fallback(initial: str, modifier: str) -> None:
    caller = initial + '\nresult="${value:' + modifier + '$(printf changed > scripts/library.sh)}"\n'
    assert _preserves(caller, _LIBRARY)


@pytest.mark.parametrize("zero", ["0", "00", "000:?missing"])
def test_zero_parameter_never_selects_the_last_function_argument(zero: str) -> None:
    condition = '[[ "${' + zero + '}" == "$0" ]]'
    safe = 'check() { if ' + condition + '; then :; else printf changed > scripts/library.sh; fi; }\ncheck other\n'
    unsafe = 'check() { if ' + condition + '; then printf changed > scripts/library.sh; fi; }\ncheck other\n'
    assert _preserves(safe, _LIBRARY)
    assert not _preserves(unsafe, _LIBRARY)


@pytest.mark.parametrize("name", ["0" * 17, "١"])
def test_unproved_numeric_parameter_spelling_refuses(name: str) -> None:
    assert not _preserves('value="${' + name + '}"\n', _LIBRARY)


@pytest.mark.parametrize("condition, outcome", [
    ("! x == x || y == y", True),
    ("! x == x && y == y", False),
    ("! x == y || y == x", True),
    ("! x == y && y == x", False),
    ("x == y && y == x || x == x", True),
    ("! ! x == x && y == y", True),
])
def test_conditional_negation_binds_before_logical_operators(condition: str, outcome: bool) -> None:
    caller = 'if [[ ' + condition + ' ]]; then printf changed > scripts/library.sh; fi\n'
    opposite = 'if [[ ' + condition + ' ]]; then :; else printf changed > scripts/library.sh; fi\n'
    assert _preserves(caller, _LIBRARY) is not outcome
    assert _preserves(opposite, _LIBRARY) is outcome


@pytest.mark.parametrize("condition", ["( x == x || y == y )", "! ( x == y )", "! " * 18 + "x == x"])
def test_unsupported_condition_grouping_or_depth_refuses(condition: str) -> None:
    assert not _preserves('if [[ ' + condition + ' ]]; then :; fi\n', _LIBRARY)


@pytest.mark.parametrize("left, operator", [("x == x", "||"), ("x == y", "&&")])
def test_skipped_conditional_operand_never_runs_substitution_effects(left: str, operator: str) -> None:
    caller = 'if [[ ' + left + ' ' + operator + ' "$(set -e; printf changed > scripts/library.sh)" ]]; then :; fi\n'
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


@pytest.mark.parametrize("index, preserved", [("8", False), ("10", True), ("010", False), ("08", False)])
def test_array_indices_never_misread_bash_octal_as_decimal(index: str, preserved: bool) -> None:
    items = [f"/tmp/output{number}" for number in range(11)]
    items[8] = "scripts/library.sh"
    caller = 'items=(' + ' '.join(items) + ')\nprintf changed > "${items[' + index + ']}"\n'
    assert _preserves(caller, _LIBRARY) is preserved


@pytest.mark.parametrize("comparison", ["010 -eq 8", "8 -eq 010", "08 -eq 8"])
def test_nondecimal_conditional_numbers_cannot_skip_source_writes(comparison: str) -> None:
    assert not _preserves('if [[ ' + comparison + ' ]]; then printf changed > scripts/library.sh; fi\n', _LIBRARY)


@pytest.mark.parametrize("number, preserved", [("8", True), ("010", False), ("08", False)])
def test_arithmetic_condition_requires_canonical_decimal(number: str, preserved: bool) -> None:
    caller = 'if (( ' + number + ' )); then :; else printf changed > scripts/library.sh; fi\n'
    assert _preserves(caller, _LIBRARY) is preserved


def test_array_element_length_never_becomes_element_contents() -> None:
    caller = 'items=(safe)\nif [[ "${#items[0]}" == safe ]]; then :; else printf changed > scripts/library.sh; fi\n'
    assert not _preserves(caller, _LIBRARY)
    assert _preserves('items=(safe)\nif [[ "${#items[@]}" == 1 ]]; then :; else printf changed > scripts/library.sh; fi\n', _LIBRARY)


@pytest.mark.parametrize("operand, arguments", [
    ('"$@x"', 'scripts/library.sh /tmp/other'),
    ('"prefix$@"', '/tmp/other scripts/library.sh'),
    ('"${@}x"', 'scripts/library.sh /tmp/other'),
])
def test_embedded_positional_list_never_flattens_distinct_write_paths(operand: str, arguments: str) -> None:
    caller = 'remove() { rm ' + operand + '; }\nremove ' + arguments + '\n'
    assert not _preserves(caller, _LIBRARY)


def test_exact_positional_list_still_preserves_each_argument() -> None:
    caller = 'remove() { rm "$@"; }\n'
    assert _preserves(caller + 'remove /tmp/one /tmp/two\n', _LIBRARY)
    assert not _preserves(caller + 'remove scripts/library.sh /tmp/two\n', _LIBRARY)


@pytest.mark.parametrize("opening, closing", [("[", " ]"), ("test", "")])
def test_ordinary_test_refuses_unproved_splitting_and_globbing(opening: str, closing: str) -> None:
    caller = "value='x y'\nif " + opening + ' $value != z' + closing + '; then :; else printf changed > scripts/library.sh; fi\n'
    assert not _preserves(caller, _LIBRARY)
    assert _preserves(caller.replace('$value', '"$value"'), _LIBRARY)
    pattern = 'if ' + opening + ' safe == s*' + closing + '; then :; else printf changed > scripts/library.sh; fi\n'
    assert not _preserves(pattern, _LIBRARY)


def test_extended_condition_keeps_its_distinct_no_splitting_and_pattern_rules() -> None:
    assert _preserves("value='x y'\nif [[ $value != z ]]; then :; else printf changed > scripts/library.sh; fi\n", _LIBRARY)
    assert _preserves('if [[ safe == s* ]]; then :; else printf changed > scripts/library.sh; fi\n', _LIBRARY)


def test_ordinary_test_substitution_output_cannot_skip_field_splitting() -> None:
    value = "$(printf '%s' 'x y')"
    caller = 'if [ ' + value + ' != z ]; then :; else printf changed > scripts/library.sh; fi\n'
    assert not _preserves(caller, _LIBRARY)
    assert _preserves(caller.replace(value, '"' + value + '"'), _LIBRARY)
    assert not _preserves("rm $(printf '%s' 'scripts/library.sh /tmp/other')\n", _LIBRARY)


def test_ordinary_test_brace_expansion_cannot_skip_argument_count_failure() -> None:
    caller = 'if [ x != {x,y} ]; then :; else printf changed > scripts/library.sh; fi\n'
    assert not _preserves(caller, _LIBRARY)
    assert _preserves(caller.replace('{x,y}', "'{x,y}'"), _LIBRARY)


@pytest.mark.parametrize("prefix, operand", [('value=~\n', '"$value"'), ('', '~')])
def test_unquoted_tilde_never_becomes_literal_in_assignment_or_condition(prefix: str, operand: str) -> None:
    caller = prefix + 'if [[ ' + operand + ' == "~" ]]; then :; else printf changed > scripts/library.sh; fi\n'
    assert not _preserves(caller, _LIBRARY)
    quoted = prefix.replace('=~', "='~'") + 'if [[ ' + ("'~'" if operand == '~' else operand) + ' == "~" ]]; then :; else printf changed > scripts/library.sh; fi\n'
    assert _preserves(quoted, _LIBRARY)


@pytest.mark.parametrize("value", ['end$', '"end$"'])
def test_terminal_dollar_is_literal_without_a_parameter_name(value: str) -> None:
    caller = 'value=' + value + '\nif [[ "$value" == \'end$\' ]]; then :; else printf changed > scripts/library.sh; fi\n'
    assert _preserves(caller, _LIBRARY)


def test_regex_anchor_stays_literal_without_claiming_regex_truth() -> None:
    condition = '[[ "$BRANCH" =~ ^[A-Za-z0-9._-]+$ ]]'
    assert _preserves('BRANCH=dev\n' + condition + '\n', _LIBRARY)
    assert not _preserves('BRANCH=dev\nif ' + condition + '; then printf changed > scripts/library.sh; fi\n', _LIBRARY)


@pytest.mark.parametrize("left, right", [
    ('"$(printf changed > scripts/library.sh)"', 'literal'),
    ('literal', '"$(printf changed > scripts/library.sh)"'),
])
def test_regex_operands_still_audit_executable_substitutions(left: str, right: str) -> None:
    assert not _preserves('[[ ' + left + ' =~ ' + right + ' ]]\n', _LIBRARY)


@pytest.mark.parametrize("value", ['$missing', '"$missing"', '$[1+1]'])
def test_real_or_unsupported_dollar_expansion_does_not_become_literal(value: str) -> None:
    assert not _preserves('value=' + value + '\nprintf changed > "$value"\n', _LIBRARY)


@pytest.mark.parametrize("operator", ["#", "##", "%", "%%"])
def test_uncalled_quoted_pattern_removal_keeps_its_function_boundary(operator: str) -> None:
    library = 'unused() {\nprintf "%s\\n" "${path' + operator + '"$ROOT"/}"\n}\n'
    assert _preserves('source scripts/library.sh\n', library)
    assert not _preserves('source scripts/library.sh\n', library + 'printf changed > scripts/library.sh\n')
    assert not _preserves('source scripts/library.sh\nunused\n', library)


def test_sourced_main_guard_without_else_returns_success() -> None:
    assert not _preserves('if source scripts/library.sh; then printf changed > scripts/library.sh; fi\n', _LIBRARY)


@pytest.mark.parametrize("statement", [
    "if false; then :; fi",
    "false; case value in other) : ;; esac",
    "false; for item in; do :; done",
    "false; while false; do :; done",
    "false; until true; do :; done",
])
def test_empty_compound_commands_do_not_retain_false_condition_status(statement: str) -> None:
    caller = statement + '\nif [[ "$?" == 0 ]]; then printf changed > scripts/library.sh; fi\n'
    assert not _preserves(caller, _LIBRARY)


@pytest.mark.parametrize("opener, closer", [("$(", ")"), ('"$(', ')"')])
def test_first_assignment_substitution_audits_only_prior_effects(opener: str, closer: str) -> None:
    caller = 'source scripts/library.sh\nhelper=' + opener + "sed -n 'p' scripts/library.sh" + closer + '\nmain\n'
    assert _preserves(caller, _LIBRARY, use_offset=caller.index("sed -n"))


@pytest.mark.parametrize("prefix, closer", [
    ("source scripts/library.sh\nmain\nhelper=$(", ")"),
    ('source scripts/library.sh\nhelper="$(printf changed > scripts/library.sh; ', ')"'),
    ('source scripts/library.sh\nhelper="$(main)$( ', ')"'),
])
def test_earlier_writes_or_executable_word_pieces_cannot_disappear(prefix: str, closer: str) -> None:
    caller = prefix + "sed -n 'p' scripts/library.sh" + closer + "\n"
    assert not _preserves(caller, _LIBRARY, use_offset=caller.index("sed -n"))


def test_collection_has_no_fake_unchanged_source_target() -> None:
    caller = 'printf changed > scripts/library.sh\n'
    resolver = _resolver(caller, _LIBRARY)

    def callback(command):
        return ShellEffectResult(command.stdout, True, ("scripts/library.sh",))

    assert source_effects_collect(resolver, len(caller), read_source=resolver.context.sources.get, native_effect=callback)
    assert not _preserves(caller, _LIBRARY, native_effect=callback)


@pytest.mark.parametrize("call", [
    "write_file scripts/library.sh",
    'target=scripts/library.sh; write_file "$target"',
    'wrapper() { write_file "$@"; }; wrapper scripts/library.sh',
    'write_file() { printf changed > scripts/library.sh; }; write_file harmless',
])
def test_actual_arguments_and_latest_overrides_retain_writes(call: str) -> None:
    assert not _preserves('source scripts/library.sh\n' + call + '\n', _LIBRARY)


@pytest.mark.parametrize("caller", [
    'source scripts/library.sh\nwrite_file /tmp/effects-output.txt\n',
    'source scripts/library.sh\nwrite_file() { :; }\nwrite_file scripts/library.sh\n',
    'source scripts/library.sh\nif false; then main; else :; fi\n',
    'source scripts/library.sh\nwrite_file() { return 0; main; }; write_file anything\n',
])
def test_disjoint_writes_and_reached_control_flow_preserve_source(caller: str) -> None:
    assert _preserves(caller, _LIBRARY)


def test_import_replaces_an_earlier_function_override() -> None:
    assert not _preserves('write_file() { :; }\nsource scripts/library.sh\nmain\n', _LIBRARY)


def test_unknown_condition_audits_both_function_definitions() -> None:
    caller = '''source scripts/library.sh
if test -e /runtime/unknown; then write_file() { :; }; fi
write_file scripts/library.sh
'''
    assert not _preserves(caller, _LIBRARY)


@pytest.mark.parametrize("condition", ['safe == "safe"*', 'safe == "s"afe', 'safex != safe"*"', r'safex != safe\*', 'safe == [[:alpha:]]*'])
def test_mixed_quoted_or_complex_patterns_cannot_skip_source_mutation(condition: str) -> None:
    caller = f'if [[ {condition} ]]; then printf changed > scripts/library.sh; fi\n'
    assert not _preserves(caller, _LIBRARY)


def test_wholly_quoted_pattern_is_literal_and_simple_unquoted_pattern_stays_supported() -> None:
    assert _preserves('if [[ safe == "safe*" ]]; then printf changed > scripts/library.sh; fi\n', _LIBRARY)
    assert not _preserves('if [[ safe == safe* ]]; then printf changed > scripts/library.sh; fi\n', _LIBRARY)


def test_expanded_extglob_pattern_cannot_skip_source_mutation() -> None:
    assert not _preserves('pattern="@(safe|other)"\nif [[ safe == $pattern ]]; then printf changed > scripts/library.sh; fi\n', _LIBRARY)


@pytest.mark.parametrize("condition", ['[[ "$$" == 123 ]]', '[[ "$$" != 123 ]]'])
def test_pid_identity_remains_unknown_in_conditional_effects(condition: str) -> None:
    assert not _preserves(f'if {condition}; then printf changed > scripts/library.sh; fi\n', _LIBRARY)


def test_pid_case_selection_and_output_are_not_exact_facts() -> None:
    assert not _preserves('case "$$" in 123) printf changed > scripts/library.sh ;; esac\n', _LIBRARY)
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult(command.stdout, True, native_identity=True)

    assert _preserves('pid="$(printf "%s" "$$")"\nprintf "%s" "$pid"\n', _LIBRARY, native_effect=callback)
    assert requests[-1].stdout is None


def test_pid_marker_cannot_be_injected_as_uploaded_literal() -> None:
    assert not _preserves(f'printf "%s" "{SHELL_PID_MARKER}"\n', _LIBRARY)
    resolver = _resolver("printf safe\n", _LIBRARY)
    resolver.context = ShellSourceContext(resolver.context.path, {**resolver.context.sources, "scripts/" + SHELL_PID_MARKER: "printf fake\n"})
    assert not source_effects_preserve(resolver, "scripts/library.sh", len(resolver.content))


def test_pid_redirect_retains_a_domain_and_conflicting_numeric_reads_refuse() -> None:
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult(command.stdout, True, native_identity=True)

    assert _preserves('(printf ready) > "/tmp/log.$$"\n', _LIBRARY, native_effect=callback)
    assert any(command.redirects == (("1>", "/tmp/log." + SHELL_PID_MARKER),) for command in requests)
    assert not _preserves('(cat /tmp/log.123) > "/tmp/log.$$"\n', _LIBRARY, native_effect=callback)


def test_sourced_top_level_write_is_not_treated_as_a_definition() -> None:
    assert not _preserves('source scripts/library.sh\n', 'printf changed > scripts/library.sh\n')


def test_child_scope_cannot_hide_filesystem_effects() -> None:
    caller = '''bash -c 'source scripts/library.sh; write_file scripts/library.sh'
'''
    assert not _preserves(caller, _LIBRARY)


def test_child_does_not_borrow_unexported_parent_binding() -> None:
    assert not _preserves('target=/tmp/harmless\nbash -c \'printf changed > "$target"\'\n', _LIBRARY)


def test_sequential_prefix_bindings_cannot_use_stale_parent_value() -> None:
    caller = '''first=/tmp/harmless
first=scripts/library.sh target="$first" bash -c 'printf changed > "$target"'
'''
    assert not _preserves(caller, _LIBRARY)


def test_prefix_bindings_respect_dynamic_local_scope() -> None:
    caller = '''wrapper() {
local first=/tmp/harmless
first=scripts/library.sh target="$first" bash -c 'printf changed > "$target"'
}
wrapper
'''
    assert not _preserves(caller, _LIBRARY)


@pytest.mark.parametrize("modifier, initial", [("-", ""), ("+", "present")])
def test_parameter_fallback_cannot_hide_executable_expansion(modifier: str, initial: str) -> None:
    caller = f'x="{initial}"\nignored="${{x:{modifier}$(printf changed > scripts/library.sh)}}"\n'
    assert not _preserves(caller, _LIBRARY)


def test_failed_status_does_not_invent_exact_exit_code() -> None:
    caller = '''failure() { return 2; }
failure
rc="$?"
if [[ "$rc" -eq 2 ]]; then printf changed > scripts/library.sh; fi
'''
    assert not _preserves(caller, _LIBRARY)


def test_unknown_cd_failure_cannot_change_the_assumed_write_directory() -> None:
    assert not _preserves('cd /tmp/may-not-exist\nrm -rf .\n', _LIBRARY)


@pytest.mark.parametrize("caller", [
    'false\nif plain=value; then printf changed > scripts/library.sh; fi\n',
    'if value="$(false)"; then :; else printf changed > scripts/library.sh; fi\n',
    'printf value > /tmp/may-fail || printf changed > scripts/library.sh\n',
])
def test_assignment_and_io_status_cannot_skip_reachable_writes(caller: str) -> None:
    assert not _preserves(caller, _LIBRARY)


@pytest.mark.parametrize("statement, expected", [
    ('set -e\nprintf ready\n', True),
    ('set -e\nprintf ready || :\n', False),
    ('set -e\nif printf ready; then :; fi\n', False),
    ('set -e\nf() { printf ready; }; if f; then :; fi\n', False),
    ('set -e\nset +e\nprintf ready\n', False),
])
def test_fail_stop_context_reaches_native_artifact_callback(statement: str, expected: bool) -> None:
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult(command.stdout, True, native_identity=not command.child_inspected)

    assert _preserves(statement, _LIBRARY, native_effect=callback)
    printf = [command for command in requests if command.argv[0] == "printf"]
    assert printf and all(command.fail_stop is expected for command in printf)


@pytest.mark.parametrize("statement, expected", [
    ('set -e\ntmp="$(mktemp -d /tmp/example.XXXXXX)"\n', True),
    ('set -e\ntmp="$(mktemp -d /tmp/example.XXXXXX; true)"\n', False),
    ('set -e\ntmp="$(mktemp -d /tmp/example.XXXXXX)" true\n', False),
])
def test_only_final_native_mktemp_borrows_assignment_failure_guard(statement: str, expected: bool) -> None:
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult("/tmp/example\n" if command.argv[0] == "mktemp" else "", True, native_identity=True)

    assert _preserves(statement, _LIBRARY, native_effect=callback)
    mktemp = [command for command in requests if command.argv[0] == "mktemp"]
    assert mktemp and all(command.fail_stop is expected for command in mktemp)


def test_unknown_parent_directory_computation_does_not_invent_file_contents() -> None:
    assert _preserves('set -e\nrepo="$(cd ./.. && pwd)"\nprintf ready\n', _LIBRARY)
    assert not _preserves('set -e\nrepo="$(cd ./.. && pwd)"\nsource "$repo/library.sh"\n', _LIBRARY)


def test_unknown_directory_success_cannot_hide_later_relative_writes() -> None:
    assert not _preserves('cd /runtime/unknown\nprintf changed > scripts/library.sh\n', _LIBRARY)


def test_checked_known_directory_does_not_skip_possible_failure_branch() -> None:
    assert not _preserves('if cd scripts; then :; else printf changed > scripts/library.sh; fi\n', _LIBRARY)


def test_dirname_option_cannot_invent_success_and_skip_mutation() -> None:
    assert not _preserves('if dirname -z; then :; else printf changed > scripts/library.sh; fi\n', _LIBRARY)


@pytest.mark.parametrize("statement, expected", [
    ('set -e\nroot="$(cd scripts && pwd)"\n', True),
    ('set -e\nroot="$(cd scripts && pwd; true)"\n', False),
    ('set -e\nroot="$(cd scripts && pwd)" true\n', False),
    ('root="$(cd scripts && pwd)"\n', False),
])
def test_only_final_native_path_probe_requires_success(statement: str, expected: bool) -> None:
    requests = []

    def callback(command):
        requests.append(command)
        output = None if command.cwd is None else command.cwd + "\n"
        return ShellEffectResult(output if command.argv[0] == "pwd" else "", True, native_identity=True)

    assert _preserves(statement, _LIBRARY, native_effect=callback)
    cd = [command for command in requests if command.argv[0] == "cd"]
    assert cd and all(command.fail_stop is expected for command in cd)


def test_local_argument_binding_restores_parent_value() -> None:
    caller = '''source scripts/library.sh
target=/tmp/parent-output
write_file /tmp/child-output
printf safe > "$target"
'''
    assert _preserves(caller, _LIBRARY)


@pytest.mark.parametrize("statement", [
    '[[ x == x ]] > scripts/library.sh',
    'printf safe > scripts/library.sh',
    '(trap "printf changed > scripts/library.sh" EXIT; :)',
    'result="$(trap "printf changed > scripts/library.sh" EXIT; :)"',
    'cp --target-directory=scripts/library.sh harmless',
])
def test_redirects_traps_and_writer_options_cannot_escape_effect_audit(statement: str) -> None:
    assert not _preserves(statement + '\n', _LIBRARY)


def test_callback_refusal_is_authoritative() -> None:
    assert not _preserves('cat harmless\n', _LIBRARY, native_effect=lambda command: None)


def test_unknown_branches_are_marked_for_artifact_callback() -> None:
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult(status=None if command.argv[0] == "test" else True)

    assert _preserves('if test -e /runtime/unknown; then echo first; else echo second; fi\n', _LIBRARY, native_effect=callback)
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
    assert requests and all(command.uncertain for command in requests)


def test_function_redirect_opens_in_caller_directory_before_body() -> None:
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult(command.stdout, True, native_identity=True)

    caller = 'emit() { cd /tmp/other; printf code; }; emit > /tmp/result\n'
    assert _preserves(caller, _LIBRARY, native_effect=callback)
    opening = next(command for command in requests if command.redirect_open)
    completion = [command for command in requests if command.child_inspected and not command.redirect_open][-1]
    assert opening.cwd == completion.cwd == "."
    assert requests.index(opening) < next(index for index, command in enumerate(requests) if command.argv[0] == "cd")


@pytest.mark.parametrize("opening, closing", [("(", ")"), ("{", "; }")])
def test_compound_output_redirect_preserves_body_effects(opening: str, closing: str) -> None:
    assert _preserves(f'{opening} printf safe {closing} > /tmp/log 2>&1\n', _LIBRARY)
    assert not _preserves(f'{opening} printf safe {closing} > scripts/library.sh 2>&1\n', _LIBRARY)
    assert not _preserves(f'{opening} printf changed > scripts/library.sh {closing} > /tmp/log 2>&1\n', _LIBRARY)


def test_subshell_redirect_opens_before_body_and_retains_caller_directory() -> None:
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult(command.stdout, True, native_identity=True)

    assert _preserves('set -e\n(cd scripts; printf safe) > /tmp/log 2>&1\n', _LIBRARY, native_effect=callback)
    opening = next(command for command in requests if command.redirect_open)
    completion = [command for command in requests if command.child_inspected and not command.redirect_open][-1]
    assert opening.cwd == completion.cwd == "."
    assert requests.index(opening) < next(index for index, command in enumerate(requests) if command.argv[0] == "cd")


def test_subshell_cannot_read_its_still_open_destination() -> None:
    assert not _preserves('(printf safe; source /tmp/log) > /tmp/log\n', _LIBRARY)


def test_exit_trap_output_is_not_dropped_from_generated_source_proof() -> None:
    requests = []

    def callback(command):
        requests.append(command)
        return ShellEffectResult(command.stdout, command.child_status if command.child_inspected else True, native_identity=True)

    caller = 'set -e\n(trap "printf extra" EXIT; printf body) > /tmp/log\n'
    assert _preserves(caller, _LIBRARY, native_effect=callback)
    completion = [command for command in requests if command.child_inspected and not command.redirect_open][-1]
    assert completion.stdout is None and completion.child_status is None


@pytest.mark.parametrize("statement", ['value="$(printf safe)"', '(printf safe)', 'printf safe &', 'printf safe | cat'])
def test_child_shell_does_not_run_parent_exit_trap(statement: str) -> None:
    caller = 'trap "printf changed > scripts/library.sh" EXIT\n' + statement + '\n'
    assert _preserves(caller, _LIBRARY)


def test_pipeline_member_retains_its_own_exit_trap_effects() -> None:
    caller = 'emit() { trap "printf changed > scripts/library.sh" EXIT; }; emit | cat\n'
    assert not _preserves(caller, _LIBRARY)


def test_child_cannot_borrow_bytes_from_its_active_output_file() -> None:
    def callback(command):
        return ShellEffectResult(command.stdout, True, native_identity=True)

    caller = 'emit() { printf prefix; cat /tmp/result; }; emit > /tmp/result\n'
    assert not _preserves(caller, _LIBRARY, native_effect=callback)


def test_path_prefix_needs_native_shell_identity() -> None:
    assert not _preserves('PATH="/tmp/replacement:$PATH" bash -c ":"\n', _LIBRARY)


def test_recursive_function_hits_shared_depth_bound() -> None:
    assert not _preserves('again() { again; }; again\n', _LIBRARY)


def test_while_argument_loop_is_finite_and_binds_each_call() -> None:
    library = '''write_files() {
while (($#)); do printf changed > "$1"; shift; done
}
'''
    assert _preserves('source scripts/library.sh\nwrite_files /tmp/a /tmp/b\n', library)
    assert not _preserves('source scripts/library.sh\nwrite_files /tmp/a scripts/library.sh\n', library)


def test_quoted_heredoc_in_uncalled_function_is_literal_data() -> None:
    library = '''unused() {
cat <<'TEXT'
} $(printf changed > scripts/library.sh)
TEXT
}
'''
    assert _preserves('source scripts/library.sh\n', library)
