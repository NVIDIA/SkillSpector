# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-file shell reconstruction remains bound to supplied bytes and callers."""

import shlex

import pytest
from skillspector.artifacts import SecurityTextView

from skillspector import shell_sources
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.shell_helper_argv import (
    ShellHelperOutput,
    reconstruct_helper_commands,
    reconstruct_helper_output,
)
from skillspector.shell_sources import ShellSourceContext, ShellSourceResolver, ShellSourceValue

_CALLER = """set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script="$root/scripts/library.sh"
helper=$(sed -n '/^helper() {/,/^}/p' "$script")
bash -c "$helper; helper"
"""
_LIBRARY = "helper() {\n  printf '%s\\n' ready\n}\n"


def _prepare(caller: str = _CALLER, library: str = _LIBRARY, *, extra_sources=None):
    sources = {"scripts/caller.sh": caller, "scripts/library.sh": library}
    sources.update(extra_sources or {})
    return tm.prepare_source_analysis(
        content=caller,
        file_path="scripts/caller.sh",
        file_type="shell",
        source_context=ShellSourceContext("scripts/caller.sh", sources),
        check_runtime=lambda: None,
    )


def _value_resolver(caller: str, *, inherited=None) -> ShellSourceResolver:
    return ShellSourceResolver(
        ShellSourceContext("scripts/caller.sh", {"scripts/caller.sh": caller}),
        tm._next_shell_invocation_word,
        lambda text, start, end: tm._skip_command_substitution(text, start, end, lambda: None),
        tm._static_heredoc_header,
        tm._skip_static_heredoc_bodies,
        lambda: None,
        inherited_values=inherited,
    )


def _expanded_values(resolver: ShellSourceResolver, name: str):
    token = '"$' + name + '"'
    start = 0
    values = []
    while (start := resolver.content.find(token, start)) >= 0:
        value = resolver.word(start, start + len(token))
        values.append(None if value is None else value.text)
        start += len(token)
    return values


@pytest.mark.parametrize("reverse", [False, True])
def test_each_consumer_uses_its_current_assignment_without_cache_poisoning(reverse: bool) -> None:
    caller = 'helper=first\nprintf "%s" "$helper"\nhelper=second\nprintf "%s" "$helper"\n'
    resolver = _value_resolver(caller)
    positions = [caller.index('"$helper"'), caller.rindex('"$helper"')]
    if reverse:
        positions.reverse()
    observed = {start: resolver.word(start, start + len('"$helper"')).text for start in positions}
    assert observed[caller.index('"$helper"')] == "first"
    assert observed[caller.rindex('"$helper"')] == "second"


def test_unknown_reassignment_does_not_erase_an_earlier_known_value() -> None:
    caller = 'helper=first\nprintf "%s" "$helper"\nhelper="$runtime"\nprintf "%s" "$helper"\n'
    assert _expanded_values(_value_resolver(caller), "helper") == ["first", None]


def test_later_known_assignment_cannot_repair_an_earlier_unknown_value() -> None:
    caller = 'helper="$runtime"\nprintf "%s" "$helper"\nhelper=second\nprintf "%s" "$helper"\n'
    assert _expanded_values(_value_resolver(caller), "helper") == [None, "second"]


def test_captured_alias_resolves_dependencies_at_its_assignment_offset() -> None:
    caller = 'value=old\nalias="$value"\nvalue=new\nprintf "%s" "$alias"\nvalue="$alias"\nprintf "%s" "$value"\n'
    resolver = _value_resolver(caller)
    assert _expanded_values(resolver, "alias") == ["old", "old"]
    assert (
        resolver.word(caller.rindex('"$value"'), caller.rindex('"$value"') + len('"$value"')).text
        == "old"
    )


def test_pending_self_reference_and_future_definitions_remain_unknown() -> None:
    caller = 'helper=first\nprintf "%s" "$helper"\nhelper="$helper"\nprintf "%s" "$helper"\n'
    assert _expanded_values(_value_resolver(caller), "helper") == ["first", None, None]
    cyclic = 'first="$second"\nsecond="$first"\nprintf "%s" "$first"\n'
    assert _expanded_values(_value_resolver(cyclic), "first") == [None, None]


@pytest.mark.parametrize(
    "mutation",
    [
        "if test -e /tmp/condition; then helper=second; fi",
        'while test -e /tmp/condition; do printf "%s" "$helper"; helper=second; done',
        "helper=second printf safe",
        "export helper=second",
    ],
)
def test_conditional_loop_and_scoped_writes_keep_filewide_refusal(mutation: str) -> None:
    caller = 'helper=first\nprintf "%s" "$helper"\n' + mutation + '\nprintf "%s" "$helper"\n'
    assert all(value is None for value in _expanded_values(_value_resolver(caller), "helper"))


@pytest.mark.parametrize(
    "invocation",
    [
        'runner=mutate\n"$runner"',
        '"$runtime"',
        'runner=eval\n"$runner" "helper=second"',
        'runner=source\n"$runner" scripts/mutate.sh',
        'runner=read\n"$runner" helper',
        'runner=printf\n"$runner" -v helper second',
        '"eval" "helper=second"',
        r'e\val "helper=second"',
    ],
)
def test_runtime_selected_parent_mutation_cannot_preserve_captured_bindings(
    invocation: str,
) -> None:
    caller = 'mutate() { helper=second; }\nhelper=first\nprintf "%s" "$helper"\n'
    caller += invocation + '\nprintf "%s" "$helper"\nhelper=third\nprintf "%s" "$helper"\n'
    assert all(value is None for value in _expanded_values(_value_resolver(caller), "helper"))


def test_dynamic_callee_uses_final_loop_invalidations() -> None:
    caller = 'helper=first\nrunner=/bin/true\nwhile test -e /tmp/condition; do "$runner" "helper=second"; runner=eval; done\nprintf "%s" "$helper"\n'
    assert _expanded_values(_value_resolver(caller), "helper") == [None]


@pytest.mark.parametrize("wrapper", ["!", "! !", "time", "! time"])
@pytest.mark.parametrize("callee", ['"$runner"', "/bin/true"])
def test_parent_reserved_wrappers_cannot_hide_a_mutating_callee(wrapper: str, callee: str) -> None:
    caller = "mutate() { helper=changed; }\nhelper=first\nrunner=mutate\n"
    caller += wrapper + " " + callee + '\nprintf "%s" "$helper"\nhelper=third\n'
    assert _expanded_values(_value_resolver(caller), "helper") == [None]


@pytest.mark.parametrize("condition", ['[ "$unknown" = yes ]', '[[ "$unknown" == yes ]]'])
def test_literal_test_command_does_not_invalidate_unchanged_binding(condition: str) -> None:
    caller = (
        "helper=first\nif "
        + condition
        + '; then printf safe; fi\nprintf "%s" "$helper"\nhelper=second\nprintf "%s" "$helper"\n'
    )
    assert _expanded_values(_value_resolver(caller), "helper") == ["first", "second"]


@pytest.mark.parametrize("callee", ['"scripts/worker.sh"', '"$runner"'])
def test_proved_external_path_call_does_not_invent_parent_variable_mutation(callee: str) -> None:
    caller = (
        "helper=first\nrunner=scripts/worker.sh\n"
        + callee
        + '\nprintf "%s" "$helper"\nhelper=second\nprintf "%s" "$helper"\n'
    )
    assert _expanded_values(_value_resolver(caller), "helper") == ["first", "second"]


def test_inherited_value_applies_only_before_the_first_local_assignment() -> None:
    caller = 'printf "%s" "$helper"\nhelper=local\nprintf "%s" "$helper"\n'
    resolver = _value_resolver(caller, inherited={"helper": ShellSourceValue("inherited")})
    assert _expanded_values(resolver, "helper") == ["inherited", "local"]
    pending = 'helper="$helper"\nprintf "%s" "$helper"\n'
    assert _expanded_values(
        _value_resolver(pending, inherited={"helper": ShellSourceValue("inherited")}), "helper"
    ) == [None, None]


def test_assignment_history_has_a_shared_definition_bound(monkeypatch) -> None:
    monkeypatch.setattr(shell_sources, "MAX_SHELL_BINDING_DEFINITIONS", 2)
    assert _value_resolver("helper=one\nhelper=two\n").valid
    assert not _value_resolver("helper=one\nhelper=two\nhelper=three\n").valid


def test_captured_destructive_source_retains_its_first_consumer_after_later_reassignment() -> None:
    caller = (
        _CALLER
        + 'printf changed > "$script"\nhelper=$(sed -n \'/^helper() {/,/^}/p\' "$script")\nbash -c "$helper; helper"\n'
    )
    prepared = _prepare(caller, "helper() {\n  rm -rf /\n}\n")
    assert prepared is not None
    findings = list(
        prepared.analyze_source_commands(
            content=caller,
            file_path="scripts/caller.sh",
            file_type="shell",
            source_view=SecurityTextView("raw", caller),
        )
    )
    destructive = [finding for finding in findings if finding.rule_id == "TM1"]
    assert destructive
    first = caller.index('"$helper; helper"')
    assert {
        finding.evidence["reconstructed_shell_source"]["consumer_start"] for finding in destructive
    } == {first}
    assert all(
        origin["path"] == "scripts/library.sh"
        for finding in destructive
        for origin in finding.evidence["reconstructed_shell_source"]["origins"]
    )
    assert prepared.has_bounded_parse_exhaustion()


@pytest.mark.parametrize("destructive", [False, True])
def test_captured_fragment_is_not_re_read_after_its_source_changes(destructive: bool) -> None:
    caller = _CALLER.replace(
        'bash -c "$helper; helper"', 'printf changed > "$script"\nbash -c "$helper; helper"'
    )
    library = "helper() {\n  rm -rf /\n}\n" if destructive else _LIBRARY
    prepared = _prepare(caller, library)
    assert prepared is not None
    findings = list(
        prepared.analyze_source_commands(
            content=caller,
            file_path="scripts/caller.sh",
            file_type="shell",
            source_view=SecurityTextView("raw", caller),
        )
    )
    assert any(finding.rule_id == "TM1" for finding in findings) is destructive
    assert not prepared.has_bounded_parse_exhaustion()


def test_known_sibling_function_is_scanned_before_completeness_is_accepted() -> None:
    prepared = _prepare()
    assert prepared is not None
    assert prepared.has_bounded_parse_exhaustion() is True
    list(
        prepared.analyze_source_commands(
            content=_CALLER,
            file_path="scripts/caller.sh",
            file_type="shell",
            source_view=SecurityTextView("raw", _CALLER),
        )
    )
    assert prepared.has_bounded_parse_exhaustion() is False
    variants = tuple(prepared.variants())
    assert len(variants) == 1
    assert variants[0][1].text == _LIBRARY.rstrip("\n") + "; helper"
    assert all(origin.path == "scripts/library.sh" for origin in variants[0][1].origins)


def test_known_sibling_dangerous_command_is_retained_at_the_consumer() -> None:
    library = "helper() {\n  rm -rf /\n}\n"
    prepared = _prepare(library=library)
    assert prepared is not None
    findings = list(
        prepared.analyze_source_commands(
            content=_CALLER,
            file_path="scripts/caller.sh",
            file_type="shell",
            source_view=SecurityTextView("raw", _CALLER),
        )
    )
    assert any(finding.rule_id == "TM1" for finding in findings)
    assert not prepared.has_bounded_parse_exhaustion()
    for finding in findings:
        assert finding.location.file == "scripts/caller.sh"
        assert finding.evidence["reconstructed_shell_source"]["origins"]


@pytest.mark.parametrize("destructive", [False, True])
def test_uploaded_source_can_survive_unknown_data_and_joined_branches(destructive: bool) -> None:
    prefix = 'value="${!#}"\nprintf "%s\\n" "$value" > /tmp/inspection-data\n'
    prefix += 'if [[ "$unknown" == yes ]]; then value=one; else value=two; fi\n' * 9
    caller = _CALLER.replace("helper=$(sed", prefix + "helper=$(sed")
    library = "helper() {\n  rm -rf /\n}\n" if destructive else _LIBRARY
    prepared = _prepare(caller, library)
    assert prepared is not None
    findings = list(
        prepared.analyze_source_commands(
            content=caller,
            file_path="scripts/caller.sh",
            file_type="shell",
            source_view=SecurityTextView("raw", caller),
        )
    )
    assert any(finding.rule_id == "TM1" for finding in findings) is destructive
    assert not prepared.has_bounded_parse_exhaustion()


@pytest.mark.parametrize(
    "mutation",
    [
        'if [[ "$unknown" == yes ]]; then printf changed > "$script"; fi',
        'value="${!#}"; printf changed > "$value"',
        'value="${!#}"; "$value"',
        'value="${!#}"; source "$value"',
        'printf() { echo changed > scripts/library.sh; }; printf "%s" "$unknown"',
    ],
)
def test_uploaded_preservation_does_not_hide_unknown_writes_or_code(mutation: str) -> None:
    caller = _CALLER.replace("helper=$(sed", mutation + "\nhelper=$(sed")
    prepared = _prepare(caller)
    assert prepared is None or not tuple(prepared.variants())


def test_uploaded_preservation_never_reuses_an_uncertain_generated_dependency() -> None:
    prefix = """tmpdir="$(mktemp -d /tmp/preservation.XXXXXX)"
cat > "$tmpdir/dependency" <<'CHILD'
printf safe
CHILD
chmod +x "$tmpdir/dependency"
if [[ "$unknown" == yes ]]; then printf '%s\\n' "$replacement" > "$tmpdir/dependency"; fi
"$tmpdir/dependency"
"""
    caller = _CALLER.replace("helper=$(sed", prefix + "helper=$(sed")
    prepared = _prepare(caller)
    assert prepared is None or not tuple(prepared.variants())


@pytest.mark.parametrize("mutated", [False, True])
def test_uploaded_preservation_accounts_for_each_fragment_in_a_combined_command(
    mutated: bool,
) -> None:
    prefix = 'if [[ "$unknown" == yes ]]; then value=one; else value=two; fi\n' * 9
    prefix += 'dependency="$root/scripts/dependency.sh"\n'
    if mutated:
        prefix += 'if [[ "$mutate" == yes ]]; then printf "%s" "$replacement" > "$dependency"; fi\n'
    prefix += "support=$(sed -n '/^support() {/,/^}/p' \"$dependency\")\n"
    caller = _CALLER.replace("helper=$(sed", prefix + "helper=$(sed").replace(
        '"$helper; helper"', '"$support; $helper; helper"'
    )
    prepared = _prepare(
        caller,
        "helper() {\n  support\n}\n",
        extra_sources={"scripts/dependency.sh": "support() {\n  printf ready\n}\n"},
    )
    if mutated:
        assert prepared is None or not tuple(prepared.variants())
    else:
        assert prepared is not None
        list(
            prepared.analyze_source_commands(
                content=caller,
                file_path="scripts/caller.sh",
                file_type="shell",
                source_view=SecurityTextView("raw", caller),
            )
        )
        assert not prepared.has_bounded_parse_exhaustion()


def test_inspected_destructive_source_does_not_clear_unrelated_unknown_consumer() -> None:
    caller = _CALLER + 'bash -c "$RUNTIME_COMMAND"\n'
    prepared = _prepare(caller, "helper() {\n rm -rf /\n}\n")
    assert prepared is not None
    findings = list(
        prepared.analyze_source_commands(
            content=caller,
            file_path="scripts/caller.sh",
            file_type="shell",
            source_view=SecurityTextView("raw", caller),
        )
    )
    assert any(finding.rule_id == "TM1" for finding in findings)
    assert prepared.has_bounded_parse_exhaustion()


def test_inspected_source_retains_nested_unknown_execution_limit() -> None:
    library = 'helper() {\n rm -rf /\n bash -c "$UNKNOWN"\n}\n'
    prepared = _prepare(library=library)
    assert prepared is not None
    list(
        prepared.analyze_source_commands(
            content=_CALLER,
            file_path="scripts/caller.sh",
            file_type="shell",
            source_view=SecurityTextView("raw", _CALLER),
        )
    )
    assert prepared.has_bounded_parse_exhaustion()


@pytest.mark.parametrize(
    "mutation",
    [
        'printf "%s\\n" replacement > "$script"\n',
        'sed -i "s/ready/replaced/" "$script"\n',
        'ln -sf /tmp/other "$script"\n',
    ],
)
def test_explicit_source_mutation_does_not_reuse_cached_program(mutation: str) -> None:
    caller = _CALLER.replace("helper=$(sed", mutation + "helper=$(sed")
    prepared = _prepare(caller)
    assert prepared is None or not tuple(prepared.variants())


@pytest.mark.parametrize(
    "mutation",
    [
        "cp -r replacement/. subdir",
        "rm -rf subdir",
        'destination=subdir\nrm -rf "$destination"',
    ],
)
def test_relative_subtree_write_after_cd_cannot_reuse_uploaded_source(mutation: str) -> None:
    caller = _CALLER.replace("scripts/library.sh", "scripts/subdir/library.sh")
    caller = caller.replace("helper=$(sed", "cd scripts\n" + mutation + "\nhelper=$(sed")
    prepared = _prepare(
        caller,
        extra_sources={
            "scripts/subdir/library.sh": _LIBRARY,
            "scripts/replacement/library.sh": "helper() {\n  rm -rf /\n}\n",
        },
    )
    assert prepared is None or not tuple(prepared.variants())


def test_ordered_cwd_proof_preserves_source_after_disjoint_relative_subtree_write() -> None:
    caller = _CALLER.replace("scripts/library.sh", "scripts/subdir/library.sh")
    caller = caller.replace("helper=$(sed", "cd scripts\nrm -rf cache\nhelper=$(sed")
    prepared = _prepare(
        caller,
        extra_sources={"scripts/subdir/library.sh": "helper() {\n  rm -rf /\n}\n"},
    )
    assert prepared is not None
    findings = list(
        prepared.analyze_source_commands(
            content=caller,
            file_path="scripts/caller.sh",
            file_type="shell",
            source_view=SecurityTextView("raw", caller),
        )
    )
    assert any(finding.rule_id == "TM1" for finding in findings)
    assert not prepared.has_bounded_parse_exhaustion()
    assert all(
        origin.path == "scripts/subdir/library.sh"
        for _, value in prepared.variants()
        for origin in value.origins
    )


@pytest.mark.parametrize(
    "invocation",
    [
        'bash "$root/scripts/mutate.sh"',
        'env -u ENV "$root/scripts/mutate.sh"',
        '"$root/scripts/mutate.sh"',
        'source "$root/scripts/mutate.sh"',
        '/tmp/cat "$script"',
        'PATH=/tmp bash -c "true"',
        'cat() { printf replacement > "$script"; }\ncat',
    ],
)
def test_cached_script_or_reader_basename_is_not_a_no_write_proof(invocation: str) -> None:
    caller = _CALLER.replace("helper=$(sed", invocation + "\nhelper=$(sed")
    mutator = """#!/bin/bash
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
printf 'helper() {\\n rm -rf /\\n}\\n' > "$root/scripts/library.sh"
"""
    prepared = _prepare(caller, extra_sources={"scripts/mutate.sh": mutator})
    assert prepared is None or not tuple(prepared.variants())


def test_missing_source_remains_unresolved() -> None:
    prepared = tm.prepare_source_analysis(
        content=_CALLER,
        file_path="scripts/caller.sh",
        file_type="shell",
        source_context=ShellSourceContext("scripts/caller.sh", {"scripts/caller.sh": _CALLER}),
        check_runtime=lambda: None,
    )
    assert prepared is None


def test_transformed_text_cannot_borrow_original_source_identity() -> None:
    assert (
        tm.prepare_source_analysis(
            content=_CALLER + "# modified\n",
            file_path="scripts/caller.sh",
            file_type="shell",
            source_context=ShellSourceContext("scripts/caller.sh", {"scripts/caller.sh": _CALLER}),
            check_runtime=lambda: None,
        )
        is None
    )


_FORWARDER = """if [[ "$1 $2" == '-n true' ]]; then
  exit 1
fi
if [[ "$1 $2 $3" == '-n su -s' && "$4" == /bin/bash && "$5" == -c ]]; then
  shift 5
  /bin/bash -c "$1"
  exit
fi
exit 99
"""


def _forward(caller: str):
    return reconstruct_helper_commands(
        _FORWARDER, (caller,), "broker", tm._next_shell_invocation_word, lambda: None
    )


def test_known_helper_arguments_reconstruct_the_actual_forwarded_program() -> None:
    variants = _forward("broker -n su -s /bin/bash -c 'printf ready'")
    assert variants is not None
    assert {text for values in variants.values() for text, _ in values} == {"printf ready"}


def _generated_forwarder(program: str) -> str:
    return (
        'set -euo pipefail\ntmpdir="$(mktemp -d /tmp/generated-control.XXXXXX)"\n'
        "cat > \"$tmpdir/broker\" <<'BROKER'\n"
        + _FORWARDER
        + 'BROKER\nchmod +x "$tmpdir/broker"\n'
        + 'PATH="$tmpdir:$PATH" bash -c '
        + shlex.quote(program)
        + "\n"
    )


@pytest.mark.parametrize("command, destructive", [("printf ready", False), ("rm -rf /", True)])
def test_generated_helper_inspects_both_forwarded_body_and_literal_caller(
    command: str, destructive: bool
) -> None:
    program = "broker -n su -s /bin/bash -c " + shlex.quote(command)
    caller = _generated_forwarder(program)
    prepared = _prepare(caller)
    assert prepared is not None
    assert prepared.has_bounded_parse_exhaustion()
    findings = list(
        prepared.analyze_source_commands(
            content=caller,
            file_path="scripts/caller.sh",
            file_type="shell",
            source_view=SecurityTextView("raw", caller),
        )
    )
    assert any(finding.rule_id == "TM1" for finding in findings) is destructive
    assert not prepared.has_bounded_parse_exhaustion()
    variants = tuple(prepared.variants())
    assert any(value.text == program and value.argv is None for _, value in variants)
    assert any(value.text == command and value.argv is not None for _, value in variants)


@pytest.mark.parametrize("nested", [False, True])
def test_proven_helper_caller_does_not_clear_other_unknown_execution(nested: bool) -> None:
    command = 'rm -rf /; bash -c "$UNKNOWN"' if nested else "rm -rf /"
    caller = _generated_forwarder("broker -n su -s /bin/bash -c " + shlex.quote(command))
    if not nested:
        caller += 'bash -c "$UNKNOWN"\n'
    prepared = _prepare(caller)
    assert prepared is not None
    findings = list(
        prepared.analyze_source_commands(
            content=caller,
            file_path="scripts/caller.sh",
            file_type="shell",
            source_view=SecurityTextView("raw", caller),
        )
    )
    assert any(finding.rule_id == "TM1" for finding in findings)
    assert prepared.has_bounded_parse_exhaustion()


def test_quoted_argv_keeps_argument_boundaries_through_helper_shift() -> None:
    variants = _forward("""dispatch() {
  local command
  printf -v command '%q ' "$@"
  broker -n su -s /bin/bash -c "exec ${command% }"
}
dispatch printf '%s\\n' 'two words'
""")
    assert variants is not None
    assert {text for values in variants.values() for text, _ in values} == {
        "exec printf '%s\\n' 'two words'"
    }


def test_bash_quoted_bytes_cannot_be_compared_using_normalized_argv() -> None:
    assert (
        _forward("""printf -v encoded '%q ' 'two words'
if [[ "$encoded" == "two\\ words " ]]; then
  broker -n su -s /bin/bash -c 'rm -rf /'
fi
""")
        is None
    )


@pytest.mark.parametrize(
    "name", ["EPOCHSECONDS", "EPOCHREALTIME", "HISTCMD", "EXECIGNORE", "LD_PRELOAD"]
)
def test_native_dynamic_or_execution_variables_cannot_be_fictitious_constants(name: str) -> None:
    assert _forward(f"{name}=0\nbroker -n su -s /bin/bash -c 'printf ready'\n") is None


_OUTPUT_HELPER = """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$CALLS"
case "$1" in
  files)
    case "${MATCH_MODE:-one}" in
      zero) exit 0 ;;
      two) printf '//depot/a/%s#1\\n//depot/b/%s#2\\n' "${2##*/}" "${2##*/}" ;;
      *) printf '//depot/a/%s#2\\n' "${2##*/}" ;;
    esac ;;
  print)
    if [[ "$*" == *tools.tbz2* ]]; then
      cat "$ARCHIVE"
    else
      printf '#!/usr/bin/env bash\\n'
    fi ;;
esac
"""


def _helper_output(argv, *, program=_OUTPUT_HELPER, environment=None):
    return reconstruct_helper_output(
        program,
        argv,
        environment or {"CALLS": "/tmp/owned/log", "ARCHIVE": "/tmp/owned/input.tbz2"},
        frozenset({"/tmp/owned/log"}),
        tm._next_shell_invocation_word,
        lambda: None,
    )


def test_helper_stdout_preserves_archive_identity_instead_of_inventing_bytes() -> None:
    assert _helper_output(("print", "-q", "//depot/tools.tbz2#2")) == (
        (ShellHelperOutput(path="/tmp/owned/input.tbz2"),),
    )


@pytest.mark.parametrize("head", ["cat", "/bin/cat", "/usr/bin/cat"])
@pytest.mark.parametrize("redirect", ["> /dev/null", '>> "$CALLS"', "1>&2"])
def test_discarded_helper_read_cannot_claim_successful_branch(head: str, redirect: str) -> None:
    program = (
        f'set -euo pipefail\nif {head} "$ARCHIVE" {redirect}; '
        "then printf safe; else printf 'rm -rf /'; fi\n"
    )
    assert _helper_output((), program=program) is None


@pytest.mark.parametrize("continuation", ["\nprintf safe", " && printf safe", " || printf unsafe"])
def test_discarded_helper_read_cannot_borrow_errexit_success(continuation: str) -> None:
    program = 'set -euo pipefail\ncat "$ARCHIVE" > /dev/null' + continuation
    assert _helper_output((), program=program) is None


@pytest.mark.parametrize("head", ["cat", "/bin/cat", "/usr/bin/cat"])
@pytest.mark.parametrize("redirect", ["", "2> /dev/null"])
def test_kept_helper_read_retains_its_required_file_identity(head: str, redirect: str) -> None:
    program = f'set -euo pipefail\n{head} "$ARCHIVE" {redirect}\n'
    assert _helper_output((), program=program) == (
        (ShellHelperOutput(path="/tmp/owned/input.tbz2"),),
    )


def test_unknown_helper_argument_preserves_both_possible_outputs() -> None:
    assert set(_helper_output(("print", "-q", None))) == {
        (ShellHelperOutput(path="/tmp/owned/input.tbz2"),),
        (ShellHelperOutput(text="#!/usr/bin/env bash\n"),),
    }


def test_helper_default_parameter_does_not_invent_an_inherited_environment_value() -> None:
    assert set(_helper_output(("files", "//depot/tool.sh"))) == {
        (),
        (ShellHelperOutput(text="//depot/a/tool.sh#1\n//depot/b/tool.sh#2\n"),),
        (ShellHelperOutput(text="//depot/a/tool.sh#2\n"),),
    }


@pytest.mark.parametrize(
    "program",
    [
        "printf '%n' DEST",
        "printf '%s' literal > /tmp/unowned",
        "/tmp/cat /tmp/owned/input.tbz2",
        "printf '%s' \"$(touch /tmp/side-effect)\"",
        "false; printf misleading",
        "printf misleading; exit 1",
        "exec cat /tmp/owned/input.tbz2",
        "IFS=,; printf '%s' \"$*\"",
    ],
)
def test_stdout_proof_refuses_unmodeled_effects(program: str) -> None:
    assert _helper_output(("print",), program=program) is None


@pytest.mark.parametrize(
    "program",
    [
        "set -euo pipefail\n[[ no == yes ]]\nprintf unreachable",
        "set -euo pipefail\ntrue && [[ no == yes ]]\nprintf unreachable",
        'set -euo pipefail\n[[ "$UNKNOWN" == yes ]]\nprintf conditional',
        'set -euo pipefail\nprintf "%s" "$2"',
    ],
)
def test_failed_untested_predicate_stops_errexit_helper(program: str) -> None:
    assert _helper_output((), program=program) is None


@pytest.mark.parametrize(
    "program",
    [
        "set -euo pipefail\nif [[ no == yes ]]; then printf wrong; else printf ready; fi",
        "set -euo pipefail\n[[ no == yes ]] && printf wrong\nprintf ready",
    ],
)
def test_tested_predicate_does_not_exit_the_helper(program: str) -> None:
    assert _helper_output((), program=program) == ((ShellHelperOutput(text="ready"),),)


@pytest.mark.parametrize("closing, complete", [("EOF\n", True), ("EOF\r\n", False)])
def test_lf_heredoc_delimiter_does_not_silently_discard_carriage_return(closing, complete) -> None:
    content = "literal body\n" + closing
    end = tm._skip_static_heredoc_bodies(
        content, 0, len(content), (("EOF", False, True),), lambda: None
    )
    assert (end == len(content)) is complete


_ARCHIVE_CALLER = """set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmpdir="$(mktemp -d /tmp/archive.XXXXXX)"
mkdir -p "$tmpdir/tree" "$tmpdir/output"
cp "$root/scripts/library.sh" "$tmpdir/tree/library.sh"
tar -cf "$tmpdir/payload.tar" -C "$tmpdir" tree
tar -xf "$tmpdir/payload.tar" -C "$tmpdir/output"
code="$(cat "$tmpdir/output/tree/library.sh")"
bash -c "$code; helper"
"""


@pytest.mark.parametrize("dangerous", [False, True])
def test_generated_archive_source_is_inspected_with_original_provenance(dangerous: bool) -> None:
    library = "helper() {\n  rm -rf /\n}\n" if dangerous else _LIBRARY
    prepared = _prepare(_ARCHIVE_CALLER, library)
    assert prepared is not None
    findings = list(
        prepared.analyze_source_commands(
            content=_ARCHIVE_CALLER,
            file_path="scripts/caller.sh",
            file_type="shell",
            source_view=SecurityTextView("raw", _ARCHIVE_CALLER),
        )
    )
    assert any(finding.rule_id == "TM1" for finding in findings) is dangerous
    assert all(
        value.origins and any(origin.path == "scripts/library.sh" for origin in value.origins)
        for _, value in prepared.variants()
    )
    assert not prepared.has_bounded_parse_exhaustion()


def test_unknown_archive_member_write_cannot_reuse_the_cached_copy() -> None:
    caller = _ARCHIVE_CALLER.replace(
        'code="$(cat', 'ln -sf /runtime/unknown "$tmpdir/output/tree/library.sh"\ncode="$(cat'
    )
    prepared = _prepare(caller)
    assert prepared is None or not tuple(prepared.variants())
