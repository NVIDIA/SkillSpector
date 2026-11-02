# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-file shell reconstruction remains bound to supplied bytes and callers."""

import shlex

import pytest

from skillspector.artifacts import SecurityTextView
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.shell_helper_argv import ShellHelperOutput, reconstruct_helper_commands, reconstruct_helper_output
from skillspector.shell_sources import ShellSourceContext


_CALLER = '''set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script="$root/scripts/library.sh"
helper=$(sed -n '/^helper() {/,/^}/p' "$script")
bash -c "$helper; helper"
'''
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


def test_known_sibling_function_is_scanned_before_completeness_is_accepted() -> None:
    prepared = _prepare()
    assert prepared is not None
    assert prepared.has_bounded_parse_exhaustion() is True
    list(prepared.analyze_source_commands(
        content=_CALLER,
        file_path="scripts/caller.sh",
        file_type="shell",
        source_view=SecurityTextView("raw", _CALLER),
    ))
    assert prepared.has_bounded_parse_exhaustion() is False
    variants = tuple(prepared.variants())
    assert len(variants) == 1
    assert variants[0][1].text == _LIBRARY.rstrip("\n") + "; helper"
    assert all(origin.path == "scripts/library.sh" for origin in variants[0][1].origins)


def test_known_sibling_dangerous_command_is_retained_at_the_consumer() -> None:
    library = "helper() {\n  rm -rf /\n}\n"
    prepared = _prepare(library=library)
    assert prepared is not None
    findings = list(prepared.analyze_source_commands(
        content=_CALLER,
        file_path="scripts/caller.sh",
        file_type="shell",
        source_view=SecurityTextView("raw", _CALLER),
    ))
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
    findings = list(prepared.analyze_source_commands(content=caller, file_path="scripts/caller.sh", file_type="shell", source_view=SecurityTextView("raw", caller)))
    assert any(finding.rule_id == "TM1" for finding in findings) is destructive
    assert not prepared.has_bounded_parse_exhaustion()


@pytest.mark.parametrize("mutation", [
    'if [[ "$unknown" == yes ]]; then printf changed > "$script"; fi',
    'value="${!#}"; printf changed > "$value"',
    'value="${!#}"; "$value"',
    'value="${!#}"; source "$value"',
    'printf() { echo changed > scripts/library.sh; }; printf "%s" "$unknown"',
])
def test_uploaded_preservation_does_not_hide_unknown_writes_or_code(mutation: str) -> None:
    caller = _CALLER.replace("helper=$(sed", mutation + "\nhelper=$(sed")
    prepared = _prepare(caller)
    assert prepared is None or not tuple(prepared.variants())


def test_uploaded_preservation_never_reuses_an_uncertain_generated_dependency() -> None:
    prefix = '''tmpdir="$(mktemp -d /tmp/preservation.XXXXXX)"
cat > "$tmpdir/dependency" <<'CHILD'
printf safe
CHILD
chmod +x "$tmpdir/dependency"
if [[ "$unknown" == yes ]]; then printf '%s\\n' "$replacement" > "$tmpdir/dependency"; fi
"$tmpdir/dependency"
'''
    caller = _CALLER.replace("helper=$(sed", prefix + "helper=$(sed")
    prepared = _prepare(caller)
    assert prepared is None or not tuple(prepared.variants())


@pytest.mark.parametrize("mutated", [False, True])
def test_uploaded_preservation_accounts_for_each_fragment_in_a_combined_command(mutated: bool) -> None:
    prefix = 'if [[ "$unknown" == yes ]]; then value=one; else value=two; fi\n' * 9
    prefix += 'dependency="$root/scripts/dependency.sh"\n'
    if mutated:
        prefix += 'if [[ "$mutate" == yes ]]; then printf "%s" "$replacement" > "$dependency"; fi\n'
    prefix += 'support=$(sed -n \'/^support() {/,/^}/p\' "$dependency")\n'
    caller = _CALLER.replace("helper=$(sed", prefix + "helper=$(sed").replace('"$helper; helper"', '"$support; $helper; helper"')
    prepared = _prepare(caller, "helper() {\n  support\n}\n", extra_sources={"scripts/dependency.sh": "support() {\n  printf ready\n}\n"})
    if mutated:
        assert prepared is None or not tuple(prepared.variants())
    else:
        assert prepared is not None
        list(prepared.analyze_source_commands(content=caller, file_path="scripts/caller.sh", file_type="shell", source_view=SecurityTextView("raw", caller)))
        assert not prepared.has_bounded_parse_exhaustion()


def test_inspected_destructive_source_does_not_clear_unrelated_unknown_consumer() -> None:
    caller = _CALLER + 'bash -c "$RUNTIME_COMMAND"\n'
    prepared = _prepare(caller, "helper() {\n rm -rf /\n}\n")
    assert prepared is not None
    findings = list(prepared.analyze_source_commands(content=caller, file_path="scripts/caller.sh", file_type="shell", source_view=SecurityTextView("raw", caller)))
    assert any(finding.rule_id == "TM1" for finding in findings)
    assert prepared.has_bounded_parse_exhaustion()


def test_inspected_source_retains_nested_unknown_execution_limit() -> None:
    library = 'helper() {\n rm -rf /\n bash -c "$UNKNOWN"\n}\n'
    prepared = _prepare(library=library)
    assert prepared is not None
    list(prepared.analyze_source_commands(content=_CALLER, file_path="scripts/caller.sh", file_type="shell", source_view=SecurityTextView("raw", _CALLER)))
    assert prepared.has_bounded_parse_exhaustion()


@pytest.mark.parametrize("mutation", [
    'printf "%s\\n" replacement > "$script"\n',
    'sed -i "s/ready/replaced/" "$script"\n',
    'ln -sf /tmp/other "$script"\n',
])
def test_explicit_source_mutation_does_not_reuse_cached_program(mutation: str) -> None:
    caller = _CALLER.replace("helper=$(sed", mutation + "helper=$(sed")
    prepared = _prepare(caller)
    assert prepared is None or not tuple(prepared.variants())


@pytest.mark.parametrize("invocation", [
    'bash "$root/scripts/mutate.sh"',
    'env -u ENV "$root/scripts/mutate.sh"',
    '"$root/scripts/mutate.sh"',
    'source "$root/scripts/mutate.sh"',
    '/tmp/cat "$script"',
    'PATH=/tmp bash -c "true"',
    'cat() { printf replacement > "$script"; }\ncat',
])
def test_cached_script_or_reader_basename_is_not_a_no_write_proof(invocation: str) -> None:
    caller = _CALLER.replace("helper=$(sed", invocation + "\nhelper=$(sed")
    mutator = '''#!/bin/bash
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
printf 'helper() {\\n rm -rf /\\n}\\n' > "$root/scripts/library.sh"
'''
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
    assert tm.prepare_source_analysis(
        content=_CALLER + "# modified\n",
        file_path="scripts/caller.sh",
        file_type="shell",
        source_context=ShellSourceContext("scripts/caller.sh", {"scripts/caller.sh": _CALLER}),
        check_runtime=lambda: None,
    ) is None


_FORWARDER = '''if [[ "$1 $2" == '-n true' ]]; then
  exit 1
fi
if [[ "$1 $2 $3" == '-n su -s' && "$4" == /bin/bash && "$5" == -c ]]; then
  shift 5
  /bin/bash -c "$1"
  exit
fi
exit 99
'''


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
        'cat > "$tmpdir/broker" <<\'BROKER\'\n' + _FORWARDER
        + 'BROKER\nchmod +x "$tmpdir/broker"\n'
        + 'PATH="$tmpdir:$PATH" bash -c ' + shlex.quote(program) + '\n'
    )


@pytest.mark.parametrize("command, destructive", [("printf ready", False), ("rm -rf /", True)])
def test_generated_helper_inspects_both_forwarded_body_and_literal_caller(command: str, destructive: bool) -> None:
    program = "broker -n su -s /bin/bash -c " + shlex.quote(command)
    caller = _generated_forwarder(program)
    prepared = _prepare(caller)
    assert prepared is not None
    assert prepared.has_bounded_parse_exhaustion()
    findings = list(prepared.analyze_source_commands(content=caller, file_path="scripts/caller.sh", file_type="shell", source_view=SecurityTextView("raw", caller)))
    assert any(finding.rule_id == "TM1" for finding in findings) is destructive
    assert not prepared.has_bounded_parse_exhaustion()
    variants = tuple(prepared.variants())
    assert any(value.text == program and value.argv is None for _, value in variants)
    assert any(value.text == command and value.argv is not None for _, value in variants)


@pytest.mark.parametrize("nested", [False, True])
def test_proven_helper_caller_does_not_clear_other_unknown_execution(nested: bool) -> None:
    command = 'rm -rf /; bash -c "$UNKNOWN"' if nested else 'rm -rf /'
    caller = _generated_forwarder("broker -n su -s /bin/bash -c " + shlex.quote(command))
    if not nested:
        caller += 'bash -c "$UNKNOWN"\n'
    prepared = _prepare(caller)
    assert prepared is not None
    findings = list(prepared.analyze_source_commands(content=caller, file_path="scripts/caller.sh", file_type="shell", source_view=SecurityTextView("raw", caller)))
    assert any(finding.rule_id == "TM1" for finding in findings)
    assert prepared.has_bounded_parse_exhaustion()


def test_quoted_argv_keeps_argument_boundaries_through_helper_shift() -> None:
    variants = _forward('''dispatch() {
  local command
  printf -v command '%q ' "$@"
  broker -n su -s /bin/bash -c "exec ${command% }"
}
dispatch printf '%s\\n' 'two words'
''')
    assert variants is not None
    assert {text for values in variants.values() for text, _ in values} == {
        "exec printf '%s\\n' 'two words'"
    }


def test_bash_quoted_bytes_cannot_be_compared_using_normalized_argv() -> None:
    assert _forward('''printf -v encoded '%q ' 'two words'
if [[ "$encoded" == "two\\ words " ]]; then
  broker -n su -s /bin/bash -c 'rm -rf /'
fi
''') is None


@pytest.mark.parametrize("name", ["EPOCHSECONDS", "EPOCHREALTIME", "HISTCMD", "EXECIGNORE", "LD_PRELOAD"])
def test_native_dynamic_or_execution_variables_cannot_be_fictitious_constants(name: str) -> None:
    assert _forward(f"{name}=0\nbroker -n su -s /bin/bash -c 'printf ready'\n") is None


_OUTPUT_HELPER = '''#!/usr/bin/env bash
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
'''


def _helper_output(argv, *, program=_OUTPUT_HELPER, environment=None):
    return reconstruct_helper_output(
        program, argv,
        environment or {"CALLS": "/tmp/owned/log", "ARCHIVE": "/tmp/owned/input.tbz2"},
        frozenset({"/tmp/owned/log"}), tm._next_shell_invocation_word, lambda: None,
    )


def test_helper_stdout_preserves_archive_identity_instead_of_inventing_bytes() -> None:
    assert _helper_output(("print", "-q", "//depot/tools.tbz2#2")) == (
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


@pytest.mark.parametrize("program", [
    "printf '%n' DEST",
    "printf '%s' literal > /tmp/unowned",
    "/tmp/cat /tmp/owned/input.tbz2",
    "printf '%s' \"$(touch /tmp/side-effect)\"",
    "false; printf misleading",
    "printf misleading; exit 1",
    "exec cat /tmp/owned/input.tbz2",
    "IFS=,; printf '%s' \"$*\"",
])
def test_stdout_proof_refuses_unmodeled_effects(program: str) -> None:
    assert _helper_output(("print",), program=program) is None


@pytest.mark.parametrize("program", [
    "set -euo pipefail\n[[ no == yes ]]\nprintf unreachable",
    "set -euo pipefail\ntrue && [[ no == yes ]]\nprintf unreachable",
    'set -euo pipefail\n[[ "$UNKNOWN" == yes ]]\nprintf conditional',
    'set -euo pipefail\nprintf "%s" "$2"',
])
def test_failed_untested_predicate_stops_errexit_helper(program: str) -> None:
    assert _helper_output((), program=program) is None


@pytest.mark.parametrize("program", [
    "set -euo pipefail\nif [[ no == yes ]]; then printf wrong; else printf ready; fi",
    "set -euo pipefail\n[[ no == yes ]] && printf wrong\nprintf ready",
])
def test_tested_predicate_does_not_exit_the_helper(program: str) -> None:
    assert _helper_output((), program=program) == ((ShellHelperOutput(text="ready"),),)


@pytest.mark.parametrize("closing, complete", [("EOF\n", True), ("EOF\r\n", False)])
def test_lf_heredoc_delimiter_does_not_silently_discard_carriage_return(closing, complete) -> None:
    content = "literal body\n" + closing
    end = tm._skip_static_heredoc_bodies(content, 0, len(content), (("EOF", False, True),), lambda: None)
    assert (end == len(content)) is complete


_ARCHIVE_CALLER = '''set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmpdir="$(mktemp -d /tmp/archive.XXXXXX)"
mkdir -p "$tmpdir/tree" "$tmpdir/output"
cp "$root/scripts/library.sh" "$tmpdir/tree/library.sh"
tar -cf "$tmpdir/payload.tar" -C "$tmpdir" tree
tar -xf "$tmpdir/payload.tar" -C "$tmpdir/output"
code="$(cat "$tmpdir/output/tree/library.sh")"
bash -c "$code; helper"
'''


@pytest.mark.parametrize("dangerous", [False, True])
def test_generated_archive_source_is_inspected_with_original_provenance(dangerous: bool) -> None:
    library = "helper() {\n  rm -rf /\n}\n" if dangerous else _LIBRARY
    prepared = _prepare(_ARCHIVE_CALLER, library)
    assert prepared is not None
    findings = list(prepared.analyze_source_commands(content=_ARCHIVE_CALLER, file_path="scripts/caller.sh", file_type="shell", source_view=SecurityTextView("raw", _ARCHIVE_CALLER)))
    assert any(finding.rule_id == "TM1" for finding in findings) is dangerous
    assert all(value.origins and any(origin.path == "scripts/library.sh" for origin in value.origins) for _, value in prepared.variants())
    assert not prepared.has_bounded_parse_exhaustion()


def test_unknown_archive_member_write_cannot_reuse_the_cached_copy() -> None:
    caller = _ARCHIVE_CALLER.replace('code="$(cat', 'ln -sf /runtime/unknown "$tmpdir/output/tree/library.sh"\ncode="$(cat')
    prepared = _prepare(caller)
    assert prepared is None or not tuple(prepared.variants())
