# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-file shell reconstruction remains bound to supplied bytes and callers."""

import pytest

from skillspector.artifacts import SecurityTextView
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.shell_helper_argv import reconstruct_helper_commands
from skillspector.shell_sources import ShellSourceContext


_CALLER = '''set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script="$root/scripts/library.sh"
helper=$(sed -n '/^helper() {/,/^}/p' "$script")
bash -c "$helper; helper"
'''
_LIBRARY = "helper() {\n  printf '%s\\n' ready\n}\n"


def _prepare(caller: str = _CALLER, library: str = _LIBRARY):
    sources = {"scripts/caller.sh": caller, "scripts/library.sh": library}
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
    for finding in findings:
        assert finding.location.file == "scripts/caller.sh"
        assert finding.evidence["reconstructed_shell_source"]["origins"]


@pytest.mark.parametrize("mutation", [
    'printf "%s\\n" replacement > "$script"\n',
    'sed -i "s/ready/replaced/" "$script"\n',
    'ln -sf /tmp/other "$script"\n',
])
def test_explicit_source_mutation_does_not_reuse_cached_program(mutation: str) -> None:
    caller = _CALLER.replace("helper=$(sed", mutation + "helper=$(sed")
    prepared = _prepare(caller)
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
