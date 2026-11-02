# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reached imported functions, not all definitions, own source-write effects."""

import pytest

from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.shell_source_effects import ShellEffectResult, source_effects_preserve
from skillspector.shell_sources import ShellSourceContext, ShellSourceResolver


def _preserves(caller: str, library: str, **kwargs) -> bool:
    resolver = ShellSourceResolver(
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
    return source_effects_preserve(resolver, "scripts/library.sh", len(caller), **kwargs)


_LIBRARY = '''set -euo pipefail
write_file() { local target="$1"; printf '%s' changed > "$target"; }
main() { write_file scripts/library.sh; }
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
'''


def test_import_installs_definitions_without_running_main_or_uncalled_writer() -> None:
    assert _preserves('source scripts/library.sh\n', _LIBRARY)


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
