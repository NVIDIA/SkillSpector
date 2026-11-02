# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cached/generated source identity survives only understood prior effects."""

import pytest

from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.shell_sources import (
    ShellSourceContext,
    ShellSourceResolver,
    generated_file_is_unmodified,
    source_is_unmodified,
)


def _resolver(content: str) -> ShellSourceResolver:
    return ShellSourceResolver(
        ShellSourceContext("scripts/caller.sh", {
            "scripts/caller.sh": content,
            "scripts/library.sh": "helper() {\n printf ready\n}\n",
        }),
        tm._next_shell_invocation_word,
        lambda text, start, end: tm._skip_command_substitution(text, start, end, lambda: None),
        tm._static_heredoc_header,
        tm._skip_static_heredoc_bodies,
        lambda: None,
    )


@pytest.mark.parametrize("command", [
    "sed -n -e 'p' unrelated.txt -e 'w scripts/library.sh'",
    "sed -n 'p' unrelated.txt -i scripts/library.sh",
    "sed -n 'p' unrelated.txt -f program.sed",
    "sed -n 'p' unrelated.txt --expression='w scripts/library.sh'",
])
def test_trailing_sed_options_cannot_hide_writes(command: str) -> None:
    resolver = _resolver(command + "\n")
    assert not source_is_unmodified(resolver, "scripts/library.sh", len(resolver.content))


@pytest.mark.parametrize("command", [
    "sed -n 'p' unrelated.txt",
    "sed -n -e 'p' first.txt second.txt",
    "sed -n -e 's/a/b/g' first.txt",
])
def test_all_literal_sed_inputs_keep_readonly_proof(command: str) -> None:
    resolver = _resolver(command + "\n")
    assert source_is_unmodified(resolver, "scripts/library.sh", len(resolver.content))


_GENERATED = '''set -euo pipefail
tmpdir="$(mktemp -d /tmp/fixture.XXXXXX)"
cat > "$tmpdir/helper" <<'HELPER'
#!/bin/bash
printf ready
HELPER
CHMOD +x "$tmpdir/helper"
PATH="$tmpdir:$PATH" bash -c 'helper'
'''


@pytest.mark.parametrize("chmod", ["chmod", "/bin/chmod", "/usr/bin/chmod"])
def test_native_chmod_establishes_generated_executable(chmod: str) -> None:
    resolver = _resolver(_GENERATED.replace("CHMOD", chmod))
    generated, = resolver.generated_files()
    caller = resolver.commands[-1]
    assert generated_file_is_unmodified(resolver, generated, (caller,))


@pytest.mark.parametrize("chmod", ["/tmp/chmod", '"$tmpdir/chmod"'])
def test_chmod_basename_does_not_establish_generated_executable(chmod: str) -> None:
    resolver = _resolver(_GENERATED.replace("CHMOD", chmod))
    generated, = resolver.generated_files()
    assert not generated_file_is_unmodified(resolver, generated, (resolver.commands[-1],))


def test_shadowed_chmod_cannot_establish_generated_executable() -> None:
    content = _GENERATED.replace("CHMOD", "chmod").replace(
        'chmod +x', 'chmod() { true; }\nchmod +x'
    )
    resolver = _resolver(content)
    assert not tuple(resolver.generated_files())


@pytest.mark.parametrize("setting", ["BASH_ENV", "LD_PRELOAD", "ENV"])
def test_caller_startup_settings_cannot_borrow_generated_identity(setting: str) -> None:
    content = _GENERATED.replace("CHMOD", "chmod").replace(
        'PATH="$tmpdir:$PATH"', f'{setting}=/tmp/startup PATH="$tmpdir:$PATH"'
    )
    resolver = _resolver(content)
    assert not tuple(resolver.generated_files())
