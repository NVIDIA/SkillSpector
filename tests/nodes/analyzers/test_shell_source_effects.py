# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cached/generated source identity survives only understood prior effects."""

import shlex

import pytest

from skillspector.artifacts import ArtifactDisposition, ContentKind, SecurityTextView
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner
from skillspector.shell_sources import (
    ShellSourceContext,
    ShellSourceResolver,
    generated_file_is_unmodified,
    source_is_unmodified,
)


def _resolver(content: str) -> ShellSourceResolver:
    return ShellSourceResolver(
        ShellSourceContext(
            "scripts/caller.sh",
            {
                "scripts/caller.sh": content,
                "scripts/library.sh": "helper() {\n printf ready\n}\n",
            },
        ),
        tm._next_shell_invocation_word,
        lambda text, start, end: tm._skip_command_substitution(text, start, end, lambda: None),
        tm._static_heredoc_header,
        tm._skip_static_heredoc_bodies,
        lambda: None,
    )


@pytest.mark.parametrize(
    "command",
    [
        "sed -n -e 'p' unrelated.txt -e 'w scripts/library.sh'",
        "sed -n 'p' unrelated.txt -i scripts/library.sh",
        "sed -n 'p' unrelated.txt -f program.sed",
        "sed -n 'p' unrelated.txt --expression='w scripts/library.sh'",
    ],
)
def test_trailing_sed_options_cannot_hide_writes(command: str) -> None:
    resolver = _resolver(command + "\n")
    assert not source_is_unmodified(resolver, "scripts/library.sh", len(resolver.content))


@pytest.mark.parametrize(
    "command",
    [
        "sed -n 'p' unrelated.txt",
        "sed -n -e 'p' first.txt second.txt",
        "sed -n -e 's/a/b/g' first.txt",
    ],
)
def test_all_literal_sed_inputs_keep_readonly_proof(command: str) -> None:
    resolver = _resolver(command + "\n")
    assert source_is_unmodified(resolver, "scripts/library.sh", len(resolver.content))


_GENERATED = """set -euo pipefail
tmpdir="$(mktemp -d /tmp/fixture.XXXXXX)"
cat > "$tmpdir/helper" <<'HELPER'
#!/bin/bash
printf ready
HELPER
CHMOD +x "$tmpdir/helper"
PATH="$tmpdir:$PATH" bash -c 'helper'
"""


@pytest.mark.parametrize("chmod", ["chmod", "/bin/chmod", "/usr/bin/chmod"])
def test_native_chmod_establishes_generated_executable(chmod: str) -> None:
    resolver = _resolver(_GENERATED.replace("CHMOD", chmod))
    (generated,) = resolver.generated_files()
    caller = resolver.commands[-1]
    assert generated_file_is_unmodified(resolver, generated, (caller,))


@pytest.mark.parametrize("chmod", ["/tmp/chmod", '"$tmpdir/chmod"'])
def test_chmod_basename_does_not_establish_generated_executable(chmod: str) -> None:
    resolver = _resolver(_GENERATED.replace("CHMOD", chmod))
    (generated,) = resolver.generated_files()
    assert not generated_file_is_unmodified(resolver, generated, (resolver.commands[-1],))


def test_shadowed_chmod_cannot_establish_generated_executable() -> None:
    content = _GENERATED.replace("CHMOD", "chmod").replace(
        "chmod +x", "chmod() { true; }\nchmod +x"
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


@pytest.mark.parametrize("mutate", [False, True])
def test_cached_cat_source_requires_unchanged_uploaded_bytes(mutate: bool) -> None:
    prefix = """set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"""
    if mutate:
        prefix += 'printf changed > "$root/scripts/library.sh"\n'
    content = prefix + 'program=$(cat "$root/scripts/library.sh")\nbash -c "$program; helper"\n'
    resolver = _resolver(content)
    value = resolver._name("program", 0, len(content))
    if mutate:
        assert value is None
    else:
        assert value is not None and value.kind == "code"
        assert value.text == "helper() {\n printf ready\n}"
        assert value.origins[0].path == "scripts/library.sh"


@pytest.mark.parametrize(
    "rebind, preserved",
    [
        ("", True),
        ("work=scripts/library.sh\n", False),
        (
            "work=scripts/library.sh\nexit 0\nwork=$(mktemp -d /tmp/fixture.XXXXXX)\n",
            False,
        ),
    ],
)
def test_child_exit_trap_uses_its_invocation_binding(rebind: str, preserved: bool) -> None:
    child = "set -e\nwork=$(mktemp -d /tmp/fixture.XXXXXX)\ntrap 'rm -rf \"$work\"' EXIT\n" + rebind
    child_resolver = _resolver(child)
    assert not source_is_unmodified(child_resolver, "scripts/library.sh", len(child))

    caller = (
        'set -e\nroot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\n'
        + "bash -c "
        + shlex.quote(child)
        + "\n"
        + 'program=$(cat "$root/scripts/library.sh")\nbash -c "$program; helper"\n'
    )
    library = "helper() {\n  rm -rf /\n}\n"
    prepared = tm.prepare_source_analysis(
        content=caller,
        file_path="scripts/caller.sh",
        file_type="shell",
        source_context=ShellSourceContext(
            "scripts/caller.sh", {"scripts/caller.sh": caller, "scripts/library.sh": library}
        ),
        check_runtime=lambda: None,
    )
    borrowed_library = False
    if prepared is not None:
        list(
            prepared.analyze_source_commands(
                content=caller,
                file_path="scripts/caller.sh",
                file_type="shell",
                source_view=SecurityTextView("raw", caller),
            )
        )
        borrowed_library = any(
            origin.path == "scripts/library.sh"
            for _, value in prepared.variants()
            for origin in value.origins
        )
    assert borrowed_library is preserved

    sources = {"scripts/caller.sh": caller, "scripts/library.sh": library}
    result = static_runner.run_static_patterns_with_ledger(
        {
            "components": ["scripts/caller.sh"],
            "file_cache": sources,
            "raw_file_cache": {path: text.encode() for path, text in sources.items()},
            "artifact_inventory": [
                {
                    "path": path,
                    "content_kind": ContentKind.TEXT,
                    "disposition": ArtifactDisposition.ANALYZED,
                    "decodable": True,
                    "contains_nul": False,
                    "size_bytes": len(text.encode()),
                }
                for path, text in sources.items()
            ],
        },
        [tm],
    )
    row = result["inspection_ledger"][0]
    if preserved:
        assert row["outcome"] is LedgerOutcome.COMPLETED
        assert any(finding.rule_id == "TM1" for finding in result["findings"])
    else:
        assert row["outcome"] is LedgerOutcome.PARTIAL
        assert row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
