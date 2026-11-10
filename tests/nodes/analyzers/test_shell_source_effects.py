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
    readonly_file_arguments,
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


@pytest.mark.parametrize(
    "arguments, preserved",
    [
        (("scripts/library.sh",), True),
        (("-bi", "--mime-type", "scripts/library.sh"), True),
        (("--", "-C"), True),
        (("-m", "-C", "scripts/library.sh"), True),
        (("--magic-file=-C", "scripts/library.sh"), True),
        (("-F", "--compile", "scripts/library.sh"), True),
        (("-C", "-m", "definitions"), False),
        (("--compile", "--magic-file=definitions"), False),
        (("scripts/library.sh", "-C"), False),
        (("-bC", "scripts/library.sh"), False),
        (("--comp", "scripts/library.sh"), False),
        (("-m", "definitions", "-C"), False),
        (("--magic-file",), False),
        (("--", None), False),
        ((None, "scripts/library.sh"), False),
        (("--uncompress", "scripts/library.sh"), False),
    ],
)
def test_file_inspection_options_cannot_hide_magic_compilation(arguments, preserved) -> None:
    assert readonly_file_arguments(arguments, lambda: None) is preserved
    if all(value is not None for value in arguments):
        content = "file " + shlex.join(arguments) + "\n"
        resolver = _resolver(content)
        assert source_is_unmodified(resolver, "scripts/library.sh", len(content)) is preserved


def test_file_inspection_proof_keeps_argument_bounds_and_cancellation() -> None:
    assert not readonly_file_arguments(("input",) * 129, lambda: None)
    assert not readonly_file_arguments(("x" * 65_537,), lambda: None)

    def cancel() -> None:
        raise TimeoutError("cancelled inspection proof")

    with pytest.raises(TimeoutError, match="cancelled inspection proof"):
        readonly_file_arguments(("-m", "definitions"), cancel)


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


@pytest.mark.parametrize(
    "command, target, preserved",
    [
        ("", "scripts/library.sh", True),
        ('file -b --mime-type "$script"', "scripts/library.sh", True),
        ("cd scripts\nrm -rf cache", "scripts/library.sh", True),
        ("tar -xf archive.tar", "scripts/library.sh", False),
        (
            'python3 -c \'open("scripts/library.sh","w").write("changed")\'',
            "scripts/library.sh",
            False,
        ),
        ('file -C -m "$root/scripts/library"', "library.mgc", False),
    ],
)
def test_cached_source_consumers_require_proven_prior_effects(
    command: str, target: str, preserved: bool
) -> None:
    library = "helper() {\n  printf ready\n}\n"
    caller = (
        'set -euo pipefail\nroot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\n'
        f'script="$root/{target}"\n'
        f"{command}\n"
        'program=$(cat "$script")\nbash -c "$program; helper"\nrm -rf /\n'
    )
    if command.startswith("cd "):
        assert not source_is_unmodified(_resolver(caller), target, caller.index("program="))
    sources = {
        "scripts/caller.sh": caller,
        target: library,
        "scripts/library": "0 string x example\n",
    }
    prepared = tm.prepare_source_analysis(
        content=caller,
        file_path="scripts/caller.sh",
        file_type="shell",
        source_context=ShellSourceContext("scripts/caller.sh", sources),
        check_runtime=lambda: None,
    )
    if preserved:
        assert prepared is not None
    variants = ()
    if prepared is not None:
        list(
            prepared.analyze_source_commands(
                content=caller,
                file_path="scripts/caller.sh",
                file_type="shell",
                source_view=SecurityTextView("raw", caller),
            )
        )
        variants = tuple(prepared.variants())
    borrowed = any(origin.path == target for _, value in variants for origin in value.origins)
    assert borrowed is preserved

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
    assert row["outcome"] is (LedgerOutcome.COMPLETED if preserved else LedgerOutcome.PARTIAL)
    if not preserved:
        assert row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
    assert any(
        finding.rule_id == "TM1"
        and str(finding.severity) == "HIGH"
        and finding.file == "scripts/caller.sh"
        and finding.start_line == len(caller.splitlines())
        and not finding.evidence.get("reconstructed_shell_source")
        for finding in result["findings"]
    )


def _inspect_source_caller(caller: str, extra_sources: dict[str, str]):
    sources = {"scripts/caller.sh": caller, **extra_sources}
    prepared = tm.prepare_source_analysis(
        content=caller,
        file_path="scripts/caller.sh",
        file_type="shell",
        source_context=ShellSourceContext("scripts/caller.sh", sources),
        check_runtime=lambda: None,
    )
    variants = ()
    if prepared is not None:
        list(
            prepared.analyze_source_commands(
                content=caller,
                file_path="scripts/caller.sh",
                file_type="shell",
                source_view=SecurityTextView("raw", caller),
            )
        )
        variants = tuple(value for _, value in prepared.variants())
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
    assert any(
        finding.rule_id == "TM1"
        and str(finding.severity) == "HIGH"
        and finding.file == "scripts/caller.sh"
        and finding.start_line == len(caller.splitlines())
        and not finding.evidence.get("reconstructed_shell_source")
        for finding in result["findings"]
    )
    return variants, result["inspection_ledger"][0]


@pytest.mark.parametrize("mode", ["root", "relative", "absolute", "without-alias"])
def test_producer_copy_respects_cwd_and_exact_source_origin(mode: str) -> None:
    producer = "set -euo pipefail\n"
    if mode != "root":
        producer += (
            'work="$(mktemp -d /tmp/producer.XXXXXX)"\n'
            'mkdir -p "$work/scripts"\n'
            'cp scripts/replacement.sh "$work/scripts/setup.sh"\ncd "$work"\n'
        )
    source = "$work/scripts/setup.sh" if mode == "absolute" else "scripts/setup.sh"
    producer += f'cp "{source}" "$1"\n'
    sources = {
        "scripts/producer.sh": producer,
        "scripts/replacement.sh": "printf generated\n",
    }
    if mode != "without-alias":
        sources["scripts/setup.sh"] = "printf uploaded\n"
    caller = (
        'set -euo pipefail\nroot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\n'
        'work="$(mktemp -d /tmp/caller.XXXXXX)"\noutput="$work/output.sh"\n'
        'bash "$root/scripts/producer.sh" "$output"\n'
        'program=$(cat "$output")\nbash -c "$program"\nrm -rf /\n'
    )
    variants, row = _inspect_source_caller(caller, sources)
    expected = "scripts/setup.sh" if mode == "root" else "scripts/replacement.sh"
    assert variants and {value.text for value in variants} == {sources[expected].rstrip("\n")}
    assert {origin.path for value in variants for origin in value.origins} == {expected}
    assert row["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize("operator", ["true ||", "false &&"])
@pytest.mark.parametrize("continuation", [" ", "\n", "\r\n", " \\\n", "\n# comment\n\n"])
def test_logical_continuation_cannot_publish_skipped_source_assignment(
    operator: str, continuation: str
) -> None:
    caller = (
        'set -euo pipefail\nroot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\n'
        'program=$(cat "$root/scripts/actual.sh")\n'
        + operator
        + continuation
        + 'program=$(cat "$root/scripts/decoy.sh")\n'
        'bash -c "$program"\nrm -rf /\n'
    )
    variants, row = _inspect_source_caller(
        caller, {"scripts/actual.sh": "printf actual\n", "scripts/decoy.sh": "printf decoy\n"}
    )
    assert not variants
    assert row["outcome"] is LedgerOutcome.PARTIAL
    assert row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize("operator", ["true ||", "false &&"])
def test_pending_logical_operator_does_not_taint_later_independent_assignment(
    operator: str,
) -> None:
    caller = (
        'set -euo pipefail\nroot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\n'
        'program=$(cat "$root/scripts/actual.sh")\n'
        + operator
        + '\nprogram=$(cat "$root/scripts/decoy.sh")\n'
        'other=$(cat "$root/scripts/actual.sh")\nbash -c "$other"\nrm -rf /\n'
    )
    variants, row = _inspect_source_caller(
        caller, {"scripts/actual.sh": "printf actual\n", "scripts/decoy.sh": "printf decoy\n"}
    )
    assert variants and {value.text for value in variants} == {"printf actual"}
    assert {origin.path for value in variants for origin in value.origins} == {"scripts/actual.sh"}
    assert row["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize(
    "operation, preserved",
    [
        ('mv -tscripts "$work/library.sh"', False),
        ('mv --target-directory=scripts "$work/library.sh"', False),
        ('mv -t scripts "$work/library.sh"', False),
        ('mv "$work/library.sh" "$work/other.sh"', True),
        ('mv -f "$work/library.sh" "$work/other.sh"', True),
        ('mv -- "$work/library.sh" "$work/other.sh"', True),
    ],
)
def test_writer_option_cannot_hide_cached_source_destination(
    operation: str, preserved: bool
) -> None:
    caller = (
        'set -euo pipefail\nroot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\n'
        'work="$(mktemp -d /tmp/source.XXXXXX)"\nprintf generated > "$work/library.sh"\n'
        + operation
        + '\nprogram=$(cat "$root/scripts/library.sh")\nbash -c "$program"\nrm -rf /\n'
    )
    variants, row = _inspect_source_caller(caller, {"scripts/library.sh": "printf uploaded\n"})
    assert bool(variants) is preserved
    assert row["outcome"] is (LedgerOutcome.COMPLETED if preserved else LedgerOutcome.PARTIAL)
    if preserved:
        assert {value.text for value in variants} == {"printf uploaded"}
        assert {origin.path for value in variants for origin in value.origins} == {
            "scripts/library.sh"
        }
    else:
        assert row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
