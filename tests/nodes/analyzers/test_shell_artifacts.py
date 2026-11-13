# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated code must retain its bytes and provenance across file operations."""

import pytest

from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.shell_artifact_flow import (
    ShellArtifactEffects,
    ShellProducer,
    helper_stdout_artifacts,
)
from skillspector.shell_artifacts import ShellArtifact, ShellArtifactStore
from skillspector.shell_helper_argv import ShellHelperOutput, reconstruct_helper_output
from skillspector.shell_sources import ShellSourceContext, ShellSourceOrigin, ShellSourceValue


def _program(text: str = "printf '%s\\n' ready\n") -> ShellArtifact:
    origin = ShellSourceOrigin("scripts/setup.sh", 0, len(text), "a" * 64)
    return ShellArtifact(ShellSourceValue(text, "code", (origin,)))


def test_effect_snapshot_changes_with_dependency_facts_but_not_repeated_invalidation() -> None:
    store = ShellArtifactStore(lambda: None)
    initial = store.effect_snapshot()
    assert store.create_empty_directory("/tmp/loop")
    empty = store.effect_snapshot()
    assert initial != empty
    assert store.exists("/tmp/loop/marker") is False
    store.invalidate("/tmp/loop/marker")
    unknown = store.effect_snapshot()
    assert unknown != empty and store.exists("/tmp/loop/marker") is None
    store.invalidate("/tmp/loop/marker")
    assert store.effect_snapshot() == unknown
    assert store.invalidate_decimal_path("/tmp/loop/log.", "")
    decimal = store.effect_snapshot()
    assert decimal != unknown
    assert store.invalidate_decimal_path("/tmp/loop/log.", "")
    assert store.effect_snapshot() == decimal
    store.invalidate(None)
    assert store.effect_snapshot() != decimal


def test_effect_snapshot_retains_prior_code_mode_and_exact_file_ownership() -> None:
    store = ShellArtifactStore(lambda: None)
    program = _program()
    assert store.write("/tmp/helper", program)
    prior = store.effect_snapshot()
    assert store.write("/tmp/helper", ShellArtifact(program.value, executable=True))
    executable = store.effect_snapshot()
    assert executable != prior
    assert store.write("/tmp/helper", _program("printf changed\n"))
    assert store.effect_snapshot() != executable
    left, right = ShellArtifactStore(lambda: None), ShellArtifactStore(lambda: None)
    assert left.create_empty_file("/tmp/file", _program("").value)
    assert right.write("/tmp/file", ShellArtifact(_program("").value, executable=False))
    assert left.read("/tmp/file") == right.read("/tmp/file")
    assert left.effect_snapshot() != right.effect_snapshot()


def test_fresh_file_owns_only_its_exact_path_and_retains_provenance() -> None:
    store = ShellArtifactStore(lambda: None)
    value = _program("").value
    assert store.create_empty_file("/tmp/credential", value)
    assert store.read("/tmp/credential") == ShellArtifact(value, executable=False)
    assert store.owns("/tmp/credential")
    assert not store.owns("/tmp")
    assert not store.owns("/tmp/credential/child")
    assert store.exists("/tmp/credential/child") is None
    assert store.inventory("/tmp/credential") is None
    assert store.write("/tmp/credential", _program())
    assert store.copy("/tmp/credential", "/tmp/copied")
    assert not store.owns("/tmp/copied")
    assert not store.create_empty_directory("/tmp/credential/child")
    assert not store.write("/tmp/credential/child", _program())


@pytest.mark.parametrize(
    "existing, directory",
    [
        ("/tmp/file", False),
        ("/tmp/file/child", False),
        ("/tmp", False),
        ("/tmp/file", True),
        ("/tmp/file/child", True),
    ],
)
def test_fresh_file_cannot_overwrite_an_existing_file_or_directory(
    existing: str, directory: bool
) -> None:
    store = ShellArtifactStore(lambda: None)
    if directory:
        assert store.create_empty_directory(existing)
    else:
        assert store.write(existing, _program())
    assert not store.create_empty_file("/tmp/file", _program("").value)
    if not directory:
        assert store.read(existing) == _program()


@pytest.mark.parametrize("path", ["/tmp/file", "/tmp", None])
def test_fresh_file_ownership_cannot_return_after_invalidation(path: str | None) -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_file("/tmp/file", _program("").value)
    store.invalidate(path)
    assert not store.owns("/tmp/file")
    assert not store.create_empty_file("/tmp/file", _program("").value)
    assert not store.write("/tmp/file", _program())


def test_fresh_file_decimal_invalidation_and_transactions_keep_ownership_boundaries() -> None:
    store = ShellArtifactStore(lambda: None)
    child = store.fork()
    assert child.create_empty_file("/tmp/log.123", _program("").value)
    assert not store.owns("/tmp/log.123")
    assert store.adopt(child)
    assert store.owns("/tmp/log.123")
    stale = store.fork()
    assert stale.create_empty_file("/tmp/second", _program("").value)
    assert store.invalidate_decimal_path("/tmp/log.", "")
    assert not store.owns("/tmp/log.123")
    assert not store.create_empty_file("/tmp/log.456", _program("").value)
    assert not store.adopt(stale)
    assert not store.owns("/tmp/second")


def test_fresh_files_share_existing_count_and_later_byte_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("skillspector.shell_artifacts.MAX_GENERATED_FILES", 2)
    store = ShellArtifactStore(lambda: None)
    assert store.write("uploaded", _program())
    assert store.create_empty_file("/tmp/file", _program("").value)
    assert not store.create_empty_file("/tmp/overflow", _program("").value)
    assert not store.owns("/tmp/overflow")
    monkeypatch.setattr("skillspector.shell_artifacts.MAX_GENERATED_BYTES", 3)
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_file("/tmp/file", _program("").value)
    assert not store.write("/tmp/file", _program("four"))
    assert not store.owns("/tmp/file")


def test_copy_archive_extract_retains_the_original_program_and_origin() -> None:
    store = ShellArtifactStore(lambda: None)
    program = _program()
    assert store.create_empty_directory("/tmp/build-owned")
    assert store.create_empty_directory("/tmp/extract-owned")
    assert store.write("scripts/setup.sh", program)
    assert store.copy("scripts/setup.sh", "/tmp/build-owned/bundle/setup.sh")
    assert store.archive("/tmp/build-owned", "/tmp/package.tgz")
    assert store.extract("/tmp/package.tgz", "/tmp/extract-owned")
    assert store.read("/tmp/extract-owned/bundle/setup.sh") == program


@pytest.mark.parametrize("with_file", [False, True])
def test_archive_roundtrip_retains_empty_directories_and_known_absence(with_file: bool) -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/input")
    assert store.create_empty_directory("/tmp/input/tree")
    assert store.create_empty_directory("/tmp/input/tree/empty/nested")
    assert store.create_empty_directory("/tmp/output")
    if with_file:
        assert store.write("/tmp/input/tree/runner", _program())
    assert store.archive("/tmp/input/tree", "/tmp/archive", member_prefix="tree")
    assert store.extract("/tmp/archive", "/tmp/output")
    for relative in ("tree", "tree/empty", "tree/empty/nested"):
        assert store.exists("/tmp/output/" + relative) is True
        assert store.read("/tmp/output/" + relative) is None
    assert store.exists("/tmp/output/tree/absent") is False
    assert (store.read("/tmp/output/tree/runner") is not None) is with_file


@pytest.mark.parametrize(
    "directories",
    [("../escape",), ("/absolute",), ("sub/../escape",), ("sub\\escape",), ("same", "same")],
)
def test_archive_directory_inventory_rejects_escapes_and_duplicates(directories) -> None:
    store = ShellArtifactStore(lambda: None)
    assert not store.write("archive", ShellArtifact(directories=directories))
    assert store.read("archive") is None


@pytest.mark.parametrize("directory", ["file", "file/child"])
def test_archive_directory_cannot_be_a_file_or_its_descendant(directory: str) -> None:
    store = ShellArtifactStore(lambda: None)
    assert not store.write(
        "archive", ShellArtifact(members=(("file", _program()),), directories=(directory,))
    )


def test_archive_directories_share_the_recursive_inventory_budget(monkeypatch) -> None:
    monkeypatch.setattr("skillspector.shell_artifacts.MAX_GENERATED_FILES", 3)
    store = ShellArtifactStore(lambda: None)
    child = ShellArtifact(directories=("empty",))
    assert store.write("at-limit", ShellArtifact(members=(("inner", child),)))
    assert not store.write(
        "over-limit", ShellArtifact(members=(("inner", child),), directories=("another",))
    )


@pytest.mark.parametrize(
    "member, relative", [("setup.sh", "setup.sh"), ("bundle", "bundle/setup.sh")]
)
def test_tar_file_and_directory_members_preserve_identity_and_prefix(
    member: str, relative: str
) -> None:
    store = ShellArtifactStore(lambda: None)
    effects = ShellArtifactEffects(store, lambda: None)
    program = ShellArtifact(_program().value, executable=True)
    assert store.create_empty_directory("/tmp/input")
    assert store.create_empty_directory("/tmp/output")
    assert store.write("/tmp/input/" + relative, program)
    assert effects.tar(("tar", "-cf", "/tmp/package.tar", "-C", "/tmp/input", member))
    assert store.read("/tmp/package.tar").members == ((relative, program),)
    assert effects.tar(("tar", "-xf", "/tmp/package.tar", "-C", "/tmp/output"))
    assert store.read("/tmp/output/" + relative) is program


@pytest.mark.parametrize("prefix", ["../setup.sh", "/setup.sh", "sub/../setup.sh", "sub\\setup.sh"])
def test_exact_file_archive_rejects_invalid_member_prefix(prefix: str) -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/input")
    assert store.write("/tmp/input/setup.sh", _program())
    assert store.write("/tmp/package.tar", _program())
    assert not store.archive("/tmp/input/setup.sh", "/tmp/package.tar", member_prefix=prefix)
    assert store.read("/tmp/package.tar") is None


@pytest.mark.parametrize("levels, complete", [(2, True), (8, True), (9, False)])
def test_exact_file_archive_keeps_nested_depth_bound(levels: int, complete: bool) -> None:
    store = ShellArtifactStore(lambda: None)
    effects = ShellArtifactEffects(store, lambda: None)
    program = _program()
    assert store.create_empty_directory("/tmp/input")
    assert store.write("/tmp/input/setup.sh", program)
    member = "setup.sh"
    for level in range(1, levels + 1):
        path = f"/tmp/input/layer{level}.tar"
        accepted = effects.tar(("tar", "-cf", path, "-C", "/tmp/input", member))
        assert accepted is (complete or level < levels)
        member = f"layer{level}.tar"
    if complete:
        for level in range(levels, 0, -1):
            destination = f"/tmp/output{level}"
            assert store.create_empty_directory(destination)
            assert effects.tar(("tar", "-xf", path, "-C", destination))
            path = destination + (f"/layer{level - 1}.tar" if level > 1 else "/setup.sh")
        assert store.read(path) is program
    else:
        assert store.read(path) is None


@pytest.mark.parametrize(
    "limit, allowance, complete",
    [
        ("MAX_GENERATED_FILES", 1, False),
        ("MAX_GENERATED_FILES", 2, True),
        ("MAX_GENERATED_BYTES", 7, False),
        ("MAX_GENERATED_BYTES", 8, True),
    ],
)
def test_exact_file_archive_keeps_count_and_byte_budgets(
    monkeypatch: pytest.MonkeyPatch, limit: str, allowance: int, complete: bool
) -> None:
    monkeypatch.setattr("skillspector.shell_artifacts." + limit, allowance)
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/input")
    assert store.write("/tmp/input/setup.sh", _program("1234"))
    assert (
        store.archive("/tmp/input/setup.sh", "/tmp/package.tar", member_prefix="setup.sh")
        is complete
    )
    assert (store.read("/tmp/package.tar") is not None) is complete


@pytest.mark.parametrize("write_path", ["/tmp/build-owned/bundle/setup.sh", "/tmp/build-owned"])
def test_unknown_write_before_packaging_cannot_reuse_original_bytes(write_path: str) -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/build-owned")
    assert store.write("/tmp/build-owned/bundle/setup.sh", _program())
    store.invalidate(write_path)
    assert not store.archive("/tmp/build-owned", "/tmp/package.tgz")
    assert store.read("/tmp/package.tgz") is None


def test_later_source_mutation_does_not_change_an_already_created_archive() -> None:
    store = ShellArtifactStore(lambda: None)
    original = _program()
    assert store.create_empty_directory("/tmp/build-owned")
    assert store.create_empty_directory("/tmp/extract-owned")
    assert store.write("/tmp/build-owned/setup.sh", original)
    assert store.archive("/tmp/build-owned", "/tmp/package.tgz")
    assert store.write("/tmp/build-owned/setup.sh", _program("rm -rf /\n"))
    assert store.extract("/tmp/package.tgz", "/tmp/extract-owned")
    assert store.read("/tmp/extract-owned/setup.sh") == original


@pytest.mark.parametrize("name", ["../setup.sh", "/setup.sh", "sub/../setup.sh", "sub\\setup.sh"])
def test_archive_member_escape_never_establishes_generated_code(name: str) -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/extract-owned")
    assert not store.write("/tmp/package.tgz", ShellArtifact(members=((name, _program()),)))
    assert not store.extract("/tmp/package.tgz", "/tmp/extract-owned")
    assert store.read("/tmp/extract-owned/setup.sh") is None


def test_duplicate_members_refuse_ambiguous_extraction() -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/extract-owned")
    archive = ShellArtifact(
        members=(("setup.sh", _program()), ("setup.sh", _program("rm -rf /\n")))
    )
    assert not store.write("/tmp/package.tgz", archive)
    assert not store.extract("/tmp/package.tgz", "/tmp/extract-owned")
    assert store.read("/tmp/extract-owned/setup.sh") is None


def test_extraction_requires_a_fresh_known_destination() -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.write("/tmp/package.tgz", ShellArtifact(members=(("setup.sh", _program()),)))
    assert not store.extract("/tmp/package.tgz", "/tmp/existing")
    assert store.read("/tmp/existing/setup.sh") is None


@pytest.mark.parametrize(
    "paths", [("root/file", "root/file/child"), ("root/file/child", "root/file")]
)
def test_file_directory_prefix_collisions_refuse_the_snapshot(paths: tuple[str, str]) -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.write(paths[0], _program())
    assert not store.write(paths[1], _program())
    assert store.read(paths[0]) is None


def test_declared_directories_cannot_be_replaced_by_fictitious_files() -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("root/child")
    assert not store.write("root", _program())
    assert store.read("root") is None


def test_fresh_directory_cannot_be_declared_beneath_a_known_file() -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.write("root/file", _program())
    assert not store.create_empty_directory("root/file/child")


def test_an_existing_directory_cannot_be_redeclared_fresh() -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("root")
    assert not store.create_empty_directory("root")


@pytest.mark.parametrize("names", [("file", "file/child"), ("file/child", "file")])
def test_archive_file_directory_collisions_are_rejected(names: tuple[str, str]) -> None:
    store = ShellArtifactStore(lambda: None)
    archive = ShellArtifact(members=tuple((name, _program()) for name in names))
    assert not store.write("package.tgz", archive)


def test_recursive_inventory_shares_one_count_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("skillspector.shell_artifacts.MAX_GENERATED_FILES", 4)
    leaf = ShellArtifact(members=(("one.sh", _program("")), ("two.sh", _program(""))))
    archive = ShellArtifact(members=(("one.tgz", leaf), ("two.tgz", leaf)))
    store = ShellArtifactStore(lambda: None)
    assert not store.write("package.tgz", archive)


def test_missing_copy_source_invalidates_previously_known_destination() -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.write("/tmp/setup.sh", _program())
    assert not store.copy("/tmp/unknown", "/tmp/setup.sh")
    assert store.read("/tmp/setup.sh") is None


def test_unknown_destination_invalidates_all_previous_proofs() -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.write("scripts/setup.sh", _program())
    store.invalidate(None)
    assert store.read("scripts/setup.sh") is None
    assert not store.write("scripts/later.sh", _program())


def test_file_budget_cannot_leave_a_trusted_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("skillspector.shell_artifacts.MAX_GENERATED_FILES", 1)
    store = ShellArtifactStore(lambda: None)
    assert store.write("one.sh", _program())
    assert not store.write("two.sh", _program())
    assert store.read("one.sh") is None


def test_byte_budget_includes_copies_and_archive_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("skillspector.shell_artifacts.MAX_GENERATED_BYTES", 8)
    store = ShellArtifactStore(lambda: None)
    assert store.write("one.sh", _program("1234"))
    assert store.copy("one.sh", "two.sh")
    assert not store.copy("one.sh", "three.sh")
    assert store.read("one.sh") is None


def test_shared_cancellation_is_not_converted_into_success() -> None:
    def cancel() -> None:
        raise TimeoutError("scan deadline")

    store = ShellArtifactStore(cancel)
    with pytest.raises(TimeoutError, match="scan deadline"):
        store.write("setup.sh", _program())


def test_typed_helper_archive_survives_copy_into_a_new_bundle() -> None:
    store = ShellArtifactStore(lambda: None)
    effects = ShellArtifactEffects(store, lambda: None)
    assert store.create_empty_directory("/tmp/input")
    assert store.create_empty_directory("/tmp/work")
    assert store.create_empty_directory("/tmp/output")
    assert store.write("/tmp/input/tool/flash", _program("#!/bin/bash\n"))
    assert effects.tar(
        ("tar", "-I", "lbzip2", "-cf", "/tmp/input.tbz2", "-C", "/tmp/input", "tool")
    )
    output = helper_stdout_artifacts(
        ((ShellHelperOutput(path="/tmp/input.tbz2"),),), store, (), lambda: None
    )
    assert output is not None
    assert effects.redirect_stdout("/tmp/work/download.tbz2", output)
    assert effects.tar(
        ("tar", "-I", "lbzip2", "-xf", "/tmp/work/download.tbz2", "-C", "/tmp/work/bundle/tools")
    )
    original = _program()
    assert store.write("scripts/setup.sh", original)
    assert effects.copy_file(("cp", "scripts/setup.sh", "/tmp/work/bundle/setup.sh"))
    assert effects.tar(("tar", "-I", "pigz", "-cf", "/tmp/bundle.tgz", "-C", "/tmp/work", "bundle"))
    assert effects.tar(("tar", "-I", "pigz", "-xf", "/tmp/bundle.tgz", "-C", "/tmp/output"))
    assert store.read("/tmp/output/bundle/setup.sh") == original
    assert store.read("/tmp/output/bundle/tools/tool/flash") is not None


@pytest.mark.parametrize(
    "pieces",
    [
        (ShellHelperOutput(text="prefix"), ShellHelperOutput(path="/tmp/archive.tgz")),
        (ShellHelperOutput(path="/tmp/archive.tgz"), ShellHelperOutput(text="suffix")),
        (ShellHelperOutput(path="/tmp/archive.tgz"), ShellHelperOutput(path="/tmp/archive.tgz")),
        (ShellHelperOutput(path="/tmp/missing.tgz"),),
    ],
)
def test_stdout_cannot_turn_a_modified_or_unknown_stream_into_an_archive(pieces) -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.write("/tmp/archive.tgz", ShellArtifact(members=(("setup.sh", _program()),)))
    assert helper_stdout_artifacts((pieces,), store, (), lambda: None) is None


def test_ambiguous_helper_output_invalidates_its_destination() -> None:
    store = ShellArtifactStore(lambda: None)
    effects = ShellArtifactEffects(store, lambda: None)
    assert store.write("/tmp/output", _program())
    assert not effects.redirect_stdout("/tmp/output", (_program("one"), _program("two")))
    assert store.read("/tmp/output") is None


@pytest.mark.parametrize(
    "argv",
    [
        ("tar", "-I", "malicious-compressor", "-cf", "/tmp/archive", "-C", "/tmp/root", "folder"),
        ("tar", "-cf", "/tmp/archive", "-C", "/tmp/root", "../folder"),
        ("tar", "-xf", "/tmp/archive", "-C", "/tmp/root", "--overwrite"),
    ],
)
def test_unmodeled_tar_arguments_never_create_proven_code(argv) -> None:
    store = ShellArtifactStore(lambda: None)
    assert not ShellArtifactEffects(store, lambda: None).tar(argv)


_PRODUCER = """set -Eeuo pipefail
TOOL_SPEC=//depot/tools.tbz2
HELPER_BASE=//depot/helpers
[[ $# -eq 1 ]] || { echo "Usage: $0 OUTPUT.tgz" >&2; exit 2; }
output="$1"
[[ "$output" == *.tgz ]] || { echo invalid >&2; exit 2; }
[[ ! -e "$output" ]] || { echo exists >&2; exit 1; }
command -v p4 >/dev/null
command -v pigz >/dev/null
command -v lbzip2 >/dev/null
resolve_one() {
  local spec="$1" result
  local -a matches=()
  result="$(p4 files "$spec")"
  mapfile -t matches < <(awk 'NF {print $1}' <<<"$result")
  [[ ${#matches[@]} -eq 1 ]] || {
    echo "Expected one match; found ${#matches[@]}" >&2
    return 1
  }
  printf '%s\\n' "${matches[0]}"
}
umask 077
work="$(mktemp -d /tmp/producer.XXXXXX)"
trap 'rm -rf "$work"' EXIT
bundle_root="$work/bundle"
root="$bundle_root/tools"
mkdir -p "$root/helpers"
cp "$(dirname "${BASH_SOURCE[0]}")/setup.sh" "$bundle_root/setup.sh"
chmod 700 "$bundle_root/setup.sh"
tool_depot="$(resolve_one "$TOOL_SPEC")"
archive="$work/tool.tbz2"
p4 print -q "$tool_depot" > "$archive"
tar -I lbzip2 -xf "$archive" -C "$root"
revisions=("$tool_depot")
for file in first.sh second.py third.py; do
  depot="$(resolve_one "$HELPER_BASE/$file")"
  p4 print -q "$depot" > "$root/helpers/$file"
  revisions+=("$depot")
done
chmod 700 "$root/helpers/first.sh"
mapfile -t tools < <(find "$root" -type f -name FlashTool -print)
[[ ${#tools[@]} -eq 1 && -s "${tools[0]}" ]] || {
  echo "Expected one nonempty FlashTool" >&2
  exit 1
}
chmod 700 "${tools[0]}"
printf '%s\\n' "${revisions[@]}" > "$root/REVISIONS"
(
  cd "$root"
  find . -type f ! -name MANIFEST.sha256 -print0 |
    sort -z |
    xargs -0 sha256sum > MANIFEST.sha256
)
tar -I pigz -cf "$output" -C "$work" bundle
chmod 600 "$output"
printf 'Created %s\\n' "$output"
"""


def _inspect_producer(program=_PRODUCER, *, modes=("one",)):
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/result")
    assert store.write("scripts/setup.sh", _program())
    archive = ShellArtifact(members=(("tool/FlashTool", _program("#!/bin/bash\n")),))

    def helper_output(argv, environment):
        if argv[1] == "files":
            name = argv[2].rpartition("/")[2]
            candidates = {
                "zero": "",
                "one": "//depot/a/" + name + "#2 - add\n",
                "two": "//depot/a/" + name + "#1 - add\n//depot/b/" + name + "#2 - add\n",
            }
            return tuple(ShellArtifact(ShellSourceValue(candidates[mode])) for mode in modes)
        assert argv[1:3] == ("print", "-q")
        return (archive,) if "tools.tbz2" in argv[3] else (_program("#!/bin/bash\n"),)

    context = ShellSourceContext(
        "scripts/producer.sh",
        {"scripts/producer.sh": program, "scripts/setup.sh": _program().value.text},
    )
    producer = ShellProducer(
        store, context, tm._next_shell_invocation_word, lambda: None, helper_output
    )
    return (
        producer.inspect(
            ("/tmp/result/package.tgz",), {}, ("/tmp/result/package.tgz",), initial_cwd="."
        ),
        store,
    )


def test_finite_producer_preserves_setup_copy_and_archive_member_prefixes() -> None:
    result, store = _inspect_producer()
    assert result is not None
    assert store.create_empty_directory("/tmp/extracted")
    assert store.extract("/tmp/result/package.tgz", "/tmp/extracted")
    copied = store.read("/tmp/extracted/bundle/setup.sh")
    assert copied is not None and copied.value == _program().value and copied.executable is True
    assert store.read("/tmp/extracted/bundle/tools/tool/FlashTool") is not None
    assert all(
        path.startswith(("/tmp/producer.__skillspector_producer_", "/tmp/result/"))
        for path in result.writes
    )


@pytest.mark.parametrize(
    "invocation, expected",
    [
        ("printf %s changed", "changed"),
        ("printf %s changed >/dev/null", ""),
        ("printf %s changed >>/dev/null", ""),
        ("emit", "changed"),
        ("emit >/dev/null", ""),
    ],
)
def test_producer_capture_never_retains_discarded_stdout(invocation: str, expected: str) -> None:
    program = (
        'set -euo pipefail\nemit() { printf %s changed; }\ncode="$('
        + invocation
        + ')"\nprintf %s "$code" > "$1"\n'
    )
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/result")
    producer = ShellProducer(
        store,
        ShellSourceContext("producer.sh", {"producer.sh": program}),
        tm._next_shell_invocation_word,
        lambda: None,
        None,
    )
    result = producer.inspect(("/tmp/result/output.sh",), {}, ("/tmp/result/output.sh",))
    assert result is not None
    assert store.read("/tmp/result/output.sh").value.text == expected


@pytest.mark.parametrize(
    "redirects", [">/dev/null > /tmp/result/extra", "> /tmp/result/extra >/dev/null"]
)
def test_producer_refuses_unmodeled_ordered_stdout_routes(redirects: str) -> None:
    program = (
        'set -euo pipefail\ncode="$(printf %s changed '
        + redirects
        + ')"\nprintf %s "$code" > "$1"\n'
    )
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/result")
    producer = ShellProducer(
        store,
        ShellSourceContext("producer.sh", {"producer.sh": program}),
        tm._next_shell_invocation_word,
        lambda: None,
        None,
    )
    assert producer.inspect(("/tmp/result/output.sh",), {}, ("/tmp/result/output.sh",)) is None


@pytest.mark.parametrize(
    "redirects, complete", [('> "$1"', True), ('> "$1" 2>&1', False), ('2>&1 > "$1"', False)]
)
def test_producer_never_claims_merged_helper_stderr_is_stdout_only(
    redirects: str, complete: bool
) -> None:
    helper_text = 'printf "%s\\n" "rm -rf /" >&2\nprintf "%s\\n" "printf ready"\n'
    program = "set -euo pipefail\np4 print " + redirects + "\n"
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/result")

    def helper(argv, environment):
        variants = reconstruct_helper_output(
            helper_text,
            argv[1:],
            environment,
            frozenset(),
            tm._next_shell_invocation_word,
            lambda: None,
        )
        assert variants is not None
        return helper_stdout_artifacts(
            variants, store, _program(helper_text).value.origins, lambda: None
        )

    producer = ShellProducer(
        store,
        ShellSourceContext("producer.sh", {"producer.sh": program}),
        tm._next_shell_invocation_word,
        lambda: None,
        helper,
    )
    result = producer.inspect(("/tmp/result/output.sh",), {}, ("/tmp/result/output.sh",))
    assert (result is not None) is complete
    if complete:
        assert store.read("/tmp/result/output.sh").value.text == "printf ready\n"
    else:
        assert store.read("/tmp/result/output.sh") is None


@pytest.mark.parametrize(
    "invocation, expected",
    [
        ("( printf %s two )", "two"),
        ("printf %s one; ( printf %s two ); printf %s three", "onetwothree"),
        ("( ( printf %s two ) )", "two"),
    ],
)
def test_producer_capture_retains_nested_subshell_stdout_in_order(
    invocation: str, expected: str
) -> None:
    program = 'set -euo pipefail\nmode="$(' + invocation + ')"\np4 "$mode" > "$1"\n'
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/result")

    def helper(argv, environment):
        return (_program(argv[1]),)

    producer = ShellProducer(
        store,
        ShellSourceContext("producer.sh", {"producer.sh": program}),
        tm._next_shell_invocation_word,
        lambda: None,
        helper,
    )
    assert producer.inspect(("/tmp/result/output.sh",), {}, ("/tmp/result/output.sh",)) is not None
    assert store.read("/tmp/result/output.sh").value.text == expected


def test_producer_subshell_keeps_local_bindings_separate_from_its_output() -> None:
    program = 'set -euo pipefail\nmode=outer\ncaptured="$( ( mode=two; printf %s "$mode" ) )"\nprintf "%s|%s" "$captured" "$mode" > "$1"\n'
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/result")
    producer = ShellProducer(
        store,
        ShellSourceContext("producer.sh", {"producer.sh": program}),
        tm._next_shell_invocation_word,
        lambda: None,
        None,
    )
    assert producer.inspect(("/tmp/result/output.sh",), {}, ("/tmp/result/output.sh",)) is not None
    assert store.read("/tmp/result/output.sh").value.text == "two|outer"


@pytest.mark.parametrize(
    "definition, complete",
    [
        ("emit() { printf %s known; }", True),
        ("( emit() { printf %s known; } )", False),
    ],
)
def test_producer_subshell_function_definition_cannot_escape(
    definition: str, complete: bool
) -> None:
    program = "set -euo pipefail\n" + definition + '\nmode="$(emit)"\nprintf %s "$mode" > "$1"\n'
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/result")
    producer = ShellProducer(
        store,
        ShellSourceContext("producer.sh", {"producer.sh": program}),
        tm._next_shell_invocation_word,
        lambda: None,
        None,
    )
    assert (
        producer.inspect(("/tmp/result/output.sh",), {}, ("/tmp/result/output.sh",)) is not None
    ) is complete
    if complete:
        assert store.read("/tmp/result/output.sh").value.text == "known"
    else:
        assert store.read("/tmp/result/output.sh") is None


@pytest.mark.parametrize(
    "installation, complete",
    [
        ("trap 'rm -rf \"$temporary\"' EXIT", True),
        ("( trap 'rm -rf \"$temporary\"' EXIT )", False),
        ('captured="$(trap \'rm -rf "$temporary"\' EXIT)"', False),
    ],
)
def test_producer_child_exit_trap_cannot_mutate_parent_cleanup(
    installation: str, complete: bool
) -> None:
    program = (
        'set -euo pipefail\ntemporary="$(mktemp -d /tmp/producer.XXXXXX)"\n'
        + installation
        + '\nprintf %s known > "$1"\n'
    )
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/result")
    producer = ShellProducer(
        store,
        ShellSourceContext("producer.sh", {"producer.sh": program}),
        tm._next_shell_invocation_word,
        lambda: None,
        None,
    )
    assert (
        producer.inspect(("/tmp/result/output.sh",), {}, ("/tmp/result/output.sh",)) is not None
    ) is complete
    if complete:
        assert store.read("/tmp/result/output.sh").value.text == "known"
    else:
        assert store.read("/tmp/result/output.sh") is None


@pytest.mark.parametrize(
    "change, complete",
    [
        ("", True),
        ('temporary="$sibling"', True),
        ("temporary=/tmp/result", False),
        ("temporary=/tmp/result\ntrap 'rm -rf \"$sibling\"' EXIT", True),
    ],
)
def test_producer_exit_trap_uses_final_variable_and_last_installation(
    change: str, complete: bool
) -> None:
    program = (
        'set -euo pipefail\ntemporary="$(mktemp -d /tmp/producer.XXXXXX)"\nsibling="$(mktemp -d /tmp/sibling.XXXXXX)"\ntrap \'rm -rf "$temporary"\' EXIT\n'
        + change
        + '\nprintf %s known > "$1"\n'
    )
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/result")
    producer = ShellProducer(
        store,
        ShellSourceContext("producer.sh", {"producer.sh": program}),
        tm._next_shell_invocation_word,
        lambda: None,
        None,
    )
    assert (
        producer.inspect(("/tmp/result/output.sh",), {}, ("/tmp/result/output.sh",)) is not None
    ) is complete
    if complete:
        assert store.read("/tmp/result/output.sh").value.text == "known"
    else:
        assert store.read("/tmp/result/output.sh") is None


@pytest.mark.parametrize(
    "body, declaration, operand",
    [
        ("mode=bad; printf %s x", "local mode=good; ", '"$mode"'),
        ("items=(bad); printf %s x", "local -a items=(); ", '"${items[0]}"'),
        (
            "mapfile -t items < <(awk 'NF {print $1}' <<<\"$listing\"); printf %s x",
            "local -a items=(); ",
            '"${items[0]}"',
        ),
        ("for mode in bad; do printf %s x; done", "local mode=good; ", '"$mode"'),
    ],
)
@pytest.mark.parametrize("local", [False, True])
def test_capture_function_requires_mutations_to_be_declared_local(
    body: str, declaration: str, operand: str, local: bool
) -> None:
    program = (
        "set -euo pipefail\nmode=good\nitems=(good)\nlisting=bad\nchoose() { "
        + (declaration if local else "")
        + body
        + '; }\ncaptured="$(choose; printf %s '
        + operand
        + ')"\nprintf %s "$captured" > "$1"\n'
    )
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/result")
    producer = ShellProducer(
        store,
        ShellSourceContext("producer.sh", {"producer.sh": program}),
        tm._next_shell_invocation_word,
        lambda: None,
        None,
    )
    assert (
        producer.inspect(("/tmp/result/output.sh",), {}, ("/tmp/result/output.sh",)) is not None
    ) is local
    if local:
        assert store.read("/tmp/result/output.sh").value.text == "xgood"
    else:
        assert store.read("/tmp/result/output.sh") is None


def test_capture_function_cannot_bypass_file_effect_guard_via_manifest_pipeline() -> None:
    program = 'set -euo pipefail\ncd /tmp/result\nemit() { find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256; }\ncaptured="$(emit)"\nprintf %s "$captured" > "$1"\n'
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/result")
    producer = ShellProducer(
        store,
        ShellSourceContext("producer.sh", {"producer.sh": program}),
        tm._next_shell_invocation_word,
        lambda: None,
        None,
    )
    assert producer.inspect(("/tmp/result/output.sh",), {}, ("/tmp/result/output.sh",)) is None
    assert store.read("/tmp/result/output.sh") is None


@pytest.mark.parametrize(
    "index, expected", [("0", "zero"), ("8", "eight"), ("10", "ten"), ("010", None), ("08", None)]
)
def test_producer_array_index_does_not_treat_shell_octal_as_decimal(
    index: str, expected: str | None
) -> None:
    program = (
        'set -euo pipefail\nitems=(zero one two three four five six seven eight nine ten)\nprintf %s "${items['
        + index
        + ']}" > "$1"\n'
    )
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/result")
    producer = ShellProducer(
        store,
        ShellSourceContext("producer.sh", {"producer.sh": program}),
        tm._next_shell_invocation_word,
        lambda: None,
        None,
    )
    assert (
        producer.inspect(("/tmp/result/output.sh",), {}, ("/tmp/result/output.sh",)) is not None
    ) is (expected is not None)
    if expected is not None:
        assert store.read("/tmp/result/output.sh").value.text == expected
    else:
        assert store.read("/tmp/result/output.sh") is None


@pytest.mark.parametrize(
    "number, complete", [("0", True), ("8", True), ("10", True), ("010", False), ("08", False)]
)
def test_producer_numeric_guard_refuses_unproved_arithmetic_base(
    number: str, complete: bool
) -> None:
    program = (
        "set -euo pipefail\n[[ "
        + number
        + " -eq "
        + number
        + ' ]] || { exit 1; }\nprintf %s known > "$1"\n'
    )
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/result")
    producer = ShellProducer(
        store,
        ShellSourceContext("producer.sh", {"producer.sh": program}),
        tm._next_shell_invocation_word,
        lambda: None,
        None,
    )
    assert (
        producer.inspect(("/tmp/result/output.sh",), {}, ("/tmp/result/output.sh",)) is not None
    ) is complete
    if not complete:
        assert store.read("/tmp/result/output.sh") is None


@pytest.mark.parametrize(
    "operand, expected",
    [
        ("'${#items[@]}'", "${#items[@]}"),
        (r'"\${#items[@]}"', "${#items[@]}"),
        ('"${#items[@]}"', "1"),
        ("${#items[@]}", "1"),
        ('"${items[0]}"', "two words"),
    ],
)
def test_producer_array_words_respect_literal_and_quoted_boundaries(
    operand: str, expected: str
) -> None:
    program = 'set -euo pipefail\nitems=("two words")\nprintf %s ' + operand + ' > "$1"\n'
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/result")
    producer = ShellProducer(
        store,
        ShellSourceContext("producer.sh", {"producer.sh": program}),
        tm._next_shell_invocation_word,
        lambda: None,
        None,
    )
    assert producer.inspect(("/tmp/result/output.sh",), {}, ("/tmp/result/output.sh",)) is not None
    assert store.read("/tmp/result/output.sh").value.text == expected


@pytest.mark.parametrize("operand", ["${items[0]}", '"prefix${#items[@]}"'])
def test_producer_unproved_array_splitting_or_embedding_refuses(operand: str) -> None:
    program = 'set -euo pipefail\nitems=("two words *")\nprintf %s ' + operand + ' > "$1"\n'
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/result")
    producer = ShellProducer(
        store,
        ShellSourceContext("producer.sh", {"producer.sh": program}),
        tm._next_shell_invocation_word,
        lambda: None,
        None,
    )
    assert producer.inspect(("/tmp/result/output.sh",), {}, ("/tmp/result/output.sh",)) is None
    assert store.read("/tmp/result/output.sh") is None


@pytest.mark.parametrize(
    "listing, expected",
    [
        (" first\tignored\n\tsecond rest\n \t\n", "first\nsecond\n"),
        ("first\u00a0second\n", None),
        ("first\u2028second\n", None),
        ("first\rsecond\n", None),
        ("first\r\n", None),
    ],
)
def test_producer_awk_fields_do_not_use_python_unicode_whitespace(
    listing: str, expected: str | None
) -> None:
    program = 'set -euo pipefail\nlisting="$(p4 files)"\nmapfile -t rows < <(awk \'NF {print $1}\' <<<"$listing")\nprintf "%s\\n" "${rows[@]}" > "$1"\n'
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/result")

    def helper(argv, environment):
        return (_program(listing),)

    producer = ShellProducer(
        store,
        ShellSourceContext("producer.sh", {"producer.sh": program}),
        tm._next_shell_invocation_word,
        lambda: None,
        helper,
    )
    result = producer.inspect(("/tmp/result/output.sh",), {}, ("/tmp/result/output.sh",))
    assert (result is not None) is (expected is not None)
    if expected is not None:
        assert store.read("/tmp/result/output.sh").value.text == expected
    else:
        assert store.read("/tmp/result/output.sh") is None


def test_producer_helper_environment_retains_only_current_exported_names() -> None:
    program = 'set -euo pipefail\nEXPORTED=changed\nLOCAL=private\np4 print > "$1"\n'
    environments = []

    def helper(argv, environment):
        environments.append(environment)
        return (_program(environment["EXPORTED"]),)

    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/result")
    producer = ShellProducer(
        store,
        ShellSourceContext("producer.sh", {"producer.sh": program}),
        tm._next_shell_invocation_word,
        lambda: None,
        helper,
    )
    assert (
        producer.inspect(
            ("/tmp/result/output.sh",), {"EXPORTED": "original"}, ("/tmp/result/output.sh",)
        )
        is not None
    )
    assert environments == [{"EXPORTED": "changed"}]
    assert store.read("/tmp/result/output.sh").value.text == "changed"


def test_failed_zero_and_multiple_match_branches_do_not_supply_fictitious_outputs() -> None:
    result, _ = _inspect_producer(modes=("zero", "two", "one"))
    assert result is not None
    result, store = _inspect_producer(modes=("zero", "two"))
    assert result is None
    assert store.read("/tmp/result/package.tgz") is None


@pytest.mark.parametrize(
    "mutation",
    [
        ('cp "$(dirname', 'eval "danger"\ncp "$(dirname'),
        ("tar -I lbzip2 -xf", "tar -P -I lbzip2 -xf"),
        ('mkdir -p "$root/helpers"', 'mkdir -p "$root/helpers"\nln -s /outside "$root/escape"'),
        ('local spec="$1" result', 'local spec="$1" result\ncp scripts/setup.sh /tmp/hidden-write'),
        ("set -Eeuo pipefail", "set +e"),
        ("resolve_one() {", "mapfile() { printf misleading; }\nresolve_one() {"),
        ("resolve_one() {", "local() { printf misleading; }\nresolve_one() {"),
        ("umask 077", "change_output() { output=/tmp/other.tgz; }\nchange_output\numask 077"),
        ("resolve_one() {", "nested() { printf misleading; }\nresolve_one() {\n nested"),
    ],
)
def test_unmodeled_producer_effect_refuses_all_outputs(mutation) -> None:
    result, store = _inspect_producer(_PRODUCER.replace(*mutation))
    assert result is None
    assert store.read("/tmp/result/package.tgz") is None


@pytest.mark.parametrize(
    "name",
    [
        "dirname",
        "awk",
        "find",
        "sort",
        "xargs",
        "sha256sum",
        "gzip",
        "pigz",
        "bzip2",
        "lbzip2",
        "xz",
        "rm",
    ],
)
def test_specialized_producer_operations_cannot_bypass_shadowed_native_names(name) -> None:
    # These names occur within substitutions, process substitutions, pipelines,
    # tar options or the cleanup trap rather than only as ordinary commands.
    program = _PRODUCER.replace(
        "resolve_one() {", name + "() { printf replaced; }\nresolve_one() {"
    )
    result, store = _inspect_producer(program)
    assert result is None
    assert store.read("/tmp/result/package.tgz") is None


def test_optional_producer_failure_cannot_invalidate_the_original_snapshot() -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.write("scripts/setup.sh", _program())
    child = store.fork()
    child.invalidate(None)
    assert not store.adopt(child)
    assert store.read("scripts/setup.sh") == _program()


def test_successful_snapshot_can_only_be_adopted_by_its_parent_once() -> None:
    store = ShellArtifactStore(lambda: None)
    child = store.fork()
    assert child.write("generated.sh", _program())
    assert store.read("generated.sh") is None
    assert not ShellArtifactStore(lambda: None).adopt(child)
    assert store.adopt(child)
    assert store.read("generated.sh") == _program()
    assert not store.adopt(child)


def test_adopting_a_stale_producer_snapshot_cannot_erase_later_writes() -> None:
    store = ShellArtifactStore(lambda: None)
    child = store.fork()
    assert child.write("first.sh", _program())
    assert store.write("later.sh", _program())
    assert not store.adopt(child)
    assert store.read("later.sh") == _program()


def test_decimal_write_domain_invalidates_numeric_aliases_and_retains_unrelated_tree() -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/work")
    for path in ("/tmp/log.123", "/tmp/log.456/child", "/tmp/log.named", "/tmp/work/known.sh"):
        assert store.write(path, _program())
    assert store.invalidate_decimal_path("/tmp/log.", "")
    assert store.read("/tmp/log.123") is None
    assert store.read("/tmp/log.456/child") is None
    assert store.read("/tmp/log.named") is not None
    assert store.exists("/tmp/log.789") is None
    assert not store.write("/tmp/log.789", _program())
    assert store.inventory("/tmp/work") is not None


def test_decimal_alias_inside_fresh_tree_is_unknown_without_erasing_disjoint_bytes() -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/work")
    assert store.write("/tmp/work/log.123.sh", _program())
    assert store.write("/tmp/work/known.sh", _program())
    assert store.invalidate_decimal_path("/tmp/work/log.", ".sh")
    assert store.read("/tmp/work/log.123.sh") is None
    assert not store.owns("/tmp/work/log.456.sh")
    assert store.owns("/tmp/work/known.sh")
    assert store.inventory("/tmp/work") is None
    assert not store.archive("/tmp/work", "/tmp/result.tar")


def test_decimal_path_domains_survive_transactional_adoption() -> None:
    store = ShellArtifactStore(lambda: None)
    child = store.fork()
    assert child.invalidate_decimal_path("/tmp/log.", "")
    assert store.adopt(child)
    assert store.exists("/tmp/log.123") is None
    assert not store.write("/tmp/log.123", _program())


def test_decimal_domain_bounds_and_invalid_paths_refuse_the_snapshot() -> None:
    store = ShellArtifactStore(lambda: None)
    assert not store.invalidate_decimal_path("/tmp/../log.", "")
    assert not store.complete


def test_unknown_sibling_bytes_do_not_remove_an_independent_path_ownership_proof() -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/work")
    store.invalidate("/tmp/work/log")
    assert store.inventory("/tmp/work") is None
    assert store.owns("/tmp/work/output")
    assert not store.owns("/tmp/work/log")
    assert not store.owns("/tmp/work/log/child")


@pytest.mark.parametrize("source_mode", [False, True, None])
@pytest.mark.parametrize("destination_mode", [False, True, None])
def test_producer_plain_copy_retains_existing_destination_mode(
    source_mode: bool | None, destination_mode: bool | None
) -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/copies")
    source = ShellArtifact(_program().value, executable=source_mode)
    assert store.write("/tmp/copies/source", source)
    assert store.write(
        "/tmp/copies/destination",
        ShellArtifact(_program("printf old\n").value, executable=destination_mode),
    )
    assert ShellArtifactEffects(store, lambda: None).copy_file(
        ("cp", "/tmp/copies/source", "/tmp/copies/destination")
    )
    result = store.read("/tmp/copies/destination")
    assert result is not None and result.value == source.value
    assert result.executable is destination_mode


@pytest.mark.parametrize("source_mode", [False, True, None])
@pytest.mark.parametrize("known_new", [False, True])
def test_producer_plain_copy_inherits_mode_only_for_proven_new_destination(
    source_mode: bool | None, known_new: bool
) -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/copies")
    source = ShellArtifact(_program().value, executable=source_mode)
    assert store.write("/tmp/copies/source", source)
    destination = "/tmp/copies/new" if known_new else "/runtime/unknown"
    assert store.exists(destination) is (False if known_new else None)
    assert ShellArtifactEffects(store, lambda: None).copy_file(
        ("cp", "/tmp/copies/source", destination)
    )
    result = store.read(destination)
    assert result is not None and result.value == source.value
    assert result.executable is (source_mode if known_new else None)
    assert store.copy("/tmp/copies/source", "/tmp/copies/generic")
    assert store.read("/tmp/copies/generic") == source


@pytest.mark.parametrize("initial_cwd", [None, ".", "/tmp/work"])
@pytest.mark.parametrize("relative_destination", [False, True])
def test_producer_relative_copy_uses_actual_cwd_for_both_operands(
    initial_cwd: str | None, relative_destination: bool
) -> None:
    destination = "scripts/result.sh" if relative_destination else "$1"
    program = f'set -euo pipefail\ncp scripts/setup.sh "{destination}"\n'
    uploaded = _program("printf uploaded\n")
    generated = _program("printf generated\n")
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/work")
    assert store.write("scripts/setup.sh", uploaded)
    assert store.write("scripts/result.sh", uploaded)
    assert store.write("/tmp/work/scripts/setup.sh", generated)
    output = "/tmp/work/scripts/result.sh"
    producer = ShellProducer(
        store,
        ShellSourceContext(
            "producer.sh",
            {
                "producer.sh": program,
                "scripts/setup.sh": uploaded.value.text,
                "scripts/result.sh": uploaded.value.text,
            },
        ),
        tm._next_shell_invocation_word,
        lambda: None,
        None,
    )
    result = producer.inspect((output,), {}, (output,), initial_cwd=initial_cwd)
    if initial_cwd is None or initial_cwd == "." and relative_destination:
        assert result is None
    else:
        assert result is not None
        assert store.read(output) == (uploaded if initial_cwd == "." else generated)
        assert store.read("scripts/result.sh") == uploaded
