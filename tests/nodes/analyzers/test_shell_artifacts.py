# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated code must retain its bytes and provenance across file operations."""

import pytest

from skillspector.shell_artifacts import ShellArtifact, ShellArtifactStore
from skillspector.shell_artifact_flow import ShellArtifactEffects, ShellProducer, helper_stdout_artifacts
from skillspector.shell_helper_argv import ShellHelperOutput
from skillspector.shell_sources import ShellSourceContext, ShellSourceOrigin, ShellSourceValue
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm


def _program(text: str = "printf '%s\\n' ready\n") -> ShellArtifact:
    origin = ShellSourceOrigin("scripts/setup.sh", 0, len(text), "a" * 64)
    return ShellArtifact(ShellSourceValue(text, "code", (origin,)))


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
    archive = ShellArtifact(members=(("setup.sh", _program()), ("setup.sh", _program("rm -rf /\n"))))
    assert not store.write("/tmp/package.tgz", archive)
    assert not store.extract("/tmp/package.tgz", "/tmp/extract-owned")
    assert store.read("/tmp/extract-owned/setup.sh") is None


def test_extraction_requires_a_fresh_known_destination() -> None:
    store = ShellArtifactStore(lambda: None)
    assert store.write("/tmp/package.tgz", ShellArtifact(members=(("setup.sh", _program()),)))
    assert not store.extract("/tmp/package.tgz", "/tmp/existing")
    assert store.read("/tmp/existing/setup.sh") is None


@pytest.mark.parametrize("paths", [("root/file", "root/file/child"), ("root/file/child", "root/file")])
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
    assert effects.tar(("tar", "-I", "lbzip2", "-cf", "/tmp/input.tbz2", "-C", "/tmp/input", "tool"))
    output = helper_stdout_artifacts(((ShellHelperOutput(path="/tmp/input.tbz2"),),), store, (), lambda: None)
    assert output is not None
    assert effects.redirect_stdout("/tmp/work/download.tbz2", output)
    assert effects.tar(("tar", "-I", "lbzip2", "-xf", "/tmp/work/download.tbz2", "-C", "/tmp/work/bundle/tools"))
    original = _program()
    assert store.write("scripts/setup.sh", original)
    assert effects.copy_file(("cp", "scripts/setup.sh", "/tmp/work/bundle/setup.sh"))
    assert effects.tar(("tar", "-I", "pigz", "-cf", "/tmp/bundle.tgz", "-C", "/tmp/work", "bundle"))
    assert effects.tar(("tar", "-I", "pigz", "-xf", "/tmp/bundle.tgz", "-C", "/tmp/output"))
    assert store.read("/tmp/output/bundle/setup.sh") == original
    assert store.read("/tmp/output/bundle/tools/tool/flash") is not None


@pytest.mark.parametrize("pieces", [
    (ShellHelperOutput(text="prefix"), ShellHelperOutput(path="/tmp/archive.tgz")),
    (ShellHelperOutput(path="/tmp/archive.tgz"), ShellHelperOutput(text="suffix")),
    (ShellHelperOutput(path="/tmp/archive.tgz"), ShellHelperOutput(path="/tmp/archive.tgz")),
    (ShellHelperOutput(path="/tmp/missing.tgz"),),
])
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


@pytest.mark.parametrize("argv", [
    ("tar", "-I", "malicious-compressor", "-cf", "/tmp/archive", "-C", "/tmp/root", "folder"),
    ("tar", "-cf", "/tmp/archive", "-C", "/tmp/root", "../folder"),
    ("tar", "-xf", "/tmp/archive", "-C", "/tmp/root", "--overwrite"),
])
def test_unmodeled_tar_arguments_never_create_proven_code(argv) -> None:
    store = ShellArtifactStore(lambda: None)
    assert not ShellArtifactEffects(store, lambda: None).tar(argv)


_PRODUCER = '''set -Eeuo pipefail
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
'''


def _inspect_producer(program=_PRODUCER, *, modes=("one",)):
    store = ShellArtifactStore(lambda: None)
    assert store.create_empty_directory("/tmp/result")
    assert store.write("scripts/setup.sh", _program())
    archive = ShellArtifact(members=(("tool/FlashTool", _program("#!/bin/bash\n")),))

    def helper_output(argv):
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

    context = ShellSourceContext("scripts/producer.sh", {"scripts/producer.sh": program, "scripts/setup.sh": _program().value.text})
    producer = ShellProducer(store, context, tm._next_shell_invocation_word, lambda: None, helper_output)
    return producer.inspect(("/tmp/result/package.tgz",), {}, ("/tmp/result/package.tgz",)), store


def test_finite_producer_preserves_setup_copy_and_archive_member_prefixes() -> None:
    result, store = _inspect_producer()
    assert result is not None
    assert store.create_empty_directory("/tmp/extracted")
    assert store.extract("/tmp/result/package.tgz", "/tmp/extracted")
    assert store.read("/tmp/extracted/bundle/setup.sh") == _program()
    assert store.read("/tmp/extracted/bundle/tools/tool/FlashTool") is not None
    assert all(path.startswith(("/tmp/producer.__skillspector_producer_", "/tmp/result/")) for path in result.writes)


def test_failed_zero_and_multiple_match_branches_do_not_supply_fictitious_outputs() -> None:
    result, _ = _inspect_producer(modes=("zero", "two", "one"))
    assert result is not None
    result, store = _inspect_producer(modes=("zero", "two"))
    assert result is None
    assert store.read("/tmp/result/package.tgz") is None


@pytest.mark.parametrize("mutation", [
    ('cp "$(dirname', 'eval "danger"\ncp "$(dirname'),
    ('tar -I lbzip2 -xf', 'tar -P -I lbzip2 -xf'),
    ('mkdir -p "$root/helpers"', 'mkdir -p "$root/helpers"\nln -s /outside "$root/escape"'),
    ('local spec="$1" result', 'local spec="$1" result\ncp scripts/setup.sh /tmp/hidden-write'),
    ('set -Eeuo pipefail', 'set +e'),
])
def test_unmodeled_producer_effect_refuses_all_outputs(mutation) -> None:
    result, store = _inspect_producer(_PRODUCER.replace(*mutation))
    assert result is None
    assert store.read("/tmp/result/package.tgz") is None
