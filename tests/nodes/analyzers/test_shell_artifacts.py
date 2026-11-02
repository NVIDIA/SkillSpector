# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated code must retain its bytes and provenance across file operations."""

import pytest

from skillspector.shell_artifacts import ShellArtifact, ShellArtifactStore
from skillspector.shell_sources import ShellSourceOrigin, ShellSourceValue


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
