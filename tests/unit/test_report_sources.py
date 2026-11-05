# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded known-source discovery and whole-output-plan barriers."""

import asyncio
import os
import shutil
from pathlib import Path

import pytest

from skillspector import file_output, report_sources
from skillspector.report_outputs import FormatChoice, ReportOutput
from skillspector.report_sources import (
    SourceProtectionError,
    SourceProtectionInventory,
    current_source_inventory,
    record_scan_sources,
    track_report_sources,
    validate_report_outputs,
)

pytestmark = pytest.mark.skipif(
    not file_output._SECURE_OUTPUT_SUPPORTED, reason="secure descriptor-relative output required"
)


def output(path=None, *, primary=False, format=FormatChoice.json):
    return ReportOutput(format, str(path) if path is not None else None, primary)


def snapshot(root):
    return {
        str(path.relative_to(root)): (
            (info := path.lstat()).st_ino,
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            os.readlink(path)
            if path.is_symlink()
            else path.read_bytes()
            if path.is_file()
            else None,
        )
        for path in [root, *root.rglob("*")]
    }


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "skill"
    root.mkdir()
    for name in ("SKILL.md", "scripts/helper.py", ".hidden/key", ".git/config", "asset.bin"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"protected")
    return root


def test_discovery_includes_every_file_without_reading_content(bundle, monkeypatch):
    expected = {path for path in bundle.rglob("*") if path.is_file()}
    original_open = os.open

    def metadata_only_open(path, flags, *args, **kwargs):
        assert flags & os.O_DIRECTORY
        assert flags & os.O_NOFOLLOW
        return original_open(path, flags, *args, **kwargs)

    def no_content(*args, **kwargs):
        pytest.fail("source discovery read file content")

    with monkeypatch.context() as patch:
        patch.setattr(report_sources.os, "open", metadata_only_open)
        patch.setattr(Path, "read_bytes", no_content)
        inventory = SourceProtectionInventory()
        files = inventory.discover_local(bundle)
    assert set(files) == expected
    assert expected <= inventory.paths
    assert len(inventory.identities) == len(inventory.paths)


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("primary", [False, True])
def test_normalized_duplicate_destinations_rejected(tmp_path, monkeypatch, existing, primary):
    path = tmp_path / "report"
    if existing:
        path.write_bytes(b"previous")
    monkeypatch.chdir(tmp_path)
    before = snapshot(tmp_path)
    entries = [output("./report", primary=primary), output(path, format=FormatChoice.sarif)]
    with pytest.raises(ValueError, match="Duplicate or aliased"):
        validate_report_outputs(entries, SourceProtectionInventory())
    assert snapshot(tmp_path) == before


def test_hardlinked_destinations_rejected(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"previous")
    os.link(first, second)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="Duplicate or aliased"):
        validate_report_outputs(
            [output(first, primary=True), output(second)], SourceProtectionInventory()
        )
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("primary", [False, True])
def test_bundled_script_source_collision_rejected(bundle, tmp_path, alias, primary):
    inventory = SourceProtectionInventory()
    inventory.discover_local(bundle)
    source = bundle / "scripts/helper.py"
    path = source
    if alias:
        path = tmp_path / "alias"
        os.link(source, path)
    before = snapshot(tmp_path)
    entries = [output(path, primary=primary), output(tmp_path / "otherwise-safe")]
    with pytest.raises(ValueError, match="protected local source"):
        validate_report_outputs(entries, inventory)
    assert snapshot(tmp_path) == before


def test_recursive_sources_beyond_skill_retention_limit_protected(tmp_path):
    root = tmp_path / "skills"
    root.mkdir()
    for index in range(40):
        child = root / str(index)
        child.mkdir()
        (child / "SKILL.md").write_bytes(b"manifest")
        (child / "helper.py").write_bytes(b"protected")
    inventory = SourceProtectionInventory()
    files = inventory.discover_local(root)
    assert len(files) == 80
    with pytest.raises(ValueError, match="protected local source"):
        validate_report_outputs([output(root / "39/helper.py")], inventory)


def test_registry_comparison_file_and_alias_protected(tmp_path):
    source = tmp_path / "comparison.json"
    source.write_bytes(b"comparison")
    alias = tmp_path / "alias.json"
    os.link(source, alias)
    inventory = SourceProtectionInventory()
    inventory.add_path(source)
    assert inventory.paths == {source}
    with pytest.raises(ValueError, match="protected local source"):
        validate_report_outputs([output(alias)], inventory)


def test_distinct_repeated_formats_and_primary_stdout_are_valid(tmp_path, capsys):
    entries = [output(primary=True), output(tmp_path / "one"), output(tmp_path / "two")]
    before = snapshot(tmp_path)
    validated = validate_report_outputs(entries, SourceProtectionInventory())
    assert [entry.output for entry in validated] == entries
    assert [entry.normalized_path for entry in validated] == [
        None,
        tmp_path / "one",
        tmp_path / "two",
    ]
    assert all(entry.existing_stat is None for entry in validated)
    assert snapshot(tmp_path) == before
    assert capsys.readouterr() == ("", "")


def test_whole_plan_preflight_rejects_missing_parent_without_changes(tmp_path):
    entries = [output(tmp_path / "first", primary=True), output(tmp_path / "missing/second")]
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="directory does not exist"):
        validate_report_outputs(entries, SourceProtectionInventory())
    assert snapshot(tmp_path) == before


def test_additional_stdout_is_rejected():
    with pytest.raises(ValueError, match="Only the primary"):
        validate_report_outputs([output()], SourceProtectionInventory())


@pytest.mark.parametrize(
    "bounds", [{"max_paths": 2}, {"max_entries": 1}, {"max_path_bytes": 1}, {"max_depth": 1}]
)
def test_discovery_bound_failure_poisoned_inventory(bundle, bounds):
    (bundle / "deep/nested").mkdir(parents=True)
    inventory = SourceProtectionInventory(**bounds)
    before = snapshot(bundle)
    with pytest.raises(SourceProtectionError, match="bound exhausted"):
        inventory.discover_local(bundle)
    assert not inventory.complete
    with pytest.raises(SourceProtectionError, match="bound exhausted"):
        validate_report_outputs([output(primary=True)], inventory)
    assert snapshot(bundle) == before


def test_runtime_bound_fails_closed(bundle):
    ticks = iter([0.0, 0.0, 3.0, 3.0])
    inventory = SourceProtectionInventory(max_seconds=1.0, clock=lambda: next(ticks))
    with pytest.raises(SourceProtectionError, match="runtime bound"):
        inventory.discover_local(bundle)
    assert not inventory.complete


@pytest.mark.parametrize("kind", ["symlink-leaf", "symlink-parent", "fifo", "missing"])
def test_unsafe_or_missing_sources_fail_closed(tmp_path, kind):
    root = tmp_path / "sources"
    root.mkdir()
    if kind == "symlink-leaf":
        (root / "link").symlink_to(tmp_path / "missing")
    elif kind == "symlink-parent":
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)
        (root / "nested").mkdir()
        root = alias / "nested"
    elif kind == "fifo":
        os.mkfifo(root / "pipe")
    else:
        root /= "missing"
    inventory = SourceProtectionInventory()
    before = snapshot(tmp_path)
    with pytest.raises(SourceProtectionError):
        inventory.discover_local(root)
    assert not inventory.complete
    assert snapshot(tmp_path) == before


def test_discovery_errors_fail_closed_and_close_descriptors(bundle, monkeypatch):
    real_open = os.open
    opened = []

    def track_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def unreadable(*args, **kwargs):
        raise PermissionError("unreadable source directory")

    monkeypatch.setattr(report_sources.os, "open", track_open)
    monkeypatch.setattr(report_sources.os, "scandir", unreadable)
    inventory = SourceProtectionInventory()
    with pytest.raises(SourceProtectionError, match="unreadable"):
        inventory.discover_local(bundle)
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)
    assert not inventory.complete


def test_complete_discovery_reused_without_rewalk(bundle, monkeypatch):
    inventory = SourceProtectionInventory()
    files = inventory.discover_local(bundle)

    def no_rewalk(*args, **kwargs):
        pytest.fail("complete source inventory was walked again")

    monkeypatch.setattr(report_sources.os, "scandir", no_rewalk)
    assert inventory.discover_local(bundle) == files
    assert inventory.discover_local(bundle / "scripts") == (bundle / "scripts/helper.py",)


def test_file_discovery_does_not_walk_siblings(tmp_path):
    source = tmp_path / "manifest"
    source.write_bytes(b"manifest")
    (tmp_path / "unrelated").symlink_to(tmp_path / "missing")
    inventory = SourceProtectionInventory()
    assert inventory.discover_local(source) == (source,)
    assert inventory.paths == {source}


@pytest.mark.parametrize("primary", [False, True])
def test_late_child_collision_aborts_entire_plan_before_emission(tmp_path, primary, capsys):
    inventory = SourceProtectionInventory()
    destination = tmp_path / "report"
    destination.write_bytes(b"previous report")
    plan = [
        output(destination, primary=primary),
        output(tmp_path / "other"),
    ]
    if not primary:
        plan.insert(0, output(primary=True))
    validate_report_outputs(plan, inventory)

    child = tmp_path / "child"
    child.mkdir()
    (child / "SKILL.md").write_bytes(b"child manifest")
    os.link(destination, child / "helper.py")
    result = {
        "skill_path": str(child),
        "components": ["SKILL.md", "helper.py"],
        "artifact_inventory": [{"path": "helper.py", "disposition": "out_of_scope"}],
    }
    with track_report_sources(inventory):
        assert record_scan_sources(result) is result
        # Simulate transitive truncation; protection does not retain the result.
        result.clear()
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="protected local source"):
        validate_report_outputs(plan, inventory)
    assert snapshot(tmp_path) == before
    assert capsys.readouterr() == ("", "")


def test_source_identities_survive_child_cleanup(tmp_path):
    child = tmp_path / "skillspector_child"
    child.mkdir()
    source = child / "script.py"
    source.write_bytes(b"protected")
    alias = tmp_path / "report"
    os.link(source, alias)
    identity = (source.stat().st_dev, source.stat().st_ino)
    inventory = SourceProtectionInventory()
    inventory.add_result({"root_path": str(child)})
    shutil.rmtree(child)
    assert source in inventory.paths
    assert identity in inventory.identities
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="protected local source"):
        validate_report_outputs([output(alias)], inventory)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("root_key", ["skill_path", "root_path"])
def test_result_root_protects_files_omitted_from_content_inventory(bundle, root_key):
    inventory = SourceProtectionInventory()
    inventory.add_result({root_key: str(bundle), "components": [], "artifact_inventory": []})
    assert bundle / "scripts/helper.py" in inventory.paths
    assert bundle / ".hidden/key" in inventory.paths


def test_result_records_new_component_after_initial_discovery(bundle):
    inventory = SourceProtectionInventory()
    inventory.discover_local(bundle)
    late = bundle / "late.py"
    late.write_bytes(b"newly discovered")
    inventory.add_result({"skill_path": str(bundle), "components": ["late.py"]})
    assert late in inventory.paths


def test_result_with_file_root_and_absolute_external_components(tmp_path):
    source = tmp_path / "manifest"
    source.write_bytes(b"manifest")
    external = tmp_path / "external.py"
    external.write_bytes(b"external")
    inventory = SourceProtectionInventory()
    inventory.add_result({"root_path": str(source), "components": ["manifest", str(external)]})
    assert inventory.paths == {source, external}


def test_virtual_archive_members_protect_physical_containers(bundle):
    archive = bundle / "container.zip"
    archive.write_bytes(b"container")
    inventory = SourceProtectionInventory()
    inventory.add_result(
        {
            "skill_path": str(bundle),
            "components": ["container.zip!/script.py"],
            "artifact_inventory": [
                {
                    "path": "container.zip!/nested.tar",
                    "inspected_members": ["container.zip!/nested.tar!/inner.py"],
                }
            ],
        }
    )
    assert archive in inventory.paths
    assert all("!/" not in str(path) for path in inventory.paths)


@pytest.mark.parametrize(
    "metadata",
    [
        {"skill_path": 42},
        {"components": ["relative.py"]},
        {"components": "script.py"},
        {"components": [42]},
        {"components": ["../outside.py"]},
        {"components": ["source::script.py"]},
        {"components": ["https://example.com/script.py"]},
        {"components": ["missing.py"]},
        {"artifact_inventory": {}},
        {"artifact_inventory": ["script.py"]},
        {"artifact_inventory": [{}]},
        {"artifact_inventory": [{"path": "SKILL.md", "inspected_members": "member.py"}]},
    ],
)
def test_invalid_or_insufficient_result_metadata_fails_closed(bundle, metadata):
    inventory = SourceProtectionInventory()
    if metadata != {"components": ["relative.py"]}:
        metadata = {"root_path": str(bundle), **metadata}
    with pytest.raises(SourceProtectionError):
        inventory.add_result(metadata)
    assert not inventory.complete
    with pytest.raises(SourceProtectionError):
        validate_report_outputs([output(primary=True)], inventory)


def test_result_metadata_is_bounded_even_for_duplicate_names(bundle):
    inventory = SourceProtectionInventory(max_entries=30)
    with pytest.raises(SourceProtectionError, match="entry bound"):
        inventory.add_result({"skill_path": str(bundle), "components": ["SKILL.md"] * 100})
    assert not inventory.complete


def test_repeated_result_recording_keeps_same_bounded_identities(bundle):
    inventory = SourceProtectionInventory()
    result = {"skill_path": str(bundle), "components": ["SKILL.md"]}
    with track_report_sources(inventory):
        record_scan_sources(result)
        paths, identities = inventory.paths, inventory.identities
        record_scan_sources(result)
    assert inventory.paths == paths
    assert inventory.identities == identities


def test_legacy_recording_is_noop_without_filesystem_access(monkeypatch):
    result = {"skill_path": "/missing/legacy-input", "components": ["missing.py"]}

    def no_checks(*args, **kwargs):
        pytest.fail("legacy source recording inspected the filesystem")

    monkeypatch.setattr(report_sources, "open_output_parent", no_checks)
    assert current_source_inventory() is None
    assert record_scan_sources(result) is result


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_tracking_context_resets_on_errors_or_interrupts(error):
    outer = SourceProtectionInventory()
    inner = SourceProtectionInventory()
    with track_report_sources(outer):
        assert current_source_inventory() is outer
        with pytest.raises(error), track_report_sources(inner):
            assert current_source_inventory() is inner
            raise error
        assert current_source_inventory() is outer
    assert current_source_inventory() is None


async def test_tracking_context_isolated_between_async_scans(tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    async def scan(path):
        inventory = SourceProtectionInventory()
        with track_report_sources(inventory):
            await asyncio.sleep(0)
            record_scan_sources({"components": [str(path)]})
            await asyncio.sleep(0)
            assert current_source_inventory() is inventory
        return inventory

    first_inventory, second_inventory = await asyncio.gather(scan(first), scan(second))
    assert first_inventory.paths == {first}
    assert second_inventory.paths == {second}
    assert current_source_inventory() is None


def test_interrupted_discovery_closes_handles_and_poisons_inventory(bundle, monkeypatch):
    opened = []
    real_open = os.open

    def track_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(report_sources.os, "open", track_open)
    monkeypatch.setattr(report_sources.os, "scandir", interrupt)
    inventory = SourceProtectionInventory()
    with pytest.raises(KeyboardInterrupt):
        inventory.discover_local(bundle)
    assert not inventory.complete
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)
    with pytest.raises(SourceProtectionError, match="interrupted"):
        validate_report_outputs([output(primary=True)], inventory)


def test_late_barrier_refreshes_current_and_historical_source_identities(tmp_path):
    source = tmp_path / "source.py"
    source.write_bytes(b"original")
    original_alias = tmp_path / "original_alias"
    os.link(source, original_alias)
    inventory = SourceProtectionInventory()
    inventory.add_path(source)
    source.unlink()
    source.write_bytes(b"new source")
    new_alias = tmp_path / "new_alias"
    os.link(source, new_alias)
    before = snapshot(tmp_path)
    for path in (original_alias, new_alias):
        with pytest.raises(ValueError, match="protected local source"):
            validate_report_outputs([output(path)], inventory)
    assert snapshot(tmp_path) == before


def test_late_source_symlink_swap_fails_closed(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"original")
    inventory = SourceProtectionInventory()
    inventory.add_path(source)
    source.unlink()
    source.symlink_to(tmp_path / "missing")
    before = snapshot(tmp_path)
    with pytest.raises(SourceProtectionError, match="symlinked"):
        validate_report_outputs([output(primary=True), output(tmp_path / "report")], inventory)
    assert snapshot(tmp_path) == before


def test_historical_source_identity_bound_fails_closed(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"original")
    alias = tmp_path / "alias"
    os.link(source, alias)
    inventory = SourceProtectionInventory(max_paths=1)
    inventory.add_path(source)
    source.unlink()
    source.write_bytes(b"new source")
    with pytest.raises(SourceProtectionError, match="identity bound"):
        inventory.add_path(source)
    assert not inventory.complete
