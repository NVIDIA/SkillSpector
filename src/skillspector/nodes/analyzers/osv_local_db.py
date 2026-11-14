# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only OSV snapshot indexes and version matching for SC4."""

from __future__ import annotations

import json
import os
import zipfile
import zlib
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, TypedDict

from packaging.utils import canonicalize_name
from packaging.version import Version as Pep440Version
from semver import Version as SemverVersion

from skillspector.input_handler import _open_regular_file_no_follow, _read_zip_directory_metadata
from skillspector.inspection_ledger import LedgerReason

# The published npm snapshot alone contains more than 230,000 records and
# expands to roughly 375 MiB. Stream individual records without extraction.
_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
_MAX_CENTRAL_DIRECTORY_BYTES = 64 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 500_000
_MAX_INDEX_ENTRIES = 1_000_000
_MAX_EXPANDED_BYTES = 768 * 1024 * 1024
_MAX_RECORD_BYTES = 2 * 1024 * 1024

SNAPSHOT_ERRORS = (
    OSError,
    ValueError,
    RuntimeError,
    OverflowError,
    zipfile.BadZipFile,
    zlib.error,
)


class SnapshotMetadata(TypedDict):
    ecosystem: str
    size_bytes: int
    file_mtime: str


@dataclass(frozen=True)
class SnapshotLimitation:
    reason: LedgerReason
    observed: int
    limit: int


class SnapshotLimitError(ValueError):
    def __init__(self, reason: LedgerReason, observed: int, limit: int) -> None:
        super().__init__(reason.value)
        self.limitation = SnapshotLimitation(reason, observed, limit)


def package_key(name: str, ecosystem: str) -> str:
    # npm names can contain underscores and dots. Do not apply PyPI rules to them.
    return canonicalize_name(name) if ecosystem == "PyPI" else name.lower()


def _affected_package(entry: dict, ecosystem: str) -> str | None:
    package = entry.get("package")
    if package is None:
        # OSV also permits Git-only entries without a package coordinate.
        return None
    if not isinstance(package, dict) or not isinstance(package.get("ecosystem"), str):
        raise ValueError("invalid affected package")
    if package["ecosystem"] != ecosystem:
        return None
    name = package.get("name")
    if not isinstance(name, str) or not name or len(name) > 256:
        raise ValueError("invalid affected package name")
    return package_key(name, ecosystem)


def _read_record(archive: zipfile.ZipFile, member: zipfile.ZipInfo) -> dict:
    if member.file_size > _MAX_RECORD_BYTES:
        raise SnapshotLimitError(
            LedgerReason.ARCHIVE_MEMBER_SIZE_LIMIT, member.file_size, _MAX_RECORD_BYTES
        )
    if member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise ValueError("unsupported advisory compression")
    with archive.open(member) as stream:
        content = stream.read(_MAX_RECORD_BYTES + 1)
    if len(content) > _MAX_RECORD_BYTES:
        raise SnapshotLimitError(
            LedgerReason.ARCHIVE_MEMBER_SIZE_LIMIT, len(content), _MAX_RECORD_BYTES
        )
    record = json.loads(content)
    if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]:
        raise ValueError("invalid advisory record")
    return record


def _entries(record: dict) -> list[dict]:
    entries = record.get("affected")
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError("invalid affected entries")
    return entries


@dataclass
class SnapshotIndex:
    """Package-to-member index tied to one opened snapshot, shared within a scan."""

    archive: zipfile.ZipFile
    metadata: SnapshotMetadata
    source: BinaryIO
    mtime_ns: int
    members: dict[str, list[zipfile.ZipInfo]] = field(default_factory=lambda: defaultdict(list))
    malformed: bool = False
    limitation: SnapshotLimitation | None = None
    timed_out: bool = False

    def build(self, remaining_seconds: Callable[[], float]) -> None:
        expanded_bytes = 0
        index_entries = 0
        for member in self.archive.infolist():
            if remaining_seconds() <= 0:
                self.timed_out = True
                break
            if not member.filename.endswith(".json") or member.is_dir():
                continue
            expanded_bytes += member.file_size
            if expanded_bytes > _MAX_EXPANDED_BYTES:
                self.limitation = SnapshotLimitation(
                    LedgerReason.ARCHIVE_SIZE_LIMIT, expanded_bytes, _MAX_EXPANDED_BYTES
                )
                break
            try:
                record = _read_record(self.archive, member)
                if record.get("withdrawn"):
                    continue
                packages = {
                    _affected_package(entry, self.metadata["ecosystem"])
                    for entry in _entries(record)
                }
            except SnapshotLimitError as exc:
                self.limitation = exc.limitation
                continue
            except SNAPSHOT_ERRORS:
                self.malformed = True
                continue
            names = packages - {None}
            index_entries += len(names)
            if index_entries > _MAX_INDEX_ENTRIES:
                self.limitation = SnapshotLimitation(
                    LedgerReason.OUTPUT_LIMIT, index_entries, _MAX_INDEX_ENTRIES
                )
                break
            for name in names:
                self.members[name].append(member)

    def unchanged(self) -> bool:
        stat = os.fstat(self.source.fileno())
        return stat.st_mtime_ns == self.mtime_ns and stat.st_size == self.metadata["size_bytes"]

    def records(self, name: str) -> Iterator[tuple[dict, list[dict]]]:
        ecosystem = self.metadata["ecosystem"]
        key = package_key(name, ecosystem)
        for member in self.members.get(key, []):
            record = _read_record(self.archive, member)
            yield (
                record,
                [entry for entry in _entries(record) if _affected_package(entry, ecosystem) == key],
            )


def open_snapshot(db_path: Path, ecosystem: str, stack: ExitStack) -> SnapshotIndex:
    """Read fixed-size ZIP headers before zipfile can allocate the directory."""
    source = stack.enter_context(_open_regular_file_no_follow(db_path / ecosystem / "all.zip"))
    stat = os.fstat(source.fileno())
    if stat.st_size > _MAX_ARCHIVE_BYTES:
        raise SnapshotLimitError(LedgerReason.ARCHIVE_SIZE_LIMIT, stat.st_size, _MAX_ARCHIVE_BYTES)
    directory = _read_zip_directory_metadata(source)
    for reason, value, limit in (
        (LedgerReason.ARCHIVE_MEMBER_LIMIT, directory.entries, _MAX_ARCHIVE_MEMBERS),
        (
            LedgerReason.ARCHIVE_SIZE_LIMIT,
            directory.central_directory_bytes,
            _MAX_CENTRAL_DIRECTORY_BYTES,
        ),
    ):
        if value > limit:
            raise SnapshotLimitError(reason, value, limit)
    archive = stack.enter_context(zipfile.ZipFile(source))
    actual_members = len(archive.infolist())
    if actual_members > _MAX_ARCHIVE_MEMBERS:
        raise SnapshotLimitError(
            LedgerReason.ARCHIVE_MEMBER_LIMIT, actual_members, _MAX_ARCHIVE_MEMBERS
        )
    if actual_members != directory.entries:
        raise zipfile.BadZipFile("OSV snapshot directory entry count mismatch")
    return SnapshotIndex(
        archive,
        {
            "ecosystem": ecosystem,
            "size_bytes": stat.st_size,
            "file_mtime": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        },
        source,
        stat.st_mtime_ns,
    )


def _version(value: object, *, semver: bool) -> Pep440Version | SemverVersion:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError("invalid version")
    if semver:
        # npm accepts a leading v. Published OSV SEMVER events normally omit it.
        return SemverVersion.parse(value.removeprefix("v"))
    return Pep440Version(value)


def _matches_range(events: object, version: str, *, semver: bool) -> bool:
    target = _version(version, semver=semver)
    if not isinstance(events, list) or not events:
        raise ValueError("missing range events")
    boundaries: list[tuple[Pep440Version | SemverVersion | None, str]] = []
    limits: list[Pep440Version | SemverVersion] = []
    open_limit = False
    kinds = set()
    for event in events:
        if not isinstance(event, dict) or len(event) != 1:
            raise ValueError("invalid range event")
        kind, value = next(iter(event.items()))
        kinds.add(kind)
        if kind == "limit":
            if value == "*":
                open_limit = True
            else:
                limits.append(_version(value, semver=semver))
        elif kind == "introduced" and value == "0":
            boundaries.append((None, kind))
        elif kind in {"introduced", "fixed", "last_affected"}:
            boundaries.append((_version(value, semver=semver), kind))
        else:
            raise ValueError("unsupported range event")
    if "introduced" not in kinds or {"fixed", "last_affected"} <= kinds:
        raise ValueError("invalid range boundaries")
    if limits and not open_limit and not any(target < limit for limit in limits):
        return False

    # OSV applies every event in sorted order. A fix at the same version as
    # an introduction closes that interval; last_affected is inclusive.
    ordered = sorted(
        ((value, kind) for value, kind in boundaries if value is not None),
        key=lambda boundary: (boundary[0], boundary[1] != "introduced"),
    )
    vulnerable = (None, "introduced") in boundaries
    for boundary, kind in ordered:
        if boundary > target or (boundary == target and kind == "last_affected"):
            break
        vulnerable = kind == "introduced"
    return vulnerable


def validate_version(version: str, ecosystem: str) -> None:
    _version(version, semver=ecosystem == "npm")


def matches_version(entries: list[dict], version: str, ecosystem: str) -> tuple[bool, bool]:
    """Return whether an advisory matches and whether coverage is complete."""
    npm = ecosystem == "npm"
    complete = True
    matched = False
    target = _version(version, semver=npm)
    for entry in entries:
        versions = entry.get("versions", [])
        ranges = entry.get("ranges", [])
        if not isinstance(versions, list) or not isinstance(ranges, list):
            complete = False
            continue
        if not versions and not ranges:
            # An empty versions list means no published affected versions.
            complete &= "versions" in entry
            continue
        for value in versions:
            try:
                matched |= value == version or target == _version(value, semver=npm)
            except ValueError:
                complete = False
        has_package_ranges = any(
            isinstance(r, dict) and r.get("type") in ("ECOSYSTEM", "SEMVER") for r in ranges
        )
        for affected_range in ranges:
            if not isinstance(affected_range, dict):
                complete = False
                continue
            kind = affected_range.get("type")
            if kind == "GIT" and (versions or has_package_ranges):
                # Named package entries can carry Git commit data alongside
                # version data. Commit ranges do not describe package versions.
                continue
            if kind not in ("ECOSYSTEM", "SEMVER"):
                complete = False
                continue
            try:
                matched |= _matches_range(
                    affected_range.get("events"), version, semver=npm or kind == "SEMVER"
                )
            except ValueError:
                complete = False
    return matched, complete
