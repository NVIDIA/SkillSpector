# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, local-only inspection of ZIP-compatible nested artifacts.

The inspector never extracts members to disk and never renders, imports, or
executes their contents.  It recognizes ZIP-compatible content by bytes, gives
every member a stable virtual path, and returns text projections solely for
deterministic analyzers.
"""

from __future__ import annotations

import io
import stat
import struct
import time
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from skillspector.artifacts import (
    ArtifactDisposition,
    ArtifactRecord,
    ContentKind,
    classify_artifact,
)
from skillspector.constants import MAX_FILE_BYTES
from skillspector.input_handler import (
    _FileOpenError,
    _open_regular_file_no_follow,
    _UnsafeFileError,
)
from skillspector.inspection_ledger import (
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    LedgerRecordType,
    ledger_event,
)

ARCHIVE_MAX_DEPTH = 3
ARCHIVE_MAX_MEMBERS = 1_000
ARCHIVE_MAX_UNCOMPRESSED_BYTES = 25 * 1024 * 1024
ARCHIVE_MAX_CENTRAL_DIRECTORY_BYTES = 4 * 1024 * 1024
ARCHIVE_MAX_COMPRESSION_RATIO = 100
ARCHIVE_MAX_SECONDS = 5.0

_ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_EXECUTABLE_SUFFIXES = frozenset(
    {
        ".app",
        ".bash",
        ".bat",
        ".bin",
        ".cmd",
        ".com",
        ".dll",
        ".dylib",
        ".exe",
        ".go",
        ".js",
        ".msi",
        ".pl",
        ".ps1",
        ".py",
        ".pyc",
        ".pyo",
        ".rb",
        ".rs",
        ".sh",
        ".so",
        ".ts",
        ".zsh",
    }
)
_OOXML_MARKERS: tuple[tuple[str, str], ...] = (
    ("word/", "docx"),
    ("xl/", "xlsx"),
    ("ppt/", "pptx"),
)
_EXPECTED_SUFFIXES: dict[str, frozenset[str]] = {
    "docx": frozenset({".docx", ".docm", ".dotx", ".dotm"}),
    "xlsx": frozenset({".xlsx", ".xlsm", ".xltx", ".xltm"}),
    "pptx": frozenset({".pptx", ".pptm", ".potx", ".potm", ".ppsx", ".ppsm"}),
    "zip": frozenset({".zip"}),
}


@dataclass
class NestedInspectionResult:
    """Virtual inventory and local-only content derived from nested artifacts."""

    components: list[str] = field(default_factory=list)
    file_cache: dict[str, str] = field(default_factory=dict)
    raw_file_cache: dict[str, bytes] = field(default_factory=dict)
    artifact_inventory: list[ArtifactRecord] = field(default_factory=list)
    metadata: list[dict[str, object]] = field(default_factory=list)
    outer_metadata: dict[str, dict[str, object]] = field(default_factory=dict)
    ledger_events: list[InspectionLedgerEvent] = field(default_factory=list)
    uncompressed_bytes: int = 0
    # Exceptions can target a top-level container before a virtual artifact row
    # exists. Preserve their canonical disposition so build_context can apply
    # the same accounting to the outer bundle inventory.
    inventory_overrides: dict[str, tuple[ArtifactDisposition, str]] = field(default_factory=dict)


@dataclass
class _Budget:
    clock: Callable[[], float]
    max_members: int
    max_uncompressed_bytes: int
    max_central_directory_bytes: int
    max_member_bytes: int
    max_depth: int
    max_compression_ratio: int
    deadline: float
    started_at: float
    runtime_limit: float
    last_checked_at: float
    members: int = 0
    uncompressed_bytes: int = 0
    halted: bool = False

    def expired(self) -> bool:
        self.last_checked_at = self.clock()
        return self.last_checked_at > self.deadline

    @property
    def elapsed(self) -> float:
        return max(0.0, self.last_checked_at - self.started_at)


@dataclass(frozen=True)
class _CentralDirectory:
    """Preflighted central-directory bounds read without constructing ZipInfo objects."""

    declared_entries: int
    size_bytes: int
    start: int
    end: int


_PARTIAL_INVENTORY_REASONS = frozenset(
    {
        LedgerReason.ARCHIVE_AMBIGUOUS_MEMBER_PATH,
        LedgerReason.ARCHIVE_COMPRESSION_RATIO,
        LedgerReason.ARCHIVE_DEPTH_LIMIT,
        LedgerReason.ARCHIVE_FORMAT_MISMATCH,
        LedgerReason.ARCHIVE_MEMBER_LIMIT,
        LedgerReason.ARCHIVE_MEMBER_SIZE_LIMIT,
        LedgerReason.ARCHIVE_SIZE_LIMIT,
        LedgerReason.ARCHIVE_TIME_LIMIT,
        LedgerReason.ARCHIVE_UNSAFE_MEMBER_PATH,
    }
)
_FAILED_INVENTORY_REASONS = frozenset(
    {
        LedgerReason.ARCHIVE_ENCRYPTED,
        LedgerReason.ARCHIVE_LINK_MEMBER,
        LedgerReason.ARCHIVE_MALFORMED,
        LedgerReason.ARCHIVE_TRUNCATED,
        LedgerReason.ARCHIVE_UNSUPPORTED_COMPRESSION,
    }
)


_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_CENTRAL_FILE_SIGNATURE = b"PK\x01\x02"
_CENTRAL_DIGITAL_SIGNATURE = b"PK\x05\x05"
_MAX_EOCD_SEARCH = 22 + 65_535


def _find_eocd(data: bytes) -> int | None:
    """Return the terminal EOCD offset, rejecting signatures embedded in comments."""
    lower_bound = max(0, len(data) - _MAX_EOCD_SEARCH)
    search_end = len(data)
    while search_end > lower_bound:
        offset = data.rfind(_EOCD_SIGNATURE, lower_bound, search_end)
        if offset < 0:
            return None
        if offset + 22 <= len(data):
            comment_length = struct.unpack_from("<H", data, offset + 20)[0]
            if offset + 22 + comment_length == len(data):
                return offset
        search_end = offset
    return None


def _zip64_record_offset(data: bytes, *, locator_offset: int) -> int | None:
    """Resolve a ZIP64 EOCD record, including archives with a prepended stub."""
    if locator_offset < 0 or data[locator_offset : locator_offset + 4] != _ZIP64_LOCATOR_SIGNATURE:
        return None
    locator_disk, reported_offset, total_disks = struct.unpack_from(
        "<IQI", data, locator_offset + 4
    )
    if locator_disk != 0 or total_disks != 1:
        return None
    candidates = [reported_offset]
    fallback = data.rfind(_ZIP64_EOCD_SIGNATURE, 0, locator_offset)
    if fallback >= 0 and fallback != reported_offset:
        candidates.append(fallback)
    for offset in candidates:
        if offset < 0 or offset + 56 > locator_offset:
            continue
        if data[offset : offset + 4] != _ZIP64_EOCD_SIGNATURE:
            continue
        record_size = struct.unpack_from("<Q", data, offset + 4)[0]
        if record_size >= 44 and offset + 12 + record_size <= locator_offset:
            return offset
    return None


def _central_directory_bounds(data: bytes) -> _CentralDirectory | None:
    """Read EOCD/ZIP64 counts and byte bounds without invoking ``zipfile``."""
    eocd_offset = _find_eocd(data)
    if eocd_offset is None:
        return None
    (
        disk_number,
        directory_disk,
        entries_on_disk,
        declared_entries,
        directory_size,
        directory_offset,
    ) = struct.unpack_from("<HHHHII", data, eocd_offset + 4)
    central_end = eocd_offset
    needs_zip64 = (
        entries_on_disk == 0xFFFF
        or declared_entries == 0xFFFF
        or directory_size == 0xFFFFFFFF
        or directory_offset == 0xFFFFFFFF
    )
    if needs_zip64:
        zip64_offset = _zip64_record_offset(data, locator_offset=eocd_offset - 20)
        if zip64_offset is None:
            return None
        (
            disk_number,
            directory_disk,
            entries_on_disk,
            declared_entries,
            directory_size,
            directory_offset,
        ) = struct.unpack_from("<IIQQQQ", data, zip64_offset + 16)
        central_end = zip64_offset
    if disk_number != 0 or directory_disk != 0 or entries_on_disk != declared_entries:
        return None
    if directory_size > central_end:
        return None
    central_start = central_end - directory_size
    # The recorded offset may omit a prepended executable stub, but it cannot
    # point beyond the actual central-directory start.
    if directory_offset > central_start:
        return None
    return _CentralDirectory(
        declared_entries=declared_entries,
        size_bytes=directory_size,
        start=central_start,
        end=central_end,
    )


def _count_central_directory_entries(
    data: bytes,
    directory: _CentralDirectory,
    *,
    stop_after: int,
) -> int | None:
    """Count central headers with constant memory, stopping once a limit is exceeded."""
    offset = directory.start
    count = 0
    while offset < directory.end:
        signature = data[offset : offset + 4]
        if signature == _CENTRAL_DIGITAL_SIGNATURE:
            if offset + 6 > directory.end:
                return None
            signature_size = struct.unpack_from("<H", data, offset + 4)[0]
            offset += 6 + signature_size
            return count if offset == directory.end else None
        if signature != _CENTRAL_FILE_SIGNATURE or offset + 46 > directory.end:
            return None
        name_length, extra_length, comment_length = struct.unpack_from("<HHH", data, offset + 28)
        record_size = 46 + name_length + extra_length + comment_length
        if record_size < 46 or offset + record_size > directory.end:
            return None
        offset += record_size
        count += 1
        if count > stop_after:
            return count
    return count if offset == directory.end else None


def _is_zip_signature(data: bytes) -> bool:
    return data.startswith(_ZIP_SIGNATURES)


def _is_hidden_path(path: str) -> bool:
    return any(part.startswith(".") for part in path.replace("\\", "/").split("/") if part)


def _container_type(names: list[str]) -> str:
    normalized = [name.replace("\\", "/").lower() for name in names]
    if "[content_types].xml" in normalized:
        for marker, container_type in _OOXML_MARKERS:
            if any(name.startswith(marker) for name in normalized):
                return container_type
    return "zip"


def _expected_container_type(path: str) -> str | None:
    suffix = Path(path).suffix.lower()
    return next(
        (
            container_type
            for container_type, suffixes in _EXPECTED_SUFFIXES.items()
            if suffix in suffixes
        ),
        None,
    )


def _safe_member_name(name: str) -> str | None:
    normalized = name.replace("\\", "/")
    if (
        not normalized
        or "\x00" in normalized
        or "!/" in normalized
        or normalized.startswith(("/", "//"))
    ):
        return None
    if len(normalized) >= 2 and normalized[1] == ":":
        return None
    parts = PurePosixPath(normalized).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        return None
    return PurePosixPath(*parts).as_posix()


def _zip_member_is_link(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return bool(mode and stat.S_ISLNK(mode))


def is_executable_content(path: str, data: bytes, mode: int = 0) -> bool:
    """Classify filesystem and archive content with one static-only policy."""
    suffix = Path(path).suffix.lower()
    executable_magic = data.startswith(
        (b"#!", b"MZ", b"\x7fELF", b"\xfe\xed\xfa", b"\xcf\xfa\xed\xfe")
    )
    return suffix in _EXECUTABLE_SUFFIXES or executable_magic or bool(mode & 0o111)


def _member_executable(info: zipfile.ZipInfo, safe_name: str, data: bytes) -> bool:
    return is_executable_content(safe_name, data, info.external_attr >> 16)


def _nested_path(outer_path: str, virtual_path: str) -> str:
    prefix = f"{outer_path}!/"
    return virtual_path[len(prefix) :] if virtual_path.startswith(prefix) else virtual_path


def _sorted_infos(infos: list[zipfile.ZipInfo]) -> list[zipfile.ZipInfo]:
    return sorted(infos, key=lambda item: item.filename)


def _record_outer_metadata(
    result: NestedInspectionResult,
    *,
    path: str,
    container_type: str,
    hidden: bool,
    disguised: bool,
) -> None:
    result.outer_metadata[path] = {
        "type": container_type,
        "container_type": container_type,
        "container_ancestry": [container_type],
        "hidden": hidden,
        "disguised": disguised,
        # Recognized and expected containers are never provider input,
        # including when their bytes cannot be fully inspected.
        "local_only": True,
    }


def _virtual_type(path: str, data: bytes, nested_type: str | None) -> str:
    if nested_type is not None:
        return nested_type
    suffix = Path(path).suffix.lower()
    return {
        ".md": "markdown",
        ".markdown": "markdown",
        ".py": "python",
        ".sh": "shell",
        ".bash": "shell",
        ".zsh": "shell",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".txt": "text",
        ".js": "javascript",
        ".ts": "typescript",
        ".rb": "ruby",
        ".go": "go",
        ".rs": "rust",
        ".xml": "xml",
    }.get(suffix, "binary" if b"\x00" in data[:8192] else "text")


def _mark_inventory_exception(
    result: NestedInspectionResult,
    *,
    path: str,
    reason: LedgerReason,
) -> None:
    """Apply a container-inspection exception to an existing virtual artifact row."""
    disposition = (
        ArtifactDisposition.PARTIAL
        if reason in _PARTIAL_INVENTORY_REASONS
        else ArtifactDisposition.FAILED
        if reason in _FAILED_INVENTORY_REASONS
        else None
    )
    if disposition is None:
        return
    result.inventory_overrides[path] = (disposition, reason.value)
    for artifact in reversed(result.artifact_inventory):
        if artifact["path"] == path:
            artifact["disposition"] = disposition
            artifact["reason"] = reason.value
            return


def _exception(
    result: NestedInspectionResult,
    *,
    path: str,
    reason: LedgerReason,
    observed_bytes: int | None = None,
    limit_bytes: int | None = None,
    observed_artifacts: int | None = None,
    limit_artifacts: int | None = None,
    observed_depth: int | None = None,
    limit_depth: int | None = None,
    observed_seconds: float | None = None,
    limit_seconds: float | None = None,
) -> None:
    _mark_inventory_exception(result, path=path, reason=reason)
    result.ledger_events.append(
        ledger_event(
            outcome=(
                LedgerOutcome.PARTIAL
                if reason in _PARTIAL_INVENTORY_REASONS
                else LedgerOutcome.SKIPPED
            ),
            record_type=LedgerRecordType.SYSTEM,
            phase="nested_artifact_inspection",
            path=path,
            reason=reason,
            observed_bytes=observed_bytes,
            limit_bytes=limit_bytes,
            observed_artifacts=observed_artifacts,
            limit_artifacts=limit_artifacts,
            observed_depth=observed_depth,
            limit_depth=limit_depth,
            observed_seconds=observed_seconds,
            limit_seconds=limit_seconds,
        )
    )


def _time_exception(
    result: NestedInspectionResult,
    *,
    path: str,
    budget: _Budget,
) -> None:
    """Record elapsed time and the effective bound from the last deadline check."""
    _exception(
        result,
        path=path,
        reason=LedgerReason.ARCHIVE_TIME_LIMIT,
        observed_seconds=budget.elapsed,
        limit_seconds=budget.runtime_limit,
    )


def _add_unreadable_component(
    result: NestedInspectionResult,
    *,
    virtual_path: str,
    outer_path: str,
    member_path: str,
    container_type: str,
    container_ancestry: tuple[str, ...],
    concealment_reasons: tuple[str, ...],
    depth: int,
    reason: LedgerReason,
    size_bytes: int,
) -> None:
    if virtual_path not in result.file_cache:
        result.components.append(virtual_path)
        # A binary sentinel lets ordinary analyzers account for the component
        # without pretending that inaccessible bytes were inspected as text.
        result.file_cache[virtual_path] = "\x00"
        disposition = (
            ArtifactDisposition.PARTIAL
            if reason in _PARTIAL_INVENTORY_REASONS
            else ArtifactDisposition.FAILED
        )
        result.artifact_inventory.append(
            {
                "path": virtual_path,
                "content_kind": ContentKind.OPAQUE,
                "disposition": disposition,
                "size_bytes": max(size_bytes, 0),
                "decodable": False,
                "contains_nul": False,
                "misleading_extension": False,
                "referenced": False,
                "reason": reason.value,
            }
        )
        result.metadata.append(
            {
                "path": virtual_path,
                "type": "binary",
                "lines": 0,
                "executable": False,
                "size_bytes": max(size_bytes, 0),
                "outer_path": outer_path,
                "nested_path": member_path,
                "container_type": container_type,
                "container_ancestry": list(container_ancestry),
                "container_depth": depth,
                "hidden": _is_hidden_path(member_path),
                "local_only": True,
                "concealed_executable": False,
                "concealment_reasons": list(concealment_reasons),
            }
        )


def _inspect_zip_bytes(
    data: bytes,
    *,
    outer_path: str,
    container_virtual_path: str,
    depth: int,
    outer_hidden: bool,
    outer_disguised: bool | None,
    outer_expected_type: str | None,
    ancestor_container_types: tuple[str, ...],
    budget: _Budget,
    result: NestedInspectionResult,
) -> None:
    if budget.expired():
        budget.halted = True
        _time_exception(result, path=container_virtual_path, budget=budget)
        return
    malformed_reason = (
        LedgerReason.ARCHIVE_MALFORMED if depth == 1 else LedgerReason.ARCHIVE_TRUNCATED
    )
    directory = _central_directory_bounds(data)
    if directory is None:
        _exception(result, path=container_virtual_path, reason=malformed_reason)
        return
    remaining_members = budget.max_members - budget.members
    if directory.declared_entries > remaining_members:
        budget.halted = True
        _exception(
            result,
            path=container_virtual_path,
            reason=LedgerReason.ARCHIVE_MEMBER_LIMIT,
            observed_artifacts=budget.members + directory.declared_entries,
            limit_artifacts=budget.max_members,
        )
        return
    if directory.size_bytes > budget.max_central_directory_bytes:
        budget.halted = True
        _exception(
            result,
            path=container_virtual_path,
            reason=LedgerReason.ARCHIVE_SIZE_LIMIT,
            observed_bytes=directory.size_bytes,
            limit_bytes=budget.max_central_directory_bytes,
        )
        return
    actual_entries = _count_central_directory_entries(
        data,
        directory,
        stop_after=remaining_members,
    )
    if actual_entries is None:
        _exception(result, path=container_virtual_path, reason=malformed_reason)
        return
    if actual_entries > remaining_members:
        budget.halted = True
        _exception(
            result,
            path=container_virtual_path,
            reason=LedgerReason.ARCHIVE_MEMBER_LIMIT,
            observed_artifacts=budget.members + actual_entries,
            limit_artifacts=budget.max_members,
        )
        return
    if actual_entries != directory.declared_entries:
        _exception(result, path=container_virtual_path, reason=malformed_reason)
        return
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError, ValueError):
        _exception(result, path=container_virtual_path, reason=malformed_reason)
        return

    with archive:
        if budget.expired():
            budget.halted = True
            _time_exception(result, path=container_virtual_path, budget=budget)
            return

        # ZipFile has already parsed the central directory. Enforce the cumulative
        # entry budget before sorting or inspecting any attacker-controlled names.
        infos = archive.filelist
        remaining_members = budget.max_members - budget.members
        if len(infos) > remaining_members:
            budget.halted = True
            _exception(
                result,
                path=container_virtual_path,
                reason=LedgerReason.ARCHIVE_MEMBER_LIMIT,
                observed_artifacts=budget.members + len(infos),
                limit_artifacts=budget.max_members,
            )
            return
        if len(infos) != actual_entries:
            _exception(result, path=container_virtual_path, reason=malformed_reason)
            return
        budget.members += len(infos)
        infos = _sorted_infos(infos)
        current_type = _container_type([info.filename for info in infos])
        effective_outer_disguised = (
            outer_disguised
            if outer_disguised is not None
            else outer_expected_type is None or outer_expected_type != current_type
        )
        if depth == 1:
            _record_outer_metadata(
                result,
                path=outer_path,
                container_type=current_type,
                hidden=outer_hidden,
                disguised=effective_outer_disguised,
            )
        container_ancestry = (*ancestor_container_types, current_type)
        inherited_reason_list: list[str] = []
        if any(item in {"docx", "xlsx", "pptx"} for item in container_ancestry):
            inherited_reason_list.append("document_container")
        if outer_hidden:
            inherited_reason_list.append("hidden_artifact")
        if effective_outer_disguised:
            inherited_reason_list.append("disguised_container")
        inherited_reasons = tuple(inherited_reason_list)
        seen_names: set[str] = set()

        for info in infos:
            if info.is_dir():
                continue
            if budget.expired():
                budget.halted = True
                _time_exception(result, path=container_virtual_path, budget=budget)
                return
            safe_name = _safe_member_name(info.filename)
            if safe_name is None:
                _exception(
                    result,
                    path=container_virtual_path,
                    reason=LedgerReason.ARCHIVE_UNSAFE_MEMBER_PATH,
                )
                continue
            virtual_path = f"{container_virtual_path}!/{safe_name}"
            if safe_name in seen_names or virtual_path in result.file_cache:
                _exception(
                    result,
                    path=virtual_path,
                    reason=LedgerReason.ARCHIVE_AMBIGUOUS_MEMBER_PATH,
                )
                continue
            seen_names.add(safe_name)
            member_path = _nested_path(outer_path, virtual_path)
            concealment_reasons = tuple(
                dict.fromkeys(
                    (
                        *inherited_reasons,
                        *(("hidden_artifact",) if _is_hidden_path(safe_name) else ()),
                    )
                )
            )

            if _zip_member_is_link(info):
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=member_path,
                    container_type=current_type,
                    container_ancestry=container_ancestry,
                    concealment_reasons=concealment_reasons,
                    depth=depth,
                    reason=LedgerReason.ARCHIVE_LINK_MEMBER,
                    size_bytes=info.file_size,
                )
                _exception(result, path=virtual_path, reason=LedgerReason.ARCHIVE_LINK_MEMBER)
                continue
            if info.flag_bits & 0x1:
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=member_path,
                    container_type=current_type,
                    container_ancestry=container_ancestry,
                    concealment_reasons=concealment_reasons,
                    depth=depth,
                    reason=LedgerReason.ARCHIVE_ENCRYPTED,
                    size_bytes=info.file_size,
                )
                _exception(result, path=virtual_path, reason=LedgerReason.ARCHIVE_ENCRYPTED)
                continue

            compressed = max(info.compress_size, 1)
            if info.file_size > compressed * budget.max_compression_ratio:
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=member_path,
                    container_type=current_type,
                    container_ancestry=container_ancestry,
                    concealment_reasons=concealment_reasons,
                    depth=depth,
                    reason=LedgerReason.ARCHIVE_COMPRESSION_RATIO,
                    size_bytes=info.file_size,
                )
                _exception(
                    result,
                    path=virtual_path,
                    reason=LedgerReason.ARCHIVE_COMPRESSION_RATIO,
                    observed_bytes=info.file_size,
                    limit_bytes=compressed * budget.max_compression_ratio,
                )
                continue
            if budget.uncompressed_bytes + info.file_size > budget.max_uncompressed_bytes:
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=member_path,
                    container_type=current_type,
                    container_ancestry=container_ancestry,
                    concealment_reasons=concealment_reasons,
                    depth=depth,
                    reason=LedgerReason.ARCHIVE_SIZE_LIMIT,
                    size_bytes=info.file_size,
                )
                _exception(
                    result,
                    path=virtual_path,
                    reason=LedgerReason.ARCHIVE_SIZE_LIMIT,
                    observed_bytes=budget.uncompressed_bytes + info.file_size,
                    limit_bytes=budget.max_uncompressed_bytes,
                )
                continue
            if info.file_size > budget.max_member_bytes:
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=member_path,
                    container_type=current_type,
                    container_ancestry=container_ancestry,
                    concealment_reasons=concealment_reasons,
                    depth=depth,
                    reason=LedgerReason.ARCHIVE_MEMBER_SIZE_LIMIT,
                    size_bytes=info.file_size,
                )
                _exception(
                    result,
                    path=virtual_path,
                    reason=LedgerReason.ARCHIVE_MEMBER_SIZE_LIMIT,
                    observed_bytes=info.file_size,
                    limit_bytes=budget.max_member_bytes,
                )
                continue

            try:
                with archive.open(info) as source:
                    member_data = source.read(budget.max_member_bytes + 1)
            except NotImplementedError:
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=member_path,
                    container_type=current_type,
                    container_ancestry=container_ancestry,
                    concealment_reasons=concealment_reasons,
                    depth=depth,
                    reason=LedgerReason.ARCHIVE_UNSUPPORTED_COMPRESSION,
                    size_bytes=info.file_size,
                )
                _exception(
                    result,
                    path=virtual_path,
                    reason=LedgerReason.ARCHIVE_UNSUPPORTED_COMPRESSION,
                )
                continue
            except RuntimeError as exc:
                reason = (
                    LedgerReason.ARCHIVE_UNSUPPORTED_COMPRESSION
                    if "compress" in str(exc).lower() or "not supported" in str(exc).lower()
                    else LedgerReason.ARCHIVE_TRUNCATED
                )
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=member_path,
                    container_type=current_type,
                    container_ancestry=container_ancestry,
                    concealment_reasons=concealment_reasons,
                    depth=depth,
                    reason=reason,
                    size_bytes=info.file_size,
                )
                _exception(result, path=virtual_path, reason=reason)
                continue
            except (zipfile.BadZipFile, EOFError, OSError, ValueError):
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=member_path,
                    container_type=current_type,
                    container_ancestry=container_ancestry,
                    concealment_reasons=concealment_reasons,
                    depth=depth,
                    reason=LedgerReason.ARCHIVE_TRUNCATED,
                    size_bytes=info.file_size,
                )
                _exception(result, path=virtual_path, reason=LedgerReason.ARCHIVE_TRUNCATED)
                continue

            if len(member_data) > budget.max_member_bytes:
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=member_path,
                    container_type=current_type,
                    container_ancestry=container_ancestry,
                    concealment_reasons=concealment_reasons,
                    depth=depth,
                    reason=LedgerReason.ARCHIVE_MEMBER_SIZE_LIMIT,
                    size_bytes=len(member_data),
                )
                _exception(
                    result,
                    path=virtual_path,
                    reason=LedgerReason.ARCHIVE_MEMBER_SIZE_LIMIT,
                    observed_bytes=len(member_data),
                    limit_bytes=budget.max_member_bytes,
                )
                continue

            if budget.expired():
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=member_path,
                    container_type=current_type,
                    container_ancestry=container_ancestry,
                    concealment_reasons=concealment_reasons,
                    depth=depth,
                    reason=LedgerReason.ARCHIVE_TIME_LIMIT,
                    size_bytes=len(member_data),
                )
                budget.halted = True
                _time_exception(result, path=virtual_path, budget=budget)
                return
            if budget.uncompressed_bytes + len(member_data) > budget.max_uncompressed_bytes:
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=member_path,
                    container_type=current_type,
                    container_ancestry=container_ancestry,
                    concealment_reasons=concealment_reasons,
                    depth=depth,
                    reason=LedgerReason.ARCHIVE_SIZE_LIMIT,
                    size_bytes=len(member_data),
                )
                _exception(
                    result,
                    path=virtual_path,
                    reason=LedgerReason.ARCHIVE_SIZE_LIMIT,
                    observed_bytes=budget.uncompressed_bytes + len(member_data),
                    limit_bytes=budget.max_uncompressed_bytes,
                )
                return
            if len(member_data) > compressed * budget.max_compression_ratio:
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=member_path,
                    container_type=current_type,
                    container_ancestry=container_ancestry,
                    concealment_reasons=concealment_reasons,
                    depth=depth,
                    reason=LedgerReason.ARCHIVE_COMPRESSION_RATIO,
                    size_bytes=len(member_data),
                )
                _exception(
                    result,
                    path=virtual_path,
                    reason=LedgerReason.ARCHIVE_COMPRESSION_RATIO,
                    observed_bytes=len(member_data),
                    limit_bytes=compressed * budget.max_compression_ratio,
                )
                continue

            budget.uncompressed_bytes += len(member_data)
            nested_zip = _is_zip_signature(member_data)
            nested_type: str | None = "zip" if nested_zip else None

            executable = _member_executable(info, safe_name, member_data)
            member_hidden = _is_hidden_path(safe_name)
            concealed = executable and bool(concealment_reasons)
            virtual_type = _virtual_type(safe_name, member_data, nested_type)
            result.components.append(virtual_path)
            result.file_cache[virtual_path] = member_data.decode("utf-8", errors="replace")
            result.raw_file_cache[virtual_path] = member_data
            result.artifact_inventory.append(classify_artifact(virtual_path, member_data))
            result.metadata.append(
                {
                    "path": virtual_path,
                    "type": virtual_type,
                    "lines": (
                        0
                        if virtual_type == "binary"
                        else len(result.file_cache[virtual_path].splitlines())
                    ),
                    "executable": executable,
                    "size_bytes": len(member_data),
                    "outer_path": outer_path,
                    "nested_path": member_path,
                    "container_type": current_type,
                    "container_ancestry": list(container_ancestry),
                    "container_depth": depth,
                    "hidden": member_hidden,
                    "outer_hidden": outer_hidden,
                    "outer_disguised": effective_outer_disguised,
                    "local_only": True,
                    "concealed_executable": concealed,
                    "concealment_reasons": list(concealment_reasons),
                }
            )

            if not nested_zip:
                continue
            if depth >= budget.max_depth:
                _exception(
                    result,
                    path=virtual_path,
                    reason=LedgerReason.ARCHIVE_DEPTH_LIMIT,
                    observed_depth=depth + 1,
                    limit_depth=budget.max_depth,
                )
                continue
            _inspect_zip_bytes(
                member_data,
                outer_path=outer_path,
                container_virtual_path=virtual_path,
                depth=depth + 1,
                outer_hidden=outer_hidden,
                outer_disguised=effective_outer_disguised,
                outer_expected_type=outer_expected_type,
                ancestor_container_types=container_ancestry,
                budget=budget,
                result=result,
            )


def inspect_nested_artifacts(
    skill_dir: Path,
    components: list[str],
    *,
    clock: Callable[[], float] = time.monotonic,
    raw_file_cache: Mapping[str, bytes] | None = None,
    max_members: int | None = None,
    max_uncompressed_bytes: int | None = None,
    max_seconds: float | None = None,
    absolute_deadline: float | None = None,
) -> NestedInspectionResult:
    """Inspect ZIP-compatible components under one bundle-wide archive budget.

    ``raw_file_cache`` may provide bytes already read under the caller's bundle
    limits. Supplying it avoids a second filesystem read without weakening the
    archive-specific member, expansion, or deadline bounds. Callers may pass
    their remaining aggregate member/expanded-byte budgets and an absolute
    deadline in the same clock domain. ``None`` preserves the module defaults.
    """
    result = NestedInspectionResult()
    member_limit = (
        ARCHIVE_MAX_MEMBERS
        if max_members is None
        else min(ARCHIVE_MAX_MEMBERS, max(0, max_members))
    )
    byte_limit = (
        ARCHIVE_MAX_UNCOMPRESSED_BYTES
        if max_uncompressed_bytes is None
        else min(ARCHIVE_MAX_UNCOMPRESSED_BYTES, max(0, max_uncompressed_bytes))
    )
    started_at = clock()
    local_seconds = (
        ARCHIVE_MAX_SECONDS
        if max_seconds is None
        else min(ARCHIVE_MAX_SECONDS, max(0.0, max_seconds))
    )
    local_deadline = started_at + local_seconds
    deadline = (
        local_deadline if absolute_deadline is None else min(local_deadline, absolute_deadline)
    )
    budget = _Budget(
        clock=clock,
        max_members=member_limit,
        max_uncompressed_bytes=byte_limit,
        max_central_directory_bytes=ARCHIVE_MAX_CENTRAL_DIRECTORY_BYTES,
        max_member_bytes=MAX_FILE_BYTES,
        max_depth=ARCHIVE_MAX_DEPTH,
        max_compression_ratio=ARCHIVE_MAX_COMPRESSION_RATIO,
        deadline=deadline,
        started_at=started_at,
        runtime_limit=max(0.0, deadline - started_at),
        last_checked_at=started_at,
    )
    for path in dict.fromkeys(components):
        if budget.halted:
            break
        if budget.expired():
            budget.halted = True
            _time_exception(result, path=path, budget=budget)
            break
        full_path = skill_dir / path
        expected_type = _expected_container_type(path)
        hidden = _is_hidden_path(path)

        supplied = raw_file_cache is not None and path in raw_file_cache
        if supplied:
            data = raw_file_cache[path]
            size = len(data)
        else:
            try:
                size = full_path.stat().st_size
            except OSError:
                continue
        if not supplied and size > budget.max_uncompressed_bytes:
            # Only classify content that begins like ZIP; avoid reading arbitrary
            # large files merely to decide whether they are containers. Caller-
            # supplied bytes were already bounded and charged by the caller;
            # ``max_uncompressed_bytes`` applies to newly expanded members.
            try:
                with _open_regular_file_no_follow(full_path) as source:
                    signature = source.read(4)
            except (OSError, _FileOpenError, _UnsafeFileError):
                continue
            if _is_zip_signature(signature):
                _record_outer_metadata(
                    result,
                    path=path,
                    container_type=expected_type or "zip",
                    hidden=hidden,
                    disguised=expected_type is None,
                )
                _exception(
                    result,
                    path=path,
                    reason=LedgerReason.ARCHIVE_SIZE_LIMIT,
                    observed_bytes=size,
                    limit_bytes=budget.max_uncompressed_bytes,
                )
            elif expected_type is not None:
                _record_outer_metadata(
                    result,
                    path=path,
                    container_type=expected_type,
                    hidden=hidden,
                    disguised=False,
                )
                _exception(
                    result,
                    path=path,
                    reason=LedgerReason.ARCHIVE_FORMAT_MISMATCH,
                )
            continue
        if not supplied:
            try:
                with _open_regular_file_no_follow(full_path) as source:
                    data = source.read(budget.max_uncompressed_bytes + 1)
            except (OSError, _FileOpenError, _UnsafeFileError):
                continue
        if not _is_zip_signature(data):
            if expected_type is not None:
                _record_outer_metadata(
                    result,
                    path=path,
                    container_type=expected_type,
                    hidden=hidden,
                    disguised=False,
                )
                _exception(
                    result,
                    path=path,
                    reason=LedgerReason.ARCHIVE_FORMAT_MISMATCH,
                )
            continue
        # Record a conservative local-only identity before parsing the central
        # directory. The bounded inspector refines this after its early checks.
        _record_outer_metadata(
            result,
            path=path,
            container_type=expected_type or "zip",
            hidden=hidden,
            disguised=expected_type is None,
        )
        _inspect_zip_bytes(
            data,
            outer_path=path,
            container_virtual_path=path,
            depth=1,
            outer_hidden=hidden,
            outer_disguised=None,
            outer_expected_type=expected_type,
            ancestor_container_types=(),
            budget=budget,
            result=result,
        )

    result.components = list(dict.fromkeys(result.components))
    result.uncompressed_bytes = budget.uncompressed_bytes
    return result
