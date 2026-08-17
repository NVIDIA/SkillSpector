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
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

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
ARCHIVE_MAX_COMPRESSION_RATIO = 100
ARCHIVE_MAX_SECONDS = 5.0

_ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_EXECUTABLE_SUFFIXES = frozenset(
    {".py", ".sh", ".bash", ".zsh", ".js", ".ts", ".rb", ".go", ".rs", ".pl"}
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
    metadata: list[dict[str, object]] = field(default_factory=list)
    outer_metadata: dict[str, dict[str, object]] = field(default_factory=dict)
    ledger_events: list[InspectionLedgerEvent] = field(default_factory=list)


@dataclass
class _Budget:
    started_at: float
    clock: Callable[[], float]
    members: int = 0
    uncompressed_bytes: int = 0

    def expired(self) -> bool:
        return self.clock() - self.started_at > ARCHIVE_MAX_SECONDS


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


def _safe_member_name(name: str) -> str | None:
    normalized = name.replace("\\", "/")
    if not normalized or "\x00" in normalized or normalized.startswith(("/", "//")):
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


def _member_executable(info: zipfile.ZipInfo, safe_name: str, data: bytes) -> bool:
    suffix = Path(safe_name).suffix.lower()
    mode = info.external_attr >> 16
    return suffix in _EXECUTABLE_SUFFIXES or data.startswith(b"#!") or bool(mode & 0o111)


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


def _exception(
    result: NestedInspectionResult,
    *,
    path: str,
    reason: LedgerReason,
    observed_bytes: int | None = None,
    limit_bytes: int | None = None,
) -> None:
    result.ledger_events.append(
        ledger_event(
            outcome=LedgerOutcome.SKIPPED,
            record_type=LedgerRecordType.SYSTEM,
            phase="nested_artifact_inspection",
            path=path,
            reason=reason,
            observed_bytes=observed_bytes,
            limit_bytes=limit_bytes,
        )
    )


def _add_unreadable_component(
    result: NestedInspectionResult,
    *,
    virtual_path: str,
    outer_path: str,
    member_path: str,
    container_type: str,
    depth: int,
) -> None:
    if virtual_path not in result.file_cache:
        result.components.append(virtual_path)
        # A binary sentinel lets ordinary analyzers account for the component
        # without pretending that inaccessible bytes were inspected as text.
        result.file_cache[virtual_path] = "\x00"
        result.metadata.append(
            {
                "path": virtual_path,
                "type": "binary",
                "lines": 0,
                "executable": False,
                "size_bytes": 0,
                "outer_path": outer_path,
                "nested_path": member_path,
                "container_type": container_type,
                "container_depth": depth,
                "hidden": _is_hidden_path(member_path),
                "local_only": True,
                "concealed_executable": False,
            }
        )


def _inspect_zip_bytes(
    data: bytes,
    *,
    outer_path: str,
    container_virtual_path: str,
    depth: int,
    outer_hidden: bool,
    outer_disguised: bool,
    budget: _Budget,
    result: NestedInspectionResult,
) -> None:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError, ValueError):
        _exception(result, path=container_virtual_path, reason=LedgerReason.ARCHIVE_MALFORMED)
        return

    with archive:
        infos = sorted(archive.infolist(), key=lambda item: item.filename)
        current_type = _container_type([info.filename for info in infos])

        for info in infos:
            if info.is_dir():
                continue
            if budget.expired():
                _exception(
                    result, path=container_virtual_path, reason=LedgerReason.ARCHIVE_TIME_LIMIT
                )
                return
            if budget.members >= ARCHIVE_MAX_MEMBERS:
                _exception(
                    result, path=container_virtual_path, reason=LedgerReason.ARCHIVE_MEMBER_LIMIT
                )
                return
            budget.members += 1

            safe_name = _safe_member_name(info.filename)
            if safe_name is None:
                _exception(
                    result,
                    path=container_virtual_path,
                    reason=LedgerReason.ARCHIVE_UNSAFE_MEMBER_PATH,
                )
                continue
            virtual_path = f"{container_virtual_path}!/{safe_name}"

            if _zip_member_is_link(info):
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=safe_name,
                    container_type=current_type,
                    depth=depth,
                )
                _exception(result, path=virtual_path, reason=LedgerReason.ARCHIVE_LINK_MEMBER)
                continue
            if info.flag_bits & 0x1:
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=safe_name,
                    container_type=current_type,
                    depth=depth,
                )
                _exception(result, path=virtual_path, reason=LedgerReason.ARCHIVE_ENCRYPTED)
                continue

            compressed = max(info.compress_size, 1)
            if info.file_size > compressed * ARCHIVE_MAX_COMPRESSION_RATIO:
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=safe_name,
                    container_type=current_type,
                    depth=depth,
                )
                _exception(
                    result,
                    path=virtual_path,
                    reason=LedgerReason.ARCHIVE_COMPRESSION_RATIO,
                    observed_bytes=info.file_size,
                    limit_bytes=compressed * ARCHIVE_MAX_COMPRESSION_RATIO,
                )
                continue
            if budget.uncompressed_bytes + info.file_size > ARCHIVE_MAX_UNCOMPRESSED_BYTES:
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=safe_name,
                    container_type=current_type,
                    depth=depth,
                )
                _exception(
                    result,
                    path=virtual_path,
                    reason=LedgerReason.ARCHIVE_SIZE_LIMIT,
                    observed_bytes=budget.uncompressed_bytes + info.file_size,
                    limit_bytes=ARCHIVE_MAX_UNCOMPRESSED_BYTES,
                )
                continue
            if info.file_size > MAX_FILE_BYTES:
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=safe_name,
                    container_type=current_type,
                    depth=depth,
                )
                _exception(
                    result,
                    path=virtual_path,
                    reason=LedgerReason.ARCHIVE_MEMBER_SIZE_LIMIT,
                    observed_bytes=info.file_size,
                    limit_bytes=MAX_FILE_BYTES,
                )
                continue

            try:
                with archive.open(info) as source:
                    member_data = source.read(MAX_FILE_BYTES + 1)
            except RuntimeError:
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=safe_name,
                    container_type=current_type,
                    depth=depth,
                )
                _exception(result, path=virtual_path, reason=LedgerReason.ARCHIVE_ENCRYPTED)
                continue
            except (zipfile.BadZipFile, EOFError, OSError, ValueError):
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=safe_name,
                    container_type=current_type,
                    depth=depth,
                )
                _exception(result, path=virtual_path, reason=LedgerReason.ARCHIVE_TRUNCATED)
                continue

            if len(member_data) > MAX_FILE_BYTES:
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=safe_name,
                    container_type=current_type,
                    depth=depth,
                )
                _exception(
                    result,
                    path=virtual_path,
                    reason=LedgerReason.ARCHIVE_MEMBER_SIZE_LIMIT,
                    observed_bytes=len(member_data),
                    limit_bytes=MAX_FILE_BYTES,
                )
                continue

            budget.uncompressed_bytes += len(member_data)
            nested_type: str | None = None
            nested_zip = _is_zip_signature(member_data)
            if nested_zip and zipfile.is_zipfile(io.BytesIO(member_data)):
                try:
                    with zipfile.ZipFile(io.BytesIO(member_data)) as nested_archive:
                        nested_type = _container_type(
                            [nested_info.filename for nested_info in nested_archive.infolist()]
                        )
                except (zipfile.BadZipFile, OSError, ValueError):
                    nested_type = "zip"

            executable = _member_executable(info, safe_name, member_data)
            member_hidden = _is_hidden_path(safe_name)
            concealed = executable and (
                current_type in {"docx", "xlsx", "pptx"}
                or outer_hidden
                or outer_disguised
                or member_hidden
            )
            virtual_type = _virtual_type(safe_name, member_data, nested_type)
            result.components.append(virtual_path)
            result.file_cache[virtual_path] = member_data.decode("utf-8", errors="replace")
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
                    "nested_path": safe_name,
                    "container_type": current_type,
                    "container_depth": depth,
                    "hidden": member_hidden,
                    "outer_hidden": outer_hidden,
                    "outer_disguised": outer_disguised,
                    "local_only": True,
                    "concealed_executable": concealed,
                }
            )

            if not nested_zip:
                continue
            if not zipfile.is_zipfile(io.BytesIO(member_data)):
                _exception(result, path=virtual_path, reason=LedgerReason.ARCHIVE_MALFORMED)
                continue
            if depth >= ARCHIVE_MAX_DEPTH:
                _exception(result, path=virtual_path, reason=LedgerReason.ARCHIVE_DEPTH_LIMIT)
                continue
            _inspect_zip_bytes(
                member_data,
                outer_path=outer_path,
                container_virtual_path=virtual_path,
                depth=depth + 1,
                outer_hidden=outer_hidden,
                outer_disguised=outer_disguised,
                budget=budget,
                result=result,
            )


def inspect_nested_artifacts(
    skill_dir: Path,
    components: list[str],
    *,
    clock: Callable[[], float] = time.monotonic,
) -> NestedInspectionResult:
    """Inspect ZIP-compatible filesystem components under cumulative bounds."""
    result = NestedInspectionResult()
    for path in components:
        full_path = skill_dir / path
        try:
            size = full_path.stat().st_size
        except OSError:
            continue
        if size > ARCHIVE_MAX_UNCOMPRESSED_BYTES:
            # Only classify content that begins like ZIP; avoid reading arbitrary
            # large files merely to decide whether they are containers.
            try:
                with _open_regular_file_no_follow(full_path) as source:
                    signature = source.read(4)
            except (OSError, _FileOpenError, _UnsafeFileError):
                continue
            if _is_zip_signature(signature):
                _exception(
                    result,
                    path=path,
                    reason=LedgerReason.ARCHIVE_SIZE_LIMIT,
                    observed_bytes=size,
                    limit_bytes=ARCHIVE_MAX_UNCOMPRESSED_BYTES,
                )
            continue
        try:
            with _open_regular_file_no_follow(full_path) as source:
                data = source.read(ARCHIVE_MAX_UNCOMPRESSED_BYTES + 1)
        except (OSError, _FileOpenError, _UnsafeFileError):
            continue
        if not _is_zip_signature(data):
            continue
        if not zipfile.is_zipfile(io.BytesIO(data)):
            _exception(result, path=path, reason=LedgerReason.ARCHIVE_MALFORMED)
            continue

        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                container_type = _container_type([info.filename for info in archive.infolist()])
        except (zipfile.BadZipFile, OSError, ValueError):
            _exception(result, path=path, reason=LedgerReason.ARCHIVE_MALFORMED)
            continue

        hidden = _is_hidden_path(path)
        disguised = Path(path).suffix.lower() not in _EXPECTED_SUFFIXES[container_type]
        result.outer_metadata[path] = {
            "type": container_type,
            "container_type": container_type,
            "hidden": hidden,
            "disguised": disguised,
            "local_only": hidden or disguised,
        }
        budget = _Budget(started_at=clock(), clock=clock)
        _inspect_zip_bytes(
            data,
            outer_path=path,
            container_virtual_path=path,
            depth=1,
            outer_hidden=hidden,
            outer_disguised=disguised,
            budget=budget,
            result=result,
        )

    result.components = list(dict.fromkeys(result.components))
    return result
