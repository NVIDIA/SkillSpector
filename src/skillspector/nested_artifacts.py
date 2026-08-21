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
    metadata: list[dict[str, object]] = field(default_factory=list)
    outer_metadata: dict[str, dict[str, object]] = field(default_factory=dict)
    ledger_events: list[InspectionLedgerEvent] = field(default_factory=list)
    uncompressed_bytes: int = 0


@dataclass
class _Budget:
    started_at: float
    clock: Callable[[], float]
    max_uncompressed_bytes: int = ARCHIVE_MAX_UNCOMPRESSED_BYTES
    max_seconds: float = ARCHIVE_MAX_SECONDS
    members: int = 0
    uncompressed_bytes: int = 0

    def expired(self) -> bool:
        return self.clock() - self.started_at > self.max_seconds


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
    container_ancestry: tuple[str, ...],
    concealment_reasons: tuple[str, ...],
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
        _exception(result, path=container_virtual_path, reason=LedgerReason.ARCHIVE_TIME_LIMIT)
        return
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError, ValueError):
        reason = LedgerReason.ARCHIVE_MALFORMED if depth == 1 else LedgerReason.ARCHIVE_TRUNCATED
        _exception(result, path=container_virtual_path, reason=reason)
        return

    with archive:
        if budget.expired():
            _exception(result, path=container_virtual_path, reason=LedgerReason.ARCHIVE_TIME_LIMIT)
            return

        # ZipFile has already parsed the central directory. Enforce the cumulative
        # entry budget before sorting or inspecting any attacker-controlled names.
        infos = archive.filelist
        remaining_members = ARCHIVE_MAX_MEMBERS - budget.members
        if len(infos) > remaining_members:
            _exception(
                result,
                path=container_virtual_path,
                reason=LedgerReason.ARCHIVE_MEMBER_LIMIT,
            )
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
                _exception(
                    result, path=container_virtual_path, reason=LedgerReason.ARCHIVE_TIME_LIMIT
                )
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
                )
                _exception(result, path=virtual_path, reason=LedgerReason.ARCHIVE_ENCRYPTED)
                continue

            compressed = max(info.compress_size, 1)
            if info.file_size > compressed * ARCHIVE_MAX_COMPRESSION_RATIO:
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=member_path,
                    container_type=current_type,
                    container_ancestry=container_ancestry,
                    concealment_reasons=concealment_reasons,
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
                )
                _exception(
                    result,
                    path=virtual_path,
                    reason=LedgerReason.ARCHIVE_SIZE_LIMIT,
                    observed_bytes=budget.uncompressed_bytes + info.file_size,
                    limit_bytes=budget.max_uncompressed_bytes,
                )
                continue
            if info.file_size > MAX_FILE_BYTES:
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=member_path,
                    container_type=current_type,
                    container_ancestry=container_ancestry,
                    concealment_reasons=concealment_reasons,
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
                )
                _exception(
                    result,
                    path=virtual_path,
                    reason=LedgerReason.ARCHIVE_UNSUPPORTED_COMPRESSION,
                )
                continue
            except RuntimeError as exc:
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=member_path,
                    container_type=current_type,
                    container_ancestry=container_ancestry,
                    concealment_reasons=concealment_reasons,
                    depth=depth,
                )
                reason = (
                    LedgerReason.ARCHIVE_UNSUPPORTED_COMPRESSION
                    if "compress" in str(exc).lower() or "not supported" in str(exc).lower()
                    else LedgerReason.ARCHIVE_TRUNCATED
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
                )
                _exception(result, path=virtual_path, reason=LedgerReason.ARCHIVE_TRUNCATED)
                continue

            if len(member_data) > MAX_FILE_BYTES:
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=member_path,
                    container_type=current_type,
                    container_ancestry=container_ancestry,
                    concealment_reasons=concealment_reasons,
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
                )
                _exception(result, path=virtual_path, reason=LedgerReason.ARCHIVE_TIME_LIMIT)
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
                )
                _exception(
                    result,
                    path=virtual_path,
                    reason=LedgerReason.ARCHIVE_SIZE_LIMIT,
                    observed_bytes=budget.uncompressed_bytes + len(member_data),
                    limit_bytes=budget.max_uncompressed_bytes,
                )
                return
            if len(member_data) > compressed * ARCHIVE_MAX_COMPRESSION_RATIO:
                _add_unreadable_component(
                    result,
                    virtual_path=virtual_path,
                    outer_path=outer_path,
                    member_path=member_path,
                    container_type=current_type,
                    container_ancestry=container_ancestry,
                    concealment_reasons=concealment_reasons,
                    depth=depth,
                )
                _exception(
                    result,
                    path=virtual_path,
                    reason=LedgerReason.ARCHIVE_COMPRESSION_RATIO,
                    observed_bytes=len(member_data),
                    limit_bytes=compressed * ARCHIVE_MAX_COMPRESSION_RATIO,
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
            if depth >= ARCHIVE_MAX_DEPTH:
                _exception(result, path=virtual_path, reason=LedgerReason.ARCHIVE_DEPTH_LIMIT)
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
    max_uncompressed_bytes: int | None = None,
    max_seconds: float | None = None,
) -> NestedInspectionResult:
    """Inspect ZIP-compatible filesystem components under cumulative bounds."""
    result = NestedInspectionResult()
    byte_limit = ARCHIVE_MAX_UNCOMPRESSED_BYTES
    if max_uncompressed_bytes is not None:
        byte_limit = min(byte_limit, max(0, max_uncompressed_bytes))
    time_limit = ARCHIVE_MAX_SECONDS
    if max_seconds is not None:
        time_limit = min(time_limit, max(0.0, max_seconds))
    budget = _Budget(
        started_at=clock(),
        clock=clock,
        max_uncompressed_bytes=byte_limit,
        max_seconds=time_limit,
    )
    for path in components:
        full_path = skill_dir / path
        expected_type = _expected_container_type(path)
        hidden = _is_hidden_path(path)

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
                    limit_bytes=ARCHIVE_MAX_UNCOMPRESSED_BYTES,
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
        try:
            with _open_regular_file_no_follow(full_path) as source:
                data = source.read(ARCHIVE_MAX_UNCOMPRESSED_BYTES + 1)
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
