# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, local-only inspection of ZIP, TAR and GZIP nested artifacts.

The inspector never extracts members to disk and never renders, imports, or
executes their contents.  It recognizes supported content by bytes, gives
every member a stable virtual path, and returns text projections solely for
deterministic analyzers.
"""

from __future__ import annotations

import io
import re
import stat
import struct
import tarfile
import time
import zipfile
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from skillspector.artifacts import (
    ArtifactDisposition,
    ArtifactRecord,
    ContentKind,
    classify_artifact,
    has_dex_magic,
    has_tar_header,
)
from skillspector.constants import MAX_ANALYZABLE_FILE_BYTES, MAX_FILE_BYTES
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
from skillspector.passive_svg import is_structurally_valid_passive_svg
from skillspector.python_ast import PythonSourceClassification, classify_python_source
from skillspector.resource_limits import bounded_capacity

ARCHIVE_MAX_DEPTH = 3
ARCHIVE_MAX_MEMBERS = bounded_capacity("SKILLSPECTOR_ARCHIVE_MAX_MEMBERS", 4_096, 10_000)
ARCHIVE_MAX_MEMBER_BYTES = bounded_capacity(
    "SKILLSPECTOR_ARCHIVE_MAX_MEMBER_BYTES", MAX_FILE_BYTES, MAX_ANALYZABLE_FILE_BYTES
)
ARCHIVE_MAX_UNCOMPRESSED_BYTES = 25 * 1024 * 1024
ARCHIVE_MAX_CENTRAL_DIRECTORY_BYTES = 4 * 1024 * 1024
ARCHIVE_MAX_COMPRESSION_RATIO = 100
ARCHIVE_MAX_SECONDS = 5.0
ARCHIVE_MAX_METADATA_DEPTH = 32
ARCHIVE_MAX_VIRTUAL_PATH_BYTES = 65535

_ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_EXECUTABLE_SUFFIXES = frozenset(
    {
        ".app",
        ".bash",
        ".bat",
        ".bin",
        ".cmd",
        ".com",
        ".cjs",
        ".class",
        ".cts",
        ".dll",
        ".dex",
        ".dylib",
        ".exe",
        ".go",
        ".js",
        ".jsx",
        ".luac",
        ".mjs",
        ".msi",
        ".mts",
        ".php",
        ".php3",
        ".php4",
        ".php5",
        ".pl",
        ".phtml",
        ".ps1",
        ".py",
        ".pyw",
        ".pyc",
        ".pyo",
        ".rb",
        ".rake",
        ".rs",
        ".sh",
        ".so",
        ".svg",
        ".ts",
        ".tsx",
        ".wasm",
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
    "tar": frozenset({".tar"}),
    "gzip": frozenset({".gz", ".tgz"}),
}


@dataclass
class NestedInspectionResult:
    """Virtual inventory and local-only content derived from nested artifacts."""

    components: list[str] = field(default_factory=list)
    file_cache: dict[str, str] = field(default_factory=dict)
    raw_file_cache: dict[str, bytes] = field(default_factory=dict)
    # Classify with the archive member's execution path, while retaining the
    # virtual path as the stable cache/report key used by downstream analyzers.
    python_source_classifications: dict[str, PythonSourceClassification] = field(
        default_factory=dict
    )
    artifact_inventory: list[ArtifactRecord] = field(default_factory=list)
    metadata: list[dict[str, object]] = field(default_factory=list)
    outer_metadata: dict[str, dict[str, object]] = field(default_factory=dict)
    # Byte-recognized ZIPs at every depth, including ones stopped by a limit.
    # Expected extensions and format mismatches are not content recognition.
    recognized_zip_paths: set[str] = field(default_factory=set)
    recognized_container_paths: set[str] = field(default_factory=set)
    # Only complete bounded traversals, including every nested member read.
    inspected_containers: dict[str, list[str]] = field(default_factory=dict)
    ledger_events: list[InspectionLedgerEvent] = field(default_factory=list)
    # Keep the actual event objects: disk and virtual paths can share both a
    # spelling and a work ID, so neither identifies which input produced it.
    disk_container_events: list[InspectionLedgerEvent] = field(default_factory=list)
    _active_disk_path: str | None = None
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
    metadata_fields: int = 0
    gzip_metadata_bytes: int = 0
    virtual_identity_bytes: int = 0
    outer_path: str = ""
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
        LedgerReason.ARCHIVE_MEMBER_INCOMPLETE,
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
        LedgerReason.ARCHIVE_UNSUPPORTED_MEMBER,
        LedgerReason.OPAQUE_CONTENT,
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
    raw_locator = struct.unpack_from("<IQI", data, locator_offset + 4)
    locator_disk = int(raw_locator[0])
    reported_offset = int(raw_locator[1])
    total_disks = int(raw_locator[2])
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
        record_size = int(struct.unpack_from("<Q", data, offset + 4)[0])
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


def is_zip_content(data: bytes) -> bool:
    """Return whether bytes begin with a supported ZIP-family signature."""
    return data.startswith(_ZIP_SIGNATURES)


def _archive_type(data: bytes) -> str | None:
    if is_zip_content(data):
        return "zip"
    if data.startswith(b"\x1f\x8b"):
        return "gzip"
    if has_tar_header(data):
        return "tar"
    return None


def is_archive_content(data: bytes) -> bool:
    """Recognize supported archives without opening or expanding their contents."""
    return _archive_type(data) is not None


def _is_hidden_path(path: str) -> bool:
    return any(part.startswith(".") for part in path.replace("\\", "/").split("/") if part)


def _container_type(names: list[str]) -> str:
    normalized = [name.replace("\\", "/").lower() for name in names]
    if "[content_types].xml" in normalized:
        for marker, container_type in _OOXML_MARKERS:
            if any(name.startswith(marker) for name in normalized):
                return container_type
    return "zip"


def expected_container_type(path: str) -> str | None:
    """Return the supported container type implied by a filename, if any."""
    suffix = Path(path).suffix.lower()
    return next(
        (
            container_type
            for container_type, suffixes in _EXPECTED_SUFFIXES.items()
            if suffix in suffixes
        ),
        None,
    )


_expected_container_type = expected_container_type


def _safe_member_name(name: str) -> str | None:
    normalized = name.replace("\\", "/")
    if (
        not normalized
        or "\x00" in normalized
        or any(0xD800 <= ord(character) <= 0xDFFF for character in normalized)
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


def _member_name(
    name: str, *, directory: bool, seen: dict[str, bool], container: str, budget: _Budget
) -> tuple[str | None, LedgerReason | None]:
    """Share path and namespace validation across archive formats."""
    if directory and name.rstrip("/") == ".":
        return None, None
    try:
        _member_identity_size(container, name, budget)
    except UnicodeEncodeError:
        return None, LedgerReason.ARCHIVE_UNSAFE_MEMBER_PATH
    safe = _safe_member_name(name)
    if safe is None:
        return None, LedgerReason.ARCHIVE_UNSAFE_MEMBER_PATH
    for existing, existing_directory in seen.items():
        _check_archive_time(budget)
        parent_file = (
            not existing_directory
            and len(existing) < len(safe)
            and safe.startswith(existing)
            and safe[len(existing)] == "/"
        )
        existing_child = (
            not directory
            and len(safe) < len(existing)
            and existing.startswith(safe)
            and existing[len(safe)] == "/"
        )
        if safe == existing or parent_file or existing_child:
            return safe, LedgerReason.ARCHIVE_AMBIGUOUS_MEMBER_PATH
    seen[safe] = directory
    return safe, None


def _zip_member_is_link(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return bool(mode and stat.S_ISLNK(mode))


_BINARY_EXECUTABLE_MAGICS = (
    b"MZ",
    b"\x7fELF",
    b"\x00asm",
    b"\x1bLua",
    b"\xfe\xed\xfa",
    b"\xce\xfa\xed\xfe",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
)

_TYPESCRIPT_DECLARATION_SUFFIXES = (".d.ts", ".d.cts", ".d.mts")
_TYPESCRIPT_AMBIENT_MEMBER = (
    r"(?:export\s+)?(?:"
    r"interface\s+[A-Za-z_$][\w$]*(?:\s*<[^\r\n{}]*>)?\s*\{[^\r\n{}]*\}|"
    r"type\s+[A-Za-z_$][\w$]*(?:\s*<[^\r\n;{}]*>)?\s*=\s*[^\r\n;{}]+;|"
    r"(?:const|let|var)\s+[A-Za-z_$][\w$]*(?:\s*:\s*[^\r\n;{}]+)?\s*;|"
    r"function\s+[A-Za-z_$][\w$]*\s*\([^\r\n{}]*\)\s*:\s*[^\r\n;{}]+;|"
    r"class\s+[A-Za-z_$][\w$]*\s*\{[^\r\n{}]*\}"
    r")"
)
_TYPESCRIPT_DECLARATION_FILE = re.compile(
    r"\A\s*(?:(?:"
    r"(?:(?:export\s+)?declare\s+(?:const|let|var|function|class)\b[^\r\n;{}]*;)|"
    rf"(?:declare\s+(?:export\s+)?namespace\s+[A-Za-z_$][\w$]*\s*\{{(?:\s*{_TYPESCRIPT_AMBIENT_MEMBER})*\s*\}}\s*;?)|"
    rf"(?:declare\s+module\s+(?:[\"'][^\"']+[\"']|[A-Za-z_$][\w$]*)\s*\{{(?:\s*{_TYPESCRIPT_AMBIENT_MEMBER})*\s*\}}\s*;?)|"
    r"(?:(?:export\s+)?interface\s+[A-Za-z_$][\w$]*(?:\s*<[^\r\n{}]*>)?\s*\{[^\r\n{}]*\}\s*;?)|"
    r"(?:(?:export\s+)?type\s+[A-Za-z_$][\w$]*(?:\s*<[^;{}]*>)?\s*=\s*[^;\r\n{}]+;)|"
    r"(?:import\s+type\s+(?:[A-Za-z_$][\w$]*(?:\s*,\s*\{[^{}]*\})?|\{[^{}]*\}|\*\s+as\s+[A-Za-z_$][\w$]*)\s+from\s+[\"'][^\"'\r\n]+[\"']\s*;)|"
    r"(?:export\s*\{[^{}]*\}\s*;?)"
    r")\s*)+\Z",
    re.DOTALL,
)


def _strip_typescript_comments(text: str) -> str | None:
    """Remove comments without interpreting comment markers inside literals."""
    result: list[str] = []
    index = 0
    while index < len(text):
        character = text[index]
        if character in "'\"`":
            quote = character
            start = index
            index += 1
            while index < len(text):
                if text[index] == "\\":
                    index += 2
                    continue
                if quote == "`" and text.startswith("${", index):
                    # Interpolated templates need a JavaScript parser to prove
                    # inert; fail closed rather than hide their expression.
                    return None
                if text[index] == quote:
                    index += 1
                    break
                index += 1
            else:
                return None
            result.append(text[start:index])
            continue
        if text.startswith("//", index):
            while index < len(text) and text[index] not in "\r\n":
                result.append(" ")
                index += 1
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end < 0:
                return None
            comment = text[index : end + 2]
            result.extend(
                "\n" if char == "\n" else "\r" if char == "\r" else " " for char in comment
            )
            index = end + 2
            continue
        result.append(character)
        index += 1
    return "".join(result)


def _looks_like_typescript_declaration(path: str, data: bytes, *, complete_content: bool) -> bool:
    """Recognize clearly inert TypeScript declaration content conservatively.

    Declaration suffixes alone are not trusted: a file named ``evil.d.cts``
    can still contain executable CommonJS. Unknown or non-text content stays
    executable so this check cannot create a name-based security bypass.
    """
    name = Path(path).name.lower()
    if not complete_content or not name.endswith(_TYPESCRIPT_DECLARATION_SUFFIXES) or not data:
        return False
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    uncommented = _strip_typescript_comments(text)
    return bool(
        uncommented and uncommented.strip() and _TYPESCRIPT_DECLARATION_FILE.fullmatch(uncommented)
    )


def has_binary_executable_magic(data: bytes) -> bool:
    """Return whether canonical bytes begin with supported executable magic."""
    return has_dex_magic(data) or data.startswith(_BINARY_EXECUTABLE_MAGICS)


def is_executable_content(
    path: str, data: bytes, mode: int = 0, *, complete_content: bool = False
) -> bool:
    """Classify content; exemptions require complete, bounded structure proofs."""
    suffix = Path(path).suffix.lower()
    executable_magic = data.startswith(b"#!") or has_binary_executable_magic(data)
    declaration_only = _looks_like_typescript_declaration(
        path, data, complete_content=complete_content
    )
    passive_svg = suffix == ".svg" and complete_content and is_structurally_valid_passive_svg(data)
    return (
        (suffix in _EXECUTABLE_SUFFIXES and not declaration_only and not passive_svg)
        or executable_magic
        or bool(mode & 0o111)
    )


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


def _virtual_type(
    path: str,
    data: bytes,
    nested_type: str | None,
    source_classification: PythonSourceClassification,
) -> str:
    if nested_type is not None:
        return nested_type
    if source_classification is PythonSourceClassification.PYTHON:
        return "python"
    suffix = Path(path).suffix.lower()
    return {
        ".md": "markdown",
        ".mdc": "markdown",
        ".markdown": "markdown",
        ".py": "python",
        ".pyw": "python",
        ".sh": "shell",
        ".bash": "shell",
        ".zsh": "shell",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".txt": "text",
        ".js": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
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


def _apply_inventory_overrides(result: NestedInspectionResult) -> None:
    """Reconcile all exception dispositions in one linear inventory pass."""
    for artifact in result.artifact_inventory:
        override = result.inventory_overrides.get(artifact["path"])
        if override is not None:
            artifact["disposition"], artifact["reason"] = override


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
    event = ledger_event(
        outcome=(
            LedgerOutcome.PARTIAL if reason in _PARTIAL_INVENTORY_REASONS else LedgerOutcome.SKIPPED
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
    result.ledger_events.append(event)
    if path == result._active_disk_path:
        result.disk_container_events.append(event)


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
        artifact: ArtifactRecord = {
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
        result.artifact_inventory.append(artifact)
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


def _record_member(
    data: bytes,
    *,
    name: str,
    mode: int,
    outer_path: str,
    container_virtual_path: str,
    container_type: str,
    container_ancestry: tuple[str, ...],
    depth: int,
    outer_hidden: bool,
    outer_disguised: bool,
    concealment_reasons: tuple[str, ...],
    budget: _Budget,
    result: NestedInspectionResult,
    complete_content: bool = True,
    virtual_path: str | None = None,
) -> str:
    """Publish one validated member through the same inventory and local-only route."""
    if virtual_path is None:
        virtual_path = _member_path(container_virtual_path, name, budget)
    if virtual_path in result.file_cache:
        result.inspected_containers.pop(container_virtual_path, None)
        _exception(
            result,
            path=container_virtual_path,
            reason=LedgerReason.ARCHIVE_AMBIGUOUS_MEMBER_PATH,
        )
        return virtual_path
    nested_type = _archive_type(data)
    executable = is_executable_content(name, data, mode, complete_content=complete_content)
    source_classification = classify_python_source(name, data)
    virtual_type = _virtual_type(name, data, nested_type, source_classification)
    result.components.append(virtual_path)
    result.file_cache[virtual_path] = data.decode("utf-8", errors="replace")
    result.raw_file_cache[virtual_path] = data
    result.python_source_classifications[virtual_path] = source_classification
    result.artifact_inventory.append(classify_artifact(virtual_path, data))
    if not complete_content:
        _exception(result, path=virtual_path, reason=LedgerReason.ARCHIVE_MEMBER_INCOMPLETE)
    result.metadata.append(
        {
            "path": virtual_path,
            "type": virtual_type,
            "lines": 0
            if virtual_type == "binary"
            else len(result.file_cache[virtual_path].splitlines()),
            "executable": executable,
            "size_bytes": len(data),
            "outer_path": outer_path,
            "nested_path": _nested_path(outer_path, virtual_path),
            "container_type": container_type,
            "container_ancestry": list(container_ancestry),
            "container_depth": depth,
            "hidden": _is_hidden_path(name),
            "outer_hidden": outer_hidden,
            "outer_disguised": outer_disguised,
            "local_only": True,
            "concealed_executable": executable and bool(concealment_reasons),
            "concealment_reasons": list(concealment_reasons),
        }
    )
    if nested_type is None:
        return virtual_path
    result.recognized_container_paths.add(virtual_path)
    if nested_type == "zip":
        result.recognized_zip_paths.add(virtual_path)
    if depth >= budget.max_depth:
        _exception(
            result,
            path=virtual_path,
            reason=LedgerReason.ARCHIVE_DEPTH_LIMIT,
            observed_depth=depth + 1,
            limit_depth=budget.max_depth,
        )
        return virtual_path
    _inspect_archive_bytes(
        data,
        outer_path=outer_path,
        container_virtual_path=virtual_path,
        depth=depth + 1,
        outer_hidden=outer_hidden,
        outer_disguised=outer_disguised,
        outer_expected_type=None,
        ancestor_container_types=container_ancestry,
        budget=budget,
        result=result,
    )
    return virtual_path


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
    initial_exceptions = len(result.ledger_events)
    member_paths: list[str] = []
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
        seen_names: dict[str, bool] = {}

        for info in infos:
            if budget.expired():
                budget.halted = True
                _time_exception(result, path=container_virtual_path, budget=budget)
                return
            directory_member = info.is_dir() and not _zip_member_is_link(info)
            safe_name, name_reason = _member_name(
                info.filename,
                directory=directory_member,
                seen=seen_names,
                container=container_virtual_path,
                budget=budget,
            )
            virtual_path = (
                _member_path(container_virtual_path, safe_name, budget)
                if safe_name is not None
                else container_virtual_path
            )
            if name_reason is not None:
                _exception(
                    result,
                    path=virtual_path,
                    reason=name_reason,
                )
                continue
            if safe_name is None or directory_member:
                continue
            if virtual_path in result.file_cache:
                _exception(
                    result,
                    # A different disk container may already own this path.
                    # Attribute that skipped copy to its current container.
                    path=container_virtual_path,
                    reason=LedgerReason.ARCHIVE_AMBIGUOUS_MEMBER_PATH,
                )
                continue
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

            if len(member_data) != info.file_size:
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
            member_paths.append(
                _record_member(
                    member_data,
                    name=safe_name,
                    mode=info.external_attr >> 16,
                    outer_path=outer_path,
                    container_virtual_path=container_virtual_path,
                    container_type=current_type,
                    container_ancestry=container_ancestry,
                    depth=depth,
                    outer_hidden=outer_hidden,
                    outer_disguised=effective_outer_disguised,
                    concealment_reasons=concealment_reasons,
                    budget=budget,
                    result=result,
                    virtual_path=virtual_path,
                )
            )
            if budget.halted:
                return

    if len(result.ledger_events) == initial_exceptions:
        result.inspected_containers[container_virtual_path] = member_paths


class _ArchiveInspectionError(Exception):
    def __init__(
        self, reason: LedgerReason, *, report_path: str | None = None, **metrics: int | float
    ) -> None:
        self.reason = reason
        self.metrics = metrics
        self.report_path = report_path


def _member_identity_size(container: str, name: str, budget: _Budget) -> int:
    """Bound raw and canonical spellings before normalization or concatenation."""

    def byte_size(value: str) -> int:
        if value.isascii():
            return len(value)
        count = 0
        for offset in range(0, len(value), 65536):
            _check_archive_time(budget)
            count += len(value[offset : offset + 65536].encode("utf-8"))
        return count

    size = byte_size(container) + 2 + byte_size(name)
    per_path_limit = min(ARCHIVE_MAX_VIRTUAL_PATH_BYTES, budget.max_central_directory_bytes)
    if size > per_path_limit:
        budget.halted = True
        raise _ArchiveInspectionError(
            LedgerReason.ARCHIVE_SIZE_LIMIT,
            report_path=budget.outer_path,
            observed_bytes=size,
            limit_bytes=per_path_limit,
        )
    return size


def _member_path(container: str, name: str, budget: _Budget) -> str:
    """Reserve exact identities before concatenating attacker-controlled ancestors."""
    size = _member_identity_size(container, name, budget)
    observed = budget.virtual_identity_bytes + size
    if observed > budget.max_central_directory_bytes:
        budget.halted = True
        raise _ArchiveInspectionError(
            LedgerReason.ARCHIVE_SIZE_LIMIT,
            report_path=budget.outer_path,
            observed_bytes=observed,
            limit_bytes=budget.max_central_directory_bytes,
        )
    budget.virtual_identity_bytes += size
    return f"{container}!/{name}"


def _check_archive_time(budget: _Budget) -> None:
    if budget.expired():
        budget.halted = True
        raise _ArchiveInspectionError(
            LedgerReason.ARCHIVE_TIME_LIMIT,
            observed_seconds=budget.elapsed,
            limit_seconds=budget.runtime_limit,
        )


def _charge_archive_member(budget: _Budget) -> None:
    _check_archive_time(budget)
    if budget.members >= budget.max_members:
        budget.halted = True
        raise _ArchiveInspectionError(
            LedgerReason.ARCHIVE_MEMBER_LIMIT,
            observed_artifacts=budget.members + 1,
            limit_artifacts=budget.max_members,
        )
    budget.members += 1


class _TarReader(io.BytesIO):
    """Bound stdlib header processing before reads, allocation and chained metadata."""

    def __init__(self, data: bytes, budget: _Budget) -> None:
        super().__init__(data)
        self.data = data
        self.budget = budget
        self.metadata_bytes = 0
        self.metadata_depth = 0

    def read(self, size: int = -1) -> bytes:
        _check_archive_time(self.budget)
        if size < 0 or self.metadata_bytes + size > self.budget.max_central_directory_bytes:
            raise _ArchiveInspectionError(
                LedgerReason.ARCHIVE_SIZE_LIMIT,
                observed_bytes=self.metadata_bytes + (len(self.data) if size < 0 else size),
                limit_bytes=self.budget.max_central_directory_bytes,
            )
        content = super().read(size)
        self.metadata_bytes += len(content)
        return content

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        _check_archive_time(self.budget)
        absolute = offset + (
            self.tell() if whence == io.SEEK_CUR else len(self.data) if whence == io.SEEK_END else 0
        )
        if absolute < 0 or absolute > len(self.data):
            raise _ArchiveInspectionError(LedgerReason.ARCHIVE_TRUNCATED)
        return super().seek(offset, whence)

    def check_pax(self, size: int) -> None:
        """Bound field expansion and reject sparse semantics before stdlib builds them."""
        if size < 0 or self.metadata_bytes + size > self.budget.max_central_directory_bytes:
            raise _ArchiveInspectionError(
                LedgerReason.ARCHIVE_SIZE_LIMIT,
                observed_bytes=self.metadata_bytes + max(size, 0),
                limit_bytes=self.budget.max_central_directory_bytes,
            )
        start = self.tell()
        end = start + size
        if end > len(self.data):
            raise _ArchiveInspectionError(LedgerReason.ARCHIVE_TRUNCATED)
        while start < end:
            _check_archive_time(self.budget)
            separator = self.data.find(b" ", start, min(end, start + 12))
            if separator < 0:
                raise _ArchiveInspectionError(LedgerReason.ARCHIVE_MALFORMED)
            digits = self.data[start:separator]
            if not digits.isdigit():
                raise _ArchiveInspectionError(LedgerReason.ARCHIVE_MALFORMED)
            length = int(digits)
            stop = start + length
            equals = self.data.find(b"=", separator + 1, min(stop, end))
            if length < 5 or stop > end or equals <= separator + 1 or self.data[stop - 1] != 10:
                raise _ArchiveInspectionError(LedgerReason.ARCHIVE_MALFORMED)
            self.budget.metadata_fields += 1
            if self.budget.metadata_fields > self.budget.max_members:
                raise _ArchiveInspectionError(
                    LedgerReason.ARCHIVE_MEMBER_LIMIT,
                    observed_artifacts=self.budget.metadata_fields,
                    limit_artifacts=self.budget.max_members,
                )
            key = self.data[separator + 1 : equals]
            if key.startswith(b"GNU.sparse.") or key == b"SCHILY.realsize":
                raise _ArchiveInspectionError(LedgerReason.ARCHIVE_UNSUPPORTED_MEMBER)
            start = stop


class _BoundedTarInfo(tarfile.TarInfo):
    def _proc_member(self, archive: tarfile.TarFile) -> _BoundedTarInfo:
        """Guard both stdlib header readers and their metadata continuations."""
        reader = archive.fileobj
        if not isinstance(reader, _TarReader):
            raise TypeError("TAR inspection requires a bounded metadata reader")
        _charge_archive_member(reader.budget)
        if self.size < 0:
            raise _ArchiveInspectionError(LedgerReason.ARCHIVE_MALFORMED)
        if self.type == tarfile.GNUTYPE_SPARSE:
            raise _ArchiveInspectionError(LedgerReason.ARCHIVE_UNSUPPORTED_MEMBER)
        reader.metadata_depth += 1
        try:
            if reader.metadata_depth > ARCHIVE_MAX_METADATA_DEPTH:
                raise _ArchiveInspectionError(
                    LedgerReason.ARCHIVE_DEPTH_LIMIT,
                    observed_depth=reader.metadata_depth,
                    limit_depth=ARCHIVE_MAX_METADATA_DEPTH,
                )
            if self.type in {tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.SOLARIS_XHDTYPE}:
                reader.check_pax(self.size)
            return super()._proc_member(archive)
        finally:
            reader.metadata_depth -= 1


def _container_concealment(
    ancestry: tuple[str, ...], *, hidden: bool, disguised: bool
) -> tuple[str, ...]:
    return (
        *(
            ("document_container",)
            if any(kind in {"docx", "xlsx", "pptx"} for kind in ancestry)
            else ()
        ),
        *(("hidden_artifact",) if hidden else ()),
        *(("disguised_container",) if disguised else ()),
    )


def _inspect_tar_bytes(
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
    initial_exceptions = len(result.ledger_events)
    disguised = outer_disguised if outer_disguised is not None else outer_expected_type != "tar"
    ancestry = (*ancestor_container_types, "tar")
    inherited = _container_concealment(ancestry, hidden=outer_hidden, disguised=disguised)
    names: dict[str, bool] = {}
    member_paths: list[str] = []
    try:
        with (
            _TarReader(data, budget) as reader,
            tarfile.open(
                fileobj=reader,
                mode="r:",
                tarinfo=_BoundedTarInfo,
                encoding="utf-8",
                errors="surrogateescape",
            ) as archive,
        ):
            for info in archive:
                _check_archive_time(budget)
                safe_name, name_reason = _member_name(
                    info.name,
                    directory=info.isdir(),
                    seen=names,
                    container=container_virtual_path,
                    budget=budget,
                )
                virtual_path = (
                    _member_path(container_virtual_path, safe_name, budget)
                    if safe_name is not None
                    else container_virtual_path
                )
                if name_reason is not None:
                    _exception(result, path=virtual_path, reason=name_reason)
                    continue
                if safe_name is None:
                    continue
                if virtual_path in result.file_cache:
                    _exception(
                        result,
                        path=container_virtual_path,
                        reason=LedgerReason.ARCHIVE_AMBIGUOUS_MEMBER_PATH,
                    )
                    continue
                if info.isdir():
                    continue
                concealment = (
                    *inherited,
                    *(
                        ("hidden_artifact",)
                        if _is_hidden_path(safe_name) and "hidden_artifact" not in inherited
                        else ()
                    ),
                )
                reason: LedgerReason | None = None
                metrics: dict[str, int] = {}
                end = info.offset_data + info.size
                if (
                    info.type in {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.CONTTYPE}
                    and info.size >= 0
                ):
                    # Apply effective PAX size overrides to the next header too.
                    archive.offset = info.offset_data + ((info.size + 511) // 512) * 512
                if info.issym() or info.islnk():
                    reason = LedgerReason.ARCHIVE_LINK_MEMBER
                elif (
                    info.type not in {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.CONTTYPE}
                    or info.sparse is not None
                ):
                    reason = LedgerReason.ARCHIVE_UNSUPPORTED_MEMBER
                elif info.size < 0 or info.offset_data < 0 or end > len(data):
                    reason = LedgerReason.ARCHIVE_TRUNCATED
                elif info.size > budget.max_member_bytes:
                    reason = LedgerReason.ARCHIVE_MEMBER_SIZE_LIMIT
                    metrics = {"observed_bytes": info.size, "limit_bytes": budget.max_member_bytes}
                elif budget.uncompressed_bytes + info.size > budget.max_uncompressed_bytes:
                    reason = LedgerReason.ARCHIVE_SIZE_LIMIT
                    metrics = {
                        "observed_bytes": budget.uncompressed_bytes + info.size,
                        "limit_bytes": budget.max_uncompressed_bytes,
                    }
                if reason is not None:
                    _add_unreadable_component(
                        result,
                        virtual_path=virtual_path,
                        outer_path=outer_path,
                        member_path=_nested_path(outer_path, virtual_path),
                        container_type="tar",
                        container_ancestry=ancestry,
                        concealment_reasons=concealment,
                        depth=depth,
                        reason=reason,
                        size_bytes=info.size,
                    )
                    _exception(result, path=virtual_path, reason=reason, **metrics)
                    continue
                budget.uncompressed_bytes += info.size
                member_paths.append(
                    _record_member(
                        data[info.offset_data : end],
                        name=safe_name,
                        mode=info.mode,
                        outer_path=outer_path,
                        container_virtual_path=container_virtual_path,
                        container_type="tar",
                        container_ancestry=ancestry,
                        depth=depth,
                        outer_hidden=outer_hidden,
                        outer_disguised=disguised,
                        concealment_reasons=concealment,
                        budget=budget,
                        result=result,
                        virtual_path=virtual_path,
                    )
                )
                if budget.halted:
                    return
            # TarFile.next() accepts missing trailers and malformed later headers.
            # Only two terminal blocks plus zero padding prove the suffix inspected.
            trailer = archive.offset
            if len(data) - trailer < 1024 or len(data) % tarfile.BLOCKSIZE:
                raise _ArchiveInspectionError(LedgerReason.ARCHIVE_TRUNCATED)
            for offset in range(trailer, len(data), 65536):
                _check_archive_time(budget)
                if any(memoryview(data)[offset : offset + 65536]):
                    raise _ArchiveInspectionError(LedgerReason.ARCHIVE_MALFORMED)
    except _ArchiveInspectionError as exc:
        _exception(
            result, path=exc.report_path or container_virtual_path, reason=exc.reason, **exc.metrics
        )
        return
    except (tarfile.TarError, EOFError, OSError, ValueError, OverflowError):
        _exception(
            result,
            path=container_virtual_path,
            reason=LedgerReason.ARCHIVE_MALFORMED if depth == 1 else LedgerReason.ARCHIVE_TRUNCATED,
        )
        return
    if len(result.ledger_events) == initial_exceptions:
        result.inspected_containers[container_virtual_path] = member_paths


def _gzip_header_end(data: bytes, start: int, budget: _Budget) -> int:
    """Preflight RFC 1952 metadata before handing any header to zlib."""
    _check_archive_time(budget)

    def consume(size: int) -> None:
        observed = budget.gzip_metadata_bytes + size
        if observed > budget.max_central_directory_bytes:
            raise _ArchiveInspectionError(
                LedgerReason.ARCHIVE_SIZE_LIMIT,
                observed_bytes=observed,
                limit_bytes=budget.max_central_directory_bytes,
            )
        budget.gzip_metadata_bytes = observed

    if len(data) - start < 10:
        raise _ArchiveInspectionError(LedgerReason.ARCHIVE_TRUNCATED)
    consume(10)
    if data[start : start + 2] != b"\x1f\x8b" or data[start + 2] != 8 or data[start + 3] & 0xE0:
        raise _ArchiveInspectionError(LedgerReason.ARCHIVE_MALFORMED)
    flags = data[start + 3]
    cursor = start + 10
    if flags & 4:
        if len(data) - cursor < 2:
            raise _ArchiveInspectionError(LedgerReason.ARCHIVE_TRUNCATED)
        consume(2)
        size = struct.unpack_from("<H", data, cursor)[0]
        cursor += 2
        if size > len(data) - cursor:
            raise _ArchiveInspectionError(LedgerReason.ARCHIVE_TRUNCATED)
        consume(size)
        cursor += size
    for flag in (8, 16):
        if not flags & flag:
            continue
        while True:
            _check_archive_time(budget)
            remaining = budget.max_central_directory_bytes - budget.gzip_metadata_bytes
            stop = min(len(data), cursor + min(65536, remaining + 1))
            terminator = data.find(b"\x00", cursor, stop)
            end = terminator + 1 if terminator >= 0 else stop
            consume(end - cursor)
            cursor = end
            if terminator >= 0:
                break
            if cursor == len(data):
                raise _ArchiveInspectionError(LedgerReason.ARCHIVE_TRUNCATED)
    if flags & 2:
        if len(data) - cursor < 2:
            raise _ArchiveInspectionError(LedgerReason.ARCHIVE_TRUNCATED)
        consume(2)
        cursor += 2
    return cursor


def _inspect_gzip_bytes(
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
    initial_exceptions = len(result.ledger_events)
    expanded = bytearray()
    cursor = 0
    verified_length = 0
    try:
        while cursor < len(data):
            _check_archive_time(budget)
            if data[cursor] == 0:
                padding = data[cursor : cursor + 65536]
                cursor += len(padding) - len(padding.lstrip(b"\x00"))
                continue
            if not data.startswith(b"\x1f\x8b", cursor):
                raise _ArchiveInspectionError(LedgerReason.ARCHIVE_MALFORMED)
            _charge_archive_member(budget)
            _gzip_header_end(data, cursor, budget)
            frame_start, expanded_start = cursor, len(expanded)
            decoder = zlib.decompressobj(wbits=31)
            while not decoder.eof:
                _check_archive_time(budget)
                stop = min(len(data), cursor + 65536)
                if stop == cursor:
                    raise _ArchiveInspectionError(LedgerReason.ARCHIVE_TRUNCATED)
                remaining = budget.max_uncompressed_bytes - budget.uncompressed_bytes
                block = decoder.decompress(data[cursor:stop], min(65536, remaining + 1))
                cursor = stop - len(decoder.unused_data if decoder.eof else decoder.unconsumed_tail)
                if len(block) > remaining:
                    raise _ArchiveInspectionError(
                        LedgerReason.ARCHIVE_SIZE_LIMIT,
                        observed_bytes=budget.uncompressed_bytes + len(block),
                        limit_bytes=budget.max_uncompressed_bytes,
                    )
                expanded.extend(block)
                budget.uncompressed_bytes += len(block)
            compressed = max(1, cursor - frame_start)
            if len(expanded) - expanded_start > compressed * budget.max_compression_ratio:
                raise _ArchiveInspectionError(
                    LedgerReason.ARCHIVE_COMPRESSION_RATIO,
                    observed_bytes=len(expanded) - expanded_start,
                    limit_bytes=compressed * budget.max_compression_ratio,
                )
            verified_length = len(expanded)
    except _ArchiveInspectionError as exc:
        _exception(
            result, path=exc.report_path or container_virtual_path, reason=exc.reason, **exc.metrics
        )
        if not verified_length:
            return
        del expanded[verified_length:]
    except (zlib.error, OverflowError, ValueError):
        _exception(result, path=container_virtual_path, reason=LedgerReason.ARCHIVE_MALFORMED)
        if not verified_length:
            return
        del expanded[verified_length:]
    disguised = outer_disguised if outer_disguised is not None else outer_expected_type != "gzip"
    ancestry = (*ancestor_container_types, "gzip")
    content = bytes(expanded)
    if _archive_type(content) is not None:
        if depth >= budget.max_depth:
            _exception(
                result,
                path=container_virtual_path,
                reason=LedgerReason.ARCHIVE_DEPTH_LIMIT,
                observed_depth=depth + 1,
                limit_depth=budget.max_depth,
            )
            return
        _inspect_archive_bytes(
            content,
            outer_path=outer_path,
            container_virtual_path=container_virtual_path,
            depth=depth + 1,
            outer_hidden=outer_hidden,
            outer_disguised=disguised,
            outer_expected_type=None,
            ancestor_container_types=ancestry,
            budget=budget,
            result=result,
        )
        if len(result.ledger_events) != initial_exceptions:
            result.inspected_containers.pop(container_virtual_path, None)
        return
    if len(content) > budget.max_member_bytes:
        _exception(
            result,
            path=container_virtual_path,
            reason=LedgerReason.ARCHIVE_MEMBER_SIZE_LIMIT,
            observed_bytes=len(content),
            limit_bytes=budget.max_member_bytes,
        )
        return
    name = container_virtual_path.rsplit("/", 1)[-1]
    name = name[:-3] if name.lower().endswith(".gz") else "content"
    name = _safe_member_name(name) or "content"
    path = _record_member(
        content,
        name=name,
        mode=0,
        outer_path=outer_path,
        container_virtual_path=container_virtual_path,
        container_type="gzip",
        container_ancestry=ancestry,
        depth=depth,
        outer_hidden=outer_hidden,
        outer_disguised=disguised,
        concealment_reasons=_container_concealment(
            ancestry, hidden=outer_hidden, disguised=disguised
        ),
        budget=budget,
        result=result,
        complete_content=len(result.ledger_events) == initial_exceptions,
    )
    if len(result.ledger_events) == initial_exceptions:
        result.inspected_containers[container_virtual_path] = [path]


def _inspect_archive_bytes(
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
    kind = _archive_type(data)
    if kind is None:
        _exception(result, path=container_virtual_path, reason=LedgerReason.ARCHIVE_FORMAT_MISMATCH)
        return
    result.recognized_container_paths.add(container_virtual_path)
    if kind == "zip":
        result.recognized_zip_paths.add(container_virtual_path)
    if depth == 1:
        _record_outer_metadata(
            result,
            path=outer_path,
            container_type=kind,
            hidden=outer_hidden,
            disguised=outer_expected_type != kind,
        )
    inspector = {"zip": _inspect_zip_bytes, "tar": _inspect_tar_bytes, "gzip": _inspect_gzip_bytes}[
        kind
    ]
    try:
        inspector(
            data,
            outer_path=outer_path,
            container_virtual_path=container_virtual_path,
            depth=depth,
            outer_hidden=outer_hidden,
            outer_disguised=outer_disguised,
            outer_expected_type=outer_expected_type,
            ancestor_container_types=ancestor_container_types,
            budget=budget,
            result=result,
        )
    except _ArchiveInspectionError as exc:
        _exception(
            result, path=exc.report_path or container_virtual_path, reason=exc.reason, **exc.metrics
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
    """Inspect supported components under one bundle-wide archive budget.

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
        max_member_bytes=ARCHIVE_MAX_MEMBER_BYTES,
        max_depth=ARCHIVE_MAX_DEPTH,
        max_compression_ratio=ARCHIVE_MAX_COMPRESSION_RATIO,
        deadline=deadline,
        started_at=started_at,
        runtime_limit=max(0.0, deadline - started_at),
        last_checked_at=started_at,
    )
    for path in dict.fromkeys(components):
        result._active_disk_path = path
        budget.outer_path = path
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
        if raw_file_cache is not None and path in raw_file_cache:
            data = raw_file_cache[path]
            size = len(data)
        else:
            try:
                size = full_path.stat().st_size
            except OSError:
                continue
        if not supplied and size > budget.max_uncompressed_bytes:
            # Only classify content that begins like an archive; avoid reading arbitrary
            # large files merely to decide whether they are containers. Caller-
            # supplied bytes were already bounded and charged by the caller;
            # ``max_uncompressed_bytes`` applies to newly expanded members.
            try:
                with _open_regular_file_no_follow(full_path) as source:
                    signature = source.read(1024)
            except (OSError, _FileOpenError, _UnsafeFileError):
                continue
            if (kind := _archive_type(signature)) is not None:
                result.recognized_container_paths.add(path)
                if kind == "zip":
                    result.recognized_zip_paths.add(path)
                _record_outer_metadata(
                    result,
                    path=path,
                    container_type=kind,
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
        if not is_archive_content(data):
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
        _inspect_archive_bytes(
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

    result._active_disk_path = None
    result.components = list(dict.fromkeys(result.components))
    result.uncompressed_bytes = budget.uncompressed_bytes
    _apply_inventory_overrides(result)
    return result
