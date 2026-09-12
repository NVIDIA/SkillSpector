# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Input handler for Skillspector.

Handles various input formats:
- Git repository URLs
- Raw file URLs
- Local zip files
- Single markdown files
- Local directories

Each remote/archive ingest path is bounded by ``INGEST_MAX_BYTES`` and
``INGEST_MAX_ZIP_MEMBERS`` so that the per-file analysis caps downstream
of ``InputHandler.resolve()`` are not defeated by an oversized download,
a zip bomb, or a too-large git clone.  This file fails closed on any
ingest budget breach (closes #21 / #131).

URL-based ingest is additionally gated by an SSRF host allowlist plus a
private-IP check, and zip extraction is guarded against zip-slip.
"""

from __future__ import annotations

import ipaddress
import os
import re
import shutil
import socket
import struct
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from errno import ELOOP, ENOENT, ENOTDIR
from pathlib import Path, PurePosixPath
from stat import S_IFMT, S_ISDIR, S_ISLNK, S_ISREG
from time import monotonic
from typing import BinaryIO, NoReturn, cast
from urllib.parse import urljoin, urlparse

import httpx

from skillspector.logging_config import get_logger

logger = get_logger(__name__)

_HAS_SECURE_DIR_FD = os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW")
_IS_WINDOWS = os.name == "nt"

ALLOWED_GIT_HOSTS = frozenset(
    {
        "github.com",
        "gitlab.com",
        "bitbucket.org",
    }
)

ALLOWED_DOWNLOAD_HOSTS = frozenset(
    {
        "github.com",
        "raw.githubusercontent.com",
        "gitlab.com",
        "bitbucket.org",
        "huggingface.co",
    }
)
_DIRECT_FILE_URL_SUFFIXES = (
    ".md",
    ".py",
    ".sh",
)

# Hard ceiling on what any single ingest path can pull into the temp dir.
# Sized above the per-file analysis cap (``MAX_FILE_BYTES`` = 1 MB) so a
# legitimate multi-file skill is not blocked at ingest, but tight enough
# to bound memory / disk DoS from a malicious source.
INGEST_MAX_BYTES = 100 * 1024 * 1024  # 100 MiB

# Hard ceiling on the number of members in a zip we are willing to
# extract.  Catches the "many tiny files" zip-bomb variant where each
# entry is small but the entry count itself exhausts the filesystem.
INGEST_MAX_ZIP_MEMBERS = 10_000

# Bounds the metadata which ``zipfile.ZipFile.infolist()`` may materialize.
# Entry count alone is insufficient because names, comments, and extra fields
# are attacker-controlled variable-length records in the central directory.
INGEST_MAX_ZIP_CENTRAL_DIRECTORY_BYTES = 16 * 1024 * 1024  # 16 MiB
INGEST_MAX_ZIP_PATH_BYTES = 4 * 1024
INGEST_MAX_ZIP_PATH_DEPTH = 64

# Bounds the post-clone filesystem walk, including ``.git`` objects.  The walk
# is iterative and only retains at most this many ``DirEntry`` objects.
INGEST_MAX_TREE_ENTRIES = 10_000

# Wall-clock bound for local post-ingest inspection and extraction.  A shared
# transitive deadline may reduce this further.
INGEST_MAX_SECONDS = 60.0

_ZIP_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_ZIP_EOCD_MIN_BYTES = 22
_ZIP_MAX_COMMENT_BYTES = 65_535
_ZIP64_LOCATOR_BYTES = 20
_COPY_CHUNK_BYTES = 64 * 1024
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "AUX",
        "CLOCK$",
        "CON",
        "CONIN$",
        "CONOUT$",
        "NUL",
        "PRN",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)


class IngestLimitExceededError(ValueError):
    """Raised when an ingest path exceeds an ``INGEST_MAX_*`` budget.

    Subclass of ``ValueError`` so existing callers that catch
    ``ValueError`` from ``InputHandler.resolve()`` continue to work.
    """


@dataclass(frozen=True, slots=True)
class IngestTruncation:
    """Sanitized machine-readable description of a transitive ingest truncation."""

    code: str
    source_type: str
    message: str

    def as_dict(self) -> dict[str, str]:
        """Return a state-safe representation without URLs or local paths."""
        return {
            "code": self.code,
            "source_type": self.source_type,
            "message": self.message,
        }


class TransitiveIngestTruncatedError(IngestLimitExceededError):
    """Signal that a transitive input was intentionally not materialized.

    The exception is public and typed so graph/CLI callers can mark the source
    incomplete.  Its payload deliberately excludes attacker-controlled URLs,
    paths, HTTP bodies, and subprocess stderr.
    """

    def __init__(self, code: str, source_type: str) -> None:
        message = f"Transitive {source_type} ingest truncated ({code})"
        self.truncation = IngestTruncation(
            code=code,
            source_type=source_type,
            message=message,
        )
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class _TreeMeasurement:
    """Bounded clone-tree measurement."""

    entries: int
    content_bytes: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class _ZipDirectoryMetadata:
    """Small EOCD-derived ZIP metadata read before central-dir materialization."""

    entries: int
    central_directory_bytes: int
    central_directory_offset: int


def _is_private_ip(host: str) -> bool:
    """Return True if host resolves to a private/reserved IP address."""
    try:
        addr = ipaddress.ip_address(host)
        return addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved
    except ValueError:
        pass
    try:
        resolved = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _, _, _, _, sockaddr in resolved:
            addr = ipaddress.ip_address(sockaddr[0])
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                return True
    except (socket.gaierror, OSError):
        return True
    return False


def _root_owned_root_alias(path: Path) -> Path | None:
    """Return a root-owned symlink directly below ``/``, if *path* is one."""
    absolute_path = Path(os.path.abspath(path))
    if absolute_path.anchor != os.path.sep or len(absolute_path.parts) != 2:
        return None
    try:
        path_stat = absolute_path.lstat()
    except OSError:
        return None
    if S_ISLNK(path_stat.st_mode) and path_stat.st_uid == 0:
        return absolute_path
    return None


def _normalize_root_owned_alias(path: Path) -> Path:
    """Resolve a trusted root-level system alias while retaining child path components."""
    absolute_path = Path(os.path.abspath(path))
    if absolute_path.anchor != os.path.sep or len(absolute_path.parts) < 3:
        return absolute_path
    root_alias = _root_owned_root_alias(Path(absolute_path.anchor, absolute_path.parts[1]))
    if root_alias is None:
        return absolute_path
    try:
        return root_alias.resolve(strict=True).joinpath(*absolute_path.parts[2:])
    except OSError:
        return absolute_path


def _has_symlinked_parent(path: Path) -> bool:
    """Return whether any parent of *path* is a symlink or Windows junction."""
    current = Path(path.anchor)
    for part in path.parts[1:-1]:
        current /= part
        try:
            if current.is_symlink() or current.is_junction():
                return True
        except OSError:
            return True
    return False


class _UnsafeFileError(ValueError):
    """Raised when a path cannot be safely treated as a regular file."""


class _FileOpenError(ValueError):
    """Raised when an otherwise safe file cannot be opened for operational reasons."""

    def __init__(self, file_path: Path, cause: OSError) -> None:
        super().__init__(f"Could not safely open file: {file_path}")
        self.error_class = type(cause).__name__


def validate_local_input_path(path: Path) -> Path:
    """Normalize a local input path after rejecting symlinks and their ancestors."""
    if path.is_symlink() and _root_owned_root_alias(path) is None:
        raise ValueError(f"Refusing to resolve a symlinked input: {path}")
    if path.is_junction():
        raise ValueError(f"Refusing to resolve a junctioned input: {path}")
    normalized_path = _normalize_root_owned_alias(path)
    if _has_symlinked_parent(normalized_path):
        raise ValueError(f"Refusing to resolve input with a symlinked parent: {path}")
    return normalized_path


def _open_regular_file_no_follow(file_path: Path) -> BinaryIO:
    """Open a regular file without following symlinks.

    Descriptor-relative opens protect every path component against replacement
    races. Windows uses a reparse-point handle and validates the opened handle's
    canonical path before exposing its contents.
    """
    absolute_path = _normalize_root_owned_alias(file_path)
    if _has_symlinked_parent(absolute_path):
        raise _UnsafeFileError(f"Could not safely open file: {file_path}")
    if _HAS_SECURE_DIR_FD:
        return _open_regular_file_from_trusted_directory(absolute_path)
    if _IS_WINDOWS:
        return _open_regular_file_from_windows_handle(absolute_path)
    raise _UnsafeFileError(
        f"Secure no-follow file opens are unavailable on this platform: {file_path}"
    )


def _open_regular_file_from_trusted_directory(file_path: Path) -> BinaryIO:
    """Open *file_path* one non-symlinked component at a time."""
    # Traversal only needs search access on each component. Prefer O_PATH where it
    # exists (Linux); O_RDONLY additionally demands read access, which sandboxes
    # such as Landlock withhold on "/". Elsewhere (e.g. macOS) this is O_RDONLY,
    # i.e. 0, leaving the flags unchanged.
    directory_flags = os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_PATH", os.O_RDONLY)
    directory_fd: int | None = None
    try:
        directory_fd = os.open(file_path.anchor, directory_flags)
        for part in file_path.parts[1:-1]:
            next_directory_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            _close_fd_safely(directory_fd)
            directory_fd = next_directory_fd
        source_fd = os.open(
            file_path.name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_fd,
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}") from None
    except OSError as exc:
        if exc.errno in {ELOOP, ENOTDIR}:
            raise _UnsafeFileError(f"Could not safely open file: {file_path}") from exc
        raise _FileOpenError(file_path, exc) from exc
    finally:
        if directory_fd is not None:
            _close_fd_safely(directory_fd)

    return _fdopen_regular_file(source_fd, file_path)


def _open_regular_file_from_windows_handle(file_path: Path) -> BinaryIO:
    """Open a regular Windows file without traversing a reparse point.

    ``FILE_FLAG_OPEN_REPARSE_POINT`` prevents a final symlink or junction from
    being dereferenced. ``GetFinalPathNameByHandleW`` then detects an ancestor
    that changed into a reparse point after the initial parent check, before the
    opened handle can be used to read content.
    """
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class _ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    get_file_information = kernel32.GetFileInformationByHandle
    get_file_information.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ByHandleFileInformation)]
    get_file_information.restype = wintypes.BOOL
    get_final_path_name = kernel32.GetFinalPathNameByHandleW
    get_final_path_name.argtypes = [
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    get_final_path_name.restype = wintypes.DWORD
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    generic_read = 0x80000000
    file_share_all = 0x00000001 | 0x00000002 | 0x00000004
    open_existing = 3
    file_attribute_reparse_point = 0x00000400
    file_flag_open_reparse_point = 0x00200000
    invalid_handle_value = ctypes.c_void_p(-1).value

    handle = create_file(
        os.fspath(file_path),
        generic_read,
        file_share_all,
        None,
        open_existing,
        file_flag_open_reparse_point,
        None,
    )
    if handle == invalid_handle_value:
        error = _windows_last_error()
        if error.errno == ENOENT:
            raise FileNotFoundError(f"File not found: {file_path}") from None
        raise _FileOpenError(file_path, error)

    try:
        information = _ByHandleFileInformation()
        if not get_file_information(handle, ctypes.byref(information)):
            raise _FileOpenError(file_path, _windows_last_error())
        if information.dwFileAttributes & file_attribute_reparse_point:
            raise _UnsafeFileError(f"Could not safely open file: {file_path}")

        opened_path = _windows_final_path_name(get_final_path_name, handle, file_path)
        if _windows_normalized_path(opened_path) != _windows_normalized_path(os.fspath(file_path)):
            raise _UnsafeFileError(f"Could not safely open file: {file_path}")

        source_fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)  # type: ignore[attr-defined]
    except BaseException:
        close_handle(handle)
        raise

    return _fdopen_regular_file(source_fd, file_path)


def _windows_final_path_name(get_final_path_name: object, handle: int, file_path: Path) -> str:
    """Return the canonical DOS path for an already-open Windows handle."""
    import ctypes

    buffer_size = 260
    while True:
        buffer = ctypes.create_unicode_buffer(buffer_size)
        result = cast(int, get_final_path_name(handle, buffer, buffer_size, 0))  # type: ignore[operator]
        if result == 0:
            raise _FileOpenError(file_path, _windows_last_error())
        if result < buffer_size:
            return buffer.value
        buffer_size = result + 1


def _windows_last_error() -> OSError:
    """Return the current Windows error as an ``OSError`` instance."""
    import ctypes

    return cast(OSError, ctypes.WinError(ctypes.get_last_error()))  # type: ignore[attr-defined]


def _windows_normalized_path(path: str) -> str:
    """Normalize a Windows DOS path for an exact opened-handle comparison."""
    long_path_prefix = "\\\\?\\"
    long_unc_prefix = "\\\\?\\UNC\\"
    if path.startswith(long_unc_prefix):
        path = "\\\\" + path[len(long_unc_prefix) :]
    elif path.startswith(long_path_prefix):
        path = path[len(long_path_prefix) :]
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _close_fd_safely(fd: int) -> None:
    """Close a descriptor without masking the operation that owns it."""
    try:
        os.close(fd)
    except OSError:
        pass


def _fdopen_regular_file(source_fd: int, file_path: Path) -> BinaryIO:
    """Transfer an opened descriptor to a validated binary file object."""
    try:
        source = os.fdopen(source_fd, "rb")
    except OSError as exc:
        _close_fd_safely(source_fd)
        raise _FileOpenError(file_path, exc) from exc
    try:
        if not S_ISREG(os.fstat(source.fileno()).st_mode):
            raise _UnsafeFileError(f"Refusing to open a symlinked or non-regular file: {file_path}")
    except OSError as exc:
        source.close()
        raise _FileOpenError(file_path, exc) from exc
    except BaseException:
        source.close()
        raise
    return source


def _find_zip_eocd(archive_file: BinaryIO) -> tuple[int, tuple[int, ...]]:
    """Locate and parse the terminal EOCD using a fixed-size tail read."""
    archive_file.seek(0, os.SEEK_END)
    archive_size = archive_file.tell()
    if archive_size < _ZIP_EOCD_MIN_BYTES:
        raise zipfile.BadZipFile("File is not a zip file")

    tail_size = min(archive_size, _ZIP_EOCD_MIN_BYTES + _ZIP_MAX_COMMENT_BYTES)
    tail_offset = archive_size - tail_size
    archive_file.seek(tail_offset)
    tail = archive_file.read(tail_size)

    # A signature can occur inside the user-controlled ZIP comment.  Accept
    # only a candidate whose declared comment ends exactly at EOF.
    search_end = len(tail)
    while True:
        index = tail.rfind(_ZIP_EOCD_SIGNATURE, 0, search_end)
        if index < 0:
            raise zipfile.BadZipFile("End-of-central-directory record not found")
        if index + _ZIP_EOCD_MIN_BYTES <= len(tail):
            fields = struct.unpack_from("<4s4H2LH", tail, index)
            comment_length = fields[-1]
            if index + _ZIP_EOCD_MIN_BYTES + comment_length == len(tail):
                # Exclude the signature and comment length from the normalized
                # integer tuple returned to the caller.
                return tail_offset + index, cast(tuple[int, ...], fields[1:-1])
        search_end = index


def _read_zip64_metadata(archive_file: BinaryIO, eocd_offset: int) -> _ZipDirectoryMetadata:
    """Read fixed-size ZIP64 locator/EOCD fields without loading the directory."""
    locator_offset = eocd_offset - _ZIP64_LOCATOR_BYTES
    if locator_offset < 0:
        raise zipfile.BadZipFile("ZIP64 locator is missing")
    archive_file.seek(locator_offset)
    locator = archive_file.read(_ZIP64_LOCATOR_BYTES)
    if len(locator) != _ZIP64_LOCATOR_BYTES:
        raise zipfile.BadZipFile("Truncated ZIP64 locator")
    signature, zip64_disk, zip64_offset, disk_count = struct.unpack("<4sLQL", locator)
    if signature != _ZIP64_LOCATOR_SIGNATURE:
        raise zipfile.BadZipFile("ZIP64 locator is missing")
    if zip64_disk != 0 or disk_count != 1:
        raise zipfile.BadZipFile("Multi-disk ZIP archives are not supported")
    if zip64_offset < 0 or zip64_offset + 56 > locator_offset:
        raise zipfile.BadZipFile("Invalid ZIP64 directory offset")

    archive_file.seek(zip64_offset)
    record = archive_file.read(56)
    if len(record) != 56:
        raise zipfile.BadZipFile("Truncated ZIP64 end-of-central-directory record")
    (
        signature,
        record_size,
        _version_made,
        _version_needed,
        disk_number,
        directory_disk,
        entries_on_disk,
        entries,
        directory_size,
        directory_offset,
    ) = struct.unpack("<4sQ2H2L4Q", record)
    if signature != _ZIP64_EOCD_SIGNATURE or record_size < 44:
        raise zipfile.BadZipFile("Invalid ZIP64 end-of-central-directory record")
    if zip64_offset + 12 + record_size > locator_offset:
        raise zipfile.BadZipFile("Invalid ZIP64 record size")
    if disk_number != 0 or directory_disk != 0 or entries_on_disk != entries:
        raise zipfile.BadZipFile("Multi-disk ZIP archives are not supported")
    if directory_offset + directory_size > zip64_offset:
        raise zipfile.BadZipFile("Invalid ZIP64 central-directory bounds")
    return _ZipDirectoryMetadata(
        entries=entries,
        central_directory_bytes=directory_size,
        central_directory_offset=directory_offset,
    )


def _read_zip_directory_metadata(archive_file: BinaryIO) -> _ZipDirectoryMetadata:
    """Read EOCD/ZIP64 counts and directory bounds with constant memory."""
    eocd_offset, fields = _find_zip_eocd(archive_file)
    (
        disk_number,
        directory_disk,
        entries_on_disk,
        entries,
        directory_size,
        directory_offset,
    ) = fields
    if disk_number != 0 or directory_disk != 0 or entries_on_disk != entries:
        raise zipfile.BadZipFile("Multi-disk ZIP archives are not supported")

    requires_zip64 = (
        entries == 0xFFFF
        or entries_on_disk == 0xFFFF
        or directory_size == 0xFFFFFFFF
        or directory_offset == 0xFFFFFFFF
    )
    if requires_zip64:
        return _read_zip64_metadata(archive_file, eocd_offset)
    if directory_offset + directory_size > eocd_offset:
        raise zipfile.BadZipFile("Invalid central-directory bounds")
    return _ZipDirectoryMetadata(
        entries=entries,
        central_directory_bytes=directory_size,
        central_directory_offset=directory_offset,
    )


def _safe_zip_target(extract_root: Path, member_name: str) -> Path:
    """Return a contained extraction path or reject an ambiguous member name."""
    if not member_name or "\x00" in member_name or "\\" in member_name:
        raise ValueError("Zip entry has an unsafe or ambiguous path (zip-slip)")
    if re.match(r"^[A-Za-z]:", member_name):
        raise ValueError("Zip entry has an absolute drive path (zip-slip)")

    normalized_name = member_name[:-1] if member_name.endswith("/") else member_name
    raw_parts = normalized_name.split("/")
    encoded_length = len(normalized_name.encode("utf-8", errors="surrogatepass"))
    pure_path = PurePosixPath(normalized_name)
    has_windows_ambiguous_part = any(
        ":" in part
        or part.rstrip(" .") != part
        or part.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES
        for part in raw_parts
    )
    if (
        not normalized_name
        or encoded_length > INGEST_MAX_ZIP_PATH_BYTES
        or len(raw_parts) > INGEST_MAX_ZIP_PATH_DEPTH
        or pure_path.is_absolute()
        or any(part in {"", ".", ".."} for part in raw_parts)
        or has_windows_ambiguous_part
    ):
        raise ValueError("Zip entry would escape extraction directory (zip-slip)")

    member_path = extract_root.joinpath(*pure_path.parts).resolve(strict=False)
    try:
        contained = member_path.is_relative_to(extract_root)
    except (OSError, ValueError):
        contained = False
    try:
        common_root = Path(os.path.commonpath((extract_root, member_path))) == extract_root
    except ValueError:
        common_root = False
    if not contained or not common_root:
        raise ValueError("Zip entry would escape extraction directory (zip-slip)")
    return member_path


def _validate_zip_member_type(info: zipfile.ZipInfo) -> None:
    """Reject links, encrypted entries, and non-file/non-directory members."""
    original_name = getattr(info, "orig_filename", info.filename)
    if original_name != info.filename or "\x00" in original_name:
        raise ValueError("Zip entry has an unsafe or ambiguous path (zip-slip)")
    if info.flag_bits & 0x1:
        raise ValueError("Encrypted zip entries are not supported")
    unix_mode = info.external_attr >> 16
    file_type = S_IFMT(unix_mode)
    if S_ISLNK(unix_mode):
        raise ValueError("Zip links are not supported")
    if file_type and not (S_ISREG(unix_mode) or S_ISDIR(unix_mode)):
        raise ValueError("Zip special-file entries are not supported")
    if info.is_dir() and file_type and not S_ISDIR(unix_mode):
        raise ValueError("Zip entry type is inconsistent")
    if not info.is_dir() and S_ISDIR(unix_mode):
        raise ValueError("Zip entry type is inconsistent")
    if info.is_dir() and (info.file_size != 0 or info.compress_size != 0):
        raise ValueError("Zip directory entry contains file data")


class InputHandler:
    """
    Handles input resolution for different source types.

    Normalizes all inputs to a local directory path for scanning.
    """

    def __init__(self, transitive_budget: object | None = None) -> None:
        self._temp_dir: Path | None = None
        self._transitive_budget = transitive_budget

    def resolve(self, input_path: str) -> tuple[Path, str]:
        """
        Resolve input to a scannable directory.

        Args:
            input_path: Path or URL to resolve

        Returns:
            Tuple of (resolved_path, source_type)
            source_type is one of: "git", "url", "zip", "file", "directory"

        Raises:
            ValueError: If input type cannot be determined, or if an
                ingest path exceeds ``INGEST_MAX_BYTES`` /
                ``INGEST_MAX_ZIP_MEMBERS`` (``IngestLimitExceededError``).
            FileNotFoundError: If local path doesn't exist.
        """
        input_path = input_path.strip()

        if self._is_git_url(input_path):
            return self._clone_git(input_path), "git"
        if self._is_file_url(input_path):
            return self._download_file(input_path), "url"
        normalized_local_path = validate_local_input_path(Path(input_path))
        if input_path.endswith(".zip"):
            return self._extract_zip(normalized_local_path), "zip"
        if input_path.endswith(".md"):
            return self._wrap_single_file(normalized_local_path), "file"
        if normalized_local_path.is_dir():
            return normalized_local_path, "directory"
        if normalized_local_path.is_file():
            return self._wrap_single_file(normalized_local_path), "file"
        raise ValueError(
            f"Cannot determine input type for: {input_path}\n"
            "Supported formats: Git URL, file URL, .zip file, .md file, or directory"
        )

    def cleanup(self) -> None:
        """Clean up temporary files created during resolution."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    def temp_dir_for_cleanup(self) -> Path | None:
        """Return the temp directory path if one was created (for caller to clean up after graph)."""
        return self._temp_dir

    def _get_temp_dir(self) -> Path:
        """Get or create a temporary directory for this session."""
        if not self._temp_dir:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="skillspector_"))
        return self._temp_dir

    def _remaining_seconds(self) -> float | None:
        remaining = getattr(self._transitive_budget, "remaining_seconds", None)
        if callable(remaining):
            try:
                return float(remaining())
            except (TypeError, ValueError):
                return None
        return None

    def _remaining_bytes(self) -> int | None:
        remaining = getattr(self._transitive_budget, "remaining_bytes", None)
        if callable(remaining):
            try:
                return int(remaining())
            except (TypeError, ValueError):
                return None
        return None

    def _remaining_artifacts(self) -> int | None:
        remaining = getattr(self._transitive_budget, "remaining_artifacts", None)
        if callable(remaining):
            try:
                return int(remaining())
            except (TypeError, ValueError):
                return None
        return None

    def _record_bytes(self, count: int) -> None:
        record = getattr(self._transitive_budget, "record_bytes", None)
        if callable(record):
            record(max(0, count))

    def _record_artifacts(self, count: int) -> None:
        record = getattr(self._transitive_budget, "record_artifacts", None)
        if callable(record):
            record(max(0, count))

    def _note_truncation(self, reason: str) -> None:
        note = getattr(self._transitive_budget, "note_truncation", None)
        if callable(note):
            note(reason)

    def _truncate(self, code: str, source_type: str) -> NoReturn:
        """Record and raise a typed transitive truncation without source data."""
        error = TransitiveIngestTruncatedError(code, source_type)
        self._note_truncation(error.truncation.message)
        raise error

    def _deadline(self) -> float:
        """Return the local ingest deadline, reduced by any shared deadline."""
        seconds = INGEST_MAX_SECONDS
        remaining = self._remaining_seconds()
        if remaining is not None:
            seconds = min(seconds, max(0.0, remaining))
        return monotonic() + seconds

    def _check_deadline(self, deadline: float, source_type: str) -> None:
        if monotonic() < deadline:
            return
        if self._transitive_budget is not None:
            self._truncate("time_budget_exhausted", source_type)
        raise IngestLimitExceededError(f"{source_type.title()} ingest exceeded its time limit")

    def _bounded_tree_measurement(self, root: Path, deadline: float) -> _TreeMeasurement:
        """Measure a clone using iterative, deterministic, bounded ``scandir``.

        Directory entries are retained only up to ``INGEST_MAX_TREE_ENTRIES``.
        If the cap is crossed the result is rejected before any further tree
        work, rather than walking an attacker-sized tree with ``Path.rglob``.
        """
        entries_seen = 0
        content_bytes = 0
        total_bytes = 0
        stack: list[tuple[Path, bool]] = [(root, False)]
        remaining_bytes = self._remaining_bytes()
        remaining_artifacts = self._remaining_artifacts()

        while stack:
            self._check_deadline(deadline, "git")
            directory, inside_git = stack.pop()
            try:
                with os.scandir(directory) as iterator:
                    directory_entries: list[os.DirEntry[str]] = []
                    for entry in iterator:
                        entries_seen += 1
                        if entries_seen > INGEST_MAX_TREE_ENTRIES:
                            if self._transitive_budget is not None:
                                self._truncate("entry_budget_exhausted", "git")
                            raise IngestLimitExceededError(
                                "Git clone exceeded ingest entry cap: "
                                f"> INGEST_MAX_TREE_ENTRIES ({INGEST_MAX_TREE_ENTRIES})"
                            )
                        if remaining_artifacts is not None and entries_seen > remaining_artifacts:
                            self._truncate("artifact_budget_exhausted", "git")
                        directory_entries.append(entry)
                        self._check_deadline(deadline, "git")
            except OSError as exc:
                raise ValueError("Could not safely inspect cloned repository") from exc

            child_directories: list[tuple[Path, bool]] = []
            for entry in sorted(
                directory_entries, key=lambda item: (item.name.casefold(), item.name)
            ):
                self._check_deadline(deadline, "git")
                try:
                    entry_stat = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise ValueError("Could not safely inspect cloned repository") from exc
                entry_path = Path(entry.path)
                entry_inside_git = inside_git or (directory == root and entry.name == ".git")
                if S_ISLNK(entry_stat.st_mode):
                    continue
                if S_ISDIR(entry_stat.st_mode):
                    child_directories.append((entry_path, entry_inside_git))
                    continue
                if not S_ISREG(entry_stat.st_mode):
                    continue

                size = max(0, entry_stat.st_size)
                total_bytes += size
                if not entry_inside_git:
                    content_bytes += size
                if total_bytes > INGEST_MAX_BYTES:
                    if self._transitive_budget is not None:
                        self._truncate("hard_byte_limit_exceeded", "git")
                    raise IngestLimitExceededError(
                        f"Git clone exceeded ingest cap: {total_bytes} bytes > "
                        f"INGEST_MAX_BYTES ({INGEST_MAX_BYTES})"
                    )
                # Git's object database, indexes, and other .git material are
                # part of the ingest work too.  The shared aggregate budget is
                # therefore checked against the complete on-disk tree, not
                # only the checkout files later visible to analyzers.
                if remaining_bytes is not None and total_bytes > remaining_bytes:
                    self._truncate("byte_budget_exhausted", "git")

            # Reverse push preserves deterministic ascending processing order.
            stack.extend(reversed(child_directories))

        return _TreeMeasurement(
            entries=entries_seen,
            content_bytes=content_bytes,
            total_bytes=total_bytes,
        )

    def _preflight_zip_entries(
        self,
        archive_file: BinaryIO,
        metadata: _ZipDirectoryMetadata,
        deadline: float,
    ) -> None:
        """Count central-directory records without materializing ``ZipInfo`` objects."""
        archive_file.seek(metadata.central_directory_offset)
        consumed = 0
        entries = 0
        while consumed < metadata.central_directory_bytes:
            self._check_deadline(deadline, "zip")
            fixed_header = archive_file.read(46)
            if len(fixed_header) != 46 or fixed_header[:4] != b"PK\x01\x02":
                raise zipfile.BadZipFile("Invalid central-directory record")
            filename_bytes, extra_bytes, comment_bytes = struct.unpack_from("<3H", fixed_header, 28)
            record_bytes = 46 + filename_bytes + extra_bytes + comment_bytes
            if record_bytes > metadata.central_directory_bytes - consumed:
                raise zipfile.BadZipFile("Truncated central-directory record")
            archive_file.seek(record_bytes - 46, os.SEEK_CUR)
            consumed += record_bytes
            entries += 1
            if entries > INGEST_MAX_ZIP_MEMBERS:
                if self._transitive_budget is not None:
                    self._truncate("entry_budget_exhausted", "zip")
                raise IngestLimitExceededError(
                    "Zip exceeded ingest cap while preflighting members: "
                    f"> INGEST_MAX_ZIP_MEMBERS ({INGEST_MAX_ZIP_MEMBERS})"
                )
        if consumed != metadata.central_directory_bytes or entries != metadata.entries:
            raise zipfile.BadZipFile("Central-directory count is inconsistent")

    def _reserve_zip_target(self, materialized_targets: set[str], target: Path) -> str:
        """Reserve one extracted filesystem object under the global count cap."""
        key = os.path.normcase(os.fspath(target)).casefold()
        if key in materialized_targets:
            return key
        if len(materialized_targets) >= INGEST_MAX_ZIP_MEMBERS:
            if self._transitive_budget is not None:
                self._truncate("entry_budget_exhausted", "zip")
            raise IngestLimitExceededError(
                "Zip exceeded extracted-entry cap: "
                f"> INGEST_MAX_ZIP_MEMBERS ({INGEST_MAX_ZIP_MEMBERS})"
            )
        remaining_artifacts = self._remaining_artifacts()
        if remaining_artifacts is not None and remaining_artifacts <= 0:
            self._truncate("artifact_budget_exhausted", "zip")
        materialized_targets.add(key)
        self._record_artifacts(1)
        return key

    def _ensure_zip_directories(
        self,
        extract_root: Path,
        directory: Path,
        materialized_targets: set[str],
        deadline: float,
    ) -> None:
        """Create and count implicit member directories one component at a time."""
        current = extract_root
        for part in directory.relative_to(extract_root).parts:
            self._check_deadline(deadline, "zip")
            current /= part
            key = os.path.normcase(os.fspath(current)).casefold()
            if key in materialized_targets:
                if not current.is_dir() or current.is_symlink():
                    raise ValueError("Zip directory entry conflicts with a file")
                continue
            self._reserve_zip_target(materialized_targets, current)
            current.mkdir()

    def _is_git_url(self, path: str) -> bool:
        """Check if path is a Git repository URL."""
        if not path.startswith(("https://", "git@")):
            return False
        parsed = urlparse(path)
        host = parsed.hostname or ""
        if any(allowed in host for allowed in ALLOWED_GIT_HOSTS):
            lower_path = parsed.path.lower()
            if (
                "/raw/" in lower_path
                or "/blob/" in lower_path
                or "/archive/" in lower_path
                or lower_path.endswith(_DIRECT_FILE_URL_SUFFIXES)
            ):
                return False
            return True
        if path.endswith(".git"):
            return True
        return False

    def _is_file_url(self, path: str) -> bool:
        """Check if path is a direct file URL."""
        if not path.startswith("https://"):
            return False
        return not self._is_git_url(path)

    def _extract_scp_host(self, url: str) -> str | None:
        """Return the host from an scp-style Git URL, or None if not scp form."""
        if "://" in url:
            return None
        m = re.match(r"^[^@/]+@([^:/]+):.+$", url)
        return m.group(1) if m else None

    def _validate_url_host(self, url: str, allowed_hosts: frozenset[str]) -> str:
        """Validate URL host against allowlist and SSRF protections.

        Returns the hostname on success, raises ValueError on blocked URLs.
        """
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if not host:
            host = self._extract_scp_host(url) or ""
        if not host:
            raise ValueError(f"URL has no valid hostname: {url}")
        if not any(host == allowed or host.endswith("." + allowed) for allowed in allowed_hosts):
            raise ValueError(
                f"Host '{host}' is not in the allowed hosts list. Allowed: {sorted(allowed_hosts)}"
            )
        if _is_private_ip(host):
            raise ValueError(
                f"URL resolves to a private/internal IP address: {url}. "
                "This is blocked to prevent SSRF attacks."
            )
        return host

    def _clone_git(self, url: str) -> Path:
        """Clone a Git repository to a temporary directory, bounded by ``INGEST_MAX_BYTES``."""
        remaining_seconds = self._remaining_seconds()
        remaining_bytes = self._remaining_bytes()
        remaining_artifacts = self._remaining_artifacts()
        if remaining_seconds is not None and remaining_seconds <= 0:
            self._truncate("time_budget_exhausted", "git")
        if remaining_bytes is not None and remaining_bytes <= 0:
            self._truncate("byte_budget_exhausted", "git")
        if remaining_artifacts is not None and remaining_artifacts <= 0:
            self._truncate("artifact_budget_exhausted", "git")
        self._validate_url_host(url, ALLOWED_GIT_HOSTS)
        deadline = self._deadline()
        self._check_deadline(deadline, "git")
        temp_dir = self._get_temp_dir()
        clone_dir = temp_dir / "repo"
        clone_command = [
            "git",
            "-c",
            "core.symlinks=false",
            "clone",
            "--depth",
            "1",
            url,
            str(clone_dir),
        ]
        if remaining_bytes is not None:
            clone_command.insert(6, f"--filter=blob:limit={remaining_bytes}")
        process: subprocess.Popen[bytes] | None = None
        final_measurement: _TreeMeasurement | None = None
        try:
            process = subprocess.Popen(
                clone_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
            )
            while True:
                self._check_deadline(deadline, "git")
                return_code = process.poll()
                if clone_dir.exists():
                    # The clone filter is only a server hint and may be ignored.
                    # Measure the materializing tree while Git is still running
                    # so an oversized pack/worktree is terminated, not merely
                    # rejected after the subprocess has filled the disk.
                    final_measurement = self._bounded_tree_measurement(clone_dir, deadline)
                if return_code is not None:
                    if return_code != 0:
                        raise ValueError("Failed to clone repository")
                    break
                try:
                    process.wait(timeout=min(0.05, max(0.001, deadline - monotonic())))
                except subprocess.TimeoutExpired:
                    continue
            if not clone_dir.is_dir():
                raise ValueError("Git clone did not produce a repository directory")
            if final_measurement is None:
                final_measurement = self._bounded_tree_measurement(clone_dir, deadline)
            self._record_bytes(final_measurement.total_bytes)
            self._record_artifacts(final_measurement.entries)
        except (IngestLimitExceededError, TransitiveIngestTruncatedError, ValueError):
            self._terminate_git_process(process)
            shutil.rmtree(clone_dir, ignore_errors=True)
            raise
        except FileNotFoundError:
            self._terminate_git_process(process)
            shutil.rmtree(clone_dir, ignore_errors=True)
            logger.warning("Git not found when cloning %s", url)
            raise ValueError(
                "Git is not installed. Please install git to scan repositories."
            ) from None
        except OSError as exc:
            self._terminate_git_process(process)
            shutil.rmtree(clone_dir, ignore_errors=True)
            raise ValueError("Failed to clone repository") from exc
        return clone_dir

    @staticmethod
    def _terminate_git_process(process: subprocess.Popen[bytes] | None) -> None:
        """Best-effort bounded shutdown for a clone rejected during materialization."""
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                logger.warning("Git clone process did not terminate promptly")

    def _download_file(self, url: str) -> Path:
        """Download a file from URL to a temporary directory.

        Streams the body to disk in chunks while running a byte counter.
        The cap check fires before each chunk is written, so a breach
        aborts immediately without accumulating the body in memory.  A
        partial file produced by a mid-stream breach is removed before
        the exception propagates.
        """
        if self._transitive_budget is not None:
            return self._download_transitive_file(url)
        self._validate_url_host(url, ALLOWED_DOWNLOAD_HOSTS)
        temp_dir = self._get_temp_dir()
        parsed = urlparse(url)
        filename = Path(parsed.path).name or "SKILL.md"
        # Write to a stable target path inside the temp dir so we can
        # rename / move it after the download succeeds without ever
        # holding the body in memory.  Use a sentinel name for the
        # download itself; we rename / replace at the end.
        download_path = temp_dir / "_download.partial"
        content_type = ""
        deadline = self._deadline()
        try:
            self._validate_url_host(url, ALLOWED_DOWNLOAD_HOSTS)
            with httpx.Client(follow_redirects=False, timeout=30) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "")
                    # Cheap up-front check: trust Content-Length when the
                    # server provides it, so we abort before reading any
                    # body bytes. Streaming check below covers the case
                    # where the header is missing or wrong.
                    declared = response.headers.get("content-length")
                    if declared is not None:
                        try:
                            declared_bytes = int(declared)
                        except ValueError:
                            declared_bytes = None
                        if declared_bytes is not None and declared_bytes > INGEST_MAX_BYTES:
                            raise IngestLimitExceededError(
                                f"Download exceeded ingest cap: Content-Length {declared} bytes > "
                                f"INGEST_MAX_BYTES ({INGEST_MAX_BYTES})"
                            )

                    received = 0
                    with download_path.open("wb") as out:
                        for chunk in response.iter_bytes(chunk_size=_COPY_CHUNK_BYTES):
                            self._check_deadline(deadline, "download")
                            received += len(chunk)
                            if received > INGEST_MAX_BYTES:
                                raise IngestLimitExceededError(
                                    f"Download exceeded ingest cap: streamed {received} bytes > "
                                    f"INGEST_MAX_BYTES ({INGEST_MAX_BYTES})"
                                )
                            out.write(chunk)
        except httpx.HTTPError as e:
            # Best-effort cleanup of any partial download.
            download_path.unlink(missing_ok=True)
            logger.warning("Download failed for %s: %s", url, e)
            raise ValueError(f"Failed to download file: {e}") from e
        except IngestLimitExceededError:
            # Don't leave the partial bomb on disk.
            download_path.unlink(missing_ok=True)
            raise

        is_zip = filename.endswith(".zip") or content_type.startswith("application/zip")
        if is_zip:
            zip_path = temp_dir / "download.zip"
            download_path.replace(zip_path)
            return self._extract_zip(zip_path)
        file_path = temp_dir / filename
        download_path.replace(file_path)
        return temp_dir

    def _download_transitive_file(self, url: str) -> Path:
        remaining_artifacts = self._remaining_artifacts()
        if remaining_artifacts is not None and remaining_artifacts <= 0:
            self._truncate("artifact_budget_exhausted", "download")
        try:
            headers, final_url, content = self._download_with_redirect_validation(url)
            filename = Path(urlparse(final_url).path).name or "SKILL.md"
        except httpx.TimeoutException:
            self._truncate("time_budget_exhausted", "download")
        except httpx.HTTPError as exc:
            raise ValueError(f"Failed to download file: {exc}") from exc
        temp_dir = self._get_temp_dir()
        self._record_artifacts(1)
        if filename.endswith(".zip") or headers.get("content-type", "").startswith(
            "application/zip"
        ):
            zip_path = temp_dir / "download.zip"
            zip_path.write_bytes(content)
            return self._extract_zip(zip_path)
        (temp_dir / filename).write_bytes(content)
        return temp_dir

    def _download_with_redirect_validation(self, url: str) -> tuple[dict[str, str], str, bytes]:
        current_url = url
        deadline = self._deadline()
        for _ in range(5):
            remaining_seconds = self._remaining_seconds()
            if remaining_seconds is not None and remaining_seconds <= 0:
                self._truncate("time_budget_exhausted", "download")
            remaining_bytes = self._remaining_bytes()
            if remaining_bytes is not None and remaining_bytes <= 0:
                self._truncate("byte_budget_exhausted", "download")
            self._check_deadline(deadline, "download")
            self._validate_url_host(current_url, ALLOWED_DOWNLOAD_HOSTS)
            request_timeout = min(30.0, max(0.001, deadline - monotonic()))
            with httpx.Client(follow_redirects=False, timeout=request_timeout) as client:
                with client.stream("GET", current_url) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError(f"Redirect response missing location: {current_url}")
                        current_url = urljoin(current_url, location)
                        continue
                    response.raise_for_status()
                    declared = response.headers.get("content-length")
                    if declared is not None:
                        try:
                            declared_bytes = int(declared)
                        except ValueError:
                            declared_bytes = None
                        if declared_bytes is not None and declared_bytes > INGEST_MAX_BYTES:
                            self._truncate("hard_byte_limit_exceeded", "download")
                        if (
                            declared_bytes is not None
                            and remaining_bytes is not None
                            and declared_bytes > remaining_bytes
                        ):
                            self._truncate("byte_budget_exhausted", "download")
                    content = bytearray()
                    for chunk in response.iter_bytes(chunk_size=_COPY_CHUNK_BYTES):
                        self._check_deadline(deadline, "download")
                        content.extend(chunk)
                        if len(content) > INGEST_MAX_BYTES:
                            self._truncate("hard_byte_limit_exceeded", "download")
                        if remaining_bytes is not None and len(content) > remaining_bytes:
                            self._truncate("byte_budget_exhausted", "download")
                        self._record_bytes(len(chunk))
                    return dict(response.headers), current_url, bytes(content)
        raise ValueError(f"Too many redirects while downloading: {url}")

    def _extract_zip(self, zip_path: Path) -> Path:
        """Extract a zip file, bounded by ``INGEST_MAX_BYTES`` and ``INGEST_MAX_ZIP_MEMBERS``.

        EOCD/ZIP64 fields are checked before ``ZipFile`` may materialize the
        central directory.  Extraction is then manual and streaming so count,
        byte, type, containment, and deadline checks remain enforceable while
        bytes are written.
        """
        remaining_bytes = self._remaining_bytes()
        deadline = self._deadline()
        with _open_regular_file_no_follow(zip_path) as archive_file:
            try:
                self._check_deadline(deadline, "zip")
                directory_metadata = _read_zip_directory_metadata(archive_file)
                if directory_metadata.entries > INGEST_MAX_ZIP_MEMBERS:
                    if self._transitive_budget is not None:
                        self._truncate("entry_budget_exhausted", "zip")
                    raise IngestLimitExceededError(
                        f"Zip exceeded ingest cap: {directory_metadata.entries} members > "
                        f"INGEST_MAX_ZIP_MEMBERS ({INGEST_MAX_ZIP_MEMBERS})"
                    )
                remaining_artifacts = self._remaining_artifacts()
                if (
                    remaining_artifacts is not None
                    and directory_metadata.entries > remaining_artifacts
                ):
                    self._truncate("artifact_budget_exhausted", "zip")
                if (
                    directory_metadata.central_directory_bytes
                    > INGEST_MAX_ZIP_CENTRAL_DIRECTORY_BYTES
                ):
                    if self._transitive_budget is not None:
                        self._truncate("metadata_budget_exhausted", "zip")
                    raise IngestLimitExceededError(
                        "Zip exceeded central-directory metadata cap: "
                        f"{directory_metadata.central_directory_bytes} bytes > "
                        "INGEST_MAX_ZIP_CENTRAL_DIRECTORY_BYTES "
                        f"({INGEST_MAX_ZIP_CENTRAL_DIRECTORY_BYTES})"
                    )
                self._preflight_zip_entries(archive_file, directory_metadata, deadline)

                archive_file.seek(0)
                with zipfile.ZipFile(archive_file, "r") as zf:
                    infos = zf.infolist()
                    self._check_deadline(deadline, "zip")
                    if len(infos) != directory_metadata.entries:
                        raise ValueError("Zip central-directory entry count is inconsistent")
                    total_uncompressed = sum(info.file_size for info in infos)
                    self._check_deadline(deadline, "zip")
                    if remaining_bytes is not None and total_uncompressed > remaining_bytes:
                        self._truncate("byte_budget_exhausted", "zip")
                    if total_uncompressed > INGEST_MAX_BYTES:
                        if self._transitive_budget is not None:
                            self._truncate("hard_byte_limit_exceeded", "zip")
                        raise IngestLimitExceededError(
                            f"Zip exceeded ingest cap: uncompressed "
                            f"{total_uncompressed} bytes > INGEST_MAX_BYTES "
                            f"({INGEST_MAX_BYTES})"
                        )
                    temp_dir = self._get_temp_dir()
                    extract_dir = temp_dir / "extracted"
                    if extract_dir.exists():
                        shutil.rmtree(extract_dir, ignore_errors=True)
                    extract_dir.mkdir()
                    extract_root = extract_dir.resolve(strict=True)
                    seen_member_targets: set[str] = set()
                    materialized_targets: set[str] = set()
                    extracted_bytes = 0
                    try:
                        for info in infos:
                            self._check_deadline(deadline, "zip")
                            _validate_zip_member_type(info)
                            member_path = _safe_zip_target(extract_root, info.filename)
                            target_key = os.path.normcase(os.fspath(member_path)).casefold()
                            if target_key in seen_member_targets:
                                raise ValueError("Zip contains duplicate extraction paths")
                            seen_member_targets.add(target_key)

                            if info.is_dir():
                                self._ensure_zip_directories(
                                    extract_root,
                                    member_path,
                                    materialized_targets,
                                    deadline,
                                )
                                continue
                            self._ensure_zip_directories(
                                extract_root,
                                member_path.parent,
                                materialized_targets,
                                deadline,
                            )
                            if target_key in materialized_targets:
                                raise ValueError("Zip file entry conflicts with a directory")
                            self._reserve_zip_target(materialized_targets, member_path)
                            member_bytes = 0
                            with zf.open(info, "r") as source, member_path.open("xb") as target:
                                while True:
                                    self._check_deadline(deadline, "zip")
                                    chunk = source.read(_COPY_CHUNK_BYTES)
                                    if not chunk:
                                        break
                                    member_bytes += len(chunk)
                                    extracted_bytes += len(chunk)
                                    if extracted_bytes > INGEST_MAX_BYTES:
                                        if self._transitive_budget is not None:
                                            self._truncate("hard_byte_limit_exceeded", "zip")
                                        raise IngestLimitExceededError(
                                            "Zip exceeded ingest cap during extraction"
                                        )
                                    if (
                                        remaining_bytes is not None
                                        and extracted_bytes > remaining_bytes
                                    ):
                                        self._truncate("byte_budget_exhausted", "zip")
                                    if member_bytes > info.file_size:
                                        raise ValueError(
                                            "Zip member expanded beyond its declared size"
                                        )
                                    target.write(chunk)
                                    self._record_bytes(len(chunk))
                            if member_bytes != info.file_size:
                                raise ValueError("Zip member size did not match its declaration")
                    except BaseException:
                        shutil.rmtree(extract_dir, ignore_errors=True)
                        raise
            except zipfile.BadZipFile:
                logger.warning("Invalid zip or extract failed: %s", zip_path)
                raise ValueError(f"Invalid zip file: {zip_path}") from None
        contents: list[Path] = []
        with os.scandir(extract_dir) as iterator:
            for entry in iterator:
                contents.append(Path(entry.path))
                if len(contents) > 1:
                    break
        if len(contents) == 1 and contents[0].is_dir():
            return contents[0]
        return extract_dir

    def _wrap_single_file(self, file_path: Path) -> Path:
        """Wrap a single file in a temporary directory for consistent handling."""
        remaining_bytes = self._remaining_bytes()
        deadline = self._deadline()
        self._check_deadline(deadline, "file")
        with _open_regular_file_no_follow(file_path) as source:
            source_size = max(0, os.fstat(source.fileno()).st_size)
            if source_size > INGEST_MAX_BYTES:
                if self._transitive_budget is not None:
                    self._truncate("hard_byte_limit_exceeded", "file")
                raise IngestLimitExceededError(
                    f"File exceeded ingest cap: {source_size} bytes > "
                    f"INGEST_MAX_BYTES ({INGEST_MAX_BYTES})"
                )
            if remaining_bytes is not None and source_size > remaining_bytes:
                self._truncate("byte_budget_exhausted", "file")
            temp_dir = self._get_temp_dir()
            dest = temp_dir / file_path.name
            copied = 0
            try:
                with dest.open("xb") as target:
                    while True:
                        self._check_deadline(deadline, "file")
                        chunk = source.read(_COPY_CHUNK_BYTES)
                        if not chunk:
                            break
                        copied += len(chunk)
                        if copied > INGEST_MAX_BYTES:
                            if self._transitive_budget is not None:
                                self._truncate("hard_byte_limit_exceeded", "file")
                            raise IngestLimitExceededError(
                                "File exceeded ingest cap while being copied"
                            )
                        if remaining_bytes is not None and copied > remaining_bytes:
                            self._truncate("byte_budget_exhausted", "file")
                        target.write(chunk)
            except BaseException:
                dest.unlink(missing_ok=True)
                raise
        return temp_dir
