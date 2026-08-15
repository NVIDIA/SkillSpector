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
import subprocess
import tempfile
import zipfile
from errno import ELOOP, ENOENT, ENOTDIR
from pathlib import Path
from stat import S_ISLNK, S_ISREG
from typing import BinaryIO, cast
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


class IngestLimitExceededError(ValueError):
    """Raised when an ingest path exceeds an ``INGEST_MAX_*`` budget.

    Subclass of ``ValueError`` so existing callers that catch
    ``ValueError`` from ``InputHandler.resolve()`` continue to work.
    """


class _TraversalBudgetError(ValueError):
    """Raised when transitive input work should truncate rather than fail."""


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
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
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

    def _note_truncation(self, reason: str) -> None:
        note = getattr(self._transitive_budget, "note_truncation", None)
        if callable(note):
            note(reason)

    def _empty_result_dir(self, reason: str, name: str) -> Path:
        self._note_truncation(reason)
        temp_dir = self._get_temp_dir()
        target = temp_dir / name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _measure_tree_bytes(self, root: Path) -> int:
        total = 0
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink() or ".git" in path.parts:
                continue
            try:
                total += path.stat().st_size
            except OSError:
                logger.debug("Could not stat cloned file: %s", path)
        return total

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
        self._validate_url_host(url, ALLOWED_GIT_HOSTS)
        temp_dir = self._get_temp_dir()
        clone_dir = temp_dir / "repo"
        remaining_seconds = self._remaining_seconds()
        remaining_bytes = self._remaining_bytes()
        if remaining_seconds is not None and remaining_seconds <= 0:
            return self._empty_result_dir(
                "Transitive time budget exhausted before git clone", "repo"
            )
        if remaining_bytes is not None and remaining_bytes <= 0:
            return self._empty_result_dir(
                "Transitive byte budget exhausted before git clone", "repo"
            )
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
        try:
            subprocess.run(
                clone_command,
                check=True,
                capture_output=True,
                timeout=remaining_seconds if remaining_seconds is not None else 60,
                shell=False,
            )
        except subprocess.CalledProcessError as e:
            logger.warning("Git clone failed for %s: %s", url, e)
            raise ValueError(f"Failed to clone repository: {e.stderr.decode()}") from e
        except subprocess.TimeoutExpired:
            logger.warning("Git clone timed out for %s", url)
            raise ValueError(
                f"Git clone timed out after {remaining_seconds or 60:.0f} seconds"
            ) from None
        except FileNotFoundError:
            logger.warning("Git not found when cloning %s", url)
            raise ValueError(
                "Git is not installed. Please install git to scan repositories."
            ) from None

        tree_bytes = self._measure_tree_bytes(clone_dir)
        if remaining_bytes is not None and tree_bytes > remaining_bytes:
            return self._empty_result_dir(
                "Transitive byte budget exceeded by cloned repository", "repo"
            )

        # Post-clone size check: a successful --depth 1 clone may still
        # land an arbitrarily large tree on disk before we can measure
        # it, so this is a fail-closed cap rather than a hard prefilter.
        # Residual window: within the 60s clone timeout an attacker can
        # transiently consume up to whatever the network + disk let
        # through before this check runs; bounded by the timeout, but
        # not zero.  ``.git/`` objects are counted toward the cap, so a
        # legitimate repo with a working tree just under ``INGEST_MAX_BYTES``
        # can still be rejected once packfiles are added.
        total = _directory_size_bytes(clone_dir)
        if total > INGEST_MAX_BYTES:
            shutil.rmtree(clone_dir, ignore_errors=True)
            logger.warning(
                "Git clone of %s exceeded ingest cap: %d > %d bytes",
                url,
                total,
                INGEST_MAX_BYTES,
            )
            raise IngestLimitExceededError(
                f"Git clone exceeded ingest cap: {total} bytes > "
                f"INGEST_MAX_BYTES ({INGEST_MAX_BYTES})"
            )
        return clone_dir

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
                        for chunk in response.iter_bytes():
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
        temp_dir = self._get_temp_dir()
        try:
            headers, final_url, content = self._download_with_redirect_validation(url)
            filename = Path(urlparse(final_url).path).name or "SKILL.md"
        except _TraversalBudgetError as exc:
            return self._empty_result_dir(str(exc), "download")
        except httpx.HTTPError as exc:
            raise ValueError(f"Failed to download file: {exc}") from exc
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
        for _ in range(5):
            remaining_seconds = self._remaining_seconds()
            if remaining_seconds is not None and remaining_seconds <= 0:
                raise _TraversalBudgetError("Transitive time budget exhausted before download")
            self._validate_url_host(current_url, ALLOWED_DOWNLOAD_HOSTS)
            with httpx.Client(follow_redirects=False, timeout=remaining_seconds or 30) as client:
                with client.stream("GET", current_url) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError(f"Redirect response missing location: {current_url}")
                        current_url = urljoin(current_url, location)
                        continue
                    response.raise_for_status()
                    remaining_bytes = self._remaining_bytes()
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
                        if (
                            declared_bytes is not None
                            and remaining_bytes is not None
                            and declared_bytes > remaining_bytes
                        ):
                            raise _TraversalBudgetError(
                                "Transitive byte budget exceeded by downloaded file"
                            )
                    content = bytearray()
                    for chunk in response.iter_bytes():
                        if self._remaining_seconds() is not None and self._remaining_seconds() <= 0:
                            raise _TraversalBudgetError(
                                "Transitive time budget exhausted during download"
                            )
                        content.extend(chunk)
                        if len(content) > INGEST_MAX_BYTES:
                            raise IngestLimitExceededError(
                                "Download exceeded ingest cap while following redirect"
                            )
                        if remaining_bytes is not None and len(content) > remaining_bytes:
                            raise _TraversalBudgetError(
                                "Transitive byte budget exceeded by downloaded file"
                            )
                    return dict(response.headers), current_url, bytes(content)
        raise ValueError(f"Too many redirects while downloading: {url}")

    def _extract_zip(self, zip_path: Path) -> Path:
        """Extract a zip file, bounded by ``INGEST_MAX_BYTES`` and ``INGEST_MAX_ZIP_MEMBERS``.

        Sums ``ZipInfo.file_size`` (uncompressed size) across all members
        before extracting and refuses to extract if either the total or
        the member count exceeds the cap.  This rejects classic zip
        bombs (small archive, huge declared uncompressed size) without
        materialising any of the bomb on disk.  A zip-slip check on each
        member name is applied before extraction to reject entries whose
        resolved path escapes the extraction directory.
        """
        remaining_bytes = self._remaining_bytes()
        if remaining_bytes is not None and remaining_bytes <= 0:
            return self._empty_result_dir(
                "Transitive byte budget exhausted before zip extraction", "extracted"
            )
        with _open_regular_file_no_follow(zip_path) as archive_file:
            temp_dir = self._get_temp_dir()
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir(exist_ok=True)
            try:
                with zipfile.ZipFile(archive_file, "r") as zf:
                    infos = zf.infolist()
                    if len(infos) > INGEST_MAX_ZIP_MEMBERS:
                        raise IngestLimitExceededError(
                            f"Zip exceeded ingest cap: {len(infos)} members > "
                            f"INGEST_MAX_ZIP_MEMBERS ({INGEST_MAX_ZIP_MEMBERS})"
                        )
                    total_uncompressed = sum(info.file_size for info in infos)
                    if remaining_bytes is not None and total_uncompressed > remaining_bytes:
                        return self._empty_result_dir(
                            "Transitive byte budget exceeded by zip extraction", "extracted"
                        )
                    if total_uncompressed > INGEST_MAX_BYTES:
                        raise IngestLimitExceededError(
                            f"Zip exceeded ingest cap: uncompressed "
                            f"{total_uncompressed} bytes > INGEST_MAX_BYTES "
                            f"({INGEST_MAX_BYTES})"
                        )
                    extract_root = extract_dir.resolve()
                    for member in zf.namelist():
                        member_path = (extract_dir / member).resolve()
                        if not str(member_path).startswith(str(extract_root)):
                            raise ValueError(
                                f"Zip entry '{member}' would escape extraction directory (zip-slip). "
                                "Archive is potentially malicious."
                            )
                    zf.extractall(extract_dir)
            except zipfile.BadZipFile:
                logger.warning("Invalid zip or extract failed: %s", zip_path)
                raise ValueError(f"Invalid zip file: {zip_path}") from None
        contents = list(extract_dir.iterdir())
        if len(contents) == 1 and contents[0].is_dir():
            return contents[0]
        return extract_dir

    def _wrap_single_file(self, file_path: Path) -> Path:
        """Wrap a single file in a temporary directory for consistent handling."""
        remaining_bytes = self._remaining_bytes()
        if remaining_bytes is not None:
            try:
                if file_path.stat().st_size > remaining_bytes:
                    return self._empty_result_dir(
                        "Transitive byte budget exceeded by single-file input", "file"
                    )
            except OSError:
                pass
        with _open_regular_file_no_follow(file_path) as source:
            temp_dir = self._get_temp_dir()
            dest = temp_dir / file_path.name
            with dest.open("wb") as target:
                shutil.copyfileobj(source, target)
        return temp_dir


def _directory_size_bytes(path: Path) -> int:
    """Return the total size of all regular files under *path*, in bytes.

    Symlinks are explicitly skipped via ``Path.is_symlink()`` — note that
    ``Path.is_file()`` follows symlinks and would otherwise return
    ``True`` for a symlink pointing at a regular file, so the
    ``not p.is_symlink()`` guard is load-bearing and must not be removed.
    This is what prevents a malicious symlink to ``/dev/zero`` (or any
    large file outside the walked tree) from inflating the count.
    """
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            try:
                total += p.stat().st_size
            except OSError:
                # File disappeared mid-walk (race with concurrent fs ops).
                # Skip rather than fail the whole ingest.
                continue
    return total
