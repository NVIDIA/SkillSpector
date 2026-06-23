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
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import httpx

from skillspector.logging_config import get_logger

logger = get_logger(__name__)

# Hard ceiling on what any single ingest path can pull into the temp dir.
# Sized above the per-file analysis cap (``MAX_FILE_BYTES`` = 1 MB) so a
# legitimate multi-file skill is not blocked at ingest, but tight enough
# to bound memory / disk DoS from a malicious source.
INGEST_MAX_BYTES = 100 * 1024 * 1024  # 100 MiB

# Hard ceiling on the number of members in a zip we are willing to
# extract.  Catches the "many tiny files" zip-bomb variant where each
# entry is small but the entry count itself exhausts the filesystem.
INGEST_MAX_ZIP_MEMBERS = 10_000

# Chunk size for streaming HTTP downloads.  Small enough that the
# byte-count breach check fires promptly; large enough to keep syscall
# overhead reasonable on legitimate inputs.
_DOWNLOAD_CHUNK_BYTES = 64 * 1024


class IngestLimitExceededError(ValueError):
    """Raised when an ingest path exceeds an ``INGEST_MAX_*`` budget.

    Subclass of ``ValueError`` so existing callers that catch
    ``ValueError`` from ``InputHandler.resolve()`` continue to work.
    """


class InputHandler:
    """
    Handles input resolution for different source types.

    Normalizes all inputs to a local directory path for scanning.
    """

    def __init__(self) -> None:
        self._temp_dir: Path | None = None

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
        if input_path.endswith(".zip"):
            return self._extract_zip(Path(input_path)), "zip"
        if input_path.endswith(".md"):
            return self._wrap_single_file(Path(input_path)), "file"
        if Path(input_path).is_dir():
            return Path(input_path).resolve(), "directory"
        if Path(input_path).is_file():
            return self._wrap_single_file(Path(input_path)), "file"
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

    def _is_git_url(self, path: str) -> bool:
        """Check if path is a Git repository URL."""
        if not path.startswith(("http://", "https://", "git@")):
            return False
        parsed = urlparse(path)
        git_hosts = ["github.com", "gitlab.com", "bitbucket.org"]
        if any(host in parsed.netloc for host in git_hosts):
            if "/raw/" in path or "/blob/" in path or path.endswith((".md", ".py", ".sh")):
                return False
            return True
        if path.endswith(".git"):
            return True
        return False

    def _is_file_url(self, path: str) -> bool:
        """Check if path is a direct file URL."""
        if not path.startswith(("http://", "https://")):
            return False
        return not self._is_git_url(path)

    def _clone_git(self, url: str) -> Path:
        """Clone a Git repository to a temporary directory, bounded by ``INGEST_MAX_BYTES``."""
        temp_dir = self._get_temp_dir()
        clone_dir = temp_dir / "repo"
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(clone_dir)],
                check=True,
                capture_output=True,
                timeout=60,
                shell=False,
            )
        except subprocess.CalledProcessError as e:
            logger.warning("Git clone failed for %s: %s", url, e)
            raise ValueError(f"Failed to clone repository: {e.stderr.decode()}") from e
        except subprocess.TimeoutExpired:
            logger.warning("Git clone timed out for %s", url)
            raise ValueError("Git clone timed out after 60 seconds") from None
        except FileNotFoundError:
            logger.warning("Git not found when cloning %s", url)
            raise ValueError(
                "Git is not installed. Please install git to scan repositories."
            ) from None

        # Post-clone size check: a successful --depth 1 clone may still
        # land an arbitrarily large tree on disk.  Reject (and clean up)
        # if it exceeds the ingest budget.
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
            with httpx.Client(follow_redirects=True, timeout=30) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "")
                    # Cheap up-front check: trust Content-Length when the
                    # server provides it, so we abort before reading any
                    # body bytes.  Streaming check below covers the case
                    # where the header is missing or wrong.
                    declared = response.headers.get("content-length")
                    if declared is not None:
                        try:
                            declared_bytes = int(declared)
                        except ValueError:
                            # Malformed header — fall through to the
                            # streamed byte counter, which is authoritative.
                            declared_bytes = None
                        if declared_bytes is not None and declared_bytes > INGEST_MAX_BYTES:
                            raise IngestLimitExceededError(
                                f"Download exceeded ingest cap: "
                                f"Content-Length {declared} bytes > "
                                f"INGEST_MAX_BYTES ({INGEST_MAX_BYTES})"
                            )

                    received = 0
                    with download_path.open("wb") as out:
                        for chunk in response.iter_bytes(_DOWNLOAD_CHUNK_BYTES):
                            received += len(chunk)
                            if received > INGEST_MAX_BYTES:
                                raise IngestLimitExceededError(
                                    f"Download exceeded ingest cap: streamed "
                                    f"{received} bytes > INGEST_MAX_BYTES "
                                    f"({INGEST_MAX_BYTES})"
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

    def _extract_zip(self, zip_path: Path) -> Path:
        """Extract a zip file, bounded by ``INGEST_MAX_BYTES`` and ``INGEST_MAX_ZIP_MEMBERS``.

        Sums ``ZipInfo.file_size`` (uncompressed size) across all members
        before extracting and refuses to extract if either the total or
        the member count exceeds the cap.  This rejects classic zip
        bombs (small archive, huge declared uncompressed size) without
        materialising any of the bomb on disk.
        """
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}") from None
        temp_dir = self._get_temp_dir()
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                infos = zf.infolist()
                if len(infos) > INGEST_MAX_ZIP_MEMBERS:
                    raise IngestLimitExceededError(
                        f"Zip exceeded ingest cap: {len(infos)} members > "
                        f"INGEST_MAX_ZIP_MEMBERS ({INGEST_MAX_ZIP_MEMBERS})"
                    )
                total_uncompressed = sum(info.file_size for info in infos)
                if total_uncompressed > INGEST_MAX_BYTES:
                    raise IngestLimitExceededError(
                        f"Zip exceeded ingest cap: uncompressed "
                        f"{total_uncompressed} bytes > INGEST_MAX_BYTES "
                        f"({INGEST_MAX_BYTES})"
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
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}") from None
        temp_dir = self._get_temp_dir()
        dest = temp_dir / file_path.name
        shutil.copy2(file_path, dest)
        return temp_dir


def _directory_size_bytes(path: Path) -> int:
    """Return the total size of all regular files under *path*, in bytes.

    Symlinks are not followed (``Path.is_file()`` returns ``True`` only
    for regular files), so a malicious symlink to ``/dev/zero`` cannot
    cause a runaway count.
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
