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

"""Tests for ingest-layer size bounds in ``InputHandler``.

Covers the three ingest paths the bounded-reads work in PR #19 deferred
to a follow-up (issues #21 / #131): URL download, zip extraction, and
git clone.  Each is bounded by ``INGEST_MAX_BYTES`` (and zip is also
bounded by ``INGEST_MAX_ZIP_MEMBERS``); each must fail closed with a
clear error message rather than letting the per-file analysis cap be
defeated upstream.
"""

from __future__ import annotations

import struct
import subprocess
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from skillspector.input_handler import (
    INGEST_MAX_BYTES,
    INGEST_MAX_ZIP_MEMBERS,
    IngestLimitExceededError,
    InputHandler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_httpx_client(monkeypatch: pytest.MonkeyPatch, handler: Callable) -> None:
    """Patch ``httpx.Client`` so ``InputHandler._download_file`` uses a MockTransport."""
    import skillspector.input_handler as ih

    real_client = httpx.Client

    def factory(*args: object, **kwargs: object) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(ih.httpx, "Client", factory)


def _make_zip(zip_path: Path, members: list[tuple[str, bytes]]) -> None:
    """Write ``members`` as a real zip file to ``zip_path``."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members:
            zf.writestr(name, data)


def _make_bomb_zip(zip_path: Path, declared_uncompressed: int) -> None:
    """Forge a zip whose ``ZipInfo.file_size`` declares an oversized member.

    We can't easily construct a true compression bomb in-test, but the
    extractor's check is against the declared uncompressed size from the
    central directory.  We write a one-member zip and then rewrite the
    uncompressed-size field in the central directory record.
    """
    name = "bomb.bin"
    payload = b"a"  # one real byte
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, payload)

    # Patch the central directory's "uncompressed size" field for the
    # one member.  Format from PKZIP APPNOTE 4.4.13:
    #   Central directory record: 4-byte sig (0x02014b50), then
    #     2 version-made-by, 2 version-needed, 2 flags, 2 method,
    #     2 mtime, 2 mdate, 4 crc32,
    #     4 compressed size, 4 uncompressed size, ...
    # So uncompressed-size offset within the record is 24 bytes from sig.
    raw = zip_path.read_bytes()
    sig = b"\x50\x4b\x01\x02"
    idx = raw.find(sig)
    assert idx >= 0, "central directory record not found"
    uncomp_offset = idx + 24
    patched = (
        raw[:uncomp_offset] + struct.pack("<I", declared_uncompressed) + raw[uncomp_offset + 4 :]
    )
    zip_path.write_bytes(patched)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


class TestDownloadBound:
    """``_download_file`` aborts oversized downloads before buffering them."""

    def test_under_cap_downloads_succeed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        body = b"# small markdown\n"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body)

        _patch_httpx_client(monkeypatch, handler)

        h = InputHandler()
        try:
            resolved, source_type = h.resolve("https://example.com/skill.md")
            assert source_type == "url"
            assert (resolved / "skill.md").read_bytes() == body
        finally:
            h.cleanup()

    def test_content_length_header_rejected_before_body_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Server declares an oversized Content-Length → reject before reading body.

        httpx normalises the ``content`` arg's length into Content-Length,
        so we ship a chunked stream and inject a forged header via a raw
        ``httpx.Response`` constructed from a byte-stream + explicit headers.
        """
        oversized = INGEST_MAX_BYTES + 1

        def handler(request: httpx.Request) -> httpx.Response:
            # Drop Transfer-Encoding to be sure Content-Length is the
            # only size signal; ship a tiny body so iter_bytes() would
            # complete almost instantly if we ever got there.
            return httpx.Response(
                200,
                stream=httpx.ByteStream(b"x"),
                headers={"content-length": str(oversized)},
            )

        _patch_httpx_client(monkeypatch, handler)

        h = InputHandler()
        try:
            with pytest.raises(IngestLimitExceededError, match="Content-Length"):
                h.resolve("https://example.com/huge.md")
        finally:
            h.cleanup()

    def test_streamed_body_overflow_rejected_when_header_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No Content-Length header → streamed byte-counter must catch overflow.

        Use a generator-backed stream so httpx cannot pre-compute and
        attach a Content-Length header, then ship oversized bytes.
        """

        def body_iter():
            chunk = b"x" * (64 * 1024)
            # Yield enough chunks to exceed the cap.
            sent = 0
            while sent <= INGEST_MAX_BYTES + 1024:
                yield chunk
                sent += len(chunk)

        class _GenStream(httpx.SyncByteStream):
            def __iter__(self):
                return body_iter()

            def close(self):  # noqa: D401 - protocol method
                pass

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=_GenStream())

        _patch_httpx_client(monkeypatch, handler)

        h = InputHandler()
        try:
            with pytest.raises(IngestLimitExceededError, match="streamed"):
                h.resolve("https://example.com/huge.bin")
        finally:
            h.cleanup()

    def test_streamed_overflow_leaves_no_partial_file_on_disk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A breach mid-stream must clean up the partial file.

        Closes the security-review finding: even when the cap fires,
        the bytes written before the breach must not survive on disk.
        Otherwise an attacker can still fill the temp dir up to
        ~INGEST_MAX_BYTES by sending exactly one byte over the cap.
        """

        def body_iter():
            chunk = b"x" * (64 * 1024)
            sent = 0
            while sent <= INGEST_MAX_BYTES + 1024:
                yield chunk
                sent += len(chunk)

        class _GenStream(httpx.SyncByteStream):
            def __iter__(self):
                return body_iter()

            def close(self):  # noqa: D401 - protocol method
                pass

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=_GenStream())

        _patch_httpx_client(monkeypatch, handler)

        h = InputHandler()
        try:
            with pytest.raises(IngestLimitExceededError):
                h.resolve("https://example.com/huge.bin")
            temp = h.temp_dir_for_cleanup()
            assert temp is not None
            # The partial download file must not survive the breach.
            assert not (temp / "_download.partial").exists()
            assert not (temp / "huge.bin").exists()
            assert not (temp / "download.zip").exists()
        finally:
            h.cleanup()

    def test_download_streams_to_disk_not_memory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A legitimate download must write incrementally to disk.

        Verifies the body is not buffered as a single ``bytes`` object
        in memory — the streaming refactor uses ``file.write()`` per
        chunk.  We can't directly measure peak memory in a unit test,
        but we can assert the on-disk file ends up at the same size as
        the bytes the server shipped, with no intermediate concatenation.
        """
        # 5 MiB body — well under the cap, large enough that a single
        # ``b''.join(chunks)`` would be a visible allocation if it ever
        # happened.
        body = b"a" * (5 * 1024 * 1024)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body)

        _patch_httpx_client(monkeypatch, handler)

        h = InputHandler()
        try:
            resolved, source_type = h.resolve("https://example.com/medium.bin")
            assert source_type == "url"
            assert (resolved / "medium.bin").stat().st_size == len(body)
            # And the sentinel partial-download path must not survive.
            assert not (resolved / "_download.partial").exists()
        finally:
            h.cleanup()


# ---------------------------------------------------------------------------
# Zip
# ---------------------------------------------------------------------------


class TestZipBound:
    """``_extract_zip`` refuses zip bombs and member-count bombs."""

    def test_under_cap_zip_succeeds(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "ok.zip"
        _make_zip(zip_path, [("SKILL.md", b"# skill")])

        h = InputHandler()
        try:
            resolved, source_type = h.resolve(str(zip_path))
            assert source_type == "zip"
            assert resolved.is_dir()
            assert (resolved / "SKILL.md").exists()
        finally:
            h.cleanup()

    def test_declared_uncompressed_oversize_rejected_before_extract(self, tmp_path: Path) -> None:
        """Classic zip bomb: small archive, declared-uncompressed size > cap."""
        zip_path = tmp_path / "bomb.zip"
        _make_bomb_zip(zip_path, declared_uncompressed=INGEST_MAX_BYTES + 1)

        h = InputHandler()
        try:
            with pytest.raises(IngestLimitExceededError, match="uncompressed"):
                h.resolve(str(zip_path))
            # Crucially: nothing extracted.  The extract dir may exist
            # (we mkdir before pre-checking) but must be empty.
            temp = h.temp_dir_for_cleanup()
            assert temp is not None
            extract_dir = temp / "extracted"
            if extract_dir.exists():
                assert list(extract_dir.iterdir()) == []
        finally:
            h.cleanup()

    def test_too_many_members_rejected(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "many.zip"
        # One byte each, but more entries than the member cap.
        members = [(f"file{i}.txt", b"x") for i in range(INGEST_MAX_ZIP_MEMBERS + 1)]
        _make_zip(zip_path, members)

        h = InputHandler()
        try:
            with pytest.raises(IngestLimitExceededError, match="members"):
                h.resolve(str(zip_path))
        finally:
            h.cleanup()


# ---------------------------------------------------------------------------
# Git clone
# ---------------------------------------------------------------------------


class TestGitCloneBound:
    """``_clone_git`` rejects clones whose on-disk size exceeds the cap."""

    def test_under_cap_clone_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def fake_run(cmd, **kwargs):
            # cmd is ["git", "clone", "--depth", "1", url, str(clone_dir)]
            clone_dir = Path(cmd[-1])
            clone_dir.mkdir(parents=True, exist_ok=True)
            (clone_dir / "SKILL.md").write_text("# small")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        monkeypatch.setattr(subprocess, "run", fake_run)

        h = InputHandler()
        try:
            resolved, source_type = h.resolve("https://github.com/foo/bar")
            assert source_type == "git"
            assert (resolved / "SKILL.md").exists()
        finally:
            h.cleanup()

    def test_oversize_clone_rejected_and_cleaned_up(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        big = b"x" * (INGEST_MAX_BYTES + 1)

        def fake_run(cmd, **kwargs):
            clone_dir = Path(cmd[-1])
            clone_dir.mkdir(parents=True, exist_ok=True)
            (clone_dir / "huge.bin").write_bytes(big)
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        monkeypatch.setattr(subprocess, "run", fake_run)

        h = InputHandler()
        try:
            with pytest.raises(IngestLimitExceededError, match="Git clone"):
                h.resolve("https://github.com/foo/huge-repo")
            # Failed clone must be cleaned up.
            temp = h.temp_dir_for_cleanup()
            assert temp is not None
            assert not (temp / "repo").exists()
        finally:
            h.cleanup()
