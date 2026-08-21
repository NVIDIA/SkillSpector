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
from stat import S_IFIFO, S_IFLNK

import httpx
import pytest

from skillspector.input_handler import (
    INGEST_MAX_BYTES,
    INGEST_MAX_ZIP_CENTRAL_DIRECTORY_BYTES,
    INGEST_MAX_ZIP_MEMBERS,
    IngestLimitExceededError,
    InputHandler,
    TransitiveIngestTruncatedError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_httpx_client(monkeypatch: pytest.MonkeyPatch, handler: Callable) -> None:
    """Patch ``httpx.Client`` so ``InputHandler._download_file`` uses a MockTransport.

    Also stubs the SSRF private-IP resolver so unit tests stay hermetic
    (the production check does a real DNS lookup on the URL host).
    """
    import skillspector.input_handler as ih

    real_client = httpx.Client

    def factory(*args: object, **kwargs: object) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(ih.httpx, "Client", factory)
    monkeypatch.setattr(ih, "_is_private_ip", lambda host: False)


# All download-path tests below hit an allowlisted host
# (``raw.githubusercontent.com``) rather than the pre-SSRF-hardening
# ``example.com`` placeholder — ``_validate_url_host`` now rejects
# hosts that are not in ``ALLOWED_DOWNLOAD_HOSTS`` before the mocked
# transport is ever reached.
_ALLOWED_HOST = "raw.githubusercontent.com"


class _CompletedGitProcess:
    """Minimal successful ``Popen`` stand-in for clone-bound tests."""

    def poll(self) -> int:
        return 0

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None


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


def _make_zip64_count_claim(zip_path: Path, entries: int) -> None:
    """Promote a small ZIP to ZIP64 while claiming an arbitrary entry count."""
    _make_zip(zip_path, [("SKILL.md", b"# skill")])
    raw = zip_path.read_bytes()
    eocd_offset = raw.rfind(b"PK\x05\x06")
    assert eocd_offset >= 0
    eocd = bytearray(raw[eocd_offset:])
    (
        _signature,
        _disk,
        _directory_disk,
        _entries_on_disk,
        _entries,
        directory_size,
        directory_offset,
        _comment_size,
    ) = struct.unpack("<4s4H2LH", eocd[:22])
    zip64_offset = eocd_offset
    zip64_eocd = struct.pack(
        "<4sQ2H2L4Q",
        b"PK\x06\x06",
        44,
        45,
        45,
        0,
        0,
        entries,
        entries,
        directory_size,
        directory_offset,
    )
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, zip64_offset, 1)
    struct.pack_into("<HHLL", eocd, 8, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
    zip_path.write_bytes(raw[:eocd_offset] + zip64_eocd + locator + eocd)


class _TransitiveBudget:
    def __init__(self, *, remaining_bytes: int, remaining_seconds: float = 60.0) -> None:
        self.bytes = remaining_bytes
        self.seconds = remaining_seconds
        self.reasons: list[str] = []

    def remaining_bytes(self) -> int:
        return self.bytes

    def remaining_seconds(self) -> float:
        return self.seconds

    def note_truncation(self, reason: str) -> None:
        self.reasons.append(reason)


class _RecordingBudget:
    """Small exact shared-budget stand-in for materialization tests."""

    def __init__(self, *, max_bytes: int, max_artifacts: int, seconds: float = 60.0) -> None:
        self.max_bytes = max_bytes
        self.max_artifacts = max_artifacts
        self.seconds = seconds
        self.scanned_bytes = 0
        self.scanned_artifacts = 0
        self.reasons: list[str] = []

    def remaining_seconds(self) -> float:
        return self.seconds

    def remaining_bytes(self) -> int:
        return max(0, self.max_bytes - self.scanned_bytes)

    def remaining_artifacts(self) -> int:
        return max(0, self.max_artifacts - self.scanned_artifacts)

    def record_bytes(self, count: int) -> None:
        self.scanned_bytes += max(0, count)

    def record_artifacts(self, count: int) -> None:
        self.scanned_artifacts += max(0, count)

    def note_truncation(self, reason: str) -> None:
        self.reasons.append(reason)


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
            resolved, source_type = h.resolve("https://raw.githubusercontent.com/skill.md")
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
                h.resolve("https://raw.githubusercontent.com/huge.md")
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
                h.resolve("https://raw.githubusercontent.com/huge.bin")
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
                h.resolve("https://raw.githubusercontent.com/huge.bin")
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
            resolved, source_type = h.resolve("https://raw.githubusercontent.com/medium.bin")
            assert source_type == "url"
            assert (resolved / "medium.bin").stat().st_size == len(body)
            # And the sentinel partial-download path must not survive.
            assert not (resolved / "_download.partial").exists()
        finally:
            h.cleanup()

    def test_shared_budget_charges_exact_download_bytes_and_artifact(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exactly-sized remote file consumes, but does not exceed, both budgets."""
        body = b"# exact\n"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body)

        _patch_httpx_client(monkeypatch, handler)
        budget = _RecordingBudget(max_bytes=len(body), max_artifacts=1)
        h = InputHandler(transitive_budget=budget)
        try:
            resolved, source_type = h.resolve(f"https://{_ALLOWED_HOST}/exact.md")
            assert source_type == "url"
            assert (resolved / "exact.md").read_bytes() == body
            assert budget.scanned_bytes == len(body)
            assert budget.scanned_artifacts == 1
            assert budget.remaining_bytes() == 0
            assert budget.remaining_artifacts() == 0
            assert budget.reasons == []
        finally:
            h.cleanup()

    def test_shared_download_artifact_budget_fails_before_network(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_httpx_client(
            monkeypatch,
            lambda _request: pytest.fail("network accessed after artifact budget exhaustion"),
        )
        budget = _RecordingBudget(max_bytes=1, max_artifacts=0)
        h = InputHandler(transitive_budget=budget)
        try:
            with pytest.raises(TransitiveIngestTruncatedError) as raised:
                h.resolve(f"https://{_ALLOWED_HOST}/blocked.md")
            assert raised.value.truncation.as_dict() == {
                "code": "artifact_budget_exhausted",
                "source_type": "download",
                "message": "Transitive download ingest truncated (artifact_budget_exhausted)",
            }
            assert h.temp_dir_for_cleanup() is None
            assert budget.reasons == [raised.value.truncation.message]
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

    def test_shared_artifact_budget_rejects_member_suffix_before_extraction(
        self, tmp_path: Path
    ) -> None:
        zip_path = tmp_path / "two-members.zip"
        _make_zip(zip_path, [("SKILL.md", b"# skill"), ("run.py", b"print(1)\n")])
        budget = _RecordingBudget(max_bytes=1024, max_artifacts=1)

        h = InputHandler(transitive_budget=budget)
        try:
            with pytest.raises(TransitiveIngestTruncatedError) as raised:
                h.resolve(str(zip_path))
            assert raised.value.truncation.as_dict() == {
                "code": "artifact_budget_exhausted",
                "source_type": "zip",
                "message": "Transitive zip ingest truncated (artifact_budget_exhausted)",
            }
            assert "SKILL.md" not in str(raised.value)
            assert "two-members.zip" not in str(raised.value)
            assert budget.scanned_artifacts == 0
            assert budget.reasons == [raised.value.truncation.message]
            assert h.temp_dir_for_cleanup() is None
        finally:
            h.cleanup()

    def test_shared_zip_budgets_allow_exact_bytes_and_implicit_directory(
        self, tmp_path: Path
    ) -> None:
        payload = b"# exact zip\n"
        zip_path = tmp_path / "exact.zip"
        _make_zip(zip_path, [("nested/SKILL.md", payload)])
        # Extraction materializes both the implicit directory and the file.
        budget = _RecordingBudget(max_bytes=len(payload), max_artifacts=2)

        h = InputHandler(transitive_budget=budget)
        try:
            resolved, source_type = h.resolve(str(zip_path))
            assert source_type == "zip"
            assert (resolved / "SKILL.md").read_bytes() == payload
            assert budget.scanned_bytes == len(payload)
            assert budget.scanned_artifacts == 2
            assert budget.remaining_bytes() == 0
            assert budget.remaining_artifacts() == 0
            assert budget.reasons == []
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
            # Crucially: nothing is materialized before the metadata check.
            temp = h.temp_dir_for_cleanup()
            if temp is not None:
                assert not (temp / "extracted").exists()
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

    def test_eocd_count_rejected_before_infolist(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An oversized ordinary EOCD count never reaches ``ZipFile.infolist``."""
        zip_path = tmp_path / "claimed-many.zip"
        _make_zip(zip_path, [("SKILL.md", b"# skill")])
        raw = bytearray(zip_path.read_bytes())
        eocd_offset = raw.rfind(b"PK\x05\x06")
        assert eocd_offset >= 0
        struct.pack_into("<HH", raw, eocd_offset + 8, 0xFFFE, 0xFFFE)
        zip_path.write_bytes(raw)
        monkeypatch.setattr(
            zipfile.ZipFile,
            "infolist",
            lambda _self: pytest.fail("infolist materialized an over-count directory"),
        )

        h = InputHandler()
        try:
            with pytest.raises(IngestLimitExceededError, match="members"):
                h.resolve(str(zip_path))
        finally:
            h.cleanup()

    def test_actual_central_records_cannot_hide_behind_small_eocd_count(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Preflight counts real records, not just attacker-controlled EOCD fields."""
        zip_path = tmp_path / "lying-count.zip"
        _make_zip(zip_path, [(f"file-{index}", b"x") for index in range(4)])
        raw = bytearray(zip_path.read_bytes())
        eocd_offset = raw.rfind(b"PK\x05\x06")
        assert eocd_offset >= 0
        struct.pack_into("<HH", raw, eocd_offset + 8, 1, 1)
        zip_path.write_bytes(raw)
        monkeypatch.setattr("skillspector.input_handler.INGEST_MAX_ZIP_MEMBERS", 2)
        monkeypatch.setattr(
            zipfile.ZipFile,
            "infolist",
            lambda _self: pytest.fail("infolist materialized a lying central directory"),
        )

        h = InputHandler()
        try:
            with pytest.raises(IngestLimitExceededError, match="preflighting members"):
                h.resolve(str(zip_path))
        finally:
            h.cleanup()

    def test_zip64_count_rejected_before_infolist(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """ZIP64 EOCD counts receive the same pre-materialization bound."""
        zip_path = tmp_path / "claimed-many-zip64.zip"
        _make_zip64_count_claim(zip_path, INGEST_MAX_ZIP_MEMBERS + 1)
        monkeypatch.setattr(
            zipfile.ZipFile,
            "infolist",
            lambda _self: pytest.fail("infolist materialized an over-count ZIP64 directory"),
        )

        h = InputHandler()
        try:
            with pytest.raises(IngestLimitExceededError, match="members"):
                h.resolve(str(zip_path))
        finally:
            h.cleanup()

    def test_zip64_under_cap_extracts(self, tmp_path: Path) -> None:
        """ZIP64 preflight preserves a valid small archive."""
        zip_path = tmp_path / "small-zip64.zip"
        _make_zip64_count_claim(zip_path, 1)

        h = InputHandler()
        try:
            resolved, source_type = h.resolve(str(zip_path))
            assert source_type == "zip"
            assert (resolved / "SKILL.md").read_bytes() == b"# skill"
        finally:
            h.cleanup()

    def test_central_directory_bytes_rejected_before_infolist(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Variable-length central-directory metadata has an independent cap."""
        zip_path = tmp_path / "metadata.zip"
        _make_zip(zip_path, [("SKILL.md", b"# skill")])
        assert INGEST_MAX_ZIP_CENTRAL_DIRECTORY_BYTES > 1
        monkeypatch.setattr("skillspector.input_handler.INGEST_MAX_ZIP_CENTRAL_DIRECTORY_BYTES", 1)
        monkeypatch.setattr(
            zipfile.ZipFile,
            "infolist",
            lambda _self: pytest.fail("infolist materialized oversized metadata"),
        )

        h = InputHandler()
        try:
            with pytest.raises(IngestLimitExceededError, match="central-directory"):
                h.resolve(str(zip_path))
        finally:
            h.cleanup()

    @pytest.mark.parametrize("mode", [S_IFLNK | 0o777, S_IFIFO | 0o600])
    def test_links_and_special_members_are_rejected(self, tmp_path: Path, mode: int) -> None:
        zip_path = tmp_path / "special.zip"
        info = zipfile.ZipInfo("unsafe")
        info.create_system = 3
        info.external_attr = mode << 16
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(info, b"target")

        h = InputHandler()
        try:
            with pytest.raises(ValueError, match="links|special-file"):
                h.resolve(str(zip_path))
            temp = h.temp_dir_for_cleanup()
            assert temp is not None
            assert not (temp / "extracted").exists()
        finally:
            h.cleanup()

    def test_prefix_sibling_zip_slip_is_rejected(self, tmp_path: Path) -> None:
        """``../extracted_evil`` must not pass a string-prefix containment check."""
        zip_path = tmp_path / "prefix-slip.zip"
        _make_zip(zip_path, [("../extracted_evil/payload", b"bad")])
        h = InputHandler()
        try:
            with pytest.raises(ValueError, match="zip-slip"):
                h.resolve(str(zip_path))
            temp = h.temp_dir_for_cleanup()
            assert temp is not None
            assert not (temp / "extracted_evil").exists()
        finally:
            h.cleanup()

    @pytest.mark.parametrize(
        "member",
        ["CON", "folder/NUL.txt", "tool.py:payload", "trailing.", "trailing "],
    )
    def test_cross_platform_ambiguous_zip_paths_are_rejected(
        self, tmp_path: Path, member: str
    ) -> None:
        zip_path = tmp_path / "ambiguous.zip"
        _make_zip(zip_path, [(member, b"bad")])
        h = InputHandler()
        try:
            with pytest.raises(ValueError, match="zip-slip"):
                h.resolve(str(zip_path))
        finally:
            h.cleanup()

    def test_extraction_deadline_is_enforced(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        zip_path = tmp_path / "slow.zip"
        _make_zip(zip_path, [("SKILL.md", b"# skill")])
        clock = iter((0.0, 0.0, 0.0, 61.0))
        monkeypatch.setattr("skillspector.input_handler.monotonic", lambda: next(clock, 61.0))

        h = InputHandler()
        try:
            with pytest.raises(IngestLimitExceededError, match="time limit"):
                h.resolve(str(zip_path))
            temp = h.temp_dir_for_cleanup()
            if temp is not None:
                assert not (temp / "extracted").exists()
        finally:
            h.cleanup()

    def test_implicit_directories_count_toward_extraction_cap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        zip_path = tmp_path / "deep-count.zip"
        _make_zip(zip_path, [("a/b/c/file", b"x")])
        monkeypatch.setattr("skillspector.input_handler.INGEST_MAX_ZIP_MEMBERS", 3)

        h = InputHandler()
        try:
            with pytest.raises(IngestLimitExceededError, match="extracted-entry cap"):
                h.resolve(str(zip_path))
            temp = h.temp_dir_for_cleanup()
            assert temp is not None
            assert not (temp / "extracted").exists()
        finally:
            h.cleanup()


# ---------------------------------------------------------------------------
# Git clone
# ---------------------------------------------------------------------------


def _stub_private_ip_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass the real DNS lookup in ``_is_private_ip`` for hermetic tests."""
    import skillspector.input_handler as ih

    monkeypatch.setattr(ih, "_is_private_ip", lambda host: False)


class TestGitCloneBound:
    """``_clone_git`` rejects clones whose on-disk size exceeds the cap."""

    def test_under_cap_clone_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _stub_private_ip_check(monkeypatch)

        def fake_popen(cmd, **kwargs):
            # cmd is ["git", "clone", "--depth", "1", url, str(clone_dir)]
            clone_dir = Path(cmd[-1])
            clone_dir.mkdir(parents=True, exist_ok=True)
            (clone_dir / "SKILL.md").write_text("# small")
            return _CompletedGitProcess()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)

        h = InputHandler()
        try:
            resolved, source_type = h.resolve("https://github.com/foo/bar")
            assert source_type == "git"
            assert (resolved / "SKILL.md").exists()
        finally:
            h.cleanup()

    def test_shared_git_budget_charges_dot_git_bytes_at_exact_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_private_ip_check(monkeypatch)
        worktree_bytes = b"x"
        git_bytes = b"abc"
        # Entries: .git, SKILL.md, .git/objects, and its pack file.
        budget = _RecordingBudget(
            max_bytes=len(worktree_bytes) + len(git_bytes),
            max_artifacts=4,
        )

        def fake_popen(cmd, **kwargs):
            clone_dir = Path(cmd[-1])
            objects_dir = clone_dir / ".git" / "objects"
            objects_dir.mkdir(parents=True)
            (objects_dir / "pack").write_bytes(git_bytes)
            (clone_dir / "SKILL.md").write_bytes(worktree_bytes)
            return _CompletedGitProcess()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        h = InputHandler(transitive_budget=budget)
        try:
            resolved, source_type = h.resolve("https://github.com/foo/exact")
            assert source_type == "git"
            assert (resolved / "SKILL.md").read_bytes() == worktree_bytes
            assert budget.scanned_bytes == len(worktree_bytes) + len(git_bytes)
            assert budget.scanned_artifacts == 4
            assert budget.remaining_bytes() == 0
            assert budget.remaining_artifacts() == 0
            assert budget.reasons == []
        finally:
            h.cleanup()

    def test_shared_git_byte_limit_includes_dot_git_and_is_sanitized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_private_ip_check(monkeypatch)
        budget = _RecordingBudget(max_bytes=3, max_artifacts=10)

        def fake_popen(cmd, **kwargs):
            clone_dir = Path(cmd[-1])
            objects_dir = clone_dir / ".git" / "objects"
            objects_dir.mkdir(parents=True)
            (objects_dir / "secret-pack-name").write_bytes(b"xxxx")
            return _CompletedGitProcess()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        h = InputHandler(transitive_budget=budget)
        try:
            with pytest.raises(TransitiveIngestTruncatedError) as raised:
                h.resolve("https://github.com/foo/private-name")
            assert raised.value.truncation.as_dict() == {
                "code": "byte_budget_exhausted",
                "source_type": "git",
                "message": "Transitive git ingest truncated (byte_budget_exhausted)",
            }
            assert "secret-pack-name" not in str(raised.value)
            assert "private-name" not in str(raised.value)
            assert budget.reasons == [raised.value.truncation.message]
            temp = h.temp_dir_for_cleanup()
            assert temp is not None
            assert not (temp / "repo").exists()
        finally:
            h.cleanup()

    def test_shared_git_artifact_limit_is_typed_and_cleans_partial_clone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_private_ip_check(monkeypatch)
        budget = _RecordingBudget(max_bytes=100, max_artifacts=1)

        def fake_popen(cmd, **kwargs):
            clone_dir = Path(cmd[-1])
            clone_dir.mkdir(parents=True)
            (clone_dir / "SKILL.md").write_bytes(b"x")
            (clone_dir / "second.txt").write_bytes(b"y")
            return _CompletedGitProcess()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        h = InputHandler(transitive_budget=budget)
        try:
            with pytest.raises(TransitiveIngestTruncatedError) as raised:
                h.resolve("https://github.com/foo/artifact-limit")
            assert raised.value.truncation.as_dict() == {
                "code": "artifact_budget_exhausted",
                "source_type": "git",
                "message": "Transitive git ingest truncated (artifact_budget_exhausted)",
            }
            assert budget.reasons == [raised.value.truncation.message]
            temp = h.temp_dir_for_cleanup()
            assert temp is not None
            assert not (temp / "repo").exists()
        finally:
            h.cleanup()

    def test_clone_tree_entry_count_is_bounded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _stub_private_ip_check(monkeypatch)
        monkeypatch.setattr("skillspector.input_handler.INGEST_MAX_TREE_ENTRIES", 3)

        def fake_popen(cmd, **kwargs):
            clone_dir = Path(cmd[-1])
            clone_dir.mkdir(parents=True, exist_ok=True)
            for index in range(4):
                (clone_dir / f"file-{index}").write_bytes(b"x")
            return _CompletedGitProcess()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        h = InputHandler()
        try:
            with pytest.raises(IngestLimitExceededError, match="entry cap"):
                h.resolve("https://github.com/foo/many-files")
            temp = h.temp_dir_for_cleanup()
            assert temp is not None
            assert not (temp / "repo").exists()
        finally:
            h.cleanup()

    def test_running_clone_is_terminated_when_inflight_tree_exceeds_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The clone process is stopped during materialization, with bounded output pipes."""
        _stub_private_ip_check(monkeypatch)
        monkeypatch.setattr("skillspector.input_handler.INGEST_MAX_BYTES", 10)
        popen_kwargs: list[dict[str, object]] = []

        class GrowingProcess:
            def __init__(self, command: list[str]) -> None:
                clone_dir = Path(command[-1])
                clone_dir.mkdir(parents=True)
                (clone_dir / "pack.bin").write_bytes(b"x" * 11)
                self.terminated = False

            def poll(self) -> int | None:
                return -15 if self.terminated else None

            def wait(self, timeout: float | None = None) -> int:
                return -15 if self.terminated else 0

            def terminate(self) -> None:
                self.terminated = True

            def kill(self) -> None:
                self.terminated = True

        processes: list[GrowingProcess] = []

        def fake_popen(command: list[str], **kwargs: object) -> GrowingProcess:
            process = GrowingProcess(command)
            processes.append(process)
            popen_kwargs.append(kwargs)
            return process

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        handler = InputHandler()
        try:
            with pytest.raises(IngestLimitExceededError, match="Git clone"):
                handler.resolve("https://github.com/foo/growing")
            assert len(processes) == 1
            assert processes[0].terminated is True
            assert popen_kwargs == [
                {
                    "stdout": subprocess.DEVNULL,
                    "stderr": subprocess.DEVNULL,
                    "shell": False,
                }
            ]
            temp = handler.temp_dir_for_cleanup()
            assert temp is not None
            assert not (temp / "repo").exists()
        finally:
            handler.cleanup()

    def test_transitive_clone_limit_is_typed_and_not_an_empty_directory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An exhausted budget must short-circuit before DNS or subprocess work.
        monkeypatch.setattr(
            "skillspector.input_handler._is_private_ip",
            lambda _host: pytest.fail("DNS accessed after budget exhaustion"),
        )
        budget = _TransitiveBudget(remaining_bytes=0)
        h = InputHandler(transitive_budget=budget)
        try:
            with pytest.raises(TransitiveIngestTruncatedError) as raised:
                h.resolve("https://github.com/foo/too-late")
            assert raised.value.truncation.as_dict() == {
                "code": "byte_budget_exhausted",
                "source_type": "git",
                "message": "Transitive git ingest truncated (byte_budget_exhausted)",
            }
            assert h.temp_dir_for_cleanup() is None
            assert budget.reasons == [raised.value.truncation.message]
        finally:
            h.cleanup()

    def test_oversize_clone_rejected_and_cleaned_up(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _stub_private_ip_check(monkeypatch)
        big = b"x" * (INGEST_MAX_BYTES + 1)

        def fake_popen(cmd, **kwargs):
            clone_dir = Path(cmd[-1])
            clone_dir.mkdir(parents=True, exist_ok=True)
            (clone_dir / "huge.bin").write_bytes(big)
            return _CompletedGitProcess()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)

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


def test_transitive_download_limit_is_typed_without_materializing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_httpx_client(
        monkeypatch,
        lambda _request: pytest.fail("network accessed after byte budget exhaustion"),
    )
    budget = _TransitiveBudget(remaining_bytes=0)
    h = InputHandler(transitive_budget=budget)
    try:
        with pytest.raises(TransitiveIngestTruncatedError) as raised:
            h.resolve(f"https://{_ALLOWED_HOST}/skill.md")
        assert raised.value.truncation.source_type == "download"
        assert raised.value.truncation.code == "byte_budget_exhausted"
        assert h.temp_dir_for_cleanup() is None
    finally:
        h.cleanup()


def test_transitive_zip_limit_is_typed_without_empty_extract_dir(tmp_path: Path) -> None:
    zip_path = tmp_path / "skill.zip"
    _make_zip(zip_path, [("SKILL.md", b"# skill")])
    budget = _TransitiveBudget(remaining_bytes=0)
    h = InputHandler(transitive_budget=budget)
    try:
        with pytest.raises(TransitiveIngestTruncatedError) as raised:
            h.resolve(str(zip_path))
        assert raised.value.truncation.source_type == "zip"
        assert raised.value.truncation.code == "byte_budget_exhausted"
        assert h.temp_dir_for_cleanup() is None
    finally:
        h.cleanup()
