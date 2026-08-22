# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for bounded, local-only nested artifact inspection."""

from __future__ import annotations

import io
import json
import stat
import struct
import zipfile
from pathlib import Path

import pytest

from skillspector.artifacts import ArtifactDisposition, ContentKind
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nested_artifacts import inspect_nested_artifacts
from skillspector.nodes.analyzers.static_patterns_supply_chain import (
    _analyze_concealed_executables,
)
from skillspector.nodes.build_context import build_context


def _zip_bytes(
    members: dict[str, bytes],
    *,
    compression: int = zipfile.ZIP_STORED,
    link: str | None = None,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        for name, content in members.items():
            if name == link:
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, content)
            else:
                archive.writestr(name, content)
    return buffer.getvalue()


def _write_archive(path: Path, members: dict[str, bytes]) -> None:
    path.write_bytes(_zip_bytes(members))


def _document_members(**extra: bytes) -> dict[str, bytes]:
    return {
        "[Content_Types].xml": b"<Types/>",
        "word/document.xml": b"<document>ordinary text</document>",
        **extra,
    }


def _with_unsupported_compression(data: bytes, method: int = 99) -> bytes:
    encoded = bytearray(data)
    local_header = encoded.find(b"PK\x03\x04")
    central_header = encoded.find(b"PK\x01\x02")
    assert local_header >= 0 and central_header >= 0
    encoded[local_header + 8 : local_header + 10] = method.to_bytes(2, "little")
    encoded[central_header + 10 : central_header + 12] = method.to_bytes(2, "little")
    return bytes(encoded)


def _with_zip64_eocd(data: bytes) -> bytes:
    """Promote a small ordinary ZIP to a standards-shaped ZIP64 EOCD fixture."""
    eocd = data.rfind(b"PK\x05\x06")
    assert eocd >= 0
    encoded = bytearray(data[eocd:])
    entries = int.from_bytes(encoded[10:12], "little")
    directory_size = int.from_bytes(encoded[12:16], "little")
    directory_offset = int.from_bytes(encoded[16:20], "little")
    encoded[8:12] = b"\xff\xff\xff\xff"
    encoded[12:20] = b"\xff" * 8
    zip64_eocd = struct.pack(
        "<4sQHHIIQQQQ",
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
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, eocd, 1)
    return data[:eocd] + zip64_eocd + locator + bytes(encoded)


def test_transitive_limit_caps_nested_uncompressed_bytes(tmp_path: Path) -> None:
    """A caller-provided child budget prevents archive expansion past its allowance."""
    path = tmp_path / "bounded.zip"
    _write_archive(path, {"payload.txt": b"0123456789"})

    result = inspect_nested_artifacts(tmp_path, [path.name], max_uncompressed_bytes=5)

    virtual_path = "bounded.zip!/payload.txt"
    assert virtual_path not in result.file_cache
    assert result.inventory_overrides[path.name] == (
        ArtifactDisposition.PARTIAL,
        LedgerReason.ARCHIVE_SIZE_LIMIT.value,
    )
    assert result.uncompressed_bytes == 0
    assert any(
        event.get("reason_code") == LedgerReason.ARCHIVE_SIZE_LIMIT
        and event.get("limit_bytes") == 5
        for event in result.ledger_events
    )


def test_build_context_accounts_nested_bytes_to_transitive_budget(tmp_path: Path) -> None:
    """Extracted child members consume the same shared budget as ordinary files."""

    class Traversal:
        def __init__(self) -> None:
            self.scanned_bytes = 0
            self.reasons: list[str] = []

        def remaining_bytes(self) -> int:
            return 1024 * 1024 - self.scanned_bytes

        def remaining_seconds(self) -> float:
            return 60.0

        def record_bytes(self, count: int) -> None:
            self.scanned_bytes += count

        def note_truncation(self, reason: str) -> None:
            self.reasons.append(reason)

    (tmp_path / "SKILL.md").write_text("# Nested child\n", encoding="utf-8")
    _write_archive(tmp_path / "bundle.zip", {"payload.py": b"print('checked')\n"})
    traversal = Traversal()

    context = build_context({"skill_path": str(tmp_path), "transitive_traversal_state": traversal})

    assert "bundle.zip!/payload.py" in context["local_file_cache"]
    assert traversal.scanned_bytes == sum(
        len(content) for content in context["raw_file_cache"].values()
    )
    assert traversal.reasons == []


def test_hidden_disguised_document_inventories_nested_executable_locally(tmp_path: Path) -> None:
    archive_path = tmp_path / ".instructions.docx.txt"
    _write_archive(archive_path, _document_members(**{"word/sync1.sh": b"#!/bin/sh\necho ok\n"}))
    (tmp_path / "SKILL.md").write_text("# Context loader\n", encoding="utf-8")

    context = build_context({"skill_path": str(tmp_path)})
    virtual_path = ".instructions.docx.txt!/word/sync1.sh"

    assert ".instructions.docx.txt" in context["components"]
    assert virtual_path in context["components"]
    assert virtual_path in context["local_file_cache"]
    assert ".instructions.docx.txt" not in context["file_cache"]
    assert virtual_path not in context["file_cache"]
    outer = next(
        item for item in context["component_metadata"] if item["path"] == archive_path.name
    )
    nested = next(item for item in context["component_metadata"] if item["path"] == virtual_path)
    assert outer["type"] == "docx"
    assert outer["hidden"] is True
    assert outer["disguised"] is True
    assert nested["executable"] is True
    assert nested["concealed_executable"] is True

    findings = _analyze_concealed_executables(context["component_metadata"])
    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_id == "SC9"
    assert finding.severity == "HIGH"
    assert finding.file == virtual_path
    assert finding.evidence["outer_path"] == archive_path.name
    assert finding.evidence["nested_path"] == "word/sync1.sh"
    assert finding.evidence["container_type"] == "docx"


def test_benign_document_without_executable_has_no_sc9(tmp_path: Path) -> None:
    archive_path = tmp_path / "notes.docx"
    _write_archive(archive_path, _document_members())

    context = build_context({"skill_path": str(tmp_path)})

    assert not _analyze_concealed_executables(context["component_metadata"])


def test_hidden_standalone_executable_has_sc9_and_stays_local(tmp_path: Path) -> None:
    (tmp_path / ".setup.sh").write_text("#!/bin/sh\necho local\n", encoding="utf-8")

    context = build_context({"skill_path": str(tmp_path)})
    findings = _analyze_concealed_executables(context["component_metadata"])

    assert ".setup.sh" in context["components"]
    assert ".setup.sh" in context["local_file_cache"]
    assert ".setup.sh" not in context["file_cache"]
    assert len(findings) == 1
    assert findings[0].file == ".setup.sh"
    assert findings[0].evidence["container_type"] == "filesystem"
    assert findings[0].evidence["concealment"] == "hidden_artifact"


def test_nested_zip_preserves_full_virtual_provenance(tmp_path: Path) -> None:
    inner = _zip_bytes({"payload.sh": b"#!/bin/sh\necho nested\n"})
    outer_path = tmp_path / ".bundle.txt"
    _write_archive(outer_path, {"nested.bin": inner})

    result = inspect_nested_artifacts(tmp_path, [outer_path.name])

    assert ".bundle.txt!/nested.bin!/payload.sh" in result.components
    metadata = next(item for item in result.metadata if item["path"].endswith("!/payload.sh"))
    assert metadata["container_depth"] == 2
    assert metadata["concealed_executable"] is True


def test_supplied_outer_bytes_preserve_exact_virtual_member_bytes(tmp_path: Path) -> None:
    payload = b"\xff\x00\x80raw-member\r\n"
    outer_path = "cached.zip"
    supplied = _zip_bytes({"payload.bin": payload})

    result = inspect_nested_artifacts(
        tmp_path,
        [outer_path],
        raw_file_cache={outer_path: supplied},
    )

    virtual_path = "cached.zip!/payload.bin"
    assert not (tmp_path / outer_path).exists()
    assert result.raw_file_cache[virtual_path] == payload
    assert result.file_cache[virtual_path] == payload.decode("utf-8", errors="replace")
    artifact = next(item for item in result.artifact_inventory if item["path"] == virtual_path)
    assert artifact["size_bytes"] == len(payload)
    assert artifact["contains_nul"] is True


def test_zip64_eocd_is_preflighted_before_member_inspection(tmp_path: Path) -> None:
    outer_path = "zip64.zip"
    supplied = _with_zip64_eocd(_zip_bytes({"payload.txt": b"zip64 member"}))

    result = inspect_nested_artifacts(
        tmp_path,
        [outer_path],
        raw_file_cache={outer_path: supplied},
    )

    virtual_path = "zip64.zip!/payload.txt"
    assert result.components == [virtual_path]
    assert result.raw_file_cache[virtual_path] == b"zip64 member"
    assert not result.ledger_events


def test_archive_member_budget_is_shared_across_outer_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import skillspector.nested_artifacts as nested

    monkeypatch.setattr(nested, "ARCHIVE_MAX_MEMBERS", 1)
    _write_archive(tmp_path / "first.zip", {"first.txt": b"first"})
    _write_archive(tmp_path / "second.zip", {"second.txt": b"second"})

    result = inspect_nested_artifacts(tmp_path, ["first.zip", "second.zip"])

    assert result.components == ["first.zip!/first.txt"]
    assert [item["path"] for item in result.artifact_inventory] == result.components
    assert any(
        event.get("reason_code") == LedgerReason.ARCHIVE_MEMBER_LIMIT
        and event.get("path") == "second.zip"
        for event in result.ledger_events
    )


def test_preflight_rejects_actual_member_count_before_zipfile_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import skillspector.nested_artifacts as nested

    encoded = bytearray(_zip_bytes({"one.txt": b"1", "two.txt": b"2"}))
    eocd = encoded.rfind(b"PK\x05\x06")
    assert eocd >= 0
    # Lie in both EOCD count fields. The bounded central-header walk must still
    # observe that the one-member caller budget would be exceeded.
    encoded[eocd + 8 : eocd + 12] = b"\x00\x00\x00\x00"
    path = tmp_path / "forged-count.zip"
    path.write_bytes(bytes(encoded))

    def forbidden_zipfile(*args: object, **kwargs: object) -> None:
        raise AssertionError("ZipFile must not run before central-directory preflight")

    monkeypatch.setattr(nested.zipfile, "ZipFile", forbidden_zipfile)
    result = inspect_nested_artifacts(tmp_path, [path.name], max_members=1)

    assert any(
        event.get("reason_code") == LedgerReason.ARCHIVE_MEMBER_LIMIT
        and event.get("outcome") == LedgerOutcome.PARTIAL
        for event in result.ledger_events
    )


def test_preflight_rejects_large_central_directory_before_zipfile_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import skillspector.nested_artifacts as nested

    path = tmp_path / "large-directory.zip"
    _write_archive(path, {"member-with-a-long-name.txt": b"content"})

    def forbidden_zipfile(*args: object, **kwargs: object) -> None:
        raise AssertionError("ZipFile must not run before central-directory preflight")

    monkeypatch.setattr(nested, "ARCHIVE_MAX_CENTRAL_DIRECTORY_BYTES", 1)
    monkeypatch.setattr(nested.zipfile, "ZipFile", forbidden_zipfile)
    result = inspect_nested_artifacts(tmp_path, [path.name])

    event = next(
        event
        for event in result.ledger_events
        if event.get("reason_code") == LedgerReason.ARCHIVE_SIZE_LIMIT
    )
    assert event["observed_bytes"] > event["limit_bytes"]
    assert event["limit_bytes"] == 1
    assert event["outcome"] == LedgerOutcome.PARTIAL


def test_caller_supplied_remaining_budgets_are_enforced(tmp_path: Path) -> None:
    outer_path = "cached.zip"
    supplied = _zip_bytes({"four.txt": b"1234"})

    byte_limited = inspect_nested_artifacts(
        tmp_path,
        [outer_path],
        raw_file_cache={outer_path: supplied},
        max_uncompressed_bytes=3,
    )
    artifact = byte_limited.artifact_inventory[0]
    assert artifact["path"] == "cached.zip!/four.txt"
    assert artifact["content_kind"] == ContentKind.OPAQUE
    assert artifact["disposition"] == ArtifactDisposition.PARTIAL
    assert artifact["reason"] == LedgerReason.ARCHIVE_SIZE_LIMIT.value

    member_limited = inspect_nested_artifacts(
        tmp_path,
        [outer_path],
        raw_file_cache={outer_path: supplied},
        max_members=0,
    )
    assert not member_limited.components
    assert any(
        event.get("reason_code") == LedgerReason.ARCHIVE_MEMBER_LIMIT
        for event in member_limited.ledger_events
    )

    deadline_limited = inspect_nested_artifacts(
        tmp_path,
        [outer_path],
        raw_file_cache={outer_path: supplied},
        clock=lambda: 11.0,
        absolute_deadline=10.0,
    )
    assert not deadline_limited.components
    assert any(
        event.get("reason_code") == LedgerReason.ARCHIVE_TIME_LIMIT
        for event in deadline_limited.ledger_events
    )


def test_caller_allowances_cannot_raise_archive_specific_ceilings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import skillspector.nested_artifacts as nested

    outer_path = "cached.zip"
    supplied = _zip_bytes({"one.txt": b"1", "two.txt": b"22"})
    monkeypatch.setattr(nested, "ARCHIVE_MAX_MEMBERS", 1)
    member_limited = inspect_nested_artifacts(
        tmp_path,
        [outer_path],
        raw_file_cache={outer_path: supplied},
        max_members=100,
    )
    assert any(
        event.get("reason_code") == LedgerReason.ARCHIVE_MEMBER_LIMIT
        for event in member_limited.ledger_events
    )

    monkeypatch.setattr(nested, "ARCHIVE_MAX_MEMBERS", 10)
    monkeypatch.setattr(nested, "ARCHIVE_MAX_UNCOMPRESSED_BYTES", 1)
    byte_limited = inspect_nested_artifacts(
        tmp_path,
        [outer_path],
        raw_file_cache={outer_path: supplied},
        max_uncompressed_bytes=100,
    )
    assert any(
        event.get("reason_code") == LedgerReason.ARCHIVE_SIZE_LIMIT
        for event in byte_limited.ledger_events
    )

    times = iter((0.0, 6.0, 6.0))
    monkeypatch.setattr(nested, "ARCHIVE_MAX_SECONDS", 5.0)
    time_limited = inspect_nested_artifacts(
        tmp_path,
        [outer_path],
        raw_file_cache={outer_path: supplied},
        clock=lambda: next(times),
        absolute_deadline=100.0,
    )
    assert any(
        event.get("reason_code") == LedgerReason.ARCHIVE_TIME_LIMIT
        for event in time_limited.ledger_events
    )


def test_archive_byte_budget_is_shared_across_outer_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import skillspector.nested_artifacts as nested

    monkeypatch.setattr(nested, "ARCHIVE_MAX_UNCOMPRESSED_BYTES", 1_000)
    first = _zip_bytes({"first.txt": b"A" * 600}, compression=zipfile.ZIP_DEFLATED)
    second = _zip_bytes({"second.txt": b"B" * 600}, compression=zipfile.ZIP_DEFLATED)
    assert len(first) < 1_000 and len(second) < 1_000
    (tmp_path / "first.zip").write_bytes(first)
    (tmp_path / "second.zip").write_bytes(second)

    result = inspect_nested_artifacts(tmp_path, ["first.zip", "second.zip"])

    assert result.raw_file_cache["first.zip!/first.txt"] == b"A" * 600
    assert "second.zip!/second.txt" not in result.raw_file_cache
    limited = next(
        item for item in result.artifact_inventory if item["path"] == "second.zip!/second.txt"
    )
    assert limited["content_kind"] == ContentKind.OPAQUE
    assert limited["disposition"] == ArtifactDisposition.PARTIAL
    assert limited["reason"] == LedgerReason.ARCHIVE_SIZE_LIMIT.value


def test_archive_deadline_is_shared_across_outer_archives(tmp_path: Path) -> None:
    _write_archive(tmp_path / "first.zip", {"first.txt": b"first"})
    _write_archive(tmp_path / "second.zip", {"second.txt": b"second"})
    clock_calls = 0

    def clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        return 0.0 if clock_calls <= 6 else 6.0

    result = inspect_nested_artifacts(
        tmp_path,
        ["first.zip", "second.zip"],
        clock=clock,
    )

    assert "first.zip!/first.txt" in result.components
    assert "second.zip!/second.txt" not in result.components
    assert any(
        event.get("reason_code") == LedgerReason.ARCHIVE_TIME_LIMIT
        and event.get("path") == "second.zip"
        for event in result.ledger_events
    )


def test_virtual_inventory_covers_readable_and_failed_members(tmp_path: Path) -> None:
    path = tmp_path / "inventory.zip"
    path.write_bytes(
        _zip_bytes(
            {"readme.txt": b"ordinary", "payload.sh": b"target.sh"},
            link="payload.sh",
        )
    )

    result = inspect_nested_artifacts(tmp_path, [path.name])
    inventory = {item["path"]: item for item in result.artifact_inventory}

    assert set(inventory) == set(result.components)
    readable = inventory["inventory.zip!/readme.txt"]
    assert readable["content_kind"] == ContentKind.TEXT
    assert readable["disposition"] == ArtifactDisposition.ANALYZED
    failed = inventory["inventory.zip!/payload.sh"]
    assert failed["content_kind"] == ContentKind.OPAQUE
    assert failed["disposition"] == ArtifactDisposition.FAILED
    assert failed["reason"] == LedgerReason.ARCHIVE_LINK_MEMBER.value
    assert "inventory.zip!/payload.sh" not in result.raw_file_cache


def test_visible_document_concealment_survives_recursive_zip(tmp_path: Path) -> None:
    inner = _zip_bytes({"payload.sh": b"#!/bin/sh\necho nested\n"})
    outer_path = tmp_path / "nested.docx"
    _write_archive(outer_path, _document_members(**{"word/embedded.zip": inner}))

    context = build_context({"skill_path": str(tmp_path)})
    virtual_path = "nested.docx!/word/embedded.zip!/payload.sh"
    metadata = next(item for item in context["component_metadata"] if item["path"] == virtual_path)
    findings = _analyze_concealed_executables(context["component_metadata"])

    assert metadata["nested_path"] == "word/embedded.zip!/payload.sh"
    assert metadata["container_ancestry"] == ["docx", "zip"]
    assert metadata["concealment_reasons"] == ["document_container"]
    finding = next(item for item in findings if item.file == virtual_path)
    assert finding.severity == "HIGH"
    assert finding.evidence["nested_path"] == "word/embedded.zip!/payload.sh"
    assert finding.evidence["container_ancestry"] == ["docx", "zip"]


@pytest.mark.parametrize(
    ("member_name", "content"),
    [
        ("payload.ps1", b"Write-Host 'review'\n"),
        ("payload.pyc", b"\x42\x0d\x0d\x0a bytecode"),
        ("payload.exe", b"MZ executable"),
    ],
)
def test_document_execution_relevant_suffixes_emit_sc9(
    tmp_path: Path, member_name: str, content: bytes
) -> None:
    outer_path = tmp_path / "payloads.docx"
    _write_archive(outer_path, _document_members(**{f"word/{member_name}": content}))

    context = build_context({"skill_path": str(tmp_path)})
    findings = _analyze_concealed_executables(context["component_metadata"])

    finding = next(item for item in findings if item.file.endswith(member_name))
    assert finding.rule_id == "SC9"
    assert finding.severity == "HIGH"
    assert finding.evidence["concealment"] == "document_container"


def test_hidden_extensionless_executable_shebang_file_emits_sc9(tmp_path: Path) -> None:
    path = tmp_path / ".bootstrap"
    path.write_text("#!/bin/sh\necho local\n", encoding="utf-8")
    path.chmod(0o755)

    context = build_context({"skill_path": str(tmp_path)})
    metadata = next(item for item in context["component_metadata"] if item["path"] == path.name)
    findings = _analyze_concealed_executables(context["component_metadata"])

    assert metadata["executable"] is True
    assert metadata["concealed_executable"] is True
    assert findings[0].severity == "HIGH"


@pytest.mark.parametrize(
    ("member", "reason"),
    [
        ("../escape.sh", LedgerReason.ARCHIVE_UNSAFE_MEMBER_PATH),
        ("/absolute.sh", LedgerReason.ARCHIVE_UNSAFE_MEMBER_PATH),
        ("C:\\escape.sh", LedgerReason.ARCHIVE_UNSAFE_MEMBER_PATH),
    ],
)
def test_unsafe_member_paths_are_not_inventoried(
    tmp_path: Path, member: str, reason: LedgerReason
) -> None:
    path = tmp_path / "unsafe.zip"
    _write_archive(path, {member: b"#!/bin/sh\n"})

    result = inspect_nested_artifacts(tmp_path, [path.name])

    assert not result.components
    event = next(event for event in result.ledger_events if event.get("reason_code") == reason)
    assert event["outcome"] == LedgerOutcome.PARTIAL
    assert result.inventory_overrides[path.name] == (
        ArtifactDisposition.PARTIAL,
        reason.value,
    )


def test_duplicate_member_marks_canonical_member_partial(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("payload.txt", b"first")
        archive.writestr("payload.txt", b"second")

    result = inspect_nested_artifacts(tmp_path, [path.name])

    virtual_path = "ambiguous.zip!/payload.txt"
    artifact = next(item for item in result.artifact_inventory if item["path"] == virtual_path)
    assert artifact["disposition"] == ArtifactDisposition.PARTIAL
    assert artifact["reason"] == LedgerReason.ARCHIVE_AMBIGUOUS_MEMBER_PATH.value
    event = next(
        item
        for item in result.ledger_events
        if item.get("reason_code") == LedgerReason.ARCHIVE_AMBIGUOUS_MEMBER_PATH
    )
    assert event["outcome"] == LedgerOutcome.PARTIAL


def test_archive_link_member_is_not_followed(tmp_path: Path) -> None:
    path = tmp_path / "links.zip"
    path.write_bytes(_zip_bytes({"payload.sh": b"target.sh"}, link="payload.sh"))

    result = inspect_nested_artifacts(tmp_path, [path.name])

    assert "links.zip!/payload.sh" in result.components
    assert result.file_cache["links.zip!/payload.sh"] == "\x00"
    assert any(
        event.get("reason_code") == LedgerReason.ARCHIVE_LINK_MEMBER
        for event in result.ledger_events
    )


def test_malformed_zip_marks_inspection_incomplete(tmp_path: Path) -> None:
    path = tmp_path / "broken.txt"
    path.write_bytes(b"PK\x03\x04not-a-zip")

    result = inspect_nested_artifacts(tmp_path, [path.name])

    assert any(
        event.get("reason_code") == LedgerReason.ARCHIVE_MALFORMED for event in result.ledger_events
    )


def test_expected_document_with_incompatible_bytes_is_incomplete_and_local_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "broken.docx"
    path.write_bytes(b"not an office container")

    context = build_context({"skill_path": str(tmp_path)})

    assert path.name not in context["file_cache"]
    assert path.name in context["local_file_cache"]
    metadata = next(item for item in context["component_metadata"] if item["path"] == path.name)
    assert metadata["local_only"] is True
    artifact = next(item for item in context["artifact_inventory"] if item["path"] == path.name)
    assert artifact["disposition"] == ArtifactDisposition.PARTIAL
    assert artifact["reason"] == LedgerReason.ARCHIVE_FORMAT_MISMATCH.value
    assert any(
        event.get("reason_code") == LedgerReason.ARCHIVE_FORMAT_MISMATCH
        and event.get("outcome") == LedgerOutcome.PARTIAL
        and event.get("path") == path.name
        for event in context["inspection_ledger"]
    )


def test_unsupported_compression_is_not_reported_as_encryption(tmp_path: Path) -> None:
    path = tmp_path / "unsupported.zip"
    path.write_bytes(_with_unsupported_compression(_zip_bytes({"payload.sh": b"#!/bin/sh\n"})))

    result = inspect_nested_artifacts(tmp_path, [path.name])
    reasons = {event.get("reason_code") for event in result.ledger_events}

    assert LedgerReason.ARCHIVE_UNSUPPORTED_COMPRESSION in reasons
    assert LedgerReason.ARCHIVE_ENCRYPTED not in reasons


def test_truncated_nested_archive_retains_full_provenance(tmp_path: Path) -> None:
    path = tmp_path / "outer.zip"
    _write_archive(path, {"nested.zip": b"PK\x03\x04truncated"})

    result = inspect_nested_artifacts(tmp_path, [path.name])

    assert any(
        event.get("reason_code") == LedgerReason.ARCHIVE_TRUNCATED
        and event.get("path") == "outer.zip!/nested.zip"
        for event in result.ledger_events
    )


def test_encrypted_member_is_inventoried_but_not_read(tmp_path: Path) -> None:
    encoded = bytearray(_zip_bytes({"secret.sh": b"#!/bin/sh\n"}))
    local_header = encoded.find(b"PK\x03\x04")
    central_header = encoded.find(b"PK\x01\x02")
    assert local_header >= 0 and central_header >= 0
    for offset in (local_header + 6, central_header + 8):
        flags = int.from_bytes(encoded[offset : offset + 2], "little") | 0x1
        encoded[offset : offset + 2] = flags.to_bytes(2, "little")
    path = tmp_path / "encrypted.zip"
    path.write_bytes(bytes(encoded))

    result = inspect_nested_artifacts(tmp_path, [path.name])

    assert "encrypted.zip!/secret.sh" in result.components
    assert result.file_cache["encrypted.zip!/secret.sh"] == "\x00"
    assert any(
        event.get("reason_code") == LedgerReason.ARCHIVE_ENCRYPTED for event in result.ledger_events
    )


def test_compression_ratio_limit_is_cumulative_safety_boundary(tmp_path: Path) -> None:
    path = tmp_path / "compressed.zip"
    path.write_bytes(_zip_bytes({"large.txt": b"A" * 100_000}, compression=zipfile.ZIP_DEFLATED))

    result = inspect_nested_artifacts(tmp_path, [path.name])

    assert any(
        event.get("reason_code") == LedgerReason.ARCHIVE_COMPRESSION_RATIO
        for event in result.ledger_events
    )


def test_depth_member_size_and_time_limits_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import skillspector.nested_artifacts as nested

    deepest = _zip_bytes({"payload.sh": b"#!/bin/sh\n"})
    for index in range(4):
        deepest = _zip_bytes({f"level-{index}.bin": deepest})
    depth_path = tmp_path / "depth.zip"
    depth_path.write_bytes(deepest)
    depth_result = inspect_nested_artifacts(tmp_path, [depth_path.name])
    depth_event = next(
        event
        for event in depth_result.ledger_events
        if event.get("reason_code") == LedgerReason.ARCHIVE_DEPTH_LIMIT
    )
    assert depth_event["observed_depth"] == 4
    assert depth_event["limit_depth"] == 3

    monkeypatch.setattr(nested, "ARCHIVE_MAX_MEMBERS", 1)
    member_path = tmp_path / "members.zip"
    _write_archive(member_path, {"one.txt": b"1", "two.txt": b"2"})
    member_result = inspect_nested_artifacts(tmp_path, [member_path.name])
    member_event = next(
        event
        for event in member_result.ledger_events
        if event.get("reason_code") == LedgerReason.ARCHIVE_MEMBER_LIMIT
    )
    assert member_event["observed_artifacts"] == 2
    assert member_event["limit_artifacts"] == 1

    monkeypatch.setattr(nested, "ARCHIVE_MAX_MEMBERS", 1_000)
    monkeypatch.setattr(nested, "ARCHIVE_MAX_UNCOMPRESSED_BYTES", 3)
    size_path = tmp_path / "size.zip"
    _write_archive(size_path, {"four.txt": b"1234"})
    size_result = inspect_nested_artifacts(tmp_path, [size_path.name])
    assert any(
        event.get("reason_code") == LedgerReason.ARCHIVE_SIZE_LIMIT
        for event in size_result.ledger_events
    )

    monkeypatch.setattr(nested, "ARCHIVE_MAX_UNCOMPRESSED_BYTES", 25 * 1024 * 1024)
    ticks = iter((0.0, 6.0))
    time_result = inspect_nested_artifacts(tmp_path, [member_path.name], clock=lambda: next(ticks))
    time_event = next(
        event
        for event in time_result.ledger_events
        if event.get("reason_code") == LedgerReason.ARCHIVE_TIME_LIMIT
    )
    assert time_event["observed_seconds"] == 6.0
    assert time_event["limit_seconds"] == 5.0


def test_build_context_applies_outer_archive_limit_to_canonical_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import skillspector.nested_artifacts as nested

    (tmp_path / "SKILL.md").write_text("# Bounded archive", encoding="utf-8")
    _write_archive(tmp_path / "limited.zip", {"payload.txt": b"payload"})
    monkeypatch.setattr(nested, "ARCHIVE_MAX_MEMBERS", 0)

    context = build_context({"skill_path": str(tmp_path)})

    artifact = next(item for item in context["artifact_inventory"] if item["path"] == "limited.zip")
    assert artifact["disposition"] == ArtifactDisposition.PARTIAL
    assert artifact["reason"] == LedgerReason.ARCHIVE_MEMBER_LIMIT.value


def test_build_context_extracts_structured_context_from_nested_cache(tmp_path: Path) -> None:
    payload = json.dumps(
        [
            {"role": "system", "content": {"protocol": "AISOP V1"}},
            {
                "role": "user",
                "content": {
                    "aisop": {"main": "graph TD"},
                    "functions": {"nested_step": {"constraints": ["read-only"]}},
                },
            },
        ]
    ).encode()
    (tmp_path / "SKILL.md").write_text("# Nested workflow", encoding="utf-8")
    _write_archive(tmp_path / "workflow.zip", {"inside.aisop.json": payload})

    context = build_context({"skill_path": str(tmp_path)})

    assert context["structured_skill_context"]["workflow_nodes"] == ["nested_step"]
    assert "workflow.zip!/inside.aisop.json" in context["components"]


def test_member_limit_is_checked_before_sorting_attacker_controlled_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import skillspector.nested_artifacts as nested

    monkeypatch.setattr(nested, "ARCHIVE_MAX_MEMBERS", 1)
    path = tmp_path / "members.zip"
    _write_archive(path, {"two.txt": b"2", "one.txt": b"1"})
    zip_sort_calls = 0

    def guarded_sort(infos: list[zipfile.ZipInfo]) -> list[zipfile.ZipInfo]:
        nonlocal zip_sort_calls
        zip_sort_calls += 1
        return infos

    monkeypatch.setattr(nested, "_sorted_infos", guarded_sort)

    result = inspect_nested_artifacts(tmp_path, [path.name])

    assert zip_sort_calls == 0
    assert any(
        event.get("reason_code") == LedgerReason.ARCHIVE_MEMBER_LIMIT
        for event in result.ledger_events
    )


def test_reserved_virtual_delimiter_cannot_collide_with_recursive_provenance(
    tmp_path: Path,
) -> None:
    inner = _zip_bytes({"payload.sh": b"#!/bin/sh\n"})
    path = tmp_path / "collision.zip"
    _write_archive(
        path,
        {
            "evil": inner,
            "evil!/payload.sh": b"#!/bin/sh\necho impersonated\n",
        },
    )

    result = inspect_nested_artifacts(tmp_path, [path.name])

    assert result.components.count("collision.zip!/evil!/payload.sh") == 1
    assert result.file_cache["collision.zip!/evil!/payload.sh"] == "#!/bin/sh\n"
    assert any(
        event.get("reason_code") == LedgerReason.ARCHIVE_UNSAFE_MEMBER_PATH
        for event in result.ledger_events
    )
