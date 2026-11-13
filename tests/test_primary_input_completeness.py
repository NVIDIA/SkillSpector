# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unsupported requested content must never become a passive-asset exclusion."""

import bz2
import gzip
import io
import json
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import pytest

import skillspector.nodes.build_context as build_context_module
from skillspector.input_handler import InputHandler
from skillspector.mcp_server import run_scan

_SKILL = b"---\nname: primary-input-check\ndescription: Summarize supplied text.\n---\n# Notes\nSummarize the supplied text.\n"
_PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(32))


def _archive_bytes(members: dict[str, bytes], *, compressed: bool = False) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for path, content in members.items():
            member = tarfile.TarInfo(path)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    data = stream.getvalue() if members else bytes(2 * tarfile.BLOCKSIZE)
    return gzip.compress(data) if compressed else data


@pytest.mark.parametrize(
    ("name", "data", "directory"),
    [
        ("payload.dat", bytes([0x80, 0x81, 0x82, 0x83, 0, 0xFF]) * 20, False),
        ("SKILL.md", _SKILL.decode().encode("utf-16"), False),
        ("SKILL.md", _SKILL.decode().encode("utf-16"), True),
        ("skill.md", _SKILL.decode().encode("utf-16-be"), True),
        ("notes.md", bz2.compress(_SKILL), False),
        ("SKILL.md", _SKILL + b"Legacy caf\xe9 text.\n", True),
        ("notes.md", _SKILL + b"\xff", False),
    ],
    ids=[
        "opaque-file",
        "utf16-file",
        "utf16-primary",
        "utf16-no-bom",
        "renamed-bzip2",
        "lossy-primary",
        "lossy-explicit",
    ],
)
async def test_unsupported_primary_is_incomplete_and_not_install_safe(
    tmp_path: Path, name: str, data: bytes, directory: bool
) -> None:
    target = tmp_path / name
    target.write_bytes(data)

    result = await run_scan(str(tmp_path if directory else target), use_llm=False)

    assert result["safe_to_install"] is False
    completeness = result["analysis_completeness"]
    assert completeness["is_complete"] is False
    assert completeness["total_components"] >= 1
    assert completeness["coverage_percent"] < 100.0
    assert result["recommendation"] != "SAFE"
    exceptions = completeness["ledger_exceptions"]
    primary = next(
        item
        for item in exceptions
        if item["path"] == name and item["reason_code"] == "unsupported_primary_content"
    )
    assert primary["message"] == (
        "The requested file or primary instructions could not be interpreted. "
        "Provide UTF-8 text, a supported ZIP, TAR, or GZIP container, "
        "or an extracted directory instead."
    )
    assert not any(item["path"] == name for item in completeness["scope_exclusions"])


@pytest.mark.parametrize("referenced", [False, True])
async def test_passive_asset_keeps_existing_reference_policy(
    tmp_path: Path, referenced: bool
) -> None:
    instructions = _SKILL + (b"\nRead [image](image.png).\n" if referenced else b"")
    (tmp_path / "SKILL.md").write_bytes(instructions)
    (tmp_path / "image.png").write_bytes(_PNG)

    result = await run_scan(str(tmp_path), use_llm=False)

    assert result["analysis_completeness"]["is_complete"] is not referenced
    assert result["safe_to_install"] is not referenced
    assert not any(
        item["reason_code"] == "unsupported_primary_content"
        for item in result["analysis_completeness"]["ledger_exceptions"]
    )
    if not referenced:
        assert any(
            item["path"] == "image.png" and item["reason_code"] == "binary_content"
            for item in result["analysis_completeness"]["scope_exclusions"]
        )
    else:
        assert any(
            item["path"] == "image.png"
            and item["reason_code"] == "opaque_content"
            and item["message"] == "Artifact contents could not be fully interpreted."
            for item in result["analysis_completeness"]["ledger_exceptions"]
        )


@pytest.mark.parametrize("nested_archive", [False, True])
async def test_primary_failure_preserves_excluded_executable_evidence(
    tmp_path: Path, nested_archive: bool
) -> None:
    (tmp_path / "SKILL.md").write_bytes(_SKILL.decode().encode("utf-16"))
    excluded = tmp_path / "node_modules" / "example"
    excluded.mkdir(parents=True)
    if nested_archive:
        with zipfile.ZipFile(excluded / "bundle.zip", "w") as archive:
            archive.writestr("worker.py", "print('inert example')\n")
        executable_path = "node_modules/example/bundle.zip!/worker.py"
    else:
        (excluded / "worker.py").write_text("print('inert example')\n", encoding="utf-8")
        executable_path = "node_modules/example/worker.py"

    result = await run_scan(str(tmp_path), use_llm=False)

    completeness = result["analysis_completeness"]
    assert completeness["status"] == "failed"
    assert result["execution_successful"] is False
    assert result["safe_to_install"] is False
    exceptions = completeness["ledger_exceptions"]
    assert any(
        item["path"] == "SKILL.md" and item["reason_code"] == "unsupported_primary_content"
        for item in exceptions
    )
    assert any(
        item["path"] == executable_path and item["reason_code"] == "excluded_executable_content"
        for item in exceptions
    )


@pytest.mark.parametrize("layout", ["flat", "nested", "empty", "renamed"])
@pytest.mark.parametrize("archive_kind", ["zip", "tar", "gzip"])
async def test_supported_archives_remain_complete(
    tmp_path: Path, layout: str, archive_kind: str
) -> None:
    prefix = "skill/" if layout == "nested" else ""
    members = {} if layout == "empty" else {prefix + "SKILL.md": _SKILL, prefix + "image.png": _PNG}
    suffix = {"zip": ".zip", "tar": ".tar", "gzip": ".tar.gz"}[archive_kind]
    name = "bundle" + suffix
    if layout == "renamed":
        name = "bundle.dat" if archive_kind == "zip" else "notes.md"
    target = tmp_path / name
    if archive_kind == "zip":
        with zipfile.ZipFile(target, "w") as archive:
            for path, content in members.items():
                archive.writestr(path, content)
    else:
        target.write_bytes(_archive_bytes(members, compressed=archive_kind == "gzip"))

    result = await run_scan(str(target), use_llm=False)

    assert result["analysis_completeness"]["is_complete"] is True
    assert result["safe_to_install"] is True


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("member_type", [tarfile.FIFOTYPE, tarfile.GNUTYPE_SPARSE])
async def test_unsupported_tar_members_keep_incomplete_reference_evidence(
    tmp_path: Path, compressed: bool, member_type: bytes
) -> None:
    name = "bundle.tar.gz" if compressed else "bundle.tar"
    info = tarfile.TarInfo("unsupported")
    info.type = member_type
    data = info.tobuf(format=tarfile.GNU_FORMAT) + bytes(1024)
    (tmp_path / name).write_bytes(gzip.compress(data) if compressed else data)
    (tmp_path / "SKILL.md").write_bytes(_SKILL + f"\nRead [archive]({name}).\n".encode())

    result = await run_scan(str(tmp_path), use_llm=False)

    assert result["safe_to_install"] is False
    assert result["analysis_completeness"]["is_complete"] is False
    assert result["recommendation"] != "SAFE"
    assert any(
        item["path"].startswith(name)
        and item["reason_code"] == "archive_unsupported_member"
        and item["message"]
        == "Archive member uses a type or feature that is not supported for inspection."
        and item["outcome"] == "skipped"
        for item in result["analysis_completeness"]["ledger_exceptions"]
    )
    assert any(issue["id"] == "AE1" and issue["severity"] == "HIGH" for issue in result["findings"])


@pytest.mark.parametrize("extension", ["zip", "dat"])
@pytest.mark.parametrize("member", ["SKILL.md", "pkg/skill.md", "one/SKILL.md"])
async def test_zip_instruction_identity_survives_container_boundaries(
    tmp_path: Path, extension: str, member: str
) -> None:
    target = tmp_path / f"bundle.{extension}"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr(member, _SKILL.decode().encode("utf-16"))
        archive.writestr(str(PurePosixPath(member).parent / "image.png"), _PNG)
        if member.startswith("one/"):
            archive.writestr("two/README.md", "Summarize text.")

    result = await run_scan(str(target), use_llm=False)

    assert result["safe_to_install"] is False
    assert result["execution_successful"] is False
    completeness = result["analysis_completeness"]
    assert completeness["status"] == "failed"
    assert any(
        row["reason_code"] == "unsupported_primary_content"
        and row["path"].endswith(member.rsplit("/", 1)[-1])
        for row in completeness["ledger_exceptions"]
    )
    assert not any(
        row["path"].endswith(("SKILL.md", "skill.md")) for row in completeness["scope_exclusions"]
    )
    assert any(row["path"].endswith("image.png") for row in completeness["scope_exclusions"])


async def test_nested_directory_instructions_are_required(tmp_path: Path) -> None:
    target = tmp_path / "pkg" / "SKILL.md"
    target.parent.mkdir()
    target.write_bytes(_PNG)

    result = await run_scan(str(tmp_path), use_llm=False)

    assert result["safe_to_install"] is False
    assert any(
        row["path"] == "pkg/SKILL.md" and row["reason_code"] == "unsupported_primary_content"
        for row in result["analysis_completeness"]["ledger_exceptions"]
    )


@pytest.mark.parametrize("prefix", [b"", _SKILL])
@pytest.mark.parametrize("limit_name", ["MAX_ANALYZABLE_FILE_BYTES", "MAX_TOTAL_CACHED_BYTES"])
async def test_utf8_codepoint_split_by_cache_limit_stays_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: bytes, limit_name: str
) -> None:
    # The cached prefix ends after the first byte of a valid multibyte letter.
    monkeypatch.setattr(build_context_module, limit_name, len(prefix) + 1)
    (tmp_path / "SKILL.md").write_bytes(prefix + "é\n".encode())

    result = await run_scan(str(tmp_path), use_llm=False)

    assert result["safe_to_install"] is False
    assert result["execution_successful"] is True
    assert result["analysis_completeness"]["status"] == "partial"
    exceptions = result["analysis_completeness"]["ledger_exceptions"]
    expected_reason = (
        "size_limit" if limit_name == "MAX_ANALYZABLE_FILE_BYTES" else "total_bytes_limit"
    )
    assert any(row["reason_code"] == expected_reason for row in exceptions)
    assert not any(row["reason_code"] == "unsupported_primary_content" for row in exceptions)


async def test_explicit_text_with_binary_extension_preserves_source_path(tmp_path: Path) -> None:
    target = tmp_path / "instructions.png"
    target.write_bytes(_SKILL)

    result = await run_scan(str(target), use_llm=False)

    assert result["analysis_completeness"]["is_complete"] is True
    payload = json.loads(result["report"])
    assert payload["analysis_completeness"]["fully_inspected_files"] == 1


@pytest.mark.parametrize(
    "text",
    [
        b"BZh is the bzip2 file prefix. Summarize the supplied document.",
        b"BZh9 is a bzip2 header. Summarize the supplied document.",
        b"# Notes\n" + b" " * 249 + b"ustar denotes the TAR format.\n",
    ],
)
async def test_archive_magic_words_remain_analyzable_text(tmp_path: Path, text: bytes) -> None:
    target = tmp_path / "notes.md"
    target.write_bytes(text)

    result = await run_scan(str(target), use_llm=False)

    assert result["safe_to_install"] is True
    assert result["analysis_completeness"]["is_complete"] is True
    assert result["analysis_completeness"]["fully_inspected_files"] == 1
    assert not result["analysis_completeness"]["ledger_exceptions"]


@pytest.mark.parametrize("archive", [False, True])
async def test_downloaded_file_keeps_primary_identity_but_zip_members_are_passive(
    monkeypatch: pytest.MonkeyPatch, archive: bool
) -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr("SKILL.md", _SKILL)
        bundle.writestr("image.png", _PNG)
    content = stream.getvalue() if archive else _PNG
    headers = {"content-type": "application/zip" if archive else "application/octet-stream"}
    target = "https://raw.githubusercontent.com/example/skill/main/notes.md"
    monkeypatch.setattr(
        InputHandler,
        "_download_with_redirect_validation",
        lambda _self, _url: (headers, target, content),
    )

    result = await run_scan(target, use_llm=False)

    assert result["analysis_completeness"]["is_complete"] is archive
    assert result["safe_to_install"] is archive
    if not archive:
        assert any(
            item["path"] == "notes.md" and item["reason_code"] == "unsupported_primary_content"
            for item in result["analysis_completeness"]["ledger_exceptions"]
        )


@pytest.mark.parametrize("limit", [1, 12])
@pytest.mark.parametrize("cap", ["bundle", "workflow"])
@pytest.mark.parametrize(
    "content",
    [_SKILL.decode().encode("utf-16"), b"---\nname: [unterminated\n---\nHello\xff"],
    ids=["utf16", "invalid-utf8-and-manifest"],
)
async def test_primary_failure_survives_ledger_overflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int, cap: str, content: bytes
) -> None:
    from typer.testing import CliRunner

    import skillspector.state as state_module
    from skillspector.cli import app

    if cap == "bundle":
        monkeypatch.setattr(build_context_module, "MAX_BUNDLE_LEDGER_EVENTS", limit)
    else:
        monkeypatch.setattr(state_module, "MAX_INSPECTION_LEDGER_EVENTS", limit)
    (tmp_path / "SKILL.md").write_bytes(content)
    excluded = tmp_path / "node_modules" / "example"
    excluded.mkdir(parents=True)
    for index in range(10):
        (excluded / f"{index}.py").write_text("pass\n", encoding="utf-8")

    result = await run_scan(str(tmp_path), use_llm=False)

    assert result["execution_successful"] is False
    assert result["safe_to_install"] is False
    completeness = result["analysis_completeness"]
    assert completeness["status"] == "failed"
    assert any(
        row["path"] == "SKILL.md"
        and row["reason_code"] == "unsupported_primary_content"
        and row["fatal"]
        for row in completeness["ledger_exceptions"]
    )
    cli = CliRunner().invoke(app, ["scan", str(tmp_path), "--no-llm", "--format", "json"])
    assert cli.exit_code == 2, cli.output


@pytest.mark.parametrize("extension", ["zip", "dat"])
async def test_supported_zip_instruction_member_remains_complete(
    tmp_path: Path, extension: str
) -> None:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("notes.md", _SKILL)
    target = tmp_path / f"bundle.{extension}"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("pkg/SKILL.md", inner.getvalue())

    result = await run_scan(str(target), use_llm=False)

    assert result["execution_successful"] is True
    assert result["analysis_completeness"]["is_complete"] is True
    assert result["safe_to_install"] is True


@pytest.mark.parametrize("extension", ["docx", "xlsx", "pptx"])
async def test_archive_extension_does_not_hide_unsupported_primary_bytes(
    tmp_path: Path, extension: str
) -> None:
    target = tmp_path / f"notes.{extension}"
    target.write_bytes(_SKILL.decode().encode("utf-16"))

    result = await run_scan(str(target), use_llm=False)

    assert result["execution_successful"] is False
    assert result["analysis_completeness"]["status"] == "failed"
    assert any(
        row["reason_code"] == "unsupported_primary_content" and row["fatal"]
        for row in result["analysis_completeness"]["ledger_exceptions"]
    )


@pytest.mark.parametrize("failure", ["malformed", "depth_limit"])
async def test_recognized_nested_zip_keeps_archive_failure_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import skillspector.nested_artifacts as nested_module

    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("notes.md", _SKILL)
    content = inner.getvalue()
    if failure == "malformed":
        content = b"PK\x03\x04truncated"
        expected_reason = "archive_truncated"
    else:
        monkeypatch.setattr(nested_module, "ARCHIVE_MAX_DEPTH", 1)
        expected_reason = "archive_depth_limit"
    target = tmp_path / "bundle.dat"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("pkg/SKILL.md", content)

    result = await run_scan(str(target), use_llm=False)

    assert result["safe_to_install"] is False
    assert result["analysis_completeness"]["is_complete"] is False
    reasons = {row["reason_code"] for row in result["analysis_completeness"]["ledger_exceptions"]}
    assert expected_reason in reasons
    assert "unsupported_primary_content" not in reasons
