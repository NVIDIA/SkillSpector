# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capacity settings preserve independent bounds and complete work evidence."""

from __future__ import annotations

import gzip
import io
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from skillspector import inspection_ledger, nested_artifacts, state
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason, ledger_event
from skillspector.resource_limits import bounded_capacity


@pytest.mark.parametrize(
    ("value", "expected", "warns"),
    [
        (None, 100, False),
        ("", 100, False),
        (" \t ", 100, False),
        ("1", 1, False),
        (" 00145 ", 145, False),
        ("1000", 1000, False),
        ("0", 100, True),
        ("-1", 100, True),
        ("+1", 100, True),
        ("1.0", 100, True),
        ("1001", 100, True),
        ("１２", 100, True),
        ("1_000", 100, True),
        ("nan", 100, True),
        ("9" * 10_000, 100, True),
        ("secret\nforged log entry", 100, True),
    ],
)
def test_bounded_capacity_parsing_does_not_log_values(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    value: str | None,
    expected: int,
    warns: bool,
) -> None:
    name = "SKILLSPECTOR_TEST_CAPACITY"
    monkeypatch.delenv(name, raising=False)
    if value is not None:
        monkeypatch.setenv(name, value)
    with caplog.at_level("WARNING", logger="skillspector.resource_limits"):
        assert bounded_capacity(name, 100, 1000) == expected
    assert bool(caplog.records) is warns
    if warns:
        assert caplog.records[-1].getMessage() == (
            f"{name} must be a decimal integer from 1 to 1000; using default 100"
        )


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, [100000, 10000, 4096, 1000000, 16777216, 1000000, 128, 1048576, 4194304, 4194304]),
        (
            {
                "SKILLSPECTOR_MAX_INSPECTION_LEDGER_EVENTS": "250000",
                "SKILLSPECTOR_ARCHIVE_MAX_MEMBERS": "8192",
                "SKILLSPECTOR_ARCHIVE_MAX_MEMBER_BYTES": "2097152",
                "SKILLSPECTOR_MAX_PYTHON_AST_SOURCE_CHARS": "2000000",
                "SKILLSPECTOR_TP4_MAX_FILES": "1024",
                "SKILLSPECTOR_TP4_MAX_FILE_CODE_BYTES": "2097152",
                "SKILLSPECTOR_TP4_MAX_TOTAL_CODE_BYTES": "16777216",
                "SKILLSPECTOR_TP4_MAX_TOTAL_INPUT_BYTES": "33554432",
            },
            [250000, 10000, 8192, 2097152, 2097152, 2000000, 1024, 2097152, 16777216, 33554432],
        ),
        (
            {"SKILLSPECTOR_ARCHIVE_MAX_MEMBER_BYTES": " \t "},
            [100000, 10000, 4096, 1000000, 16777216, 1000000, 128, 1048576, 4194304, 4194304],
        ),
        (
            {"SKILLSPECTOR_ARCHIVE_MAX_MEMBER_BYTES": "0"},
            [100000, 10000, 4096, 1000000, 16777216, 1000000, 128, 1048576, 4194304, 4194304],
        ),
        (
            {"SKILLSPECTOR_ARCHIVE_MAX_MEMBER_BYTES": "16777217"},
            [100000, 10000, 4096, 1000000, 16777216, 1000000, 128, 1048576, 4194304, 4194304],
        ),
        (
            {"SKILLSPECTOR_ARCHIVE_MAX_MEMBER_BYTES": "1024"},
            [100000, 10000, 4096, 1024, 1024, 1000000, 128, 1048576, 4194304, 4194304],
        ),
        (
            {"SKILLSPECTOR_ARCHIVE_MAX_MEMBER_BYTES": "16777216"},
            [100000, 10000, 4096, 16777216, 16777216, 1000000, 128, 1048576, 4194304, 4194304],
        ),
    ],
)
def test_capacity_settings_are_bound_on_import(
    tmp_path: Path, overrides: dict[str, str], expected: list[int]
) -> None:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("SKILLSPECTOR_")
    }
    environment.update(overrides)
    script = (
        "import json; from skillspector import inspection_ledger as l, "
        "nested_artifacts as n, python_ast as p; "
        "from skillspector.nodes.analyzers import mcp_tool_poisoning as t; "
        "print(json.dumps([l.MAX_INSPECTION_LEDGER_EVENTS, l.MAX_FINDING_OUTPUT_RECORDS, "
        "n.ARCHIVE_MAX_MEMBERS, n.ARCHIVE_MAX_MEMBER_BYTES, n.ARCHIVE_MAX_WORKSHEET_BYTES, "
        "p.MAX_PYTHON_AST_SOURCE_CHARS, t.TP4_MAX_FILES, "
        "t.TP4_MAX_FILE_CODE_BYTES, t.TP4_MAX_TOTAL_CODE_BYTES, t.TP4_MAX_TOTAL_INPUT_BYTES]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert json.loads(result.stdout) == expected


def test_ordinary_multi_analyzer_work_keeps_more_than_ten_thousand_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(state, "MAX_INSPECTION_LEDGER_EVENTS", 100_000)
    rows = [
        ledger_event(outcome=LedgerOutcome.COMPLETED, phase="static", path=f"file-{i}.txt")
        for i in range(10_001)
    ]
    merged = state.merge_inspection_ledger(rows[:5000], rows[5000:])
    assert merged == rows
    assert inspection_ledger.MAX_FINDING_OUTPUT_RECORDS == 10_000


def test_configured_ledger_limit_retains_partial_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(state, "MAX_INSPECTION_LEDGER_EVENTS", 3)
    rows = [
        ledger_event(outcome=LedgerOutcome.COMPLETED, phase="static", path=f"file-{i}.txt")
        for i in range(5)
    ]
    merged = state.merge_inspection_ledger(rows[:2], rows[2:4])
    merged = state.merge_inspection_ledger(merged, rows[4:])
    assert len(merged) == 3
    assert merged[-1]["reason_code"] == LedgerReason.OUTPUT_LIMIT
    assert merged[-1]["observed_records"] == 5
    assert merged[-1]["limit_records"] == 3


def _archive(members: int) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for i in range(members):
            archive.writestr(f"file-{i}.txt", "ordinary text\n")
    return buffer.getvalue()


def test_archive_default_admits_more_than_one_thousand_small_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nested_artifacts, "ARCHIVE_MAX_MEMBERS", 4096)
    data = _archive(1025)
    result = nested_artifacts.inspect_nested_artifacts(
        tmp_path,
        ["bundle.zip"],
        raw_file_cache={"bundle.zip": data},
        clock=lambda: 0.0,
    )
    assert len(result.file_cache) == 1025
    assert len(result.inspected_containers["bundle.zip"]) == 1025
    assert not result.ledger_events


@pytest.mark.parametrize("constraint", ["members", "bytes"])
def test_archive_shared_remaining_budget_stays_stricter(tmp_path: Path, constraint: str) -> None:
    data = _archive(3)
    settings = {"max_members": 2} if constraint == "members" else {"max_uncompressed_bytes": 15}
    result = nested_artifacts.inspect_nested_artifacts(
        tmp_path, ["bundle.zip"], raw_file_cache={"bundle.zip": data}, **settings
    )
    reason = (
        LedgerReason.ARCHIVE_MEMBER_LIMIT
        if constraint == "members"
        else LedgerReason.ARCHIVE_SIZE_LIMIT
    )
    assert any(event.get("reason_code") == reason for event in result.ledger_events)
    assert "bundle.zip" not in result.inspected_containers
    if constraint == "members":
        assert len(result.file_cache) < 3
    else:
        assert result.uncompressed_bytes <= 15
        assert any(
            disposition[1] == LedgerReason.ARCHIVE_SIZE_LIMIT
            for disposition in result.inventory_overrides.values()
        )


@pytest.mark.parametrize("size", [15, 16, 17])
@pytest.mark.parametrize("archive_kind", ["zip", "tar", "gzip"])
def test_configured_archive_member_byte_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: int, archive_kind: str
) -> None:
    monkeypatch.setattr(nested_artifacts, "ARCHIVE_MAX_MEMBER_BYTES", 16)
    buffer = io.BytesIO()
    path = f"bundle.{archive_kind}"
    member = "source.txt"
    if archive_kind == "zip":
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(member, b"x" * size)
    elif archive_kind == "tar":
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            info = tarfile.TarInfo(member)
            info.size = size
            archive.addfile(info, io.BytesIO(b"x" * size))
    else:
        path, member = "source.txt.gz", "source.txt"
        buffer.write(gzip.compress(b"x" * size))
    result = nested_artifacts.inspect_nested_artifacts(
        tmp_path, [path], raw_file_cache={path: buffer.getvalue()}
    )
    if size <= 16:
        assert result.raw_file_cache[f"{path}!/{member}"] == b"x" * size
        assert not result.ledger_events
    else:
        assert path not in result.inspected_containers
        assert any(
            event.get("reason_code") == LedgerReason.ARCHIVE_MEMBER_SIZE_LIMIT
            for event in result.ledger_events
        )


def test_larger_member_setting_preserves_compression_ratio_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nested_artifacts, "ARCHIVE_MAX_MEMBER_BYTES", 16 * 1024 * 1024)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("source.txt", b"x" * 100_000)
    result = nested_artifacts.inspect_nested_artifacts(
        tmp_path, ["bundle.zip"], raw_file_cache={"bundle.zip": buffer.getvalue()}
    )
    assert "bundle.zip" not in result.inspected_containers
    assert any(
        event.get("reason_code") == LedgerReason.ARCHIVE_COMPRESSION_RATIO
        for event in result.ledger_events
    )
