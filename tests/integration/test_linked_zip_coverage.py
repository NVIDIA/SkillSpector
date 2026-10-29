# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Linked archives retain complete inspection only while their members do."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from skillspector.graph import graph
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason, ledger_event
from skillspector.nodes.finalize_inspection_ledger import finalize_inspection_ledger

pytestmark = pytest.mark.integration


def _zip(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for path, content in members.items():
            archive.writestr(path, content)
    return stream.getvalue()


def _scan(tmp_path: Path, data: bytes) -> dict:
    (tmp_path / "SKILL.md").write_text(
        "---\nname: bundled-guide\ndescription: Read bundled documentation.\n---\n"
        "# Guide\nRead [the guide](guide.zip).\n"
    )
    (tmp_path / "guide.zip").write_bytes(data)
    return graph.invoke({"skill_path": str(tmp_path), "use_llm": False, "output_format": "json"})


@pytest.mark.parametrize("layout", ["plain", "nested", "empty", "passive"])
def test_linked_zip_inspection_is_complete(tmp_path: Path, layout: str) -> None:
    members = {"note.md": b"# Ordinary documentation\n"}
    if layout == "nested":
        members = {"inner.zip": _zip(members)}
    elif layout == "empty":
        members = {}
    elif layout == "passive":
        members = {"image.png": b"\x89PNG\r\n\x1a\n\x00"}
    result = _scan(tmp_path, _zip(members))
    report = json.loads(result["report_body"])
    assert report["analysis_completeness"]["is_complete"] is True
    assert report["analysis_completeness"]["coverage_percent"] == 100
    assert not any(issue["id"] == "AE1" for issue in report["issues"])
    assert all(
        artifact["disposition"] == "analyzed"
        for artifact in result["artifact_inventory"]
        if "inspected_members" in artifact
    )


@pytest.mark.parametrize("gap", ["late_failure", "missing_member"])
def test_member_gaps_propagate_through_nested_containers(tmp_path: Path, gap: str) -> None:
    result = _scan(tmp_path, _zip({"inner.zip": _zip({"note.md": b"Ordinary note.\n"})}))
    member = "guide.zip!/inner.zip!/note.md"
    if gap == "late_failure":
        result["inspection_ledger"].append(
            ledger_event(
                outcome=LedgerOutcome.PARTIAL,
                phase="static",
                path=member,
                reason=LedgerReason.RUNTIME_LIMIT,
            )
        )
    else:
        result["artifact_inventory"] = [
            row for row in result["artifact_inventory"] if row["path"] != member
        ]
    finalized = finalize_inspection_ledger(result)
    containers = {
        row["path"]: row["disposition"]
        for row in finalized["artifact_inventory"]
        if "inspected_members" in row
    }
    assert containers == {"guide.zip": "partial", "guide.zip!/inner.zip": "partial"}
    assert any(finding.rule_id == "AE1" for finding in finalized["findings"])
    assert finalized["analysis_completeness"]["is_complete"] is False


@pytest.mark.parametrize("data", [b"PK\x03\x04truncated", _zip({"../note.md": b"Note"})])
def test_failed_archive_inspection_retains_ae1(tmp_path: Path, data: bytes) -> None:
    result = _scan(tmp_path, data)
    assert any(finding.rule_id == "AE1" for finding in result["findings"])
    assert result["analysis_completeness"]["is_complete"] is False
    assert not any("inspected_members" in row for row in result["artifact_inventory"])
