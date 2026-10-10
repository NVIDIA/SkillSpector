# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Archive provenance remains available without changing reported member identity."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import cast

import pytest

from skillspector.models import Finding
from skillspector.nodes.analyzers.static_patterns_supply_chain import (
    _analyze_concealed_executables,
)
from skillspector.nodes.build_context import build_context
from skillspector.nodes.finalize_inspection_ledger import finalize_inspection_ledger
from skillspector.nodes.report import report
from skillspector.state import SkillspectorState

PROVENANCE_FIELDS = (
    "outer_path",
    "nested_path",
    "container_type",
    "container_ancestry",
    "container_depth",
)


@pytest.fixture(autouse=True)
def disable_provider_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "skillspector.nodes.report.is_llm_available", lambda **_: (False, "Disabled in test")
    )


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _json_report(context: dict[str, object], findings: list[Finding] | None = None) -> dict:
    state = cast(
        SkillspectorState,
        {**context, "findings": findings or [], "output_format": "json", "use_llm": False},
    )
    return json.loads(report(state)["report_body"])


def test_wheel_and_recursive_zip_keep_complete_member_identity(tmp_path: Path) -> None:
    (tmp_path / "package.whl").write_bytes(
        _zip_bytes({"package/__init__.py": b"VALUE = 1\n", "package/data.txt": b"data\n"})
    )
    (tmp_path / "outer.zip").write_bytes(
        _zip_bytes({"inner.zip": _zip_bytes({"run.sh": b"#!/bin/sh\necho hello\n"})})
    )
    context = build_context({"skill_path": str(tmp_path)})
    metadata = context["component_metadata"]
    source = {
        "source_url": "https://example.com/package.zip",
        "source_identity": "source-sha256:" + "a" * 64,
        "source_digest": "b" * 64,
    }
    for component in metadata:
        component.update(source)

    payload = _json_report(context)
    rows = {row["path"]: row for row in payload["components"]}

    assert len(payload["components"]) == len(metadata) == len(rows)
    assert rows.keys() == {component["path"] for component in metadata}
    for component in metadata:
        row = rows[component["path"]]
        for field in ("path", "type", "lines", "executable", "size_bytes"):
            assert row[field] == component[field]
        assert {field: row[field] for field in source} == source
    for path, outer, nested, depth in (
        ("package.whl!/package/__init__.py", "package.whl", "package/__init__.py", 1),
        ("package.whl!/package/data.txt", "package.whl", "package/data.txt", 1),
        ("outer.zip!/inner.zip", "outer.zip", "inner.zip", 1),
        ("outer.zip!/inner.zip!/run.sh", "outer.zip", "inner.zip!/run.sh", 2),
    ):
        row = rows[path]
        assert row["outer_path"] == outer
        assert row["nested_path"] == nested
        assert row["container_type"] == "zip"
        assert row["container_ancestry"] == ["zip"] * depth
        assert row["container_depth"] == depth
    assert rows["package.whl!/package/__init__.py"]["executable"] is True
    assert rows["outer.zip!/inner.zip!/run.sh"]["executable"] is True
    assert rows["package.whl!/package/data.txt"]["executable"] is False
    assert not any(field in rows["package.whl"] for field in PROVENANCE_FIELDS)


def test_concealed_member_keeps_security_finding_and_risk(tmp_path: Path) -> None:
    (tmp_path / ".hidden.zip").write_bytes(_zip_bytes({"run.sh": b"#!/bin/sh\necho hello\n"}))
    context = build_context({"skill_path": str(tmp_path)})
    findings = _analyze_concealed_executables(context["component_metadata"])
    finding = next(item for item in findings if item.rule_id == "SC9")
    original_fingerprint = finding.fingerprint()

    payload = _json_report(context, findings)
    issue = next(item for item in payload["issues"] if item["id"] == "SC9")
    member = next(row for row in payload["components"] if row["path"] == finding.file)

    assert issue["location"]["file"] == member["path"] == ".hidden.zip!/run.sh"
    assert member["outer_path"] == issue["evidence"]["outer_path"] == ".hidden.zip"
    assert member["nested_path"] == issue["evidence"]["nested_path"] == "run.sh"
    assert member["executable"] is True
    assert issue["severity"] == "HIGH"
    assert issue["match_fingerprint"] == original_fingerprint
    assert payload["risk_assessment"]["score"] > 0
    assert payload["risk_assessment"]["recommendation"] != "SAFE"


def test_truncated_nested_archive_preserves_incomplete_verdict(tmp_path: Path) -> None:
    (tmp_path / "outer.zip").write_bytes(_zip_bytes({"broken.zip": b"PK\x03\x04truncated"}))
    context = build_context({"skill_path": str(tmp_path), "use_llm": False})
    finalized = finalize_inspection_ledger(cast(SkillspectorState, {**context, "use_llm": False}))

    payload = _json_report({**context, **finalized})
    completeness = payload["analysis_completeness"]
    member = next(row for row in payload["components"] if row["path"] == "outer.zip!/broken.zip")

    assert member["outer_path"] == "outer.zip"
    assert member["nested_path"] == "broken.zip"
    assert completeness == finalized["analysis_completeness"]
    assert completeness["is_complete"] is False
    assert any(
        row["path"] == member["path"] and row["reason_code"] == "archive_truncated"
        for row in completeness["ledger_exceptions"]
    )
    assert payload["execution_successful"] == completeness["execution_successful"]
    assert payload["risk_assessment"]["recommendation"] != "SAFE"


def test_literal_archive_separator_in_physical_path_has_no_provenance(tmp_path: Path) -> None:
    directory = tmp_path / "notes!"
    directory.mkdir()
    (directory / "run.py").write_text("VALUE = 1\n", encoding="utf-8")
    context = build_context({"skill_path": str(tmp_path)})

    payload = _json_report(context)
    row = next(item for item in payload["components"] if item["path"] == "notes!/run.py")

    assert row["executable"] is True
    assert not any(field in row for field in PROVENANCE_FIELDS)


@pytest.mark.parametrize(
    ("outer_path", "nested_path"),
    [("other.zip", "run.py"), ("bundle.zip", "other.py"), ("", "run.py"), (None, "run.py")],
)
def test_inconsistent_provenance_is_not_exported(outer_path: str | None, nested_path: str) -> None:
    component = {
        "path": "bundle.zip!/run.py",
        "executable": True,
        "outer_path": outer_path,
        "nested_path": nested_path,
        "container_type": "zip",
        "container_ancestry": ["zip"],
        "container_depth": 1,
    }

    row = _json_report({"component_metadata": [component]})["components"][0]

    assert row["path"] == component["path"]
    assert row["executable"] is True
    assert not any(field in row for field in PROVENANCE_FIELDS)
