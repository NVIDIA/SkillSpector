# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Permanent behavioral corpus for dependency-source trust-boundary changes."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from skillspector.artifacts import classify_artifact
from skillspector.dependency_source_types import DependencyWorkBudget

DATA_DIR = Path(__file__).with_name("data")
DATA_FILES = (DATA_DIR / "sc10_findings.json", DATA_DIR / "sc10_controls.json")
STATUS_VALUES = {"fixed", "unfixed", "deferred"}
OWNER_VALUES = {"PR-1", "PR-2", "DEFERRED"}
OUTCOME_VALUES = {"finding", "inert", "limitation"}
FINDING_FIELDS = {
    "severity",
    "ecosystem",
    "surface",
    "operation",
    "scope",
    "destination",
    "destination_status",
    "file",
    "start_line",
}
ROW_FIELDS = {"id", "status", "lands_in", "expected_outcome", "files", "expected_sc10"}
PROHIBITED_FIELDS = {
    "expect",
    "expect_sc10",
    "expected_prose",
    "family",
    "generated_from",
    "index",
    "input_note",
    "kind",
    "observed_today",
    "root_cause",
}


def _load_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in DATA_FILES]
    return documents[0], documents[1]


FINDING_DOCUMENT, CONTROL_DOCUMENT = _load_documents()
FINDING_ROWS = FINDING_DOCUMENT["rows"]
CONTROL_ROWS = CONTROL_DOCUMENT["rows"]
ALL_ROWS = FINDING_ROWS + CONTROL_ROWS


def _row_marks(row: dict[str, Any]) -> list[pytest.MarkDecorator]:
    owner_mark = {
        "PR-1": pytest.mark.sc10_pr1,
        "PR-2": pytest.mark.sc10_pr2,
        "DEFERRED": pytest.mark.sc10_deferred,
    }[row["lands_in"]]
    marks = [owner_mark]
    if row["status"] != "fixed" and os.getenv("SKILLSPECTOR_SC10_GAPS") != "enforce":
        marks.append(pytest.mark.xfail(strict=True, reason=f"SC10 gap: {row['id']}"))
    return marks


BEHAVIOR_PARAMETERS = [pytest.param(row, id=row["id"], marks=_row_marks(row)) for row in ALL_ROWS]


def _normalized_finding(finding: Any) -> dict[str, Any]:
    evidence = finding.evidence
    normalized = {
        "severity": finding.severity,
        "ecosystem": evidence["ecosystem"],
        "surface": evidence["surface"],
        "operation": evidence["operation"],
        "scope": evidence["scope"],
        "destination": evidence["destination"],
        "destination_status": evidence["destination_status"],
        "file": finding.file,
        "start_line": finding.start_line,
    }
    end_line = getattr(finding, "end_line", None)
    if end_line is not None and end_line != finding.start_line:
        normalized["end_line"] = end_line
    return normalized


def _normalized_limitation(limitation: Any) -> dict[str, Any]:
    return {
        "reason": getattr(limitation.reason, "value", limitation.reason),
        "path": limitation.path,
        "range": {
            "start_line": limitation.start_line,
            "end_line": limitation.end_line,
        },
    }


def _multiset(records: list[dict[str, Any]]) -> Counter[str]:
    return Counter(json.dumps(record, sort_keys=True) for record in records)


def _mapping_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            nested_key
            for nested_value in value.values()
            for nested_key in _mapping_keys(nested_value)
        }
    if isinstance(value, list):
        return {nested_key for item in value for nested_key in _mapping_keys(item)}
    return set()


def test_corpus_schema_and_self_checks() -> None:
    for document in (FINDING_DOCUMENT, CONTROL_DOCUMENT):
        assert set(document) == {"schema_version", "expected_row_count", "rows"}
        assert type(document["schema_version"]) is int and document["schema_version"] == 1
        assert type(document["expected_row_count"]) is int
        assert document["expected_row_count"] >= 1
        assert isinstance(document["rows"], list)
        assert len(document["rows"]) == document["expected_row_count"]

    assert FINDING_ROWS, "findings corpus must not be empty"
    assert CONTROL_ROWS, "controls corpus must not be empty"

    ids = [row["id"] for row in ALL_ROWS]
    assert len(ids) == len(set(ids))
    file_inputs: list[tuple[str, str]] = []
    for row in ALL_ROWS:
        allowed_fields = ROW_FIELDS | (
            {"expected_limitation"} if "expected_limitation" in row else set()
        )
        assert set(row) == allowed_fields
        assert not (_mapping_keys(row) & PROHIBITED_FIELDS)
        assert isinstance(row["id"], str) and row["id"]
        assert isinstance(row["status"], str) and row["status"] in STATUS_VALUES
        assert isinstance(row["lands_in"], str) and row["lands_in"] in OWNER_VALUES
        assert (
            isinstance(row["expected_outcome"], str) and row["expected_outcome"] in OUTCOME_VALUES
        )
        assert isinstance(row["files"], dict) and len(row["files"]) == 1
        path, content = next(iter(row["files"].items()))
        assert isinstance(path, str) and path
        assert isinstance(content, str)
        physical_line_count = max(1, content.encode("utf-8").count(b"\n") + 1)
        assert physical_line_count >= 1
        file_inputs.append((path, content))
        assert isinstance(row["expected_sc10"], list)
        for expected in row["expected_sc10"]:
            assert isinstance(expected, dict)
            assert set(expected) == FINDING_FIELDS or set(expected) == FINDING_FIELDS | {"end_line"}
            for field in FINDING_FIELDS - {"start_line"}:
                assert isinstance(expected[field], str) and expected[field]
            assert expected["severity"] == "HIGH"
            assert expected["destination_status"] in {"resolved", "unresolved"}
            assert type(expected["start_line"]) is int
            assert 1 <= expected["start_line"] <= physical_line_count
            assert expected["file"] == path
            if "end_line" in expected:
                assert type(expected["end_line"]) is int
                assert expected["end_line"] > expected["start_line"]
                assert expected["end_line"] <= physical_line_count
        if row["expected_outcome"] == "finding":
            assert row["expected_sc10"]
            assert "expected_limitation" not in row
        elif row["expected_outcome"] == "inert":
            assert row["expected_sc10"] == []
            assert "expected_limitation" not in row
        else:
            assert row["expected_sc10"] == []
            assert isinstance(row["expected_limitation"], dict)
            assert set(row["expected_limitation"]) == {"reason", "path", "range"}
            assert isinstance(row["expected_limitation"]["reason"], str)
            assert row["expected_limitation"]["reason"]
            assert row["expected_limitation"]["reason"] in {
                "dependency_source_parse_incomplete",
                "unscanned_executable_content",
            }
            assert isinstance(row["expected_limitation"]["path"], str)
            assert row["expected_limitation"]["path"]
            assert row["expected_limitation"]["path"] == path
            limitation_range = row["expected_limitation"]["range"]
            assert isinstance(limitation_range, dict)
            assert set(limitation_range) == {"start_line", "end_line"}
            assert type(limitation_range["start_line"]) is int
            assert type(limitation_range["end_line"]) is int
            assert 1 <= limitation_range["start_line"] <= limitation_range["end_line"]
            assert limitation_range["end_line"] <= physical_line_count

    assert len(file_inputs) == len(set(file_inputs))


@pytest.mark.parametrize("row", BEHAVIOR_PARAMETERS)
def test_dependency_source_behavior(row: dict[str, Any]) -> None:
    try:
        from skillspector.dependency_sources import analyze_dependency_sources
    except ImportError as exc:
        pytest.fail(f"real dependency-source analyzer is unavailable: {exc}")

    files = row["files"]
    raw_files = {path: content.encode("utf-8") for path, content in files.items()}
    analysis = analyze_dependency_sources(
        components=sorted(files),
        local_file_cache=files,
        raw_file_cache=raw_files,
        artifact_inventory=[classify_artifact(path, raw_files[path]) for path in sorted(raw_files)],
        budget=DependencyWorkBudget(),
    )
    findings = list(getattr(analysis, "findings", analysis))
    limitations = list(getattr(analysis, "limitations", []))
    actual_sc10 = [
        _normalized_finding(finding) for finding in findings if finding.rule_id == "SC10"
    ]
    assert len(actual_sc10) == len(row["expected_sc10"])
    assert _multiset(actual_sc10) == _multiset(row["expected_sc10"])

    expected_limitations = [row["expected_limitation"]] if "expected_limitation" in row else []
    actual_limitations = [_normalized_limitation(item) for item in limitations]
    assert len(actual_limitations) == len(expected_limitations)
    assert _multiset(actual_limitations) == _multiset(expected_limitations)
    assert row["status"] == "fixed", "unimplemented corpus rows remain explicit red gates"
