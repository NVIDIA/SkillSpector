# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Permanent behavioral corpus for dependency-source trust-boundary changes."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from skillspector.artifacts import classify_artifact
from skillspector.dependency_source_types import (
    DependencyEcosystem,
    DependencySourceLimitationReason,
    DependencySourceOperation,
    DependencySourceScope,
    DependencySourceSpan,
    DependencySourceSurface,
    DependencyWorkBudget,
    DestinationStatus,
    SourceChange,
    SourceSpan,
    finding_from_source_change,
)
from skillspector.nested_artifacts import is_executable_content

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
PR1_FINDING_IDS = frozenset(
    {
        "cargo-replace-with-registry-table",
        "cargo-vendored-sources",
        "line-anchor-maven-url",
        "line-anchor-poetry-url",
        "maven-distribution-management",
        "npmrc-semicolon-inline-comment",
        "pip-conf-colon-delimiter",
        "pip-conf-continuation-drops-extra-urls",
        "pip-conf-multi-url-single-line",
        "pip-conf-multiline-continuation",
        "pipconf-colon-delimiter",
        "pyproject-uv-index-table",
        "uv-toml-index-table",
        "yarnrc-context-free-registry-key",
        "yarnrc-v1-scoped-quoted-key",
        "yarnrc-yaml-block-scalar",
        "yarnrc-yaml-explicit-alias",
        "yarnrc-yaml-flow-style",
        "yarnrc-yaml-quoted-key",
    }
)
TRANSITIONED_LIMITATION_ID = "markdown-nonstandard-filename-limitation"
PUBLIC_CONTROL_ID = re.compile(r"control-(?:[a-z][a-z0-9-]*|executable-[0-9a-f]{12})\Z")
PUBLIC_IMPORTED_CONTROL_ID = re.compile(r"control-executable-[0-9a-f]{12}\Z")


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
    assert len(FINDING_ROWS) == 85
    assert len(CONTROL_ROWS) == 165
    assert Counter(row["expected_outcome"] for row in FINDING_ROWS) == {
        "finding": 70,
        "inert": 12,
        "limitation": 3,
    }
    assert {row["id"] for row in FINDING_ROWS if row["lands_in"] == "PR-1"} == (PR1_FINDING_IDS)
    assert sum(row["lands_in"] == "PR-2" for row in FINDING_ROWS) == 66
    assert sum(row["lands_in"] == "PR-1" for row in CONTROL_ROWS) == 28
    assert sum(row["lands_in"] == "PR-2" for row in CONTROL_ROWS) == 137
    assert Counter(row["status"] for row in ALL_ROWS) == {"fixed": 250}
    imported_control_ids = [row["id"] for row in CONTROL_ROWS if row["lands_in"] == "PR-2"]
    assert imported_control_ids == sorted(imported_control_ids)
    assert all(
        PUBLIC_IMPORTED_CONTROL_ID.fullmatch(identifier) for identifier in imported_control_ids
    )
    for row in CONTROL_ROWS:
        if row["lands_in"] != "PR-2":
            continue
        canonical_files = json.dumps(
            row["files"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        expected_id = f"control-executable-{hashlib.sha256(canonical_files).hexdigest()[:12]}"
        assert row["id"] == expected_id
    assert not any(row["lands_in"] == "DEFERRED" for row in ALL_ROWS)
    transitioned = [row for row in FINDING_ROWS if row["id"] == TRANSITIONED_LIMITATION_ID]
    assert len(transitioned) == 1
    assert (
        transitioned[0]["lands_in"],
        transitioned[0]["status"],
        transitioned[0]["expected_outcome"],
    ) == ("PR-2", "fixed", "limitation")

    ids = [row["id"] for row in ALL_ROWS]
    assert len(ids) == len(set(ids))
    file_inputs: list[tuple[str, str]] = []
    for row in ALL_ROWS:
        allowed_fields = ROW_FIELDS | (
            {"expected_limitation"} if row["expected_outcome"] == "limitation" else set()
        )
        assert set(row) == allowed_fields
        assert not (_mapping_keys(row) & PROHIBITED_FIELDS)
        assert isinstance(row["id"], str) and row["id"]
        assert isinstance(row["status"], str) and row["status"] in STATUS_VALUES
        assert isinstance(row["lands_in"], str) and row["lands_in"] in OWNER_VALUES
        assert (
            isinstance(row["expected_outcome"], str) and row["expected_outcome"] in OUTCOME_VALUES
        )
        assert isinstance(row["files"], dict) and 1 <= len(row["files"]) <= 2
        for path, content in row["files"].items():
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
            path = expected["file"]
            assert path in row["files"]
            content = row["files"][path]
            physical_line_count = max(1, content.encode("utf-8").count(b"\n") + 1)
            assert 1 <= expected["start_line"] <= physical_line_count
            if "end_line" in expected:
                assert type(expected["end_line"]) is int
                assert expected["end_line"] > expected["start_line"]
                assert expected["end_line"] <= physical_line_count
            ecosystem = expected["ecosystem"]
            assert ecosystem in {item.value for item in DependencyEcosystem}
            projected = finding_from_source_change(
                SourceChange(
                    ecosystem=DependencyEcosystem(ecosystem),
                    surface=DependencySourceSurface(expected["surface"]),
                    operation=DependencySourceOperation(expected["operation"]),
                    scope=DependencySourceScope(expected["scope"]),
                    destination=expected["destination"],
                    destination_status=DestinationStatus(expected["destination_status"]),
                    span=SourceSpan(
                        path=path,
                        start_byte=0,
                        end_byte=0,
                        start_line=expected["start_line"],
                        end_line=expected.get("end_line", expected["start_line"]),
                    ),
                )
            )
            assert _normalized_finding(projected) == expected
            if expected["destination_status"] == "resolved":
                parsed_destination = urlsplit(expected["destination"])
                assert parsed_destination.username is None
                assert parsed_destination.password is None
                assert not parsed_destination.query
                assert not parsed_destination.fragment
                if parsed_destination.path not in {"", "/"}:
                    assert parsed_destination.path == "/REDACTED_PATH"
        if row["expected_outcome"] == "finding":
            assert row["expected_sc10"]
            assert "expected_limitation" not in row
        elif row["expected_outcome"] == "inert":
            assert row["expected_sc10"] == []
            assert "expected_limitation" not in row
        else:
            assert row["expected_sc10"] == []
            assert "expected_limitation" in row
        if row["expected_outcome"] == "limitation":
            assert isinstance(row["expected_limitation"], dict)
            assert set(row["expected_limitation"]) == {"reason", "path", "range"}
            assert isinstance(row["expected_limitation"]["reason"], str)
            assert row["expected_limitation"]["reason"]
            assert row["expected_limitation"]["reason"] in {
                item.value for item in DependencySourceLimitationReason
            }
            assert isinstance(row["expected_limitation"]["path"], str)
            assert row["expected_limitation"]["path"]
            path = row["expected_limitation"]["path"]
            assert path in row["files"]
            content = row["files"][path]
            physical_line_count = max(1, content.encode("utf-8").count(b"\n") + 1)
            limitation_range = row["expected_limitation"]["range"]
            assert isinstance(limitation_range, dict)
            assert set(limitation_range) == {"start_line", "end_line"}
            assert type(limitation_range["start_line"]) is int
            assert type(limitation_range["end_line"]) is int
            assert 1 <= limitation_range["start_line"] <= limitation_range["end_line"]
            assert limitation_range["end_line"] <= physical_line_count

    assert len(file_inputs) == len(set(file_inputs))
    assert all(PUBLIC_CONTROL_ID.fullmatch(row["id"]) for row in CONTROL_ROWS)


@pytest.mark.parametrize("row", BEHAVIOR_PARAMETERS)
def test_dependency_source_behavior(row: dict[str, Any]) -> None:
    try:
        from skillspector.dependency_sources import analyze_dependency_sources
    except ImportError as exc:
        pytest.fail(f"real dependency-source analyzer is unavailable: {exc}")

    files = row["files"]
    raw_files = {path: content.encode("utf-8") for path, content in files.items()}
    executable_paths = frozenset(
        DependencySourceSpan(path=path, start_line=1, end_line=1).path
        for path in sorted(raw_files)
        if is_executable_content(path, raw_files[path])
    )
    analysis = analyze_dependency_sources(
        components=sorted(files),
        local_file_cache=files,
        raw_file_cache=raw_files,
        artifact_inventory=[classify_artifact(path, raw_files[path]) for path in sorted(raw_files)],
        budget=DependencyWorkBudget(),
        executable_paths=executable_paths,
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
