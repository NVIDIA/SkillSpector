# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task 5.4: serialized character bounds are private, post-collection limits."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from skillspector import cli
from skillspector.multi_skill import MultiSkillDetectionResult, SkillDirectory
from skillspector.nodes import report as report_node
from skillspector.nodes.report import prepare_report, serialize_report
from skillspector.recursive_reports import RecursiveChild, RecursiveReportCollection
from skillspector.report_completeness import build_report_completeness
from skillspector.report_emission import emit_report_outputs
from skillspector.report_outputs import FormatChoice, ReportOutput
from skillspector.sarif_models import validate_sarif_report
from skillspector.state import Finding

CHARACTER_LIMIT = 4_194_304
FORMATS = tuple(FormatChoice)


@pytest.fixture(autouse=True)
def fixed_report_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exclude per-render timestamps from projection-isolation comparisons."""

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            fixed = cls(2026, 11, 5, 12, 0, tzinfo=UTC)
            return fixed.astimezone(tz) if tz is not None else fixed.replace(tzinfo=None)

    monkeypatch.setattr(report_node, "datetime", FixedDatetime)


@pytest.fixture
def collection(tmp_path: Path) -> RecursiveReportCollection:
    prepared = prepare_report(
        {
            "manifest": {"name": "alpha"},
            "skill_path": str(tmp_path / "alpha"),
            "use_llm": False,
            "findings": [
                Finding(
                    rule_id="I-1",
                    severity="LOW",
                    message="Retained evidence",
                    file="agent.py",
                    start_line=1,
                )
            ],
            "analysis_completeness": {
                "is_complete": True,
                "status": "complete",
                "execution_successful": True,
                "coverage_percent": 100,
                "limitations": [],
            },
        }
    )
    return RecursiveReportCollection(
        children=[
            RecursiveChild(
                skill=SkillDirectory(tmp_path / "alpha", "alpha", "alpha"),
                prepared=prepared,
                policy={
                    "risk_score": prepared.risk_score,
                    "risk_severity": prepared.risk_severity,
                    "finding_count": 1,
                    "execution_successful": True,
                },
            )
        ],
        skill_count=1,
        scanned_skill_count=1,
        retained_public_records=1,
        max_score=prepared.risk_score,
        has_findings=True,
        analysis_completeness=dict(prepared.analysis_completeness),
        risk_assessment=cli._multi_skill_risk_assessment(
            prepared.risk_score, execution_failed=False, analysis_incomplete=False
        ),
        retention_completeness=build_report_completeness(),
    )


def _padded_serializer(
    monkeypatch: pytest.MonkeyPatch, selected: FormatChoice, padding: list[str]
) -> None:
    """Expand only one format after real preparation/serialization.

    Padding is projection-only so the canonical retention budget is untouched.
    Use actual JSON/SARIF layouts, schema validation, and combined rendering.
    """

    def padded(prepared, output_format):
        result = serialize_report(prepared, output_format)
        if output_format != selected.value:
            return result
        if selected is FormatChoice.json:
            payload = json.loads(result["report_body"])
            payload["format_padding"] = padding[0]
            result["report_body"] = json.dumps(payload, indent=2)
        elif selected is FormatChoice.sarif:
            payload = deepcopy(result["sarif_report"])
            payload["runs"][0].setdefault("properties", {})["format_padding"] = padding[0]
            result["sarif_report"] = payload
            result["report_body"] = json.dumps(payload, indent=2)
        else:
            result["report_body"] += padding[0]
        return result

    monkeypatch.setattr(report_node, "serialize_report", padded)


def _assert_projection(
    rendered: dict, format: FormatChoice, collection: RecursiveReportCollection
) -> None:
    """Check emitted metadata, not only the out-of-band completeness flag."""
    body = rendered["report_body"]
    if format is FormatChoice.json:
        payload = json.loads(body)
        assert payload["analysis_completeness"] == collection.analysis_completeness
        assert payload["max_risk_score"] == collection.max_score
        assert payload["risk_severity"] == collection.risk_assessment["severity"]
        assert payload["risk_recommendation"] == collection.risk_assessment["recommendation"]
        assert payload["execution_successful"] is True
        assert payload["report_completeness"] == rendered["report_completeness"]
        assert payload["skills_scanned"] == collection.scanned_skill_count
    elif format is FormatChoice.sarif:
        payload = json.loads(body)
        validate_sarif_report(payload)
        invocation = payload["runs"][-1]["invocations"][0]
        properties = invocation["properties"]
        assert invocation["executionSuccessful"] is True
        assert properties["analysisCompleteness"] == collection.analysis_completeness
        assert properties["reportCompleteness"] == rendered["report_completeness"]
        assert properties["riskAssessment"] == {
            "maxRiskScore": collection.max_score,
            "severity": collection.risk_assessment["severity"],
            "recommendation": collection.risk_assessment["recommendation"],
        }
    elif not rendered["report_completeness"]["is_complete"]:
        assert "Report completeness: incomplete" in body
        assert "serialized_output_limit" in body
        assert "Status: complete" in body
        assert f"Maximum score: {collection.max_score}/100" in body
        assert f"Severity: {collection.risk_assessment['severity']}" in body
        assert collection.risk_assessment["recommendation"].replace("_", " ") in body


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("delta", (-1, 0, 1), ids=("below", "exact", "overflow"))
def test_each_combined_format_has_the_exact_character_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    collection: RecursiveReportCollection,
    format: FormatChoice,
    delta: int,
) -> None:
    assert cli._MULTI_SKILL_MAX_REPORT_CHARACTERS == CHARACTER_LIMIT
    before = deepcopy(collection)
    padding = [""]
    _padded_serializer(monkeypatch, format, padding)
    target = CHARACTER_LIMIT + delta

    # Calibrate wrapper/JSON overhead with an enlarged temporary ceiling.
    # Enforcement below always uses the real, unchanged public ceiling.
    with monkeypatch.context() as calibration:
        calibration.setattr(cli, "_MULTI_SKILL_MAX_REPORT_CHARACTERS", 2 * CHARACTER_LIMIT)
        for _ in range(5):
            candidate = cli._render_recursive_report(collection, format)
            difference = target - len(candidate["report_body"])
            if difference == 0:
                break
            padding[0] = "x" * (len(padding[0]) + difference)
        assert len(candidate["report_body"]) == target

    rendered = cli._render_recursive_report(collection, format)
    assert len(rendered["report_body"]) <= CHARACTER_LIMIT
    projection = rendered["report_completeness"]
    assert projection["is_complete"] is (delta <= 0)
    if delta <= 0:
        assert len(rendered["report_body"]) == target
        assert projection["omitted_records"] == projection["omitted_skills"] == 0
    else:
        assert projection["omission_reasons"] == ["serialized_output_limit"]
        assert projection["omitted_records"] == collection.retained_public_records
        assert projection["omitted_skills"] == collection.scanned_skill_count
        assert "x" * 100 not in rendered["report_body"]
    _assert_projection(rendered, format, collection)
    assert collection == before


@pytest.mark.parametrize("limited_format", FORMATS)
def test_fallback_is_format_local_and_repeated_reports_remain_independent(
    monkeypatch: pytest.MonkeyPatch,
    collection: RecursiveReportCollection,
    limited_format: FormatChoice,
) -> None:
    before = deepcopy(collection)
    unaffected = {
        format: cli._render_recursive_report(collection, format)
        for format in FORMATS
        if format is not limited_format
    }
    _padded_serializer(monkeypatch, limited_format, ["x" * CHARACTER_LIMIT])
    limited = cli._render_recursive_report(collection, limited_format)
    assert limited["report_completeness"]["is_complete"] is False
    _assert_projection(limited, limited_format, collection)
    assert collection == before
    for format, expected in unaffected.items():
        assert cli._render_recursive_report(collection, format) == expected
        assert expected["report_completeness"]["is_complete"] is True
    assert cli._render_recursive_report(collection, limited_format) == limited

    # Even caller mutation of a returned projection must not poison snapshots
    # or a subsequent projection of the same format.
    limited["report_completeness"]["omission_reasons"].append("caller_mutation")
    rerendered = cli._render_recursive_report(collection, limited_format)
    assert rerendered["report_completeness"]["omission_reasons"] == ["serialized_output_limit"]
    assert collection == before


@pytest.mark.parametrize("format", (FormatChoice.terminal, FormatChoice.markdown))
def test_non_ascii_text_can_exceed_utf8_byte_ceiling_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    collection: RecursiveReportCollection,
    format: FormatChoice,
) -> None:
    _padded_serializer(monkeypatch, format, ["é" * (CHARACTER_LIMIT // 2 + 1)])
    rendered = cli._render_recursive_report(collection, format)
    body = rendered["report_body"]
    assert len(body) < CHARACTER_LIMIT < len(body.encode("utf-8"))
    assert rendered["report_completeness"]["is_complete"] is True
    cli._ensure_recursive_output_bound(body)


@pytest.mark.parametrize("format", (FormatChoice.json, FormatChoice.sarif))
def test_machine_format_counts_serialized_unicode_escapes_not_source_bytes(
    monkeypatch: pytest.MonkeyPatch,
    collection: RecursiveReportCollection,
    format: FormatChoice,
) -> None:
    padding = "é" * (CHARACTER_LIMIT // 6 + 1)
    assert len(padding) < len(padding.encode("utf-8")) < CHARACTER_LIMIT
    assert len(json.dumps(padding)) > CHARACTER_LIMIT
    _padded_serializer(monkeypatch, format, [padding])
    rendered = cli._render_recursive_report(collection, format)
    assert len(rendered["report_body"]) <= CHARACTER_LIMIT
    assert rendered["report_completeness"]["omission_reasons"] == ["serialized_output_limit"]
    _assert_projection(rendered, format, collection)


@pytest.mark.parametrize("format", FORMATS)
def test_recursive_writer_enforces_characters_not_utf8_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, format: FormatChoice
) -> None:
    # The writer is format-agnostic; payload shape belongs to the renderer.
    body = "é" * CHARACTER_LIMIT
    written = []
    monkeypatch.setattr(cli, "write_text_no_follow", lambda path, text: written.append(text))
    cli._write_recursive_report({"report_body": body}, tmp_path / "report", format, cli.err_console)
    assert written == [body]
    assert len(body.encode("utf-8")) == 2 * CHARACTER_LIMIT
    with pytest.raises(RuntimeError, match="output budget"):
        cli._write_recursive_report(
            {"report_body": body + "é"}, tmp_path / "report", format, cli.err_console
        )
    assert written == [body]


@pytest.mark.parametrize("limited_format", FORMATS)
@pytest.mark.parametrize("limited_is_primary", (True, False), ids=("primary", "additional"))
def test_collection_finishes_before_primary_or_additional_overflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    limited_format: FormatChoice,
    limited_is_primary: bool,
) -> None:
    skills = [SkillDirectory(tmp_path / name, name, name) for name in ("alpha", "beta", "gamma")]
    for skill in skills:
        skill.path.mkdir()
    events = []

    def scan_skill(**kwargs):
        name = Path(kwargs["input_path"]).name
        events.append(("scan", name))
        prepared = prepare_report(
            {
                "manifest": {"name": name},
                "skill_path": kwargs["input_path"],
                "use_llm": False,
                "findings": [],
                "analysis_completeness": {
                    "is_complete": True,
                    "status": "complete",
                    "execution_successful": True,
                    "coverage_percent": 100,
                    "limitations": [],
                },
            }
        )
        return {**prepared.public_fields, "prepared_report": prepared}

    monkeypatch.setattr(cli, "_scan_skill", scan_skill)
    _padded_serializer(monkeypatch, limited_format, ["x" * CHARACTER_LIMIT])
    padded = report_node.serialize_report

    def observed_serializer(prepared, output_format):
        assert [event for event in events if event[0] == "scan"] == [
            ("scan", skill.name) for skill in skills
        ]
        events.append(("serialize", output_format))
        return padded(prepared, output_format)

    monkeypatch.setattr(report_node, "serialize_report", observed_serializer)
    collection = cli._collect_recursive_reports(
        MultiSkillDetectionResult(is_multi_skill=True, skills=skills),
        limited_format,
        None,
        no_llm=True,
    )
    try:
        assert events == [("scan", skill.name) for skill in skills]
        assert [child.skill.name for child in collection.children] == [
            skill.name for skill in skills
        ]
        assert collection.scanned_skill_count == 3
        assert collection.unscanned_skill_count == collection.output_omitted_skill_count == 0
        assert collection.analysis_completeness["is_complete"] is True
        before = deepcopy(collection)
        other = next(format for format in FORMATS if format is not limited_format)
        order = (
            (limited_format, other, other) if limited_is_primary else (other, limited_format, other)
        )
        outputs = [
            ReportOutput(format, str(tmp_path / f"report-{index}"), index == 0)
            for index, format in enumerate(order)
        ]
        projections = []
        writes = []

        def renderer(format):
            rendered = cli._render_recursive_report(collection, format)
            _assert_projection(rendered, format, collection)
            projections.append((format, rendered["report_completeness"]))
            return rendered

        emission = emit_report_outputs(
            outputs,
            renderer,
            lambda rendered, entry: writes.append((entry.path, len(rendered["report_body"]))),
            additional_writer=lambda path, body: writes.append((path, len(body))),
            diagnostic=lambda message: None,
        )
        assert emission.output_failed is False
        assert emission.report_incomplete is True
        assert [path for path, _ in writes] == [entry.path for entry in outputs]
        assert all(length <= CHARACTER_LIMIT for _, length in writes)
        assert [format for format, _ in projections] == list(order)
        for format, projection in projections:
            assert projection["is_complete"] is (format is not limited_format)
            if format is limited_format:
                assert projection["omitted_skills"] == 3
        assert collection == before
        assert events[:3] == [("scan", skill.name) for skill in skills]
        assert sum(event[0] == "scan" for event in events) == 3
    finally:
        collection.release()
    assert collection.children == []
