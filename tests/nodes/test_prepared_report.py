"""Reusable report preparation and format serialization contracts."""

from __future__ import annotations

import importlib
import json

import pytest

from skillspector.models import Finding
from skillspector.nodes.report import prepare_report, report, serialize_report
from skillspector.sarif_models import validate_sarif_report

report_module = importlib.import_module("skillspector.nodes.report")


def test_one_prepared_snapshot_renders_every_format(monkeypatch):
    finding = Finding(
        rule_id="TEST-1",
        severity="HIGH",
        message="A finding",
        file="script.py",
        start_line=2,
    )
    state = {
        "findings": [finding],
        "manifest": {"name": "example"},
        "skill_path": "./example",
        "use_llm": False,
    }
    prepared = prepare_report(state)

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("Serialization repeated preparation/provider resolution")

    monkeypatch.setattr(report_module, "prepare_report", unexpected_preparation)
    monkeypatch.setattr(report_module, "is_llm_available", unexpected_preparation)
    monkeypatch.setattr(report_module, "partition_findings", unexpected_preparation)
    for output_format in ("terminal", "json", "markdown", "sarif"):
        rendered = serialize_report(prepared, output_format)
        assert rendered["risk_score"] == prepared.risk_score
        assert rendered["filtered_findings"] == prepared.reported_findings
        assert "A finding" in rendered["report_body"]
        if output_format == "json":
            payload = json.loads(rendered["report_body"])
            assert payload["issues"][0]["id"] == "TEST-1"
        elif output_format == "sarif":
            validate_sarif_report(json.loads(rendered["report_body"]))


@pytest.mark.parametrize("output_format", ["terminal", "json", "markdown", "sarif"])
def test_wrapper_keeps_legacy_public_fields(output_format):
    state = {"findings": [], "use_llm": False, "output_format": output_format}
    prepared = prepare_report(state)
    rendered = report(state)
    assert set(rendered) == {*prepared.public_fields, "report_body", "sarif_report"}
    assert isinstance(rendered["report_body"], str)
    validate_sarif_report(rendered["sarif_report"])
