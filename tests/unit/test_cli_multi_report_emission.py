"""Ordinary CLI reports share one snapshot and defer every policy until emission."""

import json
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from skillspector import cli
from skillspector.nodes.report import report_rendering_is_deferred
from skillspector.report_completeness import build_report_completeness
from skillspector.sarif_models import validate_sarif_report


def install_scan(tmp_path, monkeypatch, **policy):
    source = tmp_path / "SKILL.md"
    source.write_text("---\nname: example\n---\nHello")

    def scan(**kwargs):
        assert report_rendering_is_deferred()
        prepared = cli.prepare_report(
            {
                "skill_path": str(source),
                "manifest": {"name": "example"},
                "use_llm": False,
                "analysis_completeness": {"is_complete": True, "coverage_percent": 100},
                **policy,
            }
        )
        return {**prepared.public_fields, "prepared_report": prepared, **policy}

    analyzer = Mock(side_effect=scan)
    monkeypatch.setattr(cli, "_scan_skill", analyzer)
    return source, analyzer


def test_one_scan_valid_ordered_reports_clean_stdout(tmp_path, monkeypatch):
    source, analyzer = install_scan(tmp_path, monkeypatch)
    events = []
    real = cli.serialize_report

    def render(prepared, choice):
        events.append(choice)
        return real(prepared, choice)

    monkeypatch.setattr(cli, "serialize_report", render)
    sarif, second = tmp_path / "extra.sarif", tmp_path / "second.json"
    result = CliRunner().invoke(
        cli.app,
        [
            "scan",
            str(source),
            "--no-llm",
            "--format",
            "json",
            "--report",
            f"sarif={sarif}",
            "--report",
            f"json={second}",
        ],
    )
    assert result.exit_code == 0, result.output
    analyzer.assert_called_once()
    assert events == ["json", "sarif", "json"]
    first_payload, second_payload = json.loads(result.stdout), json.loads(second.read_text())
    # Existing serializers stamp each report at emission, not at graph completion.
    first_payload["skill"].pop("scanned_at")
    second_payload["skill"].pop("scanned_at")
    assert first_payload == second_payload
    validate_sarif_report(json.loads(sarif.read_text()))
    assert "Saved sarif report" in result.stderr


@pytest.mark.parametrize("primary_failure", [False, True])
def test_renderer_failure_keeps_destination_and_attempts_later_reports(
    tmp_path, monkeypatch, primary_failure
):
    source, analyzer = install_scan(tmp_path, monkeypatch, risk_score=99)
    bad, later = tmp_path / "bad.sarif", tmp_path / "later.md"
    bad.write_text("unchanged")
    original = cli.serialize_report
    events = []

    def render(prepared, choice):
        events.append(choice)
        if choice == "sarif":
            raise RuntimeError("private provider details")
        return original(prepared, choice)

    monkeypatch.setattr(cli, "serialize_report", render)
    args = ["scan", str(source), "--no-llm"]
    if primary_failure:
        args += ["--format", "sarif", "--output", str(bad)]
    else:
        args += ["--format", "json", "--report", f"sarif={bad}"]
    args += ["--report", f"markdown={later}"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 2
    assert events[-1] == "markdown"
    assert later.exists()
    assert bad.read_text() == "unchanged"
    assert "private provider details" not in result.output
    analyzer.assert_called_once()


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("execution_failed", [False, True])
def test_additional_omission_gate_runs_after_all_attempts(
    tmp_path, monkeypatch, strict, execution_failed
):
    source, _ = install_scan(tmp_path, monkeypatch, execution_successful=not execution_failed)
    original = cli.serialize_report
    events = []

    def render(prepared, choice):
        events.append(choice)
        rendered = original(prepared, choice)
        if choice == "sarif":
            rendered["report_completeness"] = build_report_completeness(
                omission_reasons=["output_character_limit"]
            )
        return rendered

    monkeypatch.setattr(cli, "serialize_report", render)
    sarif, md = tmp_path / "extra.sarif", tmp_path / "extra.md"
    args = [
        "scan",
        str(source),
        "--no-llm",
        "--format",
        "json",
        "--report",
        f"sarif={sarif}",
        "--report",
        f"markdown={md}",
    ]
    if strict:
        args += ["--fail-on-incomplete"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == (2 if execution_failed else 1 if strict else 0), result.output
    assert events == ["json", "sarif", "markdown"]
    assert sarif.exists() and md.exists()


def test_legacy_single_report_omission_is_strictly_gated(tmp_path, monkeypatch):
    source = tmp_path / "SKILL.md"
    source.write_text("Hello")
    monkeypatch.setattr(
        cli,
        "_scan_skill",
        Mock(
            return_value={
                "report_body": "{}",
                "risk_score": 0,
                "report_completeness": {"is_complete": False},
            }
        ),
    )
    result = CliRunner().invoke(
        cli.app, ["scan", str(source), "--no-llm", "--format", "json", "--fail-on-incomplete"]
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {}
