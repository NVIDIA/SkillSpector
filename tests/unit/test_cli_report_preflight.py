"""Multi-report preflight is early, non-destructive, and explicitly opt-in."""

from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

import skillspector.cli as cli


@pytest.mark.parametrize("primary_invalid", [False, True])
def test_missing_output_parent_rejected_before_scan(tmp_path, monkeypatch, primary_invalid):
    source = tmp_path / "SKILL.md"
    source.write_text("---\nname: example\n---\nHello")
    valid = tmp_path / "valid.json"
    invalid = tmp_path / "missing" / "invalid.json"
    analyzer = Mock(side_effect=AssertionError("analysis began"))
    monkeypatch.setattr(cli, "_scan_skill", analyzer)
    args = ["scan", str(source), "--no-llm", "--format", "json"]
    if primary_invalid:
        args += ["--output", str(invalid), "--report", f"json={valid}"]
    else:
        args += ["--output", str(valid), "--report", f"json={invalid}"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 2
    assert "output directory does not exist" in result.output
    analyzer.assert_not_called()
    assert not valid.exists()
    assert not invalid.parent.exists()


def test_legacy_missing_parent_is_validated_after_analysis(tmp_path, monkeypatch):
    source = tmp_path / "SKILL.md"
    source.write_text("---\nname: example\n---\nHello")
    output = tmp_path / "missing" / "report.json"
    analyzer = Mock(return_value={"report_body": "{}", "risk_score": 0})
    monkeypatch.setattr(cli, "_scan_skill", analyzer)
    result = CliRunner().invoke(
        cli.app, ["scan", str(source), "--no-llm", "--format", "json", "--output", str(output)]
    )
    assert result.exit_code == 2
    analyzer.assert_called_once()
    assert not output.parent.exists()


def test_additional_file_requires_secure_platform_before_analysis(tmp_path, monkeypatch):
    import skillspector.file_output as file_output

    source = tmp_path / "SKILL.md"
    source.write_text("Hello")
    analyzer = Mock(side_effect=AssertionError("analysis began"))
    monkeypatch.setattr(cli, "_scan_skill", analyzer)
    monkeypatch.setattr(file_output, "_SECURE_OUTPUT_SUPPORTED", False)
    result = CliRunner().invoke(
        cli.app, ["scan", str(source), "--no-llm", "--report", f"json={tmp_path / 'extra.json'}"]
    )
    assert result.exit_code == 2
    assert "unsupported" in result.output
    analyzer.assert_not_called()
