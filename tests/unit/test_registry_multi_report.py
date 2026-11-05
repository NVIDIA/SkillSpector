"""Registry JSON reports use one completed registry result and the ordered coordinator."""

import json
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from skillspector import cli


@pytest.mark.parametrize("primary_file", [False, True])
def test_repeated_json_outputs_share_one_registry_scan(tmp_path, monkeypatch, primary_file):
    source = tmp_path / "registry.json"
    source.write_text("{}")
    scanner = Mock(return_value={"findings": [], "risk_score": 0})
    monkeypatch.setattr(cli, "scan_registry", scanner)
    extra, repeated = tmp_path / "extra.json", tmp_path / "repeated.json"
    args = [
        "scan",
        str(source),
        "--mcp-registry",
        "--format",
        "json",
        "--report",
        f"json={extra}",
        "--report",
        f"JsOn={repeated}",
    ]
    primary = tmp_path / "primary.json"
    if primary_file:
        args += ["--output", str(primary)]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    scanner.assert_called_once_with(str(source))
    payload = json.loads(primary.read_text() if primary_file else result.stdout)
    assert payload == json.loads(extra.read_text()) == json.loads(repeated.read_text())
    assert payload["report_completeness"]["is_complete"]
    assert result.stderr.count("Saved json report") == 2
    if primary_file:
        assert "Report saved to" in result.stdout


@pytest.mark.parametrize(
    "format_options",
    [
        [],
        ["--format", "sarif"],
        ["--format", "json", "--report", "terminal=bad.txt"],
        ["--format", "json", "--report", "sarif=bad.sarif"],
        ["--format", "json", "--report", "markdown=bad.md"],
    ],
)
def test_registry_rejects_primary_and_additional_formats_before_scan(
    tmp_path, monkeypatch, format_options
):
    source = tmp_path / "registry.json"
    source.write_text("{}")
    scanner = Mock()
    monkeypatch.setattr(cli, "scan_registry", scanner)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cli.app,
        [
            "scan",
            str(source),
            "--mcp-registry",
            *format_options,
            "--report",
            f"json={tmp_path / 'extra.json'}",
        ],
    )
    assert result.exit_code == 2
    assert "only --format json" in result.output
    scanner.assert_not_called()
    assert not (tmp_path / "extra.json").exists()


def test_registry_primary_write_failure_attempts_remaining_outputs(tmp_path, monkeypatch):
    source = tmp_path / "registry.json"
    source.write_text("{}")
    scanner = Mock(return_value={"findings": [{"rule": "R"}], "risk_score": 99})
    monkeypatch.setattr(cli, "scan_registry", scanner)
    primary, extra = tmp_path / "primary.json", tmp_path / "extra.json"
    primary.write_text("unchanged")
    real_write = cli.write_text_no_follow

    def write(path, body):
        if str(path) == str(primary):
            raise PermissionError("race")
        real_write(path, body)

    monkeypatch.setattr(cli, "write_text_no_follow", write)
    result = CliRunner().invoke(
        cli.app,
        [
            "scan",
            str(source),
            "--mcp-registry",
            "--format",
            "json",
            "--output",
            str(primary),
            "--report",
            f"json={extra}",
        ],
    )
    assert result.exit_code == 2
    assert primary.read_text() == "unchanged"
    assert json.loads(extra.read_text())["risk_score"] == 99
    scanner.assert_called_once()
