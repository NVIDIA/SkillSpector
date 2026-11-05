# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declaration parsing and ordered output planning, not multi-report emission."""

from pathlib import Path
from unittest.mock import MagicMock

import click
import pytest
from typer.testing import CliRunner

from skillspector import cli
from skillspector.report_outputs import (
    FormatChoice,
    ReportOutput,
    build_report_outputs,
    parse_report_declarations,
)

runner = CliRunner()


@pytest.mark.parametrize("format_choice", list(FormatChoice))
@pytest.mark.parametrize("mixed_case", [False, True])
def test_parse_every_format(format_choice: FormatChoice, mixed_case: bool) -> None:
    name = format_choice.value
    if mixed_case:
        name = "".join(char.upper() if index % 2 else char for index, char in enumerate(name))
    assert parse_report_declarations([f"{name}=report.txt"]) == [
        ReportOutput(format=format_choice, path="report.txt")
    ]


def test_repeated_formats_and_embedded_equals_preserve_order_and_path_spelling() -> None:
    declarations = [
        "JsOn=./reports/result=v1.json",
        "sarif=first.sarif",
        "JSON=second.json",
        "markdown=a=b=c.md",
        "terminal==literal-name",
        "json= paths with spaces/über.json ",
        "json=./-",
    ]
    reports = parse_report_declarations(declarations)
    assert [entry.format for entry in reports] == [
        FormatChoice.json,
        FormatChoice.sarif,
        FormatChoice.json,
        FormatChoice.markdown,
        FormatChoice.terminal,
        FormatChoice.json,
        FormatChoice.json,
    ]
    assert [entry.path for entry in reports] == [
        declaration.split("=", 1)[1] for declaration in declarations
    ]
    assert not any(entry.is_primary for entry in reports)


@pytest.mark.parametrize("declarations", [None, [], ()])
def test_no_declarations_returns_empty_list(declarations) -> None:
    assert parse_report_declarations(declarations) == []


INVALID_DECLARATIONS = [
    "",
    "json",
    "json report.json",
    "=report.json",
    "=",
    "xml=report.xml",
    " json=report.json",
    "json =report.json",
    "json=",
    "JsOn=",
    "terminal=-",
    "json=-",
    "Markdown=-",
    "SARIF=-",
]


@pytest.mark.parametrize("declaration", INVALID_DECLARATIONS)
def test_invalid_declarations_raise_value_error(declaration: str) -> None:
    with pytest.raises(ValueError):
        parse_report_declarations(["json=valid.json", declaration])


@pytest.mark.parametrize("primary_format", list(FormatChoice))
@pytest.mark.parametrize("primary_output", [None, Path("primary.json")])
def test_primary_first_plan(primary_format: FormatChoice, primary_output: Path | None) -> None:
    reports = parse_report_declarations(
        ["sarif=first.sarif", "json=second.json", "json=third.json"]
    )
    original = reports.copy()
    entries = build_report_outputs(primary_format, primary_output, reports)
    assert entries[0] == ReportOutput(
        format=primary_format,
        path=str(primary_output) if primary_output is not None else None,
        is_primary=True,
    )
    assert entries[1:] == reports == original
    assert all(entry.path is not None and not entry.is_primary for entry in entries[1:])


def test_plan_does_not_preflight_or_deduplicate_destinations(tmp_path: Path) -> None:
    destination = tmp_path / "missing-parent" / "report.json"
    reports = parse_report_declarations([f"json={destination}", f"json={destination}"])
    entries = build_report_outputs(FormatChoice.json, destination, reports)
    assert len(entries) == 3
    assert [entry.path for entry in entries] == [str(destination)] * 3
    assert not destination.parent.exists()


@pytest.mark.parametrize("declaration", INVALID_DECLARATIONS)
@pytest.mark.parametrize(
    "mode_options",
    [[], ["--recursive"], ["--transitive"], ["--mcp-registry", "--format", "json"]],
)
def test_invalid_cli_declaration_invokes_no_analysis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    declaration: str,
    mode_options: list[str],
) -> None:
    analysis = MagicMock(side_effect=AssertionError("Analysis must not begin"))
    for name in ("_scan_skill", "_scan_multi_skill", "scan_registry", "detect_skills"):
        monkeypatch.setattr(cli, name, analysis)
    monkeypatch.setattr(cli.graph, "invoke", analysis)
    write = MagicMock(side_effect=AssertionError("Output must not begin"))
    monkeypatch.setattr(cli, "_write_result", write)
    monkeypatch.setattr(cli, "write_text_no_follow", write)

    result = runner.invoke(
        cli.app,
        [
            "scan",
            str(tmp_path),
            "--no-llm",
            *mode_options,
            "--report",
            "json=valid.json",
            "--report",
            declaration,
        ],
    )

    assert result.exit_code == 2, result.output
    assert "--report" in result.output
    analysis.assert_not_called()
    write.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_cli_rejects_missing_report_option_value(monkeypatch: pytest.MonkeyPatch) -> None:
    analysis = MagicMock()
    monkeypatch.setattr(cli, "_scan_skill", analysis)
    result = runner.invoke(cli.app, ["scan", "SKILL.md", "--report"])
    assert result.exit_code == 2
    analysis.assert_not_called()


@pytest.mark.parametrize("primary_options", [[], ["--format", "JsOn"]])
def test_cli_exposes_parsed_list_and_ordered_entries_with_additive_emission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, primary_options: list[str]
) -> None:
    source = tmp_path / "SKILL.md"
    source.write_text("# Safe skill\n", encoding="utf-8")
    declarations = [
        f"TeRmInAl={tmp_path}/terminal.txt",
        f"JsOn={tmp_path}/result=v1.json",
        f"MARKDOWN={tmp_path}/report.md",
        f"SaRiF={tmp_path}/report.sarif",
        f"json={tmp_path}/second.json",
    ]
    metadata = {}
    scan_kwargs = {}

    def scan_skill(**kwargs):
        metadata.update(click.get_current_context().meta)
        scan_kwargs.update(kwargs)
        return {"report_body": "Primary only", "risk_score": 0}

    monkeypatch.setattr(cli, "_scan_skill", scan_skill)
    write = MagicMock()
    monkeypatch.setattr(cli, "_write_result", write)
    args = ["scan", str(source), "--no-llm", *primary_options]
    for declaration in declarations:
        args.extend(["--report", declaration])
    result = runner.invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    expected_reports = parse_report_declarations(declarations)
    primary_format = FormatChoice.json if primary_options else FormatChoice.terminal
    assert metadata["reports"] == expected_reports
    assert metadata["report_outputs"] == build_report_outputs(
        primary_format, None, expected_reports
    )
    assert scan_kwargs["format"] is primary_format
    assert "reports" not in scan_kwargs
    assert "report_outputs" not in scan_kwargs
    write.assert_called_once()
    assert write.call_args.args[1:] == (None, primary_format)
    assert isinstance(write.call_args.args[0]["report_body"], str)
    assert {path.name for path in tmp_path.iterdir()} == {
        source.name,
        "terminal.txt",
        "result=v1.json",
        "report.md",
        "report.sarif",
        "second.json",
    }


@pytest.mark.parametrize("output_to_file", [False, True])
def test_no_report_preserves_legacy_primary_scan_and_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, output_to_file: bool
) -> None:
    source = tmp_path / "SKILL.md"
    source.write_text("# Safe skill\n", encoding="utf-8")
    output = tmp_path / "primary.json" if output_to_file else None
    metadata = {}

    def scan_skill(**kwargs):
        metadata.update(click.get_current_context().meta)
        assert kwargs["format"] is FormatChoice.json
        return {"report_body": '{"legacy": true}', "risk_score": 0}

    scan = MagicMock(side_effect=scan_skill)
    monkeypatch.setattr(cli, "_scan_skill", scan)
    args = ["scan", str(source), "--no-llm", "--format", "json"]
    if output is not None:
        args.extend(["--output", str(output)])
    result = runner.invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    scan.assert_called_once()
    assert metadata["reports"] == []
    assert metadata["report_outputs"] == build_report_outputs(FormatChoice.json, output, [])
    if output is not None:
        assert output.read_text(encoding="utf-8") == '{"legacy": true}'
        assert "Report saved to" in result.output
    else:
        assert result.stdout == '{"legacy": true}\n'
