# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task 3.5: executable examples for report ordering and final exit policy."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from skillspector import cli
from skillspector.report_completeness import build_report_completeness
from skillspector.report_emission import emit_report_outputs
from skillspector.report_outputs import (
    FormatChoice,
    build_report_outputs,
    parse_report_declarations,
)
from skillspector.sarif_models import validate_sarif_report

EXAMPLE = (
    "skillspector scan ./my-skill/ --no-llm --format json "
    "--report sarif=report.sarif --report markdown=report.md --fail-on-incomplete"
)


def test_readme_subsection_contains_the_exercised_example_and_metadata():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    output_formats = readme.split("### Output Formats\n", 1)[1].split(
        "### Recursive Collection Scanning\n", 1
    )[0]
    section = output_formats.split("#### Report emission and exit policy\n", 1)[1]
    code = re.search(r"```bash\n(.*?)\n```", section, re.DOTALL)
    assert code is not None
    assert shlex.split(code.group(1).replace("\\\n", "")) == shlex.split(EXAMPLE)
    for field in (
        "analysis_completeness",
        "report_completeness",
        "analysisCompleteness",
        "reportCompleteness",
    ):
        assert f"`{field}`" in section


def _example_plan():
    """Parse the example with the real Click command and report parser."""
    group = get_command(cli.app)
    context = group.make_context("skillspector", [], resilient_parsing=True)
    command = group.get_command(context, "scan")
    assert command is not None
    with command.make_context("scan", shlex.split(EXAMPLE)[2:]) as scan_context:
        params = scan_context.params
        assert params["input_path"] == "./my-skill/"
        assert params["no_llm"] is True
        assert params["fail_on_incomplete"] is True
        return build_report_outputs(
            FormatChoice(params["format"]),
            params["output"],
            parse_report_declarations(params["report"]),
        )


def test_example_parses_as_json_stdout_then_sarif_and_markdown_files():
    entries = _example_plan()
    assert [(entry.format, entry.path, entry.is_primary) for entry in entries] == [
        (FormatChoice.json, None, True),
        (FormatChoice.sarif, "report.sarif", False),
        (FormatChoice.markdown, "report.md", False),
    ]


def test_example_coordinator_order_and_stderr_diagnostics(capsys):
    entries = _example_plan()
    events = []

    def render(format):
        events.append(("render", format.value))
        return {
            "report_body": json.dumps({"format": format.value}),
            "report_completeness": build_report_completeness(),
        }

    def primary(report, entry):
        assert entry is entries[0]
        events.append(("primary", "stdout"))
        print(report["report_body"])

    def additional(path, body):
        events.append(("additional", path))

    outcome = emit_report_outputs(entries, render, primary, additional_writer=additional)
    assert not outcome.output_failed and not outcome.report_incomplete
    assert events == [
        ("render", "json"),
        ("primary", "stdout"),
        ("render", "sarif"),
        ("additional", "report.sarif"),
        ("render", "markdown"),
        ("additional", "report.md"),
    ]
    streams = capsys.readouterr()
    assert json.loads(streams.out) == {"format": "json"}
    assert "Saved sarif report to 'report.sarif'." in streams.err
    assert "Saved markdown report to 'report.md'." in streams.err


def _install_example_scan(tmp_path, monkeypatch, *, analysis_complete=True, execution_failed=False):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "my-skill"
    source.mkdir()
    (source / "SKILL.md").write_text("---\nname: example\n---\nHello", encoding="utf-8")
    events = []

    def analyze(**kwargs):
        events.append(("analysis", kwargs["input_path"]))
        prepared = cli.prepare_report(
            {
                "manifest": {"name": "example"},
                "skill_path": str(source),
                "use_llm": False,
                "findings": [],
                "analysis_completeness": {
                    "is_complete": analysis_complete,
                    "coverage_percent": 100 if analysis_complete else 50,
                },
                "execution_successful": not execution_failed,
            }
        )
        return {**prepared.public_fields, "prepared_report": prepared}

    monkeypatch.setattr(cli, "_scan_skill", analyze)
    real_primary, real_write = cli._write_result, cli.write_text_no_follow

    def primary(report, output, format):
        events.append(("primary", format.value))
        return real_primary(report, output, format)

    def additional(path, body):
        events.append(("additional", Path(path).name))
        return real_write(path, body)

    monkeypatch.setattr(cli, "_write_result", primary)
    monkeypatch.setattr(cli, "write_text_no_follow", additional)
    return events


@pytest.mark.parametrize("strict", (False, True))
@pytest.mark.parametrize("omitted_format", ("json", "sarif", "markdown", None))
def test_any_report_omission_is_gated_only_after_the_last_attempt(
    tmp_path, monkeypatch, strict, omitted_format
):
    events = _install_example_scan(tmp_path, monkeypatch)
    original = cli.serialize_report

    def render(prepared, format):
        events.append(("render", format))
        result = original(prepared, format)
        if format == omitted_format:
            projection = build_report_completeness(
                omission_reasons=["serialized_output_limit"], omitted_records=1
            )
            result["report_completeness"] = projection
            if format == "json":
                payload = json.loads(result["report_body"])
                payload["report_completeness"] = projection
                result["report_body"] = json.dumps(payload)
            elif format == "sarif":
                payload = result["sarif_report"]
                payload["runs"][0]["invocations"][0]["properties"]["reportCompleteness"] = (
                    projection
                )
                result["report_body"] = json.dumps(payload)
        return result

    monkeypatch.setattr(cli, "serialize_report", render)
    args = shlex.split(EXAMPLE)[1:]
    if not strict:
        args.remove("--fail-on-incomplete")
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == (1 if strict and omitted_format is not None else 0), result.output
    assert events == [
        ("analysis", "./my-skill/"),
        ("render", "json"),
        ("primary", "json"),
        ("render", "sarif"),
        ("additional", "report.sarif"),
        ("render", "markdown"),
        ("additional", "report.md"),
    ]
    primary = json.loads(result.stdout)
    assert primary["analysis_completeness"]["is_complete"] is True
    assert primary["report_completeness"]["is_complete"] is (omitted_format != "json")
    sarif = json.loads((tmp_path / "report.sarif").read_text(encoding="utf-8"))
    validate_sarif_report(sarif)
    projection = sarif["runs"][0]["invocations"][0]["properties"]["reportCompleteness"]
    assert projection["is_complete"] is (omitted_format != "sarif")
    assert (tmp_path / "report.md").is_file()
    assert "Saved sarif report" in result.stderr
    assert "Saved markdown report" in result.stderr


@pytest.mark.parametrize("strict", (False, True))
def test_incomplete_analysis_with_complete_reports_uses_the_same_optional_gate(
    tmp_path, monkeypatch, strict
):
    events = _install_example_scan(tmp_path, monkeypatch, analysis_complete=False)
    args = shlex.split(EXAMPLE)[1:]
    if not strict:
        args.remove("--fail-on-incomplete")
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == (1 if strict else 0), result.output
    payload = json.loads(result.stdout)
    assert payload["analysis_completeness"]["is_complete"] is False
    assert payload["report_completeness"]["is_complete"] is True
    assert events[-1] == ("additional", "report.md")
    assert (tmp_path / "report.sarif").is_file()
    assert (tmp_path / "report.md").is_file()


@pytest.mark.parametrize("failure", ("render", "write", "execution"))
def test_error_exit_two_overrides_strict_gating_and_preserves_partial_success(
    tmp_path, monkeypatch, failure
):
    events = _install_example_scan(
        tmp_path, monkeypatch, analysis_complete=False, execution_failed=failure == "execution"
    )
    sarif = tmp_path / "report.sarif"
    sarif.write_text("previous report", encoding="utf-8")
    original, writer = cli.serialize_report, cli.write_text_no_follow

    def render(prepared, format):
        events.append(("render", format))
        if failure == "render" and format == "sarif":
            raise RuntimeError("private renderer detail")
        return original(prepared, format)

    def write(path, body):
        if failure == "write" and Path(path).name == "report.sarif":
            events.append(("write_failed", "report.sarif"))
            raise OSError("private writer detail")
        return writer(path, body)

    monkeypatch.setattr(cli, "serialize_report", render)
    monkeypatch.setattr(cli, "write_text_no_follow", write)
    result = CliRunner().invoke(cli.app, shlex.split(EXAMPLE)[1:])
    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["analysis_completeness"]["is_complete"] is False
    assert events[-1] == ("additional", "report.md")
    assert (tmp_path / "report.md").is_file()
    assert sum(event[0] == "analysis" for event in events) == 1
    if failure != "execution":
        assert sarif.read_text(encoding="utf-8") == "previous report"
        assert f"Failed to {failure} sarif report" in result.stderr
        assert "private renderer detail" not in result.output
        assert "private writer detail" not in result.output
