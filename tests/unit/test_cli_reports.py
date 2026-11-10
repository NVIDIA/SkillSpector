# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Additional outputs reuse existing single and recursive scan results."""

import json
import os
import sys
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from skillspector import cli
from skillspector.file_output import write_text_no_follow
from skillspector.models import Finding
from skillspector.nodes.report import report
from skillspector.report_outputs import parse_reports
from skillspector.sarif_models import validate_sarif_report

runner = CliRunner()


def scan_result(path, format):
    state = {
        "skill_path": str(path),
        "manifest": {"name": "example", "description": "Example skill"},
        "findings": [
            Finding(
                rule_id="TEST",
                severity="LOW",
                message="Example finding",
                file="SKILL.md",
            )
        ],
        "use_llm": False,
        "llm_requested": False,
        "output_format": format.value,
    }
    return {**state, **report(state)}


@pytest.mark.parametrize("recursive", [False, True])
@pytest.mark.parametrize("primary", list(cli.FormatChoice))
def test_reports_scan_each_skill_once(tmp_path, monkeypatch, recursive, primary):
    root = tmp_path / "skills"
    root.mkdir()
    skills = [root / name for name in ("one", "two")] if recursive else [root]
    for skill in skills:
        skill.mkdir(exist_ok=True)
        (skill / "SKILL.md").write_text("# Example\n")
    scan = Mock(side_effect=lambda input_path, format, **kw: scan_result(input_path, format))
    monkeypatch.setattr(cli, "_scan_skill", scan)
    original_states = []
    render = cli._render_report

    def capture_render(state, format):
        original_states.append(state.copy())
        rendered = render(state, format)
        assert state == original_states[-1]
        return rendered

    monkeypatch.setattr(cli, "_render_report", capture_render)
    args = ["scan", str(root), "--no-llm", "-f", primary.value]
    if recursive:
        args.append("--recursive")
    destinations = {}
    for format in cli.FormatChoice:
        path = tmp_path / f"report.{format.value}"
        destinations[format] = path
        args.extend(["--report", f"{format.value}={path}"])
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert scan.call_count == len(skills)
    for path in destinations.values():
        assert path.read_text()
    payload = json.loads(destinations[cli.FormatChoice.json].read_text())
    if recursive:
        assert payload["multi_skill"] is True
        assert len(payload["skills"]) == 2
        assert all(child["issues"] for child in payload["skills"])
    else:
        assert payload["issues"][0]["explanation"] == "Example finding"
    validate_sarif_report(json.loads(destinations[cli.FormatChoice.sarif].read_text()))
    if primary in (cli.FormatChoice.json, cli.FormatChoice.sarif):
        json.loads(result.stdout)  # File notifications must not contaminate stdout.


@pytest.mark.parametrize("declaration", ["json", "html=out", "json=", "json=-"])
def test_invalid_report_fails_before_scan(tmp_path, monkeypatch, declaration):
    scan = Mock()
    monkeypatch.setattr(cli, "_scan_skill", scan)
    result = runner.invoke(cli.app, ["scan", str(tmp_path), "--report", declaration])
    assert result.exit_code == 2
    scan.assert_not_called()


def test_report_paths_are_distinct_and_preserve_equals(tmp_path):
    path = tmp_path / "report=one.json"
    assert parse_reports([f"JSON={path}"], ("json",)) == [("json", path)]
    with pytest.raises(ValueError, match="distinct"):
        parse_reports([f"json={path}"], ("json",), path)
    path.touch()
    alias = tmp_path / "alias"
    os.link(path, alias)
    with pytest.raises(ValueError, match="distinct"):
        parse_reports([f"json={path}", f"json={alias}"], ("json",))


def test_report_paths_use_writer_normalization(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "report.json"
    reports = parse_reports(["json=missing/../report.json"], ("json",))
    assert reports == [("json", path)]
    with pytest.raises(ValueError, match="distinct"):
        parse_reports(["json=missing/../report.json", "json=report.json"], ("json",))
    with pytest.raises(ValueError, match="distinct"):
        parse_reports(["json=report.json"], ("json",), tmp_path / "missing" / ".." / path.name)
    write_text_no_follow(reports[0][1], '{"ok": true}')
    assert json.loads(path.read_text()) == {"ok": True}
    assert not (tmp_path / "missing").exists()
    os.link(path, tmp_path / "alias.json")
    with pytest.raises(ValueError, match="distinct"):
        parse_reports(["json=missing/../alias.json"], ("json",), path)


def test_report_paths_retain_untrusted_symlinks(tmp_path):
    directory = tmp_path / "reports"
    directory.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    # Do not resolve child symlinks into paths that the no-follow writer permits.
    destination = alias / "report.json"
    reports = parse_reports([f"json={destination}"], ("json",))
    assert reports == [("json", destination)]
    with pytest.raises(ValueError, match="symlink or is not a directory"):
        write_text_no_follow(reports[0][1], '{"ok": true}')
    assert not list(directory.iterdir())


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize(
    "input_kind,filename",
    [
        ("directory", "SKILL.md"),
        ("directory", "helper.py"),
        ("file", "SKILL.md"),
        ("recursive", "SKILL.md"),
        ("recursive", "helper.py"),
    ],
)
def test_report_does_not_overwrite_input(tmp_path, monkeypatch, alias, input_kind, filename):
    skill = tmp_path / "one" if input_kind == "recursive" else tmp_path
    skill.mkdir(exist_ok=True)
    (skill / "SKILL.md").write_text("# Example\n")
    source = skill / filename
    original = b"# Keep me\r\n# preserved bytes: \xc3\xa4\n"
    source.write_bytes(original)
    destination = skill / "missing" / ".." / filename if alias else source
    assert not (skill / "missing").exists()
    scan = Mock(side_effect=lambda input_path, format, **kw: scan_result(input_path, format))
    monkeypatch.setattr(cli, "_scan_skill", scan)
    args = [
        "scan",
        str(source if input_kind == "file" else tmp_path),
        "--no-llm",
        "--report",
        f"json={destination}",
    ]
    if input_kind == "recursive":
        args.append("--recursive")
    result = runner.invoke(cli.app, args)
    assert source.read_bytes() == original
    assert result.exit_code == 2
    assert "--output points to the input file" in result.output or (
        "--report must not overwrite" in result.output
    )
    scan.assert_not_called()


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("filename", ["SKILL.md", "helper.py"])
def test_batch_report_does_not_overwrite_input(tmp_path, monkeypatch, capsys, alias, filename):
    from contrib.batch_scan import batch_scan

    skill = tmp_path / "one"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# Example\n")
    source = skill / filename
    original = b"# Keep me\r\n# preserved bytes: \xc3\xa4\n"
    source.write_bytes(original)
    destination = skill / "missing" / ".." / filename if alias else source
    assert not (skill / "missing").exists()
    scan = Mock(
        return_value=(
            {
                "skill": {"name": "example", "language": "en"},
                "risk_assessment": {"score": 0, "severity": "LOW"},
                "issues": [],
            },
            None,
            "one",
        )
    )
    pool = Mock(return_value=None)
    monkeypatch.setattr(batch_scan, "_scan_skill_bounded", scan)
    monkeypatch.setattr(batch_scan, "create_api_key_pool_from_env", pool)
    monkeypatch.setattr(
        sys,
        "argv",
        ["batch_scan", str(tmp_path), "--no-llm", "--report", f"json={destination}"],
    )
    with pytest.raises(SystemExit) as exc:
        try:
            batch_scan.main()
        finally:
            assert source.read_bytes() == original
    assert exc.value.code == 2
    assert "--report must not overwrite a batch input" in capsys.readouterr().err
    scan.assert_not_called()
    pool.assert_not_called()


@pytest.mark.parametrize("recursive", [False, True])
def test_cleanup_on_additional_write_failure(tmp_path, monkeypatch, recursive):
    skill = tmp_path / "one" if recursive else tmp_path
    skill.mkdir(exist_ok=True)
    (skill / "SKILL.md").write_text("# Example\n")
    state = scan_result(skill, cli.FormatChoice.json)
    monkeypatch.setattr(cli, "_scan_skill", Mock(return_value=state))
    cleanup = Mock()
    monkeypatch.setattr(cli, "cleanup_result", cleanup)
    args = [
        "scan",
        str(tmp_path),
        "--no-llm",
        "-f",
        "json",
        "--report",
        f"markdown={tmp_path / 'missing' / 'report.md'}",
    ]
    if recursive:
        args.append("--recursive")
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 2
    json.loads(result.stdout)
    cleanup.assert_called_once_with(state)


def test_findings_exit_after_all_reports(tmp_path, monkeypatch):
    (tmp_path / "SKILL.md").write_text("# Example\n")
    monkeypatch.setattr(
        cli,
        "_scan_skill",
        Mock(side_effect=lambda input_path, format, **kw: scan_result(input_path, format)),
    )
    output = tmp_path / "primary.json"
    markdown = tmp_path / "additional.md"
    result = runner.invoke(
        cli.app,
        [
            "scan",
            str(tmp_path),
            "--no-llm",
            "--fail-on-findings",
            "-f",
            "json",
            "-o",
            str(output),
            "--report",
            f"markdown={markdown}",
        ],
    )
    assert result.exit_code == 1
    assert json.loads(output.read_text())["issues"]
    assert markdown.read_text()


def test_additional_recursive_report_does_not_change_other_outputs(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    skill = root / "one"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Example\n")
    monkeypatch.setattr(
        cli,
        "_scan_skill",
        Mock(side_effect=lambda input_path, format, **kw: scan_result(input_path, format)),
    )
    render = cli._render_report
    monkeypatch.setattr(cli, "_MULTI_SKILL_MAX_REPORT_CHARACTERS", 20_000)
    monkeypatch.setattr(
        cli,
        "_render_report",
        lambda state, format: (
            {"report_body": "x" * 20_001}
            if format == cli.FormatChoice.markdown
            else render(state, format)
        ),
    )
    output = tmp_path / "primary.json"
    copy = tmp_path / "copy.json"
    result = runner.invoke(
        cli.app,
        [
            "scan",
            str(root),
            "--recursive",
            "--no-llm",
            "--fail-on-incomplete",
            "-f",
            "json",
            "-o",
            str(output),
            "--report",
            f"markdown={tmp_path / 'large.md'}",
            "--report",
            f"json={copy}",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text()) == json.loads(copy.read_text())


def test_registry_additional_json_reuses_scan(tmp_path, monkeypatch):
    source = tmp_path / "registry.json"
    source.write_text("{}")
    output = tmp_path / "report.json"
    scan = Mock(return_value={"findings": [], "risk_score": 0})
    monkeypatch.setattr(cli, "scan_registry", scan)
    args = ["scan", str(source), "--mcp-registry", "-f", "json", "--report"]
    result = runner.invoke(cli.app, [*args, f"json={output}"])
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text()) == json.loads(result.stdout)
    scan.assert_called_once()
    scan.reset_mock()
    result = runner.invoke(cli.app, [*args, f"markdown={tmp_path / 'report.md'}"])
    assert result.exit_code == 2
    scan.assert_not_called()


def test_batch_reports_reuse_results(tmp_path, monkeypatch, capsys):
    from contrib.batch_scan import batch_scan

    root = tmp_path / "skills"
    skill = root / "one"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Example\n")
    scan = Mock(
        return_value=(
            {
                "skill": {"name": "example", "language": "en"},
                "risk_assessment": {"score": 0, "severity": "LOW"},
                "issues": [],
            },
            None,
            "one",
        )
    )
    monkeypatch.setattr(batch_scan, "_scan_skill_bounded", scan)
    monkeypatch.setattr(batch_scan, "create_api_key_pool_from_env", lambda: None)
    markdown = tmp_path / "report.md"
    duplicate = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_scan",
            str(root),
            "--no-llm",
            "-f",
            "json",
            "--report",
            f"markdown={markdown}",
            "--report",
            f"json={duplicate}",
        ],
    )
    batch_scan.main()
    assert scan.call_count == 1
    assert markdown.read_text()
    assert json.loads(duplicate.read_text()) == json.loads(capsys.readouterr().out)
