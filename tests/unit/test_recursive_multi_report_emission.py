"""Recursive output failures and omissions cannot stop format-neutral collection."""

import importlib
import json
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from skillspector import cli
from skillspector.sarif_models import validate_sarif_report

report_module = importlib.import_module("skillspector.nodes.report")


def install_children(tmp_path, monkeypatch, *, execution_failed=False):
    root = tmp_path / "collection"
    for name in ("a", "b", "c"):
        child = root / name
        child.mkdir(parents=True)
        (child / "SKILL.md").write_text(f"---\nname: {name}\n---\nHello")

    def scan(**kwargs):
        from pathlib import Path

        prepared = cli.prepare_report(
            {
                "manifest": {"name": Path(kwargs["input_path"]).name},
                "use_llm": False,
                "execution_successful": not execution_failed,
                "analysis_completeness": {"is_complete": True, "coverage_percent": 100},
            }
        )
        return {**prepared.public_fields, "prepared_report": prepared}

    analyzer = Mock(side_effect=scan)
    monkeypatch.setattr(cli, "_scan_skill", analyzer)
    collections = []
    original_collect = cli._collect_recursive_reports

    def collect(*args, **kwargs):
        result = original_collect(*args, **kwargs)
        collections.append(result)
        return result

    monkeypatch.setattr(cli, "_collect_recursive_reports", collect)
    return root, analyzer, collections


@pytest.mark.parametrize("primary", list(cli.FormatChoice))
def test_recursive_all_formats_and_repetitions_scan_once(tmp_path, monkeypatch, primary):
    root, analyzer, collections = install_children(tmp_path, monkeypatch)
    primary_file = tmp_path / "primary"
    args = [
        "scan",
        str(root),
        "--recursive",
        "--no-llm",
        "--format",
        primary.value,
        "--output",
        str(primary_file),
    ]
    for i, choice in enumerate((*cli.FormatChoice, cli.FormatChoice.json)):
        args += ["--report", f"{choice.value}={tmp_path / str(i)}"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert analyzer.call_count == 3
    assert json.loads((tmp_path / "1").read_text())["skills_scanned"] == 3
    validate_sarif_report(json.loads((tmp_path / "3").read_text()))
    assert all((tmp_path / str(i)).exists() for i in range(5))
    assert "Combined report saved to" in result.stdout
    assert "Multi-Skill Summary" in result.stderr
    assert collections[0].children == []


@pytest.mark.parametrize("failure", ["render", "write", "interrupt"])
def test_recursive_later_outputs_survive_failures_and_release_snapshots(
    tmp_path, monkeypatch, failure
):
    root, analyzer, collections = install_children(tmp_path, monkeypatch)
    bad, later = tmp_path / "bad.sarif", tmp_path / "later.json"
    bad.write_text("unchanged")
    original_render, original_write = report_module.serialize_report, cli.write_text_no_follow

    def render(prepared, choice):
        assert analyzer.call_count == 3
        if choice == "sarif":
            if failure == "render":
                raise ValueError("serialization")
            if failure == "interrupt":
                raise KeyboardInterrupt
        return original_render(prepared, choice)

    def write(path, body):
        if str(path) == str(bad) and failure == "write":
            raise OSError("race")
        original_write(path, body)

    monkeypatch.setattr(report_module, "serialize_report", render)
    monkeypatch.setattr(cli, "write_text_no_follow", write)
    result = CliRunner().invoke(
        cli.app,
        [
            "scan",
            str(root),
            "--recursive",
            "--no-llm",
            "--format",
            "json",
            "--report",
            f"sarif={bad}",
            "--report",
            f"json={later}",
        ],
    )
    assert result.exit_code == (130 if failure == "interrupt" else 2), result.output
    assert json.loads(result.stdout)["skills_scanned"] == 3
    assert analyzer.call_count == 3
    assert not collections[0].execution_failed
    assert collections[0].children == []
    assert bad.read_text() == "unchanged"
    assert later.exists() is (failure != "interrupt")


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("execution_failed", [False, True])
def test_recursive_any_report_omission_gate_after_every_attempt(
    tmp_path, monkeypatch, strict, execution_failed
):
    root, analyzer, collections = install_children(
        tmp_path, monkeypatch, execution_failed=execution_failed
    )
    real_render = cli._render_recursive_report

    def render(collection, choice):
        result = real_render(collection, choice)
        if choice == cli.FormatChoice.sarif:
            result["report_completeness"] = {"is_complete": False}
        return result

    monkeypatch.setattr(cli, "_render_recursive_report", render)
    sarif, later = tmp_path / "extra.sarif", tmp_path / "later.json"
    args = [
        "scan",
        str(root),
        "--recursive",
        "--no-llm",
        "--format",
        "json",
        "--report",
        f"sarif={sarif}",
        "--report",
        f"json={later}",
    ]
    if strict:
        args += ["--fail-on-incomplete"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == (2 if execution_failed else 1 if strict else 0), result.output
    assert analyzer.call_count == 3
    assert sarif.exists() and later.exists()
    assert collections[0].children == []
