"""The whole output plan is rejected before emission for newly discovered sources."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from skillspector import cli
from skillspector.report_sources import current_source_inventory, record_scan_sources


@pytest.mark.parametrize("mode", ["ordinary", "recursive", "transitive"])
@pytest.mark.parametrize("primary", [False, True])
def test_late_collision_aborts_every_output(tmp_path, monkeypatch, mode, primary):
    root = tmp_path / "input"
    root.mkdir()
    if mode == "recursive":
        for name in ("a", "b"):
            child = root / name
            child.mkdir()
            (child / "SKILL.md").write_text("---\nname: example\n---\nHello")
    else:
        (root / "SKILL.md").write_text("---\nname: example\n---\nHello")
    late = tmp_path / "late.py"
    late.write_text("protected source")
    safe = tmp_path / "safe.json"
    safe.write_text("existing report")
    calls = []

    def scan(**kwargs):
        calls.append(kwargs["input_path"])
        raw = {"skill_path": str(late), "components": [str(late)]}
        # Emulate the raw graph hook before any transitive decoration/retention.
        if mode == "transitive":
            record_scan_sources(raw)
        return {
            **raw,
            "report_body": "PRIMARY MUST NOT BE EMITTED",
            "risk_score": 0,
            "analysis_completeness": {"is_complete": True},
        }

    monkeypatch.setattr(cli, "_scan_skill", scan)
    writer = Mock(side_effect=AssertionError("emission began"))
    monkeypatch.setattr(cli, "_write_result", writer)
    monkeypatch.setattr(cli, "_write_recursive_report", writer)
    args = ["scan", str(root), "--no-llm", "--format", "terminal", "--verbose"]
    if mode == "recursive":
        args += ["--recursive"]
    if mode == "transitive":
        args += ["--transitive"]
    if primary:
        args += ["--output", str(late), "--report", f"json={safe}"]
    else:
        args += ["--report", f"json={late}", "--report", f"json={safe}"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 2, result.output
    assert calls
    assert result.stdout == ""
    writer.assert_not_called()
    assert late.read_text() == "protected source"
    assert safe.read_text() == "existing report"
    assert current_source_inventory() is None


def test_poisoned_late_inventory_fails_closed_and_cleans_result(tmp_path, monkeypatch):
    source = tmp_path / "SKILL.md"
    source.write_text("Hello")
    extra = tmp_path / "extra.json"
    cleanup = Mock()
    monkeypatch.setattr(cli, "cleanup_result", cleanup)

    def scan(**kwargs):
        try:
            record_scan_sources({"components": ["relative-without-root.py"]})
        except ValueError:
            pass  # An orchestration catch cannot make partial protection safe.
        return {"report_body": "not emitted", "risk_score": 0}

    monkeypatch.setattr(cli, "_scan_skill", scan)
    result = CliRunner().invoke(
        cli.app, ["scan", str(source), "--no-llm", "--report", f"json={extra}"]
    )
    assert result.exit_code == 2
    assert result.stdout == ""
    assert not extra.exists()
    cleanup.assert_called_once()
    assert current_source_inventory() is None


@pytest.mark.parametrize("stream", [False, True])
def test_graph_return_records_sources_before_cleanup(tmp_path, monkeypatch, stream):
    source = tmp_path / "SKILL.md"
    source.write_text("Hello")
    raw = {"skill_path": str(source), "components": [str(source)]}
    graph = Mock()
    graph.invoke.return_value = raw
    graph.stream.return_value = iter([("values", raw)])
    monkeypatch.setattr(cli, "graph", graph)
    inventory = cli.SourceProtectionInventory()
    with cli.track_report_sources(inventory):
        cli._run_graph_scan(str(source), cli.FormatChoice.json, True, stream_progress=stream)
    assert Path(source) in inventory.paths
