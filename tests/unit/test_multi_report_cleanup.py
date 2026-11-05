"""Root resources outlive report emission and are released on errors/interrupts."""

from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from skillspector import cli
from skillspector.report_sources import current_source_inventory


@pytest.mark.parametrize("failure", ["write", "render", "interrupt"])
def test_cleanup_after_partial_output_and_interrupt(tmp_path, monkeypatch, failure):
    source = tmp_path / "SKILL.md"
    source.write_text("Hello")
    resource = tmp_path / "skillspector_scan"
    resource.mkdir()
    events = []

    def scan(**kwargs):
        prepared = cli.prepare_report({"manifest": {"name": "example"}, "use_llm": False})
        return {
            **prepared.public_fields,
            "prepared_report": prepared,
            "temp_dir_for_cleanup": str(resource),
        }

    monkeypatch.setattr(cli, "_scan_skill", scan)
    original_render, original_write = cli.serialize_report, cli.write_text_no_follow
    first, second, third = [tmp_path / name for name in ("first.json", "second.md", "third.json")]

    def render(prepared, choice):
        assert resource.exists()
        events.append("render:" + choice)
        if choice == "markdown":
            if failure == "interrupt":
                raise KeyboardInterrupt
            if failure == "render":
                raise RuntimeError("renderer failed")
        return original_render(prepared, choice)

    def write(path, body):
        assert resource.exists()
        events.append("write:" + str(path))
        if str(path) == str(second) and failure == "write":
            raise OSError("write race")
        original_write(path, body)

    original_cleanup = cli.cleanup_result

    def cleanup(result):
        events.append("cleanup")
        original_cleanup(result)

    monkeypatch.setattr(cli, "serialize_report", render)
    monkeypatch.setattr(cli, "write_text_no_follow", write)
    monkeypatch.setattr(cli, "cleanup_result", cleanup)
    result = CliRunner().invoke(
        cli.app,
        [
            "scan",
            str(source),
            "--no-llm",
            "--format",
            "json",
            "--output",
            str(first),
            "--report",
            f"markdown={second}",
            "--report",
            f"json={third}",
        ],
    )
    assert result.exit_code == (130 if failure == "interrupt" else 2), result.output
    assert events[-1] == "cleanup"
    assert first.exists()
    assert not second.exists()
    assert third.exists() is (failure != "interrupt")
    assert not resource.exists()
    assert current_source_inventory() is None


def test_raw_source_tracking_failure_cleans_root_before_return(tmp_path, monkeypatch):
    resource = tmp_path / "skillspector_scan"
    resource.mkdir()
    raw = {"temp_dir_for_cleanup": str(resource), "components": ["missing-root.py"]}
    monkeypatch.setattr(cli, "_run_graph_scan_for_source", Mock(return_value=raw))
    with cli.track_report_sources(cli.SourceProtectionInventory()):
        with pytest.raises(ValueError, match="no concrete local root"):
            cli._scan_skill(
                "remote",
                cli.FormatChoice.json,
                True,
                None,
                None,
                False,
                False,
                False,
                1,
                None,
                None,
            )
    assert not resource.exists()
