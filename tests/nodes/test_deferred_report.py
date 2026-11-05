"""Internal deferred graph path never calls a serializer during analysis."""

import importlib
import json
from unittest.mock import Mock

import pytest

from skillspector.models import Finding
from skillspector.nodes.report import (
    PreparedReport,
    report,
    report_rendering_deferred,
    report_rendering_is_deferred,
    serialize_report,
)

report_module = importlib.import_module("skillspector.nodes.report")


@pytest.mark.parametrize("use_state_flag", [False, True])
def test_deferred_preparation_renders_nothing_and_retains_findings(monkeypatch, use_state_flag):
    finding = Finding(rule_id="TEST", message="preserved", severity="HIGH")
    state = {"findings": [finding], "use_llm": False}
    if use_state_flag:
        state["defer_report_rendering"] = True
    renderers = ["_build_sarif", "_format_json", "_format_terminal", "_format_markdown"]
    for name in renderers:
        monkeypatch.setattr(report_module, name, Mock(side_effect=AssertionError("rendered")))
    if use_state_flag:
        result = report(state)
    else:
        with report_rendering_deferred():
            result = report(state)
    assert "report_body" not in result
    assert "sarif_report" not in result
    prepared = result["prepared_report"]
    assert isinstance(prepared, PreparedReport)
    assert prepared.reported_findings[0].message == "preserved"
    for name in renderers:
        getattr(report_module, name).assert_not_called()
    with pytest.raises(AssertionError, match="rendered"):
        serialize_report(prepared, "sarif")
    monkeypatch.setattr(report_module, "_format_json", Mock(return_value='{"ok": true}'))
    assert json.loads(serialize_report(prepared, "json")["report_body"])["ok"] is True
    assert prepared.reported_findings[0].message == "preserved"
    assert result["risk_score"] == prepared.risk_score


def test_deferred_scope_restores_after_interrupt():
    assert not report_rendering_is_deferred()


def test_cli_graph_invocation_defers_only_inside_internal_scope(monkeypatch):
    import skillspector.cli as cli

    graph = Mock()
    graph.invoke.return_value = {}
    monkeypatch.setattr(cli, "graph", graph)
    cli._run_graph_scan("example", cli.FormatChoice.json, True)
    assert "defer_report_rendering" not in graph.invoke.call_args.args[0]
    with report_rendering_deferred():
        cli._run_graph_scan("example", cli.FormatChoice.json, True)
    assert graph.invoke.call_args.args[0]["defer_report_rendering"] is True
    with pytest.raises(KeyboardInterrupt), report_rendering_deferred():
        assert report_rendering_is_deferred()
        raise KeyboardInterrupt
    assert not report_rendering_is_deferred()
