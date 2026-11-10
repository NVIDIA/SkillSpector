# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Documentation notices do not change findings, completeness or strict exits."""

import json
from typing import cast

import pytest
from typer.testing import CliRunner

from skillspector import cli
from skillspector.inspection_ledger import (
    UNSUPPORTED_PROSE_COMMAND_NOTICE,
    UNSUPPORTED_PROSE_COMMAND_PATTERN,
    bounded_inspection_notices,
    finalize_ledger,
)
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner
from skillspector.nodes.report import report
from skillspector.state import SkillspectorState, merge_inspection_notices


def _state() -> SkillspectorState:
    message = 'Run `sh -c "echo safe"` and later eval "$PAYLOAD".'
    source = (
        "rules:\n- id: example\n  languages: [javascript]\n  severity: WARNING\n  message: "
        + json.dumps(message)
        + "\n  pattern: ordinary\n"
    )
    state = cast(
        SkillspectorState,
        {"components": ["rules.yaml"], "file_cache": {"rules.yaml": source}, "use_llm": False},
    )
    state.update(static_runner.run_static_patterns_with_ledger(state, [tm]))
    completeness, _ = finalize_ledger(state)
    state["analysis_completeness"] = completeness
    return state


@pytest.fixture(autouse=True)
def no_provider_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "skillspector.nodes.report.is_llm_available", lambda **_: (False, "Disabled")
    )


def test_notice_does_not_change_canonical_completeness() -> None:
    state = _state()
    with_notice = state["analysis_completeness"]
    without_notice, _ = finalize_ledger({**state, "inspection_notices": []})
    assert with_notice["is_complete"] is True
    assert {k: v for k, v in with_notice.items() if k != "notices"} == without_notice
    assert not with_notice.get("warnings")
    assert not with_notice["ledger_exceptions"]
    assert len(with_notice["notices"]) == 1


@pytest.mark.parametrize("output_format", ["json", "sarif", "markdown", "terminal"])
def test_every_report_renders_a_nonblocking_source_notice(output_format: str) -> None:
    state = _state()
    state["output_format"] = output_format
    result = report(state)
    assert result["risk_score"] == 0
    assert result["analysis_completeness"]["is_complete"] is True
    body = result["report_body"]
    assert "This command pattern is not allowed for inspection" in " ".join(body.split())
    if output_format == "json":
        payload = json.loads(body)
        assert payload["issues"] == []
        assert (
            payload["analysis_completeness"]["notices"][0]["message"]
            == UNSUPPORTED_PROSE_COMMAND_NOTICE
        )
        assert "llm_assessment_complete" not in payload["metadata"]
        assert payload["analysis_completeness"]["notices"][0]["path"] == "rules.yaml"
    elif output_format == "sarif":
        payload = json.loads(body)
        run = payload["runs"][0]
        assert run["results"] == []
        invocation = run["invocations"][0]
        assert invocation["executionSuccessful"] is True
        notices = invocation["toolExecutionNotifications"]
        notice = next(
            n for n in notices if n["properties"].get("kind") == "documentation_pattern_notice"
        )
        assert notice["message"]["text"] == UNSUPPORTED_PROSE_COMMAND_NOTICE
        assert notice["level"] == "note"
        assert notice["locations"][0]["physicalLocation"]["region"]["startLine"] == 5


def test_notice_projection_is_bounded_deduplicated_and_vetted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("skillspector.inspection_ledger.MAX_FINDING_OUTPUT_RECORDS", 2)
    first = _state()["inspection_notices"][0]
    second = {**first, "path": "second.yaml", "message": "untrusted display text"}
    third = {**first, "path": "third.yaml"}
    invalid = [{**first, "path": "../outside"}, {**first, "source_end_offset": 0}]
    rows = bounded_inspection_notices([*invalid, first, first, second, third])
    assert [r["path"] for r in rows] == ["rules.yaml", "second.yaml"]
    assert all(r["reason_code"] == UNSUPPORTED_PROSE_COMMAND_PATTERN for r in rows)
    assert all(r["message"] == UNSUPPORTED_PROSE_COMMAND_NOTICE for r in rows)
    assert merge_inspection_notices([first], [first, second, third]) == rows


@pytest.mark.parametrize("incomplete, expected", [(False, 0), (True, 1)])
def test_strict_cli_uses_real_completeness_not_the_notice(
    tmp_path, monkeypatch: pytest.MonkeyPatch, incomplete: bool, expected: int
) -> None:
    state = _state()
    if incomplete:
        state["analysis_completeness"] = {**state["analysis_completeness"], "is_complete": False}
    state["output_format"] = "json"
    result = report(state)
    monkeypatch.setattr(cli, "_scan_skill", lambda **_: result)
    (tmp_path / "SKILL.md").write_text("---\nname: example\ndescription: example\n---\n")
    invocation = CliRunner().invoke(
        cli.app, ["scan", str(tmp_path), "--no-llm", "--format", "json", "--fail-on-incomplete"]
    )
    assert invocation.exit_code == expected
    assert UNSUPPORTED_PROSE_COMMAND_NOTICE in invocation.output
