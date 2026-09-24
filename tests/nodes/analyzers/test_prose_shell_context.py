# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordinary prose must not establish an unresolved shell execution context."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner

_MANIFEST = (
    "---\nname: documentation-control\n"
    "description: Explain the documented interface when asked about its behavior.\n"
    "---\n\nRead [the interface guide](guide.md) for the explanation.\n"
)
_APOSTROPHE = "The interface doesn't match the spec.\n\n" + "".join(
    f"Observation {i:03d}: the interface accepts an ordinary numeric value, "
    "records its label, and displays a plain explanatory result for this example.\n"
    for i in range(45)
)
_LAYOUT = (
    "```\n```\n"
    "| **Data layout** (row-first vs column-first) | Describes the sample dimensions |\n"
    "```\n```python\n```\n```\n"
    "1. Read the example guide at `notes/<example>_guide.md`.\n"
)
_CASES = [
    pytest.param(_APOSTROPHE, id="apostrophe-long"),
    pytest.param(_APOSTROPHE.replace("doesn't", "does not"), id="apostrophe-control"),
    pytest.param(_LAYOUT, id="overview-minimal"),
    pytest.param(
        _LAYOUT.replace("row-first vs column-first", "row major vs column major"),
        id="overview-control",
    ),
]


@pytest.mark.parametrize("content", _CASES)
def test_benign_guide_has_complete_analyzer_coverage(content: str) -> None:
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": content}}, [tm]
    )
    assert result["findings"] == []
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize("content", _CASES)
def test_benign_referenced_guide_passes_strict_cli(tmp_path: Path, content: str) -> None:
    (tmp_path / "SKILL.md").write_text(_MANIFEST, encoding="utf-8")
    (tmp_path / "guide.md").write_text(content, encoding="utf-8")
    result = CliRunner().invoke(
        app, ["scan", str(tmp_path), "--no-llm", "--format", "json", "--fail-on-incomplete"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    coverage = report["analysis_completeness"]
    assert coverage["execution_successful"] is True
    assert coverage["is_complete"] is True
    assert coverage["coverage_percent"] == 100.0
    assert coverage["ledger_exceptions"] == []
    assert report["risk_assessment"]["score"] == 0
    assert report["risk_assessment"]["recommendation"] == "SAFE"
    assert report["issues"] == []


@pytest.mark.parametrize(
    "content",
    [
        f"```sh\n{_APOSTROPHE}```\n",
        f"~~~sh\n{_APOSTROPHE}~~~\n",
        "    " + _APOSTROPHE.replace("\n", "\n    "),
        "`\nThe interface doesn't match the spec.\n`\n",
        f'bash -c "\n{_APOSTROPHE}"\n',
        "$(\nThe interface doesn't match the spec.\n') -rf /",
        "$(\n\nThe interface doesn't match the spec.\n') -rf /",
        "r'\nThe interface doesn't match the spec.\n'rm -rf /",
        "<pre>\nThe interface doesn't match the spec.\n</pre>",
    ],
)
def test_prose_ownership_never_rewrites_enclosing_code(content: str) -> None:
    projected = tm._markdown_shell_text(content, lambda: None)
    apostrophe = content.index("doesn't") + len("doesn")
    assert projected[apostrophe] == "'"
    assert len(projected) == len(content)


@pytest.mark.parametrize(
    "content",
    [
        f'bash -c "\n{_APOSTROPHE}"\n',
        "$(\nThe interface doesn't match the spec.\n') -rf /",
    ],
)
def test_prose_inside_unresolved_shell_content_remains_partial(content: str) -> None:
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": content}}, [tm]
    )
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize("content", [_APOSTROPHE, _LAYOUT])
@pytest.mark.parametrize(
    "command,partial",
    [
        ("rm -rf /", False),
        ("$($CMD) -rf /", True),
        ("$($(resolve_tool)/printf %s rm) -rf /", True),
    ],
)
def test_benign_prefix_cannot_hide_later_commands(
    tmp_path: Path, content: str, command: str, partial: bool
) -> None:
    (tmp_path / "SKILL.md").write_text(_MANIFEST, encoding="utf-8")
    # All command-looking text is scan data; no fixture is executed.
    (tmp_path / "guide.md").write_text(content + "\n" + command + "\n", encoding="utf-8")
    result = CliRunner().invoke(
        app, ["scan", str(tmp_path), "--no-llm", "--format", "json", "--fail-on-incomplete"]
    )
    report = json.loads(result.output)
    assert report["analysis_completeness"]["execution_successful"] is True
    assert report["risk_assessment"]["recommendation"] != "SAFE"
    if partial:
        assert result.exit_code == 1
        assert report["analysis_completeness"]["is_complete"] is False
        assert any(
            event["path"] == "guide.md" and event["reason_code"] == "static_parse_limit"
            for event in report["analysis_completeness"]["ledger_exceptions"]
        )
        assert any(issue["id"] == "AE1" for issue in report["issues"])
    else:
        assert any(issue["id"] == "TM1" for issue in report["issues"])


@pytest.mark.parametrize(
    "arguments",
    [
        " " * 8200 + "-rf /",
        '"' + "x" * 8200 + '" -rf /',
        "/" + "x" * 8200 + " -rf",
        "-r " + "x" * 8200 + " -f /",
    ],
)
def test_runtime_operands_beyond_tokenizer_bound_stay_partial(arguments: str) -> None:
    assert tm.has_bounded_parse_exhaustion("$CMD " + arguments, lambda: None)


@pytest.mark.parametrize(
    "command",
    [
        "$(echo; $CMD -rf /)",
        "`echo; $CMD -rf /`",
        '$(echo; $WRAP -c "rm -rf /")',
    ],
)
@pytest.mark.parametrize("tail", ["'", " " * 8200 + "safe", "x" * 8200])
def test_outer_context_does_not_discard_nested_command_operands(command: str, tail: str) -> None:
    assert tm.has_bounded_parse_exhaustion(command + " " + tail, lambda: None)


def test_prose_ownership_requires_complete_context() -> None:
    assert tm.has_bounded_parse_exhaustion(
        _APOSTROPHE, lambda: None, file_type="markdown", complete_context=False
    )
    assert tm.has_bounded_parse_exhaustion(_APOSTROPHE, lambda: None, file_type="shell")


def test_prose_ownership_preserves_cancellation() -> None:
    checks = 0

    def cancel() -> None:
        nonlocal checks
        checks += 1
        if checks == 20:
            raise TimeoutError("cancelled")

    with pytest.raises(TimeoutError, match="cancelled"):
        tm.has_bounded_parse_exhaustion(_APOSTROPHE * 100, cancel, file_type="markdown")


def test_prose_ownership_checks_cancellation_after_word_jumps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed_words = 0
    original = tm._parse_shell_command_word

    def counted(*args, **kwargs):
        nonlocal parsed_words
        parsed_words += 1
        return original(*args, **kwargs)

    def cancel() -> None:
        if parsed_words >= 10:
            raise TimeoutError("cancelled")

    monkeypatch.setattr(tm, "_parse_shell_command_word", counted)
    # Each short parameter jumps over an exact multiple of the callback stride.
    content = "a" * 4095 + "$X " + ("a" * 4093 + "$X ") * 100
    content += "\n\nThe interface doesn't match the spec.\n"
    with pytest.raises(TimeoutError, match="cancelled"):
        tm._markdown_shell_text(content, cancel)
