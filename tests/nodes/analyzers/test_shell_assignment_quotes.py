# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.nodes.analyzers.static_patterns_tool_misuse import has_bounded_parse_exhaustion

_COMMENT_PADDING = "\n" + "#" * 6000 + "\n"
_APOSTROPHE_PADDING = "\n" + "y = 'b'\n" * 800


@pytest.mark.parametrize(
    "source",
    [
        'out="$(date)"',
        'v="a (b)"',
        'v="|$a|"',
        'printf "%s" "$(f "$x")"',
        '[ "$(f)" = true ]',
        'cat <<< "$(f)"',
        'v="a \\" (b)"',
        'v="a \\" |$a|"',
    ],
)
def test_complete_shell_quotes_do_not_consume_following_padding(source: str) -> None:
    assert not has_bounded_parse_exhaustion(source + "\n" + "#" * 6000 + "\n", lambda: None)


@pytest.mark.parametrize(
    "source", ['x="$(unterminated', 'x="unclosed', '"$(printf "%s" "$x")" -rf /']
)
def test_unresolved_shell_words_still_exhaust(source: str) -> None:
    assert has_bounded_parse_exhaustion(source + "\n" + "#" * 6000 + "\n", lambda: None)


def test_assignment_ownership_retains_nested_command_budget() -> None:
    nested_word = "r" * 4097
    source = f'out="$({nested_word} -rf /)"'
    assert has_bounded_parse_exhaustion(source, lambda: None)


def test_failed_word_parse_does_not_claim_following_command() -> None:
    source = 'Test-Path "$($_.FullName)\\cli-path"; $($CMD) -rf /'
    assert has_bounded_parse_exhaustion(source, lambda: None)


@pytest.mark.parametrize(
    "source", ['Run "$(resolve_tool) -rf /"', '# see "notes\n$(resolve_tool) -rf /\n# "']
)
def test_quoted_runtime_commands_keep_fail_closed_coverage(source: str) -> None:
    from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
    from skillspector.nodes.analyzers import static_patterns_tool_misuse, static_runner

    assert has_bounded_parse_exhaustion(source, lambda: None)
    path = "script.sh" if source.startswith("#") else "SKILL.md"
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "local_file_cache": {path: source}, "file_cache": {path: source}},
        [static_patterns_tool_misuse],
    )
    assert any(
        e["outcome"] == LedgerOutcome.PARTIAL
        and e["reason_code"] == LedgerReason.STATIC_PARSE_LIMIT
        for e in result["inspection_ledger"]
    )


@pytest.mark.parametrize("file_type", ["shell", "markdown", "python"])
def test_non_shell_unclosed_assignment_remains_conservative(file_type: str) -> None:
    source = 'name="value' + "\n" + "text " * 1200
    assert has_bounded_parse_exhaustion(source, lambda: None, file_type=file_type)


@pytest.mark.parametrize(
    ("source", "file_type"),
    [
        ('# use "$(cmd)" here', "shell"),
        ('```bash\nout="$(date)"\n```', "markdown"),
        ('x="$(printf \'%s\' "$X")"', "shell"),
        ("model_count=\"$(printf '%s\\n' \"$j\" | jq -r 'length')\"", "shell"),
    ],
    ids=["issue-comment-form", "markdown-fence", "printf-value", "printf-pipeline-value"],
)
def test_benign_quoted_values_do_not_consume_comment_padding(source: str, file_type: str) -> None:
    assert not has_bounded_parse_exhaustion(
        source + _COMMENT_PADDING, lambda: None, file_type=file_type
    )


@pytest.mark.parametrize(
    ("source", "file_type"),
    [
        ('# Keep `X="${X}"` so X\'s value wins.', "python"),
        ("# Keep `X='${X}'`, so X's value wins.", "python"),
        ('Use `X="${X}"`. That\'s the self-reference.', "shell"),
    ],
    ids=["double-quoted", "single-quoted", "prose"],
)
def test_backtick_enclosed_assignment_stops_at_its_closing_backtick(
    source: str, file_type: str
) -> None:
    assert not has_bounded_parse_exhaustion(
        source + _APOSTROPHE_PADDING, lambda: None, file_type=file_type
    )


@pytest.mark.parametrize(
    ("source", "file_type"),
    [
        ('ls # see "notes\n"$(resolve_tool)" -rf /\n# "', "shell"),
        ('Set name="value in prose.\n\n```bash\n"$(resolve_tool)" -rf /\n```\n"\n', "markdown"),
        ('x = f(name="value)\n"$(resolve_tool)" -rf /\n"\n', "python"),
        ('Set name="value\n$T -rf /\n"\n', "markdown"),
        ('name="a "$(resolve_tool)" -rf /"', "shell"),
    ],
    ids=[
        "trailing-comment",
        "prose-assignment-before-fence",
        "host-language-assignment",
        "prose-assignment-parameter",
        "same-line-runtime-opener",
    ],
)
def test_misaligned_quote_claims_keep_runtime_commands_partial(source: str, file_type: str) -> None:
    assert has_bounded_parse_exhaustion(source, lambda: None, file_type=file_type)


def test_issue_628_bundle_reports_complete_coverage(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text(
        "---\nname: repro\ndescription: Minimal repro.\n---\nRun `scripts/run.sh`.\n",
        encoding="utf-8",
    )
    padding = "".join(f"# padding line {index} " + "." * 50 + "\n" for index in range(90))
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.sh").write_text(
        '#!/bin/bash\nout="$(date)"\necho "$out"\n' + padding, encoding="utf-8"
    )

    result = CliRunner().invoke(app, ["scan", str(tmp_path), "--format", "json", "--no-llm"])

    # Exit 1 is the documented high-risk verdict, not a scan execution failure.
    assert result.exit_code in {0, 1}, result.output
    report = json.loads(result.output)
    assert report["analysis_completeness"]["status"] == "complete"
    assert report["analysis_completeness"]["ledger_exceptions"] == []
    assert not any(issue["id"] == "AE1" for issue in report["issues"])
