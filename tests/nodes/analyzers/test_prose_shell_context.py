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
_OBSERVATIONS = "".join(
    f"Observation {i:03d}: the interface accepts an ordinary numeric value, "
    "records its label, and displays a plain explanatory result for this example.\n"
    for i in range(45)
)
_APOSTROPHE = "The interface doesn't match the spec.\n\n" + _OBSERVATIONS
# Possessives and contractions in a prose line with other punctuation. The
# count is odd, so read as shell quotes the last one stays open to the end.
_POSSESSIVE = (
    "The router keeps no mesh/CAD/splat table; it calls each reference's probe "
    "mode, and that reference's reader (don't skip it) lists the formats.\n\n"
) + _OBSERVATIONS
# Read as shell quotes, a possessive and a later plural possessive pair up.
# Only one of them as punctuation would leave the other open to the end.
_PLURAL_POSSESSIVE = (
    "It reads each reference's notes.\n\nIt lists the drivers' files.\n\n" + _OBSERVATIONS
)
_LAYOUT = (
    "```\n```\n"
    "| **Data layout** (row-first vs column-first) | Describes the sample dimensions |\n"
    "```\n```python\n```\n```\n"
    "1. Read the example guide at `notes/<example>_guide.md`.\n"
)
_CASES = [
    pytest.param(_APOSTROPHE, id="apostrophe-long"),
    pytest.param("- " + _APOSTROPHE, id="list-apostrophe-long"),
    pytest.param(("- " + _APOSTROPHE).replace("\n", "\r\n"), id="list-apostrophe-crlf"),
    pytest.param("- " + _APOSTROPHE.replace("doesn't", "does not"), id="list-control"),
    pytest.param(_APOSTROPHE.replace("doesn't", "does not"), id="apostrophe-control"),
    pytest.param(_POSSESSIVE, id="possessive-long"),
    pytest.param("- " + _POSSESSIVE, id="list-possessive-long"),
    pytest.param(
        _POSSESSIVE.replace("'s", "s").replace("don't", "do not"), id="possessive-control"
    ),
    pytest.param(_PLURAL_POSSESSIVE, id="plural-possessive-long"),
    pytest.param(_LAYOUT, id="overview-minimal"),
    pytest.param(
        _LAYOUT.replace("row-first vs column-first", "row major vs column major"),
        id="overview-control",
    ),
]


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize(
    "paragraph",
    [
        "> - Read `--workflow <name>` to record the reader's selected interface.\n",
        "> 1. Read ``--workflow <name>`` to record the reader's selected interface.\n",
        "> > - Read `--workflow <name>` to record the reader's selected interface.\n",
        "> - Read `guide.md` for the interface.\n>   The runner's manifest identifies the source.\n",
    ],
)
def test_quoted_list_literal_code_and_prose_keep_source_ownership(paragraph, newline):
    content = (paragraph + "\n" + _OBSERVATIONS).replace("\n", newline)
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": content}}, [tm]
    )
    assert result["findings"] == []
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize(
    "paragraph",
    [
        "> - Run `$CMD -rf /` using the reader's chosen command.\n",
        "> - Run `$(cat command.txt)` using the reader's chosen command.\n",
        "> - Read ``one ` character`` from the reader's guide.\n",
        "> - Read `guide.md` for the interface.\nThe runner's manifest identifies the source.\n",
        "> - ```sh\n>   `$CMD -rf /`\n>   ```\n",
        "> - <pre>\n>   `guide.md` contains the reader's notes.\n>   </pre>\n",
        ">     `guide.md` contains the reader's notes.\n",
    ],
)
def test_quoted_list_unproven_inline_bodies_keep_backticks(paragraph):
    projected = tm._markdown_shell_text(paragraph, lambda: None, word_apostrophes=True)
    assert "`" in projected


@pytest.mark.parametrize("quote", ["'", '"'])
def test_quoted_list_cannot_steal_an_earlier_shell_string(quote):
    content = f"bash -c {quote}\n> - Read `guide.md` from the reader's guide.\n{quote}\n"
    projected = tm._markdown_shell_text(content, lambda: None, word_apostrophes=True)
    assert "`guide.md`" in projected
    assert "reader's" in projected


def test_quoted_list_literal_ownership_preserves_security_findings():
    content = "> - Run `rm -rf /` on the selected filesystem.\n"
    assert any(finding.rule_id == "TM1" for finding in tm.analyze(content, "guide.md", "markdown"))


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize(
    "paragraph",
    [
        "> Read `~/guide.md` for the reader's interface.\n",
        "> Read ``guide.md`` for the reader's interface.\n",
        "1. Read `guide.md` for the reader's interface.\n",
        "-\n  Read `guide.md` for the reader's interface.\n",
        "1.\n   Read `guide.md` for the reader's interface.\n",
        "3. Read `guide.md` for the interface.\n   The reader's copy identifies the source.\n",
        "> The reader's guide is described below.\n>\n> Read `guide.md` for the interface.\n",
    ],
)
def test_native_container_paragraphs_keep_exact_literal_boundaries(paragraph, newline):
    from skillspector.markdown_tokens import quoted_list_spans

    content = ("# Unicode: 日本語 🧪\n\n" + paragraph + "\n" + _OBSERVATIONS).replace("\n", newline)
    spans = quoted_list_spans(content, lambda: None)
    assert spans is not None
    assert spans.code
    for start, body, close, end in spans.code:
        assert content[start:body] == content[close:end]
        assert content[start:body] in {"`", "``"}
        assert content[body:close] in {"guide.md", "~/guide.md"}
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": content}}, [tm]
    )
    assert result["findings"] == []
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


def test_native_quote_paragraph_recovers_an_established_prose_boundary(monkeypatch):
    content = (
        "> The reader's guide is described below.\n>\n"
        "> Read `guide.md` for the interface.\n\n" + _OBSERVATIONS
    )
    with monkeypatch.context() as baseline:
        # The previous quote+list-only helper grants no role to this paragraph.
        baseline.setattr(tm, "quoted_list_spans", lambda *args: None)
        before = tm.bounded_parse_diagnostic(content, lambda: None, file_type="markdown")
    assert before is not None
    assert tm.bounded_parse_diagnostic(content, lambda: None, file_type="markdown") is None


def test_existing_quoted_list_heading_keeps_its_native_inline_ownership():
    from skillspector.markdown_tokens import quoted_list_spans

    content = "> - ### Read `guide.md` from the reader's guide.\n\n" + _OBSERVATIONS
    spans = quoted_list_spans(content, lambda: None)
    assert spans is not None
    start = content.index("`guide.md`")
    assert (start, start + 1, start + 9, start + 10) in spans.code
    assert tm.bounded_parse_diagnostic(content, lambda: None, file_type="markdown") is None


@pytest.mark.parametrize("quote", ["'", '"'])
@pytest.mark.parametrize("container", ["> ", "1. "])
def test_native_container_preserves_existing_enclosing_shell_quote_reading(
    monkeypatch, quote, container
):
    content = (
        f"bash -c {quote}\n"
        + container
        + "Read `guide.md` from the reader's copy.\n"
        + quote
        + "\n"
    )
    with monkeypatch.context() as baseline:
        baseline.setattr(tm, "quoted_list_spans", lambda *args: None)
        before = tm._markdown_shell_text(content, lambda: None, word_apostrophes=True)
        before_diagnostic = tm.bounded_parse_diagnostic(content, lambda: None, file_type="markdown")
    projected = tm._markdown_shell_text(content, lambda: None, word_apostrophes=True)
    assert projected == before
    assert "reader's" in projected
    assert (
        tm.bounded_parse_diagnostic(content, lambda: None, file_type="markdown")
        == before_diagnostic
    )


@pytest.mark.parametrize("container", ["> ", "1. "])
@pytest.mark.parametrize("body", ["$CMD -rf /", 'sh -c "$PAYLOAD"', "$(cat command.txt)"])
def test_container_runtime_inline_code_keeps_its_existing_fallback(monkeypatch, container, body):
    from skillspector.markdown_tokens import quoted_list_spans

    content = container + "Run `" + body + "` using the reader's command.\n\n" + _OBSERVATIONS
    spans = quoted_list_spans(content, lambda: None)
    assert spans is not None
    assert spans.code == ()
    with monkeypatch.context() as baseline:
        baseline.setattr(tm, "quoted_list_spans", lambda *args: None)
        before = tm._markdown_shell_text(content, lambda: None, word_apostrophes=True)
        before_diagnostic = tm.bounded_parse_diagnostic(content, lambda: None, file_type="markdown")
    projected = tm._markdown_shell_text(content, lambda: None, word_apostrophes=True)
    assert projected == before
    assert (
        tm.bounded_parse_diagnostic(content, lambda: None, file_type="markdown")
        == before_diagnostic
    )


@pytest.mark.parametrize("container", ["> ", "1. "])
def test_container_literal_ownership_preserves_raw_and_later_high_findings(container):
    content = container + "Read `guide.md` for the interface.\n\n```sh\nrm -rf /\n```\n"
    assert any(
        f.rule_id == "TM1" and f.severity == "HIGH"
        for f in tm.analyze(content, "guide.md", "markdown")
    )
    inline = container + "Run `rm -rf /` on the selected filesystem.\n"
    assert any(
        f.rule_id == "TM1" and f.severity == "HIGH"
        for f in tm.analyze(inline, "guide.md", "markdown")
    )


def test_quoted_list_native_parser_failure_retains_original_text(monkeypatch):
    import builtins

    from skillspector import markdown_tokens

    original_import = builtins.__import__

    def without_parser(name, *args, **kwargs):
        if name == "markdown_it":
            raise ImportError("optional parser unavailable")
        return original_import(name, *args, **kwargs)

    markdown_tokens._quoted_list_spans.cache_clear()
    monkeypatch.setattr(builtins, "__import__", without_parser)
    content = "> - Read `guide.md` from the reader's guide.\n"
    try:
        assert markdown_tokens.quoted_list_spans(content, lambda: None) is None
    finally:
        markdown_tokens._quoted_list_spans.cache_clear()


def test_quoted_list_source_bound_retains_conservative_fallback():
    from skillspector.markdown_tokens import quoted_list_spans

    content = "> - Read `guide.md` from the reader's guide.\n" + "a" * 131_072
    assert quoted_list_spans(content, lambda: None) is None


def test_quoted_list_nesting_bound_retains_conservative_fallback():
    from skillspector.markdown_tokens import quoted_list_spans

    content = "> " * 40 + "- Read `guide.md` from the reader's guide.\n"
    assert quoted_list_spans(content, lambda: None) is None


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize(
    "paragraph",
    [
        "Read `guide.md` for the interface.\nThe runner's manifest identifies the source.\n",
        "- Read `guide.md` for the interface.\n  The runner's manifest identifies the source.\n",
        "Read the reader's \"documented behavior\" for the interface.\nThe runner's manifest identifies the source.\n",
        "Read \\` literally and `guide.md` for the interface.\nThe runner's manifest identifies the source.\n",
        "| Topic | Description |\n| --- | --- |\n| Memory | The engine's limit applies. |\n",
        "| Topic | Description |\n| --- | --- |\n| Memory | The engine's limit and runner's settings don't conflict. |\n",
    ],
)
def test_completed_inline_lines_and_table_prose_own_apostrophes(
    paragraph: str, newline: str
) -> None:
    content = (paragraph + "\n" + _OBSERVATIONS).replace("\n", newline)

    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": content}}, [tm]
    )

    assert result["findings"] == []
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize(
    "content",
    [
        "Read `an unfinished span.\nThe runner's manifest identifies the source.\n",
        "Read `an unfinished span with ``paired`` ticks.\nThe runner's manifest identifies it.\n",
        "Read ``one ` character`` for the interface.\nThe runner's manifest identifies the source.\n",
        "| Command | Description |\n| --- | --- |\n| `$CMD -rf /` | The engine's limit applies. |\n",
    ],
)
def test_prose_ownership_keeps_unmatched_inline_and_runtime_commands_partial(content: str) -> None:
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": content + "\n" + _OBSERVATIONS}},
        [tm],
    )

    assert any(
        event["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
        for event in result["inspection_ledger"]
    )


def test_inline_line_ownership_preserves_later_destructive_command() -> None:
    content = (
        "Read `guide.md` for the interface.\nThe runner's manifest identifies the source.\n\n"
        "```bash\nrm -rf /\n```\n" + _OBSERVATIONS
    )

    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": content}}, [tm]
    )

    assert any(finding.rule_id == "TM1" for finding in result["findings"])


_NESTED_SHELL_QUOTES = 'echo "entry: $item => $(readlink -f "$item")"\n'


@pytest.mark.parametrize("fence", ["```", "~~~"])
@pytest.mark.parametrize("language", ["bash", "sh", "shell"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_closed_shell_fence_owns_nested_quotes(fence: str, language: str, newline: str) -> None:
    content = (
        f"## Reader\n\n{fence}{language}\n" + _NESTED_SHELL_QUOTES + f"{fence}\n\n" + _OBSERVATIONS
    ).replace("\n", newline)

    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": content}}, [tm]
    )

    assert result["findings"] == []
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize(
    "opening,closing",
    [
        ("```", "```"),
        ("```text", "```"),
        ("```python", "```"),
        ("```bash title=example", "```"),
        ("```bash", ""),
        ("```bash", "~~~"),
        ("````bash", "```"),
        ("\\```bash", "```"),
        ("```bash", "\\```"),
        ("- ```bash", "  ```"),
        ("> ```bash", "> ```"),
    ],
)
def test_unproven_fences_do_not_grant_shell_quote_ownership(opening: str, closing: str) -> None:
    content = opening + "\n" + _NESTED_SHELL_QUOTES + closing + "\n" + _OBSERVATIONS
    ranges: list[tuple[int, int]] = []

    projected = tm._markdown_shell_text(content, lambda: None, shell_fence_ranges=ranges)

    assert len(projected) == len(content)
    assert ranges == []


def test_fence_ownership_requires_complete_context_and_preserves_offsets() -> None:
    prefix = "## Reader\n\n   ```bash\n"
    content = prefix + _NESTED_SHELL_QUOTES + "   ````\n" + _OBSERVATIONS
    ranges: list[tuple[int, int]] = []
    projected = tm._markdown_shell_text(content, lambda: None, shell_fence_ranges=ranges)

    assert ranges == [(len(prefix), len(prefix) + len(_NESTED_SHELL_QUOTES))]
    start, end = ranges[0]
    assert projected[start:end] == content[start:end] == _NESTED_SHELL_QUOTES
    assert len(projected) == len(content)

    ranges.clear()
    tm._markdown_shell_text(
        content, lambda: None, complete_context=False, shell_fence_ranges=ranges
    )
    assert ranges == []


@pytest.mark.parametrize("position", ["before", "inside", "after"])
@pytest.mark.parametrize("command", ["$CMD -rf /", 'echo "$($CMD -rf /)"', "rm -rf /"])
def test_shell_fence_quote_ownership_preserves_destructive_checks(
    position: str, command: str
) -> None:
    inside = _NESTED_SHELL_QUOTES + (command + "\n" if position == "inside" else "")
    content = "```bash\n" + inside + "```\n" + _OBSERVATIONS
    if position == "before":
        content = command + "\n" + content
    elif position == "after":
        content += command + "\n"

    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": content}}, [tm]
    )

    if command == "rm -rf /":
        assert any(finding.rule_id == "TM1" for finding in result["findings"])
    else:
        event = result["inspection_ledger"][0]
        assert event["outcome"] is LedgerOutcome.PARTIAL
        assert event["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


def test_closed_fence_with_unclosed_shell_quote_stays_partial() -> None:
    content = '```bash\necho "unfinished\n```\n' + _OBSERVATIONS

    assert tm.has_bounded_parse_exhaustion(content, lambda: None, file_type="markdown")


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
    assert coverage["total_components"] == coverage["fully_inspected_files"] == 2
    assert len(coverage["references"]) == 1
    reference = coverage["references"][0]
    assert reference["target_path"] == "guide.md"
    assert reference["status"] == "resolved"
    assert reference["disposition"] == "analyzed"
    assert (reference["source_path"], reference["line"], reference["column"]) == ("SKILL.md", 6, 28)


@pytest.mark.parametrize(
    "prefix",
    [
        "- ",
        "+ ",
        "* ",
        "1. ",
        "2) ",
        "-\t",
        "-    ",
        "- - ",
        "- Details\n  - ",
        "1. Details\n   1. ",
        "- Details\n  ",
    ],
)
@pytest.mark.parametrize("long", [False, True], ids=["short", "long"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_list_prose_has_complete_analyzer_coverage(prefix: str, long: bool, newline: str) -> None:
    content = prefix + (_APOSTROPHE if long else _APOSTROPHE.split("\n\n")[0] + "\n")
    content = content.replace("\n", newline)
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": content}}, [tm]
    )
    assert result["findings"] == []
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_list_prose_preserves_unicode_offsets_and_later_findings(
    tmp_path: Path, newline: str
) -> None:
    content = "# Résumé — インターフェース\n\n- " + _APOSTROPHE
    command = "- rm -rf /\n"
    content += command + command
    # Reference the same file twice to exercise canonical artifact accounting.
    manifest = _MANIFEST + "Read [the same guide](guide.md) again.\n"
    reports = []
    for guide in (content, content.replace("doesn't", "does not")):
        (tmp_path / "SKILL.md").write_bytes(manifest.encode())
        (tmp_path / "guide.md").write_bytes(guide.replace("\n", newline).encode())
        cli = CliRunner().invoke(
            app,
            ["scan", str(tmp_path), "--no-llm", "--format", "json", "--fail-on-incomplete"],
        )
        assert cli.exit_code in (0, 1), cli.output
        report = json.loads(cli.output)
        coverage = report["analysis_completeness"]
        assert coverage["is_complete"] is True
        assert coverage["coverage_percent"] == 100.0
        assert coverage["total_components"] == coverage["fully_inspected_files"] == 2
        assert coverage["ledger_exceptions"] == []
        assert report["risk_assessment"]["recommendation"] != "SAFE"
        assert not any(issue["id"] == "AE1" for issue in report["issues"])
        tm1_issues = [issue for issue in report["issues"] if issue["id"] == "TM1"]
        # Compaction retains one match identity while reports preserve each
        # occurrence's original identifier and source-specific snippet.
        assert len(tm1_issues) == 2
        assert all(issue["match_fingerprint"] for issue in tm1_issues)
        assert len({issue["match_fingerprint"] for issue in tm1_issues}) == 1
        assert len({issue["finding_id"] for issue in tm1_issues}) == 2
        assert len({issue["code_snippet"] for issue in tm1_issues}) == 2
        locations = {
            (location["file"], location["start_line"])
            for issue in report["issues"]
            if issue["id"] == "TM1"
            for location in [issue["location"], *issue.get("occurrences", [])]
        }
        first_line = content[: content.index(command)].count("\n") + 1
        assert locations == {("guide.md", first_line), ("guide.md", first_line + 1)}
        reports.append(report)
    # Prose punctuation cannot change finding deduplication or threat scoring.
    assert reports[0]["risk_assessment"] == reports[1]["risk_assessment"]
    assert len(reports[0]["issues"]) == len(reports[1]["issues"])


@pytest.mark.parametrize("prefix", ["- ", "1. ", "- Details\n  - ", "> - "])
@pytest.mark.parametrize("command", ["$($CMD) -rf /", "$($(resolve_tool)/printf %s rm) -rf /"])
@pytest.mark.parametrize("before", [False, True], ids=["after-prose", "before-prose"])
def test_list_prose_cannot_hide_runtime_commands(prefix: str, command: str, before: bool) -> None:
    prose = "- " + _APOSTROPHE
    shell = prefix + command + "\n\n"
    content = shell + prose if before else prose + shell
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": content}}, [tm]
    )
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize(
    "content",
    [
        "- ```sh\n  " + _APOSTROPHE.replace("\n", "\n  ") + "\n  ```\n",
        "1. ~~~sh\n   " + _APOSTROPHE.replace("\n", "\n   ") + "\n   ~~~\n",
        "-     " + _APOSTROPHE.replace("\n", "\n      "),
        "- Details\n\n      " + _APOSTROPHE.replace("\n", "\n      "),
        "> - " + _APOSTROPHE.replace("\n", "\n>   "),
        '- bash -c "\n- ' + _APOSTROPHE + '"\n',
        "- $(\n- " + _APOSTROPHE + "') -rf /\n",
        "- r'\n- " + _APOSTROPHE + "'m -rf /\n",
        "- `\n- " + _APOSTROPHE + "`\n",
    ],
)
def test_list_context_does_not_rewrite_code_or_shell_owners(content: str) -> None:
    projected = tm._markdown_shell_text(content, lambda: None)
    assert len(projected) == len(content)
    apostrophe = content.index("doesn't") + len("doesn")
    assert projected[apostrophe] == "'"


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


@pytest.mark.parametrize(
    "line",
    [
        "$CMD reads each reference's probe mode.",
        "Each reference's `probe` mode.",
        'Each reference\'s "probe" mode.',
        "Each reference's probe mode, as in echo 'it'.",
        "Each reference's 'tools' mode.",
        "Each reference's probe mode \\",
        "Run the probe \\\nfor each reference's mode.",
        "$CMD \u2028reads each reference's probe mode.",
        "## Each reference's probe mode",
    ],
)
def test_possessive_beside_shell_syntax_keeps_its_quote(line: str) -> None:
    content = line + "\n\n" + _OBSERVATIONS
    projected = tm._markdown_shell_text(content, lambda: None)
    apostrophe = content.index("reference's") + len("reference")
    assert projected[apostrophe] == "'"


def test_complete_inline_code_before_prose_apostrophe_does_not_span_later_fence() -> None:
    content = (
        "- `${flow_path}` is replaced with the configuration file's directory.\n\n"
        "```\n"
        "copy.py -src <input> -dest <build_dir>/output.vc\n"
        "compile (uses -file output.vc)\n"
        "```\n"
    )

    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": content}}, [tm]
    )

    assert result["findings"] == []
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


def test_complete_inline_code_does_not_hide_runtime_selected_command() -> None:
    content = "Run `$($CMD) -rf /`; then read the reference's notes.\n"

    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": content}}, [tm]
    )

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize("separator", [" ", " \u2028"], ids=["space", "line-separator"])
def test_possessive_cannot_hide_runtime_command_operands(separator: str) -> None:
    # Read as shell, the apostrophe joins the next line to $CMD's operands.
    # A Unicode line separator does not end the shell line.
    content = f"$CMD{separator}reads each reference's\nprobe' -rf /\n\n" + _OBSERVATIONS
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": content}}, [tm]
    )
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize("separator", [" ", " \u2028"], ids=["space", "line-separator"])
def test_inline_runtime_command_cannot_hide_later_operands(separator: str) -> None:
    # Markdown owns the backtick delimiters, but the shell projection retains
    # the runtime command and must not detach its quote-joined operands.
    content = f"`$CMD`{separator}reads each reference's\nprobe' -rf /\n\n" + _OBSERVATIONS
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": content}}, [tm]
    )
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize(
    ("prefix", "partial"),
    [("", True), ("- ", False)],
    ids=["runtime-command", "list-marker-control"],
)
@pytest.mark.parametrize("padding", ["pad\n" * 2300, "pad " * 2300], ids=["lines", "words"])
def test_inline_runtime_command_operands_beyond_evidence_bound(
    prefix: str, partial: bool, padding: str
) -> None:
    # The closing quote and operands lie beyond the guard's bounded lookahead,
    # which cannot prove that they are absent.
    content = f"{prefix}`$CMD` reads each reference's\n{padding}probe' -rf /\n\n" + _OBSERVATIONS
    assert content.index("-rf /") > tm._ROOT_GLOB_COMMAND_CHARS
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": content}}, [tm]
    )
    if partial:
        assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
        assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
    else:
        assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize(
    "inline_command",
    [
        "$CMD",
        "echo 'ready; still data' $CMD",
        "echo 'ready | still data' $CMD",
        r"echo ready\; $CMD",
        r"echo ready\| $CMD",
    ],
)
def test_list_marker_keeps_inline_runtime_command_as_an_argument(inline_command: str) -> None:
    # The proven Markdown marker is the shell projection's command word, so
    # $CMD is only its argument and cannot own the later recursive operands.
    content = f"- `{inline_command}` reads each reference's\nprobe' -rf /\n\n" + _OBSERVATIONS
    projected = tm._markdown_shell_text(content, lambda: None, word_apostrophes=True)
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": content}}, [tm]
    )
    assert "$CMD" in projected
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert result["findings"] == []


@pytest.mark.parametrize(
    ("marker", "owns_command"),
    [("- ", True), ("+ ", True), ("1. ", True), ("1) ", True), ("* ", False), ("- * ", True)],
)
def test_only_inert_list_marker_owns_the_command_position(marker: str, owns_command: bool) -> None:
    # ``*`` is a shell glob that can expand to an arbitrary command word, so it
    # keeps the runtime guard; inert markers own the command word.
    content = f"{marker}`$CMD` reads each reference's\nprobe' -rf /\n\n" + _OBSERVATIONS
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": content}}, [tm]
    )
    if owns_command:
        assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    else:
        assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
        assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize(
    "inline_command",
    [
        "; $CMD",
        "echo ready; $CMD",
        "echo ready | $CMD",
        "echo ready && $CMD",
        "echo ready || $CMD",
        "echo ready & $CMD",
    ],
)
def test_list_inline_control_operator_retains_runtime_guard(inline_command: str) -> None:
    content = f"- `{inline_command}` reads each reference's\nprobe' -rf /\n\n" + _OBSERVATIONS
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": content}}, [tm]
    )
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize(
    "prefix",
    [
        "- ; ",
        "- `echo ready`; ",
        "- `echo ready` | ",
        "- `echo ready;` ",
        "- `echo ready |` ",
    ],
    ids=[
        "outside-span",
        "between-spans-semicolon",
        "between-spans-pipeline",
        "prior-span-semicolon",
        "prior-span-pipeline",
    ],
)
def test_list_control_before_inline_command_retains_runtime_guard(prefix: str) -> None:
    content = f"{prefix}`$CMD` reads each reference's\nprobe' -rf /\n\n" + _OBSERVATIONS
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": content}}, [tm]
    )
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize(
    "content", [_APOSTROPHE, _LAYOUT, "- " + _APOSTROPHE, _POSSESSIVE, _PLURAL_POSSESSIVE]
)
@pytest.mark.parametrize(
    "command,partial",
    [
        ("rm -rf /", False),
        ("r'm' -rf /", False),
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


@pytest.mark.parametrize("content", [_APOSTROPHE, "- " + _APOSTROPHE])
def test_prose_ownership_requires_complete_context(content: str) -> None:
    assert tm.has_bounded_parse_exhaustion(
        content, lambda: None, file_type="markdown", complete_context=False
    )
    assert tm.has_bounded_parse_exhaustion(content, lambda: None, file_type="shell")


def test_prose_ownership_preserves_cancellation() -> None:
    checks = 0

    def cancel() -> None:
        nonlocal checks
        checks += 1
        if checks == 20:
            raise TimeoutError("cancelled")

    with pytest.raises(TimeoutError, match="cancelled"):
        tm.has_bounded_parse_exhaustion(_APOSTROPHE * 100, cancel, file_type="markdown")


@pytest.mark.parametrize("prefix", ["", "- "])
def test_prose_ownership_checks_cancellation_after_word_jumps(
    monkeypatch: pytest.MonkeyPatch,
    prefix: str,
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
    content += "\n\n" + prefix + "The interface doesn't match the spec.\n"
    with pytest.raises(TimeoutError, match="cancelled"):
        tm._markdown_shell_text(content, cancel)
