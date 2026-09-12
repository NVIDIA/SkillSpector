# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An active GFM table distinguishes body text from interrupting blocks."""

from __future__ import annotations

from pathlib import Path

import pytest

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm_module
from skillspector.nodes.analyzers import static_runner
from tests.nodes.analyzers.test_documentation_reconstruction import (
    successful_llm_transport as successful_llm_transport,
)
from tests.nodes.analyzers.test_markdown_table_boundaries import (
    _assert_public_partial,
    _public_results,
)


def _source(middle: str) -> str:
    return "| Example |\n| --- |\n" + middle + "\n| `$(resolve_tool).example |\n| ` -rf / |\n"


# cmark-gfm 499789b49373bfa045d0e7547e5ee63444c77bca, --extension table:
# Setext-only markers have no paragraph to underline while a table is active.
# A thematic break or an empty list item does start a new block instead.
# https://github.github.com/gfm/#tables-extension-
_BODY_ROWS = [
    pytest.param("=", id="equals"),
    pytest.param("===", id="equals-run"),
    pytest.param("--", id="two-hyphens"),
    pytest.param("  =  ", id="padded-equals"),
    pytest.param("  --  ", id="padded-two-hyphens"),
    pytest.param("= =", id="spaced-equals-control"),
]
_INTERRUPTIONS = [
    pytest.param("---", id="thematic-break"),
    pytest.param("-", id="empty-dash-item"),
    pytest.param("*", id="empty-star-item"),
    pytest.param("+", id="empty-plus-item"),
    pytest.param("1.", id="empty-ordered-item"),
]


def _scan(source: str) -> dict:
    return static_runner.run_static_patterns_with_ledger(
        {"components": ["references/table.md"], "file_cache": {"references/table.md": source}},
        [tm_module],
    )


@pytest.mark.parametrize("middle", _BODY_ROWS)
def test_setext_only_text_keeps_following_table_rows_separate(middle: str) -> None:
    source = _source(middle)
    projected = tm_module._markdown_shell_text(source, lambda: None)
    result = _scan(source)

    # The final two backticks are literal text in different rendered cells.
    assert projected == source
    assert any(
        row["outcome"] is LedgerOutcome.PARTIAL
        and row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
        and row["analyzer_id"] == "static_patterns_tool_misuse"
        and row["path"] == "references/table.md"
        for row in result["inspection_ledger"]
    )


@pytest.mark.parametrize("middle", _INTERRUPTIONS)
def test_real_block_interruptions_restore_paragraph_inline_ownership(middle: str) -> None:
    source = _source(middle)
    projected = tm_module._markdown_shell_text(source, lambda: None)
    result = _scan(source)

    # After the interrupting block, these lines form one paragraph code span.
    assert projected == source.replace("`", " ")
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize(
    "marker,complete",
    [pytest.param("=", False, id="table-row"), pytest.param("+", True, id="empty-list")],
)
@pytest.mark.parametrize("use_llm", [False, True], ids=["no-llm", "llm"])
def test_table_body_boundaries_through_cli_and_mcp(
    tmp_path: Path,
    marker: str,
    complete: bool,
    use_llm: bool,
    successful_llm_transport: list[str],
) -> None:
    exit_code, mcp, reports = _public_results(
        tmp_path, _source(marker), use_llm, False, successful_llm_transport
    )
    assert exit_code == (0 if complete else 1)
    assert mcp["safe_to_install"] is complete
    assert mcp["llm_used"] is use_llm
    for report in reports:
        assert report["analysis_completeness"]["is_complete"] is complete
        if complete:
            assert report["analysis_completeness"]["coverage_percent"] == 100.0
            assert report["risk_assessment"]["recommendation"] == "SAFE"
            assert not any(issue["id"] == "AE1" for issue in report["issues"])
        else:
            assert report["risk_assessment"]["recommendation"] != "SAFE"
            _assert_public_partial(report, False)


def _header_source(preamble: str) -> str:
    return preamble + "\n| --- |\n| `$(resolve_tool).example |\n| ` -rf / |\n"


@pytest.mark.parametrize("header", ["=", "===", "--"])
def test_setext_looking_text_at_document_start_can_be_a_table_header(header: str) -> None:
    source = _header_source(header)
    projected = tm_module._markdown_shell_text(source, lambda: None)
    result = _scan(source)

    # Without a preceding paragraph these lines are not Setext underlines.
    # cmark-gfm recognizes them as headers paired with the next delimiter row.
    assert projected == source
    assert any(
        row["outcome"] is LedgerOutcome.PARTIAL
        and row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
        and row["analyzer_id"] == "static_patterns_tool_misuse"
        and row["path"] == "references/table.md"
        for row in result["inspection_ledger"]
    )


@pytest.mark.parametrize(
    "preamble",
    [
        pytest.param("---", id="thematic-break"),
        pytest.param("-", id="empty-dash-item"),
        pytest.param("*", id="empty-star-item"),
        pytest.param("+", id="empty-plus-item"),
        pytest.param("1.", id="empty-ordered-item"),
        pytest.param("Intro\n===", id="actual-setext-heading"),
    ],
)
def test_real_blocks_cannot_be_promoted_to_table_headers(preamble: str) -> None:
    source = _header_source(preamble)
    projected = tm_module._markdown_shell_text(source, lambda: None)
    result = _scan(source)

    assert projected == source.replace("`", " ")
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize(
    "marker,complete",
    [
        pytest.param("=", False, id="table-row"),
        pytest.param("+", True, id="empty-list"),
        pytest.param("+\n=", False, id="header-after-empty-list"),
    ],
)
@pytest.mark.parametrize("use_llm", [False, True], ids=["no-llm", "llm"])
def test_table_header_boundaries_through_cli_and_mcp(
    tmp_path: Path,
    marker: str,
    complete: bool,
    use_llm: bool,
    successful_llm_transport: list[str],
) -> None:
    exit_code, mcp, reports = _public_results(
        tmp_path, _header_source(marker), use_llm, False, successful_llm_transport
    )
    assert exit_code == (0 if complete else 1)
    assert mcp["safe_to_install"] is complete
    assert mcp["llm_used"] is use_llm
    for report in reports:
        assert report["analysis_completeness"]["is_complete"] is complete
        if complete:
            assert report["analysis_completeness"]["coverage_percent"] == 100.0
            assert report["risk_assessment"]["recommendation"] == "SAFE"
            assert not any(issue["id"] == "AE1" for issue in report["issues"])
        else:
            assert report["risk_assessment"]["recommendation"] != "SAFE"
            _assert_public_partial(report, False)


@pytest.mark.parametrize(
    "prefix",
    [
        pytest.param("*", id="star"),
        pytest.param("+", id="plus"),
        pytest.param("1.", id="ordered"),
        pytest.param("+\n", id="blank-line-control"),
    ],
)
def test_empty_list_item_does_not_open_a_paragraph_before_table_header(prefix: str) -> None:
    source = _header_source(prefix + "\n=")
    projected = tm_module._markdown_shell_text(source, lambda: None)
    result = _scan(source)

    # An empty item has no paragraph for the next '=' to underline. The
    # following delimiter therefore establishes a new table header.
    assert projected == source
    assert any(
        row["outcome"] is LedgerOutcome.PARTIAL
        and row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
        and row["analyzer_id"] == "static_patterns_tool_misuse"
        and row["path"] == "references/table.md"
        for row in result["inspection_ledger"]
    )
