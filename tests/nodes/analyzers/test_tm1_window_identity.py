# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TM1 ownership and projection contracts split from PR #497."""

from __future__ import annotations

import pytest

import skillspector.artifacts as artifacts_module
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm_module


def _run(content: str, path: str = "run.py") -> dict:
    return tm_module.node({"components": [path], "file_cache": {path: content}})


def _tm1(content: str, path: str = "run.py") -> list:
    return [finding for finding in _run(content, path)["findings"] if finding.rule_id == "TM1"]


@pytest.mark.parametrize(
    "source",
    [
        "plain subprocess.run(command, shell=True)",
        "ig\u00adn\u03bfre and systeｍ",
        "word\u200b\u200c boundary",
        "\N{BLACK SUN WITH RAYS}\N{VARIATION SELECTOR-16} emoji",
        "left\u0085\u0600right",
        "prefix " + "ﷺ" * 100 + " suffix",
        "a" + "\u200b" * 300 + "b",
    ],
)
@pytest.mark.parametrize("max_chars", [0, 1, 7, 31, 200])
def test_normalized_security_prefix_matches_full_projection(
    source: str,
    max_chars: int,
) -> None:
    assert hasattr(artifacts_module, "normalized_security_prefix")
    assert (
        artifacts_module.normalized_security_prefix(source, max_chars)
        == artifacts_module.normalized_security_view(source).text[:max_chars]
    )


@pytest.mark.parametrize(
    "source",
    [
        "plain source",
        "😀" * 20,
        "Cafe\u0301",
        "☀️",
        "\u034f",
        "\u0085",
        "\u0600",
        "\u200b",
        "ｓｕｂｐｒｏｃｅｓｓ",
        "shell\x00=True",
        "i g n o r e previous instructions.",
        "i g n o r e previous instructions.\ufffd",
        "i-g-n-o-r-e previous instructions",
    ],
)
def test_derived_security_view_predicate_matches_materialized_views(source: str) -> None:
    assert hasattr(artifacts_module, "_has_derived_security_view")
    assert artifacts_module._has_derived_security_view(source) is (
        len(artifacts_module.security_text_views(source)) > 1
    )


def test_bound_call_uses_runner_logical_line_coordinates() -> None:
    direct = _tm1("enabled = True\n\fsubprocess.run(command, shell=True)\n")
    bound = _tm1("enabled = True\n\fsubprocess.run(command, shell=enabled)\n")

    assert len(direct) == len(bound) == 1
    assert (direct[0].start_line, bound[0].start_line) == (3, 3)
    assert bound[0].end_line == 3


def test_embedded_direct_literal_text_does_not_duplicate_bound_call() -> None:
    findings = _tm1("enabled = True\nsubprocess.run('shell=True', shell=enabled)\n")

    assert len(findings) == 1
    assert "shell=enabled" in (findings[0].matched_text or "")


def test_normalized_expansion_slice_maps_back_to_ast_call_start() -> None:
    confusable_name = "tru\N{CYRILLIC SMALL LETTER IE}_value"
    prefix = "#" + "ﷺ" * 20_000 + "\n"
    result = _run(
        prefix
        + f"{confusable_name} = True\n"
        + f"subprocess.run(command, shell={confusable_name})\n",
        "expanded.py",
    )
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert len(findings) == 1
    assert findings[0].start_line == 3


def test_normalized_continuity_view_owns_its_bound_call_once() -> None:
    confusable_name = "tru\N{CYRILLIC SMALL LETTER IE}_value"
    payload = " " * 256_000
    result = _run(
        f"{confusable_name} = True\nsubprocess.run({payload!r}, shell={confusable_name})\n",
        "wide.py",
    )
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert len(findings) == 1


def test_distinct_bound_calls_on_one_line_are_not_deduplicated() -> None:
    findings = _tm1(
        "enabled = True\n"
        "pi_π = 1; subprocess.run('one', shell=enabled); "
        "subprocess.run('two', shell=enabled)\n"
    )

    assert len(findings) == 2
    assert {finding.matched_text for finding in findings} == {
        "subprocess.run('one', shell=enabled)",
        "subprocess.run('two', shell=enabled)",
    }


def test_long_direct_calls_with_shared_preview_keep_distinct_identity() -> None:
    payload = "x" * 240
    first_call = f'subprocess.run("{payload}A", shell=True)'
    second_call = f'subprocess.run("{payload}B", shell=True)'
    findings = _tm1(f"first = {first_call}; second = {second_call}\n")

    assert len(findings) == 2
    assert findings[0].fingerprint() != findings[1].fingerprint()


def test_long_shared_preview_identity_is_cap_stable_and_matches_bound_call(monkeypatch) -> None:
    payload = "x" * 240
    direct_source = (
        f'subprocess.run("{payload}A", shell=True); subprocess.run("{payload}B", shell=True)\n'
    )
    bound_source = (
        "enabled = True\n"
        f'subprocess.run("{payload}A", shell=enabled); '
        f'subprocess.run("{payload}B", shell=enabled)\n'
    )

    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    direct_capped = _tm1(direct_source)
    bound_capped = _tm1(bound_source)
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 2)
    direct_complete = _tm1(direct_source)
    bound_complete = _tm1(bound_source)

    assert len(direct_capped) == len(bound_capped) == 1
    assert len(direct_complete) == len(bound_complete) == 2
    assert direct_capped[0].fingerprint() == direct_complete[0].fingerprint()
    assert bound_capped[0].fingerprint() == bound_complete[0].fingerprint()
    assert [finding.fingerprint() for finding in direct_complete] == [
        finding.fingerprint() for finding in bound_complete
    ]


def test_output_cap_keeps_first_mixed_owner_in_source_order(monkeypatch) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    result = _run(
        "subprocess.run('first', shell=True)\n"
        "enabled = True\n"
        "subprocess.run('second', shell=enabled)\n"
    )
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert [finding.start_line for finding in findings] == [1]
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL


def test_output_cap_orders_bound_and_ordinary_tm1_by_source(monkeypatch) -> None:
    content = '# --skip-validation\nenabled = True\nsubprocess.run("later", shell=enabled)\n'

    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 2)
    complete = _tm1(content)
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    capped_result = _run(content)
    capped = [finding for finding in capped_result["findings"] if finding.rule_id == "TM1"]

    assert [finding.start_line for finding in complete] == [1, 3]
    assert [finding.start_line for finding in capped] == [1]
    assert capped_result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert capped_result["inspection_ledger"][0]["reason_code"] is LedgerReason.OUTPUT_LIMIT


@pytest.mark.parametrize("cap", [1, 2])
def test_same_line_bound_duplicates_match_direct_output_budget(monkeypatch, cap: int) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
    direct = _run('subprocess.run("x", shell=True); subprocess.run("x", shell=True)\n')
    bound = _run(
        'enabled = True\nsubprocess.run("x", shell=enabled); subprocess.run("x", shell=enabled)\n'
    )
    direct_findings = [finding for finding in direct["findings"] if finding.rule_id == "TM1"]
    bound_findings = [finding for finding in bound["findings"] if finding.rule_id == "TM1"]

    assert len(direct_findings) == len(bound_findings) == 1
    assert direct_findings[0].fingerprint() == bound_findings[0].fingerprint()
    assert direct["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert bound["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize("bound_name", ["a", "true_value"])
def test_bound_fingerprint_matches_direct_literal(bound_name: str) -> None:
    direct = _tm1("subprocess.run(command, shell=True, capture_output=True)\n")
    bound = _tm1(f"{bound_name} = True\nsubprocess.run(command, shell={bound_name}, text=True)\n")

    assert len(direct) == len(bound) == 1
    assert bound[0].fingerprint() == direct[0].fingerprint()


def test_cross_window_qualified_and_bare_popen_share_one_budget_owner(monkeypatch) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    content = "subprocess." + "\u200b" * 256_000 + "Popen(command, shell=True)\n"
    result = _run(content, "guide.md")
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert len(findings) == 1
    assert findings[0].matched_text == "subprocess.Popen(command, shell=True"
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize("ignored", ["\u200b", "\ufffd"])
def test_output_limit_finalizes_retained_cross_window_popen(monkeypatch, ignored: str) -> None:
    expected = _tm1("subprocess.Popen(command, shell=True)\n", "guide.md")[0].fingerprint()
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    content = (
        "subprocess."
        + ignored * 256_000
        + "Popen(command, shell=True)\n"
        + "subprocess.run(command_0, shell=True)\n"
    )

    result = _run(content, "guide.md")
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert len(findings) == 1
    assert findings[0].matched_text == "subprocess.Popen(command, shell=True"
    assert findings[0].fingerprint() == expected
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.OUTPUT_LIMIT
