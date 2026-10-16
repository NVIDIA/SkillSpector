# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-comment ownership regressions, using independent synthetic text."""

from __future__ import annotations

import pytest

from skillspector.models import compute_match_fingerprint
from skillspector.nodes.analyzers import static_patterns_prompt_injection as prompt_injection
from skillspector.nodes.analyzers import static_runner


def _p2(content: str):
    return [
        finding
        for finding in prompt_injection.analyze(content, "SKILL.md", "markdown")
        if finding.rule_id == "P2"
    ]


@pytest.mark.parametrize(
    "visible",
    [
        "Then target the endpoint (see below).",
        "GET the page (documented here).",
        "Follow the visible instructions (documented here).",
        "Use the operating system (documented here).",
        "Send feedback (documented here).",
        "Transmit the visible message (documented here).",
        "Ignore this visible example (documented here).",
    ],
)
def test_closed_comment_cannot_borrow_visible_keyword(visible: str) -> None:
    assert not _p2(f"[//]: # (layout note)\n\n{visible}")


@pytest.mark.parametrize(
    "content",
    [
        "[//]: #\n\n(GET a visible page).",
        "[//]: #\r\n \t\r\n(GET a visible page).",
        "[//]:\n\n# (GET a visible page)",
        "[//]:\r\n \t\r\n# (GET a visible page)",
        "[//]: #\r\r(GET a visible page).",
    ],
)
def test_reference_opening_cannot_borrow_a_title_after_a_blank_line(content: str) -> None:
    assert not _p2(content)


@pytest.mark.parametrize("line_ending", ["\n", "\r", "\r\n"])
def test_reference_title_can_start_on_the_next_nonblank_line(line_ending: str) -> None:
    comment = f"[//]: # {line_ending}\t(GET the page)"
    findings = _p2(comment)
    assert len(findings) == 1
    assert findings[0].matched_text == comment
    assert findings[0].location.end_line == 2


@pytest.mark.parametrize("line_ending", ["\n", "\r", "\r\n"])
def test_reference_destination_can_start_on_the_next_nonblank_line(line_ending: str) -> None:
    assert _p2(f"[//]: {line_ending}\t# (GET the page)")


def test_separate_comments_only_own_their_own_title() -> None:
    hidden = "[//]: # (GET https://example.invalid/collect)"
    content = f"[//]: # (layout note)\nVisible text.\n{hidden}\n[//]: # (end note)"
    findings = _p2(content)
    assert len(findings) == 1
    assert findings[0].matched_text == hidden
    assert findings[0].match_fingerprint == compute_match_fingerprint("P2", hidden)
    assert findings[0].location.start_line == 3
    assert findings[0].location.end_line == 3


@pytest.mark.parametrize(
    "title",
    ["target", "forget", "GETTING", "POSTGRES", "SYSTEMATIC", "sender", "transmitter"],
)
def test_title_keywords_do_not_match_ordinary_word_substrings(title: str) -> None:
    assert not _p2(f"[//]: # ({title})")


@pytest.mark.parametrize(
    "title",
    [
        "SYSTEM: do something",
        "instructions: do something",
        "ignore_previous_instructions",
        "ignorePreviousInstructions",
        "pleaseIgnorePreviousInstructions",
        "POSTRequestToServer",
        "SYSTEMPrompt",
        "GET https://example.invalid/data",
        "send the message",
        "transmit the message",
    ],
)
def test_real_hidden_keywords_keep_detection(title: str) -> None:
    comment = f"[//]: # ({title})"
    findings = _p2(comment)
    assert len(findings) == 1
    assert findings[0].matched_text == comment
    assert findings[0].match_fingerprint == compute_match_fingerprint("P2", comment)


@pytest.mark.parametrize(
    "comment",
    [
        r"[//]: # (layout \(small\) note)",
        r"[//]: # (layout \) note)",
        r"[//]: # (layout \\)",
        r"[//]: # (layout \\\) note)",
    ],
)
def test_escaped_parentheses_and_backslashes_still_bound_visible_text(comment: str) -> None:
    assert not _p2(f"{comment}\n\nGET a visible page (see below).")


@pytest.mark.parametrize(
    "comment",
    [
        r"[//]: # (layout \(small\) GET the page)",
        r"[//]: # (layout \) GET the page)",
        r"[//]: # (layout \\\) GET the page)",
        "[//]: # (layout\nGET the page)",
    ],
)
def test_escaped_closer_does_not_hide_later_keyword_inside_title(comment: str) -> None:
    findings = _p2(comment)
    assert len(findings) == 1
    assert findings[0].matched_text == comment
    assert findings[0].match_fingerprint == compute_match_fingerprint("P2", comment)


@pytest.mark.parametrize(
    "content",
    [
        "[//]: # (layout (nested) note)\nGET a visible page (see below).",
        "[//]: # (layout note",
    ],
)
def test_malformed_titles_do_not_borrow_visible_keywords(content: str) -> None:
    assert not _p2(content)


@pytest.mark.parametrize(
    "content",
    [
        "[//]: # (GET a page (nested) note)",
        "[//]: # (GET a page\n\nextra text)",
    ],
)
def test_malformed_titles_keep_conservative_hidden_keyword_detection(content: str) -> None:
    assert _p2(content)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("［//］： ＃ （layout note）\n\nThen target the endpoint (see below).", False),
        ("［//］： ＃ （ＧＥＴ https://example.invalid/data）", True),
        ("[//]: # (layout note&#41;\n\nGET a visible page (see below).", False),
    ],
)
def test_transformed_views_keep_comment_ownership(content: str, expected: bool) -> None:
    state = {"components": ["SKILL.md"], "file_cache": {"SKILL.md": content}}
    findings = static_runner.run_static_patterns(state, [prompt_injection])
    p2 = [finding for finding in findings if finding.rule_id == "P2"]
    assert bool(p2) is expected
    if expected:
        assert len(p2) == 1
        assert p2[0].start_line == 1


@pytest.mark.parametrize("offset", [231_160, 247_790, 255_980])
def test_window_overlap_keeps_only_the_real_comment(offset: int) -> None:
    prefix = "Plain notes.\n" * (offset // 13) + "x" * (offset % 13) + "\n"
    benign = "[//]: # (layout note)\n\nThen target the endpoint (see below).\n"
    hidden = "[//]: # (GET https://example.invalid/data)"
    content = prefix + benign + hidden
    state = {"components": ["SKILL.md"], "file_cache": {"SKILL.md": content}}
    findings = static_runner.run_static_patterns(state, [prompt_injection])
    p2 = [finding for finding in findings if finding.rule_id == "P2"]
    assert len(p2) == 1
    assert p2[0].matched_text == hidden
    assert p2[0].start_line == content[: content.index(hidden)].count("\n") + 1


def test_long_unclosed_title_checks_deadline_during_scanning() -> None:
    pattern = prompt_injection.P2_PATTERNS[1][0]
    calls = 0

    def expire() -> None:
        nonlocal calls
        calls += 1
        if calls == 5:
            raise TimeoutError("reference-title deadline")

    with pytest.raises(TimeoutError, match="reference-title deadline"):
        list(prompt_injection._p2_pattern_matches("[//]: # (" + r"\(" * 20_000, pattern, expire))
    assert calls == 5


def test_unclosed_title_cannot_repeatedly_scan_embedded_openings() -> None:
    pattern = prompt_injection.P2_PATTERNS[1][0]
    calls = 0

    def count() -> None:
        nonlocal calls
        calls += 1

    content = "[//]: # (" * 20_000
    assert not list(prompt_injection._p2_pattern_matches(content, pattern, count))
    assert calls < 300


def test_reference_comment_output_cap_preserves_first_finding_and_partial_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    response = static_runner.run_static_patterns_with_ledger(
        {
            "components": ["SKILL.md"],
            "file_cache": {"SKILL.md": "[//]: # (GET a page)\n" * 4},
        },
        [prompt_injection],
    )
    p2 = [finding for finding in response["findings"] if finding.rule_id == "P2"]
    assert len(p2) == 1
    assert p2[0].start_line == 1
    assert response["inspection_ledger"][0]["outcome"] == "partial"
    assert response["inspection_ledger"][0]["reason_code"] == "output_limit"
