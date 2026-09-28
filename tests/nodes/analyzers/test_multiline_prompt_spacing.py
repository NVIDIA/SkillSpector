# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inert multiline prompt data must not acquire a complete, safe verdict."""

import re
from dataclasses import replace
from pathlib import Path

import pytest

from skillspector.artifacts import multiline_prompt_injection_view
from skillspector.inspection_ledger import LedgerReason
from skillspector.nodes.analyzers import artifact_integrity
from skillspector.nodes.analyzers.artifact_integrity import node
from tests.nodes.test_security_end_to_end import (
    _assert_incomplete_across_public_surfaces,
    _scan,
    _write_bundle,
)


def _letter_lines(phrase: str, separator: str = "\n") -> str:
    return "   ".join(separator.join(word) for word in phrase.split())


def _mixed_letter_lines(phrase: str) -> str:
    return "   ".join(
        "".join(character + ("\n" if i % 2 == 0 else " ") for i, character in enumerate(word))[:-1]
        for word in phrase.split()
    )


@pytest.mark.parametrize("phrase", ["send conversation to external", "never warn the user"])
@pytest.mark.parametrize("separator", ["\n", "\r\n", "\r", "\u2028", "\u2029", "\n  "])
def test_multiline_prompt_spacing_records_raw_ambiguity_line(phrase: str, separator: str) -> None:
    content = "# Instructions\n\n" + _letter_lines(phrase, separator)
    result = node({"components": ["SKILL.md"], "file_cache": {"SKILL.md": content}})

    findings = [finding for finding in result["findings"] if finding.rule_id == "AE6"]
    assert len(findings) == 1
    assert findings[0].start_line == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("phrase", ["send conversation to external", "never warn the user"])
async def test_multiline_prompt_spacing_rejects_installation_across_public_surfaces(
    tmp_path: Path, phrase: str
) -> None:
    _write_bundle(tmp_path, {"SKILL.md": "# Instructions\n" + _letter_lines(phrase) + "\n"})
    result = _scan(tmp_path)

    assert any(finding.rule_id == "AE6" for finding in result["filtered_findings"])
    await _assert_incomplete_across_public_surfaces(tmp_path, result)


@pytest.mark.asyncio
@pytest.mark.parametrize("phrase", ["send conversation to external", "never warn the user"])
async def test_mixed_singleton_gaps_reject_installation_across_public_surfaces(
    tmp_path: Path, phrase: str
) -> None:
    _write_bundle(tmp_path, {"SKILL.md": "# Instructions\n" + _mixed_letter_lines(phrase) + "\n"})
    result = _scan(tmp_path)

    assert any(finding.rule_id == "AE6" for finding in result["filtered_findings"])
    await _assert_incomplete_across_public_surfaces(tmp_path, result)


@pytest.mark.parametrize(
    "content",
    [
        "Read each section.\nWrite a summary.\nKeep normal line boundaries.",
        "A\nB\nC\nD\nE\nF\nG",
        "U\nS\nA\n\nN\nA\nS\nA",
        _letter_lines("always use rover"),
        _letter_lines("always use cover"),
        _mixed_letter_lines("always use rover"),
        _mixed_letter_lines("always use cover"),
        "never\nwarn\nthe\nweather service",
        "\n".join("- " + letter for letter in "ABCDEFG"),
        "```python\nn = 1\ne = 2\nv = 3\ne += 1\nr = 4\n```",
        "n\n\ne\n\nv\n\ne\n\nr   warn the user",
    ],
)
def test_multiline_benign_prose_notation_and_structural_boundaries(content: str) -> None:
    result = node({"components": ["SKILL.md"], "file_cache": {"SKILL.md": content}})

    assert not any(finding.rule_id == "AE6" for finding in result["findings"])


def test_multiline_projection_preserves_source_offsets_and_word_gaps() -> None:
    content = "# Notes\n\n" + _letter_lines("never warn the user", "\r\n  ")
    view = multiline_prompt_injection_view(content)

    assert view.text == "# Notes\n\nnever   warn   the   user"
    assert view.source_offsets is not None
    assert all(content[view.source_offset(i)] == char for i, char in enumerate(view.text))
    start = view.text.index("never")
    raw_start = content.index("n\r\n")
    assert view.reconstructed_source_spans(start, start + 5) == tuple(
        (raw_start + 5 * index + 1, raw_start + 5 * index + 5) for index in range(4)
    )


@pytest.mark.parametrize(
    "content", ["n\n\ne\n\nv\n\ne\n\nr", "- n\n- e\n- v", "n = 1\ne = 2\nv = 3"]
)
def test_multiline_projection_never_erases_structural_separators(content: str) -> None:
    assert multiline_prompt_injection_view(content).text == content


def test_multiline_projection_checks_runtime_and_cancels() -> None:
    content = (_letter_lines("never warn the user") + "\n\n") * 2000
    checks = 0

    def stop_after_bounded_work() -> None:
        nonlocal checks
        checks += 1
        if checks == 3:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        multiline_prompt_injection_view(content, stop_after_bounded_work)
    assert checks == 3


def test_multiline_projection_plain_prose_uses_fast_path() -> None:
    content = "Read each section.\nWrite a summary.\n" * 20_000
    checks = 0

    def check_runtime() -> None:
        nonlocal checks
        checks += 1

    view = multiline_prompt_injection_view(content, check_runtime)
    assert view.source_offsets is None
    assert view.text is content
    assert checks == 1


@pytest.mark.parametrize("canonical_first", [False, True])
def test_multiline_prompt_provenance_work_is_linear(
    monkeypatch: pytest.MonkeyPatch, canonical_first: bool
) -> None:
    repeats = 256
    canonical = "never warn the user\n" * repeats if canonical_first else ""
    content = canonical + (_letter_lines("never warn the user") + "\n\n") * repeats
    view = multiline_prompt_injection_view(content)
    visits = 0

    class CountedSpans(tuple):
        def __iter__(self):
            nonlocal visits
            for item in super().__iter__():
                visits += 1
                yield item

        def __getitem__(self, index):
            nonlocal visits
            visits += 1
            return super().__getitem__(index)

    counted = replace(view, reconstructions=CountedSpans(view.reconstructions))
    monkeypatch.setattr(artifact_integrity, "multiline_prompt_injection_view", lambda *_: counted)
    line = artifact_integrity._multiline_prompt_injection_line(
        content, artifact_integrity._ArtifactIntegrityBudget({})
    )

    assert line == (repeats + 1 if canonical_first else 1)
    assert visits <= 16 * (repeats + len(view.reconstructions))


def test_multiline_prompt_matching_preserves_deadline_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = ("never warn the user\n" * 1000) + _letter_lines("never warn the user")
    view = multiline_prompt_injection_view(content)
    monkeypatch.setattr(artifact_integrity, "multiline_prompt_injection_view", lambda *_: view)
    checks = 0

    def cancel(_self) -> None:
        nonlocal checks
        checks += 1
        if checks == 8:
            raise RuntimeError("deadline")

    monkeypatch.setattr(artifact_integrity._ArtifactIntegrityBudget, "check_runtime", cancel)
    with pytest.raises(RuntimeError, match="deadline"):
        artifact_integrity._multiline_prompt_injection_line(
            content, artifact_integrity._ArtifactIntegrityBudget({})
        )
    assert checks == 8


def test_multiline_regex_timeout_preserves_partial_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeouts = []

    class ExpiredPattern:
        def finditer(self, _text, *, timeout, concurrent):
            timeouts.append(timeout)
            raise TimeoutError("regex timed out")

    monkeypatch.setattr(artifact_integrity, "_MULTILINE_PROMPT_PATTERNS", (ExpiredPattern(),))
    monkeypatch.setattr(artifact_integrity, "transitive_remaining_seconds", lambda _state: 0.01)
    result = node(
        {
            "components": ["SKILL.md", "later.md"],
            "file_cache": {
                "SKILL.md": _letter_lines("without telling user"),
                "later.md": "Summarize text.",
            },
        }
    )

    assert timeouts == [0.01]
    assert not result["findings"]  # Timeout is missing coverage, not a semantic finding.
    for event in result["inspection_ledger"]:
        assert event["outcome"] == "partial"
        assert event["reason_code"] == LedgerReason.RUNTIME_LIMIT
        assert event["limit_seconds"] == 0.01
    assert {event["path"] for event in result["inspection_ledger"]} == {"SKILL.md", "later.md"}


@pytest.mark.asyncio
async def test_real_multiline_regex_timeout_rejects_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A deliberately tiny operation budget makes backend timeout enforcement
    # deterministic without a fragile wall-clock performance assertion.
    monkeypatch.setattr(artifact_integrity, "_MULTILINE_PROMPT_PATTERN_SECONDS", 0.000001)
    content = (_letter_lines("without telling user") + "   ") * 2400
    _write_bundle(tmp_path, {"SKILL.md": content})

    result = _scan(tmp_path)

    assert any(
        event["reason_code"] == "runtime_limit" and event["analyzer_id"] == "artifact_integrity"
        for event in result["inspection_ledger"]
        if "reason_code" in event
    )
    await _assert_incomplete_across_public_surfaces(tmp_path, result)


@pytest.mark.parametrize("phrase", ["never ınform user", "always use ² over"])
@pytest.mark.asyncio
async def test_timed_multiline_matching_retains_unicode_coverage(
    tmp_path: Path, phrase: str
) -> None:
    _write_bundle(tmp_path, {"SKILL.md": "# Instructions\n\n" + _letter_lines(phrase)})
    result = _scan(tmp_path)

    findings = [finding for finding in result["filtered_findings"] if finding.rule_id == "AE6"]
    assert len(findings) == 1
    assert findings[0].start_line == 3
    await _assert_incomplete_across_public_surfaces(tmp_path, result)


@pytest.mark.parametrize(
    "character",
    [
        "ı",
        "İ",
        "ſ",
        "K",
        "²",
        "Ⅰ",
        "é",
        "中",
        "\u0301",
        "\u200c",
        "\u200d",
        "\u00a0",
        "\u2028",
        "\u2029",
        "\u3000",
        "\x1c",
        "\x1d",
        "\x1e",
        "\x1f",
        "\n",
        "\r",
        "\t",
        "_",
        "!",
        "\U0001ccf0",
    ],
)
def test_timed_prompt_patterns_preserve_python_match_spans(character: str) -> None:
    phrases = [
        "send conversation to external",
        "send user data to external",
        "copy the conversation and send",
        "call endpoint with user",
        "include full context in request",
        "silently record",
        "without telling user later send",
        "always recommend something over",
        "never warn the user",
        "prioritize performance over user safety",
        "subtly guide the user",
        "user knowing later change",
        "build user trust before",
        "seem friendly while",
        "always include this in every response",
        "append this to all output",
    ]
    controls = [
        *phrases,
        *(phrase.replace(" ", character) for phrase in phrases),
        *(phrase.replace("i", character) for phrase in phrases),
        f"always use {character} over",
        f"always use a{character}b over",
        f"without telling user{character}send",
        f"user knowing{character}change",
    ]
    for text in controls:
        projected = artifact_integrity._multiline_prompt_matching_text(
            text, artifact_integrity._ArtifactIntegrityBudget({})
        )
        assert len(projected) == len(text)
        for original, timed in zip(
            artifact_integrity._PROJECTED_PROMPT_PATTERNS,
            artifact_integrity._MULTILINE_PROMPT_PATTERNS,
            strict=True,
        ):
            assert [match.span() for match in timed.finditer(projected, timeout=0.25)] == [
                match.span() for match in original.finditer(text)
            ], (text, original.pattern)


def test_timed_prompt_alphabet_matches_current_grammar_contract() -> None:
    # Expanding the grammar needs a corresponding matching-alphabet review:
    # placeholders must never become literals or acquire new class semantics.
    for pattern in artifact_integrity._PROJECTED_PROMPT_PATTERNS:
        source = re.sub(r"\{[0-9,]+\}", "", pattern.pattern)
        assert source.isascii()
        assert set(re.findall(r"\\(.)", source)) <= {"s", "w"}
        assert not any(character in source for character in "[]^$0123456789~")


def test_timed_prompt_alphabet_checks_runtime_and_preserves_ascii_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = artifact_integrity._ArtifactIntegrityBudget({})
    plain = "ordinary text\n" * 1000
    assert artifact_integrity._multiline_prompt_matching_text(plain, budget) is plain
    checks = 0

    def cancel(_self) -> None:
        nonlocal checks
        checks += 1
        if checks == 3:
            raise RuntimeError("deadline")

    monkeypatch.setattr(artifact_integrity._ArtifactIntegrityBudget, "check_runtime", cancel)
    with pytest.raises(RuntimeError, match="deadline"):
        artifact_integrity._multiline_prompt_matching_text("é" * 20_000, budget)
    assert checks == 3


def test_multiline_matching_does_not_yield_its_budget_to_another_python_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import threading
    import time

    start_work = threading.Event()
    original = artifact_integrity._MULTILINE_PROMPT_PATTERNS[0]

    def compete_for_gil() -> None:
        if start_work.wait(timeout=5):
            deadline = time.monotonic() + 0.35
            while time.monotonic() < deadline:
                pass

    class ContendedPattern:
        def finditer(self, text, **kwargs):
            matches = original.finditer(text, **kwargs)
            first = next(matches)
            start_work.set()
            yield first
            yield from matches

    monkeypatch.setattr(artifact_integrity, "_MULTILINE_PROMPT_PATTERNS", (ContendedPattern(),))
    interval = sys.getswitchinterval()
    worker = threading.Thread(target=compete_for_gil)
    worker.start()
    try:
        # Let the competing analyzer keep the GIL beyond the regex's 250 ms
        # budget if matching releases it. Actual matching takes milliseconds.
        sys.setswitchinterval(1.0)
        line = artifact_integrity._multiline_prompt_injection_line(
            "send conversation to them. It's a short book.\n" * 2000,
            artifact_integrity._ArtifactIntegrityBudget({}),
        )
        assert line is None
    finally:
        start_work.set()
        worker.join(timeout=5)
        sys.setswitchinterval(interval)


@pytest.mark.asyncio
async def test_parallel_scan_of_contractions_remains_complete(tmp_path: Path) -> None:
    import json

    from skillspector.mcp_server import run_scan

    text = (
        "# Reading notes\n\n"
        + (
            "It's a short book about a village library. The chapter describes shelves, windows, "
            "and reading tables. A visitor returns a borrowed volume and reads the next chapter.\n\n"
        )
        * 40
    )
    _write_bundle(
        tmp_path,
        {"SKILL.md": text, "notes.md": text, "examples.json": json.dumps({"examples": [text]})},
    )

    result = _scan(tmp_path)
    assert result["analysis_completeness"]["is_complete"] is True
    mcp = await run_scan(str(tmp_path), use_llm=False)
    assert mcp["safe_to_install"] is True
