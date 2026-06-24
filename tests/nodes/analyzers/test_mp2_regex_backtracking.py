# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for MP2 regex pattern — catastrophic backtracking prevention."""

from __future__ import annotations

import time

import pytest

from skillspector.nodes.analyzers import static_patterns_memory_poisoning as mp_module


class TestMP2DetectsStuffing:
    """MP2 correctly detects repeated content (context window stuffing)."""

    def test_repeated_phrase_detected(self) -> None:
        """A multi-char phrase repeated 25+ times triggers MP2."""
        content = "ABCD" * 30
        findings = mp_module.analyze(content, "test.md", "markdown")
        mp2 = [f for f in findings if f.rule_id == "MP2"]
        assert len(mp2) >= 1

    def test_repeated_short_phrase_detected(self) -> None:
        """A short multi-char phrase repeated 25+ times triggers MP2."""
        content = "DEADBEEF_PAYLOAD" * 25
        findings = mp_module.analyze(content, "payload.md", "markdown")
        mp2 = [f for f in findings if f.rule_id == "MP2"]
        assert len(mp2) >= 1

    def test_short_repetition_not_detected(self) -> None:
        """Under 20 repetitions should not trigger the repetition pattern."""
        content = "hello world. " * 5
        findings = mp_module.analyze(content, "normal.md", "markdown")
        mp2_repetition = [
            f for f in findings if f.rule_id == "MP2" and "Context Window Stuffing" in f.message
        ]
        assert len(mp2_repetition) == 0

    def test_separator_line_not_detected(self) -> None:
        """Single-char separators like '=' * 80 should be suppressed."""
        content = "=" * 80
        findings = mp_module.analyze(content, "readme.md", "markdown")
        mp2 = [f for f in findings if f.rule_id == "MP2"]
        assert len(mp2) == 0

    def test_whitespace_bearing_stuffing_detected(self) -> None:
        """Repeated tokens containing whitespace (e.g. 'x ' * 30) must not be suppressed."""
        content = "x " * 30
        findings = mp_module.analyze(content, "payload.md", "markdown")
        mp2 = [f for f in findings if f.rule_id == "MP2"]
        assert len(mp2) >= 1, "Whitespace-bearing stuffing should be detected, not suppressed"


class TestMP2NoBacktracking:
    """MP2 regex completes in bounded time on adversarial inputs."""

    @pytest.mark.timeout(5)
    def test_non_matching_random_input_completes_fast(self) -> None:
        """Non-repeating input of moderate size should complete within 5 seconds.

        The old regex with nested lazy quantifier and backreference would hang
        on non-matching inputs due to catastrophic backtracking.
        """
        content = "".join(chr(65 + (i % 26)) for i in range(2000))
        start = time.monotonic()
        mp_module.analyze(content, "adversarial.txt", "text")
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"MP2 regex took {elapsed:.1f}s — possible backtracking"

    @pytest.mark.timeout(5)
    def test_near_miss_pattern_completes_fast(self) -> None:
        """Input with almost-repeating but not-quite structure completes quickly.

        This is the classic ReDoS vector: content that almost matches but
        requires the regex engine to explore many backtracking paths.
        """
        content = ("abcdefghij" * 19) + "abcdefghiX" + ("abcdefghij" * 5)
        start = time.monotonic()
        mp_module.analyze(content, "nearmiss.txt", "text")
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"MP2 regex took {elapsed:.1f}s — possible backtracking"

    @pytest.mark.timeout(5)
    def test_large_non_repeating_content(self) -> None:
        """5KB of non-repeating text should not cause regex to hang."""
        lines = [f"Line {i}: This is unique content number {i * 7 + 3}." for i in range(100)]
        content = "\n".join(lines)
        start = time.monotonic()
        mp_module.analyze(content, "large.md", "markdown")
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"MP2 regex took {elapsed:.1f}s on 5KB input"
