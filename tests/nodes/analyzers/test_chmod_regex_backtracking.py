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

"""Tests for the numeric chmod patterns — catastrophic backtracking prevention.

The option group that lets `chmod -R 4755 dir` reach the mode was written as
`(?:--?[\\w=-]*[ \\t]+)*`.  A token starting with `--` can then be matched two
ways, and when no mode follows the engine tries every combination: on
`chmod ` + `-- ` * 24 the match took seconds and doubled with each extra token.
The per-artifact runtime budget is only checked after an analyzer returns, so it
cannot interrupt a running `re.finditer`.  The group now takes a single leading
dash (`[\\w=-]*` already absorbs a long option's second one), which is linear.

Scope note: a long run of octal digits that never reaches a reportable mode
(`"chmod 0" * 8000`) is still quadratic in the number of `chmod` occurrences.
That shape is pre-existing and unaltered by this change -- it is faster on this
branch than on main -- so it is left alone here rather than folded into a fix
for the option group.
"""

from __future__ import annotations

import time

import pytest

from skillspector.nodes.analyzers import static_patterns_privilege_escalation as pe_module
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm_module

_ANALYZERS = (pe_module, tm_module)


class TestChmodOptionGroupDoesNotBacktrack:
    """A run of option-looking tokens with no mode must stay cheap."""

    @pytest.mark.timeout(10)
    @pytest.mark.parametrize("tokens", [24, 40, 120])
    def test_unmatched_option_tokens_complete_fast(self, tokens: int) -> None:
        """`chmod -- -- -- ... x` must not blow up combinatorially."""
        content = "chmod " + "-- " * tokens + "x"
        start = time.monotonic()
        for module in _ANALYZERS:
            module.analyze(content, "adversarial.sh", "shell")
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"chmod option group took {elapsed:.1f}s on {tokens} tokens"

    @pytest.mark.timeout(10)
    def test_very_long_option_run_is_linear(self) -> None:
        """A 60k-character option run still completes well inside the budget."""
        content = "chmod " + "-- " * 30000 + "x"
        start = time.monotonic()
        for module in _ANALYZERS:
            module.analyze(content, "adversarial.sh", "shell")
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"chmod option group took {elapsed:.1f}s on a 60k-char line"

    @pytest.mark.timeout(10)
    def test_long_option_prefixes_do_not_backtrack(self) -> None:
        """Tokens with no separating space are the other ReDoS shape."""
        content = "chmod " + "--" * 400 + " x"
        start = time.monotonic()
        for module in _ANALYZERS:
            module.analyze(content, "adversarial.sh", "shell")
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"chmod option group took {elapsed:.1f}s on long prefixes"


class TestChmodOptionGroupStillMatchesLongOptions:
    """The single-dash group must keep reaching modes behind long options."""

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param("chmod --recursive 4755 dir", id="long_option_setuid"),
            pytest.param("chmod --reference=other 6755 dir", id="long_option_with_equals"),
            pytest.param("chmod -R 4755 dir", id="short_option_setuid"),
        ],
    )
    def test_mode_behind_an_option_is_still_reported(self, source: str) -> None:
        """Dropping the second dash from the group must not drop coverage."""
        findings = pe_module.analyze(source, "setup.sh", "shell")
        assert any(finding.rule_id == "PE2" for finding in findings), findings

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param("chmod --recursive 777 dir", id="long_option_world_writable"),
            pytest.param("chmod -R 777 dir", id="short_option_world_writable"),
        ],
    )
    def test_world_writable_mode_behind_an_option_is_still_tm1(self, source: str) -> None:
        """The same holds for the TM1 numeric pattern."""
        findings = tm_module.analyze(source, "setup.sh", "shell")
        assert any(finding.rule_id == "TM1" for finding in findings), findings
