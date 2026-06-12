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

"""Tests for the meta_analyzer node: batch-failure resilience of the filter.

Covers the keep/drop semantics when Stage 2 batches fail (issues #9/#11):
a finding the LLM rejected is dropped, a finding the LLM never saw is kept
via the fallback path, and one failed batch must not disable filtering for
the batches that succeeded.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from skillspector.llm_analyzer_base import Batch
from skillspector.models import Finding
from skillspector.nodes.meta_analyzer import LLMMetaAnalyzer, meta_analyzer

MOCK_PATCH_TARGET = "skillspector.llm_analyzer_base.get_chat_model"


def _mock_get_chat_model(*_args, **_kwargs):
    from unittest.mock import MagicMock

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = MagicMock()
    return mock_llm


def _confirm(pattern_id: str, file: str, start_line: int) -> dict[str, object]:
    """LLM item confirming a finding, as parse_response would emit it."""
    return {
        "pattern_id": pattern_id,
        "is_vulnerability": True,
        "confidence": 0.9,
        "explanation": "confirmed by llm",
        "remediation": "fix it",
        "_file": file,
        "start_line": start_line,
        "end_line": None,
    }


@patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
class TestMetaAnalyzerPartialBatchFailure:
    def _state(self, findings: list[Finding]) -> dict[str, object]:
        return {
            "findings": findings,
            "use_llm": True,
            "file_cache": {"a.py": "code a", "b.py": "code b"},
            "manifest": {},
            "model_config": {},
        }

    def test_unanalysed_findings_survive_a_failed_batch(self) -> None:
        """Findings whose batch failed are kept (no verdict != rejection)."""
        f_confirmed = Finding(rule_id="R1", message="m", file="a.py", start_line=1)
        f_rejected = Finding(rule_id="R2", message="m", file="a.py", start_line=5)
        f_unseen = Finding(rule_id="R1", message="m", file="b.py", start_line=3)

        batch_a = Batch(file_path="a.py", content="code a", findings=[f_confirmed, f_rejected])
        batch_b = Batch(file_path="b.py", content="code b", findings=[f_unseen])

        # batch_b never returned (timeout/429): only batch_a's verdicts exist,
        # and the LLM confirmed R1 but stayed silent on R2 (= rejection).
        partial_results = [(batch_a, [_confirm("R1", "a.py", 1)])]

        with (
            patch.object(LLMMetaAnalyzer, "get_batches", return_value=[batch_a, batch_b]),
            patch.object(
                LLMMetaAnalyzer,
                "arun_batches",
                new_callable=AsyncMock,
                return_value=partial_results,
            ),
        ):
            result = meta_analyzer(self._state([f_confirmed, f_rejected, f_unseen]))

        filtered = result["filtered_findings"]
        kept = {(f.file, f.rule_id) for f in filtered}

        # the real filter still applies to the batch that came back
        assert ("a.py", "R1") in kept
        assert ("a.py", "R2") not in kept
        # the finding the LLM never saw must NOT be silently dropped
        assert ("b.py", "R1") in kept

        confirmed = next(f for f in filtered if f.file == "a.py")
        assert confirmed.explanation == "confirmed by llm"

    def test_all_batches_failed_keeps_everything_via_fallback(self) -> None:
        f1 = Finding(rule_id="R1", message="m", file="a.py", start_line=1)
        f2 = Finding(rule_id="R2", message="m", file="b.py", start_line=2)
        batch_a = Batch(file_path="a.py", content="code a", findings=[f1])
        batch_b = Batch(file_path="b.py", content="code b", findings=[f2])

        with (
            patch.object(LLMMetaAnalyzer, "get_batches", return_value=[batch_a, batch_b]),
            patch.object(
                LLMMetaAnalyzer,
                "arun_batches",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = meta_analyzer(self._state([f1, f2]))

        kept = {(f.file, f.rule_id) for f in result["filtered_findings"]}
        assert kept == {("a.py", "R1"), ("b.py", "R2")}

    def test_no_failures_keeps_strict_confirm_or_drop(self) -> None:
        """When every batch returns, unconfirmed findings are dropped as before."""
        f_confirmed = Finding(rule_id="R1", message="m", file="a.py", start_line=1)
        f_rejected = Finding(rule_id="R2", message="m", file="b.py", start_line=2)
        batch_a = Batch(file_path="a.py", content="code a", findings=[f_confirmed])
        batch_b = Batch(file_path="b.py", content="code b", findings=[f_rejected])

        full_results = [
            (batch_a, [_confirm("R1", "a.py", 1)]),
            (batch_b, []),
        ]

        with (
            patch.object(LLMMetaAnalyzer, "get_batches", return_value=[batch_a, batch_b]),
            patch.object(
                LLMMetaAnalyzer,
                "arun_batches",
                new_callable=AsyncMock,
                return_value=full_results,
            ),
        ):
            result = meta_analyzer(self._state([f_confirmed, f_rejected]))

        kept = {(f.file, f.rule_id) for f in result["filtered_findings"]}
        assert kept == {("a.py", "R1")}
