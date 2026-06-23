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

"""Unit tests for LLMMetaAnalyzer.apply_filter (no LLM / no network)."""

from skillspector.llm_analyzer_base import Batch
from skillspector.models import Finding
from skillspector.nodes.meta_analyzer import LLMMetaAnalyzer


def _analyzer() -> LLMMetaAnalyzer:
    # Skip __init__ so no LLM client / API key is needed; apply_filter is pure.
    return LLMMetaAnalyzer.__new__(LLMMetaAnalyzer)


def _finding(rule_id: str, start_line: int, end_line: int | None = None) -> Finding:
    return Finding(
        rule_id=rule_id,
        message=f"static finding {rule_id}",
        severity="CRITICAL",
        confidence=0.9,
        file="requirements.txt",
        start_line=start_line,
        end_line=end_line,
    )


def _llm_item(rule_id: str, start_line: int, **kw: object) -> dict[str, object]:
    item: dict[str, object] = {
        "pattern_id": rule_id,
        "is_vulnerability": True,
        "confidence": 1.0,
        "start_line": start_line,
        "_file": "requirements.txt",
    }
    item.update(kw)
    return item


def test_confirmed_finding_kept_when_model_returns_end_line() -> None:
    """Regression: a static finding with end_line=None must still match a
    confirmation whose end_line is populated (e.g. end_line == start_line, as
    some models return). Previously these confirmed findings were silently
    dropped. See issue #67."""
    findings = [_finding("SC4", 4), _finding("SC4", 5)]
    items = [_llm_item("SC4", 4, end_line=4), _llm_item("SC4", 5, end_line=5)]
    batch = Batch(file_path="requirements.txt", content="", findings=findings)

    kept = _analyzer().apply_filter(findings, [(batch, items)])

    assert {f.start_line for f in kept} == {4, 5}
    assert len(kept) == 2


def test_rejected_finding_still_dropped() -> None:
    """The end_line-agnostic fallback must not resurrect findings the LLM
    rejected (is_vulnerability=False)."""
    findings = [_finding("SC4", 4)]
    items = [_llm_item("SC4", 4, end_line=4, is_vulnerability=False)]
    batch = Batch(file_path="requirements.txt", content="", findings=findings)

    kept = _analyzer().apply_filter(findings, [(batch, items)])

    assert kept == []


def test_low_confidence_finding_dropped() -> None:
    """Confirmations below the confidence threshold are not kept."""
    findings = [_finding("SC4", 4)]
    items = [_llm_item("SC4", 4, end_line=4, confidence=0.3)]
    batch = Batch(file_path="requirements.txt", content="", findings=findings)

    kept = _analyzer().apply_filter(findings, [(batch, items)])

    assert kept == []


def test_exact_end_line_match_still_works() -> None:
    """Existing behaviour: when both sides carry the same concrete end_line,
    the finding is kept (no regression from the new fallback)."""
    findings = [_finding("AST1", 21, end_line=21)]
    items = [_llm_item("AST1", 21, end_line=21)]
    batch = Batch(file_path="requirements.txt", content="", findings=findings)

    kept = _analyzer().apply_filter(findings, [(batch, items)])

    assert len(kept) == 1
    assert kept[0].rule_id == "AST1"
