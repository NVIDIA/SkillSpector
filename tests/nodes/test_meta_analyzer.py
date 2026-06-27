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

"""Tests for LLMMetaAnalyzer filtering and partial batch failure handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skillspector.llm_analyzer_base import Batch
from skillspector.models import Finding
from skillspector.nodes.meta_analyzer import LLMMetaAnalyzer, meta_analyzer

MOCK_PATCH_TARGET = "skillspector.llm_analyzer_base.get_chat_model"


def _mock_get_chat_model(*_args, **_kwargs):
    from unittest.mock import MagicMock

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = MagicMock()
    return mock_llm


def _analyzer() -> LLMMetaAnalyzer:
    # Skip __init__ so no LLM client / API key is needed; apply_filter is pure.
    return LLMMetaAnalyzer.__new__(LLMMetaAnalyzer)


def _finding(
    rule_id: str, start_line: int, end_line: int | None = None, severity: str = "CRITICAL"
) -> Finding:
    return Finding(
        rule_id=rule_id,
        message=f"static finding {rule_id}",
        severity=severity,
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


def test_confirmed_finding_kept_when_model_returns_end_line() -> None:
    """A finding with end_line=None matches confirmation with end_line=start_line."""
    findings = [_finding("SC4", 4), _finding("SC4", 5)]
    items = [_llm_item("SC4", 4, end_line=4), _llm_item("SC4", 5, end_line=5)]
    batch = Batch(file_path="requirements.txt", content="", findings=findings)

    kept = _analyzer().apply_filter(findings, [(batch, items)])

    assert {f.start_line for f in kept} == {4, 5}
    assert len(kept) == 2


def test_rejected_finding_still_dropped() -> None:
    """LLM-rejected MEDIUM findings are dropped (no severity floor for MEDIUM/LOW)."""
    findings = [_finding("SC4", 4, severity="MEDIUM")]
    items = [_llm_item("SC4", 4, end_line=4, is_vulnerability=False)]
    batch = Batch(file_path="requirements.txt", content="", findings=findings)

    kept = _analyzer().apply_filter(findings, [(batch, items)])

    assert kept == []


def test_low_confidence_finding_dropped() -> None:
    """MEDIUM confirmations below the confidence threshold are dropped."""
    findings = [_finding("SC4", 4, severity="MEDIUM")]
    items = [_llm_item("SC4", 4, end_line=4, confidence=0.3)]
    batch = Batch(file_path="requirements.txt", content="", findings=findings)

    kept = _analyzer().apply_filter(findings, [(batch, items)])

    assert kept == []


def test_critical_finding_kept_when_rejected_by_llm() -> None:
    """CRITICAL findings survive LLM rejection — security floor prevents false negatives."""
    findings = [_finding("SC4", 4, severity="CRITICAL")]
    items = [_llm_item("SC4", 4, end_line=4, is_vulnerability=False)]
    batch = Batch(file_path="requirements.txt", content="", findings=findings)

    kept = _analyzer().apply_filter(findings, [(batch, items)])

    assert len(kept) == 1
    assert "llm-unconfirmed" in kept[0].tags


def test_exact_end_line_match_still_works() -> None:
    """Existing behavior: matching concrete end_line keeps the finding."""
    findings = [_finding("AST1", 21, end_line=21)]
    items = [_llm_item("AST1", 21, end_line=21)]
    batch = Batch(file_path="requirements.txt", content="", findings=findings)

    kept = _analyzer().apply_filter(findings, [(batch, items)])

    assert len(kept) == 1
    assert kept[0].rule_id == "AST1"


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

        assert ("a.py", "R1") in kept
        assert ("a.py", "R2") not in kept
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


@patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
def test_meta_analyzer_batches_large_finding_sets(monkeypatch) -> None:
    """When findings > META_BATCH_SIZE, meta_analyzer splits into multiple LLM calls."""
    import importlib

    import skillspector.constants

    monkeypatch.setenv("SKILLSPECTOR_META_BATCH_SIZE", "3")
    importlib.reload(skillspector.constants)

    # 6 findings across 6 files
    findings = [
        Finding(
            rule_id=f"E{i}",
            message=f"finding {i}",
            severity="MEDIUM",
            confidence=0.8,
            file=f"file{i}.py",
            start_line=i,
        )
        for i in range(6)
    ]
    from skillspector.state import SkillspectorState

    state = SkillspectorState(
        findings=findings,
        use_llm=True,
        file_cache={f"file{i}.py": f"# file {i}" for i in range(6)},
        manifest={},
        model_config={},
    )

    call_count = {"n": 0}

    async def fake_arun_batches(self_or_batches, batches_or_nothing=None, **kwargs):
        call_count["n"] += 1
        return []  # return empty so filtered_findings is empty (fine for count test)

    with patch("skillspector.nodes.meta_analyzer.LLMMetaAnalyzer.arun_batches", fake_arun_batches):
        meta_analyzer(state)

    assert call_count["n"] >= 2, "Should split into multiple arun_batches calls when findings > batch size"


def test_split_files_into_batches_groups_files_correctly() -> None:
    """_split_files_into_batches correctly groups files within the max size."""
    from skillspector.nodes.meta_analyzer import _split_files_into_batches

    # 3 files with 2, 3, 2 findings each; max_findings=4
    findings = (
        [Finding(rule_id="R1", message="m", severity="MEDIUM", confidence=0.8, file="a.py", start_line=i) for i in range(2)]
        + [Finding(rule_id="R1", message="m", severity="MEDIUM", confidence=0.8, file="b.py", start_line=i) for i in range(3)]
        + [Finding(rule_id="R1", message="m", severity="MEDIUM", confidence=0.8, file="c.py", start_line=i) for i in range(2)]
    )
    files = ["a.py", "b.py", "c.py"]
    groups = _split_files_into_batches(files, findings, max_findings=4)
    # a.py (2) + b.py (3) = 5 > 4, so a.py alone, then b.py alone (3<=4), then c.py
    # Actually: a.py (2) fits in first group; adding b.py (3) = 5 > 4, so b.py starts group 2;
    # adding c.py (2) to group 2 = 5 > 4, so c.py starts group 3
    assert len(groups) == 3
    assert groups[0] == ["a.py"]
    assert groups[1] == ["b.py"]
    assert groups[2] == ["c.py"]


def test_split_files_into_batches_single_group_when_under_limit() -> None:
    """All files in one group when total findings <= max_findings."""
    from skillspector.nodes.meta_analyzer import _split_files_into_batches

    findings = [
        Finding(rule_id="R1", message="m", severity="MEDIUM", confidence=0.8, file="a.py", start_line=1),
        Finding(rule_id="R1", message="m", severity="MEDIUM", confidence=0.8, file="b.py", start_line=1),
    ]
    groups = _split_files_into_batches(["a.py", "b.py"], findings, max_findings=10)
    assert len(groups) == 1
    assert groups[0] == ["a.py", "b.py"]


@patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
def test_meta_analyzer_reads_batch_size_at_call_time(monkeypatch) -> None:
    """META_BATCH_SIZE is read from constants at call time, not at import time."""
    import importlib

    import skillspector.constants

    monkeypatch.setenv("SKILLSPECTOR_META_BATCH_SIZE", "1")
    importlib.reload(skillspector.constants)

    # 2 findings in 2 files; batch size=1 means each file is its own group
    findings = [
        Finding(rule_id="E1", message="m", severity="MEDIUM", confidence=0.8, file="f1.py", start_line=1),
        Finding(rule_id="E2", message="m", severity="MEDIUM", confidence=0.8, file="f2.py", start_line=1),
    ]
    from skillspector.state import SkillspectorState

    state = SkillspectorState(
        findings=findings,
        use_llm=True,
        file_cache={"f1.py": "# f1", "f2.py": "# f2"},
        manifest={},
        model_config={},
    )

    call_count = {"n": 0}

    async def fake_arun_batches_call_time(_self, _batches, **kwargs):
        call_count["n"] += 1
        return []

    with patch("skillspector.nodes.meta_analyzer.LLMMetaAnalyzer.arun_batches", fake_arun_batches_call_time):
        meta_analyzer(state)

    assert call_count["n"] == 2, "With batch size=1 and 2 files, expect 2 separate LLM calls"


def test_skip_meta_bypasses_llm_entirely() -> None:
    """skip_meta=True must return all findings without any LLM call."""
    from skillspector.state import SkillspectorState

    state = SkillspectorState(
        findings=[_finding("E1", 1), _finding("P1", 2)],
        use_llm=True,
        skip_meta=True,
        file_cache={"SKILL.md": "content"},
        manifest={},
        model_config={},
    )
    with patch("skillspector.nodes.meta_analyzer.LLMMetaAnalyzer") as mock_cls:
        result = meta_analyzer(state)
    mock_cls.assert_not_called()
    assert len(result["filtered_findings"]) == 2


@patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
def test_meta_analyzer_llm_failure_prints_stderr_hint(capsys) -> None:
    """When LLM call fails, a stderr hint about --no-llm must be printed."""
    finding = Finding(
        rule_id="E1",
        message="E1 test finding",
        severity="HIGH",
        confidence=0.8,
        file="SKILL.md",
        start_line=1,
    )
    state: dict[str, object] = {
        "findings": [finding],
        "use_llm": True,
        "file_cache": {"SKILL.md": "# test\nsome content"},
        "manifest": {"name": "test"},
        "model_config": {},
    }
    batch = Batch(file_path="SKILL.md", content="# test\nsome content", findings=[finding])
    with (
        patch.object(LLMMetaAnalyzer, "get_batches", return_value=[batch]),
        patch.object(
            LLMMetaAnalyzer,
            "arun_batches",
            new_callable=AsyncMock,
            side_effect=Exception("provider not available"),
        ),
    ):
        result = meta_analyzer(state)

    captured = capsys.readouterr()
    assert "--no-llm" in captured.err, "stderr must mention --no-llm when LLM fails"
    assert result["filtered_findings"], "fail-closed: findings still returned"
