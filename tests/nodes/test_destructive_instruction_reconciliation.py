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

"""Destructive-operation evidence reaches existing LLM reconciliation and survives it."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from skillspector.llm_analyzer_base import Batch
from skillspector.models import Finding
from skillspector.nodes.meta_analyzer import (
    PER_FILE_ANALYSIS_PROMPT,
    LLMMetaAnalyzer,
    _format_findings_for_prompt,
    meta_analyzer,
)
from skillspector.state import SkillspectorState


def _finding(context: str = "instruction") -> Finding:
    return Finding(
        rule_id="TM1",
        message="Destructive folder operation",
        severity="HIGH",
        confidence=0.9,
        file="SKILL.md",
        start_line=7,
        end_line=8,
        start_column=3,
        end_column=42,
        context="Delete all files in the customer backups folder.",
        matched_text="Delete all files in the customer backups folder",
        tags=["destructive-operation"] + (["contextual-triage"] if context == "warning" else []),
        evidence={
            "destructive_operation": {
                "action": "delete",
                "target": "the customer backups folder",
                "form": "natural-language",
                "scope": "broad",
                "context": context,
            },
        },
        match_fingerprint="sha256:destructive-operation",
        occurrences=[{"file": "SKILL.md", "start_line": 7, "end_line": 8}],
    )


def _analyzer() -> LLMMetaAnalyzer:
    analyzer = LLMMetaAnalyzer.__new__(LLMMetaAnalyzer)
    analyzer.base_prompt = PER_FILE_ANALYSIS_PROMPT
    return analyzer


def _assert_preserved(original: Finding, returned: Finding) -> None:
    assert returned.finding_id == original.finding_id
    assert returned.severity == original.severity
    assert returned.confidence >= original.confidence
    assert returned.evidence == original.evidence
    assert returned.match_fingerprint == original.match_fingerprint
    assert returned.occurrences == original.occurrences
    assert returned.file == original.file
    assert (returned.start_line, returned.end_line) == (7, 8)
    assert (returned.start_column, returned.end_column) == (3, 42)
    assert set(original.tags) <= set(returned.tags)


@pytest.mark.parametrize("form", ["shell", "natural-language"])
@pytest.mark.parametrize("context", ["instruction", "warning"])
def test_prompt_supplies_destructive_action_target_and_context(form: str, context: str) -> None:
    finding = _finding(context)
    operation = finding.evidence["destructive_operation"]
    assert isinstance(operation, dict)
    operation["form"] = form
    batch = Batch(file_path=finding.file, content=finding.context or "", findings=[finding])

    prompt = _analyzer().build_prompt(batch)

    assert json.dumps(operation) in prompt
    assert "Location: SKILL.md:7–8" in prompt
    assert "evaluate deletion of any folder or its contents" in prompt
    assert "distinguish active\ninstructions from warnings" in prompt
    assert "Deletion alone does not\nestablish malicious intent" in prompt


def test_prompt_evidence_is_allowlisted_redacted_and_bounded() -> None:
    finding = _finding()
    finding.evidence = {
        "not_for_provider": "outer-secret",
        "destructive_operation": {
            "action": "x" * 1000,
            "target": "https://person:password@example.com/data?token=secret" + "x" * 1000,
            "form": "shell",
            "scope": "specified",
            "context": "instruction",
            "arbitrary_payload": "inner-secret",
        },
    }

    formatted = _format_findings_for_prompt([finding])
    evidence = json.loads(formatted.split("Destructive operation evidence (untrusted): ", 1)[1])

    assert set(evidence) == {"action", "target", "form", "scope", "context"}
    assert all(len(value) <= 512 for value in evidence.values())
    assert "outer-secret" not in formatted
    assert "inner-secret" not in formatted
    assert "password" not in formatted
    assert "token=secret" not in formatted
    assert "https://***@example.com/data?token=%2A%2A%2A" == evidence["target"]


@pytest.mark.parametrize("evidence", [None, "invalid", {"action": ["delete"]}, {}])
def test_prompt_ignores_invalid_operation_evidence(evidence: object) -> None:
    finding = _finding()
    finding.evidence["destructive_operation"] = evidence

    assert "Destructive operation evidence" not in _format_findings_for_prompt([finding])


@pytest.mark.parametrize("confirmed", [True, False])
@pytest.mark.parametrize("context", ["instruction", "warning"])
def test_llm_verdict_preserves_static_destructive_evidence(confirmed: bool, context: str) -> None:
    original = _finding(context)
    batch = Batch(file_path=original.file, content=original.context or "", findings=[original])
    verdict = {
        "pattern_id": "TM1",
        "start_line": original.start_line,
        "end_line": original.end_line,
        "is_vulnerability": confirmed,
        "confidence": 0.7,
        "explanation": "Operation evaluated in context",
        "remediation": "Restrict the operation to the intended cleanup target",
    }

    [returned] = _analyzer().apply_filter([original], [(batch, [verdict])])

    _assert_preserved(original, returned)
    assert ("llm-unconfirmed" in returned.tags) is not confirmed
    if confirmed:
        assert returned.explanation == verdict["explanation"]


def test_no_llm_preserves_static_destructive_evidence() -> None:
    original = _finding()
    state: SkillspectorState = {"findings": [original], "use_llm": False}

    result = meta_analyzer(state)

    [returned] = result["findings"]
    _assert_preserved(original, returned)
    assert "llm-unconfirmed" not in returned.tags
    assert result["effective_finding_ids"] == [original.finding_id]


@pytest.mark.parametrize(
    ("error", "response_received"),
    [
        (TimeoutError("provider timeout"), False),
        (RuntimeError("provider unavailable"), False),
        (ValueError("malformed structured response"), True),
    ],
)
def test_llm_failure_preserves_static_destructive_evidence(
    error: Exception, response_received: bool
) -> None:
    original = _finding()
    batch = Batch(file_path=original.file, content=original.context or "", findings=[original])
    state: SkillspectorState = {
        "findings": [original],
        "use_llm": True,
        "llm_file_cache": {original.file: batch.content},
    }
    with patch("skillspector.nodes.meta_analyzer.LLMMetaAnalyzer") as mock_cls:
        mock_cls.return_value.get_batches.return_value = [batch]
        mock_cls.return_value.arun_batches = AsyncMock(side_effect=error)
        mock_cls.return_value.response_received = response_received
        mock_cls.return_value.inference_usage = []

        result = meta_analyzer(state)

    [returned] = result["findings"]
    _assert_preserved(original, returned)
    assert "llm-unconfirmed" not in returned.tags
    assert result["effective_finding_ids"] == [original.finding_id]
