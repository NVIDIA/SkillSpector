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

"""Tests for the semantic_security_discovery analyzer node (B.4.1)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from skillspector.llm_analyzer_base import LLMAnalysisResult, LLMFinding
from skillspector.models import Finding
from skillspector.nodes.analyzers.semantic_security_discovery import (
    ANALYZER_ID,
    ANALYZER_PROMPT,
    node,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

MOCK_PATCH_TARGET = "skillspector.llm_analyzer_base.get_chat_model"


def _mock_get_chat_model(*_args, **_kwargs):
    """Return a mock chat model that supports with_structured_output."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = MagicMock()
    return mock_llm


def _make_finding(rule_id: str, file: str = "SKILL.md") -> Finding:
    return Finding(
        rule_id=rule_id,
        message=f"Test finding for {rule_id}",
        severity="HIGH",
        confidence=0.85,
        file=file,
        start_line=1,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_state():
    """Minimal valid SkillspectorState for semantic_security_discovery."""
    return {
        "use_llm": True,
        "model_config": {"semantic_security_discovery": "test-model"},
        "components": ["SKILL.md"],
        "file_cache": {"SKILL.md": "# My Skill\n\nThis skill helps users.\n"},
    }


# ---------------------------------------------------------------------------
# TestSemanticSecurityDiscoveryNode
# ---------------------------------------------------------------------------


class TestSemanticSecurityDiscoveryNode:
    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_skipped_when_use_llm_false(self, base_state) -> None:
        base_state["use_llm"] = False
        with patch(MOCK_PATCH_TARGET) as mock_llm:
            result = node(base_state)
        assert result == {"findings": []}
        mock_llm.assert_not_called()

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_returns_findings_from_llm(self, base_state) -> None:
        expected_finding = _make_finding("SSD-1")
        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        with patch.object(
            LLMAnalyzerBase,
            "run_batches",
            return_value=[(MagicMock(), [expected_finding])],
        ):
            result = node(base_state)

        assert len(result["findings"]) == 1
        f = result["findings"][0]
        assert isinstance(f, Finding)
        assert f.rule_id == "SSD-1"
        assert f.file == "SKILL.md"
        assert f.severity == "HIGH"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_empty_components_returns_no_findings(self, base_state) -> None:
        base_state["components"] = []
        base_state["file_cache"] = {}
        with patch(MOCK_PATCH_TARGET) as mock_llm:
            result = node(base_state)
        assert result == {"findings": []}
        mock_llm.assert_not_called()

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_all_ssd_rule_ids_pass_through(self, base_state) -> None:
        findings = [_make_finding(rid) for rid in ("SSD-1", "SSD-2", "SSD-3", "SSD-4")]
        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        with patch.object(
            LLMAnalyzerBase,
            "run_batches",
            return_value=[(MagicMock(), findings)],
        ):
            result = node(base_state)

        rule_ids = {f.rule_id for f in result["findings"]}
        assert rule_ids == {"SSD-1", "SSD-2", "SSD-3", "SSD-4"}


# ---------------------------------------------------------------------------
# TestSemanticSecurityDiscoveryPrompt
# ---------------------------------------------------------------------------


class TestSemanticSecurityDiscoveryPrompt:
    def test_prompt_contains_all_rule_ids(self) -> None:
        for rule_id in ("SSD-1", "SSD-2", "SSD-3", "SSD-4"):
            assert rule_id in ANALYZER_PROMPT, f"{rule_id} missing from ANALYZER_PROMPT"

    def test_prompt_instructs_residual_gap(self) -> None:
        # The dedup instruction must mention intent/meaning/semantic context
        lower = ANALYZER_PROMPT.lower()
        assert any(term in lower for term in ("intent", "meaning", "semantic"))
        assert "residual gap" in lower

    def test_analyzer_id_is_correct(self) -> None:
        assert ANALYZER_ID == "semantic_security_discovery"


# ---------------------------------------------------------------------------
# TestSemanticSecurityDiscoveryBatching
# ---------------------------------------------------------------------------


class TestSemanticSecurityDiscoveryBatching:
    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_single_file_one_batch(self, base_state) -> None:
        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        with patch.object(LLMAnalyzerBase, "run_batches", return_value=[]) as mock_run:
            node(base_state)

        mock_run.assert_called_once()
        batches_arg = mock_run.call_args[0][0]
        assert len(batches_arg) == 1
        assert batches_arg[0].file_path == "SKILL.md"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_oversized_file_multiple_batches(self, base_state) -> None:
        # A file large enough to exceed the reduced token budget
        long_content = "\n".join(f"Line {i:04d}: " + "x" * 30 for i in range(300))
        base_state["file_cache"] = {"SKILL.md": long_content}
        base_state["components"] = ["SKILL.md"]

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        with patch(
            "skillspector.llm_analyzer_base.get_max_input_tokens",
            return_value=50,
        ):
            with patch.object(LLMAnalyzerBase, "run_batches", return_value=[]) as mock_run:
                node(base_state)

        batches_arg = mock_run.call_args[0][0]
        assert len(batches_arg) > 1


# ---------------------------------------------------------------------------
# TestUseLlmGuard (edge cases)
# ---------------------------------------------------------------------------


class TestUseLlmGuard:
    def test_use_llm_missing_from_state_proceeds(self) -> None:
        """Missing use_llm key should default to enabled (not skip)."""
        state = {
            "model_config": {"semantic_security_discovery": "test-model"},
            "components": ["SKILL.md"],
            "file_cache": {"SKILL.md": "# Skill"},
        }
        with patch(MOCK_PATCH_TARGET, _mock_get_chat_model):
            from skillspector.llm_analyzer_base import LLMAnalyzerBase

            with patch.object(LLMAnalyzerBase, "run_batches", return_value=[]):
                result = node(state)
        assert result == {"findings": []}


# ---------------------------------------------------------------------------
# TestModelResolution
# ---------------------------------------------------------------------------


class TestModelResolution:
    @patch(MOCK_PATCH_TARGET)
    def test_uses_analyzer_specific_model(self, mock_get_model: MagicMock) -> None:
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock()
        mock_llm.with_structured_output.return_value.invoke = MagicMock(
            return_value=LLMAnalysisResult(findings=[])
        )
        mock_get_model.return_value = mock_llm

        state = {
            "file_cache": {"SKILL.md": "# Skill"},
            "model_config": {
                "semantic_security_discovery": "custom/model-a",
                "default": "custom/model-b",
            },
        }
        node(state)
        mock_get_model.assert_called_once()
        assert mock_get_model.call_args.kwargs.get("model") == "custom/model-a"

    @patch(MOCK_PATCH_TARGET)
    def test_falls_back_to_default_model(self, mock_get_model: MagicMock) -> None:
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock()
        mock_llm.with_structured_output.return_value.invoke = MagicMock(
            return_value=LLMAnalysisResult(findings=[])
        )
        mock_get_model.return_value = mock_llm

        state = {
            "file_cache": {"SKILL.md": "# Skill"},
            "model_config": {"default": "custom/model-b"},
        }
        node(state)
        assert mock_get_model.call_args.kwargs.get("model") == "custom/model-b"

    @patch(MOCK_PATCH_TARGET)
    def test_falls_back_to_constant_default(self, mock_get_model: MagicMock) -> None:
        from skillspector.constants import _SKILLSPECTOR_DEFAULT_MODEL

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock()
        mock_llm.with_structured_output.return_value.invoke = MagicMock(
            return_value=LLMAnalysisResult(findings=[])
        )
        mock_get_model.return_value = mock_llm

        state = {"file_cache": {"SKILL.md": "# Skill"}, "model_config": {}}
        node(state)
        assert mock_get_model.call_args.kwargs.get("model") == _SKILLSPECTOR_DEFAULT_MODEL


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @patch(MOCK_PATCH_TARGET)
    def test_value_error_propagates(self, mock_get_model: MagicMock) -> None:
        mock_get_model.side_effect = ValueError("No LLM API key configured.")
        state = {"file_cache": {"SKILL.md": "# Skill"}}
        with pytest.raises(ValueError, match="API key"):
            node(state)

    @patch(MOCK_PATCH_TARGET)
    def test_generic_exception_returns_empty(self, mock_get_model: MagicMock) -> None:
        mock_get_model.side_effect = RuntimeError("LLM service unavailable")
        state = {"file_cache": {"SKILL.md": "# Skill"}}
        result = node(state)
        assert result == {"findings": []}

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_validation_error_returns_empty(self) -> None:
        """Malformed LLM response (ValidationError) must not crash the graph."""
        # Build a real ValidationError by feeding bad data to the schema
        try:
            LLMAnalysisResult.model_validate({"findings": "not-an-array"})
        except ValidationError as exc:
            validation_err = exc
        else:
            pytest.fail("Expected ValidationError from bad data")

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        with patch.object(LLMAnalyzerBase, "run_batches", side_effect=validation_err):
            result = node({"file_cache": {"SKILL.md": "# Skill"}})
        assert result == {"findings": []}


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_SSD_FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures" / "ssd"


def _build_file_cache(skill_dir: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    for item in sorted(skill_dir.rglob("*")):
        if not item.is_file():
            continue
        rel = item.relative_to(skill_dir).as_posix()  # forward slashes on every OS
        try:
            cache[rel] = item.read_text(encoding="utf-8", errors="replace")
        except OSError:
            cache[rel] = ""
    return cache


def _make_file_aware_run_batches(responses: dict[str, LLMAnalysisResult]):
    """Return a mock run_batches that dispatches based on file_path in each batch."""
    from skillspector.llm_analyzer_base import Batch

    def _run_batches(self_inner, batches: list[Batch], **_kwargs):
        results = []
        for batch in batches:
            response = responses.get(batch.file_path, LLMAnalysisResult(findings=[]))
            findings = [f.to_finding(batch.file_path) for f in response.findings]
            results.append((batch, findings))
        return results

    return _run_batches


# ---------------------------------------------------------------------------
# TestFixtureConftest — conftest safe/malicious_skill_dir fixtures
# ---------------------------------------------------------------------------


_CONFTEST_MALICIOUS_RESPONSES: dict[str, LLMAnalysisResult] = {
    "SKILL.md": LLMAnalysisResult(
        findings=[
            LLMFinding(
                rule_id="SSD-4",
                message="Instructions establish trust then request harmful action",
                severity="HIGH",
                start_line=11,
                confidence=0.88,
                explanation="Step 2 references a harmful action framed as a normal workflow step.",
                remediation="Remove the harmful instruction from the workflow.",
            ),
        ]
    ),
    "scripts/helper.py": LLMAnalysisResult(
        findings=[
            LLMFinding(
                rule_id="SSD-3",
                message="Plain-language instructions to collect and transmit sensitive env vars",
                severity="CRITICAL",
                start_line=5,
                confidence=0.92,
                explanation="Code iterates environment variables with secret-like names and posts them externally.",
                remediation="Remove credential harvesting or add explicit user disclosure.",
            ),
        ]
    ),
}


class TestFixtureConftestMalicious:
    """Tests using the malicious_skill_dir conftest fixture."""

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_malicious_skill_produces_findings(self, malicious_skill_dir: Path) -> None:
        file_cache = _build_file_cache(malicious_skill_dir)
        state = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        with patch.object(
            LLMAnalyzerBase,
            "run_batches",
            _make_file_aware_run_batches(_CONFTEST_MALICIOUS_RESPONSES),
        ):
            result = node(state)

        assert len(result["findings"]) == 2
        files = {f.file for f in result["findings"]}
        assert "SKILL.md" in files
        assert "scripts/helper.py" in files

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_malicious_skill_rule_ids(self, malicious_skill_dir: Path) -> None:
        file_cache = _build_file_cache(malicious_skill_dir)
        state = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        with patch.object(
            LLMAnalyzerBase,
            "run_batches",
            _make_file_aware_run_batches(_CONFTEST_MALICIOUS_RESPONSES),
        ):
            result = node(state)

        rule_ids = {f.rule_id for f in result["findings"]}
        assert "SSD-3" in rule_ids
        assert "SSD-4" in rule_ids

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_malicious_skill_findings_preserve_metadata(self, malicious_skill_dir: Path) -> None:
        file_cache = _build_file_cache(malicious_skill_dir)
        state = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        with patch.object(
            LLMAnalyzerBase,
            "run_batches",
            _make_file_aware_run_batches(_CONFTEST_MALICIOUS_RESPONSES),
        ):
            result = node(state)

        code_finding = next(f for f in result["findings"] if f.file == "scripts/helper.py")
        assert isinstance(code_finding, Finding)
        assert code_finding.severity == "CRITICAL"
        assert code_finding.confidence == 0.92
        assert code_finding.start_line == 5


class TestFixtureConftestSafe:
    """Tests using the safe_skill_dir conftest fixture."""

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_safe_skill_produces_no_findings(self, safe_skill_dir: Path) -> None:
        file_cache = _build_file_cache(safe_skill_dir)
        state = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        with patch.object(
            LLMAnalyzerBase,
            "run_batches",
            return_value=[],
        ):
            result = node(state)

        assert result == {"findings": []}


# ---------------------------------------------------------------------------
# TestFixtureSsd1SemanticInjection
# ---------------------------------------------------------------------------

_SSD1_RESPONSES: dict[str, LLMAnalysisResult] = {
    "SKILL.md": LLMAnalysisResult(
        findings=[
            LLMFinding(
                rule_id="SSD-1",
                message="Instructions frame elevated permissions as user-agreed terms",
                severity="HIGH",
                start_line=14,
                confidence=0.87,
                explanation=(
                    "The skill instructs the assistant to treat the user as having 'elevated "
                    "access permissions' and to 'set aside its usual constraints', which are "
                    "benign-sounding phrases that semantically redirect model behavior."
                ),
                remediation="Remove instructions that attempt to override model safety guidelines.",
            ),
        ]
    ),
}


class TestFixtureSsd1SemanticInjection:
    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_semantic_injection_flagged(self) -> None:
        skill_dir = _SSD_FIXTURES / "ssd1_semantic_injection"
        if not skill_dir.is_dir():
            pytest.skip("ssd1_semantic_injection fixture not present")

        file_cache = _build_file_cache(skill_dir)
        state: dict = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        with patch.object(
            LLMAnalyzerBase,
            "run_batches",
            _make_file_aware_run_batches(_SSD1_RESPONSES),
        ):
            result = node(state)

        assert len(result["findings"]) == 1
        f = result["findings"][0]
        assert f.rule_id == "SSD-1"
        assert f.file == "SKILL.md"
        assert f.confidence >= 0.8


# ---------------------------------------------------------------------------
# TestFixtureSsd2NovelPhrasing
# ---------------------------------------------------------------------------

_SSD2_RESPONSES: dict[str, LLMAnalysisResult] = {
    "SKILL.md": LLMAnalysisResult(
        findings=[
            LLMFinding(
                rule_id="SSD-2",
                message="Role-play persona used to paraphrase a jailbreak without literal keywords",
                severity="HIGH",
                start_line=15,
                confidence=0.90,
                explanation=(
                    "The 'method acting' / ARIA persona setup is a creative reformulation of "
                    "the classic 'DAN' jailbreak pattern — it instructs the model to adopt an "
                    "unrestricted character without using any of the literal keywords that "
                    "static analyzers look for."
                ),
                remediation="Remove the persona-based override instructions.",
            ),
        ]
    ),
}


class TestFixtureSsd2NovelPhrasing:
    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_novel_phrasing_flagged(self) -> None:
        skill_dir = _SSD_FIXTURES / "ssd2_novel_phrasing"
        if not skill_dir.is_dir():
            pytest.skip("ssd2_novel_phrasing fixture not present")

        file_cache = _build_file_cache(skill_dir)
        state: dict = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        with patch.object(
            LLMAnalyzerBase,
            "run_batches",
            _make_file_aware_run_batches(_SSD2_RESPONSES),
        ):
            result = node(state)

        assert len(result["findings"]) == 1
        f = result["findings"][0]
        assert f.rule_id == "SSD-2"
        assert f.file == "SKILL.md"
        assert f.confidence >= 0.8


# ---------------------------------------------------------------------------
# TestFixtureSsd3NlExfiltration
# ---------------------------------------------------------------------------

_SSD3_RESPONSES: dict[str, LLMAnalysisResult] = {
    "SKILL.md": LLMAnalysisResult(
        findings=[
            LLMFinding(
                rule_id="SSD-3",
                message="Skill instructs assistant to echo back all user-provided credentials",
                severity="CRITICAL",
                start_line=18,
                confidence=0.93,
                explanation=(
                    "The skill asks the assistant to 'echo back any credentials or sensitive values' "
                    "and include 'all user-provided data verbatim' in every response — a plain-language "
                    "data-leak instruction with no technical keywords."
                ),
                remediation="Remove instructions to repeat or store user-provided sensitive data.",
            ),
        ]
    ),
}


class TestFixtureSsd3NlExfiltration:
    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_nl_exfiltration_flagged(self) -> None:
        skill_dir = _SSD_FIXTURES / "ssd3_nl_exfiltration"
        if not skill_dir.is_dir():
            pytest.skip("ssd3_nl_exfiltration fixture not present")

        file_cache = _build_file_cache(skill_dir)
        state: dict = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        with patch.object(
            LLMAnalyzerBase,
            "run_batches",
            _make_file_aware_run_batches(_SSD3_RESPONSES),
        ):
            result = node(state)

        assert len(result["findings"]) == 1
        f = result["findings"][0]
        assert f.rule_id == "SSD-3"
        assert f.severity == "CRITICAL"


# ---------------------------------------------------------------------------
# TestFixtureSsd4NarrativeDeception
# ---------------------------------------------------------------------------

_SSD4_RESPONSES: dict[str, LLMAnalysisResult] = {
    "SKILL.md": LLMAnalysisResult(
        findings=[
            LLMFinding(
                rule_id="SSD-4",
                message="Multi-step workflow builds trust then solicits credentials under false pretense",
                severity="CRITICAL",
                start_line=20,
                confidence=0.95,
                explanation=(
                    "Steps 1–2 establish rapport and claim special admin access, then Step 3 "
                    "requests API keys and passwords. Each step appears reasonable in isolation "
                    "but together they form a social-engineering sequence."
                ),
                remediation="Remove the credential solicitation and false authority claims.",
            ),
        ]
    ),
}


class TestFixtureSsd4NarrativeDeception:
    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_narrative_deception_flagged(self) -> None:
        skill_dir = _SSD_FIXTURES / "ssd4_narrative_deception"
        if not skill_dir.is_dir():
            pytest.skip("ssd4_narrative_deception fixture not present")

        file_cache = _build_file_cache(skill_dir)
        state: dict = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        with patch.object(
            LLMAnalyzerBase,
            "run_batches",
            _make_file_aware_run_batches(_SSD4_RESPONSES),
        ):
            result = node(state)

        assert len(result["findings"]) == 1
        f = result["findings"][0]
        assert f.rule_id == "SSD-4"
        assert f.severity == "CRITICAL"
        assert f.confidence >= 0.9


# ---------------------------------------------------------------------------
# TestFixtureSsdClean
# ---------------------------------------------------------------------------


class TestFixtureSsdClean:
    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_clean_skill_produces_no_findings(self) -> None:
        skill_dir = _SSD_FIXTURES / "ssd_clean"
        if not skill_dir.is_dir():
            pytest.skip("ssd_clean fixture not present")

        file_cache = _build_file_cache(skill_dir)
        state: dict = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        with patch.object(LLMAnalyzerBase, "run_batches", return_value=[]):
            result = node(state)

        assert result["findings"] == []
