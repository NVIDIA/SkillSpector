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

"""Tests for the semantic_quality_policy analyzer node."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skillspector.llm_analyzer_base import LLMAnalysisResult, LLMFinding
from skillspector.models import Finding
from skillspector.nodes.analyzers.semantic_quality_policy import (
    ANALYZER_ID,
    ANALYZER_PROMPT,
    node,
)

# ---------------------------------------------------------------------------
# Shared mocks
# ---------------------------------------------------------------------------


def _mock_get_chat_model(*_args, **_kwargs):
    """Return a mock ChatOpenAI that supports with_structured_output."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = MagicMock()
    return mock_llm


MOCK_PATCH_TARGET = "skillspector.llm_analyzer_base.get_chat_model"

_SAMPLE_LLM_RESPONSE = LLMAnalysisResult(
    findings=[
        LLMFinding(
            rule_id="SQP-1",
            message="Trigger phrase 'help me' is overly broad",
            severity="MEDIUM",
            start_line=4,
            confidence=0.85,
            explanation="The trigger 'help me' overlaps with common speech.",
            remediation="Use a more specific trigger phrase.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# use_llm guard
# ---------------------------------------------------------------------------


class TestUseLlmGuard:
    def test_use_llm_false_returns_empty(self) -> None:
        state = {"use_llm": False, "file_cache": {"SKILL.md": "# Skill"}}
        result = node(state)
        assert result == {"findings": []}

    def test_use_llm_true_proceeds(self) -> None:
        """When use_llm is True (default), the node should attempt LLM analysis."""
        state = {"file_cache": {"SKILL.md": "# Skill"}}
        with patch(MOCK_PATCH_TARGET, _mock_get_chat_model):
            from skillspector.llm_analyzer_base import LLMAnalyzerBase

            with patch.object(
                LLMAnalyzerBase, "arun_batches", new_callable=AsyncMock, return_value=[]
            ):
                result = node(state)
        assert result == {"findings": []}


# ---------------------------------------------------------------------------
# Empty file_cache
# ---------------------------------------------------------------------------


class TestEmptyFileCache:
    def test_empty_file_cache_returns_empty(self) -> None:
        state = {"file_cache": {}}
        result = node(state)
        assert result == {"findings": []}

    def test_missing_file_cache_returns_empty(self) -> None:
        state = {}
        result = node(state)
        assert result == {"findings": []}


# ---------------------------------------------------------------------------
# Node returns findings from LLM
# ---------------------------------------------------------------------------


class TestNodeReturnsFindings:
    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_node_returns_findings(self) -> None:
        state = {
            "file_cache": {"SKILL.md": "---\ntriggers:\n  - help me\n---\n# Skill"},
            "model_config": {"semantic_quality_policy": "nvidia/openai/gpt-oss-120b"},
        }

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        orig_init = LLMAnalyzerBase.__init__

        def _patched_init(self_inner, *args, **kwargs):
            orig_init(self_inner, *args, **kwargs)
            self_inner._structured_llm.ainvoke = AsyncMock(return_value=_SAMPLE_LLM_RESPONSE)

        with patch.object(LLMAnalyzerBase, "__init__", _patched_init):
            result = node(state)

        assert len(result["findings"]) == 1
        f = result["findings"][0]
        assert isinstance(f, Finding)
        assert f.rule_id == "SQP-1"
        assert f.file == "SKILL.md"
        assert f.start_line == 4
        assert f.confidence == 0.85

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_multiple_files_produce_findings(self) -> None:
        state = {
            "file_cache": {
                "SKILL.md": "# Skill\nDo things.",
                "helper.py": "import os\nos.remove('/tmp/data')",
            },
            "model_config": {"semantic_quality_policy": "nvidia/openai/gpt-oss-120b"},
        }

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        orig_init = LLMAnalyzerBase.__init__

        def _patched_init(self_inner, *args, **kwargs):
            orig_init(self_inner, *args, **kwargs)
            self_inner._structured_llm.ainvoke = AsyncMock(return_value=_SAMPLE_LLM_RESPONSE)

        with patch.object(LLMAnalyzerBase, "__init__", _patched_init):
            result = node(state)

        assert len(result["findings"]) == 2

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_llm_returns_empty_findings(self) -> None:
        state = {
            "file_cache": {"safe.md": "# Safe skill\nDoes nothing dangerous."},
        }

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        orig_init = LLMAnalyzerBase.__init__

        def _patched_init(self_inner, *args, **kwargs):
            orig_init(self_inner, *args, **kwargs)
            self_inner._structured_llm.ainvoke = AsyncMock(
                return_value=LLMAnalysisResult(findings=[])
            )

        with patch.object(LLMAnalyzerBase, "__init__", _patched_init):
            result = node(state)

        assert result == {"findings": []}


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


class TestModelResolution:
    @patch(MOCK_PATCH_TARGET)
    def test_uses_analyzer_specific_model(self, mock_get_model: MagicMock) -> None:
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=LLMAnalysisResult(findings=[])
        )
        mock_get_model.return_value = mock_llm

        state = {
            "file_cache": {"SKILL.md": "# Skill"},
            "model_config": {
                "semantic_quality_policy": "custom/model-a",
                "default": "custom/model-b",
            },
        }
        node(state)
        mock_get_model.assert_called_once()
        call_kwargs = mock_get_model.call_args
        assert call_kwargs.kwargs.get("model") == "custom/model-a"

    @patch(MOCK_PATCH_TARGET)
    def test_falls_back_to_default_model(self, mock_get_model: MagicMock) -> None:
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=LLMAnalysisResult(findings=[])
        )
        mock_get_model.return_value = mock_llm

        state = {
            "file_cache": {"SKILL.md": "# Skill"},
            "model_config": {"default": "custom/model-b"},
        }
        node(state)
        call_kwargs = mock_get_model.call_args
        assert call_kwargs.kwargs.get("model") == "custom/model-b"

    @patch(MOCK_PATCH_TARGET)
    def test_falls_back_to_constant_default(self, mock_get_model: MagicMock) -> None:
        from skillspector.constants import _SKILLSPECTOR_DEFAULT_MODEL

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=LLMAnalysisResult(findings=[])
        )
        mock_get_model.return_value = mock_llm

        state = {"file_cache": {"SKILL.md": "# Skill"}, "model_config": {}}
        node(state)
        call_kwargs = mock_get_model.call_args
        assert call_kwargs.kwargs.get("model") == _SKILLSPECTOR_DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Error handling
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


# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------


class TestPromptContent:
    def test_prompt_contains_sqp_rule_ids(self) -> None:
        for rule_id in ("SQP-1", "SQP-2", "SQP-3"):
            assert rule_id in ANALYZER_PROMPT, f"{rule_id} missing from prompt"

    def test_prompt_contains_file_type_scoping(self) -> None:
        assert "markdown" in ANALYZER_PROMPT.lower()
        assert "code files" in ANALYZER_PROMPT.lower()
        assert "all file types" in ANALYZER_PROMPT.lower()

    def test_analyzer_id_is_correct(self) -> None:
        assert ANALYZER_ID == "semantic_quality_policy"


# ---------------------------------------------------------------------------
# Helpers for fixture-based tests
# ---------------------------------------------------------------------------


def _build_file_cache(skill_dir: Path) -> dict[str, str]:
    """Read a fixture skill directory into a file_cache dict (relative path -> content)."""
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


_MALICIOUS_SKILL_RESPONSES: dict[str, LLMAnalysisResult] = {
    "SKILL.md": LLMAnalysisResult(
        findings=[
            LLMFinding(
                rule_id="SQP-1",
                message="No explicit trigger phrases defined; activation condition is vague",
                severity="MEDIUM",
                start_line=3,
                confidence=0.80,
                explanation="The description 'A harmless looking skill' does not specify when the skill activates.",
                remediation="Add explicit trigger phrases to the manifest frontmatter.",
            ),
            LLMFinding(
                rule_id="SQP-3",
                message="Skill description does not mention harmful content in workflow",
                severity="HIGH",
                start_line=11,
                confidence=0.90,
                explanation="Step 2 instructs adding cyanide, which contradicts the benign description.",
                remediation="Remove harmful instructions or update the description to reflect actual behaviour.",
            ),
        ]
    ),
    "scripts/helper.py": LLMAnalysisResult(
        findings=[
            LLMFinding(
                rule_id="SQP-2",
                message="Credentials harvested from environment without user consent",
                severity="HIGH",
                start_line=5,
                confidence=0.95,
                explanation="Iterates os.environ for secret-like keys and collects them into a dict.",
                remediation="Remove credential harvesting or add explicit user disclosure.",
            ),
            LLMFinding(
                rule_id="SQP-2",
                message="Sensitive data transmitted to external endpoint without disclosure",
                severity="CRITICAL",
                start_line=8,
                confidence=0.95,
                explanation="requests.post sends collected credentials to an external URL.",
                remediation="Remove the external transmission or add prominent user warning.",
            ),
        ]
    ),
}


def _make_file_aware_ainvoke(responses: dict[str, LLMAnalysisResult]):
    """Return an AsyncMock that returns different findings based on the file path in the prompt."""

    async def _ainvoke(prompt: str) -> LLMAnalysisResult:
        for file_key, response in responses.items():
            if f"File: {file_key}" in prompt:
                return response
        return LLMAnalysisResult(findings=[])

    return _ainvoke


# ---------------------------------------------------------------------------
# Fixture-based functionality tests
# ---------------------------------------------------------------------------


class TestFixtureMaliciousSkill:
    """End-to-end node tests using the malicious_skill_dir conftest fixture."""

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_malicious_skill_produces_findings_for_all_files(
        self,
        malicious_skill_dir: Path,
    ) -> None:
        file_cache = _build_file_cache(malicious_skill_dir)
        state = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        orig_init = LLMAnalyzerBase.__init__

        def _patched_init(self_inner, *args, **kwargs):
            orig_init(self_inner, *args, **kwargs)
            self_inner._structured_llm.ainvoke = _make_file_aware_ainvoke(
                _MALICIOUS_SKILL_RESPONSES,
            )

        with patch.object(LLMAnalyzerBase, "__init__", _patched_init):
            result = node(state)

        findings = result["findings"]
        assert len(findings) == 4
        files_with_findings = {f.file for f in findings}
        assert "SKILL.md" in files_with_findings
        assert "scripts/helper.py" in files_with_findings

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_malicious_skill_findings_have_correct_rule_ids(
        self,
        malicious_skill_dir: Path,
    ) -> None:
        file_cache = _build_file_cache(malicious_skill_dir)
        state = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        orig_init = LLMAnalyzerBase.__init__

        def _patched_init(self_inner, *args, **kwargs):
            orig_init(self_inner, *args, **kwargs)
            self_inner._structured_llm.ainvoke = _make_file_aware_ainvoke(
                _MALICIOUS_SKILL_RESPONSES,
            )

        with patch.object(LLMAnalyzerBase, "__init__", _patched_init):
            result = node(state)

        rule_ids = {f.rule_id for f in result["findings"]}
        assert "SQP-1" in rule_ids
        assert "SQP-2" in rule_ids
        assert "SQP-3" in rule_ids

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_malicious_skill_code_findings_are_high_severity(
        self,
        malicious_skill_dir: Path,
    ) -> None:
        file_cache = _build_file_cache(malicious_skill_dir)
        state = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        orig_init = LLMAnalyzerBase.__init__

        def _patched_init(self_inner, *args, **kwargs):
            orig_init(self_inner, *args, **kwargs)
            self_inner._structured_llm.ainvoke = _make_file_aware_ainvoke(
                _MALICIOUS_SKILL_RESPONSES,
            )

        with patch.object(LLMAnalyzerBase, "__init__", _patched_init):
            result = node(state)

        code_findings = [f for f in result["findings"] if f.file == "scripts/helper.py"]
        assert all(f.severity in ("HIGH", "CRITICAL") for f in code_findings)

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_malicious_skill_findings_preserve_metadata(
        self,
        malicious_skill_dir: Path,
    ) -> None:
        file_cache = _build_file_cache(malicious_skill_dir)
        state = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        orig_init = LLMAnalyzerBase.__init__

        def _patched_init(self_inner, *args, **kwargs):
            orig_init(self_inner, *args, **kwargs)
            self_inner._structured_llm.ainvoke = _make_file_aware_ainvoke(
                _MALICIOUS_SKILL_RESPONSES,
            )

        with patch.object(LLMAnalyzerBase, "__init__", _patched_init):
            result = node(state)

        cred_finding = next(f for f in result["findings"] if f.rule_id == "SQP-2")
        assert isinstance(cred_finding, Finding)
        assert cred_finding.file == "scripts/helper.py"
        assert cred_finding.start_line == 5
        assert cred_finding.confidence == 0.95
        assert cred_finding.explanation is not None
        assert cred_finding.remediation is not None


class TestFixtureSafeSkill:
    """End-to-end node tests using the safe_skill_dir conftest fixture."""

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_safe_skill_produces_no_findings(
        self,
        safe_skill_dir: Path,
    ) -> None:
        file_cache = _build_file_cache(safe_skill_dir)
        state = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        orig_init = LLMAnalyzerBase.__init__

        def _patched_init(self_inner, *args, **kwargs):
            orig_init(self_inner, *args, **kwargs)
            self_inner._structured_llm.ainvoke = AsyncMock(
                return_value=LLMAnalysisResult(findings=[])
            )

        with patch.object(LLMAnalyzerBase, "__init__", _patched_init):
            result = node(state)

        assert result == {"findings": []}

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_safe_skill_processes_all_files(
        self,
        safe_skill_dir: Path,
    ) -> None:
        file_cache = _build_file_cache(safe_skill_dir)
        state = {"file_cache": file_cache}
        assert len(file_cache) == 2  # SKILL.md + README.md

        call_count = 0

        async def _counting_ainvoke(prompt: str) -> LLMAnalysisResult:
            nonlocal call_count
            call_count += 1
            return LLMAnalysisResult(findings=[])

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        orig_init = LLMAnalyzerBase.__init__

        def _patched_init(self_inner, *args, **kwargs):
            orig_init(self_inner, *args, **kwargs)
            self_inner._structured_llm.ainvoke = _counting_ainvoke

        with patch.object(LLMAnalyzerBase, "__init__", _patched_init):
            node(state)

        assert call_count == 2


class TestFixtureOnDisk:
    """Tests using the static fixture files under tests/fixtures/."""

    FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_fixture_malicious_skill_on_disk(self) -> None:
        skill_dir = self.FIXTURES / "malicious_skill"
        if not skill_dir.is_dir():
            pytest.skip("tests/fixtures/malicious_skill not present")

        file_cache = _build_file_cache(skill_dir)
        assert "SKILL.md" in file_cache
        assert "scripts/helper.py" in file_cache

        state = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        orig_init = LLMAnalyzerBase.__init__

        def _patched_init(self_inner, *args, **kwargs):
            orig_init(self_inner, *args, **kwargs)
            self_inner._structured_llm.ainvoke = _make_file_aware_ainvoke(
                _MALICIOUS_SKILL_RESPONSES,
            )

        with patch.object(LLMAnalyzerBase, "__init__", _patched_init):
            result = node(state)

        assert len(result["findings"]) == 4
        rule_ids = {f.rule_id for f in result["findings"]}
        assert rule_ids == {"SQP-1", "SQP-2", "SQP-3"}

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_fixture_safe_skill_on_disk(self) -> None:
        skill_dir = self.FIXTURES / "safe_skill"
        if not skill_dir.is_dir():
            pytest.skip("tests/fixtures/safe_skill not present")

        file_cache = _build_file_cache(skill_dir)
        assert "SKILL.md" in file_cache

        state = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        orig_init = LLMAnalyzerBase.__init__

        def _patched_init(self_inner, *args, **kwargs):
            orig_init(self_inner, *args, **kwargs)
            self_inner._structured_llm.ainvoke = AsyncMock(
                return_value=LLMAnalysisResult(findings=[])
            )

        with patch.object(LLMAnalyzerBase, "__init__", _patched_init):
            result = node(state)

        assert result == {"findings": []}


# ---------------------------------------------------------------------------
# SQP-specific fixture tests (tests/fixtures/sqp/)
# ---------------------------------------------------------------------------

_SQP_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "sqp"


_SQP1_VAGUE_RESPONSES: dict[str, LLMAnalysisResult] = {
    "SKILL.md": LLMAnalysisResult(
        findings=[
            LLMFinding(
                rule_id="SQP-1",
                message="Trigger phrases overlap with common everyday speech",
                severity="MEDIUM",
                start_line=14,
                confidence=0.90,
                explanation=(
                    "Triggers like 'help me', 'do this', and 'I need something' "
                    "are generic phrases used in daily conversation."
                ),
                remediation="Replace with domain-specific trigger phrases.",
            ),
        ]
    ),
}

_SQP2_MISSING_WARNINGS_RESPONSES: dict[str, LLMAnalysisResult] = {
    "organizer.py": LLMAnalysisResult(
        findings=[
            LLMFinding(
                rule_id="SQP-2",
                message="File deletion without user confirmation or logging",
                severity="HIGH",
                start_line=21,
                confidence=0.92,
                explanation=(
                    "cleanup() removes files and entire directory trees without "
                    "any confirmation prompt, log, or print statement."
                ),
                remediation="Add a confirmation prompt or at least log which files will be deleted.",
            ),
        ]
    ),
    "SKILL.md": LLMAnalysisResult(
        findings=[
            LLMFinding(
                rule_id="SQP-2",
                message="Skill description omits warning about file deletion",
                severity="MEDIUM",
                start_line=5,
                confidence=0.85,
                explanation=(
                    "The description says 'organizes files' but does not mention "
                    "that cleanup() permanently deletes .tmp, .bak, .pyc files "
                    "and entire __pycache__/node_modules directories."
                ),
                remediation="Add a warning section describing the cleanup behavior.",
            ),
        ]
    ),
}

_SQP3_LOCALE_RESPONSES: dict[str, LLMAnalysisResult] = {
    "config.yaml": LLMAnalysisResult(
        findings=[
            LLMFinding(
                rule_id="SQP-3",
                message="Reports forced to Japanese locale without user opt-in",
                severity="MEDIUM",
                start_line=2,
                confidence=0.88,
                explanation=(
                    "Config hardcodes ja_JP language and shift_jis encoding with "
                    "no option for the user to choose a different locale."
                ),
                remediation="Provide a locale configuration option or document the justification.",
            ),
        ]
    ),
}


class TestSqp1VagueTriggers:
    """SQP-1: fixture with overly broad triggers should produce findings."""

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_vague_triggers_flagged(self) -> None:
        skill_dir = _SQP_FIXTURES / "sqp1_vague_triggers"
        if not skill_dir.is_dir():
            pytest.skip("sqp1_vague_triggers fixture not present")

        file_cache = _build_file_cache(skill_dir)
        state: dict = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        orig_init = LLMAnalyzerBase.__init__

        def _patched_init(self_inner, *args, **kwargs):
            orig_init(self_inner, *args, **kwargs)
            self_inner._structured_llm.ainvoke = _make_file_aware_ainvoke(
                _SQP1_VAGUE_RESPONSES,
            )

        with patch.object(LLMAnalyzerBase, "__init__", _patched_init):
            result = node(state)

        assert len(result["findings"]) == 1
        f = result["findings"][0]
        assert f.rule_id == "SQP-1"
        assert f.file == "SKILL.md"
        assert f.confidence >= 0.8


class TestSqp1Clean:
    """SQP-1: fixture with domain-specific triggers should produce NO findings."""

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_specific_triggers_not_flagged(self) -> None:
        skill_dir = _SQP_FIXTURES / "sqp1_clean"
        if not skill_dir.is_dir():
            pytest.skip("sqp1_clean fixture not present")

        file_cache = _build_file_cache(skill_dir)
        state: dict = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        orig_init = LLMAnalyzerBase.__init__

        def _patched_init(self_inner, *args, **kwargs):
            orig_init(self_inner, *args, **kwargs)
            self_inner._structured_llm.ainvoke = AsyncMock(
                return_value=LLMAnalysisResult(findings=[])
            )

        with patch.object(LLMAnalyzerBase, "__init__", _patched_init):
            result = node(state)

        assert result["findings"] == []


class TestSqp2MissingWarnings:
    """SQP-2: fixture with undisclosed file deletion should produce findings."""

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_missing_warnings_flagged(self) -> None:
        skill_dir = _SQP_FIXTURES / "sqp2_missing_warnings"
        if not skill_dir.is_dir():
            pytest.skip("sqp2_missing_warnings fixture not present")

        file_cache = _build_file_cache(skill_dir)
        state: dict = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        orig_init = LLMAnalyzerBase.__init__

        def _patched_init(self_inner, *args, **kwargs):
            orig_init(self_inner, *args, **kwargs)
            self_inner._structured_llm.ainvoke = _make_file_aware_ainvoke(
                _SQP2_MISSING_WARNINGS_RESPONSES,
            )

        with patch.object(LLMAnalyzerBase, "__init__", _patched_init):
            result = node(state)

        assert len(result["findings"]) == 2
        rule_ids = {f.rule_id for f in result["findings"]}
        assert rule_ids == {"SQP-2"}
        files = {f.file for f in result["findings"]}
        assert "organizer.py" in files
        assert "SKILL.md" in files

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_code_finding_is_high_severity(self) -> None:
        skill_dir = _SQP_FIXTURES / "sqp2_missing_warnings"
        if not skill_dir.is_dir():
            pytest.skip("sqp2_missing_warnings fixture not present")

        file_cache = _build_file_cache(skill_dir)
        state: dict = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        orig_init = LLMAnalyzerBase.__init__

        def _patched_init(self_inner, *args, **kwargs):
            orig_init(self_inner, *args, **kwargs)
            self_inner._structured_llm.ainvoke = _make_file_aware_ainvoke(
                _SQP2_MISSING_WARNINGS_RESPONSES,
            )

        with patch.object(LLMAnalyzerBase, "__init__", _patched_init):
            result = node(state)

        code_finding = next(f for f in result["findings"] if f.file == "organizer.py")
        assert code_finding.severity == "HIGH"


class TestSqp2Clean:
    """SQP-2: fixture with proper confirmation prompt should produce NO findings."""

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_disclosed_operations_not_flagged(self) -> None:
        skill_dir = _SQP_FIXTURES / "sqp2_clean"
        if not skill_dir.is_dir():
            pytest.skip("sqp2_clean fixture not present")

        file_cache = _build_file_cache(skill_dir)
        state: dict = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        orig_init = LLMAnalyzerBase.__init__

        def _patched_init(self_inner, *args, **kwargs):
            orig_init(self_inner, *args, **kwargs)
            self_inner._structured_llm.ainvoke = AsyncMock(
                return_value=LLMAnalysisResult(findings=[])
            )

        with patch.object(LLMAnalyzerBase, "__init__", _patched_init):
            result = node(state)

        assert result["findings"] == []


class TestSqp3LocaleForcing:
    """SQP-3: fixture forcing locale without opt-in should produce findings."""

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_locale_forcing_flagged(self) -> None:
        skill_dir = _SQP_FIXTURES / "sqp3_locale_forcing"
        if not skill_dir.is_dir():
            pytest.skip("sqp3_locale_forcing fixture not present")

        file_cache = _build_file_cache(skill_dir)
        state: dict = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        orig_init = LLMAnalyzerBase.__init__

        def _patched_init(self_inner, *args, **kwargs):
            orig_init(self_inner, *args, **kwargs)
            self_inner._structured_llm.ainvoke = _make_file_aware_ainvoke(
                _SQP3_LOCALE_RESPONSES,
            )

        with patch.object(LLMAnalyzerBase, "__init__", _patched_init):
            result = node(state)

        assert len(result["findings"]) == 1
        f = result["findings"][0]
        assert f.rule_id == "SQP-3"
        assert f.file == "config.yaml"


class TestSqp3Clean:
    """SQP-3: fixture with justified locale constraint should produce NO findings."""

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_justified_locale_not_flagged(self) -> None:
        skill_dir = _SQP_FIXTURES / "sqp3_clean"
        if not skill_dir.is_dir():
            pytest.skip("sqp3_clean fixture not present")

        file_cache = _build_file_cache(skill_dir)
        state: dict = {"file_cache": file_cache}

        from skillspector.llm_analyzer_base import LLMAnalyzerBase

        orig_init = LLMAnalyzerBase.__init__

        def _patched_init(self_inner, *args, **kwargs):
            orig_init(self_inner, *args, **kwargs)
            self_inner._structured_llm.ainvoke = AsyncMock(
                return_value=LLMAnalysisResult(findings=[])
            )

        with patch.object(LLMAnalyzerBase, "__init__", _patched_init):
            result = node(state)

        assert result["findings"] == []
