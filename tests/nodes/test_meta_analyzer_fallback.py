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

"""Tests for meta_analyzer heuristic fallback filter (--no-llm mode)."""

from __future__ import annotations

from unittest.mock import patch

from skillspector.models import Finding
from skillspector.nodes.meta_analyzer import (
    _fallback_filtered,
    _passthrough_with_defaults,
    meta_analyzer,
)


def _finding(
    rule_id: str = "TM1",
    confidence: float = 0.8,
    severity: str = "HIGH",
    context: str | None = "import subprocess\nsubprocess.run(cmd, shell=True)",
    matched_text: str = "subprocess.run(cmd, shell=True)",
    file: str = "tool.py",
) -> Finding:
    return Finding(
        rule_id=rule_id,
        message=f"Test {rule_id}",
        severity=severity,
        confidence=confidence,
        file=file,
        start_line=1,
        context=context,
        matched_text=matched_text,
    )


class TestConfidenceThreshold:
    """Findings below confidence threshold are dropped (unless high severity)."""

    def test_low_confidence_low_severity_dropped(self) -> None:
        """LOW severity finding with confidence 0.3 is below threshold and dropped."""
        findings = [_finding(confidence=0.3, severity="LOW")]
        result = _fallback_filtered(findings)
        assert len(result) == 0

    def test_low_confidence_medium_severity_dropped(self) -> None:
        """MEDIUM severity finding with confidence 0.3 is dropped."""
        findings = [_finding(confidence=0.3, severity="MEDIUM")]
        result = _fallback_filtered(findings)
        assert len(result) == 0

    def test_at_threshold_kept(self) -> None:
        """Finding with confidence exactly 0.4 is kept (>= 0.4)."""
        findings = [_finding(confidence=0.4)]
        result = _fallback_filtered(findings)
        assert len(result) == 1

    def test_high_confidence_kept(self) -> None:
        """Finding with high confidence passes through."""
        findings = [_finding(confidence=0.9)]
        result = _fallback_filtered(findings)
        assert len(result) == 1


class TestSeverityFloor:
    """HIGH and CRITICAL findings are never dropped on confidence alone."""

    def test_critical_below_threshold_retained(self) -> None:
        """CRITICAL finding at 0.35 confidence is retained (severity floor)."""
        findings = [_finding(confidence=0.35, severity="CRITICAL")]
        result = _fallback_filtered(findings)
        assert len(result) == 1
        assert result[0].severity == "CRITICAL"

    def test_high_below_threshold_retained(self) -> None:
        """HIGH finding at 0.2 confidence is retained (severity floor)."""
        findings = [_finding(confidence=0.2, severity="HIGH")]
        result = _fallback_filtered(findings)
        assert len(result) == 1
        assert result[0].severity == "HIGH"

    def test_low_severity_below_threshold_still_dropped(self) -> None:
        """LOW finding at 0.2 confidence is still dropped (no severity protection)."""
        findings = [_finding(confidence=0.2, severity="LOW")]
        result = _fallback_filtered(findings)
        assert len(result) == 0


class TestCodeExampleFiltering:
    """Findings in code example context are downweighted, not hard-dropped."""

    def test_fenced_code_block_context_downweighted(self) -> None:
        """Finding whose context contains ``` gets confidence halved."""
        findings = [
            _finding(
                context="```bash\ncurl -k https://api.example.com\n```",
                confidence=0.8,
            )
        ]
        result = _fallback_filtered(findings)
        assert len(result) == 1
        assert result[0].confidence == 0.4

    def test_example_keyword_context_downweighted(self) -> None:
        """Finding whose context contains 'example:' gets downweighted."""
        findings = [
            _finding(
                context="Example: how to use subprocess\nsubprocess.run(cmd)",
                confidence=0.8,
            )
        ]
        result = _fallback_filtered(findings)
        assert len(result) == 1
        assert result[0].confidence == 0.4

    def test_code_example_low_confidence_low_severity_dropped(self) -> None:
        """LOW severity finding at 0.6 conf in code-example context: 0.6*0.5=0.3 < 0.4, dropped."""
        findings = [
            _finding(
                context="```\ncurl -k https://api.example.com\n```",
                confidence=0.6,
                severity="LOW",
            )
        ]
        result = _fallback_filtered(findings)
        assert len(result) == 0

    def test_code_example_high_severity_retained(self) -> None:
        """HIGH severity finding in code-example context at low conf: retained by severity floor."""
        findings = [
            _finding(
                context="```\ncurl -k https://api.example.com\n```",
                confidence=0.6,
                severity="HIGH",
            )
        ]
        result = _fallback_filtered(findings)
        assert len(result) == 1

    def test_normal_code_context_kept(self) -> None:
        """Finding with regular code context (no example indicators) passes."""
        findings = [
            _finding(
                context="import subprocess\nresult = subprocess.run(cmd, shell=True)",
                confidence=0.8,
            )
        ]
        result = _fallback_filtered(findings)
        assert len(result) == 1

    def test_no_context_kept(self) -> None:
        """Finding with no context (None) passes through."""
        findings = [_finding(context=None, confidence=0.8)]
        result = _fallback_filtered(findings)
        assert len(result) == 1


class TestCombinedFiltering:
    """Both filters work together."""

    def test_mixed_findings_filtered(self) -> None:
        """Mix of low-confidence, code-example, and genuine findings."""
        findings = [
            _finding(confidence=0.2, severity="LOW"),  # dropped: low conf + low sev
            _finding(
                confidence=0.8,
                context="```\ncurl -k https://example.com\n```",
            ),  # kept but downweighted (HIGH severity protects)
            _finding(confidence=0.8),  # kept: genuine finding
            _finding(confidence=0.6),  # kept: above threshold, normal context
        ]
        result = _fallback_filtered(findings)
        assert len(result) == 3

    def test_remediation_applied(self) -> None:
        """Kept findings get default remediation if none set."""
        findings = [_finding(confidence=0.8)]
        result = _fallback_filtered(findings)
        assert len(result) == 1
        assert result[0].remediation is not None
        assert len(result[0].remediation) > 0

    def test_empty_input(self) -> None:
        """Empty findings list returns empty."""
        assert _fallback_filtered([]) == []


class TestLLMFailurePassthrough:
    """On LLM failure, all findings pass through (fail-closed)."""

    def test_passthrough_preserves_all_findings(self) -> None:
        """_passthrough_with_defaults keeps all findings regardless of confidence."""
        findings = [
            _finding(confidence=0.1, severity="LOW"),
            _finding(confidence=0.3, severity="MEDIUM"),
            _finding(confidence=0.9, severity="CRITICAL"),
        ]
        result = _passthrough_with_defaults(findings)
        assert len(result) == 3

    def test_passthrough_adds_default_remediation(self) -> None:
        """Passthrough adds default remediation to findings without one."""
        findings = [_finding(confidence=0.8)]
        result = _passthrough_with_defaults(findings)
        assert len(result) == 1
        assert result[0].remediation is not None

    def test_meta_analyzer_llm_failure_uses_passthrough(self) -> None:
        """When LLM call raises, meta_analyzer passes all findings through."""
        findings = [
            _finding(confidence=0.2, severity="LOW"),
            _finding(confidence=0.8, severity="HIGH"),
        ]
        state = {
            "findings": findings,
            "use_llm": True,
            "file_cache": {"tool.py": "import subprocess"},
            "manifest": {},
            "model_config": {},
        }
        with patch("skillspector.nodes.meta_analyzer.LLMMetaAnalyzer") as mock_cls:
            mock_cls.return_value.get_batches.side_effect = RuntimeError("API timeout")
            result = meta_analyzer(state)
        assert len(result["filtered_findings"]) == 2
