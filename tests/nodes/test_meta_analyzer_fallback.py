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

"""Tests for fail-closed meta_analyzer fallback behavior."""

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


class TestConfidencePreservation:
    """Fallback preserves deterministic findings at every confidence."""

    def test_low_confidence_low_severity_retained(self) -> None:
        """LOW-severity deterministic findings remain visible."""
        findings = [_finding(confidence=0.3, severity="LOW")]
        result = _fallback_filtered(findings)
        assert len(result) == 1

    def test_low_confidence_medium_severity_retained(self) -> None:
        """MEDIUM-severity deterministic findings remain visible."""
        findings = [_finding(confidence=0.3, severity="MEDIUM")]
        result = _fallback_filtered(findings)
        assert len(result) == 1

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


class TestSeverityPreservation:
    """Fallback preserves deterministic findings across all severities."""

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

    def test_low_severity_below_threshold_retained(self) -> None:
        """LOW findings are retained even at low confidence."""
        findings = [_finding(confidence=0.2, severity="LOW")]
        result = _fallback_filtered(findings)
        assert len(result) == 1

    def test_none_severity_treated_as_low(self) -> None:
        """Finding with None severity does not crash — treated as LOW."""
        findings = [_finding(confidence=0.8, severity=None)]
        result = _fallback_filtered(findings)
        assert len(result) == 1

    def test_none_severity_below_threshold_retained(self) -> None:
        """Missing severity does not cause a deterministic finding to disappear."""
        findings = [_finding(confidence=0.3, severity=None)]
        result = _fallback_filtered(findings)
        assert len(result) == 1


class TestCodeExamplePreservation:
    """Attacker-controlled example framing cannot downweight findings."""

    def test_fenced_code_block_context_preserves_confidence(self) -> None:
        """Fenced-code framing leaves deterministic confidence unchanged."""
        findings = [
            _finding(
                context="```bash\ncurl -k https://api.example.com\n```",
                confidence=0.8,
            )
        ]
        result = _fallback_filtered(findings)
        assert len(result) == 1
        assert result[0].confidence == 0.8

    def test_example_keyword_context_preserves_confidence(self) -> None:
        """Example-keyword framing leaves deterministic confidence unchanged."""
        findings = [
            _finding(
                context="Example: how to use subprocess\nsubprocess.run(cmd)",
                confidence=0.8,
            )
        ]
        result = _fallback_filtered(findings)
        assert len(result) == 1
        assert result[0].confidence == 0.8

    def test_code_example_low_confidence_low_severity_retained(self) -> None:
        """Example framing cannot remove a LOW-severity finding."""
        findings = [
            _finding(
                context="```\ncurl -k https://api.example.com\n```",
                confidence=0.6,
                severity="LOW",
            )
        ]
        result = _fallback_filtered(findings)
        assert len(result) == 1
        assert result[0].confidence == 0.6

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


class TestCombinedFallback:
    """Mixed deterministic findings all survive fallback."""

    def test_mixed_findings_retained(self) -> None:
        """Confidence, framing, and severity do not remove findings."""
        findings = [
            _finding(confidence=0.2, severity="LOW"),
            _finding(
                confidence=0.8,
                context="```\ncurl -k https://example.com\n```",
            ),
            _finding(confidence=0.8),
            _finding(confidence=0.6),
        ]
        result = _fallback_filtered(findings)
        assert len(result) == 4

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

    def test_fallback_and_passthrough_preserve_security_metadata(self) -> None:
        original = Finding(
            rule_id="TM1",
            message="deterministic",
            severity="MEDIUM",
            confidence=0.2,
            file="tool.py",
            start_line=3,
            intent="malicious",
            evidence={"source": "static", "local_only": True},
            match_fingerprint="sha256:deterministic",
            occurrences=[{"file": "tool.py", "start_line": 3, "end_line": 3}],
        )

        for clone in (
            _fallback_filtered([original])[0],
            _passthrough_with_defaults([original])[0],
        ):
            assert clone.intent == original.intent
            assert clone.evidence == original.evidence
            assert clone.match_fingerprint == original.match_fingerprint
            assert clone.occurrences == original.occurrences

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
        assert len(result["findings"]) == 2
        assert "filtered_findings" not in result
