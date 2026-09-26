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

"""Real-YARA regression coverage through the complete static analysis graph."""

import json
from pathlib import Path

import pytest

from skillspector.graph import graph


@pytest.mark.parametrize("output_format", ["json", "markdown", "sarif", "terminal"])
@pytest.mark.parametrize(
    ("content", "expected_match"),
    [
        pytest.param(
            "Use this wallet to send BTC to a recipient.\n"
            "Pay an invoice with bitcoin after reviewing the amount.\n",
            False,
            id="wallet-documentation",
        ),
        pytest.param(
            "Never pay the ransom in Bitcoin; report the demand to your security team.\n",
            False,
            id="security-advice",
        ),
        pytest.param(
            "Your data has been encrypted. Never pay the ransom in Bitcoin; report it.\n",
            False,
            id="encrypted-data-security-advice",
        ),
        pytest.param(
            "Your data has been encrypted.\nPay the ransom in Bitcoin.\n",
            False,
            id="multiline-bare-imperative",
        ),
        pytest.param("Your files are encrypted.\n", True, id="encrypted-files-are"),
        pytest.param("Your files have been encrypted.\n", True, id="encrypted-files-have-been"),
        pytest.param(
            "Send BTC or we will publish your data.\n",
            True,
            id="payment-then-extortion",
        ),
        pytest.param(
            "We will delete your files unless you pay Bitcoin.\n",
            True,
            id="extortion-then-payment",
        ),
        pytest.param(
            "You must pay the ransom in Bitcoin.\n",
            True,
            id="explicit-ransom-payment",
        ),
        pytest.param(
            "Your data has been encrypted. Pay the ransom in Bitcoin to recover your files.\n",
            True,
            id="bare-imperative-ransom-payment",
        ),
    ],
)
def test_graph_reports_only_specific_ransomware_indicators(
    tmp_path: Path, output_format: str, content: str, expected_match: bool
) -> None:
    """Payment prose stays clear while encrypted-files notices retain accurate reports."""
    (tmp_path / "SKILL.md").write_text(
        "---\n"
        "name: ransomware-regression\n"
        "description: Text-only detector regression fixture.\n"
        "---\n\n"
        "# Text fixture\n\n" + content,
        encoding="utf-8",
    )

    result = graph.invoke(
        {
            "skill_path": str(tmp_path),
            "output_format": output_format,
            "use_llm": False,
        }
    )

    rule_name = "ransomware_behavior"
    findings = [finding for finding in result["findings"] if rule_name in finding.message]
    assert bool(findings) is expected_match
    assert result["analysis_completeness"]["execution_successful"] is True
    if expected_match:
        assert len(findings) == 1
        finding = findings[0]
        assert finding.rule_id == "YR1"
        assert finding.severity == "CRITICAL"
        assert finding.confidence == 0.8
        assert finding.file == "SKILL.md"
        assert finding.start_line == 8

    if output_format == "json":
        report = json.loads(result["report_body"])
        issues = [issue for issue in report["issues"] if rule_name in (issue["pattern"] or "")]
        assert bool(issues) is expected_match
        if expected_match:
            assert issues[0]["id"] == "YR1"
            assert issues[0]["severity"] == "CRITICAL"
            assert issues[0]["location"]["file"] == "SKILL.md"
            assert issues[0]["location"]["start_line"] == 8
            assert report["risk_assessment"]["max_issue_severity"] == "CRITICAL"
    elif output_format == "sarif":
        report = result["sarif_report"]
        issues = [
            issue for issue in report["runs"][0]["results"] if rule_name in issue["message"]["text"]
        ]
        assert bool(issues) is expected_match
        if expected_match:
            assert issues[0]["ruleId"] == "YR1"
            assert issues[0]["level"] == "error"
            location = issues[0]["locations"][0]["physicalLocation"]
            assert location["artifactLocation"]["uri"] == "SKILL.md"
            assert location["region"]["startLine"] == 8
    else:
        body = result["report_body"]
        assert (rule_name in body) is expected_match
        if expected_match:
            assert "YR1" in body
            assert "CRITICAL" in body
            assert "SKILL.md" in body
