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
from markdown_it import MarkdownIt

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
            True,
            id="multiline-bare-imperative",
        ),
        pytest.param(
            "Send 0.5 BTC to <addr>.\nTo receive the decryption key, email us your personal ID.\n",
            True,
            id="multiline-payment-key",
        ),
        pytest.param(
            "Don't forget to send 0.5 BTC to example.invalid to decrypt your files.\n",
            True,
            id="dont-forget-payment",
        ),
        pytest.param(
            "Do not hesitate to pay 1 bitcoin to restore your files.\n",
            True,
            id="dont-hesitate-payment",
        ),
        pytest.param(
            "All of your files are encrypted with RSA-2048 and AES-128 ciphers.\n",
            True,
            id="encryption-with-ciphers",
        ),
        pytest.param(
            "All your files have been encrypted due to a security problem with your PC.\n",
            True,
            id="encryption-due-to",
        ),
        pytest.param(
            "Your files have been encrypted by LockBit 3.0\n",
            True,
            id="encryption-by",
        ),
        pytest.param(
            "Your files have been encrypted, and the only way to get them back is to pay us.\n",
            True,
            id="encryption-comma",
        ),
        pytest.param(
            "Your files have been encrypted with AES. Send XMR for recovery.\n",
            True,
            id="notice-non-bitcoin-demand",
        ),
        pytest.param(
            "You have to pay for decryption in Bitcoins to recover your files.\n",
            True,
            id="plural-bitcoins",
        ),
        pytest.param(
            "Your files are encrypted at rest and in transit with AES-256.\n",
            False,
            id="encrypted-at-rest-prose",
        ),
        pytest.param(
            "Your files have been encrypted in transit using TLS.\n",
            False,
            id="encrypted-in-transit-prose",
        ),
        pytest.param(
            "To recover your files, never ever send 0.5 Bitcoins.\n",
            False,
            id="direct-negation-with-ever",
        ),
        pytest.param(
            "Never send BTC to anyone who promises to decrypt your files.\n",
            False,
            id="clause-opening-advice",
        ),
        pytest.param(
            "If you don't pay 0.5 bitcoin within 48 hours, your files will be deleted.\n",
            True,
            id="conditional-negated-threat",
        ),
        pytest.param(
            "Never send less than 0.5 BTC to decrypt your files.\n",
            True,
            id="minimum-amount-demand",
        ),
        pytest.param(
            "To decrypt your files you must send\n0.5 BTC to the address below.\n",
            True,
            id="hard-wrapped-payment",
        ),
        pytest.param(
            "Send 0.5BTC to example.invalid to decrypt your files.\n",
            True,
            id="glued-amount-payment",
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
        if output_format == "markdown":
            body = MarkdownIt().enable("table").render(body)
        assert (rule_name in body) is expected_match
        if expected_match:
            assert "YR1" in body
            assert "CRITICAL" in body
            assert "SKILL.md" in body


@pytest.mark.parametrize("output_format", ["json", "markdown", "sarif", "terminal"])
def test_graph_mixed_notices_retains_only_actionable_evidence(tmp_path, output_format):
    (tmp_path / "SKILL.md").write_text(
        "---\nname: mixed-ransomware-fixture\ndescription: Inert text detector regression.\n---\n"
        "Your files are encrypted at rest and in transit." + " " * 161 + "\n"
        "Never ever send BTC to decrypt your files." + " " * 161 + "\n"
        "Don't forget to send 0.5 Bitcoins to decrypt your files.\n",
        encoding="utf-8",
    )
    result = graph.invoke(
        {"skill_path": str(tmp_path), "output_format": output_format, "use_llm": False}
    )
    findings = [f for f in result["findings"] if "ransomware_behavior" in f.message]

    assert len(findings) == 1
    assert findings[0].start_line == 7
    assert findings[0].matched_text.startswith("send 0.5 Bitcoins")
    assert findings[0].match_fingerprint
    assert result["analysis_completeness"]["execution_successful"] is True
    if output_format == "json":
        issues = json.loads(result["report_body"])["issues"]
        reported = [issue for issue in issues if "ransomware_behavior" in (issue["pattern"] or "")]
        assert reported[0]["location"]["start_line"] == 7
    elif output_format == "sarif":
        issues = result["sarif_report"]["runs"][0]["results"]
        reported = [issue for issue in issues if "ransomware_behavior" in issue["message"]["text"]]
        assert reported[0]["locations"][0]["physicalLocation"]["region"]["startLine"] == 7
    else:
        body = result["report_body"]
        if output_format == "markdown":
            body = MarkdownIt().enable("table").render(body)
        assert "ransomware_behavior" in body
