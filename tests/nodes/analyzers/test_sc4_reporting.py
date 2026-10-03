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

"""SC4 report guidance must match the evidence, including incomplete lookups."""

import json
from dataclasses import replace

import pytest

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.llm_analyzer_base import Batch
from skillspector.models import Finding, Severity
from skillspector.nodes.analyzers import static_patterns_supply_chain as sc
from skillspector.nodes.analyzers.osv_client import (
    OsvQueryLimitation,
    QueryBatchResults,
    VulnResult,
)
from skillspector.nodes.analyzers.pattern_defaults import get_explanation, get_remediation
from skillspector.nodes.analyzers.static_runner import analyzer_finding_to_finding
from skillspector.nodes.meta_analyzer import (
    LLMMetaAnalyzer,
    MetaAnalyzerFinding,
    MetaAnalyzerResult,
)
from skillspector.nodes.report import report


@pytest.fixture
def advisory():
    # Synthetic data: no real package vulnerability or live OSV lookup.
    return VulnResult("GHSA-test", "Synthetic advisory", "CRITICAL", ("CVE-test",))


@pytest.fixture
def failed_lookup(monkeypatch):
    limitation = OsvQueryLimitation(
        reason=LedgerReason.ANALYZER_RUNTIME_ERROR, error_class="TimeoutError"
    )

    def query(packages, _ecosystem, **_kwargs):
        return QueryBatchResults([[] for _ in packages], limitations=(limitation,))

    monkeypatch.setattr(sc, "query_batch", query)
    monkeypatch.setattr(sc, "was_osv_reachable", lambda: False)
    return limitation


def _sc4(findings):
    return [analyzer_finding_to_finding(f) for f in findings if f.rule_id == "SC4"]


def _assert_report_guidance(finding, output_format):
    result = report(
        {
            "filtered_findings": [finding],
            "component_metadata": [],
            "manifest": {},
            "skill_path": None,
            "has_executable_scripts": False,
            "use_llm": False,
            "output_format": output_format,
        }
    )
    body = result["report_body"]
    if output_format == "json":
        issue = json.loads(body)["issues"][0]
        assert issue["explanation"] == finding.explanation
        assert issue["remediation"] == finding.remediation
    elif output_format == "sarif":
        properties = json.loads(body)["runs"][0]["results"][0]["properties"]
        assert properties["explanation"] == finding.explanation
        assert properties["remediation"] == finding.remediation
    else:
        assert finding.remediation in body


@pytest.mark.parametrize("output_format", ["json", "markdown", "sarif"])
@pytest.mark.parametrize("version", [None, "1.0.0"])
def test_osv_guidance_survives_report_conversion(monkeypatch, advisory, output_format, version):
    monkeypatch.setattr(sc, "query_batch", lambda *_args, **_kwargs: [[advisory]])
    raw, covered = sc._sc4_from_osv(
        [("examplepkg", version, 3)], "PyPI", "requirements.txt", ["supply-chain"]
    )
    finding = _sc4(raw)[0]

    assert covered == {"examplepkg"}
    assert finding.file == "requirements.txt"
    assert finding.start_line == 3
    assert "CVE-test" in finding.message
    if version is None:
        assert raw[0].severity is Severity.LOW
        assert finding.confidence == 0.4
        assert "unknown" in finding.explanation.lower()
        assert "resolved version" in finding.remediation.lower()
        assert "Dependency has known vulnerabilities" not in finding.explanation
    else:
        assert raw[0].severity is Severity.CRITICAL
        assert "1.0.0" in finding.message
        assert "resolved version" in finding.explanation.lower()
        # The OSV result carries advisory IDs, not an available fixed release.
        assert "if a fixed release is available" in finding.remediation.lower()

    _assert_report_guidance(finding, output_format)


@pytest.mark.parametrize("output_format", ["json", "markdown", "sarif"])
@pytest.mark.parametrize("content", ["examplepkg==1.0.0\n", "examplepkg>=1.0.0\n"])
def test_failed_lookup_guidance_and_partial_status_are_preserved(
    failed_lookup, content, output_format
):
    response = sc.node(
        {
            "skill_path": "",
            "components": ["requirements.txt"],
            "file_cache": {"requirements.txt": content},
            "local_file_cache": {"requirements.txt": content},
            "manifest": {},
            "component_metadata": [],
        }
    )
    findings = [f for f in response["findings"] if f.rule_id == "SC4"]
    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity == "LOW"
    assert "OSV.dev unreachable" in finding.message
    assert "incomplete" in finding.explanation.lower()
    assert "does not establish" in finding.explanation.lower()
    assert "retry" in finding.remediation.lower()
    assert "patched version" not in finding.remediation.lower()
    assert any(
        event["outcome"] is LedgerOutcome.PARTIAL and event["reason_code"] is failed_lookup.reason
        for event in response["inspection_ledger"]
    )
    assert response["analyzer_status_events"][0]["status"] == "degraded"
    _assert_report_guidance(finding, output_format)


@pytest.mark.parametrize("max_safe", [None, "2.0.0"])
def test_fallback_evidence_keeps_vulnerability_and_appropriate_guidance(max_safe):
    raw = sc._sc4_from_fallback(
        [("examplepkg", "1.0.0", 5)],
        [("examplepkg", max_safe, "Synthetic advisory", 0.8)],
        "requirements.txt",
        ["supply-chain"],
    )
    finding = _sc4(raw)[0]
    assert finding.severity == "HIGH"
    assert finding.confidence == 0.8
    assert finding.start_line == 5
    assert "static fallback" in finding.explanation.lower()
    if max_safe is None:
        assert "patched version" not in finding.remediation.lower()
        assert "replace" in finding.remediation.lower()
    else:
        assert "2.0.0" in finding.remediation


def test_failed_osv_lookup_retains_positive_fallback_evidence(monkeypatch, failed_lookup):
    monkeypatch.setattr(
        sc, "_FALLBACK_VULNERABLE_PYPI", [("examplepkg", "2.0.0", "Synthetic", 0.8)]
    )
    raw, limitations, count = sc._analyze_dependencies_detailed(
        "examplepkg==1.0.0\n", "requirements.txt"
    )
    findings = _sc4(raw)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"
    assert "static fallback" in findings[0].explanation.lower()
    assert limitations == [failed_lookup]
    assert count == 1


@pytest.mark.parametrize("content", ["examplepkg==1.0.0\n", "examplepkg>=1.0.0\n"])
def test_successful_empty_lookup_does_not_add_a_vulnerability(monkeypatch, content):
    monkeypatch.setattr(sc, "query_batch", lambda *_args, **_kwargs: [[]])
    monkeypatch.setattr(sc, "was_osv_reachable", lambda: True)
    raw, limitations, count = sc._analyze_dependencies_detailed(content, "requirements.txt")
    assert _sc4(raw) == []
    assert limitations == []
    assert count == 1


def _confirmed_sc4(findings, *, text=None, coarse=False):
    # Exercise the real schema/parse/filter path, without constructing a provider.
    analyzer = LLMMetaAnalyzer.__new__(LLMMetaAnalyzer)
    batch = Batch(file_path="requirements.txt", content="", findings=findings)
    fields = {} if text is None else {"explanation": text, "remediation": text}
    response = MetaAnalyzerResult(
        findings=[
            MetaAnalyzerFinding(
                pattern_id="SC4",
                start_line=None if coarse else finding.start_line,
                end_line=None if coarse else finding.end_line,
                is_vulnerability=True,
                confidence=0.7,
                intent="negligent",
                impact="low",
                **fields,
            )
            for finding in findings
        ]
    )
    return analyzer.apply_filter(findings, [(batch, analyzer.parse_response(response, batch))])


@pytest.mark.parametrize("output_format", ["json", "markdown", "sarif"])
@pytest.mark.parametrize("text", [None, "", " \t\n"])
@pytest.mark.parametrize("evidence", ["resolved", "unknown", "fallback", "threshold", "failed"])
def test_confirmed_empty_llm_text_preserves_sc4_evidence_in_reports(
    monkeypatch, advisory, failed_lookup, output_format, text, evidence
):
    if evidence == "failed":
        raw, _, _ = sc._analyze_dependencies_detailed("examplepkg==1.0.0\n", "requirements.txt")
    elif evidence in {"resolved", "unknown"}:
        monkeypatch.setattr(sc, "query_batch", lambda *_args, **_kwargs: [[advisory]])
        version = "1.0.0" if evidence == "resolved" else None
        raw, _ = sc._sc4_from_osv(
            [("examplepkg", version, 3)], "PyPI", "requirements.txt", ["supply-chain"]
        )
    else:
        threshold = "2.0.0" if evidence == "threshold" else None
        raw = sc._sc4_from_fallback(
            [("examplepkg", "1.0.0", 5)],
            [("examplepkg", threshold, "Synthetic advisory", 0.8)],
            "requirements.txt",
            ["supply-chain"],
        )
    [original] = _sc4(raw)
    [confirmed] = _confirmed_sc4([original], text=text)

    assert confirmed.explanation == original.explanation
    assert confirmed.remediation == original.remediation
    assert confirmed.severity == original.severity
    assert confirmed.confidence == max(original.confidence, 0.7)
    assert confirmed.finding_id == original.finding_id
    assert confirmed.evidence == original.evidence
    assert confirmed.occurrences == original.occurrences
    assert confirmed.tags == original.tags
    assert original.message == raw[0].message
    _assert_report_guidance(confirmed, output_format)


@pytest.mark.parametrize("text", [None, "", " \t\n"])
@pytest.mark.parametrize("original_text", [None, "", " \t\n"])
def test_confirmed_sc4_without_evidence_text_uses_neutral_defaults(text, original_text):
    original = Finding(
        rule_id="SC4",
        message="Dependency lookup requires review",
        confidence=0.8,
        file="requirements.txt",
        explanation=original_text,
        remediation=original_text,
    )
    [confirmed] = _confirmed_sc4([original], text=text)
    assert confirmed.explanation == get_explanation("SC4")
    assert confirmed.remediation == get_remediation("SC4")
    assert "does not by itself confirm" in confirmed.explanation.lower()
    assert "if the resolved release is affected" in confirmed.remediation.lower()
    assert "verified fixed release when available" in confirmed.remediation.lower()
    assert confirmed.confidence == original.confidence


def test_coarse_sc4_confirmation_retains_each_findings_own_guidance():
    first = Finding(
        rule_id="SC4",
        message="unknown version",
        file="requirements.txt",
        start_line=3,
        explanation="Resolved version is unknown.",
        remediation="Resolve the version first.",
    )
    second = replace(
        first,
        start_line=7,
        message="lookup failed",
        explanation="Coverage is incomplete.",
        remediation="Retry the lookup.",
    )
    confirmed = _confirmed_sc4([first, second], coarse=True)
    assert [f.explanation for f in confirmed] == [first.explanation, second.explanation]
    assert [f.remediation for f in confirmed] == [first.remediation, second.remediation]


@pytest.mark.parametrize(
    ("explanation", "remediation"),
    [
        ("Reviewed advisory details", ""),
        ("", "Review the fixed release"),
        ("Reviewed advisory details", "Review the fixed release"),
    ],
)
def test_explicit_llm_guidance_still_enriches_sc4(explanation, remediation):
    original = Finding(
        rule_id="SC4",
        message="Original message",
        file="requirements.txt",
        explanation="Original evidence",
        remediation="Original guidance",
    )
    batch = Batch(file_path=original.file, content="", findings=[original])
    item = {
        "pattern_id": "SC4",
        "is_vulnerability": True,
        "confidence": 0.8,
        "explanation": explanation,
        "remediation": remediation,
    }
    [confirmed] = LLMMetaAnalyzer.__new__(LLMMetaAnalyzer).apply_filter(
        [original], [(batch, [item])]
    )
    assert confirmed.explanation == (explanation or original.explanation)
    assert confirmed.remediation == (remediation or original.remediation)


def test_other_rules_keep_their_existing_empty_llm_text_defaults():
    original = Finding(
        rule_id="SC1",
        message="Original message",
        file="requirements.txt",
        explanation="Original evidence",
        remediation="Original guidance",
    )
    batch = Batch(file_path=original.file, content="", findings=[original])
    item = {"pattern_id": "SC1", "is_vulnerability": True, "confidence": 0.8}
    [confirmed] = LLMMetaAnalyzer.__new__(LLMMetaAnalyzer).apply_filter(
        [original], [(batch, [item])]
    )
    assert confirmed.explanation == get_explanation("SC1")
    assert confirmed.remediation == get_remediation("SC1")


@pytest.mark.parametrize(
    "limitation",
    [
        OsvQueryLimitation(reason=LedgerReason.RUNTIME_LIMIT),
        OsvQueryLimitation(reason=LedgerReason.ANALYZER_RUNTIME_ERROR, error_class="ValueError"),
        OsvQueryLimitation(reason=LedgerReason.OUTPUT_LIMIT, observed_records=1, limit_records=0),
    ],
    ids=["timeout", "malformed-response", "query-limit"],
)
def test_failed_lookup_advice_does_not_assume_network_failure(monkeypatch, limitation):
    monkeypatch.setattr(
        sc,
        "query_batch",
        lambda packages, *_args, **_kwargs: QueryBatchResults(
            [[] for _ in packages], limitations=(limitation,)
        ),
    )
    monkeypatch.setattr(sc, "was_osv_reachable", lambda: False)
    raw, limitations, _ = sc._analyze_dependencies_detailed(
        "examplepkg==1.0.0\n", "requirements.txt"
    )
    [finding] = _sc4(raw)
    assert limitations == [limitation]
    assert "retry the scan" in finding.remediation.lower()
    assert "if the lookup timed out or the network was unavailable" in finding.remediation.lower()
    assert "verify dependency versions" in finding.remediation.lower()
    assert "incomplete" in finding.explanation.lower()
