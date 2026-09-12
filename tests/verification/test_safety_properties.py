# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differential and adversarial properties for the scanner's safety envelope."""

from __future__ import annotations

import random
from dataclasses import replace

from hypothesis import given, settings
from hypothesis import strategies as st

from skillspector.llm_analyzer_base import Batch
from skillspector.models import Finding
from skillspector.nodes.meta_analyzer import LLMMetaAnalyzer
from skillspector.nodes.report import _compute_risk_score
from tests.verification.risk_reference import ReferenceFinding, reference_risk_score

SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN")


@st.composite
def reference_findings(draw: st.DrawFn) -> ReferenceFinding:
    rule_id = draw(st.sampled_from(("R1", "R2", "SC8", "BH2", "BH3", "")))
    severity = draw(st.sampled_from(SEVERITIES))
    confidence = draw(
        st.floats(min_value=-1.0, max_value=2.0, allow_nan=False, allow_infinity=False)
    )
    file = draw(st.sampled_from(("SKILL.md", "run.py", "nested/run.py")))
    identity = draw(st.one_of(st.none(), st.sampled_from(("scope-a", "scope-b"))))
    evidence: dict[str, object] = {}
    if rule_id in {"BH2", "BH3"}:
        evidence["activation_state"] = draw(st.sampled_from(("conditional", "inactive")))
    if rule_id == "BH2":
        evidence["proof_status"] = draw(st.sampled_from(("closed", "open")))
    return ReferenceFinding(
        rule_id=rule_id,
        severity=severity,
        confidence=confidence,
        file=file,
        source_identity=identity,
        evidence=evidence,
    )


def _production_finding(item: ReferenceFinding) -> Finding:
    return Finding(
        rule_id=item.rule_id,
        message="verification finding",
        severity=item.severity,
        confidence=item.confidence,
        file=item.file,
        source_identity=item.source_identity,
        source_url=item.source_url,
        source_digest=item.source_digest,
        evidence=dict(item.evidence),
    )


@given(st.lists(reference_findings(), max_size=12), st.booleans())
@settings(max_examples=500, deadline=None)
def test_risk_core_matches_independent_reference_model(
    findings: list[ReferenceFinding], has_executable_scripts: bool
) -> None:
    metadata: list[dict[str, object]] = [
        {
            "path": item.file,
            "source_identity": item.source_identity,
            "executable": item.file.endswith(".py"),
        }
        for item in findings
    ]
    expected = reference_risk_score(findings, has_executable_scripts, metadata)
    actual = _compute_risk_score(
        [_production_finding(item) for item in findings], has_executable_scripts, metadata
    )
    assert actual == expected
    assert 0 <= actual[0] <= 100


@given(st.lists(reference_findings(), max_size=12), st.booleans(), st.randoms())
@settings(max_examples=250, deadline=None)
def test_risk_score_is_permutation_invariant(
    findings: list[ReferenceFinding], has_executable_scripts: bool, rng: random.Random
) -> None:
    metadata: list[dict[str, object]] = [
        {
            "path": item.file,
            "source_identity": item.source_identity,
            "executable": item.file.endswith(".py"),
        }
        for item in findings
    ]
    shuffled = list(findings)
    rng.shuffle(shuffled)
    assert reference_risk_score(findings, has_executable_scripts, metadata) == reference_risk_score(
        shuffled, has_executable_scripts, metadata
    )
    assert _compute_risk_score(
        [_production_finding(item) for item in findings], has_executable_scripts, metadata
    ) == _compute_risk_score(
        [_production_finding(item) for item in shuffled], has_executable_scripts, metadata
    )


@given(st.lists(reference_findings(), max_size=12), reference_findings(), st.booleans())
@settings(max_examples=500, deadline=None)
def test_adding_arbitrary_evidence_cannot_lower_score(
    findings: list[ReferenceFinding],
    added: ReferenceFinding,
    has_executable_scripts: bool,
) -> None:
    combined = [*findings, added]
    metadata: list[dict[str, object]] = [
        {
            "path": item.file,
            "source_identity": item.source_identity,
            "executable": item.file.endswith(".py"),
        }
        for item in combined
    ]

    reference_before = reference_risk_score(findings, has_executable_scripts, metadata)[0]
    reference_after = reference_risk_score(combined, has_executable_scripts, metadata)[0]
    production_before = _compute_risk_score(
        [_production_finding(item) for item in findings], has_executable_scripts, metadata
    )[0]
    production_after = _compute_risk_score(
        [_production_finding(item) for item in combined], has_executable_scripts, metadata
    )[0]

    assert reference_after >= reference_before
    assert production_after >= production_before


@st.composite
def llm_outcomes(draw: st.DrawFn) -> list[dict[str, object]]:
    return draw(
        st.lists(
            st.fixed_dictionaries(
                {
                    "pattern_id": st.sampled_from(("R1", "R2", "INVENTED")),
                    "is_vulnerability": st.booleans(),
                    "confidence": st.floats(
                        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
                    ),
                    "start_line": st.integers(min_value=1, max_value=5),
                    "_file": st.sampled_from(("run.py", "other.py")),
                    "explanation": st.text(max_size=30),
                    "remediation": st.text(max_size=30),
                }
            ),
            max_size=12,
        )
    )


@given(llm_outcomes())
@settings(max_examples=500, deadline=None)
def test_arbitrary_llm_outcomes_cannot_weaken_deterministic_findings(
    outcomes: list[dict[str, object]],
) -> None:
    originals = [
        Finding(
            rule_id="R1",
            message="first",
            severity="HIGH",
            confidence=0.82,
            file="run.py",
            start_line=1,
            evidence={"origin": "deterministic"},
        ),
        Finding(
            rule_id="R2",
            message="second",
            severity="LOW",
            confidence=0.41,
            file="run.py",
            start_line=2,
            evidence={"origin": "deterministic"},
        ),
    ]
    batch = Batch(file_path="run.py", content="", findings=originals)
    analyzer = LLMMetaAnalyzer.__new__(LLMMetaAnalyzer)
    result = analyzer.apply_filter(originals, [(batch, outcomes)])
    by_id = {finding.finding_id: finding for finding in result}

    assert set(by_id) == {finding.finding_id for finding in originals}
    for original in originals:
        returned = by_id[original.finding_id]
        assert returned.rule_id == original.rule_id
        assert returned.severity == original.severity
        assert returned.confidence >= original.confidence
        assert returned.evidence == original.evidence


def test_reference_model_detects_the_historical_equal_severity_order_bug() -> None:
    weak = ReferenceFinding(rule_id="R1", severity="CRITICAL", confidence=0.01)
    strong = replace(weak, confidence=1.0)
    assert reference_risk_score([weak, strong], False) == reference_risk_score(
        [strong, weak], False
    )
    assert _compute_risk_score([_production_finding(weak), _production_finding(strong)], False) == (
        50,
        "MEDIUM",
        "CAUTION",
    )
    assert _compute_risk_score([_production_finding(strong), _production_finding(weak)], False) == (
        50,
        "MEDIUM",
        "CAUTION",
    )


def test_higher_severity_evidence_cannot_steal_weight_from_stronger_evidence() -> None:
    strong_high = ReferenceFinding(rule_id="R1", severity="HIGH", confidence=0.1)
    weak_critical = ReferenceFinding(rule_id="R1", severity="CRITICAL", confidence=0.01)

    before = reference_risk_score([strong_high], False)
    after = reference_risk_score([strong_high, weak_critical], False)

    assert before == (2, "LOW", "SAFE")
    assert after == (2, "LOW", "SAFE")
    assert _compute_risk_score([_production_finding(strong_high)], False) == before
    assert (
        _compute_risk_score(
            [_production_finding(strong_high), _production_finding(weak_critical)], False
        )
        == after
    )


def test_every_integer_score_has_exactly_one_documented_band() -> None:
    bands = ((81, "CRITICAL"), (51, "HIGH"), (21, "MEDIUM"), (0, "LOW"))
    for score in range(101):
        actual = next(label for threshold, label in bands if score >= threshold)
        expected = (
            "LOW"
            if score <= 20
            else "MEDIUM"
            if score <= 50
            else "HIGH"
            if score <= 80
            else "CRITICAL"
        )
        assert actual == expected
