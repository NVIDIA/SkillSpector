# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for finding identity."""

from skillspector.models import Finding
from skillspector.state import merge_findings_by_id


def test_finding_has_unique_instance_id_without_changing_rule_id() -> None:
    """Each logical finding has an opaque identity while rule IDs stay compatible."""
    first = Finding(rule_id="P1", message="first")
    second = Finding(rule_id="P1", message="second")

    assert first.finding_id.startswith("finding-")
    assert second.finding_id.startswith("finding-")
    assert first.finding_id != second.finding_id
    assert first.to_dict()["id"] == "P1"
    assert first.to_dict()["finding_id"] == first.finding_id


def test_finding_reducer_replaces_same_id_without_duplicating_payload() -> None:
    """An enriched finding replaces its canonical instance in reducer order."""
    original = Finding(rule_id="P1", message="raw", finding_id="finding-a")
    enriched = Finding(
        rule_id="P1",
        message="confirmed",
        finding_id="finding-a",
        explanation="confirmed by meta-analysis",
    )

    merged = merge_findings_by_id([original], [enriched])

    assert len(merged) == 1
    assert merged[0].finding_id == "finding-a"
    assert merged[0].message == "confirmed"


def test_finding_serializes_structured_evidence() -> None:
    finding = Finding(
        rule_id="SC9",
        message="concealed executable",
        evidence={"outer_path": "archive.docx", "nested_path": "payload.sh"},
    )

    assert finding.to_dict()["evidence"] == {
        "outer_path": "archive.docx",
        "nested_path": "payload.sh",
    }
