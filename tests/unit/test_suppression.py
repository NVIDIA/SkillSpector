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

"""Unit tests for baseline / false-positive suppression."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from skillspector.models import Finding
from skillspector.suppression import (
    Baseline,
    SuppressionRule,
    baseline_from_dict,
    build_baseline_dict,
    dump_baseline,
    finding_fingerprint,
    load_baseline,
    partition_findings,
)


def _finding(
    rule_id: str = "SQP-1",
    file: str = "skill-a/SKILL.md",
    message: str = "Overly broad trigger phrases",
    severity: str = "MEDIUM",
    start_line: int = 3,
) -> Finding:
    return Finding(
        rule_id=rule_id,
        message=message,
        severity=severity,
        confidence=0.7,
        file=file,
        start_line=start_line,
    )


# --- fingerprint --------------------------------------------------------------


def test_fingerprint_is_stable_and_prefixed() -> None:
    f = _finding()
    assert finding_fingerprint(f) == finding_fingerprint(_finding())
    assert finding_fingerprint(f).startswith("sha256:")


def test_fingerprint_differs_on_field_change() -> None:
    base = finding_fingerprint(_finding())
    assert finding_fingerprint(_finding(rule_id="SQP-2")) != base
    assert finding_fingerprint(_finding(file="skill-b/SKILL.md")) != base
    assert finding_fingerprint(_finding(start_line=99)) != base


# --- rule matching ------------------------------------------------------------


def test_rule_matches_exact_rule_id() -> None:
    rule = SuppressionRule(rule_id="SQP-1", reason="nit")
    assert rule.matches(_finding(rule_id="SQP-1"))
    assert not rule.matches(_finding(rule_id="SQP-2"))


def test_rule_matches_glob_rule_id() -> None:
    rule = SuppressionRule(rule_id="SQP-*", reason="all quality-policy nits")
    assert rule.matches(_finding(rule_id="SQP-1"))
    assert rule.matches(_finding(rule_id="SQP-12"))
    assert not rule.matches(_finding(rule_id="SDI-2"))


def test_rule_scoped_by_path_and_rule_id() -> None:
    rule = SuppressionRule(rule_id="SSD-2", path="*deploy-topology*/SKILL.md", reason="lab phrase")
    assert rule.matches(_finding(rule_id="SSD-2", file="deploy-topology-execute-scripts/SKILL.md"))
    # Right rule, wrong file -> not suppressed
    assert not rule.matches(_finding(rule_id="SSD-2", file="other/SKILL.md"))
    # Right file, wrong rule -> not suppressed
    assert not rule.matches(
        _finding(rule_id="SQP-1", file="deploy-topology-execute-scripts/SKILL.md")
    )


def test_rule_message_glob_is_case_insensitive_substring() -> None:
    rule = SuppressionRule(message="*telemetry*", reason="first-party telemetry")
    assert rule.matches(_finding(message="Mandates completion TELEMETRY call"))
    assert not rule.matches(_finding(message="Reads environment variables"))


def test_double_star_is_alias_for_star() -> None:
    rule = SuppressionRule(path="**/SKILL.md", reason="any skill file")
    assert rule.matches(_finding(file="a/b/c/SKILL.md"))


def test_empty_rule_never_matches() -> None:
    assert not SuppressionRule().matches(_finding())


# --- Baseline.reason_for ------------------------------------------------------


def test_baseline_reason_for_rule_then_fingerprint() -> None:
    f = _finding()
    by_rule = Baseline(rules=[SuppressionRule(rule_id="SQP-1", reason="rule wins")])
    assert by_rule.reason_for(f) == "rule wins"

    by_fp = Baseline(fingerprints={finding_fingerprint(f): "fp reason"})
    assert by_fp.reason_for(f) == "fp reason"

    assert Baseline().reason_for(f) is None


def test_baseline_default_reason_when_blank() -> None:
    f = _finding()
    assert Baseline(rules=[SuppressionRule(rule_id="SQP-1")]).reason_for(f) == (
        "matched suppression rule"
    )
    assert Baseline(fingerprints={finding_fingerprint(f): ""}).reason_for(f) == (
        "matched baseline fingerprint"
    )


# --- partition_findings -------------------------------------------------------


def test_partition_no_baseline_keeps_all() -> None:
    findings = [_finding(), _finding(rule_id="SDI-2")]
    kept, suppressed = partition_findings(findings, None)
    assert kept == findings
    assert suppressed == []


def test_partition_empty_baseline_keeps_all() -> None:
    findings = [_finding()]
    kept, suppressed = partition_findings(findings, Baseline())
    assert len(kept) == 1
    assert suppressed == []


def test_partition_splits_and_records_reason() -> None:
    keep = _finding(rule_id="SDI-2", message="real issue")
    drop = _finding(rule_id="SQP-1")
    baseline = Baseline(rules=[SuppressionRule(rule_id="SQP-1", reason="fp")])
    kept, suppressed = partition_findings([keep, drop], baseline)
    assert kept == [keep]
    assert len(suppressed) == 1
    assert suppressed[0].finding is drop
    assert suppressed[0].reason == "fp"


def test_suppressed_finding_to_dict() -> None:
    baseline = Baseline(rules=[SuppressionRule(rule_id="SQP-1", reason="fp")])
    _, suppressed = partition_findings([_finding()], baseline)
    d = suppressed[0].to_dict()
    assert d["suppressed"] is True
    assert d["suppression_reason"] == "fp"
    assert d["id"] == "SQP-1"


# --- baseline_from_dict parsing ----------------------------------------------


def test_baseline_from_dict_full() -> None:
    data = {
        "version": 1,
        "rules": [
            {"id": "SQP-*", "reason": "nits"},
            {"rule_id": "SSD-2", "file": "*/SKILL.md", "message": "*exploit*", "reason": "fp"},
        ],
        "fingerprints": [
            "sha256:deadbeefdeadbeef",
            {"hash": "sha256:cafebabecafebabe", "reason": "accepted"},
        ],
    }
    baseline = baseline_from_dict(data)
    assert len(baseline.rules) == 2
    assert baseline.rules[1].path == "*/SKILL.md"
    assert baseline.fingerprints["sha256:deadbeefdeadbeef"] == ""
    assert baseline.fingerprints["sha256:cafebabecafebabe"] == "accepted"


def test_baseline_from_dict_rejects_all_wildcard_rule() -> None:
    with pytest.raises(ValueError, match="at least one of"):
        baseline_from_dict({"rules": [{"reason": "oops, suppresses everything"}]})


def test_baseline_from_dict_rejects_non_mapping() -> None:
    with pytest.raises(ValueError):
        baseline_from_dict(["not", "a", "mapping"])  # type: ignore[arg-type]


# --- load / dump round-trip ---------------------------------------------------


def test_load_baseline_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_baseline(tmp_path / "nope.yaml")


def test_build_dump_load_round_trip(tmp_path: Path) -> None:
    findings = [_finding(), _finding(rule_id="SDI-2", file="x/SKILL.md")]
    data = build_baseline_dict(findings, reason="accepted in CI")
    out = tmp_path / "baseline.yaml"
    dump_baseline(data, out)
    assert out.exists()

    baseline = load_baseline(out)
    # Every original finding is now suppressed by fingerprint.
    kept, suppressed = partition_findings(findings, baseline)
    assert kept == []
    assert len(suppressed) == 2
    assert all(sf.reason == "accepted in CI" for sf in suppressed)


def test_dump_baseline_json_extension(tmp_path: Path) -> None:
    data = build_baseline_dict([_finding()])
    out = tmp_path / "baseline.json"
    dump_baseline(data, out)
    # Valid JSON and loadable back through the YAML-or-JSON loader.
    import json

    parsed = json.loads(out.read_text())
    assert parsed["version"] == 1
    assert load_baseline(out).fingerprints


def test_load_baseline_parses_yaml_content(tmp_path: Path) -> None:
    out = tmp_path / "b.yaml"
    out.write_text(
        yaml.safe_dump({"version": 1, "rules": [{"id": "SQP-1", "reason": "r"}]}),
        encoding="utf-8",
    )
    baseline = load_baseline(out)
    assert baseline.rules[0].rule_id == "SQP-1"
