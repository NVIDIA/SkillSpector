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

"""Tests for cross-analyzer finding deduplication."""

from __future__ import annotations

from copy import deepcopy

import pytest

from skillspector.models import Finding
from skillspector.nodes.deduplicate import deduplicate


def _finding(
    rule_id: str = "TM1",
    file: str = "tool.py",
    matched_text: str = "subprocess.run(cmd, shell=True)",
    confidence: float = 0.8,
    severity: str = "HIGH",
    start_line: int = 1,
) -> Finding:
    return Finding(
        rule_id=rule_id,
        message=f"Test finding {rule_id}",
        severity=severity,
        confidence=confidence,
        file=file,
        start_line=start_line,
        matched_text=matched_text,
    )


class TestSameFileDedup:
    """Same rule_id + same file + same matched_text → keep highest confidence."""

    def test_exact_duplicates_reduced_to_one(self) -> None:
        """Two identical findings in same file → one output."""
        findings = [
            _finding(file="a.py", start_line=1),
            _finding(file="a.py", start_line=5),
        ]
        result = deduplicate(findings)
        assert len(result) == 1

    def test_keeps_highest_confidence(self) -> None:
        """When duplicates exist, the highest confidence one is kept."""
        findings = [
            _finding(file="a.py", confidence=0.6),
            _finding(file="a.py", confidence=0.9),
            _finding(file="a.py", confidence=0.3),
        ]
        result = deduplicate(findings)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_different_rules_same_file_not_deduped(self) -> None:
        """Different rule_ids in same file are independent findings."""
        findings = [
            _finding(rule_id="TM1", file="a.py"),
            _finding(rule_id="TM2", file="a.py"),
        ]
        result = deduplicate(findings)
        assert len(result) == 2

    def test_different_matched_text_same_file_not_deduped(self) -> None:
        """Same rule but different matched text in same file → separate findings."""
        findings = [
            _finding(file="a.py", matched_text="subprocess.run(cmd, shell=True)"),
            _finding(file="a.py", matched_text="subprocess.Popen(cmd, shell=True)"),
        ]
        result = deduplicate(findings)
        assert len(result) == 2

    @pytest.mark.parametrize(
        ("field_name", "different_value"),
        [
            ("message", "Different message"),
            ("severity", "MEDIUM"),
            ("category", "Different category"),
            ("pattern", "Different pattern"),
            ("explanation", "Different explanation"),
            ("remediation", "Different remediation"),
            ("intent", "different intent"),
            ("tags", ["contextual-triage", "likely-benign-context"]),
            ("evidence", {"classification": "different"}),
        ],
    )
    def test_different_report_metadata_is_not_deduplicated(
        self,
        field_name: str,
        different_value: object,
    ) -> None:
        first = _finding(file="a.py")
        second = deepcopy(first)
        second.file = "b.py"
        setattr(second, field_name, different_value)

        result = deduplicate([first, second])

        assert len(result) == 2

    @pytest.mark.parametrize("field_name", ["finding", "code_snippet", "context"])
    def test_location_context_does_not_change_dedup_identity(self, field_name: str) -> None:
        first = _finding(file="a.py")
        second = deepcopy(first)
        second.file = "b.py"
        setattr(first, field_name, "context from a.py")
        setattr(second, field_name, "context from b.py")

        result = deduplicate([first, second])

        assert len(result) == 1
        assert {item["file"] for item in result[0].occurrences} == {"a.py", "b.py"}

    def test_evidence_mapping_order_does_not_change_dedup_identity(self) -> None:
        first = _finding(file="a.py")
        first.evidence = {"outer": {"a": 1, "b": [2, 3]}}
        second = _finding(file="b.py")
        second.evidence = {"outer": {"b": [2, 3], "a": 1}}

        result = deduplicate([first, second])

        assert len(result) == 1

    def test_tag_order_does_not_change_dedup_identity(self) -> None:
        first = _finding(file="a.py")
        first.tags = ["primary", "secondary"]
        second = _finding(file="b.py")
        second.tags = ["secondary", "primary"]

        result = deduplicate([first, second])

        assert len(result) == 1
        assert {item["file"] for item in result[0].occurrences} == {"a.py", "b.py"}

    def test_non_json_evidence_fails_closed_without_raising(self) -> None:
        first = _finding(file="a.py")
        first.evidence = {"raw": b"same"}
        second = _finding(file="b.py")
        second.evidence = {"raw": b"same"}

        result = deduplicate([first, second])

        assert len(result) == 2

    def test_cyclic_evidence_fails_closed_without_raising(self) -> None:
        first = _finding(file="a.py")
        first.evidence["cycle"] = first.evidence
        second = _finding(file="b.py")
        second.evidence["cycle"] = second.evidence

        result = deduplicate([first, second])

        assert len(result) == 2

    def test_same_line_benign_and_unsafe_matches_keep_local_classification(self) -> None:
        safe = _finding(rule_id="PE3", file="build.sh", matched_text="/etc/passwd")
        safe.tags = ["Privilege Escalation", "contextual-triage", "likely-benign-context"]
        safe.code_snippet = "docker run -v /etc/passwd:/etc/passwd:ro image"
        unsafe = _finding(rule_id="PE3", file="build.sh", matched_text="/etc/passwd")
        unsafe.tags = ["Privilege Escalation"]
        unsafe.code_snippet = "cat /etc/passwd"

        for findings in ([safe, unsafe], [unsafe, safe]):
            result = deduplicate(findings)
            assert len(result) == 2
            assert {(tuple(item.tags), item.code_snippet) for item in result} == {
                (tuple(safe.tags), safe.code_snippet),
                (tuple(unsafe.tags), unsafe.code_snippet),
            }


class TestCrossFileDedup:
    """Same rule_id + same matched_text across files → keep best."""

    def test_same_pattern_across_files_deduplicated(self) -> None:
        """Same rule + same matched text in different files → one output."""
        findings = [
            _finding(file="step1.py"),
            _finding(file="step2.py"),
            _finding(file="step3.py"),
            _finding(file="step4.py"),
        ]
        result = deduplicate(findings)
        assert len(result) == 1

    def test_cross_file_keeps_highest_confidence(self) -> None:
        """Cross-file dedup keeps the highest confidence finding."""
        findings = [
            _finding(file="a.py", confidence=0.5),
            _finding(file="b.py", confidence=0.9),
            _finding(file="c.py", confidence=0.7),
        ]
        result = deduplicate(findings)
        assert len(result) == 1
        assert result[0].confidence == 0.9
        assert result[0].file == "b.py"

    def test_same_pattern_from_different_transitive_sources_is_preserved(self) -> None:
        first = _finding(file="tool.py")
        first.source_url = "https://github.com/org/first"
        second = _finding(file="tool.py")
        second.source_url = "https://github.com/org/second"

        result = deduplicate([first, second])

        assert len(result) == 2

    def test_same_display_url_with_different_source_identities_is_preserved(self) -> None:
        first = _finding(file="tool.py")
        first.source_url = "https://github.com/org/repository"
        first.source_identity = "external/first"
        first.source_digest = "sha256:" + "a" * 64
        second = _finding(file="tool.py")
        second.source_url = first.source_url
        second.source_identity = "external/second"
        second.source_digest = "sha256:" + "b" * 64

        result = deduplicate([first, second])

        assert len(result) == 2

    def test_same_immutable_source_deduplicates_across_display_urls(self) -> None:
        first = _finding(file="tool.py", start_line=1)
        first.source_url = "https://github.com/org/repository/tree/main"
        first.source_identity = "external/source"
        first.source_digest = "sha256:" + "a" * 64
        second = _finding(file="tool.py", start_line=2)
        second.source_url = "https://github.com/org/repository/tree/release"
        second.source_identity = first.source_identity
        second.source_digest = first.source_digest

        result = deduplicate([first, second])

        assert len(result) == 1
        assert {item["source_identity"] for item in result[0].occurrences} == {"external/source"}
        assert {item["source_digest"] for item in result[0].occurrences} == {"sha256:" + "a" * 64}
        assert {item["source_url"] for item in result[0].occurrences} == {
            first.source_url,
            second.source_url,
        }

    def test_occurrence_only_source_identities_are_not_cross_deduplicated(self) -> None:
        first = _finding(file="tool.py")
        first.occurrences = [
            {"file": "tool.py", "start_line": 1, "source_identity": "external/first"}
        ]
        second = _finding(file="tool.py")
        second.occurrences = [
            {"file": "tool.py", "start_line": 1, "source_identity": "external/second"}
        ]

        assert len(deduplicate([first, second])) == 2

    def test_different_patterns_across_files_not_deduped(self) -> None:
        """Different matched texts are independent even with same rule_id."""
        findings = [
            _finding(file="a.py", matched_text="curl -k"),
            _finding(file="b.py", matched_text="wget --no-check-certificate"),
        ]
        result = deduplicate(findings)
        assert len(result) == 2

    def test_different_rules_same_pattern_not_deduped(self) -> None:
        """Different rules with same matched text are independent."""
        findings = [
            _finding(rule_id="TM1", file="a.py", matched_text="curl -k"),
            _finding(rule_id="SC1", file="b.py", matched_text="curl -k"),
        ]
        result = deduplicate(findings)
        assert len(result) == 2


class TestNoMatchedText:
    """Findings without matched_text are never cross-file deduplicated."""

    def test_no_matched_text_kept_independently(self) -> None:
        """Findings with empty/None matched_text are all kept."""
        findings = [
            _finding(file="a.py", matched_text=""),
            _finding(file="b.py", matched_text=""),
        ]
        result = deduplicate(findings)
        assert len(result) == 2

    def test_none_matched_text_kept(self) -> None:
        """Findings with None matched_text are preserved."""
        f1 = Finding(rule_id="TM1", message="Test", file="a.py", start_line=1, matched_text=None)
        f2 = Finding(rule_id="TM1", message="Test", file="b.py", start_line=1, matched_text=None)
        result = deduplicate([f1, f2])
        assert len(result) == 2


class TestEdgeCases:
    """Edge cases and ordering."""

    def test_empty_list(self) -> None:
        """Empty input returns empty output."""
        assert deduplicate([]) == []

    def test_single_finding_unchanged(self) -> None:
        """A single finding passes through unchanged."""
        findings = [_finding()]
        result = deduplicate(findings)
        assert len(result) == 1
        assert result[0].rule_id == "TM1"

    def test_output_sorted_by_severity_then_file(self) -> None:
        """Output is sorted: CRITICAL > HIGH > MEDIUM > LOW, then by file."""
        findings = [
            _finding(rule_id="A", severity="LOW", file="z.py", matched_text="low"),
            _finding(rule_id="B", severity="CRITICAL", file="a.py", matched_text="crit"),
            _finding(rule_id="C", severity="HIGH", file="m.py", matched_text="high"),
            _finding(rule_id="D", severity="MEDIUM", file="b.py", matched_text="med"),
        ]
        result = deduplicate(findings)
        assert len(result) == 4
        assert [r.severity for r in result] == ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

    def test_real_world_repetitive_skill(self) -> None:
        """Simulates a skill with subprocess in 5 files — should deduplicate to 1."""
        findings = [
            _finding(
                rule_id="TM1",
                file=f"step{i}.py",
                matched_text="subprocess.run(cmd, shell=True)",
                confidence=0.8,
            )
            for i in range(5)
        ]
        result = deduplicate(findings)
        assert len(result) == 1

    def test_mixed_dedup_scenario(self) -> None:
        """Mix of same-file, cross-file, and unique findings."""
        findings = [
            # Same pattern in 3 files → should become 1
            _finding(rule_id="TM1", file="a.py", matched_text="shell=True"),
            _finding(rule_id="TM1", file="b.py", matched_text="shell=True"),
            _finding(rule_id="TM1", file="c.py", matched_text="shell=True"),
            # Different pattern, unique
            _finding(rule_id="E1", file="a.py", matched_text="requests.post(url)"),
            # Same rule different pattern
            _finding(rule_id="TM1", file="d.py", matched_text="--force delete"),
        ]
        result = deduplicate(findings)
        # TM1 shell=True (1) + E1 requests.post (1) + TM1 --force (1) = 3
        assert len(result) == 3

    def test_whitespace_normalization(self) -> None:
        """Leading/trailing whitespace in matched_text is trimmed for key."""
        findings = [
            _finding(file="a.py", matched_text="  curl -k  "),
            _finding(file="b.py", matched_text="curl -k"),
        ]
        result = deduplicate(findings)
        assert len(result) == 1

    def test_long_matched_text_uses_complete_fingerprint(self) -> None:
        """Matches sharing a long prefix remain distinct when their suffix differs."""
        base = "x" * 100
        findings = [
            _finding(file="a.py", matched_text=base + "AAAA"),
            _finding(file="b.py", matched_text=base + "BBBB"),
        ]
        result = deduplicate(findings)
        assert len(result) == 2
