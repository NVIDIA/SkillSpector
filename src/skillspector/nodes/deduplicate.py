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

"""Cross-analyzer finding deduplication.

Merges findings that represent the same conceptual issue observed multiple
times — either within the same file or across files with identical patterns.

Deduplication strategy:
1. Same-file dedup: Same rule_id + same file + same matched_text
   → keep highest confidence instance
2. Cross-file consolidation: Same rule_id + same matched_text across files
   → keep highest confidence instance
"""

from __future__ import annotations

from skillspector.logging_config import get_logger
from skillspector.models import Finding

logger = get_logger(__name__)


def _same_file_key(finding: Finding) -> tuple[str, str, str]:
    """Build a deduplication key for same-file matches."""
    matched = (finding.matched_text or "").strip()[:100]
    return (finding.rule_id, finding.file, matched)


def _cross_file_key(finding: Finding) -> tuple[str, str]:
    """Build a cross-file deduplication key from rule_id and normalized matched_text."""
    matched = (finding.matched_text or "").strip()[:100]
    return (finding.rule_id, matched)


def deduplicate(findings: list[Finding]) -> list[Finding]:
    """Deduplicate a list of findings, returning a reduced list.

    Two-pass deduplication:
    1. Same-file: identical (rule_id, file, matched_text) → keep highest confidence
    2. Cross-file: identical (rule_id, matched_text) across different files
       → keep highest confidence representative

    Findings without matched_text are never cross-file deduplicated (they lack
    a reliable identity signal).
    """
    if not findings:
        return []

    original_count = len(findings)

    # Pass 1: Same-file deduplication
    same_file_best: dict[tuple[str, str, str], Finding] = {}
    for f in findings:
        key = _same_file_key(f)
        existing = same_file_best.get(key)
        if existing is None or f.confidence > existing.confidence:
            same_file_best[key] = f

    after_same_file = list(same_file_best.values())

    # Pass 2: Cross-file deduplication (only for findings WITH matched_text)
    cross_file_best: dict[tuple[str, str], Finding] = {}
    no_text_findings: list[Finding] = []

    for f in after_same_file:
        matched = (f.matched_text or "").strip()
        if not matched:
            no_text_findings.append(f)
            continue
        key = _cross_file_key(f)
        existing = cross_file_best.get(key)
        if existing is None or f.confidence > existing.confidence:
            cross_file_best[key] = f

    deduplicated = list(cross_file_best.values()) + no_text_findings

    removed = original_count - len(deduplicated)
    if removed > 0:
        logger.info(
            "Deduplication: %d → %d findings (%d duplicates removed)",
            original_count,
            len(deduplicated),
            removed,
        )

    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    deduplicated.sort(
        key=lambda f: (severity_order.get(f.severity.upper(), 4), f.file, f.start_line)
    )

    return deduplicated
