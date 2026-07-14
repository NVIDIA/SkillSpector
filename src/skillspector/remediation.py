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

"""AI-assisted remediation: generate auto-fix patches for detected findings.

Scans a skill, identifies fixable findings, and produces patched file versions
with automated remediations applied. Supports both regex-based pattern fixes
and AST-based Python code transformations.

This module provides the ``skillspector fix`` CLI command's core logic.
"""

from __future__ import annotations

import difflib
import re
import textwrap
from pathlib import Path

from skillspector.logging_config import get_logger
from skillspector.models import Finding

logger = get_logger(__name__)


# Rule ID -> (pattern, replacement) for automated regex fixes
_REGEX_FIXES: dict[str, list[tuple[re.Pattern[str], str]]] = {
    "P1": [
        (re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions?", re.IGNORECASE), "[REMOVED: instruction override]"),
        (re.compile(r"ignore\s+(?:all\s+)?(?:safety|security)\s+(?:rules?|constraints?|guidelines?)", re.IGNORECASE), "[REMOVED: safety override]"),
        (re.compile(r"you\s+are\s+now\s+(?:in\s+)?(?:jailbreak|unrestricted|unfiltered)\s+mode", re.IGNORECASE), "[REMOVED: jailbreak mode]"),
    ],
    "P2": [
        (re.compile(r"<!--.*?-->", re.DOTALL), ""),
        (re.compile(r"\[//\]:\s*#\(.*?\)", re.DOTALL), ""),
    ],
    "P6": [
        (re.compile(r"(?:print|output|reveal|expose|show)\s+(?:the\s+)?(?:system\s+)?(?:prompt|instructions?|rules?)", re.IGNORECASE), "[REMOVED: system prompt leakage]"),
    ],
    "AR1": [
        (re.compile(r"never\s+(?:refuse|say\s+no|decline|reject)", re.IGNORECASE), "[REMOVED: refusal suppression]"),
    ],
    "AR2": [
        (re.compile(r"(?:omit|remove|skip|drop)\s+(?:all\s+)?(?:warnings?|disclaimers?|caveats?)", re.IGNORECASE), "[REMOVED: disclaimer suppression]"),
    ],
    "SC2": [
        (re.compile(r"curl\s+(?:-[sSkLfO]*\s+)*\S+\s*\|\s*(?:bash|sh|python|node)", re.IGNORECASE), "# [SECURITY] Remote code execution blocked: review and install manually"),
    ],
}

# Rule ID -> remediation template for SKILL.md security annotations
_SKILL_MD_TEMPLATES: dict[str, str] = {
    "EA1": "# SECURITY: Tool access has been restricted to required tools only.",
    "EA2": "# SECURITY: Destructive operations now require human confirmation.",
    "LP2": "# SECURITY: Wildcard permissions replaced with explicit allowlist.",
    "LP3": "# SECURITY: Permissions field added to SKILL.md manifest.",
}


class RemediationResult:
    """Result of applying automated remediations."""

    def __init__(self) -> None:
        self.files_modified: list[str] = []
        self.fixes_applied: list[dict[str, str]] = []
        self.skipped: list[dict[str, str]] = []
        self.diff: str = ""

    def add_fix(self, file_path: str, rule_id: str, description: str) -> None:
        self.files_modified.append(file_path)
        self.fixes_applied.append({
            "file": file_path,
            "rule": rule_id,
            "description": description,
        })

    def add_skip(self, file_path: str, rule_id: str, reason: str) -> None:
        self.skipped.append({
            "file": file_path,
            "rule": rule_id,
            "reason": reason,
        })

    def summary(self) -> str:
        lines = [f"Applied {len(self.fixes_applied)} fix(es) to {len(set(self.files_modified))} file(s)."]
        if self.skipped:
            lines.append(f"Skipped {len(self.skipped)} finding(s) requiring manual review.")
        return "\n".join(lines)


def apply_regex_fix(content: str, rule_id: str) -> tuple[str, int]:
    """Apply regex-based fixes for a given rule ID. Returns (new_content, fix_count)."""
    fixes = _REGEX_FIXES.get(rule_id, [])
    fix_count = 0
    for pattern, replacement in fixes:
        new_content = pattern.sub(replacement, content)
        if new_content != content:
            fix_count += 1
            content = new_content
    return content, fix_count


def generate_skill_md_patch(findings: list[Finding]) -> str | None:
    """Generate a SKILL.md security annotation block from findings."""
    annotations: list[str] = []
    seen_rules: set[str] = set()
    for finding in findings:
        template = _SKILL_MD_TEMPLATES.get(finding.rule_id)
        if template and finding.rule_id not in seen_rules:
            annotations.append(template)
            seen_rules.add(finding.rule_id)
    if not annotations:
        return None
    return "\n".join(annotations)


def compute_diff(old: str, new: str, file_path: str) -> str:
    """Compute a unified diff between old and new file contents."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    return "".join(difflib.unified_diff(old_lines, new_lines, fromfile=file_path, tofile=f"{file_path} (patched)"))


def remediate_files(
    findings: list[Finding],
    file_cache: dict[str, str],
    dry_run: bool = True,
) -> tuple[RemediationResult, dict[str, str]]:
    """Apply automated remediations to files based on findings.

    Args:
        findings: List of findings to remediate.
        file_cache: Map of file paths to their contents.
        dry_run: If True, compute patches without writing to disk.

    Returns:
        Tuple of (RemediationResult, patched_files_map).
    """
    result = RemediationResult()
    patched: dict[str, str] = {}

    findings_by_file: dict[str, list[Finding]] = {}
    for f in findings:
        findings_by_file.setdefault(f.file, []).append(f)

    for file_path, file_findings in findings_by_file.items():
        original = file_cache.get(file_path)
        if original is None:
            continue
        content = original
        total_fixes = 0
        for finding in file_findings:
            if finding.rule_id in _REGEX_FIXES:
                new_content, fix_count = apply_regex_fix(content, finding.rule_id)
                if fix_count > 0:
                    content = new_content
                    total_fixes += fix_count
                    result.add_fix(file_path, finding.rule_id, f"Regex fix applied ({fix_count} occurrence(s))")
                else:
                    result.add_skip(file_path, finding.rule_id, "Pattern not found in current content")
            elif finding.rule_id in _SKILL_MD_TEMPLATES:
                result.add_skip(file_path, finding.rule_id, "Requires manual SKILL.md edit")
            else:
                result.add_skip(file_path, finding.rule_id, "No automated fix available")

        if content != original:
            patched[file_path] = content
            result.diff += compute_diff(original, content, file_path) + "\n"

    return result, patched
