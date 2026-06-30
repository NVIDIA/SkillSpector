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

"""Baseline / false-positive suppression for SkillSpector.

A *baseline* is a YAML (or JSON) file that tells the report node which findings
to drop before scoring and reporting. It supports two complementary mechanisms:

* ``rules`` — human-authored, glob-based suppressions. A finding is suppressed
  when every field a rule specifies (``id``, ``path``, ``message``) glob-matches
  the finding. Unspecified fields match anything. This covers both global
  pattern suppression (e.g. ``id: "SQP-1"``) and skill/file-scoped suppression
  (e.g. ``id: "SSD-2"`` + ``path: "deploy-topology-execute-scripts/SKILL.md"``).

* ``fingerprints`` — machine-generated exact suppressions. Each entry is the
  stable hash of one known finding, so re-scans only surface *new* findings.
  Generate these with ``skillspector baseline <path>`` for incremental CI use.

Example baseline::

    version: 1
    rules:
      - id: "SQP-1"
        reason: "Trigger-phrase breadth is a description nit, not a vuln"
      - id: "SSD-2"
        path: "*deploy-topology*/SKILL.md"
        message: "*run the exploit*"
        reason: "False positive: 'run the exploit' is a lab test-workflow phrase"
    fingerprints:
      - hash: "sha256:1a2b3c4d5e6f7081"
        rule_id: "SDI-2"
        file: "baas-build-analysis/SKILL.md"
        reason: "Accepted 2026-06-19 — first-party env detection"

Glob semantics use :func:`fnmatch.fnmatch`, where ``*`` matches across path
separators (so ``*SKILL.md`` matches ``a/b/SKILL.md``); ``**`` is accepted as a
friendly alias for ``*``. Message globs are matched case-insensitively, so wrap
a keyword in ``*`` (e.g. ``"*telemetry*"``) for substring matching.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from skillspector.logging_config import get_logger
from skillspector.models import Finding

logger = get_logger(__name__)

BASELINE_VERSION = 1


def _match_glob(value: str, pattern: str) -> bool:
    """Case-insensitive glob match; ``**`` is treated as an alias for ``*``.

    Patterns use :func:`fnmatch.fnmatch` semantics, so ``*``, ``?`` and ``[...]``
    are treated as glob metacharacters. Rule ids and the messages we match are
    plain text in practice, but if you ever need to match one of those characters
    literally, escape it with :func:`fnmatch.translate` / ``[`` brackets rather
    than relying on literal matching here.
    """
    normalized = pattern.replace("**", "*")
    return fnmatch.fnmatch(value.lower(), normalized.lower())


def finding_fingerprint(finding: Finding) -> str:
    """Return a stable short fingerprint for *finding*.

    Derived from rule id, file, line span, and message so the same finding hashes
    identically across runs. Note that edits which shift line numbers or reword an
    LLM message will change the fingerprint — regenerate the baseline when a skill
    changes materially. Use ``rules`` for drift-tolerant suppression.
    """
    raw = "|".join(
        [
            finding.rule_id or "",
            finding.file or "",
            str(finding.start_line or ""),
            str(finding.end_line or ""),
            (finding.message or "").strip(),
        ]
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"sha256:{digest}"


@dataclass(frozen=True)
class SuppressionRule:
    """A glob-based suppression rule. Empty rules (no field set) never match."""

    rule_id: str | None = None
    path: str | None = None
    message: str | None = None
    reason: str = ""

    def matches(self, finding: Finding) -> bool:
        """True when every field this rule specifies glob-matches *finding*."""
        if self.rule_id is None and self.path is None and self.message is None:
            return False  # guard: an all-wildcard rule would suppress everything
        if self.rule_id is not None and not _match_glob(finding.rule_id or "", self.rule_id):
            return False
        if self.path is not None and not _match_glob(finding.file or "", self.path):
            return False
        if self.message is not None and not _match_glob(finding.message or "", self.message):
            return False
        return True


@dataclass(frozen=True)
class SuppressedFinding:
    """A finding paired with the reason it was suppressed."""

    finding: Finding
    reason: str

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form: the full finding plus its suppression reason."""
        data = self.finding.to_dict()
        data["suppressed"] = True
        data["suppression_reason"] = self.reason
        return data


@dataclass
class Baseline:
    """Loaded baseline: glob rules plus exact fingerprint suppressions."""

    rules: list[SuppressionRule] = field(default_factory=list)
    fingerprints: dict[str, str] = field(default_factory=dict)  # hash -> reason

    def reason_for(self, finding: Finding) -> str | None:
        """Return the suppression reason for *finding*, or None if not suppressed."""
        for rule in self.rules:
            if rule.matches(finding):
                return rule.reason or "matched suppression rule"
        fp = finding_fingerprint(finding)
        if fp in self.fingerprints:
            return self.fingerprints[fp] or "matched baseline fingerprint"
        return None

    def is_empty(self) -> bool:
        """True when the baseline has no rules and no fingerprints."""
        return not self.rules and not self.fingerprints


def baseline_from_dict(data: dict[str, Any]) -> Baseline:
    """Build a :class:`Baseline` from a parsed mapping (YAML/JSON)."""
    if not isinstance(data, dict):
        raise ValueError(f"baseline must be a mapping (got {type(data).__name__})")

    version = data.get("version", BASELINE_VERSION)
    if version != BASELINE_VERSION:
        logger.warning(
            "Baseline version %s does not match supported version %s; attempting to load anyway",
            version,
            BASELINE_VERSION,
        )

    rules: list[SuppressionRule] = []
    for raw in data.get("rules") or []:
        if not isinstance(raw, dict):
            raise ValueError(f"each baseline rule must be a mapping, got: {raw!r}")
        rule = SuppressionRule(
            rule_id=raw.get("id") or raw.get("rule_id"),
            path=raw.get("path") or raw.get("file"),
            message=raw.get("message"),
            reason=raw.get("reason", ""),
        )
        if rule.rule_id is None and rule.path is None and rule.message is None:
            raise ValueError(
                "a baseline rule must set at least one of: id, path, message "
                f"(offending rule: {raw!r})"
            )
        rules.append(rule)

    fingerprints: dict[str, str] = {}
    for raw in data.get("fingerprints") or []:
        if isinstance(raw, str):
            fingerprints[raw] = ""
        elif isinstance(raw, dict) and raw.get("hash"):
            fingerprints[str(raw["hash"])] = raw.get("reason", "")
        else:
            raise ValueError(
                f"each fingerprint must be a string or have a 'hash' key, got: {raw!r}"
            )

    return Baseline(rules=rules, fingerprints=fingerprints)


def load_baseline(path: str | Path) -> Baseline:
    """Load a baseline file (YAML or JSON) into a :class:`Baseline`.

    Raises FileNotFoundError if *path* is missing, ValueError if it is malformed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Baseline file not found: {p}")
    text = p.read_text(encoding="utf-8")
    try:
        # yaml.safe_load parses JSON too, so a single path handles both formats.
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:  # pragma: no cover - error path
        raise ValueError(f"Could not parse baseline file {p}: {e}") from e
    return baseline_from_dict(data)


def partition_findings(
    findings: list[Finding], baseline: Baseline | None
) -> tuple[list[Finding], list[SuppressedFinding]]:
    """Split *findings* into (kept, suppressed) using *baseline*.

    With no baseline, everything is kept. Suppressed findings never count toward
    the risk score and are excluded from the SARIF results.
    """
    if baseline is None or baseline.is_empty():
        return list(findings), []
    kept: list[Finding] = []
    suppressed: list[SuppressedFinding] = []
    for finding in findings:
        reason = baseline.reason_for(finding)
        if reason is None:
            kept.append(finding)
        else:
            suppressed.append(SuppressedFinding(finding=finding, reason=reason))
    if suppressed:
        logger.debug("Suppressed %d finding(s) via baseline", len(suppressed))
    return kept, suppressed


def build_baseline_dict(
    findings: list[Finding],
    reason: str = "Accepted finding (auto-generated baseline)",
) -> dict[str, object]:
    """Build a baseline mapping that fingerprint-suppresses every given finding."""
    return {
        "version": BASELINE_VERSION,
        "rules": [],
        "fingerprints": [
            {
                "hash": finding_fingerprint(f),
                "rule_id": f.rule_id,
                "file": f.file,
                "reason": reason,
            }
            for f in findings
        ],
    }


def dump_baseline(data: dict[str, object], path: str | Path) -> None:
    """Write a baseline mapping to *path* as YAML (``.json`` extension -> JSON)."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        header = (
            "# SkillSpector baseline — findings listed here are suppressed on future scans.\n"
            "# Edit 'reason' fields and add glob 'rules' as needed. See docs/SUPPRESSION.md.\n"
        )
        p.write_text(header + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
