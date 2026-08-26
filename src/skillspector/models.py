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

"""Shared models for the Skillspector v2 LangGraph workflow."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import InitVar, dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

if TYPE_CHECKING:
    from skillspector.state import SkillspectorState


class Severity(StrEnum):
    """Severity levels for findings (used by all analyzers)."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


def compute_match_fingerprint(rule_id: str, matched_text: str) -> str:
    """Return the canonical SHA-256 identity for one rule-bound match."""
    normalized = " ".join(matched_text.strip().split())
    return sha256(f"{rule_id}\x1f{normalized}".encode()).hexdigest()


@dataclass
class Location:
    """Location of a finding within a file (used by all analyzers)."""

    file: str
    start_line: int
    end_line: int | None = None


_analyzer_finding_observer: ContextVar[Callable[[AnalyzerFinding], None] | None] = ContextVar(
    "skillspector_analyzer_finding_observer",
    default=None,
)


@dataclass
class AnalyzerFinding:
    """
    Common finding type produced by any analyzer (static, behavioral, MCP, semantic).
    Converted to Finding for graph state; use severity, location, tags for consistency.
    """

    rule_id: str
    message: str
    severity: Severity
    location: Location
    confidence: float = 0.5
    remediation: str | None = None
    tags: list[str] = field(default_factory=list)
    context: str | None = None
    matched_text: str | None = None
    evidence: dict[str, object] = field(default_factory=dict)
    # Canonical rule+match digest; source binding is derived by ``Finding.fingerprint``.
    match_fingerprint: str | None = None
    complete_match: InitVar[str | None] = None

    def __post_init__(self, complete_match: str | None) -> None:
        """Notify an optional runner-owned resource guard after construction.

        Static analyzers are trusted code, but the number of findings they
        construct is controlled by untrusted input.  A context-local observer
        lets the shared runner stop an analyzer while it is still building its
        private result list instead of waiting for that list to become large.
        Other analyzer families pay no cost beyond this single context lookup.
        """
        if complete_match is not None:
            self.match_fingerprint = compute_match_fingerprint(self.rule_id, complete_match)
        observer = _analyzer_finding_observer.get()
        if observer is not None:
            observer(self)


@contextmanager
def observe_analyzer_findings(
    observer: Callable[[AnalyzerFinding], None],
) -> Iterator[None]:
    """Install a task-local observer for newly constructed analyzer findings."""
    token = _analyzer_finding_observer.set(observer)
    try:
        yield
    finally:
        _analyzer_finding_observer.reset(token)


def _new_finding_id() -> str:
    """Return an opaque, run-unique identity for one logical finding."""
    return f"finding-{uuid4().hex}"


@dataclass
class Finding:
    """Finding model for graph state and report output (shape aligned with to_dict)."""

    rule_id: str
    message: str
    finding_id: str = field(default_factory=_new_finding_id)
    severity: str = "LOW"
    confidence: float = 0.5
    file: str = "SKILL.md"
    start_line: int = 1
    end_line: int | None = None
    category: str | None = None
    pattern: str | None = None
    finding: str | None = None  # short matched snippet
    explanation: str | None = None
    remediation: str | None = None
    code_snippet: str | None = None
    intent: str | None = None
    tags: list[str] = field(default_factory=list)
    context: str | None = None
    matched_text: str | None = None
    transitive_depth: int = 0
    source_url: str | None = None
    # ``source_url`` is display metadata and can be mutable (for example a
    # branch URL). These values are the report-safe, immutable provenance
    # attached by transitive traversal: an opaque source scope and the digest
    # of the exact tree/content that was inspected.
    source_identity: str | None = None
    source_digest: str | None = None
    evidence: dict[str, object] = field(default_factory=dict)
    # Canonical unbound rule+match digest. Never replace it with a source-bound digest.
    match_fingerprint: str | None = None
    occurrences: list[dict[str, object]] = field(default_factory=list)

    def fingerprint(self) -> str | None:
        """Return a full-match fingerprint without exposing the matched payload."""
        has_source_provenance = bool(
            self.source_identity or self.source_digest or self.source_url or self.transitive_depth
        )
        if self.match_fingerprint and not has_source_provenance:
            return self.match_fingerprint
        if not self.match_fingerprint and not self.matched_text:
            return None
        provenance = {
            "source_identity": self.source_identity or "",
            "source_digest": self.source_digest or "",
            # URL is only a compatibility discriminator when immutable source
            # provenance is unavailable; it is display-only otherwise.
            "source_url": (
                self.source_url if not self.source_identity and not self.source_digest else ""
            )
            or "",
            "transitive_depth": self.transitive_depth,
        }
        provenance_json = json.dumps(
            provenance, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        provenance_hash = sha256(provenance_json.encode()).hexdigest()
        source_prefix = f"source-sha256:{provenance_hash}:"
        # A source-bound fingerprint is tagged with its provenance hash so
        # compaction remains idempotent while a changed source is re-bound.
        if self.match_fingerprint and self.match_fingerprint.startswith(source_prefix):
            return self.match_fingerprint
        normalized = (
            self.match_fingerprint
            if self.match_fingerprint
            else " ".join((self.matched_text or "").strip().split())
        )
        if not has_source_provenance:
            return (
                self.match_fingerprint
                if self.match_fingerprint
                else compute_match_fingerprint(self.rule_id, normalized)
            )
        payload = {
            "rule_id": self.rule_id,
            "match": normalized,
            "source": provenance,
        }
        canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return f"{source_prefix}{sha256(canonical.encode()).hexdigest()}"

    def _serialized_occurrences(self) -> list[dict[str, object]]:
        """Return locations with the finding's immutable provenance attached."""
        occurrences = list(self.occurrences) or [
            {
                "file": self.file,
                "start_line": self.start_line,
                "end_line": self.end_line,
            }
        ]
        serialized: list[dict[str, object]] = []
        for raw in occurrences:
            occurrence = dict(raw)
            if self.source_identity:
                occurrence.setdefault("source_identity", self.source_identity)
            if self.source_digest:
                occurrence.setdefault("source_digest", self.source_digest)
            if self.source_url:
                occurrence.setdefault("source_url", self.source_url)
            if self.transitive_depth:
                occurrence.setdefault("transitive_depth", self.transitive_depth)
            serialized.append(occurrence)
        return serialized

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dict representation (full finding shape)."""
        data: dict[str, object] = {
            "id": self.rule_id,
            "finding_id": self.finding_id,
            "category": self.category,
            "pattern": self.pattern,
            "severity": self.severity,
            "confidence": self.confidence,
            "location": {
                "file": self.file,
                "start_line": self.start_line,
                "end_line": self.end_line,
            },
            "finding": self.finding,
            "explanation": self.explanation or self.message,
            "remediation": self.remediation,
            "code_snippet": self.code_snippet or self.context,
            "intent": self.intent,
            # Tags surface markers like "llm-unconfirmed" (a high-severity static
            # finding the LLM filter did not confirm but which is preserved anyway).
            "tags": list(self.tags),
            "evidence": dict(self.evidence),
            "match_fingerprint": self.fingerprint(),
            "occurrences": self._serialized_occurrences(),
        }
        if self.transitive_depth:
            data["transitive_depth"] = self.transitive_depth
        if self.source_url:
            data["source_url"] = self.source_url
        if self.source_identity:
            data["source_identity"] = self.source_identity
        if self.source_digest:
            data["source_digest"] = self.source_digest
        return data

    def __str__(self) -> str:
        return f"{self.rule_id}: {self.message} ({self.file}:{self.start_line})"


class AnalyzerPlugin(Protocol):
    """Analyzer plugin protocol: name/stage/availability and an ``analyze`` entry point."""

    name: str
    stage: str
    requires_api_key: bool

    def analyze(self, state: SkillspectorState) -> list[Finding]:
        """Analyze graph state and return findings."""

    def is_available(self) -> bool:
        """Return whether the analyzer can run in current environment."""
