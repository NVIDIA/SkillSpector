"""event_model.py - Normalized SecurityEvent for all analyzers."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class SecurityEvent:
    source: str  # runtime | static | secrets | privacy | permission | dependency | diff | provenance | sbom
    category: str  # filesystem | network | processes | env | executables | package | mcp | dns | outbound | secrets | privacy
    action: str  # read | write | connect | execute | install | call | resolve | exfiltrate
    subject: str  # skill.py / skill name
    target: str  # file path, domain, package, env var
    capability: str  # capability identifier e.g. filesystem.write, network.outbound
    evidence: str  # code snippet / trace
    severity: str = "medium"
    confidence: float = 0.9
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    rule_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        raw = f"{self.source}|{self.category}|{self.action}|{self.subject}|{self.target}|{self.capability}|{self.evidence[:200]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_finding(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id or f"EVT-{self.category.upper()}",
            "category": self.category,
            "severity": self.severity,
            "message": f"[{self.source}] {self.category}/{self.action} {self.subject} -> {self.target} ({self.capability})",
            "file": self.subject,
            "line": None,
            "evidence": self.evidence[:500],
            "fix": self.metadata.get("fix"),
            "confidence": self.confidence,
            "source": self.source,
            "capability": self.capability,
            "target": self.target,
        }


@dataclass
class EventGraph:
    events: list[SecurityEvent] = field(default_factory=list)

    def add(self, e: SecurityEvent):
        self.events.append(e)

    def by_category(self, cat: str) -> list[SecurityEvent]:
        return [e for e in self.events if e.category == cat]

    def by_source(self, src: str) -> list[SecurityEvent]:
        return [e for e in self.events if e.source == src]

    def to_findings(self) -> list[dict[str, Any]]:
        return [e.to_finding() for e in self.events]

    def to_dict(self) -> dict[str, Any]:
        return {"events": [e.to_dict() for e in self.events], "count": len(self.events)}
