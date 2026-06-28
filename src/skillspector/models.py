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

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from skillspector.state import SkillspectorState


class Severity(StrEnum):
    """Severity levels for findings (used by all analyzers)."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Location:
    """Location of a finding within a file (used by all analyzers)."""

    file: str
    start_line: int
    end_line: int | None = None


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


@dataclass
class Finding:
    """Finding model for graph state and report output (shape aligned with to_dict)."""

    rule_id: str
    message: str
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

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dict representation (full finding shape)."""
        return {
            "id": self.rule_id,
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
        }

    def __str__(self) -> str:
        return f"{self.rule_id}: {self.message} ({self.file}:{self.start_line})"


class AnalyzerPlugin(Protocol):
    """Analyzer protocol from SADD A.1.1."""

    name: str
    stage: str
    requires_api_key: bool

    def analyze(self, state: SkillspectorState) -> list[Finding]:
        """Analyze graph state and return findings."""

    def is_available(self) -> bool:
        """Return whether the analyzer can run in current environment."""
