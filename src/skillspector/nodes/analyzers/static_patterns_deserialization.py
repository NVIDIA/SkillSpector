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

"""Static patterns: insecure deserialization across languages (DS1–DS4).

Python deserialization is detected with AST/taint precision by ``behavioral_ast``
(AST10) and ``behavioral_taint_tracking`` (TT6). This module provides *breadth* for
the non-Python scripts a skill may bundle (PHP, Ruby, and JavaScript/TypeScript) via
language-gated regex signatures. Matching is anchored to each language's dangerous
deserializer so a signature only runs against files of that language, keeping false
positives low. Node and analyze() live in one module.
"""

from __future__ import annotations

import re
import sys

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import get_context, get_line_number
from .pattern_defaults import PatternCategory

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_deserialization"

# File extension -> language. Python is intentionally excluded: it is covered with
# AST/taint precision by behavioral_ast (AST10) and behavioral_taint_tracking (TT6),
# so scanning it here too would only produce duplicate, lower-quality findings.
_LANG_BY_EXT: dict[str, str] = {
    ".php": "php",
    ".php3": "php",
    ".php4": "php",
    ".php5": "php",
    ".phtml": "php",
    ".rb": "ruby",
    ".rake": "ruby",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "javascript",
    ".tsx": "javascript",
}

# language -> [(rule_id, message, severity, [(regex, confidence), ...]), ...]
_LANG_RULES: dict[str, list[tuple[str, str, Severity, list[tuple[str, float]]]]] = {
    "php": [
        (
            "DS1",
            "PHP object injection via unserialize()",
            Severity.HIGH,
            [(r"\bunserialize\s*\(", 0.8)],
        ),
    ],
    "ruby": [
        (
            "DS2",
            "Ruby Marshal deserialization of untrusted data",
            Severity.HIGH,
            [(r"\bMarshal\s*\.\s*(?:load|restore)\b", 0.85)],
        ),
        (
            "DS3",
            "Unsafe Ruby YAML/Oj deserialization",
            Severity.MEDIUM,
            [
                (r"\b(?:YAML|Psych)\s*\.\s*load\s*\(", 0.65),
                (r"\bOj\s*\.\s*load\s*\(", 0.6),
            ],
        ),
    ],
    "javascript": [
        (
            "DS4",
            "Unsafe JavaScript deserialization (node-serialize/funcster)",
            Severity.HIGH,
            [
                (r"""require\(\s*['"]node-serialize['"]\s*\)""", 0.75),
                (r"""require\(\s*['"]serialize-to-js['"]\s*\)""", 0.7),
                (r"\bfuncster\b", 0.6),
                (r"\.unserialize\s*\(", 0.6),
            ],
        ),
    ],
}

# Pre-compiled: language -> [(rule_id, message, severity, compiled_regex, confidence), ...]
_COMPILED: dict[str, list[tuple[str, str, Severity, re.Pattern[str], float]]] = {
    language: [
        (rule_id, message, severity, re.compile(pattern, re.IGNORECASE), confidence)
        for rule_id, message, severity, patterns in rules
        for pattern, confidence in patterns
    ]
    for language, rules in _LANG_RULES.items()
}


def _language_for(file_path: str) -> str | None:
    """Map a file path to a supported language by extension, or None."""
    idx = file_path.rfind(".")
    if idx < 0:
        return None
    return _LANG_BY_EXT.get(file_path[idx:].lower())


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Detect insecure deserialization signatures in non-Python skill scripts (DS1–DS4)."""
    language = _language_for(file_path)
    if language is None:
        return []
    tag = [PatternCategory.DESERIALIZATION.value]
    findings: list[AnalyzerFinding] = []
    for rule_id, message, severity, regex, confidence in _COMPILED[language]:
        for match in regex.finditer(content):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id=rule_id,
                    message=message,
                    severity=severity,
                    location=Location(file=file_path, start_line=line_num),
                    confidence=confidence,
                    tags=tag,
                    context=get_context(content, match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run multi-language deserialization patterns and return findings."""
    findings = static_runner.run_static_patterns(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(findings))
    return {"findings": findings}
