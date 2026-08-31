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

"""Shared runner for static pattern nodes: file-type inference, conversion, run_static_patterns."""

from __future__ import annotations

import re
import time
import unicodedata
from array import array
from bisect import bisect_right
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import cast

from skillspector.artifacts import (
    ContentKind,
    SecurityTextView,
    _contains_default_ignorable,
    is_default_ignorable,
    security_text_views,
)
from skillspector.inspection_ledger import (
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    LedgerRecordType,
    analyzer_status_for_events,
    ledger_event,
)
from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Finding, observe_analyzer_findings
from skillspector.nodes.deduplicate import classification_metadata_key
from skillspector.python_ast import (
    MAX_PYTHON_AST_SOURCE_CHARS,
    ParsedPythonFile,
    get_python_ast,
)
from skillspector.security_reconstruction import (
    MAX_DECLARED_MARKER_RIGHT_CONTEXT_CHARS,
    MAX_MARKER_LOOKAHEAD_CHARS,
    build_declared_marker_views,
)
from skillspector.state import AnalyzerNodeResponse, SkillspectorState, transitive_remaining_seconds

from .common import (
    LINE_BREAK_CHARS,
    LOGICAL_LINE_BREAK,
    MARKDOWN_FENCE_CLOSE,
    MARKDOWN_FENCE_OPEN,
)
from .pattern_defaults import get_category, get_explanation, get_pattern_name, get_remediation

logger = get_logger(__name__)

# Extension -> file type (match v1 InventoryBuilder.FILE_TYPES)
FILE_TYPES: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".py": "python",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".txt": "text",
    ".js": "javascript",
    ".ts": "typescript",
    ".rb": "ruby",
    ".go": "go",
    ".rs": "rust",
}

MAX_FILE_CHARS = MAX_PYTHON_AST_SOURCE_CHARS
SECURITY_VIEW_WINDOW_CHARS = 256_000
_WINDOW_OVERLAP_CHARS = 8192
_RAW_WINDOW_OWNED_CHARS = SECURITY_VIEW_WINDOW_CHARS - 2 * _WINDOW_OVERLAP_CHARS
_VIEW_START_EVIDENCE = "_security_view_start"
_SOURCE_START_EVIDENCE = "_security_source_start"
_VIEW_ORIGIN_TAGS = frozenset({"normalized-view", "declared-marker-view"})
_BENIGN_CONTEXT_TAGS = frozenset({"contextual-triage", "likely-benign-context"})
_ViewFindingKey = tuple[str, str, int, str | None, tuple[object, ...]]
_ViewScopeKey = tuple[str, str, int, str | None]
assert _RAW_WINDOW_OWNED_CHARS > 0
DECLARED_MARKER_LEFT_CONTEXT_CHARS = MAX_MARKER_LOOKAHEAD_CHARS
DECLARED_MARKER_RIGHT_CONTEXT_CHARS = MAX_DECLARED_MARKER_RIGHT_CONTEXT_CHARS
DECLARED_MARKER_OWNED_CHARS = (
    SECURITY_VIEW_WINDOW_CHARS
    - DECLARED_MARKER_LEFT_CONTEXT_CHARS
    - DECLARED_MARKER_RIGHT_CONTEXT_CHARS
)
assert DECLARED_MARKER_OWNED_CHARS > 0
# The continuity projection keeps enough of an attacker-controlled separator
# that bounded-gap expressions cannot be turned into matches.  Only expressions
# which already accept an unbounded separator (for example ``\s+``) can bridge
# it.  Each auxiliary view is therefore still substantially smaller than the
# ordinary module-input ceiling.
_CONTINUITY_SEPARATOR_CHARS = _WINDOW_OVERLAP_CHARS
_CONTINUITY_CONTEXT_CHARS = 2048
_CONTINUITY_MAX_CHAIN_RUNS = 24
MAX_FINDINGS_PER_ARTIFACT = 10_000
MAX_FINDINGS_PER_ANALYZER = 10_000
MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT = 30.0

_LICENSE_FILE_TYPES = frozenset({"markdown", "text", "other"})
_LICENSE_BASENAME = re.compile(r"^(?:license|licenses|copying|notice|notices)(?:[._-].*)?$")
_LICENSE_OTHER_SUFFIXES = frozenset({".lesser"})
_ASCII_CONTINUITY_SEPARATOR_RUN = re.compile(r"[\s\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")
_ASCII_NON_NEWLINE_WHITESPACE = re.compile(r"[ \t\r\f\v]")


def _advance_markdown_fence(active: tuple[str, int] | None, line: str) -> tuple[str, int] | None:
    stripped = line.rstrip(LINE_BREAK_CHARS)
    closing = MARKDOWN_FENCE_CLOSE.fullmatch(stripped)
    if active is not None:
        if closing and closing.group(1)[0] == active[0] and len(closing.group(1)) >= active[1]:
            return None
        return active
    opening = MARKDOWN_FENCE_OPEN.fullmatch(stripped)
    if opening:
        marker = opening.group(1)
        return marker[0], len(marker)
    return None


def _markdown_fence_states(
    content: str, offsets: tuple[int, ...]
) -> tuple[dict[int, tuple[str, int] | None], dict[int, tuple[str, int, str, int, int]]]:
    states: dict[int, tuple[str, int] | None] = {}
    transitions: dict[int, tuple[str, int, str, int, int]] = {}
    active: tuple[str, int] | None = None
    offset_index = 0
    content_offset = 0
    for line in content.splitlines(keepends=True):
        line_end = content_offset + len(line)
        complete = line.endswith(tuple(LINE_BREAK_CHARS))
        stripped = line.rstrip(LINE_BREAK_CHARS)
        opening = MARKDOWN_FENCE_OPEN.fullmatch(stripped) if complete else None
        closing = MARKDOWN_FENCE_CLOSE.fullmatch(stripped) if complete else None
        while offset_index < len(offsets) and offsets[offset_index] < line_end:
            offset = offsets[offset_index]
            states[offset] = active
            if offset > content_offset:
                if active is None and opening is not None:
                    marker = opening.group(1)
                    transitions[offset] = (marker[0], len(marker), "open", line_end, content_offset)
                elif (
                    active is not None
                    and closing is not None
                    and closing.group(1)[0] == active[0]
                    and len(closing.group(1)) >= active[1]
                ):
                    marker = closing.group(1)
                    transitions[offset] = (
                        marker[0],
                        len(marker),
                        "close",
                        line_end,
                        content_offset,
                    )
            offset_index += 1
        if not complete:
            break
        active = _advance_markdown_fence(active, line)
        content_offset = line_end
        while offset_index < len(offsets) and offsets[offset_index] == line_end:
            states[offsets[offset_index]] = active
            offset_index += 1
    while offset_index < len(offsets):
        states[offsets[offset_index]] = active
        offset_index += 1
    return states, transitions


def _window_view_with_markdown_context(
    view: SecurityTextView, prefix_length: int
) -> SecurityTextView:
    if prefix_length == 0:
        return view
    if view.source_offsets is None:
        offsets = array("I", (max(0, offset - prefix_length) for offset in range(len(view.text))))
    else:
        offsets = array("I", (max(0, offset - prefix_length) for offset in view.source_offsets))
    return SecurityTextView(view.name, view.text, offsets)


def _markdown_context_prefix(
    content: str,
    window_start: int,
    window_end: int,
    fence_states: dict[int, tuple[str, int] | None],
    fence_transitions: dict[int, tuple[str, int, str, int, int]],
) -> str:
    """Return synthetic fence context for one window that starts mid-document."""
    fence = fence_states.get(window_start)
    transition = fence_transitions.get(window_start)
    if transition is not None and transition[2] == "close" and transition[3] <= window_end:
        closing_prefix = content[transition[4] : window_start]
        return transition[0] * transition[1] + "\n" + closing_prefix
    if fence is not None:
        return fence[0] * fence[1] + "\n"
    if transition is not None and transition[3] <= window_end:
        return transition[0] * transition[1] + "\n"
    return ""


def _normalize_license_line(line: str) -> str:
    return " ".join(line.casefold().split())


# Each range contains the complete adjacent text and the only suppressible line offset.
_LICENSE_CANONICAL_RANGES: tuple[tuple[tuple[str, ...], int], ...] = (
    (
        (
            '"source" form shall mean the preferred form for making modifications,',
            "including but not limited to software source code, documentation",
            "source, and configuration files.",
        ),
        1,
    ),
    (
        (
            "transformation or translation of a source form, including but",
            "not limited to compiled object code, generated documentation,",
            "and conversions to other media types.",
        ),
        1,
    ),
    (
        (
            'the copyright owner. For the purposes of this definition, "submitted"',
            "means any form of electronic, verbal, or written communication sent",
            "to the Licensor or its representatives, including but not limited to",
            "communication on electronic mailing lists, source code control systems,",
        ),
        2,
    ),
    (
        (
            "result of this License or out of the use or inability to use the",
            "Work (including but not limited to damages for loss of goodwill,",
            "work stoppage, computer failure or malfunction, or any and all",
        ),
        1,
    ),
    (
        (
            'the software is provided "as is", without warranty of any kind, express or',
            "implied, including but not limited to the warranties of merchantability,",
            "fitness for a particular purpose and NONINFRINGEMENT. in no event shall the",
        ),
        1,
    ),
    (
        (
            'this software is provided by the copyright holders and contributors "as is"',
            "and any express or implied warranties, including, but not limited to, the",
            "implied warranties of merchantability and fitness for a particular purpose are",
        ),
        1,
    ),
)


def _infer_file_type(path: str) -> str:
    """Infer file type from path (extension)."""
    idx = path.rfind(".")
    suffix = path[idx:].lower() if idx >= 0 else ""
    return FILE_TYPES.get(suffix, "other")


def _is_license_basename(path: str, file_type: str) -> bool:
    """Return whether a text-like path has a conventional legal-file basename."""
    if file_type not in _LICENSE_FILE_TYPES:
        return False
    basename = path.replace("\\", "/").rsplit("/", 1)[-1]
    if file_type == "other" and "." in basename:
        suffix = "." + basename.rsplit(".", 1)[-1].casefold()
        if suffix not in _LICENSE_OTHER_SUFFIXES:
            return False
    return _LICENSE_BASENAME.fullmatch(basename.casefold()) is not None


def _is_license_boilerplate_line(content: str, start_line: int) -> bool:
    """Return whether start_line occupies a registered canonical license range."""
    return _is_license_boilerplate_in_normalized_lines(
        tuple(_normalize_license_line(line) for line in content.splitlines()),
        start_line,
    )


def _is_license_boilerplate_in_normalized_lines(
    normalized_lines: tuple[str, ...], start_line: int
) -> bool:
    """Check one line against pre-normalized license text."""
    if start_line < 1 or start_line > len(normalized_lines):
        return False
    for canonical_lines, match_offset in _LICENSE_CANONICAL_RANGES:
        range_start = start_line - match_offset - 1
        range_end = range_start + len(canonical_lines)
        normalized_canonical_lines = tuple(
            _normalize_license_line(line) for line in canonical_lines
        )
        if (
            range_start >= 0
            and normalized_lines[range_start:range_end] == normalized_canonical_lines
        ):
            return True
    return False


_NULL_BYTE_SAMPLE_SIZE = 512


def _is_binary_file(path: str, content: str) -> bool:
    """Compatibility helper: extensions alone never classify an artifact as binary."""
    del path
    return "\x00" in content[:_NULL_BYTE_SAMPLE_SIZE]


_PE3_ENV_TEMPLATE_SETUP = re.compile(
    r"(?:[-*]\s*)?(?:cp|copy|mv|rename)\s+\.env\.(?:example|sample|template)\s+"
    r"(?:to\s+)?\.env(?:\s+(?:before\s+(?:running|starting)(?:\s+the\s+app)?|"
    r"for\s+local\s+development))?[.:]?",
    re.IGNORECASE,
)
_PE3_ENV_FILE_SETUP = re.compile(
    r"(?:create|configure|set\s+up|make|add)\s+(?:an?\s+|the\s+)?\.env(?:\s+file)?"
    r"(?:\s+in\s+the\s+project\s+root)?(?:\s+with\s+(?:your\s+)?api\s+keys?|"
    r"\s+for\s+(?:local\s+)?(?:development|testing))?[.:]?",
    re.IGNORECASE,
)
_PE3_DOTENV_SETUP = re.compile(
    r"(?:install|use)\s+(?:python-)?dotenv\s+to\s+load\s+(?:the\s+)?\.env\s+file[.:]?",
    re.IGNORECASE,
)


def _is_env_file_reference_in_docs(
    finding: AnalyzerFinding,
    file_type: str,
    file_path: str = "",
    content: str | None = None,
    content_lines: list[str] | None = None,
) -> bool:
    """Return True if a PE3 finding is a documentation reference to .env files, not actual access.

    SKILL.md is exempt: it is the agent's primary instruction file, so `.env`
    references there may be genuine credential-access instructions.
    """
    if finding.rule_id != "PE3":
        return False
    if file_type not in ("markdown", "text"):
        return False
    if file_path.replace("\\", "/").lower().endswith("skill.md"):
        return False
    if not finding.context:
        return False

    if content is not None:
        lines = content.splitlines() if content_lines is None else content_lines
        index = finding.location.start_line - 1
        if index < 0 or index >= len(lines):
            return False
        line = lines[index]
    else:
        candidate_lines = [line for line in finding.context.splitlines() if ".env" in line.lower()]
        if len(candidate_lines) != 1:
            return False
        line = candidate_lines[0]

    normalized_line = line.replace("`", "").strip()
    return any(
        pattern.fullmatch(normalized_line) is not None
        for pattern in (_PE3_ENV_TEMPLATE_SETUP, _PE3_ENV_FILE_SETUP, _PE3_DOTENV_SETUP)
    )


def analyzer_finding_to_finding(
    af: AnalyzerFinding,
    get_remediation_fn: Callable[[str], str] | None = None,
) -> Finding:
    """Convert an AnalyzerFinding (from any analyzer) to graph-state Finding."""
    rem_fn = get_remediation_fn or get_remediation
    remediation = af.remediation or rem_fn(af.rule_id)
    category = (af.tags[0] if af.tags else None) or get_category(af.rule_id)
    pattern = af.message or get_pattern_name(af.rule_id)
    finding_snippet = af.matched_text[:200] if af.matched_text else None
    return Finding(
        rule_id=af.rule_id,
        message=af.message,
        severity=af.severity.value,
        confidence=af.confidence,
        file=af.location.file,
        start_line=af.location.start_line,
        end_line=af.location.end_line,
        remediation=remediation,
        tags=list(af.tags),
        context=af.context,
        matched_text=af.matched_text,
        category=category,
        pattern=pattern,
        finding=finding_snippet,
        explanation=get_explanation(af.rule_id),
        code_snippet=af.context,
        intent=None,
        evidence=dict(af.evidence),
    )


def _uses_python_ast(module: object) -> bool:
    """Return whether a pattern module explicitly opts into the shared AST hook."""
    return getattr(module, "USES_PYTHON_AST", False) is True


class _StaticResourceLimitError(RuntimeError):
    """Internal control-flow signal for one attacker-controlled work ceiling."""

    def __init__(
        self,
        reason: LedgerReason,
        metrics: dict[str, int | float],
        *,
        partial_findings: list[Finding] | None = None,
    ) -> None:
        super().__init__(reason.value)
        self.reason = reason
        self.metrics = metrics
        self.partial_findings = partial_findings or []


@dataclass
class _FindingBudget:
    """Bound findings while modules construct and return their private results."""

    max_findings: int
    started_at: float
    deadline: float
    clock: Callable[[], float]
    created_findings: int = 0
    emitted_findings: int = 0
    current_created: list[AnalyzerFinding] = field(default_factory=list)

    def _runtime_metrics(self, now: float) -> dict[str, int | float]:
        return {
            "observed_seconds": max(0.0, now - self.started_at),
            "limit_seconds": max(0.0, self.deadline - self.started_at),
        }

    def check_runtime(self) -> None:
        now = self.clock()
        if now >= self.deadline:
            raise _StaticResourceLimitError(
                LedgerReason.RUNTIME_LIMIT,
                self._runtime_metrics(now),
            )

    def begin_module(self) -> None:
        self.current_created = []
        self.check_runtime()

    def observe_creation(self, finding: AnalyzerFinding) -> None:
        """Stop list-building analyzers before a large private list is materialized."""
        self.check_runtime()
        self.created_findings += 1
        if self.created_findings > self.max_findings:
            raise _StaticResourceLimitError(
                LedgerReason.OUTPUT_LIMIT,
                {
                    "observed_findings": self.created_findings,
                    "limit_findings": self.max_findings,
                },
            )
        self.current_created.append(finding)

    def observe_emission(self) -> None:
        """Bound generators and modules returning preconstructed finding objects."""
        self.check_runtime()
        self.emitted_findings += 1
        if self.emitted_findings > self.max_findings:
            raise _StaticResourceLimitError(
                LedgerReason.OUTPUT_LIMIT,
                {
                    "observed_findings": self.emitted_findings,
                    "limit_findings": self.max_findings,
                },
            )


@dataclass(frozen=True)
class _ContinuityView:
    """One bounded cross-window projection with exact raw line locations."""

    view: SecurityTextView
    source_lines: tuple[int, ...]


@dataclass(frozen=True)
class _WindowSourceContext:
    """Shared whole-artifact coordinates for marker and raw window scans."""

    line_starts: tuple[int, ...]
    fence_states: dict[int, tuple[str, int] | None]
    fence_transitions: dict[int, tuple[str, int, str, int, int]]


def _build_window_source_context(
    path: str,
    content: str,
    raw_starts: tuple[int, ...],
) -> _WindowSourceContext:
    """Build line and Markdown state once for every scanner window origin."""
    line_starts = (
        0,
        *(separator.end() for separator in LOGICAL_LINE_BREAK.finditer(content)),
    )
    fence_states, fence_transitions = (
        _markdown_fence_states(content, raw_starts)
        if _infer_file_type(path) in {"markdown", "text"}
        else ({}, {})
    )
    return _WindowSourceContext(line_starts, fence_states, fence_transitions)


def _convert_analyzer_finding(
    af: AnalyzerFinding,
    *,
    path: str,
    file_type: str,
    content: str,
    content_lines: list[str],
    normalized_license_lines: tuple[str, ...] | None,
) -> Finding | None:
    """Apply contextual filters and convert one already-budgeted finding."""
    if (
        af.rule_id == "EA3"
        and normalized_license_lines is not None
        and _is_license_boilerplate_in_normalized_lines(
            normalized_license_lines,
            af.location.start_line,
        )
    ):
        logger.debug("Filtered EA3 license boilerplate finding: %s", path)
        return None
    if _is_env_file_reference_in_docs(
        af,
        file_type,
        path,
        content,
        content_lines,
    ):
        for triage_tag in ("contextual-triage", "likely-benign-context"):
            if triage_tag not in af.tags:
                af.tags.append(triage_tag)
    return analyzer_finding_to_finding(af)


def _scan_path(
    path: str,
    content: str,
    pattern_modules: list,
    finding_budget: _FindingBudget,
    python_ast_cache_key: str | None = None,
) -> tuple[list[Finding], _StaticResourceLimitError | None]:
    """Run pattern modules with construction, emission, and runtime guards."""
    findings: list[Finding] = []
    file_type = _infer_file_type(path)
    content_lines = content.splitlines()
    normalized_license_lines = (
        tuple(_normalize_license_line(line) for line in content_lines)
        if _is_license_basename(path, file_type)
        else None
    )
    python_ast: ParsedPythonFile | None = None
    if file_type == "python" and any(_uses_python_ast(module) for module in pattern_modules):
        finding_budget.check_runtime()
        python_ast = get_python_ast(python_ast_cache_key, content, path)
        finding_budget.check_runtime()

    for module in pattern_modules:
        module_finding_start = len(findings)
        finding_budget.begin_module()
        try:
            with observe_analyzer_findings(finding_budget.observe_creation):
                if file_type == "python" and _uses_python_ast(module):
                    raw = module.analyze(
                        content=content,
                        file_path=path,
                        file_type=file_type,
                        python_ast=python_ast,
                    )
                else:
                    raw = module.analyze(content=content, file_path=path, file_type=file_type)
                finding_budget.check_runtime()
                for af in raw:
                    finding_budget.observe_emission()
                    converted = _convert_analyzer_finding(
                        af,
                        path=path,
                        file_type=file_type,
                        content=content,
                        content_lines=content_lines,
                        normalized_license_lines=normalized_license_lines,
                    )
                    if converted is not None:
                        findings.append(converted)
        except _StaticResourceLimitError as exc:
            # A list-building module may be interrupted before it can return.
            # Preserve the bounded prefix it constructed so high-severity
            # evidence is not discarded merely because the output ceiling hit.
            if len(findings) == module_finding_start:
                for af in finding_budget.current_created:
                    if finding_budget.emitted_findings >= finding_budget.max_findings:
                        break
                    finding_budget.emitted_findings += 1
                    converted = _convert_analyzer_finding(
                        af,
                        path=path,
                        file_type=file_type,
                        content=content,
                        content_lines=content_lines,
                        normalized_license_lines=normalized_license_lines,
                    )
                    if converted is not None:
                        findings.append(converted)
            return findings, exc
    return findings, None


def _view_finding_key(finding: Finding) -> _ViewFindingKey:
    return (
        finding.rule_id,
        finding.file,
        finding.start_line,
        finding.fingerprint(),
        classification_metadata_key(finding, ignored_tags=_VIEW_ORIGIN_TAGS),
    )


def _view_scope_key(finding: Finding) -> _ViewScopeKey:
    return (
        finding.rule_id,
        finding.file,
        finding.start_line,
        finding.fingerprint(),
    )


def _extend_unique_findings(
    result: list[Finding],
    seen: set[_ViewFindingKey],
    candidates: list[Finding],
    *,
    max_findings: int,
) -> _StaticResourceLimitError | None:
    """Append distinct final findings and enforce the user-visible output cap."""
    for finding in candidates:
        key = _view_finding_key(finding)
        if key in seen:
            continue
        seen.add(key)
        result.append(finding)
        if len(result) > max_findings:
            return _StaticResourceLimitError(
                LedgerReason.OUTPUT_LIMIT,
                {
                    "observed_findings": len(result),
                    "limit_findings": max_findings,
                },
            )
    return None


def _deduplicate_view_findings(findings: list[Finding]) -> list[Finding]:
    """Remove only location- and classification-equivalent view duplicates."""
    result: list[Finding] = []
    seen: set[_ViewFindingKey] = set()
    raw_keys = {
        _view_finding_key(finding) for finding in findings if "normalized-view" not in finding.tags
    }
    raw_non_benign_scopes = {
        _view_scope_key(finding)
        for finding in findings
        if "normalized-view" not in finding.tags and not _BENIGN_CONTEXT_TAGS.issubset(finding.tags)
    }
    for finding in findings:
        key = _view_finding_key(finding)
        if "normalized-view" in finding.tags and key in raw_keys:
            continue
        if (
            "normalized-view" in finding.tags
            and _BENIGN_CONTEXT_TAGS.issubset(finding.tags)
            and _view_scope_key(finding) in raw_non_benign_scopes
        ):
            # Normalization may make an ambiguous raw occurrence look benign.
            # Prefer the raw non-benign signal, but never suppress a derived
            # unsafe classification that exposes obfuscated content.
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(finding)
    return result


def _scan_view_windows(
    path: str,
    view: SecurityTextView,
    pattern_modules: list,
    finding_budget: _FindingBudget,
    python_ast_cache_key: str | None,
) -> tuple[list[Finding], _StaticResourceLimitError | None]:
    """Scan one already-bounded view."""
    findings, resource_limit = _scan_path(
        path,
        view.text,
        pattern_modules,
        finding_budget,
        python_ast_cache_key,
    )
    for finding in findings:
        local_start = finding.evidence.pop(_VIEW_START_EVIDENCE, None)
        if isinstance(local_start, int) and 0 <= local_start < len(view.text):
            finding.evidence[_SOURCE_START_EVIDENCE] = view.source_offset(local_start)
    if view.name != "raw":
        for finding in findings:
            if "normalized-view" not in finding.tags:
                finding.tags.append("normalized-view")
    if view.name.startswith("declared-marker-"):
        for finding in findings:
            if "declared-marker-view" not in finding.tags:
                finding.tags.append("declared-marker-view")
    return findings, resource_limit


def _bounded_view_slices(view: SecurityTextView) -> Iterator[SecurityTextView]:
    """Split an expanded derived view before any pattern module sees it."""
    if len(view.text) <= SECURITY_VIEW_WINDOW_CHARS:
        yield view
        return
    step = SECURITY_VIEW_WINDOW_CHARS - _WINDOW_OVERLAP_CHARS
    for start in range(0, len(view.text), step):
        end = min(len(view.text), start + SECURITY_VIEW_WINDOW_CHARS)
        offsets = None if view.source_offsets is None else view.source_offsets[start:end]
        yield SecurityTextView(
            name=view.name,
            text=view.text[start:end],
            source_offsets=offsets,
        )
        if end == len(view.text):
            break


def _is_continuity_separator(character: str) -> bool:
    """Return whether a character separates tokens in a security text view."""
    return (
        character.isspace()
        or character == "\u00ad"
        or character == "\ufffd"
        or is_default_ignorable(character)
        or unicodedata.category(character) in {"Cf", "Cc"}
    )


def _continuity_separator_runs(
    content: str,
    finding_budget: _FindingBudget,
) -> Iterator[tuple[int, int]]:
    """Yield long separator runs without allocating a whole-file projection."""
    if content.isascii():
        # Keep ordinary source files on the regex engine's bounded C-level
        # fast path.  Unicode category inspection below is reserved for input
        # that can actually contain normalized-away format characters.
        for match in _ASCII_CONTINUITY_SEPARATOR_RUN.finditer(content):
            finding_budget.check_runtime()
            if match.end() - match.start() > _WINDOW_OVERLAP_CHARS:
                yield match.start(), match.end()
        return

    # A printable Unicode artifact with no ASCII whitespace/control,
    # replacement character, or pinned default-ignorable cannot contain any
    # character accepted by ``_is_continuity_separator``. Keep that common
    # multilingual-text case on C-level predicates instead of walking every
    # code point in Python.
    if (
        content.isprintable()
        and _ASCII_CONTINUITY_SEPARATOR_RUN.search(content) is None
        and "\ufffd" not in content
        and not _contains_default_ignorable(content)
    ):
        finding_budget.check_runtime()
        return

    run_start: int | None = None
    for index, character in enumerate(content):
        if index % _WINDOW_OVERLAP_CHARS == 0:
            finding_budget.check_runtime()
        if _is_continuity_separator(character):
            if run_start is None:
                run_start = index
            continue
        if run_start is not None and index - run_start > _WINDOW_OVERLAP_CHARS:
            yield run_start, index
        run_start = None
    if run_start is not None and len(content) - run_start > _WINDOW_OVERLAP_CHARS:
        yield run_start, len(content)


def _append_projected_piece(
    text_parts: list[str],
    source_lines: list[int],
    piece: str,
    source_line: int,
) -> int:
    """Append one contiguous raw piece and extend its exact line projection."""
    text_parts.append(piece)
    for _ in LOGICAL_LINE_BREAK.finditer(piece):
        source_line += 1
        source_lines.append(source_line)
    return source_line


def _continuity_views(
    content: str,
    finding_budget: _FindingBudget,
) -> Iterator[_ContinuityView]:
    """Build bounded neighborhoods that preserve lexical state across raw windows.

    Separator runs wider than the normal overlap can otherwise place two
    adjacent lexical tokens in different windows.  Retaining up to 8 KiB of
    the original run preserves ASCII whitespace boundaries and newlines while
    keeping every bounded-gap expression bounded.  Expressions that already
    accept an unbounded separator see the same token sequence.  The source-line
    map is constructed per view, so neither a whole-file normalized copy nor a
    whole-file offset table exists.
    """
    separator_runs = list(_continuity_separator_runs(content, finding_budget))
    previous_left = 0
    previous_left_line = 1
    for run_index, (run_start, _) in enumerate(separator_runs):
        finding_budget.check_runtime()
        last_run_index = run_index
        while (
            last_run_index + 1 < len(separator_runs)
            and last_run_index - run_index + 1 < _CONTINUITY_MAX_CHAIN_RUNS
            and separator_runs[last_run_index + 1][0] - separator_runs[last_run_index][1]
            <= _CONTINUITY_CONTEXT_CHARS
        ):
            last_run_index += 1
        selected_runs = separator_runs[run_index : last_run_index + 1]
        left = max(0, run_start - _CONTINUITY_CONTEXT_CHARS)
        right = min(len(content), selected_runs[-1][1] + _CONTINUITY_CONTEXT_CHARS)
        previous_left_line += sum(
            1 for _ in LOGICAL_LINE_BREAK.finditer(content, previous_left, left)
        )
        previous_left = left
        source_lines = [previous_left_line]
        text_parts: list[str] = []
        current_line = previous_left_line
        cursor = left
        for selected_start, selected_end in selected_runs:
            current_line = _append_projected_piece(
                text_parts,
                source_lines,
                content[cursor:selected_start],
                current_line,
            )
            run_length = selected_end - selected_start
            if run_length <= _CONTINUITY_SEPARATOR_CHARS:
                current_line = _append_projected_piece(
                    text_parts,
                    source_lines,
                    content[selected_start:selected_end],
                    current_line,
                )
            else:
                head_length = _CONTINUITY_SEPARATOR_CHARS // 2
                tail_length = _CONTINUITY_SEPARATOR_CHARS - head_length
                head_end = selected_start + head_length
                tail_start = selected_end - tail_length
                current_line = _append_projected_piece(
                    text_parts,
                    source_lines,
                    content[selected_start:head_end],
                    current_line,
                )
                skipped_newlines = sum(
                    1 for _ in LOGICAL_LINE_BREAK.finditer(content, head_end, tail_start)
                )
                if skipped_newlines:
                    # Retain a line boundary so DOT-without-DOTALL and anchors
                    # do not acquire semantics absent from the original source.
                    text_parts.append("\n")
                    current_line += skipped_newlines
                    source_lines.append(current_line)
                elif _ASCII_NON_NEWLINE_WHITESPACE.search(content, head_end, tail_start):
                    # Never let truncation erase a real word boundary and turn
                    # separated tokens into a normalized security match.
                    text_parts.append(" ")
                current_line = _append_projected_piece(
                    text_parts,
                    source_lines,
                    content[tail_start:selected_end],
                    current_line,
                )
            cursor = selected_end
        _append_projected_piece(
            text_parts,
            source_lines,
            content[cursor:right],
            current_line,
        )

        projected = "".join(text_parts)
        # Context, the retained separators, and the bounded text between
        # chained runs remain below the ordinary module-input ceiling.
        assert len(projected) <= SECURITY_VIEW_WINDOW_CHARS
        yield _ContinuityView(
            view=SecurityTextView("continuity", projected),
            source_lines=tuple(source_lines),
        )


def _restore_continuity_lines(
    findings: list[Finding],
    source_lines: tuple[int, ...],
) -> None:
    """Restore projected finding lines without scanning an unbounded prefix."""
    if not source_lines:
        return
    for finding in findings:
        start_index = min(max(finding.start_line - 1, 0), len(source_lines) - 1)
        finding.start_line = source_lines[start_index]
        if finding.end_line is not None:
            end_index = min(max(finding.end_line - 1, 0), len(source_lines) - 1)
            finding.end_line = source_lines[end_index]


def _continuity_finding_key(finding: Finding) -> tuple[object, ...]:
    """Identify equivalent raw/continuity signals without match-text drift."""
    return (
        finding.rule_id,
        finding.file,
        finding.start_line,
        finding.end_line,
        finding.message,
        finding.severity,
        finding.confidence,
        classification_metadata_key(finding, ignored_tags=_VIEW_ORIGIN_TAGS),
    )


def _line_start_offset(text: str, line_number: int) -> int:
    """Return the local character offset for a 1-based line number."""
    if line_number <= 1:
        return 0
    offset = 0
    for _ in range(line_number - 1):
        separator = LOGICAL_LINE_BREAK.search(text, offset)
        if separator is None:
            return len(text)
        offset = separator.end()
    return offset


def _restore_source_lines(
    findings: list[Finding],
    *,
    raw_window: str,
    window_line: int,
    view: SecurityTextView,
    window_start: int = 0,
    source_line_starts: tuple[int, ...] | None = None,
) -> None:
    """Map normalized/window-relative locations to raw whole-file lines."""

    def source_line(raw_offset: int) -> int:
        if source_line_starts is not None:
            return bisect_right(source_line_starts, window_start + raw_offset)
        return window_line + sum(1 for _ in LOGICAL_LINE_BREAK.finditer(raw_window, 0, raw_offset))

    for finding in findings:
        derived_start = _line_start_offset(view.text, finding.start_line)
        raw_start = view.source_offset(derived_start)
        finding.start_line = source_line(raw_start)
        if finding.end_line is not None:
            derived_end = _line_start_offset(view.text, finding.end_line)
            raw_end = view.source_offset(derived_end)
            finding.end_line = source_line(raw_end)
        finding.evidence.pop(_SOURCE_START_EVIDENCE, None)


def _scan_declared_marker_views(
    path: str,
    content: str,
    pattern_modules: list,
    finding_budget: _FindingBudget,
    *,
    owned_starts: tuple[int, ...],
    raw_starts: tuple[int, ...],
    source_context: _WindowSourceContext,
) -> tuple[list[Finding], bool, _StaticResourceLimitError | None]:
    """Reconstruct marker payloads with directive-relative context windows."""
    findings: list[Finding] = []

    def check_runtime() -> None:
        try:
            finding_budget.check_runtime()
        except _StaticResourceLimitError as exc:
            raise _StaticResourceLimitError(
                exc.reason,
                exc.metrics,
                partial_findings=list(findings),
            ) from exc

    check_runtime()
    projection_limited = False
    seen_views: set[tuple[str, int, int]] = set()
    seen_findings: set[_ViewFindingKey] = set()

    for owned_start, raw_start in zip(owned_starts, raw_starts, strict=True):
        check_runtime()
        owned_end = min(len(content), owned_start + DECLARED_MARKER_OWNED_CHARS)
        raw_end = min(len(content), owned_end + DECLARED_MARKER_RIGHT_CONTEXT_CHARS)
        raw_window = content[raw_start:raw_end]
        owned_source_start = owned_start - raw_start
        owned_source_end = owned_end - raw_start if owned_end < len(content) else None
        context_prefix = _markdown_context_prefix(
            content,
            raw_start,
            raw_end,
            source_context.fence_states,
            source_context.fence_transitions,
        )

        check_runtime()
        full_views = tuple(
            _window_view_with_markdown_context(full_view, len(context_prefix))
            for full_view in security_text_views(context_prefix + raw_window)
        )
        check_runtime()
        for full_view in full_views:
            reconstruction = build_declared_marker_views(
                full_view,
                check_runtime=check_runtime,
                owned_source_start=owned_source_start,
                owned_source_end=owned_source_end,
                source_end_is_truncated=raw_end < len(content),
            )
            projection_limited = projection_limited or reconstruction.limited
            for marker_view in reconstruction.views:
                if not marker_view.source_offsets:
                    continue
                marker_key = (
                    marker_view.text,
                    raw_start + marker_view.source_offsets[0],
                    raw_start + marker_view.source_offsets[-1],
                )
                if marker_key in seen_views:
                    continue
                seen_views.add(marker_key)
                for view in _bounded_view_slices(marker_view):
                    check_runtime()
                    view_budget = _FindingBudget(
                        max_findings=finding_budget.max_findings,
                        started_at=finding_budget.started_at,
                        deadline=finding_budget.deadline,
                        clock=finding_budget.clock,
                    )
                    view_findings, resource_limit = _scan_view_windows(
                        path,
                        view,
                        pattern_modules,
                        view_budget,
                        None,
                    )
                    _restore_source_lines(
                        view_findings,
                        raw_window=raw_window,
                        window_line=1,
                        view=view,
                        window_start=raw_start,
                        source_line_starts=source_context.line_starts,
                    )
                    for finding in view_findings:
                        key = _view_finding_key(finding)
                        if key in seen_findings:
                            continue
                        seen_findings.add(key)
                        findings.append(finding)
                        if len(findings) > finding_budget.max_findings:
                            return (
                                findings,
                                projection_limited,
                                _StaticResourceLimitError(
                                    LedgerReason.OUTPUT_LIMIT,
                                    {
                                        "observed_findings": len(findings),
                                        "limit_findings": finding_budget.max_findings,
                                    },
                                ),
                            )
                    if resource_limit is not None:
                        return findings, projection_limited, resource_limit

        if owned_end == len(content):
            break

    return findings, projection_limited, None


def _scan_all_views_detailed(
    path: str,
    content: str,
    pattern_modules: list,
    python_ast_cache_key: str | None,
    *,
    max_findings: int = MAX_FINDINGS_PER_ARTIFACT,
    timeout_seconds: float | None = None,
) -> tuple[list[Finding], LedgerReason | None, dict[str, int | float]]:
    """Scan bounded raw windows and return any limit with observed/limit metrics."""
    ast_modules = [module for module in pattern_modules if _uses_python_ast(module)]
    lexical_modules = [module for module in pattern_modules if not _uses_python_ast(module)]
    findings: list[Finding] = []
    seen_findings: set[_ViewFindingKey] = set()
    started_at = time.monotonic()
    runtime_limit = MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT
    if timeout_seconds is not None:
        runtime_limit = min(runtime_limit, max(0.0, timeout_seconds))
    deadline = started_at + runtime_limit
    finding_budget = _FindingBudget(
        max_findings=max(0, max_findings),
        started_at=started_at,
        deadline=deadline,
        clock=time.monotonic,
    )
    marker_projection_limited = False
    modules_for_windows = lexical_modules or ([] if ast_modules else pattern_modules)
    bounded_parse_limited = False
    marker_owned_starts: tuple[int, ...] = ()
    marker_raw_starts: tuple[int, ...] = ()
    raw_owned_starts: tuple[int, ...] = ()
    raw_starts: tuple[int, ...] = ()
    source_context: _WindowSourceContext | None = None
    whole_artifact_window = False

    if modules_for_windows:
        marker_owned_starts = tuple(range(0, max(1, len(content)), DECLARED_MARKER_OWNED_CHARS))
        marker_raw_starts = tuple(
            max(0, owned_start - DECLARED_MARKER_LEFT_CONTEXT_CHARS)
            for owned_start in marker_owned_starts
        )
        whole_artifact_window = len(content) <= SECURITY_VIEW_WINDOW_CHARS
        raw_owned_starts = (
            (0,)
            if whole_artifact_window
            else tuple(range(0, max(1, len(content)), _RAW_WINDOW_OWNED_CHARS))
        )
        raw_starts = tuple(
            0 if whole_artifact_window else max(0, owned_start - _WINDOW_OVERLAP_CHARS)
            for owned_start in raw_owned_starts
        )
        marker_budget = _FindingBudget(
            max_findings=max(0, max_findings),
            started_at=started_at,
            deadline=deadline,
            clock=time.monotonic,
        )
        try:
            finding_budget.check_runtime()
            source_context = _build_window_source_context(
                path,
                content,
                tuple(sorted(set(marker_raw_starts).union(raw_starts))),
            )
            finding_budget.check_runtime()
            marker_findings, marker_projection_limited, resource_limit = (
                _scan_declared_marker_views(
                    path,
                    content,
                    modules_for_windows,
                    marker_budget,
                    owned_starts=marker_owned_starts,
                    raw_starts=marker_raw_starts,
                    source_context=source_context,
                )
            )
        except _StaticResourceLimitError as exc:
            _extend_unique_findings(
                findings,
                seen_findings,
                exc.partial_findings,
                max_findings=max_findings,
            )
            return (
                findings[:max_findings],
                exc.reason,
                exc.metrics,
            )
        unique_limit = _extend_unique_findings(
            findings,
            seen_findings,
            marker_findings,
            max_findings=max_findings,
        )
        if unique_limit is not None:
            return findings[:max_findings], unique_limit.reason, unique_limit.metrics
        if resource_limit is not None:
            return (
                _deduplicate_view_findings(findings)[:max_findings],
                resource_limit.reason,
                resource_limit.metrics,
            )

    if ast_modules and len(content) <= MAX_FILE_CHARS:
        try:
            ast_findings, resource_limit = _scan_path(
                path,
                content,
                ast_modules,
                finding_budget,
                python_ast_cache_key,
            )
        except _StaticResourceLimitError as exc:
            return _deduplicate_view_findings(findings), exc.reason, exc.metrics
        unique_limit = _extend_unique_findings(
            findings,
            seen_findings,
            ast_findings,
            max_findings=max_findings,
        )
        if unique_limit is not None:
            return findings[:max_findings], unique_limit.reason, unique_limit.metrics
        if resource_limit is not None:
            return (
                _deduplicate_view_findings(findings)[:max_findings],
                resource_limit.reason,
                resource_limit.metrics,
            )

    if modules_for_windows:
        assert source_context is not None
        for owned_start, raw_start in zip(raw_owned_starts, raw_starts, strict=True):
            now = time.monotonic()
            if now >= deadline:
                return (
                    _deduplicate_view_findings(findings),
                    LedgerReason.RUNTIME_LIMIT,
                    {
                        "observed_seconds": max(0.0, now - started_at),
                        "limit_seconds": runtime_limit,
                    },
                )
            owned_end = (
                len(content)
                if whole_artifact_window
                else min(len(content), owned_start + _RAW_WINDOW_OWNED_CHARS)
            )
            raw_start = 0 if whole_artifact_window else max(0, owned_start - _WINDOW_OVERLAP_CHARS)
            raw_end = (
                len(content)
                if whole_artifact_window
                else min(len(content), owned_end + _WINDOW_OVERLAP_CHARS)
            )
            raw_window = content[raw_start:raw_end]
            owned_source_start = owned_start - raw_start
            owned_source_end = owned_end - raw_start
            context_prefix = _markdown_context_prefix(
                content,
                raw_start,
                raw_end,
                source_context.fence_states,
                source_context.fence_transitions,
            )
            for full_view in security_text_views(context_prefix + raw_window):
                full_view = _window_view_with_markdown_context(full_view, len(context_prefix))
                try:
                    for module in modules_for_windows:
                        exhaustion_hook = getattr(
                            module,
                            "has_bounded_parse_exhaustion",
                            None,
                        )
                        if callable(exhaustion_hook):
                            finding_budget.check_runtime()
                            bounded_parse_limited = bounded_parse_limited or bool(
                                exhaustion_hook(
                                    full_view.text,
                                    finding_budget.check_runtime,
                                )
                            )
                except _StaticResourceLimitError as exc:
                    return (
                        _deduplicate_view_findings(findings)[:max_findings],
                        exc.reason,
                        exc.metrics,
                    )
                for view in _bounded_view_slices(full_view):
                    try:
                        finding_budget.check_runtime()
                        view_budget = _FindingBudget(
                            max_findings=max(0, max_findings),
                            started_at=started_at,
                            deadline=deadline,
                            clock=finding_budget.clock,
                        )
                        view_findings, resource_limit = _scan_view_windows(
                            path,
                            view,
                            modules_for_windows,
                            view_budget,
                            None,
                        )
                    except _StaticResourceLimitError as exc:
                        return (
                            _deduplicate_view_findings(findings)[:max_findings],
                            exc.reason,
                            exc.metrics,
                        )
                    owned_findings: list[Finding] = []
                    for finding in view_findings:
                        source_start = finding.evidence.get(_SOURCE_START_EVIDENCE)
                        if isinstance(source_start, int) and not (
                            owned_source_start <= source_start < owned_source_end
                        ):
                            continue
                        owned_findings.append(finding)
                    view_findings = owned_findings
                    _restore_source_lines(
                        view_findings,
                        raw_window=raw_window,
                        window_line=1,
                        view=view,
                        window_start=raw_start,
                        source_line_starts=source_context.line_starts,
                    )
                    unique_limit = _extend_unique_findings(
                        findings,
                        seen_findings,
                        view_findings,
                        max_findings=max_findings,
                    )
                    if unique_limit is not None:
                        return findings[:max_findings], unique_limit.reason, unique_limit.metrics
                    if resource_limit is not None:
                        return (
                            _deduplicate_view_findings(findings)[:max_findings],
                            resource_limit.reason,
                            resource_limit.metrics,
                        )
            if owned_end == len(content):
                break

        # Raw windows intentionally remain small, but a separator wider than
        # their overlap can split a lexical expression even though the
        # analyzer's own expression accepts that separator without a bound.
        # Scan only bounded neighborhoods of those runs.  This is additive:
        # raw findings win, padding-only auxiliary findings are discarded, and
        # all resource accounting remains on the same artifact budget.
        continuity_seen = {_continuity_finding_key(finding) for finding in findings}
        try:
            for continuity in _continuity_views(content, finding_budget):
                for full_view in security_text_views(continuity.view.text):
                    named_view = SecurityTextView(
                        name=f"continuity-{full_view.name}",
                        text=full_view.text,
                        source_offsets=full_view.source_offsets,
                    )
                    for view in _bounded_view_slices(named_view):
                        finding_budget.check_runtime()
                        view_budget = _FindingBudget(
                            max_findings=max(0, max_findings),
                            started_at=started_at,
                            deadline=deadline,
                            clock=finding_budget.clock,
                        )
                        view_findings, resource_limit = _scan_view_windows(
                            path,
                            view,
                            modules_for_windows,
                            view_budget,
                            None,
                        )
                        _restore_source_lines(
                            view_findings,
                            raw_window=continuity.view.text,
                            window_line=1,
                            view=view,
                        )
                        _restore_continuity_lines(
                            view_findings,
                            continuity.source_lines,
                        )
                        for finding in view_findings:
                            key = _continuity_finding_key(finding)
                            if finding.rule_id == "P9" or key in continuity_seen:
                                continue
                            continuity_seen.add(key)
                            unique_limit = _extend_unique_findings(
                                findings,
                                seen_findings,
                                [finding],
                                max_findings=max_findings,
                            )
                            if unique_limit is not None:
                                return (
                                    findings[:max_findings],
                                    unique_limit.reason,
                                    unique_limit.metrics,
                                )
                        if resource_limit is not None:
                            return (
                                _deduplicate_view_findings(findings)[:max_findings],
                                resource_limit.reason,
                                resource_limit.metrics,
                            )
        except _StaticResourceLimitError as exc:
            return (
                _deduplicate_view_findings(findings)[:max_findings],
                exc.reason,
                exc.metrics,
            )

    deduplicated = _deduplicate_view_findings(findings)
    if len(deduplicated) > max_findings:
        return (
            deduplicated[:max_findings],
            LedgerReason.OUTPUT_LIMIT,
            {
                "observed_findings": len(deduplicated),
                "limit_findings": max_findings,
            },
        )
    return (
        deduplicated,
        (
            LedgerReason.STATIC_PARSE_LIMIT
            if bounded_parse_limited
            else LedgerReason.OBFUSCATED_INSTRUCTION_TEXT
            if marker_projection_limited
            else None
        ),
        {},
    )


def _scan_all_views(
    path: str,
    content: str,
    pattern_modules: list,
    python_ast_cache_key: str | None,
    *,
    max_findings: int = MAX_FINDINGS_PER_ARTIFACT,
    timeout_seconds: float | None = None,
) -> list[Finding]:
    findings, _, _ = _scan_all_views_detailed(
        path,
        content,
        pattern_modules,
        python_ast_cache_key,
        max_findings=max_findings,
        timeout_seconds=timeout_seconds,
    )
    return findings


def run_static_patterns(
    state: Mapping[str, object],
    pattern_modules: list,
) -> list[Finding]:
    """
    Run one or more pattern modules over state components/file_cache.

    For each path in state["components"], loads content from state["file_cache"],
    infers file_type, runs each module's analyze(content, path, file_type),
    converts all AnalyzerFindings to Finding via analyzer_finding_to_finding, returns combined list.
    """
    components = cast(list[str], state.get("components") or [])
    file_cache = cast(
        dict[str, str], state.get("local_file_cache") or state.get("file_cache") or {}
    )
    python_ast_cache_key = cast(str | None, state.get("python_ast_cache_key"))
    container_paths = {
        str(metadata.get("path", ""))
        for metadata in cast(list[dict[str, object]], state.get("component_metadata") or [])
        if metadata.get("container_type") in {"zip", "docx", "xlsx", "pptx"}
        and "!/" not in str(metadata.get("path", ""))
    }
    raw_inventory = state.get("artifact_inventory", [])
    binary_paths = (
        {
            str(item.get("path", ""))
            for item in raw_inventory
            if isinstance(item, dict) and item.get("content_kind") == ContentKind.BINARY
        }
        if isinstance(raw_inventory, list)
        else set()
    )
    findings: list[Finding] = []

    for path in components:
        if path in container_paths:
            continue
        content = file_cache.get(path)
        if content is None:
            logger.debug("Skipping %s: no content in file_cache", path)
            continue
        if path in binary_paths or (not binary_paths and _is_binary_file(path, content)):
            continue
        remaining = MAX_FINDINGS_PER_ANALYZER - len(findings)
        if remaining <= 0:
            break
        shared_remaining = transitive_remaining_seconds(cast(SkillspectorState, state))
        if shared_remaining is not None and shared_remaining <= 0:
            break
        findings.extend(
            _scan_all_views(
                path,
                content,
                pattern_modules,
                python_ast_cache_key,
                max_findings=min(MAX_FINDINGS_PER_ARTIFACT, remaining),
                timeout_seconds=shared_remaining,
            )
        )

    return findings


def run_static_patterns_with_ledger(
    state: Mapping[str, object],
    pattern_modules: list,
) -> AnalyzerNodeResponse:
    """Run one static analyzer and account for every planned file work item."""
    analyzer_id = str(getattr(pattern_modules[0], "ANALYZER_ID", "static_patterns"))
    components = cast(list[str], state.get("components") or [])
    file_cache = cast(
        dict[str, str], state.get("local_file_cache") or state.get("file_cache") or {}
    )
    python_ast_cache_key = cast(str | None, state.get("python_ast_cache_key"))
    container_paths = {
        str(metadata.get("path", ""))
        for metadata in cast(list[dict[str, object]], state.get("component_metadata") or [])
        if metadata.get("container_type") in {"zip", "docx", "xlsx", "pptx"}
        and "!/" not in str(metadata.get("path", ""))
    }
    findings: list[Finding] = []
    events: list[InspectionLedgerEvent] = []
    raw_inventory = state.get("artifact_inventory", [])
    inventory: dict[str, dict[str, object]] = (
        {str(item.get("path", "")): item for item in raw_inventory if isinstance(item, dict)}
        if isinstance(raw_inventory, list)
        else {}
    )

    for path in components:
        if path in container_paths:
            event = ledger_event(
                outcome=LedgerOutcome.COMPLETED,
                phase="static",
                analyzer_id=analyzer_id,
                path=path,
            )
        else:
            artifact = inventory.get(path, {})
        if path not in container_paths and artifact.get("content_kind") == ContentKind.OPAQUE:
            event = ledger_event(
                outcome=(
                    LedgerOutcome.FAILED
                    if artifact.get("disposition") == "failed"
                    else LedgerOutcome.PARTIAL
                ),
                phase="static",
                analyzer_id=analyzer_id,
                path=path,
                reason=LedgerReason.OPAQUE_CONTENT,
            )
        elif path not in container_paths and artifact.get("content_kind") == ContentKind.BINARY:
            referenced = bool(artifact.get("referenced"))
            event = ledger_event(
                outcome=LedgerOutcome.PARTIAL if referenced else LedgerOutcome.OUT_OF_SCOPE,
                record_type=(
                    LedgerRecordType.WORK_ITEM if referenced else LedgerRecordType.SCOPE_BOUNDARY
                ),
                phase="static",
                analyzer_id=analyzer_id,
                path=path,
                reason=(LedgerReason.OPAQUE_CONTENT if referenced else LedgerReason.BINARY_CONTENT),
            )
        elif path not in container_paths:
            content = file_cache.get(path)
            if content is None:
                event = ledger_event(
                    outcome=LedgerOutcome.FAILED,
                    phase="static",
                    analyzer_id=analyzer_id,
                    path=path,
                    reason=LedgerReason.MISSING_FILE_CACHE,
                )
            elif len(findings) >= MAX_FINDINGS_PER_ANALYZER:
                event = ledger_event(
                    outcome=LedgerOutcome.PARTIAL,
                    phase="static",
                    analyzer_id=analyzer_id,
                    path=path,
                    reason=LedgerReason.OUTPUT_LIMIT,
                    observed_findings=len(findings),
                    limit_findings=MAX_FINDINGS_PER_ANALYZER,
                )
            else:
                remaining = MAX_FINDINGS_PER_ANALYZER - len(findings)
                shared_remaining = transitive_remaining_seconds(cast(SkillspectorState, state))
                path_findings: list[Finding]
                resource_limit: LedgerReason | None
                resource_metrics: dict[str, int | float]
                if shared_remaining is not None and shared_remaining <= 0:
                    path_findings = []
                    resource_limit = LedgerReason.RUNTIME_LIMIT
                    resource_metrics = {
                        "observed_seconds": 0.0,
                        "limit_seconds": 0.0,
                    }
                else:
                    try:
                        path_findings, resource_limit, resource_metrics = _scan_all_views_detailed(
                            path,
                            content,
                            pattern_modules,
                            python_ast_cache_key,
                            max_findings=min(MAX_FINDINGS_PER_ARTIFACT, remaining),
                            timeout_seconds=shared_remaining,
                        )
                    except Exception as exc:
                        logger.warning("%s: scan error on %s: %s", analyzer_id, path, exc)
                        event = ledger_event(
                            outcome=LedgerOutcome.FAILED,
                            phase="static",
                            analyzer_id=analyzer_id,
                            path=path,
                            reason=LedgerReason.ANALYZER_RUNTIME_ERROR,
                            error_class=type(exc).__name__,
                        )
                        events.append(event)
                        continue
                if len(path_findings) > remaining:
                    resource_metrics = {
                        "observed_findings": len(findings) + len(path_findings),
                        "limit_findings": MAX_FINDINGS_PER_ANALYZER,
                    }
                    path_findings = path_findings[:remaining]
                    resource_limit = LedgerReason.OUTPUT_LIMIT
                findings.extend(path_findings)
                partial = resource_limit is not None or (
                    _infer_file_type(path) == "python"
                    and len(content) > MAX_FILE_CHARS
                    and any(_uses_python_ast(module) for module in pattern_modules)
                )
                partial_reason = resource_limit or LedgerReason.SIZE_LIMIT
                event = ledger_event(
                    outcome=LedgerOutcome.PARTIAL if partial else LedgerOutcome.COMPLETED,
                    phase="static",
                    analyzer_id=analyzer_id,
                    path=path,
                    reason=partial_reason if partial else None,
                    emitted_finding_ids=[finding.finding_id for finding in path_findings],
                    observed_characters=(
                        len(content) if partial_reason is LedgerReason.SIZE_LIMIT else None
                    ),
                    limit_characters=(
                        MAX_FILE_CHARS if partial_reason is LedgerReason.SIZE_LIMIT else None
                    ),
                    observed_findings=(
                        int(resource_metrics.get("observed_findings", len(path_findings)))
                        if partial_reason is LedgerReason.OUTPUT_LIMIT
                        else None
                    ),
                    limit_findings=(
                        int(resource_metrics.get("limit_findings", MAX_FINDINGS_PER_ARTIFACT))
                        if partial_reason is LedgerReason.OUTPUT_LIMIT
                        else None
                    ),
                    observed_seconds=(
                        float(resource_metrics.get("observed_seconds", 0.0))
                        if partial_reason is LedgerReason.RUNTIME_LIMIT
                        else None
                    ),
                    limit_seconds=(
                        float(
                            resource_metrics.get(
                                "limit_seconds", MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT
                            )
                        )
                        if partial_reason is LedgerReason.RUNTIME_LIMIT
                        else None
                    ),
                )
        events.append(event)

    return {
        "findings": findings,
        "inspection_ledger": events,
        "analyzer_status_events": [analyzer_status_for_events(analyzer_id, events)],
    }
