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

"""YARA analyzer node — runs curated and user-supplied YARA rules against skill artifacts.

Built-in rules ship in ``src/skillspector/yara_rules/`` (webshells, crypto miners, malware,
hack tools) based on industry open-source patterns. Users can supply additional rules via the
``--yara-rules-dir`` CLI flag; both directories are compiled together.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from pathlib import Path

import yara  # type: ignore[import-not-found]

from skillspector.inspection_ledger import (
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    analyzer_status_event,
    ledger_event,
)
from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from .common import get_context_from_lines
from .pattern_defaults import PatternCategory
from .static_runner import MAX_FILE_CHARS, analyzer_finding_to_finding

ANALYZER_ID = "static_yara"
logger = get_logger(__name__)

_BUILTIN_RULES_DIR = Path(__file__).resolve().parent.parent.parent / "yara_rules"

_RULE_EXTENSIONS = ("*.yar", "*.yara", "*.yar.b64", "*.yara.b64")
_ENCODED_RULE_SUFFIXES = (".yar.b64", ".yara.b64")

_CATEGORY_MAP: dict[str, tuple[str, Severity]] = {
    "malware": ("YR1", Severity.CRITICAL),
    "webshell": ("YR2", Severity.CRITICAL),
    "cryptominer": ("YR3", Severity.HIGH),
    "hack_tool": ("YR4", Severity.HIGH),
    "exploit": ("YR4", Severity.HIGH),
}
_DEFAULT_RULE_ID = "YR4"
_DEFAULT_SEVERITY = Severity.MEDIUM
_DEFAULT_CONFIDENCE = 0.7
_DESTRUCTIVE_AUTONOMY_NAMESPACE = "agent_skills"
_DESTRUCTIVE_AUTONOMY_RULE = "agent_skill_destructive_autonomous_actions"
_MAX_DESTRUCTIVE_AUTONOMY_LINE_DISTANCE = 3

# Module-level cache keyed by a content hash of all rule directories.
_compiled_rules: yara.Rules | None = None
_rules_hash: str | None = None


def _collect_rule_files(*dirs: Path) -> list[Path]:
    """Collect YARA files deterministically while preserving directory precedence."""
    files: list[Path] = []
    seen: set[Path] = set()
    for d in dirs:
        if not d.is_dir():
            continue
        directory_files: set[Path] = set()
        for ext in _RULE_EXTENSIONS:
            directory_files.update(d.rglob(ext))
        for rule_file in sorted(directory_files):
            if rule_file not in seen:
                seen.add(rule_file)
                files.append(rule_file)
    return files


def _content_hash(rule_files: list[Path]) -> str:
    """Hash over rule file paths and content for cache invalidation.

    Uses actual file content (not just size) so that edits which preserve
    file length still invalidate the cache.
    """
    h = hashlib.sha256()
    for p in rule_files:
        h.update(str(p).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _rule_namespace(rule_file: Path) -> str:
    """Derive a stable namespace from a rule file name."""
    for suffix in _ENCODED_RULE_SUFFIXES:
        if rule_file.name.endswith(suffix):
            return rule_file.name[: -len(suffix)]
    return rule_file.stem


def _read_rule_source(rule_file: Path) -> str:
    """Read a YARA rule source, decoding embedded packaged rules when needed."""
    if not rule_file.name.endswith(_ENCODED_RULE_SUFFIXES):
        return rule_file.read_text(encoding="utf-8")

    encoded_source = rule_file.read_text(encoding="utf-8")
    return base64.b64decode("".join(encoded_source.split())).decode("utf-8")


def _build_namespace_map(
    rule_files: list[Path], temp_dir: Path | None = None
) -> tuple[dict[str, str], int]:
    """Build a {namespace: source} dict and count malformed rule files."""
    del temp_dir
    sources: dict[str, str] = {}
    skipped = 0
    for rf in rule_files:
        ns = _rule_namespace(rf)
        if ns in sources:
            ns = f"{rf.parent.name}/{ns}"
        try:
            sources[ns] = _read_rule_source(rf)
        except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
            skipped += 1
            logger.debug("%s: skipping malformed encoded rule %s: %s", ANALYZER_ID, rf, exc)
    return sources, skipped


def _compile_rules(sources: dict[str, str]) -> tuple[yara.Rules | None, int]:
    """Compile YARA rules from a namespace map. Falls back to per-source compilation on error.

    Returns (compiled_rules, skipped_count).
    """
    try:
        return yara.compile(sources=sources), 0
    except yara.SyntaxError:
        pass

    logger.debug("%s: bulk compile failed, falling back to per-source compilation", ANALYZER_ID)
    good: dict[str, str] = {}
    skipped = 0
    for ns, source in sources.items():
        try:
            yara.compile(source=source)
            good[ns] = source
        except (yara.SyntaxError, yara.Error) as exc:
            skipped += 1
            logger.debug("%s: skipping %s: %s", ANALYZER_ID, ns, exc)

    compiled = yara.compile(sources=good) if good else None
    return compiled, skipped


def _load_rules(extra_dir: Path | None = None) -> yara.Rules | None:
    """Compile YARA rules from built-in and optional user-supplied directories.

    Results are cached at module level and reused if directory contents haven't changed.
    """
    global _compiled_rules, _rules_hash  # noqa: PLW0603

    dirs = [_BUILTIN_RULES_DIR]
    if extra_dir and extra_dir.is_dir():
        dirs.append(extra_dir)
    elif extra_dir:
        logger.warning("%s: user rules directory %s does not exist", ANALYZER_ID, extra_dir)

    rule_files = _collect_rule_files(*dirs)
    if not rule_files:
        logger.info("%s: no YARA rule files found", ANALYZER_ID)
        return None

    current_hash = _content_hash(rule_files)
    if _compiled_rules is not None and _rules_hash == current_hash:
        return _compiled_rules

    sources, materialize_skipped = _build_namespace_map(rule_files)
    compiled, compile_skipped = _compile_rules(sources)
    skipped = materialize_skipped + compile_skipped

    if compiled is None:
        logger.warning("%s: failed to compile any YARA rules", ANALYZER_ID)
        return None

    _compiled_rules = compiled
    _rules_hash = current_hash
    loaded = len(sources) - compile_skipped
    logger.info("%s: compiled %d YARA rule file(s) (%d skipped)", ANALYZER_ID, loaded, skipped)
    return compiled


def _extract_match_strings(match: yara.Match) -> tuple[int, str | None]:
    """Extract the first match offset and a joined matched-text snippet from a YARA match."""
    first_offset: int | None = None
    parts: list[str] = []
    for sd in match.strings or []:
        for inst in sd.instances or []:
            if first_offset is None or inst.offset < first_offset:
                first_offset = inst.offset
            matched_bytes = inst.matched_data
            if isinstance(matched_bytes, bytes):
                parts.append(matched_bytes.decode("utf-8", errors="replace"))
    matched_text = "; ".join(parts)[:200] if parts else None
    return first_offset if first_offset is not None else 0, matched_text


def _line_number_from_byte_offset(data: bytes, offset: int) -> int:
    """Return the 1-based line number for a YARA byte offset in *data*."""
    return data[:offset].count(b"\n") + 1


def _has_local_destructive_autonomy_evidence(match: yara.Match, data: bytes) -> bool:
    """Require destructive and autonomy evidence to occur in one local context.

    YARA string conditions are file-wide. Without this post-match check, a
    scoped workspace reset near the start of a long skill combines with unrelated
    prose such as "do not prompt per file" much later and becomes a false HIGH.
    Root deletion remains blocking without autonomy evidence, matching the rule's
    explicit condition.
    """
    destructive_lines: list[int] = []
    autonomy_lines: list[int] = []
    for string_match in match.strings or []:
        identifier = str(string_match.identifier)
        for instance in string_match.instances or []:
            line = _line_number_from_byte_offset(data, instance.offset)
            if identifier == "$destructive_rm_root":
                return True
            if identifier.startswith("$destructive_"):
                destructive_lines.append(line)
            elif identifier.startswith("$autonomy_"):
                autonomy_lines.append(line)

    return any(
        abs(destructive_line - autonomy_line) <= _MAX_DESTRUCTIVE_AUTONOMY_LINE_DISTANCE
        for destructive_line in destructive_lines
        for autonomy_line in autonomy_lines
    )


def _parse_meta(match: yara.Match) -> tuple[str, Severity, float, str | None]:
    """Extract rule_id, severity, confidence, and description from a YARA match's meta."""
    meta: dict[str, object] = match.meta or {}
    category = str(meta.get("category", "")).lower()
    rule_id, severity = _CATEGORY_MAP.get(category, (_DEFAULT_RULE_ID, _DEFAULT_SEVERITY))

    severity_override = str(meta.get("severity", "")).upper()
    if severity_override in Severity.__members__:
        severity = Severity[severity_override]

    try:
        confidence = float(str(meta.get("confidence", _DEFAULT_CONFIDENCE)))
    except (ValueError, TypeError):
        confidence = _DEFAULT_CONFIDENCE

    description = str(meta.get("description", "")) or None
    return rule_id, severity, confidence, description


def _build_message(rule_name: str, namespace: str, description: str | None) -> str:
    """Build a human-readable finding message from YARA match metadata."""
    msg = f"YARA rule '{rule_name}'"
    if description:
        msg += f": {description}"
    if namespace != "default":
        msg += f" [{namespace}]"
    return msg


def _match_file(rules: yara.Rules, content: str, file_path: str) -> list[AnalyzerFinding]:
    """Run compiled YARA rules against *content* and return AnalyzerFindings."""
    data = content.encode("utf-8", errors="replace")
    matches = rules.match(data=data)

    findings: list[AnalyzerFinding] = []
    for match in matches:
        if (
            match.namespace == _DESTRUCTIVE_AUTONOMY_NAMESPACE
            and match.rule == _DESTRUCTIVE_AUTONOMY_RULE
            and not _has_local_destructive_autonomy_evidence(match, data)
        ):
            logger.debug(
                "%s: ignored cross-context destructive/autonomy match in %s",
                ANALYZER_ID,
                file_path,
            )
            continue
        rule_id, severity, confidence, description = _parse_meta(match)
        first_offset, matched_text = _extract_match_strings(match)
        start_line = _line_number_from_byte_offset(data, first_offset)

        findings.append(
            AnalyzerFinding(
                rule_id=rule_id,
                message=_build_message(match.rule, match.namespace, description),
                severity=severity,
                location=Location(file=file_path, start_line=start_line),
                confidence=confidence,
                tags=[PatternCategory.YARA_MATCH.value],
                context=get_context_from_lines(content.splitlines(), start_line),
                matched_text=matched_text,
            )
        )
    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run YARA rules against all skill artifacts and return findings."""
    extra_dir_str: str | None = state.get("yara_rules_dir")
    extra_dir = Path(extra_dir_str) if extra_dir_str else None

    rules = _load_rules(extra_dir)
    if rules is None:
        logger.info("%s: 0 findings (no rules available)", ANALYZER_ID)
        return {
            "findings": [],
            "inspection_ledger": [],
            "analyzer_status_events": [
                analyzer_status_event(
                    analyzer_id=ANALYZER_ID,
                    status="unavailable",
                    reason=LedgerReason.RULES_UNAVAILABLE,
                )
            ],
        }

    components: list[str] = state.get("components") or []
    file_cache: dict[str, str] = state.get("local_file_cache") or state.get("file_cache") or {}
    findings = []
    events: list[InspectionLedgerEvent] = []

    for path in components:
        content = file_cache.get(path)
        if content is None:
            events.append(
                ledger_event(
                    analyzer_id=ANALYZER_ID,
                    outcome=LedgerOutcome.FAILED,
                    phase="static",
                    path=path,
                    reason=LedgerReason.MISSING_FILE_CACHE,
                )
            )
            continue
        if len(content) > MAX_FILE_CHARS:
            logger.debug(
                "%s: skipping %s (exceeds %d-character limit)",
                ANALYZER_ID,
                path,
                MAX_FILE_CHARS,
            )
            events.append(
                ledger_event(
                    analyzer_id=ANALYZER_ID,
                    outcome=LedgerOutcome.SKIPPED,
                    phase="static",
                    path=path,
                    reason=LedgerReason.SIZE_LIMIT,
                    observed_characters=len(content),
                    limit_characters=MAX_FILE_CHARS,
                    observed_bytes=len(content.encode("utf-8")),
                )
            )
            continue
        try:
            path_findings = [
                analyzer_finding_to_finding(af) for af in _match_file(rules, content, path)
            ]
        except Exception as exc:
            logger.warning("%s: match error on %s: %s", ANALYZER_ID, path, exc)
            events.append(
                ledger_event(
                    analyzer_id=ANALYZER_ID,
                    outcome=LedgerOutcome.FAILED,
                    phase="static",
                    path=path,
                    reason=LedgerReason.ANALYZER_RUNTIME_ERROR,
                    error_class=type(exc).__name__,
                )
            )
            continue
        findings.extend(path_findings)
        events.append(
            ledger_event(
                analyzer_id=ANALYZER_ID,
                outcome=LedgerOutcome.COMPLETED,
                phase="static",
                path=path,
                emitted_finding_ids=[finding.finding_id for finding in path_findings],
            )
        )

    logger.info("%s: %d findings", ANALYZER_ID, len(findings))
    if not events:
        status = analyzer_status_event(
            analyzer_id=ANALYZER_ID,
            status="not_applicable",
            reason=LedgerReason.NO_APPLICABLE_FILES,
        )
    else:
        status = analyzer_status_event(
            analyzer_id=ANALYZER_ID,
            status=(
                "failed"
                if any(event["outcome"] is LedgerOutcome.FAILED for event in events)
                else "degraded"
                if any(event["outcome"] is LedgerOutcome.SKIPPED for event in events)
                else "completed"
            ),
            planned_work=[
                {
                    "work_id": event["work_id"],
                    "path": event["path"],
                    "start_line": event["start_line"],
                    "end_line": event["end_line"],
                }
                for event in events
            ],
        )
    return {
        "findings": findings,
        "inspection_ledger": events,
        "analyzer_status_events": [status],
    }
