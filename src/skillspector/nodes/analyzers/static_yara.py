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

import hashlib
from pathlib import Path

import yara

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from .common import get_context, get_line_number
from .pattern_defaults import PatternCategory
from .static_runner import MAX_FILE_BYTES, analyzer_finding_to_finding

ANALYZER_ID = "static_yara"
logger = get_logger(__name__)

_BUILTIN_RULES_DIR = Path(__file__).resolve().parent.parent.parent / "yara_rules"

_RULE_EXTENSIONS = ("*.yar", "*.yara")

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

# Module-level cache keyed by a content hash of all rule directories.
_compiled_rules: yara.Rules | None = None
_rules_hash: str | None = None


def _collect_rule_files(*dirs: Path) -> list[Path]:
    """Collect all YARA rule files under one or more directories, sorted for determinism."""
    files: set[Path] = set()
    for d in dirs:
        if not d.is_dir():
            continue
        for ext in _RULE_EXTENSIONS:
            files.update(d.rglob(ext))
    return sorted(files)


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


def _build_namespace_map(rule_files: list[Path]) -> dict[str, str]:
    """Build a {namespace: filepath} dict from rule files, deduplicating namespace names."""
    filepaths: dict[str, str] = {}
    for rf in rule_files:
        ns = rf.stem
        if ns in filepaths:
            ns = f"{rf.parent.name}/{rf.stem}"
        filepaths[ns] = str(rf)
    return filepaths


def _compile_rules(filepaths: dict[str, str]) -> tuple[yara.Rules | None, int]:
    """Compile YARA rules from a namespace map. Falls back to per-file compilation on error.

    Returns (compiled_rules, skipped_count).
    """
    try:
        return yara.compile(filepaths=filepaths), 0
    except yara.SyntaxError:
        pass

    logger.debug("%s: bulk compile failed, falling back to per-file compilation", ANALYZER_ID)
    good: dict[str, str] = {}
    skipped = 0
    for ns, fp in filepaths.items():
        try:
            yara.compile(filepath=fp)
            good[ns] = fp
        except (yara.SyntaxError, yara.Error) as exc:
            skipped += 1
            logger.debug("%s: skipping %s: %s", ANALYZER_ID, fp, exc)

    compiled = yara.compile(filepaths=good) if good else None
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

    filepaths = _build_namespace_map(rule_files)
    compiled, skipped = _compile_rules(filepaths)

    if compiled is None:
        logger.warning("%s: failed to compile any YARA rules", ANALYZER_ID)
        return None

    _compiled_rules = compiled
    _rules_hash = current_hash
    loaded = len(filepaths) - skipped
    logger.info("%s: compiled %d YARA rule file(s) (%d skipped)", ANALYZER_ID, loaded, skipped)
    return compiled


def _extract_match_strings(match: yara.Match) -> tuple[int, str | None]:
    """Extract the first match offset and a joined matched-text snippet from a YARA match."""
    first_offset = 0
    parts: list[str] = []
    for sd in match.strings or []:
        for inst in sd.instances or []:
            if first_offset == 0:
                first_offset = inst.offset
            matched_bytes = inst.matched_data
            if isinstance(matched_bytes, bytes):
                parts.append(matched_bytes.decode("utf-8", errors="replace"))
    matched_text = "; ".join(parts)[:200] if parts else None
    return first_offset, matched_text


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
    try:
        matches = rules.match(data=data)
    except Exception as exc:
        logger.debug("%s: match error on %s: %s", ANALYZER_ID, file_path, exc)
        return []

    findings: list[AnalyzerFinding] = []
    for match in matches:
        rule_id, severity, confidence, description = _parse_meta(match)
        first_offset, matched_text = _extract_match_strings(match)

        findings.append(
            AnalyzerFinding(
                rule_id=rule_id,
                message=_build_message(match.rule, match.namespace, description),
                severity=severity,
                location=Location(
                    file=file_path, start_line=get_line_number(content, first_offset)
                ),
                confidence=confidence,
                tags=[PatternCategory.YARA_MATCH.value],
                context=get_context(content, first_offset),
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
        return {"findings": []}

    components: list[str] = state.get("components") or []
    file_cache: dict[str, str] = state.get("file_cache") or {}
    findings = []

    for path in components:
        content = file_cache.get(path)
        if content is None:
            continue
        if len(content) > MAX_FILE_BYTES:
            logger.debug("%s: skipping %s (exceeds size limit)", ANALYZER_ID, path)
            continue
        for af in _match_file(rules, content, path):
            findings.append(analyzer_finding_to_finding(af))

    logger.info("%s: %d findings", ANALYZER_ID, len(findings))
    return {"findings": findings}
