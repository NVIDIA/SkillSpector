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
import math
import os
import stat
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

import yara  # type: ignore[import-not-found]

from skillspector.constants import MAX_ANALYZABLE_FILE_BYTES
from skillspector.input_handler import (
    _FileOpenError,
    _open_regular_file_no_follow,
    _UnsafeFileError,
)
from skillspector.inspection_ledger import (
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    analyzer_status_event,
    ledger_event,
)
from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Finding, Location, Severity
from skillspector.state import (
    AnalyzerNodeResponse,
    SkillspectorState,
    transitive_remaining_seconds,
)

from .pattern_defaults import PatternCategory
from .static_runner import (
    MAX_FINDINGS_PER_ANALYZER,
    MAX_FINDINGS_PER_ARTIFACT,
    MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT,
    analyzer_finding_to_finding,
)

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
MAX_YARA_MATCH_INSTANCES_PER_RULE = 4_096
MAX_YARA_RULE_FILES = 1_024
MAX_YARA_RULE_DIRECTORY_ENTRIES = 10_000
MAX_YARA_RULE_TRAVERSAL_DEPTH = 64
MAX_YARA_RULE_FILE_BYTES = 1 * 1024 * 1024
MAX_YARA_RULE_TOTAL_BYTES = 16 * 1024 * 1024
MAX_YARA_MATCH_FINGERPRINT_BYTES_PER_FILE = MAX_ANALYZABLE_FILE_BYTES
MAX_YARA_RULE_LOAD_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class _YaraRuleResourceLimitError(Exception):
    """Sanitized resource signal for optional and built-in rule materialization."""

    reason: LedgerReason
    metrics: dict[str, int | float]


@dataclass(frozen=True, slots=True)
class _YaraRuleLoadBudget:
    """Bound active rule work while retaining the workflow's wall-clock deadline.

    Analyzer nodes run concurrently.  A CPU-heavy sibling can prevent this Python
    thread from being scheduled for several seconds, which must not consume the
    rule loader's own processing allowance.  The enclosing workflow wall-clock
    deadline remains authoritative during such scheduler contention.
    """

    active_started_at: float
    active_limit_seconds: float
    workflow_started_at: float
    workflow_limit_seconds: float


_RULE_LOAD_DEADLINE: ContextVar[_YaraRuleLoadBudget | None] = ContextVar(
    "skillspector_yara_rule_load_deadline", default=None
)


def _new_rule_load_budget(
    active_limit_seconds: float,
    *,
    workflow_limit_seconds: float,
    workflow_started_at: float | None = None,
) -> _YaraRuleLoadBudget:
    """Create one active-processing budget nested inside a wall-clock budget."""
    return _YaraRuleLoadBudget(
        active_started_at=time.thread_time(),
        active_limit_seconds=active_limit_seconds,
        workflow_started_at=(
            time.monotonic() if workflow_started_at is None else workflow_started_at
        ),
        workflow_limit_seconds=workflow_limit_seconds,
    )


def _check_rule_load_budget(budget: _YaraRuleLoadBudget) -> None:
    """Raise a sanitized signal when either rule-load deadline is exhausted."""
    workflow_elapsed = max(0.0, time.monotonic() - budget.workflow_started_at)
    if workflow_elapsed >= budget.workflow_limit_seconds:
        raise _YaraRuleResourceLimitError(
            LedgerReason.RUNTIME_LIMIT,
            {
                "observed_seconds": workflow_elapsed,
                "limit_seconds": budget.workflow_limit_seconds,
            },
        )

    active_elapsed = max(0.0, time.thread_time() - budget.active_started_at)
    if active_elapsed >= budget.active_limit_seconds:
        raise _YaraRuleResourceLimitError(
            LedgerReason.RUNTIME_LIMIT,
            {
                "observed_seconds": active_elapsed,
                "limit_seconds": budget.active_limit_seconds,
            },
        )


def _enforce_rule_load_deadline() -> None:
    """Cooperatively stop rule discovery/materialization/compilation at its deadline."""
    budget = _RULE_LOAD_DEADLINE.get()
    if budget is not None:
        _check_rule_load_budget(budget)


# Module-level cache keyed by a content hash of all rule directories.
_compiled_rules: yara.Rules | None = None
_rules_hash: str | None = None


def _collect_rule_files(*dirs: Path) -> list[Path]:
    """Collect YARA files with bounded no-follow deterministic traversal."""
    files: list[Path] = []
    seen: set[Path] = set()
    entries_seen = 0
    budget = _RULE_LOAD_DEADLINE.get() or _new_rule_load_budget(
        MAX_YARA_RULE_LOAD_SECONDS,
        workflow_limit_seconds=MAX_YARA_RULE_LOAD_SECONDS,
    )

    def check_deadline() -> None:
        _check_rule_load_budget(budget)

    suffixes = tuple(pattern.removeprefix("*") for pattern in _RULE_EXTENSIONS)
    for root in dirs:
        try:
            root_stat = root.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise _YaraRuleResourceLimitError(LedgerReason.READ_ERROR, {}) from exc
        if not stat.S_ISDIR(root_stat.st_mode):
            continue
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack:
            check_deadline()
            directory, depth = stack.pop()
            if depth > MAX_YARA_RULE_TRAVERSAL_DEPTH:
                raise _YaraRuleResourceLimitError(
                    LedgerReason.TRAVERSAL_DEPTH_LIMIT,
                    {
                        "observed_depth": depth,
                        "limit_depth": MAX_YARA_RULE_TRAVERSAL_DEPTH,
                    },
                )
            try:
                with os.scandir(directory) as scanner:
                    entries: list[os.DirEntry[str]] = []
                    for entry in scanner:
                        check_deadline()
                        entries_seen += 1
                        if entries_seen > MAX_YARA_RULE_DIRECTORY_ENTRIES:
                            raise _YaraRuleResourceLimitError(
                                LedgerReason.ARTIFACT_COUNT_LIMIT,
                                {
                                    "observed_artifacts": entries_seen,
                                    "limit_artifacts": MAX_YARA_RULE_DIRECTORY_ENTRIES,
                                },
                            )
                        entries.append(entry)
            except _YaraRuleResourceLimitError:
                raise
            except OSError as exc:
                raise _YaraRuleResourceLimitError(LedgerReason.READ_ERROR, {}) from exc

            child_directories: list[tuple[Path, int]] = []
            for entry in sorted(entries, key=lambda item: (item.name.casefold(), item.name)):
                check_deadline()
                try:
                    entry_stat = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise _YaraRuleResourceLimitError(LedgerReason.READ_ERROR, {}) from exc
                if stat.S_ISLNK(entry_stat.st_mode):
                    continue
                path = Path(entry.path)
                if stat.S_ISDIR(entry_stat.st_mode):
                    child_directories.append((path, depth + 1))
                    continue
                if not stat.S_ISREG(entry_stat.st_mode) or not entry.name.endswith(suffixes):
                    continue
                if path in seen:
                    continue
                seen.add(path)
                files.append(path)
                if len(files) > MAX_YARA_RULE_FILES:
                    raise _YaraRuleResourceLimitError(
                        LedgerReason.ARTIFACT_COUNT_LIMIT,
                        {
                            "observed_artifacts": len(files),
                            "limit_artifacts": MAX_YARA_RULE_FILES,
                        },
                    )
            stack.extend(reversed(child_directories))
    return files


def _read_rule_bytes_cache(rule_files: list[Path]) -> dict[Path, bytes]:
    """Read exact rule bytes once under per-file, aggregate, and deadline caps."""
    raw_cache: dict[Path, bytes] = {}
    total_bytes = 0
    budget = _RULE_LOAD_DEADLINE.get() or _new_rule_load_budget(
        MAX_YARA_RULE_LOAD_SECONDS,
        workflow_limit_seconds=MAX_YARA_RULE_LOAD_SECONDS,
    )
    for path in rule_files:
        _check_rule_load_budget(budget)
        try:
            with _open_regular_file_no_follow(path) as source:
                data = source.read(MAX_YARA_RULE_FILE_BYTES + 1)
        except (OSError, _FileOpenError, _UnsafeFileError) as exc:
            raise _YaraRuleResourceLimitError(LedgerReason.READ_ERROR, {}) from exc
        if len(data) > MAX_YARA_RULE_FILE_BYTES:
            raise _YaraRuleResourceLimitError(
                LedgerReason.SIZE_LIMIT,
                {
                    "observed_bytes": len(data),
                    "limit_bytes": MAX_YARA_RULE_FILE_BYTES,
                },
            )
        total_bytes += len(data)
        if total_bytes > MAX_YARA_RULE_TOTAL_BYTES:
            raise _YaraRuleResourceLimitError(
                LedgerReason.TOTAL_BYTES_LIMIT,
                {
                    "observed_bytes": total_bytes,
                    "limit_bytes": MAX_YARA_RULE_TOTAL_BYTES,
                },
            )
        raw_cache[path] = data
    return raw_cache


def _content_hash(rule_files: list[Path], raw_cache: dict[Path, bytes] | None = None) -> str:
    """Hash over rule file paths and content for cache invalidation.

    Uses actual file content (not just size) so that edits which preserve
    file length still invalidate the cache.
    """
    if raw_cache is None:
        raw_cache = _read_rule_bytes_cache(rule_files)
    h = hashlib.sha256()
    for p in rule_files:
        _enforce_rule_load_deadline()
        h.update(str(p).encode())
        h.update(raw_cache[p])
    return h.hexdigest()


def _rule_namespace(rule_file: Path) -> str:
    """Derive a stable namespace from a rule file name."""
    for suffix in _ENCODED_RULE_SUFFIXES:
        if rule_file.name.endswith(suffix):
            return rule_file.name[: -len(suffix)]
    return rule_file.stem


def _read_rule_source(rule_file: Path, data: bytes | None = None) -> str:
    """Read a YARA rule source, decoding embedded packaged rules when needed."""
    if data is None:
        data = _read_rule_bytes_cache([rule_file])[rule_file]
    if not rule_file.name.endswith(_ENCODED_RULE_SUFFIXES):
        return data.decode("utf-8")

    encoded_source = data.decode("utf-8")
    return base64.b64decode("".join(encoded_source.split())).decode("utf-8")


def _build_namespace_map(
    rule_files: list[Path],
    temp_dir: Path | None = None,
    *,
    raw_cache: dict[Path, bytes] | None = None,
) -> tuple[dict[str, str], int]:
    """Build a {namespace: source} dict and count malformed rule files."""
    del temp_dir
    sources: dict[str, str] = {}
    skipped = 0
    if raw_cache is None:
        raw_cache = _read_rule_bytes_cache(rule_files)
    for rf in rule_files:
        _enforce_rule_load_deadline()
        ns = _rule_namespace(rf)
        if ns in sources:
            ns = f"{rf.parent.name}/{ns}"
        try:
            sources[ns] = _read_rule_source(rf, raw_cache[rf])
        except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
            skipped += 1
            logger.debug("%s: skipping malformed encoded rule %s: %s", ANALYZER_ID, rf, exc)
    return sources, skipped


def _compile_rules(sources: dict[str, str]) -> tuple[yara.Rules | None, int]:
    """Compile YARA rules from a namespace map. Falls back to per-source compilation on error.

    Returns (compiled_rules, skipped_count).
    """
    _enforce_rule_load_deadline()
    try:
        compiled = yara.compile(sources=sources)
        _enforce_rule_load_deadline()
        return compiled, 0
    except yara.SyntaxError:
        pass

    logger.debug("%s: bulk compile failed, falling back to per-source compilation", ANALYZER_ID)
    good: dict[str, str] = {}
    skipped = 0
    for ns, source in sources.items():
        _enforce_rule_load_deadline()
        try:
            yara.compile(source=source)
            good[ns] = source
        except (yara.SyntaxError, yara.Error) as exc:
            skipped += 1
            logger.debug("%s: skipping %s: %s", ANALYZER_ID, ns, exc)

    _enforce_rule_load_deadline()
    compiled = yara.compile(sources=good) if good else None
    _enforce_rule_load_deadline()
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

    raw_cache = _read_rule_bytes_cache(rule_files)
    current_hash = _content_hash(rule_files, raw_cache)
    if _compiled_rules is not None and _rules_hash == current_hash:
        return _compiled_rules

    sources, materialize_skipped = _build_namespace_map(rule_files, raw_cache=raw_cache)
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


def _bounded_match_instances(
    match: yara.Match,
) -> tuple[list[tuple[str, object]], bool]:
    """Materialize only a bounded prefix of one rule's string instances."""
    instances: list[tuple[str, object]] = []
    for string_match in match.strings or []:
        identifier = str(string_match.identifier)
        for instance in string_match.instances or []:
            if len(instances) >= MAX_YARA_MATCH_INSTANCES_PER_RULE:
                return instances, True
            instances.append((identifier, instance))
    return instances, False


def _extract_match_strings(instances: list[tuple[str, object]]) -> tuple[int, str | None]:
    """Extract the first match offset and a joined matched-text snippet from a YARA match."""
    first_offset: int | None = None
    parts: list[str] = []
    output_characters = 0
    for _identifier, instance in instances:
        offset = int(getattr(instance, "offset", 0))
        if first_offset is None or offset < first_offset:
            first_offset = offset
        matched_bytes = getattr(instance, "matched_data", None)
        if isinstance(matched_bytes, bytes) and output_characters < 200:
            # Four source bytes per remaining output character is enough for
            # valid UTF-8 and keeps a malicious wide YARA match bounded before
            # decoding. Replacement decoding is sliced again below.
            remaining = 200 - output_characters
            part = matched_bytes[: remaining * 4].decode("utf-8", errors="replace")[:remaining]
            parts.append(part)
            output_characters += len(part)
    matched_text = "; ".join(parts)[:200] if parts else None
    return first_offset if first_offset is not None else 0, matched_text


class _YaraFingerprintLimitError(RuntimeError):
    """Signal that complete-match hashing exhausted its per-file byte budget."""

    def __init__(self, observed_bytes: int, limit_bytes: int) -> None:
        super().__init__("YARA full-match fingerprint byte budget exceeded")
        self.observed_bytes = observed_bytes
        self.limit_bytes = limit_bytes


@dataclass
class _YaraFingerprintBudget:
    """Bound complete-match hashing and cache repeated source spans."""

    limit_bytes: int
    hashed_bytes: int = 0
    span_digests: dict[tuple[int, int], bytes] = field(default_factory=dict)

    def digest(
        self,
        data: bytes,
        offset: int,
        matched_length: int,
        matched_data: bytes,
    ) -> bytes:
        end = offset + matched_length
        if offset >= 0 and end <= len(data):
            span = (offset, matched_length)
            cached = self.span_digests.get(span)
            if cached is not None:
                return cached
            payload = memoryview(data)[offset:end]
        else:
            span = None
            payload = memoryview(matched_data)

        observed_bytes = self.hashed_bytes + len(payload)
        if observed_bytes > self.limit_bytes:
            raise _YaraFingerprintLimitError(observed_bytes, self.limit_bytes)
        payload_digest = hashlib.sha256(payload).digest()
        self.hashed_bytes = observed_bytes
        if span is not None:
            self.span_digests[span] = payload_digest
        return payload_digest


def _match_instances_fingerprint(
    rule_id: str,
    instances: list[tuple[str, object]],
    data: bytes,
    budget: _YaraFingerprintBudget,
) -> str | None:
    """Hash complete raw YARA matches without retaining them in finding state."""
    records: list[tuple[bytes, int, bytes]] = []
    for identifier, instance in instances:
        matched_data = getattr(instance, "matched_data", None)
        if not isinstance(matched_data, bytes):
            continue
        offset = int(getattr(instance, "offset", 0))
        matched_length = int(getattr(instance, "matched_length", len(matched_data)))
        if matched_length < 0:
            matched_length = len(matched_data)
        records.append(
            (
                identifier.encode("utf-8", errors="surrogatepass"),
                matched_length,
                budget.digest(data, offset, matched_length, matched_data),
            )
        )
    if not records:
        return None

    digest = hashlib.sha256()
    digest.update(b"skillspector-yara-match-v1\x00")

    def update_framed(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    update_framed(rule_id.encode("utf-8", errors="surrogatepass"))
    for identifier_bytes, matched_length, payload_digest in sorted(records):
        update_framed(identifier_bytes)
        digest.update(matched_length.to_bytes(8, "big", signed=False))
        update_framed(payload_digest)
    return digest.hexdigest()


def _line_number_from_byte_offset(data: bytes, offset: int) -> int:
    """Return the 1-based line number for a YARA byte offset in *data*."""
    return data[:offset].count(b"\n") + 1


def _cached_line_number(data: bytes, offset: int, cache: dict[int, int]) -> int:
    """Return a cached line number without allocating a byte prefix."""
    if offset not in cache:
        cache[offset] = data.count(b"\n", 0, offset) + 1
    return cache[offset]


def _bounded_context(data: bytes, offset: int) -> str:
    """Render a small local byte window around a YARA match."""
    start = max(0, offset - 400)
    end = min(len(data), offset + 600)
    return data[start:end].decode("utf-8", errors="replace")[:1000]


def _has_local_destructive_autonomy_evidence(
    instances: list[tuple[str, object]],
    data: bytes,
    line_cache: dict[int, int],
) -> bool:
    """Require destructive and autonomy evidence to occur in one local context.

    YARA string conditions are file-wide. Without this post-match check, a
    scoped workspace reset near the start of a long skill combines with unrelated
    prose such as "do not prompt per file" much later and becomes a false HIGH.
    Root deletion remains blocking without autonomy evidence, matching the rule's
    explicit condition.
    """
    destructive_lines: list[int] = []
    autonomy_lines: list[int] = []
    for identifier, instance in instances:
        offset = int(getattr(instance, "offset", 0))
        line = _cached_line_number(data, offset, line_cache)
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


@dataclass(frozen=True)
class _YaraFileResult:
    findings: list[AnalyzerFinding]
    reason: LedgerReason | None = None
    metrics: dict[str, int | float] | None = None


def _match_file(
    rules: yara.Rules,
    data: bytes | str,
    file_path: str,
    content: str | None = None,
    *,
    max_findings: int = MAX_FINDINGS_PER_ARTIFACT,
    timeout_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> _YaraFileResult:
    """Run compiled YARA rules against canonical raw bytes."""
    if isinstance(data, str):
        content = data if content is None else content
        data = data.encode("utf-8", errors="replace")
    if content is None:
        content = data.decode("utf-8", errors="replace")
    started_at = clock()
    runtime_limit = MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT
    if timeout_seconds is not None:
        runtime_limit = min(runtime_limit, max(0.0, timeout_seconds))
    # yara-python accepts only a positive whole-second engine timeout. Do not
    # begin work that cannot be contained within the shared remaining budget.
    if runtime_limit < 1.0:
        return _YaraFileResult(
            findings=[],
            reason=LedgerReason.RUNTIME_LIMIT,
            metrics={"observed_seconds": 0.0, "limit_seconds": runtime_limit},
        )
    deadline = started_at + runtime_limit
    observed_matches = 0

    def _match_callback(_match_data: dict[str, object]) -> int:
        nonlocal observed_matches
        observed_matches += 1
        return int(
            yara.CALLBACK_ABORT if observed_matches > max_findings else yara.CALLBACK_CONTINUE
        )

    matches = rules.match(
        data=data,
        callback=_match_callback,
        which_callbacks=yara.CALLBACK_MATCHES,
        # Round down so the engine timeout never exceeds min(shared, 30s).
        timeout=max(1, math.floor(runtime_limit)),
        # YARA still evaluates full rule conditions, but stops retaining every
        # repeated string instance after the condition is decided. Without
        # this, one-byte custom rules can materialize millions of instances.
        fast=True,
    )

    findings: list[AnalyzerFinding] = []
    instance_limited = False
    line_cache: dict[int, int] = {}
    fingerprint_budget = _YaraFingerprintBudget(max(0, MAX_YARA_MATCH_FINGERPRINT_BYTES_PER_FILE))
    for match_index, match in enumerate(matches):
        now = clock()
        if now >= deadline:
            return _YaraFileResult(
                findings=findings,
                reason=LedgerReason.RUNTIME_LIMIT,
                metrics={
                    "observed_seconds": max(0.0, now - started_at),
                    "limit_seconds": runtime_limit,
                },
            )
        if match_index >= max_findings:
            observed_matches = max(observed_matches, match_index + 1)
            break
        instances, limited = _bounded_match_instances(match)
        instance_limited = instance_limited or limited
        if (
            match.namespace == _DESTRUCTIVE_AUTONOMY_NAMESPACE
            and match.rule == _DESTRUCTIVE_AUTONOMY_RULE
            # A bounded-prefix hit cannot safely justify suppression. Retain
            # the high-severity rule and mark this work item partial instead.
            and not limited
            and not _has_local_destructive_autonomy_evidence(instances, data, line_cache)
        ):
            logger.debug(
                "%s: ignored cross-context destructive/autonomy match in %s",
                ANALYZER_ID,
                file_path,
            )
            continue
        rule_id, severity, confidence, description = _parse_meta(match)
        first_offset, matched_text = _extract_match_strings(instances)
        try:
            match_fingerprint = _match_instances_fingerprint(
                rule_id,
                instances,
                data,
                fingerprint_budget,
            )
        except _YaraFingerprintLimitError as exc:
            return _YaraFileResult(
                findings=findings,
                reason=LedgerReason.SIZE_LIMIT,
                metrics={
                    "observed_bytes": exc.observed_bytes,
                    "limit_bytes": exc.limit_bytes,
                },
            )
        start_line = _cached_line_number(data, first_offset, line_cache)

        findings.append(
            AnalyzerFinding(
                rule_id=rule_id,
                message=_build_message(match.rule, match.namespace, description),
                severity=severity,
                location=Location(file=file_path, start_line=start_line),
                confidence=confidence,
                tags=[PatternCategory.YARA_MATCH.value],
                context=_bounded_context(data, first_offset),
                matched_text=matched_text,
                match_fingerprint=match_fingerprint,
            )
        )
    finished_at = clock()
    if finished_at >= deadline:
        return _YaraFileResult(
            findings=findings,
            reason=LedgerReason.RUNTIME_LIMIT,
            metrics={
                "observed_seconds": max(0.0, finished_at - started_at),
                "limit_seconds": runtime_limit,
            },
        )
    if observed_matches > max_findings:
        return _YaraFileResult(
            findings=findings,
            reason=LedgerReason.OUTPUT_LIMIT,
            metrics={
                "observed_findings": observed_matches,
                "limit_findings": max_findings,
            },
        )
    if instance_limited:
        return _YaraFileResult(
            findings=findings,
            reason=LedgerReason.OUTPUT_LIMIT,
            metrics={
                "observed_records": MAX_YARA_MATCH_INSTANCES_PER_RULE + 1,
                "limit_records": MAX_YARA_MATCH_INSTANCES_PER_RULE,
            },
        )
    return _YaraFileResult(findings=findings)


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run YARA rules against all skill artifacts and return findings."""
    extra_dir_str: str | None = state.get("yara_rules_dir")
    extra_dir = Path(extra_dir_str) if extra_dir_str else None
    components: list[str] = state.get("components") or []

    def _rule_limit_response(
        reason: LedgerReason,
        metrics: dict[str, int | float],
    ) -> AnalyzerNodeResponse:
        limit_events = [
            ledger_event(
                analyzer_id=ANALYZER_ID,
                outcome=LedgerOutcome.PARTIAL,
                phase="static",
                path=path,
                reason=reason,
                observed_bytes=(
                    int(metrics["observed_bytes"]) if "observed_bytes" in metrics else None
                ),
                limit_bytes=(int(metrics["limit_bytes"]) if "limit_bytes" in metrics else None),
                observed_artifacts=(
                    int(metrics["observed_artifacts"]) if "observed_artifacts" in metrics else None
                ),
                limit_artifacts=(
                    int(metrics["limit_artifacts"]) if "limit_artifacts" in metrics else None
                ),
                observed_depth=(
                    int(metrics["observed_depth"]) if "observed_depth" in metrics else None
                ),
                limit_depth=(int(metrics["limit_depth"]) if "limit_depth" in metrics else None),
                observed_seconds=(
                    float(metrics["observed_seconds"]) if "observed_seconds" in metrics else None
                ),
                limit_seconds=(
                    float(metrics["limit_seconds"]) if "limit_seconds" in metrics else None
                ),
            )
            for path in components
        ]
        return {
            "findings": [],
            "inspection_ledger": limit_events,
            "analyzer_status_events": [
                analyzer_status_event(
                    analyzer_id=ANALYZER_ID,
                    status="degraded",
                    reason=reason,
                    planned_work=[
                        {
                            "work_id": event["work_id"],
                            "path": event["path"],
                            "start_line": event["start_line"],
                            "end_line": event["end_line"],
                        }
                        for event in limit_events
                    ],
                )
            ],
        }

    workflow_load_started_at = time.monotonic()
    initial_remaining = transitive_remaining_seconds(state)
    if initial_remaining is not None and initial_remaining < 1.0:
        return _rule_limit_response(
            LedgerReason.RUNTIME_LIMIT,
            {
                "observed_seconds": 0.0,
                "limit_seconds": max(0.0, initial_remaining),
            },
        )

    rule_load_seconds = min(
        MAX_YARA_RULE_LOAD_SECONDS,
        max(0.0, initial_remaining)
        if initial_remaining is not None
        else MAX_YARA_RULE_LOAD_SECONDS,
    )
    workflow_load_seconds = (
        max(0.0, initial_remaining) if initial_remaining is not None else MAX_YARA_RULE_LOAD_SECONDS
    )
    load_budget = _new_rule_load_budget(
        rule_load_seconds,
        workflow_limit_seconds=workflow_load_seconds,
        workflow_started_at=workflow_load_started_at,
    )
    deadline_token = _RULE_LOAD_DEADLINE.set(load_budget)
    try:
        rules = _load_rules(extra_dir)
    except _YaraRuleResourceLimitError as exc:
        return _rule_limit_response(exc.reason, dict(exc.metrics))
    finally:
        _RULE_LOAD_DEADLINE.reset(deadline_token)
    remaining_after_load = transitive_remaining_seconds(state)
    if remaining_after_load is not None and remaining_after_load < 1.0:
        return _rule_limit_response(
            LedgerReason.RUNTIME_LIMIT,
            {
                "observed_seconds": max(0.0, time.monotonic() - load_budget.workflow_started_at),
                "limit_seconds": workflow_load_seconds,
            },
        )
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

    file_cache: dict[str, str] = state.get("local_file_cache") or state.get("file_cache") or {}
    raw_file_cache: dict[str, bytes] = state.get("raw_file_cache") or {}
    findings: list[Finding] = []
    events: list[InspectionLedgerEvent] = []

    for component_index, path in enumerate(components):
        shared_remaining = transitive_remaining_seconds(state)
        if shared_remaining is not None and shared_remaining < 1.0:
            events.extend(
                ledger_event(
                    analyzer_id=ANALYZER_ID,
                    outcome=LedgerOutcome.PARTIAL,
                    phase="static",
                    path=remaining_path,
                    reason=LedgerReason.RUNTIME_LIMIT,
                    observed_seconds=0.0,
                    limit_seconds=max(0.0, shared_remaining),
                )
                for remaining_path in components[component_index:]
            )
            break
        content = file_cache.get(path)
        data = raw_file_cache.get(path)
        if data is None and content is not None:
            data = content.encode("utf-8", errors="replace")
        if data is None:
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
        remaining = MAX_FINDINGS_PER_ANALYZER - len(findings)
        if remaining <= 0:
            events.append(
                ledger_event(
                    analyzer_id=ANALYZER_ID,
                    outcome=LedgerOutcome.PARTIAL,
                    phase="static",
                    path=path,
                    reason=LedgerReason.OUTPUT_LIMIT,
                    observed_findings=len(findings) + 1,
                    limit_findings=MAX_FINDINGS_PER_ANALYZER,
                )
            )
            continue
        try:
            matched = _match_file(
                rules,
                data,
                path,
                content,
                max_findings=min(MAX_FINDINGS_PER_ARTIFACT, remaining),
                timeout_seconds=shared_remaining,
                clock=time.monotonic,
            )
            path_findings = [analyzer_finding_to_finding(af) for af in matched.findings]
        except yara.TimeoutError:
            runtime_limit = MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT
            if shared_remaining is not None:
                runtime_limit = min(runtime_limit, max(0.0, shared_remaining))
            events.append(
                ledger_event(
                    analyzer_id=ANALYZER_ID,
                    outcome=LedgerOutcome.PARTIAL,
                    phase="static",
                    path=path,
                    reason=LedgerReason.RUNTIME_LIMIT,
                    observed_seconds=runtime_limit,
                    limit_seconds=runtime_limit,
                )
            )
            continue
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
        metrics = matched.metrics or {}
        events.append(
            ledger_event(
                analyzer_id=ANALYZER_ID,
                outcome=(
                    LedgerOutcome.PARTIAL if matched.reason is not None else LedgerOutcome.COMPLETED
                ),
                phase="static",
                path=path,
                reason=matched.reason,
                emitted_finding_ids=[finding.finding_id for finding in path_findings],
                observed_findings=(
                    int(metrics["observed_findings"]) if "observed_findings" in metrics else None
                ),
                limit_findings=(
                    int(metrics["limit_findings"]) if "limit_findings" in metrics else None
                ),
                observed_records=(
                    int(metrics["observed_records"]) if "observed_records" in metrics else None
                ),
                limit_records=(
                    int(metrics["limit_records"]) if "limit_records" in metrics else None
                ),
                observed_seconds=(
                    float(metrics["observed_seconds"]) if "observed_seconds" in metrics else None
                ),
                limit_seconds=(
                    float(metrics["limit_seconds"]) if "limit_seconds" in metrics else None
                ),
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
                if any(
                    event["outcome"] in {LedgerOutcome.PARTIAL, LedgerOutcome.SKIPPED}
                    for event in events
                )
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
