# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed contracts and safe factories for inspection-work accounting."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from enum import StrEnum
from hashlib import sha256
from typing import Final, NotRequired, cast

from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class LedgerOutcome(StrEnum):
    """Terminal outcome of one inspection work item."""

    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    OUT_OF_SCOPE = "out_of_scope"


class LedgerRecordType(StrEnum):
    """Kind of ledger record."""

    WORK_ITEM = "work_item"
    SYSTEM = "system"
    SCOPE_BOUNDARY = "scope_boundary"


class LedgerReason(StrEnum):
    """Allowlisted reasons for omitted, skipped, or failed inspection work."""

    EXCLUDED_DIRECTORY = "excluded_directory"
    HIDDEN_FILE = "hidden_file"
    FILE_DISAPPEARED = "file_disappeared"
    NOT_REGULAR_FILE = "not_regular_file"
    STAT_ERROR = "stat_error"
    READ_ERROR = "read_error"
    MISSING_FILE_CACHE = "missing_file_cache"
    SIZE_LIMIT = "size_limit"
    BINARY_CONTENT = "binary_content"
    EVAL_DATASET = "eval_dataset"
    SYNTAX_ERROR = "syntax_error"
    LLM_BATCH_FAILED = "llm_batch_failed"
    LLM_STRUCTURED_RESPONSE_INVALID = "llm_structured_response_invalid"
    LLM_CONNECTION_RETRIES_EXHAUSTED = "llm_connection_retries_exhausted"
    LLM_RATE_LIMIT_RETRIES_EXHAUSTED = "llm_rate_limit_retries_exhausted"
    ANALYZER_RUNTIME_ERROR = "analyzer_runtime_error"
    UNACCOUNTED_WORK = "unaccounted_work"
    FINDING_ACCOUNTING_ERROR = "finding_accounting_error"
    DISABLED_BY_CONFIGURATION = "disabled_by_configuration"
    MISSING_CREDENTIALS = "missing_credentials"
    RULES_UNAVAILABLE = "rules_unavailable"
    MANIFEST_ABSENT = "manifest_absent"
    NO_APPLICABLE_FILES = "no_applicable_files"
    OMS_SIGNATURE = "oms_signature"
    BASELINE_FILE = "baseline_file"
    ARCHIVE_FORMAT_MISMATCH = "archive_format_mismatch"
    ARCHIVE_MALFORMED = "archive_malformed"
    ARCHIVE_ENCRYPTED = "archive_encrypted"
    ARCHIVE_UNSUPPORTED_COMPRESSION = "archive_unsupported_compression"
    ARCHIVE_TRUNCATED = "archive_truncated"
    ARCHIVE_UNSAFE_MEMBER_PATH = "archive_unsafe_member_path"
    ARCHIVE_AMBIGUOUS_MEMBER_PATH = "archive_ambiguous_member_path"
    ARCHIVE_LINK_MEMBER = "archive_link_member"
    ARCHIVE_DEPTH_LIMIT = "archive_depth_limit"
    ARCHIVE_MEMBER_LIMIT = "archive_member_limit"
    ARCHIVE_SIZE_LIMIT = "archive_size_limit"
    ARCHIVE_MEMBER_SIZE_LIMIT = "archive_member_size_limit"
    ARCHIVE_COMPRESSION_RATIO = "archive_compression_ratio"
    ARCHIVE_TIME_LIMIT = "archive_time_limit"


REASON_MESSAGES: Final[dict[LedgerReason, str]] = {
    LedgerReason.EXCLUDED_DIRECTORY: ("Directory tree is excluded from the configured scan scope."),
    LedgerReason.HIDDEN_FILE: "Hidden file is excluded from the configured scan scope.",
    LedgerReason.FILE_DISAPPEARED: ("Inventoried file disappeared before it could be inspected."),
    LedgerReason.NOT_REGULAR_FILE: "Inventoried path is no longer a regular file.",
    LedgerReason.STAT_ERROR: "Filesystem metadata could not be read.",
    LedgerReason.READ_ERROR: "File content could not be read.",
    LedgerReason.MISSING_FILE_CACHE: "Applicable analyzer could not obtain file content.",
    LedgerReason.SIZE_LIMIT: "File exceeds this analyzer's character limit.",
    LedgerReason.BINARY_CONTENT: "Binary content is unsupported by this analyzer.",
    LedgerReason.EVAL_DATASET: (
        "Evaluation dataset prose is excluded from static pattern analysis."
    ),
    LedgerReason.SYNTAX_ERROR: "Python source could not be parsed.",
    LedgerReason.LLM_BATCH_FAILED: "LLM analysis failed for this file range.",
    LedgerReason.LLM_STRUCTURED_RESPONSE_INVALID: (
        "LLM returned a malformed structured response after bounded retries."
    ),
    LedgerReason.LLM_CONNECTION_RETRIES_EXHAUSTED: ("LLM connection failed after bounded retries."),
    LedgerReason.LLM_RATE_LIMIT_RETRIES_EXHAUSTED: (
        "LLM provider rate limit persisted after bounded retries."
    ),
    LedgerReason.ANALYZER_RUNTIME_ERROR: ("Analyzer failed after beginning applicable work."),
    LedgerReason.UNACCOUNTED_WORK: ("Planned inspection work has no unique terminal outcome."),
    LedgerReason.FINDING_ACCOUNTING_ERROR: (
        "Finding identity could not be reconciled with completed work."
    ),
    LedgerReason.DISABLED_BY_CONFIGURATION: (
        "Analyzer was disabled by the requested configuration."
    ),
    LedgerReason.MISSING_CREDENTIALS: ("Analyzer credentials were unavailable before execution."),
    LedgerReason.RULES_UNAVAILABLE: ("Analyzer rules were unavailable before execution."),
    LedgerReason.MANIFEST_ABSENT: ("No compatible manifest was present for this analyzer."),
    LedgerReason.NO_APPLICABLE_FILES: ("No files matched this analyzer's applicability contract."),
    LedgerReason.OMS_SIGNATURE: (
        "Recognized OMS signature metadata is excluded from content analysis."
    ),
    LedgerReason.BASELINE_FILE: (
        "The explicitly selected suppression baseline is excluded from content analysis."
    ),
    LedgerReason.ARCHIVE_FORMAT_MISMATCH: (
        "Artifact bytes do not match the container format implied by its filename."
    ),
    LedgerReason.ARCHIVE_MALFORMED: "ZIP-compatible content is malformed.",
    LedgerReason.ARCHIVE_ENCRYPTED: "Archive member is encrypted and could not be inspected.",
    LedgerReason.ARCHIVE_UNSUPPORTED_COMPRESSION: (
        "Archive member uses an unsupported compression method."
    ),
    LedgerReason.ARCHIVE_TRUNCATED: "Archive member is truncated or unreadable.",
    LedgerReason.ARCHIVE_UNSAFE_MEMBER_PATH: (
        "Archive member uses an absolute, traversal, or otherwise unsafe path."
    ),
    LedgerReason.ARCHIVE_AMBIGUOUS_MEMBER_PATH: (
        "Archive member name has an ambiguous or duplicate provenance identity."
    ),
    LedgerReason.ARCHIVE_LINK_MEMBER: "Archive link member was not followed or read.",
    LedgerReason.ARCHIVE_DEPTH_LIMIT: "Nested archive depth limit was reached.",
    LedgerReason.ARCHIVE_MEMBER_LIMIT: "Cumulative archive member limit was reached.",
    LedgerReason.ARCHIVE_SIZE_LIMIT: "Cumulative archive content size limit was reached.",
    LedgerReason.ARCHIVE_MEMBER_SIZE_LIMIT: "Archive member exceeds the per-file analysis limit.",
    LedgerReason.ARCHIVE_COMPRESSION_RATIO: (
        "Archive member exceeds the permitted compression ratio."
    ),
    LedgerReason.ARCHIVE_TIME_LIMIT: "Cumulative archive inspection time limit was reached.",
}


def outcome_for_llm_batch_failure(reason: LedgerReason) -> LedgerOutcome:
    """Return the terminal ledger outcome for an exhausted LLM batch."""
    return (
        LedgerOutcome.SKIPPED
        if reason is LedgerReason.LLM_STRUCTURED_RESPONSE_INVALID
        else LedgerOutcome.FAILED
    )


class PlannedWorkTarget(TypedDict):
    """One analyzer work item expected to have a terminal ledger row."""

    work_id: str
    path: str
    start_line: int | None
    end_line: int | None


class InspectionLedgerEvent(TypedDict):
    """Internal terminal evidence for one work item or scope boundary."""

    work_id: str
    record_type: LedgerRecordType
    outcome: LedgerOutcome
    phase: str
    path: str
    start_line: int | None
    end_line: int | None
    input_finding_ids: list[str]
    emitted_finding_ids: list[str]
    analyzer_id: NotRequired[str]
    reason_code: NotRequired[LedgerReason]
    message: NotRequired[str]
    error_class: NotRequired[str]
    stage: NotRequired[str]
    observed_characters: NotRequired[int]
    limit_characters: NotRequired[int]
    observed_bytes: NotRequired[int]
    limit_bytes: NotRequired[int]


class AnalyzerStatusEvent(TypedDict):
    """Run-level analyzer status and its internal planned work targets."""

    analyzer_id: str
    status: str
    planned_work: list[PlannedWorkTarget]
    reason_code: NotRequired[LedgerReason]
    message: NotRequired[str]


class InspectionLedgerException(TypedDict):
    """Public exceptional projection derived from an internal ledger event."""

    outcome: LedgerOutcome
    phase: str
    reason_code: LedgerReason
    message: str
    path: str
    start_line: int | None
    end_line: int | None
    error_class: NotRequired[str]
    analyzers: NotRequired[list[str]]
    fatal: NotRequired[bool]


class AnalysisCompleteness(TypedDict):
    """Public inspection-completeness projection derived during finalization."""

    total_components: int
    scanned_components: int
    coverage_percent: float
    is_complete: bool
    execution_successful: bool
    fully_inspected_files: int
    partially_inspected_files: int
    entirely_uninspected_files: int
    ledger_exceptions: list[InspectionLedgerException]
    scope_exclusions: list[InspectionLedgerException]
    analyzer_statuses: list[dict[str, object]]
    limitations: NotRequired[list[str]]
    findings_before_filtering: NotRequired[int]
    findings_after_filtering: NotRequired[int]


def _normalize_relative_path(path: str, *, scope_boundary: bool = False) -> str:
    """Return a normalized report-safe relative POSIX path."""
    raw_path = path.replace("\\", "/")
    if not raw_path or raw_path.startswith("/") or raw_path.startswith("//"):
        raise ValueError("path must be a relative POSIX path")
    if len(raw_path) >= 2 and raw_path[1] == ":":
        raise ValueError("path must be a relative POSIX path")

    parts = raw_path.split("/")
    if any(part == ".." for part in parts):
        raise ValueError("path must be a relative POSIX path without parent traversal")
    normalized_parts = [part for part in parts if part not in ("", ".")]
    if not normalized_parts:
        raise ValueError("path must be a relative POSIX path")

    normalized = "/".join(normalized_parts)
    return f"{normalized}/" if scope_boundary else normalized


def _deduplicate_ids(finding_ids: Iterable[str]) -> list[str]:
    """Deduplicate finding IDs while retaining their first-seen order."""
    unique_ids: list[str] = []
    seen: set[str] = set()
    for finding_id in finding_ids:
        if finding_id not in seen:
            unique_ids.append(finding_id)
            seen.add(finding_id)
    return unique_ids


def _validate_range(start_line: int | None, end_line: int | None) -> None:
    """Reject incomplete or invalid source ranges."""
    if (start_line is None) != (end_line is None):
        raise ValueError("start_line and end_line must both be set or both be None")
    if start_line is not None:
        if end_line is None or start_line < 1 or end_line < start_line:
            raise ValueError("line ranges must be positive and inclusive")


def inspection_work_id(
    analyzer_id: str,
    path: str,
    start_line: int | None,
    end_line: int | None,
) -> str:
    """Build a deterministic ID for one analyzer/path/range work item."""
    _validate_range(start_line, end_line)
    normalized_path = _normalize_relative_path(path)
    canonical = "\x1f".join((analyzer_id, normalized_path, str(start_line), str(end_line)))
    return f"work-{sha256(canonical.encode('utf-8')).hexdigest()}"


def _is_meta_phase(phase: str) -> bool:
    """Return whether a row tracks meta-analysis finding lineage."""
    return phase == "meta"


def ledger_event(
    *,
    outcome: LedgerOutcome,
    phase: str,
    path: str,
    analyzer_id: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    record_type: LedgerRecordType = LedgerRecordType.WORK_ITEM,
    reason: LedgerReason | None = None,
    input_finding_ids: Iterable[str] = (),
    emitted_finding_ids: Iterable[str] = (),
    error_class: str | None = None,
    stage: str | None = None,
    observed_characters: int | None = None,
    limit_characters: int | None = None,
    observed_bytes: int | None = None,
    limit_bytes: int | None = None,
) -> InspectionLedgerEvent:
    """Create one validated terminal ledger record without sensitive payloads."""
    _validate_range(start_line, end_line)
    normalized_path = _normalize_relative_path(
        path,
        scope_boundary=(
            record_type is LedgerRecordType.SCOPE_BOUNDARY and path.endswith(("/", "\\"))
        ),
    )
    input_ids = _deduplicate_ids(input_finding_ids)
    emitted_ids = _deduplicate_ids(emitted_finding_ids)
    is_meta = _is_meta_phase(phase)

    if outcome is LedgerOutcome.COMPLETED:
        if reason is not None:
            raise ValueError("completed ledger events cannot include a reason")
    elif reason is None:
        raise ValueError("non-completed ledger events require a reason")

    if not is_meta:
        if input_ids:
            raise ValueError("producer ledger events cannot consume findings")
        if outcome is not LedgerOutcome.COMPLETED and emitted_ids:
            raise ValueError("non-completed producers cannot reference findings")
    elif outcome is LedgerOutcome.COMPLETED and not set(emitted_ids).issubset(input_ids):
        raise ValueError("completed meta events must emit a subset of input findings")
    elif outcome in (LedgerOutcome.FAILED, LedgerOutcome.SKIPPED):
        if emitted_ids != input_ids:
            raise ValueError("failed or skipped meta events must pass every input finding through")
    elif outcome is not LedgerOutcome.COMPLETED:
        if input_ids or emitted_ids:
            raise ValueError("non-completed meta events cannot reference findings")

    work_identity = analyzer_id or f"{record_type.value}:{phase}"
    event: InspectionLedgerEvent = {
        "work_id": inspection_work_id(work_identity, normalized_path, start_line, end_line),
        "record_type": record_type,
        "outcome": outcome,
        "phase": phase,
        "path": normalized_path,
        "start_line": start_line,
        "end_line": end_line,
        "input_finding_ids": input_ids,
        "emitted_finding_ids": emitted_ids,
    }
    if analyzer_id is not None:
        event["analyzer_id"] = analyzer_id
    if reason is not None:
        event["reason_code"] = reason
        event["message"] = REASON_MESSAGES[reason]
    if error_class is not None:
        event["error_class"] = error_class
    if stage is not None:
        event["stage"] = stage
    if observed_characters is not None:
        event["observed_characters"] = observed_characters
    if limit_characters is not None:
        event["limit_characters"] = limit_characters
    if observed_bytes is not None:
        event["observed_bytes"] = observed_bytes
    if limit_bytes is not None:
        event["limit_bytes"] = limit_bytes
    return event


def analyzer_status_event(
    *,
    analyzer_id: str,
    status: str,
    planned_work: Iterable[PlannedWorkTarget] = (),
    reason: LedgerReason | None = None,
) -> AnalyzerStatusEvent:
    """Create a run-level analyzer status with normalized expected-work targets."""
    normalized_work: list[PlannedWorkTarget] = []
    for target in planned_work:
        start_line = target["start_line"]
        end_line = target["end_line"]
        _validate_range(start_line, end_line)
        normalized_work.append(
            {
                "work_id": target["work_id"],
                "path": _normalize_relative_path(target["path"]),
                "start_line": start_line,
                "end_line": end_line,
            }
        )

    event: AnalyzerStatusEvent = {
        "analyzer_id": analyzer_id,
        "status": status,
        "planned_work": normalized_work,
    }
    if reason is not None:
        event["reason_code"] = reason
        event["message"] = REASON_MESSAGES[reason]
    return event


def analyzer_status_for_events(
    analyzer_id: str, events: Iterable[InspectionLedgerEvent]
) -> AnalyzerStatusEvent:
    """Summarize an analyzer's terminal work without exposing event payloads."""
    terminal_events = list(events)
    if not terminal_events:
        return analyzer_status_event(
            analyzer_id=analyzer_id,
            status="not_applicable",
            reason=LedgerReason.NO_APPLICABLE_FILES,
        )

    outcomes = {event["outcome"] for event in terminal_events}
    status = (
        "failed"
        if LedgerOutcome.FAILED in outcomes
        else "degraded"
        if LedgerOutcome.SKIPPED in outcomes
        else "completed"
    )
    return analyzer_status_event(
        analyzer_id=analyzer_id,
        status=status,
        planned_work=[
            {
                "work_id": event["work_id"],
                "path": event["path"],
                "start_line": event["start_line"],
                "end_line": event["end_line"],
            }
            for event in terminal_events
        ],
    )


def _reason(value: object, fallback: LedgerReason) -> LedgerReason:
    """Return an allowlisted reason code without trusting untyped graph state."""
    try:
        return LedgerReason(str(value))
    except ValueError:
        return fallback


def _safe_path(path: object, components: list[str]) -> str:
    """Choose a report-safe path for a synthetic finalization exception."""
    if isinstance(path, str):
        try:
            return _normalize_relative_path(path)
        except ValueError:
            pass
    if components:
        return components[0]
    return "SKILL.md"


def _exception(
    *,
    outcome: LedgerOutcome,
    phase: str,
    reason: LedgerReason,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    error_class: str | None = None,
    analyzers: Iterable[str] = (),
    fatal: bool,
) -> InspectionLedgerException:
    """Build the public, safe projection of one exceptional ledger fact."""
    exception: InspectionLedgerException = {
        "outcome": outcome,
        "phase": phase,
        "reason_code": reason,
        "message": REASON_MESSAGES[reason],
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "fatal": fatal,
    }
    analyzer_ids = sorted({analyzer for analyzer in analyzers if analyzer})
    if analyzer_ids:
        exception["analyzers"] = analyzer_ids
    if error_class:
        exception["error_class"] = error_class
    return exception


def _exception_from_event(
    event: InspectionLedgerEvent, *, fatal: bool
) -> InspectionLedgerException:
    """Project a non-completed internal event without exposing internal IDs."""
    outcome = event["outcome"]
    fallback = (
        LedgerReason.UNACCOUNTED_WORK
        if outcome == LedgerOutcome.FAILED
        else LedgerReason.NO_APPLICABLE_FILES
    )
    return _exception(
        outcome=outcome,
        phase=str(event["phase"]),
        reason=_reason(event.get("reason_code"), fallback),
        path=str(event["path"]),
        start_line=event.get("start_line"),
        end_line=event.get("end_line"),
        error_class=event.get("error_class"),
        analyzers=[str(event.get("analyzer_id", ""))],
        fatal=fatal,
    )


def _merge_exception_projection(
    exceptions: Iterable[InspectionLedgerException],
) -> list[InspectionLedgerException]:
    """Group duplicate public rows while retaining all contributing analyzers."""
    grouped: dict[tuple[object, ...], InspectionLedgerException] = {}
    for exception in exceptions:
        key = (
            exception["outcome"],
            exception["phase"],
            exception["reason_code"],
            exception["message"],
            exception["path"],
            exception["start_line"],
            exception["end_line"],
            exception.get("error_class"),
        )
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = cast(InspectionLedgerException, dict(exception))
            continue
        existing["fatal"] = bool(existing.get("fatal")) or bool(exception.get("fatal"))
        analyzer_ids = set(existing.get("analyzers", [])) | set(exception.get("analyzers", []))
        if analyzer_ids:
            existing["analyzers"] = sorted(analyzer_ids)

    return sorted(
        grouped.values(),
        key=lambda item: (
            item["path"],
            item.get("start_line") or 0,
            item.get("end_line") or 0,
            str(item["phase"]),
            str(item["reason_code"]),
        ),
    )


def _legacy_effective_ids(
    findings: list[object],
    legacy_filtered: object,
) -> list[str]:
    """Map pre-ledger meta output back to canonical IDs during the transition.

    The stacked producer MR preserves IDs directly. This compatibility path only
    supports the older meta node, which copied findings before opaque IDs existed.
    """
    known_ids = {getattr(finding, "finding_id", "") for finding in findings}
    if not isinstance(legacy_filtered, list):
        return [str(getattr(finding, "finding_id", "")) for finding in findings]

    by_shape: dict[tuple[object, ...], list[str]] = {}
    for finding in findings:
        shape = (
            getattr(finding, "rule_id", None),
            getattr(finding, "file", None),
            getattr(finding, "start_line", None),
            getattr(finding, "end_line", None),
        )
        by_shape.setdefault(shape, []).append(str(getattr(finding, "finding_id", "")))

    selected: list[str] = []
    consumed: set[str] = set()
    for finding in legacy_filtered:
        finding_id = str(getattr(finding, "finding_id", ""))
        if finding_id in known_ids:
            selected.append(finding_id)
            consumed.add(finding_id)
            continue
        shape = (
            getattr(finding, "rule_id", None),
            getattr(finding, "file", None),
            getattr(finding, "start_line", None),
            getattr(finding, "end_line", None),
        )
        candidate = next((item for item in by_shape.get(shape, []) if item not in consumed), None)
        if candidate:
            selected.append(candidate)
            consumed.add(candidate)
    return selected


def finalize_ledger(state: Mapping[str, object]) -> tuple[AnalysisCompleteness, list[str]]:
    """Validate ledger accounting and derive the canonical public projection.

    Full internal rows remain in graph state. Reports receive only scope boundaries,
    skipped/failed work, analyzer summaries, and safe policy-derived fatality.
    """
    raw_components = state.get("components", [])
    components = (
        [_safe_path(component, []) for component in raw_components if isinstance(component, str)]
        if isinstance(raw_components, list)
        else []
    )
    components = list(dict.fromkeys(components))
    raw_findings = state.get("findings", [])
    findings = list(raw_findings) if isinstance(raw_findings, list) else []
    raw_events = state.get("inspection_ledger", [])
    events = (
        [cast(InspectionLedgerEvent, event) for event in raw_events if isinstance(event, dict)]
        if isinstance(raw_events, list)
        else []
    )
    raw_statuses = state.get("analyzer_status_events", [])
    statuses = (
        [cast(AnalyzerStatusEvent, status) for status in raw_statuses if isinstance(status, dict)]
        if isinstance(raw_statuses, list)
        else []
    )

    findings_by_id: dict[str, object] = {}
    accounting_exceptions: list[InspectionLedgerException] = []

    def accounting_error(path: object = None) -> None:
        accounting_exceptions.append(
            _exception(
                outcome=LedgerOutcome.FAILED,
                phase="finalization",
                reason=LedgerReason.FINDING_ACCOUNTING_ERROR,
                path=_safe_path(path, components),
                fatal=True,
            )
        )

    for finding in findings:
        finding_id = str(getattr(finding, "finding_id", ""))
        if not finding_id or finding_id in findings_by_id:
            accounting_error(getattr(finding, "file", None))
            continue
        findings_by_id[finding_id] = finding

    events_by_work_id: dict[str, list[InspectionLedgerEvent]] = {}
    for event in events:
        events_by_work_id.setdefault(str(event.get("work_id", "")), []).append(event)

    producer_origins: dict[str, int] = {}
    producer_rows_present = False
    for event in events:
        outcome = event.get("outcome")
        phase = str(event.get("phase", ""))
        input_ids = list(event.get("input_finding_ids", []))
        emitted_ids = list(event.get("emitted_finding_ids", []))
        is_meta = _is_meta_phase(phase)
        is_producer = event.get("record_type") == LedgerRecordType.WORK_ITEM and not is_meta
        if is_producer:
            producer_rows_present = True
        if is_producer and input_ids:
            accounting_error(event.get("path"))
        if is_producer and outcome != LedgerOutcome.COMPLETED and emitted_ids:
            accounting_error(event.get("path"))
        if (
            is_meta
            and outcome == LedgerOutcome.COMPLETED
            and not set(emitted_ids).issubset(input_ids)
        ):
            accounting_error(event.get("path"))
        if (
            is_meta
            and outcome in (LedgerOutcome.FAILED, LedgerOutcome.SKIPPED)
            and emitted_ids != input_ids
        ):
            accounting_error(event.get("path"))
        for finding_id in [*input_ids, *emitted_ids]:
            if finding_id not in findings_by_id:
                accounting_error(event.get("path"))
        if is_producer and outcome == LedgerOutcome.COMPLETED:
            for finding_id in emitted_ids:
                producer_origins[finding_id] = producer_origins.get(finding_id, 0) + 1

    if producer_rows_present:
        for finding_id, finding in findings_by_id.items():
            if producer_origins.get(finding_id, 0) != 1:
                accounting_error(getattr(finding, "file", None))

    explicit_effective = state.get("effective_finding_ids")
    if isinstance(explicit_effective, list):
        effective_ids = [str(finding_id) for finding_id in explicit_effective]
    else:
        effective_ids = _legacy_effective_ids(findings, state.get("filtered_findings"))

    seen_effective: set[str] = set()
    validated_effective: list[str] = []
    for finding_id in effective_ids:
        if finding_id in seen_effective or finding_id not in findings_by_id:
            accounting_error(getattr(findings_by_id.get(finding_id), "file", None))
            continue
        seen_effective.add(finding_id)
        validated_effective.append(finding_id)

    meta_planned_ids = {
        target["work_id"]
        for status in statuses
        if status.get("analyzer_id") == "meta_analyzer"
        for target in status.get("planned_work", [])
    }
    if meta_planned_ids:
        meta_effective = _deduplicate_ids(
            finding_id
            for event in events
            if event.get("work_id") in meta_planned_ids
            and _is_meta_phase(str(event.get("phase", "")))
            for finding_id in event.get("emitted_finding_ids", [])
        )
        if meta_effective != validated_effective:
            accounting_error()

    unaccounted_exceptions: list[InspectionLedgerException] = []
    status_summaries: list[dict[str, object]] = []
    primary_targets: list[tuple[str, PlannedWorkTarget, list[InspectionLedgerEvent]]] = []
    for status in statuses:
        analyzer_id = str(status.get("analyzer_id", ""))
        planned_work = cast(list[PlannedWorkTarget], status.get("planned_work", []))
        outcome_counts = {"completed": 0, "skipped": 0, "failed": 0, "unaccounted": 0}
        for target in planned_work:
            work_id = str(target.get("work_id", ""))
            matches = events_by_work_id.get(work_id, [])
            if len(matches) != 1:
                outcome_counts["unaccounted"] += 1
                unaccounted_exceptions.append(
                    _exception(
                        outcome=LedgerOutcome.FAILED,
                        phase="finalization",
                        reason=LedgerReason.UNACCOUNTED_WORK,
                        path=_safe_path(target.get("path"), components),
                        start_line=target.get("start_line"),
                        end_line=target.get("end_line"),
                        analyzers=[analyzer_id],
                        fatal=True,
                    )
                )
            else:
                outcome_name = str(matches[0].get("outcome", "failed"))
                outcome_counts[outcome_name if outcome_name in outcome_counts else "failed"] += 1
            if analyzer_id != "meta_analyzer":
                primary_targets.append((analyzer_id, target, matches))
        summary: dict[str, object] = {
            "analyzer_id": analyzer_id,
            "status": status.get("status", "unknown"),
            "planned_work": len(planned_work),
            **outcome_counts,
        }
        if status.get("reason_code") is not None:
            summary["reason_code"] = str(status["reason_code"])
        if status.get("message") is not None:
            summary["message"] = str(status["message"])
        status_summaries.append(summary)

    scope_rows = [
        _exception_from_event(event, fatal=False)
        for event in events
        if event.get("outcome") == LedgerOutcome.OUT_OF_SCOPE
    ]
    exceptional_rows = [
        _exception_from_event(event, fatal=event.get("outcome") == LedgerOutcome.FAILED)
        for event in events
        if event.get("outcome") in (LedgerOutcome.SKIPPED, LedgerOutcome.FAILED)
    ]
    exceptional_rows.extend(unaccounted_exceptions)
    exceptional_rows.extend(accounting_exceptions)
    ledger_exceptions = _merge_exception_projection(exceptional_rows)
    scope_exclusions = _merge_exception_projection(scope_rows)

    per_component: dict[str, list[LedgerOutcome]] = {component: [] for component in components}
    if primary_targets:
        for _analyzer_id, target, matches in primary_targets:
            path = _safe_path(target.get("path"), components)
            outcomes = per_component.setdefault(path, [])
            if len(matches) == 1:
                outcomes.append(matches[0]["outcome"])
            else:
                outcomes.append(LedgerOutcome.FAILED)
    else:
        cache_failures = {
            str(event.get("path"))
            for event in events
            if event.get("phase") == "cache" and event.get("outcome") == LedgerOutcome.FAILED
        }
        for component in components:
            per_component[component].append(
                LedgerOutcome.FAILED if component in cache_failures else LedgerOutcome.COMPLETED
            )

    fully_inspected = 0
    partially_inspected = 0
    entirely_uninspected = 0
    for component in components:
        outcomes = per_component.get(component, [])
        if outcomes and all(outcome == LedgerOutcome.COMPLETED for outcome in outcomes):
            fully_inspected += 1
        elif any(outcome == LedgerOutcome.COMPLETED for outcome in outcomes):
            partially_inspected += 1
        else:
            entirely_uninspected += 1

    total_components = len(components)
    coverage_percent = (
        round(fully_inspected / total_components * 100, 1) if total_components else 100.0
    )
    limitations: list[str] = []
    for status_summary in status_summaries:
        status_name = str(status_summary["status"])
        if status_name not in {"completed", "not_applicable"}:
            message = status_summary.get("message")
            limitations.append(
                str(message)
                if message
                else f"Analyzer {status_summary['analyzer_id']} status: {status_name}."
            )
    is_complete = not ledger_exceptions and not limitations
    execution_successful = not any(exception.get("fatal") for exception in ledger_exceptions)

    completeness: AnalysisCompleteness = {
        "total_components": total_components,
        "scanned_components": fully_inspected,
        "coverage_percent": coverage_percent,
        "is_complete": is_complete,
        "execution_successful": execution_successful,
        "fully_inspected_files": fully_inspected,
        "partially_inspected_files": partially_inspected,
        "entirely_uninspected_files": entirely_uninspected,
        "ledger_exceptions": ledger_exceptions,
        "scope_exclusions": scope_exclusions,
        "analyzer_statuses": sorted(status_summaries, key=lambda item: str(item["analyzer_id"])),
        "limitations": limitations,
        "findings_before_filtering": len(findings_by_id),
        "findings_after_filtering": len(validated_effective),
    }
    return completeness, validated_effective


def guard_analyzer_node(
    analyzer_id: str,
    node: Callable[[object], dict[str, object]],
) -> Callable[[object], dict[str, object]]:
    """Convert an unexpected analyzer exception into safe, terminal ledger facts."""

    def guarded(state: object) -> dict[str, object]:
        try:
            return node(state)
        except Exception as exc:  # pragma: no cover - exact exception is node-dependent
            logger.warning("Analyzer %s raised %s", analyzer_id, type(exc).__name__, exc_info=True)
            state_mapping = cast(Mapping[str, object], state)
            raw_components = state_mapping.get("components", [])
            components = (
                [str(component) for component in raw_components]
                if isinstance(raw_components, list)
                else []
            )
            events = [
                ledger_event(
                    outcome=LedgerOutcome.FAILED,
                    phase="analyzer",
                    analyzer_id=analyzer_id,
                    path=_safe_path(component, components),
                    reason=LedgerReason.ANALYZER_RUNTIME_ERROR,
                    error_class=type(exc).__name__,
                )
                for component in components
            ]
            planned_work: list[PlannedWorkTarget] = [
                cast(
                    PlannedWorkTarget,
                    {
                        "work_id": event["work_id"],
                        "path": event["path"],
                        "start_line": event["start_line"],
                        "end_line": event["end_line"],
                    },
                )
                for event in events
            ]
            return {
                "findings": [],
                "inspection_ledger": events,
                "analyzer_status_events": [
                    analyzer_status_event(
                        analyzer_id=analyzer_id,
                        status="failed",
                        planned_work=planned_work,
                        reason=LedgerReason.ANALYZER_RUNTIME_ERROR,
                    )
                ],
            }

    return guarded
