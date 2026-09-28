# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-node adapter for canonical inspection-ledger finalization."""

from __future__ import annotations

import math
import re
import zlib
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath

from skillspector.artifacts import ContentKind, ReferenceKind
from skillspector.constants import MAX_ANALYZABLE_FILE_BYTES
from skillspector.inspection_ledger import (
    MAX_FINDING_OUTPUT_RECORDS,
    MAX_INSPECTION_LEDGER_EVENTS,
    REASON_MESSAGES,
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    LedgerRecordType,
    analyzer_status_for_events,
    finalize_ledger,
    inspection_work_id,
    ledger_event,
)
from skillspector.models import Finding
from skillspector.nodes.analyzers import ANALYZER_LOAD_ERRORS, ANALYZER_MODULES
from skillspector.semantic_runtime import (
    has_semantic_runtime_event,
    semantic_runtime_intent,
    semantic_runtime_ledger_event,
)
from skillspector.state import SkillspectorState

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_ALLOWED_CHUNKS = frozenset({b"IHDR", b"PLTE", b"IDAT", b"IEND"})
_PNG_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_PNG_BIT_DEPTHS = {
    0: frozenset({1, 2, 4, 8, 16}),
    2: frozenset({8, 16}),
    3: frozenset({1, 2, 4, 8}),
    4: frozenset({8, 16}),
    6: frozenset({8, 16}),
}
_MAX_PASSIVE_PNG_CHUNKS = 4_096
_MAX_REFERENCE_REASONS = 16
_REFERENCE_LIMIT_FIELDS = frozenset(
    f"{prefix}_{unit}"
    for prefix in ("observed", "limit")
    for unit in ("characters", "bytes", "findings", "artifacts", "depth", "records", "seconds")
)


def _reference_analysis_evidence(
    target_path: str,
    disposition: str,
    events: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Project bounded diagnostic facts without copying arbitrary ledger payloads."""
    evidence: dict[str, object] = {
        "target_path": target_path,
        "target_disposition": disposition,
    }
    reasons: list[dict[str, object]] = []
    for event in events:
        if str(event.get("outcome")) not in {"partial", "failed", "skipped", "out_of_scope"}:
            continue
        try:
            reason = LedgerReason(str(event.get("reason_code")))
        except ValueError:
            continue
        phase = event.get("phase")
        analyzer = event.get("analyzer_id")
        row: dict[str, object] = {
            "reason_code": reason.value,
            "message": REASON_MESSAGES[reason],
            "phase": (
                phase
                if isinstance(phase, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", phase)
                else "unknown"
            ),
            "analyzers": (
                [analyzer]
                if isinstance(analyzer, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", analyzer)
                else []
            ),
        }
        for field in ("start_line", "end_line"):
            value = event.get(field)
            if type(value) is int and 0 < value < 2**63:
                row[field] = value
        limits: dict[str, int | float] = {}
        for field in sorted(_REFERENCE_LIMIT_FIELDS):
            value = event.get(field)
            if type(value) is int and 0 <= value < 2**63:
                limits[field] = value
            elif (
                field.endswith("_seconds")
                and type(value) is float
                and math.isfinite(value)
                and value >= 0
            ):
                limits[field] = value
        if limits:
            row["limits"] = limits
        if row in reasons:
            continue
        if len(reasons) == _MAX_REFERENCE_REASONS:
            evidence["reasons_truncated"] = True
            break
        reasons.append(row)
    evidence["reasons"] = reasons
    return evidence


def _reference_analysis_remediation(evidence: Mapping[str, object]) -> str:
    """Give resolved references advice appropriate to their inspection limitation."""
    reasons = {
        str(row.get("reason_code"))
        for row in evidence.get("reasons", [])
        if isinstance(row, Mapping)
    }
    actions: list[str] = []
    if LedgerReason.STATIC_PARSE_LIMIT in reasons:
        actions.append(
            "Review the reported expression and analyzer's parsing limitation; "
            "correct the scanner if it misinterprets valid source, then rerun the scan."
        )
    if reasons & {LedgerReason.RUNTIME_LIMIT, LedgerReason.SIZE_LIMIT}:
        actions.append(
            "Review the reported analysis bounds and input size; "
            "resolve the scanner limit or provide fully inspectable source, then rerun the scan."
        )
    if reasons & {
        LedgerReason.READ_ERROR,
        LedgerReason.STAT_ERROR,
        LedgerReason.FILE_DISAPPEARED,
        LedgerReason.MISSING_FILE_CACHE,
    }:
        actions.append(
            "Ensure the resolved target remains readable throughout the scan, then rerun."
        )
    if reasons & {LedgerReason.BINARY_CONTENT, LedgerReason.OPAQUE_CONTENT}:
        actions.append("Provide inspectable source or analysis support for the referenced format.")
    if not actions or len(reasons) > 1 or evidence.get("reasons_truncated"):
        actions.append(
            "Review the target's analysis-completeness ledger and resolve all limitations."
        )
    actions.append(
        "Keep required references; incomplete analysis is not proof of malicious evasion."
    )
    return " ".join(actions)


def _png_scanlines_are_valid(compressed: bytes, expected_size: int, row_size: int) -> bool:
    """Decode one bounded PNG stream and validate its non-interlaced scanlines."""
    if expected_size <= 0 or expected_size > MAX_ANALYZABLE_FILE_BYTES:
        return False
    try:
        decoder = zlib.decompressobj()
        decoded = decoder.decompress(compressed, expected_size + 1)
        if len(decoded) > expected_size or decoder.unconsumed_tail:
            return False
        remaining = expected_size + 1 - len(decoded)
        if remaining > 0:
            decoded += decoder.flush(remaining)
    except zlib.error:
        return False
    return (
        len(decoded) == expected_size
        and decoder.eof
        and not decoder.unused_data
        and not decoder.unconsumed_tail
        and all(decoded[offset] <= 4 for offset in range(0, expected_size, row_size))
    )


def _is_structurally_valid_passive_png(data: bytes) -> bool:
    """Accept only a bounded, minimal, fully decoded non-interlaced PNG."""
    if len(data) > MAX_ANALYZABLE_FILE_BYTES or not data.startswith(_PNG_SIGNATURE):
        return False

    offset = len(_PNG_SIGNATURE)
    chunk_count = 0
    seen_ihdr = False
    seen_plte = False
    seen_idat = False
    idat_finished = False
    color_type = -1
    bit_depth = -1
    expected_size = 0
    row_size = 0
    compressed_parts: list[bytes] = []

    while offset < len(data):
        chunk_count += 1
        if chunk_count > _MAX_PASSIVE_PNG_CHUNKS or len(data) - offset < 12:
            return False
        chunk_length = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = data[offset + 4 : offset + 8]
        chunk_end = offset + 12 + chunk_length
        if (
            chunk_length > 0x7FFFFFFF
            or chunk_end > len(data)
            or chunk_type not in _PNG_ALLOWED_CHUNKS
        ):
            return False
        chunk_data = data[offset + 8 : offset + 8 + chunk_length]
        recorded_crc = int.from_bytes(data[offset + 8 + chunk_length : chunk_end], "big")
        computed_crc = zlib.crc32(chunk_type)
        computed_crc = zlib.crc32(chunk_data, computed_crc) & 0xFFFFFFFF
        if recorded_crc != computed_crc:
            return False

        if chunk_type == b"IHDR":
            if seen_ihdr or chunk_count != 1 or chunk_length != 13:
                return False
            width = int.from_bytes(chunk_data[0:4], "big")
            height = int.from_bytes(chunk_data[4:8], "big")
            bit_depth, color_type, compression, filtering, interlace = chunk_data[8:13]
            if (
                width == 0
                or height == 0
                or bit_depth not in _PNG_BIT_DEPTHS.get(color_type, ())
                or compression != 0
                or filtering != 0
                or interlace != 0
            ):
                return False
            scanline_bytes = (width * _PNG_CHANNELS[color_type] * bit_depth + 7) // 8
            row_size = scanline_bytes + 1
            expected_size = row_size * height
            if expected_size > MAX_ANALYZABLE_FILE_BYTES:
                return False
            seen_ihdr = True
        elif chunk_type == b"PLTE":
            if (
                not seen_ihdr
                or seen_plte
                or seen_idat
                or color_type in {0, 4}
                or chunk_length == 0
                or chunk_length > 768
                or chunk_length % 3 != 0
                or (color_type == 3 and chunk_length // 3 > 2**bit_depth)
            ):
                return False
            seen_plte = True
        elif chunk_type == b"IDAT":
            if not seen_ihdr or idat_finished or (color_type == 3 and not seen_plte):
                return False
            seen_idat = True
            compressed_parts.append(chunk_data)
        else:
            if (
                chunk_type != b"IEND"
                or not seen_ihdr
                or not seen_idat
                or chunk_length != 0
                or chunk_end != len(data)
            ):
                return False
            return _png_scanlines_are_valid(
                b"".join(compressed_parts),
                expected_size,
                row_size,
            )

        if seen_idat and chunk_type != b"IDAT":
            idat_finished = True
        offset = chunk_end
    return False


def _has_verified_passive_png(
    target_path: str,
    inventory_item: Mapping[str, object] | None,
    raw_file_cache: Mapping[str, object],
) -> bool:
    """Require complete canonical bytes for the one proven-passive image format."""
    if PurePosixPath(target_path).suffix.lower() != ".png" or not inventory_item:
        return False
    raw = raw_file_cache.get(target_path)
    size_bytes = inventory_item.get("size_bytes")
    return (
        isinstance(raw, bytes)
        and type(size_bytes) is int
        and size_bytes == len(raw)
        and _is_structurally_valid_passive_png(raw)
    )


def _has_only_format_limitations(
    target_path: str,
    inventory_item: Mapping[str, object] | None,
    events: list[Mapping[str, object]],
    raw_file_cache: Mapping[str, object],
) -> bool:
    """Recognize unsupported content without hiding other coverage failures."""
    if not _has_verified_passive_png(target_path, inventory_item, raw_file_cache):
        return False
    if not inventory_item or str(inventory_item.get("content_kind")) not in {
        ContentKind.BINARY,
        ContentKind.OPAQUE,
    }:
        return False
    if inventory_item.get("misleading_extension") is True:
        return False
    if str(inventory_item.get("disposition")) not in {"partial", "out_of_scope"}:
        return False
    if inventory_item.get("referenced") is not True:
        return False
    format_reasons = {LedgerReason.BINARY_CONTENT, LedgerReason.OPAQUE_CONTENT}
    for reason_field in ("reason", "inherited_exclusion_reason"):
        inventory_reason = inventory_item.get(reason_field)
        if inventory_reason is not None and str(inventory_reason) not in format_reasons:
            return False
    # Classifying an asset is insufficient: require explicit ledger evidence.
    # Missing/unknown reasons, skipped work, failed work and mixed limitations
    # must continue to produce AE1. Completed analyzers do not erase exceptions.
    exceptions = [event for event in events if event.get("outcome") != "completed"]
    return (
        bool(exceptions)
        and not any(event.get("fatal") for event in events)
        and not any(
            str(event.get("outcome")) == LedgerOutcome.COMPLETED
            and event.get("reason_code") is not None
            for event in events
        )
        and all(
            str(event.get("outcome")) in {"partial", "out_of_scope"}
            and str(event.get("reason_code")) in format_reasons
            for event in exceptions
        )
    )


def _ledger_evidence_was_truncated(events: Sequence[Mapping[str, object]]) -> bool:
    """Return whether a ledger cap may have hidden a non-format limitation."""
    for event in events:
        if str(event.get("reason_code")) != LedgerReason.OUTPUT_LIMIT:
            continue
        if str(event.get("phase")) == "ledger_output":
            return True
        observed = event.get("observed_records")
        limit = event.get("limit_records")
        if (
            str(event.get("record_type")) == LedgerRecordType.SYSTEM
            and type(observed) is int
            and type(limit) is int
            and limit == MAX_INSPECTION_LEDGER_EVENTS
            and observed > limit
        ):
            return True
    return False


def _path_is_canonical(value: object, *, scope_boundary: bool = False) -> bool:
    """Return whether a stored ledger path already has canonical POSIX form."""
    if not isinstance(value, str) or not value or "\\" in value or value.startswith("/"):
        return False
    if len(value) >= 2 and value[1] == ":":
        return False
    has_trailing_slash = value.endswith("/")
    if has_trailing_slash and not scope_boundary:
        return False
    core = value[:-1] if has_trailing_slash else value
    path = PurePosixPath(core)
    if not core or path.is_absolute() or any(part in {"", ".", ".."} for part in core.split("/")):
        return False
    canonical = path.as_posix() + ("/" if has_trailing_slash else "")
    return value == canonical


def _ledger_event_is_canonical(event: Mapping[str, object]) -> bool:
    """Validate the fields used to prove that no other limitation was present."""
    try:
        outcome = LedgerOutcome(str(event.get("outcome")))
        record_type = LedgerRecordType(str(event.get("record_type")))
        reason_value = event.get("reason_code")
        if outcome is LedgerOutcome.COMPLETED:
            if reason_value is not None:
                return False
        else:
            LedgerReason(str(reason_value))
        phase = event.get("phase")
        path = event.get("path")
        work_id = event.get("work_id")
        analyzer_id = event.get("analyzer_id")
        start_line = event.get("start_line")
        end_line = event.get("end_line")
        input_ids = event.get("input_finding_ids")
        emitted_ids = event.get("emitted_finding_ids")
        if (
            not isinstance(phase, str)
            or not phase
            or not isinstance(path, str)
            or not _path_is_canonical(
                path, scope_boundary=record_type is LedgerRecordType.SCOPE_BOUNDARY
            )
            or not isinstance(work_id, str)
            or not work_id
            or (analyzer_id is not None and not isinstance(analyzer_id, str))
            or (start_line is not None and type(start_line) is not int)
            or (end_line is not None and type(end_line) is not int)
            or not isinstance(input_ids, list)
            or not all(isinstance(finding_id, str) for finding_id in input_ids)
            or not isinstance(emitted_ids, list)
            or not all(isinstance(finding_id, str) for finding_id in emitted_ids)
        ):
            return False
        identity = analyzer_id or f"{record_type.value}:{phase}"
        return work_id == inspection_work_id(identity, path, start_line, end_line)
    except (TypeError, ValueError):
        return False


def _status_paths_with_incomplete_evidence(
    statuses: object,
    events: Sequence[Mapping[str, object]],
) -> tuple[bool, set[str]]:
    """Validate analyzer summaries and return targets whose status contradicts their rows."""
    if statuses is None:
        return True, set()
    if not isinstance(statuses, list) or not all(isinstance(status, dict) for status in statuses):
        return False, set()
    events_by_work_id: dict[str, list[Mapping[str, object]]] = {}
    for event in events:
        events_by_work_id.setdefault(str(event.get("work_id", "")), []).append(event)
    invalid_paths: set[str] = set()
    for status in statuses:
        analyzer_id = status.get("analyzer_id")
        status_name = status.get("status")
        planned_work = status.get("planned_work")
        if (
            not isinstance(analyzer_id, str)
            or not analyzer_id
            or not isinstance(status_name, str)
            or not isinstance(planned_work, list)
        ):
            return False, set()
        planned_paths: set[str] = set()
        outcomes: set[str] = set()
        reason_value = status.get("reason_code")
        if reason_value is not None:
            try:
                status_reason = LedgerReason(str(reason_value))
            except ValueError:
                return False, set()
        else:
            status_reason = None
        for target in planned_work:
            if not isinstance(target, Mapping):
                return False, set()
            work_id = target.get("work_id")
            path = target.get("path")
            start_line = target.get("start_line")
            end_line = target.get("end_line")
            if (
                not isinstance(work_id, str)
                or not isinstance(path, str)
                or not _path_is_canonical(path)
                or (start_line is not None and type(start_line) is not int)
                or (end_line is not None and type(end_line) is not int)
            ):
                return False, set()
            planned_paths.add(path)
            matches = events_by_work_id.get(work_id, [])
            if len(matches) != 1:
                invalid_paths.add(path)
                continue
            match = matches[0]
            if (
                str(match.get("path", "")) != path
                or match.get("start_line") != start_line
                or match.get("end_line") != end_line
            ):
                invalid_paths.add(path)
                continue
            outcomes.add(str(match.get("outcome")))
        if planned_work:
            expected_status = (
                "failed"
                if "failed" in outcomes
                else "degraded"
                if outcomes & {"partial", "skipped"}
                else "completed"
            )
            if status_name != expected_status:
                invalid_paths.update(planned_paths)
            if status_reason not in {
                None,
                LedgerReason.BINARY_CONTENT,
                LedgerReason.OPAQUE_CONTENT,
            }:
                invalid_paths.update(planned_paths)
    return True, invalid_paths


def _reference_coverage_findings(
    state: SkillspectorState,
) -> list[Finding]:
    """Create AE1 only for canonical resolved targets with incomplete disposition."""
    raw_references = state.get("artifact_references") or []
    raw_cache = state.get("raw_file_cache")
    raw_file_cache: Mapping[str, object] = raw_cache if isinstance(raw_cache, Mapping) else {}
    raw_inventory = state.get("artifact_inventory")
    inventory_shape_complete = isinstance(raw_inventory, list) and all(
        isinstance(item, dict) and _path_is_canonical(item.get("path")) for item in raw_inventory
    )
    inventory: dict[str, Mapping[str, object]] = {}
    duplicate_inventory_paths: set[str] = set()
    for item in raw_inventory or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", ""))
        if path in inventory:
            duplicate_inventory_paths.add(path)
        inventory[path] = item
    events_by_path: dict[str, list[Mapping[str, object]]] = {}
    raw_ledger_events = state.get("inspection_ledger")
    ledger_shape_complete = isinstance(raw_ledger_events, list) and all(
        isinstance(event, dict) and _ledger_event_is_canonical(event) for event in raw_ledger_events
    )
    ledger_events = (
        [event for event in raw_ledger_events if isinstance(event, dict)]
        if isinstance(raw_ledger_events, list)
        else []
    )
    for event in ledger_events:
        events_by_path.setdefault(str(event.get("path", "")), []).append(event)
    status_shape_complete, invalid_status_paths = _status_paths_with_incomplete_evidence(
        state.get("analyzer_status_events"), ledger_events
    )
    fatal_paths: set[str] = set()
    canonicalization_succeeded = False
    if ledger_shape_complete and inventory_shape_complete and status_shape_complete:
        try:
            preexisting_completeness, _ = finalize_ledger(state)
            fatal_paths = {
                str(exception.get("path", ""))
                for exception in preexisting_completeness["ledger_exceptions"]
                if exception.get("fatal") is True
            }
            canonicalization_succeeded = True
        except (AttributeError, KeyError, TypeError, ValueError):
            # Persisted or manually shaped state can predate the canonical ledger
            # schema. Such evidence cannot prove that a format limitation was the
            # target's only inspection gap, but it must not crash AE1 classification.
            pass
    ledger_evidence_complete = (
        ledger_shape_complete
        and inventory_shape_complete
        and status_shape_complete
        and canonicalization_succeeded
        and not _ledger_evidence_was_truncated(ledger_events)
    )
    findings: list[Finding] = []
    seen_locations: set[tuple[str, int, str]] = set()
    diagnostics_by_target: dict[tuple[str, str], tuple[dict[str, object], str]] = {}
    for reference in raw_references:
        if not isinstance(reference, dict):
            continue
        status = str(reference.get("status", ""))
        if status != "resolved":
            continue
        target = reference.get("target_path")
        target_path = str(target) if target else ""
        inventory_item = inventory.get(target_path)
        disposition = str(inventory_item.get("disposition", "")) if inventory_item else ""
        reference_disposition = str(reference.get("disposition", ""))
        target_events = events_by_path.get(target_path, [])
        incomplete_dispositions = {str(event.get("outcome", "")) for event in target_events} | {
            disposition,
            reference_disposition,
        }
        final_disposition = (
            "failed"
            if "failed" in incomplete_dispositions
            else "partial"
            if "partial" in incomplete_dispositions
            else "out_of_scope"
            if "out_of_scope" in incomplete_dispositions
            else ""
        )
        if final_disposition not in {"partial", "failed", "out_of_scope"}:
            continue
        # The all-reasons check is the safety boundary: executables, concealed
        # content, read failures, and any mixed limitation must retain AE1.
        if (
            ledger_evidence_complete
            and target_path not in fatal_paths
            and target_path not in invalid_status_paths
            and target_path not in duplicate_inventory_paths
            and reference_disposition == disposition
            and str(reference.get("reference_kind", "")) == ReferenceKind.MARKDOWN_IMAGE
            and _has_only_format_limitations(
                target_path,
                inventory_item,
                target_events,
                raw_file_cache,
            )
        ):
            continue
        line_value = reference.get("line", 1)
        source_path = str(reference.get("source_path", "SKILL.md"))
        line = line_value if isinstance(line_value, int) else 1
        # A Markdown label and destination may resolve to the same artifact.
        # Findings identify source lines, so emit that coverage gap only once.
        location = (source_path, line, target_path)
        if location in seen_locations:
            continue
        seen_locations.add(location)
        evidence = str(reference.get("evidence", ""))[:160]
        diagnostic_key = (target_path, final_disposition)
        if diagnostic_key not in diagnostics_by_target:
            analysis_evidence = _reference_analysis_evidence(
                target_path, final_disposition, target_events
            )
            diagnostics_by_target[diagnostic_key] = (
                analysis_evidence,
                _reference_analysis_remediation(analysis_evidence),
            )
        analysis_evidence, remediation = diagnostics_by_target[diagnostic_key]
        findings.append(
            Finding(
                rule_id="AE1",
                message="Referenced artifact was not completely inspected",
                pattern="Incomplete referenced artifact analysis",
                severity="HIGH",
                confidence=1.0,
                file=source_path,
                start_line=line,
                category="analysis-evasion",
                tags=["coverage", "reference", f"target-disposition:{final_disposition}"],
                finding=f"{target_path} ({final_disposition})"[:200],
                code_snippet=evidence,
                matched_text=target_path,
                evidence=analysis_evidence,
                remediation=remediation,
            )
        )
    return findings


def _size_coverage_findings(
    state: SkillspectorState,
    covered_paths: set[str],
) -> list[Finding]:
    """Create AE7 for artifacts truncated by the per-file size cap.

    A file too large to fully analyze must not be able to produce a
    zero-finding report: the unreviewed region is itself the finding.
    Scoped to the per-file cap (``size_limit``); aggregate budget
    exhaustion already fails closed through the completeness projection.
    Paths already reported by AE1 (referenced artifacts) are skipped.
    """
    findings: list[Finding] = []
    for item in state.get("artifact_inventory") or []:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("disposition", "")) != "partial":
            continue
        if str(item.get("reason", "")) != LedgerReason.SIZE_LIMIT.value:
            continue
        if str(item.get("path", "")) in covered_paths:
            continue
        path = str(item.get("path", ""))
        size_bytes = item.get("size_bytes", 0)
        findings.append(
            Finding(
                rule_id="AE7",
                message=(
                    "File exceeds the analyzable size limit; trailing content was not inspected"
                ),
                severity="HIGH",
                confidence=1.0,
                file=path,
                start_line=1,
                category="analysis-evasion",
                tags=["coverage", "size-limit"],
                finding=f"{path} ({size_bytes} bytes, partially inspected)"[:200],
                matched_text=path,
                remediation=(
                    "Keep analyzable files under the per-file size limit, or "
                    "split oversized content so every byte can be inspected."
                ),
            )
        )
    return findings


def _analyzer_load_error_events() -> list[InspectionLedgerEvent]:
    """Surface analyzer modules dropped by the registry before any node ran.

    ``_discover_analyzers`` logs an import failure and moves on, so a module
    that cannot be imported never gets a graph node and never emits a
    work-item event of its own. Recorded here as a SYSTEM record so a dropped
    analyzer degrades ``analysis_completeness`` instead of leaving a scan
    silently short of whatever that analyzer would have looked for.
    """
    return [
        ledger_event(
            outcome=LedgerOutcome.PARTIAL,
            record_type=LedgerRecordType.SYSTEM,
            phase="analyzer_registry",
            path=f"analyzer_registry/{module_name}",
            reason=LedgerReason.ANALYZER_LOAD_ERROR,
        )
        for module_name in sorted(ANALYZER_LOAD_ERRORS)
    ]


def finalize_inspection_ledger(state: SkillspectorState) -> dict[str, object]:
    """Validate full internal facts and derive the public completeness projection."""
    reference_findings = _reference_coverage_findings(state)
    size_findings = _size_coverage_findings(
        state,
        covered_paths={str(finding.matched_text or "") for finding in reference_findings},
    )
    coverage_findings = [*reference_findings, *size_findings]
    # Work IDs are scoped to analyzer, source file and line range. Distinct
    # targets on one line must share a terminal row with all emitted findings.
    coverage_ids_by_line: dict[tuple[str, int | None], list[str]] = {}
    for finding in coverage_findings:
        coverage_ids_by_line.setdefault((finding.file, finding.start_line), []).append(
            finding.finding_id
        )
    reference_events: list[InspectionLedgerEvent] = [
        ledger_event(
            outcome=LedgerOutcome.COMPLETED,
            phase="reference",
            analyzer_id="reference_coverage",
            path=path,
            start_line=line,
            end_line=line,
            emitted_finding_ids=finding_ids,
        )
        for (path, line), finding_ids in coverage_ids_by_line.items()
    ]
    merged_state = dict(state)
    all_findings = [*(state.get("findings") or []), *coverage_findings]
    output_events: list[InspectionLedgerEvent] = []
    finding_output_records = sum(max(1, len(finding.occurrences)) for finding in all_findings)
    if finding_output_records > MAX_FINDING_OUTPUT_RECORDS:
        output_events.append(
            ledger_event(
                outcome=LedgerOutcome.PARTIAL,
                record_type=LedgerRecordType.SYSTEM,
                phase="finding_output",
                path=next(
                    (finding.file for finding in all_findings if finding.occurrences),
                    all_findings[MAX_FINDING_OUTPUT_RECORDS].file
                    if len(all_findings) > MAX_FINDING_OUTPUT_RECORDS
                    else "SKILL.md",
                ),
                reason=LedgerReason.OUTPUT_LIMIT,
                observed_findings=finding_output_records,
                limit_findings=MAX_FINDING_OUTPUT_RECORDS,
            )
        )
    load_error_events = _analyzer_load_error_events()
    merged_state["findings"] = all_findings
    merged_state["effective_finding_ids"] = [
        *(state.get("effective_finding_ids") or []),
        *(finding.finding_id for finding in coverage_findings),
    ]
    llm_requested, llm_enabled = semantic_runtime_intent(merged_state)
    runtime_event = semantic_runtime_ledger_event(
        requested=llm_requested,
        enabled=llm_enabled,
        result=merged_state,
        discovered_modules=ANALYZER_MODULES,
    )
    runtime_events = (
        [runtime_event]
        if runtime_event is not None
        and not has_semantic_runtime_event(state.get("inspection_ledger") or [], runtime_event)
        else []
    )
    merged_state["inspection_ledger"] = [
        *(state.get("inspection_ledger") or []),
        *reference_events,
        *output_events,
        *runtime_events,
        *load_error_events,
    ]
    reference_statuses = (
        [analyzer_status_for_events("reference_coverage", reference_events)]
        if reference_events
        else []
    )
    merged_state["analyzer_status_events"] = [
        *(state.get("analyzer_status_events") or []),
        *reference_statuses,
    ]
    completeness, effective_finding_ids = finalize_ledger(merged_state)
    if coverage_findings and completeness["status"] == "complete":
        completeness["status"] = "partial"
        completeness["is_complete"] = False
        limitations = completeness.setdefault("limitations", [])
        limitations.append("One or more artifacts were not completely inspected.")
    return {
        "analysis_completeness": completeness,
        "execution_successful": completeness["execution_successful"],
        "findings": coverage_findings,
        "effective_finding_ids": effective_finding_ids,
        "inspection_ledger": [
            *reference_events,
            *output_events,
            *runtime_events,
            *load_error_events,
        ],
        "analyzer_status_events": reference_statuses,
    }
