# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-independent semantic analyzer runtime accounting."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from skillspector.inspection_ledger import (
    FORMAT_RECOVERY_SEMANTIC_ANALYZERS,
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    LedgerRecordType,
    ledger_event,
)

CANONICAL_SEMANTIC_ANALYZER_IDS = FORMAT_RECOVERY_SEMANTIC_ANALYZERS

SEMANTIC_PREFLIGHT_UNAVAILABLE_MESSAGE = (
    "Requested semantic analysis was unavailable before execution."
)
SEMANTIC_RUNTIME_INCOMPLETE_MESSAGE = (
    "Requested semantic analysis did not produce complete per-source runtime telemetry."
)

_ROOT_SOURCE_SCOPE = "<root>"


def required_semantic_analyzer_ids(
    discovered_modules: Mapping[str, object],
) -> frozenset[str]:
    """Return stable requirements plus newly discovered credential-gated analyzers."""
    discovered = frozenset(
        analyzer_id
        for analyzer_id, module in discovered_modules.items()
        if getattr(module, "requires_api_key", False)
    )
    return CANONICAL_SEMANTIC_ANALYZER_IDS | discovered


def semantic_runtime_intent(result: Mapping[str, object]) -> tuple[bool, bool]:
    """Return strict ``(requested, enabled)`` semantic-analysis intent."""
    if "use_llm" not in result and "llm_requested" not in result:
        return False, False
    enabled = result.get("use_llm") is not False
    raw_requested = result.get("llm_requested")
    requested = raw_requested if isinstance(raw_requested, bool) else enabled
    return requested, enabled


def successful_llm_record(record: object) -> bool:
    """Return whether ``record`` is a well-formed successful LLM call."""
    return (
        isinstance(record, Mapping)
        and isinstance(record.get("node"), str)
        and bool(record.get("node"))
        and record.get("ok") is True
        and record.get("error") is None
    )


def _source_scope(record: Mapping[str, object]) -> str | None:
    """Return a stable scope key, rejecting malformed explicit identities."""
    source_identity = record.get("source_identity")
    if source_identity is None:
        return _ROOT_SOURCE_SCOPE
    if isinstance(source_identity, str) and source_identity:
        return source_identity
    return None


def _has_effective_findings(result: Mapping[str, object]) -> bool:
    """Return whether meta-analysis had effective findings to process."""
    required = result.get("meta_review_required")
    if isinstance(required, bool):
        return required
    findings = result.get("findings")
    if isinstance(findings, list) and findings:
        for finding in findings:
            tags: object = getattr(finding, "tags", None)
            if tags is None and isinstance(finding, Mapping):
                tags = finding.get("tags")
            if not isinstance(tags, list) or "coverage" not in tags:
                return True
        return False
    effective_ids = result.get("effective_finding_ids")
    if isinstance(effective_ids, list):
        return bool(effective_ids)
    filtered_findings = result.get("filtered_findings")
    return isinstance(filtered_findings, list) and bool(filtered_findings)


def llm_runtime_available(
    *,
    preflight_available: bool,
    result: Mapping[str, object],
) -> bool:
    """Return provider availability after applying meta-analysis runtime evidence."""
    if not preflight_available:
        return False
    call_log = result.get("llm_call_log")
    if not isinstance(call_log, list):
        return True
    meta_analyzer_records = [
        record
        for record in call_log
        if isinstance(record, Mapping) and record.get("node") == "meta_analyzer"
    ]
    return all(successful_llm_record(record) for record in meta_analyzer_records)


def semantic_runtime_accounting(
    *,
    enabled: bool,
    result: Mapping[str, object],
    discovered_modules: Mapping[str, object],
) -> tuple[bool, bool]:
    """Return ``(used, complete)`` for an enabled semantic LLM pass.

    A requested pass is complete only when every required semantic analyzer
    explicitly reports either ``completed`` with successful telemetry or
    ``not_applicable``. Empty telemetry never proves use. Meta-analysis also
    needs a successful record when effective findings exist.
    """
    if not enabled:
        return False, False

    raw_call_log = result.get("llm_call_log", [])
    if not isinstance(raw_call_log, list):
        return False, False
    used = any(successful_llm_record(record) for record in raw_call_log)
    if not all(successful_llm_record(record) for record in raw_call_log):
        return used, False

    calls_by_scope_and_node: dict[tuple[str, str], int] = {}
    for record in raw_call_log:
        # successful_llm_record() above established the Mapping and node shape.
        assert isinstance(record, Mapping)
        scope = _source_scope(record)
        node = record.get("node")
        if scope is None or not isinstance(node, str):
            return used, False
        key = (scope, node)
        calls_by_scope_and_node[key] = calls_by_scope_and_node.get(key, 0) + 1

    raw_statuses = result.get("analyzer_status_events")
    if not isinstance(raw_statuses, list):
        return used, False
    required_analyzer_ids = required_semantic_analyzer_ids(discovered_modules)
    statuses_by_scope_and_analyzer: dict[tuple[str, str], list[str]] = {}
    semantic_scopes: set[str] = set()
    for status in raw_statuses:
        if not isinstance(status, Mapping):
            return used, False
        scope = _source_scope(status)
        analyzer_id = status.get("analyzer_id")
        analyzer_status = status.get("status")
        if (
            scope is None
            or not isinstance(analyzer_id, str)
            or not analyzer_id
            or not isinstance(analyzer_status, str)
            or not analyzer_status
        ):
            return used, False
        if analyzer_id in required_analyzer_ids:
            semantic_scopes.add(scope)
            statuses_by_scope_and_analyzer.setdefault((scope, analyzer_id), []).append(
                analyzer_status
            )

    if not semantic_scopes:
        return used, False

    for scope in semantic_scopes:
        for analyzer_id in required_analyzer_ids:
            statuses = statuses_by_scope_and_analyzer.get((scope, analyzer_id))
            if statuses is None or len(statuses) != 1:
                return used, False
            status = statuses[0]
            successful_calls = calls_by_scope_and_node.get((scope, analyzer_id), 0)
            if status == "completed":
                if successful_calls != 1:
                    return used, False
            elif status == "not_applicable":
                if successful_calls != 0:
                    return used, False
            else:
                return used, False

    # A call must never borrow an identically named status from another source.
    for (scope, analyzer_id), _count in calls_by_scope_and_node.items():
        if analyzer_id in required_analyzer_ids and scope not in semantic_scopes:
            return used, False

    if _has_effective_findings(result) and not any(
        successful_llm_record(record) and record.get("node") == "meta_analyzer"
        for record in raw_call_log
    ):
        return used, False

    return used, True


def is_format_recovery_event(event: Mapping[str, object]) -> bool:
    """Identify a completed recovery procedure whose assessment remains unknown."""
    analyzer = event.get("analyzer_id")
    return (
        event.get("record_type") == LedgerRecordType.WORK_ITEM
        and event.get("outcome") == LedgerOutcome.PARTIAL
        and event.get("reason_code") == LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED
        and isinstance(analyzer, str)
        and bool(analyzer)
        and event.get("phase") == ("meta" if analyzer == "meta_analyzer" else "semantic")
    )


def format_recovery_runtime_accepted(
    result: Mapping[str, object],
    discovered_modules: Mapping[str, object],
) -> bool:
    """Prove format-only policy acceptance separately from semantic completeness.

    Every warning must bind to one submitted work item and successful recovery
    telemetry in the same source. Projecting those statuses here never changes
    the original ledger, call log, or reported inspection completeness.
    """
    requested, enabled = semantic_runtime_intent(result)
    events = result.get("inspection_ledger")
    statuses = result.get("analyzer_status_events")
    calls = result.get("llm_call_log")
    if (
        not requested
        or not enabled
        or not all(isinstance(value, list) for value in (events, statuses, calls))
    ):
        return False
    assert isinstance(events, list) and isinstance(statuses, list) and isinstance(calls, list)
    allowed_analyzers = required_semantic_analyzer_ids(discovered_modules) | {
        "meta_analyzer",
        "mcp_tool_poisoning",
    }
    by_work: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    warning_work: set[tuple[str, str]] = set()
    warning_scopes: set[tuple[str, str]] = set()
    for event in events:
        if not isinstance(event, Mapping):
            return False
        scope, work_id = _source_scope(event), event.get("work_id")
        if scope is None or not isinstance(work_id, str) or not work_id:
            return False
        key = (scope, work_id)
        by_work.setdefault(key, []).append(event)
        if event.get("reason_code") == LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED:
            analyzer = event.get("analyzer_id")
            if not is_format_recovery_event(event) or analyzer not in allowed_analyzers:
                return False
            assert isinstance(analyzer, str)
            warning_work.add(key)
            warning_scopes.add((scope, analyzer))
    if not warning_work:
        return False
    projected_statuses: list[Mapping[str, object]] = []
    accounted_work: set[tuple[str, str]] = set()
    accounted_scopes: set[tuple[str, str]] = set()
    for status in statuses:
        if not isinstance(status, Mapping):
            return False
        scope, analyzer = _source_scope(status), status.get("analyzer_id")
        if scope is None or not isinstance(analyzer, str):
            return False
        key = (scope, analyzer)
        if key not in warning_scopes:
            projected_statuses.append(status)
            continue
        targets = status.get("planned_work")
        if (
            key in accounted_scopes
            or status.get("status") != "degraded"
            or not isinstance(targets, list)
            or not targets
        ):
            return False
        for target in targets:
            if not isinstance(target, Mapping) or not isinstance(target.get("work_id"), str):
                return False
            work_key = (scope, str(target["work_id"]))
            matches = by_work.get(work_key, [])
            if len(matches) != 1:
                return False
            event = matches[0]
            # TP4 shares its status with the same node's completed static phase.
            own_static_work = (
                analyzer == "mcp_tool_poisoning"
                and event.get("analyzer_id") == "mcp_tool_poisoning_static"
                and event.get("record_type") == LedgerRecordType.WORK_ITEM
                and event.get("phase") == "static"
                and event.get("path")
                == ("SKILL.md" if scope == _ROOT_SOURCE_SCOPE else scope + "/SKILL.md")
            )
            if (event.get("analyzer_id") != analyzer and not own_static_work) or any(
                event.get(field) != target.get(field)
                for field in ("path", "start_line", "end_line")
            ):
                return False
            if is_format_recovery_event(event):
                if work_key in accounted_work:
                    return False
                accounted_work.add(work_key)
            elif (
                event.get("outcome") != LedgerOutcome.COMPLETED
                or event.get("reason_code") is not None
            ):
                return False
        if (
            sum(
                1
                for call in calls
                if isinstance(call, Mapping)
                and _source_scope(call) == scope
                and call.get("node") == analyzer
                and successful_llm_record(call)
            )
            != 1
        ):
            return False
        accounted_scopes.add(key)
        projected_statuses.append({**status, "status": "completed"})
    if accounted_work != warning_work or accounted_scopes != warning_scopes:
        return False
    _used, complete = semantic_runtime_accounting(
        enabled=True,
        result={**result, "analyzer_status_events": projected_statuses},
        discovered_modules=discovered_modules,
    )
    return complete


def _refused_runtime_is_accounted(
    result: Mapping[str, object],
    discovered_modules: Mapping[str, object],
) -> bool:
    """Distinguish reported refusals from missing execution evidence."""
    events = result.get("inspection_ledger")
    statuses = result.get("analyzer_status_events")
    calls = result.get("llm_call_log")
    if not all(isinstance(value, list) for value in (events, statuses, calls)):
        return False
    assert isinstance(events, list) and isinstance(statuses, list) and isinstance(calls, list)
    by_work: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    refusals: set[tuple[str, str]] = set()
    refused_nodes: set[tuple[str, str]] = set()
    for event in events:
        if not isinstance(event, Mapping):
            return False
        scope, work_id = _source_scope(event), event.get("work_id")
        if scope is None or not isinstance(work_id, str) or not work_id:
            return False
        key = (scope, work_id)
        by_work.setdefault(key, []).append(event)
        if not isinstance(event.get("phase"), str):
            return False
        if (
            event.get("phase") in {"semantic", "meta"}
            and event.get("outcome") != LedgerOutcome.COMPLETED
        ):
            if (
                event.get("record_type") != LedgerRecordType.WORK_ITEM
                or event.get("phase") != "semantic"
                or event.get("outcome") != LedgerOutcome.SKIPPED
                or event.get("reason_code") != LedgerReason.LLM_RESPONSE_REFUSED
            ):
                return False
            analyzer = event.get("analyzer_id")
            if not isinstance(analyzer, str) or not analyzer:
                return False
            refusals.add(key)
            refused_nodes.add((scope, analyzer))
    if not refusals:
        return False

    projected_statuses: list[Mapping[str, object]] = []
    accounted_work: set[tuple[str, str]] = set()
    accounted_refusals: set[tuple[str, str]] = set()
    for status in statuses:
        if not isinstance(status, Mapping):
            return False
        scope, analyzer = _source_scope(status), status.get("analyzer_id")
        if scope is None or not isinstance(analyzer, str) or not analyzer:
            return False
        if (scope, analyzer) not in refused_nodes:
            projected_statuses.append(status)
            continue
        targets = status.get("planned_work")
        if status.get("status") != "degraded" or not isinstance(targets, list) or not targets:
            return False
        node_refusals: set[tuple[str, str]] = set()
        for target in targets:
            if not isinstance(target, Mapping):
                return False
            work_id, path = target.get("work_id"), target.get("path")
            if not isinstance(work_id, str) or not work_id or not isinstance(path, str) or not path:
                return False
            start, end = target.get("start_line"), target.get("end_line")
            if (start is not None and (type(start) is not int or start < 1)) or (
                end is not None and (type(end) is not int or start is None or end < start)
            ):
                return False
            key = (scope, work_id)
            matches = by_work.get(key, [])
            if key in accounted_work or len(matches) != 1:
                return False
            event = matches[0]
            if (
                event.get("record_type") != LedgerRecordType.WORK_ITEM
                or event.get("analyzer_id") != analyzer
                or event.get("phase") != "semantic"
                or any(
                    event.get(field) != target.get(field)
                    for field in ("path", "start_line", "end_line")
                )
                or not (
                    key in refusals
                    or (
                        event.get("outcome") == LedgerOutcome.COMPLETED
                        and event.get("reason_code") is None
                    )
                )
            ):
                return False
            accounted_work.add(key)
            if key in refusals:
                node_refusals.add(key)
        if not node_refusals:
            return False
        accounted_refusals.update(node_refusals)
        projected_statuses.append({**status, "status": "completed"})
    if accounted_refusals != refusals:
        return False

    projected_calls: list[object] = []
    accounted_nodes: set[tuple[str, str]] = set()
    for call in calls:
        if not isinstance(call, Mapping):
            return False
        scope, node = _source_scope(call), call.get("node")
        if scope is None or not isinstance(node, str) or not node:
            return False
        key = (scope, node)
        if call.get("ok") is False and key in refused_nodes:
            # Refusal batch outcomes have no transport error. Other failures
            # must retain their missing-runtime warning.
            if "error" not in call or call["error"] is not None:
                return False
            accounted_nodes.add(key)
            projected_calls.append({**call, "ok": True})
        else:
            projected_calls.append(call)
    if accounted_nodes != refused_nodes:
        return False
    _used, complete = semantic_runtime_accounting(
        enabled=True,
        result={
            **result,
            "analyzer_status_events": projected_statuses,
            "llm_call_log": projected_calls,
        },
        discovered_modules=discovered_modules,
    )
    return complete


def semantic_runtime_limitation(
    *,
    requested: bool,
    enabled: bool,
    result: Mapping[str, object],
    discovered_modules: Mapping[str, object],
) -> str | None:
    """Return the canonical limitation for an unmet requested semantic pass."""
    if not requested:
        return None
    if not enabled:
        return SEMANTIC_PREFLIGHT_UNAVAILABLE_MESSAGE
    _used, complete = semantic_runtime_accounting(
        enabled=True,
        result=result,
        discovered_modules=discovered_modules,
    )
    return None if complete else SEMANTIC_RUNTIME_INCOMPLETE_MESSAGE


def semantic_runtime_ledger_event(
    *,
    requested: bool,
    enabled: bool,
    result: Mapping[str, object],
    discovered_modules: Mapping[str, object],
) -> InspectionLedgerEvent | None:
    """Project an unmet semantic requirement into the canonical inspection ledger."""
    # Finalization is also used by focused analyzers and compatibility callers
    # that predate semantic intent telemetry.  Only graph states that explicitly
    # declare that intent can owe semantic work.
    if "use_llm" not in result and "llm_requested" not in result:
        return None
    limitation = semantic_runtime_limitation(
        requested=requested,
        enabled=enabled,
        result=result,
        discovered_modules=discovered_modules,
    )
    if limitation is None:
        return None
    if enabled and _refused_runtime_is_accounted(result, discovered_modules):
        return None
    event = ledger_event(
        outcome=LedgerOutcome.PARTIAL,
        record_type=LedgerRecordType.SYSTEM,
        phase="semantic_runtime",
        path="SKILL.md",
        reason=(
            LedgerReason.SEMANTIC_RUNTIME_INCOMPLETE
            if enabled
            else LedgerReason.MISSING_CREDENTIALS
        ),
        stage="runtime_telemetry" if enabled else "preflight",
    )
    event["message"] = limitation
    return event


def semantic_runtime_event_key(event: Mapping[str, object]) -> tuple[str, str, str, str]:
    """Return the stable identity used to deduplicate semantic runtime gaps."""
    return (
        str(event.get("work_id", "")),
        str(event.get("reason_code", "")),
        str(event.get("stage", "")),
        str(event.get("source_identity", _ROOT_SOURCE_SCOPE)),
    )


def has_semantic_runtime_event(events: Iterable[object], candidate: Mapping[str, object]) -> bool:
    """Return whether ``events`` already contain this exact runtime limitation."""
    candidate_key = semantic_runtime_event_key(candidate)
    return any(
        isinstance(event, Mapping)
        and event.get("phase") == "semantic_runtime"
        and semantic_runtime_event_key(event) == candidate_key
        for event in events
    )
