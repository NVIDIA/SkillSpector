# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-independent semantic analyzer runtime accounting."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from skillspector.inspection_ledger import (
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    LedgerRecordType,
    ledger_event,
)

CANONICAL_SEMANTIC_ANALYZER_IDS = frozenset(
    {
        "semantic_developer_intent",
        "semantic_quality_policy",
        "semantic_security_discovery",
    }
)

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
