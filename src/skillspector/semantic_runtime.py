# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-independent semantic analyzer runtime accounting."""

from __future__ import annotations

from collections.abc import Mapping

CANONICAL_SEMANTIC_ANALYZER_IDS = frozenset(
    {
        "semantic_developer_intent",
        "semantic_quality_policy",
        "semantic_security_discovery",
    }
)


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


def successful_llm_record(record: object) -> bool:
    """Return whether ``record`` is a well-formed successful LLM call."""
    return (
        isinstance(record, Mapping)
        and isinstance(record.get("node"), str)
        and bool(record.get("node"))
        and record.get("ok") is True
        and record.get("error") is None
    )


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

    raw_statuses = result.get("analyzer_status_events")
    if not isinstance(raw_statuses, list):
        return used, False
    required_analyzer_ids = required_semantic_analyzer_ids(discovered_modules)
    statuses_by_analyzer: dict[str, list[str]] = {}
    for status in raw_statuses:
        if not isinstance(status, Mapping):
            return used, False
        analyzer_id = status.get("analyzer_id")
        analyzer_status = status.get("status")
        if (
            not isinstance(analyzer_id, str)
            or not analyzer_id
            or not isinstance(analyzer_status, str)
            or not analyzer_status
        ):
            return used, False
        if analyzer_id in required_analyzer_ids:
            statuses_by_analyzer.setdefault(analyzer_id, []).append(analyzer_status)

    for analyzer_id in required_analyzer_ids:
        statuses = statuses_by_analyzer.get(analyzer_id)
        if statuses is None or len(statuses) != 1:
            return used, False
        status = statuses[0]
        has_successful_call = any(
            successful_llm_record(record) and record.get("node") == analyzer_id
            for record in raw_call_log
        )
        if status == "completed":
            if not has_successful_call:
                return used, False
        elif status == "not_applicable":
            if has_successful_call:
                return used, False
        else:
            return used, False

    if _has_effective_findings(result) and not any(
        successful_llm_record(record) and record.get("node") == "meta_analyzer"
        for record in raw_call_log
    ):
        return used, False

    return used, True
