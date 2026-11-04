# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for provider-independent semantic runtime accounting."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from skillspector.inspection_ledger import (
    LedgerOutcome,
    LedgerReason,
    analyzer_status_for_events,
    ledger_event,
)
from skillspector.semantic_runtime import (
    format_recovery_runtime_accepted,
    required_semantic_analyzer_ids,
    semantic_runtime_accounting,
    successful_llm_record,
)


def test_empty_discovery_registry_cannot_shrink_canonical_semantic_requirements() -> None:
    """An import failure cannot erase a required semantic completion check."""
    assert required_semantic_analyzer_ids({}) == frozenset(
        {
            "semantic_developer_intent",
            "semantic_quality_policy",
            "semantic_security_discovery",
        }
    )


def test_successful_llm_record_requires_strict_well_formed_evidence() -> None:
    """Truthy substitutes and errored records cannot prove a successful call."""
    assert successful_llm_record({"node": "meta_analyzer", "ok": True, "error": None})
    assert not successful_llm_record({"node": "meta_analyzer", "ok": "false", "error": None})
    assert not successful_llm_record({"node": "", "ok": True, "error": None})
    assert not successful_llm_record(
        {"node": "meta_analyzer", "ok": True, "error": "runtime failure"}
    )


def test_discovered_api_key_analyzers_extend_canonical_semantic_requirements() -> None:
    """Future credential-gated analyzers automatically join the required set."""

    class _FutureSemanticAnalyzer:
        requires_api_key = True

    class _StaticAnalyzer:
        requires_api_key = False

    discovered = {
        "semantic_future_policy": _FutureSemanticAnalyzer(),
        "static_example": _StaticAnalyzer(),
    }

    assert required_semantic_analyzer_ids(discovered) == frozenset(
        {
            "semantic_developer_intent",
            "semantic_future_policy",
            "semantic_quality_policy",
            "semantic_security_discovery",
        }
    )


def test_incomplete_registry_cannot_make_incomplete_canonical_telemetry_complete() -> None:
    """Runtime accounting still requires canonical analyzers absent from discovery."""
    result = {
        "llm_call_log": [],
        "analyzer_status_events": [
            {"analyzer_id": "semantic_developer_intent", "status": "not_applicable"},
            {"analyzer_id": "semantic_quality_policy", "status": "not_applicable"},
        ],
    }

    assert semantic_runtime_accounting(
        enabled=True,
        result=result,
        discovered_modules={},
    ) == (False, False)


def _semantic_statuses(source_identity: str | None = None) -> list[dict[str, object]]:
    provenance = {"source_identity": source_identity} if source_identity is not None else {}
    return [
        {"analyzer_id": analyzer_id, "status": "completed", **provenance}
        for analyzer_id in sorted(required_semantic_analyzer_ids({}))
    ]


def _semantic_calls(source_identity: str | None = None) -> list[dict[str, object]]:
    provenance = {"source_identity": source_identity} if source_identity is not None else {}
    return [
        {"node": analyzer_id, "ok": True, "error": None, **provenance}
        for analyzer_id in sorted(required_semantic_analyzer_ids({}))
    ]


def test_legacy_coverage_only_findings_do_not_require_meta_call() -> None:
    """Legacy results without the flag must ignore post-review coverage findings."""
    result = {
        "findings": [SimpleNamespace(tags=["coverage"])],
        "analyzer_status_events": _semantic_statuses(),
        "llm_call_log": _semantic_calls(),
    }

    assert semantic_runtime_accounting(
        enabled=True,
        result=result,
        discovered_modules={},
    ) == (True, True)


def test_complete_root_and_child_telemetry_is_validated_per_source_scope() -> None:
    """Identical analyzer IDs in independent complete sources are not duplicates."""
    child_scope = "external/child-digest"
    result = {
        "analyzer_status_events": [
            *_semantic_statuses(),
            *_semantic_statuses(child_scope),
        ],
        "llm_call_log": [
            *_semantic_calls(),
            *_semantic_calls(child_scope),
        ],
    }

    assert semantic_runtime_accounting(
        enabled=True,
        result=result,
        discovered_modules={},
    ) == (True, True)


def test_duplicate_semantic_status_within_one_source_scope_is_rejected() -> None:
    """Source scoping must not weaken duplicate-within-scope detection."""
    statuses = _semantic_statuses()
    statuses.append(dict(statuses[0]))

    assert semantic_runtime_accounting(
        enabled=True,
        result={"analyzer_status_events": statuses, "llm_call_log": _semantic_calls()},
        discovered_modules={},
    ) == (True, False)


def test_child_call_cannot_borrow_an_unscoped_root_status() -> None:
    """Call evidence and terminal status must carry the same structural scope key."""
    child_scope = "external/child-digest"

    assert semantic_runtime_accounting(
        enabled=True,
        result={
            "analyzer_status_events": _semantic_statuses(child_scope),
            "llm_call_log": _semantic_calls(),
        },
        discovered_modules={},
    ) == (True, False)


def _format_recovery_state() -> dict:
    analyzer = "semantic_security_discovery"
    event = ledger_event(
        analyzer_id=analyzer,
        outcome=LedgerOutcome.PARTIAL,
        phase="semantic",
        path="SKILL.md",
        reason=LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED,
    )
    statuses = [status for status in _semantic_statuses() if status["analyzer_id"] != analyzer]
    return {
        "use_llm": True,
        "inspection_ledger": [event],
        "analyzer_status_events": [*statuses, analyzer_status_for_events(analyzer, [event])],
        "llm_call_log": _semantic_calls(),
    }


def test_format_policy_acceptance_does_not_claim_semantic_completion() -> None:
    state = _format_recovery_state()
    assert format_recovery_runtime_accepted(state, {})
    assert semantic_runtime_accounting(enabled=True, result=state, discovered_modules={}) == (
        True,
        False,
    )
    assert state["analyzer_status_events"][-1]["status"] == "degraded"


def test_format_recovery_remains_bound_to_its_transitive_source() -> None:
    state = _format_recovery_state()
    source = "external/child-digest"
    for event in state["inspection_ledger"]:
        event["source_identity"] = source
        event["path"] = source + "/" + event["path"]
    for status in state["analyzer_status_events"]:
        status["source_identity"] = source
        for target in status.get("planned_work", []):
            target["path"] = source + "/" + target["path"]
    for call in state["llm_call_log"]:
        call["source_identity"] = source
    state["analyzer_status_events"].extend(_semantic_statuses())
    state["llm_call_log"].extend(_semantic_calls())
    assert format_recovery_runtime_accepted(state, {})
    state["inspection_ledger"][0]["source_identity"] = "external/different"
    assert not format_recovery_runtime_accepted(state, {})


@pytest.mark.parametrize("source", [None, "external/child-digest"])
@pytest.mark.parametrize(
    "mutation",
    [None, "other_analyzer", "wrong_phase", "wrong_path", "partial", "failed", "reason"],
)
def test_tp4_warning_accepts_only_its_completed_static_phase(source, mutation) -> None:
    path = "SKILL.md" if source is None else source + "/SKILL.md"
    static = ledger_event(
        analyzer_id="mcp_tool_poisoning_static",
        outcome=LedgerOutcome.COMPLETED,
        phase="static",
        path=path,
    )
    semantic = ledger_event(
        analyzer_id="mcp_tool_poisoning",
        outcome=LedgerOutcome.PARTIAL,
        phase="semantic",
        path=path,
        reason=LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED,
    )
    if mutation == "other_analyzer":
        static["analyzer_id"] = "unrelated_static"
    elif mutation == "wrong_phase":
        static["phase"] = "semantic"
    elif mutation == "wrong_path":
        static["path"] = path.replace("SKILL.md", "other.py")
    elif mutation in {"partial", "failed"}:
        static["outcome"] = LedgerOutcome(mutation)
    elif mutation == "reason":
        static["reason_code"] = LedgerReason.RUNTIME_LIMIT
    events = [static, semantic]
    status = analyzer_status_for_events("mcp_tool_poisoning", events)
    call = {"node": "mcp_tool_poisoning", "ok": True, "error": None}
    if source is not None:
        for record in [*events, status, call]:
            record["source_identity"] = source
    state = {
        "use_llm": True,
        "inspection_ledger": events,
        "analyzer_status_events": [*_semantic_statuses(source), status],
        "llm_call_log": [*_semantic_calls(source), call],
    }
    assert format_recovery_runtime_accepted(state, {}) is (mutation is None)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_status",
        "missing_call",
        "failed_call",
        "other_source_call",
        "wrong_path",
        "duplicate_event",
        "duplicate_target",
        "wrong_reason",
        "wrong_phase",
        "wrong_status",
        "unknown_analyzer",
        "disabled",
    ],
)
def test_format_policy_cannot_waive_unaccounted_or_unrelated_work(mutation: str) -> None:
    state = _format_recovery_state()
    event = state["inspection_ledger"][0]
    status = state["analyzer_status_events"][-1]
    call = next(call for call in state["llm_call_log"] if call["node"] == event["analyzer_id"])
    if mutation == "missing_status":
        state["analyzer_status_events"].pop(0)
    elif mutation == "missing_call":
        state["llm_call_log"].remove(call)
    elif mutation == "failed_call":
        call["ok"] = False
    elif mutation == "other_source_call":
        call["source_identity"] = "external/other"
    elif mutation == "wrong_path":
        event["path"] = "different.py"
    elif mutation == "duplicate_event":
        state["inspection_ledger"].append(dict(event))
    elif mutation == "duplicate_target":
        status["planned_work"].append(dict(status["planned_work"][0]))
    elif mutation == "wrong_reason":
        event["reason_code"] = LedgerReason.LLM_RESPONSE_REFUSED
    elif mutation == "wrong_phase":
        event["phase"] = "static"
    elif mutation == "wrong_status":
        status["status"] = "completed"
    elif mutation == "unknown_analyzer":
        event["analyzer_id"] = "static_patterns"
    else:
        state["use_llm"] = False
    assert not format_recovery_runtime_accepted(state, {})
