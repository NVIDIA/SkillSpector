# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for provider-independent semantic runtime accounting."""

from __future__ import annotations

from copy import deepcopy
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
    semantic_runtime_ledger_event,
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


def _refused_semantic_state() -> dict:
    events = []
    statuses = []
    calls = []
    for analyzer in sorted(required_semantic_analyzer_ids({})):
        own_events = [
            ledger_event(
                analyzer_id=analyzer,
                outcome=LedgerOutcome.COMPLETED,
                phase="semantic",
                path="SKILL.md",
            ),
            ledger_event(
                analyzer_id=analyzer,
                outcome=LedgerOutcome.SKIPPED,
                phase="semantic",
                path="rules.yaml",
                reason=LedgerReason.LLM_RESPONSE_REFUSED,
            ),
        ]
        events.extend(own_events)
        statuses.append(analyzer_status_for_events(analyzer, own_events))
        calls.append({"node": analyzer, "ok": False, "error": None})
    return {
        "use_llm": True,
        "inspection_ledger": events,
        "analyzer_status_events": statuses,
        "llm_call_log": calls,
    }


def _runtime_event(state: dict, discovered: dict | None = None):
    return semantic_runtime_ledger_event(
        requested=True, enabled=True, result=state, discovered_modules=discovered or {}
    )


def test_accounted_refusals_do_not_claim_missing_runtime_telemetry() -> None:
    state = _refused_semantic_state()
    before = deepcopy(state)

    assert _runtime_event(state) is None
    assert state == before
    assert semantic_runtime_accounting(enabled=True, result=state, discovered_modules={}) == (
        False,
        False,
    )
    assert all(status["status"] == "degraded" for status in state["analyzer_status_events"])
    assert (
        sum(
            event.get("reason_code") == LedgerReason.LLM_RESPONSE_REFUSED
            for event in state["inspection_ledger"]
        )
        == 3
    )


def test_accounted_refusal_keeps_findings_and_requires_actual_meta_execution() -> None:
    state = _refused_semantic_state()
    state["findings"] = [{"id": "TM1", "severity": "HIGH", "tags": []}]
    state["meta_review_required"] = True
    assert _runtime_event(state) is not None
    state["llm_call_log"].append({"node": "meta_analyzer", "ok": True, "error": None})
    assert _runtime_event(state) is None
    assert state["findings"][0]["severity"] == "HIGH"
    assert semantic_runtime_accounting(enabled=True, result=state, discovered_modules={}) == (
        True,
        False,
    )


def test_accounted_refusal_is_validated_independently_in_each_source() -> None:
    state = _refused_semantic_state()
    source = "external/child-digest"
    for event in state["inspection_ledger"]:
        event["source_identity"] = source
        event["path"] = source + "/" + event["path"]
    for status in state["analyzer_status_events"]:
        status["source_identity"] = source
        for target in status["planned_work"]:
            target["path"] = source + "/" + target["path"]
    for call in state["llm_call_log"]:
        call["source_identity"] = source
    state["analyzer_status_events"].extend(_semantic_statuses())
    state["llm_call_log"].extend(_semantic_calls())
    assert _runtime_event(state) is None
    state["llm_call_log"][0].pop("source_identity")
    assert _runtime_event(state) is not None


def test_one_refusing_analyzer_does_not_hide_other_completed_assessments() -> None:
    state = _refused_semantic_state()
    refused = state["analyzer_status_events"][-1]["analyzer_id"]
    for event in state["inspection_ledger"]:
        if event["analyzer_id"] != refused:
            event["outcome"] = LedgerOutcome.COMPLETED
            event.pop("reason_code", None)
            event.pop("message", None)
    state["analyzer_status_events"] = [
        analyzer_status_for_events(
            analyzer,
            [event for event in state["inspection_ledger"] if event["analyzer_id"] == analyzer],
        )
        for analyzer in sorted(required_semantic_analyzer_ids({}))
    ]
    for call in state["llm_call_log"]:
        call["ok"] = call["node"] != refused
    assert _runtime_event(state) is None
    assert semantic_runtime_accounting(enabled=True, result=state, discovered_modules={}) == (
        True,
        False,
    )


def test_static_limit_does_not_become_a_missing_semantic_telemetry_warning() -> None:
    state = _refused_semantic_state()
    event = ledger_event(
        analyzer_id="static_patterns_tool_misuse",
        outcome=LedgerOutcome.PARTIAL,
        phase="static",
        path="rules.yaml",
        reason=LedgerReason.STATIC_PARSE_LIMIT,
    )
    state["inspection_ledger"].append(event)
    state["analyzer_status_events"].append(
        analyzer_status_for_events("static_patterns_tool_misuse", [event])
    )
    assert _runtime_event(state) is None
    assert event["reason_code"] == LedgerReason.STATIC_PARSE_LIMIT


def test_refusal_accounting_includes_new_required_analyzers() -> None:
    state = _refused_semantic_state()
    analyzer = "semantic_future_policy"
    event = ledger_event(
        analyzer_id=analyzer,
        outcome=LedgerOutcome.SKIPPED,
        phase="semantic",
        path="rules.yaml",
        reason=LedgerReason.LLM_RESPONSE_REFUSED,
    )
    state["inspection_ledger"].append(event)
    state["analyzer_status_events"].append(analyzer_status_for_events(analyzer, [event]))
    state["llm_call_log"].append({"node": analyzer, "ok": False, "error": None})
    discovered = {analyzer: SimpleNamespace(requires_api_key=True)}
    assert _runtime_event(state, discovered) is None
    state["llm_call_log"].pop()
    assert _runtime_event(state, discovered) is not None


@pytest.mark.parametrize("analyzer", ["semantic_unregistered", "static_patterns_tool_misuse"])
def test_refusal_cannot_promote_an_unrecognized_analyzer(analyzer: str) -> None:
    state = _refused_semantic_state()
    event = ledger_event(
        analyzer_id=analyzer,
        outcome=LedgerOutcome.SKIPPED,
        phase="semantic",
        path="rules.yaml",
        reason=LedgerReason.LLM_RESPONSE_REFUSED,
    )
    state["inspection_ledger"].append(event)
    state["analyzer_status_events"].append(analyzer_status_for_events(analyzer, [event]))
    state["llm_call_log"].append({"node": analyzer, "ok": False, "error": None})
    assert _runtime_event(state) is not None


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_event",
        "duplicate_event",
        "missing_call",
        "duplicate_call",
        "missing_status",
        "duplicate_status",
        "wrong_source",
        "wrong_analyzer",
        "wrong_path",
        "wrong_range",
        "wrong_phase",
        "provider_failure",
        "unknown_failure",
        "unplanned_refusal",
        "malformed_event",
        "malformed_phase",
        "malformed_call",
        "missing_error",
        "transport_error",
        "duplicate_target",
        "invalid_range",
        "invalid_end",
        "missing_required",
    ],
)
def test_accounted_refusal_cannot_hide_unproven_execution(mutation: str) -> None:
    state = _refused_semantic_state()
    events, statuses, calls = (
        state["inspection_ledger"],
        state["analyzer_status_events"],
        state["llm_call_log"],
    )
    if mutation == "missing_event":
        events.pop()
    elif mutation == "duplicate_event":
        events.append(dict(events[-1]))
    elif mutation == "missing_call":
        calls.pop()
    elif mutation == "duplicate_call":
        calls.append(dict(calls[0]))
    elif mutation == "missing_status":
        statuses.pop()
    elif mutation == "duplicate_status":
        statuses.append(deepcopy(statuses[0]))
    elif mutation == "wrong_source":
        events[-1]["source_identity"] = "external/other"
    elif mutation == "wrong_analyzer":
        events[-1]["analyzer_id"] = "another_analyzer"
    elif mutation == "wrong_path":
        events[-1]["path"] = "another.yaml"
    elif mutation == "wrong_range":
        events[-1]["start_line"] = 2
    elif mutation == "wrong_phase":
        events[-1]["phase"] = "static"
    elif mutation == "provider_failure":
        events[-1]["reason_code"] = LedgerReason.LLM_CONNECTION_RETRIES_EXHAUSTED
        events[-1]["outcome"] = LedgerOutcome.FAILED
    elif mutation == "unknown_failure":
        events[-1]["reason_code"] = LedgerReason.LLM_STRUCTURED_RESPONSE_INVALID
    elif mutation == "unplanned_refusal":
        events.append(
            ledger_event(
                analyzer_id="semantic_security_discovery",
                outcome=LedgerOutcome.SKIPPED,
                phase="semantic",
                path="other.yaml",
                reason=LedgerReason.LLM_RESPONSE_REFUSED,
            )
        )
    elif mutation == "malformed_event":
        events[-1] = None
    elif mutation == "malformed_phase":
        events[-1]["phase"] = []
    elif mutation == "malformed_call":
        calls[-1]["node"] = []
    elif mutation == "missing_error":
        calls[-1].pop("error")
    elif mutation == "transport_error":
        calls[-1]["error"] = "connection failed"
    elif mutation == "duplicate_target":
        statuses[-1]["planned_work"].append(dict(statuses[-1]["planned_work"][0]))
    elif mutation == "invalid_range":
        events[-1]["start_line"] = statuses[-1]["planned_work"][-1]["start_line"] = False
    elif mutation == "invalid_end":
        events[-1]["end_line"] = statuses[-1]["planned_work"][-1]["end_line"] = 2
    elif mutation == "missing_required":
        analyzer = statuses[-1]["analyzer_id"]
        state["inspection_ledger"] = [event for event in events if event["analyzer_id"] != analyzer]
        statuses.pop()
        calls.pop()
    assert _runtime_event(state) is not None


def _refused_tp4_meta_state() -> dict:
    state = _refused_semantic_state()
    for analyzer in ("mcp_tool_poisoning", "meta_analyzer"):
        phase = "meta" if analyzer == "meta_analyzer" else "semantic"
        own_events = [
            ledger_event(
                analyzer_id=analyzer,
                outcome=LedgerOutcome.COMPLETED,
                phase=phase,
                path="complete.py",
            ),
            ledger_event(
                analyzer_id=analyzer,
                outcome=LedgerOutcome.SKIPPED,
                phase=phase,
                path="rules.yaml",
                reason=LedgerReason.LLM_RESPONSE_REFUSED,
            ),
        ]
        if analyzer == "mcp_tool_poisoning":
            own_events.insert(
                0,
                ledger_event(
                    analyzer_id="mcp_tool_poisoning_static",
                    outcome=LedgerOutcome.COMPLETED,
                    phase="static",
                    path="SKILL.md",
                ),
            )
        state["inspection_ledger"].extend(own_events)
        state["analyzer_status_events"].append(analyzer_status_for_events(analyzer, own_events))
        state["llm_call_log"].append({"node": analyzer, "ok": False, "error": None})
    state["findings"] = [{"id": "TM1", "severity": "HIGH", "tags": []}]
    state["meta_review_required"] = True
    return state


def test_bound_tp4_and_meta_refusals_retain_original_failure_and_findings() -> None:
    state = _refused_tp4_meta_state()
    before = deepcopy(state)
    assert _runtime_event(state) is None
    assert state == before
    assert semantic_runtime_accounting(enabled=True, result=state, discovered_modules={}) == (
        False,
        False,
    )
    assert state["findings"][0]["severity"] == "HIGH"
    assert all(call["ok"] is False for call in state["llm_call_log"])
    assert all(status["status"] == "degraded" for status in state["analyzer_status_events"])


def test_bound_tp4_and_meta_refusals_require_matching_child_scope() -> None:
    state = _refused_tp4_meta_state()
    scope = "external/child-digest"
    for event in state["inspection_ledger"]:
        event["source_identity"] = scope
        event["path"] = scope + "/" + event["path"]
    for status in state["analyzer_status_events"]:
        status["source_identity"] = scope
        for target in status["planned_work"]:
            target["path"] = scope + "/" + target["path"]
    for call in state["llm_call_log"]:
        call["source_identity"] = scope
    assert _runtime_event(state) is None
    state["llm_call_log"][-1].pop("source_identity")
    assert _runtime_event(state) is not None


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_meta_call",
        "duplicate_meta_call",
        "contradictory_meta_call",
        "missing_meta_event",
        "duplicate_meta_event",
        "missing_meta_status",
        "duplicate_meta_status",
        "duplicate_tp4_target",
        "tp4_static_path",
        "tp4_static_failure",
        "meta_phase",
        "tp4_phase",
        "provider_failure",
        "missing_error",
        "mixed_provider_error",
        "unknown_analyzer",
    ],
)
def test_tp4_and_meta_refusal_proof_rejects_incomplete_or_contradictory_work(mutation: str) -> None:
    state = _refused_tp4_meta_state()
    events, statuses, calls = (
        state["inspection_ledger"],
        state["analyzer_status_events"],
        state["llm_call_log"],
    )
    if mutation == "missing_meta_call":
        calls.pop()
    elif mutation == "duplicate_meta_call":
        calls.append(dict(calls[-1]))
    elif mutation == "contradictory_meta_call":
        calls.append({"node": "meta_analyzer", "ok": True, "error": None})
    elif mutation == "missing_meta_event":
        events.pop()
    elif mutation == "duplicate_meta_event":
        events.append(dict(events[-1]))
    elif mutation == "missing_meta_status":
        statuses.pop()
    elif mutation == "duplicate_meta_status":
        statuses.append(deepcopy(statuses[-1]))
    elif mutation == "duplicate_tp4_target":
        statuses[-2]["planned_work"].append(dict(statuses[-2]["planned_work"][0]))
    elif mutation == "tp4_static_path":
        static = next(event for event in events if event["phase"] == "static")
        static["path"] = statuses[-2]["planned_work"][0]["path"] = "wrong.md"
    elif mutation == "tp4_static_failure":
        static = next(event for event in events if event["phase"] == "static")
        static["outcome"] = LedgerOutcome.PARTIAL
        static["reason_code"] = LedgerReason.RUNTIME_LIMIT
    elif mutation == "meta_phase":
        events[-1]["phase"] = "semantic"
    elif mutation == "tp4_phase":
        events[-3]["phase"] = "meta"
    elif mutation == "provider_failure":
        events[-1]["outcome"] = LedgerOutcome.FAILED
        events[-1]["reason_code"] = LedgerReason.LLM_CONNECTION_RETRIES_EXHAUSTED
    elif mutation == "missing_error":
        calls[-1].pop("error")
    elif mutation == "mixed_provider_error":
        calls[-2]["error"] = "connection failed"
    elif mutation == "unknown_analyzer":
        events[-1]["analyzer_id"] = "static_patterns_tool_misuse"
    before = deepcopy(state)
    assert _runtime_event(state) is not None
    assert state == before


@pytest.mark.parametrize("analyzer", ["mcp_tool_poisoning", "meta_analyzer"])
def test_refusal_status_cannot_split_disjoint_work_across_duplicate_nodes(analyzer: str) -> None:
    state = _refused_tp4_meta_state()
    extra = ledger_event(
        analyzer_id=analyzer,
        outcome=LedgerOutcome.SKIPPED,
        phase="meta" if analyzer == "meta_analyzer" else "semantic",
        path="second-refusal.yaml",
        reason=LedgerReason.LLM_RESPONSE_REFUSED,
    )
    extra_status = analyzer_status_for_events(analyzer, [extra])
    original = next(s for s in state["analyzer_status_events"] if s["analyzer_id"] == analyzer)
    state["inspection_ledger"].append(extra)
    original["planned_work"].extend(extra_status["planned_work"])
    assert _runtime_event(state) is None
    original["planned_work"].pop()
    state["analyzer_status_events"].append(extra_status)
    assert not {t["work_id"] for t in original["planned_work"]} & {
        t["work_id"] for t in extra_status["planned_work"]
    }
    before = deepcopy(state)
    assert _runtime_event(state) is not None
    assert state == before


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
