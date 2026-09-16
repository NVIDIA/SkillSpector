# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for provider-independent semantic runtime accounting."""

from __future__ import annotations

from skillspector.semantic_runtime import (
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
