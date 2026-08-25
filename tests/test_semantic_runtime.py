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
