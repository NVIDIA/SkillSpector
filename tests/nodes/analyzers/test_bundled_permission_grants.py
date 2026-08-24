# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for bundled project permission mode analysis."""

from __future__ import annotations

import re
from dataclasses import FrozenInstanceError

import pytest

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers.bundled_permission_grants import (
    PermissionAnalysis,
    PermissionSourceLines,
    analyze_permission_grants,
    build_bh3_finding,
)


def _analyze(permissions: object, *, source_kind: str = "project_settings") -> PermissionAnalysis:
    return analyze_permission_grants(
        {"permissions": permissions},
        source_kind=source_kind,
        content_digest="sha256:" + "1" * 64,
        source_identity_digest="sha256:" + "2" * 64,
        source_lines=PermissionSourceLines(
            permissions_line=2,
            default_mode_line=3,
            disable_bypass_line=4,
            disable_auto_line=5,
            skip_dangerous_prompt_line=6,
        ),
    )


def test_mapping_without_permissions_is_not_applicable() -> None:
    result = analyze_permission_grants(
        {"env": {"SAFE": "1"}},
        source_kind="project_settings",
        content_digest="sha256:" + "1" * 64,
        source_identity_digest="sha256:" + "2" * 64,
        source_lines=PermissionSourceLines(),
    )

    assert result == PermissionAnalysis(False, None, None, (), (), None)
    assert build_bh3_finding(result, source_path=".claude/settings.json") is None


def test_non_object_permissions_fail_without_grants() -> None:
    result = _analyze(["not", "an", "object"])

    assert result.applicable is True
    assert result.outcome is LedgerOutcome.FAILED
    assert result.reason is LedgerReason.INVALID_CONFIGURATION
    assert result.grants == ()
    assert build_bh3_finding(result, source_path=".claude/settings.json") is None


@pytest.mark.parametrize(
    "permissions",
    ({}, {"allow": []}, {"ask": []}, {"deny": []}, {"additionalDirectories": []}),
)
def test_empty_permissions_and_recognized_empty_arrays_are_completed(permissions: object) -> None:
    result = _analyze(permissions)

    assert result.outcome is LedgerOutcome.COMPLETED
    assert result.reason is None
    assert result.grants == ()


def test_records_are_frozen() -> None:
    result = _analyze({"defaultMode": "acceptEdits"})

    with pytest.raises(FrozenInstanceError):
        result.applicable = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mode", "diagnostic"),
    [
        ("default", None),
        ("plan", None),
        ("dontAsk", None),
        ("manual", "legacy_manual"),
        ("auto", "auto_ignored"),
    ],
)
def test_non_grant_modes_are_completed(mode: str, diagnostic: str | None) -> None:
    result = _analyze({"defaultMode": mode})

    assert result.outcome is LedgerOutcome.COMPLETED
    assert result.grants == ()
    assert [item.diagnostic_kind for item in result.diagnostics] == (
        [] if diagnostic is None else [diagnostic]
    )
    assert build_bh3_finding(result, source_path=".claude/settings.json") is None


@pytest.mark.parametrize(
    ("source_kind", "tracking_status"),
    [("project_settings", "not_applicable"), ("project_local_settings", "unknown")],
)
def test_bypass_mode_emits_blocking_critical_grant_with_source_context(
    source_kind: str, tracking_status: str
) -> None:
    result = _analyze({"defaultMode": "bypassPermissions"}, source_kind=source_kind)

    assert result.outcome is LedgerOutcome.COMPLETED
    assert len(result.grants) == 1
    grant = result.grants[0]
    assert grant.grant_kind == "permission_mode_bypass"
    assert grant.severity == "CRITICAL"
    assert grant.blocking_critical is True
    assert grant.activation_requirement == "interface_and_external_policy"
    assert grant.interface_applicability == "permission_mode_interface_dependent"
    assert grant.tracking_status == tracking_status

    finding = build_bh3_finding(result, source_path=".claude/settings.json")
    assert finding is not None
    assert finding.rule_id == "BH3"
    assert finding.severity == "CRITICAL"
    assert finding.start_line == 3
    assert finding.evidence["grant_kinds"] == "permission_mode_bypass"
    assert all(isinstance(value, (str, int, bool)) for value in finding.evidence.values())
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", finding.matched_text or "")


def test_accept_edits_mode_emits_non_blocking_medium_grant() -> None:
    result = _analyze({"defaultMode": "acceptEdits"})

    assert result.outcome is LedgerOutcome.COMPLETED
    assert [
        (grant.grant_kind, grant.severity, grant.blocking_critical) for grant in result.grants
    ] == [("permission_mode_accept_edits", "MEDIUM", False)]


@pytest.mark.parametrize("mode", ["delegate", "canary-mode"])
def test_unknown_mode_is_incomplete_and_fails_without_valid_sibling(mode: str) -> None:
    result = _analyze({"defaultMode": mode})

    assert result.outcome is LedgerOutcome.FAILED
    assert result.reason is LedgerReason.INVALID_CONFIGURATION
    assert result.grants == ()
    assert [(item.diagnostic_kind, item.affects_completeness) for item in result.diagnostics] == [
        ("unknown_mode", True)
    ]
    assert mode not in repr(result)


def test_same_document_disable_neutralizes_bypass() -> None:
    result = _analyze(
        {"defaultMode": "bypassPermissions", "disableBypassPermissionsMode": "disable"}
    )

    assert result.outcome is LedgerOutcome.COMPLETED
    assert result.grants == ()
    assert {item.diagnostic_kind for item in result.diagnostics} == {"bypass_disabled"}


def test_disable_before_bypass_also_neutralizes_the_document() -> None:
    result = _analyze(
        {"disableBypassPermissionsMode": "disable", "defaultMode": "bypassPermissions"}
    )

    assert result.outcome is LedgerOutcome.COMPLETED
    assert result.grants == ()
    assert {item.diagnostic_kind for item in result.diagnostics} == {"bypass_disabled"}


@pytest.mark.parametrize("key", ["disableBypassPermissionsMode", "disableAutoMode"])
def test_malformed_disable_controls_are_incomplete(key: str) -> None:
    result = _analyze({key: True})

    assert result.outcome is LedgerOutcome.FAILED
    assert result.reason is LedgerReason.INVALID_CONFIGURATION
    assert [(item.diagnostic_kind, item.affects_completeness) for item in result.diagnostics] == [
        ("wrong_type", True)
    ]


def test_disable_auto_mode_is_recognized() -> None:
    result = _analyze({"disableAutoMode": "disable"})

    assert result.outcome is LedgerOutcome.COMPLETED
    assert [item.diagnostic_kind for item in result.diagnostics] == ["auto_disabled"]


@pytest.mark.parametrize(
    ("source_kind", "diagnostic"),
    [
        ("project_settings", "skip_dangerous_prompt_ignored"),
        ("project_local_settings", "local_skip_dangerous_prompt_declared"),
    ],
)
def test_skip_dangerous_prompt_source_semantics(source_kind: str, diagnostic: str) -> None:
    result = _analyze({"skipDangerousModePermissionPrompt": True}, source_kind=source_kind)

    assert result.outcome is LedgerOutcome.COMPLETED
    assert result.grants == ()
    assert [item.diagnostic_kind for item in result.diagnostics] == [diagnostic]


def test_false_skip_dangerous_prompt_is_a_recognized_noop() -> None:
    result = _analyze({"skipDangerousModePermissionPrompt": False})

    assert result.outcome is LedgerOutcome.COMPLETED
    assert result.diagnostics == ()


def test_local_skip_dangerous_prompt_keeps_bypass_blocking_and_changes_identity() -> None:
    bypass = _analyze({"defaultMode": "bypassPermissions"}, source_kind="project_local_settings")
    prompted = _analyze(
        {"defaultMode": "bypassPermissions", "skipDangerousModePermissionPrompt": True},
        source_kind="project_local_settings",
    )

    assert prompted.grants[0].severity == "CRITICAL"
    assert prompted.grants[0].blocking_critical is True
    assert prompted.aggregate_digest != bypass.aggregate_digest


def test_wrong_type_skip_dangerous_prompt_is_incomplete() -> None:
    result = _analyze({"skipDangerousModePermissionPrompt": "true"})

    assert result.outcome is LedgerOutcome.FAILED
    assert result.reason is LedgerReason.INVALID_CONFIGURATION
    assert [item.diagnostic_kind for item in result.diagnostics] == ["wrong_type"]


@pytest.mark.parametrize("key", ["allow", "ask", "deny", "additionalDirectories"])
def test_nonempty_deferred_rule_grammar_is_incomplete(key: str) -> None:
    result = _analyze({key: ["canary-rule"]})

    assert result.outcome is LedgerOutcome.FAILED
    assert result.reason is LedgerReason.INVALID_CONFIGURATION
    assert [(item.diagnostic_kind, item.affects_completeness) for item in result.diagnostics] == [
        ("unknown_rule", True)
    ]
    assert "canary-rule" not in repr(result)


def test_invalid_digest_is_rejected_without_echoing_it() -> None:
    invalid = "sha256:NOT-A-DIGEST-canary"

    with pytest.raises(ValueError) as exc_info:
        analyze_permission_grants(
            {"permissions": {}},
            source_kind="project_settings",
            content_digest=invalid,
            source_identity_digest="sha256:" + "2" * 64,
            source_lines=PermissionSourceLines(),
        )

    assert invalid not in str(exc_info.value)


def test_structural_item_limit_is_atomic() -> None:
    result = _analyze({f"unknown-{index}": None for index in range(2049)})

    assert result.outcome is LedgerOutcome.FAILED
    assert result.reason is LedgerReason.COMPONENT_LIMIT
    assert result.grants == ()
    assert result.diagnostics == ()
    assert result.aggregate_digest is None
