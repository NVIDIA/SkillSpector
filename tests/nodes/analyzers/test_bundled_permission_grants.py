# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for bundled project permission mode analysis."""

from __future__ import annotations

import json
import re
import time
from dataclasses import FrozenInstanceError

import pytest

import skillspector.nodes.analyzers.bundled_permission_grants as permission_grants
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers.bundled_permission_grants import (
    GRANT_KIND_ALLOWLIST,
    PermissionAnalysis,
    PermissionSourceLines,
    _bounded_glob_match,
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


@pytest.mark.parametrize("surrogate_escape", (r"\ud800", r"\udc00"))
@pytest.mark.parametrize(
    ("path", "document_template", "diagnostic_kind"),
    [
        (
            "unknown_key",
            r'{"permissions":{"CANARY-prefix-%s-suffix":null}}',
            "unknown_permission_key",
        ),
        (
            "unknown_mode",
            r'{"permissions":{"defaultMode":"CANARY-prefix-%s-suffix"}}',
            "unknown_mode",
        ),
        (
            "deferred_rule",
            r'{"permissions":{"allow":["CANARY-prefix-%s-suffix"]}}',
            "unknown_rule",
        ),
    ],
)
def test_json_loaded_unpaired_surrogates_fail_closed_without_leaking(
    surrogate_escape: str,
    path: str,
    document_template: str,
    diagnostic_kind: str,
) -> None:
    raw = json.loads(document_template % surrogate_escape)
    permissions = raw["permissions"]
    assert isinstance(permissions, dict)
    if path == "unknown_key":
        canary = next(iter(permissions))
    elif path == "unknown_mode":
        canary = permissions["defaultMode"]
    else:
        canary = permissions["allow"][0]
    assert isinstance(canary, str)

    try:
        result = analyze_permission_grants(
            raw,
            source_kind="project_settings",
            content_digest="sha256:" + "1" * 64,
            source_identity_digest="sha256:" + "2" * 64,
            source_lines=PermissionSourceLines(),
        )
    except Exception as exc:
        assert canary not in str(exc)
        assert canary not in repr(exc)
        pytest.fail("unpaired JSON surrogate must not raise")

    assert result.outcome is LedgerOutcome.FAILED
    assert result.reason is LedgerReason.INVALID_CONFIGURATION
    assert [item.diagnostic_kind for item in result.diagnostics] == [diagnostic_kind]
    assert canary not in repr(result)
    finding = build_bh3_finding(result, source_path=".claude/settings.json")
    assert finding is None


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


def test_unknown_allow_rule_is_incomplete() -> None:
    result = _analyze({"allow": ["canary-rule"]})

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


@pytest.mark.parametrize(
    ("rule", "grant_kind", "severity", "blocking"),
    [
        ("Bash", "tool_wide_execution", "CRITICAL", True),
        ("Bash(*)", "tool_wide_execution", "CRITICAL", True),
        ("PowerShell", "tool_wide_execution", "CRITICAL", True),
        ("PowerShell(*)", "tool_wide_execution", "CRITICAL", True),
        ("Monitor", "tool_wide_execution", "CRITICAL", True),
        ("Monitor(*)", "tool_wide_execution", "CRITICAL", True),
        ("Read", "tool_wide_read", "CRITICAL", True),
        ("Edit", "tool_wide_edit", "CRITICAL", True),
        ("Read(//**)", "root_or_home_wide_read", "CRITICAL", True),
        ("Edit(~/**)", "root_or_home_wide_edit", "CRITICAL", True),
        ("Write", "tool_wide_write", "CRITICAL", True),
        ("Read(~/.ssh/**)", "sensitive_read", "HIGH", False),
        ("NotebookEdit", "broad_notebook_edit", "HIGH", False),
        ("MultiEdit", "broad_multi_edit", "HIGH", False),
        ("WebFetch", "all_domain_fetch", "HIGH", False),
        ("WebFetch(domain:*)", "all_domain_fetch", "HIGH", False),
        ("Edit(../shared/**)", "broad_external_edit", "HIGH", False),
        ("Edit(//tmp/**)", "broad_external_edit", "HIGH", False),
        ("mcp__billing", "mcp_server_wide", "HIGH", False),
        ("mcp__billing__*", "mcp_server_wide", "HIGH", False),
        ("Artifact", "external_content_upload", "HIGH", False),
        ("ShareOnboardingGuide", "external_content_upload", "HIGH", False),
        ("Workflow", "autonomous_workflow", "HIGH", False),
        ("EnterWorktree", "workspace_boundary_change", "HIGH", False),
        ("Bash(npm test:*)", "scoped_execution", "MEDIUM", False),
        ("Monitor(npm test:*)", "scoped_execution", "MEDIUM", False),
        ("Glob", "filesystem_enumeration", "MEDIUM", False),
        ("Grep", "filesystem_search", "MEDIUM", False),
        ("LSP", "code_intelligence", "MEDIUM", False),
        ("WebSearch", "network_search", "MEDIUM", False),
        ("WebFetch(domain:docs.example)", "scoped_domain_fetch", "MEDIUM", False),
        ("WebFetch(domain:*.example.com)", "scoped_domain_fetch", "MEDIUM", False),
        ("WebFetch(domain:example.*)", "scoped_domain_fetch", "MEDIUM", False),
        ("mcp__billing__lookup", "mcp_exact_tool", "MEDIUM", False),
        ("mcp__billing__get_*", "mcp_partial_tool", "MEDIUM", False),
        ("Edit(../shared/config.json)", "scoped_edit", "MEDIUM", False),
        ("Edit(../shared/report-*.md)", "scoped_edit", "MEDIUM", False),
        ("Edit(./generated/**)", "scoped_edit", "MEDIUM", False),
        ("Edit(/tmp/**)", "scoped_edit", "MEDIUM", False),
        ("Read(../shared/report.md)", "external_read", "MEDIUM", False),
        ("Skill", "skill_invocation", "MEDIUM", False),
        ("Skill(commit)", "skill_invocation", "MEDIUM", False),
        ("ExitPlanMode", "approval_gate_transition", "MEDIUM", False),
    ],
)
def test_allow_rule_severity_and_kind(
    rule: str, grant_kind: str, severity: str, blocking: bool
) -> None:
    result = _analyze({"allow": [rule]})

    assert [
        (grant.grant_kind, grant.severity, grant.blocking_critical) for grant in result.grants
    ] == [(grant_kind, severity, blocking)]


@pytest.mark.parametrize("rule", ["Read(src/main.py)"])
def test_allow_rule_silent_controls(rule: str) -> None:
    result = _analyze({"allow": [rule]})

    assert result.outcome is LedgerOutcome.COMPLETED
    assert result.grants == ()
    assert build_bh3_finding(result, source_path=".claude/settings.json") is None


@pytest.mark.parametrize("rule", ["Bash(npx prettier:*)", "Bash(npx prettier *)"])
def test_prettier_execution_is_not_safelisted(rule: str) -> None:
    result = _analyze({"allow": [rule]})

    assert [(grant.grant_kind, grant.severity) for grant in result.grants] == [
        ("scoped_execution", "MEDIUM")
    ]


@pytest.mark.parametrize(
    ("rule", "diagnostic_kind"),
    [
        ("*", "ignored_allow_rule_glob"),
        ("B*", "ignored_allow_rule_glob"),
        ("mcp__*", "ignored_allow_rule_glob"),
        ("WebFetch(*)", "unsupported_allow_specifier"),
    ],
)
def test_deterministically_ignored_allow_forms_are_complete(
    rule: str, diagnostic_kind: str
) -> None:
    result = _analyze({"allow": [rule]})

    assert result.outcome is LedgerOutcome.COMPLETED
    assert result.grants == ()
    assert [(item.diagnostic_kind, item.affects_completeness) for item in result.diagnostics] == [
        (diagnostic_kind, False)
    ]


@pytest.mark.parametrize("tool", ["Grep", "LSP"])
def test_scoped_search_and_intelligence_are_runtime_uncertain(tool: str) -> None:
    result = _analyze({"allow": [f"{tool}(src/**)"]})

    assert result.outcome is LedgerOutcome.FAILED
    assert result.reason is LedgerReason.INVALID_CONFIGURATION
    assert result.grants == ()
    assert [(item.diagnostic_kind, item.affects_completeness) for item in result.diagnostics] == [
        ("runtime_uncertain_rule", True)
    ]


def test_pinned_read_mcp_resource_directory_tool_is_known_non_grant() -> None:
    result = _analyze({"allow": ["ReadMcpResourceDirTool"]})

    assert result.outcome is LedgerOutcome.COMPLETED
    assert result.grants == ()
    assert [item.diagnostic_kind for item in result.diagnostics] == ["known_non_grant_tool"]


@pytest.mark.parametrize(
    ("directory", "grant_kind", "severity", "blocking"),
    [
        (".", None, None, False),
        ("./", None, None, False),
        ("child", None, None, False),
        ("./child", None, None, False),
        ("child/../docs", None, None, False),
        ("../docs", "external_additional_directory", "MEDIUM", False),
        ("child/../../docs", "external_additional_directory", "MEDIUM", False),
        ("/tmp", "external_additional_directory", "MEDIUM", False),
        ("/", "root_or_home_additional_directory", "CRITICAL", True),
        ("//", "root_or_home_additional_directory", "CRITICAL", True),
        ("///", "root_or_home_additional_directory", "CRITICAL", True),
        ("~", "root_or_home_additional_directory", "CRITICAL", True),
        ("~/", "root_or_home_additional_directory", "CRITICAL", True),
        ("~/docs", "external_additional_directory", "MEDIUM", False),
        ("~/.ssh", "sensitive_additional_directory", "HIGH", False),
    ],
)
def test_additional_directory_posix_semantics(
    directory: str, grant_kind: str | None, severity: str | None, blocking: bool
) -> None:
    result = _analyze({"additionalDirectories": [directory]})

    assert result.outcome is LedgerOutcome.COMPLETED
    assert [
        (grant.grant_kind, grant.severity, grant.blocking_critical) for grant in result.grants
    ] == ([] if grant_kind is None else [(grant_kind, severity, blocking)])
    assert [(item.diagnostic_kind, item.affects_completeness) for item in result.diagnostics] == [
        ("directory_existence_static_unknown", False)
    ]


@pytest.mark.parametrize(
    ("directory", "grant_kind", "severity", "blocking"),
    [
        ("C:\\", "root_or_home_additional_directory", "CRITICAL", True),
        ("C:/", "root_or_home_additional_directory", "CRITICAL", True),
        ("\\", "root_or_home_additional_directory", "CRITICAL", True),
        (r"\\", "root_or_home_additional_directory", "CRITICAL", True),
        (r"C:\Users\x\.ssh", "sensitive_additional_directory", "HIGH", False),
        ("C:/Users/x/docs", "external_additional_directory", "MEDIUM", False),
        (r"\\server\share", "external_additional_directory", "MEDIUM", False),
        ("//server/share", "external_additional_directory", "MEDIUM", False),
        (r"\\server", "external_additional_directory", "MEDIUM", False),
        ("//server", "external_additional_directory", "MEDIUM", False),
        (r"\\secret", "sensitive_additional_directory", "HIGH", False),
        (r"\\server\share\.ssh", "sensitive_additional_directory", "HIGH", False),
        (r"\\server\.ssh\..", "sensitive_additional_directory", "HIGH", False),
        ("//server/.ssh/..", "sensitive_additional_directory", "HIGH", False),
        (r"\\server\share\..\..", "external_additional_directory", "MEDIUM", False),
        ("//server/share/../..", "external_additional_directory", "MEDIUM", False),
    ],
)
def test_additional_directory_windows_absolute_is_conditional(
    directory: str, grant_kind: str, severity: str, blocking: bool
) -> None:
    result = _analyze({"additionalDirectories": [directory]})

    assert result.outcome is LedgerOutcome.PARTIAL
    assert result.reason is LedgerReason.INVALID_CONFIGURATION
    assert [
        (grant.grant_kind, grant.severity, grant.blocking_critical) for grant in result.grants
    ] == [(grant_kind, severity, blocking)]
    assert {item.diagnostic_kind for item in result.diagnostics} == {
        "platform_dependent_path",
        "directory_existence_static_unknown",
    }


@pytest.mark.parametrize(
    "directory",
    [
        r"\server",
        r"\\server\\share",
        "//server//share",
        r"\\\server\share",
        "///server/share",
        r"\\server\..\share",
        "//server/../share",
        r"\\.\share",
        "//./share",
    ],
)
def test_additional_directory_malformed_windows_scope_is_not_guessed(directory: str) -> None:
    result = _analyze({"additionalDirectories": [directory]})

    assert result.outcome is LedgerOutcome.FAILED
    assert result.reason is LedgerReason.INVALID_CONFIGURATION
    assert result.grants == ()
    assert [(item.diagnostic_kind, item.affects_completeness) for item in result.diagnostics] == [
        ("platform_dependent_path", True)
    ]


def test_additional_directory_drive_relative_does_not_guess_scope() -> None:
    result = _analyze({"additionalDirectories": ["C:docs"]})

    assert result.outcome is LedgerOutcome.FAILED
    assert result.reason is LedgerReason.INVALID_CONFIGURATION
    assert result.grants == ()
    assert [(item.diagnostic_kind, item.affects_completeness) for item in result.diagnostics] == [
        ("platform_dependent_path", True)
    ]


@pytest.mark.parametrize("directory", ["", "bad\0path", "~someone/docs", "$HOME/docs"])
def test_invalid_additional_directory_fails_closed(directory: str) -> None:
    result = _analyze({"additionalDirectories": [directory]})

    assert result.outcome is LedgerOutcome.FAILED
    assert result.reason is LedgerReason.INVALID_CONFIGURATION
    assert result.grants == ()
    assert [(item.diagnostic_kind, item.affects_completeness) for item in result.diagnostics] == [
        ("invalid_path", True)
    ]


@pytest.mark.parametrize(
    ("key", "rule"),
    [
        ("ask", "Tool(param:value)"),
        ("deny", "UnknownDynamicTool(scope)"),
        ("deny", "Agent(Explore)"),
        ("ask", "ReadMcpResourceDirTool(resource:*)"),
    ],
)
def test_valid_restrictive_rules_are_completed(key: str, rule: str) -> None:
    result = _analyze({key: [rule]})

    assert result.outcome is LedgerOutcome.COMPLETED
    assert result.reason is None
    assert result.grants == ()
    assert [(item.diagnostic_kind, item.affects_completeness) for item in result.diagnostics] == [
        ("restrictive_rule", False)
    ]


@pytest.mark.parametrize(
    ("allow", "restriction_key", "restriction"),
    [
        ("Bash(curl:*)", "ask", "Bash"),
        ("Bash(ls:*)", "deny", "Bash(ls *)"),
        ("Bash(*)", "deny", "Bash"),
        ("PowerShell(Get-ChildItem:*)", "ask", "powershell(get-childitem *)"),
        ("WebFetch(domain:EXAMPLE.com.)", "deny", "WebFetch(domain:example.com)"),
        ("WebFetch(domain:*.EXAMPLE.com.)", "ask", "WebFetch(domain:*.example.com)"),
        ("mcp__billing", "deny", "mcp__billing__*"),
        ("Bash(npm test:*)", "deny", "*"),
        ("Bash(npm test:*)", "ask", "B*"),
        ("mcp__billing__lookup", "deny", "mcp__*"),
        ("Monitor(npm test:*)", "ask", "Bash(npm test *)"),
        ("Monitor(npm test:*)", "deny", "Bash"),
    ],
)
def test_proven_restriction_coverage_mitigates_allow(
    allow: str, restriction_key: str, restriction: str
) -> None:
    result = _analyze({"allow": [allow], restriction_key: [restriction]})

    assert result.outcome is LedgerOutcome.COMPLETED
    assert result.grants == ()
    assert {item.diagnostic_kind for item in result.diagnostics} == {
        "restrictive_rule",
        "mitigated_allow",
    }


@pytest.mark.parametrize(
    ("allow", "restriction_key", "restriction", "grant_kind"),
    [
        ("Read(~/.ssh/**)", "deny", "Read(~/.ssh/id_rsa)", "sensitive_read"),
        ("Read(../shared/**)", "ask", "Read(../shared/*)", "external_read"),
        (
            "WebFetch(domain:*.example.com)",
            "deny",
            "WebFetch(domain:api.example.com)",
            "scoped_domain_fetch",
        ),
        ("Bash(npm test:*)", "deny", "Bash(npm *)", "scoped_execution"),
        ("Monitor", "deny", "Bash", "tool_wide_execution"),
    ],
)
def test_unproven_overlap_does_not_mitigate_allow(
    allow: str, restriction_key: str, restriction: str, grant_kind: str
) -> None:
    result = _analyze({"allow": [allow], restriction_key: [restriction]})

    assert [grant.grant_kind for grant in result.grants] == [grant_kind]
    assert "mitigated_allow" not in {item.diagnostic_kind for item in result.diagnostics}


@pytest.mark.parametrize("restriction_key", ["ask", "deny"])
def test_global_restriction_neutralizes_bypass(restriction_key: str) -> None:
    result = _analyze({"defaultMode": "bypassPermissions", restriction_key: ["*"]})

    assert result.outcome is LedgerOutcome.COMPLETED
    assert result.grants == ()
    assert {item.diagnostic_kind for item in result.diagnostics} == {
        "restrictive_rule",
        "bypass_global_restriction",
    }


@pytest.mark.parametrize(
    ("restriction_key", "restriction"),
    [("ask", "Bash"), ("deny", "Read"), ("ask", "B*"), ("deny", "mcp__*")],
)
def test_narrow_restriction_does_not_neutralize_bypass(
    restriction_key: str, restriction: str
) -> None:
    result = _analyze({"defaultMode": "bypassPermissions", restriction_key: [restriction]})

    assert [
        (grant.grant_kind, grant.severity, grant.blocking_critical) for grant in result.grants
    ] == [("permission_mode_bypass", "CRITICAL", True)]


def test_dont_ask_does_not_remove_preapproved_allow() -> None:
    result = _analyze({"defaultMode": "dontAsk", "allow": ["Bash(npm test:*)"]})

    assert [grant.severity for grant in result.grants] == ["MEDIUM"]


def test_grant_kind_allowlist_is_closed_and_exact() -> None:
    assert GRANT_KIND_ALLOWLIST == {
        "permission_mode_bypass",
        "permission_mode_accept_edits",
        "tool_wide_execution",
        "scoped_execution",
        "tool_wide_read",
        "root_or_home_wide_read",
        "sensitive_read",
        "external_read",
        "tool_wide_edit",
        "root_or_home_wide_edit",
        "sensitive_edit",
        "broad_external_edit",
        "scoped_edit",
        "tool_wide_write",
        "broad_notebook_edit",
        "broad_multi_edit",
        "filesystem_enumeration",
        "filesystem_search",
        "code_intelligence",
        "all_domain_fetch",
        "scoped_domain_fetch",
        "network_search",
        "mcp_server_wide",
        "mcp_exact_tool",
        "mcp_partial_tool",
        "root_or_home_additional_directory",
        "sensitive_additional_directory",
        "external_additional_directory",
        "external_content_upload",
        "skill_invocation",
        "autonomous_workflow",
        "workspace_boundary_change",
        "approval_gate_transition",
    }


def test_webfetch_domain_total_length_boundaries() -> None:
    valid = ".".join(("a" * 63, "b" * 63, "c" * 63, "d" * 61))
    invalid = f"{valid}e"

    accepted = _analyze({"allow": [f"WebFetch(domain:{valid})"]})
    rejected = _analyze({"allow": [f"WebFetch(domain:{invalid})"]})

    assert len(valid) == 253
    assert [grant.grant_kind for grant in accepted.grants] == ["scoped_domain_fetch"]
    assert rejected.outcome is LedgerOutcome.FAILED
    assert [item.diagnostic_kind for item in rejected.diagnostics] == ["unknown_rule"]


def test_webfetch_domain_label_length_boundaries() -> None:
    accepted = _analyze({"allow": [f"WebFetch(domain:{'a' * 63}.example)"]})
    rejected = _analyze({"allow": [f"WebFetch(domain:{'a' * 64}.example)"]})

    assert [grant.grant_kind for grant in accepted.grants] == ["scoped_domain_fetch"]
    assert rejected.outcome is LedgerOutcome.FAILED


def test_webfetch_all_domain_spellings_share_one_identity_and_earliest_line() -> None:
    result = analyze_permission_grants(
        {"permissions": {"allow": ["WebFetch", "WebFetch(domain:*)", "WebFetch(domain:*.)"]}},
        source_kind="project_settings",
        content_digest="sha256:" + "1" * 64,
        source_identity_digest="sha256:" + "2" * 64,
        source_lines=PermissionSourceLines(
            permissions_line=2,
            allow_lines=(12, 3, 8),
        ),
    )

    assert result.outcome is LedgerOutcome.COMPLETED
    assert [(grant.grant_kind, grant.source_line) for grant in result.grants] == [
        ("all_domain_fetch", 3)
    ]


def test_terminal_dot_all_domain_restriction_mitigates_scoped_webfetch() -> None:
    result = _analyze(
        {
            "allow": ["WebFetch(domain:docs.example)"],
            "deny": ["WebFetch(domain:*.)"],
        }
    )

    assert result.grants == ()
    assert {item.diagnostic_kind for item in result.diagnostics} == {
        "restrictive_rule",
        "mitigated_allow",
    }


@pytest.mark.parametrize(
    "domain",
    ["xn--bcher-kva.example", "*.example.com", "example.*", "a*b.example", "EXAMPLE.com."],
)
def test_webfetch_valid_ascii_domain_patterns(domain: str) -> None:
    result = _analyze({"allow": [f"WebFetch(domain:{domain})"]})

    assert result.outcome is LedgerOutcome.COMPLETED
    assert [grant.grant_kind for grant in result.grants] == ["scoped_domain_fetch"]


@pytest.mark.parametrize(
    "domain",
    [
        "https://example.com",
        "user@example.com",
        "example.com:443",
        "example.com/path",
        "example .com",
        "example..com",
        "example.com?x=1",
        "-example.com",
        "example-.com",
        "example.com..",
        "bücher.example",
    ],
)
def test_webfetch_invalid_domain_patterns_fail_closed(domain: str) -> None:
    result = _analyze({"allow": [f"WebFetch(domain:{domain})"]})

    assert result.outcome is LedgerOutcome.FAILED
    assert result.reason is LedgerReason.INVALID_CONFIGURATION
    assert result.grants == ()
    assert [item.diagnostic_kind for item in result.diagnostics] == ["unknown_rule"]


KNOWN_NON_GRANT_TOOLS = {
    "Agent",
    "AskUserQuestion",
    "Cd",
    "CronCreate",
    "CronDelete",
    "CronList",
    "EndConversation",
    "EnterPlanMode",
    "ExitWorktree",
    "ListAgents",
    "ListMcpResourcesTool",
    "PushNotification",
    "ReadMcpResourceDirTool",
    "ReadMcpResourceTool",
    "RemoteTrigger",
    "ReportFindings",
    "ScheduleWakeup",
    "SendMessage",
    "SendUserFile",
    "SendUserMessage",
    "Task",
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskOutput",
    "TaskStop",
    "TaskUpdate",
    "TodoWrite",
    "ToolSearch",
    "WaitForMcpServers",
}


def test_canonical_exact_names_belong_to_exactly_one_route() -> None:
    routes = (
        {"Bash", "PowerShell", "Monitor"},
        {"Read", "Edit", "Write", "NotebookEdit", "MultiEdit", "Glob", "Grep", "LSP"},
        {"WebFetch", "WebSearch"},
        {"Artifact", "ShareOnboardingGuide"},
        {"Skill"},
        {"Workflow", "EnterWorktree", "ExitPlanMode"},
        KNOWN_NON_GRANT_TOOLS,
    )

    for tool in set().union(*routes):
        assert sum(tool in route for route in routes) == 1


@pytest.mark.parametrize("tool", sorted(KNOWN_NON_GRANT_TOOLS))
def test_every_known_non_grant_has_exact_bare_and_scoped_routes(tool: str) -> None:
    bare = _analyze({"allow": [tool]})
    scoped = _analyze({"allow": [f"{tool}(scope)"]})

    assert bare.outcome is LedgerOutcome.COMPLETED
    assert bare.grants == ()
    assert [item.diagnostic_kind for item in bare.diagnostics] == ["known_non_grant_tool"]
    assert scoped.outcome is LedgerOutcome.COMPLETED
    assert scoped.grants == ()
    assert [item.diagnostic_kind for item in scoped.diagnostics] == ["unsupported_allow_specifier"]


@pytest.mark.parametrize(
    "rule",
    [
        "Bash(",
        "Bash)",
        "Bash()",
        "Bash((pwd))",
        "Bash(pwd)tail",
        "Bash(pwd)(whoami)",
        "Bad Tool",
        "UnknownTool(*)",
        "mcp____lookup",
        "mcp__server__tool__extra",
        "mcp__ser*ver__tool",
    ],
)
def test_malformed_or_unknown_allow_rules_fail_closed(rule: str) -> None:
    result = _analyze({"allow": [rule]})

    assert result.outcome is LedgerOutcome.FAILED
    assert result.reason is LedgerReason.INVALID_CONFIGURATION
    assert result.grants == ()
    assert [item.diagnostic_kind for item in result.diagnostics] == ["unknown_rule"]


@pytest.mark.parametrize(
    "rule",
    [
        "Read(../shared/../secret)",
        "Read(C:/secret)",
        r"Read(\\server\share)",
        "Read(~someone/.ssh)",
        "Read(///tmp)",
        "Edit(bad\0path)",
    ],
)
def test_invalid_read_edit_paths_fail_closed(rule: str) -> None:
    result = _analyze({"allow": [rule]})

    assert result.outcome is LedgerOutcome.FAILED
    assert result.grants == ()
    assert [item.diagnostic_kind for item in result.diagnostics] == ["unknown_rule"]


@pytest.mark.parametrize(
    "rule",
    [
        "Read(~/.claude/settings.json)",
        "Read(~/.zsh_history)",
        "Read(~/.ssh/id_rsa)",
        "Read(~/.aws/credentials)",
        "Read(~/.config/gcloud/application_default_credentials.json)",
        "Read(~/.kube/config)",
        "Read(~/.docker/config.json)",
        "Read(~/.npmrc)",
        "Read(~/.cargo/credentials)",
        "Read(~/.git-credentials)",
        "Read(.env.production)",
        "Read(config/secrets/token.json)",
    ],
)
def test_sensitive_read_categories_are_high(rule: str) -> None:
    result = _analyze({"allow": [rule]})

    assert [(grant.grant_kind, grant.severity) for grant in result.grants] == [
        ("sensitive_read", "HIGH")
    ]


@pytest.mark.parametrize(
    "rule",
    [
        "Read(src/tokenizer.py)",
        "Read(docs/tokenization.md)",
        "Read(docs/secretariat.md)",
        "Read(docs/ſecret.md)",
    ],
)
def test_sensitive_markers_do_not_match_ascii_alphanumeric_continuations(rule: str) -> None:
    result = _analyze({"allow": [rule]})

    assert result.outcome is LedgerOutcome.COMPLETED
    assert result.grants == ()


@pytest.mark.parametrize(
    "rule",
    [
        "Read(config/token)",
        "Read(config/API-TOKEN.txt)",
        "Read(config/my_secret_backup.json)",
        "Read(~/.SSH/ID_RSA)",
    ],
)
def test_sensitive_markers_match_complete_or_punctuation_delimited_tokens(rule: str) -> None:
    result = _analyze({"allow": [rule]})

    assert [(grant.grant_kind, grant.severity) for grant in result.grants] == [
        ("sensitive_read", "HIGH")
    ]


def test_pinned_2_1_241_multiedit_fixture_remains_broad_edit() -> None:
    pinned_canonical_edit_tools = "Edit MultiEdit NotebookEdit Write"

    assert "MultiEdit" in pinned_canonical_edit_tools.split()
    result = _analyze({"allow": ["MultiEdit"]})
    assert [(grant.grant_kind, grant.severity) for grant in result.grants] == [
        ("broad_multi_edit", "HIGH")
    ]


def test_every_allowlisted_grant_kind_is_reachable() -> None:
    rules = [
        "Bash",
        "Bash(npm test:*)",
        "Read",
        "Read(//)",
        "Read(~/.ssh/config)",
        "Read(../shared/report.md)",
        "Edit",
        "Edit(~)",
        "Edit(~/.ssh/config)",
        "Edit(../shared/**)",
        "Edit(src/generated/**)",
        "Write",
        "NotebookEdit",
        "MultiEdit",
        "Glob",
        "Grep",
        "LSP",
        "WebFetch",
        "WebFetch(domain:docs.example)",
        "WebSearch",
        "mcp__files",
        "mcp__files__read",
        "mcp__files__get_*",
        "Artifact",
        "Skill(commit)",
        "Workflow",
        "EnterWorktree",
        "ExitPlanMode",
    ]
    rule_result = _analyze({"allow": rules, "additionalDirectories": ["/", "~/.ssh", "/tmp"]})
    bypass = _analyze({"defaultMode": "bypassPermissions"})
    accept_edits = _analyze({"defaultMode": "acceptEdits"})
    reached = {
        grant.grant_kind
        for result in (rule_result, bypass, accept_edits)
        for grant in result.grants
    }

    assert reached == GRANT_KIND_ALLOWLIST


@pytest.mark.parametrize(
    ("source_kind", "activation", "tracking"),
    [
        ("project_settings", "workspace_trust", "not_applicable"),
        (
            "project_local_settings",
            "local_provenance_and_session_policy",
            "unknown",
        ),
    ],
)
@pytest.mark.parametrize(
    "permissions",
    [{"allow": ["Bash"]}, {"additionalDirectories": ["/tmp"]}],
)
def test_rule_and_directory_grants_have_source_context(
    source_kind: str, activation: str, tracking: str, permissions: dict[str, list[str]]
) -> None:
    result = _analyze(permissions, source_kind=source_kind)

    assert len(result.grants) == 1
    grant = result.grants[0]
    assert grant.activation_requirement == activation
    assert grant.interface_applicability == "claude_code_settings_consumers"
    assert grant.tracking_status == tracking


def test_semantic_duplicates_and_permutations_have_stable_projections() -> None:
    first = _analyze(
        {
            "allow": [
                "Bash(ls:*)",
                "mcp__files",
                "WebFetch(domain:EXAMPLE.com.)",
            ],
            "additionalDirectories": ["child", "../docs"],
        }
    )
    repeated = _analyze(
        {
            "additionalDirectories": ["../docs", "./child", "../docs"],
            "allow": [
                "WebFetch(domain:example.com)",
                "mcp__files__*",
                "Bash(ls *)",
                "Bash(ls:*)",
            ],
        }
    )

    assert repeated.grants == first.grants
    assert repeated.diagnostics == first.diagnostics
    assert repeated.aggregate_digest == first.aggregate_digest


def test_mixed_valid_and_non_string_rule_retains_grant_as_partial() -> None:
    result = _analyze({"allow": ["Bash", 7]})

    assert result.outcome is LedgerOutcome.PARTIAL
    assert result.reason is LedgerReason.INVALID_CONFIGURATION
    assert [grant.grant_kind for grant in result.grants] == ["tool_wide_execution"]
    assert [(item.diagnostic_kind, item.affects_completeness) for item in result.diagnostics] == [
        ("wrong_type", True)
    ]


def test_untrusted_rule_canaries_never_leave_safe_records() -> None:
    canary = "CANARY-prefix-\ud800-\x01-suffix"
    result = _analyze(
        {
            "allow": ["Bash", f"UnknownTool({canary})"],
            "ask": [f"DynamicTool({canary})"],
            "deny": [f"OtherTool({canary})"],
            "additionalDirectories": [f"~{canary}"],
        }
    )

    assert result.outcome is LedgerOutcome.PARTIAL
    assert [grant.grant_kind for grant in result.grants] == ["tool_wide_execution"]
    assert canary not in repr(result)
    finding = build_bh3_finding(result, source_path=".claude/settings.json")
    assert finding is not None
    assert canary not in repr(finding)


def test_rule_source_lines_use_safe_positive_fallback() -> None:
    result = analyze_permission_grants(
        {"permissions": {"allow": ["Bash"], "additionalDirectories": ["/tmp"]}},
        source_kind="project_settings",
        content_digest="sha256:" + "1" * 64,
        source_identity_digest="sha256:" + "2" * 64,
        source_lines=PermissionSourceLines(
            permissions_line=7,
            allow_lines=(0,),
            additional_directory_lines=(-3,),
        ),
    )

    assert {grant.source_line for grant in result.grants} == {7}
    assert {diagnostic.source_line for diagnostic in result.diagnostics} == {7}


def test_external_edit_with_final_all_entry_wildcard_is_broad() -> None:
    result = _analyze({"allow": ["Edit(../shared/*)"]})

    assert [(grant.grant_kind, grant.severity) for grant in result.grants] == [
        ("broad_external_edit", "HIGH")
    ]


@pytest.mark.parametrize("rule", ["Read(secret.json)", "Read(config/api-token.txt)"])
def test_secret_and_token_material_are_sensitive(rule: str) -> None:
    result = _analyze({"allow": [rule]})

    assert [(grant.grant_kind, grant.severity) for grant in result.grants] == [
        ("sensitive_read", "HIGH")
    ]


def test_distinct_windows_drive_roots_remain_distinct_grants() -> None:
    result = _analyze({"additionalDirectories": ["C:/", "D:/"]})

    assert len(result.grants) == 2
    assert all(grant.grant_kind == "root_or_home_additional_directory" for grant in result.grants)


def test_semantic_duplicate_uses_earliest_positive_source_line() -> None:
    result = analyze_permission_grants(
        {"permissions": {"allow": ["Bash", "Bash(*)"]}},
        source_kind="project_settings",
        content_digest="sha256:" + "1" * 64,
        source_identity_digest="sha256:" + "2" * 64,
        source_lines=PermissionSourceLines(permissions_line=7, allow_lines=(10, 20)),
    )

    assert len(result.grants) == 1
    assert result.grants[0].source_line == 10


@pytest.mark.parametrize("tool", ["Write", "NotebookEdit", "MultiEdit", "Glob"])
def test_known_ignored_path_qualifiers_are_complete(tool: str) -> None:
    result = _analyze({"allow": [f"{tool}(../outside/**)"]})

    assert result.outcome is LedgerOutcome.COMPLETED
    assert result.grants == ()
    assert [(item.diagnostic_kind, item.affects_completeness) for item in result.diagnostics] == [
        ("ignored_path_qualifier", False)
    ]


@pytest.mark.parametrize(
    "tool",
    ["Artifact", "ShareOnboardingGuide", "Workflow", "EnterWorktree", "ExitPlanMode", "WebSearch"],
)
def test_known_bare_only_grant_routes_reject_scoped_forms_neutrally(tool: str) -> None:
    result = _analyze({"allow": [f"{tool}(scope)"]})

    assert result.outcome is LedgerOutcome.COMPLETED
    assert result.grants == ()
    assert [item.diagnostic_kind for item in result.diagnostics] == ["unsupported_allow_specifier"]


def test_exact_path_spelling_is_mitigated() -> None:
    result = _analyze(
        {"allow": ["Edit(./generated/report.md)"], "deny": ["Edit(./generated/report.md)"]}
    )

    assert result.grants == ()
    assert "mitigated_allow" in {item.diagnostic_kind for item in result.diagnostics}


def test_distinct_path_spelling_is_not_normalized_for_mitigation() -> None:
    result = _analyze(
        {"allow": ["Edit(./generated/report.md)"], "deny": ["Edit(generated/report.md)"]}
    )

    assert [grant.grant_kind for grant in result.grants] == ["scoped_edit"]
    assert "mitigated_allow" not in {item.diagnostic_kind for item in result.diagnostics}


@pytest.mark.parametrize("rule", ["Read(~//docs)", "Read(.//docs)", "Edit(..//shared)"])
def test_ambiguous_permission_path_separators_fail_closed(rule: str) -> None:
    result = _analyze({"allow": [rule]})

    assert result.outcome is LedgerOutcome.FAILED
    assert result.grants == ()
    assert [item.diagnostic_kind for item in result.diagnostics] == ["unknown_rule"]


def test_hostile_tool_glob_near_miss_scales_linearly() -> None:
    def duration(size: int) -> float:
        started = time.perf_counter()
        assert _bounded_glob_match(f"*{'a' * size}b", "a" * (size * 2)) is False
        return time.perf_counter() - started

    small = duration(3_000)
    large = duration(12_000)

    assert large < small * 8 + 0.02


def test_duplicate_restrictions_are_deduplicated_before_large_candidate_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert hasattr(permission_grants, "_compile_tool_glob")
    assert hasattr(permission_grants, "_match_compiled_tool_glob")
    duplicate_count = 1_000
    server = "a" * 500_000
    deny_rule = "*Z*"
    compile_calls = 0
    match_calls = 0
    original_compile = permission_grants._compile_tool_glob
    original_match = permission_grants._match_compiled_tool_glob

    def counting_compile(pattern: str) -> permission_grants._CompiledToolGlob:
        nonlocal compile_calls
        compile_calls += 1
        return original_compile(pattern)

    def counting_match(compiled: permission_grants._CompiledToolGlob, value: str) -> bool:
        nonlocal match_calls
        match_calls += 1
        return original_match(compiled, value)

    monkeypatch.setattr(permission_grants, "_compile_tool_glob", counting_compile)
    monkeypatch.setattr(permission_grants, "_match_compiled_tool_glob", counting_match)
    result = analyze_permission_grants(
        {
            "permissions": {
                "allow": [f"mcp__{server}__read"],
                "deny": [deny_rule] * duplicate_count,
            }
        },
        source_kind="project_settings",
        content_digest="sha256:" + "1" * 64,
        source_identity_digest="sha256:" + "2" * 64,
        source_lines=PermissionSourceLines(
            permissions_line=2,
            allow_lines=(4,),
            deny_lines=tuple(range(duplicate_count + 2, 2, -1)),
        ),
    )

    assert compile_calls == 1
    assert match_calls == 1
    assert [grant.grant_kind for grant in result.grants] == ["mcp_exact_tool"]
    assert [diagnostic.source_line for diagnostic in result.diagnostics] == [3]


def test_duplicate_allow_candidates_are_deduplicated_before_coverage_with_earliest_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert hasattr(permission_grants, "_match_compiled_tool_glob")
    match_calls = 0
    original_match = permission_grants._match_compiled_tool_glob

    def counting_match(compiled: permission_grants._CompiledToolGlob, value: str) -> bool:
        nonlocal match_calls
        match_calls += 1
        return original_match(compiled, value)

    monkeypatch.setattr(permission_grants, "_match_compiled_tool_glob", counting_match)
    result = analyze_permission_grants(
        {
            "permissions": {
                "allow": ["Bash(ls:*)", "Bash(ls *)", "Bash(ls:*)"],
                "deny": ["Z*"],
            }
        },
        source_kind="project_settings",
        content_digest="sha256:" + "1" * 64,
        source_identity_digest="sha256:" + "2" * 64,
        source_lines=PermissionSourceLines(
            permissions_line=2,
            allow_lines=(12, 3, 8),
            deny_lines=(5,),
        ),
    )

    assert match_calls == 1
    assert [(grant.grant_kind, grant.source_line) for grant in result.grants] == [
        ("scoped_execution", 3)
    ]


def test_indexed_restrictions_avoid_glob_matching_for_proven_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert hasattr(permission_grants, "_match_compiled_tool_glob")
    match_calls = 0
    original_match = permission_grants._match_compiled_tool_glob

    def counting_match(compiled: permission_grants._CompiledToolGlob, value: str) -> bool:
        nonlocal match_calls
        match_calls += 1
        return original_match(compiled, value)

    monkeypatch.setattr(permission_grants, "_match_compiled_tool_glob", counting_match)
    result = _analyze(
        {
            "allow": [
                "Bash(echo hi)",
                "WebFetch(domain:docs.example)",
                "mcp__files__read",
            ],
            "deny": ["Z*", "Bash(echo hi)", "WebFetch", "mcp__files"],
        }
    )

    assert result.grants == ()
    assert match_calls == 0


def test_glob_match_budget_accepts_exact_limit_and_rejects_next_character(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert permission_grants.MAX_PERMISSION_GLOB_MATCH_CHARS_PER_DOCUMENT == 8_388_608
    candidate = "mcp__s__t"
    glob = "Z*"
    exact_charge = len(candidate) + len(glob)

    monkeypatch.setattr(
        permission_grants,
        "MAX_PERMISSION_GLOB_MATCH_CHARS_PER_DOCUMENT",
        exact_charge,
    )
    accepted = _analyze({"allow": [candidate], "deny": [glob]})

    monkeypatch.setattr(
        permission_grants,
        "MAX_PERMISSION_GLOB_MATCH_CHARS_PER_DOCUMENT",
        exact_charge - 1,
    )
    rejected = _analyze({"allow": [candidate], "deny": [glob]})

    assert [grant.grant_kind for grant in accepted.grants] == ["mcp_exact_tool"]
    assert rejected.outcome is LedgerOutcome.FAILED
    assert rejected.reason is LedgerReason.COMPONENT_LIMIT
    assert rejected.grants == ()
    assert rejected.diagnostics == ()
    assert rejected.aggregate_digest is None


def test_near_megabyte_unique_glob_cross_product_hits_atomic_deterministic_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert hasattr(permission_grants, "_compile_tool_glob")
    assert hasattr(permission_grants, "_match_compiled_tool_glob")
    rule_count = 200
    rule_width = 2_400

    def padded_rule(prefix: str, suffix: str) -> str:
        return f"{prefix}{'a' * (rule_width - len(prefix) - len(suffix))}{suffix}"

    allow = [padded_rule(f"mcp__s{index:03d}", "__tool") for index in range(rule_count)]
    deny = [padded_rule(f"Z{index:03d}", "*X") for index in range(rule_count)]
    permissions = {"allow": allow, "deny": deny}
    encoded_size = len(json.dumps({"permissions": permissions}, separators=(",", ":")))
    charge = rule_width * 2
    expected_matches = permission_grants.MAX_PERMISSION_GLOB_MATCH_CHARS_PER_DOCUMENT // charge
    compile_calls = 0
    match_calls = 0
    original_compile = permission_grants._compile_tool_glob
    original_match = permission_grants._match_compiled_tool_glob

    def counting_compile(pattern: str) -> permission_grants._CompiledToolGlob:
        nonlocal compile_calls
        compile_calls += 1
        return original_compile(pattern)

    def counting_match(compiled: permission_grants._CompiledToolGlob, value: str) -> bool:
        nonlocal match_calls
        match_calls += 1
        return original_match(compiled, value)

    monkeypatch.setattr(permission_grants, "_compile_tool_glob", counting_compile)
    monkeypatch.setattr(permission_grants, "_match_compiled_tool_glob", counting_match)

    first = _analyze(permissions)
    first_counts = (compile_calls, match_calls)
    compile_calls = match_calls = 0
    second = _analyze({"allow": list(reversed(allow)), "deny": list(reversed(deny))})
    second_counts = (compile_calls, match_calls)

    assert 900_000 <= encoded_size < 1_000_000
    assert first == second
    assert first.outcome is LedgerOutcome.FAILED
    assert first.reason is LedgerReason.COMPONENT_LIMIT
    assert first.grants == ()
    assert first.diagnostics == ()
    assert first.aggregate_digest is None
    assert first_counts == second_counts == (rule_count, expected_matches)


def test_equivalent_mcp_server_candidates_have_order_stable_normalized_glob_coverage() -> None:
    bare_first = _analyze(
        {
            "allow": ["mcp__files", "mcp__files__*"],
            "deny": ["mcp__files_*"],
        }
    )
    wildcard_first = _analyze(
        {
            "allow": ["mcp__files__*", "mcp__files"],
            "deny": ["mcp__files_*"],
        }
    )

    assert bare_first.grants == wildcard_first.grants == ()
    assert bare_first.diagnostics == wildcard_first.diagnostics
    assert bare_first.aggregate_digest == wildcard_first.aggregate_digest


@pytest.mark.parametrize(
    ("allow", "deny"),
    [
        ("Bash(echo hi)", "Bash(*)"),
        ("WebFetch(domain:docs.example)", "WebFetch(domain:*)"),
        ("mcp__billing__lookup", "mcp__billing"),
        ("Monitor(echo hi)", "Monitor"),
    ],
)
def test_semantically_tool_wide_restriction_covers_scoped_allow(allow: str, deny: str) -> None:
    result = _analyze({"allow": [allow], "deny": [deny]})

    assert result.grants == ()
    assert "mitigated_allow" in {item.diagnostic_kind for item in result.diagnostics}


def test_windows_environment_variable_directory_is_invalid() -> None:
    result = _analyze({"additionalDirectories": ["%USERPROFILE%/docs"]})

    assert result.outcome is LedgerOutcome.FAILED
    assert result.grants == ()
    assert [(item.diagnostic_kind, item.affects_completeness) for item in result.diagnostics] == [
        ("invalid_path", True)
    ]


def test_unpaired_percent_in_directory_is_literal() -> None:
    result = _analyze({"additionalDirectories": ["reports/100%"]})

    assert result.outcome is LedgerOutcome.COMPLETED
    assert result.grants == ()
    assert [item.diagnostic_kind for item in result.diagnostics] == [
        "directory_existence_static_unknown"
    ]


@pytest.mark.parametrize("directory", ["/tmp/..", "~/docs/.."])
def test_lexically_normalized_root_or_home_directory_is_critical(directory: str) -> None:
    result = _analyze({"additionalDirectories": [directory]})

    assert [
        (grant.grant_kind, grant.severity, grant.blocking_critical) for grant in result.grants
    ] == [("root_or_home_additional_directory", "CRITICAL", True)]


@pytest.mark.parametrize("directory", ["/..", "/tmp/../.."])
def test_absolute_parent_traversal_clamps_at_filesystem_root(directory: str) -> None:
    result = _analyze({"additionalDirectories": [directory]})

    assert result.outcome is LedgerOutcome.COMPLETED
    assert [
        (grant.grant_kind, grant.severity, grant.blocking_critical) for grant in result.grants
    ] == [("root_or_home_additional_directory", "CRITICAL", True)]


@pytest.mark.parametrize("directory", ["C:/..", "C:/Users/../.."])
def test_drive_parent_traversal_clamps_at_drive_root(directory: str) -> None:
    result = _analyze({"additionalDirectories": [directory]})

    assert result.outcome is LedgerOutcome.PARTIAL
    assert [
        (grant.grant_kind, grant.severity, grant.blocking_critical) for grant in result.grants
    ] == [("root_or_home_additional_directory", "CRITICAL", True)]


def test_powershell_tool_glob_remains_case_sensitive_after_identifier_normalization() -> None:
    matched = _analyze({"allow": ["PowerShell(Get-ChildItem:*)"], "deny": ["power*"]})
    unmatched = _analyze({"allow": ["PowerShell(Get-ChildItem:*)"], "deny": ["POWER*"]})

    assert matched.grants == ()
    assert [grant.grant_kind for grant in unmatched.grants] == ["scoped_execution"]


def test_powershell_command_normalization_folds_ascii_only() -> None:
    ascii_equivalent = _analyze(
        {
            "allow": ["PowerShell(WRITE-ß)"],
            "deny": ["powershell(write-ß)"],
        }
    )
    unicode_distinct = _analyze(
        {
            "allow": ["PowerShell(WRITE-ß)"],
            "deny": ["powershell(write-ss)"],
        }
    )

    assert ascii_equivalent.grants == ()
    assert [grant.grant_kind for grant in unicode_distinct.grants] == ["scoped_execution"]


@pytest.mark.parametrize("directory", ["~/.config/gcloud", "~/.config/gh", "~/.config/glab"])
def test_exact_cloud_credential_store_directory_is_sensitive(directory: str) -> None:
    result = _analyze({"additionalDirectories": [directory]})

    assert [(grant.grant_kind, grant.severity) for grant in result.grants] == [
        ("sensitive_additional_directory", "HIGH")
    ]
