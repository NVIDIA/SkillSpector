# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sanitized, bounded interpretation of bundled Claude Code permission settings.

This module deliberately accepts already-parsed settings only.  It retains no
permission rule, mode, key, path, or other configuration value outside the
function-local classification boundary.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Final

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.models import Finding

_EVIDENCE_SCHEMA: Final = "skillspector.bundled_permission.v1"
_SEMANTICS_SNAPSHOT: Final = "2.1.241"
MAX_PERMISSION_STRUCTURAL_ITEMS_PER_DOCUMENT: Final = 2048

_SHA256_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SUPPORTED_SOURCE_KINDS: Final = frozenset({"project_settings", "project_local_settings"})
_RECOGNIZED_KEYS: Final = frozenset(
    {
        "allow",
        "ask",
        "deny",
        "additionalDirectories",
        "defaultMode",
        "disableBypassPermissionsMode",
        "disableAutoMode",
        "skipDangerousModePermissionPrompt",
    }
)
_RULE_KEYS: Final = frozenset({"allow", "ask", "deny", "additionalDirectories"})
_SEVERITY_RANK: Final = {"MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


@dataclass(frozen=True, slots=True)
class PermissionGrant:
    """One safe, reportable permission capability classification."""

    grant_kind: str
    severity: str
    activation_requirement: str
    interface_applicability: str
    tracking_status: str
    blocking_critical: bool
    grant_digest: str
    source_line: int


@dataclass(frozen=True, slots=True)
class PermissionDiagnostic:
    """A safe configuration classification which does not retain raw input."""

    diagnostic_kind: str
    affects_completeness: bool
    diagnostic_digest: str
    source_line: int


@dataclass(frozen=True, slots=True)
class PermissionSourceLines:
    """Recovered structural locations, without keys or values."""

    permissions_line: int = 1
    permission_key_lines: tuple[int, ...] = ()
    allow_lines: tuple[int, ...] = ()
    ask_lines: tuple[int, ...] = ()
    deny_lines: tuple[int, ...] = ()
    additional_directory_lines: tuple[int, ...] = ()
    default_mode_line: int | None = None
    disable_bypass_line: int | None = None
    disable_auto_line: int | None = None
    skip_dangerous_prompt_line: int | None = None


@dataclass(frozen=True, slots=True)
class PermissionAnalysis:
    """The immutable result for one physical project settings document."""

    applicable: bool
    outcome: LedgerOutcome | None
    reason: LedgerReason | None
    grants: tuple[PermissionGrant, ...]
    diagnostics: tuple[PermissionDiagnostic, ...]
    aggregate_digest: str | None


def _digest(domain: str, value: bytes) -> str:
    payload = _EVIDENCE_SCHEMA.encode() + b"\0" + domain.encode() + b"\0" + value
    return f"sha256:{sha256(payload).hexdigest()}"


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def _identity_digest(domain: str, value: str) -> str:
    canonical_json_string = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return _digest(f"identity.{domain}", canonical_json_string.encode("ascii"))


def _safe_line(candidate: object, fallback: int) -> int:
    if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
        return candidate
    if isinstance(fallback, int) and not isinstance(fallback, bool) and fallback > 0:
        return fallback
    return 1


def _line_for_key(source_lines: PermissionSourceLines, key: str) -> int:
    candidates = {
        "defaultMode": source_lines.default_mode_line,
        "disableBypassPermissionsMode": source_lines.disable_bypass_line,
        "disableAutoMode": source_lines.disable_auto_line,
        "skipDangerousModePermissionPrompt": source_lines.skip_dangerous_prompt_line,
    }
    return _safe_line(candidates.get(key), _safe_line(source_lines.permissions_line, 1))


def _line_for_rule(source_lines: PermissionSourceLines, key: str, index: int) -> int:
    candidates = {
        "allow": source_lines.allow_lines,
        "ask": source_lines.ask_lines,
        "deny": source_lines.deny_lines,
        "additionalDirectories": source_lines.additional_directory_lines,
    }[key]
    candidate = candidates[index] if index < len(candidates) else None
    return _safe_line(candidate, _safe_line(source_lines.permissions_line, 1))


def _line_for_unknown_key(source_lines: PermissionSourceLines, index: int) -> int:
    candidate = (
        source_lines.permission_key_lines[index]
        if index < len(source_lines.permission_key_lines)
        else None
    )
    return _safe_line(candidate, _safe_line(source_lines.permissions_line, 1))


def _mode_context(source_kind: str) -> tuple[str, str, str]:
    return (
        "interface_and_external_policy",
        "permission_mode_interface_dependent",
        "not_applicable" if source_kind == "project_settings" else "unknown",
    )


def _rule_context(source_kind: str) -> tuple[str, str, str]:
    return (
        "workspace_trust"
        if source_kind == "project_settings"
        else "local_provenance_and_session_policy",
        "claude_code_settings_consumers",
        "not_applicable" if source_kind == "project_settings" else "unknown",
    )


def _diagnostic(
    kind: str,
    affects_completeness: bool,
    source_line: int,
    *,
    identity: str,
) -> PermissionDiagnostic:
    safe = {
        "diagnostic_kind": kind,
        "affects_completeness": affects_completeness,
        "identity_digest": _identity_digest("diagnostic", identity),
    }
    return PermissionDiagnostic(
        diagnostic_kind=kind,
        affects_completeness=affects_completeness,
        diagnostic_digest=_digest("diagnostic.v1", _canonical_bytes(safe)),
        source_line=source_line,
    )


def _grant(
    grant_kind: str,
    severity: str,
    source_kind: str,
    source_line: int,
    *,
    identity: str,
) -> PermissionGrant:
    activation_requirement, interface_applicability, tracking_status = _mode_context(source_kind)
    blocking_critical = severity == "CRITICAL"
    safe = {
        "grant_kind": grant_kind,
        "severity": severity,
        "activation_requirement": activation_requirement,
        "interface_applicability": interface_applicability,
        "tracking_status": tracking_status,
        "blocking_critical": blocking_critical,
        "identity_digest": _identity_digest("grant", identity),
    }
    return PermissionGrant(
        grant_kind=grant_kind,
        severity=severity,
        activation_requirement=activation_requirement,
        interface_applicability=interface_applicability,
        tracking_status=tracking_status,
        blocking_critical=blocking_critical,
        grant_digest=_digest("grant.v1", _canonical_bytes(safe)),
        source_line=source_line,
    )


def _validate_digest(digest: str) -> None:
    if not _SHA256_DIGEST.fullmatch(digest):
        raise ValueError("invalid SHA-256 digest")


def _structural_item_count(permissions: Mapping[object, object]) -> int:
    count = len(permissions)
    for key in _RULE_KEYS:
        value = permissions.get(key)
        if isinstance(value, list):
            count += len(value)
    return count


def _safe_identity(value: object) -> str:
    """Return a local identity only for safe-to-serialize scalar input shapes."""
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return "other"


def _aggregate_digest(
    *,
    source_kind: str,
    content_digest: str,
    source_identity_digest: str,
    grants: tuple[PermissionGrant, ...],
    diagnostics: tuple[PermissionDiagnostic, ...],
) -> str:
    max_severity = max(
        (grant.severity for grant in grants), key=_SEVERITY_RANK.__getitem__, default="LOW"
    )
    safe = {
        "schema": _EVIDENCE_SCHEMA,
        "claude_semantics_snapshot": _SEMANTICS_SNAPSHOT,
        "source_kind": source_kind,
        "source_identity_digest": source_identity_digest,
        "content_digest": content_digest,
        "grant_digests": sorted(grant.grant_digest for grant in grants),
        "diagnostic_digests": sorted(diagnostic.diagnostic_digest for diagnostic in diagnostics),
        "mitigated_allow_count": 0,
        "max_severity": max_severity,
        "blocking_critical": any(grant.blocking_critical for grant in grants),
    }
    return _digest("aggregate.v1", _canonical_bytes(safe))


def analyze_permission_grants(
    raw: Mapping[str, object],
    *,
    source_kind: str,
    content_digest: str,
    source_identity_digest: str,
    source_lines: PermissionSourceLines,
) -> PermissionAnalysis:
    """Classify permission modes without retaining any untrusted setting value."""
    if source_kind not in _SUPPORTED_SOURCE_KINDS:
        raise ValueError("unsupported permission source")
    _validate_digest(content_digest)
    _validate_digest(source_identity_digest)
    if "permissions" not in raw:
        return PermissionAnalysis(False, None, None, (), (), None)

    raw_permissions = raw["permissions"]
    if not isinstance(raw_permissions, dict):
        return PermissionAnalysis(
            True,
            LedgerOutcome.FAILED,
            LedgerReason.INVALID_CONFIGURATION,
            (),
            (),
            None,
        )
    permissions: dict[object, object] = raw_permissions
    if _structural_item_count(permissions) > MAX_PERMISSION_STRUCTURAL_ITEMS_PER_DOCUMENT:
        return PermissionAnalysis(
            True,
            LedgerOutcome.FAILED,
            LedgerReason.COMPONENT_LIMIT,
            (),
            (),
            None,
        )

    grants: list[PermissionGrant] = []
    diagnostics: list[PermissionDiagnostic] = []
    has_valid_content = not permissions
    bypass_declared = False
    bypass_disabled = False
    bypass_line = _line_for_key(source_lines, "defaultMode")

    for key_index, (raw_key, value) in enumerate(permissions.items()):
        key = raw_key if isinstance(raw_key, str) else None
        if key not in _RECOGNIZED_KEYS:
            diagnostics.append(
                _diagnostic(
                    "unknown_permission_key",
                    True,
                    _line_for_unknown_key(source_lines, key_index),
                    identity=key if key is not None else "non_string_key",
                )
            )
            continue

        if key in _RULE_KEYS:
            if not isinstance(value, list):
                diagnostics.append(
                    _diagnostic("wrong_type", True, _line_for_key(source_lines, key), identity=key)
                )
                continue
            if not value:
                has_valid_content = True
                continue
            for item_index, item in enumerate(value):
                diagnostics.append(
                    _diagnostic(
                        "unknown_rule",
                        True,
                        _line_for_rule(source_lines, key, item_index),
                        identity=f"{key}:{_safe_identity(item)}",
                    )
                )
            continue

        line = _line_for_key(source_lines, key)
        if key == "defaultMode":
            if not isinstance(value, str):
                diagnostics.append(_diagnostic("wrong_type", True, line, identity=key))
                continue
            if value == "bypassPermissions":
                has_valid_content = True
                bypass_declared = True
                bypass_line = line
            elif value == "acceptEdits":
                has_valid_content = True
                grants.append(
                    _grant(
                        "permission_mode_accept_edits", "MEDIUM", source_kind, line, identity=key
                    )
                )
            elif value in {"default", "plan", "dontAsk"}:
                has_valid_content = True
            elif value == "manual":
                has_valid_content = True
                diagnostics.append(_diagnostic("legacy_manual", False, line, identity=key))
            elif value == "auto":
                has_valid_content = True
                diagnostics.append(_diagnostic("auto_ignored", False, line, identity=key))
            else:
                diagnostics.append(_diagnostic("unknown_mode", True, line, identity=value))
            continue

        if key in {"disableBypassPermissionsMode", "disableAutoMode"}:
            if value != "disable":
                diagnostics.append(_diagnostic("wrong_type", True, line, identity=key))
            elif key == "disableBypassPermissionsMode":
                has_valid_content = True
                diagnostics.append(_diagnostic("bypass_disabled", False, line, identity=key))
                bypass_disabled = True
            else:
                has_valid_content = True
                diagnostics.append(_diagnostic("auto_disabled", False, line, identity=key))
            continue

        if not isinstance(value, bool):
            diagnostics.append(_diagnostic("wrong_type", True, line, identity=key))
        elif value:
            has_valid_content = True
            diagnostics.append(
                _diagnostic(
                    "skip_dangerous_prompt_ignored"
                    if source_kind == "project_settings"
                    else "local_skip_dangerous_prompt_declared",
                    False,
                    line,
                    identity=key,
                )
            )
        else:
            has_valid_content = True

    if bypass_declared and not bypass_disabled:
        grants.append(
            _grant(
                "permission_mode_bypass",
                "CRITICAL",
                source_kind,
                bypass_line,
                identity="defaultMode",
            )
        )

    unique_grants = {grant.grant_digest: grant for grant in grants}
    unique_diagnostics = {diagnostic.diagnostic_digest: diagnostic for diagnostic in diagnostics}
    sorted_grants = tuple(
        sorted(unique_grants.values(), key=lambda item: (item.grant_digest, item.source_line))
    )
    sorted_diagnostics = tuple(
        sorted(
            unique_diagnostics.values(), key=lambda item: (item.diagnostic_digest, item.source_line)
        )
    )
    incomplete = any(item.affects_completeness for item in sorted_diagnostics)
    outcome = LedgerOutcome.PARTIAL if incomplete and has_valid_content else LedgerOutcome.COMPLETED
    reason = LedgerReason.INVALID_CONFIGURATION if incomplete else None
    if incomplete and not has_valid_content:
        outcome = LedgerOutcome.FAILED
    aggregate_digest = _aggregate_digest(
        source_kind=source_kind,
        content_digest=content_digest,
        source_identity_digest=source_identity_digest,
        grants=sorted_grants,
        diagnostics=sorted_diagnostics,
    )
    return PermissionAnalysis(
        True, outcome, reason, sorted_grants, sorted_diagnostics, aggregate_digest
    )


def build_bh3_finding(analysis: PermissionAnalysis, *, source_path: str) -> Finding | None:
    """Build one structurally safe BH3 finding for retained reportable grants."""
    if not analysis.grants or analysis.aggregate_digest is None:
        return None
    max_severity = max(
        (grant.severity for grant in analysis.grants), key=_SEVERITY_RANK.__getitem__
    )
    evidence: dict[str, object] = {
        "schema": _EVIDENCE_SCHEMA,
        "claude_semantics_snapshot": _SEMANTICS_SNAPSHOT,
        "source_kind": "project_settings"
        if all(grant.tracking_status == "not_applicable" for grant in analysis.grants)
        else "project_local_settings",
        "declaration_status": "declared",
        "artifact_effect_status": "conditional",
        "activation_requirement": ",".join(
            sorted({grant.activation_requirement for grant in analysis.grants})
        ),
        "interface_applicability": ",".join(
            sorted({grant.interface_applicability for grant in analysis.grants})
        ),
        "tracking_status": ",".join(sorted({grant.tracking_status for grant in analysis.grants})),
        "runtime_status": "external_unknown",
        "grant_count": len(analysis.grants),
        "critical_grant_count": sum(grant.severity == "CRITICAL" for grant in analysis.grants),
        "high_grant_count": sum(grant.severity == "HIGH" for grant in analysis.grants),
        "medium_grant_count": sum(grant.severity == "MEDIUM" for grant in analysis.grants),
        "grant_kinds": ",".join(sorted({grant.grant_kind for grant in analysis.grants})),
        "diagnostic_count": len(analysis.diagnostics),
        "diagnostic_kinds": ",".join(
            sorted({diagnostic.diagnostic_kind for diagnostic in analysis.diagnostics})
        ),
        "max_severity": max_severity,
        "blocking_critical": any(grant.blocking_critical for grant in analysis.grants),
        "aggregate_digest": analysis.aggregate_digest,
    }
    return Finding(
        rule_id="BH3",
        message=(
            "Bundled settings declare "
            f"{len(analysis.grants)} permission grant(s) with maximum {max_severity} severity."
        ),
        severity=max_severity,
        confidence=1.0,
        file=source_path,
        start_line=min(grant.source_line for grant in analysis.grants),
        category="Bundled Execution Surface",
        pattern="Bundled Permission Grant",
        explanation="The artifact declares a permission capability subject to external policy.",
        remediation="Review bundled project permission settings before trusting the artifact.",
        tags=["bundled-execution-surface", "structural"],
        matched_text=analysis.aggregate_digest,
        finding=analysis.aggregate_digest,
        evidence=evidence,
    )
