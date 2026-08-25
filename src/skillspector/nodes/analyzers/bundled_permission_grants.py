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
MAX_PERMISSION_GLOB_MATCH_CHARS_PER_DOCUMENT: Final = 8_388_608

_SHA256_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SUPPORTED_SOURCE_KINDS: Final = frozenset({"project_settings", "project_local_settings"})
_AGGREGATE_PREFIX: Final = b"skillspector.bundled_permission.aggregate.v1\0"
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
GRANT_KIND_ALLOWLIST: Final = frozenset(
    {
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
)
_GRANT_SEVERITY_BY_KIND: Final = {
    "permission_mode_bypass": "CRITICAL",
    "permission_mode_accept_edits": "MEDIUM",
    "tool_wide_execution": "CRITICAL",
    "scoped_execution": "MEDIUM",
    "tool_wide_read": "CRITICAL",
    "root_or_home_wide_read": "CRITICAL",
    "sensitive_read": "HIGH",
    "external_read": "MEDIUM",
    "tool_wide_edit": "CRITICAL",
    "root_or_home_wide_edit": "CRITICAL",
    "sensitive_edit": "HIGH",
    "broad_external_edit": "HIGH",
    "scoped_edit": "MEDIUM",
    "tool_wide_write": "CRITICAL",
    "broad_notebook_edit": "HIGH",
    "broad_multi_edit": "HIGH",
    "filesystem_enumeration": "MEDIUM",
    "filesystem_search": "MEDIUM",
    "code_intelligence": "MEDIUM",
    "all_domain_fetch": "HIGH",
    "scoped_domain_fetch": "MEDIUM",
    "network_search": "MEDIUM",
    "mcp_server_wide": "HIGH",
    "mcp_exact_tool": "MEDIUM",
    "mcp_partial_tool": "MEDIUM",
    "root_or_home_additional_directory": "CRITICAL",
    "sensitive_additional_directory": "HIGH",
    "external_additional_directory": "MEDIUM",
    "external_content_upload": "HIGH",
    "skill_invocation": "MEDIUM",
    "autonomous_workflow": "HIGH",
    "workspace_boundary_change": "HIGH",
    "approval_gate_transition": "MEDIUM",
}
_MODE_GRANT_KINDS: Final = frozenset({"permission_mode_bypass", "permission_mode_accept_edits"})
_DIAGNOSTIC_COMPLETENESS: Final = {
    "auto_ignored": False,
    "legacy_manual": False,
    "bypass_disabled": False,
    "bypass_global_restriction": False,
    "auto_disabled": False,
    "skip_dangerous_prompt_ignored": False,
    "local_skip_dangerous_prompt_declared": False,
    "ignored_allow_rule_glob": False,
    "ignored_path_qualifier": False,
    "runtime_uncertain_rule": True,
    "unsupported_allow_specifier": False,
    "known_non_grant_tool": False,
    "restrictive_rule": False,
    "mitigated_allow": False,
    "platform_dependent_path": True,
    "directory_existence_static_unknown": False,
    "unknown_permission_key": True,
    "unknown_mode": True,
    "unknown_rule": True,
    "wrong_type": True,
    "invalid_path": True,
}
_GRANT_ACTIVATION_REQUIREMENTS: Final = frozenset(
    {
        "workspace_trust",
        "local_provenance_and_session_policy",
        "interface_and_external_policy",
    }
)
_GRANT_INTERFACE_APPLICABILITY: Final = frozenset(
    {"claude_code_settings_consumers", "permission_mode_interface_dependent"}
)
_GRANT_TRACKING_STATUS: Final = frozenset({"not_applicable", "unknown"})

_SHELL_TOOLS: Final = frozenset({"Bash", "PowerShell", "Monitor"})
_FILESYSTEM_TOOLS: Final = frozenset(
    {"Read", "Edit", "Write", "NotebookEdit", "MultiEdit", "Glob", "Grep", "LSP"}
)
_BARE_EXTERNAL_UPLOAD_TOOLS: Final = frozenset({"Artifact", "ShareOnboardingGuide"})
_KNOWN_NON_GRANT_TOOLS: Final = frozenset(
    {
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
)
_BARE_ROUTE_GRANTS: Final = {
    "Workflow": ("autonomous_workflow", "HIGH"),
    "EnterWorktree": ("workspace_boundary_change", "HIGH"),
    "ExitPlanMode": ("approval_gate_transition", "MEDIUM"),
}
_WHOLE_PERMISSION_PATHS: Final = frozenset({"//", "//**", "//**/*", "~", "~/", "~/**", "~/**/*"})
_ECMASCRIPT_TRIM_CHARACTERS: Final = frozenset(
    chr(code_point)
    for code_point in (
        *range(0x0009, 0x000E),
        0x0020,
        0x00A0,
        0x1680,
        *range(0x2000, 0x200B),
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
        0xFEFF,
    )
)


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


@dataclass(frozen=True, slots=True)
class _ParsedRule:
    tool: str
    specifier: str | None


@dataclass(frozen=True, slots=True)
class _PathClassification:
    scope: str
    normalized: str
    broad: bool


@dataclass(frozen=True, slots=True)
class _AllowCandidate:
    grant: PermissionGrant
    tool_identifier: str
    original_tool_identifier: str
    normalized_identity: str
    mcp_server: str | None


@dataclass(frozen=True, slots=True)
class _Restriction:
    tool_identifier: str
    original_tool_identifier: str
    normalized_identity: str
    tool_glob: bool
    tool_wide: bool
    mcp_server: str | None
    source_line: int


@dataclass(frozen=True, slots=True)
class _CompiledLiteral:
    value: str
    prefix: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _CompiledToolGlob:
    character_count: int
    exact_literal: str | None
    leading_wildcard: bool
    trailing_wildcard: bool
    segments: tuple[_CompiledLiteral, ...]


@dataclass(frozen=True, slots=True)
class _RestrictionIndex:
    exact_identities: frozenset[str]
    tool_wide_identifiers: frozenset[str]
    original_tool_wide_identifiers: frozenset[str]
    mcp_servers: frozenset[str]
    tool_globs: tuple[_CompiledToolGlob, ...]


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


def _valid_tool_identifier(value: str, *, allow_glob: bool) -> bool:
    if not value or not value.isascii():
        return False
    for character in value:
        if character.isalnum() or character in {"_", "-"}:
            continue
        if allow_glob and character == "*":
            continue
        return False
    return True


def _parse_permission_rule(rule: str) -> _ParsedRule | None:
    """Parse the closed bare-or-single-specifier grammar without retaining input."""
    opening = rule.find("(")
    if opening < 0:
        if ")" in rule or not _valid_tool_identifier(rule, allow_glob=True):
            return None
        return _ParsedRule(rule, None)
    if opening == 0 or not rule.endswith(")"):
        return None
    tool = rule[:opening]
    specifier = rule[opening + 1 : -1]
    if (
        not specifier
        or "(" in specifier
        or ")" in specifier
        or not _valid_tool_identifier(tool, allow_glob=False)
    ):
        return None
    return _ParsedRule(tool, specifier)


def _ascii_lower(value: str) -> str:
    return "".join(
        chr(ord(character) + 32) if "A" <= character <= "Z" else character for character in value
    )


def _has_sensitive_ascii_token(value: str) -> bool:
    sensitive_tokens = {"credential", "credentials", "secret", "secrets", "token", "tokens"}
    token_start: int | None = None
    for index, character in enumerate(value):
        is_ascii_alphanumeric = (
            "a" <= character <= "z" or "A" <= character <= "Z" or "0" <= character <= "9"
        )
        if is_ascii_alphanumeric:
            if token_start is None:
                token_start = index
        elif token_start is not None:
            if value[token_start:index] in sensitive_tokens:
                return True
            token_start = None
    return token_start is not None and value[token_start:] in sensitive_tokens


def _sensitive_path(parts: tuple[str, ...], *, credential_pair_start: int = 0) -> bool:
    literal_lowered = tuple(_ascii_lower(part) for part in parts)
    lowered = tuple(part for part in literal_lowered if part not in {"", ".", "*", "**"})
    sensitive_segments = {
        ".agents",
        ".anthropic",
        ".aws",
        ".azure",
        ".bash_history",
        ".claude",
        ".codex",
        ".config/gcloud",
        ".cursor",
        ".docker",
        ".env",
        ".git-credentials",
        ".kube",
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".ssh",
        ".zsh_history",
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "kubeconfig",
    }
    if any(part in sensitive_segments or part.startswith(".env.") for part in lowered):
        return True
    if any(_has_sensitive_ascii_token(part) for part in lowered):
        return True
    credential_store_names = {"gcloud", "gh", "glab"}
    if any(
        literal_lowered[index] == ".config" and literal_lowered[index + 1] in credential_store_names
        for index in range(credential_pair_start, len(literal_lowered) - 1)
    ):
        return True
    if any(part.endswith((".key", ".pem")) for part in lowered):
        return True
    return False


def _classify_path_specifier(specifier: str) -> _PathClassification:
    if (
        not specifier
        or "\0" in specifier
        or "\\" in specifier
        or specifier.startswith("$")
        or (len(specifier) >= 2 and specifier[0].isalpha() and specifier[1] == ":")
        or (specifier.startswith("~") and specifier != "~" and not specifier.startswith("~/"))
        or specifier.startswith("///")
        or ("//" in specifier and not specifier.startswith("//"))
    ):
        return _PathClassification("invalid", "invalid", False)

    if specifier in _WHOLE_PERMISSION_PATHS:
        scope = "root" if specifier.startswith("//") else "home"
        return _PathClassification(scope, specifier, True)

    anchor = "project"
    remainder = specifier
    prefix = ""
    if specifier.startswith("//"):
        anchor = "external"
        remainder = specifier[2:]
        prefix = "//"
    elif specifier == "~" or specifier.startswith("~/"):
        anchor = "external"
        remainder = specifier[2:] if specifier.startswith("~/") else ""
        prefix = "~/"
    elif specifier.startswith("/"):
        remainder = specifier[1:]
        prefix = "/"
    else:
        while remainder.startswith("../"):
            anchor = "external"
            prefix += "../"
            remainder = remainder[3:]
        if remainder == "..":
            anchor = "external"
            prefix += ".."
            remainder = ""
        elif remainder.startswith("./"):
            remainder = remainder[2:]
        elif remainder == ".":
            remainder = ""

    if "//" in remainder:
        return _PathClassification("invalid", "invalid", False)
    raw_parts = tuple(part for part in remainder.split("/") if part)
    if any(part == ".." for part in raw_parts):
        return _PathClassification("invalid", "invalid", False)
    if _sensitive_path(raw_parts):
        normalized = f"{prefix}{'/'.join(raw_parts)}"
        return _PathClassification("sensitive", normalized, False)

    normalized = f"{prefix}{'/'.join(raw_parts)}"
    broad = bool(
        anchor == "external"
        and (
            (raw_parts and raw_parts[-1] in {"*", "**"})
            or normalized.endswith("/**")
            or normalized.endswith("/**/*")
        )
    )
    return _PathClassification(anchor, normalized, broad)


def _normalize_bash_specifier(specifier: str) -> str:
    if specifier.endswith(":*"):
        return f"{specifier[:-2]} *"
    return specifier


def _valid_domain_pattern(value: str) -> str | None:
    if not value or not value.isascii():
        return None
    normalized = value[:-1] if value.endswith(".") else value
    if not normalized or len(normalized) > 253 or normalized.endswith("."):
        return None
    labels = normalized.split(".")
    for label in labels:
        if not label or len(label) > 63 or label.startswith("-") or label.endswith("-"):
            return None
        if any(not (character.isalnum() or character in {"-", "*"}) for character in label):
            return None
    return normalized.lower()


def _mcp_classification(tool: str) -> tuple[str, str, str] | None:
    parts = tool.split("__")
    if len(parts) not in {2, 3} or parts[0] != "mcp":
        return None
    server = parts[1]
    if not _valid_tool_identifier(server, allow_glob=False):
        return None
    if len(parts) == 2:
        return "mcp_server_wide", "HIGH", f"mcp__{server}__*"
    mcp_tool = parts[2]
    if not _valid_tool_identifier(mcp_tool, allow_glob=True):
        return None
    if mcp_tool == "*":
        return "mcp_server_wide", "HIGH", f"mcp__{server}__*"
    if "*" in mcp_tool:
        return "mcp_partial_tool", "MEDIUM", tool
    return "mcp_exact_tool", "MEDIUM", tool


def _normalized_rule_identity(parsed: _ParsedRule) -> tuple[str, str]:
    tool = "powershell" if _ascii_lower(parsed.tool) == "powershell" else parsed.tool
    specifier = parsed.specifier
    if tool == "Monitor" and specifier is not None:
        tool = "Bash"
    if tool in {"Bash", "powershell", "Monitor"}:
        if specifier is None or specifier == "*":
            return tool, f"{tool}(*)"
        normalized_specifier = _normalize_bash_specifier(specifier)
        if tool == "powershell":
            normalized_specifier = _ascii_lower(normalized_specifier)
        return tool, f"{tool}({normalized_specifier})"
    if tool == "WebFetch":
        if specifier is None:
            return tool, "WebFetch(domain:*)"
        if specifier.startswith("domain:"):
            domain = _valid_domain_pattern(specifier[7:])
            if domain is not None:
                return tool, f"WebFetch(domain:{domain})"
    if specifier is None and tool.startswith("mcp__"):
        mcp = _mcp_classification(tool)
        if mcp is not None:
            return mcp[2], mcp[2]
    identity = tool if specifier is None else f"{tool}({specifier})"
    return tool, identity


def _classify_restriction(
    rule: str, *, key: str, source_line: int
) -> tuple[_Restriction | None, PermissionDiagnostic, bool]:
    parsed = _parse_permission_rule(rule)
    if parsed is None:
        return (
            None,
            _diagnostic("unknown_rule", True, source_line, identity=f"{key}:{rule}"),
            False,
        )
    tool_identifier, normalized_identity = _normalized_rule_identity(parsed)
    original_tool = "powershell" if _ascii_lower(parsed.tool) == "powershell" else parsed.tool
    mcp = _mcp_classification(parsed.tool) if parsed.specifier is None else None
    mcp_server = (
        parsed.tool.split("__")[1] if mcp is not None and mcp[0] == "mcp_server_wide" else None
    )
    tool_wide = parsed.specifier is None or normalized_identity in {
        "Bash(*)",
        "powershell(*)",
        "WebFetch(domain:*)",
    }
    restriction = _Restriction(
        tool_identifier=tool_identifier,
        original_tool_identifier=original_tool,
        normalized_identity=normalized_identity,
        tool_glob=parsed.specifier is None and "*" in parsed.tool,
        tool_wide=tool_wide,
        mcp_server=mcp_server,
        source_line=source_line,
    )
    diagnostic = _diagnostic(
        "restrictive_rule", False, source_line, identity=f"{key}:{normalized_identity}"
    )
    return restriction, diagnostic, True


def _compile_literal(value: str) -> _CompiledLiteral:
    prefix = [0] * len(value)
    matched = 0
    for index in range(1, len(value)):
        while matched and value[index] != value[matched]:
            matched = prefix[matched - 1]
        if value[index] == value[matched]:
            matched += 1
            prefix[index] = matched
    return _CompiledLiteral(value, tuple(prefix))


def _literal_index(value: str, literal: _CompiledLiteral, start: int, end: int) -> int:
    matched = 0
    for index in range(start, end):
        while matched and value[index] != literal.value[matched]:
            matched = literal.prefix[matched - 1]
        if value[index] == literal.value[matched]:
            matched += 1
            if matched == len(literal.value):
                return index - len(literal.value) + 1
    return -1


def _compile_tool_glob(pattern: str) -> _CompiledToolGlob:
    if "*" not in pattern:
        return _CompiledToolGlob(len(pattern), pattern, False, False, ())
    segments = tuple(_compile_literal(segment) for segment in pattern.split("*") if segment)
    return _CompiledToolGlob(
        len(pattern),
        None,
        pattern.startswith("*"),
        pattern.endswith("*"),
        segments,
    )


def _match_compiled_tool_glob(compiled: _CompiledToolGlob, value: str) -> bool:
    if compiled.exact_literal is not None:
        return compiled.exact_literal == value
    segments = compiled.segments
    if not segments:
        return True

    start = 0
    end = len(value)
    first_segment = 0
    last_segment = len(segments)
    if not compiled.leading_wildcard:
        leading = segments[0].value
        if not value.startswith(leading):
            return False
        start = len(leading)
        first_segment = 1
    if not compiled.trailing_wildcard:
        trailing = segments[-1].value
        if not value.endswith(trailing):
            return False
        end -= len(trailing)
        last_segment -= 1
    if start > end:
        return False
    for literal in segments[first_segment:last_segment]:
        found = _literal_index(value, literal, start, end)
        if found < 0:
            return False
        start = found + len(literal.value)
    return start <= end


def _bounded_glob_match(pattern: str, value: str) -> bool:
    return _match_compiled_tool_glob(_compile_tool_glob(pattern), value)


def _deduplicate_allow_candidates(
    candidates: list[_AllowCandidate],
) -> tuple[_AllowCandidate, ...]:
    unique: dict[tuple[str, ...], _AllowCandidate] = {}
    for candidate in candidates:
        semantic_identity: tuple[str, ...]
        if candidate.mcp_server is not None:
            semantic_identity = (
                "mcp",
                candidate.normalized_identity,
                candidate.mcp_server,
            )
        else:
            semantic_identity = (
                "rule",
                candidate.normalized_identity,
                candidate.tool_identifier,
                candidate.original_tool_identifier,
            )
        previous = unique.get(semantic_identity)
        if previous is None or candidate.grant.source_line < previous.grant.source_line:
            unique[semantic_identity] = candidate
    return tuple(unique.values())


def _deduplicate_restrictions(
    restrictions: tuple[_Restriction, ...],
) -> tuple[_Restriction, ...]:
    unique: dict[tuple[str, ...], _Restriction] = {}
    for restriction in restrictions:
        semantic_identity: tuple[str, ...]
        if restriction.mcp_server is not None:
            semantic_identity = ("mcp_server", restriction.mcp_server)
        elif restriction.tool_glob:
            semantic_identity = ("tool_glob", restriction.tool_identifier)
        else:
            semantic_identity = (
                "rule",
                restriction.normalized_identity,
                restriction.tool_identifier,
                restriction.original_tool_identifier,
                "wide" if restriction.tool_wide else "exact",
            )
        previous = unique.get(semantic_identity)
        if previous is None or restriction.source_line < previous.source_line:
            unique[semantic_identity] = restriction
    return tuple(unique.values())


def _index_restrictions(restrictions: tuple[_Restriction, ...]) -> _RestrictionIndex:
    glob_patterns = sorted(
        {
            restriction.tool_identifier
            for restriction in restrictions
            if restriction.tool_glob and restriction.mcp_server is None
        }
    )
    return _RestrictionIndex(
        exact_identities=frozenset(restriction.normalized_identity for restriction in restrictions),
        tool_wide_identifiers=frozenset(
            restriction.tool_identifier
            for restriction in restrictions
            if restriction.tool_wide
            and not restriction.tool_glob
            and restriction.mcp_server is None
        ),
        original_tool_wide_identifiers=frozenset(
            restriction.original_tool_identifier
            for restriction in restrictions
            if restriction.tool_wide
            and not restriction.tool_glob
            and restriction.mcp_server is None
        ),
        mcp_servers=frozenset(
            restriction.mcp_server
            for restriction in restrictions
            if restriction.mcp_server is not None
        ),
        tool_globs=tuple(_compile_tool_glob(pattern) for pattern in glob_patterns),
    )


def _indexed_coverage_without_globs(
    candidate: _AllowCandidate,
    restriction_index: _RestrictionIndex,
) -> bool:
    return (
        candidate.normalized_identity in restriction_index.exact_identities
        or (
            candidate.mcp_server is not None
            and candidate.mcp_server in restriction_index.mcp_servers
        )
        or candidate.tool_identifier in restriction_index.tool_wide_identifiers
        or candidate.original_tool_identifier in restriction_index.original_tool_wide_identifiers
    )


def _indexed_restriction_coverage(
    candidates: tuple[_AllowCandidate, ...],
    restrictions: tuple[_Restriction, ...],
) -> tuple[bool, ...] | None:
    restriction_index = _index_restrictions(restrictions)
    indexed_coverage = tuple(
        _indexed_coverage_without_globs(candidate, restriction_index) for candidate in candidates
    )
    unmatched_tool_identifiers = tuple(
        sorted(
            {
                candidate.tool_identifier
                for candidate, covered in zip(candidates, indexed_coverage, strict=True)
                if not covered
            }
        )
    )
    glob_covered_identifiers: set[str] = set()
    charged_characters = 0
    for compiled in restriction_index.tool_globs:
        for tool_identifier in unmatched_tool_identifiers:
            charge = compiled.character_count + len(tool_identifier)
            if charged_characters + charge > MAX_PERMISSION_GLOB_MATCH_CHARS_PER_DOCUMENT:
                return None
            charged_characters += charge
            if _match_compiled_tool_glob(compiled, tool_identifier):
                glob_covered_identifiers.add(tool_identifier)
    return tuple(
        covered or candidate.tool_identifier in glob_covered_identifiers
        for candidate, covered in zip(candidates, indexed_coverage, strict=True)
    )


def _diagnostic(
    kind: str,
    affects_completeness: bool,
    source_line: int,
    *,
    identity: str,
) -> PermissionDiagnostic:
    if _DIAGNOSTIC_COMPLETENESS.get(kind) is not affects_completeness:
        raise ValueError("unsupported permission diagnostic")
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
    mode_context: bool = False,
) -> PermissionGrant:
    if _GRANT_SEVERITY_BY_KIND.get(grant_kind) != severity:
        raise ValueError("unsupported permission grant kind")
    context = _mode_context(source_kind) if mode_context else _rule_context(source_kind)
    activation_requirement, interface_applicability, tracking_status = context
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


def _allow_grant_candidate(
    parsed: _ParsedRule,
    grant_kind: str,
    severity: str,
    source_kind: str,
    source_line: int,
    *,
    identity: str | None = None,
) -> _AllowCandidate:
    tool_identifier, normalized_identity = _normalized_rule_identity(parsed)
    original_tool = "powershell" if _ascii_lower(parsed.tool) == "powershell" else parsed.tool
    mcp = _mcp_classification(parsed.tool) if parsed.specifier is None else None
    mcp_server = parsed.tool.split("__")[1] if mcp is not None else None
    safe_identity = normalized_identity if identity is None else identity
    return _AllowCandidate(
        _grant(grant_kind, severity, source_kind, source_line, identity=safe_identity),
        tool_identifier,
        original_tool,
        normalized_identity,
        mcp_server,
    )


def _classify_allow_rule(
    rule: str,
    *,
    source_kind: str,
    source_line: int,
) -> tuple[_AllowCandidate | None, PermissionDiagnostic | None, bool]:
    parsed = _parse_permission_rule(rule)
    if parsed is None:
        return (
            None,
            _diagnostic("unknown_rule", True, source_line, identity=rule),
            False,
        )

    tool = parsed.tool
    specifier = parsed.specifier
    if _ascii_lower(tool) == "powershell":
        tool = "PowerShell"
        parsed = _ParsedRule(tool, specifier)

    if tool.startswith("mcp__"):
        if specifier is not None:
            return (
                None,
                _diagnostic("unknown_rule", True, source_line, identity=rule),
                False,
            )
        mcp = _mcp_classification(tool)
        if mcp is not None:
            grant_kind, severity, identity = mcp
            return (
                _allow_grant_candidate(
                    parsed,
                    grant_kind,
                    severity,
                    source_kind,
                    source_line,
                    identity=identity,
                ),
                None,
                True,
            )
        if tool == "mcp__*":
            return (
                None,
                _diagnostic("ignored_allow_rule_glob", False, source_line, identity=tool),
                True,
            )
        return None, _diagnostic("unknown_rule", True, source_line, identity=rule), False

    if specifier is None and "*" in tool:
        return (
            None,
            _diagnostic("ignored_allow_rule_glob", False, source_line, identity=tool),
            True,
        )

    if tool in _SHELL_TOOLS:
        if specifier in {None, "*"}:
            return (
                _allow_grant_candidate(
                    parsed, "tool_wide_execution", "CRITICAL", source_kind, source_line
                ),
                None,
                True,
            )
        return (
            _allow_grant_candidate(parsed, "scoped_execution", "MEDIUM", source_kind, source_line),
            None,
            True,
        )

    if tool in _FILESYSTEM_TOOLS:
        bare_grants = {
            "Read": ("tool_wide_read", "CRITICAL"),
            "Edit": ("tool_wide_edit", "CRITICAL"),
            "Write": ("tool_wide_write", "CRITICAL"),
            "NotebookEdit": ("broad_notebook_edit", "HIGH"),
            "MultiEdit": ("broad_multi_edit", "HIGH"),
            "Glob": ("filesystem_enumeration", "MEDIUM"),
            "Grep": ("filesystem_search", "MEDIUM"),
            "LSP": ("code_intelligence", "MEDIUM"),
        }
        if specifier is None:
            grant_kind, severity = bare_grants[tool]
            return (
                _allow_grant_candidate(parsed, grant_kind, severity, source_kind, source_line),
                None,
                True,
            )
        if tool in {"Grep", "LSP"}:
            return (
                None,
                _diagnostic("runtime_uncertain_rule", True, source_line, identity=rule),
                False,
            )
        if tool not in {"Read", "Edit"}:
            return (
                None,
                _diagnostic("ignored_path_qualifier", False, source_line, identity=rule),
                True,
            )
        path = _classify_path_specifier(specifier)
        if path.scope == "invalid":
            return None, _diagnostic("unknown_rule", True, source_line, identity=rule), False
        if tool == "Read":
            if path.scope in {"root", "home"}:
                kind, severity = "root_or_home_wide_read", "CRITICAL"
            elif path.scope == "sensitive":
                kind, severity = "sensitive_read", "HIGH"
            elif path.scope == "external":
                kind, severity = "external_read", "MEDIUM"
            else:
                return None, None, True
        elif path.scope in {"root", "home"}:
            kind, severity = "root_or_home_wide_edit", "CRITICAL"
        elif path.scope == "sensitive":
            kind, severity = "sensitive_edit", "HIGH"
        elif path.broad:
            kind, severity = "broad_external_edit", "HIGH"
        else:
            kind, severity = "scoped_edit", "MEDIUM"
        return (
            _allow_grant_candidate(parsed, kind, severity, source_kind, source_line),
            None,
            True,
        )

    if tool == "WebFetch":
        if specifier is None:
            return (
                _allow_grant_candidate(
                    parsed, "all_domain_fetch", "HIGH", source_kind, source_line
                ),
                None,
                True,
            )
        if specifier == "*":
            return (
                None,
                _diagnostic("unsupported_allow_specifier", False, source_line, identity=rule),
                True,
            )
        if not specifier.startswith("domain:"):
            return None, _diagnostic("unknown_rule", True, source_line, identity=rule), False
        domain = _valid_domain_pattern(specifier[7:])
        if domain is None:
            return None, _diagnostic("unknown_rule", True, source_line, identity=rule), False
        kind, severity = (
            ("all_domain_fetch", "HIGH") if domain == "*" else ("scoped_domain_fetch", "MEDIUM")
        )
        normalized = _ParsedRule(tool, f"domain:{domain}")
        return (
            _allow_grant_candidate(normalized, kind, severity, source_kind, source_line),
            None,
            True,
        )

    if tool == "WebSearch":
        if specifier is not None:
            return (
                None,
                _diagnostic("unsupported_allow_specifier", False, source_line, identity=rule),
                True,
            )
        return (
            _allow_grant_candidate(parsed, "network_search", "MEDIUM", source_kind, source_line),
            None,
            True,
        )

    if tool in _BARE_EXTERNAL_UPLOAD_TOOLS:
        if specifier is not None:
            return (
                None,
                _diagnostic("unsupported_allow_specifier", False, source_line, identity=rule),
                True,
            )
        return (
            _allow_grant_candidate(
                parsed, "external_content_upload", "HIGH", source_kind, source_line
            ),
            None,
            True,
        )

    if tool == "Skill":
        return (
            _allow_grant_candidate(parsed, "skill_invocation", "MEDIUM", source_kind, source_line),
            None,
            True,
        )

    if tool in _BARE_ROUTE_GRANTS:
        if specifier is not None:
            return (
                None,
                _diagnostic("unsupported_allow_specifier", False, source_line, identity=rule),
                True,
            )
        kind, severity = _BARE_ROUTE_GRANTS[tool]
        return (
            _allow_grant_candidate(parsed, kind, severity, source_kind, source_line),
            None,
            True,
        )

    if tool in _KNOWN_NON_GRANT_TOOLS:
        kind = "known_non_grant_tool" if specifier is None else "unsupported_allow_specifier"
        return None, _diagnostic(kind, False, source_line, identity=rule), True
    return None, _diagnostic("unknown_rule", True, source_line, identity=rule), False


def _collapse_lexical_parts(value: str, *, clamp_root: bool = False) -> tuple[str, ...]:
    collapsed: list[str] = []
    for part in value.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if collapsed and collapsed[-1] != "..":
                collapsed.pop()
            elif not clamp_root:
                collapsed.append(part)
            continue
        collapsed.append(part)
    return tuple(collapsed)


def _ecmascript_trim(value: str) -> str:
    start = 0
    end = len(value)
    while start < end and value[start] in _ECMASCRIPT_TRIM_CHARACTERS:
        start += 1
    while end > start and value[end - 1] in _ECMASCRIPT_TRIM_CHARACTERS:
        end -= 1
    return value[start:end]


def _has_ascii_drive_prefix(value: str) -> bool:
    return (
        len(value) >= 2 and ("A" <= value[0] <= "Z" or "a" <= value[0] <= "z") and value[1] == ":"
    )


def _is_unicode_scalar(character: str) -> bool:
    code_point = ord(character)
    return not 0xD800 <= code_point <= 0xDFFF


def _valid_unc_server(value: str) -> bool:
    forbidden = frozenset('\\/:*?"<>|,')
    return (
        bool(value)
        and value not in {".", ".."}
        and all(
            _is_unicode_scalar(character)
            and not ord(character) <= 0x1F
            and ord(character) != 0x7F
            and not character.isspace()
            and character not in forbidden
            for character in value
        )
    )


def _valid_unc_share(value: str) -> bool:
    forbidden = frozenset('"\\/[]:|<>+=;,*?')
    utf16_units = sum(1 if ord(character) <= 0xFFFF else 2 for character in value)
    return (
        bool(value)
        and value not in {".", ".."}
        and utf16_units <= 80
        and all(
            _is_unicode_scalar(character)
            and not ord(character) <= 0x1F
            and character not in forbidden
            for character in value
        )
    )


def _normalize_win32_tail_parts(value: str) -> tuple[str, ...]:
    collapsed: list[str] = []
    trailing_separator = value.endswith("/")
    raw_parts = value.split("/")
    final_index = len(raw_parts) - 1
    for index, part in enumerate(raw_parts):
        if not part:
            continue
        if part == ".":
            continue
        if part == "..":
            if collapsed:
                collapsed.pop()
            continue

        is_final = index == final_index and not trailing_separator
        normalized = part
        if is_final:
            normalized = normalized.rstrip(" ")
            if normalized == ".":
                continue
            if normalized == "..":
                if collapsed:
                    collapsed.pop()
                continue
            normalized = normalized.rstrip(" .")
            if not normalized:
                continue
        else:
            terminal_dot_count = len(normalized) - len(normalized.rstrip("."))
            if terminal_dot_count == 1:
                normalized = normalized[:-1]
        if normalized:
            collapsed.append(normalized)
    return tuple(collapsed)


def _normalize_unc_parts(
    value: str, *, minimum_components: int, win32_tail: bool
) -> tuple[str, ...] | None:
    trimmed = value.rstrip("/")
    if not trimmed or value.startswith("/") or "//" in trimmed:
        return None
    raw_parts = tuple(trimmed.split("/"))
    if len(raw_parts) < minimum_components:
        return None
    if not _valid_unc_server(raw_parts[0]):
        return None
    if len(raw_parts) == 1:
        return raw_parts
    if not _valid_unc_share(raw_parts[1]):
        return None
    tail = "/".join(raw_parts[2:])
    if len(raw_parts) > 2 and value.endswith("/"):
        tail += "/"
    normalized_tail = (
        _normalize_win32_tail_parts(tail)
        if win32_tail
        else _collapse_lexical_parts(tail, clamp_root=True)
    )
    return (*raw_parts[:2], *normalized_tail)


def _platform_dependent_directory(
    identity: str, *, source_line: int
) -> tuple[PermissionGrant | None, tuple[PermissionDiagnostic, ...], bool]:
    return (
        None,
        (_diagnostic("platform_dependent_path", True, source_line, identity=identity),),
        False,
    )


def _conditional_windows_directory(
    identity: str,
    parts: tuple[str, ...],
    *,
    whole_root: bool,
    credential_pair_start: int = 0,
    source_kind: str,
    source_line: int,
) -> tuple[PermissionGrant | None, tuple[PermissionDiagnostic, ...], bool]:
    if whole_root:
        grant_kind, severity = "root_or_home_additional_directory", "CRITICAL"
    elif _sensitive_path(parts, credential_pair_start=credential_pair_start):
        grant_kind, severity = "sensitive_additional_directory", "HIGH"
    else:
        grant_kind, severity = "external_additional_directory", "MEDIUM"
    grant = _grant(
        grant_kind,
        severity,
        source_kind,
        source_line,
        identity=f"additional:{identity}",
    )
    return (
        grant,
        (
            _diagnostic("platform_dependent_path", True, source_line, identity=identity),
            _diagnostic(
                "directory_existence_static_unknown", False, source_line, identity=identity
            ),
        ),
        True,
    )


def _classify_additional_directory(
    value: str,
    *,
    source_kind: str,
    source_line: int,
) -> tuple[PermissionGrant | None, tuple[PermissionDiagnostic, ...], bool]:
    value = _ecmascript_trim(value)
    if "\0" in value:
        return (
            None,
            (_diagnostic("invalid_path", True, source_line, identity=value),),
            False,
        )

    normalized_windows = value.replace("\\", "/")
    if normalized_windows == "/??" or normalized_windows.startswith("/??/"):
        return _platform_dependent_directory(value, source_line=source_line)
    if normalized_windows == "//??" or normalized_windows.startswith("//??/"):
        return _platform_dependent_directory(value, source_line=source_line)

    if normalized_windows == "//?" or normalized_windows.startswith("//?/"):
        if normalized_windows == "//?":
            return _platform_dependent_directory(value, source_line=source_line)
        extended = normalized_windows[4:]
        if _has_ascii_drive_prefix(extended):
            remainder = extended[2:]
            if not remainder.startswith("/"):
                return _platform_dependent_directory(value, source_line=source_line)
            extended_drive_parts = _collapse_lexical_parts(remainder, clamp_root=True)
            identity = f"drive:{extended[0].upper()}:/{'/'.join(extended_drive_parts)}"
            return _conditional_windows_directory(
                identity,
                extended_drive_parts,
                whole_root=not extended_drive_parts,
                source_kind=source_kind,
                source_line=source_line,
            )
        if len(extended) >= 4 and _ascii_lower(extended[:3]) == "unc" and extended[3] == "/":
            extended_unc_parts = _normalize_unc_parts(
                extended[4:], minimum_components=2, win32_tail=False
            )
            if extended_unc_parts is not None:
                identity = f"unc:/{'/'.join(extended_unc_parts)}"
                return _conditional_windows_directory(
                    identity,
                    extended_unc_parts,
                    whole_root=False,
                    credential_pair_start=1,
                    source_kind=source_kind,
                    source_line=source_line,
                )
        return _platform_dependent_directory(value, source_line=source_line)

    if normalized_windows == "//." or normalized_windows == "//./":
        return _conditional_windows_directory(
            "drive-current:/",
            (),
            whole_root=True,
            source_kind=source_kind,
            source_line=source_line,
        )
    if normalized_windows.startswith("//./"):
        return _platform_dependent_directory(value, source_line=source_line)

    is_drive = _has_ascii_drive_prefix(value)
    is_backslash_root = bool(value) and all(character == "\\" for character in value)
    is_unc = (value.startswith("\\\\") or value.startswith("//")) and not all(
        character == "/" for character in value
    )
    if is_drive or is_backslash_root or is_unc:
        if is_drive:
            remainder = value[2:]
            if not remainder.startswith(("/", "\\")):
                return _platform_dependent_directory(value, source_line=source_line)
            normalized_remainder = remainder.replace("\\", "/")
            drive_parts = _normalize_win32_tail_parts(normalized_remainder)
            identity = f"drive:{value[0].upper()}:/{'/'.join(drive_parts)}"
            return _conditional_windows_directory(
                identity,
                drive_parts,
                whole_root=not drive_parts,
                source_kind=source_kind,
                source_line=source_line,
            )
        elif is_backslash_root:
            return _conditional_windows_directory(
                "drive-current:/",
                (),
                whole_root=True,
                source_kind=source_kind,
                source_line=source_line,
            )
        else:
            normalized_remainder = value[2:].replace("\\", "/")
            unc_parts = _normalize_unc_parts(
                normalized_remainder, minimum_components=1, win32_tail=True
            )
            if unc_parts is None:
                return _platform_dependent_directory(value, source_line=source_line)
            identity = f"unc:/{'/'.join(unc_parts)}"
            return _conditional_windows_directory(
                identity,
                unc_parts,
                whole_root=False,
                credential_pair_start=1,
                source_kind=source_kind,
                source_line=source_line,
            )

    if "\\" in value:
        return _platform_dependent_directory(value, source_line=source_line)

    posix_grant_kind: str | None
    posix_severity: str | None
    if value and all(character == "/" for character in value):
        normalized = "/"
        posix_grant_kind, posix_severity = "root_or_home_additional_directory", "CRITICAL"
    elif value in {"~", "~/"}:
        normalized = "~/"
        posix_grant_kind, posix_severity = "root_or_home_additional_directory", "CRITICAL"
    else:
        home = value.startswith("~/")
        absolute = value.startswith("/")
        lexical_value = value[2:] if home else value.lstrip("/") if absolute else value
        parts = _collapse_lexical_parts(lexical_value, clamp_root=absolute)
        external = home or absolute or bool(parts and parts[0] == "..")
        normalized = (("~/" if home else "/" if absolute else "") + "/".join(parts)) or "."
        if (home or absolute) and not parts:
            posix_grant_kind, posix_severity = (
                "root_or_home_additional_directory",
                "CRITICAL",
            )
        elif not external:
            posix_grant_kind = posix_severity = None
        elif _sensitive_path(parts):
            posix_grant_kind, posix_severity = "sensitive_additional_directory", "HIGH"
        else:
            posix_grant_kind, posix_severity = "external_additional_directory", "MEDIUM"

    posix_grant = (
        None
        if posix_grant_kind is None or posix_severity is None
        else _grant(
            posix_grant_kind,
            posix_severity,
            source_kind,
            source_line,
            identity=f"additional:{normalized}",
        )
    )
    diagnostic = _diagnostic(
        "directory_existence_static_unknown", False, source_line, identity=normalized
    )
    return posix_grant, (diagnostic,), True


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
    mitigated_allow_count: int,
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
        "mitigated_allow_count": mitigated_allow_count,
        "max_severity": max_severity,
        "blocking_critical": any(grant.blocking_critical for grant in grants),
    }
    payload = _AGGREGATE_PREFIX + _canonical_bytes(safe)
    return f"sha256:{sha256(payload).hexdigest()}"


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
    allow_candidates: list[_AllowCandidate] = []
    deny_restrictions: list[_Restriction] = []
    ask_restrictions: list[_Restriction] = []
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
                item_line = _line_for_rule(source_lines, key, item_index)
                if not isinstance(item, str):
                    diagnostics.append(
                        _diagnostic(
                            "wrong_type",
                            True,
                            item_line,
                            identity=f"{key}:{_safe_identity(item)}",
                        )
                    )
                    continue
                if key == "additionalDirectories":
                    grant, directory_diagnostics, valid = _classify_additional_directory(
                        item, source_kind=source_kind, source_line=item_line
                    )
                    has_valid_content = has_valid_content or valid
                    if grant is not None:
                        grants.append(grant)
                    diagnostics.extend(directory_diagnostics)
                    continue
                if key == "allow":
                    candidate, diagnostic, valid = _classify_allow_rule(
                        item, source_kind=source_kind, source_line=item_line
                    )
                    has_valid_content = has_valid_content or valid
                    if candidate is not None:
                        allow_candidates.append(candidate)
                    if diagnostic is not None:
                        diagnostics.append(diagnostic)
                    continue
                restriction, diagnostic, valid = _classify_restriction(
                    item, key=key, source_line=item_line
                )
                has_valid_content = has_valid_content or valid
                diagnostics.append(diagnostic)
                if restriction is not None:
                    if key == "deny":
                        deny_restrictions.append(restriction)
                    else:
                        ask_restrictions.append(restriction)
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
                        "permission_mode_accept_edits",
                        "MEDIUM",
                        source_kind,
                        line,
                        identity=key,
                        mode_context=True,
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

    validated_candidates = _deduplicate_allow_candidates(allow_candidates)
    validated_restrictions = _deduplicate_restrictions((*deny_restrictions, *ask_restrictions))
    restriction_coverage = _indexed_restriction_coverage(
        validated_candidates, validated_restrictions
    )
    if restriction_coverage is None:
        return PermissionAnalysis(
            True,
            LedgerOutcome.FAILED,
            LedgerReason.COMPONENT_LIMIT,
            (),
            (),
            None,
        )

    global_restriction = any(
        restriction.tool_glob and restriction.tool_identifier == "*"
        for restriction in validated_restrictions
    )
    if bypass_declared and not bypass_disabled:
        if global_restriction:
            diagnostics.append(
                _diagnostic(
                    "bypass_global_restriction",
                    False,
                    bypass_line,
                    identity="global_restriction",
                )
            )
        else:
            grants.append(
                _grant(
                    "permission_mode_bypass",
                    "CRITICAL",
                    source_kind,
                    bypass_line,
                    identity="defaultMode",
                    mode_context=True,
                )
            )

    mitigated_allow_count = 0
    for candidate, mitigated in zip(validated_candidates, restriction_coverage, strict=True):
        if mitigated:
            mitigated_allow_count += 1
            diagnostics.append(
                _diagnostic(
                    "mitigated_allow",
                    False,
                    candidate.grant.source_line,
                    identity=candidate.normalized_identity,
                )
            )
        else:
            grants.append(candidate.grant)

    unique_grants: dict[str, PermissionGrant] = {}
    for grant in grants:
        previous = unique_grants.get(grant.grant_digest)
        if previous is None or grant.source_line < previous.source_line:
            unique_grants[grant.grant_digest] = grant
    unique_diagnostics: dict[str, PermissionDiagnostic] = {}
    for diagnostic in diagnostics:
        previous_diagnostic = unique_diagnostics.get(diagnostic.diagnostic_digest)
        if previous_diagnostic is None or diagnostic.source_line < previous_diagnostic.source_line:
            unique_diagnostics[diagnostic.diagnostic_digest] = diagnostic
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
        mitigated_allow_count=mitigated_allow_count,
    )
    return PermissionAnalysis(
        True, outcome, reason, sorted_grants, sorted_diagnostics, aggregate_digest
    )


def _is_full_digest(value: object) -> bool:
    return isinstance(value, str) and _SHA256_DIGEST.fullmatch(value) is not None


def _is_positive_source_line(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_reportable_grant(grant: object) -> bool:
    if not isinstance(grant, PermissionGrant):
        return False
    if not all(
        type(value) is str
        for value in (
            grant.grant_kind,
            grant.severity,
            grant.activation_requirement,
            grant.interface_applicability,
            grant.tracking_status,
        )
    ):
        return False
    if _GRANT_SEVERITY_BY_KIND.get(grant.grant_kind) != grant.severity:
        return False
    if grant.activation_requirement not in _GRANT_ACTIVATION_REQUIREMENTS:
        return False
    if grant.interface_applicability not in _GRANT_INTERFACE_APPLICABILITY:
        return False
    if grant.tracking_status not in _GRANT_TRACKING_STATUS:
        return False
    if not isinstance(grant.blocking_critical, bool):
        return False
    if grant.blocking_critical is not (grant.severity == "CRITICAL"):
        return False
    if not _is_full_digest(grant.grant_digest):
        return False
    if not _is_positive_source_line(grant.source_line):
        return False

    if grant.grant_kind in _MODE_GRANT_KINDS:
        return (
            grant.activation_requirement == "interface_and_external_policy"
            and grant.interface_applicability == "permission_mode_interface_dependent"
        )
    expected_activation = (
        "workspace_trust"
        if grant.tracking_status == "not_applicable"
        else "local_provenance_and_session_policy"
    )
    return (
        grant.activation_requirement == expected_activation
        and grant.interface_applicability == "claude_code_settings_consumers"
    )


def _valid_reportable_diagnostic(diagnostic: object) -> bool:
    if not isinstance(diagnostic, PermissionDiagnostic):
        return False
    if type(diagnostic.diagnostic_kind) is not str:
        return False
    if not isinstance(diagnostic.affects_completeness, bool):
        return False
    if (
        _DIAGNOSTIC_COMPLETENESS.get(diagnostic.diagnostic_kind)
        is not diagnostic.affects_completeness
    ):
        return False
    return _is_full_digest(diagnostic.diagnostic_digest) and _is_positive_source_line(
        diagnostic.source_line
    )


def _validated_finding_source_kind(analysis: PermissionAnalysis) -> str:
    """Validate an internal analysis before projecting any report-visible data."""
    if analysis.applicable is not True:
        raise ValueError("invalid permission analysis")
    if not isinstance(analysis.grants, tuple) or not isinstance(analysis.diagnostics, tuple):
        raise ValueError("invalid permission analysis")
    if not all(_valid_reportable_grant(grant) for grant in analysis.grants):
        raise ValueError("invalid permission analysis")
    if not all(_valid_reportable_diagnostic(diagnostic) for diagnostic in analysis.diagnostics):
        raise ValueError("invalid permission analysis")
    if not _is_full_digest(analysis.aggregate_digest):
        raise ValueError("invalid permission analysis")
    if len({grant.grant_digest for grant in analysis.grants}) != len(analysis.grants):
        raise ValueError("invalid permission analysis")
    if len({item.diagnostic_digest for item in analysis.diagnostics}) != len(analysis.diagnostics):
        raise ValueError("invalid permission analysis")

    incomplete = any(item.affects_completeness for item in analysis.diagnostics)
    expected_outcome = LedgerOutcome.PARTIAL if incomplete else LedgerOutcome.COMPLETED
    expected_reason = LedgerReason.INVALID_CONFIGURATION if incomplete else None
    if analysis.outcome is not expected_outcome or analysis.reason is not expected_reason:
        raise ValueError("invalid permission analysis")

    tracking_statuses = {grant.tracking_status for grant in analysis.grants}
    if tracking_statuses == {"not_applicable"}:
        return "project_settings"
    if tracking_statuses == {"unknown"}:
        return "project_local_settings"
    raise ValueError("invalid permission analysis")


def build_bh3_finding(analysis: PermissionAnalysis, *, source_path: str) -> Finding | None:
    """Build one structurally safe BH3 finding for retained reportable grants."""
    if not isinstance(analysis, PermissionAnalysis):
        raise ValueError("invalid permission analysis")
    if not analysis.grants:
        return None
    if not isinstance(source_path, str):
        raise ValueError("invalid permission analysis")
    source_kind = _validated_finding_source_kind(analysis)
    max_severity = max(
        (grant.severity for grant in analysis.grants), key=_SEVERITY_RANK.__getitem__
    )
    evidence: dict[str, object] = {
        "schema": _EVIDENCE_SCHEMA,
        "claude_semantics_snapshot": _SEMANTICS_SNAPSHOT,
        "source_kind": source_kind,
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
