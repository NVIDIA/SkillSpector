# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, cache-backed inventory for Claude Code hook declarations."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Final, cast

import yaml  # type: ignore[import-untyped]
from yaml.resolver import BaseResolver  # type: ignore[import-untyped]

from skillspector.inspection_ledger import (
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    analyzer_status_for_events,
    inspection_work_id,
    ledger_event,
)
from skillspector.models import Finding
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from .bundled_hook_flow import (
    DocumentFlowInput,
    FlowWorkRef,
    FlowWorkResult,
    HandlerFlowInput,
    UserConfigProfile,
    analyze_documents,
    build_user_config_profile,
    capture_handler,
)
from .bundled_hook_runtime import (
    HookRegistration,
    registration_severity,
)
from .bundled_hook_runtime import (
    normalize_registration as _normalize_registration,
)
from .bundled_permission_grants import (
    PermissionAnalysis,
    PermissionSourceLines,
    analyze_permission_grants,
    build_bh3_finding,
)
from .static_runner import MAX_FILE_CHARS

ANALYZER_ID: Final = "bundled_execution_surface"
_PLUGIN_DEFAULT_PATH: Final = "hooks/hooks.json"
_EVIDENCE_SCHEMA: Final = "skillspector.bundled_hook.v1"
_SEMANTICS_SNAPSHOT: Final = "2.1.238"
_PLUGIN_METADATA_DIRECTORY: Final = ".claude-plugin"
_PLUGIN_MANIFEST_FILENAME: Final = "plugin.json"
_PLUGIN_MARKETPLACE_FILENAME: Final = "marketplace.json"
_MANIFEST_COMPONENT_FIELDS: Final = frozenset(
    {
        "hooks",
        "skills",
        "commands",
        "agents",
        "mcpServers",
        "lspServers",
        "outputStyles",
        "workflows",
        "experimental",
    }
)
# Settings-file hooks can load in never-trusted headless sessions. These labels
# describe session scope rather than implying that permission grants were trusted.
_PROJECT_SETTINGS: Final = {
    ".claude/settings.json": ("project_settings", "project_session"),
    ".claude/settings.local.json": ("project_local_settings", "project_local_session"),
}
_FRONTMATTER_DELIMITER: Final = re.compile(r"^(?:---|\.\.\.)[ \t]*$")
_MAX_YAML_COLLECTION_DEPTH: Final = 64
_MAX_YAML_NODES: Final = 2048
_MAX_REGISTRATIONS_PER_DOCUMENT: Final = 2048
_MAX_HOOK_STRUCTURE_ITEMS: Final = 8192


class InvalidHookConfigurationError(ValueError):
    """A supported runtime source cannot be safely interpreted."""


class BinaryHookConfigurationError(InvalidHookConfigurationError):
    """A hook configuration contains binary data."""


class HookConfigurationSizeLimitError(InvalidHookConfigurationError):
    """A hook configuration exceeds the bounded parser input limit."""

    def __init__(self, observed_characters: int) -> None:
        super().__init__("hook configuration exceeds character limit")
        self.observed_characters = observed_characters


class HookRegistrationLimitError(InvalidHookConfigurationError):
    """A hook document exceeds the bounded registration cardinality."""


class _DuplicateKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader which rejects duplicate mapping keys at every depth."""


def _construct_unique_mapping(
    loader: _DuplicateKeySafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            if key in mapping:
                raise InvalidHookConfigurationError("duplicate YAML key")
        except TypeError as exc:
            raise InvalidHookConfigurationError("YAML mapping key is not scalar") from exc
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_DuplicateKeySafeLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


@dataclass(frozen=True)
class HookDocument:
    """One immutable, cache-backed hook declaration document."""

    source_kind: str
    declaration_roles: tuple[str, ...]
    source_path: str
    activation_lifetime: str
    content_digest: str
    registrations: tuple[HookRegistration, ...]
    flow_inputs: tuple[HandlerFlowInput, ...] = field(repr=False)
    runtime_status: str = "declared_unclassified"


@dataclass(frozen=True)
class _SettingsWork:
    """One parse-once project settings document and its safe permission result."""

    source_path: str
    source_kind: str
    content_digest: str
    source_identity_digest: str
    raw: dict[str, object] | None
    parse_error: BaseException | None
    permission_analysis: PermissionAnalysis | None
    permission_source_lines: PermissionSourceLines


@dataclass(frozen=True)
class _RegistrationSet:
    """Parallel normalized and raw-flow records for one parsed hook map."""

    registrations: tuple[HookRegistration, ...]
    flow_inputs: tuple[HandlerFlowInput, ...] = field(repr=False)


@dataclass(frozen=True)
class MarketplaceEntry:
    """A validated local or remote marketplace plugin declaration."""

    marketplace_path: str
    ledger_path: str
    index: int
    plugin_root: str | None
    strict: bool
    hooks: object | None
    skills: object | None
    commands: object | None
    handler_lines: tuple[int, ...] = ()
    source_is_root: bool = False


def _digest(domain: str, value: str) -> str:
    payload = f"skillspector.bundled_hook.v1\0{domain}\0{value}".encode()
    return f"sha256:{sha256(payload).hexdigest()}"


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidHookConfigurationError("duplicate JSON key")
        result[key] = value
    return result


def _load_json(content: str) -> dict[str, object]:
    if "\x00" in content:
        raise BinaryHookConfigurationError("binary hook configuration")
    if len(content) > MAX_FILE_CHARS:
        raise HookConfigurationSizeLimitError(len(content))

    def reject_nonfinite_json_constant(value: str) -> object:
        raise InvalidHookConfigurationError(f"non-finite JSON constant: {value}")

    try:
        raw = json.loads(
            content,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_nonfinite_json_constant,
        )
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise InvalidHookConfigurationError("malformed JSON") from exc
    if not isinstance(raw, dict):
        raise InvalidHookConfigurationError("JSON root must be an object")
    return cast(dict[str, object], raw)


def _validate_yaml_before_construction(frontmatter: str) -> None:
    """Reject alias graphs and oversized YAML collections before object construction."""
    collection_depth = 0
    node_count = 0
    try:
        for event in yaml.parse(frontmatter):
            if isinstance(event, yaml.events.AliasEvent):
                raise InvalidHookConfigurationError("YAML aliases are unsupported")
            if isinstance(event, (yaml.events.MappingStartEvent, yaml.events.SequenceStartEvent)):
                collection_depth += 1
                node_count += 1
                if collection_depth > _MAX_YAML_COLLECTION_DEPTH:
                    raise InvalidHookConfigurationError("YAML collection depth exceeds limit")
            elif isinstance(event, (yaml.events.MappingEndEvent, yaml.events.SequenceEndEvent)):
                collection_depth -= 1
            elif isinstance(event, yaml.events.ScalarEvent):
                node_count += 1
            if node_count > _MAX_YAML_NODES:
                raise InvalidHookConfigurationError("YAML node count exceeds limit")
    except (yaml.YAMLError, RecursionError, ValueError) as exc:
        raise InvalidHookConfigurationError("malformed YAML frontmatter") from exc


def _load_frontmatter(content: str) -> dict[str, object] | None:
    """Load only a leading YAML frontmatter mapping from bounded cached content."""
    if "\x00" in content:
        raise BinaryHookConfigurationError("binary hook configuration")
    if len(content) > MAX_FILE_CHARS:
        raise HookConfigurationSizeLimitError(len(content))

    lines = content.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        return None
    for index, line in enumerate(lines[1:], start=1):
        if _FRONTMATTER_DELIMITER.fullmatch(line.rstrip("\r\n")):
            frontmatter = "".join(lines[1:index])
            try:
                _validate_yaml_before_construction(frontmatter)
                raw = yaml.load(frontmatter, Loader=_DuplicateKeySafeLoader)
            except (yaml.YAMLError, RecursionError, ValueError) as exc:
                raise InvalidHookConfigurationError("malformed YAML frontmatter") from exc
            if raw is None:
                return {}
            if not isinstance(raw, dict):
                raise InvalidHookConfigurationError("YAML frontmatter must be a mapping")
            return cast(dict[str, object], raw)
    raise InvalidHookConfigurationError("unterminated YAML frontmatter")


def _frontmatter_has_explicit_hooks_key(content: str) -> bool:
    """Recognize a top-level hooks key from bounded YAML parser events."""
    if "\x00" in content or len(content) > MAX_FILE_CHARS:
        return False
    lines = content.splitlines()
    if not lines or lines[0] != "---":
        return False
    frontmatter_lines: list[str] = []
    for line in lines[1:]:
        if _FRONTMATTER_DELIMITER.fullmatch(line):
            break
        frontmatter_lines.append(line)

    collection_depth = 0
    collection_roles: list[bool | None] = []
    root_is_mapping = False
    expecting_key = False
    node_count = 0
    try:
        for event in yaml.parse("\n".join(frontmatter_lines)):
            if isinstance(event, (yaml.events.MappingStartEvent, yaml.events.SequenceStartEvent)):
                role = expecting_key if root_is_mapping and collection_depth == 1 else None
                collection_roles.append(role)
                if collection_depth == 0:
                    root_is_mapping = isinstance(event, yaml.events.MappingStartEvent)
                    expecting_key = root_is_mapping
                collection_depth += 1
                node_count += 1
                if collection_depth > _MAX_YAML_COLLECTION_DEPTH or node_count > _MAX_YAML_NODES:
                    return False
                continue
            if isinstance(event, (yaml.events.MappingEndEvent, yaml.events.SequenceEndEvent)):
                role = collection_roles.pop()
                collection_depth -= 1
                if root_is_mapping and collection_depth == 1 and role is not None:
                    expecting_key = not role
                continue
            if isinstance(event, (yaml.events.ScalarEvent, yaml.events.AliasEvent)):
                node_count += 1
                if node_count > _MAX_YAML_NODES:
                    return False
                if root_is_mapping and collection_depth == 1:
                    if expecting_key:
                        if isinstance(event, yaml.events.ScalarEvent) and event.value == "hooks":
                            return True
                        if isinstance(event, yaml.events.AliasEvent):
                            return True
                    expecting_key = not expecting_key
    except (yaml.YAMLError, RecursionError, ValueError):
        return False
    return False


def _registrations(
    hook_map: object,
    *,
    source_kind: str,
    source_path: str,
    activation_lifetime: str,
    source_line: int = 1,
    source_lines: Iterator[int] | None = None,
    execution_root: str | None = None,
    runtime_confirmed: bool = True,
    registration_limit: int = _MAX_REGISTRATIONS_PER_DOCUMENT,
) -> _RegistrationSet:
    if not isinstance(hook_map, dict):
        raise InvalidHookConfigurationError("hooks must be an event-map object")

    registrations: list[HookRegistration] = []
    flow_inputs: list[HandlerFlowInput] = []
    structure_items = 0
    for event, matcher_groups in hook_map.items():
        structure_items += 1
        if structure_items > _MAX_HOOK_STRUCTURE_ITEMS:
            raise HookRegistrationLimitError("hook structure cardinality limit exceeded")
        if not isinstance(event, str) or not isinstance(matcher_groups, list):
            raise InvalidHookConfigurationError("hook events must map to matcher arrays")
        for matcher_group in matcher_groups:
            structure_items += 1
            if structure_items > _MAX_HOOK_STRUCTURE_ITEMS:
                raise HookRegistrationLimitError("hook structure cardinality limit exceeded")
            if not isinstance(matcher_group, dict) or not isinstance(
                matcher_group.get("hooks"), list
            ):
                raise InvalidHookConfigurationError(
                    "hook matcher groups must contain handler arrays"
                )
            handlers = cast(list[object], matcher_group["hooks"])
            if len(handlers) > registration_limit - len(registrations):
                raise HookRegistrationLimitError("hook registration limit exceeded")
            for handler in handlers:
                structure_items += 1
                if (
                    structure_items > _MAX_HOOK_STRUCTURE_ITEMS
                    or len(registrations) >= registration_limit
                ):
                    raise HookRegistrationLimitError("hook registration limit exceeded")
                if not isinstance(handler, dict):
                    raise InvalidHookConfigurationError("hook handlers must be objects")
                try:
                    normalized_group = cast(dict[str, object], dict(matcher_group))
                    normalized_group["hooks"] = [handler]
                    registration = _normalize_registration(
                        event,
                        normalized_group,
                        cast(dict[str, object], handler),
                        source_kind=source_kind,
                        activation_lifetime=activation_lifetime,
                        source_line=(
                            next(source_lines, source_line) if source_lines else source_line
                        ),
                        source_path=source_path,
                        execution_root=execution_root,
                        runtime_confirmed=runtime_confirmed,
                    )
                except (RecursionError, TypeError, ValueError) as exc:
                    raise InvalidHookConfigurationError("recursive hook handler") from exc
                if registration.handler_status == "invalid" or (
                    registration.event_status == "known" and registration.matcher_kind == "invalid"
                ):
                    raise InvalidHookConfigurationError(
                        "documented hook matcher or handler fields are invalid"
                    )
                registrations.append(registration)
                flow_inputs.append(capture_handler(registration, cast(dict[str, object], handler)))
    return _RegistrationSet(tuple(registrations), tuple(flow_inputs))


def _document(
    *,
    source_kind: str,
    source_path: str,
    activation_lifetime: str,
    hook_map: object,
    content_identity: str,
    execution_root: str | None,
    source_lines: Iterator[int] | None = None,
    runtime_confirmed: bool = True,
    registration_limit: int = _MAX_REGISTRATIONS_PER_DOCUMENT,
) -> HookDocument:
    parsed = _registrations(
        hook_map,
        source_kind=source_kind,
        source_path=source_path,
        activation_lifetime=activation_lifetime,
        source_lines=source_lines,
        execution_root=execution_root,
        runtime_confirmed=runtime_confirmed,
        registration_limit=registration_limit,
    )
    return HookDocument(
        source_kind=source_kind,
        declaration_roles=(source_kind,),
        source_path=source_path,
        activation_lifetime=activation_lifetime,
        content_digest=_digest("content", content_identity),
        registrations=parsed.registrations,
        flow_inputs=parsed.flow_inputs,
    )


def _mapping_value_node(node: yaml.MappingNode, key: str) -> yaml.Node | None:
    for key_node, value_node in node.value:
        if isinstance(key_node, yaml.ScalarNode) and key_node.value == key:
            return value_node
    return None


def _mapping_keys(node: yaml.MappingNode) -> set[str]:
    return {
        key_node.value
        for key_node, _value_node in node.value
        if isinstance(key_node, yaml.ScalarNode)
    }


def _handler_type_line(node: yaml.MappingNode) -> int:
    """Return the handler type-key line, falling back to the mapping start."""
    for key_node, _value_node in node.value:
        if isinstance(key_node, yaml.ScalarNode) and key_node.value == "type":
            return cast(int, key_node.start_mark.line) + 1
    return cast(int, node.start_mark.line) + 1


def _event_map_handler_lines(node: yaml.Node | None) -> tuple[int, ...]:
    """Return handler lines from one structurally validated event map node."""
    if not isinstance(node, yaml.MappingNode):
        return ()
    result: list[int] = []
    for _event_node, matcher_groups in node.value:
        if not isinstance(matcher_groups, yaml.SequenceNode):
            continue
        for matcher_group in matcher_groups.value:
            if not isinstance(matcher_group, yaml.MappingNode):
                continue
            handlers = _mapping_value_node(matcher_group, "hooks")
            if not isinstance(handlers, yaml.SequenceNode):
                continue
            result.extend(
                _handler_type_line(handler)
                for handler in handlers.value
                if isinstance(handler, yaml.MappingNode)
            )
    return tuple(result)


def _inline_declaration_handler_lines(node: yaml.Node | None) -> tuple[int, ...]:
    """Return handler lines from a manifest-style object, path, or mixed array."""
    items = node.value if isinstance(node, yaml.SequenceNode) else [node]
    result: list[int] = []
    for item in items:
        if not isinstance(item, yaml.MappingNode):
            continue
        event_map: yaml.Node | None = item
        if _mapping_keys(item) == {"hooks"}:
            event_map = _mapping_value_node(item, "hooks")
        result.extend(_event_map_handler_lines(event_map))
    return tuple(result)


def _json_root_node(content: str) -> yaml.MappingNode | None:
    """Compose already validated JSON solely to recover structural source locations."""
    try:
        root = yaml.compose(content, Loader=yaml.BaseLoader)
    except (yaml.YAMLError, RecursionError):
        return None
    return root if isinstance(root, yaml.MappingNode) else None


def _node_line(node: yaml.Node | None, fallback: int = 1) -> int:
    """Return one positive parser location without retaining its source value."""
    if node is None:
        return max(1, fallback)
    line = cast(int, node.start_mark.line) + 1
    return line if line > 0 else max(1, fallback)


def _mapping_key_line(node: yaml.MappingNode | None, key: str, fallback: int = 1) -> int:
    if node is not None:
        for key_node, _value_node in node.value:
            if isinstance(key_node, yaml.ScalarNode) and key_node.value == key:
                return _node_line(key_node, fallback)
    return max(1, fallback)


def _known_list_lines(node: yaml.MappingNode | None, key: str) -> tuple[int, ...]:
    value = _mapping_value_node(node, key) if node is not None else None
    if not isinstance(value, yaml.SequenceNode):
        return ()
    return tuple(_node_line(item) for item in value.value)


def _permission_source_lines(
    raw: dict[str, object], root: yaml.MappingNode | None
) -> PermissionSourceLines:
    """Recover only structural permission locations from one composed syntax tree."""
    permissions_line = _mapping_key_line(root, "permissions")
    permissions_node = _mapping_value_node(root, "permissions") if root is not None else None
    permissions_mapping = (
        permissions_node if isinstance(permissions_node, yaml.MappingNode) else None
    )
    raw_permissions = raw.get("permissions")
    raw_permission_mapping = raw_permissions if isinstance(raw_permissions, dict) else {}
    raw_permission_count = len(raw_permission_mapping)
    recovered_key_lines = (
        tuple(_node_line(key_node, permissions_line) for key_node, _ in permissions_mapping.value)
        if permissions_mapping is not None
        else ()
    )
    permission_key_lines = recovered_key_lines[:raw_permission_count] + (permissions_line,) * max(
        0, raw_permission_count - len(recovered_key_lines)
    )

    return PermissionSourceLines(
        permissions_line=permissions_line,
        permission_key_lines=permission_key_lines,
        allow_lines=_known_list_lines(permissions_mapping, "allow"),
        ask_lines=_known_list_lines(permissions_mapping, "ask"),
        deny_lines=_known_list_lines(permissions_mapping, "deny"),
        additional_directory_lines=_known_list_lines(permissions_mapping, "additionalDirectories"),
        default_mode_line=(
            _mapping_key_line(permissions_mapping, "defaultMode", permissions_line)
            if "defaultMode" in raw_permission_mapping
            else None
        ),
        disable_bypass_line=(
            _mapping_key_line(permissions_mapping, "disableBypassPermissionsMode", permissions_line)
            if "disableBypassPermissionsMode" in raw_permission_mapping
            else None
        ),
        disable_auto_line=(
            _mapping_key_line(permissions_mapping, "disableAutoMode", permissions_line)
            if "disableAutoMode" in raw_permission_mapping
            else None
        ),
        skip_dangerous_prompt_line=(
            _mapping_key_line(
                permissions_mapping, "skipDangerousModePermissionPrompt", permissions_line
            )
            if "skipDangerousModePermissionPrompt" in raw_permission_mapping
            else None
        ),
    )


def _json_handler_lines_from_root(root: yaml.MappingNode | None) -> tuple[int, ...]:
    """Locate handler declarations from an already composed JSON syntax tree."""
    return _event_map_handler_lines(
        _mapping_value_node(root, "hooks") if root is not None else None
    )


def _json_handler_lines(content: str) -> tuple[int, ...]:
    """Locate handler declarations under a JSON document's top-level hook map."""
    return _json_handler_lines_from_root(_json_root_node(content))


def _manifest_handler_lines(content: str) -> tuple[int, ...]:
    """Locate only inline handler declarations in a plugin manifest."""
    root = _json_root_node(content)
    return _inline_declaration_handler_lines(
        _mapping_value_node(root, "hooks") if root is not None else None
    )


def _marketplace_handler_lines(content: str) -> tuple[tuple[int, ...], ...]:
    """Locate inline handlers per marketplace entry without matching metadata fields."""
    root = _json_root_node(content)
    plugins = _mapping_value_node(root, "plugins") if root is not None else None
    if not isinstance(plugins, yaml.SequenceNode):
        return ()
    return tuple(
        _inline_declaration_handler_lines(_mapping_value_node(entry, "hooks"))
        if isinstance(entry, yaml.MappingNode)
        else ()
        for entry in plugins.value
    )


def _yaml_handler_lines(content: str) -> tuple[int, ...]:
    """Return source lines for handler mappings in leading YAML frontmatter."""
    lines = content.splitlines(keepends=True)
    delimiter = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if _FRONTMATTER_DELIMITER.fullmatch(line.rstrip("\r\n"))
        ),
        None,
    )
    if delimiter is None:
        return ()
    try:
        root = yaml.compose("".join(lines[1:delimiter]), Loader=yaml.BaseLoader)
    except yaml.YAMLError:
        return ()
    if not isinstance(root, yaml.MappingNode):
        return ()

    def mapping_value(node: yaml.MappingNode, key: str) -> yaml.Node | None:
        for key_node, value_node in node.value:
            if isinstance(key_node, yaml.ScalarNode) and key_node.value == key:
                return value_node
        return None

    hook_map = mapping_value(root, "hooks")
    if not isinstance(hook_map, yaml.MappingNode):
        return ()
    result: list[int] = []
    for _event_node, matcher_groups in hook_map.value:
        if not isinstance(matcher_groups, yaml.SequenceNode):
            continue
        for matcher_group in matcher_groups.value:
            if not isinstance(matcher_group, yaml.MappingNode):
                continue
            handlers = mapping_value(matcher_group, "hooks")
            if not isinstance(handlers, yaml.SequenceNode):
                continue
            result.extend(
                handler.start_mark.line + 2
                for handler in handlers.value
                if isinstance(handler, yaml.MappingNode)
            )
    return tuple(result)


def _archive_or_project_root(path: str) -> str:
    namespace, _parts = _path_parts(path)
    return f"{namespace}!/" if namespace else ""


def _parse_hook_document(
    path: str,
    content: str,
    source_kind: str,
    activation_lifetime: str,
    *,
    execution_root: str | None,
    registration_limit: int = _MAX_REGISTRATIONS_PER_DOCUMENT,
) -> HookDocument:
    raw = _load_json(content)
    return _parse_hook_mapping_document(
        path,
        raw,
        source_kind,
        activation_lifetime,
        content_digest=_digest("content", content),
        execution_root=execution_root,
        source_lines=_json_handler_lines(content),
        registration_limit=registration_limit,
    )


def _parse_hook_mapping_document(
    path: str,
    raw: dict[str, object],
    source_kind: str,
    activation_lifetime: str,
    *,
    content_digest: str,
    execution_root: str | None,
    source_lines: tuple[int, ...],
    registration_limit: int = _MAX_REGISTRATIONS_PER_DOCUMENT,
) -> HookDocument:
    """Build one hook document from a retained duplicate-safe JSON mapping."""
    if "hooks" not in raw:
        raise InvalidHookConfigurationError("hook document must contain hooks")
    parsed = _registrations(
        raw["hooks"],
        source_kind=source_kind,
        source_path=path,
        activation_lifetime=activation_lifetime,
        execution_root=execution_root,
        source_lines=iter(source_lines),
        registration_limit=registration_limit,
    )
    return HookDocument(
        source_kind=source_kind,
        declaration_roles=(source_kind,),
        source_path=path,
        activation_lifetime=activation_lifetime,
        content_digest=content_digest,
        registrations=parsed.registrations,
        flow_inputs=parsed.flow_inputs,
    )


def _parse_frontmatter_document(
    path: str,
    content: str,
    source_kind: str,
    activation_lifetime: str,
    execution_root: str | None,
    runtime_status: str = "declared_unclassified",
    registration_limit: int = _MAX_REGISTRATIONS_PER_DOCUMENT,
) -> HookDocument | None:
    """Return a frontmatter hook document, or None when no hooks are declared."""
    raw = _load_frontmatter(content)
    if raw is None or "hooks" not in raw:
        return None
    document = _document(
        source_kind=source_kind,
        source_path=path,
        activation_lifetime=activation_lifetime,
        hook_map=raw["hooks"],
        content_identity=content,
        execution_root=execution_root,
        source_lines=iter(_yaml_handler_lines(content)),
        runtime_confirmed=runtime_status != "runtime_unconfirmed",
        registration_limit=registration_limit,
    )
    return replace(document, runtime_status=runtime_status)


def _is_plugin_metadata_path(path: str, filename: str) -> bool:
    """Return whether a cache key ends in an exact plugin metadata path."""
    _namespace_value, parts = _path_parts(path)
    return parts[-2:] == (_PLUGIN_METADATA_DIRECTORY, filename)


def _plugin_metadata_root(path: str, filename: str) -> str:
    """Return the root owning an exact metadata file without slicing raw strings."""
    namespace, parts = _path_parts(path)
    if parts[-2:] != (_PLUGIN_METADATA_DIRECTORY, filename):
        raise ValueError("not a plugin metadata path")
    root = "/".join(parts[:-2])
    if not namespace:
        return root
    return f"{namespace}!/{root}" if root else f"{namespace}!/"


def _plugin_metadata_path(plugin_root: str, filename: str) -> str:
    """Build one normalized metadata path inside a project or archive root."""
    namespace, root_parts = _path_parts(plugin_root)
    joined = "/".join((*root_parts, _PLUGIN_METADATA_DIRECTORY, filename))
    return f"{namespace}!/{joined}" if namespace else joined


def _is_manifest_path(path: str) -> bool:
    return _is_plugin_metadata_path(path, _PLUGIN_MANIFEST_FILENAME)


def _is_marketplace_path(path: str) -> bool:
    return _is_plugin_metadata_path(path, _PLUGIN_MARKETPLACE_FILENAME)


def _manifest_root(path: str) -> str:
    return _plugin_metadata_root(path, _PLUGIN_MANIFEST_FILENAME)


def _marketplace_root(path: str) -> str:
    return _plugin_metadata_root(path, _PLUGIN_MARKETPLACE_FILENAME)


def _resolve_local_path(
    root: str, reference: str, *, allow_dot: bool = True, allow_bare: bool = False
) -> str:
    """Resolve a marketplace-local path without leaving its cache namespace."""
    if not isinstance(reference, str) or "\x00" in reference or "\\" in reference:
        raise InvalidHookConfigurationError("marketplace path is not safe")
    if reference == "." and allow_dot:
        relative_parts: tuple[str, ...] = ()
    else:
        if not reference.startswith("./") and not allow_bare:
            raise InvalidHookConfigurationError("marketplace path must be relative")
        if "!/" in reference:
            raise InvalidHookConfigurationError("marketplace path changes archive namespace")
        parsed = PurePosixPath(reference)
        if parsed.is_absolute() or any(
            part == ".." or (len(part) >= 2 and part[1] == ":") for part in parsed.parts
        ):
            raise InvalidHookConfigurationError("marketplace path escapes its root")
        relative_parts = tuple(part for part in parsed.parts if part != ".")
        if not relative_parts and not allow_dot:
            raise InvalidHookConfigurationError("marketplace path must name a component")
    namespace, root_parts = _path_parts(root)
    joined = "/".join((*root_parts, *relative_parts))
    return f"{namespace}!/{joined}" if namespace else joined


def _manifest_path(plugin_root: str) -> str:
    return _plugin_metadata_path(plugin_root, _PLUGIN_MANIFEST_FILENAME)


def _marketplace_entry_path(path: str, index: int, reserved_paths: set[str] | None = None) -> str:
    """Return a safe synthetic ledger path for one marketplace entry."""
    base = f"{path}#plugin[{index}]"
    candidate = base
    suffix = 0
    while reserved_paths is not None and candidate in reserved_paths:
        suffix += 1
        candidate = f"{base}#ledger[{suffix}]"
    return candidate


def _validate_marketplace_component(
    value: object, plugin_root: str | None, *, component_kind: str
) -> object:
    if not isinstance(value, (str, list)):
        raise InvalidHookConfigurationError(
            "marketplace component must be a relative path or array"
        )
    values = [value] if isinstance(value, str) else value
    if not all(isinstance(item, str) for item in values):
        raise InvalidHookConfigurationError("marketplace component entries must be paths")
    if plugin_root is not None:
        for item in values:
            if item == "." and component_kind != "skills":
                raise InvalidHookConfigurationError(
                    "only marketplace skills may use the bare-dot plugin root"
                )
            _resolve_local_path(plugin_root, cast(str, item), allow_dot=True)
    return value


def _validate_marketplace_hooks(value: object) -> object:
    if not isinstance(value, (str, dict, list)):
        raise InvalidHookConfigurationError("marketplace hooks must be an object, path, or array")
    if isinstance(value, list) and not all(isinstance(item, (str, dict)) for item in value):
        raise InvalidHookConfigurationError("marketplace hook items must be paths or objects")
    return value


def _required_nonempty_string(mapping: dict[str, object], field: str, owner: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise InvalidHookConfigurationError(f"{owner} {field} is required")
    return value


def _validate_manifest_identity(manifest: dict[str, object]) -> None:
    _required_nonempty_string(manifest, "name", "plugin manifest")


def _validate_marketplace_identity(marketplace: dict[str, object]) -> list[object]:
    _required_nonempty_string(marketplace, "name", "marketplace")
    owner = marketplace.get("owner")
    if not isinstance(owner, dict):
        raise InvalidHookConfigurationError("marketplace owner must be an object")
    _required_nonempty_string(cast(dict[str, object], owner), "name", "marketplace owner")
    plugins = marketplace.get("plugins")
    if not isinstance(plugins, list):
        raise InvalidHookConfigurationError("marketplace plugins must be an array")
    return plugins


def _validate_remote_plugin_source(source: dict[str, object]) -> None:
    source_type = source.get("source")
    required_fields = {
        "github": ("repo",),
        "url": ("url",),
        "git-subdir": ("url", "path"),
        "npm": ("package",),
        "archive": ("url",),
        "command": ("command",),
    }
    if not isinstance(source_type, str) or source_type not in required_fields:
        raise InvalidHookConfigurationError("remote marketplace source is malformed")
    for required_field in required_fields[source_type]:
        _required_nonempty_string(source, required_field, f"remote {source_type} source")
    for optional_field in ("ref", "sha", "sha256", "version", "registry", "mode"):
        if optional_field in source and not isinstance(source[optional_field], str):
            raise InvalidHookConfigurationError(
                f"remote marketplace source {optional_field} must be a string"
            )


def _default_path(plugin_root: str) -> str:
    if not plugin_root:
        return _PLUGIN_DEFAULT_PATH
    separator = "" if plugin_root.endswith("/") else "/"
    return f"{plugin_root}{separator}{_PLUGIN_DEFAULT_PATH}"


def _namespace(path: str) -> str:
    return path.rsplit("!/", 1)[0] if "!/" in path else ""


def _resolve_reference(plugin_root: str, reference: str) -> str:
    """Resolve a documented relative manifest ref without crossing path namespaces."""
    if not reference.startswith("./"):
        raise InvalidHookConfigurationError("hook reference must be relative")
    if "\x00" in reference or "\\" in reference:
        raise InvalidHookConfigurationError("hook reference contains NUL")
    root_namespace = _namespace(plugin_root)
    if "!/" in reference or _namespace(reference) not in {"", root_namespace}:
        raise InvalidHookConfigurationError("hook reference changes archive namespace")

    root_prefix = plugin_root.rsplit("!/", 1)[-1].strip("/")
    reference_path = PurePosixPath(reference)
    if reference_path.is_absolute() or any(
        part == ".." or (len(part) >= 2 and part[1] == ":") for part in reference_path.parts
    ):
        raise InvalidHookConfigurationError("hook reference escapes plugin root")
    inner = "/".join(part for part in reference_path.parts if part != ".")
    if not inner:
        raise InvalidHookConfigurationError("hook reference must name a configuration document")
    joined = "/".join(part for part in (root_prefix, inner) if part)
    return f"{root_namespace}!/{joined}" if root_namespace else joined


def _path_parts(path: str) -> tuple[str, tuple[str, ...]]:
    """Split a normal or archive-backed cache key into namespace and POSIX parts."""
    namespace = _namespace(path)
    member = path.rsplit("!/", 1)[-1] if namespace else path
    return namespace, tuple(part for part in member.split("/") if part)


def _permission_source_identity_digest(path: str) -> str:
    """Hash the full normalized cache identity, including every archive namespace."""
    payload = b"skillspector.bundled_permission.source.v1\0" + path.encode("utf-8")
    return f"sha256:{sha256(payload).hexdigest()}"


def _is_within_root(path: str, root: str) -> bool:
    path_namespace, path_parts = _path_parts(path)
    root_namespace, root_parts = _path_parts(root)
    return path_namespace == root_namespace and path_parts[: len(root_parts)] == root_parts


def _relative_parts(path: str, root: str) -> tuple[str, ...] | None:
    if not _is_within_root(path, root):
        return None
    return _path_parts(path)[1][len(_path_parts(root)[1]) :]


def _resolve_component_reference(plugin_root: str, reference: str) -> str:
    """Resolve a manifest component path without filesystem access or namespace escape."""
    if (
        not reference.startswith("./")
        or "\x00" in reference
        or "\\" in reference
        or "!/" in reference
    ):
        raise InvalidHookConfigurationError("component reference is not a safe relative path")
    parsed = PurePosixPath(reference)
    if parsed.is_absolute() or any(
        part == ".." or (len(part) >= 2 and part[1] == ":") for part in parsed.parts
    ):
        raise InvalidHookConfigurationError("component reference escapes plugin root")
    relative = tuple(part for part in parsed.parts if part != ".")
    namespace, root_parts = _path_parts(plugin_root)
    joined = "/".join((*root_parts, *relative))
    return f"{namespace}!/{joined}" if namespace else joined


def _manifest_component_paths(
    plugin_root: str,
    references: object,
    *,
    component_kind: str,
    candidates: list[str],
) -> tuple[set[str], tuple[str, ...]]:
    """Expand explicit file/directory manifest components from known cache keys."""
    if not isinstance(references, (str, list)):
        raise InvalidHookConfigurationError(
            f"manifest {component_kind} must be a relative path or array"
        )
    raw_references = [references] if isinstance(references, str) else references
    resolved: set[str] = set()
    missing: list[str] = []
    for reference in raw_references:
        if not isinstance(reference, str):
            raise InvalidHookConfigurationError(
                f"manifest {component_kind} entries must be relative paths"
            )
        if reference == "." and component_kind != "skills":
            raise InvalidHookConfigurationError(
                "only manifest skills may use the bare-dot plugin root"
            )
        target = _resolve_local_path(plugin_root, reference, allow_dot=True)
        target_parts = _path_parts(target)[1]
        is_file = bool(target_parts) and target_parts[-1].lower().endswith(".md")
        if is_file:
            if target not in candidates:
                missing.append(target)
                continue
            resolved.add(target)
            continue
        if not any(_is_within_root(path, target) for path in candidates):
            missing.append(target)
            continue
        if component_kind == "skills":
            resolved.update(
                path
                for path in candidates
                if _is_within_root(path, target) and _path_parts(path)[1][-1] == "SKILL.md"
            )
        else:
            resolved.update(
                path
                for path in candidates
                if _is_within_root(path, target) and path.lower().endswith(".md")
            )
    return resolved, tuple(dict.fromkeys(missing))


def _default_plugin_skill_paths(plugin_root: str, candidates: list[str]) -> set[str]:
    return {
        path
        for path in candidates
        if (relative := _relative_parts(path, plugin_root)) is not None
        and len(relative) == 3
        and relative[0] == "skills"
        and relative[-1] == "SKILL.md"
    }


def _default_plugin_command_paths(plugin_root: str, candidates: list[str]) -> set[str]:
    return {
        path
        for path in candidates
        if (relative := _relative_parts(path, plugin_root)) is not None
        and len(relative) >= 2
        and relative[0] == "commands"
        and relative[-1].lower().endswith(".md")
    }


def _has_default_plugin_skills_directory(plugin_root: str, candidates: list[str]) -> bool:
    return any(
        (relative := _relative_parts(path, plugin_root)) is not None
        and len(relative) >= 2
        and relative[0] == "skills"
        for path in candidates
    )


def _manifest_inline_map(raw_item: dict[str, object]) -> object:
    """Accept direct event maps and an unambiguous one-key compatibility wrapper."""
    if set(raw_item) == {"hooks"}:
        return raw_item["hooks"]
    return raw_item


def _bh1_finding(document: HookDocument, known_paths: set[str]) -> Finding:
    chain_digest = _digest(
        "BH1",
        "\0".join(
            (
                document.source_kind,
                *document.declaration_roles,
                document.source_path,
                document.activation_lifetime,
                _SEMANTICS_SNAPSHOT,
                document.content_digest,
                *(registration.chain_digest for registration in document.registrations),
            )
        ),
    )
    severity_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    severity = max(
        (
            registration_severity(registration, known_paths)
            for registration in document.registrations
        ),
        key=severity_rank.__getitem__,
        default="LOW",
    )
    handler_types = ",".join(
        sorted({registration.handler_type for registration in document.registrations})
    )
    events = ",".join(
        sorted(
            {
                registration.event if registration.event_status == "known" else "unknown"
                for registration in document.registrations
            }
        )
    )
    runnable_count = sum(registration.runnable for registration in document.registrations)
    ambient_count = sum(registration.ambient for registration in document.registrations)
    if document.runtime_status == "runtime_unconfirmed":
        runtime_status = document.runtime_status
    elif runnable_count:
        runtime_status = "runnable"
    elif document.registrations and all(
        registration.runtime_status == "dormant" for registration in document.registrations
    ):
        runtime_status = "all_dormant"
    else:
        runtime_status = "unconfirmed"
    evidence: dict[str, object] = {
        "schema": _EVIDENCE_SCHEMA,
        "claude_semantics_snapshot": _SEMANTICS_SNAPSHOT,
        "source_kind": document.source_kind,
        "declaration_roles": ",".join(document.declaration_roles),
        "activation_lifetime": document.activation_lifetime,
        "runtime_status": runtime_status,
        "handler_count": len(document.registrations),
        "runnable_handler_count": runnable_count,
        "ambient_handler_count": ambient_count,
        "handler_types": handler_types,
        "events": events,
        "chain_digest": chain_digest,
    }
    return Finding(
        rule_id="BH1",
        message=(
            "Bundled hook document declares "
            f"{len(document.registrations)} handler(s) for automatic execution."
        ),
        severity=severity,
        confidence=1.0,
        file=document.source_path,
        start_line=min(
            (registration.source_line for registration in document.registrations), default=1
        ),
        category="Bundled Execution Surface",
        pattern="Bundled Hook Declaration",
        explanation="The artifact declares hooks that may execute when their activation fires.",
        remediation="Review each bundled hook before trusting or enabling the artifact.",
        tags=["bundled-execution-surface", "structural"],
        matched_text=chain_digest,
        finding=chain_digest,
        evidence=evidence,
    )


def _failure_reason(error: BaseException) -> LedgerReason:
    return (
        LedgerReason.MISSING_FILE_CACHE
        if isinstance(error, KeyError)
        else LedgerReason.BINARY_CONTENT
        if isinstance(error, BinaryHookConfigurationError)
        else LedgerReason.SIZE_LIMIT
        if isinstance(error, HookConfigurationSizeLimitError)
        else LedgerReason.COMPONENT_LIMIT
        if isinstance(error, HookRegistrationLimitError)
        else LedgerReason.INVALID_CONFIGURATION
    )


def _failure(
    path: str, error: BaseException, *, phase: str = "bundled_hook"
) -> InspectionLedgerEvent:
    reason = _failure_reason(error)
    if isinstance(error, HookConfigurationSizeLimitError):
        return ledger_event(
            outcome=LedgerOutcome.FAILED,
            phase=phase,
            analyzer_id=ANALYZER_ID,
            path=path,
            reason=reason,
            error_class=type(error).__name__,
            stage="parse",
            observed_characters=error.observed_characters,
            limit_characters=MAX_FILE_CHARS,
        )
    return ledger_event(
        outcome=LedgerOutcome.FAILED,
        phase=phase,
        analyzer_id=ANALYZER_ID,
        path=path,
        reason=reason,
        error_class=type(error).__name__,
        stage="parse",
    )


def _completed(
    path: str, findings: list[Finding], *, phase: str = "bundled_hook"
) -> InspectionLedgerEvent:
    return ledger_event(
        outcome=LedgerOutcome.COMPLETED,
        phase=phase,
        analyzer_id=ANALYZER_ID,
        path=path,
        emitted_finding_ids=[finding.finding_id for finding in findings],
    )


def _settings_terminal(
    work: _SettingsWork,
    hook_result: tuple[LedgerOutcome, LedgerReason | None] | None,
    findings: list[Finding],
) -> InspectionLedgerEvent | None:
    """Reduce hook and permission subanalyses to one settings producer row."""
    if work.parse_error is not None:
        return _failure(work.source_path, work.parse_error, phase="bundled_settings")

    permission = work.permission_analysis
    permission_result = (
        (permission.outcome, permission.reason)
        if permission is not None and permission.applicable
        else None
    )
    applicable_results = [
        result for result in (hook_result, permission_result) if result is not None
    ]
    if not applicable_results:
        return None

    failed_results = [result for result in applicable_results if result[0] is LedgerOutcome.FAILED]
    retained_valid_result = any(
        result[0] in {LedgerOutcome.COMPLETED, LedgerOutcome.PARTIAL}
        for result in applicable_results
    )
    incomplete_result = any(result[0] is LedgerOutcome.PARTIAL for result in applicable_results)

    reason = (
        LedgerReason.COMPONENT_LIMIT
        if any(result[1] is LedgerReason.COMPONENT_LIMIT for result in applicable_results)
        else LedgerReason.INVALID_CONFIGURATION
    )
    if failed_results:
        outcome = LedgerOutcome.PARTIAL if retained_valid_result else LedgerOutcome.FAILED
    elif incomplete_result:
        outcome = LedgerOutcome.PARTIAL
    else:
        outcome = LedgerOutcome.COMPLETED

    if outcome is LedgerOutcome.COMPLETED:
        return _completed(work.source_path, findings, phase="bundled_settings")
    return ledger_event(
        outcome=outcome,
        phase="bundled_settings",
        analyzer_id=ANALYZER_ID,
        path=work.source_path,
        reason=reason,
        emitted_finding_ids=[finding.finding_id for finding in findings],
        stage="analyze",
    )


def _flow_terminal(work: FlowWorkResult, findings: list[Finding]) -> InspectionLedgerEvent:
    """Convert one sanitized flow result into its unique producer ledger row."""
    common: dict[str, object] = {
        "phase": "bundled_hook",
        "analyzer_id": ANALYZER_ID,
        "path": work.ref.path,
        "start_line": work.ref.start_line,
        "end_line": work.ref.end_line,
    }
    if work.outcome is LedgerOutcome.COMPLETED:
        return ledger_event(
            outcome=LedgerOutcome.COMPLETED,
            emitted_finding_ids=[finding.finding_id for finding in findings],
            **common,  # type: ignore[arg-type]
        )
    return ledger_event(
        outcome=work.outcome,
        reason=work.reason or LedgerReason.ANALYZER_RUNTIME_ERROR,
        error_class=work.error_class,
        observed_characters=work.observed_characters,
        limit_characters=work.limit_characters,
        **common,  # type: ignore[arg-type]
    )


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Discover supported hook documents from deterministic cache state only."""
    component_paths = cast(list[str], state.get("components") or [])
    cache = cast(dict[str, str], state.get("local_file_cache") or state.get("file_cache") or {})
    paths = list(dict.fromkeys(component_paths))
    known_paths = list(dict.fromkeys([*paths, *cache]))
    known_path_set = set(known_paths)
    cache_path_set = set(cache)
    manifest_limited_paths = {
        str(artifact.get("path", ""))
        for artifact in state.get("artifact_inventory", []) or []
        if str(artifact.get("reason", "")) in {"manifest_parse_error", "manifest_parse_limit"}
    }
    path_rank = {path: index for index, path in enumerate(paths)}
    root_candidate_index: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for path in known_paths:
        namespace, path_parts = _path_parts(path)
        for prefix_length in range(len(path_parts) + 1):
            root_candidate_index.setdefault((namespace, path_parts[:prefix_length]), []).append(
                path
            )
    user_config_by_root: dict[str, UserConfigProfile] = {}

    def candidates_for_root(root: str) -> list[str]:
        return root_candidate_index.get(_path_parts(root), [])

    def project_settings_metadata(path: str) -> tuple[str, str] | None:
        _namespace_value, member_parts = _path_parts(path)
        if len(member_parts) != 2:
            return None
        return _PROJECT_SETTINGS.get("/".join(member_parts))

    documents: list[HookDocument] = []
    document_indexes: dict[str, int] = {}
    events: list[InspectionLedgerEvent] = []
    marketplace_entry_events: dict[str, InspectionLedgerEvent] = {}
    handled_paths: set[str] = set()

    def record_marketplace_entry_failure(path: str, error: BaseException) -> None:
        """Record at most one terminal failure for one logical marketplace entry."""
        if path in marketplace_entry_events:
            return
        event = _failure(path, error)
        marketplace_entry_events[path] = event
        events.append(event)

    def add_document(document: HookDocument) -> None:
        if len(document.registrations) > _MAX_REGISTRATIONS_PER_DOCUMENT:
            raise HookRegistrationLimitError("aggregated hook registration limit exceeded")
        if document.source_path in document_indexes:
            index = document_indexes[document.source_path]
            existing = documents[index]
            if (
                existing.source_kind == "marketplace_plugin_inline"
                and document.source_kind == "marketplace_plugin_inline"
            ):
                if (
                    len(existing.registrations) + len(document.registrations)
                    > _MAX_REGISTRATIONS_PER_DOCUMENT
                ):
                    raise HookRegistrationLimitError(
                        "aggregated marketplace registration limit exceeded"
                    )
                documents[index] = replace(
                    existing,
                    registrations=(*existing.registrations, *document.registrations),
                    flow_inputs=(*existing.flow_inputs, *document.flow_inputs),
                )
            add_declaration_role(document.source_path, document.source_kind)
            return
        document_indexes[document.source_path] = len(documents)
        documents.append(document)

    def discard_document(path: str) -> None:
        """Remove a partially aggregated physical document before recording failure."""
        index = document_indexes.pop(path, None)
        if index is None:
            return
        documents.pop(index)
        for shifted_index in range(index, len(documents)):
            document_indexes[documents[shifted_index].source_path] = shifted_index

    def add_declaration_role(path: str, role: str) -> None:
        index = document_indexes.get(path)
        if index is None:
            return
        document = documents[index]
        documents[index] = replace(
            document,
            activation_lifetime=(
                "plugin_enabled"
                if role in {"plugin_manifest_reference", "marketplace_plugin_reference"}
                else document.activation_lifetime
            ),
            declaration_roles=tuple(sorted({*document.declaration_roles, role})),
        )

    settings_work_by_path: dict[str, _SettingsWork] = {}
    settings_handler_lines_by_path: dict[str, tuple[int, ...]] = {}
    settings_hook_results: dict[str, tuple[LedgerOutcome, LedgerReason | None]] = {}
    for path in known_paths:
        settings = project_settings_metadata(path)
        if settings is None:
            continue

        content = cache.get(path)
        source_identity_digest = _permission_source_identity_digest(path)
        if content is None:
            settings_work_by_path[path] = _SettingsWork(
                source_path=path,
                source_kind=settings[0],
                content_digest=_digest("content", ""),
                source_identity_digest=source_identity_digest,
                raw=None,
                parse_error=KeyError(path),
                permission_analysis=None,
                permission_source_lines=PermissionSourceLines(),
            )
            continue

        try:
            raw = _load_json(content)
        except (InvalidHookConfigurationError, TypeError) as exc:
            settings_work_by_path[path] = _SettingsWork(
                source_path=path,
                source_kind=settings[0],
                content_digest=_digest("content", ""),
                source_identity_digest=source_identity_digest,
                raw=None,
                parse_error=exc,
                permission_analysis=None,
                permission_source_lines=PermissionSourceLines(),
            )
            continue

        content_digest = _digest("content", content)
        syntax_root = _json_root_node(content)
        permission_source_lines = _permission_source_lines(raw, syntax_root)
        permission_analysis = analyze_permission_grants(
            raw,
            source_kind=settings[0],
            content_digest=content_digest,
            source_identity_digest=source_identity_digest,
            source_lines=permission_source_lines,
        )
        settings_work_by_path[path] = _SettingsWork(
            source_path=path,
            source_kind=settings[0],
            content_digest=content_digest,
            source_identity_digest=source_identity_digest,
            raw=raw,
            parse_error=None,
            permission_analysis=permission_analysis,
            permission_source_lines=permission_source_lines,
        )
        settings_handler_lines_by_path[path] = _json_handler_lines_from_root(syntax_root)

    marketplace_entries: list[MarketplaceEntry] = []
    marketplace_declared_roots: set[str] = set()
    marketplace_managed_manifests: set[str] = set()
    invalid_manifest_paths: set[str] = set()
    marketplace_paths = [path for path in paths if _is_marketplace_path(path)]
    marketplace_path_set = set(marketplace_paths)
    for marketplace_path in marketplace_paths:
        content = cache.get(marketplace_path)
        if content is None:
            handled_paths.add(marketplace_path)
            events.append(_failure(marketplace_path, KeyError(marketplace_path)))
            continue
        try:
            marketplace = _load_json(content)
            raw_plugins = _validate_marketplace_identity(marketplace)
            metadata = marketplace.get("metadata", {})
            if not isinstance(metadata, dict):
                raise InvalidHookConfigurationError("marketplace metadata must be an object")
            plugin_root_ref = metadata.get("pluginRoot", ".")
            if not isinstance(plugin_root_ref, str):
                raise InvalidHookConfigurationError("marketplace pluginRoot must be a path")
            catalog_root = _marketplace_root(marketplace_path)
            catalog_plugin_root = _resolve_local_path(catalog_root, plugin_root_ref)
            explicit_plugin_root = "pluginRoot" in metadata
        except (InvalidHookConfigurationError, TypeError) as exc:
            handled_paths.add(marketplace_path)
            events.append(_failure(marketplace_path, exc))
            continue

        handler_lines_by_entry = _marketplace_handler_lines(content)
        for index, raw_entry in enumerate(raw_plugins):
            entry_path = _marketplace_entry_path(marketplace_path, index, known_path_set)
            entry_handler_lines = (
                handler_lines_by_entry[index] if index < len(handler_lines_by_entry) else ()
            )
            try:
                if not isinstance(raw_entry, dict):
                    raise InvalidHookConfigurationError(
                        "marketplace plugin entry must be an object"
                    )
                _required_nonempty_string(
                    cast(dict[str, object], raw_entry), "name", "marketplace plugin entry"
                )
                source = raw_entry.get("source")
                if source is None:
                    raise InvalidHookConfigurationError("marketplace plugin source is required")
                strict = raw_entry.get("strict", True)
                if not isinstance(strict, bool):
                    raise InvalidHookConfigurationError("marketplace strict must be boolean")
                plugin_root: str | None
                source_is_root = isinstance(source, str) and source in {".", "./"}
                if isinstance(source, dict):
                    _validate_remote_plugin_source(cast(dict[str, object], source))
                    plugin_root = None
                elif isinstance(source, str):
                    plugin_root = _resolve_local_path(
                        catalog_plugin_root,
                        source,
                        allow_bare=explicit_plugin_root,
                    )
                else:
                    raise InvalidHookConfigurationError(
                        "marketplace source must be local or remote"
                    )
                hooks = (
                    _validate_marketplace_hooks(raw_entry["hooks"])
                    if "hooks" in raw_entry
                    else None
                )
                skills = (
                    _validate_marketplace_component(
                        raw_entry["skills"], plugin_root, component_kind="skills"
                    )
                    if "skills" in raw_entry
                    else None
                )
                commands = (
                    _validate_marketplace_component(
                        raw_entry["commands"], plugin_root, component_kind="commands"
                    )
                    if "commands" in raw_entry
                    else None
                )
                entry = MarketplaceEntry(
                    marketplace_path=marketplace_path,
                    ledger_path=entry_path,
                    index=index,
                    plugin_root=plugin_root,
                    strict=strict,
                    hooks=hooks,
                    skills=skills,
                    commands=commands,
                    handler_lines=entry_handler_lines,
                    source_is_root=source_is_root,
                )
                if plugin_root is not None:
                    marketplace_declared_roots.add(plugin_root)
                    manifest_path = _manifest_path(plugin_root)
                    if not candidates_for_root(plugin_root):
                        raise KeyError(plugin_root)
                    if not strict:
                        marketplace_managed_manifests.add(manifest_path)
                        manifest_content = cache.get(manifest_path)
                        if manifest_content is not None:
                            manifest = _load_json(manifest_content)
                            _validate_manifest_identity(manifest)
                            if any(key in manifest for key in _MANIFEST_COMPONENT_FIELDS):
                                raise InvalidHookConfigurationError(
                                    "strict-false marketplace definition conflicts with plugin manifest"
                                )
                            user_config_by_root[plugin_root] = build_user_config_profile(
                                manifest.get("userConfig")
                            )
                marketplace_entries.append(entry)
            except (InvalidHookConfigurationError, TypeError, KeyError) as exc:
                record_marketplace_entry_failure(entry_path, exc)

    manifests = [
        path
        for path in known_paths
        if _is_manifest_path(path) and path not in marketplace_managed_manifests
    ]
    manifest_path_set = set(manifests)
    marketplace_owned_paths = {
        candidate for root in marketplace_declared_roots for candidate in candidates_for_root(root)
    }
    default_paths = {
        path
        for path in paths
        if _path_parts(path)[1] == tuple(_PLUGIN_DEFAULT_PATH.split("/"))
        and path not in marketplace_owned_paths
    }

    for path in paths:
        if path not in default_paths:
            continue
        handled_paths.add(path)
        content = cache.get(path)
        if content is None:
            events.append(_failure(path, KeyError(path)))
            continue
        try:
            namespace, parts = _path_parts(path)
            root_parts = parts[: -len(tuple(_PLUGIN_DEFAULT_PATH.split("/")))]
            execution_root = "/".join(root_parts)
            if namespace:
                execution_root = f"{namespace}!/{execution_root}"
            document = _parse_hook_document(
                path,
                content,
                "plugin_default",
                "plugin_enabled",
                execution_root=execution_root,
            )
        except (InvalidHookConfigurationError, TypeError) as exc:
            events.append(_failure(path, exc))
            continue
        add_document(document)

    for path, settings_work in settings_work_by_path.items():
        if (
            settings_work.parse_error is not None
            or settings_work.raw is None
            or "hooks" not in settings_work.raw
        ):
            continue
        settings = project_settings_metadata(path)
        assert settings is not None
        try:
            document = _parse_hook_mapping_document(
                path,
                settings_work.raw,
                settings_work.source_kind,
                settings[1],
                content_digest=settings_work.content_digest,
                execution_root=_archive_or_project_root(path),
                source_lines=settings_handler_lines_by_path.get(path, ()),
            )
        except (InvalidHookConfigurationError, TypeError) as exc:
            handled_paths.add(path)
            settings_hook_results[path] = (
                LedgerOutcome.FAILED,
                _failure_reason(exc),
            )
            continue
        handled_paths.add(path)
        settings_hook_results[path] = (LedgerOutcome.COMPLETED, None)
        add_document(document)

    referenced_paths: dict[str, set[str]] = {}
    manifest_components: dict[str, dict[str, list[str]]] = {}
    manifest_fields: dict[str, set[str]] = {}
    for manifest_path in manifests:
        content = cache.get(manifest_path)
        if content is None:
            handled_paths.add(manifest_path)
            invalid_manifest_paths.add(manifest_path)
            events.append(_failure(manifest_path, KeyError(manifest_path)))
            continue
        try:
            manifest = _load_json(content)
            _validate_manifest_identity(manifest)
            manifest_root = _manifest_root(manifest_path)
            user_config_by_root[manifest_root] = build_user_config_profile(
                manifest.get("userConfig")
            )
            manifest_line_iterator = iter(_manifest_handler_lines(content))
            component_fields = {field for field in ("skills", "commands") if field in manifest}
            component_references: dict[str, list[str]] = {}
            for field in component_fields:
                value = manifest[field]
                if not isinstance(value, (str, list)):
                    raise InvalidHookConfigurationError(
                        f"manifest {field} must be a relative path or array"
                    )
                values = [value] if isinstance(value, str) else value
                if not all(isinstance(item, str) for item in values):
                    raise InvalidHookConfigurationError(
                        f"manifest {field} entries must be relative paths"
                    )
                for item in values:
                    if item == "." and field != "skills":
                        raise InvalidHookConfigurationError(
                            "only manifest skills may use the bare-dot plugin root"
                        )
                    _resolve_local_path(manifest_root, item, allow_dot=True)
                component_references[field] = cast(list[str], values)
            if "hooks" in manifest:
                declared_hooks = manifest["hooks"]
                if not isinstance(declared_hooks, (str, dict, list)):
                    raise InvalidHookConfigurationError(
                        "manifest hooks must be an object, path, or array"
                    )
                items = declared_hooks if isinstance(declared_hooks, list) else [declared_hooks]
                inline_registrations: list[HookRegistration] = []
                inline_flow_inputs: list[HandlerFlowInput] = []
                manifest_references: set[str] = set()
                for item in items:
                    if isinstance(item, str):
                        reference_path = _resolve_reference(manifest_root, item)
                        if reference_path == manifest_path:
                            raise InvalidHookConfigurationError("manifest hook reference is cyclic")
                        manifest_references.add(reference_path)
                        continue
                    if not isinstance(item, dict):
                        raise InvalidHookConfigurationError(
                            "manifest hook items must be paths or objects"
                        )
                    parsed_inline = _registrations(
                        _manifest_inline_map(item),
                        source_kind="plugin_manifest_inline",
                        source_path=manifest_path,
                        activation_lifetime="plugin_enabled",
                        source_lines=manifest_line_iterator,
                        execution_root=manifest_root,
                        registration_limit=(
                            _MAX_REGISTRATIONS_PER_DOCUMENT - len(inline_registrations)
                        ),
                    )
                    inline_registrations.extend(parsed_inline.registrations)
                    inline_flow_inputs.extend(parsed_inline.flow_inputs)
                if inline_registrations:
                    add_document(
                        HookDocument(
                            source_kind="plugin_manifest_inline",
                            declaration_roles=("plugin_manifest_inline",),
                            source_path=manifest_path,
                            activation_lifetime="plugin_enabled",
                            content_digest=_digest("content", content),
                            registrations=tuple(inline_registrations),
                            flow_inputs=tuple(inline_flow_inputs),
                        )
                    )
                    handled_paths.add(manifest_path)
                for reference_path in manifest_references:
                    referenced_paths.setdefault(reference_path, set()).add(manifest_root)
            manifest_components[manifest_path] = component_references
            manifest_fields[manifest_path] = component_fields
        except (InvalidHookConfigurationError, TypeError) as exc:
            handled_paths.add(manifest_path)
            invalid_manifest_paths.add(manifest_path)
            events.append(_failure(manifest_path, exc))

    for manifest_path in manifest_fields:
        default_path = _default_path(_manifest_root(manifest_path))
        if default_path in handled_paths or default_path not in known_paths:
            continue
        handled_paths.add(default_path)
        content = cache.get(default_path)
        if content is None:
            events.append(_failure(default_path, KeyError(default_path)))
            continue
        try:
            add_document(
                _parse_hook_document(
                    default_path,
                    content,
                    "plugin_default",
                    "plugin_enabled",
                    execution_root=_manifest_root(manifest_path),
                )
            )
        except (InvalidHookConfigurationError, TypeError) as exc:
            events.append(_failure(default_path, exc))

    marketplace_manifest_roots = {_manifest_root(path) for path in manifest_fields}
    for entry in marketplace_entries:
        if not entry.strict or entry.plugin_root is None:
            continue
        if _manifest_path(entry.plugin_root) in invalid_manifest_paths:
            continue
        if entry.plugin_root in marketplace_manifest_roots:
            continue
        default_path = _default_path(entry.plugin_root)
        if default_path in handled_paths or default_path not in known_paths:
            continue
        handled_paths.add(default_path)
        content = cache.get(default_path)
        if content is None:
            events.append(_failure(default_path, KeyError(default_path)))
            continue
        try:
            add_document(
                _parse_hook_document(
                    default_path,
                    content,
                    "plugin_default",
                    "plugin_enabled",
                    execution_root=entry.plugin_root,
                )
            )
        except (InvalidHookConfigurationError, TypeError) as exc:
            events.append(_failure(default_path, exc))

    def reference_order(path: str) -> tuple[int, str]:
        return (path_rank.get(path, len(paths)), path)

    def inspect_referenced_document(
        reference_path: str,
        source_kind: str,
        activation_roots: set[str],
    ) -> None:
        """Inventory every distinct execution root for one physical hook document."""
        settings_work = settings_work_by_path.get(reference_path)
        if (
            settings_work is None
            and (settings := project_settings_metadata(reference_path)) is not None
        ):
            settings_work = _SettingsWork(
                source_path=reference_path,
                source_kind=settings[0],
                content_digest=_digest("content", ""),
                source_identity_digest=_permission_source_identity_digest(reference_path),
                raw=None,
                parse_error=KeyError(reference_path),
                permission_analysis=None,
                permission_source_lines=PermissionSourceLines(),
            )
            settings_work_by_path[reference_path] = settings_work
        existing_index = document_indexes.get(reference_path)
        if settings_work is not None and settings_work.parse_error is not None:
            handled_paths.add(reference_path)
            settings_hook_results.setdefault(
                reference_path,
                (LedgerOutcome.FAILED, _failure_reason(settings_work.parse_error)),
            )
            return
        if (
            settings_work is not None
            and settings_hook_results.get(reference_path, (None, None))[0] is LedgerOutcome.FAILED
        ):
            handled_paths.add(reference_path)
            return
        if settings_work is None and reference_path in handled_paths and existing_index is None:
            add_declaration_role(reference_path, source_kind)
            return
        content = cache.get(reference_path) if settings_work is None else None
        if settings_work is None and content is None:
            handled_paths.add(reference_path)
            events.append(_failure(reference_path, KeyError(reference_path)))
            return

        existing = documents[existing_index] if existing_index is not None else None
        existing_roots = (
            {
                registration.execution_root
                for registration in existing.registrations
                if registration.source_kind != "marketplace_plugin_inline"
            }
            if existing is not None
            else set()
        )
        pending_roots = sorted(activation_roots - existing_roots)
        if not pending_roots:
            handled_paths.add(reference_path)
            add_declaration_role(reference_path, source_kind)
            return

        try:
            added_registrations: list[HookRegistration] = []
            added_flow_inputs: list[HandlerFlowInput] = []
            template: HookDocument | None = None
            base_count = len(existing.registrations) if existing is not None else 0
            for execution_root in pending_roots:
                remaining = _MAX_REGISTRATIONS_PER_DOCUMENT - base_count - len(added_registrations)
                parsed = (
                    _parse_hook_mapping_document(
                        reference_path,
                        cast(_SettingsWork, settings_work).raw or {},
                        source_kind,
                        "plugin_enabled",
                        content_digest=cast(_SettingsWork, settings_work).content_digest,
                        execution_root=execution_root,
                        source_lines=settings_handler_lines_by_path.get(reference_path, ()),
                        registration_limit=remaining,
                    )
                    if settings_work is not None
                    else _parse_hook_document(
                        reference_path,
                        cast(str, content),
                        source_kind,
                        "plugin_enabled",
                        execution_root=execution_root,
                        registration_limit=remaining,
                    )
                )
                template = parsed
                added_registrations.extend(parsed.registrations)
                added_flow_inputs.extend(parsed.flow_inputs)
            if existing is not None:
                assert existing_index is not None
                documents[existing_index] = replace(
                    existing,
                    registrations=(*existing.registrations, *added_registrations),
                    flow_inputs=(*existing.flow_inputs, *added_flow_inputs),
                )
                add_declaration_role(reference_path, source_kind)
            elif template is not None:
                add_document(
                    replace(
                        template,
                        registrations=tuple(added_registrations),
                        flow_inputs=tuple(added_flow_inputs),
                    )
                )
            handled_paths.add(reference_path)
            if settings_work is not None:
                settings_hook_results[reference_path] = (LedgerOutcome.COMPLETED, None)
        except (InvalidHookConfigurationError, TypeError) as exc:
            discard_document(reference_path)
            handled_paths.add(reference_path)
            if settings_work is not None:
                settings_hook_results[reference_path] = (
                    LedgerOutcome.FAILED,
                    _failure_reason(exc),
                )
            else:
                events.append(_failure(reference_path, exc))

    for reference_path in sorted(referenced_paths, key=reference_order):
        inspect_referenced_document(
            reference_path,
            "plugin_manifest_reference",
            referenced_paths[reference_path],
        )

    marketplace_references: dict[str, set[str]] = {}
    staged_marketplace_registrations: dict[str, list[HookRegistration]] = {}
    staged_marketplace_flow_inputs: dict[str, list[HandlerFlowInput]] = {}
    failed_marketplace_documents: set[str] = set()

    def marketplace_document_failed(path: str) -> bool:
        return path in failed_marketplace_documents or (
            path in handled_paths and path not in document_indexes
        )

    def remaining_marketplace_registrations(path: str, pending_count: int) -> int:
        existing_index = document_indexes.get(path)
        existing_count = (
            len(documents[existing_index].registrations) if existing_index is not None else 0
        )
        return (
            _MAX_REGISTRATIONS_PER_DOCUMENT
            - existing_count
            - len(staged_marketplace_registrations.get(path, []))
            - pending_count
        )

    def fail_marketplace_document(path: str, error: HookRegistrationLimitError) -> None:
        """Discard every staged inline entry and fail the physical marketplace once."""
        staged_marketplace_registrations.pop(path, None)
        staged_marketplace_flow_inputs.pop(path, None)
        discard_document(path)
        if path in failed_marketplace_documents:
            return
        failed_marketplace_documents.add(path)
        handled_paths.add(path)
        events.append(_failure(path, error))

    for entry in marketplace_entries:
        entry_path = entry.ledger_path
        if (
            entry.plugin_root is not None
            and _manifest_path(entry.plugin_root) in invalid_manifest_paths
        ):
            continue
        content = cache.get(entry.marketplace_path)
        if content is None:
            record_marketplace_entry_failure(entry_path, KeyError(entry_path))
            continue
        if entry.plugin_root is None:
            # Remote sources cannot be mapped to the cache, but inline declarations
            # remain useful and are intentionally retained.
            if entry.hooks is not None:
                try:
                    items = entry.hooks if isinstance(entry.hooks, list) else [entry.hooks]
                    remote_inline_registrations: list[HookRegistration] = []
                    remote_inline_flow_inputs: list[HandlerFlowInput] = []
                    entry_line_iterator = iter(entry.handler_lines)
                    for item in items:
                        if isinstance(item, str):
                            continue
                        if marketplace_document_failed(entry.marketplace_path):
                            continue
                        parsed_inline = _registrations(
                            _manifest_inline_map(item),
                            source_kind="marketplace_plugin_inline",
                            source_path=entry.marketplace_path,
                            activation_lifetime="plugin_enabled",
                            source_lines=entry_line_iterator,
                            execution_root=None,
                            registration_limit=remaining_marketplace_registrations(
                                entry.marketplace_path,
                                len(remote_inline_registrations),
                            ),
                        )
                        remote_inline_registrations.extend(parsed_inline.registrations)
                        remote_inline_flow_inputs.extend(parsed_inline.flow_inputs)
                    if remote_inline_registrations:
                        staged_marketplace_registrations.setdefault(
                            entry.marketplace_path, []
                        ).extend(remote_inline_registrations)
                        staged_marketplace_flow_inputs.setdefault(
                            entry.marketplace_path, []
                        ).extend(remote_inline_flow_inputs)
                except HookRegistrationLimitError as exc:
                    fail_marketplace_document(entry.marketplace_path, exc)
                    record_marketplace_entry_failure(entry_path, KeyError(entry_path))
                    continue
                except (InvalidHookConfigurationError, TypeError) as exc:
                    record_marketplace_entry_failure(entry_path, exc)
                    continue
            record_marketplace_entry_failure(entry_path, KeyError(entry_path))
            continue

        if entry.hooks is not None:
            try:
                items = entry.hooks if isinstance(entry.hooks, list) else [entry.hooks]
                entry_inline_registrations: list[HookRegistration] = []
                entry_inline_flow_inputs: list[HandlerFlowInput] = []
                entry_references: set[str] = set()
                entry_line_iterator = iter(entry.handler_lines)
                for item in items:
                    if isinstance(item, str):
                        reference_path = _resolve_local_path(
                            entry.plugin_root, item, allow_dot=False
                        )
                        entry_references.add(reference_path)
                        continue
                    if marketplace_document_failed(entry.marketplace_path):
                        continue
                    parsed_inline = _registrations(
                        _manifest_inline_map(item),
                        source_kind="marketplace_plugin_inline",
                        source_path=entry.marketplace_path,
                        activation_lifetime="plugin_enabled",
                        source_lines=entry_line_iterator,
                        execution_root=entry.plugin_root,
                        registration_limit=remaining_marketplace_registrations(
                            entry.marketplace_path,
                            len(entry_inline_registrations),
                        ),
                    )
                    entry_inline_registrations.extend(parsed_inline.registrations)
                    entry_inline_flow_inputs.extend(parsed_inline.flow_inputs)
                if entry_inline_registrations:
                    staged_marketplace_registrations.setdefault(entry.marketplace_path, []).extend(
                        entry_inline_registrations
                    )
                    staged_marketplace_flow_inputs.setdefault(entry.marketplace_path, []).extend(
                        entry_inline_flow_inputs
                    )
                for reference_path in entry_references:
                    marketplace_references.setdefault(reference_path, set()).add(entry.plugin_root)
            except HookRegistrationLimitError as exc:
                fail_marketplace_document(entry.marketplace_path, exc)
            except (InvalidHookConfigurationError, TypeError) as exc:
                record_marketplace_entry_failure(entry_path, exc)

    for marketplace_path, registrations in staged_marketplace_registrations.items():
        if not registrations or marketplace_document_failed(marketplace_path):
            continue
        content = cache[marketplace_path]
        existing_index = document_indexes.get(marketplace_path)
        if existing_index is None:
            add_document(
                HookDocument(
                    source_kind="marketplace_plugin_inline",
                    declaration_roles=("marketplace_plugin_inline",),
                    source_path=marketplace_path,
                    activation_lifetime="plugin_enabled",
                    content_digest=_digest("content", content),
                    registrations=tuple(registrations),
                    flow_inputs=tuple(staged_marketplace_flow_inputs[marketplace_path]),
                )
            )
            continue
        existing = documents[existing_index]
        if len(existing.registrations) + len(registrations) > _MAX_REGISTRATIONS_PER_DOCUMENT:
            fail_marketplace_document(
                marketplace_path,
                HookRegistrationLimitError("aggregated marketplace registration limit exceeded"),
            )
            continue
        documents[existing_index] = replace(
            existing,
            registrations=(*existing.registrations, *registrations),
            flow_inputs=(
                *existing.flow_inputs,
                *staged_marketplace_flow_inputs[marketplace_path],
            ),
        )
        add_declaration_role(marketplace_path, "marketplace_plugin_inline")

    for reference_path in sorted(marketplace_references, key=reference_order):
        inspect_referenced_document(
            reference_path,
            "marketplace_plugin_reference",
            marketplace_references[reference_path],
        )

    frontmatter_attempted: set[str] = set()
    frontmatter_activations: dict[str, set[tuple[str, str, str | None, str]]] = {}
    frontmatter_failed: set[str] = set()
    frontmatter_hookless: set[str] = set()

    def inspect_frontmatter(
        path: str,
        source_kind: str,
        activation_lifetime: str,
        execution_root: str | None,
        runtime_status: str = "declared_unclassified",
    ) -> None:
        """Aggregate each distinct runtime activation of one physical Markdown document."""
        _namespace_value, path_parts = _path_parts(path)
        if source_kind.endswith("skill") and path_parts and path_parts[-1] == "skill.md":
            runtime_status = "runtime_unconfirmed"
        if path in frontmatter_failed or path in frontmatter_hookless:
            return
        existing_index = document_indexes.get(path)
        if path in handled_paths and existing_index is None:
            return
        if existing_index is not None and path not in frontmatter_activations:
            add_declaration_role(path, source_kind)
            return
        activation = (source_kind, activation_lifetime, execution_root, runtime_status)
        activations = frontmatter_activations.setdefault(path, set())
        if activation in activations:
            add_declaration_role(path, source_kind)
            return
        activations.add(activation)
        frontmatter_attempted.add(path)
        content = cache.get(path)
        if content is None:
            handled_paths.add(path)
            frontmatter_failed.add(path)
            events.append(_failure(path, KeyError(path)))
            return
        if path in manifest_limited_paths and not _frontmatter_has_explicit_hooks_key(content):
            frontmatter_hookless.add(path)
            return
        try:
            existing = documents[existing_index] if existing_index is not None else None
            document = _parse_frontmatter_document(
                path,
                content,
                source_kind,
                activation_lifetime,
                execution_root,
                runtime_status,
                registration_limit=(
                    _MAX_REGISTRATIONS_PER_DOCUMENT
                    - (len(existing.registrations) if existing is not None else 0)
                ),
            )
        except (InvalidHookConfigurationError, TypeError) as exc:
            discard_document(path)
            handled_paths.add(path)
            frontmatter_failed.add(path)
            events.append(_failure(path, exc))
            return
        if document is None:
            frontmatter_hookless.add(path)
            return
        handled_paths.add(path)
        if existing is None:
            add_document(document)
            return
        assert existing_index is not None
        documents[existing_index] = replace(
            existing,
            registrations=(*existing.registrations, *document.registrations),
            flow_inputs=(*existing.flow_inputs, *document.flow_inputs),
        )
        add_declaration_role(path, source_kind)

    plugin_roots_by_manifest = {
        manifest_path: _manifest_root(manifest_path) for manifest_path in manifest_fields
    }
    marketplace_command_overrides = {
        entry.plugin_root
        for entry in marketplace_entries
        if entry.plugin_root is not None and entry.commands is not None
    }
    marketplace_skill_overrides = {
        entry.plugin_root
        for entry in marketplace_entries
        if entry.strict
        and entry.plugin_root is not None
        and entry.skills is not None
        and entry.source_is_root
    }
    for manifest_path, plugin_root in plugin_roots_by_manifest.items():
        if plugin_root not in marketplace_skill_overrides:
            for path in sorted(
                _default_plugin_skill_paths(plugin_root, candidates_for_root(plugin_root)),
                key=reference_order,
            ):
                inspect_frontmatter(
                    path, "plugin_default_skill", "invocation_through_session", plugin_root
                )

        fields = manifest_fields.get(manifest_path, set())
        components = manifest_components.get(manifest_path, {})
        if "skills" in fields:
            custom_skills, missing_skills = _manifest_component_paths(
                plugin_root,
                components["skills"],
                component_kind="skills",
                candidates=candidates_for_root(plugin_root),
            )
            for missing_path in missing_skills:
                if missing_path not in frontmatter_attempted and missing_path not in handled_paths:
                    frontmatter_attempted.add(missing_path)
                    handled_paths.add(missing_path)
                    events.append(_failure(missing_path, KeyError(missing_path)))
            for path in sorted(custom_skills, key=reference_order):
                inspect_frontmatter(
                    path, "plugin_manifest_skill", "invocation_through_session", plugin_root
                )

        if "commands" in fields:
            custom_commands, missing_commands = _manifest_component_paths(
                plugin_root,
                components["commands"],
                component_kind="commands",
                candidates=candidates_for_root(plugin_root),
            )
            for missing_path in missing_commands:
                if missing_path not in frontmatter_attempted and missing_path not in handled_paths:
                    frontmatter_attempted.add(missing_path)
                    handled_paths.add(missing_path)
                    events.append(_failure(missing_path, KeyError(missing_path)))
            for path in sorted(custom_commands, key=reference_order):
                inspect_frontmatter(
                    path, "plugin_manifest_command", "invocation_through_session", plugin_root
                )
        else:
            for path in sorted(
                _default_plugin_command_paths(plugin_root, candidates_for_root(plugin_root)),
                key=reference_order,
            ):
                if plugin_root not in marketplace_command_overrides:
                    inspect_frontmatter(
                        path, "plugin_default_command", "invocation_through_session", plugin_root
                    )

        root_skill = _resolve_component_reference(plugin_root, "./SKILL.md")
        if (
            root_skill in known_paths
            and "skills" not in fields
            and not _has_default_plugin_skills_directory(
                plugin_root, candidates_for_root(plugin_root)
            )
        ):
            inspect_frontmatter(
                root_skill, "plugin_root_skill", "invocation_through_session", plugin_root
            )

    manifest_roots = set(plugin_roots_by_manifest.values())
    for entry in marketplace_entries:
        plugin_root = entry.plugin_root
        if plugin_root is None:
            continue
        if _manifest_path(plugin_root) in invalid_manifest_paths:
            continue
        if entry.strict and plugin_root not in manifest_roots:
            if plugin_root not in marketplace_skill_overrides:
                for path in sorted(
                    _default_plugin_skill_paths(plugin_root, candidates_for_root(plugin_root)),
                    key=reference_order,
                ):
                    inspect_frontmatter(
                        path, "plugin_default_skill", "invocation_through_session", plugin_root
                    )
                if entry.skills is None and not _has_default_plugin_skills_directory(
                    plugin_root, candidates_for_root(plugin_root)
                ):
                    root_skill = _resolve_component_reference(plugin_root, "./SKILL.md")
                    if root_skill in known_paths:
                        inspect_frontmatter(
                            root_skill,
                            "plugin_root_skill",
                            "invocation_through_session",
                            plugin_root,
                        )
            if entry.commands is None:
                for path in sorted(
                    _default_plugin_command_paths(plugin_root, candidates_for_root(plugin_root)),
                    key=reference_order,
                ):
                    inspect_frontmatter(
                        path, "plugin_default_command", "invocation_through_session", plugin_root
                    )

        for field, value, source_kind in (
            ("skills", entry.skills, "marketplace_plugin_skill"),
            ("commands", entry.commands, "marketplace_plugin_command"),
        ):
            if value is None:
                continue
            try:
                candidates = [
                    path
                    for path in candidates_for_root(plugin_root)
                    if path not in marketplace_path_set and path not in manifest_path_set
                ]
                selected, missing = _manifest_component_paths(
                    plugin_root,
                    value,
                    component_kind=field,
                    candidates=candidates,
                )
                for missing_path in missing:
                    if (
                        missing_path not in frontmatter_attempted
                        and missing_path not in handled_paths
                    ):
                        frontmatter_attempted.add(missing_path)
                        handled_paths.add(missing_path)
                        events.append(_failure(missing_path, KeyError(missing_path)))
                if field == "skills" and entry.strict and not selected and missing:
                    for fallback_path in sorted(
                        _default_plugin_skill_paths(plugin_root, candidates_for_root(plugin_root)),
                        key=reference_order,
                    ):
                        inspect_frontmatter(
                            fallback_path,
                            "plugin_default_skill",
                            "invocation_through_session",
                            plugin_root,
                        )
                for path in sorted(selected, key=reference_order):
                    inspect_frontmatter(
                        path, source_kind, "invocation_through_session", plugin_root
                    )
            except (InvalidHookConfigurationError, TypeError) as exc:
                record_marketplace_entry_failure(entry.ledger_path, exc)

    all_plugin_root_values = tuple(_manifest_root(manifest_path) for manifest_path in manifests)
    plugin_content_paths = {
        candidate for root in all_plugin_root_values for candidate in candidates_for_root(root)
    }
    for path in paths:
        _, parts = _path_parts(path)
        if not parts or path in frontmatter_attempted:
            continue
        is_plugin_content = path in plugin_content_paths
        if len(parts) == 1 and parts[0] in {"SKILL.md", "skill.md"} and not is_plugin_content:
            inspect_frontmatter(
                path,
                "root_skill",
                "invocation_through_session",
                _archive_or_project_root(path),
                "runtime_unconfirmed" if parts[0] == "skill.md" else "declared_unclassified",
            )
            continue
        if parts[:2] == (".claude", "skills") and parts[-1] == "SKILL.md":
            inspect_frontmatter(
                path,
                "project_skill",
                "invocation_through_session",
                _archive_or_project_root(path),
            )
            continue
        if (
            len(parts) >= 3
            and parts[:2] == (".claude", "commands")
            and parts[-1].lower().endswith(".md")
        ):
            inspect_frontmatter(
                path,
                "project_command",
                "invocation_through_session",
                _archive_or_project_root(path),
            )
            continue
        if (
            len(parts) == 3
            and parts[:2] == (".claude", "agents")
            and parts[-1].lower().endswith(".md")
            and not is_plugin_content
        ):
            inspect_frontmatter(
                path,
                "project_agent",
                "project_subagent",
                _archive_or_project_root(path),
            )

    documents.sort(key=lambda document: reference_order(document.source_path))
    flow_batch = analyze_documents(
        tuple(
            DocumentFlowInput(
                source_kind=document.source_kind,
                declaration_roles=document.declaration_roles,
                source_path=document.source_path,
                activation_lifetime=document.activation_lifetime,
                content_digest=document.content_digest,
                handlers=document.flow_inputs,
            )
            for document in documents
        ),
        local_file_cache=cast(dict[str, str], state.get("local_file_cache") or {}),
        user_config_by_root=user_config_by_root,
        python_ast_cache_key=cast(str | None, state.get("python_ast_cache_key")),
    )
    findings_by_owner: dict[FlowWorkRef, list[Finding]] = {}
    for owned in flow_batch.findings:
        findings_by_owner.setdefault(owned.owner, []).append(owned.finding)

    settings_permission_findings: dict[str, Finding] = {}
    for path, settings_work in settings_work_by_path.items():
        if settings_work.permission_analysis is None:
            continue
        permission_finding = build_bh3_finding(
            settings_work.permission_analysis,
            source_path=path,
        )
        if permission_finding is not None:
            settings_permission_findings[path] = permission_finding

    findings: list[Finding] = []
    document_owners: set[FlowWorkRef] = set()
    documents_by_owner: dict[FlowWorkRef, HookDocument] = {}
    settings_findings_by_path: dict[str, list[Finding]] = {}
    for document in documents:
        owner = FlowWorkRef(document.source_path)
        document_owners.add(owner)
        documents_by_owner[owner] = document
        document_findings = (
            [_bh1_finding(document, cache_path_set)] if document.registrations else []
        )
        document_findings.extend(findings_by_owner.get(owner, []))
        permission_finding = settings_permission_findings.get(document.source_path)
        if permission_finding is not None:
            document_findings.append(permission_finding)
        findings.extend(document_findings)
        if document.source_path in settings_work_by_path:
            settings_findings_by_path[document.source_path] = document_findings
        else:
            events.append(_completed(document.source_path, document_findings))

    for path in sorted(settings_work_by_path, key=reference_order):
        if path not in settings_findings_by_path:
            permission_finding = settings_permission_findings.get(path)
            path_findings = [permission_finding] if permission_finding is not None else []
            findings.extend(path_findings)
            settings_findings_by_path[path] = path_findings
        terminal = _settings_terminal(
            settings_work_by_path[path],
            settings_hook_results.get(path),
            settings_findings_by_path[path],
        )
        if terminal is not None:
            events.append(terminal)

    findings.extend(
        owned.finding for owned in flow_batch.findings if owned.owner not in document_owners
    )
    occupied_work_refs = {
        FlowWorkRef(event["path"], event["start_line"], event["end_line"]) for event in events
    }
    for work in flow_batch.work:
        original_ref = work.ref
        if original_ref in occupied_work_refs:
            if original_ref in document_owners and work.outcome is LedgerOutcome.COMPLETED:
                continue
            collision_document = documents_by_owner.get(original_ref)
            source_line = (
                min(
                    (registration.source_line for registration in collision_document.registrations),
                    default=1,
                )
                if collision_document is not None
                else original_ref.start_line or 1
            )
            activation_ref = FlowWorkRef(original_ref.path, source_line, source_line)
            while activation_ref in occupied_work_refs:
                source_line += 1
                activation_ref = FlowWorkRef(original_ref.path, source_line, source_line)
            work = replace(work, ref=activation_ref)
        occupied_work_refs.add(work.ref)
        events.append(_flow_terminal(work, findings_by_owner.get(original_ref, [])))

    synthetic_event_ids = {id(event) for event in marketplace_entry_events.values()}
    occupied_work_ids = {
        event["work_id"] for event in events if id(event) not in synthetic_event_ids
    }
    for base_path, event in marketplace_entry_events.items():
        if event["work_id"] in occupied_work_ids:
            suffix = 1
            candidate_path = f"{base_path}#ledger[{suffix}]"
            candidate_work_id = inspection_work_id(
                ANALYZER_ID,
                candidate_path,
                event["start_line"],
                event["end_line"],
            )
            while candidate_work_id in occupied_work_ids:
                suffix += 1
                candidate_path = f"{base_path}#ledger[{suffix}]"
                candidate_work_id = inspection_work_id(
                    ANALYZER_ID,
                    candidate_path,
                    event["start_line"],
                    event["end_line"],
                )
            event["path"] = candidate_path
            event["work_id"] = candidate_work_id
        occupied_work_ids.add(event["work_id"])

    return {
        "findings": findings,
        "inspection_ledger": events,
        "analyzer_status_events": [analyzer_status_for_events(ANALYZER_ID, events)],
    }
