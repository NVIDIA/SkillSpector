# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Structured AISOP/AISP bundle detection helpers."""

from __future__ import annotations

import json
from pathlib import Path

_SKIP_DIRS = frozenset(
    {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".pytest_cache"}
)

_AISOP_PROTOCOL_PREFIXES = ("AISOP V", "AISP V")


def extract_structured_skill_context(skill_dir: Path) -> dict[str, object] | None:
    """Return structured-skill context for the first valid bundle under *skill_dir*."""
    if not skill_dir.is_dir():
        return None

    for path in _iter_aisop_files(skill_dir):
        context = _parse_bundle_path(path)
        if context is not None:
            return context

    return None


def _iter_aisop_files(skill_dir: Path) -> list[Path]:
    """Yield candidate *.aisop.json files under a directory, skipping noisy paths."""
    files: list[Path] = []
    for path in sorted(skill_dir.rglob("*.aisop.json")):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if any(part.startswith(".") and part != ".aisop" for part in path.parts[:-1]):
            # Keep hidden metadata directories out of structured-skill detection.
            continue
        if path.is_file():
            files.append(path)
    return files


def _parse_bundle_path(bundle_path: Path) -> dict[str, object] | None:
    """Parse and validate one AISOP/AISP bundle path."""
    try:
        data = json.loads(bundle_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    return _parse_bundle_payload(bundle_path, data)


def _parse_bundle_payload(bundle_path: Path, payload: object) -> dict[str, object] | None:
    """Parse the minimal phase-1 AISOP/AISP payload contract."""
    if not isinstance(payload, list) or len(payload) != 2:
        return None

    system_msg = _normalize_mapping(payload[0])
    user_msg = _normalize_mapping(payload[1])
    if system_msg is None or user_msg is None:
        return None

    system_content = _normalize_mapping(system_msg.get("content"))
    if system_content is None:
        return None

    protocol = system_content.get("protocol")
    if not isinstance(protocol, str) or not protocol.startswith(_AISOP_PROTOCOL_PREFIXES):
        return None
    if system_msg.get("role") != "system":
        return None

    user_content = user_msg.get("content")
    contract = _find_contract_payload(user_content)
    if contract is None:
        return None

    if user_msg.get("role") != "user":
        return None

    layout_kind = protocol.split()[0]
    declared_tools = _first_non_empty(
        (
            system_content.get("declared_tools"),
            system_content.get("tools"),
            contract.get("declared_tools"),
            contract.get("tools"),
        )
    )
    functions = contract.get("functions")
    function_names = _extract_function_names(functions)
    constraint_anchors = _extract_constraint_anchors(functions)
    resource_anchors = _extract_resource_anchors(contract.get("resources"))

    return {
        "layout_kind": layout_kind,
        "format": system_content.get("format", layout_kind),
        "protocol": protocol,
        "bundle_path": str(bundle_path.resolve()),
        "declared_tools": declared_tools,
        "workflow_nodes": function_names,
        "constraint_anchors": constraint_anchors,
        "resource_anchors": resource_anchors,
    }


def _normalize_mapping(value: object) -> dict[str, object] | None:
    """Return a dict if *value* is a mapping object."""
    return value if isinstance(value, dict) else None


def _find_contract_payload(content: object) -> dict[str, object] | None:
    """Locate the AISOP/AISP contract payload in a user message."""
    container = _normalize_mapping(content)
    if container is None:
        return None

    for key in ("aisop", "aisp_contract"):
        value = container.get(key)
        if isinstance(value, dict):
            return value
    return None


def _first_non_empty(values: tuple[object, ...]) -> list[str]:
    """Return a stable deduplicated string list from candidate values."""
    result: list[str] = []
    seen = set[str]()
    for value in values:
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, str):
                continue
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
    return result


def _extract_function_names(functions: object) -> list[str]:
    """Extract function names from a dictionary/list of workflow nodes."""
    names: list[str] = []
    seen: set[str] = set()

    if isinstance(functions, dict):
        items = functions.items()
        for name, node in items:
            if isinstance(name, str):
                n = name.strip()
                if n and n not in seen:
                    seen.add(n)
                    names.append(n)
            if isinstance(node, dict):
                names.extend(_extract_function_names(node.get("functions")))
    elif isinstance(functions, list):
        for item in functions:
            if not isinstance(item, dict):
                continue
            node_name = item.get("name")
            if isinstance(node_name, str):
                n = node_name.strip()
                if n and n not in seen:
                    seen.add(n)
                    names.append(n)
            names.extend(_extract_function_names(item.get("functions")))

    return names


def _extract_constraint_anchors(functions: object) -> list[str]:
    """Extract anchors from content.functions.*.constraints."""
    anchors: list[str] = []
    seen: set[str] = set()

    def _walk(nodes: object) -> None:
        if isinstance(nodes, dict):
            for maybe_node in nodes.values():
                if isinstance(maybe_node, dict):
                    constraints = maybe_node.get("constraints")
                    if isinstance(constraints, list):
                        for constraint in constraints:
                            if not isinstance(constraint, dict):
                                continue
                            anchor = constraint.get("anchor")
                            if isinstance(anchor, str):
                                a = anchor.strip()
                                if a and a not in seen:
                                    seen.add(a)
                                    anchors.append(a)
                    _walk(maybe_node.get("functions"))
                elif isinstance(maybe_node, list):
                    _walk(maybe_node)
        elif isinstance(nodes, list):
            for item in nodes:
                if isinstance(item, dict):
                    _walk(item)

    _walk(functions)
    return anchors


def _extract_resource_anchors(resources: object) -> list[str]:
    """Extract resource path anchors from content.aisp_contract.resources."""
    paths: list[str] = []
    seen: set[str] = set()

    def _collect(path: str) -> None:
        p = path.strip()
        if p and p not in seen:
            seen.add(p)
            paths.append(p)

    def _walk(value: object) -> None:
        if isinstance(value, dict):
            for val in value.values():
                if isinstance(val, dict):
                    resource_path = val.get("path")
                    if isinstance(resource_path, str):
                        _collect(resource_path)
                    _walk(val.get("resources"))
                elif isinstance(val, str):
                    _collect(val)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    _collect(item)
                elif isinstance(item, dict):
                    resource_path = item.get("path")
                    if isinstance(resource_path, str):
                        _collect(resource_path)
                    _walk(item.get("resources"))

    _walk(resources)
    return paths
