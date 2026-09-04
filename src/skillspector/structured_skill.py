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

"""Resource-bounded structured AISOP/AISP bundle detection helpers.

The primary API consumes content that bundle discovery and caching already
accepted. It never performs a second filesystem traversal or rereads a file,
so structured-skill detection cannot bypass the enclosing scan's bounds.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from skillspector.input_handler import (
    _FileOpenError,
    _open_regular_file_no_follow,
    _UnsafeFileError,
)

_SKIP_DIRS = frozenset(
    {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".pytest_cache"}
)

_AISOP_PROTOCOL_PREFIXES = ("AISOP V", "AISP V")

# These limits apply to one structured-skill extraction, not to each file.
# The enclosing bundle scanner may impose tighter aggregate bounds.
MAX_STRUCTURED_CANDIDATES = 64
MAX_STRUCTURED_DOCUMENT_BYTES = 256 * 1024
MAX_STRUCTURED_TOTAL_INPUT_BYTES = 1024 * 1024
MAX_STRUCTURED_NESTING = 64
MAX_STRUCTURED_NODES = 4096
MAX_STRUCTURED_OUTPUT_RECORDS = 512
MAX_STRUCTURED_RUNTIME_SECONDS = 2.0

# The legacy filesystem wrapper is retained for multi-skill discovery. Its
# traversal is independently bounded and feeds the same cache-only core.
MAX_STRUCTURED_DISCOVERY_ENTRIES = 4096
MAX_STRUCTURED_DIRECTORY_ENTRIES = 1024
MAX_STRUCTURED_DISCOVERY_DEPTH = 64


@dataclass(frozen=True)
class StructuredSkillLimitation:
    """One partial-coverage record using inspection-ledger-compatible fields."""

    path: str
    reason_code: str
    resource: str
    observed_bytes: int | None = None
    limit_bytes: int | None = None
    observed_artifacts: int | None = None
    limit_artifacts: int | None = None
    observed_depth: int | None = None
    limit_depth: int | None = None
    observed_records: int | None = None
    limit_records: int | None = None
    observed_seconds: float | None = None
    limit_seconds: float | None = None

    def as_ledger_metadata(self) -> dict[str, object]:
        """Return fields a caller can translate directly into a ledger event."""
        result: dict[str, object] = {
            "outcome": "partial",
            "record_type": "system",
            "phase": "structured_skill",
            "path": self.path,
            "reason_code": self.reason_code,
            "resource": self.resource,
        }
        for name in (
            "observed_bytes",
            "limit_bytes",
            "observed_artifacts",
            "limit_artifacts",
            "observed_depth",
            "limit_depth",
            "observed_records",
            "limit_records",
            "observed_seconds",
            "limit_seconds",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True)
class StructuredSkillExtractionResult:
    """Bounded structured context together with explicit coverage accounting."""

    context: dict[str, object] | None
    limitations: tuple[StructuredSkillLimitation, ...] = ()
    candidates_examined: int = 0
    input_bytes_examined: int = 0
    nodes_examined: int = 0
    output_records: int = 0

    @property
    def complete(self) -> bool:
        """Whether structured detection exhausted all applicable bounded work."""
        return not self.limitations


@dataclass
class _ExtractionBudget:
    started_at: float
    deadline: float
    runtime_limit: float
    clock: Callable[[], float]
    input_bytes: int = 0
    nodes: int = 0
    outputs: int = 0
    limitations: list[StructuredSkillLimitation] = field(default_factory=list)

    def check_runtime(self, path: str) -> None:
        now = self.clock()
        elapsed = max(0.0, now - self.started_at)
        if now > self.deadline:
            raise _LimitReachedError(
                StructuredSkillLimitation(
                    path=path,
                    reason_code="runtime_limit",
                    resource="structured_runtime",
                    observed_seconds=elapsed,
                    limit_seconds=self.runtime_limit,
                )
            )

    def visit_node(self, path: str, *, depth: int) -> None:
        self.check_runtime(path)
        if depth > MAX_STRUCTURED_NESTING:
            raise _LimitReachedError(
                StructuredSkillLimitation(
                    path=path,
                    reason_code="traversal_depth_limit",
                    resource="structured_nesting",
                    observed_depth=depth,
                    limit_depth=MAX_STRUCTURED_NESTING,
                )
            )
        self.nodes += 1
        if self.nodes > MAX_STRUCTURED_NODES:
            raise _LimitReachedError(
                StructuredSkillLimitation(
                    path=path,
                    reason_code="artifact_count_limit",
                    resource="structured_nodes",
                    observed_artifacts=self.nodes,
                    limit_artifacts=MAX_STRUCTURED_NODES,
                )
            )

    def add_output(self, path: str) -> None:
        self.check_runtime(path)
        self.outputs += 1
        if self.outputs > MAX_STRUCTURED_OUTPUT_RECORDS:
            raise _LimitReachedError(
                StructuredSkillLimitation(
                    path=path,
                    reason_code="output_limit",
                    resource="structured_output_records",
                    observed_records=self.outputs,
                    limit_records=MAX_STRUCTURED_OUTPUT_RECORDS,
                )
            )


class _LimitReachedError(Exception):
    """Internal structured-work budget signal."""

    def __init__(self, limitation: StructuredSkillLimitation):
        super().__init__(limitation.reason_code)
        self.limitation = limitation


def _result(
    context: dict[str, object] | None,
    budget: _ExtractionBudget,
    *,
    candidates_examined: int,
) -> StructuredSkillExtractionResult:
    return StructuredSkillExtractionResult(
        context=context,
        limitations=tuple(budget.limitations),
        candidates_examined=candidates_examined,
        input_bytes_examined=budget.input_bytes,
        nodes_examined=budget.nodes,
        output_records=budget.outputs,
    )


def _record_once(
    limitations: list[StructuredSkillLimitation], limitation: StructuredSkillLimitation
) -> None:
    key = (limitation.path, limitation.reason_code, limitation.resource)
    if not any((item.path, item.reason_code, item.resource) == key for item in limitations):
        limitations.append(limitation)


def _safe_component_path(path: str) -> str | None:
    normalized = path.replace("\\", "/")
    if not normalized or normalized.startswith(("/", "//")) or "\x00" in normalized:
        return None
    if len(normalized) >= 2 and normalized[1] == ":":
        return None
    parts = PurePosixPath(normalized).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None
    return PurePosixPath(*parts).as_posix()


def _is_candidate(path: str) -> bool:
    if not path.lower().endswith(".aisop.json"):
        return False
    parts = PurePosixPath(path).parts
    if any(part in _SKIP_DIRS for part in parts):
        return False
    return not any(part.startswith(".") and part != ".aisop" for part in parts[:-1])


def _bounded_utf8(text: str, limit: int) -> tuple[bytes, bool]:
    """Encode at most *limit* bytes without first allocating the full encoding."""
    output = bytearray()
    offset = 0
    chunk_chars = min(16 * 1024, limit + 1)
    while offset < len(text) and len(output) <= limit:
        chunk = text[offset : offset + chunk_chars].encode("utf-8")
        remaining = limit + 1 - len(output)
        output.extend(chunk[:remaining])
        offset += chunk_chars
    return bytes(output), offset < len(text) or len(output) > limit


def _candidate_bytes(
    path: str,
    *,
    raw_file_cache: Mapping[str, bytes] | None,
    file_cache: Mapping[str, str] | None,
) -> tuple[bytes, bool, int] | None:
    raw = raw_file_cache.get(path) if raw_file_cache is not None else None
    if isinstance(raw, bytes):
        observed = len(raw)
        return (
            raw[: MAX_STRUCTURED_DOCUMENT_BYTES + 1],
            observed > MAX_STRUCTURED_DOCUMENT_BYTES,
            observed,
        )

    text = file_cache.get(path) if file_cache is not None else None
    if not isinstance(text, str):
        return None
    bounded, truncated = _bounded_utf8(text, MAX_STRUCTURED_DOCUMENT_BYTES)
    observed = MAX_STRUCTURED_DOCUMENT_BYTES + 1 if truncated else len(bounded)
    return bounded, truncated, observed


def _validate_json_structure(payload: object, budget: _ExtractionBudget, path: str) -> None:
    """Bound all JSON nodes and nesting before semantic traversals begin."""
    stack: list[tuple[object, int]] = [(payload, 0)]
    while stack:
        value, depth = stack.pop()
        budget.visit_node(path, depth=depth)
        if isinstance(value, dict):
            if len(value) > MAX_STRUCTURED_NODES - budget.nodes:
                raise _LimitReachedError(
                    StructuredSkillLimitation(
                        path=path,
                        reason_code="artifact_count_limit",
                        resource="structured_nodes",
                        observed_artifacts=budget.nodes + len(value),
                        limit_artifacts=MAX_STRUCTURED_NODES,
                    )
                )
            stack.extend((item, depth + 1) for item in reversed(value.values()))
        elif isinstance(value, list):
            if len(value) > MAX_STRUCTURED_NODES - budget.nodes:
                raise _LimitReachedError(
                    StructuredSkillLimitation(
                        path=path,
                        reason_code="artifact_count_limit",
                        resource="structured_nodes",
                        observed_artifacts=budget.nodes + len(value),
                        limit_artifacts=MAX_STRUCTURED_NODES,
                    )
                )
            stack.extend((item, depth + 1) for item in reversed(value))


def _add_unique_output(
    value: object,
    output: list[str],
    seen: set[str],
    budget: _ExtractionBudget,
    path: str,
) -> None:
    if not isinstance(value, str):
        return
    normalized = value.strip()
    if not normalized or normalized in seen:
        return
    budget.add_output(path)
    seen.add(normalized)
    output.append(normalized)


def _collect_declared_tools(
    values: tuple[object, ...], budget: _ExtractionBudget, path: str
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        budget.check_runtime(path)
        if isinstance(value, list):
            for item in value:
                _add_unique_output(item, result, seen, budget, path)
    return result


def _collect_functions(
    functions: object,
    budget: _ExtractionBudget,
    path: str,
) -> tuple[list[str], list[str]]:
    names: list[str] = []
    constraints: list[str] = []
    seen_names: set[str] = set()
    seen_constraints: set[str] = set()

    def collect_constraint(value: object) -> None:
        if isinstance(value, str):
            _add_unique_output(value, constraints, seen_constraints, budget, path)
        elif isinstance(value, dict):
            _add_unique_output(value.get("anchor"), constraints, seen_constraints, budget, path)

    def walk(nodes: object, depth: int = 0) -> None:
        budget.check_runtime(path)
        if depth > MAX_STRUCTURED_NESTING:
            raise _LimitReachedError(
                StructuredSkillLimitation(
                    path=path,
                    reason_code="traversal_depth_limit",
                    resource="structured_nesting",
                    observed_depth=depth,
                    limit_depth=MAX_STRUCTURED_NESTING,
                )
            )
        if isinstance(nodes, dict):
            for name, node in nodes.items():
                _add_unique_output(name, names, seen_names, budget, path)
                if not isinstance(node, dict):
                    continue
                node_constraints = node.get("constraints")
                if isinstance(node_constraints, list):
                    for constraint in node_constraints:
                        collect_constraint(constraint)
                walk(node.get("functions"), depth + 1)
        elif isinstance(nodes, list):
            for item in nodes:
                if not isinstance(item, dict):
                    continue
                _add_unique_output(item.get("name"), names, seen_names, budget, path)
                item_constraints = item.get("constraints")
                if isinstance(item_constraints, list):
                    for constraint in item_constraints:
                        collect_constraint(constraint)
                walk(item.get("functions"), depth + 1)

    walk(functions)
    return names, constraints


def _collect_resources(
    resources: object,
    budget: _ExtractionBudget,
    path: str,
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    def walk(value: object, depth: int = 0) -> None:
        budget.check_runtime(path)
        if depth > MAX_STRUCTURED_NESTING:
            raise _LimitReachedError(
                StructuredSkillLimitation(
                    path=path,
                    reason_code="traversal_depth_limit",
                    resource="structured_nesting",
                    observed_depth=depth,
                    limit_depth=MAX_STRUCTURED_NESTING,
                )
            )
        if isinstance(value, dict):
            for item in value.values():
                if isinstance(item, dict):
                    _add_unique_output(item.get("path"), result, seen, budget, path)
                    walk(item.get("resources"), depth + 1)
                elif isinstance(item, str):
                    _add_unique_output(item, result, seen, budget, path)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    _add_unique_output(item, result, seen, budget, path)
                elif isinstance(item, dict):
                    _add_unique_output(item.get("path"), result, seen, budget, path)
                    walk(item.get("resources"), depth + 1)

    walk(resources)
    return result


def _bundle_display_path(skill_dir: Path, component_path: str) -> str:
    # Avoid Path.resolve(): the cache API must not consult the filesystem.
    root = skill_dir if skill_dir.is_absolute() else skill_dir.absolute()
    return str(root.joinpath(*PurePosixPath(component_path).parts))


def _parse_bundle_payload(
    bundle_path: Path | str,
    payload: object,
    *,
    budget: _ExtractionBudget | None = None,
    ledger_path: str | None = None,
) -> dict[str, object] | None:
    """Parse the minimal phase-1 AISOP/AISP payload contract under a budget."""
    path = str(bundle_path)
    work_path = ledger_path or path
    owned_budget = budget is None
    if budget is None:
        clock = time.monotonic
        started_at = clock()
        budget = _ExtractionBudget(
            started_at=started_at,
            deadline=started_at + MAX_STRUCTURED_RUNTIME_SECONDS,
            runtime_limit=MAX_STRUCTURED_RUNTIME_SECONDS,
            clock=clock,
        )
        _validate_json_structure(payload, budget, work_path)

    if not isinstance(payload, list) or len(payload) != 2:
        return None

    system_msg = payload[0] if isinstance(payload[0], dict) else None
    user_msg = payload[1] if isinstance(payload[1], dict) else None
    if system_msg is None or user_msg is None:
        return None

    system_content = system_msg.get("content")
    user_content = user_msg.get("content")
    if not isinstance(system_content, dict) or not isinstance(user_content, dict):
        return None

    protocol = system_content.get("protocol")
    if (
        not isinstance(protocol, str)
        or not protocol.startswith(_AISOP_PROTOCOL_PREFIXES)
        or system_msg.get("role") != "system"
        or user_msg.get("role") != "user"
    ):
        return None

    aisop_payload = user_content.get("aisop")
    aisp_contract = user_content.get("aisp_contract")
    aisop_payload = aisop_payload if isinstance(aisop_payload, dict) else None
    aisp_contract = aisp_contract if isinstance(aisp_contract, dict) else None
    if aisop_payload is None and aisp_contract is None:
        return None

    declared_tools = _collect_declared_tools(
        (
            system_content.get("declared_tools"),
            system_content.get("tools"),
            user_content.get("declared_tools"),
            user_content.get("tools"),
            aisop_payload.get("declared_tools") if aisop_payload else None,
            aisop_payload.get("tools") if aisop_payload else None,
            aisp_contract.get("declared_tools") if aisp_contract else None,
            aisp_contract.get("tools") if aisp_contract else None,
        ),
        budget,
        work_path,
    )
    functions = user_content.get("functions")
    if functions is None and aisop_payload is not None:
        functions = aisop_payload.get("functions")
    if functions is None and aisp_contract is not None:
        functions = aisp_contract.get("functions")
    function_names, constraint_anchors = _collect_functions(functions, budget, work_path)
    resource_anchors = _collect_resources(
        aisp_contract.get("resources") if aisp_contract is not None else None,
        budget,
        work_path,
    )

    if not function_names and not resource_anchors:
        return None

    layout_kind = protocol.split()[0]
    result = {
        "layout_kind": layout_kind,
        "format": system_content.get("format", layout_kind),
        "protocol": protocol,
        "bundle_path": path,
        "declared_tools": declared_tools,
        "workflow_nodes": function_names,
        "constraint_anchors": constraint_anchors,
        "resource_anchors": resource_anchors,
    }
    if owned_budget:
        budget.check_runtime(work_path)
    return result


def extract_structured_skill_context_from_cache(
    skill_dir: Path,
    component_paths: Iterable[str] | None = None,
    *,
    raw_file_cache: Mapping[str, bytes] | None = None,
    file_cache: Mapping[str, str] | None = None,
    clock: Callable[[], float] = time.monotonic,
    deadline: float | None = None,
) -> StructuredSkillExtractionResult:
    """Extract the first valid structured context from already-bounded caches.

    Applicable candidates are collected only up to
    :data:`MAX_STRUCTURED_CANDIDATES`. If the candidate set exceeds that bound,
    no arbitrary subset is selected: extraction returns partial without parsing
    a candidate. Otherwise candidates are processed in stable lexical order.
    This function performs no filesystem reads or traversal. ``deadline`` is an
    absolute value from ``clock``; when supplied, the tighter of that shared
    caller deadline and the local runtime ceiling is enforced.
    """
    started_at = clock()
    own_deadline = started_at + MAX_STRUCTURED_RUNTIME_SECONDS
    effective_deadline = own_deadline if deadline is None else min(own_deadline, deadline)
    budget = _ExtractionBudget(
        started_at=started_at,
        deadline=effective_deadline,
        runtime_limit=max(0.0, effective_deadline - started_at),
        clock=clock,
    )
    source_paths: Iterable[str]
    if component_paths is None:
        source_paths = dict.fromkeys(
            [
                *(raw_file_cache.keys() if raw_file_cache is not None else ()),
                *(file_cache.keys() if file_cache is not None else ()),
            ]
        )
    else:
        source_paths = component_paths

    candidates: list[str] = []
    seen: set[str] = set()
    try:
        for raw_path in source_paths:
            budget.check_runtime(str(raw_path))
            safe_path = _safe_component_path(str(raw_path))
            if safe_path is None or safe_path in seen or not _is_candidate(safe_path):
                continue
            seen.add(safe_path)
            candidates.append(safe_path)
            if len(candidates) > MAX_STRUCTURED_CANDIDATES:
                _record_once(
                    budget.limitations,
                    StructuredSkillLimitation(
                        path=safe_path,
                        reason_code="artifact_count_limit",
                        resource="structured_candidates",
                        observed_artifacts=len(candidates),
                        limit_artifacts=MAX_STRUCTURED_CANDIDATES,
                    ),
                )
                return _result(None, budget, candidates_examined=0)
    except _LimitReachedError as exc:
        _record_once(budget.limitations, exc.limitation)
        return _result(None, budget, candidates_examined=0)

    examined = 0
    for path in sorted(candidates):
        examined += 1
        try:
            budget.check_runtime(path)
            candidate = _candidate_bytes(
                path,
                raw_file_cache=raw_file_cache,
                file_cache=file_cache,
            )
            if candidate is None:
                continue
            data, per_document_truncated, observed_bytes = candidate
            if per_document_truncated:
                _record_once(
                    budget.limitations,
                    StructuredSkillLimitation(
                        path=path,
                        reason_code="size_limit",
                        resource="structured_document_bytes",
                        observed_bytes=observed_bytes,
                        limit_bytes=MAX_STRUCTURED_DOCUMENT_BYTES,
                    ),
                )
                continue
            if budget.input_bytes + len(data) > MAX_STRUCTURED_TOTAL_INPUT_BYTES:
                _record_once(
                    budget.limitations,
                    StructuredSkillLimitation(
                        path=path,
                        reason_code="total_bytes_limit",
                        resource="structured_total_input_bytes",
                        observed_bytes=budget.input_bytes + len(data),
                        limit_bytes=MAX_STRUCTURED_TOTAL_INPUT_BYTES,
                    ),
                )
                break
            budget.input_bytes += len(data)
            try:
                payload = json.loads(data.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            except (ValueError, OverflowError):
                # Safe JSON scalar constructors may reject syntactically valid,
                # attacker-sized numeric values (for example Python's integer
                # digit ceiling). Treat that as incomplete structured parsing,
                # never as a graph crash or a clean non-candidate.
                raise _LimitReachedError(
                    StructuredSkillLimitation(
                        path=path,
                        reason_code="output_limit",
                        resource="structured_scalar_conversion",
                        observed_records=1,
                        limit_records=0,
                    )
                ) from None
            except RecursionError:
                raise _LimitReachedError(
                    StructuredSkillLimitation(
                        path=path,
                        reason_code="traversal_depth_limit",
                        resource="structured_nesting",
                        observed_depth=MAX_STRUCTURED_NESTING + 1,
                        limit_depth=MAX_STRUCTURED_NESTING,
                    )
                ) from None
            budget.check_runtime(path)
            _validate_json_structure(payload, budget, path)
            context = _parse_bundle_payload(
                _bundle_display_path(skill_dir, path),
                payload,
                budget=budget,
                ledger_path=path,
            )
            budget.check_runtime(path)
            if context is not None:
                return _result(context, budget, candidates_examined=examined)
        except _LimitReachedError as exc:
            _record_once(budget.limitations, exc.limitation)
            if exc.limitation.reason_code == "runtime_limit":
                break
            if exc.limitation.resource in {
                "structured_nodes",
                "structured_nesting",
                "structured_output_records",
            }:
                break

    return _result(None, budget, candidates_examined=examined)


def _bounded_filesystem_cache(
    skill_dir: Path,
    *,
    clock: Callable[[], float],
    deadline: float,
) -> tuple[list[str], dict[str, bytes], bool]:
    """Discover and read candidates for the compatibility API under hard bounds."""
    if not skill_dir.is_dir():
        return [], {}, True

    entries_seen = 0
    total_bytes = 0
    candidates: list[str] = []
    raw_cache: dict[str, bytes] = {}
    stack: list[tuple[Path, PurePosixPath, int]] = [(skill_dir, PurePosixPath("."), 0)]

    while stack:
        if clock() > deadline:
            return [], {}, False
        directory, relative_dir, depth = stack.pop()
        if depth > MAX_STRUCTURED_DISCOVERY_DEPTH:
            return [], {}, False
        bounded_entries: list[os.DirEntry[str]] = []
        try:
            with os.scandir(directory) as scanner:
                for entry in scanner:
                    bounded_entries.append(entry)
                    if len(bounded_entries) > MAX_STRUCTURED_DIRECTORY_ENTRIES:
                        return [], {}, False
        except OSError:
            return [], {}, False

        child_directories: list[tuple[Path, PurePosixPath, int]] = []
        for entry in sorted(bounded_entries, key=lambda item: item.name):
            entries_seen += 1
            if entries_seen > MAX_STRUCTURED_DISCOVERY_ENTRIES:
                return [], {}, False
            if clock() > deadline:
                return [], {}, False
            relative = (
                PurePosixPath(entry.name)
                if relative_dir == PurePosixPath(".")
                else relative_dir / entry.name
            )
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if entry.name in _SKIP_DIRS or (
                        entry.name.startswith(".") and entry.name != ".aisop"
                    ):
                        continue
                    child_directories.append((Path(entry.path), relative, depth + 1))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
            except OSError:
                return [], {}, False

            relative_path = relative.as_posix()
            if not _is_candidate(relative_path):
                continue
            if len(candidates) >= MAX_STRUCTURED_CANDIDATES:
                return [], {}, False
            try:
                size = entry.stat(follow_symlinks=False).st_size
            except OSError:
                return [], {}, False
            if size > MAX_STRUCTURED_DOCUMENT_BYTES:
                return [], {}, False
            if total_bytes + size > MAX_STRUCTURED_TOTAL_INPUT_BYTES:
                return [], {}, False
            try:
                with _open_regular_file_no_follow(Path(entry.path)) as source:
                    data = source.read(MAX_STRUCTURED_DOCUMENT_BYTES + 1)
            except (OSError, _FileOpenError, _UnsafeFileError):
                return [], {}, False
            if len(data) > MAX_STRUCTURED_DOCUMENT_BYTES:
                return [], {}, False
            total_bytes += len(data)
            candidates.append(relative_path)
            raw_cache[relative_path] = data

        # Push in reverse lexical order so the next visited directory is stable.
        stack.extend(reversed(child_directories))

    return candidates, raw_cache, True


def extract_structured_skill_context(skill_dir: Path) -> dict[str, object] | None:
    """Compatibility wrapper using bounded discovery and bounded no-follow reads."""
    clock = time.monotonic
    deadline = clock() + MAX_STRUCTURED_RUNTIME_SECONDS
    candidates, raw_cache, complete = _bounded_filesystem_cache(
        skill_dir,
        clock=clock,
        deadline=deadline,
    )
    if not complete:
        return None
    return extract_structured_skill_context_from_cache(
        skill_dir,
        candidates,
        raw_file_cache=raw_cache,
        clock=clock,
        deadline=deadline,
    ).context
