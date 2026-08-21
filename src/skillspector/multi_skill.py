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

"""Resource-bounded multi-skill directory detection.

Multi-skill discovery runs before the normal bundle inventory is built. It
therefore needs its own explicit resource profile; otherwise an attacker can
make ``--recursive`` spend unbounded memory merely by adding directory
entries, or make the display-name parser consume an unbounded manifest.
"""

from __future__ import annotations

import json
import os
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from skillspector.input_handler import (
    _FileOpenError,
    _open_regular_file_no_follow,
    _UnsafeFileError,
    validate_local_input_path,
)
from skillspector.logging_config import get_logger
from skillspector.structured_skill import (
    _SKIP_DIRS,
    MAX_STRUCTURED_DOCUMENT_BYTES,
    _is_candidate,
    extract_structured_skill_context_from_cache,
)

logger = get_logger(__name__)


# Discovery keeps, at most, one directory-local list of this size for stable
# lexical ordering. The global entry ceiling includes structured-bundle
# subtrees and prevents many individually-small directories from evading it.
MAX_MULTI_SKILL_DIRECTORY_ENTRIES = 1_024
MAX_MULTI_SKILL_DISCOVERY_ENTRIES = 10_000
MAX_MULTI_SKILL_TRAVERSAL_DEPTH = 64
MAX_MULTI_SKILL_STRUCTURED_CANDIDATES = 1_024
MAX_MULTI_SKILL_STRUCTURED_TOTAL_BYTES = 16 * 1024 * 1024
MAX_MULTI_SKILL_RUNTIME_SECONDS = 2.0

# Only a bounded prefix is needed to obtain the optional display name. The
# scan itself parses this manifest again from the bundle's bounded raw cache.
MAX_MULTI_SKILL_MANIFEST_FRONTMATTER_BYTES = 256 * 1024
MAX_MULTI_SKILL_NAME_CHARACTERS = 256


@dataclass(frozen=True, slots=True)
class SkillDirectory:
    """A detected skill within a multi-skill directory."""

    path: Path
    name: str
    relative_path: str


@dataclass(frozen=True, slots=True)
class MultiSkillDetectionLimitation:
    """Sanitized accounting for a discovery operation that did not complete."""

    reason_code: str
    resource: str
    observed_artifacts: int | None = None
    limit_artifacts: int | None = None
    observed_bytes: int | None = None
    limit_bytes: int | None = None
    observed_depth: int | None = None
    limit_depth: int | None = None
    observed_seconds: float | None = None
    limit_seconds: float | None = None
    observed_characters: int | None = None
    limit_characters: int | None = None

    def as_ledger_metadata(self) -> dict[str, object]:
        """Return inspection-ledger-compatible data without attacker paths."""
        result: dict[str, object] = {
            "outcome": "partial",
            "record_type": "system",
            "phase": "multi_skill_discovery",
            "path": ".",
            "reason_code": self.reason_code,
            "resource": self.resource,
        }
        for name in (
            "observed_artifacts",
            "limit_artifacts",
            "observed_bytes",
            "limit_bytes",
            "observed_depth",
            "limit_depth",
            "observed_seconds",
            "limit_seconds",
            "observed_characters",
            "limit_characters",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass
class MultiSkillDetectionResult:
    """Result of scanning a directory for multiple skills."""

    is_multi_skill: bool
    skills: list[SkillDirectory] = field(default_factory=list)
    has_root_skill: bool = False
    limitations: tuple[MultiSkillDetectionLimitation, ...] = ()
    entries_examined: int = 0
    structured_candidates_examined: int = 0
    structured_input_bytes_examined: int = 0

    @property
    def complete(self) -> bool:
        """Whether every applicable directory entry was classified."""
        return not self.limitations


class _DetectionIncompleteError(Exception):
    """Internal control flow carrying one sanitized resource limitation."""

    def __init__(self, limitation: MultiSkillDetectionLimitation) -> None:
        super().__init__(limitation.reason_code)
        self.limitation = limitation


@dataclass
class _DetectionBudget:
    """Aggregate resource accounting shared by all candidate directories."""

    started_at: float
    deadline: float
    clock: Callable[[], float]
    entries: int = 0
    structured_candidates: int = 0
    structured_bytes: int = 0

    def check_runtime(self) -> None:
        """Stop before starting more work once the shared deadline expires."""
        now = self.clock()
        if now >= self.deadline:
            raise _DetectionIncompleteError(
                MultiSkillDetectionLimitation(
                    reason_code="runtime_limit",
                    resource="multi_skill_runtime",
                    observed_seconds=max(0.0, now - self.started_at),
                    limit_seconds=MAX_MULTI_SKILL_RUNTIME_SECONDS,
                )
            )

    def consume_entry(self) -> None:
        """Charge one filesystem entry to the aggregate discovery ceiling."""
        self.check_runtime()
        self.entries += 1
        if self.entries > MAX_MULTI_SKILL_DISCOVERY_ENTRIES:
            raise _DetectionIncompleteError(
                MultiSkillDetectionLimitation(
                    reason_code="artifact_count_limit",
                    resource="multi_skill_discovery_entries",
                    observed_artifacts=self.entries,
                    limit_artifacts=MAX_MULTI_SKILL_DISCOVERY_ENTRIES,
                )
            )

    def consume_structured_candidate(self) -> None:
        """Charge one AISOP/AISP candidate across the entire invocation."""
        self.structured_candidates += 1
        if self.structured_candidates > MAX_MULTI_SKILL_STRUCTURED_CANDIDATES:
            raise _DetectionIncompleteError(
                MultiSkillDetectionLimitation(
                    reason_code="artifact_count_limit",
                    resource="multi_skill_structured_candidates",
                    observed_artifacts=self.structured_candidates,
                    limit_artifacts=MAX_MULTI_SKILL_STRUCTURED_CANDIDATES,
                )
            )

    def consume_structured_bytes(self, amount: int) -> None:
        """Charge bytes read only for structured-skill classification."""
        self.structured_bytes += amount
        if self.structured_bytes > MAX_MULTI_SKILL_STRUCTURED_TOTAL_BYTES:
            raise _DetectionIncompleteError(
                MultiSkillDetectionLimitation(
                    reason_code="total_bytes_limit",
                    resource="multi_skill_structured_bytes",
                    observed_bytes=self.structured_bytes,
                    limit_bytes=MAX_MULTI_SKILL_STRUCTURED_TOTAL_BYTES,
                )
            )


def _incomplete_result(
    limitation: MultiSkillDetectionLimitation,
    *,
    budget: _DetectionBudget | None = None,
) -> MultiSkillDetectionResult:
    """Discard arbitrary partial classifications and return a fail-closed result."""
    return MultiSkillDetectionResult(
        is_multi_skill=False,
        skills=[],
        has_root_skill=False,
        limitations=(limitation,),
        entries_examined=budget.entries if budget is not None else 0,
        structured_candidates_examined=(budget.structured_candidates if budget is not None else 0),
        structured_input_bytes_examined=budget.structured_bytes if budget is not None else 0,
    )


def detect_skills(directory: Path) -> MultiSkillDetectionResult:
    """Detect immediate child skills using bounded, deterministic discovery.

    A directory is considered multi-skill when it has no root ``SKILL.md`` and
    at least two immediate child directories contain a manifest or supported
    structured skill bundle. Any discovery limit or filesystem ambiguity
    discards all partial classifications and returns ``complete == False``.
    Callers can then fall back to a bounded monolithic scan and propagate the
    supplied limitation to their public completeness surfaces.
    """
    absolute_directory = Path(os.path.abspath(directory))
    try:
        directory = validate_local_input_path(absolute_directory)
    except (OSError, ValueError):
        return _incomplete_result(
            MultiSkillDetectionLimitation(
                reason_code="read_error",
                resource="multi_skill_input_path",
            )
        )
    try:
        root_stat = directory.stat(follow_symlinks=False)
    except FileNotFoundError:
        return MultiSkillDetectionResult(is_multi_skill=False)
    except OSError:
        return _incomplete_result(
            MultiSkillDetectionLimitation(
                reason_code="read_error",
                resource="multi_skill_input_path",
            )
        )
    if not stat.S_ISDIR(root_stat.st_mode):
        return MultiSkillDetectionResult(is_multi_skill=False)

    clock = time.monotonic
    started_at = clock()
    budget = _DetectionBudget(
        started_at=started_at,
        deadline=started_at + MAX_MULTI_SKILL_RUNTIME_SECONDS,
        clock=clock,
    )
    try:
        has_root = _has_skill_md(directory, budget=budget)
        if has_root:
            return MultiSkillDetectionResult(is_multi_skill=False, has_root_skill=True)

        skills: list[SkillDirectory] = []
        for entry in _bounded_scandir(directory, budget=budget):
            budget.check_runtime()
            child = Path(entry.path)
            try:
                if entry.is_symlink() or _is_link_or_junction(child):
                    continue
                if not entry.is_dir(follow_symlinks=False):
                    continue
            except OSError as exc:
                raise _read_error("multi_skill_directory_entry") from exc
            if entry.name in _SKIP_DIRS:
                continue

            has_manifest = _has_skill_md(child, budget=budget)
            is_structured = False
            if not has_manifest:
                is_structured = _is_structured_skill_bundle(child, budget=budget)
            if not (has_manifest or is_structured):
                continue

            name = _sanitize_display_component(child.name)
            if has_manifest:
                name = _extract_skill_name(child, budget=budget)
            skills.append(
                SkillDirectory(
                    path=child,
                    name=name,
                    relative_path=_sanitize_display_component(entry.name),
                )
            )
    except _DetectionIncompleteError as exc:
        return _incomplete_result(exc.limitation, budget=budget)

    return MultiSkillDetectionResult(
        is_multi_skill=len(skills) >= 2,
        skills=skills,
        has_root_skill=False,
        entries_examined=budget.entries,
        structured_candidates_examined=budget.structured_candidates,
        structured_input_bytes_examined=budget.structured_bytes,
    )


def _read_error(resource: str) -> _DetectionIncompleteError:
    """Build a sanitized filesystem-failure signal."""
    return _DetectionIncompleteError(
        MultiSkillDetectionLimitation(
            reason_code="read_error",
            resource=resource,
        )
    )


def _bounded_scandir(
    directory: Path,
    *,
    budget: _DetectionBudget,
) -> list[os.DirEntry[str]]:
    """Collect at most one bounded directory and return it in lexical order."""
    entries: list[os.DirEntry[str]] = []
    try:
        with os.scandir(directory) as scanner:
            for entry in scanner:
                budget.consume_entry()
                entries.append(entry)
                if len(entries) > MAX_MULTI_SKILL_DIRECTORY_ENTRIES:
                    raise _DetectionIncompleteError(
                        MultiSkillDetectionLimitation(
                            reason_code="artifact_count_limit",
                            resource="multi_skill_directory_entries",
                            observed_artifacts=len(entries),
                            limit_artifacts=MAX_MULTI_SKILL_DIRECTORY_ENTRIES,
                        )
                    )
    except _DetectionIncompleteError:
        raise
    except OSError as exc:
        raise _read_error("multi_skill_directory_entries") from exc
    budget.check_runtime()
    return sorted(entries, key=lambda item: item.name)


def _is_structured_skill_bundle(child_dir: Path, *, budget: _DetectionBudget) -> bool:
    """Classify an AISOP/AISP child from an aggregate-bounded local cache."""
    component_paths, raw_cache = _structured_candidate_cache(child_dir, budget=budget)
    result = extract_structured_skill_context_from_cache(
        child_dir,
        component_paths,
        raw_file_cache=raw_cache,
        clock=budget.clock,
        deadline=budget.deadline,
    )
    if result.limitations:
        limitation = result.limitations[0]
        raise _DetectionIncompleteError(
            MultiSkillDetectionLimitation(
                reason_code=limitation.reason_code,
                resource="multi_skill_structured_extraction",
                observed_artifacts=limitation.observed_artifacts,
                limit_artifacts=limitation.limit_artifacts,
                observed_bytes=limitation.observed_bytes,
                limit_bytes=limitation.limit_bytes,
                observed_depth=limitation.observed_depth,
                limit_depth=limitation.limit_depth,
                observed_seconds=limitation.observed_seconds,
                limit_seconds=limitation.limit_seconds,
            )
        )
    budget.check_runtime()
    return result.context is not None


def _structured_candidate_cache(
    skill_dir: Path,
    *,
    budget: _DetectionBudget,
) -> tuple[list[str], dict[str, bytes]]:
    """Build a small no-follow cache for structured classification."""
    candidates: list[str] = []
    raw_cache: dict[str, bytes] = {}
    stack: list[tuple[Path, PurePosixPath, int]] = [(skill_dir, PurePosixPath("."), 0)]

    while stack:
        budget.check_runtime()
        directory, relative_dir, depth = stack.pop()
        if depth > MAX_MULTI_SKILL_TRAVERSAL_DEPTH:
            raise _DetectionIncompleteError(
                MultiSkillDetectionLimitation(
                    reason_code="traversal_depth_limit",
                    resource="multi_skill_structured_depth",
                    observed_depth=depth,
                    limit_depth=MAX_MULTI_SKILL_TRAVERSAL_DEPTH,
                )
            )

        child_directories: list[tuple[Path, PurePosixPath, int]] = []
        for entry in _bounded_scandir(directory, budget=budget):
            path = Path(entry.path)
            relative = (
                PurePosixPath(entry.name)
                if relative_dir == PurePosixPath(".")
                else relative_dir / entry.name
            )
            try:
                if entry.is_symlink() or _is_link_or_junction(path):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if entry.name in _SKIP_DIRS or (
                        entry.name.startswith(".") and entry.name != ".aisop"
                    ):
                        continue
                    child_directories.append((path, relative, depth + 1))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
            except OSError as exc:
                raise _read_error("multi_skill_structured_entry") from exc

            relative_path = relative.as_posix()
            if not _is_candidate(relative_path):
                continue
            budget.consume_structured_candidate()
            try:
                size = entry.stat(follow_symlinks=False).st_size
            except OSError as exc:
                raise _read_error("multi_skill_structured_metadata") from exc
            if size > MAX_STRUCTURED_DOCUMENT_BYTES:
                raise _DetectionIncompleteError(
                    MultiSkillDetectionLimitation(
                        reason_code="size_limit",
                        resource="multi_skill_structured_document_bytes",
                        observed_bytes=size,
                        limit_bytes=MAX_STRUCTURED_DOCUMENT_BYTES,
                    )
                )
            try:
                with _open_regular_file_no_follow(path) as source:
                    data = source.read(MAX_STRUCTURED_DOCUMENT_BYTES + 1)
            except (OSError, _FileOpenError, _UnsafeFileError) as exc:
                raise _read_error("multi_skill_structured_content") from exc
            if len(data) > MAX_STRUCTURED_DOCUMENT_BYTES:
                raise _DetectionIncompleteError(
                    MultiSkillDetectionLimitation(
                        reason_code="size_limit",
                        resource="multi_skill_structured_document_bytes",
                        observed_bytes=len(data),
                        limit_bytes=MAX_STRUCTURED_DOCUMENT_BYTES,
                    )
                )
            budget.consume_structured_bytes(len(data))
            candidates.append(relative_path)
            raw_cache[relative_path] = data

        # Reversed push makes the next visited directory lexically smallest.
        stack.extend(reversed(child_directories))

    return candidates, raw_cache


def _manifest_file(directory: Path) -> Path | None:
    """Return a regular manifest path without following links or junctions."""
    for name in ("SKILL.md", "skill.md"):
        path = directory / name
        try:
            path_stat = path.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise _read_error("multi_skill_manifest_metadata") from exc
        if stat.S_ISLNK(path_stat.st_mode) or _is_link_or_junction(path):
            continue
        if stat.S_ISREG(path_stat.st_mode):
            return path
    return None


def _has_skill_md(directory: Path, *, budget: _DetectionBudget) -> bool:
    """Check for a root manifest using constant, no-follow metadata work."""
    budget.check_runtime()
    return _manifest_file(directory) is not None


def _is_link_or_junction(path: Path) -> bool:
    """Return true for links; metadata ambiguity is an incomplete detection."""
    try:
        return path.is_symlink() or path.is_junction()
    except OSError as exc:
        raise _read_error("multi_skill_path_metadata") from exc


def _extract_skill_name(skill_dir: Path, *, budget: _DetectionBudget) -> str:
    """Read only bounded frontmatter and parse a strict scalar ``name`` value."""
    fallback = _sanitize_display_component(skill_dir.name)
    path = _manifest_file(skill_dir)
    if path is None:
        return fallback
    try:
        budget.check_runtime()
        with _open_regular_file_no_follow(path) as source:
            observed = source.read(MAX_MULTI_SKILL_MANIFEST_FRONTMATTER_BYTES + 1)
        budget.check_runtime()
    except _DetectionIncompleteError:
        raise
    except (OSError, _FileOpenError, _UnsafeFileError) as exc:
        raise _read_error("multi_skill_manifest_content") from exc

    prefix = observed[:MAX_MULTI_SKILL_MANIFEST_FRONTMATTER_BYTES]
    if not prefix.startswith(b"---"):
        return fallback
    content = prefix.decode("utf-8", errors="replace")

    lines = content.splitlines()
    closing_index: int | None = None
    for index, line in enumerate(lines[1:], 1):
        budget.check_runtime()
        if line.strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        if len(observed) > MAX_MULTI_SKILL_MANIFEST_FRONTMATTER_BYTES:
            raise _DetectionIncompleteError(
                MultiSkillDetectionLimitation(
                    reason_code="manifest_parse_limit",
                    resource="multi_skill_manifest_bytes",
                    observed_bytes=len(observed),
                    limit_bytes=MAX_MULTI_SKILL_MANIFEST_FRONTMATTER_BYTES,
                )
            )
        return fallback

    for line in lines[1:closing_index]:
        budget.check_runtime()
        # A top-level key must begin in column zero. This deliberately avoids
        # YAML construction, aliases, tags, merge keys, and container values.
        if not line.startswith("name:"):
            continue
        candidate = _parse_strict_name_scalar(line[len("name:") :])
        if candidate is None:
            return fallback
        if len(candidate) > MAX_MULTI_SKILL_NAME_CHARACTERS:
            raise _DetectionIncompleteError(
                MultiSkillDetectionLimitation(
                    reason_code="output_limit",
                    resource="multi_skill_name_characters",
                    observed_characters=len(candidate),
                    limit_characters=MAX_MULTI_SKILL_NAME_CHARACTERS,
                )
            )
        return _sanitize_display_component(candidate)
    return fallback


def _parse_strict_name_scalar(raw_value: str) -> str | None:
    """Parse only plain or wholly quoted scalar text, never general YAML."""
    value = raw_value.strip()
    if not value:
        return None
    if value.startswith('"'):
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, RecursionError):
            return None
        return decoded if isinstance(decoded, str) else None
    if value.startswith("'"):
        if len(value) < 2 or not value.endswith("'"):
            return None
        return value[1:-1].replace("''", "'")
    if value[0] in "[{&*!|>@`'\"" or value.endswith(("]", "}")):
        return None
    comment = value.find(" #")
    if comment >= 0:
        value = value[:comment].rstrip()
    return value or None


def _sanitize_display_component(value: str) -> str:
    """Neutralize control/Rich-markup characters in console-facing names."""
    sanitized = "".join(
        character if character.isalnum() or character in {"-", "_", ".", " "} else "_"
        for character in value[:MAX_MULTI_SKILL_NAME_CHARACTERS]
    ).strip()
    return sanitized or "skill"
