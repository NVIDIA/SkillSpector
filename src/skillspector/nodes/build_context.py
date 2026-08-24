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

"""Build-context node for Skillspector workflow.

Builds flat ScanContext fields (components, file_cache, manifest, etc.)
from a local skill directory.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from stat import S_ISREG
from time import monotonic
from typing import cast

import yaml

from skillspector.artifacts import (
    ArtifactDisposition,
    ArtifactRecord,
    ContentKind,
    classify_artifact,
    decode_text,
)
from skillspector.constants import MAX_ANALYZABLE_FILE_BYTES, MAX_FILE_BYTES, build_model_config
from skillspector.input_handler import (
    _FileOpenError,
    _open_regular_file_no_follow,
    _UnsafeFileError,
    validate_local_input_path,
)
from skillspector.inspection_ledger import (
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    LedgerRecordType,
    ledger_event,
)
from skillspector.logging_config import get_logger
from skillspector.nested_artifacts import (
    inspect_nested_artifacts,
    is_executable_content,
)
from skillspector.python_ast import prewarm_python_ast_cache
from skillspector.references import (
    MAX_ACCEPTED_REFERENCES,
    MAX_RAW_REFERENCE_CANDIDATES,
    MAX_REFERENCE_RECORDS,
    MAX_REFERENCE_SOURCE_BYTES,
    ReferenceResolutionResult,
    resolve_bundle_references_with_metadata,
)
from skillspector.state import (
    SkillspectorState,
    ensure_workflow_resource_budget,
    transitive_note_truncation,
    transitive_record_artifacts,
    transitive_remaining_artifacts,
    transitive_remaining_bytes,
    transitive_remaining_seconds,
    transitive_traversal_state,
)
from skillspector.structured_skill import extract_structured_skill_context_from_cache

logger = get_logger(__name__)

# Directories to skip when walking
_SKIP_DIRS = frozenset({"__pycache__", "node_modules", ".venv", "venv", ".tox", ".pytest_cache"})

# Bundle-wide bounds complement the per-artifact read and analyzer limits.  A
# limit hit is always recorded as partial coverage; it is never treated as a
# clean scan of the subset accumulated before the bound.
MAX_DISCOVERED_ARTIFACTS = 10_000
MAX_DIRECTORY_ENTRIES = 10_000
MAX_BUNDLE_TRAVERSAL_DEPTH = 64
MAX_TOTAL_CACHED_BYTES = 64 * 1024 * 1024
MAX_BUNDLE_DISCOVERY_SECONDS = 30.0
MAX_BUNDLE_CACHE_SECONDS = 60.0
MAX_BUNDLE_LEDGER_EVENTS = 10_000
MAX_MANIFEST_FRONTMATTER_BYTES = 256 * 1024
MAX_MANIFEST_YAML_NODES = 10_000
MAX_MANIFEST_YAML_DEPTH = 64
MAX_MANIFEST_PARSE_SECONDS = 1.0
MAX_MANIFEST_OUTPUT_RECORDS = 1_024
MAX_MANIFEST_OUTPUT_CHARACTERS = 256 * 1024

# File type by extension
_FILE_TYPES: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".py": "python",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".txt": "text",
    ".js": "javascript",
    ".ts": "typescript",
    ".rb": "ruby",
    ".go": "go",
    ".rs": "rust",
}
_OMS_SIGNATURE_PATH = "skill.oms.sig"
_SIGSTORE_BUNDLE_MEDIA_TYPE = "application/vnd.dev.sigstore.bundle.v0.3+json"
_IN_TOTO_PAYLOAD_TYPE = "application/vnd.in-toto+json"
_IN_TOTO_STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
_OMS_PREDICATE_TYPE_PREFIX = "https://model_signing/signature/"


def _resolve_skill_dir(state: SkillspectorState) -> Path:
    """Resolve state skill_path to an existing directory Path."""
    skill_path = state.get("skill_path")
    if not skill_path or not isinstance(skill_path, str) or not skill_path.strip():
        raise ValueError("skill_path is required; provide input_path or skill_path to scan")
    try:
        resolved = validate_local_input_path(Path(skill_path))
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid skill_path: {skill_path}") from e
    if not resolved.is_dir():
        raise ValueError(f"Invalid skill_path: {skill_path} is not an existing directory")
    return cast(Path, resolved)


def _selected_baseline_component(
    state: SkillspectorState,
    skill_dir: Path,
    inventoried_components: list[str],
) -> str | None:
    """Return the selected baseline's component path when it is inside the skill.

    The CLI records the exact path selected by ``scan --baseline`` or targeted
    by ``baseline -o``. Excluding only that file prevents a rule's own sensitive
    message glob from producing a fresh finding (or entering regenerated
    fingerprints) while leaving every sibling YAML/JSON file in normal scope.
    """
    raw_path = state.get("baseline_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    baseline_path = Path(raw_path)
    candidates: list[Path] = [baseline_path]
    try:
        resolved = baseline_path.resolve()
    except (OSError, RuntimeError):
        resolved = None
    if resolved is not None and resolved != baseline_path:
        candidates.append(resolved)

    inventory = frozenset(inventoried_components)
    for candidate in candidates:
        try:
            relative = candidate.relative_to(skill_dir).as_posix()
        except ValueError:
            continue
        if relative in inventory:
            return relative
    return None


def _is_symlink(path: Path) -> bool:
    """Return whether *path* is a link or junction without masking later stat errors."""
    try:
        return path.is_symlink() or path.is_junction()
    except OSError:
        return False


def _resolves_outside(path: Path, root: Path) -> bool:
    """Return whether *path* resolves outside an already-resolved *root*."""
    try:
        return not path.resolve(strict=False).is_relative_to(root)
    except OSError:
        return False


def _append_bounded_ledger_event(
    events: list[InspectionLedgerEvent], event: InspectionLedgerEvent
) -> bool:
    """Append one event without allowing bundle-level ledger output to grow unbounded."""
    limit = max(1, MAX_BUNDLE_LEDGER_EVENTS)
    if len(events) < limit:
        events.append(event)
        return True
    if events[-1].get("reason_code") != LedgerReason.OUTPUT_LIMIT:
        events[-1] = ledger_event(
            outcome=LedgerOutcome.PARTIAL,
            record_type=LedgerRecordType.SYSTEM,
            phase=str(event["phase"]),
            path=str(event["path"]),
            reason=LedgerReason.OUTPUT_LIMIT,
            observed_records=limit + 1,
            limit_records=limit,
        )
    return False


def _bounded_ledger_output(
    events: list[InspectionLedgerEvent],
) -> list[InspectionLedgerEvent]:
    """Return a deterministic capped ledger projection with explicit truncation evidence."""
    limit = max(1, MAX_BUNDLE_LEDGER_EVENTS)
    if len(events) <= limit:
        return events
    overflow = events[limit]
    return [
        *events[: limit - 1],
        ledger_event(
            outcome=LedgerOutcome.PARTIAL,
            record_type=LedgerRecordType.SYSTEM,
            phase=str(overflow["phase"]),
            path=str(overflow["path"]),
            reason=LedgerReason.OUTPUT_LIMIT,
            observed_records=len(events),
            limit_records=limit,
        ),
    ]


def _discovery_scope_path(relative_root: Path, dirnames: list[str], filenames: list[str]) -> str:
    """Choose a deterministic report-safe path for a bundle discovery limit."""
    if filenames:
        return (relative_root / filenames[0]).as_posix()
    if dirnames:
        return f"{(relative_root / dirnames[0]).as_posix()}/"
    if relative_root.parts:
        return f"{relative_root.as_posix()}/"
    return "SKILL.md"


def _read_text_no_follow(path: Path, *, max_bytes: int | None = None) -> str:
    """Read a regular file without following symlinks at open time."""
    with _open_regular_file_no_follow(path) as source:
        data = source.read() if max_bytes is None else source.read(max_bytes + 1)
        return cast(bytes, data).decode("utf-8", errors="replace")


def _read_bytes_no_follow(path: Path, *, max_bytes: int | None = None) -> bytes:
    """Read a regular file as canonical bytes without following symlinks."""
    with _open_regular_file_no_follow(path) as source:
        return cast(bytes, source.read() if max_bytes is None else source.read(max_bytes))


def _walk_skill_files(
    skill_dir: Path,
    state: SkillspectorState | None = None,
) -> tuple[list[str], list[InspectionLedgerEvent]]:
    """Walk skill files and record scan-scope exclusions.

    Skips profile-permitted generated trees and symlinks. Hidden artifacts are
    inventoried normally. Within ``.git`` only configuration and active hooks
    are inspected; sample hooks and object/history storage remain outside the
    bounded scope.
    """
    paths: list[str] = []
    exclusions: list[InspectionLedgerEvent] = []
    skill_root = skill_dir.resolve(strict=False)
    started = monotonic()
    initial_shared_seconds = transitive_remaining_seconds(state) if state is not None else None
    discovery_runtime_limit = min(
        MAX_BUNDLE_DISCOVERY_SECONDS,
        max(0.0, initial_shared_seconds)
        if initial_shared_seconds is not None
        else MAX_BUNDLE_DISCOVERY_SECONDS,
    )
    # A directory counts as discovery work too. This prevents a directory-only
    # tree from bypassing the artifact-count ceiling.
    discovered_entries = 0
    stack: list[tuple[Path, Path]] = [(skill_dir, Path())]

    def _elapsed() -> float:
        return monotonic() - started

    def _scope(relative_root: Path) -> str:
        return f"{relative_root.as_posix()}/" if relative_root.parts else "SKILL.md"

    def _record_runtime_limit(
        relative_root: Path,
        *,
        elapsed: float,
        shared_seconds: float | None,
        affected_path: str | None = None,
    ) -> None:
        """Record one canonical discovery deadline boundary."""
        scope = affected_path or _scope(relative_root)
        if state is not None and shared_seconds is not None and shared_seconds <= 0:
            transitive_note_truncation(state, f"time budget exhausted during discovery at {scope}")
        _append_bounded_ledger_event(
            exclusions,
            ledger_event(
                outcome=LedgerOutcome.PARTIAL,
                record_type=LedgerRecordType.SYSTEM,
                phase="discovery",
                path=scope,
                reason=LedgerReason.RUNTIME_LIMIT,
                observed_seconds=max(0.0, elapsed),
                limit_seconds=discovery_runtime_limit,
            ),
        )

    while stack:
        root_path, relative_root = stack.pop()
        elapsed = _elapsed()
        shared_seconds = transitive_remaining_seconds(state) if state is not None else None
        if elapsed >= MAX_BUNDLE_DISCOVERY_SECONDS or (
            shared_seconds is not None and shared_seconds <= 0
        ):
            _record_runtime_limit(
                relative_root,
                elapsed=elapsed,
                shared_seconds=shared_seconds,
            )
            break

        # scandir is lazy. Keep only a bounded directory-local list, and do not
        # sort any attacker-controlled collection until that ceiling is known
        # to hold. If the directory itself exceeds the ceiling, none of its
        # nondeterministic enumeration prefix is retained.
        entries: list[tuple[str, bool, bool]] = []
        directory_overflow = False
        shared_artifacts = transitive_remaining_artifacts(state) if state is not None else None
        directory_entry_limit = (
            MAX_DIRECTORY_ENTRIES
            if shared_artifacts is None
            else min(MAX_DIRECTORY_ENTRIES, max(0, shared_artifacts))
        )
        try:
            with os.scandir(root_path) as iterator:
                for entry in iterator:
                    elapsed = _elapsed()
                    shared_seconds = (
                        transitive_remaining_seconds(state) if state is not None else None
                    )
                    if elapsed >= MAX_BUNDLE_DISCOVERY_SECONDS or (
                        shared_seconds is not None and shared_seconds <= 0
                    ):
                        _record_runtime_limit(
                            relative_root,
                            elapsed=elapsed,
                            shared_seconds=shared_seconds,
                        )
                        directory_overflow = True
                        break
                    if len(entries) >= directory_entry_limit:
                        if state is not None and shared_artifacts is not None:
                            transitive_note_truncation(
                                state,
                                f"artifact budget exhausted during discovery at {_scope(relative_root)}",
                            )
                        _append_bounded_ledger_event(
                            exclusions,
                            ledger_event(
                                outcome=LedgerOutcome.PARTIAL,
                                record_type=LedgerRecordType.SYSTEM,
                                phase="discovery",
                                path=_scope(relative_root),
                                reason=LedgerReason.ARTIFACT_COUNT_LIMIT,
                                observed_artifacts=len(entries) + 1,
                                limit_artifacts=directory_entry_limit,
                            ),
                        )
                        directory_overflow = True
                        break
                    try:
                        is_link = entry.is_symlink()
                        is_directory = entry.is_dir(follow_symlinks=False)
                    except OSError:
                        is_link = False
                        is_directory = False
                    entries.append((entry.name, is_directory, is_link))
        except OSError as exc:
            _append_bounded_ledger_event(
                exclusions,
                ledger_event(
                    outcome=LedgerOutcome.FAILED,
                    record_type=LedgerRecordType.SYSTEM,
                    phase="discovery",
                    path=_scope(relative_root),
                    reason=LedgerReason.READ_ERROR,
                    error_class=type(exc).__name__,
                ),
            )
            continue
        if directory_overflow:
            break

        child_directories: list[tuple[Path, Path]] = []
        normalized_root = relative_root.as_posix()
        sorted_entries = sorted(entries, key=lambda item: item[0])
        elapsed = _elapsed()
        shared_seconds = transitive_remaining_seconds(state) if state is not None else None
        if elapsed >= MAX_BUNDLE_DISCOVERY_SECONDS or (
            shared_seconds is not None and shared_seconds <= 0
        ):
            _record_runtime_limit(
                relative_root,
                elapsed=elapsed,
                shared_seconds=shared_seconds,
            )
            break

        for name, is_directory, is_link in sorted_entries:
            relative_path_obj = relative_root / name
            relative_path = relative_path_obj.as_posix()
            affected_path = f"{relative_path}/" if is_directory else relative_path

            elapsed = _elapsed()
            shared_seconds = transitive_remaining_seconds(state) if state is not None else None
            if elapsed >= MAX_BUNDLE_DISCOVERY_SECONDS or (
                shared_seconds is not None and shared_seconds <= 0
            ):
                _record_runtime_limit(
                    relative_root,
                    elapsed=elapsed,
                    shared_seconds=shared_seconds,
                    affected_path=affected_path,
                )
                return sorted(paths), exclusions

            shared_artifacts = transitive_remaining_artifacts(state) if state is not None else None
            if discovered_entries >= MAX_DISCOVERED_ARTIFACTS or (
                shared_artifacts is not None and shared_artifacts <= 0
            ):
                if state is not None and shared_artifacts is not None and shared_artifacts <= 0:
                    transitive_note_truncation(
                        state, f"artifact budget exhausted before discovering {relative_path}"
                    )
                _append_bounded_ledger_event(
                    exclusions,
                    ledger_event(
                        outcome=LedgerOutcome.PARTIAL,
                        record_type=LedgerRecordType.SYSTEM,
                        phase="discovery",
                        path=f"{relative_path}/" if is_directory else relative_path,
                        reason=LedgerReason.ARTIFACT_COUNT_LIMIT,
                        observed_artifacts=discovered_entries + 1,
                        limit_artifacts=min(
                            MAX_DISCOVERED_ARTIFACTS,
                            discovered_entries + max(0, shared_artifacts)
                            if shared_artifacts is not None
                            else MAX_DISCOVERED_ARTIFACTS,
                        ),
                    ),
                )
                return sorted(paths), exclusions
            discovered_entries += 1
            if state is not None:
                transitive_record_artifacts(state, 1)

            full = root_path / name
            unsafe_path = is_link or _is_symlink(full) or _resolves_outside(full, skill_root)
            elapsed = _elapsed()
            shared_seconds = transitive_remaining_seconds(state) if state is not None else None
            if elapsed >= MAX_BUNDLE_DISCOVERY_SECONDS or (
                shared_seconds is not None and shared_seconds <= 0
            ):
                _record_runtime_limit(
                    relative_root,
                    elapsed=elapsed,
                    shared_seconds=shared_seconds,
                    affected_path=affected_path,
                )
                return sorted(paths), exclusions
            if unsafe_path:
                _append_bounded_ledger_event(
                    exclusions,
                    ledger_event(
                        outcome=LedgerOutcome.OUT_OF_SCOPE,
                        record_type=LedgerRecordType.SCOPE_BOUNDARY,
                        phase="discovery",
                        path=f"{relative_path}/" if is_directory else relative_path,
                        reason=LedgerReason.NOT_REGULAR_FILE,
                    ),
                )
                continue

            if normalized_root == ".git" and (
                is_directory and name != "hooks" or not is_directory and name != "config"
            ):
                _append_bounded_ledger_event(
                    exclusions,
                    ledger_event(
                        outcome=LedgerOutcome.OUT_OF_SCOPE,
                        record_type=LedgerRecordType.SCOPE_BOUNDARY,
                        phase="discovery",
                        path=f"{relative_path}/" if is_directory else relative_path,
                        reason=LedgerReason.VCS_METADATA,
                    ),
                )
                continue
            if normalized_root == ".git/hooks" and (is_directory or name.endswith(".sample")):
                _append_bounded_ledger_event(
                    exclusions,
                    ledger_event(
                        outcome=LedgerOutcome.OUT_OF_SCOPE,
                        record_type=LedgerRecordType.SCOPE_BOUNDARY,
                        phase="discovery",
                        path=f"{relative_path}/" if is_directory else relative_path,
                        reason=LedgerReason.VCS_METADATA,
                    ),
                )
                continue

            if is_directory:
                if name in _SKIP_DIRS:
                    _append_bounded_ledger_event(
                        exclusions,
                        ledger_event(
                            outcome=LedgerOutcome.OUT_OF_SCOPE,
                            record_type=LedgerRecordType.SCOPE_BOUNDARY,
                            phase="discovery",
                            path=f"{relative_path}/",
                            reason=LedgerReason.EXCLUDED_DIRECTORY,
                        ),
                    )
                    continue
                depth = len(relative_path_obj.parts)
                if depth > MAX_BUNDLE_TRAVERSAL_DEPTH:
                    _append_bounded_ledger_event(
                        exclusions,
                        ledger_event(
                            outcome=LedgerOutcome.PARTIAL,
                            record_type=LedgerRecordType.SYSTEM,
                            phase="discovery",
                            path=f"{relative_path}/",
                            reason=LedgerReason.TRAVERSAL_DEPTH_LIMIT,
                            observed_depth=depth,
                            limit_depth=MAX_BUNDLE_TRAVERSAL_DEPTH,
                        ),
                    )
                    continue
                child_directories.append((full, relative_path_obj))
                continue

            # Other non-regular entries remain inventoried so the cache phase
            # records their exact failure disposition.
            paths.append(relative_path)

        # Reverse push gives a stable lexical depth-first traversal.
        stack.extend(reversed(child_directories))

    return sorted(paths), exclusions


def _infer_file_type(path: str) -> str:
    """Infer file type from path (extension)."""
    idx = path.rfind(".")
    suffix = path[idx:].lower() if idx >= 0 else ""
    return _FILE_TYPES.get(suffix, "other")


def _is_hidden_component(path: str) -> bool:
    """Return whether any segment of a relative component path is hidden."""
    return any(part.startswith(".") for part in path.replace("\\", "/").split("/") if part)


def _decode_base64_json(value: object) -> dict[str, object] | None:
    """Decode a strict base64 JSON object, returning ``None`` on malformed input."""
    if not isinstance(value, str) or not value:
        return None
    try:
        decoded = base64.b64decode(value, validate=True)
        parsed = json.loads(decoded.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _is_valid_oms_signature_bytes(data: bytes) -> bool:
    """Recognize the minimal OMS DSSE/in-toto structure from bounded bytes."""
    try:
        if len(data) > MAX_FILE_BYTES:
            return False
        bundle = json.loads(decode_text(data))
    except json.JSONDecodeError:
        return False

    if not isinstance(bundle, dict):
        return False
    if bundle.get("mediaType") != _SIGSTORE_BUNDLE_MEDIA_TYPE:
        return False
    if not isinstance(bundle.get("verificationMaterial"), dict):
        return False

    envelope = bundle.get("dsseEnvelope")
    if not isinstance(envelope, dict):
        return False
    if envelope.get("payloadType") != _IN_TOTO_PAYLOAD_TYPE:
        return False

    signatures = envelope.get("signatures")
    if not isinstance(signatures, list) or len(signatures) != 1:
        return False
    signature = signatures[0]
    if not isinstance(signature, dict):
        return False
    signature_bytes = signature.get("sig")
    if not isinstance(signature_bytes, str) or not signature_bytes:
        return False
    try:
        base64.b64decode(signature_bytes, validate=True)
    except (binascii.Error, ValueError):
        return False

    statement = _decode_base64_json(envelope.get("payload"))
    predicate_type = statement.get("predicateType") if statement else None
    return bool(
        statement
        and statement.get("_type") == _IN_TOTO_STATEMENT_TYPE
        and isinstance(predicate_type, str)
        and predicate_type.startswith(_OMS_PREDICATE_TYPE_PREFIX)
    )


def _is_valid_oms_signature(file_path: Path) -> bool:
    """Compatibility wrapper using one bounded, no-follow read."""
    try:
        if file_path.stat().st_size > MAX_FILE_BYTES:
            return False
        data = _read_bytes_no_follow(file_path, max_bytes=MAX_FILE_BYTES + 1)
    except (OSError, _FileOpenError, _UnsafeFileError):
        return False
    return _is_valid_oms_signature_bytes(data)


def _count_lines(file_path: Path) -> int:
    """Count lines in a file, handling binary and errors gracefully."""
    try:
        content = _read_text_no_follow(file_path, max_bytes=MAX_FILE_BYTES)
        return len(content.splitlines())
    except (OSError, _FileOpenError, _UnsafeFileError):
        logger.debug("Could not read file for line count: %s", file_path)
        return 0


def _build_component_metadata(
    skill_dir: Path,
    components: list[str],
    file_cache: dict[str, str],
    recognized_oms_signatures: frozenset[str] = frozenset(),
    *,
    clock: Callable[[], float] = monotonic,
    started_at: float | None = None,
    deadline: float | None = None,
    runtime_limitations: list[tuple[str, float]] | None = None,
) -> tuple[list[dict[str, object]], bool]:
    """Build component_metadata list and has_executable_scripts from paths."""
    metadata: list[dict[str, object]] = []
    has_executable = False
    effective_started_at = clock() if started_at is None else started_at

    def _expired(path: str) -> bool:
        if deadline is None:
            return False
        now = clock()
        if now < deadline:
            return False
        if runtime_limitations is not None and not runtime_limitations:
            runtime_limitations.append((path, max(0.0, now - effective_started_at)))
        return True

    for path in components:
        if _expired(path):
            break
        full = skill_dir / path
        file_type = "oms_signature" if path in recognized_oms_signatures else _infer_file_type(path)
        content = file_cache.get(path)
        lines = (
            len(content.splitlines())
            if content is not None
            else _count_lines(full)
            if path in recognized_oms_signatures
            else 0
        )
        try:
            file_stat = full.stat()
            size_bytes = file_stat.st_size
            mode = file_stat.st_mode
        except OSError:
            logger.debug("Could not stat file: %s", path)
            size_bytes = 0
            mode = 0
        data = content.encode("utf-8", errors="replace") if content is not None else b""
        executable = is_executable_content(path, data, mode)
        if executable:
            has_executable = True
        component: dict[str, object] = {
            "path": path,
            "type": file_type,
            "lines": lines,
            "executable": executable,
            "size_bytes": size_bytes,
        }
        if _is_hidden_component(path):
            component["hidden"] = True
            component["local_only"] = True
            if executable:
                component.update(
                    {
                        "outer_path": path,
                        "nested_path": path,
                        "container_type": "filesystem",
                        "container_ancestry": ["filesystem"],
                        "container_depth": 0,
                        "outer_hidden": True,
                        "concealed_executable": True,
                        "concealment_reasons": ["hidden_artifact"],
                    }
                )
        metadata.append(component)
        if _expired(path):
            break
    return metadata, has_executable


def _redact_for_external_model(path: str, content: str) -> str:
    """Redact values from local environment files before external-model use."""
    name = Path(path).name.lower()
    if name != ".env" and not name.startswith(".env."):
        return content
    lines: list[str] = []
    for line in content.splitlines(keepends=True):
        match = re.match(r"^(\s*(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*\s*=)(.*?)(\r?\n)?$", line)
        if match:
            lines.append(f"{match.group(1)}<redacted>{match.group(3) or ''}")
        else:
            lines.append(line)
    return "".join(lines)


def _is_hidden_path(path: str) -> bool:
    """Return whether any bundle path segment is hidden."""
    return any(part.startswith(".") for part in Path(path).parts)


def _opaque_artifact_record(
    path: str,
    *,
    disposition: ArtifactDisposition,
    reason: LedgerReason,
    referenced: bool,
    size_bytes: int = 0,
) -> ArtifactRecord:
    """Return an explicit inventory row for content that was not cached."""
    return {
        "path": path,
        "content_kind": ContentKind.OPAQUE,
        "disposition": disposition,
        "size_bytes": max(0, size_bytes),
        "decodable": False,
        "contains_nul": False,
        "misleading_extension": False,
        "referenced": referenced,
        "reason": reason.value,
    }


def _read_file_cache(
    skill_dir: Path,
    components: list[str],
    referenced_paths: frozenset[str] = frozenset(),
    *,
    started_at: float | None = None,
    state: SkillspectorState | None = None,
) -> tuple[
    dict[str, str],
    dict[str, bytes],
    dict[str, str],
    list[ArtifactRecord],
    list[InspectionLedgerEvent],
]:
    """Build canonical byte/text caches, inventory rows, and cache-failure events."""
    file_cache: dict[str, str] = {}
    raw_file_cache: dict[str, bytes] = {}
    llm_file_cache: dict[str, str] = {}
    inventory: list[ArtifactRecord] = []
    ledger_events: list[InspectionLedgerEvent] = []
    skill_root = skill_dir.resolve(strict=False)
    traversal = transitive_traversal_state(state) if state is not None else None
    started = monotonic() if started_at is None else started_at
    initial_shared_seconds = transitive_remaining_seconds(state) if state is not None else None
    cache_runtime_limit = min(
        MAX_BUNDLE_CACHE_SECONDS,
        max(0.0, initial_shared_seconds)
        if initial_shared_seconds is not None
        else MAX_BUNDLE_CACHE_SECONDS,
    )
    total_cached_bytes = 0

    def _record_cache_runtime_limit(
        path: str,
        component_index: int,
        *,
        action: str,
        current_size: int = 0,
    ) -> bool:
        """Record an expired cache deadline and its deterministic affected suffix."""
        elapsed = monotonic() - started
        remaining_seconds = transitive_remaining_seconds(state) if state is not None else None
        if elapsed < MAX_BUNDLE_CACHE_SECONDS and not (
            remaining_seconds is not None and remaining_seconds <= 0
        ):
            return False
        if state is not None and remaining_seconds is not None and remaining_seconds <= 0:
            transitive_note_truncation(state, f"time budget exhausted {action} {path}")
        ledger_events.append(
            ledger_event(
                outcome=LedgerOutcome.PARTIAL,
                record_type=LedgerRecordType.SYSTEM,
                phase="cache",
                path=path,
                reason=LedgerReason.RUNTIME_LIMIT,
                observed_seconds=max(0.0, elapsed),
                limit_seconds=cache_runtime_limit,
            )
        )
        inventory.append(
            _opaque_artifact_record(
                path,
                disposition=ArtifactDisposition.PARTIAL,
                reason=LedgerReason.RUNTIME_LIMIT,
                referenced=path in referenced_paths,
                size_bytes=current_size,
            )
        )
        inventory.extend(
            _opaque_artifact_record(
                omitted,
                disposition=ArtifactDisposition.PARTIAL,
                reason=LedgerReason.RUNTIME_LIMIT,
                referenced=omitted in referenced_paths,
            )
            for omitted in components[component_index + 1 :]
        )
        return True

    for component_index, path in enumerate(components):
        if _record_cache_runtime_limit(
            path,
            component_index,
            action="before reading",
        ):
            break
        local_remaining_bytes = MAX_TOTAL_CACHED_BYTES - total_cached_bytes
        shared_remaining_bytes = transitive_remaining_bytes(state) if state is not None else None
        remaining_bundle_bytes = (
            local_remaining_bytes
            if shared_remaining_bytes is None
            else min(local_remaining_bytes, shared_remaining_bytes)
        )
        effective_total_limit = total_cached_bytes + max(0, remaining_bundle_bytes)
        if remaining_bundle_bytes <= 0:
            if (
                state is not None
                and shared_remaining_bytes is not None
                and shared_remaining_bytes <= 0
            ):
                transitive_note_truncation(state, f"byte budget exhausted before reading {path}")
            ledger_events.append(
                ledger_event(
                    outcome=LedgerOutcome.PARTIAL,
                    record_type=LedgerRecordType.SYSTEM,
                    phase="cache",
                    path=path,
                    reason=LedgerReason.TOTAL_BYTES_LIMIT,
                    observed_bytes=total_cached_bytes + 1,
                    limit_bytes=min(MAX_TOTAL_CACHED_BYTES, total_cached_bytes),
                )
            )
            inventory.extend(
                _opaque_artifact_record(
                    omitted,
                    disposition=ArtifactDisposition.PARTIAL,
                    reason=LedgerReason.TOTAL_BYTES_LIMIT,
                    referenced=omitted in referenced_paths,
                )
                for omitted in components[component_index:]
            )
            break
        full = skill_dir / path
        unsafe_path = _is_symlink(full) or _resolves_outside(full, skill_root)
        if _record_cache_runtime_limit(
            path,
            component_index,
            action="while validating",
        ):
            break
        if unsafe_path:
            ledger_events.append(
                ledger_event(
                    outcome=LedgerOutcome.OUT_OF_SCOPE,
                    record_type=LedgerRecordType.SCOPE_BOUNDARY,
                    phase="cache",
                    path=path,
                    reason=LedgerReason.NOT_REGULAR_FILE,
                )
            )
            inventory.append(
                _opaque_artifact_record(
                    path,
                    disposition=ArtifactDisposition.OUT_OF_SCOPE,
                    reason=LedgerReason.NOT_REGULAR_FILE,
                    referenced=path in referenced_paths,
                )
            )
            continue
        try:
            file_stat = full.stat()
        except FileNotFoundError as exc:
            if _record_cache_runtime_limit(
                path,
                component_index,
                action="while stating",
            ):
                break
            ledger_events.append(
                ledger_event(
                    outcome=LedgerOutcome.FAILED,
                    record_type=LedgerRecordType.SYSTEM,
                    phase="cache",
                    path=path,
                    reason=LedgerReason.FILE_DISAPPEARED,
                    error_class=type(exc).__name__,
                )
            )
            inventory.append(
                _opaque_artifact_record(
                    path,
                    disposition=ArtifactDisposition.FAILED,
                    reason=LedgerReason.FILE_DISAPPEARED,
                    referenced=path in referenced_paths,
                )
            )
            continue
        except OSError as exc:
            if _record_cache_runtime_limit(
                path,
                component_index,
                action="while stating",
            ):
                break
            ledger_events.append(
                ledger_event(
                    outcome=LedgerOutcome.FAILED,
                    record_type=LedgerRecordType.SYSTEM,
                    phase="cache",
                    path=path,
                    reason=LedgerReason.STAT_ERROR,
                    error_class=type(exc).__name__,
                )
            )
            inventory.append(
                _opaque_artifact_record(
                    path,
                    disposition=ArtifactDisposition.FAILED,
                    reason=LedgerReason.STAT_ERROR,
                    referenced=path in referenced_paths,
                )
            )
            continue
        if _record_cache_runtime_limit(
            path,
            component_index,
            action="while stating",
            current_size=file_stat.st_size,
        ):
            break
        if not S_ISREG(file_stat.st_mode):
            ledger_events.append(
                ledger_event(
                    outcome=LedgerOutcome.FAILED,
                    record_type=LedgerRecordType.SYSTEM,
                    phase="cache",
                    path=path,
                    reason=LedgerReason.NOT_REGULAR_FILE,
                )
            )
            inventory.append(
                _opaque_artifact_record(
                    path,
                    disposition=ArtifactDisposition.FAILED,
                    reason=LedgerReason.NOT_REGULAR_FILE,
                    referenced=path in referenced_paths,
                    size_bytes=file_stat.st_size,
                )
            )
            continue
        try:
            # Always bound the post-stat read as well.  A file can grow between
            # stat and open, and a stale size must not turn this into an
            # unbounded allocation.
            per_file_limit = min(MAX_ANALYZABLE_FILE_BYTES, remaining_bundle_bytes)
            observed = _read_bytes_no_follow(full, max_bytes=per_file_limit + 1)
            per_file_truncated = (
                file_stat.st_size > MAX_ANALYZABLE_FILE_BYTES
                or len(observed) > MAX_ANALYZABLE_FILE_BYTES
            )
            aggregate_truncated = (
                file_stat.st_size > remaining_bundle_bytes or len(observed) > remaining_bundle_bytes
            )
            truncated = per_file_truncated or aggregate_truncated
            raw = observed[:per_file_limit]
            observed_total_bytes = total_cached_bytes + max(file_stat.st_size, len(observed))
            record_bytes = getattr(traversal, "record_bytes", None)
            if callable(record_bytes):
                record_bytes(len(raw))
            if _record_cache_runtime_limit(
                path,
                component_index,
                action="while reading",
                current_size=max(file_stat.st_size, len(observed)),
            ):
                break
            content = decode_text(raw)
            if _record_cache_runtime_limit(
                path,
                component_index,
                action="while decoding",
                current_size=max(file_stat.st_size, len(observed)),
            ):
                break
            artifact = classify_artifact(path, raw, referenced=path in referenced_paths)
            if _record_cache_runtime_limit(
                path,
                component_index,
                action="while classifying",
                current_size=max(file_stat.st_size, len(observed)),
            ):
                break
            total_cached_bytes += len(raw)
            raw_file_cache[path] = raw
            file_cache[path] = content
            if truncated:
                observed_size = max(file_stat.st_size, len(observed))
                artifact["size_bytes"] = observed_size
                artifact["disposition"] = ArtifactDisposition.PARTIAL
                artifact["reason"] = "total_bytes_limit" if aggregate_truncated else "size_limit"
                if per_file_truncated:
                    ledger_events.append(
                        ledger_event(
                            outcome=LedgerOutcome.PARTIAL,
                            record_type=LedgerRecordType.SYSTEM,
                            phase="cache",
                            path=path,
                            reason=LedgerReason.SIZE_LIMIT,
                            observed_bytes=observed_size,
                            limit_bytes=MAX_ANALYZABLE_FILE_BYTES,
                        )
                    )
                if aggregate_truncated:
                    if (
                        state is not None
                        and shared_remaining_bytes is not None
                        and shared_remaining_bytes <= local_remaining_bytes
                    ):
                        transitive_note_truncation(
                            state, f"byte budget exhausted while reading {path}"
                        )
                    ledger_events.append(
                        ledger_event(
                            outcome=LedgerOutcome.PARTIAL,
                            record_type=LedgerRecordType.SYSTEM,
                            phase="cache",
                            path=path,
                            reason=LedgerReason.TOTAL_BYTES_LIMIT,
                            observed_bytes=observed_total_bytes,
                            limit_bytes=effective_total_limit,
                        )
                    )
            inventory.append(artifact)
            if not truncated and not _is_hidden_path(path) and artifact["content_kind"] == "text":
                llm_file_cache[path] = _redact_for_external_model(path, content)
            if aggregate_truncated:
                inventory.extend(
                    _opaque_artifact_record(
                        omitted,
                        disposition=ArtifactDisposition.PARTIAL,
                        reason=LedgerReason.TOTAL_BYTES_LIMIT,
                        referenced=omitted in referenced_paths,
                    )
                    for omitted in components[component_index + 1 :]
                )
                break
        except FileNotFoundError as exc:
            ledger_events.append(
                ledger_event(
                    outcome=LedgerOutcome.FAILED,
                    record_type=LedgerRecordType.SYSTEM,
                    phase="cache",
                    path=path,
                    reason=LedgerReason.FILE_DISAPPEARED,
                    error_class=type(exc).__name__,
                )
            )
            inventory.append(
                _opaque_artifact_record(
                    path,
                    disposition=ArtifactDisposition.FAILED,
                    reason=LedgerReason.FILE_DISAPPEARED,
                    referenced=path in referenced_paths,
                )
            )
        except _UnsafeFileError:
            ledger_events.append(
                ledger_event(
                    outcome=LedgerOutcome.OUT_OF_SCOPE,
                    record_type=LedgerRecordType.SCOPE_BOUNDARY,
                    phase="cache",
                    path=path,
                    reason=LedgerReason.NOT_REGULAR_FILE,
                )
            )
            inventory.append(
                _opaque_artifact_record(
                    path,
                    disposition=ArtifactDisposition.OUT_OF_SCOPE,
                    reason=LedgerReason.NOT_REGULAR_FILE,
                    referenced=path in referenced_paths,
                )
            )
        except _FileOpenError as exc:
            logger.debug("Could not read file: %s", path)
            ledger_events.append(
                ledger_event(
                    outcome=LedgerOutcome.FAILED,
                    record_type=LedgerRecordType.SYSTEM,
                    phase="cache",
                    path=path,
                    reason=LedgerReason.READ_ERROR,
                    error_class=exc.error_class,
                )
            )
            inventory.append(
                _opaque_artifact_record(
                    path,
                    disposition=ArtifactDisposition.FAILED,
                    reason=LedgerReason.READ_ERROR,
                    referenced=path in referenced_paths,
                    size_bytes=file_stat.st_size,
                )
            )
        except OSError as exc:
            logger.debug("Could not read file: %s", path)
            ledger_events.append(
                ledger_event(
                    outcome=LedgerOutcome.FAILED,
                    record_type=LedgerRecordType.SYSTEM,
                    phase="cache",
                    path=path,
                    reason=LedgerReason.READ_ERROR,
                    error_class=type(exc).__name__,
                )
            )
            inventory.append(
                _opaque_artifact_record(
                    path,
                    disposition=ArtifactDisposition.FAILED,
                    reason=LedgerReason.READ_ERROR,
                    referenced=path in referenced_paths,
                    size_bytes=file_stat.st_size,
                )
            )
    return file_cache, raw_file_cache, llm_file_cache, inventory, ledger_events


class _ManifestLimitError(yaml.YAMLError):
    """Internal signal that bounded YAML composition exhausted a resource."""

    def __init__(self, kind: str, observed: int | float) -> None:
        super().__init__(kind)
        self.kind = kind
        self.observed = observed


class _ManifestSchemaError(yaml.YAMLError):
    """Internal signal for unsupported manifest value shapes."""


class _BoundedManifestLoader(yaml.SafeLoader):
    """SafeLoader with explicit node, nesting, and elapsed-time ceilings."""

    def __init__(
        self,
        stream: str,
        *,
        clock: Callable[[], float] = monotonic,
        started_at: float | None = None,
        deadline: float | None = None,
    ) -> None:
        super().__init__(stream)
        self._manifest_clock = clock
        self._manifest_started = clock() if started_at is None else started_at
        self._manifest_deadline = (
            self._manifest_started + MAX_MANIFEST_PARSE_SECONDS if deadline is None else deadline
        )
        self._manifest_nodes = 0
        self._manifest_depth = 0

    def flatten_mapping(self, node: yaml.MappingNode) -> None:
        """Reject YAML merge keys before SafeConstructor can amplify aliases."""
        if any(key_node.tag == "tag:yaml.org,2002:merge" for key_node, _ in node.value):
            raise _ManifestSchemaError("merge_key")
        super().flatten_mapping(node)

    def compose_node(self, parent: object, index: object) -> yaml.Node:
        now = self._manifest_clock()
        elapsed = max(0.0, now - self._manifest_started)
        if now >= self._manifest_deadline:
            raise _ManifestLimitError("runtime", elapsed)
        self._manifest_nodes += 1
        if self._manifest_nodes > MAX_MANIFEST_YAML_NODES:
            raise _ManifestLimitError("nodes", self._manifest_nodes)
        self._manifest_depth += 1
        if self._manifest_depth > MAX_MANIFEST_YAML_DEPTH:
            self._manifest_depth -= 1
            raise _ManifestLimitError("depth", self._manifest_depth + 1)
        try:
            return super().compose_node(parent, index)
        finally:
            self._manifest_depth -= 1


def _validate_manifest_graph(
    value: object,
    *,
    started_at: float,
    deadline: float,
    clock: Callable[[], float] = monotonic,
) -> None:
    """Reject cyclic or unexpectedly complex constructed YAML object graphs."""
    active: set[int] = set()
    visited: set[int] = set()
    nodes = 0

    def _visit(item: object, depth: int) -> None:
        nonlocal nodes
        now = clock()
        elapsed = max(0.0, now - started_at)
        if now >= deadline:
            raise _ManifestLimitError("runtime", elapsed)
        nodes += 1
        if nodes > MAX_MANIFEST_YAML_NODES:
            raise _ManifestLimitError("nodes", nodes)
        if depth > MAX_MANIFEST_YAML_DEPTH:
            raise _ManifestLimitError("depth", depth)
        if not isinstance(item, (Mapping, list, tuple)):
            return
        identity = id(item)
        if identity in active:
            raise _ManifestLimitError("depth", MAX_MANIFEST_YAML_DEPTH + 1)
        if identity in visited:
            return
        active.add(identity)
        visited.add(identity)
        try:
            children = (
                (child for pair in item.items() for child in pair)
                if isinstance(item, Mapping)
                else iter(item)
            )
            for child in children:
                _visit(child, depth + 1)
        finally:
            active.remove(identity)

    _visit(value, 0)


def _record_manifest_limit(
    ledger_events: list[InspectionLedgerEvent] | None,
    *,
    path: str,
    kind: str,
    observed: int | float,
    runtime_limit: float = MAX_MANIFEST_PARSE_SECONDS,
) -> None:
    """Record one parser limitation without exposing manifest contents."""
    if ledger_events is None:
        return
    observed_characters: int | None = None
    limit_characters: int | None = None
    observed_bytes: int | None = None
    limit_bytes: int | None = None
    observed_records: int | None = None
    limit_records: int | None = None
    observed_depth: int | None = None
    limit_depth: int | None = None
    observed_seconds: float | None = None
    limit_seconds: float | None = None
    if kind == "bytes":
        observed_bytes = int(observed)
        limit_bytes = MAX_MANIFEST_FRONTMATTER_BYTES
    elif kind in {"nodes", "output_records"}:
        observed_records = int(observed)
        limit_records = MAX_MANIFEST_YAML_NODES if kind == "nodes" else MAX_MANIFEST_OUTPUT_RECORDS
    elif kind == "characters":
        observed_characters = int(observed)
        limit_characters = MAX_MANIFEST_OUTPUT_CHARACTERS
    elif kind == "depth":
        observed_depth = int(observed)
        limit_depth = MAX_MANIFEST_YAML_DEPTH
    else:
        observed_seconds = float(observed)
        limit_seconds = runtime_limit
    ledger_events.append(
        ledger_event(
            outcome=LedgerOutcome.PARTIAL,
            record_type=LedgerRecordType.SYSTEM,
            phase="manifest",
            path=path,
            reason=LedgerReason.MANIFEST_PARSE_LIMIT,
            observed_characters=observed_characters,
            limit_characters=limit_characters,
            observed_bytes=observed_bytes,
            limit_bytes=limit_bytes,
            observed_records=observed_records,
            limit_records=limit_records,
            observed_depth=observed_depth,
            limit_depth=limit_depth,
            observed_seconds=observed_seconds,
            limit_seconds=limit_seconds,
        )
    )


def _record_manifest_parse_error(
    ledger_events: list[InspectionLedgerEvent] | None,
    *,
    path: str,
    error_class: str | None = None,
) -> None:
    """Record malformed claimed frontmatter as incomplete analysis."""
    if ledger_events is None:
        return
    ledger_events.append(
        ledger_event(
            outcome=LedgerOutcome.PARTIAL,
            record_type=LedgerRecordType.SYSTEM,
            phase="manifest",
            path=path,
            reason=LedgerReason.MANIFEST_PARSE_ERROR,
            error_class=error_class,
        )
    )


def _project_manifest(
    data: Mapping[str, object],
    *,
    started_at: float,
    deadline: float,
    clock: Callable[[], float] = monotonic,
) -> dict[str, object]:
    """Project only the supported manifest schema under output/runtime ceilings."""
    records = 0
    characters = 0

    def _check() -> None:
        now = clock()
        elapsed = max(0.0, now - started_at)
        if now >= deadline:
            raise _ManifestLimitError("runtime", elapsed)

    def _consume(text: str = "") -> None:
        nonlocal records, characters
        _check()
        records += 1
        if records > MAX_MANIFEST_OUTPUT_RECORDS:
            raise _ManifestLimitError("output_records", records)
        characters += len(text)
        if characters > MAX_MANIFEST_OUTPUT_CHARACTERS:
            raise _ManifestLimitError("characters", characters)

    def _scalar_text(value: object) -> str:
        if not isinstance(value, (str, int, float, bool)):
            raise _ManifestSchemaError(type(value).__name__)
        text = str(value)
        _consume(text)
        return text

    def _clone_bounded(value: object, *, depth: int, active: set[int]) -> object:
        """Clone parameter metadata while charging every projected occurrence.

        YAML aliases share constructed Python objects.  Charging the output
        projection rather than unique object identities prevents a small
        alias graph from expanding into an unbounded returned manifest.
        """
        _check()
        if depth > MAX_MANIFEST_YAML_DEPTH:
            raise _ManifestLimitError("depth", depth)
        if value is None:
            _consume()
            return None
        if isinstance(value, (str, int, float, bool)):
            _consume(str(value))
            return value
        if not isinstance(value, (Mapping, list, tuple)):
            raise _ManifestSchemaError(type(value).__name__)
        identity = id(value)
        if identity in active:
            raise _ManifestLimitError("depth", MAX_MANIFEST_YAML_DEPTH + 1)
        active.add(identity)
        _consume()
        try:
            if isinstance(value, Mapping):
                cloned: dict[str, object] = {}
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise _ManifestSchemaError(type(key).__name__)
                    _consume(key)
                    cloned[key] = _clone_bounded(item, depth=depth + 1, active=active)
                return cloned
            return [_clone_bounded(item, depth=depth + 1, active=active) for item in value]
        finally:
            active.remove(identity)

    def _string_list(value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise _ManifestSchemaError(type(value).__name__)
        return [_scalar_text(item) for item in value]

    manifest: dict[str, object] = {}
    name = data.get("name")
    if name is not None:
        if not isinstance(name, str):
            raise _ManifestSchemaError(type(name).__name__)
        _consume(name)
        manifest["name"] = name
    description = data.get("description")
    if description is not None:
        if not isinstance(description, str):
            raise _ManifestSchemaError(type(description).__name__)
        _consume(description)
        manifest["description"] = description

    manifest["triggers"] = _string_list(data.get("triggers", []))
    manifest["permissions"] = _string_list(data.get("permissions", []))
    allowed_tools = data.get("allowed-tools", [])
    if isinstance(allowed_tools, str):
        tools: list[str] = []
        cursor = 0
        while cursor <= len(allowed_tools):
            _check()
            separator = allowed_tools.find(",", cursor)
            if separator < 0:
                separator = len(allowed_tools)
            item = allowed_tools[cursor:separator].strip()
            if item:
                tools.append(_scalar_text(item))
            if separator == len(allowed_tools):
                break
            cursor = separator + 1
        manifest["allowed-tools"] = tools
    elif isinstance(allowed_tools, list):
        manifest["allowed-tools"] = [
            item.strip() for item in _string_list(allowed_tools) if item.strip()
        ]
    else:
        # Preserve the established compatibility behavior for malformed
        # scalar declarations without converting arbitrary containers.
        manifest["allowed-tools"] = []

    raw_parameters = data.get("parameters", [])
    if not isinstance(raw_parameters, list):
        raw_parameters = []
    parameters: list[dict[str, object]] = []
    for raw_parameter in raw_parameters:
        if not isinstance(raw_parameter, Mapping):
            continue
        parameter = _clone_bounded(raw_parameter, depth=0, active=set())
        if not isinstance(parameter, dict):  # defensive narrowing
            raise _ManifestSchemaError(type(parameter).__name__)
        parameters.append(parameter)
    manifest["parameters"] = parameters
    _check()
    return manifest


def _parse_manifest(
    skill_dir: Path,
    *,
    raw_file_cache: Mapping[str, bytes] | None = None,
    ledger_events: list[InspectionLedgerEvent] | None = None,
    clock: Callable[[], float] = monotonic,
    deadline: float | None = None,
) -> dict[str, object]:
    """Parse SKILL.md or skill.md YAML frontmatter into a manifest dict.

    Returns dict with name, description, triggers (list), permissions (list),
    allowed-tools (list), parameters (list). Returns {} if no file or parse fails.
    Parsing is restricted to a bounded byte prefix, including for direct helper
    callers that do not provide the bundle's already-bounded raw cache.
    """
    started_at = clock()
    local_deadline = started_at + MAX_MANIFEST_PARSE_SECONDS
    effective_deadline = local_deadline if deadline is None else min(local_deadline, deadline)
    runtime_limit = max(0.0, effective_deadline - started_at)

    def _check_runtime() -> None:
        now = clock()
        if now >= effective_deadline:
            raise _ManifestLimitError("runtime", max(0.0, now - started_at))

    skill_root = skill_dir.resolve(strict=False)
    for name in ("SKILL.md", "skill.md"):
        path = skill_dir / name
        try:
            _check_runtime()
            if raw_file_cache is not None:
                # The cache is the canonical, no-follow snapshot selected by
                # bounded discovery.  Do not consult the mutable filesystem
                # again: doing so would let a post-cache removal or symlink
                # swap suppress otherwise valid cached frontmatter.
                cached = raw_file_cache.get(name)
                if cached is None:
                    continue
                observed = cached[: MAX_MANIFEST_FRONTMATTER_BYTES + 1]
            else:
                if _is_symlink(path) or _resolves_outside(path, skill_root) or not path.is_file():
                    continue
                observed = _read_bytes_no_follow(path, max_bytes=MAX_MANIFEST_FRONTMATTER_BYTES + 1)
            _check_runtime()
        except _ManifestLimitError as exc:
            _record_manifest_limit(
                ledger_events,
                path=name,
                kind=exc.kind,
                observed=exc.observed,
                runtime_limit=runtime_limit,
            )
            return {}
        except (OSError, _FileOpenError, _UnsafeFileError):
            logger.debug("Could not read manifest file: %s", name)
            return {}
        truncated = len(observed) > MAX_MANIFEST_FRONTMATTER_BYTES
        try:
            _check_runtime()
            content = decode_text(observed[:MAX_MANIFEST_FRONTMATTER_BYTES])
            _check_runtime()
        except _ManifestLimitError as exc:
            _record_manifest_limit(
                ledger_events,
                path=name,
                kind=exc.kind,
                observed=exc.observed,
                runtime_limit=runtime_limit,
            )
            return {}
        if not content.startswith("---"):
            return {}
        end_match = re.search(r"\n---\s*\n", content[3:])
        try:
            _check_runtime()
        except _ManifestLimitError as exc:
            _record_manifest_limit(
                ledger_events,
                path=name,
                kind=exc.kind,
                observed=exc.observed,
                runtime_limit=runtime_limit,
            )
            return {}
        if not end_match:
            if truncated:
                _record_manifest_limit(
                    ledger_events,
                    path=name,
                    kind="bytes",
                    observed=len(observed),
                )
            else:
                _record_manifest_parse_error(ledger_events, path=name)
            return {}
        frontmatter = content[3 : end_match.start() + 3]
        loader: _BoundedManifestLoader | None = None
        try:
            _check_runtime()
            loader = _BoundedManifestLoader(
                frontmatter,
                clock=clock,
                started_at=started_at,
                deadline=effective_deadline,
            )
            try:
                data = loader.get_single_data()
            except (ValueError, OverflowError, KeyError, AttributeError, IndexError) as exc:
                # SafeLoader's bounded scalar constructors may still reject a
                # syntactically valid scalar during conversion (for example,
                # Python's integer digit ceiling).  Treat only these expected
                # conversion failures as malformed manifest input; unrelated
                # exceptions remain visible to callers.
                logger.debug("Manifest scalar conversion failed for %s", name)
                _record_manifest_parse_error(
                    ledger_events,
                    path=name,
                    error_class=type(exc).__name__,
                )
                return {}
            _validate_manifest_graph(
                data,
                started_at=started_at,
                deadline=effective_deadline,
                clock=clock,
            )
        except _ManifestLimitError as exc:
            _record_manifest_limit(
                ledger_events,
                path=name,
                kind=exc.kind,
                observed=exc.observed,
                runtime_limit=runtime_limit,
            )
            return {}
        except (yaml.YAMLError, RecursionError) as exc:
            logger.debug("Manifest parse failed for %s", name)
            _record_manifest_parse_error(
                ledger_events,
                path=name,
                error_class=type(exc).__name__,
            )
            return {}
        finally:
            if loader is not None:
                loader.dispose()
        if not isinstance(data, dict):
            _record_manifest_parse_error(ledger_events, path=name)
            return {}
        try:
            return _project_manifest(
                data,
                started_at=started_at,
                deadline=effective_deadline,
                clock=clock,
            )
        except _ManifestLimitError as exc:
            _record_manifest_limit(
                ledger_events,
                path=name,
                kind=exc.kind,
                observed=exc.observed,
                runtime_limit=runtime_limit,
            )
            return {}
        except _ManifestSchemaError as exc:
            _record_manifest_parse_error(
                ledger_events,
                path=name,
                error_class=type(exc).__name__,
            )
            return {}
    return {}


def build_context(state: SkillspectorState) -> dict[str, object]:
    """Build flat ScanContext fields from state skill_path (local directory).

    Resolves skill_path to a directory, walks files, builds file_cache
    and manifest. Returns only context keys; leaves findings untouched.
    Raises ValueError if skill_path is missing or not an existing directory.
    """
    # Start one graph-wide deadline before any discovery or preprocessing. A
    # transitive traversal supplied by the CLI is reused, preserving its
    # stricter cross-child byte/time allowances.
    workflow_budget = ensure_workflow_resource_budget(state)
    budgeted_state = dict(state)
    budgeted_state["workflow_resource_budget"] = workflow_budget
    state = cast(SkillspectorState, budgeted_state)

    skill_dir = _resolve_skill_dir(state)

    inventoried_components, discovery_events = _walk_skill_files(skill_dir, state)
    selected_baseline = _selected_baseline_component(state, skill_dir, inventoried_components)
    selected_baselines = frozenset({selected_baseline} if selected_baseline else set())
    cache_candidates = [path for path in inventoried_components if path not in selected_baselines]
    processing_started = monotonic()
    processing_deadline = processing_started + MAX_BUNDLE_CACHE_SECONDS
    shared_remaining_seconds = transitive_remaining_seconds(state)
    if shared_remaining_seconds is not None:
        processing_deadline = min(
            processing_deadline,
            processing_started + max(0.0, shared_remaining_seconds),
        )
    (
        ordinary_file_cache,
        raw_file_cache,
        llm_file_cache,
        artifact_inventory,
        cache_events,
    ) = _read_file_cache(
        skill_dir,
        cache_candidates,
        started_at=processing_started,
        state=state,
    )

    inventory_by_path = {item["path"]: item for item in artifact_inventory}
    prework_events: list[InspectionLedgerEvent] = []
    processing_runtime_limit = max(0.0, processing_deadline - processing_started)

    def _record_processing_runtime(*, phase: str, path: str, now: float) -> None:
        artifact = inventory_by_path.get(path)
        if artifact is not None and artifact.get("disposition") != ArtifactDisposition.FAILED:
            if artifact.get("disposition") != ArtifactDisposition.PARTIAL:
                artifact["reason"] = LedgerReason.RUNTIME_LIMIT.value
            artifact["disposition"] = ArtifactDisposition.PARTIAL
        prework_events.append(
            ledger_event(
                outcome=LedgerOutcome.PARTIAL,
                record_type=LedgerRecordType.SYSTEM,
                phase=phase,
                path=path,
                reason=LedgerReason.RUNTIME_LIMIT,
                observed_seconds=max(0.0, now - processing_started),
                limit_seconds=processing_runtime_limit,
            )
        )

    recognized_oms_signature_paths: set[str] = set()
    if _OMS_SIGNATURE_PATH in raw_file_cache:
        signature_started = monotonic()
        if signature_started >= processing_deadline:
            _record_processing_runtime(
                phase="signature_recognition",
                path=_OMS_SIGNATURE_PATH,
                now=signature_started,
            )
        else:
            signature_valid = _is_valid_oms_signature_bytes(raw_file_cache[_OMS_SIGNATURE_PATH])
            signature_finished = monotonic()
            if signature_finished >= processing_deadline:
                _record_processing_runtime(
                    phase="signature_recognition",
                    path=_OMS_SIGNATURE_PATH,
                    now=signature_finished,
                )
            elif signature_valid:
                recognized_oms_signature_paths.add(_OMS_SIGNATURE_PATH)
    recognized_oms_signatures = frozenset(recognized_oms_signature_paths)
    signature_events = [
        ledger_event(
            outcome=LedgerOutcome.OUT_OF_SCOPE,
            record_type=LedgerRecordType.SCOPE_BOUNDARY,
            phase="discovery",
            path=path,
            reason=LedgerReason.OMS_SIGNATURE,
        )
        for path in sorted(recognized_oms_signatures)
    ]
    baseline_events = [
        ledger_event(
            outcome=LedgerOutcome.OUT_OF_SCOPE,
            record_type=LedgerRecordType.SCOPE_BOUNDARY,
            phase="discovery",
            path=path,
            reason=LedgerReason.BASELINE_FILE,
        )
        for path in sorted(selected_baselines)
    ]
    for artifact in artifact_inventory:
        if artifact["path"] in recognized_oms_signatures:
            artifact["disposition"] = ArtifactDisposition.OUT_OF_SCOPE
            artifact["reason"] = LedgerReason.OMS_SIGNATURE.value
            llm_file_cache.pop(artifact["path"], None)

    primary_path = next(
        (path for path in ("SKILL.md", "skill.md") if path in inventoried_components), None
    )
    references = []
    reference_events: list[InspectionLedgerEvent] = []
    reference_resolution: dict[str, object] = {}
    inventory_by_path = {item["path"]: item for item in artifact_inventory}
    if primary_path is not None and primary_path in raw_file_cache:
        primary_raw = raw_file_cache[primary_path]
        reference_started = monotonic()
        if reference_started >= processing_deadline:
            resolution = ReferenceResolutionResult(
                records=[],
                complete=False,
                limitations=("runtime",),
                input_bytes_examined=0,
                raw_candidates_considered=0,
                accepted_references=0,
                runtime_seconds=0.0,
                runtime_seconds_limit=0.0,
            )
        else:
            primary_text = decode_text(primary_raw[: MAX_REFERENCE_SOURCE_BYTES + 1])
            reference_after_decode = monotonic()
            if reference_after_decode >= processing_deadline:
                resolution = ReferenceResolutionResult(
                    records=[],
                    complete=False,
                    limitations=("runtime",),
                    input_bytes_examined=0,
                    raw_candidates_considered=0,
                    accepted_references=0,
                    runtime_seconds=max(0.0, reference_after_decode - reference_started),
                    runtime_seconds_limit=max(0.0, processing_deadline - reference_started),
                )
            else:
                resolution = resolve_bundle_references_with_metadata(
                    skill_dir,
                    source_path=primary_path,
                    source_text=primary_text,
                    known_paths=inventoried_components,
                    clock=monotonic,
                    deadline=processing_deadline,
                )
        references = resolution.records
        primary_partial = (
            inventory_by_path.get(primary_path, {}).get("disposition")
            == ArtifactDisposition.PARTIAL
        )
        limitations = list(resolution.limitations)
        if primary_partial:
            limitations.append("source_partial")
        limitations = list(dict.fromkeys(limitations))
        reference_resolution = {
            "complete": resolution.complete and not primary_partial,
            "limitations": limitations,
            "input_bytes_examined": resolution.input_bytes_examined,
            "input_bytes_limit": MAX_REFERENCE_SOURCE_BYTES,
            "raw_candidates_considered": resolution.raw_candidates_considered,
            "raw_candidates_limit": MAX_RAW_REFERENCE_CANDIDATES,
            "accepted_references": resolution.accepted_references,
            "accepted_references_limit": MAX_ACCEPTED_REFERENCES,
            "output_records": len(resolution.records),
            "output_records_limit": MAX_REFERENCE_RECORDS,
            "runtime_seconds": resolution.runtime_seconds,
            "runtime_seconds_limit": resolution.runtime_seconds_limit,
        }
        if limitations:
            primary_artifact = inventory_by_path.get(primary_path)
            if (
                primary_artifact is not None
                and primary_artifact.get("disposition") != ArtifactDisposition.FAILED
            ):
                if primary_artifact.get("disposition") != ArtifactDisposition.PARTIAL:
                    primary_artifact["reason"] = LedgerReason.REFERENCE_EXTRACTION_LIMIT.value
                primary_artifact["disposition"] = ArtifactDisposition.PARTIAL
        for limitation in limitations:
            observed_bytes: int | None = None
            limit_bytes: int | None = None
            observed_artifacts: int | None = None
            limit_artifacts: int | None = None
            observed_records: int | None = None
            limit_records: int | None = None
            observed_seconds: float | None = None
            limit_seconds: float | None = None
            if limitation == "input_bytes":
                observed_bytes = resolution.input_bytes_examined
                limit_bytes = MAX_REFERENCE_SOURCE_BYTES
            elif limitation == "raw_candidates":
                observed_artifacts = resolution.raw_candidates_considered
                limit_artifacts = MAX_RAW_REFERENCE_CANDIDATES
            elif limitation == "accepted_references":
                observed_artifacts = resolution.accepted_references
                limit_artifacts = MAX_ACCEPTED_REFERENCES
            elif limitation == "output_records":
                observed_records = len(resolution.records)
                limit_records = MAX_REFERENCE_RECORDS
            elif limitation == "runtime":
                observed_seconds = resolution.runtime_seconds
                limit_seconds = resolution.runtime_seconds_limit
            reference_events.append(
                ledger_event(
                    outcome=LedgerOutcome.PARTIAL,
                    record_type=LedgerRecordType.SYSTEM,
                    phase="reference_resolution",
                    path=primary_path,
                    reason=LedgerReason.REFERENCE_EXTRACTION_LIMIT,
                    stage=limitation,
                    observed_bytes=observed_bytes,
                    limit_bytes=limit_bytes,
                    observed_artifacts=observed_artifacts,
                    limit_artifacts=limit_artifacts,
                    observed_records=observed_records,
                    limit_records=limit_records,
                    observed_seconds=observed_seconds,
                    limit_seconds=limit_seconds,
                )
            )
        reference_events.extend(
            ledger_event(
                outcome=LedgerOutcome.PARTIAL,
                record_type=LedgerRecordType.SYSTEM,
                phase="reference_resolution",
                path=primary_path,
                start_line=int(reference["line"]),
                end_line=int(reference["line"]),
                reason=LedgerReason.REFERENCE_UNRESOLVED,
            )
            for reference in references
            if reference["status"] in {"missing", "ambiguous"}
        )

    referenced_paths = frozenset(
        str(reference["target_path"])
        for reference in references
        if reference["status"] == "resolved" and reference["target_path"]
    )
    for artifact in artifact_inventory:
        if artifact["path"] in referenced_paths:
            artifact["referenced"] = True

    # Omitted paths remain represented in artifact_inventory, but are not fed
    # to analyzers without content. Genuine read failures remain analyzer work
    # so their fatal accounting is preserved.
    ordinary_components = [
        path
        for path in cache_candidates
        if path not in recognized_oms_signatures
        and (
            path in raw_file_cache
            or inventory_by_path.get(path, {}).get("disposition")
            in {ArtifactDisposition.FAILED, ArtifactDisposition.OUT_OF_SCOPE}
        )
    ]

    remaining_artifacts = max(0, MAX_DISCOVERED_ARTIFACTS - len(artifact_inventory))
    shared_remaining_artifacts = transitive_remaining_artifacts(state)
    if shared_remaining_artifacts is not None:
        remaining_artifacts = min(remaining_artifacts, max(0, shared_remaining_artifacts))
    remaining_bytes = max(
        0,
        MAX_TOTAL_CACHED_BYTES - sum(len(data) for data in raw_file_cache.values()),
    )
    shared_nested_bytes = transitive_remaining_bytes(state)
    if shared_nested_bytes is not None:
        remaining_bytes = min(remaining_bytes, max(0, shared_nested_bytes))
    shared_nested_seconds = transitive_remaining_seconds(state)
    nested_deadline = processing_deadline
    if shared_nested_seconds is not None:
        nested_deadline = min(
            nested_deadline,
            monotonic() + max(0.0, shared_nested_seconds),
        )
    nested = inspect_nested_artifacts(
        skill_dir,
        [path for path in ordinary_components if path in raw_file_cache],
        raw_file_cache=raw_file_cache,
        max_members=remaining_artifacts,
        max_uncompressed_bytes=remaining_bytes,
        absolute_deadline=nested_deadline,
        clock=monotonic,
    )
    traversal = transitive_traversal_state(state)
    record_bytes = getattr(traversal, "record_bytes", None)
    if callable(record_bytes):
        record_bytes(nested.uncompressed_bytes)
    if state is not None:
        transitive_record_artifacts(state, len(nested.artifact_inventory))
    nested_reasons = {event.get("reason_code") for event in nested.ledger_events}
    if shared_nested_bytes is not None and LedgerReason.ARCHIVE_SIZE_LIMIT in nested_reasons:
        transitive_note_truncation(state, "byte budget exhausted during nested inspection")
    if shared_nested_seconds is not None and LedgerReason.ARCHIVE_TIME_LIMIT in nested_reasons:
        transitive_note_truncation(state, "time budget exhausted during nested inspection")
    if (
        shared_remaining_artifacts is not None
        and LedgerReason.ARCHIVE_MEMBER_LIMIT in nested_reasons
    ):
        transitive_note_truncation(state, "artifact budget exhausted during nested inspection")
    local_file_cache = dict(ordinary_file_cache)
    local_file_cache.update(nested.file_cache)
    raw_file_cache.update(nested.raw_file_cache)
    artifact_inventory.extend(nested.artifact_inventory)
    for artifact in artifact_inventory:
        override = nested.inventory_overrides.get(artifact["path"])
        if override is not None:
            artifact["disposition"], artifact["reason"] = override
    inventory_by_path = {item["path"]: item for item in artifact_inventory}

    recognized_containers = frozenset(nested.outer_metadata)
    components = sorted(
        dict.fromkeys(
            [
                *ordinary_components,
                *nested.components,
            ]
        )
    )
    for path in [*recognized_containers, *recognized_oms_signatures]:
        llm_file_cache.pop(path, None)
    llm_components = sorted(llm_file_cache)
    file_cache = dict(llm_file_cache)

    manifest_events: list[InspectionLedgerEvent] = []
    manifest = _parse_manifest(
        skill_dir,
        raw_file_cache=raw_file_cache,
        ledger_events=manifest_events,
        clock=monotonic,
        deadline=processing_deadline,
    )
    if manifest_events and primary_path is not None:
        manifest_reason = manifest_events[-1].get("reason_code", LedgerReason.MANIFEST_PARSE_LIMIT)
        for artifact in artifact_inventory:
            if artifact["path"] == primary_path:
                artifact["disposition"] = ArtifactDisposition.PARTIAL
                artifact["reason"] = (
                    manifest_reason.value
                    if isinstance(manifest_reason, LedgerReason)
                    else str(manifest_reason)
                )
                break

    structured_candidates = sorted(dict.fromkeys([*cache_candidates, *nested.components]))
    structured = extract_structured_skill_context_from_cache(
        skill_dir,
        structured_candidates,
        raw_file_cache=raw_file_cache,
        file_cache=local_file_cache,
        clock=monotonic,
        deadline=processing_deadline,
    )
    structured_events: list[InspectionLedgerEvent] = []
    for structured_limitation in structured.limitations:
        reason = LedgerReason(structured_limitation.reason_code)
        affected_paths = {structured_limitation.path}
        if structured_limitation.resource == "structured_candidates":
            affected_paths.update(
                path for path in structured_candidates if path.lower().endswith(".aisop.json")
            )
        elif structured_limitation.resource in {
            "structured_total_input_bytes",
            "structured_nesting",
            "structured_nodes",
            "structured_output_records",
        }:
            affected_paths.update(
                path
                for path in structured_candidates
                if path >= structured_limitation.path and path.lower().endswith(".aisop.json")
            )
        for path in affected_paths:
            structured_artifact = inventory_by_path.get(path)
            if (
                structured_artifact is None
                or structured_artifact.get("disposition") == ArtifactDisposition.FAILED
            ):
                continue
            if structured_artifact.get("disposition") != ArtifactDisposition.PARTIAL:
                structured_artifact["reason"] = reason.value
            structured_artifact["disposition"] = ArtifactDisposition.PARTIAL
        structured_events.append(
            ledger_event(
                outcome=LedgerOutcome.PARTIAL,
                record_type=LedgerRecordType.SYSTEM,
                phase="structured_skill",
                path=structured_limitation.path,
                reason=reason,
                observed_bytes=structured_limitation.observed_bytes,
                limit_bytes=structured_limitation.limit_bytes,
                observed_artifacts=structured_limitation.observed_artifacts,
                limit_artifacts=structured_limitation.limit_artifacts,
                observed_depth=structured_limitation.observed_depth,
                limit_depth=structured_limitation.limit_depth,
                observed_records=structured_limitation.observed_records,
                limit_records=structured_limitation.limit_records,
                observed_seconds=structured_limitation.observed_seconds,
                limit_seconds=structured_limitation.limit_seconds,
            )
        )

    disposition_by_path = {item["path"]: item["disposition"] for item in artifact_inventory}
    for reference in references:
        target = reference["target_path"]
        if target and target in disposition_by_path:
            reference["disposition"] = disposition_by_path[target]

    postprocessing_events: list[InspectionLedgerEvent] = []
    runtime_limit = max(0.0, processing_deadline - processing_started)

    def _mark_runtime_partial(affected_paths: list[str], first_limited_path: str) -> None:
        limited = False
        for affected_path in affected_paths:
            if affected_path == first_limited_path:
                limited = True
            if not limited:
                continue
            affected_artifact = inventory_by_path.get(affected_path)
            if (
                affected_artifact is None
                or affected_artifact.get("disposition") == ArtifactDisposition.FAILED
            ):
                continue
            if affected_artifact.get("disposition") != ArtifactDisposition.PARTIAL:
                affected_artifact["reason"] = LedgerReason.RUNTIME_LIMIT.value
            affected_artifact["disposition"] = ArtifactDisposition.PARTIAL

    ast_runtime_limitations: list[tuple[str, float]] = []
    python_ast_cache_key = prewarm_python_ast_cache(
        components,
        local_file_cache,
        clock=monotonic,
        started_at=processing_started,
        deadline=processing_deadline,
        runtime_limitations=ast_runtime_limitations,
    )
    if ast_runtime_limitations:
        path, elapsed = ast_runtime_limitations[0]
        python_components = [
            component
            for component in components
            if component.lower().endswith(".py") and component in local_file_cache
        ]
        _mark_runtime_partial(python_components, path)
        postprocessing_events.append(
            ledger_event(
                outcome=LedgerOutcome.PARTIAL,
                record_type=LedgerRecordType.SYSTEM,
                phase="python_ast_prewarm",
                path=path,
                reason=LedgerReason.RUNTIME_LIMIT,
                observed_seconds=elapsed,
                limit_seconds=runtime_limit,
            )
        )
    metadata_components = [
        path for path in inventoried_components if path not in selected_baselines
    ]
    metadata_runtime_limitations: list[tuple[str, float]] = []
    component_metadata, has_executable_scripts = _build_component_metadata(
        skill_dir,
        metadata_components,
        local_file_cache,
        recognized_oms_signatures,
        clock=monotonic,
        started_at=processing_started,
        deadline=processing_deadline,
        runtime_limitations=metadata_runtime_limitations,
    )
    if metadata_runtime_limitations:
        path, elapsed = metadata_runtime_limitations[0]
        _mark_runtime_partial(metadata_components, path)
        postprocessing_events.append(
            ledger_event(
                outcome=LedgerOutcome.PARTIAL,
                record_type=LedgerRecordType.SYSTEM,
                phase="component_metadata",
                path=path,
                reason=LedgerReason.RUNTIME_LIMIT,
                observed_seconds=elapsed,
                limit_seconds=runtime_limit,
            )
        )
    for metadata in component_metadata:
        path = str(metadata.get("path", ""))
        if path in nested.outer_metadata:
            metadata.update(nested.outer_metadata[path])
            metadata["lines"] = 0
    component_metadata.extend(nested.metadata)
    has_executable_scripts = has_executable_scripts or any(
        bool(metadata.get("executable")) for metadata in nested.metadata
    )

    result: dict[str, object] = {
        "components": components,
        "llm_components": llm_components,
        "file_cache": file_cache,
        "local_file_cache": local_file_cache,
        "raw_file_cache": raw_file_cache,
        "llm_file_cache": llm_file_cache,
        "artifact_inventory": artifact_inventory,
        "artifact_references": references,
        "reference_resolution": reference_resolution,
        "inspection_ledger": _bounded_ledger_output(
            [
                *discovery_events,
                *prework_events,
                *signature_events,
                *baseline_events,
                *reference_events,
                *cache_events,
                *nested.ledger_events,
                *manifest_events,
                *structured_events,
                *postprocessing_events,
            ]
        ),
        "ast_cache": {},
        "python_ast_cache_key": python_ast_cache_key,
        "manifest": manifest,
        "previous_manifest": None,
        "model_config": build_model_config(),
        "component_metadata": component_metadata,
        "has_executable_scripts": has_executable_scripts,
        "workflow_resource_budget": workflow_budget,
    }

    if structured.context is not None:
        result["structured_skill_context"] = structured.context

    return result
