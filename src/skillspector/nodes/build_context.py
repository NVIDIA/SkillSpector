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
from pathlib import Path
from stat import S_ISREG

import yaml

from skillspector.constants import MAX_FILE_BYTES, build_model_config
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
from skillspector.python_ast import prewarm_python_ast_cache
from skillspector.state import SkillspectorState
from skillspector.structured_skill import extract_structured_skill_context

logger = get_logger(__name__)

# Directories to skip when walking
_SKIP_DIRS = frozenset(
    {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".pytest_cache"}
)

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
_EXECUTABLE_EXTENSIONS = frozenset(
    {".py", ".sh", ".bash", ".zsh", ".js", ".ts", ".rb", ".go", ".rs", ".pl"}
)

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
    return resolved


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


def _read_text_no_follow(path: Path) -> str:
    """Read a regular file without following symlinks at open time."""
    with _open_regular_file_no_follow(path) as source:
        return source.read().decode("utf-8", errors="replace")


def _walk_skill_files(
    skill_dir: Path,
) -> tuple[list[str], list[InspectionLedgerEvent]]:
    """Walk skill files and record scan-scope exclusions.

    Skips _SKIP_DIRS, hidden files except those starting with .claude, and
    symlinks, which must never supply content to remote LLM analyzers.
    """
    paths: list[str] = []
    exclusions: list[InspectionLedgerEvent] = []
    skill_root = skill_dir.resolve(strict=False)
    for root, dirnames, filenames in os.walk(skill_dir, followlinks=False):
        root_path = Path(root)
        dirnames.sort()
        filenames.sort()
        relative_root = root_path.relative_to(skill_dir)

        skipped_dirnames = [name for name in dirnames if name in _SKIP_DIRS]
        symlinked_dirnames = [name for name in dirnames if _is_symlink(root_path / name)]
        dirnames[:] = [
            name for name in dirnames if name not in _SKIP_DIRS and name not in symlinked_dirnames
        ]
        for dirname in skipped_dirnames:
            boundary = (relative_root / dirname).as_posix()
            exclusions.append(
                ledger_event(
                    outcome=LedgerOutcome.OUT_OF_SCOPE,
                    record_type=LedgerRecordType.SCOPE_BOUNDARY,
                    phase="discovery",
                    path=f"{boundary}/",
                    reason=LedgerReason.EXCLUDED_DIRECTORY,
                )
            )
        for dirname in symlinked_dirnames:
            boundary = (relative_root / dirname).as_posix()
            exclusions.append(
                ledger_event(
                    outcome=LedgerOutcome.OUT_OF_SCOPE,
                    record_type=LedgerRecordType.SCOPE_BOUNDARY,
                    phase="discovery",
                    path=f"{boundary}/",
                    reason=LedgerReason.NOT_REGULAR_FILE,
                )
            )

        for filename in filenames:
            relative_path = (relative_root / filename).as_posix()
            if filename.startswith(".") and not filename.startswith(".claude"):
                exclusions.append(
                    ledger_event(
                        outcome=LedgerOutcome.OUT_OF_SCOPE,
                        record_type=LedgerRecordType.SCOPE_BOUNDARY,
                        phase="discovery",
                        path=relative_path,
                        reason=LedgerReason.HIDDEN_FILE,
                    )
                )
                continue

            # Use forward slashes on every OS: these relative paths are dict keys
            # and SARIF/URI locations, so they must be portable.  Other
            # non-regular entries remain inventoried for cache-phase evidence;
            # symlinks are excluded before they can be read.
            full = root_path / filename
            if _is_symlink(full) or _resolves_outside(full, skill_root):
                exclusions.append(
                    ledger_event(
                        outcome=LedgerOutcome.OUT_OF_SCOPE,
                        record_type=LedgerRecordType.SCOPE_BOUNDARY,
                        phase="discovery",
                        path=relative_path,
                        reason=LedgerReason.NOT_REGULAR_FILE,
                    )
                )
                continue
            paths.append(relative_path)
    paths.sort()
    return paths, exclusions


def _infer_file_type(path: str) -> str:
    """Infer file type from path (extension)."""
    idx = path.rfind(".")
    suffix = path[idx:].lower() if idx >= 0 else ""
    return _FILE_TYPES.get(suffix, "other")


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


def _is_valid_oms_signature(file_path: Path) -> bool:
    """Recognize the minimal root-level OMS DSSE/in-toto signature structure.

    This intentionally does not parse verification material or verify the
    cryptographic signature. Its purpose is to distinguish detached OMS
    metadata from agent-facing content before analyzers inspect the skill.
    """
    try:
        if file_path.stat().st_size > MAX_FILE_BYTES:
            return False
        content = file_path.read_text(encoding="utf-8")
        bundle = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
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
    return bool(
        statement
        and statement.get("_type") == _IN_TOTO_STATEMENT_TYPE
        and isinstance(statement.get("predicateType"), str)
        and statement["predicateType"].startswith(_OMS_PREDICATE_TYPE_PREFIX)
    )


def _count_lines(file_path: Path) -> int:
    """Count lines in a file, handling binary and errors gracefully."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return len(content.splitlines())
    except OSError:
        logger.debug("Could not read file for line count: %s", file_path)
        return 0


def _build_component_metadata(
    skill_dir: Path,
    components: list[str],
    file_cache: dict[str, str],
    recognized_oms_signatures: frozenset[str] = frozenset(),
) -> tuple[list[dict[str, object]], bool]:
    """Build component_metadata list and has_executable_scripts from paths."""
    metadata: list[dict[str, object]] = []
    has_executable = False
    for path in components:
        full = skill_dir / path
        suffix = full.suffix.lower()
        file_type = "oms_signature" if path in recognized_oms_signatures else _infer_file_type(path)
        content = file_cache.get(path)
        lines = (
            len(content.splitlines())
            if content is not None
            else _count_lines(full)
            if path in recognized_oms_signatures
            else 0
        )
        executable = suffix in _EXECUTABLE_EXTENSIONS
        if executable:
            has_executable = True
        try:
            size_bytes = full.stat().st_size
        except OSError:
            logger.debug("Could not stat file: %s", path)
            size_bytes = 0
        metadata.append(
            {
                "path": path,
                "type": file_type,
                "lines": lines,
                "executable": executable,
                "size_bytes": size_bytes,
            }
        )
    return metadata, has_executable


def _read_file_cache(
    skill_dir: Path, components: list[str]
) -> tuple[dict[str, str], list[InspectionLedgerEvent]]:
    """Build readable file content and terminal events for cache failures."""
    file_cache: dict[str, str] = {}
    ledger_events: list[InspectionLedgerEvent] = []
    skill_root = skill_dir.resolve(strict=False)
    for path in components:
        full = skill_dir / path
        if _is_symlink(full) or _resolves_outside(full, skill_root):
            ledger_events.append(
                ledger_event(
                    outcome=LedgerOutcome.OUT_OF_SCOPE,
                    record_type=LedgerRecordType.SCOPE_BOUNDARY,
                    phase="cache",
                    path=path,
                    reason=LedgerReason.NOT_REGULAR_FILE,
                )
            )
            continue
        try:
            file_stat = full.stat()
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
            continue
        except OSError as exc:
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
            continue
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
            continue
        try:
            content = _read_text_no_follow(full)
            file_cache[path] = content
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
    return file_cache, ledger_events


def _parse_manifest(skill_dir: Path) -> dict[str, object]:
    """Parse SKILL.md or skill.md YAML frontmatter into a manifest dict.

    Returns dict with name, description, triggers (list), permissions (list),
    allowed-tools (list), parameters (list). Returns {} if no file or parse fails.
    """
    skill_root = skill_dir.resolve(strict=False)
    for name in ("SKILL.md", "skill.md"):
        path = skill_dir / name
        if _is_symlink(path) or _resolves_outside(path, skill_root) or not path.is_file():
            continue
        try:
            content = _read_text_no_follow(path)
        except (OSError, _FileOpenError, _UnsafeFileError):
            logger.debug("Could not read manifest file: %s", name)
            return {}
        if not content.startswith("---"):
            return {}
        end_match = re.search(r"\n---\s*\n", content[3:])
        if not end_match:
            return {}
        frontmatter = content[3 : end_match.start() + 3]
        try:
            data = yaml.safe_load(frontmatter)
        except yaml.YAMLError:
            logger.debug("Manifest parse failed for %s", name)
            return {}
        if not isinstance(data, dict):
            return {}
        manifest: dict[str, object] = {}
        if "name" in data:
            manifest["name"] = data["name"]
        if "description" in data:
            manifest["description"] = data["description"]
        triggers = data.get("triggers", [])
        manifest["triggers"] = [str(t) for t in triggers] if isinstance(triggers, list) else []
        permissions = data.get("permissions", [])
        manifest["permissions"] = (
            [str(p) for p in permissions] if isinstance(permissions, list) else []
        )
        # `allowed-tools` (Agent Skills standard) — accept list or comma string.
        allowed_tools = data.get("allowed-tools", [])
        if isinstance(allowed_tools, list):
            manifest["allowed-tools"] = [str(t).strip() for t in allowed_tools if str(t).strip()]
        elif isinstance(allowed_tools, str):
            manifest["allowed-tools"] = [t.strip() for t in allowed_tools.split(",") if t.strip()]
        else:
            manifest["allowed-tools"] = []
        # Preserve parameter definitions as dicts so the MCP tool-poisoning
        # analyzer (TP1/TP2/TP3 parameter checks) can inspect them. Without
        # this, those checks never fire on real scans because the manifest
        # carried no `parameters` key.
        parameters = data.get("parameters", [])
        manifest["parameters"] = (
            [p for p in parameters if isinstance(p, dict)] if isinstance(parameters, list) else []
        )
        return manifest
    return {}


def build_context(state: SkillspectorState) -> dict[str, object]:
    """Build flat ScanContext fields from state skill_path (local directory).

    Resolves skill_path to a directory, walks files, builds file_cache
    and manifest. Returns only context keys; leaves findings untouched.
    Raises ValueError if skill_path is missing or not an existing directory.
    """
    skill_dir = _resolve_skill_dir(state)

    inventoried_components, discovery_events = _walk_skill_files(skill_dir)
    recognized_oms_signatures = frozenset(
        {_OMS_SIGNATURE_PATH}
        if _OMS_SIGNATURE_PATH in inventoried_components
        and _is_valid_oms_signature(skill_dir / _OMS_SIGNATURE_PATH)
        else set()
    )
    selected_baseline = _selected_baseline_component(state, skill_dir, inventoried_components)
    selected_baselines = frozenset({selected_baseline} if selected_baseline else set())
    components = [
        path
        for path in inventoried_components
        if path not in recognized_oms_signatures and path not in selected_baselines
    ]
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
    file_cache, cache_events = _read_file_cache(skill_dir, components)
    python_ast_cache_key = prewarm_python_ast_cache(components, file_cache)
    manifest = _parse_manifest(skill_dir)
    metadata_components = [
        path for path in inventoried_components if path not in selected_baselines
    ]
    component_metadata, has_executable_scripts = _build_component_metadata(
        skill_dir, metadata_components, file_cache, recognized_oms_signatures
    )
    structured_skill_context = extract_structured_skill_context(skill_dir)

    result = {
        "components": components,
        "file_cache": file_cache,
        "inspection_ledger": [
            *discovery_events,
            *signature_events,
            *baseline_events,
            *cache_events,
        ],
        "ast_cache": {},
        "python_ast_cache_key": python_ast_cache_key,
        "manifest": manifest,
        "previous_manifest": None,
        "model_config": build_model_config(),
        "component_metadata": component_metadata,
        "has_executable_scripts": has_executable_scripts,
    }

    if structured_skill_context is not None:
        result["structured_skill_context"] = structured_skill_context

    return result
