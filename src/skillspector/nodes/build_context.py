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

import os
import re
from pathlib import Path
from stat import S_ISREG

import yaml

from skillspector.constants import build_model_config
from skillspector.inspection_ledger import (
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    LedgerRecordType,
    ledger_event,
)
from skillspector.logging_config import get_logger
from skillspector.state import SkillspectorState

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


def _resolve_skill_dir(state: SkillspectorState) -> Path:
    """Resolve state skill_path to an existing directory Path."""
    skill_path = state.get("skill_path")
    if not skill_path or not isinstance(skill_path, str) or not skill_path.strip():
        raise ValueError("skill_path is required; provide input_path or skill_path to scan")
    try:
        resolved = Path(skill_path).resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid skill_path: {skill_path}") from e
    if not resolved.is_dir():
        raise ValueError(f"Invalid skill_path: {skill_path} is not an existing directory")
    return resolved


def _walk_skill_files(
    skill_dir: Path,
) -> tuple[list[str], list[InspectionLedgerEvent]]:
    """Walk skill files and record scan-scope exclusions.

    Skips _SKIP_DIRS and hidden files except those starting with .claude.
    """
    paths: list[str] = []
    exclusions: list[InspectionLedgerEvent] = []
    for root, dirnames, filenames in os.walk(skill_dir):
        root_path = Path(root)
        dirnames.sort()
        filenames.sort()
        relative_root = root_path.relative_to(skill_dir)

        skipped_dirnames = [name for name in dirnames if name in _SKIP_DIRS]
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
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
            # and SARIF/URI locations, so they must be portable.  Do not filter
            # on ``is_file()`` here: it follows symlinks and silently discards
            # dangling or non-regular entries before the cache phase can record
            # their terminal ledger evidence.
            paths.append(relative_path)
    paths.sort()
    return paths, exclusions


def _infer_file_type(path: str) -> str:
    """Infer file type from path (extension)."""
    idx = path.rfind(".")
    suffix = path[idx:].lower() if idx >= 0 else ""
    return _FILE_TYPES.get(suffix, "other")


def _build_component_metadata(
    skill_dir: Path, components: list[str], file_cache: dict[str, str]
) -> tuple[list[dict[str, object]], bool]:
    """Build component_metadata list and has_executable_scripts from paths."""
    metadata: list[dict[str, object]] = []
    has_executable = False
    for path in components:
        full = skill_dir / path
        suffix = full.suffix.lower()
        file_type = _infer_file_type(path)
        content = file_cache.get(path)
        lines = len(content.splitlines()) if content is not None else 0
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
    for path in components:
        full = skill_dir / path
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
            content = full.read_text(encoding="utf-8", errors="replace")
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
    for name in ("SKILL.md", "skill.md"):
        path = skill_dir / name
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
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

    components, discovery_events = _walk_skill_files(skill_dir)
    file_cache, cache_events = _read_file_cache(skill_dir, components)
    manifest = _parse_manifest(skill_dir)
    component_metadata, has_executable_scripts = _build_component_metadata(
        skill_dir, components, file_cache
    )

    return {
        "components": components,
        "file_cache": file_cache,
        "inspection_ledger": [*discovery_events, *cache_events],
        "ast_cache": {},
        "manifest": manifest,
        "previous_manifest": None,
        "model_config": build_model_config(),
        "component_metadata": component_metadata,
        "has_executable_scripts": has_executable_scripts,
    }
