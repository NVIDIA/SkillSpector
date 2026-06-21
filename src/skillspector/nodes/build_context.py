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

import re
from pathlib import Path

import yaml

from skillspector.constants import MODEL_CONFIG
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

# Per-file read cap. Files larger than this fail the scan rather than being
# skipped, because skipping would let malicious content hide in an oversized
# file. Aligned with static_runner.MAX_FILE_BYTES.
_MAX_READ_BYTES = 50 * 1024 * 1024


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


def _walk_skill_files(skill_dir: Path) -> list[str]:
    """Walk skill directory and return sorted relative path strings.

    Skips _SKIP_DIRS and hidden files except those starting with .claude.
    """
    paths: list[str] = []
    for item in skill_dir.rglob("*"):
        if not item.is_file():
            continue
        if any(skip in item.parts for skip in _SKIP_DIRS):
            continue
        if item.name.startswith(".") and not item.name.startswith(".claude"):
            continue
        try:
            rel = item.relative_to(skill_dir)
            paths.append(str(rel))
        except ValueError:
            logger.debug("Skipping path (not under skill_dir): %s", item)
            continue
    paths.sort()
    return paths


def _infer_file_type(path: str) -> str:
    """Infer file type from path (extension)."""
    idx = path.rfind(".")
    suffix = path[idx:].lower() if idx >= 0 else ""
    return _FILE_TYPES.get(suffix, "other")


def _count_lines(file_path: Path) -> int:
    """Count lines in a file, handling binary and errors gracefully.

    Reads the file in fixed-size binary chunks and counts newline bytes so
    peak memory stays bounded regardless of file size *or* line length — a
    multi-gigabyte file with no newlines must not be materialized in memory
    (consistent with the _MAX_READ_BYTES cap in _read_file_cache). The count
    matches ``len(text.splitlines())`` for the common ``\\n`` / ``\\r\\n`` cases:
    number of newline bytes, plus one for a final line without a trailing
    newline.
    """
    try:
        newline_count = 0
        last_byte = b""
        with file_path.open("rb") as fh:
            while chunk := fh.read(65536):
                newline_count += chunk.count(b"\n")
                last_byte = chunk[-1:]
        if last_byte and last_byte != b"\n":
            newline_count += 1  # final line with no trailing newline
        return newline_count
    except OSError:
        logger.debug("Could not read file for line count: %s", file_path)
        return 0


def _build_component_metadata(
    skill_dir: Path, components: list[str]
) -> tuple[list[dict[str, object]], bool]:
    """Build component_metadata list and has_executable_scripts from paths."""
    metadata: list[dict[str, object]] = []
    has_executable = False
    for path in components:
        full = skill_dir / path
        if not full.is_file():
            continue
        suffix = full.suffix.lower()
        file_type = _infer_file_type(path)
        lines = _count_lines(full)
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


def _read_file_cache(skill_dir: Path, components: list[str]) -> dict[str, str]:
    """Build file_cache: relative path -> file contents. Uses utf-8 with replace for errors."""
    file_cache: dict[str, str] = {}
    for path in components:
        full = skill_dir / path
        if not full.is_file():
            continue
        try:
            content = full.read_text(encoding="utf-8", errors="replace")
            file_cache[path] = content
        except OSError:
            logger.debug("Could not read file: %s", path)
            file_cache[path] = ""
    return file_cache


def _validate_file_sizes(skill_dir: Path, components: list[str]) -> None:
    """Fail the scan if any discovered file exceeds the per-file read cap."""
    oversized: list[tuple[str, int]] = []
    for path in components:
        full = skill_dir / path
        if not full.is_file():
            continue
        try:
            size_bytes = full.stat().st_size
        except OSError:
            logger.debug("Could not stat file for size validation: %s", path)
            continue
        if size_bytes > _MAX_READ_BYTES:
            oversized.append((path, size_bytes))

    if not oversized:
        return

    details = ", ".join(f"{path} ({size_bytes} bytes)" for path, size_bytes in oversized[:5])
    if len(oversized) > 5:
        details += f", and {len(oversized) - 5} more"
    raise ValueError(
        "Scan aborted: file size exceeds the per-file analysis limit "
        f"({_MAX_READ_BYTES} bytes): {details}"
    )


def _parse_manifest(skill_dir: Path) -> dict[str, object]:
    """Parse SKILL.md or skill.md YAML frontmatter into a manifest dict.

    Returns dict with name, description, triggers (list), permissions (list).
    Returns {} if no file or parse fails.
    """
    for name in ("SKILL.md", "skill.md"):
        path = skill_dir / name
        if not path.is_file():
            continue
        try:
            # Only the leading YAML frontmatter is needed, and it sits at the
            # top of the file. Read a bounded *byte* prefix (binary mode, then
            # decode) so an oversized SKILL.md (a huge body, or a malicious
            # multi-GB file) is never materialized whole. Reading bytes rather
            # than text makes the cap a true byte ceiling: a text-mode
            # read(_MAX_READ_BYTES) bounds characters, which multibyte content
            # could inflate well past _MAX_READ_BYTES of memory. If a skill's
            # frontmatter somehow exceeds this, the closing delimiter falls
            # outside the prefix and parsing degrades to {} below.
            with path.open("rb") as fh:
                content = fh.read(_MAX_READ_BYTES).decode("utf-8", errors="replace")
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

    components = _walk_skill_files(skill_dir)
    _validate_file_sizes(skill_dir, components)
    file_cache = _read_file_cache(skill_dir, components)
    manifest = _parse_manifest(skill_dir)
    component_metadata, has_executable_scripts = _build_component_metadata(skill_dir, components)

    return {
        "components": components,
        "file_cache": file_cache,
        "ast_cache": {},
        "manifest": manifest,
        "previous_manifest": None,
        "model_config": MODEL_CONFIG,
        "component_metadata": component_metadata,
        "has_executable_scripts": has_executable_scripts,
    }
