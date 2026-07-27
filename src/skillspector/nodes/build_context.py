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
from dataclasses import dataclass
from pathlib import Path

import yaml

from skillspector.constants import build_model_config
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


@dataclass(frozen=True)
class _FileInspection:
    """Content classification used to decide whether LLM analyzers can read a file."""

    content: str
    content_kind: str
    llm_analysis_status: str
    llm_skip_reason: str | None = None


def _has_media_signature(data: bytes) -> bool:
    """Return whether bytes have a recognized image, audio, or video signature."""
    if data.startswith(
        (
            b"\x89PNG\r\n\x1a\n",
            b"\xff\xd8\xff",
            b"GIF87a",
            b"GIF89a",
            b"BM",
            b"\x00\x00\x01\x00",
            b"II*\x00",
            b"MM\x00*",
            b"fLaC",
            b"OggS",
            b"\x1aE\xdf\xa3",
            b"\x00\x00\x01\xba",
            b"\x00\x00\x01\xb3",
        )
    ):
        return True
    if data.startswith(b"RIFF") and data[8:12] in {b"WEBP", b"WAVE", b"AVI "}:
        return True
    if data.startswith(b"ID3") or (
        len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE6 in {0xE0, 0xE2, 0xE4, 0xE6}
    ):
        return True
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return data[8:12].lower() in {
            b"3gp4",
            b"3gp5",
            b"avif",
            b"avis",
            b"heic",
            b"heix",
            b"hevc",
            b"hevx",
            b"isom",
            b"m4a ",
            b"m4v ",
            b"mif1",
            b"mp41",
            b"mp42",
            b"msf1",
            b"qt  ",
        }
    return False


def _has_abnormal_controls(content: str) -> bool:
    """Return whether decoded text contains controls not used for normal layout."""
    return any(
        (ord(char) < 32 and char not in "\t\n\r\f") or 127 <= ord(char) <= 159 for char in content
    )


def _is_strong_binary(data: bytes) -> bool:
    """Detect strong binary evidence while leaving uncertain encodings analyzable."""
    if b"\x00" in data:
        return True
    if not data:
        return False
    abnormal_controls = sum(
        (byte < 32 and byte not in {9, 10, 12, 13}) or byte == 127 for byte in data
    )
    return abnormal_controls / len(data) >= 0.3


def _inspect_bytes(data: bytes) -> _FileInspection:
    """Classify bytes, preferring analyzable text and failing open when uncertain."""
    try:
        strict_content = data.decode("utf-8")
    except UnicodeDecodeError:
        strict_content = None

    if strict_content is not None and not _has_abnormal_controls(strict_content):
        return _FileInspection(strict_content, "text", "included")
    if _has_media_signature(data):
        return _FileInspection(
            data.decode("utf-8", errors="replace"),
            "media",
            "excluded",
            "media_content",
        )
    if _is_strong_binary(data):
        return _FileInspection(
            data.decode("utf-8", errors="replace"),
            "binary",
            "excluded",
            "binary_content",
        )

    # Invalid or unusual text that is not confidently binary remains in scope.
    return _FileInspection(data.decode("utf-8", errors="replace"), "text", "included")


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
            # Use forward slashes on every OS: these relative paths are dict keys
            # and SARIF/URI locations, so they must be portable (not OS-specific
            # backslashes on Windows).
            paths.append(rel.as_posix())
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


def _build_component_metadata(
    skill_dir: Path,
    components: list[str],
    inspections: dict[str, _FileInspection],
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
        inspection = inspections[path]
        lines = len(inspection.content.splitlines())
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
                "content_kind": inspection.content_kind,
                "llm_analysis_status": inspection.llm_analysis_status,
                "llm_skip_reason": inspection.llm_skip_reason,
            }
        )
    return metadata, has_executable


def _read_file_cache(
    skill_dir: Path, components: list[str]
) -> tuple[dict[str, str], dict[str, str], dict[str, _FileInspection]]:
    """Build the shared and LLM-eligible caches and classify every component."""
    file_cache: dict[str, str] = {}
    llm_file_cache: dict[str, str] = {}
    inspections: dict[str, _FileInspection] = {}
    for path in components:
        full = skill_dir / path
        if not full.is_file():
            continue
        try:
            inspection = _inspect_bytes(full.read_bytes())
        except OSError:
            logger.debug("Could not read file: %s", path)
            inspection = _FileInspection("", "unknown", "excluded", "read_error")
        inspections[path] = inspection
        file_cache[path] = inspection.content
        if inspection.llm_analysis_status == "included":
            llm_file_cache[path] = inspection.content
        else:
            logger.info("Excluding %s from LLM analysis: %s", path, inspection.llm_skip_reason)
    return file_cache, llm_file_cache, inspections


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

    components = _walk_skill_files(skill_dir)
    file_cache, llm_file_cache, inspections = _read_file_cache(skill_dir, components)
    manifest = _parse_manifest(skill_dir)
    component_metadata, has_executable_scripts = _build_component_metadata(
        skill_dir, components, inspections
    )

    return {
        "components": components,
        "file_cache": file_cache,
        "llm_file_cache": llm_file_cache,
        "ast_cache": {},
        "manifest": manifest,
        "previous_manifest": None,
        "model_config": build_model_config(),
        "component_metadata": component_metadata,
        "has_executable_scripts": has_executable_scripts,
    }
