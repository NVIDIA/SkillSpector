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

"""Shared runner for static pattern nodes: file-type inference, conversion, run_static_patterns."""

from __future__ import annotations

import re
from collections.abc import Callable

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Finding

from .common import is_code_example
from .pattern_defaults import get_category, get_explanation, get_pattern_name, get_remediation

logger = get_logger(__name__)

# Extension -> file type (match v1 InventoryBuilder.FILE_TYPES)
FILE_TYPES: dict[str, str] = {
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

MAX_FILE_BYTES = 1_000_000
_EVAL_DATASET_FILES = {
    "evals/evals.json",
    "evals/evals.jsonl",
    "evals/evals.yaml",
    "evals/evals.yml",
    "eval/dataset.json",
    "eval/dataset.jsonl",
    "eval/dataset.yaml",
    "eval/dataset.yml",
}


def _infer_file_type(path: str) -> str:
    """Infer file type from path (extension)."""
    idx = path.rfind(".")
    suffix = path[idx:].lower() if idx >= 0 else ""
    return FILE_TYPES.get(suffix, "other")


_BINARY_EXTENSIONS = frozenset({
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a",
    ".pyc", ".pyo", ".class", ".wasm",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".webm",
    ".sqlite", ".db",
})

_NULL_BYTE_SAMPLE_SIZE = 512


def _is_binary_file(path: str, content: str) -> bool:
    """Detect binary files by extension or null-byte presence in the first 512 chars."""
    idx = path.rfind(".")
    if idx >= 0 and path[idx:].lower() in _BINARY_EXTENSIONS:
        return True
    return "\x00" in content[:_NULL_BYTE_SAMPLE_SIZE]


_PE3_ENV_REFERENCE_CONTEXT = re.compile(
    r"(?:create|copy|rename|add|set up|configure|make)\s+.*\.env",
    re.IGNORECASE,
)


def _is_env_file_reference_in_docs(finding: AnalyzerFinding, file_type: str, file_path: str = "") -> bool:
    """Return True if a PE3 finding is a documentation reference to .env files, not actual access.

    SKILL.md is exempt: it is the agent's primary instruction file, so `.env`
    references there may be genuine credential-access instructions.
    """
    if finding.rule_id != "PE3":
        return False
    if file_type not in ("markdown", "text"):
        return False
    if file_path.replace("\\", "/").lower().endswith("skill.md"):
        return False
    if not finding.context:
        return False
    if _PE3_ENV_REFERENCE_CONTEXT.search(finding.context):
        return True
    ctx_lower = finding.context.lower()
    doc_phrases = (
        ".env.example",
        "cp .env",
        "copy .env",
        "mv .env",
        "rename .env",
        ".env file",
        "environment file",
        "dotenv",
    )
    return any(phrase in ctx_lower for phrase in doc_phrases)


def _is_eval_dataset(path: str) -> bool:
    """Return True for authored eval datasets that contain test-case prose."""
    return path.replace("\\", "/") in _EVAL_DATASET_FILES


_DOCUMENTATION_DIR_NAMES = (
    "docs",
    "documentation",
    "procedures",
    "references",
    "examples",
    "guides",
)

_DOCUMENTATION_CONFIDENCE_FACTOR = 0.3
_CODE_EXAMPLE_CONFIDENCE_FACTOR = 0.5

_NON_EXECUTABLE_FILE_TYPES = frozenset({"markdown", "text", "json", "yaml", "toml", "other"})


def _is_documentation_markdown(path: str) -> bool:
    """Return True for markdown files in documentation subdirectories (not SKILL.md)."""
    normalized = path.replace("\\", "/").lower()
    if not normalized.endswith((".md", ".markdown")):
        return False
    if normalized.endswith("skill.md"):
        return False
    parts = normalized.split("/")
    return any(part in _DOCUMENTATION_DIR_NAMES for part in parts[:-1])


def analyzer_finding_to_finding(
    af: AnalyzerFinding,
    get_remediation_fn: Callable[[str], str] | None = None,
) -> Finding:
    """Convert an AnalyzerFinding (from any analyzer) to graph-state Finding."""
    rem_fn = get_remediation_fn or get_remediation
    remediation = af.remediation or rem_fn(af.rule_id)
    category = (af.tags[0] if af.tags else None) or get_category(af.rule_id)
    pattern = af.message or get_pattern_name(af.rule_id)
    finding_snippet = af.matched_text[:200] if af.matched_text else None
    return Finding(
        rule_id=af.rule_id,
        message=af.message,
        severity=af.severity.value,
        confidence=af.confidence,
        file=af.location.file,
        start_line=af.location.start_line,
        end_line=af.location.end_line,
        remediation=remediation,
        tags=list(af.tags),
        context=af.context,
        matched_text=af.matched_text[:200] if af.matched_text else None,
        category=category,
        pattern=pattern,
        finding=finding_snippet,
        explanation=get_explanation(af.rule_id),
        code_snippet=af.context,
        intent=None,
    )


def run_static_patterns(
    state: dict[str, object],
    pattern_modules: list,
) -> list[Finding]:
    """
    Run one or more pattern modules over state components/file_cache.

    For each path in state["components"], loads content from state["file_cache"],
    infers file_type, runs each module's analyze(content, path, file_type),
    converts all AnalyzerFindings to Finding via analyzer_finding_to_finding, returns combined list.
    """
    components: list[str] = state.get("components") or []
    file_cache: dict[str, str] = state.get("file_cache") or {}
    findings: list[Finding] = []

    for path in components:
        if _is_eval_dataset(path):
            logger.debug("Skipping eval dataset prose for static pattern scan: %s", path)
            continue
        content = file_cache.get(path)
        if content is None:
            logger.debug("Skipping %s: no content in file_cache", path)
            continue
        if len(content) > MAX_FILE_BYTES:
            logger.debug(
                "Skipping %s: size %d exceeds MAX_FILE_BYTES (%d)",
                path,
                len(content),
                MAX_FILE_BYTES,
            )
            continue
        if _is_binary_file(path, content):
            logger.debug("Skipping binary file: %s", path)
            continue
        file_type = _infer_file_type(path)
        is_doc_markdown = _is_documentation_markdown(path)
        is_non_executable = file_type in _NON_EXECUTABLE_FILE_TYPES
        for module in pattern_modules:
            raw = module.analyze(content=content, file_path=path, file_type=file_type)
            for af in raw:
                if _is_env_file_reference_in_docs(af, file_type, path):
                    logger.debug(
                        "Filtered PE3 .env doc reference: %s in %s:%d",
                        af.rule_id, path, af.location.start_line,
                    )
                    continue
                if af.context and is_code_example(af.context):
                    if is_non_executable:
                        logger.debug(
                            "Filtered code-example finding in non-executable: %s in %s:%d",
                            af.rule_id,
                            path,
                            af.location.start_line,
                        )
                        continue
                    af.confidence *= _CODE_EXAMPLE_CONFIDENCE_FACTOR
                    logger.debug(
                        "Downweighted code-example finding in executable: %s in %s:%d (conf=%.2f)",
                        af.rule_id,
                        path,
                        af.location.start_line,
                        af.confidence,
                    )
                if is_doc_markdown:
                    af.confidence *= _DOCUMENTATION_CONFIDENCE_FACTOR
                findings.append(analyzer_finding_to_finding(af))

    return findings
