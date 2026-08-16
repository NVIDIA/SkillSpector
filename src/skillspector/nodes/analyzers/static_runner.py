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
from collections.abc import Callable, Mapping
from typing import cast

from skillspector.inspection_ledger import (
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    analyzer_status_for_events,
    ledger_event,
)
from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Finding
from skillspector.python_ast import (
    MAX_PYTHON_AST_SOURCE_CHARS,
    ParsedPythonFile,
    get_python_ast,
)
from skillspector.state import AnalyzerNodeResponse

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

MAX_FILE_CHARS = MAX_PYTHON_AST_SOURCE_CHARS
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


_BINARY_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".eot",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bin",
        ".o",
        ".a",
        ".pyc",
        ".pyo",
        ".class",
        ".wasm",
        ".mp3",
        ".mp4",
        ".wav",
        ".avi",
        ".mov",
        ".webm",
        ".sqlite",
        ".db",
    }
)

_NULL_BYTE_SAMPLE_SIZE = 512


def _is_binary_file(path: str, content: str) -> bool:
    """Detect binary files by extension or null-byte presence in the first 512 chars."""
    idx = path.rfind(".")
    if idx >= 0 and path[idx:].lower() in _BINARY_EXTENSIONS:
        return True
    return "\x00" in content[:_NULL_BYTE_SAMPLE_SIZE]


_PE3_ENV_TEMPLATE_SETUP = re.compile(
    r"(?:[-*]\s*)?(?:cp|copy|mv|rename)\s+\.env\.(?:example|sample|template)\s+"
    r"(?:to\s+)?\.env(?:\s+(?:before\s+(?:running|starting)(?:\s+the\s+app)?|"
    r"for\s+local\s+development))?[.:]?",
    re.IGNORECASE,
)
_PE3_ENV_FILE_SETUP = re.compile(
    r"(?:create|configure|set\s+up|make|add)\s+(?:an?\s+|the\s+)?\.env(?:\s+file)?"
    r"(?:\s+in\s+the\s+project\s+root)?(?:\s+with\s+(?:your\s+)?api\s+keys?|"
    r"\s+for\s+(?:local\s+)?(?:development|testing))?[.:]?",
    re.IGNORECASE,
)
_PE3_DOTENV_SETUP = re.compile(
    r"(?:install|use)\s+(?:python-)?dotenv\s+to\s+load\s+(?:the\s+)?\.env\s+file[.:]?",
    re.IGNORECASE,
)


def _is_env_file_reference_in_docs(
    finding: AnalyzerFinding,
    file_type: str,
    file_path: str = "",
    content: str | None = None,
) -> bool:
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

    if content is not None:
        lines = content.splitlines()
        index = finding.location.start_line - 1
        if index < 0 or index >= len(lines):
            return False
        line = lines[index]
    else:
        candidate_lines = [line for line in finding.context.splitlines() if ".env" in line.lower()]
        if len(candidate_lines) != 1:
            return False
        line = candidate_lines[0]

    normalized_line = line.replace("`", "").strip()
    return any(
        pattern.fullmatch(normalized_line) is not None
        for pattern in (_PE3_ENV_TEMPLATE_SETUP, _PE3_ENV_FILE_SETUP, _PE3_DOTENV_SETUP)
    )


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

_NON_EXECUTABLE_FILE_TYPES = frozenset({"markdown", "text", "json", "yaml", "toml"})
_DOC_PROSE_FILE_TYPES = frozenset({"markdown", "text"})

# PE3 is intentionally excluded: its analyzer and the exact .env setup grammar
# above own the narrowly reviewed safe cases. A generic prose classification
# must not hide credential-access instructions.
_SEMANTIC_STRING_DOC_PRONE_RULES = frozenset({"RA1", "TM1", "AR2"})
_EXECUTION_SIGNAL = re.compile(
    r"(?:\b\w+\s*=|\bos\.(?:environ|getenv|system)\b|\bshutil\.rmtree\b|\b(?:subprocess|eval|exec)\b|[|>]"
    r"|\b(?:open|read_text|write_text)\s*\()",
    re.IGNORECASE,
)


# Markdown syntax that collides with shell metacharacters. A table row is delimited by "|" and
# a quoted line begins with ">": neither is a pipe or a redirection, but _EXECUTION_SIGNAL reads
# them as one and the prose classification below is then skipped for the whole line.
#
# Only the *delimiters* are removed — the leading and trailing bar of a row and the quote marker.
# A bar inside a cell may well be a real pipe in a documented command, and it must keep counting
# as an execution signal.
_MD_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_MD_BLOCKQUOTE = re.compile(r"^\s*>+\s?")
_MD_ESCAPED_BAR = "\\|"
_BAR_PLACEHOLDER = "\x00"


def _strip_markdown_structure(line: str) -> str:
    r"""Drop markdown delimiters that would otherwise read as shell metacharacters.

    In a table row an unescaped ``|`` separates cells; a literal pipe inside a cell has to be
    written ``\|`` (CommonMark). That distinction is what makes this safe: the delimiters are
    removed, while a documented ``cmd \| tee log`` keeps its pipe and still counts as an
    execution signal.
    """
    if _MD_TABLE_ROW.match(line):
        line = line.replace(_MD_ESCAPED_BAR, _BAR_PLACEHOLDER)
        line = line.replace("|", " ")
        line = line.replace(_BAR_PLACEHOLDER, "|")
    return _MD_BLOCKQUOTE.sub("", line)


def _is_documentation_context(af: AnalyzerFinding, file_type: str, path: str, content: str) -> bool:
    """Return true when a governed finding is prose or a comment without execution signals."""
    if af.rule_id not in _SEMANTIC_STRING_DOC_PRONE_RULES:
        return False
    if path.replace("\\", "/").lower().endswith("skill.md"):
        return False
    lines = content.splitlines()
    matched_line = (
        lines[af.location.start_line - 1]
        if 0 < af.location.start_line <= len(lines)
        else af.context or ""
    )
    if file_type in _DOC_PROSE_FILE_TYPES:
        if _EXECUTION_SIGNAL.search(_strip_markdown_structure(matched_line)):
            return False
        return True
    return bool(matched_line and matched_line.lstrip().startswith(("#", "//")))


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


def _uses_python_ast(module: object) -> bool:
    """Return whether a pattern module explicitly opts into the shared AST hook."""
    return getattr(module, "USES_PYTHON_AST", False) is True


def _scan_path(
    path: str,
    content: str,
    pattern_modules: list,
    python_ast_cache_key: str | None = None,
) -> list[Finding]:
    """Run pattern modules for one already-applicable file path."""
    findings: list[Finding] = []
    file_type = _infer_file_type(path)
    is_doc_markdown = _is_documentation_markdown(path)
    is_non_executable = file_type in _NON_EXECUTABLE_FILE_TYPES
    python_ast: ParsedPythonFile | None = None
    if file_type == "python" and any(_uses_python_ast(module) for module in pattern_modules):
        python_ast = get_python_ast(python_ast_cache_key, content, path)

    for module in pattern_modules:
        if file_type == "python" and _uses_python_ast(module):
            raw = module.analyze(
                content=content,
                file_path=path,
                file_type=file_type,
                python_ast=python_ast,
            )
        else:
            raw = module.analyze(content=content, file_path=path, file_type=file_type)
        for af in raw:
            if _is_env_file_reference_in_docs(af, file_type, path, content):
                logger.debug(
                    "Filtered PE3 .env doc reference: %s in %s:%d",
                    af.rule_id,
                    path,
                    af.location.start_line,
                )
                continue
            # PE3's analyzer owns its narrowly qualified safe references.
            # Generic documentation words are attacker-controlled and must
            # not hard-drop HIGH credential-access findings here.
            if af.rule_id != "PE3" and af.context and is_code_example(af.context, path=path):
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
            if _is_documentation_context(af, file_type, path, content):
                logger.debug(
                    "Filtered documentation-context finding: %s in %s:%d",
                    af.rule_id,
                    path,
                    af.location.start_line,
                )
                continue
            if is_doc_markdown:
                af.confidence *= _DOCUMENTATION_CONFIDENCE_FACTOR
            findings.append(analyzer_finding_to_finding(af))
    return findings


def run_static_patterns(
    state: Mapping[str, object],
    pattern_modules: list,
) -> list[Finding]:
    """
    Run one or more pattern modules over state components/file_cache.

    For each path in state["components"], loads content from state["file_cache"],
    infers file_type, runs each module's analyze(content, path, file_type),
    converts all AnalyzerFindings to Finding via analyzer_finding_to_finding, returns combined list.
    """
    components = cast(list[str], state.get("components") or [])
    file_cache = cast(dict[str, str], state.get("file_cache") or {})
    python_ast_cache_key = cast(str | None, state.get("python_ast_cache_key"))
    findings: list[Finding] = []

    for path in components:
        if _is_eval_dataset(path):
            logger.debug("Skipping eval dataset prose for static pattern scan: %s", path)
            continue
        content = file_cache.get(path)
        if content is None:
            logger.debug("Skipping %s: no content in file_cache", path)
            continue
        if len(content) > MAX_FILE_CHARS:
            logger.debug(
                "Skipping %s: size %d characters exceeds MAX_FILE_CHARS (%d)",
                path,
                len(content),
                MAX_FILE_CHARS,
            )
            continue
        if _is_binary_file(path, content):
            logger.debug("Skipping binary file: %s", path)
            continue
        findings.extend(_scan_path(path, content, pattern_modules, python_ast_cache_key))

    return findings


def run_static_patterns_with_ledger(
    state: Mapping[str, object],
    pattern_modules: list,
) -> AnalyzerNodeResponse:
    """Run one static analyzer and account for every planned file work item."""
    analyzer_id = str(getattr(pattern_modules[0], "ANALYZER_ID", "static_patterns"))
    components = cast(list[str], state.get("components") or [])
    file_cache = cast(dict[str, str], state.get("file_cache") or {})
    python_ast_cache_key = cast(str | None, state.get("python_ast_cache_key"))
    findings: list[Finding] = []
    events: list[InspectionLedgerEvent] = []

    for path in components:
        if _is_eval_dataset(path):
            event = ledger_event(
                outcome=LedgerOutcome.SKIPPED,
                phase="static",
                analyzer_id=analyzer_id,
                path=path,
                reason=LedgerReason.EVAL_DATASET,
            )
        else:
            content = file_cache.get(path)
            if content is None:
                event = ledger_event(
                    outcome=LedgerOutcome.FAILED,
                    phase="static",
                    analyzer_id=analyzer_id,
                    path=path,
                    reason=LedgerReason.MISSING_FILE_CACHE,
                )
            elif len(content) > MAX_FILE_CHARS:
                event = ledger_event(
                    outcome=LedgerOutcome.SKIPPED,
                    phase="static",
                    analyzer_id=analyzer_id,
                    path=path,
                    reason=LedgerReason.SIZE_LIMIT,
                    observed_characters=len(content),
                    limit_characters=MAX_FILE_CHARS,
                    observed_bytes=len(content.encode("utf-8")),
                )
            elif _is_binary_file(path, content):
                event = ledger_event(
                    outcome=LedgerOutcome.SKIPPED,
                    phase="static",
                    analyzer_id=analyzer_id,
                    path=path,
                    reason=LedgerReason.BINARY_CONTENT,
                )
            else:
                try:
                    path_findings = _scan_path(path, content, pattern_modules, python_ast_cache_key)
                except Exception as exc:
                    logger.warning("%s: scan error on %s: %s", analyzer_id, path, exc)
                    event = ledger_event(
                        outcome=LedgerOutcome.FAILED,
                        phase="static",
                        analyzer_id=analyzer_id,
                        path=path,
                        reason=LedgerReason.ANALYZER_RUNTIME_ERROR,
                        error_class=type(exc).__name__,
                    )
                else:
                    findings.extend(path_findings)
                    event = ledger_event(
                        outcome=LedgerOutcome.COMPLETED,
                        phase="static",
                        analyzer_id=analyzer_id,
                        path=path,
                        emitted_finding_ids=[finding.finding_id for finding in path_findings],
                    )
        events.append(event)

    return {
        "findings": findings,
        "inspection_ledger": events,
        "analyzer_status_events": [analyzer_status_for_events(analyzer_id, events)],
    }
