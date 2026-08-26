# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-pinned extraction, parser boundary, and bounded shell modeling.

This module intentionally contains no package-manager policy or host shell state.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Callable, Generator
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cache
from importlib import import_module
from math import ceil, isfinite
from time import monotonic
from typing import TYPE_CHECKING, Any, Final, cast
from warnings import catch_warnings, filterwarnings

from skillspector.dependency_source_types import (
    MAX_DEPENDENCY_FILE_BYTES,
    MAX_DEPENDENCY_RETAINED_LITERAL_BYTES,
    MAX_DEPENDENCY_RETAINED_SHELL_IR,
    MAX_DEPENDENCY_SHELL_UNITS_PER_FILE,
    MAX_DEPENDENCY_SHELL_VALUE_BYTES_PER_FILE,
    MAX_DEPENDENCY_SOURCE_MAP_ENTRIES_PER_FILE,
    AssignmentSite,
    CommandProducerReachability,
    CommandResolutionKind,
    CommandSite,
    DependencyFileBudget,
    DependencyWorkBudget,
    DependencyWorkExhaustion,
    DependencyWorkResource,
    GeneratedConfig,
    ShellDialect,
    ShellExtractionResult,
    ShellFrontendResult,
    ShellIssue,
    ShellIssueReason,
    ShellTruncationClaimStatus,
    ShellUnit,
    ShellUnitKind,
    ShellWorkItem,
    ShellWorkOutcome,
    SiteProvenance,
    SourceMap,
    SourceMapEntry,
    SourceSpan,
    StaticValue,
    StaticValueState,
)

if TYPE_CHECKING:
    from tree_sitter import Language, Node, Parser, Tree

EXPECTED_BASH_ABI_VERSION: Final = 15
EXPECTED_BASH_SEMANTIC_VERSION: Final = (0, 25, 1)
MAX_TREE_SITTER_READ_BYTES: Final = 4_096
_GENERATED_UNKNOWN_MARKER: Final = b"SKILLSPECTOR_UNRESOLVED"
_MARKDOWN_SUFFIXES: Final = frozenset({".md", ".markdown", ".mdown", ".mkd"})
_UNSUPPORTED_SHELL_SUFFIXES: Final = frozenset({".zsh", ".envrc", ".ksh"})
_SUPPORTED_FENCE_DIALECTS: Final = {
    b"bash": ShellDialect.BASH,
    b"sh": ShellDialect.SH,
    b"shell-script": ShellDialect.SH,
    b"console": ShellDialect.SH,
}
_SUPPORTED_SHEBANG_DIALECTS: Final = {
    b"bash": ShellDialect.BASH,
    b"sh": ShellDialect.SH,
    b"dash": ShellDialect.DASH,
}
_UNSUPPORTED_SHEBANG_NAMES: Final = frozenset({b"zsh", b"ksh"})


@dataclass(frozen=True, slots=True)
class _PhysicalLine:
    start: int
    content_end: int
    full_end: int
    number: int


def _physical_lines(raw: bytes) -> tuple[_PhysicalLine, ...]:
    """Split only CRLF, LF, and lone CR physical boundaries."""
    lines: list[_PhysicalLine] = []
    line_start = 0
    line_number = 1
    cursor = 0
    while cursor < len(raw):
        byte = raw[cursor]
        if byte == 13:
            full_end = cursor + (2 if cursor + 1 < len(raw) and raw[cursor + 1] == 10 else 1)
        elif byte == 10:
            full_end = cursor + 1
        else:
            cursor += 1
            continue
        lines.append(_PhysicalLine(line_start, cursor, full_end, line_number))
        line_start = full_end
        line_number += 1
        cursor = full_end
    lines.append(_PhysicalLine(line_start, len(raw), len(raw), line_number))
    return tuple(lines)


def _line_starts(lines: tuple[_PhysicalLine, ...]) -> tuple[int, ...]:
    return tuple(line.start for line in lines)


def _span_for_bytes(
    path: str,
    starts: tuple[int, ...],
    start_byte: int,
    end_byte: int,
) -> SourceSpan:
    """Build an inclusive-line span around an exact half-open byte range."""
    start_index = max(0, bisect_right(starts, start_byte) - 1)
    if end_byte > start_byte:
        end_index = max(0, bisect_right(starts, end_byte - 1) - 1)
        end_column = end_byte - starts[end_index]
    else:
        end_index = start_index
        end_column = start_byte - starts[start_index]
    return SourceSpan(
        path,
        start_byte,
        end_byte,
        start_index + 1,
        end_index + 1,
        start_column=start_byte - starts[start_index],
        end_column=end_column,
    )


def _normalized_path(path: object) -> str:
    probe = SourceSpan(
        path,  # type: ignore[arg-type]
        0,
        0,
        1,
        1,
        start_column=0,
        end_column=0,
    )
    if probe.path != path:
        raise ValueError("path must already be normalized")
    return probe.path


def _path_suffix(path: str) -> str:
    basename = path.rsplit("/", 1)[-1].casefold()
    dot = basename.rfind(".")
    return basename[dot:] if dot >= 0 else ""


def _is_unsupported_executable_path(
    path: str,
    raw: bytes,
    lines: tuple[_PhysicalLine, ...],
) -> bool:
    basename = path.rsplit("/", 1)[-1]
    suffix = _path_suffix(path)
    if suffix in _UNSUPPORTED_SHELL_SUFFIXES:
        return True
    if basename == "Dockerfile" or basename.startswith("Dockerfile."):
        for line in lines:
            content = raw[line.start : line.content_end].lstrip(b" \t")
            if content[:3].lower() == b"run" and (len(content) == 3 or content[3] in {9, 32}):
                return True
        return False
    if basename in {"Makefile", "makefile", "GNUmakefile"} or suffix == ".mk":
        return any(raw[line.start : line.content_end].startswith(b"\t") for line in lines)
    return False


def _shebang_dialect(first_line: bytes) -> tuple[ShellDialect | None, bool]:
    if not first_line.startswith(b"#!"):
        return None, False
    words = first_line[2:].strip(b" \t").split()
    if not words:
        return None, False
    executable = words[0].rsplit(b"/", 1)[-1]
    if executable == b"env":
        word_index = 1
        if word_index < len(words) and words[word_index] == b"-S":
            word_index += 1
        if word_index >= len(words):
            return None, False
        executable = words[word_index].rsplit(b"/", 1)[-1]
    dialect = _SUPPORTED_SHEBANG_DIALECTS.get(executable)
    return dialect, executable in _UNSUPPORTED_SHEBANG_NAMES


def _fence_opener(line: bytes) -> tuple[int, int, bytes] | None:
    indentation = 0
    while indentation < len(line) and line[indentation] == 32:
        indentation += 1
    if indentation > 3 or indentation >= len(line):
        return None
    delimiter = line[indentation]
    if delimiter not in {96, 126}:
        return None
    delimiter_end = indentation
    while delimiter_end < len(line) and line[delimiter_end] == delimiter:
        delimiter_end += 1
    length = delimiter_end - indentation
    if length < 3:
        return None
    info = line[delimiter_end:].strip(b" \t")
    token = info.split(maxsplit=1)[0].lower() if info else b""
    return delimiter, length, token


def _is_fence_closer(line: bytes, delimiter: int, minimum_length: int) -> bool:
    indentation = 0
    while indentation < len(line) and line[indentation] == 32:
        indentation += 1
    if indentation > 3:
        return False
    delimiter_end = indentation
    while delimiter_end < len(line) and line[delimiter_end] == delimiter:
        delimiter_end += 1
    return (
        delimiter_end - indentation >= minimum_length and line[delimiter_end:].strip(b" \t") == b""
    )


def _retain_issue(
    issues: list[ShellIssue],
    issue: ShellIssue,
    *,
    file_budget: DependencyFileBudget,
) -> bool:
    exhaustion = file_budget.charge_shell_issues(1)
    if exhaustion is None:
        issues.append(issue)
        return True
    if file_budget.claim_reserved_shell_truncation_issue() is ShellTruncationClaimStatus.CLAIMED:
        issues.append(
            ShellIssue(
                reason=ShellIssueReason.RESOURCE_LIMIT,
                outcome=ShellWorkOutcome.PARTIAL,
                span=issue.span,
                exhaustion=exhaustion,
            )
        )
    return False


def _resource_issue(span: SourceSpan, exhaustion: DependencyWorkExhaustion) -> ShellIssue:
    return ShellIssue(
        reason=ShellIssueReason.RESOURCE_LIMIT,
        outcome=ShellWorkOutcome.PARTIAL,
        span=span,
        exhaustion=exhaustion,
    )


def _reserve_shell_unit(
    file_budget: DependencyFileBudget,
    *,
    source_map_entries: int,
) -> DependencyWorkExhaustion | None:
    next_units = file_budget.used(DependencyWorkResource.SHELL_UNITS) + 1
    next_entries = (
        file_budget.used(DependencyWorkResource.SHELL_SOURCE_MAP_ENTRIES) + source_map_entries
    )
    if next_units > MAX_DEPENDENCY_SHELL_UNITS_PER_FILE:
        return DependencyWorkExhaustion(
            DependencyWorkResource.SHELL_UNITS,
            next_units,
            MAX_DEPENDENCY_SHELL_UNITS_PER_FILE,
        )
    if next_entries > MAX_DEPENDENCY_SOURCE_MAP_ENTRIES_PER_FILE:
        return DependencyWorkExhaustion(
            DependencyWorkResource.SHELL_SOURCE_MAP_ENTRIES,
            next_entries,
            MAX_DEPENDENCY_SOURCE_MAP_ENTRIES_PER_FILE,
        )
    if exhaustion := file_budget.charge_shell_units(1):
        return exhaustion
    if exhaustion := file_budget.charge_source_map_entries(source_map_entries):
        raise RuntimeError("atomic shell-unit reservation invariant failed")
    return None


def _append_unit(
    units: list[ShellUnit],
    issues: list[ShellIssue],
    *,
    path: str,
    raw: bytes,
    starts: tuple[int, ...],
    unit_start: int,
    unit_end: int,
    dialect: ShellDialect,
    kind: ShellUnitKind,
    provenance: SiteProvenance,
    file_budget: DependencyFileBudget,
    mapped: bool,
) -> bool:
    unit_raw = raw[unit_start:unit_end]
    invalid_range: tuple[int, int] | None = None
    try:
        unit_raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        invalid_range = (error.start, max(error.end, error.start + 1))
    entry_count = 1 if invalid_range is None and mapped and unit_raw else 0
    span = _span_for_bytes(path, starts, unit_start, unit_end)
    if exhaustion := _reserve_shell_unit(file_budget, source_map_entries=entry_count):
        _retain_issue(issues, _resource_issue(span, exhaustion), file_budget=file_budget)
        return False
    if invalid_range is not None:
        invalid_start, invalid_end = invalid_range
        invalid_span = _span_for_bytes(
            path,
            starts,
            unit_start + invalid_start,
            unit_start + invalid_end,
        )
        _retain_issue(
            issues,
            ShellIssue(
                reason=ShellIssueReason.SYNTAX_ERROR,
                outcome=ShellWorkOutcome.PARTIAL,
                span=invalid_span,
            ),
            file_budget=file_budget,
        )
        return True
    source_map = (
        SourceMap(
            path=path,
            entries=(SourceMapEntry(0, len(unit_raw), unit_start, unit_end),) if unit_raw else (),
            child_size_bytes=len(unit_raw),
            physical_size_bytes=len(raw),
            physical_line_starts=starts,
        )
        if mapped
        else None
    )
    units.append(
        ShellUnit(
            dialect=dialect,
            kind=kind,
            provenance=provenance,
            raw_bytes=unit_raw,
            origin_span=span,
            source_map=source_map,
        )
    )
    return True


def _unsupported_issue(path: str, starts: tuple[int, ...], raw: bytes) -> ShellIssue:
    return ShellIssue(
        reason=ShellIssueReason.UNSUPPORTED_SEMANTICS,
        outcome=ShellWorkOutcome.PARTIAL,
        span=_span_for_bytes(path, starts, 0, len(raw)),
    )


def _extract_markdown_units(
    *,
    path: str,
    raw: bytes,
    lines: tuple[_PhysicalLine, ...],
    starts: tuple[int, ...],
    file_budget: DependencyFileBudget,
) -> ShellExtractionResult:
    units: list[ShellUnit] = []
    issues: list[ShellIssue] = []
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        opener = _fence_opener(raw[line.start : line.content_end])
        if opener is None:
            line_index += 1
            continue
        delimiter, minimum_length, token = opener
        closer_index: int | None = None
        candidate_index = line_index + 1
        while candidate_index < len(lines):
            candidate = lines[candidate_index]
            if _is_fence_closer(
                raw[candidate.start : candidate.content_end],
                delimiter,
                minimum_length,
            ):
                closer_index = candidate_index
                break
            candidate_index += 1
        content_start = line.full_end
        content_end = lines[closer_index].start if closer_index is not None else len(raw)
        dialect = _SUPPORTED_FENCE_DIALECTS.get(token)
        if dialect is not None:
            retained = _append_unit(
                units,
                issues,
                path=path,
                raw=raw,
                starts=starts,
                unit_start=content_start,
                unit_end=content_end,
                dialect=dialect,
                kind=ShellUnitKind.MARKDOWN_FENCE,
                provenance=SiteProvenance.MARKDOWN_FENCE,
                file_budget=file_budget,
                mapped=True,
            )
            if not retained:
                break
            if closer_index is None:
                issue_span = _span_for_bytes(path, starts, line.start, len(raw))
                unit_id = units[-1].unit_id if units else None
                _retain_issue(
                    issues,
                    ShellIssue(
                        reason=ShellIssueReason.SYNTAX_ERROR,
                        outcome=ShellWorkOutcome.PARTIAL,
                        span=issue_span,
                        unit_id=unit_id,
                    ),
                    file_budget=file_budget,
                )
        elif not token:
            _retain_issue(
                issues,
                ShellIssue(
                    reason=ShellIssueReason.UNSUPPORTED_SEMANTICS,
                    outcome=ShellWorkOutcome.PARTIAL,
                    span=_span_for_bytes(path, starts, line.start, content_end),
                ),
                file_budget=file_budget,
            )
        if closer_index is None:
            break
        line_index = closer_index + 1
    return ShellExtractionResult(units=tuple(units), issues=tuple(issues))


def extract_shell_units(
    path: str,
    raw_bytes: bytes,
    *,
    executable_paths: frozenset[str],
    budget: DependencyWorkBudget,
) -> ShellExtractionResult:
    """Extract bounded applicable units without interpreting shell commands."""
    normalized_path = _normalized_path(path)
    if type(raw_bytes) is not bytes:
        raise TypeError("raw_bytes must be canonical immutable bytes")
    if not isinstance(executable_paths, frozenset):
        raise ValueError("executable_paths must be an immutable set")
    if not isinstance(budget, DependencyWorkBudget):
        raise ValueError("budget must be a DependencyWorkBudget")
    file_budget = budget.for_file(normalized_path)
    file_budget.register_shell_file_size(len(raw_bytes))
    lines = _physical_lines(raw_bytes)
    starts = _line_starts(lines)
    suffix = _path_suffix(normalized_path)
    if suffix in _MARKDOWN_SUFFIXES:
        return _extract_markdown_units(
            path=normalized_path,
            raw=raw_bytes,
            lines=lines,
            starts=starts,
            file_budget=file_budget,
        )
    if _is_unsupported_executable_path(normalized_path, raw_bytes, lines):
        issues: list[ShellIssue] = []
        _retain_issue(
            issues,
            _unsupported_issue(normalized_path, starts, raw_bytes),
            file_budget=file_budget,
        )
        return ShellExtractionResult(issues=tuple(issues))
    first_line = raw_bytes[lines[0].start : lines[0].content_end]
    shebang_dialect, unsupported_shebang = _shebang_dialect(first_line)
    if unsupported_shebang:
        issues = []
        _retain_issue(
            issues,
            _unsupported_issue(normalized_path, starts, raw_bytes),
            file_budget=file_budget,
        )
        return ShellExtractionResult(issues=tuple(issues))
    dialect = shebang_dialect
    provenance = SiteProvenance.SHEBANG
    if dialect is None and suffix in {".sh", ".bash"}:
        dialect = ShellDialect.BASH if suffix == ".bash" else ShellDialect.SH
        provenance = SiteProvenance.FILE_SUFFIX
    if dialect is not None:
        units: list[ShellUnit] = []
        issues = []
        _append_unit(
            units,
            issues,
            path=normalized_path,
            raw=raw_bytes,
            starts=starts,
            unit_start=0,
            unit_end=len(raw_bytes),
            dialect=dialect,
            kind=ShellUnitKind.STANDALONE,
            provenance=provenance,
            file_budget=file_budget,
            mapped=False,
        )
        return ShellExtractionResult(units=tuple(units), issues=tuple(issues))
    if normalized_path in executable_paths:
        issues = []
        _retain_issue(
            issues,
            _unsupported_issue(normalized_path, starts, raw_bytes),
            file_budget=file_budget,
        )
        return ShellExtractionResult(issues=tuple(issues))
    return ShellExtractionResult()


class ShellParserOutcome(StrEnum):
    """Terminal classification for a parser-runtime failure."""

    FAILED = "failed"
    PARTIAL = "partial"


class ShellParserFailureReason(StrEnum):
    """Content-free parser-runtime failure reason."""

    SHELL_PARSER_UNAVAILABLE = "shell_parser_unavailable"
    RUNTIME_LIMIT = "runtime_limit"


class ShellParserError(RuntimeError):
    """A sanitized, classified parser-runtime failure."""

    def __init__(
        self,
        *,
        outcome: ShellParserOutcome,
        reason: ShellParserFailureReason,
        deadline_tripped: bool,
    ) -> None:
        self.outcome = outcome
        self.reason = reason
        self.deadline_tripped = deadline_tripped
        super().__init__(f"{outcome.value}: {reason.value}")


def _unavailable_error(*, meaningful_work: bool = False) -> ShellParserError:
    return ShellParserError(
        outcome=(ShellParserOutcome.PARTIAL if meaningful_work else ShellParserOutcome.FAILED),
        reason=ShellParserFailureReason.SHELL_PARSER_UNAVAILABLE,
        deadline_tripped=False,
    )


@cache
def load_bash_language() -> Language:
    """Load and validate the exact Bash grammar once, on first use."""
    try:
        tree_sitter = import_module("tree_sitter")
        tree_sitter_bash = import_module("tree_sitter_bash")
        language = cast("Language", tree_sitter.Language(tree_sitter_bash.language()))
        if language.abi_version != EXPECTED_BASH_ABI_VERSION:
            raise ValueError("unexpected Bash grammar ABI")
        if language.semantic_version != EXPECTED_BASH_SEMANTIC_VERSION:
            raise ValueError("unexpected Bash grammar semantic version")
    except ShellParserError:
        raise
    except Exception:
        raise _unavailable_error() from None
    return language


def create_bash_parser(*, deadline_monotonic: float | None = None) -> Parser:
    """Create one fresh parser around the cached immutable language."""
    try:
        tree_sitter = import_module("tree_sitter")
        language = load_bash_language()
        timeout_micros = _native_timeout_micros(deadline_monotonic)
        if timeout_micros is None:
            return cast("Parser", tree_sitter.Parser(language))
        with catch_warnings():
            filterwarnings(
                "ignore",
                message=r"Use the progress_callback in parse\(\)",
                category=DeprecationWarning,
            )
            return cast(
                "Parser",
                tree_sitter.Parser(
                    language,
                    timeout_micros=timeout_micros,
                ),
            )
    except ShellParserError:
        raise
    except Exception:
        raise _unavailable_error() from None


def _bounded_reader(
    source: bytes,
    *,
    deadline_expired: Callable[[], bool],
) -> Callable[[int, Any], bytes]:
    if type(source) is not bytes:
        raise TypeError("source must be immutable bytes")

    def read(byte_offset: int, _position: Any) -> bytes:
        if deadline_expired():
            return b""
        if byte_offset < 0 or byte_offset >= len(source):
            return b""
        return source[byte_offset : byte_offset + MAX_TREE_SITTER_READ_BYTES]

    return read


def _native_timeout_micros(deadline_monotonic: float | None) -> int | None:
    if deadline_monotonic is None:
        return None
    if isinstance(deadline_monotonic, bool) or not isinstance(deadline_monotonic, (int, float)):
        raise TypeError("deadline_monotonic must be a finite monotonic timestamp")
    if not isfinite(deadline_monotonic):
        raise ValueError("deadline_monotonic must be finite")
    return max(1, ceil((deadline_monotonic - monotonic()) * 1_000_000))


def parse_bash_source(
    source: bytes,
    *,
    deadline_monotonic: float | None = None,
    meaningful_work: bool = False,
) -> Tree:
    """Parse immutable bytes through a bounded reader with classified failures."""
    deadline_tripped = False

    def deadline_expired() -> bool:
        nonlocal deadline_tripped
        if deadline_monotonic is not None and monotonic() >= deadline_monotonic:
            deadline_tripped = True
        return deadline_tripped

    try:
        reader = _bounded_reader(source, deadline_expired=deadline_expired)
        parser = create_bash_parser(deadline_monotonic=deadline_monotonic)
        tree = parser.parse(reader)
    except ShellParserError as error:
        if meaningful_work and error.reason is ShellParserFailureReason.SHELL_PARSER_UNAVAILABLE:
            raise _unavailable_error(meaningful_work=True) from None
        raise
    except Exception:
        if deadline_expired():
            raise ShellParserError(
                outcome=ShellParserOutcome.PARTIAL,
                reason=ShellParserFailureReason.RUNTIME_LIMIT,
                deadline_tripped=True,
            ) from None
        raise _unavailable_error(meaningful_work=meaningful_work) from None

    if deadline_expired():
        raise ShellParserError(
            outcome=ShellParserOutcome.PARTIAL,
            reason=ShellParserFailureReason.RUNTIME_LIMIT,
            deadline_tripped=True,
        )
    if tree is None:
        raise _unavailable_error(meaningful_work=meaningful_work)
    return tree


class _ExecutionRegionKind(StrEnum):
    """Code-owned structural regions retained for later same-parse analysis."""

    PROGRAM = "program"
    LIST = "list"
    PIPELINE = "pipeline"
    PIPELINE_STAGE = "pipeline_stage"
    IF = "if"
    ELIF = "elif"
    ELSE = "else"
    FOR = "for"
    C_STYLE_FOR = "c_style_for"
    WHILE = "while"
    UNTIL = "until"
    DO = "do"
    CASE = "case"
    CASE_ITEM = "case_item"
    FUNCTION = "function"
    COMPOUND = "compound"
    SUBSHELL = "subshell"
    COMMAND_SUBSTITUTION = "command_substitution"
    PROCESS_SUBSTITUTION = "process_substitution"
    NEGATION = "negation"
    REDIRECTED = "redirected"
    ASYNC = "async"


class _RedirectKind(StrEnum):
    """Narrow syntax-proven redirect facts; deliberately not an FD model."""

    STDIN_HEREDOC = "stdin_heredoc"
    STDIN_HERE_STRING = "stdin_here_string"
    STDOUT_TRUNCATE = "stdout_truncate"
    STDOUT_CLOBBER = "stdout_clobber"
    STDOUT_APPEND = "stdout_append"
    STDOUT_STDERR_TRUNCATE = "stdout_stderr_truncate"


class _ValueAtomKind(StrEnum):
    LITERAL = "literal"
    VARIABLE = "variable"


class _ExportState(StrEnum):
    EXPORTED = "exported"
    UNEXPORTED = "unexported"
    UNKNOWN = "unknown"


class _CommandResolutionKind(StrEnum):
    EXTERNAL = "external"
    FUNCTION = "function"
    AMBIGUOUS = "ambiguous"


_IMPORTED_FUNCTION_ID: Final = -1


class _ShellEventKind(StrEnum):
    ASSIGNMENT = "assignment"
    COMMAND = "command"
    UNSET = "unset"
    FUNCTION_DEFINITION = "function_definition"
    LOOP_BINDING = "loop_binding"
    LOOP_UPDATE = "loop_update"


class _ControlRole(StrEnum):
    STRAIGHT = "straight"
    CONDITIONAL = "conditional"
    BOOLEAN_LEFT = "boolean_left"
    BOOLEAN_RIGHT = "boolean_right"
    LOOP = "loop"
    GROUP = "group"
    ASYNC = "async"
    ISOLATED = "isolated"
    FUNCTION = "function"


class _NestedExecutionKind(StrEnum):
    SHELL = "shell"
    EVAL = "eval"


_REGION_NODE_TYPES: Final[dict[str, _ExecutionRegionKind]] = {
    "program": _ExecutionRegionKind.PROGRAM,
    "list": _ExecutionRegionKind.LIST,
    "pipeline": _ExecutionRegionKind.PIPELINE,
    "if_statement": _ExecutionRegionKind.IF,
    "elif_clause": _ExecutionRegionKind.ELIF,
    "else_clause": _ExecutionRegionKind.ELSE,
    "for_statement": _ExecutionRegionKind.FOR,
    "c_style_for_statement": _ExecutionRegionKind.C_STYLE_FOR,
    "while_statement": _ExecutionRegionKind.WHILE,
    "until_statement": _ExecutionRegionKind.UNTIL,
    "do_group": _ExecutionRegionKind.DO,
    "case_statement": _ExecutionRegionKind.CASE,
    "case_item": _ExecutionRegionKind.CASE_ITEM,
    "function_definition": _ExecutionRegionKind.FUNCTION,
    "compound_statement": _ExecutionRegionKind.COMPOUND,
    "subshell": _ExecutionRegionKind.SUBSHELL,
    "command_substitution": _ExecutionRegionKind.COMMAND_SUBSTITUTION,
    "process_substitution": _ExecutionRegionKind.PROCESS_SUBSTITUTION,
    "negated_command": _ExecutionRegionKind.NEGATION,
    "redirected_statement": _ExecutionRegionKind.REDIRECTED,
}
_SUBSTITUTION_NODE_TYPES: Final = frozenset({"command_substitution", "process_substitution"})
_DYNAMIC_VALUE_NODE_TYPES: Final = frozenset(
    {
        "simple_expansion",
        "expansion",
        "parameter_expansion",
        "arithmetic_expansion",
        "command_substitution",
        "process_substitution",
    }
)
_DECLARATION_KEYWORDS: Final = frozenset({"declare", "export", "local", "readonly", "typeset"})
_DECLARATION_KEYWORD_BYTES: Final[frozenset[bytes]] = frozenset(
    keyword.encode("ascii") for keyword in _DECLARATION_KEYWORDS
)
_SUPPORTED_REDIRECT_OPERATORS: Final = frozenset({">", ">|", ">>", "&>"})
_COMPOUND_REDIRECT_BODIES: Final = frozenset(
    {
        "compound_statement",
        "subshell",
        "if_statement",
        "for_statement",
        "c_style_for_statement",
        "while_statement",
        "until_statement",
        "case_statement",
        "function_definition",
    }
)
_PINNED_CST_FIELDS: Final[dict[str, frozenset[str]]] = {
    "program": frozenset(),
    "command": frozenset({"name", "argument", "redirect"}),
    "variable_assignment": frozenset({"name", "value"}),
    "declaration_command": frozenset(),
    "unset_command": frozenset(),
    "redirected_statement": frozenset({"body", "redirect"}),
    "file_redirect": frozenset({"descriptor", "destination"}),
    "function_definition": frozenset({"name", "body"}),
    "if_statement": frozenset({"condition"}),
    "for_statement": frozenset({"variable", "value", "body"}),
    "c_style_for_statement": frozenset({"initializer", "condition", "update", "body"}),
    "postfix_expression": frozenset({"operator"}),
    "while_statement": frozenset({"condition", "body"}),
    "case_statement": frozenset({"value"}),
    "case_item": frozenset({"value", "termination"}),
    "pipeline": frozenset(),
    "list": frozenset(),
    "negated_command": frozenset(),
    "subshell": frozenset(),
    "compound_statement": frozenset(),
    "command_substitution": frozenset(),
    "process_substitution": frozenset(),
    "heredoc_redirect": frozenset({"argument"}),
    "ERROR": frozenset(),
    "MISSING": frozenset(),
}


@dataclass(frozen=True, slots=True)
class _ValueFragment:
    """A physical source fragment mapped into one retained logical value."""

    span: SourceSpan
    value_start_byte: int
    value_end_byte: int
    exact: bool
    unit_id: str = field(repr=False)
    local_start_byte: int = field(repr=False)
    local_end_byte: int = field(repr=False)


@dataclass(frozen=True, slots=True)
class _ValueAtom:
    kind: _ValueAtomKind
    value: StaticValue
    span: SourceSpan
    fragments: tuple[_ValueFragment, ...]
    name: str | None = field(default=None, repr=False)
    quoted: bool = False


@dataclass(frozen=True, slots=True)
class _GeneratedProofEntry:
    child_start_byte: int
    child_end_byte: int
    physical_start_byte: int
    physical_end_byte: int


@dataclass(frozen=True, slots=True)
class _GeneratedValueProof:
    raw_bytes: bytes = field(repr=False)
    entries: tuple[_GeneratedProofEntry, ...] = field(repr=False)
    unknown_ranges: tuple[tuple[int, int], ...] = field(repr=False)
    unknowns_quoted: bool
    path: str
    physical_size_bytes: int
    physical_line_starts: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _ProvenGeneratedConfig(GeneratedConfig):
    """Private proof carried beside the stable public generated-config fields."""

    target_proof: _GeneratedValueProof | None = field(default=None, repr=False, compare=False)
    home_relative_target: bytes | None = field(default=None, repr=False, compare=False)
    content_proof: _GeneratedValueProof | None = field(default=None, repr=False, compare=False)
    physical_size_bytes: int = field(default=0, repr=False, compare=False)
    physical_line_starts: tuple[int, ...] = field(default=(), repr=False, compare=False)
    producer_program_id: str = field(default="", repr=False, compare=False)
    producer_command_start_byte: int | None = field(default=None, repr=False, compare=False)
    producer_function_id: int | None = field(default=None, repr=False, compare=False)
    execution_limitation_span: SourceSpan | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True, slots=True)
class _BindingIR:
    name: str = field(repr=False)
    value: StaticValue
    export_state: _ExportState
    fragments: tuple[_ValueFragment, ...]


@dataclass(frozen=True, slots=True)
class _StateUpdateIR:
    order: int
    region_id: int | None
    function_id: int | None
    frame_id: int
    binding: _BindingIR


@dataclass(frozen=True, slots=True)
class _StateFrameFact:
    frame_id: int
    parent_frame_id: int | None
    functions_unknown: bool


@dataclass(frozen=True, slots=True)
class _StateEventIR:
    order: int
    kind: _ShellEventKind
    role: _ControlRole
    span: SourceSpan
    region_id: int | None
    function_id: int | None
    local_start_byte: int = field(repr=False)
    local_end_byte: int = field(repr=False)
    name: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _CommandResolution:
    kind: _CommandResolutionKind
    function_id: int | None = None


@dataclass(frozen=True, slots=True)
class _ArgumentIR:
    value: StaticValue
    span: SourceSpan
    fragments: tuple[_ValueFragment, ...]
    atoms: tuple[_ValueAtom, ...]
    local_start_byte: int = field(repr=False)
    local_end_byte: int = field(repr=False)
    home_relative_target: bytes | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _ExecutionRegion:
    region_id: int
    order: int
    kind: _ExecutionRegionKind
    span: SourceSpan
    parent_region_id: int | None
    function_id: int | None
    local_start_byte: int = field(repr=False)
    local_end_byte: int = field(repr=False)
    control_split_byte: int | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _FunctionContext:
    function_id: int
    name: StaticValue
    span: SourceSpan
    fragments: tuple[_ValueFragment, ...]
    parent_function_id: int | None
    definition_region_id: int | None
    local_start_byte: int = field(repr=False)
    local_end_byte: int = field(repr=False)


@dataclass(frozen=True, slots=True)
class _RedirectFact:
    kind: _RedirectKind
    target: _ArgumentIR
    span: SourceSpan
    local_start_byte: int = field(default=0, repr=False)
    expands: bool = field(default=False, repr=False)
    complete: bool = field(default=True, repr=False)
    value_proof: _GeneratedValueProof | None = field(default=None, repr=False)
    execution_limitation_span: SourceSpan | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _AssignmentIR:
    site: AssignmentSite
    order: int
    region_id: int | None
    function_id: int | None
    prefix_for_command_start_byte: int | None
    value_fragments: tuple[_ValueFragment, ...]
    value_atoms: tuple[_ValueAtom, ...]
    declaration_keyword: bytes | None = field(default=None, repr=False)
    declaration_command_start_byte: int | None = field(default=None, repr=False)
    local_start_byte: int = field(default=0, repr=False)
    local_end_byte: int = field(default=0, repr=False)


@dataclass(frozen=True, slots=True)
class _CommandIR:
    site: CommandSite
    order: int
    region_id: int | None
    function_id: int | None
    arguments: tuple[_ArgumentIR, ...]
    prefix_assignments: tuple[AssignmentSite, ...]
    redirects: tuple[_RedirectFact, ...]
    redirects_complete: bool = True
    prefix_bindings: tuple[_BindingIR, ...] = ()
    program_id: str = field(default="", repr=False)
    containing_function_program_id: str | None = field(default=None, repr=False)
    containing_function_id: int | None = field(default=None, repr=False)
    execution_order: int = 0
    state_update_order: int = 0
    state_frame_id: int = 0
    resolution: _CommandResolution = _CommandResolution(_CommandResolutionKind.EXTERNAL)
    local_start_byte: int = field(default=0, repr=False)
    local_end_byte: int = field(default=0, repr=False)


@dataclass(frozen=True, slots=True)
class _ShellProgramIR:
    program_id: str = field(default="", repr=False)
    regions: tuple[_ExecutionRegion, ...] = ()
    functions: tuple[_FunctionContext, ...] = ()
    commands: tuple[_CommandIR, ...] = ()
    assignments: tuple[_AssignmentIR, ...] = ()
    state_updates: tuple[_StateUpdateIR, ...] = ()
    state_events: tuple[_StateEventIR, ...] = ()
    state_frames: tuple[_StateFrameFact, ...] = ()
    initial_bindings: tuple[_BindingIR, ...] = ()
    initial_functions: tuple[tuple[bytes, int | None], ...] = field(default=(), repr=False)
    initial_functions_unknown: bool = False
    execution_commands: tuple[_CommandIR, ...] = ()
    nested_programs: tuple[_NestedProgramIR, ...] = ()
    generated_configs: tuple[GeneratedConfig, ...] = ()
    recovered_generated_drafts: tuple[_RecoveredGeneratedDraft, ...] = field(default=(), repr=False)


@dataclass(frozen=True, slots=True)
class _NestedProgramIR:
    order: int
    unit: ShellUnit
    depth: int
    execution_kind: _NestedExecutionKind
    parent_program_id: str
    parent_command_start_byte: int
    program: _ShellProgramIR
    outcome: ShellWorkOutcome


@dataclass(frozen=True, slots=True)
class _NestedRequest:
    execution_kind: _NestedExecutionKind
    command: _CommandIR
    payload: _ArgumentIR
    frame: _StateFrame = field(repr=False, compare=False)
    depth: int


@dataclass(frozen=True, slots=True)
class _NestedResponse:
    bindings: dict[str, _BindingIR] | None = field(default=None, repr=False)
    functions: dict[bytes, int | None] | None = field(default=None, repr=False)
    functions_unknown: bool = False


@dataclass(frozen=True, slots=True)
class _ModeledState:
    program: _ShellProgramIR
    bindings: dict[str, _BindingIR] = field(repr=False)
    functions: dict[bytes, int | None] = field(repr=False)
    functions_unknown: bool


@dataclass(slots=True)
class _ProgramJob:
    lowerer: _ShellLowerer
    unit: ShellUnit
    depth: int
    execution_kind: _NestedExecutionKind | None
    parent_request: _NestedRequest | None
    nested_order: int | None
    initial_bindings: dict[str, _BindingIR]
    initial_functions: dict[bytes, int | None]
    initial_functions_unknown: bool
    generator: Generator[_NestedRequest, _NestedResponse, _ModeledState]
    started: bool = False
    pending_response: _NestedResponse | None = None


@dataclass(frozen=True, slots=True)
class _ShellAnalysisResult:
    public: ShellFrontendResult
    program: _ShellProgramIR


@dataclass(frozen=True, slots=True)
class _FoldedValue:
    value: StaticValue
    fragments: tuple[tuple[int, int, int, int, bool], ...]
    unquoted_tilde_offsets: tuple[int, ...] = ()
    unquoted_assignment_delimiter_offsets: tuple[int, ...] = ()
    quoted_empty_offsets: tuple[int, ...] = ()
    atoms: tuple[_FoldAtom, ...] | None = None


@dataclass(frozen=True, slots=True)
class _FoldAtom:
    kind: _ValueAtomKind
    value: StaticValue
    fragments: tuple[tuple[int, int, int, int, bool], ...]
    source_start_byte: int
    source_end_byte: int
    name: str | None = field(default=None, repr=False)
    quoted: bool = False


@dataclass(frozen=True, slots=True)
class _NodeGroup:
    nodes: tuple[Node, ...]
    start_byte: int
    end_byte: int
    raw_syntax: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class _CommandDraft:
    node: Node = field(repr=False)
    region_id: int | None
    function_id: int | None
    substitution_depth: int
    non_file_redirect_count: int


@dataclass(frozen=True, slots=True)
class _AssignmentDraft:
    node: Node = field(repr=False)
    region_id: int | None
    function_id: int | None
    prefix_for_command_start_byte: int | None
    declaration_command_start_byte: int | None


@dataclass(frozen=True, slots=True)
class _FunctionDraft:
    node: Node = field(repr=False)
    function_id: int
    parent_function_id: int | None


@dataclass(frozen=True, slots=True)
class _RedirectOwner:
    node: Node = field(repr=False)
    statement_start_byte: int
    body_start_byte: int
    body_end_byte: int
    body_type: str
    substitution_depth: int
    has_error_child: bool
    redirect_count: int


@dataclass(frozen=True, slots=True)
class _RedirectDraft:
    node: Node = field(repr=False)
    command_start_byte: int | None
    statement_start_byte: int | None


@dataclass(frozen=True, slots=True)
class _RecoveredGeneratedDraft:
    input_fact: _RedirectFact
    output_fact: _RedirectFact
    function_id: int | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _WrapperArgumentsDraft:
    nodes: tuple[Node, ...] = field(repr=False)
    statement_start_byte: int


@dataclass(frozen=True, slots=True)
class _ShellEventDraft:
    kind: _ShellEventKind
    node: Node = field(repr=False)
    region_id: int | None
    function_id: int | None


@dataclass(slots=True)
class _StateFrame:
    frame_id: int
    parent: _StateFrame | None
    bindings: dict[str, _BindingIR] = field(default_factory=dict)
    functions: dict[bytes, int | None] = field(default_factory=dict)
    functions_unknown: bool = False
    sticky_unknown_names: set[str] = field(default_factory=set, repr=False)
    persistent_unknown_names: set[str] = field(default_factory=set, repr=False)
    persistent_bindings_unknown: bool = field(default=False, repr=False)


@dataclass(frozen=True, slots=True)
class _WalkFrame:
    node: Node = field(repr=False)
    parent_type: str | None
    parent_start_byte: int | None
    region_id: int | None
    function_id: int | None
    substitution_depth: int
    command_owner_start_byte: int | None
    redirect_owner_start_byte: int | None
    exiting: bool = False


def _node_key(node: Node) -> tuple[int, int, str]:
    return node.start_byte, node.end_byte, node.type


def _shell_work_item(unit: ShellUnit, outcome: ShellWorkOutcome) -> ShellWorkItem:
    return ShellWorkItem(
        unit_id=unit.unit_id,
        dialect=unit.dialect,
        kind=unit.kind,
        provenance=unit.provenance,
        span=unit.origin_span,
        outcome=outcome,
    )


class _ShellLowerer:
    """One bounded, iterative CST walk plus typed syntax-only projection."""

    def __init__(
        self,
        unit: ShellUnit,
        budget: DependencyWorkBudget,
        file_budget: DependencyFileBudget,
        *,
        accounting_unit: ShellUnit | None = None,
    ) -> None:
        self.unit = unit
        self.accounting_unit = accounting_unit or unit
        self.raw = unit.raw_bytes
        self.budget = budget
        self.file_budget = file_budget
        self.local_lines = _physical_lines(self.raw)
        self.local_line_starts = _line_starts(self.local_lines)
        self.issues: list[ShellIssue] = []
        self.regions: list[_ExecutionRegion] = []
        self.function_drafts: list[_FunctionDraft] = []
        self.command_drafts: list[_CommandDraft] = []
        self.assignment_drafts: list[_AssignmentDraft] = []
        self.redirect_owners: dict[int, _RedirectOwner] = {}
        self.redirect_drafts: list[_RedirectDraft] = []
        self.heredoc_facts_by_statement: dict[int, list[_RedirectFact]] = {}
        self.recovered_generated_drafts: list[_RecoveredGeneratedDraft] = []
        self.consumed_error_nodes: set[tuple[int, int, str]] = set()
        self.consumed_redirect_nodes: set[tuple[int, int, str]] = set()
        self.masked_data_ranges: list[tuple[int, int]] = []
        self.wrapper_argument_drafts: list[_WrapperArgumentsDraft] = []
        self.event_drafts: list[_ShellEventDraft] = []
        self.region_chain_cache: dict[int | None, tuple[_ExecutionRegion, ...]] = {}
        self.branch_container_ids: dict[int, int] = {}
        self.pipeline_input_region_ids: frozenset[int] | None = None
        self.persistent_unknown_function_ids: frozenset[int] = frozenset()
        self.next_state_frame_id = 1
        self.syntax_error_command_starts: set[int] = set()
        self.syntax_error_redirect_starts: set[int] = set()
        self.halted = False
        self.partial = False

    def _identity_span(self, start_byte: int, end_byte: int) -> SourceSpan:
        local = _span_for_bytes(
            self.unit.origin_span.path,
            self.local_line_starts,
            start_byte,
            end_byte,
        )
        origin_column = self.unit.origin_span.start_column or 0
        start_column = local.start_column
        end_column = local.end_column
        if local.start_line == 1 and start_column is not None:
            start_column += origin_column
        if local.end_line == 1 and end_column is not None:
            end_column += origin_column
        return SourceSpan(
            local.path,
            self.unit.origin_span.start_byte + local.start_byte,
            self.unit.origin_span.start_byte + local.end_byte,
            self.unit.origin_span.start_line + local.start_line - 1,
            self.unit.origin_span.start_line + local.end_line - 1,
            start_column=start_column,
            end_column=end_column,
        )

    def _span(self, start_byte: int, end_byte: int) -> SourceSpan | None:
        if end_byte <= start_byte:
            return None
        if self.unit.source_map is None:
            return self._identity_span(start_byte, end_byte)
        return self.unit.source_map.map_range(start_byte, end_byte)

    def _point_span(self, byte_offset: int) -> SourceSpan:
        bounded = min(max(byte_offset, 0), len(self.raw))
        if self.unit.source_map is None:
            return self._identity_span(bounded, bounded)
        if bounded < len(self.raw):
            following = self.unit.source_map.map_range(bounded, bounded + 1)
            if following is not None:
                return replace(
                    following, end_byte=following.start_byte, end_column=following.start_column
                )
        if bounded > 0:
            preceding = self.unit.source_map.map_range(bounded - 1, bounded)
            if preceding is not None:
                return SourceSpan(
                    preceding.path,
                    preceding.end_byte,
                    preceding.end_byte,
                    preceding.end_line,
                    preceding.end_line,
                    start_column=preceding.end_column,
                    end_column=preceding.end_column,
                )
        return replace(
            self.unit.origin_span,
            end_byte=self.unit.origin_span.start_byte,
            end_line=self.unit.origin_span.start_line,
            end_column=self.unit.origin_span.start_column,
        )

    def _node_span(self, node: Node) -> SourceSpan | None:
        return self._span(node.start_byte, node.end_byte)

    def _issue(
        self,
        reason: ShellIssueReason,
        span: SourceSpan,
        *,
        outcome: ShellWorkOutcome = ShellWorkOutcome.PARTIAL,
        exhaustion: DependencyWorkExhaustion | None = None,
    ) -> None:
        self.partial = True
        issue = ShellIssue(
            reason=reason,
            outcome=outcome,
            span=span,
            unit_id=self.unit.unit_id,
            exhaustion=exhaustion,
        )
        if issue in self.issues:
            return
        _retain_issue(
            self.issues,
            issue,
            file_budget=self.file_budget,
        )

    def _resource(self, exhaustion: DependencyWorkExhaustion, span: SourceSpan) -> None:
        self._issue(ShellIssueReason.RESOURCE_LIMIT, span, exhaustion=exhaustion)
        self.halted = True

    def _reserve(self, count: int, *, value_bytes: int = 0, span: SourceSpan) -> bool:
        next_ir = self.budget.used(DependencyWorkResource.RETAINED_SHELL_IR) + count
        if next_ir > MAX_DEPENDENCY_RETAINED_SHELL_IR:
            self._resource(
                DependencyWorkExhaustion(
                    DependencyWorkResource.RETAINED_SHELL_IR,
                    next_ir,
                    MAX_DEPENDENCY_RETAINED_SHELL_IR,
                ),
                span,
            )
            return False
        next_file_values = (
            self.file_budget.used(DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES) + value_bytes
        )
        if next_file_values > MAX_DEPENDENCY_SHELL_VALUE_BYTES_PER_FILE:
            self._resource(
                DependencyWorkExhaustion(
                    DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES,
                    next_file_values,
                    MAX_DEPENDENCY_SHELL_VALUE_BYTES_PER_FILE,
                ),
                span,
            )
            return False
        next_global_values = (
            self.budget.used(DependencyWorkResource.RETAINED_LITERAL_BYTES) + value_bytes
        )
        if next_global_values > MAX_DEPENDENCY_RETAINED_LITERAL_BYTES:
            self._resource(
                DependencyWorkExhaustion(
                    DependencyWorkResource.RETAINED_LITERAL_BYTES,
                    next_global_values,
                    MAX_DEPENDENCY_RETAINED_LITERAL_BYTES,
                ),
                span,
            )
            return False
        if self.file_budget.charge_retained_shell_ir(self.accounting_unit, count) is not None:
            raise RuntimeError("atomic retained shell IR reservation invariant failed")
        if self.file_budget.reserve_shell_value_bytes(value_bytes) is not None:
            raise RuntimeError("atomic retained shell value reservation invariant failed")
        return True

    def _mapped_fragments(
        self,
        fragments: tuple[tuple[int, int, int, int, bool], ...],
        *,
        fallback_start: int,
    ) -> tuple[_ValueFragment, ...] | None:
        mapped: list[_ValueFragment] = []
        for source_start, source_end, value_start, value_end, exact in fragments:
            span = self._span(source_start, source_end)
            if span is None:
                self._issue(
                    ShellIssueReason.UNSUPPORTED_SEMANTICS,
                    self._point_span(source_start if source_start >= 0 else fallback_start),
                )
                return None
            mapped.append(
                _ValueFragment(
                    span,
                    value_start,
                    value_end,
                    exact,
                    self.unit.unit_id,
                    source_start,
                    source_end,
                )
            )
        return tuple(mapped)

    def _mapped_atoms(
        self,
        atoms: tuple[_FoldAtom, ...] | None,
    ) -> tuple[_ValueAtom, ...] | None:
        if atoms is None:
            return ()
        mapped: list[_ValueAtom] = []
        for atom in atoms:
            span = self._span(atom.source_start_byte, atom.source_end_byte)
            if span is None:
                self._issue(
                    ShellIssueReason.UNSUPPORTED_SEMANTICS,
                    self._point_span(atom.source_start_byte),
                )
                return None
            fragments = self._mapped_fragments(
                atom.fragments,
                fallback_start=atom.source_start_byte,
            )
            if fragments is None:
                return None
            mapped.append(
                _ValueAtom(
                    atom.kind,
                    atom.value,
                    span,
                    fragments,
                    atom.name,
                    atom.quoted,
                )
            )
        return tuple(mapped)

    @staticmethod
    def _append_surviving_fragment(
        fragments: list[tuple[int, int, int, int, bool]],
        *,
        source_byte: int,
        value_byte: int,
    ) -> None:
        if fragments and fragments[-1][1] == source_byte and fragments[-1][3] == value_byte:
            source_start, _source_end, value_start, _value_end, exact = fragments[-1]
            fragments[-1] = (
                source_start,
                source_byte + 1,
                value_start,
                value_byte + 1,
                exact,
            )
            return
        fragments.append((source_byte, source_byte + 1, value_byte, value_byte + 1, True))

    @staticmethod
    def _decode_unquoted(
        raw: bytes,
        *,
        source_start_byte: int,
    ) -> tuple[
        bytes | None,
        bool,
        tuple[int, ...],
        tuple[int, ...],
        tuple[tuple[int, int, int, int, bool], ...],
    ]:
        decoded = bytearray()
        has_pathname_expansion = False
        tilde_offsets: list[int] = []
        assignment_delimiter_offsets: list[int] = []
        fragments: list[tuple[int, int, int, int, bool]] = []
        index = 0
        while index < len(raw):
            byte = raw[index]
            if byte == 92:
                if index + 1 >= len(raw):
                    return None, False, (), (), ()
                following = raw[index + 1]
                if following == 10:
                    index += 2
                    continue
                value_offset = len(decoded)
                decoded.append(following)
                _ShellLowerer._append_surviving_fragment(
                    fragments,
                    source_byte=source_start_byte + index + 1,
                    value_byte=value_offset,
                )
                index += 2
                continue
            if byte in {0, 10, 12, 13}:
                return None, False, (), (), ()
            if byte in {42, 63, 91}:
                has_pathname_expansion = True
            elif byte == 126:
                tilde_offsets.append(len(decoded))
            elif byte == 58:
                assignment_delimiter_offsets.append(len(decoded))
            value_offset = len(decoded)
            decoded.append(byte)
            _ShellLowerer._append_surviving_fragment(
                fragments,
                source_byte=source_start_byte + index,
                value_byte=value_offset,
            )
            index += 1
        return (
            bytes(decoded),
            has_pathname_expansion,
            tuple(tilde_offsets),
            tuple(assignment_delimiter_offsets),
            tuple(fragments),
        )

    @staticmethod
    def _decode_double_quoted(
        raw: bytes,
        *,
        source_start_byte: int,
    ) -> tuple[bytes | None, tuple[tuple[int, int, int, int, bool], ...]]:
        decoded = bytearray()
        fragments: list[tuple[int, int, int, int, bool]] = []
        index = 0
        while index < len(raw):
            byte = raw[index]
            if byte == 92 and index + 1 < len(raw):
                following = raw[index + 1]
                if following == 10:
                    index += 2
                    continue
                if following in {36, 96, 34, 92}:
                    value_offset = len(decoded)
                    decoded.append(following)
                    _ShellLowerer._append_surviving_fragment(
                        fragments,
                        source_byte=source_start_byte + index + 1,
                        value_byte=value_offset,
                    )
                    index += 2
                    continue
            if byte in {0, 10, 13}:
                return None, ()
            value_offset = len(decoded)
            decoded.append(byte)
            _ShellLowerer._append_surviving_fragment(
                fragments,
                source_byte=source_start_byte + index,
                value_byte=value_offset,
            )
            index += 1
        return bytes(decoded), tuple(fragments)

    @staticmethod
    def _unknown_fold(node: Node) -> _FoldedValue:
        return _FoldedValue(
            StaticValue.unknown(),
            ((node.start_byte, node.end_byte, 0, 0, False),),
        )

    def _simple_variable_reference(self, node: Node) -> str | None:
        children = node.children
        if node.type == "simple_expansion":
            if len(children) != 2 or children[0].type != "$" or children[1].type != "variable_name":
                return None
            variable = children[1]
        elif node.type == "expansion":
            if (
                len(children) != 3
                or children[0].type != "${"
                or children[1].type != "variable_name"
                or children[2].type != "}"
            ):
                return None
            variable = children[1]
        else:
            return None
        return self._identifier(self.raw[variable.start_byte : variable.end_byte])

    @staticmethod
    def _apply_tilde_context(
        folded: _FoldedValue,
        *,
        assignment_value: bool,
    ) -> _FoldedValue:
        if folded.value.state is not StaticValueState.EXACT:
            return folded
        dynamic = False
        delimiters = folded.unquoted_assignment_delimiter_offsets
        quoted_empty_offsets = folded.quoted_empty_offsets
        for offset in folded.unquoted_tilde_offsets:
            quote_index = bisect_left(quoted_empty_offsets, offset)
            if (
                quote_index < len(quoted_empty_offsets)
                and quoted_empty_offsets[quote_index] == offset
            ):
                continue
            if offset == 0:
                dynamic = True
                break
            if assignment_value:
                delimiter_index = bisect_left(delimiters, offset - 1)
                if delimiter_index < len(delimiters) and delimiters[delimiter_index] == offset - 1:
                    dynamic = True
                    break
        if not dynamic:
            return folded
        return _FoldedValue(
            StaticValue.unknown(),
            tuple(
                (source_start, source_end, 0, 0, False)
                for source_start, source_end, _value_start, _value_end, _exact in folded.fragments
            ),
        )

    def _fold_node(self, root: Node) -> _FoldedValue:
        results: dict[tuple[int, int, str], _FoldedValue] = {}
        stack: list[tuple[Node, bool]] = [(root, False)]
        while stack:
            node, visited = stack.pop()
            key = _node_key(node)
            if not visited:
                if node.type in {"simple_expansion", "expansion"}:
                    name = self._simple_variable_reference(node)
                    if name is None:
                        results[key] = self._unknown_fold(node)
                    else:
                        results[key] = _FoldedValue(
                            StaticValue.unknown(),
                            ((node.start_byte, node.end_byte, 0, 0, False),),
                            atoms=(
                                _FoldAtom(
                                    _ValueAtomKind.VARIABLE,
                                    StaticValue.unknown(),
                                    (),
                                    node.start_byte,
                                    node.end_byte,
                                    name=name,
                                ),
                            ),
                        )
                    continue
                if node.type in _DYNAMIC_VALUE_NODE_TYPES:
                    results[key] = self._unknown_fold(node)
                    continue
                stack.append((node, True))
                stack.extend((child, False) for child in reversed(node.children))
                continue
            node_type = node.type
            raw = self.raw[node.start_byte : node.end_byte]
            if node_type in _DYNAMIC_VALUE_NODE_TYPES:
                results[key] = self._unknown_fold(node)
                continue
            if node_type == "raw_string":
                value = raw[1:-1] if len(raw) >= 2 else b""
                raw_fragments = (
                    (
                        (
                            node.start_byte + 1,
                            max(node.start_byte + 1, node.end_byte - 1),
                            0,
                            len(value),
                            True,
                        ),
                    )
                    if value
                    else ()
                )
                results[key] = _FoldedValue(
                    StaticValue.exact(value),
                    raw_fragments,
                    quoted_empty_offsets=(0,) if not value else (),
                    atoms=(
                        _FoldAtom(
                            _ValueAtomKind.LITERAL,
                            StaticValue.exact(value),
                            raw_fragments,
                            node.start_byte,
                            node.end_byte,
                            quoted=True,
                        ),
                    ),
                )
                continue
            if node_type in {"word", "variable_name", "number"}:
                (
                    decoded_unquoted,
                    has_pathname_expansion,
                    tilde_offsets,
                    assignment_delimiter_offsets,
                    surviving_fragments,
                ) = self._decode_unquoted(raw, source_start_byte=node.start_byte)
                if decoded_unquoted is None:
                    self._issue(
                        ShellIssueReason.UNSUPPORTED_SEMANTICS,
                        self._node_span(node) or self._point_span(node.start_byte),
                    )
                    results[key] = self._unknown_fold(node)
                elif node_type == "word" and has_pathname_expansion:
                    results[key] = self._unknown_fold(node)
                else:
                    results[key] = _FoldedValue(
                        StaticValue.exact(decoded_unquoted),
                        surviving_fragments,
                        tilde_offsets if node_type == "word" else (),
                        assignment_delimiter_offsets if node_type == "word" else (),
                        atoms=(
                            _FoldAtom(
                                _ValueAtomKind.LITERAL,
                                StaticValue.exact(decoded_unquoted),
                                surviving_fragments,
                                node.start_byte,
                                node.end_byte,
                            ),
                        ),
                    )
                continue
            if node_type == "string_content":
                decoded_quoted, surviving_fragments = self._decode_double_quoted(
                    raw,
                    source_start_byte=node.start_byte,
                )
                if decoded_quoted is None:
                    self._issue(
                        ShellIssueReason.UNSUPPORTED_SEMANTICS,
                        self._node_span(node) or self._point_span(node.start_byte),
                    )
                    results[key] = self._unknown_fold(node)
                else:
                    results[key] = _FoldedValue(
                        StaticValue.exact(decoded_quoted),
                        surviving_fragments,
                        atoms=(
                            _FoldAtom(
                                _ValueAtomKind.LITERAL,
                                StaticValue.exact(decoded_quoted),
                                surviving_fragments,
                                node.start_byte,
                                node.end_byte,
                                quoted=True,
                            ),
                        ),
                    )
                continue
            if node_type == "command_name":
                named = node.named_children
                results[key] = (
                    results[_node_key(named[0])] if len(named) == 1 else self._unknown_fold(node)
                )
                continue
            if node_type == "variable_assignment":
                name_node = node.child_by_field_name("name")
                value_node = node.child_by_field_name("value")
                if name_node is None:
                    results[key] = self._unknown_fold(node)
                    continue
                name_fold = results[_node_key(name_node)]
                value_fold = (
                    results[_node_key(value_node)]
                    if value_node is not None
                    else _FoldedValue(StaticValue.exact(b""), ())
                )
                if name_fold.value.state is StaticValueState.EXACT:
                    name_bytes = cast(bytes, name_fold.value.exact_bytes)
                    value_bytes = (
                        cast(bytes, value_fold.value.exact_bytes)
                        if value_fold.value.state is StaticValueState.EXACT
                        else b""
                    )
                    combined = name_bytes + b"=" + value_bytes
                    value_offset = len(name_bytes) + 1
                    prefix_end = value_node.start_byte if value_node is not None else node.end_byte
                    prefix_fragments = ((node.start_byte, prefix_end, 0, value_offset, True),)
                    assignment_fragments = list(prefix_fragments)
                    if value_fold.value.state is StaticValueState.EXACT:
                        assignment_fragments.extend(
                            (
                                source_start,
                                source_end,
                                value_offset + value_start,
                                value_offset + value_end,
                                exact,
                            )
                            for source_start, source_end, value_start, value_end, exact in value_fold.fragments
                        )
                    atoms = (
                        (
                            _FoldAtom(
                                _ValueAtomKind.LITERAL,
                                StaticValue.exact(name_bytes + b"="),
                                prefix_fragments,
                                node.start_byte,
                                prefix_end,
                            ),
                            *value_fold.atoms,
                        )
                        if value_fold.atoms is not None
                        else None
                    )
                    results[key] = self._apply_tilde_context(
                        _FoldedValue(
                            StaticValue.exact(combined)
                            if value_fold.value.state is StaticValueState.EXACT
                            else StaticValue.unknown(),
                            tuple(assignment_fragments)
                            if value_fold.value.state is StaticValueState.EXACT
                            else ((node.start_byte, node.end_byte, 0, 0, False),),
                            tuple(
                                value_offset + offset
                                for offset in value_fold.unquoted_tilde_offsets
                            ),
                            (
                                len(name_bytes),
                                *(
                                    value_offset + offset
                                    for offset in value_fold.unquoted_assignment_delimiter_offsets
                                ),
                            ),
                            tuple(
                                value_offset + offset for offset in value_fold.quoted_empty_offsets
                            ),
                            atoms,
                        ),
                        assignment_value=True,
                    )
                else:
                    results[key] = self._unknown_fold(node)
                continue
            if node_type in {"string", "concatenation"}:
                parts = [
                    results[_node_key(child)]
                    for child in node.named_children
                    if child.type not in {'"', "'"}
                ]
                exact = all(part.value.state is StaticValueState.EXACT for part in parts)
                output = bytearray()
                combined_fragments: list[tuple[int, int, int, int, bool]] = []
                combined_tilde_offsets: list[int] = []
                combined_assignment_delimiter_offsets: list[int] = []
                combined_quoted_empty_offsets: list[int] = []
                combined_atoms: list[_FoldAtom] | None = []
                output_cursor = 0
                for part in parts:
                    part_bytes = part.value.exact_bytes if exact else None
                    for (
                        source_start,
                        source_end,
                        value_start,
                        value_end,
                        fragment_exact,
                    ) in part.fragments:
                        combined_fragments.append(
                            (
                                source_start,
                                source_end,
                                output_cursor + value_start if exact else 0,
                                output_cursor + value_end if exact else 0,
                                fragment_exact and exact,
                            )
                        )
                    if part_bytes is not None:
                        combined_tilde_offsets.extend(
                            output_cursor + offset for offset in part.unquoted_tilde_offsets
                        )
                        combined_assignment_delimiter_offsets.extend(
                            output_cursor + offset
                            for offset in part.unquoted_assignment_delimiter_offsets
                        )
                        combined_quoted_empty_offsets.extend(
                            output_cursor + offset for offset in part.quoted_empty_offsets
                        )
                        output.extend(part_bytes)
                        output_cursor += len(part_bytes)
                    if combined_atoms is not None:
                        if part.atoms is None:
                            combined_atoms = None
                        else:
                            combined_atoms.extend(
                                replace(atom, quoted=True)
                                if node_type == "string" and atom.kind is _ValueAtomKind.VARIABLE
                                else atom
                                for atom in part.atoms
                            )
                if node_type == "string" and exact and not output:
                    combined_quoted_empty_offsets.append(0)
                results[key] = _FoldedValue(
                    StaticValue.exact(bytes(output)) if exact else StaticValue.unknown(),
                    tuple(combined_fragments) or ((node.start_byte, node.end_byte, 0, 0, exact),),
                    tuple(combined_tilde_offsets),
                    tuple(combined_assignment_delimiter_offsets),
                    tuple(combined_quoted_empty_offsets),
                    tuple(combined_atoms) if combined_atoms is not None else None,
                )
                continue
            if not node.is_named:
                fragments = ((node.start_byte, node.end_byte, 0, len(raw), True),) if raw else ()
                results[key] = _FoldedValue(
                    StaticValue.exact(raw),
                    fragments,
                    atoms=(
                        _FoldAtom(
                            _ValueAtomKind.LITERAL,
                            StaticValue.exact(raw),
                            fragments,
                            node.start_byte,
                            node.end_byte,
                        ),
                    ),
                )
                continue
            self._issue(
                ShellIssueReason.UNSUPPORTED_SEMANTICS,
                self._node_span(node) or self._point_span(node.start_byte),
            )
            results[key] = self._unknown_fold(node)
        return results[_node_key(root)]

    def _fold_group(
        self,
        group: _NodeGroup,
        *,
        assignment_value: bool = False,
        assignment_equals_offset: int | None = None,
    ) -> _FoldedValue:
        parts = [self._fold_node(node) for node in group.nodes]
        exact = all(part.value.state is StaticValueState.EXACT for part in parts)
        output = bytearray()
        output_cursor = 0
        fragments: list[tuple[int, int, int, int, bool]] = []
        tilde_offsets: list[int] = []
        assignment_delimiter_offsets: list[int] = []
        quoted_empty_offsets: list[int] = []
        atoms: list[_FoldAtom] | None = []
        for part in parts:
            for source_start, source_end, value_start, value_end, fragment_exact in part.fragments:
                fragments.append(
                    (
                        source_start,
                        source_end,
                        output_cursor + value_start if exact else 0,
                        output_cursor + value_end if exact else 0,
                        fragment_exact and exact,
                    )
                )
            if exact:
                value = cast(bytes, part.value.exact_bytes)
                tilde_offsets.extend(
                    output_cursor + offset for offset in part.unquoted_tilde_offsets
                )
                assignment_delimiter_offsets.extend(
                    output_cursor + offset for offset in part.unquoted_assignment_delimiter_offsets
                )
                quoted_empty_offsets.extend(
                    output_cursor + offset for offset in part.quoted_empty_offsets
                )
                output.extend(value)
                output_cursor += len(value)
            if atoms is not None:
                if part.atoms is None:
                    atoms = None
                else:
                    atoms.extend(part.atoms)
        if assignment_equals_offset is not None:
            equals_index = bisect_left(assignment_delimiter_offsets, assignment_equals_offset)
            if (
                equals_index == len(assignment_delimiter_offsets)
                or assignment_delimiter_offsets[equals_index] != assignment_equals_offset
            ):
                assignment_delimiter_offsets.insert(equals_index, assignment_equals_offset)
        return self._apply_tilde_context(
            _FoldedValue(
                StaticValue.exact(bytes(output)) if exact else StaticValue.unknown(),
                tuple(fragments),
                tuple(tilde_offsets),
                tuple(assignment_delimiter_offsets),
                tuple(quoted_empty_offsets),
                tuple(atoms) if atoms is not None else None,
            ),
            assignment_value=assignment_value,
        )

    @staticmethod
    def _is_line_continuation_gap(gap: bytes) -> bool:
        saw_continuation = False
        index = 0
        while index < len(gap):
            if gap[index] in {9, 32}:
                index += 1
                continue
            if gap[index : index + 2] == b"\\\n":
                saw_continuation = True
                index += 2
                continue
            return False
        return saw_continuation

    def _group_nodes(self, nodes: list[Node]) -> list[_NodeGroup]:
        groups: list[list[Node]] = []
        for node in sorted(nodes, key=lambda item: (item.start_byte, item.end_byte)):
            if groups and self.raw[groups[-1][-1].end_byte : node.start_byte] == b"\\\n":
                groups[-1].append(node)
            else:
                groups.append([node])
        return [
            _NodeGroup(
                tuple(group),
                group[0].start_byte,
                group[-1].end_byte,
                b"".join(self.raw[node.start_byte : node.end_byte] for node in group),
            )
            for group in groups
        ]

    def _argument(self, group: _NodeGroup) -> _ArgumentIR | None:
        span = self._span(group.start_byte, group.end_byte)
        if span is None:
            self._issue(
                ShellIssueReason.UNSUPPORTED_SEMANTICS,
                self._point_span(group.start_byte),
            )
            return None
        folded = self._fold_group(group)
        fragments = self._mapped_fragments(
            folded.fragments,
            fallback_start=group.start_byte,
        )
        if fragments is None:
            return None
        atoms = (
            () if folded.value.state is StaticValueState.EXACT else self._mapped_atoms(folded.atoms)
        )
        if atoms is None:
            return None
        value_bytes = (
            len(cast(bytes, folded.value.exact_bytes))
            if folded.value.state is StaticValueState.EXACT
            else sum(
                len(cast(bytes, atom.value.exact_bytes))
                for atom in atoms
                if atom.value.state is StaticValueState.EXACT
            )
        )
        if not self._reserve(
            1 + len(fragments) + len(atoms),
            value_bytes=value_bytes,
            span=span,
        ):
            return None
        return _ArgumentIR(
            folded.value,
            span,
            fragments,
            atoms,
            group.start_byte,
            group.end_byte,
            self._home_relative_generated_target(group.raw_syntax),
        )

    @staticmethod
    def _home_relative_generated_target(raw_syntax: bytes) -> bytes | None:
        """Retain only the two code-owned home forms needed for config dispatch."""
        if raw_syntax in {b'"$HOME/.npmrc"', b"~/.npmrc"}:
            return b".npmrc"
        return None

    @staticmethod
    def _identifier(raw: bytes) -> str | None:
        if not raw or not (raw[0] == 95 or 65 <= raw[0] <= 90 or 97 <= raw[0] <= 122):
            return None
        if any(
            not (byte == 95 or 48 <= byte <= 57 or 65 <= byte <= 90 or 97 <= byte <= 122)
            for byte in raw[1:]
        ):
            return None
        return raw.decode("ascii")

    def _group_assignment_name(self, group: _NodeGroup) -> tuple[str, int] | None:
        equals = group.raw_syntax.find(b"=")
        if equals <= 0:
            return None
        name = self._identifier(group.raw_syntax[:equals])
        return (name, equals) if name is not None else None

    @staticmethod
    def _slice_fold_fragments(
        fragments: tuple[tuple[int, int, int, int, bool], ...],
        start_byte: int,
    ) -> tuple[tuple[int, int, int, int, bool], ...]:
        sliced: list[tuple[int, int, int, int, bool]] = []
        for source_start, source_end, value_start, value_end, exact in fragments:
            if value_end <= start_byte:
                continue
            clipped_start = max(value_start, start_byte)
            source_clip = clipped_start - value_start if exact else 0
            sliced.append(
                (
                    source_start + source_clip,
                    source_end,
                    clipped_start - start_byte if exact else 0,
                    value_end - start_byte if exact else 0,
                    exact,
                )
            )
        return tuple(sliced)

    @classmethod
    def _slice_fold_atoms(
        cls,
        atoms: tuple[_FoldAtom, ...] | None,
        start_byte: int,
    ) -> tuple[_FoldAtom, ...] | None:
        if atoms is None:
            return None
        remaining = start_byte
        sliced: list[_FoldAtom] = []
        for atom in atoms:
            if remaining:
                if atom.value.state is not StaticValueState.EXACT:
                    return None
                literal = cast(bytes, atom.value.exact_bytes)
                if remaining >= len(literal):
                    remaining -= len(literal)
                    continue
                fragments = cls._slice_fold_fragments(atom.fragments, remaining)
                sliced.append(
                    replace(
                        atom,
                        value=StaticValue.exact(literal[remaining:]),
                        fragments=fragments,
                    )
                )
                remaining = 0
                continue
            sliced.append(atom)
        return tuple(sliced) if remaining == 0 else None

    def _group_assignment_parts(
        self,
        group: _NodeGroup,
    ) -> (
        tuple[
            str,
            StaticValue,
            tuple[_ValueFragment, ...],
            tuple[_ValueAtom, ...],
        ]
        | None
    ):
        syntax = self._group_assignment_name(group)
        if syntax is None:
            return None
        name, equals = syntax
        folded = self._fold_group(
            group,
            assignment_value=True,
            assignment_equals_offset=equals,
        )
        value_offset = equals + 1
        if folded.value.state is StaticValueState.EXACT:
            exact = cast(bytes, folded.value.exact_bytes)
            value = StaticValue.exact(exact[value_offset:])
        else:
            value = StaticValue.unknown()
        value_fragments = self._slice_fold_fragments(folded.fragments, value_offset)
        mapped = self._mapped_fragments(value_fragments, fallback_start=group.start_byte)
        if mapped is None:
            return None
        atoms = (
            ()
            if value.state is StaticValueState.EXACT
            else self._mapped_atoms(self._slice_fold_atoms(folded.atoms, value_offset))
        )
        if atoms is None:
            return None
        return name, value, mapped, atoms

    def _assignment_from_group(
        self,
        group: _NodeGroup,
        *,
        order: int,
        region_id: int | None,
        function_id: int | None,
        prefix_for_command_start_byte: int | None,
        declaration_keyword: bytes | None = None,
        declaration_command_start_byte: int | None = None,
    ) -> _AssignmentIR | None:
        parts = self._group_assignment_parts(group)
        if parts is None:
            return None
        name, value, fragments, atoms = parts
        span = self._span(group.start_byte, group.end_byte)
        if span is None:
            self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, self._point_span(group.start_byte))
            return None
        value_bytes = (
            len(cast(bytes, value.exact_bytes))
            if value.state is StaticValueState.EXACT
            else sum(
                len(cast(bytes, atom.value.exact_bytes))
                for atom in atoms
                if atom.value.state is StaticValueState.EXACT
            )
        )
        if not self._reserve(
            2 + len(fragments) + len(atoms),
            value_bytes=value_bytes,
            span=span,
        ):
            return None
        site = AssignmentSite(
            unit_id=self.unit.unit_id,
            provenance=self.unit.provenance,
            span=span,
            name=name,
            value=value,
        )
        return _AssignmentIR(
            site,
            order,
            region_id,
            function_id,
            prefix_for_command_start_byte,
            fragments,
            atoms,
            declaration_keyword=declaration_keyword,
            declaration_command_start_byte=declaration_command_start_byte,
            local_start_byte=group.start_byte,
            local_end_byte=group.end_byte,
        )

    def _assignment_from_node(
        self,
        draft: _AssignmentDraft,
        *,
        order: int,
    ) -> _AssignmentIR | None:
        node = draft.node
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return None
        name = self._identifier(self.raw[name_node.start_byte : name_node.end_byte])
        if name is None:
            self._issue(
                ShellIssueReason.UNSUPPORTED_SEMANTICS,
                self._node_span(node) or self._point_span(node.start_byte),
            )
            return None
        value_node = node.child_by_field_name("value")
        if value_node is None:
            value = StaticValue.exact(b"")
            fragments: tuple[_ValueFragment, ...] = ()
            atoms: tuple[_ValueAtom, ...] = ()
        else:
            folded = self._apply_tilde_context(
                self._fold_node(value_node),
                assignment_value=True,
            )
            value = folded.value
            mapped = self._mapped_fragments(folded.fragments, fallback_start=value_node.start_byte)
            if mapped is None:
                return None
            fragments = mapped
            if value.state is StaticValueState.EXACT:
                atoms = ()
            else:
                mapped_atoms = self._mapped_atoms(folded.atoms)
                if mapped_atoms is None:
                    return None
                atoms = mapped_atoms
        span = self._node_span(node)
        if span is None:
            self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, self._point_span(node.start_byte))
            return None
        value_bytes = (
            len(cast(bytes, value.exact_bytes))
            if value.state is StaticValueState.EXACT
            else sum(
                len(cast(bytes, atom.value.exact_bytes))
                for atom in atoms
                if atom.value.state is StaticValueState.EXACT
            )
        )
        if not self._reserve(
            2 + len(fragments) + len(atoms),
            value_bytes=value_bytes,
            span=span,
        ):
            return None
        return _AssignmentIR(
            AssignmentSite(
                unit_id=self.unit.unit_id,
                provenance=self.unit.provenance,
                span=span,
                name=name,
                value=value,
            ),
            order,
            draft.region_id,
            draft.function_id,
            draft.prefix_for_command_start_byte,
            fragments,
            atoms,
            declaration_keyword=None,
            declaration_command_start_byte=None,
            local_start_byte=node.start_byte,
            local_end_byte=node.end_byte,
        )

    def _add_region(
        self,
        node: Node,
        kind: _ExecutionRegionKind,
        *,
        parent_region_id: int | None,
        function_id: int | None,
    ) -> int | None:
        span = self._node_span(node)
        if span is None:
            self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, self._point_span(node.start_byte))
            return parent_region_id
        if not self._reserve(1, span=span):
            return parent_region_id
        region_id = len(self.regions)
        self.regions.append(
            _ExecutionRegion(
                region_id,
                region_id,
                kind,
                span,
                parent_region_id,
                function_id,
                node.start_byte,
                node.end_byte,
                next(
                    (child.start_byte for child in node.children if child.type in {"&&", "||"}),
                    None,
                ),
            )
        )
        return region_id

    @staticmethod
    def _normalized_heredoc_delimiter(
        raw: bytes,
        start: int,
        limit: int,
    ) -> tuple[bytes, bool, int] | None:
        """Apply only heredoc delimiter quote removal inside an anchored header."""
        index = start
        while index < limit and raw[index] in {9, 32}:
            index += 1
        output = bytearray()
        quoted = False
        quote: int | None = None
        ansi_quote = False
        while index < limit:
            byte = raw[index]
            if quote is None:
                if byte in {9, 10, 13, 32, 38, 40, 41, 59, 60, 62, 124}:
                    break
                if byte == 36 and index + 1 < limit and raw[index + 1] in {34, 39}:
                    quoted = True
                    ansi_quote = raw[index + 1] == 39
                    quote = raw[index + 1]
                    index += 2
                    continue
                if byte in {34, 39}:
                    quoted = True
                    quote = byte
                    ansi_quote = False
                    index += 1
                    continue
                if byte == 92:
                    quoted = True
                    if index + 1 >= limit:
                        return None
                    if raw[index + 1] == 10:
                        index += 2
                        continue
                    output.append(raw[index + 1])
                    index += 2
                    continue
                if byte == 0:
                    return None
                output.append(byte)
                index += 1
                continue
            if byte == quote:
                quote = None
                ansi_quote = False
                index += 1
                continue
            if byte == 0 or byte in {10, 13}:
                return None
            if byte == 92 and quote == 34:
                if index + 1 >= limit:
                    return None
                following = raw[index + 1]
                if following == 10:
                    index += 2
                    continue
                if following in {34, 36, 92, 96}:
                    output.append(following)
                    index += 2
                    continue
            if byte == 92 and ansi_quote:
                if index + 1 >= limit or raw[index + 1] not in {39, 92}:
                    return None
                output.append(raw[index + 1])
                index += 2
                continue
            output.append(byte)
            index += 1
        if quote is not None or not output:
            return None
        return bytes(output), quoted, index

    def _recovery_step(self, count: int, span: SourceSpan) -> bool:
        if exhaustion := self.file_budget.charge_shell_cst_visits(
            self.accounting_unit,
            count,
        ):
            self._resource(exhaustion, span)
            return False
        return True

    def _mask_data_range(self, start_byte: int, end_byte: int) -> None:
        if end_byte <= start_byte:
            return
        index = bisect_left(self.masked_data_ranges, (start_byte, -1))
        if index > 0 and self.masked_data_ranges[index - 1][1] >= start_byte:
            index -= 1
            start_byte = min(start_byte, self.masked_data_ranges[index][0])
            end_byte = max(end_byte, self.masked_data_ranges[index][1])
        right = index
        while right < len(self.masked_data_ranges):
            candidate_start, candidate_end = self.masked_data_ranges[right]
            if candidate_start > end_byte:
                break
            end_byte = max(end_byte, candidate_end)
            right += 1
        self.masked_data_ranges[index:right] = [(start_byte, end_byte)]

    def _is_masked_data_node(self, node: Node) -> bool:
        index = bisect_right(self.masked_data_ranges, (node.start_byte, len(self.raw))) - 1
        return index >= 0 and node.end_byte <= self.masked_data_ranges[index][1]

    def _body_argument(
        self,
        start_byte: int,
        end_byte: int,
        *,
        strip_tabs: bool,
        fallback_span: SourceSpan,
    ) -> _ArgumentIR | None:
        output = bytearray()
        raw_fragments: list[tuple[int, int, int, int, bool]] = []
        if strip_tabs:
            line_index = max(0, bisect_right(self.local_line_starts, start_byte) - 1)
            while line_index < len(self.local_lines):
                line = self.local_lines[line_index]
                if line.start >= end_byte:
                    break
                line_index += 1
                if line.full_end <= start_byte:
                    continue
                segment_start = max(line.start, start_byte)
                segment_end = min(line.full_end, end_byte)
                while (
                    segment_start < min(line.content_end, segment_end)
                    and self.raw[segment_start] == 9
                ):
                    segment_start += 1
                if segment_end <= segment_start:
                    continue
                value_start = len(output)
                output.extend(self.raw[segment_start:segment_end])
                raw_fragments.append((segment_start, segment_end, value_start, len(output), True))
        elif end_byte > start_byte:
            output.extend(self.raw[start_byte:end_byte])
            raw_fragments.append((start_byte, end_byte, 0, end_byte - start_byte, True))
        fragments = self._mapped_fragments(tuple(raw_fragments), fallback_start=start_byte)
        if fragments is None:
            return None
        span = self._span(start_byte, end_byte) if end_byte > start_byte else fallback_span
        if span is None:
            self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, fallback_span)
            return None
        if not self._reserve(
            1 + len(fragments),
            value_bytes=len(output),
            span=span,
        ):
            return None
        return _ArgumentIR(
            StaticValue.exact(bytes(output)),
            span,
            fragments,
            (),
            start_byte,
            end_byte,
        )

    @staticmethod
    def _decode_heredoc_literal(
        raw: bytes,
        *,
        source_start_byte: int,
    ) -> tuple[bytes | None, tuple[tuple[int, int, int, int, bool], ...], bool]:
        """Decode only Bash's unquoted-heredoc backslash rules."""
        output = bytearray()
        fragments: list[tuple[int, int, int, int, bool]] = []
        dynamic_marker = False
        index = 0
        while index < len(raw):
            byte = raw[index]
            if byte == 0:
                return None, (), True
            if byte == 92 and index + 1 < len(raw):
                following = raw[index + 1]
                if following == 10:
                    index += 2
                    continue
                if following in {36, 92, 96}:
                    value_offset = len(output)
                    output.append(following)
                    self_fragment = (
                        source_start_byte + index + 1,
                        source_start_byte + index + 2,
                        value_offset,
                        value_offset + 1,
                        True,
                    )
                    fragments.append(self_fragment)
                    index += 2
                    continue
            if byte in {36, 96}:
                dynamic_marker = True
            value_offset = len(output)
            output.append(byte)
            _ShellLowerer._append_surviving_fragment(
                fragments,
                source_byte=source_start_byte + index,
                value_byte=value_offset,
            )
            index += 1
        return bytes(output), tuple(fragments), dynamic_marker

    def _native_expanding_body_argument(
        self,
        body: Node,
        *,
        fallback_span: SourceSpan,
    ) -> _ArgumentIR | None:
        expansion_nodes = sorted(
            (child for child in body.named_children if child.type != "heredoc_content"),
            key=lambda child: (child.start_byte, child.end_byte),
        )
        atoms: list[_FoldAtom] = []
        cursor = body.start_byte
        literal_bytes = 0
        for expansion in (*expansion_nodes, None):
            end = expansion.start_byte if expansion is not None else body.end_byte
            if end < cursor:
                self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, fallback_span)
                return None
            if end > cursor:
                literal, raw_fragments, dynamic_marker = self._decode_heredoc_literal(
                    self.raw[cursor:end],
                    source_start_byte=cursor,
                )
                if literal is None or dynamic_marker:
                    self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, fallback_span)
                    if not self._reserve(1, span=fallback_span):
                        return None
                    return _ArgumentIR(
                        StaticValue.unknown(),
                        fallback_span,
                        (),
                        (),
                        body.start_byte,
                        body.end_byte,
                    )
                literal_bytes += len(literal)
                atoms.append(
                    _FoldAtom(
                        _ValueAtomKind.LITERAL,
                        StaticValue.exact(literal),
                        raw_fragments,
                        cursor,
                        end,
                    )
                )
            if expansion is None:
                break
            name = self._simple_variable_reference(expansion)
            if name is None:
                span = self._node_span(expansion) or fallback_span
                self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, span)
                if not self._reserve(1, span=fallback_span):
                    return None
                return _ArgumentIR(
                    StaticValue.unknown(),
                    fallback_span,
                    (),
                    (),
                    body.start_byte,
                    body.end_byte,
                )
            atoms.append(
                _FoldAtom(
                    _ValueAtomKind.VARIABLE,
                    StaticValue.unknown(),
                    (),
                    expansion.start_byte,
                    expansion.end_byte,
                    name=name,
                    quoted=True,
                )
            )
            cursor = expansion.end_byte
        mapped_atoms = self._mapped_atoms(tuple(atoms))
        if mapped_atoms is None:
            return None
        span = self._span(body.start_byte, body.end_byte) or fallback_span
        fragment_count = sum(len(atom.fragments) for atom in mapped_atoms)
        if not self._reserve(
            1 + len(mapped_atoms) + fragment_count,
            value_bytes=literal_bytes,
            span=span,
        ):
            return None
        if all(atom.kind is _ValueAtomKind.LITERAL for atom in mapped_atoms):
            exact_value = StaticValue.exact(
                b"".join(cast(bytes, atom.value.exact_bytes) for atom in mapped_atoms)
            )
            resolved_fragments: list[_ValueFragment] = []
            output_cursor = 0
            for atom in mapped_atoms:
                atom_bytes = cast(bytes, atom.value.exact_bytes)
                resolved_fragments.extend(
                    replace(
                        fragment,
                        value_start_byte=output_cursor + fragment.value_start_byte,
                        value_end_byte=output_cursor + fragment.value_end_byte,
                    )
                    for fragment in atom.fragments
                )
                output_cursor += len(atom_bytes)
            return _ArgumentIR(
                exact_value,
                span,
                tuple(resolved_fragments),
                (),
                body.start_byte,
                body.end_byte,
            )
        return _ArgumentIR(
            StaticValue.unknown(),
            span,
            (),
            mapped_atoms,
            body.start_byte,
            body.end_byte,
        )

    def _recover_cst_anchored_body_argument(
        self,
        start_byte: int,
        end_byte: int,
        *,
        strip_tabs: bool,
        expands: bool,
        fallback_span: SourceSpan,
    ) -> _ArgumentIR | None:
        if not expands:
            return self._body_argument(
                start_byte,
                end_byte,
                strip_tabs=strip_tabs,
                fallback_span=fallback_span,
            )
        if strip_tabs:
            literal = self._body_argument(
                start_byte,
                end_byte,
                strip_tabs=True,
                fallback_span=fallback_span,
            )
            if literal is None or literal.value.state is not StaticValueState.EXACT:
                return literal
            if not any(
                marker in cast(bytes, literal.value.exact_bytes) for marker in (b"$", b"`", b"\\")
            ):
                return literal
            self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, literal.span)
            return replace(literal, value=StaticValue.unknown(), fragments=())
        value, raw_fragments, dynamic_marker = self._decode_heredoc_literal(
            self.raw[start_byte:end_byte],
            source_start_byte=start_byte,
        )
        span = self._span(start_byte, end_byte) or fallback_span
        if value is None or dynamic_marker:
            self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, span)
            if not self._reserve(1, span=span):
                return None
            return _ArgumentIR(
                StaticValue.unknown(),
                span,
                (),
                (),
                start_byte,
                end_byte,
            )
        fragments = self._mapped_fragments(raw_fragments, fallback_start=start_byte)
        if fragments is None:
            return None
        if not self._reserve(
            1 + len(fragments),
            value_bytes=len(value),
            span=span,
        ):
            return None
        return _ArgumentIR(
            StaticValue.exact(value),
            span,
            fragments,
            (),
            start_byte,
            end_byte,
        )

    def _here_string_argument(self, redirect: Node, operator_start: int) -> _ArgumentIR | None:
        destination = redirect.child_by_field_name("destination")
        if destination is None:
            named = redirect.named_children
            if (
                redirect.type != "herestring_redirect"
                or len(named) != 1
                or not redirect.children
                or redirect.children[0].type != "<<<"
            ):
                return None
            destination = named[0]
        argument = self._argument(
            _NodeGroup(
                (destination,),
                destination.start_byte,
                destination.end_byte,
                self.raw[destination.start_byte : destination.end_byte],
            )
        )
        if argument is None:
            return None
        if argument.value.state is StaticValueState.EXACT:
            value = cast(bytes, argument.value.exact_bytes) + b"\n"
            if not self._reserve(1, value_bytes=1, span=argument.span):
                return None
            argument = replace(argument, value=StaticValue.exact(value))
        elif argument.atoms:
            if not self._reserve(1, value_bytes=1, span=argument.span):
                return None
            argument = replace(
                argument,
                atoms=(
                    *argument.atoms,
                    _ValueAtom(
                        _ValueAtomKind.LITERAL,
                        StaticValue.exact(b"\n"),
                        argument.span,
                        (),
                        quoted=True,
                    ),
                ),
            )
        span = self._span(operator_start, redirect.end_byte)
        if span is None:
            self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, argument.span)
            return None
        return argument

    def _native_heredoc_fact(self, node: Node) -> _RedirectFact | None:
        start_node = next((child for child in node.children if child.type == "heredoc_start"), None)
        body_node = next((child for child in node.children if child.type == "heredoc_body"), None)
        end_node = next((child for child in node.children if child.type == "heredoc_end"), None)
        operator_node = next(
            (child for child in node.children if child.type in {"<<", "<<-"}),
            None,
        )
        if (
            start_node is None
            or body_node is None
            or end_node is None
            or end_node.is_missing
            or any(child.is_error or child.is_missing for child in node.children)
        ):
            return None
        descriptor = node.child_by_field_name("descriptor")
        if descriptor is not None and self.raw[descriptor.start_byte : descriptor.end_byte] != b"0":
            self._issue(
                ShellIssueReason.UNSUPPORTED_SEMANTICS,
                self._node_span(descriptor) or self._point_span(descriptor.start_byte),
            )
            return None
        operator = operator_node.type if operator_node is not None else ""
        operator_start = operator_node.start_byte if operator_node is not None else node.start_byte
        start_raw = self.raw[start_node.start_byte : start_node.end_byte]
        if operator not in {"<<", "<<-"} or not start_raw:
            return None
        normalized = self._normalized_heredoc_delimiter(start_raw, 0, len(start_raw))
        if normalized is None:
            return None
        delimiter, quoted, word_end = normalized
        if word_end != len(start_raw):
            return None
        end_line_index = max(0, bisect_right(self.local_line_starts, end_node.start_byte) - 1)
        end_line = self.local_lines[end_line_index]
        candidate_start = end_line.start
        if operator == "<<-":
            while candidate_start < end_line.content_end and self.raw[candidate_start] == 9:
                candidate_start += 1
        if (
            self.raw[candidate_start : end_line.content_end] != delimiter
            or end_node.start_byte != candidate_start
            or end_node.end_byte != end_line.content_end
        ):
            return None
        body_line_index = max(0, bisect_right(self.local_line_starts, body_node.start_byte) - 1)
        body_start = self.local_lines[body_line_index].start
        operator_span = self._span(operator_start, max(operator_start + 2, start_node.end_byte))
        if operator_span is None:
            return None
        argument = (
            self._native_expanding_body_argument(
                body_node,
                fallback_span=operator_span,
            )
            if not quoted and operator == "<<" and body_node.start_byte == body_start
            else self._recover_cst_anchored_body_argument(
                body_start,
                end_line.start,
                strip_tabs=operator == "<<-",
                expands=not quoted,
                fallback_span=operator_span,
            )
        )
        if argument is None:
            return None
        self._mask_data_range(body_start, end_line.full_end)
        return _RedirectFact(
            _RedirectKind.STDIN_HEREDOC,
            argument,
            operator_span,
            operator_start,
            expands=not quoted,
        )

    def _candidate_operator_nodes(
        self, root: Node
    ) -> tuple[list[Node], list[tuple[int, bool, Node | None, bool | None]]]:
        nodes: list[Node] = []
        candidates: list[tuple[int, bool, Node | None, bool | None]] = []
        stack = [root]
        while stack:
            node = stack.pop()
            nodes.append(node)
            children = list(node.children)
            for index, child in enumerate(children[:-1]):
                following = children[index + 1]
                if not child.is_error or following.type != "file_redirect":
                    continue
                child_raw_end = child.end_byte
                if following.start_byte != child_raw_end:
                    continue
                error_raw = self.raw[child.start_byte : child.end_byte]
                if error_raw == b"<<":
                    candidates.append((child.end_byte - 2, True, following, False))
                elif error_raw.endswith(b"<"):
                    named_types = [named.type for named in child.named_children]
                    separate_segment = (
                        True
                        if named_types == ["word", "file_redirect"]
                        else False
                        if error_raw == b"<" and not named_types
                        else None
                    )
                    candidates.append((child.end_byte - 1, False, following, separate_segment))
            if node.type != "heredoc_body":
                stack.extend(reversed(children))
        return nodes, candidates

    def _recover_cst_anchored_heredoc_facts(
        self,
        statement: Node,
        heredocs: tuple[Node, ...],
        function_id: int | None,
    ) -> tuple[list[_RedirectFact], list[_RecoveredGeneratedDraft]] | None:
        statement_span = self._node_span(statement) or self._point_span(statement.start_byte)
        nodes, split_candidates = self._candidate_operator_nodes(statement)
        if not self._recovery_step(len(nodes), statement_span):
            return None
        operator_candidates: dict[int, tuple[bool, Node | None, bool | None]] = {}
        for heredoc in heredocs:
            operator = next(
                (child for child in heredoc.children if child.type in {"<<", "<<-"}),
                None,
            )
            if operator is not None and operator.end_byte > operator.start_byte:
                operator_candidates[operator.start_byte] = (False, None, False)
                continue
            start_node = next(
                (child for child in heredoc.children if child.type == "heredoc_start"),
                None,
            )
            if start_node is not None:
                relative = self.raw[start_node.start_byte : start_node.end_byte].find(b"<<")
                if relative >= 0:
                    operator_candidates[start_node.start_byte + relative] = (False, None, False)
        for position, here_candidate, redirect, separate_segment in split_candidates:
            if position < statement.start_byte or position + 2 > statement.end_byte:
                continue
            if here_candidate and self.raw[position : position + 3] == b"<<<":
                operator_candidates[position] = (True, redirect, False)
            elif not here_candidate and self.raw[position : position + 2] in {b"<<", b"<<-"}:
                if separate_segment is None:
                    return None
                operator_candidates[position] = (False, redirect, separate_segment)
        if not operator_candidates:
            return None
        ordered_candidates = sorted(operator_candidates.items())
        first_operator = ordered_candidates[0][0]
        header_end = first_operator
        while header_end < len(self.raw):
            if self.raw[header_end] == 10:
                backslashes = 0
                cursor = header_end
                while cursor > first_operator and self.raw[cursor - 1] == 92:
                    backslashes += 1
                    cursor -= 1
                if backslashes % 2 == 0:
                    break
            header_end += 1
        if header_end >= len(self.raw):
            return None
        if not self._recovery_step(header_end - first_operator + 1, statement_span):
            return None

        specs: list[tuple[int, bytes, bool, bool, bool, int]] = []
        here_facts: list[_RedirectFact] = []
        separate_input_positions: set[int] = set()
        candidate_redirects_by_position: dict[int, Node] = {}
        for position, (is_here_string, redirect, separate_segment) in ordered_candidates:
            descriptor_start = position
            while (
                descriptor_start > statement.start_byte
                and 48 <= self.raw[descriptor_start - 1] <= 57
            ):
                descriptor_start -= 1
            descriptor = self.raw[descriptor_start:position]
            if descriptor not in {b"", b"0"}:
                self._issue(
                    ShellIssueReason.UNSUPPORTED_SEMANTICS,
                    self._span(descriptor_start, position + 2) or statement_span,
                )
                return None
            if is_here_string:
                if redirect is None:
                    return None
                self.consumed_redirect_nodes.add(_node_key(redirect))
                argument = self._here_string_argument(redirect, position)
                span = self._span(position, redirect.end_byte)
                if argument is None or span is None:
                    return None
                here_facts.append(
                    _RedirectFact(
                        _RedirectKind.STDIN_HERE_STRING,
                        argument,
                        span,
                        position,
                        expands=True,
                    )
                )
                continue
            if redirect is not None:
                self.consumed_redirect_nodes.add(_node_key(redirect))
                candidate_redirects_by_position[position] = redirect
            operator_length = 3 if self.raw[position : position + 3] == b"<<-" else 2
            normalized = self._normalized_heredoc_delimiter(
                self.raw,
                position + operator_length,
                header_end,
            )
            if normalized is None:
                return None
            delimiter, quoted, word_end = normalized
            separator_after = self.raw[word_end : word_end + 1] == b";"
            specs.append(
                (position, delimiter, quoted, operator_length == 3, separator_after, word_end)
            )
            if separate_segment:
                separate_input_positions.add(position)

        facts: list[_RedirectFact] = list(here_facts)
        data_cursor = header_end + 1
        separator_after_by_position: dict[int, bool] = {}
        word_end_by_position: dict[int, int] = {}
        for position, delimiter, quoted, strip_tabs, separator_after, word_end in specs:
            separator_after_by_position[position] = separator_after
            word_end_by_position[position] = word_end
            body_start = data_cursor
            terminator_line: _PhysicalLine | None = None
            line_index = max(0, bisect_right(self.local_line_starts, data_cursor) - 1)
            while line_index < len(self.local_lines):
                line = self.local_lines[line_index]
                if line.start < data_cursor:
                    line_index += 1
                    continue
                candidate_start = line.start
                if strip_tabs:
                    while candidate_start < line.content_end and self.raw[candidate_start] == 9:
                        candidate_start += 1
                if self.raw[candidate_start : line.content_end] == delimiter:
                    terminator_line = line
                    break
                line_index += 1
            body_end = terminator_line.start if terminator_line is not None else len(self.raw)
            operator_span = self._span(position, position + (3 if strip_tabs else 2))
            if operator_span is None:
                return None
            if not self._recovery_step(max(0, body_end - body_start), operator_span):
                return None
            argument = self._recover_cst_anchored_body_argument(
                body_start,
                body_end,
                strip_tabs=strip_tabs,
                expands=not quoted,
                fallback_span=operator_span,
            )
            if argument is None:
                return None
            complete = terminator_line is not None
            facts.append(
                _RedirectFact(
                    _RedirectKind.STDIN_HEREDOC,
                    argument,
                    operator_span,
                    position,
                    expands=not quoted,
                    complete=complete,
                )
            )
            data_end = terminator_line.full_end if terminator_line is not None else len(self.raw)
            self._mask_data_range(body_start, data_end)
            data_cursor = data_end
            if not complete:
                self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, operator_span)
                break

        recovered: list[_RecoveredGeneratedDraft] = []
        ordered_inputs = sorted(facts, key=lambda item: item.local_start_byte)
        if len(separate_input_positions) > 1:
            self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, statement_span)
            return None
        for index, input_fact in enumerate(ordered_inputs[1:], start=1):
            previous_input = ordered_inputs[index - 1]
            if input_fact.local_start_byte not in separate_input_positions:
                continue
            if not separator_after_by_position.get(previous_input.local_start_byte, False):
                self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, input_fact.span)
                return None
            proxy = candidate_redirects_by_position.get(input_fact.local_start_byte)
            if proxy is None or proxy.end_byte != word_end_by_position.get(
                input_fact.local_start_byte
            ):
                self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, input_fact.span)
                return None
            local_inputs = [input_fact]
            local_outputs: list[_RedirectFact] = []
            local_redirect_nodes = sorted(
                (
                    node
                    for node in nodes
                    if node.type in {"file_redirect", "herestring_redirect"}
                    and previous_input.local_start_byte < node.start_byte < header_end
                ),
                key=lambda node: (node.start_byte, node.end_byte),
            )
            for redirect_node in local_redirect_nodes:
                if _node_key(redirect_node) == _node_key(proxy):
                    continue
                if redirect_node.type == "herestring_redirect":
                    operator = next(
                        (child for child in redirect_node.children if child.type == "<<<"),
                        None,
                    )
                    if operator is None:
                        self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, input_fact.span)
                        return None
                    here_argument = self._here_string_argument(
                        redirect_node,
                        operator.start_byte,
                    )
                    here_span = self._span(operator.start_byte, redirect_node.end_byte)
                    if here_argument is None or here_span is None:
                        return None
                    local_inputs.append(
                        _RedirectFact(
                            _RedirectKind.STDIN_HERE_STRING,
                            here_argument,
                            here_span,
                            operator.start_byte,
                            expands=True,
                        )
                    )
                    self.consumed_redirect_nodes.add(_node_key(redirect_node))
                    continue
                output, _extras, supported = self._redirect_parts(
                    _RedirectDraft(redirect_node, None, statement.start_byte),
                    allow_fact=True,
                )
                if not supported or output is None:
                    return None
                local_outputs.append(output)
                self.consumed_redirect_nodes.add(_node_key(redirect_node))
            if (
                not any(
                    output.local_start_byte < input_fact.local_start_byte
                    for output in local_outputs
                )
                or not local_outputs
            ):
                continue
            final_input = max(local_inputs, key=lambda item: item.local_start_byte)
            final_output = max(local_outputs, key=lambda item: item.local_start_byte)
            if (
                final_input.target.value.state is not StaticValueState.EXACT
                or final_output.target.value.state is not StaticValueState.EXACT
            ):
                self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, final_input.span)
                return None
            recovered.append(_RecoveredGeneratedDraft(final_input, final_output, function_id))
            for local_input in local_inputs:
                if local_input in facts:
                    facts.remove(local_input)

        for node in nodes:
            if node.is_missing and node.type == "heredoc_end":
                self.consumed_error_nodes.add(_node_key(node))
            elif node.is_error:
                raw = self.raw[node.start_byte : node.end_byte]
                allowed = (
                    raw == b"&"
                    or raw in {b"<", b"<<"}
                    or (
                        raw.endswith(b"<")
                        and any(child.type == "file_redirect" for child in node.children)
                    )
                )
                if not allowed:
                    return None
                self.consumed_error_nodes.add(_node_key(node))
            elif node.is_missing:
                return None
        return facts, recovered

    def _preceding_pending_heredoc_error(self, statement: Node) -> Node | None:
        separator = statement.prev_sibling
        candidate = separator.prev_sibling if separator is not None else None
        follows_pending_statement = candidate is not None and (
            candidate.end_point.row == statement.start_point.row
            or (
                separator is not None
                and self.raw[separator.end_byte : statement.start_byte].strip(b" \t") == b"\\\n"
            )
        )
        if (
            separator is None
            or separator.type not in {";", "&"}
            or candidate is None
            or not candidate.is_error
            or not follows_pending_statement
        ):
            return None
        child_types = {child.type for child in candidate.children}
        return candidate if child_types & {"<<", "<<-"} and "heredoc_start" in child_types else None

    def _prepare_redirected_statement(self, node: Node, function_id: int | None) -> None:
        preceding_pending = self._preceding_pending_heredoc_error(node)
        heredocs = tuple(
            child
            for index, child in enumerate(node.children)
            if node.field_name_for_child(index) == "redirect" and child.type == "heredoc_redirect"
        )
        native: list[_RedirectFact] = []
        native_complete = bool(heredocs)
        for heredoc in heredocs:
            fact = self._native_heredoc_fact(heredoc)
            if fact is None:
                native_complete = False
                break
            native.append(fact)
        if preceding_pending is not None:
            pending_span = self._node_span(preceding_pending) or self._point_span(
                preceding_pending.start_byte
            )
            if self._recovery_step(len(preceding_pending.children) + 2, pending_span):
                self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, pending_span)
            return
        direct_children = list(node.children)
        has_here_string_shape = any(
            child.is_error
            and self.raw[child.start_byte : child.end_byte] == b"<<"
            and index + 1 < len(direct_children)
            and direct_children[index + 1].type == "file_redirect"
            and direct_children[index + 1].start_byte == child.end_byte
            and self.raw[child.start_byte : direct_children[index + 1].start_byte + 1] == b"<<<"
            for index, child in enumerate(direct_children)
        )
        if native_complete and not has_here_string_shape:
            self.heredoc_facts_by_statement[node.start_byte] = native
            return
        if not heredocs and not has_here_string_shape:
            return
        recovered = self._recover_cst_anchored_heredoc_facts(node, heredocs, function_id)
        if recovered is None:
            return
        facts, generated = recovered
        self.heredoc_facts_by_statement[node.start_byte] = facts
        self.recovered_generated_drafts.extend(generated)

    def walk(self, root: Node) -> None:
        stack = [_WalkFrame(root, None, None, None, None, 0, None, None)]
        while stack and not self.halted:
            frame = stack.pop()
            node = frame.node
            if frame.exiting:
                event_kind = {
                    "command": _ShellEventKind.COMMAND,
                    "declaration_command": _ShellEventKind.COMMAND,
                    "variable_assignment": _ShellEventKind.ASSIGNMENT,
                    "unset_command": _ShellEventKind.UNSET,
                    "function_definition": _ShellEventKind.FUNCTION_DEFINITION,
                }.get(node.type)
                if event_kind is not None:
                    self.event_drafts.append(
                        _ShellEventDraft(
                            event_kind,
                            node,
                            frame.region_id,
                            frame.function_id,
                        )
                    )
                continue
            if frame.parent_type == "program" and self._is_masked_data_node(node):
                continue
            node_span = self._node_span(node) or self._point_span(node.start_byte)
            if exhaustion := self.file_budget.charge_shell_cst_visits(self.accounting_unit, 1):
                self._resource(exhaustion, node_span)
                break
            if node.is_error:
                if _node_key(node) in self.consumed_error_nodes:
                    continue
                if frame.command_owner_start_byte is not None:
                    self.syntax_error_command_starts.add(frame.command_owner_start_byte)
                if frame.redirect_owner_start_byte is not None:
                    self.syntax_error_redirect_starts.add(frame.redirect_owner_start_byte)
                self._issue(ShellIssueReason.SYNTAX_ERROR, node_span)
                continue
            if node.is_missing:
                if _node_key(node) in self.consumed_error_nodes:
                    continue
                if frame.command_owner_start_byte is not None:
                    self.syntax_error_command_starts.add(frame.command_owner_start_byte)
                if frame.redirect_owner_start_byte is not None:
                    self.syntax_error_redirect_starts.add(frame.redirect_owner_start_byte)
                self._issue(ShellIssueReason.SYNTAX_ERROR, node_span)
                continue

            region_id = frame.region_id
            function_id = frame.function_id
            substitution_depth = frame.substitution_depth
            command_owner_start_byte = frame.command_owner_start_byte
            redirect_owner_start_byte = frame.redirect_owner_start_byte
            if node.type in {"command", "declaration_command"}:
                command_owner_start_byte = node.start_byte
            elif node.type == "redirected_statement":
                redirect_owner_start_byte = node.start_byte
            kind = _REGION_NODE_TYPES.get(node.type)
            if kind is not None:
                if node.type in _SUBSTITUTION_NODE_TYPES:
                    substitution_depth += 1
                if node.type == "function_definition":
                    new_function_id = len(self.function_drafts)
                    if not self._reserve(1, span=node_span):
                        break
                    self.function_drafts.append(_FunctionDraft(node, new_function_id, function_id))
                    function_id = new_function_id
                region_id = self._add_region(
                    node,
                    kind,
                    parent_region_id=frame.region_id,
                    function_id=function_id,
                )
            if node.type == "for_statement":
                variable = node.child_by_field_name("variable")
                if variable is not None:
                    self.event_drafts.append(
                        _ShellEventDraft(
                            _ShellEventKind.LOOP_BINDING,
                            variable,
                            region_id,
                            function_id,
                        )
                    )
            elif node.type == "c_style_for_statement":
                self.event_drafts.append(
                    _ShellEventDraft(
                        _ShellEventKind.LOOP_UPDATE,
                        node,
                        region_id,
                        function_id,
                    )
                )

            if node.type in {"command", "declaration_command"}:
                if not self._reserve(1, span=node_span):
                    break
                self.command_drafts.append(
                    _CommandDraft(
                        node,
                        region_id,
                        function_id,
                        substitution_depth,
                        sum(
                            node.field_name_for_child(index) == "redirect"
                            and child.type != "file_redirect"
                            for index, child in enumerate(node.children)
                        ),
                    )
                )
            elif node.type == "variable_assignment":
                prefix = frame.parent_start_byte if frame.parent_type == "command" else None
                declaration = (
                    frame.parent_start_byte if frame.parent_type == "declaration_command" else None
                )
                if not self._reserve(1, span=node_span):
                    break
                self.assignment_drafts.append(
                    _AssignmentDraft(node, region_id, function_id, prefix, declaration)
                )
            elif node.type == "redirected_statement":
                self._prepare_redirected_statement(node, function_id)
                body = node.child_by_field_name("body")
                if body is not None:
                    if not self._reserve(1, span=node_span):
                        break
                    self.redirect_owners[node.start_byte] = _RedirectOwner(
                        node,
                        node.start_byte,
                        body.start_byte,
                        body.end_byte,
                        body.type,
                        substitution_depth,
                        any(
                            (child.is_error or child.is_missing)
                            and _node_key(child) not in self.consumed_error_nodes
                            for child in node.children
                        ),
                        sum(
                            node.field_name_for_child(index) == "redirect"
                            for index in range(len(node.children))
                        ),
                    )
            elif node.type == "file_redirect":
                if _node_key(node) in self.consumed_redirect_nodes:
                    continue
                if not self._reserve(1, span=node_span):
                    break
                self.redirect_drafts.append(
                    _RedirectDraft(
                        node,
                        frame.parent_start_byte if frame.parent_type == "command" else None,
                        (
                            frame.parent_start_byte
                            if frame.parent_type == "redirected_statement"
                            else frame.redirect_owner_start_byte
                        ),
                    )
                )
            elif node.type == "heredoc_redirect" and frame.parent_type == "redirected_statement":
                argument_nodes = tuple(
                    child
                    for index, child in enumerate(node.children)
                    if node.field_name_for_child(index) == "argument"
                )
                if argument_nodes:
                    if not self._reserve(1, span=node_span):
                        break
                    self.wrapper_argument_drafts.append(
                        _WrapperArgumentsDraft(
                            argument_nodes,
                            cast(int, frame.parent_start_byte),
                        )
                    )

            if node.type == "heredoc_body":
                continue

            child_frames: list[_WalkFrame] = []
            children = list(node.children)
            async_regions: dict[tuple[int, int, str], int | None] = {}
            pipeline_stage_regions: dict[tuple[int, int, str], int | None] = {}
            if node.type == "pipeline":
                for child in children:
                    if child.is_named:
                        pipeline_stage_regions[_node_key(child)] = self._add_region(
                            child,
                            _ExecutionRegionKind.PIPELINE_STAGE,
                            parent_region_id=region_id,
                            function_id=function_id,
                        )
                        if self.halted:
                            break
            if self.halted:
                break
            for index, child in enumerate(children[:-1]):
                if child.is_named and children[index + 1].type == "&":
                    async_regions[_node_key(child)] = self._add_region(
                        child,
                        _ExecutionRegionKind.ASYNC,
                        parent_region_id=region_id,
                        function_id=function_id,
                    )
                    if self.halted:
                        break
            if self.halted:
                break
            previous_child: Node | None = None
            for child in children:
                child_command_owner = command_owner_start_byte
                child_redirect_owner = redirect_owner_start_byte
                if (
                    (child.is_error or child.is_missing)
                    and previous_child is not None
                    and previous_child.end_point.row == child.start_point.row
                ):
                    if previous_child.type in {"command", "declaration_command"}:
                        child_command_owner = previous_child.start_byte
                    elif previous_child.type == "redirected_statement":
                        child_redirect_owner = previous_child.start_byte
                child_frames.append(
                    _WalkFrame(
                        child,
                        node.type,
                        node.start_byte,
                        async_regions.get(
                            _node_key(child),
                            pipeline_stage_regions.get(_node_key(child), region_id),
                        ),
                        function_id,
                        substitution_depth,
                        child_command_owner,
                        child_redirect_owner,
                    )
                )
                previous_child = child
            stack.append(
                _WalkFrame(
                    node,
                    frame.parent_type,
                    frame.parent_start_byte,
                    region_id,
                    function_id,
                    substitution_depth,
                    command_owner_start_byte,
                    redirect_owner_start_byte,
                    True,
                )
            )
            stack.extend(reversed(child_frames))

    def _command_groups(self, node: Node) -> list[_NodeGroup]:
        nodes: list[Node] = []
        if node.type == "declaration_command":
            if node.children:
                nodes.append(node.children[0])
            nodes.extend(node.named_children)
        else:
            for index, child in enumerate(node.children):
                if child.type == "variable_assignment" or node.field_name_for_child(index) in {
                    "name",
                    "argument",
                }:
                    nodes.append(child)
        return self._group_nodes(nodes)

    @staticmethod
    def _redirect_owner_command_start(
        owner: _RedirectOwner,
        commands_by_depth: dict[int, tuple[_CommandDraft, ...]],
        starts_by_depth: dict[int, tuple[int, ...]],
    ) -> int | None:
        if owner.body_type in _COMPOUND_REDIRECT_BODIES:
            return None
        commands = commands_by_depth.get(owner.substitution_depth, ())
        starts = starts_by_depth.get(owner.substitution_depth, ())
        left = bisect_left(starts, owner.body_start_byte)
        right = bisect_left(starts, owner.body_end_byte)
        for index in range(right - 1, left - 1, -1):
            command = commands[index]
            if command.node.end_byte <= owner.body_end_byte:
                return command.node.start_byte
        return None

    def _redirect_command_start(
        self,
        draft: _RedirectDraft,
        commands_by_depth: dict[int, tuple[_CommandDraft, ...]],
        starts_by_depth: dict[int, tuple[int, ...]],
    ) -> int | None:
        if draft.command_start_byte is not None:
            return draft.command_start_byte
        if draft.statement_start_byte is None:
            return None
        owner = self.redirect_owners.get(draft.statement_start_byte)
        if owner is None:
            return None
        return self._redirect_owner_command_start(owner, commands_by_depth, starts_by_depth)

    def _redirect_parts(
        self,
        draft: _RedirectDraft,
        *,
        allow_fact: bool,
    ) -> tuple[_RedirectFact | None, list[_ArgumentIR], bool]:
        node = draft.node
        descriptor = node.child_by_field_name("descriptor")
        destination_nodes = [
            child
            for index, child in enumerate(node.children)
            if node.field_name_for_child(index) == "destination"
        ]
        groups = self._group_nodes(destination_nodes)
        extras: list[_ArgumentIR] = []
        if not groups:
            self._issue(
                ShellIssueReason.UNSUPPORTED_SEMANTICS,
                self._node_span(node) or self._point_span(node.start_byte),
            )
            return None, extras, False
        for group in groups[1:]:
            argument = self._argument(group)
            if argument is not None:
                extras.append(argument)
            if self.halted:
                return None, extras, False
        operator_node = next(
            (
                child
                for child in node.children
                if not child.is_named and child.type in _SUPPORTED_REDIRECT_OPERATORS
            ),
            None,
        )
        operator = operator_node.type if operator_node is not None else ""
        descriptor_raw = (
            self.raw[descriptor.start_byte : descriptor.end_byte] if descriptor is not None else b""
        )
        kind: _RedirectKind | None = None
        if operator == ">" and descriptor_raw in {b"", b"1"}:
            kind = _RedirectKind.STDOUT_TRUNCATE
        elif operator == ">|" and descriptor_raw in {b"", b"1"}:
            kind = _RedirectKind.STDOUT_CLOBBER
        elif operator == ">>" and descriptor_raw in {b"", b"1"}:
            kind = _RedirectKind.STDOUT_APPEND
        elif operator == "&>" and not descriptor_raw:
            kind = _RedirectKind.STDOUT_STDERR_TRUNCATE
        raw_operator = next(
            (
                child.type
                for child in node.children
                if not child.is_named and child.type not in {"\n"}
            ),
            "",
        )
        unsupported = kind is None and (
            raw_operator in {"<", "<>", "<<", "<<-", "<<<", "&>>", ">&", "<&"}
            or descriptor is not None
            or operator in _SUPPORTED_REDIRECT_OPERATORS
        )
        if unsupported:
            self._issue(
                ShellIssueReason.UNSUPPORTED_SEMANTICS,
                self._node_span(node) or self._point_span(node.start_byte),
            )
            return None, extras, False
        if kind is None:
            return None, extras, True
        if not allow_fact:
            return None, extras, False
        target = self._argument(groups[0])
        if target is None:
            return None, extras, False
        span = self._node_span(node)
        if span is None:
            self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, self._point_span(node.start_byte))
            return None, extras, False
        if not self._reserve(1, span=span):
            return None, extras, False
        return _RedirectFact(kind, target, span, node.start_byte), extras, True

    def _functions(self) -> tuple[_FunctionContext, ...]:
        functions: list[_FunctionContext] = []
        function_regions = {
            region.function_id: region
            for region in self.regions
            if region.kind is _ExecutionRegionKind.FUNCTION and region.function_id is not None
        }
        for draft in self.function_drafts:
            name_node = draft.node.child_by_field_name("name")
            if name_node is None:
                continue
            group = _NodeGroup(
                (name_node,),
                name_node.start_byte,
                name_node.end_byte,
                self.raw[name_node.start_byte : name_node.end_byte],
            )
            argument = self._argument(group)
            if argument is None:
                if self.halted:
                    break
                continue
            if not self._reserve(1, span=argument.span):
                break
            functions.append(
                _FunctionContext(
                    draft.function_id,
                    argument.value,
                    argument.span,
                    argument.fragments,
                    draft.parent_function_id,
                    (
                        function_regions[draft.function_id].parent_region_id
                        if draft.function_id in function_regions
                        else None
                    ),
                    draft.node.start_byte,
                    draft.node.end_byte,
                )
            )
        return tuple(functions)

    def _region_chain(self, region_id: int | None) -> tuple[_ExecutionRegion, ...]:
        cached = self.region_chain_cache.get(region_id)
        if cached is not None:
            return cached
        chain: list[_ExecutionRegion] = []
        original_region_id = region_id
        while region_id is not None:
            region = self.regions[region_id]
            chain.append(region)
            region_id = region.parent_region_id
        retained = tuple(chain)
        self.region_chain_cache[original_region_id] = retained
        return retained

    def _control_role(self, draft: _ShellEventDraft) -> _ControlRole:
        chain = self._region_chain(draft.region_id)
        if any(region.kind is _ExecutionRegionKind.ASYNC for region in chain):
            return _ControlRole.ASYNC
        for region in chain:
            if region.kind is _ExecutionRegionKind.LIST:
                split = region.control_split_byte
                return (
                    _ControlRole.BOOLEAN_LEFT
                    if split is None or draft.node.start_byte < split
                    else _ControlRole.BOOLEAN_RIGHT
                )
        if any(
            region.kind
            in {
                _ExecutionRegionKind.IF,
                _ExecutionRegionKind.ELIF,
                _ExecutionRegionKind.ELSE,
                _ExecutionRegionKind.CASE,
                _ExecutionRegionKind.CASE_ITEM,
                _ExecutionRegionKind.NEGATION,
            }
            for region in chain
        ):
            return _ControlRole.CONDITIONAL
        if any(
            region.kind
            in {
                _ExecutionRegionKind.FOR,
                _ExecutionRegionKind.C_STYLE_FOR,
                _ExecutionRegionKind.WHILE,
                _ExecutionRegionKind.UNTIL,
                _ExecutionRegionKind.DO,
            }
            for region in chain
        ):
            return _ControlRole.LOOP
        for index, region in enumerate(chain):
            if region.kind is not _ExecutionRegionKind.COMPOUND:
                continue
            parent_kind = chain[index + 1].kind if index + 1 < len(chain) else None
            if parent_kind is not _ExecutionRegionKind.FUNCTION:
                return _ControlRole.GROUP
        if any(
            region.kind
            in {
                _ExecutionRegionKind.PIPELINE,
                _ExecutionRegionKind.PIPELINE_STAGE,
                _ExecutionRegionKind.SUBSHELL,
                _ExecutionRegionKind.COMMAND_SUBSTITUTION,
                _ExecutionRegionKind.PROCESS_SUBSTITUTION,
            }
            for region in chain
        ):
            return _ControlRole.ISOLATED
        if draft.function_id is not None:
            return _ControlRole.FUNCTION
        return _ControlRole.STRAIGHT

    def _scope_region_id(self, region_id: int | None) -> int | None:
        for region in self._region_chain(region_id):
            if region.kind in {
                _ExecutionRegionKind.PIPELINE_STAGE,
                _ExecutionRegionKind.SUBSHELL,
                _ExecutionRegionKind.COMMAND_SUBSTITUTION,
                _ExecutionRegionKind.PROCESS_SUBSTITUTION,
            }:
                return region.region_id
        return None

    def _uncertain_scope_ids(self, draft: _ShellEventDraft) -> tuple[int, ...]:
        chain = self._region_chain(draft.region_id)
        explicit_arm_kinds = {
            _ExecutionRegionKind.ELIF,
            _ExecutionRegionKind.ELSE,
            _ExecutionRegionKind.CASE_ITEM,
        }
        shadowed_containers: set[int] = set()
        nearest_if_id: int | None = None
        nearest_case_id: int | None = None
        for region in reversed(chain):
            if region.kind is _ExecutionRegionKind.IF:
                nearest_if_id = region.region_id
            elif region.kind is _ExecutionRegionKind.CASE:
                nearest_case_id = region.region_id
            elif region.kind in {_ExecutionRegionKind.ELIF, _ExecutionRegionKind.ELSE}:
                if nearest_if_id is not None:
                    shadowed_containers.add(nearest_if_id)
                    self.branch_container_ids[region.region_id] = nearest_if_id
            elif region.kind is _ExecutionRegionKind.CASE_ITEM and nearest_case_id is not None:
                shadowed_containers.add(nearest_case_id)
                self.branch_container_ids[region.region_id] = nearest_case_id
        uncertain_kinds = {
            _ExecutionRegionKind.ASYNC,
            _ExecutionRegionKind.LIST,
            *explicit_arm_kinds,
            _ExecutionRegionKind.IF,
            _ExecutionRegionKind.CASE,
            _ExecutionRegionKind.NEGATION,
            _ExecutionRegionKind.FOR,
            _ExecutionRegionKind.C_STYLE_FOR,
            _ExecutionRegionKind.WHILE,
            _ExecutionRegionKind.UNTIL,
        }
        scopes: list[int] = []
        for index, region in enumerate(chain):
            include = region.kind in uncertain_kinds and region.region_id not in shadowed_containers
            if region.kind is _ExecutionRegionKind.COMPOUND:
                parent_kind = chain[index + 1].kind if index + 1 < len(chain) else None
                include = parent_kind is not _ExecutionRegionKind.FUNCTION
            if include:
                scopes.append(region.region_id)
        return tuple(reversed(scopes))

    @staticmethod
    def _lookup_binding(frame: _StateFrame, name: str) -> _BindingIR:
        current: _StateFrame | None = frame
        while current is not None:
            if name in current.bindings:
                return current.bindings[name]
            current = current.parent
        return _BindingIR(name, StaticValue.unknown(), _ExportState.UNKNOWN, ())

    @staticmethod
    def _persistent_bindings_are_unknown(frame: _StateFrame) -> bool:
        current: _StateFrame | None = frame
        while current is not None:
            if current.persistent_bindings_unknown:
                return True
            current = current.parent
        return False

    @staticmethod
    def _nameref_declaration_mode(
        argv: tuple[StaticValue, ...],
        arguments: tuple[_ArgumentIR, ...] | list[_ArgumentIR],
        declaration_starts: set[int],
    ) -> bool:
        if (
            not argv
            or argv[0].state is not StaticValueState.EXACT
            or cast(bytes, argv[0].exact_bytes) not in {b"declare", b"local", b"typeset"}
        ):
            return False
        return any(
            argument.value.state is not StaticValueState.EXACT
            or b"n" in cast(bytes, argument.value.exact_bytes)[1:]
            for argument in arguments[1:]
            if argument.local_start_byte not in declaration_starts
            and (
                argument.value.state is not StaticValueState.EXACT
                or cast(bytes, argument.value.exact_bytes).startswith((b"-", b"+"))
            )
        )

    @staticmethod
    def _lookup_function(frame: _StateFrame, name: bytes) -> tuple[bool, int | None]:
        current: _StateFrame | None = frame
        while current is not None:
            if name in current.functions:
                return True, current.functions[name]
            if current.functions_unknown:
                return True, None
            current = current.parent
        return False, None

    def _resolve_value(
        self,
        value: StaticValue,
        fragments: tuple[_ValueFragment, ...],
        atoms: tuple[_ValueAtom, ...],
        frame: _StateFrame,
        *,
        assignment_value: bool,
        span: SourceSpan,
    ) -> tuple[StaticValue, tuple[_ValueFragment, ...]]:
        if value.state is StaticValueState.EXACT or not atoms:
            return value, fragments
        output = bytearray()
        resolved_fragments: list[_ValueFragment] = []
        saw_unbound = False
        for atom in atoms:
            if atom.kind is _ValueAtomKind.LITERAL:
                if atom.value.state is not StaticValueState.EXACT:
                    return StaticValue.unknown(), fragments
                atom_value = cast(bytes, atom.value.exact_bytes)
                cursor = len(output)
                output.extend(atom_value)
                resolved_fragments.extend(
                    replace(
                        fragment,
                        value_start_byte=cursor + fragment.value_start_byte,
                        value_end_byte=cursor + fragment.value_end_byte,
                    )
                    for fragment in atom.fragments
                )
                continue
            if atom.name is None or (not assignment_value and not atom.quoted):
                return StaticValue.unknown(), fragments
            binding = self._lookup_binding(frame, atom.name)
            if binding.value.state is StaticValueState.UNKNOWN:
                return StaticValue.unknown(), fragments
            if binding.value.state is StaticValueState.UNBOUND:
                saw_unbound = True
                continue
            binding_value = cast(bytes, binding.value.exact_bytes)
            cursor = len(output)
            output.extend(binding_value)
            resolved_fragments.extend(
                replace(
                    fragment,
                    value_start_byte=cursor + fragment.value_start_byte,
                    value_end_byte=cursor + fragment.value_end_byte,
                )
                for fragment in binding.fragments
            )
        if saw_unbound and assignment_value:
            return StaticValue.unknown(), ()
        if saw_unbound and len(atoms) == 1 and not output:
            return StaticValue.unbound(), ()
        exact = StaticValue.exact(bytes(output))
        if not self._reserve(
            len(resolved_fragments),
            value_bytes=len(output),
            span=span,
        ):
            return StaticValue.unknown(), fragments
        return exact, tuple(resolved_fragments)

    def _resolve_assignment(
        self,
        assignment: _AssignmentIR,
        frame: _StateFrame,
    ) -> _AssignmentIR:
        value, fragments = self._resolve_value(
            assignment.site.value,
            assignment.value_fragments,
            assignment.value_atoms,
            frame,
            assignment_value=True,
            span=assignment.site.span,
        )
        return replace(
            assignment,
            site=replace(assignment.site, value=value),
            value_fragments=fragments,
            value_atoms=(),
        )

    def _state_frame(
        self,
        draft: _ShellEventDraft,
        root: _StateFrame,
        frames: dict[tuple[str, int], _StateFrame],
        persistent_unknown_function_ids: set[int],
    ) -> _StateFrame:
        base = root
        if draft.function_id is not None:
            key = ("function", draft.function_id)
            frame = frames.get(key)
            if frame is None:
                frame = self._new_state_frame(None)
                source = root
                parent_function_id = self.function_drafts[draft.function_id].parent_function_id
                if parent_function_id is not None:
                    source = frames.get(("function", parent_function_id), root)
                frame.functions.update(_visible_functions(source))
                frame.functions_unknown = _visible_functions_unknown(source)
                frame.persistent_bindings_unknown = (
                    draft.function_id in persistent_unknown_function_ids
                    or self._persistent_bindings_are_unknown(source)
                )
                frames[key] = frame
            base = frame
        scope_region_id = self._scope_region_id(draft.region_id)
        if scope_region_id is None:
            return base
        key = ("region", scope_region_id)
        frame = frames.get(key)
        if frame is not None:
            return frame
        parent = base
        parent_region_id = self.regions[scope_region_id].parent_region_id
        while parent_region_id is not None:
            candidate = self._scope_region_id(parent_region_id)
            if candidate is None or candidate == scope_region_id:
                break
            parent = frames.get(("region", candidate), base)
            break
        frame = self._new_state_frame(parent)
        frames[key] = frame
        return frame

    def _new_state_frame(self, parent: _StateFrame | None) -> _StateFrame:
        frame = _StateFrame(self.next_state_frame_id, parent)
        self.next_state_frame_id += 1
        return frame

    @staticmethod
    def _uncertain_role(role: _ControlRole) -> bool:
        return role in {
            _ControlRole.CONDITIONAL,
            _ControlRole.BOOLEAN_LEFT,
            _ControlRole.BOOLEAN_RIGHT,
            _ControlRole.LOOP,
            _ControlRole.GROUP,
            _ControlRole.ASYNC,
        }

    def _apply_binding(
        self,
        *,
        frame: _StateFrame,
        binding: _BindingIR,
        role: _ControlRole,
        draft: _ShellEventDraft,
        updates: list[_StateUpdateIR],
    ) -> None:
        retained = (
            _BindingIR(
                binding.name,
                StaticValue.unknown(),
                _ExportState.UNKNOWN,
                (),
            )
            if binding.name in frame.sticky_unknown_names
            or binding.name in frame.persistent_unknown_names
            or self._persistent_bindings_are_unknown(frame)
            or draft.function_id in self.persistent_unknown_function_ids
            else binding
        )
        if not self._reserve(
            2,
            span=retained.fragments[0].span
            if retained.fragments
            else (self._node_span(draft.node) or self._point_span(draft.node.start_byte)),
        ):
            return
        frame.bindings[binding.name] = retained
        updates.append(
            _StateUpdateIR(
                len(updates),
                draft.region_id,
                draft.function_id,
                frame.frame_id,
                retained,
            )
        )

    def _event_name(self, draft: _ShellEventDraft) -> str | None:
        if draft.kind is _ShellEventKind.ASSIGNMENT:
            name = draft.node.child_by_field_name("name")
            return (
                self._identifier(self.raw[name.start_byte : name.end_byte])
                if name is not None
                else None
            )
        if draft.kind is _ShellEventKind.LOOP_UPDATE and draft.node.type == "c_style_for_statement":
            return None
        if draft.kind in {_ShellEventKind.LOOP_BINDING, _ShellEventKind.LOOP_UPDATE}:
            return self._identifier(self.raw[draft.node.start_byte : draft.node.end_byte])
        if draft.kind is _ShellEventKind.FUNCTION_DEFINITION:
            name = draft.node.child_by_field_name("name")
            return (
                self._identifier(self.raw[name.start_byte : name.end_byte])
                if name is not None
                else None
            )
        if draft.kind is _ShellEventKind.UNSET:
            named = draft.node.named_children
            if len(named) == 1 and named[0].type == "variable_name":
                return self._identifier(self.raw[named[0].start_byte : named[0].end_byte])
        return None

    def _has_pipeline_input(self, command: _CommandIR) -> bool:
        if self.pipeline_input_region_ids is None:
            grouped: dict[int | None, list[_ExecutionRegion]] = {}
            regions_by_id = {region.region_id: region for region in self.regions}
            for region in self.regions:
                if region.kind is _ExecutionRegionKind.PIPELINE_STAGE:
                    grouped.setdefault(region.parent_region_id, []).append(region)
            input_ids: set[int] = set()
            for stages in grouped.values():
                ordered = sorted(stages, key=lambda item: item.local_start_byte)
                parent_id = ordered[0].parent_region_id if ordered else None
                pipeline = regions_by_id.get(parent_id) if parent_id is not None else None
                first_input = bool(
                    pipeline is not None
                    and self.raw[pipeline.local_start_byte : pipeline.local_start_byte + 1] == b"|"
                )
                input_ids.update(region.region_id for region in ordered[0 if first_input else 1 :])
            self.pipeline_input_region_ids = frozenset(input_ids)
        stage = next(
            (
                region
                for region in self._region_chain(command.region_id)
                if region.kind is _ExecutionRegionKind.PIPELINE_STAGE
            ),
            None,
        )
        return stage is not None and stage.region_id in self.pipeline_input_region_ids

    def _nested_request(
        self,
        command: _CommandIR,
        frame: _StateFrame,
        *,
        depth: int,
    ) -> _NestedRequest | None:
        argv = command.site.argv
        if (
            command.resolution.kind is _CommandResolutionKind.AMBIGUOUS
            and argv[0].state is StaticValueState.EXACT
            and cast(bytes, argv[0].exact_bytes) in {b"bash", b"sh", b"dash", b"eval"}
        ):
            self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, command.site.span)
            return None
        if command.resolution.kind is not _CommandResolutionKind.EXTERNAL:
            return None
        if argv[0].state is not StaticValueState.EXACT:
            return None
        executable = cast(bytes, argv[0].exact_bytes)
        if executable == b"env" and len(argv) > 1 and argv[1] == StaticValue.exact(b"-S"):
            self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, command.site.span)
            return None
        if executable == b"xargs":
            self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, command.site.span)
            return None
        if executable in {b"bash", b"sh", b"dash"}:
            if len(argv) > 1 and argv[1] in {
                StaticValue.exact(b"-c"),
                StaticValue.exact(b"-lc"),
            }:
                if len(command.arguments) < 3 or argv[2].state is not StaticValueState.EXACT:
                    span = (
                        command.arguments[2].span
                        if len(command.arguments) > 2
                        else command.site.span
                    )
                    self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, span)
                    return None
                return _NestedRequest(
                    _NestedExecutionKind.SHELL,
                    command,
                    command.arguments[2],
                    frame,
                    depth + 1,
                )
            if self._has_pipeline_input(command):
                self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, command.site.span)
            return None
        if executable == b"eval":
            if len(command.arguments) != 2 or argv[1].state is not StaticValueState.EXACT:
                span = (
                    command.arguments[1].span if len(command.arguments) > 1 else command.site.span
                )
                self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, span)
                return None
            return _NestedRequest(
                _NestedExecutionKind.EVAL,
                command,
                command.arguments[1],
                frame,
                depth + 1,
            )
        return None

    def _merge_control_frame(
        self,
        *,
        frame: _StateFrame,
        parent: _StateFrame,
        scope_id: int,
        draft: _ShellEventDraft,
        updates: list[_StateUpdateIR],
    ) -> None:
        span = self.regions[scope_id].span
        for name, candidate in frame.bindings.items():
            previous = self._lookup_binding(parent, name)
            value = previous.value if candidate.value == previous.value else StaticValue.unknown()
            export_state = (
                previous.export_state
                if candidate.export_state is previous.export_state
                else _ExportState.UNKNOWN
            )
            fragments = previous.fragments if value == previous.value else ()
            merged = _BindingIR(name, value, export_state, fragments)
            if not self._reserve(2, span=span):
                return
            parent.bindings[name] = merged
            updates.append(
                _StateUpdateIR(
                    len(updates),
                    scope_id,
                    draft.function_id,
                    parent.frame_id,
                    merged,
                )
            )
        for function_name, function_id in frame.functions.items():
            found, previous_function = self._lookup_function(parent, function_name)
            parent.functions[function_name] = (
                function_id if found and previous_function == function_id else None
            )
        parent.functions_unknown = parent.functions_unknown or frame.functions_unknown
        parent.persistent_unknown_names.update(frame.persistent_unknown_names)
        parent.persistent_bindings_unknown = (
            parent.persistent_bindings_unknown or frame.persistent_bindings_unknown
        )

    def _resolve_redirect_fact(
        self,
        fact: _RedirectFact,
        frame: _StateFrame,
    ) -> _RedirectFact:
        input_fact = fact.kind in {
            _RedirectKind.STDIN_HEREDOC,
            _RedirectKind.STDIN_HERE_STRING,
        }
        unquoted_output_variable = not input_fact and any(
            atom.kind is _ValueAtomKind.VARIABLE and not atom.quoted for atom in fact.target.atoms
        )
        value, fragments = self._resolve_value(
            fact.target.value,
            fact.target.fragments,
            fact.target.atoms,
            frame,
            assignment_value=input_fact,
            span=fact.target.span,
        )
        if unquoted_output_variable:
            if value.state is StaticValueState.UNBOUND:
                value = StaticValue.unknown()
                fragments = ()
            proof = None
        else:
            proof = self._generated_value_proof(fact.target, value, frame)
        if fact.expands and value.state is StaticValueState.EXACT:
            fragments = tuple(
                fragment
                for fragment in fragments
                if fact.target.local_start_byte <= fragment.local_start_byte
                and fragment.local_end_byte <= fact.target.local_end_byte
            )
        return replace(
            fact,
            target=replace(
                fact.target,
                value=value,
                fragments=fragments,
                atoms=(),
            ),
            value_proof=proof,
            execution_limitation_span=(fact.target.span if unquoted_output_variable else None),
        )

    def _generated_value_proof(
        self,
        argument: _ArgumentIR,
        resolved_value: StaticValue,
        frame: _StateFrame,
    ) -> _GeneratedValueProof | None:
        """Retain only ordered value uncertainty needed by generated-config dispatch."""
        if resolved_value.state is StaticValueState.UNBOUND:
            return None
        output = bytearray()
        entries: list[_GeneratedProofEntry] = []
        unknown_ranges: list[tuple[int, int]] = []
        unknowns_quoted = True
        saw_variable = False
        if not argument.atoms:
            if resolved_value.state is not StaticValueState.EXACT:
                return None
            output.extend(cast(bytes, resolved_value.exact_bytes))
            entries.extend(
                _GeneratedProofEntry(
                    fragment.value_start_byte,
                    fragment.value_end_byte,
                    fragment.span.start_byte,
                    fragment.span.end_byte,
                )
                for fragment in argument.fragments
                if fragment.exact and fragment.value_end_byte > fragment.value_start_byte
            )
        else:
            for atom in argument.atoms:
                child_start = len(output)
                if atom.kind is _ValueAtomKind.LITERAL:
                    if atom.value.state is not StaticValueState.EXACT:
                        return None
                    literal = cast(bytes, atom.value.exact_bytes)
                    output.extend(literal)
                    for fragment in atom.fragments:
                        if (
                            not fragment.exact
                            or fragment.value_end_byte <= fragment.value_start_byte
                            or fragment.span.end_byte - fragment.span.start_byte
                            != fragment.value_end_byte - fragment.value_start_byte
                        ):
                            return None
                        entries.append(
                            _GeneratedProofEntry(
                                child_start + fragment.value_start_byte,
                                child_start + fragment.value_end_byte,
                                fragment.span.start_byte,
                                fragment.span.end_byte,
                            )
                        )
                    continue
                saw_variable = True
                if atom.name is None:
                    return None
                binding = self._lookup_binding(frame, atom.name)
                if binding.value.state is StaticValueState.EXACT:
                    value = cast(bytes, binding.value.exact_bytes)
                else:
                    value = _GENERATED_UNKNOWN_MARKER
                    unknown_ranges.append((child_start, child_start + len(value)))
                    unknowns_quoted = unknowns_quoted and atom.quoted
                output.extend(value)
                if value:
                    entries.append(
                        _GeneratedProofEntry(
                            child_start,
                            child_start + len(value),
                            atom.span.start_byte,
                            atom.span.end_byte,
                        )
                    )
        entries.sort(key=lambda item: item.child_start_byte)
        if any(
            right.child_start_byte < left.child_end_byte
            or right.physical_start_byte < left.physical_end_byte
            for left, right in zip(entries, entries[1:], strict=False)
        ):
            return None
        transformed = saw_variable or not entries or entries[0].child_start_byte != 0
        transformed = transformed or entries[-1].child_end_byte != len(output)
        transformed = transformed or any(
            right.child_start_byte != left.child_end_byte
            or right.physical_start_byte != left.physical_end_byte
            for left, right in zip(entries, entries[1:], strict=False)
        )
        if not transformed:
            return None
        if resolved_value.state is StaticValueState.EXACT and bytes(output) != cast(
            bytes, resolved_value.exact_bytes
        ):
            return None
        if not self._reserve(
            1 + len(entries) + len(unknown_ranges),
            value_bytes=len(output),
            span=argument.span,
        ):
            return None
        parent_map = self.unit.source_map
        return _GeneratedValueProof(
            raw_bytes=bytes(output),
            entries=tuple(entries),
            unknown_ranges=tuple(unknown_ranges),
            unknowns_quoted=unknowns_quoted,
            path=self.unit.origin_span.path,
            physical_size_bytes=(
                parent_map.physical_size_bytes if parent_map is not None else len(self.raw)
            ),
            physical_line_starts=(
                parent_map.physical_line_starts
                if parent_map is not None
                else self.local_line_starts
            ),
        )

    def _generated_source_map(self, content: _ArgumentIR) -> SourceMap | None:
        if content.value.state is not StaticValueState.EXACT:
            return None
        entries: list[SourceMapEntry] = []
        for fragment in sorted(content.fragments, key=lambda item: item.value_start_byte):
            if (
                not fragment.exact
                or fragment.value_end_byte <= fragment.value_start_byte
                or fragment.span.end_byte - fragment.span.start_byte
                != fragment.value_end_byte - fragment.value_start_byte
            ):
                return None
            candidate = SourceMapEntry(
                fragment.value_start_byte,
                fragment.value_end_byte,
                fragment.span.start_byte,
                fragment.span.end_byte,
            )
            if entries and (
                candidate.child_start_byte < entries[-1].child_end_byte
                or candidate.physical_start_byte < entries[-1].physical_end_byte
            ):
                return None
            if (
                entries
                and entries[-1].child_end_byte == candidate.child_start_byte
                and entries[-1].physical_end_byte == candidate.physical_start_byte
            ):
                previous = entries[-1]
                entries[-1] = SourceMapEntry(
                    previous.child_start_byte,
                    candidate.child_end_byte,
                    previous.physical_start_byte,
                    candidate.physical_end_byte,
                )
            else:
                entries.append(candidate)
        if exhaustion := self.file_budget.charge_source_map_entries(len(entries)):
            self._resource(exhaustion, content.span)
            return None
        parent_map = self.unit.source_map
        physical_size = parent_map.physical_size_bytes if parent_map is not None else len(self.raw)
        physical_line_starts = (
            parent_map.physical_line_starts if parent_map is not None else self.local_line_starts
        )
        try:
            return SourceMap(
                path=self.unit.origin_span.path,
                entries=tuple(entries),
                child_size_bytes=len(cast(bytes, content.value.exact_bytes)),
                physical_size_bytes=physical_size,
                physical_line_starts=physical_line_starts,
            )
        except ValueError:
            self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, content.span)
            return None

    def _generated_config(
        self,
        input_fact: _RedirectFact,
        output_fact: _RedirectFact,
        *,
        producer_command_start_byte: int | None,
        producer_function_id: int | None,
    ) -> GeneratedConfig | None:
        target = output_fact.target.value
        content = input_fact.target.value
        if target.state is StaticValueState.UNBOUND or content.state is StaticValueState.UNBOUND:
            return None
        source_map = self._generated_source_map(input_fact.target)
        if content.state is StaticValueState.EXACT and source_map is None:
            self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, input_fact.target.span)
            return None
        span = SourceSpan(
            output_fact.span.path,
            min(output_fact.span.start_byte, input_fact.span.start_byte),
            max(output_fact.span.end_byte, input_fact.target.span.end_byte),
            min(output_fact.span.start_line, input_fact.span.start_line),
            max(output_fact.span.end_line, input_fact.target.span.end_line),
        )
        if not self._reserve(1, span=span):
            return None
        parent_map = self.unit.source_map
        return _ProvenGeneratedConfig(
            unit_id=self.unit.unit_id,
            provenance=SiteProvenance.GENERATED_CONFIG,
            span=span,
            target=target,
            content=content,
            source_map=source_map,
            target_proof=output_fact.value_proof,
            home_relative_target=output_fact.target.home_relative_target,
            content_proof=input_fact.value_proof,
            physical_size_bytes=(
                parent_map.physical_size_bytes if parent_map is not None else len(self.raw)
            ),
            physical_line_starts=(
                parent_map.physical_line_starts
                if parent_map is not None
                else self.local_line_starts
            ),
            producer_program_id=self.unit.unit_id,
            producer_command_start_byte=producer_command_start_byte,
            producer_function_id=producer_function_id,
            execution_limitation_span=output_fact.execution_limitation_span,
        )

    def _tee_output_fact(self, command: _CommandIR) -> _RedirectFact | None:
        argv = command.site.argv
        if (
            not argv
            or argv[0].state is not StaticValueState.EXACT
            or cast(bytes, argv[0].exact_bytes) != b"tee"
        ):
            return None
        if command.resolution.kind is not _CommandResolutionKind.EXTERNAL:
            self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, command.site.span)
            return None
        append = False
        end_options = False
        outputs: list[_ArgumentIR] = []
        for argument in command.arguments[1:]:
            if argument.value.state is not StaticValueState.EXACT:
                self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, argument.span)
                return None
            literal = cast(bytes, argument.value.exact_bytes)
            if not end_options and literal == b"--":
                end_options = True
                continue
            if not end_options and literal in {b"-a", b"--append"}:
                append = True
                continue
            if not end_options and literal.startswith(b"-"):
                self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, argument.span)
                return None
            outputs.append(argument)
        if len(outputs) != 1:
            if outputs:
                self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, command.site.span)
            return None
        output = outputs[0]
        return _RedirectFact(
            _RedirectKind.STDOUT_APPEND if append else _RedirectKind.STDOUT_TRUNCATE,
            output,
            output.span,
            output.local_start_byte,
        )

    def _generated_from_command(self, command: _CommandIR) -> tuple[GeneratedConfig, ...]:
        if not command.redirects_complete:
            return ()
        stdin: _RedirectFact | None = None
        stdout: _RedirectFact | None = None
        for fact in sorted(command.redirects, key=lambda item: item.local_start_byte):
            if fact.kind in {
                _RedirectKind.STDIN_HEREDOC,
                _RedirectKind.STDIN_HERE_STRING,
            }:
                stdin = fact
            elif fact.kind in {
                _RedirectKind.STDOUT_TRUNCATE,
                _RedirectKind.STDOUT_CLOBBER,
                _RedirectKind.STDOUT_APPEND,
                _RedirectKind.STDOUT_STDERR_TRUNCATE,
            }:
                stdout = fact
        if stdin is None:
            return ()
        outputs = [stdout] if stdout is not None else []
        tee_output = self._tee_output_fact(command)
        if tee_output is not None:
            outputs.append(tee_output)
        return tuple(
            generated
            for output in outputs
            if (
                generated := self._generated_config(
                    stdin,
                    output,
                    producer_command_start_byte=command.site.span.start_byte,
                    producer_function_id=command.function_id,
                )
            )
            is not None
        )

    def _model_state(
        self,
        program: _ShellProgramIR,
        *,
        initial_bindings: dict[str, _BindingIR] | None = None,
        initial_functions: dict[bytes, int | None] | None = None,
        initial_functions_unknown: bool = False,
        depth: int = 0,
    ) -> Generator[_NestedRequest, _NestedResponse, _ModeledState]:
        commands = list(program.commands)
        assignments = list(program.assignments)
        generated_configs: list[GeneratedConfig] = []
        commands_by_start = {
            command.local_start_byte: index for index, command in enumerate(commands)
        }
        assignments_by_start: dict[int, list[int]] = {}
        prefix_assignments_by_command: dict[int, list[int]] = {}
        declaration_assignments_by_command: dict[int, list[int]] = {}
        for index, assignment in enumerate(assignments):
            assignments_by_start.setdefault(assignment.local_start_byte, []).append(index)
            if assignment.prefix_for_command_start_byte is not None:
                prefix_assignments_by_command.setdefault(
                    assignment.prefix_for_command_start_byte, []
                ).append(index)
            if assignment.declaration_command_start_byte is not None:
                declaration_assignments_by_command.setdefault(
                    assignment.declaration_command_start_byte, []
                ).append(index)
        functions_by_start = {function.local_start_byte: function for function in program.functions}
        functions_by_id = {function.function_id: function for function in program.functions}
        function_binding_writes: dict[int, set[str]] = {}
        function_export_writes: dict[int, set[str]] = {}
        function_function_writes: dict[int, set[bytes]] = {}
        function_binding_writes_unknown: set[int] = set()
        function_export_writes_unknown: set[int] = set()
        function_function_writes_unknown: set[int] = set()
        loop_binding_writes: dict[int, set[str]] = {}
        loop_export_writes: dict[int, set[str]] = {}
        loop_function_writes: dict[int, set[bytes]] = {}
        loop_binding_writes_unknown: set[int] = set()
        loop_export_writes_unknown: set[int] = set()
        loop_function_writes_unknown: set[int] = set()
        loop_command_names: dict[int, set[bytes]] = {}
        loop_dynamic_commands: set[int] = set()
        loop_region_kinds = {
            _ExecutionRegionKind.FOR,
            _ExecutionRegionKind.C_STYLE_FOR,
            _ExecutionRegionKind.WHILE,
            _ExecutionRegionKind.UNTIL,
        }
        isolated_region_kinds = {
            _ExecutionRegionKind.PIPELINE_STAGE,
            _ExecutionRegionKind.SUBSHELL,
            _ExecutionRegionKind.COMMAND_SUBSTITUTION,
            _ExecutionRegionKind.PROCESS_SUBSTITUTION,
        }
        functions_by_scope_and_name: dict[tuple[int | None, bytes], list[int]] = {}
        function_names_by_scope: dict[int | None, set[bytes]] = {}
        function_ids_by_scope: dict[int | None, list[int]] = {}
        function_isolation_scopes: dict[int, int | None] = {}
        uncertain_function_ids: set[int] = set()
        for function in program.functions:
            function_ids_by_scope.setdefault(function.parent_function_id, []).append(
                function.function_id
            )
            function_isolation_scopes[function.function_id] = self._scope_region_id(
                function.definition_region_id
            )
            if function.name.state is StaticValueState.EXACT:
                function_name = cast(bytes, function.name.exact_bytes)
                function_names_by_scope.setdefault(function.parent_function_id, set()).add(
                    function_name
                )
                functions_by_scope_and_name.setdefault(
                    (
                        function.parent_function_id,
                        function_name,
                    ),
                    [],
                ).append(function.function_id)
            definition_draft = _ShellEventDraft(
                _ShellEventKind.FUNCTION_DEFINITION,
                self.function_drafts[function.function_id].node,
                function.definition_region_id,
                function.parent_function_id,
            )
            if self._uncertain_scope_ids(definition_draft):
                uncertain_function_ids.add(function.function_id)
        function_calls: dict[int, set[int]] = {}
        barrier_function_lexical_calls: dict[int, set[int]] = {}
        barrier_function_call_names: dict[int, set[bytes]] = {}
        barrier_function_dynamic_calls: set[int] = set()
        root_function_call_events: list[tuple[_ShellEventDraft, bytes | None]] = []
        persistent_barrier_call_events: list[tuple[_ShellEventDraft, bytes | None]] = []
        persistent_barrier_scopes: set[tuple[int | None, int | None]] = set()
        persistent_unknown_function_ids: set[int] = set()

        def possible_function_ids(
            event: _ShellEventDraft,
            name: bytes,
            *,
            include_future: bool = False,
        ) -> tuple[int, ...]:
            scope_id = event.function_id
            possible: list[int] = []
            event_region_ids = {region.region_id for region in self._region_chain(event.region_id)}
            while True:
                candidates = tuple(
                    candidate
                    for candidate in functions_by_scope_and_name.get((scope_id, name), ())
                    if function_isolation_scopes[candidate] is None
                    or function_isolation_scopes[candidate] in event_region_ids
                )
                if include_future:
                    possible.extend(candidates)
                else:
                    preceding = tuple(
                        candidate
                        for candidate in candidates
                        if functions_by_id[candidate].local_start_byte < event.node.start_byte
                    )
                    if preceding:
                        latest_unconditional = next(
                            (
                                index
                                for index in range(len(preceding) - 1, -1, -1)
                                if preceding[index] not in uncertain_function_ids
                            ),
                            None,
                        )
                        if latest_unconditional is not None:
                            possible.extend(preceding[latest_unconditional:])
                            return tuple(possible)
                        possible.extend(preceding)
                if scope_id is None:
                    return tuple(possible)
                scope_id = functions_by_id[scope_id].parent_function_id

        def possible_dynamic_function_ids(event: _ShellEventDraft) -> tuple[int, ...]:
            scope_id = event.function_id
            include_future = scope_id is not None
            possible: list[int] = []
            event_region_ids = {region.region_id for region in self._region_chain(event.region_id)}
            shadowed_names: set[bytes] = set()
            while True:
                exact_ids: set[int] = set()
                for name in function_names_by_scope.get(scope_id, ()):
                    candidates = tuple(
                        candidate
                        for candidate in functions_by_scope_and_name.get((scope_id, name), ())
                        if (
                            include_future
                            or functions_by_id[candidate].local_start_byte < event.node.start_byte
                        )
                        and (
                            function_isolation_scopes[candidate] is None
                            or function_isolation_scopes[candidate] in event_region_ids
                        )
                    )
                    exact_ids.update(candidates)
                    if not candidates or name in shadowed_names:
                        continue
                    if include_future:
                        possible.extend(candidates)
                        continue
                    latest_unconditional = next(
                        (
                            index
                            for index in range(len(candidates) - 1, -1, -1)
                            if candidates[index] not in uncertain_function_ids
                        ),
                        None,
                    )
                    if latest_unconditional is None:
                        possible.extend(candidates)
                    else:
                        possible.extend(candidates[latest_unconditional:])
                        shadowed_names.add(name)
                possible.extend(
                    candidate
                    for candidate in function_ids_by_scope.get(scope_id, ())
                    if candidate not in exact_ids
                    and (
                        include_future
                        or functions_by_id[candidate].local_start_byte < event.node.start_byte
                    )
                    and (
                        function_isolation_scopes[candidate] is None
                        or function_isolation_scopes[candidate] in event_region_ids
                    )
                )
                if scope_id is None:
                    return tuple(possible)
                scope_id = functions_by_id[scope_id].parent_function_id

        for event in self.event_drafts:
            write_region_id = event.region_id
            write_function_id = event.function_id
            binding_names: set[str] = set()
            export_names: set[str] = set()
            function_names: set[bytes] = set()
            bindings_unknown = False
            exports_unknown = False
            functions_unknown = False
            if event.kind is _ShellEventKind.ASSIGNMENT:
                for assignment_index in assignments_by_start.get(event.node.start_byte, ()):
                    assignment = assignments[assignment_index]
                    if assignment.prefix_for_command_start_byte is None:
                        binding_names.add(assignment.site.name)
                        if assignment.declaration_keyword == b"export":
                            export_names.add(assignment.site.name)
            elif event.kind is _ShellEventKind.LOOP_BINDING:
                if event_name := self._event_name(event):
                    binding_names.add(event_name)
            elif event.kind is _ShellEventKind.LOOP_UPDATE:
                event_name = self._event_name(event)
                if event_name is None:
                    bindings_unknown = True
                else:
                    binding_names.add(event_name)
            elif event.kind is _ShellEventKind.UNSET:
                unset_names = {
                    name
                    for child in event.node.named_children
                    if child.type == "variable_name"
                    and (name := self._identifier(self.raw[child.start_byte : child.end_byte]))
                    is not None
                }
                if unset_names:
                    binding_names.update(unset_names)
                    export_names.update(unset_names)
                    function_names.update(name.encode("ascii") for name in unset_names)
                else:
                    bindings_unknown = True
                    exports_unknown = True
                    functions_unknown = True
            elif event.kind is _ShellEventKind.FUNCTION_DEFINITION:
                function_context = functions_by_start.get(event.node.start_byte)
                if function_context is None:
                    functions_unknown = True
                else:
                    write_region_id = function_context.definition_region_id
                    write_function_id = function_context.parent_function_id
                    if function_context.name.state is StaticValueState.EXACT:
                        function_names.add(cast(bytes, function_context.name.exact_bytes))
                    else:
                        functions_unknown = True
            elif event.kind is _ShellEventKind.COMMAND:
                command_index = commands_by_start.get(event.node.start_byte)
                if command_index is not None:
                    pre_model_command = commands[command_index]
                    pre_model_name = pre_model_command.site.argv[0]
                    if event.function_id is None:
                        root_function_call_events.append(
                            (
                                event,
                                cast(bytes, pre_model_name.exact_bytes)
                                if pre_model_name.state is StaticValueState.EXACT
                                else None,
                            )
                        )
                    declaration_starts = {
                        assignments[index].local_start_byte
                        for index in declaration_assignments_by_command.get(
                            event.node.start_byte,
                            (),
                        )
                    }
                    if self._nameref_declaration_mode(
                        pre_model_command.site.argv,
                        pre_model_command.arguments,
                        declaration_starts,
                    ):
                        persistent_barrier_scopes.add(
                            (event.function_id, self._scope_region_id(event.region_id))
                        )
                    event_chain = self._region_chain(event.region_id)
                    event_region_ids = {region.region_id for region in event_chain}
                    persistent_barrier_active = (
                        event.function_id,
                        None,
                    ) in persistent_barrier_scopes or any(
                        (event.function_id, region.region_id) in persistent_barrier_scopes
                        for region in event_chain
                        if region.kind in isolated_region_kinds
                    )
                    callees = tuple(
                        candidate
                        for candidate in (
                            possible_function_ids(
                                event,
                                cast(bytes, pre_model_name.exact_bytes),
                                include_future=event.function_id is not None,
                            )
                            if pre_model_name.state is StaticValueState.EXACT
                            and (event.function_id is not None or persistent_barrier_active)
                            else ()
                        )
                        if function_isolation_scopes[candidate] is None
                        or function_isolation_scopes[candidate] in event_region_ids
                    )
                    if event.function_id is not None:
                        if pre_model_name.state is StaticValueState.EXACT:
                            barrier_function_call_names.setdefault(event.function_id, set()).add(
                                cast(bytes, pre_model_name.exact_bytes)
                            )
                        else:
                            barrier_function_dynamic_calls.add(event.function_id)
                        barrier_callees = (
                            callees
                            if pre_model_name.state is StaticValueState.EXACT
                            else possible_dynamic_function_ids(event)
                        )
                        lexical_barrier_callees = {
                            candidate
                            for candidate in barrier_callees
                            if functions_by_id[candidate].parent_function_id is not None
                        }
                        if lexical_barrier_callees:
                            barrier_function_lexical_calls.setdefault(
                                event.function_id, set()
                            ).update(lexical_barrier_callees)
                    if persistent_barrier_active:
                        persistent_barrier_call_events.append(
                            (
                                event,
                                cast(bytes, pre_model_name.exact_bytes)
                                if pre_model_name.state is StaticValueState.EXACT
                                else None,
                            )
                        )
                    if pre_model_name == StaticValue.exact(b"eval"):
                        bindings_unknown = True
                        exports_unknown = True
                        functions_unknown = True
                    elif pre_model_name == StaticValue.exact(b"export"):
                        for argument in pre_model_command.arguments[1:]:
                            if argument.local_start_byte in declaration_starts:
                                continue
                            if argument.value.state is not StaticValueState.EXACT:
                                exports_unknown = True
                                continue
                            argument_bytes = cast(bytes, argument.value.exact_bytes)
                            if argument_bytes == b"-n":
                                continue
                            if export_name := self._identifier(argument_bytes):
                                export_names.add(export_name)
                    if (
                        write_function_id is not None
                        and pre_model_name.state is StaticValueState.EXACT
                        and not any(
                            region.kind in isolated_region_kinds
                            for region in self._region_chain(write_region_id)
                        )
                    ):
                        if callees:
                            function_calls.setdefault(write_function_id, set()).update(callees)
                    elif (
                        write_function_id is not None
                        and pre_model_name.state is not StaticValueState.EXACT
                    ):
                        bindings_unknown = True
                        exports_unknown = True
                        functions_unknown = True
            if not (
                binding_names
                or export_names
                or function_names
                or bindings_unknown
                or exports_unknown
                or functions_unknown
            ):
                continue
            write_chain = self._region_chain(write_region_id)
            if write_function_id is not None and not any(
                region.kind in isolated_region_kinds for region in write_chain
            ):
                function_binding_writes.setdefault(write_function_id, set()).update(binding_names)
                function_export_writes.setdefault(write_function_id, set()).update(export_names)
                function_function_writes.setdefault(write_function_id, set()).update(function_names)
                if bindings_unknown:
                    function_binding_writes_unknown.add(write_function_id)
                if exports_unknown:
                    function_export_writes_unknown.add(write_function_id)
                if functions_unknown:
                    function_function_writes_unknown.add(write_function_id)
            crossed_isolation = False
            for region in write_chain:
                if region.kind in isolated_region_kinds:
                    crossed_isolation = True
                if (
                    region.kind in loop_region_kinds
                    and not crossed_isolation
                    and region.function_id == write_function_id
                ):
                    loop_binding_writes.setdefault(region.region_id, set()).update(binding_names)
                    loop_export_writes.setdefault(region.region_id, set()).update(export_names)
                    loop_function_writes.setdefault(region.region_id, set()).update(function_names)
                    if bindings_unknown:
                        loop_binding_writes_unknown.add(region.region_id)
                    if exports_unknown:
                        loop_export_writes_unknown.add(region.region_id)
                    if functions_unknown:
                        loop_function_writes_unknown.add(region.region_id)

        def call_event_function_ids(
            event: _ShellEventDraft,
            name: bytes | None,
        ) -> tuple[int, ...]:
            return (
                possible_function_ids(
                    event,
                    name,
                    include_future=event.function_id is not None,
                )
                if name is not None
                else possible_dynamic_function_ids(event)
            )

        def summarized_function_callees(
            event: _ShellEventDraft,
            function_id: int,
        ) -> set[int]:
            callees = set(barrier_function_lexical_calls.get(function_id, ()))
            for call_name in barrier_function_call_names.get(function_id, ()):
                callees.update(call_event_function_ids(event, call_name))
            if function_id in barrier_function_dynamic_calls:
                callees.update(possible_dynamic_function_ids(event))
            return callees

        def apply_persistent_barrier(
            resolution_event: _ShellEventDraft,
            barrier_name: bytes | None,
        ) -> None:
            pending_persistent_functions = list(
                call_event_function_ids(resolution_event, barrier_name)
            )
            visited_persistent_functions: set[int] = set()
            while pending_persistent_functions:
                pending_function_id = pending_persistent_functions.pop()
                if pending_function_id in visited_persistent_functions:
                    continue
                visited_persistent_functions.add(pending_function_id)
                persistent_unknown_function_ids.add(pending_function_id)
                pending_persistent_functions.extend(
                    summarized_function_callees(
                        resolution_event,
                        pending_function_id,
                    )
                )

        barrier_names_by_function: dict[int, set[bytes | None]] = {}
        for barrier_event, barrier_name in persistent_barrier_call_events:
            apply_persistent_barrier(barrier_event, barrier_name)
            if barrier_event.function_id is not None:
                barrier_names_by_function.setdefault(barrier_event.function_id, set()).add(
                    barrier_name
                )

        for root_event, root_name in root_function_call_events:
            pending_origin_functions = list(call_event_function_ids(root_event, root_name))
            visited_origin_functions: set[int] = set()
            while pending_origin_functions:
                origin_function_id = pending_origin_functions.pop()
                if origin_function_id in visited_origin_functions:
                    continue
                visited_origin_functions.add(origin_function_id)
                for barrier_name in barrier_names_by_function.get(origin_function_id, ()):
                    apply_persistent_barrier(root_event, barrier_name)
                pending_origin_functions.extend(
                    summarized_function_callees(root_event, origin_function_id)
                )
        self.persistent_unknown_function_ids = frozenset(persistent_unknown_function_ids)

        function_effect_cache: dict[
            int,
            tuple[
                frozenset[str],
                frozenset[str],
                frozenset[bytes],
                bool,
                bool,
                bool,
            ],
        ] = {}
        function_effect_cache_items = 0

        def function_effects(
            function_id: int,
        ) -> tuple[
            frozenset[str],
            frozenset[str],
            frozenset[bytes],
            bool,
            bool,
            bool,
        ]:
            nonlocal function_effect_cache_items
            cached = function_effect_cache.get(function_id)
            if cached is not None:
                return cached
            binding_names: set[str] = set()
            export_names: set[str] = set()
            function_names: set[bytes] = set()
            bindings_unknown = False
            exports_unknown = False
            functions_unknown = False
            pending = [function_id]
            visited: set[int] = set()
            while pending:
                candidate = pending.pop()
                if candidate in visited:
                    continue
                visited.add(candidate)
                if candidate not in functions_by_id:
                    bindings_unknown = True
                    exports_unknown = True
                    functions_unknown = True
                    continue
                binding_names.update(function_binding_writes.get(candidate, ()))
                export_names.update(function_export_writes.get(candidate, ()))
                function_names.update(function_function_writes.get(candidate, ()))
                bindings_unknown = bindings_unknown or candidate in function_binding_writes_unknown
                exports_unknown = exports_unknown or candidate in function_export_writes_unknown
                functions_unknown = (
                    functions_unknown or candidate in function_function_writes_unknown
                )
                pending.extend(function_calls.get(candidate, ()))
            effect = (
                frozenset(binding_names),
                frozenset(export_names),
                frozenset(function_names),
                bindings_unknown,
                exports_unknown,
                functions_unknown,
            )
            effect_items = len(binding_names) + len(export_names) + len(function_names) + 3
            if function_effect_cache_items + effect_items <= MAX_DEPENDENCY_RETAINED_SHELL_IR:
                function_effect_cache[function_id] = effect
                function_effect_cache_items += effect_items
            return effect

        for event in self.event_drafts:
            if event.kind is not _ShellEventKind.COMMAND:
                continue
            command_index = commands_by_start.get(event.node.start_byte)
            if command_index is None:
                continue
            pre_model_name = commands[command_index].site.argv[0]
            event_loop_ids: list[int] = []
            crossed_isolation = False
            for region in self._region_chain(event.region_id):
                if region.kind in isolated_region_kinds:
                    crossed_isolation = True
                if (
                    region.kind in loop_region_kinds
                    and not crossed_isolation
                    and region.function_id == event.function_id
                ):
                    event_loop_ids.append(region.region_id)
            for loop_id in event_loop_ids:
                if pre_model_name.state is StaticValueState.EXACT:
                    loop_command_names.setdefault(loop_id, set()).add(
                        cast(bytes, pre_model_name.exact_bytes)
                    )
                else:
                    loop_dynamic_commands.add(loop_id)
            if pre_model_name.state is not StaticValueState.EXACT:
                continue
            callees = possible_function_ids(
                event,
                cast(bytes, pre_model_name.exact_bytes),
                include_future=event.function_id is not None,
            )
            if not callees:
                continue
            callee_binding_write_names: set[str] = set()
            callee_export_write_names: set[str] = set()
            callee_function_write_names: set[bytes] = set()
            callee_bindings_unknown = False
            callee_exports_unknown = False
            callee_functions_unknown = False
            for callee in callees:
                (
                    nested_binding_names,
                    nested_export_names,
                    nested_function_names,
                    nested_bindings_unknown,
                    nested_exports_unknown,
                    nested_functions_unknown,
                ) = function_effects(callee)
                callee_binding_write_names.update(nested_binding_names)
                callee_export_write_names.update(nested_export_names)
                callee_function_write_names.update(nested_function_names)
                callee_bindings_unknown = callee_bindings_unknown or nested_bindings_unknown
                callee_exports_unknown = callee_exports_unknown or nested_exports_unknown
                callee_functions_unknown = callee_functions_unknown or nested_functions_unknown
            for loop_id in event_loop_ids:
                loop_binding_writes.setdefault(loop_id, set()).update(callee_binding_write_names)
                loop_export_writes.setdefault(loop_id, set()).update(callee_export_write_names)
                loop_function_writes.setdefault(loop_id, set()).update(callee_function_write_names)
                if callee_bindings_unknown:
                    loop_binding_writes_unknown.add(loop_id)
                if callee_exports_unknown:
                    loop_export_writes_unknown.add(loop_id)
                if callee_functions_unknown:
                    loop_function_writes_unknown.add(loop_id)

        function_mutation_positions: dict[int | None, dict[bytes, list[int]]] = {}
        function_namespace_unknown_positions: dict[int | None, list[int]] = {}
        for event in self.event_drafts:
            if event.kind is _ShellEventKind.FUNCTION_DEFINITION:
                mutation_function = functions_by_start.get(event.node.start_byte)
                if (
                    mutation_function is not None
                    and mutation_function.name.state is StaticValueState.EXACT
                ):
                    function_mutation_positions.setdefault(
                        mutation_function.parent_function_id, {}
                    ).setdefault(cast(bytes, mutation_function.name.exact_bytes), []).append(
                        event.node.start_byte
                    )
                continue
            if event.kind is _ShellEventKind.UNSET:
                unset_names = {
                    name
                    for child in event.node.named_children
                    if child.type == "variable_name"
                    and (name := self._identifier(self.raw[child.start_byte : child.end_byte]))
                    is not None
                }
                if not unset_names:
                    function_namespace_unknown_positions.setdefault(event.function_id, []).append(
                        event.node.start_byte
                    )
                    continue
                for unset_name in unset_names:
                    function_mutation_positions.setdefault(event.function_id, {}).setdefault(
                        unset_name.encode("ascii"), []
                    ).append(event.node.start_byte)
                continue
            if event.kind is not _ShellEventKind.COMMAND:
                continue
            command_index = commands_by_start.get(event.node.start_byte)
            if command_index is None:
                continue
            pre_model_command_name = commands[command_index].site.argv[0]
            if (
                pre_model_command_name.state is not StaticValueState.EXACT
                or pre_model_command_name == StaticValue.exact(b"eval")
            ):
                function_namespace_unknown_positions.setdefault(event.function_id, []).append(
                    event.node.start_byte
                )
        for mutations in function_mutation_positions.values():
            for positions in mutations.values():
                positions.sort()
        for positions in function_namespace_unknown_positions.values():
            positions.sort()
        function_resolution_change_cache: dict[tuple[int, bytes], bool] = {}

        def function_resolution_may_change(function_id: int, name: bytes) -> bool:
            cache_key = (function_id, name)
            cached = function_resolution_change_cache.get(cache_key)
            if cached is not None:
                return cached
            function = functions_by_id[function_id]
            scope_id = function.parent_function_id
            while True:
                name_positions = function_mutation_positions.get(scope_id, {}).get(name, ())
                if bisect_right(name_positions, function.local_end_byte) < len(name_positions):
                    function_resolution_change_cache[cache_key] = True
                    return True
                unknown_positions = function_namespace_unknown_positions.get(scope_id, ())
                if bisect_right(unknown_positions, function.local_end_byte) < len(
                    unknown_positions
                ):
                    function_resolution_change_cache[cache_key] = True
                    return True
                if scope_id is None:
                    break
                scope_id = functions_by_id[scope_id].parent_function_id
            function_resolution_change_cache[cache_key] = False
            return False

        root = _StateFrame(
            0,
            None,
            dict(initial_bindings or {}),
            dict(initial_functions or {}),
            initial_functions_unknown,
        )
        frames: dict[tuple[str, int], _StateFrame] = {}
        control_frames: dict[int, _StateFrame] = {}
        control_parents: dict[int, _StateFrame] = {}
        active_control_scopes: list[int] = []
        updates: list[_StateUpdateIR] = []
        events: list[_StateEventIR] = []
        modeled_assignment_indices: set[int] = set()
        command_execution_order = 0
        opaque_initial_function_names = frozenset(initial_functions or {})

        def enter_control_scope(
            scope_id: int,
            parent: _StateFrame,
            draft: _ShellEventDraft,
        ) -> _StateFrame:
            retained = control_frames.get(scope_id)
            if retained is not None:
                return retained
            retained = self._new_state_frame(parent)
            control_frames[scope_id] = retained
            control_parents[scope_id] = parent
            active_control_scopes.append(scope_id)
            if self.regions[scope_id].kind not in loop_region_kinds:
                return retained

            called_names = loop_command_names.get(scope_id, set())
            has_dynamic_call = scope_id in loop_dynamic_commands
            value_names = set(loop_binding_writes.get(scope_id, ()))
            export_names = set(loop_export_writes.get(scope_id, ()))
            function_names = set(loop_function_writes.get(scope_id, ()))
            bindings_unknown = scope_id in loop_binding_writes_unknown
            exports_unknown = scope_id in loop_export_writes_unknown
            functions_unknown = scope_id in loop_function_writes_unknown
            for called_name in called_names:
                found, function_id = self._lookup_function(retained, called_name)
                if not found:
                    continue
                (
                    live_binding_names,
                    live_export_names,
                    live_function_names,
                    live_bindings_unknown,
                    live_exports_unknown,
                    live_functions_unknown,
                ) = function_effects(
                    _IMPORTED_FUNCTION_ID
                    if called_name in opaque_initial_function_names
                    or function_id is None
                    or function_id not in functions_by_id
                    else function_id
                )
                value_names.update(live_binding_names)
                export_names.update(live_export_names)
                function_names.update(live_function_names)
                bindings_unknown = bindings_unknown or live_bindings_unknown
                exports_unknown = exports_unknown or live_exports_unknown
                functions_unknown = functions_unknown or live_functions_unknown
            if has_dynamic_call:
                bindings_unknown = True
                exports_unknown = True
                functions_unknown = True
            if bindings_unknown:
                value_names.update(_visible_bindings(retained))
            if exports_unknown:
                export_names.update(_visible_bindings(retained))
            for binding_name in sorted(value_names | export_names):
                previous = self._lookup_binding(retained, binding_name)
                self._apply_binding(
                    frame=retained,
                    binding=_BindingIR(
                        binding_name,
                        (StaticValue.unknown() if binding_name in value_names else previous.value),
                        (
                            _ExportState.UNKNOWN
                            if binding_name in export_names
                            else previous.export_state
                        ),
                        () if binding_name in value_names else previous.fragments,
                    ),
                    role=_ControlRole.LOOP,
                    draft=draft,
                    updates=updates,
                )
            for function_name in sorted(function_names):
                retained.functions[function_name] = None
            if functions_unknown:
                for function_name, function_id in _visible_functions(retained).items():
                    retained.functions[function_name] = (
                        _IMPORTED_FUNCTION_ID if function_id == _IMPORTED_FUNCTION_ID else None
                    )
                retained.functions_unknown = True
            return retained

        for draft in self.event_drafts:
            if self.halted:
                break
            effective_draft = draft
            if draft.kind is _ShellEventKind.FUNCTION_DEFINITION:
                definition_function = functions_by_start.get(draft.node.start_byte)
                if definition_function is not None:
                    effective_draft = replace(
                        draft,
                        region_id=definition_function.definition_region_id,
                        function_id=definition_function.parent_function_id,
                    )
            effective_chain = self._region_chain(effective_draft.region_id)
            current_region_ids = {region.region_id for region in effective_chain}
            for scope_id in tuple(reversed(active_control_scopes)):
                if scope_id in current_region_ids:
                    continue
                scope_kind = self.regions[scope_id].kind
                if (
                    scope_kind
                    in {
                        _ExecutionRegionKind.ELIF,
                        _ExecutionRegionKind.ELSE,
                        _ExecutionRegionKind.CASE_ITEM,
                    }
                    and self.branch_container_ids.get(scope_id) in current_region_ids
                ):
                    continue
                self._merge_control_frame(
                    frame=control_frames[scope_id],
                    parent=control_parents[scope_id],
                    scope_id=scope_id,
                    draft=effective_draft,
                    updates=updates,
                )
                active_control_scopes.remove(scope_id)
            span = self._node_span(draft.node) or self._point_span(draft.node.start_byte)
            role = self._control_role(effective_draft)
            name = self._event_name(draft)
            if not self._reserve(1, span=span):
                break
            events.append(
                _StateEventIR(
                    len(events),
                    draft.kind,
                    role,
                    span,
                    effective_draft.region_id,
                    effective_draft.function_id,
                    draft.node.start_byte,
                    draft.node.end_byte,
                    name,
                )
            )
            frame = self._state_frame(
                effective_draft,
                root,
                frames,
                persistent_unknown_function_ids,
            )
            control_scope_ids = self._uncertain_scope_ids(effective_draft)
            if control_scope_ids:
                region_positions = {
                    region.region_id: position for position, region in enumerate(effective_chain)
                }
                isolation_scope_id = self._scope_region_id(effective_draft.region_id)
                isolation_position = (
                    region_positions[isolation_scope_id] if isolation_scope_id is not None else None
                )
                outer_control_ids = (
                    tuple(
                        scope_id
                        for scope_id in control_scope_ids
                        if region_positions[scope_id] > isolation_position
                    )
                    if isolation_position is not None
                    else ()
                )
                inner_control_ids = tuple(
                    scope_id
                    for scope_id in control_scope_ids
                    if isolation_position is None or region_positions[scope_id] < isolation_position
                )
                parent = frame.parent or root if outer_control_ids else frame
                for scope_id in outer_control_ids:
                    parent = enter_control_scope(scope_id, parent, effective_draft)
                if outer_control_ids:
                    frame.parent = parent
                    parent = frame
                for scope_id in inner_control_ids:
                    parent = enter_control_scope(scope_id, parent, effective_draft)
                frame = parent
                if self.halted:
                    break

            if draft.kind is _ShellEventKind.ASSIGNMENT:
                for assignment_index in assignments_by_start.get(draft.node.start_byte, ()):
                    assignment = assignments[assignment_index]
                    if (
                        assignment.prefix_for_command_start_byte is not None
                        or assignment.declaration_keyword is not None
                    ):
                        continue
                    assignment = self._resolve_assignment(assignment, frame)
                    assignments[assignment_index] = assignment
                    modeled_assignment_indices.add(assignment_index)
                    previous = self._lookup_binding(frame, assignment.site.name)
                    export_state = (
                        _ExportState.UNEXPORTED
                        if previous.value.state is StaticValueState.UNBOUND
                        else previous.export_state
                    )
                    self._apply_binding(
                        frame=frame,
                        binding=_BindingIR(
                            assignment.site.name,
                            assignment.site.value,
                            export_state,
                            assignment.value_fragments,
                        ),
                        role=role,
                        draft=draft,
                        updates=updates,
                    )
                continue

            if draft.kind in {
                _ShellEventKind.LOOP_BINDING,
                _ShellEventKind.LOOP_UPDATE,
            }:
                if draft.kind is _ShellEventKind.LOOP_UPDATE:
                    self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, span)
                affected_names = (
                    (name,)
                    if name is not None
                    else (
                        tuple(_visible_bindings(frame))
                        if draft.kind is _ShellEventKind.LOOP_UPDATE
                        else ()
                    )
                )
                for binding_name in affected_names:
                    self._apply_binding(
                        frame=frame,
                        binding=_BindingIR(
                            binding_name,
                            StaticValue.unknown(),
                            _ExportState.UNKNOWN,
                            (),
                        ),
                        role=_ControlRole.LOOP,
                        draft=draft,
                        updates=updates,
                    )
                if draft.kind is _ShellEventKind.LOOP_UPDATE:
                    frame.sticky_unknown_names.update(affected_names)
                continue

            if draft.kind is _ShellEventKind.UNSET:
                named = draft.node.named_children
                if name is not None:
                    frame.functions[name.encode("ascii")] = None
                    self._apply_binding(
                        frame=frame,
                        binding=_BindingIR(
                            name,
                            StaticValue.unbound(),
                            _ExportState.UNEXPORTED,
                            (),
                        ),
                        role=role,
                        draft=draft,
                        updates=updates,
                    )
                else:
                    self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, span)
                    changed = False
                    for child in named:
                        if child.type != "variable_name":
                            continue
                        child_name = self._identifier(self.raw[child.start_byte : child.end_byte])
                        if child_name is not None:
                            changed = True
                            frame.functions[self.raw[child.start_byte : child.end_byte]] = None
                            self._apply_binding(
                                frame=frame,
                                binding=_BindingIR(
                                    child_name,
                                    StaticValue.unknown(),
                                    _ExportState.UNKNOWN,
                                    (),
                                ),
                                role=role,
                                draft=draft,
                                updates=updates,
                            )
                    if not changed:
                        for binding_name in tuple(_visible_bindings(frame)):
                            self._apply_binding(
                                frame=frame,
                                binding=_BindingIR(
                                    binding_name,
                                    StaticValue.unknown(),
                                    _ExportState.UNKNOWN,
                                    (),
                                ),
                                role=role,
                                draft=draft,
                                updates=updates,
                            )
                        for function_name in tuple(_visible_functions(frame)):
                            frame.functions[function_name] = None
                continue

            if draft.kind is _ShellEventKind.FUNCTION_DEFINITION:
                modeled_function = functions_by_start.get(draft.node.start_byte)
                if (
                    modeled_function is not None
                    and modeled_function.name.state is StaticValueState.EXACT
                ):
                    function_name_bytes = cast(bytes, modeled_function.name.exact_bytes)
                    frame.functions[function_name_bytes] = modeled_function.function_id
                continue

            if draft.kind is not _ShellEventKind.COMMAND:
                continue
            command_index = commands_by_start.get(draft.node.start_byte)
            if command_index is None:
                continue
            command = commands[command_index]
            prefix_bindings: list[_BindingIR] = []
            prefix_assignment_indices = prefix_assignments_by_command.get(
                command.local_start_byte, []
            )
            declaration_indices = declaration_assignments_by_command.get(
                command.local_start_byte, []
            )
            declaration_starts = {
                assignments[index].local_start_byte for index in declaration_indices
            }
            prefix_name_counts: dict[str, int] = {}
            for assignment_index in prefix_assignment_indices:
                prefix_name = assignments[assignment_index].site.name
                prefix_name_counts[prefix_name] = prefix_name_counts.get(prefix_name, 0) + 1
            for assignment_index in prefix_assignment_indices:
                original_assignment = assignments[assignment_index]
                assignment = self._resolve_assignment(original_assignment, frame)
                if any(
                    atom.kind is _ValueAtomKind.VARIABLE
                    and atom.name in prefix_name_counts
                    and (
                        atom.name != original_assignment.site.name
                        or prefix_name_counts[original_assignment.site.name] > 1
                    )
                    for atom in original_assignment.value_atoms
                ):
                    assignment = replace(
                        assignment,
                        site=replace(
                            assignment.site,
                            value=StaticValue.unknown(),
                        ),
                        value_fragments=(),
                    )
                assignments[assignment_index] = assignment
                prefix_bindings.append(
                    _BindingIR(
                        assignment.site.name,
                        assignment.site.value,
                        _ExportState.EXPORTED,
                        assignment.value_fragments,
                    )
                )
            arguments: list[_ArgumentIR] = []
            for argument in command.arguments:
                value, value_fragments = self._resolve_value(
                    argument.value,
                    argument.fragments,
                    argument.atoms,
                    frame,
                    assignment_value=argument.local_start_byte in declaration_starts,
                    span=argument.span,
                )
                arguments.append(
                    replace(
                        argument,
                        value=value,
                        fragments=value_fragments,
                        atoms=(),
                    )
                )
            argv = tuple(argument.value for argument in arguments)
            redirects = tuple(
                self._resolve_redirect_fact(redirect, frame) for redirect in command.redirects
            )
            resolution = _CommandResolution(_CommandResolutionKind.AMBIGUOUS)
            if argv[0].state is StaticValueState.EXACT:
                command_name = cast(bytes, argv[0].exact_bytes)
                if draft.function_id is not None:
                    own_function = functions_by_id.get(draft.function_id)
                    own_name = (
                        cast(bytes, own_function.name.exact_bytes)
                        if own_function is not None
                        and own_function.name.state is StaticValueState.EXACT
                        else None
                    )
                    if own_name == command_name:
                        resolution = _CommandResolution(
                            _CommandResolutionKind.FUNCTION,
                            draft.function_id,
                        )
                    elif function_resolution_may_change(draft.function_id, command_name):
                        resolution = _CommandResolution(_CommandResolutionKind.AMBIGUOUS)
                    else:
                        found, function_id = self._lookup_function(frame, command_name)
                        resolution = (
                            _CommandResolution(_CommandResolutionKind.EXTERNAL)
                            if not found
                            else _CommandResolution(
                                _CommandResolutionKind.FUNCTION
                                if function_id is not None
                                else _CommandResolutionKind.AMBIGUOUS,
                                function_id,
                            )
                        )
                else:
                    found, function_id = self._lookup_function(frame, command_name)
                    resolution = (
                        _CommandResolution(_CommandResolutionKind.EXTERNAL)
                        if not found
                        else _CommandResolution(
                            _CommandResolutionKind.FUNCTION
                            if function_id is not None
                            else _CommandResolutionKind.AMBIGUOUS,
                            function_id,
                        )
                    )
            if not self._reserve(1 + len(prefix_bindings), span=command.site.span):
                break
            command = replace(
                command,
                site=replace(
                    command.site,
                    argv=argv,
                    argument_spans=tuple(argument.span for argument in arguments),
                    resolution=CommandResolutionKind(resolution.kind.value),
                    prefix_assignments=tuple(
                        assignments[index].site for index in prefix_assignment_indices
                    ),
                ),
                arguments=tuple(arguments),
                redirects=redirects,
                prefix_assignments=tuple(
                    assignments[index].site for index in prefix_assignment_indices
                ),
                prefix_bindings=tuple(prefix_bindings),
                program_id=self.unit.unit_id,
                execution_order=command_execution_order,
                state_update_order=len(updates),
                state_frame_id=frame.frame_id,
                resolution=resolution,
            )
            command_execution_order += 1
            commands[command_index] = command
            if (
                command.resolution.kind is _CommandResolutionKind.EXTERNAL
                and argv[0].state is StaticValueState.EXACT
                and cast(bytes, argv[0].exact_bytes) == b"tee"
                and self._has_pipeline_input(command)
            ):
                self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, command.site.span)
            generated_configs.extend(self._generated_from_command(command))
            modeled_assignment_indices.update(prefix_assignment_indices)

            declaration_keyword = (
                cast(bytes, argv[0].exact_bytes)
                if argv[0].state is StaticValueState.EXACT
                and cast(bytes, argv[0].exact_bytes) in _DECLARATION_KEYWORD_BYTES
                else None
            )
            declaration_operands = [
                argument
                for argument in arguments[1:]
                if argument.local_start_byte not in declaration_starts
            ]
            declaration_options = [
                argument
                for argument in declaration_operands
                if argument.value.state is not StaticValueState.EXACT
                or cast(bytes, argument.value.exact_bytes).startswith((b"-", b"+"))
            ]
            post_assignment_export_n = (
                declaration_keyword == b"export"
                and bool(declaration_starts)
                and len(declaration_options) == 1
                and declaration_options[0].value == StaticValue.exact(b"-n")
                and declaration_options[0].local_start_byte > min(declaration_starts)
            )
            unsupported_export_assignment = (
                declaration_keyword == b"export"
                and bool(declaration_indices)
                and bool(declaration_options)
                and not post_assignment_export_n
            )
            unsupported_declaration_mode = declaration_keyword in _DECLARATION_KEYWORD_BYTES - {
                b"export"
            } and bool(declaration_options)
            persistent_declaration_mode = declaration_keyword == b"readonly" or (
                declaration_keyword in {b"declare", b"local", b"typeset"}
                and any(
                    option.value.state is not StaticValueState.EXACT
                    or any(
                        byte in b"aAgilnrtux" for byte in cast(bytes, option.value.exact_bytes)[1:]
                    )
                    for option in declaration_options
                )
            )
            nameref_declaration_mode = self._nameref_declaration_mode(
                argv,
                arguments,
                declaration_starts,
            )
            unsupported_declaration_semantics = (
                unsupported_export_assignment
                or unsupported_declaration_mode
                or persistent_declaration_mode
            )
            if unsupported_declaration_semantics:
                self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, command.site.span)
            resolved_declarations: list[tuple[int, _AssignmentIR]] = []
            for assignment_index in declaration_indices:
                assignment = self._resolve_assignment(assignments[assignment_index], frame)
                if unsupported_declaration_semantics:
                    assignment = replace(
                        assignment,
                        site=replace(assignment.site, value=StaticValue.unknown()),
                        value_fragments=(),
                    )
                assignments[assignment_index] = assignment
                modeled_assignment_indices.add(assignment_index)
                resolved_declarations.append((assignment_index, assignment))
            exported_assignments: list[AssignmentSite] = []
            for _assignment_index, assignment in resolved_declarations:
                previous = self._lookup_binding(frame, assignment.site.name)
                export_state = (
                    _ExportState.EXPORTED
                    if assignment.declaration_keyword == b"export"
                    and not unsupported_declaration_semantics
                    else previous.export_state
                )
                if unsupported_declaration_semantics:
                    export_state = _ExportState.UNKNOWN
                self._apply_binding(
                    frame=frame,
                    binding=_BindingIR(
                        assignment.site.name,
                        assignment.site.value,
                        export_state,
                        assignment.value_fragments,
                    ),
                    role=role,
                    draft=draft,
                    updates=updates,
                )
                if (
                    assignment.declaration_keyword == b"export"
                    and not unsupported_declaration_semantics
                ):
                    exported_assignments.append(assignment.site)

            if persistent_declaration_mode:
                persistent_names = {
                    assignment.site.name for _index, assignment in resolved_declarations
                }
                for operand in declaration_operands:
                    if operand in declaration_options:
                        continue
                    if operand.value.state is not StaticValueState.EXACT:
                        persistent_names.update(_visible_bindings(frame))
                        continue
                    if persistent_name := self._identifier(cast(bytes, operand.value.exact_bytes)):
                        persistent_names.add(persistent_name)
                if nameref_declaration_mode:
                    persistent_names.update(_visible_bindings(frame))
                frame.persistent_unknown_names.update(persistent_names)
                frame.persistent_bindings_unknown = (
                    frame.persistent_bindings_unknown or nameref_declaration_mode
                )
                for persistent_name in sorted(persistent_names):
                    self._apply_binding(
                        frame=frame,
                        binding=_BindingIR(
                            persistent_name,
                            StaticValue.unknown(),
                            _ExportState.UNKNOWN,
                            (),
                        ),
                        role=role,
                        draft=draft,
                        updates=updates,
                    )

            if argv[0] == StaticValue.exact(b"export"):
                export_operand_arguments = [
                    argument
                    for argument in arguments[1:]
                    if argument.local_start_byte not in declaration_starts
                ]
                export_operands = [argument.value for argument in export_operand_arguments]
                post_assignment_no_export = (
                    bool(declaration_starts)
                    and len(export_operands) == 2
                    and export_operands[0] == StaticValue.exact(b"-n")
                    and export_operand_arguments[0].local_start_byte > min(declaration_starts)
                )
                if (
                    len(export_operands) == 1
                    and export_operands[0].state is StaticValueState.EXACT
                    and (
                        export_name := self._identifier(cast(bytes, export_operands[0].exact_bytes))
                    )
                    is not None
                ):
                    previous = self._lookup_binding(frame, export_name)
                    self._apply_binding(
                        frame=frame,
                        binding=replace(previous, export_state=_ExportState.EXPORTED),
                        role=role,
                        draft=draft,
                        updates=updates,
                    )
                    exported_assignments.append(
                        AssignmentSite(
                            unit_id=command.site.unit_id,
                            provenance=command.site.provenance,
                            span=export_operand_arguments[0].span,
                            name=export_name,
                            value=previous.value,
                        )
                    )
                elif (
                    post_assignment_no_export
                    and export_operands[1].state is StaticValueState.EXACT
                    and (
                        export_name := self._identifier(cast(bytes, export_operands[1].exact_bytes))
                    )
                    is not None
                ):
                    self._issue(
                        ShellIssueReason.UNSUPPORTED_SEMANTICS,
                        command.site.span,
                    )
                    previous = self._lookup_binding(frame, export_name)
                    self._apply_binding(
                        frame=frame,
                        binding=replace(
                            previous,
                            export_state=_ExportState.UNKNOWN,
                        ),
                        role=role,
                        draft=draft,
                        updates=updates,
                    )
                elif (
                    len(export_operands) == 2
                    and export_operands[0] == StaticValue.exact(b"-n")
                    and export_operands[1].state is StaticValueState.EXACT
                    and (
                        export_name := self._identifier(cast(bytes, export_operands[1].exact_bytes))
                    )
                    is not None
                ):
                    previous = self._lookup_binding(frame, export_name)
                    self._apply_binding(
                        frame=frame,
                        binding=replace(previous, export_state=_ExportState.UNEXPORTED),
                        role=role,
                        draft=draft,
                        updates=updates,
                    )
                elif export_operands:
                    self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, command.site.span)
                    for argv_value in export_operands:
                        if argv_value.state is not StaticValueState.EXACT:
                            continue
                        export_name = self._identifier(cast(bytes, argv_value.exact_bytes))
                        if export_name is None:
                            continue
                        previous = self._lookup_binding(frame, export_name)
                        self._apply_binding(
                            frame=frame,
                            binding=replace(
                                previous,
                                export_state=_ExportState.UNKNOWN,
                            ),
                            role=role,
                            draft=draft,
                            updates=updates,
                        )

            if exported_assignments:
                command = replace(
                    command,
                    site=replace(
                        command.site,
                        exported_assignments=tuple(exported_assignments),
                    ),
                )
                commands[command_index] = command

            effect_function_ids: tuple[int, ...] = ()
            if (
                command.resolution.kind is _CommandResolutionKind.FUNCTION
                and command.resolution.function_id is not None
            ):
                effect_function_ids = (command.resolution.function_id,)
            elif (
                command.resolution.kind is _CommandResolutionKind.AMBIGUOUS
                and argv[0].state is StaticValueState.EXACT
            ):
                effect_function_ids = possible_function_ids(
                    draft,
                    cast(bytes, argv[0].exact_bytes),
                    include_future=draft.function_id is not None,
                )
            if effect_function_ids:
                if command.resolution.kind is _CommandResolutionKind.FUNCTION and any(
                    function_id not in functions_by_id for function_id in effect_function_ids
                ):
                    self._issue(
                        ShellIssueReason.UNSUPPORTED_SEMANTICS,
                        command.site.span,
                    )
                effect_binding_names: set[str] = set()
                effect_export_names: set[str] = set()
                effect_function_names: set[bytes] = set()
                effect_bindings_unknown = False
                effect_exports_unknown = False
                effect_functions_unknown = False
                for function_id in effect_function_ids:
                    (
                        nested_binding_names,
                        nested_export_names,
                        nested_function_names,
                        nested_bindings_unknown,
                        nested_exports_unknown,
                        nested_functions_unknown,
                    ) = function_effects(function_id)
                    effect_binding_names.update(nested_binding_names)
                    effect_export_names.update(nested_export_names)
                    effect_function_names.update(nested_function_names)
                    effect_bindings_unknown = effect_bindings_unknown or nested_bindings_unknown
                    effect_exports_unknown = effect_exports_unknown or nested_exports_unknown
                    effect_functions_unknown = effect_functions_unknown or nested_functions_unknown
                value_names = set(effect_binding_names)
                export_names = set(effect_export_names)
                if effect_bindings_unknown:
                    value_names.update(_visible_bindings(frame))
                if effect_exports_unknown:
                    export_names.update(_visible_bindings(frame))
                for binding_name in sorted(value_names | export_names):
                    previous = self._lookup_binding(frame, binding_name)
                    self._apply_binding(
                        frame=frame,
                        binding=_BindingIR(
                            binding_name,
                            (
                                StaticValue.unknown()
                                if binding_name in value_names
                                else previous.value
                            ),
                            (
                                _ExportState.UNKNOWN
                                if binding_name in export_names
                                else previous.export_state
                            ),
                            () if binding_name in value_names else previous.fragments,
                        ),
                        role=role,
                        draft=draft,
                        updates=updates,
                    )
                for function_name in sorted(effect_function_names):
                    frame.functions[function_name] = None
                if effect_functions_unknown:
                    for function_name in _visible_functions(frame):
                        frame.functions[function_name] = None
                    frame.functions_unknown = True

            nested = self._nested_request(command, frame, depth=depth)
            if nested is not None:
                response = yield nested
                if nested.execution_kind is _NestedExecutionKind.EVAL:
                    if response.bindings is not None:
                        prefix_binding_names = {binding.name for binding in command.prefix_bindings}
                        for binding_name, binding in response.bindings.items():
                            if binding_name in prefix_binding_names:
                                continue
                            if self._lookup_binding(frame, binding_name) == binding:
                                continue
                            self._apply_binding(
                                frame=frame,
                                binding=binding,
                                role=role,
                                draft=draft,
                                updates=updates,
                            )
                    if response.functions is not None:
                        frame.functions.update(response.functions)
                    frame.functions_unknown = frame.functions_unknown or response.functions_unknown
            elif command.resolution.kind in {
                _CommandResolutionKind.EXTERNAL,
                _CommandResolutionKind.AMBIGUOUS,
            } and command.site.argv[0] == StaticValue.exact(b"eval"):
                if command.resolution.kind is _CommandResolutionKind.AMBIGUOUS:
                    self._issue(
                        ShellIssueReason.UNSUPPORTED_SEMANTICS,
                        command.site.span,
                    )
                for binding_name in tuple(_visible_bindings(frame)):
                    self._apply_binding(
                        frame=frame,
                        binding=_BindingIR(
                            binding_name,
                            StaticValue.unknown(),
                            _ExportState.UNKNOWN,
                            (),
                        ),
                        role=role,
                        draft=draft,
                        updates=updates,
                    )
                for function_name in tuple(_visible_functions(frame)):
                    frame.functions[function_name] = None
                frame.functions_unknown = True

        final_draft = self.event_drafts[-1] if self.event_drafts else None
        if final_draft is not None:
            for scope_id in reversed(active_control_scopes):
                self._merge_control_frame(
                    frame=control_frames[scope_id],
                    parent=control_parents[scope_id],
                    scope_id=scope_id,
                    draft=final_draft,
                    updates=updates,
                )

        for recovered in program.recovered_generated_drafts:
            input_fact = self._resolve_redirect_fact(recovered.input_fact, root)
            output_fact = self._resolve_redirect_fact(recovered.output_fact, root)
            generated = self._generated_config(
                input_fact,
                output_fact,
                producer_command_start_byte=None,
                producer_function_id=recovered.function_id,
            )
            if generated is not None:
                generated_configs.append(generated)

        generated_configs = list(
            {
                (
                    config.target,
                    config.content,
                    config.span.path,
                    config.span.start_byte,
                    config.span.end_byte,
                ): config
                for config in generated_configs
            }.values()
        )

        assignments = [
            assignment
            for assignment_index, assignment in enumerate(assignments)
            if assignment_index in modeled_assignment_indices
        ]
        assignments.sort(key=lambda item: item.site.span.start_byte)
        assignments = [replace(item, order=index) for index, item in enumerate(assignments)]
        commands = [command for command in commands if command.program_id == self.unit.unit_id]
        commands.sort(key=lambda item: item.site.span.start_byte)
        commands = [replace(item, order=index) for index, item in enumerate(commands)]
        state_frames_by_id = {
            frame.frame_id: frame for frame in (root, *frames.values(), *control_frames.values())
        }
        state_frames: tuple[_StateFrameFact, ...] = ()
        retained_initial_bindings: tuple[_BindingIR, ...] = ()
        retained_initial_functions: tuple[tuple[bytes, int | None], ...] = ()
        retained_initial_functions_unknown = False
        state_span = program.regions[0].span if program.regions else self.unit.origin_span
        state_snapshot_retained = self._reserve(
            len(state_frames_by_id) + len(initial_bindings or {}) + len(initial_functions or {}),
            span=state_span,
        )
        if state_snapshot_retained:
            state_frames = tuple(
                _StateFrameFact(
                    frame.frame_id,
                    frame.parent.frame_id if frame.parent is not None else None,
                    frame.functions_unknown,
                )
                for frame in sorted(state_frames_by_id.values(), key=lambda item: item.frame_id)
            )
            retained_initial_bindings = tuple(
                binding for _, binding in sorted((initial_bindings or {}).items())
            )
            retained_initial_functions = tuple(sorted((initial_functions or {}).items()))
            retained_initial_functions_unknown = initial_functions_unknown
        else:
            commands = []
            assignments = []
            updates = []
            events = []
        return _ModeledState(
            replace(
                program,
                program_id=self.unit.unit_id,
                commands=tuple(commands),
                assignments=tuple(assignments),
                state_updates=tuple(updates),
                state_events=tuple(events),
                state_frames=state_frames,
                initial_bindings=retained_initial_bindings,
                initial_functions=retained_initial_functions,
                initial_functions_unknown=retained_initial_functions_unknown,
                generated_configs=tuple(generated_configs),
            ),
            dict(root.bindings),
            dict(root.functions),
            root.functions_unknown,
        )

    def lower(self) -> _ShellProgramIR:
        if self.halted:
            return _ShellProgramIR(
                program_id=self.unit.unit_id,
                regions=tuple(self.regions),
            )
        sorted_command_drafts = sorted(self.command_drafts, key=lambda item: item.node.start_byte)
        mutable_commands_by_depth: dict[int, list[_CommandDraft]] = {}
        for command in sorted_command_drafts:
            mutable_commands_by_depth.setdefault(command.substitution_depth, []).append(command)
        commands_by_depth = {
            depth: tuple(commands) for depth, commands in mutable_commands_by_depth.items()
        }
        starts_by_depth = {
            depth: tuple(command.node.start_byte for command in commands)
            for depth, commands in commands_by_depth.items()
        }
        redirects_by_command: dict[int, list[_RedirectDraft]] = {}
        for redirect in self.redirect_drafts:
            command_start = self._redirect_command_start(
                redirect,
                commands_by_depth,
                starts_by_depth,
            )
            if command_start is None:
                owner = (
                    self.redirect_owners.get(redirect.statement_start_byte)
                    if redirect.statement_start_byte is not None
                    else None
                )
                if owner is not None and owner.body_type in _COMPOUND_REDIRECT_BODIES:
                    self._issue(
                        ShellIssueReason.UNSUPPORTED_SEMANTICS,
                        self._node_span(redirect.node)
                        or self._point_span(redirect.node.start_byte),
                    )
                continue
            redirects_by_command.setdefault(command_start, []).append(redirect)

        heredocs_by_command: dict[int, list[_RedirectFact]] = {}
        for statement_start, statement_facts in self.heredoc_facts_by_statement.items():
            owner = self.redirect_owners.get(statement_start)
            if owner is None:
                continue
            command_start = self._redirect_owner_command_start(
                owner,
                commands_by_depth,
                starts_by_depth,
            )
            if command_start is not None:
                heredocs_by_command.setdefault(command_start, []).extend(statement_facts)

        wrapper_arguments_by_command: dict[int, list[_NodeGroup]] = {}
        for wrapper in self.wrapper_argument_drafts:
            owner = self.redirect_owners.get(wrapper.statement_start_byte)
            if owner is None:
                continue
            command_start = self._redirect_owner_command_start(
                owner,
                commands_by_depth,
                starts_by_depth,
            )
            if command_start is None:
                self._issue(
                    ShellIssueReason.UNSUPPORTED_SEMANTICS,
                    self._point_span(owner.body_start_byte),
                )
                continue
            wrapper_arguments_by_command.setdefault(command_start, []).extend(
                self._group_nodes(list(wrapper.nodes))
            )

        assignments: list[_AssignmentIR] = []
        assignment_keys: set[tuple[int, int]] = set()
        prefix_sites_by_command: dict[int, list[AssignmentSite]] = {}
        grouped_declaration_starts = {
            draft.node.start_byte
            for draft in sorted_command_drafts
            if draft.node.type == "declaration_command"
            and draft.node.children
            and self.raw[draft.node.children[0].start_byte : draft.node.children[0].end_byte]
            in _DECLARATION_KEYWORD_BYTES
        }
        for order, draft in enumerate(
            sorted(self.assignment_drafts, key=lambda item: item.node.start_byte)
        ):
            if (
                draft.prefix_for_command_start_byte is not None
                or draft.declaration_command_start_byte in grouped_declaration_starts
            ):
                continue
            assignment = self._assignment_from_node(draft, order=order)
            if assignment is not None:
                assignments.append(assignment)
                assignment_keys.add((draft.node.start_byte, draft.node.end_byte))
                if assignment.prefix_for_command_start_byte is not None:
                    prefix_sites_by_command.setdefault(
                        assignment.prefix_for_command_start_byte, []
                    ).append(assignment.site)
            if self.halted:
                break

        commands: list[_CommandIR] = []
        for command_draft in sorted_command_drafts:
            if self.halted:
                break
            groups = self._command_groups(command_draft.node)
            if not groups:
                continue
            prefix_groups: list[_NodeGroup] = []
            if command_draft.node.type == "command":
                while groups and self._group_assignment_name(groups[0]) is not None:
                    prefix_groups.append(groups.pop(0))
            persistent_prefix_count = 0
            if prefix_groups and groups:
                boundary_groups = (*prefix_groups[1:], groups[0])
                for index, (prefix_group, next_group) in enumerate(
                    zip(prefix_groups, boundary_groups, strict=True)
                ):
                    prefix_end_node = prefix_group.nodes[-1]
                    next_start_node = next_group.nodes[0]
                    gap = self.raw[prefix_end_node.end_byte : next_start_node.start_byte]
                    if (
                        prefix_end_node.end_point.row < next_start_node.start_point.row
                        and not self._is_line_continuation_gap(gap)
                    ):
                        persistent_prefix_count = index + 1
            for index, group in enumerate(prefix_groups):
                persistent_assignment = index < persistent_prefix_count
                assignment = self._assignment_from_group(
                    group,
                    order=len(assignments),
                    region_id=command_draft.region_id,
                    function_id=command_draft.function_id,
                    prefix_for_command_start_byte=(
                        None if persistent_assignment else command_draft.node.start_byte
                    ),
                )
                if assignment is not None:
                    assignments.append(assignment)
                    if not persistent_assignment:
                        prefix_sites_by_command.setdefault(
                            command_draft.node.start_byte, []
                        ).append(assignment.site)
                if self.halted:
                    break
            if self.halted or not groups:
                continue

            timed = (
                command_draft.node.type == "command"
                and groups[0].raw_syntax == b"time"
                and len(groups[0].nodes) == 1
            )
            if timed:
                groups.pop(0)
                if groups and groups[0].raw_syntax == b"-p" and len(groups[0].nodes) == 1:
                    groups.pop(0)
                if not groups:
                    continue

            name_group = groups.pop(0)
            name_argument = self._argument(name_group)
            if name_argument is None:
                continue
            arguments: list[_ArgumentIR] = [name_argument]
            for group in groups:
                argument = self._argument(group)
                if argument is not None:
                    arguments.append(argument)
                if self.halted:
                    break
            if self.halted:
                break

            for group in wrapper_arguments_by_command.get(command_draft.node.start_byte, ()):
                argument = self._argument(group)
                if argument is not None:
                    arguments.append(argument)
                if self.halted:
                    break
            if self.halted:
                break

            facts: list[_RedirectFact] = list(
                heredocs_by_command.get(command_draft.node.start_byte, ())
            )
            redirects_complete = True
            command_prefix_start = (
                prefix_groups[persistent_prefix_count].start_byte
                if persistent_prefix_count < len(prefix_groups)
                else name_group.start_byte
            )
            span_start = name_group.start_byte if timed else command_prefix_start
            span_end = max(argument.local_end_byte for argument in arguments)
            command_redirects = sorted(
                redirects_by_command.get(command_draft.node.start_byte, ()),
                key=lambda item: item.node.start_byte,
            )
            for redirect in command_redirects:
                owner = (
                    self.redirect_owners.get(redirect.statement_start_byte)
                    if redirect.statement_start_byte is not None
                    else None
                )
                allow_fact = not (
                    command_draft.node.start_byte in self.syntax_error_command_starts
                    or (
                        owner is not None
                        and (
                            owner.has_error_child
                            or owner.statement_start_byte in self.syntax_error_redirect_starts
                        )
                    )
                )
                fact, extras, supported = self._redirect_parts(
                    redirect,
                    allow_fact=allow_fact,
                )
                redirects_complete = redirects_complete and supported and allow_fact
                if fact is not None:
                    facts.append(fact)
                arguments.extend(extras)
                span_start = min(span_start, redirect.node.start_byte)
                span_end = max(span_end, redirect.node.end_byte)
                if self.halted:
                    break
            if self.halted:
                break
            command_span = self._span(span_start, span_end)
            if command_span is None:
                self._issue(
                    ShellIssueReason.UNSUPPORTED_SEMANTICS,
                    self._point_span(span_start),
                )
                continue
            ordered_arguments = [
                arguments[0],
                *sorted(arguments[1:], key=lambda item: item.local_start_byte),
            ]
            if not self._reserve(2, span=command_span):
                break
            site = CommandSite(
                unit_id=self.unit.unit_id,
                provenance=self.unit.provenance,
                span=command_span,
                argv=tuple(argument.value for argument in ordered_arguments),
                argument_spans=tuple(argument.span for argument in ordered_arguments),
            )
            prefix_sites = tuple(prefix_sites_by_command.get(command_draft.node.start_byte, ()))
            commands.append(
                _CommandIR(
                    site,
                    len(commands),
                    command_draft.region_id,
                    command_draft.function_id,
                    tuple(ordered_arguments),
                    prefix_sites,
                    tuple(sorted(facts, key=lambda item: item.local_start_byte)),
                    redirects_complete,
                    local_start_byte=command_draft.node.start_byte,
                    local_end_byte=command_draft.node.end_byte,
                )
            )

            if (
                site.argv[0].state is StaticValueState.EXACT
                and site.argv[0].exact_bytes in _DECLARATION_KEYWORD_BYTES
            ):
                for group in groups:
                    key = (group.start_byte, group.end_byte)
                    if key in assignment_keys:
                        continue
                    assignment = self._assignment_from_group(
                        group,
                        order=len(assignments),
                        region_id=command_draft.region_id,
                        function_id=command_draft.function_id,
                        prefix_for_command_start_byte=None,
                        declaration_keyword=cast(bytes, site.argv[0].exact_bytes),
                        declaration_command_start_byte=command_draft.node.start_byte,
                    )
                    if assignment is not None:
                        assignments.append(assignment)
                        assignment_keys.add(key)
                    if self.halted:
                        break

        assignments.sort(key=lambda item: item.site.span.start_byte)
        assignments = [replace(item, order=index) for index, item in enumerate(assignments)]
        commands.sort(key=lambda item: item.site.span.start_byte)
        commands = [replace(item, order=index) for index, item in enumerate(commands)]
        program = _ShellProgramIR(
            program_id=self.unit.unit_id,
            regions=tuple(self.regions),
            functions=self._functions() if not self.halted else (),
            commands=tuple(commands),
            assignments=tuple(assignments),
            recovered_generated_drafts=tuple(self.recovered_generated_drafts),
        )
        return program


def _visible_bindings(frame: _StateFrame) -> dict[str, _BindingIR]:
    chain: list[_StateFrame] = []
    current: _StateFrame | None = frame
    while current is not None:
        chain.append(current)
        current = current.parent
    visible: dict[str, _BindingIR] = {}
    for current in reversed(chain):
        visible.update(current.bindings)
    return visible


def _visible_functions(frame: _StateFrame) -> dict[bytes, int | None]:
    chain: list[_StateFrame] = []
    current: _StateFrame | None = frame
    while current is not None:
        chain.append(current)
        current = current.parent
    visible: dict[bytes, int | None] = {}
    for current in reversed(chain):
        visible.update(current.functions)
    return visible


def _visible_functions_unknown(frame: _StateFrame) -> bool:
    current: _StateFrame | None = frame
    while current is not None:
        if current.functions_unknown:
            return True
        current = current.parent
    return False


def _unknown_nested_response(request: _NestedRequest) -> _NestedResponse:
    if request.execution_kind is not _NestedExecutionKind.EVAL:
        return _NestedResponse()
    bindings = {
        name: _BindingIR(name, StaticValue.unknown(), _ExportState.UNKNOWN, ())
        for name in _visible_bindings(request.frame)
    }
    functions = dict.fromkeys(_visible_functions(request.frame))
    return _NestedResponse(bindings, functions, True)


def _identity_source_map(unit: ShellUnit) -> SourceMap | None:
    if unit.source_map is not None:
        return unit.source_map
    if unit.origin_span.start_byte != 0:
        return None
    entries = (
        (
            SourceMapEntry(
                0,
                len(unit.raw_bytes),
                unit.origin_span.start_byte,
                unit.origin_span.end_byte,
            ),
        )
        if unit.raw_bytes
        else ()
    )
    return SourceMap(
        path=unit.origin_span.path,
        entries=entries,
        child_size_bytes=len(unit.raw_bytes),
        physical_size_bytes=unit.origin_span.end_byte,
        physical_line_starts=_line_starts(_physical_lines(unit.raw_bytes)),
    )


def _nested_source_map(
    parent: _ShellLowerer,
    payload: _ArgumentIR,
) -> SourceMap | None:
    if payload.value.state is not StaticValueState.EXACT:
        return None
    payload_bytes = cast(bytes, payload.value.exact_bytes)
    parent_map = _identity_source_map(parent.unit)
    if parent_map is None:
        return None
    parent_local = all(fragment.unit_id == parent.unit.unit_id for fragment in payload.fragments)
    entries: list[SourceMapEntry] = []
    logical_cursor = 0
    physical_cursor = 0
    for fragment in payload.fragments:
        physical_start = fragment.local_start_byte if parent_local else fragment.span.start_byte
        physical_end = fragment.local_end_byte if parent_local else fragment.span.end_byte
        if (
            not fragment.exact
            or fragment.value_start_byte != logical_cursor
            or fragment.value_end_byte <= fragment.value_start_byte
            or physical_end <= physical_start
            or fragment.value_end_byte - fragment.value_start_byte != physical_end - physical_start
            or physical_start < physical_cursor
        ):
            return None
        candidate = SourceMapEntry(
            fragment.value_start_byte,
            fragment.value_end_byte,
            physical_start,
            physical_end,
        )
        if (
            entries
            and entries[-1].child_end_byte == candidate.child_start_byte
            and entries[-1].physical_end_byte == candidate.physical_start_byte
        ):
            previous = entries[-1]
            entries[-1] = SourceMapEntry(
                previous.child_start_byte,
                candidate.child_end_byte,
                previous.physical_start_byte,
                candidate.physical_end_byte,
            )
        else:
            entries.append(candidate)
        logical_cursor = fragment.value_end_byte
        physical_cursor = physical_end
    if logical_cursor != len(payload_bytes):
        return None
    if not payload_bytes:
        entries = []
    intermediate_map = SourceMap(
        path=parent.unit.origin_span.path,
        entries=tuple(entries),
        child_size_bytes=len(payload_bytes),
        physical_size_bytes=(
            len(parent.unit.raw_bytes) if parent_local else parent_map.physical_size_bytes
        ),
        physical_line_starts=(
            _line_starts(_physical_lines(parent.unit.raw_bytes))
            if parent_local
            else parent_map.physical_line_starts
        ),
    )
    if not parent_local:
        return intermediate_map
    try:
        return intermediate_map.compose(parent_map)
    except ValueError:
        return None


def _nested_unit(
    parent: _ShellLowerer,
    request: _NestedRequest,
    source_map: SourceMap,
) -> ShellUnit:
    payload = cast(bytes, request.payload.value.exact_bytes)
    physical_starts = source_map.physical_line_starts
    mapped_starts = [entry.physical_start_byte for entry in source_map.entries]
    mapped_ends = [entry.physical_end_byte for entry in source_map.entries]
    origin_start = min([request.command.site.span.start_byte, *mapped_starts])
    origin_end = max([request.command.site.span.end_byte, *mapped_ends])
    origin = _span_for_bytes(
        parent.unit.origin_span.path,
        physical_starts,
        origin_start,
        origin_end,
    )
    return ShellUnit(
        dialect=(
            ShellDialect.BASH
            if request.execution_kind is _NestedExecutionKind.EVAL
            else {
                b"bash": ShellDialect.BASH,
                b"sh": ShellDialect.SH,
                b"dash": ShellDialect.DASH,
            }[cast(bytes, request.command.site.argv[0].exact_bytes)]
        ),
        kind=ShellUnitKind.NESTED_LITERAL,
        provenance=SiteProvenance.NESTED_LITERAL,
        raw_bytes=payload,
        origin_span=origin,
        source_map=source_map,
    )


def _initial_nested_state(
    request: _NestedRequest,
) -> tuple[dict[str, _BindingIR], dict[bytes, int | None], bool]:
    visible = _visible_bindings(request.frame)
    if request.execution_kind is _NestedExecutionKind.EVAL:
        visible.update({binding.name: binding for binding in request.command.prefix_bindings})
        return (
            visible,
            _visible_functions(request.frame),
            _visible_functions_unknown(request.frame),
        )
    exported = {
        name: binding
        for name, binding in visible.items()
        if binding.export_state is _ExportState.EXPORTED
    }
    exported.update({binding.name: binding for binding in request.command.prefix_bindings})
    return exported, {}, False


def _retain_nested_issue(
    lowerer: _ShellLowerer,
    reason: ShellIssueReason,
    span: SourceSpan,
    *,
    exhaustion: DependencyWorkExhaustion | None = None,
) -> None:
    lowerer._issue(reason, span, exhaustion=exhaustion)


def _retain_typed_function_generated_configs(
    root_lowerer: _ShellLowerer,
    root_program: _ShellProgramIR,
    nested_programs: tuple[_NestedProgramIR, ...],
    commands: tuple[_CommandIR, ...],
    configs: list[GeneratedConfig],
) -> list[GeneratedConfig]:
    """Publish function-body configs only through exact typed call reachability."""

    type FunctionKey = tuple[str, int]
    commands_by_key: dict[tuple[str, int], list[_CommandIR]] = {}
    for command in commands:
        commands_by_key.setdefault((command.program_id, command.site.span.start_byte), []).append(
            command
        )

    programs_by_id: dict[str, list[_ShellProgramIR]] = {}
    programs_by_id.setdefault(root_program.program_id, []).append(root_program)
    for nested in nested_programs:
        programs_by_id.setdefault(nested.program.program_id, []).append(nested.program)

    functions_by_key: dict[FunctionKey, list[_FunctionContext]] = {}
    function_names: dict[FunctionKey, bytes] = {}
    for program_id, programs in programs_by_id.items():
        for program in programs:
            for function in program.functions:
                key = (program_id, function.function_id)
                functions_by_key.setdefault(key, []).append(function)
                if function.name.state is not StaticValueState.EXACT:
                    continue
                name = cast(bytes, function.name.exact_bytes)
                previous = function_names.get(key)
                if previous is not None and previous != name:
                    function_names.pop(key, None)
                    continue
                function_names[key] = name
    functions_by_name: dict[bytes, set[FunctionKey]] = {}
    functions_by_name_and_id: dict[tuple[bytes, int], set[FunctionKey]] = {}
    for key, name in function_names.items():
        functions_by_name.setdefault(name, set()).add(key)
        functions_by_name_and_id.setdefault((name, key[1]), set()).add(key)

    nested_by_program_id: dict[str, list[_NestedProgramIR]] = {}
    for nested in nested_programs:
        nested_by_program_id.setdefault(nested.program.program_id, []).append(nested)

    unique_programs = {
        program_id: programs[0]
        for program_id, programs in programs_by_id.items()
        if len(programs) == 1
    }
    initial_functions_by_program = {
        program_id: dict(program.initial_functions)
        for program_id, program in unique_programs.items()
    }
    regions_by_program = {
        program_id: {region.region_id: region for region in program.regions}
        for program_id, program in unique_programs.items()
    }

    def command_owner(command: _CommandIR) -> FunctionKey | None:
        if command.function_id is not None:
            return (command.program_id, command.function_id)
        if (
            command.containing_function_program_id is not None
            and command.containing_function_id is not None
        ):
            return (
                command.containing_function_program_id,
                command.containing_function_id,
            )
        return None

    def parent_program_command(program_id: str) -> tuple[str, _CommandIR] | None:
        records = nested_by_program_id.get(program_id, ())
        if len(records) != 1:
            return None
        record = records[0]
        parents = commands_by_key.get(
            (record.parent_program_id, record.parent_command_start_byte), ()
        )
        if len(parents) != 1:
            return None
        return record.parent_program_id, parents[0]

    def resolve_function_call(command: _CommandIR) -> FunctionKey | None:
        if (
            command.resolution.kind is not _CommandResolutionKind.FUNCTION
            or type(command.resolution.function_id) is not int
            or not command.site.argv
            or command.site.argv[0].state is not StaticValueState.EXACT
        ):
            return None
        name = cast(bytes, command.site.argv[0].exact_bytes)
        resolved_id = cast(int, command.resolution.function_id)
        seen_programs: set[str] = set()
        program_id: str | None = command.program_id
        reference = command
        while program_id is not None and program_id not in seen_programs:
            seen_programs.add(program_id)
            key = (program_id, resolved_id)
            functions = functions_by_key.get(key, ())
            function = functions[0] if len(functions) == 1 else None
            initial_id = initial_functions_by_program.get(program_id, {}).get(name)
            definition_region = (
                regions_by_program.get(program_id, {}).get(function.definition_region_id)
                if function is not None and function.definition_region_id is not None
                else None
            )
            inherited_id_collision_is_ambiguous = (
                initial_id == resolved_id
                and reference.function_id is None
                and definition_region is not None
                and definition_region.kind is not _ExecutionRegionKind.PROGRAM
            )
            if inherited_id_collision_is_ambiguous:
                return None
            if (
                function is not None
                and function_names.get(key) == name
                and (
                    reference.function_id is not None
                    or function.local_start_byte < reference.local_start_byte
                )
            ):
                return key
            if initial_id != resolved_id:
                break
            parent = parent_program_command(program_id)
            if parent is None:
                break
            program_id, reference = parent
        candidates = (
            functions_by_name.get(name, set())
            if resolved_id == _IMPORTED_FUNCTION_ID
            else functions_by_name_and_id.get((name, resolved_id), set())
        )
        return next(iter(candidates)) if len(candidates) == 1 else None

    owner_by_config_id: dict[int, FunctionKey | None] = {}
    invalid_configs: set[int] = set()
    config_functions: set[FunctionKey] = set()
    for config in configs:
        if type(config) is not _ProvenGeneratedConfig:
            invalid_configs.add(id(config))
            root_lowerer._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, config.span)
            continue
        proven = cast(_ProvenGeneratedConfig, config)
        owner: FunctionKey | None
        owner_known = True
        if proven.producer_function_id is not None:
            owner = (proven.producer_program_id, proven.producer_function_id)
        elif proven.producer_command_start_byte is not None:
            producers = commands_by_key.get(
                (
                    proven.producer_program_id,
                    proven.producer_command_start_byte,
                ),
                (),
            )
            if len(producers) != 1:
                owner = None
                owner_known = False
            else:
                owner = command_owner(producers[0])
        elif proven.producer_program_id == root_program.program_id:
            owner = None
        else:
            records = nested_by_program_id.get(proven.producer_program_id, ())
            inherited_owners: set[FunctionKey | None] = set()
            for record in records:
                parents = commands_by_key.get(
                    (record.parent_program_id, record.parent_command_start_byte), ()
                )
                if len(parents) == 1:
                    inherited_owners.add(command_owner(parents[0]))
            if len(inherited_owners) != 1:
                owner = None
                owner_known = False
            else:
                owner = next(iter(inherited_owners))
        if owner is not None and (
            len(functions_by_key.get(owner, ())) != 1 or owner not in function_names
        ):
            owner_known = False
        if not owner_known:
            invalid_configs.add(id(config))
            root_lowerer._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, config.span)
            continue
        owner_by_config_id[id(config)] = owner
        if owner is not None:
            config_functions.add(owner)

    if not config_functions:
        retained_root_configs: list[GeneratedConfig] = []
        for config in configs:
            if id(config) in invalid_configs:
                continue
            if owner_by_config_id[id(config)] is not None:
                continue
            proven = cast(_ProvenGeneratedConfig, config)
            if proven.execution_limitation_span is not None:
                root_lowerer._issue(
                    ShellIssueReason.UNSUPPORTED_SEMANTICS,
                    proven.execution_limitation_span,
                )
            retained_root_configs.append(config)
        return retained_root_configs

    calls_by_owner: dict[FunctionKey, set[FunctionKey]] = {}
    root_calls: set[FunctionKey] = set()
    unresolved_calls: list[tuple[FunctionKey | None, _CommandIR, bytes | None]] = []
    for command in commands:
        if not command.site.argv:
            continue
        owner = command_owner(command)
        command_name: bytes | None = (
            cast(bytes, command.site.argv[0].exact_bytes)
            if command.site.argv[0].state is StaticValueState.EXACT
            else None
        )
        if command.resolution.kind is _CommandResolutionKind.FUNCTION:
            if command_name is None:
                unresolved_calls.append((owner, command, None))
                continue
            target = resolve_function_call(command)
            if target is None:
                unresolved_calls.append((owner, command, command_name))
            elif owner is None:
                root_calls.add(target)
            else:
                calls_by_owner.setdefault(owner, set()).add(target)
        elif command.resolution.kind is _CommandResolutionKind.AMBIGUOUS:
            unresolved_calls.append((owner, command, command_name))

    active_functions: set[FunctionKey] = set()
    pending_functions = list(root_calls)
    while pending_functions:
        key = pending_functions.pop()
        if key in active_functions:
            continue
        active_functions.add(key)
        pending_functions.extend(calls_by_owner.get(key, ()))

    reverse_calls: dict[FunctionKey, set[FunctionKey]] = {}
    for caller, targets in calls_by_owner.items():
        for target in targets:
            reverse_calls.setdefault(target, set()).add(caller)

    unresolved_owners_by_name: dict[bytes, set[FunctionKey]] = {}
    dynamic_unresolved_owners: set[FunctionKey] = set()
    for owner, _command, unresolved_name in unresolved_calls:
        if owner is None:
            continue
        if unresolved_name is None:
            dynamic_unresolved_owners.add(owner)
        else:
            unresolved_owners_by_name.setdefault(unresolved_name, set()).add(owner)

    functions_reaching_configs = set(config_functions)
    pending_functions = [*config_functions, *dynamic_unresolved_owners]
    expanded_unresolved_names: set[bytes] = set()
    while pending_functions:
        key = pending_functions.pop()
        if key not in functions_reaching_configs:
            functions_reaching_configs.add(key)
        for caller in reverse_calls.get(key, ()):
            if caller in functions_reaching_configs:
                continue
            functions_reaching_configs.add(caller)
            pending_functions.append(caller)
        reachable_name = function_names.get(key)
        if reachable_name is None or reachable_name in expanded_unresolved_names:
            continue
        expanded_unresolved_names.add(reachable_name)
        for caller in unresolved_owners_by_name.get(reachable_name, ()):
            if caller in functions_reaching_configs:
                continue
            functions_reaching_configs.add(caller)
            pending_functions.append(caller)

    reachable_names: set[bytes] = set()
    for key in functions_reaching_configs:
        candidate_name = function_names.get(key)
        if candidate_name is not None:
            reachable_names.add(candidate_name)
    for owner, command, unresolved_name in unresolved_calls:
        if owner is not None and owner not in active_functions:
            continue
        if unresolved_name is None or unresolved_name in reachable_names:
            root_lowerer._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, command.site.span)

    retained: list[GeneratedConfig] = []
    for config in configs:
        if id(config) in invalid_configs:
            continue
        owner = owner_by_config_id[id(config)]
        if owner is None:
            proven = cast(_ProvenGeneratedConfig, config)
            if proven.execution_limitation_span is not None:
                root_lowerer._issue(
                    ShellIssueReason.UNSUPPORTED_SEMANTICS,
                    proven.execution_limitation_span,
                )
            retained.append(config)
            continue
        if owner not in active_functions:
            continue
        proven = cast(_ProvenGeneratedConfig, config)
        if proven.execution_limitation_span is not None:
            root_lowerer._issue(
                ShellIssueReason.UNSUPPORTED_SEMANTICS,
                proven.execution_limitation_span,
            )
        if proven.target_proof is not None or proven.content_proof is not None:
            root_lowerer._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, config.span)
            continue
        retained.append(config)
    return retained


def _annotate_command_producer_reachability(
    root_program: _ShellProgramIR,
    nested_programs: tuple[_NestedProgramIR, ...],
    commands: tuple[_CommandIR, ...],
) -> tuple[_CommandIR, ...]:
    """Attach conservative function-producer reachability to public command sites."""

    type FunctionKey = tuple[str, int]
    functions: dict[FunctionKey, _FunctionContext] = {}
    for program in (root_program, *(nested.program for nested in nested_programs)):
        for function in program.functions:
            key = (program.program_id, function.function_id)
            if key in functions:
                functions.pop(key, None)
            else:
                functions[key] = function
    names: dict[FunctionKey, bytes] = {
        key: cast(bytes, function.name.exact_bytes)
        for key, function in functions.items()
        if function.name.state is StaticValueState.EXACT
    }
    by_name: dict[bytes, set[FunctionKey]] = {}
    for key, name in names.items():
        by_name.setdefault(name, set()).add(key)

    def owner(command: _CommandIR) -> FunctionKey | None:
        if command.function_id is not None:
            return (command.program_id, command.function_id)
        if (
            command.containing_function_program_id is not None
            and command.containing_function_id is not None
        ):
            return (
                command.containing_function_program_id,
                command.containing_function_id,
            )
        return None

    def command_name(command: _CommandIR) -> bytes | None:
        value = command.site.argv[0]
        return cast(bytes, value.exact_bytes) if value.state is StaticValueState.EXACT else None

    def exact_target(command: _CommandIR) -> FunctionKey | None:
        function_id = command.resolution.function_id
        if function_id is None:
            return None
        local = (command.program_id, function_id)
        if local in functions:
            return local
        name = command_name(command)
        candidates = (
            {key for key in by_name.get(name, ()) if key[1] == function_id}
            if name is not None
            else set()
        )
        return next(iter(candidates)) if len(candidates) == 1 else None

    definite: dict[FunctionKey | None, set[FunctionKey]] = {}
    possible_names: dict[FunctionKey | None, set[bytes]] = {}
    dynamic_possible: set[FunctionKey | None] = set()
    for command in commands:
        caller = owner(command)
        if command.resolution.kind is _CommandResolutionKind.FUNCTION:
            target = exact_target(command)
            if target is not None:
                definite.setdefault(caller, set()).add(target)
            else:
                call_name = command_name(command)
                if call_name is None:
                    dynamic_possible.add(caller)
                else:
                    possible_names.setdefault(caller, set()).add(call_name)
        elif command.resolution.kind is _CommandResolutionKind.AMBIGUOUS:
            call_name = command_name(command)
            if call_name is None:
                dynamic_possible.add(caller)
            else:
                possible_names.setdefault(caller, set()).add(call_name)

    active: set[FunctionKey] = set(definite.get(None, ()))
    ambiguous: set[FunctionKey] = set()
    active_queue = list(active)
    ambiguous_queue: list[FunctionKey] = []
    expanded_possible_names: set[bytes] = set()
    dynamic_expanded = False

    def mark_ambiguous(target: FunctionKey) -> None:
        if target not in active and target not in ambiguous:
            ambiguous.add(target)
            ambiguous_queue.append(target)

    def expand_possible_targets(caller: FunctionKey | None) -> None:
        nonlocal dynamic_expanded
        for call_name in possible_names.get(caller, ()):
            if call_name in expanded_possible_names:
                continue
            expanded_possible_names.add(call_name)
            for target in by_name.get(call_name, ()):
                mark_ambiguous(target)
        if caller in dynamic_possible and not dynamic_expanded:
            dynamic_expanded = True
            for target in functions:
                mark_ambiguous(target)

    expand_possible_targets(None)
    active_cursor = 0
    ambiguous_cursor = 0
    while active_cursor < len(active_queue) or ambiguous_cursor < len(ambiguous_queue):
        while active_cursor < len(active_queue):
            caller = active_queue[active_cursor]
            active_cursor += 1
            for target in definite.get(caller, ()):
                if target not in active:
                    active.add(target)
                    ambiguous.discard(target)
                    active_queue.append(target)
            expand_possible_targets(caller)
        while ambiguous_cursor < len(ambiguous_queue):
            caller = ambiguous_queue[ambiguous_cursor]
            ambiguous_cursor += 1
            if caller in active:
                continue
            for target in definite.get(caller, ()):
                mark_ambiguous(target)
            expand_possible_targets(caller)

    annotated: list[_CommandIR] = []
    for command in commands:
        command_owner = owner(command)
        producer = (
            CommandProducerReachability.ACTIVE
            if command_owner is None or command_owner in active
            else CommandProducerReachability.AMBIGUOUS
            if command_owner in ambiguous
            else CommandProducerReachability.INERT
        )
        annotated.append(replace(command, site=replace(command.site, producer=producer)))
    return tuple(annotated)


def _run_program_queue(
    root_lowerer: _ShellLowerer,
    root_program: _ShellProgramIR,
    *,
    budget: DependencyWorkBudget,
    file_budget: DependencyFileBudget,
    accounting_unit: ShellUnit,
    deadline_monotonic: float | None,
) -> tuple[_ShellProgramIR, tuple[ShellWorkItem, ...], bool]:
    root_generator = root_lowerer._model_state(root_program)
    stack = [
        _ProgramJob(
            lowerer=root_lowerer,
            unit=root_lowerer.unit,
            depth=0,
            execution_kind=None,
            parent_request=None,
            nested_order=None,
            initial_bindings={},
            initial_functions={},
            initial_functions_unknown=False,
            generator=root_generator,
        )
    ]
    nested_programs: list[_NestedProgramIR] = []
    nested_work_items: list[tuple[int, ShellWorkItem]] = []
    completed_nested_programs: list[tuple[int, _ShellProgramIR]] = []
    root_modeled: _ModeledState | None = None
    any_partial = root_lowerer.partial
    next_nested_order = 0

    while stack:
        job = stack[-1]
        try:
            if not job.started:
                request = next(job.generator)
                job.started = True
            else:
                response = job.pending_response or _NestedResponse()
                job.pending_response = None
                request = job.generator.send(response)
        except StopIteration as stopped:
            modeled = cast(_ModeledState, stopped.value)
            stack.pop()
            if job.execution_kind is None:
                root_modeled = modeled
                continue
            outcome = (
                ShellWorkOutcome.PARTIAL if job.lowerer.partial else ShellWorkOutcome.COMPLETED
            )
            root_lowerer.issues.extend(job.lowerer.issues)
            any_partial = any_partial or outcome is not ShellWorkOutcome.COMPLETED
            if job.nested_order is None or job.parent_request is None:
                raise RuntimeError("nested program completed without parent context") from None
            retained_program = (
                _ShellProgramIR(program_id=job.unit.unit_id)
                if job.lowerer.halted
                else modeled.program
            )
            if not job.lowerer.halted:
                completed_nested_programs.append((job.nested_order, modeled.program))
            nested_programs.append(
                _NestedProgramIR(
                    order=job.nested_order,
                    unit=job.unit,
                    depth=job.depth,
                    execution_kind=job.execution_kind,
                    parent_program_id=job.parent_request.command.program_id,
                    parent_command_start_byte=(job.parent_request.command.site.span.start_byte),
                    program=retained_program,
                    outcome=outcome,
                )
            )
            nested_work_items.append((job.nested_order, _shell_work_item(job.unit, outcome)))
            if not stack:
                raise RuntimeError("nested program completed without a parent") from None
            if job.execution_kind is _NestedExecutionKind.EVAL and not job.lowerer.partial:
                child_function_names = {
                    cast(bytes, function.name.exact_bytes)
                    for function in modeled.program.functions
                    if function.name.state is StaticValueState.EXACT
                }
                stack[-1].pending_response = _NestedResponse(
                    modeled.bindings,
                    {
                        name: (
                            None
                            if function_id is None
                            else (
                                _IMPORTED_FUNCTION_ID
                                if name in child_function_names
                                else function_id
                            )
                        )
                        for name, function_id in modeled.functions.items()
                    },
                    modeled.functions_unknown,
                )
            else:
                stack[-1].pending_response = (
                    _unknown_nested_response(cast(_NestedRequest, job.parent_request))
                    if job.lowerer.partial and job.parent_request is not None
                    else _NestedResponse()
                )
            continue

        exhaustion = file_budget.observe_shell_nested_depth(accounting_unit, request.depth)
        if exhaustion is not None:
            _retain_nested_issue(
                job.lowerer,
                ShellIssueReason.RESOURCE_LIMIT,
                request.payload.span,
                exhaustion=exhaustion,
            )
            any_partial = True
            job.pending_response = _unknown_nested_response(request)
            continue
        source_map = _nested_source_map(job.lowerer, request.payload)
        if source_map is None:
            _retain_nested_issue(
                job.lowerer,
                ShellIssueReason.UNSUPPORTED_SEMANTICS,
                request.payload.span,
            )
            any_partial = True
            job.pending_response = _unknown_nested_response(request)
            continue
        child = _nested_unit(job.lowerer, request, source_map)
        nested_order = next_nested_order
        next_nested_order += 1
        unit_exhaustion = _reserve_shell_unit(
            file_budget,
            source_map_entries=len(source_map.entries),
        )
        if unit_exhaustion is not None:
            _retain_nested_issue(
                job.lowerer,
                ShellIssueReason.RESOURCE_LIMIT,
                request.payload.span,
                exhaustion=unit_exhaustion,
            )
            nested_programs.append(
                _NestedProgramIR(
                    order=nested_order,
                    unit=child,
                    depth=request.depth,
                    execution_kind=request.execution_kind,
                    parent_program_id=request.command.program_id,
                    parent_command_start_byte=request.command.site.span.start_byte,
                    program=_ShellProgramIR(program_id=child.unit_id),
                    outcome=ShellWorkOutcome.SKIPPED,
                )
            )
            nested_work_items.append(
                (nested_order, _shell_work_item(child, ShellWorkOutcome.SKIPPED))
            )
            any_partial = True
            job.pending_response = _unknown_nested_response(request)
            continue
        parse_exhaustion = file_budget.reserve_shell_parse(len(child.raw_bytes))
        if parse_exhaustion is not None:
            _retain_nested_issue(
                job.lowerer,
                ShellIssueReason.RESOURCE_LIMIT,
                request.payload.span,
                exhaustion=parse_exhaustion,
            )
            nested_programs.append(
                _NestedProgramIR(
                    order=nested_order,
                    unit=child,
                    depth=request.depth,
                    execution_kind=request.execution_kind,
                    parent_program_id=request.command.program_id,
                    parent_command_start_byte=request.command.site.span.start_byte,
                    program=_ShellProgramIR(program_id=child.unit_id),
                    outcome=ShellWorkOutcome.SKIPPED,
                )
            )
            nested_work_items.append(
                (nested_order, _shell_work_item(child, ShellWorkOutcome.SKIPPED))
            )
            any_partial = True
            job.pending_response = _unknown_nested_response(request)
            continue
        try:
            tree = parse_bash_source(
                child.raw_bytes,
                deadline_monotonic=deadline_monotonic,
                meaningful_work=True,
            )
        except ShellParserError as error:
            reason = (
                ShellIssueReason.RUNTIME_LIMIT
                if error.reason is ShellParserFailureReason.RUNTIME_LIMIT
                else ShellIssueReason.SHELL_PARSER_UNAVAILABLE
            )
            _retain_nested_issue(job.lowerer, reason, request.payload.span)
            nested_programs.append(
                _NestedProgramIR(
                    order=nested_order,
                    unit=child,
                    depth=request.depth,
                    execution_kind=request.execution_kind,
                    parent_program_id=request.command.program_id,
                    parent_command_start_byte=request.command.site.span.start_byte,
                    program=_ShellProgramIR(program_id=child.unit_id),
                    outcome=ShellWorkOutcome.PARTIAL,
                )
            )
            nested_work_items.append(
                (nested_order, _shell_work_item(child, ShellWorkOutcome.PARTIAL))
            )
            any_partial = True
            job.pending_response = _unknown_nested_response(request)
            continue
        child_lowerer = _ShellLowerer(
            child,
            budget,
            file_budget,
            accounting_unit=accounting_unit,
        )
        child_lowerer.walk(tree.root_node)
        child_program = child_lowerer.lower()
        initial_bindings, initial_functions, initial_functions_unknown = _initial_nested_state(
            request
        )
        if child_lowerer.halted:
            root_lowerer.issues.extend(child_lowerer.issues)
            nested_programs.append(
                _NestedProgramIR(
                    order=nested_order,
                    unit=child,
                    depth=request.depth,
                    execution_kind=request.execution_kind,
                    parent_program_id=request.command.program_id,
                    parent_command_start_byte=request.command.site.span.start_byte,
                    program=_ShellProgramIR(program_id=child.unit_id),
                    outcome=ShellWorkOutcome.PARTIAL,
                )
            )
            nested_work_items.append(
                (nested_order, _shell_work_item(child, ShellWorkOutcome.PARTIAL))
            )
            any_partial = True
            job.pending_response = _unknown_nested_response(request)
            continue
        child_generator = child_lowerer._model_state(
            child_program,
            initial_bindings=initial_bindings,
            initial_functions=initial_functions,
            initial_functions_unknown=initial_functions_unknown,
            depth=request.depth,
        )
        stack.append(
            _ProgramJob(
                lowerer=child_lowerer,
                unit=child,
                depth=request.depth,
                execution_kind=request.execution_kind,
                parent_request=request,
                nested_order=nested_order,
                initial_bindings=initial_bindings,
                initial_functions=initial_functions,
                initial_functions_unknown=initial_functions_unknown,
                generator=child_generator,
            )
        )

    if root_modeled is None:
        raise RuntimeError("root shell program did not complete")
    publication_cost = sum(
        1 + len(program.commands) + len(program.assignments)
        for _, program in completed_nested_programs
    )
    if publication_cost:
        first_completed_order = min(order for order, _ in completed_nested_programs)
        publication_span = next(
            nested.unit.origin_span
            for nested in nested_programs
            if nested.order == first_completed_order
        )
        if not root_lowerer._reserve(publication_cost, span=publication_span):
            discarded_orders = {order for order, _ in completed_nested_programs}
            completed_nested_programs = []
            nested_programs = [
                nested for nested in nested_programs if nested.order not in discarded_orders
            ]
            nested_work_items = [
                (
                    order,
                    (
                        replace(item, outcome=ShellWorkOutcome.PARTIAL)
                        if order in discarded_orders
                        else item
                    ),
                )
                for order, item in nested_work_items
            ]
            any_partial = True

    retained_parent_commands = {
        (command.program_id, command.site.span.start_byte)
        for command in root_modeled.program.commands
    }
    retained_nested_programs: list[_NestedProgramIR] = []
    retained_nested_orders: set[int] = set()
    for nested in sorted(nested_programs, key=lambda item: item.order):
        parent_key = (nested.parent_program_id, nested.parent_command_start_byte)
        if parent_key not in retained_parent_commands:
            continue
        retained_nested_programs.append(nested)
        retained_nested_orders.add(nested.order)
        retained_parent_commands.update(
            (command.program_id, command.site.span.start_byte)
            for command in nested.program.commands
        )
    nested_programs = retained_nested_programs
    nested_work_items = [
        (order, item) for order, item in nested_work_items if order in retained_nested_orders
    ]
    completed_nested_programs = [
        (order, program)
        for order, program in completed_nested_programs
        if order in retained_nested_orders
    ]

    assignments = list(root_modeled.program.assignments)
    generated_configs = list(root_modeled.program.generated_configs)
    for _, program in sorted(completed_nested_programs):
        assignments.extend(program.assignments)
        generated_configs.extend(program.generated_configs)
    assignments.sort(key=lambda item: (item.site.span.start_byte, item.site.span.end_byte))
    ordered_nested_programs = tuple(sorted(nested_programs, key=lambda item: item.order))
    nested_children: dict[tuple[str, int], list[_NestedProgramIR]] = {}
    for nested in ordered_nested_programs:
        nested_children.setdefault(
            (nested.parent_program_id, nested.parent_command_start_byte), []
        ).append(nested)
    execution_commands: list[_CommandIR] = []
    execution_stack: list[tuple[_CommandIR, tuple[str, int] | None]] = [
        (command, None) for command in reversed(root_modeled.program.commands)
    ]
    while execution_stack:
        command, inherited_function = execution_stack.pop()
        if inherited_function is not None:
            command = replace(
                command,
                containing_function_program_id=inherited_function[0],
                containing_function_id=inherited_function[1],
            )
        execution_commands.append(command)
        child_function = (
            (command.program_id, command.function_id)
            if command.function_id is not None
            else inherited_function
        )
        children = nested_children.get((command.program_id, command.site.span.start_byte), ())
        for child_record in reversed(children):
            for child_command in reversed(child_record.program.commands):
                execution_stack.append((child_command, child_function))

    ordered_commands = tuple(
        replace(command, order=index) for index, command in enumerate(execution_commands)
    )
    ordered_commands = _annotate_command_producer_reachability(
        root_modeled.program,
        ordered_nested_programs,
        ordered_commands,
    )
    generated_configs = _retain_typed_function_generated_configs(
        root_lowerer,
        root_modeled.program,
        ordered_nested_programs,
        ordered_commands,
        generated_configs,
    )
    combined = replace(
        root_modeled.program,
        commands=ordered_commands,
        assignments=tuple(
            replace(assignment, order=index) for index, assignment in enumerate(assignments)
        ),
        execution_commands=ordered_commands,
        nested_programs=ordered_nested_programs,
        generated_configs=tuple(
            sorted(
                generated_configs,
                key=lambda item: (item.span.start_byte, item.span.end_byte),
            )
        ),
    )
    return (
        combined,
        tuple(item for _, item in sorted(nested_work_items)),
        any_partial or root_lowerer.partial,
    )


def _analyze_shell_unit(
    unit: ShellUnit,
    *,
    budget: DependencyWorkBudget,
    deadline_monotonic: float | None = None,
) -> _ShellAnalysisResult:
    """Private same-parse boundary retaining bounded structure for later tasks."""
    if not isinstance(unit, ShellUnit):
        raise ValueError("unit must be a ShellUnit")
    if not isinstance(budget, DependencyWorkBudget):
        raise ValueError("budget must be a DependencyWorkBudget")
    file_budget = budget.for_file(unit.origin_span.path)
    physical_size = (
        unit.source_map.physical_size_bytes
        if unit.source_map is not None
        else unit.origin_span.end_byte
    )
    if physical_size > MAX_DEPENDENCY_FILE_BYTES:
        exhaustion = DependencyWorkExhaustion(
            DependencyWorkResource.PHYSICAL_BYTES,
            physical_size,
            MAX_DEPENDENCY_FILE_BYTES,
        )
        issues: list[ShellIssue] = []
        _retain_issue(
            issues,
            _resource_issue(unit.origin_span, exhaustion),
            file_budget=file_budget,
        )
        public = ShellFrontendResult(
            issues=tuple(issues),
            work_items=(_shell_work_item(unit, ShellWorkOutcome.SKIPPED),),
        )
        return _ShellAnalysisResult(public, _ShellProgramIR())
    file_budget.register_shell_file_size(physical_size)
    if parse_exhaustion := file_budget.reserve_shell_parse(len(unit.raw_bytes)):
        issues = []
        _retain_issue(
            issues,
            _resource_issue(unit.origin_span, parse_exhaustion),
            file_budget=file_budget,
        )
        public = ShellFrontendResult(
            issues=tuple(issues),
            work_items=(_shell_work_item(unit, ShellWorkOutcome.SKIPPED),),
        )
        return _ShellAnalysisResult(public, _ShellProgramIR())

    try:
        tree = parse_bash_source(
            unit.raw_bytes,
            deadline_monotonic=deadline_monotonic,
            meaningful_work=False,
        )
    except ShellParserError as error:
        reason = (
            ShellIssueReason.RUNTIME_LIMIT
            if error.reason is ShellParserFailureReason.RUNTIME_LIMIT
            else ShellIssueReason.SHELL_PARSER_UNAVAILABLE
        )
        outcome = (
            ShellWorkOutcome.PARTIAL
            if error.outcome is ShellParserOutcome.PARTIAL
            else ShellWorkOutcome.FAILED
        )
        issues = []
        _retain_issue(
            issues,
            ShellIssue(
                reason=reason,
                outcome=outcome,
                span=unit.origin_span,
                unit_id=unit.unit_id,
            ),
            file_budget=file_budget,
        )
        public = ShellFrontendResult(
            issues=tuple(issues),
            work_items=(_shell_work_item(unit, outcome),),
        )
        return _ShellAnalysisResult(public, _ShellProgramIR())

    lowerer = _ShellLowerer(unit, budget, file_budget)
    lowerer.walk(tree.root_node)
    program = lowerer.lower()
    if lowerer.halted:
        program = replace(
            program,
            functions=(),
            commands=(),
            assignments=(),
            generated_configs=(),
            recovered_generated_drafts=(),
        )
    nested_work_items: tuple[ShellWorkItem, ...] = ()
    any_partial = lowerer.partial
    if not lowerer.halted:
        program, nested_work_items, any_partial = _run_program_queue(
            lowerer,
            program,
            budget=budget,
            file_budget=file_budget,
            accounting_unit=unit,
            deadline_monotonic=deadline_monotonic,
        )
    outcome = ShellWorkOutcome.PARTIAL if any_partial else ShellWorkOutcome.COMPLETED
    public = ShellFrontendResult(
        commands=tuple(command.site for command in program.commands),
        assignments=tuple(assignment.site for assignment in program.assignments),
        generated_configs=program.generated_configs,
        issues=tuple(lowerer.issues),
        work_items=(_shell_work_item(unit, outcome), *nested_work_items),
    )
    return _ShellAnalysisResult(public, program)


def analyze_shell_unit(
    unit: ShellUnit,
    *,
    budget: DependencyWorkBudget,
    deadline_monotonic: float | None = None,
) -> ShellFrontendResult:
    """Parse and lower one shell unit exactly once into bounded syntax-only sites."""
    return _analyze_shell_unit(
        unit,
        budget=budget,
        deadline_monotonic=deadline_monotonic,
    ).public
