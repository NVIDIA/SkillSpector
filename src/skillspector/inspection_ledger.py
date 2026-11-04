# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed contracts and safe factories for inspection-work accounting."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from hashlib import sha256
from typing import Final, NotRequired, cast

from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

MAX_INSPECTION_LEDGER_EVENTS: Final = 10_000
MAX_FINDING_OUTPUT_RECORDS: Final = 10_000
MAX_EFFECTIVE_FINDINGS: Final = MAX_FINDING_OUTPUT_RECORDS
RESOURCE_MEASUREMENT_UNITS: Final = (
    "characters",
    "bytes",
    "findings",
    "artifacts",
    "depth",
    "records",
    "seconds",
)
FORMAT_RECOVERY_SEMANTIC_ANALYZERS: Final = frozenset(
    {"semantic_security_discovery", "semantic_developer_intent", "semantic_quality_policy"}
)


class LedgerOutcome(StrEnum):
    """Terminal outcome of one inspection work item."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    FAILED = "failed"
    OUT_OF_SCOPE = "out_of_scope"


class LedgerRecordType(StrEnum):
    """Kind of ledger record."""

    WORK_ITEM = "work_item"
    SYSTEM = "system"
    SCOPE_BOUNDARY = "scope_boundary"
    # An analyzer's own configuration (e.g. its YARA rule set), not a skill
    # artifact. Its ``path`` is only a report-safe label, and every relative
    # path is also a legal file name, so path-keyed accounting must exclude
    # these records by type -- no choice of label can be collision-free.
    RULE_SET = "rule_set"


RULE_SET_SCOPE: Final = "rule_set"
"""Public ``scope`` of exception rows describing a rule set rather than a file."""


class LedgerReason(StrEnum):
    """Allowlisted reasons for omitted, skipped, or failed inspection work."""

    EXCLUDED_DIRECTORY = "excluded_directory"
    HIDDEN_FILE = "hidden_file"
    FILE_DISAPPEARED = "file_disappeared"
    NOT_REGULAR_FILE = "not_regular_file"
    STAT_ERROR = "stat_error"
    READ_ERROR = "read_error"
    MISSING_FILE_CACHE = "missing_file_cache"
    SIZE_LIMIT = "size_limit"
    BINARY_CONTENT = "binary_content"
    BINARY_TEXT_PROJECTION = "binary_text_projection"
    BINARY_EXECUTABLE_TEXT = "binary_executable_text"
    EVAL_DATASET = "eval_dataset"
    SYNTAX_ERROR = "syntax_error"
    LLM_BATCH_FAILED = "llm_batch_failed"
    LLM_STRUCTURED_RESPONSE_INVALID = "llm_structured_response_invalid"
    LLM_RESPONSE_REFUSED = "llm_response_refused"
    LLM_RESPONSE_TRUNCATED = "llm_response_truncated"
    LLM_RESPONSE_INCOMPLETE = "llm_response_incomplete"
    LLM_RESPONSE_MISSING_FIELDS = "llm_response_missing_fields"
    LLM_RESPONSE_INVALID_TOOL = "llm_response_invalid_tool"
    LLM_RESPONSE_INVALID_SCHEMA = "llm_response_invalid_schema"
    LLM_RESPONSE_EMPTY = "llm_response_empty"
    LLM_RESPONSE_LIMIT = "llm_response_limit"
    LLM_FORMAT_CHANGED = "llm_format_changed"
    LLM_FORMAT_RECOVERY_EXHAUSTED = "llm_format_recovery_exhausted"
    LLM_CONNECTION_RETRIES_EXHAUSTED = "llm_connection_retries_exhausted"
    ANALYZER_RUNTIME_ERROR = "analyzer_runtime_error"
    ANALYZER_LOAD_ERROR = "analyzer_load_error"
    UNACCOUNTED_WORK = "unaccounted_work"
    SEMANTIC_RUNTIME_INCOMPLETE = "semantic_runtime_incomplete"
    FINDING_ACCOUNTING_ERROR = "finding_accounting_error"
    DISABLED_BY_CONFIGURATION = "disabled_by_configuration"
    MISSING_CREDENTIALS = "missing_credentials"
    RULES_UNAVAILABLE = "rules_unavailable"
    MANIFEST_ABSENT = "manifest_absent"
    NO_APPLICABLE_FILES = "no_applicable_files"
    OMS_SIGNATURE = "oms_signature"
    BASELINE_FILE = "baseline_file"
    ARCHIVE_FORMAT_MISMATCH = "archive_format_mismatch"
    ARCHIVE_MALFORMED = "archive_malformed"
    ARCHIVE_ENCRYPTED = "archive_encrypted"
    ARCHIVE_UNSUPPORTED_COMPRESSION = "archive_unsupported_compression"
    ARCHIVE_UNSUPPORTED_MEMBER = "archive_unsupported_member"
    ARCHIVE_TRUNCATED = "archive_truncated"
    ARCHIVE_UNSAFE_MEMBER_PATH = "archive_unsafe_member_path"
    ARCHIVE_AMBIGUOUS_MEMBER_PATH = "archive_ambiguous_member_path"
    ARTIFACT_PATH_COLLISION = "artifact_path_collision"
    ARCHIVE_LINK_MEMBER = "archive_link_member"
    ARCHIVE_DEPTH_LIMIT = "archive_depth_limit"
    ARCHIVE_MEMBER_LIMIT = "archive_member_limit"
    ARCHIVE_SIZE_LIMIT = "archive_size_limit"
    ARCHIVE_MEMBER_SIZE_LIMIT = "archive_member_size_limit"
    ARCHIVE_COMPRESSION_RATIO = "archive_compression_ratio"
    ARCHIVE_TIME_LIMIT = "archive_time_limit"
    VCS_METADATA = "vcs_metadata"
    OPAQUE_CONTENT = "opaque_content"
    UNSUPPORTED_PRIMARY_CONTENT = "unsupported_primary_content"
    REFERENCED_UNINSPECTED = "referenced_uninspected"
    REFERENCE_EXTRACTION_LIMIT = "reference_extraction_limit"
    REFERENCE_MISSING = "reference_missing"
    REFERENCE_UNRESOLVED = "reference_unresolved"
    MANIFEST_PARSE_ERROR = "manifest_parse_error"
    DEPENDENCY_PARSE_ERROR = "dependency_parse_error"
    MANIFEST_PARSE_LIMIT = "manifest_parse_limit"
    ARTIFACT_COUNT_LIMIT = "artifact_count_limit"
    TRAVERSAL_DEPTH_LIMIT = "traversal_depth_limit"
    TOTAL_BYTES_LIMIT = "total_bytes_limit"
    RUNTIME_LIMIT = "runtime_limit"
    EXCLUDED_EXECUTABLE_CONTENT = "excluded_executable_content"
    OUTPUT_LIMIT = "output_limit"
    TRANSITIVE_CHILD_SCAN_FAILED = "transitive_child_scan_failed"
    STATIC_PARSE_LIMIT = "static_parse_limit"
    PYTHON_SOURCE_AMBIGUOUS = "python_source_ambiguous"
    PYTHON_SOURCE_DECODE_ERROR = "python_source_decode_error"
    OBFUSCATED_INSTRUCTION_TEXT = "obfuscated_instruction_text"


def only_missing_reference_exceptions(exceptions: object) -> bool:
    """Return whether non-empty ledger exceptions are only missing references."""
    return (
        isinstance(exceptions, list)
        and bool(exceptions)
        and all(
            isinstance(exception, Mapping)
            and exception.get("reason_code") == LedgerReason.REFERENCE_MISSING
            for exception in exceptions
        )
    )


def format_recovery_policy_accepted(completeness: Mapping[str, object]) -> bool:
    """Validate the derived format-only acceptance without claiming complete inspection.

    The producer also proves runtime accounting before setting ``policy_accepted``.
    Consumers must retain that projection intact: a flag alone, extra limitations,
    or a warning for a different file cannot waive a remaining inspection gap.
    """
    if (
        completeness.get("policy_accepted") is not True
        or completeness.get("execution_successful") is not True
        or completeness.get("status") != "partial"
        or completeness.get("is_complete") is not False
        or completeness.get("entirely_uninspected_files") != 0
        or any(completeness.get(name) for name in ("exceptions", "errors", "failures"))
    ):
        return False
    scope_rows = completeness.get("scope_exclusions")
    if not isinstance(scope_rows, list) or any(
        not isinstance(row, Mapping)
        or row.get("outcome") != LedgerOutcome.OUT_OF_SCOPE
        or row.get("fatal") is not False
        or row.get("reason_code")
        not in (LedgerReason.BASELINE_FILE, LedgerReason.OMS_SIGNATURE, LedgerReason.VCS_METADATA)
        for row in scope_rows
    ):
        return False
    counts = [
        completeness.get(name)
        for name in (
            "total_components",
            "scanned_components",
            "fully_inspected_files",
            "partially_inspected_files",
            "entirely_uninspected_files",
        )
    ]
    if not all(type(value) is int and value >= 0 for value in counts):
        return False
    total, scanned, fully, partially, entirely = (cast(int, value) for value in counts)
    if total != fully + partially + entirely or scanned != fully:
        return False
    rows = completeness.get("ledger_exceptions")
    warnings = completeness.get("warnings")
    statuses = completeness.get("analyzer_statuses")
    limitations = completeness.get("limitations")
    if not all(isinstance(value, list) for value in (rows, warnings, statuses, limitations)):
        return False
    assert isinstance(rows, list) and isinstance(warnings, list)
    assert isinstance(statuses, list) and isinstance(limitations, list)
    if not all(isinstance(value, str) for value in limitations):
        return False
    if not rows or not warnings or len(warnings) > MAX_INSPECTION_LEDGER_EVENTS:
        return False
    format_rows: list[dict[str, object]] = []
    warning_counts: dict[tuple[str, str], int] = {}
    partial_paths: set[tuple[str, str]] = set()
    identities: set[tuple[object, ...]] = set()
    runtime_rows: list[Mapping[str, object]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("fatal") is not False:
            return False
        if row.get("outcome") != LedgerOutcome.PARTIAL:
            return False
        reason = row.get("reason_code")
        if reason == LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED:
            analyzers = row.get("analyzers")
            phase = row.get("phase")
            path = row.get("path")
            start_line, end_line = row.get("start_line"), row.get("end_line")
            if (
                phase not in ("semantic", "meta")
                or not isinstance(path, str)
                or not path
                or not isinstance(analyzers, list)
                or not analyzers
                or not all(isinstance(value, str) and value for value in analyzers)
                or any(
                    value is not None and type(value) is not int for value in (start_line, end_line)
                )
            ):
                return False
            if len(set(analyzers)) != len(analyzers):
                return False
            allowed_analyzers = (
                {"meta_analyzer"}
                if phase == "meta"
                else FORMAT_RECOVERY_SEMANTIC_ANALYZERS | {"mcp_tool_poisoning"}
            )
            if not set(analyzers) <= allowed_analyzers:
                return False
            try:
                if _normalize_relative_path(path) != path:
                    return False
                _validate_range(cast(int | None, start_line), cast(int | None, end_line))
            except ValueError:
                return False
            source = row.get("source_identity", "")
            if not isinstance(source, str):
                return False
            if source and not str(row["path"]).startswith(source + "/"):
                return False
            identity = (
                source,
                row["path"],
                row.get("start_line"),
                row.get("end_line"),
                row["phase"],
                tuple(analyzers),
            )
            if identity in identities:
                return False
            identities.add(identity)
            for analyzer in analyzers:
                key = (source, analyzer)
                warning_counts[key] = warning_counts.get(key, 0) + 1
            if row["phase"] == "semantic":
                partial_paths.add((source, str(row["path"])))
            format_rows.append(row)
        elif reason == LedgerReason.REFERENCE_MISSING:
            # These existing caveats describe absent references, not uninspected bytes.
            path, source = row.get("path"), row.get("source_identity", "")
            if (
                row.get("phase") != "reference_resolution"
                or row.get("analyzers")
                or row.get("error_class")
                or not isinstance(path, str)
                or not path
                or not isinstance(source, str)
                or (source and not path.startswith(source + "/"))
                or any(
                    value is not None and type(value) is not int
                    for value in (row.get("start_line"), row.get("end_line"))
                )
            ):
                return False
            try:
                if _normalize_relative_path(path) != path:
                    return False
                _validate_range(
                    cast(int | None, row.get("start_line")),
                    cast(int | None, row.get("end_line")),
                )
            except ValueError:
                return False
        elif not (
            reason == LedgerReason.SEMANTIC_RUNTIME_INCOMPLETE
            and row.get("phase") == "semantic_runtime"
            and row.get("start_line") is None
            and row.get("end_line") is None
            and not row.get("analyzers")
            and not row.get("error_class")
        ):
            return False
        else:
            runtime_rows.append(row)
    expected_warnings = [
        {**row, "assessment_complete": False, "accepted_by_policy": True} for row in format_rows
    ]
    if not format_rows or warnings != expected_warnings:
        return False
    if completeness.get("partially_inspected_files") != len(partial_paths):
        return False
    runtime_scopes: set[str] = set()
    for row in runtime_rows:
        source = row.get("source_identity", "")
        if (
            not isinstance(source, str)
            or source in runtime_scopes
            or row.get("path") != (source + "/SKILL.md" if source else "SKILL.md")
            or (source and not any(scope == source for scope, _ in warning_counts))
        ):
            return False
        runtime_scopes.add(source)
    expected_limitations: list[str] = []
    seen_statuses: set[tuple[str, str]] = set()
    semantic_scopes = {scope for scope, _analyzer in warning_counts}
    for status in statuses:
        if not isinstance(status, Mapping):
            return False
        source, analyzer = status.get("source_identity", ""), status.get("analyzer_id")
        if not isinstance(source, str) or not isinstance(analyzer, str) or not analyzer:
            return False
        key = (source, analyzer)
        if key in seen_statuses:
            return False
        seen_statuses.add(key)
        if analyzer in FORMAT_RECOVERY_SEMANTIC_ANALYZERS:
            semantic_scopes.add(source)
        counts = [status.get(name) for name in ("planned_work", "completed", "partial")]
        if not all(type(value) is int and value >= 0 for value in counts):
            return False
        planned, completed, partial = (cast(int, value) for value in counts)
        if (
            planned != completed + partial
            or partial != warning_counts.get(key, 0)
            or any(
                type(status.get(name)) is not int or status.get(name) != 0
                for name in ("failed", "skipped", "unaccounted")
            )
        ):
            return False
        if status.get("status") == "not_applicable" and planned != 0:
            return False
        if status.get("status") in ("completed", "not_applicable") and partial == 0:
            continue
        if status.get("status") != "degraded" or key not in warning_counts:
            return False
        if status.get("reason_code") not in (None, LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED):
            return False
        expected_limitations.append(
            str(status.get("message") or f"Analyzer {status['analyzer_id']} status: degraded.")
        )
    return (
        warning_counts.keys() <= seen_statuses
        and sorted(limitations) == sorted(expected_limitations)
        and all(
            (source, analyzer) in seen_statuses
            for source in semantic_scopes
            for analyzer in FORMAT_RECOVERY_SEMANTIC_ANALYZERS
        )
    )


REASON_MESSAGES: Final[dict[LedgerReason, str]] = {
    LedgerReason.EXCLUDED_DIRECTORY: ("Directory tree is excluded from the configured scan scope."),
    LedgerReason.HIDDEN_FILE: "Hidden file is excluded from the configured scan scope.",
    LedgerReason.FILE_DISAPPEARED: ("Inventoried file disappeared before it could be inspected."),
    LedgerReason.NOT_REGULAR_FILE: "Inventoried path is no longer a regular file.",
    LedgerReason.STAT_ERROR: "Filesystem metadata could not be read.",
    LedgerReason.READ_ERROR: "File content could not be read.",
    LedgerReason.MISSING_FILE_CACHE: "Applicable analyzer could not obtain file content.",
    LedgerReason.SIZE_LIMIT: "File exceeds this analyzer's character limit.",
    LedgerReason.BINARY_CONTENT: "Binary content is unsupported by this analyzer.",
    LedgerReason.BINARY_TEXT_PROJECTION: (
        "Only bounded printable text was inspected; binary structure and behavior are outside scope."
    ),
    LedgerReason.BINARY_EXECUTABLE_TEXT: (
        "Readable text was inspected, but binary executable behavior remains outside analyzer coverage."
    ),
    LedgerReason.EVAL_DATASET: (
        "Evaluation dataset prose is excluded from static pattern analysis."
    ),
    LedgerReason.SYNTAX_ERROR: "Python source could not be parsed.",
    LedgerReason.LLM_BATCH_FAILED: "LLM analysis failed for this file range.",
    LedgerReason.LLM_STRUCTURED_RESPONSE_INVALID: (
        "LLM response could not be validated after bounded recovery."
    ),
    LedgerReason.LLM_RESPONSE_REFUSED: "The model declined to complete this assessment.",
    LedgerReason.LLM_RESPONSE_TRUNCATED: "The model's answer was cut short by its output limit.",
    LedgerReason.LLM_RESPONSE_INCOMPLETE: "The model did not finish this assessment.",
    LedgerReason.LLM_RESPONSE_MISSING_FIELDS: "The model's assessment omitted required information.",
    LedgerReason.LLM_RESPONSE_INVALID_TOOL: "The model returned missing, unexpected, or ambiguous tool calls.",
    LedgerReason.LLM_RESPONSE_INVALID_SCHEMA: "The model's assessment contained invalid field values.",
    LedgerReason.LLM_RESPONSE_EMPTY: "The model returned no assessment.",
    LedgerReason.LLM_RESPONSE_LIMIT: "The model's response exceeded processing or formatting limits.",
    LedgerReason.LLM_FORMAT_CHANGED: "Formatting could not preserve the complete original assessment.",
    LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED: (
        "The AI answered, but its full assessment could not be verified after bounded recovery. "
        "The formatting-recovery policy permits continuing with this warning; known findings remain."
    ),
    LedgerReason.LLM_CONNECTION_RETRIES_EXHAUSTED: (
        "Transient LLM provider failure persisted after bounded retries."
    ),
    LedgerReason.ANALYZER_RUNTIME_ERROR: ("Analyzer failed after beginning applicable work."),
    LedgerReason.ANALYZER_LOAD_ERROR: (
        "Analyzer module failed to load and never began any inspection work."
    ),
    LedgerReason.UNACCOUNTED_WORK: ("Planned inspection work has no unique terminal outcome."),
    LedgerReason.SEMANTIC_RUNTIME_INCOMPLETE: (
        "Requested semantic analysis did not produce complete per-source runtime telemetry."
    ),
    LedgerReason.FINDING_ACCOUNTING_ERROR: (
        "Finding identity could not be reconciled with completed work."
    ),
    LedgerReason.DISABLED_BY_CONFIGURATION: (
        "Analyzer was disabled by the requested configuration."
    ),
    LedgerReason.MISSING_CREDENTIALS: ("Analyzer credentials were unavailable before execution."),
    LedgerReason.RULES_UNAVAILABLE: ("Analyzer rules were unavailable before execution."),
    LedgerReason.MANIFEST_ABSENT: ("No compatible manifest was present for this analyzer."),
    LedgerReason.NO_APPLICABLE_FILES: ("No files matched this analyzer's applicability contract."),
    LedgerReason.OMS_SIGNATURE: (
        "DSSE content decoding is incomplete or unsupported; raw fields were retained for inspection. Signature authenticity was not verified."
    ),
    LedgerReason.BASELINE_FILE: (
        "The explicitly selected suppression baseline is excluded from content analysis."
    ),
    LedgerReason.ARCHIVE_FORMAT_MISMATCH: (
        "Artifact bytes do not match the container format implied by its filename."
    ),
    LedgerReason.ARCHIVE_MALFORMED: "Archive or compressed content is malformed.",
    LedgerReason.ARCHIVE_ENCRYPTED: "Archive member is encrypted and could not be inspected.",
    LedgerReason.ARCHIVE_UNSUPPORTED_COMPRESSION: (
        "Archive member uses an unsupported compression method."
    ),
    LedgerReason.ARCHIVE_UNSUPPORTED_MEMBER: (
        "Archive member uses a type or feature that is not supported for inspection."
    ),
    LedgerReason.ARCHIVE_TRUNCATED: "Archive member is truncated or unreadable.",
    LedgerReason.ARCHIVE_UNSAFE_MEMBER_PATH: (
        "Archive member uses an absolute, traversal, or otherwise unsafe path."
    ),
    LedgerReason.ARCHIVE_AMBIGUOUS_MEMBER_PATH: (
        "Archive member name has an ambiguous or duplicate provenance identity."
    ),
    LedgerReason.ARTIFACT_PATH_COLLISION: (
        "A disk path containing '!/' collides with an archive member. Rename the directory "
        "ending in '!'; the colliding archive member was not analyzed."
    ),
    LedgerReason.ARCHIVE_LINK_MEMBER: "Archive link member was not followed or read.",
    LedgerReason.ARCHIVE_DEPTH_LIMIT: "Nested archive depth limit was reached.",
    LedgerReason.ARCHIVE_MEMBER_LIMIT: "Cumulative archive member limit was reached.",
    LedgerReason.ARCHIVE_SIZE_LIMIT: "Cumulative archive content size limit was reached.",
    LedgerReason.ARCHIVE_MEMBER_SIZE_LIMIT: "Archive member exceeds the per-file analysis limit.",
    LedgerReason.ARCHIVE_COMPRESSION_RATIO: (
        "Archive member exceeds the permitted compression ratio."
    ),
    LedgerReason.ARCHIVE_TIME_LIMIT: "Cumulative archive inspection time limit was reached.",
    LedgerReason.VCS_METADATA: (
        "VCS object and history metadata is outside the bounded artifact inspection profile."
    ),
    LedgerReason.OPAQUE_CONTENT: "Artifact contents could not be fully interpreted.",
    LedgerReason.UNSUPPORTED_PRIMARY_CONTENT: (
        "The requested file or primary instructions could not be interpreted. "
        "Provide UTF-8 text, a supported ZIP, TAR, or GZIP container, "
        "or an extracted directory instead."
    ),
    LedgerReason.REFERENCED_UNINSPECTED: ("A referenced artifact was not completely inspected."),
    LedgerReason.REFERENCE_EXTRACTION_LIMIT: (
        "Reference extraction reached an explicit resource bound before completion."
    ),
    LedgerReason.REFERENCE_MISSING: (
        "A local path-like reference does not match any bundled artifact,"
        " such as a file the skill writes at runtime."
    ),
    LedgerReason.REFERENCE_UNRESOLVED: (
        "A local path-like reference matched more than one bundled artifact"
        " and could not be resolved to a single target."
    ),
    LedgerReason.MANIFEST_PARSE_ERROR: (
        "Manifest frontmatter is malformed or uses an unsupported value shape."
    ),
    LedgerReason.DEPENDENCY_PARSE_ERROR: (
        "Dependency metadata could not be parsed into supported registry coordinates."
    ),
    LedgerReason.MANIFEST_PARSE_LIMIT: (
        "Manifest frontmatter could not be completely examined within its resource limits."
    ),
    LedgerReason.ARTIFACT_COUNT_LIMIT: ("Bundle discovery reached its artifact-count limit."),
    LedgerReason.TRAVERSAL_DEPTH_LIMIT: ("Bundle discovery reached its directory-depth limit."),
    LedgerReason.TOTAL_BYTES_LIMIT: "Bundle caching reached its aggregate byte limit.",
    LedgerReason.RUNTIME_LIMIT: "Inspection reached its configured runtime limit.",
    LedgerReason.EXCLUDED_EXECUTABLE_CONTENT: (
        "Executable content was inventoried but excluded from content analysis."
    ),
    LedgerReason.OUTPUT_LIMIT: "Inspection reached its configured output limit.",
    LedgerReason.TRANSITIVE_CHILD_SCAN_FAILED: (
        "A transitive child scan failed before complete inspection."
    ),
    LedgerReason.STATIC_PARSE_LIMIT: (
        "A security-relevant expression could not be fully resolved by the bounded static parser."
    ),
    LedgerReason.PYTHON_SOURCE_AMBIGUOUS: (
        "Python execution intent depends on runtime or platform-specific shebang semantics."
    ),
    LedgerReason.PYTHON_SOURCE_DECODE_ERROR: (
        "Python source bytes could not be decoded under their declared encoding."
    ),
    LedgerReason.OBFUSCATED_INSTRUCTION_TEXT: (
        "Obfuscated instruction text could not be fully evaluated by the deterministic layer."
    ),
}


def outcome_for_llm_batch_failure(reason: LedgerReason) -> LedgerOutcome:
    """Return the terminal ledger outcome for an exhausted LLM batch."""
    return (
        LedgerOutcome.SKIPPED
        if reason
        in {
            LedgerReason.LLM_STRUCTURED_RESPONSE_INVALID,
            LedgerReason.LLM_RESPONSE_REFUSED,
            LedgerReason.LLM_RESPONSE_TRUNCATED,
            LedgerReason.LLM_RESPONSE_INCOMPLETE,
            LedgerReason.LLM_RESPONSE_MISSING_FIELDS,
            LedgerReason.LLM_RESPONSE_INVALID_TOOL,
            LedgerReason.LLM_RESPONSE_INVALID_SCHEMA,
            LedgerReason.LLM_RESPONSE_EMPTY,
            LedgerReason.LLM_RESPONSE_LIMIT,
            LedgerReason.LLM_FORMAT_CHANGED,
        }
        else LedgerOutcome.FAILED
    )


class PlannedWorkTarget(TypedDict):
    """One analyzer work item expected to have a terminal ledger row."""

    work_id: str
    path: str
    start_line: int | None
    end_line: int | None


class ResourceMeasurements(TypedDict, total=False):
    """Optional measured resource values with units encoded in their field names."""

    observed_characters: int
    limit_characters: int
    observed_bytes: int
    limit_bytes: int
    observed_findings: int
    limit_findings: int
    observed_artifacts: int
    limit_artifacts: int
    observed_depth: int
    limit_depth: int
    observed_records: int
    limit_records: int
    observed_seconds: float
    limit_seconds: float


class ParserCause(StrEnum):
    RESOURCE_LIMIT = "resource_limit"
    BOUNDARY_UNRESOLVED = "boundary_unresolved"
    MALFORMED_INPUT = "malformed_input"
    UNSUPPORTED_SYNTAX = "unsupported_syntax"
    RUNTIME_DEPENDENT = "runtime_dependent"


class ParserCheck(StrEnum):
    COMMAND_WORD_CHARACTERS = "command_word_unquoted_characters"
    COMMAND_ARGUMENT_CHARACTERS = "command_argument_characters"
    COMMAND_WORD_BOUNDARY = "command_word_boundary"
    COMMAND_ARGUMENT_BOUNDARY = "command_argument_boundary"
    COMMAND_IDENTITY = "runtime_command_identity"
    PRINTF_INPUT_CHARACTERS = "printf_input_characters"
    PRINTF_ARGUMENT_COUNT = "printf_argument_count"
    PRINTF_ARGUMENT_CHARACTERS = "printf_argument_characters"
    PRINTF_OUTPUT_CHARACTERS = "printf_output_characters"
    PRINTF_FORMAT = "printf_format"
    PRINTF_INVOCATION = "printf_invocation"
    WRAPPER_DEPTH = "wrapper_depth"
    WRAPPER_ARGUMENTS = "wrapper_arguments"
    WRAPPER_CLAUSE_STEPS = "wrapper_clause_steps"
    COMMAND_STRING_DEPTH = "command_string_depth"
    COMMAND_STRING_RUNTIME = "command_string_runtime"
    COMMAND_STRING_EFFECTS = "command_string_effects"
    STARTUP_PROOF_CHARACTERS = "startup_proof_characters"
    BRACE_CHARACTERS = "brace_characters"
    BRACE_ALTERNATIVES = "brace_alternatives"
    BRACE_PRODUCTS = "brace_products"
    BRACE_ROUNDS = "brace_rounds"
    BRACE_SYNTAX = "brace_syntax"
    MARKDOWN_LITERAL_OWNERSHIP = "markdown_literal_ownership"
    HEREDOC_HEADER = "heredoc_header"
    HEREDOC_COUNT = "heredoc_count"
    HEREDOC_BODY = "heredoc_body"
    HEREDOC_GRAMMAR = "heredoc_grammar"
    EXPANSION_BOUNDARY = "expansion_boundary"


class ParserUncheckedScope(StrEnum):
    COMMAND_WORD = "command_word"
    COMMAND_ARGUMENTS = "command_arguments"
    COMMAND_STRING = "command_string"
    EXPRESSION = "expression"
    REMAINING_INPUT = "remaining_input"


@dataclass(frozen=True)
class StaticParserDiagnostic:
    """Internal parser evidence in the current view's character coordinates."""

    cause: ParserCause
    check: ParserCheck
    start: int
    end: int
    unchecked_scope: ParserUncheckedScope
    measurements: ResourceMeasurements = field(default_factory=ResourceMeasurements)
    observed_is_lower_bound: bool = False
    full_input_checked: bool = False
    location_precision: str = "source_span"

    def shifted(self, offset: int) -> StaticParserDiagnostic:
        return replace(self, start=self.start + offset, end=self.end + offset)

    def enclosed(self, start: int, end: int) -> StaticParserDiagnostic:
        """Attribute decoded text to its carrier without inventing inner columns."""
        return replace(self, start=start, end=end, location_precision="enclosing_expression")


class ParserDiagnostic(ResourceMeasurements):
    """Payload-free public explanation of a parser's first unresolved scope."""

    cause: ParserCause
    check: ParserCheck
    source_start_offset: int
    source_end_offset: int
    start_line: int
    end_line: int
    unchecked_scope: ParserUncheckedScope
    observed_is_lower_bound: bool
    full_input_checked: bool
    location_precision: str


def sanitize_parser_diagnostic(value: object) -> ParserDiagnostic | None:
    """Accept only known parser identifiers, source coordinates and measurements."""
    if not isinstance(value, Mapping):
        return None
    try:
        if not all(
            isinstance(value.get(key), str) for key in ("cause", "check", "unchecked_scope")
        ):
            return None
        cause = ParserCause(value.get("cause"))
        check = ParserCheck(value.get("check"))
        scope = ParserUncheckedScope(value.get("unchecked_scope"))
    except (TypeError, ValueError):
        return None
    offsets = [value.get("source_start_offset"), value.get("source_end_offset")]
    lines = [value.get("start_line"), value.get("end_line")]
    if not all(type(number) is int and 0 <= number <= (1 << 63) - 1 for number in offsets + lines):
        return None
    start, end = cast(list[int], offsets)
    start_line, end_line = cast(list[int], lines)
    if start >= end or start_line < 1 or end_line < start_line:
        return None
    precision = value.get("location_precision")
    lower_bound = value.get("observed_is_lower_bound")
    checked = value.get("full_input_checked")
    if (
        type(precision) is not str
        or precision not in {"source_span", "enclosing_expression"}
        or type(lower_bound) is not bool
        or type(checked) is not bool
    ):
        return None
    if cause is ParserCause.MALFORMED_INPUT and (not checked or precision != "source_span"):
        cause = ParserCause.BOUNDARY_UNRESOLVED
    result: ParserDiagnostic = {
        "cause": cause,
        "check": check,
        "source_start_offset": start,
        "source_end_offset": end,
        "start_line": start_line,
        "end_line": end_line,
        "unchecked_scope": scope,
        "observed_is_lower_bound": lower_bound,
        "full_input_checked": checked,
        "location_precision": precision,
    }
    result.update(resource_measurements(value))
    return result


def resource_measurements(event: Mapping[str, object]) -> ResourceMeasurements:
    """Allow only bounded numeric diagnostics, never arbitrary graph payloads."""
    measurements: dict[str, int | float] = {}
    for unit in RESOURCE_MEASUREMENT_UNITS:
        for prefix in ("observed_", "limit_"):
            key = prefix + unit
            value = event.get(key)
            # Machine resource counters fit in 63 bits; larger graph values
            # must not amplify report output through unbounded decimal text.
            if type(value) is int and 0 <= value <= (1 << 63) - 1:
                measurements[key] = value
            elif unit == "seconds" and type(value) is float and math.isfinite(value) and value >= 0:
                measurements[key] = value
    return cast(ResourceMeasurements, measurements)


class InspectionLedgerEvent(ResourceMeasurements):
    """Internal terminal evidence for one work item or scope boundary."""

    work_id: str
    record_type: LedgerRecordType
    outcome: LedgerOutcome
    phase: str
    path: str
    start_line: int | None
    end_line: int | None
    input_finding_ids: list[str]
    emitted_finding_ids: list[str]
    analyzer_id: NotRequired[str]
    reason_code: NotRequired[LedgerReason]
    message: NotRequired[str]
    error_class: NotRequired[str]
    stage: NotRequired[str]
    parser_diagnostic: NotRequired[ParserDiagnostic]


class AnalyzerStatusEvent(TypedDict):
    """Run-level analyzer status and its internal planned work targets."""

    analyzer_id: str
    status: str
    planned_work: list[PlannedWorkTarget]
    reason_code: NotRequired[LedgerReason]
    message: NotRequired[str]
    source_url: NotRequired[str]
    source_identity: NotRequired[str]
    source_digest: NotRequired[str]


class InspectionLedgerException(ResourceMeasurements):
    """Public exceptional projection derived from an internal ledger event."""

    outcome: LedgerOutcome
    phase: str
    reason_code: LedgerReason
    message: str
    path: str
    start_line: int | None
    end_line: int | None
    error_class: NotRequired[str]
    analyzers: NotRequired[list[str]]
    fatal: NotRequired[bool]
    scope: NotRequired[str]
    parser_diagnostic: NotRequired[ParserDiagnostic]
    source_identity: NotRequired[str]


class AnalysisCompleteness(TypedDict):
    """Public inspection-completeness projection derived during finalization."""

    total_components: int
    scanned_components: int
    coverage_percent: float
    is_complete: bool
    status: str
    execution_successful: bool
    fully_inspected_files: int
    partially_inspected_files: int
    entirely_uninspected_files: int
    ledger_exceptions: list[InspectionLedgerException]
    scope_exclusions: list[InspectionLedgerException]
    analyzer_statuses: list[dict[str, object]]
    references: NotRequired[list[dict[str, object]]]
    limitations: NotRequired[list[str]]
    findings_before_filtering: NotRequired[int]
    findings_after_filtering: NotRequired[int]
    dependency_inspection: NotRequired[dict[str, object]]
    policy_accepted: NotRequired[bool]
    warnings: NotRequired[list[dict[str, object]]]


def _normalize_relative_path(path: str, *, scope_boundary: bool = False) -> str:
    """Return a normalized report-safe relative POSIX path."""
    raw_path = path.replace("\\", "/")
    if not raw_path or raw_path.startswith("/") or raw_path.startswith("//"):
        raise ValueError("path must be a relative POSIX path")
    if len(raw_path) >= 2 and raw_path[1] == ":":
        raise ValueError("path must be a relative POSIX path")

    parts = raw_path.split("/")
    if any(part == ".." for part in parts):
        raise ValueError("path must be a relative POSIX path without parent traversal")
    normalized_parts = [part for part in parts if part not in ("", ".")]
    if not normalized_parts:
        raise ValueError("path must be a relative POSIX path")

    normalized = "/".join(normalized_parts)
    return f"{normalized}/" if scope_boundary else normalized


def _deduplicate_ids(finding_ids: Iterable[str]) -> list[str]:
    """Deduplicate finding IDs while retaining their first-seen order."""
    unique_ids: list[str] = []
    seen: set[str] = set()
    for finding_id in finding_ids:
        if finding_id not in seen:
            unique_ids.append(finding_id)
            seen.add(finding_id)
    return unique_ids


def _validate_range(start_line: int | None, end_line: int | None) -> None:
    """Reject incomplete or invalid source ranges."""
    if (start_line is None) != (end_line is None):
        raise ValueError("start_line and end_line must both be set or both be None")
    if start_line is not None:
        if end_line is None or start_line < 1 or end_line < start_line:
            raise ValueError("line ranges must be positive and inclusive")


def inspection_work_id(
    analyzer_id: str,
    path: str,
    start_line: int | None,
    end_line: int | None,
) -> str:
    """Build a deterministic ID for one analyzer/path/range work item."""
    _validate_range(start_line, end_line)
    normalized_path = _normalize_relative_path(path)
    canonical = "\x1f".join((analyzer_id, normalized_path, str(start_line), str(end_line)))
    return f"work-{sha256(canonical.encode('utf-8')).hexdigest()}"


def _is_meta_phase(phase: str) -> bool:
    """Return whether a row tracks meta-analysis finding lineage."""
    return phase == "meta"


def ledger_event(
    *,
    outcome: LedgerOutcome,
    phase: str,
    path: str,
    analyzer_id: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    record_type: LedgerRecordType = LedgerRecordType.WORK_ITEM,
    reason: LedgerReason | None = None,
    input_finding_ids: Iterable[str] = (),
    emitted_finding_ids: Iterable[str] = (),
    error_class: str | None = None,
    stage: str | None = None,
    observed_characters: int | None = None,
    limit_characters: int | None = None,
    observed_bytes: int | None = None,
    limit_bytes: int | None = None,
    observed_findings: int | None = None,
    limit_findings: int | None = None,
    observed_artifacts: int | None = None,
    limit_artifacts: int | None = None,
    observed_depth: int | None = None,
    limit_depth: int | None = None,
    observed_records: int | None = None,
    limit_records: int | None = None,
    observed_seconds: float | None = None,
    limit_seconds: float | None = None,
    parser_diagnostic: ParserDiagnostic | None = None,
) -> InspectionLedgerEvent:
    """Create one validated terminal ledger record without sensitive payloads."""
    _validate_range(start_line, end_line)
    normalized_path = _normalize_relative_path(
        path,
        scope_boundary=(
            record_type is LedgerRecordType.SCOPE_BOUNDARY and path.endswith(("/", "\\"))
        ),
    )
    input_ids = _deduplicate_ids(input_finding_ids)
    emitted_ids = _deduplicate_ids(emitted_finding_ids)
    is_meta = _is_meta_phase(phase)

    if outcome is LedgerOutcome.COMPLETED:
        if reason is not None:
            raise ValueError("completed ledger events cannot include a reason")
    elif reason is None:
        raise ValueError("non-completed ledger events require a reason")

    if not is_meta:
        if input_ids:
            raise ValueError("producer ledger events cannot consume findings")
        if outcome not in (LedgerOutcome.COMPLETED, LedgerOutcome.PARTIAL) and emitted_ids:
            raise ValueError("non-completed producers cannot reference findings")
    elif outcome is LedgerOutcome.COMPLETED and not set(emitted_ids).issubset(input_ids):
        raise ValueError("completed meta events must emit a subset of input findings")
    elif outcome in (LedgerOutcome.FAILED, LedgerOutcome.SKIPPED, LedgerOutcome.PARTIAL):
        if emitted_ids != input_ids:
            raise ValueError("failed or skipped meta events must pass every input finding through")
    elif outcome is not LedgerOutcome.COMPLETED:
        if input_ids or emitted_ids:
            raise ValueError("non-completed meta events cannot reference findings")

    work_identity = analyzer_id or f"{record_type.value}:{phase}"
    event: InspectionLedgerEvent = {
        "work_id": inspection_work_id(work_identity, normalized_path, start_line, end_line),
        "record_type": record_type,
        "outcome": outcome,
        "phase": phase,
        "path": normalized_path,
        "start_line": start_line,
        "end_line": end_line,
        "input_finding_ids": input_ids,
        "emitted_finding_ids": emitted_ids,
    }
    if analyzer_id is not None:
        event["analyzer_id"] = analyzer_id
    if reason is not None:
        event["reason_code"] = reason
        event["message"] = REASON_MESSAGES[reason]
    if error_class is not None:
        event["error_class"] = error_class
    if stage is not None:
        event["stage"] = stage
    if observed_characters is not None:
        event["observed_characters"] = observed_characters
    if limit_characters is not None:
        event["limit_characters"] = limit_characters
    if observed_bytes is not None:
        event["observed_bytes"] = observed_bytes
    if limit_bytes is not None:
        event["limit_bytes"] = limit_bytes
    if observed_findings is not None:
        event["observed_findings"] = observed_findings
    if limit_findings is not None:
        event["limit_findings"] = limit_findings
    if observed_artifacts is not None:
        event["observed_artifacts"] = observed_artifacts
    if limit_artifacts is not None:
        event["limit_artifacts"] = limit_artifacts
    if observed_depth is not None:
        event["observed_depth"] = observed_depth
    if limit_depth is not None:
        event["limit_depth"] = limit_depth
    if observed_records is not None:
        event["observed_records"] = observed_records
    if limit_records is not None:
        event["limit_records"] = limit_records
    if observed_seconds is not None:
        event["observed_seconds"] = observed_seconds
    if limit_seconds is not None:
        event["limit_seconds"] = limit_seconds
    safe_parser_diagnostic = sanitize_parser_diagnostic(parser_diagnostic)
    if safe_parser_diagnostic is not None:
        event["parser_diagnostic"] = safe_parser_diagnostic
    return event


def analyzer_status_event(
    *,
    analyzer_id: str,
    status: str,
    planned_work: Iterable[PlannedWorkTarget] = (),
    reason: LedgerReason | None = None,
) -> AnalyzerStatusEvent:
    """Create a run-level analyzer status with normalized expected-work targets."""
    normalized_work: list[PlannedWorkTarget] = []
    for target in planned_work:
        start_line = target["start_line"]
        end_line = target["end_line"]
        _validate_range(start_line, end_line)
        normalized_work.append(
            {
                "work_id": target["work_id"],
                "path": _normalize_relative_path(target["path"]),
                "start_line": start_line,
                "end_line": end_line,
            }
        )

    event: AnalyzerStatusEvent = {
        "analyzer_id": analyzer_id,
        "status": status,
        "planned_work": normalized_work,
    }
    if reason is not None:
        event["reason_code"] = reason
        event["message"] = REASON_MESSAGES[reason]
    return event


def analyzer_status_for_events(
    analyzer_id: str, events: Iterable[InspectionLedgerEvent]
) -> AnalyzerStatusEvent:
    """Summarize analyzer work; scope-boundary records remain separate exclusions."""
    terminal_events = [
        event for event in events if event.get("record_type") != LedgerRecordType.SCOPE_BOUNDARY
    ]
    if not terminal_events:
        return analyzer_status_event(
            analyzer_id=analyzer_id,
            status="not_applicable",
            reason=LedgerReason.NO_APPLICABLE_FILES,
        )

    outcomes = {event["outcome"] for event in terminal_events}
    status = (
        "failed"
        if LedgerOutcome.FAILED in outcomes
        else "degraded"
        if LedgerOutcome.SKIPPED in outcomes or LedgerOutcome.PARTIAL in outcomes
        else "completed"
    )
    return analyzer_status_event(
        analyzer_id=analyzer_id,
        status=status,
        planned_work=[
            {
                "work_id": event["work_id"],
                "path": event["path"],
                "start_line": event["start_line"],
                "end_line": event["end_line"],
            }
            for event in terminal_events
        ],
    )


def _reason(value: object, fallback: LedgerReason) -> LedgerReason:
    """Return an allowlisted reason code without trusting untyped graph state."""
    try:
        return LedgerReason(str(value))
    except ValueError:
        return fallback


def _safe_path(path: object, components: list[str]) -> str:
    """Choose a report-safe path for a synthetic finalization exception."""
    if isinstance(path, str):
        try:
            return _normalize_relative_path(path)
        except ValueError:
            pass
    if components:
        return components[0]
    return "SKILL.md"


def _exception(
    *,
    outcome: LedgerOutcome,
    phase: str,
    reason: LedgerReason,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    error_class: str | None = None,
    analyzers: Iterable[str] = (),
    fatal: bool,
    scope: str | None = None,
) -> InspectionLedgerException:
    """Build the public, safe projection of one exceptional ledger fact."""
    exception: InspectionLedgerException = {
        "outcome": outcome,
        "phase": phase,
        "reason_code": reason,
        "message": REASON_MESSAGES[reason],
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "fatal": fatal,
    }
    analyzer_ids = sorted({analyzer for analyzer in analyzers if analyzer})
    if analyzer_ids:
        exception["analyzers"] = analyzer_ids
    if error_class:
        exception["error_class"] = error_class
    if scope:
        exception["scope"] = scope
    return exception


def _is_rule_set_record(event: Mapping[str, object]) -> bool:
    """Return whether a ledger row describes a rule set rather than an artifact."""
    return event.get("record_type") == LedgerRecordType.RULE_SET


def _exception_from_event(
    event: InspectionLedgerEvent, *, fatal: bool
) -> InspectionLedgerException:
    """Project a non-completed internal event without exposing internal IDs."""
    outcome = event["outcome"]
    fallback = (
        LedgerReason.UNACCOUNTED_WORK
        if outcome == LedgerOutcome.FAILED
        else LedgerReason.NO_APPLICABLE_FILES
    )
    exception = _exception(
        outcome=outcome,
        phase=str(event["phase"]),
        reason=_reason(event.get("reason_code"), fallback),
        path=str(event["path"]),
        start_line=event.get("start_line"),
        end_line=event.get("end_line"),
        error_class=event.get("error_class"),
        analyzers=[str(event.get("analyzer_id", ""))],
        fatal=fatal,
        scope=RULE_SET_SCOPE if _is_rule_set_record(event) else None,
    )
    exception.update(resource_measurements(event))
    parser_diagnostic = sanitize_parser_diagnostic(event.get("parser_diagnostic"))
    if parser_diagnostic is not None:
        exception["parser_diagnostic"] = parser_diagnostic
    source_identity = event.get("source_identity")
    if isinstance(source_identity, str) and source_identity:
        exception["source_identity"] = source_identity
    return exception


def _merge_exception_projection(
    exceptions: Iterable[InspectionLedgerException],
) -> list[InspectionLedgerException]:
    """Group duplicate public rows while retaining all contributing analyzers."""
    grouped: dict[tuple[object, ...], InspectionLedgerException] = {}
    for exception in exceptions:
        parser_diagnostic = sanitize_parser_diagnostic(exception.get("parser_diagnostic"))
        key = (
            exception["outcome"],
            exception["phase"],
            exception["reason_code"],
            exception["message"],
            exception["path"],
            exception["start_line"],
            exception["end_line"],
            exception.get("error_class"),
            # A rule-set row and a real file's row can share a path label;
            # they must never merge into one public row.
            exception.get("scope"),
            exception.get("source_identity"),
            tuple(resource_measurements(exception).items()),
            tuple(parser_diagnostic.items()) if parser_diagnostic else (),
        )
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = cast(InspectionLedgerException, dict(exception))
            continue
        existing["fatal"] = bool(existing.get("fatal")) or bool(exception.get("fatal"))
        analyzer_ids = set(existing.get("analyzers", [])) | set(exception.get("analyzers", []))
        if analyzer_ids:
            existing["analyzers"] = sorted(analyzer_ids)

    return sorted(
        grouped.values(),
        key=lambda item: (
            item["path"],
            item.get("start_line") or 0,
            item.get("end_line") or 0,
            str(item["phase"]),
            str(item["reason_code"]),
        ),
    )


def _legacy_effective_ids(
    findings: list[object],
    legacy_filtered: object,
) -> list[str]:
    """Map pre-ledger meta output back to canonical IDs during the transition.

    The stacked producer MR preserves IDs directly. This compatibility path only
    supports the older meta node, which copied findings before opaque IDs existed.
    """
    known_ids = {getattr(finding, "finding_id", "") for finding in findings}
    if not isinstance(legacy_filtered, list):
        return [str(getattr(finding, "finding_id", "")) for finding in findings]

    by_shape: dict[tuple[object, ...], list[str]] = {}
    for finding in findings:
        shape = (
            getattr(finding, "rule_id", None),
            getattr(finding, "file", None),
            getattr(finding, "start_line", None),
            getattr(finding, "end_line", None),
        )
        by_shape.setdefault(shape, []).append(str(getattr(finding, "finding_id", "")))

    selected: list[str] = []
    consumed: set[str] = set()
    for finding in legacy_filtered:
        finding_id = str(getattr(finding, "finding_id", ""))
        if finding_id in known_ids:
            selected.append(finding_id)
            consumed.add(finding_id)
            continue
        shape = (
            getattr(finding, "rule_id", None),
            getattr(finding, "file", None),
            getattr(finding, "start_line", None),
            getattr(finding, "end_line", None),
        )
        candidate = next((item for item in by_shape.get(shape, []) if item not in consumed), None)
        if candidate:
            selected.append(candidate)
            consumed.add(candidate)
    return selected


def finalize_ledger(state: Mapping[str, object]) -> tuple[AnalysisCompleteness, list[str]]:
    """Validate ledger accounting and derive the canonical public projection.

    Full internal rows remain in graph state. Reports receive only scope boundaries,
    skipped/failed work, analyzer summaries, and safe policy-derived fatality.
    """
    raw_components = state.get("components", [])
    components = (
        [_safe_path(component, []) for component in raw_components if isinstance(component, str)]
        if isinstance(raw_components, list)
        else []
    )
    components = list(dict.fromkeys(components))
    raw_findings = state.get("findings", [])
    findings = list(raw_findings) if isinstance(raw_findings, list) else []
    raw_events = state.get("inspection_ledger", [])
    events = (
        [cast(InspectionLedgerEvent, event) for event in raw_events if isinstance(event, dict)]
        if isinstance(raw_events, list)
        else []
    )
    ledger_output_limited = any(
        event.get("phase") == "ledger_output"
        and event.get("reason_code") == LedgerReason.OUTPUT_LIMIT
        for event in events
    )
    raw_statuses = state.get("analyzer_status_events", [])
    statuses = (
        [cast(AnalyzerStatusEvent, status) for status in raw_statuses if isinstance(status, dict)]
        if isinstance(raw_statuses, list)
        else []
    )

    findings_by_id: dict[str, object] = {}
    accounting_exceptions: list[InspectionLedgerException] = []

    def accounting_error(path: object = None) -> None:
        accounting_exceptions.append(
            _exception(
                outcome=LedgerOutcome.FAILED,
                phase="finalization",
                reason=LedgerReason.FINDING_ACCOUNTING_ERROR,
                path=_safe_path(path, components),
                fatal=True,
            )
        )

    for finding in findings:
        finding_id = str(getattr(finding, "finding_id", ""))
        if not finding_id or finding_id in findings_by_id:
            accounting_error(getattr(finding, "file", None))
            continue
        findings_by_id[finding_id] = finding

    events_by_work_id: dict[str, list[InspectionLedgerEvent]] = {}
    for event in events:
        events_by_work_id.setdefault(str(event.get("work_id", "")), []).append(event)

    producer_origins: dict[str, int] = {}
    producer_rows_present = False
    for event in events:
        outcome = event.get("outcome")
        phase = str(event.get("phase", ""))
        input_ids = list(event.get("input_finding_ids", []))
        emitted_ids = list(event.get("emitted_finding_ids", []))
        is_meta = _is_meta_phase(phase)
        is_producer = event.get("record_type") == LedgerRecordType.WORK_ITEM and not is_meta
        if is_producer:
            producer_rows_present = True
        if is_producer and input_ids:
            accounting_error(event.get("path"))
        if (
            is_producer
            and outcome not in (LedgerOutcome.COMPLETED, LedgerOutcome.PARTIAL)
            and emitted_ids
        ):
            accounting_error(event.get("path"))
        if (
            is_meta
            and outcome == LedgerOutcome.COMPLETED
            and not set(emitted_ids).issubset(input_ids)
        ):
            accounting_error(event.get("path"))
        if (
            is_meta
            and outcome in (LedgerOutcome.FAILED, LedgerOutcome.SKIPPED, LedgerOutcome.PARTIAL)
            and emitted_ids != input_ids
        ):
            accounting_error(event.get("path"))
        for finding_id in [*input_ids, *emitted_ids]:
            if finding_id not in findings_by_id:
                accounting_error(event.get("path"))
        if is_producer and outcome in (LedgerOutcome.COMPLETED, LedgerOutcome.PARTIAL):
            for finding_id in emitted_ids:
                producer_origins[finding_id] = producer_origins.get(finding_id, 0) + 1

    if producer_rows_present and not ledger_output_limited:
        for finding_id, finding in findings_by_id.items():
            if producer_origins.get(finding_id, 0) != 1:
                accounting_error(getattr(finding, "file", None))

    explicit_effective = state.get("effective_finding_ids")
    if isinstance(explicit_effective, list):
        effective_ids = [str(finding_id) for finding_id in explicit_effective]
    else:
        effective_ids = _legacy_effective_ids(findings, state.get("filtered_findings"))

    seen_effective: set[str] = set()
    validated_effective: list[str] = []
    for finding_id in effective_ids:
        if finding_id in seen_effective or finding_id not in findings_by_id:
            accounting_error(getattr(findings_by_id.get(finding_id), "file", None))
            continue
        seen_effective.add(finding_id)
        validated_effective.append(finding_id)

    # Deterministic analyzer findings are primary evidence. Meta analysis may
    # enrich or annotate those objects but cannot select them out of the public
    # machine-readable result.
    validated_effective = list(findings_by_id)[:MAX_EFFECTIVE_FINDINGS]

    unaccounted_exceptions: list[InspectionLedgerException] = []
    status_summaries: list[dict[str, object]] = []
    primary_targets: list[tuple[str, PlannedWorkTarget, list[InspectionLedgerEvent]]] = []
    for status in statuses:
        analyzer_id = str(status.get("analyzer_id", ""))
        planned_work = cast(list[PlannedWorkTarget], status.get("planned_work", []))
        outcome_counts = {
            "completed": 0,
            "partial": 0,
            "skipped": 0,
            "failed": 0,
            "unaccounted": 0,
        }
        for target in planned_work:
            work_id = str(target.get("work_id", ""))
            matches = events_by_work_id.get(work_id, [])
            if len(matches) == 0 and ledger_output_limited:
                outcome_counts["partial"] += 1
            elif len(matches) != 1:
                outcome_counts["unaccounted"] += 1
                unaccounted_exceptions.append(
                    _exception(
                        outcome=LedgerOutcome.FAILED,
                        phase="finalization",
                        reason=LedgerReason.UNACCOUNTED_WORK,
                        path=_safe_path(target.get("path"), components),
                        start_line=target.get("start_line"),
                        end_line=target.get("end_line"),
                        analyzers=[analyzer_id],
                        fatal=True,
                    )
                )
            else:
                outcome_name = str(matches[0].get("outcome", "failed"))
                outcome_counts[outcome_name if outcome_name in outcome_counts else "failed"] += 1
            if analyzer_id != "meta_analyzer":
                primary_targets.append((analyzer_id, target, matches))
        summary: dict[str, object] = {
            "analyzer_id": analyzer_id,
            "status": status.get("status", "unknown"),
            "planned_work": len(planned_work),
            **outcome_counts,
        }
        if status.get("reason_code") is not None:
            summary["reason_code"] = str(status["reason_code"])
        if status.get("message") is not None:
            summary["message"] = str(status["message"])
        if isinstance(status.get("source_identity"), str) and status["source_identity"]:
            summary["source_identity"] = status["source_identity"]
        status_summaries.append(summary)

    scope_rows = [
        _exception_from_event(event, fatal=False)
        for event in events
        if event.get("outcome") == LedgerOutcome.OUT_OF_SCOPE
    ]
    exceptional_rows = [
        _exception_from_event(event, fatal=event.get("outcome") == LedgerOutcome.FAILED)
        for event in events
        if event.get("outcome")
        in (LedgerOutcome.PARTIAL, LedgerOutcome.SKIPPED, LedgerOutcome.FAILED)
    ]
    exceptional_rows.extend(unaccounted_exceptions)
    exceptional_rows.extend(accounting_exceptions)
    raw_inventory = state.get("artifact_inventory", [])
    inventory = (
        [item for item in raw_inventory if isinstance(item, dict)]
        if isinstance(raw_inventory, list)
        else []
    )
    fatal_reasons = {
        (row["path"], row["reason_code"]) for row in exceptional_rows if row.get("fatal")
    }
    # The bounded detail ledger may omit a cache failure. Canonical inventory
    # still owns the artifact's disposition: truncation cannot restore success.
    for artifact in inventory:
        if artifact.get("disposition") != "failed":
            continue
        path = _safe_path(artifact.get("path"), components)
        reason = _reason(artifact.get("reason"), LedgerReason.READ_ERROR)
        if (path, reason) not in fatal_reasons:
            exceptional_rows.append(
                _exception(
                    outcome=LedgerOutcome.FAILED,
                    phase="cache",
                    reason=reason,
                    path=path,
                    fatal=True,
                )
            )
            fatal_reasons.add((path, reason))
    ledger_exceptions = _merge_exception_projection(exceptional_rows)
    scope_exclusions = _merge_exception_projection(scope_rows)

    # Rule-set rows describe an analyzer's configuration, not an artifact. Their
    # path is only a label, so folding them into per-component coverage would
    # charge the rule set's incompleteness to any real file with that name.
    rule_set_work_ids = {
        str(event.get("work_id", "")) for event in events if _is_rule_set_record(event)
    }
    per_component: dict[str, list[LedgerOutcome]] = {component: [] for component in components}
    if primary_targets:
        for _analyzer_id, target, matches in primary_targets:
            if str(target.get("work_id", "")) in rule_set_work_ids:
                continue
            path = _safe_path(target.get("path"), components)
            outcomes = per_component.setdefault(path, [])
            if len(matches) == 1:
                outcomes.append(matches[0]["outcome"])
            elif ledger_output_limited:
                outcomes.append(LedgerOutcome.PARTIAL)
            else:
                outcomes.append(LedgerOutcome.FAILED)
    else:
        cache_failures = {
            str(event.get("path"))
            for event in events
            if event.get("phase") == "cache" and event.get("outcome") == LedgerOutcome.FAILED
        }
        for component in components:
            per_component[component].append(
                LedgerOutcome.FAILED if component in cache_failures else LedgerOutcome.COMPLETED
            )

    disposition_by_path = {
        str(item.get("path", "")): str(item.get("disposition", "")) for item in inventory
    }
    raw_references_for_paths = state.get("artifact_references", [])
    referenced_paths = (
        {
            str(item.get("target_path"))
            for item in raw_references_for_paths
            if isinstance(item, dict)
            and item.get("status") == "resolved"
            and item.get("target_path")
        }
        if isinstance(raw_references_for_paths, list)
        else set()
    )
    coverage_components = list(
        dict.fromkeys(
            [
                *components,
                *(str(item.get("path", "")) for item in inventory if item.get("path")),
            ]
        )
    )
    relevant_components = [
        component
        for component in coverage_components
        if disposition_by_path.get(component) != "out_of_scope" or component in referenced_paths
    ]

    fully_inspected = 0
    partially_inspected = 0
    entirely_uninspected = 0
    for component in relevant_components:
        inventory_disposition = disposition_by_path.get(component)
        if inventory_disposition in {"failed", "out_of_scope"}:
            # Canonical inventory state is authoritative for content access. A
            # downstream analyzer can complete against an opaque sentinel, but
            # that must not make inaccessible bytes count as inspected. The
            # only out-of-scope artifacts retained above are resolved targets.
            entirely_uninspected += 1
            continue
        if inventory_disposition == "partial":
            # Components handed to analyzers contain at least a bounded prefix
            # or an explicit sentinel. Omitted inventory-only rows contain no
            # inspected bytes and therefore remain entirely uninspected.
            if component in per_component:
                partially_inspected += 1
            else:
                entirely_uninspected += 1
            continue
        outcomes = [
            outcome
            for outcome in per_component.get(component, [])
            if outcome != LedgerOutcome.OUT_OF_SCOPE
        ]
        if outcomes and all(outcome == LedgerOutcome.COMPLETED for outcome in outcomes):
            fully_inspected += 1
        elif any(
            outcome in (LedgerOutcome.COMPLETED, LedgerOutcome.PARTIAL) for outcome in outcomes
        ):
            partially_inspected += 1
        else:
            entirely_uninspected += 1

    total_components = len(relevant_components)
    coverage_percent = (
        round(fully_inspected / total_components * 100, 1) if total_components else 100.0
    )
    limitations: list[str] = []
    for status_summary in status_summaries:
        status_name = str(status_summary["status"])
        explicitly_optional = (
            state.get("use_llm") is False
            and status_name == "disabled"
            and str(status_summary["analyzer_id"])
            in {
                "meta_analyzer",
                "semantic_security_discovery",
                "semantic_developer_intent",
                "semantic_quality_policy",
            }
        )
        if explicitly_optional:
            continue
        if status_name not in {"completed", "not_applicable"}:
            message = status_summary.get("message")
            limitations.append(
                str(message)
                if message
                else f"Analyzer {status_summary['analyzer_id']} status: {status_name}."
            )
    execution_successful = not any(exception.get("fatal") for exception in ledger_exceptions)
    reference_caveat_only = (
        only_missing_reference_exceptions(ledger_exceptions)
        and not limitations
        and partially_inspected == 0
        and entirely_uninspected == 0
    )
    completeness_status = (
        "failed"
        if not execution_successful
        else "complete_with_caveats"
        if reference_caveat_only
        else "partial"
        if ledger_exceptions or limitations or partially_inspected or entirely_uninspected
        else "complete"
    )
    is_complete = completeness_status in {"complete", "complete_with_caveats"}

    raw_references = state.get("artifact_references", [])
    public_references = (
        [dict(item) for item in raw_references if isinstance(item, dict)]
        if isinstance(raw_references, list)
        else []
    )

    completeness: AnalysisCompleteness = {
        "total_components": total_components,
        "scanned_components": fully_inspected,
        "coverage_percent": coverage_percent,
        "is_complete": is_complete,
        "status": completeness_status,
        "execution_successful": execution_successful,
        "fully_inspected_files": fully_inspected,
        "partially_inspected_files": partially_inspected,
        "entirely_uninspected_files": entirely_uninspected,
        "ledger_exceptions": ledger_exceptions,
        "scope_exclusions": scope_exclusions,
        "analyzer_statuses": sorted(status_summaries, key=lambda item: str(item["analyzer_id"])),
        "references": public_references,
        "limitations": limitations,
        "findings_before_filtering": len(findings_by_id),
        "findings_after_filtering": len(validated_effective),
    }
    patterns = state.get("exclude_patterns", [])
    dependency_inspection = state.get("dependency_inspection")
    if isinstance(dependency_inspection, dict):
        completeness["dependency_inspection"] = dict(dependency_inspection)
    if patterns:
        completeness["exclude_patterns"] = list(patterns)
        completeness["excluded_file_count"] = sum(
            item.get("reason") == LedgerReason.USER_EXCLUSION for item in inventory
        )
    format_warnings = [
        {**row, "assessment_complete": False, "accepted_by_policy": True}
        for row in ledger_exceptions
        if row["reason_code"] == LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED
    ]
    if format_warnings:
        # Imports stay lazy: ordinary static-only ledger projection does not
        # need to load analyzer discovery or semantic-runtime accounting.
        from skillspector.nodes.analyzers import ANALYZER_MODULES
        from skillspector.semantic_runtime import format_recovery_runtime_accepted

        completeness["warnings"] = format_warnings
        completeness["policy_accepted"] = format_recovery_runtime_accepted(state, ANALYZER_MODULES)
        completeness["policy_accepted"] = format_recovery_policy_accepted(completeness)
        if not completeness["policy_accepted"]:
            for warning in format_warnings:
                warning["accepted_by_policy"] = False
    return completeness, validated_effective


def guard_analyzer_node(
    analyzer_id: str,
    node: Callable[[object], dict[str, object]],
) -> Callable[[object], dict[str, object]]:
    """Convert an unexpected analyzer exception into safe, terminal ledger facts."""

    def guarded(state: object) -> dict[str, object]:
        try:
            return node(state)
        except Exception as exc:  # pragma: no cover - exact exception is node-dependent
            logger.warning("Analyzer %s raised %s", analyzer_id, type(exc).__name__, exc_info=True)
            state_mapping = cast(Mapping[str, object], state)
            raw_components = state_mapping.get("components", [])
            components = (
                [str(component) for component in raw_components]
                if isinstance(raw_components, list)
                else []
            )
            events = [
                ledger_event(
                    outcome=LedgerOutcome.FAILED,
                    phase="analyzer",
                    analyzer_id=analyzer_id,
                    path=_safe_path(component, components),
                    reason=LedgerReason.ANALYZER_RUNTIME_ERROR,
                    error_class=type(exc).__name__,
                )
                for component in components
            ]
            planned_work: list[PlannedWorkTarget] = [
                cast(
                    PlannedWorkTarget,
                    {
                        "work_id": event["work_id"],
                        "path": event["path"],
                        "start_line": event["start_line"],
                        "end_line": event["end_line"],
                    },
                )
                for event in events
            ]
            return {
                "findings": [],
                "inspection_ledger": events,
                "analyzer_status_events": [
                    analyzer_status_event(
                        analyzer_id=analyzer_id,
                        status="failed",
                        planned_work=planned_work,
                        reason=LedgerReason.ANALYZER_RUNTIME_ERROR,
                    )
                ],
            }

    return guarded
