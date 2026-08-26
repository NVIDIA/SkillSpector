# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared semantic and resource contracts for dependency-source analysis."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import blake2s
from typing import Final

from skillspector.inspection_ledger import (
    MAX_FINDING_OUTPUT_RECORDS,
    MAX_INSPECTION_LEDGER_EVENTS,
)
from skillspector.models import Finding
from skillspector.url_redaction import redact_text, redact_url

MAX_DEPENDENCY_CONFIG_NODES: Final = 50_000
MAX_DEPENDENCY_RETAINED_LITERAL_BYTES: Final = 2_000_000
MAX_DEPENDENCY_SOURCE_RECORDS: Final = 50_000
MAX_DEPENDENCY_SOURCE_CHANGES: Final = 10_000
MAX_DEPENDENCY_FINDING_OUTPUT_RECORDS: Final = MAX_FINDING_OUTPUT_RECORDS
MAX_DEPENDENCY_LEDGER_EVENTS: Final = MAX_INSPECTION_LEDGER_EVENTS
MAX_DEPENDENCY_FILE_BYTES: Final = 1_000_000
MAX_DEPENDENCY_YAML_ALIASES: Final = 256
MAX_DEPENDENCY_CONFIG_DEPTH: Final = 64
MAX_DEPENDENCY_DESTINATION_CHARACTERS: Final = 16_384
MAX_DEPENDENCY_SHELL_UNITS_PER_FILE: Final = 256
MAX_DEPENDENCY_SHELL_PARSER_CALLS_PER_FILE: Final = 512
DEPENDENCY_SHELL_PARSED_BYTE_REVISIT_FACTOR: Final = 2
MAX_DEPENDENCY_SHELL_PARSED_BYTES: Final = 6_000_000
DEPENDENCY_SHELL_CST_VISIT_FACTOR: Final = 12
DEPENDENCY_SHELL_CST_VISIT_BASE: Final = 1_024
MAX_DEPENDENCY_SHELL_NESTED_LITERAL_DEPTH: Final = 2
MAX_DEPENDENCY_RETAINED_SHELL_IR: Final = 50_000
MAX_DEPENDENCY_SOURCE_MAP_ENTRIES_PER_FILE: Final = 50_000
MAX_DEPENDENCY_SHELL_VALUE_BYTES_PER_FILE: Final = 2_000_000
MAX_DEPENDENCY_SHELL_LOCALIZED_ISSUES: Final = 10_000


class DestinationStatus(StrEnum):
    """Whether a source destination is literal or conservatively unresolved."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


class DependencyCandidateRank(StrEnum):
    """Strength used when equivalent transient semantic sinks overlap."""

    EXACT = "exact"
    RECOVERED = "recovered"


class DependencyEcosystem(StrEnum):
    """Code-owned dependency ecosystems implemented by source parsers."""

    NPM = "npm"
    YARN = "yarn"
    PNPM = "pnpm"
    PIP = "pip"
    POETRY = "poetry"
    PDM = "pdm"
    UV = "uv"
    CARGO = "cargo"
    MAVEN = "maven"
    GRADLE = "gradle"
    NUGET = "nuget"
    RUBYGEMS = "rubygems"
    GO = "go"
    GENERIC = "generic"


class DependencySourceSurface(StrEnum):
    """Coarse code-owned surface where a dependency source was declared."""

    NPMRC = ".npmrc"
    PIP_CONFIG = "pip config"
    YARN_CONFIG = "yarn-config"
    PYTHON_PROJECT_CONFIG = "python-project-config"
    CARGO_CONFIG = "cargo-config"
    MAVEN_CONFIG = "maven-config"
    SOURCE = "source"
    REPOSITORY = "repository"
    MIRROR = "mirror"
    COMMAND = "command"
    INVOCATION = "invocation"
    ENVIRONMENT = "environment"
    GENERATED_CONFIG = "generated-config"


class DependencySourceOperation(StrEnum):
    """Code-owned semantic operation represented by a source change."""

    ADD = "add"
    REPLACE = "replace"
    REMOVE = "remove"
    SET = "set"
    USE = "use"


class DependencySourceScope(StrEnum):
    """Coarse code-owned scope category, never a raw package or section name."""

    GLOBAL = "global"
    SCOPED = "scoped"
    PROJECT = "project"
    SOURCE = "source"
    REGISTRY = "registry"
    MIRROR = "mirror"
    REPOSITORY = "repository"
    COMMAND = "command"
    INVOCATION = "invocation"
    ENVIRONMENT = "environment"
    GENERATED_CONFIG = "generated-config"


class DependencySourceLimitationReason(StrEnum):
    """Safe local reason codes mapped to ledger reasons only at integration time."""

    PARSE_INCOMPLETE = "dependency_source_parse_incomplete"
    UNSCANNED_EXECUTABLE_CONTENT = "unscanned_executable_content"


class DependencyWorkResource(StrEnum):
    """Code-owned names for every dependency-source resource counter."""

    CONFIG_NODES = "config_nodes"
    RETAINED_LITERAL_BYTES = "retained_literal_bytes"
    SOURCE_RECORDS = "source_records"
    EMITTED_CHANGES = "emitted_changes"
    FINDING_OUTPUT_RECORDS = "finding_output_records"
    LEDGER_EVENTS = "ledger_events"
    PHYSICAL_BYTES = "physical_bytes"
    YAML_ALIASES = "yaml_aliases"
    DEPTH = "depth"
    SHELL_UNITS = "shell_units"
    SHELL_PARSER_CALLS = "shell_parser_calls"
    SHELL_PARSED_REVISIT_BYTES = "shell_parsed_revisit_bytes"
    SHELL_PARSED_BYTES = "shell_parsed_bytes"
    SHELL_CST_VISITS = "shell_cst_visits"
    SHELL_NESTED_DEPTH = "shell_nested_depth"
    RETAINED_SHELL_IR = "retained_shell_ir"
    SHELL_SOURCE_MAP_ENTRIES = "shell_source_map_entries"
    SHELL_RETAINED_VALUE_BYTES = "shell_retained_value_bytes"
    SHELL_LOCALIZED_ISSUES = "shell_localized_issues"


class LedgerTruncationClaimStatus(StrEnum):
    """Outcome of claiming the scan's single reserved truncation row."""

    CLAIMED = "claimed"
    ALREADY_CLAIMED = "already_claimed"
    NO_CAPACITY = "no_capacity"


class ShellTruncationClaimStatus(StrEnum):
    """Outcome of claiming the scan's reserved localized shell-limit issue."""

    CLAIMED = "claimed"
    ALREADY_CLAIMED = "already_claimed"
    NO_CAPACITY = "no_capacity"


class ShellDialect(StrEnum):
    """Code-owned shell dialects accepted by the Bash grammar frontend."""

    BASH = "bash"
    SH = "sh"
    DASH = "dash"


class ShellUnitKind(StrEnum):
    """Code-owned origin shape for one independently parsed shell unit."""

    STANDALONE = "standalone"
    MARKDOWN_FENCE = "markdown_fence"
    NESTED_LITERAL = "nested_literal"
    GENERATED_CONFIG = "generated_config"


class SiteProvenance(StrEnum):
    """Code-owned provenance attached to transient shell sites."""

    FILE_SUFFIX = "file_suffix"
    SHEBANG = "shebang"
    EXECUTABLE_PATH = "executable_path"
    MARKDOWN_FENCE = "markdown_fence"
    NESTED_LITERAL = "nested_literal"
    GENERATED_CONFIG = "generated_config"


class StaticValueState(StrEnum):
    """Three-state conservative shell value classification."""

    EXACT = "exact"
    UNKNOWN = "unknown"
    UNBOUND = "unbound"


class CommandResolutionKind(StrEnum):
    """Whether a command word is proven external or function-resolved."""

    EXTERNAL = "external"
    FUNCTION = "function"
    AMBIGUOUS = "ambiguous"


class CommandProducerReachability(StrEnum):
    """Whether the command's enclosing producer is proven to execute."""

    ACTIVE = "active"
    INERT = "inert"
    AMBIGUOUS = "ambiguous"


class ShellIssueReason(StrEnum):
    """Content-free reasons for localized shell-analysis incompleteness."""

    SYNTAX_ERROR = "syntax_error"
    UNSUPPORTED_SEMANTICS = "unsupported_semantics"
    RUNTIME_LIMIT = "runtime_limit"
    SHELL_PARSER_UNAVAILABLE = "shell_parser_unavailable"
    RESOURCE_LIMIT = "resource_limit"


class ShellWorkOutcome(StrEnum):
    """Terminal shell work status retained for later ledger integration."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    FAILED = "failed"


def _require_nonnegative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _normalize_relative_posix_path(path: object) -> str:
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise ValueError("path must be a relative POSIX path")
    if path.startswith("/") or path.startswith("//"):
        raise ValueError("path must be a relative POSIX path")
    if len(path) >= 2 and path[1] == ":":
        raise ValueError("path must be a relative POSIX path")
    parts = path.split("/")
    if any(part == ".." for part in parts):
        raise ValueError("path must not contain parent traversal")
    normalized = "/".join(part for part in parts if part not in {"", "."})
    if not normalized:
        raise ValueError("path must identify a file")
    return normalized


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """A source range using canonical UTF-8 byte and one-based line coordinates."""

    path: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    start_column: int | None = field(default=None, kw_only=True)
    end_column: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_relative_posix_path(self.path))
        start_byte = _require_nonnegative_integer(self.start_byte, "start_byte")
        end_byte = _require_nonnegative_integer(self.end_byte, "end_byte")
        start_line = _require_nonnegative_integer(self.start_line, "start_line")
        end_line = _require_nonnegative_integer(self.end_line, "end_line")
        if end_byte < start_byte:
            raise ValueError("byte range must be zero-based and half-open")
        if start_line < 1 or end_line < start_line:
            raise ValueError("line range must be positive and inclusive")
        if (self.start_column is None) != (self.end_column is None):
            raise ValueError("byte columns must be supplied as a pair")
        if self.start_column is not None and self.end_column is not None:
            start_column = _require_nonnegative_integer(self.start_column, "start_column")
            end_column = _require_nonnegative_integer(self.end_column, "end_column")
            if start_line == end_line and end_column < start_column:
                raise ValueError("column range must be zero-based and half-open")


@dataclass(frozen=True, slots=True)
class SourceMapEntry:
    """One affine child-byte to canonical physical-byte interval."""

    child_start_byte: int
    child_end_byte: int
    physical_start_byte: int
    physical_end_byte: int

    def __post_init__(self) -> None:
        child_start = _require_nonnegative_integer(self.child_start_byte, "child_start_byte")
        child_end = _require_nonnegative_integer(self.child_end_byte, "child_end_byte")
        physical_start = _require_nonnegative_integer(
            self.physical_start_byte, "physical_start_byte"
        )
        physical_end = _require_nonnegative_integer(self.physical_end_byte, "physical_end_byte")
        if child_end <= child_start or physical_end <= physical_start:
            raise ValueError("source-map intervals must be non-empty and half-open")
        if child_end - child_start != physical_end - physical_start:
            raise ValueError("source-map entries must be affine")


@dataclass(frozen=True, slots=True)
class SourceMap:
    """Compact ordered mapping from transient unit bytes to physical bytes."""

    path: str
    entries: tuple[SourceMapEntry, ...]
    child_size_bytes: int
    physical_size_bytes: int
    physical_line_starts: tuple[int, ...] = field(repr=False)
    _child_starts: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_relative_posix_path(self.path))
        entries = tuple(self.entries)
        child_size = _require_nonnegative_integer(self.child_size_bytes, "child_size_bytes")
        physical_size = _require_nonnegative_integer(
            self.physical_size_bytes, "physical_size_bytes"
        )
        if len(entries) > MAX_DEPENDENCY_SOURCE_MAP_ENTRIES_PER_FILE:
            raise ValueError("source map exceeds the entry limit")
        if not all(isinstance(entry, SourceMapEntry) for entry in entries):
            raise ValueError("entries must contain SourceMapEntry values")
        previous_child_end = -1
        previous_physical_end = -1
        for entry in entries:
            if entry.child_end_byte > child_size or entry.physical_end_byte > physical_size:
                raise ValueError("source-map interval exceeds a canonical byte domain")
            if entry.child_start_byte < previous_child_end:
                raise ValueError("source-map child intervals must be ordered and nonoverlapping")
            if entry.physical_start_byte < previous_physical_end:
                raise ValueError("source-map physical intervals must be ordered and nonoverlapping")
            previous_child_end = entry.child_end_byte
            previous_physical_end = entry.physical_end_byte
        line_starts = tuple(self.physical_line_starts)
        if not line_starts or line_starts[0] != 0:
            raise ValueError("physical line starts must begin at byte zero")
        if any(type(value) is not int or value < 0 for value in line_starts) or any(
            right <= left for left, right in zip(line_starts, line_starts[1:], strict=False)
        ):
            raise ValueError("physical line starts must be strictly ordered byte offsets")
        if line_starts[-1] > physical_size:
            raise ValueError("physical line starts exceed canonical physical bytes")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "physical_line_starts", line_starts)
        object.__setattr__(
            self,
            "_child_starts",
            tuple(entry.child_start_byte for entry in entries),
        )

    def _physical_position(self, byte_offset: int) -> tuple[int, int]:
        line_index = bisect_right(self.physical_line_starts, byte_offset) - 1
        return line_index + 1, byte_offset - self.physical_line_starts[line_index]

    def map_range(self, start_byte: int, end_byte: int) -> SourceSpan | None:
        """Map one exact child range, returning None across any gap or discontinuity."""
        start = _require_nonnegative_integer(start_byte, "start_byte")
        end = _require_nonnegative_integer(end_byte, "end_byte")
        if end <= start:
            return None
        cursor = start
        mapped_start: int | None = None
        mapped_end: int | None = None
        entry_index = max(0, bisect_right(self._child_starts, start) - 1)
        while entry_index < len(self.entries):
            entry = self.entries[entry_index]
            if entry.child_end_byte <= cursor:
                entry_index += 1
                continue
            if entry.child_start_byte > cursor:
                return None
            segment_end = min(end, entry.child_end_byte)
            physical_start = entry.physical_start_byte + (cursor - entry.child_start_byte)
            physical_end = physical_start + (segment_end - cursor)
            if mapped_end is not None and mapped_end != physical_start:
                return None
            if mapped_start is None:
                mapped_start = physical_start
            mapped_end = physical_end
            cursor = segment_end
            if cursor == end:
                break
            entry_index += 1
        if cursor != end or mapped_start is None or mapped_end is None:
            return None
        start_line, start_column = self._physical_position(mapped_start)
        end_line_index = bisect_right(self.physical_line_starts, mapped_end - 1) - 1
        end_line = end_line_index + 1
        end_column = mapped_end - self.physical_line_starts[end_line_index]
        return SourceSpan(
            self.path,
            mapped_start,
            mapped_end,
            start_line,
            end_line,
            start_column=start_column,
            end_column=end_column,
        )

    def compose(self, parent: SourceMap) -> SourceMap:
        """Compose child-to-parent and parent-to-physical intervals exactly."""
        if not isinstance(parent, SourceMap) or parent.path != self.path:
            raise ValueError("source maps must share one canonical path")
        if self.physical_size_bytes != parent.child_size_bytes:
            raise ValueError("source-map intermediate byte domains must match")
        composed: list[SourceMapEntry] = []
        parent_index = 0
        for entry in self.entries:
            parent_cursor = entry.physical_start_byte
            while parent_cursor < entry.physical_end_byte:
                while (
                    parent_index < len(parent.entries)
                    and parent.entries[parent_index].child_end_byte <= parent_cursor
                ):
                    parent_index += 1
                if parent_index >= len(parent.entries):
                    raise ValueError("source-map composition encountered an unmapped range")
                parent_entry = parent.entries[parent_index]
                if parent_entry.child_start_byte > parent_cursor:
                    raise ValueError("source-map composition encountered an unmapped range")
                parent_end = min(entry.physical_end_byte, parent_entry.child_end_byte)
                length = parent_end - parent_cursor
                child_start = entry.child_start_byte + (parent_cursor - entry.physical_start_byte)
                physical_start = parent_entry.physical_start_byte + (
                    parent_cursor - parent_entry.child_start_byte
                )
                candidate = SourceMapEntry(
                    child_start,
                    child_start + length,
                    physical_start,
                    physical_start + length,
                )
                if (
                    composed
                    and composed[-1].child_end_byte == candidate.child_start_byte
                    and composed[-1].physical_end_byte == candidate.physical_start_byte
                ):
                    previous = composed[-1]
                    composed[-1] = SourceMapEntry(
                        previous.child_start_byte,
                        candidate.child_end_byte,
                        previous.physical_start_byte,
                        candidate.physical_end_byte,
                    )
                else:
                    composed.append(candidate)
                if len(composed) > MAX_DEPENDENCY_SOURCE_MAP_ENTRIES_PER_FILE:
                    raise ValueError("source-map composition exceeds the entry limit")
                parent_cursor = parent_end
        return SourceMap(
            path=parent.path,
            entries=tuple(composed),
            child_size_bytes=self.child_size_bytes,
            physical_size_bytes=parent.physical_size_bytes,
            physical_line_starts=parent.physical_line_starts,
        )


def _require_shell_unit_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("unit_id must be an internally generated opaque identifier")
    return value


@dataclass(frozen=True, slots=True)
class ShellUnit:
    """Immutable unit-local canonical bytes plus physical origin metadata."""

    dialect: ShellDialect
    kind: ShellUnitKind
    provenance: SiteProvenance
    raw_bytes: bytes = field(repr=False)
    origin_span: SourceSpan
    source_map: SourceMap | None = None
    unit_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.dialect, ShellDialect):
            raise ValueError("dialect must be a code-owned ShellDialect")
        if not isinstance(self.kind, ShellUnitKind):
            raise ValueError("kind must be a code-owned ShellUnitKind")
        if not isinstance(self.provenance, SiteProvenance):
            raise ValueError("provenance must be code-owned")
        if type(self.raw_bytes) is not bytes:
            raise ValueError("raw_bytes must be canonical immutable bytes")
        if not isinstance(self.origin_span, SourceSpan):
            raise ValueError("origin_span must be a SourceSpan")
        if self.source_map is not None:
            if not isinstance(self.source_map, SourceMap):
                raise ValueError("source_map must be a SourceMap")
            if self.source_map.path != self.origin_span.path:
                raise ValueError("source_map and origin_span paths must match")
            if self.source_map.child_size_bytes != len(self.raw_bytes):
                raise ValueError("source_map child size must match unit bytes")
            if self.source_map.physical_size_bytes < self.origin_span.end_byte:
                raise ValueError("source_map physical size must cover the unit origin")
            if self.source_map.entries and (
                self.source_map.entries[0].child_start_byte != 0
                or self.source_map.entries[-1].child_end_byte != len(self.raw_bytes)
            ):
                raise ValueError("source_map must cover the unit byte boundaries")
            if any(
                entry.physical_start_byte < self.origin_span.start_byte
                or entry.physical_end_byte > self.origin_span.end_byte
                for entry in self.source_map.entries
            ):
                raise ValueError("source_map intervals must stay inside the unit origin")
        elif self.origin_span.end_byte - self.origin_span.start_byte != len(self.raw_bytes):
            raise ValueError("an identity-mapped unit must match its physical byte span")
        coordinate_key = (
            f"{self.origin_span.path}\0{self.origin_span.start_byte}\0"
            f"{self.origin_span.end_byte}\0{self.kind.value}\0{self.provenance.value}"
        ).encode()
        object.__setattr__(
            self,
            "unit_id",
            blake2s(coordinate_key, digest_size=16, person=b"SC10UNIT").hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class StaticValue:
    """Exact bytes or one of two content-free conservative states."""

    state: StaticValueState
    exact_bytes: bytes | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.state, StaticValueState):
            raise ValueError("state must be code-owned")
        if self.state is StaticValueState.EXACT:
            if type(self.exact_bytes) is not bytes:
                raise ValueError("exact values require immutable bytes")
        elif self.exact_bytes is not None:
            raise ValueError("unknown and unbound values cannot retain bytes")

    @classmethod
    def exact(cls, value: bytes) -> StaticValue:
        return cls(StaticValueState.EXACT, value)

    @classmethod
    def unknown(cls) -> StaticValue:
        return cls(StaticValueState.UNKNOWN)

    @classmethod
    def unbound(cls) -> StaticValue:
        return cls(StaticValueState.UNBOUND)


def _validate_site(unit_id: object, provenance: object, span: object) -> None:
    _require_shell_unit_id(unit_id)
    if not isinstance(provenance, SiteProvenance):
        raise ValueError("provenance must be code-owned")
    if not isinstance(span, SourceSpan):
        raise ValueError("span must be a SourceSpan")


@dataclass(frozen=True, slots=True)
class CommandSite:
    """Policy-free physical command descriptor consumed by later adapters."""

    unit_id: str
    provenance: SiteProvenance
    span: SourceSpan
    argv: tuple[StaticValue, ...]
    argument_spans: tuple[SourceSpan, ...] = ()
    resolution: CommandResolutionKind = CommandResolutionKind.EXTERNAL
    producer: CommandProducerReachability = CommandProducerReachability.ACTIVE
    prefix_assignments: tuple[AssignmentSite, ...] = ()
    exported_assignments: tuple[AssignmentSite, ...] = ()

    def __post_init__(self) -> None:
        _validate_site(self.unit_id, self.provenance, self.span)
        argv = tuple(self.argv)
        if not argv or not all(isinstance(value, StaticValue) for value in argv):
            raise ValueError("argv must contain at least one StaticValue")
        object.__setattr__(self, "argv", argv)
        argument_spans = tuple(self.argument_spans) or tuple(self.span for _ in argv)
        if len(argument_spans) != len(argv) or not all(
            isinstance(argument_span, SourceSpan) for argument_span in argument_spans
        ):
            raise ValueError("argument_spans must align one-to-one with argv")
        if any(argument_span.path != self.span.path for argument_span in argument_spans):
            raise ValueError("argument_spans must share the command path")
        if not isinstance(self.resolution, CommandResolutionKind):
            raise ValueError("resolution must be code-owned")
        if not isinstance(self.producer, CommandProducerReachability):
            raise ValueError("producer must be code-owned")
        prefix_assignments = tuple(self.prefix_assignments)
        exported_assignments = tuple(self.exported_assignments)
        if not all(isinstance(site, AssignmentSite) for site in prefix_assignments):
            raise ValueError("prefix_assignments must contain AssignmentSite values")
        if not all(isinstance(site, AssignmentSite) for site in exported_assignments):
            raise ValueError("exported_assignments must contain AssignmentSite values")
        if any(
            site.unit_id != self.unit_id or site.span.path != self.span.path
            for site in (*prefix_assignments, *exported_assignments)
        ):
            raise ValueError("command assignments must share the command unit and path")
        object.__setattr__(self, "argument_spans", argument_spans)
        object.__setattr__(self, "prefix_assignments", prefix_assignments)
        object.__setattr__(self, "exported_assignments", exported_assignments)


@dataclass(frozen=True, slots=True)
class AssignmentSite:
    """One bounded shell assignment with no environment evaluation."""

    unit_id: str
    provenance: SiteProvenance
    span: SourceSpan
    name: str = field(repr=False)
    value: StaticValue = field(repr=False)

    def __post_init__(self) -> None:
        _validate_site(self.unit_id, self.provenance, self.span)
        if (
            not isinstance(self.name, str)
            or not self.name
            or not (self.name[0].isascii() and (self.name[0].isalpha() or self.name[0] == "_"))
            or any(
                not (character.isascii() and (character.isalnum() or character == "_"))
                for character in self.name[1:]
            )
        ):
            raise ValueError("name must be a shell identifier")
        if not isinstance(self.value, StaticValue):
            raise ValueError("value must be a StaticValue")


@dataclass(frozen=True, slots=True)
class GeneratedConfig:
    """Transient, policy-free generated configuration descriptor."""

    unit_id: str
    provenance: SiteProvenance
    span: SourceSpan
    target: StaticValue = field(repr=False)
    content: StaticValue = field(repr=False)
    source_map: SourceMap | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _validate_site(self.unit_id, self.provenance, self.span)
        if not isinstance(self.target, StaticValue) or not isinstance(self.content, StaticValue):
            raise ValueError("generated config values must be StaticValue values")
        if self.source_map is not None and not isinstance(self.source_map, SourceMap):
            raise ValueError("source_map must be a SourceMap")


@dataclass(frozen=True, slots=True)
class ShellIssue:
    """Sanitized localized shell issue with terminal outcome metadata."""

    reason: ShellIssueReason
    outcome: ShellWorkOutcome
    span: SourceSpan
    unit_id: str | None = None
    exhaustion: DependencyWorkExhaustion | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, ShellIssueReason):
            raise ValueError("reason must be a code-owned ShellIssueReason")
        if not isinstance(self.outcome, ShellWorkOutcome):
            raise ValueError("outcome must be a code-owned ShellWorkOutcome")
        if not isinstance(self.span, SourceSpan):
            raise ValueError("span must be a SourceSpan")
        if self.unit_id is not None:
            _require_shell_unit_id(self.unit_id)
        if self.reason is ShellIssueReason.RESOURCE_LIMIT:
            if not isinstance(self.exhaustion, DependencyWorkExhaustion):
                raise ValueError("resource-limit issues require typed exhaustion")
        elif self.exhaustion is not None:
            raise ValueError("only resource-limit issues carry exhaustion")
        if (
            self.reason is ShellIssueReason.RUNTIME_LIMIT
            and self.outcome is not ShellWorkOutcome.PARTIAL
        ):
            raise ValueError("runtime limits are localized partial outcomes")
        if (
            self.reason
            in {
                ShellIssueReason.SYNTAX_ERROR,
                ShellIssueReason.UNSUPPORTED_SEMANTICS,
                ShellIssueReason.RESOURCE_LIMIT,
            }
            and self.outcome is not ShellWorkOutcome.PARTIAL
        ):
            raise ValueError("localized shell issues are partial outcomes")
        if self.reason is ShellIssueReason.SHELL_PARSER_UNAVAILABLE and self.outcome not in {
            ShellWorkOutcome.PARTIAL,
            ShellWorkOutcome.FAILED,
        }:
            raise ValueError("parser unavailability must be partial or failed")


@dataclass(frozen=True, slots=True)
class ShellWorkItem:
    """Sanitized terminal shell work record safe for public analysis state."""

    unit_id: str
    dialect: ShellDialect
    kind: ShellUnitKind
    provenance: SiteProvenance
    span: SourceSpan
    outcome: ShellWorkOutcome

    def __post_init__(self) -> None:
        _require_shell_unit_id(self.unit_id)
        if not isinstance(self.dialect, ShellDialect):
            raise ValueError("dialect must be code-owned")
        if not isinstance(self.kind, ShellUnitKind):
            raise ValueError("kind must be code-owned")
        if not isinstance(self.provenance, SiteProvenance):
            raise ValueError("provenance must be code-owned")
        if not isinstance(self.span, SourceSpan):
            raise ValueError("span must be a SourceSpan")
        if not isinstance(self.outcome, ShellWorkOutcome):
            raise ValueError("outcome must be code-owned")


@dataclass(frozen=True, slots=True)
class ShellExtractionResult:
    """Transient bounded unit extraction plus sanitized localized issues."""

    units: tuple[ShellUnit, ...] = ()
    issues: tuple[ShellIssue, ...] = ()

    def __post_init__(self) -> None:
        units = tuple(self.units)
        issues = tuple(self.issues)
        if not all(isinstance(unit, ShellUnit) for unit in units):
            raise ValueError("units must contain ShellUnit values")
        if not all(isinstance(issue, ShellIssue) for issue in issues):
            raise ValueError("issues must contain ShellIssue values")
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "issues", issues)


@dataclass(frozen=True, slots=True)
class ShellFrontendResult:
    """Typed syntax-only frontend output; no manager policy or findings."""

    commands: tuple[CommandSite, ...] = ()
    assignments: tuple[AssignmentSite, ...] = ()
    generated_configs: tuple[GeneratedConfig, ...] = ()
    issues: tuple[ShellIssue, ...] = ()
    work_items: tuple[ShellWorkItem, ...] = ()

    def __post_init__(self) -> None:
        normalized: tuple[tuple[str, type[object]], ...] = (
            ("commands", CommandSite),
            ("assignments", AssignmentSite),
            ("generated_configs", GeneratedConfig),
            ("issues", ShellIssue),
            ("work_items", ShellWorkItem),
        )
        for name, value_type in normalized:
            values = tuple(getattr(self, name))
            if not all(isinstance(value, value_type) for value in values):
                raise ValueError(f"{name} contains an invalid value")
            object.__setattr__(self, name, values)


@dataclass(frozen=True, slots=True)
class SourceChange:
    """One sanitized, command-independent dependency-source semantic change."""

    ecosystem: DependencyEcosystem
    surface: DependencySourceSurface
    operation: DependencySourceOperation
    scope: DependencySourceScope
    destination: str
    destination_status: DestinationStatus
    span: SourceSpan

    def __post_init__(self) -> None:
        for name, enum_type in (
            ("ecosystem", DependencyEcosystem),
            ("surface", DependencySourceSurface),
            ("operation", DependencySourceOperation),
            ("scope", DependencySourceScope),
        ):
            try:
                normalized = enum_type(getattr(self, name))
            except (TypeError, ValueError):
                raise ValueError(f"{name} is not a code-owned semantic") from None
            object.__setattr__(self, name, normalized)
        try:
            status = DestinationStatus(self.destination_status)
        except (TypeError, ValueError):
            raise ValueError("destination_status is invalid") from None
        object.__setattr__(self, "destination_status", status)
        if not isinstance(self.span, SourceSpan):
            raise ValueError("span must be a SourceSpan")
        if status is DestinationStatus.UNRESOLVED:
            if self.destination != "unresolved":
                raise ValueError("an unresolved destination must use the canonical placeholder")
            return
        if (
            not isinstance(self.destination, str)
            or not self.destination.strip()
            or len(self.destination) > MAX_DEPENDENCY_DESTINATION_CHARACTERS
            or any(ord(character) < 32 or ord(character) == 127 for character in self.destination)
            or self.destination == "unresolved"
            or redact_url(self.destination) != self.destination
            or redact_text(self.destination) != self.destination
        ):
            raise ValueError("a resolved destination must already be safely redacted")


@dataclass(frozen=True, slots=True)
class DependencySourceCandidate:
    """One sanitized unreserved semantic candidate awaiting orchestration."""

    ecosystem: DependencyEcosystem
    surface: DependencySourceSurface
    operation: DependencySourceOperation
    scope: DependencySourceScope
    destination: str
    destination_status: DestinationStatus
    span: SourceSpan
    producer_unit_id: str | None = None
    rank: DependencyCandidateRank = DependencyCandidateRank.EXACT
    canonical_default: bool = False

    def __post_init__(self) -> None:
        validated = SourceChange(
            ecosystem=self.ecosystem,
            surface=self.surface,
            operation=self.operation,
            scope=self.scope,
            destination=self.destination,
            destination_status=self.destination_status,
            span=self.span,
        )
        for name in (
            "ecosystem",
            "surface",
            "operation",
            "scope",
            "destination_status",
        ):
            object.__setattr__(self, name, getattr(validated, name))
        if self.producer_unit_id is not None:
            _require_shell_unit_id(self.producer_unit_id)
        try:
            rank = DependencyCandidateRank(self.rank)
        except (TypeError, ValueError):
            raise ValueError("rank must be code-owned") from None
        object.__setattr__(self, "rank", rank)
        if type(self.canonical_default) is not bool:
            raise ValueError("canonical_default must be a boolean")


_METRIC_FIELDS: Final = (
    "observed_bytes",
    "limit_bytes",
    "observed_findings",
    "limit_findings",
    "observed_depth",
    "limit_depth",
    "observed_records",
    "limit_records",
)


@dataclass(frozen=True, slots=True)
class DependencySourceLimitation:
    """Localized, content-free incomplete-analysis evidence for ledger integration."""

    reason: DependencySourceLimitationReason
    path: str
    start_line: int
    end_line: int
    observed_bytes: int | None = None
    limit_bytes: int | None = None
    observed_findings: int | None = None
    limit_findings: int | None = None
    observed_depth: int | None = None
    limit_depth: int | None = None
    observed_records: int | None = None
    limit_records: int | None = None

    def __post_init__(self) -> None:
        try:
            reason = DependencySourceLimitationReason(self.reason)
        except (TypeError, ValueError):
            raise ValueError("limitation reason is invalid") from None
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "path", _normalize_relative_posix_path(self.path))
        start_line = _require_nonnegative_integer(self.start_line, "start_line")
        end_line = _require_nonnegative_integer(self.end_line, "end_line")
        if start_line < 1 or end_line < start_line:
            raise ValueError("limitation line range must be positive and inclusive")
        for field_name in _METRIC_FIELDS:
            value = getattr(self, field_name)
            if value is not None:
                _require_nonnegative_integer(value, field_name)
        for observed_name, limit_name in (
            ("observed_bytes", "limit_bytes"),
            ("observed_findings", "limit_findings"),
            ("observed_depth", "limit_depth"),
            ("observed_records", "limit_records"),
        ):
            if (getattr(self, observed_name) is None) != (getattr(self, limit_name) is None):
                raise ValueError("limitation metrics must use observed/limit pairs")

    def ledger_metrics(self) -> dict[str, int]:
        """Return only ledger-compatible numeric fields that are present."""
        return {
            field_name: value
            for field_name in _METRIC_FIELDS
            if (value := getattr(self, field_name)) is not None
        }


@dataclass(frozen=True, slots=True)
class DependencySourceSpan:
    """Sanitized whole-file or localized line range for integration accounting."""

    path: str
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_relative_posix_path(self.path))
        start_line = _require_nonnegative_integer(self.start_line, "start_line")
        end_line = _require_nonnegative_integer(self.end_line, "end_line")
        if start_line < 1 or end_line < start_line:
            raise ValueError("span line range must be positive and inclusive")


@dataclass(frozen=True, slots=True)
class DependencySourceParseResult:
    """Sanitized parser or adapter output."""

    candidates: tuple[DependencySourceCandidate, ...] = ()
    limitations: tuple[DependencySourceLimitation, ...] = ()

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        limitations = tuple(self.limitations)
        if not all(isinstance(candidate, DependencySourceCandidate) for candidate in candidates):
            raise ValueError("candidates must contain DependencySourceCandidate values")
        if not all(isinstance(item, DependencySourceLimitation) for item in limitations):
            raise ValueError("limitations must contain DependencySourceLimitation values")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "limitations", limitations)


@dataclass(frozen=True, slots=True)
class DependencySourceAnalysis:
    """Public deterministic findings plus any localized analysis limitations."""

    findings: tuple[Finding, ...] = ()
    finding_producer_unit_ids: tuple[str | None, ...] = ()
    limitations: tuple[DependencySourceLimitation, ...] = ()
    applicable_spans: tuple[DependencySourceSpan, ...] = ()
    inspected_spans: tuple[DependencySourceSpan, ...] = ()
    shell_work_items: tuple[ShellWorkItem, ...] = ()
    shell_issues: tuple[ShellIssue, ...] = ()
    ledger_exhaustion: DependencyWorkExhaustion | None = None

    def __post_init__(self) -> None:
        findings = tuple(self.findings)
        producer_unit_ids = tuple(self.finding_producer_unit_ids) or tuple(
            None for _finding in findings
        )
        limitations = tuple(self.limitations)
        applicable_spans = tuple(self.applicable_spans)
        inspected_spans = tuple(self.inspected_spans)
        shell_work_items = tuple(self.shell_work_items)
        shell_issues = tuple(self.shell_issues)
        if not all(isinstance(finding, Finding) for finding in findings):
            raise ValueError("findings must contain Finding values")
        if len(producer_unit_ids) != len(findings):
            raise ValueError("finding producer identities must align one-to-one with findings")
        for producer_unit_id in producer_unit_ids:
            if producer_unit_id is not None:
                _require_shell_unit_id(producer_unit_id)
        if not all(isinstance(item, DependencySourceLimitation) for item in limitations):
            raise ValueError("limitations must contain DependencySourceLimitation values")
        if not all(isinstance(item, DependencySourceSpan) for item in applicable_spans):
            raise ValueError("applicable_spans must contain DependencySourceSpan values")
        if not all(isinstance(item, DependencySourceSpan) for item in inspected_spans):
            raise ValueError("inspected_spans must contain DependencySourceSpan values")
        if not all(isinstance(item, ShellWorkItem) for item in shell_work_items):
            raise ValueError("shell_work_items must contain ShellWorkItem values")
        if not all(isinstance(item, ShellIssue) for item in shell_issues):
            raise ValueError("shell_issues must contain ShellIssue values")
        if self.ledger_exhaustion is not None and not isinstance(
            self.ledger_exhaustion, DependencyWorkExhaustion
        ):
            raise ValueError("ledger_exhaustion must be DependencyWorkExhaustion")
        object.__setattr__(self, "findings", findings)
        object.__setattr__(self, "finding_producer_unit_ids", producer_unit_ids)
        object.__setattr__(self, "limitations", limitations)
        object.__setattr__(self, "applicable_spans", applicable_spans)
        object.__setattr__(self, "inspected_spans", inspected_spans)
        object.__setattr__(self, "shell_work_items", shell_work_items)
        object.__setattr__(self, "shell_issues", shell_issues)


def finding_from_source_change(change: SourceChange) -> Finding:
    """Convert one sanitized semantic change at the sole public finding boundary."""
    evidence: dict[str, object] = {
        "ecosystem": change.ecosystem.value,
        "surface": change.surface.value,
        "operation": change.operation.value,
        "scope": change.scope.value,
        "destination": change.destination,
        "destination_status": change.destination_status.value,
    }
    return Finding(
        rule_id="SC10",
        message="Dependency source redirects away from its canonical default",
        severity="HIGH",
        confidence=1.0,
        file=change.span.path,
        start_line=change.span.start_line,
        end_line=change.span.end_line,
        category="supply-chain",
        finding=f"{change.operation.value} source: {change.destination}",
        remediation="Review the configured dependency source before installing dependencies.",
        tags=["dependency-source", change.ecosystem.value],
        matched_text=change.destination,
        evidence=evidence,
    )


@dataclass(frozen=True, slots=True)
class DependencyWorkExhaustion:
    """Content-free typed evidence that one resource charge could not be reserved."""

    resource: DependencyWorkResource
    observed: int
    limit: int

    def __post_init__(self) -> None:
        try:
            resource = DependencyWorkResource(self.resource)
        except (TypeError, ValueError):
            raise ValueError("dependency work resource is invalid") from None
        object.__setattr__(self, "resource", resource)
        _require_nonnegative_integer(self.observed, "observed")
        _require_nonnegative_integer(self.limit, "limit")
        if self.observed <= self.limit:
            raise ValueError("resource exhaustion requires an observation above its limit")

    def ledger_metrics(self) -> dict[str, int]:
        """Project the resource count into compatible inspection-ledger metrics."""
        if self.resource in {
            DependencyWorkResource.PHYSICAL_BYTES,
            DependencyWorkResource.RETAINED_LITERAL_BYTES,
            DependencyWorkResource.SHELL_PARSED_REVISIT_BYTES,
            DependencyWorkResource.SHELL_PARSED_BYTES,
            DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES,
        }:
            prefix = "bytes"
        elif self.resource in {
            DependencyWorkResource.EMITTED_CHANGES,
            DependencyWorkResource.FINDING_OUTPUT_RECORDS,
        }:
            prefix = "findings"
        elif self.resource in {
            DependencyWorkResource.DEPTH,
            DependencyWorkResource.SHELL_NESTED_DEPTH,
        }:
            prefix = "depth"
        else:
            prefix = "records"
        return {f"observed_{prefix}": self.observed, f"limit_{prefix}": self.limit}


_SCAN_LIMITS: Final[dict[DependencyWorkResource, int]] = {
    DependencyWorkResource.CONFIG_NODES: MAX_DEPENDENCY_CONFIG_NODES,
    DependencyWorkResource.RETAINED_LITERAL_BYTES: MAX_DEPENDENCY_RETAINED_LITERAL_BYTES,
    DependencyWorkResource.SOURCE_RECORDS: MAX_DEPENDENCY_SOURCE_RECORDS,
    DependencyWorkResource.EMITTED_CHANGES: MAX_DEPENDENCY_SOURCE_CHANGES,
    DependencyWorkResource.FINDING_OUTPUT_RECORDS: MAX_DEPENDENCY_FINDING_OUTPUT_RECORDS,
    DependencyWorkResource.LEDGER_EVENTS: MAX_DEPENDENCY_LEDGER_EVENTS,
    DependencyWorkResource.SHELL_PARSED_BYTES: MAX_DEPENDENCY_SHELL_PARSED_BYTES,
    DependencyWorkResource.RETAINED_SHELL_IR: MAX_DEPENDENCY_RETAINED_SHELL_IR,
    DependencyWorkResource.SHELL_LOCALIZED_ISSUES: MAX_DEPENDENCY_SHELL_LOCALIZED_ISSUES,
}
_FILE_LIMITS: Final[dict[DependencyWorkResource, int]] = {
    DependencyWorkResource.PHYSICAL_BYTES: MAX_DEPENDENCY_FILE_BYTES,
    DependencyWorkResource.YAML_ALIASES: MAX_DEPENDENCY_YAML_ALIASES,
    DependencyWorkResource.DEPTH: MAX_DEPENDENCY_CONFIG_DEPTH,
    DependencyWorkResource.SHELL_UNITS: MAX_DEPENDENCY_SHELL_UNITS_PER_FILE,
    DependencyWorkResource.SHELL_PARSER_CALLS: MAX_DEPENDENCY_SHELL_PARSER_CALLS_PER_FILE,
    DependencyWorkResource.SHELL_SOURCE_MAP_ENTRIES: MAX_DEPENDENCY_SOURCE_MAP_ENTRIES_PER_FILE,
    DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES: MAX_DEPENDENCY_SHELL_VALUE_BYTES_PER_FILE,
}
_DYNAMIC_FILE_RESOURCES: Final = frozenset({DependencyWorkResource.SHELL_PARSED_REVISIT_BYTES})
_UNIT_RESOURCES: Final = frozenset(
    {
        DependencyWorkResource.SHELL_CST_VISITS,
        DependencyWorkResource.SHELL_NESTED_DEPTH,
        DependencyWorkResource.RETAINED_SHELL_IR,
    }
)


class DependencyWorkBudget:
    """The sole owner of scan-wide SC10 resource counters and per-file views."""

    def __init__(
        self,
        *,
        existing_finding_output_records: int = 0,
        existing_ledger_events: int = 0,
    ) -> None:
        existing_findings = _require_nonnegative_integer(
            existing_finding_output_records, "existing_finding_output_records"
        )
        existing_ledger = _require_nonnegative_integer(
            existing_ledger_events, "existing_ledger_events"
        )
        if existing_findings > MAX_DEPENDENCY_FINDING_OUTPUT_RECORDS:
            raise ValueError("existing finding output exceeds the dependency-source ceiling")
        if existing_ledger > MAX_DEPENDENCY_LEDGER_EVENTS:
            raise ValueError("existing ledger output exceeds the dependency-source ceiling")
        self._used: dict[DependencyWorkResource, int] = dict.fromkeys(_SCAN_LIMITS, 0)
        self._used[DependencyWorkResource.FINDING_OUTPUT_RECORDS] = existing_findings
        self._used[DependencyWorkResource.LEDGER_EVENTS] = existing_ledger
        self._truncation_slot_available = existing_ledger < MAX_DEPENDENCY_LEDGER_EVENTS
        self._truncation_slot_claimed = False
        self._shell_truncation_slot_claimed = False
        self._file_budgets: dict[str, DependencyFileBudget] = {}

    @classmethod
    def from_existing(
        cls,
        *,
        findings: Iterable[Finding],
        ledger_events: Iterable[object],
    ) -> DependencyWorkBudget:
        """Initialize remaining capacity from real public-output and ledger footprints."""
        finding_records = sum(max(1, len(finding.occurrences)) for finding in findings)
        ledger_records = sum(1 for _event in ledger_events)
        return cls(
            existing_finding_output_records=finding_records,
            existing_ledger_events=ledger_records,
        )

    def for_file(self, path: str) -> DependencyFileBudget:
        """Return the persistent per-file view for one normalized artifact path."""
        normalized = _normalize_relative_posix_path(path)
        child = self._file_budgets.get(normalized)
        if child is None:
            child = DependencyFileBudget(self, normalized)
            self._file_budgets[normalized] = child
        return child

    def used(self, resource: DependencyWorkResource) -> int:
        """Return a scan-wide counter without exposing mutable budget state."""
        try:
            normalized = DependencyWorkResource(resource)
        except (TypeError, ValueError):
            raise ValueError("dependency work resource is invalid") from None
        if normalized not in _SCAN_LIMITS:
            raise ValueError("resource is per-file")
        return self._used[normalized]

    def _charge(
        self,
        resource: DependencyWorkResource,
        count: int,
    ) -> DependencyWorkExhaustion | None:
        value = _require_nonnegative_integer(count, "count")
        current = self._used[resource]
        limit = _SCAN_LIMITS[resource]
        observed = current + value
        if observed > limit:
            return DependencyWorkExhaustion(resource, observed, limit)
        self._used[resource] = observed
        return None

    def charge_config_nodes(self, count: int) -> DependencyWorkExhaustion | None:
        return self._charge(DependencyWorkResource.CONFIG_NODES, count)

    def charge_retained_literal_bytes(self, count: int) -> DependencyWorkExhaustion | None:
        return self._charge(DependencyWorkResource.RETAINED_LITERAL_BYTES, count)

    def charge_source_records(self, count: int) -> DependencyWorkExhaustion | None:
        return self._charge(DependencyWorkResource.SOURCE_RECORDS, count)

    def charge_emitted_changes(self, count: int) -> DependencyWorkExhaustion | None:
        return self._charge(DependencyWorkResource.EMITTED_CHANGES, count)

    def charge_finding_output_records(self, count: int) -> DependencyWorkExhaustion | None:
        return self._charge(DependencyWorkResource.FINDING_OUTPUT_RECORDS, count)

    def reserve_source_changes(self, count: int = 1) -> DependencyWorkExhaustion | None:
        """Atomically reserve semantic-change and public-finding record capacity."""
        value = _require_nonnegative_integer(count, "count")
        changes = DependencyWorkResource.EMITTED_CHANGES
        findings = DependencyWorkResource.FINDING_OUTPUT_RECORDS
        next_changes = self._used[changes] + value
        next_findings = self._used[findings] + value
        if next_changes > _SCAN_LIMITS[changes]:
            return DependencyWorkExhaustion(changes, next_changes, _SCAN_LIMITS[changes])
        if next_findings > _SCAN_LIMITS[findings]:
            return DependencyWorkExhaustion(findings, next_findings, _SCAN_LIMITS[findings])
        self._used[changes] = next_changes
        self._used[findings] = next_findings
        return None

    def reserve_dependency_outputs(
        self,
        *,
        emitted_changes: int,
        finding_output_records: int,
        new_ledger_events: int,
    ) -> DependencyWorkExhaustion | None:
        """Atomically reserve finalized SC10 changes, findings, and new ledger rows."""
        requested = {
            DependencyWorkResource.EMITTED_CHANGES: _require_nonnegative_integer(
                emitted_changes, "emitted_changes"
            ),
            DependencyWorkResource.FINDING_OUTPUT_RECORDS: _require_nonnegative_integer(
                finding_output_records, "finding_output_records"
            ),
            DependencyWorkResource.LEDGER_EVENTS: _require_nonnegative_integer(
                new_ledger_events, "new_ledger_events"
            ),
        }
        next_used: dict[DependencyWorkResource, int] = {}
        for resource, count in requested.items():
            observed = self._used[resource] + count
            if resource is DependencyWorkResource.LEDGER_EVENTS and self._truncation_slot_available:
                observed += 1
            limit = _SCAN_LIMITS[resource]
            if observed > limit:
                return DependencyWorkExhaustion(resource, observed, limit)
            next_used[resource] = self._used[resource] + count
        self._used.update(next_used)
        return None

    def reserve_source_batch(
        self,
        *,
        source_records: int,
        retained_literal_bytes: int,
        emitted_changes: int,
    ) -> DependencyWorkExhaustion | None:
        """Atomically reserve every output counter for one structured source file."""
        requested = {
            DependencyWorkResource.SOURCE_RECORDS: _require_nonnegative_integer(
                source_records, "source_records"
            ),
            DependencyWorkResource.RETAINED_LITERAL_BYTES: _require_nonnegative_integer(
                retained_literal_bytes, "retained_literal_bytes"
            ),
            DependencyWorkResource.EMITTED_CHANGES: _require_nonnegative_integer(
                emitted_changes, "emitted_changes"
            ),
            DependencyWorkResource.FINDING_OUTPUT_RECORDS: _require_nonnegative_integer(
                emitted_changes, "emitted_changes"
            ),
        }
        next_used: dict[DependencyWorkResource, int] = {}
        for resource, count in requested.items():
            observed = self._used[resource] + count
            limit = _SCAN_LIMITS[resource]
            if observed > limit:
                return DependencyWorkExhaustion(resource, observed, limit)
            next_used[resource] = observed
        self._used.update(next_used)
        return None

    def charge_ledger_events(self, count: int) -> DependencyWorkExhaustion | None:
        """Reserve normal ledger rows without consuming the truncation slot."""
        value = _require_nonnegative_integer(count, "count")
        resource = DependencyWorkResource.LEDGER_EVENTS
        current = self._used[resource]
        reserved = 1 if self._truncation_slot_available else 0
        observed_with_reserve = current + value + reserved
        limit = _SCAN_LIMITS[resource]
        if observed_with_reserve > limit:
            return DependencyWorkExhaustion(resource, observed_with_reserve, limit)
        self._used[resource] = current + value
        return None

    def claim_reserved_truncation_event(self) -> LedgerTruncationClaimStatus:
        """Claim the scan's one reserved truncation row, if physical capacity exists."""
        resource = DependencyWorkResource.LEDGER_EVENTS
        current = self._used[resource]
        limit = _SCAN_LIMITS[resource]
        if self._truncation_slot_claimed:
            return LedgerTruncationClaimStatus.ALREADY_CLAIMED
        if not self._truncation_slot_available or current >= limit:
            return LedgerTruncationClaimStatus.NO_CAPACITY
        self._used[resource] = current + 1
        self._truncation_slot_available = False
        self._truncation_slot_claimed = True
        return LedgerTruncationClaimStatus.CLAIMED

    def charge_shell_issues(self, count: int) -> DependencyWorkExhaustion | None:
        """Reserve localized issues while preserving one scan-wide truncation issue."""
        value = _require_nonnegative_integer(count, "count")
        resource = DependencyWorkResource.SHELL_LOCALIZED_ISSUES
        current = self._used[resource]
        reserved = 0 if self._shell_truncation_slot_claimed else 1
        observed_with_reserve = current + value + reserved
        limit = _SCAN_LIMITS[resource]
        if observed_with_reserve > limit:
            return DependencyWorkExhaustion(resource, observed_with_reserve, limit)
        self._used[resource] = current + value
        return None

    def claim_reserved_shell_truncation_issue(self) -> ShellTruncationClaimStatus:
        """Claim the one localized issue slot reserved for shell truncation."""
        resource = DependencyWorkResource.SHELL_LOCALIZED_ISSUES
        current = self._used[resource]
        limit = _SCAN_LIMITS[resource]
        if self._shell_truncation_slot_claimed:
            return ShellTruncationClaimStatus.ALREADY_CLAIMED
        if current >= limit:
            return ShellTruncationClaimStatus.NO_CAPACITY
        self._used[resource] = current + 1
        self._shell_truncation_slot_claimed = True
        return ShellTruncationClaimStatus.CLAIMED


@dataclass(slots=True)
class DependencyFileBudget:
    """Persistent local ceilings plus delegation to one shared scan budget."""

    _root: DependencyWorkBudget
    path: str
    _used: dict[DependencyWorkResource, int] = field(
        default_factory=lambda: dict.fromkeys((*_FILE_LIMITS, *_DYNAMIC_FILE_RESOURCES), 0)
    )
    _shell_file_size: int | None = None
    _unit_used: dict[tuple[str, DependencyWorkResource], int] = field(default_factory=dict)

    def used(self, resource: DependencyWorkResource) -> int:
        normalized = DependencyWorkResource(resource)
        if normalized in _FILE_LIMITS or normalized in _DYNAMIC_FILE_RESOURCES:
            return self._used[normalized]
        if normalized in _UNIT_RESOURCES:
            raise ValueError("resource requires a shell unit")
        return self._root.used(normalized)

    def _validate_unit(self, unit: ShellUnit) -> None:
        if not isinstance(unit, ShellUnit) or unit.origin_span.path != self.path:
            raise ValueError("shell unit must belong to this normalized file budget")

    def used_for_unit(
        self,
        unit: ShellUnit,
        resource: DependencyWorkResource,
    ) -> int:
        """Return persistent accounting for one opaque unit identity."""
        self._validate_unit(unit)
        normalized = DependencyWorkResource(resource)
        if normalized not in _UNIT_RESOURCES:
            raise ValueError("resource is not tracked per shell unit")
        return self._unit_used.get((unit.unit_id, normalized), 0)

    def _charge_local(
        self,
        resource: DependencyWorkResource,
        count: int,
    ) -> DependencyWorkExhaustion | None:
        value = _require_nonnegative_integer(count, "count")
        current = self._used[resource]
        limit = _FILE_LIMITS[resource]
        observed = current + value
        if observed > limit:
            return DependencyWorkExhaustion(resource, observed, limit)
        self._used[resource] = observed
        return None

    def charge_physical_bytes(self, count: int) -> DependencyWorkExhaustion | None:
        return self._charge_local(DependencyWorkResource.PHYSICAL_BYTES, count)

    def charge_yaml_aliases(self, count: int) -> DependencyWorkExhaustion | None:
        return self._charge_local(DependencyWorkResource.YAML_ALIASES, count)

    def observe_depth(self, depth: int) -> DependencyWorkExhaustion | None:
        value = _require_nonnegative_integer(depth, "depth")
        resource = DependencyWorkResource.DEPTH
        limit = _FILE_LIMITS[resource]
        if value > limit:
            return DependencyWorkExhaustion(resource, value, limit)
        self._used[resource] = max(self._used[resource], value)
        return None

    def charge_shell_units(self, count: int) -> DependencyWorkExhaustion | None:
        return self._charge_local(DependencyWorkResource.SHELL_UNITS, count)

    def charge_source_map_entries(self, count: int) -> DependencyWorkExhaustion | None:
        return self._charge_local(DependencyWorkResource.SHELL_SOURCE_MAP_ENTRIES, count)

    def register_shell_file_size(self, size_bytes: int) -> None:
        """Record canonical physical size without recharging the shared byte counter."""
        size = _require_nonnegative_integer(size_bytes, "size_bytes")
        if size > MAX_DEPENDENCY_FILE_BYTES:
            raise ValueError("shell file exceeds the existing physical byte limit")
        if self._shell_file_size is not None and self._shell_file_size != size:
            raise ValueError("shell file size cannot change during one scan")
        self._shell_file_size = size

    def reserve_shell_parse(self, parsed_bytes: int) -> DependencyWorkExhaustion | None:
        """Atomically reserve one parser call plus file and scan parsed bytes."""
        value = _require_nonnegative_integer(parsed_bytes, "parsed_bytes")
        if self._shell_file_size is None:
            raise ValueError("shell file size must be registered before parsing")
        calls = DependencyWorkResource.SHELL_PARSER_CALLS
        revisits = DependencyWorkResource.SHELL_PARSED_REVISIT_BYTES
        aggregate = DependencyWorkResource.SHELL_PARSED_BYTES
        next_calls = self._used[calls] + 1
        next_revisits = self._used[revisits] + value
        revisit_limit = self._shell_file_size * DEPENDENCY_SHELL_PARSED_BYTE_REVISIT_FACTOR
        next_aggregate = self._root._used[aggregate] + value
        if next_calls > MAX_DEPENDENCY_SHELL_PARSER_CALLS_PER_FILE:
            return DependencyWorkExhaustion(
                calls, next_calls, MAX_DEPENDENCY_SHELL_PARSER_CALLS_PER_FILE
            )
        if next_revisits > revisit_limit:
            return DependencyWorkExhaustion(revisits, next_revisits, revisit_limit)
        if next_aggregate > MAX_DEPENDENCY_SHELL_PARSED_BYTES:
            return DependencyWorkExhaustion(
                aggregate, next_aggregate, MAX_DEPENDENCY_SHELL_PARSED_BYTES
            )
        self._used[calls] = next_calls
        self._used[revisits] = next_revisits
        self._root._used[aggregate] = next_aggregate
        return None

    def charge_shell_cst_visits(
        self,
        unit: ShellUnit,
        count: int,
    ) -> DependencyWorkExhaustion | None:
        """Charge visits against a persistent size-derived unit ceiling."""
        self._validate_unit(unit)
        value = _require_nonnegative_integer(count, "count")
        resource = DependencyWorkResource.SHELL_CST_VISITS
        key = (unit.unit_id, resource)
        current = self._unit_used.get(key, 0)
        limit = DEPENDENCY_SHELL_CST_VISIT_FACTOR * len(unit.raw_bytes) + (
            DEPENDENCY_SHELL_CST_VISIT_BASE
        )
        observed = current + value
        if observed > limit:
            return DependencyWorkExhaustion(resource, observed, limit)
        self._unit_used[key] = observed
        return None

    def observe_shell_nested_depth(
        self,
        unit: ShellUnit,
        depth: int,
    ) -> DependencyWorkExhaustion | None:
        """Persist the deepest literal-shell nesting observed for one unit."""
        self._validate_unit(unit)
        value = _require_nonnegative_integer(depth, "depth")
        resource = DependencyWorkResource.SHELL_NESTED_DEPTH
        if value > MAX_DEPENDENCY_SHELL_NESTED_LITERAL_DEPTH:
            return DependencyWorkExhaustion(
                resource, value, MAX_DEPENDENCY_SHELL_NESTED_LITERAL_DEPTH
            )
        key = (unit.unit_id, resource)
        self._unit_used[key] = max(self._unit_used.get(key, 0), value)
        return None

    def charge_retained_shell_ir(
        self,
        unit: ShellUnit,
        count: int,
    ) -> DependencyWorkExhaustion | None:
        """Charge scan-wide retained IR while preserving per-unit accounting."""
        self._validate_unit(unit)
        value = _require_nonnegative_integer(count, "count")
        resource = DependencyWorkResource.RETAINED_SHELL_IR
        next_global = self._root._used[resource] + value
        limit = _SCAN_LIMITS[resource]
        if next_global > limit:
            return DependencyWorkExhaustion(resource, next_global, limit)
        key = (unit.unit_id, resource)
        self._root._used[resource] = next_global
        self._unit_used[key] = self._unit_used.get(key, 0) + value
        return None

    def reserve_shell_value_bytes(self, count: int) -> DependencyWorkExhaustion | None:
        """Atomically charge global literals and this file's shell values."""
        value = _require_nonnegative_integer(count, "count")
        global_resource = DependencyWorkResource.RETAINED_LITERAL_BYTES
        file_resource = DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES
        next_global = self._root._used[global_resource] + value
        next_file = self._used[file_resource] + value
        global_limit = _SCAN_LIMITS[global_resource]
        file_limit = _FILE_LIMITS[file_resource]
        if next_file > file_limit:
            return DependencyWorkExhaustion(file_resource, next_file, file_limit)
        if next_global > global_limit:
            return DependencyWorkExhaustion(global_resource, next_global, global_limit)
        self._root._used[global_resource] = next_global
        self._used[file_resource] = next_file
        return None

    def charge_config_nodes(self, count: int) -> DependencyWorkExhaustion | None:
        return self._root.charge_config_nodes(count)

    def charge_retained_literal_bytes(self, count: int) -> DependencyWorkExhaustion | None:
        return self._root.charge_retained_literal_bytes(count)

    def charge_source_records(self, count: int) -> DependencyWorkExhaustion | None:
        return self._root.charge_source_records(count)

    def charge_emitted_changes(self, count: int) -> DependencyWorkExhaustion | None:
        return self._root.charge_emitted_changes(count)

    def charge_finding_output_records(self, count: int) -> DependencyWorkExhaustion | None:
        return self._root.charge_finding_output_records(count)

    def reserve_source_changes(self, count: int = 1) -> DependencyWorkExhaustion | None:
        return self._root.reserve_source_changes(count)

    def reserve_source_batch(
        self,
        *,
        source_records: int,
        retained_literal_bytes: int,
        emitted_changes: int,
    ) -> DependencyWorkExhaustion | None:
        return self._root.reserve_source_batch(
            source_records=source_records,
            retained_literal_bytes=retained_literal_bytes,
            emitted_changes=emitted_changes,
        )

    def charge_ledger_events(self, count: int) -> DependencyWorkExhaustion | None:
        return self._root.charge_ledger_events(count)

    def claim_reserved_truncation_event(self) -> LedgerTruncationClaimStatus:
        return self._root.claim_reserved_truncation_event()

    def charge_shell_issues(self, count: int) -> DependencyWorkExhaustion | None:
        return self._root.charge_shell_issues(count)

    def claim_reserved_shell_truncation_issue(self) -> ShellTruncationClaimStatus:
        return self._root.claim_reserved_shell_truncation_issue()
