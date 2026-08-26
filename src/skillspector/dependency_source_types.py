# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared semantic and resource contracts for dependency-source analysis."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
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


class DestinationStatus(StrEnum):
    """Whether a source destination is literal or conservatively unresolved."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


class DependencyEcosystem(StrEnum):
    """Code-owned dependency ecosystems implemented by source parsers."""

    NPM = "npm"
    YARN = "yarn"
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


class LedgerTruncationClaimStatus(StrEnum):
    """Outcome of claiming the scan's single reserved truncation row."""

    CLAIMED = "claimed"
    ALREADY_CLAIMED = "already_claimed"
    NO_CAPACITY = "no_capacity"


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

    changes: tuple[SourceChange, ...] = ()
    limitations: tuple[DependencySourceLimitation, ...] = ()

    def __post_init__(self) -> None:
        changes = tuple(self.changes)
        limitations = tuple(self.limitations)
        if not all(isinstance(change, SourceChange) for change in changes):
            raise ValueError("changes must contain SourceChange values")
        if not all(isinstance(item, DependencySourceLimitation) for item in limitations):
            raise ValueError("limitations must contain DependencySourceLimitation values")
        object.__setattr__(self, "changes", changes)
        object.__setattr__(self, "limitations", limitations)


@dataclass(frozen=True, slots=True)
class DependencySourceAnalysis:
    """Public deterministic findings plus any localized analysis limitations."""

    findings: tuple[Finding, ...] = ()
    limitations: tuple[DependencySourceLimitation, ...] = ()
    applicable_spans: tuple[DependencySourceSpan, ...] = ()
    inspected_spans: tuple[DependencySourceSpan, ...] = ()
    ledger_exhaustion: DependencyWorkExhaustion | None = None

    def __post_init__(self) -> None:
        findings = tuple(self.findings)
        limitations = tuple(self.limitations)
        applicable_spans = tuple(self.applicable_spans)
        inspected_spans = tuple(self.inspected_spans)
        if not all(isinstance(finding, Finding) for finding in findings):
            raise ValueError("findings must contain Finding values")
        if not all(isinstance(item, DependencySourceLimitation) for item in limitations):
            raise ValueError("limitations must contain DependencySourceLimitation values")
        if not all(isinstance(item, DependencySourceSpan) for item in applicable_spans):
            raise ValueError("applicable_spans must contain DependencySourceSpan values")
        if not all(isinstance(item, DependencySourceSpan) for item in inspected_spans):
            raise ValueError("inspected_spans must contain DependencySourceSpan values")
        if self.ledger_exhaustion is not None and not isinstance(
            self.ledger_exhaustion, DependencyWorkExhaustion
        ):
            raise ValueError("ledger_exhaustion must be DependencyWorkExhaustion")
        object.__setattr__(self, "findings", findings)
        object.__setattr__(self, "limitations", limitations)
        object.__setattr__(self, "applicable_spans", applicable_spans)
        object.__setattr__(self, "inspected_spans", inspected_spans)


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
        }:
            prefix = "bytes"
        elif self.resource in {
            DependencyWorkResource.EMITTED_CHANGES,
            DependencyWorkResource.FINDING_OUTPUT_RECORDS,
        }:
            prefix = "findings"
        elif self.resource is DependencyWorkResource.DEPTH:
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
}
_FILE_LIMITS: Final[dict[DependencyWorkResource, int]] = {
    DependencyWorkResource.PHYSICAL_BYTES: MAX_DEPENDENCY_FILE_BYTES,
    DependencyWorkResource.YAML_ALIASES: MAX_DEPENDENCY_YAML_ALIASES,
    DependencyWorkResource.DEPTH: MAX_DEPENDENCY_CONFIG_DEPTH,
}


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


@dataclass(slots=True)
class DependencyFileBudget:
    """Persistent local ceilings plus delegation to one shared scan budget."""

    _root: DependencyWorkBudget
    path: str
    _used: dict[DependencyWorkResource, int] = field(
        default_factory=lambda: dict.fromkeys(_FILE_LIMITS, 0)
    )

    def used(self, resource: DependencyWorkResource) -> int:
        normalized = DependencyWorkResource(resource)
        if normalized in _FILE_LIMITS:
            return self._used[normalized]
        return self._root.used(normalized)

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
