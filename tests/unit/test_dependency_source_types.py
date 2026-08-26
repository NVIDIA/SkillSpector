# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit contracts for dependency-source semantics and resource accounting."""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable
from typing import Any

import pytest

from skillspector.models import Finding


def _api() -> Any:
    """Import the real contract module while keeping the initial TDD run collectable."""
    try:
        return importlib.import_module("skillspector.dependency_source_types")
    except ImportError:
        pytest.fail("dependency-source semantic contracts are unavailable")


def _span(api: Any) -> Any:
    return api.SourceSpan(
        path="config/.npmrc",
        start_byte=2,
        end_byte=9,
        start_line=1,
        end_line=1,
    )


def test_source_span_normalizes_relative_posix_path_and_preserves_utf8_byte_offsets() -> None:
    api = _api()

    span = api.SourceSpan(
        path="./config//pip.conf",
        start_byte=len("é".encode()),
        end_byte=len("éindex".encode()),
        start_line=2,
        end_line=3,
    )

    assert span.path == "config/pip.conf"
    assert (span.start_byte, span.end_byte) == (2, 7)
    assert (span.start_line, span.end_line) == (2, 3)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", ""),
        ("path", "/absolute/npmrc"),
        ("path", "../outside/npmrc"),
        ("path", "config\\npmrc"),
        ("start_byte", -1),
        ("start_byte", True),
        ("end_byte", -1),
        ("end_byte", False),
        ("start_line", 0),
        ("start_line", True),
        ("end_line", 0),
        ("end_line", False),
    ],
)
def test_source_span_rejects_unsafe_paths_and_non_integer_or_negative_ranges(
    field: str,
    value: object,
) -> None:
    api = _api()
    values: dict[str, object] = {
        "path": "config/npmrc",
        "start_byte": 0,
        "end_byte": 4,
        "start_line": 1,
        "end_line": 1,
    }
    values[field] = value

    with pytest.raises(ValueError):
        api.SourceSpan(**values)


@pytest.mark.parametrize(
    ("start_byte", "end_byte", "start_line", "end_line"),
    [(5, 4, 1, 1), (0, 1, 2, 1)],
)
def test_source_span_rejects_reversed_ranges(
    start_byte: int,
    end_byte: int,
    start_line: int,
    end_line: int,
) -> None:
    api = _api()

    with pytest.raises(ValueError):
        api.SourceSpan(
            path="config/npmrc",
            start_byte=start_byte,
            end_byte=end_byte,
            start_line=start_line,
            end_line=end_line,
        )


def test_source_change_accepts_only_redacted_resolved_destinations() -> None:
    api = _api()
    raw_secret = "change-secret-4f387"
    raw_destination = f"https://alice:{raw_secret}@packages.example.invalid/private"

    with pytest.raises(ValueError) as error:
        api.SourceChange(
            ecosystem="npm",
            surface="source",
            operation="replace",
            scope="global",
            destination=raw_destination,
            destination_status=api.DestinationStatus.RESOLVED,
            span=_span(api),
        )

    assert raw_secret not in str(error.value)

    change = api.SourceChange(
        ecosystem="npm",
        surface="source",
        operation="replace",
        scope="global",
        destination="https://packages.example.invalid/REDACTED_PATH",
        destination_status=api.DestinationStatus.RESOLVED,
        span=_span(api),
    )
    assert change.destination == "https://packages.example.invalid/REDACTED_PATH"
    assert change.ecosystem is api.DependencyEcosystem.NPM
    assert change.surface is api.DependencySourceSurface.SOURCE
    assert change.operation is api.DependencySourceOperation.REPLACE
    assert change.scope is api.DependencySourceScope.GLOBAL
    assert change.destination_status is api.DestinationStatus.RESOLVED


@pytest.mark.parametrize(
    ("value", "member_name"),
    [
        ("yarn", "YARN"),
        ("poetry", "POETRY"),
        ("pdm", "PDM"),
        ("uv", "UV"),
    ],
)
def test_dependency_ecosystem_has_fixed_pr2_parser_categories(
    value: str,
    member_name: str,
) -> None:
    api = _api()

    member = getattr(api.DependencyEcosystem, member_name)

    assert api.DependencyEcosystem(value) is member


@pytest.mark.parametrize(
    ("value", "member_name"),
    [
        (".npmrc", "NPMRC"),
        ("pip config", "PIP_CONFIG"),
        ("cargo-config", "CARGO_CONFIG"),
        ("maven-config", "MAVEN_CONFIG"),
    ],
)
def test_dependency_surface_has_fixed_direct_config_categories(
    value: str,
    member_name: str,
) -> None:
    api = _api()

    member = getattr(api.DependencySourceSurface, member_name)

    assert api.DependencySourceSurface(value) is member


@pytest.mark.parametrize(
    ("value", "member_name"),
    [
        ("source", "SOURCE"),
        ("registry", "REGISTRY"),
        ("mirror", "MIRROR"),
        ("repository", "REPOSITORY"),
    ],
)
def test_dependency_scope_has_fixed_cargo_and_maven_categories(
    value: str,
    member_name: str,
) -> None:
    api = _api()

    member = getattr(api.DependencySourceScope, member_name)

    assert api.DependencySourceScope(value) is member


@pytest.mark.parametrize(
    "raw_destination",
    [
        "token=type-boundary-secret",
        "ftp://user:type-boundary-secret@packages.example.invalid/private",
        "https://packages.example.invalid/private?apikey=type-boundary-secret",
        "https://packages.example.invalid/private?channel=stable;authToken=type-boundary-secret",
    ],
)
def test_source_change_rejects_raw_destination_redaction_bypasses(
    raw_destination: str,
) -> None:
    api = _api()

    with pytest.raises(ValueError) as error:
        api.SourceChange(
            ecosystem="npm",
            surface="source",
            operation="replace",
            scope="global",
            destination=raw_destination,
            destination_status=api.DestinationStatus.RESOLVED,
            span=_span(api),
        )

    assert "type-boundary-secret" not in str(error.value)


@pytest.mark.parametrize(
    "destination",
    [
        "[REDACTED_URL]",
        "//packages.example.invalid/REDACTED_PATH",
        "https://safe.invalid/REDACTED_PATH",
    ],
)
def test_source_change_accepts_only_stable_sanitized_destinations(
    destination: str,
) -> None:
    api = _api()

    change = api.SourceChange(
        ecosystem="npm",
        surface="source",
        operation="replace",
        scope="global",
        destination=destination,
        destination_status=api.DestinationStatus.RESOLVED,
        span=_span(api),
    )

    assert change.destination == destination


def test_source_change_rejects_noncanonical_non_url_destinations() -> None:
    api = _api()
    sentinel = "non-url-destination-secret"

    with pytest.raises(ValueError) as error:
        api.SourceChange(
            ecosystem="npm",
            surface="source",
            operation="replace",
            scope="global",
            destination=f"token={sentinel}",
            destination_status=api.DestinationStatus.RESOLVED,
            span=_span(api),
        )

    assert sentinel not in str(error.value)


def test_source_change_uses_one_exact_unresolved_representation() -> None:
    api = _api()

    change = api.SourceChange(
        ecosystem="pip",
        surface="source",
        operation="replace",
        scope="global",
        destination="unresolved",
        destination_status="unresolved",
        span=_span(api),
    )

    assert change.destination_status is api.DestinationStatus.UNRESOLVED
    assert change.destination == "unresolved"
    for invalid in ("", "${REGISTRY}", "UNRESOLVED"):
        with pytest.raises(ValueError):
            dataclasses.replace(change, destination=invalid)


def test_source_change_rejects_empty_semantic_fields_and_has_no_raw_payload_slots() -> None:
    api = _api()
    base = api.SourceChange(
        ecosystem="pip",
        surface="source",
        operation="replace",
        scope="global",
        destination="unresolved",
        destination_status=api.DestinationStatus.UNRESOLVED,
        span=_span(api),
    )

    for field in ("ecosystem", "surface", "operation", "scope"):
        with pytest.raises(ValueError):
            dataclasses.replace(base, **{field: ""})

    assert {field.name for field in dataclasses.fields(api.SourceChange)} == {
        "ecosystem",
        "surface",
        "operation",
        "scope",
        "destination",
        "destination_status",
        "span",
    }


@pytest.mark.parametrize("field", ["ecosystem", "surface", "operation", "scope"])
@pytest.mark.parametrize("unsafe", ["attacker-secret", "safe\x00value"])
def test_source_change_semantics_reject_attacker_controlled_labels(
    field: str,
    unsafe: str,
) -> None:
    api = _api()
    base = api.SourceChange(
        ecosystem="pip",
        surface="source",
        operation="replace",
        scope="global",
        destination="unresolved",
        destination_status=api.DestinationStatus.UNRESOLVED,
        span=_span(api),
    )

    with pytest.raises(ValueError):
        dataclasses.replace(base, **{field: unsafe})


@pytest.mark.parametrize("destination", ["", "   ", "https://host.invalid/\x00path"])
def test_resolved_destination_rejects_blank_or_control_bearing_values(destination: str) -> None:
    api = _api()

    with pytest.raises(ValueError):
        api.SourceChange(
            ecosystem="npm",
            surface="source",
            operation="replace",
            scope="global",
            destination=destination,
            destination_status=api.DestinationStatus.RESOLVED,
            span=_span(api),
        )


def test_resolved_destination_rejects_values_above_its_explicit_bound() -> None:
    api = _api()
    destination = "https://packages.example.invalid/" + (
        "a" * api.MAX_DEPENDENCY_DESTINATION_CHARACTERS
    )

    with pytest.raises(ValueError):
        api.SourceChange(
            ecosystem="npm",
            surface="source",
            operation="replace",
            scope="global",
            destination=destination,
            destination_status=api.DestinationStatus.RESOLVED,
            span=_span(api),
        )


def test_parse_and_analysis_results_freeze_iterables_as_tuples() -> None:
    api = _api()
    change = api.SourceChange(
        ecosystem="pip",
        surface="source",
        operation="replace",
        scope="global",
        destination="unresolved",
        destination_status=api.DestinationStatus.UNRESOLVED,
        span=_span(api),
    )
    limitation = api.DependencySourceLimitation(
        reason=api.DependencySourceLimitationReason.PARSE_INCOMPLETE,
        path="config/pip.conf",
        start_line=1,
        end_line=1,
        observed_records=51,
        limit_records=50,
    )

    parsed = api.DependencySourceParseResult(changes=[change], limitations=[limitation])
    finding = Finding(rule_id="SC10", message="source changed")
    analysis = api.DependencySourceAnalysis(findings=[finding], limitations=[limitation])

    assert parsed.changes == (change,)
    assert parsed.limitations == (limitation,)
    assert analysis.findings == (finding,)
    assert analysis.limitations == (limitation,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        parsed.changes = ()


def test_limitation_exposes_only_safe_path_range_and_ledger_numeric_metrics() -> None:
    api = _api()

    limitation = api.DependencySourceLimitation(
        reason="dependency_source_parse_incomplete",
        path="./config//pip.conf",
        start_line=3,
        end_line=4,
        observed_bytes=1_000_001,
        limit_bytes=1_000_000,
    )

    assert limitation.reason is api.DependencySourceLimitationReason.PARSE_INCOMPLETE
    assert limitation.path == "config/pip.conf"
    assert limitation.ledger_metrics() == {
        "observed_bytes": 1_000_001,
        "limit_bytes": 1_000_000,
    }
    assert {field.name for field in dataclasses.fields(api.DependencySourceLimitation)} == {
        "reason",
        "path",
        "start_line",
        "end_line",
        "observed_bytes",
        "limit_bytes",
        "observed_findings",
        "limit_findings",
        "observed_depth",
        "limit_depth",
        "observed_records",
        "limit_records",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_line", 0),
        ("end_line", 0),
        ("observed_bytes", -1),
        ("limit_bytes", True),
        ("observed_records", False),
    ],
)
def test_limitation_rejects_invalid_ranges_and_metrics(field: str, value: object) -> None:
    api = _api()
    values: dict[str, object] = {
        "reason": api.DependencySourceLimitationReason.PARSE_INCOMPLETE,
        "path": "pip.conf",
        "start_line": 1,
        "end_line": 1,
        "observed_records": 2,
        "limit_records": 1,
    }
    values[field] = value

    with pytest.raises(ValueError):
        api.DependencySourceLimitation(**values)


def test_source_change_conversion_is_the_single_safe_finding_boundary() -> None:
    api = _api()
    change = api.SourceChange(
        ecosystem="npm",
        surface="source",
        operation="replace",
        scope="scoped",
        destination="https://packages.example.invalid/REDACTED_PATH",
        destination_status=api.DestinationStatus.RESOLVED,
        span=_span(api),
    )

    finding = api.finding_from_source_change(change)

    assert finding.rule_id == "SC10"
    assert finding.severity == "HIGH"
    assert finding.file == "config/.npmrc"
    assert (finding.start_line, finding.end_line) == (1, 1)
    assert finding.evidence == {
        "ecosystem": "npm",
        "surface": "source",
        "operation": "replace",
        "scope": "scoped",
        "destination": "https://packages.example.invalid/REDACTED_PATH",
        "destination_status": "resolved",
    }


def test_file_children_share_every_scan_wide_counter() -> None:
    api = _api()
    budget = api.DependencyWorkBudget()
    first = budget.for_file("config/first.conf")
    second = budget.for_file("config/second.conf")

    assert first.charge_config_nodes(30_000) is None
    assert second.charge_config_nodes(20_000) is None
    exhaustion = first.charge_config_nodes(1)

    assert exhaustion == api.DependencyWorkExhaustion(
        resource=api.DependencyWorkResource.CONFIG_NODES,
        observed=50_001,
        limit=50_000,
    )
    assert budget.used(api.DependencyWorkResource.CONFIG_NODES) == 50_000


@pytest.mark.parametrize(
    ("method_name", "resource", "limit"),
    [
        ("charge_config_nodes", "config_nodes", 50_000),
        ("charge_retained_literal_bytes", "retained_literal_bytes", 2_000_000),
        ("charge_source_records", "source_records", 50_000),
        ("charge_emitted_changes", "emitted_changes", 10_000),
        ("charge_finding_output_records", "finding_output_records", 10_000),
    ],
)
def test_scan_budget_accepts_exact_limit_and_rejects_one_over_atomically(
    method_name: str,
    resource: str,
    limit: int,
) -> None:
    api = _api()
    budget = api.DependencyWorkBudget()
    charge: Callable[[int], Any] = getattr(budget, method_name)

    assert charge(limit) is None
    exhaustion = charge(1)

    assert exhaustion.resource is api.DependencyWorkResource(resource)
    assert (exhaustion.observed, exhaustion.limit) == (limit + 1, limit)
    assert budget.used(api.DependencyWorkResource(resource)) == limit
    assert set(dataclasses.asdict(exhaustion)) == {"resource", "observed", "limit"}


@pytest.mark.parametrize(
    ("method_name", "resource", "limit"),
    [
        ("charge_physical_bytes", "physical_bytes", 1_000_000),
        ("charge_yaml_aliases", "yaml_aliases", 256),
        ("observe_depth", "depth", 64),
    ],
)
def test_file_budget_accepts_exact_limit_and_rejects_one_over_atomically(
    method_name: str,
    resource: str,
    limit: int,
) -> None:
    api = _api()
    file_budget = api.DependencyWorkBudget().for_file("config/source.conf")
    charge: Callable[[int], Any] = getattr(file_budget, method_name)

    assert charge(limit) is None
    exhaustion = charge(limit + 1 if method_name == "observe_depth" else 1)

    assert exhaustion.resource is api.DependencyWorkResource(resource)
    assert exhaustion.limit == limit
    assert file_budget.used(api.DependencyWorkResource(resource)) == limit


def test_file_children_have_independent_physical_limits_without_multiplying_scan_limits() -> None:
    api = _api()
    budget = api.DependencyWorkBudget()
    first = budget.for_file("config/first.conf")
    second = budget.for_file("config/second.conf")

    assert first.charge_physical_bytes(1_000_000) is None
    assert second.charge_physical_bytes(1_000_000) is None
    assert first.charge_physical_bytes(1) is not None
    assert first.charge_source_records(30_000) is None
    assert second.charge_source_records(20_000) is None
    assert second.charge_source_records(1) is not None


def test_reopening_same_normalized_path_cannot_reset_per_file_capacity() -> None:
    api = _api()
    budget = api.DependencyWorkBudget()
    first = budget.for_file("./config//source.yml")

    assert first.charge_physical_bytes(1_000_000) is None
    reopened = budget.for_file("config/source.yml")
    exhaustion = reopened.charge_physical_bytes(1)

    assert exhaustion.resource is api.DependencyWorkResource.PHYSICAL_BYTES
    assert reopened.used(api.DependencyWorkResource.PHYSICAL_BYTES) == 1_000_000


def test_failed_scan_charge_does_not_mutate_target_or_related_counters() -> None:
    api = _api()
    budget = api.DependencyWorkBudget()
    first = budget.for_file("first.conf")
    second = budget.for_file("second.conf")
    assert first.charge_emitted_changes(10_000) is None
    before = {
        resource: budget.used(resource)
        for resource in api.DependencyWorkResource
        if resource
        not in {
            api.DependencyWorkResource.PHYSICAL_BYTES,
            api.DependencyWorkResource.YAML_ALIASES,
            api.DependencyWorkResource.DEPTH,
        }
    }

    exhaustion = second.charge_emitted_changes(1)

    assert exhaustion.resource is api.DependencyWorkResource.EMITTED_CHANGES
    assert {resource: budget.used(resource) for resource in before} == before


def test_source_change_reservation_charges_change_and_finding_capacity_atomically() -> None:
    api = _api()
    budget = api.DependencyWorkBudget()
    assert budget.charge_emitted_changes(9_999) is None
    assert budget.charge_finding_output_records(9_999) is None

    assert budget.reserve_source_changes() is None

    assert budget.used(api.DependencyWorkResource.EMITTED_CHANGES) == 10_000
    assert budget.used(api.DependencyWorkResource.FINDING_OUTPUT_RECORDS) == 10_000


def test_source_change_reservation_mutates_neither_counter_when_finding_capacity_is_full() -> None:
    api = _api()
    budget = api.DependencyWorkBudget()
    assert budget.charge_emitted_changes(9_999) is None
    assert budget.charge_finding_output_records(10_000) is None

    exhaustion = budget.reserve_source_changes()

    assert exhaustion.resource is api.DependencyWorkResource.FINDING_OUTPUT_RECORDS
    assert budget.used(api.DependencyWorkResource.EMITTED_CHANGES) == 9_999
    assert budget.used(api.DependencyWorkResource.FINDING_OUTPUT_RECORDS) == 10_000


def test_source_change_reservation_mutates_neither_counter_when_change_capacity_is_full() -> None:
    api = _api()
    budget = api.DependencyWorkBudget()
    assert budget.charge_emitted_changes(10_000) is None
    assert budget.charge_finding_output_records(9_999) is None

    exhaustion = budget.reserve_source_changes()

    assert exhaustion.resource is api.DependencyWorkResource.EMITTED_CHANGES
    assert budget.used(api.DependencyWorkResource.EMITTED_CHANGES) == 10_000
    assert budget.used(api.DependencyWorkResource.FINDING_OUTPUT_RECORDS) == 9_999


def test_finding_capacity_starts_from_existing_public_output_record_footprint() -> None:
    api = _api()
    existing = Finding(
        rule_id="SC1",
        message="existing",
        occurrences=[{"file": "SKILL.md", "start_line": 1}] * 9_999,
    )
    budget = api.DependencyWorkBudget.from_existing(findings=[existing], ledger_events=[])

    assert budget.charge_finding_output_records(1) is None
    exhaustion = budget.charge_finding_output_records(1)

    assert exhaustion.observed == 10_001
    assert exhaustion.limit == 10_000
    assert budget.used(api.DependencyWorkResource.FINDING_OUTPUT_RECORDS) == 10_000


def test_ledger_budget_reserves_one_truncation_slot_at_9_999_existing_rows() -> None:
    api = _api()
    budget = api.DependencyWorkBudget.from_existing(findings=[], ledger_events=[{}] * 9_999)

    normal_exhaustion = budget.charge_ledger_events(1)

    assert normal_exhaustion.resource is api.DependencyWorkResource.LEDGER_EVENTS
    assert budget.used(api.DependencyWorkResource.LEDGER_EVENTS) == 9_999
    assert budget.claim_reserved_truncation_event() is api.LedgerTruncationClaimStatus.CLAIMED
    assert budget.used(api.DependencyWorkResource.LEDGER_EVENTS) == 10_000
    assert (
        budget.claim_reserved_truncation_event() is api.LedgerTruncationClaimStatus.ALREADY_CLAIMED
    )


def test_ledger_budget_allows_one_normal_row_plus_reserved_slot_at_9_998() -> None:
    api = _api()
    budget = api.DependencyWorkBudget.from_existing(findings=[], ledger_events=[{}] * 9_998)

    assert budget.charge_ledger_events(1) is None
    assert budget.charge_ledger_events(1) is not None
    assert budget.claim_reserved_truncation_event() is api.LedgerTruncationClaimStatus.CLAIMED
    assert budget.used(api.DependencyWorkResource.LEDGER_EVENTS) == 10_000


def test_reserved_truncation_slot_can_be_claimed_once_across_file_siblings() -> None:
    api = _api()
    budget = api.DependencyWorkBudget.from_existing(findings=[], ledger_events=[{}] * 9_999)
    first = budget.for_file("first.conf")
    second = budget.for_file("second.conf")

    assert first.claim_reserved_truncation_event() is api.LedgerTruncationClaimStatus.CLAIMED
    assert (
        second.claim_reserved_truncation_event() is api.LedgerTruncationClaimStatus.ALREADY_CLAIMED
    )
    assert budget.used(api.DependencyWorkResource.LEDGER_EVENTS) == 10_000


def test_full_existing_ledger_has_no_fabricated_truncation_slot() -> None:
    api = _api()
    budget = api.DependencyWorkBudget.from_existing(findings=[], ledger_events=[{}] * 10_000)

    status = budget.claim_reserved_truncation_event()

    assert status is api.LedgerTruncationClaimStatus.NO_CAPACITY
    assert budget.used(api.DependencyWorkResource.LEDGER_EVENTS) == 10_000


def test_dependency_work_exhaustion_requires_a_real_one_over_capacity_observation() -> None:
    api = _api()

    with pytest.raises(ValueError):
        api.DependencyWorkExhaustion(
            resource=api.DependencyWorkResource.LEDGER_EVENTS,
            observed=2,
            limit=10_000,
        )
    with pytest.raises(ValueError):
        api.DependencyWorkExhaustion(
            resource=api.DependencyWorkResource.LEDGER_EVENTS,
            observed=10_000,
            limit=10_000,
        )


@pytest.mark.parametrize(
    ("finding_records", "ledger_events"),
    [(10_001, 0), (0, 10_001)],
)
def test_preexisting_output_counts_above_global_ceiling_are_rejected(
    finding_records: int,
    ledger_events: int,
) -> None:
    api = _api()

    with pytest.raises(ValueError):
        api.DependencyWorkBudget(
            existing_finding_output_records=finding_records,
            existing_ledger_events=ledger_events,
        )


@pytest.mark.parametrize(
    "method_name",
    [
        "charge_config_nodes",
        "charge_retained_literal_bytes",
        "charge_source_records",
        "charge_emitted_changes",
        "charge_finding_output_records",
        "charge_ledger_events",
    ],
)
@pytest.mark.parametrize("invalid", [-1, True, False])
def test_scan_charges_reject_negative_and_boolean_counts(
    method_name: str,
    invalid: int | bool,
) -> None:
    api = _api()
    budget = api.DependencyWorkBudget()

    with pytest.raises(ValueError):
        getattr(budget, method_name)(invalid)


@pytest.mark.parametrize(
    "method_name", ["charge_physical_bytes", "charge_yaml_aliases", "observe_depth"]
)
@pytest.mark.parametrize("invalid", [-1, True, False])
def test_file_charges_reject_negative_and_boolean_counts(
    method_name: str,
    invalid: int | bool,
) -> None:
    api = _api()
    file_budget = api.DependencyWorkBudget().for_file("config/source.conf")

    with pytest.raises(ValueError):
        getattr(file_budget, method_name)(invalid)
