# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit contracts for dependency-source semantics and resource accounting."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
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


def _shell_unit(
    api: Any,
    *,
    path: str = "scripts/setup.sh",
    raw: bytes = b"x",
) -> Any:
    return api.ShellUnit(
        dialect=api.ShellDialect.BASH,
        kind=api.ShellUnitKind.STANDALONE,
        provenance=api.SiteProvenance.FILE_SUFFIX,
        raw_bytes=raw,
        origin_span=api.SourceSpan(
            path,
            0,
            len(raw),
            1,
            1,
            start_column=0,
            end_column=len(raw),
        ),
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


def test_source_span_keeps_legacy_positionals_and_adds_keyword_only_byte_columns() -> None:
    api = _api()

    legacy = api.SourceSpan("scripts/setup.sh", 0, 6, 1, 1)
    exact = api.SourceSpan(
        "scripts/setup.sh",
        3,
        6,
        1,
        1,
        start_column=3,
        end_column=6,
    )
    parameters = inspect.signature(api.SourceSpan).parameters

    assert (legacy.start_column, legacy.end_column) == (None, None)
    assert (exact.start_column, exact.end_column) == (3, 6)
    assert parameters["start_column"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["end_column"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize(
    ("start_column", "end_column"),
    [(0, None), (None, 0), (-1, 0), (True, 1), (4, 3)],
)
def test_source_span_rejects_incomplete_or_invalid_byte_columns(
    start_column: object,
    end_column: object,
) -> None:
    api = _api()

    with pytest.raises(ValueError):
        api.SourceSpan(
            "scripts/setup.sh",
            0,
            1,
            1,
            1,
            start_column=start_column,
            end_column=end_column,
        )


def test_shell_contracts_are_frozen_code_owned_and_privacy_safe() -> None:
    api = _api()
    secret = b"credential-8f42c"
    unit = _shell_unit(api, raw=secret)
    exact = api.StaticValue.exact(secret)
    command = api.CommandSite(
        unit_id=unit.unit_id,
        provenance=api.SiteProvenance.FILE_SUFFIX,
        span=unit.origin_span,
        argv=[exact, api.StaticValue.unknown()],
    )
    assignment = api.AssignmentSite(
        unit_id=unit.unit_id,
        provenance=api.SiteProvenance.FILE_SUFFIX,
        span=unit.origin_span,
        name="REGISTRY",
        value=exact,
    )
    generated = api.GeneratedConfig(
        unit_id=unit.unit_id,
        provenance=api.SiteProvenance.GENERATED_CONFIG,
        span=unit.origin_span,
        target=api.StaticValue.unbound(),
        content=exact,
    )

    assert {item.value for item in api.ShellDialect} == {"bash", "sh", "dash"}
    assert command.argv == (exact, api.StaticValue.unknown())
    assert secret.decode() not in repr(unit)
    assert secret.decode() not in repr(exact)
    assert secret.decode() not in repr(command)
    assert secret.decode() not in repr(assignment)
    assert secret.decode() not in repr(generated)
    with pytest.raises(dataclasses.FrozenInstanceError):
        unit.raw_bytes = b"changed"
    with pytest.raises(TypeError):
        api.ShellUnit(
            unit_id="caller-chosen",
            dialect=api.ShellDialect.BASH,
            kind=api.ShellUnitKind.STANDALONE,
            provenance=api.SiteProvenance.FILE_SUFFIX,
            raw_bytes=b"x",
            origin_span=unit.origin_span,
        )


def test_command_site_exposes_only_typed_execution_facts_with_aligned_spans() -> None:
    api = _api()
    unit = _shell_unit(api, raw=b"npm config")
    prefix = api.AssignmentSite(
        unit_id=unit.unit_id,
        provenance=unit.provenance,
        span=unit.origin_span,
        name="NPM_CONFIG_REGISTRY",
        value=api.StaticValue.exact(b"https://packages.example.invalid"),
    )

    command = api.CommandSite(
        unit_id=unit.unit_id,
        provenance=unit.provenance,
        span=unit.origin_span,
        argv=(api.StaticValue.exact(b"npm"), api.StaticValue.exact(b"config")),
        argument_spans=(unit.origin_span, unit.origin_span),
        resolution=api.CommandResolutionKind.EXTERNAL,
        producer=api.CommandProducerReachability.ACTIVE,
        prefix_assignments=(prefix,),
        exported_assignments=(),
    )

    assert len(command.argument_spans) == len(command.argv)
    assert command.prefix_assignments == (prefix,)
    assert {item.value for item in api.CommandResolutionKind} == {
        "external",
        "function",
        "ambiguous",
    }
    assert {item.value for item in api.CommandProducerReachability} == {
        "active",
        "inert",
        "ambiguous",
    }
    with pytest.raises(ValueError, match="argument_spans"):
        api.CommandSite(
            unit_id=unit.unit_id,
            provenance=unit.provenance,
            span=unit.origin_span,
            argv=(api.StaticValue.exact(b"npm"), api.StaticValue.exact(b"config")),
            argument_spans=(unit.origin_span,),
        )


def test_shell_unit_ids_are_deterministic_opaque_coordinates_not_content_hashes() -> None:
    api = _api()
    first = _shell_unit(api, raw=b"a")
    repeated = _shell_unit(api, raw=b"b")
    shifted = api.ShellUnit(
        dialect=api.ShellDialect.BASH,
        kind=api.ShellUnitKind.STANDALONE,
        provenance=api.SiteProvenance.FILE_SUFFIX,
        raw_bytes=b"b",
        origin_span=api.SourceSpan(
            "scripts/setup.sh",
            1,
            2,
            1,
            1,
            start_column=1,
            end_column=2,
        ),
    )
    different_origin_kind = api.ShellUnit(
        dialect=api.ShellDialect.BASH,
        kind=api.ShellUnitKind.NESTED_LITERAL,
        provenance=api.SiteProvenance.NESTED_LITERAL,
        raw_bytes=b"c",
        origin_span=first.origin_span,
    )

    assert first.unit_id == repeated.unit_id
    assert shifted.unit_id != first.unit_id
    assert different_origin_kind.unit_id != first.unit_id
    assert len(first.unit_id) == 32
    assert set(first.unit_id) <= set("0123456789abcdef")


@pytest.mark.parametrize("dialect", ["bash", "fish", object()])
def test_shell_unit_rejects_free_form_dialects(dialect: object) -> None:
    api = _api()

    with pytest.raises(ValueError, match="code-owned"):
        api.ShellUnit(
            dialect=dialect,
            kind=api.ShellUnitKind.STANDALONE,
            provenance=api.SiteProvenance.FILE_SUFFIX,
            raw_bytes=b"x",
            origin_span=api.SourceSpan(
                "setup",
                0,
                1,
                1,
                1,
                start_column=0,
                end_column=1,
            ),
        )


def test_static_values_enforce_exact_unknown_and_unbound_states() -> None:
    api = _api()

    assert api.StaticValue.exact(b"literal").state is api.StaticValueState.EXACT
    assert api.StaticValue.unknown().state is api.StaticValueState.UNKNOWN
    assert api.StaticValue.unbound().state is api.StaticValueState.UNBOUND
    with pytest.raises(ValueError):
        api.StaticValue(state=api.StaticValueState.UNKNOWN, exact_bytes=b"leak")
    with pytest.raises(ValueError):
        api.StaticValue(state=api.StaticValueState.EXACT, exact_bytes=None)


def test_source_map_maps_affine_ranges_and_refuses_gaps_or_non_affine_spans() -> None:
    api = _api()
    affine = api.SourceMap(
        path="docs/guide.md",
        entries=[api.SourceMapEntry(0, 4, 10, 14)],
        child_size_bytes=4,
        physical_size_bytes=30,
        physical_line_starts=(0, 10),
    )
    mapped = affine.map_range(1, 3)

    assert mapped == api.SourceSpan(
        "docs/guide.md",
        11,
        13,
        2,
        2,
        start_column=1,
        end_column=3,
    )

    non_affine = api.SourceMap(
        path="docs/guide.md",
        entries=[
            api.SourceMapEntry(0, 2, 10, 12),
            api.SourceMapEntry(2, 4, 20, 22),
        ],
        child_size_bytes=4,
        physical_size_bytes=30,
        physical_line_starts=(0, 10, 20),
    )
    assert non_affine.map_range(0, 1).start_byte == 10
    assert non_affine.map_range(0, 4) is None
    assert non_affine.map_range(4, 5) is None


@pytest.mark.parametrize(
    ("raw", "line_starts"),
    [
        ("é\n".encode(), (0, 3)),
        ("é\r\n".encode(), (0, 4)),
        ("é\r".encode(), (0, 3)),
    ],
)
def test_source_map_uses_inclusive_end_lines_at_physical_line_boundaries(
    raw: bytes,
    line_starts: tuple[int, ...],
) -> None:
    api = _api()
    source_map = api.SourceMap(
        path="docs/guide.md",
        entries=[api.SourceMapEntry(0, len(raw), 0, len(raw))],
        child_size_bytes=len(raw),
        physical_size_bytes=len(raw),
        physical_line_starts=line_starts,
    )

    assert source_map.map_range(0, len(raw)) == api.SourceSpan(
        "docs/guide.md",
        0,
        len(raw),
        1,
        1,
        start_column=0,
        end_column=len(raw),
    )


def test_source_map_preserves_multibyte_columns_after_an_empty_final_line_boundary() -> None:
    api = _api()
    raw = "x\né".encode()
    source_map = api.SourceMap(
        path="docs/guide.md",
        entries=[api.SourceMapEntry(0, len(raw), 0, len(raw))],
        child_size_bytes=len(raw),
        physical_size_bytes=len(raw),
        physical_line_starts=(0, 2),
    )

    assert source_map.map_range(2, 4) == api.SourceSpan(
        "docs/guide.md",
        2,
        4,
        2,
        2,
        start_column=0,
        end_column=2,
    )


@pytest.mark.parametrize(
    "entries",
    [
        [(0, 2, 10, 11)],
        [(0, 2, 10, 12), (1, 3, 20, 22)],
        [(2, 4, 10, 12), (0, 2, 20, 22)],
    ],
)
def test_source_map_rejects_non_affine_or_ambiguous_entries(
    entries: list[tuple[int, int, int, int]],
) -> None:
    api = _api()

    with pytest.raises(ValueError):
        api.SourceMap(
            path="docs/guide.md",
            entries=[api.SourceMapEntry(*entry) for entry in entries],
            child_size_bytes=4,
            physical_size_bytes=30,
            physical_line_starts=(0, 10, 20),
        )


def test_source_map_constructor_enforces_entry_and_canonical_domain_bounds() -> None:
    api = _api()
    entry = api.SourceMapEntry(0, 1, 0, 1)

    with pytest.raises(ValueError, match="entry limit"):
        api.SourceMap(
            path="docs/guide.md",
            entries=[entry] * 50_001,
            child_size_bytes=1,
            physical_size_bytes=1,
            physical_line_starts=(0,),
        )
    with pytest.raises(ValueError, match="canonical byte domain"):
        api.SourceMap(
            path="docs/guide.md",
            entries=[api.SourceMapEntry(0, 1, 10, 11)],
            child_size_bytes=1,
            physical_size_bytes=10,
            physical_line_starts=(0,),
        )


def test_source_map_composition_is_exact_and_fails_closed_across_parent_gaps() -> None:
    api = _api()
    child = api.SourceMap(
        path="docs/guide.md",
        entries=[api.SourceMapEntry(0, 4, 2, 6)],
        child_size_bytes=4,
        physical_size_bytes=6,
        physical_line_starts=(0,),
    )
    parent = api.SourceMap(
        path="docs/guide.md",
        entries=[api.SourceMapEntry(2, 6, 20, 24)],
        child_size_bytes=6,
        physical_size_bytes=30,
        physical_line_starts=(0, 20),
    )

    composed = child.compose(parent)

    assert composed.entries == (api.SourceMapEntry(0, 4, 20, 24),)
    assert composed.map_range(1, 3).start_byte == 21

    gapped_parent = api.SourceMap(
        path="docs/guide.md",
        entries=[
            api.SourceMapEntry(2, 3, 20, 21),
            api.SourceMapEntry(4, 6, 22, 24),
        ],
        child_size_bytes=6,
        physical_size_bytes=30,
        physical_line_starts=(0, 20),
    )
    with pytest.raises(ValueError, match="unmapped"):
        child.compose(gapped_parent)


@pytest.mark.timeout(10)
def test_source_map_lookup_and_composition_remain_bounded_at_the_50k_entry_limit() -> None:
    api = _api()
    entry_count = 50_000
    child = api.SourceMap(
        path="docs/large.md",
        entries=[
            api.SourceMapEntry(index * 2, index * 2 + 1, index * 2, index * 2 + 1)
            for index in range(entry_count)
        ],
        child_size_bytes=entry_count * 2 - 1,
        physical_size_bytes=entry_count * 2 - 1,
        physical_line_starts=(0,),
    )
    parent = api.SourceMap(
        path="docs/large.md",
        entries=[
            api.SourceMapEntry(index * 2, index * 2 + 1, index * 3, index * 3 + 1)
            for index in range(entry_count)
        ],
        child_size_bytes=entry_count * 2 - 1,
        physical_size_bytes=entry_count * 3 - 2,
        physical_line_starts=(0,),
    )

    composed = child.compose(parent)
    mapped_offsets = [
        composed.map_range(index * 2, index * 2 + 1).start_byte
        for _repeat in range(5)
        for index in range(entry_count)
    ]

    assert len(composed.entries) == entry_count
    assert mapped_offsets[0] == 0
    assert mapped_offsets[-1] == (entry_count - 1) * 3


def test_analysis_retains_only_sanitized_shell_work_and_issues() -> None:
    api = _api()
    unit = _shell_unit(api, raw=b"private-value")
    work = api.ShellWorkItem(
        unit_id=unit.unit_id,
        dialect=unit.dialect,
        kind=unit.kind,
        provenance=unit.provenance,
        span=unit.origin_span,
        outcome=api.ShellWorkOutcome.COMPLETED,
    )
    issue = api.ShellIssue(
        unit_id=unit.unit_id,
        reason=api.ShellIssueReason.UNSUPPORTED_SEMANTICS,
        outcome=api.ShellWorkOutcome.PARTIAL,
        span=unit.origin_span,
    )

    analysis = api.DependencySourceAnalysis(shell_work_items=[work], shell_issues=[issue])

    assert analysis.shell_work_items == (work,)
    assert analysis.shell_issues == (issue,)
    assert "private-value" not in repr(analysis)
    with pytest.raises(TypeError):
        api.DependencySourceAnalysis(shell_units=[unit])


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


def test_transient_candidate_is_sanitized_ranked_and_linked_to_its_shell_producer() -> None:
    api = _api()
    candidate = api.DependencySourceCandidate(
        ecosystem=api.DependencyEcosystem.NPM,
        surface=api.DependencySourceSurface.COMMAND,
        operation=api.DependencySourceOperation.SET,
        scope=api.DependencySourceScope.PROJECT,
        destination="https://packages.example.invalid/REDACTED_PATH",
        destination_status=api.DestinationStatus.RESOLVED,
        span=_span(api),
        producer_unit_id="a" * 32,
        rank=api.DependencyCandidateRank.RECOVERED,
        canonical_default=False,
    )

    assert candidate.producer_unit_id == "a" * 32
    assert candidate.rank is api.DependencyCandidateRank.RECOVERED
    assert candidate.canonical_default is False
    assert "REDACTED_PATH" in repr(candidate)
    assert {field.name for field in dataclasses.fields(api.DependencySourceCandidate)} == {
        "ecosystem",
        "surface",
        "operation",
        "scope",
        "destination",
        "destination_status",
        "span",
        "producer_unit_id",
        "rank",
        "canonical_default",
    }


def test_transient_candidate_rejects_invalid_producer_identity_and_unredacted_destination() -> None:
    api = _api()
    values = {
        "ecosystem": api.DependencyEcosystem.NPM,
        "surface": api.DependencySourceSurface.COMMAND,
        "operation": api.DependencySourceOperation.SET,
        "scope": api.DependencySourceScope.PROJECT,
        "destination": "https://packages.example.invalid",
        "destination_status": api.DestinationStatus.RESOLVED,
        "span": _span(api),
    }

    with pytest.raises(ValueError):
        api.DependencySourceCandidate(**values, producer_unit_id="attacker-controlled")
    unsafe_values = dict(values)
    unsafe_values["destination"] = "https://user:secret@packages.example.invalid/path?token=abc"
    with pytest.raises(ValueError):
        api.DependencySourceCandidate(**unsafe_values)


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


def test_resolved_destination_accepts_exact_bound_and_rejects_one_over() -> None:
    api = _api()
    suffix = "://x.invalid/"
    exact = "a" * (api.MAX_DEPENDENCY_DESTINATION_CHARACTERS - len(suffix)) + suffix
    accepted = api.SourceChange(
        ecosystem="npm",
        surface="source",
        operation="replace",
        scope="global",
        destination=exact,
        destination_status=api.DestinationStatus.RESOLVED,
        span=_span(api),
    )

    assert accepted.destination == exact

    with pytest.raises(ValueError):
        api.SourceChange(
            ecosystem="npm",
            surface="source",
            operation="replace",
            scope="global",
            destination=exact + "a",
            destination_status=api.DestinationStatus.RESOLVED,
            span=_span(api),
        )


def test_parse_and_analysis_results_freeze_iterables_as_tuples() -> None:
    api = _api()
    candidate = api.DependencySourceCandidate(
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

    parsed = api.DependencySourceParseResult(candidates=[candidate], limitations=[limitation])
    finding = Finding(rule_id="SC10", message="source changed")
    analysis = api.DependencySourceAnalysis(findings=[finding], limitations=[limitation])

    assert parsed.candidates == (candidate,)
    assert not hasattr(parsed, "changes")
    assert parsed.limitations == (limitation,)
    assert analysis.findings == (finding,)
    assert analysis.limitations == (limitation,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        parsed.candidates = ()


def test_analysis_keeps_one_validated_producer_identity_per_finding() -> None:
    api = _api()
    finding = Finding(rule_id="SC10", message="source changed")

    analysis = api.DependencySourceAnalysis(
        findings=[finding],
        finding_producer_unit_ids=["a" * 32],
    )

    assert analysis.finding_producer_unit_ids == ("a" * 32,)
    with pytest.raises(ValueError):
        api.DependencySourceAnalysis(
            findings=[finding],
            finding_producer_unit_ids=["a" * 32, "b" * 32],
        )
    with pytest.raises(ValueError):
        api.DependencySourceAnalysis(
            findings=[finding],
            finding_producer_unit_ids=["attacker-controlled"],
        )


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
    scan_resources = (
        api.DependencyWorkResource.CONFIG_NODES,
        api.DependencyWorkResource.RETAINED_LITERAL_BYTES,
        api.DependencyWorkResource.SOURCE_RECORDS,
        api.DependencyWorkResource.EMITTED_CHANGES,
        api.DependencyWorkResource.FINDING_OUTPUT_RECORDS,
        api.DependencyWorkResource.LEDGER_EVENTS,
        api.DependencyWorkResource.SHELL_PARSED_BYTES,
        api.DependencyWorkResource.RETAINED_SHELL_IR,
        api.DependencyWorkResource.SHELL_LOCALIZED_ISSUES,
    )
    before = {resource: budget.used(resource) for resource in scan_resources}

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


def test_dependency_output_reservation_charges_changes_findings_and_new_ledger_rows_atomically() -> (
    None
):
    api = _api()
    budget = api.DependencyWorkBudget(
        existing_finding_output_records=9_998,
        existing_ledger_events=9_997,
    )
    assert budget.charge_emitted_changes(9_998) is None

    exhaustion = budget.reserve_dependency_outputs(
        emitted_changes=2,
        finding_output_records=2,
        new_ledger_events=2,
    )

    assert exhaustion is None
    assert budget.used(api.DependencyWorkResource.EMITTED_CHANGES) == 10_000
    assert budget.used(api.DependencyWorkResource.FINDING_OUTPUT_RECORDS) == 10_000
    assert budget.used(api.DependencyWorkResource.LEDGER_EVENTS) == 9_999


def test_dependency_output_reservation_rolls_back_every_counter_when_ledger_capacity_denies() -> (
    None
):
    api = _api()
    budget = api.DependencyWorkBudget(
        existing_finding_output_records=9_998,
        existing_ledger_events=9_999,
    )
    assert budget.charge_emitted_changes(9_998) is None
    before = {
        resource: budget.used(resource)
        for resource in (
            api.DependencyWorkResource.EMITTED_CHANGES,
            api.DependencyWorkResource.FINDING_OUTPUT_RECORDS,
            api.DependencyWorkResource.LEDGER_EVENTS,
        )
    }

    exhaustion = budget.reserve_dependency_outputs(
        emitted_changes=1,
        finding_output_records=1,
        new_ledger_events=1,
    )

    assert exhaustion == api.DependencyWorkExhaustion(
        api.DependencyWorkResource.LEDGER_EVENTS,
        10_001,
        10_000,
    )
    assert {resource: budget.used(resource) for resource in before} == before


def test_dependency_output_reservation_does_not_charge_an_existing_producer_row() -> None:
    api = _api()
    budget = api.DependencyWorkBudget(
        existing_finding_output_records=9_999,
        existing_ledger_events=9_999,
    )
    before_ledger = budget.used(api.DependencyWorkResource.LEDGER_EVENTS)

    exhaustion = budget.reserve_dependency_outputs(
        emitted_changes=1,
        finding_output_records=1,
        new_ledger_events=0,
    )

    assert exhaustion is None
    assert budget.used(api.DependencyWorkResource.EMITTED_CHANGES) == 1
    assert budget.used(api.DependencyWorkResource.FINDING_OUTPUT_RECORDS) == 10_000
    assert budget.used(api.DependencyWorkResource.LEDGER_EVENTS) == before_ledger


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


@pytest.mark.parametrize(
    ("method_name", "resource", "limit"),
    [
        ("charge_shell_units", "shell_units", 256),
        ("charge_source_map_entries", "shell_source_map_entries", 50_000),
    ],
)
def test_shell_file_counters_accept_exact_limit_and_reject_one_over(
    method_name: str,
    resource: str,
    limit: int,
) -> None:
    api = _api()
    file_budget = api.DependencyWorkBudget().for_file("scripts/setup.sh")
    charge: Callable[[int], Any] = getattr(file_budget, method_name)

    assert charge(limit) is None
    exhaustion = charge(1)

    assert exhaustion == api.DependencyWorkExhaustion(
        api.DependencyWorkResource(resource), limit + 1, limit
    )
    assert file_budget.used(api.DependencyWorkResource(resource)) == limit


def test_shell_parse_reservation_atomically_charges_calls_file_revisits_and_scan_bytes() -> None:
    api = _api()
    budget = api.DependencyWorkBudget()
    file_budget = budget.for_file("scripts/setup.sh")
    file_budget.register_shell_file_size(4)

    assert file_budget.reserve_shell_parse(4) is None
    assert file_budget.reserve_shell_parse(4) is None
    before = {
        "calls": file_budget.used(api.DependencyWorkResource.SHELL_PARSER_CALLS),
        "file_bytes": file_budget.used(api.DependencyWorkResource.SHELL_PARSED_REVISIT_BYTES),
        "scan_bytes": budget.used(api.DependencyWorkResource.SHELL_PARSED_BYTES),
    }

    exhaustion = file_budget.reserve_shell_parse(1)

    assert exhaustion == api.DependencyWorkExhaustion(
        api.DependencyWorkResource.SHELL_PARSED_REVISIT_BYTES, 9, 8
    )
    assert {
        "calls": file_budget.used(api.DependencyWorkResource.SHELL_PARSER_CALLS),
        "file_bytes": file_budget.used(api.DependencyWorkResource.SHELL_PARSED_REVISIT_BYTES),
        "scan_bytes": budget.used(api.DependencyWorkResource.SHELL_PARSED_BYTES),
    } == before


def test_shell_parser_call_limit_is_persistent_for_zero_byte_units() -> None:
    api = _api()
    file_budget = api.DependencyWorkBudget().for_file("scripts/empty.sh")
    file_budget.register_shell_file_size(0)

    for _ in range(512):
        assert file_budget.reserve_shell_parse(0) is None
    exhaustion = file_budget.reserve_shell_parse(0)

    assert exhaustion.resource is api.DependencyWorkResource.SHELL_PARSER_CALLS
    assert (exhaustion.observed, exhaustion.limit) == (513, 512)
    assert file_budget.used(api.DependencyWorkResource.SHELL_PARSER_CALLS) == 512


def test_scan_parsed_byte_limit_is_shared_across_files_without_mutating_denied_file() -> None:
    api = _api()
    budget = api.DependencyWorkBudget()
    for index in range(3):
        file_budget = budget.for_file(f"scripts/{index}.sh")
        file_budget.register_shell_file_size(1_000_000)
        assert file_budget.reserve_shell_parse(1_000_000) is None
        assert file_budget.reserve_shell_parse(1_000_000) is None

    denied = budget.for_file("scripts/denied.sh")
    denied.register_shell_file_size(1)
    exhaustion = denied.reserve_shell_parse(1)

    assert exhaustion == api.DependencyWorkExhaustion(
        api.DependencyWorkResource.SHELL_PARSED_BYTES,
        6_000_001,
        6_000_000,
    )
    assert denied.used(api.DependencyWorkResource.SHELL_PARSER_CALLS) == 0
    assert denied.used(api.DependencyWorkResource.SHELL_PARSED_REVISIT_BYTES) == 0


def test_shell_cst_visits_and_nested_depth_persist_by_path_and_opaque_unit_id() -> None:
    api = _api()
    budget = api.DependencyWorkBudget()
    unit = _shell_unit(api, raw=b"abcd")
    file_budget = budget.for_file(unit.origin_span.path)
    visit_limit = 12 * len(unit.raw_bytes) + 1_024

    assert file_budget.charge_shell_cst_visits(unit, visit_limit) is None
    visit_exhaustion = budget.for_file("scripts/setup.sh").charge_shell_cst_visits(unit, 1)
    assert visit_exhaustion == api.DependencyWorkExhaustion(
        api.DependencyWorkResource.SHELL_CST_VISITS,
        visit_limit + 1,
        visit_limit,
    )
    assert file_budget.observe_shell_nested_depth(unit, 2) is None
    depth_exhaustion = file_budget.observe_shell_nested_depth(unit, 3)
    assert depth_exhaustion == api.DependencyWorkExhaustion(
        api.DependencyWorkResource.SHELL_NESTED_DEPTH,
        3,
        2,
    )


def test_retained_shell_ir_is_scan_wide_and_keeps_per_unit_accounting() -> None:
    api = _api()
    budget = api.DependencyWorkBudget()
    first = _shell_unit(api, path="scripts/first.sh")
    second = _shell_unit(api, path="scripts/second.sh")

    assert budget.for_file(first.origin_span.path).charge_retained_shell_ir(first, 30_000) is None
    assert budget.for_file(second.origin_span.path).charge_retained_shell_ir(second, 20_000) is None
    exhaustion = budget.for_file(first.origin_span.path).charge_retained_shell_ir(first, 1)

    assert exhaustion.resource is api.DependencyWorkResource.RETAINED_SHELL_IR
    assert budget.used(api.DependencyWorkResource.RETAINED_SHELL_IR) == 50_000
    assert (
        budget.for_file(first.origin_span.path).used_for_unit(
            first, api.DependencyWorkResource.RETAINED_SHELL_IR
        )
        == 30_000
    )


def test_shell_value_bytes_charge_global_and_file_limits_atomically() -> None:
    api = _api()
    budget = api.DependencyWorkBudget()
    first = budget.for_file("scripts/first.sh")
    second = budget.for_file("scripts/second.sh")

    assert first.reserve_shell_value_bytes(1_999_999) is None
    assert second.reserve_shell_value_bytes(1) is None
    before = {
        "global": budget.used(api.DependencyWorkResource.RETAINED_LITERAL_BYTES),
        "file": second.used(api.DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES),
    }
    exhaustion = second.reserve_shell_value_bytes(1)

    assert exhaustion.resource is api.DependencyWorkResource.RETAINED_LITERAL_BYTES
    assert {
        "global": budget.used(api.DependencyWorkResource.RETAINED_LITERAL_BYTES),
        "file": second.used(api.DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES),
    } == before


def test_shell_value_file_limit_is_independently_typed_and_atomic() -> None:
    api = _api()
    budget = api.DependencyWorkBudget()
    file_budget = budget.for_file("scripts/setup.sh")

    assert file_budget.reserve_shell_value_bytes(2_000_000) is None
    before = {
        "global": budget.used(api.DependencyWorkResource.RETAINED_LITERAL_BYTES),
        "file": file_budget.used(api.DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES),
    }
    exhaustion = file_budget.reserve_shell_value_bytes(1)

    assert exhaustion.resource is api.DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES
    assert {
        "global": budget.used(api.DependencyWorkResource.RETAINED_LITERAL_BYTES),
        "file": file_budget.used(api.DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES),
    } == before


def test_shell_issue_budget_reserves_exactly_one_truncation_issue() -> None:
    api = _api()
    budget = api.DependencyWorkBudget()

    assert budget.charge_shell_issues(9_999) is None
    exhaustion = budget.charge_shell_issues(1)

    assert exhaustion.resource is api.DependencyWorkResource.SHELL_LOCALIZED_ISSUES
    assert budget.used(api.DependencyWorkResource.SHELL_LOCALIZED_ISSUES) == 9_999
    assert budget.claim_reserved_shell_truncation_issue() is api.ShellTruncationClaimStatus.CLAIMED
    assert budget.used(api.DependencyWorkResource.SHELL_LOCALIZED_ISSUES) == 10_000
    assert (
        budget.claim_reserved_shell_truncation_issue()
        is api.ShellTruncationClaimStatus.ALREADY_CLAIMED
    )
    one_over = budget.charge_shell_issues(1)
    assert one_over == api.DependencyWorkExhaustion(
        api.DependencyWorkResource.SHELL_LOCALIZED_ISSUES,
        10_001,
        10_000,
    )


def test_shell_only_charges_do_not_mutate_pr1_config_output_or_ledger_counters() -> None:
    api = _api()
    budget = api.DependencyWorkBudget()
    before = {
        resource: budget.used(resource)
        for resource in (
            api.DependencyWorkResource.CONFIG_NODES,
            api.DependencyWorkResource.SOURCE_RECORDS,
            api.DependencyWorkResource.EMITTED_CHANGES,
            api.DependencyWorkResource.FINDING_OUTPUT_RECORDS,
            api.DependencyWorkResource.LEDGER_EVENTS,
        )
    }
    unit = _shell_unit(api)
    file_budget = budget.for_file(unit.origin_span.path)
    file_budget.register_shell_file_size(1)

    assert file_budget.charge_shell_units(1) is None
    assert file_budget.reserve_shell_parse(1) is None
    assert file_budget.charge_shell_cst_visits(unit, 1) is None
    assert file_budget.charge_source_map_entries(1) is None
    assert file_budget.charge_retained_shell_ir(unit, 1) is None
    assert budget.charge_shell_issues(1) is None

    assert {resource: budget.used(resource) for resource in before} == before
