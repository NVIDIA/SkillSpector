# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task 5.2: canonical recursive bounds, independent of report plans.

Exercise the real collector, accounting walker and serializers with deterministic
analyzer results and a manually advanced clock. The standalone emission
coordinator lets these acceptance tests cover additional-format plans without
depending on the still-separate CLI emission integration tasks.
"""

import importlib
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from skillspector import cli
from skillspector.models import Finding
from skillspector.multi_skill import MultiSkillDetectionResult, SkillDirectory
from skillspector.nodes.report import (
    PreparedReport,
    prepare_report,
    report_rendering_is_deferred,
)
from skillspector.recursive_reports import prepared_payload_usage
from skillspector.report_emission import emit_report_outputs
from skillspector.report_outputs import build_report_outputs, parse_report_declarations
from skillspector.state import MAX_WORKFLOW_BYTES, MAX_WORKFLOW_SECONDS
from skillspector.suppression import SuppressedFinding

report_module = importlib.import_module("skillspector.nodes.report")
FORMATS = tuple(cli.FormatChoice)
ADDITIONAL_PLANS = (
    (),
    FORMATS,
    tuple(reversed(FORMATS)),
    (
        cli.FormatChoice.json,
        cli.FormatChoice.sarif,
        cli.FormatChoice.json,
        cli.FormatChoice.markdown,
        cli.FormatChoice.sarif,
    ),
)


@dataclass
class Clock:
    now: float = 100.0

    def __call__(self) -> float:
        return self.now


def _detection(tmp_path: Path, count: int = 4) -> MultiSkillDetectionResult:
    return MultiSkillDetectionResult(
        is_multi_skill=True,
        skills=[SkillDirectory(tmp_path / f"s{i}", f"s{i}", f"s{i}") for i in range(count)],
    )


def _result(name: str, occurrences: int = 3) -> dict[str, object]:
    finding = Finding(
        finding_id=f"finding-{name}",
        rule_id="TEST-CANONICAL",
        message="fixed result",
        severity="LOW",
        occurrences=[{"file": "SKILL.md", "start_line": i + 1} for i in range(occurrences)],
    )
    prepared = prepare_report(
        {
            "manifest": {"name": name},
            "findings": [finding],
            "use_llm": False,
            "analysis_completeness": {
                "is_complete": True,
                "execution_successful": True,
                "coverage_percent": 100,
            },
        }
    )
    return {**prepared.public_fields, "prepared_report": prepared}


def _usage(prepared: PreparedReport) -> tuple[int, int]:
    return prepared_payload_usage(
        prepared,
        max_bytes=MAX_WORKFLOW_BYTES,
        max_records=1_000_000,
    )


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    controlled = Clock()
    monkeypatch.setattr(cli, "monotonic", controlled)
    return controlled


def test_initial_caps_preserve_existing_workflow_and_record_limits() -> None:
    assert cli._MULTI_SKILL_MAX_PREPARED_BYTES == MAX_WORKFLOW_BYTES
    assert cli._MULTI_SKILL_MAX_METADATA_RECORDS == cli._MULTI_SKILL_MAX_PUBLIC_RECORDS
    assert cli._MULTI_SKILL_MAX_PUBLIC_RECORDS == 10_000
    assert cli._MULTI_SKILL_MAX_SKILLS == 32
    assert cli._TransitiveBudget().max_seconds == MAX_WORKFLOW_SECONDS


@pytest.mark.parametrize("bound", ["bytes", "metadata", "occurrences"])
@pytest.mark.parametrize("delta", [-1, 0, 1])
@pytest.mark.parametrize("primary", FORMATS)
@pytest.mark.parametrize(
    "additional", ADDITIONAL_PLANS, ids=["sole", "all", "reordered", "repeated"]
)
def test_canonical_boundary_selection_is_invariant_under_output_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clock: Clock,
    bound: str,
    delta: int,
    primary: cli.FormatChoice,
    additional: tuple[cli.FormatChoice, ...],
) -> None:
    results = [_result(f"s{i}") for i in range(4)]
    first_bytes, first_metadata = _usage(results[0]["prepared_report"])
    # Equal-length deterministic identities make all four child costs identical.
    assert [_usage(result["prepared_report"]) for result in results] == [
        (first_bytes, first_metadata)
    ] * 4
    cap_name, per_child = {
        "bytes": ("_MULTI_SKILL_MAX_PREPARED_BYTES", first_bytes),
        "metadata": ("_MULTI_SKILL_MAX_METADATA_RECORDS", first_metadata),
        "occurrences": ("_MULTI_SKILL_MAX_PUBLIC_RECORDS", 3),
    }[bound]
    monkeypatch.setattr(cli, cap_name, 2 * per_child + delta)
    analyzed = []
    traversals = []
    caches = []
    collecting = True
    real_serializer = report_module.serialize_report

    def scan(**kwargs):
        assert collecting
        assert report_rendering_is_deferred()
        name = Path(kwargs["input_path"]).name
        analyzed.append(name)
        traversals.append(kwargs["transitive_traversal"])
        caches.append(kwargs["transitive_cache"])
        return results[int(name[1:])]

    def serialize(*args):
        assert not collecting, "Collection must not serialize even to estimate retention"
        # Arbitrary presentation time cannot consume the completed scan's deadline.
        clock.now += 10 * MAX_WORKFLOW_SECONDS
        return real_serializer(*args)

    monkeypatch.setattr(cli, "_scan_skill", scan)
    monkeypatch.setattr(report_module, "serialize_report", serialize)
    collection = cli._collect_recursive_reports(
        _detection(tmp_path),
        primary,
        None,
        no_llm=True,
    )
    collecting = False
    expected_analyzed = ["s0", "s1"] if delta <= 0 else ["s0", "s1", "s2"]
    expected_retained = ["s0"] if delta < 0 else ["s0", "s1"]
    assert analyzed == expected_analyzed
    assert [child.skill.name for child in collection.children] == expected_retained
    assert collection.scanned_skill_count == len(expected_analyzed)
    assert collection.unscanned_skill_count == 4 - len(expected_analyzed)
    assert not collection.execution_failed
    assert all(traversal is traversals[0] for traversal in traversals)
    assert all(cache is caches[0] for cache in caches)
    assert traversals[0].budget.max_bytes == MAX_WORKFLOW_BYTES
    assert collection.retained_prepared_bytes == len(expected_retained) * first_bytes
    assert collection.retained_metadata_records == len(expected_retained) * first_metadata
    assert collection.retained_public_records == len(expected_retained) * 3
    reason = {
        "bytes": "prepared_payload_limit",
        "metadata": "metadata_record_limit",
        "occurrences": "recursive_finding_record_limit",
    }[bound]
    assert collection.output_omitted_skill_count == (0 if delta == 0 else 1)
    if delta != 0:
        assert reason in collection.retention_completeness["omission_reasons"]
    before = deepcopy(collection)
    snapshots = [child.prepared for child in collection.children]
    rendered_formats = []

    def render(output_format):
        rendered_formats.append(output_format)
        return cli._render_recursive_report(collection, output_format)

    plan = build_report_outputs(
        primary,
        None,
        parse_report_declarations(
            [
                f"{output_format.value}={tmp_path / f'report-{i}'}"
                for i, output_format in enumerate(additional)
            ]
        ),
    )
    emission = emit_report_outputs(
        plan,
        render,
        primary_writer=lambda result, entry: None,
        additional_writer=lambda path, body: None,
        diagnostic=lambda message: None,
    )
    assert not emission.output_failed
    assert rendered_formats == [primary, *additional]
    assert analyzed == expected_analyzed
    assert collection == before
    assert all(
        child.prepared is snapshot
        for child, snapshot in zip(
            collection.children,
            snapshots,
            strict=True,
        )
    )


@pytest.mark.parametrize("bound", ["source", "time", "skill"])
@pytest.mark.parametrize("delta", [-1, 0, 1])
@pytest.mark.parametrize("primary", FORMATS)
def test_shared_analysis_budget_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clock: Clock,
    bound: str,
    delta: int,
    primary: cli.FormatChoice,
) -> None:
    analyzed = []
    traversals = []
    max_skills = 2 + delta if bound == "skill" else None
    if bound == "source":
        monkeypatch.setattr(cli, "MAX_WORKFLOW_BYTES", 4 + delta)
    if bound == "time":
        real_budget = cli._TransitiveBudget
        monkeypatch.setattr(
            cli,
            "_TransitiveBudget",
            lambda **kwargs: real_budget(max_seconds=4 + delta, **kwargs),
        )

    def scan(**kwargs):
        traversal = kwargs["transitive_traversal"]
        traversals.append(traversal)
        name = Path(kwargs["input_path"]).name
        analyzed.append(name)
        # Controlled analyzer consumption, using the actual shared budget API.
        if bound == "source":
            traversal.record_bytes(2)
        if bound == "time":
            clock.now += 2
        return _result(name)

    monkeypatch.setattr(cli, "_scan_skill", scan)
    collection = cli._collect_recursive_reports(
        _detection(tmp_path),
        primary,
        None,
        no_llm=True,
        max_skills=max_skills,
    )
    count = 2 + delta if bound == "skill" else 2 if delta <= 0 else 3
    assert analyzed == [f"s{i}" for i in range(count)]
    assert [child.skill.name for child in collection.children] == analyzed
    assert collection.scanned_skill_count == count
    assert collection.unscanned_skill_count == 4 - count
    assert collection.analysis_completeness["entirely_uninspected_files"] == 4 - count
    assert not collection.analysis_completeness["is_complete"]
    assert collection.retention_completeness["is_complete"]
    assert all(traversal is traversals[0] for traversal in traversals)
    if bound == "source":
        assert traversals[0].scanned_bytes == count * 2
        assert traversals[0].budget.max_bytes == 4 + delta
    if bound == "time":
        assert clock.now - traversals[0].started_at == count * 2


def test_default_skill_boundary_keeps_exactly_32_children(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clock: Clock,
) -> None:
    analyzed = []

    def scan(**kwargs):
        name = Path(kwargs["input_path"]).name
        analyzed.append(name)
        return _result(name, occurrences=0)

    monkeypatch.setattr(cli, "_scan_skill", scan)
    collection = cli._collect_recursive_reports(
        _detection(tmp_path, 33),
        cli.FormatChoice.json,
        None,
        no_llm=True,
    )
    assert analyzed == [f"s{i}" for i in range(32)]
    assert [child.skill.name for child in collection.children] == analyzed
    assert collection.unscanned_skill_count == 1


@pytest.mark.parametrize("bound", ["bytes", "metadata", "occurrences", "source", "time", "skill"])
@pytest.mark.parametrize("delta", [-1, 0, 1])
@pytest.mark.parametrize("primary", FORMATS)
@pytest.mark.parametrize(
    "additional", ADDITIONAL_PLANS, ids=["sole", "all", "reordered", "repeated"]
)
def test_real_cli_report_arguments_do_not_change_bounded_child_identities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clock: Clock,
    bound: str,
    delta: int,
    primary: cli.FormatChoice,
    additional: tuple[cli.FormatChoice, ...],
) -> None:
    """Cover the actual parser/orchestrator, not just a hypothetical output plan.

    Additional-file emission belongs to other tasks; this assertion deliberately
    concerns only 5.2's analyzed/retained identities and shared budget accounting.
    """
    root = tmp_path / "skills"
    root.mkdir()
    for i in range(4):
        child = root / f"s{i}"
        child.mkdir()
        (child / "SKILL.md").write_text(f"---\nname: s{i}\n---\n# Fixed skill\n")
    results = [_result(f"s{i}") for i in range(4)]
    byte_cost, metadata_cost = _usage(results[0]["prepared_report"])
    caps = {
        "bytes": ("_MULTI_SKILL_MAX_PREPARED_BYTES", byte_cost),
        "metadata": ("_MULTI_SKILL_MAX_METADATA_RECORDS", metadata_cost),
        "occurrences": ("_MULTI_SKILL_MAX_PUBLIC_RECORDS", 3),
    }
    if bound in caps:
        cap_name, cost = caps[bound]
        monkeypatch.setattr(cli, cap_name, 2 * cost + delta)
    elif bound == "source":
        monkeypatch.setattr(cli, "MAX_WORKFLOW_BYTES", 4 + delta)
    elif bound == "time":
        real_budget = cli._TransitiveBudget
        monkeypatch.setattr(
            cli,
            "_TransitiveBudget",
            lambda **kwargs: real_budget(max_seconds=4 + delta, **kwargs),
        )
    analyzed = []
    collected = []
    real_collect = cli._collect_recursive_reports

    def scan(**kwargs):
        assert report_rendering_is_deferred()
        name = Path(kwargs["input_path"]).name
        analyzed.append(name)
        if bound == "source":
            kwargs["transitive_traversal"].record_bytes(2)
        if bound == "time":
            clock.now += 2
        return results[int(name[1:])]

    def collect(*args, **kwargs):
        collection = real_collect(*args, **kwargs)
        # The orchestrator legitimately releases snapshots after output.
        collected.append(
            (
                [child.skill.name for child in collection.children],
                collection.scanned_skill_count,
                collection.retained_public_records,
            )
        )
        return collection

    monkeypatch.setattr(cli, "_scan_skill", scan)
    monkeypatch.setattr(cli, "_collect_recursive_reports", collect)
    arguments = [
        "scan",
        str(root),
        "--recursive",
        "--no-llm",
        "--format",
        primary.value,
    ]
    if bound == "skill":
        arguments.extend(["--max-skills", str(2 + delta)])
    for i, output_format in enumerate(additional):
        arguments.extend(["--report", f"{output_format.value}={tmp_path / f'report-{i}'}"])
    invocation = CliRunner().invoke(cli.app, arguments)
    assert invocation.exit_code == 0, invocation.output
    if bound in caps:
        analyzed_count = 2 if delta <= 0 else 3
        retained_count = 1 if delta < 0 else 2
    else:
        analyzed_count = 2 + delta if bound == "skill" else 2 if delta <= 0 else 3
        retained_count = analyzed_count
    assert analyzed == [f"s{i}" for i in range(analyzed_count)]
    assert collected == [
        (
            [f"s{i}" for i in range(retained_count)],
            analyzed_count,
            3 * retained_count,
        )
    ]


@pytest.mark.parametrize("show_suppressed", [False, True])
def test_public_budget_counts_active_and_hidden_suppressed_occurrences_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clock: Clock,
    show_suppressed: bool,
) -> None:
    result = _result("s0", occurrences=3)
    prepared = result["prepared_report"]
    suppressed_finding = Finding(
        finding_id="suppressed-s0",
        rule_id="SUPPRESSED",
        message="hidden finding",
        occurrences=[
            {"file": "helper.py", "start_line": 1},
            {"file": "helper.py", "start_line": 2},
        ],
    )
    suppressed = [SuppressedFinding(suppressed_finding, "baseline")]
    public = {
        **prepared.public_fields,
        "filtered_findings": [*prepared.reported_findings, suppressed_finding],
        "suppressed_findings": suppressed,
    }
    prepared = replace(
        prepared,
        public_fields=public,
        reported_findings=public["filtered_findings"],
        suppressed=suppressed,
        show_suppressed=show_suppressed,
    )
    result = {**public, "prepared_report": prepared}
    assert cli._multi_skill_public_record_count(result) == 5
    monkeypatch.setattr(cli, "_scan_skill", lambda **kwargs: result)
    monkeypatch.setattr(cli, "_MULTI_SKILL_MAX_PUBLIC_RECORDS", 5)
    collection = cli._collect_recursive_reports(
        _detection(tmp_path, 1),
        cli.FormatChoice.json,
        None,
        no_llm=True,
        show_suppressed=show_suppressed,
    )
    assert collection.retained_public_records == 5
    assert len(collection.children) == 1
    monkeypatch.setattr(cli, "_MULTI_SKILL_MAX_PUBLIC_RECORDS", 4)
    overflow = cli._collect_recursive_reports(
        _detection(tmp_path, 1),
        cli.FormatChoice.json,
        None,
        no_llm=True,
        show_suppressed=show_suppressed,
    )
    assert not overflow.children
    assert overflow.retention_completeness["omitted_records"] == 5
    assert overflow.scanned_skill_count == 1


@pytest.mark.parametrize("occurrences", [0, 1, 4])
def test_canonical_public_records_are_occurrences_not_finding_objects(occurrences: int) -> None:
    result = _result("s0", occurrences=occurrences)
    assert len(result["filtered_findings"]) == 1
    assert cli._multi_skill_public_record_count(result) == max(1, occurrences)


def test_payload_uses_utf8_bytes_not_characters_or_serialized_escaping() -> None:
    prepared = _result("s0")["prepared_report"]
    ascii_payload = replace(prepared, manifest={"name": "a" * 20})
    unicode_payload = replace(prepared, manifest={"name": "é" * 20})
    ascii_bytes, ascii_records = _usage(ascii_payload)
    unicode_bytes, unicode_records = _usage(unicode_payload)
    assert unicode_bytes - ascii_bytes == 20
    assert unicode_records == ascii_records
    escaped_payload = replace(prepared, manifest={"name": '"' * 20})
    assert _usage(escaped_payload) == (ascii_bytes, ascii_records)


@pytest.mark.parametrize(
    "field",
    [
        "component_metadata",
        "llm_call_log",
        "inference_usage",
        "structured_summaries",
    ],
)
def test_metadata_collections_and_empty_records_have_canonical_cost(field: str) -> None:
    prepared = _result("s0")["prepared_report"]
    empty = replace(prepared, **{field: []})
    populated = replace(prepared, **{field: [{}, {}, {}]})
    empty_bytes, empty_records = _usage(empty)
    populated_bytes, populated_records = _usage(populated)
    assert populated_records - empty_records == 3
    assert populated_bytes - empty_bytes == 3 * 16


def test_metadata_string_lists_charge_entries_even_when_text_is_empty() -> None:
    prepared = _result("s0")["prepared_report"]
    empty = replace(prepared, transitive_truncation_reasons=[])
    populated = replace(prepared, transitive_truncation_reasons=["", "", ""])
    empty_bytes, empty_records = _usage(empty)
    populated_bytes, populated_records = _usage(populated)
    assert populated_records - empty_records == 3
    assert populated_bytes == empty_bytes


def test_aliases_are_charged_once_but_distinct_metadata_records_are_not() -> None:
    prepared = _result("s0")["prepared_report"]
    row = {"label": "fixed"}
    shared = replace(prepared, component_metadata=[row], structured_summaries=[row])
    distinct = replace(prepared, component_metadata=[row], structured_summaries=[dict(row)])
    shared_bytes, shared_records = _usage(shared)
    distinct_bytes, distinct_records = _usage(distinct)
    assert distinct_records - shared_records == 1
    assert distinct_bytes - shared_bytes == 16 + len("label") + len("fixed")


@pytest.mark.parametrize("bound", ["bytes", "records"])
def test_payload_walker_accepts_exact_limit_and_signals_one_unit_overflow(bound: str) -> None:
    prepared = _result("s0")["prepared_report"]
    byte_count, record_count = _usage(prepared)
    assert prepared_payload_usage(
        prepared,
        max_bytes=byte_count,
        max_records=record_count,
    ) == (byte_count, record_count)
    measured_bytes, measured_records = prepared_payload_usage(
        prepared,
        max_bytes=byte_count - (bound == "bytes"),
        max_records=record_count - (bound == "records"),
    )
    assert (
        measured_bytes > byte_count - 1 if bound == "bytes" else measured_records > record_count - 1
    )


def test_payload_walker_does_not_dump_models_or_serialize_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import BaseModel

    class Metadata(BaseModel):
        text: str

    prepared = replace(
        _result("s0")["prepared_report"],
        llm_provenance=Metadata(text="é"),
    )
    expected = _usage(prepared)

    def fail(*args, **kwargs):
        pytest.fail("Canonical accounting must not construct a serialized projection")

    monkeypatch.setattr(BaseModel, "model_dump", fail)
    monkeypatch.setattr(Finding, "to_dict", fail)
    monkeypatch.setattr(report_module, "serialize_report", fail)
    assert _usage(prepared) == expected


def test_unknown_prepared_metadata_fails_closed() -> None:
    prepared = replace(_result("s0")["prepared_report"], llm_provenance=object())
    byte_count, _ = prepared_payload_usage(prepared, max_bytes=100_000, max_records=10_000)
    assert byte_count > 100_000
