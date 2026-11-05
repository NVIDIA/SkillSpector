# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task 4.2 acceptance regressions at the real CLI/traversal/emission boundary."""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
from copy import deepcopy
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from skillspector import cli, dependency_inspection, transitive
from skillspector.inspection_ledger import (
    LedgerOutcome,
    LedgerReason,
    LedgerRecordType,
    analyzer_status_for_events,
    finalize_ledger,
    ledger_event,
)
from skillspector.models import Finding
from skillspector.nodes.report import PreparedReport
from skillspector.report_sources import current_source_inventory
from skillspector.sarif_models import validate_sarif_report
from skillspector.state import SkillspectorState

report_module = importlib.import_module("skillspector.nodes.report")
FORMATS = ("terminal", "json", "markdown", "sarif")
TARGET = "https://github.com/acceptance/shared-budget-child"
SECOND_TARGET = "https://github.com/acceptance/second-budget-child"
BLOB = b"controlled locked artifact"
EXPANDED_BYTES = 7
CHARGE = len(BLOB) + EXPANDED_BYTES


@pytest.fixture
def budget_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Mock acquisition/analyzer results, not traversal, inspection or rendering."""
    clock = SimpleNamespace(now=100.0)
    for module in (cli, dependency_inspection):
        monkeypatch.setattr(module, "monotonic", lambda: clock.now)
    for name in (
        "extract_external_refs_with_metadata",
        "plan_transitive_targets_with_metadata",
        "BoundedTransitiveFrontier",
    ):
        monkeypatch.setattr(
            transitive, name, partial(getattr(transitive, name), clock=lambda: clock.now)
        )
    date = Mock(wraps=datetime)
    date.now.return_value = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(report_module, "datetime", date)
    monkeypatch.setattr(cli.err_console, "_force_terminal", False)
    provider = Mock(side_effect=AssertionError("offline fixture must not resolve a provider"))
    monkeypatch.setattr(report_module, "is_llm_available", provider)
    real_budget = cli._TransitiveBudget
    real_traversal = cli._TransitiveTraversalState
    real_report = report_module.report
    real_inspect = dependency_inspection.inspect_dependencies
    real_cleanup = cli.cleanup_result
    real_serialize = report_module.serialize_report
    real_prepare = report_module.prepare_report

    def make(*, recursive=False, second_target=False, limits=None, child_seconds=0.0):
        index = len(list(tmp_path.glob("case-*")))
        home = tmp_path / f"case-{index}"
        home.mkdir()
        root = home / "input"
        root.mkdir()
        roots = [root / "a", root / "b"] if recursive else [root]
        refs = TARGET + ("\n" + SECOND_TARGET if second_target else "")
        raw_lock = json.dumps(
            {
                "lockfileVersion": 3,
                "packages": {
                    "": {},
                    "node_modules/fixture": {
                        "version": "1.0.0",
                        "resolved": "https://registry.npmjs.org/fixture/-/fixture-1.0.0.tgz",
                        "integrity": "sha512-"
                        + base64.b64encode(hashlib.sha512(BLOB).digest()).decode(),
                    },
                },
            }
        ).encode()
        for local in roots:
            local.mkdir(exist_ok=True)
            (local / "SKILL.md").write_text(
                f"---\nname: {local.name}\n---\n{refs}\n", encoding="utf-8"
            )
            (local / "package-lock.json").write_bytes(raw_lock)
        resolved = {}
        for i, target in enumerate((TARGET, SECOND_TARGET)):
            local = home / f"resolved-{i}"
            local.mkdir()
            (local / "SKILL.md").write_text("# Child\n", encoding="utf-8")
            (local / "package-lock.json").write_bytes(raw_lock)
            resolved[target] = local

        case = SimpleNamespace(
            home=home,
            root=root,
            roots=roots,
            resolved=resolved,
            recursive=recursive,
            calls=[],
            traversals=[],
            inspections=[],
            downloads=[],
            workers=[],
            prepared=[],
            cleanup=[],
            late=None,
            clock=clock,
            provider=provider,
        )
        clock.now = 100.0

        def budget(**kwargs):
            return real_budget(**{**kwargs, **(limits or {})})

        monkeypatch.setattr(cli, "_TransitiveBudget", budget)
        monkeypatch.setattr(
            cli,
            "_TransitiveTraversalState",
            lambda **kwargs: real_traversal(**{"budget": budget(), **kwargs}),
        )

        def download(artifact, *, max_bytes, deadline, usage):
            assert max_bytes >= len(BLOB)
            assert deadline > clock.now
            case.downloads.append((artifact.url, max_bytes, deadline))
            usage["bytes"] += len(BLOB)
            return BLOB

        def worker(artifact, blob, prefix, worker_limits, timeout):
            assert blob == BLOB
            assert timeout > 0
            case.workers.append((worker_limits.copy(), timeout))
            return {
                "expanded_bytes": EXPANDED_BYTES,
                "members": 1,
                "limitations": [],
                "files": [
                    {
                        "path": prefix + "/index.js",
                        "size_bytes": EXPANDED_BYTES,
                        "disposition": "analyzed",
                        "findings": [],
                    }
                ],
            }

        monkeypatch.setattr(dependency_inspection, "_download_artifact", download)
        monkeypatch.setattr(dependency_inspection, "_run_worker", worker)

        def invoke(state, config=None):
            source = str(state["input_path"])
            local = resolved.get(source, Path(source))
            traversal = state["transitive_traversal_state"]
            assert state.get("inspect_dependencies") is True
            case.calls.append(source)
            case.traversals.append(traversal)
            traversal.remaining_seconds()
            clock.now += 1.0
            content = "# Child\n" if source in resolved else refs
            finding = Finding(
                rule_id="P1",
                finding_id="fixture-" + local.name,
                message="Controlled low-risk finding",
                severity="LOW",
                confidence=1.0,
                file="SKILL.md",
                start_line=1,
            )
            canonical = {
                **state,
                "skill_path": str(local),
                "manifest": {"name": local.name},
                "components": ["SKILL.md", "package-lock.json"],
                "file_cache": {"SKILL.md": content},
                "local_file_cache": {"SKILL.md": content},
                "raw_file_cache": {"package-lock.json": raw_lock},
                "findings": [finding],
                "filtered_findings": [finding],
                "effective_finding_ids": [finding.finding_id],
                "inspection_ledger": [
                    dict(
                        ledger_event(
                            outcome=LedgerOutcome.COMPLETED,
                            phase="static_analysis",
                            analyzer_id="static_patterns",
                            path="SKILL.md",
                            emitted_finding_ids=[finding.finding_id],
                        )
                    ),
                    dict(
                        ledger_event(
                            outcome=LedgerOutcome.COMPLETED,
                            phase="static_analysis",
                            analyzer_id="static_patterns",
                            path="package-lock.json",
                        )
                    ),
                ],
                "component_metadata": [{"path": "SKILL.md", "type": "markdown"}],
            }
            dependency = real_inspect(cast(SkillspectorState, canonical))
            case.inspections.append(deepcopy(dependency["dependency_inspection"]))
            assert dependency["dependency_inspection"]["status"] == "completed"
            assert dependency["dependency_inspection"]["inspected_artifacts"] == 1
            # The graph fixture exposes physical inspected files for the real
            # no-follow source inventory, rather than bypassing its checks.
            for row in dependency["artifact_inventory"]:
                physical = local / row["path"]
                physical.parent.mkdir(parents=True, exist_ok=True)
                physical.write_bytes(b"fixture")
            for key in ("inspection_ledger", "component_metadata"):
                canonical[key] += dependency[key]
            canonical["artifact_inventory"] = dependency["artifact_inventory"]
            canonical["analyzer_status_events"] = [
                analyzer_status_for_events(
                    "static_patterns",
                    [
                        event
                        for event in canonical["inspection_ledger"]
                        if event.get("analyzer_id") == "static_patterns"
                    ],
                ),
                *dependency["analyzer_status_events"],
            ]
            canonical["dependency_inspection"] = dependency["dependency_inspection"]
            if source in resolved:
                clock.now += child_seconds
                if case.late is not None:
                    # Absolute raw component is deliberately absent from all
                    # report caches/metadata and outside the resolved root.
                    canonical["components"].append(str(case.late))
            return {**canonical, **real_report(cast(SkillspectorState, canonical))}

        def prepare(state):
            prepared = real_prepare(state)
            case.prepared.append(prepared)
            return prepared

        def cleanup(result):
            inventory = current_source_inventory()
            case.cleanup.append(
                (result.get("skill_path"), inventory.paths if inventory else frozenset())
            )
            real_cleanup(result)

        monkeypatch.setattr(report_module, "prepare_report", prepare)
        monkeypatch.setattr(cli.graph, "invoke", invoke)
        monkeypatch.setattr(cli, "cleanup_result", cleanup)
        return case

    return make, real_serialize


def _run(case, *, primary="json", extras=(), strict=False, output=None):
    args = [
        "scan", str(case.root), "--no-llm", "--transitive",
        "--inspect-dependencies", "--format", primary,
    ]
    if case.recursive:
        args.append("--recursive")
    if strict:
        args.append("--fail-on-incomplete")
    if output is not None:
        args.extend(["--output", str(output)])
    paths = []
    for i, fmt in enumerate(extras):
        path = case.home / f"extra-{i}.{fmt}"
        paths.append(path)
        args.extend(["--report", f"{fmt}={path}"])
    return CliRunner().invoke(cli.app, args), paths


def _metrics(case):
    traversal = case.traversals[0]
    assert all(item is traversal for item in case.traversals)
    return (
        [TARGET if source == TARGET else SECOND_TARGET if source == SECOND_TARGET
         else Path(source).name for source in case.calls],
        len(case.inspections),
        len(case.downloads),
        len(case.workers),
        traversal.scanned_targets,
        traversal.scanned_bytes,
        traversal.scanned_artifacts,
        sorted(traversal.cache),
        traversal.truncation_reasons.copy(),
    )


@pytest.mark.parametrize("primary", FORMATS)
def test_extra_formats_share_recursive_transitive_cache_and_dependency_budget(
    budget_case, monkeypatch, primary
):
    make, real_serialize = budget_case
    single = make(recursive=True)
    result, _ = _run(single, primary=primary)
    assert result.exit_code == 0, result.output
    expected = _metrics(single)
    multiple = make(recursive=True)
    attempts = []

    def serialize(prepared, fmt):
        # Every graph and dependency inspection finished before any renderer.
        assert multiple.calls == [str(multiple.roots[0]), TARGET, str(multiple.roots[1])]
        before = _metrics(multiple)
        attempts.append(fmt)
        multiple.clock.now += 1_000  # Serialization cannot restart/consume traversal.
        rendered = real_serialize(prepared, fmt)
        assert _metrics(multiple) == before
        return rendered

    monkeypatch.setattr(report_module, "serialize_report", serialize)
    result, paths = _run(multiple, primary=primary, extras=(*reversed(FORMATS), "json"))
    assert result.exit_code == 0, result.output
    assert _metrics(multiple) == expected
    assert expected[1:7] == (3, 3, 3, 1, 3 * CHARGE, 3)
    assert expected[7] == [(TARGET, False)]
    assert attempts == [fmt for fmt in (primary, *reversed(FORMATS), "json") for _ in range(2)]
    assert all(path.is_file() for path in paths)
    json_report = json.loads(paths[-1].read_text())
    assert json_report["skills_scanned"] == 2
    assert [skill["transitive_finding_count"] for skill in json_report["skills"]] == [1, 1]
    assert json_report["transitive_sources"] == [TARGET]
    multiple.provider.assert_not_called()
    assert current_source_inventory() is None


@pytest.mark.parametrize(
    ("limits", "child_seconds", "reason"),
    [
        ({"max_targets": 1}, 0.0, "target budget 1 reached"),
        ({"max_bytes": 2 * CHARGE}, 0.0, f"byte budget {2 * CHARGE} reached"),
        ({"max_seconds": 5.0}, 5.0, "time budget 5s reached"),
    ],
    ids=["targets", "dependency-bytes", "deadline"],
)
def test_additional_formats_do_not_reset_exhausted_transitive_allowances(
    budget_case, limits, child_seconds, reason
):
    make, _ = budget_case
    single = make(second_target=True, limits=limits, child_seconds=child_seconds)
    result, _ = _run(single, strict=True)
    assert result.exit_code == 1, result.output
    expected = _metrics(single)
    multiple = make(second_target=True, limits=limits, child_seconds=child_seconds)
    result, paths = _run(multiple, extras=("sarif", "markdown", "json", "sarif"), strict=True)
    assert result.exit_code == 1, result.output
    assert _metrics(multiple) == expected
    assert multiple.calls == [str(multiple.root), TARGET]
    assert expected[1:7] == (2, 2, 2, 1, 2 * CHARGE, 2)
    assert reason in expected[-1]
    assert all(path.is_file() for path in paths)
    payload = json.loads(result.stdout)
    assert payload["analysis_completeness"]["is_complete"] is False
    assert payload["metadata"]["transitive_targets_scanned"] == 1
    assert json.loads(paths[2].read_text()) == payload
    validate_sarif_report(json.loads(paths[0].read_text()))
    if "max_seconds" in limits:
        # Both acquisitions receive the same absolute deadline, not fresh
        # root/child/output allowances. Worker allowances decrease too.
        assert [call[2] for call in multiple.downloads] == [105.0, 105.0]
        assert [call[1] for call in multiple.workers] == [4.0, 3.0]


@pytest.mark.parametrize("strict", [False, True])
def test_transitive_record_bounds_do_not_stop_targets_or_expand_for_extra_formats(
    budget_case, strict
):
    make, _ = budget_case
    limits = {"max_findings": 2, "max_components": 2}
    single = make(second_target=True, limits=limits)
    result, _ = _run(single, strict=strict)
    assert result.exit_code == (1 if strict else 0), result.output
    expected = _metrics(single)
    multiple = make(second_target=True, limits=limits)
    result, paths = _run(multiple, extras=FORMATS + ("json",), strict=strict)
    assert result.exit_code == (1 if strict else 0), result.output
    assert _metrics(multiple) == expected
    assert expected[1:7] == (3, 3, 3, 2, 3 * CHARGE, 3)
    prepared = multiple.prepared[-1]
    assert len(prepared.reported_findings) <= 2
    assert len(prepared.component_metadata) <= 2
    assert any("finding budget 2" in reason for reason in expected[-1])
    assert any("component" in reason and "budget 2" in reason for reason in expected[-1])
    payload = json.loads(result.stdout)
    assert len(payload["issues"]) == 2
    assert payload["analysis_completeness"]["execution_successful"] is True
    assert payload["analysis_completeness"]["is_complete"] is False
    assert all(
        row["reason_code"] != "finding_accounting_error"
        for row in payload["analysis_completeness"]["ledger_exceptions"]
    )
    assert payload["metadata"]["transitive_targets_scanned"] == 2
    assert json.loads(paths[-1].read_text()) == payload
    # The second child was analyzed and cached in full, not discarded by the
    # merged finding budget. Reconciliation must leave its original producer
    # references available to another recursive root's independent merge.
    cached = multiple.traversals[0].cache[(SECOND_TARGET, False)]
    assert len(cached.findings) == 1
    assert any(
        cached.findings[0].finding_id in event.get("emitted_finding_ids", [])
        for event in cached.inspection_ledger
    )
    validate_sarif_report(json.loads(paths[3].read_text()))


def _ledger_result(*, ghost=False, failed=False):
    findings = [
        Finding(
            rule_id="P1",
            finding_id=f"finding-{i}",
            message="Controlled low-risk finding",
            severity="LOW",
            confidence=1.0,
            file="SKILL.md",
            start_line=i + 1,
        )
        for i in range(2)
    ]
    ids = [item.finding_id for item in findings]
    events = [
        dict(
            ledger_event(
                outcome=LedgerOutcome.COMPLETED,
                analyzer_id="static_patterns",
                phase="static_analysis",
                path="SKILL.md",
                emitted_finding_ids=[*ids, *(["unknown-finding"] if ghost else [])],
            )
        ),
        dict(
            ledger_event(
                outcome=LedgerOutcome.COMPLETED,
                analyzer_id="meta_analyzer",
                phase="meta",
                path="SKILL.md",
                input_finding_ids=ids,
                emitted_finding_ids=ids,
            )
        ),
    ]
    if failed:
        events.append(
            dict(
                ledger_event(
                    outcome=LedgerOutcome.FAILED,
                    analyzer_id="static_archive",
                    phase="archive_analysis",
                    path="SKILL.md",
                    reason=LedgerReason.READ_ERROR,
                )
            )
        )
    return {
        "findings": findings,
        "filtered_findings": findings,
        # Different order tests that known budget-omitted effective IDs are
        # removed too, rather than assuming all three collections align.
        "effective_finding_ids": list(reversed(ids)),
        "inspection_ledger": events,
        "components": ["SKILL.md"],
        "component_metadata": [{"path": "SKILL.md", "type": "markdown"}],
        "file_cache": {},
        "manifest": {"name": "bounded"},
        "use_llm": False,
    }


def _fixed_ledger_clock(monkeypatch):
    monkeypatch.setattr(cli, "monotonic", lambda: 100.0)
    for name in (
        "extract_external_refs_with_metadata",
        "plan_transitive_targets_with_metadata",
        "BoundedTransitiveFrontier",
    ):
        monkeypatch.setattr(
            transitive, name, partial(getattr(transitive, name), clock=lambda: 100.0)
        )


@pytest.mark.parametrize("ghost", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_root_budget_reconciles_producer_meta_and_effective_ids_without_hiding_failures(
    monkeypatch, ghost, failed
):
    _fixed_ledger_clock(monkeypatch)
    source = _ledger_result(ghost=ghost, failed=failed)
    before = deepcopy(source)
    traversal = cli._TransitiveTraversalState(budget=cli._TransitiveBudget(max_findings=1))
    with report_module.report_rendering_deferred():
        result = cli._scan_transitive(
            initial_result=source,
            format=cli.FormatChoice.json,
            no_llm=True,
            max_depth=1,
            transitive_allow_prefix=None,
            transitive_deny_prefix=None,
            baseline=None,
            show_suppressed=False,
            visited=set(),
            traversal=traversal,
        )
    assert source == before
    assert len(result["filtered_findings"]) == 1
    assert "finding budget 1 reached" in traversal.truncation_reasons
    completeness = result["analysis_completeness"]
    assert completeness["execution_successful"] is (not ghost and not failed)
    assert completeness["is_complete"] is False
    reasons = {row["reason_code"] for row in completeness["ledger_exceptions"]}
    assert ("finding_accounting_error" in reasons) is ghost
    assert ("read_error" in reasons) is failed
    assert traversal.scanned_targets == 0


@pytest.mark.parametrize("ghost", [False, True])
def test_child_cache_bound_reconciles_only_known_omitted_ids(monkeypatch, ghost):
    _fixed_ledger_clock(monkeypatch)
    monkeypatch.setattr(cli, "_TRANSITIVE_MAX_FINDINGS", 1)
    source = _ledger_result(ghost=ghost)
    before = deepcopy(source)
    traversal = cli._TransitiveTraversalState()
    cached = cli._cache_transitive_result(TARGET, source, traversal)
    assert source == before
    assert len(cached.findings) == 1
    retained_id = cached.findings[0].finding_id
    assert cached.effective_finding_ids == [retained_id]
    producer, meta = cached.inspection_ledger
    unknown_id = cli._scoped_finding_id(cached.source_identity, "unknown-finding")
    assert producer["emitted_finding_ids"] == [retained_id, *([unknown_id] if ghost else [])]
    assert meta["input_finding_ids"] == meta["emitted_finding_ids"] == [retained_id]
    assert any("finding budget 1" in reason for reason in traversal.truncation_reasons)


@pytest.mark.parametrize("phase", ["static_analysis", "meta"])
def test_budget_projection_preserves_malformed_contracts_and_never_mutates_rows(phase):
    event = dict(
        ledger_event(
            outcome=LedgerOutcome.COMPLETED,
            analyzer_id="meta_analyzer" if phase == "meta" else "static_patterns",
            phase=phase,
            path="SKILL.md",
            **(
                {"input_finding_ids": ["keep"], "emitted_finding_ids": ["keep"]}
                if phase == "meta"
                else {"emitted_finding_ids": ["keep"]}
            ),
        )
    )
    # Corruption predates retention. Deleting this known omitted reference
    # would mask an illegal producer input or an illegal meta output.
    event["emitted_finding_ids" if phase == "meta" else "input_finding_ids"] = ["omitted"]
    events = [event]
    before = deepcopy(events)
    projected = cli._reconcile_budget_omitted_ledger_findings(events, {"omitted"})
    assert projected[:1] == before
    assert projected is not events
    assert projected[0] is not event
    assert events == before
    if phase == "static_analysis":
        assert projected[-1]["outcome"] == LedgerOutcome.FAILED
        assert projected[-1]["reason_code"] == LedgerReason.FINDING_ACCOUNTING_ERROR


@pytest.mark.parametrize("lineage", ["duplicate", "unproduced", "silent-unproduced"])
@pytest.mark.parametrize("bound", ["root", "cache"])
def test_budget_omission_cannot_erase_existing_producer_conservation_failure(
    monkeypatch, lineage, bound
):
    _fixed_ledger_clock(monkeypatch)
    source = _ledger_result()
    dropped_id = source["findings"][1].finding_id
    if lineage == "duplicate":
        source["inspection_ledger"].append(
            dict(
                ledger_event(
                    outcome=LedgerOutcome.COMPLETED,
                    analyzer_id="duplicate_patterns",
                    phase="static_analysis",
                    path="SKILL.md",
                    emitted_finding_ids=[dropped_id],
                )
            )
        )
    else:
        source["inspection_ledger"][0]["emitted_finding_ids"].remove(dropped_id)
        if lineage == "silent-unproduced":
            source["inspection_ledger"].pop(1)  # No meta reference can preserve the failure.
    before = deepcopy(source)
    original_completeness, _ = finalize_ledger(source)
    assert not original_completeness["execution_successful"]
    assert any(
        row["reason_code"] == "finding_accounting_error"
        for row in original_completeness["ledger_exceptions"]
    )
    traversal = cli._TransitiveTraversalState(budget=cli._TransitiveBudget(max_findings=1))
    if bound == "cache":
        monkeypatch.setattr(cli, "_TRANSITIVE_MAX_FINDINGS", 1)
        cached = cli._cache_transitive_result(TARGET, source, traversal)
        bounded = {
            **source,
            "findings": cached.findings,
            "effective_finding_ids": cached.effective_finding_ids,
            "inspection_ledger": cached.inspection_ledger,
            "components": cached.components,
        }
        completeness, _ = finalize_ledger(bounded)
    else:
        with report_module.report_rendering_deferred():
            result = cli._scan_transitive(
                initial_result=source,
                format=cli.FormatChoice.json,
                no_llm=True,
                max_depth=1,
                transitive_allow_prefix=None,
                transitive_deny_prefix=None,
                baseline=None,
                show_suppressed=False,
                visited=set(),
                traversal=traversal,
            )
        completeness = result["analysis_completeness"]
    assert not completeness["execution_successful"]
    assert any(
        row["reason_code"] == "finding_accounting_error"
        for row in completeness["ledger_exceptions"]
    )
    assert source == before
    assert traversal.scanned_targets == 0


def test_budget_projection_obeys_ledger_output_limit_conservation_semantics():
    source = _ledger_result()
    dropped_id = source["findings"][1].finding_id
    source["inspection_ledger"].extend(
        [
            dict(
                ledger_event(
                    outcome=LedgerOutcome.COMPLETED,
                    analyzer_id="duplicate_patterns",
                    phase="static_analysis",
                    path="SKILL.md",
                    emitted_finding_ids=[dropped_id],
                )
            ),
            dict(
                ledger_event(
                    outcome=LedgerOutcome.PARTIAL,
                    record_type=LedgerRecordType.SYSTEM,
                    phase="ledger_output",
                    path="SKILL.md",
                    reason=LedgerReason.OUTPUT_LIMIT,
                    observed_records=5,
                    limit_records=4,
                )
            ),
        ]
    )
    # Origin conservation is intentionally not evaluated for output-limited
    # ledgers. Match that existing contract rather than inventing a new failure.
    completeness, _ = finalize_ledger(source)
    assert completeness["execution_successful"]
    before = deepcopy(source)
    bounded = {
        **source,
        "findings": source["findings"][:1],
        "effective_finding_ids": [source["findings"][0].finding_id],
        "inspection_ledger": cli._reconcile_budget_omitted_ledger_findings(
            source["inspection_ledger"], {dropped_id}
        ),
    }
    completeness, _ = finalize_ledger(bounded)
    assert completeness["execution_successful"]
    assert not completeness["is_complete"]
    assert all(
        row["reason_code"] != "finding_accounting_error"
        for row in completeness["ledger_exceptions"]
    )
    assert source == before


@pytest.mark.parametrize("strict", [False, True])
def test_recursive_transitive_output_limit_is_local_to_sarif(
    budget_case, monkeypatch, strict
):
    make, _ = budget_case
    case = make(recursive=True)
    real_build = report_module._build_sarif

    def oversized_sarif(*args, **kwargs):
        sarif = real_build(*args, **kwargs)
        sarif["runs"][0]["properties"] = {"fixturePadding": "é" * 60_000}
        return sarif

    monkeypatch.setattr(report_module, "_build_sarif", oversized_sarif)
    ceiling = 50_000
    monkeypatch.setattr(cli, "_MULTI_SKILL_MAX_REPORT_CHARACTERS", ceiling)
    result, paths = _run(case, extras=("sarif", "json", "markdown"), strict=strict)
    assert result.exit_code == (1 if strict else 0), result.output
    assert _metrics(case)[1:7] == (3, 3, 3, 1, 3 * CHARGE, 3)
    primary = json.loads(result.stdout)
    extra_json = json.loads(paths[1].read_text())
    assert primary == extra_json
    assert primary["skills_scanned"] == 2
    assert primary["analysis_completeness"]["is_complete"] is True, json.dumps(
        primary["analysis_completeness"], indent=2
    )
    assert primary["report_completeness"]["is_complete"] is True
    sarif = json.loads(paths[0].read_text())
    validate_sarif_report(sarif)
    invocation = sarif["runs"][0]["invocations"][0]["properties"]
    assert invocation["reportCompleteness"]["is_complete"] is False
    assert invocation["analysisCompleteness"]["is_complete"] is True
    assert all(len(path.read_text()) <= ceiling for path in paths)
    assert len(result.stdout.rstrip("\n")) <= ceiling
    assert paths[2].is_file()


@pytest.mark.parametrize("primary_collision", [False, True])
@pytest.mark.parametrize("hardlink", [False, True], ids=["same-name", "hardlink"])
def test_late_raw_source_identity_survives_transitive_caps_and_aborts_all_output(
    budget_case, monkeypatch, primary_collision, hardlink
):
    make, _ = budget_case
    case = make(limits={"max_components": 1})
    late = case.home / "unretained-source.py"
    late.write_bytes(b"protected raw source")
    case.late = late
    alias = case.home / "collision.json"
    if hardlink:
        os.link(late, alias)
    else:
        alias = late
    safe = case.home / "safe.json"
    safe.write_bytes(b"existing safe report")
    absent = case.home / "not-created.sarif"
    before = {path: (path.read_bytes(), path.stat().st_ino) for path in (late, alias, safe)}
    # Cap the cached child as well as the final merge. The raw absolute
    # component is beyond this slice; the independent inventory must keep it.
    monkeypatch.setattr(cli, "_TRANSITIVE_MAX_COMPONENTS", 1)
    emitter = Mock(side_effect=AssertionError("late barrier allowed emission"))
    monkeypatch.setattr(cli, "emit_report_outputs", emitter)
    writer = Mock(side_effect=AssertionError("late barrier allowed a write"))
    monkeypatch.setattr(cli, "write_text_no_follow", writer)
    args = [
        "scan", str(case.root), "--no-llm", "--transitive",
        "--inspect-dependencies", "--format", "json",
    ]
    if primary_collision:
        args.extend(["--output", str(alias)])
    else:
        args.extend(["--report", f"json={alias}"])
    args.extend(["--report", f"json={safe}", "--report", f"sarif={absent}"])
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 2, result.output
    assert "protected local source" in result.stderr
    assert result.stdout == ""
    assert case.calls == [str(case.root), TARGET]  # Late, not a preflight rejection.
    assert len(case.inspections) == 2
    cached = case.traversals[0].cache[(TARGET, False)]
    assert len(cached.components) == 1
    assert all(str(late) not in path for path in cached.components)
    prepared = case.prepared[-1]
    assert isinstance(prepared, PreparedReport)
    assert len(prepared.component_metadata) == 1
    assert all(str(late) not in row["path"] for row in prepared.component_metadata)
    assert any(late in paths for _, paths in case.cleanup)
    assert len(case.cleanup) == 2  # Child and root cleanup still run.
    emitter.assert_not_called()
    writer.assert_not_called()
    assert {path: (path.read_bytes(), path.stat().st_ino) for path in before} == before
    assert not absent.exists()
    assert current_source_inventory() is None
