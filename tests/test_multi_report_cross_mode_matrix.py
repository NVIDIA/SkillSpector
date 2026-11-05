# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task 6.1: cross-mode CLI acceptance complementing the mode-specific suites.

Keep discovery, graph orchestration, transitive merging, preparation, serializers,
the emission coordinator and secure writes real. Only acquisition/analyzer
results are controlled. Detailed parser, retention/budget, character-bound,
cleanup and source-discovery permutations belong to the existing suites.
"""

from __future__ import annotations

import importlib
import json
import os
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from skillspector import cli, file_output
from skillspector.inspection_ledger import (
    LedgerOutcome,
    LedgerReason,
    analyzer_status_for_events,
    finalize_ledger,
    ledger_event,
)
from skillspector.models import Finding
from skillspector.report_completeness import build_report_completeness
from skillspector.report_sources import current_source_inventory
from skillspector.sarif_models import validate_sarif_report
from skillspector.state import SkillspectorState

report_module = importlib.import_module("skillspector.nodes.report")
MODES = ("ordinary", "recursive", "transitive", "registry")
SKILL_MODES = MODES[:-1]
PRIMARY_FORMATS = ("default", "terminal", "json", "markdown", "sarif")
TARGET = "https://github.com/acceptance/cross-mode-child"

pytestmark = pytest.mark.skipif(
    not file_output._SECURE_OUTPUT_SUPPORTED,
    reason="matrix exercises descriptor-relative secure file output",
)


@pytest.fixture
def matrix_case(tmp_path, monkeypatch):
    """Two recursive children or one real transitive target, never live providers."""
    clock = Mock(wraps=datetime)
    clock.now.return_value = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(report_module, "datetime", clock)
    monkeypatch.setattr(cli.err_console, "_force_terminal", False)
    provider = Mock(side_effect=AssertionError("matrix must not resolve a provider"))
    monkeypatch.setattr(report_module, "is_llm_available", provider)
    original_report = report_module.report
    original_emit = cli.emit_report_outputs
    original_recursive_render = cli._render_recursive_report

    def make(mode, condition="complete"):
        root = tmp_path / "input"
        if mode == "registry":
            root.write_text("{}", encoding="utf-8")
            locals_ = [root]
        else:
            root.mkdir()
            locals_ = [root / "a", root / "b"] if mode == "recursive" else [root]
            for local in locals_:
                local.mkdir(exist_ok=True)
                (local / "SKILL.md").write_text(
                    f"---\nname: {local.name}\n---\n{TARGET if mode == 'transitive' else 'Hello'}\n",
                    encoding="utf-8",
                )
        resolved = tmp_path / "resolved"
        resolved.mkdir()
        (resolved / "SKILL.md").write_text("# Dependency\n", encoding="utf-8")
        expected = (
            [str(root)]
            if mode == "registry"
            else [str(local) for local in locals_] + ([TARGET] if mode == "transitive" else [])
        )
        case = SimpleNamespace(
            mode=mode,
            root=root,
            source=root if mode == "registry" else locals_[0] / "SKILL.md",
            expected=expected,
            calls=[],
            events=[],
            snapshots=[],
            policies=[],
            successful_writes={},
            discovered_source=None,
            after_analysis=None,
            before_render=None,
            fail_render=None,
            fail_write=None,
            outputs=[],
            provider=provider,
        )

        def graph_result(state, config=None):
            assert state["defer_report_rendering"] is True
            source = str(state["input_path"])
            local = resolved if source == TARGET else Path(source)
            case.calls.append(source)
            findings = []
            if condition in {"findings", "risk", "execution", "serialized"}:
                findings = [
                    Finding(
                        rule_id="MATRIX",
                        finding_id=f"matrix-{local.name}-{i}",
                        message="Controlled evidence",
                        severity="CRITICAL" if condition in {"risk", "execution"} else "LOW",
                        confidence=1.0,
                        file="SKILL.md",
                        start_line=i + 1,
                    )
                    # Two CRITICAL occurrences score 75, above (not at) the
                    # real risk threshold. A sole LOW finding isolates policy.
                    for i in range(2 if condition in {"risk", "execution"} else 1)
                ]
            outcome = (
                LedgerOutcome.PARTIAL
                if condition in {"analysis", "coverage"}
                else LedgerOutcome.COMPLETED
            )
            reason = LedgerReason.SIZE_LIMIT if outcome == LedgerOutcome.PARTIAL else None
            producer = ledger_event(
                outcome=outcome,
                phase="static_analysis",
                analyzer_id="static_patterns",
                path="SKILL.md",
                reason=reason,
                emitted_finding_ids=[finding.finding_id for finding in findings],
            )
            events = [producer]
            if condition == "execution":
                # Evidence from a completed producer remains reportable when
                # a separate analyzer fails. FAILED producers cannot emit IDs.
                events.append(
                    ledger_event(
                        outcome=LedgerOutcome.FAILED,
                        phase="static_analysis",
                        analyzer_id="behavioral_ast",
                        path="SKILL.md",
                        reason=LedgerReason.ANALYZER_RUNTIME_ERROR,
                    )
                )
            canonical = {
                **state,
                "skill_path": str(local),
                "manifest": {"name": local.name},
                "components": ["SKILL.md"],
                "component_metadata": [{"path": "SKILL.md", "type": "markdown"}],
                "file_cache": {
                    "SKILL.md": TARGET if mode == "transitive" and source != TARGET else "Hello"
                },
                "findings": findings,
                "filtered_findings": findings,
                "effective_finding_ids": [finding.finding_id for finding in findings],
                "inspection_ledger": events,
                "analyzer_status_events": [
                    analyzer_status_for_events(
                        analyzer,
                        [event for event in events if event["analyzer_id"] == analyzer],
                    )
                    for analyzer in dict.fromkeys(event["analyzer_id"] for event in events)
                ],
                "use_llm": False,
                "llm_requested": False,
                # Three notes exercise the real SARIF notification fallback.
                "structured_summaries": [
                    {
                        "id": f"summary-{i}",
                        "message": (
                            "Controlled serialized evidence é " * 500
                            if condition == "serialized"
                            else "Controlled summary"
                        ),
                        "file": "SKILL.md",
                    }
                    for i in range(3)
                ]
                if condition in {"report", "serialized"}
                else [],
            }
            if case.discovered_source is not None and case.calls == case.expected:
                inventory = current_source_inventory()
                assert inventory is not None
                assert case.discovered_source not in inventory.paths
                # The real graph-return hook must protect this raw source
                # before preparation/merging can discard its report content.
                canonical["components"].append(str(case.discovered_source))
            completeness, effective_ids = finalize_ledger(cast(SkillspectorState, canonical))
            canonical["analysis_completeness"] = completeness
            canonical["execution_successful"] = completeness["execution_successful"]
            canonical["effective_finding_ids"] = effective_ids
            assert completeness["execution_successful"] is (condition != "execution")
            assert not any(
                row["reason_code"] == LedgerReason.FINDING_ACCOUNTING_ERROR
                for row in completeness["ledger_exceptions"]
            )
            if condition == "coverage":
                assert completeness["coverage_percent"] < 100
            result = {**canonical, **original_report(cast(SkillspectorState, canonical))}
            if case.calls == case.expected and case.after_analysis is not None:
                case.after_analysis()
            return result

        def registry_result(source):
            case.calls.append(source)
            result = {
                "findings": [{"rule": "MATRIX"}]
                if condition in {"findings", "risk", "execution"}
                else [],
                "risk_score": 99 if condition in {"risk", "execution"} else 0,
                "execution_successful": condition != "execution",
                "analysis_completeness": {
                    "is_complete": condition not in {"analysis", "execution"}
                },
                "report_completeness": build_report_completeness(
                    omission_reasons=["fixture_retention_limit"] if condition == "report" else [],
                    omitted_records=1 if condition == "report" else 0,
                ),
            }
            if case.after_analysis is not None:
                case.after_analysis()
            return result

        def emit(entries, renderer, primary_writer, additional_writer, diagnostic):
            # This is an observation/fault-injection seam, not a replacement
            # coordinator. In particular registry retains its JSON-only lambda.
            assert case.calls == case.expected
            case.outputs = list(entries)
            index = 0

            def render(choice):
                nonlocal index
                current = index
                index += 1
                entry = case.outputs[current]
                case.events.append(("render", choice.value, entry.path))
                if case.before_render is not None:
                    case.before_render(current)
                if case.fail_render == current:
                    raise RuntimeError("private-render-sentinel")
                rendered = renderer(choice)
                case.snapshots.append(deepcopy(rendered))
                if mode != "recursive":
                    case.policies.append(
                        {
                            "risk_score": rendered["risk_score"],
                            "analysis_completeness": deepcopy(rendered["analysis_completeness"]),
                        }
                    )
                return rendered

            def primary(rendered, entry):
                case.events.append(("write", entry.path))
                if case.fail_write is not None and case.fail_write == entry.path:
                    raise PermissionError("private-write-sentinel")
                primary_writer(rendered, entry)
                if entry.path is not None:
                    path = Path(entry.path)
                    case.successful_writes[path] = (path.read_bytes(), path.stat().st_ino)

            def write(path, body):
                case.events.append(("write", path))
                if case.fail_write == path:
                    raise PermissionError("private-write-sentinel")
                additional_writer(path, body)
                written = Path(path)
                case.successful_writes[written] = (written.read_bytes(), written.stat().st_ino)

            return original_emit(entries, render, primary, write, diagnostic)

        def recursive_render(collection, choice):
            before = deepcopy((collection.risk_assessment, collection.analysis_completeness))
            rendered = original_recursive_render(collection, choice)
            assert (collection.risk_assessment, collection.analysis_completeness) == before
            case.policies.append(
                {
                    "risk_score": collection.max_score,
                    "analysis_completeness": deepcopy(collection.analysis_completeness),
                }
            )
            return rendered

        monkeypatch.setattr(cli.graph, "invoke", graph_result)
        monkeypatch.setattr(cli, "scan_registry", registry_result)
        monkeypatch.setattr(cli, "emit_report_outputs", emit)
        monkeypatch.setattr(cli, "_render_recursive_report", recursive_render)
        if condition == "report" and mode != "registry":
            monkeypatch.setattr(report_module, "MAX_FINDING_OUTPUT_RECORDS", 2)
        return case

    return make


def _extra_formats(case):
    return ("json", "json", "json") if case.mode == "registry" else ("json", "sarif", "json")


def _run(
    case,
    tmp_path,
    *,
    primary="json",
    primary_file=False,
    options=(),
    extra_paths=None,
    additional_reports=True,
):
    args = ["scan", str(case.root), "--no-llm", *options]
    if case.mode == "recursive":
        args.append("--recursive")
    elif case.mode == "transitive":
        args.append("--transitive")
    elif case.mode == "registry":
        args.append("--mcp-registry")
    if primary != "default":
        args += ["--format", primary]
    primary_format = "terminal" if primary == "default" else primary
    primary_path = tmp_path / f"primary.{primary_format}" if primary_file else None
    if primary_path is not None:
        args += ["--output", str(primary_path)]
    formats = _extra_formats(case) if additional_reports else ()
    paths = extra_paths or [tmp_path / f"extra-{i}.{fmt}" for i, fmt in enumerate(formats)]
    for fmt, path in zip(formats, paths, strict=True):
        args += ["--report", f"{fmt}={path}"]
    return CliRunner().invoke(cli.app, args), primary_path, paths


def _assert_attempts(case, primary, primary_path, paths, *, render_failure=None):
    formats = _extra_formats(case)
    entries = [(primary, str(primary_path) if primary_path else None)]
    entries += [(fmt, str(path)) for fmt, path in zip(formats, paths, strict=True)]
    expected = []
    for index, (fmt, path) in enumerate(entries):
        expected.append(("render", fmt, path))
        if index != render_failure:
            expected.append(("write", path))
    assert case.events == expected, case.events
    assert case.calls == case.expected  # One graph per child/target, one registry call.
    for path, original in case.successful_writes.items():
        assert (path.read_bytes(), path.stat().st_ino) == original
    case.provider.assert_not_called()
    assert current_source_inventory() is None


def _json_pair(case, paths):
    first, last = json.loads(paths[0].read_text()), json.loads(paths[-1].read_text())
    assert first == last
    if case.mode == "recursive":
        assert first["skills_scanned"] == 2
        assert [row["name"] for row in first["skills"]] == ["a", "b"]
    elif case.mode == "transitive":
        assert first["metadata"]["transitive_targets_scanned"] == 1
    return first


def _sarif_properties(case, path):
    sarif = json.loads(path.read_text())
    validate_sarif_report(sarif)
    # Recursive output appends an aggregate invocation after its child runs.
    run = sarif["runs"][-1] if case.mode == "recursive" else sarif["runs"][0]
    return run["invocations"][0]["properties"]


@pytest.mark.parametrize(
    ("mode", "primary", "primary_file"),
    [
        (mode, primary, primary_file)
        for mode in MODES
        for primary in (("json",) if mode == "registry" else PRIMARY_FORMATS)
        for primary_file in (False, True)
    ],
)
def test_primary_streams_and_files_share_ordered_repeated_reports(
    matrix_case, tmp_path, mode, primary, primary_file
):
    case = matrix_case(mode)
    result, output, paths = _run(case, tmp_path, primary=primary, primary_file=primary_file)
    assert result.exit_code == 0, result.output
    _assert_attempts(case, "terminal" if primary == "default" else primary, output, paths)
    payload = _json_pair(case, paths)
    assert payload["analysis_completeness"]["is_complete"] is True
    assert payload["report_completeness"]["is_complete"] is True
    if mode != "registry":
        _sarif_properties(case, paths[1])
    assert result.stderr.count("Saved ") == 3
    assert "Saved " not in result.stdout
    body = output.read_text() if primary_file else result.stdout
    if primary == "json":
        assert json.loads(body) == payload
    elif primary == "sarif":
        validate_sarif_report(json.loads(body))
    else:
        assert "Report completeness" in body
        assert str(paths[0]) not in result.stdout
    if primary_file:
        message = "Combined report saved to" if mode == "recursive" else "Report saved to"
        assert message in result.stdout
    assert len(case.policies) == 4
    assert all(policy["risk_score"] == 0 for policy in case.policies)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("condition", "options", "exit_code"),
    [
        ("complete", (), 0),
        ("analysis", (), 0),
        ("analysis", ("--fail-on-incomplete",), 1),
        ("report", (), 0),
        ("report", ("--fail-on-incomplete",), 1),
        ("findings", ("--fail-on-findings",), 1),
        ("risk", (), 1),
        ("execution", ("--fail-on-findings", "--fail-on-incomplete"), 2),
    ],
)
def test_completed_results_emit_before_cross_mode_policy_exits(
    matrix_case, tmp_path, mode, condition, options, exit_code
):
    case = matrix_case(mode, condition)
    result, output, paths = _run(case, tmp_path, options=options)
    assert result.exit_code == exit_code, result.output
    _assert_attempts(case, "json", output, paths)
    payload = _json_pair(case, paths)
    assert json.loads(result.stdout) == payload  # No policy/status stdout contamination.
    assert payload["analysis_completeness"]["is_complete"] is (
        condition not in {"analysis", "execution"}
    )
    assert payload["report_completeness"]["is_complete"] is (
        mode != "registry" or condition != "report"
    )
    if mode != "registry":
        sarif = _sarif_properties(case, paths[1])
        assert sarif["reportCompleteness"]["is_complete"] is (condition != "report")
        # Preserve the established aggregate vs per-skill SARIF schemas;
        # assert the exact field, rather than accepting either opportunistically.
        complete_key = "is_complete" if mode == "recursive" else "isComplete"
        assert sarif["analysisCompleteness"][complete_key] is (
            condition not in {"analysis", "execution"}
        )
        if condition == "report":
            assert sarif["reportCompleteness"]["omitted_records"] > 0
            assert "notification_record_limit" in sarif["reportCompleteness"]["omission_reasons"]
            assert all(policy["analysis_completeness"]["is_complete"] for policy in case.policies)
    if condition == "findings":
        score = (
            payload["risk_score"]
            if mode == "registry"
            else payload["max_risk_score"]
            if mode == "recursive"
            else payload["risk_assessment"]["score"]
        )
        assert score <= cli.RISK_THRESHOLD  # Policy, not the risk gate, caused exit 1.
    if condition in {"risk", "execution"}:
        assert all(policy["risk_score"] > cli.RISK_THRESHOLD for policy in case.policies)
    assert len(case.policies) == 4
    assert len({policy["risk_score"] for policy in case.policies}) == 1
    assert all(
        policy["analysis_completeness"] == case.policies[0]["analysis_completeness"]
        for policy in case.policies
    )
    if mode == "recursive":
        assert sarif["riskAssessment"]["maxRiskScore"] == payload["max_risk_score"]
    assert result.stderr.count("Saved ") == 3


@pytest.mark.parametrize("mode", SKILL_MODES)
def test_coverage_policy_waits_for_repeated_outputs(matrix_case, tmp_path, mode):
    case = matrix_case(mode, "coverage")
    result, output, paths = _run(case, tmp_path, options=("--min-coverage", "100"))
    assert result.exit_code == 1, result.output
    _assert_attempts(case, "json", output, paths)
    assert json.loads(result.stdout) == _json_pair(case, paths)
    _sarif_properties(case, paths[1])


@pytest.mark.parametrize("primary", ("terminal", "json", "markdown", "sarif"))
@pytest.mark.parametrize("strict", [False, True], ids=("default-off", "strict"))
def test_recursive_sole_report_real_serialized_fallback_preserves_analysis_and_gates(
    matrix_case, tmp_path, monkeypatch, primary, strict
):
    """No --report, retention omission, injected flag, or padded serializer."""
    case = matrix_case("recursive", "serialized")
    ceiling = 20_000
    assert cli._MULTI_SKILL_MAX_REPORT_CHARACTERS == 4_194_304
    monkeypatch.setattr(cli, "_MULTI_SKILL_MAX_REPORT_CHARACTERS", ceiling)
    # Keep the fallback notice on one line for exact terminal-body comparison.
    monkeypatch.setattr(cli.console, "width", 200)
    real_serialize = report_module.serialize_report
    real_render = cli._render_recursive_report
    child_lengths = []
    projections = []

    def serialize(prepared, choice):
        assert choice == primary
        assert case.calls == case.expected  # Collection finishes before rendering.
        rendered = real_serialize(prepared, choice)
        assert rendered["report_completeness"]["is_complete"] is True
        child_lengths.append(len(rendered["report_body"]))
        return rendered

    def render(collection, choice):
        assert collection.scanned_skill_count == len(collection.children) == 2
        assert [child.skill.name for child in collection.children] == ["a", "b"]
        assert collection.unscanned_skill_count == collection.output_omitted_skill_count == 0
        assert collection.retained_public_records == 2
        assert collection.retention_completeness == build_report_completeness()
        assert collection.analysis_completeness["is_complete"] is True
        rendered = real_render(collection, choice)
        projections.append(deepcopy(rendered))
        return rendered

    monkeypatch.setattr(report_module, "serialize_report", serialize)
    monkeypatch.setattr(cli, "_render_recursive_report", render)
    result, output, paths = _run(
        case,
        tmp_path,
        primary=primary,
        options=("--fail-on-incomplete",) if strict else (),
        additional_reports=False,
    )
    assert result.exit_code == (1 if strict else 0), result.output
    assert output is None and paths == []
    assert case.calls == case.expected
    assert case.events == []  # The legacy sole-report path, not the coordinator.
    assert len(child_lengths) == 2 and sum(child_lengths) > ceiling
    assert len(projections) == len(case.policies) == 1
    rendered = projections[0]
    omission = rendered["report_completeness"]
    assert omission == {
        "is_complete": False,
        "omission_reasons": ["serialized_output_limit"],
        "omitted_records": 2,
        "omitted_skills": 2,
    }
    assert len(rendered["report_body"]) <= ceiling
    if primary == "terminal":
        # Legacy terminal progress belongs on stdout; machine reports below
        # must still match their serialized body exactly without that prefix.
        marker = "--- Recursive Risk Assessment ---"
        assert result.stdout.count(marker) == 1
        progress, _, suffix = result.stdout.partition(marker)
        assert "Multi-Skill Summary" in progress
        assert progress.count("[1/2] Scanning a") == progress.count("[2/2] Scanning b") == 1
        assert marker + suffix.rstrip("\n") == rendered["report_body"].rstrip("\n")
    else:
        assert result.stdout.rstrip("\n") == rendered["report_body"].rstrip("\n")
    assert case.policies[0]["risk_score"] == 5 <= cli.RISK_THRESHOLD
    analysis = case.policies[0]["analysis_completeness"]
    assert analysis["is_complete"] is analysis["execution_successful"] is True
    assert analysis["coverage_percent"] == 100
    assert "Controlled serialized evidence" not in result.stdout
    if primary == "json":
        payload = json.loads(result.stdout)
        assert payload["report_completeness"] == omission
        assert payload["analysis_completeness"] == analysis
        assert payload["skills_scanned"] == payload["skills_output_omitted"] == 2
        assert payload["max_risk_score"] == 5
        assert payload["execution_successful"] is True
    elif primary == "sarif":
        payload = json.loads(result.stdout)
        validate_sarif_report(payload)
        invocation = payload["runs"][-1]["invocations"][0]
        assert invocation["executionSuccessful"] is True
        properties = invocation["properties"]
        assert properties["reportCompleteness"] == omission
        assert properties["analysisCompleteness"] == analysis
        assert properties["riskAssessment"]["maxRiskScore"] == 5
    else:
        assert "Report completeness: incomplete" in result.stdout
        assert "serialized_output_limit" in result.stdout
        assert "Status: complete" in result.stdout
        assert "Maximum score: 5/100" in result.stdout
    assert "Saved " not in result.stderr
    assert current_source_inventory() is None
    case.provider.assert_not_called()
    assert not list(tmp_path.rglob(".skillspector-output-*"))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("stage", ("render", "write"))
@pytest.mark.parametrize("index", (0, 2), ids=("primary", "additional"))
@pytest.mark.parametrize("condition", ("risk", "report", "execution"))
def test_output_error_overrides_risk_omission_and_execution_gates_without_rollback(
    matrix_case, tmp_path, mode, stage, index, condition
):
    case = matrix_case(mode, condition)
    output = tmp_path / "primary.json"
    paths = [tmp_path / f"extra-{i}.{fmt}" for i, fmt in enumerate(_extra_formats(case))]
    destinations = [output, *paths]
    bad = destinations[index]
    bad.write_text("previous report", encoding="utf-8")
    before = (bad.read_bytes(), bad.stat().st_ino)
    if stage == "render":
        case.fail_render = index
    else:
        case.fail_write = str(bad)
    result, output, paths = _run(
        case, tmp_path, primary_file=True, options=("--fail-on-incomplete", "--fail-on-findings")
    )
    assert result.exit_code == 2, result.output
    _assert_attempts(
        case, "json", output, paths, render_failure=index if stage == "render" else None
    )
    assert (bad.read_bytes(), bad.stat().st_ino) == before
    for i, path in enumerate([output, *paths]):
        if i != index:
            payload = json.loads(path.read_text())
            if path.suffix == ".sarif":
                validate_sarif_report(payload)
            else:
                assert payload["execution_successful"] is (condition != "execution")
    assert f"Failed to {stage}" in result.stderr
    assert bad.name in result.stderr
    assert "private-render-sentinel" not in result.output
    assert "private-write-sentinel" not in result.output
    assert "Failed to" not in result.stdout
    assert result.stderr.count("Saved ") == (3 if index == 0 else 2)
    assert not list(tmp_path.rglob(".skillspector-output-*"))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("race", ["leaf_symlink", "parent_symlink"])
def test_post_barrier_filesystem_race_preserves_targets_and_later_reports(
    matrix_case, tmp_path, mode, race
):
    case = matrix_case(mode, "risk")
    sentinel = tmp_path / "outside.txt"
    sentinel.write_text("never follow this target", encoding="utf-8")
    sentinel_before = (sentinel.read_bytes(), sentinel.stat().st_ino)
    fmt = "json" if mode == "registry" else "sarif"
    archived = tmp_path / "old-report"
    parent = tmp_path / "raced-parent"
    displaced = tmp_path / "displaced-parent"
    parent.mkdir()
    bad = (parent if race == "parent_symlink" else tmp_path) / f"extra-1.{fmt}"
    bad.write_text("old report", encoding="utf-8")
    bad_before = (bad.read_bytes(), bad.stat().st_ino)

    def mutate(index):
        if index != 2:
            return
        # All plan validation and an earlier report write have completed.
        assert (tmp_path / "extra-0.json").is_file()
        if race == "leaf_symlink":
            bad.rename(archived)
            bad.symlink_to(sentinel)
        else:
            parent.rename(displaced)
            parent.symlink_to(tmp_path, target_is_directory=True)

    case.before_render = mutate
    result, output, paths = _run(
        case,
        tmp_path,
        options=("--fail-on-findings",),
        extra_paths=[tmp_path / "extra-0.json", bad, tmp_path / "extra-2.json"],
    )
    assert result.exit_code == 2, result.output
    _assert_attempts(case, "json", output, paths)
    assert json.loads(result.stdout) == _json_pair(case, paths)
    assert (sentinel.read_bytes(), sentinel.stat().st_ino) == sentinel_before
    if race == "leaf_symlink":
        assert bad.is_symlink()
        assert (archived.read_bytes(), archived.stat().st_ino) == bad_before
    else:
        assert parent.is_symlink()
        assert (
            (displaced / bad.name).read_bytes(),
            (displaced / bad.name).stat().st_ino,
        ) == bad_before
        assert not (tmp_path / bad.name).exists()
    assert "Failed to write" in result.stderr
    assert bad.name in result.stderr
    assert not list(tmp_path.rglob(".skillspector-output-*"))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("primary_collision", [False, True])
def test_analysis_time_destination_alias_aborts_before_any_emission(
    matrix_case, tmp_path, mode, primary_collision
):
    case = matrix_case(mode, "risk")
    bad = tmp_path / ("primary.json" if primary_collision else "extra-0.json")
    safe = tmp_path / "extra-2.json"
    safe.write_text("preserved existing report", encoding="utf-8")
    safe_before = (safe.read_bytes(), safe.stat().st_ino)
    source_before = (case.source.read_bytes(), case.source.stat().st_ino)

    def acquire_alias():
        # This destination was absent at preflight. In particular registry's
        # inventory must refresh known identities despite no prepared report.
        assert not bad.exists()
        os.link(case.source, bad)

    case.after_analysis = acquire_alias
    result, output, paths = _run(case, tmp_path, primary_file=primary_collision)
    assert result.exit_code == 2, result.output
    assert case.calls == case.expected  # A genuinely late rejection.
    assert case.events == []
    assert result.stdout == ""
    assert "protected local source" in result.stderr
    assert (case.source.read_bytes(), case.source.stat().st_ino) == source_before
    assert (bad.read_bytes(), bad.stat().st_ino) == source_before
    assert (safe.read_bytes(), safe.stat().st_ino) == safe_before
    assert not paths[1].exists()
    case.provider.assert_not_called()
    assert current_source_inventory() is None
    assert not list(tmp_path.rglob(".skillspector-output-*"))


@pytest.mark.parametrize("mode", SKILL_MODES)
@pytest.mark.parametrize("primary_collision", [False, True])
def test_new_raw_source_aborts_entire_plan_before_stdout_or_replacement(
    matrix_case, tmp_path, mode, primary_collision
):
    """Unlike alias races above, this file is unknown to pre-scan discovery."""
    case = matrix_case(mode, "risk")
    late = tmp_path / "late-source.py"
    late.write_text("# Newly acquired source\n", encoding="utf-8")
    safe = tmp_path / "safe.json"
    safe.write_text("existing report", encoding="utf-8")
    before = {path: (path.read_bytes(), path.stat().st_ino) for path in (case.source, late, safe)}
    case.discovered_source = late
    formats = _extra_formats(case)
    paths = [tmp_path / f"extra-{i}.{fmt}" for i, fmt in enumerate(formats)]
    paths[-1] = safe
    if primary_collision:
        # _run's primary path aliases the newly acquired file but not any
        # source discoverable under the local input root before analysis.
        primary = tmp_path / "primary.json"
        os.link(late, primary)
        before[primary] = (primary.read_bytes(), primary.stat().st_ino)
    else:
        paths[0] = late
    result, _, _ = _run(case, tmp_path, primary_file=primary_collision, extra_paths=paths)
    assert result.exit_code == 2, result.output
    assert case.calls == case.expected
    assert case.events == []
    assert result.stdout == ""
    assert "protected local source" in result.stderr
    for path, original in before.items():
        assert (path.read_bytes(), path.stat().st_ino) == original
    assert not paths[1].exists()
    case.provider.assert_not_called()
    assert current_source_inventory() is None
    assert not list(tmp_path.rglob(".skillspector-output-*"))
