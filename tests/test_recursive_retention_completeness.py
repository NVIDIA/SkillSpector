# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task 5.3: retained content is not the analyzed scope or source inventory."""

import json
import os
from copy import deepcopy
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from skillspector import cli
from skillspector.models import Finding
from skillspector.multi_skill import MultiSkillDetectionResult, SkillDirectory
from skillspector.nodes.report import prepare_report
from skillspector.recursive_reports import prepared_payload_usage
from skillspector.report_outputs import build_report_outputs, parse_report_declarations
from skillspector.report_sources import (
    SourceProtectionInventory,
    current_source_inventory,
    track_report_sources,
    validate_report_outputs,
)
from skillspector.state import MAX_WORKFLOW_BYTES

BOUNDS = {
    "occurrences": ("_MULTI_SKILL_MAX_PUBLIC_RECORDS", "recursive_finding_record_limit"),
    "bytes": ("_MULTI_SKILL_MAX_PREPARED_BYTES", "prepared_payload_limit"),
    "metadata": ("_MULTI_SKILL_MAX_METADATA_RECORDS", "metadata_record_limit"),
}


def _result(
    name: str,
    *,
    findings: bool = False,
    score: int = 0,
    status: str = "complete",
) -> dict[str, object]:
    canonical = (
        [
            Finding(
                finding_id=f"finding-{name}",
                rule_id="TEST-RETENTION",
                message="fixed analyzed evidence",
                severity="LOW",
                occurrences=[{"file": "SKILL.md", "start_line": i + 1} for i in range(3)],
            )
        ]
        if findings
        else []
    )
    completeness = {
        "is_complete": status == "complete",
        "execution_successful": status != "failed",
        "status": status,
        "coverage_percent": 100 if status == "complete" else 75,
    }
    prepared = prepare_report(
        {
            "manifest": {"name": name},
            "findings": canonical,
            "use_llm": False,
            "analysis_completeness": completeness,
        }
    )
    severity, recommendation = cli.risk_severity_and_recommendation(score)
    public = {
        **prepared.public_fields,
        "risk_score": score,
        "risk_severity": severity,
        "risk_recommendation": recommendation,
        "execution_successful": status != "failed",
        "analysis_completeness": completeness,
        "transitive_finding_count": 7 if findings else 0,
        "transitive_sources": ["https://example.test/omitted-dependency"] if findings else [],
    }
    prepared = replace(
        prepared,
        public_fields=public,
        risk_score=score,
        risk_severity=severity,
        risk_recommendation=recommendation,
        analysis_completeness=completeness,
        execution_successful=status != "failed",
    )
    return {**public, "prepared_report": prepared}


def _limit_for_omitted_second(
    monkeypatch: pytest.MonkeyPatch,
    bound: str,
    results: list[dict[str, object]],
) -> str:
    cap_name, reason = BOUNDS[bound]
    if bound == "occurrences":
        cap = 2  # First has no occurrences; second has three.
    else:
        index = 0 if bound == "bytes" else 1
        costs = [
            prepared_payload_usage(
                result["prepared_report"],
                max_bytes=MAX_WORKFLOW_BYTES,
                max_records=1_000_000,
            )[index]
            for result in results[:2]
        ]
        cap = sum(costs) - 1
        assert costs[0] < cap  # Analyze the second child before refusing its content.
    monkeypatch.setattr(cli, cap_name, cap)
    return reason


def _detection(tmp_path: Path, tail: int) -> MultiSkillDetectionResult:
    return MultiSkillDetectionResult(
        is_multi_skill=True,
        skills=[SkillDirectory(tmp_path / f"s{i}", f"s{i}", f"s{i}") for i in range(2 + tail)],
    )


@pytest.mark.parametrize("bound", BOUNDS)
@pytest.mark.parametrize("tail", [0, 2])
@pytest.mark.parametrize("status", ["complete", "partial", "failed"])
@pytest.mark.parametrize("score", [0, 100])
@pytest.mark.parametrize("output_format", list(cli.FormatChoice))
def test_shared_retention_omissions_preserve_analysis_and_policy_in_every_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bound: str,
    tail: int,
    status: str,
    score: int,
    output_format: cli.FormatChoice,
) -> None:
    monkeypatch.setattr(cli, "monotonic", lambda: 100.0)
    results = [_result("s0"), _result("s1", findings=True, score=score, status=status)]
    reason = _limit_for_omitted_second(monkeypatch, bound, results)
    analyzed = []
    cleaned = []

    def scan(**kwargs):
        name = Path(kwargs["input_path"]).name
        analyzed.append(name)
        assert name in ("s0", "s1"), "Uninspected tail must not invoke an analyzer"
        return results[int(name[1:])]

    monkeypatch.setattr(cli, "_scan_skill", scan)
    monkeypatch.setattr(cli, "cleanup_result", lambda result: cleaned.append(result))
    collection = cli._collect_recursive_reports(
        _detection(tmp_path, tail),
        output_format,
        None,
        no_llm=True,
        min_coverage=90,
    )
    assert analyzed == ["s0", "s1"]
    assert len(cleaned) == 2
    assert cleaned[0] is results[0] and cleaned[1] is results[1]
    assert [child.skill.name for child in collection.children] == ["s0"]
    assert collection.children[0].policy["risk_score"] == 0
    assert collection.children[0].prepared.report_completeness["is_complete"]
    assert collection.scanned_skill_count == 2
    assert collection.unscanned_skill_count == tail
    assert collection.output_omitted_skill_count == 1
    assert collection.retained_public_records == 0
    assert collection.retention_completeness == {
        "is_complete": False,
        "omission_reasons": [reason],
        "omitted_skills": 1,
        "omitted_records": 3,
    }
    analysis = collection.analysis_completeness
    assert analysis["is_complete"] is (tail == 0 and status == "complete")
    assert analysis["execution_successful"] is (status != "failed")
    assert analysis["fully_inspected_files"] == (2 if status == "complete" else 1)
    assert analysis["partially_inspected_files"] == (1 if status == "partial" else 0)
    assert analysis["entirely_uninspected_files"] == tail + (status == "failed")
    assert analysis["total_files"] == 2 + tail
    if tail == 0 and status == "complete":
        assert analysis["status"] == "complete"
        assert analysis["coverage_percent"] == 100
        assert analysis["limitations"] == []
    # Even when its full snapshot is discarded, the second child drives policy.
    assert collection.max_score == score
    assert collection.has_findings
    assert collection.execution_failed is (status == "failed")
    assert collection.coverage_failed is (tail > 0 or status == "partial")
    assert collection.transitive_finding_count == 7
    assert collection.transitive_sources == ["https://example.test/omitted-dependency"]
    recommendation = (
        "DO_NOT_INSTALL"
        if score == 100 or status == "failed"
        else "CAUTION"
        if tail or status == "partial"
        else "SAFE"
    )
    assert collection.risk_assessment["recommendation"] == recommendation
    before = deepcopy(collection)
    rendered = cli._render_recursive_report(collection, output_format)
    assert rendered["report_completeness"] == collection.retention_completeness
    assert rendered["report_completeness"] is not collection.retention_completeness
    assert collection == before
    if output_format == cli.FormatChoice.json:
        payload = json.loads(rendered["report_body"])
        assert payload["analysis_completeness"] == analysis
        assert payload["report_completeness"] == collection.retention_completeness
        assert payload["max_risk_score"] == score
        assert payload["risk_recommendation"] == recommendation
        assert payload["skills_scanned"] == 2
        assert payload["skills_omitted"] == tail
        assert payload["skills_output_omitted"] == 1
        assert payload["transitive_finding_count"] == 7
        assert payload["transitive_sources"] == collection.transitive_sources
        assert [entry["name"] for entry in payload["skills"] if "name" in entry] == ["s0"]
        omissions = {
            entry["reason"]: entry["omitted_count"]
            for entry in payload["skills"]
            if entry.get("omitted")
        }
        assert omissions["aggregate_output_limit"] == 1
        if tail:
            assert omissions["aggregate_scan_limit"] == tail
        else:
            assert "aggregate_scan_limit" not in omissions
    elif output_format == cli.FormatChoice.sarif:
        aggregate = json.loads(rendered["report_body"])["runs"][-1]
        assert aggregate["properties"]["kind"] == "recursiveAggregate"
        invocation = aggregate["invocations"][0]
        properties = invocation["properties"]
        assert properties["analysisCompleteness"] == analysis
        assert properties["reportCompleteness"] == collection.retention_completeness
        assert properties["riskAssessment"]["maxRiskScore"] == score
        assert properties["riskAssessment"]["recommendation"] == recommendation
        assert invocation["executionSuccessful"] is (status != "failed")
        notifications = invocation["toolExecutionNotifications"]
        assert any(
            row["properties"]["kind"] == "report_omission" and row["message"]["text"] == reason
            for row in notifications
        )
        if tail == 0 and status == "complete":
            assert all(row["properties"]["kind"] == "report_omission" for row in notifications)
    else:
        body = rendered["report_body"]
        assert "Report completeness: incomplete" in body
        assert reason in body
        assert "omitted records: 3" in body
        assert "omitted skills: 1" in body
        assert f"Maximum score: {score}/100" in body
        assert f"Recommendation: {recommendation.replace('_', ' ')}" in body
        assert f"Status: {analysis['status']}" in body
    # Mutating a consumer's projection cannot contaminate the shared result.
    rendered["report_completeness"]["omission_reasons"].append("consumer_only")
    assert collection == before


@pytest.mark.parametrize("bound", ["skill", "source", "time"])
@pytest.mark.parametrize("output_format", list(cli.FormatChoice))
def test_analysis_only_stop_does_not_invent_a_shared_retention_omission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bound: str,
    output_format: cli.FormatChoice,
) -> None:
    now = [100.0]
    monkeypatch.setattr(cli, "monotonic", lambda: now[0])
    if bound == "source":
        monkeypatch.setattr(cli, "MAX_WORKFLOW_BYTES", 2)
    elif bound == "time":
        real_budget = cli._TransitiveBudget
        monkeypatch.setattr(
            cli,
            "_TransitiveBudget",
            lambda **kwargs: real_budget(max_seconds=2, **kwargs),
        )
    analyzed = []

    def scan(**kwargs):
        name = Path(kwargs["input_path"]).name
        analyzed.append(name)
        assert name == "s0"
        if bound == "source":
            kwargs["transitive_traversal"].record_bytes(2)
        if bound == "time":
            now[0] += 2
        return _result(name)

    monkeypatch.setattr(cli, "_scan_skill", scan)
    collection = cli._collect_recursive_reports(
        _detection(tmp_path, 0),
        output_format,
        None,
        no_llm=True,
        max_skills=1 if bound == "skill" else None,
    )
    assert analyzed == ["s0"]
    assert collection.scanned_skill_count == 1
    assert collection.unscanned_skill_count == 1
    assert collection.output_omitted_skill_count == 0
    assert collection.analysis_completeness["entirely_uninspected_files"] == 1
    assert not collection.analysis_completeness["is_complete"]
    assert collection.retention_completeness == {
        "is_complete": True,
        "omission_reasons": [],
        "omitted_skills": 0,
        "omitted_records": 0,
    }
    assert collection.max_score == 0
    assert not collection.has_findings
    assert collection.transitive_finding_count == 0
    assert collection.risk_assessment["recommendation"] == "CAUTION"
    rendered = cli._render_recursive_report(collection, output_format)
    assert rendered["report_completeness"]["is_complete"]
    if output_format == cli.FormatChoice.json:
        payload = json.loads(rendered["report_body"])
        assert payload["skills_output_omitted"] == 0
        assert payload["skills_omitted"] == 1
        assert [row["reason"] for row in payload["skills"] if row.get("omitted")] == [
            "aggregate_scan_limit",
        ]
    elif output_format == cli.FormatChoice.sarif:
        invocation = json.loads(rendered["report_body"])["runs"][-1]["invocations"][0]
        assert not invocation["properties"]["analysisCompleteness"]["is_complete"]
        assert invocation["properties"]["reportCompleteness"]["is_complete"]
        assert all(
            row["properties"]["kind"] != "report_omission"
            for row in invocation["toolExecutionNotifications"]
        )
    else:
        assert "Status: partial" in rendered["report_body"]
        assert "Report completeness: incomplete" not in rendered["report_body"]


@pytest.mark.parametrize("output_format", list(cli.FormatChoice))
def test_several_shared_retention_limits_count_one_omitted_child_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    output_format: cli.FormatChoice,
) -> None:
    monkeypatch.setattr(cli, "monotonic", lambda: 100.0)
    results = [_result("s0"), _result("s1", findings=True)]
    reasons = [_limit_for_omitted_second(monkeypatch, bound, results) for bound in BOUNDS]
    monkeypatch.setattr(
        cli, "_scan_skill", lambda **kwargs: results[int(Path(kwargs["input_path"]).name[1:])]
    )
    collection = cli._collect_recursive_reports(
        _detection(tmp_path, 0),
        output_format,
        None,
        no_llm=True,
    )
    assert collection.analysis_completeness["is_complete"]
    omissions = collection.retention_completeness
    assert not omissions["is_complete"]
    assert omissions["omitted_skills"] == 1
    assert omissions["omitted_records"] == 3
    # The bounded walker may stop at metadata before measuring the entire
    # payload. Every observed limit is valid; counts must not multiply with it.
    assert reasons[0] in omissions["omission_reasons"]
    assert len(omissions["omission_reasons"]) >= 2
    assert set(omissions["omission_reasons"]) <= set(reasons)
    assert cli._render_recursive_report(collection, output_format)["report_completeness"] == (
        collection.retention_completeness
    )


@pytest.mark.parametrize("bound", BOUNDS)
@pytest.mark.parametrize("transitive", [False, True])
def test_raw_source_identity_survives_discarded_snapshot_and_temp_root_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bound: str,
    transitive: bool,
) -> None:
    monkeypatch.setattr(cli, "monotonic", lambda: 100.0)
    detection = _detection(tmp_path, 0)
    for skill in detection.skills:
        skill.path.mkdir()
        (skill.path / "SKILL.md").write_text("fixed input")
    temp_root = tmp_path / "skillspector_omitted"
    temp_root.mkdir()
    late_source = temp_root / "excluded.py"
    late_source.write_text("protected physical source")
    alias = tmp_path / "aliased-report.json"
    os.link(late_source, alias)
    identity = (late_source.stat().st_dev, late_source.stat().st_ino)
    results = [_result("s0"), _result("s1", findings=True)]
    for skill, result in zip(detection.skills, results, strict=True):
        result.update(skill_path=str(skill.path), components=["SKILL.md"])
    results[1].update(
        artifact_inventory=[{"path": str(late_source), "excluded": True}],
        temp_dir_for_cleanup=str(temp_root),
    )
    reason = _limit_for_omitted_second(monkeypatch, bound, results)
    graph = Mock()
    graph.invoke.side_effect = results
    monkeypatch.setattr(cli, "graph", graph)
    if transitive:
        # The raw graph hook owns protection. A merged/scoped result is not a
        # local filesystem inventory and must not be passed to add_result.
        monkeypatch.setattr(
            cli,
            "_scan_transitive",
            lambda **kwargs: {
                **kwargs["initial_result"],
                "components": ["source::dependency.py"],
                "artifact_inventory": [{"path": "https://example.test/scoped/source.py"}],
            },
        )
    inventory = SourceProtectionInventory(clock=lambda: 100.0)
    plan = build_report_outputs(
        cli.FormatChoice.json,
        None,
        parse_report_declarations([f"json={alias}"]),
    )
    validate_report_outputs(plan, inventory)  # The alias is not a known source yet.
    cleanup_observations = []
    real_cleanup = cli.cleanup_result

    def cleanup(result):
        if result.get("temp_dir_for_cleanup"):
            cleanup_observations.append(
                (late_source in inventory.paths, identity in inventory.identities)
            )
        real_cleanup(result)

    monkeypatch.setattr(cli, "cleanup_result", cleanup)
    with track_report_sources(inventory):
        collection = cli._collect_recursive_reports(
            detection,
            cli.FormatChoice.json,
            None,
            no_llm=True,
            transitive_enabled=transitive,
        )
    assert graph.invoke.call_count == 2
    assert collection.scanned_skill_count == 2
    assert collection.analysis_completeness["is_complete"]
    assert collection.retention_completeness["omission_reasons"] == [reason]
    assert [child.skill.name for child in collection.children] == ["s0"]
    assert cleanup_observations == [(True, True)]
    assert not temp_root.exists()
    assert late_source in inventory.paths
    assert identity in inventory.identities
    assert inventory.complete
    assert inventory.protects(alias, alias.stat())
    collection.release()
    assert identity in inventory.identities
    with pytest.raises(ValueError, match="protected local source"):
        validate_report_outputs(plan, inventory)
    assert alias.read_text() == "protected physical source"
    assert current_source_inventory() is None


@pytest.mark.parametrize("bound", BOUNDS)
@pytest.mark.parametrize("tail", [0, 2])
def test_shared_retention_projection_is_isolated_across_repeated_formats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bound: str,
    tail: int,
) -> None:
    monkeypatch.setattr(cli, "monotonic", lambda: 100.0)
    results = [_result("s0"), _result("s1", findings=True, score=100)]
    reason = _limit_for_omitted_second(monkeypatch, bound, results)
    scan = Mock(side_effect=results)
    monkeypatch.setattr(cli, "_scan_skill", scan)
    collection = cli._collect_recursive_reports(
        _detection(tmp_path, tail), cli.FormatChoice.json, None, no_llm=True
    )
    assert scan.call_count == 2
    before = deepcopy(collection)
    formats = [*cli.FormatChoice, *reversed(list(cli.FormatChoice)), cli.FormatChoice.json]
    for output_format in formats:
        rendered = cli._render_recursive_report(collection, output_format)
        assert rendered["report_completeness"] == {
            "is_complete": False,
            "omission_reasons": [reason],
            "omitted_skills": 1,
            "omitted_records": 3,
        }
        rendered["report_completeness"]["omission_reasons"].append("consumer_only")
        rendered["report_completeness"]["omitted_skills"] = 99
        assert collection == before
        assert scan.call_count == 2, "Rendering must not reanalyze the omitted child or tail"


@pytest.mark.parametrize("bound", BOUNDS)
@pytest.mark.parametrize("additional", [False, True], ids=["sole-primary", "multi-report"])
@pytest.mark.parametrize(
    ("score", "status", "policy_flags", "expected_exit"),
    [
        pytest.param(0, "complete", [], 0, id="retention-default-off"),
        pytest.param(0, "complete", ["--fail-on-incomplete"], 1, id="retention-strict"),
        pytest.param(0, "complete", ["--fail-on-findings"], 1, id="omitted-findings"),
        pytest.param(100, "complete", [], 1, id="omitted-risk"),
        pytest.param(0, "partial", [], 0, id="coverage-default-off"),
        pytest.param(0, "partial", ["--min-coverage", "90"], 1, id="omitted-coverage"),
        pytest.param(0, "failed", [], 2, id="omitted-execution"),
        pytest.param(
            100,
            "failed",
            ["--fail-on-incomplete", "--fail-on-findings", "--min-coverage", "90"],
            2,
            id="execution-overrides-policy",
        ),
    ],
)
def test_cli_preserves_omitted_child_policy_until_all_reports_are_emitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bound: str,
    additional: bool,
    score: int,
    status: str,
    policy_flags: list[str],
    expected_exit: int,
) -> None:
    monkeypatch.setattr(cli, "monotonic", lambda: 100.0)
    # JSON's scanned_at is generated by the serializer, not the snapshot.
    # Fix the wall clock too so repeated reports can be compared in full.
    report_module = import_module("skillspector.nodes.report")
    wall_clock = Mock(wraps=report_module.datetime)
    wall_clock.now.return_value = report_module.datetime(2026, 11, 5, tzinfo=report_module.UTC)
    monkeypatch.setattr(report_module, "datetime", wall_clock)
    root = tmp_path / "input"
    root.mkdir()
    results = [_result("s0"), _result("s1", findings=True, score=score, status=status)]
    for i, result in enumerate(results):
        child = root / f"s{i}"
        child.mkdir()
        (child / "SKILL.md").write_text(f"---\nname: s{i}\n---\nfixed input")
        result.update(skill_path=str(child), components=["SKILL.md"])
    reason = _limit_for_omitted_second(monkeypatch, bound, results)
    graph = Mock()
    graph.invoke.side_effect = results
    monkeypatch.setattr(cli, "graph", graph)
    collections = []
    emissions = []
    real_collect = cli._collect_recursive_reports
    real_render = cli._render_recursive_report

    def collect(*args, **kwargs):
        collection = real_collect(*args, **kwargs)
        assert [child.skill.name for child in collection.children] == ["s0"]
        assert collection.max_score == score
        assert collection.has_findings
        assert collection.execution_failed is (status == "failed")
        assert collection.coverage_failed is ("--min-coverage" in policy_flags and status == "partial")
        collections.append(collection)
        return collection

    def render(collection, output_format):
        assert graph.invoke.call_count == 2, "Both children must finish analysis before emission"
        rendered = real_render(collection, output_format)
        assert rendered["report_completeness"] == collection.retention_completeness
        emissions.append(output_format)
        return rendered

    monkeypatch.setattr(cli, "_collect_recursive_reports", collect)
    monkeypatch.setattr(cli, "_render_recursive_report", render)
    sarif_path = tmp_path / "extra.sarif"
    json_path = tmp_path / "repeated.json"
    arguments = ["scan", str(root), "--recursive", "--no-llm", "--format", "json", *policy_flags]
    if additional:
        arguments.extend(["--report", f"sarif={sarif_path}", "--report", f"json={json_path}"])
    invocation = CliRunner().invoke(cli.app, arguments)
    assert invocation.exit_code == expected_exit, invocation.output
    assert graph.invoke.call_count == 2
    assert [Path(call.args[0]["input_path"]).name for call in graph.invoke.call_args_list] == [
        "s0",
        "s1",
    ]
    assert emissions == [
        cli.FormatChoice.json,
        *([cli.FormatChoice.sarif, cli.FormatChoice.json] if additional else []),
    ]
    payload = json.loads(invocation.stdout)
    assert payload["skills_scanned"] == 2
    assert payload["skills_omitted"] == 0
    assert payload["skills_output_omitted"] == 1
    assert [row["name"] for row in payload["skills"] if "name" in row] == ["s0"]
    assert payload["max_risk_score"] == score
    assert payload["execution_successful"] is (status != "failed")
    assert payload["analysis_completeness"]["is_complete"] is (status == "complete")
    assert payload["report_completeness"] == {
        "is_complete": False,
        "omission_reasons": [reason],
        "omitted_skills": 1,
        "omitted_records": 3,
    }
    if additional:
        assert json.loads(json_path.read_text()) == payload
        invocation_properties = json.loads(sarif_path.read_text())["runs"][-1]["invocations"][0]
        assert invocation_properties["properties"]["analysisCompleteness"] == payload[
            "analysis_completeness"
        ]
        assert invocation_properties["properties"]["reportCompleteness"] == payload[
            "report_completeness"
        ]
        assert invocation_properties["executionSuccessful"] is (status != "failed")
    assert len(collections) == 1
    assert collections[0].children == [], "Release snapshots after emission, including policy exits"
    assert current_source_inventory() is None


@pytest.mark.parametrize("bound", BOUNDS)
@pytest.mark.parametrize("primary_collision", [False, True])
@pytest.mark.parametrize("tail", [0, 2])
def test_cli_active_inventory_protects_source_of_analyzed_but_unretained_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bound: str,
    primary_collision: bool,
    tail: int,
) -> None:
    monkeypatch.setattr(cli, "monotonic", lambda: 100.0)
    root = tmp_path / "input"
    root.mkdir()
    for i in range(2 + tail):
        child = root / f"s{i}"
        child.mkdir()
        (child / "SKILL.md").write_text(f"---\nname: s{i}\n---\nfixed input")
    late = tmp_path / "late.py"
    late.write_text("source stays intact")
    alias = tmp_path / "late-alias.json"
    os.link(late, alias)
    safe = tmp_path / "safe.json"
    safe.write_text("previous report stays intact")
    results = [_result("s0"), _result("s1", findings=True, score=100)]
    for i, result in enumerate(results):
        result.update(skill_path=str(root / f"s{i}"), components=["SKILL.md"])
    results[1]["artifact_inventory"] = [{"path": str(late), "excluded": True}]
    _limit_for_omitted_second(monkeypatch, bound, results)
    analyzed = []
    inventories = []
    collected = []
    real_collect = cli._collect_recursive_reports

    def invoke(state, **kwargs):
        name = Path(state["input_path"]).name
        analyzed.append(name)
        inventories.append(current_source_inventory())
        assert inventories[-1] is not None, "CLI must activate raw graph tracking"
        return results[int(name[1:])]

    def collect(*args, **kwargs):
        collection = real_collect(*args, **kwargs)
        collected.append(
            (
                [child.skill.name for child in collection.children],
                collection.scanned_skill_count,
                collection.unscanned_skill_count,
                collection.analysis_completeness["is_complete"],
            )
        )
        return collection

    graph = Mock()
    graph.invoke.side_effect = invoke
    monkeypatch.setattr(cli, "graph", graph)
    monkeypatch.setattr(cli, "_collect_recursive_reports", collect)
    render = Mock(side_effect=AssertionError("Source barrier must precede all rendering"))
    writer = Mock(side_effect=AssertionError("Source barrier must precede all emission"))
    monkeypatch.setattr(cli, "_render_recursive_report", render)
    monkeypatch.setattr(cli, "write_text_no_follow", writer)
    arguments = ["scan", str(root), "--recursive", "--no-llm", "--format", "terminal", "--verbose"]
    if primary_collision:
        arguments.extend(["--output", str(alias), "--report", f"json={safe}"])
    else:
        arguments.extend(["--report", f"json={alias}", "--report", f"json={safe}"])
    invocation = CliRunner().invoke(cli.app, arguments)
    assert invocation.exit_code == 2, invocation.output
    assert invocation.stdout == ""
    assert analyzed == ["s0", "s1"]
    assert collected == [(["s0"], 2, tail, tail == 0)]
    assert all(inventory is inventories[0] for inventory in inventories)
    inventory = inventories[0]
    assert late in inventory.paths
    assert inventory.protects(alias, alias.stat())
    render.assert_not_called()
    writer.assert_not_called()
    assert late.read_text() == alias.read_text() == "source stays intact"
    assert safe.read_text() == "previous report stays intact"
    assert current_source_inventory() is None
