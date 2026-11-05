# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recursive collection is independent of rendering, output, and primary format."""

import gc
import importlib
import json
import weakref
from dataclasses import replace
from pathlib import Path

import pytest
import typer

from skillspector import cli
from skillspector.models import Finding
from skillspector.multi_skill import MultiSkillDetectionResult, SkillDirectory
from skillspector.nodes.report import prepare_report, report_rendering_is_deferred
from skillspector.recursive_reports import prepared_payload_usage
from skillspector.sarif_models import validate_sarif_report

report_module = importlib.import_module("skillspector.nodes.report")


def detection(tmp_path: Path, count: int = 3) -> MultiSkillDetectionResult:
    return MultiSkillDetectionResult(
        is_multi_skill=True,
        skills=[SkillDirectory(tmp_path / f"s{i}", f"s{i}", f"s{i}") for i in range(count)],
    )


def child_result(name: str, *, records: int = 0, score: int = 0) -> dict[str, object]:
    findings = [
        Finding(
            rule_id=f"R-{name}-{index}",
            message=f"finding {name} {index}",
            severity="LOW",
            file="SKILL.md",
            start_line=1,
        )
        for index in range(records)
    ]
    prepared = prepare_report(
        {
            "manifest": {"name": name},
            "findings": findings,
            "use_llm": False,
            "analysis_completeness": {"is_complete": True, "coverage_percent": 100},
        }
    )
    public = {**prepared.public_fields, "risk_score": score}
    prepared = replace(prepared, public_fields=public, risk_score=score)
    return {**public, "prepared_report": prepared}


@pytest.mark.parametrize("primary_format", list(cli.FormatChoice))
def test_all_analysis_finishes_before_first_serializer_and_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, primary_format: cli.FormatChoice
) -> None:
    events = []
    real_serializer = report_module.serialize_report

    def scan(**kwargs):
        assert report_rendering_is_deferred()
        name = Path(kwargs["input_path"]).name
        events.append(("scan", name))
        return child_result(name)

    def serialize(prepared, output_format):
        assert not report_rendering_is_deferred()
        assert [event for event in events if event[0] == "scan"] == [
            ("scan", "s0"), ("scan", "s1"), ("scan", "s2")
        ]
        events.append(("render", prepared.manifest["name"]))
        return real_serializer(prepared, output_format)

    def write(*args):
        events.append(("write", "primary"))

    monkeypatch.setattr(cli, "_scan_skill", scan)
    monkeypatch.setattr(report_module, "serialize_report", serialize)
    monkeypatch.setattr(cli, "write_text_no_follow", write)
    cli._scan_multi_skill(detection(tmp_path), primary_format, tmp_path / "out", no_llm=True)
    assert events[:3] == [("scan", "s0"), ("scan", "s1"), ("scan", "s2")]
    assert events[-1] == ("write", "primary")


@pytest.mark.parametrize("primary_format", list(cli.FormatChoice))
@pytest.mark.parametrize("limit", [None, 1, 3])
def test_child_order_and_skill_limit_are_primary_format_invariant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    primary_format: cli.FormatChoice,
    limit: int | None,
) -> None:
    names = []

    def scan(**kwargs):
        name = Path(kwargs["input_path"]).name
        names.append(name)
        return child_result(name)

    monkeypatch.setattr(cli, "_scan_skill", scan)
    monkeypatch.setattr(
        report_module, "serialize_report", lambda *args: pytest.fail("Collection rendered a format")
    )
    collection = cli._collect_recursive_reports(
        detection(tmp_path), primary_format, None, no_llm=True, max_skills=limit
    )
    expected = ["s0"] if limit == 1 else ["s0", "s1", "s2"]
    assert names == [child.skill.name for child in collection.children] == expected
    assert collection.scanned_skill_count == len(expected)
    assert collection.unscanned_skill_count == 3 - len(expected)
    assert collection.analysis_completeness["is_complete"] is (limit != 1)
    assert collection.retention_completeness["is_complete"]


def test_one_collection_can_render_every_format_without_reanalysis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def scan(**kwargs):
        calls.append(kwargs)
        return child_result(Path(kwargs["input_path"]).name, records=1)

    monkeypatch.setattr(cli, "_scan_skill", scan)
    collection = cli._collect_recursive_reports(
        detection(tmp_path), cli.FormatChoice.json, None, no_llm=True
    )
    snapshots = [child.prepared for child in collection.children]
    for output_format in list(cli.FormatChoice) + [cli.FormatChoice.json]:
        result = cli._render_recursive_report(collection, output_format)
        assert result["report_completeness"]["is_complete"]
        assert [child.prepared for child in collection.children] == snapshots
        if output_format == cli.FormatChoice.json:
            payload = json.loads(result["report_body"])
            assert [entry["name"] for entry in payload["skills"]] == ["s0", "s1", "s2"]
        if output_format == cli.FormatChoice.sarif:
            validate_sarif_report(json.loads(result["report_body"]))
    assert len(calls) == 3
    assert calls[0]["transitive_cache"] is calls[1]["transitive_cache"]
    assert calls[0]["transitive_traversal"] is calls[1]["transitive_traversal"]


@pytest.mark.parametrize("primary_format", list(cli.FormatChoice))
def test_primary_output_size_does_not_stop_collection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, primary_format: cli.FormatChoice
) -> None:
    monkeypatch.setattr(cli, "_MULTI_SKILL_MAX_REPORT_CHARACTERS", 1_500)
    names = []

    def scan(**kwargs):
        name = Path(kwargs["input_path"]).name
        names.append(name)
        # A legacy rendered body is not a canonical retention input.
        return {**child_result(name), "report_body": "x" * 10_000_000}

    monkeypatch.setattr(cli, "_scan_skill", scan)
    collection = cli._collect_recursive_reports(
        detection(tmp_path), primary_format, None, no_llm=True
    )
    assert names == ["s0", "s1", "s2"]
    assert collection.retention_completeness["is_complete"]
    result = cli._render_recursive_report(collection, primary_format)
    assert len(result["report_body"]) <= 1_500
    assert collection.analysis_completeness["is_complete"]
    assert collection.risk_assessment["recommendation"] == "SAFE"


def test_raw_graph_and_caches_are_released_before_rendering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class GraphResult(dict):
        pass

    refs = []
    cleaned = []

    def scan(**kwargs):
        name = Path(kwargs["input_path"]).name
        graph_result = GraphResult(
            **child_result(name),
            file_cache={"unused": "raw graph cache"},
            local_file_cache={"unused": "local graph cache"},
            arbitrary_graph_field={"unretained": True},
        )
        refs.append(weakref.ref(graph_result))
        return graph_result

    monkeypatch.setattr(cli, "_scan_skill", scan)
    monkeypatch.setattr(cli, "cleanup_result", lambda result: cleaned.append(id(result)))
    collection = cli._collect_recursive_reports(
        detection(tmp_path), cli.FormatChoice.json, None, no_llm=True
    )
    gc.collect()
    assert len(cleaned) == 3
    assert all(ref() is None for ref in refs)
    assert all(child.prepared is not None for child in collection.children)
    assert all("file_cache" not in child.policy for child in collection.children)


def test_omitted_analyzed_child_keeps_policy_without_mislabeling_analysis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "_MULTI_SKILL_MAX_PUBLIC_RECORDS", 1)
    monkeypatch.setattr(cli, "_scan_skill", lambda **kwargs: child_result("risk", records=2, score=100))
    collection = cli._collect_recursive_reports(
        detection(tmp_path, 1), cli.FormatChoice.json, None, no_llm=True
    )
    assert collection.children == []
    assert collection.scanned_skill_count == 1
    assert collection.max_score == 100
    assert collection.has_findings
    assert collection.analysis_completeness["is_complete"]
    assert collection.retention_completeness["omitted_skills"] == 1
    assert not collection.retention_completeness["is_complete"]


def test_retention_stop_reports_uninspected_tail_separately(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "_MULTI_SKILL_MAX_PUBLIC_RECORDS", 1)
    monkeypatch.setattr(cli, "_scan_skill", lambda **kwargs: child_result("risk", records=2))
    collection = cli._collect_recursive_reports(
        detection(tmp_path), cli.FormatChoice.json, None, no_llm=True
    )
    assert collection.scanned_skill_count == 1
    assert collection.unscanned_skill_count == 2
    assert not collection.analysis_completeness["is_complete"]
    assert collection.analysis_completeness["entirely_uninspected_files"] == 2
    assert collection.retention_completeness["omitted_skills"] == 1


def test_prepared_payload_cap_is_used_without_any_serializer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = child_result("s0")
    size, _ = prepared_payload_usage(result["prepared_report"], max_bytes=10_000_000, max_records=10_000)
    monkeypatch.setattr(cli, "_MULTI_SKILL_MAX_PREPARED_BYTES", size - 1)
    monkeypatch.setattr(cli, "_scan_skill", lambda **kwargs: result)
    monkeypatch.setattr(
        report_module, "serialize_report", lambda *args: pytest.fail("Retention invoked a serializer")
    )
    collection = cli._collect_recursive_reports(
        detection(tmp_path, 1), cli.FormatChoice.json, None, no_llm=True
    )
    assert collection.children == []
    assert collection.analysis_completeness["is_complete"]
    assert "prepared_payload_limit" in collection.retention_completeness["omission_reasons"]


def test_renderer_failure_leaves_complete_collection_available_to_other_formats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "_scan_skill", lambda **kwargs: child_result("s0"))
    collection = cli._collect_recursive_reports(
        detection(tmp_path, 1), cli.FormatChoice.sarif, None, no_llm=True
    )
    real_serializer = report_module.serialize_report

    def serialize(prepared, output_format):
        if output_format == "sarif":
            raise RuntimeError("SARIF failed after collection")
        return real_serializer(prepared, output_format)

    monkeypatch.setattr(report_module, "serialize_report", serialize)
    with pytest.raises(RuntimeError, match="SARIF failed"):
        cli._render_recursive_report(collection, cli.FormatChoice.sarif)
    assert collection.analysis_completeness["is_complete"]
    assert not collection.execution_failed
    payload = json.loads(cli._render_recursive_report(collection, cli.FormatChoice.json)["report_body"])
    assert payload["skills"][0]["name"] == "s0"


@pytest.mark.parametrize("failure", ["render", "write", "interrupt"])
def test_snapshots_release_on_emission_error_or_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    monkeypatch.setattr(cli, "_scan_skill", lambda **kwargs: child_result("s0"))
    collection = cli._collect_recursive_reports(
        detection(tmp_path, 1), cli.FormatChoice.json, None, no_llm=True
    )
    monkeypatch.setattr(cli, "_collect_recursive_reports", lambda **kwargs: collection)

    def fail(*args):
        if failure == "interrupt":
            raise KeyboardInterrupt
        raise RuntimeError("output failed")

    if failure == "write":
        monkeypatch.setattr(cli, "write_text_no_follow", fail)
    else:
        monkeypatch.setattr(cli, "_render_recursive_report", fail)
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else RuntimeError):
        cli._scan_multi_skill(detection(tmp_path, 1), cli.FormatChoice.json, tmp_path / "out", no_llm=True)
    assert collection.children == []


def test_strict_gate_counts_retention_omissions_after_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "_MULTI_SKILL_MAX_PUBLIC_RECORDS", 1)
    monkeypatch.setattr(cli, "_scan_skill", lambda **kwargs: child_result("s0", records=2))
    output = tmp_path / "out.json"
    with pytest.raises(typer.Exit) as exc:
        cli._scan_multi_skill(
            detection(tmp_path, 1), cli.FormatChoice.json, output, no_llm=True, fail_on_incomplete=True
        )
    assert exc.value.exit_code == 1
    payload = json.loads(output.read_text())
    assert payload["analysis_completeness"]["is_complete"]
    assert not payload["report_completeness"]["is_complete"]
