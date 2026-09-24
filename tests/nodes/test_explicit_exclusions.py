# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller exclusions omit content while retaining truthful scan accounting."""

import importlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.nodes.build_context import build_context


def make_skill(root: Path) -> None:
    (root / "SKILL.md").write_text("# Demo\nA harmless example skill.\n")
    (root / "fixtures").mkdir()
    (root / "fixtures/attack.sh").write_text("#!/bin/sh\nrm -rf /\n")


def test_excluded_bytes_never_enter_content_caches(tmp_path, monkeypatch):
    make_skill(tmp_path)
    (tmp_path / "SKILL.md").write_text("# Demo\nSee [fixture](fixtures/attack.sh).\n")
    # Exercise both ordinary files and the separately audited generated tree.
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules/attack.sh").write_text("#!/bin/sh\nrm -rf /\n")
    module = importlib.import_module("skillspector.nodes.build_context")
    original = module._open_regular_file_no_follow

    def guarded_open(path, *args, **kwargs):
        assert Path(path).name != "attack.sh", "Excluded file content was opened"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(module, "_open_regular_file_no_follow", guarded_open)
    state = build_context(
        {"skill_path": str(tmp_path), "exclude_patterns": ["fixtures/*", "node_modules/*"]}
    )
    for key in ("file_cache", "raw_file_cache", "local_file_cache", "llm_file_cache"):
        assert set(state[key]) == {"SKILL.md"}
    assert state["components"] == ["SKILL.md"]
    excluded = [row for row in state["artifact_inventory"] if row.get("reason") == "user_exclusion"]
    assert {row["path"] for row in excluded} == {"fixtures/attack.sh", "node_modules/attack.sh"}
    assert all(row["disposition"] == "partial" for row in excluded)


@pytest.mark.parametrize("pattern", ["*", "SKILL.md", "*.md", "", "../*", "/tmp/*", "a\\b", "a\nb"])
def test_invalid_or_manifest_exclusions_rejected(tmp_path, pattern):
    make_skill(tmp_path)
    with pytest.raises(ValueError):
        build_context({"skill_path": str(tmp_path), "exclude_patterns": [pattern]})


def test_patterns_are_case_sensitive_and_recorded_even_without_matches(tmp_path):
    make_skill(tmp_path)
    state = build_context({"skill_path": str(tmp_path), "exclude_patterns": ["Fixtures/*"]})
    assert "fixtures/attack.sh" in state["file_cache"]


@pytest.mark.parametrize("option", ["--recursive", "--transitive", "--mcp-registry"])
def test_unsupported_modes_reject_exclusions(tmp_path, option):
    make_skill(tmp_path)
    result = CliRunner().invoke(app, ["scan", str(tmp_path), "--exclude", "fixtures/*", option])
    assert result.exit_code == 2
    assert "--exclude requires a local single-skill directory" in result.output


def test_real_cli_exclusion_keeps_coverage_and_findings_truthful(tmp_path):
    make_skill(tmp_path)
    runner = CliRunner()
    control_path = tmp_path.parent / f"{tmp_path.name}-control.json"
    excluded_path = tmp_path.parent / f"{tmp_path.name}-excluded.json"
    common = ["scan", str(tmp_path), "--no-llm", "--format", "json"]
    control = runner.invoke(app, [*common, "--output", str(control_path)])
    assert control.exit_code in (0, 1), control.output
    before = json.loads(control_path.read_text())
    assert any(issue["id"] == "TM1" for issue in before["issues"])
    result = runner.invoke(
        app,
        [
            *common,
            "--exclude",
            "fixtures/*",
            "--exclude",
            "missing/*",
            "--exclude",
            "*/attack.sh",
            "--fail-on-incomplete",
            "--output",
            str(excluded_path),
        ],
    )
    assert result.exit_code == 1, result.output
    after = json.loads(excluded_path.read_text())
    assert not after["issues"]
    assert after["risk_assessment"]["recommendation"] == "CAUTION"
    completeness = after["analysis_completeness"]
    assert completeness["exclude_patterns"] == ["fixtures/*", "missing/*", "*/attack.sh"]
    assert completeness["excluded_file_count"] == 1
    assert completeness["entirely_uninspected_files"] == 1
    assert completeness["total_components"] == 2
    assert completeness["coverage_percent"] == 50.0
    assert completeness["status"] == "partial"
    assert completeness["ledger_exceptions"][0]["path"] == "fixtures/attack.sh"


@pytest.mark.parametrize("output_format", ["sarif", "markdown", "terminal"])
def test_report_formats_show_explicit_scope(tmp_path, output_format):
    make_skill(tmp_path)
    output = tmp_path.parent / f"{tmp_path.name}-{output_format}.txt"
    result = CliRunner().invoke(
        app,
        [
            "scan",
            str(tmp_path),
            "--no-llm",
            "--exclude",
            "fixtures/*",
            "--format",
            output_format,
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    text = output.read_text()
    assert "fixtures/*" in text
    assert "fixtures/attack.sh" in text
    assert "user_exclusion" in text
    if output_format == "sarif":
        sarif = json.loads(text)
        projection = sarif["runs"][0]["invocations"][0]["properties"]["analysisCompleteness"]
        assert projection["excludePatterns"] == ["fixtures/*"]
        assert projection["excludedFileCount"] == 1
        assert projection["isComplete"] is False


def test_excluded_reference_remains_uninspected(tmp_path):
    make_skill(tmp_path)
    (tmp_path / "SKILL.md").write_text("# Demo\nSee [fixture](fixtures/attack.sh).\n")
    state = build_context({"skill_path": str(tmp_path), "exclude_patterns": ["fixtures/*"]})
    assert "fixtures/attack.sh" not in state["file_cache"]
    artifact = next(
        row for row in state["artifact_inventory"] if row["path"] == "fixtures/attack.sh"
    )
    assert artifact["disposition"] == "partial"
    assert artifact["referenced"] is True
    assert state["artifact_references"][0]["disposition"] == "partial"


def test_default_and_unmatched_exclusions_leave_content_present(tmp_path):
    make_skill(tmp_path)
    baseline = build_context({"skill_path": str(tmp_path)})
    unmatched = build_context({"skill_path": str(tmp_path), "exclude_patterns": ["Fixtures/*"]})
    assert baseline["file_cache"] == unmatched["file_cache"]
    assert not any(row.get("reason") == "user_exclusion" for row in unmatched["inspection_ledger"])


def test_unmatched_pattern_remains_complete_and_audited(tmp_path):
    (tmp_path / "SKILL.md").write_text("# Demo\nA harmless example skill.\n")
    output = tmp_path.parent / f"{tmp_path.name}-unmatched.json"
    result = CliRunner().invoke(
        app,
        [
            "scan",
            str(tmp_path),
            "--no-llm",
            "--exclude",
            "fixtures/*",
            "--format",
            "json",
            "--fail-on-incomplete",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text())
    completeness = report["analysis_completeness"]
    assert completeness["status"] == "complete"
    assert completeness["exclude_patterns"] == ["fixtures/*"]
    assert completeness["excluded_file_count"] == 0
    assert report["risk_assessment"]["recommendation"] == "SAFE"
