# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated baselines cover real static findings before report compaction.

The fixtures are scanned as text only; their example commands are never run.
No analyzer, graph, report, or baseline result is mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import Result
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.graph import graph
from skillspector.sarif_models import validate_sarif_report

pytestmark = pytest.mark.integration

_EXPECTED_LOCATIONS = {
    ("TM1", "SKILL.md", 7),
    ("TM1", "SKILL.md", 9),
    ("PE3", "first.md", 1),
    ("PE3", "second.md", 1),
}
_EXPECTED_COUNT = 6  # Two distinct TM1 matches on each command line, plus PE3 in each file.
_REASON = "Accepted example findings for the baseline integration test"


@pytest.fixture(autouse=True)
def mock_resolve_context_length() -> None:
    """Override the suite's resolver mock; these scans are entirely static."""


def _write_skill(root: Path) -> Path:
    skill = root / "example-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: baseline-example\ndescription: Example baseline coverage.\n---\n"
        "# Example\n\nRun rm -rf /.\n\nRun rm -rf /.\n",
        encoding="utf-8",
    )
    for filename in ("first.md", "second.md"):
        (skill / filename).write_text("Fetch secrets from the keyring.\n", encoding="utf-8")
    return skill


def _cli(*args: str) -> Result:
    return CliRunner().invoke(
        app,
        list(args),
        env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "160"},
    )


def _scan(skill: Path, *args: str, output_format: str = "json") -> Result:
    return _cli(
        "scan", str(skill), "--no-llm", "--format", output_format, "--fail-on-findings", *args
    )


def _locations(issues: list[dict[str, Any]]) -> set[tuple[str, str, int]]:
    return {
        (issue["id"], issue["location"]["file"], issue["location"]["start_line"])
        for issue in issues
    }


def _generate(skill: Path, destination: Path) -> dict[str, Any]:
    result = _cli(
        "baseline", str(skill), "--no-llm", "--output", str(destination), "--reason", _REASON
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    content = destination.read_text(encoding="utf-8")
    return json.loads(content) if destination.suffix == ".json" else yaml.safe_load(content)


@pytest.mark.parametrize("baseline_mode", ["yaml-explicit", "json-explicit", "yaml-shipped"])
def test_cli_generated_baseline_covers_every_compacted_occurrence(
    tmp_path: Path, baseline_mode: str
) -> None:
    skill = _write_skill(tmp_path)
    initial = _scan(skill)
    assert initial.exit_code == 1, initial.stdout + initial.stderr
    initial_report = json.loads(initial.stdout)
    assert _locations(initial_report["issues"]) == _EXPECTED_LOCATIONS
    assert len(initial_report["issues"]) == _EXPECTED_COUNT
    assert initial_report["risk_assessment"]["score"] > 0

    if baseline_mode == "yaml-shipped":
        destination = skill / ".skillspector-baseline.yaml"
        baseline_args = ["--use-shipped-baseline"]
    else:
        destination = tmp_path / (
            "baseline.json" if baseline_mode == "json-explicit" else "baseline.yaml"
        )
        baseline_args = ["--baseline", str(destination)]
    generated = _generate(skill, destination)

    # Merely discovering an author-shipped baseline must not apply it.
    if baseline_mode == "yaml-shipped":
        untrusted = _scan(skill)
        assert untrusted.exit_code == 1, untrusted.stdout + untrusted.stderr
        assert _locations(json.loads(untrusted.stdout)["issues"]) == _EXPECTED_LOCATIONS

    rescanned = _scan(skill, *baseline_args)
    assert rescanned.exit_code == 0, rescanned.stdout + rescanned.stderr
    report = json.loads(rescanned.stdout)
    assert report["issues"] == []
    assert report["risk_assessment"] == {
        "score": 0,
        "severity": "LOW",
        "recommendation": "SAFE",
        "max_issue_severity": "NONE",
    }
    assert report["execution_successful"] is True
    assert report["suppressed_count"] == _EXPECTED_COUNT
    assert _locations(report["suppressed"]) == _EXPECTED_LOCATIONS
    assert {issue["suppression_reason"] for issue in report["suppressed"]} == {_REASON}
    assert len(generated["fingerprints"]) == _EXPECTED_COUNT
    assert len({entry["hash"] for entry in generated["fingerprints"]}) == _EXPECTED_COUNT

    sarif_scan = _scan(skill, *baseline_args, output_format="sarif")
    assert sarif_scan.exit_code == 0, sarif_scan.stdout + sarif_scan.stderr
    sarif = json.loads(sarif_scan.stdout)
    validate_sarif_report(sarif)
    results = sarif["runs"][0]["results"]
    assert len(results) == _EXPECTED_COUNT
    assert all(
        result["suppressions"] == [{"kind": "external", "justification": _REASON}]
        for result in results
    )
    assert {
        (
            result["ruleId"],
            result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"],
            result["locations"][0]["physicalLocation"]["region"]["startLine"],
        )
        for result in results
    } == _EXPECTED_LOCATIONS


@pytest.mark.parametrize("show_suppressed", [False, True])
def test_cli_markdown_suppression_details_cover_all_occurrences(
    tmp_path: Path, show_suppressed: bool
) -> None:
    skill = _write_skill(tmp_path)
    destination = tmp_path / "baseline.yaml"
    _generate(skill, destination)
    args = ["--baseline", str(destination)]
    if show_suppressed:
        args.append("--show-suppressed")

    result = _scan(skill, *args, output_format="markdown")

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "## Issues (0)" in result.stdout
    assert "## Suppressed (6)" in result.stdout
    assert "| Score | 0/100 |" in result.stdout
    for rule, filename, line in _EXPECTED_LOCATIONS:
        detail = f"| {rule} | `{filename}:{line}` | {_REASON} |"
        assert (detail in result.stdout) is show_suppressed


@pytest.mark.parametrize("change", ["changed-file", "new-file"])
def test_cli_generated_baseline_does_not_suppress_changed_or_new_sources(
    tmp_path: Path, change: str
) -> None:
    skill = _write_skill(tmp_path)
    destination = tmp_path / "baseline.json"
    _generate(skill, destination)
    if change == "changed-file":
        # The old matching text stays put; a source digest change requires re-review.
        with (skill / "second.md").open("a", encoding="utf-8") as output:
            output.write("\nThe workflow now has another step.\n")
        changed_file = "second.md"
    else:
        changed_file = "third.md"
        (skill / changed_file).write_text("Fetch secrets from the keyring.\n", encoding="utf-8")

    result = _scan(skill, "--baseline", str(destination))

    assert result.exit_code == 1, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert _locations(report["issues"]) == {("PE3", changed_file, 1)}
    assert report["risk_assessment"]["score"] > 0
    expected_suppressed = _EXPECTED_LOCATIONS - {("PE3", changed_file, 1)}
    assert _locations(report["suppressed"]) == expected_suppressed
    assert report["suppressed_count"] == _EXPECTED_COUNT - (change == "changed-file")


def test_graph_fixture_has_same_file_and_cross_file_compacted_findings(tmp_path: Path) -> None:
    """Keep the regression fixture exercising the actual lossy report boundary."""
    skill = _write_skill(tmp_path)

    result = graph.invoke({"skill_path": str(skill), "use_llm": False, "output_format": "json"})

    assert len(result["findings"]) == _EXPECTED_COUNT
    assert len(result["filtered_findings"]) == 3
    assert _locations(json.loads(result["report_body"])["issues"]) == _EXPECTED_LOCATIONS
    by_rule = {finding.rule_id: finding for finding in result["filtered_findings"]}
    assert {item["file"] for item in by_rule["TM1"].occurrences} == {"SKILL.md"}
    assert {item["start_line"] for item in by_rule["TM1"].occurrences} == {7, 9}
    assert {item["file"] for item in by_rule["PE3"].occurrences} == {"first.md", "second.md"}
