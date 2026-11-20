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


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_cli_baseline_control_character_reason_round_trip(tmp_path: Path, suffix: str) -> None:
    skill = _write_skill(tmp_path)
    output = tmp_path / f"baseline{suffix}"
    reason = "Accepted 🚀 \x7f\x80\x85\x9f\ufffe\uffff controls"
    generated = _cli("baseline", str(skill), "--no-llm", "-o", str(output), "--reason", reason)
    assert generated.exit_code == 0, generated.stdout + generated.stderr
    rescanned = _scan(skill, "--baseline", str(output))
    assert rescanned.exit_code == 0, rescanned.stdout + rescanned.stderr
    report = json.loads(rescanned.stdout)
    assert report["issues"] == []
    assert {finding["suppression_reason"] for finding in report["suppressed"]} == {reason}
    assert report["suppressed_count"] == _EXPECTED_COUNT


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
@pytest.mark.parametrize(
    "filename",
    ["notes\u2028--- x.md", "notes \u2029 y.md", "notes\x7f\x85\x86.md"],
    ids=["line-separator-document-marker", "paragraph-separator-spaces", "del-c1-controls"],
)
def test_cli_baseline_untrusted_file_name_round_trip(
    tmp_path: Path, suffix: str, filename: str
) -> None:
    """File names come from the scanned skill and must survive the loader exactly."""
    skill = tmp_path / "example-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: baseline-names\ndescription: Example file names.\n---\n# Example\n",
        encoding="utf-8",
    )
    try:
        (skill / filename).write_text("Fetch secrets from the keyring.\n", encoding="utf-8")
    except (OSError, UnicodeEncodeError):
        pytest.skip("filesystem cannot store this file name")
    output = tmp_path / f"baseline{suffix}"

    generated = _cli("baseline", str(skill), "--no-llm", "-o", str(output), "--reason", _REASON)

    assert generated.exit_code == 0, generated.stdout + generated.stderr
    # PyYAML reads both formats, exactly as load_baseline does.
    written = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert {entry["file"] for entry in written["fingerprints"]} == {filename}
    rescanned = _scan(skill, "--baseline", str(output))
    assert rescanned.exit_code == 0, rescanned.stdout + rescanned.stderr
    report = json.loads(rescanned.stdout)
    assert report["issues"] == []
    assert report["suppressed_count"] == len(written["fingerprints"])
    assert {issue["location"]["file"] for issue in report["suppressed"]} == {filename}
    assert {issue["suppression_reason"] for issue in report["suppressed"]} == {_REASON}


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


@pytest.mark.parametrize("extension", ["yaml", "json"])
@pytest.mark.parametrize("existing_output", [False, True])
def test_cli_failed_scan_cannot_create_or_replace_baseline(
    tmp_path: Path, extension: str, existing_output: bool
) -> None:
    skill = _write_skill(tmp_path)
    # Unsupported primary text is a real fatal acquisition outcome. Readable
    # sibling files still produce static findings, reproducing the unsafe path.
    primary = skill / "SKILL.md"
    primary.write_bytes(primary.read_text(encoding="utf-8").encode("utf-16"))
    initial = _scan(skill)
    assert initial.exit_code == 2, initial.stdout + initial.stderr
    report = json.loads(initial.stdout)
    assert report["execution_successful"] is False
    assert report["analysis_completeness"]["status"] == "failed"
    assert any(issue["id"] == "PE3" for issue in report["issues"])

    destination = tmp_path / f"baseline.{extension}"
    previous = b"An existing reviewed baseline must remain byte-for-byte intact.\n"
    if existing_output:
        destination.write_bytes(previous)
    result = _cli("baseline", str(skill), "--no-llm", "--output", str(destination))

    assert result.exit_code == 2, result.stdout + result.stderr
    assert "scan execution failed" in result.stderr
    assert "Wrote baseline" not in result.stdout
    if existing_output:
        assert destination.read_bytes() == previous
    else:
        assert not destination.exists()


def test_cli_partial_scan_baseline_warns_and_does_not_clear_coverage_gaps(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path)
    with (skill / "SKILL.md").open("a", encoding="utf-8") as primary:
        primary.write("\nRead [opaque instructions](payload.bin).\n")
    (skill / "payload.bin").write_bytes(bytes([0x80, 0x81, 0x82, 0x83, 0, 0xFF]) * 20)
    initial = _scan(skill)
    assert initial.exit_code == 1, initial.stdout + initial.stderr
    initial_report = json.loads(initial.stdout)
    assert initial_report["execution_successful"] is True
    assert initial_report["analysis_completeness"]["status"] == "partial"
    assert initial_report["issues"]

    destination = tmp_path / "baseline.yaml"
    generated = _cli("baseline", str(skill), "--no-llm", "--output", str(destination))
    assert generated.exit_code == 0, generated.stdout + generated.stderr
    assert "only observed findings" in generated.stderr
    assert "coverage gaps remain" in generated.stderr
    assert destination.exists()

    rescanned = _scan(skill, "--baseline", str(destination))
    assert rescanned.exit_code == 1, rescanned.stdout + rescanned.stderr
    report = json.loads(rescanned.stdout)
    assert report["issues"] == []
    assert report["suppressed_count"] == len(initial_report["issues"])
    assert report["analysis_completeness"]["status"] == "partial"
    assert report["risk_assessment"]["recommendation"] != "SAFE"
