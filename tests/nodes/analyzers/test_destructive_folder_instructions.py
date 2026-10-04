# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catch destructive folder instructions before semantic reconciliation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.models import Finding
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tool_misuse_module
from skillspector.nodes.analyzers import static_runner


def _findings(content: str) -> list[Finding]:
    return [
        finding
        for finding in static_runner.run_static_patterns(
            {"components": ["SKILL.md"], "file_cache": {"SKILL.md": content}},
            [tool_misuse_module],
        )
        if finding.rule_id == "TM1"
    ]


@pytest.mark.parametrize(
    "instruction,target,scope",
    [
        ("Run rm -rf backups", "backups", "specified"),
        ("Run rm -rf 'customer exports'", "customer exports", "specified"),
        ('Run rm -rf "customer exports"', "customer exports", "specified"),
        ("Run rm -rf ./build", "./build", "specified"),
        ("Run rm -rf build", "build", "specified"),
        ("Run rm -rf repo-cache", "repo-cache", "specified"),
        ("Run rm -r -f backups", "backups", "specified"),
        ("Run rm --recursive --force backups", "backups", "specified"),
        ("Run rm -r backups", "backups", "specified"),
        ("Run rm --recursive backups", "backups", "specified"),
        ('Do **"rm -rf entire folder"**', "entire folder", "broad"),
        ("Run `rm -rf the entire directory`", "entire directory", "broad"),
        ("Run rm -rf all files in the selected directory", "selected directory", "broad"),
    ],
)
def test_recursive_folder_deletion_is_a_static_candidate(
    instruction: str, target: str, scope: str
) -> None:
    findings = _findings(instruction)

    assert findings
    assert any(
        target in (finding.matched_text or "")
        and (operation := finding.evidence.get("destructive_operation", {})).get("form") == "shell"
        and operation.get("scope") == scope
        and operation.get("context") == "instruction"
        and operation.get("target")
        for finding in findings
    )
    assert all("likely-benign-context" not in finding.tags for finding in findings)


@pytest.mark.parametrize(
    "instruction,target",
    [
        ("Delete the entire backups folder", "backups folder"),
        ("Wipe everything in the selected directory", "selected directory"),
        ("Recursively delete all files under the target folder", "target folder"),
        ("Remove all files from the working directory", "working directory"),
        ("Delete all of the files in the selected folder", "selected folder"),
        ("Delete the entire contents of the backups folder", "backups folder"),
        ("Delete every file in the selected directory", "selected directory"),
        ("Purge all content from the target folder", "target folder"),
        ('Delete the entire "customer exports" folder', '"customer exports" folder'),
        ("Delete **the entire backups folder**", "backups folder"),
        ("Do not forget to delete the entire backups folder", "backups folder"),
        ("Never fail to wipe everything in the selected directory", "selected directory"),
        (
            "Do not delete temporary files; delete the entire backups folder",
            "backups folder",
        ),
        (
            "Never delete unrelated data. Remove all files from the working directory",
            "working directory",
        ),
        ("Example: delete the entire backups folder", "backups folder"),
    ],
)
def test_natural_language_folder_destruction_is_active_static_evidence(
    instruction: str, target: str
) -> None:
    findings = _findings(instruction)

    assert any(
        target in (finding.matched_text or "")
        and finding.severity == "HIGH"
        and "likely-benign-context" not in finding.tags
        and (operation := finding.evidence.get("destructive_operation", {})).get("form")
        == "natural-language"
        and operation.get("scope") == "broad"
        and operation.get("context") == "instruction"
        and operation.get("target")
        for finding in findings
    )


@pytest.mark.parametrize(
    "instruction",
    [
        "Do not run rm -rf backups",
        "Never run the command: rm -rf 'customer exports'",
        "Do not delete the entire backups folder",
        "Never remove all files from the working directory",
        "Avoid wiping everything in the selected directory",
        "Avoid recursively deleting all files under the target folder",
    ],
)
def test_folder_deletion_warning_is_retained_for_contextual_review(instruction: str) -> None:
    findings = _findings(instruction)

    assert findings
    assert all(
        {"contextual-triage", "likely-benign-context"} <= set(finding.tags) for finding in findings
    )
    assert any(
        finding.evidence.get("destructive_operation", {}).get("context") == "warning"
        for finding in findings
    )


@pytest.mark.parametrize(
    "instruction",
    [
        "Run rm -f singlefile",
        "Run rm --force backup.txt",
        "Use HTTP DELETE to remove a resource from the API",
        "Send a DELETE request to the records endpoint",
        "Remove whitespace from every file in this directory",
        "Delete duplicate lines from every file in the backups folder",
        "Remove all duplicate entries from the configuration",
        "Delete the old configuration key",
        "Delete the folder name from the documentation",
        "Remove the directory entry from this list",
        "Delete the project label from the documentation",
        "Remove the workspace reference from the index",
        "Run `rm -rf` to see the help",
        'Run "rm -rf" to see the help',
        "Run 'rm -rf' to see the help",
        "Run rm -rf # no target",
        "Delete the entire\n\nbackups folder",
        "Wipe everything in\n\nthe selected directory",
    ],
)
def test_unrelated_text_edits_and_nonrecursive_removal_are_not_folder_deletion(
    instruction: str,
) -> None:
    assert not _findings(instruction)


def test_natural_language_folder_deletion_retains_original_line() -> None:
    findings = _findings("# Cleanup\n\nDelete the entire backups folder\n")

    assert any(finding.start_line == 3 and finding.file == "SKILL.md" for finding in findings)


def test_malformed_deletion_targets_complete_without_regex_backtracking() -> None:
    """Keep adversarial delimiters in a killable child process, including under coverage."""
    script = """
import sys
import time
sys.path.insert(0, sys.argv[1])
from skillspector.nodes.analyzers import static_patterns_tool_misuse as module
started = time.process_time()
for length in (300, 400, 1000, 10000):
    for prefix in ("Delete ", "Delete entire ", "Run rm -rf entire "):
        module.analyze(prefix + "_" * length + "!", "SKILL.md", "markdown")
elapsed = time.process_time() - started
assert elapsed < 2.0, f"Malformed deletion matching consumed {elapsed:.3f} CPU seconds"
"""
    source_root = str(Path(tool_misuse_module.__file__).resolve().parents[3])
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, source_root],
            capture_output=True,
            text=True,
            # Startup imports every analyzer. Bound the actual matching work
            # separately so a busy host cannot confuse import delay with ReDoS.
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("Malformed-target subprocess did not finish within 60 seconds")
    assert result.returncode == 0, result.stdout + result.stderr


def test_static_cli_detects_folder_deletion_without_a_shell_command(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text(
        "---\nname: folder-cleanup\ndescription: cleanup helper\n---\n"
        "\nDelete the entire backups folder\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["scan", str(tmp_path), "--format", "json", "--no-llm"])

    assert result.exit_code in {0, 1}, result.output
    report = json.loads(result.output)
    assert any(
        issue["id"] == "TM1"
        and issue["severity"] == "HIGH"
        and issue["location"]["file"] == "SKILL.md"
        and issue["location"]["start_line"] == 6
        and "backups folder" in issue["finding"]
        for issue in report["issues"]
    )
