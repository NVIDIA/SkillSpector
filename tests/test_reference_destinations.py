# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve Markdown destinations without weakening missing-reference coverage."""

import base64
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from skillspector import references as references_module
from skillspector.cli import app
from skillspector.mcp_server import run_scan
from skillspector.references import resolve_bundle_references_with_metadata


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("Read [guide](<docs/user guide.md>).", "docs/user guide.md"),
        (r'Read [guide](<docs/user guide.md> "A \"quoted\" title").', "docs/user guide.md"),
        (r"Read [guide](<docs/user guide.md> 'A \'quoted\' title').", "docs/user guide.md"),
        (r"Read [guide](<docs/user guide.md> (A \(quoted\) title)).", "docs/user guide.md"),
        (r"Read [guide](<\<note\>.md>).", "<note>.md"),
        (r"Read [manual](\<tool.1\>).", "<tool.1>"),
        ("Read [guide](<docs/user.md guide.md>).", "docs/user.md guide.md"),
        ("Read [guide](docs/guide(v1).md).", "docs/guide(v1).md"),
        (r"Read [guide](docs/guide\(v1\).md).", "docs/guide(v1).md"),
        ("Read [guide](docs/guide(a(b(c))).md).", "docs/guide(a(b(c))).md"),
        ('Read [guide](docs/guide.md "Guide title").', "docs/guide.md"),
        ("Read [guide][manual].\n\n[manual]: docs/user%20guide.md", "docs/user guide.md"),
        ("Read [guide][manual].\n\n[manual]: <docs/user guide.md>", "docs/user guide.md"),
        ("Read [guide][manual].\n\n[manual]: docs/guide(v1).md", "docs/guide(v1).md"),
        ("Read [guide](docs/part%23one.md#summary).", "docs/part#one.md"),
        ("Read [guide](docs/part%3Fone.md?view=1).", "docs/part?one.md"),
        ("Read [guide](docs/part%252Fone.md).", "docs/part%2Fone.md"),
        ("Read [guide](docs/caf%C3%A9.md).", "docs/café.md"),
        ("Read [manual](tool.1).", "tool.1"),
        ("Read `tool.1`.", "tool.1"),
    ],
)
@pytest.mark.parametrize("present", [True, False])
def test_markdown_destinations_preserve_present_and_missing_targets(
    tmp_path: Path, source: str, target: str, present: bool
) -> None:
    result = resolve_bundle_references_with_metadata(
        tmp_path,
        source_path="SKILL.md",
        source_text=source,
        known_paths=["SKILL.md", target] if present else ["SKILL.md"],
    )
    assert result.complete is True  # Extraction completed; resolution may be missing.
    assert result.records
    assert {record["status"] for record in result.records} == {"resolved" if present else "missing"}
    assert {record["target_path"] for record in result.records} == {target if present else None}


@pytest.mark.parametrize(
    "target",
    ["../outside.md", "%2e%2e/outside.md", "%2Foutside.md", "%5Coutside.md", "C%3A/file.md"],
)
def test_decoded_destination_cannot_escape_bundle(tmp_path: Path, target: str) -> None:
    result = resolve_bundle_references_with_metadata(
        tmp_path,
        source_path="SKILL.md",
        source_text=f"Read [guide]({target}).",
        known_paths=["SKILL.md"],
    )
    assert result.complete is True
    assert result.records
    assert all(record["status"] == "rejected" for record in result.records)


def test_reference_definitions_do_not_restore_slash_prose_false_positives(tmp_path: Path) -> None:
    result = resolve_bundle_references_with_metadata(
        tmp_path,
        source_path="SKILL.md",
        source_text="Compare process I/O, reads/writes, and environment/profile settings.",
        known_paths=["SKILL.md"],
    )
    assert result.complete is True
    assert result.records == []


@pytest.mark.parametrize("channel", ["cli", "mcp"])
@pytest.mark.parametrize("present", [True, False])
@pytest.mark.parametrize(
    ("body", "target"),
    [
        ("Read [guide][manual].\n\n[manual]: docs/user%20guide.md", "docs/user guide.md"),
        ("Read [guide](<docs/user.md guide.md>).", "docs/user.md guide.md"),
        (r'Read [guide](<docs/user guide.md> "A \"quoted\" title").', "docs/user guide.md"),
        (r"Read [guide](<docs/user guide.md> 'A \'quoted\' title').", "docs/user guide.md"),
        (r"Read [guide](<docs/user guide.md> (A \(quoted\) title)).", "docs/user guide.md"),
        (r"Read [guide](<\<note\>.md>).", "<note>.md"),
        (r"Read [manual](\<tool.1\>).", "<tool.1>"),
        ("Read [guide](docs/guide(v1).md).", "docs/guide(v1).md"),
        ("Read [guide](docs/part%23one.md#summary).", "docs/part#one.md"),
    ],
)
async def test_cli_and_mcp_reference_completeness_agree(
    tmp_path: Path, body: str, target: str, present: bool, channel: str
) -> None:
    if "<" in target and os.name == "nt":
        pytest.skip("Literal angle filenames are unsupported on Windows")
    (tmp_path / "SKILL.md").write_text(
        f"---\nname: reference-control\ndescription: Summarize the guide.\n---\n{body}\n",
        encoding="utf-8",
    )
    if present:
        path = tmp_path / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Guide\nSummarize the supplied text.\n", encoding="utf-8")
    if channel == "cli":
        result = CliRunner().invoke(
            app, ["scan", str(tmp_path), "--no-llm", "--format", "json", "--fail-on-incomplete"]
        )
        assert result.exit_code == (0 if present else 1), result.output
        report = json.loads(result.stdout)
    else:
        result = await run_scan(str(tmp_path), use_llm=False, output_format="json")
        assert result["safe_to_install"] is present
        report = json.loads(result["report"])
    assert report["execution_successful"] is True
    assert report["analysis_completeness"]["is_complete"] is present
    assert report["risk_assessment"]["recommendation"] == ("SAFE" if present else "CAUTION")
    assert {r["status"] for r in report["analysis_completeness"]["references"]} == {
        "resolved" if present else "missing"
    }


@pytest.mark.parametrize(
    ("body", "target"),
    [("Read [manual](tool.1).", "tool.1"), (r"Read [manual](\<tool.1\>).", "<tool.1>")],
)
async def test_numeric_extension_reference_retains_uninspected_artifact_gap(
    tmp_path: Path, body: str, target: str
) -> None:
    if "<" in target and os.name == "nt":
        pytest.skip("Literal angle filenames are unsupported on Windows")
    (tmp_path / "SKILL.md").write_text(f"# Guide\n{body}\n", encoding="utf-8")
    if target != "tool.1":
        (tmp_path / "tool.1").write_text(
            "# Safe decoy\nSummarize supplied text.\n", encoding="utf-8"
        )
    # A real raster artifact, independent of its man-page-like filename.
    (tmp_path / target).write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6"
            "SAAAAABJRU5ErkJggg=="
        )
    )
    result = await run_scan(str(tmp_path), use_llm=False, output_format="json")
    report = json.loads(result["report"])
    assert result["safe_to_install"] is False
    assert report["analysis_completeness"]["is_complete"] is False
    assert any(r["target_path"] == target for r in report["analysis_completeness"]["references"])
    assert any(finding["id"] == "AE1" for finding in report["issues"])


@pytest.mark.parametrize("separator", ["%2F", "%5C"])
def test_encoded_directory_does_not_fall_back_to_another_basename(
    tmp_path: Path, separator: str
) -> None:
    result = resolve_bundle_references_with_metadata(
        tmp_path,
        source_path="SKILL.md",
        source_text=f"Read [guide](missing{separator}guide.md).",
        known_paths=["SKILL.md", "other/guide.md"],
    )
    assert result.complete is True
    assert len(result.records) == 1
    assert result.records[0]["status"] == "missing"
    assert result.records[0]["target_path"] is None


@pytest.mark.parametrize("body", ["[status]: All checks passed.", "[note]: Read the guide."])
def test_prose_after_bracket_label_is_not_a_reference(tmp_path: Path, body: str) -> None:
    result = resolve_bundle_references_with_metadata(
        tmp_path, source_path="SKILL.md", source_text=body, known_paths=["SKILL.md"]
    )
    assert result.complete is True
    assert result.records == []


def test_escaped_angle_filename_does_not_resolve_different_file(tmp_path: Path) -> None:
    result = resolve_bundle_references_with_metadata(
        tmp_path,
        source_path="SKILL.md",
        source_text=r"Read [manual](\<tool.1\>).",
        known_paths=["SKILL.md", "tool.1"],
    )
    assert len(result.records) == 1
    assert result.records[0]["status"] == "missing"
    assert result.records[0]["target_path"] is None


def test_malformed_markdown_openings_observe_deadline_before_yield(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = references_module._MARKDOWN_REFERENCE_START
    observed = 0

    class CountedStarts:
        def finditer(self, line: str):
            nonlocal observed
            for match in original.finditer(line):
                observed += 1
                yield match

    monkeypatch.setattr(references_module, "_MARKDOWN_REFERENCE_START", CountedStarts())
    result = resolve_bundle_references_with_metadata(
        tmp_path,
        source_path="SKILL.md",
        source_text="[x](" * 5000,
        known_paths=["SKILL.md"],
        clock=lambda: observed / 1000,
    )
    assert "runtime" in result.limitations
    assert result.complete is False
    assert observed <= 2001
