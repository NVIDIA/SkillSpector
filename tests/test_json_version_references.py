# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON version literals are metadata, not references to local artifacts."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.references import resolve_bundle_references_with_metadata


@pytest.mark.parametrize(
    "body",
    [
        '"VERSION": "1.0.0",\n',
        '```json\n{\n  "VERSION": "1.0.0",\n  "name": "example"\n}\n```\n',
        '"VERSION" : "22.4.7"\n',
    ],
)
def test_strict_json_version_metadata_is_not_a_missing_file(tmp_path: Path, body: str) -> None:
    result = resolve_bundle_references_with_metadata(
        tmp_path,
        source_path="SKILL.md",
        source_text=body,
        known_paths=["SKILL.md"],
    )
    assert result.complete
    assert result.records == []
    assert result.accepted_references == 0


@pytest.mark.parametrize(
    ("body", "expected_missing"),
    [
        ('"version": "1.0.0",\n', "1.0.0"),
        ('"VERSION": "1.x",\n', "1.x"),
        ('"VERSION": "0.0.0.0",\n', "0.0.0.0"),
        ('"VERSION": "1.0.0", "file": "missing.md"\n', "missing.md"),
        ("Read [version artifact](1.0.0).\n", "1.0.0"),
        ('Read "1.0.0".\n', "1.0.0"),
        ("Run `python3 scripts/emit.py`.\n", "scripts/emit.py"),
    ],
)
def test_json_version_exemption_preserves_other_missing_references(
    tmp_path: Path, body: str, expected_missing: str
) -> None:
    result = resolve_bundle_references_with_metadata(
        tmp_path,
        source_path="SKILL.md",
        source_text=body,
        known_paths=["SKILL.md"],
    )
    assert result.complete
    assert any(
        r["status"] == "missing" and r["column"] == body.index(expected_missing) + 1
        for r in result.records
    )


@pytest.mark.parametrize("present", [True, False])
def test_version_shaped_filename_is_still_a_reference(tmp_path: Path, present: bool) -> None:
    result = resolve_bundle_references_with_metadata(
        tmp_path,
        source_path="SKILL.md",
        source_text="Read [manual](1.0.0).\n",
        known_paths=["SKILL.md", "1.0.0"] if present else ["SKILL.md"],
    )
    assert result.complete
    assert len(result.records) == 1
    assert result.records[0]["status"] == ("resolved" if present else "missing")


def test_json_version_only_real_cli_is_not_missing(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text(
        "---\nname: json-version-proof\ndescription: Explain a JSON object.\n---\n"
        '```json\n{\n  "VERSION": "1.0.0",\n  "name": "example"\n}\n```\n',
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app,
        ["scan", str(tmp_path), "--no-llm", "--format", "json", "--fail-on-incomplete"],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["execution_successful"] is True
    coverage = report["analysis_completeness"]
    assert not any("1.0.0" in r["evidence"] for r in coverage["references"])


def test_json_version_does_not_hide_real_missing_cli_reference(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text(
        "---\nname: json-with-missing-file\ndescription: Explain a JSON object.\n---\n"
        '```json\n{\n  "VERSION": "1.0.0",\n  "name": "example"\n}\n```\n'
        "Run `python3 scripts/emit.py` to produce a report.\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app,
        ["scan", str(tmp_path), "--no-llm", "--format", "json", "--fail-on-incomplete"],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    references = report["analysis_completeness"]["references"]
    assert not any("1.0.0" in r["evidence"] for r in references)
    assert any(r["status"] == "missing" and "scripts/emit.py" in r["evidence"] for r in references)


def test_json_version_matcher_rejects_long_malformed_suffix_promptly() -> None:
    # Isolate the matcher so a backtracking regression cannot hang the suite.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from skillspector.references import _JSON_VERSION_FIELD\n"
            'line = \'"VERSION": "1.0.0"\' + " " * 250_000 + "x"\n'
            "assert _JSON_VERSION_FIELD.fullmatch(line) is None",
        ],
        check=True,
        timeout=5,
    )
