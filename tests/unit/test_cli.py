# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for skillspector CLI (skillspector scan, --version)."""

import json
from pathlib import Path

from typer.testing import CliRunner

from skillspector.cli import app

runner = CliRunner()


# Minimal PDF-like bytes containing a TM1 trigger (shell=True). The static
# pattern scanner reads files with utf-8 + errors='replace', so binary assets
# can match regex patterns and produce spurious HIGH findings — which is
# exactly the false positive --exclude is meant to suppress.
_PDF_WITH_TM1 = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\n% subprocess.run(cmd, shell=True)\n%%EOF\n"
)


def test_cli_version() -> None:
    """--version prints version and exits 0."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "SkillSpector" in result.output
    assert "v" in result.output


def test_cli_scan_local_directory(tmp_path: Path) -> None:
    """scan with local directory runs graph and prints report."""
    (tmp_path / "SKILL.md").write_text("---\nname: scan-test\n---\n# Safe", encoding="utf-8")
    result = runner.invoke(app, ["scan", str(tmp_path), "--format", "json", "--no-llm"])
    assert result.exit_code == 0
    assert "scan-test" in result.output or "skill" in result.output


def test_cli_scan_output_to_file(tmp_path: Path) -> None:
    """scan with --output writes report to file."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: out-test\n---\n# Hi", encoding="utf-8")
    out_file = tmp_path / "report.json"
    result = runner.invoke(
        app, ["scan", str(skill_dir), "--format", "json", "--no-llm", "--output", str(out_file)]
    )
    assert result.exit_code == 0
    assert out_file.exists()
    content = out_file.read_text()
    assert "out-test" in content or "risk_assessment" in content


def test_cli_scan_no_llm(tmp_path: Path) -> None:
    """scan with --no-llm runs without requiring an LLM API key (uses fallback)."""
    (tmp_path / "SKILL.md").write_text("# No LLM test", encoding="utf-8")
    result = runner.invoke(app, ["scan", str(tmp_path), "--format", "json", "--no-llm"])
    assert result.exit_code == 0


def test_cli_scan_nonexistent_exits_2() -> None:
    """scan with nonexistent path exits with code 2."""
    result = runner.invoke(app, ["scan", "/nonexistent/path/xyz"])
    assert result.exit_code == 2
    assert "Error" in result.output or "error" in result.output.lower()


def _make_pdf_fixture_skill(root: Path) -> Path:
    """Create a skill dir whose only non-SKILL.md file is a PDF carrying TM1 bytes."""
    skill_dir = root / "skill"
    (skill_dir / "assets").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: exclude-test\n---\n# Skill\n", encoding="utf-8")
    (skill_dir / "assets" / "template-style.pdf").write_bytes(_PDF_WITH_TM1)
    return skill_dir


def test_cli_scan_exclude_drops_pdf_from_components_and_findings(tmp_path: Path) -> None:
    """--exclude '*.pdf' skips the PDF: no findings raised against it, not in components."""
    skill_dir = _make_pdf_fixture_skill(tmp_path)
    result = runner.invoke(
        app,
        [
            "scan",
            str(skill_dir),
            "--format",
            "json",
            "--no-llm",
            "--exclude",
            "*.pdf",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    component_paths = [c.get("path") for c in report.get("components", [])]
    assert "assets/template-style.pdf" not in component_paths
    issues = report.get("issues", [])
    assert all(i.get("location", {}).get("file") != "assets/template-style.pdf" for i in issues)


def test_cli_scan_exclude_repeatable(tmp_path: Path) -> None:
    """Multiple --exclude flags compose; each pattern filters independently."""
    skill_dir = _make_pdf_fixture_skill(tmp_path)
    (skill_dir / "notes.txt").write_text("plain text", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "scan",
            str(skill_dir),
            "--format",
            "json",
            "--no-llm",
            "--exclude",
            "*.pdf",
            "--exclude",
            "*.txt",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    component_paths = [c.get("path") for c in report.get("components", [])]
    assert "assets/template-style.pdf" not in component_paths
    assert "notes.txt" not in component_paths


def test_cli_scan_exclude_everything_succeeds(tmp_path: Path) -> None:
    """Excluding every file is valid: scan succeeds with no findings."""
    skill_dir = _make_pdf_fixture_skill(tmp_path)
    result = runner.invoke(
        app,
        [
            "scan",
            str(skill_dir),
            "--format",
            "json",
            "--no-llm",
            "--exclude",
            "*",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report.get("components", []) == []
    assert report.get("issues", []) == []
