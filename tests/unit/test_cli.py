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

from pathlib import Path

from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.nodes.build_context import _MAX_READ_BYTES

runner = CliRunner()


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


def test_cli_scan_oversized_file_exits_2(tmp_path: Path) -> None:
    """scan fails instead of silently skipping files above the analysis limit."""
    (tmp_path / "SKILL.md").write_text("---\nname: oversized\n---\n# Skill", encoding="utf-8")
    huge = tmp_path / "huge.txt"
    with huge.open("wb") as fh:
        fh.seek(_MAX_READ_BYTES)
        fh.write(b"x")

    result = runner.invoke(app, ["scan", str(tmp_path), "--format", "json", "--no-llm"])

    assert result.exit_code == 2
    assert "file size exceeds" in result.output
    assert "huge.txt" in result.output
