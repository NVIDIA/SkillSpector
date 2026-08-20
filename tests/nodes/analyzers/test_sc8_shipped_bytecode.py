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

import json
from pathlib import Path

from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.nodes.analyzers import static_patterns_supply_chain as supply_chain


def test_sc8_flags_pycache_and_pyc(tmp_path: Path) -> None:
    cache = tmp_path / "scripts" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "evil.cpython-312.pyc").write_bytes(b"\x00")
    (tmp_path / "orphan.pyc").write_bytes(b"\x00")
    (tmp_path / "clean.py").write_text("print('ok')\n", encoding="utf-8")

    findings = supply_chain._analyze_shipped_bytecode(str(tmp_path))
    rule_ids = {f.rule_id for f in findings}
    assert rule_ids == {"SC8"}
    paths = {f.file for f in findings}
    assert "scripts/__pycache__/" in paths
    assert "scripts/__pycache__/evil.cpython-312.pyc" in paths
    assert "orphan.pyc" in paths
    assert all(f.severity == "HIGH" for f in findings)


def test_sc8_clean_tree_has_no_findings(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text("# demo\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    assert supply_chain._analyze_shipped_bytecode(str(tmp_path)) == []


def test_sc8_single_pyc_blocks_install_and_cli_exit(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text(
        "---\nname: shipped-bytecode\n---\n# Shipped bytecode\n", encoding="utf-8"
    )
    (tmp_path / "payload.pyc").write_bytes(b"\x00")

    result = CliRunner().invoke(
        app,
        ["scan", str(tmp_path), "--format", "json", "--no-llm"],
    )

    assert result.exit_code == 1, result.output
    report = json.loads(result.output)
    assert report["risk_assessment"] == {
        "score": 51,
        "severity": "HIGH",
        "recommendation": "DO_NOT_INSTALL",
        "max_issue_severity": "HIGH",
    }
    assert any(issue["id"] == "SC8" for issue in report["issues"])
