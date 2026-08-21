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
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
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
    assert report["risk_assessment"]["score"] >= 51
    assert report["risk_assessment"]["severity"] in {"HIGH", "CRITICAL"}
    assert report["risk_assessment"]["recommendation"] == "DO_NOT_INSTALL"
    assert report["risk_assessment"]["max_issue_severity"] == "HIGH"
    assert any(issue["id"] == "SC8" for issue in report["issues"])


def test_sc8_directory_entry_overflow_discards_nondeterministic_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "a.pyc").write_bytes(b"\x00")
    (tmp_path / "b.pyc").write_bytes(b"\x00")
    monkeypatch.setattr(supply_chain, "MAX_SC8_DIRECTORY_ENTRIES", 1)

    result = supply_chain._scan_shipped_bytecode(str(tmp_path))

    assert result.findings == []
    assert len(result.limitations) == 1
    assert result.limitations[0].reason is LedgerReason.ARTIFACT_COUNT_LIMIT
    assert result.limitations[0].path == "SKILL.md"


def test_sc8_depth_and_output_limits_are_explicit(tmp_path: Path, monkeypatch) -> None:
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "deep.pyc").write_bytes(b"\x00")
    (tmp_path / "one.pyc").write_bytes(b"\x00")
    (tmp_path / "two.pyc").write_bytes(b"\x00")
    monkeypatch.setattr(supply_chain, "MAX_SC8_TRAVERSAL_DEPTH", 1)
    monkeypatch.setattr(supply_chain, "MAX_SC8_FINDINGS", 1)

    result = supply_chain._scan_shipped_bytecode(str(tmp_path))

    assert [finding.file for finding in result.findings] == ["one.pyc"]
    reasons = {limitation.reason for limitation in result.limitations}
    assert reasons == {LedgerReason.OUTPUT_LIMIT}

    # With room for findings, the independently bounded deep subtree is also
    # represented rather than silently treated as clean.
    monkeypatch.setattr(supply_chain, "MAX_SC8_FINDINGS", 10)
    depth_result = supply_chain._scan_shipped_bytecode(str(tmp_path))
    assert any(
        limitation.reason is LedgerReason.TRAVERSAL_DEPTH_LIMIT
        for limitation in depth_result.limitations
    )


def test_sc8_expired_deadline_is_partial_in_node_ledger(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "payload.pyc").write_bytes(b"\x00")
    monkeypatch.setattr(supply_chain, "MAX_SC8_ANALYSIS_SECONDS", 0.0)
    monkeypatch.setattr(
        supply_chain.static_runner,
        "run_static_patterns_with_ledger",
        lambda _state, _modules: {
            "findings": [],
            "inspection_ledger": [],
            "analyzer_status_events": [],
        },
    )

    response = supply_chain.node(
        {
            "skill_path": str(tmp_path),
            "components": [],
            "file_cache": {},
            "local_file_cache": {},
            "manifest": {},
            "component_metadata": [],
        }
    )

    partial = [
        event
        for event in response["inspection_ledger"]
        if event["outcome"] is LedgerOutcome.PARTIAL
    ]
    assert len(partial) == 1
    assert partial[0]["reason_code"] is LedgerReason.RUNTIME_LIMIT
    assert response["analyzer_status_events"][0]["status"] == "degraded"


def test_sc8_cli_projects_truncation_into_analysis_completeness(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "SKILL.md").write_text("---\nname: bounded\n---\n", encoding="utf-8")
    (tmp_path / "payload.pyc").write_bytes(b"\x00")
    monkeypatch.setattr(supply_chain, "MAX_SC8_DIRECTORY_ENTRIES", 1)

    result = CliRunner().invoke(
        app,
        ["scan", str(tmp_path), "--format", "json", "--no-llm"],
    )

    assert result.exit_code in {0, 1}, result.output
    report = json.loads(result.output)
    completeness = report["analysis_completeness"]
    assert completeness["is_complete"] is False
    assert completeness["status"] == "partial"
    assert any(
        item["reason_code"] == LedgerReason.ARTIFACT_COUNT_LIMIT.value
        for item in completeness["ledger_exceptions"]
    )
