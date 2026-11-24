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

"""SC11: a symlink shipped inside a skill bundle is a finding, not only a coverage event.

Discovery already refuses to follow links (#374, #495). These tests pin that the
link itself is reported, with severity driven by where it points, and that the
walker still never reads through it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from skillspector.nodes.analyzers import static_patterns_supply_chain as supply_chain


def _link(target: Path | str, link: Path) -> None:
    try:
        os.symlink(str(target), str(link))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")


def _sc11(skill_dir: Path):
    return [
        f for f in supply_chain._analyze_shipped_bytecode(str(skill_dir)) if f.rule_id == "SC11"
    ]


def test_sc11_outward_link_is_high(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "notes.txt").write_text("secret\n", encoding="utf-8")
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# demo\n", encoding="utf-8")
    _link(outside, skill / "helpdocs")

    findings = _sc11(skill)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.file == "helpdocs"
    assert finding.severity == "HIGH"
    assert "outside the skill bundle" in finding.message
    # Evidence stays skill-relative: no resolved host path is reported.
    assert str(outside) not in (finding.context or "")
    assert str(outside.resolve()) not in (finding.context or "")


def test_sc11_sensitive_target_is_high_even_when_dangling(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# demo\n", encoding="utf-8")
    _link("~/.claude/settings.json", skill / "project_settings.json")

    findings = _sc11(skill)

    assert len(findings) == 1
    assert findings[0].severity == "HIGH"
    assert "credential store or agent configuration" in findings[0].message


def test_sc11_dangling_internal_link_is_high(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# demo\n", encoding="utf-8")
    _link("missing.txt", skill / "docs.txt")

    findings = _sc11(skill)

    assert len(findings) == 1
    assert findings[0].severity == "HIGH"
    assert "dangling" in findings[0].message


def test_sc11_internal_link_is_medium(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# demo\n", encoding="utf-8")
    (skill / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    _link("AGENTS.md", skill / "CLAUDE.md")

    findings = _sc11(skill)

    assert len(findings) == 1
    assert findings[0].severity == "MEDIUM"
    assert findings[0].file == "CLAUDE.md"
    assert findings[0].matched_text == "CLAUDE.md -> AGENTS.md"


def test_sc11_does_not_descend_through_directory_links(tmp_path: Path) -> None:
    outside = tmp_path / "outside" / "__pycache__"
    outside.mkdir(parents=True)
    (outside / "evil.pyc").write_bytes(b"\x00")
    skill = tmp_path / "skill"
    skill.mkdir()
    _link(tmp_path / "outside", skill / "vendor")

    findings = supply_chain._analyze_shipped_bytecode(str(skill))

    assert {f.rule_id for f in findings} == {"SC11"}
    assert not any(f.rule_id == "SC8" for f in findings)


def test_sc11_clean_tree_has_no_findings(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text("# demo\n", encoding="utf-8")
    assert _sc11(tmp_path) == []


def test_sc11_internal_link_stays_medium_when_skill_lives_under_dot_claude(tmp_path: Path) -> None:
    skill = tmp_path / ".claude" / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# demo\n", encoding="utf-8")
    (skill / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    _link("AGENTS.md", skill / "CLAUDE.md")

    findings = _sc11(skill)

    assert len(findings) == 1
    assert findings[0].severity == "MEDIUM"


def test_sc11_reports_links_under_sc8_skip_dirs_without_sc8(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    skill = tmp_path / "skill"
    vendor = skill / "node_modules" / "pkg"
    vendor.mkdir(parents=True)
    (vendor / "__pycache__").mkdir()
    (vendor / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    _link(outside, vendor / "helpdocs")

    findings = supply_chain._analyze_shipped_bytecode(str(skill))

    sc11 = [f for f in findings if f.rule_id == "SC11"]
    assert [f.file for f in sc11] == ["node_modules/pkg/helpdocs"]
    assert sc11[0].severity == "HIGH"
    assert not any(f.rule_id == "SC8" for f in findings)


@pytest.mark.skipif(os.name != "nt", reason="NTFS junctions exist only on Windows")
def test_sc11_reports_windows_junction_and_does_not_descend(tmp_path: Path) -> None:
    import subprocess

    outside = tmp_path / "outside" / "__pycache__"
    outside.mkdir(parents=True)
    (outside / "evil.pyc").write_bytes(b"\x00")
    skill = tmp_path / "skill"
    skill.mkdir()
    junction = skill / "vendor"
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(tmp_path / "outside")],
        check=True,
        capture_output=True,
    )

    findings = supply_chain._analyze_shipped_bytecode(str(skill))

    assert {f.rule_id for f in findings} == {"SC11"}
    assert findings[0].file == "vendor"


def _forbid_access_outside(monkeypatch: pytest.MonkeyPatch, root: Path) -> list[str]:
    """Record metadata probes on any path outside *root*.

    Violations are recorded rather than raised: raising inside os.stat would also break
    pytest's own failure reporting while the patches are active. Callers undo the patches
    before asserting.
    """
    root_text = os.path.abspath(root)
    violations: list[str] = []
    real_lstat, real_stat = os.lstat, os.stat
    real_exists, real_realpath = os.path.exists, os.path.realpath

    def check(path: object) -> None:
        text = os.path.abspath(os.fspath(path))
        if not (text == root_text or text.startswith(root_text + os.sep)):
            violations.append(text)

    def lstat(path, *args, **kwargs):
        check(path)
        return real_lstat(path, *args, **kwargs)

    def stat_(path, *args, **kwargs):
        check(path)
        return real_stat(path, *args, **kwargs)

    def exists(path):
        check(path)
        return real_exists(path)

    def realpath(path, *args, **kwargs):
        check(path)
        return real_realpath(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", lstat)
    monkeypatch.setattr(os, "stat", stat_)
    monkeypatch.setattr(os.path, "exists", exists)
    monkeypatch.setattr(os.path, "realpath", realpath)
    return violations


@pytest.mark.parametrize(
    "target",
    [
        "/etc",
        "/etc/passwd",
        "\\\\attacker\\share\\payload",
        "//attacker/share/payload",
        "../../outside",
    ],
)
def test_sc11_outward_and_unc_targets_are_high_without_accessing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# demo\n", encoding="utf-8")
    _link(target, skill / "helpdocs")
    violations = _forbid_access_outside(monkeypatch, skill)

    findings = _sc11(skill)
    monkeypatch.undo()

    assert violations == []
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"
    assert "destination was not accessed" in (findings[0].context or "")


def test_sc11_internal_chain_that_escapes_is_high_without_accessing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = tmp_path / "skill"
    (skill / "docs").mkdir(parents=True)
    _link("../escape", skill / "docs" / "guide")
    _link("../../outside", skill / "escape")
    violations = _forbid_access_outside(monkeypatch, skill)

    findings = {f.file: f for f in _sc11(skill)}
    monkeypatch.undo()

    assert violations == []
    assert findings["docs/guide"].severity == "HIGH"
    assert "outside the skill bundle" in findings["docs/guide"].message
    assert findings["escape"].severity == "HIGH"


def test_sc11_absolute_internal_target_under_dot_claude_install_stays_medium(
    tmp_path: Path,
) -> None:
    skill = tmp_path / ".claude" / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    _link(str(skill / "AGENTS.md"), skill / "CLAUDE.md")

    findings = _sc11(skill)

    assert len(findings) == 1
    assert findings[0].severity == "MEDIUM"
    assert "resolves to AGENTS.md" in (findings[0].context or "")


def test_sc11_internal_link_to_sensitive_bundle_path_is_high(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    (skill / "config").mkdir(parents=True)
    (skill / "config" / ".env").write_text("X=1\n", encoding="utf-8")
    _link("config/.env", skill / "settings")

    findings = _sc11(skill)

    assert findings[0].severity == "HIGH"


def test_sc11_links_only_trees_do_not_consume_bytecode_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = tmp_path / "skill"
    vendor = skill / ".git" / "objects"
    vendor.mkdir(parents=True)
    for index in range(40):
        (vendor / f"blob{index:02d}").write_bytes(b"x")
    cache = skill / "scripts" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "evil.cpython-312.pyc").write_bytes(b"\x00")
    monkeypatch.setattr(supply_chain, "MAX_SC8_DISCOVERED_ENTRIES", 6)
    monkeypatch.setattr(supply_chain, "MAX_SC11_LINK_ONLY_ENTRIES", 5)

    result = supply_chain._scan_shipped_bytecode(str(skill))

    sc8_paths = {f.file for f in result.findings if f.rule_id == "SC8"}
    assert "scripts/__pycache__/evil.cpython-312.pyc" in sc8_paths
    # The vendor tree hit its own budget and says so, without stopping SC8.
    assert any(
        limitation.path.startswith(".git") and limitation.limit_artifacts == 5
        for limitation in result.limitations
    )
