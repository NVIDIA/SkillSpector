import json
from pathlib import Path

from skillspector.security_inspection.diff_security import diff_against_previous_version
from skillspector.security_inspection.scanner import SecurityScanner


def test_hostile_git_config_not_executed(tmp_path: Path) -> None:
    skill_dir = tmp_path / "hostile_skill"
    skill_dir.mkdir()
    (skill_dir / "skill.json").write_text(
        json.dumps({"name": "hostile", "version": "1.0.0"}), encoding="utf-8"
    )
    (skill_dir / "main.py").write_text("print('hello')\n", encoding="utf-8")

    canary = tmp_path / "canary_executed.txt"
    bad_script = tmp_path / "evil.sh"
    bad_script.write_text(f"#!/bin/sh\necho pwned > '{canary}'\n", encoding="utf-8")
    bad_script.chmod(0o755)

    git_dir = skill_dir / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text(
        f"[diff]\n    external = {bad_script}\n[core]\n    fsmonitor = {bad_script}\n",
        encoding="utf-8",
    )

    diff_against_previous_version(skill_dir)
    assert not canary.exists(), (
        "Hostile .git/config was executed during diff_against_previous_version!"
    )

    scanner = SecurityScanner(skill_dir)
    scanner.scan()
    assert not canary.exists(), "Hostile .git/config was executed during scanner.scan()!"
