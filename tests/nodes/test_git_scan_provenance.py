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

"""Exercise real Git acquisition and artifact provenance without network access."""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from skillspector import graph
from skillspector.cleanup import remove_temp_tree
from skillspector.input_handler import TransitiveIngestTruncatedError
from skillspector.nodes.resolve_input import resolve_input

_GIT_URL = "https://github.com/example/template-provenance.git"
_TRACKED_SCRIPT = ".tracked.sh"
_LOCAL_SCRIPT = ".local.sh"
_TEMPLATE_HOOK = ".git/hooks/post-checkout"
_SCRIPT = b"#!/bin/sh\nprintf '%s\\n' 'fixture'\n"


def _git(*args: str) -> None:
    subprocess.run(["git", *args], check=True, capture_output=True)


def _write_script(path: Path) -> None:
    path.write_bytes(_SCRIPT)
    path.chmod(0o755)


@pytest.fixture
def git_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    if shutil.which("git") is None:
        pytest.skip("Git is required for the clone fixture")
    monkeypatch.chdir(tmp_path)
    for key in os.environ.copy():
        if key.startswith("GIT_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "git.config"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    monkeypatch.setattr("skillspector.input_handler._is_private_ip", lambda _host: False)
    empty_hooks = tmp_path / "empty-hooks"
    empty_hooks.mkdir()
    _git("config", "--global", "core.hooksPath", str(empty_hooks))

    repo = tmp_path / "template-provenance"
    _git("init", "--quiet", "--template=", str(repo))
    _git("-C", str(repo), "config", "uploadpack.allowFilter", "true")
    (repo / "SKILL.md").write_text(
        "---\nname: template-provenance\ndescription: Documents the fixture.\n---\n# Fixture\n",
        encoding="utf-8",
    )
    _write_script(repo / _TRACKED_SCRIPT)
    _git("-C", str(repo), "add", "--", "SKILL.md", _TRACKED_SCRIPT)
    _git(
        "-C",
        str(repo),
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.test",
        "commit",
        "--quiet",
        "--no-gpg-sign",
        "-m",
        "Create fixture",
    )
    # Let Git handle the real clone using only the fixture's file transport.
    _git("config", "--global", f"url.{repo.as_uri()}.insteadOf", _GIT_URL)
    return repo


def _scan_artifacts(target: str) -> tuple[set[str], set[tuple[str, str]], dict[str, str]]:
    result = graph.invoke({"input_path": target, "use_llm": False, "output_format": "json"})
    temp_dir = result.get("temp_dir_for_cleanup")
    try:
        report = json.loads(result["report_body"])
        assert report["execution_successful"] is True
        assert result["raw_file_cache"][_TRACKED_SCRIPT] == _SCRIPT
        return (
            set(result["raw_file_cache"]),
            {(finding["id"], finding["location"]["file"]) for finding in report["issues"]},
            report["skill"]["origin"],
        )
    finally:
        if temp_dir:
            remove_temp_tree(Path(temp_dir))


@pytest.mark.parametrize("template_setting", ["GIT_TEMPLATE_DIR", "init.templateDir"])
def test_git_url_scan_ignores_local_templates(
    git_repository: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, template_setting: str
) -> None:
    clean_snapshot, clean_findings, clean_origin = _scan_artifacts(_GIT_URL)
    assert _TRACKED_SCRIPT in clean_snapshot
    assert ("SC9", _TRACKED_SCRIPT) in clean_findings
    assert _TEMPLATE_HOOK not in clean_snapshot
    assert clean_origin == {"type": "git", "source": _GIT_URL}

    template = tmp_path / "template"
    (template / "hooks").mkdir(parents=True)
    _write_script(template / "hooks" / "post-checkout")
    if template_setting == "GIT_TEMPLATE_DIR":
        monkeypatch.setenv(template_setting, str(template))
    else:
        _git("config", "--global", template_setting, str(template))

    assert not (git_repository / _TEMPLATE_HOOK).exists()
    assert _scan_artifacts(_GIT_URL) == (clean_snapshot, clean_findings, clean_origin)


def test_local_scan_includes_existing_hidden_executable(git_repository: Path) -> None:
    _write_script(git_repository / _LOCAL_SCRIPT)
    hook = git_repository / _TEMPLATE_HOOK
    hook.parent.mkdir()
    _write_script(hook)

    snapshot, findings, origin = _scan_artifacts(str(git_repository))

    assert {_LOCAL_SCRIPT, _TEMPLATE_HOOK} <= snapshot
    assert {("SC9", _LOCAL_SCRIPT), ("SC9", _TEMPLATE_HOOK)} <= findings
    assert origin == {"type": "directory", "source": str(git_repository)}


@pytest.mark.parametrize("failure", ["clone", "budget"])
def test_failed_git_url_scan_removes_owned_checkout(
    git_repository: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(materialized))
    if failure == "clone":
        # The rewrite deliberately addresses a repository that no longer exists.
        git_repository.rename(tmp_path / "moved")
        expected = pytest.raises(ValueError, match="Failed to clone repository")
    else:
        monkeypatch.setattr("skillspector.input_handler.INGEST_MAX_TREE_ENTRIES", 1)
        expected = pytest.raises(TransitiveIngestTruncatedError, match="entry_budget_exhausted")

    with expected:
        resolve_input({"input_path": _GIT_URL})

    assert list(materialized.iterdir()) == []
