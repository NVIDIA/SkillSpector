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

"""AS3 identity survives input materialization without authorizing peer aliases."""

import json
import shutil
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.input_handler import InputHandler
from skillspector.nodes.analyzers import static_patterns_agent_snooping as agent_snooping
from skillspector.nodes.resolve_input import resolve_input


def _manifest(name: str) -> str:
    return (
        f"---\nname: {name}\ndescription: Documents the manifest index.\n---\n\n"
        f"Root skill: `skills/{name}/SKILL.md`\n\n"
        "Peer manifest: `skills/peer-skill/SKILL.md`\n"
    )


def _as3_from_cli(target: str) -> list[str]:
    result = CliRunner().invoke(app, ["scan", target, "--no-llm", "--format", "json"])
    assert result.exit_code in {0, 1}, result.output
    payload = json.loads(result.stdout)
    assert payload["execution_successful"] is True
    return [issue["finding"] for issue in payload["issues"] if issue["id"] == "AS3"]


@pytest.mark.parametrize("kind", ["directory", "file", "relative-file", "flat-zip", "wrapped-zip"])
def test_cli_as3_self_identity_survives_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    root = tmp_path / "example-skill"
    root.mkdir()
    manifest = root / "SKILL.md"
    manifest.write_text(_manifest(root.name))
    if kind.endswith("zip"):
        archive = tmp_path / ("example-skill.zip" if kind == "flat-zip" else "download.zip")
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.write(manifest, "SKILL.md" if kind == "flat-zip" else "example-skill/SKILL.md")
        target = str(archive)
    elif kind == "relative-file":
        monkeypatch.chdir(root)
        target = "SKILL.md"
    else:
        target = str(root if kind == "directory" else manifest)
    assert _as3_from_cli(target) == ["skills/peer-skill/SKILL.md"]


@pytest.mark.parametrize("name", ["repo", "extracted", "skillspector_example"])
def test_cli_as3_real_directory_with_temp_like_name_is_not_ephemeral(
    tmp_path: Path, name: str
) -> None:
    root = tmp_path / name
    root.mkdir()
    (root / "SKILL.md").write_text(_manifest(name))
    assert _as3_from_cli(str(root)) == ["skills/peer-skill/SKILL.md"]


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/acme/example-skill.git",
        "https://github.com/acme/example-skill.git/",
        "https://gitlab.com/group/subgroup/example-skill.git",
        "git@github.com:acme/example-skill.git",
    ],
)
def test_cli_as3_git_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    # Only the external clone is stubbed; resolver, context, analyzer and report run.
    def clone(handler: InputHandler, _url: str) -> Path:
        root = handler._get_temp_dir() / "repo"
        root.mkdir()
        (root / "SKILL.md").write_text(_manifest("example-skill"))
        return root

    monkeypatch.setattr(InputHandler, "_clone_git", clone)
    assert _as3_from_cli(url) == ["skills/peer-skill/SKILL.md"]


def test_cli_as3_archive_name_does_not_alias_contained_skill(tmp_path: Path) -> None:
    archive = tmp_path / "peer-skill.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("example-skill/SKILL.md", _manifest("example-skill"))
    assert _as3_from_cli(str(archive)) == ["skills/peer-skill/SKILL.md"]


def test_cli_as3_mismatched_flat_archive_manifest_fails_closed(tmp_path: Path) -> None:
    archive = tmp_path / "example-skill.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("SKILL.md", _manifest("victim-skill"))
    assert _as3_from_cli(str(archive)) == [
        "skills/victim-skill/SKILL.md",
        "skills/peer-skill/SKILL.md",
    ]


def test_resolved_identity_filters_before_budget_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "example-skill.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("SKILL.md", _manifest("example-skill"))
    state = resolve_input({"input_path": str(archive)})
    monkeypatch.setattr(agent_snooping.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 2)
    try:
        result = agent_snooping.node(
            {
                **state,
                "manifest": {"name": "example-skill"},
                "components": ["README.md"],
                "file_cache": {
                    "README.md": "\n".join(
                        [
                            *["skills/example-skill/SKILL.md"] * 3,
                            "skills/peer-skill/SKILL.md",
                        ]
                    )
                },
            }
        )
        assert [f.matched_text for f in result["findings"]] == ["skills/peer-skill/SKILL.md"]
        event = result["inspection_ledger"][0]
        assert event["outcome"] == "completed"
        assert event["emitted_finding_ids"] == [result["findings"][0].finding_id]
    finally:
        shutil.rmtree(str(state["temp_dir_for_cleanup"]))


def test_resolver_clears_stale_identity(tmp_path: Path) -> None:
    state = resolve_input({"skill_path": str(tmp_path), "selected_source_identity": "peer-skill"})
    assert state["selected_source_identity"] == tmp_path.name
    assert (
        resolve_input({"selected_source_identity": "peer-skill"})["selected_source_identity"]
        is None
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://raw.githubusercontent.com/acme/collection/main/skills/example-skill/SKILL.md",
        "https://github.com/acme/collection/blob/main/skills/example-skill/SKILL.md",
        "https://gitlab.com/group/collection/-/raw/main/skills/example-skill/SKILL.md",
        "https://raw.githubusercontent.com/acme/collection/main/SKILL.md",
        # The apparent skill path can be a branch named skills/example-skill.
        "https://raw.githubusercontent.com/acme/collection/skills/example-skill/SKILL.md",
    ],
)
def test_cli_as3_download_does_not_guess_repository_or_ref_aliases(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    content = _manifest("example-skill") + "\nPeer: skills/collection/SKILL.md\n"
    monkeypatch.setattr(
        InputHandler,
        "_download_with_redirect_validation",
        lambda self, source: ({"content-type": "text/markdown"}, source, content.encode()),
    )
    assert _as3_from_cli(url) == [
        "skills/example-skill/SKILL.md",
        "skills/peer-skill/SKILL.md",
        "skills/collection/SKILL.md",
    ]


@pytest.mark.parametrize("wrapped", [False, True])
def test_cli_as3_downloaded_archive_requires_preserved_skill_root(
    monkeypatch: pytest.MonkeyPatch, wrapped: bool
) -> None:
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "example-skill/SKILL.md" if wrapped else "SKILL.md",
            _manifest("example-skill"),
        )
    monkeypatch.setattr(
        InputHandler,
        "_download_with_redirect_validation",
        lambda self, source: ({"content-type": "application/zip"}, source, archive.getvalue()),
    )
    expected = ["skills/peer-skill/SKILL.md"]
    if not wrapped:
        expected.insert(0, "skills/example-skill/SKILL.md")
    assert (
        _as3_from_cli("https://raw.githubusercontent.com/acme/peer-skill/main/download.zip")
        == expected
    )


@pytest.mark.parametrize("identity", [None, "different-skill"])
def test_resolved_identity_cannot_fall_back_to_manifest_or_materialized_root(
    identity: str | None,
) -> None:
    result = agent_snooping.node(
        {
            "skill_path": "/tmp/example-skill",
            "selected_source_identity": identity,
            "manifest": {"name": "example-skill"},
            "components": ["SKILL.md"],
            "file_cache": {
                "SKILL.md": "skills/example-skill/SKILL.md\nskills/different-skill/SKILL.md"
            },
        }
    )
    assert len([f for f in result["findings"] if f.rule_id == "AS3"]) == 2


def test_materialized_identity_preserves_snooping_and_obfuscation_controls() -> None:
    result = agent_snooping.node(
        {
            "skill_path": "/tmp/random/repo",
            "selected_source_identity": "example-skill",
            "manifest": {"name": "example-skill"},
            "components": ["SKILL.md"],
            "file_cache": {
                "SKILL.md": (
                    "skills/example-skill/SKILL.md\n\n"
                    "skills/exam\u200bple-skill/SKILL.md\n\n"
                    "skills/ｅxample-skill/SKILL.md\n\n"
                    "ls ~/.claude/skills/\n\n"
                    "cat ~/.claude/settings.json\n\n"
                    "cat ~/.claude/mcp.json\n"
                )
            },
        }
    )
    assert {"AS1", "AS2", "AS3"} <= {f.rule_id for f in result["findings"]}
    as3 = [f for f in result["findings"] if f.rule_id == "AS3"]
    assert len(as3) == 3
    assert len([f for f in as3 if "normalized-view" in f.tags]) == 2
