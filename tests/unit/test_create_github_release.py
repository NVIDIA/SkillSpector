# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the public GitHub release helper."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_RELEASE_SCRIPT = REPO_ROOT / "scripts" / "release" / "public" / "create_github_release.py"


def _write_release_notes(root: Path, version: str = "2.4.3") -> Path:
    release_notes = root / "docs" / "release" / f"skillspector-{version}.md"
    release_notes.parent.mkdir(parents=True)
    release_notes.write_text("# SkillSpector release notes\n", encoding="utf-8")
    return release_notes


def _write_existing_release_gh(root: Path) -> tuple[dict[str, str], Path]:
    """Create a fake gh CLI that records mutations for an existing release."""
    bin_dir = root / "bin"
    bin_dir.mkdir()
    calls_file = root / "gh-calls.jsonl"
    gh = bin_dir / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "if args[:1] == ['api']:\n"
        "    endpoint = next((arg.lstrip('/') for arg in args if arg.lstrip('/').startswith('repos/')), '')\n"
        "    if endpoint == 'repos/NVIDIA/SkillSpector/git/ref/tags/v2.4.3':\n"
        "        print(json.dumps({'object': {'type': 'commit', 'sha': 'deadbeef'}}))\n"
        "        raise SystemExit(0)\n"
        "if args[:2] == ['release', 'view']:\n"
        "    print(json.dumps({'isDraft': True}))\n"
        "    raise SystemExit(0)\n"
        "with Path(os.environ['GH_CALLS_FILE']).open('a', encoding='utf-8') as calls:\n"
        "    calls.write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["GH_CALLS_FILE"] = str(calls_file)
    return env, calls_file


def test_dry_run_derives_public_tag_from_project_version(tmp_path: Path) -> None:
    """A dry run reports the exact GitHub release that would be created."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "skillspector"\nversion = "2.4.3"\n',
        encoding="utf-8",
    )
    _write_release_notes(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(PUBLIC_RELEASE_SCRIPT),
            "--repository",
            "NVIDIA/SkillSpector",
            "--target",
            "deadbeef",
            "--dry-run",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "v2.4.3" in result.stdout
    assert "NVIDIA/SkillSpector" in result.stdout
    assert "deadbeef" in result.stdout


def test_creates_github_release_with_supported_distribution_artifacts(tmp_path: Path) -> None:
    """The helper attaches the wheel and sdist to the GitHub release."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "skillspector"\nversion = "2.4.3"\n',
        encoding="utf-8",
    )
    release_notes = _write_release_notes(tmp_path)
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    wheel = dist_dir / "skillspector-2.4.3-py3-none-any.whl"
    wheel.touch()
    source_distribution = dist_dir / "skillspector-2.4.3.tar.gz"
    source_distribution.touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    arguments_file = tmp_path / "gh-arguments.txt"
    tag_lookup_file = tmp_path / "tag-looked-up"
    gh = bin_dir / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "if args[:1] == ['api']:\n"
        "    endpoint = next((arg.lstrip('/') for arg in args if arg.lstrip('/').startswith('repos/')), '')\n"
        "    tag_lookup = Path(os.environ['GH_TAG_LOOKUP_FILE'])\n"
        "    if endpoint == 'repos/NVIDIA/SkillSpector/git/ref/tags/v2.4.3':\n"
        "        if not tag_lookup.exists():\n"
        "            tag_lookup.touch()\n"
        "            print('gh: Not Found (HTTP 404)', file=sys.stderr)\n"
        "            raise SystemExit(1)\n"
        "        print(json.dumps({'object': {'type': 'commit', 'sha': 'deadbeef'}}))\n"
        "        raise SystemExit(0)\n"
        "    if endpoint == 'repos/NVIDIA/SkillSpector/git/refs':\n"
        "        print(json.dumps({'ref': 'refs/tags/v2.4.3'}))\n"
        "        raise SystemExit(0)\n"
        "if args[:2] == ['release', 'view']:\n"
        "    print('release not found', file=sys.stderr)\n"
        "    raise SystemExit(1)\n"
        "Path(os.environ['GH_ARGUMENTS_FILE']).write_text('\\n'.join(sys.argv[1:]))\n"
        "print('https://github.com/NVIDIA/SkillSpector/releases/tag/v2.4.3')\n",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["GH_ARGUMENTS_FILE"] = str(arguments_file)
    env["GH_TAG_LOOKUP_FILE"] = str(tag_lookup_file)

    result = subprocess.run(
        [
            sys.executable,
            str(PUBLIC_RELEASE_SCRIPT),
            "--repository",
            "NVIDIA/SkillSpector",
            "--target",
            "deadbeef",
            "--asset",
            str(wheel),
            "--asset",
            str(source_distribution),
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert arguments_file.read_text(encoding="utf-8").splitlines() == [
        "release",
        "create",
        "v2.4.3",
        "--repo",
        "NVIDIA/SkillSpector",
        "--verify-tag",
        "--title",
        "SkillSpector v2.4.3",
        "--notes-file",
        str(release_notes.relative_to(tmp_path)),
        str(wheel),
        str(source_distribution),
    ]
    assert "https://github.com/NVIDIA/SkillSpector/releases/tag/v2.4.3" in result.stdout


@pytest.mark.parametrize("include_assets", [False, True], ids=["notes-only", "notes-and-assets"])
def test_reconciles_and_publishes_when_rerunning_an_existing_release(
    tmp_path: Path,
    include_assets: bool,
) -> None:
    """A retry restores artifacts and publishes an interrupted draft release."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "skillspector"\nversion = "2.4.3"\n',
        encoding="utf-8",
    )
    _write_release_notes(tmp_path)
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    wheel = dist_dir / "skillspector-2.4.3-py3-none-any.whl"
    wheel.touch()
    source_distribution = dist_dir / "skillspector-2.4.3.tar.gz"
    source_distribution.touch()
    env, calls_file = _write_existing_release_gh(tmp_path)

    asset_arguments = (
        ["--asset", str(wheel), "--asset", str(source_distribution)] if include_assets else []
    )
    result = subprocess.run(
        [
            sys.executable,
            str(PUBLIC_RELEASE_SCRIPT),
            "--repository",
            "NVIDIA/SkillSpector",
            "--target",
            "deadbeef",
            *asset_arguments,
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    expected_calls = []
    if include_assets:
        expected_calls.append(
            [
                "release",
                "upload",
                "v2.4.3",
                "--repo",
                "NVIDIA/SkillSpector",
                "--clobber",
                str(wheel),
                str(source_distribution),
            ]
        )
    expected_calls.append(
        [
            "release",
            "edit",
            "v2.4.3",
            "--repo",
            "NVIDIA/SkillSpector",
            "--notes-file",
            "docs/release/skillspector-2.4.3.md",
            "--draft=false",
        ]
    )
    assert [
        json.loads(line) for line in calls_file.read_text(encoding="utf-8").splitlines()
    ] == expected_calls


def test_rejects_an_existing_version_tag_at_another_commit(tmp_path: Path) -> None:
    """A labeled PR cannot overwrite or release an already-used version tag."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "skillspector"\nversion = "2.4.3"\n',
        encoding="utf-8",
    )
    _write_release_notes(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "print(json.dumps({'object': {'type': 'commit', 'sha': 'other-commit'}}))\n",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"

    result = subprocess.run(
        [
            sys.executable,
            str(PUBLIC_RELEASE_SCRIPT),
            "--repository",
            "NVIDIA/SkillSpector",
            "--target",
            "merged-commit",
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "v2.4.3" in result.stderr
    assert "other-commit" in result.stderr
    assert "merged-commit" in result.stderr


def test_rejects_a_release_when_its_versioned_notes_are_missing(tmp_path: Path) -> None:
    """The release body must come from the matching versioned release note."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "skillspector"\nversion = "2.4.3"\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PUBLIC_RELEASE_SCRIPT),
            "--repository",
            "NVIDIA/SkillSpector",
            "--target",
            "deadbeef",
            "--dry-run",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "docs/release/skillspector-2.4.3.md" in result.stderr
