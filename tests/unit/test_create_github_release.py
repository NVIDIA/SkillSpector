# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the public GitHub release helper."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_RELEASE_SCRIPT = REPO_ROOT / "scripts" / "release" / "public" / "create_github_release.py"


def test_dry_run_derives_public_tag_from_project_version(tmp_path: Path) -> None:
    """A dry run reports the exact GitHub release that would be created."""
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

    assert result.returncode == 0, result.stderr
    assert "v2.4.3" in result.stdout
    assert "NVIDIA/SkillSpector" in result.stdout
    assert "deadbeef" in result.stdout


def test_creates_github_release_at_requested_commit(tmp_path: Path) -> None:
    """The helper creates and verifies the version tag before the release."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "skillspector"\nversion = "2.4.3"\n',
        encoding="utf-8",
    )
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
        "    if endpoint == 'repos/NVIDIA/SkillSpector/releases/tags/v2.4.3':\n"
        "        print('gh: Not Found (HTTP 404)', file=sys.stderr)\n"
        "        raise SystemExit(1)\n"
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
        "--generate-notes",
    ]
    assert "https://github.com/NVIDIA/SkillSpector/releases/tag/v2.4.3" in result.stdout


def test_rejects_an_existing_version_tag_at_another_commit(tmp_path: Path) -> None:
    """A labeled PR cannot overwrite or release an already-used version tag."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "skillspector"\nversion = "2.4.3"\n',
        encoding="utf-8",
    )
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
