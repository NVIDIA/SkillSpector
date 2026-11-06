# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the publisher's committed-source boundary without a registry."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/hotfix-image.yml"


def test_manual_publisher_keeps_checkout_credentials_out_of_the_build() -> None:
    assert WORKFLOW.exists(), "Missing manual hotfix publisher"
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"workflow_dispatch"}
    job = workflow["jobs"]["publish"]
    assert job["if"] == "github.repository == 'NVIDIA/SkillSpector'"
    assert workflow["permissions"] == {"contents": "read", "packages": "write"}
    actions = [step for step in job["steps"] if "uses" in step]
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"]) for step in actions)
    checkout = next(step for step in actions if step["uses"].startswith("actions/checkout@"))
    assert checkout["with"]["persist-credentials"] == "false"
    upload = next(step for step in actions if step["uses"].startswith("actions/upload-artifact@"))
    assert upload["with"]["path"] == "hotfix.env"


def test_build_context_and_wheel_exclude_unrelated_and_uncommitted_files(tmp_path: Path) -> None:
    assert WORKFLOW.exists(), "Missing manual hotfix publisher"
    dockerfile = ROOT / "Dockerfile.hotfix"
    assert dockerfile.exists(), "Missing package transport Dockerfile"
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    build = next(step for step in workflow["jobs"]["publish"]["steps"] if step.get("id") == "build")

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for name in ("Dockerfile.hotfix", "pyproject.toml", "README.md", "LICENSE"):
        shutil.copyfile(ROOT / name, checkout / name)
    shutil.copytree(
        ROOT / "src/skillspector",
        checkout / "src/skillspector",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    fixture = checkout / "tests/customer-fixture.txt"
    fixture.parent.mkdir()
    fixture.write_text("unrelated-committed-fixture\n")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "Public package fixture",
        ],
        cwd=checkout,
        check=True,
    )
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    (checkout / "README.md").write_text("uncommitted-readme-marker\n")
    (checkout / "src/skillspector/untracked-fixture.txt").write_text("untracked-fixture-marker\n")
    (checkout / ".env").write_text("SYNTHETIC_VALUE=not-for-the-build\n")

    # Docker is the external boundary; execute the real context preparation,
    # then capture its actual input instead of contacting a daemon or registry.
    commands = tmp_path / "bin"
    commands.mkdir()
    docker = commands / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import os, shutil, sys\n"
        "assert sys.argv[1] == 'build'\n"
        "shutil.copytree(sys.argv[-1], os.environ['CAPTURED_CONTEXT'])\n"
    )
    docker.chmod(0o755)
    context = tmp_path / "context"
    subprocess.run(
        ["bash", "-euo", "pipefail", "-c", build["run"]],
        cwd=checkout,
        env={
            **os.environ,
            "PATH": f"{commands}{os.pathsep}{os.environ['PATH']}",
            "SOURCE_SHA": sha,
            "IMAGE": "ghcr.io/nvidia/skillspector-hotfix",
            "CAPTURED_CONTEXT": str(context),
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert {path.name for path in context.iterdir()} == {
        "Dockerfile.hotfix",
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "src",
    }
    assert {path.name for path in (context / "src").iterdir()} == {"skillspector"}
    assert (context / "README.md").read_bytes() == (ROOT / "README.md").read_bytes()
    assert not (context / "src/skillspector/untracked-fixture.txt").exists()

    wheels = tmp_path / "wheels"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--wheel",
            "--outdir",
            str(wheels),
            str(context),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    built = list(wheels.glob("*.whl"))
    assert len(built) == 1
    with zipfile.ZipFile(built[0]) as wheel:
        assert "skillspector/__init__.py" in wheel.namelist()
        licenses = [
            name for name in wheel.namelist() if name.endswith(".dist-info/licenses/LICENSE")
        ]
        assert len(licenses) == 1
        assert wheel.read(licenses[0]) == (ROOT / "LICENSE").read_bytes()
        assert all(
            name.startswith("skillspector/") or re.match(r"skillspector-[^/]+\.dist-info/", name)
            for name in wheel.namelist()
        )
        for name in wheel.namelist():
            data = wheel.read(name)
            assert b"unrelated-committed-fixture" not in data
            assert b"uncommitted-readme-marker" not in data
            assert b"untracked-fixture-marker" not in data
