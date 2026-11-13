# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup import boundaries and parser compatibility for the console entry point."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner


def _entry_point() -> tuple[str, str]:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return tuple(project["project"]["scripts"]["skillspector"].split(":"))


@pytest.mark.parametrize("flag", ["--version", "-v"])
@pytest.mark.parametrize("provider", ["nv_build", "anthropic", "invalid-provider"])
def test_version_does_not_import_scan_stack(flag: str, provider: str) -> None:
    module, name = _entry_point()
    code = f"""
import importlib
import importlib.abc
import sys

class RejectScanStack(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = (
            "skillspector.cli", "skillspector.constants", "skillspector.providers",
            "skillspector.nodes", "skillspector.graph", "langgraph",
            "langchain_core.runnables", "langchain_core.language_models",
            "langchain_openai", "langchain_anthropic", "langchain_aws",
            "openai", "anthropic", "boto3",
        )
        if any(fullname == p or fullname.startswith(p + ".") for p in blocked):
            raise AssertionError("Version loaded scanning dependency: " + fullname)

sys.meta_path.insert(0, RejectScanStack())
app = getattr(importlib.import_module({module!r}), {name!r})
app()
"""
    env = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP")
        if key in os.environ
    }
    env.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"),
        SKILLSPECTOR_PROVIDER=provider,
        SKILLSPECTOR_MODEL="not-a-registered-model",
        SKILLSPECTOR_STRICT_MODEL_VALIDATION="true",
        NO_COLOR="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", code, flag],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    from skillspector import __version__

    assert result.stdout == f"SkillSpector v{__version__}\n"
    assert result.stderr == ""


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--version"],
        ["-v"],
        ["-vv"],
        ["--version", "scan", "--unknown"],
        ["--version", "--help"],
        ["--help", "--version"],
        ["--unknown", "--version"],
        ["--version=false"],
        ["--", "--version"],
        ["scan", "--version"],
        ["scna"],
        ["scan", "--help"],
        ["baseline", "--help"],
        ["mcp", "--help"],
    ],
)
def test_entry_point_preserves_cli_parsing(args: list[str], monkeypatch) -> None:
    from importlib import import_module

    from skillspector.cli import app as original_app

    monkeypatch.setenv("NO_COLOR", "1")
    module, name = _entry_point()
    entry_app = getattr(import_module(module), name)
    runner = CliRunner()
    original = runner.invoke(original_app, args, prog_name="skillspector")
    entry = runner.invoke(entry_app, args, prog_name="skillspector")

    assert entry.exit_code == original.exit_code
    assert entry.stdout == original.stdout
    assert entry.stderr == original.stderr
