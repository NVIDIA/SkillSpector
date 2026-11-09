# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup import boundaries and parser compatibility for the console entry point."""

from __future__ import annotations

import json
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


class _WindowsOs:
    """Report ``os.name == "nt"`` so Typer applies its Windows argv expansion on any platform."""

    name = "nt"

    def __getattr__(self, attribute: str) -> object:
        return getattr(os, attribute)


@pytest.mark.parametrize(
    ("module", "name"),
    [_entry_point(), ("skillspector.cli", "app")],
    ids=["console-script", "python-m"],
)
def test_entry_point_keeps_exclude_globs_verbatim(
    module: str, name: str, tmp_path: Path, monkeypatch, capsys
) -> None:
    from importlib import import_module

    (tmp_path / "SKILL.md").write_text("# Demo\nA harmless example skill.\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "a.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "tests" / "b.py").write_text("print(2)\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("typer.core.os", _WindowsOs())
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)  # app() replaces it
    monkeypatch.setattr(
        sys,
        "argv",
        ["skillspector", "scan", ".", "--no-llm", "--format", "json", "--exclude", "tests/*"],
    )

    with pytest.raises(SystemExit) as exited:
        getattr(import_module(module), name)()

    captured = capsys.readouterr()
    assert exited.value.code == 0, captured.err
    report = json.loads(captured.out)
    assert report["analysis_completeness"]["exclude_patterns"] == ["tests/*"]
    assert [component["path"] for component in report["components"]] == ["SKILL.md"]
