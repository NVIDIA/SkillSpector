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


@pytest.mark.parametrize(
    ("args", "expected_exit", "expected_text"),
    [
        (["--help"], 0, "Security scanner"),
        ([], 2, "Security scanner"),
        (["scan", "--help"], 0, "--mcp-registry"),
        (["baseline", "--help"], 0, "--reason"),
        (["mcp", "--help"], 0, "--transport"),
        (["scna"], 2, "No such command"),
        (["scan", "missing", "--format", "invalid"], 2, "Invalid value"),
    ],
)
def test_help_and_validation_do_not_import_scanner(
    args: list[str], expected_exit: int, expected_text: str
) -> None:
    result = _run_without_scanner(args)
    assert result.returncode == expected_exit, result.stderr
    assert expected_text in result.stdout + result.stderr


def _run_without_scanner(
    args: list[str],
    *,
    extra_env: dict[str, str] | None = None,
    entry_module: str = "skillspector.cli_entry",
) -> subprocess.CompletedProcess[str]:
    # A real process catches accidental transitive imports that an in-process
    # test would miss after another test has populated sys.modules.
    code = """
import importlib.abc
import sys
class RejectScanner(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ("skillspector.constants", "skillspector.providers", "skillspector.nodes",
                   "skillspector.graph", "skillspector.llm_utils", "langgraph",
                   "langchain_core.runnables", "langchain_core.language_models",
                   "langchain_openai", "langchain_anthropic", "langchain_aws",
                   "openai", "anthropic", "boto3")
        if any(fullname == p or fullname.startswith(p + ".") for p in blocked):
            raise AssertionError("Unexpected scanner import: " + fullname)
sys.meta_path.insert(0, RejectScanner())
from skillspector.cli_entry import app
app(prog_name="skillspector")
"""
    code = code.replace("from skillspector.cli_entry import app", f"from {entry_module} import app")
    env = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP")
        if key in os.environ
    }
    env.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"),
        SKILLSPECTOR_PROVIDER="invalid-provider",
        SKILLSPECTOR_MODEL="not-a-registered-model",
        SKILLSPECTOR_STRICT_MODEL_VALIDATION="true",
        NO_COLOR="1",
    )
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("entry_module", ["skillspector.cli_entry", "skillspector.cli"])
@pytest.mark.parametrize("additional_report", [False, True])
def test_registry_scan_does_not_import_scanner(
    tmp_path: Path, entry_module: str, additional_report: bool
) -> None:
    payload = tmp_path / "registry.json"
    payload.write_text(
        '{"servers": [{"server": {"name": "example/demo", "version": "1.0.0"}}]}', encoding="utf-8"
    )
    report = tmp_path / "report.json"
    additional = tmp_path / "additional.json"
    options = ["--report", f"json={additional}"] if additional_report else []
    result = _run_without_scanner(
        ["scan", str(payload), "--mcp-registry", "-f", "JSON", "-o", str(report), *options],
        entry_module=entry_module,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text())
    assert data["mcp_registry"] is True
    assert len(data["servers"]) == 1
    assert data["servers"][0]["snapshot"]["name"] == "example/demo"
    if additional_report:
        assert json.loads(additional.read_text()) == data
        assert "Report saved to" in result.stderr
    else:
        assert result.stderr == ""


@pytest.mark.parametrize(
    "options",
    [
        ["--recursive"],
        ["--baseline", "missing.yaml"],
        ["--show-suppressed"],
        ["--yara-rules-dir", "missing"],
        ["--min-coverage", "50"],
        ["--use-shipped-baseline"],
        ["--inspect-dependencies"],
        ["--transitive"],
        ["--transitive-depth", "1"],
        ["--transitive-allow-prefix", "https://example.com/"],
        ["--transitive-deny-prefix", "https://example.com/"],
        ["--exclude", "*.py"],
    ],
)
def test_registry_option_errors_stay_lightweight(options: list[str]) -> None:
    result = _run_without_scanner(
        ["scan", "missing.json", "--mcp-registry", "-f", "json", *options]
    )
    assert result.returncode == 2, result.stderr
    assert "Error:" in result.stderr
    assert "Unexpected scanner import" not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("words", ["skillspector sc", "skillspector scan --trans"])
def test_shell_completion_keeps_scanner_unloaded(words: str) -> None:
    result = _run_without_scanner(
        [],
        extra_env={
            "_SKILLSPECTOR_COMPLETE": "complete_bash",
            "COMP_WORDS": words,
            "COMP_CWORD": str(len(words.split()) - 1),
        },
    )
    assert result.returncode == 0, result.stderr
    expected = "scan" if words.endswith(" sc") else "--transitive"
    assert expected in result.stdout
    assert result.stderr == ""


def test_registry_compare_and_stdout_keep_scanner_unloaded(tmp_path: Path) -> None:
    payload = tmp_path / "registry.json"
    payload.write_text(
        '{"servers": [{"server": {"name": "example/demo", "version": "1.0.0"}}]}',
        encoding="utf-8",
    )
    previous = tmp_path / "previous.json"
    initial = _run_without_scanner(["scan", str(payload), "--mcp-registry", "-f", "json"])
    assert initial.returncode == 0, initial.stderr
    previous.write_text(initial.stdout, encoding="utf-8")
    result = _run_without_scanner(
        [
            "scan",
            str(payload),
            "--mcp-registry",
            "-f",
            "json",
            "--mcp-registry-compare",
            str(previous),
        ]
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["server_count"] == 1
    assert "comparison" in report
    assert result.stderr == ""
