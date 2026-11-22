# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for NVIDIA/SkillSpector#1166.

Windows Application Control (WDAC) blocks the unsigned ``ormsgpack`` .pyd
at import, so every scan fails, and the recursive runner hid the real error
behind a generic message. These tests simulate the blocked-import error path
(the sandbox has no WDAC-managed Windows endpoint, so the exact OS wording
is simulated, not observed live).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from skillspector.cli import FormatChoice, _scan_multi_skill, app
from skillspector.graph_proxy import (
    GraphLoadError,
    LazyGraph,
    blocked_native_import,
    format_blocked_native_import_diagnostic,
    translate_graph_load_error,
)
from skillspector.multi_skill import MultiSkillDetectionResult, SkillDirectory

runner = CliRunner()


def _wdac_import_error(module: str, *, policy_blocked: bool = True) -> ImportError:
    """Simulate a native extension import refused by the OS at load time."""
    if policy_blocked:
        text = (
            f"DLL load failed while importing {module}: the module was "
            "blocked by Windows Application Control."
        )
    else:
        text = f"DLL load failed while importing {module}: the specified module could not be found."
    return ImportError(text, name=module, path=f"{module}.pyd")


def test_blocked_native_import_detects_policy_blocked_dll() -> None:
    """A WDAC-blocked .pyd load is recognised, with the module named."""
    info = blocked_native_import(_wdac_import_error("ormsgpack"))
    assert info is not None
    assert info.module == "ormsgpack"
    assert info.policy_blocked is True


def test_blocked_native_import_detects_plain_dll_load_failure() -> None:
    """A plain DLL load failure is recognised without claiming WDAC."""
    info = blocked_native_import(_wdac_import_error("yara", policy_blocked=False))
    assert info is not None
    assert info.module == "yara"
    assert info.policy_blocked is False


def test_blocked_native_import_ignores_missing_module() -> None:
    """A module that is simply not installed is a different problem."""
    assert (
        blocked_native_import(ImportError("No module named 'ormsgpack'", name="ormsgpack")) is None
    )


def test_blocked_native_import_ignores_unrelated_errors() -> None:
    """Non-import failures are never reinterpreted as blocked native modules."""
    assert blocked_native_import(RuntimeError("boom")) is None
    assert blocked_native_import(ValueError("bad value")) is None
    assert blocked_native_import(ImportError("cannot import name 'x' from 'y'")) is None


def test_translate_graph_load_error_wraps_blocked_import() -> None:
    """A blocked import becomes a GraphLoadError carrying the diagnostic."""
    translated = translate_graph_load_error(_wdac_import_error("ormsgpack"))
    assert isinstance(translated, GraphLoadError)
    assert isinstance(translated, ImportError)
    assert translated.module == "ormsgpack"
    message = str(translated)
    assert "ormsgpack" in message
    assert "Windows Application Control" in message
    assert "WSL" in message
    assert "Docker" in message


def test_translate_graph_load_error_leaves_other_errors_alone() -> None:
    """Errors that are not blocked native imports pass through untouched."""
    error = RuntimeError("boom")
    assert translate_graph_load_error(error) is error
    missing = ImportError("No module named 'ormsgpack'", name="ormsgpack")
    assert translate_graph_load_error(missing) is missing


def test_format_diagnostic_does_not_echo_raw_error_text() -> None:
    """The diagnostic is constructed, never a copy of the raw exception text."""
    info = blocked_native_import(_wdac_import_error("ormsgpack"))
    assert info is not None
    message = format_blocked_native_import_diagnostic(info)
    assert "ormsgpack" in message
    assert "DLL load failed while importing" not in message


def test_lazy_graph_import_failure_raises_graph_load_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blocked import while compiling the graph surfaces as GraphLoadError."""
    failing = types.ModuleType("skillspector.graph")

    def _blocked_import(_name: str) -> object:
        raise _wdac_import_error("ormsgpack")

    failing.__getattr__ = _blocked_import  # type: ignore[method-assign]
    monkeypatch.setitem(sys.modules, "skillspector.graph", failing)

    with pytest.raises(GraphLoadError) as exc_info:
        LazyGraph().invoke({})

    assert "ormsgpack" in str(exc_info.value)
    assert "Windows Application Control" in str(exc_info.value)


@pytest.mark.parametrize("module", ["ormsgpack", "yara"])
def test_recursive_child_failure_surfaces_blocked_native_error(tmp_path: Path, module: str) -> None:
    """First blocked child failure names the module and stops the remaining children."""
    skills = [
        SkillDirectory(path=tmp_path / name, name=name, relative_path=name)
        for name in ("one", "two")
    ]
    detection = MultiSkillDetectionResult(is_multi_skill=True, skills=skills)
    output = tmp_path / "combined.json"

    with patch(
        "skillspector.cli.graph.invoke",
        side_effect=_wdac_import_error(module),
    ):
        with pytest.raises(typer.Exit) as exit_info:
            _scan_multi_skill(
                detection,
                FormatChoice.json,
                output,
                no_llm=True,
                yara_rules_dir=None,
                verbose=False,
            )

    assert exit_info.value.exit_code == 2
    payload = json.loads(output.read_text())
    assert payload["execution_successful"] is False
    # Fail fast: the second child is never attempted once the graph cannot load.
    attempted = [skill["name"] for skill in payload["skills"] if "name" in skill]
    assert attempted == ["one"]
    omitted = [skill for skill in payload["skills"] if skill.get("omitted")]
    assert omitted and omitted[0]["omitted_count"] == 1
    error = payload["skills"][0]["error"]
    assert error.startswith("A recursive child scan failed before complete inspection.")
    assert module in error
    assert "WSL" in error
    limitations = payload["analysis_completeness"]["limitations"]
    assert any(module in item for item in limitations)


def test_recursive_child_failure_keeps_generic_message_for_other_errors(
    tmp_path: Path,
) -> None:
    """Non-import child crashes still get exactly the generic message."""
    skills = [SkillDirectory(path=tmp_path / "one", name="one", relative_path="one")]
    detection = MultiSkillDetectionResult(is_multi_skill=True, skills=skills)
    output = tmp_path / "combined.json"

    with patch(
        "skillspector.cli.graph.invoke",
        side_effect=RuntimeError("TOKEN=child-scan-secret"),
    ):
        with pytest.raises(typer.Exit):
            _scan_multi_skill(
                detection,
                FormatChoice.json,
                output,
                no_llm=True,
                yara_rules_dir=None,
                verbose=False,
            )

    payload = json.loads(output.read_text())
    assert payload["skills"][0]["error"] == (
        "A recursive child scan failed before complete inspection."
    )
    assert "TOKEN=child-scan-secret" not in output.read_text()


def test_single_skill_scan_fails_fast_with_wdac_diagnostic(tmp_path: Path) -> None:
    """A blocked graph import fails the scan with the diagnostic on stderr."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Skill\n", encoding="utf-8")

    with patch(
        "skillspector.cli.graph.invoke",
        side_effect=_wdac_import_error("ormsgpack"),
    ):
        result = runner.invoke(app, ["scan", str(skill_dir), "--no-llm"])

    assert result.exit_code == 2
    # Rich wraps the stderr line, so compare against the unwrapped text.
    stderr = " ".join(result.stderr.split())
    assert "ormsgpack" in stderr
    assert "Windows Application Control" in stderr
    assert "WSL" in stderr
    assert "DLL load failed while importing" not in stderr
