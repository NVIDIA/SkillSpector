# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""String ownership must not discard evidence from later shell execution."""

from pathlib import Path

import pytest

from skillspector.graph import graph
from skillspector.inspection_ledger import LedgerOutcome
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner
from skillspector.python_tokens import python_string_value_bounds_allowed

_TAIL = "".join(f"padding_{index} = {index}\n" for index in range(700))
_OPTIONS = 'options = ["--recursive", "--force", "/tmp/build"]\n'
_WARNING = 'warnings = [f"Inspected directory source `{source_directory}` and found {detail}."]\n'


def _exhausted(source: str) -> bool:
    return tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="python")


@pytest.mark.parametrize(
    "sink",
    [
        "import os; os.system(command)",
        "import os; os.popen(command)",
        "import os as o; o.system(command)",
        "from os import system as run; run(command)",
        "import os; run = os.system; run(command)",
        "import os; run, = (os.system,); run(command)",
        "import os; o = (os,)[0]; o.system(command)",
        'import os; getattr(os, "system")(command)',
        "import os; getattr(os, selected)(command)",
        '__import__("os").system(command)',
        'import importlib; importlib.import_module("os").system(command)',
        'from importlib import import_module as load; load("os").system(command)',
        'import os; os.execv("/bin/sh", ["sh", "-c", command])',
        'import os; os.spawnv(os.P_WAIT, "/bin/bash", ["bash", "-c", command])',
        "import subprocess; subprocess.run(command, shell=True)",
        "import subprocess as sp; sp.run(command, shell=flag)",
        "from subprocess import run; run(command, **kwargs)",
        "import subprocess; run = subprocess.run; run(command, shell=True)",
        'import subprocess; subprocess.run(["sh", "-c", command], shell=False)',
        'import subprocess; subprocess.run(["/bin/bash", "-lc", command])',
        'import subprocess; subprocess.run(["bash", "-e", "-c", command])',
        'import subprocess; subprocess.run(["bash", "--noprofile", "-c", command])',
        'import subprocess; subprocess.run(["env", "bash", "-e", "-c", command])',
        'import subprocess; subprocess.run(["cmd.exe", "/S", "/C", command])',
        'import subprocess; subprocess.run(["bash.exe", "-c", command])',
        'import subprocess; subprocess.run(["pwsh.exe", "-c", command])',
        'import subprocess; subprocess.run([r"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe", "-c", command])',
        'import subprocess; subprocess.run(["bash"] + ["-c", command])',
        "import os; os.path.os.system(command)",
        "import asyncio; asyncio.get_event_loop().subprocess_shell(None, command)",
        'import subprocess; subprocess.run(["env", "sh", "-c", command])',
        'import subprocess; subprocess.run(command, executable="/bin/sh")',
        "import subprocess; subprocess.Popen(command, -1, None, None, None, None, None, True, True)",
        "import asyncio; asyncio.create_subprocess_shell(command)",
        "from asyncio import create_subprocess_shell as run; run(command)",
        'import os; message = f"status: {os.system(command)}"',
        "evaluate = eval; evaluate(command)",
        "from builtins import exec as execute; execute(command)",
        "from helper import *; run(command)",
    ],
)
def test_recognized_or_ambiguous_shell_paths_disable_local_string_value_bounds(sink: str) -> None:
    source = 'command = f"Report `{tool}`"\ncommand += " -rf /"\n' + sink + "\n"
    assert not python_string_value_bounds_allowed(source, lambda: None)
    assert _exhausted(source)


@pytest.mark.parametrize(
    "call",
    [
        "import subprocess; subprocess.run(argv)",
        "import subprocess as sp; sp.run(argv, shell=False)",
        "from subprocess import run as invoke; invoke(argv, check=True)",
        'import subprocess; subprocess.run(["python", "worker.py", value])',
        'import os; print(os.environ.get("WORK_ROOT"), os.path.join(root, leaf))',
        "from os import path as p; print(p.join(root, leaf))",
        "import os.path as p; print(p.join(root, leaf))",
        "import posixpath as p; print(p.join(root, leaf))",
    ],
)
def test_default_or_explicit_nonshell_subprocess_keeps_report_ownership(call: str) -> None:
    source = _WARNING + _OPTIONS + call + "\n" + _TAIL
    assert python_string_value_bounds_allowed(source, lambda: None)
    assert not _exhausted(source)


@pytest.mark.parametrize(
    "sink",
    [
        "from os.path import os as x; x.system(command)",
        "import posix as x; x.system(command)",
        "import nt as x; x.system(command)",
        "from posix import system as run; run(command)",
        "__builtins__.eval(command)",
        '__builtins__["eval"](command)',
        'import importlib; importlib.sys.modules["subprocess"].run(command, shell=True)',
        'import builtins; builtins.getattr(builtins.__import__("os"), "system")(command)',
        'import os; os.getenv.__globals__["system"](command)',
        'from os import getenv as g; g.__globals__["system"](command)',
        'import os; os.environ.__class__.__init__.__globals__["system"](command)',
        "runtime.eval(command)",
        "runtime.exec(command)",
        "from os import path as p; p.os.system(command)",
        'import os; os.path.os.__getattribute__("system")(command)',
        'from os import path as p; p.os.__getattribute__("system")(command)',
        'import os; import subprocess; os.path.sys.modules["subprocess"].run(command, shell=True)',
        'import os; import subprocess; os.path.sys.modules["subprocess"].run(command, **opts)',
        "import os; path = os.path; path.os.system(command)",
        'import posixpath; posixpath.os.__getattribute__("system")(command)',
        'import subprocess; argv = ["bash", "-c", command]; subprocess.run(argv)',
        'from subprocess import run; argv = ["bash", "-c", command]; run(argv)',
    ],
)
def test_indirect_shell_paths_keep_report_shaped_owner_fallback(sink: str) -> None:
    source = 'command = f"Report `{tool}`"\ncommand += " -rf /"\n' + sink + "\n"
    assert not python_string_value_bounds_allowed(source, lambda: None)
    assert _exhausted(source)


@pytest.mark.parametrize(
    "transform",
    [
        'command = command.removeprefix("Report ")',
        'command = command[len("Report "):]',
        'command = command.replace("Report ", "")',
    ],
)
def test_later_transformed_report_string_keeps_incomplete_ledger(transform: str) -> None:
    source = (
        'import subprocess\ntool = "echo rm"\ncommand = f"Report `{tool}`"\n'
        + transform
        + '\ncommand += " -rf /"\n'
        + 'subprocess.run(["bash", "--noprofile", "-c", command])\n'
    )
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["command.py"], "file_cache": {"command.py": source}}, [tm]
    )
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL


def test_referenced_shell_flow_is_not_reported_fully_inspected(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text(
        "---\nname: synthetic-command\ndescription: Synthetic command inspection fixture.\n---\n"
        "Read `command.py`.\n",
        encoding="utf-8",
    )
    (tmp_path / "command.py").write_text(
        'import subprocess\ntool = "echo rm"\ncommand = f"Report `{tool}`"\n'
        'command = command.removeprefix("Report ")\ncommand += " -rf /"\n'
        'subprocess.run(["bash", "--noprofile", "-c", command])\n',
        encoding="utf-8",
    )
    result = graph.invoke({"input_path": str(tmp_path), "use_llm": False, "output_format": "json"})
    assert result["analysis_completeness"]["status"] == "partial"
    assert any(f.rule_id == "AE1" and str(f.severity) == "HIGH" for f in result["findings"])


def test_string_value_guard_preserves_runtime_interruptions() -> None:
    def interrupted() -> None:
        raise TimeoutError("synthetic inspection deadline")

    with pytest.raises(TimeoutError, match="inspection deadline"):
        python_string_value_bounds_allowed("value = 'report'", interrupted)


def test_string_value_guard_caches_only_completed_proofs(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _WARNING + _OPTIONS + _TAIL
    owner = tm._PythonSourceOwnership(source, lambda: None)
    calls = 0

    def guard(_source: str, _check_runtime: object) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("synthetic guard deadline")
        return True

    monkeypatch.setattr(tm, "python_string_value_bounds_allowed", guard)
    start = source.index("`")
    with pytest.raises(TimeoutError, match="guard deadline"):
        owner.string_value(start, start + 1)
    assert owner.string_value(start, start + 1) is not None
    assert owner.string_value(start, start + 1) is not None
    assert calls == 2


@pytest.mark.parametrize(
    "sink",
    [
        'import sys, subprocess; sys.modules["subprocess"].run(command, shell=True)',
        'import sys as runtime, subprocess; runtime.modules["subprocess"].run(["bash", "-c", command])',
        'from sys import modules as loaded; loaded["subprocess"].run(command, shell=True)',
    ],
)
def test_module_registry_reflection_keeps_report_string_fallback(sink):
    source = 'tool = "echo rm"\ncommand = f"Report `{tool}`"\ncommand += " -rf /"\n' + sink + "\n"
    assert not python_string_value_bounds_allowed(source, lambda: None)
    assert _exhausted(source)
