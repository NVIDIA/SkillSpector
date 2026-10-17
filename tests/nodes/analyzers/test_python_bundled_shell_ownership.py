# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundled imports cannot erase existing shell inspection uncertainty."""

import pytest

from skillspector import python_import_bounds
from skillspector.inspection_ledger import LedgerOutcome
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner
from skillspector.python_import_bounds import PythonImportBounds

_CASES = [
    (
        "selected_output_beside_unrelated_shell",
        {
            "caller.py": "from helpers import output\n"
            'tool = "echo rm"\n'
            'command = f"Report `{tool}`"\n'
            'command = command.removeprefix("Report ")\n'
            'command += " -rf /"\n'
            "output(command)\n",
            "helpers.py": "import subprocess\n"
            "def output(value):\n"
            "    print(value)\n"
            "def unrelated(value):\n"
            '    subprocess.run(["bash", "-c", value])\n',
        },
        "caller.py",
        True,
    ),
    (
        "module_shell_initialization",
        {
            "caller.py": "from helpers import output\n"
            'tool = "echo rm"\n'
            'command = f"Report `{tool}`"\n'
            'command = command.removeprefix("Report ")\n'
            'command += " -rf /"\n'
            "output(command)\n",
            "helpers.py": "import subprocess\n"
            'subprocess.run(["bash", "-c", "printf init"])\n'
            "def output(value):\n"
            "    print(value)\n",
        },
        "caller.py",
        False,
    ),
    (
        "module_reflection_initialization",
        {
            "caller.py": "from helpers import output\n"
            'tool = "echo rm"\n'
            'command = f"Report `{tool}`"\n'
            'command = command.removeprefix("Report ")\n'
            'command += " -rf /"\n'
            "output(command)\n",
            "helpers.py": "import sys, subprocess\n"
            'runner = sys.modules["subprocess"].run\n'
            "def output(value):\n"
            "    print(value)\n",
        },
        "caller.py",
        False,
    ),
    (
        "transitive_shell_helper",
        {
            "caller.py": "from helpers import output\n"
            'tool = "echo rm"\n'
            'command = f"Report `{tool}`"\n'
            'command = command.removeprefix("Report ")\n'
            'command += " -rf /"\n'
            "output(command)\n",
            "helpers.py": "from engine import execute\ndef output(value):\n    execute(value)\n",
            "engine.py": "import subprocess\n"
            "def execute(value):\n"
            '    subprocess.run(["bash", "-c", value])\n',
        },
        "caller.py",
        False,
    ),
    (
        "module_alias_shell_helper",
        {
            "caller.py": "import helpers as h\n"
            'tool = "echo rm"\n'
            'command = f"Report `{tool}`"\n'
            'command = command.removeprefix("Report ")\n'
            'command += " -rf /"\n'
            "h.execute(command)\n",
            "helpers.py": "import subprocess\n"
            "def execute(value):\n"
            '    subprocess.run(["bash", "-c", value])\n',
        },
        "caller.py",
        False,
    ),
    (
        "from_alias_shell_helper",
        {
            "caller.py": "from helpers import execute as h\n"
            'tool = "echo rm"\n'
            'command = f"Report `{tool}`"\n'
            'command = command.removeprefix("Report ")\n'
            'command += " -rf /"\n'
            "h(command)\n",
            "helpers.py": "import subprocess\n"
            "def execute(value):\n"
            '    subprocess.run(["bash", "-c", value])\n',
        },
        "caller.py",
        False,
    ),
    (
        "relative_shell_helper",
        {
            "pkg/__init__.py": "",
            "pkg/caller.py": "from .helpers import execute\n"
            'tool = "echo rm"\n'
            'command = f"Report `{tool}`"\n'
            'command = command.removeprefix("Report ")\n'
            'command += " -rf /"\n'
            "execute(command)\n",
            "pkg/helpers.py": "import subprocess\n"
            "def execute(value):\n"
            '    subprocess.run(["bash", "-c", value])\n',
        },
        "pkg/caller.py",
        False,
    ),
    (
        "missing_local_member",
        {
            "caller.py": "from helpers import missing\n"
            'tool = "echo rm"\n'
            'command = f"Report `{tool}`"\n'
            'command = command.removeprefix("Report ")\n'
            'command += " -rf /"\n'
            "missing(command)\n",
            "helpers.py": "def output(value):\n    print(value)\n",
        },
        "caller.py",
        False,
    ),
    (
        "ambiguous_local_module",
        {
            "caller.py": "from helpers import output\n"
            'tool = "echo rm"\n'
            'command = f"Report `{tool}`"\n'
            'command = command.removeprefix("Report ")\n'
            'command += " -rf /"\n'
            "output(command)\n",
            "helpers.py": "def output(value):\n    print(value)\n",
            "helpers/__init__.py": "import subprocess\n"
            "def output(value):\n"
            '    subprocess.run(["bash", "-c", value])\n',
        },
        "caller.py",
        False,
    ),
    (
        "reachable_import_cycle",
        {
            "caller.py": "from helpers import output\n"
            'tool = "echo rm"\n'
            'command = f"Report `{tool}`"\n'
            'command = command.removeprefix("Report ")\n'
            'command += " -rf /"\n'
            "output(command)\n",
            "helpers.py": "from engine import output as other\ndef output(value):\n    other(value)\n",
            "engine.py": "from helpers import output as other\ndef output(value):\n    other(value)\n",
        },
        "caller.py",
        False,
    ),
    (
        "caller_argv_generic_helper",
        {
            "caller.py": "from helpers import execute\n"
            'tool = "echo rm"\n'
            'command = f"Report `{tool}`"\n'
            'command = command.removeprefix("Report ")\n'
            'command += " -rf /"\n'
            'argv = ["bash", "--noprofile", "-c", command]\n'
            "execute(argv)\n",
            "helpers.py": "import subprocess\ndef execute(argv):\n    subprocess.run(argv)\n",
        },
        "caller.py",
        False,
    ),
    (
        "unsupported_callable_class",
        {
            "caller.py": "from helpers import Runner\n"
            'tool = "echo rm"\n'
            'command = f"Report `{tool}`"\n'
            'command = command.removeprefix("Report ")\n'
            'command += " -rf /"\n'
            "run = Runner()\n"
            "run(command)\n",
            "helpers.py": "import subprocess\n"
            "class Runner:\n"
            "    def __call__(self, value):\n"
            '        subprocess.run(["bash", "-c", value])\n',
        },
        "caller.py",
        False,
    ),
    (
        "unsupported_lambda",
        {
            "caller.py": "from helpers import execute\n"
            'tool = "echo rm"\n'
            'command = f"Report `{tool}`"\n'
            'command = command.removeprefix("Report ")\n'
            'command += " -rf /"\n'
            "execute(command)\n",
            "helpers.py": "import subprocess\n"
            'execute = lambda value: subprocess.run(["bash", "-c", value])\n',
        },
        "caller.py",
        False,
    ),
    (
        "function_rebound_to_shell",
        {
            "caller.py": "from helpers import output\n"
            'tool = "echo rm"\n'
            'command = f"Report `{tool}`"\n'
            'command = command.removeprefix("Report ")\n'
            'command += " -rf /"\n'
            "output(command)\n",
            "helpers.py": "import subprocess\n"
            "def output(value):\n"
            "    print(value)\n"
            'output = lambda value: subprocess.run(["bash", "-c", value])\n',
        },
        "caller.py",
        False,
    ),
    (
        "selected_function_decorator",
        {
            "caller.py": "from helpers import output\n"
            'tool = "echo rm"\n'
            'command = f"Report `{tool}`"\n'
            'command = command.removeprefix("Report ")\n'
            'command += " -rf /"\n'
            "output(command)\n",
            "helpers.py": "import subprocess\n"
            "def wrapper(func):\n"
            "    def execute(value):\n"
            '        subprocess.run(["bash", "-c", value])\n'
            "    return execute\n"
            "@wrapper\n"
            "def output(value):\n"
            "    print(value)\n",
        },
        "caller.py",
        False,
    ),
]


@pytest.mark.parametrize("name,files,entry,allowed", _CASES, ids=[case[0] for case in _CASES])
def test_bundled_import_context_and_ledger(name, files, entry, allowed):
    context = PythonImportBounds(list(files), files)
    assert context.allows(entry, files[entry], lambda: None) is allowed
    result = static_runner.run_static_patterns_with_ledger(
        {"components": list(files), "file_cache": files}, [tm]
    )
    event = next(event for event in result["inspection_ledger"] if event["path"] == entry)
    assert event["outcome"] is (LedgerOutcome.COMPLETED if allowed else LedgerOutcome.PARTIAL)


def test_unused_bundled_import_initialization_is_inspected():
    files = {
        "caller.py": "from helpers import output\n",
        "helpers.py": "import side_effect\ndef output(value):\n    print(value)\n",
        "side_effect.py": 'import subprocess\nsubprocess.run(["bash", "-c", "printf init"])\n',
    }
    assert not PythonImportBounds(list(files), files).allows(
        "caller.py", files["caller.py"], lambda: None
    )


def test_missing_bundled_cache_does_not_prove_ownership():
    files = {"caller.py": "from helpers import output\n"}
    assert not PythonImportBounds([*files, "helpers.py"], files).allows(
        "caller.py", files["caller.py"], lambda: None
    )


def test_import_budget_exhaustion_retains_fallback(monkeypatch):
    files = {
        "caller.py": "from helpers import output\n",
        "helpers.py": "def output(value):\n    print(value)\n",
    }
    monkeypatch.setattr(python_import_bounds, "_MAX_IMPORT_DEPTH", 0)
    assert not PythonImportBounds(list(files), files).allows(
        "caller.py", files["caller.py"], lambda: None
    )


def test_interrupted_import_proof_is_not_cached():
    files = {
        "caller.py": "from helpers import output\n",
        "helpers.py": "def output(value):\n    print(value)\n",
    }
    context = PythonImportBounds(list(files), files)

    def interrupted():
        raise TimeoutError("synthetic deadline")

    with pytest.raises(TimeoutError, match="synthetic deadline"):
        context.allows("caller.py", files["caller.py"], interrupted)
    assert context.allows("caller.py", files["caller.py"], lambda: None)


def test_cached_proof_does_not_own_different_projected_source():
    files = {
        "caller.py": "value = 'report'\n",
        "helpers.py": "def output(value):\n    print(value)\n",
    }
    context = PythonImportBounds(list(files), files)
    assert context.allows("caller.py", files["caller.py"], lambda: None)
    assert not context.allows("caller.py", "from helpers import missing\n", lambda: None)


@pytest.mark.parametrize("decorator", ["cache", "lru_cache(maxsize=8)"])
def test_standard_cache_wrapper_preserves_named_output_function(decorator):
    files = {
        "caller.py": "from helpers import output\n",
        "helpers.py": f"from functools import cache, lru_cache\n@{decorator}\ndef output(value):\n    return value.upper()\n",
    }
    assert PythonImportBounds(list(files), files).allows(
        "caller.py", files["caller.py"], lambda: None
    )


def test_bundled_functools_cannot_claim_standard_cache_semantics():
    files = {
        "caller.py": "from helpers import output\n",
        "helpers.py": "from functools import cache\n@cache\ndef output(value):\n    print(value)\n",
        "functools.py": "def cache(value):\n    return value\n",
    }
    assert not PythonImportBounds(list(files), files).allows(
        "caller.py", files["caller.py"], lambda: None
    )


@pytest.mark.parametrize("parent", ["pkg", "pkg/nested"])
def test_imported_module_checks_ancestor_package_initialization(parent):
    files = {
        "caller.py": "from pkg.nested.helpers import output\n",
        "pkg/nested/helpers.py": "def output(value):\n    print(value)\n",
        parent
        + "/__init__.py": 'import subprocess\nsubprocess.run(["bash", "-c", "printf init"])\n',
    }
    assert not PythonImportBounds(list(files), files).allows(
        "caller.py", files["caller.py"], lambda: None
    )


def test_rebound_cache_decorator_does_not_gain_standard_wrapper_ownership():
    files = {
        "caller.py": "from helpers import output\n",
        "helpers.py": "from functools import cache\nfrom custom import replacement as cache\n@cache\ndef output(value):\n    print(value)\n",
    }
    assert not PythonImportBounds(list(files), files).allows(
        "caller.py", files["caller.py"], lambda: None
    )


@pytest.mark.parametrize("initialization", ["initialize()", "list(map(initialize, [None]))"])
def test_local_initialization_call_cannot_install_an_unchecked_executor(initialization):
    files = {
        "caller.py": "from helpers import output\n",
        "helpers.py": (
            "import subprocess\n"
            "def initialize(unused=None):\n"
            "    global runner\n"
            '    runner = lambda value: subprocess.run(["bash", "-c", value])\n'
            f"{initialization}\n"
            "def output(value):\n"
            "    runner(value)\n"
        ),
    }
    assert not PythonImportBounds(list(files), files).allows(
        "caller.py", files["caller.py"], lambda: None
    )
