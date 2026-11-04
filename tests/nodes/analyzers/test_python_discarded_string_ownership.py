# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A discarded string statement cannot carry shell operands to later execution.

The 2.13 shell-ownership backport checks backtick words inside Python strings
as their own candidates. In a module with any recognized shell path, string
values keep the whole-source fallback, so a docstring such as
``Use `tool` on PATH`` made ordinary helpers partially inspected. A docstring
is evaluated and dropped, so its own body bounds those operands unless the
module reads docstrings back.
"""

from __future__ import annotations

import pytest

from skillspector.inspection_ledger import LedgerOutcome
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner
from skillspector.python_tokens import (
    python_discarded_string_spans,
    python_string_value_bounds_allowed,
)

_SHELL_SINK = (
    "import subprocess\n\n\ndef run(cmd):\n    return subprocess.run(cmd, shell=True)\n\n\n"
)
_PADDING = "".join(f"value_{index} = {index}\n" for index in range(300))
_DOCUMENTED = (
    'def find():\n    """Use TOOL_BIN when set, else `tool` on PATH, or None."""\n    return None\n'
)


def _exhausted(source: str) -> bool:
    return tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="python")


@pytest.mark.parametrize(
    "docstring",
    [
        pytest.param('"""Use TOOL_BIN when set, else `tool` on PATH, or None."""', id="function"),
        pytest.param(
            '"""Append `data` to `path`, up to `limit` bytes.\n\n    Whole lines are kept."""',
            id="multiline",
        ),
        pytest.param('"Register the `sync` subcommand."', id="single-quoted"),
    ],
)
def test_docstring_command_words_complete_beside_a_shell_path(docstring: str) -> None:
    source = (
        _SHELL_SINK
        + f"def helper():\n    {docstring}\n    return None\n"
        + 'options = ["--recursive", "--force", "/tmp/build"]\n'
        + _PADDING
    )
    assert not python_string_value_bounds_allowed(source, lambda: None)
    assert not _exhausted(source)


def test_documented_helper_reaches_completed_ledger_outcome() -> None:
    path = "scripts/helper.py"
    source = _SHELL_SINK + _DOCUMENTED + 'options = ["-rf", "/tmp/x"]\n' + _PADDING
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: source}}, [tm]
    )
    row = result["inspection_ledger"][0]
    assert row["outcome"] is LedgerOutcome.COMPLETED
    assert "reason_code" not in row


@pytest.mark.parametrize(
    "tail",
    [
        pytest.param("run(find.__doc__ + ' -rf /')\n", id="doc-attribute"),
        pytest.param("run(getattr(find, '__doc__') + ' -rf /')\n", id="doc-getattr"),
        pytest.param("import inspect\nrun(inspect.getdoc(find) + ' -rf /')\n", id="inspect-getdoc"),
    ],
)
def test_docstrings_read_back_keep_the_whole_source_fallback(tail: str) -> None:
    assert _exhausted(_SHELL_SINK + _DOCUMENTED + tail + _PADDING)


def test_module_docstring_read_through_doc_name_keeps_the_fallback() -> None:
    source = '"""Use `tool` on PATH."""\n' + _SHELL_SINK + "run(__doc__ + ' -rf /')\n" + _PADDING
    assert _exhausted(source)


def test_destructive_operands_inside_a_docstring_remain_partial() -> None:
    source = _SHELL_SINK + 'def helper():\n    """Then `pick` -rf /"""\n' + _PADDING
    assert _exhausted(source)


def test_assigned_string_flowing_to_a_shell_keeps_the_fallback() -> None:
    source = _SHELL_SINK + 'command = f"Report `{tool}`"\ncommand += " -rf /"\nrun(command)\n'
    assert _exhausted(source + _PADDING)


def test_discarded_string_spans_cover_only_string_statements() -> None:
    source = (
        '"""Module `x`."""\n'
        "import argparse\n"
        "def f():\n"
        '    """Run `tool` on PATH."""\n'
        '    value = "kept `y`"\n'
        '    "stray `z`"\n'
        "    é = 1\n"
        '    "after `é`"\n'
        "argparse.ArgumentParser(description=__doc__)\n"
    )
    spans = python_discarded_string_spans(source, lambda: None)
    assert [source[start:end] for start, end in spans] == [
        '"""Run `tool` on PATH."""',
        '"stray `z`"',
        '"after `é`"',
    ]


@pytest.mark.parametrize(
    "source",
    [
        pytest.param('def f(:\n    """`x`"""\n', id="syntax-error"),
        pytest.param('def f():\n    """`x`"""\n\x0c\nprint(1)\n', id="form-feed"),
        pytest.param('def f():\r    """`x`"""\r', id="lone-carriage-return"),
    ],
)
def test_unprovable_sources_have_no_discarded_spans(source: str) -> None:
    assert python_discarded_string_spans(source, lambda: None) == ()


@pytest.mark.parametrize(
    "use",
    [
        "import os\nos.makedirs('out', exist_ok=True)\n",
        "import os\nos.kill(1, 0)\n",
        "import os\nos.readlink('current')\n",
        "try:\n    pass\nexcept Exception as error:\n    print(type(error).__name__)\n",
        "import module\nprint(module.__file__)\n",
    ],
)
def test_non_executing_calls_and_name_attributes_keep_string_bounds(use: str) -> None:
    assert python_string_value_bounds_allowed(use, lambda: None)


@pytest.mark.parametrize(
    "use",
    [
        "import os\nos.system(command)\n",
        "import os\nos.putenv('PATH', value)\n",
        "value = object()\nprint(value.__class__)\n",
        "value = object()\nprint(value.__dict__)\n",
        "value = object()\nprint(value.__globals__)\n",
    ],
)
def test_shell_paths_and_object_escapes_still_disable_string_bounds(use: str) -> None:
    assert not python_string_value_bounds_allowed(use, lambda: None)
