# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static imports stay light without changing graph warning or deadline behavior."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_NETWORK_GUARD = """
import socket
network_attempts = 0

def deny_network(*args, **kwargs):
    global network_attempts
    network_attempts += 1
    raise AssertionError('graph initialization attempted network access')

socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
socket.create_connection = deny_network
"""


def _environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "PYTHONPATH", "LANG", "PYTHONDONTWRITEBYTECODE"}
    } | {"LANGSMITH_TRACING": "false"}


def test_cold_package_import_does_not_initialize_workflow_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            _NETWORK_GUARD
            + """
import sys
import skillspector

assert 'langchain_core' not in sys.modules
assert 'skillspector.graph' not in sys.modules
assert 'skillspector._workflow_warnings' not in sys.modules
assert callable(skillspector.create_graph)
assert network_attempts == 0
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_environment(),
        timeout=3,
    )
    assert result.stdout == ""


@pytest.mark.parametrize("first_access", ["lazy", "factory", "direct", "preloaded"])
def test_graph_initialization_preserves_warning_order_without_inference(first_access: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            _NETWORK_GUARD
            + """
import importlib
import sys
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    if sys.argv[1] == 'preloaded':
        import langchain_core
    import skillspector
    if sys.argv[1] == 'factory':
        compiled = skillspector.create_graph()
    elif sys.argv[1] == 'direct':
        compiled = importlib.import_module('skillspector.graph')
    else:
        compiled = skillspector.graph
    for name in ('invoke', 'ainvoke', 'stream'):
        assert callable(getattr(compiled, name))
    assert 'skillspector._workflow_warnings' in sys.modules
    assert 'langchain_core' in sys.modules
    warnings.warn('The default value of `allowed_objects` will change soon', UserWarning)
    warnings.warn('unrelated warning remains visible', UserWarning)
    visible = [str(item.message) for item in caught]
    assert not any('The default value of `allowed_objects` will change' in item for item in visible)
    assert 'unrelated warning remains visible' in visible
    assert network_attempts == 0
""",
            first_access,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_environment(),
        timeout=30,
    )
    assert result.stdout == ""


def test_import_change_does_not_remove_end_to_end_timeout() -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            [sys.executable, "-B", "-c", "import skillspector\nwhile True: pass\n"],
            check=True,
            capture_output=True,
            text=True,
            env=_environment(),
            timeout=3,
        )
