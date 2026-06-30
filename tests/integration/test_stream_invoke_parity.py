# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parity test: non-verbose (stream) and verbose (invoke) paths must produce the same CLI-consumed keys.

The CLI ``scan`` command has two code paths:
  - ``--verbose``: uses ``graph.invoke()`` → returns full final state.
  - default (non-verbose): uses ``graph.stream()`` and manually accumulates a
    subset of keys into a ``result`` dict.

If the streaming accumulation loop drifts (e.g. a new key is consumed downstream
but never accumulated), the non-verbose path silently produces wrong output.
This test guards against that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillspector.graph import graph


# Keys the CLI reads from the result dict *after* the graph run.
# Derived from cli.py: _write_result, _cleanup_result, exit-code check.
_CLI_CONSUMED_KEYS = frozenset(
    {
        "report_body",
        "sarif_report",
        "risk_score",
        "temp_dir_for_cleanup",
    }
)


def _stream_result(state: dict) -> dict:
    """Simulate the non-verbose streaming accumulation from cli.py."""
    result: dict = dict(state)
    for update in graph.stream(state, stream_mode="updates"):
        for _node_name, node_output in update.items():
            if "temp_dir_for_cleanup" in node_output:
                result["temp_dir_for_cleanup"] = node_output["temp_dir_for_cleanup"]
            if "report_body" in node_output:
                result["report_body"] = node_output["report_body"]
            if "sarif_report" in node_output:
                result["sarif_report"] = node_output["sarif_report"]
            if "risk_score" in node_output:
                result["risk_score"] = node_output["risk_score"]
    return result


@pytest.mark.integration
def test_stream_and_invoke_produce_same_cli_keys(tmp_path: Path) -> None:
    """Non-verbose (stream) result contains every key that verbose (invoke) produces and the CLI consumes."""
    (tmp_path / "SKILL.md").write_text(
        "---\nname: parity-test\n---\n# Safe skill\n", encoding="utf-8"
    )
    state: dict = {
        "skill_path": str(tmp_path),
        "output_format": "json",
        "use_llm": False,
    }

    invoke_result = graph.invoke(dict(state))
    stream_result = _stream_result(dict(state))

    # Every key the CLI consumes must be present in *both* results.
    for key in _CLI_CONSUMED_KEYS:
        assert key in invoke_result, f"invoke result missing CLI key: {key}"
        assert key in stream_result, f"stream result missing CLI key: {key}"

    # The actual *values* of the CLI-consumed keys should match (structurally).
    # For report_body we compare parsed JSON keys because timestamps differ
    # between separate runs.
    for key in _CLI_CONSUMED_KEYS:
        inv = invoke_result.get(key)
        stm = stream_result.get(key)
        if key == "report_body":
            # Both should parse to JSON with the same top-level keys
            inv_parsed = json.loads(inv)
            stm_parsed = json.loads(stm)
            assert set(inv_parsed.keys()) == set(stm_parsed.keys()), (
                f"report_body top-level keys differ: "
                f"invoke={set(inv_parsed.keys())}, stream={set(stm_parsed.keys())}"
            )
        else:
            assert inv == stm, (
                f"value mismatch for CLI key {key!r}: "
                f"invoke={inv!r}, stream={stm!r}"
            )
