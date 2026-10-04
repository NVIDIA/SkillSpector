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

"""Parity test for the interactive stream and direct invoke graph paths.

The CLI ``scan`` command has two code paths:
  - ``--verbose``: uses ``graph.invoke()`` → returns full final state.
  - interactive default: uses ``graph.stream()`` for progress and returns the
    final ``values`` state.

The stream path must preserve the complete state because the CLI and transitive
scanner consume much more than the rendered report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillspector.cli import FormatChoice, _run_graph_scan

# Keys the CLI reads from the result dict *after* the graph run.
# Derived from cli.py: _write_result, _cleanup_result, exit-code check.
_CLI_CONSUMED_KEYS = frozenset(
    {
        "report_body",
        "sarif_report",
        "risk_score",
        "temp_dir_for_cleanup",
        "execution_successful",
        "analysis_completeness",
        "findings",
        "filtered_findings",
        "effective_finding_ids",
    }
)


@pytest.mark.integration
def test_stream_and_invoke_produce_same_cli_keys(tmp_path: Path) -> None:
    """Non-verbose (stream) result contains every key that verbose (invoke) produces and the CLI consumes."""
    (tmp_path / "SKILL.md").write_text(
        "---\nname: parity-test\n---\n# Safe skill\n", encoding="utf-8"
    )
    invoke_result = _run_graph_scan(
        input_path=str(tmp_path),
        format=FormatChoice.json,
        no_llm=True,
    )
    stream_result = _run_graph_scan(
        input_path=str(tmp_path),
        format=FormatChoice.json,
        no_llm=True,
        stream_progress=True,
    )

    assert set(stream_result) == set(invoke_result)

    # Every key the CLI consumes must be present in *both* results.
    for key in _CLI_CONSUMED_KEYS:
        assert key in invoke_result, f"invoke result missing CLI key: {key}"
        assert key in stream_result, f"stream result missing CLI key: {key}"

    # Reports carry per-run timestamps, so compare their public structure.
    invoke_report = json.loads(invoke_result["report_body"])
    stream_report = json.loads(stream_result["report_body"])
    assert set(invoke_report) == set(stream_report)

    for key in ("risk_score", "sarif_report", "temp_dir_for_cleanup", "execution_successful"):
        assert invoke_result.get(key) == stream_result.get(key)
