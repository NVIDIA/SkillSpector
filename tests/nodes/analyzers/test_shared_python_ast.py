# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the shared AST cache across analyzer consumers."""

from __future__ import annotations

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

import skillspector.python_ast as python_ast
from skillspector.graph import graph
from skillspector.nodes.analyzers import (
    behavioral_ast,
    behavioral_taint_tracking,
    static_patterns_data_exfiltration,
    static_patterns_output_handling,
)
from skillspector.nodes.build_context import build_context
from skillspector.python_ast import ParsedPythonFile, get_python_ast


def test_preparsed_python_is_reused_by_all_ast_analyzers(tmp_path, monkeypatch) -> None:
    """One scan parses each eligible Python file once before analyzer fan-out."""
    (tmp_path / "script.py").write_text(
        "import os\n"
        "import subprocess\n"
        "payload = input()\n"
        "environment = os.environ.copy()\n"
        "subprocess.run(output)\n"
        "exec(payload)\n",
        encoding="utf-8",
    )
    original_parse = python_ast.ast.parse
    parse_calls = 0

    def count_parse(*args, **kwargs):
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(*args, **kwargs)

    monkeypatch.setattr(python_ast.ast, "parse", count_parse)
    state = build_context({"skill_path": str(tmp_path)})

    python_ast_cache_key = state["python_ast_cache_key"]
    assert isinstance(python_ast_cache_key, str)
    parsed = get_python_ast(
        python_ast_cache_key,
        state["file_cache"]["script.py"],
        "script.py",
    )
    assert isinstance(parsed, ParsedPythonFile)
    assert parsed.is_parseable
    assert parse_calls == 1

    data_findings = static_patterns_data_exfiltration.node(state)["findings"]
    output_findings = static_patterns_output_handling.node(state)["findings"]
    ast_findings = behavioral_ast.node(state)["findings"]
    taint_findings = behavioral_taint_tracking.node(state)["findings"]

    assert any(finding.rule_id == "E2" for finding in data_findings)
    assert any(finding.rule_id == "OH1" for finding in output_findings)
    assert any(finding.rule_id == "AST1" for finding in ast_findings)
    assert any(finding.rule_id == "TT5" for finding in taint_findings)
    assert parse_calls == 1


def test_uppercase_python_path_reuses_preparsed_ast_for_static_analyzers(
    tmp_path, monkeypatch
) -> None:
    """Static Python inference and cache eligibility use the same case handling."""
    (tmp_path / "script.PY").write_text(
        "import os\nimport subprocess\nos.environ.copy()\nsubprocess.run(output)\n",
        encoding="utf-8",
    )
    original_parse = python_ast.ast.parse
    parse_calls = 0

    def count_parse(*args, **kwargs):
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(*args, **kwargs)

    monkeypatch.setattr(python_ast.ast, "parse", count_parse)
    state = build_context({"skill_path": str(tmp_path)})

    data_findings = static_patterns_data_exfiltration.node(state)["findings"]
    output_findings = static_patterns_output_handling.node(state)["findings"]

    assert any(finding.rule_id == "E2" for finding in data_findings)
    assert any(finding.rule_id == "OH1" for finding in output_findings)
    assert parse_calls == 1


def test_graph_scan_parses_python_once_before_parallel_analyzers(tmp_path, monkeypatch) -> None:
    """The runtime cache shares one parse across the graph's analyzer fan-out."""
    (tmp_path / "script.py").write_text(
        "import os\n"
        "import subprocess\n"
        "payload = input()\n"
        "environment = os.environ.copy()\n"
        "subprocess.run(output)\n"
        "exec(payload)\n",
        encoding="utf-8",
    )
    original_parse = python_ast.ast.parse
    parse_calls = 0

    def count_parse(*args, **kwargs):
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(*args, **kwargs)

    monkeypatch.setattr(python_ast.ast, "parse", count_parse)

    result = graph.invoke({"skill_path": str(tmp_path), "use_llm": False})

    assert {"E2", "OH1", "AST1", "TT5"} <= {finding.rule_id for finding in result["findings"]}
    assert parse_calls == 1
    assert JsonPlusSerializer().dumps_typed(result)
