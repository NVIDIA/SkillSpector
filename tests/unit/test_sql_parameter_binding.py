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

"""SQL parameter binding and adverse OH1 ownership regressions."""

from __future__ import annotations

import pytest

from skillspector.nodes.analyzers import static_patterns_output_handling as output_handling
from skillspector.python_ast import parse_python_source


def _database_source(statement: str, driver: str = "psycopg") -> str:
    return (
        f"import {driver}\n"
        f"connection = {driver}.connect('example')\n"
        "cursor = connection.cursor()\n"
        f"{statement}\n"
    )


@pytest.mark.parametrize(
    "statement,driver",
    [
        ('cursor.execute("SELECT name FROM users WHERE id = %s", (output,))', "psycopg"),
        ('cursor.execute("SELECT name FROM users WHERE id = %(id)s", {"id": output})', "psycopg2"),
        (
            'cursor.execute(query="SELECT name FROM users WHERE id = %s", params=(response_id,))',
            "psycopg",
        ),
        (
            "cursor.execute(\"SELECT name FROM users WHERE id = ? AND note = '50%'\", (output,))",
            "sqlite3",
        ),
        ('cursor.execute("SELECT name FROM users WHERE id = %s", (output,))', "psycopg2"),
        (
            "connection.execute(\"SELECT name FROM users WHERE id = ? AND note = '50%'\", (output,))",
            "sqlite3",
        ),
    ],
)
def test_bound_database_query_is_not_output_injection(statement: str, driver: str) -> None:
    findings = output_handling.analyze(_database_source(statement, driver), "query.py", "python")
    assert not [finding for finding in findings if finding.rule_id == "OH1"]


@pytest.mark.parametrize(
    "query",
    [
        '"SELECT name FROM users WHERE id = " + output',
        '"SELECT name FROM users WHERE id = %s" % output',
        '"SELECT name FROM users WHERE id = {}".format(output)',
        'f"SELECT name FROM users WHERE id = {output}"',
    ],
)
def test_query_interpolation_remains_detected_even_with_parameters(query: str) -> None:
    source = _database_source(f"cursor.execute({query}, (user_id,))")
    assert any(
        finding.rule_id == "OH1"
        for finding in output_handling.analyze(source, "query.py", "python")
    )


@pytest.mark.parametrize(
    "source",
    [
        'cursor.execute("SELECT name FROM users WHERE id = %s", (output,))',
        _database_source(
            'initialize_plugin()\ncursor.execute("SELECT name FROM users WHERE id = %s", (output,))'
        ),
        'custom.query("SELECT name FROM users WHERE id = %s", (output,))',
        'execute("SELECT name FROM users WHERE id = %s", (output,))',
        _database_source(
            'cursor = custom\ncursor.execute("SELECT name FROM users WHERE id = %s", (output,))'
        ),
        _database_source(
            'cursor.execute = custom\ncursor.execute("SELECT name FROM users WHERE id = %s", (output,))'
        ),
        'import psycopg\npsycopg.connect = custom\nconnection = psycopg.connect("example")\ncursor = connection.cursor()\ncursor.execute("SELECT name FROM users WHERE id = %s", (output,))',
        'import psycopg\ndef run(psycopg):\n    cursor = psycopg.connect("example").cursor()\n    cursor.execute("SELECT name FROM users WHERE id = %s", (output,))',
        'import psycopg\nconnection = psycopg.connect("example", cursor_factory=custom)\ncursor = connection.cursor()\ncursor.execute("SELECT name FROM users WHERE id = %s", (output,))',
        _database_source(
            'custom(cursor)\ncursor.execute("SELECT name FROM users WHERE id = %s", (output,))'
        ),
        _database_source(
            'other = cursor\nother.execute = custom\ncursor.execute("SELECT name FROM users WHERE id = %s", (output,))'
        ),
        _database_source(
            'cursor.execute("SELECT name FROM users WHERE id = %s", (output,))\nif incomplete:'
        ),
    ],
)
def test_unknown_mutated_or_malformed_database_owner_stays_conservative(source: str) -> None:
    assert any(
        finding.rule_id == "OH1"
        for finding in output_handling.analyze(source, "query.py", "python")
    )


def test_safe_call_cannot_hide_adjacent_interpolation_with_unicode_offsets() -> None:
    source = _database_source(
        'label = "café"; cursor.execute("SELECT name FROM users WHERE id = %s", (output,)); '
        'cursor.execute("SELECT name FROM users WHERE id = " + output)'
    )
    findings = [
        finding
        for finding in output_handling.analyze(source, "query.py", "python")
        if finding.rule_id == "OH1"
    ]
    assert len(findings) == 1
    assert " + output" in (findings[0].matched_text or "")
    assert findings[0].location.start_line == 4


def test_embedded_python_keeps_lexical_sql_fallback() -> None:
    content = (
        "```python\n"
        + _database_source('cursor.execute("SELECT name FROM users WHERE id = %s", (output,))')
        + "```\n"
    )
    assert any(
        finding.rule_id == "OH1"
        for finding in output_handling.analyze(content, "SKILL.md", "markdown")
    )


def test_mismatched_source_parse_keeps_lexical_fallback() -> None:
    source = _database_source('cursor.execute("SELECT name FROM users WHERE id = %s", (output,))')
    parsed = parse_python_source(source.replace("SELECT", "SELECT\u200b"), "query.py")
    findings = output_handling.analyze(source, "query.py", "python", python_ast=parsed)
    assert any(finding.rule_id == "OH1" for finding in findings)


def test_intervening_import_keeps_lexical_finding() -> None:
    source = _database_source(
        'import plugin\ncursor.execute("SELECT name FROM users WHERE id = %s", (output,))'
    )
    assert any(f.rule_id == "OH1" for f in output_handling.analyze(source, "query.py", "python"))


@pytest.mark.parametrize(
    "setup",
    [
        'import sqlite3\nconnection = sqlite3.connect("example", factory=custom)\ncursor = connection.cursor()',
        'import psycopg\nconnection = psycopg.connect("example")\ncursor = connection.cursor(factory=custom)',
        'import psycopg\nconnection = psycopg.connect("example")\ncursor = connection.cursor()\nsetattr(cursor, "execute", custom)',
        'import psycopg\nconnection = psycopg.connect("example")\ncursor = connection.cursor()\ndef patch():\n    cursor.execute = custom',
    ],
)
def test_custom_factories_and_reflective_or_deferred_writes_stay_conservative(setup: str) -> None:
    source = setup + '\ncursor.execute("SELECT name FROM users WHERE id = %s", (output,))'
    assert any(f.rule_id == "OH1" for f in output_handling.analyze(source, "query.py", "python"))


def test_proof_budget_exhaustion_retains_lexical_finding() -> None:
    source = _database_source('cursor.execute("SELECT name FROM users WHERE id = %s", (output,))')
    source += "0\n" * 10_001
    assert any(f.rule_id == "OH1" for f in output_handling.analyze(source, "query.py", "python"))


def test_outer_binding_cannot_hide_nested_sql_interpolation() -> None:
    source = _database_source(
        'cursor.execute("SELECT \'100%\' WHERE value=?", (cursor.execute("SELECT " + output),))',
        "sqlite3",
    )
    assert any(f.rule_id == "OH1" for f in output_handling.analyze(source, "query.py", "python"))


def test_relative_driver_import_is_not_trusted() -> None:
    source = (
        "from .sqlite3 import connect\n"
        'connection = connect("example")\n'
        "cursor = connection.cursor()\n"
        "cursor.execute(\"SELECT '100%' WHERE value=?\", (output,))\n"
    )
    assert any(f.rule_id == "OH1" for f in output_handling.analyze(source, "query.py", "python"))


@pytest.mark.parametrize("pattern", ["cursor", "[first, *cursor]", '{"item": value, **cursor}'])
def test_match_capture_rebinding_stays_conservative(pattern: str) -> None:
    source = _database_source(
        f"match custom:\n    case {pattern}:\n"
        "        cursor.execute(\"SELECT '100%' WHERE value=?\", (output,))"
    )
    assert any(f.rule_id == "OH1" for f in output_handling.analyze(source, "query.py", "python"))


@pytest.mark.parametrize(
    "statement",
    [
        "with manager:\n    cursor.execute(\"SELECT '100%' WHERE value=?\", (output,))",
        "other = external.attribute\ncursor.execute(\"SELECT '100%' WHERE value=?\", (output,))",
        "other = external[0]\ncursor.execute(\"SELECT '100%' WHERE value=?\", (output,))",
        "other = external + value\ncursor.execute(\"SELECT '100%' WHERE value=?\", (output,))",
    ],
)
def test_implicit_side_effect_before_query_stays_conservative(statement: str) -> None:
    source = _database_source(statement, "sqlite3")
    assert any(f.rule_id == "OH1" for f in output_handling.analyze(source, "query.py", "python"))


def test_conditional_driver_binding_stays_conservative() -> None:
    source = (
        "if enabled:\n    import sqlite3\n"
        'connection = sqlite3.connect("example")\n'
        "cursor = connection.cursor()\n"
        "cursor.execute(\"SELECT '100%' WHERE value=?\", (output,))\n"
    )
    assert any(f.rule_id == "OH1" for f in output_handling.analyze(source, "query.py", "python"))


def test_many_bound_queries_preserve_trailing_sql_injection() -> None:
    source = _database_source(
        'cursor.execute("SELECT name FROM users WHERE id = %s", (output,))\n' * 1000
        + 'cursor.execute("SELECT name FROM users WHERE id = " + output)'
    )
    findings = [
        finding
        for finding in output_handling.analyze(source, "query.py", "python")
        if finding.rule_id == "OH1"
    ]
    assert len(findings) == 1
    assert " + output" in (findings[0].matched_text or "")


def test_wildcard_driver_import_keeps_lexical_finding() -> None:
    source = _database_source(
        'from psycopg import *\ncursor.execute("SELECT name FROM users WHERE id = %s", (output,))'
    )
    assert any(f.rule_id == "OH1" for f in output_handling.analyze(source, "query.py", "python"))
