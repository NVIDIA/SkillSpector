# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TypeScript template literals must not be mistaken for shell backticks."""

from __future__ import annotations

import pytest

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner
from skillspector.typescript_tokens import typescript_template_spans

_PADDING = "\n// ordinary padding\n" * 400


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "export const label = (n: number): string => `${format(n)}`;",
            id="typed-arrow-call",
        ),
        pytest.param("const value = `${item.name}`;", id="member-expression"),
        pytest.param("const value = `${ item }`;", id="spaced-identifier"),
        pytest.param("const value = `${left ?? right}`;", id="nullish-expression"),
        pytest.param("const value = `${items[0]}`;", id="computed-member"),
        pytest.param("const value = `outer ${`inner ${item.name}`}`;", id="nested-template"),
        pytest.param("type Value = `prefix-${string}`;", id="template-literal-type"),
        pytest.param(
            "function identity<T>(value: T): T { return value; }\n"
            "const label = `${identity('x')}`;",
            id="generic-function",
        ),
    ],
)
def test_benign_typescript_templates_do_not_exhaust_shell_parser(source: str) -> None:
    assert (
        tm.has_bounded_parse_exhaustion(source + _PADDING, lambda: None, file_type="typescript")
        is False
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("exec(`${command} -rf /`);", id="direct-exec"),
        pytest.param(
            "const child = require('child_process'); child.exec(`${tool} -rf /`);",
            id="child-process-alias",
        ),
        pytest.param(
            "import { spawnSync } from 'node:child_process'; spawnSync(`${tool} -rf /`);",
            id="node-child-process-import",
        ),
        pytest.param("import { $ } from 'zx'; $`rm -rf /`;", id="zx-tagged-template"),
        pytest.param("const value = `unterminated;", id="unterminated-template"),
        pytest.param("const = `malformed ${value}`;", id="malformed-declaration"),
    ],
)
def test_unproven_typescript_keeps_conservative_parse_result(source: str) -> None:
    assert (
        tm.has_bounded_parse_exhaustion(source + _PADDING, lambda: None, file_type="typescript")
        is True
    )


def test_tagged_template_without_shell_surface_is_not_owned() -> None:
    assert typescript_template_spans("String.raw`${value}`;", lambda: None) is None


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("tag<string>`rm -rf /`;", id="generic-tag"),
        pytest.param("tag!`rm -rf /`;", id="postfix-tag"),
    ],
)
def test_typed_tags_are_not_owned(source: str) -> None:
    assert typescript_template_spans(source, lambda: None) is None


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("#!/bin/sh\nvalue=`rm -rf /`;\n", id="shell"),
        pytest.param("#!/usr/bin/env python\nvalue=`rm -rf /`;\n", id="python"),
    ],
)
def test_non_host_shebang_is_not_owned(source: str) -> None:
    assert typescript_template_spans(source, lambda: None) is None


def test_host_shebang_allows_templates() -> None:
    source = "#!/usr/bin/env -S tsx\nconst value = `${item.name}`;\n"

    assert typescript_template_spans(source, lambda: None) is not None


def test_unclosed_code_delimiters_are_not_owned() -> None:
    assert typescript_template_spans("foo(`${value}`;", lambda: None) is None


def test_interpolation_cannot_close_outer_delimiters() -> None:
    assert typescript_template_spans("foo(`${bar)}`;", lambda: None) is None


def test_regex_literal_backtick_is_not_owned_as_template() -> None:
    source = "if (ok) /`/.test(value); const label = `${item.name}`;"

    starts, _ = typescript_template_spans(source, lambda: None) or ((), ())
    assert starts == (source.index("`", source.index("const label")),)


def test_division_before_template_is_not_mistaken_for_regex() -> None:
    source = "const ratio = left / right; const label = `${item.name}`;"

    assert typescript_template_spans(source, lambda: None) is not None


def test_template_ownership_requires_complete_context() -> None:
    source = "const label = `${item.name}`;" + _PADDING

    assert (
        tm.has_bounded_parse_exhaustion(
            source, lambda: None, file_type="typescript", complete_context=False
        )
        is True
    )


def test_destructive_command_in_executable_template_still_has_tm1() -> None:
    findings = tm.analyze("exec(`rm -rf /`);", "install.ts", "typescript")

    assert any(finding.rule_id == "TM1" for finding in findings)


def test_ledger_completes_benign_templates_and_preserves_dynamic_partial_result() -> None:
    benign_path = "scripts/render.ts"
    benign_source = "const label = `frame ${index}.png`;" + _PADDING
    benign = static_runner.run_static_patterns_with_ledger(
        {"components": [benign_path], "file_cache": {benign_path: benign_source}}, [tm]
    )
    benign_row = benign["inspection_ledger"][0]
    assert benign_row["outcome"] is LedgerOutcome.COMPLETED
    assert "reason_code" not in benign_row

    dynamic_path = "scripts/run.ts"
    dynamic_source = "exec(`${command} -rf /`);" + _PADDING
    dynamic = static_runner.run_static_patterns_with_ledger(
        {"components": [dynamic_path], "file_cache": {dynamic_path: dynamic_source}}, [tm]
    )
    dynamic_row = dynamic["inspection_ledger"][0]
    assert dynamic_row["outcome"] is LedgerOutcome.PARTIAL
    assert dynamic_row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


def test_tsx_paths_use_the_typescript_ownership_path() -> None:
    path = "scripts/render.tsx"
    source = "const label = `${format(index)}`;" + _PADDING

    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: source}}, [tm]
    )

    row = result["inspection_ledger"][0]
    assert row["outcome"] is LedgerOutcome.COMPLETED
    assert "reason_code" not in row


def test_declaration_keywords_do_not_read_as_execution_calls() -> None:
    # ``function`` is a call name in the shared JavaScript list but a
    # declaration keyword here, so a generic function body stays benign.
    source = (
        "function identity<T>(value: T): T { return value; }\nconst label = `${identity('x')}`;"
    )

    assert typescript_template_spans(source, lambda: None) is not None
    assert (
        tm.has_bounded_parse_exhaustion(source + _PADDING, lambda: None, file_type="typescript")
        is False
    )
