# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JavaScript template literals must not be mistaken for shell backticks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from skillspector import javascript_ast as js
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.javascript_tokens import javascript_template_spans
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner

_PADDING = "\n// ordinary padding\n" * 400


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "const output = path.resolve(outDir, `${id}_${String(frame).padStart(5, '0')}.png`);",
            id="filename",
        ),
        pytest.param("const r = (s) => random(`${startFrame}-${i}-${s}`);", id="random-seed"),
        pytest.param(
            "pts.push(`${cx + Math.cos(t) * rx} ${cy + Math.sin(t) * ry}`);",
            id="svg-points",
        ),
        pytest.param(
            'throw new Error(`Unknown scene kind "${kind}". Known: ${Object.keys(kinds).join(", ")}`);',
            id="scene-error",
        ),
    ],
)
def test_benign_template_literals_do_not_exhaust_shell_parser(source: str) -> None:
    assert (
        tm.has_bounded_parse_exhaustion(source + _PADDING, lambda: None, file_type="javascript")
        is False
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("exec(`${command} -rf /`);", id="direct-argument"),
        pytest.param("const command = `${tool} -rf /`;\nexec(command);", id="assigned-template"),
    ],
)
def test_dynamic_template_in_file_with_exec_stays_partial(source: str) -> None:
    source += _PADDING

    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript") is True


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "const { execSync } = require('child_process'); execSync(`$TOOL -rf /`);",
            id="static-template-with-shell-variable",
        ),
        pytest.param(
            "const child = require('child_process'); child.execSync(`${tool} -rf /`);",
            id="required-module-alias",
        ),
        pytest.param(
            "const childProcess = require('child_process'); childProcess.execSync(`${tool} -rf /`);",
            id="child-process-alias",
        ),
        pytest.param(
            "const proc = require('child_process'); proc.spawnSync(`${tool} -rf /`);",
            id="proc-spawn-alias",
        ),
        pytest.param(
            "require('child_process').execSync(`${tool} -rf /`);",
            id="require-call-member-receiver",
        ),
        pytest.param("shelljs.exec(`${tool} -rf /`);", id="shelljs-member-call"),
        pytest.param("import execa from 'execa'; execa(`${tool} -rf /`);", id="execa-import"),
        pytest.param("import { $ } from 'zx'; $`rm -rf /`;", id="zx-tagged-template"),
    ],
)
def test_shell_execution_forms_keep_template_files_partial(source: str) -> None:
    assert (
        tm.has_bounded_parse_exhaustion(source + _PADDING, lambda: None, file_type="javascript")
        is True
    )


@pytest.mark.parametrize(
    "source", ["const value = `unterminated;", "const = `malformed ${value}`;"]
)
def test_unproven_javascript_keeps_conservative_parse_result(source: str) -> None:
    assert (
        tm.has_bounded_parse_exhaustion(source + _PADDING, lambda: None, file_type="javascript")
        is True
    )


def test_template_ownership_requires_complete_context() -> None:
    source = "const output = `${frame}.png`;" + _PADDING

    assert (
        tm.has_bounded_parse_exhaustion(
            source, lambda: None, file_type="javascript", complete_context=False
        )
        is True
    )


def test_destructive_command_in_executable_template_still_has_tm1() -> None:
    findings = tm.analyze("exec(`rm -rf /`);", "install.js", "javascript")

    assert any(finding.rule_id == "TM1" for finding in findings)


def test_ledger_completes_benign_templates_and_preserves_dynamic_partial_result() -> None:
    benign_path = "scripts/render.js"
    benign_source = "const output = `frame ${index}.png`;" + _PADDING
    benign = static_runner.run_static_patterns_with_ledger(
        {"components": [benign_path], "file_cache": {benign_path: benign_source}}, [tm]
    )
    benign_row = benign["inspection_ledger"][0]
    assert benign_row["outcome"] is LedgerOutcome.COMPLETED
    assert "reason_code" not in benign_row

    dynamic_path = "scripts/run.js"
    dynamic_source = "exec(`${command} -rf /`);" + _PADDING
    dynamic = static_runner.run_static_patterns_with_ledger(
        {"components": [dynamic_path], "file_cache": {dynamic_path: dynamic_source}}, [tm]
    )
    dynamic_row = dynamic["inspection_ledger"][0]
    assert dynamic_row["outcome"] is LedgerOutcome.PARTIAL
    assert dynamic_row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize(
    "member",
    [
        pytest.param("[0]", id="number"),
        pytest.param("[null]", id="null"),
        pytest.param("[false]", id="boolean"),
    ],
)
def test_non_string_computed_member_with_template_does_not_crash(member: str) -> None:
    source = f"items{member}(); const name = `file-${{id}}`;" + _PADDING

    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript") is False


@pytest.mark.parametrize(
    "source",
    [
        "const config = {...defaults, limit: 10_000}; const label = `item ${config?.name ?? 'none'}`;",
        "import {fileURLToPath} from 'node:url'; const label = `${fileURLToPath(import.meta.url)}`;",
        "class View { #name = 'chart'; label() { return `chart ${this.#name}`; } }",
        "const label = `outer ${items.map(item => `inner ${item?.name ?? ''}`).join(',')}`;",
        "#!/usr/bin/env node\nconst result = await Promise.resolve('done'); const label = `result ${result}`;",
    ],
)
def test_modern_javascript_preserves_template_host_ownership(source: str) -> None:
    assert (
        tm.has_bounded_parse_exhaustion(source + _PADDING, lambda: None, file_type="javascript")
        is False
    )


def test_shared_parser_retains_exact_unicode_and_comment_offsets() -> None:
    source = "// café 😀\nconst label = `日本語 ${value?.name ?? 'none'}`;"
    program = js.parse_javascript_source(source, lambda: None)
    assert program is not None
    comment = next(node for node in program.nodes if node.type == "comment")
    template = next(node for node in program.nodes if node.type == "template_string")
    assert program.text(comment) == "// café 😀"
    assert program.span(template) == (source.index("`"), len(source) - 1)
    assert program.text(template) == "`日本語 ${value?.name ?? 'none'}`"


@pytest.mark.parametrize(
    "source",
    [
        "const label = `unterminated ${value};",
        "const = `invalid ${value}`;",
        "const typed: string = `unsupported ${value}`;",
        "const value = '\x00';",
        "const value = '\ud800';",
        "entry?.label = `value ${input}`;",
        "entry?.count++; const value = `${input}`;",
        "(entry?.label) = `value ${input}`;",
        "new Constructor?.(`${input}`);",
        "entry?.tag`value ${input}`;",
        "#!/bin/sh\nconst value = `${input}`;",
        "const value = `${input}`;\n#!/usr/bin/env node",
    ],
)
def test_shared_parser_returns_no_ownership_for_unusable_source(source: str) -> None:
    assert js.parse_javascript_source(source, lambda: None) is None


@pytest.mark.parametrize(
    "source",
    [
        "target[entry?.key] = 1;",
        "target[entry?.key]++;",
        "(entry?.target).value = 1;",
        "new (entry?.Constructor)();",
        "new Constructor(entry?.value);",
        "(entry?.tag)`value`;",
    ],
)
def test_independent_optional_expressions_keep_valid_syntax(source: str) -> None:
    assert js.parse_javascript_source(source, lambda: None) is not None


@pytest.mark.parametrize(
    "source",
    [
        'const metadata = {eval: false, Function: "render", label: "plot"}; console.log(`${metadata?.label}`);',
        "const label = source[key]; console.log(`Value: ${label}`);",
    ],
)
def test_data_properties_do_not_acquire_execution_semantics(source: str) -> None:
    assert javascript_template_spans(source, lambda: None) is not None
    assert (
        tm.has_bounded_parse_exhaustion(source + _PADDING, lambda: None, file_type="javascript")
        is False
    )


@pytest.mark.parametrize("name", ["run", "exec", "spawn"])
@pytest.mark.parametrize("value", ["1", "'frame'", "true", "false", "null"])
def test_const_scalar_execution_names_retain_template_ownership(name: str, value: str) -> None:
    source = f"const {name} = {value}; const label = `frame ${{{name}}}.png`;"
    unresolved: list[int] = []
    assert (
        javascript_template_spans(source, lambda: None, unresolved_executables=unresolved)
        is not None
    )
    assert not unresolved
    assert (
        tm.has_bounded_parse_exhaustion(source + _PADDING, lambda: None, file_type="javascript")
        is False
    )


@pytest.mark.parametrize(
    "source",
    [
        "let run = 1; const label = `${run}`;",
        "function label() { const run = 1; return `${run}`; }",
        "export const run = 1; const label = `${run}`;",
        "const run = 1; { const run = command; const label = `${run}`; }",
        "const run = 1; function label(run) { return `${run}`; }",
        "const run = 1; function label({run}) { return `${run}`; }",
        "const run = 1; run = command; const label = `${run}`;",
        "const run = 1; run++; const label = `${run}`;",
        "const run = 1; const alias = run; const label = `${run}`;",
        "const run = 1; run(); const label = `${run}`;",
        "const run = 1; run.member; const label = `${run}`;",
        "const run = 1; accept(run); const label = `${run}`;",
        "const run = 1; const receiver = {run}; const label = `${run}`;",
        "const run = 1; receiver.run(); const label = `${run}`;",
        "const run = 1; with (receiver) { const label = `${run}`; }",
        "const label = `${run}`; const run = 1;",
        "const eval = 1; const label = `${eval}`;",
        "const Function = 1; const label = `${Function}`;",
    ],
)
def test_unproven_scalar_execution_names_remain_partial(source: str) -> None:
    unresolved: list[int] = []
    assert (
        javascript_template_spans(source, lambda: None, unresolved_executables=unresolved) is None
    )
    assert unresolved
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript") is True


def test_proven_assertion_message_retains_host_ownership() -> None:
    source = (
        "import assert from 'node:assert/strict'; assert.equal(actual, expected, `value ${label}`);"
    )
    unresolved: list[int] = []
    assert (
        javascript_template_spans(source, lambda: None, unresolved_executables=unresolved)
        is not None
    )
    assert not unresolved


@pytest.mark.parametrize(
    ("limit", "value", "source"),
    [
        ("MAX_JAVASCRIPT_SOURCE_CHARS", 10, "const label = `long`;"),
        ("_MAX_NODES", 20, "const values = [" + ",".join(str(i) for i in range(30)) + "];"),
        ("_MAX_DEPTH", 8, "const value = " + "(" * 20 + "1" + ")" * 20 + ";"),
    ],
)
def test_shared_parser_preserves_resource_bounds(
    monkeypatch: pytest.MonkeyPatch, limit: str, value: int, source: str
) -> None:
    monkeypatch.setattr(js, limit, value)
    assert js.parse_javascript_source(source, lambda: None) is None


@pytest.mark.parametrize("expire_at", [1, 2, 3], ids=["entry", "before-native", "after-native"])
def test_parser_preserves_caller_runtime_signal(expire_at: int) -> None:
    checks = 0

    def check_runtime() -> None:
        nonlocal checks
        checks += 1
        if checks >= expire_at:
            raise TimeoutError("test runtime exhausted")

    with pytest.raises(TimeoutError, match="test runtime exhausted"):
        js.parse_javascript_source("const label = `value ${item}`;", check_runtime)


def test_native_timeout_does_not_depend_on_python_clock_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Freeze only the helper's clock: the native parser must enforce its own
    # finite budget rather than relying on a check after unlimited native work.
    monkeypatch.setattr(js, "time", SimpleNamespace(monotonic=lambda: 0.0))
    source = "const values = [" + "'value'," * 5_000 + "];"
    assert js.parse_javascript_source(source, lambda: None) is not None
    monkeypatch.setattr(js, "_MAX_SECONDS", 0.000001)
    assert js.parse_javascript_source(source, lambda: None) is None


@pytest.mark.parametrize(
    "source",
    [
        "import { exec as launch } from 'node:child_process'; launch(`${command}`);",
        "import {exec} from 'node:child_process'; const marker = 'node:assert'; exec(`${command}`);",
        "const cp = require('node:child_process'); const launch = cp.exec; launch(`${command}`);",
        "const launch = require(moduleName); launch(`${command}`);",
        r"\u0065xec(`${command}`);",
        "runner['exec'](`${command}`);",
        "Reflect.apply(runner, null, [`${command}`]);",
        "const runner = execute.bind(null); runner(`${command}`);",
        "getRunner()(`${command}`);",
        "const invoke = globalThis[key]; invoke?.(`${command}`);",
        "const 調用 = globalThis[key]; 調用?.(`${command}`);",
        "const [invoke] = [globalThis[key]]; invoke?.(`${command}`);",
        "const {invoke} = {invoke: source[key]}; invoke?.(`${command}`);",
        "const receiver = {invoke: globalThis[key]}; receiver.invoke(`${command}`);",
        "const root = globalThis; const receiver = {invoke: root[key]}; receiver.invoke?.(`${command}`);",
        "const root = flag ? globalThis : window; const receiver = {invoke: root[key]}; receiver.invoke(`${command}`);",
        "import {launch} from 'unknown'; launch?.(`${command}`);",
        "import {launch} from 'unknown'; const text = `${command}`; const alias = text; launch?.(alias);",
        "import * as tools from 'unknown'; tools.launch?.(`${command}`);",
        "let invoke; invoke = globalThis[key]; invoke(`${command}`);",
        "function main(invoke) { return invoke(`${command}`); }",
        "unknown`rm -rf /`;",
    ],
)
def test_indirect_or_unknown_execution_retains_conservative_templates(source: str) -> None:
    unresolved: list[int] = []
    assert (
        javascript_template_spans(source, lambda: None, unresolved_executables=unresolved) is None
    )
    assert unresolved
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript") is True
    assert (
        tm.has_bounded_parse_exhaustion(source + _PADDING, lambda: None, file_type="javascript")
        is True
    )


def test_literal_shell_template_keeps_raw_inspection_and_findings() -> None:
    source = "import { execSync } from 'node:child_process'; execSync(`rm -rf /`);"
    unresolved: list[int] = []
    assert (
        javascript_template_spans(source, lambda: None, unresolved_executables=unresolved) is None
    )
    assert not unresolved
    assert any(
        finding.rule_id == "TM1" for finding in tm.analyze(source, "install.mjs", "javascript")
    )


@pytest.mark.parametrize("source", ["exec(`${command}`);", "exec(`$COMMAND`);", "const = `bad`;"])
def test_short_unknown_templates_produce_partial_ledger_evidence(source: str) -> None:
    path = "scripts/run.mjs"
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: source}}, [tm]
    )
    row = result["inspection_ledger"][0]
    assert row["outcome"] is LedgerOutcome.PARTIAL
    assert row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
