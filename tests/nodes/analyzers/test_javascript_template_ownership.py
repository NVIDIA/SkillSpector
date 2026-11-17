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
    "source",
    [
        pytest.param(
            "export function check(value, max = Number.MAX_SAFE_INTEGER) {\n"
            "  const n = Number(value);\n"
            "  return n <= (max ?? Number.MAX_SAFE_INTEGER) ? `ok: ${n}` : `too large: ${n}`;\n"
            "}\n"
            "console.log(check(process.argv[2] ?? '0'));\n",
            id="number-max-safe-integer-default",
        ),
        pytest.param(
            "export function check(value, max) {\n"
            "  const n = Number(value);\n"
            "  return n <= (max ?? Number.MAX_SAFE_INTEGER) ? `ok: ${n}` : `too large: ${n}`;\n"
            "}\n"
            "console.log(check(process.argv[2] ?? '0'));\n",
            id="plain-parameter-control",
        ),
        pytest.param(
            "function defaultPath() { return 'a.json'; }\n"
            "export function load(p = defaultPath()) { return `${p}:${defaultPath()}`; }\n",
            id="helper-call-default",
        ),
        pytest.param(
            "function defaultPath() { return 'a.json'; }\n"
            "export function load(p) { return `${p ?? defaultPath()}`; }\n",
            id="plain-parameter-helper-control",
        ),
        pytest.param(
            "export function show({max = Number.MAX_SAFE_INTEGER} = {}) {\n"
            "  const n = Number(max);\n"
            "  return `max=${n}`;\n"
            "}\n",
            id="destructured-object-default",
        ),
        pytest.param(
            "export function show([first = Number.MAX_SAFE_INTEGER] = []) {\n"
            "  const n = Number(first);\n"
            "  return `n=${n}`;\n"
            "}\n",
            id="destructured-array-default",
        ),
        pytest.param(
            "const check = (value, max = Number.MAX_SAFE_INTEGER) => {\n"
            "  const n = Number(value);\n"
            "  return `n=${n}`;\n"
            "};\n",
            id="arrow-function-default",
        ),
        pytest.param(
            "export function show(a, ...rest) {\n  const n = Number(a);\n  return `n=${n}`;\n}\n",
            id="rest-parameter-control",
        ),
        pytest.param(
            "class Checker {\n"
            "  check(value, max = Number.MAX_SAFE_INTEGER) {\n"
            "    const n = Number(value);\n"
            "    return `n=${n}`;\n"
            "  }\n"
            "}\n",
            id="class-method-default",
        ),
        pytest.param(
            "export function show(a, b = a) {\n  const n = Number(a);\n  return `n=${n}`;\n}\n",
            id="parameter-referenced-by-later-default",
        ),
    ],
)
def test_parameter_default_values_do_not_keep_template_files_partial(source: str) -> None:
    assert (
        tm.has_bounded_parse_exhaustion(source + _PADDING, lambda: None, file_type="javascript")
        is False
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "export function run(cb) { return cb(`rm ${x}`); }",
            id="uncertain-parameter-call",
        ),
        pytest.param(
            "export function run(cb = getHandler()) { return cb(`rm ${x}`); }",
            id="uncertain-parameter-with-default-call",
        ),
        pytest.param(
            "export function run({handler = getHandler()} = {}) { return handler(`rm ${x}`); }",
            id="uncertain-destructured-parameter-call",
        ),
        pytest.param(
            "function f(x = eval) { return `v=${x}`; }",
            id="shell-call-in-default-value",
        ),
        pytest.param(
            "function f(p = cp.exec(`ls ${dir}`)) { return `v=${p}`; }",
            id="shell-call-expression-in-default-value",
        ),
    ],
)
def test_uncertain_parameter_calls_keep_template_files_partial(source: str) -> None:
    assert (
        tm.has_bounded_parse_exhaustion(source + _PADDING, lambda: None, file_type="javascript")
        is True
    )


def test_ledger_completes_parameter_default_value_template() -> None:
    path = "scripts/check.mjs"
    source = (
        "export function check(value, max = Number.MAX_SAFE_INTEGER) {\n"
        "  const n = Number(value);\n"
        "  return n <= (max ?? Number.MAX_SAFE_INTEGER) ? `ok: ${n}` : `too large: ${n}`;\n"
        "}\n"
        "console.log(check(process.argv[2] ?? '0'));\n" + _PADDING
    )
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: source}}, [tm]
    )
    row = result["inspection_ledger"][0]
    assert row["outcome"] is LedgerOutcome.COMPLETED
    assert "reason_code" not in row


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
        "const reflect = Reflect; const invoke = reflect.apply; "
        "invoke(callback, null, [`${command}`]);",
        "const box = {reflection: Reflect}; "
        "box.reflection.apply(source[key], null, [`${command}`]);",
        "globalThis.Reflect.apply(callback, null, [`${command}`]);",
        "const runner = execute.bind(null); runner(`${command}`);",
        "getRunner()(`${command}`);",
        "const invoke = globalThis[key]; invoke?.(`${command}`);",
        "const 調用 = globalThis[key]; 調用?.(`${command}`);",
        "const [invoke] = [globalThis[key]]; invoke?.(`${command}`);",
        "const {invoke} = {invoke: source[key]}; invoke?.(`${command}`);",
        "const receiver = {invoke: globalThis[key]}; receiver.invoke(`${command}`);",
        "const receiver = {invoke: globalThis.eval}; receiver.invoke(`${command}`);",
        "const receiver = {invoke: tool.exec}; receiver.invoke(`${command}`);",
        "const handlers = [tool.exec]; handlers[0](`${command}`);",
        "const handlers = {nested: {invoke: tool.exec}}; handlers.nested.invoke(`${command}`);",
        "const source = {invoke: tool.exec}; const alias = source.invoke; "
        "const handlers = {invoke: alias}; handlers.invoke(`${command}`);",
        "const source = {invoke: tool.exec}; const handlers = {}; "
        "Object.assign(handlers, {invoke: source.invoke}); handlers.invoke(`${command}`);",
        "const {invoke} = {invoke: tool.exec}; const handlers = {invoke}; "
        "handlers.invoke(`${command}`);",
        "const [invoke] = [tool.exec]; const handlers = {invoke}; handlers.invoke(`${command}`);",
        "const invoke = await tool.exec; const handlers = {invoke}; handlers.invoke(`${command}`);",
        "function* pick() { yield tool.exec; } pick().next().value(`${command}`);",
        "class Handlers { static invoke = tool.exec; } Handlers.invoke(`${command}`);",
        "class Handler extends tool.exec {} new Handler(`${command}`);",
        "for (const invoke of [tool.exec]) { const handlers = {invoke}; "
        "handlers.invoke(`${command}`); }",
        "const handlers = {}; handlers.invoke = tool.exec; handlers.invoke(`${command}`);",
        "const handlers = []; handlers[0] = tool.exec; handlers[0](`${command}`);",
        "const handlers = {nested: {}}; handlers.nested.invoke = tool.exec; "
        "handlers.nested.invoke(`${command}`);",
        "const execute = tool.exec; const handlers = {}; handlers.invoke = execute; "
        "handlers.invoke(`${command}`);",
        "const handlers = {}; Object.assign(handlers, {invoke: tool.exec}); "
        "handlers.invoke(`${command}`);",
        "const handlers = {}; Object.defineProperty(handlers, 'invoke', {value: tool.exec}); "
        "handlers.invoke(`${command}`);",
        "function pick() { return {invoke: tool.exec}; } pick().invoke(`${command}`);",
        "const execute = tool.exec; const pick = () => ({invoke: execute}); "
        "pick().invoke(`${command}`);",
        "const root = globalThis; const receiver = {invoke: root[key]}; receiver.invoke?.(`${command}`);",
        "const root = flag ? globalThis : window; const receiver = {invoke: root[key]}; receiver.invoke(`${command}`);",
        "import {launch} from 'unknown'; launch?.(`${command}`);",
        "import {launch} from 'unknown'; const text = `${command}`; const alias = text; launch?.(alias);",
        "import * as tools from 'unknown'; tools.launch?.(`${command}`);",
        "let invoke; invoke = globalThis[key]; invoke(`${command}`);",
        "function main(invoke) { return invoke(`${command}`); }",
        "format?.bind(null, `hello ${name}`)();",
        "const {render: go = format} = source; go(`hello ${name}`);",
        "function invoke(go = format) { go(`hello ${name}`); }",
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


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("interpreter", ["#!/usr/bin/env node", "#!/usr/bin/node --no-warnings"])
def test_node_hashbang_preserves_nested_template_ownership(interpreter: str, newline: str) -> None:
    source = (
        interpreter
        + newline
        + (
            "const rows = [['a', 1]];\n"
            'console.log(`Rows: ${rows.map(([key, value]) => `${key}=${value}`).join(", ")}`);'
        )
        + _PADDING
    )
    spans = javascript_template_spans(source, lambda: None)
    assert spans is not None
    assert all(source[start] == source[end - 1] == "`" for start, end in zip(*spans, strict=True))
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript") is False


@pytest.mark.parametrize(
    "expression",
    [
        "entry?.label",
        "entry?.['label']",
        "formatter?.(entry)",
        "entry?.children?.[0]?.label",
        "entry?.querySelector('.label').textContent",
        "`${entry?.label}: ${entry?.children?.length}`",
    ],
)
def test_optional_chain_keeps_template_boundaries(expression: str) -> None:
    source = f"const value = {expression}; console.log(`Value: ${{value}}`);" + _PADDING
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript") is False


def test_optional_chain_projection_preserves_literal_and_unicode_offsets() -> None:
    source = (
        "const emoji = '🌍'; const text = 'literal?.value'; const pattern = /\\?\\./;\n"
        "const value = entry?.label; console.log(`literal ?. ${value}`);"
    ) + _PADDING
    spans = javascript_template_spans(source, lambda: None)
    start = source.index("`literal")
    end = source.index("`", start + 1) + 1
    assert spans == ((start,), (end,))


@pytest.mark.parametrize(
    "source",
    [
        "entry?.label = `value ${input}`;",
        "entry?.count++; const value = `${input}`;",
        "new Constructor?.(`${input}`);",
        "entry?.tag`value ${input}`;",
        "const value = entry?.; console.log(`${input}`);",
        "#!/bin/sh\nconst value = `${input}`;",
        " const value = `${input}`;\n#!/usr/bin/env node",
    ],
)
def test_invalid_or_unproven_host_syntax_keeps_inspection_partial(source: str) -> None:
    source += _PADDING
    unresolved: list[int] = []
    assert (
        javascript_template_spans(source, lambda: None, unresolved_executables=unresolved) is None
    )
    # A failed complete host parse cannot prove executable template ownership,
    # even when the ordinary shell fallback would fit within its span limit.
    assert unresolved
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript")


@pytest.mark.parametrize(
    "source",
    [
        "#!/usr/bin/env node\nexec(`${command} -rf /`);",
        "child?.exec(`${command} -rf /`);",
        "eval?.(`${command}`);",
        "const evaluate = eval; evaluate(`${command}`);",
        "const Evaluate = Function; new Evaluate(`${command}`);",
        "new Function(`${command}`);",
        "console.log(`${child?.exec(command)}`);",
        "const child = require('child_process'); child?.exec(`${command}`);",
    ],
)
@pytest.mark.parametrize("suffix", ["", _PADDING], ids=["short", "padded"])
def test_supported_host_syntax_cannot_hide_code_execution(source: str, suffix: str) -> None:
    source += suffix
    assert javascript_template_spans(source, lambda: None) is None
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript") is True


@pytest.mark.parametrize(
    "statement",
    [
        "target[entry?.key] = 1;",
        "target[entry?.key]++;",
        "(entry?.target).value = 1;",
        "new (entry?.Constructor)();",
        "new Constructor(entry?.value);",
        "(entry?.tag)`value`;",
    ],
)
def test_independent_optional_expressions_do_not_invalidate_templates(statement: str) -> None:
    source = statement + "console.log(`Value: ${entry?.value}`);" + _PADDING
    assert js.parse_javascript_source(source, lambda: None) is not None
    if statement == "(entry?.tag)`value`;":
        # Valid syntax does not establish the behavior of an unknown tag.
        assert javascript_template_spans(source, lambda: None) is None
        assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript")
    else:
        assert javascript_template_spans(source, lambda: None) is not None
        assert not tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript")


@pytest.mark.parametrize(
    "statement",
    [
        "eval?.(`${command} -rf /`);",
        "const evaluate = eval; evaluate(`${command} -rf /`);",
        "const Evaluate = Function; new Evaluate(`${command} -rf /`);",
        "const child = require('child_process'); child?.exec(`${command} -rf /`);",
    ],
)
def test_unknown_executable_template_keeps_destructive_operands_partial(statement: str) -> None:
    assert tm.has_bounded_parse_exhaustion(
        statement + _PADDING, lambda: None, file_type="javascript"
    )


def test_metadata_keys_do_not_become_evaluator_references() -> None:
    content = (
        'const metadata = {eval: false, Function: "render", label: "plot"}; '
        "console.log(`${metadata?.label}`);"
    )
    assert javascript_template_spans(content, lambda: None) is not None


def test_optional_chain_cancellation_propagates() -> None:
    class CancelledError(RuntimeError):
        pass

    calls = 0

    def cancel() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise CancelledError

    source = "const value = entry?.label;" * 1000 + "console.log(`${value}`);"
    with pytest.raises(CancelledError):
        javascript_template_spans(source, cancel)


@pytest.mark.parametrize(
    "member",
    [
        'globalThis["eval"]',
        'globalThis?.["eval"]',
        'globalThis["Function"]',
        'globalThis?.["Function"]',
    ],
)
def test_computed_evaluator_aliases_cannot_own_executable_templates(member: str) -> None:
    source = f"const evaluate = {member}; evaluate(`${{command}} -rf /`);" + _PADDING
    assert javascript_template_spans(source, lambda: None) is None
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript")


@pytest.mark.parametrize(
    "statement",
    [
        "const evaluate = globalThis?.['ev' + 'al']; evaluate(`${command} -rf /`);",
        "const key = 'eval'; globalThis?.[key](`${command} -rf /`);",
        "const root = globalThis; const other = root; other[key](`${command} -rf /`);",
        "const evaluate = source[key]; const run = evaluate; run(`${command} -rf /`);",
        "let run; run = source[key]; run.call(null, `${command} -rf /`);",
        "source[key](`${command} -rf /`);",
        "source[key].call(null, `${command} -rf /`);",
        "const invoke = source[key].apply; invoke(null, [`${command} -rf /`]);",
        "source[key].bind(null)(`${command} -rf /`);",
        "const {eval: run} = globalThis; run(`${command} -rf /`);",
        "const {'Function': run} = globalThis; new run(`${command} -rf /`);",
        "const root = globalThis; const {[key]: run} = root; run(`${command} -rf /`);",
        "const run = source[key]; run.call`${command} -rf /`;",
        "source[key].call`${command} -rf /`;",
    ],
)
@pytest.mark.parametrize("suffix", ["", _PADDING], ids=["short", "padded"])
def test_unresolved_computed_evaluators_cannot_own_templates(statement: str, suffix: str) -> None:
    source = statement + suffix
    assert javascript_template_spans(source, lambda: None) is None
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript")


def test_computed_data_reads_do_not_become_executable_aliases() -> None:
    source = "const label = source[key]; console.log(`Value: ${label}`);" + _PADDING
    assert javascript_template_spans(source, lambda: None) is not None


@pytest.mark.parametrize(
    "statement",
    [
        "tool.exec.call(tool, `${command} -rf /`);",
        "tool.exec?.call(tool, `${command} -rf /`);",
        "tool.exec?.apply(tool, [`${command} -rf /`]);",
        "tool.exec?.bind(tool, `${command} -rf /`)();",
        "tool.exec['call']?.(tool, `${command} -rf /`);",
        "tool['exec']?.['apply'](tool, [`${command} -rf /`]);",
        "tool.exec.call?.call(tool.exec, tool, `${command} -rf /`);",
        "const go = tool.exec; go?.(`${command} -rf /`);",
        "const go = exec; const next = go; next?.(`${command} -rf /`);",
        "let go; go = tool.exec; go.call(null, `${command} -rf /`);",
        "const go = tool.exec.bind(tool); go?.(`${command} -rf /`);",
        "const {exec: go} = tool; go?.(`${command} -rf /`);",
        "const {'exec': go = fallback} = tool; go?.(`${command} -rf /`);",
        "function invoke({exec: go}) { go?.(`${command} -rf /`); }",
        "tool.exec.call`${command} -rf /`;",
        "Reflect.apply(tool.exec, tool, [`${command} -rf /`]);",
        "Reflect?.apply(tool.exec, tool, [`${command} -rf /`]);",
        "Reflect.construct(tool.exec, [`${command} -rf /`]);",
        "const invoke = Reflect.apply; invoke?.(tool.exec, tool, [`${command} -rf /`]);",
        "const go = flag ? tool.exec : format; go?.(`${command} -rf /`);",
        "const go = flag ? format : tool.exec; go?.(`${command} -rf /`);",
        "const go = tool.exec || format; go?.(`${command} -rf /`);",
        "const go = flag && tool.exec; go?.(`${command} -rf /`);",
        "const go = (observe(), tool.exec); go?.(`${command} -rf /`);",
        "const first = tool.exec; const go = flag ? first : format; go?.(`${command} -rf /`);",
        "(flag ? tool.exec : format)?.call(tool, `${command} -rf /`);",
        "const first = tool.exec; (flag ? first : format)?.call(tool, `${command} -rf /`);",
        "(selected = tool.exec)?.(`${command} -rf /`);",
        "const first = tool.exec; const invoke = first.call; "
        "invoke.call(first, tool, `${command} -rf /`);",
        "const first = tool.exec; const invoke = first['apply']; "
        "invoke.call(first, tool, [`${command} -rf /`]);",
        "const first = tool.exec; const invoke = first?.bind; "
        "invoke.call(first, tool, `${command} -rf /`)();",
        "const first = tool.exec; const invoke = first.call.apply; "
        "invoke.call(first.call, first, [tool, `${command} -rf /`]);",
        "const first = tool.exec; const invoke = flag ? first.call : format; "
        "invoke.call(first, tool, `${command} -rf /`);",
        "const first = tool.exec; let invoke; invoke = first.call; "
        "invoke.call(first, tool, `${command} -rf /`);",
        "const {render: go = tool.exec} = source; go(`${command} -rf /`);",
        "const first = tool.exec; const {render: go = first} = source; go(`${command} -rf /`);",
        "const first = tool.exec; const {render: go = first.call} = source; "
        "go.call(first, tool, `${command} -rf /`);",
        "const {render: go = flag ? format : tool.exec} = source; go(`${command} -rf /`);",
        "const {nested: {render: go = tool.exec}} = source; go(`${command} -rf /`);",
        "let go; ({render: go = tool.exec} = source); go(`${command} -rf /`);",
        "const [go = tool.exec] = source; go(`${command} -rf /`);",
        "function invoke(go = tool.exec) { go(`${command} -rf /`); }",
        "function invoke({render: go = tool.exec}) { go(`${command} -rf /`); }",
        "function invoke([go = tool.exec]) { go(`${command} -rf /`); }",
    ],
)
@pytest.mark.parametrize("suffix", ["", _PADDING], ids=["short", "padded"])
def test_execution_helpers_and_aliases_keep_dynamic_templates_unverified(
    statement: str, suffix: str
) -> None:
    source = statement + suffix
    assert javascript_template_spans(source, lambda: None) is None
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript")


@pytest.mark.parametrize(
    "statement",
    [
        "format?.call(null, `hello ${name}`);",
        "format?.apply(null, [`hello ${name}`]);",
        "format['call']?.(null, `hello ${name}`);",
        "const {exec: label} = source; console.log(`Value: ${label}`);",
        "const format = flag ? first : second; format?.call(null, `hello ${name}`);",
        "(observe(), format)?.call(null, `hello ${name}`);",
        "const first = format; const invoke = first.call; "
        "invoke.call(first, null, `hello ${name}`);",
        "const first = format; const invoke = first['apply']; "
        "invoke.call(first, null, [`hello ${name}`]);",
        "function pick() { return {label: source[key]}; } console.log(`Value: ${pick().label}`);",
        "function label(values, key) { return values[key].replace(/ +/g, ' '); } "
        "console.log(`Value: ${label(items, key)}`);",
        "function label(values, key) { return values[key].split(':')[0].split(',')[0]; } "
        "console.log(`Value: ${label(items, key)}`);",
    ],
)
@pytest.mark.parametrize("suffix", ["", _PADDING], ids=["short", "padded"])
def test_data_reads_and_ordinary_function_helpers_retain_template_ownership(
    statement: str, suffix: str
) -> None:
    source = statement + suffix
    assert javascript_template_spans(source, lambda: None) is not None
    assert not tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript")


@pytest.mark.parametrize("extracted", [False, True])
def test_benign_function_helper_chain_has_bounded_traversal(extracted: bool) -> None:
    helper = "format" + ".call" * 48
    source = (f"const invoke = {helper}; invoke" if extracted else helper) + "(`hello ${name}`);"
    checks = 0

    def check_runtime() -> None:
        nonlocal checks
        checks += 1
        assert checks <= 1000

    assert javascript_template_spans(source, check_runtime) is not None


@pytest.mark.parametrize(
    "source",
    [
        "const config = {exec: 'label'}; console.log(config?.exec, `hello ${name}`);",
        "const label = source.exec; console.log(`Value: ${label}`);",
        "const label = flag ? source.exec : 'label'; console.log(`Value: ${label}`);",
        "const {render: label = tool.exec} = source; console.log(`Value: ${label}`);",
        "const first = tool.exec; const label = first.call; console.log(`Value: ${label}`);",
        "const handlers = {invoke: tool.exec}; console.log(`Value: ${name}`);",
        "const handlers = [tool.exec]; console.log(`Value: ${name}`);",
    ],
)
def test_executor_member_reads_keep_existing_conservative_coverage(source: str) -> None:
    # The syntax tree does not prove that an executor-named member is data-only.
    # Keep the existing file-local refusal rather than infer arbitrary value flow.
    unresolved: list[int] = []
    assert (
        javascript_template_spans(source, lambda: None, unresolved_executables=unresolved) is None
    )
    assert unresolved
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript")


@pytest.mark.parametrize(
    "statement",
    [
        "const match = /^(\\d+)(?:\\.(\\d+))?$/.exec(text);",
        "const match = /exec/.exec(text); const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);",
        "const NUMBER = /^[0-9]+$/; const valid = value => NUMBER.test(value); const match = /[0-9]+/.exec(text);",
        "import {posix} from 'node:path'; const path = posix.join(posix.dirname(source), `${name}.json`);",
        "import {win32 as p} from 'node:path'; const path = p.join(`${name}`, 'file');",
        "import {join as makePath} from 'path'; const path = makePath(`${name}`, 'file');",
        "import {join as run} from 'node:path'; const path = run(`${name}`, 'file');",
        "function hash(values) { const r = (x, n) => (x >>> n) | (x << (32 - n)); return r(values[0], 7); } function other(r) { return r; }",
        "function hash(values) { const run = x => x * 2; return run(values[0]); } function other(r) { return r; }",
        "function first(values) { const r = x => x + 1; return r(values[0]); } function second(r) { return r; }",
    ],
)
@pytest.mark.parametrize("suffix", ["", _PADDING], ids=["short", "padded"])
def test_proven_native_and_local_numeric_calls_retain_templates(
    statement: str, suffix: str
) -> None:
    source = statement + " console.log(`Value: ${name}`);" + suffix
    assert javascript_template_spans(source, lambda: None) is not None
    assert not tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript")


@pytest.mark.parametrize(
    "statement",
    [
        "RegExp.prototype.exec = executor; /x/.exec(text);",
        "function mutate(x, f) { x['__pro' + 'to__']['e' + 'xec'] = f; } mutate(/x/, handler); /x/.exec(`${command}`);",
        "mutate(/x/, handler); /x/.exec(`${command}`);",
        "const NUMBER = /x/; mutate(NUMBER, handler); /x/.exec(`${command}`);",
        "const NUMBER = /x/; const alias = NUMBER; /x/.exec(`${command}`);",
        "const NUMBER = /x/; function use(NUMBER) { return NUMBER.test(value); } /x/.exec(`${command}`);",
        "const NUMBER = /x/; NUMBER.test = handler; /x/.exec(`${command}`);",
        "const NUMBER = /x/; const method = NUMBER.test; method(value); /x/.exec(`${command}`);",
        "export const NUMBER = /x/; /x/.exec(`${command}`);",
        "import {join as eval} from 'node:path'; eval(`${command}`);",
        "import {posix as p} from 'node:path'; const load = require; const other = load('node:path'); other.posix.join = handler; p.join(`${command}`);",
        "import {posix as p} from 'node:path'; const other = process.getBuiltinModule('node:path'); other.posix.join = handler; p.join(`${command}`);",
        "import {posix as p} from 'node:path'; import other from 'node:path/posix'; other.join = handler; p.join(`${command}`);",
        "import {join as Function} from 'node:path'; Function(`${command}`);",
        "const prototype = RegExp.prototype; prototype[key] = executor; /x/.exec(text);",
        "Object.getPrototypeOf(/x/)[key] = executor; /x/.exec(text);",
        "Object.defineProperty(Object.getPrototypeOf(/x/), key, {value: executor}); /x/.exec(text);",
        "const expression = /x/; expression[key] = executor; /x/.exec(text);",
        "(/x/)[key][other] = executor; /x/.exec(text);",
        "(/x/).constructor.prototype[key] = executor; /x/.exec(text);",
        "this['Reg' + 'Exp'].prototype[key] = executor; /x/.exec(text);",
        "const Object = tools; Object.prototype.hasOwnProperty.call(value, key); /x/.exec(text);",
        "const method = /x/.exec; method(`${command}`);",
        "tool.exec(`${command}`); /x/.exec(text);",
        "import {posix} from 'node:path'; posix.join = executor; posix.join(`${command}`);",
        "import {posix} from 'node:path'; const alias = posix; alias.join = executor; posix.join(`${command}`);",
        "import {posix} from 'node:path'; mutate(posix); posix.join(`${command}`);",
        "import {posix} from 'node:path'; function use(posix) { return posix.join(`${command}`); }",
        "import {posix} from 'node:path'; const method = posix.join; method(`${command}`);",
        "import {join} from 'node:path'; function use(join) { return join(`${command}`); }",
        "import {join} from 'unknown'; join(`${command}`);",
        "function hash(r) { return r(`${command}`); }",
        "function hash() { let r = x => x + 1; r = source[key]; return r(`${command}`); }",
        "function hash() { const r = x => x + 1; function use(r) { return r(`${command}`); } return r(1); }",
        "function hash() { const r = x => x + 1; const alias = r; return alias(`${command}`); } function other(r) { return r; }",
        "const r = x => x + 1; r = source[key]; r(`${command}`);",
        "const r = x => executor(x); r(`${command}`); function other(r) { return r; }",
        "const r = x => object[key]; r(`${command}`); function other(r) { return r; }",
        "const run = x => x + 1; run.call(executor, `${command}`);",
        "const eval = x => x + 1; eval(`${command}`);",
        "const run = x => x + 1; with (tools) { run(`${command}`); }",
        "import {join as run} from 'node:path'; const alias = run; alias(`${command}`);",
        "import {posix as p, posix as q} from 'node:path'; q.join = executor; p.join(`${command}`);",
        "import {posix as p} from 'node:path'; import {posix as q} from 'node:path'; q.join = executor; p.join(`${command}`);",
        "import path, {posix as p} from 'node:path'; path.posix.join = executor; p.join(`${command}`);",
        "import {posix as p} from 'node:path'; const other = require('node:path'); other.posix.join = executor; p.join(`${command}`);",
        "import {posix as p} from 'node:path'; const other = await import('node:path'); other.posix.join = executor; p.join(`${command}`);",
        "import {join} from 'node:path'; const other = require('node:path'); other.join = executor; syncBuiltinESMExports(); join(`${command}`);",
        "import {posix} from 'node:path'; with (tools) { posix.join(`${command}`); }",
    ],
)
def test_unproved_native_or_local_callable_overrides_stay_partial(statement: str) -> None:
    source = statement + " console.log(`Value: ${name}`);"
    assert javascript_template_spans(source, lambda: None) is None
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript")


def test_known_native_callee_cannot_hide_security_findings() -> None:
    source = (
        'const match = /value/.exec(text); console.log(`Value: ${name}`); tool.exec("rm -rf /");'
    )
    assert javascript_template_spans(source, lambda: None) is None
    assert any(
        finding.rule_id == "TM1" for finding in tm.analyze(source, "display.js", "javascript")
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_quoted_backtick_values_cannot_consume_later_host_code(newline: str) -> None:
    source = newline.join(
        [
            "const values = [['`first` before', '`first` after']];",
            "// " + ("padding " * 1000),
            "const other = ['`second` before', '`second` after'];",
        ]
    )
    assert not tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript")


@pytest.mark.parametrize(
    "suffix",
    [
        "exec(`${tool} -rf /`);",
        "new Function(`${code}`);",
        "import {exec} from 'node:child_process'; exec(`${command}`);",
    ],
)
def test_completed_quoted_value_never_clears_later_runtime_uncertainty(suffix: str) -> None:
    source = "const values = ['`name` before', '`name` after'];\n" + suffix
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript")


def test_regex_refinement_preserves_neighbor_raw_high_commands() -> None:
    source = (
        "const values = ['`name` before', '`name` after'];\n"
        "import {execSync} from 'node:child_process'; execSync('rm -rf /');\n"
    )
    assert any(
        finding.rule_id == "TM1" for finding in tm.analyze(source, "control.mjs", "javascript")
    )


@pytest.mark.parametrize(
    "source",
    [
        "const value = '`name`'; eval(`${value}${payload}`);",
        "let value = '`name`'; value += `${payload}`; Function(value)();",
        "const values = ['`name`']; eval(`${values[0]}${payload}`);",
        "import {exec as launch} from 'node:child_process'; "
        "const value = '`name`'; launch(`${value}${payload}`);",
    ],
)
def test_regex_refinement_preserves_established_indirect_runtime_diagnostics(
    source: str, monkeypatch
) -> None:
    with monkeypatch.context() as baseline:
        baseline.setattr(tm._JavaScriptTemplateOwnership, "regex_literal_end", lambda *args: None)
        before = tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript")
    after = tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript")
    assert before is not None
    assert before.cause is tm.ParserCause.RUNTIME_DEPENDENT
    assert after == before


@pytest.mark.parametrize("value", ["`$TOOL -rf /`", "$(" + "a" * 5000])
def test_ordinary_quoted_data_keeps_its_existing_complete_result(value, monkeypatch) -> None:
    source = "const values = ['`name` before', '`name` after'];\nconst other = '" + value + "';"
    with monkeypatch.context() as baseline:
        baseline.setattr(tm._JavaScriptTemplateOwnership, "regex_literal_end", lambda *args: None)
        before = tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript")
    after = tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript")
    assert before is None
    assert after is None


def test_plain_quoted_files_keep_no_backtick_fast_path(monkeypatch) -> None:
    from skillspector import javascript_tokens

    def must_not_parse(*args):
        raise AssertionError("no-backtick source should not require a native parse")

    monkeypatch.setattr(javascript_tokens, "parse_javascript_source", must_not_parse)
    regexes: list[tuple[int, int]] = []
    assert (
        javascript_template_spans("const label = 'ordinary';", lambda: None, regex_literals=regexes)
        is None
    )
    assert regexes == []


_JS_REGEX_BOUNDARY_SOURCE = (
    "const text = 'safe';\n"
    "const captured = text.match(/read `([^`]+policy\\.md)` before review/);\n"
    + "// "
    + "padding " * 1024
    + "\n"
    + "for (const row of [['label', '`render_proof` passed']]) { inspect(row); }\n"
)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_native_regex_origin_does_not_borrow_a_distant_string_boundary(newline, monkeypatch):
    source = _JS_REGEX_BOUNDARY_SOURCE.replace("\n", newline)
    with monkeypatch.context() as baseline:
        baseline.setattr(tm._JavaScriptTemplateOwnership, "regex_literal_end", lambda *args: None)
        before = tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript")
    assert before is not None
    assert before.cause is tm.ParserCause.BOUNDARY_UNRESOLVED
    assert tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript") is None
    owner = tm._JavaScriptTemplateOwnership(source, lambda: None)
    start = source.index("/read")
    end = source.index("/);", start) + 1
    assert owner.regex_literal_end(source.index("`", start)) == end
    assert owner.regex_literal_end(end) is None


@pytest.mark.parametrize(
    "source",
    [
        "eval(/read `([^`]+policy\\.md)` before review/);",
        "new Function(/read `([^`]+policy\\.md)` before review/);",
        "launch(/read `([^`]+policy\\.md)` before review/);",
        "const rule = /read `([^`]+policy\\.md)` before review/; eval(rule.source);",
        "const rule = /read `([^`]+policy\\.md)` before review/; eval(rule['source']);",
    ],
)
def test_regex_unproved_consumers_do_not_receive_native_pattern_boundaries(source, monkeypatch):
    regexes = []
    javascript_template_spans(source, lambda: None, regex_literals=regexes)
    assert regexes == []
    with monkeypatch.context() as baseline:
        baseline.setattr(tm._JavaScriptTemplateOwnership, "regex_literal_end", lambda *args: None)
        before = tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript")
    after = tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript")
    assert after == before


def test_regex_origin_keeps_later_executable_limits_and_raw_findings():
    runtime = _JS_REGEX_BOUNDARY_SOURCE + "exec(`${tool} -rf /`);"
    assert tm.has_bounded_parse_exhaustion(runtime, lambda: None, file_type="javascript")
    high = _JS_REGEX_BOUNDARY_SOURCE + "exec('rm -rf /');"
    assert any(f.rule_id == "TM1" for f in tm.analyze(high, "example.mjs", "javascript"))


def test_regex_origin_keeps_later_overbudget_host_word(monkeypatch):
    source = _JS_REGEX_BOUNDARY_SOURCE + "r" * (tm._SHELL_COMMAND_WORD_CHARS + 1) + ";\n"
    with monkeypatch.context() as baseline:
        baseline.setattr(tm._JavaScriptTemplateOwnership, "regex_literal_end", lambda *args: None)
        before = tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript")
    after = tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript")
    assert before is not None
    assert after is not None
    assert after.cause is tm.ParserCause.RESOURCE_LIMIT
    assert after.start >= len(_JS_REGEX_BOUNDARY_SOURCE)


@pytest.mark.parametrize(
    "source",
    [
        "import {exec} from 'node:child_process'; "
        "const receiver={match(pattern){exec(String(pattern).slice(2,-1));}}; "
        "receiver.match(/`$TOOL -rf \\//);",
        "import cp from 'node:child_process'; cp.exec(/`$TOOL -rf \\//);",
        "import {exec as launch} from 'node:child_process'; "
        "const runner={exec:launch}; runner.exec(/`$TOOL -rf \\//);",
        "const receiver={match(pattern){consume(pattern);}}; receiver.match(/`name/);",
        "text.match = custom; text.match(/`name/);",
        "text['match'] = custom; text.match(/`name/);",
        "String.prototype.match = custom; text.match(/`name/);",
        "RegExp.prototype.exec = custom; text.match(/`name/);",
        "const rule = /`name/; use(rule['so'+'urce']);",
    ],
)
def test_regex_execution_and_override_barriers_keep_existing_incompleteness(source, monkeypatch):
    regexes = []
    javascript_template_spans(source, lambda: None, regex_literals=regexes)
    assert regexes == []
    with monkeypatch.context() as baseline:
        baseline.setattr(tm._JavaScriptTemplateOwnership, "regex_literal_end", lambda *args: None)
        before = tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript")
    after = tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript")
    assert after == before


@pytest.mark.parametrize(
    "prefix,receiver",
    [
        ("import value from './runtime.mjs';", "value"),
        ("const value = getValue();", "value"),
        ("let value = 'safe'; value = getValue();", "value"),
        ("const value = 'safe'; function shadow(value) {}", "value"),
        ("function check(value) {", "value"),
    ],
)
def test_unknown_or_shadowed_string_receivers_keep_the_regex_fallback(prefix, receiver):
    source = prefix + receiver + ".match(/`$TOOL -rf \\//);"
    if prefix.startswith("function"):
        source += "}"
    regexes = []
    javascript_template_spans(source, lambda: None, regex_literals=regexes)
    assert regexes == []


def test_direct_primitive_string_can_prove_its_matching_regex_boundaries():
    source = "'safe'.match(/`label/);"
    regexes = []
    javascript_template_spans(source, lambda: None, regex_literals=regexes)
    start = source.index("/`label")
    assert regexes == [(start, start + len("/`label/"))]


def test_with_scope_cannot_borrow_a_top_level_primitive_string_binding(monkeypatch):
    source = "const text='safe'; with (runtimeScope) { text.match(/`$TOOL -rf \\//); }"
    regexes = []
    javascript_template_spans(source, lambda: None, regex_literals=regexes)
    assert regexes == []
    with monkeypatch.context() as baseline:
        baseline.setattr(tm._JavaScriptTemplateOwnership, "regex_literal_end", lambda *args: None)
        before = tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript")
    assert before is not None
    assert tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript") == before


def test_quoted_backtick_data_keeps_the_no_regex_execution_fast_path(monkeypatch):
    from skillspector import javascript_tokens

    def must_not_analyze_execution(*args):
        raise AssertionError("ordinary quoted data needs no regex execution proof")

    monkeypatch.setattr(javascript_tokens, "_has_shell_execution_call", must_not_analyze_execution)
    regexes = []
    assert (
        javascript_template_spans("const label='`name`';", lambda: None, regex_literals=regexes)
        is None
    )
    assert regexes == []


def test_native_matching_pattern_dollars_are_data_not_runtime_commands(monkeypatch):
    source = "'safe'.match(/`$TOOL -rf \\//);"
    with monkeypatch.context() as baseline:
        baseline.setattr(tm._JavaScriptTemplateOwnership, "regex_literal_end", lambda *args: None)
        before = tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript")
    assert before is not None
    assert before.cause is tm.ParserCause.RUNTIME_DEPENDENT
    assert tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript") is None


@pytest.mark.parametrize("comment", ["/* pattern */ ", "// pattern\n", "/* first */ /* second */ "])
def test_first_regex_argument_can_follow_native_comments(comment, monkeypatch):
    source = _JS_REGEX_BOUNDARY_SOURCE.replace("text.match(", "text.match(" + comment)
    with monkeypatch.context() as baseline:
        baseline.setattr(tm._JavaScriptTemplateOwnership, "regex_literal_end", lambda *args: None)
        before = tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript")
    assert before is not None
    assert tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript") is None


def test_comments_do_not_make_a_second_argument_a_matching_pattern(monkeypatch):
    source = _JS_REGEX_BOUNDARY_SOURCE.replace(
        "text.match(", "text.match(unresolved, /* pattern */ "
    )
    with monkeypatch.context() as baseline:
        baseline.setattr(tm._JavaScriptTemplateOwnership, "regex_literal_end", lambda *args: None)
        before = tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript")
    assert before is not None
    assert tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript") == before


def test_native_pattern_boundaries_cannot_hide_a_later_dynamic_consumer():
    source = "'safe'.match(/`$TOOL -rf \\//);\nexec(`${tool} -rf /`);"
    diagnostic = tm.bounded_parse_diagnostic(source, lambda: None, file_type="javascript")
    assert diagnostic is not None
    assert diagnostic.cause is tm.ParserCause.RUNTIME_DEPENDENT
    assert any(
        f.rule_id == "TM1"
        for f in tm.analyze(source + "exec('rm -rf /');", "control.mjs", "javascript")
    )
