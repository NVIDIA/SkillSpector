# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JavaScript template literals must not be mistaken for shell backticks."""

from __future__ import annotations

import pytest

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
def test_invalid_or_unproven_host_syntax_keeps_conservative_fallback(source: str) -> None:
    source += _PADDING
    assert javascript_template_spans(source, lambda: None) is None
    # This hook reports bounded shell inspection, not JavaScript validity.
    # Refusing ownership must leave the ordinary conservative scan unchanged.
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript") == (
        tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="text")
    )


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
def test_supported_host_syntax_cannot_hide_code_execution(source: str) -> None:
    source += _PADDING
    assert javascript_template_spans(source, lambda: None) is None
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="javascript") == (
        tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="text")
    )


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
