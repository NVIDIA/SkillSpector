# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JavaScript template literals must not be mistaken for shell backticks."""

from __future__ import annotations

import pytest

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
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
