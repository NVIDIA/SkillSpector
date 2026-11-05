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


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            'import * as cp from "node:child_\\x70rocess";\ncp["exec"](`rm -rf / ${tool}`);',
            id="hex-escaped-module",
        ),
        pytest.param(
            'import * as cp from "node:child_\\u0070rocess";\ncp["execSync"](`rm -rf / ${tool}`);',
            id="unicode-escaped-module",
        ),
        pytest.param(
            'import cp from "node:child_\\u{70}rocess";\ncp.spawn(`rm -rf / ${site}`);',
            id="braced-unicode-escaped-module",
        ),
        pytest.param(
            'const cp = require("child_\\x70rocess");\ncp.exec(`rm -rf / ${site}`);',
            id="escaped-require-module",
        ),
        pytest.param(
            'import cp from "node:child_\\x70rocess";\ncp[`exec`](`rm -rf / ${tool}`);',
            id="escaped-module-direct-call",
        ),
    ],
)
def test_escaped_module_spellings_keep_the_execution_surface(source: str) -> None:
    assert typescript_template_spans(source, lambda: None) is None
    assert (
        tm.has_bounded_parse_exhaustion(source + _PADDING, lambda: None, file_type="typescript")
        is True
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param('const cp = load();\ncp["exec"](`rm -rf / ${tool}`);', id="string-exec"),
        pytest.param(
            "const cp = load();\ncp[`spawnSync`](`rm -rf / ${tool}`);", id="template-exec"
        ),
        pytest.param('const cp = load();\ncp["EXEC"](`rm -rf / ${tool}`);', id="upper-case-exec"),
    ],
)
def test_computed_execution_members_keep_the_execution_surface(source: str) -> None:
    assert typescript_template_spans(source, lambda: None) is None
    assert (
        tm.has_bounded_parse_exhaustion(source + _PADDING, lambda: None, file_type="typescript")
        is True
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            'import cp from "node:child_\\x7";\ncp.exec(`rm -rf / ${tool}`);', id="truncated-hex"
        ),
        pytest.param(
            'import cp from "node:child_\\xZZrocess";\ncp.exec(`rm -rf / ${tool}`);',
            id="bad-hex",
        ),
        pytest.param(
            'import cp from "node:child_\\u{d800}rocess";\ncp.spawn(`rm -rf / ${tool}`);',
            id="surrogate",
        ),
        pytest.param(
            'import cp from "node:child_\\u{110000}rocess";\ncp.execSync(`rm -rf / ${tool}`);',
            id="out-of-range",
        ),
    ],
)
def test_an_undecodable_specifier_with_a_shell_call_fails_closed(source: str) -> None:
    # Decoding fails, so the specifier cannot be compared with the allowlist. A shell
    # execution call in the same file keeps the conservative read.
    assert typescript_template_spans(source, lambda: None) is None
    assert (
        tm.has_bounded_parse_exhaustion(source + _PADDING, lambda: None, file_type="typescript")
        is True
    )


def test_an_undecodable_benign_string_does_not_claim_ownership() -> None:
    # Nothing destructive is reachable, so no conservative evidence is due; an undecodable
    # escape must also not be read as a shell surface.
    source = 'const label = "node:child_\\x7"; const value = `${label}`;'

    assert typescript_template_spans(source, lambda: None) is not None
    assert (
        tm.has_bounded_parse_exhaustion(source + _PADDING, lambda: None, file_type="typescript")
        is False
    )


def test_simple_escapes_resolve_before_a_member_is_matched() -> None:
    # The computed member resolves to a tab, which is not a shell call; the raw spelling
    # ``\\t`` tells a substring matcher nothing on its own.
    source = 'const cp = load(); cp["\\t"](`rm -rf / ${x}`);'

    assert typescript_template_spans(source, lambda: None) is not None


def test_decoded_computed_member_on_a_non_module_receiver_is_not_a_shell_call() -> None:
    source = 'const labels = { kind: "x" };\nconst kind = labels["kind"];\nconst value = `${kind}`;'

    assert typescript_template_spans(source, lambda: None) is not None


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            'import * as cp from "node:child_\\x70rocess";\ncp["exec"](`rm -rf / ${tool}`);',
            id="hex-escaped-module",
        ),
        pytest.param(
            "const cp = load();\ncp[`exec`](`rm -rf / ${tool}`);", id="template-exec-member"
        ),
    ],
)
def test_ledger_keeps_partial_status_for_escaped_execution_surfaces(source: str) -> None:
    path = "scripts/run.ts"
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: source + _PADDING}}, [tm]
    )

    row = result["inspection_ledger"][0]
    assert row["outcome"] is LedgerOutcome.PARTIAL
    assert row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


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


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("const code = new Function(`rm -rf / ${payload}`)();", id="new-operator"),
        pytest.param("const code = Function(`return ${payload}`)();", id="bare-call"),
    ],
)
def test_the_function_constructor_stays_a_shell_execution_surface(source: str) -> None:
    # The shared call list is compared casefolded, so ``function`` is excluded
    # by exact spelling; ``Function``, which builds code the template supplies,
    # must still be rejected.
    assert typescript_template_spans(source, lambda: None) is None
    assert (
        tm.has_bounded_parse_exhaustion(source + _PADDING, lambda: None, file_type="typescript")
        is True
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "const cp = await import('node:child_process');\nconst label = `${command}`;",
            id="static-import",
        ),
        pytest.param(
            "const cp = await import(`node:child_process`);\nconst label = `${command}`;",
            id="template-import",
        ),
    ],
)
def test_module_loaders_checked_through_the_call_parens_keep_ownership(
    source: str,
) -> None:
    # ``require(...)`` and ``import(...)`` are calls, so the module string sits
    # after ``(``; the callee name must survive that token for the check.
    assert typescript_template_spans(source, lambda: None) is None


def test_a_loaded_execution_module_keeps_its_samples_unowned() -> None:
    source = "const launch = require('execa');\nlaunch(`${command.name}`);"

    assert typescript_template_spans(source, lambda: None) is None
    assert (
        tm.has_bounded_parse_exhaustion(source + _PADDING, lambda: None, file_type="typescript")
        is True
    )


def test_shell_execution_module_loaded_without_reaching_a_sink_is_not_claimed() -> None:
    # The module string alone is not destructive; a member call is. Loading
    # ``fs`` is plain code, so the template stays owned.
    source = "const fs = require('node:fs');\nconst label = `${file}`;"

    assert typescript_template_spans(source, lambda: None) is not None


def test_postfix_increment_before_a_slash_does_not_open_a_regex() -> None:
    # A collided ``+`` used to leave ``+`` as the last token, so
    # ``/ eval(payload) /`` read as a regex and hid the ``eval`` guard.
    source = "const n = counter++ / eval(payload) / 2;\nconst label = `${item.name}`;"

    assert typescript_template_spans(source, lambda: None) is None
    assert (
        tm.has_bounded_parse_exhaustion(source + _PADDING, lambda: None, file_type="typescript")
        is True
    )


def test_an_incomplete_block_does_not_claim_ownership() -> None:
    # The host block is never closed, so the source is incomplete and the
    # conservative shell reading must survive.
    source = "if (ok) { const label = `${item.name}`;"

    assert typescript_template_spans(source, lambda: None) is None
    assert (
        tm.has_bounded_parse_exhaustion(source + _PADDING, lambda: None, file_type="typescript")
        is True
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("const label = `${x}`};\n", id="extra-close-brace"),
        pytest.param("const label = `${fn([)]}`;", id="crossed-square-parens"),
    ],
)
def test_delimiters_must_nest_in_the_matching_order(source: str) -> None:
    assert typescript_template_spans(source, lambda: None) is None


def test_closing_blocks_keep_later_templates_owned() -> None:
    # Balance tracking must survive ordinary closed blocks, not only reject
    # incomplete ones.
    source = "if (ok) { render(); }\nfor (const x of xs) { use(x); }\nconst label = `${item.name}`;"

    assert typescript_template_spans(source, lambda: None) is not None
