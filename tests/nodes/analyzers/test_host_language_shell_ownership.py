# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proven host-language syntax must not exhaust the bounded shell parser.

Python string and comment tokens own their bytes, and Perl ``eval BLOCK`` is
exception handling rather than a string reparse. Neither may turn ordinary
host source into a ``static_parse_limit`` result, while genuinely unresolved
shell text keeps its conservative partial outcome. Verified Node assertion
messages may own template delimiters without hiding executable string contents.
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

from skillspector import python_tokens
from skillspector.graph import graph
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner

# More than the 4096-character command-word budget of ordinary code after
# the construct under test, with no quotes, backticks or comments of its own.
_TAIL = "".join(f"value_{index} = {index}\n" for index in range(600))
# A view-level prefilter only considers runtime-selected command operands
# when recursive/force option text and a path occur somewhere in the source.
_OPTION_TEXT = 'BUILD_ARGS = ["--recursive", "--force", "/tmp/build"]\n'
# Read as shell text, the fence backticks open a command substitution that the
# tail exhausts. Only proven Python token ownership makes them literal.
_FENCE_SOURCE = 'FENCE = "\\n```\\n"\n' + _TAIL


@pytest.fixture(autouse=True)
def _fresh_ownership_memo() -> Iterator[None]:
    # Ownership proofs are memoized across scans. Each test starts without
    # them, so its parse, warning, and runtime-check assertions are its own.
    python_tokens._PYTHON_LITERAL_SPANS_CACHE.clear()
    yield


def _python(content: str, *, complete_context: bool = True) -> bool:
    return tm.has_bounded_parse_exhaustion(
        content, lambda: None, file_type="python", complete_context=complete_context
    )


def _perl(content: str, *, complete_context: bool = True) -> bool:
    return tm.has_bounded_parse_exhaustion(
        content, lambda: None, file_type="perl", complete_context=complete_context
    )


def _shell(content: str) -> bool:
    return tm.has_bounded_parse_exhaustion(content, lambda: None, file_type="shell")


def _ledger(path: str, content: str) -> dict:
    return static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: content}}, [tm]
    )


# --- Perl eval BLOCK -------------------------------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        'eval { require $module; 1 } or die "load failed: $@";',
        "eval { require $module };",
        "eval { $handler->(); 1 };",
        "eval\n{ require $module; 1 };",
        '{ local $^W = 0; eval { require $module; 1 } or die "load failed: $@"; }',
        'if (defined &Plugin::init) { eval { Plugin::init(); 1 } or warn "init: $@"; }',
    ],
)
@pytest.mark.parametrize("complete_context", [True, False], ids=["complete", "fragment"])
def test_perl_eval_block_is_not_a_shell_string_eval(statement: str, complete_context: bool) -> None:
    content = "use strict;\nuse warnings;\n" + statement + "\n"

    assert _perl(content, complete_context=complete_context) is False


def test_perl_eval_block_clause_is_not_a_command_string() -> None:
    content = "eval { require $module; 1 };\n"

    assert tm._command_string_from_clause(content, 0, lambda: None, perl_eval_blocks=True) == (
        False,
        None,
    )
    # Shell has no block form: the same bytes in a shell file stay partial.
    assert tm._command_string_from_clause(content, 0, lambda: None) == (True, None)
    assert _shell(content) is True


@pytest.mark.parametrize(
    "statement",
    [
        "eval $code;",
        'eval "$code";',
        "eval qq{$code};",
        'eval "require $module; 1" or die;',
        "EVAL { $code };",
        "eval { require $module; 1 } or die; eval $code;",
    ],
)
def test_perl_string_eval_keeps_conservative_parse_status(statement: str) -> None:
    assert _perl(statement + "\n") is True


@pytest.mark.parametrize(
    "statement",
    [
        "eval { my $out = `$TOOL -rf /`; 1 };",
        'eval { qx(sh -c "$cmd") };',
        "eval { rm -rf {/,{1..2}} };",
    ],
)
def test_commands_inside_perl_eval_block_remain_scanned(statement: str) -> None:
    assert _perl(statement + "\n") is True


def test_destructive_command_inside_perl_eval_block_keeps_tm1() -> None:
    result = _ledger("scripts/helper.pl", 'eval { system("rm -rf /") } or warn "failed: $@";\n')

    assert any(finding.rule_id == "TM1" for finding in result["findings"])


# --- Python string and comment ownership -----------------------------------


_BENIGN_PYTHON = [
    pytest.param('FENCE = "\\n```\\n"\n', id="fence-in-double-quoted-string"),
    pytest.param(
        'def render(lang):\n    return [f"```{lang}", "body", "```"]\n',
        id="fence-in-f-string",
    ),
    pytest.param(
        'def quote():\n    """Wrap inline code in a ` character pair."""\n    return ""\n',
        id="docstring-unpaired-backtick",
    ),
    pytest.param(
        "def token():\n    '''Return the user's ``token``.'''\n    return None\n",
        id="docstring-apostrophe-and-double-backticks",
    ),
    pytest.param("# The prefix doesn't include the array name.\n", id="comment-apostrophe"),
    pytest.param(
        "# Patch the shared `client` module so helper()'s requests are intercepted.\n",
        id="comment-backtick-and-apostrophe",
    ),
    pytest.param('assert "see `latest" not in "text"\n', id="string-single-backtick"),
    pytest.param(
        'import re\nWORD = re.compile(r"(?:^|[\\s;&|`(/])(tool)(?=\\s|$)")\n',
        id="raw-regex-character-class",
    ),
    pytest.param('print("done", end="`")\n', id="keyword-argument-backtick"),
    pytest.param('PARTS = ("first `part"\n         "second part")\n', id="implicit-concatenation"),
    pytest.param(
        "import signal\n\n\ndef arm(timeout):\n"
        "    if (\n        timeout > 0\n    ):\n        signal.alarm(timeout)\n",
        id="python-name-matching-shell-wrapper",
    ),
    pytest.param(
        _OPTION_TEXT + '# Pass $0 = the launcher\'s path so the `dirname "$0"`\n'
        "# branch runs (and `command -v` is skipped).\n",
        id="comment-runtime-parameter",
    ),
    # Markdown inline code and parameter names in report strings. Read past
    # the closing quote, they would take the following code as operands.
    pytest.param(
        _OPTION_TEXT + "def report(path, detail):\n"
        '    return [f"Inspected `{path}` and found {detail}."]\n',
        id="f-string-inline-code",
    ),
    pytest.param(
        _OPTION_TEXT
        + "def lines(payload):\n    return [\n"
        + "".join(f"        f\"- Field {i}: `{{payload['field_{i}']}}`\",\n" for i in range(8))
        + "    ]\n",
        id="f-string-inline-code-list",
    ),
    pytest.param(
        _OPTION_TEXT + "def message(tool):\n    return (\n"
        '        f"{tool} was not found at "\n'
        '        f"$TOOL_ROOT, $UPSTREAM_ROOT/{tool}, "\n'
        '        f"or $HOME/.cache/upstreams/{tool} "\n'
        '        "or elsewhere"\n    )\n',
        id="concatenated-f-strings-naming-parameters",
    ),
    pytest.param(
        _OPTION_TEXT + 'def run():\n    """Run `$TOOL` once, then report."""\n',
        id="docstring-runtime-parameter",
    ),
]


@pytest.mark.parametrize("source", _BENIGN_PYTHON)
def test_python_literals_and_comments_own_their_bytes(source: str) -> None:
    content = source + _TAIL

    assert _python(content) is False
    # The same bytes read as shell text remain conservative.
    assert _shell(content) is True


@pytest.mark.parametrize("newline", ["\r\n"], ids=["crlf"])
def test_python_ownership_preserves_crlf_offsets(newline: str) -> None:
    content = ('"""Module `doc."""\nFENCE = "\\n```\\n"  # it\'s fine\n' + _TAIL).replace(
        "\n", newline
    )

    assert _python(content) is False


def test_python_ownership_without_trailing_newline_or_with_non_ascii_text() -> None:
    assert _python('FENCE = "\\n```\\n"\n' + _TAIL.rstrip("\n")) is False
    assert _python('LABEL = "caf\u00e9 \\n```\\n"  # na\u00efve\n' + _TAIL) is False


def test_python_literal_spans_are_exact_source_offsets() -> None:
    content = (
        "#!/usr/bin/env python3\r\n"
        "x = f\"a{y!r:>{w}}b\" + rb'\\d'  # c\u00e9\r\n"
        "z = f\"{f'{q}'}\"\r\n"
        '"""doc\r\nstring"""\r\n'
    )

    spans = tm._python_literal_spans(content, lambda: None)

    assert spans is not None
    assert [content[start:end] for start, end in zip(*spans, strict=True)] == [
        "#!/usr/bin/env python3",
        'f"a{y!r:>{w}}b"',
        "rb'\\d'",
        "# c\u00e9",
        "f\"{f'{q}'}\"",
        '"""doc\r\nstring"""',
    ]


@pytest.mark.parametrize(
    "content",
    [
        pytest.param('FENCE = "\\n```\\n"\nnot python $(\n', id="invalid-python"),
        pytest.param('#!/bin/sh\nFENCE = "\\n```\\n"\n', id="shell-shebang"),
        pytest.param('FENCE = "\\n```\\n"\rvalue = 1\n', id="lone-carriage-return"),
        pytest.param('FENCE = "\\n```\\n"\n\x00\n', id="nul-byte"),
    ],
)
def test_unproven_python_source_has_no_token_ownership(content: str) -> None:
    assert tm._python_literal_spans(content, lambda: None) is None
    assert _python(content + _TAIL) is True


def test_python_ownership_requires_complete_context() -> None:
    # A fragment can start inside a string and invert code and literal bytes.
    assert _python('FENCE = "\\n```\\n"\n' + _TAIL, complete_context=False) is True


def test_bom_prefixed_python_keeps_conservative_result() -> None:
    # The file cache decodes with utf-8, not utf-8-sig, so a leading U+FEFF is
    # not Python syntax. Ownership stays unproven until that decoding changes.
    content = "\ufeff" + _FENCE_SOURCE

    assert tm._python_literal_spans(content, lambda: None) is None
    assert _python(content) is True


# A valid module whose invalid escape makes the compiler emit a SyntaxWarning.
_WARNING_SOURCE = 'PATTERN = "\\d+"\n' + _FENCE_SOURCE


def test_python_ownership_parse_emits_no_warnings() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        filters = list(warnings.filters)

        assert _python(_WARNING_SOURCE) is False
        assert list(warnings.filters) == filters

    assert [str(warning.message) for warning in caught] == []


@pytest.mark.filterwarnings("error::SyntaxWarning")
def test_python_ownership_survives_syntax_warnings_as_errors() -> None:
    assert _python(_WARNING_SOURCE) is False


def test_concurrent_ownership_parses_restore_warning_filters() -> None:
    # Analyzer nodes run on graph worker threads. Interleaved catch_warnings
    # exits must not leave the ownership parse's ignore filter installed.
    filters = list(warnings.filters)
    barrier = threading.Barrier(8)
    proven: list[bool] = []

    def parse(worker: int) -> None:
        barrier.wait()
        for index in range(10):
            # A distinct source per call misses the memo, so every call parses.
            source = f"{_WARNING_SOURCE}# {worker}-{index}\n"
            proven.append(tm._python_literal_spans(source, lambda: None) is not None)

    threads = [threading.Thread(target=parse, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert proven == [True] * 80
    assert list(warnings.filters) == filters


@pytest.mark.parametrize(
    "source",
    [
        pytest.param('COMMAND = "echo `' + "word " * 900 + '"\n', id="single-literal"),
        pytest.param('COMMAND = "echo " "`' + "word " * 900 + '"\n', id="adjacent-literal"),
        pytest.param("# doesn't " + "word " * 900 + "\n", id="single-comment"),
    ],
)
def test_long_unresolved_shell_inside_one_python_token_stays_partial(source: str) -> None:
    assert _python(source + "value = 1\n") is True


def test_runtime_command_with_destructive_operands_in_python_comment_stays_partial() -> None:
    assert _python(_OPTION_TEXT + "# $TOOL -rf /\nvalue = 1\n" + _TAIL) is True


@pytest.mark.parametrize(
    "joiner", ["\n           ", "  # tool\n           + "], ids=["implicit", "plus"]
)
@pytest.mark.parametrize(
    "operands,partial",
    [('"-rf /"', True), ('"--force --recursive /"', True), ('"-rf build"', False)],
    ids=["root", "long-options", "relative"],
)
def test_runtime_command_takes_operands_from_concatenated_python_literals(
    joiner: str, operands: str, partial: bool
) -> None:
    source = f'COMMAND = (f"run `{{tool}}` "{joiner}{operands})\n'

    assert _python(_OPTION_TEXT + source + _TAIL) is partial


@pytest.mark.parametrize(
    "source",
    [
        pytest.param('COMMAND = f"run `{tool}` -rf /"\n', id="same-literal"),
        pytest.param('COMMAND = f"Inspected `{path}` and found %s." % detail\n', id="percent"),
        pytest.param('COMMAND = "Inspected `$P` and found {}.".format(detail)\n', id="format"),
        pytest.param('COMMAND = f"Inspected `{path}` and " + detail\n', id="non-literal"),
        pytest.param('COMMAND = (f"run `{tool}` %s") % "-rf /"\n', id="parenthesized-percent"),
        pytest.param('COMMAND = ("run $(tool) ") + "-rf /"\n', id="parenthesized-plus"),
        pytest.param('COMMAND = "run $(tool) "[0:] + "-rf /"\n', id="subscript"),
        pytest.param('COMMAND = f"run `{tool}` {flag} /"\n', id="replacement-field"),
    ],
)
def test_runtime_command_in_python_string_keeps_undecided_operands_partial(source: str) -> None:
    # Code can still append operands to a string that is formatted, sliced,
    # joined with a non-literal, or followed by a method call, also after a
    # closing parenthesis. A replacement field can supply the options for a
    # root target.
    assert _python(_OPTION_TEXT + source + _TAIL) is True


@pytest.mark.parametrize(
    "source,shell_text",
    [
        pytest.param('X = f"a `{b}` c"\n', "`{b}` c", id="f-string"),
        pytest.param(
            "X = (\"a `{b}` \"  # note\n     + \"c\" \\\n     r'''d''')\n",
            "`{b}` cd",
            id="concatenated",
        ),
        pytest.param('X = ["a `{b}` ", "c"]\n', "`{b}` ", id="list-items"),
        pytest.param('X = "a `{b}` " if c else "d"\n', "`{b}` ", id="conditional"),
        pytest.param('X = "a `{b}` %s" % c\n', None, id="percent"),
        pytest.param('X = "a `{b}` ".join(c)\n', None, id="method"),
        pytest.param('X = "a `{b}` " * c\n', None, id="repetition"),
        pytest.param('X = "a `{b}` " + c\n', None, id="non-literal"),
        pytest.param('X = ("a `{b}` ") % c\n', None, id="parenthesized"),
        pytest.param('X = "a `{b}` "[c]\n', None, id="subscript"),
        pytest.param(
            'X = f"a `{b}` {c} {{d}}" "{e}" f"{{{g}}}"\n',
            "`{b}`  -rf {c} {{d}}{e}{{ -rf {g}}}",
            id="replacement-fields",
        ),
    ],
)
def test_python_string_value_joins_only_concatenated_literals(
    source: str, shell_text: str | None
) -> None:
    start = source.index("`")
    value = tm._PythonSourceOwnership(source, lambda: None).string_value(start, start + 5)

    # The shell word and the operands that the rest of the value supplies.
    assert (value if value is None else value[0][value[1] :]) == shell_text


def test_python_closing_quote_starts_no_shell_word() -> None:
    source = 'X = ("a "\n     "$B -rf /")\n'
    start = source.index('"\n')

    assert tm._PythonSourceOwnership(source, lambda: None).string_value(start, start + 1) == (
        "",
        0,
        0,
    )


def test_shell_file_with_unmatched_backtick_and_long_tail_stays_partial() -> None:
    content = "#!/bin/sh\necho `date\n" + "echo ordinary line\n" * 300

    assert _shell(content) is True
    result = _ledger("scripts/run.sh", content)
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


def test_python_token_ownership_honors_runtime_checks() -> None:
    content = 'FENCE = "\\n```\\n"\n' + "".join(
        f"value_{index} = 'x'  # note\n" for index in range(5000)
    )
    checks = 0

    def check_runtime() -> None:
        nonlocal checks
        checks += 1
        # The first checks bracket the module parse; later ones must come
        # from the bounded token scan itself.
        if checks >= 4:
            raise TimeoutError("test runtime bound")

    with pytest.raises(TimeoutError, match="test runtime bound"):
        tm._python_literal_spans(content, check_runtime)


def test_python_ownership_keeps_tool_misuse_detection() -> None:
    content = (
        'FENCE = "\\n```\\n"\nimport subprocess\nsubprocess.run("rm -rf /", shell=True)\n' + _TAIL
    )

    result = _ledger("scripts/report.py", content)

    assert any(finding.rule_id == "TM1" for finding in result["findings"])
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


# --- Node assertion diagnostic templates ----------------------------------


_JS_ASSERT_IMPORT = 'import check from "node:assert/strict";\n'
_JS_ASSERT_CALL = 'check.equal(typeof row.name, "string", `${row.key} must be present`);\n'


def _javascript(content: str, *, complete_context: bool = True) -> bool:
    return tm.has_bounded_parse_exhaustion(
        content, lambda: None, file_type="javascript", complete_context=complete_context
    )


@pytest.mark.parametrize(
    "declaration",
    [
        'import check from "node:assert";',
        'import check from "node:assert/strict";',
        "import * as check from 'node:assert';",
        "import * as check from 'node:assert/strict';",
    ],
)
def test_imported_node_assert_message_template_is_complete(declaration: str) -> None:
    source = declaration + "\n" + _JS_ASSERT_CALL

    assert _javascript(source) is False
    assert _shell(source) is True


@pytest.mark.parametrize(
    "method,arguments",
    [
        ("ok", "valid"),
        ("equal", "actual, expected"),
        ("notEqual", "actual, expected"),
        ("strictEqual", "actual, expected"),
        ("notStrictEqual", "actual, expected"),
        ("deepEqual", "actual, expected"),
        ("notDeepEqual", "actual, expected"),
        ("deepStrictEqual", "actual, expected"),
        ("notDeepStrictEqual", "actual, expected"),
        ("match", "actual, /value/"),
        ("doesNotMatch", "actual, /value/"),
        ("throws", "() => work(), undefined"),
        ("doesNotThrow", "() => work(), undefined"),
    ],
)
def test_node_assert_owns_only_its_message_argument(method: str, arguments: str) -> None:
    source = _JS_ASSERT_IMPORT + f"check.{method}({arguments}, `${{row.key}} needs a value`);\n"

    assert _javascript(source) is False


@pytest.mark.parametrize("suffix", [".js", ".mjs", ".cjs", ".MJS"])
def test_node_assert_message_ownership_reaches_static_ledger(suffix: str) -> None:
    result = _ledger("scripts/check" + suffix, _JS_ASSERT_IMPORT + _JS_ASSERT_CALL)

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert not any(finding.rule_id == "TM1" for finding in result["findings"])


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_node_assert_template_offsets_preserve_dangerous_neighbor(newline: str) -> None:
    source = (
        _JS_ASSERT_IMPORT
        + "check.ok(ready, `café ${row.key} is ready`);\n"
        + 'execSync("rm -rf /");\n'
    ).replace("\n", newline)

    result = _ledger("scripts/check.mjs", source)

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert any(
        finding.rule_id == "TM1" and finding.start_line == 3 for finding in result["findings"]
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(_JS_ASSERT_CALL, id="unbound-name"),
        pytest.param(
            'import check from "custom-assert";\n' + _JS_ASSERT_CALL, id="non-node-import"
        ),
        pytest.param(
            'import { strict as check } from "node:assert";\n' + _JS_ASSERT_CALL,
            id="unsupported-named-import",
        ),
        pytest.param(_JS_ASSERT_IMPORT + "check = run;\n" + _JS_ASSERT_CALL, id="rebound"),
        pytest.param(
            _JS_ASSERT_IMPORT + "function verify(check) {\n" + _JS_ASSERT_CALL + "}\n",
            id="shadowed-parameter",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + "const alias = check;\n" + _JS_ASSERT_CALL, id="escaped-binding"
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + "const run = check.equal;\n" + _JS_ASSERT_CALL,
            id="aliased-method",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + "const message = `${check}`;\n" + _JS_ASSERT_CALL,
            id="binding-interpolated-elsewhere",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + "const message = `${check.equal}`;\n" + _JS_ASSERT_CALL,
            id="method-interpolated-elsewhere",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + "check.equal = run;\n" + _JS_ASSERT_CALL,
            id="method-replacement",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + 'check["equal"] = run;\n' + _JS_ASSERT_CALL,
            id="computed-replacement",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + "Object.assign(check, { equal: run });\n" + _JS_ASSERT_CALL,
            id="object-mutation",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT
            + 'import alternate from "node:assert";\nalternate.equal = run;\n'
            + _JS_ASSERT_CALL,
            id="second-import-alias",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + 'require("node:assert/strict").equal = run;\n' + _JS_ASSERT_CALL,
            id="require-alias",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT
            + r'import alternate from "node:\u0061ssert";'
            + "\nalternate.equal = run;\n"
            + _JS_ASSERT_CALL,
            id="escaped-module-import",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT
            + r'require("node:\u0061ssert").equal = run;'
            + "\n"
            + _JS_ASSERT_CALL,
            id="escaped-module-require",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT
            + "const of = 1; const n = of / (check.equal = run) / divisor;\n"
            + _JS_ASSERT_CALL,
            id="division-after-contextual-keyword",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT
            + "const n = holder.return / (check.equal = run) / divisor;\n"
            + _JS_ASSERT_CALL,
            id="division-after-keyword-property",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + 'check["equal"](a, b, `${row.key} is required`);\n',
            id="computed-call",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + "check?.equal(a, b, `${row.key} is required`);\n",
            id="optional-call",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + "check(a, `${row.key} is required`);\n",
            id="unsupported-callable-assert",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + "check.equal(`${row.key}`, b);\n",
            id="template-is-actual-value",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + "check.equal(a, b, `${row.key}`, extra);\n",
            id="extra-argument",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + "check.equal(...values, `${row.key}`);\n",
            id="spread-argument",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + "check.equal(a, b, tag`${row.key}`);\n",
            id="tagged-message",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + "check.equal(a, b, (`${row.key}`));\n",
            id="parenthesized-message",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + 'check.equal(a, b, `${row.key}` + " more");\n',
            id="concatenated-message",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + "check.equal(a, b, message = `${row.key}`);\n",
            id="assigned-message",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + "const message = `${row.key}`;\n",
            id="unrelated-assignment",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + "function message() { return `${row.key}`; }\n",
            id="unrelated-return",
        ),
        pytest.param("#!/bin/sh\n" + _JS_ASSERT_IMPORT + _JS_ASSERT_CALL, id="shell-shebang"),
        pytest.param(_JS_ASSERT_IMPORT + "check.equal(a, b, `${row.key}\n", id="unclosed-template"),
        pytest.param(_JS_ASSERT_IMPORT + "check.equal(a, b, `${row.key`);\n", id="unclosed-field"),
        pytest.param(_JS_ASSERT_IMPORT + _JS_ASSERT_CALL + "/* unfinished", id="unclosed-comment"),
        pytest.param(
            _JS_ASSERT_IMPORT + "const pattern = /unterminated\n" + _JS_ASSERT_CALL,
            id="unclosed-regexp",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + "const pattern = /value/gg;\n" + _JS_ASSERT_CALL,
            id="invalid-regexp-flags",
        ),
        pytest.param(
            _JS_ASSERT_IMPORT + "const pattern = /value\\\nmore/;\n" + _JS_ASSERT_CALL,
            id="escaped-newline-in-regexp",
        ),
    ],
)
def test_unproven_node_assert_binding_or_message_is_not_projected(source: str) -> None:
    from skillspector.javascript_tokens import assertion_template_shell_text

    assert assertion_template_shell_text(source, lambda: None) == source
    assert _javascript(source) is _shell(source)


@pytest.mark.parametrize(
    "operand",
    [
        "value!",
        "value !",
        "value /* gap */ !",
        "value! /* gap */",
        "holder.value!",
        "getValue()!",
        "(value)!",
        "values[0]!",
    ],
)
@pytest.mark.parametrize("file_type", ["javascript", "typescript"])
def test_postfix_non_null_assertion_cannot_hide_receiver_mutation(
    operand: str, file_type: str
) -> None:
    from skillspector.javascript_tokens import assertion_template_shell_text

    source = (
        _JS_ASSERT_IMPORT
        + f"const n = {operand} / (check.equal = run) / divisor;\n"
        + _JS_ASSERT_CALL
    )

    assert assertion_template_shell_text(source, lambda: None) == source
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type=file_type) is True


@pytest.mark.parametrize(
    "expression",
    ["!value", "!!value", "!/value/.test(input)", "value != other", "value !== other"],
)
def test_prefix_negation_and_inequality_keep_assert_message_ownership(expression: str) -> None:
    source = _JS_ASSERT_IMPORT + f"const ready = {expression};\n" + _JS_ASSERT_CALL

    assert _javascript(source) is False
    assert _shell(source) is True


@pytest.mark.parametrize(
    "statement",
    [
        "",
        "const n = value! / (check.equal = run) / divisor;\n",
        "const n = factory<number> / (check.equal = run) / divisor;\n",
    ],
)
def test_typescript_does_not_borrow_javascript_assert_ownership(statement: str) -> None:
    source = _JS_ASSERT_IMPORT + statement + _JS_ASSERT_CALL

    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="typescript") is True
    result = _ledger("scripts/check.ts", source)
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize(
    "prefix,message",
    [
        ("// check.equal = run;\n", "ready ${row.key}"),
        ("/* check is a diagnostic receiver */\n", "ready ${row.key}"),
        (r"""const pattern = /['"]/;""" + "\n", "ready ${row.key}"),
        ("", r"escaped \` delimiter ${row.key}"),
        ("", "multiple ${label} and ${row.key} values"),
    ],
)
def test_node_assert_projection_changes_only_proven_delimiters(prefix: str, message: str) -> None:
    from skillspector.javascript_tokens import assertion_template_shell_text

    source = _JS_ASSERT_IMPORT + prefix + f"check.equal(a, b, `{message}`);\n"
    opening = source.index("`", len(_JS_ASSERT_IMPORT + prefix))
    closing = source.rindex("`")

    projected = assertion_template_shell_text(source, lambda: None)

    assert len(projected) == len(source)
    assert [
        index
        for index, pair in enumerate(zip(source, projected, strict=True))
        if pair[0] != pair[1]
    ] == [
        opening,
        closing,
    ]
    assert projected[opening] == projected[closing] == " "


@pytest.mark.parametrize(
    "expression",
    [
        "execSync(obj.cmd)",
        "new Command(obj.cmd)",
        "(target = obj.cmd)",
        "counter++",
        "++counter",
        "obj[key]",
        "obj?.cmd",
        "`nested ${obj.cmd}`",
    ],
)
def test_effectful_or_unproven_assert_interpolation_is_not_projected(expression: str) -> None:
    from skillspector.javascript_tokens import assertion_template_shell_text

    source = _JS_ASSERT_IMPORT + f"check.equal(a, b, `${{{expression}}} ready`);\n"

    assert assertion_template_shell_text(source, lambda: None) == source
    assert _javascript(source) is _shell(source)


@pytest.mark.parametrize(
    "statement",
    [
        "execSync(`${obj.cmd}`);",
        "run(`${obj.cmd}`);",
        "execSync(`${getCmd()}`);",
        "execSync(`${cmd} -rf /`);",
        "execSync(`${obj.cmd} -rf /`);",
        'execSync(`${obj.cmd}` + " -rf /");',
    ],
)
def test_node_assert_message_fix_keeps_unresolved_shell_commands_partial(statement: str) -> None:
    source = _JS_ASSERT_IMPORT + _JS_ASSERT_CALL + statement + "\n"

    assert _javascript(source) is True
    result = _ledger("scripts/check.mjs", source)
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize(
    "statement",
    [
        "run(`${cmd}`);",
        "sh`${obj.cmd}`;",
        "sh`${obj.cmd} -rf /`;",
        "execSync(String.raw`${obj.cmd}`);",
        'execSync(`${cmd}` + " -rf /");',
    ],
)
def test_unrecognized_template_consumers_keep_existing_behavior(statement: str) -> None:
    from skillspector.javascript_tokens import assertion_template_shell_text

    source = _JS_ASSERT_IMPORT + statement + "\n"

    assert assertion_template_shell_text(source, lambda: None) == source
    assert _javascript(source) is _shell(source)


@pytest.mark.parametrize(
    "statement",
    [
        "execSync(`rm -rf /`);",
        "check.equal(a, b, `${execSync('rm -rf /')} ready`);",
        "check.equal(a, b, `rm -rf / ${row.key}`);",
    ],
)
def test_assert_template_ownership_preserves_destructive_text(statement: str) -> None:
    result = _ledger("scripts/check.mjs", _JS_ASSERT_IMPORT + statement + "\n")

    assert any(finding.rule_id == "TM1" for finding in result["findings"])


def test_node_assert_message_ownership_requires_complete_javascript_context() -> None:
    source = _JS_ASSERT_IMPORT + _JS_ASSERT_CALL

    assert _javascript(source, complete_context=False) is True
    assert _shell(source) is True


def test_node_assert_template_scan_propagates_runtime_interruptions() -> None:
    from skillspector.javascript_tokens import assertion_template_shell_text

    def check_runtime() -> None:
        raise TimeoutError("test runtime bound")

    with pytest.raises(TimeoutError, match="test runtime bound"):
        assertion_template_shell_text(_JS_ASSERT_IMPORT + _JS_ASSERT_CALL * 1000, check_runtime)


def test_oversized_javascript_does_not_gain_template_ownership() -> None:
    from skillspector.javascript_tokens import assertion_template_shell_text

    source = _JS_ASSERT_IMPORT + "/*" + "x" * 1_100_000 + "*/\n" + _JS_ASSERT_CALL

    assert assertion_template_shell_text(source, lambda: None) == source


# --- Whole-scan accounting -------------------------------------------------


def _write_bundle(root: Path, files: dict[str, str]) -> None:
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


_MANIFEST = (
    "---\nname: report-builder\ndescription: Build a Markdown report from local results.\n"
    "---\n\nRun `scripts/report.py` to build the report.\n\n"
    "Run `scripts/load.pl` to load the optional plugin.\n"
)
_REPORT_SCRIPT = (
    '"""Render results as Markdown (see the `results` directory)."""\n\n'
    "# The renderer doesn't escape table cells.\n"
    'FENCE = "\\n```\\n"\n\n\n'
    "def render(lang, body):\n"
    '    return f"```{lang}\\n{body}" + FENCE\n' + _TAIL
)
_LOADER_SCRIPT = (
    "#!/usr/bin/env perl\nuse strict;\nuse warnings;\n"
    'my $module = shift @ARGV or die "usage: load.pl <module>";\n'
    '{ local $^W = 0; eval { require $module; 1 } or die "load failed: $@"; }\n'
)


def test_referenced_python_and_perl_helpers_are_complete_without_ae1(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path,
        {
            "SKILL.md": _MANIFEST,
            "scripts/report.py": _REPORT_SCRIPT,
            "scripts/load.pl": _LOADER_SCRIPT,
        },
    )

    result = graph.invoke({"input_path": str(tmp_path), "use_llm": False, "output_format": "json"})

    assert result["analysis_completeness"]["status"] == "complete"
    assert not any(finding.rule_id == "AE1" for finding in result["findings"])


def test_referenced_python_helper_with_unresolved_shell_literal_keeps_ae1(
    tmp_path: Path,
) -> None:
    _write_bundle(
        tmp_path,
        {
            "SKILL.md": _MANIFEST,
            "scripts/report.py": 'COMMAND = "echo `' + "word " * 900 + '"\n' + _TAIL,
            "scripts/load.pl": _LOADER_SCRIPT,
        },
    )

    result = graph.invoke({"input_path": str(tmp_path), "use_llm": False, "output_format": "json"})

    assert result["analysis_completeness"]["status"] == "partial"
    assert any(
        finding.rule_id == "AE1" and finding.evidence.get("target_path") == "scripts/report.py"
        for finding in result["findings"]
    )
