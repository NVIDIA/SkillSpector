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
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from skillspector import python_string_constants, python_tokens
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


def test_concatenated_fstrings_keep_runtime_operand_ambiguity() -> None:
    content = (
        _OPTION_TEXT + "def message(tool):\n    return (\n"
        '        f"{tool} was not found at "\n'
        '        f"$TOOL_ROOT, $UPSTREAM_ROOT/{tool}, "\n'
        '        f"or $HOME/.cache/upstreams/{tool} "\n'
        '        "or elsewhere"\n    )\n' + _TAIL
    )
    # The body-start runtime command is now inspected. An empty upstream
    # parameter can leave a root path, while a replacement field supplies flags.
    # Concatenation must retain those possible operands even in report strings.
    assert _python(content) is True


_CONSTANT_CHECKOUT_SOURCE = """\
from urllib.parse import urlparse
from settings import active_branch

REPOSITORY = "https://example.com/tools/widget.git"

def checkout_name(repository):
    name = urlparse(repository).path.rstrip("/").rsplit("/", 1)[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name

CHECKOUT = checkout_name(REPOSITORY)

def message():
    return (
        f"$TOOL_ROOT, $UPSTREAM_ROOT/{CHECKOUT}, "
        f"or $HOME/.cache/upstreams/{CHECKOUT} "
        f"checked out to {active_branch}"
    )
"""


@pytest.mark.parametrize("repository", ["widget", "widget/", "widget.git", "widget.git/"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_constant_checkout_fstrings_do_not_invent_runtime_options(
    repository: str, newline: str
) -> None:
    source = _CONSTANT_CHECKOUT_SOURCE.replace("tools/widget.git", f"tools/{repository}")
    # The branch remains unknown; only the checkout component has a fixed value.
    content = (_OPTION_TEXT + "# Unicode offset: café\n" + source + _TAIL).replace("\n", newline)

    assert _python(content) is False


@pytest.mark.parametrize(
    "before,after",
    [
        pytest.param(
            'REPOSITORY = "https://example.com/tools/widget.git"',
            "REPOSITORY = supplied_repository",
            id="unknown-url",
        ),
        pytest.param(
            "CHECKOUT = checkout_name(REPOSITORY)",
            "REPOSITORY = supplied_repository\nCHECKOUT = checkout_name(REPOSITORY)",
            id="rebound-url",
        ),
        pytest.param(
            "CHECKOUT = checkout_name(REPOSITORY)",
            "CHECKOUT = checkout_name(REPOSITORY)\nCHECKOUT = supplied_checkout",
            id="rebound-result",
        ),
        pytest.param(
            "CHECKOUT = checkout_name(REPOSITORY)",
            "CHECKOUT = checkout_name(REPOSITORY)\ndel CHECKOUT",
            id="deleted-result",
        ),
        pytest.param("def message():", "def message(CHECKOUT):", id="parameter-shadow"),
        pytest.param(
            "def message():",
            "def message():\n    CHECKOUT = supplied_checkout",
            id="local-shadow",
        ),
        pytest.param(
            "def message():",
            "def message():\n    global CHECKOUT\n    CHECKOUT = supplied_checkout",
            id="global-write",
        ),
        pytest.param(
            "def checkout_name(repository):",
            "urlparse = external\n\ndef checkout_name(repository):",
            id="rebound-import",
        ),
        pytest.param(
            "def checkout_name(repository):",
            "external(urlparse)\n\ndef checkout_name(repository):",
            id="escaped-import",
        ),
        pytest.param(
            "def checkout_name(repository):",
            "@external\ndef checkout_name(repository):",
            id="decorated-helper",
        ),
        pytest.param(
            "CHECKOUT = checkout_name(REPOSITORY)",
            "checkout_name = external\nCHECKOUT = checkout_name(REPOSITORY)",
            id="rebound-helper",
        ),
        pytest.param(
            "CHECKOUT = checkout_name(REPOSITORY)",
            "external(checkout_name)\nCHECKOUT = checkout_name(REPOSITORY)",
            id="escaped-helper",
        ),
        pytest.param(
            "    return name",
            "    return checkout_name(repository)",
            id="recursive-helper",
        ),
        pytest.param(
            "def message():",
            'globals()["CHECKOUT"] = supplied_checkout\n\ndef message():',
            id="reflective-write",
        ),
        pytest.param(
            "from urllib.parse import urlparse",
            'import sys\nsys.modules["urllib.parse"] = external\nfrom urllib.parse import urlparse',
            id="import-registry-write",
        ),
        pytest.param(
            "from urllib.parse import urlparse",
            "import sys\nsys.meta_path.insert(0, external)\nfrom urllib.parse import urlparse",
            id="import-meta-hook",
        ),
        pytest.param(
            "from urllib.parse import urlparse",
            "import sys\nsys.path_hooks.append(external)\nfrom urllib.parse import urlparse",
            id="import-path-hook",
        ),
        pytest.param(
            "from urllib.parse import urlparse",
            'import sys\nsys.path_importer_cache["."] = external\nfrom urllib.parse import urlparse',
            id="import-cache-write",
        ),
        pytest.param(
            "from urllib.parse import urlparse",
            '__builtins__["__import__"] = external\nfrom urllib.parse import urlparse',
            id="builtin-loader-write",
        ),
        pytest.param(
            "def checkout_name(repository):",
            "def unrelated(urlparse=external(urlparse)):\n    pass\n\ndef checkout_name(repository):",
            id="import-escape-in-shadowing-default",
        ),
        pytest.param(
            "CHECKOUT = checkout_name(REPOSITORY)",
            "def unrelated(checkout_name=external(checkout_name)):\n    pass\n"
            "CHECKOUT = checkout_name(REPOSITORY)",
            id="helper-escape-in-shadowing-default",
        ),
        pytest.param(
            "def checkout_name(repository):",
            "def unrelated(urlparse: external(urlparse)):\n    pass\n\ndef checkout_name(repository):",
            id="import-escape-in-shadowing-annotation",
        ),
        pytest.param(
            "CHECKOUT = checkout_name(REPOSITORY)",
            "def unrelated(checkout_name: external(checkout_name)):\n    pass\n"
            "CHECKOUT = checkout_name(REPOSITORY)",
            id="helper-escape-in-shadowing-annotation",
        ),
    ],
)
def test_constant_checkout_proof_rejects_unknown_or_changed_bindings(
    before: str, after: str
) -> None:
    assert before in _CONSTANT_CHECKOUT_SOURCE
    source = _CONSTANT_CHECKOUT_SOURCE.replace(before, after, 1)

    assert _python(_OPTION_TEXT + source + _TAIL) is True


@pytest.mark.parametrize("field", ["CHECKOUT!r", "CHECKOUT:>20", "CHECKOUT:{width}"])
def test_constant_checkout_proof_rejects_unmodeled_field_formatting(field: str) -> None:
    source = _CONSTANT_CHECKOUT_SOURCE.replace("{CHECKOUT}", "{" + field + "}")

    assert python_string_constants.python_constant_string_fields(source, lambda: None) == ()


@pytest.mark.parametrize("extension", [" % detail", ".format(detail)", "[0:]", " + detail"])
def test_constant_checkout_proof_keeps_extended_string_values_partial(extension: str) -> None:
    source = _CONSTANT_CHECKOUT_SOURCE.replace("    )\n", "    )" + extension + "\n")

    assert _python(_OPTION_TEXT + source + _TAIL) is True


@pytest.mark.parametrize(
    "checkout",
    [
        pytest.param("-rf", id="literal-options"),
        pytest.param("-rf /", id="literal-root-options"),
        pytest.param("", id="empty-component"),
    ],
)
def test_constant_field_proof_rejects_unsafe_or_empty_components(checkout: str) -> None:
    source = _CONSTANT_CHECKOUT_SOURCE.replace(
        "CHECKOUT = checkout_name(REPOSITORY)", f"CHECKOUT = {checkout!r}"
    )

    assert python_string_constants.python_constant_string_fields(source, lambda: None) == ()
    assert _python(_OPTION_TEXT + source + _TAIL) is True


def test_constant_checkout_proof_preserves_real_shell_findings() -> None:
    source = (
        _OPTION_TEXT
        + _CONSTANT_CHECKOUT_SOURCE
        + '\nimport subprocess\nsubprocess.run("rm -rf /", shell=True)\n'
        + _TAIL
    )

    assert _python(source) is True
    result = _ledger("scripts/report.py", source)
    assert any(finding.rule_id == "TM1" for finding in result["findings"])


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_constant_field_spans_exclude_literal_braces_and_format_specs(newline: str) -> None:
    content = (
        "# Unicode offset: café\n"
        + _CONSTANT_CHECKOUT_SOURCE.replace('f"$TOOL_ROOT', 'f"café $TOOL_ROOT')
        + 'LITERAL = "{CHECKOUT}"\n'
        + 'ESCAPED = f"{{CHECKOUT}}"\n'
        + 'NESTED = f"{unknown:{CHECKOUT}}"\n'
        + "LAMBDA = f\"{(lambda: f'{CHECKOUT},')()}\"\n"
        + 'DEBUG = f"{CHECKOUT=}"\n'
    ).replace("\n", newline)

    fields = python_string_constants.python_constant_string_fields(content, lambda: None)

    assert [(content[start:end], value) for start, end, value in fields] == [
        ("{CHECKOUT}", "widget"),
        ("{CHECKOUT}", "widget"),
    ]
    first = content.index("{CHECKOUT}")
    second = content.index("{CHECKOUT}", first + 1)
    assert [start for start, _, _ in fields] == [first, second]


@pytest.mark.parametrize(
    "body",
    [
        '    message = f"$TOOL -rf /{CHECKOUT}"\n',
        '    def message(self, value=f"$TOOL -rf /{CHECKOUT}"):\n        return value\n',
        '    messages = [value for value in (f"$TOOL -rf /{CHECKOUT}",)]\n',
    ],
    ids=["class-body", "method-default", "comprehension-first-iterable"],
)
def test_constant_field_proof_refuses_class_lookup_contexts(body: str) -> None:
    source = 'CHECKOUT = "widget"\nclass Report:\n    CHECKOUT = ""\n' + body

    assert python_string_constants.python_constant_string_fields(source, lambda: None) == ()


@pytest.mark.parametrize("warning_filter", ["always", "error"])
def test_constant_field_proof_suppresses_source_warnings(warning_filter: str) -> None:
    source = 'PATTERN = "\\d+"\n' + _CONSTANT_CHECKOUT_SOURCE
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(warning_filter)
        filters = list(warnings.filters)

        fields = python_string_constants.python_constant_string_fields(source, lambda: None)

        assert [value for _, _, value in fields] == ["widget", "widget"]
        assert list(warnings.filters) == filters
    assert caught == []


def test_constant_field_proof_bounds_total_output_and_honors_cancellation() -> None:
    content = 'VALUE = "' + "x" * 4096 + '"\n' + 'f"{VALUE}"\n' * 9

    assert python_string_constants.python_constant_string_fields(content, lambda: None) == ()

    checks = 0

    def cancel() -> None:
        nonlocal checks
        checks += 1
        if checks == 4:
            raise TimeoutError("constant fields deadline")

    with pytest.raises(TimeoutError, match="constant fields deadline"):
        python_string_constants.python_constant_string_fields(
            _CONSTANT_CHECKOUT_SOURCE + _TAIL, cancel
        )
    assert checks == 4


def test_constant_field_proof_respects_bundled_standard_library_override() -> None:
    checked = []

    def bundled_override(module: str, check_runtime: Callable[[], None]) -> bool:
        checked.append(module)
        check_runtime()
        return False

    assert (
        python_string_constants.python_constant_string_fields(
            _CONSTANT_CHECKOUT_SOURCE, lambda: None, stdlib_check=bundled_override
        )
        == ()
    )
    assert checked == ["urllib"]


def _constant_cutoff_source(layout: str, suffix: str, *, field_count: int = 600) -> str:
    field = "{LONG_CONSTANT_NAME}"
    if layout in {"single", "boundary"}:
        # Short continued lines bound offset work. Align the raw window with
        # a complete field, so an accidentally cut brace cannot mask the bug.
        unit = field + "\\\n"
        padding = " " * ((tm._ROOT_GLOB_COMMAND_CHARS + 1) % len(unit))
        if layout == "boundary":
            field_count = (tm._ROOT_GLOB_COMMAND_CHARS + 1 - len(padding)) // len(unit)
        literals = ['f"$TOOL' + padding + unit * field_count + '"']
    else:
        literals = ['f"$TOOL "'] + [f'f"{field}"'] * field_count
    if suffix == "operands":
        literals.append('" -rf /"')
        extension = ""
    else:
        extension = suffix
    joiner = " +\n        " if layout == "plus" else "\n        "
    return (
        _OPTION_TEXT
        + 'LONG_CONSTANT_NAME = "x"\n'
        + "def message(detail):\n    return (\n        "
        + joiner.join(literals)
        + "\n    )"
        + extension
        + "\n"
        + _TAIL
    )


@pytest.mark.parametrize(
    "layout,suffix",
    [
        pytest.param("single", "operands", id="initial-literal"),
        pytest.param("boundary", "operands", id="exact-source-boundary"),
        pytest.param("implicit", "operands", id="implicit-literals"),
        pytest.param("plus", "operands", id="plus-literals"),
        pytest.param("single", ".format(detail)", id="initial-late-format"),
        pytest.param("implicit", ".format(detail)", id="implicit-late-format"),
        pytest.param("plus", " % detail", id="late-percent"),
        pytest.param("plus", " + detail", id="late-unknown-concatenation"),
    ],
)
def test_constant_field_shrinkage_cannot_hide_operand_window_cutoff(
    layout: str, suffix: str
) -> None:
    source = _constant_cutoff_source(layout, suffix)
    start = source.index("$TOOL")
    fields = python_string_constants.python_constant_string_fields(source, lambda: None)
    assert len(fields) >= 300
    assert {value for _, _, value in fields} == {"x"}
    ownership = tm._PythonSourceOwnership(source, lambda: None)

    assert ownership.string_value(start, start + len("$TOOL")) is None
    assert _python(source) is True


@pytest.mark.parametrize("layout", ["single", "implicit", "plus"])
def test_constant_field_shrinkage_within_operand_window_stays_complete(layout: str) -> None:
    source = _constant_cutoff_source(layout, "", field_count=16)
    start = source.index("$TOOL")
    fields = python_string_constants.python_constant_string_fields(source, lambda: None)
    assert len(fields) == 16
    ownership = tm._PythonSourceOwnership(source, lambda: None)

    assert ownership.string_value(start, start + len("$TOOL")) is not None
    assert _python(source) is False


@pytest.mark.parametrize("target,partial", [("/", True), ("build", False)])
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize("leading", ["", "\n"], ids=["body-start", "next-line"])
def test_completed_python_word_cannot_claim_a_later_string(
    target: str, partial: bool, newline: str, leading: str
) -> None:
    # A quote in the replacement expression can lexically pair with the next
    # literal's quote. That apparent completed word must not hide a later
    # runtime-selected command, even when the literal starts with it.
    content = (
        "message = f\"{' '.join(parts)} ready\"\n"
        f'REPORT = r"""{leading}$TOOL -rf {target}\nlabel(\'value\')\n"""\n'
    ).replace("\n", newline)

    assert _python(content) is partial


@pytest.mark.parametrize(
    "javascript",
    [
        "execSync(`${{obj.cmd}}`);",
        "run(`${{obj.cmd}}`);",
        'execSync(`${{obj.cmd}}` + " -rf /");',
        'const invoke = execSync; invoke(`${{obj.cmd}}` + " -rf /");',
    ],
)
def test_python_constant_text_keeps_embedded_executable_javascript_partial(
    javascript: str,
) -> None:
    assert _python('REPORT = f"""' + javascript + '"""\n') is True


@pytest.mark.parametrize("prefix", ["r", "f", "rf"])
@pytest.mark.parametrize("quote", ['"', '"""'])
@pytest.mark.parametrize("target,dangerous", [("/", True), ("build", False)])
def test_short_prefixed_python_commands_keep_raw_findings(
    prefix: str, quote: str, target: str, dangerous: bool
) -> None:
    content = f"COMMAND = {prefix}{quote}rm -rf {target}{quote}\n"

    result = _ledger("scripts/command.py", content)

    assert any(finding.rule_id == "TM1" for finding in result["findings"]) is dangerous
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


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


# --- Escaped browser list values in Python HTML ----------------------------

_BROWSER_ASSIGNMENT = """panel.innerHTML = `<section>${listMarkup(rows.map((item) => `${item.name} :: ${item.value}`))}</section>`;"""
_BROWSER_SCRIPT = (
    """
const data = JSON.parse(document.getElementById('data').textContent);
const panel = document.getElementById('panel');
const rows = data.items;
function escapeText(value) {
  return String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;').replaceAll('"', '&quot;');
}
function listMarkup(items, emptyText='None') {
  if (!items || !items.length) {
    return `<div class="empty">${escapeText(emptyText)}</div>`;
  }
  return `<ul>${items.map((item) => `<li>${escapeText(typeof item === 'string' ? item : JSON.stringify(item))}</li>`).join('')}</ul>`;
}
"""
    + _BROWSER_ASSIGNMENT
)


def _browser_source(script: str = _BROWSER_SCRIPT, *, consumer: str | None = None) -> str:
    encoded = script.replace("{", "{{").replace("}", "}}")
    return (
        "from pathlib import Path\nfrom html import escape\nimport json\n"
        "def render(data):\n"
        "    payload = json.dumps(data).replace('</', '<\\\\/')\n"
        "    label = escape(data['label'])\n"
        "    return f'''<!doctype html><html><body><p>{label}</p><div id=\"panel\"></div>"
        '<script id="data" type="application/json">{payload}</script><script>'
        + encoded
        + "</script></body></html>'''\n"
        + (consumer or "Path('page.html').write_text(render(data), encoding='utf-8')")
        + "\n"
    )


def test_escaped_browser_list_projects_only_selected_backticks() -> None:
    source = _browser_source()
    projected = python_tokens.python_browser_template_shell_text(source, lambda: None)

    assert _python(source) is False
    assert (
        _ledger("scripts/page.py", source)["inspection_ledger"][0]["outcome"]
        is LedgerOutcome.COMPLETED
    )
    changes = [i for i, (a, b) in enumerate(zip(source, projected, strict=True)) if a != b]
    assert len(changes) == 2
    assert all(source[i] == "`" and projected[i] == " " for i in changes)
    assert projected.count("${{") == source.count("${{")


def test_escaped_browser_list_keeps_optional_property_reads() -> None:
    source = _browser_source(_BROWSER_SCRIPT.replace("data.items", "data?.items"))
    projected = python_tokens.python_browser_template_shell_text(source, lambda: None)

    assert _python(source) is False
    changes = [i for i, (a, b) in enumerate(zip(source, projected, strict=True)) if a != b]
    assert len(changes) == 2
    assert all(source[i] == "`" and projected[i] == " " for i in changes)
    assert "data?.items" in projected


def test_escaped_browser_list_keeps_literal_security_findings() -> None:
    script = _BROWSER_SCRIPT + '\nconst commandText = "rm -rf /";\n'
    source = _browser_source(script)
    result = _ledger("scripts/page.py", source)

    assert any(finding.rule_id == "TM1" for finding in result["findings"])
    # Delimiter ownership cannot complete a separate destructive shell candidate.
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL


def test_escaped_browser_list_traces_local_parameters() -> None:
    assignment = _BROWSER_ASSIGNMENT.replace("rows.map", "values.map")
    script = _BROWSER_SCRIPT.replace(
        _BROWSER_ASSIGNMENT,
        "document.getElementById('filter').value = '';\nfunction renderList(values) {\n"
        + assignment
        + "\n}\nrenderList(rows);",
    )
    source = _browser_source(script)

    assert _python(source) is False
    projected = python_tokens.python_browser_template_shell_text(source, lambda: None)
    assert sum(a != b for a, b in zip(source, projected, strict=True)) == 2


def test_escaped_browser_list_traces_json_object_entries() -> None:
    script = _BROWSER_SCRIPT.replace(
        "const rows = data.items;",
        "const rows = Object.entries(data.items).map(([name, value]) => ({name, value}));",
    )
    source = _browser_source(script)

    assert _python(source) is False
    projected = python_tokens.python_browser_template_shell_text(source, lambda: None)
    assert sum(a != b for a, b in zip(source, projected, strict=True)) == 2


@pytest.mark.parametrize("binding_kind", ["parameter", "local-const"])
@pytest.mark.parametrize("receiver,partial", [("rows", False), ("external", True)])
def test_browser_undefined_binding_requires_proven_data(
    binding_kind: str, receiver: str, partial: bool
) -> None:
    assignment = _BROWSER_ASSIGNMENT.replace("rows.map", "undefined.map")
    if binding_kind == "parameter":
        replacement = (
            f"function renderList(undefined) {{\n{assignment}\n}}\nrenderList({receiver});"
        )
    else:
        replacement = (
            f"function renderList() {{\nconst undefined = {receiver};\n"
            f"{assignment}\n}}\nrenderList();"
        )
    source = _browser_source(_BROWSER_SCRIPT.replace(_BROWSER_ASSIGNMENT, replacement))
    projected = python_tokens.python_browser_template_shell_text(source, lambda: None)

    assert _python(source) is partial
    if partial:
        assert projected == source
        assert (
            _ledger("scripts/page.py", source)["inspection_ledger"][0]["reason_code"]
            is LedgerReason.STATIC_PARSE_LIMIT
        )
    else:
        assert sum(a != b for a, b in zip(source, projected, strict=True)) == 2


def test_escaped_browser_list_traces_native_filter_values() -> None:
    script = _BROWSER_SCRIPT.replace(
        "const rows = data.items;",
        "function matches(row) {\n"
        "  const terms = [row.name, ...row.tags].join(' ').trim().toLowerCase();\n"
        "  return terms.includes(data.query);\n"
        "}\nconst rows = data.items.filter(matches);",
    )
    source = _browser_source(script)

    assert _python(source) is False
    projected = python_tokens.python_browser_template_shell_text(source, lambda: None)
    assert sum(a != b for a, b in zip(source, projected, strict=True)) == 2


def test_tagged_browser_template_keeps_existing_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _BROWSER_SCRIPT.replace(
        "`${item.name} :: ${item.value}`", "tag`${item.name} :: ${item.value}`"
    )
    source = _browser_source(script)
    assert python_tokens.python_browser_template_shell_text(source, lambda: None) == source
    observed = _python(source)
    monkeypatch.setattr(tm, "python_browser_template_shell_text", lambda text, check: text)
    assert _python(source) is observed


@pytest.mark.parametrize(
    "script",
    [
        _BROWSER_SCRIPT.replace(
            _BROWSER_ASSIGNMENT, "execSync(rows.map((item) => `${item.name}`));"
        ),
        _BROWSER_SCRIPT.replace(_BROWSER_ASSIGNMENT, "run(rows.map((item) => `${item.name}`));"),
        _BROWSER_SCRIPT.replace(
            _BROWSER_ASSIGNMENT, "const invoke = execSync; invoke(`${data.command}`);"
        ),
        _BROWSER_SCRIPT.replace("${item.name} ::", "${run(item.name)} ::"),
        _BROWSER_SCRIPT.replace(_BROWSER_ASSIGNMENT, "panel.innerHTML = `${data.command}`;"),
        _BROWSER_SCRIPT.replace(_BROWSER_ASSIGNMENT, "escapeText = run;\n" + _BROWSER_ASSIGNMENT),
        _BROWSER_SCRIPT.replace(
            _BROWSER_ASSIGNMENT, "function display(escapeText) {\n" + _BROWSER_ASSIGNMENT + "\n}"
        ),
        _BROWSER_SCRIPT.replace(_BROWSER_ASSIGNMENT, "mutate(data);\n" + _BROWSER_ASSIGNMENT),
        *[
            _BROWSER_SCRIPT.replace(_BROWSER_ASSIGNMENT, call + "\n" + _BROWSER_ASSIGNMENT)
            for call in (
                "external?.(rows);",
                "external?.method(rows);",
                "external.method?.(rows);",
                "external['method']?.(rows);",
                "external?.['method'](rows);",
                "external?.['method']?.(rows);",
                "(external.method)?.(rows);",
                "external /* call */ ?. (rows);",
                "external?.(panel);",
            )
        ],
        _BROWSER_SCRIPT.replace(
            _BROWSER_ASSIGNMENT, "external.map(() => data);\n" + _BROWSER_ASSIGNMENT
        ),
        _BROWSER_SCRIPT.replace(
            _BROWSER_ASSIGNMENT,
            "function leak(value) { external.map(() => value); }\nleak(data);\n"
            + _BROWSER_ASSIGNMENT,
        ),
        _BROWSER_SCRIPT.replace(
            _BROWSER_ASSIGNMENT,
            "function invoke(strings, value) { external.map(() => value); }\ninvoke`${data}`;\n"
            + _BROWSER_ASSIGNMENT,
        ),
        _BROWSER_SCRIPT.replace(
            _BROWSER_ASSIGNMENT,
            "rows.forEach((item) => external.map(() => item));\n" + _BROWSER_ASSIGNMENT,
        ),
        _BROWSER_SCRIPT.replace(
            _BROWSER_ASSIGNMENT, "external.map(() => panel);\n" + _BROWSER_ASSIGNMENT
        ),
        _BROWSER_SCRIPT.replace(
            _BROWSER_ASSIGNMENT,
            "external.map(() => document.getElementById('panel'));\n" + _BROWSER_ASSIGNMENT,
        ),
        _BROWSER_SCRIPT.replace(
            _BROWSER_ASSIGNMENT,
            "const holder = {}; holder.value = data; external.map(() => holder.value);\n"
            + _BROWSER_ASSIGNMENT,
        ),
        _BROWSER_SCRIPT.replace(
            _BROWSER_ASSIGNMENT,
            "const holder = {}; holder.value = panel; external.map(() => holder.value);\n"
            + _BROWSER_ASSIGNMENT,
        ),
        _BROWSER_SCRIPT.replace(
            _BROWSER_ASSIGNMENT,
            "function renderInto(panel) {\n" + _BROWSER_ASSIGNMENT + "\n}\nrenderInto(external);",
        ),
        _BROWSER_SCRIPT.replace(
            _BROWSER_ASSIGNMENT, "rows.forEach((panel) => {\n" + _BROWSER_ASSIGNMENT + "\n});"
        ),
        _BROWSER_SCRIPT.replace("const rows = data.items;", "const rows = {map: eval};"),
        _BROWSER_SCRIPT.replace(
            "const rows = data.items;", "const rows = Object.entries(external);"
        ),
        _BROWSER_SCRIPT.replace(
            _BROWSER_ASSIGNMENT,
            "const terms = external.toLowerCase(); terms.includes(data.query);\n"
            + _BROWSER_ASSIGNMENT,
        ),
        _BROWSER_SCRIPT.replace(
            _BROWSER_ASSIGNMENT,
            "data.query.toLowerCase(data.query).includes(data.query);\n" + _BROWSER_ASSIGNMENT,
        ),
        _BROWSER_SCRIPT.replace(
            "const rows = data.items;",
            "const rows = {[\"map\"]: document.getElementById('p').onclick};",
        ),
        _BROWSER_SCRIPT.replace("<li>${escapeText", '<li onclick="mutate()">${escapeText'),
        _BROWSER_SCRIPT.replace("${listMarkup(rows.map", "${execSync(listMarkup(rows.map").replace(
            "))}</section>", ")))}</section>"
        ),
    ],
    ids=[
        "exec",
        "unknown-call",
        "alias",
        "executable-field",
        "unescaped-output",
        "reassign",
        "shadow",
        "data-escape",
        "optional-call",
        "optional-member-call",
        "member-optional-call",
        "computed-optional-call",
        "optional-computed-member-call",
        "optional-computed-optional-call",
        "grouped-optional-call",
        "commented-optional-call",
        "optional-dom-escape",
        "callback-escape",
        "forwarded-data-escape",
        "tagged-local-escape",
        "forwarded-callback-escape",
        "dom-receiver-escape",
        "dom-lookup-escape",
        "stored-data-escape",
        "stored-dom-escape",
        "dom-function-shadow",
        "dom-callback-shadow",
        "fake-map",
        "unknown-object-entries",
        "unknown-string-receiver",
        "unexpected-string-argument",
        "computed-map",
        "formatter-handler",
        "formatter-executed",
    ],
)
def test_unproven_browser_consumers_remain_partial(script: str) -> None:
    source = _browser_source(script)

    assert python_tokens.python_browser_template_shell_text(source, lambda: None) == source
    assert _python(source) is True
    assert (
        _ledger("scripts/page.py", source)["inspection_ledger"][0]["reason_code"]
        is LedgerReason.STATIC_PARSE_LIMIT
    )


@pytest.mark.parametrize(
    "consumer",
    [
        "import subprocess\nsubprocess.run(render(data), shell=True)",
        "unknown(render(data))",
        "unknown(Path)\nPath('page.html').write_text(render(data))",
        "unknown(json)\nPath('page.html').write_text(render(data))",
        "p = Path('page.html')\nunknown(p)\np.write_text(render(data))",
        "p = Path('page.html')\nunknown(p.write_text)\np.write_text(render(data))",
        "p = Path('page.html')\nunknown(p.resolve)\np.write_text(render(data))",
        "p = Path('page.html')\nunknown(p.parent)\np.write_text(render(data))",
        "p = Path('page.html')\nunknown(p.resolve())\np.write_text(render(data))",
        "Path.resolve = unknown\nPath('page.html').write_text(render(data))",
        "(Path('page.html') / unknown).write_text(render(data))",
        "def outer(Path):\n    def inner():\n        Path('page.html').write_text(render(data))",
        "[Path('page.html').write_text(render(data)) for Path in commands]",
    ],
)
def test_browser_html_requires_explicit_pathlib_consumers(consumer: str) -> None:
    consumer = consumer.replace(
        "write_text(render(data))", "write_text(render(data), encoding='utf-8')"
    )
    source = _browser_source(consumer=consumer)

    assert python_tokens.python_browser_template_shell_text(source, lambda: None) == source
    assert _python(source) is True


@pytest.mark.parametrize(
    "arguments",
    [
        "render(data)",
        "render(data), encoding='utf-16'",
        "render(data), encoding=codec",
        "render(data), encoding='utf-8', errors='ignore'",
        "render(data), 'utf-8', 'ignore'",
    ],
)
def test_browser_html_requires_known_utf8_write(arguments: str) -> None:
    source = _browser_source(consumer=f"Path('page.html').write_text({arguments})")

    assert python_tokens.python_browser_template_shell_text(source, lambda: None) == source
    assert _python(source) is True


@pytest.mark.parametrize(
    "before,after",
    [
        ("label = escape(data['label'])", "label = data['label']"),
        ("payload = json.dumps(data).replace('</', '<\\\\/')", "payload = json.dumps(data)"),
        ("<p>{label}</p>", '<iframe srcdoc="{label}"></iframe>'),
        ("function escapeText(value)", "function String(value)"),
    ],
)
def test_browser_boundary_and_builtin_changes_do_not_gain_ownership(
    before: str, after: str
) -> None:
    source = _browser_source().replace(before, after)

    assert _python(source) is True
    assert python_tokens.python_browser_template_shell_text(source, lambda: None) == source


def test_browser_ownership_preserves_fragment_bounds_and_cancellation() -> None:
    source = _browser_source()
    assert _python(source, complete_context=False) is True
    assert _python(source.replace("</script></body>", "</body>")) is True
    assert _python(_browser_source("(" * 150 + _BROWSER_SCRIPT + ")" * 150)) is True
    oversized = _browser_source("/*" + "x" * 1_100_000 + "*/" + _BROWSER_SCRIPT)
    assert python_tokens.python_browser_template_shell_text(oversized, lambda: None) == oversized

    def cancel() -> None:
        raise TimeoutError("browser ownership deadline")

    with pytest.raises(TimeoutError, match="browser ownership deadline"):
        python_tokens.python_browser_template_shell_text(source, cancel)


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
