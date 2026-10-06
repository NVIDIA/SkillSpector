# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Valid Python and Perl syntax must not exhaust the bounded shell parser.

Python string and comment tokens own their bytes, and Perl ``eval BLOCK`` is
exception handling rather than a string reparse. Neither may turn ordinary
host source into a ``static_parse_limit`` result, while genuinely unresolved
shell text keeps its conservative partial outcome.
"""

from __future__ import annotations

import threading
import warnings
from pathlib import Path

import pytest

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

    def parse() -> None:
        barrier.wait()
        for _ in range(10):
            proven.append(tm._python_literal_spans(_WARNING_SOURCE, lambda: None) is not None)

    threads = [threading.Thread(target=parse) for _ in range(8)]
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
