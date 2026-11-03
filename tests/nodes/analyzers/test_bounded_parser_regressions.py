# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused regressions for bounded parsing of technical documentation."""

from __future__ import annotations

import json
import shlex
from types import SimpleNamespace

import pytest

from skillspector.inspection_ledger import (
    LedgerOutcome,
    LedgerReason,
    ParserCause,
    ParserCheck,
    ParserUncheckedScope,
    StaticParserDiagnostic,
)
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner


def _exhausted(source: str, file_type: str = "markdown") -> bool:
    return tm.has_bounded_parse_exhaustion(source, lambda: None, file_type=file_type)


def _public_result(source: str, path: str):
    return static_runner.run_static_patterns_with_ledger(
        {
            "components": [path],
            "file_cache": {path: source},
            "local_file_cache": {path: source},
        },
        [tm],
    )


def _assert_static_parse_limit(source: str, path: str) -> None:
    result = _public_result(source, path)
    assert any(
        event["outcome"] == LedgerOutcome.PARTIAL
        and event["reason_code"] == LedgerReason.STATIC_PARSE_LIMIT
        for event in result["inspection_ledger"]
    )


@pytest.mark.parametrize(
    "source",
    [
        "CHIP_CONFIG := $($(GET_OUTPUT_DIRS) project/defs/chip)/config.yml\n",
        "  - export CHIP_CONFIG ?= $($(GET_OUTPUT_DIRS) /project/defs/chip)/config.yml\n",
    ],
)
def test_computed_make_value_with_static_path_suffix_stays_conservative(source: str) -> None:
    _assert_static_parse_limit(source, "Makefile")


@pytest.mark.parametrize(
    "source",
    [
        "X := $($(FUNC) -rf /)\n",
        "X := $($(FUNC) --recursive --force /)\n",
        "X := $($(FUNC) value) -rf /\n",
        "X := $($(FUNC) value) ; rm -rf /\n",
        "echo $($(FUNC) value)/config.yml\n",
    ],
)
def test_computed_make_value_does_not_hide_destructive_operands(source: str) -> None:
    assert _exhausted(source)


@pytest.mark.parametrize(
    "source",
    [
        "X := $($(pick) $(printf '\\055\\162\\146') /)\n",
        "X := $($(FUNC) $(OPTS) /)\n",
        "X := $($(FUNC) -rf $(TARGET))\n",
        "X := $($(FUNC) -$(R)$(F) /)\n",
        "X := $($(FUNC) $(DASH)rf /)\n",
        "X := $($(FUNC) $(OPTS)/tmp)\n",
        "X := $($(FUNC) ${ROOT}/defs/chip)\n",
        "X := $($(FUNC) $(ROOT)/$(ARCH)/defs)\n",
    ],
)
def test_computed_make_value_keeps_unresolved_options_conservative(source: str) -> None:
    _assert_static_parse_limit(source, "Makefile")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("$($CMD) -force /\n", False),
        ("$($CMD) -rf /\n", True),
        ("$($CMD) -frv /\n", True),
        ("$($CMD) -r -f /\n", True),
        ("$($CMD) --recursive --force /\n", True),
    ],
)
def test_only_valid_rm_short_option_clusters_are_classified(source: str, expected: bool) -> None:
    assert _exhausted(source, "shell") is expected


@pytest.mark.parametrize("options", ["-rfx", "-rfP", "-rfW"])
@pytest.mark.parametrize("command", ["rm", 'r""m'])
def test_bsd_rm_clusters_keep_destructive_commands_visible(
    options: str,
    command: str,
) -> None:
    result = _public_result(f"{command} {options} *\n", "script.sh")
    assert any(finding.rule_id == "TM1" for finding in result["findings"])


@pytest.mark.parametrize("options", ["-rfx", "-rfP", "-rfW"])
def test_bsd_rm_clusters_keep_runtime_commands_incomplete(options: str) -> None:
    _assert_static_parse_limit(f"$($CMD) {options} /\n", "script.sh")


def test_static_quoted_assignment_word_does_not_own_following_lines() -> None:
    source = (
        "tool -attr 'run_mode=fmod' -fsdb off\n"
        'The diagnostic said can\'t read "::env(CHIPLET)": no such variable.\n'
    )
    assert not _exhausted(source)


def test_dynamic_quoted_assignment_keeps_nested_command_visible() -> None:
    findings = tm.analyze("tool 'MODE=$(rm -rf /)' safe\n", "SKILL.md", "markdown")
    assert any(finding.rule_id == "TM1" for finding in findings)


@pytest.mark.parametrize("wrapper", ["sh -c", "bash -c", "eval"])
def test_quoted_assignment_does_not_hide_executable_wrapper_body(wrapper: str) -> None:
    _assert_static_parse_limit(
        f"""{wrapper} 'MODE=x; r""m -r -f /'\n""",
        "script.sh",
    )


@pytest.mark.parametrize("wrapper", ["bash -c", "eval"])
@pytest.mark.parametrize("separator", [r"\x3b", r"\073", r"\n", r"\012", r"\u003b"])
def test_ansi_c_assignment_does_not_hide_executable_wrapper_body(
    wrapper: str,
    separator: str,
) -> None:
    _assert_static_parse_limit(
        f"{wrapper} $'MODE=x{separator}rm -rf /'\n",
        "script.sh",
    )


@pytest.mark.parametrize("language", ["tcl", "verilog", "systemverilog", "sv", "svh"])
def test_verilog_numeric_apostrophe_is_owned_in_typed_fence(language: str) -> None:
    source = (
        f"```{language}\n"
        "::HW::Utils::program_field $cfg {.*REG} {.*FIELD} 1'b1\n"
        "```\n"
        "| `$cfg program` / `program -force` | Commit staged register writes |"
        " Important runtime boundary inside event |\n"
    )
    assert not _exhausted(source)


@pytest.mark.parametrize("language", ["", "bash"])
def test_verilog_numeric_apostrophe_stays_conservative_outside_typed_fence(
    language: str,
) -> None:
    source = (
        f"```{language}\n"
        "::HW::Utils::program_field $cfg {.*REG} {.*FIELD} 1'b1\n"
        "```\n"
        "| `$cfg program` / `program -force` | Commit staged register writes |"
        " Important runtime boundary inside event |\n"
    )
    assert _exhausted(source)


def test_typed_fence_does_not_hide_destructive_content() -> None:
    projected = tm._markdown_shell_text(
        "```tcl\nset value 1'b1\nrm -rf /\n```\n",
        lambda: None,
    )
    assert "rm -rf /" in projected


@pytest.mark.parametrize("language", ["tcl", "verilog", "systemverilog", "sv", "svh"])
def test_typed_fence_does_not_treat_shell_quote_as_unsized_literal(language: str) -> None:
    _assert_static_parse_limit(
        f"```{language}\n// echo '1'\nsh -c '$CMD -rf /'\n```\n",
        "guide.md",
    )


def test_tcl_exec_body_preserves_embedded_shell_quote_ownership() -> None:
    _assert_static_parse_limit(
        "```tcl\nexec sh -c {echo '1'\nsh -c '$CMD -rf /'\n}\n```\n",
        "guide.md",
    )


@pytest.mark.parametrize("language", ["verilog", "systemverilog", "sv", "svh"])
@pytest.mark.parametrize("wrapper", ['sh -c "$CMD -rf /"', "eval '$CMD -rf /'"])
def test_unsized_literal_does_not_own_later_executable_quote(
    language: str,
    wrapper: str,
) -> None:
    _assert_static_parse_limit(
        f"```{language}\nassign x='1;\n{wrapper}\n```\n",
        "guide.md",
    )


@pytest.mark.parametrize("language", ["verilog", "systemverilog", "sv", "svh"])
def test_unsized_literal_is_owned_in_hdl_expression(language: str) -> None:
    source = f"```{language}\nassign x = '1;\n```\n"
    projected = tm._markdown_shell_text(source, lambda: None)
    assert "assign x =  1;" in projected


@pytest.mark.parametrize(
    "first_lines",
    [
        "echo x=1'b1; harmless'\n",
        "echo x='1; harmless'\n",
        "assign x=1'b1;\nharmless'\n",
    ],
)
def test_hardware_literal_does_not_reassign_later_shell_quote(first_lines: str) -> None:
    _assert_static_parse_limit(
        f"```systemverilog\n{first_lines}sh -c '$CMD -rf /'\n```\n",
        "guide.md",
    )


def test_indented_typed_fence_preserves_offsets_and_dangerous_content() -> None:
    source = "- Example:\n   ```SystemVerilog\n   set value 1'b1\n   rm -rf /\n   ```\n"
    projected = tm._markdown_shell_text(source, lambda: None)
    assert len(projected) == len(source)
    assert "1 b1" in projected
    assert "rm -rf /" in projected


@pytest.mark.parametrize(
    ("source", "cause", "check"),
    [
        ("r" + "a" * 4096, ParserCause.RESOURCE_LIMIT, ParserCheck.COMMAND_WORD_CHARACTERS),
        ("$CMD -rf /", ParserCause.RUNTIME_DEPENDENT, ParserCheck.COMMAND_IDENTITY),
        ("$(printf '%d' 2) -rf /", ParserCause.UNSUPPORTED_SYNTAX, ParserCheck.PRINTF_FORMAT),
        (
            "$(printf '%s' '" + "x" * 250 + "') -rf /",
            ParserCause.RESOURCE_LIMIT,
            ParserCheck.PRINTF_INPUT_CHARACTERS,
        ),
        (
            "$(printf '%s' " + "x " * 33 + ") -rf /",
            ParserCause.RESOURCE_LIMIT,
            ParserCheck.PRINTF_ARGUMENT_COUNT,
        ),
        (
            "$(printf '%s%s' '" + "x" * 40 + "' '" + "y" * 40 + "') -rf /",
            ParserCause.RESOURCE_LIMIT,
            ParserCheck.PRINTF_OUTPUT_CHARACTERS,
        ),
        (
            "rm -rf 'unterminated",
            ParserCause.MALFORMED_INPUT,
            ParserCheck.COMMAND_ARGUMENT_BOUNDARY,
        ),
        ("sh -c 'rm -rf /'", ParserCause.UNSUPPORTED_SYNTAX, ParserCheck.COMMAND_STRING_EFFECTS),
        (
            "BASH_ENV=/tmp/profile sh -c 'echo ok'",
            ParserCause.RUNTIME_DEPENDENT,
            ParserCheck.COMMAND_STRING_RUNTIME,
        ),
        ("eval $PROGRAM", ParserCause.RUNTIME_DEPENDENT, ParserCheck.COMMAND_STRING_RUNTIME),
        (
            "eval <<EOF\necho ok\nEOF",
            ParserCause.UNSUPPORTED_SYNTAX,
            ParserCheck.COMMAND_STRING_RUNTIME,
        ),
        (
            "sh " + "-e " * 16 + "-c 'echo ok'",
            ParserCause.RESOURCE_LIMIT,
            ParserCheck.WRAPPER_ARGUMENTS,
        ),
    ],
)
def test_detailed_parser_reports_the_actual_first_cause(
    source: str, cause: ParserCause, check: ParserCheck
) -> None:
    diagnostic = tm.bounded_parse_diagnostic(source, lambda: None, file_type="shell")
    assert isinstance(diagnostic, StaticParserDiagnostic)
    assert (diagnostic.cause, diagnostic.check) == (cause, check)
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is True
    assert 0 <= diagnostic.start < diagnostic.end <= len(source)
    if cause is not ParserCause.RESOURCE_LIMIT:
        assert diagnostic.measurements == {}


@pytest.mark.parametrize(
    "source", ["", "echo ok\n", "# `printf` is literal\necho ok\n", "rm -f build.log\n"]
)
def test_complete_parser_result_stays_boolean_and_has_no_diagnostic(source: str) -> None:
    assert tm.bounded_parse_diagnostic(source, lambda: None, file_type="shell") is None
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is False


@pytest.mark.parametrize(
    ("source", "file_type"),
    [
        ("tool.exec?.apply(tool, [`${command}`]);", "javascript"),
        ("& $command -rf /", "powershell"),
        ("& @'\n/bin/rm\n'@ -rf /\n", "powershell"),
        ("& (@'\n/bin/rm\n'@) -rf /\n", "powershell"),
        ("$value = '$CMD -rf /'", "powershell"),
        ("# Example\n```powershell\n& $command -rf /\n```\n", "markdown"),
    ],
)
def test_native_language_partial_paths_retain_typed_diagnostics(
    source: str, file_type: str
) -> None:
    diagnostic = tm.bounded_parse_diagnostic(source, lambda: None, file_type=file_type)
    assert isinstance(diagnostic, StaticParserDiagnostic)
    assert 0 <= diagnostic.start < diagnostic.end <= len(source)
    assert diagnostic.full_input_checked is False
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type=file_type) is True


@pytest.mark.parametrize("prefix", ["rm -rf ", "& $command -rf "])
def test_native_powershell_operand_limit_keeps_its_measured_diagnostic(prefix: str) -> None:
    source = prefix + "'" + "x" * (tm._ROOT_GLOB_COMMAND_CHARS + 1) + "'\n"
    diagnostic = tm.bounded_parse_diagnostic(source, lambda: None, file_type="powershell")
    assert isinstance(diagnostic, StaticParserDiagnostic)
    assert diagnostic.cause is ParserCause.RESOURCE_LIMIT
    assert diagnostic.check is ParserCheck.COMMAND_ARGUMENT_CHARACTERS
    assert diagnostic.measurements == {
        "observed_characters": tm._ROOT_GLOB_COMMAND_CHARS + 1,
        "limit_characters": tm._ROOT_GLOB_COMMAND_CHARS,
    }
    assert diagnostic.observed_is_lower_bound is True
    assert diagnostic.full_input_checked is False
    assert 0 <= diagnostic.start < diagnostic.end < len(source)


@pytest.mark.parametrize("opener,closer", [("$(", ")"), ("`", "`")])
def test_substitution_startup_diagnostic_keeps_original_source_coordinates(
    opener: str, closer: str
) -> None:
    source = "echo " + opener + "read -z VALUE; sh -c 'echo ok'" + closer
    commands = list(tm._shell_command_strings(source, lambda: None))
    unresolved = [command for command in commands if command.content is None]
    assert len(unresolved) == 1
    diagnostic = unresolved[0].diagnostic
    assert isinstance(diagnostic, StaticParserDiagnostic)
    assert diagnostic.cause is ParserCause.UNSUPPORTED_SYNTAX
    assert diagnostic.check is ParserCheck.COMMAND_STRING_RUNTIME
    assert diagnostic.start == source.index("read")
    assert diagnostic.start < diagnostic.end <= len(source) - len(closer)


def test_command_word_limit_records_a_prefix_instead_of_an_invented_total() -> None:
    source = "r" + "a" * 20_000
    diagnostic = tm.bounded_parse_diagnostic(source, lambda: None)
    assert diagnostic is not None
    assert diagnostic.measurements == {"observed_characters": 4097, "limit_characters": 4096}
    assert diagnostic.observed_is_lower_bound is True
    assert diagnostic.end == 4097
    assert diagnostic.end < len(source)


@pytest.mark.parametrize("complete_context", [False, True])
def test_missing_quote_requires_complete_source_for_malformed_classification(
    complete_context: bool,
) -> None:
    diagnostic = tm.bounded_parse_diagnostic(
        "rm -rf 'unterminated", lambda: None, file_type="shell", complete_context=complete_context
    )
    assert diagnostic is not None
    assert diagnostic.cause == (
        ParserCause.MALFORMED_INPUT if complete_context else ParserCause.BOUNDARY_UNRESOLVED
    )
    assert diagnostic.full_input_checked is complete_context


def test_operand_boundary_before_budget_is_not_called_malformed_input() -> None:
    source = "rm -rf '" + "x" * 20_000 + "'"
    diagnostics = []
    _tokens, end, exhausted = tm._bounded_shell_tokens(source, 0, 2, diagnostics=diagnostics)
    assert exhausted is True
    assert end == 2 + tm._ROOT_GLOB_COMMAND_CHARS
    diagnostic = diagnostics[0]
    assert diagnostic.cause is ParserCause.BOUNDARY_UNRESOLVED
    assert diagnostic.full_input_checked is False
    assert diagnostic.observed_is_lower_bound is True
    assert diagnostic.measurements == {"observed_characters": 8192, "limit_characters": 8192}


@pytest.mark.parametrize(
    ("word", "check", "observed", "limit", "unit", "lower_bound"),
    [
        ("{" + "a" * 257 + ",b}", ParserCheck.BRACE_CHARACTERS, 261, 256, "characters", False),
        ("{" + ",".join(["a"] * 9) + "}", ParserCheck.BRACE_ALTERNATIVES, 9, 8, "records", False),
        ("{a,b,c,d,e,f,g,h}{a,b,c,d,e}", ParserCheck.BRACE_PRODUCTS, 35, 32, "records", True),
        ("{a,b}{c,d}{e,f}{g,h}{i,j}", ParserCheck.BRACE_ROUNDS, 4, 4, "depth", True),
    ],
)
def test_brace_limits_describe_the_exact_counter_that_stopped(
    word: str, check: ParserCheck, observed: int, limit: int, unit: str, lower_bound: bool
) -> None:
    diagnostics = []
    assert tm._static_brace_expansions(word, diagnostics) is None
    diagnostic = diagnostics[0]
    assert diagnostic.cause is ParserCause.RESOURCE_LIMIT
    assert diagnostic.check is check
    assert diagnostic.measurements == {"observed_" + unit: observed, "limit_" + unit: limit}
    assert diagnostic.observed_is_lower_bound is lower_bound


def test_unsupported_brace_grammar_has_no_fabricated_resource_measurements() -> None:
    diagnostics = []
    assert tm._static_brace_expansions("{a,b{c}}", diagnostics) is None
    assert diagnostics[0].cause is ParserCause.UNSUPPORTED_SYNTAX
    assert diagnostics[0].check is ParserCheck.BRACE_SYNTAX
    assert diagnostics[0].measurements == {}


def test_known_command_string_depth_limit_keeps_its_existing_boolean_stop() -> None:
    diagnostic = tm._has_shell_command_word_exhaustion(
        "sh -c 'echo ok'", lambda: None, _command_string_depth=8
    )
    assert diagnostic is not None
    assert diagnostic.check is ParserCheck.COMMAND_STRING_DEPTH
    assert diagnostic.measurements == {"observed_depth": 8, "limit_depth": 8}
    assert diagnostic.observed_is_lower_bound is True


def test_unsupported_heredoc_does_not_claim_the_remaining_input_was_checked() -> None:
    source = "$(cat <<'E-O-F'\necho ok\nE-O-F\n)" + "a" * 5000
    diagnostic = tm.bounded_parse_diagnostic(source, lambda: None)
    assert diagnostic is not None
    assert diagnostic.cause is ParserCause.UNSUPPORTED_SYNTAX
    assert diagnostic.check is ParserCheck.HEREDOC_HEADER
    assert diagnostic.full_input_checked is False
    assert diagnostic.measurements == {}


def test_heredoc_count_budget_reports_only_the_observed_prefix() -> None:
    source = "$(cat " + " <<END" * 33 + ")"
    diagnostics = []
    assert tm._skip_command_substitution(source, 0, len(source), diagnostics=diagnostics) is None
    diagnostic = diagnostics[0]
    assert diagnostic.check is ParserCheck.HEREDOC_COUNT
    assert diagnostic.measurements == {"observed_records": 33, "limit_records": 32}
    assert diagnostic.observed_is_lower_bound is True


def test_cached_unresolved_boundary_never_invents_full_input_evidence() -> None:
    source = "$(unsupported)"
    diagnostics = []
    assert (
        tm._skip_command_substitution(
            source, 0, len(source), end_cache={0: None}, diagnostics=diagnostics
        )
        is None
    )
    assert diagnostics[0].cause is ParserCause.BOUNDARY_UNRESOLVED
    assert diagnostics[0].full_input_checked is False


def test_startup_proof_budget_is_preserved_as_a_resource_cause() -> None:
    source = "export VALUE=example"
    diagnostics = []
    assert tm._shell_clause_sets_startup(source, 0, lambda: None, [1], diagnostics)
    diagnostic = diagnostics[0]
    assert diagnostic.cause is ParserCause.RESOURCE_LIMIT
    assert diagnostic.check is ParserCheck.STARTUP_PROOF_CHARACTERS
    assert (
        diagnostic.measurements["observed_characters"] > diagnostic.measurements["limit_characters"]
    )
    assert diagnostic.observed_is_lower_bound is True


def test_sliced_startup_body_reports_its_shared_work_budget(monkeypatch) -> None:
    source = "echo `read -z VALUE; sh -c 'echo ok'`"
    original = tm._shell_clause_sets_startup

    def nearly_exhausted_body(content, start, check_runtime, remaining_work, diagnostics, **kwargs):
        if content != source:
            # Earlier clauses have consumed all but one shared work unit.
            remaining_work[0] = 1
        return original(content, start, check_runtime, remaining_work, diagnostics, **kwargs)

    monkeypatch.setattr(tm, "_shell_clause_sets_startup", nearly_exhausted_body)
    unresolved = [
        command
        for command in tm._shell_command_strings(source, lambda: None)
        if command.content is None
    ]
    assert len(unresolved) == 1
    diagnostic = unresolved[0].diagnostic
    assert isinstance(diagnostic, StaticParserDiagnostic)
    limit = 4 * len(source) + tm._ROOT_GLOB_COMMAND_CHARS
    assert diagnostic.check is ParserCheck.STARTUP_PROOF_CHARACTERS
    assert diagnostic.measurements == {"observed_characters": limit + 3, "limit_characters": limit}
    assert diagnostic.start == source.index("read")


def test_unsupported_startup_proof_option_is_not_a_numeric_limit() -> None:
    source = "read -z VALUE; sh -c 'echo ok'"
    diagnostic = tm.bounded_parse_diagnostic(source, lambda: None)
    assert diagnostic is not None
    assert diagnostic.cause is ParserCause.UNSUPPORTED_SYNTAX
    assert diagnostic.check is ParserCheck.COMMAND_STRING_RUNTIME
    assert diagnostic.measurements == {}


@pytest.mark.parametrize("budget", [False, True])
@pytest.mark.parametrize("nesting", [0, 1, 2])
@pytest.mark.parametrize("repeat_eval", [False, True])
def test_eval_retains_the_first_startup_proof_cause_and_original_carrier(
    budget: bool, nesting: int, repeat_eval: bool
) -> None:
    body = (
        "export " + " ".join(f"NAME{index}=0" for index in range(32)) if budget else "read -z VALUE"
    )
    for _ in range(nesting):
        body = "eval " + shlex.quote(body)
    carrier = "eval " + shlex.quote(body)
    source = carrier + ("; eval 'echo ok'" if repeat_eval else "") + "; sh -c 'echo ok'"
    result = _public_result(source, "script.sh")
    event = result["inspection_ledger"][0]
    diagnostic = event["parser_diagnostic"]
    assert event["outcome"] is LedgerOutcome.PARTIAL
    assert event["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
    assert diagnostic["source_start_offset"] == 0
    assert diagnostic["source_end_offset"] == len(carrier)
    assert diagnostic["location_precision"] == "enclosing_expression"
    assert diagnostic["full_input_checked"] is False
    assert body not in json.dumps(diagnostic)
    if budget:
        assert diagnostic["cause"] is ParserCause.RESOURCE_LIMIT
        assert diagnostic["check"] is ParserCheck.WRAPPER_ARGUMENTS
        assert diagnostic["observed_records"] == diagnostic["limit_records"] == 32
        assert diagnostic["observed_is_lower_bound"] is True
    else:
        assert diagnostic["cause"] is ParserCause.UNSUPPORTED_SYNTAX
        assert diagnostic["check"] is ParserCheck.COMMAND_STRING_RUNTIME
        assert "observed_records" not in diagnostic
        assert "limit_records" not in diagnostic


@pytest.mark.parametrize("budget", [False, True])
@pytest.mark.parametrize("nested_eval", [False, True])
def test_child_shell_startup_proof_cause_does_not_leak_into_outer_siblings(
    budget: bool, nested_eval: bool
) -> None:
    body = (
        "export " + " ".join(f"NAME{index}=0" for index in range(32)) if budget else "read -z VALUE"
    )
    if nested_eval:
        body = "eval " + shlex.quote(body)
    source = "sh -c " + shlex.quote(body) + "; sh -c 'echo ok'"
    assert tm.bounded_parse_diagnostic(source, lambda: None) is None
    assert tm.has_bounded_parse_exhaustion(source, lambda: None) is False


@pytest.mark.parametrize("budget", [False, True])
@pytest.mark.parametrize("consumer_nesting", [0, 1, 2])
def test_later_nested_eval_consumer_keeps_the_first_proof_origin(
    budget: bool, consumer_nesting: int
) -> None:
    body = (
        "export " + " ".join(f"NAME{index}=0" for index in range(32)) if budget else "read -z VALUE"
    )
    first_carrier = "eval " + shlex.quote(body)
    consumer = "sh -c 'echo ok'"
    for _ in range(consumer_nesting + 1):
        consumer = "eval " + shlex.quote(consumer)
    source = first_carrier + "; " + consumer
    diagnostic = _public_result(source, "script.sh")["inspection_ledger"][0]["parser_diagnostic"]
    assert diagnostic["source_start_offset"] == 0
    assert diagnostic["source_end_offset"] == len(first_carrier)
    assert diagnostic["location_precision"] == "enclosing_expression"
    if budget:
        assert diagnostic["cause"] is ParserCause.RESOURCE_LIMIT
        assert diagnostic["check"] is ParserCheck.WRAPPER_ARGUMENTS
        assert diagnostic["observed_records"] == diagnostic["limit_records"] == 32
        assert diagnostic["observed_is_lower_bound"] is True
    else:
        assert diagnostic["cause"] is ParserCause.UNSUPPORTED_SYNTAX
        assert diagnostic["check"] is ParserCheck.COMMAND_STRING_RUNTIME
        assert "observed_records" not in diagnostic


@pytest.mark.parametrize("budget", [False, True])
@pytest.mark.parametrize("consumer_nesting", [0, 1, 2])
def test_new_limitation_inside_later_eval_keeps_its_own_cause_and_carrier(
    budget: bool, consumer_nesting: int
) -> None:
    body = (
        "export " + " ".join(f"NAME{index}=0" for index in range(32)) if budget else "read -z VALUE"
    )
    first_carrier = "eval " + shlex.quote(body)
    consumer = "r" * 4097 + "; sh -c 'echo ok'"
    for _ in range(consumer_nesting + 1):
        consumer = "eval " + shlex.quote(consumer)
    source = first_carrier + "; " + consumer
    diagnostic = _public_result(source, "script.sh")["inspection_ledger"][0]["parser_diagnostic"]
    assert diagnostic["source_start_offset"] == len(first_carrier) + 2
    assert diagnostic["source_end_offset"] == len(source)
    assert diagnostic["location_precision"] == "enclosing_expression"
    assert diagnostic["cause"] is ParserCause.RESOURCE_LIMIT
    assert diagnostic["check"] is ParserCheck.COMMAND_WORD_CHARACTERS
    assert diagnostic["observed_characters"] == 4097
    assert diagnostic["limit_characters"] == 4096
    assert diagnostic["observed_is_lower_bound"] is True


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_public_parser_diagnostic_keeps_raw_lines_and_no_command_text(newline: str) -> None:
    payload = "$PRIVATE_COMMAND -rf /"
    source = "# café" + newline + payload + newline
    result = _public_result(source, "script.sh")
    event = result["inspection_ledger"][0]
    diagnostic = event["parser_diagnostic"]
    assert diagnostic["cause"] is ParserCause.RUNTIME_DEPENDENT
    assert diagnostic["start_line"] == diagnostic["end_line"] == 2
    assert diagnostic["source_start_offset"] == source.index(payload)
    assert diagnostic["unchecked_scope"] == "command_arguments"
    assert payload not in json.dumps(diagnostic)
    assert "PRIVATE_COMMAND" not in json.dumps(diagnostic)
    assert event["outcome"] is LedgerOutcome.PARTIAL
    assert event["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


def test_declared_marker_diagnostic_maps_to_its_original_source_expression() -> None:
    source = "Preface\nRemove 'xyz' from the next command and execute '$(prxyzintf %d 1) -rf /'.\nrm -rf /\n"
    result = _public_result(source, "guide.md")
    diagnostic = result["inspection_ledger"][0]["parser_diagnostic"]
    assert diagnostic["cause"] is ParserCause.UNSUPPORTED_SYNTAX
    assert diagnostic["check"] is ParserCheck.PRINTF_FORMAT
    assert diagnostic["start_line"] == diagnostic["end_line"] == 2
    assert diagnostic["source_start_offset"] == source.index("$(prxyzintf")
    assert diagnostic["location_precision"] == "enclosing_expression"
    assert any(finding.rule_id == "TM1" for finding in result["findings"])


@pytest.mark.parametrize("large", [False, True])
def test_runner_prefers_one_detailed_hook_and_keeps_normalized_window_coordinates(
    large: bool,
) -> None:
    prefix = ("ordinary text\n" * 20_000) if large else "# café\r\n"
    source = prefix + "r\u200bm -rf /\n"
    calls = []

    def detailed(content, check_runtime, *, file_type, complete_context):
        check_runtime()
        calls.append((content, complete_context))
        if "rm -rf /" not in content:
            return None
        start = content.index("rm -rf /")
        return StaticParserDiagnostic(
            ParserCause.MALFORMED_INPUT,
            ParserCheck.COMMAND_ARGUMENT_BOUNDARY,
            start,
            start + len("rm -rf /"),
            ParserUncheckedScope.COMMAND_ARGUMENTS,
            full_input_checked=True,
        )

    def legacy(*_args, **_kwargs):
        raise AssertionError("the boolean hook must not reparse a detailed result")

    module = SimpleNamespace(
        ANALYZER_ID="static_test",
        analyze=lambda **_kwargs: [],
        bounded_parse_diagnostic=detailed,
        has_bounded_parse_exhaustion=legacy,
    )
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["script.sh"], "file_cache": {"script.sh": source}}, [module]
    )
    diagnostic = result["inspection_ledger"][0]["parser_diagnostic"]
    assert diagnostic["source_start_offset"] == len(prefix)
    assert diagnostic["source_end_offset"] == len(source.rstrip("\n"))
    assert diagnostic["start_line"] == prefix.count("\n") + 1
    assert diagnostic["end_line"] == diagnostic["start_line"]
    assert diagnostic["cause"] is ParserCause.BOUNDARY_UNRESOLVED
    assert diagnostic["location_precision"] == "enclosing_expression"
    assert diagnostic["full_input_checked"] is (not large)
    assert sum("rm -rf /" in content for content, _complete in calls) == 1


def test_parser_diagnostic_survives_a_later_resource_limit_without_hiding_findings(
    monkeypatch,
) -> None:
    from skillspector.models import Finding

    finding = Finding(rule_id="TM1", message="retained finding", file="script.sh")
    diagnostic = StaticParserDiagnostic(
        ParserCause.RUNTIME_DEPENDENT,
        ParserCheck.COMMAND_IDENTITY,
        0,
        10,
        ParserUncheckedScope.COMMAND_ARGUMENTS,
    )
    metrics = static_runner._StaticScanMetrics({"observed_seconds": 2.0, "limit_seconds": 1.0})
    metrics.parser_diagnostic = diagnostic
    monkeypatch.setattr(
        static_runner,
        "_scan_all_views_detailed",
        lambda *_args, **_kwargs: ([finding], LedgerReason.RUNTIME_LIMIT, metrics),
    )
    module = SimpleNamespace(ANALYZER_ID="static_test", analyze=lambda **_kwargs: [])
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["script.sh"], "file_cache": {"script.sh": "$CMD -rf /"}}, [module]
    )
    event = result["inspection_ledger"][0]
    assert event["reason_code"] is LedgerReason.RUNTIME_LIMIT
    assert event["observed_seconds"] == 2.0
    assert event["parser_diagnostic"]["cause"] is ParserCause.RUNTIME_DEPENDENT
    assert event["emitted_finding_ids"] == [finding.finding_id]


@pytest.mark.parametrize("with_parser_gap", [False, True])
def test_output_reconciliation_only_clears_resolved_output_incompleteness(
    monkeypatch, with_parser_gap: bool
) -> None:
    from skillspector.models import Finding

    finding = Finding(rule_id="TM1", message="retained finding", file="script.sh")
    metrics = static_runner._StaticScanMetrics({"observed_findings": 20, "limit_findings": 10})
    metrics.fully_scanned_aggregate_overflow = True
    if with_parser_gap:
        metrics.parser_diagnostic = StaticParserDiagnostic(
            ParserCause.RUNTIME_DEPENDENT,
            ParserCheck.COMMAND_IDENTITY,
            0,
            10,
            ParserUncheckedScope.COMMAND_ARGUMENTS,
        )
    monkeypatch.setattr(
        static_runner,
        "_scan_all_views_detailed",
        lambda *_args, **_kwargs: ([finding, finding], LedgerReason.OUTPUT_LIMIT, metrics),
    )
    module = SimpleNamespace(
        ANALYZER_ID="static_test",
        analyze=lambda **_kwargs: [],
        postprocess_path_findings=lambda _content, _findings: [finding],
    )
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["script.sh"], "file_cache": {"script.sh": "$CMD -rf /"}}, [module]
    )
    event = result["inspection_ledger"][0]
    assert event["emitted_finding_ids"] == [finding.finding_id]
    assert result["findings"] == [finding]
    assert "observed_findings" not in event
    if with_parser_gap:
        assert event["outcome"] is LedgerOutcome.PARTIAL
        assert event["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
        assert event["parser_diagnostic"]["cause"] is ParserCause.RUNTIME_DEPENDENT
    else:
        assert event["outcome"] is LedgerOutcome.COMPLETED
        assert "reason_code" not in event
        assert "parser_diagnostic" not in event
