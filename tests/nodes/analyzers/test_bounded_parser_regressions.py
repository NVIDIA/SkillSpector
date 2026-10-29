# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused regressions for bounded parsing of technical documentation."""

from __future__ import annotations

import pytest

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
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
