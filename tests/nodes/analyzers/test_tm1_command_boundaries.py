# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unsafe options belong to the command that actually supplies them."""

import subprocess
import sys

import pytest

from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm

COMMANDS = [
    ("git push -u origin topic", "git push --force-with-lease origin topic"),
    ("curl https://example.com", "curl --insecure https://example.com"),
    ("curl https://example.com", "curl -k https://example.com"),
    ("wget https://example.com", "wget --no-check-certificate https://example.com"),
    ("chmod u+r file", "chmod a+rwx file"),
    ("chown user file", "chown a+rwx file"),
]


def tm1(source):
    return [
        f for f in tm.analyze(source, "references/commands.md", "markdown") if f.rule_id == "TM1"
    ]


@pytest.mark.parametrize("safe,unsafe", COMMANDS)
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_separate_examples_keep_their_own_locations(safe, unsafe, newline):
    source = newline.join(
        [
            "```powershell",
            "wsl.exe -- " + safe,
            "```",
            "",
            "Explain the option.",
            "",
            "```powershell",
            unsafe,
            "```",
            "",
        ]
    )
    findings = tm1(source)
    assert {f.location.start_line for f in findings} == {8}
    assert any(f.matched_text.startswith(unsafe.split()[0]) for f in findings)
    assert all("Explain" not in f.matched_text for f in findings)


@pytest.mark.parametrize("safe,unsafe", COMMANDS)
@pytest.mark.parametrize("separator", ["\n", "; ", " && ", " || ", " | "])
def test_separate_commands_do_not_share_options(safe, unsafe, separator):
    findings = [
        f for f in tm1(safe + separator + unsafe) if f.matched_text.startswith(unsafe.split()[0])
    ]
    assert len(findings) == 1
    assert unsafe.startswith(findings[0].matched_text)


@pytest.mark.parametrize("safe,unsafe", COMMANDS)
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("continuation", ["\\", "`"])
def test_supported_continuations_keep_the_unsafe_option(safe, unsafe, newline, continuation):
    words = unsafe.split()
    source = words[0] + " " + continuation + newline + " ".join(words[1:])
    findings = [f for f in tm1(source) if f.matched_text.startswith(words[0])]
    assert findings
    assert all(f.location.start_line == 1 for f in findings)


@pytest.mark.parametrize("quote", ["'", '"'])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_balanced_multiline_argument_and_quoted_option(quote, newline):
    source = f'git push {quote}first{newline}second{quote} "--force-with-lease" origin topic'
    findings = tm1(source)
    assert len(findings) == 1
    assert findings[0].location.start_line == 1


def test_balanced_substitution_can_precede_an_option():
    assert tm1('git push $(printf "%s"\n topic) --force origin topic')


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_balanced_backtick_argument_keeps_the_option(newline):
    assert tm1(f'git push `printf "%s"{newline}topic` --force origin topic')


@pytest.mark.parametrize("newline", ["", "\n", "\r\n"])
def test_powershell_escaped_quote_keeps_the_option(newline):
    assert tm1(f'git push "first`" ;{newline} second" --force origin topic')


def test_unclosed_quote_cannot_cross_a_markdown_fence():
    source = '```sh\ngit push "topic\n```\n\n```sh\ngit push --force origin topic\n```\n'
    findings = tm1(source)
    assert [f.location.start_line for f in findings] == [6]


@pytest.mark.parametrize(
    "source",
    [
        "Run `curl https://example.com`\nExplain --insecure here.",
        'git push "`printf topic`"\nThe --force option is dangerous.',
        "curl https://example.com/items/[id\nExplain --insecure here.",
    ],
)
def test_literal_argument_text_does_not_open_a_multiline_command(source):
    assert not tm1(source)


def test_list_indented_fences_keep_their_command_locations():
    source = '- examples:\n\n    ```sh\n    git push "topic\n    ```\n\n    ```sh\n    git push --force origin topic\n    ```\n'
    assert {f.location.start_line for f in tm1(source)} == {8}


def test_node_retains_only_the_actual_force_push():
    source = "```sh\ngit push origin topic\n```\n\n```sh\ngit push --force origin topic\n```\n"
    findings = tm.node({"components": ["commands.md"], "file_cache": {"commands.md": source}})[
        "findings"
    ]
    assert {f.start_line for f in findings if f.rule_id == "TM1"} == {6}


def test_long_quoted_arguments_finish_in_bounded_time():
    script = """
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
import time
started = time.monotonic()
source = 'git push ' + ('"' + 'x' * 60000 + '" ') + '--force origin topic'
assert any(f.rule_id == 'TM1' for f in tm.analyze(source, 'commands.md', 'markdown'))
source = 'git push ' + ('"x" ' * 30000)
assert not any(f.rule_id == 'TM1' for f in tm.analyze(source, 'commands.md', 'markdown'))
assert time.monotonic() - started < 5
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)
