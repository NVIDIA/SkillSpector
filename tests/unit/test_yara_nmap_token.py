# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the offensive-tool nmap command signature."""

from __future__ import annotations

from pathlib import Path

import pytest
import yara

from skillspector.nodes.analyzers import static_yara

RULES_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "skillspector" / "yara_rules" / "hacktools.yar"
)
_OTHER_ENCODED_SEPARATORS = (
    r"\r",
    r"\v",
    r"\f",
    r"\x0d",
    r"\x0b",
    r"\x0c",
    r"\x26",
    r"\x3b",
    r"\x7c",
    r"\u000d",
    r"\u000b",
    r"\u000c",
    r"\u0026",
    r"\u003b",
    r"\u007c",
    r"\U0000000d",
    r"\U0000000b",
    r"\U0000000c",
    r"\U00000026",
    r"\U0000003b",
    r"\U0000007c",
    r"\015",
    r"\013",
    r"\014",
    r"\046",
    r"\073",
    r"\174",
)
_ENCODED_OPEN_PARENS = (r"\x28", r"\u0028", r"\U00000028", r"\050")


def _matched_identifiers(data: bytes) -> set[str]:
    identifiers: set[str] = set()
    for match in yara.compile(filepath=str(RULES_PATH)).match(data=data):
        identifiers.update(string.identifier for string in match.strings)
    return identifiers


def test_nmap_signature_ignores_identifier_suffix_across_lines() -> None:
    for sample in (
        b"set noAtePinMap\n  -skipScanSetup true\n",
        b"hasNmapConfiguration -sV enabled\n",
        b"run_nmap -sS x\n",
    ):
        assert "$nmap_scan" not in _matched_identifiers(sample)


def test_nmap_signature_still_matches_command_tokens() -> None:
    for sample in (b"nmap -sS host\n", b"sudo nmap -p 443 host\n", b";nmap -A host\n"):
        assert "$nmap_scan" in _matched_identifiers(sample), sample


def test_nmap_signature_reports_the_command_line() -> None:
    content = "#!/bin/sh\nnmap -sS host\n"
    result = static_yara.node({"components": ["script.sh"], "file_cache": {"script.sh": content}})
    finding = next(
        finding for finding in result["findings"] if "offensive_tool_references" in finding.message
    )

    assert finding.start_line == 2


@pytest.mark.parametrize("separator", [r"\n", r"\x0a", r"\u000a", r"\U0000000a", r"\012"])
def test_nmap_signature_matches_after_encoded_newline(separator: str) -> None:
    content = f'import os\nos.system("echo ready{separator}nmap -sV localhost")\n'
    result = static_yara.node({"components": ["install.py"], "file_cache": {"install.py": content}})
    finding = next(
        finding for finding in result["findings"] if "offensive_tool_references" in finding.message
    )

    assert finding.start_line == 2


@pytest.mark.parametrize("separator", [r"\n", r"\x0a", r"\u000a", r"\U0000000a", r"\012"])
def test_nmap_signature_ignores_double_escaped_newline(separator: str) -> None:
    content = f'print("\\{separator}nmap -sV localhost")\n'
    result = static_yara.node({"components": ["install.py"], "file_cache": {"install.py": content}})

    assert not any("offensive_tool_references" in finding.message for finding in result["findings"])


@pytest.mark.parametrize(
    "separator",
    [
        r"\t",
        r"\x09",
        r"\u0009",
        r"\U00000009",
        r"\011",
        r"\x20",
        r"\u0020",
        r"\U00000020",
        r"\040",
    ],
)
def test_nmap_signature_matches_after_encoded_shell_whitespace(separator: str) -> None:
    content = f'import os\nos.system("echo ready;{separator}nmap -sV localhost")\n'
    result = static_yara.node({"components": ["install.py"], "file_cache": {"install.py": content}})
    finding = next(
        finding for finding in result["findings"] if "offensive_tool_references" in finding.message
    )

    assert finding.start_line == 2


@pytest.mark.parametrize(
    "separator",
    [
        r"\t",
        r"\x09",
        r"\u0009",
        r"\U00000009",
        r"\011",
        r"\x20",
        r"\u0020",
        r"\U00000020",
        r"\040",
    ],
)
def test_nmap_signature_ignores_double_escaped_shell_whitespace(separator: str) -> None:
    content = f'print("\\{separator}nmap -sV localhost")\n'
    result = static_yara.node({"components": ["install.py"], "file_cache": {"install.py": content}})

    assert not any("offensive_tool_references" in finding.message for finding in result["findings"])


@pytest.mark.parametrize("separator", _OTHER_ENCODED_SEPARATORS)
def test_nmap_signature_matches_after_other_encoded_separators(separator: str) -> None:
    content = f'import os\nos.system("echo ready{separator}nmap -sV localhost")\n'
    result = static_yara.node({"components": ["install.py"], "file_cache": {"install.py": content}})
    finding = next(
        finding for finding in result["findings"] if "offensive_tool_references" in finding.message
    )

    assert finding.start_line == 2


@pytest.mark.parametrize("separator", _OTHER_ENCODED_SEPARATORS)
def test_nmap_signature_ignores_other_double_escaped_separators(separator: str) -> None:
    content = f'print("\\{separator}nmap -sV localhost")\n'
    result = static_yara.node({"components": ["install.py"], "file_cache": {"install.py": content}})

    assert not any("offensive_tool_references" in finding.message for finding in result["findings"])


@pytest.mark.parametrize("separator", _ENCODED_OPEN_PARENS)
def test_nmap_signature_matches_after_encoded_subshell_opener(separator: str) -> None:
    content = f'import os\nos.system("echo ready;{separator}nmap -sV localhost)")\n'
    result = static_yara.node({"components": ["install.py"], "file_cache": {"install.py": content}})
    finding = next(
        finding for finding in result["findings"] if "offensive_tool_references" in finding.message
    )

    assert finding.start_line == 2


@pytest.mark.parametrize("separator", _ENCODED_OPEN_PARENS)
def test_nmap_signature_ignores_double_escaped_subshell_opener(separator: str) -> None:
    content = f'print("\\{separator}nmap -sV localhost)")\n'
    result = static_yara.node({"components": ["install.py"], "file_cache": {"install.py": content}})

    assert not any("offensive_tool_references" in finding.message for finding in result["findings"])
