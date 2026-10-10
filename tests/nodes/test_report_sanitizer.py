# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for report-output sanitization (ANSI / control-byte stripping)."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from markdown_it import MarkdownIt

from skillspector.llm_analyzer_base import LLMFinding
from skillspector.models import Finding
from skillspector.nodes.report import _clean_text, _format_markdown, _sanitize_finding, report
from skillspector.state import SkillspectorState
from skillspector.suppression import SuppressedFinding


def _dirty_finding() -> Finding:
    return Finding(
        rule_id="E2",
        message="creds \x1b[31mleak\x1b[0m here\x00",
        pattern="pattern \x1b[31mleak\x1b[0m here\x00",
        severity="HIGH",
        confidence=0.9,
        file="a/SKILL.md",
        start_line=5,
        remediation="redact \x1b[1mnow\x1b[0m",
        context="line with \x07 bell and \x1b[0m reset",
    )


def test_clean_text_strips_ansi_and_control_keeps_readable() -> None:
    assert _clean_text("a\x1b[31mb\x1b[0mc\x00d") == "abcd"
    # Tabs and newlines are preserved.
    assert _clean_text("a\tb\nc") == "a\tb\nc"
    # Emoji / multibyte UTF-8 is untouched.
    assert _clean_text("🔴 HIGH") == "🔴 HIGH"
    # Non-strings pass through.
    assert _clean_text(None) is None


def test_sanitize_finding_cleans_text_fields_only() -> None:
    cleaned = _sanitize_finding(_dirty_finding())
    assert "\x1b" not in cleaned.message and "\x00" not in cleaned.message
    assert "leak" in cleaned.message and "here" in cleaned.message
    assert cleaned.pattern == "pattern leak here"
    assert "\x1b" not in (cleaned.remediation or "")
    assert "\x07" not in (cleaned.context or "")
    assert "\x1b" not in (cleaned.pattern or "") and "\x00" not in (cleaned.pattern or "")
    # Non-text fields are unchanged.
    assert cleaned.rule_id == "E2"
    assert cleaned.start_line == 5


@pytest.mark.parametrize("fmt", ["markdown", "json", "sarif", "terminal"])
def test_report_emits_clean_utf8_for_all_formats(fmt: str) -> None:
    """No ANSI/control bytes leak into any report format."""
    state: SkillspectorState = {
        "filtered_findings": [_dirty_finding()],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "skill_path": None,
        "output_format": fmt,
    }
    body = report(state)["report_body"]
    assert "\x00" not in body, f"NUL leaked into {fmt}"
    assert "\x1b" not in body, f"ESC leaked into {fmt}"
    # The readable content survives the sanitization.
    assert "leak" in body and "here" in body


@pytest.mark.parametrize("fmt", ["markdown", "json", "sarif", "terminal"])
@pytest.mark.parametrize("scheme", ["https", "ssh", "git+https", "sparse+https"])
def test_report_redacts_url_credentials_from_every_finding_field(fmt: str, scheme: str) -> None:
    username = "output-user-sentinel"
    password = "output-password-sentinel"
    token = "output-token-sentinel"
    url = f"{scheme}://{username}:{password}@packages.example.invalid/?token={token}"
    finding = Finding(
        rule_id="E2",
        message=f"credential-bearing destination {url}",
        severity="HIGH",
        confidence=0.9,
        file="setup.sh",
        start_line=1,
        pattern=url,
        finding=url,
        explanation=url,
        remediation=url,
        context=url,
        matched_text=url,
        code_snippet=url,
        evidence={
            "destination": url,
            "redirects": [{"nested": {url: [url, {"destination": url}]}}],
            "tuple": (url, {"destination": url}),
        },
    )
    state: SkillspectorState = {
        "filtered_findings": [finding],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "skill_path": None,
        "output_format": fmt,
    }

    result = report(state)
    rendered = result["report_body"]
    if fmt == "markdown":
        rendered = MarkdownIt().enable("table").render(rendered)
    serialized_findings = json.dumps([item.to_dict() for item in result["filtered_findings"]])
    for secret in (username, password, token):
        assert secret not in rendered
        assert secret not in serialized_findings


@pytest.mark.parametrize("fmt", ["json", "sarif"])
def test_report_sanitizes_llm_message_copied_to_pattern(fmt: str) -> None:
    message = "review \x1b[31mhttps://user:secret@example.invalid/?token=secret-token\x1b[0m"
    finding = LLMFinding(
        rule_id="E1",
        message=message,
        severity="HIGH",
        start_line=1,
    ).to_finding("SKILL.md")
    state: SkillspectorState = {
        "filtered_findings": [finding],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "skill_path": None,
        "output_format": fmt,
    }

    result = report(state)
    rendered = result["report_body"]
    serialized_findings = json.dumps([item.to_dict() for item in result["filtered_findings"]])
    assert "\x1b" not in rendered
    assert "secret" not in rendered
    assert "secret" not in serialized_findings


def test_nested_evidence_preserves_scalar_types_and_original_finding() -> None:
    scalar_values = [None, True, False, 42, 1.25]
    finding = _dirty_finding()
    finding.evidence = {
        "nested": [{"values": scalar_values, "dirty\x00key": "readable\x1b[31m text\x00"}],
        "tuple": (None, True, 42),
    }

    cleaned = _sanitize_finding(finding)

    assert cleaned.evidence == {
        "nested": [{"values": scalar_values, "dirtykey": "readable text"}],
        "tuple": (None, True, 42),
    }
    for actual, original in zip(
        cleaned.evidence["nested"][0]["values"], scalar_values, strict=True
    ):
        assert type(actual) is type(original)
    assert "dirty\x00key" in finding.evidence["nested"][0]
    assert "\x1b" in finding.evidence["nested"][0]["dirty\x00key"]


@pytest.mark.parametrize(
    "payload",
    [
        "safe` | LOW |\r\n## Issues (0)\rNo security issues detected.\n<!--",
        "` `` ``` <script>alert(1)</script> [safe](https://example.invalid)",
        "\\| **safe** &lt;!--",
        "~~hidden~~",
        "\x1b[2J\x00\u202eLOW\u202c\x9b\u2066safe\u2069",
    ],
)
@pytest.mark.parametrize(
    "field",
    [
        "name",
        "skill_path",
        "degraded_notice",
        "component_path",
        "component_type",
        "rule_id",
        "severity",
        "message",
        "remediation",
        "file",
        "source_url",
        "evidence_key",
        "evidence_value",
        "summary_id",
        "summary_message",
        "summary_file",
        "summary_protocol",
        "suppressed_reason",
        "suppressed_file",
        "ledger_path",
        "ledger_message",
        "ledger_reason_code",
        "exclude_pattern",
        "limitation",
    ],
)
def test_markdown_report_contains_untrusted_fields(payload: str, field: str) -> None:
    finding = Finding(
        rule_id="P1",
        message="finding",
        severity="HIGH",
        file="run.py",
        source_url="https://example.invalid",
        remediation="review",
        evidence={"match": "value"},
    )
    suppressed = SuppressedFinding(
        Finding(rule_id="P2", message="suppressed", file="notes.md"), "reviewed"
    )
    arguments = {
        "findings": [finding],
        "component_metadata": [{"path": "run.py", "type": "python", "lines": 1}],
        "manifest": {"name": "sample"},
        "skill_path": "sample",
        "risk_score": 90,
        "risk_severity": "CRITICAL",
        "risk_recommendation": "DO_NOT_INSTALL",
        "has_executable_scripts": True,
        "use_llm": False,
        "degraded_notice": "notice",
        "structured_summaries": [
            {"id": "SSR-1", "message": "summary", "file": "run.py", "protocol": "mcp"}
        ],
        "suppressed": [suppressed],
        "show_suppressed": True,
        "analysis_completeness": {
            "ledger_exceptions": [
                {"reason_code": "partial", "path": "run.py", "message": "inspect"}
            ],
            "exclude_patterns": ["cache"],
            "limitations": ["limitation"],
        },
    }
    parser = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
    original_blocks = [token.type for token in parser.parse(_format_markdown(**arguments))]
    if field == "name":
        arguments["manifest"]["name"] = payload
    elif field in {"skill_path", "degraded_notice"}:
        arguments[field] = payload
    elif field.startswith("component_"):
        arguments["component_metadata"][0][field.removeprefix("component_")] = payload
    elif field.startswith("summary_"):
        arguments["structured_summaries"][0][field.removeprefix("summary_")] = payload
    elif field == "suppressed_reason":
        arguments["suppressed"] = [SuppressedFinding(suppressed.finding, payload)]
    elif field == "suppressed_file":
        suppressed.finding.file = payload
    elif field.startswith("ledger_"):
        arguments["analysis_completeness"]["ledger_exceptions"][0][
            field.removeprefix("ledger_")
        ] = payload
    elif field in {"exclude_pattern", "limitation"}:
        arguments["analysis_completeness"][
            "exclude_patterns" if field == "exclude_pattern" else "limitations"
        ] = [payload]
    elif field == "evidence_key":
        finding.evidence = {payload: "value"}
    elif field == "evidence_value":
        finding.evidence = {"match": payload}
    else:
        setattr(finding, field, payload)
    original = deepcopy(arguments)

    rendered = _format_markdown(**arguments)
    tokens = parser.parse(rendered)

    assert all(character.isprintable() or character in "\n\t" for character in rendered)
    assert [token.type for token in tokens] == original_blocks
    for token in tokens:
        assert token.type not in {"html_block", "fence", "code_block"}
        assert not any(
            child.type in {"html_inline", "link_open", "image", "s_open"}
            for child in token.children or []
        )
    assert arguments == original


@pytest.mark.parametrize("value", ["a|b", r"a\|b", "`a`", "a``b`", "<script> & value"])
@pytest.mark.parametrize("table_cell", [False, True])
def test_markdown_code_preserves_literal_values(value: str, table_cell: bool) -> None:
    from skillspector.nodes.report import _markdown_code

    source = _markdown_code(value, table_cell=table_cell)
    if table_cell:
        source = f"| Path |\n|---|\n| {source} |"
    tokens = MarkdownIt("commonmark").enable("table").parse(source)
    code = [
        child.content
        for token in tokens
        for child in token.children or []
        if child.type == "code_inline"
    ]
    assert code == [value]
