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

import pytest

from skillspector.models import Finding
from skillspector.nodes.report import _clean_text, _sanitize_finding, report
from skillspector.state import SkillspectorState


def _dirty_finding() -> Finding:
    return Finding(
        rule_id="E2",
        message="creds \x1b[31mleak\x1b[0m here\x00",
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
    assert "\x1b" not in (cleaned.remediation or "")
    assert "\x07" not in (cleaned.context or "")
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
        finding=url,
        explanation=url,
        remediation=url,
        context=url,
        matched_text=url,
        code_snippet=url,
        evidence={"destination": url},
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
    serialized_findings = json.dumps([item.to_dict() for item in result["filtered_findings"]])
    for secret in (username, password, token):
        assert secret not in rendered
        assert secret not in serialized_findings
