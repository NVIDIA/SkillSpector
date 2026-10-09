# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the reported file surface of a finding."""

from __future__ import annotations

import json

import pytest

from skillspector.models import AnalyzerFinding, Finding, Location, Severity
from skillspector.nodes.analyzers import (
    static_patterns_data_exfiltration as data_exfiltration_module,
)
from skillspector.nodes.analyzers import static_runner
from skillspector.nodes.analyzers.static_runner import analyzer_finding_to_finding
from skillspector.nodes.deduplicate import deduplicate
from skillspector.nodes.report import _build_sarif_properties, _expand_occurrences, _format_json
from skillspector.surface import (
    CODE,
    COMMENTS,
    CONFIG,
    DOCS,
    INSTRUCTIONS,
    SURFACES,
    TESTS,
    infer_surface,
)


@pytest.mark.parametrize(
    ("file_path", "expected"),
    [
        ("SKILL.md", INSTRUCTIONS),
        ("skills/pdf/Skill.MD", INSTRUCTIONS),
        ("src/cli.py", CODE),
        ("scripts/install.sh", CODE),
        ("tests/test_runner.py", TESTS),
        ("pkg/__tests__/widget.js", TESTS),
        ("runner_test.go", TESTS),
        ("README.md", DOCS),
        ("docs/getting-started.md", DOCS),
        ("LICENSE", DOCS),
        (".mcp.json", CONFIG),
        ("config/settings.yaml", CONFIG),
        ("requirements.txt", CONFIG),
        ("requirements-dev.txt", CONFIG),
        ("requirements/runtime.txt", CONFIG),
        ("docs/install.sh", CODE),
        ("config/hook.py", CODE),
        (".env.local", CONFIG),
    ],
)
def test_infer_surface_classifies_the_path(file_path: str, expected: str) -> None:
    """Every documented surface is reachable from a path alone."""
    assert expected in SURFACES
    assert infer_surface(file_path) == expected


@pytest.mark.parametrize(
    ("file_path", "line_text", "expected"),
    [
        ("src/main.py", "    # indented", COMMENTS),
        ("src/main.py", "value = 1", CODE),
        ("app.js", "// note", COMMENTS),
        ("app.js", "/* comment", COMMENTS),
        ("app.js", "/**/ eval(x)", CODE),
        ("app.js", "*/ eval(x)", CODE),
        ("app.js", " * continuation", CODE),
        ("app.js", "const value = 1", CODE),
        ("page.html", "<!-- comment", COMMENTS),
        ("page.html", "<!-- x --><script>eval(x)</script>", CODE),
        ("script.lua", "-- note", COMMENTS),
        ("script.lua", "--[[x]] os.execute('id')", CODE),
        ("script.ps1", "#> Invoke-Expression $payload", CODE),
        ("script.ps1", "##> Invoke-Expression $payload", CODE),
        ("script.ps1", "# note #> Invoke-Expression $payload", CODE),
        ("script.php", "// note ?><?php system($_GET['c']); ?>", CODE),
        ("script.lua", "--[=[ x ]=] os.execute('id')", CODE),
        ("page.html", "<!--><script>eval(x)</script>", CODE),
        ("page.html", "<!---><script>eval(x)</script>", CODE),
        ("page.html", "<!-- x --!><script>eval(x)</script>", CODE),
        ("query.sql", "-- note", COMMENTS),
        # a comment inside a config file is still a comment
        ("setup.cfg", "# pinned", COMMENTS),
        ("config.yaml", "key: value", CONFIG),
        # docs and instructions never become comments
        ("README.md", "<!-- hidden -->", DOCS),
        ("SKILL.md", "<!-- hidden -->", INSTRUCTIONS),
        # a comment inside a test file stays a test
        ("tests/test_a.py", "# comment", TESTS),
    ],
)
def test_comment_refines_code_and_config_only(
    file_path: str, line_text: str, expected: str
) -> None:
    """The matched line refines code/config paths and leaves other labels alone."""
    assert infer_surface(file_path, line_text) == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("# only comment\n# still comment\n", COMMENTS),
        ("# first line\npayload = 1\n", CODE),
    ],
)
def test_multiline_finding_requires_every_line_to_be_comment(content: str, expected: str) -> None:
    """A multi-line finding is comments only when all covered lines are comments."""
    finding = AnalyzerFinding(
        rule_id="RP1",
        message="multi-line",
        severity=Severity.MEDIUM,
        location=Location(file="src/main.py", start_line=1, end_line=2),
    )
    converted = static_runner._convert_analyzer_finding(
        finding,
        path="src/main.py",
        file_type="python",
        content=content,
        content_lines=content.splitlines(),
        normalized_license_lines=None,
    )

    assert converted is not None
    assert converted.surface == expected


def test_ast_line_numbers_ignore_non_cpython_line_breaks() -> None:
    """AST findings index source without splitlines-only separators."""
    content = "import os\n\f# Collect settings for the report\nenv = dict(os.environ)\n"
    state = {"components": ["script.py"], "file_cache": {"script.py": content}}

    findings = static_runner.run_static_patterns(state, [data_exfiltration_module])
    e2 = next(finding for finding in findings if finding.rule_id == "E2")

    assert e2.start_line == 3
    assert e2.surface == CODE


def test_deduplicate_to_expand_preserves_each_occurrence_surface() -> None:
    """The report expansion path keeps occurrence labels after compaction."""
    findings = [
        analyzer_finding_to_finding(_analyzer_finding("SKILL.md"), line_text="# Role"),
        analyzer_finding_to_finding(_analyzer_finding("scripts/helper.py"), line_text="    # x"),
        analyzer_finding_to_finding(_analyzer_finding("docs/guide.md"), line_text="# Guide"),
    ]
    for finding in findings:
        finding.match_fingerprint = "shared-fingerprint"

    expanded = _expand_occurrences(deduplicate(findings))

    assert {finding.file: finding.surface for finding in expanded} == {
        "SKILL.md": INSTRUCTIONS,
        "scripts/helper.py": COMMENTS,
        "docs/guide.md": DOCS,
    }


def _analyzer_finding(file_path: str) -> AnalyzerFinding:
    return AnalyzerFinding(
        rule_id="RP1",
        message="hidden instruction",
        severity=Severity.MEDIUM,
        location=Location(file=file_path, start_line=1),
        matched_text="do this",
    )


@pytest.mark.parametrize(
    ("file_path", "line_text", "expected"),
    [
        ("SKILL.md", "# Role", INSTRUCTIONS),
        ("src/helper.py", "payload = 1", CODE),
        ("tests/test_helper.py", "payload = 1", TESTS),
        ("src/helper.py", "# payload = 1", COMMENTS),
        ("config/settings.json", '{"a": 1}', CONFIG),
    ],
)
def test_one_rule_reports_each_surface_of_its_hits(
    file_path: str, line_text: str, expected: str
) -> None:
    """The same rule is labelled by where it landed, not by which rule fired."""
    finding = analyzer_finding_to_finding(_analyzer_finding(file_path), line_text=line_text)

    assert finding.to_dict()["surface"] == expected


def test_serialization_derives_a_surface_and_keeps_an_explicit_one() -> None:
    """The label is additive: absent labels are derived, present ones are kept."""
    derived = Finding(rule_id="RP1", message="m", file="docs/guide.md")
    explicit = Finding(rule_id="RP1", message="m", file="docs/guide.md", surface=TESTS)

    assert derived.surface is None
    assert derived.to_dict()["surface"] == DOCS
    assert explicit.to_dict()["surface"] == TESTS


def test_json_and_sarif_reports_carry_the_surface() -> None:
    """Both report formats expose the label without touching severity."""
    high = Finding(rule_id="RP1", message="m", severity="HIGH", file="docs/guide.md")
    payload = json.loads(
        _format_json(
            [high, Finding(rule_id="RP1", message="m", file="tests/test_a.py")],
            component_metadata=[],
            manifest={"name": "demo"},
            skill_path="demo",
            risk_score=10,
            risk_severity="LOW",
            risk_recommendation="review",
            has_executable_scripts=False,
        )
    )

    assert [issue["surface"] for issue in payload["issues"]] == [DOCS, TESTS]
    assert payload["issues"][0]["severity"] == "HIGH"
    properties = _build_sarif_properties(high)
    assert properties is not None
    assert (properties["severity"], properties["surface"]) == ("HIGH", DOCS)


def test_dedup_keeps_a_surface_per_occurrence() -> None:
    """Exact-match compaction across files keeps each occurrence's own label."""
    findings = [
        analyzer_finding_to_finding(_analyzer_finding("SKILL.md"), line_text="# Role"),
        analyzer_finding_to_finding(_analyzer_finding("scripts/helper.py"), line_text="    # x"),
        analyzer_finding_to_finding(_analyzer_finding("docs/guide.md"), line_text="# Guide"),
    ]
    for finding in findings:
        finding.match_fingerprint = "shared-fingerprint"

    compacted = deduplicate(findings)

    assert len(compacted) == 1
    assert {
        str(occurrence["file"]): occurrence["surface"] for occurrence in compacted[0].occurrences
    } == {"SKILL.md": INSTRUCTIONS, "scripts/helper.py": COMMENTS, "docs/guide.md": DOCS}


def test_sarif_properties_prefer_the_occurrence_surface() -> None:
    """A SARIF result for an occurrence reports that occurrence's surface."""
    occurrence = {"file": "scripts/helper.py", "start_line": 5, "surface": COMMENTS}
    finding = Finding(
        rule_id="P5",
        message="m",
        file="SKILL.md",
        surface=INSTRUCTIONS,
        occurrences=[occurrence],
    )

    properties = _build_sarif_properties(finding, occurrence)

    assert properties is not None
    assert properties["surface"] == COMMENTS
