# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Additional outputs reuse existing single and recursive scan results."""

import json
import os
import sys
from copy import deepcopy
from datetime import UTC, datetime
from importlib import import_module
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from skillspector import cli
from skillspector.file_output import write_text_no_follow
from skillspector.input_handler import InputHandler
from skillspector.models import Finding
from skillspector.nodes.report import report
from skillspector.report_outputs import parse_reports
from skillspector.sarif_models import validate_sarif_report
from skillspector.suppression import baseline_from_dict, build_baseline_dict, serialize_baseline

runner = CliRunner()


@pytest.fixture
def transitive_states(tmp_path, monkeypatch):
    """Fresh source-aware findings, exact suppression, and successful runtime evidence."""
    report_module = import_module("skillspector.nodes.report")

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 12, tzinfo=UTC)

    monkeypatch.setattr(report_module, "datetime", FixedDatetime)
    monkeypatch.setattr(report_module, "is_llm_available", lambda **kw: (True, None))
    semantic_nodes = [
        name
        for name, module in cli.ANALYZER_MODULES.items()
        if getattr(module, "requires_api_key", False)
    ]
    target = "https://github.com/example/dependency"

    def factory(primary, depth, degraded=False, incomplete=False, show_suppressed=True):
        def state(child=False):
            cache = {"SKILL.md": "# Child" if child else target, "helper.py": "print('safe')"}
            findings = [
                Finding(
                    rule_id="CHILD" if child else "EXEC",
                    finding_id="child-finding" if child else "root-finding",
                    severity="LOW" if child else "MEDIUM",
                    confidence=1.0,
                    file="helper.py",
                    message="Child finding" if child else "Executable finding",
                )
            ]
            if not child:
                findings.append(
                    Finding(
                        rule_id="ACCEPTED",
                        finding_id="accepted-finding",
                        severity="HIGH",
                        file="SKILL.md",
                        message="Accepted",
                    )
                )
            return {
                "skill_path": str(tmp_path),
                "manifest": {"name": "state-fixture"},
                "output_format": primary.value,
                "findings": findings,
                "filtered_findings": findings[:1],
                "components": list(cache),
                "component_metadata": [
                    {
                        "path": path,
                        "type": "python" if path.endswith(".py") else "markdown",
                        "executable": path.endswith(".py"),
                        "lines": 1,
                        "size_bytes": len(content),
                    }
                    for path, content in cache.items()
                ],
                "file_cache": cache,
                "local_file_cache": dict(cache),
                "has_executable_scripts": True,
                "use_llm": True,
                "llm_requested": True,
                "llm_call_log": [
                    {"node": node, "ok": not degraded, "error": "failed" if degraded else None}
                    for node in [*semantic_nodes, "meta_analyzer"]
                ],
                "analyzer_status_events": [
                    {"analyzer_id": node, "status": "failed" if degraded else "completed"}
                    for node in semantic_nodes
                ],
                "analysis_completeness": {
                    "is_complete": not incomplete,
                    "execution_successful": True,
                    "coverage_percent": 50.0 if incomplete else 100.0,
                },
            }

        root = state()
        root["baseline"] = baseline_from_dict(
            build_baseline_dict(
                root["findings"][1:],
                file_cache=root["file_cache"],
                scanner_version=cli.__version__,
            )
        )
        root["show_suppressed"] = show_suppressed
        child_scan = Mock(side_effect=lambda **kw: state(child=True))
        monkeypatch.setattr(cli, "_run_graph_scan_for_source", child_scan)
        result = cli._scan_transitive(
            initial_result=root,
            format=primary,
            no_llm=False,
            max_depth=depth,
            transitive_allow_prefix=[],
            transitive_deny_prefix=[],
            baseline=None,
            show_suppressed=show_suppressed,
            visited=set(),
        )
        assert child_scan.call_count == (1 if depth else 0)
        return root, result, child_scan

    return factory


@pytest.mark.parametrize("depth", [0, 1])
@pytest.mark.parametrize("primary", list(cli.FormatChoice))
def test_transitive_reports_preserve_full_state(tmp_path, transitive_states, depth, primary):
    root, result, child_scan = transitive_states(primary, depth)
    snapshot = deepcopy(result)
    # Independent expectation: MEDIUM executable = 10 * 1.3, child LOW = int(5 * 1.3).
    assert result["risk_score"] == (19 if depth else 13)
    assert [f.rule_id for f in result["active_findings"]] == (
        ["EXEC", "CHILD"] if depth else ["EXEC"]
    )
    assert len(result["suppressed_findings"]) == 1
    primary_path = tmp_path / "primary"
    formats = list(cli.FormatChoice)
    outputs = [(choice, tmp_path / f"extra-{choice.value}") for choice in formats]
    outputs += [(formats[0], tmp_path / "repeat")]
    cli._write_reports(result, primary, primary_path, outputs)
    assert child_scan.call_count == depth
    primary_body = primary_path.read_text()
    original_bodies = {path: path.read_text() for _, path in outputs}
    fresh_scans = []
    for choice, path in outputs:
        _, fresh, fresh_scan = transitive_states(choice, depth)
        fresh_scans.append(fresh_scan)
        assert path.read_text() == cli._result_body(fresh)
    cli._write_reports(result, primary, primary_path, list(reversed(outputs)))
    assert {path: path.read_text() for _, path in outputs} == original_bodies
    assert all(scan.call_count == depth for scan in fresh_scans)
    assert primary_path.read_text() == primary_body == cli._result_body(result)
    assert result == snapshot  # Deep comparison includes findings, metadata, caches, and telemetry.
    child_scan.assert_called_once() if depth else child_scan.assert_not_called()
    assert result["findings"][:2] == root["findings"]  # canonical, not suppressed projection
    payload = json.loads((tmp_path / "extra-json").read_text())
    assert payload["risk_assessment"]["score"] == (19 if depth else 13)
    assert payload["suppressed"][0]["id"] == "ACCEPTED"
    assert len(payload["components"]) == (4 if depth else 2)
    assert payload["metadata"]["has_executable_scripts"] is True
    assert payload["metadata"]["llm_requested"] is True
    assert payload["metadata"].get("llm_degraded", False) is False
    assert payload["metadata"]["llm_calls_succeeded"] == len(root["llm_call_log"]) * (depth + 1)
    assert any(c.get("source_identity") for c in payload["components"]) is bool(depth)
    validate_sarif_report(json.loads((tmp_path / "extra-sarif").read_text()))
    assert "Accepted" in (tmp_path / "extra-markdown").read_text()
    assert "Executable scripts: Yes" in (tmp_path / "extra-terminal").read_text()


@pytest.mark.parametrize(
    "depth,degraded,incomplete,show",
    [
        (0, True, False, False),
        (1, True, False, True),
        (0, False, True, True),
    ],
)
def test_transitive_extra_reports_retain_runtime_edges(
    tmp_path,
    transitive_states,
    depth,
    degraded,
    incomplete,
    show,
):
    _, result, scan = transitive_states(
        cli.FormatChoice.markdown, depth, degraded, incomplete, show
    )
    snapshot = deepcopy(result)
    rendered = cli._render_report(result, cli.FormatChoice.json)
    payload = json.loads(rendered["report_body"])
    assert payload["metadata"]["llm_requested"] is True
    assert payload["metadata"].get("llm_degraded", False) is degraded
    assert payload["metadata"]["llm_calls_succeeded"] == (
        0 if degraded else len(result["llm_call_log"])
    )
    assert payload["suppressed_count"] == 1
    assert {issue["id"] for issue in payload["issues"]} == (
        {"EXEC", "CHILD"} if depth else {"EXEC"}
    )
    if incomplete or (degraded and depth):
        assert payload["analysis_completeness"]["is_complete"] is False
    markdown = cli._result_body(cli._render_report(result, cli.FormatChoice.markdown))
    assert ("Accepted" in markdown) is show
    assert result == snapshot
    assert scan.call_count == depth


@pytest.fixture
def baseline_files(tmp_path):
    def factory(shipped=False):
        skill = tmp_path / "skill"
        skill.mkdir()
        manifest = skill / "SKILL.md"
        manifest.write_bytes(b"# Preserve input\r\n")
        baseline = skill / ".skillspector-baseline.yaml" if shipped else tmp_path / "baseline.yaml"
        data = build_baseline_dict([], file_cache={}, scanner_version=cli.__version__)
        original = ("# Preserve ä\r\n" + serialize_baseline(data)).encode()
        baseline.write_bytes(original)
        return skill, baseline, original

    return factory


@pytest.mark.parametrize(
    "source",
    [
        "local",
        "http://github.com/example/skill",
        "https://github.com/example/skill",
        "git@github.com:example/skill.git",
    ],
)
@pytest.mark.parametrize("option", ["--output", "--report"])
@pytest.mark.parametrize("alias", ["exact", "missing", "relative", "hardlink"])
def test_explicit_baseline_collisions_preflight(
    tmp_path,
    monkeypatch,
    baseline_files,
    source,
    option,
    alias,
):
    skill, baseline, original = baseline_files()
    monkeypatch.chdir(tmp_path)
    destination = baseline
    if alias == "missing":
        destination = tmp_path / "missing" / ".." / baseline.name
    elif alias == "relative":
        destination = baseline.relative_to(tmp_path)
    elif alias == "hardlink":
        destination = tmp_path / "hardlink.yaml"
        try:
            os.link(baseline, destination)
        except OSError as exc:
            pytest.skip(f"Hardlinks unsupported: {exc}")
    guards = [
        Mock(side_effect=AssertionError("preflight must precede scan/fetch/write"))
        for _ in range(3)
    ]
    monkeypatch.setattr(cli, "_scan_skill", guards[0])
    monkeypatch.setattr(InputHandler, "resolve", guards[1])
    monkeypatch.setattr(cli, "write_text_no_follow", guards[2])
    result = runner.invoke(
        cli.app,
        [
            "scan",
            str(skill) if source == "local" else source,
            "--no-llm",
            "--baseline",
            str(baseline),
            option,
            f"json={destination}" if option == "--report" else str(destination),
        ],
    )
    assert baseline.read_bytes() == original
    assert result.exit_code == 2, result.output
    assert "--output points to the input file" in result.output
    for guard in guards:
        guard.assert_not_called()


@pytest.mark.parametrize("option", ["--output", "--report"])
@pytest.mark.parametrize("alias", [False, True])
def test_shipped_baseline_collisions_preflight(
    monkeypatch,
    baseline_files,
    option,
    alias,
):
    skill, baseline, original = baseline_files(shipped=True)
    destination = skill / "missing" / ".." / baseline.name if alias else baseline
    scan = Mock(side_effect=AssertionError("scan must not run"))
    writer = Mock(side_effect=AssertionError("write must not run"))
    monkeypatch.setattr(cli, "_scan_skill", scan)
    monkeypatch.setattr(cli, "write_text_no_follow", writer)
    result = runner.invoke(
        cli.app,
        [
            "scan",
            str(skill),
            "--no-llm",
            "--use-shipped-baseline",
            option,
            f"json={destination}" if option == "--report" else str(destination),
        ],
    )
    assert baseline.read_bytes() == original
    assert result.exit_code == 2, result.output
    assert (
        "--output points to the input file" in result.output
        or "--report must not overwrite" in result.output
    )
    scan.assert_not_called()
    writer.assert_not_called()


@pytest.mark.parametrize("destination_kind", ["new", "existing", "symlink", "parent_symlink"])
@pytest.mark.parametrize("option", ["--output", "--report"])
def test_baseline_outputs_allow_regular_files_but_refuse_symlinks(
    tmp_path,
    monkeypatch,
    baseline_files,
    destination_kind,
    option,
):
    skill, baseline, original = baseline_files()
    manifest_bytes = (skill / "SKILL.md").read_bytes()
    destination = tmp_path / "report.json"
    protected = tmp_path / "protected.json"
    protected.write_bytes(b"Preserve symlink target\r\n")
    if destination_kind == "existing":
        destination.write_text("Old regular report")
    elif destination_kind in ("symlink", "parent_symlink"):
        try:
            if destination_kind == "symlink":
                destination.symlink_to(protected)
            else:
                (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
                destination = tmp_path / "alias" / protected.name
        except OSError as exc:
            pytest.skip(f"Symlinks unsupported: {exc}")
    scan = Mock(side_effect=lambda input_path, format, **kw: scan_result(input_path, format))
    monkeypatch.setattr(cli, "_scan_skill", scan)
    result = runner.invoke(
        cli.app,
        [
            "scan",
            str(skill),
            "--no-llm",
            "--baseline",
            str(baseline),
            "-f",
            "json",
            option,
            f"json={destination}" if option == "--report" else str(destination),
        ],
    )
    assert baseline.read_bytes() == original
    assert (skill / "SKILL.md").read_bytes() == manifest_bytes
    assert protected.read_bytes() == b"Preserve symlink target\r\n"
    scan.assert_called_once()
    if "symlink" in destination_kind:
        assert result.exit_code == 2
        assert "symlink" in result.output or "non-regular output" in result.output
    else:
        assert result.exit_code == 0, result.output
        assert json.loads(destination.read_text())["issues"]


@pytest.mark.integration
def test_offline_transitive_multi_report_cli(tmp_path, monkeypatch):
    """Run actual ingest, static analyzers, transitive merge, rendering, and secure writing."""
    skill = tmp_path / "skill"
    skill.mkdir()
    inputs = {
        "SKILL.md": b"---\nname: offline-report\n---\n# Safe skill\n",
        "helper.py": b"print('safe')\n",
    }
    for name, content in inputs.items():
        (skill / name).write_bytes(content)
    # Any accidental external fetch is a failure, not an authenticated/network scan.
    monkeypatch.setattr(InputHandler, "_clone_git", Mock(side_effect=AssertionError("no fetch")))
    monkeypatch.setattr(
        InputHandler, "_download_file", Mock(side_effect=AssertionError("no fetch"))
    )
    primary = tmp_path / "primary.md"
    additional = tmp_path / "additional.json"
    sarif = tmp_path / "additional.sarif"
    result = runner.invoke(
        cli.app,
        [
            "scan",
            str(skill),
            "--no-llm",
            "--transitive",
            "-f",
            "markdown",
            "-o",
            str(primary),
            "--report",
            f"json={additional}",
            "--report",
            f"sarif={sarif}",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(additional.read_text())
    assert {c["path"] for c in payload["components"]} == set(inputs)
    assert payload["metadata"]["has_executable_scripts"] is True
    assert payload["metadata"]["llm_requested"] is False
    assert payload["metadata"]["transitive_targets_scanned"] == 0
    assert f"| Score | {payload['risk_assessment']['score']}/100 |" in primary.read_text()
    validate_sarif_report(json.loads(sarif.read_text()))
    for name, original in inputs.items():
        assert (skill / name).read_bytes() == original


def scan_result(path, format):
    state = {
        "skill_path": str(path),
        "manifest": {"name": "example", "description": "Example skill"},
        "findings": [
            Finding(
                rule_id="TEST",
                severity="LOW",
                message="Example finding",
                file="SKILL.md",
            )
        ],
        "use_llm": False,
        "llm_requested": False,
        "output_format": format.value,
    }
    return {**state, **report(state)}


@pytest.mark.parametrize("recursive", [False, True])
@pytest.mark.parametrize("primary", list(cli.FormatChoice))
def test_reports_scan_each_skill_once(tmp_path, monkeypatch, recursive, primary):
    root = tmp_path / "skills"
    root.mkdir()
    skills = [root / name for name in ("one", "two")] if recursive else [root]
    for skill in skills:
        skill.mkdir(exist_ok=True)
        (skill / "SKILL.md").write_text("# Example\n")
    scan = Mock(side_effect=lambda input_path, format, **kw: scan_result(input_path, format))
    monkeypatch.setattr(cli, "_scan_skill", scan)
    original_states = []
    render = cli._render_report

    def capture_render(state, format):
        original_states.append(state.copy())
        rendered = render(state, format)
        assert state == original_states[-1]
        return rendered

    monkeypatch.setattr(cli, "_render_report", capture_render)
    args = ["scan", str(root), "--no-llm", "-f", primary.value]
    if recursive:
        args.append("--recursive")
    destinations = {}
    for format in cli.FormatChoice:
        path = tmp_path / f"report.{format.value}"
        destinations[format] = path
        args.extend(["--report", f"{format.value}={path}"])
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert scan.call_count == len(skills)
    for path in destinations.values():
        assert path.read_text()
    payload = json.loads(destinations[cli.FormatChoice.json].read_text())
    if recursive:
        assert payload["multi_skill"] is True
        assert len(payload["skills"]) == 2
        assert all(child["issues"] for child in payload["skills"])
    else:
        assert payload["issues"][0]["explanation"] == "Example finding"
    validate_sarif_report(json.loads(destinations[cli.FormatChoice.sarif].read_text()))
    if primary in (cli.FormatChoice.json, cli.FormatChoice.sarif):
        json.loads(result.stdout)  # File notifications must not contaminate stdout.


@pytest.mark.parametrize("declaration", ["json", "html=out", "json=", "json=-"])
def test_invalid_report_fails_before_scan(tmp_path, monkeypatch, declaration):
    scan = Mock()
    monkeypatch.setattr(cli, "_scan_skill", scan)
    result = runner.invoke(cli.app, ["scan", str(tmp_path), "--report", declaration])
    assert result.exit_code == 2
    scan.assert_not_called()


def test_report_paths_are_distinct_and_preserve_equals(tmp_path):
    path = tmp_path / "report=one.json"
    assert parse_reports([f"JSON={path}"], ("json",)) == [("json", path)]
    with pytest.raises(ValueError, match="distinct"):
        parse_reports([f"json={path}"], ("json",), path)
    path.touch()
    alias = tmp_path / "alias"
    os.link(path, alias)
    with pytest.raises(ValueError, match="distinct"):
        parse_reports([f"json={path}", f"json={alias}"], ("json",))


def test_report_paths_use_writer_normalization(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "report.json"
    reports = parse_reports(["json=missing/../report.json"], ("json",))
    assert reports == [("json", path)]
    with pytest.raises(ValueError, match="distinct"):
        parse_reports(["json=missing/../report.json", "json=report.json"], ("json",))
    with pytest.raises(ValueError, match="distinct"):
        parse_reports(["json=report.json"], ("json",), tmp_path / "missing" / ".." / path.name)
    write_text_no_follow(reports[0][1], '{"ok": true}')
    assert json.loads(path.read_text()) == {"ok": True}
    assert not (tmp_path / "missing").exists()
    os.link(path, tmp_path / "alias.json")
    with pytest.raises(ValueError, match="distinct"):
        parse_reports(["json=missing/../alias.json"], ("json",), path)


def test_report_paths_retain_untrusted_symlinks(tmp_path):
    directory = tmp_path / "reports"
    directory.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    # Do not resolve child symlinks into paths that the no-follow writer permits.
    destination = alias / "report.json"
    reports = parse_reports([f"json={destination}"], ("json",))
    assert reports == [("json", destination)]
    with pytest.raises(ValueError, match="symlink or is not a directory"):
        write_text_no_follow(reports[0][1], '{"ok": true}')
    assert not list(directory.iterdir())


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize(
    "input_kind,filename",
    [
        ("directory", "SKILL.md"),
        ("directory", "helper.py"),
        ("file", "SKILL.md"),
        ("recursive", "SKILL.md"),
        ("recursive", "helper.py"),
    ],
)
def test_report_does_not_overwrite_input(tmp_path, monkeypatch, alias, input_kind, filename):
    skill = tmp_path / "one" if input_kind == "recursive" else tmp_path
    skill.mkdir(exist_ok=True)
    (skill / "SKILL.md").write_text("# Example\n")
    source = skill / filename
    original = b"# Keep me\r\n# preserved bytes: \xc3\xa4\n"
    source.write_bytes(original)
    destination = skill / "missing" / ".." / filename if alias else source
    assert not (skill / "missing").exists()
    scan = Mock(side_effect=lambda input_path, format, **kw: scan_result(input_path, format))
    monkeypatch.setattr(cli, "_scan_skill", scan)
    args = [
        "scan",
        str(source if input_kind == "file" else tmp_path),
        "--no-llm",
        "--report",
        f"json={destination}",
    ]
    if input_kind == "recursive":
        args.append("--recursive")
    result = runner.invoke(cli.app, args)
    assert source.read_bytes() == original
    assert result.exit_code == 2
    assert "--output points to the input file" in result.output or (
        "--report must not overwrite" in result.output
    )
    scan.assert_not_called()


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("filename", ["SKILL.md", "helper.py"])
def test_batch_report_does_not_overwrite_input(tmp_path, monkeypatch, capsys, alias, filename):
    from contrib.batch_scan import batch_scan

    skill = tmp_path / "one"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# Example\n")
    source = skill / filename
    original = b"# Keep me\r\n# preserved bytes: \xc3\xa4\n"
    source.write_bytes(original)
    destination = skill / "missing" / ".." / filename if alias else source
    assert not (skill / "missing").exists()
    scan = Mock(
        return_value=(
            {
                "skill": {"name": "example", "language": "en"},
                "risk_assessment": {"score": 0, "severity": "LOW"},
                "issues": [],
            },
            None,
            "one",
        )
    )
    pool = Mock(return_value=None)
    monkeypatch.setattr(batch_scan, "_scan_skill_bounded", scan)
    monkeypatch.setattr(batch_scan, "create_api_key_pool_from_env", pool)
    monkeypatch.setattr(
        sys,
        "argv",
        ["batch_scan", str(tmp_path), "--no-llm", "--report", f"json={destination}"],
    )
    with pytest.raises(SystemExit) as exc:
        try:
            batch_scan.main()
        finally:
            assert source.read_bytes() == original
    assert exc.value.code == 2
    assert "--report must not overwrite a batch input" in capsys.readouterr().err
    scan.assert_not_called()
    pool.assert_not_called()


@pytest.mark.parametrize("recursive", [False, True])
def test_cleanup_on_additional_write_failure(tmp_path, monkeypatch, recursive):
    skill = tmp_path / "one" if recursive else tmp_path
    skill.mkdir(exist_ok=True)
    (skill / "SKILL.md").write_text("# Example\n")
    state = scan_result(skill, cli.FormatChoice.json)
    original = (skill / "SKILL.md").read_bytes()
    baseline = tmp_path / "baseline.yaml"
    baseline.write_bytes(b"# Preserve baseline\r\n")
    temp_dir = tmp_path / "skillspector_scan"
    temp_dir.mkdir()
    (temp_dir / "temporary.md").write_text("# temporary")
    state["temp_dir_for_cleanup"] = str(temp_dir)
    monkeypatch.setattr(cli, "_scan_skill", Mock(return_value=state))
    cleanup = Mock(wraps=cli.cleanup_result)
    monkeypatch.setattr(cli, "cleanup_result", cleanup)
    args = [
        "scan",
        str(tmp_path),
        "--no-llm",
        "-f",
        "json",
        "--report",
        f"markdown={tmp_path / 'missing' / 'report.md'}",
    ]
    if recursive:
        args.append("--recursive")
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 2
    json.loads(result.stdout)
    cleanup.assert_called_once_with(state)
    assert not temp_dir.exists()
    assert (skill / "SKILL.md").read_bytes() == original
    assert baseline.read_bytes() == b"# Preserve baseline\r\n"


def test_findings_exit_after_all_reports(tmp_path, monkeypatch):
    (tmp_path / "SKILL.md").write_text("# Example\n")
    monkeypatch.setattr(
        cli,
        "_scan_skill",
        Mock(side_effect=lambda input_path, format, **kw: scan_result(input_path, format)),
    )
    output = tmp_path / "primary.json"
    markdown = tmp_path / "additional.md"
    result = runner.invoke(
        cli.app,
        [
            "scan",
            str(tmp_path),
            "--no-llm",
            "--fail-on-findings",
            "-f",
            "json",
            "-o",
            str(output),
            "--report",
            f"markdown={markdown}",
        ],
    )
    assert result.exit_code == 1
    assert json.loads(output.read_text())["issues"]
    assert markdown.read_text()


def test_additional_recursive_report_does_not_change_other_outputs(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    skill = root / "one"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Example\n")
    monkeypatch.setattr(
        cli,
        "_scan_skill",
        Mock(side_effect=lambda input_path, format, **kw: scan_result(input_path, format)),
    )
    render = cli._render_report
    monkeypatch.setattr(cli, "_MULTI_SKILL_MAX_REPORT_CHARACTERS", 20_000)
    monkeypatch.setattr(
        cli,
        "_render_report",
        lambda state, format: (
            {"report_body": "x" * 20_001}
            if format == cli.FormatChoice.markdown
            else render(state, format)
        ),
    )
    output = tmp_path / "primary.json"
    copy = tmp_path / "copy.json"
    result = runner.invoke(
        cli.app,
        [
            "scan",
            str(root),
            "--recursive",
            "--no-llm",
            "--fail-on-incomplete",
            "-f",
            "json",
            "-o",
            str(output),
            "--report",
            f"markdown={tmp_path / 'large.md'}",
            "--report",
            f"json={copy}",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text()) == json.loads(copy.read_text())


def test_registry_additional_json_reuses_scan(tmp_path, monkeypatch):
    source = tmp_path / "registry.json"
    source.write_text("{}")
    output = tmp_path / "report.json"
    scan = Mock(return_value={"findings": [], "risk_score": 0})
    monkeypatch.setattr(cli, "scan_registry", scan)
    args = ["scan", str(source), "--mcp-registry", "-f", "json", "--report"]
    result = runner.invoke(cli.app, [*args, f"json={output}"])
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text()) == json.loads(result.stdout)
    scan.assert_called_once()
    scan.reset_mock()
    result = runner.invoke(cli.app, [*args, f"markdown={tmp_path / 'report.md'}"])
    assert result.exit_code == 2
    scan.assert_not_called()


def test_batch_reports_reuse_results(tmp_path, monkeypatch, capsys):
    from contrib.batch_scan import batch_scan

    root = tmp_path / "skills"
    skill = root / "one"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Example\n")
    scan = Mock(
        return_value=(
            {
                "skill": {"name": "example", "language": "en"},
                "risk_assessment": {"score": 0, "severity": "LOW"},
                "issues": [],
            },
            None,
            "one",
        )
    )
    monkeypatch.setattr(batch_scan, "_scan_skill_bounded", scan)
    monkeypatch.setattr(batch_scan, "create_api_key_pool_from_env", lambda: None)
    markdown = tmp_path / "report.md"
    duplicate = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_scan",
            str(root),
            "--no-llm",
            "-f",
            "json",
            "--report",
            f"markdown={markdown}",
            "--report",
            f"json={duplicate}",
        ],
    )
    batch_scan.main()
    assert scan.call_count == 1
    assert markdown.read_text()
    assert json.loads(duplicate.read_text()) == json.loads(capsys.readouterr().out)
