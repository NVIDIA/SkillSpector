"""Task 2.5: opt-in validation must not broaden the legacy output boundary."""

import json
import os
import re
import shlex
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.main import get_command
from typer.testing import CliRunner

import skillspector.cli as cli
import skillspector.file_output as file_output
from skillspector.report_outputs import FormatChoice, parse_report_declarations
from skillspector.report_sources import record_scan_sources


@pytest.fixture
def legacy_boundary(monkeypatch):
    """Fail loudly if a no--report scan enters either multi-report barrier."""
    forbidden = Mock(side_effect=AssertionError("multi-report protection enabled"))
    for name in ("preflight_output", "SourceProtectionInventory", "validate_report_outputs"):
        monkeypatch.setattr(cli, name, forbidden)
    yield forbidden
    forbidden.assert_not_called()


@pytest.fixture
def skill(tmp_path):
    root = tmp_path / "skill"
    root.mkdir()
    (root / "SKILL.md").write_text("---\nname: example\n---\nHello", encoding="utf-8")
    (root / "script.py").write_text("print('source')", encoding="utf-8")
    return root


def invoke_legacy(source, output, *options):
    return CliRunner().invoke(
        cli.app,
        ["scan", str(source), "--no-llm", "--format", "json", "--output", str(output), *options],
    )


@pytest.mark.parametrize("mode", ["ordinary", "recursive", "registry"])
@pytest.mark.parametrize("invalid_path", ["missing_parent", "symlink_parent", "symlink", "fifo"])
def test_no_report_full_path_validation_happens_after_analysis(
    tmp_path, skill, monkeypatch, legacy_boundary, mode, invalid_path
):
    target = tmp_path / "existing.json"
    target.write_text("unchanged", encoding="utf-8")
    if invalid_path == "missing_parent":
        output = tmp_path / "missing" / "report.json"
    elif invalid_path == "symlink_parent":
        parent = tmp_path / "linked"
        parent.symlink_to(skill, target_is_directory=True)
        output = parent / "report.json"
    elif invalid_path == "symlink":
        output = tmp_path / "linked.json"
        output.symlink_to(target)
    else:
        output = tmp_path / "fifo"
        os.mkfifo(output)

    events = []
    real_write = cli.write_text_no_follow

    def write(path, body):
        events.append("write")
        return real_write(path, body)

    def analyze(**kwargs):
        events.append("analysis")
        assert not (tmp_path / "missing").exists()
        assert target.read_text(encoding="utf-8") == "unchanged"
        return {"report_body": '{"legacy": true}', "risk_score": 0, "findings": []}

    analyzer = Mock(side_effect=analyze)
    monkeypatch.setattr(cli, "_scan_skill", analyzer)
    monkeypatch.setattr(cli, "write_text_no_follow", write)
    source = skill
    options = []
    if mode == "recursive":
        source = tmp_path / "collection"
        child = source / "child"
        child.mkdir(parents=True)
        (child / "SKILL.md").write_text("Hello", encoding="utf-8")
        options = ["--recursive"]
    elif mode == "registry":
        source = tmp_path / "registry.json"
        source.write_text("{}", encoding="utf-8")
        scanner = Mock(side_effect=lambda *args, **kwargs: analyze())
        monkeypatch.setattr(cli, "scan_registry", scanner)
        options = ["--mcp-registry"]
    result = invoke_legacy(source, output, *options)

    assert result.exit_code == 2, result.output
    assert events == ["analysis", "write"]
    if mode == "registry":
        scanner.assert_called_once()
    else:
        analyzer.assert_called_once()
    assert target.read_text(encoding="utf-8") == "unchanged"
    assert not (tmp_path / "missing").exists()
    assert not (skill / "report.json").exists()
    if invalid_path in {"symlink", "symlink_parent"}:
        assert (output if invalid_path == "symlink" else output.parent).is_symlink()


@pytest.mark.parametrize("destination", ["directory", "manifest", "manifest_hardlink"])
def test_no_report_retains_early_directory_and_manifest_protection(
    tmp_path, skill, monkeypatch, legacy_boundary, destination
):
    manifest = skill / "SKILL.md"
    original = manifest.read_bytes()
    if destination == "directory":
        output = tmp_path / "reports"
        output.mkdir()
    elif destination == "manifest":
        output = manifest
    else:
        output = tmp_path / "manifest-alias.md"
        os.link(manifest, output)
    analyzer = Mock(side_effect=AssertionError("analysis began"))
    monkeypatch.setattr(cli, "_scan_skill", analyzer)

    result = invoke_legacy(skill, output)

    assert result.exit_code == 2, result.output
    analyzer.assert_not_called()
    assert manifest.read_bytes() == original
    expected_error = "non-regular path" if destination == "directory" else "input file"
    assert expected_error in result.output


def test_no_report_write_permission_failure_remains_after_analysis(
    tmp_path, skill, monkeypatch, legacy_boundary
):
    """Inject EACCES rather than relying on process credentials or chmod."""
    output = tmp_path / "report.json"
    output.write_text("unchanged", encoding="utf-8")
    events = []
    real_open = file_output.os.open

    def denied_open(path, flags, *args, **kwargs):
        if path == output.name and flags & os.O_WRONLY:
            events.append("permission_failure")
            raise PermissionError(13, "Permission denied", str(output))
        return real_open(path, flags, *args, **kwargs)

    def analyze(**kwargs):
        events.append("analysis")
        return {"report_body": "{}", "risk_score": 0}

    analyzer = Mock(side_effect=analyze)
    monkeypatch.setattr(cli, "_scan_skill", analyzer)
    monkeypatch.setattr(file_output.os, "open", denied_open)

    result = invoke_legacy(skill, output)

    assert result.exit_code == 2, result.output
    analyzer.assert_called_once()
    assert events == ["analysis", "permission_failure"]
    assert output.read_text(encoding="utf-8") == "unchanged"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["report.json", "skill"]


def test_no_report_secure_platform_check_remains_early(
    tmp_path, skill, monkeypatch, legacy_boundary
):
    monkeypatch.setattr(file_output, "_SECURE_OUTPUT_SUPPORTED", False)
    analyzer = Mock(side_effect=AssertionError("analysis began"))
    monkeypatch.setattr(cli, "_scan_skill", analyzer)
    output = tmp_path / "report.json"

    result = invoke_legacy(skill, output)

    assert result.exit_code == 2
    assert "unsupported" in result.output
    analyzer.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("hardlink", [False, True])
def test_no_report_does_not_extend_protection_to_bundled_scripts(
    tmp_path, skill, monkeypatch, legacy_boundary, hardlink
):
    """Compatibility assertion, not a recommendation to overwrite sources."""
    source = skill / "script.py"
    output = tmp_path / "script-alias.py" if hardlink else source
    if hardlink:
        os.link(source, output)
    manifest_bytes = (skill / "SKILL.md").read_bytes()
    analyzer = Mock(return_value={"report_body": '{"legacy": true}', "risk_score": 0})
    monkeypatch.setattr(cli, "_scan_skill", analyzer)

    result = invoke_legacy(skill, output)

    assert result.exit_code == 0, result.output
    analyzer.assert_called_once()
    assert output.read_text(encoding="utf-8") == '{"legacy": true}'
    assert "Report saved to" in result.stdout
    assert result.stderr == ""
    assert (skill / "SKILL.md").read_bytes() == manifest_bytes
    if hardlink:
        assert source.read_text(encoding="utf-8") == "print('source')"


@pytest.mark.parametrize("manifest_destination", [False, True])
def test_no_report_recursive_scope_protects_omitted_manifests_not_bundled_sources(
    tmp_path, monkeypatch, legacy_boundary, manifest_destination
):
    root = tmp_path / "collection"
    for name in ("a", "z"):
        child = root / name
        child.mkdir(parents=True)
        (child / "SKILL.md").write_text(f"---\nname: {name}\n---\nHello", encoding="utf-8")
        (child / "source.py").write_text("protected only with --report", encoding="utf-8")
    output = root / "z" / ("SKILL.md" if manifest_destination else "source.py")
    original = output.read_bytes()
    analyzer = Mock(return_value={"risk_score": 0, "findings": []})
    monkeypatch.setattr(cli, "_scan_skill", analyzer)

    result = invoke_legacy(root, output, "--recursive", "--max-skills", "1")

    assert result.exit_code == (2 if manifest_destination else 0), result.output
    if manifest_destination:
        analyzer.assert_not_called()
        assert output.read_bytes() == original
    else:
        analyzer.assert_called_once()
        assert Path(analyzer.call_args.kwargs["input_path"]) == root / "a"
        assert isinstance(json.loads(output.read_text(encoding="utf-8")), dict)
        assert (root / "z" / "SKILL.md").read_text(encoding="utf-8").endswith("Hello")


@pytest.mark.parametrize("comparison", [False, True])
def test_no_report_registry_input_and_comparison_aliases_remain_protected(
    tmp_path, monkeypatch, legacy_boundary, comparison
):
    source = tmp_path / "registry.json"
    compare = tmp_path / "compare.json"
    source.write_text("{}", encoding="utf-8")
    compare.write_text('{"comparison": true}', encoding="utf-8")
    output = tmp_path / "alias.json"
    protected = compare if comparison else source
    original = protected.read_bytes()
    os.link(protected, output)
    scanner = Mock(side_effect=AssertionError("registry scanned"))
    monkeypatch.setattr(cli, "scan_registry", scanner)

    result = invoke_legacy(source, output, "--mcp-registry", "--mcp-registry-compare", str(compare))

    assert result.exit_code == 2, result.output
    scanner.assert_not_called()
    assert protected.read_bytes() == output.read_bytes() == original


def test_no_report_does_not_apply_late_discovered_source_barrier(
    tmp_path, skill, monkeypatch, legacy_boundary
):
    output = tmp_path / "discovered.md"
    output.write_text("source first learned during analysis", encoding="utf-8")
    original = output.read_bytes()

    def analyze(**kwargs):
        result = {
            "report_body": '{"legacy": true}',
            "risk_score": 0,
            "components": [str(output)],
            "transitive_sources": [str(output)],
        }
        record_scan_sources(result)
        assert output.read_bytes() == original
        return result

    analyzer = Mock(side_effect=analyze)
    monkeypatch.setattr(cli, "_scan_skill", analyzer)

    result = invoke_legacy(skill, output, "--transitive")

    assert result.exit_code == 0, result.output
    analyzer.assert_called_once()
    assert output.read_text(encoding="utf-8") == '{"legacy": true}'


def test_readme_output_format_examples_match_report_parser():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    section = readme.read_text(encoding="utf-8").split("### Output Formats\n", 1)[1]
    section = section.split("\n### ", 1)[0]
    examples = [
        ("json=report.json", FormatChoice.json, "report.json"),
        ("sarif=report.sarif", FormatChoice.sarif, "report.sarif"),
        ("JsOn=result=v1.json", FormatChoice.json, "result=v1.json"),
        ("json=second.json", FormatChoice.json, "second.json"),
    ]
    for declaration, expected_format, expected_path in examples:
        assert f"--report {declaration}" in section
        entry = parse_report_declarations([declaration])[0]
        assert (entry.format, entry.path) == (expected_format, expected_path)
    repeated = parse_report_declarations([examples[2][0], examples[3][0]])
    assert [entry.format for entry in repeated] == [FormatChoice.json, FormatChoice.json]
    assert repeated[0].path != repeated[1].path


@pytest.mark.parametrize("multi_report", [False, True], ids=["sole", "additional"])
def test_readme_recursive_examples_parse_and_scan_once(tmp_path, monkeypatch, multi_report):
    """Exercise README commands through the parser and recursive orchestration."""
    from skillspector.nodes.report import prepare_report, report_rendering_is_deferred
    from skillspector.sarif_models import validate_sarif_report

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    section = readme.split("### Recursive Collection Scanning\n", 1)[1].split("\n### ", 1)[0]
    examples = re.findall(
        r"^skillspector scan .*?(?<!\\)\n",
        section,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert len(examples) == 2
    arguments = shlex.split(examples[int(multi_report)].replace("\\\n", ""))[2:]
    group = get_command(cli.app)
    context = group.make_context("skillspector", [], resilient_parsing=True)
    command = group.get_command(context, "scan")
    assert command is not None
    with command.make_context("scan", list(arguments)) as scan_context:
        params = scan_context.params
        assert params["recursive"] is True
        assert params["max_skills"] == 100
        assert params["no_llm"] is True
        assert params["format"] == "json"
        assert params["output"] is None
        entries = parse_report_declarations(params["report"])
        assert [(entry.format, entry.path) for entry in entries] == (
            [
                (FormatChoice.sarif, "recursive.sarif"),
                (FormatChoice.markdown, "recursive.md"),
            ]
            if multi_report
            else []
        )

    monkeypatch.chdir(tmp_path)
    for name in ("alpha", "beta"):
        child = tmp_path / "my-skills" / name
        child.mkdir(parents=True)
        (child / "SKILL.md").write_text(f"---\nname: {name}\n---\nHello", encoding="utf-8")
    analyzed = []

    def analyze(**kwargs):
        assert report_rendering_is_deferred()
        name = Path(kwargs["input_path"]).name
        analyzed.append(name)
        prepared = prepare_report(
            {
                "manifest": {"name": name},
                "findings": [],
                "use_llm": False,
                "analysis_completeness": {
                    "is_complete": True,
                    "execution_successful": True,
                    "coverage_percent": 100,
                },
            }
        )
        return {**prepared.public_fields, "prepared_report": prepared}

    monkeypatch.setattr(cli, "_scan_skill", analyze)
    result = CliRunner().invoke(cli.app, ["scan", *arguments])
    assert result.exit_code == 0, result.output
    assert analyzed == ["alpha", "beta"]
    payload = json.loads(result.stdout)
    assert payload["skills_scanned"] == 2
    assert payload["analysis_completeness"]["is_complete"]
    assert payload["report_completeness"]["is_complete"]
    if multi_report:
        validate_sarif_report(json.loads((tmp_path / "recursive.sarif").read_text()))
        assert "Report completeness" in (tmp_path / "recursive.md").read_text()
        assert "recursive.sarif" in result.stderr
        assert "recursive.md" in result.stderr
    else:
        assert not (tmp_path / "recursive.sarif").exists()
        assert not (tmp_path / "recursive.md").exists()
