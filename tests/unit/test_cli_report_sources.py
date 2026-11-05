"""All known local sources and aliased destinations are rejected before analysis."""

import os
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

import skillspector.cli as cli


@pytest.fixture
def skill(tmp_path):
    root = tmp_path / "skill"
    root.mkdir()
    (root / "SKILL.md").write_text("---\nname: example\n---\nHello")
    (root / "script.py").write_text("print('source')")
    return root


@pytest.mark.parametrize("primary", [False, True])
@pytest.mark.parametrize("hardlink", [False, True])
def test_source_aliases_protected_for_primary_and_additional(
    tmp_path, skill, monkeypatch, primary, hardlink
):
    source = skill / "script.py"
    destination = tmp_path / "alias.py" if hardlink else source
    if hardlink:
        os.link(source, destination)
    graph = Mock(side_effect=AssertionError("analysis began"))
    monkeypatch.setattr(cli, "_scan_skill", graph)
    args = ["scan", str(skill), "--no-llm"]
    if primary:
        args += ["--output", str(destination), "--report", f"json={tmp_path / 'safe.json'}"]
    else:
        args += ["--report", f"json={destination}"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 2
    assert "protected local source" in result.output
    graph.assert_not_called()
    assert source.read_text() == "print('source')"
    assert not (tmp_path / "safe.json").exists()


@pytest.mark.parametrize("alias", ["same", "relative", "hardlink"])
def test_destination_aliases_rejected_before_analysis(tmp_path, skill, monkeypatch, alias):
    primary = tmp_path / "report.json"
    primary.write_text("unchanged")
    if alias == "hardlink":
        additional = tmp_path / "other.json"
        os.link(primary, additional)
    elif alias == "relative":
        monkeypatch.chdir(tmp_path)
        additional = "report.json"
    else:
        additional = primary
    analyzer = Mock(side_effect=AssertionError("analysis began"))
    monkeypatch.setattr(cli, "_scan_skill", analyzer)
    result = CliRunner().invoke(
        cli.app,
        [
            "scan",
            str(skill),
            "--no-llm",
            "--output",
            str(primary),
            "--report",
            f"json={additional}",
        ],
    )
    assert result.exit_code == 2
    assert "Duplicate or aliased" in result.output
    analyzer.assert_not_called()
    assert primary.read_text() == "unchanged"


def test_recursive_source_beyond_skill_limit_is_protected(tmp_path, monkeypatch):
    root = tmp_path / "collection"
    for name in ("a", "z"):
        child = root / name
        child.mkdir(parents=True)
        (child / "SKILL.md").write_text(f"---\nname: {name}\n---\nHello")
        (child / "source.py").write_text("protected")
    analyzer = Mock(side_effect=AssertionError("analysis began"))
    monkeypatch.setattr(cli, "_scan_multi_skill", analyzer)
    result = CliRunner().invoke(
        cli.app,
        [
            "scan",
            str(root),
            "--recursive",
            "--max-skills",
            "1",
            "--no-llm",
            "--report",
            f"json={root / 'z' / 'source.py'}",
        ],
    )
    assert result.exit_code == 2
    analyzer.assert_not_called()


def test_registry_comparison_alias_protected(tmp_path, monkeypatch):
    source = tmp_path / "registry.json"
    compare = tmp_path / "compare.json"
    alias = tmp_path / "alias.json"
    source.write_text("{}")
    compare.write_text("{}")
    os.link(compare, alias)
    scanner = Mock(side_effect=AssertionError("registry scanned"))
    monkeypatch.setattr(cli, "scan_registry", scanner)
    result = CliRunner().invoke(
        cli.app,
        [
            "scan",
            str(source),
            "--mcp-registry",
            "--mcp-registry-compare",
            str(compare),
            "--format",
            "json",
            "--report",
            f"json={alias}",
        ],
    )
    assert result.exit_code == 2
    scanner.assert_not_called()
