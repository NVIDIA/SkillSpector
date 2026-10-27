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

"""Regression tests for PR #498 security inspection fixes:
- Finding A: Runtime monitoring error handling, availability, and CLI behavior.
- Finding B: Dependency graph skill discovery, nested directory handling, and root manifest.
- Section 8: Input-discovery verification against project contract.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.input_handler import InputHandler
from skillspector.security_inspection.dependency_graph import (
    build_dependency_graph,
    discover_skills,
)
from skillspector.security_inspection.event_model import EventGraph, SecurityEvent
from skillspector.security_inspection.permission_manifest import load_manifest
from skillspector.security_inspection.runtime_monitor import (
    RuntimeMonitoringUnavailableError,
    is_runtime_available,
    run_isolated,
)
from skillspector.security_inspection.scanner import SecurityScanner


# ==============================================================================
# Finding A: Runtime Monitoring Tests
# ==============================================================================
class TestFindingARuntimeMonitoring:
    """Regression tests verifying runtime monitoring does not silently do nothing."""

    def test_runtime_execution_and_event_collection_succeeding(self, tmp_path: Path):
        """When isolated runtime sandbox is available and returns events, scanner collects them."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test-skill\n---\n")

        mock_graph = EventGraph()
        mock_graph.add(
            SecurityEvent(
                source="runtime",
                category="network",
                action="connect",
                subject="test-skill",
                target="evil.example.com:443",
                capability="network.outbound",
                evidence="Outbound socket connection detected",
                severity="critical",
                rule_id="RT-NET-001",
            )
        )

        with (
            patch(
                "skillspector.security_inspection.runtime_monitor.is_runtime_available",
                return_value=True,
            ),
            patch(
                "skillspector.security_inspection.runtime_monitor.run_isolated",
                return_value=mock_graph,
            ),
        ):
            scanner = SecurityScanner(skills_root=tmp_path, db_path=tmp_path / "test.db")
            result = scanner.scan(enable_runtime=True)

            assert "runtime_error" not in result
            assert result["summary"]["total_findings"] >= 1
            # Verify event was captured in the global graph
            events = result["event_graph"]["events"]
            assert any(e["rule_id"] == "RT-NET-001" for e in events)

    def test_runtime_valid_execution_producing_zero_events(self, tmp_path: Path):
        """When execution succeeds but produces zero events, zero count is legitimately preserved."""
        skill_dir = tmp_path / "benign-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: benign-skill\n---\n")

        runner = CliRunner()
        mock_empty_graph = EventGraph()

        with (
            patch(
                "skillspector.security_inspection.runtime_monitor.is_runtime_available",
                return_value=True,
            ),
            patch(
                "skillspector.security_inspection.runtime_monitor.run_isolated",
                return_value=mock_empty_graph,
            ),
        ):
            res = runner.invoke(app, ["runtime-scan", str(skill_dir)])
            assert res.exit_code == 0
            assert "Runtime events: 0" in res.output

    def test_isolated_execution_unavailable_or_stubbed(self, tmp_path: Path):
        """The real/current isolated execution is unavailable and raises RuntimeMonitoringUnavailableError."""
        assert is_runtime_available() is False
        with pytest.raises(RuntimeMonitoringUnavailableError) as exc_info:
            run_isolated(tmp_path)
        assert "Runtime monitoring is unavailable" in str(exc_info.value)
        assert "disabled pending a real OS-level sandbox" in str(exc_info.value)

    def test_runtime_collection_failing(self, tmp_path: Path):
        """When runtime execution fails unexpectedly, runtime-scan exits 1 with actionable message."""
        runner = CliRunner()
        with (
            patch(
                "skillspector.security_inspection.runtime_monitor.is_runtime_available",
                return_value=True,
            ),
            patch(
                "skillspector.security_inspection.runtime_monitor.run_isolated",
                side_effect=TimeoutError("Container execution timed out after 10s"),
            ),
        ):
            res = runner.invoke(app, ["runtime-scan", str(tmp_path)])
            assert res.exit_code == 1
            assert "Runtime scan failed" in res.output or "timed out" in res.output

    def test_runtime_scan_exit_code_and_actionable_error_message(self, tmp_path: Path):
        """runtime-scan command with stubbed/unavailable sandbox exits 1 with actionable error."""
        runner = CliRunner()
        res = runner.invoke(app, ["runtime-scan", str(tmp_path)])
        assert res.exit_code == 1
        assert "Error: Runtime monitoring is unavailable" in res.output
        assert "disabled pending a real OS-level" in res.output

    def test_offline_scan_runtime_flag_surfaces_warning(self, tmp_path: Path):
        """offline-scan --runtime surfaces an explicit warning when runtime monitoring is unavailable."""
        skill_dir = tmp_path / "skill-test"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: skill-test\n---\n")

        runner = CliRunner()
        res = runner.invoke(app, ["offline-scan", str(tmp_path), "--runtime"])
        assert res.exit_code == 0
        assert "Warning: Runtime monitoring is unavailable" in res.output
        assert "Offline scan complete" in res.output

    def test_ordinary_offline_scan_functional_without_runtime(self, tmp_path: Path):
        """offline-scan without --runtime runs purely static scan without runtime warning."""
        skill_dir = tmp_path / "skill-test"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: skill-test\n---\n")

        runner = CliRunner()
        res = runner.invoke(app, ["offline-scan", str(tmp_path)])
        assert res.exit_code == 0
        assert "Warning: Runtime monitoring" not in res.output
        assert "Offline scan complete" in res.output

    def test_cli_output_never_reports_success_when_unmonitored(self, tmp_path: Path):
        """CLI never prints 'Runtime events: 0' when runtime execution never occurred."""
        runner = CliRunner()
        res = runner.invoke(app, ["runtime-scan", str(tmp_path)])
        assert res.exit_code == 1
        assert "Runtime events: 0" not in res.output


# ==============================================================================
# Finding B: Dependency Graph and Skill Discovery Tests
# ==============================================================================
class TestFindingBDependencyGraph:
    """Regression tests verifying skill discovery correctly classifies skill directories."""

    def test_root_skill_manifest_discovered_and_analyzed(self, tmp_path: Path):
        """A root-level SKILL.md is discovered as a single skill and analyzed."""
        (tmp_path / "SKILL.md").write_text("---\nname: root-skill\n---\n# Root Skill\n")
        (tmp_path / "main.py").write_text("print('hello')\n")

        discovered = discover_skills(tmp_path)
        assert len(discovered) == 1
        assert tmp_path.name in discovered
        assert discovered[tmp_path.name] == tmp_path

        graph, skills = build_dependency_graph(tmp_path)
        assert tmp_path.name in graph.nodes
        assert tmp_path.name in skills

    def test_conventional_skill_with_nested_scripts(self, tmp_path: Path):
        """A conventional skill directory (my-skill/SKILL.md + my-skill/scripts/run.py) produces the expected skill."""
        repo_dir = tmp_path / "my-repo"
        repo_dir.mkdir()
        skill_dir = repo_dir / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\n# My Skill\n")

        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "run.py").write_text("import os\nprint('run')\n")

        discovered = discover_skills(repo_dir)
        assert "my-skill" in discovered
        assert discovered["my-skill"] == skill_dir

    def test_scripts_directory_not_misclassified_as_separate_skill(self, tmp_path: Path):
        """A scripts/ directory containing Python files is not misclassified as a skill."""
        repo_dir = tmp_path / "my-repo"
        repo_dir.mkdir()
        skill_dir = repo_dir / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\n# My Skill\n")

        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "run.py").write_text("print('run')\n")

        discovered = discover_skills(repo_dir)
        assert "scripts" not in discovered

        graph, _ = build_dependency_graph(repo_dir)
        assert "scripts" not in graph.nodes

    def test_readme_only_directory_not_misclassified(self, tmp_path: Path):
        """A directory containing only README.md is not misclassified as a skill."""
        repo_dir = tmp_path / "my-repo"
        repo_dir.mkdir()
        docs_dir = repo_dir / "docs"
        docs_dir.mkdir()
        (docs_dir / "README.md").write_text("# Documentation\nNot a skill.\n")

        skill_dir = repo_dir / "valid-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: valid-skill\n---\n")

        discovered = discover_skills(repo_dir)
        assert "docs" not in discovered
        assert "valid-skill" in discovered

    def test_repository_containing_multiple_skills_discovers_all(self, tmp_path: Path):
        """A repository containing multiple skills discovers all of them."""
        skill_a = tmp_path / "skill-alpha"
        skill_a.mkdir()
        (skill_a / "SKILL.md").write_text("---\nname: skill-alpha\n---\n")

        skill_b = tmp_path / "skill-beta"
        skill_b.mkdir()
        (skill_b / "SKILL.md").write_text("---\nname: skill-beta\n---\n")

        discovered = discover_skills(tmp_path)
        assert "skill-alpha" in discovered
        assert "skill-beta" in discovered
        assert len(discovered) == 2

    def test_nested_directories_not_skills_do_not_create_nodes(self, tmp_path: Path):
        """Nested directories like utils/, tests/, data/ inside a skill do not create skill nodes."""
        skill = tmp_path / "complex-skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text("---\nname: complex-skill\n---\n")

        for d in ["utils", "tests", "data", "lib"]:
            sub = skill / d
            sub.mkdir()
            (sub / f"{d}.py").write_text(f"# {d} module\n")

        discovered = discover_skills(tmp_path)
        assert list(discovered.keys()) == ["complex-skill"]

        graph, _ = build_dependency_graph(tmp_path)
        assert "complex-skill" in graph.nodes
        for d in ["utils", "tests", "data", "lib"]:
            assert d not in graph.nodes

    def test_nested_skills_in_subdirectories_handled(self, tmp_path: Path):
        """Nested skills inside intermediate multi-skill categories are discovered."""
        category = tmp_path / "skills-category"
        category.mkdir()

        nested_a = category / "nested-skill-a"
        nested_a.mkdir()
        (nested_a / "SKILL.md").write_text("---\nname: nested-skill-a\n---\n")

        nested_b = category / "nested-skill-b"
        nested_b.mkdir()
        (nested_b / "SKILL.md").write_text("---\nname: nested-skill-b\n---\n")

        discovered = discover_skills(tmp_path)
        assert "nested-skill-a" in discovered
        assert "nested-skill-b" in discovered
        assert "skills-category" not in discovered

    def test_dependency_graph_nodes_and_edges_reflect_actual_relationships(self, tmp_path: Path):
        """Dependency graph edges reflect actual skill relationships, not directory containment."""
        skill_caller = tmp_path / "caller-skill"
        skill_caller.mkdir()
        (skill_caller / "SKILL.md").write_text(
            "---\nname: caller-skill\ndependencies:\n  - callee-skill\n---\n"
        )
        scripts = skill_caller / "scripts"
        scripts.mkdir()
        (scripts / "run.py").write_text("import callee_skill\n")

        skill_callee = tmp_path / "callee-skill"
        skill_callee.mkdir()
        (skill_callee / "SKILL.md").write_text("---\nname: callee-skill\n---\n")

        graph, skills = build_dependency_graph(tmp_path)
        assert "caller-skill" in graph.nodes
        assert "callee-skill" in graph.nodes
        assert "scripts" not in graph.nodes
        assert graph.has_edge("caller-skill", "callee-skill")

    def test_traversal_is_deterministic_and_no_duplicate_nodes(self, tmp_path: Path):
        """Discovery and graph building produce identical, sorted, deduplicated results."""
        for name in ["zebra-skill", "alpha-skill", "middle-skill"]:
            s = tmp_path / name
            s.mkdir()
            (s / "SKILL.md").write_text(f"---\nname: {name}\n---\n")

        res1 = list(discover_skills(tmp_path).keys())
        res2 = list(discover_skills(tmp_path).keys())
        assert res1 == ["alpha-skill", "middle-skill", "zebra-skill"]
        assert res1 == res2
        assert len(res1) == len(set(res1))

    def test_load_manifest_parses_skill_md_frontmatter(self, tmp_path: Path):
        """load_manifest successfully parses YAML frontmatter in SKILL.md."""
        skill_dir = tmp_path / "frontmatter-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: frontmatter-skill\npermissions:\n  network: full\n  filesystem: read\n---\n"
        )

        manifest, src = load_manifest(skill_dir)
        assert src.endswith("SKILL.md")
        assert manifest.get("name") == "frontmatter-skill"
        assert manifest.get("permissions", {}).get("network") == "full"


# ==============================================================================
# Section 8: Input-Discovery Contract Tests
# ==============================================================================
class TestInputDiscoveryContract:
    """Verification of SkillSpector input-discovery and classification contract."""

    def test_existing_valid_skill_directory_recognized(self, tmp_path: Path):
        """An existing valid skill directory is recognized as 'directory'."""
        skill_dir = tmp_path / "valid-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: valid-skill\n---\n")

        handler = InputHandler()
        try:
            target, kind = handler.resolve(str(skill_dir))
            assert kind == "directory"
            assert target == skill_dir.resolve()
        finally:
            handler.cleanup()

    def test_root_level_skill_manifest_recognized(self, tmp_path: Path):
        """A root-level skill manifest file (.md) is recognized as 'file'."""
        manifest_file = tmp_path / "SKILL.md"
        manifest_file.write_text("---\nname: root-skill\n---\n")

        handler = InputHandler()
        try:
            target, kind = handler.resolve(str(manifest_file))
            assert kind == "file"
            assert target.is_dir()  # wrap_single_file creates temp directory containing file
        finally:
            handler.cleanup()

    def test_conventional_skill_directory_with_nested_scripts_recognized(self, tmp_path: Path):
        """A conventional skill directory containing nested scripts is recognized as 'directory'."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\n")
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "run.py").write_text("print('running')\n")

        handler = InputHandler()
        try:
            target, kind = handler.resolve(str(skill_dir))
            assert kind == "directory"
            assert target == skill_dir.resolve()
        finally:
            handler.cleanup()

    def test_missing_directory_fails_with_clear_accurate_message(self, tmp_path: Path):
        """A missing directory path fails with clear ValueError explaining supported formats."""
        nonexistent = tmp_path / "nonexistent-dir"
        handler = InputHandler()
        try:
            with pytest.raises(ValueError) as exc_info:
                handler.resolve(str(nonexistent))
            err_msg = str(exc_info.value)
            assert f"Cannot determine input type for: {nonexistent}" in err_msg
            assert (
                "Supported formats: Git URL, file URL, .zip file, .md file, or directory" in err_msg
            )
        finally:
            handler.cleanup()

    def test_unsupported_format_rejected(self, tmp_path: Path):
        """A file with an unsupported format or extension that does not exist is rejected."""
        unsupported = tmp_path / "missing_file.unknown_extension"
        handler = InputHandler()
        try:
            with pytest.raises(ValueError) as exc_info:
                handler.resolve(str(unsupported))
            assert "Cannot determine input type" in str(exc_info.value)
        finally:
            handler.cleanup()

    def test_git_url_file_url_zip_markdown_behavior_preserved(self, tmp_path: Path):
        """Git URLs, file URLs, and Markdown paths retain their recognized input types."""
        handler = InputHandler()
        try:
            assert handler._is_git_url("https://github.com/owner/repo.git") is True
            assert handler._is_file_url("https://example.com/archive.zip") is True

            # Existing markdown file
            md_file = tmp_path / "sample.md"
            md_file.write_text("# Markdown\n")
            target, kind = handler.resolve(str(md_file))
            assert kind == "file"
        finally:
            handler.cleanup()

    def test_container_path_handling_when_mounted_correctly(self, tmp_path: Path):
        """Simulated container mount path (absolute path that exists) resolves correctly."""
        container_mount = tmp_path / "scan" / "my-skill"
        container_mount.mkdir(parents=True)
        (container_mount / "SKILL.md").write_text("---\nname: container-skill\n---\n")

        handler = InputHandler()
        try:
            target, kind = handler.resolve(str(container_mount))
            assert kind == "directory"
            assert target == container_mount.resolve()
        finally:
            handler.cleanup()
