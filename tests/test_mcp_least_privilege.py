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

"""Tests for mcp_least_privilege analyzer (B.3.1 LP1-LP4)."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from skillspector.nodes.analyzers import mcp_least_privilege

# ---------------------------------------------------------------------------
# Fixture directory path
# ---------------------------------------------------------------------------
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Executable extensions (must match build_context._EXECUTABLE_EXTENSIONS)
_EXECUTABLE_EXTENSIONS = frozenset(
    {".py", ".sh", ".bash", ".zsh", ".js", ".ts", ".rb", ".go", ".rs", ".pl"}
)

_FILE_TYPES: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".py": "python",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".txt": "text",
    ".js": "javascript",
    ".ts": "typescript",
    ".rb": "ruby",
    ".go": "go",
    ".rs": "rust",
}

_SKIP_DIRS = frozenset(
    {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".pytest_cache"}
)


def _infer_file_type(path: str) -> str:
    idx = path.rfind(".")
    suffix = path[idx:].lower() if idx >= 0 else ""
    return _FILE_TYPES.get(suffix, "other")


def _parse_yaml_frontmatter(skill_md_content: str) -> dict:
    """Parse YAML frontmatter from SKILL.md content. Returns raw dict (no normalization)."""
    if not skill_md_content.startswith("---"):
        return {}
    end_match = re.search(r"\n---\s*\n", skill_md_content[3:])
    if not end_match:
        return {}
    frontmatter = skill_md_content[3 : end_match.start() + 3]
    try:
        data = yaml.safe_load(frontmatter)
    except yaml.YAMLError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _make_state(fixture_name: str) -> dict:
    """Build a minimal SkillspectorState from a fixture directory.

    Parses SKILL.md YAML frontmatter, builds file_cache and component_metadata.
    Mirrors what build_context.build_context() produces.
    """
    fixture_dir = FIXTURES_DIR / fixture_name

    # Collect files (skip hidden and skip dirs)
    components: list[str] = []
    for item in fixture_dir.rglob("*"):
        if not item.is_file():
            continue
        if any(skip in item.parts for skip in _SKIP_DIRS):
            continue
        if item.name.startswith(".") and not item.name.startswith(".claude"):
            continue
        rel = item.relative_to(fixture_dir).as_posix()  # forward slashes on every OS
        components.append(rel)
    components.sort()

    # Build file_cache
    file_cache: dict[str, str] = {}
    for path in components:
        full = fixture_dir / path
        try:
            file_cache[path] = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            file_cache[path] = ""

    # Build component_metadata
    component_metadata: list[dict] = []
    has_executable = False
    for path in components:
        full = fixture_dir / path
        suffix = full.suffix.lower()
        executable = suffix in _EXECUTABLE_EXTENSIONS
        if executable:
            has_executable = True
        file_type = _infer_file_type(path)
        try:
            content = file_cache.get(path, "")
            lines = len(content.splitlines())
            size_bytes = full.stat().st_size
        except OSError:
            lines = 0
            size_bytes = 0
        component_metadata.append(
            {
                "path": path,
                "type": file_type,
                "lines": lines,
                "executable": executable,
                "size_bytes": size_bytes,
            }
        )

    # Parse manifest from SKILL.md
    manifest: dict = {}
    skill_md_content = file_cache.get("SKILL.md", "")
    if skill_md_content:
        raw = _parse_yaml_frontmatter(skill_md_content)
        if raw:
            manifest = raw
            # Normalize permissions: if not present, leave as None (absent)
            # build_context normalizes to [], but we preserve raw for fidelity
            # The implementation must handle both None and [].
            if "permissions" not in manifest:
                manifest["permissions"] = None
            elif isinstance(manifest["permissions"], list):
                manifest["permissions"] = [str(p) for p in manifest["permissions"]]
            if "triggers" not in manifest:
                manifest["triggers"] = []
            elif isinstance(manifest["triggers"], list):
                manifest["triggers"] = [str(t) for t in manifest["triggers"]]

    return {
        "manifest": manifest,
        "file_cache": file_cache,
        "component_metadata": component_metadata,
        "has_executable_scripts": has_executable,
        "components": components,
    }


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestLP2WildcardPermission:
    def test_wildcard_detected(self):
        """mcp_overprivileged_skill has '*' in permissions → LP2 finding (MEDIUM, >=0.90)."""
        state = _make_state("mcp_overprivileged_skill")
        result = mcp_least_privilege.node(state)
        findings = result["findings"]
        lp2_findings = [f for f in findings if f.rule_id == "LP2"]
        assert len(lp2_findings) >= 1, f"Expected LP2 finding, got: {[f.rule_id for f in findings]}"
        lp2 = lp2_findings[0]
        assert lp2.severity == "MEDIUM", f"Expected MEDIUM severity, got {lp2.severity}"
        assert lp2.confidence >= 0.90, f"Expected confidence >= 0.90, got {lp2.confidence}"
        assert lp2.category == "MCP Least Privilege"
        assert "ASI02" in lp2.tags
        assert lp2.file == "SKILL.md"


class TestLP1UnderdeclaredCapability:
    def test_underdeclared_detected(self):
        """mcp_underdeclared_skill uses network, shell, env but declares no permissions → LP3 (no LP1 since permissions is empty)."""
        state = _make_state("mcp_underdeclared_skill")
        result = mcp_least_privilege.node(state)
        findings = result["findings"]
        # Underdeclared skill has permissions=None/[], so LP3 fires (not LP1)
        lp3_findings = [f for f in findings if f.rule_id == "LP3"]
        assert len(lp3_findings) >= 1, (
            f"Expected LP3 finding for underdeclared skill, got: {[f.rule_id for f in findings]}"
        )
        lp3 = lp3_findings[0]
        assert lp3.severity == "MEDIUM"
        assert lp3.confidence >= 0.70
        assert lp3.category == "MCP Least Privilege"
        assert "ASI02" in lp3.tags

    def test_underdeclared_has_high_severity_for_lp1(self):
        """When a skill declares some permissions but misses capabilities, LP1 findings are HIGH."""
        # Use a custom state where some permissions are declared but capabilities underdeclared
        state: dict = {
            "manifest": {
                "name": "partial-skill",
                "permissions": ["read"],
                "triggers": [],
            },
            "file_cache": {
                "scripts/agent.py": (
                    "import httpx\n"
                    "import subprocess\n"
                    "response = httpx.get('http://example.com')\n"
                    "subprocess.run(['ls'])\n"
                ),
            },
            "component_metadata": [
                {
                    "path": "scripts/agent.py",
                    "type": "python",
                    "executable": True,
                    "lines": 4,
                    "size_bytes": 100,
                },
            ],
            "has_executable_scripts": True,
            "components": ["scripts/agent.py"],
        }
        result = mcp_least_privilege.node(state)
        findings = result["findings"]
        lp1_findings = [f for f in findings if f.rule_id == "LP1"]
        assert len(lp1_findings) >= 1, (
            f"Expected LP1 findings, got: {[f.rule_id for f in findings]}"
        )
        for lp1 in lp1_findings:
            assert lp1.severity == "HIGH", f"Expected HIGH severity for LP1, got {lp1.severity}"
            assert lp1.confidence >= 0.70


class TestLP3NoPermissions:
    def test_no_permissions_field(self):
        """mcp_underdeclared_skill has no permissions field → LP3 (MEDIUM, >=0.70)."""
        state = _make_state("mcp_underdeclared_skill")
        result = mcp_least_privilege.node(state)
        findings = result["findings"]
        lp3_findings = [f for f in findings if f.rule_id == "LP3"]
        assert len(lp3_findings) >= 1, f"Expected LP3 finding, got: {[f.rule_id for f in findings]}"
        lp3 = lp3_findings[0]
        assert lp3.severity == "MEDIUM"
        assert lp3.confidence >= 0.70
        assert lp3.file == "SKILL.md"


class TestLP4OverDeclared:
    def test_over_declared_detected(self):
        """mcp_overprivileged_skill declares bash/network/write but code only reads → LP4 for network and write."""
        state = _make_state("mcp_overprivileged_skill")
        result = mcp_least_privilege.node(state)
        findings = result["findings"]
        lp4_findings = [f for f in findings if f.rule_id == "LP4"]
        assert len(lp4_findings) >= 1, (
            f"Expected LP4 findings for over-declared permissions, got: {[f.rule_id for f in findings]}"
        )
        for lp4 in lp4_findings:
            assert lp4.severity == "LOW", f"Expected LOW severity for LP4, got {lp4.severity}"
            assert lp4.confidence >= 0.60
            assert lp4.category == "MCP Least Privilege"
            assert "ASI02" in lp4.tags
            assert lp4.file == "SKILL.md"


class TestCleanSkill:
    def test_no_findings(self):
        """mcp_clean_skill: permissions match capabilities → zero LP findings."""
        state = _make_state("mcp_clean_skill")
        result = mcp_least_privilege.node(state)
        findings = result["findings"]
        lp_findings = [f for f in findings if f.rule_id.startswith("LP")]
        assert len(lp_findings) == 0, (
            f"Expected zero LP findings for clean skill, got: {[(f.rule_id, f.message) for f in lp_findings]}"
        )


class TestEdgeCases:
    def test_no_manifest_skips(self):
        """Empty state (no manifest) → empty findings."""
        state: dict = {
            "manifest": {},
            "file_cache": {},
            "component_metadata": [],
            "has_executable_scripts": False,
            "components": [],
        }
        result = mcp_least_privilege.node(state)
        assert result["findings"] == []

    def test_docs_only_skill_skips(self):
        """Skill with only .md files → no executable files → empty findings."""
        state: dict = {
            "manifest": {
                "name": "docs-skill",
                "permissions": None,
                "triggers": [],
            },
            "file_cache": {
                "SKILL.md": "---\nname: docs-skill\n---\n# Docs only\n",
                "README.md": "# Read me\n",
            },
            "component_metadata": [
                {
                    "path": "SKILL.md",
                    "type": "markdown",
                    "executable": False,
                    "lines": 4,
                    "size_bytes": 40,
                },
                {
                    "path": "README.md",
                    "type": "markdown",
                    "executable": False,
                    "lines": 2,
                    "size_bytes": 15,
                },
            ],
            "has_executable_scripts": False,
            "components": ["README.md", "SKILL.md"],
        }
        result = mcp_least_privilege.node(state)
        assert result["findings"] == []

    def test_permission_matching_case_insensitive(self):
        """Permissions 'BASH' and 'Network' should match shell/network capabilities."""
        state: dict = {
            "manifest": {
                "name": "case-test",
                "permissions": ["BASH", "Network"],
                "triggers": [],
            },
            "file_cache": {
                "scripts/run.py": (
                    "import subprocess\n"
                    "import httpx\n"
                    "subprocess.run(['ls'])\n"
                    "httpx.get('http://example.com')\n"
                ),
            },
            "component_metadata": [
                {
                    "path": "scripts/run.py",
                    "type": "python",
                    "executable": True,
                    "lines": 4,
                    "size_bytes": 90,
                },
            ],
            "has_executable_scripts": True,
            "components": ["scripts/run.py"],
        }
        result = mcp_least_privilege.node(state)
        findings = result["findings"]
        # shell and network are covered → no LP1 for those capabilities
        lp1_for_shell = [
            f for f in findings if f.rule_id == "LP1" and "shell" in (f.message or "").lower()
        ]
        lp1_for_network = [
            f for f in findings if f.rule_id == "LP1" and "network" in (f.message or "").lower()
        ]
        assert len(lp1_for_shell) == 0, "BASH permission should cover shell capability"
        assert len(lp1_for_network) == 0, "Network permission should cover network capability"

    def test_lp1_test_files_reduced_confidence(self):
        """Capability found only in test_helper.py → LP1 confidence < 0.75."""
        state: dict = {
            "manifest": {
                "name": "test-confidence-skill",
                "permissions": ["read"],
                "triggers": [],
            },
            "file_cache": {
                "test_helper.py": (
                    "import httpx\ndef test_fetch():\n    return httpx.get('http://example.com')\n"
                ),
            },
            "component_metadata": [
                {
                    "path": "test_helper.py",
                    "type": "python",
                    "executable": True,
                    "lines": 3,
                    "size_bytes": 80,
                },
            ],
            "has_executable_scripts": True,
            "components": ["test_helper.py"],
        }
        result = mcp_least_privilege.node(state)
        findings = result["findings"]
        # Network detected only in test file; LP1 should have reduced confidence
        lp1_findings = [f for f in findings if f.rule_id == "LP1"]
        assert len(lp1_findings) >= 1, (
            f"Expected LP1 finding for undeclared network in test file, got: {[f.rule_id for f in findings]}"
        )
        for lp1 in lp1_findings:
            assert lp1.confidence < 0.75, (
                f"Expected reduced confidence for test-file-only capability, got {lp1.confidence}"
            )
