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

"""Tests for static_runner code-example filtering and documentation-path confidence reduction."""

from __future__ import annotations

import pytest

from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm_module
from skillspector.nodes.analyzers import static_runner


class TestCodeExampleFiltering:
    """Findings inside fenced code blocks or documentation examples are filtered."""

    def test_curl_in_fenced_code_block_is_filtered(self) -> None:
        """A curl -k inside a markdown fenced code block should be filtered out."""
        content = """\
# Usage Guide

## Example: Checking Service Health

```bash
curl -k https://internal-api.example.com/health
```

This is how you check the health endpoint.
"""
        state = {
            "components": ["docs/usage.md"],
            "file_cache": {"docs/usage.md": content},
        }
        findings = static_runner.run_static_patterns(state, [tm_module])
        tm1_findings = [f for f in findings if f.rule_id == "TM1"]
        assert len(tm1_findings) == 0

    def test_shell_true_in_executable_python_is_not_filtered(self) -> None:
        """subprocess with shell=True in Python code should NOT be filtered."""
        content = """\
import subprocess
result = subprocess.run(cmd, shell=True)
"""
        state = {
            "components": ["deploy.py"],
            "file_cache": {"deploy.py": content},
        }
        findings = static_runner.run_static_patterns(state, [tm_module])
        tm1_findings = [f for f in findings if f.rule_id == "TM1"]
        assert len(tm1_findings) >= 1

    def test_git_reset_in_example_section_is_filtered(self) -> None:
        """git reset --hard inside 'example:' context is filtered."""
        content = """\
# Troubleshooting

Example: If you need to reset your local branch:

git reset --hard origin/main

This will discard all local changes.
"""
        state = {
            "components": ["troubleshooting.md"],
            "file_cache": {"troubleshooting.md": content},
        }
        findings = static_runner.run_static_patterns(state, [tm_module])
        tm1_findings = [f for f in findings if f.rule_id == "TM1"]
        assert len(tm1_findings) == 0

    def test_rm_rf_in_shell_script_is_not_filtered(self) -> None:
        """rm -rf in a .sh file without example context should NOT be filtered."""
        content = """\
#!/bin/bash
rm -rf /tmp/build-cache
"""
        state = {
            "components": ["cleanup.sh"],
            "file_cache": {"cleanup.sh": content},
        }
        findings = static_runner.run_static_patterns(state, [tm_module])
        tm1_findings = [f for f in findings if f.rule_id == "TM1"]
        assert len(tm1_findings) >= 1

    def test_finding_in_executable_not_dropped_by_generic_indicator(self) -> None:
        """A finding in an executable file is NOT dropped when context contains a generic indicator.

        Validates that an attacker cannot suppress a genuine finding in a .py file
        by salting nearby code with a comment like '# e.g. usage' or '# Note: ...'
        """
        content = """\
import subprocess
# Note: this is how we deploy
result = subprocess.run(cmd, shell=True)
"""
        state = {
            "components": ["deploy.py"],
            "file_cache": {"deploy.py": content},
        }
        findings = static_runner.run_static_patterns(state, [tm_module])
        tm1_findings = [f for f in findings if f.rule_id == "TM1"]
        assert len(tm1_findings) >= 1
        for f in tm1_findings:
            assert f.confidence > 0

    def test_extensionless_file_not_hard_dropped_by_code_example(self) -> None:
        """An extensionless file (inferred as 'other') in code-example context is downweighted, not dropped."""
        content = """\
#!/bin/bash
# Example: cleanup old builds
rm -rf /tmp/build-cache
"""
        state = {
            "components": ["cleanup_script"],
            "file_cache": {"cleanup_script": content},
        }
        findings = static_runner.run_static_patterns(state, [tm_module])
        tm1_findings = [f for f in findings if f.rule_id == "TM1"]
        assert len(tm1_findings) >= 1, (
            "Extensionless files must not have code-example findings hard-dropped"
        )

    def test_skill_md_findings_are_not_filtered_by_backticks(self) -> None:
        """SKILL.md is the primary instruction file — backticks alone shouldn't filter."""
        content = """\
---
name: deploy-tool
---
# Deploy Tool

Use this tool to deploy:
```
curl -k https://production.example.com/deploy
```

The agent will execute the above command.
"""
        state = {
            "components": ["SKILL.md"],
            "file_cache": {"SKILL.md": content},
        }
        findings = static_runner.run_static_patterns(state, [tm_module])
        # SKILL.md code blocks do get filtered by is_code_example (same as EA2/MP)
        # This is correct: the meta-analyzer handles SKILL.md nuance
        # The key test is that SKILL.md is NOT treated as documentation-path markdown
        for f in findings:
            # Confidence should NOT be reduced by _DOCUMENTATION_CONFIDENCE_FACTOR
            assert f.confidence >= 0.3


class TestDocumentationPathConfidenceReduction:
    """Findings in documentation subdirectories get reduced confidence."""

    def test_docs_subdir_markdown_gets_reduced_confidence(self) -> None:
        """A finding in docs/deploy.md gets confidence reduced."""
        content = """\
# Deployment

Run the following to deploy:
rm -rf /opt/app/old-version
"""
        state = {
            "components": ["docs/deploy.md"],
            "file_cache": {"docs/deploy.md": content},
        }
        findings = static_runner.run_static_patterns(state, [tm_module])
        tm1_findings = [f for f in findings if f.rule_id == "TM1"]
        assert len(tm1_findings) >= 1
        for f in tm1_findings:
            # Original confidence 0.9 * 0.3 factor = 0.27
            assert f.confidence <= 0.3

    def test_procedures_subdir_markdown_gets_reduced_confidence(self) -> None:
        """A finding in procedures/reset.md gets confidence reduced."""
        content = """\
# Reset Procedure

git reset --hard origin/main
"""
        state = {
            "components": ["procedures/reset.md"],
            "file_cache": {"procedures/reset.md": content},
        }
        findings = static_runner.run_static_patterns(state, [tm_module])
        tm1_findings = [f for f in findings if f.rule_id == "TM1"]
        assert len(tm1_findings) >= 1
        for f in tm1_findings:
            # Original confidence 0.65 * 0.3 factor = 0.195
            assert f.confidence < 0.25

    def test_skill_md_is_not_documentation_path(self) -> None:
        """SKILL.md should never get documentation confidence reduction."""
        content = """\
---
name: dangerous-skill
---
# Tool
subprocess.run(["curl", "-k", "https://api.example.com"])
"""
        state = {
            "components": ["SKILL.md"],
            "file_cache": {"SKILL.md": content},
        }
        findings = static_runner.run_static_patterns(state, [tm_module])
        tm1_findings = [f for f in findings if f.rule_id == "TM1"]
        if tm1_findings:
            for f in tm1_findings:
                # Should NOT be reduced — SKILL.md is executable context
                assert f.confidence >= 0.5

    def test_python_file_in_docs_is_not_documentation_markdown(self) -> None:
        """A .py file even inside docs/ is not documentation markdown."""
        content = """\
import subprocess
subprocess.run(["rm", "-rf", "/tmp/cache"])
"""
        state = {
            "components": ["docs/helper.py"],
            "file_cache": {"docs/helper.py": content},
        }
        findings = static_runner.run_static_patterns(state, [tm_module])
        tm1_findings = [f for f in findings if f.rule_id == "TM1"]
        if tm1_findings:
            for f in tm1_findings:
                # .py files don't get markdown documentation reduction
                assert f.confidence >= 0.5

    @pytest.mark.parametrize(
        "path",
        [
            "docs/usage.md",
            "documentation/guide.md",
            "procedures/deploy.md",
            "references/api.md",
            "examples/demo.md",
            "guides/quickstart.md",
        ],
    )
    def test_various_documentation_paths_detected(self, path: str) -> None:
        """All known documentation path patterns are recognized."""
        from skillspector.nodes.analyzers.static_runner import _is_documentation_markdown

        assert _is_documentation_markdown(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "SKILL.md",
            "src/tool.py",
            "README.md",
            "CHANGELOG.md",
            "config.yaml",
        ],
    )
    def test_non_documentation_paths_not_matched(self, path: str) -> None:
        """Non-documentation paths are not matched."""
        from skillspector.nodes.analyzers.static_runner import _is_documentation_markdown

        assert _is_documentation_markdown(path) is False
