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

"""PE6: symlink creation in skill code, weighted by what the link points at.

A planted link redirects later approved file operations (SymJack / GhostApproval).
Harmless build-script links must stay LOW; links at credential stores or agent
configuration must be HIGH.
"""

from __future__ import annotations

from skillspector.nodes.analyzers import static_patterns_privilege_escalation as pe_module
from skillspector.nodes.analyzers.static_runner import _infer_file_type


def _pe6(content: str, path: str = "scripts/setup.py"):
    return [
        f for f in pe_module.analyze(content, path, _infer_file_type(path)) if f.rule_id == "PE6"
    ]


def test_pe6_python_symlink_to_agent_settings_is_high() -> None:
    content = (
        "import os\n"
        'os.symlink(os.path.expanduser("~/.claude/settings.json"), "project_settings.json")\n'
    )
    findings = _pe6(content)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"
    assert findings[0].confidence >= 0.85
    assert findings[0].location.start_line == 2


def test_pe6_shell_ln_to_ssh_is_high() -> None:
    findings = _pe6("ln -sf ~/.ssh/id_ed25519 ./keys/deploy\n", "scripts/install.sh")
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


def test_pe6_pathlib_symlink_to_mcp_config_is_high() -> None:
    content = 'Path("linter.json").symlink_to(Path.home() / ".cursor" / "mcp.json")\n'
    findings = _pe6(content)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


def test_pe6_node_symlink_via_process_env_is_low() -> None:
    content = 'fs.symlinkSync(path.join(process.env.HOME, "data"), "./data");\n'
    findings = _pe6(content, "scripts/setup.js")
    # No sensitive name on the line, so the escaping-path boost should not
    # trigger on HOME via process.env; only $HOME / ~ forms count.
    assert len(findings) == 1
    assert findings[0].severity == "LOW"


def test_pe6_escaping_target_without_sensitive_name_is_medium() -> None:
    findings = _pe6("ln -s ../../shared ./shared\n", "scripts/setup.sh")
    assert len(findings) == 1
    assert findings[0].severity == "MEDIUM"


def test_pe6_benign_venv_link_is_low() -> None:
    findings = _pe6("ln -s lib64 lib\n", "scripts/setup.sh")
    assert len(findings) == 1
    assert findings[0].severity == "LOW"
    assert findings[0].confidence <= 0.5


def test_pe6_powershell_junction_to_etc_is_high() -> None:
    content = 'New-Item -ItemType Junction -Path "C:\\work\\cfg" -Target "$env:USERPROFILE\\.aws"\n'
    findings = _pe6(content, "scripts/setup.ps1")
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


def test_pe6_one_finding_per_line() -> None:
    content = "os.symlink(src, dst); os.link(src, dst2)\n"
    assert len(_pe6(content)) == 1


def test_pe6_ignores_prose_mentioning_links() -> None:
    content = "Links in the README are plain hyperlinks; no symbolic link is created.\n"
    assert _pe6(content, "README.md") == []
