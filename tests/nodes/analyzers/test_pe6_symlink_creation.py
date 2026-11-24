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


def test_pe6_split_option_groups_before_symbolic_flag_are_high() -> None:
    for line in (
        "ln -f -s ~/.claude/settings.json ./project_settings.json\n",
        "ln --force -s ~/.claude/settings.json ./project_settings.json\n",
    ):
        findings = _pe6(line, "scripts/install.sh")
        assert len(findings) == 1, line
        assert findings[0].severity == "HIGH", line


def test_pe6_any_absolute_target_is_medium() -> None:
    assert _pe6("ln -s /opt/payload ./docs\n", "scripts/setup.sh")[0].severity == "MEDIUM"
    assert _pe6('os.symlink("/tmp/shared", "shared")\n')[0].severity == "MEDIUM"


def test_pe6_division_and_relative_paths_do_not_count_as_absolute() -> None:
    findings = _pe6("os.symlink(src, dst)  # ratio = a / b\n")
    assert findings[0].severity == "LOW"
    assert _pe6("ln -s ./lib64 lib\n", "scripts/setup.sh")[0].severity == "LOW"


def test_pe6_quoted_powershell_item_types_are_detected() -> None:
    for item_type in ('"SymbolicLink"', "'Junction'"):
        line = f'New-Item -ItemType {item_type} -Path cfg -Target "$env:USERPROFILE\\.claude"\n'
        findings = _pe6(line, "scripts/setup.ps1")
        assert len(findings) == 1, line
        assert findings[0].severity == "HIGH", line


def test_pe6_appsettings_json_is_not_agent_configuration() -> None:
    findings = _pe6("ln -s ./appsettings.json ./current.json\n", "scripts/setup.sh")
    assert findings[0].severity == "LOW"


def test_pe6_root_and_windows_targets_are_medium() -> None:
    assert _pe6("ln -s / ./root\n", "scripts/setup.sh")[0].severity == "MEDIUM"
    assert _pe6("ln -s /root ./r\n", "scripts/setup.sh")[0].severity == "MEDIUM"
    assert _pe6('os.symlink("/", "root")\n')[0].severity == "MEDIUM"
    assert _pe6('os.symlink("C:/shared/data", "data")\n')[0].severity == "MEDIUM"


def test_pe6_pathlib_division_in_arguments_is_not_a_root_target() -> None:
    assert _pe6("os.symlink(base / name, dst)\n")[0].severity == "LOW"


def test_pe6_sensitive_path_in_another_statement_does_not_raise_severity() -> None:
    line = 'os.symlink("lib64", "lib"); print("see ~/.claude/settings.json")\n'
    assert _pe6(line)[0].severity == "LOW"


def test_pe6_dense_single_line_is_linear_and_builds_context_once(monkeypatch) -> None:
    windows: list[int] = []
    real_window = pe_module._pe6_argument_window

    def recording_window(*args, **kwargs):
        result = real_window(*args, **kwargs)
        windows.append(len(result))
        return result

    contexts: list[int] = []
    real_context = pe_module.get_context_from_lines

    def recording_context(*args, **kwargs):
        contexts.append(1)
        return real_context(*args, **kwargs)

    monkeypatch.setattr(pe_module, "_pe6_argument_window", recording_window)
    monkeypatch.setattr(pe_module, "get_context_from_lines", recording_context)
    content = 'fs.symlinkSync("a","b");' * 3000 + "\n"

    findings = _pe6(content, "dist/bundle.js")

    assert len(findings) == 1
    assert len(windows) == 3000
    assert max(windows) <= pe_module._PE6_MAX_ARGUMENT_WINDOW
    assert len(contexts) == 1
