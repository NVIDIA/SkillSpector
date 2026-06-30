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

"""Tests for behavioral_ast analyzer: AST-based dangerous execution detection."""

from __future__ import annotations

from skillspector.nodes.analyzers import behavioral_ast


def _run(code: str, filename: str = "script.py") -> list:
    state = {
        "components": [filename],
        "file_cache": {filename: code},
    }
    result = behavioral_ast.node(state)
    return result["findings"]


class TestExecDetection:
    def test_exec_produces_ast1(self):
        findings = _run('exec("print(1)")')
        ast1 = [f for f in findings if f.rule_id == "AST1"]
        assert len(ast1) == 1
        assert ast1[0].severity == "HIGH"
        assert ast1[0].file == "script.py"
        assert ast1[0].start_line == 1

    def test_exec_with_variable(self):
        findings = _run("code = 'x = 1'\nexec(code)")
        assert any(f.rule_id == "AST1" for f in findings)


class TestEvalDetection:
    def test_eval_produces_ast2(self):
        findings = _run('result = eval("2 + 2")')
        ast2 = [f for f in findings if f.rule_id == "AST2"]
        assert len(ast2) == 1
        assert ast2[0].severity == "HIGH"

    def test_eval_in_function(self):
        code = "def run(expr):\n    return eval(expr)\n"
        findings = _run(code)
        assert any(f.rule_id == "AST2" for f in findings)


class TestDunderImport:
    def test_dunder_import_produces_ast3(self):
        findings = _run('mod = __import__("os")')
        ast3 = [f for f in findings if f.rule_id == "AST3"]
        assert len(ast3) == 1
        assert ast3[0].severity == "MEDIUM"


class TestSubprocess:
    def test_subprocess_run_produces_ast4(self):
        code = 'import subprocess\nsubprocess.run(["ls", "-la"])'
        findings = _run(code)
        ast4 = [f for f in findings if f.rule_id == "AST4"]
        assert len(ast4) == 1
        assert ast4[0].severity == "MEDIUM"

    def test_subprocess_popen_produces_ast4(self):
        code = 'import subprocess\nsubprocess.Popen(["cat", "/etc/passwd"])'
        findings = _run(code)
        assert any(f.rule_id == "AST4" for f in findings)

    def test_subprocess_check_output_produces_ast4(self):
        code = 'import subprocess\nsubprocess.check_output(["whoami"])'
        findings = _run(code)
        assert any(f.rule_id == "AST4" for f in findings)


class TestOsSystem:
    def test_os_system_produces_ast5(self):
        code = 'import os\nos.system("rm -rf /")'
        findings = _run(code)
        ast5 = [f for f in findings if f.rule_id == "AST5"]
        assert len(ast5) == 1
        assert ast5[0].severity == "HIGH"

    def test_os_popen_produces_ast5(self):
        code = 'import os\nos.popen("whoami")'
        findings = _run(code)
        assert any(f.rule_id == "AST5" for f in findings)


class TestCompile:
    def test_compile_produces_ast6(self):
        code = 'code = compile("x = 1", "<string>", "exec")'
        findings = _run(code)
        ast6 = [f for f in findings if f.rule_id == "AST6"]
        assert len(ast6) == 1
        assert ast6[0].severity == "MEDIUM"


class TestDynamicGetattr:
    def test_getattr_with_variable_produces_ast7(self):
        code = "attr = 'secret'\nval = getattr(obj, attr)"
        findings = _run(code)
        ast7 = [f for f in findings if f.rule_id == "AST7"]
        assert len(ast7) == 1
        assert ast7[0].severity == "LOW"

    def test_getattr_with_literal_no_finding(self):
        code = 'val = getattr(obj, "name")'
        findings = _run(code)
        assert not any(f.rule_id == "AST7" for f in findings)


class TestReflectiveGetattrExec:
    """getattr(obj, "<sink>")(...) is a reflective handle on an exec/os sink.

    It evades AST1/AST5 (the inner getattr has a *constant* name so AST7 is skipped,
    and the outer call's func is an ast.Call whose name does not resolve), so it must
    be caught directly as AST9.
    """

    def test_getattr_os_system_produces_ast9(self):
        findings = _run("import os\ngetattr(os, 'system')('id')")
        ast9 = [f for f in findings if f.rule_id == "AST9"]
        assert len(ast9) == 1
        assert ast9[0].severity == "HIGH"

    def test_getattr_builtins_exec_produces_ast9(self):
        findings = _run("import builtins\ngetattr(builtins, 'exec')(payload)")
        assert any(f.rule_id == "AST9" for f in findings)

    def test_getattr_eval_double_quotes_produces_ast9(self):
        findings = _run('import builtins\ngetattr(builtins, "eval")("2+2")')
        assert any(f.rule_id == "AST9" for f in findings)

    def test_getattr_os_popen_produces_ast9(self):
        findings = _run("import os\nhandle = getattr(os, 'popen')('whoami')")
        assert any(f.rule_id == "AST9" for f in findings)

    def test_reflective_getattr_does_not_emit_ast7(self):
        # A constant name must not also trip the non-literal AST7 rule.
        findings = _run("import os\ngetattr(os, 'system')('id')")
        assert not any(f.rule_id == "AST7" for f in findings)

    def test_benign_constant_attr_no_ast9(self):
        # Common, safe reflective access must stay unflagged (near-zero false positives).
        for name in ("name", "timeout", "value", "data", "run", "compile"):
            findings = _run(f"v = getattr(config, '{name}')")
            assert not any(f.rule_id == "AST9" for f in findings), name


class TestDangerousChains:
    def test_exec_compile_chain_produces_ast8(self):
        code = 'exec(compile("x = 1", "<string>", "exec"))'
        findings = _run(code)
        ast8 = [f for f in findings if f.rule_id == "AST8"]
        assert len(ast8) >= 1
        assert ast8[0].severity == "CRITICAL"
        assert "compile" in ast8[0].message

    def test_eval_base64_chain_produces_ast8(self):
        code = "import base64\neval(base64.b64decode(payload))"
        findings = _run(code)
        ast8 = [f for f in findings if f.rule_id == "AST8"]
        assert len(ast8) >= 1
        assert "base64" in ast8[0].message

    def test_exec_urllib_chain_produces_ast8(self):
        code = "import urllib.request\nexec(urllib.request.urlopen(url).read())"
        findings = _run(code)
        ast8 = [f for f in findings if f.rule_id == "AST8"]
        assert len(ast8) >= 1

    def test_exec_import_chain_produces_ast8(self):
        code = 'exec(__import__("os").system("id"))'
        findings = _run(code)
        ast8 = [f for f in findings if f.rule_id == "AST8"]
        assert len(ast8) >= 1


class TestEdgeCases:
    def test_non_python_files_skipped(self):
        state = {
            "components": ["readme.md"],
            "file_cache": {"readme.md": "exec('hello')"},
        }
        result = behavioral_ast.node(state)
        assert result["findings"] == []

    def test_syntax_error_skipped(self):
        findings = _run("def broken(\n")
        assert findings == []

    def test_empty_file_no_findings(self):
        findings = _run("")
        assert findings == []

    def test_safe_code_no_findings(self):
        code = "import json\ndata = json.loads('{}')\nprint(data)\n"
        findings = _run(code)
        assert findings == []

    def test_finding_has_remediation(self):
        findings = _run('exec("x = 1")')
        assert findings[0].remediation is not None
        assert len(findings[0].remediation) > 0

    def test_finding_has_context(self):
        findings = _run('x = 1\nexec("y = 2")\nz = 3')
        ast1 = [f for f in findings if f.rule_id == "AST1"]
        assert ast1[0].context is not None

    def test_finding_has_matched_text(self):
        findings = _run('exec("code")')
        assert findings[0].matched_text is not None

    def test_empty_components(self):
        state = {"components": [], "file_cache": {}}
        result = behavioral_ast.node(state)
        assert result["findings"] == []

    def test_missing_file_in_cache(self):
        state = {"components": ["missing.py"], "file_cache": {}}
        result = behavioral_ast.node(state)
        assert result["findings"] == []


class TestImportAliasEvasion:
    """Dangerous calls must be detected through ``from ... import`` and ``import ... as``.

    A skill can otherwise dodge the prefix-based matching simply by importing the
    primitive under another name (e.g. ``from os import system``).
    """

    def test_from_os_import_system(self):
        findings = _run("from os import system\nsystem('id')")
        assert any(f.rule_id == "AST5" for f in findings)

    def test_import_os_as_alias(self):
        findings = _run("import os as o\no.system('id')")
        assert any(f.rule_id == "AST5" for f in findings)

    def test_from_subprocess_import_run(self):
        findings = _run("from subprocess import run\nrun(['id'])")
        assert any(f.rule_id == "AST4" for f in findings)

    def test_import_subprocess_as_alias(self):
        findings = _run("import subprocess as sp\nsp.Popen(['id'])")
        assert any(f.rule_id == "AST4" for f in findings)

    def test_aliased_chain_via_from_import(self):
        """``from base64 import b64decode; eval(b64decode(...))`` is still a chain (AST8)."""
        findings = _run("from base64 import b64decode\neval(b64decode(payload))")
        ast8 = [f for f in findings if f.rule_id == "AST8"]
        assert len(ast8) >= 1
        assert "base64" in ast8[0].message

    def test_aliased_safe_import_no_false_positive(self):
        findings = _run("import json as j\ndata = j.loads('{}')\nprint(data)\n")
        assert findings == []


class TestMultipleFindings:
    def test_multiple_dangerous_calls_in_one_file(self):
        code = (
            "import os, subprocess\n"
            'exec("x = 1")\n'
            'eval("2 + 2")\n'
            'os.system("ls")\n'
            'subprocess.run(["id"])\n'
        )
        findings = _run(code)
        rule_ids = {f.rule_id for f in findings}
        assert "AST1" in rule_ids
        assert "AST2" in rule_ids
        assert "AST4" in rule_ids
        assert "AST5" in rule_ids


# ── builtins / importlib import-chain evasion ─────────────────────────


class TestBuiltinsImportEvasion:
    """Dangerous builtins hidden behind the ``builtins`` module must still alert.

    The analyzer matches dangerous builtins by their bare name (``exec``/``eval``/
    ``compile``/``__import__``). Writing ``from builtins import exec`` or
    ``import builtins; builtins.exec(...)`` resolves, through the import-alias map,
    to the qualified spelling ``builtins.exec`` — which would slip past the bare-name
    checks unless it is canonicalized back. Since ``builtins.exec is exec``, the
    collapse is semantically exact. Complements the ``getattr`` branch (PR #166).
    """

    def test_from_builtins_import_exec(self):
        """``from builtins import exec; exec(code)`` must still raise AST1."""
        findings = _run("from builtins import exec\nexec('x = 1')\n")
        assert any(f.rule_id == "AST1" for f in findings)

    def test_from_builtins_import_eval(self):
        """``from builtins import eval`` must still raise AST2."""
        findings = _run("from builtins import eval\neval('2 + 2')\n")
        assert any(f.rule_id == "AST2" for f in findings)

    def test_from_builtins_import_compile(self):
        """``from builtins import compile`` must still raise AST6."""
        findings = _run("from builtins import compile\ncompile('x', '<s>', 'exec')\n")
        assert any(f.rule_id == "AST6" for f in findings)

    def test_from_builtins_import_dunder_import(self):
        """``from builtins import __import__`` must still raise AST3."""
        findings = _run("from builtins import __import__\n__import__('os')\n")
        assert any(f.rule_id == "AST3" for f in findings)

    def test_import_builtins_dot_exec(self):
        """``import builtins; builtins.exec(...)`` must still raise AST1."""
        findings = _run("import builtins\nbuiltins.exec('x = 1')\n")
        assert any(f.rule_id == "AST1" for f in findings)

    def test_import_builtins_as_alias_dot_exec(self):
        """``import builtins as b2; b2.exec(...)`` must still raise AST1."""
        findings = _run("import builtins as b2\nb2.exec('x = 1')\n")
        assert any(f.rule_id == "AST1" for f in findings)

    def test_from_builtins_import_exec_as_alias(self):
        """``from builtins import exec as e; e(...)`` must still raise AST1."""
        findings = _run("from builtins import exec as e\ne('x = 1')\n")
        assert any(f.rule_id == "AST1" for f in findings)

    def test_user_module_exec_helper_no_false_positive(self):
        """A benign helper merely *named* like a sink must not match (FP-neighbor).

        ``from mymod import exec_helper; exec_helper()`` imports an unrelated
        third-party callable — it is not ``builtins.exec`` and must stay clean.
        """
        findings = _run("from mymod import exec_helper\nexec_helper()\n")
        assert findings == []


class TestImportlibDynamicChainEvasion:
    """``importlib.import_module('mod').attr(...)`` is a dynamic-import sink chain.

    It mirrors ``__import__('mod')`` but lets the dangerous module name live in a
    string literal so it never appears as a static ``import``. The chain is resolved
    to the canonical dotted sink (``os.system``/``subprocess.run``) so it re-enters
    the existing ``os.``/``subprocess.`` sink ladders.
    """

    def test_importlib_import_module_os_system(self):
        """``importlib.import_module('os').system(...)`` must raise AST5."""
        findings = _run("import importlib\nimportlib.import_module('os').system('id')\n")
        assert any(f.rule_id == "AST5" for f in findings)

    def test_importlib_import_module_subprocess_run(self):
        """``importlib.import_module('subprocess').run(...)`` must raise AST4."""
        findings = _run("import importlib\nimportlib.import_module('subprocess').run(['id'])\n")
        assert any(f.rule_id == "AST4" for f in findings)

    def test_from_importlib_import_module_os_system(self):
        """Bare-imported ``import_module('os').system(...)`` must raise AST5."""
        findings = _run("from importlib import import_module\nimport_module('os').system('id')\n")
        assert any(f.rule_id == "AST5" for f in findings)

    def test_importlib_import_module_benign_no_false_positive(self):
        """A benign dynamic import (``json.loads``) must not match a sink ladder."""
        findings = _run("import importlib\nimportlib.import_module('json').loads('{}')\n")
        assert findings == []
