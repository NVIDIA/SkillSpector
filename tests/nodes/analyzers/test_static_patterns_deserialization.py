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

"""Tests for static_patterns_deserialization: multi-language deserialization (DS1–DS4)."""

from __future__ import annotations

from skillspector.nodes.analyzers import static_patterns_deserialization


def _run(code: str, filename: str) -> list:
    state = {
        "components": [filename],
        "file_cache": {filename: code},
    }
    return static_patterns_deserialization.node(state)["findings"]


class TestPHP:
    def test_unserialize_produces_ds1(self):
        findings = _run("<?php $obj = unserialize($_GET['data']); ?>", "exploit.php")
        ds1 = [f for f in findings if f.rule_id == "DS1"]
        assert len(ds1) == 1
        assert ds1[0].severity == "HIGH"

    def test_clean_php_no_finding(self):
        findings = _run("<?php $obj = json_decode($_GET['data'], true); ?>", "clean.php")
        assert not any(f.rule_id == "DS1" for f in findings)


class TestRuby:
    def test_marshal_load_produces_ds2(self):
        findings = _run("data = Marshal.load(untrusted_blob)\n", "loader.rb")
        assert any(f.rule_id == "DS2" for f in findings)

    def test_marshal_restore_produces_ds2(self):
        findings = _run("data = Marshal.restore(untrusted_blob)\n", "loader.rb")
        assert any(f.rule_id == "DS2" for f in findings)

    def test_yaml_load_produces_ds3(self):
        findings = _run("obj = YAML.load(params[:payload])\n", "config.rb")
        assert any(f.rule_id == "DS3" for f in findings)

    def test_oj_load_produces_ds3(self):
        findings = _run("obj = Oj.load(input_str)\n", "config.rb")
        assert any(f.rule_id == "DS3" for f in findings)

    def test_yaml_safe_load_no_finding(self):
        findings = _run("obj = YAML.safe_load(params[:payload])\n", "config.rb")
        assert not any(f.rule_id == "DS3" for f in findings)


class TestJavaScript:
    def test_node_serialize_require_produces_ds4(self):
        code = "const serialize = require('node-serialize');\nserialize.unserialize(payload);\n"
        findings = _run(code, "handler.js")
        assert any(f.rule_id == "DS4" for f in findings)

    def test_unserialize_method_produces_ds4(self):
        findings = _run("obj.unserialize(userInput);\n", "handler.ts")
        assert any(f.rule_id == "DS4" for f in findings)

    def test_json_parse_no_finding(self):
        findings = _run("const obj = JSON.parse(userInput);\n", "handler.js")
        assert not any(f.rule_id.startswith("DS") for f in findings)


class TestLanguageGating:
    def test_python_file_not_scanned_here(self):
        # Python is owned by behavioral_ast (AST10) / taint (TT6); this module skips it
        # so it does not emit duplicate, lower-quality findings.
        findings = _run("import pickle\npickle.loads(data)\n", "script.py")
        assert findings == []

    def test_php_pattern_does_not_fire_on_ruby(self):
        # bare unserialize() is PHP-only; a Ruby file must not match DS1.
        findings = _run("x = unserialize(data)\n", "thing.rb")
        assert not any(f.rule_id == "DS1" for f in findings)

    def test_unknown_extension_no_findings(self):
        findings = _run("unserialize(data)\n", "notes.txt")
        assert findings == []


class TestFindingMetadata:
    def test_finding_has_remediation_and_context(self):
        findings = _run("<?php unserialize($x); ?>", "x.php")
        ds1 = [f for f in findings if f.rule_id == "DS1"]
        assert ds1[0].remediation
        assert ds1[0].context is not None
        assert ds1[0].category == "Insecure Deserialization"
