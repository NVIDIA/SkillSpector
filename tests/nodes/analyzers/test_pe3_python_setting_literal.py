# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An exact Python setting name is not credential-store access."""

import sys

import pytest

from skillspector.nodes.analyzers import static_patterns_privilege_escalation as pe
from skillspector.python_ast import clear_python_ast_cache, prewarm_python_ast_cache


def pe3(source, file_type="python"):
    return [f for f in pe.analyze(source, "script.py", file_type) if f.rule_id == "PE3"]


@pytest.mark.parametrize(
    "source",
    [
        'allowed = {"GLAB_USE_KEYRING"}\n',
        'enabled = os.environ.get("GLAB_USE_KEYRING")\n',
        "setting = 'GLAB_USE_KEYRING'\n",
        'setting = r"GLAB_USE_KEYRING"\n',
        'setting = f"GLAB_USE_KEYRING"\n',
        'setting = """GLAB_USE_KEYRING"""\n',
        'setting = ("GLAB_USE_" "KEYRING")\n',
        'label = "🦄é"; allowed = {"GLAB_USE_KEYRING"}\n',
        'label = "é"\r\nallowed = {"GLAB_USE_KEYRING"}\r\n',
        'label = "é\u2028🦄"; allowed = {"GLAB_USE_KEYRING"}\n',
        'label = "é\u2029🦄"; allowed = {"GLAB_USE_KEYRING"}\n',
        'label = "é\f🦄"; allowed = {"GLAB_USE_KEYRING"}\n',
    ],
)
def test_exact_setting_literal_is_not_credential_access(source):
    assert pe3(source) == []


@pytest.mark.parametrize(
    "source",
    [
        'keyring.get_password("service", "user")\n',
        "value = settings.GLAB_USE_KEYRING\n",
        "value = GLAB_USE_KEYRING\n",
        'value = "prefix_GLAB_USE_KEYRING"\n',
        'value = "GLAB_USE_KEYRING_suffix"\n',
        'value = f"GLAB_USE_KEYRING{suffix}"\n',
        'value = f"{prefix}GLAB_USE_KEYRING"\n',
        "value = f\"{keyring.get_password('s', 'u')}GLAB_USE_KEYRING\"\n",
        'value = f"{secret:GLAB_USE_KEYRING}"\n',
        'value = "GLAB_USE_KEYRING" + keyring.get_password("s", "u")\n',
        'allowed = {"GLAB_USE_KEYRING"\n',
        'value = "~/.ssh/id_rsa"\n',
        'value = "~/.aws/credentials"\n',
    ],
)
def test_unproven_or_sensitive_references_stay_visible(source):
    assert pe3(source)


@pytest.mark.skipif(sys.version_info < (3, 14), reason="Template strings require Python 3.14")
@pytest.mark.parametrize(
    "source",
    [
        'value = t"GLAB_USE_KEYRING{suffix}"\n',
        'value = t"{prefix}GLAB_USE_KEYRING"\n',
        "value = t\"{keyring.get_password('s', 'u')}GLAB_USE_KEYRING\"\n",
        'value = t"{secret:GLAB_USE_KEYRING}"\n',
    ],
)
def test_template_interpolation_stays_visible(source):
    assert pe3(source)


def test_setting_does_not_hide_later_access_on_unicode_line():
    source = 'label = "🦄é"; setting = "GLAB_USE_KEYRING"; value = keyring.get_password("s", "u")\n'
    findings = pe3(source)
    assert len(findings) == 1
    assert findings[0].matched_text == "keyring"


def test_non_python_setting_reference_keeps_existing_detection():
    assert pe3('setting = "GLAB_USE_KEYRING"', "shell")


def test_node_uses_the_same_setting_qualification():
    result = pe.node(
        {
            "components": ["script.py"],
            "file_cache": {"script.py": 'allowed = {"GLAB_USE_KEYRING"}\n'},
        }
    )
    assert not [f for f in result["findings"] if f.rule_id == "PE3"]


def test_windowed_setting_keeps_constructed_credential_access():
    source = (
        "from os.path import join as j\n"
        "def configure():\n"
        + "    # filler\n" * 25000
        + '    allowed = {"GLAB_USE_KEYRING"}\n'
        + '    credential = j("/etc", "passwd")\n'
    )
    cache = {"script.py": source}
    key = prewarm_python_ast_cache(["script.py"], cache)
    try:
        findings = pe.node(
            {"components": ["script.py"], "file_cache": cache, "python_ast_cache_key": key}
        )["findings"]
    finally:
        clear_python_ast_cache(key)
    pe3_findings = [f for f in findings if f.rule_id == "PE3"]
    assert pe3_findings
    assert all(f.matched_text == "/etc/passwd" for f in pe3_findings)
