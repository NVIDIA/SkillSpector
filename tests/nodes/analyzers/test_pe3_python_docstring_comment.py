# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PE3 does not flag credential words in Python docstrings, comments, and messages.

Reproduces NVIDIA/SkillSpector #1144: the issue's demo skill raised six HIGH
PE3 findings, all from a module docstring, two guard docstrings, a comment,
and a message string that read nothing. Docstrings and comments never execute,
and a prose phrase inside a message string is documentation, not access, so
none of them is credential access. String literals that can reach a file API
stay visible: the pattern loop still fires on the path itself.
"""

from skillspector.nodes.analyzers import static_patterns_privilege_escalation as pe


def pe3(source, file_type="python"):
    return [f for f in pe.analyze(source, "script.py", file_type) if f.rule_id == "PE3"]


DEMO_TOOL = '''"""Helper for the demo skill.

Two values can live in ``<config_dir>/credentials.json`` (mode 600). Storage
in a flat JSON file (not a keychain) is intentional.
"""

from pathlib import Path


def safe_input_path(value: str) -> Path:
    """Reject absolute paths: a defence against an agent supplying ``/etc/passwd``."""
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe path")
    return path


def contained(value: str) -> Path:
    """Refuse a symlink such as ``notes.txt -> ~/.ssh/id_rsa`` that leaves the cwd."""
    path = Path(value).resolve()
    if not path.is_relative_to(Path.cwd().resolve()):
        raise ValueError("outside the working directory")
    return path


# Backend calls send the access token as a Bearer <redacted>
HINT = "use a citation URL that contains no username, password, or access token"
'''


def test_issue_1144_demo_skill_has_no_pe3_findings():
    findings = pe3(DEMO_TOOL)
    assert findings == []


def test_module_docstring_credential_paths_are_not_access():
    assert pe3('"""Store tokens in ~/.aws/credentials, never in /etc/passwd."""\n') == []


def test_class_and_function_docstrings_are_not_access():
    source = (
        "class Guard:\n"
        '    """Rejects ``~/.ssh/id_rsa`` and ``credentials.json`` outright."""\n'
        "    def check(self):\n"
        '        """Refuse a symlink to ``/etc/passwd``."""\n'
        "        return True\n"
    )
    assert pe3(source) == []


def test_same_line_function_docstring_is_not_access():
    assert pe3('def f(): """Mentioning /etc/passwd here reads nothing."""\n') == []


def test_bare_string_statement_is_a_block_comment():
    assert pe3('"TODO: rotate the key in ~/.ssh/id_rsa before release."\n') == []
    assert pe3('x = 1; "Block comment naming credentials.json."\n') == []


def test_docstring_code_example_is_not_access():
    source = (
        "def example():\n"
        '    """Usage::\n'
        "\n"
        '        open("~/.ssh/id_rsa").read()\n'
        '    """\n'
        "    return None\n"
    )
    assert pe3(source) == []


def test_comment_credential_words_are_not_access():
    assert pe3("# Never read ~/.ssh/id_rsa or /etc/passwd here.\n") == []
    assert pe3("# Backend calls send the access token as a Bearer <redacted>\n") == []


def test_credential_type_label_comment_is_not_access():
    # The disclosure-label exemption is markdown-only, but in a Python file
    # the label line is a comment, so #1144 silences it all the same.
    assert pe3("# **Credential Type(s):** [Other [Perforce access credentials]]\n") == []


def test_message_string_prose_is_not_access():
    assert (
        pe3('HINT = "use a citation URL that contains no username, password, or access token"\n')
        == []
    )
    assert pe3('raise ValueError("do not access the credentials file")\n') == []
    assert pe3('message = f"the access token expires soon {name}"\n') == []


def test_bare_fstring_literal_text_is_not_access():
    assert pe3('f"TODO: rotate ~/.ssh/id_rsa before release."\n') == []


def test_bare_fstring_replacement_field_still_executes():
    assert pe3("f\"{open('/etc/passwd').read()}\"\n")


def test_docstring_after_non_ascii_text_is_not_access():
    source = 'label = "\U0001f984\u00e9"; setting = "x"\n"""Docstring naming /etc/passwd."""\n'
    assert pe3(source) == []


def test_docstring_with_crlf_endings_is_not_access():
    assert pe3('"""Docstring naming /etc/passwd."""\r\nx = 1\r\n') == []


def test_unparseable_python_keeps_existing_detection():
    assert pe3('"""unterminated docstring naming /etc/passwd\n')


def test_non_python_files_keep_existing_detection():
    assert pe3("See /etc/passwd for the user list.\n", "markdown")


def test_assigned_credential_path_stays_visible():
    assert pe3('value = "~/.ssh/id_rsa"\n')
    assert pe3('value = "~/.aws/credentials"\n')


def test_credential_path_reaching_a_file_api_stays_visible():
    assert pe3('open("~/.ssh/id_rsa").read()\n')
    assert pe3('open("/etc/passwd").read()\n')
    assert pe3('from pathlib import Path\nPath("~/.ssh/id_rsa").read_text()\n')
    assert pe3('from pathlib import Path\nPath("credentials.json").read_text()\n')


def test_tool_own_credential_store_reaching_a_file_api_stays_visible():
    source = (
        "from pathlib import Path\n"
        "\n"
        "\n"
        "def config_dir() -> Path:\n"
        '    return Path.home() / ".config" / "demo"\n'
        "\n"
        "\n"
        'store = config_dir() / "credentials.json"\n'
        'store.write_bytes(b"{}")\n'
    )
    findings = pe3(source)
    assert [f.matched_text for f in findings] == ["credentials.json"]


def test_shell_command_string_with_credential_path_stays_visible():
    assert pe3('import subprocess\nsubprocess.run("cat ~/.ssh/id_rsa", shell=True)\n')
