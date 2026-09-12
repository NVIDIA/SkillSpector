# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused coverage for issue #475's straight-line Python binding form."""

from __future__ import annotations

import ast
from collections import Counter

import pytest

from skillspector.artifacts import normalized_security_view
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm_module
from skillspector.nodes.analyzers import static_python_shell_truthiness as shell_truthiness
from skillspector.nodes.deduplicate import deduplicate
from skillspector.python_ast import (
    MAX_PYTHON_SHEBANG_CHARS,
    PythonSourceClassification,
    classify_python_source,
)


def _run(content: str, path: str = "run.py") -> dict:
    return tm_module.node(
        {
            "components": [path],
            "file_cache": {path: content},
        }
    )


def _tm1(content: str, path: str = "run.py") -> list:
    return [finding for finding in _run(content, path)["findings"] if finding.rule_id == "TM1"]


def test_function_finalizer_safety_requires_retained_name_provenance() -> None:
    statement = ast.parse(
        "def holder(value=opaque, *, other=marker) -> annotation:\n    pass\n"
    ).body[0]

    assert isinstance(statement, ast.FunctionDef)
    assert not shell_truthiness._function_result_is_finalizer_safe(statement, set())
    assert shell_truthiness._function_result_is_finalizer_safe(
        statement,
        {"opaque", "marker", "annotation"},
    )


def test_class_finalizer_safety_requires_retained_member_provenance() -> None:
    statement = ast.parse(
        "class Holder:\n"
        "    values = [opaque]\n"
        "    def method(self, marker=default):\n"
        "        pass\n"
    ).body[0]

    assert isinstance(statement, ast.ClassDef)
    assert not shell_truthiness._class_result_is_finalizer_safe(statement, set())
    assert shell_truthiness._class_result_is_finalizer_safe(
        statement,
        {"opaque", "default"},
    )


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("True", id="issue-body-boolean"),
        pytest.param("'True'", id="reporter-attachment-string"),
    ],
)
def test_issue_475_multiline_binding_matches_direct_tm1(value: str) -> None:
    findings = _tm1(
        "import subprocess\n"
        "command = f'python a.py'\n"
        f"a = {value}\n"
        "result = subprocess.run(\n"
        "    command,\n"
        "    shell=a,\n"
        "    capture_output=True,\n"
        "    text=True,\n"
        ")\n"
    )

    assert len(findings) == 1
    assert findings[0].start_line == 4
    assert findings[0].severity == "HIGH"
    assert findings[0].confidence == pytest.approx(0.9)
    assert "shell=a" in findings[0].matched_text


@pytest.mark.parametrize(
    ("path", "prefix", "expected_line"),
    [
        pytest.param("run.pyw", "", 3, id="pyw"),
        pytest.param("runner", "#!/usr/bin/python3\n", 4, id="direct-python"),
        pytest.param(
            "runner",
            "#!/usr/bin/python3.14t\n",
            4,
            id="direct-free-threaded-python",
        ),
        pytest.param("runner", "#!/usr/bin/env python3\n", 4, id="env-python"),
        pytest.param(
            "runner",
            "#!/usr/bin/python3# xnu comment\n",
            4,
            id="darwin-direct-python-comment",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/python3 -W ignore /tmp/other.py\n",
            4,
            id="platform-dependent-python-warning-argument",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -i python3\n",
            4,
            id="darwin-env-ignore-environment",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -u SKILLSPECTOR_NAME python3\n",
            4,
            id="darwin-env-unset",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env python3 -I\n",
            4,
            id="darwin-python-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env python3 -vv\n",
            4,
            id="darwin-python-repeated-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env python3 -IB\n",
            4,
            id="darwin-python-clustered-options",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env python3 -bbb\n",
            4,
            id="darwin-python-repeated-bytes-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env python3 -OOO\n",
            4,
            id="darwin-python-repeated-optimize-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -i python3 # xnu comment\n",
            4,
            id="darwin-env-comment",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env python3.13t\n",
            4,
            id="env-free-threaded-python",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S python3 -B\n",
            4,
            id="env-split-string",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/python3 -t\n",
            4,
            id="direct-legacy-tab-compatibility-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S python3 -tt\n",
            4,
            id="env-legacy-tab-compatibility-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S python3 -tEB\n",
            4,
            id="env-clustered-legacy-tab-compatibility-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -Spython3 -B\n",
            4,
            id="env-attached-split-string",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S 'python3' -B\n",
            4,
            id="env-single-quoted-interpreter",
        ),
        pytest.param(
            "runner",
            '#!/usr/bin/env -S "python3" -B\n',
            4,
            id="env-double-quoted-interpreter",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S-P/usr/bin:/bin python3\n",
            4,
            id="env-split-attached-path-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -i python3\n",
            4,
            id="env-split-ignore-environment-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -i-v python3\n",
            4,
            id="env-freebsd-clustered-compatibility-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S-iv -P/usr/bin:/bin python3\n",
            4,
            id="env-split-clustered-flags",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -a alternate python3\n",
            4,
            id="env-split-argv0-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S --argv0=alternate python3\n",
            4,
            id="env-split-long-argv0-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S --argv0= python3\n",
            4,
            id="env-split-empty-argv0-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -L root python3\n",
            4,
            id="env-split-freebsd-login-class-separate-operand",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -Lroot python3\n",
            4,
            id="env-split-freebsd-login-class-attached-operand",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -ivLroot python3\n",
            4,
            id="env-split-freebsd-clustered-login-class-operand",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S =x python3\n",
            4,
            id="env-split-gnu-empty-name-assignment",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S FOO=bar =x python3\n",
            4,
            id="env-split-gnu-empty-name-assignment-after-assignment",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -i =x python3\n",
            4,
            id="env-split-gnu-empty-name-assignment-after-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -- - python3\n",
            4,
            id="env-split-gnu-post-terminator-legacy-dash",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -- - FOO=bar python3\n",
            4,
            id="env-split-gnu-post-terminator-dash-before-assignment",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S --env0-from=environment python3\n",
            4,
            id="env-split-gnu-environment-file-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S PYTHONSAFEPATH=1 python3\n",
            4,
            id="env-split-assignment",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S PATH=/usr/bin:${PATH} python3" "\n",
            4,
            id="env-split-dynamic-assignment",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -- python3\n",
            4,
            id="env-split-option-terminator",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -- PYTHONSAFEPATH=1 python3\n",
            4,
            id="env-split-assignment-after-option-terminator",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -u PYTHONPATH python3\n",
            4,
            id="env-split-separate-option-operand",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -vS python3\n",
            4,
            id="env-verbose-split-string",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -iS python3\n",
            4,
            id="env-ignore-split-string",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -ivS python3\n",
            4,
            id="env-clustered-outer-split-string",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -i-vSpython3\n",
            4,
            id="env-freebsd-clustered-outer-split-string",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env --split-string=python3\n",
            4,
            id="env-long-split-string",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env --spl=python3\n",
            4,
            id="gnu-only-outer-abbreviated-split-string",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S --split-string=python3\n",
            4,
            id="gnu-only-nested-long-split-string",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S --ignore-environment python3\n",
            4,
            id="gnu-only-ignore-environment",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S --unset=FOO python3\n",
            4,
            id="gnu-only-unset",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env --split=python3\n",
            4,
            id="env-abbreviated-long-split-string",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S --chd=/tmp python3\n",
            4,
            id="env-abbreviated-long-chdir",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S --i python3\n",
            4,
            id="env-freebsd-double-dash-cluster",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S --unknown python3\n",
            4,
            id="env-freebsd-double-dash-unset-operand",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S FOO=bar\\ baz python3\n",
            4,
            id="env-freebsd-escaped-space",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S FOO=café python3\n",
            4,
            id="env-unicode-assignment",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S python3\_-B" "\n",
            4,
            id="env-escaped-argument-separator",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S python3\c node" "\n",
            4,
            id="env-escaped-string-terminator",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -Spython3\n",
            4,
            id="nested-env-split-string",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S -i${SKILLSPECTOR_EMPTY} python3" "\n",
            4,
            id="env-dynamic-empty-option-suffix",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S -u${SKILLSPECTOR_MAYBE} python3" "\n",
            4,
            id="env-dynamic-attached-unset-operand",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S -u ${SKILLSPECTOR_MAYBE} python3" "\n",
            4,
            id="env-dynamic-separate-unset-operand",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S py${SKILLSPECTOR_EMPTY}thon3" "\n",
            4,
            id="env-dynamic-empty-interpreter-fragment",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S -P${PATH} python3" "\n",
            4,
            id="env-dynamic-bsd-path-operand",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S -C${PWD} /usr/bin/python3" "\n",
            4,
            id="env-dynamic-chdir-operand",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S -P ${PATH} /usr/bin/python3" "\n",
            4,
            id="env-dynamic-separate-path-operand",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S -u ${SKILLSPECTOR_NAME}#suffix python3" "\n",
            4,
            id="env-dynamic-prefix-before-comment-marker",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S /opt/${SKILLSPECTOR_ROOT}/python3" "\n",
            4,
            id="env-fixed-python-basename-after-dynamic-path",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S /opt/${SKILLSPECTOR_ROOT}/pypy3" "\n",
            4,
            id="env-fixed-pypy-basename-after-dynamic-path",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S -- -X${SKILLSPECTOR_VALUE}=1 python3" "\n",
            4,
            id="env-dynamic-dash-assignment-after-terminator",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S -- -x${SKILLSPECTOR_ROOT}/python3" "\n",
            4,
            id="env-dynamic-dash-path-after-terminator",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S -u${SKILLSPECTOR_NAME} /usr/bin/true python3" "\n",
            4,
            id="env-dynamic-attached-operand-python-after-shift",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S -u ${SKILLSPECTOR_NAME} /usr/bin/true python3" "\n",
            4,
            id="env-dynamic-separate-operand-python-after-shift",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S --unset ${SKILLSPECTOR_NAME} python3 /usr/bin/true"
            "\n",
            4,
            id="env-dynamic-long-operand-python-before-shift",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S - -i python3\n",
            4,
            id="platform-dependent-lone-dash",
        ),
        pytest.param("runner", "#!/bin/env python3\n", 4, id="bin-env-python"),
        pytest.param(
            "runner",
            r'#!/usr/bin/env -S -S "FOO=a\\\nb /usr/bin/python3"' "\n",
            4,
            id="nested-env-split-freebsd-escaped-newline",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S --unset /usr/bin/true python3\n",
            4,
            id="platform-dependent-double-dash-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/python3"
            + " " * (MAX_PYTHON_SHEBANG_CHARS - len("#!/usr/bin/python3") + 1)
            + "\n",
            4,
            id="over-maximum-length-shebang",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S ${SKILLSPECTOR_INTERPRETER}" "\n",
            4,
            id="dynamic-env-utility",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S X${SKILLSPECTOR_ASSIGNMENT} python3" "\n",
            4,
            id="dynamic-env-token-role",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S python3 ${SKILLSPECTOR_ARGUMENT}" "\n",
            4,
            id="dynamic-python-pre-script-argument",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S python3 -- ${SKILLSPECTOR_ARGUMENT}" "\n",
            4,
            id="dynamic-python-argument-after-option-terminator",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S python3 -- ${SKILLSPECTOR_ARGUMENT} /tmp/other.py"
            "\n",
            4,
            id="dynamic-python-argument-before-fixed-other-script",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env /tmp/a b/python3\n",
            4,
            id="plain-env-opaque-python-path-with-spaces",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/python3 -c__import__('builtins').exec(open(__import__('sys').argv[1]).read())\n",
            4,
            id="python-command-can-load-appended-source",
        ),
        pytest.param(
            "eval(input())",
            "#!/usr/bin/env -S python3 -c\n",
            4,
            id="python-command-executes-active-appended-source-spelling",
        ),
        pytest.param(
            "evilmod",
            "#!/usr/bin/env -S python3 -m\n",
            4,
            id="python-module-can-select-sibling-module",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/python3 -ic\n",
            4,
            id="python-forced-interactive-command-can-recover-appended-source",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S python3 -i -m\n",
            4,
            id="env-python-forced-interactive-module-can-recover-appended-source",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/python3 -\n",
            4,
            id="python-stdin-can-load-appended-source",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S python3 -\n",
            4,
            id="env-python-stdin-can-load-appended-source",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/python3 -X\n",
            4,
            id="python-xoption-operand-exposes-appended-source",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S python3 -X\n",
            4,
            id="env-python-xoption-operand-exposes-appended-source",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/python3 -W\n",
            4,
            id="python-warning-operand-exposes-appended-source",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S python3 -W\n",
            4,
            id="env-python-warning-operand-exposes-appended-source",
        ),
        pytest.param(
            "default",
            "#!/usr/bin/python3 --check-hash-based-pycs\n",
            4,
            id="python-hash-option-consumes-default-source-name",
        ),
        pytest.param(
            "tools/always",
            "#!/usr/bin/env -S python3 --check-hash-based-pycs\n",
            4,
            id="env-python-hash-option-consumes-subdirectory-source-name",
        ),
        pytest.param(
            "bundle.zip!/never",
            "#!/usr/bin/env -S python3 --check-hash-based-pycs\n",
            4,
            id="env-python-hash-option-consumes-nested-source-name",
        ),
        pytest.param(
            "-h",
            "#!/usr/bin/python3\n",
            4,
            id="direct-implicit-source-can-be-help-option",
        ),
        pytest.param(
            "tools/-V",
            "#!/usr/bin/env -S python3\n",
            4,
            id="env-implicit-source-can-be-version-option",
        ),
        pytest.param(
            "tools/-options/runner",
            "#!/usr/bin/python3 -B\n",
            4,
            id="direct-implicit-source-has-dash-intermediate-component",
        ),
        pytest.param(
            "bundle.zip!/-nested/runner",
            "#!/usr/bin/env -S python3\n",
            4,
            id="env-implicit-nested-source-has-dash-component",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S python3 --check-hash-based-pycs default\n",
            4,
            id="unversioned-python-hash-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S python3 -P\n",
            4,
            id="unversioned-python-safe-path-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/python3 runner\n",
            4,
            id="python-selects-analyzed-source-explicitly",
        ),
        pytest.param(
            "Runner",
            "#!/usr/bin/python3 runner\n",
            4,
            id="python-selects-case-insensitive-source-alias",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/Python3\n",
            4,
            id="python-interpreter-case-alias",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/Env python3\n",
            4,
            id="env-path-case-alias",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/../bin/env python3\n",
            4,
            id="env-path-dot-segment-alias",
        ),
        pytest.param(
            "runner",
            "#!//usr/bin/env python3\n",
            4,
            id="env-path-double-leading-slash-alias",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env python3 node\n",
            4,
            id="external-python-script-via-plain-env",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/python3 /tmp/other.py\n",
            4,
            id="external-python-script-direct",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S python3 /tmp/other.py\n",
            4,
            id="external-python-script-via-env-split",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S python3 argument\ with-space" "\n",
            4,
            id="external-python-script-with-escaped-space",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S -u${SKILLSPECTOR_NAME} python3 /usr/bin/true" "\n",
            4,
            id="external-script-after-dynamic-env-option",
        ),
        pytest.param(
            "runner",
            r'#!/usr/bin/env -S -u "${SKILLSPECTOR_NAME}" python3 /usr/bin/true' "\n",
            4,
            id="external-script-after-quoted-env-option",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S -uPREFIX${SKILLSPECTOR_NAME} python3 /usr/bin/true"
            "\n",
            4,
            id="external-script-after-literal-env-option",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S /opt/${SKILLSPECTOR_ROOT}/python3 /usr/bin/true" "\n",
            4,
            id="external-script-after-dynamic-python-path",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S -- -x${SKILLSPECTOR_ROOT}/python3 /usr/bin/true" "\n",
            4,
            id="external-script-after-dynamic-dash-path",
        ),
    ],
)
def test_bound_shell_truthiness_covers_python_execution_surfaces(
    path: str, prefix: str, expected_line: int
) -> None:
    bound_content = (
        prefix + "import subprocess\nenabled = True\nsubprocess.run(command, shell=enabled)\n"
    )
    direct = _tm1(
        prefix + "import subprocess\nsubprocess.run(command, shell=True)\n",
        path,
    )
    result = _run(bound_content, path)
    bound = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert len(direct) == len(bound) == 1
    assert bound[0].severity == direct[0].severity == "HIGH"
    assert bound[0].confidence == direct[0].confidence == pytest.approx(0.9)
    assert bound[0].fingerprint() == direct[0].fingerprint()
    assert bound[0].start_line == expected_line
    classification = classify_python_source(path, bound_content)
    expected_outcome = (
        LedgerOutcome.PARTIAL
        if classification is PythonSourceClassification.AMBIGUOUS
        else LedgerOutcome.COMPLETED
    )
    assert result["inspection_ledger"][0]["outcome"] is expected_outcome
    if expected_outcome is LedgerOutcome.PARTIAL:
        assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.PYTHON_SOURCE_AMBIGUOUS


@pytest.mark.parametrize(
    "prefix",
    [
        pytest.param(
            "#!/usr/bin/python3 -t\n",
            id="direct",
        ),
        pytest.param(
            "#!/usr/bin/env -S python3 -tt\n",
            id="env-split",
        ),
        pytest.param(
            "#!/usr/bin/env -S python3 -tEB\n",
            id="clustered",
        ),
    ],
)
def test_legacy_tab_compatibility_option_tm1_ledger_is_complete(prefix: str) -> None:
    content = prefix + "enabled = True\nsubprocess.run(command, shell=enabled)\n"

    result = _run(content, "runner")
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert len(findings) == 1
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert result["inspection_ledger"][0].get("reason_code") is None


@pytest.mark.parametrize(
    ("path", "prefix"),
    [
        pytest.param("-h", "#!/usr/bin/python3\n", id="root-dash-basename"),
        pytest.param(
            "tools/-V",
            "#!/usr/bin/env -S python3\n",
            id="subdirectory-dash-basename",
        ),
        pytest.param(
            "tools/-options/runner",
            "#!/usr/bin/python3 -B\n",
            id="dash-intermediate-component",
        ),
        pytest.param(
            "bundle.zip!/-nested/runner",
            "#!/usr/bin/env -S python3\n",
            id="nested-dash-component",
        ),
    ],
)
def test_implicit_dash_source_path_tm1_ledger_is_partial(path: str, prefix: str) -> None:
    content = prefix + "enabled = True\nsubprocess.run(command, shell=enabled)\n"

    result = _run(content, path)
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert len(findings) == 1
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.PYTHON_SOURCE_AMBIGUOUS


def test_python_option_terminator_dash_source_tm1_ledger_is_complete() -> None:
    content = (
        "#!/usr/bin/env -S python3 --\nenabled = True\nsubprocess.run(command, shell=enabled)\n"
    )

    result = _run(content, "tools/-h")
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert len(findings) == 1
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert result["inspection_ledger"][0].get("reason_code") is None


@pytest.mark.parametrize(
    ("path", "prefix"),
    [
        pytest.param("runner", "#!/usr/bin/env node\n", id="env-node"),
        pytest.param(
            "runner",
            "#!/usr/bin/env X=/tmp/python3\n",
            id="plain-env-assignment-not-utility",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -P/tmp/python3\n",
            id="plain-env-option-operand-not-utility",
        ),
        pytest.param("runner", "#!/usr/bin/env node python3\n", id="deceptive-env"),
        pytest.param("runner", "#!/usr/bin/env -S node python3\n", id="deceptive-env-s"),
        pytest.param(
            "runner",
            "#!/usr/bin/env -L default python3\n",
            id="darwin-rejects-freebsd-login-class-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -L python3\n",
            id="env-split-freebsd-login-class-missing-utility",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -iL python3\n",
            id="env-split-freebsd-clustered-ignore-login-class-operand",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -vL python3\n",
            id="env-split-freebsd-clustered-verbose-login-class-operand",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -iLSpython3\n",
            id="env-freebsd-login-class-operand-is-not-outer-split",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -U name python3\n",
            id="darwin-rejects-freebsd-unset-alt-option",
        ),
        pytest.param("runner", "#!/usr/bin/env python3 -h\n", id="darwin-python-help"),
        pytest.param("runner", "#!/usr/bin/env python3 -?\n", id="darwin-python-help-alias"),
        pytest.param("runner", "#!/usr/bin/env python3 -VV\n", id="darwin-python-version"),
        pytest.param(
            "runner",
            "#!/usr/bin/env python3 --check-hash-based-pycs=default\n",
            id="darwin-python-invalid-hash-option-equals-form",
        ),
        pytest.param("runner", "#!/usr/bin/env -Snode python3\n", id="deceptive-attached-s"),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S PYTHONSAFEPATH=1 -i python3\n",
            id="env-option-after-assignment",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S - node python3\n",
            id="lone-dash-still-selects-node",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S --zunknown python3\n",
            id="unknown-long-and-freebsd-cluster-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -a alternate -P/usr/bin python3\n",
            id="mixed-gnu-freebsd-short-options",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S --debug -P/usr/bin python3\n",
            id="mixed-gnu-long-freebsd-short-options",
        ),
        pytest.param(
            "runner",
            '#!/usr/bin/env -S -S "-a alternate -P/usr/bin python3"\n',
            id="nested-mixed-gnu-freebsd-options",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S FOO=bar\\ baz -a alternate python3\n",
            id="mixed-freebsd-lexer-gnu-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env --S-a alternate python3\n",
            id="mixed-freebsd-outer-gnu-inner-option",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S pyth\on3" "\n",
            id="invalid-env-escape",
        ),
        pytest.param(
            "runner",
            r'#!/usr/bin/env -S "python3\c"' "\n",
            id="env-string-terminator-in-double-quotes",
        ),
        pytest.param(
            "runner",
            r'#!/usr/bin/env -S "python3\_-I"' "\n",
            id="env-quoted-escaped-separator",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -xS python3\n",
            id="unknown-outer-env-option",
        ),
        pytest.param("runner", "#!/usr/bin/env -S -P\n", id="missing-env-option-operand"),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S 'python3 -I'\n",
            id="quoted-interpreter-with-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S 'python3\n",
            id="unterminated-env-quote",
        ),
        pytest.param("runner", "#!/bin/sh python3\n", id="shell-with-python-argument"),
        pytest.param(
            "runner",
            "#!/usr/bin/node\0/usr/bin/python3\n",
            id="nul-terminated-node-before-python",
        ),
        pytest.param("typing.pyi", "", id="pyi-without-shebang"),
    ],
)
def test_bound_shell_truthiness_rejects_non_python_execution_surfaces(
    path: str, prefix: str
) -> None:
    assert not _tm1(
        prefix + "import subprocess\nenabled = True\nsubprocess.run(command, shell=enabled)\n",
        path,
    )


def test_non_python_shebang_keeps_direct_lexical_tm1_only() -> None:
    prefix = "#!/usr/bin/env node python3\n"

    assert (
        len(_tm1(prefix + "import subprocess\nsubprocess.run(command, shell=True)\n", "runner"))
        == 1
    )
    assert not _tm1(
        prefix + "import subprocess\nenabled = True\nsubprocess.run(command, shell=enabled)\n",
        "runner",
    )


def test_python_shebang_file_type_reaches_later_static_windows() -> None:
    content = (
        "#!/usr/bin/env python3\n"
        + "#"
        + "x" * (tm_module.static_runner.SECURITY_VIEW_WINDOW_CHARS + 1)
        + "\nsubprocess.run(command, shell=True)\n"
    )

    findings = _tm1(content, "runner")

    assert len(findings) == 1
    assert findings[0].confidence == pytest.approx(0.9)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("1", id="integer"),
        pytest.param("-1", id="negative-integer"),
        pytest.param("+True", id="positive-boolean"),
        pytest.param("-True", id="negative-boolean"),
        pytest.param("(0,)", id="tuple"),
        pytest.param("not False", id="negation"),
    ],
)
def test_simple_immutable_truthy_values_are_tracked(value: str) -> None:
    findings = _tm1(
        f"import subprocess\nenabled = {value}\nsubprocess.run(command, shell=enabled)\n"
    )

    assert len(findings) == 1


def test_simple_immutable_alias_chain_is_tracked() -> None:
    findings = _tm1(
        "import subprocess\n"
        "first = 'enabled'\n"
        "second = first\n"
        "third = second\n"
        "subprocess.check_output(command, shell=third)\n"
    )

    assert len(findings) == 1
    assert findings[0].start_line == 5


@pytest.mark.parametrize("value", ["(unknown,)", "(unknown, 0)"])
def test_nonempty_passive_tuple_truth_does_not_depend_on_element_truth(value: str) -> None:
    findings = _tm1(f"enabled = {value}\nsubprocess.run(command, shell=enabled)\n")

    assert len(findings) == 1


def test_deep_supported_unary_chain_is_not_silently_skipped() -> None:
    findings = _tm1(f"enabled = {'not ' * 65}False\nsubprocess.run(command, shell=enabled)\n")

    assert len(findings) == 1


def test_bare_popen_direct_expression_is_tracked() -> None:
    findings = _tm1("enabled = True\nPopen(command, shell=enabled)\n")

    assert len(findings) == 1


def test_explicit_popen_definition_rejects_bound_calls() -> None:
    assert not _tm1(
        "def Popen(*args, **kwargs):\n"
        "    return None\n"
        "enabled = True\n"
        "Popen('one', shell=enabled)\n"
        "Popen('two', shell=enabled)\n"
    )


def test_explicit_popen_import_reestablishes_bare_receiver() -> None:
    findings = _tm1(
        "Popen = None\n"
        "from subprocess import Popen\n"
        "enabled = True\n"
        "Popen(command, shell=enabled)\n"
    )

    assert len(findings) == 1


def test_relative_subprocess_import_does_not_establish_bare_popen() -> None:
    assert not _tm1(
        "from .subprocess import Popen\nenabled = True\nPopen(command, shell=enabled)\n"
    )


@pytest.mark.parametrize(
    "rebind",
    [
        pytest.param("enabled = False", id="false-assignment"),
        pytest.param("enabled = dynamic", id="unknown-assignment"),
        pytest.param("import pathlib as enabled", id="import"),
        pytest.param("from settings import enabled", id="from-import"),
    ],
)
def test_rebinding_invalidates_truthy_fact(rebind: str) -> None:
    assert not _tm1(
        f"import subprocess\nenabled = True\n{rebind}\nsubprocess.run(command, shell=enabled)\n"
    )


def test_import_side_effect_boundary_clears_all_facts() -> None:
    assert not _tm1(
        "import subprocess\nenabled = True\nimport attacker\n"
        "subprocess.run(command, shell=enabled)\n"
    )


def test_wildcard_import_clears_all_facts() -> None:
    assert not _tm1(
        "enabled = True\nfrom settings import *\nsubprocess.run(command, shell=enabled)\n"
    )


def test_function_local_binding_is_tracked_without_erasing_outer_fact() -> None:
    findings = _tm1(
        "import subprocess\n"
        "outer = True\n"
        "def execute(command):\n"
        "    enabled = 'True'\n"
        "    subprocess.run('true', shell=enabled)\n"
        "subprocess.run('true', shell=outer)\n"
        "execute('true')\n"
    )

    assert [finding.start_line for finding in findings] == [5, 6]


@pytest.mark.parametrize(
    "definition",
    [
        pytest.param("def helper():\n    pass", id="plain"),
        pytest.param("def helper(value=1):\n    pass", id="passive-default"),
        pytest.param("async def helper():\n    pass", id="async"),
    ],
)
def test_passive_function_definition_preserves_unrelated_truth_fact(definition: str) -> None:
    findings = _tm1(f"enabled = True\n{definition}\nsubprocess.run(command, shell=enabled)\n")

    assert len(findings) == 1


@pytest.mark.parametrize(
    "shadow",
    [
        pytest.param("subprocess = Proxy()", id="assignment"),
        pytest.param("import other as subprocess", id="import-alias"),
        pytest.param("for subprocess in values:\n    pass", id="compound-binder"),
        pytest.param("class subprocess:\n    pass", id="class-binder"),
        pytest.param("subprocess, other = pair", id="unpacking"),
        pytest.param("subprocess.run = Proxy()", id="attribute-assignment"),
        pytest.param(
            "subprocess.run: object = Proxy()",
            id="annotated-attribute-assignment",
        ),
    ],
)
def test_explicit_subprocess_shadow_rejects_later_bound_call(shadow: str) -> None:
    assert not _tm1(f"{shadow}\nenabled = True\nsubprocess.run(command, shell=enabled)\n")


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param("item = (subprocess := Proxy())", id="assignment-value"),
        pytest.param(
            "item: (subprocess := Proxy()) = 1",
            id="assignment-annotation",
        ),
        pytest.param("item: int = (subprocess := Proxy())", id="annotated-value"),
        pytest.param("consume(subprocess := Proxy())", id="call-argument"),
        pytest.param(
            "subprocess.run((subprocess := Proxy()), shell=False)",
            id="rejected-direct-call-argument",
        ),
    ],
)
def test_explicit_nested_subprocess_rebinding_rejects_later_bound_call(mutation: str) -> None:
    assert not _tm1(
        f"import subprocess\n{mutation}\nenabled = True\nsubprocess.run(command, shell=enabled)\n"
    )


@pytest.mark.parametrize(
    "compound",
    [
        pytest.param("if True:\n    from attacker import *", id="conditional"),
        pytest.param(
            "try:\n    from attacker import *\nexcept ImportError:\n    pass",
            id="try",
        ),
    ],
)
def test_compound_wildcard_import_invalidates_direct_receivers(compound: str) -> None:
    assert not _tm1(f"{compound}\nenabled = True\nsubprocess.run(command, shell=enabled)\n")


def test_subprocess_parameter_rejects_bound_attribute_lookup() -> None:
    assert not _tm1(
        "def execute(subprocess, command):\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
    )


def test_outer_subprocess_shadow_is_inherited_by_nested_function() -> None:
    assert not _tm1(
        "subprocess = Proxy()\n"
        "def execute(command):\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
    )


@pytest.mark.parametrize(
    "later_binding",
    [
        pytest.param("subprocess = Proxy()", id="assignment"),
        pytest.param("import other as subprocess", id="import-alias"),
        pytest.param("subprocess.run = Proxy()", id="attribute-mutation"),
    ],
)
def test_later_global_receiver_mutation_suppresses_function_body(
    later_binding: str,
) -> None:
    assert not _tm1(
        "def execute(command):\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        f"{later_binding}\n"
    )


@pytest.mark.parametrize(
    "later_binding",
    [
        pytest.param("subprocess = Proxy()", id="assignment"),
        pytest.param("import other as subprocess", id="import-alias"),
        pytest.param("subprocess.run = Proxy()", id="attribute-mutation"),
    ],
)
def test_function_call_before_later_global_receiver_mutation_is_reported(
    later_binding: str,
) -> None:
    findings = _tm1(
        "import subprocess\n"
        "def execute(command):\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        "execute(command)\n"
        f"{later_binding}\n"
    )

    assert len(findings) == 1
    assert findings[0].start_line == 4


@pytest.mark.parametrize(
    "call_statement",
    [
        pytest.param("result = execute(command)", id="assignment"),
        pytest.param("result: object = execute(command)", id="annotated-assignment"),
    ],
)
def test_function_call_in_assignment_before_receiver_mutation_is_reported(
    call_statement: str,
) -> None:
    findings = _tm1(
        "import subprocess\n"
        "def execute(command):\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        f"{call_statement}\n"
        "subprocess = Proxy()\n"
    )

    assert len(findings) == 1
    assert findings[0].start_line == 4


def test_function_alias_called_before_receiver_mutation_is_reported() -> None:
    findings = _tm1(
        "import subprocess\n"
        "def execute(command):\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        "alias = execute\n"
        "alias(command)\n"
        "subprocess = Proxy()\n"
    )

    assert len(findings) == 1
    assert findings[0].start_line == 4


@pytest.mark.parametrize(
    "unknown_call",
    [
        pytest.param("unknown()", id="expression"),
        pytest.param("result = unknown()", id="assignment"),
        pytest.param("result: object = unknown()", id="annotated-assignment"),
    ],
)
def test_unknown_call_before_local_function_invalidates_receiver_generation(
    unknown_call: str,
) -> None:
    assert not _tm1(
        "import subprocess\n"
        "def victim(command):\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        f"{unknown_call}\n"
        "victim(command)\n"
        "subprocess = Proxy()\n"
    )


def test_called_local_mutator_invalidates_receiver_before_later_function_call() -> None:
    assert not _tm1(
        "import subprocess\n"
        "def victim(command):\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        "def mutate():\n"
        "    global subprocess\n"
        "    subprocess = Proxy()\n"
        "mutate()\n"
        "victim(command)\n"
        "subprocess = Proxy()\n"
    )


def test_called_local_mutator_invalidates_later_callable_identity() -> None:
    assert not _tm1(
        "import subprocess\n"
        "def victim(command):\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        "def mutate():\n"
        "    global victim\n"
        "    victim = lambda command: None\n"
        "mutate()\n"
        "import subprocess\n"
        "victim(command)\n"
        "subprocess = Proxy()\n"
    )


@pytest.mark.parametrize(
    "import_statement",
    [
        pytest.param("import attacker", id="import"),
        pytest.param("from attacker import hook", id="import-from"),
    ],
)
def test_import_hook_invalidates_later_callable_identity(import_statement: str) -> None:
    assert not _tm1(
        "import subprocess\n"
        "def victim(command):\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        f"{import_statement}\n"
        "victim(command)\n"
        "subprocess = None\n"
    )


@pytest.mark.parametrize(
    "import_statement",
    [
        pytest.param("import attacker", id="import"),
        pytest.param("from attacker import hook", id="import-from"),
        pytest.param("import subprocess, attacker", id="later-alias"),
    ],
)
def test_import_hook_invalidates_direct_receiver_trust(import_statement: str) -> None:
    assert not _tm1(
        "import subprocess\n"
        f"{import_statement}\n"
        "enabled = True\n"
        "subprocess.run('true', shell=enabled)\n"
    )


def test_arbitrary_import_alias_prevents_later_exact_reestablishment() -> None:
    assert not _tm1(
        "import attacker, subprocess\nenabled = True\nsubprocess.run('true', shell=enabled)\n"
    )


def test_import_from_hook_clears_other_receiver_before_reestablishing_popen() -> None:
    findings = _tm1(
        "import subprocess\n"
        "from subprocess import Popen\n"
        "enabled = True\n"
        "Popen('true', shell=enabled)\n"
        "subprocess.run('true', shell=enabled)\n"
    )

    assert len(findings) == 1
    assert (findings[0].matched_text or "").startswith("Popen(")


@pytest.mark.parametrize(
    ("imports", "call"),
    [
        pytest.param(
            "import subprocess\nimport subprocess",
            "subprocess.run('true', shell=enabled)",
            id="subprocess",
        ),
        pytest.param(
            "from subprocess import Popen\nfrom subprocess import Popen",
            "Popen('true', shell=enabled)",
            id="popen",
        ),
    ],
)
def test_repeated_exact_import_keeps_finalizer_safe_receiver(
    imports: str,
    call: str,
) -> None:
    findings = _tm1(f"{imports}\nenabled = True\n{call}\n")

    assert len(findings) == 1


@pytest.mark.parametrize(
    "import_statement",
    [
        pytest.param(
            "from subprocess import PIPE as x, Popen",
            id="unsafe-release-before-popen",
        ),
        pytest.param(
            "from subprocess import Popen, PIPE as x",
            id="unsafe-release-after-popen",
        ),
    ],
)
def test_mixed_import_from_never_reestablishes_receiver(import_statement: str) -> None:
    assert not _tm1(
        "class Evil:\n"
        "    def __del__(self):\n"
        "        global Popen\n"
        "        Popen = None\n"
        "x = Evil()\n"
        f"{import_statement}\n"
        "enabled = True\n"
        "Popen('true', shell=enabled)\n"
    )


@pytest.mark.parametrize(
    "arbitrary_effect",
    [
        pytest.param("unknown()", id="call"),
        pytest.param("import attacker", id="import"),
    ],
)
def test_arbitrary_effect_prevents_later_exact_import_reestablishment(
    arbitrary_effect: str,
) -> None:
    assert not _tm1(
        f"{arbitrary_effect}\n"
        "import subprocess\n"
        "enabled = True\n"
        "subprocess.run('true', shell=enabled)\n"
    )


def test_import_hook_poisoned_value_remains_finalizer_unsafe() -> None:
    assert not _tm1(
        "x = None\n"
        "import subprocess\n"
        "import attacker\n"
        "import subprocess\n"
        "x = None\n"
        "enabled = True\n"
        "subprocess.run('true', shell=enabled)\n"
    )


@pytest.mark.parametrize(
    "replacement",
    [
        pytest.param("x = None", id="assignment"),
        pytest.param("del x", id="delete"),
        pytest.param("x = subprocess.run('true')", id="subprocess-result-assignment"),
        pytest.param("def x():\n    pass", id="function-definition"),
        pytest.param("class x:\n    pass", id="class-definition"),
        pytest.param("import math as x", id="import"),
        pytest.param("from math import pi as x", id="import-from"),
    ],
)
def test_releasing_unsafe_prior_binding_invalidates_receiver(replacement: str) -> None:
    assert not _tm1(
        "class Proxy:\n"
        "    pass\n"
        "class Evil:\n"
        "    def __del__(self):\n"
        "        global subprocess\n"
        "        subprocess = Proxy()\n"
        "x = Evil()\n"
        "import subprocess\n"
        f"{replacement}\n"
        "enabled = True\n"
        "subprocess.run('true', shell=enabled)\n"
    )


def test_import_cannot_reestablish_receiver_while_releasing_unsafe_prior_value() -> None:
    assert not _tm1(
        "class Proxy:\n"
        "    pass\n"
        "class Evil:\n"
        "    def __del__(self):\n"
        "        global subprocess\n"
        "        subprocess = Proxy()\n"
        "subprocess = Evil()\n"
        "import subprocess\n"
        "enabled = True\n"
        "subprocess.run('true', shell=enabled)\n"
    )


@pytest.mark.parametrize(
    ("signature", "declaration", "call"),
    [
        pytest.param("x", "", "victim(value)", id="parameter"),
        pytest.param("", "global x\n    ", "victim()", id="global"),
    ],
)
def test_rebinding_unsafe_parameter_or_global_invalidates_receiver(
    signature: str,
    declaration: str,
    call: str,
) -> None:
    assert not _tm1(
        "import subprocess\n"
        f"def victim({signature}):\n"
        f"    {declaration}x = None\n"
        "    enabled = True\n"
        "    subprocess.run('true', shell=enabled)\n"
        f"{call}\n"
        "subprocess = None\n"
    )


@pytest.mark.parametrize(
    "binder",
    [
        pytest.param("(x := Evil())", id="named-expression"),
        pytest.param("for x in values:\n    pass", id="for-target"),
        pytest.param("with manager as x:\n    pass", id="with-target"),
        pytest.param(
            "try:\n    unknown()\nexcept Error as caught:\n    x = caught",
            id="except-body-assignment",
        ),
        pytest.param("match value:\n    case x:\n        pass", id="match-capture"),
    ],
)
def test_compound_or_expression_binding_retains_unsafe_finalizer_state(binder: str) -> None:
    assert not _tm1(
        f"{binder}\n"
        "import subprocess\n"
        "x = None\n"
        "enabled = True\n"
        "subprocess.run('true', shell=enabled)\n"
    )


def test_class_global_store_releasing_unsafe_value_invalidates_receiver() -> None:
    assert not _tm1(
        "x = Evil()\n"
        "import subprocess\n"
        "class Container:\n"
        "    global x\n"
        "    x = None\n"
        "enabled = True\n"
        "subprocess.run('true', shell=enabled)\n"
    )


@pytest.mark.parametrize(
    ("setup", "trailing"),
    [
        pytest.param(
            "def execute(command):\n"
            "    enabled = True\n"
            "    subprocess.run(command, shell=enabled)\n"
            "subprocess = None\n"
            "import subprocess\n",
            "",
            id="reimport-after-definition",
        ),
        pytest.param(
            "subprocess = Proxy()\n"
            "def execute(command):\n"
            "    enabled = True\n"
            "    subprocess.run(command, shell=enabled)\n"
            "import subprocess\n",
            "subprocess = Proxy()\n",
            id="definition-before-reimport-and-later-mutation",
        ),
    ],
)
def test_import_reestablishes_receiver_but_not_prior_callable_identity(
    setup: str,
    trailing: str,
) -> None:
    assert not _tm1(f"{setup}execute(command)\n{trailing}")


def test_local_call_defined_after_receiver_reimport_is_trusted() -> None:
    findings = _tm1(
        "subprocess = None\n"
        "import subprocess\n"
        "def execute(command):\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        "execute(command)\n"
        "subprocess = None\n"
    )

    assert len(findings) == 1
    assert "subprocess.run" in (findings[0].matched_text or "")


def test_receiver_reimport_does_not_restore_callable_after_unsafe_assignment() -> None:
    assert not _tm1(
        "def execute(command):\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        "subprocess = Proxy()\n"
        "import subprocess\n"
        "execute(command)\n"
        "subprocess = Proxy()\n"
    )


@pytest.mark.parametrize(
    "unknown_call",
    [
        pytest.param("unknown()", id="expression"),
        pytest.param("result = unknown()", id="assignment"),
        pytest.param("result: object = unknown()", id="annotated-assignment"),
    ],
)
def test_unknown_top_level_call_invalidates_direct_receiver_trust(
    unknown_call: str,
) -> None:
    assert not _tm1(
        "import subprocess\n"
        f"{unknown_call}\n"
        "enabled = True\n"
        "subprocess.run(command, shell=enabled)\n"
    )


@pytest.mark.parametrize(
    "unsafe_statement",
    [
        pytest.param("result = (unknown(), 1)", id="nested-call-assignment"),
        pytest.param("result = [unknown()]", id="nested-list-call-assignment"),
        pytest.param("holder.value = 1", id="attribute-setter"),
        pytest.param("holder[0] = 1", id="subscript-setter"),
        pytest.param(
            "holder[0] = subprocess.run(command, shell=False)",
            id="trusted-call-subscript-setter",
        ),
        pytest.param("counter += 1", id="augmented-assignment"),
        pytest.param("holder.value += 1", id="attribute-augmented-assignment"),
        pytest.param("if unknown():\n    pass", id="compound-call"),
        pytest.param("if condition:\n    unknown()", id="compound-body-call"),
        pytest.param("with unknown():\n    pass", id="context-manager-call"),
        pytest.param("class Container:\n    unknown()", id="class-body-call"),
        pytest.param(
            "class Container(unknown()):\n    pass",
            id="class-base-call",
        ),
        pytest.param("del holder.item", id="descriptor-delete"),
        pytest.param("left + right", id="operator-expression"),
        pytest.param("holder.value", id="attribute-expression"),
    ],
)
def test_unsafe_evaluation_invalidates_direct_receiver_trust(
    unsafe_statement: str,
) -> None:
    assert not _tm1(
        "import subprocess\n"
        f"{unsafe_statement}\n"
        "enabled = True\n"
        "subprocess.run(command, shell=enabled)\n"
    )


@pytest.mark.parametrize(
    "unsafe_statement",
    [
        pytest.param("result = (unknown(), 1)", id="nested-call-assignment"),
        pytest.param("result = [unknown()]", id="nested-list-call-assignment"),
        pytest.param("holder.value = 1", id="attribute-setter"),
        pytest.param("holder[0] = 1", id="subscript-setter"),
        pytest.param(
            "holder[0] = subprocess.run(command, shell=False)",
            id="trusted-call-subscript-setter",
        ),
        pytest.param("counter += 1", id="augmented-assignment"),
        pytest.param("holder.value += 1", id="attribute-augmented-assignment"),
        pytest.param("if unknown():\n    pass", id="compound-call"),
        pytest.param("if condition:\n    unknown()", id="compound-body-call"),
        pytest.param("with unknown():\n    pass", id="context-manager-call"),
        pytest.param("class Container:\n    unknown()", id="class-body-call"),
        pytest.param(
            "class Container(unknown()):\n    pass",
            id="class-base-call",
        ),
        pytest.param("del holder.item", id="descriptor-delete"),
        pytest.param("left + right", id="operator-expression"),
        pytest.param("holder.value", id="attribute-expression"),
    ],
)
def test_unsafe_evaluation_invalidates_receiver_generation(
    unsafe_statement: str,
) -> None:
    assert not _tm1(
        "import subprocess\n"
        "def victim(command):\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        f"{unsafe_statement}\n"
        "victim(command)\n"
        "subprocess = Proxy()\n"
    )


@pytest.mark.parametrize(
    "trusted_call",
    [
        pytest.param("subprocess.run('true', shell=False)", id="expression"),
        pytest.param("subprocess.run(['true'], shell=False)", id="literal-container"),
        pytest.param("subprocess.run('tr' + 'ue', shell=False)", id="constant-operator"),
        pytest.param(
            "result = subprocess.run('true', shell=False)",
            id="assignment",
        ),
    ],
)
def test_passive_trusted_subprocess_call_preserves_direct_receiver(
    trusted_call: str,
) -> None:
    findings = _tm1(
        "import subprocess\n"
        f"{trusted_call}\n"
        "enabled = True\n"
        "subprocess.run(command, shell=enabled)\n"
    )

    assert len(findings) == 1
    assert findings[0].start_line == 4


@pytest.mark.parametrize(
    ("content", "expected_lines"),
    [
        pytest.param(
            "enabled = True\n"
            "subprocess.run('one', shell=enabled)\n"
            "subprocess.run('two', shell=enabled)\n",
            [2, 3],
            id="consecutive-bound-calls",
        ),
        pytest.param(
            "disabled = False\n"
            "subprocess.run('one', shell=disabled)\n"
            "enabled = True\n"
            "subprocess.run('two', shell=enabled)\n",
            [4],
            id="known-false-shell-then-bound-call",
        ),
        pytest.param(
            "enabled = True\n"
            "subprocess.run('one', check=enabled, shell=False)\n"
            "subprocess.run('two', shell=enabled)\n",
            [3],
            id="safe-name-in-other-keyword",
        ),
        pytest.param(
            "command = 'one'\n"
            "subprocess.run(command, shell=False)\n"
            "enabled = True\n"
            "subprocess.run('two', shell=enabled)\n",
            [4],
            id="safe-command-name",
        ),
        pytest.param(
            "enabled = True\n"
            "alias = enabled\n"
            "subprocess.run('one', shell=alias)\n"
            "subprocess.run('two', shell=enabled)\n",
            [3, 4],
            id="safe-shell-alias",
        ),
        pytest.param(
            "enabled = True\nsubprocess.run('one', shell=enabled)\nPopen('two', shell=enabled)\n",
            [2, 3],
            id="cross-receiver-bound-calls",
        ),
        pytest.param(
            "enabled = True\n"
            "subprocess.run('one', shell=enabled)\n"
            "subprocess.run(command, shell=enabled)\n"
            "subprocess.run('three', shell=enabled)\n",
            [2, 3],
            id="bound-then-dynamic-command-barrier",
        ),
    ],
)
def test_protocol_safe_name_provenance_preserves_receiver_trust(
    content: str,
    expected_lines: list[int],
) -> None:
    assert [finding.start_line for finding in _tm1(content)] == expected_lines


def test_unknown_name_inside_safe_container_does_not_preserve_receiver_trust() -> None:
    assert not _tm1(
        "item = unknown\n"
        "command = (item,)\n"
        "subprocess.run(command, shell=False)\n"
        "enabled = True\n"
        "subprocess.run('later', shell=enabled)\n"
    )


def test_safe_bound_call_preserves_receiver_generation_for_local_call() -> None:
    findings = _tm1(
        "import subprocess\n"
        "def victim():\n"
        "    enabled = True\n"
        "    subprocess.run('victim', shell=enabled)\n"
        "enabled = True\n"
        "subprocess.run('safe', shell=enabled)\n"
        "victim()\n"
        "subprocess = None\n"
    )

    assert [finding.start_line for finding in findings] == [4, 6]


@pytest.mark.parametrize(
    ("victim_call", "safe_call", "later_binding"),
    [
        pytest.param(
            "Popen(command, shell=enabled)",
            "subprocess.run('true', shell=False)",
            "Popen = Proxy()",
            id="subprocess-call-preserves-popen",
        ),
        pytest.param(
            "subprocess.run(command, shell=enabled)",
            "Popen('true', shell=False)",
            "subprocess = Proxy()",
            id="popen-call-preserves-subprocess",
        ),
    ],
)
def test_passive_direct_call_preserves_other_receiver_generation(
    victim_call: str,
    safe_call: str,
    later_binding: str,
) -> None:
    findings = _tm1(
        "def victim(command):\n"
        "    enabled = True\n"
        f"    {victim_call}\n"
        f"{safe_call}\n"
        "victim(command)\n"
        f"{later_binding}\n"
    )

    assert len(findings) == 1
    assert findings[0].start_line == 3


@pytest.mark.parametrize(
    "trusted_call",
    [
        pytest.param("subprocess.run(path, shell=False)", id="positional"),
        pytest.param("result = subprocess.run(path, shell=False)", id="assignment"),
        pytest.param(
            "subprocess.run('true', cwd=path, shell=False)",
            id="keyword",
        ),
    ],
)
def test_pathlike_protocol_hook_invalidates_receiver_after_trusted_call(
    trusted_call: str,
) -> None:
    assert not _tm1(
        "class EvilPath:\n"
        "    def __fspath__(self):\n"
        "        global subprocess\n"
        "        subprocess = Proxy()\n"
        "path = EvilPath()\n"
        "import subprocess\n"
        f"{trusted_call}\n"
        "enabled = True\n"
        "subprocess.run(command, shell=enabled)\n"
    )


@pytest.mark.parametrize(
    "replacement",
    [
        pytest.param("execute = replacement", id="direct-name"),
        pytest.param("alias = execute\nalias = replacement", id="alias"),
    ],
)
def test_rebound_function_name_before_call_does_not_mark_old_body(
    replacement: str,
) -> None:
    called_name = "alias" if replacement.startswith("alias") else "execute"
    assert not _tm1(
        "import subprocess\n"
        "def execute(command):\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        f"{replacement}\n"
        f"{called_name}(command)\n"
        "subprocess = Proxy()\n"
    )


def test_function_called_only_after_global_receiver_mutation_is_suppressed() -> None:
    assert not _tm1(
        "import subprocess\n"
        "def execute(command):\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        "subprocess = Proxy()\n"
        "execute(command)\n"
    )


def test_reestablished_receiver_before_function_body_is_trusted() -> None:
    findings = _tm1(
        "subprocess = None\n"
        "import subprocess\n"
        "def execute(command):\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
    )

    assert len(findings) == 1


def test_function_default_side_effect_invalidates_subprocess_receiver() -> None:
    assert not _tm1(
        "def helper(value=mutate_subprocess()):\n"
        "    pass\n"
        "enabled = True\n"
        "subprocess.run(command, shell=enabled)\n"
    )


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param("argument: (subprocess := object())", id="argument"),
        pytest.param("*args: (subprocess := object())", id="vararg"),
        pytest.param("**kwargs: (subprocess := object())", id="kwarg"),
    ],
)
def test_later_nested_function_annotation_binds_outer_subprocess(annotation: str) -> None:
    assert not _tm1(
        "import subprocess\n"
        "def outer():\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        f"    def inner({annotation}):\n"
        "        pass\n"
    )


def test_later_nested_function_return_annotation_binds_outer_subprocess() -> None:
    assert not _tm1(
        "import subprocess\n"
        "def outer():\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        "    def inner() -> (subprocess := object()):\n"
        "        pass\n"
    )


def test_explicit_subprocess_import_reestablishes_direct_receiver() -> None:
    findings = _tm1(
        "subprocess = None\n"
        "import subprocess\n"
        "enabled = True\n"
        "subprocess.run(command, shell=enabled)\n"
    )

    assert len(findings) == 1


@pytest.mark.parametrize(
    "class_body",
    [
        pytest.param(
            "    global subprocess\n    subprocess = Proxy()",
            id="global-rebinding",
        ),
        pytest.param("    subprocess.run = Proxy()", id="attribute-mutation"),
    ],
)
def test_explicit_class_body_mutation_invalidates_subprocess_receiver(class_body: str) -> None:
    assert not _tm1(
        "import subprocess\n"
        "class Container:\n"
        f"{class_body}\n"
        "enabled = True\n"
        "subprocess.run(command, shell=enabled)\n"
    )


@pytest.mark.parametrize(
    "nested_body",
    [
        pytest.param(
            "        global subprocess\n        subprocess = Proxy()",
            id="global-rebinding",
        ),
        pytest.param("        subprocess.run = Proxy()", id="attribute-mutation"),
    ],
)
def test_nested_class_body_mutation_invalidates_subprocess_receiver(nested_body: str) -> None:
    assert not _tm1(
        "import subprocess\n"
        "class Outer:\n"
        "    class Inner:\n"
        f"{nested_body}\n"
        "enabled = True\n"
        "subprocess.run(command, shell=enabled)\n"
    )


def test_nested_class_initializer_call_invalidates_subprocess_receiver() -> None:
    assert not _tm1(
        "import subprocess\n"
        "class Outer:\n"
        "    class Inner:\n"
        "        subprocess = Proxy()\n"
        "enabled = True\n"
        "subprocess.run(command, shell=enabled)\n"
    )


@pytest.mark.parametrize(
    "class_body",
    [
        "    subprocess = subprocess\n    subprocess.run = Proxy()",
        "    import subprocess\n    subprocess.run = Proxy()",
    ],
)
def test_class_local_genuine_receiver_mutation_invalidates_outer_receiver(
    class_body: str,
) -> None:
    assert not _tm1(
        "import subprocess\n"
        "class Container:\n"
        f"{class_body}\n"
        "enabled = True\n"
        "subprocess.run(command, shell=enabled)\n"
    )


def test_class_local_proxy_calls_invalidate_outer_receiver() -> None:
    assert not _tm1(
        "import subprocess\n"
        "class Container:\n"
        "    subprocess = Proxy()\n"
        "    subprocess.run = Proxy()\n"
        "enabled = True\n"
        "subprocess.run(command, shell=enabled)\n"
    )


def test_passive_class_body_preserves_subprocess_receiver() -> None:
    findings = _tm1(
        "import subprocess\n"
        "class Container:\n"
        "    value = 1\n"
        "enabled = True\n"
        "subprocess.run(command, shell=enabled)\n"
    )

    assert len(findings) == 1


def test_class_descriptor_name_assignment_invalidates_subprocess_receiver() -> None:
    assert not _tm1(
        "class Descriptor:\n"
        "    def __set_name__(self, owner, name):\n"
        "        global subprocess\n"
        "        subprocess = Proxy()\n"
        "descriptor = Descriptor()\n"
        "import subprocess\n"
        "class Container:\n"
        "    value = descriptor\n"
        "enabled = True\n"
        "subprocess.run(command, shell=enabled)\n"
    )


def test_deep_class_expression_does_not_fail_later_module_scan() -> None:
    deep_expression = "+".join(["1"] * 500)
    result = _run(
        "class Container:\n"
        f"    value = {deep_expression}\n"
        "enabled = True\n"
        "subprocess.run(command, shell=enabled)\n"
    )
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert len(findings) == 1


@pytest.mark.parametrize(
    "imports",
    [
        pytest.param("import subprocess", id="single-exact-import"),
        pytest.param("import subprocess\nimport subprocess", id="repeated-exact-import"),
        pytest.param("import subprocess\nsubprocess = subprocess", id="self-assignment"),
    ],
)
def test_exact_subprocess_import_last_preserves_receiver(imports: str) -> None:
    findings = _tm1(f"{imports}\nenabled = True\nsubprocess.run(command, shell=enabled)\n")

    assert len(findings) == 1


@pytest.mark.parametrize(
    "later_binding",
    [
        pytest.param("subprocess = Proxy()", id="assignment"),
        pytest.param("import subprocess", id="import"),
        pytest.param("for subprocess in values:\n        pass", id="loop-target"),
        pytest.param("del subprocess", id="deletion"),
        pytest.param(
            "match value:\n        case _ as subprocess:\n            pass",
            id="match-capture",
        ),
    ],
)
def test_later_function_local_subprocess_binding_rejects_earlier_lookup(
    later_binding: str,
) -> None:
    assert not _tm1(
        "def execute(command):\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        f"    {later_binding}\n"
    )


def test_later_function_local_popen_binding_rejects_earlier_lookup() -> None:
    assert not _tm1(
        "def execute(command):\n"
        "    enabled = True\n"
        "    Popen(command, shell=enabled)\n"
        "    Popen = Proxy()\n"
    )


def test_valid_deeply_nested_function_body_is_not_silently_skipped() -> None:
    levels = 65
    lines = ["    " * depth + f"def level_{depth}():" for depth in range(levels)]
    lines.extend(
        [
            "    " * levels + "enabled = True",
            "    " * levels + "subprocess.run(command, shell=enabled)",
        ]
    )

    findings = _tm1("\n".join(lines) + "\n")

    assert len(findings) == 1


@pytest.mark.parametrize(
    "compound",
    [
        pytest.param("if condition:\n    pass", id="if"),
        pytest.param("for item in values:\n    pass", id="for"),
        pytest.param("with provider():\n    pass", id="with"),
        pytest.param("try:\n    pass\nexcept Exception:\n    pass", id="try"),
        pytest.param("class Local:\n    pass", id="class"),
    ],
)
def test_compound_statement_conservatively_clears_outer_facts(compound: str) -> None:
    assert not _tm1(f"enabled = True\n{compound}\nsubprocess.run(command, shell=enabled)\n")


def test_calls_inside_compound_statements_are_out_of_scope() -> None:
    assert not _tm1(
        "if condition:\n    enabled = True\n    subprocess.run(command, shell=enabled)\n"
    )


def test_unsupported_assignment_clears_other_facts() -> None:
    assert not _tm1("enabled = True\nresult = factory()\nsubprocess.run(command, shell=enabled)\n")


def test_deep_unsupported_expression_does_not_fail_or_shadow_receiver() -> None:
    deep_expression = "+".join(["1"] * 500)
    result = _run(
        f"result = {deep_expression}\nenabled = True\nsubprocess.run(command, shell=enabled)\n"
    )
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert len(findings) == 1


def test_named_expression_in_call_is_rejected_and_clears_facts() -> None:
    assert not _tm1(
        "enabled = True\n"
        "subprocess.run(command, shell=enabled, marker=(enabled := False))\n"
        "subprocess.run(command, shell=enabled)\n"
    )


@pytest.mark.parametrize(
    "argument",
    [
        pytest.param("disable()", id="call"),
        pytest.param("mutator.command", id="attribute"),
        pytest.param("holder[0]", id="subscript"),
        pytest.param("left + right", id="operator"),
        pytest.param("f'{value}'", id="formatted-string"),
        pytest.param("[item for item in items]", id="comprehension"),
        pytest.param("*commands", id="starred-expansion"),
    ],
)
def test_side_effect_capable_argument_before_shell_clears_facts(argument: str) -> None:
    assert not _tm1(f"enabled = True\nsubprocess.run({argument}, shell=enabled)\n")


@pytest.mark.parametrize(
    "keyword",
    [
        pytest.param("env={}", id="empty-dict"),
        pytest.param("env={'A': 'B'}", id="constant-dict"),
        pytest.param("timeout=-1", id="unary-number"),
        pytest.param("markers={'safe'}", id="constant-set"),
        pytest.param("marker=not None", id="not-constant"),
        pytest.param("marker=not ()", id="not-empty-tuple"),
        pytest.param("timeout=--1", id="nested-unary-number"),
        pytest.param("timeout=~-1", id="mixed-unary-number"),
    ],
)
def test_passive_container_and_unary_arguments_preserve_bound_parity(keyword: str) -> None:
    direct = _tm1(f"subprocess.run(command, shell=True, {keyword})\n")
    bound = _tm1(f"enabled = True\nsubprocess.run(command, shell=enabled, {keyword})\n")

    assert len(direct) == len(bound) == 1


def test_dict_key_with_user_hash_is_not_treated_as_passive() -> None:
    assert not _tm1("enabled = True\nsubprocess.run(command, shell=enabled, env={key: 'value'})\n")


@pytest.mark.parametrize("value", ["~1.0", "~1j", "~(+1.0)", "+(~1.0)"])
def test_invalid_invert_expression_is_not_treated_as_passive(value: str) -> None:
    assert not _tm1(f"enabled = True\nsubprocess.run(command, shell=enabled, timeout={value})\n")


def test_side_effect_capable_argument_on_assignment_rhs_is_not_reported() -> None:
    assert not _tm1("enabled = True\nresult = subprocess.run(disable(), shell=enabled)\n")


def test_unsafe_annotation_clears_facts() -> None:
    assert not _tm1(
        "enabled = True\nitem: (enabled := False) = 1\nsubprocess.run(command, shell=enabled)\n"
    )


def test_annotated_assignment_is_outside_side_effect_free_contract() -> None:
    assert not _tm1("enabled: bool = True\nsubprocess.run(command, shell=enabled)\n")


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("result: object", id="name"),
        pytest.param("holder.result: object", id="attribute"),
        pytest.param("holder[0]: object", id="subscript"),
    ],
)
def test_annotated_assignment_rhs_call_is_inspected_before_fact_reset(target: str) -> None:
    findings = _tm1(
        f"import subprocess\nenabled = True\n{target} = subprocess.run(command, shell=enabled)\n"
    )

    assert len(findings) == 1
    assert findings[0].start_line == 3


def test_no_value_annotated_subscript_target_clears_facts() -> None:
    assert not _tm1(
        "enabled = True\nholder[(enabled := False)]: int\nsubprocess.run(command, shell=enabled)\n"
    )


def test_true_prefixed_identifier_is_owned_by_lexical_tm1_only() -> None:
    findings = _tm1(
        "import subprocess\ntrue_value = True\nsubprocess.run(command, shell=true_value)\n"
    )

    assert len(findings) == 1
    assert findings[0].confidence == pytest.approx(0.9)


def test_multibyte_prefix_maps_ast_byte_column_to_lexical_character_offset() -> None:
    findings = _tm1("true_value = True\npi_π = 1; subprocess.run(command, shell=true_value)\n")

    assert len(findings) == 1
    assert findings[0].confidence == pytest.approx(0.9)


@pytest.mark.parametrize("line_break", ["\n", "\r\n", "\r"])
def test_true_prefixed_owner_uses_python_line_boundaries(line_break: str) -> None:
    findings = _tm1(
        line_break.join(
            [
                "true_value = True",
                "subprocess.run(command, shell=true_value)",
                "",
            ]
        )
    )

    assert len(findings) == 1
    assert findings[0].confidence == pytest.approx(0.9)


def test_bound_call_uses_runner_logical_line_coordinates() -> None:
    direct = _tm1("enabled = True\n\fsubprocess.run(command, shell=True)\n")
    bound = _tm1("enabled = True\n\fsubprocess.run(command, shell=enabled)\n")

    assert len(direct) == len(bound) == 1
    assert (direct[0].start_line, bound[0].start_line) == (3, 3)
    assert bound[0].end_line == 3


def test_true_prefixed_identifier_is_ast_owned_when_lexical_shape_cannot_match() -> None:
    findings = _tm1("true_value = True\nsubprocess.run((command), shell=true_value)\n")

    assert len(findings) == 1
    assert "shell=true_value" in findings[0].matched_text


def test_confusable_true_prefixed_identifier_has_one_lexical_owner() -> None:
    confusable_name = "tru\N{CYRILLIC SMALL LETTER IE}_value"
    findings = _tm1(f"{confusable_name} = True\nsubprocess.run(command, shell={confusable_name})\n")

    assert len(findings) == 1


def test_embedded_direct_literal_text_does_not_duplicate_bound_call() -> None:
    findings = _tm1("enabled = True\nsubprocess.run('shell=True', shell=enabled)\n")

    assert len(findings) == 1
    assert "shell=enabled" in (findings[0].matched_text or "")


def test_normalized_embedded_direct_literal_text_does_not_duplicate_bound_call() -> None:
    confusable_true = "Tru\N{CYRILLIC SMALL LETTER IE}"
    findings = _tm1(f"enabled = True\nsubprocess.run('shell={confusable_true}', shell=enabled)\n")

    assert len(findings) == 1


def test_ascii_control_embedded_direct_literal_text_does_not_duplicate_bound_call() -> None:
    findings = _tm1("enabled = True\nsubprocess.run('shell=\x01True', shell=enabled)\n")

    assert len(findings) == 1


def test_nested_string_lexical_match_does_not_suppress_outer_bound_call() -> None:
    findings = _tm1(
        "enabled = True\n"
        "subprocess.run((command), 'subprocess.run(x, shell=True)', shell=enabled)\n"
    )

    assert len(findings) == 2
    assert sum("shell=enabled" in (finding.matched_text or "") for finding in findings) == 1


def test_multiline_nested_string_lexical_match_keeps_outer_call_location() -> None:
    findings = _tm1(
        "enabled = True\n"
        "subprocess.run(\n"
        "    (command),\n"
        "    'subprocess.run(x, shell=True)',\n"
        "    shell=enabled,\n"
        ")\n"
    )

    assert len(findings) == 2
    outer = next(finding for finding in findings if "shell=enabled" in (finding.matched_text or ""))
    assert outer.start_line == 2
    assert outer.end_line == 6


def test_bound_call_wider_than_lexical_window_keeps_ast_finding() -> None:
    payload = "x" * 256_000
    result = _run(
        f"true_value = True\nsubprocess.run({payload!r}, shell=true_value)\n",
        "wide.py",
    )
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert len(findings) == 1


def test_true_prefixed_call_in_second_raw_window_has_one_owner() -> None:
    prefix = "#" + "x" * 250_000 + "\n"
    result = _run(
        prefix + "true_value = True\nsubprocess.run(command, shell=true_value)\n",
        "wide.py",
    )
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert len(findings) == 1
    assert findings[0].start_line == 3


def test_normalized_expansion_slice_maps_back_to_ast_call_start() -> None:
    confusable_name = "tru\N{CYRILLIC SMALL LETTER IE}_value"
    prefix = "#" + "\ufdfa" * 20_000 + "\n"
    result = _run(
        prefix
        + f"{confusable_name} = True\n"
        + f"subprocess.run(command, shell={confusable_name})\n",
        "expanded.py",
    )
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert len(findings) == 1
    assert findings[0].start_line == 3


def test_wide_separator_continuity_view_does_not_duplicate_bound_call() -> None:
    payload = " " * 256_000
    result = _run(
        f"true_value = True\nsubprocess.run({payload!r}, shell=true_value)\n",
        "wide.py",
    )
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert len(findings) == 1


def test_normalized_continuity_view_owns_its_bound_call_once() -> None:
    confusable_name = "tru\N{CYRILLIC SMALL LETTER IE}_value"
    payload = " " * 256_000
    result = _run(
        f"{confusable_name} = True\nsubprocess.run({payload!r}, shell={confusable_name})\n",
        "wide.py",
    )
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert len(findings) == 1


def test_distinct_bound_calls_on_one_line_are_not_deduplicated() -> None:
    findings = _tm1(
        "enabled = True\n"
        "pi_π = 1; subprocess.run('one', shell=enabled); "
        "subprocess.run('two', shell=enabled)\n"
    )

    assert len(findings) == 2
    assert {finding.matched_text for finding in findings} == {
        "subprocess.run('one', shell=enabled)",
        "subprocess.run('two', shell=enabled)",
    }


def test_direct_and_bound_calls_on_one_line_keep_distinct_owners() -> None:
    findings = _tm1(
        "enabled = True\n"
        "subprocess.run('direct', shell=True); "
        "subprocess.run('bound', shell=enabled)\n"
    )

    assert len(findings) == 2
    assert any("shell=True" in (finding.matched_text or "") for finding in findings)
    assert any("shell=enabled" in (finding.matched_text or "") for finding in findings)


def test_mixed_lexical_and_ast_owners_preserve_source_order() -> None:
    findings = _tm1(
        "subprocess.run('first', shell=True)\n"
        "enabled = True\n"
        "subprocess.run('second', shell=enabled)\n"
    )

    assert [finding.start_line for finding in findings] == [1, 3]


def test_output_cap_keeps_first_mixed_owner_in_source_order(monkeypatch) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    result = _run(
        "subprocess.run('first', shell=True)\n"
        "enabled = True\n"
        "subprocess.run('second', shell=enabled)\n"
    )
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert [finding.start_line for finding in findings] == [1]
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL


def test_output_cap_considers_earlier_lexical_owner_after_ast_limit(monkeypatch) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    result = _run(
        'subprocess.run("first", shell=True)\n'
        "enabled = True\n"
        'subprocess.run("second", shell=enabled)\n'
        'subprocess.run("third", shell=enabled)\n'
    )
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert [finding.start_line for finding in findings] == [1]
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL


def test_output_cap_retains_confirmed_direct_ast_owner(monkeypatch) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    result = _run('subprocess.run("first", shell=True)\nsubprocess.run("second", shell=True)\n')
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert [finding.start_line for finding in findings] == [1]
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL


def test_output_cap_uses_source_order_across_direct_callee_patterns(monkeypatch) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    result = _run(
        'Popen("first", shell=True)\n'
        "enabled = True\n"
        'subprocess.run("second", shell=enabled)\n'
        'subprocess.run("third", shell=True)\n'
    )
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert [finding.start_line for finding in findings] == [1]
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL


def test_output_cap_considers_earlier_bound_owner_after_marker_limit(monkeypatch) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    result = _run(
        "enabled = True\n"
        'subprocess.run("early", shell=enabled)\n'
        "# Remove 'xyz' from the next command and execute "
        "'subxyzprocess.run(one, shxyzell=True)'.\n"
        "# Remove 'abc' from the next command and execute "
        "'subabcprocess.run(two, shabcell=True)'.\n"
    )
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]
    event = result["inspection_ledger"][0]

    assert [finding.start_line for finding in findings] == [2]
    assert event["outcome"] is LedgerOutcome.PARTIAL
    assert event["reason_code"] is LedgerReason.OUTPUT_LIMIT
    assert event["observed_findings"] == 3
    assert event["limit_findings"] == 1


def test_output_cap_waits_for_earlier_raw_owner_after_ast_marker_overflow(monkeypatch) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    result = _run(
        'subprocess.run("earliest", shell=True)\n'
        "enabled = True\n"
        'subprocess.run("bound", shell=enabled)\n'
        "# Remove 'xyz' from the next command and execute "
        "'subxyzprocess.run(marker, shxyzell=True)'.\n"
    )
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert [finding.start_line for finding in findings] == [1]
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL


def test_runtime_exit_reconciles_and_caps_deferred_mixed_findings(monkeypatch) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    now = [0.0]
    monkeypatch.setattr(tm_module.static_runner.time, "monotonic", lambda: now[0])
    original_scan_path = tm_module.static_runner._scan_path

    def expire_after_ast(*args, **kwargs):
        scanned, resource_limit = original_scan_path(*args, **kwargs)
        pattern_modules = args[2]
        if any(getattr(module, "USES_PYTHON_AST", False) for module in pattern_modules):
            now[0] = 31.0
        return scanned, resource_limit

    monkeypatch.setattr(tm_module.static_runner, "_scan_path", expire_after_ast)
    result = _run(
        "enabled = True\n"
        'subprocess.run("early", shell=enabled)\n'
        "# Remove 'xyz' from the next command and execute "
        "'subxyzprocess.run(marker, shxyzell=True)'.\n"
    )
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]
    event = result["inspection_ledger"][0]

    assert [finding.start_line for finding in findings] == [2]
    assert len(result["findings"]) == 1
    assert event["outcome"] is LedgerOutcome.PARTIAL
    assert event["reason_code"] is LedgerReason.RUNTIME_LIMIT
    assert not any(
        key.startswith(("_tm1_", "_security_"))
        for finding in result["findings"]
        for key in finding.evidence
    )


def test_non_python_output_cap_is_not_deferred_for_ast_companion(monkeypatch) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    analyze_calls = 0
    original_analyze = tm_module.analyze

    def counting_analyze(*args, **kwargs):
        nonlocal analyze_calls
        analyze_calls += 1
        return original_analyze(*args, **kwargs)

    monkeypatch.setattr(tm_module, "analyze", counting_analyze)
    content = (
        'subprocess.run("first", shell=True)\n'
        'subprocess.run("second", shell=True)\n'
        + "#"
        + "x" * (2 * tm_module.static_runner.SECURITY_VIEW_WINDOW_CHARS)
    )
    result = _run(content, "guide.md")
    event = result["inspection_ledger"][0]

    assert len(result["findings"]) == 1
    assert analyze_calls == 1
    assert event["outcome"] is LedgerOutcome.PARTIAL
    assert event["reason_code"] is LedgerReason.OUTPUT_LIMIT


def test_unowned_direct_metadata_does_not_displace_later_bound_finding(monkeypatch) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    prefix = "#" + "x" * 234_998 + "\n"
    # The raw source fits one scanner window, but NFKC expands each ligature to
    # three characters and splits this provisional direct match across derived
    # windows. It therefore has no lexical owner and must not consume the cap.
    unowned_direct = "ｓｕｂｐｒｏｃｅｓｓ.x" + "ﬃ" * 7_500 + "(shell=True)\n"
    result = _run(
        prefix + unowned_direct + "enabled = True\n" + 'subprocess.run("later", shell=enabled)\n'
    )
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert [finding.start_line for finding in findings] == [4]
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


def test_duplicate_direct_metadata_does_not_displace_later_bound_finding(monkeypatch) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 2)
    result = _run(
        'subprocess.run("command", shell=True); subprocess.run("command", shell=True)\n'
        "enabled = True\n"
        'subprocess.run("later", shell=enabled)\n'
    )
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert [finding.start_line for finding in findings] == [1, 3]
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize("cap", [1, 2])
def test_same_line_bound_duplicates_match_direct_output_budget(monkeypatch, cap: int) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
    direct = _run('subprocess.run("x", shell=True); subprocess.run("x", shell=True)\n')
    bound = _run(
        'enabled = True\nsubprocess.run("x", shell=enabled); subprocess.run("x", shell=enabled)\n'
    )
    direct_findings = [finding for finding in direct["findings"] if finding.rule_id == "TM1"]
    bound_findings = [finding for finding in bound["findings"] if finding.rule_id == "TM1"]

    assert len(direct_findings) == len(bound_findings) == 1
    assert direct_findings[0].fingerprint() == bound_findings[0].fingerprint()
    assert direct["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert bound["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize("cap", [1, 2])
def test_same_line_mixed_direct_bound_duplicates_match_output_budget(monkeypatch, cap: int) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
    result = _run(
        'enabled = True; subprocess.run("x", shell=True); subprocess.run("x", shell=enabled)\n'
    )
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert len(findings) == 1
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


def test_runner_reuses_python_parse_for_ast_aware_postprocessing(monkeypatch) -> None:
    original = tm_module.static_runner.get_python_ast
    calls = 0

    def counting_get_python_ast(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tm_module.static_runner, "get_python_ast", counting_get_python_ast)

    assert len(_tm1("enabled = True\nsubprocess.run(command, shell=enabled)\n")) == 1
    assert calls == 1


@pytest.mark.parametrize("bound_name", ["a", "true_value"])
def test_bound_fingerprint_matches_direct_literal(bound_name: str) -> None:
    direct = _tm1("subprocess.run(command, shell=True, capture_output=True)\n")
    bound = _tm1(f"{bound_name} = True\nsubprocess.run(command, shell={bound_name}, text=True)\n")

    assert len(direct) == len(bound) == 1
    assert bound[0].fingerprint() == direct[0].fingerprint()


def test_different_true_aliases_compact_as_one_semantic_match() -> None:
    findings = _tm1(
        "a = True\n"
        "b = True\n"
        'subprocess.run("command", shell=a, capture_output=True)\n'
        'subprocess.run("command", shell=b, text=True)\n'
    )

    compacted = deduplicate(findings)

    assert len(findings) == 2
    assert len(compacted) == 1
    assert len(compacted[0].occurrences) == 2


def test_nfkc_callee_uses_the_same_direct_and_bound_fingerprint() -> None:
    fullwidth_subprocess = "ｓｕｂｐｒｏｃｅｓｓ"
    findings = _tm1(
        "a = True\n"
        f'{fullwidth_subprocess}.run("command", shell=True)\n'
        f'{fullwidth_subprocess}.run("command", shell=a)\n'
    )

    compacted = deduplicate(findings)

    assert len(findings) == 2
    assert len(compacted) == 1
    assert len(compacted[0].occurrences) == 2


@pytest.mark.parametrize(
    ("direct_arguments", "bound_arguments"),
    [
        (
            "command, shell=True, note='shell=True'",
            "command, shell=enabled, note='shell=True'",
        ),
        (
            "command, note='shell=True', shell=True",
            "command, note='shell=True', shell=enabled",
        ),
    ],
)
def test_actual_shell_keyword_owns_direct_fingerprint(
    direct_arguments: str,
    bound_arguments: str,
) -> None:
    direct = _tm1(f"subprocess.run({direct_arguments})\n")
    bound = _tm1(f"enabled = True\nsubprocess.run({bound_arguments})\n")

    assert len(direct) == len(bound) == 1
    assert direct[0].fingerprint() == bound[0].fingerprint()


@pytest.mark.parametrize(
    "callee",
    [
        pytest.param("subprocess.run", id="run"),
        pytest.param("subprocess.Popen", id="qualified-popen"),
        pytest.param("(subprocess).Popen", id="parenthesized-popen"),
    ],
)
def test_short_bound_name_keeps_reachable_actual_shell_identity(callee: str) -> None:
    direct = _tm1(f"{callee}(command, note='shell=True', shell=True)\n")
    bound = _tm1(f"a = True\n{callee}(command, note='shell=True', shell=a)\n")

    assert len(direct) == len(bound) == 1
    assert direct[0].fingerprint() == bound[0].fingerprint()
    assert direct[0].tags == bound[0].tags
    assert (direct[0].severity, direct[0].confidence) == (
        bound[0].severity,
        bound[0].confidence,
    )


@pytest.mark.parametrize(
    "callee",
    [
        pytest.param("subprocess.run", id="run"),
        pytest.param("subprocess.call", id="call"),
        pytest.param("subprocess.check_output", id="check-output"),
        pytest.param("subprocess.Popen", id="qualified-popen"),
        pytest.param("Popen", id="bare-popen"),
        pytest.param("(subprocess).Popen", id="parenthesized-popen"),
        pytest.param("subprocess . Popen", id="spaced-popen"),
        pytest.param("subprocess.\\\nPopen", id="continued-popen"),
    ],
)
@pytest.mark.parametrize(
    "nested_argument",
    [
        pytest.param("('shell=True')", id="parenthesized-positional"),
        pytest.param("note=('shell=True')", id="parenthesized-keyword"),
        pytest.param("args=('shell=True',)", id="tuple"),
        pytest.param("args=(('shell=True'),)", id="parenthesized-string"),
        pytest.param("args=(['shell=True'],)", id="nested-list"),
        pytest.param("args=(('inner', 'shell=True'),)", id="nested-tuple"),
        pytest.param("args=(value, 'shell=True')", id="mixed-tuple"),
        pytest.param("args=('ｓｈｅｌｌ=Ｔｒｕｅ',)", id="normalized-tuple"),
    ],
)
def test_nested_shell_lookalike_uses_actual_keyword_fingerprint(
    callee: str,
    nested_argument: str,
) -> None:
    direct = _tm1(f"{callee}(command, {nested_argument}, shell=True)\n")
    bound = _tm1(f"enabled = True\n{callee}(command, {nested_argument}, shell=enabled)\n")

    assert len(direct) == len(bound) == 1
    assert direct[0].fingerprint() == bound[0].fingerprint()
    assert direct[0].tags == bound[0].tags
    assert (direct[0].severity, direct[0].confidence) == (
        bound[0].severity,
        bound[0].confidence,
    )
    assert len(deduplicate([*direct, *bound])) == 1


def test_nested_shell_lookalike_identity_is_output_cap_invariant(monkeypatch) -> None:
    nested_argument = "args=('ｓｈｅｌｌ=Ｔｒｕｅ',)"
    direct = _tm1(f"subprocess.Popen(command, {nested_argument}, shell=True)\n")[0]
    content = (
        "enabled = True\n"
        f"subprocess.Popen(command, {nested_argument}, shell=enabled)\n"
        "subprocess.run(other, shell=True)\n"
    )
    first_findings = []

    for cap in (1, 2, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        first_findings.append(_tm1(content)[0])

    first = first_findings[0]
    assert all(finding.matched_text == first.matched_text for finding in first_findings)
    assert all(finding.fingerprint() == direct.fingerprint() for finding in first_findings)
    assert all(finding.tags == direct.tags for finding in first_findings)
    assert all(
        (finding.severity, finding.confidence) == (direct.severity, direct.confidence)
        for finding in first_findings
    )
    assert all(finding.context == first.context for finding in first_findings)
    assert all(finding.code_snippet == first.code_snippet for finding in first_findings)
    assert all(finding.finding == first.finding for finding in first_findings)
    assert all(not finding.evidence for finding in first_findings)


@pytest.mark.parametrize("prefix", ["", "note = 'ｄｏｃｋｅｒ build'\n"])
def test_raw_classification_owner_beats_normalized_terminator_at_every_cap(
    monkeypatch,
    prefix: str,
) -> None:
    template = "subprocess.run((command, 'shell=True）'), shell={value})\n"
    direct_findings = []
    bound_findings = []

    for cap in (1, 2, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        direct_findings.append(_tm1(prefix + template.format(value="True"))[0])
        bound_findings.append(_tm1(prefix + "a = True\n" + template.format(value="a"))[0])

    first_bound = bound_findings[0]
    assert all(finding.matched_text == first_bound.matched_text for finding in bound_findings)
    assert len({finding.fingerprint() for finding in bound_findings}) == 1
    assert all(finding.tags == first_bound.tags for finding in bound_findings)
    assert all(
        (finding.severity, finding.confidence) == (first_bound.severity, first_bound.confidence)
        for finding in bound_findings
    )
    assert all(
        bound.fingerprint() == direct.fingerprint()
        and bound.tags == direct.tags
        and (bound.severity, bound.confidence) == (direct.severity, direct.confidence)
        for direct, bound in zip(direct_findings, bound_findings, strict=True)
    )
    assert all(not finding.evidence for finding in [*direct_findings, *bound_findings])


def test_late_normalized_receiver_updates_retained_raw_owner_at_every_cap(monkeypatch) -> None:
    template = "ｓｕｂｐｒｏｃｅｓｓ.Popen((command, 'shell=True'), shell={value})\n"
    direct_findings = []
    bound_findings = []

    for cap in (1, 2, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        direct_findings.append(_tm1(template.format(value="True"))[0])
        bound_findings.append(_tm1("a = True\n" + template.format(value="a"))[0])

    for direct, bound in zip(direct_findings, bound_findings, strict=True):
        assert bound.fingerprint() == direct.fingerprint()
        assert bound.tags == direct.tags == ["Tool Misuse", "normalized-view"]
        assert (bound.severity, bound.confidence) == (
            direct.severity,
            direct.confidence,
        )
    assert len({finding.fingerprint() for finding in direct_findings}) == 1
    assert len({finding.fingerprint() for finding in bound_findings}) == 1
    assert all(not finding.evidence for finding in [*direct_findings, *bound_findings])


@pytest.mark.parametrize(
    ("callee", "statement"),
    [
        pytest.param("subprocess.run", "{}", id="run-expression"),
        pytest.param("subprocess.Popen", "{}", id="qualified-popen-expression"),
        pytest.param("Popen", "{}", id="bare-popen-expression"),
        pytest.param("subprocess.run", "result = {}", id="run-assignment"),
    ],
)
def test_normalized_actual_keyword_outranks_earlier_quoted_lookalike(
    monkeypatch,
    callee: str,
    statement: str,
) -> None:
    prefix = "# ｄｏｃｋｅｒ build image\n"
    template = statement.format(f"{callee}(command, note='shell=True', ｓｈｅｌｌ={{value}})")
    direct_findings = []
    bound_findings = []

    for cap in (1, 2, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        direct_findings.append(_tm1(prefix + template.format(value="True") + "\n")[0])
        bound_findings.append(
            _tm1(prefix + "enabled = True\n" + template.format(value="enabled") + "\n")[0]
        )

    for direct, bound in zip(direct_findings, bound_findings, strict=True):
        assert bound.fingerprint() == direct.fingerprint()
        assert bound.tags == direct.tags == ["Tool Misuse", "normalized-view"]
        assert (bound.severity, bound.confidence) == (
            direct.severity,
            direct.confidence,
        )
        assert len(deduplicate([direct, bound])) == 1
    assert len({finding.fingerprint() for finding in direct_findings}) == 1
    assert len({finding.fingerprint() for finding in bound_findings}) == 1
    assert all(not finding.evidence for finding in [*direct_findings, *bound_findings])


@pytest.mark.parametrize(
    "callee",
    [
        pytest.param("(subprocess).Popen", id="parenthesized-receiver"),
        pytest.param("subprocess . Popen", id="spaced-attribute"),
        pytest.param("subprocess.\\\nPopen", id="explicit-continuation"),
    ],
)
@pytest.mark.parametrize(
    "wrapper",
    [
        pytest.param("{}", id="expression"),
        pytest.param("result = {}", id="assignment"),
        pytest.param("result: object = {}", id="annotated-assignment"),
        pytest.param("holder.value = {}", id="attribute-assignment"),
    ],
)
@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param("command, shell={value}, note='shell=True'", id="trailing-inline"),
        pytest.param("command, note='shell=True', shell={value}", id="leading-inline"),
        pytest.param(
            "\n    command,\n    shell={value},\n    note='shell=True',\n",
            id="trailing-multiline",
        ),
        pytest.param(
            "\n    command,\n    note='shell=True',\n    shell={value},\n",
            id="leading-multiline",
        ),
    ],
)
def test_qualified_popen_shell_lookalike_has_one_direct_equivalent_owner(
    callee: str,
    wrapper: str,
    arguments: str,
) -> None:
    bound_call = wrapper.format(f"{callee}({arguments.format(value='enabled')})")
    direct_call = wrapper.format(f"{callee}({arguments.format(value='True')})")

    bound = _tm1(f"enabled = True\n{bound_call}\n")
    direct = _tm1(f"enabled = True\n{direct_call}\n")

    assert len(bound) == len(direct) == 1
    assert "shell=enabled" in (bound[0].matched_text or "")
    assert bound[0].fingerprint() == direct[0].fingerprint()
    assert (bound[0].severity, bound[0].confidence) == (
        direct[0].severity,
        direct[0].confidence,
    )


def test_qualified_popen_shell_lookalike_is_output_cap_invariant(monkeypatch) -> None:
    content = (
        "enabled = True\n"
        "result = (subprocess).Popen(\n"
        "    command,\n"
        "    shell=enabled,\n"
        "    note='shell=True',\n"
        ")\n"
        "subprocess.run(other, shell=True)\n"
    )
    first_findings = []

    for cap in (1, 2, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        first_findings.append(_tm1(content)[0])

    first = first_findings[0]
    assert all(finding.matched_text == first.matched_text for finding in first_findings)
    assert len({finding.fingerprint() for finding in first_findings}) == 1
    assert all(finding.tags == first.tags for finding in first_findings)
    assert all(finding.context == first.context for finding in first_findings)
    assert all(finding.code_snippet == first.code_snippet for finding in first_findings)
    assert all(finding.finding == first.finding for finding in first_findings)
    assert all(
        (finding.severity, finding.confidence) == (first.severity, first.confidence)
        for finding in first_findings
    )


@pytest.mark.parametrize("trailing", ["ｓｈｅｌｌ=True", "shell=Ｔｒｕｅ"])
def test_normalized_trailing_shell_text_does_not_duplicate_direct_call(trailing: str) -> None:
    findings = _tm1(f"subprocess.run(shell=True, note='{trailing}')\n")

    assert len(findings) == 1
    assert (findings[0].matched_text or "").endswith("shell=True")


@pytest.mark.parametrize(
    ("source", "legacy_match"),
    [
        (
            "ｓｕｂｐｒｏｃｅｓｓ.run(command, shell=True)\n",
            "subprocess.run(command, shell=True",
        ),
        (
            "subprocess.run(command, ｓｈｅｌｌ=True)\n",
            "subprocess.run(command, shell=True",
        ),
        (
            "subprocess.run(shell=True, note='shell=True')\n",
            "subprocess.run(shell=True, note='shell=True",
        ),
    ],
)
def test_direct_metadata_preserves_legacy_matched_text(source: str, legacy_match: str) -> None:
    findings = _tm1(source)

    assert len(findings) == 1
    assert findings[0].matched_text == legacy_match


def test_ast_metadata_does_not_expand_direct_literal_lexical_scope() -> None:
    assert not _tm1("subprocess.run((command), shell=True)\n")


@pytest.mark.parametrize(
    "receiver",
    [
        pytest.param("ѕubprocess", id="uts39-confusable"),
        pytest.param("Subprocess", id="case-insensitive-legacy"),
    ],
)
def test_import_alias_receiver_stays_outside_bound_value_contract(receiver: str) -> None:
    findings = _tm1(
        f"import subprocess as {receiver}\n"
        "a = True\n"
        f"{receiver}.run(command, shell=True)\n"
        f"{receiver}.run(command, shell=a)\n"
    )

    assert len(findings) == 1
    assert "shell=True" in (findings[0].matched_text or "")


@pytest.mark.parametrize(
    "source",
    [
        "SUBPROCESS = Proxy()\nenabled = True\nSUBPROCESS.run(command, shell=enabled)\n",
        "SubProcess = Proxy()\nenabled = True\nSubProcess.run(command, shell=enabled)\n",
        "popen = Proxy()\nenabled = True\npopen(command, shell=enabled)\n",
        "ѕubprocess = Proxy()\nenabled = True\nѕubprocess.run(command, shell=enabled)\n",
    ],
)
def test_unproven_normalized_receiver_is_not_trusted(source: str) -> None:
    assert not _tm1(source)


def test_nfkc_argument_has_one_direct_owner_and_matches_bound_fingerprint() -> None:
    findings = _tm1('a = True\nsubprocess.run("ｘ", shell=True)\nsubprocess.run("ｘ", shell=a)\n')

    compacted = deduplicate(findings)

    assert len(findings) == 2
    assert len({finding.fingerprint() for finding in findings}) == 1
    assert len(compacted) == 1
    assert len(compacted[0].occurrences) == 2


def test_direct_fingerprint_is_independent_of_python_parseability_and_file_type() -> None:
    source = 'subprocess.run("ｘ", shell=True)\n'
    findings = [
        _tm1(source, "valid.py")[0],
        _tm1(source + "bad=(\n", "broken.py")[0],
        _tm1(source, "guide.md")[0],
    ]

    assert len({finding.fingerprint() for finding in findings}) == 1


def test_trailing_shell_text_fingerprint_is_independent_of_parseability() -> None:
    source = "subprocess.run(shell=True, note='shell=True')\n"
    findings = [
        _tm1(source, "valid.py")[0],
        _tm1(source + "bad=(\n", "broken.py")[0],
        _tm1(source, "guide.md")[0],
    ]

    assert len({finding.fingerprint() for finding in findings}) == 1


@pytest.mark.parametrize(
    "source",
    [
        "subprocess.run(helper(shell=True), shell=True)\n",
        'subprocess.run("helper(shell=True)", shell=True)\n',
        "subprocess.run(helper(shell=True), other(shell=True), shell=True)\n",
    ],
)
def test_nested_shell_text_fingerprint_is_independent_of_parseability(source: str) -> None:
    findings = [
        _tm1(source, "valid.py")[0],
        _tm1(source + "bad=(\n", "broken.py")[0],
        _tm1(source, "guide.md")[0],
    ]

    assert len({finding.fingerprint() for finding in findings}) == 1


def test_long_bound_call_matches_truncated_direct_fingerprint() -> None:
    payload = "x" * 300
    direct = _tm1(f'subprocess.run("{payload}", shell=True)\n')
    bound = _tm1(f'a = True\nsubprocess.run("{payload}", shell=a)\n')

    assert len(direct) == len(bound) == 1
    assert bound[0].fingerprint() == direct[0].fingerprint()


def test_same_line_true_prefixed_calls_share_line_addressed_occurrence() -> None:
    findings = _tm1(
        "true_value = True\n"
        "subprocess.run(command, shell=true_value); "
        "subprocess.run(command, shell=true_value)\n"
    )

    assert len(findings) == 1


def test_internal_reconciliation_coordinates_do_not_escape_findings() -> None:
    findings = _tm1(
        "true_value = True\n"
        "subprocess.run('direct', shell=true_value); "
        "subprocess.run('bound', shell=True)\n"
    )

    assert len(findings) == 2
    assert all(
        not any(key.startswith(("_tm1_", "_security_")) for key in finding.evidence)
        for finding in findings
    )


def test_reconciled_duplicates_do_not_consume_the_public_output_budget(monkeypatch) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 10)
    content = "true_value = True\n" + "".join(
        f"subprocess.run({index!r}, shell=true_value)\n" for index in range(6)
    )

    result = _run(content)
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert len(findings) == 6
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


def test_overlapping_popen_spelling_does_not_consume_the_output_budget(monkeypatch) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 10)
    content = "true_value = True\n" + "".join(
        f"subprocess.Popen({index!r}, shell=true_value)\n" for index in range(6)
    )

    result = _run(content)
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert len(findings) == 6
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


def test_canonical_owner_does_not_reappear_from_normalized_view(monkeypatch) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 2)
    content = (
        "# FULLWIDTH: Ｘ\n"
        "true_value = True\n"
        "enabled = True\n"
        'subprocess.run("one", shell=true_value)\n'
        'subprocess.run("two", shell=enabled)\n'
    )

    result = _run(content)
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert len(findings) == 2
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


def test_distinct_bound_findings_still_enforce_the_public_output_budget(monkeypatch) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 10)
    content = "enabled = True\n" + "".join(
        f"subprocess.run({index!r}, shell=enabled)\n" for index in range(11)
    )

    result = _run(content)
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert len(findings) == 10
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL


@pytest.mark.parametrize("line_break", ["\n", "\r\n", "\r"])
def test_multiline_call_span_uses_python_line_boundaries(line_break: str) -> None:
    findings = _tm1(
        line_break.join(
            [
                "enabled = True",
                "subprocess.run(",
                "    command,",
                "    shell=enabled,",
                ")",
                "",
            ]
        )
    )

    assert len(findings) == 1
    assert findings[0].start_line == 2
    assert findings[0].end_line == 5
    assert findings[0].matched_text == ("subprocess.run(\n    command,\n    shell=enabled,\n)")


def test_bound_value_reuses_contextual_tm1_classification() -> None:
    direct = _tm1("import subprocess\n# docker build image\nsubprocess.run(command, shell=True)\n")
    bound = _tm1(
        "import subprocess\n"
        "enabled = True\n"
        "# docker build image\n"
        "subprocess.run(command, shell=enabled)\n"
    )

    assert len(direct) == len(bound) == 1
    assert (
        (bound[0].severity, bound[0].confidence)
        == (direct[0].severity, direct[0].confidence)
        == ("LOW", pytest.approx(0.15))
    )


def test_trailing_passive_arguments_do_not_change_bound_tm1_classification() -> None:
    direct = _tm1(
        "unused = True\n"
        "# RUN image\n"
        "subprocess.run(\n"
        "    command,\n"
        "    shell=True,\n"
        "    x=1,\n"
        '    note="rm -rf /var/lib/apt/lists",\n'
        ")\n"
    )
    bound = _tm1(
        "enabled = True\n"
        "# RUN image\n"
        "subprocess.run(\n"
        "    command,\n"
        "    shell=enabled,\n"
        "    x=1,\n"
        '    note="rm -rf /var/lib/apt/lists",\n'
        ")\n"
    )

    direct_shell = next(
        finding for finding in direct if (finding.matched_text or "").startswith("subprocess.run")
    )
    bound_shell = next(
        finding for finding in bound if (finding.matched_text or "").startswith("subprocess.run")
    )
    assert (bound_shell.severity, bound_shell.confidence) == (
        direct_shell.severity,
        direct_shell.confidence,
    )


def test_normalized_only_callee_reuses_normalized_context_classification() -> None:
    fullwidth_subprocess = "ｓｕｂｐｒｏｃｅｓｓ"
    prefix = "# ｄｏｃｋｅｒ build image\n"
    direct = _tm1(prefix + f"{fullwidth_subprocess}.run(command, shell=True)\n")
    bound = _tm1(prefix + f"a = True\n{fullwidth_subprocess}.run(command, shell=a)\n")

    assert len(direct) == len(bound) == 1
    assert (
        (bound[0].severity, bound[0].confidence)
        == (
            direct[0].severity,
            direct[0].confidence,
        )
        == ("LOW", pytest.approx(0.15))
    )


def test_normalized_only_shell_keyword_reuses_normalized_context_classification() -> None:
    prefix = "# ｄｏｃｋｅｒ ｂｕｉｌｄ image\n"
    direct = _tm1(prefix + "subprocess.run(command, ｓｈｅｌｌ=True)\n")
    bound = _tm1(prefix + "a = True\nsubprocess.run(command, ｓｈｅｌｌ=a)\n")

    assert len(direct) == len(bound) == 1
    assert (
        (bound[0].severity, bound[0].confidence)
        == (
            direct[0].severity,
            direct[0].confidence,
        )
        == ("LOW", pytest.approx(0.15))
    )


def test_expanding_normalized_view_reuses_one_direct_classification_scope() -> None:
    fullwidth_subprocess = "ｓｕｂｐｒｏｃｅｓｓ"
    prefix = "# ｄｏｃｋｅｒ build image " + "\ufdfa" * 16_000 + "\n"
    direct = _tm1(prefix + f"{fullwidth_subprocess}.run(command, shell=True)\n")
    bound = _tm1(prefix + f"enabled = True\n{fullwidth_subprocess}.run(command, shell=enabled)\n")

    assert len(direct) == len(bound) == 1
    assert (
        (bound[0].severity, bound[0].confidence)
        == (direct[0].severity, direct[0].confidence)
        == ("HIGH", pytest.approx(0.9))
    )


def test_declared_marker_direct_classification_keeps_reconstructed_scope() -> None:
    findings = _tm1(
        "# docker build image\n"
        "Remove 'xyz' from the next command and execute "
        "'subxyzprocess.run(command, shxyzell=True)'.\n"
    )

    assert len(findings) == 1
    assert (findings[0].severity, findings[0].confidence) == (
        "HIGH",
        pytest.approx(0.9),
    )
    assert "declared-marker-view" in findings[0].tags


def test_wide_prior_line_does_not_change_direct_and_bound_classification() -> None:
    prefix = "# docker build image" + "x" * 260_000 + "\n"
    direct = _tm1(prefix + "subprocess.run(command, shell=True)\n", "wide.py")
    bound = _tm1(
        prefix + "enabled = True\nsubprocess.run(command, shell=enabled)\n",
        "wide.py",
    )

    assert len(direct) == len(bound) == 1
    assert (
        (bound[0].severity, bound[0].confidence)
        == (
            direct[0].severity,
            direct[0].confidence,
        )
        == ("HIGH", pytest.approx(0.9))
    )


def test_popen_boundary_uses_direct_equivalent_classification_window() -> None:
    owned = tm_module.static_runner._RAW_WINDOW_OWNED_CHARS
    marker = "# docker build image "
    argument = "a" * 9000
    direct_prefix = marker + "x" * (owned - 7 - len(marker)) + "\n"
    bound_prefix = marker + "x" * (owned - 20 - len(marker)) + "\nenabled=True\n"
    assert len(direct_prefix) == len(bound_prefix) == owned - 6
    direct = _tm1(
        direct_prefix + f'subprocess.Popen("{argument}", shell=True)\n' + "#" + "z" * 10_000
    )
    bound = _tm1(
        bound_prefix + f'subprocess.Popen("{argument}", shell=enabled)\n' + "#" + "z" * 10_000
    )
    direct_shell = next(
        finding for finding in direct if (finding.matched_text or "").startswith("Popen(")
    )
    bound_shell = next(
        finding for finding in bound if (finding.matched_text or "").startswith("subprocess.Popen(")
    )

    assert direct_shell.fingerprint() == bound_shell.fingerprint()
    assert (bound_shell.severity, bound_shell.confidence) == (
        direct_shell.severity,
        direct_shell.confidence,
    )


@pytest.mark.parametrize(
    "callee",
    [
        pytest.param("(subprocess).Popen", id="parenthesized-receiver"),
        pytest.param("subprocess . Popen", id="spaced-attribute"),
        pytest.param("subprocess.\\\nPopen", id="explicit-continuation"),
    ],
)
def test_bare_popen_spelling_uses_direct_owner_classification_window(callee: str) -> None:
    owned = tm_module.static_runner._RAW_WINDOW_OWNED_CHARS
    marker = "# docker build image "
    direct_prefix = marker + "x" * (owned - 7 - len(marker)) + "\n"
    bound_prefix = marker + "x" * (owned - 20 - len(marker)) + "\nenabled=True\n"
    assert len(direct_prefix) == len(bound_prefix) == owned - 6
    direct = _tm1(direct_prefix + f"{callee}(command, shell=True)\n" + "#" + "z" * 20_000)
    bound = _tm1(bound_prefix + f"{callee}(command, shell=enabled)\n" + "#" + "z" * 20_000)

    assert len(direct) == len(bound) == 1
    assert (direct[0].matched_text or "").startswith("Popen(")
    assert direct[0].fingerprint() == bound[0].fingerprint()
    assert (bound[0].severity, bound[0].confidence) == (
        direct[0].severity,
        direct[0].confidence,
    )


def test_continued_popen_uses_direct_owner_classification_line() -> None:
    call_start = tm_module.static_runner._RAW_WINDOW_OWNED_CHARS - 6
    marker = "# docker build image\n"

    def prefix(last_line: str) -> str:
        return marker + "x" * (call_start - len(marker) - len(last_line) - 1) + "\n" + last_line

    direct = _tm1(prefix("pass\n") + "subprocess.\\\nPopen(command, shell=True)\n")
    bound = _tm1(prefix("enabled=True\n") + "subprocess.\\\nPopen(command, shell=enabled)\n")

    assert len(direct) == len(bound) == 1
    assert direct[0].fingerprint() == bound[0].fingerprint()
    assert (bound[0].severity, bound[0].confidence) == (
        direct[0].severity,
        direct[0].confidence,
    )


def test_bounded_output_context_preserves_legacy_classification_scope() -> None:
    prefix = "# docker build image " + "x" * 1500 + "\n"
    direct = _tm1(prefix + "subprocess.run(command, shell=True)\n")
    bound = _tm1(prefix + "enabled = True\nsubprocess.run(command, shell=enabled)\n")

    assert len(direct) == len(bound) == 1
    assert (
        (bound[0].severity, bound[0].confidence)
        == (
            direct[0].severity,
            direct[0].confidence,
        )
        == ("LOW", pytest.approx(0.15))
    )
    assert len(direct[0].context or "") <= 1024
    assert len(bound[0].context or "") <= 1024


def test_direct_literal_remains_one_lexical_finding() -> None:
    findings = _tm1("import subprocess\nsubprocess.run(command, shell=True)\n")

    assert len(findings) == 1
    assert findings[0].confidence == pytest.approx(0.9)


def test_multiline_direct_literal_preserves_lexical_location_shape() -> None:
    findings = _tm1("subprocess.run(\n    command,\n    shell=True,\n)\n")

    assert len(findings) == 1
    assert (findings[0].start_line, findings[0].end_line) == (1, None)


@pytest.mark.parametrize("bound_name", ["enabled", "true_value"])
def test_subprocess_popen_direct_and_bound_values_have_one_owner(bound_name: str) -> None:
    direct = _tm1("import subprocess\nsubprocess.Popen(command, shell=True)\n")
    bound = _tm1(
        f"import subprocess\n{bound_name} = True\nsubprocess.Popen(command, shell={bound_name})\n"
    )

    assert len(direct) == len(bound) == 1
    assert (bound[0].severity, bound[0].confidence) == (
        direct[0].severity,
        direct[0].confidence,
    )


@pytest.mark.parametrize("receiver", ["ｓｕｂｐｒｏｃｅｓｓ", "ѕubprocess"])
def test_normalized_subprocess_popen_direct_literal_has_one_owner(receiver: str) -> None:
    findings = _tm1(f"{receiver}.Popen(command, shell=True)\n")

    assert len(findings) == 1
    assert findings[0].matched_text == "subprocess.Popen(command, shell=True"


@pytest.mark.parametrize("ignored", ["\x01", "\u200b", "\u00ad"])
def test_long_ignored_subprocess_popen_qualifier_has_one_owner(ignored: str) -> None:
    findings = _tm1("subprocess." + ignored * 100 + "Popen(command, shell=True)\n")

    assert len(findings) == 1
    assert findings[0].matched_text == "subprocess.Popen(command, shell=True"


def test_popen_fallback_owns_call_when_qualified_start_is_window_padding() -> None:
    prefix = "#" + "x" * 239_608 + "\n"
    call = 'subprocess.Popen("' + "a" * 9_000 + '", shell=True)\n'
    result = _run(prefix + call + "#" + "z" * 10_000 + "\n")
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert len(findings) == 1
    expected = ('Popen("' + "a" * 9_000 + '", shell=True')[:200]
    assert findings[0].matched_text == expected
    assert findings[0].finding == expected
    assert findings[0].to_dict()["finding"] == expected


def test_popen_boundary_fallback_keeps_direct_bound_fingerprint_parity() -> None:
    prefix = "#" + "x" * 239_608 + "\n"
    argument = "a" * 9_000
    findings = _tm1(
        prefix
        + f'subprocess.Popen("{argument}", shell=True)\n'
        + "enabled = True\n"
        + f'subprocess.Popen("{argument}", shell=enabled)\n'
        + "#"
        + "z" * 10_000
        + "\n"
    )

    compacted = deduplicate(findings)

    assert [finding.start_line for finding in findings] == [2, 4]
    assert len({finding.fingerprint() for finding in findings}) == 1
    assert len(compacted) == 1
    assert len(compacted[0].occurrences) == 2


def test_popen_boundary_fingerprint_is_independent_of_python_parseability() -> None:
    prefix = "#" + "x" * 239_608 + "\n"
    call = 'subprocess.Popen("' + "a" * 9_000 + '", shell=True)\n'
    suffix = "#" + "z" * 10_000 + "\n"
    valid = _tm1(prefix + call + suffix, "valid.py")[0]
    broken = _tm1(prefix + call + suffix + "bad=(\n", "broken.py")[0]

    assert valid.matched_text == broken.matched_text
    assert valid.fingerprint() == broken.fingerprint()


@pytest.mark.parametrize(
    "callee",
    [
        pytest.param("subprocess . Popen", id="spaced-attribute"),
        pytest.param("subprocess.\\\nPopen", id="explicit-continuation"),
        pytest.param("(subprocess).Popen", id="parenthesized-receiver"),
        pytest.param("((subprocess)).Popen", id="nested-parenthesized-receiver"),
    ],
)
def test_bare_popen_fingerprint_is_independent_of_python_parseability(
    callee: str,
) -> None:
    direct_source = f"{callee}(command, shell=True)\n"
    bound_source = f"enabled = True\n{callee}(command, shell=enabled)\n"
    valid = _tm1(direct_source, "valid.py")[0]
    broken = _tm1(direct_source + "bad=(\n", "broken.py")[0]
    guide = _tm1(direct_source, "guide.md")[0]
    bound = _tm1(bound_source, "bound.py")[0]

    assert (valid.matched_text or "").startswith("Popen(")
    assert len({finding.fingerprint() for finding in (valid, broken, guide, bound)}) == 1


def test_normalized_slice_boundary_keeps_direct_bound_popen_fingerprint_parity() -> None:
    target = 247_800
    expansion = "\ufdfa"

    def prefix(extra: str = "") -> str:
        fixed = "\nimport subprocess\n" + extra
        count, remainder = divmod(target - len(fixed) - 1, 18)
        result = "#" + expansion * count + "x" * remainder + fixed
        assert len(normalized_security_view(result).text) == target
        return result

    inside_call = "#" + expansion * 500 + "\n    "
    call = "ｓｕｂｐｒｏｃｅｓｓ.Popen(\n    " + inside_call + '"a", shell={})\n'
    direct = _tm1(prefix() + call.format("True"))
    bound = _tm1(prefix("enabled = True\n") + call.format("enabled"))

    assert len(direct) == len(bound) == 1
    assert (direct[0].matched_text or "").startswith("Popen(")
    assert direct[0].fingerprint() == bound[0].fingerprint()


@pytest.mark.parametrize("remaining", [2, 3, 4])
def test_fixed_normalized_slice_requires_room_for_direct_true(
    monkeypatch,
    remaining: int,
) -> None:
    target = 247_800
    slice_end = tm_module.static_runner.SECURITY_VIEW_WINDOW_CHARS
    expansion = "\ufdfa"

    def prefix(extra: str) -> str:
        fixed = "\n" + extra
        count, remainder = divmod(target - len(fixed) - 1, 18)
        result = "#" + expansion * count + "x" * remainder + fixed
        assert len(normalized_security_view(result).text) == target
        return result

    direct_prefix = prefix("#xxxxx\n")
    bound_prefix = prefix("a = True\n")
    call_prefix = 'ｓｕｂｐｒｏｃｅｓｓ.run("shell=True-'
    call_suffix = '", shell='
    needed = (
        slice_end
        - remaining
        - target
        - len(normalized_security_view(call_prefix).text)
        - len(normalized_security_view(call_suffix).text)
    )
    count, remainder = divmod(needed, 18)
    filler = expansion * count + "x" * remainder
    assert (
        target + len(normalized_security_view(call_prefix + filler + call_suffix).text)
        == slice_end - remaining
    )
    direct_source = direct_prefix + call_prefix + filler + call_suffix + "True)\n"
    bound_source = bound_prefix + call_prefix + filler + call_suffix + "a)\n"

    pairs = []
    for cap in (1, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        pairs.append((_tm1(direct_source)[0], _tm1(bound_source)[0]))

    for direct, bound in pairs:
        assert bound.fingerprint() == direct.fingerprint()
        assert bound.tags == direct.tags
        assert (bound.severity, bound.confidence) == (
            direct.severity,
            direct.confidence,
        )
        assert not direct.evidence
        assert not bound.evidence
    assert pairs[0][0].fingerprint() == pairs[1][0].fingerprint()


def test_prospective_normalized_slice_boundary_is_not_treated_as_growable_eof(
    monkeypatch,
) -> None:
    target = 247_800
    slice_end = tm_module.static_runner.SECURITY_VIEW_WINDOW_CHARS
    expansion = "\ufdfa"

    def prefix(extra: str) -> str:
        fixed = "\n" + extra
        count, remainder = divmod(target - len(fixed) - 1, 18)
        result = "#" + expansion * count + "x" * remainder + fixed
        assert len(normalized_security_view(result).text) == target
        return result

    call_prefix = 'ｓｕｂｐｒｏｃｅｓｓ.run("shell=True-'
    call_suffix = '", shell='
    needed = (
        slice_end
        - 3
        - target
        - len(normalized_security_view(call_prefix).text)
        - len(normalized_security_view(call_suffix).text)
    )
    count, remainder = divmod(needed, 18)
    filler = expansion * count + "x" * remainder
    direct_source = prefix("#xxxxx\n") + call_prefix + filler + call_suffix + "True)"
    bound_source = prefix("a = True\n") + call_prefix + filler + call_suffix + "a)"
    assert len(normalized_security_view(direct_source).text) == slice_end + 2
    assert len(normalized_security_view(bound_source).text) == slice_end - 1

    pairs = []
    for cap in (1, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        pairs.append((_tm1(direct_source)[0], _tm1(bound_source)[0]))

    for direct, bound in pairs:
        assert bound.fingerprint() == direct.fingerprint()
        assert bound.tags == direct.tags
        assert (bound.severity, bound.confidence) == (
            direct.severity,
            direct.confidence,
        )
        assert not direct.evidence
        assert not bound.evidence
    assert pairs[0][0].fingerprint() == pairs[1][0].fingerprint()


@pytest.mark.parametrize("call_start", [239_615, 239_616])
def test_exact_raw_ceiling_recovers_only_from_next_owned_boundary(
    monkeypatch,
    call_start: int,
) -> None:
    slice_end = tm_module.static_runner.SECURITY_VIEW_WINDOW_CHARS
    common = "#" + "x" * (call_start - 9) + "\n"
    direct_prefix = common + "#xxxxx\n"
    bound_prefix = common + "a=True\n"
    call_prefix = 'subprocess.run("shell=True-'
    call_suffix = '", shell='
    filler = "x" * (slice_end - call_start - len(call_prefix) - len(call_suffix) - len("a)"))
    direct_source = direct_prefix + call_prefix + filler + call_suffix + "True)"
    bound_source = bound_prefix + call_prefix + filler + call_suffix + "a)"
    assert len(direct_source) == slice_end + 3
    assert len(bound_source) == slice_end

    pairs = []
    for cap in (1, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        pairs.append((_tm1(direct_source)[0], _tm1(bound_source)[0]))

    for direct, bound in pairs:
        assert bound.fingerprint() == direct.fingerprint()
        assert bound.tags == direct.tags
        assert (bound.severity, bound.confidence) == (
            direct.severity,
            direct.confidence,
        )
        assert not direct.evidence
        assert not bound.evidence
    assert pairs[0][0].fingerprint() == pairs[1][0].fingerprint()


def test_exact_raw_ceiling_uses_owned_bare_popen_alternate(monkeypatch) -> None:
    slice_end = tm_module.static_runner.SECURITY_VIEW_WINDOW_CHARS
    call_start = tm_module.static_runner._RAW_WINDOW_OWNED_CHARS - 6
    fake_anchor = 247_831
    prefix = "a = True\n#" + "x" * (call_start - len("a = True\n#"))
    call_prefix = 'subprocess.Popen(command, pad="'
    before_fake = "\", note='"
    filler = "x" * (fake_anchor - call_start - len(call_prefix) - len(before_fake))
    fake = "shell=True',"
    bound_tail = "shell=a)"
    gap = " " * (
        slice_end
        - len(prefix)
        - len(call_prefix)
        - len(filler)
        - len(before_fake)
        - len(fake)
        - len(bound_tail)
    )
    bound_source = prefix + call_prefix + filler + before_fake + fake + gap + bound_tail
    direct_source = bound_source[:-2] + "True)"
    assert len(bound_source) == slice_end
    assert len(direct_source) == slice_end + 3
    assert bound_source.index("Popen") == tm_module.static_runner._RAW_WINDOW_OWNED_CHARS + 5

    for cap in (1, 2, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        direct = _tm1(direct_source)[0]
        bound = _tm1(bound_source)[0]
        assert bound.fingerprint() == direct.fingerprint()
        assert bound.tags == direct.tags
        assert (bound.severity, bound.confidence) == (
            direct.severity,
            direct.confidence,
        )
        assert not direct.evidence
        assert not bound.evidence


def test_direct_equivalent_growth_uses_windowed_classification(monkeypatch) -> None:
    call_start = 235_000
    ceiling = tm_module.static_runner.SECURITY_VIEW_WINDOW_CHARS
    common = "#" + "x" * (call_start - 9) + "\n"
    direct_prefix = common + "#xxxxx\n"
    bound_prefix = common + "a=True\n"
    call_prefix = 'subprocess.run("shell=True-'
    call_suffix = '", shell='
    filler_length = ceiling - call_start - len(call_prefix) - len(call_suffix) - len("a)")
    marker = " docker build "
    marker_offset = 250_000 - (call_start + len(call_prefix))
    filler = "x" * marker_offset + marker + "x" * (filler_length - marker_offset - len(marker))
    direct_source = direct_prefix + call_prefix + filler + call_suffix + "True)"
    bound_source = bound_prefix + call_prefix + filler + call_suffix + "a)"
    assert len(bound_source) == ceiling
    assert len(direct_source) == ceiling + 3

    for cap in (1, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        direct = _tm1(direct_source)[0]
        bound = _tm1(bound_source)[0]
        assert (
            (bound.severity, bound.confidence)
            == (
                direct.severity,
                direct.confidence,
            )
            == ("HIGH", pytest.approx(0.9))
        )
        assert bound.fingerprint() == direct.fingerprint()


def test_direct_equivalent_window_projects_shifted_trailing_context(monkeypatch) -> None:
    call_start = 239_500
    direct_assignment = "#" + "d" * 11 + "\n"
    bound_assignment = "enabled=True\n"
    prefix = "# pre "
    common = prefix + "x" * (call_start - len(direct_assignment) - len(prefix) - 1) + "\n"
    direct_prefix = common + direct_assignment
    bound_prefix = common + bound_assignment
    direct_call = "subprocess.run(command, shell=True)"
    bound_call = "subprocess.run(command, shell=enabled)"
    marker = "docker build"
    separator = " #"
    filler = "y" * (247_808 - len(direct_prefix) - len(direct_call) - len(separator) - len(marker))
    suffix = separator + filler + marker + "\n" + "z" * 20_000
    direct_source = direct_prefix + direct_call + suffix
    bound_source = bound_prefix + bound_call + suffix

    for cap in (1, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        direct = _tm1(direct_source)[0]
        bound = _tm1(bound_source)[0]
        assert (
            (bound.severity, bound.confidence)
            == (
                direct.severity,
                direct.confidence,
            )
            == ("LOW", pytest.approx(0.15))
        )
        assert bound.fingerprint() == direct.fingerprint()


def test_direct_equivalent_shrink_recovers_whole_artifact_identity(monkeypatch) -> None:
    call_start = 235_000
    direct_assignment = "#" + "x" * 11 + "\n"
    bound_assignment = "enabled=True\n"
    marker = "# docker build image "
    common = marker + "x" * (call_start - len(direct_assignment) - len(marker) - 1) + "\n"
    direct_prefix = common + direct_assignment
    bound_prefix = common + bound_assignment
    call_prefix = 'subprocess.run("shell=True-'
    call_suffix = '", shell='
    direct_tail = "True)"
    bound_tail = "enabled)"
    filler = "x" * (
        tm_module.static_runner.SECURITY_VIEW_WINDOW_CHARS
        - len(direct_prefix)
        - len(call_prefix)
        - len(call_suffix)
        - len(direct_tail)
    )
    direct_source = direct_prefix + call_prefix + filler + call_suffix + direct_tail
    bound_source = bound_prefix + call_prefix + filler + call_suffix + bound_tail
    assert len(direct_source) == tm_module.static_runner.SECURITY_VIEW_WINDOW_CHARS
    assert len(bound_source) == tm_module.static_runner.SECURITY_VIEW_WINDOW_CHARS + 3

    for cap in (1, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        direct = _tm1(direct_source)[0]
        bound = _tm1(bound_source)[0]
        assert bound.fingerprint() == direct.fingerprint()
        assert (bound.severity, bound.confidence) == (
            direct.severity,
            direct.confidence,
        )


def test_direct_equivalent_popen_keeps_legacy_ranked_qualifier(monkeypatch) -> None:
    call_start = tm_module.static_runner._RAW_WINDOW_OWNED_CHARS - len("subprocess.")
    prefix = "a=True\n#" + "x" * (call_start - len("a=True\n#") - 1) + "\n"
    call_prefix = 'subprocess.Popen("shell=True-'
    call_suffix = '", shell='
    filler = "x" * (
        tm_module.static_runner.SECURITY_VIEW_WINDOW_CHARS
        - len(prefix)
        - len(call_prefix)
        - len(call_suffix)
        - len("a)")
    )
    bound_source = prefix + call_prefix + filler + call_suffix + "a)"
    direct_source = prefix + call_prefix + filler + call_suffix + "True)"
    assert bound_source.index("Popen") == tm_module.static_runner._RAW_WINDOW_OWNED_CHARS

    for cap in (1, 2, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        direct = _tm1(direct_source)[0]
        bound = _tm1(bound_source)[0]
        assert bound.fingerprint() == direct.fingerprint()
        assert (bound.severity, bound.confidence) == (
            direct.severity,
            direct.confidence,
        )


def test_fixed_continuity_identity_is_bound_output_cap_invariant(monkeypatch) -> None:
    base = 'a=True\nsubprocess.run("shell=True",'
    separator = " " * 9_000
    right = len(base) + len(separator) + tm_module.static_runner._CONTINUITY_RIGHT_CONTEXT_CHARS
    value_start = right - 3
    tail_prefix = '"'
    tail_suffix = '", shell='
    tail = (
        tail_prefix
        + "x" * (value_start - len(base) - len(separator) - len(tail_prefix) - len(tail_suffix))
        + tail_suffix
    )
    bound_source = base + separator + tail + "a)\n#" + "z" * 10_000
    direct_source = base + separator + tail + "True)\n#" + "z" * 10_000
    direct = _tm1(direct_source)[0]

    bound_findings = []
    for cap in (1, 2, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        bound_findings.append(_tm1(bound_source)[0])

    assert all(finding.fingerprint() == direct.fingerprint() for finding in bound_findings)
    assert len({finding.fingerprint() for finding in bound_findings}) == 1


def test_fixed_normalized_identity_is_bound_output_cap_invariant(monkeypatch) -> None:
    base = 'a=True\nsubprocess.run("shell=True-'
    tail = '", shell='
    filler = "\ufdfa" * 14_219 + "x" * 16
    common = base + filler + tail
    assert len(normalized_security_view(common).text) == 256_001
    bound_source = common + "a)\n#" + "z" * 10_000
    direct_source = common + "True)\n#" + "z" * 10_000
    direct = _tm1(direct_source)[0]

    bound_findings = []
    for cap in (1, 2, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        bound_findings.append(_tm1(bound_source)[0])

    assert all(finding.fingerprint() == direct.fingerprint() for finding in bound_findings)
    assert len({finding.fingerprint() for finding in bound_findings}) == 1


def test_external_normalization_identity_is_bound_output_cap_invariant(monkeypatch) -> None:
    fixed = "a=True\n#"
    call = '\nsubprocess.run("shell=True", shell='
    target = 256_001
    count, remainder = divmod(
        target - len(normalized_security_view(fixed + call).text),
        18,
    )
    common = fixed + "\ufdfa" * count + "x" * remainder + call
    assert len(normalized_security_view(common).text) == target
    assert common[common.index("subprocess") :].isascii()
    bound_source = common + "a)\n#" + "z" * 10_000
    direct_source = common + "True)\n#" + "z" * 10_000
    direct = _tm1(direct_source)[0]

    for cap in (1, 2, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        bound = _tm1(bound_source)[0]
        assert bound.fingerprint() == direct.fingerprint()
        assert (bound.severity, bound.confidence) == (
            direct.severity,
            direct.confidence,
        )


def test_derived_lookalike_does_not_add_bound_only_normalized_tag(monkeypatch) -> None:
    bound_source = 'a=True\nsubprocess.run("ｓｈｅｌｌ=True）", shell=a)\n'
    direct_source = bound_source.replace("shell=a", "shell=True")

    for cap in (1, 2, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        direct = _tm1(direct_source)[0]
        bound = _tm1(bound_source)[0]
        assert bound.fingerprint() == direct.fingerprint()
        assert (bound.severity, bound.confidence) == (
            direct.severity,
            direct.confidence,
        )
        assert bound.tags == direct.tags == ["Tool Misuse"]


def test_long_bound_value_crossing_window_end_replays_shifted_context(monkeypatch) -> None:
    name = "a" * 20_000
    call_start = 235_000
    assignment = name + "=True\n"
    prefix = assignment + "#" + "x" * (call_start - len(assignment) - 2) + "\n"
    bound_call = f"subprocess.run(command, shell={name})"
    direct_call = "subprocess.run(command, shell=True)"
    suffix = " # docker build\n#" + "z" * 25_000
    bound_source = prefix + bound_call + suffix
    direct_source = prefix + direct_call + suffix

    for cap in (1, 2, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        direct = _tm1(direct_source)[0]
        bound = _tm1(bound_source)[0]
        assert (
            (bound.severity, bound.confidence)
            == (
                direct.severity,
                direct.confidence,
            )
            == ("LOW", pytest.approx(0.15))
        )
        assert bound.fingerprint() == direct.fingerprint()


def test_parenthesized_popen_replays_alternate_owner_window(monkeypatch) -> None:
    call_start = 239_610
    marker_start = 487_409
    prefix = "#" + "x" * (call_start - 9) + "\n" + "a=True\n"
    call_prefix = '(subprocess).Popen("'
    call_suffix = '", shell='
    filler = "x" * (marker_start - call_start - len(call_prefix) - len(call_suffix) - len("a)"))
    marker = " # docker build "
    suffix = "\n#" + "z" * 20_000 + "\n"
    bound_source = prefix + call_prefix + filler + call_suffix + "a)" + marker + suffix
    value_start = call_start + len(call_prefix) + len(filler) + len(call_suffix)
    direct_source = bound_source[:value_start] + "True" + bound_source[value_start + 1 :]
    assert bound_source.index("Popen") == call_start + len("(subprocess).")
    assert bound_source.index("docker build") + len("docker build") == 487_424

    for cap in (1, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        direct = _tm1(direct_source)[0]
        bound = _tm1(bound_source)[0]
        assert (
            (bound.severity, bound.confidence)
            == (
                direct.severity,
                direct.confidence,
            )
            == ("HIGH", pytest.approx(0.9))
        )
        assert bound.fingerprint() == direct.fingerprint()


def test_same_line_bound_metadata_is_output_cap_independent(monkeypatch) -> None:
    call_start = 235_000
    marker_start = 250_000
    source_length = 255_996
    prefix = "#" + "x" * (call_start - 9) + "\n" + "a=True\n"
    call = 'subprocess.run("' + "x" * 300 + '",shell=a)'
    head = call + ";" + call + " #"
    marker = " docker build "
    padding = "x" * (marker_start - call_start - len(head))
    bound_source = (
        prefix
        + head
        + padding
        + marker
        + "x" * (source_length - call_start - len(head) - len(padding) - len(marker))
    )
    direct_source = bound_source.replace("shell=a", "shell=True")
    assert len(bound_source) == source_length

    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 10)
    direct = _tm1(direct_source)[0]
    assert (direct.severity, direct.confidence) == ("HIGH", pytest.approx(0.9))

    for cap in (1, 2, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        bound_findings = _tm1(bound_source)
        assert len(bound_findings) == 1
        assert (bound_findings[0].severity, bound_findings[0].confidence) == (
            direct.severity,
            direct.confidence,
        )
        assert bound_findings[0].fingerprint() == direct.fingerprint()


def test_normalized_slice_prefers_later_complete_shell_anchor(monkeypatch) -> None:
    target = 247_808
    fixed = "\nenabled = True\n"
    count, remainder = divmod(target - len(fixed) - 1, 18)
    prefix = "#" + "\ufdfa" * count + "x" * remainder + fixed
    assert len(normalized_security_view(prefix).text) == target
    callee = "ｓｕｂｐｒｏｃｅｓｓ.run"
    gap = " " * 8150
    suffix = "#" + "z" * 20_000 + "\n"
    template = prefix + callee + "(command, note='shell=True'," + gap
    direct_source = template + "shell=True)\n" + suffix
    bound_source = template + "shell=enabled)\n" + suffix
    direct_findings = []
    bound_findings = []

    for cap in (1, 2, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        direct_findings.append(_tm1(direct_source)[0])
        bound_findings.append(_tm1(bound_source)[0])

    for direct, bound in zip(direct_findings, bound_findings, strict=True):
        assert bound.fingerprint() == direct.fingerprint()
        assert bound.tags == direct.tags
        assert (bound.severity, bound.confidence) == (
            direct.severity,
            direct.confidence,
        )
    assert len({finding.fingerprint() for finding in direct_findings}) == 1
    assert len({finding.fingerprint() for finding in bound_findings}) == 1
    assert all(not finding.evidence for finding in [*direct_findings, *bound_findings])


def test_continued_qualified_popen_keeps_direct_bound_fingerprint_parity() -> None:
    findings = _tm1(
        "enabled = True\n"
        'subprocess.\\\nPopen("command", shell=True)\n'
        'subprocess.\\\nPopen("command", shell=enabled)\n'
    )

    compacted = deduplicate(findings)

    assert len(findings) == 2
    assert (findings[0].matched_text or "").startswith("Popen(")
    assert len({finding.fingerprint() for finding in findings}) == 1
    assert len(compacted) == 1
    assert len(compacted[0].occurrences) == 2


def test_cross_window_qualified_and_bare_popen_share_one_budget_owner(monkeypatch) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    content = "subprocess." + "\u200b" * 256_000 + "Popen(command, shell=True)\n"
    result = _run(content, "guide.md")
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert len(findings) == 1
    assert findings[0].matched_text == "subprocess.Popen(command, shell=True"
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize("ignored", ["\u200b", "\ufffd"])
def test_cross_window_popen_fingerprint_is_output_cap_invariant(monkeypatch, ignored: str) -> None:
    expected = _tm1("subprocess.Popen(command, shell=True)\n", "guide.md")[0].fingerprint()
    content = "subprocess." + ignored * 256_000 + "Popen(command, shell=True)\n"

    for cap in (1, 2, 3):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        findings = _tm1(content, "guide.md")

        assert len(findings) == 1
        assert findings[0].fingerprint() == expected


@pytest.mark.parametrize("ignored", ["\u200b", "\ufffd", "\x85"])
@pytest.mark.parametrize(("cap", "later_calls"), [(1, 1), (2, 2)])
def test_output_limit_finalizes_retained_cross_window_popen(
    monkeypatch, cap: int, later_calls: int, ignored: str
) -> None:
    expected = _tm1("subprocess.Popen(command, shell=True)\n", "guide.md")[0].fingerprint()
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
    content = (
        "subprocess."
        + ignored * 256_000
        + "Popen(command, shell=True)\n"
        + "".join(f"subprocess.run(command_{index}, shell=True)\n" for index in range(later_calls))
    )

    result = _run(content, "guide.md")
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert len(findings) == cap
    assert findings[0].matched_text == "subprocess.Popen(command, shell=True"
    assert findings[0].fingerprint() == expected
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.OUTPUT_LIMIT


@pytest.mark.parametrize("ignored", ["\u200b", "\ufffd", "\x85"])
def test_cross_window_popen_public_context_is_output_cap_invariant(
    monkeypatch, ignored: str
) -> None:
    content = (
        "subprocess."
        + ignored * 256_000
        + "Popen(command, shell=True)\n"
        + "subprocess.run(other, shell=True)\n"
    )

    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    limited = _tm1(content, "guide.md")[0]
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 2)
    complete = _tm1(content, "guide.md")[0]

    assert limited.matched_text == complete.matched_text
    assert limited.fingerprint() == complete.fingerprint()
    assert limited.tags == complete.tags
    assert limited.context == complete.context
    assert limited.code_snippet == complete.code_snippet
    assert limited.finding == complete.finding
    assert (limited.severity, limited.confidence) == (complete.severity, complete.confidence)
    assert limited.context == complete.context
    assert limited.code_snippet == complete.code_snippet
    assert limited.finding == complete.finding


@pytest.mark.parametrize("middle", ["\u2003" * 2000, "\ufffd" * 10 + "\u2003" * 1990])
def test_continuity_truncation_preserves_unicode_whitespace_boundary(middle: str) -> None:
    separator = "\u200b" * 4096 + middle + "\u200b" * 4096
    expected = _tm1("Popen(command, shell=True)\n", "guide.md")[0]

    finding = _tm1(
        "subprocess." + separator + "Popen(command, shell=True)\n",
        "guide.md",
    )[0]

    assert finding.matched_text == expected.matched_text
    assert finding.fingerprint() == expected.fingerprint()
    assert finding.tags == expected.tags


@pytest.mark.parametrize("cap", [1, 2])
def test_long_cross_window_popen_call_keeps_qualified_identity(monkeypatch, cap: int) -> None:
    argument = "a" * 2048
    baseline = _tm1(f'subprocess.Popen("{argument}", shell=True)\n', "guide.md")[0]
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
    content = (
        "subprocess."
        + "\u200b" * 256_000
        + f'Popen("{argument}", shell=True)\n'
        + "subprocess.run(other, shell=True)\n"
    )

    finding = _tm1(content, "guide.md")[0]

    assert (finding.matched_text or "").startswith("subprocess.Popen(")
    assert finding.fingerprint() == baseline.fingerprint()


def test_chained_continuity_context_is_output_cap_invariant(monkeypatch) -> None:
    separator = "\u200b" * 256_000
    content = (
        "subprocess."
        + separator
        + "Popen(a, shell=True)\n"
        + "subprocess."
        + separator
        + "Popen(b, shell=True) # docker build image\n"
    )

    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    limited = _tm1(content, "guide.md")[0]
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 2)
    complete = _tm1(content, "guide.md")[0]

    assert limited.severity == complete.severity
    assert limited.confidence == complete.confidence
    assert limited.matched_text == complete.matched_text
    assert limited.fingerprint() == complete.fingerprint()
    assert limited.tags == complete.tags
    assert limited.context == complete.context
    assert limited.code_snippet == complete.code_snippet
    assert limited.finding == complete.finding


@pytest.mark.parametrize(
    "first_call",
    [
        pytest.param(
            "sub"
            + "\u200b" * 9000
            + "process."
            + "\u200b" * 256_000
            + "Popen(command, shell=True)",
            id="multiple-long-default-ignorable-runs",
        ),
        pytest.param(
            "sub"
            + "\u200b" * 260_000
            + "p"
            + "-" * 2038
            + "r-o-c-e-s"
            + "\u200b" * 9000
            + "s.Popen(command, shell=True)",
            id="chain-gap-at-inclusive-limit",
        ),
        pytest.param(
            "subpro" + "\u2003" * 9000 + "cess.Popen(command, shell=True)",
            id="long-internal-compactable-gap",
        ),
        pytest.param(
            "sub" + "\u200b" * 260_000 + "process.Popen(command, shell=True)",
            id="long-internal-default-ignorable-gap",
        ),
        pytest.param(
            "s-u-b-p-r-o-c-e-s-s." + "\u200b" * 256_000 + "Popen(command, shell=True)",
            id="letter-spacing-prefix-before-long-run",
        ),
        pytest.param(
            ("-" * 500).join("subprocess") + "." + "\u200b" * 63 + "Popen(command, shell=True)",
            id="letter-spacing-prefix-with-short-tail",
        ),
        pytest.param(
            "sub"
            + "\u200b" * 260_000
            + ("-" * 400).join("process")
            + ".Popen(command, shell=True)",
            id="letter-spacing-tail-after-long-run",
        ),
        pytest.param(
            ("-" * 100).join("subprocess") + ".Popen(a-a-a-a-a-a, shell=True)",
            id="letter-spacing-prefix-and-argument",
        ),
        pytest.param(
            "s-u-b-p-r-o-c-e-s-s." + "\u200b" * 63 + 'Popen("a-a-a-a-a-a", shell=True)',
            id="letter-spacing-prefix-short-tail-and-argument",
        ),
    ],
)
def test_retained_popen_qualifier_is_output_cap_invariant(monkeypatch, first_call: str) -> None:
    content = first_call + "\nsubprocess.run(other, shell=True)\n"

    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    limited = _tm1(content, "guide.md")[0]
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 10)
    complete = _tm1(content, "guide.md")[0]

    assert (limited.matched_text or "").startswith("subprocess.Popen(")
    assert limited.matched_text == complete.matched_text
    assert limited.fingerprint() == complete.fingerprint()
    assert limited.tags == complete.tags
    assert limited.context == complete.context
    assert limited.code_snippet == complete.code_snippet
    assert limited.finding == complete.finding
    assert (limited.severity, limited.confidence) == (complete.severity, complete.confidence)


def test_output_limit_finalizer_does_not_outreach_continuity_producer(monkeypatch) -> None:
    suffix = ("-" * 1000).join("process") + "."
    content = (
        "sub"
        + "\u200b" * 260_000
        + suffix
        + 'Popen("'
        + "a" * 2500
        + '", shell=True)\n'
        + "subprocess.run(other, shell=True)\n"
    )

    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    limited = _tm1(content, "guide.md")[0]
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 10)
    complete = _tm1(content, "guide.md")[0]

    assert (complete.matched_text or "").startswith("Popen(")
    assert limited.matched_text == complete.matched_text
    assert limited.fingerprint() == complete.fingerprint()
    assert limited.tags == complete.tags
    assert limited.context == complete.context
    assert limited.code_snippet == complete.code_snippet
    assert limited.finding == complete.finding
    assert (limited.severity, limited.confidence) == (complete.severity, complete.confidence)


def test_in_call_continuity_identity_is_output_cap_invariant(monkeypatch) -> None:
    owned = tm_module.static_runner._RAW_WINDOW_OWNED_CHARS
    qualifier_start = owned - 6
    content = (
        "x" * qualifier_start
        + "subprocess.Popen("
        + "\u200b" * 9000
        + "command, shell=True)\n"
        + "z" * 20_000
        + "\nsubprocess.run(other, shell=True)\n"
    )
    first_findings = []

    for cap in (1, 2, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        first_findings.append(_tm1(content, "guide.md")[0])

    assert all(
        finding.matched_text == "subprocess.Popen(command, shell=True" for finding in first_findings
    )
    assert len({finding.fingerprint() for finding in first_findings}) == 1
    assert all(finding.tags == first_findings[0].tags for finding in first_findings)
    assert all(finding.context == first_findings[0].context for finding in first_findings)
    assert all(finding.code_snippet == first_findings[0].code_snippet for finding in first_findings)
    assert all(finding.finding == first_findings[0].finding for finding in first_findings)
    assert all(
        (finding.severity, finding.confidence)
        == (first_findings[0].severity, first_findings[0].confidence)
        for finding in first_findings
    )


def test_retained_owner_keeps_continuity_identity_across_coalesce_passes(monkeypatch) -> None:
    owned = tm_module.static_runner._RAW_WINDOW_OWNED_CHARS
    target = owned - 32
    prefix = "enabled = True\n#" + "x" * (target - len("enabled = True\n#") - 1) + "\n"
    gap = " " * (tm_module.static_runner._WINDOW_OVERLAP_CHARS + 1)
    suffix = "#" + "z" * 20_000 + "\n"
    template = prefix + "subprocess.run(command, note='shell=True'," + gap
    direct = template + "shell=True)\n" + suffix
    bound = template + "shell=enabled)\n" + suffix
    direct_findings = []
    bound_findings = []

    for cap in (1, 2, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        direct_findings.append(_tm1(direct)[0])
        bound_findings.append(_tm1(bound)[0])

    for direct_finding, bound_finding in zip(
        direct_findings,
        bound_findings,
        strict=True,
    ):
        assert bound_finding.fingerprint() == direct_finding.fingerprint()
        assert bound_finding.tags == direct_finding.tags
        assert (bound_finding.severity, bound_finding.confidence) == (
            direct_finding.severity,
            direct_finding.confidence,
        )
    assert len({finding.fingerprint() for finding in direct_findings}) == 1
    assert len({finding.fingerprint() for finding in bound_findings}) == 1
    assert all(not finding.evidence for finding in [*direct_findings, *bound_findings])


@pytest.mark.parametrize("tail_length", [8180, 8181])
def test_normalized_window_owner_is_output_cap_invariant(monkeypatch, tail_length: int) -> None:
    owned = tm_module.static_runner._RAW_WINDOW_OWNED_CHARS
    content = (
        "x" * (owned - 6)
        + "subprocess.Popen("
        + "\u200b" * 9000
        + "a" * tail_length
        + ", shell=True)\n"
        + "z" * 20_000
        + "\nsubprocess.run(other, shell=True)\n"
    )
    first_findings = []

    for cap in (1, 2, 10):
        monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", cap)
        first_findings.append(_tm1(content, "guide.md")[0])

    assert all(
        (finding.matched_text or "").startswith("subprocess.Popen(") for finding in first_findings
    )
    assert all(finding.matched_text == first_findings[0].matched_text for finding in first_findings)
    assert len({finding.fingerprint() for finding in first_findings}) == 1
    assert all(finding.tags == first_findings[0].tags for finding in first_findings)
    assert all(finding.context == first_findings[0].context for finding in first_findings)
    assert all(finding.code_snippet == first_findings[0].code_snippet for finding in first_findings)
    assert all(finding.finding == first_findings[0].finding for finding in first_findings)
    assert all(
        (finding.severity, finding.confidence)
        == (first_findings[0].severity, first_findings[0].confidence)
        for finding in first_findings
    )


def test_output_limit_reuses_one_shared_continuity_projection(monkeypatch) -> None:
    projection_calls = 0
    original = tm_module.static_runner._anchored_continuity_view

    def count_projection(content, anchor, check_runtime):
        nonlocal projection_calls
        projection_calls += 1
        return original(content, anchor, check_runtime)

    monkeypatch.setattr(
        tm_module.static_runner,
        "_anchored_continuity_view",
        count_projection,
    )
    count = 20
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", count)
    content = (
        "subprocess."
        + "\u200b" * 260_000
        + "Popen(0, shell=True)\n"
        + "".join(f"Popen({index}, shell=True)\n" for index in range(1, count))
        + "subprocess.run(extra, shell=True)\n"
    )

    result = _run(content, "guide.md")
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert len(findings) == count
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.OUTPUT_LIMIT
    assert projection_calls == 1


def test_output_limit_batches_negative_continuity_searches(monkeypatch) -> None:
    projection_calls = 0
    original = tm_module.static_runner._anchored_continuity_view

    def count_projection(content, anchor, check_runtime):
        nonlocal projection_calls
        projection_calls += 1
        return original(content, anchor, check_runtime)

    monkeypatch.setattr(
        tm_module.static_runner,
        "_anchored_continuity_view",
        count_projection,
    )
    count = 200
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", count)
    content = (
        "".join(f"Popen({index}, shell=True)\n" for index in range(count))
        + "subprocess.run(extra, shell=True)\n"
    )

    result = _run(content, "guide.md")
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]

    assert len(findings) == count
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.OUTPUT_LIMIT
    assert projection_calls == 0


@pytest.mark.parametrize(
    ("argument_length", "qualified"),
    [
        pytest.param(8169, True, id="qualified-fits-prefixed-slice"),
        pytest.param(8170, False, id="qualified-exceeds-prefixed-slice"),
    ],
)
def test_output_limit_replays_markdown_window_prefix(
    monkeypatch, argument_length: int, qualified: bool
) -> None:
    opening = "```python\n"
    qualifier_start = 479_220
    content = (
        opening
        + "x" * (qualifier_start - len(opening))
        + 'subprocess.Popen("'
        + "a" * argument_length
        + '", shell=True)\n'
        + "subprocess.run(other, shell=True)\n"
    )

    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    limited = _tm1(content, "guide.md")[0]
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 2)
    complete = _tm1(content, "guide.md")[0]

    assert (complete.matched_text or "").startswith("subprocess.Popen(" if qualified else "Popen(")
    assert limited.matched_text == complete.matched_text
    assert limited.fingerprint() == complete.fingerprint()
    assert limited.tags == complete.tags
    assert limited.context == complete.context
    assert limited.code_snippet == complete.code_snippet
    assert limited.finding == complete.finding
    assert (limited.severity, limited.confidence) == (complete.severity, complete.confidence)


def test_retained_popen_qualifier_preserves_buried_ascii_whitespace(monkeypatch) -> None:
    internal_gap = "\u200b" * 5000 + "\u00a0 \u200b" + "\u200b" * 5000
    content = (
        "sub"
        + internal_gap
        + "process."
        + "\u200b" * 256_000
        + "Popen(command, shell=True)\n"
        + "subprocess.run(other, shell=True)\n"
    )

    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    limited = _tm1(content, "guide.md")[0]
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 10)
    complete = _tm1(content, "guide.md")[0]

    assert (limited.matched_text or "").startswith("Popen(")
    assert limited.matched_text == complete.matched_text
    assert limited.fingerprint() == complete.fingerprint()
    assert limited.tags == complete.tags


@pytest.mark.parametrize("separator", ["\v", "\f", "\x1c", "\x1d", "\x1e", "\x85"])
def test_retained_popen_qualifier_preserves_removable_line_controls(
    monkeypatch, separator: str
) -> None:
    content = (
        "subprocess."
        + separator * 9000
        + "Popen(command, shell=True)\n"
        + "subprocess.run(other, shell=True)\n"
    )

    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    limited = _tm1(content, "guide.md")[0]
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 2)
    complete = _tm1(content, "guide.md")[0]

    assert (limited.matched_text or "").startswith("subprocess.Popen(")
    assert limited.matched_text == complete.matched_text
    assert limited.fingerprint() == complete.fingerprint()
    assert limited.tags == complete.tags


def test_output_limit_reconciliation_never_projects_a_whole_expanding_prefix(
    monkeypatch,
) -> None:
    observed_lengths: list[int] = []
    original = tm_module.security_text_views

    def record_projection_length(content: str):
        observed_lengths.append(len(content))
        return original(content)

    monkeypatch.setattr(tm_module, "security_text_views", record_projection_length)
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    content = (
        "ﷺ" * 300_000
        + "\u200b" * 64
        + "Popen(command, shell=True)\n"
        + "subprocess.run(other, shell=True)\n"
    )

    result = _run(content, "guide.md")

    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.OUTPUT_LIMIT
    assert max(observed_lengths, default=0) <= tm_module.static_runner.SECURITY_VIEW_WINDOW_CHARS


def test_output_limit_continuity_reconciliation_honors_runtime_deadline(monkeypatch) -> None:
    now = [0.0]
    monkeypatch.setattr(tm_module.static_runner.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    original = tm_module.reconcile_retained_findings

    def expire_before_reconciliation(content, findings, check_runtime, source_context):
        now[0] = 31.0
        return original(content, findings, check_runtime, source_context)

    monkeypatch.setattr(
        tm_module,
        "reconcile_retained_findings",
        expire_before_reconciliation,
    )
    content = (
        "subprocess."
        + "\u200b" * 256_000
        + "Popen(command, shell=True)\n"
        + "subprocess.run(other, shell=True)\n"
    )

    result = _run(content, "guide.md")
    event = result["inspection_ledger"][0]

    assert len(result["findings"]) == 1
    assert event["outcome"] is LedgerOutcome.PARTIAL
    assert event["reason_code"] is LedgerReason.RUNTIME_LIMIT


def test_output_limit_does_not_scan_unrelated_continuity_runs(monkeypatch) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)

    def unexpected_continuity_scan(*_args, **_kwargs):
        raise AssertionError(
            "output-limit identity finalization must not discover continuity views"
        )

    monkeypatch.setattr(
        tm_module.static_runner,
        "_continuity_views",
        unexpected_continuity_scan,
    )
    content = (
        ("x" + " " * 9000) * 5
        + "Popen(command, shell=True)\n"
        + "subprocess.run(other, shell=True)\n"
    )

    result = _run(content, "guide.md")

    assert len(result["findings"]) == 1
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.OUTPUT_LIMIT


@pytest.mark.parametrize("path", ["guide.md", "run.js"])
def test_non_python_lexical_fallback_remains_case_insensitive(path: str) -> None:
    assert len(_tm1("Popen(command, shell=true)", path)) == 1


def test_qualified_popen_legacy_lexical_fallback_is_preserved() -> None:
    findings = _tm1("pkg.Popen(command, shell=True)", "run.py")

    assert len(findings) == 1


def test_malformed_python_retains_direct_literal_fallback() -> None:
    assert len(_tm1("subprocess.run(command, shell=True", "broken.py")) == 1


@pytest.mark.parametrize(
    ("shell_value", "expected_findings"),
    [("enabled", 0), ("True", 1)],
)
def test_malformed_python_marks_ast_companion_work_partial(
    shell_value: str,
    expected_findings: int,
) -> None:
    result = _run(
        f"enabled = True\nsubprocess.run(command, shell={shell_value})\nbad=(\n",
        "broken.py",
    )
    event = result["inspection_ledger"][0]

    assert len([finding for finding in result["findings"] if finding.rule_id == "TM1"]) == (
        expected_findings
    )
    assert event["outcome"] is LedgerOutcome.PARTIAL
    assert event["reason_code"] is LedgerReason.SYNTAX_ERROR


def test_malformed_python_does_not_defer_lexical_output_limit(
    monkeypatch,
) -> None:
    monkeypatch.setattr(tm_module.static_runner, "MAX_FINDINGS_PER_ARTIFACT", 1)
    monkeypatch.setattr(
        tm_module.static_runner,
        "_scan_declared_marker_views",
        lambda *_args, **_kwargs: ([], False, None),
    )
    scanned_views: list[str] = []
    original = tm_module.static_runner._scan_view_windows

    def record_view(
        path,
        view,
        pattern_modules,
        finding_budget,
        python_ast_cache_key,
        *,
        python_source,
    ):
        scanned_views.append(view.text)
        return original(
            path,
            view,
            pattern_modules,
            finding_budget,
            python_ast_cache_key,
            python_source=python_source,
        )

    monkeypatch.setattr(tm_module.static_runner, "_scan_view_windows", record_view)
    late_sentinel = "LATE_RAW_WINDOW_SENTINEL"
    content = (
        "if (\n"
        "subprocess.run(a, shell=True)\n"
        "subprocess.run(b, shell=True)\n"
        + "x" * tm_module.static_runner.SECURITY_VIEW_WINDOW_CHARS
        + late_sentinel
    )

    result = _run(content, "bad.py")
    event = result["inspection_ledger"][0]

    assert len(result["findings"]) == 1
    assert event["reason_code"] is LedgerReason.OUTPUT_LIMIT
    assert not any(late_sentinel in view for view in scanned_views)


def test_direct_popen_boundary_classification_is_ast_parse_invariant() -> None:
    prefix = "#" + "x" * 239_608 + "\n"
    call = 'subprocess.Popen("' + "a" * 9_000 + '", shell=True) # docker build image\n'
    suffix = "#" + "z" * 10_000 + "\n"

    parsed = _tm1(prefix + call + suffix, "run.py")[0]
    malformed = _tm1(prefix + call + suffix + "invalid = (\n", "broken.py")[0]
    lexical = _tm1(prefix + call + suffix, "guide.md")[0]

    assert (parsed.severity, parsed.confidence) == (lexical.severity, lexical.confidence)
    assert (malformed.severity, malformed.confidence) == (
        lexical.severity,
        lexical.confidence,
    )


def test_large_flat_module_completes_and_retains_finding() -> None:
    content = "\n".join(
        [
            "import subprocess",
            "enabled = True",
            *(f"unrelated_{index} = {index}" for index in range(10_000)),
            "result = subprocess.run(command, shell=enabled)",
        ]
    )
    result = _run(content, "large.py")

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert sum(finding.rule_id == "TM1" for finding in result["findings"]) == 1


def test_many_same_line_calls_keep_classification_context_bounded(monkeypatch) -> None:
    calls: Counter[str] = Counter()

    def record(name: str):
        def predicate(*_args: str) -> bool:
            calls[name] += 1
            return False

        return predicate

    monkeypatch.setattr(tm_module, "_is_safe_container_command", record("container"))
    monkeypatch.setattr(tm_module, "_is_safe_dockerfile_idiom", record("dockerfile"))
    monkeypatch.setattr(tm_module, "_is_safe_cache_cleanup", record("cache"))
    content = "".join(
        f"true_value=True;subprocess.run({index},shell=true_value);import subprocess;"
        for index in range(50)
    )

    findings = _tm1(content)

    assert len(findings) == 50
    assert max(len(finding.context or "") for finding in findings) <= 1024
    assert calls == {"container": 1, "dockerfile": 1, "cache": 1}


def test_many_bare_popen_short_ignored_prefixes_scale_linearly(monkeypatch) -> None:
    projected_characters = 0
    original = tm_module.security_text_views

    def record_projection(content: str):
        nonlocal projected_characters
        projected_characters += len(content)
        return original(content)

    monkeypatch.setattr(tm_module, "security_text_views", record_projection)
    content = "".join("\u200b" * 64 + f"Popen({index}, shell=True)\n" for index in range(500))

    result = _run(content, "guide.md")

    assert len([finding for finding in result["findings"] if finding.rule_id == "TM1"]) == 500
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert projected_characters <= 4 * len(content)
