# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared Python AST parsing utility."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import skillspector.python_ast as python_ast
from skillspector.python_ast import (
    PythonSourceClassification,
    build_python_ast_cache,
    classify_python_source,
    clear_python_ast_cache,
    decode_python_source,
    get_python_ast,
    is_python_source,
    may_be_python_source,
    parse_python_source,
    prewarm_python_ast_cache,
)


@pytest.mark.parametrize(
    ("path", "content"),
    [
        pytest.param("run.py", "pass\n", id="py"),
        pytest.param("run.PY", "pass\n", id="uppercase-py"),
        pytest.param("run.pyw", "pass\n", id="pyw"),
        pytest.param("run.PYW", "pass\n", id="uppercase-pyw"),
        pytest.param("runner", "#!/usr/bin/python3\npass\n", id="direct-python"),
        pytest.param(
            "runner",
            "#!/usr/bin/python3.14t\npass\n",
            id="direct-free-threaded-python",
        ),
        pytest.param("runner", "#! /usr/local/bin/python3.12 -B\npass\n", id="direct-option"),
        pytest.param(
            "runner",
            "#!/usr/bin/python3 -t\npass\n",
            id="direct-legacy-tab-compatibility-option",
        ),
        pytest.param("runner", "#!/usr/bin/pypy3\npass\n", id="direct-pypy"),
        pytest.param("runner", "#!/usr/bin/env python3\npass\n", id="env-python"),
        pytest.param("runner", "#!/bin/env python3\npass\n", id="bin-env-python"),
        pytest.param(
            "runner",
            b"#!/usr/bin/env python3\0 node\npass\n",
            id="nul-terminated-env-python-bytes",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S python3\0 node\npass\n",
            id="nul-terminated-env-split-python",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env python3.13t\npass\n",
            id="env-free-threaded-python",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S python3 -B\npass\n",
            id="env-split-string",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S python3 -tt\npass\n",
            id="env-legacy-tab-compatibility-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S python3 -tEB\npass\n",
            id="env-clustered-legacy-tab-compatibility-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -Spython3 -B\npass\n",
            id="env-attached-split-string",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S 'python3' -B\npass\n",
            id="env-single-quoted-interpreter",
        ),
        pytest.param(
            "runner",
            '#!/usr/bin/env -S "python3" -B\npass\n',
            id="env-double-quoted-interpreter",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -i python3\npass\n",
            id="env-split-ignore-environment-option",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S PYTHONSAFEPATH=1 python3\npass\n",
            id="env-split-assignment",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -- python3\npass\n",
            id="env-split-option-terminator",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -- PYTHONSAFEPATH=1 python3\npass\n",
            id="env-split-assignment-after-option-terminator",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -u PYTHONPATH python3\npass\n",
            id="env-split-separate-option-operand",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -vS python3\npass\n",
            id="env-verbose-split-string",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -iS python3\npass\n",
            id="env-ignore-split-string",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -ivS python3\npass\n",
            id="env-clustered-outer-split-string",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S FOO=café python3\npass\n",
            id="env-split-unicode-assignment",
        ),
        pytest.param(
            "runner",
            b"#!/usr/bin/env -S FOO=caf\xc3\xa9 python3\npass\n",
            id="env-split-unicode-assignment-bytes",
        ),
        pytest.param(
            "runner",
            b"#!/usr/bin/env -S FOO=bar\fpython3\npass\n",
            id="env-split-form-feed-separator",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S python3\_-B" "\npass\n",
            id="env-escaped-argument-separator",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -Spython3\npass\n",
            id="nested-env-split-string",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/python3"
            + " " * (python_ast.MAX_PYTHON_SHEBANG_CHARS - len("#!/usr/bin/python3"))
            + "\npass\n",
            id="maximum-length-shebang",
        ),
        pytest.param(
            "bundle.zip!/runner",
            b"#!/usr/bin/env -S python3\r\npass\n",
            id="nested-env-split-bytes-crlf",
        ),
        pytest.param(
            "typing.pyi",
            "#!/usr/bin/python3\npass\n",
            id="pyi-with-execution-intent",
        ),
    ],
)
def test_is_python_source_accepts_supported_execution_surfaces(
    path: str, content: str | bytes
) -> None:
    assert is_python_source(path, content)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(
            "#!/usr/local/bin/python3.14-intel64\npass\n",
            id="python-org-versioned-intel64",
        ),
        pytest.param(
            "#!/usr/bin/env python3-intel64\npass\n",
            id="python-org-env-intel64",
        ),
        pytest.param(
            "#!/usr/bin/env -S python3.14t-intel64 -B\npass\n",
            id="python-org-free-threaded-intel64",
        ),
        pytest.param(
            "#!/usr/local/bin/python3.12-32\npass\n",
            id="python-org-legacy-32-bit",
        ),
        pytest.param(
            "#!/usr/bin/python3.11d\npass\n",
            id="cpython-debug-abi-flag",
        ),
        pytest.param(
            "#!/usr/bin/env python3.11-dbg\npass\n",
            id="debian-debug-interpreter",
        ),
        pytest.param(
            "#!/usr/bin/env python3.7m\npass\n",
            id="legacy-pymalloc-abi-flag",
        ),
        pytest.param(
            "#!/usr/bin/env -S python3.2dmu -B\npass\n",
            id="legacy-debug-pymalloc-unicode-abi-flags",
        ),
    ],
)
def test_python_org_macos_interpreter_aliases_are_python(content: str) -> None:
    assert classify_python_source("runner", content) is PythonSourceClassification.PYTHON


@pytest.mark.parametrize(
    "interpreter",
    [
        pytest.param("python3.14-config", id="config-tool"),
        pytest.param("python3.14-intel640", id="invalid-intel-suffix"),
        pytest.param("python3.14-arm64", id="unsupported-arm-suffix"),
        pytest.param("python3.11-debug", id="invalid-debug-suffix"),
        pytest.param("python3.7md", id="misordered-legacy-abi-flags"),
        pytest.param("python3.2dmm", id="repeated-legacy-abi-flag"),
    ],
)
def test_python_like_macos_tools_are_not_interpreters(interpreter: str) -> None:
    assert (
        classify_python_source("runner", f"#!/usr/local/bin/{interpreter}\npass\n")
        is PythonSourceClassification.NON_PYTHON
    )


@pytest.mark.parametrize(
    ("path", "content"),
    [
        pytest.param("typing.pyi", "value: int\n", id="pyi-without-shebang"),
        pytest.param("runner", None, id="missing-content"),
        pytest.param("runner", "python3 is installed\n", id="prose"),
        pytest.param("runner", "#!/usr/bin/env node\n", id="env-node"),
        pytest.param("runner", "#!/usr/bin/env node python3\n", id="deceptive-env"),
        pytest.param(
            "runner",
            "#!/usr/bin/env -L default python3\n",
            id="darwin-rejects-freebsd-login-class-option",
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
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -u FOO=BAR python3\n",
            id="env-split-invalid-short-unset-name",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S --unset '' python3\n",
            id="env-split-empty-long-unset-name",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S --unset= python3\n",
            id="env-split-empty-attached-long-unset-name",
        ),
        pytest.param("runner", "#!/usr/bin/env -S node python3\n", id="deceptive-env-s"),
        pytest.param("runner", "#!/usr/bin/env -Snode python3\n", id="deceptive-attached-s"),
        pytest.param(
            "runner",
            "#!/usr/bin/env python3 node\n",
            id="non-split-extra-argument",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S --ignore python3\n",
            id="ambiguous-long-option-abbreviation",
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
            "#!/usr/bin/env -S PYTHONSAFEPATH=1 -i python3\n",
            id="env-option-after-assignment",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S node ${SKILLSPECTOR_ARGUMENT}" "\n",
            id="non-python-with-dynamic-argument",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S /opt/${SKILLSPECTOR_ROOT}/node" "\n",
            id="fixed-node-basename-after-dynamic-path",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/env -S - node python3\n",
            id="lone-dash-still-selects-node",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S pyth\on3" "\n",
            id="invalid-env-escape",
        ),
        pytest.param(
            "runner",
            r"#!/usr/bin/env -S python3\ -I" "\n",
            id="freebsd-escaped-space-inside-utility",
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
        pytest.param(
            "runner",
            "#!/usr/bin/env -S -P\n",
            id="missing-env-option-operand",
        ),
        pytest.param("runner", "#!/bin/sh python3\n", id="shell-with-python-argument"),
        pytest.param(
            "runner",
            b"#!/usr/bin/node\0/usr/bin/python3\npass\n",
            id="nul-terminated-node-before-python-bytes",
        ),
        pytest.param(
            "runner",
            "#!/usr/bin/node\0/usr/bin/python3\npass\n",
            id="nul-terminated-node-before-python-text",
        ),
        pytest.param(
            "bundle.zip!/runner",
            b"#!/usr/bin/env python3\r\npass\n",
            id="plain-env-bytes-crlf",
        ),
        pytest.param("runner", "#!/tmp/env python3\n", id="untrusted-env-path"),
        pytest.param("runner", " #!/usr/bin/env python3\n", id="leading-space"),
        pytest.param("runner", "\ufeff#!/usr/bin/env python3\n", id="leading-bom"),
        pytest.param("runner", "#!/usr/bin/env 'python3'\n", id="quoted-interpreter"),
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
        pytest.param("runner", "#!/usr/bin/env pyth\u03bfn3\n", id="unicode-confusable"),
        pytest.param("runner", "#!/usr/bin/env -S python\u0663\n", id="unicode-version-digit"),
    ],
)
def test_is_python_source_rejects_non_python_execution_surfaces(
    path: str, content: str | bytes | None
) -> None:
    assert not is_python_source(path, content)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(
            r"#!/usr/bin/env -S -i${SKILLSPECTOR_EMPTY} python3" "\npass\n",
            id="dynamic-option-suffix",
        ),
        pytest.param(
            r"#!/usr/bin/env -S PATH=/usr/bin:${PATH} python3" "\npass\n",
            id="dynamic-assignment-value",
        ),
        pytest.param(
            r"#!/usr/bin/env -S X${SKILLSPECTOR_VALUE}=1 python3" "\npass\n",
            id="dynamic-assignment-name-suffix",
        ),
        pytest.param(
            r"#!/usr/bin/env -S -C${PWD}/work /usr/bin/python3" "\npass\n",
            id="dynamic-chdir-operand-suffix",
        ),
        pytest.param(
            r"#!/usr/bin/env -S -- -X${SKILLSPECTOR_VALUE}=1 python3" "\npass\n",
            id="dynamic-dash-assignment-after-terminator",
        ),
        pytest.param(
            r"#!/usr/bin/env -S -u${SKILLSPECTOR_MAYBE} python3" "\npass\n",
            id="dynamic-attached-operand-can-consume-python-utility",
        ),
        pytest.param(
            r"#!/usr/bin/env -S -u ${SKILLSPECTOR_MAYBE} python3" "\npass\n",
            id="dynamic-separate-operand-can-consume-python-utility",
        ),
        pytest.param(
            r"#!/usr/bin/env -S /opt/${SKILLSPECTOR_ROOT}/python3" "\npass\n",
            id="dynamic-python-path-can-become-assignment",
        ),
        pytest.param(
            r"#!/usr/bin/env -S /opt/${SKILLSPECTOR_ROOT}/pypy3" "\npass\n",
            id="dynamic-pypy-path-can-become-assignment",
        ),
        pytest.param(
            r"#!/usr/bin/env -S -C${PWD} /usr/bin/python3" "\npass\n",
            id="dynamic-attached-chdir-can-consume-python-utility",
        ),
        pytest.param(
            r"#!/usr/bin/env -S -- -x${SKILLSPECTOR_ROOT}/python3" "\npass\n",
            id="dynamic-dash-path-can-become-assignment",
        ),
        pytest.param(
            r"#!/usr/bin/env -S -u${SKILLSPECTOR_NAME} /usr/bin/true python3"
            "\npass\n",
            id="dynamic-attached-operand-python-after-shift",
        ),
        pytest.param(
            r"#!/usr/bin/env -S -u ${SKILLSPECTOR_NAME} /usr/bin/true python3"
            "\npass\n",
            id="dynamic-separate-operand-python-after-shift",
        ),
        pytest.param(
            r"#!/usr/bin/env -S --unset ${SKILLSPECTOR_NAME} /usr/bin/true python3"
            "\npass\n",
            id="dynamic-long-separate-operand-shift",
        ),
        pytest.param(
            r"#!/usr/bin/env -S -P${PATH} /usr/bin/true python3" "\npass\n",
            id="dynamic-bsd-attached-operand-shift",
        ),
        pytest.param(
            "#!/usr/bin/python3# xnu comment\npass\n",
            id="platform-dependent-direct-comment",
        ),
        pytest.param(
            "#!/usr/bin/Python3\npass\n",
            id="platform-dependent-interpreter-case-alias",
        ),
        pytest.param(
            "#!/usr/bin/env Python3\npass\n",
            id="platform-dependent-env-utility-case-alias",
        ),
        pytest.param(
            "#!/usr/bin/Env python3\npass\n",
            id="platform-dependent-env-path-case-alias",
        ),
        pytest.param(
            "#!/usr//bin/env python3\npass\n",
            id="platform-dependent-env-repeated-slash-alias",
        ),
        pytest.param(
            "#!/usr/bin/./env python3\npass\n",
            id="platform-dependent-env-dot-segment-alias",
        ),
        pytest.param(
            "#!/usr/bin/../bin/env python3\npass\n",
            id="platform-dependent-env-parent-segment-alias",
        ),
        pytest.param(
            "#!//usr/bin/env python3\npass\n",
            id="platform-dependent-env-double-leading-slash-alias",
        ),
        pytest.param(
            "#!/usr/bin/python3 -W ignore /tmp/other.py\npass\n",
            id="platform-dependent-python-warning-argument",
        ),
        pytest.param(
            "#!/usr/bin/python3 -\npass\n",
            id="python-stdin-can-load-appended-source",
        ),
        pytest.param(
            "#!/usr/bin/env -S python3 -\npass\n",
            id="env-python-stdin-can-load-appended-source",
        ),
        pytest.param(
            "#!/usr/bin/python3 -X\npass\n",
            id="python-xoption-operand-exposes-appended-source",
        ),
        pytest.param(
            "#!/usr/bin/env -S python3 -X\npass\n",
            id="env-python-xoption-operand-exposes-appended-source",
        ),
        pytest.param(
            "#!/usr/bin/python3 -W\npass\n",
            id="python-warning-operand-exposes-appended-source",
        ),
        pytest.param(
            "#!/usr/bin/env -S python3 -W\npass\n",
            id="env-python-warning-operand-exposes-appended-source",
        ),
        pytest.param(
            "#!/usr/bin/env -i python3\npass\n",
            id="platform-dependent-env-ignore-environment",
        ),
        pytest.param(
            "#!/usr/bin/env -u SKILLSPECTOR_NAME python3\npass\n",
            id="platform-dependent-env-unset",
        ),
        pytest.param(
            "#!/usr/bin/env -C /tmp python3\npass\n",
            id="platform-dependent-env-chdir",
        ),
        pytest.param(
            "#!/usr/bin/env python3 -I\npass\n",
            id="platform-dependent-python-option",
        ),
        pytest.param(
            "#!/usr/bin/env python3 -vv\npass\n",
            id="platform-dependent-python-repeated-option",
        ),
        pytest.param(
            "#!/usr/bin/env python3 -IB\npass\n",
            id="platform-dependent-python-clustered-options",
        ),
        pytest.param(
            "#!/usr/bin/env python3 -bbb\npass\n",
            id="platform-dependent-python-repeated-bytes-option",
        ),
        pytest.param(
            "#!/usr/bin/env python3 -OOO\npass\n",
            id="platform-dependent-python-repeated-optimize-option",
        ),
        pytest.param(
            "#!/usr/bin/env -i python3 # xnu comment\npass\n",
            id="platform-dependent-env-comment",
        ),
        pytest.param(
            "#!/bin/sh#/python3\npass\n",
            id="platform-dependent-direct-interpreter-identity",
        ),
        pytest.param(
            "#!/usr/bin/env -S - -i python3\npass\n",
            id="platform-dependent-lone-dash",
        ),
        pytest.param(
            "#!/usr/bin/env -S --unset /usr/bin/true python3\npass\n",
            id="platform-dependent-double-dash-option",
        ),
        pytest.param(
            "#!/usr/bin/env -S-P/usr/bin:/bin python3\npass\n",
            id="freebsd-darwin-attached-path-option",
        ),
        pytest.param(
            "#!/usr/bin/env -S -i-v python3\npass\n",
            id="freebsd-clustered-compatibility-option",
        ),
        pytest.param(
            "#!/usr/bin/env -S-iv -P/usr/bin:/bin python3\npass\n",
            id="freebsd-darwin-clustered-path-option",
        ),
        pytest.param(
            "#!/usr/bin/env -S -a alternate python3\npass\n",
            id="gnu-argv0-option",
        ),
        pytest.param(
            "#!/usr/bin/env -S --argv0=alternate python3\npass\n",
            id="gnu-long-argv0-option",
        ),
        pytest.param(
            "#!/usr/bin/env -S --argv0= python3\npass\n",
            id="gnu-empty-long-argv0-option",
        ),
        pytest.param(
            "#!/usr/bin/env -S -L root python3\npass\n",
            id="freebsd-login-class-separate-operand",
        ),
        pytest.param(
            "#!/usr/bin/env -S -Lroot python3\npass\n",
            id="freebsd-login-class-attached-operand",
        ),
        pytest.param(
            "#!/usr/bin/env -S -ivLroot python3\npass\n",
            id="freebsd-clustered-login-class-operand",
        ),
        pytest.param(
            "#!/usr/bin/env -S =x python3\npass\n",
            id="gnu-empty-name-assignment",
        ),
        pytest.param(
            "#!/usr/bin/env -S FOO=bar =x python3\npass\n",
            id="gnu-empty-name-assignment-after-assignment",
        ),
        pytest.param(
            "#!/usr/bin/env -S -i =x python3\npass\n",
            id="gnu-empty-name-assignment-after-option",
        ),
        pytest.param(
            "#!/usr/bin/env -S -- - python3\npass\n",
            id="gnu-post-terminator-legacy-dash",
        ),
        pytest.param(
            "#!/usr/bin/env -S -- - FOO=bar python3\npass\n",
            id="gnu-post-terminator-legacy-dash-before-assignment",
        ),
        pytest.param(
            "#!/usr/bin/env -S --env0-from=environment python3\npass\n",
            id="gnu-environment-file-option",
        ),
        pytest.param(
            r"#!/usr/bin/env -S -P/usr/bin:${PATH} python3" "\npass\n",
            id="freebsd-dynamic-path-operand",
        ),
        pytest.param(
            r"#!/usr/bin/env -S -P${PATH} python3" "\npass\n",
            id="freebsd-dynamic-attached-path-arity",
        ),
        pytest.param(
            r'#!/usr/bin/env -S -P "${PATH}" /usr/bin/python3' "\npass\n",
            id="freebsd-quoted-dynamic-path-operand",
        ),
        pytest.param(
            r"#!/usr/bin/env -S -P ${PATH} /usr/bin/python3" "\npass\n",
            id="freebsd-dynamic-separate-path-arity",
        ),
        pytest.param(
            r"#!/usr/bin/env -S -u ${SKILLSPECTOR_NAME}#suffix python3" "\npass\n",
            id="platform-dependent-dynamic-prefix-before-comment-marker",
        ),
        pytest.param(
            "#!/usr/bin/env -i-vSpython3\npass\n",
            id="freebsd-clustered-outer-split-string",
        ),
        pytest.param(
            "#!/usr/bin/env --split=python3\npass\n",
            id="gnu-outer-abbreviated-split-string",
        ),
        pytest.param(
            "#!/usr/bin/env -S --chd=/tmp python3\npass\n",
            id="gnu-abbreviated-long-chdir",
        ),
        pytest.param(
            "#!/usr/bin/env -S --ignore-e python3\npass\n",
            id="gnu-abbreviated-long-ignore-environment",
        ),
        pytest.param(
            "#!/usr/bin/env -S --i python3\npass\n",
            id="freebsd-double-dash-cluster",
        ),
        pytest.param(
            "#!/usr/bin/env -S --unknown python3\npass\n",
            id="freebsd-double-dash-unset-operand",
        ),
        pytest.param(
            "#!/usr/bin/env -S --uns=PYTHONPATH python3\npass\n",
            id="gnu-abbreviated-long-unset",
        ),
        pytest.param(
            "#!/usr/bin/env -S FOO=bar\\ baz python3\npass\n",
            id="freebsd-escaped-space",
        ),
        pytest.param(
            r'#!/usr/bin/env -S -S "FOO=a\\\nb /usr/bin/python3"' "\npass\n",
            id="freebsd-nested-escaped-newline",
        ),
        pytest.param(
            "#!/usr/bin/env --split-string=python3\npass\n",
            id="gnu-only-outer-long-split",
        ),
        pytest.param(
            "#!/usr/bin/env --spl=python3\npass\n",
            id="gnu-only-outer-abbreviated-split",
        ),
        pytest.param(
            "#!/usr/bin/env -S --split-string=python3\npass\n",
            id="gnu-only-nested-long-split",
        ),
        pytest.param(
            "#!/usr/bin/env -S --spl=python3\npass\n",
            id="gnu-only-nested-abbreviated-split",
        ),
        pytest.param(
            "#!/usr/bin/env -S --ignore-environment python3\npass\n",
            id="gnu-only-nested-ignore-environment",
        ),
        pytest.param(
            "#!/usr/bin/env -S --unset=FOO python3\npass\n",
            id="gnu-only-nested-unset",
        ),
        pytest.param(
            "#!/usr/bin/env -S --unset FOO=BAR python3\npass\n",
            id="gnu-invalid-unset-name-bsd-assignment",
        ),
        pytest.param(
            "#!/usr/bin/env -S -u '' python3\npass\n",
            id="darwin-xnu-retains-quotes-after-split-payload",
        ),
        pytest.param(
            r"#!/usr/bin/env -S -- ${SKILLSPECTOR_NAME}=X python3" "\npass\n",
            id="dynamic-assignment-name-after-option-terminator",
        ),
        pytest.param(
            r"#!/usr/bin/env -S python3 ${SKILLSPECTOR_ARGUMENT}" "\npass\n",
            id="dynamic-python-pre-script-argument",
        ),
        pytest.param(
            r"#!/usr/bin/env -S python3\c node" "\npass\n",
            id="platform-dependent-env-string-terminator",
        ),
        pytest.param(
            r"#!/usr/bin/env -S python3 -- ${SKILLSPECTOR_ARGUMENT}" "\npass\n",
            id="dynamic-python-argument-after-option-terminator",
        ),
        pytest.param(
            r"#!/usr/bin/env -S python3 -- ${SKILLSPECTOR_ARGUMENT} /tmp/other.py"
            "\npass\n",
            id="dynamic-python-argument-before-fixed-other-script",
        ),
        pytest.param(
            r"#!/usr/bin/env -S py${SKILLSPECTOR_EMPTY}thon3" "\npass\n",
            id="dynamic-interpreter-fragment",
        ),
        pytest.param(
            r"#!/usr/bin/env -S ${SKILLSPECTOR_INTERPRETER}" "\npass\n",
            id="dynamic-utility",
        ),
        pytest.param(
            r"#!/usr/bin/env -S X${SKILLSPECTOR_ASSIGNMENT} python3" "\npass\n",
            id="dynamic-token-role",
        ),
        pytest.param(
            r"#!/usr/bin/env -S ${SKILLSPECTOR_INTERPRETER}#suffix python3" "\npass\n",
            id="dynamic-utility-before-comment-marker",
        ),
        pytest.param(
            r"#!/usr/bin/env -S ${SKILLSPECTOR_OPTION} python3" "\npass\n",
            id="dynamic-leading-token",
        ),
        pytest.param(
            r"#!/usr/bin/env -S --split-string=${SKILLSPECTOR_SPLIT}" "\npass\n",
            id="dynamic-nested-split-string",
        ),
        pytest.param(
            r"#!/usr/bin/env -S --spl=${SKILLSPECTOR_SPLIT}" "\npass\n",
            id="dynamic-abbreviated-nested-split-string",
        ),
        pytest.param(
            r"#!/usr/bin/env -S -S ${SKILLSPECTOR_SPLIT}" "\npass\n",
            id="dynamic-nested-short-split-string",
        ),
        pytest.param(
            "#!/usr/bin/env python3 node\npass\n",
            id="external-python-script-via-plain-env",
        ),
        pytest.param(
            "#!/usr/bin/python3 /tmp/other.py\npass\n",
            id="external-python-script-direct",
        ),
        pytest.param(
            "#!/usr/bin/env -S python3 /tmp/other.py\npass\n",
            id="external-python-script-via-env-split",
        ),
        pytest.param(
            r"#!/usr/bin/env -S python3 argument\ with-space" "\npass\n",
            id="external-python-script-with-escaped-space",
        ),
        pytest.param(
            r"#!/usr/bin/env -S -u${SKILLSPECTOR_NAME} python3 /usr/bin/true"
            "\npass\n",
            id="external-script-after-dynamic-attached-operand",
        ),
        pytest.param(
            r"#!/usr/bin/env -S -u ${SKILLSPECTOR_NAME} python3 /usr/bin/true"
            "\npass\n",
            id="external-script-after-dynamic-separate-operand",
        ),
        pytest.param(
            r"#!/usr/bin/env -S -u${SKILLSPECTOR_NAME} python3 ''" "\npass\n",
            id="platform-dependent-empty-script-argument",
        ),
        pytest.param(
            r"#!/usr/bin/env -S /opt/${SKILLSPECTOR_ROOT}/python3 /usr/bin/true"
            "\npass\n",
            id="external-script-after-dynamic-python-path",
        ),
        pytest.param(
            r"#!/usr/bin/env -S -- -x${SKILLSPECTOR_ROOT}/python3 /usr/bin/true"
            "\npass\n",
            id="external-script-after-dynamic-dash-path",
        ),
        pytest.param(
            "#!/usr/bin/python3"
            + " " * (python_ast.MAX_PYTHON_SHEBANG_CHARS - len("#!/usr/bin/python3") + 1)
            + "\npass\n",
            id="over-maximum-length-shebang",
        ),
    ],
)
def test_uncertain_shebang_is_explicitly_ambiguous(content: str) -> None:
    assert classify_python_source("runner", content) is PythonSourceClassification.AMBIGUOUS
    assert not is_python_source("runner", content)
    assert may_be_python_source("runner", content)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("#!/usr/bin/env -S -L python3\n", id="separate-missing-utility"),
        pytest.param("#!/usr/bin/env -S -iL python3\n", id="clustered-after-ignore"),
        pytest.param("#!/usr/bin/env -S -vL python3\n", id="clustered-after-verbose"),
        pytest.param("#!/usr/bin/env -iLSpython3\n", id="outer-operand-not-split"),
    ],
)
def test_freebsd_login_class_option_requires_an_operand(content: str) -> None:
    assert classify_python_source("runner", content) is PythonSourceClassification.NON_PYTHON


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("#!/usr/bin/env -S =BAD python3\n", id="empty-name"),
        pytest.param("#!/usr/bin/env -S -- =BAD python3\n", id="after-option-terminator"),
        pytest.param("#!/usr/bin/env -S == python3\n", id="repeated-equals"),
        pytest.param("#!/usr/bin/env -S = python3\n", id="bare-equals"),
        pytest.param(
            "#!/usr/bin/env -S FOO=bar =x python3\n",
            id="after-valid-assignment",
        ),
        pytest.param("#!/usr/bin/env -S -i =x python3\n", id="after-ignore-option"),
    ],
)
def test_empty_env_assignment_name_is_platform_ambiguous(content: str) -> None:
    assert classify_python_source("runner", content) is PythonSourceClassification.AMBIGUOUS


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("#!/usr/bin/env -S -- - python3\n", id="direct-utility"),
        pytest.param(
            "#!/usr/bin/env -S -- - FOO=bar python3\n",
            id="before-assignment",
        ),
    ],
)
def test_gnu_post_terminator_legacy_dash_is_platform_ambiguous(content: str) -> None:
    assert classify_python_source("runner", content) is PythonSourceClassification.AMBIGUOUS


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("#!/usr/bin/env -S -- - - python3\n", id="repeated-dash"),
        pytest.param("#!/usr/bin/env -S -- FOO=bar - python3\n", id="after-assignment"),
    ],
)
def test_gnu_legacy_dash_is_only_recognized_immediately_after_getopt(content: str) -> None:
    assert classify_python_source("runner", content) is PythonSourceClassification.NON_PYTHON


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param("-i -u A=B python3", id="short-separate-clear-before"),
        pytest.param("-u A=B -i python3", id="short-separate-clear-after"),
        pytest.param("-i -uA=B python3", id="short-attached-clear-before"),
        pytest.param("-uA=B -i python3", id="short-attached-clear-after"),
        pytest.param(
            "--ignore-environment --unset A=B python3",
            id="long-separate-clear-before",
        ),
        pytest.param(
            "--unset A=B --ignore-environment python3",
            id="long-separate-clear-after",
        ),
        pytest.param(
            "--ignore-environment --unset=A=B python3",
            id="long-attached-clear-before",
        ),
        pytest.param(
            "--unset=A=B --ignore-environment python3",
            id="long-attached-clear-after",
        ),
        pytest.param("-i -u '' python3", id="empty-short-operand"),
        pytest.param("-i --unset= python3", id="empty-long-attached-operand"),
        pytest.param("-i -u = python3", id="equals-short-operand"),
        pytest.param("-u A=B - python3", id="legacy-clear-after"),
        pytest.param("-iuA=B python3", id="same-cluster-clear-before"),
        pytest.param("-iS-uA=B python3", id="nested-split-clear-before"),
    ],
)
def test_gnu_clear_environment_skips_invalid_queued_unsets(payload: str) -> None:
    content = f"#!/usr/bin/env -S {payload}\npass\n"
    assert classify_python_source("runner", content) is PythonSourceClassification.AMBIGUOUS


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(
            "#!/usr/bin/env -iS-uA=B python3\npass\n",
            id="outer-short-attached-unset",
        ),
        pytest.param(
            "#!/usr/bin/env -viS-u A=B python3\npass\n",
            id="outer-cluster-separate-unset",
        ),
    ],
)
def test_gnu_outer_split_clear_environment_is_retained(content: str) -> None:
    assert classify_python_source("runner", content) is PythonSourceClassification.AMBIGUOUS


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param("-u A=B python3", id="short-separate"),
        pytest.param("-uA=B python3", id="short-attached"),
        pytest.param(
            "--split-string='--unset A=B python3'",
            id="gnu-outer-long-separate",
        ),
        pytest.param("--unset=A=B python3", id="long-attached"),
        pytest.param("--unset= python3", id="long-attached-empty"),
        pytest.param("- -u A=B python3", id="legacy-clear-before-ends-options"),
        pytest.param(
            "-i --env0-from=environment -u A=B python3",
            id="env0-from-after-clear",
        ),
        pytest.param(
            "-u A=B -i --env0-from environment python3",
            id="env0-from-after-unset",
        ),
    ],
)
def test_invalid_gnu_unset_without_effective_clear_cannot_execute(payload: str) -> None:
    content = f"#!/usr/bin/env -S {payload}\npass\n"
    assert classify_python_source("runner", content) is PythonSourceClassification.NON_PYTHON


@pytest.mark.parametrize(
    "split_payload",
    [
        pytest.param(
            r"-u${SKILLSPECTOR_MAYBE} python3",
            id="attached-unset-operand",
        ),
        pytest.param(
            r"-u ${SKILLSPECTOR_MAYBE} python3",
            id="separate-unset-operand",
        ),
    ],
)
def test_real_env_split_substitution_has_executing_and_nonexecuting_branches(
    tmp_path: Path,
    split_payload: str,
) -> None:
    source = tmp_path / "runner"
    marker = "ENV_BRANCH_EXECUTED"
    source.write_text(f'print("{marker}")\n', encoding="utf-8")
    assert not os.access(source, os.X_OK)

    runtime_environment = dict(os.environ)
    runtime_environment["SKILLSPECTOR_MAYBE"] = "SKILLSPECTOR_UNUSED"
    executed = subprocess.run(
        ["/usr/bin/env", "-S", split_payload, str(source)],
        env=runtime_environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert executed.returncode == 0
    assert marker in executed.stdout

    runtime_environment.pop("SKILLSPECTOR_MAYBE")
    not_executed = subprocess.run(
        ["/usr/bin/env", "-S", split_payload, str(source)],
        env=runtime_environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert not_executed.returncode != 0
    assert marker not in not_executed.stdout


def test_plain_env_opaque_path_with_spaces_is_platform_ambiguous() -> None:
    content = "#!/usr/bin/env /tmp/a b/python3\npass\n"

    assert classify_python_source("runner", content) is PythonSourceClassification.AMBIGUOUS
    assert may_be_python_source("runner", content)


def test_bare_python_command_option_consumes_inert_appended_path() -> None:
    assert (
        classify_python_source("runner", "#!/usr/bin/python3 -c\npass\n")
        is PythonSourceClassification.NON_PYTHON
    )


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("eval(input())", id="active-basename"),
        pytest.param("tools/eval(input())", id="active-relative-basename"),
    ],
)
def test_bare_python_command_can_execute_active_source_spelling(path: str) -> None:
    assert (
        classify_python_source(path, "#!/usr/bin/env -S python3 -c\npass\n")
        is PythonSourceClassification.AMBIGUOUS
    )


@pytest.mark.parametrize("path", ["runner", "pass", "invalid("])
def test_bare_python_command_inert_or_invalid_source_spelling_is_non_python(path: str) -> None:
    assert (
        classify_python_source(path, "#!/usr/bin/env -S python3 -c\npass\n")
        is PythonSourceClassification.NON_PYTHON
    )


@pytest.mark.parametrize("path", ["runner", "pkg.evil", "tools/a-b", "eval(input())"])
def test_bare_python_module_can_select_external_module_for_source(path: str) -> None:
    assert (
        classify_python_source(path, "#!/usr/bin/env -S python3 -m\npass\n")
        is PythonSourceClassification.AMBIGUOUS
    )


@pytest.mark.parametrize(
    ("launcher", "option"),
    [
        pytest.param("/usr/bin/python3", "-ic", id="direct-command-cluster"),
        pytest.param("/usr/bin/python3", "-im", id="direct-module-cluster"),
        pytest.param("/usr/bin/python3", "-Iic", id="direct-isolated-command-cluster"),
        pytest.param("/usr/bin/python3", "-Iim", id="direct-isolated-module-cluster"),
        pytest.param("/usr/bin/env -S python3", "-i -c", id="env-command-separate"),
        pytest.param("/usr/bin/env -S python3", "-ic", id="env-command-cluster"),
        pytest.param("/usr/bin/env -S python3", "-I -i -c", id="env-isolated-command-separate"),
        pytest.param("/usr/bin/env -S python3", "-Iic", id="env-isolated-command-cluster"),
        pytest.param("/usr/bin/env -S python3", "-i -m", id="env-module-separate"),
        pytest.param("/usr/bin/env -S python3", "-im", id="env-module-cluster"),
        pytest.param("/usr/bin/env -S python3", "-I -i -m", id="env-isolated-module-separate"),
        pytest.param("/usr/bin/env -S python3", "-Iim", id="env-isolated-module-cluster"),
    ],
)
def test_forced_interactive_bare_command_option_can_recover_appended_path(
    launcher: str, option: str
) -> None:
    assert (
        classify_python_source("runner", f"#!{launcher} {option}\npass\n")
        is PythonSourceClassification.AMBIGUOUS
    )


@pytest.mark.parametrize(
    "option",
    ["-i -V", "-iV", "-i -h", "-ih", "-i --version", "-i --help"],
)
def test_forced_interactive_terminal_option_remains_non_python(option: str) -> None:
    assert (
        classify_python_source(
            "runner",
            f"#!/usr/bin/env -S python3 {option}\npass\n",
        )
        is PythonSourceClassification.NON_PYTHON
    )


def test_forced_interactive_missing_hash_option_operand_remains_non_python() -> None:
    assert (
        classify_python_source(
            "runner",
            "#!/usr/bin/env -S python3 -i --check-hash-based-pycs\npass\n",
        )
        is PythonSourceClassification.NON_PYTHON
    )


@pytest.mark.parametrize(
    "arguments",
    [
        "PYTHONINSPECT=1 python3 -c",
        "PYTHONINSPECT=0 python3 -m",
        "-i PYTHONINSPECT=yes python3 -c",
        "PYTHONINSPECT= PYTHONINSPECT=enabled python3 -m",
    ],
)
def test_static_pythoninspect_can_recover_bare_command_path(arguments: str) -> None:
    assert (
        classify_python_source(
            "runner",
            f"#!/usr/bin/env -S {arguments}\npass\n",
        )
        is PythonSourceClassification.AMBIGUOUS
    )


@pytest.mark.parametrize(
    "arguments",
    [
        "PYTHONINSPECT= python3 -c",
        "PYTHONINSPECT=enabled PYTHONINSPECT= python3 -c",
        "PYTHONINSPECT=enabled python3 -E -c",
        "PYTHONINSPECT=enabled python3 -I -c",
    ],
)
def test_pythoninspect_disabled_or_ignored_keeps_bare_command_non_python(
    arguments: str,
) -> None:
    assert (
        classify_python_source(
            "runner",
            f"#!/usr/bin/env -S {arguments}\npass\n",
        )
        is PythonSourceClassification.NON_PYTHON
    )


@pytest.mark.parametrize(
    "arguments",
    [
        "--env0-from=/tmp/environment python3 -c",
        "-L login python3 -c",
        "-U user python3 -m",
        "-u PYTHONINSPECT -L login python3 -c",
        "-L login -u PYTHONINSPECT python3 -m",
        "-L login PYTHONINSPECT=enabled python3 -c",
    ],
)
def test_environment_sources_make_pythoninspect_runtime_dependent(arguments: str) -> None:
    assert (
        classify_python_source(
            "runner",
            f"#!/usr/bin/env -S {arguments}\npass\n",
        )
        is PythonSourceClassification.AMBIGUOUS
    )


@pytest.mark.parametrize(
    "arguments",
    [
        "--env0-from=/tmp/environment -u PYTHONINSPECT python3 -c",
        "-u PYTHONINSPECT --env0-from=/tmp/environment python3 -c",
        "--env0-from=/tmp/environment --unset=PYTHONINSPECT python3 -c",
        "--unset PYTHONINSPECT --env0-from=/tmp/environment python3 -c",
        "--env0-from=/tmp/environment PYTHONINSPECT= python3 -c",
        "-L login PYTHONINSPECT= python3 -c",
        "--env0-from=/tmp/environment python3 -E -c",
        "-L login python3 -I -c",
    ],
)
def test_final_pythoninspect_removal_disables_environment_repl(arguments: str) -> None:
    assert (
        classify_python_source(
            "runner",
            f"#!/usr/bin/env -S {arguments}\npass\n",
        )
        is PythonSourceClassification.NON_PYTHON
    )


@pytest.mark.parametrize(
    ("path", "launcher"),
    [
        pytest.param("default", "/usr/bin/python3", id="direct-default"),
        pytest.param("tools/always", "/usr/bin/python3", id="direct-subdirectory-always"),
        pytest.param(
            "bundle.zip!/never",
            "/usr/bin/env -S python3",
            id="env-nested-like-never",
        ),
        pytest.param(
            "tools/Default",
            "/usr/bin/env -S python3",
            id="env-case-insensitive-default-alias",
        ),
    ],
)
def test_bare_python_hash_option_can_consume_runtime_source_alias(path: str, launcher: str) -> None:
    assert (
        classify_python_source(
            path,
            f"#!{launcher} --check-hash-based-pycs\npass\n",
        )
        is PythonSourceClassification.AMBIGUOUS
    )


@pytest.mark.parametrize("path", ["runner", "tools/default.txt", "/tmp/runner"])
@pytest.mark.parametrize("launcher", ["/usr/bin/python3", "/usr/bin/env -S python3"])
def test_bare_python_hash_option_rejects_non_value_source_path(path: str, launcher: str) -> None:
    assert (
        classify_python_source(
            path,
            f"#!{launcher} --check-hash-based-pycs\npass\n",
        )
        is PythonSourceClassification.NON_PYTHON
    )


def test_explicit_python_hash_option_operand_preserves_appended_source() -> None:
    assert (
        classify_python_source(
            "runner",
            "#!/usr/bin/env -S python3 --check-hash-based-pycs default\npass\n",
        )
        is PythonSourceClassification.AMBIGUOUS
    )
    assert (
        classify_python_source(
            "runner",
            "#!/usr/bin/python3 --check-hash-based-pycs default\npass\n",
        )
        is PythonSourceClassification.AMBIGUOUS
    )


@pytest.mark.parametrize(
    "interpreter",
    [
        pytest.param("python3.6", id="known-old-version"),
        pytest.param("python3", id="unversioned-minor"),
        pytest.param("python3.14", id="known-new-version-conservative"),
    ],
)
@pytest.mark.parametrize(
    "option",
    [
        pytest.param("--check-hash-based-pycs default", id="hash-based-pycs"),
        pytest.param("-I", id="isolated-mode"),
        pytest.param("-P", id="safe-path"),
        pytest.param("-q", id="quiet-mode"),
    ],
)
def test_version_dependent_python_option_is_ambiguous(interpreter: str, option: str) -> None:
    assert (
        classify_python_source(
            "runner",
            f"#!/usr/bin/env -S {interpreter} {option}\npass\n",
        )
        is PythonSourceClassification.AMBIGUOUS
    )


def test_version_dependent_python_option_followed_by_help_is_non_python() -> None:
    assert (
        classify_python_source(
            "runner",
            "#!/usr/bin/env -S python3 -Ph\npass\n",
        )
        is PythonSourceClassification.NON_PYTHON
    )


@pytest.mark.parametrize(
    ("path", "launcher"),
    [
        pytest.param("-h", "/usr/bin/python3", id="root-dash-basename-direct"),
        pytest.param(
            "tools/-V",
            "/usr/bin/env -S python3",
            id="subdirectory-dash-basename-env-split",
        ),
        pytest.param(
            "tools/-options/runner",
            "/usr/bin/python3 -B",
            id="dash-intermediate-component-direct",
        ),
        pytest.param(
            "bundle.zip!/-nested/runner",
            "/usr/bin/env -S python3",
            id="nested-dash-component-env-split",
        ),
    ],
)
def test_implicit_python_source_path_can_be_reparsed_as_an_option(path: str, launcher: str) -> None:
    assert (
        classify_python_source(path, f"#!{launcher}\npass\n")
        is PythonSourceClassification.AMBIGUOUS
    )


def test_python_option_terminator_protects_dash_prefixed_appended_source() -> None:
    assert (
        classify_python_source(
            "tools/-h",
            "#!/usr/bin/env -S python3 --\npass\n",
        )
        is PythonSourceClassification.PYTHON
    )


@pytest.mark.parametrize("option", ["-cpass", "-mrunpy", "-m trace --trace"])
def test_supplied_python_command_can_execute_appended_path(option: str) -> None:
    assert (
        classify_python_source("runner", f"#!/usr/bin/python3 {option}\npass\n")
        is PythonSourceClassification.AMBIGUOUS
    )


@pytest.mark.parametrize(
    ("path", "argument"),
    [
        pytest.param("/tmp/runner", "/tmp/runner", id="same-absolute-path"),
    ],
)
def test_python_script_argument_can_select_analyzed_source(path: str, argument: str) -> None:
    assert (
        classify_python_source(path, f"#!/usr/bin/python3 {argument}\npass\n")
        is PythonSourceClassification.PYTHON
    )


@pytest.mark.parametrize("argument", ["runner", "./runner"])
def test_python_relative_script_argument_depends_on_invocation_cwd(argument: str) -> None:
    assert (
        classify_python_source("runner", f"#!/usr/bin/python3 {argument}\npass\n")
        is PythonSourceClassification.AMBIGUOUS
    )


def test_python_script_alias_to_analyzed_source_is_ambiguous() -> None:
    assert (
        classify_python_source("runner", "#!/usr/bin/python3 /tmp/runner\npass\n")
        is PythonSourceClassification.AMBIGUOUS
    )


@pytest.mark.parametrize(
    ("path", "argument"),
    [
        pytest.param("Runner", "runner", id="case-insensitive-volume"),
        pytest.param("rúnner", "ru\u0301nner", id="unicode-normalizing-volume"),
    ],
)
def test_python_script_filesystem_alias_is_ambiguous(path: str, argument: str) -> None:
    assert (
        classify_python_source(path, f"#!/usr/bin/python3 {argument}\npass\n")
        is PythonSourceClassification.AMBIGUOUS
    )


@pytest.mark.parametrize(
    ("path", "argument"),
    [
        pytest.param("dir/runner", r"dir\runner", id="backslash-in-argument"),
        pytest.param(r"dir\runner", "dir/runner", id="backslash-in-source-path"),
        pytest.param("runner", "link/../runner", id="parent-through-symlink"),
    ],
)
def test_python_script_lexical_path_alias_is_not_certified_as_self(
    path: str, argument: str
) -> None:
    assert (
        classify_python_source(path, f"#!/usr/bin/python3 {argument}\npass\n")
        is PythonSourceClassification.AMBIGUOUS
    )


@pytest.mark.parametrize(
    "command",
    [
        pytest.param("X=/tmp/python3", id="assignment"),
        pytest.param("/tmp/X=/python3", id="assignment-with-path-like-name"),
        pytest.param("-P/tmp/python3", id="freebsd-option-operand"),
    ],
)
def test_plain_env_opaque_non_utility_is_not_python(command: str) -> None:
    assert (
        classify_python_source("runner", f"#!/usr/bin/env {command}\npass\n")
        is PythonSourceClassification.NON_PYTHON
    )


def test_linux_shebang_buffer_boundaries_are_explicitly_ambiguous() -> None:
    prefix = "#!/usr/bin/env -S python3"
    line_at_127_bytes = prefix + " " * (125 - len(prefix)) + "-h"
    line_at_128_bytes = prefix + " " * (126 - len(prefix)) + "-h"
    line_at_255_bytes = prefix + " " * (253 - len(prefix)) + "-h"
    line_at_256_bytes = prefix + " " * (254 - len(prefix)) + "-h"

    assert len(line_at_127_bytes.encode()) == 127
    assert len(line_at_128_bytes.encode()) == 128
    assert len(line_at_255_bytes.encode()) == 255
    assert len(line_at_256_bytes.encode()) == 256
    assert (
        classify_python_source("runner", line_at_127_bytes + "\npass\n")
        is PythonSourceClassification.NON_PYTHON
    )
    for line in (line_at_128_bytes, line_at_255_bytes, line_at_256_bytes):
        assert (
            classify_python_source("runner", line + "\npass\n")
            is PythonSourceClassification.AMBIGUOUS
        )
        assert (
            classify_python_source("runner", (line + "\npass\n").encode())
            is PythonSourceClassification.AMBIGUOUS
        )


def test_legacy_linux_shebang_buffer_is_included_in_classification() -> None:
    prefix = "#!/usr/bin/env -S python3"
    line = prefix + " " * (130 - len(prefix)) + "-h"

    assert len(line.encode()) == 132
    assert (
        classify_python_source("runner", line + "\npass\n") is PythonSourceClassification.AMBIGUOUS
    )


def test_linux_shebang_views_with_same_identity_remain_definite() -> None:
    line = "#!/usr/bin/env -S python3" + " " * 300

    assert classify_python_source("runner", line + "\npass\n") is PythonSourceClassification.PYTHON


def test_linux_shebang_buffer_boundary_is_measured_in_bytes() -> None:
    line = "#!/usr/bin/env -S FOO=" + "é" * 105 + " python3" + " " * 30 + "node"

    assert len(line) < 256
    assert len(line.encode()) >= 256
    assert (
        classify_python_source("runner", line + "\npass\n") is PythonSourceClassification.AMBIGUOUS
    )


@pytest.mark.parametrize(
    ("raw", "marker"),
    [
        pytest.param(
            b"#! latin-1 comment: \xff\n# coding: latin-1\nvalue = '\xff'\n",
            "ÿ",
            id="non-utf8-first-comment-line-second-line-latin1-cookie",
        ),
        pytest.param(
            b"# comment: \xff\n# coding: iso_latin_1\nvalue = '\xff'\n",
            "ÿ",
            id="canonical-latin1-alias",
        ),
        pytest.param(
            b"# comment: \xff\r# coding: latin-1\rvalue = '\xff'\r",
            "ÿ",
            id="second-physical-line-cookie-with-cr-newlines",
        ),
        pytest.param(
            b"# comment: \xff\r\n# coding: latin-1\r\nvalue = '\xff'\r\n",
            "ÿ",
            id="crlf-counts-as-one-physical-newline",
        ),
        pytest.param(
            b"# coding: UTF_8\nvalue = '\xc3\xa9'\n",
            "é",
            id="canonical-utf8-alias",
        ),
        pytest.param(
            b"\xef\xbb\xbf# comment\n# coding: utf-8\nvalue = 'ok'\n",
            "value = 'ok'",
            id="utf8-bom-compatible-cookie",
        ),
    ],
)
def test_decode_python_source_matches_python314_pep263_detection(raw: bytes, marker: str) -> None:
    decoded = decode_python_source(raw)

    assert marker in decoded
    assert not decoded.startswith("\ufeff")


@pytest.mark.parametrize(
    ("raw", "error_type"),
    [
        pytest.param(
            b"value = '\xff'\n# coding: latin-1\n",
            SyntaxError,
            id="cookie-after-non-comment-source-line",
        ),
        pytest.param(
            b"\xef\xbb\xbf# comment\n# coding: latin-1\n",
            SyntaxError,
            id="utf8-bom-conflicts-with-latin1-cookie",
        ),
        pytest.param(
            b"# comment\n# coding: definitely-unknown\n",
            SyntaxError,
            id="unknown-codec",
        ),
        pytest.param(
            b"# coding: utf-8\n# second\nvalue = '\xff'\n",
            UnicodeDecodeError,
            id="full-buffer-invalid-decode",
        ),
        pytest.param(
            b"# first\n# second\n# coding: latin-1\nvalue = '\xff'\n",
            UnicodeDecodeError,
            id="third-line-cookie-is-ignored",
        ),
        pytest.param(
            b"# first\r# second\r# coding: latin-1\rvalue = '\xff'\r",
            UnicodeDecodeError,
            id="third-physical-line-cookie-with-cr-newlines",
        ),
        pytest.param(
            b"# first\r# second\n# coding: latin-1\nvalue = '\xff'\n",
            UnicodeDecodeError,
            id="third-physical-line-cookie-with-mixed-newlines",
        ),
    ],
)
def test_decode_python_source_rejects_python314_pep263_errors(
    raw: bytes, error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        decode_python_source(raw)


def test_decode_python_source_normalizes_mixed_newlines_before_detection() -> None:
    raw = b"\r\n\t#coding:utf_16be\rx=1\n"

    with pytest.raises(SyntaxError):
        decode_python_source(raw)


def test_decode_python_source_normalizes_crlf_before_full_decode() -> None:
    raw = b"\t#coding=utf_16be\r\nx=1\r\n"
    normalized = b"\t#coding=utf_16be\nx=1\n"

    decoded = decode_python_source(raw)

    assert decoded == normalized.decode("utf_16be")
    assert parse_python_source(decoded, "encoded.py").is_parseable


def test_long_shebang_with_same_linux_and_full_identity_is_not_ambiguous() -> None:
    line = "#!/usr/bin/node" + " " * 300

    assert (
        classify_python_source("runner", line + "\npass\n") is PythonSourceClassification.NON_PYTHON
    )


def test_parse_python_source_exposes_import_aliases() -> None:
    parsed = parse_python_source(
        "import os as operating_system\nfrom subprocess import run\n", "script.py"
    )

    assert parsed.is_parseable
    assert parsed.tree is not None
    assert parsed.import_aliases == {
        "operating_system": "os",
        "run": "subprocess.run",
    }
    assert parsed.lines == ["import os as operating_system", "from subprocess import run"]
    assert parsed.content == "import os as operating_system\nfrom subprocess import run\n"


def test_parse_python_source_retains_syntax_error_result() -> None:
    parsed = parse_python_source("def broken(\n", "broken.py")

    assert not parsed.is_parseable
    assert parsed.tree is None
    assert parsed.import_aliases == {}
    assert parsed.parse_error == "SyntaxError"


def test_parse_python_source_retains_value_error_result(monkeypatch) -> None:
    def raise_value_error(*args, **kwargs):
        raise ValueError("invalid source")

    monkeypatch.setattr(python_ast.ast, "parse", raise_value_error)

    parsed = parse_python_source("x = 1\n", "broken.py")

    assert not parsed.is_parseable
    assert parsed.parse_error == "ValueError"


def test_parse_python_source_retains_recursion_error_result(monkeypatch) -> None:
    def raise_recursion_error(*args, **kwargs):
        raise RecursionError("expression is too deep")

    monkeypatch.setattr(python_ast.ast, "parse", raise_recursion_error)

    parsed = parse_python_source("x = 1\n", "deep.py")

    assert not parsed.is_parseable
    assert parsed.parse_error == "RecursionError"


def test_build_python_ast_cache_caches_failures_and_skips_oversized_files() -> None:
    cache = build_python_ast_cache(
        ["valid.py", "uppercase.PY", "broken.py", "oversized.py", "readme.md"],
        {
            "valid.py": "x = 1\n",
            "uppercase.PY": "x = 2\n",
            "broken.py": "def bad(\n",
            "oversized.py": "x" * 11,
            "readme.md": "not Python",
        },
        max_source_chars=10,
    )

    assert set(cache) == {"valid.py", "uppercase.PY", "broken.py"}
    assert cache["valid.py"].is_parseable
    assert cache["uppercase.PY"].is_parseable
    assert not cache["broken.py"].is_parseable


def test_build_python_ast_cache_includes_all_python_execution_surfaces() -> None:
    cache = build_python_ast_cache(
        [
            "window.pyw",
            "runner",
            "bundle.zip!/nested-runner",
            "typing.pyi",
            "node-runner",
        ],
        {
            "window.pyw": "value = 1\n",
            "runner": "#!/usr/bin/env python3\nvalue = 2\n",
            "bundle.zip!/nested-runner": "#!/usr/bin/python3\nvalue = 3\n",
            "typing.pyi": "value: int\n",
            "node-runner": "#!/usr/bin/env node python3\nvalue = 4\n",
        },
    )

    assert set(cache) == {"window.pyw", "runner", "bundle.zip!/nested-runner"}
    assert all(parsed.is_parseable for parsed in cache.values())


def test_build_python_ast_cache_respects_aggregate_source_budget() -> None:
    cache = build_python_ast_cache(
        ["first.py", "second.py"],
        {
            "first.py": "x = 1\n",
            "second.py": "x = 2\n",
        },
        max_cache_source_chars=8,
    )

    assert set(cache) == {"first.py"}


def test_build_python_ast_cache_checks_deadline_before_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_classification(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("classification must not start after the deadline")

    monkeypatch.setattr(python_ast, "may_be_python_source", forbidden_classification)
    limitations: list[tuple[str, float]] = []

    cache = build_python_ast_cache(
        ["runner"],
        {"runner": "#!/usr/bin/env python3\npass\n"},
        clock=lambda: 1.0,
        started_at=0.0,
        deadline=0.5,
        runtime_limitations=limitations,
    )

    assert cache == {}
    assert limitations == [("runner", 1.0)]


def test_get_python_ast_reparses_when_cached_source_changes() -> None:
    cache_key = prewarm_python_ast_cache(["script.py"], {"script.py": "import os as old_name\n"})
    assert cache_key is not None

    parsed = get_python_ast(cache_key, "import os as new_name\n", "script.py")

    assert parsed.import_aliases == {"new_name": "os"}
    clear_python_ast_cache(cache_key)


def test_runtime_ast_cache_registry_is_bounded_for_checkpoint_cache_misses() -> None:
    cache_keys = [
        f"resumed-scan-{index}" for index in range(python_ast._MAX_RUNTIME_AST_CACHES + 5)
    ]

    for cache_key in cache_keys:
        get_python_ast(cache_key, "x = 1\n", "script.py")

    assert len(python_ast._runtime_ast_caches) <= python_ast._MAX_RUNTIME_AST_CACHES

    for cache_key in cache_keys:
        clear_python_ast_cache(cache_key)
