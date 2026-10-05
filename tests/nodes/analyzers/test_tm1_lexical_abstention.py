# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lexical TM1 remains a signal when conservative Python dataflow abstains."""

from __future__ import annotations

import pytest

from skillspector.nodes.analyzers import static_patterns_tool_misuse as tool_misuse
from skillspector.nodes.analyzers import static_python_shell_truthiness as truthiness
from skillspector.python_ast import parse_python_source


def _findings(source: str) -> list:
    response = tool_misuse.node({"components": ["run.py"], "file_cache": {"run.py": source}})
    return [finding for finding in response["findings"] if finding.rule_id == "TM1"]


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "import subprocess\nif True:\n    enabled = True\n    subprocess.run(command, shell=enabled)\n",
            id="literal-compound",
        ),
        pytest.param(
            'import subprocess\ndef run(command):\n    enabled = True\n    subprocess.run(command, shell=enabled)\nif __name__ == "__main__":\n    run("ls")\n',
            id="ordinary-main-guard",
        ),
        pytest.param(
            'import subprocess\ncommand = f"python {sys.argv[1]}"\nenabled = True\nsubprocess.run(command, shell=enabled)\n',
            id="formatted-command-before-binding",
        ),
        pytest.param(
            "import subprocess\ndef run(command):\n    enabled = True\n    return subprocess.run(command, shell=enabled)\n",
            id="return-call",
        ),
        pytest.param(
            "import subprocess\nif False:\n    enabled = False\nenabled = True\nsubprocess.run(command, shell=enabled)\n",
            id="untaken-flag-store",
        ),
        pytest.param(
            "import subprocess\nenabled = True\nif False:\n    subprocess = proxy\nsubprocess.run(command, shell=enabled)\n",
            id="untaken-receiver-store",
        ),
        pytest.param(
            "import subprocess\nenabled = True\nfor subprocess in values:\n    pass\nsubprocess.run(command, shell=enabled)\n",
            id="possibly-empty-receiver-loop",
        ),
        pytest.param(
            "import subprocess\nenabled = True\nunknown_effect()\nsubprocess = subprocess\nsubprocess.run(command, shell=enabled)\n",
            id="self-store-retains-binding",
        ),
        pytest.param(
            "import subprocess\nenabled = True\nunknown_effect()\nenabled: bool = True\nsubprocess.run(command, shell=enabled)\n",
            id="annotated-true-after-unknown",
        ),
    ],
)
def test_companion_abstention_keeps_lexical_high(source: str) -> None:
    findings = _findings(source)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"
    assert "shell=enabled" in (findings[0].matched_text or "")
    assert not findings[0].evidence


@pytest.mark.parametrize(
    "prefix",
    ["payload = input()", "sys.path.insert(0, path)", "load_dotenv()", "ignored = (helper(),)"],
)
@pytest.mark.parametrize("placement", ["module", "function"])
def test_fresh_import_after_unknown_effect_keeps_lexical_high(prefix: str, placement: str) -> None:
    source = f"import os\n{prefix}\nimport subprocess\n"
    if placement == "function":
        source += (
            "def run(command):\n    enabled = True\n    subprocess.run(command, shell=enabled)\n"
        )
    else:
        source += "enabled = True\nsubprocess.run(command, shell=enabled)\n"
    assert len(_findings(source)) == 1
    assert _findings(source)[0].severity == "HIGH"


@pytest.mark.parametrize(
    "effect",
    [
        "helper()",
        "result = helper()",
        "result = (helper(),)",
        "result: object = [helper()]",
        "assert (helper(),)",
        "result = f'{value}'",
        "[value for value in values]",
    ],
)
@pytest.mark.parametrize("placement", ["module", "function", "class", "deferred"])
def test_generic_eager_effect_does_not_prove_receiver_replacement(
    effect: str, placement: str
) -> None:
    body = f"{effect}\nenabled = True\nsubprocess.run(command, shell=enabled)\n"
    if placement in {"function", "class"}:
        header = "def run():\n" if placement == "function" else "class Run:\n"
        source = (
            "import subprocess\n"
            + header
            + "".join("    " + line + "\n" for line in body.splitlines())
        )
    elif placement == "deferred":
        source = (
            "import subprocess\ndef run():\n    enabled = True\n    subprocess.run(command, shell=enabled)\n"
            + effect
            + "\nrun()\n"
        )
    else:
        source = "import subprocess\n" + body
    # Preserve conservative receiver invalidation in the AST companion itself.
    assert not truthiness.analyze(source, "run.py", "python")
    findings = _findings(source)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


@pytest.mark.parametrize(
    "replacement",
    [
        "enabled = False",
        "enabled = dynamic",
        "del enabled",
        "import settings as enabled",
        "from settings import enabled",
        "enabled: bool = False",
    ],
)
def test_observed_flag_replacement_removes_stale_lexical_binding(replacement: str) -> None:
    source = f"import subprocess\nenabled = True\n{replacement}\nsubprocess.run(command, shell=enabled)\n"
    assert not _findings(source)


@pytest.mark.parametrize(
    "replacement",
    [
        "subprocess = proxy",
        "subprocess = True",
        "subprocess.run = proxy",
        "subprocess['run'] = proxy",
        "del subprocess",
        "import settings as subprocess",
    ],
)
def test_observed_receiver_replacement_removes_lexical_candidate(replacement: str) -> None:
    source = f"import subprocess\n{replacement}\nenabled = True\nsubprocess.run(command, shell=enabled)\n"
    assert not _findings(source)


def test_fresh_direct_import_supersedes_explicit_receiver_replacement() -> None:
    source = "subprocess = proxy\nhelper()\nimport subprocess\nenabled = True\nsubprocess.run(command, shell=enabled)\n"
    findings = _findings(source)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


def test_positive_companion_decision_has_one_owner() -> None:
    source = "import subprocess\nenabled = True\nsubprocess.run(command, shell=enabled)\n"
    parsed = parse_python_source(source, "run.py")
    ownership, emitted = truthiness.bound_shell_call_state("run.py", parsed)
    assert len(emitted) == 1
    assert next(iter(ownership.values())) is True
    assert len(_findings(source)) == 1


def test_unknown_companion_decision_is_distinct_from_emission() -> None:
    source = "import subprocess\nhelper()\nenabled = True\nsubprocess.run(command, shell=enabled)\n"
    ownership, emitted = truthiness.bound_shell_call_state(
        "run.py", parse_python_source(source, "run.py")
    )
    assert next(iter(ownership.values())) is False
    assert not emitted
    assert len(_findings(source)) == 1
