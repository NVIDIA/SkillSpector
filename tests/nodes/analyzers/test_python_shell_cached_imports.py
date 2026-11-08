# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cached module slots retain their state across imports in deferred scopes."""

from __future__ import annotations

import pytest

from skillspector.nodes.analyzers import static_python_shell_truthiness as truthiness
from skillspector.python_ast import parse_python_source


@pytest.mark.parametrize(
    "source",
    [
        "import subprocess\nsubprocess.run = proxy\n"
        "def run():\n    import subprocess\n    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\nrun()\n",
        "import subprocess\nsubprocess.Popen = proxy\n"
        "def run():\n    from subprocess import Popen\n    enabled = True\n"
        "    Popen(command, shell=enabled)\nrun()\n",
        "import subprocess\ndef outer():\n    subprocess.run = proxy\n"
        "    def inner():\n        import subprocess\n        enabled = True\n"
        "        subprocess.run(command, shell=enabled)\n    inner()\nouter()\n",
        "import subprocess\nsubprocess.run = proxy\nclass Tool:\n"
        "    def run(self):\n        import subprocess\n        enabled = True\n"
        "        subprocess.run(command, shell=enabled)\n",
        "import subprocess\nclass Tool:\n    def alter(self):\n"
        "        global subprocess\n        subprocess = proxy\n"
        "subprocess.run = handler\ndef run():\n    import subprocess\n    enabled = 1\n"
        "    subprocess.run(command, shell=enabled)\n",
    ],
)
def test_local_import_does_not_restore_changed_cached_module_slot(source: str) -> None:
    assert truthiness.analyze(source, "run.py", "python") == []


@pytest.mark.parametrize(
    "source",
    [
        "import subprocess\ndef run():\n    import subprocess\n    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\nrun()\nsubprocess.run = proxy\n",
        "import subprocess\ndef alter():\n    subprocess.run = proxy\n"
        "def run():\n    import subprocess\n    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\nrun()\n",
        "subprocess = proxy\nsubprocess.run = proxy.run\ndef run():\n"
        "    import subprocess\n    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\nrun()\n",
        "import subprocess\nsubprocess.Popen = proxy\ndef run():\n"
        "    import subprocess\n    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\nrun()\n",
        "import subprocess\ndef outer(subprocess):\n    subprocess.run = proxy\n"
        "    def inner():\n        import subprocess\n        enabled = 1\n"
        "        subprocess.run(command, shell=enabled)\n    inner()\n",
        "import subprocess\nclass Tool:\n    global subprocess\n    subprocess = proxy\n"
        "subprocess.run = handler\ndef run():\n    import subprocess\n    enabled = 1\n"
        "    subprocess.run(command, shell=enabled)\n",
        "import subprocess\nclass Outer:\n    class Inner:\n"
        "        global subprocess\n        subprocess = proxy\n"
        "subprocess.run = handler\ndef run():\n    import subprocess\n    enabled = 1\n"
        "    subprocess.run(command, shell=enabled)\n",
        "import subprocess\nsubprocess.run = (subprocess := proxy)\nimport subprocess\n"
        "def run():\n    import subprocess\n    enabled = 1\n"
        "    subprocess.run(command, shell=enabled)\nrun()\n",
        "import subprocess\ndef replace_receiver():\n    global subprocess\n"
        "    subprocess = proxy\n    return handler\n"
        "subprocess.run = replace_receiver()\nimport subprocess\ndef run():\n"
        "    import subprocess\n    enabled = 1\n"
        "    subprocess.run(command, shell=enabled)\nrun()\n",
    ],
)
def test_cached_module_tracking_preserves_observed_and_unaffected_calls(source: str) -> None:
    findings = truthiness.analyze(source, "run.py", "python")
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


@pytest.mark.parametrize(
    ("source", "replacement"),
    [
        (
            "import subprocess\nsubprocess.run = proxy\nenabled = True\n"
            "subprocess.run(command, shell=enabled)\n",
            True,
        ),
        (
            "import subprocess\nsubprocess.Popen = proxy\nfrom subprocess import Popen\n"
            "enabled = True\nPopen(command, shell=enabled)\n",
            True,
        ),
        (
            "from subprocess import Popen\nimport subprocess\nsubprocess.Popen = proxy\n"
            "enabled = True\nPopen(command, shell=enabled)\n",
            False,
        ),
        (
            "import subprocess\nhelper()\nenabled = True\nsubprocess.run(command, shell=enabled)\n",
            False,
        ),
        (
            "import subprocess\ndef run(subprocess, command):\n"
            "    subprocess.run = proxy\n    import subprocess\n    enabled = 1\n"
            "    subprocess.run(command, shell=enabled)\n",
            False,
        ),
        (
            "import subprocess\nsubprocess.run = proxy\ntrue_flag = True\n"
            "subprocess.run(command, shell=true_flag)\n",
            True,
        ),
        (
            "import subprocess\ndef run():\n    import subprocess\n    enabled = True\n"
            "    subprocess.run(command, shell=enabled)\nrun()\n"
            "subprocess.run = proxy\nrun()\n",
            False,
        ),
    ],
)
def test_cached_replacement_is_separate_from_unknown_receiver_trust(
    source: str, replacement: bool
) -> None:
    parsed = parse_python_source(source, "run.py")
    ownership, emitted, replacements = truthiness.bound_shell_call_analysis("run.py", parsed)
    assert len(ownership) == 1
    assert bool(replacements) is replacement
    assert replacements.issubset(ownership)
    assert truthiness.bound_shell_call_state("run.py", parsed) == (ownership, emitted)


def test_cached_replacement_analysis_keeps_invalid_python_empty() -> None:
    parsed = parse_python_source("def incomplete(", "run.py")
    assert truthiness.bound_shell_call_analysis("run.py", parsed) == ({}, set(), set())


@pytest.mark.parametrize(
    "prefix",
    [
        "box[redirect()] = subprocess.run = handler\n",
        "def unused(default=(subprocess := proxy)):\n    pass\nsubprocess.run = handler\n",
        "class Tool((subprocess := proxy)):\n    pass\nsubprocess.run = handler\n",
        "original = subprocess.run\ndef restore():\n    subprocess.run = original\n"
        "subprocess.run = handler\nrestore()\n",
        "original = subprocess.run\ndef restore():\n    subprocess.run = original\n"
        "subprocess.run = handler\ndef unused(default=restore()):\n    pass\n",
        "subprocess.run = proxy\nsubprocess.__class__ = Restoring\n",
        "subprocess.__class__ = Restoring\nimport subprocess\nsubprocess.run = proxy\n",
        "restorer = Untrusted()\nimport subprocess\nsubprocess.run = proxy\ndel restorer\n",
        "subprocess.run = proxy\ndel subprocess.run\n",
        "subprocess.run = proxy\nsubprocess.run = handler\n",
        "subprocess.run = proxy\nimport restore_module\n",
        "subprocess.run = proxy\nfrom restore_module import restored\n",
        "subprocess.run = proxy\nimport subprocess, restore_module\n",
        "restorer = Untrusted()\nimport subprocess\nsubprocess.run = proxy\nrestorer = 0\n",
    ],
)
def test_unknown_eager_effect_invalidates_prior_replacement_proof(prefix: str) -> None:
    source = (
        "import subprocess\n" + prefix + "import subprocess\ndef work():\n"
        "    import subprocess\n    enabled = 1\n"
        "    subprocess.run(command, shell=enabled)\nwork()\n"
    )
    assert len(truthiness.analyze(source, "run.py", "python")) == 1
    _, _, replacements = truthiness.bound_shell_call_analysis(
        "run.py", parse_python_source(source, "run.py")
    )
    assert not replacements


@pytest.mark.parametrize(
    ("expression", "replacement"),
    [
        ("return Popen(command, shell=enabled)", True),
        ("wrapper(Popen(command, shell=enabled))", True),
        ("return [Popen(command, shell=enabled)]", True),
        ("return wrapper(helper(), Popen(command, shell=enabled))", False),
        ("return lambda: Popen(command, shell=enabled)", False),
        ("return (Popen(command, shell=enabled) for item in items)", False),
    ],
)
def test_cached_replacement_proof_respects_eager_expression_order(
    expression: str, replacement: bool
) -> None:
    source = (
        "import subprocess\nsubprocess.Popen = proxy\ndef run():\n"
        "    from subprocess import Popen\n    enabled = True\n"
        f"    {expression}\n"
    )
    _, emitted, replacements = truthiness.bound_shell_call_analysis(
        "run.py", parse_python_source(source, "run.py")
    )
    assert bool(replacements) is replacement
    assert not emitted


@pytest.mark.parametrize(
    "source",
    [
        "restorer = 0\nimport subprocess\nclass Tool:\n"
        "    helper()\n    import subprocess\n    subprocess.run = proxy\n"
        "restorer = 0\nenabled = True\nsubprocess.run(command, shell=enabled)\n",
        "helper()\nimport subprocess\nsubprocess.run = proxy\nrestorer = 0\n"
        "enabled = True\nsubprocess.run(command, shell=enabled)\n",
        "import subprocess\nenabled = True\nhelper()\nimport subprocess\n"
        "subprocess.run = proxy\nsubprocess.run(command, shell=enabled)\n",
        "import subprocess\nsubprocess.run = proxy\ndef work():\n"
        "    import subprocess\n    enabled = 1\n"
        "    subprocess.run(command, shell=enabled)\nwork()\nhelper()\n"
        "import subprocess\nsubprocess.run = proxy\nwork()\n",
    ],
    ids=["class-outer-finalizer", "unknown-new-binding", "unknown-protocol", "later-observation"],
)
def test_unknown_effect_cannot_reestablish_cached_replacement_proof(source: str) -> None:
    # Reimporting the same cached module does not establish that an unknown
    # earlier effect left its lookup protocol or other finalizer bindings intact.
    ownership, _, replacements = truthiness.bound_shell_call_analysis(
        "run.py", parse_python_source(source, "run.py")
    )
    assert len(ownership) == 1
    assert not replacements


@pytest.mark.parametrize(
    ("prefix", "method"),
    [
        ("original = subprocess.run\nsubprocess.run = original\n", "run"),
        ("subprocess.run = subprocess.check_call\n", "run"),
        (
            "native_module = subprocess\noriginal = native_module.run\nsubprocess.run = original\n",
            "run",
        ),
        (
            "import subprocess as native_module\nsubprocess.run = native_module.check_output\n",
            "run",
        ),
        ("original, = (subprocess.run,)\nsubprocess.run = original\n", "run"),
        ("[original] = [subprocess.run]\nsubprocess.run = original\n", "run"),
        ("from subprocess import Popen\nsubprocess.run = Popen\n", "run"),
        ("from subprocess import Popen\noriginal = Popen\nsubprocess.run = original\n", "run"),
        ("subprocess.Popen = subprocess.run\n", "Popen"),
    ],
)
def test_native_callable_value_does_not_prove_a_custom_replacement(
    prefix: str, method: str
) -> None:
    source = (
        "import subprocess\n" + prefix + "def work():\n"
        "    import subprocess\n    enabled = True\n"
        f"    subprocess.{method}(command, shell=enabled)\nwork()\n"
    )
    assert len(truthiness.analyze(source, "run.py", "python")) == 1
    _, _, replacements = truthiness.bound_shell_call_analysis(
        "run.py", parse_python_source(source, "run.py")
    )
    assert not replacements
