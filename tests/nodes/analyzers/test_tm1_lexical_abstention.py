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


def test_positive_dataflow_without_retained_finding_keeps_lexical_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "import subprocess\nenabled = True\nsubprocess.run(command, shell=enabled)\n"
    monkeypatch.setattr(truthiness, "analyze", lambda *args, **kwargs: [])
    findings = _findings(source)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


@pytest.mark.parametrize("replacement", ["enabled = False", "subprocess = proxy"])
def test_uncalled_deferred_global_store_is_not_counterevidence(replacement: str) -> None:
    name = replacement.partition(" =")[0]
    source = (
        "payload = input()\nimport subprocess\nenabled = True\n"
        f"def disable():\n    global {name}\n    {replacement}\n"
        "subprocess.run(command, shell=enabled)\n"
    )
    findings = _findings(source)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


@pytest.mark.parametrize(
    "mutation", ["subprocess.label = 'documentation'", "subprocess.check_call = proxy"]
)
def test_unrelated_receiver_attribute_is_not_counterevidence(mutation: str) -> None:
    source = (
        f"import subprocess\n{mutation}\nenabled = True\nsubprocess.run(command, shell=enabled)\n"
    )
    findings = _findings(source)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


def test_reimport_restores_method_after_mutating_a_replaced_receiver() -> None:
    source = (
        "subprocess = proxy\nsubprocess.run = proxy.run\nimport subprocess\n"
        "enabled = True\nsubprocess.run(command, shell=enabled)\n"
    )
    findings = _findings(source)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


@pytest.mark.parametrize(
    "assignment",
    [
        "subprocess.run = subprocess.run",
        "subprocess.run: object = subprocess.run",
        "subprocess.run = saved = subprocess.run",
        "saved = subprocess.run = subprocess.run",
        "subprocess.run, saved = subprocess.run, 1",
        "saved, [subprocess.run] = 1, [subprocess.run]",
        "subprocess.run, subprocess.run = proxy, subprocess.run",
        "[subprocess.run, subprocess.run] = [proxy, subprocess.run]",
        "subprocess.run = [subprocess.run] = [subprocess.run]",
    ],
)
@pytest.mark.parametrize("intervening", ["", "helper()\n"])
def test_called_method_self_store_preserves_lexical_finding(
    assignment: str, intervening: str
) -> None:
    source = (
        f"import subprocess\nenabled = True\n{assignment}\n"
        f"{intervening}subprocess.run(command, shell=enabled)\n"
    )
    findings = _findings(source)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


@pytest.mark.parametrize(
    ("intervening", "expected"),
    [("import subprocess\n", False), ("helper()\nimport subprocess\n", True)],
)
def test_reimport_does_not_prove_slot_replacement_after_unknown_effect(
    intervening: str, expected: bool
) -> None:
    source = (
        "import subprocess\nsubprocess.run = proxy\n"
        f"{intervening}enabled = True\nsubprocess.run(command, shell=enabled)\n"
    )
    findings = _findings(source)
    assert bool(findings) is expected
    if expected:
        assert findings[0].severity == "HIGH"


def test_later_fresh_import_restores_deferred_receiver_signal() -> None:
    source = (
        "import subprocess\ndef run():\n    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        "subprocess = proxy\nimport subprocess\nrun()\n"
    )
    findings = _findings(source)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


@pytest.mark.parametrize(
    "annotation", ["enabled: bool", "subprocess: object", "subprocess.run: object"]
)
def test_annotation_without_value_is_not_runtime_counterevidence(annotation: str) -> None:
    source = (
        f"import subprocess\nenabled = True\n{annotation}\nsubprocess.run(command, shell=enabled)\n"
    )
    findings = _findings(source)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


@pytest.mark.parametrize(
    "replacement",
    ["enabled = False\nenabled = enabled", "subprocess = proxy\nsubprocess = subprocess"],
)
def test_self_store_does_not_undo_observed_replacement(replacement: str) -> None:
    source = f"import subprocess\nenabled = True\n{replacement}\nsubprocess.run(command, shell=enabled)\n"
    assert not _findings(source)


def test_future_module_method_replacement_suppresses_deferred_call() -> None:
    source = (
        "import subprocess\ndef run():\n    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\nsubprocess.run = proxy\nrun()\n"
    )
    assert not _findings(source)


def test_later_method_store_in_same_function_preserves_earlier_call() -> None:
    source = (
        "import subprocess\ndef run():\n    helper()\n    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n    subprocess.run = proxy\n"
    )
    findings = _findings(source)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


def test_local_import_restores_module_after_outer_proxy_slot_mutation() -> None:
    source = (
        "subprocess = proxy\nsubprocess.run = handler\ndef run(command):\n"
        "    import subprocess\n    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
    )
    findings = _findings(source)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


def test_later_class_proxy_store_keeps_cached_replacement_unknown() -> None:
    source = (
        "import subprocess\nclass Tool:\n    subprocess.run = proxy\n"
        "    subprocess = other\n    subprocess.run = other\n    def run(self):\n"
        "        enabled = True\n        subprocess.run(command, shell=enabled)\n"
    )
    # The later write on an unknown proxy may have protocol side effects.
    assert len(_findings(source)) == 1


@pytest.mark.parametrize("local_import", [False, True])
def test_bare_popen_import_captures_cached_module_slot(local_import: bool) -> None:
    prefix = "import subprocess\nsubprocess.Popen = proxy\n"
    body = "from subprocess import Popen\nenabled = True\nPopen(command, shell=enabled)\n"
    source = prefix + (
        "def run():\n" + "".join("    " + line + "\n" for line in body.splitlines())
        if local_import
        else body
    )
    assert not _findings(source)


def test_bare_popen_import_before_module_mutation_keeps_captured_callable() -> None:
    source = (
        "import subprocess\nfrom subprocess import Popen\nsubprocess.Popen = proxy\n"
        "enabled = True\nPopen(command, shell=enabled)\n"
    )
    findings = _findings(source)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


@pytest.mark.parametrize(
    "assignment",
    ["subprocess.run = subprocess = proxy", "subprocess.run, subprocess = proxy, other"],
)
def test_class_assignment_target_order_keeps_earlier_module_mutation(assignment: str) -> None:
    source = (
        f"import subprocess\nclass Tool:\n    {assignment}\n    def run(self):\n"
        "        enabled = True\n        subprocess.run(command, shell=enabled)\n"
    )
    assert not _findings(source)


def test_class_assignment_target_order_ignores_later_proxy_slot_mutation() -> None:
    source = (
        "import subprocess\nclass Tool:\n    subprocess, subprocess.run = other, proxy\n"
        "    def run(self):\n        enabled = True\n        subprocess.run(command, shell=enabled)\n"
    )
    findings = _findings(source)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


@pytest.mark.parametrize(
    "statement",
    [
        "return Popen(command, shell=enabled)",
        "wrapper(Popen(command, shell=enabled))",
        "return [Popen(command, shell=enabled)]",
    ],
)
def test_local_popen_cached_replacement_in_eager_return_expression(statement: str) -> None:
    source = (
        "import subprocess\nsubprocess.Popen = proxy\ndef run():\n"
        "    from subprocess import Popen\n    enabled = True\n"
        f"    {statement}\nrun()\n"
    )
    assert not _findings(source)


@pytest.mark.parametrize("intervening", ["", "restore()\n"])
def test_unknown_helper_can_restore_a_changed_module_slot(intervening: str) -> None:
    source = (
        "import subprocess\nsubprocess.run = proxy\n"
        f"{intervening}enabled = True\nsubprocess.run(command, shell=enabled)\n"
    )
    findings = _findings(source)
    assert bool(findings) is bool(intervening)
    if findings:
        assert findings[0].severity == "HIGH"


@pytest.mark.parametrize("intervening", ["", "restore()\n", "import subprocess\n"])
def test_direct_effectful_slot_store_has_only_local_lifetime(intervening: str) -> None:
    source = (
        "subprocess.run = Proxy()\n"
        + intervening
        + "enabled = True\nsubprocess.run(command, shell=enabled)\n"
    )
    assert bool(_findings(source)) is bool(intervening)


@pytest.mark.parametrize(
    "statement",
    [
        "subprocess.run: restore() = Proxy()",
        "subprocess.run = other.slot = Proxy()",
        "other.slot = subprocess.run = Proxy()",
    ],
)
def test_composed_or_annotated_store_does_not_supply_local_slot_proof(statement: str) -> None:
    source = (
        "import subprocess\n"
        + statement
        + "\nenabled = True\nsubprocess.run(command, shell=enabled)\n"
    )
    assert len(_findings(source)) == 1


@pytest.mark.parametrize(
    "prefix",
    [
        "subprocess.run = Proxy()\n",
        "subprocess.__class__ = OtherModule\n",
    ],
)
def test_later_single_store_cannot_override_companion_abstention(prefix: str) -> None:
    source = (
        prefix + "subprocess.run = proxy\nenabled = True\nsubprocess.run(command, shell=enabled)\n"
    )
    assert len(_findings(source)) == 1


@pytest.mark.parametrize(
    "store",
    [
        pytest.param("False and (enabled := False)", id="and-flag"),
        pytest.param("False and (subprocess := None)", id="and-receiver"),
        pytest.param("True or (enabled := False)", id="or-flag"),
        pytest.param("ready and (enabled := False)", id="unknown-and-flag"),
        pytest.param("result = 1 if ready else (enabled := False)", id="ifexp-flag"),
        pytest.param("result = (subprocess := None) if False else 1", id="ifexp-receiver"),
        pytest.param("result = [(enabled := False) for _ in values]", id="listcomp-flag"),
        pytest.param("result = ((subprocess := None) for _ in values)", id="genexp-receiver"),
    ],
)
def test_untaken_expression_store_is_not_counterevidence(store: str) -> None:
    source = (
        f'import subprocess\nenabled = True\n{store}\nsubprocess.run("echo ok", shell=enabled)\n'
    )
    findings = _findings(source)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


@pytest.mark.parametrize(
    "store",
    [
        "True and (enabled := False)",
        "False or (subprocess := None)",
        "result = (enabled := False) if True else 1",
        "result = 1 if False else (subprocess := None)",
    ],
)
def test_executed_expression_store_removes_lexical_candidate(store: str) -> None:
    source = (
        f'import subprocess\nenabled = True\n{store}\nsubprocess.run("echo ok", shell=enabled)\n'
    )
    assert not _findings(source)


@pytest.mark.parametrize(
    ("prefix", "call"),
    [
        pytest.param(
            "import subprocess\nsaved = subprocess\nsubprocess = saved\n",
            'subprocess.run("echo ok", shell=enabled)',
            id="native-alias-round-trip",
        ),
        pytest.param(
            "import subprocess as saved\nimport subprocess\nsubprocess = saved\n",
            'subprocess.run("echo ok", shell=enabled)',
            id="native-import-alias",
        ),
        pytest.param(
            "import subprocess\nsubprocess, saved = subprocess, 1\n",
            'subprocess.run("echo ok", shell=enabled)',
            id="unpacked-receiver-self-store",
        ),
        pytest.param(
            "import subprocess\n[subprocess, saved] = [subprocess, 1]\n",
            'subprocess.run("echo ok", shell=enabled)',
            id="list-unpacked-receiver-self-store",
        ),
        pytest.param(
            "from subprocess import Popen\nsaved = Popen\nPopen = saved\n",
            'Popen("echo ok", shell=enabled)',
            id="native-popen-alias",
        ),
        pytest.param(
            "import subprocess\nfrom subprocess import Popen\nPopen = subprocess.Popen\n",
            'Popen("echo ok", shell=enabled)',
            id="native-popen-attribute",
        ),
    ],
)
def test_native_identity_store_keeps_lexical_high(prefix: str, call: str) -> None:
    source = f"{prefix}enabled = True\n{call}\n"
    findings = _findings(source)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


@pytest.mark.parametrize(
    "store",
    [
        "enabled, ignored = True, 1",
        "[enabled, ignored] = [True, 1]",
        "enabled, ignored = 'yes', 1",
        "(enabled := True)",
        "enabled = 1",
    ],
)
def test_truthy_flag_store_keeps_lexical_high(store: str) -> None:
    source = (
        f'import subprocess\nenabled = True\n{store}\nsubprocess.run("echo ok", shell=enabled)\n'
    )
    findings = _findings(source)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


@pytest.mark.parametrize(
    "store",
    [
        "enabled, ignored = False, 1",
        "enabled, ignored = dynamic, 1",
        "subprocess, saved = proxy, 1",
        "saved = proxy\nsubprocess = saved",
        "saved = subprocess\nsaved = proxy\nsubprocess = saved",
    ],
)
def test_unpacked_or_aliased_replacement_removes_lexical_candidate(store: str) -> None:
    source = (
        f'import subprocess\nenabled = True\n{store}\nsubprocess.run("echo ok", shell=enabled)\n'
    )
    assert not _findings(source)


_DEFERRED_RUN = (
    "import subprocess\ndef run(command):\n    enabled = True\n"
    "    subprocess.run(command, shell=enabled)\n"
)


@pytest.mark.parametrize(
    "invocation",
    [
        pytest.param('if True:\n    run("echo ok")\n', id="literal-branch"),
        pytest.param('if __name__ == "__main__":\n    run("echo ok")\n', id="main-guard"),
        pytest.param('try:\n    run("echo ok")\nfinally:\n    pass\n', id="try"),
        pytest.param('for _ in range(1):\n    run("echo ok")\n', id="loop"),
        pytest.param('with context:\n    run("echo ok")\n', id="with"),
        pytest.param(
            'threading.Thread(target=run, args=("echo ok",)).start()\n', id="thread-start"
        ),
        pytest.param('def main():\n    run("echo ok")\nmain()\n', id="nested-caller"),
    ],
)
@pytest.mark.parametrize("replacement", ["subprocess = None", "subprocess = proxy"])
def test_invocation_before_future_store_keeps_lexical_high(
    invocation: str, replacement: str
) -> None:
    source = f"{_DEFERRED_RUN}{invocation}{replacement}\n"
    findings = _findings(source)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


def test_method_invoked_before_future_store_keeps_lexical_high() -> None:
    source = (
        "import subprocess\nclass Tool:\n    def run(self, command):\n        enabled = True\n"
        '        subprocess.run(command, shell=enabled)\nif True:\n    Tool().run("echo ok")\n'
        "subprocess = None\n"
    )
    findings = _findings(source)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


def test_reimport_before_invocation_restores_deferred_receiver() -> None:
    source = (
        "import subprocess\nsubprocess = proxy\n"
        f"{_DEFERRED_RUN.removeprefix('import subprocess')}"
        'import subprocess\nif True:\n    run("echo ok")\n'
    )
    findings = _findings(source)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


@pytest.mark.parametrize(
    "tail",
    [
        pytest.param('subprocess = None\nrun("echo ok")\n', id="store-before-call"),
        pytest.param('subprocess = None\nif True:\n    run("echo ok")\n', id="store-before-branch"),
        pytest.param("subprocess = None\n", id="never-called-in-module"),
    ],
)
def test_store_before_every_invocation_removes_deferred_candidate(tail: str) -> None:
    assert not _findings(f"{_DEFERRED_RUN}{tail}")
