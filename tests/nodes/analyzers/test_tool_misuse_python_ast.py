# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused coverage for issue #475's ordinary-Python binding form."""

from __future__ import annotations

import pytest

from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm_module
from skillspector.nodes.analyzers import static_python_shell_truthiness as python_tm_module
from skillspector.nodes.deduplicate import deduplicate


def _run(content: str, path: str = "run.py") -> dict:
    return tm_module.node({"components": [path], "file_cache": {path: content}})


def _tm1(content: str, path: str = "run.py") -> list:
    return [finding for finding in _run(content, path)["findings"] if finding.rule_id == "TM1"]


def _tm1_ast(content: str, path: str = "run.py") -> list:
    return python_tm_module.analyze(content, path, "python")


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
        f"enabled = {value}\n"
        "result = subprocess.run(\n"
        "    command,\n"
        "    shell=enabled,\n"
        "    capture_output=True,\n"
        "    text=True,\n"
        ")\n"
    )

    assert len(findings) == 1
    assert findings[0].start_line == 4
    assert findings[0].severity == "HIGH"
    assert findings[0].confidence == pytest.approx(0.9)
    assert "shell=enabled" in findings[0].matched_text
    assert not findings[0].evidence


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("1", id="integer"),
        pytest.param("-1", id="negative-integer"),
        pytest.param("(0,)", id="nonempty-tuple"),
        pytest.param("not False", id="negation"),
    ],
)
def test_simple_immutable_truthy_values_are_tracked(value: str) -> None:
    assert (
        len(_tm1(f"import subprocess\nenabled = {value}\nsubprocess.run(cmd, shell=enabled)\n"))
        == 1
    )


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("False", id="boolean"),
        pytest.param("0", id="integer"),
        pytest.param("''", id="string"),
        pytest.param("()", id="tuple"),
        pytest.param("None", id="none"),
    ],
)
def test_definitely_false_values_are_not_tracked(value: str) -> None:
    assert not _tm1(f"import subprocess\nenabled = {value}\nsubprocess.run(cmd, shell=enabled)\n")


def test_simple_alias_chain_and_bare_popen_are_tracked() -> None:
    findings = _tm1(
        "first = 'enabled'\nsecond = first\nthird = second\nPopen(command, shell=third)\n"
    )

    assert len(findings) == 1
    assert findings[0].start_line == 4


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


def test_import_side_effect_boundary_clears_truth_facts() -> None:
    assert not _tm1(
        "import subprocess\nenabled = True\nimport attacker\n"
        "subprocess.run(command, shell=enabled)\n"
    )


@pytest.mark.parametrize(
    "shadow",
    [
        pytest.param("subprocess = Proxy()", id="assignment"),
        pytest.param("import other as subprocess", id="import-alias"),
        pytest.param("for subprocess in values:\n    pass", id="compound-binder"),
        pytest.param("subprocess.run = Proxy()", id="attribute-mutation"),
        pytest.param("subprocess, other = pair", id="unpacking"),
    ],
)
def test_explicit_subprocess_shadow_rejects_bound_call(shadow: str) -> None:
    assert not _tm1(f"{shadow}\nenabled = True\nsubprocess.run(cmd, shell=enabled)\n")


def test_explicit_import_reestablishes_direct_receivers() -> None:
    assert (
        len(
            _tm1(
                "subprocess = Proxy()\n"
                "Popen = Proxy()\n"
                "import subprocess\n"
                "from subprocess import Popen\n"
                "enabled = True\n"
                "subprocess.run('/usr/bin/true', shell=enabled)\n"
                "Popen('/usr/bin/true', shell=enabled)\n"
            )
        )
        == 2
    )


def test_relative_import_does_not_establish_bare_popen() -> None:
    assert not _tm1(
        "Popen = proxy\nfrom .subprocess import Popen\nenabled = True\n"
        "Popen(command, shell=enabled)\n"
    )


def test_function_local_binding_and_outer_fact_are_independent() -> None:
    findings = _tm1(
        "outer = True\n"
        "def execute(command):\n"
        "    enabled = 'True'\n"
        "    subprocess.run(command, shell=enabled)\n"
        "subprocess.run(command, shell=outer)\n"
    )

    assert [finding.start_line for finding in findings] == [4, 5]


def test_function_compile_time_receiver_shadow_rejects_earlier_lookup() -> None:
    assert not _tm1(
        "def execute(command):\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        "    subprocess = Proxy()\n"
    )


def test_later_global_receiver_mutation_does_not_suppress_earlier_function_call() -> None:
    findings = _tm1(
        "def execute(command):\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        "execute(command)\n"
        "subprocess = Proxy()\n"
    )

    assert [finding.start_line for finding in findings] == [3]


def test_later_global_receiver_mutation_suppresses_unobserved_function_body() -> None:
    assert not _tm1(
        "def execute(command):\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        "subprocess = Proxy()\n"
    )


def test_passive_function_definition_preserves_outer_fact() -> None:
    assert (
        len(
            _tm1(
                "enabled = True\n"
                "def helper(value=1):\n"
                "    pass\n"
                "subprocess.run(command, shell=enabled)\n"
            )
        )
        == 1
    )


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
def test_compound_statement_conservatively_clears_truth_facts(compound: str) -> None:
    assert not _tm1(f"enabled = True\n{compound}\nsubprocess.run(command, shell=enabled)\n")


def test_calls_inside_compound_statements_are_out_of_scope() -> None:
    assert not _tm1(
        "if condition:\n    enabled = True\n    subprocess.run(command, shell=enabled)\n"
    )


def test_class_body_same_scope_binding_is_tracked() -> None:
    findings = _tm1(
        "class Runner:\n    enabled = True\n    subprocess.run(command, shell=enabled)\n"
    )

    assert [finding.start_line for finding in findings] == [3]


def test_generic_call_invalidates_receiver_trust_in_class_body() -> None:
    assert not _tm1(
        "class Runner:\n"
        "    replace_subprocess()\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
    )


def test_class_method_does_not_close_over_class_binding() -> None:
    assert not _tm1(
        "class Runner:\n"
        "    enabled = True\n"
        "    def run(self):\n"
        "        subprocess.run(command, shell=enabled)\n"
    )


def test_class_body_lookup_precedes_later_class_binding() -> None:
    findings = _tm1(
        "enabled = True\n"
        "class Runner:\n"
        "    subprocess.run(command, shell=enabled)\n"
        "    enabled = False\n"
    )

    assert [finding.start_line for finding in findings] == [1]


@pytest.mark.parametrize(
    "class_body",
    [
        pytest.param(
            "    enabled = subprocess.run(command, shell=enabled)\n",
            id="assignment-rhs-before-store",
        ),
        pytest.param(
            "    def enabled(value=subprocess.run(command, shell=enabled)):\n        pass\n",
            id="function-default-before-name-binding",
        ),
        pytest.param(
            "    with manager(subprocess.run(command, shell=enabled)) as enabled:\n        pass\n",
            id="with-target-after-context",
        ),
        pytest.param(
            "    try:\n"
            "        raise Error\n"
            "    except subprocess.run(command, shell=enabled) as enabled:\n"
            "        pass\n",
            id="except-target-after-type",
        ),
    ],
)
def test_class_binding_is_installed_after_value_evaluation(class_body: str) -> None:
    findings = _tm1("enabled = True\nclass Runner:\n" + class_body)

    assert [finding.start_line for finding in findings] == [1]


def test_class_local_import_does_not_establish_method_receiver_trust() -> None:
    assert not _tm1(
        "subprocess = proxy\n"
        "class Runner:\n"
        "    import subprocess\n"
        "    def run(self):\n"
        "        enabled = True\n"
        "        subprocess.run(command, shell=enabled)\n"
    )


def test_class_local_shadow_does_not_hide_global_method_receiver() -> None:
    findings = _tm1(
        "import subprocess\n"
        "class Runner:\n"
        "    subprocess = proxy\n"
        "    def run(self):\n"
        "        enabled = True\n"
        "        subprocess.run(command, shell=enabled)\n"
    )

    assert [finding.start_line for finding in findings] == [6]


def test_class_body_generic_call_invalidates_deferred_method_receiver() -> None:
    assert not _tm1(
        "import subprocess\n"
        "class Runner:\n"
        "    replace_subprocess()\n"
        "    def run(self):\n"
        "        enabled = True\n"
        "        subprocess.run(command, shell=enabled)\n"
    )


def test_called_class_method_preserves_receiver_before_later_invalidation() -> None:
    findings = _tm1(
        "import subprocess\n"
        "class Runner:\n"
        "    def run():\n"
        "        enabled = True\n"
        "        subprocess.run(command, shell=enabled)\n"
        "    run()\n"
        "    replace_subprocess()\n"
    )

    assert [finding.start_line for finding in findings] == [5]


def test_class_method_call_after_invalidation_is_rejected() -> None:
    assert not _tm1(
        "import subprocess\n"
        "class Runner:\n"
        "    def run():\n"
        "        enabled = True\n"
        "        subprocess.run(command, shell=enabled)\n"
        "    replace_subprocess()\n"
        "    run()\n"
    )


@pytest.mark.parametrize(
    "class_header",
    [
        pytest.param("@replace_subprocess()\nclass Runner:", id="decorator"),
        pytest.param("class Runner(replace_subprocess()):", id="base"),
    ],
)
def test_effectful_class_header_invalidates_deferred_method_receiver(class_header: str) -> None:
    assert not _tm1(
        "import subprocess\n"
        f"{class_header}\n"
        "    def run(self):\n"
        "        enabled = True\n"
        "        subprocess.run(command, shell=enabled)\n"
    )


def test_class_for_target_is_bound_after_iterable_evaluation() -> None:
    findings = _tm1(
        "enabled = True\n"
        "class Runner:\n"
        "    for enabled in subprocess.run(command, shell=enabled):\n"
        "        pass\n"
    )

    assert [finding.start_line for finding in findings] == [1]


def test_comprehension_target_shadows_outer_shell_flag() -> None:
    assert not _tm1(
        "enabled = True\n"
        "class Runner:\n"
        "    values = [\n"
        "        subprocess.run(command, shell=enabled)\n"
        "        for enabled in items\n"
        "    ]\n"
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
def test_side_effect_capable_call_arguments_are_rejected(argument: str) -> None:
    assert not _tm1(f"enabled = True\nsubprocess.run({argument}, shell=enabled)\n")


@pytest.mark.parametrize(
    "statement",
    [
        pytest.param(
            "subprocess.run(command, shell=enabled, env=build_env())",
            id="expression",
        ),
        pytest.param(
            "result = subprocess.run(command, shell=enabled, env=build_env())",
            id="assignment",
        ),
        pytest.param(
            "result: object = subprocess.run(command, shell=enabled, env=build_env())",
            id="annotated-assignment",
        ),
    ],
)
def test_later_keyword_effect_preserves_captured_shell_value(statement: str) -> None:
    findings = _tm1(f"enabled = True\n{statement}\n")
    literal_findings = _tm1(statement.replace("shell=enabled", "shell=True"))

    assert len(findings) == len(literal_findings) == 1
    assert findings[0].start_line == 2
    assert findings[0].severity == literal_findings[0].severity
    assert findings[0].confidence == literal_findings[0].confidence


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            "subprocess.run(command, env=build_env(), shell=enabled)",
            id="earlier-keyword",
        ),
        pytest.param(
            "subprocess.run(build_command(), shell=enabled)",
            id="earlier-positional",
        ),
        pytest.param(
            "subprocess.run(shell=enabled, *build_args())",
            id="starred-positional-written-later",
        ),
        pytest.param(
            "subprocess.run(command, **build_options(), shell=enabled)",
            id="earlier-keyword-expansion",
        ),
    ],
)
def test_earlier_argument_effect_keeps_shell_value_uncertain(call: str) -> None:
    assert not _tm1(f"enabled = True\n{call}\n")


def test_later_keyword_expansion_preserves_captured_shell_value() -> None:
    findings = _tm1("enabled = True\nsubprocess.run(command, shell=enabled, **build_options())\n")

    assert len(findings) == 1
    assert findings[0].start_line == 2


def test_later_argument_effect_invalidates_fact_after_captured_call() -> None:
    findings = _tm1(
        "enabled = True\n"
        "subprocess.run(command, shell=enabled, env=build_env())\n"
        "subprocess.run(command, shell=enabled)\n"
    )

    assert [finding.start_line for finding in findings] == [2]


@pytest.mark.parametrize(
    "statement",
    [
        pytest.param(
            "subprocess.run(command, shell=enabled, env=replace_subprocess())",
            id="expression",
        ),
        pytest.param(
            "result = subprocess.run(command, shell=enabled, env=replace_subprocess())",
            id="assignment",
        ),
        pytest.param(
            "result: object = subprocess.run(command, shell=enabled, env=replace_subprocess())",
            id="annotated-assignment",
        ),
    ],
)
def test_later_argument_effect_invalidates_receiver_after_captured_call(statement: str) -> None:
    findings = _tm1(
        "import subprocess\n"
        "from helpers import replace_subprocess\n"
        "enabled = True\n"
        f"{statement}\n"
        "later_enabled = True\n"
        "subprocess.run(command, shell=later_enabled)\n"
    )

    assert [finding.start_line for finding in findings] == [4]


def test_later_argument_effect_invalidates_receiver_for_called_function() -> None:
    findings = _tm1(
        "import subprocess\n"
        "from helpers import replace_subprocess\n"
        "enabled = True\n"
        "subprocess.run(command, shell=enabled, env=replace_subprocess())\n"
        "def execute():\n"
        "    later_enabled = True\n"
        "    subprocess.run(command, shell=later_enabled)\n"
        "execute()\n"
    )

    assert [finding.start_line for finding in findings] == [4]


@pytest.mark.parametrize(
    "statement",
    [
        pytest.param("replace_subprocess()", id="expression"),
        pytest.param("result = replace_subprocess()", id="assignment"),
        pytest.param("result: object = replace_subprocess()", id="annotated-assignment"),
    ],
)
def test_generic_call_invalidates_receiver_trust(statement: str) -> None:
    findings = _tm1(
        "import subprocess\n"
        "from helpers import replace_subprocess\n"
        f"{statement}\n"
        "enabled = True\n"
        "subprocess.run(command, shell=enabled)\n"
    )

    assert not findings


@pytest.mark.parametrize(
    "statement",
    [
        pytest.param("replace_subprocess()", id="expression"),
        pytest.param("result = replace_subprocess()", id="assignment"),
        pytest.param("result: object = replace_subprocess()", id="annotated-assignment"),
    ],
)
def test_generic_call_invalidates_receiver_trust_for_called_function(statement: str) -> None:
    findings = _tm1(
        "import subprocess\n"
        "from helpers import replace_subprocess\n"
        "def execute():\n"
        "    enabled = True\n"
        "    subprocess.run(command, shell=enabled)\n"
        f"{statement}\n"
        "execute()\n"
    )

    assert not findings


def test_generic_call_invalidates_receiver_trust_for_nested_closure() -> None:
    assert not _tm1(
        "import subprocess\n"
        "def outer():\n"
        "    enabled = True\n"
        "    def inner():\n"
        "        subprocess.run(command, shell=enabled)\n"
        "    replace_subprocess()\n"
        "    inner()\n"
        "outer()\n"
    )


def test_generic_call_invalidates_true_prefixed_nested_closure() -> None:
    assert not _tm1(
        "import subprocess\n"
        "def outer():\n"
        "    true_value = True\n"
        "    def inner():\n"
        "        subprocess.run(command, shell=true_value)\n"
        "    replace_subprocess()\n"
        "    inner()\n"
        "outer()\n"
    )


@pytest.mark.parametrize(
    "statement",
    [
        pytest.param(
            "subprocess.run('/usr/bin/true', shell=enabled)",
            id="expression",
        ),
        pytest.param(
            "result = subprocess.run('/usr/bin/true', shell=enabled)",
            id="assignment",
        ),
        pytest.param(
            "result: object = subprocess.run('/usr/bin/true', shell=enabled)",
            id="annotated-assignment",
        ),
    ],
)
def test_direct_subprocess_call_remains_detected(statement: str) -> None:
    findings = _tm1_ast(
        "import subprocess\n"
        "def execute():\n"
        "    later_enabled = True\n"
        "    subprocess.run('/usr/bin/true', shell=later_enabled)\n"
        "enabled = True\n"
        f"{statement}\n"
        "execute()\n"
    )

    assert [finding.location.start_line for finding in findings] == [4, 6]


def test_blank_line_before_assignment_has_one_tm1_owner() -> None:
    findings = _tm1("import subprocess\n\nenabled = True\nsubprocess.run(command, shell=enabled)\n")

    assert len(findings) == 1
    assert findings[0].start_line == 4


def test_unsupported_assignment_clears_existing_facts() -> None:
    assert not _tm1("enabled = True\nresult = factory()\nsubprocess.run(cmd, shell=enabled)\n")


def test_simple_name_store_with_unsafe_prior_binding_invalidates_truth_facts() -> None:
    findings = _tm1(
        "import subprocess\n"
        "class Trigger:\n"
        "    def __del__(self):\n"
        "        global enabled\n"
        "        enabled = False\n"
        "trigger = Trigger()\n"
        "enabled = True\n"
        "trigger = 0\n"
        "subprocess.run('/usr/bin/true', shell=enabled)\n"
    )

    assert not findings


def test_external_name_store_treats_prior_binding_as_finalizer_capable() -> None:
    findings = _tm1(
        "import subprocess\n"
        "class Trigger:\n"
        "    def __del__(self):\n"
        "        global enabled\n"
        "        enabled = False\n"
        "trigger = Trigger()\n"
        "enabled = False\n"
        "def execute():\n"
        "    global enabled, trigger\n"
        "    enabled = True\n"
        "    trigger = 0\n"
        "    subprocess.run('/usr/bin/true', shell=enabled)\n"
        "execute()\n"
    )

    assert not findings


def test_protocol_consuming_direct_call_invalidates_later_truth_fact() -> None:
    findings = _tm1(
        "class MutatingArgs:\n"
        "    def __iter__(self):\n"
        "        global enabled\n"
        "        enabled = False\n"
        "        return iter(('/usr/bin/true',))\n"
        "mutator = MutatingArgs()\n"
        "import subprocess\n"
        "enabled = True\n"
        "subprocess.run(mutator, shell=enabled)\n"
        "subprocess.run('/usr/bin/true', shell=enabled)\n"
    )

    assert [finding.start_line for finding in findings] == [9]


def test_annotated_assignment_is_outside_side_effect_free_contract() -> None:
    assert not _tm1("enabled: bool = True\nsubprocess.run(command, shell=enabled)\n")


def test_assignment_rhs_direct_call_is_inspected_before_invalidation() -> None:
    findings = _tm1("enabled = True\nresult = subprocess.run(command, shell=enabled)\n")

    assert len(findings) == 1
    assert findings[0].start_line == 2


def test_true_prefixed_identifier_has_one_lexical_owner() -> None:
    findings = _tm1("true_value = True\nsubprocess.run(command, shell=true_value)\n")

    assert len(findings) == 1


def test_true_prefixed_closure_has_one_lexical_owner() -> None:
    findings = _tm1(
        "def outer():\n"
        "    true_value = True\n"
        "    def inner():\n"
        "        subprocess.run(command, shell=true_value)\n"
    )

    assert len(findings) == 1
    assert findings[0].start_line == 4


def test_extensionless_python_true_prefixed_closure_has_one_lexical_owner() -> None:
    findings = _tm1(
        "#!/usr/bin/env python3\n"
        "import subprocess\n"
        "true_value = True\n"
        "def inner():\n"
        "    subprocess.run(command, shell=true_value)\n",
        "run",
    )

    assert len(findings) == 1
    assert findings[0].start_line == 5


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(
            "enabled = True\ndef run(enabled):\n    subprocess.run(command, shell=enabled)\n",
            id="parameter-shadow",
        ),
        pytest.param(
            "enabled = True\ndef run():\n"
            "    subprocess.run(command, shell=enabled)\n    enabled = False\n",
            id="later-local-shadow",
        ),
        pytest.param(
            "def outer():\n    enabled = True\n    def inner():\n"
            "        global enabled\n        subprocess.run(command, shell=enabled)\n",
            id="global-redirect",
        ),
    ],
)
def test_cross_scope_binding_must_resolve_to_literal_assignment(content: str) -> None:
    assert not _tm1(content)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(
            "import subprocess\n"
            "enabled = True\n"
            "values = [(subprocess.run(command, shell=enabled), (enabled := False)) "
            "for item in items]\n",
            id="module",
        ),
        pytest.param(
            "import subprocess\n"
            "def execute():\n"
            "    enabled = True\n"
            "    return [(subprocess.run(command, shell=enabled), "
            "(enabled := False)) for item in items]\n",
            id="function",
        ),
        pytest.param(
            "import subprocess\n"
            "enabled = True\n"
            "values = [[(subprocess.run(command, shell=enabled), (enabled := False)) "
            "for inner in inners] for outer in outers]\n",
            id="nested-comprehension",
        ),
    ],
)
def test_comprehension_walrus_binds_in_containing_scope(content: str) -> None:
    findings = _tm1(content)

    assert len(findings) == 1


def test_malformed_python_keeps_bounded_lexical_fallback() -> None:
    findings = _tm1("enabled = True\nsubprocess.run(command, shell=enabled)\nif:\n")

    assert len(findings) == 1
    assert "_tm1_variable_shell_flag" not in findings[0].evidence


def test_oversized_python_keeps_bounded_lexical_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = "enabled = True\nsubprocess.run(command, shell=enabled)\n" + "# padding\n" * 20
    monkeypatch.setattr(tm_module.static_runner, "MAX_FILE_CHARS", 80)

    findings = _tm1(content)

    assert len(findings) == 1
    assert "_tm1_variable_shell_flag" not in findings[0].evidence


def test_long_same_line_calls_keep_exact_coordinates_and_distinct_identity() -> None:
    payload = "x" * 240
    first_call = f'subprocess.run("{payload}A", shell=enabled)'
    second_call = f'subprocess.run("{payload}B", shell=enabled)'
    call_line = f"first = {first_call}; second = {second_call}"

    findings = _tm1(f"import subprocess\nenabled = True\n{call_line}\n")

    assert len(findings) == 2
    assert [(finding.start_column, finding.end_column) for finding in findings] == [
        (
            call_line.index(first_call),
            call_line.index(first_call) + len(first_call),
        ),
        (
            call_line.index(second_call),
            call_line.index(second_call) + len(second_call),
        ),
    ]
    assert findings[0].fingerprint() != findings[1].fingerprint()
    assert len(deduplicate(findings)) == 2


@pytest.mark.parametrize("path", ["run", "run.sh"])
def test_non_py_surfaces_do_not_enable_ast_companion(path: str) -> None:
    assert not _tm1("enabled = True\nsubprocess.run(command, shell=enabled)\n", path)


def test_python_window_surface_enables_ast_companion() -> None:
    findings = _tm1(
        "import subprocess\nenabled = True\nsubprocess.run(command, shell=enabled)\n",
        "run.pyw",
    )

    assert len(findings) == 1
