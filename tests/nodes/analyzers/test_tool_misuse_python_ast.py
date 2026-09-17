# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused coverage for issue #475's ordinary-Python binding form."""

from __future__ import annotations

import pytest

from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm_module


def _run(content: str, path: str = "run.py") -> dict:
    return tm_module.node({"components": [path], "file_cache": {path: content}})


def _tm1(content: str, path: str = "run.py") -> list:
    return [finding for finding in _run(content, path)["findings"] if finding.rule_id == "TM1"]


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
                "import subprocess\n"
                "Popen = Proxy()\n"
                "from subprocess import Popen\n"
                "enabled = True\n"
                "subprocess.run(command, shell=enabled)\n"
                "Popen(command, shell=enabled)\n"
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


def test_later_global_receiver_mutation_suppresses_function_body() -> None:
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


def test_unsupported_assignment_clears_existing_facts() -> None:
    assert not _tm1("enabled = True\nresult = factory()\nsubprocess.run(cmd, shell=enabled)\n")


def test_annotated_assignment_is_outside_side_effect_free_contract() -> None:
    assert not _tm1("enabled: bool = True\nsubprocess.run(command, shell=enabled)\n")


def test_assignment_rhs_direct_call_is_inspected_before_invalidation() -> None:
    findings = _tm1("enabled = True\nresult = subprocess.run(command, shell=enabled)\n")

    assert len(findings) == 1
    assert findings[0].start_line == 2


def test_true_prefixed_identifier_has_one_lexical_owner() -> None:
    findings = _tm1("true_value = True\nsubprocess.run(command, shell=true_value)\n")

    assert len(findings) == 1


@pytest.mark.parametrize("path", ["run.pyw", "run", "run.sh"])
def test_non_py_surfaces_do_not_enable_ast_companion(path: str) -> None:
    assert not _tm1("enabled = True\nsubprocess.run(command, shell=enabled)\n", path)
