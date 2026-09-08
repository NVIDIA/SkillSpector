# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused coverage for issue #475's straight-line Python binding form."""

from __future__ import annotations

import pytest

from skillspector.inspection_ledger import LedgerOutcome
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm_module


def _run(content: str, path: str = "run.py") -> dict:
    return tm_module.node(
        {
            "components": [path],
            "file_cache": {path: content},
        }
    )


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
    "value",
    [
        pytest.param("1", id="integer"),
        pytest.param("-1", id="negative-integer"),
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
        "third: object = second\n"
        "subprocess.check_output(command, shell=third)\n"
    )

    assert len(findings) == 1
    assert findings[0].start_line == 5


def test_bare_popen_direct_expression_is_tracked() -> None:
    findings = _tm1("enabled = True\nPopen(command, shell=enabled)\n")

    assert len(findings) == 1


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


def test_wildcard_import_clears_all_facts() -> None:
    assert not _tm1(
        "enabled = True\nfrom settings import *\nsubprocess.run(command, shell=enabled)\n"
    )


def test_function_local_straight_line_binding_is_tracked_fresh() -> None:
    findings = _tm1(
        "outer = True\n"
        "def execute(command):\n"
        "    enabled = 'True'\n"
        "    subprocess.run(command, shell=enabled)\n"
        "subprocess.run(command, shell=outer)\n"
    )

    assert len(findings) == 1
    assert findings[0].start_line == 4


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


def test_named_expression_in_call_is_rejected_and_clears_facts() -> None:
    assert not _tm1(
        "enabled = True\n"
        "subprocess.run(command, shell=enabled, marker=(enabled := False))\n"
        "subprocess.run(command, shell=enabled)\n"
    )


def test_unsafe_annotation_clears_facts() -> None:
    assert not _tm1(
        "enabled = True\nitem: (enabled := False) = 1\nsubprocess.run(command, shell=enabled)\n"
    )


def test_true_prefixed_identifier_is_owned_by_lexical_tm1_only() -> None:
    findings = _tm1(
        "import subprocess\ntrue_value = True\nsubprocess.run(command, shell=true_value)\n"
    )

    assert len(findings) == 1
    assert findings[0].confidence == pytest.approx(0.9)


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


def test_direct_literal_remains_one_lexical_finding() -> None:
    findings = _tm1("import subprocess\nsubprocess.run(command, shell=True)\n")

    assert len(findings) == 1
    assert findings[0].confidence == pytest.approx(0.9)


@pytest.mark.parametrize("path", ["guide.md", "run.js"])
def test_non_python_lexical_fallback_remains_case_insensitive(path: str) -> None:
    assert len(_tm1("Popen(command, shell=true)", path)) == 1


def test_malformed_python_retains_direct_literal_fallback() -> None:
    assert len(_tm1("subprocess.run(command, shell=True", "broken.py")) == 1


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
