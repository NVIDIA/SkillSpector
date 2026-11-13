# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repeated boolean targets must use their final assigned value."""

from __future__ import annotations

import pytest

from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm


def _scan(statement: str) -> list:
    source = (
        "import subprocess\n"
        "enabled = True\n" + statement + "subprocess.run('echo ok', shell=enabled)\n"
    )
    result = tm.node({"components": ["runner.py"], "file_cache": {"runner.py": source}})
    return [finding for finding in result["findings"] if finding.rule_id == "TM1"]


@pytest.mark.parametrize("false_value", ["False", "0", "''", "None", "()"])
@pytest.mark.parametrize(
    "assignment",
    [
        "enabled, enabled = True, {value}\n",
        "[enabled, enabled] = [True, {value}]\n",
        "enabled, (enabled, enabled) = False, (True, {value})\n",
        "enabled = enabled, enabled = True, {value}\n",
    ],
)
def test_last_false_store_drops_earlier_truthy_value(assignment: str, false_value: str) -> None:
    assert not _scan(assignment.format(value=false_value))


@pytest.mark.parametrize(
    "assignment",
    [
        "enabled, enabled = False, True\n",
        "[enabled, enabled] = [False, True]\n",
        "enabled, (enabled, enabled) = False, (False, True)\n",
        "enabled = enabled, enabled = False, True\n",
    ],
)
def test_last_true_store_keeps_detection(assignment: str) -> None:
    findings = _scan(assignment)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


def test_direct_boolean_controls() -> None:
    assert len(_scan("")) == 1
    assert not _scan("enabled = False\n")


@pytest.mark.parametrize(
    "assignment",
    [
        "enabled, enabled, enabled = True, False, enabled\n",
        "[enabled, enabled, enabled] = [True, False, enabled]\n",
        "enabled, (enabled, enabled) = True, (False, enabled)\n",
        "enabled = enabled, enabled = True, enabled\n",
        "enabled, enabled = True, dynamic\n",
    ],
)
def test_other_final_values_preserve_existing_event_policy(assignment: str) -> None:
    assert len(_scan(assignment)) == 1


@pytest.mark.parametrize(
    "assignment",
    [
        "enabled, enabled, enabled = True, opaque, False\n",
        "enabled, enabled, enabled = True, factory(), False\n",
        "enabled, enabled, other.value = True, False, 0\n",
    ],
)
def test_opaque_assignment_effects_preserve_existing_event_policy(assignment: str) -> None:
    assert len(_scan(assignment)) == 1


@pytest.mark.parametrize(
    "statement",
    [
        "other = factory()\nenabled, enabled, other = True, False, None\n",
        "enabled = factory()\nenabled, enabled = True, False\n",
    ],
)
def test_unknown_prior_values_preserve_existing_event_policy(statement: str) -> None:
    assert len(_scan(statement)) == 1


@pytest.mark.parametrize("parameter", [True, False])
def test_local_target_requires_known_prior_value(parameter: bool) -> None:
    source = "import subprocess\ndef execute(" + ("other" if parameter else "") + "):\n"
    if not parameter:
        source += "    other = None\n"
    source += (
        "    enabled = True\n"
        "    enabled, enabled, other = True, False, None\n"
        "    subprocess.run('echo ok', shell=enabled)\n"
    )
    response = tm.node({"components": ["runner.py"], "file_cache": {"runner.py": source}})
    hits = [finding for finding in response["findings"] if finding.rule_id == "TM1"]
    assert len(hits) == int(parameter)


def test_class_scope_preserves_existing_event_policy() -> None:
    source = (
        "import subprocess\n"
        "class Options:\n"
        "    enabled = True\n"
        "    enabled, enabled = True, False\n"
        "    subprocess.run('echo ok', shell=enabled)\n"
    )
    response = tm.node({"components": ["runner.py"], "file_cache": {"runner.py": source}})
    hits = [finding for finding in response["findings"] if finding.rule_id == "TM1"]
    assert len(hits) == 1


def test_other_imports_preserve_existing_event_policy() -> None:
    assert len(_scan("import math\nenabled, enabled = True, False\n")) == 1


@pytest.mark.parametrize(
    "parameter, native_import",
    [("Popen", "import subprocess"), ("subprocess", "from subprocess import Popen")],
)
def test_native_import_does_not_make_other_parameter_safe(
    parameter: str, native_import: str
) -> None:
    source = (
        "import subprocess\n"
        f"def execute({parameter}):\n"
        f"    {native_import}\n"
        "    enabled = True\n"
        f"    enabled, enabled, {parameter} = True, False, None\n"
        + (
            "    subprocess.run('echo ok', shell=enabled)\n"
            if parameter == "Popen"
            else "    Popen('echo ok', shell=enabled)\n"
        )
    )
    response = tm.node({"components": ["runner.py"], "file_cache": {"runner.py": source}})
    hits = [finding for finding in response["findings"] if finding.rule_id == "TM1"]
    assert len(hits) == 1


@pytest.mark.parametrize(
    "target, native_import, receiver",
    [
        ("Popen", "from subprocess import Popen", "subprocess.run"),
        ("subprocess", "import subprocess", "Popen"),
    ],
)
def test_fresh_native_import_keeps_its_own_binding_safe(
    target: str, native_import: str, receiver: str
) -> None:
    source = (
        "import subprocess\n"
        "from subprocess import Popen\n"
        "def execute():\n"
        f"    {native_import}\n"
        "    enabled = True\n"
        f"    enabled, enabled, {target} = True, False, None\n"
        f"    {receiver}('echo ok', shell=enabled)\n"
    )
    response = tm.node({"components": ["runner.py"], "file_cache": {"runner.py": source}})
    hits = [finding for finding in response["findings"] if finding.rule_id == "TM1"]
    assert not hits
