# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic regression tests for dependency-source redirection."""

from __future__ import annotations

import json

import pytest

from skillspector.dependency_sources import analyze_dependency_sources
from skillspector.llm_analyzer_base import Batch
from skillspector.models import Finding
from skillspector.nodes.meta_analyzer import (
    PER_FILE_ANALYSIS_PROMPT,
    LLMMetaAnalyzer,
    _fallback_filtered,
    _passthrough_with_defaults,
)
from skillspector.nodes.report import report
from skillspector.state import SkillspectorState


def _analyze(
    files: dict[str, str], metadata: list[dict[str, object]] | None = None
) -> list[Finding]:
    return analyze_dependency_sources(sorted(files), files, metadata or [])


def test_generated_npm_and_yarn_configs_resolve_simple_local_indirection() -> None:
    script = """#!/bin/sh
SOURCE_URL="https://packages.example.invalid"
cat > "$PROJECT/.npmrc" << EOF
registry=${SOURCE_URL}
EOF
cat > "$PROJECT/.yarnrc" << EOF
registry "${SOURCE_URL}"
EOF
"""

    findings = _analyze({"scripts/setup.sh": script})

    assert [(finding.evidence["ecosystem"], finding.start_line) for finding in findings] == [
        ("npm", 4),
        ("yarn", 7),
    ]
    assert all(finding.rule_id == "SC10" for finding in findings)
    assert all(finding.severity == "HIGH" for finding in findings)
    assert all(finding.evidence["operation"] == "replace" for finding in findings)
    assert all(
        finding.evidence["destination"] == "https://packages.example.invalid"
        for finding in findings
    )


def test_supported_direct_configuration_surfaces_cover_all_ecosystems() -> None:
    files = {
        ".npmrc": "@team:registry=https://npm.example.invalid\n",
        ".yarnrc.yml": (
            "npmScopes:\n  team:\n    npmRegistryServer: https://yarn.example.invalid\n"
        ),
        "pip.conf": (
            "[global]\n"
            "index-url = https://python.example.invalid/simple\n"
            "extra-index-url = https://extra.example.invalid/simple\n"
        ),
        "pyproject.toml": (
            "[[tool.poetry.source]]\n"
            'name = "mirror"\n'
            'url = "https://poetry.example.invalid/simple"\n'
        ),
        "settings.xml": (
            "<settings><mirrors><mirror><id>all</id><mirrorOf>*</mirrorOf>"
            "<url>https://maven.example.invalid/repository</url>"
            "</mirror></mirrors></settings>"
        ),
        ".cargo/config.toml": (
            '[source.crates-io]\nreplace-with = "mirror"\n'
            '[source.mirror]\nregistry = "sparse+https://cargo.example.invalid/index"\n'
        ),
    }

    findings = _analyze(files)

    assert {finding.evidence["ecosystem"] for finding in findings} == {
        "npm",
        "yarn",
        "pip",
        "poetry",
        "maven",
        "cargo",
    }
    npm = next(finding for finding in findings if finding.evidence["ecosystem"] == "npm")
    assert npm.evidence["scope"] == "@team"
    yarn = next(finding for finding in findings if finding.evidence["ecosystem"] == "yarn")
    assert yarn.evidence["scope"] == "team"
    pip_operations = {
        finding.evidence["operation"]
        for finding in findings
        if finding.evidence["ecosystem"] == "pip"
    }
    assert pip_operations == {"replace", "add"}
    cargo = [finding for finding in findings if finding.evidence["ecosystem"] == "cargo"]
    assert any(finding.evidence["operation"] == "replace" for finding in cargo)


def test_supported_command_and_environment_surfaces() -> None:
    script = """#!/bin/sh
npm config set registry https://npm.example.invalid
yarn config set npmRegistryServer https://yarn.example.invalid
pip install --index-url https://pip.example.invalid/simple example
pip config set global.extra-index-url https://extra.example.invalid/simple
poetry source add private https://poetry.example.invalid/simple
mvn -Dmaven.repo.remote=https://maven.example.invalid/repo verify
export CARGO_REGISTRIES_PRIVATE_INDEX=sparse+https://cargo.example.invalid/index
"""

    findings = _analyze({"setup.sh": script})

    assert {finding.evidence["ecosystem"] for finding in findings} == {
        "npm",
        "yarn",
        "pip",
        "poetry",
        "maven",
        "cargo",
    }
    assert all(finding.evidence["destination_status"] == "resolved" for finding in findings)


@pytest.mark.parametrize(
    ("name", "destination", "ecosystem", "operation", "scope"),
    [
        (
            "NPM_CONFIG_REGISTRY",
            "https://npm.example.invalid",
            "npm",
            "replace",
            "global",
        ),
        (
            "PIP_INDEX_URL",
            "https://pip.example.invalid/simple",
            "pip",
            "replace",
            "global",
        ),
        (
            "PIP_EXTRA_INDEX_URL",
            "https://extra.example.invalid/simple",
            "pip",
            "add",
            "global",
        ),
        (
            "CARGO_REGISTRIES_PRIVATE_INDEX",
            "sparse+https://cargo.example.invalid/index",
            "cargo",
            "add",
            "private",
        ),
    ],
)
@pytest.mark.parametrize(
    "template",
    [
        "MARKER=1 {name}={destination}",
        "export MARKER=1 {name}={destination}",
        "{name}={destination} MARKER=1",
        "export {name}={destination} MARKER",
    ],
)
def test_dependency_environment_variable_can_be_any_assignment_word(
    name: str,
    destination: str,
    ecosystem: str,
    operation: str,
    scope: str,
    template: str,
) -> None:
    script = template.format(name=name, destination=destination) + "\n"

    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_id == "SC10"
    assert finding.severity == "HIGH"
    assert finding.evidence["ecosystem"] == ecosystem
    assert finding.evidence["operation"] == operation
    assert finding.evidence["surface"] == "environment variable"
    assert finding.evidence["scope"] == scope
    assert finding.evidence["destination"] == destination
    assert finding.evidence["destination_status"] == "resolved"


@pytest.mark.parametrize(
    "script",
    [
        "NPM_CONFIG_REGISTRY=https://registry.npmjs.org/ MARKER=1\n",
        "export MARKER=1 NPM_CONFIG_REGISTRY=https://registry.npmjs.org/\n",
        "PIP_INDEX_URL=https://pypi.org/simple/ MARKER=1\n",
        "export MARKER=1 PIP_INDEX_URL=https://pypi.org/simple/\n",
    ],
)
def test_canonical_environment_assignment_with_other_words_is_not_high(script: str) -> None:
    assert _analyze({"setup.sh": script}) == []


def test_multiple_dependency_environment_assignments_are_independent() -> None:
    script = """export MARKER=1 NPM_CONFIG_REGISTRY=https://npm.example.invalid PIP_INDEX_URL=https://pip.example.invalid/simple
"""

    findings = _analyze({"setup.sh": script})

    assert [finding.evidence["ecosystem"] for finding in findings] == ["npm", "pip"]
    assert [finding.evidence["destination"] for finding in findings] == [
        "https://npm.example.invalid",
        "https://pip.example.invalid/simple",
    ]


def test_nonfirst_environment_assignment_resolves_prior_literal_value() -> None:
    script = """SOURCE=https://packages.example.invalid
MARKER=1 NPM_CONFIG_REGISTRY=$SOURCE
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.evidence["destination"] == "https://packages.example.invalid"
    assert finding.evidence["destination_status"] == "resolved"


def test_nonfirst_dynamic_environment_assignment_remains_unresolved() -> None:
    finding = _analyze({"setup.sh": "MARKER=1 NPM_CONFIG_REGISTRY=$RUNTIME_SOURCE\n"})[0]

    assert finding.evidence["destination"] == "unresolved"
    assert finding.evidence["destination_status"] == "unresolved"
    assert finding.severity == "HIGH"


@pytest.mark.parametrize(
    "script",
    [
        "export NPM_CONFIG_REGISTRY=https://packages.example.invalid "
        "NPM_CONFIG_REGISTRY=https://registry.npmjs.org/\n",
        "env NPM_CONFIG_REGISTRY=https://packages.example.invalid "
        "NPM_CONFIG_REGISTRY=https://registry.npmjs.org/ npm install\n",
    ],
)
def test_last_duplicate_environment_assignment_takes_precedence(script: str) -> None:
    assert _analyze({"setup.sh": script}) == []


@pytest.mark.parametrize(
    "script",
    [
        "export NPM_CONFIG_REGISTRY=https://registry.npmjs.org/ "
        "NPM_CONFIG_REGISTRY=https://packages.example.invalid\n",
        "env NPM_CONFIG_REGISTRY=https://registry.npmjs.org/ "
        "NPM_CONFIG_REGISTRY=https://packages.example.invalid npm install\n",
    ],
)
def test_last_noncanonical_duplicate_environment_assignment_is_high(script: str) -> None:
    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].evidence["destination"] == "https://packages.example.invalid"


@pytest.mark.parametrize(
    "script",
    [
        "export MARKER NPM_CONFIG_REGISTRY=https://packages.example.invalid\n",
        "export -- MARKER=1 NPM_CONFIG_REGISTRY=https://packages.example.invalid\n",
        'export MARKER=1 "NPM_CONFIG_REGISTRY=https://packages.example.invalid"\n',
    ],
)
def test_export_assignment_operand_forms_are_detected(script: str) -> None:
    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].evidence["destination"] == "https://packages.example.invalid"


@pytest.mark.parametrize(
    "script",
    [
        "MARKER=1 NPM_CONFIG_REGISTRY=$(printf %s https://packages.example.invalid)\n",
        "export MARKER=1 NPM_CONFIG_REGISTRY=$(printf %s https://packages.example.invalid)\n",
        "env NPM_CONFIG_REGISTRY=$(printf %s https://packages.example.invalid) npm install\n",
        "SOURCE=https://registry.npmjs.org/\n"
        r"MARKER=1 NPM_CONFIG_REGISTRY=\$SOURCE" + "\n",
    ],
)
def test_complex_or_escaped_environment_values_remain_unresolved(script: str) -> None:
    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].evidence["destination"] == "unresolved"
    assert findings[0].evidence["destination_status"] == "unresolved"


@pytest.mark.parametrize(
    "line",
    [
        "NPM_CONFIG_REGISTRY='$SOURCE'",
        "env NPM_CONFIG_REGISTRY='$SOURCE' npm install",
        "npm config set registry '$SOURCE'",
    ],
)
def test_single_quoted_dependency_source_variable_is_literal_and_unresolved(line: str) -> None:
    script = f"SOURCE=https://registry.npmjs.org/\n{line}\n"

    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].evidence["destination"] == "unresolved"
    assert findings[0].evidence["destination_status"] == "unresolved"


@pytest.mark.parametrize(
    "line",
    [
        'NPM_CONFIG_REGISTRY="$SOURCE"',
        'env NPM_CONFIG_REGISTRY="$SOURCE" npm install',
        'npm config set registry "$SOURCE"',
    ],
)
def test_double_quoted_dependency_source_variable_expands_statically(line: str) -> None:
    script = f"SOURCE=https://registry.npmjs.org/\n{line}\n"

    assert _analyze({"setup.sh": script}) == []


@pytest.mark.parametrize(
    "substitution",
    [
        "$(printf %s https://packages.example.invalid | tr a-z A-Z)",
        "$(printf %s https://packages.example.invalid; printf /simple)",
    ],
)
def test_assignment_command_substitution_keeps_internal_control_operators(
    substitution: str,
) -> None:
    script = f"MARKER=1 NPM_CONFIG_REGISTRY={substitution}\n"

    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].evidence["destination"] == "unresolved"
    assert findings[0].evidence["destination_status"] == "unresolved"


@pytest.mark.parametrize(
    "script",
    [
        "npm config set registry `printf %s https://packages.example.invalid | tr a-z A-Z`\n",
        "MARKER=1 NPM_CONFIG_REGISTRY="
        "`printf %s https://packages.example.invalid; printf /simple`\n",
        "env NPM_CONFIG_REGISTRY="
        "`printf %s https://packages.example.invalid | tr a-z A-Z` npm install\n",
    ],
)
def test_backtick_substitution_keeps_internal_control_operators(script: str) -> None:
    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].evidence["destination"] == "unresolved"
    assert findings[0].evidence["destination_status"] == "unresolved"


def test_generated_configs_support_pip_poetry_maven_and_cargo() -> None:
    script = """#!/bin/sh
cat > "$ROOT/pip.conf" << EOF
[global]
index-url = https://pip.example.invalid/simple
EOF
cat > "$ROOT/pyproject.toml" << EOF
[[tool.poetry.source]]
name = "private"
url = "https://poetry.example.invalid/simple"
EOF
cat > "$ROOT/settings.xml" << EOF
<settings><mirrors><mirror><mirrorOf>*</mirrorOf><url>https://maven.example.invalid/repo</url></mirror></mirrors></settings>
EOF
cat > "$ROOT/.cargo/config.toml" << EOF
[registries.private]
index = "sparse+https://cargo.example.invalid/index"
EOF
"""

    findings = _analyze({"generate.sh": script})

    assert {finding.evidence["ecosystem"] for finding in findings} == {
        "pip",
        "poetry",
        "maven",
        "cargo",
    }
    assert all(
        str(finding.evidence["surface"]).startswith("generated")
        or finding.evidence["ecosystem"] == "pip"
        for finding in findings
    )


def test_canonical_defaults_do_not_produce_sc10() -> None:
    files = {
        ".npmrc": "registry=https://registry.npmjs.org/\n",
        ".yarnrc": 'registry "https://registry.npmjs.org"\n',
        "pip.conf": "[global]\nindex-url=https://pypi.org/simple/\n",
        "pyproject.toml": (
            '[[tool.poetry.source]]\nname = "pypi"\nurl = "https://pypi.org/simple"\n'
        ),
        "settings.xml": (
            "<settings><profiles><profile><repositories><repository>"
            "<url>https://repo.maven.apache.org/maven2/</url>"
            "</repository></repositories></profile></profiles></settings>"
        ),
        ".cargo/config.toml": (
            '[source.crates-io]\nreplace-with = "canonical"\n'
            '[source.canonical]\nregistry = "sparse+https://index.crates.io/"\n'
        ),
    }

    assert _analyze(files) == []


@pytest.mark.parametrize("filename", [".yarnrc", ".yarnrc.yml"])
def test_yarn_documented_public_default_does_not_produce_sc10(filename: str) -> None:
    content = (
        'registry "https://registry.yarnpkg.com"\n'
        if filename == ".yarnrc"
        else "npmRegistryServer: https://registry.yarnpkg.com\n"
    )

    assert _analyze({filename: content}) == []


def test_variable_resolution_uses_assignment_visible_at_command_line() -> None:
    script = """SRC=https://packages.example.invalid
npm config set registry "$SRC"
SRC=https://registry.npmjs.org/
"""

    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].start_line == 2
    assert findings[0].evidence["destination"] == "https://packages.example.invalid"


def test_assignment_text_in_unrelated_heredoc_cannot_suppress_sc10() -> None:
    script = """#!/bin/sh
SRC=https://packages.example.invalid
cat <<'EOF' > instructions.txt
SRC=https://registry.npmjs.org/
EOF
npm config set registry "$SRC"
"""

    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].start_line == 6
    assert findings[0].evidence["destination"] == "https://packages.example.invalid"


def test_assignment_in_uncalled_function_cannot_suppress_sc10() -> None:
    script = """#!/bin/sh
SRC=https://packages.example.invalid
configure_later() {
  SRC=https://registry.npmjs.org/
}
npm config set registry "$SRC"
"""

    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].start_line == 6
    assert findings[0].evidence["destination"] == "https://packages.example.invalid"


def test_assignment_in_split_line_function_declaration_cannot_suppress_sc10() -> None:
    script = """#!/bin/sh
SRC=https://packages.example.invalid
configure_later()
{
  SRC=https://registry.npmjs.org/
}
npm config set registry "$SRC"
"""

    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].start_line == 7
    assert findings[0].evidence["destination"] == "https://packages.example.invalid"


def test_called_function_assignment_keeps_possible_redirect_high() -> None:
    script = """#!/bin/sh
SRC=https://registry.npmjs.org/
use_private() {
  SRC=https://packages.example.invalid
}
use_private
npm config set registry "$SRC"
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 7
    assert finding.severity == "HIGH"
    assert finding.evidence["destination"] == "unresolved"
    assert finding.evidence["destination_status"] == "unresolved"


def test_conditionally_called_function_keeps_possible_redirect_high() -> None:
    script = """SRC=https://registry.npmjs.org/
use_private() { SRC=https://packages.example.invalid; }
if test -f use-private; then use_private; fi
npm config set registry "$SRC"
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 4
    assert finding.severity == "HIGH"
    assert finding.evidence["destination"] == "unresolved"


@pytest.mark.parametrize(
    "invocation",
    [
        "if use_private; then :; fi",
        "MARKER=1 use_private",
        "{ use_private; }",
    ],
)
def test_function_invocation_shapes_keep_possible_redirect_high(invocation: str) -> None:
    script = f"""SRC=https://registry.npmjs.org/
use_private() {{ SRC=https://packages.example.invalid; }}
{invocation}
npm config set registry "$SRC"
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 4
    assert finding.severity == "HIGH"
    assert finding.evidence["destination"] == "unresolved"
    assert finding.evidence["destination_status"] == "unresolved"


@pytest.mark.parametrize(
    ("script", "ecosystem", "surface"),
    [
        (
            "MARKER=1 npm config set registry https://packages.example.invalid\n",
            "npm",
            "npm config set",
        ),
        (
            "if :; then yarn config set registry https://packages.example.invalid; fi\n",
            "yarn",
            "yarn config set",
        ),
        (
            "{ pip install --index-url https://packages.example.invalid demo; }\n",
            "pip",
            "pip --index-url",
        ),
        (
            "while false; do pip config set global.index-url "
            "https://packages.example.invalid; done\n",
            "pip",
            "pip config set",
        ),
        (
            "MARKER=1 pip install --extra-index-url https://packages.example.invalid demo\n",
            "pip",
            "pip --extra-index-url",
        ),
        (
            "{ pip config set global.extra-index-url https://packages.example.invalid; }\n",
            "pip",
            "pip config set",
        ),
        (
            "MARKER=1 poetry source add private https://packages.example.invalid\n",
            "poetry",
            "poetry source add",
        ),
        (
            "{ poetry config repositories.private https://packages.example.invalid; }\n",
            "poetry",
            "poetry config repositories",
        ),
        (
            "if :; then mvn -Dmaven.repo.remote=https://packages.example.invalid verify; fi\n",
            "maven",
            "Maven CLI repository",
        ),
    ],
)
def test_package_manager_commands_remain_detectable_in_shell_wrappers(
    script: str, ecosystem: str, surface: str
) -> None:
    finding = _analyze({"setup.sh": script})[0]

    assert finding.rule_id == "SC10"
    assert finding.severity == "HIGH"
    assert finding.evidence["ecosystem"] == ecosystem
    assert finding.evidence["surface"] == surface
    assert finding.evidence["destination"] == "https://packages.example.invalid"


@pytest.mark.parametrize(
    ("command", "ecosystem", "surface"),
    [
        (
            "npm config set registry https://packages.example.invalid",
            "npm",
            "npm config set",
        ),
        (
            "yarn config set npmRegistryServer https://packages.example.invalid",
            "yarn",
            "yarn config set",
        ),
        (
            "python3 -m pip install --index-url https://packages.example.invalid demo",
            "pip",
            "pip --index-url",
        ),
        (
            "pip install --extra-index-url https://packages.example.invalid demo",
            "pip",
            "pip --extra-index-url",
        ),
        (
            "pip config set global.index-url https://packages.example.invalid",
            "pip",
            "pip config set",
        ),
        (
            "pip config set global.extra-index-url https://packages.example.invalid",
            "pip",
            "pip config set",
        ),
        (
            "poetry source add private https://packages.example.invalid",
            "poetry",
            "poetry source add",
        ),
        (
            "poetry config repositories.private https://packages.example.invalid",
            "poetry",
            "poetry config repositories",
        ),
        (
            "mvn -Dmaven.repo.remote=https://packages.example.invalid verify",
            "maven",
            "Maven CLI repository",
        ),
    ],
)
@pytest.mark.parametrize(
    "wrapper",
    [
        "env MARKER=1 {command}",
        "sudo -E {command}",
        "command -- {command}",
        "( {command} )",
    ],
)
def test_common_static_execution_wrappers_cover_every_command_family(
    command: str, ecosystem: str, surface: str, wrapper: str
) -> None:
    findings = _analyze({"setup.sh": wrapper.format(command=command) + "\n"})

    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_id == "SC10"
    assert finding.severity == "HIGH"
    assert finding.evidence["ecosystem"] == ecosystem
    assert finding.evidence["surface"] == surface
    assert finding.evidence["destination"] == "https://packages.example.invalid"


@pytest.mark.parametrize(
    "wrapper",
    [
        "( {command}; )",
        "( {command} ) >review.log",
        "( {command} ) 2>/dev/null",
        "( {command} ) >review.log 2>&1",
        "( ( {command} ) )",
        "( {command} && true )",
        "( true && {command} )",
    ],
)
def test_bounded_subshell_variants_preserve_actionable_command(wrapper: str) -> None:
    command = "npm config set registry https://packages.example.invalid"

    findings = _analyze({"setup.sh": wrapper.format(command=command) + "\n"})

    assert len(findings) == 1
    assert findings[0].evidence["surface"] == "npm config set"
    assert findings[0].evidence["destination"] == "https://packages.example.invalid"


@pytest.mark.parametrize(
    "script",
    [
        "env -i -u HOME MARKER=1 command -- npm config set registry "
        "https://packages.example.invalid\n",
        "sudo -u root -E -- npm config set registry https://packages.example.invalid\n",
        "( sudo -E env -i MARKER=1 command -- npm config set registry "
        "https://packages.example.invalid )\n",
    ],
)
def test_nested_and_option_bearing_execution_wrappers_are_bounded(script: str) -> None:
    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].evidence["destination"] == "https://packages.example.invalid"


@pytest.mark.parametrize(
    ("script", "ecosystem"),
    [
        (
            "env MARKER=1 NPM_CONFIG_REGISTRY=https://npm.example.invalid npm install\n",
            "npm",
        ),
        (
            "env MARKER=1 PIP_INDEX_URL=https://pip.example.invalid/simple pip install demo\n",
            "pip",
        ),
        (
            "env MARKER=1 CARGO_REGISTRIES_PRIVATE_INDEX="
            "sparse+https://cargo.example.invalid/index cargo build\n",
            "cargo",
        ),
    ],
)
def test_env_wrapped_dependency_environment_assignments_are_detected(
    script: str, ecosystem: str
) -> None:
    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].evidence["ecosystem"] == ecosystem
    assert findings[0].evidence["surface"] == "environment variable"


def test_python_module_pip3_uses_pip_environment_source() -> None:
    script = (
        "env PIP_INDEX_URL=https://packages.example.invalid/simple python -m pip3 install demo\n"
    )

    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].evidence["ecosystem"] == "pip"
    assert findings[0].evidence["surface"] == "environment variable"
    assert findings[0].evidence["destination"] == "https://packages.example.invalid/simple"


@pytest.mark.parametrize(
    "script",
    [
        "PIP_INDEX_URL=https://packages.example.invalid/simple "
        "pip_index_url=https://pypi.org/simple/\n",
        "env PIP_INDEX_URL=https://packages.example.invalid/simple "
        "pip_index_url=https://pypi.org/simple/ pip install demo\n",
    ],
)
def test_last_write_wins_only_for_exact_environment_variable_name(script: str) -> None:
    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].evidence["ecosystem"] == "pip"
    assert findings[0].evidence["destination"] == "https://packages.example.invalid/simple"


@pytest.mark.parametrize(
    "script",
    [
        "PIP_INDEX_URL=https://packages.example.invalid/simple env -i pip install demo\n",
        "PIP_INDEX_URL=https://packages.example.invalid/simple "
        "env --ignore-environment pip install demo\n",
        "PIP_INDEX_URL=https://packages.example.invalid/simple "
        "env -u PIP_INDEX_URL pip install demo\n",
        "PIP_INDEX_URL=https://packages.example.invalid/simple "
        "env --unset=PIP_INDEX_URL pip install demo\n",
        "env PIP_INDEX_URL=https://packages.example.invalid/simple env -i pip install demo\n",
        "env PIP_INDEX_URL=https://packages.example.invalid/simple "
        "env --unset PIP_INDEX_URL pip install demo\n",
    ],
)
def test_env_clear_and_unset_remove_accumulated_assignments(script: str) -> None:
    assert _analyze({"setup.sh": script}) == []


@pytest.mark.parametrize(
    "script",
    [
        "PIP_INDEX_URL=https://pypi.org/simple env -i "
        "PIP_INDEX_URL=https://packages.example.invalid/simple pip install demo\n",
        "env PIP_INDEX_URL=https://pypi.org/simple env -u PIP_INDEX_URL "
        "PIP_INDEX_URL=https://packages.example.invalid/simple pip install demo\n",
    ],
)
def test_env_assignments_after_clear_or_unset_remain_effective(script: str) -> None:
    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].evidence["destination"] == "https://packages.example.invalid/simple"


@pytest.mark.parametrize(
    "script",
    [
        "command -v npm config set registry https://packages.example.invalid\n",
        "command -V npm config set registry https://packages.example.invalid\n",
        "sudo -V npm config set registry https://packages.example.invalid\n",
        "sudo -l npm config set registry https://packages.example.invalid\n",
        "env -S 'npm config set registry https://packages.example.invalid'\n",
        "( npm config set registry https://packages.example.invalid\n",
        '"npm config set registry https://packages.example.invalid"\n',
        "'npm config set registry https://packages.example.invalid'\n",
        r"npm\ config\ set\ registry\ https://packages.example.invalid" + "\n",
        '( "npm config set registry https://packages.example.invalid" )\n',
        'env "npm config set registry https://packages.example.invalid"\n',
        "npm 'config set registry https://packages.example.invalid'\n",
        "pip 'install --index-url https://packages.example.invalid demo'\n",
        "poetry 'source add private https://packages.example.invalid'\n",
        "mvn '-Dmaven.repo.remote=https://packages.example.invalid verify'\n",
        '"NPM_CONFIG_REGISTRY=https://packages.example.invalid MARKER=1"\n',
        "command NPM_CONFIG_REGISTRY=https://packages.example.invalid npm install\n",
        "command -- NPM_CONFIG_REGISTRY=https://packages.example.invalid npm install\n",
    ],
)
def test_nonexecuting_or_malformed_wrappers_are_not_actionable(script: str) -> None:
    assert _analyze({"setup.sh": script}) == []


@pytest.mark.parametrize(
    "script",
    [
        "( npm config set registry https://packages.example.invalid ) arbitrary-tail\n",
        "( npm config set registry https://packages.example.invalid ) >\n",
        "( npm config set registry https://packages.example.invalid\n",
        "( ( npm config set registry https://packages.example.invalid )\n",
        "(( npm config set registry https://packages.example.invalid ))\n",
        "( 'npm config set registry https://packages.example.invalid'; )\n",
        "( npm 'config set registry https://packages.example.invalid'; )\n",
        "( npm config set registry https://packages.example.invalid ) >review.log arbitrary-tail\n",
        "echo `printf 'npm config set registry https://packages.example.invalid' | cat`\n",
    ],
)
def test_malformed_or_inert_grouping_is_not_actionable(script: str) -> None:
    assert _analyze({"setup.sh": script}) == []


def test_wrapped_command_text_in_unrelated_heredoc_is_not_actionable() -> None:
    script = """cat <<'EOF'
env MARKER=1 npm config set registry https://packages.example.invalid
sudo -E pip config set global.index-url https://packages.example.invalid
EOF
"""

    assert _analyze({"setup.sh": script}) == []


@pytest.mark.parametrize("output_format", ["terminal", "json", "markdown", "sarif"])
def test_assignment_prefixed_command_is_preserved_in_all_reports(output_format: str) -> None:
    finding = _analyze(
        {"setup.sh": ("MARKER=1 npm config set registry https://packages.example.invalid\n")}
    )[0]
    state: SkillspectorState = {
        "filtered_findings": [finding],
        "component_metadata": [],
        "has_executable_scripts": True,
        "manifest": {},
        "output_format": output_format,
    }

    result = report(state)
    rendered = json.dumps(result.get("sarif_report", result.get("report_body", "")))

    assert "SC10" in rendered
    assert "packages.example.invalid" in rendered


@pytest.mark.parametrize("output_format", ["terminal", "json", "markdown", "sarif"])
@pytest.mark.parametrize(
    "script",
    [
        "MARKER=1 NPM_CONFIG_REGISTRY=https://packages.example.invalid\n",
        "env MARKER=1 npm config set registry https://packages.example.invalid\n",
        "sudo -E npm config set registry https://packages.example.invalid\n",
        "command -- npm config set registry https://packages.example.invalid\n",
        "( npm config set registry https://packages.example.invalid )\n",
        "cat > .npmrc <<END$OF\nregistry=https://packages.example.invalid\nEND$OF\n",
        "cat > .npmrc <<END${OF}\nregistry=https://packages.example.invalid\nEND${OF}\n",
    ],
)
def test_reviewed_dependency_source_forms_survive_all_reports(
    output_format: str, script: str
) -> None:
    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    state: SkillspectorState = {
        "filtered_findings": findings,
        "component_metadata": [],
        "has_executable_scripts": True,
        "manifest": {},
        "output_format": output_format,
    }
    result = report(state)
    rendered = json.dumps(result.get("sarif_report", result.get("report_body", "")))

    assert "SC10" in rendered
    assert "packages.example.invalid" in rendered


@pytest.mark.parametrize("output_format", ["terminal", "json", "markdown", "sarif"])
def test_nonfirst_environment_assignment_credentials_are_redacted_in_all_reports(
    output_format: str,
) -> None:
    username = "second-assignment-user-sentinel"
    password = "second-assignment-password-sentinel"
    token = "second-assignment-token-sentinel"
    destination = f"https://{username}:{password}@packages.example.invalid/simple?token={token}"
    findings = _analyze({"setup.sh": f"export MARKER=1 PIP_INDEX_URL={destination}\n"})

    assert len(findings) == 1
    state: SkillspectorState = {
        "filtered_findings": findings,
        "component_metadata": [],
        "has_executable_scripts": True,
        "manifest": {},
        "output_format": output_format,
    }
    result = report(state)
    rendered = json.dumps(result.get("sarif_report", result.get("report_body", "")))

    for secret in (username, password, token):
        assert secret not in json.dumps(findings[0].to_dict())
        assert secret not in rendered
    assert "packages.example.invalid" in rendered


def test_assignment_in_case_arm_keeps_possible_redirect_high() -> None:
    script = """SRC=https://registry.npmjs.org/
case "$MODE" in
  private) SRC=https://packages.example.invalid ;;
esac
npm config set registry "$SRC"
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 5
    assert finding.severity == "HIGH"
    assert finding.evidence["destination"] == "unresolved"
    assert finding.evidence["destination_status"] == "unresolved"


@pytest.mark.parametrize(
    "case_body",
    [
        "SRC=https://packages.example.invalid",
        "use_private",
    ],
)
def test_one_line_case_arm_keeps_possible_redirect_high(case_body: str) -> None:
    function = (
        "use_private() { SRC=https://packages.example.invalid; }\n"
        if case_body == "use_private"
        else ""
    )
    script = f"""MODE=private
SRC=https://registry.npmjs.org/
{function}case "$MODE" in private) {case_body} ;; esac
npm config set registry "$SRC"
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.severity == "HIGH"
    assert finding.evidence["destination"] == "unresolved"
    assert finding.evidence["destination_status"] == "unresolved"


def test_definite_assignment_after_one_line_case_clears_ambiguity() -> None:
    script = """MODE=private
SRC=https://packages.example.invalid
case "$MODE" in private) SRC=https://other.example.invalid ;; esac
SRC=https://registry.npmjs.org/
npm config set registry "$SRC"
"""

    assert _analyze({"setup.sh": script}) == []


def test_assignment_shaped_command_cannot_override_real_assignment() -> None:
    script = """SRC=https://packages.example.invalid
SRC = https://registry.npmjs.org/ || true
npm config set registry "$SRC"
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.severity == "HIGH"
    assert finding.evidence["destination"] == "https://packages.example.invalid"
    assert finding.evidence["destination_status"] == "resolved"


def test_export_assignment_remains_effective_with_trailing_variable_name() -> None:
    script = """SRC=https://registry.npmjs.org/
export SRC=https://packages.example.invalid MARKER
npm config set registry "$SRC"
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.severity == "HIGH"
    assert finding.evidence["destination"] == "https://packages.example.invalid"
    assert finding.evidence["destination_status"] == "resolved"


def test_multiple_assignment_words_update_each_variable() -> None:
    script = """SRC=https://registry.npmjs.org/
MARKER=1 SRC=https://packages.example.invalid
npm config set registry "$SRC"
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 3
    assert finding.severity == "HIGH"
    assert finding.evidence["destination"] == "https://packages.example.invalid"
    assert finding.evidence["destination_status"] == "resolved"


def test_conditional_assignment_keeps_possible_noncanonical_redirect_high() -> None:
    script = """#!/bin/sh
SRC=https://packages.example.invalid
if test -f use-default; then
  SRC=https://registry.npmjs.org/
fi
npm config set registry "$SRC"
"""

    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].start_line == 6
    assert findings[0].severity == "HIGH"
    assert findings[0].evidence["destination"] == "unresolved"
    assert findings[0].evidence["destination_status"] == "unresolved"


def test_inline_conditional_assignment_keeps_possible_redirect_high() -> None:
    script = """SRC=https://registry.npmjs.org/
if test -f use-private; then SRC=https://packages.example.invalid; fi
npm config set registry "$SRC"
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 3
    assert finding.severity == "HIGH"
    assert finding.evidence["destination"] == "unresolved"


@pytest.mark.parametrize("operator", ["&&", "||"])
def test_short_circuit_assignment_keeps_possible_redirect_high(operator: str) -> None:
    script = f"""SRC=https://registry.npmjs.org/
test -f use-private {operator} SRC=https://packages.example.invalid
npm config set registry "$SRC"
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 3
    assert finding.severity == "HIGH"
    assert finding.evidence["destination"] == "unresolved"


def test_definite_assignment_after_inline_conditional_clears_ambiguity() -> None:
    script = """SRC=https://packages.example.invalid
if test -f use-private; then SRC=https://other.example.invalid; fi
SRC=https://registry.npmjs.org/
npm config set registry "$SRC"
"""

    assert _analyze({"setup.sh": script}) == []


def test_single_prior_literal_assignment_resolves_statically() -> None:
    script = """SRC=https://packages.example.invalid
npm config set registry "${SRC}"
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.evidence["destination"] == "https://packages.example.invalid"
    assert finding.evidence["destination_status"] == "resolved"


@pytest.mark.parametrize(
    "expression",
    [
        "${SRC:-https://packages.example.invalid}",
        "$(printf https://packages.example.invalid)",
        "`printf https://packages.example.invalid`",
        "$UNASSIGNED_SOURCE",
    ],
)
def test_dynamic_or_unsupported_shell_expansions_remain_unresolved(expression: str) -> None:
    finding = _analyze({"setup.sh": f"npm config set registry {expression}\n"})[0]

    assert finding.evidence["destination"] == "unresolved"
    assert finding.evidence["destination_status"] == "unresolved"
    assert finding.severity == "HIGH"


def test_unresolved_destination_is_high_trust_boundary_change() -> None:
    script = """#!/bin/sh
cat > .npmrc << EOF
registry=${SOURCE_FROM_RUNTIME}
EOF
"""

    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].severity == "HIGH"
    assert findings[0].evidence["destination"] == "unresolved"
    assert findings[0].evidence["destination_status"] == "unresolved"


def test_prose_comments_and_unrelated_registry_words_do_not_change_result() -> None:
    docs = """# Package Registry Notes
The word registry appears here with https://packages.example.invalid.
```text
npm config set registry https://packages.example.invalid
```
"""
    script = """#!/bin/sh
# This audited internal registry is completely safe.
# npm config set registry https://comment.example.invalid
echo registry
"""

    assert _analyze({"README.md": docs, "setup.sh": script}) == []


def test_actionable_shell_fence_is_analyzed_without_trusting_surrounding_prose() -> None:
    markdown = """# Setup
This source is approved and audited.
```bash
npm config set registry https://packages.example.invalid
```
"""

    findings = _analyze({"SKILL.md": markdown})

    assert len(findings) == 1
    assert findings[0].evidence["ecosystem"] == "npm"


@pytest.mark.parametrize("output_format", ["terminal", "json", "markdown", "sarif"])
def test_url_credentials_are_redacted_from_findings_and_all_reports(output_format: str) -> None:
    username = "registry-user-sentinel"
    password = "registry-password-sentinel"
    query_token = "registry-token-sentinel"
    content = (
        f"registry=https://{username}:{password}@packages.example.invalid/"
        f"?token={query_token}&channel=stable\n"
    )
    finding = _analyze({".npmrc": content})[0]

    serialized_finding = json.dumps(finding.to_dict())
    for secret in (username, password, query_token):
        assert secret not in serialized_finding
    assert "***@packages.example.invalid" in serialized_finding

    state: SkillspectorState = {
        "filtered_findings": [finding],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": output_format,
    }
    rendered = report(state)["report_body"]
    for secret in (username, password, query_token):
        assert secret not in rendered


@pytest.mark.parametrize(
    "destination",
    [
        "ssh://registry-user-sentinel:registry-password-sentinel@packages.example.invalid/index?token=registry-token-sentinel",
        "git+https://registry-user-sentinel:registry-password-sentinel@packages.example.invalid/index?token=registry-token-sentinel",
        "sparse+https://registry-user-sentinel:registry-password-sentinel@packages.example.invalid/index?token=registry-token-sentinel",
    ],
)
def test_cargo_url_credentials_are_redacted_for_supported_schemes(destination: str) -> None:
    content = f'[registries.private]\nindex = "{destination}"\n'

    finding = _analyze({".cargo/config.toml": content})[0]
    serialized = json.dumps(finding.to_dict())

    for secret in (
        "registry-user-sentinel",
        "registry-password-sentinel",
        "registry-token-sentinel",
    ):
        assert secret not in serialized
    assert "packages.example.invalid" in serialized


def test_sc10_credentials_are_redacted_before_provider_prompt_construction() -> None:
    username = "provider-user-sentinel"
    password = "provider-password-sentinel"
    token = "provider-token-sentinel"
    content = f"registry=ssh://{username}:{password}@packages.example.invalid/index?token={token}\n"
    finding = _analyze({".npmrc": content})[0]
    analyzer = LLMMetaAnalyzer.__new__(LLMMetaAnalyzer)
    analyzer.base_prompt = PER_FILE_ANALYSIS_PROMPT
    analyzer._input_budget = 100_000

    batch = analyzer.get_batches([".npmrc"], {".npmrc": content}, [finding])[0]
    prompt = analyzer.build_prompt(batch, metadata_text="No metadata available")

    for secret in (username, password, token):
        assert secret not in batch.content
        assert secret not in prompt
    assert "packages.example.invalid" in prompt


def test_hidden_source_finding_is_marked_local_only() -> None:
    findings = _analyze(
        {".npmrc": "registry=https://packages.example.invalid\n"},
        [{"path": ".npmrc", "local_only": True}],
    )

    assert findings[0].evidence["local_only"] is True
    assert "local-only" in findings[0].tags


def test_sc10_survives_optional_llm_filtering_when_unconfirmed() -> None:
    content = "registry=https://packages.example.invalid\n"
    finding = _analyze({".npmrc": content})[0]
    batch = Batch(file_path=".npmrc", content=content, findings=[finding])
    analyzer = LLMMetaAnalyzer.__new__(LLMMetaAnalyzer)

    kept = analyzer.apply_filter([finding], [(batch, [])])

    assert len(kept) == 1
    assert kept[0].rule_id == "SC10"
    assert kept[0].severity == "HIGH"
    assert kept[0] is finding
    assert kept[0].tags == ["supply-chain", "dependency-source"]


def test_sc10_provider_confirmation_cannot_replace_deterministic_fields() -> None:
    finding = _analyze({".npmrc": "registry=https://packages.example.invalid\n"})[0]
    original = finding.to_dict()
    batch = Batch(file_path=".npmrc", content="redacted", findings=[finding])
    provider_item = {
        "pattern_id": "SC10",
        "is_vulnerability": True,
        "confidence": 0.6,
        "start_line": finding.start_line,
        "explanation": "provider alternate explanation",
        "remediation": "provider alternate remediation",
        "_file": ".npmrc",
    }
    analyzer = LLMMetaAnalyzer.__new__(LLMMetaAnalyzer)

    kept = analyzer.apply_filter([finding], [(batch, [provider_item])])

    assert kept == [finding]
    assert kept[0].to_dict() == original
    assert kept[0].confidence == 1.0
    assert kept[0].message == finding.message


def test_sc10_static_only_and_provider_failure_paths_preserve_canonical_record() -> None:
    finding = _analyze({".npmrc": "registry=https://packages.example.invalid\n"})[0]

    assert _fallback_filtered([finding]) == [finding]
    assert _passthrough_with_defaults([finding]) == [finding]


def test_common_heredoc_redirection_order_is_detected_at_config_line() -> None:
    script = """cat <<EOF > .npmrc
registry=https://packages.example.invalid
EOF
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 2
    assert finding.evidence["surface"] == ".npmrc"


@pytest.mark.parametrize("delimiter", ["'END-OF'", "END-OF"])
def test_hyphenated_heredoc_delimiter_is_detected(delimiter: str) -> None:
    script = f"""cat > "$HOME/.npmrc" <<{delimiter}
registry=https://packages.example.invalid
END-OF
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 2
    assert finding.evidence["surface"] == ".npmrc"
    assert finding.evidence["destination"] == "https://packages.example.invalid"


@pytest.mark.parametrize("delimiter", ["END'-'OF", 'END"-"OF', r"END\-OF"])
def test_word_quoted_heredoc_delimiter_generates_config(delimiter: str) -> None:
    script = f"""cat > "$HOME/.npmrc" <<{delimiter}
registry=https://packages.example.invalid
END-OF
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 2
    assert finding.severity == "HIGH"
    assert finding.evidence["surface"] == ".npmrc"
    assert finding.evidence["destination"] == "https://packages.example.invalid"


@pytest.mark.parametrize("delimiter", ["END'-'OF", 'END"-"OF', r"END\-OF"])
def test_word_quoted_unrelated_heredoc_data_is_not_actionable(delimiter: str) -> None:
    script = f"""cat <<{delimiter} > instructions.txt
npm config set registry https://packages.example.invalid
END-OF
"""

    assert _analyze({"setup.sh": script}) == []


@pytest.mark.parametrize(
    ("target", "body", "ecosystem"),
    [
        (".npmrc", "registry=https://packages.example.invalid", "npm"),
        (".yarnrc", 'registry "https://packages.example.invalid"', "yarn"),
        (
            "pip.conf",
            "[global]\nindex-url=https://packages.example.invalid/simple",
            "pip",
        ),
        (
            "pyproject.toml",
            '[[tool.poetry.source]]\nname="private"\nurl="https://packages.example.invalid/simple"',
            "poetry",
        ),
        (
            "settings.xml",
            "<settings><mirrors><mirror><mirrorOf>*</mirrorOf>"
            "<url>https://packages.example.invalid/repository</url>"
            "</mirror></mirrors></settings>",
            "maven",
        ),
        (
            ".cargo/config.toml",
            '[registries.private]\nindex="sparse+https://packages.example.invalid/index"',
            "cargo",
        ),
    ],
)
def test_literal_dollar_heredoc_generates_every_supported_config(
    target: str, body: str, ecosystem: str
) -> None:
    script = f"cat > {target} <<END$OF\n{body}\nEND$OF\n"

    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].start_line >= 2
    assert findings[0].severity == "HIGH"
    assert findings[0].evidence["ecosystem"] == ecosystem
    assert "packages.example.invalid" in str(findings[0].evidence["destination"])


@pytest.mark.parametrize(
    "header",
    [
        "tee instructions.txt <<END$OF",
        "cat <<END$OF",
        "cat <<END$OF > instructions.txt",
        "cat <<END$OF >> instructions.txt",
        "cat 3<<END$OF 1>&3",
    ],
)
def test_literal_dollar_heredoc_data_is_not_actionable(header: str) -> None:
    script = f"""{header}
npm config set registry https://packages.example.invalid
MARKER=1 PIP_INDEX_URL=https://packages.example.invalid/simple
END$OF
"""

    assert _analyze({"setup.sh": script}) == []


def test_command_after_complete_literal_dollar_heredoc_is_actionable() -> None:
    script = """cat <<END$OF
npm config set registry https://inert.example.invalid
END$OF
npm config set registry https://packages.example.invalid
"""

    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].start_line == 4
    assert findings[0].evidence["destination"] == "https://packages.example.invalid"


def test_literal_dollar_heredoc_assignment_text_does_not_change_shell_state() -> None:
    script = """SRC=https://packages.example.invalid
cat <<END$OF
SRC=https://registry.npmjs.org/
END$OF
npm config set registry "$SRC"
"""

    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].start_line == 5
    assert findings[0].evidence["destination"] == "https://packages.example.invalid"


def test_generated_config_nested_in_literal_dollar_heredoc_is_inert() -> None:
    script = """tee instructions.txt <<END$OF
cat <<EOF > .npmrc
registry=https://packages.example.invalid
EOF
END$OF
"""

    assert _analyze({"setup.sh": script}) == []


@pytest.mark.parametrize(
    ("delimiter", "status"),
    [
        ("END$OF", "resolved"),
        ("'END$OF'", "unresolved"),
        ('"END$OF"', "unresolved"),
        (r"END\$OF", "unresolved"),
        ("END'$'OF", "unresolved"),
    ],
)
def test_literal_dollar_delimiter_preserves_expansion_semantics(
    delimiter: str, status: str
) -> None:
    script = f"""SOURCE=https://packages.example.invalid
cat > .npmrc <<{delimiter}
registry=$SOURCE
END$OF
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.evidence["destination_status"] == status
    assert finding.evidence["destination"] == (
        "https://packages.example.invalid" if status == "resolved" else "unresolved"
    )


def test_tab_stripping_literal_dollar_heredoc_is_supported() -> None:
    script = "cat > .npmrc <<-END$OF\n\tregistry=https://packages.example.invalid\n\tEND$OF\n"

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 2
    assert finding.evidence["destination"] == "https://packages.example.invalid"


@pytest.mark.parametrize("delimiter", ["END${OF}", "END$?", "END#OF", "END!OF"])
def test_static_punctuation_heredoc_words_are_literal_and_inert(delimiter: str) -> None:
    script = f"""cat <<{delimiter}
npm config set registry https://packages.example.invalid
{delimiter}
"""

    assert _analyze({"setup.sh": script}) == []


@pytest.mark.parametrize("delimiter", ["END${OF}", "END$?", "END#OF", "END!OF"])
def test_static_punctuation_heredoc_words_generate_config(delimiter: str) -> None:
    script = f"""cat > .npmrc <<{delimiter}
registry=https://packages.example.invalid
{delimiter}
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 2
    assert finding.evidence["destination"] == "https://packages.example.invalid"


def test_bare_braced_dollar_heredoc_keeps_body_expansion_enabled() -> None:
    script = """SOURCE=https://packages.example.invalid
cat > .npmrc <<END${OF}
registry=$SOURCE
END${OF}
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.evidence["destination"] == "https://packages.example.invalid"
    assert finding.evidence["destination_status"] == "resolved"


def test_simple_ansi_c_quoted_heredoc_word_disables_body_expansion() -> None:
    script = """SOURCE=https://packages.example.invalid
cat > .npmrc <<$'END$OF'
registry=$SOURCE
END$OF
"""

    finding = _analyze({"setup.bash": script})[0]

    assert finding.evidence["destination"] == "unresolved"
    assert finding.evidence["destination_status"] == "unresolved"


def test_locale_quoted_heredoc_word_is_inert_and_disables_expansion() -> None:
    script = """SOURCE=https://packages.example.invalid
cat > .npmrc <<$"END$OF"
registry=$SOURCE
END$OF
"""

    finding = _analyze({"setup.bash": script})[0]

    assert finding.evidence["destination"] == "unresolved"
    assert finding.evidence["destination_status"] == "unresolved"


def test_heredoc_inside_quoted_command_substitution_is_inert() -> None:
    script = """value="$(cat <<END$OF
npm config set registry https://inert.example.invalid
END$OF
)"
npm config set registry https://packages.example.invalid
"""

    findings = _analyze({"setup.bash": script})

    assert len(findings) == 1
    assert findings[0].start_line == 5
    assert findings[0].evidence["destination"] == "https://packages.example.invalid"


def test_final_stdin_heredoc_controls_generated_config() -> None:
    script = """cat > .npmrc <<DECOY <<END$OF
registry=https://registry.npmjs.org/
DECOY
registry=https://packages.example.invalid
END$OF
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 4
    assert finding.evidence["destination"] == "https://packages.example.invalid"


def test_overridden_stdin_heredoc_does_not_generate_config() -> None:
    script = """cat > .npmrc <<DECOY <<END$OF
registry=https://packages.example.invalid
DECOY
registry=https://registry.npmjs.org/
END$OF
"""

    assert _analyze({"setup.sh": script}) == []


def test_nonstdin_heredoc_does_not_override_final_stdin_body() -> None:
    script = """cat > .npmrc <<FIRST 3<<SIDE <<FINAL
registry=https://registry.npmjs.org/
FIRST
npm config set registry https://inert.example.invalid
SIDE
registry=https://packages.example.invalid
FINAL
"""

    findings = _analyze({"setup.sh": script})

    assert len(findings) == 1
    assert findings[0].start_line == 6
    assert findings[0].evidence["destination"] == "https://packages.example.invalid"


@pytest.mark.parametrize(
    ("redirect", "expected"),
    [
        (">", True),
        (">>", True),
        ("1>", True),
        ("1>>", True),
        ("2>", False),
        ("3>>", False),
    ],
)
def test_generated_config_requires_stdout_file_redirect(redirect: str, expected: bool) -> None:
    script = f"""cat {redirect} .npmrc <<END$OF
registry=https://packages.example.invalid
END$OF
"""

    findings = _analyze({"setup.sh": script})

    assert bool(findings) is expected
    if expected:
        assert findings[0].evidence["destination"] == "https://packages.example.invalid"


def test_last_stdout_redirect_controls_generated_target() -> None:
    inert = """cat > .npmrc > instructions.txt <<END$OF
registry=https://packages.example.invalid
END$OF
"""
    generated = """cat > instructions.txt >> .npmrc <<END$OF
registry=https://packages.example.invalid
END$OF
"""

    assert _analyze({"setup.sh": inert}) == []
    assert _analyze({"setup.sh": generated})[0].start_line == 2


@pytest.mark.parametrize(
    ("operands", "expected"),
    [
        ("", True),
        ("-", True),
        ("-n -", True),
        ("existing.txt", False),
        ("-- existing.txt", False),
        ("existing.txt -", True),
    ],
)
def test_generated_cat_body_requires_a_stdin_operand(operands: str, expected: bool) -> None:
    script = f"""cat {operands} > .npmrc <<END$OF
registry=https://packages.example.invalid
END$OF
"""

    findings = _analyze({"setup.sh": script})

    assert bool(findings) is expected
    if expected:
        assert findings[0].evidence["destination"] == "https://packages.example.invalid"


def test_final_ordinary_stdin_redirect_overrides_generated_heredoc() -> None:
    overridden = """cat > .npmrc <<END$OF < existing.txt
registry=https://packages.example.invalid
END$OF
"""
    effective = """cat > .npmrc < existing.txt <<END$OF
registry=https://packages.example.invalid
END$OF
"""

    assert _analyze({"setup.sh": overridden}) == []
    assert _analyze({"setup.sh": effective})[0].start_line == 2


@pytest.mark.parametrize("arithmetic", ["x=$((1 << 2))", "(( x << 2 ))"])
def test_arithmetic_shift_does_not_mask_later_command(arithmetic: str) -> None:
    script = f"""{arithmetic}
npm config set registry https://packages.example.invalid
2
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 2
    assert finding.evidence["destination"] == "https://packages.example.invalid"


def test_generated_heredoc_header_scan_is_bounded_on_one_long_line() -> None:
    script = "cat " + "x>" * 20_000 + " no-redirection-target\n"

    assert _analyze({"setup.sh": script}) == []


def test_hyphenated_generic_heredoc_does_not_hide_later_command() -> None:
    script = """cat <<END-OF > instructions.txt
not executable
END-OF
npm config set registry https://packages.example.invalid
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 4
    assert finding.evidence["surface"] == "npm config set"


def test_unterminated_literal_dollar_heredoc_does_not_hide_later_command() -> None:
    script = """cat <<END$OF > instructions.txt
not executable
npm config set registry https://packages.example.invalid
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 3
    assert finding.evidence["surface"] == "npm config set"


def test_mismatched_literal_dollar_terminator_does_not_hide_later_command() -> None:
    script = """cat <<END$OF > instructions.txt
not executable
ENDOF
npm config set registry https://packages.example.invalid
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 4
    assert finding.evidence["surface"] == "npm config set"


def test_dynamic_heredoc_word_is_not_partially_accepted() -> None:
    script = """cat <<END$(printf OF) > instructions.txt
npm config set registry https://packages.example.invalid
END$
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 2
    assert finding.evidence["surface"] == "npm config set"


def test_unmatched_word_quote_does_not_partially_consume_later_command() -> None:
    script = """cat <<END'-OF > instructions.txt
not executable
npm config set registry https://packages.example.invalid
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 3
    assert finding.evidence["surface"] == "npm config set"


def test_command_text_in_unrelated_heredoc_is_not_actionable() -> None:
    script = """#!/bin/sh
cat <<'EOF' > instructions.txt
npm config set registry https://packages.example.invalid
EOF
"""

    assert _analyze({"setup.sh": script}) == []


@pytest.mark.parametrize(
    "header",
    [
        "tee instructions.txt <<'EOF'",
        "cat <<'EOF'",
        "cat <<'EOF' >> instructions.txt",
        "cat 3<<'EOF' 1>&3",
    ],
)
def test_command_text_in_generic_heredoc_is_not_actionable(header: str) -> None:
    script = f"{header}\nnpm config set registry https://packages.example.invalid\nEOF\n"

    assert _analyze({"setup.sh": script}) == []


def test_generated_config_text_nested_in_unrelated_heredoc_is_not_actionable() -> None:
    script = """tee instructions.txt <<'OUTER'
cat <<EOF > .npmrc
registry=https://packages.example.invalid
EOF
OUTER
"""

    assert _analyze({"setup.sh": script}) == []


def test_dependency_source_command_in_pipeline_stage_is_actionable() -> None:
    script = "printf y | npm config set registry https://packages.example.invalid\n"

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 1
    assert finding.severity == "HIGH"
    assert finding.evidence["ecosystem"] == "npm"
    assert finding.evidence["destination"] == "https://packages.example.invalid"


def test_quoted_heredoc_delimiter_does_not_expand_variables() -> None:
    script = """SOURCE=https://packages.example.invalid
cat <<'EOF' > .npmrc
registry=${SOURCE}
EOF
"""

    finding = _analyze({"setup.sh": script})[0]

    assert finding.start_line == 3
    assert finding.evidence["destination"] == "unresolved"
    assert finding.evidence["destination_status"] == "unresolved"


@pytest.mark.parametrize("delimiter", ["EOF", "END$OF"])
def test_repeated_unmatched_heredocs_are_bounded_and_do_not_produce_sc10(
    delimiter: str,
) -> None:
    script = "\n".join(f"cat <<{delimiter} > .npmrc" for _ in range(2_000))

    assert _analyze({"setup.sh": script}) == []


def test_echoed_and_source_language_command_text_is_not_actionable() -> None:
    destination = "https://packages.example.invalid"
    files = {
        "setup.sh": f"echo npm config set registry {destination}\n",
        "example.py": f'command = "npm config set registry {destination}"\n',
        "example.js": f'const command = "npm config set registry {destination}";\n',
    }

    assert _analyze(files) == []


def test_pip_short_index_option_is_detected() -> None:
    finding = _analyze(
        {"setup.sh": "pip install -i https://packages.example.invalid/simple package-name\n"}
    )[0]

    assert finding.evidence["ecosystem"] == "pip"
    assert finding.evidence["operation"] == "replace"


def test_extensionless_executable_shell_script_is_actionable() -> None:
    content = "#!/bin/sh\nnpm config set registry https://packages.example.invalid\n"
    metadata = [{"path": "bootstrap", "executable": True}]

    finding = _analyze({"bootstrap": content}, metadata)[0]

    assert finding.start_line == 2
    assert finding.evidence["ecosystem"] == "npm"
