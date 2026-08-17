# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic regression tests for dependency-source redirection."""

from __future__ import annotations

import json

import pytest

from skillspector.dependency_sources import analyze_dependency_sources
from skillspector.llm_analyzer_base import Batch
from skillspector.models import Finding
from skillspector.nodes.meta_analyzer import LLMMetaAnalyzer
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
    assert "llm-unconfirmed" in kept[0].tags
