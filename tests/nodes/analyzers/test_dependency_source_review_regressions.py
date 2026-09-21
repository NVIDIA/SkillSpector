# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Review regressions for dependency-source context and workflow resource bounds."""

from types import SimpleNamespace

import pytest

from skillspector import dependency_sources
from skillspector.dependency_sources import (
    analyze_dependency_sources,
    analyze_dependency_sources_detailed,
)
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.models import Finding
from skillspector.nodes.analyzers import static_patterns_supply_chain as supply_chain


@pytest.mark.parametrize(
    "same_line",
    [
        'SRC=https://packages.example.invalid; npm config set registry "$SRC"',
        'npm config set registry "$SRC"; SRC=https://packages.example.invalid',
        'use_private; npm config set registry "$SRC"',
    ],
)
def test_same_line_state_changes_cannot_reuse_an_old_canonical_value(same_line):
    content = (
        "use_private() { SRC=https://packages.example.invalid; }\n"
        "SRC=https://registry.npmjs.org/\n"
        f"{same_line}\n"
    )
    findings = analyze_dependency_sources(["setup.sh"], {"setup.sh": content})

    assert len(findings) == 1
    assert findings[0].start_line == 3
    assert findings[0].evidence["destination_status"] == "unresolved"


@pytest.mark.parametrize("name", ["README.md", "SKILL.md"])
def test_markdown_prose_assignments_cannot_resolve_shell_fence_variables(name):
    content = 'SRC=https://registry.npmjs.org/\n```bash\nnpm config set registry "$SRC"\n```\n'
    findings = analyze_dependency_sources([name], {name: content})

    assert len(findings) == 1
    assert findings[0].start_line == 3
    assert findings[0].evidence["destination"] == "unresolved"


@pytest.mark.parametrize(
    ("path", "content"),
    [
        (".npmrc", "SRC=https://registry.npmjs.org/\nregistry=${SRC}\n"),
        (".yarnrc", 'SRC=https://registry.yarnpkg.com/\nregistry "$SRC"\n'),
        ("pip.conf", "[global]\nSRC=https://pypi.org/simple/\nindex-url=$SRC\n"),
        (
            "pyproject.toml",
            'SRC="https://pypi.org/simple/"\n[[tool.poetry.source]]\nurl="$SRC"\n',
        ),
        (
            "settings.xml",
            "<settings>\nSRC=https://repo.maven.apache.org/maven2/\n"
            "<mirrors><mirror><url>${SRC}</url></mirror></mirrors>\n</settings>",
        ),
        (
            ".cargo/config.toml",
            'SRC="sparse+https://index.crates.io/"\n[source.crates-io]\nregistry="$SRC"\n',
        ),
    ],
)
def test_direct_config_assignment_shaped_data_cannot_shadow_environment(path, content):
    findings = analyze_dependency_sources([path], {path: content})

    assert len(findings) == 1
    assert findings[0].evidence["destination"] == "unresolved"


@pytest.mark.parametrize("tool", ['"bad"', "3", "[]", "false"])
def test_non_mapping_toml_tool_preserves_earlier_supply_chain_results(monkeypatch, tool):
    earlier = Finding(rule_id="SC2", severity="HIGH", file="setup.sh", message="Existing risk")
    monkeypatch.setattr(
        supply_chain.static_runner,
        "run_static_patterns_with_ledger",
        lambda *_args: {
            "findings": [earlier],
            "inspection_ledger": [],
            "analyzer_status_events": [],
        },
    )
    files = {"pyproject.toml": f"tool = {tool}\n"}

    response = supply_chain.node({"components": list(files), "file_cache": files})

    assert response["findings"] == [earlier]
    assert not any(
        event["outcome"] is LedgerOutcome.FAILED for event in response["inspection_ledger"]
    )


@pytest.mark.parametrize(
    ("target", "canonical", "body"),
    [
        (
            "pyproject.toml",
            "https://pypi.org/simple/",
            '[[tool.poetry.source]]\nname="private"\nurl="$SRC"\n',
        ),
        (
            "settings.xml",
            "https://repo.maven.apache.org/maven2/",
            "<settings><mirrors><mirror>\n<id>private</id>\n<url>$SRC</url>\n"
            "</mirror></mirrors></settings>\n",
        ),
        (
            ".cargo/config.toml",
            "sparse+https://index.crates.io/",
            '[source.crates-io]\n# generated source\nregistry="$SRC"\n',
        ),
    ],
)
@pytest.mark.parametrize("latest_is_canonical", [False, True])
def test_generated_config_resolves_at_original_script_position(
    target, canonical, body, latest_is_canonical
):
    custom = "https://packages.example.invalid/"
    first, latest = (custom, canonical) if latest_is_canonical else (canonical, custom)
    script = f"SRC={first}\n# padding\n# padding\nSRC={latest}\ncat > {target} << EOF\n{body}EOF\n"

    findings = analyze_dependency_sources(["setup.sh"], {"setup.sh": script})

    if latest_is_canonical:
        assert findings == []
    else:
        assert len(findings) == 1
        assert findings[0].start_line == 8
        assert findings[0].evidence["destination"] == custom
        assert findings[0].evidence["destination_status"] == "resolved"


def test_large_registry_file_stops_before_allocating_all_changes(monkeypatch):
    content = "registry=https://packages.example.invalid\n" * 20_001
    resolved = 0
    original = dependency_sources._resolve_value

    def count_resolutions(*args):
        nonlocal resolved
        resolved += 1
        return original(*args)

    monkeypatch.setattr(dependency_sources, "_resolve_value", count_resolutions)
    result = analyze_dependency_sources_detailed([".npmrc"], {".npmrc": content})

    assert len(result.findings) == 10_000
    assert resolved == 10_000
    assert [finding.start_line for finding in result.findings] == list(range(1, 10_001))
    assert len(result.limitations) == 1
    assert result.limitations[0].reason is LedgerReason.OUTPUT_LIMIT
    assert result.limitations[0].limit_findings == 10_000


def test_finding_allowance_is_shared_across_components():
    files = {f"{name}/.npmrc": "registry=https://packages.example.invalid\n" for name in "abc"}
    result = analyze_dependency_sources_detailed(list(files), files, max_findings=2)

    assert [finding.file for finding in result.findings] == ["a/.npmrc", "b/.npmrc"]
    assert result.limitations[0].path == "c/.npmrc"
    assert result.limitations[0].reason is LedgerReason.OUTPUT_LIMIT


def test_deadline_expiring_inside_a_config_preserves_collected_findings(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(dependency_sources, "time", SimpleNamespace(monotonic=lambda: clock.now))
    original = dependency_sources._finding

    def consume_time(*args, **kwargs):
        finding = original(*args, **kwargs)
        clock.now = 0.5
        return finding

    monkeypatch.setattr(dependency_sources, "_finding", consume_time)
    content = "registry=https://packages.example.invalid\n" * 20
    result = analyze_dependency_sources_detailed(
        [".npmrc"], {".npmrc": content}, timeout_seconds=0.25
    )

    assert len(result.findings) == 1
    assert result.limitations[0].reason is LedgerReason.RUNTIME_LIMIT
    assert result.limitations[0].observed_seconds == 0.5
    assert result.limitations[0].limit_seconds == 0.25


@pytest.mark.parametrize("max_findings", [1, 2])
def test_node_passes_allowance_remaining_after_previous_findings(monkeypatch, max_findings):
    earlier = Finding(rule_id="SC2", severity="HIGH", file="setup.sh", message="Existing risk")
    monkeypatch.setattr(supply_chain, "MAX_FINDING_OUTPUT_RECORDS", max_findings)
    monkeypatch.setattr(
        supply_chain.static_runner,
        "run_static_patterns_with_ledger",
        lambda *_args: {
            "findings": [earlier],
            "inspection_ledger": [],
            "analyzer_status_events": [],
        },
    )
    files = {".npmrc": "registry=https://packages.example.invalid\n" * 3}

    response = supply_chain.node({"components": list(files), "file_cache": files})

    assert len(response["findings"]) == max_findings
    assert response["findings"][0] is earlier
    partial = [
        event
        for event in response["inspection_ledger"]
        if event["outcome"] is LedgerOutcome.PARTIAL
    ]
    assert len(partial) == 1
    assert partial[0]["reason_code"] is LedgerReason.OUTPUT_LIMIT
    assert partial[0]["path"] == ".npmrc"
    assert "dependency_source" in partial[0]["analyzer_id"]
    assert response["analyzer_status_events"][0]["status"] == "degraded"


def test_node_reports_expired_workflow_deadline_without_losing_previous_findings(monkeypatch):
    earlier = Finding(rule_id="SC2", severity="HIGH", file="setup.sh", message="Existing risk")
    monkeypatch.setattr(
        supply_chain.static_runner,
        "run_static_patterns_with_ledger",
        lambda *_args: {
            "findings": [earlier],
            "inspection_ledger": [],
            "analyzer_status_events": [],
        },
    )
    files = {".npmrc": "registry=https://packages.example.invalid\n"}
    response = supply_chain.node(
        {
            "components": list(files),
            "file_cache": files,
            "workflow_resource_budget": SimpleNamespace(remaining_seconds=lambda: 0.0),
        }
    )

    assert response["findings"] == [earlier]
    assert any(
        event["outcome"] is LedgerOutcome.PARTIAL
        and event["reason_code"] is LedgerReason.RUNTIME_LIMIT
        and "dependency_source" in event["analyzer_id"]
        for event in response["inspection_ledger"]
    )
    assert response["analyzer_status_events"][0]["status"] == "degraded"


@pytest.mark.parametrize(
    ("path", "module", "parse_name", "parsed"),
    [
        ("pyproject.toml", "tomllib", "loads", {}),
        (".cargo/config.toml", "tomllib", "loads", {}),
        ("settings.xml", "ET", "fromstring", dependency_sources.ET.fromstring("<settings/>")),
    ],
)
def test_whole_document_parse_expiration_is_partial_even_without_sources(
    monkeypatch, path, module, parse_name, parsed
):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(dependency_sources, "time", SimpleNamespace(monotonic=lambda: clock.now))

    def slow_parse(_content):
        clock.now = 0.5
        return parsed

    monkeypatch.setattr(getattr(dependency_sources, module), parse_name, slow_parse)
    result = analyze_dependency_sources_detailed([path], {path: "data"}, timeout_seconds=0.25)

    assert result.findings == []
    assert len(result.limitations) == 1
    assert result.limitations[0].reason is LedgerReason.RUNTIME_LIMIT
    assert result.limitations[0].path == path


def test_exact_finding_cap_on_final_record_is_complete():
    result = analyze_dependency_sources_detailed(
        [".npmrc"], {".npmrc": "registry=https://packages.example.invalid\n"}, max_findings=1
    )

    assert len(result.findings) == 1
    assert result.limitations == []
