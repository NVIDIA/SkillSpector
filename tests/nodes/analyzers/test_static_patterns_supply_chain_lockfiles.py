# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pytest

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_supply_chain as supply_chain
from skillspector.nodes.analyzers.osv_client import OsvQueryLimitation, QueryBatchResults


def _capture_osv_packages(monkeypatch):
    seen = {}

    def fake_query_batch(packages, ecosystem, **_kwargs):
        seen["packages"] = packages
        seen["ecosystem"] = ecosystem
        return [[] for _ in packages]

    monkeypatch.setattr(supply_chain, "query_batch", fake_query_batch)
    return seen


def test_uv_lock_versions_are_passed_to_osv(monkeypatch):
    seen = _capture_osv_packages(monkeypatch)
    content = """
version = 1
[[package]]
name = "mlx"
version = "0.31.2"
[[package]]
name = "requests"
version = "2.31.0"
"""
    supply_chain._analyze_dependencies(content, "uv.lock")
    assert seen["ecosystem"] == supply_chain.ECOSYSTEM_PYPI
    assert ("mlx", "0.31.2") in seen["packages"]
    assert ("requests", "2.31.0") in seen["packages"]


def test_poetry_lock_versions_are_passed_to_osv(monkeypatch):
    seen = _capture_osv_packages(monkeypatch)
    content = """
[[package]]
name = "jinja2"
version = "3.1.6"
description = "A fast template engine."
"""
    supply_chain._analyze_dependencies(content, "poetry.lock")
    assert seen["ecosystem"] == supply_chain.ECOSYSTEM_PYPI
    assert ("jinja2", "3.1.6") in seen["packages"]


def test_pyproject_unpinned_dependency_uses_locked_version_for_osv(monkeypatch):
    seen = _capture_osv_packages(monkeypatch)
    content = """
[project]
dependencies = [
    "mlx",
]
"""
    supply_chain._analyze_dependencies(content, "pyproject.toml", {"mlx": "0.31.2"})
    assert ("mlx", "0.31.2") in seen["packages"]


def test_requirements_unpinned_dependency_uses_locked_version_for_osv(monkeypatch):
    seen = _capture_osv_packages(monkeypatch)
    content = """
fastmcp
"""
    supply_chain._analyze_dependencies(content, "requirements.txt", {"fastmcp": "3.3.1"})
    assert ("fastmcp", "3.3.1") in seen["packages"]


def test_toml_lock_parser_anchors_line_numbers_to_package_blocks():
    content = """
[[package]]
name = "root"
version = "1.0.0"
dependencies = [
    { name = "requests" },
]
[[package]]
name = "requests"
version = "2.31.0"
"""
    packages = supply_chain._extract_packages_from_toml_lock(content)
    line_by_name = {name: line_num for name, _version, line_num in packages}
    assert content.splitlines()[line_by_name["requests"] - 1].strip() == 'name = "requests"'


def test_toml_lock_parser_returns_empty_for_malformed_toml():
    content = """
[[package]
name = "broken"
"""
    assert supply_chain._extract_packages_from_toml_lock(content) == []


@pytest.mark.parametrize("path", ["package.json", "package-lock.json", "npm-shrinkwrap.json"])
@pytest.mark.parametrize("is_cached", [True, False], ids=["present-empty", "missing"])
def test_node_distinguishes_empty_dependency_file_from_missing_cache_entry(
    monkeypatch, path, is_cached
):
    monkeypatch.setattr(
        supply_chain.static_runner,
        "run_static_patterns_with_ledger",
        lambda _state, _modules: {
            "findings": [],
            "inspection_ledger": [],
            "analyzer_status_events": [],
        },
    )
    file_cache = {path: ""} if is_cached else {}
    response = supply_chain.node(
        {
            "components": [path],
            "file_cache": file_cache,
            "local_file_cache": file_cache,
            "manifest": {},
            "component_metadata": [],
        }
    )

    has_parse_error = any(
        event["reason_code"] is LedgerReason.DEPENDENCY_PARSE_ERROR
        for event in response["inspection_ledger"]
    )
    assert has_parse_error is is_cached


def test_node_treats_empty_pyproject_toml_as_valid(monkeypatch):
    monkeypatch.setattr(
        supply_chain.static_runner,
        "run_static_patterns_with_ledger",
        lambda _state, _modules: {
            "findings": [],
            "inspection_ledger": [],
            "analyzer_status_events": [],
        },
    )
    response = supply_chain.node(
        {
            "components": ["pyproject.toml"],
            "file_cache": {"pyproject.toml": ""},
            "local_file_cache": {"pyproject.toml": ""},
            "manifest": {},
            "component_metadata": [],
        }
    )

    assert not any(
        event["reason_code"] is LedgerReason.DEPENDENCY_PARSE_ERROR
        for event in response["inspection_ledger"]
    )


@pytest.mark.parametrize(
    "path",
    [
        "requirements/dev.txt",
        "requirements/base.txt",
        "dev-requirements.txt",
        "test_requirements.txt",
        "tools/docs-requirements.in",
    ],
)
def test_nested_requirements_manifests_retain_sc4_advisory_findings(monkeypatch, path):
    seen = _capture_osv_packages(monkeypatch)
    monkeypatch.setattr(supply_chain, "was_osv_reachable", lambda budget=None: True)
    monkeypatch.setattr(
        supply_chain.static_runner,
        "run_static_patterns_with_ledger",
        lambda _state, _modules: {
            "findings": [],
            "inspection_ledger": [],
            "analyzer_status_events": [],
        },
    )
    content = "pyyaml==5.3.1\n"
    response = supply_chain.node(
        {
            "components": [path],
            "file_cache": {path: content},
            "local_file_cache": {path: content},
            "manifest": {},
            "component_metadata": [],
        }
    )

    assert ("pyyaml", "5.3.1") in seen["packages"]
    assert any(
        finding.rule_id == "SC4" and finding.file == path and "CVE-2020-14343" in finding.message
        for finding in response["findings"]
    )


@pytest.mark.parametrize(
    "path",
    [
        "docs/pyproject.toml.md",
        "docs/package.json.md",
        "docs/uv.lock.md",
        "docs/requirements/dev.txt.md",
    ],
)
def test_manifest_documentation_is_not_analyzed_as_dependency_metadata(monkeypatch, path):
    monkeypatch.setattr(
        supply_chain.static_runner,
        "run_static_patterns_with_ledger",
        lambda _state, _modules: {
            "findings": [],
            "inspection_ledger": [],
            "analyzer_status_events": [],
        },
    )
    content = "This guide describes package.json and pyproject.toml files.\n"
    response = supply_chain.node(
        {
            "components": [path],
            "file_cache": {path: content},
            "local_file_cache": {path: content},
            "manifest": {},
            "component_metadata": [],
        }
    )

    assert not any(
        event["reason_code"] is LedgerReason.DEPENDENCY_PARSE_ERROR
        for event in response["inspection_ledger"]
    )
    assert not any(finding.rule_id == "SC4" for finding in response["findings"])


@pytest.mark.parametrize(
    ("path", "content"),
    [
        ("pyproject.toml", "[project]\ndependencies = ["),
        ("package.json", '{"dependencies": {'),
        ("package-lock.json", '{"packages": '),
        ("npm-shrinkwrap.json", '{"dependencies": '),
        ("uv.lock", '[[package]\nname = "broken"'),
    ],
)
def test_malformed_dependency_manifests_are_reported_as_partial(path, content):
    _findings, limitations, _packages_seen = supply_chain._analyze_dependencies_detailed(
        content,
        path,
    )

    assert any(item.reason is LedgerReason.DEPENDENCY_PARSE_ERROR for item in limitations)


@pytest.mark.parametrize(
    ("path", "prefix"),
    [
        ("pyproject.toml", "[project]\nname = "),
        ("uv.lock", "version = "),
    ],
)
def test_oversized_toml_integer_is_reported_as_dependency_parse_error(path, prefix):
    content = prefix + "9" * 5_000

    _findings, limitations, _packages_seen = supply_chain._analyze_dependencies_detailed(
        content,
        path,
    )

    assert any(item.reason is LedgerReason.DEPENDENCY_PARSE_ERROR for item in limitations)


@pytest.mark.parametrize(
    ("path", "content"),
    [
        ("pyproject.toml", "[project]\nname = 'example'\n"),
        ("package.json", '{"name": "example"}'),
        ("uv.lock", "version = 1\n"),
    ],
)
def test_valid_dependency_free_manifests_remain_complete(path, content):
    _findings, limitations, _packages_seen = supply_chain._analyze_dependencies_detailed(
        content,
        path,
    )

    assert not any(item.reason is LedgerReason.STATIC_PARSE_LIMIT for item in limitations)


def test_malformed_package_json_keeps_line_scan_fallback(monkeypatch):
    seen = _capture_osv_packages(monkeypatch)
    content = '{"dependencies": {\n  "demo-package": "1.2.3"\n'

    _findings, limitations, packages_seen = supply_chain._analyze_dependencies_detailed(
        content,
        "package.json",
    )

    assert packages_seen == 1
    assert ("demo-package", "1.2.3") in seen["packages"]
    assert any(item.reason is LedgerReason.DEPENDENCY_PARSE_ERROR for item in limitations)


def test_toml_lock_parser_keeps_package_without_version():
    content = """
[[package]]
name = "local-package"
"""
    packages = supply_chain._extract_packages_from_toml_lock(content)
    assert packages[0][:2] == ("local-package", None)


def test_many_requirements_are_parsed_and_queried_under_one_cap(monkeypatch):
    seen = _capture_osv_packages(monkeypatch)
    monkeypatch.setattr(supply_chain, "MAX_DEPENDENCY_PACKAGES_PER_FILE", 3)
    content = "".join(f"package-{index}==1.0.0\n" for index in range(20))

    _findings, limitations, packages_seen = supply_chain._analyze_dependencies_detailed(
        content,
        "requirements.txt",
    )

    assert packages_seen == 3
    assert len(seen["packages"]) == 3
    assert any(item.reason is LedgerReason.OUTPUT_LIMIT for item in limitations)


def test_dependency_findings_are_capped_with_partial_metadata(monkeypatch):
    _capture_osv_packages(monkeypatch)
    content = "nose==1.3.7\nreqeusts==2.31.0\n"

    findings, limitations, packages_seen = supply_chain._analyze_dependencies_detailed(
        content,
        "requirements.txt",
        max_findings=1,
    )

    assert packages_seen == 2
    assert len(findings) == 1
    assert any(item.reason is LedgerReason.OUTPUT_LIMIT for item in limitations)


def test_oversized_requirement_spec_is_omitted_as_partial(monkeypatch):
    seen = _capture_osv_packages(monkeypatch)
    monkeypatch.setattr(supply_chain, "MAX_DEPENDENCY_SPEC_CHARS", 16)

    _findings, limitations, packages_seen = supply_chain._analyze_dependencies_detailed(
        f"{'a' * 32}==1.0.0\n",
        "requirements.txt",
    )

    assert packages_seen == 0
    assert seen["packages"] == []
    assert any(item.reason is LedgerReason.SIZE_LIMIT for item in limitations)


def test_dependency_provider_limit_is_partial_in_node_ledger(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("demo==1.0.0\n", encoding="utf-8")

    monkeypatch.setattr(
        supply_chain.static_runner,
        "run_static_patterns_with_ledger",
        lambda _state, _modules: {
            "findings": [],
            "inspection_ledger": [],
            "analyzer_status_events": [],
        },
    )

    def limited_query(packages, _ecosystem, **_kwargs):
        return QueryBatchResults(
            [[] for _ in packages],
            limitations=(
                OsvQueryLimitation(
                    reason=LedgerReason.TOTAL_BYTES_LIMIT,
                    observed_bytes=9,
                    limit_bytes=8,
                ),
            ),
        )

    monkeypatch.setattr(supply_chain, "query_batch", limited_query)
    response = supply_chain.node(
        {
            "skill_path": "",
            "components": ["requirements.txt"],
            "file_cache": {"requirements.txt": "demo==1.0.0\n"},
            "local_file_cache": {"requirements.txt": "demo==1.0.0\n"},
            "manifest": {},
            "component_metadata": [],
        }
    )

    partial = [
        event
        for event in response["inspection_ledger"]
        if event["outcome"] is LedgerOutcome.PARTIAL
    ]
    assert len(partial) == 1
    assert partial[0]["reason_code"] is LedgerReason.TOTAL_BYTES_LIMIT
    assert response["analyzer_status_events"][0]["status"] == "degraded"


def test_dependency_file_cap_is_aggregate_across_bundle(monkeypatch):
    monkeypatch.setattr(supply_chain, "MAX_DEPENDENCY_FILES_PER_SCAN", 1)
    monkeypatch.setattr(
        supply_chain.static_runner,
        "run_static_patterns_with_ledger",
        lambda _state, _modules: {
            "findings": [],
            "inspection_ledger": [],
            "analyzer_status_events": [],
        },
    )
    calls = 0

    def empty_query(packages, _ecosystem, **_kwargs):
        nonlocal calls
        calls += 1
        return QueryBatchResults([[] for _ in packages])

    monkeypatch.setattr(supply_chain, "query_batch", empty_query)
    response = supply_chain.node(
        {
            "skill_path": "",
            "components": ["requirements-a.txt", "requirements-b.txt"],
            "file_cache": {
                "requirements-a.txt": "one==1.0.0\n",
                "requirements-b.txt": "two==1.0.0\n",
            },
            "local_file_cache": {
                "requirements-a.txt": "one==1.0.0\n",
                "requirements-b.txt": "two==1.0.0\n",
            },
            "manifest": {},
            "component_metadata": [],
        }
    )

    assert calls == 1
    assert any(
        event["outcome"] is LedgerOutcome.PARTIAL
        and event["path"] == "requirements-b.txt"
        and event["reason_code"] is LedgerReason.OUTPUT_LIMIT
        for event in response["inspection_ledger"]
    )


@pytest.mark.parametrize("lock_version", [1, 2, 3])
@pytest.mark.parametrize("reverse", [False, True])
def test_npm_locks_do_not_resolve_other_projects(lock_version, reverse):
    def lock(version):
        field = (
            {"dependencies": {"node-forge": {"version": version}}}
            if lock_version == 1
            else {"packages": {"node_modules/node-forge": {"version": version}}}
        )
        return json.dumps({"lockfileVersion": lock_version, **field})

    cache = {
        "a/package-lock.json": lock("1.2.1"),
        "b/package-lock.json": lock("1.4.2"),
        "b/package.json": '{"dependencies": {"node-forge": "^1.4.2"}}',
        "c/package.json": '{"dependencies": {"node-forge": "^1.4.2"}}',
        "b/independent/package.json": '{"dependencies": {"node-forge": "^1.4.2"}}',
    }
    components = list(reversed(cache)) if reverse else list(cache)
    resolved = supply_chain._collect_npm_locked_versions(cache, components)
    assert resolved["a"]["node-forge"] == "1.2.1"
    assert resolved["b"]["node-forge"] == "1.4.2"
    assert not resolved.get("c")
    assert not resolved.get("b/independent")
    # Every installed version still participates in its own lockfile audit.
    assert supply_chain._extract_packages_from_npm_lock(cache["a/package-lock.json"])[0][:2] == (
        "node-forge",
        "1.2.1",
    )


@pytest.mark.parametrize("manifest_name", ["package.json", "PACKAGE.JSON"])
def test_npm_workspace_uses_nearest_install_and_keeps_hoisted_dependencies(manifest_name):
    cache = {
        "package-lock.json": json.dumps(
            {
                "lockfileVersion": 3,
                "packages": {
                    "": {"workspaces": ["packages/*"]},
                    "packages/a": {"name": "a", "version": "1.0.0"},
                    "packages/b": {"name": "b", "version": "1.0.0"},
                    "node_modules/node-forge": {"version": "1.4.2"},
                    "node_modules/@scope/helper": {"version": "2.0.0"},
                    "packages/a/node_modules/node-forge": {"version": "1.2.1"},
                    "node_modules/unrelated/node_modules/node-forge": {"version": "0.9.0"},
                },
            }
        ),
        f"packages/a/{manifest_name}": "{}",
        f"packages/b/{manifest_name}": "{}",
        f"packages/c/{manifest_name}": "{}",
    }
    resolved = supply_chain._collect_npm_locked_versions(cache, list(cache))
    assert resolved["."]["node-forge"] == "1.4.2"
    assert dict(resolved["packages/a"]) == {"node-forge": "1.2.1", "@scope/helper": "2.0.0"}
    assert dict(resolved["packages/b"]) == {"node-forge": "1.4.2", "@scope/helper": "2.0.0"}
    assert not resolved.get("packages/c")


@pytest.mark.parametrize("own_lock", ["", "{", '{"packages": {}}'])
@pytest.mark.parametrize("reverse", [False, True])
def test_npm_workspace_own_lock_prevents_ancestor_resolution(own_lock, reverse):
    cache = {
        "package-lock.json": json.dumps(
            {"packages": {"packages/a": {}, "node_modules/ws": {"version": "8.20.0"}}}
        ),
        "packages/a/package.json": '{"dependencies": {"ws": "^8.20.0"}}',
        "packages/a/package-lock.json": own_lock,
    }
    components = list(reversed(cache)) if reverse else list(cache)
    resolved = supply_chain._collect_npm_locked_versions(cache, components)
    assert not resolved.get("packages/a")


def test_deep_npm_own_lock_keeps_parse_error_without_borrowing_parent_pin(monkeypatch):
    seen = []

    def capture_packages(packages, ecosystem, **kwargs):
        seen.extend(packages)
        return [[] for _ in packages]

    monkeypatch.setattr(supply_chain, "query_batch", capture_packages)
    cache = {
        "package-lock.json": json.dumps(
            {"packages": {"packages/a": {}, "node_modules/ws": {"version": "8.20.0"}}}
        ),
        "packages/a/package.json": '{"dependencies": {"ws": "^8.20.0"}}',
        "packages/a/package-lock.json": "[" * 20_000 + "]" * 20_000,
    }
    response = supply_chain.node(
        {
            "components": list(cache),
            "file_cache": cache,
            "local_file_cache": cache,
            "manifest": {},
            "component_metadata": [],
        }
    )

    assert seen == [("ws", "8.20.0"), ("ws", None)]
    assert any(
        event["path"] == "packages/a/package-lock.json"
        and event.get("reason_code") is LedgerReason.DEPENDENCY_PARSE_ERROR
        and event.get("error_class") == "RecursionError"
        for event in response["inspection_ledger"]
    )


@pytest.mark.parametrize(
    "lock_names",
    [("package-lock.json", "npm-shrinkwrap.json"), ("PACKAGE-LOCK.JSON", "NPM-SHRINKWRAP.JSON")],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_npm_shrinkwrap_takes_precedence_and_keeps_both_lock_audits(lock_names, reverse):
    package_lock, shrinkwrap = lock_names
    cache = {
        package_lock: '{"packages": {"node_modules/ws": {"version": "8.19.0"}}}',
        shrinkwrap: '{"packages": {"node_modules/ws": {"version": "8.20.0"}}}',
    }
    components = list(reversed(cache)) if reverse else list(cache)
    resolved = supply_chain._collect_npm_locked_versions(cache, components)
    assert resolved["."]["ws"] == "8.20.0"
    assert supply_chain._extract_packages_from_npm_lock(cache[package_lock])[0][:2] == (
        "ws",
        "8.19.0",
    )


def test_npm_lock_collection_respects_file_and_time_bounds():
    cache = {
        "a/package-lock.json": '{"packages": {"node_modules/ws": {"version": "8.19.0"}}}',
        "b/package-lock.json": '{"packages": {"node_modules/ws": {"version": "8.20.0"}}}',
    }
    assert not supply_chain._collect_npm_locked_versions(cache, list(cache), max_files=0)
    assert not supply_chain._collect_npm_locked_versions(cache, list(cache), timeout_seconds=0)
    assert list(supply_chain._collect_npm_locked_versions(cache, list(cache), max_files=1)) == ["a"]


@pytest.mark.parametrize(
    "location", ["../other", "/other", "folder/../../other", "folder\\..\\other"]
)
def test_npm_lock_paths_cannot_resolve_manifests_outside_the_project(location):
    cache = {
        "project/package-lock.json": json.dumps(
            {"packages": {location: {}, "node_modules/ws": {"version": "8.20.0"}}}
        ),
        "other/package.json": "{}",
    }
    assert not supply_chain._collect_npm_locked_versions(cache, list(cache)).get("other")


def test_npm_alias_install_does_not_pin_the_alias_registry_name():
    content = json.dumps(
        {"packages": {"node_modules/lodash": {"name": "node-forge", "version": "1.4.2"}}}
    )
    cache = {"package-lock.json": content}
    resolved = supply_chain._collect_npm_locked_versions(cache, list(cache))
    manifest = '{"dependencies": {"lodash": "npm:node-forge@1.4.2"}}'
    packages = supply_chain._apply_locked_versions(
        supply_chain._extract_packages_from_package_json(manifest),
        resolved["."],
        supply_chain._normalize_npm_package_name,
    )
    assert packages == [("lodash", None, 1)]
    assert supply_chain._extract_packages_from_npm_lock(content) == [("node-forge", "1.4.2", 1)]


@pytest.mark.parametrize("lock_version", [2, 3])
@pytest.mark.parametrize("lock_name", ["package-lock.json", "npm-shrinkwrap.json"])
@pytest.mark.parametrize("alias_name", ["lodash", "@scope/helper"])
@pytest.mark.parametrize("reverse", [False, True])
def test_npm_workspace_alias_shadows_ancestor_registry_pin(
    lock_version, lock_name, alias_name, reverse
):
    content = json.dumps(
        {
            "lockfileVersion": lock_version,
            "packages": {
                "": {"workspaces": ["packages/*"]},
                "packages/a": {"name": "a", "version": "1.0.0"},
                "packages/b": {"name": "b", "version": "1.0.0"},
                f"node_modules/{alias_name}": {"version": "4.17.20"},
                "node_modules/ws": {"version": "8.20.0"},
                f"packages/a/node_modules/{alias_name}": {
                    "name": "node-forge",
                    "version": "1.4.2",
                },
            },
        }
    )
    cache = {lock_name: content, "packages/a/package.json": "{}", "packages/b/package.json": "{}"}
    components = list(reversed(cache)) if reverse else list(cache)
    resolved = supply_chain._collect_npm_locked_versions(cache, components)
    packages = supply_chain._apply_locked_versions(
        [(alias_name, None, 3), ("ws", None, 4), ("explicit", "1.0.0", 5)],
        resolved["packages/a"],
        supply_chain._normalize_npm_package_name,
    )
    assert packages == [(alias_name, None, 3), ("ws", "8.20.0", 4), ("explicit", "1.0.0", 5)]
    assert resolved["."][alias_name] == "4.17.20"
    assert resolved["packages/b"][alias_name] == "4.17.20"
    # Alias targets and ordinary ancestor packages still receive independent audits.
    audited = supply_chain._extract_packages_from_npm_lock(content)
    assert ("node-forge", "1.4.2") in [(name, version) for name, version, _line in audited]
    assert (alias_name, "4.17.20") in [(name, version) for name, version, _line in audited]


@pytest.mark.parametrize("lock_version", [2, 3])
@pytest.mark.parametrize("lock_name", ["package-lock.json", "npm-shrinkwrap.json"])
@pytest.mark.parametrize("linked_name", ["lodash", "@scope/helper"])
@pytest.mark.parametrize(
    "slot",
    [
        {"resolved": "lib/local", "link": True},
        # A version on a link entry still describes local code, not a registry release.
        {"resolved": "lib/local", "link": True, "version": "1.0.0"},
        # Any occupied slot shadows ancestor installs, even when it records no version.
        {"resolved": "lib/local"},
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_npm_workspace_link_shadows_ancestor_registry_pin(
    lock_version, lock_name, linked_name, slot, reverse
):
    content = json.dumps(
        {
            "lockfileVersion": lock_version,
            "packages": {
                "": {"workspaces": ["packages/*"]},
                "packages/a": {"name": "a", "version": "1.0.0"},
                "packages/b": {"name": "b", "version": "1.0.0"},
                "lib/local": {"name": linked_name, "version": "1.0.0"},
                f"node_modules/{linked_name}": {"version": "4.17.20"},
                "node_modules/ws": {"version": "8.20.0"},
                f"packages/a/node_modules/{linked_name}": slot,
            },
        }
    )
    cache = {lock_name: content, "packages/a/package.json": "{}", "packages/b/package.json": "{}"}
    components = list(reversed(cache)) if reverse else list(cache)
    resolved = supply_chain._collect_npm_locked_versions(cache, components)
    manifest = json.dumps(
        {
            "dependencies": {
                linked_name: "file:../../lib/local",
                "ws": "^8.20.0",
                "explicit": "1.0.0",
            }
        }
    )
    packages = supply_chain._apply_locked_versions(
        supply_chain._extract_packages_from_package_json(manifest),
        resolved["packages/a"],
        supply_chain._normalize_npm_package_name,
    )
    assert [(name, version) for name, version, _line in packages] == [
        (linked_name, None),
        ("ws", "8.20.0"),
        ("explicit", "1.0.0"),
    ]
    assert resolved["."][linked_name] == "4.17.20"
    assert resolved["packages/b"][linked_name] == "4.17.20"
    # The ancestor install still receives its own lockfile audit.
    audited = supply_chain._extract_packages_from_npm_lock(content)
    assert (linked_name, "4.17.20") in [(name, version) for name, version, _line in audited]


def test_npm_workspace_link_does_not_send_the_ancestor_pin_to_osv(monkeypatch):
    queries = []

    def capture_packages(packages, ecosystem, **_kwargs):
        queries.append(list(packages))
        return [[] for _ in packages]

    monkeypatch.setattr(supply_chain, "query_batch", capture_packages)
    cache = {
        "package-lock.json": json.dumps(
            {
                "lockfileVersion": 3,
                "packages": {
                    "": {"workspaces": ["packages/*"]},
                    "packages/a": {"name": "a", "version": "1.0.0"},
                    "lib/local": {"name": "lodash", "version": "1.0.0"},
                    "node_modules/lodash": {"version": "4.17.20"},
                    "packages/a/node_modules/lodash": {"resolved": "lib/local", "link": True},
                },
            }
        ),
        "packages/a/package.json": '{"dependencies": {"lodash": "file:../../lib/local"}}',
    }
    supply_chain.node(
        {
            "components": list(cache),
            "file_cache": cache,
            "local_file_cache": cache,
            "manifest": {},
            "component_metadata": [],
        }
    )

    assert [("lodash", None)] in queries
    assert [("lodash", "4.17.20")] not in queries


def test_npm_v1_workspace_uses_recorded_link_and_nearest_locked_install():
    cache = {
        "package-lock.json": json.dumps(
            {
                "lockfileVersion": 1,
                "dependencies": {
                    "workspace-a": {
                        "version": "file:packages/a",
                        "dependencies": {"node-forge": {"version": "1.2.1"}},
                    },
                    "workspace-b": {"version": "file:packages/b"},
                    "node-forge": {"version": "1.4.2"},
                },
            }
        ),
        "packages/a/package.json": "{}",
        "packages/b/package.json": "{}",
        "packages/c/package.json": "{}",
    }
    resolved = supply_chain._collect_npm_locked_versions(cache, list(cache))
    assert resolved["."]["node-forge"] == "1.4.2"
    assert resolved["packages/a"]["node-forge"] == "1.2.1"
    assert resolved["packages/b"]["node-forge"] == "1.4.2"
    assert not resolved.get("packages/c")
    cache["packages/b/package-lock.json"] = "{"
    assert not supply_chain._collect_npm_locked_versions(cache, list(cache)).get("packages/b")


@pytest.mark.parametrize(
    "location", ["../other", "/other", "folder/../../other", "folder\\..\\other"]
)
def test_npm_v1_links_cannot_resolve_manifests_outside_the_project(location):
    cache = {
        "project/package-lock.json": json.dumps(
            {
                "dependencies": {
                    "linked": {"version": f"file:{location}"},
                    "ws": {"version": "8.20.0"},
                }
            }
        ),
        "other/package.json": "{}",
    }
    assert not supply_chain._collect_npm_locked_versions(cache, list(cache)).get("other")


def test_npm_v1_self_link_does_not_replace_the_root_install_tree():
    cache = {
        "package-lock.json": json.dumps(
            {"dependencies": {"self": {"version": "file:./"}, "ws": {"version": "8.20.0"}}}
        ),
        "package.json": "{}",
    }
    assert supply_chain._collect_npm_locked_versions(cache, list(cache))["."]["ws"] == "8.20.0"
