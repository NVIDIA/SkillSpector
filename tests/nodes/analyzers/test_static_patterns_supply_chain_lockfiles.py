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

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_supply_chain as supply_chain
from skillspector.nodes.analyzers.osv_client import OsvQueryLimitation, QueryBatchResults


def _capture_osv_packages(monkeypatch):
    seen = {}

    def fake_query_batch(packages, ecosystem):
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
