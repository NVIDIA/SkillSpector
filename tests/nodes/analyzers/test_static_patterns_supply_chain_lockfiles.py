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

from skillspector.nodes.analyzers import static_patterns_supply_chain as supply_chain


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
