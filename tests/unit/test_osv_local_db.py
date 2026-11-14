# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from skillspector.inspection_ledger import LedgerReason
from skillspector.nodes.analyzers import osv_client, osv_local_db, static_patterns_supply_chain
from skillspector.nodes.analyzers.osv_client import OsvQueryBudget, query_batch


@pytest.fixture(autouse=True)
def offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SKILLSPECTOR_OSV_DB_PATH", str(tmp_path))

    def deny_http(*args, **kwargs):
        pytest.fail("offline lookup attempted to create an HTTP client")

    monkeypatch.setattr(osv_client.httpx, "Client", deny_http)
    osv_client.clear_cache()
    yield
    osv_client.clear_cache()


def advisory(
    identifier: str,
    ecosystem: str,
    name: str,
    **affected: object,
) -> dict:
    return {
        "id": identifier,
        "summary": "Example advisory",
        "aliases": ["CVE-2026-12345"],
        "database_specific": {"severity": "MODERATE"},
        "affected": [{"package": {"name": name, "ecosystem": ecosystem}, **affected}],
    }


def snapshot(root: Path, ecosystem: str, *records: dict | str) -> Path:
    path = root / ecosystem / "all.zip"
    path.parent.mkdir(exist_ok=True)
    replacement = path.with_suffix(".new")
    with ZipFile(replacement, "w") as archive:
        for i, record in enumerate(records):
            archive.writestr(
                f"{i}.json", json.dumps(record) if isinstance(record, dict) else record
            )
    replacement.chmod(0o444)
    replacement.replace(path)
    return path


def test_pypi_ordering_metadata_and_remote_cache_isolation(tmp_path: Path) -> None:
    snapshot(
        tmp_path,
        "PyPI",
        advisory(
            "PYSEC-example",
            "PyPI",
            "Demo_pkg",
            ranges=[
                {
                    "type": "ECOSYSTEM",
                    "events": [{"fixed": "2!1.0.post1"}, {"introduced": "2!1.0rc1"}],
                }
            ],
        ),
    )
    osv_client._put_cache(
        osv_client._cache_key("demo-pkg", "2!1.0rc2", "PyPI"),
        [osv_client.VulnResult("REMOTE-ONLY", "remote cached", "HIGH", ())],
    )
    result = query_batch(
        [("Demo.Pkg", version) for version in ("1!5.0", "2!1.0rc2", "2!1.0", "2!1.0.post1")],
        "PyPI",
    )
    assert result.local_checked == frozenset(range(4))
    assert not result.limitations
    assert result == [
        [],
        [osv_client.VulnResult("PYSEC-example", "Example advisory", "MEDIUM", ("CVE-2026-12345",))],
        [osv_client.VulnResult("PYSEC-example", "Example advisory", "MEDIUM", ("CVE-2026-12345",))],
        [],
    ]
    assert osv_client.is_available()


def test_npm_range_boundaries_and_package_names(tmp_path: Path) -> None:
    snapshot(
        tmp_path,
        "npm",
        advisory(
            "GHSA-range",
            "npm",
            "@scope/demo_pkg",
            ranges=[
                {
                    "type": "SEMVER",
                    "events": [
                        {"introduced": "1.0.0-beta.2"},
                        {"fixed": "1.0.0-beta.11"},
                        {"introduced": "2.0.0"},
                        {"fixed": "2.3.0"},
                        {"limit": "2.2.0"},
                    ],
                },
                {
                    "type": "ECOSYSTEM",
                    "events": [{"introduced": "3.0.0"}, {"last_affected": "3.2.0"}],
                },
            ],
            versions=["5.0.0"],
        ),
    )
    versions = (
        "1.0.0-beta.1",
        "1.0.0-beta.9+build",
        "1.0.0-beta.11",
        "1.0.0",
        "2.0.0",
        "2.2.0",
        "3.2.0",
        "3.2.1",
        "5.0.0",
    )
    result = query_batch(
        [("@scope/demo_pkg", v) for v in versions] + [("@scope/demo-pkg", "2.0.0")], "npm"
    )
    assert [bool(matches) for matches in result] == [
        False,
        True,
        False,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert result.local_checked == frozenset(range(10))
    assert not result.limitations


def test_withdrawn_unsupported_and_unresolved(tmp_path: Path) -> None:
    withdrawn = advisory("WITHDRAWN", "PyPI", "demo", versions=["1.0"])
    withdrawn["withdrawn"] = "2025-01-01T00:00:00Z"
    snapshot(
        tmp_path,
        "PyPI",
        withdrawn,
        advisory("GIT", "PyPI", "demo", ranges=[{"type": "GIT", "events": []}]),
        advisory("EXPLICIT", "PyPI", "demo", versions=["1.0"]),
    )
    result = query_batch([("demo", "1.0.0"), ("demo", None), ("unknown", "latest")], "PyPI")
    assert [v.vuln_id for v in result[0]] == ["EXPLICIT"]
    assert {v.vuln_id for v in result[1]} == {"GIT", "EXPLICIT"}
    assert result.local_checked == frozenset()
    assert {item.error_class for item in result.limitations} == {
        "UnsupportedOsvRecord",
        "UnresolvedOsvPackageVersion",
        "UnsupportedOsvPackageVersion",
    }


def test_missing_corrupt_and_malformed_are_not_clean(tmp_path: Path) -> None:
    missing = query_batch([("demo", "1")], "PyPI")
    assert missing == [[]]
    assert missing.local_checked == frozenset()
    assert missing.limitations[0].error_class == "FileNotFoundError"
    assert not osv_client.is_available()

    path = snapshot(tmp_path, "PyPI")
    path.chmod(0o644)
    path.write_bytes(b"not a zip archive")
    corrupt = query_batch([("demo", "1")], "PyPI")
    assert corrupt.local_checked == frozenset()
    assert corrupt.limitations[0].error_class == "BadZipFile"

    snapshot(tmp_path, "PyPI", "not JSON", advisory("MATCH", "PyPI", "demo", versions=["1"]))
    partial = query_batch([("demo", "1")], "PyPI")
    assert [v.vuln_id for v in partial[0]] == ["MATCH"]
    assert partial.local_checked == frozenset()
    assert partial.limitations[0].error_class == "InvalidOsvRecord"


def test_index_reuse_and_archive_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot(tmp_path, "PyPI", advisory("FIRST", "PyPI", "demo", versions=["1"]))
    original = osv_client.open_snapshot
    opened = 0

    def count_opens(*args, **kwargs):
        nonlocal opened
        opened += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(osv_client, "open_snapshot", count_opens)
    budget = OsvQueryBudget.create()
    try:
        assert query_batch([("demo", "1")], "PyPI", budget=budget)[0][0].vuln_id == "FIRST"
        snapshot(tmp_path, "PyPI", advisory("SECOND", "PyPI", "demo", versions=["1"]))
        assert query_batch([("demo", "1")], "PyPI", budget=budget)[0][0].vuln_id == "FIRST"
        assert opened == 1
    finally:
        budget.close()
    assert query_batch([("demo", "1")], "PyPI")[0][0].vuln_id == "SECOND"
    assert opened == 2


def test_archive_and_result_budgets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot(tmp_path, "PyPI", advisory("FIRST", "PyPI", "demo", versions=["1"]))
    with monkeypatch.context() as patch:
        patch.setattr(osv_local_db, "_MAX_ARCHIVE_MEMBERS", 0)
        result = query_batch([("demo", "1")], "PyPI")
        assert result.limitations[0].reason is LedgerReason.ARCHIVE_MEMBER_LIMIT
    with monkeypatch.context() as patch:
        patch.setattr(osv_local_db, "_MAX_RECORD_BYTES", 16)
        result = query_batch([("demo", "1")], "PyPI")
        assert result.local_checked == frozenset()
        assert result.limitations[0].reason is LedgerReason.ARCHIVE_MEMBER_SIZE_LIMIT
    expired = query_batch([("demo", "1")], "PyPI", timeout_seconds=0)
    assert expired.limitations[0].reason is LedgerReason.RUNTIME_LIMIT
    budget = OsvQueryBudget.create()
    budget.max_results = 1
    try:
        assert query_batch([("demo", "1")], "PyPI", budget=budget).local_checked == {0}
        limited = query_batch([("demo", "1")], "PyPI", budget=budget)
        assert limited == [[]]
        assert limited.limitations[0].reason is LedgerReason.OUTPUT_LIMIT
    finally:
        budget.close()


def test_complete_empty_lookup_skips_fallback_but_unavailable_lookup_uses_it(
    tmp_path: Path,
) -> None:
    analyze = static_patterns_supply_chain._analyze_dependencies_detailed
    snapshot(tmp_path, "PyPI")
    findings, limits, _ = analyze("requests==2.30.0\n", "requirements.txt")
    assert not any(finding.rule_id == "SC4" for finding in findings)
    assert not limits

    (tmp_path / "PyPI" / "all.zip").unlink()
    findings, limits, _ = analyze("requests==2.30.0\n", "requirements.txt")
    sc4 = [finding for finding in findings if finding.rule_id == "SC4"]
    assert sc4 and "static fallback" in sc4[0].explanation
    assert not any("OSV.dev unreachable" in finding.message for finding in findings)
    assert limits[0].error_class == "FileNotFoundError"


def test_snapshot_modified_in_place_cannot_reuse_stale_index(tmp_path: Path) -> None:
    path = snapshot(tmp_path, "PyPI", advisory("OLD", "PyPI", "demo", versions=["1"]))
    budget = OsvQueryBudget.create()
    try:
        assert query_batch([("demo", "1")], "PyPI", budget=budget)[0][0].vuln_id == "OLD"
        path.chmod(0o644)
        with path.open("ab") as changed:
            changed.write(b"changed after indexing")
        result = query_batch([("demo", "1")], "PyPI", budget=budget)
        assert result == [[]]
        assert result.local_checked == frozenset()
        assert result.limitations[0].error_class == "OsvSnapshotChanged"
    finally:
        budget.close()


def test_empty_configured_path_does_not_enable_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKILLSPECTOR_OSV_DB_PATH", "")
    result = query_batch([("demo", "1")], "PyPI")
    assert result == [[]]
    assert result.local_checked == frozenset()
    assert result.limitations[0].error_class == "InvalidOsvDatabasePath"
