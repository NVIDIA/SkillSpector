# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MCP Registry owner and posture checks."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from skillspector.mcp_registry import (
    OFFICIAL_META_KEY,
    REGISTRY_URL,
    normalize_payload,
    posture_findings,
    record_hash,
    scan_registry,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "mcp_registry"


def payload() -> dict:
    return json.loads((FIXTURES / "mcp_registry.json").read_text(encoding="utf-8"))


def one_server(server: dict[str, Any], official: dict[str, Any] | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"server": server}
    if official is not None:
        entry["_meta"] = {OFFICIAL_META_KEY: official}
    return {"servers": [entry]}


def pinned_package(**overrides: Any) -> dict[str, Any]:
    package: dict[str, Any] = {
        "registryType": "npm",
        "identifier": "example",
        "version": "1.0.0",
        "fileSha256": "a" * 64,
        "transport": {"type": "stdio"},
    }
    package.update(overrides)
    return package


def pinned_server(**overrides: Any) -> dict[str, Any]:
    server: dict[str, Any] = {
        "name": "safe/example",
        "repository": {"url": "https://github.com/example/project", "source": "github"},
        "packages": [pinned_package()],
        "remotes": [{"type": "streamable-http", "url": "https://example.invalid/mcp"}],
    }
    server.update(overrides)
    return server


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def test_snapshot_normalizes_owner_fields_and_serializes() -> None:
    first = normalize_payload(payload(), source="fixture")[0]
    data = first.to_dict()
    assert data.pop("record_hash") == first.record_hash
    assert data.pop("scanned_at") == first.scanned_at
    assert data == {
        "source": "fixture",
        "name": "ac.tandem/docs-mcp",
        "title": "Tandem Docs",
        "description": "Remote MCP server for Tandem docs.",
        "version": "0.3.2",
        "website_url": "https://tandem.ac/docs-mcp",
        "repository": {
            "url": "https://github.com/frumu-ai/tandem",
            "source": "github",
            "id": None,
            "subfolder": None,
        },
        "packages": [],
        "remotes": [{"type": "streamable-http", "url": "https://tandem.ac/mcp"}],
        "status": "active",
        "published_at": "2026-04-22T21:06:34.500049Z",
        "updated_at": "2026-04-22T21:06:34.500049Z",
        "is_latest": True,
    }


def test_snapshot_preserves_package_transport() -> None:
    server = pinned_server(
        packages=[
            pinned_package(
                transport={"type": "streamable-http", "url": "https://example.invalid/mcp"}
            )
        ]
    )
    package = normalize_payload(one_server(server), source="fixture")[0].packages[0]
    assert package.transport_type == "streamable-http"
    assert package.transport_url == "https://example.invalid/mcp"


def test_snapshot_preserves_template_transport_url() -> None:
    server = pinned_server(
        packages=[pinned_package(transport={"type": "streamable-http", "url": "{baseUrl}/mcp"})]
    )
    snapshot = normalize_payload(
        one_server(server, official={"status": "active"}), source="fixture"
    )[0]
    assert snapshot.packages[0].transport_url == "{baseUrl}/mcp"
    assert posture_findings(snapshot) == []


def test_snapshot_treats_wrong_typed_optional_fields_as_absent() -> None:
    server = pinned_server(packages=[pinned_package(version=7, fileSha256=7)])
    snapshot = normalize_payload(one_server(server, official={"status": 0}), source="fixture")[0]
    package = snapshot.packages[0]
    assert package.version is None
    assert package.file_sha256 is None
    assert snapshot.status is None
    assert all(finding["evidence"] == "unavailable" for finding in posture_findings(snapshot))


def test_contract_isolation_uses_normalized_snapshots() -> None:
    report = scan_registry(str(FIXTURES / "mcp_registry.json"))
    assert report["mcp_registry"] is True
    assert report["snapshots"][0]["repository"]["url"].startswith("https://")
    assert all("server" not in server["snapshot"] for server in report["servers"])


def test_registry_url_scan_follows_next_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    page_one = {
        "servers": [{"server": {"name": "page/one"}}],
        "metadata": {"nextCursor": "cursor-2"},
    }
    page_two = {"servers": [{"server": {"name": "page/two"}}], "metadata": {}}
    calls: list[tuple[str, dict[str, str] | None]] = []

    def fake_get(url: str, *, params: dict[str, str] | None = None, timeout: int) -> _Response:
        calls.append((url, params))
        return _Response(page_one if params is None else page_two)

    monkeypatch.setattr("skillspector.mcp_registry.httpx.get", fake_get)

    report = scan_registry(REGISTRY_URL)

    assert report["server_count"] == 2
    assert [server["snapshot"]["name"] for server in report["servers"]] == ["page/one", "page/two"]
    assert calls == [(REGISTRY_URL, None), (REGISTRY_URL, {"cursor": "cursor-2"})]


def test_untrusted_registry_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="only the official registry URL is supported"):
        scan_registry("https://untrusted.invalid/servers")


def test_server_identifier_scan_follows_next_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    page_one = {
        "servers": [{"server": {"name": "page/one"}}],
        "metadata": {"nextCursor": "cursor-2"},
    }
    page_two = {"servers": [{"server": {"name": "page/two"}}], "metadata": {}}

    def fake_get(url: str, *, params: dict[str, str] | None = None, timeout: int) -> _Response:
        return _Response(page_one if params is None else page_two)

    monkeypatch.setattr("skillspector.mcp_registry.httpx.get", fake_get)

    report = scan_registry("page/two")

    assert report["server_count"] == 1
    assert report["servers"][0]["snapshot"]["name"] == "page/two"


def test_server_identifier_scan_selects_latest_version(monkeypatch: pytest.MonkeyPatch) -> None:
    page = {
        "servers": [
            {
                "server": {"name": "dup/example", "version": "1.0.0"},
                "_meta": {OFFICIAL_META_KEY: {"status": "deprecated", "isLatest": False}},
            },
            {
                "server": {"name": "dup/example", "version": "1.1.0"},
                "_meta": {OFFICIAL_META_KEY: {"status": "active", "isLatest": True}},
            },
        ],
        "metadata": {},
    }

    def fake_get(url: str, *, params: dict[str, str] | None = None, timeout: int) -> _Response:
        return _Response(page)

    monkeypatch.setattr("skillspector.mcp_registry.httpx.get", fake_get)

    report = scan_registry("dup/example")

    assert report["server_count"] == 1
    assert report["servers"][0]["snapshot"]["version"] == "1.1.0"
    assert all(finding["id"] != "MCP-OFFICIAL-STATUS" for finding in report["findings"])


def test_record_hash_is_stable_when_record_keys_are_reordered() -> None:
    left = {
        "server": {
            "name": "example",
            "version": "1",
            "remotes": [{"url": "https://example.invalid"}],
        },
        OFFICIAL_META_KEY: {"status": "active"},
    }
    right = {
        OFFICIAL_META_KEY: {"status": "active"},
        "server": {
            "remotes": [{"url": "https://example.invalid"}],
            "version": "1",
            "name": "example",
        },
    }
    assert record_hash(left) == record_hash(right)


def test_record_hash_includes_official_metadata() -> None:
    server = {
        "name": "example",
        "version": "1",
        "remotes": [{"url": "https://example.invalid"}],
    }
    active = {"server": server, OFFICIAL_META_KEY: {"status": "active"}}
    deprecated = {"server": server, OFFICIAL_META_KEY: {"status": "deprecated"}}
    assert record_hash(active) != record_hash(deprecated)


def test_posture_findings_cover_registry_boundaries() -> None:
    findings = [
        finding
        for snapshot in normalize_payload(payload(), source="fixture")
        for finding in posture_findings(snapshot)
    ]
    ids = {finding["id"] for finding in findings}
    assert {"MCP-REPOSITORY", "MCP-OFFICIAL-STATUS", "MCP-PLAIN-HTTP"} <= ids


def test_posture_flags_unrecognized_status_instead_of_failing() -> None:
    snapshot = normalize_payload(
        one_server(pinned_server(), official={"status": "suspended"}), source="fixture"
    )[0]
    findings = posture_findings(snapshot)
    assert [finding["id"] for finding in findings] == ["MCP-OFFICIAL-STATUS"]
    assert findings[0]["message"] == "Official status is suspended"
    assert findings[0]["risk_score"] > 0


def test_posture_flags_unpinned_package_and_missing_hash() -> None:
    server = pinned_server(packages=[pinned_package(version="latest")])
    del server["packages"][0]["fileSha256"]
    snapshot = normalize_payload(
        one_server(server, official={"status": "active"}), source="fixture"
    )[0]
    findings = posture_findings(snapshot)
    by_id = {finding["id"]: finding for finding in findings}
    assert by_id["MCP-PACKAGE-VERSION"]["evidence"] == "registry_assertion"
    assert by_id["MCP-PACKAGE-SHA256"]["evidence"] == "unavailable"


def test_posture_flags_missing_status_as_unavailable() -> None:
    snapshot = normalize_payload(one_server(pinned_server()), source="fixture")[0]
    findings = posture_findings(snapshot)
    assert [finding["id"] for finding in findings] == ["MCP-OFFICIAL-STATUS"]
    assert findings[0]["evidence"] == "unavailable"
    assert findings[0]["risk_score"] == 0


def test_negative_space_pinned_active_server_has_no_findings() -> None:
    snapshot = normalize_payload(
        one_server(pinned_server(), official={"status": "active"}), source="fixture"
    )[0]
    assert posture_findings(snapshot) == []


def test_negative_space_absent_optional_facts_are_unavailable() -> None:
    snapshot = normalize_payload(
        {"servers": [{"server": {"name": "unknown/example"}}]}, source="fixture"
    )[0]
    findings = posture_findings(snapshot)
    assert findings
    assert all(finding["evidence"] == "unavailable" for finding in findings)
    assert all(finding["risk_score"] == 0 for finding in findings)


@pytest.mark.parametrize(
    "version, file_sha256",
    [("latest", "not-a-sha256"), ("", "")],
)
def test_negative_space_asserted_bad_version_and_hash_are_flagged(
    version: str, file_sha256: str
) -> None:
    server = pinned_server(packages=[pinned_package(version=version, fileSha256=file_sha256)])
    snapshot = normalize_payload(
        one_server(server, official={"status": "active"}), source="fixture"
    )[0]
    findings = posture_findings(snapshot)
    ids = {finding["id"] for finding in findings}
    assert {"MCP-PACKAGE-VERSION", "MCP-PACKAGE-SHA256"} <= ids
    assert {finding["evidence"] for finding in findings} == {"registry_assertion"}


@pytest.mark.parametrize(
    "version",
    ["latest", "LATEST", "next", "beta", "1", "1.2", "1.x", "1.0.0||2.0.0"],
)
def test_negative_space_mutable_version_tags_are_flagged(version: str) -> None:
    server = pinned_server(packages=[pinned_package(version=version)])
    snapshot = normalize_payload(
        one_server(server, official={"status": "active"}), source="fixture"
    )[0]
    ids = {finding["id"] for finding in posture_findings(snapshot)}
    assert "MCP-PACKAGE-VERSION" in ids


@pytest.mark.parametrize(
    "registry_type, version",
    [("npm", "1.0.0-experimental"), ("npm", "1.0.0+linux-x64"), ("oci", "1.0.0-linux-x64")],
)
def test_negative_space_exact_versions_with_suffixes_are_pinned(
    registry_type: str, version: str
) -> None:
    server = pinned_server(packages=[pinned_package(registryType=registry_type, version=version)])
    snapshot = normalize_payload(
        one_server(server, official={"status": "active"}), source="fixture"
    )[0]
    assert posture_findings(snapshot) == []


def test_negative_space_empty_repository_url_is_unavailable() -> None:
    server = pinned_server(repository={"url": "", "source": "github"})
    snapshot = normalize_payload(
        one_server(server, official={"status": "active"}), source="fixture"
    )[0]
    findings = posture_findings(snapshot)
    assert [finding["id"] for finding in findings] == ["MCP-REPOSITORY"]
    assert findings[0]["evidence"] == "unavailable"


def test_error_on_malformed_payload() -> None:
    with pytest.raises(ValueError, match="MCP Registry"):
        scan_registry(str(FIXTURES / "malformed.json"))


def test_error_on_missing_servers_list() -> None:
    with pytest.raises(ValueError, match="servers list"):
        normalize_payload({}, source="fixture")


@pytest.mark.parametrize(
    "field_name, value", [("packages", {}), ("packages", None), ("remotes", {})]
)
def test_error_on_invalid_collection_shapes(field_name: str, value: object) -> None:
    with pytest.raises(ValueError, match="invalid .* collection"):
        normalize_payload(
            {"servers": [{"server": {"name": "broken/example", field_name: value}}]},
            source="fixture",
        )


def test_error_on_invalid_repository_shape() -> None:
    with pytest.raises(ValueError, match="invalid repository object"):
        normalize_payload(
            {"servers": [{"server": {"name": "broken/example", "repository": []}}]},
            source="fixture",
        )


def test_error_on_repeated_next_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    page = {"servers": [{"server": {"name": "page/one"}}], "metadata": {"nextCursor": "cursor-1"}}

    def fake_get(url: str, *, params: dict[str, str] | None = None, timeout: int) -> _Response:
        return _Response(page)

    monkeypatch.setattr("skillspector.mcp_registry.httpx.get", fake_get)

    with pytest.raises(ValueError, match="repeated pagination cursor"):
        scan_registry(REGISTRY_URL)


def test_error_on_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, *, params: dict[str, str] | None = None, timeout: int) -> _Response:
        raise httpx.HTTPError("network down")

    monkeypatch.setattr("skillspector.mcp_registry.httpx.get", fake_get)

    with pytest.raises(ValueError, match="MCP Registry source failed"):
        scan_registry(REGISTRY_URL)


def test_scan_registry_scans_partial_paginated_capture(tmp_path: Path) -> None:
    capture = tmp_path / "registry.json"
    capture.write_text(
        json.dumps(
            {
                "servers": [{"server": {"name": "page/one"}}],
                "metadata": {"nextCursor": "page-2"},
            }
        ),
        encoding="utf-8",
    )
    report = scan_registry(str(capture))
    assert report["server_count"] == 1
    assert report["servers"][0]["snapshot"]["name"] == "page/one"


def test_scan_registry_aggregates_risk_score() -> None:
    report = scan_registry(str(FIXTURES / "mcp_registry.json"))
    assert report["risk_score"] == 45
    assert report["max_risk_score"] == 25
