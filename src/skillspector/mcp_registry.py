# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP Registry acquisition, normalized snapshots, and posture checks."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path
from typing import Any, TypedDict

import httpx

REGISTRY_URL = "https://registry.modelcontextprotocol.io/v0/servers"
OFFICIAL_META_KEY = "io.modelcontextprotocol.registry/official"
FILE_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
MUTABLE_VERSION_TAGS = frozenset(
    {
        "latest",
        "next",
        "beta",
        "alpha",
        "stable",
        "canary",
        "edge",
        "main",
        "master",
        "dev",
        "nightly",
        "preview",
    }
)
RANGE_SYNTAX_RE = re.compile(r"[\^~*><=|]|\s")
WILDCARD_SEGMENT_RE = re.compile(r"(?:^|\.)[xX*](?:\.|$)")
NPM_EXACT_VERSION_RE = re.compile(r"^v?\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")


class RegistryFinding(TypedDict):
    id: str
    target: str
    message: str
    severity: str
    evidence: str
    risk_score: int


class RegistryServerReport(TypedDict):
    snapshot: dict[str, Any]
    findings: list[RegistryFinding]


@dataclass(frozen=True)
class RepositoryReference:
    url: str | None = None
    source: str | None = None
    id: str | None = None
    subfolder: str | None = None


@dataclass(frozen=True)
class PackageReference:
    registry_type: str | None = None
    identifier: str | None = None
    version: str | None = None
    file_sha256: str | None = None
    transport_type: str | None = None
    transport_url: str | None = None


@dataclass(frozen=True)
class RemoteReference:
    type: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class RegistryServerSnapshot:
    source: str
    name: str
    title: str | None
    description: str | None
    version: str | None
    website_url: str | None
    repository: RepositoryReference | None
    packages: tuple[PackageReference, ...]
    remotes: tuple[RemoteReference, ...]
    status: str | None
    published_at: str | None
    updated_at: str | None
    is_latest: bool | None
    record_hash: str
    scanned_at: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["packages"] = [asdict(package) for package in self.packages]
        data["remotes"] = [asdict(remote) for remote in self.remotes]
        return data


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def record_hash(record: dict[str, Any]) -> str:
    """Hash a normalized owner record independently of JSON object key order."""
    return hashlib.sha256(_canonical_json(record).encode("utf-8")).hexdigest()


def _optional_string(value: Any) -> str | None:
    # The registry owns field semantics; non-string values are recorded as
    # absent so checks report unavailable evidence instead of failing the scan.
    return value if isinstance(value, str) else None


def _official_meta(record: dict[str, Any]) -> dict[str, Any]:
    meta = record.get("_meta", {})
    official = meta.get(OFFICIAL_META_KEY, {}) if isinstance(meta, dict) else {}
    return official if isinstance(official, dict) else {}


def _is_specific_package_version(registry_type: str | None, version: str | None) -> bool:
    if version is None:
        return False
    if version.casefold() in MUTABLE_VERSION_TAGS:
        return False
    if not any(char.isdigit() for char in version):
        return False
    if registry_type == "npm":
        return NPM_EXACT_VERSION_RE.fullmatch(version) is not None
    # Prerelease/build suffixes like 1.0.0-linux-x64 are exact versions; only
    # range operators and whole x/* segments (1.x, 1.*) mark a mutable range.
    return not (RANGE_SYNTAX_RE.search(version) or WILDCARD_SEGMENT_RE.search(version))


def _is_valid_file_sha256(file_sha256: str | None) -> bool:
    return file_sha256 is not None and FILE_SHA256_RE.fullmatch(file_sha256) is not None


def _record_dict_list(
    record: dict[str, Any], field_name: str, *, source: str, server_name: str
) -> list[dict[str, Any]]:
    if field_name not in record:
        return []
    value = record[field_name]
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(
            f"MCP Registry payload has an invalid {field_name} collection for {server_name} from {source}"
        )
    return value


def _normalize_package_reference(package: dict[str, Any]) -> PackageReference:
    transport = package.get("transport")
    transport = transport if isinstance(transport, dict) else {}
    return PackageReference(
        registry_type=_optional_string(package.get("registryType")),
        identifier=_optional_string(package.get("identifier")),
        version=_optional_string(package.get("version")),
        file_sha256=_optional_string(package.get("fileSha256")),
        transport_type=_optional_string(transport.get("type")),
        transport_url=_optional_string(transport.get("url")),
    )


def normalize_server(
    entry: dict[str, Any], *, source: str, scanned_at: str | None = None
) -> RegistryServerSnapshot:
    if not isinstance(entry, dict) or not isinstance(entry.get("server"), dict):
        raise ValueError(f"MCP Registry payload has an invalid server record from {source}")
    record = entry["server"]
    name = _optional_string(record.get("name"))
    if not name:
        raise ValueError(f"MCP Registry payload has a server without a name from {source}")
    repository_data = record.get("repository")
    repository = None
    if repository_data is not None and not isinstance(repository_data, dict):
        raise ValueError(
            f"MCP Registry payload has an invalid repository object for {name} from {source}"
        )
    if isinstance(repository_data, dict):
        repository = RepositoryReference(
            url=_optional_string(repository_data.get("url")),
            source=_optional_string(repository_data.get("source")),
            id=_optional_string(repository_data.get("id")),
            subfolder=_optional_string(repository_data.get("subfolder")),
        )
    packages = tuple(
        _normalize_package_reference(package)
        for package in _record_dict_list(record, "packages", source=source, server_name=name)
    )
    remotes = tuple(
        RemoteReference(
            type=_optional_string(remote.get("type")),
            url=_optional_string(remote.get("url")),
        )
        for remote in _record_dict_list(record, "remotes", source=source, server_name=name)
    )
    official = _official_meta(entry)
    return RegistryServerSnapshot(
        source=source,
        name=name,
        title=_optional_string(record.get("title")),
        description=_optional_string(record.get("description")),
        version=_optional_string(record.get("version")),
        website_url=_optional_string(record.get("websiteUrl")),
        repository=repository,
        packages=packages,
        remotes=remotes,
        status=_optional_string(official.get("status")),
        published_at=_optional_string(official.get("publishedAt")),
        updated_at=_optional_string(official.get("updatedAt")),
        is_latest=official.get("isLatest") if isinstance(official.get("isLatest"), bool) else None,
        record_hash=record_hash({"server": record, OFFICIAL_META_KEY: official}),
        scanned_at=scanned_at or datetime.now(UTC).isoformat(),
    )


def normalize_payload(payload: dict[str, Any], *, source: str) -> list[RegistryServerSnapshot]:
    if not isinstance(payload, dict) or not isinstance(payload.get("servers"), list):
        raise ValueError(f"MCP Registry payload from {source} must contain a servers list")
    scanned_at = datetime.now(UTC).isoformat()
    return [
        normalize_server(entry, source=source, scanned_at=scanned_at)
        for entry in payload["servers"]
    ]


def _finding(
    rule: str,
    message: str,
    target: str,
    *,
    severity: str,
    evidence: str,
    risk_score: int,
) -> RegistryFinding:
    return {
        "id": rule,
        "target": target,
        "message": message,
        "severity": severity,
        "evidence": evidence,
        "risk_score": risk_score,
    }


def _unavailable(rule: str, message: str, target: str) -> RegistryFinding:
    return _finding(
        rule,
        message,
        target,
        severity="info",
        evidence="unavailable",
        risk_score=0,
    )


def _registry_assertion(
    rule: str, message: str, target: str, *, severity: str, risk_score: int
) -> RegistryFinding:
    return _finding(
        rule,
        message,
        target,
        severity=severity,
        evidence="registry_assertion",
        risk_score=risk_score,
    )


def posture_findings(snapshot: RegistryServerSnapshot) -> list[RegistryFinding]:
    findings: list[RegistryFinding] = []
    for index, package in enumerate(snapshot.packages):
        target = package.identifier or f"package[{index}]"
        if package.version is None:
            findings.append(
                _unavailable("MCP-PACKAGE-VERSION", "Package version is unavailable", target)
            )
        elif not _is_specific_package_version(package.registry_type, package.version):
            findings.append(
                _registry_assertion(
                    "MCP-PACKAGE-VERSION",
                    "Package version is not pinned",
                    target,
                    severity="high",
                    risk_score=30,
                )
            )
        if package.file_sha256 is None:
            findings.append(
                _unavailable("MCP-PACKAGE-SHA256", "Package fileSha256 is unavailable", target)
            )
        elif not _is_valid_file_sha256(package.file_sha256):
            findings.append(
                _registry_assertion(
                    "MCP-PACKAGE-SHA256",
                    "Package fileSha256 is invalid",
                    target,
                    severity="high",
                    risk_score=25,
                )
            )
    if snapshot.repository is None or not snapshot.repository.url:
        findings.append(
            _unavailable("MCP-REPOSITORY", "Repository reference is unavailable", snapshot.name)
        )
    if snapshot.status is None:
        findings.append(
            _unavailable("MCP-OFFICIAL-STATUS", "Official status is unavailable", snapshot.name)
        )
    elif snapshot.status != "active":
        findings.append(
            _registry_assertion(
                "MCP-OFFICIAL-STATUS",
                f"Official status is {snapshot.status}",
                snapshot.name,
                severity="medium",
                risk_score=20,
            )
        )
    for remote in snapshot.remotes:
        if remote.url and remote.url.lower().startswith("http://"):
            findings.append(
                _registry_assertion(
                    "MCP-PLAIN-HTTP",
                    "Remote endpoint uses plain HTTP",
                    remote.url,
                    severity="high",
                    risk_score=25,
                )
            )
    return findings


def _dict_payload(payload: object, *, source: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"MCP Registry source failed: {source}: payload must be a JSON object")
    return payload


def _load_payload(input_path: str) -> dict[str, Any]:
    source = input_path
    try:
        if Path(input_path).is_file():
            return _dict_payload(
                json.loads(Path(input_path).read_text(encoding="utf-8")),
                source=source,
            )
        if input_path.startswith(("http://", "https://")):
            if input_path != REGISTRY_URL:
                raise ValueError(
                    f"MCP Registry source failed: {input_path}: only the official registry URL is supported"
                )
            return _load_paginated_registry(input_path)
        payload = _load_paginated_registry(REGISTRY_URL)
        matches = [
            entry
            for entry in payload.get("servers", [])
            if isinstance(entry, dict)
            and isinstance(entry.get("server"), dict)
            and entry["server"].get("name") == input_path
        ]
        if not matches:
            raise ValueError(f"MCP Registry server identifier was not found: {source}")
        # The registry lists every published version of a server; a name scan
        # assesses the owner's latest record, not the historical tail.
        latest = [entry for entry in matches if _official_meta(entry).get("isLatest") is True]
        return {"servers": latest or matches}
    except (OSError, json.JSONDecodeError, httpx.HTTPError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("MCP Registry source"):
            raise
        raise ValueError(f"MCP Registry source failed: {source}: {exc}") from exc


def _load_paginated_registry(url: str) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    seen_cursors: set[str] = set()
    cursor: str | None = None

    while True:
        params = {"cursor": cursor} if cursor is not None else None
        response = httpx.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = _dict_payload(response.json(), source=url)
        if not isinstance(payload.get("servers"), list):
            raise ValueError(f"MCP Registry payload from {url} must contain a servers list")
        pages.append(payload)

        metadata = payload.get("metadata")
        next_cursor = metadata.get("nextCursor") if isinstance(metadata, dict) else None
        if not isinstance(next_cursor, str) or not next_cursor:
            break
        if next_cursor in seen_cursors:
            raise ValueError(f"MCP Registry source failed: {url}: repeated pagination cursor")
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    return {
        "servers": list(chain.from_iterable(page["servers"] for page in pages)),
        "metadata": pages[-1].get("metadata", {}),
    }


def scan_registry(input_path: str = REGISTRY_URL) -> dict[str, Any]:
    """Acquire, normalize, and assess one MCP Registry payload."""
    snapshots = normalize_payload(_load_payload(input_path), source=input_path)
    per_server: list[RegistryServerReport] = [
        {"snapshot": snapshot.to_dict(), "findings": posture_findings(snapshot)}
        for snapshot in snapshots
    ]
    findings = [finding for server in per_server for finding in server["findings"]]
    risk_score = min(sum(finding["risk_score"] for finding in findings), 100)
    max_risk_score = max((finding["risk_score"] for finding in findings), default=0)
    return {
        "mcp_registry": True,
        "source": input_path,
        "server_count": len(snapshots),
        "risk_score": risk_score,
        "max_risk_score": max_risk_score,
        "findings": findings,
        "snapshots": [snapshot.to_dict() for snapshot in snapshots],
        "servers": per_server,
    }
