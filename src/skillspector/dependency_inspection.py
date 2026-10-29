# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in acquisition and static inspection of exact locked npm artifacts.

Only public registry HTTPS tarballs with SHA-512 SRI are supported. This pass
never invokes npm, resolves new versions, reads .npmrc, or uses host credentials.
All archive parsing and content analysis happen in the restricted worker.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from time import monotonic
from urllib.parse import urlsplit

import httpx

from skillspector.inspection_ledger import (
    LedgerOutcome,
    LedgerReason,
    LedgerRecordType,
    analyzer_status_event,
    analyzer_status_for_events,
    ledger_event,
)
from skillspector.logging_config import get_logger
from skillspector.models import Finding
from skillspector.state import SkillspectorState, ensure_workflow_resource_budget

ANALYZER_ID = "dependency_contents"
REGISTRY = "registry.npmjs.org"
LOCKFILES = frozenset({"package-lock.json", "npm-shrinkwrap.json"})
PACKAGE_NAME = re.compile(r"(?:@[a-z0-9._-]+/)?[a-z0-9._-]+", re.IGNORECASE)
logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DependencyInspectionLimits:
    """Per-invocation ceilings, further restricted by the shared workflow budget."""

    packages: int = 128
    lockfile_bytes: int = 1024 * 1024
    compressed_bytes: int = 32 * 1024 * 1024
    artifact_bytes: int = 8 * 1024 * 1024
    expanded_bytes: int = 64 * 1024 * 1024
    artifact_expanded_bytes: int = 16 * 1024 * 1024
    file_bytes: int = 1024 * 1024
    members: int = 4096
    artifact_members: int = 1024
    findings: int = 1000
    compression_ratio: int = 200
    seconds: float = 120.0
    worker_seconds: float = 30.0


@dataclass(slots=True)
class LockedArtifact:
    name: str
    version: str
    url: str
    digests: tuple[bytes, ...]
    references: list[dict[str, str]] = field(default_factory=list)


def _configured_limits() -> DependencyInspectionLimits:
    limits = DependencyInspectionLimits()
    for suffix, field_name, conversion in (
        ("PACKAGES", "packages", int),
        ("SECONDS", "seconds", float),
        ("DOWNLOAD_BYTES", "compressed_bytes", int),
        ("EXPANDED_BYTES", "expanded_bytes", int),
    ):
        name = "SKILLSPECTOR_MAX_DEPENDENCY_INSPECTION_" + suffix
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            parsed = conversion(value)
            if parsed <= 0 or not math.isfinite(parsed):
                raise ValueError("invalid dependency inspection limit")
            limits = replace(limits, **{field_name: parsed})
        except (ValueError, OverflowError):
            logger.warning("%s must be positive and finite; using the default", name)
    return limits


class DependencyInspectionError(ValueError):
    """An inspection gap described by an allowlisted reason."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _validate_artifact_url(value: object) -> str:
    if not isinstance(value, str) or len(value) > 2048 or any(c.isspace() for c in value):
        raise DependencyInspectionError("dependency_unsupported_source")
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise DependencyInspectionError("dependency_unsupported_source") from exc
    if (
        parsed.scheme != "https"
        or parsed.netloc != REGISTRY
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith(".tgz")
        or "\\" in value
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise DependencyInspectionError("dependency_unsupported_source")
    return value


def _integrity_digests(value: object) -> tuple[bytes, ...]:
    if not isinstance(value, str) or len(value) > 2048:
        raise DependencyInspectionError("dependency_integrity_missing")
    digests: list[bytes] = []
    for token in value.split():
        if not token.startswith("sha512-"):
            continue
        try:
            digest = base64.b64decode(token[7:], validate=True)
        except ValueError as exc:
            raise DependencyInspectionError("dependency_integrity_missing") from exc
        if len(digest) != 64:
            raise DependencyInspectionError("dependency_integrity_missing")
        digests.append(digest)
    if not digests:
        raise DependencyInspectionError("dependency_integrity_missing")
    return tuple(sorted(set(digests)))


def _event(
    path: str,
    reason: str | None = None,
    *,
    findings: list[Finding] | None = None,
    system: bool = False,
) -> dict:
    return ledger_event(
        analyzer_id=ANALYZER_ID,
        phase="dependency_contents",
        path=path,
        record_type=LedgerRecordType.SYSTEM if system else LedgerRecordType.WORK_ITEM,
        outcome=LedgerOutcome.PARTIAL if reason else LedgerOutcome.COMPLETED,
        reason=LedgerReason(reason) if reason else None,
        emitted_finding_ids=[finding.finding_id for finding in findings or []],
    )


def _collect_artifacts(
    state: SkillspectorState, limits: DependencyInspectionLimits, *, deadline: float | None = None
) -> tuple[list[LockedArtifact], list[dict], int]:
    cache = state.get("raw_file_cache") or {}
    locks = sorted(path for path in cache if PurePosixPath(path).name in LOCKFILES)
    events: list[dict] = []
    artifacts: dict[tuple, LockedArtifact] = {}
    selected: list[str] = []
    locations = set(locks)
    for path in locks:
        sibling = str(PurePosixPath(path).with_name("npm-shrinkwrap.json"))
        if PurePosixPath(path).name == "package-lock.json" and sibling in locations:
            continue
        selected.append(path)
    if not selected:
        return [], [_event("package-lock.json", "dependency_lockfile_missing", system=True)], 0
    selected_dirs = {str(PurePosixPath(path).parent) for path in selected}
    for path in cache:
        if (
            PurePosixPath(path).name == "package.json"
            and str(PurePosixPath(path).parent) not in selected_dirs
        ):
            events.append(_event(path, "dependency_lockfile_missing", system=True))
    entries = 0
    for lock_index, path in enumerate(selected):
        if lock_index >= limits.packages:
            events.append(_event(path, "artifact_count_limit", system=True))
            break
        if deadline is not None and monotonic() >= deadline:
            events.append(_event(path, "runtime_limit", system=True))
            break
        raw = cache[path]
        try:
            if len(raw) > limits.lockfile_bytes:
                raise DependencyInspectionError("size_limit")
            lock = json.loads(raw, object_pairs_hook=_unique_object)
            if (
                not isinstance(lock, dict)
                or type(lock.get("lockfileVersion")) is not int
                or lock["lockfileVersion"] not in (2, 3)
            ):
                raise DependencyInspectionError("dependency_unsupported_lockfile")
            packages = lock.get("packages")
            if not isinstance(packages, dict) or not isinstance(packages.get(""), dict):
                raise DependencyInspectionError("dependency_parse_error")
            manifest_path = str(PurePosixPath(path).with_name("package.json"))
            if manifest_path in cache:
                if len(cache[manifest_path]) > limits.lockfile_bytes:
                    raise DependencyInspectionError("size_limit")
                manifest = json.loads(cache[manifest_path], object_pairs_hook=_unique_object)
                if not isinstance(manifest, dict):
                    raise DependencyInspectionError("dependency_parse_error")
                for key in (
                    "dependencies",
                    "devDependencies",
                    "optionalDependencies",
                    "peerDependencies",
                ):
                    if manifest.get(key, {}) != packages[""].get(key, {}):
                        raise DependencyInspectionError("dependency_lockfile_mismatch")
            for index, (location, package) in enumerate(packages.items()):
                if location == "":
                    continue
                entries += 1
                if entries > limits.packages:
                    events.append(_event(path, "artifact_count_limit", system=True))
                    return list(artifacts.values()), events, entries
                reference_path = f"{path}!/packages/{index}"
                try:
                    if (
                        not isinstance(package, dict)
                        or not isinstance(location, str)
                        or len(location) > 2048
                        or "\\" in location
                        or any(part in {"", ".", ".."} for part in location.split("/"))
                    ):
                        raise DependencyInspectionError("dependency_parse_error")
                    if (
                        package.get("link")
                        or package.get("inBundle")
                        or not location.startswith("node_modules/")
                    ):
                        raise DependencyInspectionError("dependency_unsupported_source")
                    installed_name = location.rsplit("node_modules/", 1)[-1]
                    name = package.get("name", installed_name)
                    version = package.get("version")
                    if (
                        not isinstance(name, str)
                        or len(name) > 214
                        or not PACKAGE_NAME.fullmatch(name)
                        or not isinstance(version, str)
                        or len(version) > 128
                        or not re.fullmatch(
                            r"[0-9]+\.[0-9]+\.[0-9]+(?:-[a-zA-Z0-9.-]+)?(?:\+[a-zA-Z0-9.-]+)?",
                            version,
                        )
                    ):
                        raise DependencyInspectionError("dependency_parse_error")
                    url = _validate_artifact_url(package.get("resolved"))
                    digests = _integrity_digests(package.get("integrity"))
                    key = (name, version, digests)
                    if key not in artifacts:
                        artifacts[key] = LockedArtifact(name, version, url, digests)
                    artifacts[key].references.append({"lockfile": path, "location": location})
                except DependencyInspectionError as exc:
                    events.append(_event(reference_path, exc.reason, system=True))
        except DependencyInspectionError as exc:
            events.append(_event(path, exc.reason, system=True))
        except (ValueError, TypeError, UnicodeError, RecursionError):
            events.append(_event(path, "dependency_parse_error", system=True))
    return list(artifacts.values()), events, entries


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _resolve_registry_addresses(deadline: float) -> list[str]:
    """Bound DNS resolution independently of the operating-system resolver."""
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise DependencyInspectionError("runtime_limit")
    code = (
        "import json,socket; "
        "rows=socket.getaddrinfo('registry.npmjs.org',443,type=socket.SOCK_STREAM); "
        "print(json.dumps(list(dict.fromkeys(row[4][0] for row in rows))[:32]))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code],
            capture_output=True,
            check=False,
            close_fds=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            timeout=min(10.0, remaining),
        )
        if result.returncode != 0 or len(result.stdout) > 4096:
            raise ValueError("registry DNS resolution failed")
        addresses = json.loads(result.stdout)
        if not isinstance(addresses, list) or not all(isinstance(ip, str) for ip in addresses):
            raise ValueError("invalid registry DNS result")
        return addresses
    except subprocess.TimeoutExpired as exc:
        reason = "runtime_limit" if monotonic() >= deadline else "dependency_download_failed"
        raise DependencyInspectionError(reason) from exc
    except (OSError, ValueError) as exc:
        raise DependencyInspectionError("dependency_download_failed") from exc


def _download_artifact(
    artifact: LockedArtifact, *, max_bytes: int, deadline: float, usage: dict[str, int]
) -> bytes:
    """Enforce the absolute acquisition deadline outside all HTTP operations."""
    ips = _resolve_registry_addresses(deadline)
    if not ips or any(not ipaddress.ip_address(address).is_global for address in ips):
        raise DependencyInspectionError("dependency_download_failed")
    request = json.dumps(
        {"url": artifact.url, "address": ips[0], "max_bytes": max_bytes, "deadline": deadline}
    ).encode()
    with tempfile.TemporaryFile() as output:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise DependencyInspectionError("runtime_limit")
        try:
            # subprocess.run kills and reaps the downloader on timeout, even if
            # headers, a transport read, or HTTP client cleanup never returns.
            process = subprocess.run(
                [sys.executable, "-I", "-B", "-m", "skillspector.dependency_downloader"],
                input=request,
                stdout=output,
                stderr=subprocess.DEVNULL,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                timeout=remaining,
                check=False,
                close_fds=True,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            # A terminated child cannot report exact partial consumption. Charge
            # its grant rather than letting failed acquisition renew the budget.
            usage["bytes"] += max_bytes
            reason = (
                "runtime_limit"
                if isinstance(exc, subprocess.TimeoutExpired)
                else "dependency_download_failed"
            )
            raise DependencyInspectionError(reason) from exc
        expired = monotonic() >= deadline
        if process.returncode != 0 or expired:
            usage["bytes"] += max_bytes
            reason = (
                "runtime_limit"
                if expired
                else {3: "runtime_limit", 4: "size_limit"}.get(
                    process.returncode, "dependency_download_failed"
                )
            )
            raise DependencyInspectionError(reason)
        output.seek(0)
        blob = output.read(max_bytes + 1)
    usage["bytes"] += len(blob)
    if len(blob) > max_bytes:
        raise DependencyInspectionError("size_limit")
    return blob


def _stream_artifact(
    url: str, address: str, *, max_bytes: int, deadline: float, usage: dict[str, int]
) -> bytes:
    """Download only in the timed child; retain TLS SNI and certificate checks."""
    if monotonic() >= deadline:
        raise DependencyInspectionError("runtime_limit")
    host = f"[{address}]" if ":" in address else address
    path = urlsplit(url).path
    timeout = min(10.0, max(0.001, deadline - monotonic()))
    with httpx.Client(trust_env=False, follow_redirects=False, timeout=timeout) as client:
        with client.stream(
            "GET",
            f"https://{host}{path}",
            headers={"Host": REGISTRY, "Accept-Encoding": "identity"},
            extensions={"sni_hostname": REGISTRY},
        ) as response:
            if monotonic() >= deadline:
                raise DependencyInspectionError("runtime_limit")
            if (
                response.status_code != 200
                or response.headers.get("content-encoding", "identity") != "identity"
            ):
                raise DependencyInspectionError("dependency_download_failed")
            body = bytearray()
            # Yield transport chunks immediately: a fixed chunk size buffers
            # trickling bytes and can suppress deadline checks until 64 KiB.
            for chunk in response.iter_raw():
                usage["bytes"] += len(chunk)
                if monotonic() >= deadline:
                    raise DependencyInspectionError("runtime_limit")
                if len(body) + len(chunk) > max_bytes:
                    raise DependencyInspectionError("size_limit")
                body.extend(chunk)
    if monotonic() >= deadline:
        raise DependencyInspectionError("runtime_limit")
    return bytes(body)


def _run_worker(
    artifact: LockedArtifact, blob: bytes, prefix: str, limits: dict, timeout: float
) -> dict:
    request = json.dumps(
        {
            "name": artifact.name,
            "version": artifact.version,
            "artifact": base64.b64encode(blob).decode("ascii"),
            "prefix": prefix,
            "limits": limits,
        }
    ).encode()
    # File-backed output bounds parent memory even if a worker is compromised.
    # The kernel file-size limit is set by the worker before it reads input.
    with tempfile.TemporaryDirectory(prefix="skillspector-npm-") as directory:
        with tempfile.TemporaryFile() as output:
            try:
                process = subprocess.run(
                    [sys.executable, "-I", "-B", "-m", "skillspector.dependency_worker"],
                    input=request,
                    stdout=output,
                    stderr=subprocess.DEVNULL,
                    cwd=directory,
                    env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                    timeout=timeout,
                    check=False,
                    close_fds=True,
                )
            except subprocess.TimeoutExpired as exc:
                raise DependencyInspectionError("runtime_limit") from exc
            output.seek(0)
            raw = output.read(4 * 1024 * 1024 + 1)
    if process.returncode != 0 or len(raw) > 4 * 1024 * 1024:
        raise DependencyInspectionError("dependency_worker_failed")
    try:
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("invalid worker response")
    except (ValueError, RecursionError) as exc:
        raise DependencyInspectionError("dependency_worker_failed") from exc
    if "error" in result:
        reason = result["error"]
        try:
            LedgerReason(reason)
        except (ValueError, TypeError):
            reason = "dependency_worker_failed"
        raise DependencyInspectionError(reason)
    return result


def _validate_worker_result(
    result: dict, prefix: str, limits: dict, existing_paths: set[str]
) -> None:
    """Validate the bounded worker protocol before accepting any report state."""
    try:
        for key, ceiling in (
            ("expanded_bytes", limits["expanded_bytes"]),
            ("members", limits["members"]),
        ):
            if type(result[key]) is not int or not 0 <= result[key] <= ceiling:
                raise ValueError("invalid worker accounting")
        files = result["files"]
        if not isinstance(files, list) or not 0 < len(files) <= result["members"]:
            raise ValueError("invalid worker inventory")
        limitations = result["limitations"]
        if not isinstance(limitations, list) or len(limitations) > 128:
            raise ValueError("invalid worker limitations")
        for reason in limitations:
            LedgerReason(reason)
        paths: set[str] = set()
        finding_ids: set[str] = set()
        total_findings = 0
        total_bytes = 0
        for row in files:
            path = row["path"]
            if not isinstance(path, str) or not path.startswith(prefix + "/"):
                raise ValueError("invalid worker path")
            relative = path[len(prefix) + 1 :]
            if (
                "\\" in relative
                or any(part in {"", ".", ".."} for part in relative.split("/"))
                or path in paths
                or path in existing_paths
            ):
                raise ValueError("colliding worker path")
            paths.add(path)
            if type(row["size_bytes"]) is not int or row["size_bytes"] < 0:
                raise ValueError("invalid worker file size")
            total_bytes += row["size_bytes"]
            if row["disposition"] not in {"analyzed", "partial"}:
                raise ValueError("invalid worker disposition")
            if row["disposition"] == "partial":
                LedgerReason(row["reason"])
            elif row.get("reason") is not None:
                raise ValueError("successful worker file carries a limitation")
            if not isinstance(row["findings"], list):
                raise ValueError("invalid worker findings")
            for item in row["findings"]:
                finding = Finding(**item)
                if (
                    finding.file != path
                    or finding.finding_id in finding_ids
                    or not isinstance(finding.finding_id, str)
                    or type(finding.start_line) is not int
                    or finding.start_line < 1
                ):
                    raise ValueError("invalid worker finding identity")
                finding_ids.add(finding.finding_id)
            total_findings += len(row["findings"])
        if total_findings > limits["findings"] or total_bytes > result["expanded_bytes"]:
            raise ValueError("worker output exceeds allowance")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise DependencyInspectionError("dependency_worker_failed") from exc


def inspect_dependencies(
    state: SkillspectorState, *, limits: DependencyInspectionLimits | None = None
) -> dict[str, object]:
    """Inspect requested locked packages and append canonical coverage evidence."""
    summary: dict[str, object] = {
        "enabled": state.get("inspect_dependencies") is True,
        "status": "not_requested",
        "mode": "static_local",
        "locked_entries": 0,
        "inspected_artifacts": 0,
        "artifacts": [],
    }
    if not summary["enabled"]:
        return {"dependency_inspection": summary}
    limits = limits or _configured_limits()
    budget = ensure_workflow_resource_budget(state)
    deadline = monotonic() + min(limits.seconds, budget.remaining_seconds())
    artifacts, events, entries = _collect_artifacts(state, limits, deadline=deadline)
    findings: list[Finding] = []
    inventory = [dict(row) for row in state.get("artifact_inventory") or []]
    metadata = [dict(row) for row in state.get("component_metadata") or []]
    usage = {"bytes": 0}
    expanded = members = 0
    records: list[dict] = []
    inspected = 0
    for artifact in artifacts:
        reference = artifact.references[0]
        location = f"{reference['lockfile']}!/artifacts/{len(records)}"
        record = {
            "name": artifact.name,
            "version": artifact.version,
            "references": artifact.references,
            "status": "partial",
        }
        records.append(record)
        try:
            remaining = min(deadline - monotonic(), budget.remaining_seconds())
            available = min(
                limits.artifact_bytes,
                limits.compressed_bytes - usage["bytes"],
                budget.remaining_bytes(),
            )
            available_expanded = min(
                limits.artifact_expanded_bytes,
                limits.expanded_bytes - expanded,
                budget.remaining_bytes(),
            )
            available_members = min(
                limits.artifact_members, limits.members - members, budget.remaining_artifacts()
            )
            if remaining <= 0:
                raise DependencyInspectionError("runtime_limit")
            if available <= 0 or available_expanded <= 0:
                raise DependencyInspectionError("total_bytes_limit")
            if available_members <= 0:
                raise DependencyInspectionError("artifact_count_limit")
            if len(findings) >= limits.findings:
                raise DependencyInspectionError("output_limit")
            try:
                previous_downloaded = usage["bytes"]
                try:
                    blob = _download_artifact(
                        artifact, max_bytes=available, deadline=deadline, usage=usage
                    )
                finally:
                    budget.record_bytes(usage["bytes"] - previous_downloaded)
            except (httpx.HTTPError, OSError, ValueError) as exc:
                if isinstance(exc, DependencyInspectionError):
                    raise
                raise DependencyInspectionError("dependency_download_failed") from exc
            digest = hashlib.sha512(blob).digest()
            if not any(hmac.compare_digest(digest, expected) for expected in artifact.digests):
                raise DependencyInspectionError("dependency_integrity_mismatch")
            digest_text = "sha256:" + hashlib.sha256(blob).hexdigest()
            prefix = (
                "npm/"
                + hashlib.sha256(
                    (artifact.name + artifact.version + digest_text).encode()
                ).hexdigest()
            )
            record["digest"] = digest_text
            available_expanded = min(available_expanded, budget.remaining_bytes())
            if available_expanded <= 0:
                raise DependencyInspectionError("total_bytes_limit")
            remaining = min(
                limits.worker_seconds, deadline - monotonic(), budget.remaining_seconds()
            )
            if remaining <= 0:
                raise DependencyInspectionError("runtime_limit")
            worker_limits = {
                "expanded_bytes": available_expanded,
                "file_bytes": limits.file_bytes,
                "members": available_members,
                "findings": limits.findings - len(findings),
                "compression_ratio": limits.compression_ratio,
                "seconds": remaining,
            }
            try:
                result = _run_worker(artifact, blob, prefix, worker_limits, remaining)
                _validate_worker_result(
                    result, prefix, worker_limits, {row["path"] for row in inventory}
                )
            except DependencyInspectionError:
                expanded += available_expanded
                members += available_members
                budget.record_bytes(available_expanded)
                budget.record_artifacts(available_members)
                raise
            expanded += result["expanded_bytes"]
            members += result["members"]
            budget.record_bytes(result["expanded_bytes"])
            budget.record_artifacts(result["members"])
            for row in result["files"]:
                row_findings = [Finding(**item) for item in row.pop("findings")]
                for finding in row_findings:
                    finding.source_digest = digest_text
                    finding.source_url = artifact.url
                    finding.source_identity = "external/" + prefix.removeprefix("npm/")
                    finding.tags = list(dict.fromkeys([*finding.tags, "dependency", "local-only"]))
                    finding.evidence.update(
                        dependency={
                            "name": artifact.name,
                            "version": artifact.version,
                            "digest": digest_text,
                            "references": artifact.references,
                        },
                        local_only=True,
                    )
                findings.extend(row_findings)
                events.append(_event(row["path"], row.get("reason"), findings=row_findings))
                inventory.append(row)
                metadata.append(
                    {
                        "path": row["path"],
                        "type": "npm-dependency",
                        "lines": 0,
                        "size_bytes": row["size_bytes"],
                        "executable": False,
                        "local_only": True,
                        "source_digest": digest_text,
                    }
                )
            for reason in result["limitations"]:
                events.append(_event(location, reason, system=True))
            if not result["limitations"]:
                inspected += 1
                record["status"] = "completed"
        except (DependencyInspectionError, OSError, ValueError, KeyError, TypeError) as exc:
            reason = (
                exc.reason
                if isinstance(exc, DependencyInspectionError)
                else "dependency_worker_failed"
            )
            record["reason_code"] = reason
            events.append(_event(location, reason, system=True))
            # Failed acquisition/parsing must not grant another fresh allowance.
            # Stop work after resource exhaustion or an unaccountable worker.
            if reason in {"size_limit", "runtime_limit", "dependency_worker_failed"}:
                usage["bytes"] = limits.compressed_bytes
                expanded = limits.expanded_bytes
    summary.update(
        status="partial"
        if any(event.get("outcome") == "partial" for event in events)
        else "completed",
        locked_entries=entries,
        inspected_artifacts=inspected,
        artifacts=records,
        download_bytes_charged=usage["bytes"],
        expanded_bytes_charged=expanded,
    )
    inventory_by_path = {row["path"]: row for row in inventory}
    for event in events:
        if event.get("outcome") != "partial":
            continue
        path = event["path"]
        if path not in inventory_by_path:
            row = {
                "path": path,
                "size_bytes": 0,
                "content_kind": "opaque",
                "disposition": "partial",
                "decodable": False,
                "contains_nul": False,
                "misleading_extension": False,
                "referenced": False,
            }
            inventory.append(row)
            inventory_by_path[path] = row
        inventory_by_path[path].update(disposition="partial", reason=event["reason_code"])
    status = analyzer_status_for_events(
        ANALYZER_ID, [event for event in events if event["record_type"] == "work_item"]
    )
    if summary["status"] == "partial" and status["status"] in {"completed", "not_applicable"}:
        reason = next(event["reason_code"] for event in events if event.get("outcome") == "partial")
        status = analyzer_status_event(
            analyzer_id=ANALYZER_ID,
            status="degraded",
            planned_work=status["planned_work"],
            reason=LedgerReason(reason),
        )
    return {
        "dependency_inspection": summary,
        "findings": findings,
        "inspection_ledger": events,
        "analyzer_status_events": [status],
        "artifact_inventory": inventory,
        "component_metadata": metadata,
    }
