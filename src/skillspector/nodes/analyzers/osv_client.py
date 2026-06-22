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

"""OSV.dev API client for live vulnerability lookups (SC4).

Queries the OSV.dev batch API to check whether dependencies have known
vulnerabilities.  Falls back to a small static list when the API is
unreachable (network error, timeout, air-gapped environment).

See https://google.github.io/osv.dev/post-v1-querybatch/ for API docs.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import httpx

from skillspector.logging_config import get_logger

logger = get_logger(__name__)

_OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
_OSV_VULN_URL = "https://api.osv.dev/v1/vulns"
_REQUEST_TIMEOUT: float = 30.0
if (env_val := os.environ.get("SKILLSPECTOR_OSV_TIMEOUT")) is not None:
    try:
        _REQUEST_TIMEOUT = float(env_val)
    except ValueError:
        logger.warning(
            "SKILLSPECTOR_OSV_TIMEOUT=%r is not numeric, using default %.1fs",
            env_val,
            _REQUEST_TIMEOUT,
        )

# Tracks whether the last query_batch() API call succeeded.
# Used by the supply-chain analyzer to surface fallback warnings.
_last_query_ok: bool = True

# Ecosystem identifiers expected by OSV.dev (case-sensitive).
ECOSYSTEM_PYPI = "PyPI"
ECOSYSTEM_NPM = "npm"


@dataclass(frozen=True)
class VulnResult:
    """A single vulnerability found for a package."""

    vuln_id: str
    summary: str
    severity: str
    aliases: tuple[str, ...]


# ---------------------------------------------------------------------------
# In-memory cache: (name, version, ecosystem) -> list[VulnResult]
# ---------------------------------------------------------------------------
_cache: dict[tuple[str, str | None, str], tuple[float, list[VulnResult]]] = {}
_CACHE_TTL_SECS = 3600.0  # 1 hour


def _cache_key(name: str, version: str | None, ecosystem: str) -> tuple[str, str | None, str]:
    return (name.lower().replace("_", "-"), version, ecosystem)


def _get_cached(key: tuple[str, str | None, str]) -> list[VulnResult] | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, results = entry
    if (time.monotonic() - ts) > _CACHE_TTL_SECS:
        del _cache[key]
        return None
    return results


def _put_cache(key: tuple[str, str | None, str], results: list[VulnResult]) -> None:
    _cache[key] = (time.monotonic(), results)


def clear_cache() -> None:
    """Clear the in-memory vulnerability cache."""
    _cache.clear()


# ---------------------------------------------------------------------------
# OSV API helpers
# ---------------------------------------------------------------------------


def _build_query(name: str, version: str | None, ecosystem: str) -> dict:
    q: dict = {"package": {"name": name, "ecosystem": ecosystem}}
    if version:
        q["version"] = version
    return q


_CVSS_VECTOR_RE = re.compile(r"CVSS:[34][.\d]*/(.+)")

# Worst-case metric values used to estimate severity from a CVSS vector.
# Not a full CVSS calculator — intentionally coarse for triage purposes.
_CVSS_HIGH_METRICS = {
    # v3 base metrics
    "AV:N",
    "AC:L",
    "PR:N",
    "UI:N",
    "S:C",
    "C:H",
    "I:H",
    "A:H",
    # v4 additions (vulnerable & subsequent system impact)
    "AT:N",
    "VC:H",
    "VI:H",
    "VA:H",
    "SC:H",
    "SI:H",
    "SA:H",
}


def _estimate_cvss_severity(vector: str) -> str | None:
    """Estimate severity from a CVSS v3 or v4 vector string.

    Counts how many base metrics are at their most-severe value.
    This avoids adding a CVSS library dependency while giving a reasonable
    approximation for triage purposes.
    """
    m = _CVSS_VECTOR_RE.match(vector)
    if not m:
        return None
    metrics = m.group(1).split("/")
    high_count = sum(1 for metric in metrics if metric in _CVSS_HIGH_METRICS)
    total = len(metrics)
    if total == 0:
        return None
    ratio = high_count / total
    if ratio >= 0.75:
        return "CRITICAL"
    if ratio >= 0.5:
        return "HIGH"
    if ratio >= 0.25:
        return "MEDIUM"
    return "LOW"


def _severity_from_vuln(vuln: dict) -> str:
    """Extract the highest severity string from an OSV vulnerability object.

    Priority order:
    1. database_specific.severity — GHSA sets this reliably (e.g. "HIGH").
    2. affected[].ecosystem_specific.severity — set by some ecosystems.
    3. severity[].score CVSS vector — parsed to estimate severity band.
    4. Default to "HIGH" when no severity info is available.
    """
    db_specific = vuln.get("database_specific", {})
    ghsa_severity = db_specific.get("severity", "")
    if ghsa_severity:
        return ghsa_severity.upper()
    for affected in vuln.get("affected", []):
        eco_specific = affected.get("ecosystem_specific", {})
        sev = eco_specific.get("severity", "")
        if sev:
            return sev.upper()
    for severity_entry in vuln.get("severity", []):
        score_str = severity_entry.get("score", "")
        if score_str:
            estimated = _estimate_cvss_severity(score_str)
            if estimated:
                return estimated
    return "HIGH"


def _parse_vuln(vuln: dict) -> VulnResult:
    aliases = tuple(vuln.get("aliases", []))
    return VulnResult(
        vuln_id=vuln.get("id", "UNKNOWN"),
        summary=vuln.get("summary", vuln.get("details", "")[:200]),
        severity=_severity_from_vuln(vuln),
        aliases=aliases,
    )


def _fetch_vuln_details(vuln_ids: list[str]) -> list[VulnResult]:
    """Fetch full vulnerability details for a list of IDs."""
    if len(vuln_ids) > 10:
        logger.warning("Processing 10 of %d vulnerabilities, truncating the rest", len(vuln_ids))
    results: list[VulnResult] = []
    with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
        for vid in vuln_ids[:10]:
            try:
                resp = client.get(f"{_OSV_VULN_URL}/{vid}")
                resp.raise_for_status()
                results.append(_parse_vuln(resp.json()))
            except (httpx.HTTPError, KeyError, ValueError):
                results.append(
                    VulnResult(
                        vuln_id=vid,
                        summary="",
                        severity="HIGH",
                        aliases=(),
                    )
                )
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def query_batch(
    packages: list[tuple[str, str | None]],
    ecosystem: str,
) -> list[list[VulnResult]]:
    """Query OSV.dev for vulnerabilities across a batch of packages.

    Args:
        packages: List of (name, version_or_None) tuples.
        ecosystem: ``"PyPI"`` or ``"npm"``.

    Returns:
        A list parallel to *packages* where each element is a
        (possibly empty) list of :class:`VulnResult`.

    Raises nothing — on network/API failure returns empty lists for all
    packages (caller should fall back to static data).
    """
    global _last_query_ok

    if not packages:
        return []

    all_results: list[list[VulnResult]] = [[] for _ in packages]

    uncached_indices: list[int] = []
    uncached_queries: list[dict] = []

    for i, (name, version) in enumerate(packages):
        key = _cache_key(name, version, ecosystem)
        cached = _get_cached(key)
        if cached is not None:
            all_results[i] = cached
        else:
            uncached_indices.append(i)
            uncached_queries.append(_build_query(name, version, ecosystem))

    if not uncached_queries:
        return all_results

    try:
        with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
            resp = client.post(_OSV_BATCH_URL, json={"queries": uncached_queries})
            resp.raise_for_status()
            batch_results = resp.json().get("results", [])

        _last_query_ok = True

        for batch_idx, idx in enumerate(uncached_indices):
            if batch_idx >= len(batch_results):
                break
            vulns_raw = batch_results[batch_idx].get("vulns", [])
            if not vulns_raw:
                name, version = packages[idx]
                _put_cache(_cache_key(name, version, ecosystem), [])
                logger.info(
                    "OSV.dev: no vulnerabilities found for %s==%s (passed)",
                    name,
                    version or "unspecified",
                )
                continue

            vuln_ids = [v["id"] for v in vulns_raw if "id" in v]
            vuln_details = _fetch_vuln_details(vuln_ids)
            all_results[idx] = vuln_details

            name, version = packages[idx]
            _put_cache(_cache_key(name, version, ecosystem), vuln_details)

    except (httpx.HTTPError, httpx.TimeoutException, ValueError, KeyError) as exc:
        logger.warning("OSV.dev API request failed, falling back to static data: %s", exc)
        _last_query_ok = False
        return [[] for _ in packages]

    return all_results


def is_available() -> bool:
    """Quick connectivity check against the OSV.dev API (HEAD-like POST)."""
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                _OSV_BATCH_URL,
                json={"queries": [{"package": {"name": "pip", "ecosystem": "PyPI"}}]},
            )
            return resp.status_code == 200
    except (httpx.HTTPError, httpx.TimeoutException):
        return False


def was_osv_reachable() -> bool:
    """Return True if the last query_batch() call succeeded.

    Callers can use this to decide whether to surface a fallback warning
    when query_batch returns empty results.
    """
    return _last_query_ok
