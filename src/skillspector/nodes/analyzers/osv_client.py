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

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from skillspector.inspection_ledger import LedgerReason
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

# All OSV limits are aggregate for one ``OsvQueryBudget``.  The supply-chain
# node creates one budget and shares it across every dependency manifest, so a
# bundle containing many manifests cannot multiply any of these ceilings.
MAX_OSV_PACKAGES = 256
MAX_OSV_QUERY_BATCHES = 4
MAX_OSV_QUERIES_PER_BATCH = 64
MAX_OSV_DETAIL_REQUESTS = 64
MAX_OSV_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_OSV_RESULTS = 256
MAX_OSV_VULNS_PER_PACKAGE = 16
MAX_OSV_LIMITATIONS = 16
MAX_OSV_CACHE_ENTRIES = 4_096
MAX_OSV_PACKAGE_NAME_CHARS = 256
MAX_OSV_PACKAGE_VERSION_CHARS = 128
MAX_OSV_ID_CHARS = 256
MAX_OSV_SUMMARY_CHARS = 512
MAX_OSV_ALIASES = 16

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


@dataclass(frozen=True)
class OsvQueryLimitation:
    """Content-free metadata describing an intentionally incomplete lookup."""

    reason: LedgerReason
    observed_records: int | None = None
    limit_records: int | None = None
    observed_bytes: int | None = None
    limit_bytes: int | None = None
    observed_characters: int | None = None
    limit_characters: int | None = None
    observed_seconds: float | None = None
    limit_seconds: float | None = None
    error_class: str | None = None


class QueryBatchResults(list[list[VulnResult]]):
    """Bounded list result carrying non-fatal lookup limitations."""

    def __init__(
        self,
        values: list[list[VulnResult]],
        *,
        limitations: tuple[OsvQueryLimitation, ...] = (),
    ) -> None:
        super().__init__(values)
        self.limitations = limitations


@dataclass
class OsvQueryBudget:
    """Aggregate request, response, result, and deadline budget for OSV."""

    started_at: float
    deadline: float
    limit_seconds: float
    max_packages: int
    max_batches: int
    max_queries_per_batch: int
    max_detail_requests: int
    max_response_bytes: int
    max_results: int
    packages_seen: int = 0
    batches_sent: int = 0
    detail_requests: int = 0
    response_bytes: int = 0
    results_retained: int = 0
    limitations: list[OsvQueryLimitation] | None = None
    limitation_generation: int = 0
    last_limitation: OsvQueryLimitation | None = None

    @classmethod
    def create(cls, timeout_seconds: float | None = None) -> OsvQueryBudget:
        """Create a budget capped by both OSV and the shared workflow deadline."""
        started_at = time.monotonic()
        requested = max(0.0, _REQUEST_TIMEOUT)
        if timeout_seconds is not None:
            requested = min(requested, max(0.0, timeout_seconds))
        return cls(
            started_at=started_at,
            deadline=started_at + requested,
            limit_seconds=requested,
            max_packages=max(0, MAX_OSV_PACKAGES),
            max_batches=max(0, MAX_OSV_QUERY_BATCHES),
            max_queries_per_batch=max(1, MAX_OSV_QUERIES_PER_BATCH),
            max_detail_requests=max(0, MAX_OSV_DETAIL_REQUESTS),
            max_response_bytes=max(0, MAX_OSV_RESPONSE_BYTES),
            max_results=max(0, MAX_OSV_RESULTS),
            limitations=[],
        )

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def note(self, limitation: OsvQueryLimitation) -> None:
        """Record bounded, deduplicated limitation metadata."""
        self.limitation_generation += 1
        self.last_limitation = limitation
        if self.limitations is None:
            self.limitations = []
        if limitation in self.limitations:
            return
        if len(self.limitations) < max(1, MAX_OSV_LIMITATIONS):
            self.limitations.append(limitation)
            return
        self.limitations[-1] = OsvQueryLimitation(
            reason=LedgerReason.OUTPUT_LIMIT,
            observed_records=len(self.limitations) + 1,
            limit_records=max(1, MAX_OSV_LIMITATIONS),
        )

    def note_runtime_limit(self) -> None:
        self.note(
            OsvQueryLimitation(
                reason=LedgerReason.RUNTIME_LIMIT,
                observed_seconds=max(0.0, time.monotonic() - self.started_at),
                limit_seconds=self.limit_seconds,
            )
        )


def _limitations_since(
    budget: OsvQueryBudget,
    *,
    start_index: int,
    start_generation: int,
) -> tuple[OsvQueryLimitation, ...]:
    """Return at least one limitation when this call recorded an omitted work item."""
    retained = tuple((budget.limitations or [])[start_index:])
    if retained or budget.limitation_generation == start_generation:
        return retained
    return (budget.last_limitation,) if budget.last_limitation is not None else ()


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
    return list(results[:MAX_OSV_VULNS_PER_PACKAGE])


def _put_cache(key: tuple[str, str | None, str], results: list[VulnResult]) -> None:
    # The cache is process-global, so keep both its keys and values bounded.
    if key not in _cache and len(_cache) >= max(1, MAX_OSV_CACHE_ENTRIES):
        del _cache[next(iter(_cache))]
    _cache[key] = (time.monotonic(), list(results[:MAX_OSV_VULNS_PER_PACKAGE]))


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
    ghsa_severity = db_specific.get("severity", "") if isinstance(db_specific, dict) else ""
    if isinstance(ghsa_severity, str) and ghsa_severity:
        return ghsa_severity[:32].upper()
    raw_affected = vuln.get("affected", [])
    for affected in raw_affected if isinstance(raw_affected, list) else []:
        if not isinstance(affected, dict):
            continue
        eco_specific = affected.get("ecosystem_specific", {})
        sev = eco_specific.get("severity", "") if isinstance(eco_specific, dict) else ""
        if isinstance(sev, str) and sev:
            return sev[:32].upper()
    raw_severity = vuln.get("severity", [])
    for severity_entry in raw_severity if isinstance(raw_severity, list) else []:
        if not isinstance(severity_entry, dict):
            continue
        score_str = severity_entry.get("score", "")
        if isinstance(score_str, str) and score_str:
            estimated = _estimate_cvss_severity(score_str[:1_024])
            if estimated:
                return estimated
    return "HIGH"


def _parse_vuln(vuln: dict) -> VulnResult:
    raw_aliases = vuln.get("aliases", [])
    aliases = (
        tuple(
            alias[:MAX_OSV_ID_CHARS]
            for alias in raw_aliases[:MAX_OSV_ALIASES]
            if isinstance(alias, str)
        )
        if isinstance(raw_aliases, list)
        else ()
    )
    vuln_id = vuln.get("id", "UNKNOWN")
    if not isinstance(vuln_id, str):
        vuln_id = "UNKNOWN"
    summary = vuln.get("summary")
    if not isinstance(summary, str):
        details = vuln.get("details", "")
        summary = details if isinstance(details, str) else ""
    return VulnResult(
        vuln_id=vuln_id[:MAX_OSV_ID_CHARS],
        summary=summary[:MAX_OSV_SUMMARY_CHARS],
        severity=_severity_from_vuln(vuln),
        aliases=aliases,
    )


class _OsvLimitReachedError(RuntimeError):
    """Private control-flow exception for a recorded resource limit."""


def _fallback_vuln(vuln_id: str) -> VulnResult:
    """Retain a vulnerability signal when bounded detail enrichment is omitted."""
    return VulnResult(
        vuln_id=vuln_id[:MAX_OSV_ID_CHARS],
        summary="",
        severity="HIGH",
        aliases=(),
    )


def _request_json_bounded(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    budget: OsvQueryBudget,
    payload: dict[str, object] | None = None,
) -> Any:
    """Read and parse one response without exceeding the shared byte/deadline cap."""
    remaining_seconds = budget.remaining_seconds()
    if remaining_seconds <= 0:
        budget.note_runtime_limit()
        raise _OsvLimitReachedError("runtime")

    remaining_bytes = budget.max_response_bytes - budget.response_bytes
    if remaining_bytes <= 0:
        budget.note(
            OsvQueryLimitation(
                reason=LedgerReason.TOTAL_BYTES_LIMIT,
                observed_bytes=budget.response_bytes + 1,
                limit_bytes=budget.max_response_bytes,
            )
        )
        raise _OsvLimitReachedError("response_bytes")

    # Real httpx responses are streamed, so an oversized provider body is
    # stopped before it is materialized.  The fallback branch keeps support
    # for lightweight response doubles used by downstream callers.
    response_double: object | None = None
    with client.stream(
        method,
        url,
        json=payload,
        timeout=remaining_seconds,
    ) as response:
        if isinstance(response, httpx.Response):
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            if content_length and content_length.isdigit():
                declared = int(content_length)
                if declared > remaining_bytes:
                    budget.note(
                        OsvQueryLimitation(
                            reason=LedgerReason.TOTAL_BYTES_LIMIT,
                            observed_bytes=budget.response_bytes + declared,
                            limit_bytes=budget.max_response_bytes,
                        )
                    )
                    raise _OsvLimitReachedError("response_bytes")
            body = bytearray()
            for chunk in response.iter_bytes():
                if len(body) + len(chunk) > remaining_bytes:
                    budget.note(
                        OsvQueryLimitation(
                            reason=LedgerReason.TOTAL_BYTES_LIMIT,
                            observed_bytes=budget.response_bytes + len(body) + len(chunk),
                            limit_bytes=budget.max_response_bytes,
                        )
                    )
                    raise _OsvLimitReachedError("response_bytes")
                body.extend(chunk)
                if budget.remaining_seconds() <= 0:
                    budget.note_runtime_limit()
                    raise _OsvLimitReachedError("runtime")
            budget.response_bytes += len(body)
            return json.loads(body)
        response_double = response

    # ``MagicMock``-style clients historically supplied ``post``/``get``
    # response doubles rather than a streaming response.  Bound their parsed
    # representation too; production always takes the streaming branch above.
    if response_double is not None:
        if method == "POST":
            fallback_response = client.post(
                url,
                json=payload,
                timeout=remaining_seconds,
            )
        else:
            fallback_response = client.get(url, timeout=remaining_seconds)
        fallback_response.raise_for_status()
        parsed = fallback_response.json()
        encoded = json.dumps(parsed, separators=(",", ":")).encode("utf-8")
        if len(encoded) > remaining_bytes:
            budget.note(
                OsvQueryLimitation(
                    reason=LedgerReason.TOTAL_BYTES_LIMIT,
                    observed_bytes=budget.response_bytes + len(encoded),
                    limit_bytes=budget.max_response_bytes,
                )
            )
            raise _OsvLimitReachedError("response_bytes")
        budget.response_bytes += len(encoded)
        return parsed
    raise ValueError("OSV response was unavailable")


def _fetch_vuln_details(
    vuln_ids: list[str],
    *,
    client: httpx.Client | None = None,
    budget: OsvQueryBudget | None = None,
) -> list[VulnResult]:
    """Fetch vulnerability details under one aggregate request/result budget."""
    active_budget = budget or OsvQueryBudget.create()
    owns_client = client is None
    active_client = client or httpx.Client(timeout=active_budget.remaining_seconds())
    results: list[VulnResult] = []
    try:
        for vuln_id in vuln_ids[:MAX_OSV_VULNS_PER_PACKAGE]:
            if active_budget.results_retained >= active_budget.max_results:
                active_budget.note(
                    OsvQueryLimitation(
                        reason=LedgerReason.OUTPUT_LIMIT,
                        observed_records=active_budget.results_retained + 1,
                        limit_records=active_budget.max_results,
                    )
                )
                break
            if active_budget.detail_requests >= active_budget.max_detail_requests:
                active_budget.note(
                    OsvQueryLimitation(
                        reason=LedgerReason.OUTPUT_LIMIT,
                        observed_records=active_budget.detail_requests + 1,
                        limit_records=active_budget.max_detail_requests,
                    )
                )
                result = _fallback_vuln(vuln_id)
            elif active_budget.remaining_seconds() <= 0:
                active_budget.note_runtime_limit()
                result = _fallback_vuln(vuln_id)
            else:
                active_budget.detail_requests += 1
                try:
                    payload = _request_json_bounded(
                        active_client,
                        "GET",
                        f"{_OSV_VULN_URL}/{quote(vuln_id, safe='')}",
                        budget=active_budget,
                    )
                    result = (
                        _parse_vuln(payload)
                        if isinstance(payload, dict)
                        else _fallback_vuln(vuln_id)
                    )
                except _OsvLimitReachedError:
                    result = _fallback_vuln(vuln_id)
                except httpx.TimeoutException:
                    active_budget.note_runtime_limit()
                    result = _fallback_vuln(vuln_id)
                except (
                    httpx.HTTPError,
                    ValueError,
                    KeyError,
                    TypeError,
                    RecursionError,
                ) as exc:
                    active_budget.note(
                        OsvQueryLimitation(
                            reason=LedgerReason.ANALYZER_RUNTIME_ERROR,
                            error_class=type(exc).__name__,
                        )
                    )
                    result = _fallback_vuln(vuln_id)
            results.append(result)
            active_budget.results_retained += 1
    finally:
        if owns_client:
            active_client.close()
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def query_batch(
    packages: list[tuple[str, str | None]],
    ecosystem: str,
    *,
    timeout_seconds: float | None = None,
    budget: OsvQueryBudget | None = None,
) -> QueryBatchResults:
    """Query OSV.dev for vulnerabilities across a batch of packages.

    Args:
        packages: List of (name, version_or_None) tuples.
        ecosystem: ``"PyPI"`` or ``"npm"``.

    Returns:
        A bounded list parallel to the retained prefix of *packages* where
        each element is a (possibly empty) list of :class:`VulnResult`.
        ``result.limitations`` describes any omitted work without provider
        payloads or exception text.

    Raises nothing — on network/API failure returns empty lists for all
    packages (caller should fall back to static data).
    """
    global _last_query_ok

    active_budget = budget or OsvQueryBudget.create(timeout_seconds)
    limitations_start = len(active_budget.limitations or [])
    limitations_generation = active_budget.limitation_generation
    if not packages:
        return QueryBatchResults([])

    remaining_packages = max(0, active_budget.max_packages - active_budget.packages_seen)
    retained_packages = packages[:remaining_packages]
    if len(packages) > remaining_packages:
        active_budget.note(
            OsvQueryLimitation(
                reason=LedgerReason.OUTPUT_LIMIT,
                observed_records=active_budget.packages_seen + len(packages),
                limit_records=active_budget.max_packages,
            )
        )
    active_budget.packages_seen += len(retained_packages)
    all_results: list[list[VulnResult]] = [[] for _ in retained_packages]
    if not retained_packages:
        return QueryBatchResults(
            all_results,
            limitations=_limitations_since(
                active_budget,
                start_index=limitations_start,
                start_generation=limitations_generation,
            ),
        )

    uncached_indices: list[int] = []
    uncached_queries: list[dict] = []

    for i, (name, version) in enumerate(retained_packages):
        if not isinstance(name, str) or (version is not None and not isinstance(version, str)):
            active_budget.note(
                OsvQueryLimitation(
                    reason=LedgerReason.OPAQUE_CONTENT,
                    error_class="InvalidPackageCoordinate",
                )
            )
            continue
        if len(name) > MAX_OSV_PACKAGE_NAME_CHARS:
            active_budget.note(
                OsvQueryLimitation(
                    reason=LedgerReason.SIZE_LIMIT,
                    observed_characters=len(name),
                    limit_characters=MAX_OSV_PACKAGE_NAME_CHARS,
                )
            )
            continue
        if version is not None and len(version) > MAX_OSV_PACKAGE_VERSION_CHARS:
            active_budget.note(
                OsvQueryLimitation(
                    reason=LedgerReason.SIZE_LIMIT,
                    observed_characters=len(version),
                    limit_characters=MAX_OSV_PACKAGE_VERSION_CHARS,
                )
            )
            continue
        key = _cache_key(name, version, ecosystem)
        cached = _get_cached(key)
        if cached is not None:
            all_results[i] = cached
        else:
            uncached_indices.append(i)
            uncached_queries.append(_build_query(name, version, ecosystem))

    if not uncached_queries:
        return QueryBatchResults(
            all_results,
            limitations=_limitations_since(
                active_budget,
                start_index=limitations_start,
                start_generation=limitations_generation,
            ),
        )

    provider_failed = False
    successful_batches = 0
    try:
        with httpx.Client(timeout=max(0.001, active_budget.remaining_seconds())) as client:
            for start in range(0, len(uncached_queries), active_budget.max_queries_per_batch):
                if active_budget.remaining_seconds() <= 0:
                    active_budget.note_runtime_limit()
                    break
                if active_budget.batches_sent >= active_budget.max_batches:
                    active_budget.note(
                        OsvQueryLimitation(
                            reason=LedgerReason.OUTPUT_LIMIT,
                            observed_records=active_budget.batches_sent + 1,
                            limit_records=active_budget.max_batches,
                        )
                    )
                    break
                query_chunk = uncached_queries[start : start + active_budget.max_queries_per_batch]
                index_chunk = uncached_indices[start : start + active_budget.max_queries_per_batch]
                active_budget.batches_sent += 1
                payload = _request_json_bounded(
                    client,
                    "POST",
                    _OSV_BATCH_URL,
                    budget=active_budget,
                    payload={"queries": query_chunk},
                )
                if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                    raise ValueError("OSV batch response shape is invalid")
                successful_batches += 1
                batch_results = payload["results"]
                if len(batch_results) > len(index_chunk):
                    active_budget.note(
                        OsvQueryLimitation(
                            reason=LedgerReason.OUTPUT_LIMIT,
                            observed_records=len(batch_results),
                            limit_records=len(index_chunk),
                        )
                    )
                if len(batch_results) < len(index_chunk):
                    active_budget.note(
                        OsvQueryLimitation(
                            reason=LedgerReason.ANALYZER_RUNTIME_ERROR,
                            error_class="IncompleteBatchResponse",
                        )
                    )
                for batch_item, idx in zip(
                    batch_results[: len(index_chunk)], index_chunk, strict=False
                ):
                    if not isinstance(batch_item, dict):
                        active_budget.note(
                            OsvQueryLimitation(
                                reason=LedgerReason.ANALYZER_RUNTIME_ERROR,
                                error_class="InvalidBatchResult",
                            )
                        )
                        continue
                    vulns_raw = batch_item.get("vulns", [])
                    if not isinstance(vulns_raw, list):
                        active_budget.note(
                            OsvQueryLimitation(
                                reason=LedgerReason.ANALYZER_RUNTIME_ERROR,
                                error_class="InvalidVulnerabilityList",
                            )
                        )
                        continue
                    name, version = retained_packages[idx]
                    if not vulns_raw:
                        _put_cache(_cache_key(name, version, ecosystem), [])
                        logger.info(
                            "OSV.dev: no vulnerabilities found for %s==%s (passed)",
                            name,
                            version or "unspecified",
                        )
                        continue
                    package_generation = active_budget.limitation_generation
                    if len(vulns_raw) > MAX_OSV_VULNS_PER_PACKAGE:
                        active_budget.note(
                            OsvQueryLimitation(
                                reason=LedgerReason.OUTPUT_LIMIT,
                                observed_records=len(vulns_raw),
                                limit_records=MAX_OSV_VULNS_PER_PACKAGE,
                            )
                        )
                    vuln_ids: list[str] = []
                    for raw_vuln in vulns_raw[:MAX_OSV_VULNS_PER_PACKAGE]:
                        if not isinstance(raw_vuln, dict):
                            active_budget.note(
                                OsvQueryLimitation(
                                    reason=LedgerReason.OPAQUE_CONTENT,
                                    error_class="InvalidVulnerabilityRecord",
                                )
                            )
                            continue
                        vuln_id = raw_vuln.get("id")
                        if not isinstance(vuln_id, str) or not vuln_id:
                            active_budget.note(
                                OsvQueryLimitation(
                                    reason=LedgerReason.OPAQUE_CONTENT,
                                    error_class="InvalidVulnerabilityId",
                                )
                            )
                            continue
                        if len(vuln_id) > MAX_OSV_ID_CHARS:
                            active_budget.note(
                                OsvQueryLimitation(
                                    reason=LedgerReason.SIZE_LIMIT,
                                    observed_characters=len(vuln_id),
                                    limit_characters=MAX_OSV_ID_CHARS,
                                )
                            )
                            continue
                        if vuln_id not in vuln_ids:
                            vuln_ids.append(vuln_id)
                    vuln_details = _fetch_vuln_details(
                        vuln_ids,
                        client=client,
                        budget=active_budget,
                    )
                    all_results[idx] = vuln_details
                    if active_budget.limitation_generation == package_generation:
                        _put_cache(_cache_key(name, version, ecosystem), vuln_details)

        _last_query_ok = successful_batches > 0
    except _OsvLimitReachedError:
        # The exact limit is already recorded on the result.  Keep completed
        # prefix results and let static fallback cover the remainder.
        _last_query_ok = successful_batches > 0
    except httpx.TimeoutException:
        logger.warning("OSV.dev API request timed out, falling back to static data")
        active_budget.note_runtime_limit()
        provider_failed = True
        _last_query_ok = False
    except (
        httpx.HTTPError,
        ValueError,
        KeyError,
        TypeError,
        RecursionError,
    ) as exc:
        logger.warning("OSV.dev API request failed, falling back to static data: %s", exc)
        active_budget.note(
            OsvQueryLimitation(
                reason=LedgerReason.ANALYZER_RUNTIME_ERROR,
                error_class=type(exc).__name__,
            )
        )
        provider_failed = True
        _last_query_ok = False

    if not provider_failed and _last_query_ok:
        _last_query_ok = True
    return QueryBatchResults(
        all_results,
        limitations=_limitations_since(
            active_budget,
            start_index=limitations_start,
            start_generation=limitations_generation,
        ),
    )


def is_available() -> bool:
    """Run a bounded connectivity check against the OSV.dev API."""
    try:
        budget = OsvQueryBudget.create(timeout_seconds=15.0)
        with httpx.Client(timeout=max(0.001, budget.remaining_seconds())) as client:
            payload = _request_json_bounded(
                client,
                "POST",
                _OSV_BATCH_URL,
                budget=budget,
                payload={"queries": [{"package": {"name": "pip", "ecosystem": ECOSYSTEM_PYPI}}]},
            )
            return isinstance(payload, dict) and isinstance(payload.get("results"), list)
    except (
        _OsvLimitReachedError,
        httpx.HTTPError,
        httpx.TimeoutException,
        ValueError,
        TypeError,
        RecursionError,
    ):
        return False


def was_osv_reachable() -> bool:
    """Return True if the last query_batch() call succeeded.

    Callers can use this to decide whether to surface a fallback warning
    when query_batch returns empty results.
    """
    return _last_query_ok
