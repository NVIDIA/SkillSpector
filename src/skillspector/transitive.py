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

"""Shared helpers for opt-in transitive external-source traversal."""

from __future__ import annotations

import heapq
import posixpath
import re
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from urllib.parse import ParseResult, unquote, urlparse, urlunparse

from skillspector.input_handler import ALLOWED_DOWNLOAD_HOSTS, ALLOWED_GIT_HOSTS

_LEADING_PUNCTUATION = "([{\"'<"
_TRAILING_PUNCTUATION = "),.!?;:>\"'`]}"

_SUPPORTED_FILE_EXTENSIONS = frozenset(
    {
        ".md",
        ".py",
        ".sh",
        ".bash",
        ".zsh",
        ".js",
        ".ts",
        ".rb",
        ".go",
        ".rs",
        ".pl",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".txt",
        ".zip",
    }
)

MAX_EXTERNAL_REFERENCE_SOURCES = 1024
MAX_EXTERNAL_REFERENCE_SOURCE_BYTES = 1_000_000
MAX_RAW_EXTERNAL_REFERENCE_CANDIDATES = 4096
MAX_ACCEPTED_EXTERNAL_REFERENCES = 256
MAX_EXTERNAL_REFERENCE_RECORDS = 1024
MAX_EXTERNAL_REFERENCE_SECONDS = 2.0
MAX_EXTERNAL_REFERENCE_TOKEN_CHARACTERS = 2048

MAX_TRANSITIVE_PLAN_INPUT_REFERENCES = 4096
MAX_TRANSITIVE_PLAN_TARGETS = 32
MAX_TRANSITIVE_PLAN_PREFIXES = 128
MAX_TRANSITIVE_PLAN_SECONDS = 1.0

MAX_TRANSITIVE_FRONTIER_WAVES = 32
MAX_TRANSITIVE_FRONTIER_REFERENCES = 4096

_EXTERNAL_REF_PATTERN = re.compile(
    rf"(?:https?://|git@)[^\s\"'<>`]{{1,{MAX_EXTERNAL_REFERENCE_TOKEN_CHARACTERS}}}"
    rf"(?![^\s\"'<>`])"
)

_EXCLUDED_HOSTS = frozenset(
    {
        "img.shields.io",
        "badge.fury.io",
        "travis-ci.com",
        "travis-ci.org",
    }
)

_UNRESERVED_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)
_PERCENT_ENCODED_RE = re.compile(r"%[0-9A-Fa-f]{2}")


@dataclass(frozen=True, slots=True)
class TransitiveResourceLimitation:
    """One deterministic resource ceiling reached during reference traversal."""

    resource: str
    observed: int | float
    limit: int | float
    source_scope: str | None = None


@dataclass(frozen=True, slots=True)
class ExternalReferenceRecord:
    """One bounded, accepted occurrence with an opaque report-safe source key."""

    source_scope: str
    source_url: str


@dataclass(frozen=True, slots=True)
class ExternalReferenceLimits:
    """Independent extraction limits; defaults are safe for compatibility callers."""

    max_sources: int = MAX_EXTERNAL_REFERENCE_SOURCES
    max_source_bytes: int = MAX_EXTERNAL_REFERENCE_SOURCE_BYTES
    max_raw_candidates: int = MAX_RAW_EXTERNAL_REFERENCE_CANDIDATES
    max_accepted_references: int = MAX_ACCEPTED_EXTERNAL_REFERENCES
    max_output_records: int = MAX_EXTERNAL_REFERENCE_RECORDS
    max_seconds: float = MAX_EXTERNAL_REFERENCE_SECONDS


@dataclass(frozen=True, slots=True)
class ExternalReferenceExtractionResult:
    """Bounded external references plus explicit input/work/output accounting."""

    references: list[str]
    records: list[ExternalReferenceRecord]
    complete: bool
    limitations: tuple[TransitiveResourceLimitation, ...]
    sources_observed: int
    sources_limit: int
    source_bytes_examined: int
    source_bytes_observed: int
    source_bytes_limit: int
    raw_candidates_observed: int
    raw_candidates_limit: int
    accepted_references_observed: int
    accepted_references_limit: int
    output_records_observed: int
    output_records_limit: int
    runtime_seconds: float
    runtime_seconds_limit: float


@dataclass(frozen=True, slots=True)
class TransitivePlanLimits:
    """Independent limits for one target-planning wave."""

    max_input_references: int = MAX_TRANSITIVE_PLAN_INPUT_REFERENCES
    max_targets: int = MAX_TRANSITIVE_PLAN_TARGETS
    max_prefixes: int = MAX_TRANSITIVE_PLAN_PREFIXES
    max_seconds: float = MAX_TRANSITIVE_PLAN_SECONDS


@dataclass(frozen=True, slots=True)
class TransitiveTargetPlan:
    """Bounded next-wave targets and the limits that affected the plan."""

    targets: list[str]
    complete: bool
    limitations: tuple[TransitiveResourceLimitation, ...]
    input_references_observed: int
    input_references_limit: int
    targets_observed: int
    targets_limit: int
    prefixes_observed: int
    prefixes_limit: int
    runtime_seconds: float
    runtime_seconds_limit: float


@dataclass(frozen=True, slots=True)
class TransitiveFrontierWave:
    """One bounded breadth-first traversal wave."""

    depth: int
    references: tuple[str, ...]


@dataclass(slots=True)
class BoundedTransitiveFrontier:
    """Small FIFO frontier that never retains attacker-controlled unbounded lists."""

    deadline: float
    clock: Callable[[], float] = time.monotonic
    max_waves: int = MAX_TRANSITIVE_FRONTIER_WAVES
    max_references: int = MAX_TRANSITIVE_FRONTIER_REFERENCES
    _waves: deque[TransitiveFrontierWave] = field(default_factory=deque, init=False)
    _queued_references: int = field(default=0, init=False)
    _references_observed: int = field(default=0, init=False)
    _limitations: list[TransitiveResourceLimitation] = field(default_factory=list, init=False)
    _started_at: float = field(init=False)

    def __post_init__(self) -> None:
        self._started_at = self.clock()

    def append(self, depth: int, references: Sequence[str] | Iterable[str]) -> bool:
        """Append a bounded immutable wave, returning whether it was retained."""
        if self._deadline_exhausted():
            return False
        if len(self._waves) >= max(0, self.max_waves):
            self._record_limit("frontier_waves", len(self._waves) + 1, self.max_waves)
            return False

        remaining = max(0, self.max_references - self._queued_references)
        retained: list[str] = []
        for reference in references:
            if self._deadline_exhausted():
                return False
            self._references_observed += 1
            if len(retained) >= remaining:
                self._record_limit(
                    "frontier_references",
                    self._queued_references + len(retained) + 1,
                    self.max_references,
                )
                break
            if isinstance(reference, str):
                retained.append(reference)

        if not retained:
            return False
        wave = TransitiveFrontierWave(depth=max(1, depth), references=tuple(retained))
        self._waves.append(wave)
        self._queued_references += len(retained)
        return True

    def popleft(self) -> TransitiveFrontierWave | None:
        """Pop one wave while releasing its reference allowance."""
        if not self._waves:
            return None
        if self._deadline_exhausted():
            self._waves.clear()
            self._queued_references = 0
            return None
        wave = self._waves.popleft()
        self._queued_references -= len(wave.references)
        return wave

    @property
    def limitations(self) -> tuple[TransitiveResourceLimitation, ...]:
        return tuple(self._limitations)

    @property
    def references_observed(self) -> int:
        return self._references_observed

    def __bool__(self) -> bool:
        return bool(self._waves)

    def __len__(self) -> int:
        return len(self._waves)

    def _record_limit(self, resource: str, observed: int | float, limit: int | float) -> None:
        if any(item.resource == resource for item in self._limitations):
            return
        self._limitations.append(
            TransitiveResourceLimitation(resource=resource, observed=observed, limit=limit)
        )

    def _deadline_exhausted(self) -> bool:
        now = self.clock()
        if now < self.deadline:
            return False
        self._record_limit(
            "runtime",
            max(0.0, now - self._started_at),
            max(0.0, self.deadline - self._started_at),
        )
        return True


@dataclass(frozen=True, slots=True)
class _ReversePath:
    """Heap wrapper that keeps the lexicographically greatest selected path on top."""

    value: str

    def __lt__(self, other: _ReversePath) -> bool:
        return self.value > other.value


def canonicalize_source_identity(url: str) -> str:
    """Return canonical URL identity used for dedupe and visited-state control."""
    token = _clean_token(url).strip()
    if not token:
        raise ValueError(f"Unsupported URL: {url}")

    parsed = _parse_url(token)
    if parsed.scheme.lower() != "https":
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"

    path = _normalize_path(parsed.path or "/")
    path = path.removesuffix(".git")
    path = path.rstrip("/")
    return urlunparse(("https", netloc, path if path else "/", "", "", ""))


def report_safe_source_scope_key(source_path: str) -> str:
    """Return an opaque, relative POSIX key safe to surface in public reports."""
    digest = sha256()
    # Incremental encoding avoids a second unbounded allocation for synthetic
    # nested-cache keys while keeping the same stable digest as one-shot UTF-8.
    for offset in range(0, len(source_path), 4096):
        digest.update(source_path[offset : offset + 4096].encode("utf-8", errors="replace"))
    return f"transitive-reference-source/{digest.hexdigest()[:24]}"


def _append_limitation(
    limitations: list[TransitiveResourceLimitation],
    *,
    resource: str,
    observed: int | float,
    limit: int | float,
    source_scope: str | None = None,
) -> None:
    """Record a stable first limitation for a resource and source scope."""
    if any(item.resource == resource and item.source_scope == source_scope for item in limitations):
        return
    limitations.append(
        TransitiveResourceLimitation(
            resource=resource,
            observed=observed,
            limit=limit,
            source_scope=source_scope,
        )
    )


def _select_sorted_cache_paths(
    file_cache: Mapping[str, str],
    *,
    max_sources: int,
    deadline: float,
    clock: Callable[[], float],
) -> tuple[list[str], int, bool, bool]:
    """Select the smallest cache paths with bounded memory and deadline checks."""
    heap: list[_ReversePath] = []
    observed = 0
    limit = max(0, max_sources)
    for path, content in file_cache.items():
        if clock() >= deadline:
            # Returning no paths fails closed: a partial mapping walk cannot
            # prove which paths belong in the deterministic lexicographic set.
            return [], observed, observed > limit, True
        if not isinstance(path, str) or not isinstance(content, str):
            continue
        observed += 1
        wrapped = _ReversePath(path)
        if len(heap) < limit:
            heapq.heappush(heap, wrapped)
        elif heap and path < heap[0].value:
            heapq.heapreplace(heap, wrapped)
    return sorted(item.value for item in heap), observed, observed > limit, False


def _bounded_utf8_prefix(text: str, byte_limit: int) -> tuple[str, int, bool]:
    """Return a valid UTF-8 prefix without encoding attacker-controlled remainder."""
    limit = max(0, byte_limit)
    probe = text[: limit + 1]
    encoded = probe.encode("utf-8", errors="replace")
    overflow = len(text) > len(probe) or len(encoded) > limit
    examined = min(len(encoded), limit)
    bounded = encoded[:limit].decode("utf-8", errors="ignore")
    return bounded, examined, overflow


def extract_external_refs_with_metadata(
    file_cache: Mapping[str, str],
    *,
    limits: ExternalReferenceLimits | None = None,
    clock: Callable[[], float] = time.monotonic,
    deadline: float | None = None,
) -> ExternalReferenceExtractionResult:
    """Extract external references with deterministic input/work/output bounds."""
    limits = limits or ExternalReferenceLimits()
    started_at = clock()
    local_deadline = started_at + max(0.0, limits.max_seconds)
    effective_deadline = local_deadline if deadline is None else min(local_deadline, deadline)
    runtime_limit = max(0.0, effective_deadline - started_at)
    limitations: list[TransitiveResourceLimitation] = []

    paths, sources_observed, sources_limited, path_deadline_exhausted = _select_sorted_cache_paths(
        file_cache,
        max_sources=limits.max_sources,
        deadline=effective_deadline,
        clock=clock,
    )
    if sources_limited:
        _append_limitation(
            limitations,
            resource="sources",
            observed=min(sources_observed, max(0, limits.max_sources) + 1),
            limit=max(0, limits.max_sources),
        )
    if path_deadline_exhausted:
        _append_limitation(
            limitations,
            resource="runtime",
            observed=max(0.0, clock() - started_at),
            limit=runtime_limit,
        )

    references: list[str] = []
    records: list[ExternalReferenceRecord] = []
    accepted: set[str] = set()
    source_bytes_examined = 0
    source_bytes_observed = 0
    raw_candidates_observed = 0
    accepted_references_observed = 0
    output_records_observed = 0
    stop = path_deadline_exhausted

    for source_path in paths:
        if stop:
            break
        now = clock()
        if now >= effective_deadline:
            _append_limitation(
                limitations,
                resource="runtime",
                observed=max(0.0, now - started_at),
                limit=runtime_limit,
            )
            break

        content = file_cache.get(source_path)
        if not isinstance(content, str):
            continue
        remaining_bytes = max(0, limits.max_source_bytes - source_bytes_examined)
        bounded_content, examined, byte_overflow = _bounded_utf8_prefix(content, remaining_bytes)
        source_bytes_examined += examined
        source_bytes_observed = source_bytes_examined
        source_scope = report_safe_source_scope_key(source_path)
        if byte_overflow:
            source_bytes_observed = max(0, limits.max_source_bytes) + 1
            _append_limitation(
                limitations,
                resource="source_bytes",
                observed=source_bytes_observed,
                limit=max(0, limits.max_source_bytes),
                source_scope=source_scope,
            )

        for match in _EXTERNAL_REF_PATTERN.finditer(bounded_content):
            now = clock()
            if now >= effective_deadline:
                _append_limitation(
                    limitations,
                    resource="runtime",
                    observed=max(0.0, now - started_at),
                    limit=runtime_limit,
                    source_scope=source_scope,
                )
                stop = True
                break

            raw_candidates_observed += 1
            if raw_candidates_observed > max(0, limits.max_raw_candidates):
                _append_limitation(
                    limitations,
                    resource="raw_candidates",
                    observed=raw_candidates_observed,
                    limit=max(0, limits.max_raw_candidates),
                    source_scope=source_scope,
                )
                stop = True
                break

            try:
                identity = canonicalize_source_identity(match.group(0))
            except ValueError:
                continue
            if not _is_source_reference(identity):
                continue

            is_new_identity = identity not in accepted
            if is_new_identity:
                accepted_references_observed += 1
                if accepted_references_observed > max(0, limits.max_accepted_references):
                    _append_limitation(
                        limitations,
                        resource="accepted_references",
                        observed=accepted_references_observed,
                        limit=max(0, limits.max_accepted_references),
                        source_scope=source_scope,
                    )
                    stop = True
                    break

            output_records_observed += 1
            if output_records_observed > max(0, limits.max_output_records):
                _append_limitation(
                    limitations,
                    resource="output_records",
                    observed=output_records_observed,
                    limit=max(0, limits.max_output_records),
                    source_scope=source_scope,
                )
                stop = True
                break
            if is_new_identity:
                accepted.add(identity)
                references.append(identity)
            records.append(ExternalReferenceRecord(source_scope=source_scope, source_url=identity))

        if byte_overflow:
            # The shared byte allowance is exhausted even when no URL occurred
            # in the retained prefix; later cache paths must not be inspected.
            break

    runtime_seconds = max(0.0, clock() - started_at)
    if runtime_seconds >= runtime_limit and not any(
        limitation.resource == "runtime" for limitation in limitations
    ):
        _append_limitation(
            limitations,
            resource="runtime",
            observed=runtime_seconds,
            limit=runtime_limit,
        )

    return ExternalReferenceExtractionResult(
        references=references,
        records=records,
        complete=not limitations,
        limitations=tuple(limitations),
        sources_observed=sources_observed,
        sources_limit=max(0, limits.max_sources),
        source_bytes_examined=source_bytes_examined,
        source_bytes_observed=source_bytes_observed,
        source_bytes_limit=max(0, limits.max_source_bytes),
        raw_candidates_observed=raw_candidates_observed,
        raw_candidates_limit=max(0, limits.max_raw_candidates),
        accepted_references_observed=accepted_references_observed,
        accepted_references_limit=max(0, limits.max_accepted_references),
        output_records_observed=output_records_observed,
        output_records_limit=max(0, limits.max_output_records),
        runtime_seconds=runtime_seconds,
        runtime_seconds_limit=runtime_limit,
    )


def extract_external_refs(file_cache: Mapping[str, str]) -> list[str]:
    """Compatibility wrapper returning a safely bounded reference list."""
    return extract_external_refs_with_metadata(file_cache).references


def plan_transitive_targets(
    refs: Sequence[str],
    visited: set[str],
    current_depth: int,
    max_depth: int,
    allow_prefixes: tuple[str, ...],
    deny_prefixes: tuple[str, ...],
) -> list[str]:
    """Compatibility wrapper returning a safely bounded next target wave."""
    return plan_transitive_targets_with_metadata(
        refs=refs,
        visited=visited,
        current_depth=current_depth,
        max_depth=max_depth,
        allow_prefixes=allow_prefixes,
        deny_prefixes=deny_prefixes,
    ).targets


def plan_transitive_targets_with_metadata(
    refs: Sequence[str],
    visited: set[str],
    current_depth: int,
    max_depth: int,
    allow_prefixes: tuple[str, ...],
    deny_prefixes: tuple[str, ...],
    *,
    limits: TransitivePlanLimits | None = None,
    clock: Callable[[], float] = time.monotonic,
    deadline: float | None = None,
) -> TransitiveTargetPlan:
    """Plan the next transitive wave with input/output and absolute-time bounds."""
    limits = limits or TransitivePlanLimits()
    started_at = clock()
    local_deadline = started_at + max(0.0, limits.max_seconds)
    effective_deadline = local_deadline if deadline is None else min(local_deadline, deadline)
    runtime_limit = max(0.0, effective_deadline - started_at)
    limitations: list[TransitiveResourceLimitation] = []
    input_references_observed = 0
    targets_observed = 0
    prefixes_observed = 0

    if current_depth > max_depth or max_depth <= 0:
        return TransitiveTargetPlan(
            targets=[],
            complete=True,
            limitations=(),
            input_references_observed=0,
            input_references_limit=max(0, limits.max_input_references),
            targets_observed=0,
            targets_limit=max(0, limits.max_targets),
            prefixes_observed=0,
            prefixes_limit=max(0, limits.max_prefixes),
            runtime_seconds=max(0.0, clock() - started_at),
            runtime_seconds_limit=runtime_limit,
        )
    if current_depth < 1:
        current_depth = 1

    normalized_prefix_groups: list[tuple[str, ...]] = []
    for prefixes in (allow_prefixes, deny_prefixes):
        normalized: list[str] = []
        for prefix in prefixes:
            now = clock()
            if now >= effective_deadline:
                _append_limitation(
                    limitations,
                    resource="runtime",
                    observed=max(0.0, now - started_at),
                    limit=runtime_limit,
                )
                break
            prefixes_observed += 1
            if prefixes_observed > max(0, limits.max_prefixes):
                _append_limitation(
                    limitations,
                    resource="prefixes",
                    observed=prefixes_observed,
                    limit=max(0, limits.max_prefixes),
                )
                break
            normalized.append(_normalize_prefix(prefix))
        normalized_prefix_groups.append(tuple(normalized))
        if limitations:
            break
    while len(normalized_prefix_groups) < 2:
        normalized_prefix_groups.append(())
    normalized_allow_prefixes = normalized_prefix_groups[0]
    normalized_deny_prefixes = normalized_prefix_groups[1]
    if limitations:
        # A partially normalized deny list could authorize a target that the
        # caller intended to block. An incomplete prefix plan therefore has no
        # approved targets rather than a permissive partial result.
        runtime_seconds = max(0.0, clock() - started_at)
        return TransitiveTargetPlan(
            targets=[],
            complete=False,
            limitations=tuple(limitations),
            input_references_observed=0,
            input_references_limit=max(0, limits.max_input_references),
            targets_observed=0,
            targets_limit=max(0, limits.max_targets),
            prefixes_observed=prefixes_observed,
            prefixes_limit=max(0, limits.max_prefixes),
            runtime_seconds=runtime_seconds,
            runtime_seconds_limit=runtime_limit,
        )

    targets: list[str] = []
    for ref in refs:
        now = clock()
        if now >= effective_deadline:
            _append_limitation(
                limitations,
                resource="runtime",
                observed=max(0.0, now - started_at),
                limit=runtime_limit,
            )
            break
        input_references_observed += 1
        if input_references_observed > max(0, limits.max_input_references):
            _append_limitation(
                limitations,
                resource="input_references",
                observed=input_references_observed,
                limit=max(0, limits.max_input_references),
            )
            break
        try:
            identity = canonicalize_source_identity(ref)
        except ValueError:
            continue
        if not _is_source_reference(identity):
            continue
        if identity in visited:
            continue
        if normalized_allow_prefixes and not _matches_any_prefix(
            identity, normalized_allow_prefixes
        ):
            continue
        if normalized_deny_prefixes and _matches_any_prefix(identity, normalized_deny_prefixes):
            continue
        targets_observed += 1
        if targets_observed > max(0, limits.max_targets):
            _append_limitation(
                limitations,
                resource="output_records",
                observed=targets_observed,
                limit=max(0, limits.max_targets),
            )
            break
        visited.add(identity)
        targets.append(identity)

    runtime_seconds = max(0.0, clock() - started_at)
    if runtime_seconds >= runtime_limit and not any(
        limitation.resource == "runtime" for limitation in limitations
    ):
        _append_limitation(
            limitations,
            resource="runtime",
            observed=runtime_seconds,
            limit=runtime_limit,
        )
    return TransitiveTargetPlan(
        targets=targets,
        complete=not limitations,
        limitations=tuple(limitations),
        input_references_observed=input_references_observed,
        input_references_limit=max(0, limits.max_input_references),
        targets_observed=targets_observed,
        targets_limit=max(0, limits.max_targets),
        prefixes_observed=prefixes_observed,
        prefixes_limit=max(0, limits.max_prefixes),
        runtime_seconds=runtime_seconds,
        runtime_seconds_limit=runtime_limit,
    )


def normalize_prefixes(
    allow_prefixes: tuple[str, ...] | list[str] | None,
    deny_prefixes: tuple[str, ...] | list[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Canonicalize CLI prefixes before any root scan starts."""
    return (
        tuple(_normalize_prefix(prefix) for prefix in (allow_prefixes or ())),
        tuple(_normalize_prefix(prefix) for prefix in (deny_prefixes or ())),
    )


def _parse_url(url: str) -> ParseResult:
    token = _clean_token(url)
    if token.startswith("git@"):
        return _parse_git_ssh_url(token)
    parsed = urlparse(token)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Unsupported URL: {url}")
    return parsed


def _parse_git_ssh_url(url: str) -> ParseResult:
    match = re.fullmatch(r"git@([^:]+):(.+)", url)
    if not match:
        raise ValueError(f"Unsupported git URL format: {url}")
    host = match.group(1).strip()
    path = match.group(2).strip().lstrip("/")
    return urlparse(f"https://{host}/{path}")


def _clean_token(token: str) -> str:
    cleaned = token.strip()
    while cleaned and cleaned[0] in _LEADING_PUNCTUATION:
        cleaned = cleaned[1:]
    while cleaned and cleaned[-1] in _TRAILING_PUNCTUATION:
        cleaned = cleaned[:-1]
    return cleaned.strip()


def _decode_unreserved(text: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        decoded = chr(int(match.group(0)[1:], 16))
        if decoded in _UNRESERVED_CHARACTERS:
            return decoded
        return match.group(0).upper()

    return _PERCENT_ENCODED_RE.sub(replacer, text)


def _normalize_path(path: str) -> str:
    normalized = posixpath.normpath(_decode_unreserved(path) or "/")
    if normalized == ".":
        return "/"
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    return normalized


def _normalize_prefix(prefix: str) -> str:
    if not prefix:
        return ""
    return canonicalize_source_identity(prefix)


def _matches_any_prefix(url: str, prefixes: tuple[str, ...]) -> bool:
    return any(_matches_prefix(url, prefix) for prefix in prefixes if prefix)


def _matches_prefix(url: str, prefix: str) -> bool:
    if url == prefix:
        return True
    if prefix.endswith("/"):
        return url.startswith(prefix)
    return url.startswith(prefix + "/")


def _is_source_reference(identity: str) -> bool:
    parsed = urlparse(identity)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return False
    if host in _EXCLUDED_HOSTS:
        return False
    if not _is_allowed_host(host):
        return False

    lower_path = unquote(parsed.path).lower()
    if _has_excluded_path_marker(lower_path):
        return False

    if _looks_like_git_reference(host, lower_path):
        return True
    return _looks_like_file_reference(host, lower_path, parsed.path)


def _has_excluded_path_marker(path: str) -> bool:
    if path.endswith(".svg"):
        return True
    segments = [segment for segment in path.split("/") if segment]
    if len(segments) < 3:
        return False
    ui_segment = segments[2]
    return ui_segment in {
        "actions",
        "badge",
        "badges",
        "blob",
        "checks",
        "ci",
        "issues",
        "pull",
        "pulls",
        "tree",
        "wiki",
        "workflows",
    }


def _looks_like_git_reference(host: str, path: str) -> bool:
    if not _host_in_allowed_git_hosts(host):
        return False
    if not path or path == "/":
        return False
    if path.startswith("/raw/"):
        return False
    if path.startswith("/blob/"):
        return False
    if "/tree/" in path:
        return False
    if "/archive/" in path:
        return False

    segments = [segment for segment in path.split("/") if segment]
    if len(segments) < 2:
        return False
    if len(segments) >= 3 and segments[2] == "actions":
        return False

    return True


def _looks_like_file_reference(host: str, lower_path: str, raw_path: str) -> bool:
    if not _is_allowed_host(host):
        return False
    if raw_path.endswith("/"):
        return False
    extension = _split_extension(lower_path)
    if not extension:
        return False
    return extension in _SUPPORTED_FILE_EXTENSIONS


def _is_allowed_host(host: str) -> bool:
    return (
        host in ALLOWED_GIT_HOSTS
        or host in {f"www.{entry}" for entry in ALLOWED_GIT_HOSTS}
        or host in ALLOWED_DOWNLOAD_HOSTS
        or host in {f"www.{entry}" for entry in ALLOWED_DOWNLOAD_HOSTS}
    )


def _host_in_allowed_git_hosts(host: str) -> bool:
    return host in ALLOWED_GIT_HOSTS or host in {f"www.{entry}" for entry in ALLOWED_GIT_HOSTS}


def _split_extension(path: str) -> str:
    return (
        "." + path.rsplit("/", 1)[-1].rsplit(".", 1)[-1] if "." in path.rsplit("/", 1)[-1] else ""
    )
