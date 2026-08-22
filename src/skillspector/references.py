# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded canonical resolver for references made by the primary skill file."""

from __future__ import annotations

import heapq
import posixpath
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from io import StringIO
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from skillspector.artifacts import ArtifactDisposition, BundleReference

MAX_REFERENCE_SOURCE_BYTES = 1_000_000
MAX_RAW_REFERENCE_CANDIDATES = 4096
MAX_ACCEPTED_REFERENCES = 256
MAX_REFERENCE_RECORDS = 1024
MAX_REFERENCE_RUNTIME_SECONDS = 2.0
_MAX_EVIDENCE = 160
_MARKDOWN_DESTINATION = re.compile(r"\[[^\]\n]{1,200}\]\(([^)\n]{1,512})\)")
_QUOTED_OR_CODE_PATH = re.compile(
    r"(?:`|'|\")((?:\./)?(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,12})(?:`|'|\")"
)


@dataclass(frozen=True)
class ReferenceResolutionResult:
    """Bounded reference records plus explicit extraction accounting."""

    records: list[BundleReference]
    complete: bool
    limitations: tuple[str, ...]
    input_bytes_examined: int
    raw_candidates_considered: int
    accepted_references: int
    runtime_seconds: float
    runtime_seconds_limit: float


_PLAIN_RELATIVE_PATH = re.compile(
    r"(?<![\w:/.-])((?:\./)?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+(?:\.[A-Za-z0-9]{1,12})?)(?![\w/.-])"
)


def _evidence(cleaned_line: str, column: int) -> str:
    """Return a bounded one-line evidence preview."""
    if len(cleaned_line) <= _MAX_EVIDENCE:
        return cleaned_line
    start = max(0, min(column - 1, len(cleaned_line)) - _MAX_EVIDENCE // 2)
    return cleaned_line[start : start + _MAX_EVIDENCE]


def _candidate_strings(
    text: str,
    *,
    deadline: float,
    clock: Callable[[], float],
) -> tuple[list[tuple[str, int, int, str]], tuple[str, ...]]:
    """Extract path-like strings without materializing all matches or lines.

    Each regular expression contributes at most one pending match to a small
    merge heap.  This preserves source ordering while ensuring a dense,
    attacker-controlled line cannot be fully enumerated and sorted before the
    candidate and time ceilings are enforced.
    """
    candidates: list[tuple[str, int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()
    patterns = (_MARKDOWN_DESTINATION, _QUOTED_OR_CODE_PATH, _PLAIN_RELATIVE_PATH)
    for line_number, line in enumerate(StringIO(text), 1):
        if clock() >= deadline:
            return candidates, ("runtime",)
        cleaned_line = " ".join(line.strip().split())
        iterators: list[Iterator[re.Match[str]]] = [pattern.finditer(line) for pattern in patterns]
        pending: list[tuple[int, int, int, re.Match[str]]] = []
        for pattern_index, iterator in enumerate(iterators):
            match = next(iterator, None)
            if match is not None:
                heapq.heappush(
                    pending,
                    (match.start(1), match.end(1), pattern_index, match),
                )
            if clock() >= deadline:
                return candidates, ("runtime",)
        while pending:
            if clock() >= deadline:
                return candidates, ("runtime",)
            _, _, pattern_index, match = heapq.heappop(pending)
            raw = match.group(1).strip().split(maxsplit=1)[0]
            key = (line_number, match.start(1), raw)
            if key not in seen:
                seen.add(key)
                candidates.append(
                    (
                        raw,
                        line_number,
                        match.start(1) + 1,
                        _evidence(cleaned_line, match.start(1) + 1),
                    )
                )
                if len(candidates) >= MAX_RAW_REFERENCE_CANDIDATES:
                    return candidates, ("raw_candidates",)
            next_match = next(iterators[pattern_index], None)
            if next_match is not None:
                heapq.heappush(
                    pending,
                    (
                        next_match.start(1),
                        next_match.end(1),
                        pattern_index,
                        next_match,
                    ),
                )
    return candidates, ()


def _normalize_candidate(raw: str, source_path: str) -> str | None:
    """Return a contained relative POSIX candidate, or None when unsupported."""
    raw = unquote(raw.strip().strip("<>"))
    split = urlsplit(raw)
    if split.scheme or split.netloc or raw.startswith(("/", "\\", "#")):
        return None
    path_part = split.path.replace("\\", "/")
    if not path_part:
        return None
    if len(path_part) >= 2 and path_part[1] == ":":
        return None
    source_parent = PurePosixPath(source_path).parent.as_posix()
    joined = posixpath.normpath(posixpath.join(source_parent, path_part))
    if joined in {"", ".", ".."} or joined.startswith("../"):
        return None
    return joined.removeprefix("./")


def resolve_bundle_references_with_metadata(
    skill_dir: Path,
    *,
    source_path: str,
    source_text: str,
    known_paths: list[str],
    clock: Callable[[], float] = time.monotonic,
    deadline: float | None = None,
) -> ReferenceResolutionResult:
    """Resolve references with separate deterministic input/work/output bounds."""
    started_at = clock()
    local_deadline = started_at + MAX_REFERENCE_RUNTIME_SECONDS
    effective_deadline = local_deadline if deadline is None else min(local_deadline, deadline)
    runtime_limit = max(0.0, effective_deadline - started_at)
    if clock() >= effective_deadline:
        return ReferenceResolutionResult(
            records=[],
            complete=False,
            limitations=("runtime",),
            input_bytes_examined=0,
            raw_candidates_considered=0,
            accepted_references=0,
            runtime_seconds=max(0.0, clock() - started_at),
            runtime_seconds_limit=runtime_limit,
        )
    # UTF-8 always uses at least one byte per code point, so this character
    # prefix is sufficient to determine whether the byte limit was crossed
    # without first encoding an unbounded compatibility-wrapper input.
    source_prefix = source_text[: MAX_REFERENCE_SOURCE_BYTES + 1]
    encoded_prefix = source_prefix.encode("utf-8")
    input_limited = (
        len(source_text) > MAX_REFERENCE_SOURCE_BYTES
        or len(encoded_prefix) > MAX_REFERENCE_SOURCE_BYTES
    )
    bounded_source = encoded_prefix[:MAX_REFERENCE_SOURCE_BYTES].decode("utf-8", errors="ignore")
    input_bytes_examined = min(len(encoded_prefix), MAX_REFERENCE_SOURCE_BYTES)

    known = set(known_paths)
    basename_index: dict[str, list[str]] = {}
    for path in sorted(known):
        basename_index.setdefault(PurePosixPath(path).name, []).append(path)

    candidates, candidate_limitations = _candidate_strings(
        bounded_source,
        deadline=effective_deadline,
        clock=clock,
    )
    limitations = ["input_bytes"] if input_limited else []
    limitations.extend(candidate_limitations)
    records: list[BundleReference] = []
    accepted_keys: set[tuple[str, str]] = set()
    for raw, line, column, evidence in candidates:
        if clock() > effective_deadline:
            limitations.append("runtime")
            break
        target = _normalize_candidate(raw, source_path)
        status = "rejected"
        disposition = ArtifactDisposition.OUT_OF_SCOPE
        resolved_target: str | None = None
        if target is not None:
            # Resolution is intentionally confined to the caller's already
            # bounded discovery inventory. Re-probing the filesystem here
            # could reintroduce a path omitted by an artifact, depth, or
            # runtime limit and silently expand analyzer work past that bound.
            if target in known:
                resolved_target = target
                status = "resolved"
                disposition = ArtifactDisposition.ANALYZED
            elif "/" not in raw.replace("\\", "/"):
                matches = basename_index.get(PurePosixPath(target).name, [])
                if len(matches) == 1:
                    resolved_target = matches[0]
                    status = "resolved"
                    disposition = ArtifactDisposition.ANALYZED
                elif len(matches) > 1:
                    status = "ambiguous"
                    disposition = ArtifactDisposition.PARTIAL
                else:
                    status = "missing"
                    disposition = ArtifactDisposition.PARTIAL
            else:
                status = "missing"
                disposition = ArtifactDisposition.PARTIAL
        if status != "rejected":
            accepted_key = (status, resolved_target or target or raw)
            if accepted_key not in accepted_keys:
                if len(accepted_keys) >= MAX_ACCEPTED_REFERENCES:
                    limitations.append("accepted_references")
                    break
                accepted_keys.add(accepted_key)
        if len(records) >= MAX_REFERENCE_RECORDS:
            limitations.append("output_records")
            break
        records.append(
            {
                "source_path": source_path,
                "line": line,
                "column": column,
                "evidence": evidence,
                "target_path": resolved_target,
                "status": status,
                "disposition": disposition,
            }
        )
    stable_limitations = tuple(dict.fromkeys(limitations))
    runtime_seconds = max(0.0, clock() - started_at)
    if runtime_seconds >= runtime_limit and "runtime" not in stable_limitations:
        stable_limitations = (*stable_limitations, "runtime")
    return ReferenceResolutionResult(
        records=records,
        complete=not stable_limitations,
        limitations=stable_limitations,
        input_bytes_examined=input_bytes_examined,
        raw_candidates_considered=len(candidates),
        accepted_references=len(accepted_keys),
        runtime_seconds=runtime_seconds,
        runtime_seconds_limit=runtime_limit,
    )


def resolve_bundle_references(
    skill_dir: Path,
    *,
    source_path: str,
    source_text: str,
    known_paths: list[str],
    clock: Callable[[], float] = time.monotonic,
    deadline: float | None = None,
) -> list[BundleReference]:
    """Compatibility wrapper returning bounded reference records."""
    return resolve_bundle_references_with_metadata(
        skill_dir,
        source_path=source_path,
        source_text=source_text,
        known_paths=known_paths,
        clock=clock,
        deadline=deadline,
    ).records
