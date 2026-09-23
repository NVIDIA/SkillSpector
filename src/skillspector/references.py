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

from skillspector.artifacts import ArtifactDisposition, BundleReference, ReferenceKind

MAX_REFERENCE_SOURCE_BYTES = 1_000_000
MAX_RAW_REFERENCE_CANDIDATES = 4096
MAX_ACCEPTED_REFERENCES = 256
MAX_REFERENCE_RECORDS = 1024
MAX_REFERENCE_RUNTIME_SECONDS = 2.0
_MAX_EVIDENCE = 160
_MAX_MARKDOWN_DESTINATION_CHARS = 512
_MARKDOWN_REFERENCE_START = re.compile(
    r"(?:!\[[^\]\n]{0,200}\]|(?<!!)\[[^\]\n]{1,200}\])\("
    r"|^[ \t]{0,3}\[[^\]\n]{1,200}\]:[ \t]*"
)
_MARKDOWN_TITLE = r"""(?:"(?:\\.|[^"\\\r\n])*"|'(?:\\.|[^'\\\r\n])*'|\((?:\\.|[^)\\\r\n])*\))"""
_MARKDOWN_INLINE_END = re.compile(r"[ \t]*(?:" + _MARKDOWN_TITLE + r")?[ \t]*\)")
_MARKDOWN_DEFINITION_END = re.compile(r"[ \t]*(?:" + _MARKDOWN_TITLE + r")?[ \t]*(?:\r?\n)?\Z")
_MARKDOWN_STRUCTURAL_ESCAPE = re.compile(r"\\([\\()<>])")
_PASSIVE_IMAGE_DESTINATION = re.compile(
    r"[^\s\\()\[\]<>]+(?:[ \t]+(?:\"[^\"\\\r\n()]*\"|'[^'\\\r\n()]*'))?"
)
_MARKDOWN_FENCE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})([^\r\n]*)$")
_MARKDOWN_CONTAINER_PREFIX = re.compile(
    r"[ ]{0,3}(?:(>)[ ]?|(?:[-+*]|\d{1,9}[.)])(?:[ ]{1,4}(?![ ])|[ ](?=[ ]{4})))"
)
_HTML_CONTEXT_START = re.compile(
    r"<!--|<\?|<!\[CDATA\[|<![A-Z]|</?([A-Za-z][A-Za-z0-9-]*)(?=[\s/>])",
    re.IGNORECASE,
)
_QUOTED_OR_CODE_PATH = re.compile(
    r"(?:`|'|\")((?:\./)?(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,12})(?:`|'|\")"
)
# An interpreter command is an explicit path signal, but only its immediate
# script operand is eligible.  Searching the rest of the code span would turn
# strings passed to ``-c``/``-e`` and option values into fabricated references.
_INLINE_CODE_COMMAND_PATH = re.compile(
    r"`(?:python(?:\d+(?:\.\d+)*)?|py|node|deno|bun|bash|sh|zsh|fish|ruby|perl|php|"
    r"pwsh|powershell)\b[ \t]+"
    r"((?:\./)?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+)(?![\w/.-])"
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
    r"(?<![\w:/.-])((?:\./(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+|"
    r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,12}))(?![\w/.-])"
)


def _evidence(cleaned_line: str, column: int) -> str:
    """Return a bounded one-line evidence preview."""
    if len(cleaned_line) <= _MAX_EVIDENCE:
        return cleaned_line
    start = max(0, min(column - 1, len(cleaned_line)) - _MAX_EVIDENCE // 2)
    return cleaned_line[start : start + _MAX_EVIDENCE]


@dataclass(frozen=True)
class _ReferenceCandidate:
    raw: str
    start: int
    end: int
    kind: ReferenceKind
    syntax_start: int
    passive_image: bool = False


def _markdown_candidates(
    line: str, *, deadline: float, clock: Callable[[], float], limitations: set[str]
) -> Iterator[_ReferenceCandidate]:
    """Read bounded destinations without splitting spaces or balanced parentheses.

    Explicit reference definitions use the same destination grammar as inline
    links. Recognizing them separately keeps slash-separated prose excluded.
    """
    consumed_until = 0
    for opening in _MARKDOWN_REFERENCE_START.finditer(line):
        # Malformed openings may never yield a candidate. Bound their work too.
        if clock() >= deadline:
            return
        if opening.start() < consumed_until:
            continue
        start = opening.end()
        while start < len(line) and line[start] in " \t":
            start += 1
        if start >= len(line):
            continue
        limit = min(len(line), start + _MAX_MARKDOWN_DESTINATION_CHARS + 2)
        end = start
        if line[start] == "<":
            start += 1
            end = start
            while end < limit and line[end] not in "<>\r\n":
                if line[end] == "\\" and end + 1 < limit:
                    end += 1
                end += 1
            if end - start > _MAX_MARKDOWN_DESTINATION_CHARS:
                limitations.add("markdown_destination")
                continue
            if end >= limit or line[end] != ">":
                continue
            raw = line[start:end]
            destination_end = end + 1
        else:
            depth = 0
            while end < limit and not line[end].isspace():
                char = line[end]
                if char == "\\" and end + 1 < limit:
                    end += 2
                    continue
                if char == "(":
                    depth += 1
                elif char == ")":
                    if depth == 0:
                        break
                    depth -= 1
                elif char in "<>":
                    break
                end += 1
            if end - start > _MAX_MARKDOWN_DESTINATION_CHARS:
                limitations.add("markdown_destination")
                continue
            if depth or end == start:
                continue
            raw = line[start:end]
            destination_end = end
        if opening.group().endswith("("):
            ending = _MARKDOWN_INLINE_END.match(
                line, destination_end, min(len(line), destination_end + 512)
            )
            if ending is None:
                if len(line) - destination_end > 512:
                    limitations.add("markdown_title")
                continue
        else:
            if len(line) - destination_end > 512:
                limitations.add("markdown_title")
                continue
            ending = _MARKDOWN_DEFINITION_END.fullmatch(line, destination_end)
            # A destination can only be followed by a title, not arbitrary prose.
            if ending is None:
                continue
        consumed_until = ending.end()
        is_image = opening.group().startswith("![")
        label = line[opening.start() + 2 : opening.end() - 2] if is_image else ""
        # Retain main's conservative passive-image grammar even when the
        # destination parser can recover a more complex literal filename.
        passive_image = (
            is_image
            and not any(char in label for char in "[]\\")
            and bool(_PASSIVE_IMAGE_DESTINATION.fullmatch(line[opening.end() : consumed_until - 1]))
        )
        yield _ReferenceCandidate(
            _MARKDOWN_STRUCTURAL_ESCAPE.sub(r"\1", raw),
            start,
            consumed_until,
            ReferenceKind.MARKDOWN_IMAGE if is_image else ReferenceKind.MARKDOWN_LINK,
            opening.start(),
            passive_image,
        )


def _pattern_candidates(
    pattern: re.Pattern[str], line: str, kind: ReferenceKind
) -> Iterator[_ReferenceCandidate]:
    for match in pattern.finditer(line):
        yield _ReferenceCandidate(
            match.group(1), match.start(1), match.end(1), kind, match.start(1)
        )


def _advance_inline_code_state(
    line: str,
    end: int,
    cursor: int,
    delimiter_length: int | None,
) -> tuple[int, int | None]:
    """Advance a bounded CommonMark-style backtick-span state to ``end``."""
    while cursor < end:
        if line[cursor] != "`":
            cursor += 1
            continue
        preceding_backslashes = 0
        probe = cursor - 1
        while probe >= 0 and line[probe] == "\\":
            preceding_backslashes += 1
            probe -= 1
        run_end = cursor + 1
        while run_end < end and line[run_end] == "`":
            run_end += 1
        if preceding_backslashes % 2 == 0:
            run_length = run_end - cursor
            if delimiter_length is None:
                delimiter_length = run_length
            elif delimiter_length == run_length:
                delimiter_length = None
        cursor = run_end
    return cursor, delimiter_length


def _is_escaped_marker(line: str, offset: int) -> bool:
    """Return whether the character at ``offset`` has an odd backslash escape."""
    backslashes = 0
    offset -= 1
    while offset >= 0 and line[offset] == "\\":
        backslashes += 1
        offset -= 1
    return backslashes % 2 == 1


def _markdown_block_view(line: str) -> tuple[str, int, bool, int]:
    """Expose fences inside containers without claiming to parse their rendering."""
    expanded = line.expandtabs(4)
    cursor = 0
    quote_depth = 0
    has_list = False
    # More than four spaces after a list marker are content indentation;
    # retain that indentation so a code block cannot become a passive image.
    while match := _MARKDOWN_CONTAINER_PREFIX.match(expanded, cursor):
        quote_depth += int(match.group(1) is not None)
        has_list |= match.group(1) is None
        cursor = match.end()
    return expanded[cursor:], quote_depth, has_list, cursor


def _html_context_end(match: re.Match[str]) -> re.Pattern[str] | None:
    """Recognize literal terminators; other HTML contexts end at a blank line."""
    marker = match.group(0)
    if marker == "<!--":
        return re.compile(r"-->")
    if marker == "<?":
        return re.compile(r"\?>")
    if marker.upper() == "<![CDATA[":
        return re.compile(r"\]\]>")
    if marker.startswith("<!"):
        return re.compile(r">")
    tag = (match.group(1) or "").lower()
    if tag in {"pre", "script", "style", "textarea"}:
        return re.compile(rf"</{tag}\s*>", re.IGNORECASE)
    return None


def _candidate_strings(
    text: str,
    *,
    deadline: float,
    clock: Callable[[], float],
) -> tuple[list[tuple[str, int, int, str, ReferenceKind]], tuple[str, ...]]:
    """Extract path-like strings without materializing all matches or lines.

    Each candidate iterator contributes at most one pending match to a small
    merge heap.  This preserves source ordering while ensuring a dense,
    attacker-controlled line cannot be fully enumerated and sorted before the
    candidate and time ceilings are enforced.
    """
    candidates: list[tuple[str, int, int, str, ReferenceKind]] = []
    limitations: set[str] = set()
    seen: set[tuple[int, int, str]] = set()
    active_fence: tuple[str, int, int, int] | None = None
    in_html = False
    html_end: re.Pattern[str] | None = None
    inline_code_delimiter: int | None = None
    for line_number, line in enumerate(StringIO(text), 1):
        if clock() >= deadline:
            return candidates, ("runtime",)
        stripped_line = line.rstrip("\r\n")
        block_line, quote_depth, has_list, prefix_width = _markdown_block_view(stripped_line)
        fence_match = _MARKDOWN_FENCE.fullmatch(block_line)
        line_in_fence = active_fence is not None
        closes_fence = False
        if active_fence is not None and fence_match is not None:
            marker = fence_match.group(1)
            closes_fence = (
                marker[0] == active_fence[0]
                and len(marker) >= active_fence[1]
                and quote_depth == active_fence[2]
                and prefix_width + fence_match.start(1) >= active_fence[3]
                and not has_list
                and not fence_match.group(2).strip(" \t")
            )
        elif active_fence is None and fence_match is not None and not in_html:
            marker = fence_match.group(1)
            required_indent = prefix_width + fence_match.start(1) if has_list else 0
            active_fence = (marker[0], len(marker), quote_depth, required_indent)
            line_in_fence = True
        line_is_indented_code = stripped_line.expandtabs(4).startswith(
            "    "
        ) or block_line.startswith("    ")
        if in_html and html_end is None and not block_line.strip():
            in_html = False
        line_in_html = in_html
        html_search_start = 0
        if not line_in_fence and not line_is_indented_code:
            while True:
                if clock() >= deadline:
                    return candidates, ("runtime",)
                if not in_html:
                    html_start = _HTML_CONTEXT_START.search(block_line, html_search_start)
                    if html_start is None:
                        break
                    in_html = True
                    line_in_html = True
                    html_end = _html_context_end(html_start)
                    html_search_start = html_start.end()
                if html_end is None:
                    break
                html_close = html_end.search(block_line, html_search_start)
                if html_close is None:
                    break
                html_search_start = html_close.end()
                in_html = False
                html_end = None
        cleaned_line = " ".join(line.strip().split())
        iterators = [
            _markdown_candidates(line, deadline=deadline, clock=clock, limitations=limitations),
            _pattern_candidates(_INLINE_CODE_COMMAND_PATH, line, ReferenceKind.INLINE_COMMAND),
            _pattern_candidates(_QUOTED_OR_CODE_PATH, line, ReferenceKind.QUOTED_OR_CODE),
            _pattern_candidates(_PLAIN_RELATIVE_PATH, line, ReferenceKind.PLAIN_PATH),
        ]
        pending: list[tuple[int, int, _ReferenceCandidate]] = []
        image_label_spans: list[tuple[int, int, str]] = []
        markdown_destination_span: tuple[int, int] | None = None
        inline_code_cursor = 0
        for pattern_index, iterator in enumerate(iterators):
            match = next(iterator, None)
            if match is not None:
                heapq.heappush(pending, (match.syntax_start, pattern_index, match))
            if clock() >= deadline:
                return candidates, ("runtime",)
        while pending:
            if clock() >= deadline:
                return candidates, ("runtime",)
            sort_start, pattern_index, match = heapq.heappop(pending)
            reference_kind = match.kind
            if not line_in_fence and not line_is_indented_code and not line_in_html:
                inline_code_cursor, inline_code_delimiter = _advance_inline_code_state(
                    line, sort_start, inline_code_cursor, inline_code_delimiter
                )
            if reference_kind is ReferenceKind.MARKDOWN_IMAGE:
                if (
                    line_in_fence
                    or line_is_indented_code
                    or line_in_html
                    or inline_code_delimiter is not None
                    or not match.passive_image
                ):
                    reference_kind = ReferenceKind.QUOTED_OR_CODE
                elif _is_escaped_marker(line, match.syntax_start):
                    reference_kind = ReferenceKind.PLAIN_PATH
            raw = match.raw
            if reference_kind is ReferenceKind.MARKDOWN_IMAGE:
                # Same-target image alt text is redundant; different visible
                # targets and command operands still require their own records.
                image_label_spans.append((match.syntax_start, match.start, raw))
            redundant_image_label = reference_kind is ReferenceKind.PLAIN_PATH and any(
                start <= match.start < end and raw == destination
                for start, end, destination in image_label_spans
            )
            # Markdown candidates sort at their opening marker so their labels
            # can be classified. Suppress only destination/title text, not an
            # independent artifact named in a visible link or image label.
            inside_destination = (
                pattern_index != 0
                and markdown_destination_span is not None
                and markdown_destination_span[0] <= match.start < markdown_destination_span[1]
            )
            if pattern_index == 0:
                markdown_destination_span = (match.start, match.end)
            if not inside_destination and not redundant_image_label:
                key = (line_number, match.start, raw)
                if key not in seen:
                    seen.add(key)
                    candidates.append(
                        (
                            raw,
                            line_number,
                            match.start + 1,
                            _evidence(cleaned_line, match.start + 1),
                            reference_kind,
                        )
                    )
                    if len(candidates) >= MAX_RAW_REFERENCE_CANDIDATES:
                        return candidates, ("raw_candidates",)
            next_match = next(iterators[pattern_index], None)
            if next_match is not None:
                heapq.heappush(pending, (next_match.syntax_start, pattern_index, next_match))
        if not line_in_fence and not line_is_indented_code and not line_in_html:
            _, inline_code_delimiter = _advance_inline_code_state(
                line,
                len(line),
                inline_code_cursor,
                inline_code_delimiter,
            )
        if closes_fence:
            active_fence = None
    return candidates, tuple(sorted(limitations))


def _normalize_candidate(raw: str, source_path: str) -> str | None:
    """Return a contained relative POSIX candidate, or None when unsupported."""
    raw = raw.strip()
    split = urlsplit(raw)
    if split.scheme or split.netloc or raw.startswith(("/", "\\", "#")):
        return None
    # Split URI syntax before decoding so %23/%3F remain filename characters.
    # Decode exactly once, then apply containment checks to the decoded path.
    path_part = unquote(split.path).replace("\\", "/")
    if not path_part or path_part.startswith("/"):
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
    for raw, line, column, evidence, reference_kind in candidates:
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
            elif "/" not in unquote(raw).replace("\\", "/"):
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
                "reference_kind": reference_kind,
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
