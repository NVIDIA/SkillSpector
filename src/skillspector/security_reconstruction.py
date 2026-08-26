# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, non-executing reconstruction of explicitly declared text markers."""

from __future__ import annotations

import re
from array import array
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from io import StringIO
from typing import Final

from skillspector.artifacts import SecurityTextView

MAX_MARKER_LENGTH: Final = 16
MAX_MARKER_SCOPE_CHARS: Final = 768
MAX_MARKER_LOOKAHEAD_CHARS: Final = 8192
MAX_PAYLOAD_CHARS: Final = 700
MAX_ACTIVE_DIRECTIVES: Final = 8
MAX_MARKER_REMOVALS: Final = 64
MAX_NEGATION_PREFIX_CHARS: Final = 80

_REMOVAL_VERBS = (
    r"remove|removing|strip|stripping|delete|deleting|drop|dropping|"
    r"omit|omitting|erase|erasing|ignore|ignoring"
)
_DIRECTIVE_PREFILTER_RE: Final = re.compile(
    rf"\b(?:{_REMOVAL_VERBS})\b",
    re.IGNORECASE,
)
_QUOTED_DIRECTIVE_START_RE: Final = re.compile(
    rf"\b(?:{_REMOVAL_VERBS})\b[^.!?\n]{{0,80}}?(?P<quote>['\"`])",
    re.IGNORECASE,
)
_ENCODED_DIRECTIVE_START_RE: Final = re.compile(
    rf"\b(?:{_REMOVAL_VERBS})\b[^.!?\n]{{0,80}}?"
    r"(?P<quote>&(?:#x0*27|#0*39|apos|#x0*22|#0*34|quot);|\\(?:x27|u0027|x22|u0022))",
    re.IGNORECASE,
)
_TAG_DIRECTIVE_START_RE: Final = re.compile(
    rf"\b(?:{_REMOVAL_VERBS})\b[ \t]+(?:(?:the|this)[ \t]+)?(?P<open><)",
    re.IGNORECASE,
)
_ENCODED_TAG_DIRECTIVE_START_RE: Final = re.compile(
    rf"\b(?:{_REMOVAL_VERBS})\b[ \t]+(?:(?:the|this)[ \t]+)?"
    r"(?P<open>&(?:lt|#0*60|#x0*3c);)",
    re.IGNORECASE,
)
_ENCODED_TAG_END_RE: Final = re.compile(r"&(?:gt|#0*62|#x0*3e);", re.IGNORECASE)
_TAG_MARKER_RE: Final = re.compile(r"</?[A-Za-z][A-Za-z0-9:_-]*(?:[ \t]*/)?\>")
_ACTION_RE: Final = re.compile(
    r"\b(?:run|execute|invoke|issue|launch|perform|carry[ \t]+out)\b",
    re.IGNORECASE,
)
_NEGATED_ACTION_PREFIX_RE: Final = re.compile(
    r"(?:\bdo[ \t]+not|\bdon't|\bnever|\bavoid|\bmust[ \t]+not|\bnot)"
    r"(?:[ \t]+\w+){0,3}[ \t]*$",
    re.IGNORECASE,
)
_QUOTED_PAYLOAD_RE: Final = re.compile(
    rf"(?P<quote>['\"`])(?P<body>[^'\"`\n]{{0,{MAX_PAYLOAD_CHARS}}})(?P=quote)"
)
_INLINE_PREFIX_RE: Final = re.compile(
    r"[ \t]*(?:(?:the|this|following|next)[ \t]+)?"
    r"(?:(?:command|instruction|payload|request)\b[ \t]*)?(?::|=)?[ \t]*",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY_RE: Final = re.compile(r"\n|[!?](?=[ \t]|$)|\.(?=[ \t]|$)")
_FORWARD_PAYLOAD_REFERENCE_RE: Final = re.compile(
    r"\b(?:next|following|coming)[ \t]+"
    r"(?:command|instruction|prompt|payload|request)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DeclaredMarkerViewResult:
    """Deterministic payload views and whether any active form was unsupported."""

    views: tuple[SecurityTextView, ...]
    limited: bool


@dataclass(frozen=True)
class _Directive:
    marker: str
    start: int
    end: int
    is_tag: bool = False
    encoded: bool = False
    exhausted: bool = False


@dataclass(frozen=True)
class _Payload:
    start: int
    end: int


@dataclass(frozen=True)
class _ProjectionCandidate:
    directive: _Directive
    payload: _Payload
    positions: tuple[int, ...]


@dataclass(frozen=True)
class _DirectiveClassification:
    candidate: _ProjectionCandidate | None
    active: bool
    limited: bool


def _quoted_directives(
    text: str,
    check_runtime: Callable[[], None] | None,
    *,
    end_is_truncated: bool,
) -> Iterator[_Directive]:
    for match in _QUOTED_DIRECTIVE_START_RE.finditer(text):
        if check_runtime is not None:
            check_runtime()
        quote = match.group("quote")
        marker_start = match.end()
        marker_end_limit = min(len(text), marker_start + MAX_MARKER_LOOKAHEAD_CHARS)
        cursor = marker_start
        while cursor < marker_end_limit:
            character = text[cursor]
            if character == quote:
                if cursor > marker_start:
                    yield _Directive(text[marker_start:cursor], match.start(), cursor + 1)
                break
            if character.isspace() or character in "'\"`":
                break
            cursor += 1
        else:
            if marker_end_limit < len(text) or end_is_truncated:
                yield _Directive("", match.start(), marker_end_limit, exhausted=True)


def _tag_directives(
    text: str,
    check_runtime: Callable[[], None] | None,
    *,
    end_is_truncated: bool,
) -> Iterator[_Directive]:
    for match in _TAG_DIRECTIVE_START_RE.finditer(text):
        if check_runtime is not None:
            check_runtime()
        marker_start = match.start("open")
        marker_end_limit = min(len(text), marker_start + MAX_MARKER_LOOKAHEAD_CHARS)
        marker_end = text.find(">", marker_start + 1, marker_end_limit)
        if marker_end < 0:
            if marker_end_limit < len(text) or end_is_truncated:
                yield _Directive("", match.start(), marker_end_limit, is_tag=True, exhausted=True)
            continue
        marker = text[marker_start : marker_end + 1]
        if _TAG_MARKER_RE.fullmatch(marker) is not None:
            yield _Directive(marker, match.start(), marker_end + 1, is_tag=True)


def _encoded_directives(
    text: str,
    check_runtime: Callable[[], None] | None,
    *,
    end_is_truncated: bool,
) -> Iterator[_Directive]:
    folded_text = text.casefold()
    for match in _ENCODED_DIRECTIVE_START_RE.finditer(text):
        if check_runtime is not None:
            check_runtime()
        quote = match.group("quote")
        marker_start = match.end()
        marker_end_limit = min(len(text), marker_start + MAX_MARKER_LOOKAHEAD_CHARS)
        marker_end = folded_text.find(quote.casefold(), marker_start, marker_end_limit)
        if marker_end < 0:
            if marker_end_limit < len(text) or end_is_truncated:
                yield _Directive("", match.start(), marker_end_limit, encoded=True, exhausted=True)
            continue
        marker = text[marker_start:marker_end]
        if marker and not any(character.isspace() for character in marker):
            yield _Directive(
                marker,
                match.start(),
                marker_end + len(quote),
                encoded=True,
            )


def _encoded_tag_directives(
    text: str,
    check_runtime: Callable[[], None] | None,
    *,
    end_is_truncated: bool,
) -> Iterator[_Directive]:
    for match in _ENCODED_TAG_DIRECTIVE_START_RE.finditer(text):
        if check_runtime is not None:
            check_runtime()
        marker_start = match.start("open")
        marker_end_limit = min(len(text), marker_start + MAX_MARKER_LOOKAHEAD_CHARS)
        marker_end = _ENCODED_TAG_END_RE.search(text, match.end(), marker_end_limit)
        if marker_end is None:
            if marker_end_limit < len(text) or end_is_truncated:
                yield _Directive(
                    "",
                    match.start(),
                    marker_end_limit,
                    is_tag=True,
                    encoded=True,
                    exhausted=True,
                )
            continue
        yield _Directive(
            text[marker_start : marker_end.end()],
            match.start(),
            marker_end.end(),
            is_tag=True,
            encoded=True,
        )


def _directives(
    text: str,
    check_runtime: Callable[[], None] | None,
    *,
    end_is_truncated: bool,
) -> list[_Directive]:
    candidates = [
        *_quoted_directives(text, check_runtime, end_is_truncated=end_is_truncated),
        *_tag_directives(text, check_runtime, end_is_truncated=end_is_truncated),
        *_encoded_directives(text, check_runtime, end_is_truncated=end_is_truncated),
        *_encoded_tag_directives(text, check_runtime, end_is_truncated=end_is_truncated),
    ]
    candidates.sort(key=lambda item: (item.start, item.end, item.marker))
    unique: list[_Directive] = []
    seen: set[tuple[int, int, str]] = set()
    for candidate in candidates:
        key = (candidate.start, candidate.end, candidate.marker)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _bounded_sentence_end(
    text: str,
    start: int,
    maximum_end: int,
    *,
    end_is_truncated: bool,
) -> tuple[int, bool]:
    search_end = min(len(text), maximum_end + 1)
    boundary = _SENTENCE_BOUNDARY_RE.search(text, start, search_end)
    if boundary is not None and boundary.start() <= maximum_end:
        return boundary.start(), False
    exhausted = maximum_end < len(text) or end_is_truncated
    return maximum_end, exhausted


def _previous_sentence_boundary(text: str, end: int, maximum_span: int) -> int:
    start = max(0, end - maximum_span)
    last_end = start
    for boundary in _SENTENCE_BOUNDARY_RE.finditer(text, start, end):
        last_end = boundary.end()
    return last_end


def _verb_is_negated(text: str, verb_start: int, lower_bound: int) -> bool:
    prefix_start = max(lower_bound, verb_start - MAX_NEGATION_PREFIX_CHARS)
    return _NEGATED_ACTION_PREFIX_RE.search(text, prefix_start, verb_start) is not None


def _actions(text: str, start: int, end: int) -> list[re.Match[str]]:
    return [
        action
        for action in _ACTION_RE.finditer(text, start, end)
        if not _verb_is_negated(text, action.start(), start)
    ]


def _quoted_payloads(text: str, directive: _Directive, scope_end: int) -> list[_Payload]:
    actions = _actions(text, directive.end, scope_end)
    if not actions:
        return []
    payloads: set[tuple[int, int]] = set()
    for quoted in _QUOTED_PAYLOAD_RE.finditer(text, directive.end, scope_end):
        if quoted.end() < len(text) and text[quoted.end()].isalnum():
            continue
        body_start = quoted.start("body")
        body_end = quoted.end("body")
        if text.find(directive.marker, body_start, body_end) < 0:
            continue
        if any(action.end() <= quoted.start() for action in actions):
            payloads.add((body_start, body_end))
    return [_Payload(start, end) for start, end in sorted(payloads)]


def _inline_payloads(text: str, directive: _Directive, scope_end: int) -> list[_Payload]:
    payloads: set[tuple[int, int]] = set()
    for action in _actions(text, directive.end, scope_end):
        prefix = _INLINE_PREFIX_RE.match(text, action.end(), scope_end)
        start = prefix.end() if prefix is not None else action.end()
        if start >= scope_end or text[start] in "'\"`":
            continue
        end = min(scope_end, start + MAX_PAYLOAD_CHARS)
        if text.find(directive.marker, start, end) >= 0:
            payloads.add((start, end))
    return [_Payload(start, end) for start, end in sorted(payloads)]


def _has_active_unsupported_form(
    text: str,
    directive: _Directive,
    lookahead_end: int,
) -> bool:
    clause_start = _previous_sentence_boundary(text, directive.start, MAX_MARKER_LOOKAHEAD_CHARS)
    tail = text[directive.end : lookahead_end]
    marker_in_tail = directive.marker in tail
    marker_case_variant = not marker_in_tail and directive.marker.casefold() in tail.casefold()
    actions_after = _actions(text, directive.end, lookahead_end)
    if (marker_in_tail or marker_case_variant) and actions_after:
        return True

    actions_before = _actions(text, clause_start, directive.start)
    if actions_before and (marker_in_tail or marker_case_variant):
        return True

    marker_before = text.find(directive.marker, clause_start, directive.start) >= 0
    if marker_before and actions_after:
        return True

    if marker_in_tail:
        decoded_tail = tail.replace(directive.marker, "")
        if _actions(decoded_tail, 0, len(decoded_tail)):
            return True
    return False


def _positions_and_overlap(
    text: str, marker: str, start: int, end: int
) -> tuple[tuple[int, ...], bool]:
    positions: list[int] = []
    cursor = start
    last_end = start
    overlapping = False
    while cursor < end:
        found = text.find(marker, cursor, end)
        if found < 0:
            break
        if found < last_end:
            overlapping = True
        else:
            positions.append(found)
            last_end = found + len(marker)
        cursor = found + 1
    return tuple(positions), overlapping


def _paired_tag(marker: str) -> str | None:
    opening = re.fullmatch(r"<([A-Za-z][A-Za-z0-9:_-]*)>", marker)
    if opening is not None:
        return f"</{opening.group(1)}>"
    closing = re.fullmatch(r"</([A-Za-z][A-Za-z0-9:_-]*)>", marker)
    if closing is not None:
        return f"<{closing.group(1)}>"
    return None


def _retained_ranges(
    start: int, end: int, marker: str, positions: tuple[int, ...]
) -> Iterator[tuple[int, int]]:
    cursor = start
    for position in positions:
        if cursor < position:
            yield cursor, position
        cursor = position + len(marker)
    if cursor < end:
        yield cursor, end


def _project_payload(view: SecurityTextView, candidate: _ProjectionCandidate) -> SecurityTextView:
    output = StringIO()
    offsets = array("I")
    for start, end in _retained_ranges(
        candidate.payload.start,
        candidate.payload.end,
        candidate.directive.marker,
        candidate.positions,
    ):
        output.write(view.text[start:end])
        if view.source_offsets is None:
            offsets.extend(range(start, end))
        else:
            offsets.extend(view.source_offsets[start:end])
    return SecurityTextView(
        name=f"declared-marker-{view.name}",
        text=output.getvalue(),
        source_offsets=offsets,
    )


def _classify_directive(
    view: SecurityTextView,
    directive: _Directive,
    *,
    end_is_truncated: bool,
) -> _DirectiveClassification:
    if directive.exhausted:
        return _DirectiveClassification(None, False, True)

    scope_cap = min(len(view.text), directive.end + MAX_MARKER_SCOPE_CHARS)
    scope_end, _ = _bounded_sentence_end(
        view.text,
        directive.end,
        scope_cap,
        end_is_truncated=end_is_truncated,
    )
    lookahead_cap = min(len(view.text), directive.end + MAX_MARKER_LOOKAHEAD_CHARS)
    lookahead_end, lookahead_exhausted = _bounded_sentence_end(
        view.text,
        directive.end,
        lookahead_cap,
        end_is_truncated=end_is_truncated,
    )
    first_sentence_end = lookahead_end
    clause_start = _previous_sentence_boundary(
        view.text,
        directive.start,
        MAX_MARKER_LOOKAHEAD_CHARS,
    )
    directive_clause = view.text[clause_start:first_sentence_end]
    if (
        first_sentence_end < lookahead_cap
        and _FORWARD_PAYLOAD_REFERENCE_RE.search(directive_clause) is not None
    ):
        lookahead_end, continuation_exhausted = _bounded_sentence_end(
            view.text,
            first_sentence_end + 1,
            lookahead_cap,
            end_is_truncated=end_is_truncated,
        )
        lookahead_exhausted = lookahead_exhausted or continuation_exhausted
    payloads = _quoted_payloads(view.text, directive, scope_end)
    if not payloads:
        payloads = _inline_payloads(view.text, directive, scope_end)
    active = bool(payloads) or _has_active_unsupported_form(
        view.text,
        directive,
        lookahead_end,
    )
    if not active:
        return _DirectiveClassification(None, False, lookahead_exhausted)

    if directive.encoded or len(directive.marker) > MAX_MARKER_LENGTH or len(payloads) != 1:
        return _DirectiveClassification(None, True, True)

    payload = payloads[0]
    if len(directive.marker) == 1 and directive.marker.isalnum():
        return _DirectiveClassification(None, True, True)
    paired_tag = _paired_tag(directive.marker) if directive.is_tag else None
    if paired_tag is not None and paired_tag in view.text[payload.start : payload.end]:
        return _DirectiveClassification(None, True, True)
    positions, overlapping = _positions_and_overlap(
        view.text,
        directive.marker,
        payload.start,
        payload.end,
    )
    if overlapping or len(positions) > MAX_MARKER_REMOVALS:
        return _DirectiveClassification(None, True, True)
    candidate = _ProjectionCandidate(directive, payload, positions) if positions else None
    return _DirectiveClassification(candidate, True, lookahead_exhausted)


def _resolve_candidates(
    view: SecurityTextView,
    candidates: list[_ProjectionCandidate],
) -> tuple[tuple[SecurityTextView, ...], bool]:
    by_payload: dict[tuple[int, int], list[_ProjectionCandidate]] = {}
    for candidate in candidates:
        by_payload.setdefault((candidate.payload.start, candidate.payload.end), []).append(
            candidate
        )

    limited = False
    views: list[SecurityTextView] = []
    seen_views: set[tuple[str, int, int]] = set()
    for payload_key, payload_candidates in by_payload.items():
        if len({candidate.directive.marker for candidate in payload_candidates}) > 1:
            limited = True
            continue
        projected = _project_payload(view, payload_candidates[0])
        key = (projected.text, *payload_key)
        if projected.text and key not in seen_views:
            views.append(projected)
            seen_views.add(key)
    return tuple(views), limited


def build_declared_marker_views(
    view: SecurityTextView,
    *,
    check_runtime: Callable[[], None] | None = None,
    owned_source_start: int | None = None,
    owned_source_end: int | None = None,
) -> DeclaredMarkerViewResult:
    """Build one-pass payload views for explicit literal-removal instructions.

    The function never evaluates projected text. It accepts one unnegated,
    action-bound payload per directive; ambiguous or resource-bounded active
    forms set ``limited`` so the caller can fail closed without guessing.

    The optional ownership bounds are source coordinates in the current raw
    window. Directives before ``owned_source_start`` belong to the previous
    window; directives at or beyond exclusive ``owned_source_end`` belong to
    the next window.
    """
    if check_runtime is not None:
        check_runtime()
    if _DIRECTIVE_PREFILTER_RE.search(view.text) is None:
        return DeclaredMarkerViewResult((), False)

    active_directives = 0
    limited = False
    candidates: list[_ProjectionCandidate] = []
    end_is_truncated = owned_source_end is not None
    for directive in _directives(
        view.text,
        check_runtime,
        end_is_truncated=end_is_truncated,
    ):
        if check_runtime is not None:
            check_runtime()
        directive_source_start = view.source_offset(directive.start)
        if (owned_source_start is not None and directive_source_start < owned_source_start) or (
            owned_source_end is not None and directive_source_start >= owned_source_end
        ):
            continue

        clause_start = _previous_sentence_boundary(
            view.text,
            directive.start,
            MAX_MARKER_LOOKAHEAD_CHARS,
        )
        if _verb_is_negated(view.text, directive.start, clause_start):
            continue

        classification = _classify_directive(
            view,
            directive,
            end_is_truncated=end_is_truncated,
        )
        limited = limited or classification.limited
        if not classification.active:
            continue

        active_directives += 1
        if active_directives > MAX_ACTIVE_DIRECTIVES:
            limited = True
            break
        if classification.candidate is not None:
            candidates.append(classification.candidate)

    views, conflict_limited = _resolve_candidates(view, candidates)
    return DeclaredMarkerViewResult(views, limited or conflict_limited)
