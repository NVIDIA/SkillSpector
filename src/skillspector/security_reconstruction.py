# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, non-executing reconstruction of explicitly declared text markers."""

from __future__ import annotations

import re
from array import array
from collections import Counter
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
MAX_DIRECTIVE_HEADER_CHARS: Final = 256
MAX_DECLARED_MARKER_RIGHT_CONTEXT_CHARS: Final = (
    MAX_DIRECTIVE_HEADER_CHARS + 2 * MAX_MARKER_LOOKAHEAD_CHARS
)
_MAX_DIRECTIVE_HEADER_SPACING_CHARS: Final = 8
_MAX_FALLBACK_PREFIX_CHARS: Final = MAX_DIRECTIVE_HEADER_CHARS
_MAX_ENTITY_ZERO_PADDING: Final = 8
_ENTITY_ZERO_PADDING = rf"0{{0,{_MAX_ENTITY_ZERO_PADDING}}}"
_FALLBACK_ENTITY_ZERO_PADDING = r"0{0,64}"

_TAKE_OUT_VERBS = (
    rf"take[ \t]{{0,{_MAX_DIRECTIVE_HEADER_SPACING_CHARS}}}out|"
    rf"taking[ \t]{{0,{_MAX_DIRECTIVE_HEADER_SPACING_CHARS}}}out"
)
_FALLBACK_TAKE_OUT_VERBS = (
    rf"take[ \t]{{0,{_MAX_FALLBACK_PREFIX_CHARS}}}out|"
    rf"taking[ \t]{{0,{_MAX_FALLBACK_PREFIX_CHARS}}}out"
)
_REMOVAL_VERBS = (
    r"remove|removing|strip|stripping|delete|deleting|drop|dropping|"
    r"omit|omitting|erase|erasing|ignore|ignoring|excise|excising|"
    rf"{_TAKE_OUT_VERBS}"
)
_FALLBACK_REMOVAL_VERBS = (
    r"remove|removing|strip|stripping|delete|deleting|drop|dropping|"
    r"omit|omitting|erase|erasing|ignore|ignoring|excise|excising|"
    rf"{_FALLBACK_TAKE_OUT_VERBS}"
)
_REPLACEMENT_VERBS = r"replace|replacing|substitute|substituting"
_DIRECTIVE_CUES = rf"(?:{_FALLBACK_REMOVAL_VERBS}|{_REPLACEMENT_VERBS})"
_DECLARED_MARKER_PREFIX = (
    rf"[ \t]{{1,{_MAX_DIRECTIVE_HEADER_SPACING_CHARS}}}"
    rf"(?:(?:(?:the|this|both|all|every|each|literal|declared|specified|"
    rf"following|next|invisible)[ \t]{{1,{_MAX_DIRECTIVE_HEADER_SPACING_CHARS}}})"
    rf"{{0,3}}(?:"
    rf"(?:markers?|tokens?|substrings?|strings?|text|characters?|occurrences?)"
    rf"(?:[ \t]{{1,{_MAX_DIRECTIVE_HEADER_SPACING_CHARS}}}of)?"
    rf"[ \t]{{1,{_MAX_DIRECTIVE_HEADER_SPACING_CHARS}}})?)?"
)
_QUOTE_OPEN_TO_CLOSE: Final = {
    "'": "'",
    '"': '"',
    "`": "`",
    "‘": "’",
    "“": "”",
}
_QUOTE_OPEN_CLASS = "'\"`‘“"
_ALL_QUOTE_CHARACTERS = frozenset((*_QUOTE_OPEN_TO_CLOSE, *_QUOTE_OPEN_TO_CLOSE.values()))
_PAYLOAD_QUOTE_OPEN_RE: Final = re.compile(rf"[{_QUOTE_OPEN_CLASS}]")
_DIRECTIVE_PREFILTER_RE: Final = re.compile(
    rf"\b{_DIRECTIVE_CUES}\b",
    re.IGNORECASE,
)
_ASCII_LETTER_ENTITY_RE: Final = re.compile(
    r"&#(?:(?P<hex>x[0-9a-f]{1,6})|(?P<decimal>[0-9]{1,7}));",
    re.IGNORECASE,
)
_PASSIVE_DIRECTIVE_PREFILTER_RE: Final = re.compile(
    r"\b(?:with|once|after)\b[^.!?\n]{0,64}\b"
    r"(?:removed|stripped|deleted|dropped|omitted|erased|ignored|excised|"
    r"taken[ \t]+out)\b",
    re.IGNORECASE,
)
_SPACED_SECURITY_WORDS = (
    "remove",
    "removing",
    "strip",
    "stripping",
    "delete",
    "deleting",
    "drop",
    "dropping",
    "omit",
    "omitting",
    "erase",
    "erasing",
    "ignore",
    "ignoring",
    "excise",
    "excising",
    "takeout",
    "takingout",
    "replace",
    "replacing",
    "substitute",
    "substituting",
    "run",
    "execute",
    "invoke",
    "issue",
    "launch",
    "perform",
    "carryout",
    "call",
    "eval",
    "type",
    "submit",
    "enter",
    "paste",
)
_SPACED_DIRECTIVE_PREFILTER_RE: Final = re.compile(
    r"(?<!\w)(?:"
    + "|".join(
        rf"(?:[^\w\r\n]|_){{0,{_MAX_DIRECTIVE_HEADER_SPACING_CHARS}}}".join(
            re.escape(character) for character in word
        )
        for word in _SPACED_SECURITY_WORDS
    )
    + r")(?!\w)",
    re.IGNORECASE,
)
_OVERPADDED_ENCODED_ENTITY_RE: Final = re.compile(
    rf"&(?:#x?0{{{_MAX_ENTITY_ZERO_PADDING + 1}}})",
    re.IGNORECASE,
)
_QUOTED_DIRECTIVE_START_RE: Final = re.compile(
    rf"\b(?:{_REMOVAL_VERBS})\b{_DECLARED_MARKER_PREFIX}"
    rf"(?P<quote>[{_QUOTE_OPEN_CLASS}])",
    re.IGNORECASE,
)
_PASSIVE_QUOTED_DIRECTIVE_START_RE: Final = re.compile(
    rf"\b(?:with|once|after){_DECLARED_MARKER_PREFIX}"
    rf"(?P<quote>[{_QUOTE_OPEN_CLASS}])",
    re.IGNORECASE,
)
_PASSIVE_ENCODED_DIRECTIVE_START_RE: Final = re.compile(
    rf"\b(?:with|once|after){_DECLARED_MARKER_PREFIX}"
    rf"(?P<quote>&(?:#x{_ENTITY_ZERO_PADDING}27|#{_ENTITY_ZERO_PADDING}39|apos|"
    rf"#x{_ENTITY_ZERO_PADDING}22|#{_ENTITY_ZERO_PADDING}34|quot);|"
    r"\\(?:x27|u0027|x22|u0022))",
    re.IGNORECASE,
)
_PASSIVE_REMOVAL_SUFFIX_RE: Final = re.compile(
    rf"[ \t]{{1,{_MAX_DIRECTIVE_HEADER_SPACING_CHARS}}}"
    rf"(?:(?:the[ \t]+)?(?:markers?|tokens?|substrings?|strings?|text|characters?)"
    rf"[ \t]{{1,{_MAX_DIRECTIVE_HEADER_SPACING_CHARS}}})?"
    r"(?:is[ \t]+)?"
    r"(?:removed|stripped|deleted|dropped|omitted|erased|ignored|excised|"
    r"taken[ \t]+out)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_QUOTED_DIRECTIVE_START_RE: Final = re.compile(
    rf"\b(?:{_FALLBACK_REMOVAL_VERBS})\b[^.!?\n]{{0,{_MAX_FALLBACK_PREFIX_CHARS}}}?"
    rf"(?P<quote>[{_QUOTE_OPEN_CLASS}])",
    re.IGNORECASE,
)
_EMPTY_REPLACEMENT_DIRECTIVE_START_RE: Final = re.compile(
    rf"\b(?:{_REPLACEMENT_VERBS})\b{_DECLARED_MARKER_PREFIX}"
    rf"(?P<quote>[{_QUOTE_OPEN_CLASS}])",
    re.IGNORECASE,
)
_EMPTY_REPLACEMENT_SUFFIX_RE: Final = re.compile(
    r"[ \t]{1,32}with[ \t]+(?:(?:an?|the)[ \t]+)?"
    r"(?:nothing\b|empty[ \t]+(?:string|text|value)\b|''|\"\")",
    re.IGNORECASE,
)
_ENCODED_DIRECTIVE_START_RE: Final = re.compile(
    rf"\b(?:{_REMOVAL_VERBS})\b{_DECLARED_MARKER_PREFIX}"
    rf"(?P<quote>&(?:#x{_ENTITY_ZERO_PADDING}27|#{_ENTITY_ZERO_PADDING}39|apos|"
    rf"#x{_ENTITY_ZERO_PADDING}22|#{_ENTITY_ZERO_PADDING}34|quot);|"
    r"\\(?:x27|u0027|x22|u0022))",
    re.IGNORECASE,
)
_UNSUPPORTED_ENCODED_REPLACEMENT_DIRECTIVE_START_RE: Final = re.compile(
    rf"\b(?:{_REPLACEMENT_VERBS})\b{_DECLARED_MARKER_PREFIX}"
    rf"(?P<quote>&(?:#x{_ENTITY_ZERO_PADDING}27|#{_ENTITY_ZERO_PADDING}39|apos|"
    rf"#x{_ENTITY_ZERO_PADDING}22|#{_ENTITY_ZERO_PADDING}34|quot);|"
    r"\\(?:x27|u0027|x22|u0022))",
    re.IGNORECASE,
)
_UNSUPPORTED_ENCODED_DIRECTIVE_START_RE: Final = re.compile(
    rf"\b(?:{_FALLBACK_REMOVAL_VERBS})\b[^.!?\n]{{0,{_MAX_FALLBACK_PREFIX_CHARS}}}?"
    rf"(?P<quote>&(?:#x{_FALLBACK_ENTITY_ZERO_PADDING}27|"
    rf"#{_FALLBACK_ENTITY_ZERO_PADDING}39|apos|"
    rf"#x{_FALLBACK_ENTITY_ZERO_PADDING}22|#{_FALLBACK_ENTITY_ZERO_PADDING}34|quot);|"
    r"\\(?:x27|u0027|x22|u0022))",
    re.IGNORECASE,
)
_TAG_DIRECTIVE_START_RE: Final = re.compile(
    rf"\b(?:{_REMOVAL_VERBS})\b{_DECLARED_MARKER_PREFIX}(?P<open><)",
    re.IGNORECASE,
)
_UNSUPPORTED_TAG_DIRECTIVE_START_RE: Final = re.compile(
    rf"\b(?:{_FALLBACK_REMOVAL_VERBS})\b[^.!?\n]{{0,{_MAX_FALLBACK_PREFIX_CHARS}}}?"
    r"(?P<open><)",
    re.IGNORECASE,
)
_ENCODED_TAG_DIRECTIVE_START_RE: Final = re.compile(
    rf"\b(?:{_REMOVAL_VERBS})\b{_DECLARED_MARKER_PREFIX}"
    rf"(?P<open>&(?:lt|#{_ENTITY_ZERO_PADDING}60|#x{_ENTITY_ZERO_PADDING}3c);)",
    re.IGNORECASE,
)
_UNSUPPORTED_ENCODED_TAG_DIRECTIVE_START_RE: Final = re.compile(
    rf"\b(?:{_FALLBACK_REMOVAL_VERBS})\b[^.!?\n]{{0,{_MAX_FALLBACK_PREFIX_CHARS}}}?"
    rf"(?P<open>&(?:lt|#{_FALLBACK_ENTITY_ZERO_PADDING}60|"
    rf"#x{_FALLBACK_ENTITY_ZERO_PADDING}3c);)",
    re.IGNORECASE,
)
_ENCODED_TAG_END_RE: Final = re.compile(
    rf"&(?:gt|#{_ENTITY_ZERO_PADDING}62|#x{_ENTITY_ZERO_PADDING}3e);",
    re.IGNORECASE,
)
_ENCODED_SINGLE_QUOTE_RE: Final = re.compile(
    rf"&(?:#x{_FALLBACK_ENTITY_ZERO_PADDING}27|"
    rf"#{_FALLBACK_ENTITY_ZERO_PADDING}39|apos);|\\(?:x27|u0027)",
    re.IGNORECASE,
)
_ENCODED_DOUBLE_QUOTE_RE: Final = re.compile(
    rf"&(?:#x{_FALLBACK_ENTITY_ZERO_PADDING}22|"
    rf"#{_FALLBACK_ENTITY_ZERO_PADDING}34|quot);|\\(?:x22|u0022)",
    re.IGNORECASE,
)
_TAG_MARKER_RE: Final = re.compile(r"</?[A-Za-z][A-Za-z0-9:_-]*(?:[ \t]*/)?\>")
_UNAMBIGUOUS_ACTION_RE: Final = re.compile(
    r"\b(?:run|execute|invoke|issue|launch|perform|carry[ \t]+out)\b",
    re.IGNORECASE,
)
_CONTEXTUAL_ACTION_RE: Final = re.compile(
    r"\b(?:call|eval|type|submit|enter)\b",
    re.IGNORECASE,
)
_PASTE_ACTION_RE: Final = re.compile(r"\bpaste\b", re.IGNORECASE)
_ACTION_PAYLOAD_PREFIX_RE: Final = re.compile(
    r"[ \t]*(?:(?:exactly|verbatim|literally|directly|immediately)[ \t]+){0,2}"
    r"(?:(?:the|this|following|next)[ \t]+)?"
    r"(?:(?:command|instruction|payload|request|shell[ \t]+command)\b[ \t]*)?"
    r"(?::|=)?[ \t]*",
    re.IGNORECASE,
)
_EXPLICIT_ACTION_PAYLOAD_PREFIX_RE: Final = re.compile(
    r"[ \t]*(?:(?:the|this|following|next)[ \t]+)?"
    r"(?:command|instruction|payload|request|shell[ \t]+command)\b[ \t]*(?::|=)?[ \t]*",
    re.IGNORECASE,
)
_PASTE_TARGET_RE: Final = re.compile(
    r"[ \t]+into[ \t]+(?:(?:the|a)[ \t]+)?"
    r"(?:terminal|shell|console|command[ \t]+prompt|bash|zsh|powershell|cmd(?:\.exe)?)\b",
    re.IGNORECASE,
)
_REMOVAL_PAYLOAD_PREFIX_RE: Final = re.compile(
    r"[ \t]*(?:from|in)[ \t]+"
    r"(?:(?:the|this|following|next)[ \t]+){0,2}"
    r"(?:(?:command|instruction|payload|request)[ \t]*)?(?::|=)?[ \t]*",
    re.IGNORECASE,
)
_COPY_PAYLOAD_PREFIX_RE: Final = re.compile(
    r"[ \t]*(?:,|;)?[ \t]*(?:then[ \t]+)?copy[ \t]+"
    r"(?:(?:the|this|following|next)[ \t]+)?"
    r"(?:(?:command|instruction|payload|request)[ \t]*)?(?::|=)?[ \t]*",
    re.IGNORECASE,
)
_ANAPHORIC_ACTION_RE: Final = re.compile(
    r"\b(?P<action>run|execute|invoke|issue|launch|perform|call|eval|type|submit|"
    r"enter|paste)\b[ \t]+(?:it|this|that|(?:the|this|that)[ \t]+"
    r"(?:result|command|instruction|payload))\b",
    re.IGNORECASE,
)
_INLINE_SHELL_WRAPPER_RE: Final = re.compile(
    r"(?:sudo|command|nohup)[ \t]+[^\s]*$",
    re.IGNORECASE,
)
_DECODED_ACTIVE_TOKEN_RE: Final = re.compile(
    r"\b(?:run|execute|invoke|issue|launch|perform|call|eval|type|submit|enter|"
    r"rm|del|erase|sudo|bash|zsh|powershell)\b",
    re.IGNORECASE,
)
_AFFIRMATIVE_NEGATION_PREFIX_RE: Final = re.compile(
    r"(?:(?:\bdo[ \t]+not|\bdon't|\bnever)[ \t,]+"
    r"(?:forget|fail|hesitate|neglect)(?:[ \t]+at[ \t]+all)?[ \t]+to"
    r"(?:[ \t]+(?!(?:never|not|don't)\b)\w+){0,2}|"
    r"(?:\bdo[ \t]+not|\bdon't|\bnever)[ \t,]+forget"
    r"(?:[ \t]+that)?(?:[ \t]+you)?[ \t]+(?:must|should|shall|will|need[ \t]+to)|"
    r"\bnot[ \t]+(?:only|just)(?:[ \t]+\w+){0,2}|"
    r"(?:\bdo[ \t]+not|\bdon't|\bnever|\bavoid)[ \t]+"
    r"(?:only|just|merely)|"
    r"(?:\bdo[ \t]+not|\bdon't|\bnever)[ \t]+stop[ \t]+at)[ \t]*$",
    re.IGNORECASE,
)
_NEGATED_ACTION_PREFIX_RE: Final = re.compile(
    r"(?:(?:\bdo[ \t]+not|\bdon't|\bnever|\bavoid|\bmust[ \t]+not)"
    r"(?:(?:[ \t]*,[ \t]*|[ \t]+)"
    r"(?!(?:so|but|yet|then|therefore|however|instead)\b)\w+){0,8}[ \t,]*|"
    r"\bnot[ \t,]*|"
    r"\bnot(?:[ \t]*,[ \t]*|[ \t]+)"
    r"(?:ever|directly|immediately|actually)[ \t,]*|"
    r"\bnot(?:[ \t]*,[ \t]*|[ \t]+)(?:under|in)"
    r"(?:[ \t]*,[ \t]*|[ \t]+)(?:any|all|these|those)"
    r"(?:(?:[ \t]*,[ \t]*|[ \t]+)\w+){0,4}[ \t,]*|"
    r"\bunder(?:[ \t]*,[ \t]*|[ \t]+)no"
    r"(?:[ \t]*,[ \t]*|[ \t]+)(?:circumstances?|conditions?)[ \t,]*|"
    r"\bin(?:[ \t]*,[ \t]*|[ \t]+)no"
    r"(?:[ \t]*,[ \t]*|[ \t]+)case[ \t,]*|"
    r"\brefrain(?:ing)?[ \t]+from[ \t]*)$",
    re.IGNORECASE,
)
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
    unsupported: bool = False
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


def _decode_ascii_letter_entities_view(view: SecurityTextView) -> SecurityTextView:
    """Decode bounded numeric HTML entities only when they spell ASCII letters."""
    replacements: list[tuple[re.Match[str], str]] = []
    for match in _ASCII_LETTER_ENTITY_RE.finditer(view.text):
        digits = match.group("hex") or match.group("decimal")
        base = 16 if match.group("hex") is not None else 10
        value = int(digits[1:] if base == 16 else digits, base)
        character = chr(value) if value <= 0x7F else ""
        if character.isascii() and character.isalpha():
            replacements.append((match, character))
    if not replacements:
        return view

    output = StringIO()
    offsets = array("I")
    cursor = 0
    for match, character in replacements:
        output.write(view.text[cursor : match.start()])
        if view.source_offsets is None:
            offsets.extend(range(cursor, match.start()))
        else:
            offsets.extend(view.source_offsets[cursor : match.start()])
        output.write(character)
        offsets.append(view.source_offset(match.start()))
        cursor = match.end()
    output.write(view.text[cursor:])
    if view.source_offsets is None:
        offsets.extend(range(cursor, len(view.text)))
    else:
        offsets.extend(view.source_offsets[cursor:])
    return SecurityTextView(f"entity-{view.name}", output.getvalue(), offsets)


def _compact_spaced_security_word_view(view: SecurityTextView) -> SecurityTextView:
    """Collapse bounded separators inside exact directive and action words.

    The shared compact view intentionally requires six letters before removing
    spacing. Several directive verbs are shorter, so relying on that generic
    threshold creates a security-only eligibility mismatch. This projection is
    restricted to complete allowlisted verbs and retains exact source offsets.
    """
    matches = tuple(
        match
        for match in _SPACED_DIRECTIVE_PREFILTER_RE.finditer(view.text)
        if not match.group().isalpha()
    )
    if not matches:
        return view

    output = StringIO()
    offsets = array("I")
    cursor = 0
    for match in matches:
        output.write(view.text[cursor : match.start()])
        if view.source_offsets is None:
            offsets.extend(range(cursor, match.start()))
        else:
            offsets.extend(view.source_offsets[cursor : match.start()])
        for source_offset in range(match.start(), match.end()):
            character = view.text[source_offset]
            if not character.isalpha():
                continue
            output.write(character)
            offsets.append(view.source_offset(source_offset))
        cursor = match.end()
    output.write(view.text[cursor:])
    if view.source_offsets is None:
        offsets.extend(range(cursor, len(view.text)))
    else:
        offsets.extend(view.source_offsets[cursor:])
    return SecurityTextView(f"marker-{view.name}", output.getvalue(), offsets)


def _quoted_directives(
    text: str,
    check_runtime: Callable[[], None] | None,
    *,
    end_is_truncated: bool,
    pattern: re.Pattern[str] = _QUOTED_DIRECTIVE_START_RE,
    unsupported_header: bool = False,
) -> Iterator[_Directive]:
    for match in pattern.finditer(text):
        if check_runtime is not None:
            check_runtime()
        quote = _QUOTE_OPEN_TO_CLOSE[match.group("quote")]
        marker_start = match.end()
        marker_end_limit = min(len(text), marker_start + MAX_MARKER_LOOKAHEAD_CHARS)
        cursor = marker_start
        unsupported = unsupported_header
        while cursor < marker_end_limit:
            character = text[cursor]
            if character == quote:
                if cursor > marker_start:
                    yield _Directive(
                        text[marker_start:cursor],
                        match.start(),
                        cursor + 1,
                        unsupported=unsupported,
                    )
                break
            if character.isspace() or character in _ALL_QUOTE_CHARACTERS:
                unsupported = True
            cursor += 1
        else:
            if marker_end_limit < len(text) or end_is_truncated:
                yield _Directive("", match.start(), marker_end_limit, exhausted=True)


def _passive_quoted_directives(
    text: str,
    check_runtime: Callable[[], None] | None,
    *,
    end_is_truncated: bool,
) -> Iterator[_Directive]:
    """Parse bounded ``with '<marker>' removed`` declarations."""
    for match in _PASSIVE_QUOTED_DIRECTIVE_START_RE.finditer(text):
        if check_runtime is not None:
            check_runtime()
        close = _QUOTE_OPEN_TO_CLOSE[match.group("quote")]
        marker_start = match.end()
        marker_end_limit = min(len(text), marker_start + MAX_MARKER_LOOKAHEAD_CHARS)
        marker_end = text.find(close, marker_start, marker_end_limit)
        if marker_end < 0:
            if marker_end_limit < len(text) or end_is_truncated:
                yield _Directive("", match.start(), marker_end_limit, exhausted=True)
            continue
        suffix = _PASSIVE_REMOVAL_SUFFIX_RE.match(
            text,
            marker_end + len(close),
            min(len(text), marker_end + len(close) + MAX_DIRECTIVE_HEADER_CHARS),
        )
        if suffix is None or marker_end == marker_start:
            continue
        marker = text[marker_start:marker_end]
        unsupported = any(
            character.isspace() or character in _ALL_QUOTE_CHARACTERS for character in marker
        )
        yield _Directive(
            marker,
            match.start(),
            suffix.end(),
            unsupported=unsupported,
        )


def _passive_encoded_directives(
    text: str,
    check_runtime: Callable[[], None] | None,
    *,
    end_is_truncated: bool,
) -> Iterator[_Directive]:
    """Parse passive declarations using an encoded quote delimiter."""
    for match in _PASSIVE_ENCODED_DIRECTIVE_START_RE.finditer(text):
        if check_runtime is not None:
            check_runtime()
        opening = match.group("quote")
        closing_pattern = (
            _ENCODED_SINGLE_QUOTE_RE
            if _ENCODED_SINGLE_QUOTE_RE.fullmatch(opening) is not None
            else _ENCODED_DOUBLE_QUOTE_RE
        )
        marker_start = match.end()
        marker_end_limit = min(len(text), marker_start + MAX_MARKER_LOOKAHEAD_CHARS)
        marker_end = closing_pattern.search(text, marker_start, marker_end_limit)
        if marker_end is None:
            if marker_end_limit < len(text) or end_is_truncated:
                yield _Directive(
                    "",
                    match.start(),
                    marker_end_limit,
                    encoded=True,
                    exhausted=True,
                )
            continue
        suffix = _PASSIVE_REMOVAL_SUFFIX_RE.match(
            text,
            marker_end.end(),
            min(len(text), marker_end.end() + MAX_DIRECTIVE_HEADER_CHARS),
        )
        marker = text[marker_start : marker_end.start()]
        if suffix is not None and marker:
            yield _Directive(
                marker,
                match.start(),
                suffix.end(),
                encoded=True,
            )


def _empty_replacement_directives(
    text: str,
    check_runtime: Callable[[], None] | None,
    *,
    end_is_truncated: bool,
) -> Iterator[_Directive]:
    """Parse bounded declarations that explicitly replace a literal with emptiness."""
    for match in _EMPTY_REPLACEMENT_DIRECTIVE_START_RE.finditer(text):
        if check_runtime is not None:
            check_runtime()
        close = _QUOTE_OPEN_TO_CLOSE[match.group("quote")]
        marker_start = match.end()
        marker_end_limit = min(len(text), marker_start + MAX_MARKER_LOOKAHEAD_CHARS)
        marker_end = text.find(close, marker_start, marker_end_limit)
        if marker_end < 0:
            if marker_end_limit < len(text) or end_is_truncated:
                yield _Directive("", match.start(), marker_end_limit, exhausted=True)
            continue
        marker = text[marker_start:marker_end]
        suffix = _EMPTY_REPLACEMENT_SUFFIX_RE.match(
            text,
            marker_end + len(close),
            min(len(text), marker_end + len(close) + MAX_DIRECTIVE_HEADER_CHARS),
        )
        if not marker or suffix is None:
            continue
        unsupported = any(
            character.isspace() or character in _ALL_QUOTE_CHARACTERS for character in marker
        )
        yield _Directive(
            marker,
            match.start(),
            suffix.end(),
            unsupported=unsupported,
        )


def _tag_directives(
    text: str,
    check_runtime: Callable[[], None] | None,
    *,
    end_is_truncated: bool,
    pattern: re.Pattern[str] = _TAG_DIRECTIVE_START_RE,
    unsupported_header: bool = False,
) -> Iterator[_Directive]:
    for match in pattern.finditer(text):
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
            yield _Directive(
                marker,
                match.start(),
                marker_end + 1,
                is_tag=True,
                unsupported=unsupported_header,
            )


def _encoded_directives(
    text: str,
    check_runtime: Callable[[], None] | None,
    *,
    end_is_truncated: bool,
    pattern: re.Pattern[str] = _ENCODED_DIRECTIVE_START_RE,
    unsupported_header: bool = False,
) -> Iterator[_Directive]:
    for match in pattern.finditer(text):
        if check_runtime is not None:
            check_runtime()
        quote = match.group("quote")
        closing_pattern = (
            _ENCODED_SINGLE_QUOTE_RE
            if _ENCODED_SINGLE_QUOTE_RE.fullmatch(quote) is not None
            else _ENCODED_DOUBLE_QUOTE_RE
        )
        marker_start = match.end()
        marker_end_limit = min(len(text), marker_start + MAX_MARKER_LOOKAHEAD_CHARS)
        marker_end_match = closing_pattern.search(text, marker_start, marker_end_limit)
        if marker_end_match is None:
            overpadded_end = _OVERPADDED_ENCODED_ENTITY_RE.search(
                text,
                marker_start,
                marker_end_limit,
            )
            if overpadded_end is not None:
                marker = text[marker_start : overpadded_end.start()]
                if marker:
                    yield _Directive(
                        marker,
                        match.start(),
                        overpadded_end.end(),
                        encoded=True,
                        unsupported=True,
                    )
            elif marker_end_limit < len(text) or end_is_truncated:
                yield _Directive("", match.start(), marker_end_limit, encoded=True, exhausted=True)
            continue
        marker = text[marker_start : marker_end_match.start()]
        if marker and not any(character.isspace() for character in marker):
            yield _Directive(
                marker,
                match.start(),
                marker_end_match.end(),
                encoded=True,
                unsupported=unsupported_header,
            )


def _encoded_tag_directives(
    text: str,
    check_runtime: Callable[[], None] | None,
    *,
    end_is_truncated: bool,
    pattern: re.Pattern[str] = _ENCODED_TAG_DIRECTIVE_START_RE,
    unsupported_header: bool = False,
) -> Iterator[_Directive]:
    for match in pattern.finditer(text):
        if check_runtime is not None:
            check_runtime()
        marker_start = match.start("open")
        marker_end_limit = min(len(text), marker_start + MAX_MARKER_LOOKAHEAD_CHARS)
        marker_end = _ENCODED_TAG_END_RE.search(text, match.end(), marker_end_limit)
        if marker_end is None:
            overpadded_end = _OVERPADDED_ENCODED_ENTITY_RE.search(
                text,
                match.end(),
                marker_end_limit,
            )
            if overpadded_end is not None:
                marker = text[marker_start : overpadded_end.end()]
                yield _Directive(
                    marker,
                    match.start(),
                    overpadded_end.end(),
                    is_tag=True,
                    encoded=True,
                    unsupported=True,
                )
            elif marker_end_limit < len(text) or end_is_truncated:
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
            unsupported=unsupported_header,
        )


def _directives(
    text: str,
    check_runtime: Callable[[], None] | None,
    *,
    end_is_truncated: bool,
) -> list[_Directive]:
    candidates = [
        *_passive_quoted_directives(
            text,
            check_runtime,
            end_is_truncated=end_is_truncated,
        ),
        *_passive_encoded_directives(
            text,
            check_runtime,
            end_is_truncated=end_is_truncated,
        ),
        *_empty_replacement_directives(
            text,
            check_runtime,
            end_is_truncated=end_is_truncated,
        ),
        *_quoted_directives(text, check_runtime, end_is_truncated=end_is_truncated),
        *_quoted_directives(
            text,
            check_runtime,
            end_is_truncated=end_is_truncated,
            pattern=_UNSUPPORTED_QUOTED_DIRECTIVE_START_RE,
            unsupported_header=True,
        ),
        *_tag_directives(text, check_runtime, end_is_truncated=end_is_truncated),
        *_tag_directives(
            text,
            check_runtime,
            end_is_truncated=end_is_truncated,
            pattern=_UNSUPPORTED_TAG_DIRECTIVE_START_RE,
            unsupported_header=True,
        ),
        *_encoded_directives(text, check_runtime, end_is_truncated=end_is_truncated),
        *_encoded_directives(
            text,
            check_runtime,
            end_is_truncated=end_is_truncated,
            pattern=_UNSUPPORTED_ENCODED_REPLACEMENT_DIRECTIVE_START_RE,
            unsupported_header=True,
        ),
        *_encoded_directives(
            text,
            check_runtime,
            end_is_truncated=end_is_truncated,
            pattern=_UNSUPPORTED_ENCODED_DIRECTIVE_START_RE,
            unsupported_header=True,
        ),
        *_encoded_tag_directives(text, check_runtime, end_is_truncated=end_is_truncated),
        *_encoded_tag_directives(
            text,
            check_runtime,
            end_is_truncated=end_is_truncated,
            pattern=_UNSUPPORTED_ENCODED_TAG_DIRECTIVE_START_RE,
            unsupported_header=True,
        ),
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
    for boundary_start, _ in _sentence_boundaries(text, start, min(len(text), maximum_end + 1)):
        if boundary_start <= maximum_end:
            return boundary_start, False
    exhausted = maximum_end < len(text) or end_is_truncated
    return maximum_end, exhausted


def _previous_sentence_boundary(text: str, end: int, maximum_span: int) -> int:
    start = max(0, end - maximum_span)
    last_end = start
    for _, boundary_end in _sentence_boundaries(text, start, end):
        last_end = boundary_end
    return last_end


def _sentence_boundaries(text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
    """Yield prose boundaries while ignoring punctuation inside balanced quotes."""
    quote_close: str | None = None
    cursor = start
    while cursor < end:
        character = text[cursor]
        if quote_close is not None:
            if character == "\\" and quote_close in {"'", '"', "`"}:
                cursor += 2
                continue
            if character == quote_close:
                quote_close = None
            cursor += 1
            continue
        if character in _QUOTE_OPEN_TO_CLOSE and not (
            character == "'"
            and cursor > 0
            and cursor + 1 < len(text)
            and text[cursor - 1].isalnum()
            and text[cursor + 1].isalnum()
        ):
            quote_close = _QUOTE_OPEN_TO_CLOSE[character]
        elif character == "\n":
            yield cursor, cursor + 1
        elif character in ".!?" and (cursor + 1 == len(text) or text[cursor + 1].isspace()):
            yield cursor, cursor + 1
        cursor += 1


def _verb_is_negated(text: str, verb_start: int, lower_bound: int) -> bool:
    prefix_start = max(lower_bound, verb_start - MAX_NEGATION_PREFIX_CHARS)
    if _AFFIRMATIVE_NEGATION_PREFIX_RE.search(text, prefix_start, verb_start) is not None:
        return False
    return _NEGATED_ACTION_PREFIX_RE.search(text, prefix_start, verb_start) is not None


def _contextual_action_is_imperative(text: str, action_start: int, lower_bound: int) -> bool:
    prefix = text[max(lower_bound, action_start - 48) : action_start].rstrip()
    if not prefix or prefix[-1] in ".!?;:\n":
        return True
    previous = re.search(r"([A-Za-z]+)[^A-Za-z]*$", prefix)
    return previous is not None and previous.group(1).casefold() in {
        "and",
        "then",
        "please",
        "to",
        "do",
        "must",
        "should",
        "shall",
        "can",
        "will",
        "now",
        "next",
        "only",
        "just",
        "immediately",
    }


def _actions(
    text: str,
    start: int,
    end: int,
    *,
    include_contextual: bool = True,
) -> list[re.Match[str]]:
    matches = [
        action
        for action in _UNAMBIGUOUS_ACTION_RE.finditer(text, start, end)
        if not _verb_is_negated(text, action.start(), start)
    ]
    if include_contextual:
        matches.extend(
            action
            for action in _CONTEXTUAL_ACTION_RE.finditer(text, start, end)
            if _contextual_action_is_imperative(text, action.start(), start)
            and not _verb_is_negated(text, action.start(), start)
        )
        matches.extend(
            action
            for action in _PASTE_ACTION_RE.finditer(text, start, end)
            if _contextual_action_is_imperative(text, action.start(), start)
            and not _verb_is_negated(text, action.start(), start)
        )
    return sorted(matches, key=lambda action: action.start())


def _action_governs_quoted_payload(
    text: str,
    action: re.Match[str],
    quote_start: int,
    quote_end: int,
    scope_end: int,
) -> bool:
    """Require a direct action-to-payload edge instead of word co-occurrence."""
    gap = text[action.end() : quote_start]
    if _ACTION_PAYLOAD_PREFIX_RE.fullmatch(gap) is None:
        return False
    if action.group().casefold() != "paste":
        return True
    return _PASTE_TARGET_RE.match(text, quote_end, scope_end) is not None


def _unsupported_form_actions(
    text: str,
    start: int,
    end: int,
    *,
    include_contextual: bool = False,
) -> list[re.Match[str]]:
    """Return actions strong enough to fail closed without a parsed payload."""
    return _actions(text, start, end, include_contextual=include_contextual)


def _has_action_continuation(text: str, boundary: int, end: int) -> bool:
    cursor = boundary + 1
    while cursor < end and text[cursor].isspace():
        cursor += 1
    then = re.match(r"then\b[ \t]*", text[cursor:end], re.IGNORECASE)
    if then is not None:
        cursor += then.end()
    actions = _actions(text, cursor, min(end, cursor + 64))
    return bool(actions) and actions[0].start() == cursor


def _action_immediately_precedes_boundary(text: str, start: int, boundary: int) -> bool:
    actions = _actions(text, start, boundary)
    if not actions:
        return False
    return text[actions[-1].end() : boundary].strip() in {"", ":"}


def _quoted_spans(text: str, start: int, end: int) -> Iterator[tuple[int, int, int, int]]:
    """Yield bounded quote/body spans for ASCII and typographic quote pairs."""
    cursor = start
    while cursor < end:
        opening = _PAYLOAD_QUOTE_OPEN_RE.search(text, cursor, end)
        if opening is None:
            return
        close = _QUOTE_OPEN_TO_CLOSE[opening.group()]
        body_start = opening.end()
        close_at = text.find(close, body_start, min(end, body_start + MAX_PAYLOAD_CHARS + 1))
        if close_at < 0:
            cursor = opening.end()
            continue
        if any(character in _ALL_QUOTE_CHARACTERS for character in text[body_start:close_at]):
            cursor = close_at + 1
            continue
        yield opening.start(), body_start, close_at, close_at + len(close)
        cursor = close_at + len(close)


def _quoted_payloads(text: str, directive: _Directive, scope_end: int) -> list[_Payload]:
    actions = _actions(text, directive.end, scope_end)
    if not actions:
        return []
    payloads: set[tuple[int, int]] = set()
    for quote_start, body_start, body_end, quote_end in _quoted_spans(
        text,
        directive.end,
        scope_end,
    ):
        if quote_end < len(text) and text[quote_end].isalnum():
            continue
        if text.find(directive.marker, body_start, body_end) < 0:
            continue
        if any(
            action.end() <= quote_start
            and _action_governs_quoted_payload(
                text,
                action,
                quote_start,
                quote_end,
                scope_end,
            )
            for action in actions
        ):
            payloads.add((body_start, body_end))
    return [_Payload(start, end) for start, end in sorted(payloads)]


def _anaphoric_payloads(text: str, directive: _Directive, lookahead_end: int) -> list[_Payload]:
    """Bind a declared source quote to a later explicit ``execute it/result`` action."""
    payloads: set[tuple[int, int]] = set()
    for quote_start, body_start, body_end, quote_end in _quoted_spans(
        text,
        directive.end,
        lookahead_end,
    ):
        source_prefix = text[directive.end : quote_start]
        if (
            _REMOVAL_PAYLOAD_PREFIX_RE.fullmatch(source_prefix) is None
            and _COPY_PAYLOAD_PREFIX_RE.fullmatch(source_prefix) is None
        ):
            continue
        if text.find(directive.marker, body_start, body_end) < 0:
            continue
        for action in _ANAPHORIC_ACTION_RE.finditer(text, quote_end, lookahead_end):
            if _verb_is_negated(text, action.start(), quote_end):
                continue
            if (
                action.group("action").casefold() == "paste"
                and _PASTE_TARGET_RE.match(
                    text,
                    action.end(),
                    lookahead_end,
                )
                is None
            ):
                continue
            payloads.add((body_start, body_end))
            break
    return [_Payload(start, end) for start, end in sorted(payloads)]


def _inline_payloads(text: str, directive: _Directive, scope_end: int) -> list[_Payload]:
    payloads: set[tuple[int, int]] = set()
    for action in _actions(text, directive.end, scope_end):
        if action.group().casefold() == "paste":
            continue
        explicit_prefix = _EXPLICIT_ACTION_PAYLOAD_PREFIX_RE.match(
            text,
            action.end(),
            scope_end,
        )
        prefix = explicit_prefix or _ACTION_PAYLOAD_PREFIX_RE.match(
            text,
            action.end(),
            scope_end,
        )
        start = prefix.end() if prefix is not None else action.end()
        if start >= scope_end or text[start] in _ALL_QUOTE_CHARACTERS:
            continue
        end = min(scope_end, start + MAX_PAYLOAD_CHARS)
        if explicit_prefix is None:
            first_token_end = start
            while first_token_end < end and not text[first_token_end].isspace():
                first_token_end += 1
            marker_at = text.find(directive.marker, start, end)
            if marker_at < 0:
                continue
            before_marker = text[start:marker_at]
            if (
                marker_at >= first_token_end
                and _INLINE_SHELL_WRAPPER_RE.fullmatch(before_marker) is None
            ):
                continue
        if text.find(directive.marker, start, end) >= 0:
            payloads.add((start, end))
    return [_Payload(start, end) for start, end in sorted(payloads)]


def _has_coordinated_marker(
    text: str,
    directive: _Directive,
    payloads: list[_Payload],
    scope_end: int,
) -> bool:
    """Return whether another declared literal also participates in the payload."""
    actions = _actions(text, directive.end, scope_end)
    if not actions or not payloads:
        return False
    for _, marker_start, marker_end, _ in _quoted_spans(
        text,
        directive.end,
        actions[0].start(),
    ):
        marker = text[marker_start:marker_end]
        if not marker or marker == directive.marker or len(marker) > MAX_MARKER_LENGTH:
            continue
        if any(marker in text[payload.start : payload.end] for payload in payloads):
            return True
    return False


def _has_active_unsupported_form(
    text: str,
    directive: _Directive,
    lookahead_end: int,
) -> bool:
    clause_start = _previous_sentence_boundary(text, directive.start, MAX_MARKER_LOOKAHEAD_CHARS)
    tail = text[directive.end : lookahead_end]
    marker_in_tail = directive.marker in tail
    actions_after = _unsupported_form_actions(
        text,
        directive.end,
        lookahead_end,
        include_contextual=True,
    )
    actions_before = _unsupported_form_actions(
        text,
        clause_start,
        directive.start,
        include_contextual=True,
    )

    for quote_start, body_start, body_end, quote_end in _quoted_spans(
        text,
        directive.end,
        lookahead_end,
    ):
        body = text[body_start:body_end]
        if directive.marker.casefold() not in body.casefold():
            continue
        if any(
            action.end() <= quote_start
            and _action_governs_quoted_payload(
                text,
                action,
                quote_start,
                quote_end,
                lookahead_end,
            )
            for action in actions_after
        ):
            return True
        for action in actions_after:
            if (
                action.end() > quote_start
                or _UNAMBIGUOUS_ACTION_RE.fullmatch(action.group()) is None
                or not _contextual_action_is_imperative(text, action.start(), directive.end)
            ):
                continue
            unsupported_gap = text[action.end() : quote_start]
            if len(unsupported_gap) <= 48 and re.fullmatch(
                r"[ \t]+(?:[A-Za-z-]+[ \t]+){0,2}",
                unsupported_gap,
            ):
                return True

    # ``Execute this after removing 'x' from '<payload>'`` puts the action
    # before the declaration but the governed source payload after it.  The
    # order is unambiguous enough to fail closed, but not to reconstruct and
    # claim a definite finding.
    if (
        actions_before
        and re.search(
            r"\bafter(?:[ \t]+(?:this|that|it|the[ \t]+result))?[ \t]+$",
            text[actions_before[-1].end() : directive.start],
            re.IGNORECASE,
        )
        is not None
    ):
        for quote_start, body_start, body_end, _ in _quoted_spans(
            text,
            directive.end,
            lookahead_end,
        ):
            if (
                _REMOVAL_PAYLOAD_PREFIX_RE.fullmatch(text[directive.end : quote_start]) is not None
                and directive.marker in text[body_start:body_end]
            ):
                return True

    if marker_in_tail:
        for action in actions_after:
            if action.group().casefold() == "paste":
                continue
            prefix = _ACTION_PAYLOAD_PREFIX_RE.match(text, action.end(), lookahead_end)
            if prefix is None:
                continue
            payload_start = prefix.end()
            while payload_start < lookahead_end and text[payload_start].isspace():
                payload_start += 1
            marker_at = text.find(directive.marker, payload_start, lookahead_end)
            if marker_at >= 0 and not any(
                character.isspace() for character in text[payload_start:marker_at]
            ):
                return True

    # Handle the natural reverse order: the governed payload appears first and
    # the literal-removal declaration follows ("execute '<payload>' after
    # removing '<marker>'").  This is ambiguous rather than safe, so callers
    # report an incomplete deterministic analysis instead of reconstructing it.
    for quote_start, body_start, body_end, quote_end in _quoted_spans(
        text,
        clause_start,
        directive.start,
    ):
        if text.find(directive.marker, body_start, body_end) < 0:
            continue
        if any(
            action.end() <= quote_start
            and _action_governs_quoted_payload(
                text,
                action,
                quote_start,
                quote_end,
                directive.start,
            )
            for action in actions_before
        ):
            return True

    for action in actions_before:
        if action.group().casefold() == "paste":
            continue
        explicit_prefix = _EXPLICIT_ACTION_PAYLOAD_PREFIX_RE.match(
            text,
            action.end(),
            directive.start,
        )
        prefix = explicit_prefix or _ACTION_PAYLOAD_PREFIX_RE.match(
            text,
            action.end(),
            directive.start,
        )
        if prefix is None:
            continue
        marker_at = text.find(directive.marker, prefix.end(), directive.start)
        if marker_at < 0:
            continue
        if explicit_prefix is not None or not text[prefix.end() : marker_at].strip():
            return True

    # A quoted payload may precede the declaration while an explicit anaphoric
    # action follows it: ``'<payload>'; remove '<marker>' and execute it``.
    if any(
        directive.marker in text[body_start:body_end]
        for _, body_start, body_end, _ in _quoted_spans(
            text,
            clause_start,
            directive.start,
        )
    ) and any(
        not _verb_is_negated(text, action.start(), directive.end)
        for action in _ANAPHORIC_ACTION_RE.finditer(text, directive.end, lookahead_end)
    ):
        return True

    if marker_in_tail:
        # A newly exposed command/action token matters only in instruction
        # text. Quoted examples are inert unless a parsed action explicitly
        # governs the quote, which is handled by the payload binders above.
        unquoted_tail = list(tail)
        for quote_start, _, _, quote_end in _quoted_spans(tail, 0, len(tail)):
            unquoted_tail[quote_start:quote_end] = " " * (quote_end - quote_start)
        active_tail = "".join(unquoted_tail)
        decoded_tail = active_tail.replace(directive.marker, "")
        original_action_names = Counter(
            action.group().casefold()
            for action in _unsupported_form_actions(
                active_tail,
                0,
                len(active_tail),
                include_contextual=True,
            )
        )
        decoded_actions = _unsupported_form_actions(
            decoded_tail,
            0,
            len(decoded_tail),
            include_contextual=True,
        )
        decoded_action_names = Counter(action.group().casefold() for action in decoded_actions)
        if any(count > original_action_names[name] for name, count in decoded_action_names.items()):
            return True
        original_active_tokens = Counter(
            match.group().casefold() for match in _DECODED_ACTIVE_TOKEN_RE.finditer(active_tail)
        )
        decoded_active_tokens = Counter(
            match.group().casefold() for match in _DECODED_ACTIVE_TOKEN_RE.finditer(decoded_tail)
        )
        if any(
            count > original_active_tokens[name] for name, count in decoded_active_tokens.items()
        ):
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
    if first_sentence_end < lookahead_cap and (
        _FORWARD_PAYLOAD_REFERENCE_RE.search(directive_clause) is not None
        or _has_action_continuation(view.text, first_sentence_end, lookahead_cap)
        or _action_immediately_precedes_boundary(
            view.text,
            directive.end,
            first_sentence_end,
        )
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
        payloads = _anaphoric_payloads(view.text, directive, lookahead_end)
    if not payloads:
        payloads = _inline_payloads(view.text, directive, scope_end)
    coordinated_marker = _has_coordinated_marker(
        view.text,
        directive,
        payloads,
        scope_end,
    )
    active = bool(payloads) or _has_active_unsupported_form(
        view.text,
        directive,
        lookahead_end,
    )
    if not active:
        return _DirectiveClassification(None, False, lookahead_exhausted)

    if (
        directive.encoded
        or directive.unsupported
        or coordinated_marker
        or len(directive.marker) > MAX_MARKER_LENGTH
        or len(payloads) != 1
    ):
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
    source_end_is_truncated: bool = False,
) -> DeclaredMarkerViewResult:
    """Build one-pass payload views for explicit literal-removal instructions.

    The function never evaluates projected text. It accepts one unnegated,
    action-bound payload per directive; ambiguous or resource-bounded active
    forms set ``limited`` so the caller can fail closed without guessing.

    The optional ownership bounds are source coordinates in the current raw
    window. Directives before ``owned_source_start`` belong to the previous
    window; directives at or beyond exclusive ``owned_source_end`` belong to
    the next window. ``source_end_is_truncated`` separately records whether the
    physical view ends before the source, so ownership alone never degrades a
    complete end-of-file directive.
    """
    if check_runtime is not None:
        check_runtime()
    view = _decode_ascii_letter_entities_view(view)
    view = _compact_spaced_security_word_view(view)
    if (
        _DIRECTIVE_PREFILTER_RE.search(view.text) is None
        and _PASSIVE_DIRECTIVE_PREFILTER_RE.search(view.text) is None
    ):
        return DeclaredMarkerViewResult((), False)

    active_directives = 0
    limited = False
    projection_blocked = False
    candidates: list[_ProjectionCandidate] = []
    for directive in _directives(
        view.text,
        check_runtime,
        end_is_truncated=source_end_is_truncated,
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
            end_is_truncated=source_end_is_truncated,
        )
        limited = limited or classification.limited
        if classification.active and classification.limited:
            projection_blocked = True
            candidates.clear()
        if not classification.active:
            continue

        active_directives += 1
        if active_directives > MAX_ACTIVE_DIRECTIVES:
            limited = True
            break
        if classification.candidate is not None and not projection_blocked:
            candidates.append(classification.candidate)

    views, conflict_limited = _resolve_candidates(view, candidates)
    return DeclaredMarkerViewResult(views, limited or conflict_limited)
