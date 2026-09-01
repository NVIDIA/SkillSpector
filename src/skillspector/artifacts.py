# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical artifact classification and security-oriented text views.

The scanner keeps raw bytes as the source of truth.  Text analyzers consume
derived views with source-offset maps so decoding and Unicode normalization do
not create an untracked gap between the bytes that were supplied and the text
that was inspected.
"""

from __future__ import annotations

import re
import unicodedata
from array import array
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from io import StringIO
from typing import NotRequired

from typing_extensions import TypedDict

from skillspector.unicode_confusables import ASCII_CONFUSABLE_SKELETON


class ContentKind(StrEnum):
    """Byte-derived artifact content classification."""

    TEXT = "text"
    BINARY = "binary"
    OPAQUE = "opaque"


class ArtifactDisposition(StrEnum):
    """Normative disposition used by coverage and reference accounting."""

    ANALYZED = "analyzed"
    PARTIAL = "partial"
    FAILED = "failed"
    OUT_OF_SCOPE = "out_of_scope"


class ArtifactRecord(TypedDict):
    """Serializable inventory row for one discovered bundle artifact."""

    path: str
    content_kind: ContentKind
    disposition: ArtifactDisposition
    size_bytes: int
    decodable: bool
    contains_nul: bool
    misleading_extension: bool
    referenced: bool
    reason: NotRequired[str]


class BundleReference(TypedDict):
    """Canonical, report-safe intra-bundle reference record."""

    source_path: str
    line: int
    column: int
    evidence: str
    target_path: str | None
    status: str
    disposition: ArtifactDisposition


@dataclass(frozen=True)
class SecurityTextView:
    """A bounded derived text view and mapping to raw character offsets."""

    name: str
    text: str
    source_offsets: array[int] | None = None

    def source_offset(self, derived_offset: int) -> int:
        """Map a derived character offset to the corresponding source offset."""
        if self.source_offsets is None:
            return min(max(derived_offset, 0), len(self.text))
        if not self.source_offsets:
            return 0
        index = min(max(derived_offset, 0), len(self.source_offsets) - 1)
        return self.source_offsets[index]


@dataclass(frozen=True)
class _ObfuscatedInstructionMatch:
    """One context-bound instruction action reconstructed from source text."""

    start: int
    end: int
    gaps: tuple[tuple[int, int], ...]
    evidence_offset: int


@dataclass(frozen=True)
class _ObfuscatedWordMatch:
    """One fixed security word plus source gaps removed to reconstruct it."""

    end: int
    gaps: tuple[tuple[int, int], ...] = ()
    requires_targeted_reconstruction: bool = False


@dataclass(frozen=True)
class _ObfuscatedBackwardWordMatch:
    """One backward-matched security word plus reconstructed source gaps."""

    start: int
    gaps: tuple[tuple[int, int], ...] = ()
    requires_targeted_reconstruction: bool = False


@dataclass(frozen=True)
class _ObfuscatedBackwardState:
    """One bounded partial state while matching a context word backward."""

    last_letter_start: int
    gaps: tuple[tuple[int, int], ...] = ()
    requires_targeted_reconstruction: bool = False
    gap_has_existing_projection: bool = False
    gap_has_ascii_whitespace: bool = False
    gap_has_targeted_character: bool = False


@dataclass(frozen=True)
class _ObfuscatedIgnoreState:
    """One constant-size partial state in a fixed instruction-word automaton."""

    start: int
    last_letter_end: int
    gaps: tuple[tuple[int, int], ...] = ()
    requires_targeted_reconstruction: bool = False
    gap_has_existing_projection: bool = False
    gap_has_ascii_whitespace: bool = False
    gap_has_targeted_character: bool = False


_BINARY_MAGIC = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"PK\x03\x04",
    b"\x7fELF",
    b"MZ",
    b"\x00asm",
    b"%PDF-",
)

_BINARY_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".pdf",
        ".zip",
        ".gz",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".wasm",
        ".pyc",
        ".class",
        ".mp3",
        ".mp4",
        ".sqlite",
    }
)
_TEXT_EXTENSIONS = frozenset(
    {
        ".md",
        ".markdown",
        ".txt",
        ".py",
        ".sh",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".js",
        ".ts",
        ".rb",
        ".go",
        ".rs",
    }
)

_ALLOWED_FORMAT_CHARS = frozenset({"\n", "\r", "\t"})
_IGNORED_ASCII_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_LETTER_SPACING_CANDIDATE = re.compile(
    r"(?:[^\W\d_](?:[^\w\r\n]|_)+){5}[^\W\d_]",
    re.UNICODE,
)
_CONCEALED_INSTRUCTION_CANDIDATE = re.compile(
    r"(?:[^\W\d_](?:[^\w]|_)+){5}[^\W\d_]",
    re.UNICODE,
)
_PROMPT_SPACED_PAIR = re.compile(r"(?<![A-Za-z])\S \S(?![A-Za-z])")
_MIN_LETTER_SPACING_RUN_LETTERS = 6
# Unicode 15.1.0 DerivedCoreProperties.txt: Default_Ignorable_Code_Point.
_DEFAULT_IGNORABLE_RANGES = (
    (0x00AD, 0x00AD),
    (0x034F, 0x034F),
    (0x061C, 0x061C),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180F),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x206F),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0),
    (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)
_DEFAULT_IGNORABLE_CODEPOINTS = frozenset(
    codepoint for start, end in _DEFAULT_IGNORABLE_RANGES for codepoint in range(start, end + 1)
)
_DEFAULT_IGNORABLE_PATTERN = re.compile(
    "["
    + "".join(
        re.escape(chr(start)) if start == end else f"{re.escape(chr(start))}-{re.escape(chr(end))}"
        for start, end in _DEFAULT_IGNORABLE_RANGES
    )
    + "]"
)
_DEFAULT_IGNORABLE_RUN_PATTERN = re.compile(_DEFAULT_IGNORABLE_PATTERN.pattern + "+")
_REPEATED_CHARACTER_RUN_PATTERN = re.compile(r"(.)\1+")
_ASCII_CONFUSABLE_PATTERN = re.compile(
    "[" + "".join(re.escape(chr(codepoint)) for codepoint in ASCII_CONFUSABLE_SKELETON) + "]"
)
_OBFUSCATED_INSTRUCTION_ACTIONS = (
    "ignore",
    "override",
    "bypass",
    "disregard",
    "forget",
)
_OBFUSCATED_ACTION_INITIALS = frozenset(action[0] for action in _OBFUSCATED_INSTRUCTION_ACTIONS)
_ASCII_OBFUSCATED_ACTION_START_PATTERN = re.compile(
    "[" + re.escape("".join(sorted(_OBFUSCATED_ACTION_INITIALS))) + "]",
    re.ASCII | re.IGNORECASE,
)
_OBFUSCATED_ACTION_START_PATTERN = re.compile(
    r"["
    + re.escape(
        "iIdDfFoObB"
        + "ᴵᵢᶥᶦⁱⒾⓘ𞁌𞁨🄸"
        + "".join(
            chr(codepoint)
            for codepoint in ASCII_CONFUSABLE_SKELETON
            if unicodedata.normalize("NFKC", chr(codepoint))
            .translate(ASCII_CONFUSABLE_SKELETON)
            .casefold()
            in _OBFUSCATED_ACTION_INITIALS
        )
    )
    + "]",
    re.IGNORECASE,
)
_LOGICAL_LINE_BREAK_CHARACTERS = frozenset(
    {"\r", "\n", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"}
)
_REMOVE_ALLOWED_FORMAT_CHARACTERS = str.maketrans("", "", "".join(_ALLOWED_FORMAT_CHARS))


def _suffix(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    index = name.rfind(".")
    return name[index:].lower() if index >= 0 else ""


def classify_artifact(path: str, data: bytes, *, referenced: bool = False) -> ArtifactRecord:
    """Classify from bytes and decodability; an extension is never authoritative."""
    contains_nul = b"\x00" in data
    has_binary_magic = any(data.startswith(magic) for magic in _BINARY_MAGIC)
    try:
        decoded = data.decode("utf-8")
        decodable = True
    except UnicodeDecodeError:
        decoded = data.decode("utf-8", errors="replace")
        decodable = False

    if has_binary_magic:
        kind = ContentKind.BINARY
    elif decodable:
        kind = ContentKind.TEXT
    elif not data:
        kind = ContentKind.TEXT
    else:
        printable = sum(ch.isprintable() or ch in _ALLOWED_FORMAT_CHARS for ch in decoded)
        replacement_ratio = decoded.count("\ufffd") / max(1, len(decoded))
        if printable / max(1, len(decoded)) >= 0.85 and replacement_ratio <= 0.10:
            kind = ContentKind.TEXT
        else:
            kind = ContentKind.BINARY

    suffix = _suffix(path)
    misleading = (suffix in _BINARY_EXTENSIONS and kind is ContentKind.TEXT) or (
        suffix in _TEXT_EXTENSIONS and kind is ContentKind.BINARY
    )
    disposition = (
        ArtifactDisposition.PARTIAL
        if referenced and kind is not ContentKind.TEXT
        else ArtifactDisposition.OUT_OF_SCOPE
        if kind is ContentKind.BINARY
        else ArtifactDisposition.ANALYZED
    )
    return {
        "path": path,
        "content_kind": kind,
        "disposition": disposition,
        "size_bytes": len(data),
        "decodable": decodable,
        "contains_nul": contains_nul,
        "misleading_extension": misleading,
        "referenced": referenced,
    }


def decode_text(data: bytes) -> str:
    """Return the loss-tolerant local text projection for static analyzers."""
    return data.decode("utf-8", errors="replace")


def _is_emoji_base(ch: str) -> bool:
    """Return whether one character is a base for emoji presentation forms."""
    codepoint = ord(ch)
    return (
        0x1F000 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or codepoint in (0x00A9, 0x00AE, 0x203C, 0x2049, 0x2122, 0x2139, 0x3030, 0x303D)
    )


def is_default_ignorable(ch: str) -> bool:
    """Return the pinned Unicode Default_Ignorable_Code_Point property."""
    return ord(ch) in _DEFAULT_IGNORABLE_CODEPOINTS


def _contains_default_ignorable(text: str) -> bool:
    """Return whether *text* contains a pinned default-ignorable code point."""
    return _DEFAULT_IGNORABLE_PATTERN.search(text) is not None


def _is_unconditionally_ignored(ch: str) -> bool:
    return (
        bool(_IGNORED_ASCII_CONTROL.fullmatch(ch))
        or unicodedata.category(ch) in {"Cf", "Cc"}
        and ch not in _ALLOWED_FORMAT_CHARS
    )


def _is_word_character(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _is_security_word_character(ch: str) -> bool:
    """Return whether *ch* is word-like after the security normalization."""
    return _is_word_character(ch) or any(
        _is_word_character(character) for character in _security_skeleton_piece(ch)
    )


def _is_non_ascii_separator(ch: str) -> bool:
    return not ch.isascii() and unicodedata.category(ch).startswith("Z")


def _is_letter_spacing_separator(ch: str) -> bool:
    """Return whether *ch* can separate single-letter obfuscation tokens."""
    if ch in {"\n", "\r", "\u2028", "\u2029"} or ch.isalnum():
        return False
    category = unicodedata.category(ch)
    return (
        ch.isspace()
        or category.startswith(("P", "S", "Z"))
        or _is_unconditionally_ignored(ch)
        or is_default_ignorable(ch)
        or ch == "\ufffd"
    )


def _letter_spacing_gap_signature(gap: str) -> tuple[str, str] | None:
    """Return a stable signature for one unambiguous inter-letter gap."""
    if not gap:
        return None
    if all(ch.isspace() for ch in gap):
        return ("spacing", gap) if len(set(gap)) == 1 else None

    marker = "".join(ch for ch in gap if not ch.isspace())
    if not marker or len(set(marker)) != 1:
        return None
    return ("marked", marker[0])


def _letter_spacing_run_spans(
    text: str,
    check_runtime: Callable[[], None] | None = None,
    *,
    require_consistent_separator_class: bool = True,
) -> Iterator[tuple[int, int]]:
    """Yield maximal runs of six or more separator-delimited single letters."""
    if check_runtime is not None:
        check_runtime()
    # Keep large benign Unicode artifacts on the C-level fast path. The
    # candidate is deliberately broader than the exact scanner below, but it
    # covers Unicode letters and every supported separator without a Python
    # character-by-character pass when no six-letter run can exist.
    if _LETTER_SPACING_CANDIDATE.search(text) is None:
        if check_runtime is not None:
            check_runtime()
        return
    offset = 0
    while offset < len(text):
        if check_runtime is not None and offset % 4096 == 0:
            check_runtime()
        if not text[offset].isalpha() or (offset > 0 and text[offset - 1].isalpha()):
            offset += 1
            continue

        run_start = offset
        last_letter_end = offset + 1
        run_signature: tuple[str, str] | None = None
        letter_count = 1
        cursor = last_letter_end

        while cursor < len(text):
            gap_start = cursor
            while cursor < len(text) and _is_letter_spacing_separator(text[cursor]):
                if check_runtime is not None and cursor % 4096 == 0:
                    check_runtime()
                cursor += 1
            if gap_start == cursor or cursor >= len(text) or not text[cursor].isalpha():
                break

            next_letter_end = cursor + 1
            if next_letter_end < len(text) and text[next_letter_end].isalpha():
                break

            gap_signature = _letter_spacing_gap_signature(text[gap_start:cursor])
            if gap_signature is None:
                break
            if run_signature is None:
                run_signature = gap_signature
            elif require_consistent_separator_class and gap_signature != run_signature:
                break

            letter_count += 1
            last_letter_end = next_letter_end
            cursor = next_letter_end

        if letter_count >= _MIN_LETTER_SPACING_RUN_LETTERS:
            yield run_start, last_letter_end
            offset = last_letter_end
        else:
            offset = run_start + 1


def _concealed_instruction_run_spans(
    text: str,
    check_runtime: Callable[[], None] | None = None,
) -> Iterator[tuple[int, int]]:
    """Yield broad, bounded single-letter runs for security-term evidence only."""
    if check_runtime is not None:
        check_runtime()
    if _CONCEALED_INSTRUCTION_CANDIDATE.search(text) is None:
        if check_runtime is not None:
            check_runtime()
        return

    offset = 0
    while offset < len(text):
        if check_runtime is not None and offset % 4096 == 0:
            check_runtime()
        if not text[offset].isalpha() or (offset > 0 and text[offset - 1].isalpha()):
            offset += 1
            continue

        run_start = offset
        last_letter_end = offset + 1
        letter_count = 1
        cursor = last_letter_end
        while cursor < len(text):
            gap_start = cursor
            while cursor < len(text) and not text[cursor].isalnum():
                if check_runtime is not None and cursor % 4096 == 0:
                    check_runtime()
                cursor += 1
            if gap_start == cursor or cursor >= len(text) or not text[cursor].isalpha():
                break

            letter_count += 1
            last_letter_end = cursor + 1
            cursor = last_letter_end
            if cursor < len(text) and text[cursor].isalpha():
                break

        if letter_count >= _MIN_LETTER_SPACING_RUN_LETTERS:
            yield run_start, last_letter_end
            offset = last_letter_end
        else:
            offset = run_start + 1


def _security_skeleton_piece(character: str) -> str:
    """Return the case-folded NFKC/confusable skeleton for one source character."""
    folded = (
        unicodedata.normalize("NFKC", character).translate(ASCII_CONFUSABLE_SKELETON).casefold()
    )
    # Python's Unicode IGNORECASE treats LATIN CAPITAL I WITH DOT ABOVE as
    # equivalent to ``i``. Mirror that finite equivalence without stripping
    # arbitrary combining marks or accents.
    return "i" if folded == "i\u0307" else folded


def _fold_security_character(character: str) -> str:
    """Return a one-letter ASCII security skeleton, or an empty sentinel."""
    folded = _security_skeleton_piece(character)
    return folded if len(folded) == 1 and folded.isascii() and folded.isalpha() else ""


def _is_normalization_ignored_offset(text: str, offset: int) -> bool:
    return _is_unconditionally_ignored(text[offset]) or _is_contextual_default_ignorable_offset(
        text,
        offset,
    )


def _check_security_runtime(
    check_runtime: Callable[[], None] | None,
    offset: int,
) -> None:
    if check_runtime is not None and offset % 4096 == 0:
        check_runtime()


def _has_security_word_character_before(
    text: str,
    offset: int,
    check_runtime: Callable[[], None] | None = None,
) -> bool:
    cursor = offset
    while (
        cursor > 0
        and text[cursor - 1] not in _LOGICAL_LINE_BREAK_CHARACTERS
        and _is_normalization_ignored_offset(text, cursor - 1)
    ):
        _check_security_runtime(check_runtime, cursor)
        cursor -= 1
    return cursor > 0 and _is_security_word_character(text[cursor - 1])


def _has_security_word_character_after(
    text: str,
    offset: int,
    check_runtime: Callable[[], None] | None = None,
) -> bool:
    cursor = offset
    while (
        cursor < len(text)
        and text[cursor] not in _LOGICAL_LINE_BREAK_CHARACTERS
        and _is_normalization_ignored_offset(text, cursor)
    ):
        _check_security_runtime(check_runtime, cursor)
        cursor += 1
    return cursor < len(text) and _is_security_word_character(text[cursor])


def _match_security_word(
    text: str,
    start: int,
    expected: str,
    check_runtime: Callable[[], None] | None = None,
) -> int | None:
    """Match one whole, separator-free security word through the shared skeleton."""
    if start > 0 and _is_word_character(text[start - 1]):
        return None
    cursor = start
    expected_offset = 0
    while expected_offset < len(expected):
        _check_security_runtime(check_runtime, cursor)
        if cursor >= len(text):
            return None
        if _is_normalization_ignored_offset(text, cursor):
            cursor += 1
            continue
        piece = _security_skeleton_piece(text[cursor])
        if (
            not piece
            or not piece.isascii()
            or not piece.isalpha()
            or not expected.startswith(piece, expected_offset)
        ):
            return None
        expected_offset += len(piece)
        cursor += 1
    while cursor < len(text) and _is_normalization_ignored_offset(text, cursor):
        _check_security_runtime(check_runtime, cursor)
        cursor += 1
    if cursor < len(text) and _is_word_character(text[cursor]):
        return None
    return cursor


def _match_security_word_backward(
    text: str,
    end: int,
    expected: str,
    check_runtime: Callable[[], None] | None = None,
) -> int | None:
    """Match one whole security word ending at *end* without copying source text."""
    if end < len(text) and _is_word_character(text[end]):
        return None
    cursor = end
    while cursor > 0 and _is_normalization_ignored_offset(text, cursor - 1):
        _check_security_runtime(check_runtime, cursor)
        cursor -= 1
    expected_end = len(expected)
    while expected_end > 0:
        _check_security_runtime(check_runtime, cursor)
        if cursor == 0:
            return None
        cursor -= 1
        piece = _security_skeleton_piece(text[cursor])
        if (
            not piece
            or not piece.isascii()
            or not piece.isalpha()
            or not expected[:expected_end].endswith(piece)
        ):
            return None
        expected_end -= len(piece)
    while cursor > 0 and _is_normalization_ignored_offset(text, cursor - 1):
        _check_security_runtime(check_runtime, cursor)
        cursor -= 1
    if cursor > 0 and _is_word_character(text[cursor - 1]):
        return None
    return cursor


def _skip_required_security_whitespace(
    text: str,
    start: int,
    check_runtime: Callable[[], None] | None = None,
) -> int | None:
    cursor = start
    has_whitespace = False
    while cursor < len(text):
        _check_security_runtime(check_runtime, cursor)
        if text[cursor].isspace():
            has_whitespace = True
        elif not _is_normalization_ignored_offset(text, cursor):
            break
        cursor += 1
    return cursor if has_whitespace else None


def _skip_required_security_whitespace_backward(
    text: str,
    end: int,
    check_runtime: Callable[[], None] | None = None,
) -> int | None:
    cursor = end
    has_whitespace = False
    while cursor > 0:
        _check_security_runtime(check_runtime, cursor)
        if text[cursor - 1].isspace():
            has_whitespace = True
        elif not _is_normalization_ignored_offset(text, cursor - 1):
            break
        cursor -= 1
    return cursor if has_whitespace else None


def _obfuscated_instruction_right_context(
    text: str,
    action: str,
    action_end: int,
    check_runtime: Callable[[], None] | None = None,
) -> _ObfuscatedWordMatch | None:
    """Return the end of an existing right-side P1 context, when present."""
    cursor = _skip_required_security_whitespace(text, action_end, check_runtime)
    if cursor is None:
        return None
    gaps: tuple[tuple[int, int], ...] = ()
    requires_targeted = False

    def keep(match: _ObfuscatedWordMatch) -> None:
        nonlocal gaps, requires_targeted
        gaps += match.gaps
        requires_targeted = requires_targeted or match.requires_targeted_reconstruction

    def match_one(start: int, expected_words: tuple[str, ...]) -> _ObfuscatedWordMatch | None:
        for expected in expected_words:
            match = _match_obfuscated_security_word(
                text,
                start,
                expected,
                check_runtime,
            )
            if match is not None:
                return match
        return None

    if action in {"ignore", "disregard", "forget"}:
        all_match = _match_obfuscated_security_word(
            text,
            cursor,
            "all",
            check_runtime,
        )
        if all_match is not None:
            next_cursor = _skip_required_security_whitespace(
                text,
                all_match.end,
                check_runtime,
            )
            if next_cursor is None:
                return None
            keep(all_match)
            cursor = next_cursor

    if action == "ignore":
        previous_match = _match_obfuscated_security_word(
            text,
            cursor,
            "previous",
            check_runtime,
        )
        if previous_match is not None:
            target_start = _skip_required_security_whitespace(
                text,
                previous_match.end,
                check_runtime,
            )
            if target_start is None:
                return None
            target_match = match_one(target_start, ("instruction", "instructions"))
            if target_match is None:
                return None
            keep(previous_match)
            keep(target_match)
            return _ObfuscatedWordMatch(target_match.end, gaps, requires_targeted)

        policy_match = match_one(cursor, ("safety", "security"))
        if policy_match is None:
            return None
        target_start = _skip_required_security_whitespace(
            text,
            policy_match.end,
            check_runtime,
        )
        if target_start is None:
            return None
        target_match = match_one(
            target_start,
            ("rule", "rules", "constraint", "constraints", "guideline", "guidelines"),
        )
        if target_match is None:
            return None
        keep(policy_match)
        keep(target_match)
        return _ObfuscatedWordMatch(target_match.end, gaps, requires_targeted)

    if action == "disregard":
        context_match = match_one(cursor, ("previous", "safety", "security"))
        if context_match is None:
            return None
        keep(context_match)
        return _ObfuscatedWordMatch(context_match.end, gaps, requires_targeted)

    if action == "forget":
        context_match = match_one(cursor, ("previous", "your"))
        if context_match is None:
            return None
        target_start = _skip_required_security_whitespace(
            text,
            context_match.end,
            check_runtime,
        )
        if target_start is None:
            return None
        target_match = match_one(target_start, ("instruction", "instructions"))
        if target_match is None:
            return None
        keep(context_match)
        keep(target_match)
        return _ObfuscatedWordMatch(target_match.end, gaps, requires_targeted)

    expected_contexts = (
        ("safety", "security", "system")
        if action == "override"
        else ("safety", "security", "restriction", "restrictions", "constraint", "constraints")
    )
    context_match = match_one(cursor, expected_contexts)
    if context_match is None:
        return None
    keep(context_match)
    return _ObfuscatedWordMatch(context_match.end, gaps, requires_targeted)


def _obfuscated_instruction_left_context(
    text: str,
    action_start: int,
    check_runtime: Callable[[], None] | None = None,
) -> _ObfuscatedBackwardWordMatch | None:
    """Match the existing ``you must [always]`` P1 form before an action."""
    cursor = _skip_required_security_whitespace_backward(text, action_start, check_runtime)
    if cursor is None:
        return None
    gaps: tuple[tuple[int, int], ...] = ()
    requires_targeted = False

    def keep(match: _ObfuscatedBackwardWordMatch) -> None:
        nonlocal gaps, requires_targeted
        gaps += match.gaps
        requires_targeted = requires_targeted or match.requires_targeted_reconstruction

    always_match = _match_obfuscated_security_word_backward(
        text,
        cursor,
        "always",
        check_runtime,
    )
    if always_match is not None:
        cursor = _skip_required_security_whitespace_backward(
            text,
            always_match.start,
            check_runtime,
        )
        if cursor is None:
            return None
        keep(always_match)

    must_match = _match_obfuscated_security_word_backward(
        text,
        cursor,
        "must",
        check_runtime,
    )
    if must_match is None:
        return None
    cursor = _skip_required_security_whitespace_backward(
        text,
        must_match.start,
        check_runtime,
    )
    if cursor is None:
        return None
    you_match = _match_obfuscated_security_word_backward(
        text,
        cursor,
        "you",
        check_runtime,
    )
    if you_match is None:
        return None
    keep(must_match)
    keep(you_match)
    return _ObfuscatedBackwardWordMatch(
        you_match.start,
        tuple(sorted(gaps)),
        requires_targeted,
    )


def _is_existing_projection_gap_character(character: str) -> bool:
    """Return whether existing normalized/compact views already remove *character*."""
    return (
        _is_unconditionally_ignored(character)
        or is_default_ignorable(character)
        or _is_non_ascii_separator(character)
        or character == "\ufffd"
    )


def _existing_projection_run_end(text: str, start: int) -> int | None:
    """Return the end of a gap run whose DP transition is idempotent."""
    character = text[start]
    if is_default_ignorable(character):
        match = _DEFAULT_IGNORABLE_RUN_PATTERN.match(text, start)
        return match.end() if match is not None else None
    if (
        character in _LOGICAL_LINE_BREAK_CHARACTERS
        or character.isalpha()
        or character.isascii()
        and character.isspace()
        or not _is_existing_projection_gap_character(character)
        or _fold_security_character(character)
    ):
        return None
    match = _REPEATED_CHARACTER_RUN_PATTERN.match(text, start)
    return match.end() if match is not None else None


def _match_obfuscated_security_word(
    text: str,
    start: int,
    expected: str,
    check_runtime: Callable[[], None] | None = None,
) -> _ObfuscatedWordMatch | None:
    """Match one fixed context word while retaining only bounded filler gaps."""
    if _has_security_word_character_before(text, start, check_runtime):
        return None

    cursor = start
    while (
        cursor < len(text)
        and text[cursor] not in _LOGICAL_LINE_BREAK_CHARACTERS
        and _is_normalization_ignored_offset(text, cursor)
    ):
        _check_security_runtime(check_runtime, cursor)
        cursor += 1
    if cursor >= len(text):
        return None

    first_piece = _security_skeleton_piece(text[cursor])
    if (
        not first_piece
        or not first_piece.isascii()
        or not first_piece.isalpha()
        or not expected.startswith(first_piece)
    ):
        return None

    first_offset = len(first_piece)
    first_state = _ObfuscatedIgnoreState(
        start=cursor,
        last_letter_end=cursor + 1,
    )
    if first_offset == len(expected):
        if _has_security_word_character_after(text, cursor + 1, check_runtime):
            return None
        return _ObfuscatedWordMatch(cursor + 1)

    states: dict[int, tuple[_ObfuscatedIgnoreState, ...]] = {first_offset: (first_state,)}
    cursor += 1

    def keep_best(
        target: dict[int, tuple[_ObfuscatedIgnoreState, ...]],
        expected_offset: int,
        candidate: _ObfuscatedIgnoreState,
    ) -> None:
        bucket = list(target.get(expected_offset, ()))
        for index, existing in enumerate(bucket):
            if existing.start != candidate.start:
                continue
            candidate_rank = (
                candidate.last_letter_end,
                candidate.requires_targeted_reconstruction,
            )
            existing_rank = (
                existing.last_letter_end,
                existing.requires_targeted_reconstruction,
            )
            if candidate_rank > existing_rank:
                bucket[index] = candidate
            target[expected_offset] = tuple(bucket)
            return
        bucket.append(candidate)
        target[expected_offset] = tuple(bucket[-2:])

    while states and cursor < len(text):
        _check_security_runtime(check_runtime, cursor)
        character = text[cursor]
        piece = _security_skeleton_piece(character)
        next_states: dict[int, tuple[_ObfuscatedIgnoreState, ...]] = {}
        completions: list[_ObfuscatedWordMatch] = []

        for expected_offset, candidates in states.items():
            for state in candidates:
                if (
                    piece
                    and piece.isascii()
                    and piece.isalpha()
                    and expected.startswith(piece, expected_offset)
                ):
                    has_gap = cursor > state.last_letter_end
                    next_gaps = (
                        state.gaps + ((state.last_letter_end, cursor),) if has_gap else state.gaps
                    )
                    requires_targeted = state.requires_targeted_reconstruction or (
                        has_gap and state.gap_has_targeted_character
                    )
                    next_expected_offset = expected_offset + len(piece)
                    if next_expected_offset == len(expected):
                        completions.append(
                            _ObfuscatedWordMatch(
                                cursor + 1,
                                next_gaps,
                                requires_targeted,
                            )
                        )
                    elif next_expected_offset < len(expected):
                        keep_best(
                            next_states,
                            next_expected_offset,
                            _ObfuscatedIgnoreState(
                                start=state.start,
                                last_letter_end=cursor + 1,
                                gaps=next_gaps,
                                requires_targeted_reconstruction=requires_targeted,
                            ),
                        )

                if not character.isalpha() or _is_existing_projection_gap_character(character):
                    existing_projection = _is_existing_projection_gap_character(character)
                    has_existing_projection = (
                        state.gap_has_existing_projection or existing_projection
                    )
                    has_ascii_whitespace = state.gap_has_ascii_whitespace or (
                        character.isascii() and character.isspace()
                    )
                    crosses_line = character in _LOGICAL_LINE_BREAK_CHARACTERS
                    if not crosses_line and not (has_existing_projection and has_ascii_whitespace):
                        keep_best(
                            next_states,
                            expected_offset,
                            _ObfuscatedIgnoreState(
                                start=state.start,
                                last_letter_end=state.last_letter_end,
                                gaps=state.gaps,
                                requires_targeted_reconstruction=(
                                    state.requires_targeted_reconstruction
                                ),
                                gap_has_existing_projection=has_existing_projection,
                                gap_has_ascii_whitespace=has_ascii_whitespace,
                                gap_has_targeted_character=(
                                    state.gap_has_targeted_character or not existing_projection
                                ),
                            ),
                        )

        for completion in completions:
            if not _has_security_word_character_after(text, completion.end, check_runtime):
                return completion
        states = next_states
        cursor += 1
    return None


def _match_obfuscated_security_word_backward(
    text: str,
    end: int,
    expected: str,
    check_runtime: Callable[[], None] | None = None,
) -> _ObfuscatedBackwardWordMatch | None:
    """Match one fixed context word backward with the same bounded gap rules."""
    if _has_security_word_character_after(text, end, check_runtime):
        return None

    cursor = end
    while (
        cursor > 0
        and text[cursor - 1] not in _LOGICAL_LINE_BREAK_CHARACTERS
        and _is_normalization_ignored_offset(text, cursor - 1)
    ):
        _check_security_runtime(check_runtime, cursor)
        cursor -= 1
    if cursor == 0:
        return None

    cursor -= 1
    first_piece = _security_skeleton_piece(text[cursor])
    if (
        not first_piece
        or not first_piece.isascii()
        or not first_piece.isalpha()
        or not expected.endswith(first_piece)
    ):
        return None

    expected_end = len(expected) - len(first_piece)
    first_state = _ObfuscatedBackwardState(last_letter_start=cursor)
    if expected_end == 0:
        if _has_security_word_character_before(text, cursor, check_runtime):
            return None
        return _ObfuscatedBackwardWordMatch(cursor)

    states: dict[int, tuple[_ObfuscatedBackwardState, ...]] = {expected_end: (first_state,)}
    cursor -= 1

    def keep_best(
        target: dict[int, tuple[_ObfuscatedBackwardState, ...]],
        target_end: int,
        candidate: _ObfuscatedBackwardState,
    ) -> None:
        bucket = list(target.get(target_end, ()))
        candidate_rank = (
            -candidate.last_letter_start,
            candidate.requires_targeted_reconstruction,
        )
        for index, existing in enumerate(bucket):
            existing_rank = (
                -existing.last_letter_start,
                existing.requires_targeted_reconstruction,
            )
            if candidate_rank > existing_rank:
                bucket[index] = candidate
            target[target_end] = tuple(bucket)
            return
        bucket.append(candidate)
        target[target_end] = tuple(bucket[-2:])

    while states and cursor >= 0:
        _check_security_runtime(check_runtime, cursor)
        character = text[cursor]
        piece = _security_skeleton_piece(character)
        next_states: dict[int, tuple[_ObfuscatedBackwardState, ...]] = {}
        completions: list[_ObfuscatedBackwardWordMatch] = []

        for target_end, candidates in states.items():
            for state in candidates:
                if (
                    piece
                    and piece.isascii()
                    and piece.isalpha()
                    and expected[:target_end].endswith(piece)
                ):
                    has_gap = cursor + 1 < state.last_letter_start
                    next_gaps = (
                        ((cursor + 1, state.last_letter_start),) + state.gaps
                        if has_gap
                        else state.gaps
                    )
                    requires_targeted = state.requires_targeted_reconstruction or (
                        has_gap and state.gap_has_targeted_character
                    )
                    next_target_end = target_end - len(piece)
                    if next_target_end == 0:
                        completions.append(
                            _ObfuscatedBackwardWordMatch(
                                cursor,
                                next_gaps,
                                requires_targeted,
                            )
                        )
                    elif next_target_end > 0:
                        keep_best(
                            next_states,
                            next_target_end,
                            _ObfuscatedBackwardState(
                                last_letter_start=cursor,
                                gaps=next_gaps,
                                requires_targeted_reconstruction=requires_targeted,
                            ),
                        )

                if not character.isalpha() or _is_existing_projection_gap_character(character):
                    existing_projection = _is_existing_projection_gap_character(character)
                    has_existing_projection = (
                        state.gap_has_existing_projection or existing_projection
                    )
                    has_ascii_whitespace = state.gap_has_ascii_whitespace or (
                        character.isascii() and character.isspace()
                    )
                    crosses_line = character in _LOGICAL_LINE_BREAK_CHARACTERS
                    if not crosses_line and not (has_existing_projection and has_ascii_whitespace):
                        keep_best(
                            next_states,
                            target_end,
                            _ObfuscatedBackwardState(
                                last_letter_start=state.last_letter_start,
                                gaps=state.gaps,
                                requires_targeted_reconstruction=(
                                    state.requires_targeted_reconstruction
                                ),
                                gap_has_existing_projection=has_existing_projection,
                                gap_has_ascii_whitespace=has_ascii_whitespace,
                                gap_has_targeted_character=(
                                    state.gap_has_targeted_character or not existing_projection
                                ),
                            ),
                        )

        for completion in completions:
            if not _has_security_word_character_before(text, completion.start, check_runtime):
                return completion
        states = next_states
        cursor -= 1
    return None


def _obfuscated_action_completions(
    text: str,
    check_runtime: Callable[[], None] | None = None,
) -> Iterator[tuple[str, int, int, tuple[tuple[int, int], ...], bool]]:
    """Yield fixed-action DP reconstructions in one forward source pass.

    A non-alphabetic character may be both filler and a Unicode skeleton
    letter (for example, a symbol resembling ``e``). All existing P1 action
    verbs are folded into the same automaton. The earliest and latest source
    starts are retained at each action/offset: the former preserves a valid
    left context while the latter recovers after an unsafe earlier gap. The
    action set and state count are fixed, so time is linear and auxiliary state
    is constant.
    """
    states: dict[tuple[str, int], tuple[_ObfuscatedIgnoreState, ...]] = {}
    cursor = 0
    action_start_pattern = (
        _ASCII_OBFUSCATED_ACTION_START_PATTERN
        if text.isascii()
        else _OBFUSCATED_ACTION_START_PATTERN
    )
    if check_runtime is not None:
        check_runtime()

    def keep_extremes(
        target: dict[tuple[str, int], tuple[_ObfuscatedIgnoreState, ...]],
        state_key: tuple[str, int],
        candidate: _ObfuscatedIgnoreState,
    ) -> None:
        bucket = list(target.get(state_key, ()))
        for index, existing in enumerate(bucket):
            if existing.start != candidate.start:
                continue
            candidate_rank = (
                candidate.last_letter_end,
                candidate.requires_targeted_reconstruction,
            )
            existing_rank = (
                existing.last_letter_end,
                existing.requires_targeted_reconstruction,
            )
            if candidate_rank > existing_rank:
                bucket[index] = candidate
            target[state_key] = tuple(bucket)
            return

        bucket.append(candidate)
        if len(bucket) > 2:
            bucket = [
                min(bucket, key=lambda state: state.start),
                max(bucket, key=lambda state: state.start),
            ]
        target[state_key] = tuple(sorted(bucket, key=lambda state: state.start))

    while cursor < len(text):
        if not states:
            candidate = action_start_pattern.search(text, cursor)
            if check_runtime is not None:
                check_runtime()
            if candidate is None:
                return
            cursor = candidate.start()

        _check_security_runtime(check_runtime, cursor)
        character = text[cursor]
        run_end = _existing_projection_run_end(text, cursor) if states else None
        skeleton = _fold_security_character(character)
        next_states: dict[tuple[str, int], tuple[_ObfuscatedIgnoreState, ...]] = {}
        completions: list[tuple[str, int, int, tuple[tuple[int, int], ...], bool]] = []

        for (expected, expected_offset), candidates in states.items():
            for state in candidates:
                if skeleton != expected[expected_offset]:
                    pass
                else:
                    has_gap = cursor > state.last_letter_end
                    next_gaps = (
                        state.gaps + ((state.last_letter_end, cursor),) if has_gap else state.gaps
                    )
                    requires_targeted = state.requires_targeted_reconstruction or (
                        has_gap and state.gap_has_targeted_character
                    )
                    next_expected_offset = expected_offset + 1
                    if next_expected_offset == len(expected):
                        completions.append(
                            (
                                expected,
                                state.start,
                                cursor + 1,
                                next_gaps,
                                requires_targeted,
                            )
                        )
                    else:
                        keep_extremes(
                            next_states,
                            (expected, next_expected_offset),
                            _ObfuscatedIgnoreState(
                                start=state.start,
                                last_letter_end=cursor + 1,
                                gaps=next_gaps,
                                requires_targeted_reconstruction=requires_targeted,
                            ),
                        )

                if not character.isalpha() or _is_existing_projection_gap_character(character):
                    existing_projection = _is_existing_projection_gap_character(character)
                    has_existing_projection = (
                        state.gap_has_existing_projection or existing_projection
                    )
                    has_ascii_whitespace = state.gap_has_ascii_whitespace or (
                        character.isascii() and character.isspace()
                    )
                    crosses_line = character in _LOGICAL_LINE_BREAK_CHARACTERS
                    if not crosses_line and not (has_existing_projection and has_ascii_whitespace):
                        keep_extremes(
                            next_states,
                            (expected, expected_offset),
                            _ObfuscatedIgnoreState(
                                start=state.start,
                                last_letter_end=state.last_letter_end,
                                gaps=state.gaps,
                                requires_targeted_reconstruction=(
                                    state.requires_targeted_reconstruction
                                ),
                                gap_has_existing_projection=has_existing_projection,
                                gap_has_ascii_whitespace=has_ascii_whitespace,
                                gap_has_targeted_character=(
                                    state.gap_has_targeted_character or not existing_projection
                                ),
                            ),
                        )

        if skeleton in _OBFUSCATED_ACTION_INITIALS and not _has_security_word_character_before(
            text,
            cursor,
            check_runtime,
        ):
            for expected in _OBFUSCATED_INSTRUCTION_ACTIONS:
                if skeleton == expected[0]:
                    keep_extremes(
                        next_states,
                        (expected, 1),
                        _ObfuscatedIgnoreState(
                            start=cursor,
                            last_letter_end=cursor + 1,
                        ),
                    )

        states = next_states
        cursor = run_end if run_end is not None else cursor + 1
        yield from completions


def _obfuscated_instruction_matches(
    text: str,
    check_runtime: Callable[[], None] | None = None,
) -> Iterator[_ObfuscatedInstructionMatch]:
    """Yield context-bound P1 phrases with non-alpha inter-letter fillers.

    Fixed action and context state machines avoid global digit stripping:
    opaque identifiers remain unchanged unless the reconstructed phrase is an
    existing P1 instruction-override form. Each source character is visited a
    constant number of times and only bounded gap ranges are retained.
    """
    accepted_until = 0
    for (
        action,
        action_start,
        action_end,
        action_gaps,
        action_requires_targeted,
    ) in _obfuscated_action_completions(
        text,
        check_runtime,
    ):
        if action_start < accepted_until or (
            _has_security_word_character_after(text, action_end, check_runtime)
        ):
            continue

        context_match = _obfuscated_instruction_right_context(
            text,
            action,
            action_end,
            check_runtime,
        )
        if context_match is None:
            if action != "ignore":
                continue
            left_context_match = _obfuscated_instruction_left_context(
                text,
                action_start,
                check_runtime,
            )
            if left_context_match is None:
                continue
            if not (
                action_requires_targeted or left_context_match.requires_targeted_reconstruction
            ):
                continue
            match_end = action_end
            gaps = tuple(sorted(left_context_match.gaps + action_gaps))
        else:
            if not (action_requires_targeted or context_match.requires_targeted_reconstruction):
                continue
            match_end = context_match.end
            gaps = action_gaps + context_match.gaps

        yield _ObfuscatedInstructionMatch(
            start=action_start,
            end=match_end,
            gaps=gaps,
            evidence_offset=next(
                offset
                for gap_start, gap_end in gaps
                for offset in range(gap_start, gap_end)
                if not _is_existing_projection_gap_character(text[offset])
            ),
        )
        accepted_until = match_end


def _obfuscated_instruction_gap_offsets(text: str) -> Iterator[int]:
    """Yield filler offsets only for context-bound obfuscated instructions."""
    for match in _obfuscated_instruction_matches(text):
        for start, end in match.gaps:
            yield from range(start, end)


def _has_letter_spacing_run(text: str) -> bool:
    """Use a C-level ASCII prefilter before the exact Unicode-aware scan."""
    return next(_letter_spacing_run_spans(text), None) is not None


def _letter_spacing_gap_offsets(text: str) -> Iterator[int]:
    """Yield only the separator offsets inside confirmed letter-spacing runs."""
    for start, end in _letter_spacing_run_spans(text):
        for offset in range(start, end):
            if _is_letter_spacing_separator(text[offset]):
                yield offset


def _is_token_gap_character(ch: str) -> bool:
    return (
        _is_unconditionally_ignored(ch)
        or is_default_ignorable(ch)
        or _is_non_ascii_separator(ch)
        or ch == "\ufffd"
    )


def _token_bridging_gap_spans(
    text: str,
    *,
    require_word_boundaries: bool = True,
    check_runtime: Callable[[], None] | None = None,
) -> Iterator[tuple[int, int]]:
    """Yield contextual noise runs in one pass without crossing ASCII spaces."""
    offset = 0
    while offset < len(text):
        if check_runtime is not None and offset % 4096 == 0:
            check_runtime()
        if not _is_token_gap_character(text[offset]):
            offset += 1
            continue
        start = offset
        while offset < len(text) and _is_token_gap_character(text[offset]):
            if check_runtime is not None and offset % 4096 == 0:
                check_runtime()
            if is_default_ignorable(text[offset]):
                run = _DEFAULT_IGNORABLE_RUN_PATTERN.match(text, offset)
                if run is not None:
                    offset = run.end()
                    continue
            offset += 1
        before_is_word = start > 0 and _is_word_character(text[start - 1])
        after_is_word = offset < len(text) and _is_word_character(text[offset])
        is_contextual = (
            before_is_word and after_is_word
            if require_word_boundaries
            else before_is_word or after_is_word
        )
        if is_contextual:
            yield start, offset


def _is_contextual_default_ignorable_offset(text: str, offset: int) -> bool:
    """Return whether one offset is an ignorable outside an emoji presentation form."""
    ch = text[offset]
    if not is_default_ignorable(ch) or _is_unconditionally_ignored(ch):
        return False
    previous = text[offset - 1] if offset else ""
    following = text[offset + 1] if offset + 1 < len(text) else ""
    return not (
        0xFE00 <= ord(ch) <= 0xFE0F
        and (
            previous
            and _is_emoji_base(previous)
            or following
            and unicodedata.category(following) == "Me"
        )
    )


def _contextual_default_ignorable_spans(text: str) -> Iterator[tuple[int, int]]:
    """Yield removable ignorable spans without walking homogeneous runs in Python."""
    for gap_start, gap_end in _token_bridging_gap_spans(
        text,
        require_word_boundaries=False,
    ):
        for match in _DEFAULT_IGNORABLE_RUN_PATTERN.finditer(text, gap_start, gap_end):
            start, end = match.span()
            character = text[start]
            if text.count(character, start, end) == end - start:
                if _is_unconditionally_ignored(character):
                    continue
                contextual_start = (
                    start if _is_contextual_default_ignorable_offset(text, start) else start + 1
                )
                contextual_end = (
                    end if _is_contextual_default_ignorable_offset(text, end - 1) else end - 1
                )
                if contextual_start < contextual_end:
                    yield contextual_start, contextual_end
                continue

            span_start: int | None = None
            for offset in range(start, end):
                if _is_contextual_default_ignorable_offset(text, offset):
                    if span_start is None:
                        span_start = offset
                elif span_start is not None:
                    yield span_start, offset
                    span_start = None
            if span_start is not None:
                yield span_start, end


def _normalization_ignored_spans(text: str) -> Iterator[tuple[int, int]]:
    """Yield whole default-ignorable runs removable by the normalized view."""
    for gap_start, gap_end in _token_bridging_gap_spans(
        text,
        require_word_boundaries=False,
    ):
        for match in _DEFAULT_IGNORABLE_RUN_PATTERN.finditer(text, gap_start, gap_end):
            start, end = match.span()
            ignored_start = (
                start
                if _is_unconditionally_ignored(text[start])
                or _is_contextual_default_ignorable_offset(text, start)
                else start + 1
            )
            ignored_end = (
                end
                if _is_unconditionally_ignored(text[end - 1])
                or _is_contextual_default_ignorable_offset(text, end - 1)
                else end - 1
            )
            if ignored_start < ignored_end:
                yield ignored_start, ignored_end


def _contextual_default_ignorable_offsets(text: str) -> Iterator[int]:
    """Yield non-format default-ignorables next to text without altering emoji forms."""
    for start, end in _contextual_default_ignorable_spans(text):
        yield from range(start, end)


def _contextual_default_ignorable_boundary_spans(
    text: str,
    check_runtime: Callable[[], None] | None = None,
) -> Iterator[tuple[int, int]]:
    """Yield token-boundary gaps containing a non-emoji ignorable.

    A token-boundary gap has a word character on exactly one side. This keeps
    whole-token concealment evidence separate from in-token normalization.
    """
    if not _contains_default_ignorable(text):
        if check_runtime is not None:
            check_runtime()
        return

    for start, end in _token_bridging_gap_spans(
        text,
        require_word_boundaries=False,
        check_runtime=check_runtime,
    ):
        before_is_word = start > 0 and _is_word_character(text[start - 1])
        after_is_word = end < len(text) and _is_word_character(text[end])
        if before_is_word == after_is_word:
            continue
        for offset in range(start, end):
            if check_runtime is not None and offset % 4096 == 0:
                check_runtime()
            if _is_contextual_default_ignorable_offset(text, offset):
                yield start, end
                break


def _compact_gap_offsets(text: str) -> Iterator[int]:
    """Yield word-bounded separator runs that the compact view may remove."""
    for start, end in _token_bridging_gap_spans(text):
        character = text[start]
        if text.count(character, start, end) == end - start:
            if _is_non_ascii_separator(character):
                yield from range(start, end)
            continue
        if any(_is_non_ascii_separator(text[offset]) for offset in range(start, end)):
            yield from range(start, end)


def _next_offset(offsets: Iterator[int]) -> int | None:
    return next(offsets, None)


def normalized_security_view(text: str) -> SecurityTextView:
    """Build an NFKC/UTS #39 ASCII-skeleton view with compact offsets."""
    output = StringIO()
    offsets = array("I")
    contextual_spans = iter(_normalization_ignored_spans(text))
    next_contextual = next(contextual_spans, None)
    source_offset = 0
    while source_offset < len(text):
        if next_contextual is not None and source_offset == next_contextual[0]:
            source_offset = next_contextual[1]
            next_contextual = next(contextual_spans, None)
            continue
        ch = text[source_offset]
        if _is_unconditionally_ignored(ch):
            source_offset += 1
            continue
        normalized = unicodedata.normalize("NFKC", ch).translate(ASCII_CONFUSABLE_SKELETON)
        for normalized_char in normalized:
            output.write(normalized_char)
            offsets.append(source_offset)
        source_offset += 1
    return SecurityTextView("normalized", output.getvalue(), offsets)


def obfuscated_instruction_view(text: str) -> SecurityTextView:
    """Normalize text while removing only context-bound instruction fillers."""
    output = StringIO()
    offsets = array("I")
    contextual_offsets = iter(_contextual_default_ignorable_offsets(text))
    instruction_offsets = iter(_obfuscated_instruction_gap_offsets(text))
    next_contextual = _next_offset(contextual_offsets)
    next_instruction = _next_offset(instruction_offsets)
    for source_offset, ch in enumerate(text):
        is_contextual = source_offset == next_contextual
        is_instruction = source_offset == next_instruction
        if is_contextual:
            next_contextual = _next_offset(contextual_offsets)
        if is_instruction:
            next_instruction = _next_offset(instruction_offsets)
        if (
            _is_unconditionally_ignored(ch)
            and ch not in _LOGICAL_LINE_BREAK_CHARACTERS
            or is_contextual
            or is_instruction
        ):
            continue
        normalized = unicodedata.normalize("NFKC", ch).translate(ASCII_CONFUSABLE_SKELETON)
        for normalized_char in normalized:
            output.write(normalized_char)
            offsets.append(source_offset)
    return SecurityTextView("obfuscated-instruction", output.getvalue(), offsets)


def compact_letter_view(text: str) -> SecurityTextView:
    """Remove compact binary/format noise between letters without joining words."""
    output = StringIO()
    offsets = array("I")
    contextual_offsets = iter(_contextual_default_ignorable_offsets(text))
    compact_offsets = iter(_compact_gap_offsets(text))
    letter_spacing_offsets = iter(_letter_spacing_gap_offsets(text))
    obfuscated_instruction_offsets = iter(_obfuscated_instruction_gap_offsets(text))
    next_contextual = _next_offset(contextual_offsets)
    next_compact = _next_offset(compact_offsets)
    next_letter_spacing = _next_offset(letter_spacing_offsets)
    next_obfuscated_instruction = _next_offset(obfuscated_instruction_offsets)
    for source_offset, ch in enumerate(text):
        is_contextual = source_offset == next_contextual
        is_compact = source_offset == next_compact
        is_letter_spacing = source_offset == next_letter_spacing
        is_obfuscated_instruction = source_offset == next_obfuscated_instruction
        if is_contextual:
            next_contextual = _next_offset(contextual_offsets)
        if is_compact:
            next_compact = _next_offset(compact_offsets)
        if is_letter_spacing:
            next_letter_spacing = _next_offset(letter_spacing_offsets)
        if is_obfuscated_instruction:
            next_obfuscated_instruction = _next_offset(obfuscated_instruction_offsets)
        if (
            _is_unconditionally_ignored(ch)
            or ch == "\ufffd"
            or is_contextual
            or is_compact
            or is_letter_spacing
            or is_obfuscated_instruction
        ):
            continue
        normalized = unicodedata.normalize("NFKC", ch).translate(ASCII_CONFUSABLE_SKELETON)
        for normalized_char in normalized:
            output.write(normalized_char)
            offsets.append(source_offset)
    return SecurityTextView("compact", output.getvalue(), offsets)


def prompt_injection_letter_spacing_view(
    text: str,
    check_runtime: Callable[[], None] | None = None,
    *,
    preserve_identifier_boundaries: bool = True,
) -> SecurityTextView:
    """Collapse ASCII-spaced tokens without inventing word boundaries.

    The prompt-injection analyzer alone consumes this projection. Existing
    multi-space word boundaries are retained verbatim, while one-space token
    interiors are removed with exact raw offsets. A completely boundary-free
    run therefore remains one condensed token and cannot acquire a guessed P3
    or P4 segmentation. Identifier-adjacent runs are preserved for semantic
    classification; artifact-integrity may opt into their projection solely to
    produce an ambiguity finding.
    """
    if check_runtime is not None:
        check_runtime()
    if _PROMPT_SPACED_PAIR.search(text) is None:
        return SecurityTextView("prompt-letter-spacing", text)
    processed_since_check = 0

    def record_work(characters: int = 1) -> None:
        nonlocal processed_since_check
        if check_runtime is None:
            return
        processed_since_check += characters
        if processed_since_check >= 4096:
            check_runtime()
            processed_since_check %= 4096

    output = StringIO()
    offsets = array("I")
    transformed = False

    def append_source(start: int, end: int) -> None:
        for source_offset in range(start, end):
            record_work()
            output.write(text[source_offset])
            offsets.append(source_offset)

    cursor = 0
    index = 0
    while index < len(text):
        record_work()
        if (
            text[index].isspace()
            or index + 2 >= len(text)
            or text[index + 1] != " "
            or text[index + 2].isspace()
        ):
            index += 1
            continue

        run_start = index
        run_end = index + 1
        while run_end + 1 < len(text) and text[run_end] == " " and not text[run_end + 1].isspace():
            run_end += 2
            record_work(2)

        # Do not rewrite a letter-spaced fragment embedded in an identifier.
        # Advance past rejected runs too, so attacker-controlled near misses
        # cannot force quadratic rescanning from every interior character.
        left_identifier = run_start > 0 and (
            text[run_start - 1].isascii()
            and (text[run_start - 1].isalnum() or text[run_start - 1] == "_")
        )
        right_identifier = run_end < len(text) and (
            text[run_end].isascii() and (text[run_end].isalnum() or text[run_end] == "_")
        )
        left_letter = (
            run_start > 0 and text[run_start - 1].isascii() and text[run_start - 1].isalpha()
        )
        right_letter = run_end < len(text) and text[run_end].isascii() and text[run_end].isalpha()
        boundary_is_safe = (
            not left_identifier and not right_identifier
            if preserve_identifier_boundaries
            else not left_letter and not right_letter
        )
        if boundary_is_safe:
            append_source(cursor, run_start)
            for source_offset in range(run_start, run_end, 2):
                record_work(2)
                output.write(text[source_offset])
                offsets.append(source_offset)
            cursor = run_end
            transformed = True
        index = run_end

    if not transformed:
        return SecurityTextView("prompt-letter-spacing", text)
    append_source(cursor, len(text))
    return SecurityTextView("prompt-letter-spacing", output.getvalue(), offsets)


def _requires_normalized_security_view(text: str) -> bool:
    """Return whether normalization can produce a distinct security view."""
    if _IGNORED_ASCII_CONTROL.search(text) is not None:
        return True
    if _contains_default_ignorable(text):
        return True
    if not unicodedata.is_normalized("NFKC", text):
        return True
    if _ASCII_CONFUSABLE_PATTERN.search(text) is not None:
        return True
    if text.isprintable():
        return False
    # Newline, carriage return, and tab are retained unchanged by the
    # projection. Any other non-printable character still needs the exact
    # category-aware path in ``normalized_security_view``.
    return not text.translate(_REMOVE_ALLOWED_FORMAT_CHARACTERS).isprintable()


def security_text_views(text: str) -> tuple[SecurityTextView, ...]:
    """Return distinct raw, normalized, and compact views deterministically."""
    raw = SecurityTextView("raw", text)
    has_letter_spacing = _has_letter_spacing_run(text)
    has_obfuscated_instruction = next(_obfuscated_instruction_matches(text), None) is not None
    if (
        text.isascii()
        and _IGNORED_ASCII_CONTROL.search(text) is None
        and not has_letter_spacing
        and not has_obfuscated_instruction
    ):
        return (raw,)
    unique = [raw]
    seen = {text}
    builders: list[Callable[[str], SecurityTextView]] = []
    if _requires_normalized_security_view(text):
        builders.append(normalized_security_view)
    if (
        "\ufffd" in text
        or _next_offset(iter(_compact_gap_offsets(text))) is not None
        or has_letter_spacing
        or has_obfuscated_instruction
    ):
        builders.append(compact_letter_view)
    if has_obfuscated_instruction:
        builders.append(obfuscated_instruction_view)
    for build_view in builders:
        view = build_view(text)
        if view.text not in seen:
            seen.add(view.text)
            unique.append(view)
    return tuple(unique)


def unicode_anomaly_density(text: str) -> float:
    """Return the density of format controls and token-bridging ignorables."""
    if not text:
        return 0.0
    contextual_offsets = iter(_contextual_default_ignorable_offsets(text))
    next_contextual = _next_offset(contextual_offsets)
    ignored = 0
    for offset, ch in enumerate(text):
        is_contextual = offset == next_contextual
        if is_contextual:
            next_contextual = _next_offset(contextual_offsets)
        ignored += _is_unconditionally_ignored(ch) or is_contextual
    return ignored / len(text)


def has_mixed_script_token(text: str) -> bool:
    """Detect bounded tokens that combine ASCII with Greek/Cyrillic letters."""
    token_scripts: set[str] = set()
    for ch in text:
        if ch.isascii() and ch.isalpha():
            token_scripts.add("latin")
        elif ch.isalpha():
            name = unicodedata.name(ch, "")
            if "CYRILLIC" in name:
                token_scripts.add("cyrillic")
            elif "GREEK" in name:
                token_scripts.add("greek")
        elif ch.isalnum() or ch in {"_", "-"}:
            continue
        else:
            if "latin" in token_scripts and len(token_scripts) > 1:
                return True
            token_scripts.clear()
    return "latin" in token_scripts and len(token_scripts) > 1
