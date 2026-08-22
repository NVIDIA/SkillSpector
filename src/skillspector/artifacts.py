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
_DEFAULT_IGNORABLE_PATTERN = re.compile(
    "["
    + "".join(
        re.escape(chr(start)) if start == end else f"{re.escape(chr(start))}-{re.escape(chr(end))}"
        for start, end in _DEFAULT_IGNORABLE_RANGES
    )
    + "]"
)
_ASCII_CONFUSABLE_PATTERN = re.compile(
    "[" + "".join(re.escape(chr(codepoint)) for codepoint in ASCII_CONFUSABLE_SKELETON) + "]"
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


def is_default_ignorable(ch: str) -> bool:
    """Return the pinned Unicode Default_Ignorable_Code_Point property."""
    codepoint = ord(ch)
    for start, end in _DEFAULT_IGNORABLE_RANGES:
        if codepoint < start:
            return False
        if codepoint <= end:
            return True
    return False


def _is_unconditionally_ignored(ch: str) -> bool:
    return (
        bool(_IGNORED_ASCII_CONTROL.fullmatch(ch))
        or unicodedata.category(ch) in {"Cf", "Cc"}
        and ch not in _ALLOWED_FORMAT_CHARS
    )


def _is_word_character(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


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
) -> Iterator[tuple[int, int]]:
    """Yield word-bounded noise runs in one pass without crossing ASCII spaces."""
    offset = 0
    while offset < len(text):
        if not _is_token_gap_character(text[offset]):
            offset += 1
            continue
        start = offset
        while offset < len(text) and _is_token_gap_character(text[offset]):
            offset += 1
        if (
            start > 0
            and offset < len(text)
            and _is_word_character(text[start - 1])
            and _is_word_character(text[offset])
        ):
            yield start, offset


def _contextual_default_ignorable_offsets(text: str) -> Iterator[int]:
    """Yield non-format default-ignorables only when they bridge word tokens."""
    for start, end in _token_bridging_gap_spans(text):
        for offset in range(start, end):
            ch = text[offset]
            if is_default_ignorable(ch) and not _is_unconditionally_ignored(ch):
                yield offset


def _compact_gap_offsets(text: str) -> Iterator[int]:
    """Yield word-bounded separator runs that the compact view may remove."""
    for start, end in _token_bridging_gap_spans(text):
        if any(_is_non_ascii_separator(text[offset]) for offset in range(start, end)):
            yield from range(start, end)


def _next_offset(offsets: Iterator[int]) -> int | None:
    return next(offsets, None)


def normalized_security_view(text: str) -> SecurityTextView:
    """Build an NFKC/UTS #39 ASCII-skeleton view with compact offsets."""
    output = StringIO()
    offsets = array("I")
    contextual_offsets = iter(_contextual_default_ignorable_offsets(text))
    next_contextual = _next_offset(contextual_offsets)
    for source_offset, ch in enumerate(text):
        is_contextual = source_offset == next_contextual
        if is_contextual:
            next_contextual = _next_offset(contextual_offsets)
        if _is_unconditionally_ignored(ch) or is_contextual:
            continue
        normalized = unicodedata.normalize("NFKC", ch).translate(ASCII_CONFUSABLE_SKELETON)
        for normalized_char in normalized:
            output.write(normalized_char)
            offsets.append(source_offset)
    return SecurityTextView("normalized", output.getvalue(), offsets)


def compact_letter_view(text: str) -> SecurityTextView:
    """Remove compact binary/format noise between letters without joining words."""
    output = StringIO()
    offsets = array("I")
    contextual_offsets = iter(_contextual_default_ignorable_offsets(text))
    compact_offsets = iter(_compact_gap_offsets(text))
    letter_spacing_offsets = iter(_letter_spacing_gap_offsets(text))
    next_contextual = _next_offset(contextual_offsets)
    next_compact = _next_offset(compact_offsets)
    next_letter_spacing = _next_offset(letter_spacing_offsets)
    for source_offset, ch in enumerate(text):
        is_contextual = source_offset == next_contextual
        is_compact = source_offset == next_compact
        is_letter_spacing = source_offset == next_letter_spacing
        if is_contextual:
            next_contextual = _next_offset(contextual_offsets)
        if is_compact:
            next_compact = _next_offset(compact_offsets)
        if is_letter_spacing:
            next_letter_spacing = _next_offset(letter_spacing_offsets)
        if (
            _is_unconditionally_ignored(ch)
            or ch == "\ufffd"
            or is_contextual
            or is_compact
            or is_letter_spacing
        ):
            continue
        normalized = unicodedata.normalize("NFKC", ch).translate(ASCII_CONFUSABLE_SKELETON)
        for normalized_char in normalized:
            output.write(normalized_char)
            offsets.append(source_offset)
    return SecurityTextView("compact", output.getvalue(), offsets)


def _requires_normalized_security_view(text: str) -> bool:
    """Return whether normalization can produce a distinct security view."""
    if _IGNORED_ASCII_CONTROL.search(text) is not None:
        return True
    if _DEFAULT_IGNORABLE_PATTERN.search(text) is not None:
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
    if text.isascii() and _IGNORED_ASCII_CONTROL.search(text) is None and not has_letter_spacing:
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
    ):
        builders.append(compact_letter_view)
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
