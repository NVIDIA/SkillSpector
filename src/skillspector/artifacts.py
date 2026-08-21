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


def _is_ignored_format(ch: str) -> bool:
    return (
        ch == "\u00ad"
        or unicodedata.category(ch) in {"Cf", "Cc"}
        and ch not in _ALLOWED_FORMAT_CHARS
    )


def normalized_security_view(text: str) -> SecurityTextView:
    """Build an NFKC/UTS #39 ASCII-skeleton view with compact offsets."""
    output = StringIO()
    offsets = array("I")
    for source_offset, ch in enumerate(text):
        if _is_ignored_format(ch):
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
    for source_offset, ch in enumerate(text):
        if _is_ignored_format(ch) or ch == "\ufffd":
            continue
        normalized = unicodedata.normalize("NFKC", ch).translate(ASCII_CONFUSABLE_SKELETON)
        for normalized_char in normalized:
            output.write(normalized_char)
            offsets.append(source_offset)
    return SecurityTextView("compact", output.getvalue(), offsets)


def security_text_views(text: str) -> tuple[SecurityTextView, ...]:
    """Return distinct raw, normalized, and compact views deterministically."""
    raw = SecurityTextView("raw", text)
    if text.isascii() and _IGNORED_ASCII_CONTROL.search(text) is None:
        return (raw,)
    unique = [raw]
    seen = {text}
    builders = [normalized_security_view]
    if "\ufffd" in text:
        builders.append(compact_letter_view)
    for build_view in builders:
        view = build_view(text)
        if view.text not in seen:
            seen.add(view.text)
            unique.append(view)
    return tuple(unique)


def unicode_anomaly_density(text: str) -> float:
    """Return the density of soft-hyphen/default-ignorable format characters."""
    if not text:
        return 0.0
    return sum(_is_ignored_format(ch) for ch in text) / len(text)


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
