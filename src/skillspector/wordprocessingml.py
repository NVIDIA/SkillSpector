# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identify typographic contractions in literal WordprocessingML paragraphs."""

from __future__ import annotations

import re
from collections.abc import Callable
from xml.parsers import expat

_WORD = "http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_STRICT_WORD = "http://purl.oclc.org/ooxml/wordprocessingml/main}"
_CONTRACTION = re.compile(r"(?i)\b[a-z]+(')(?:t|re|ve|ll|d|s|m)\b")
_PLAIN_PROSE = re.compile(r"[A-Za-z ,.:?']+")


def paragraph_contractions(
    content: str,
    check_runtime: Callable[[], None],
    *,
    paragraphs: list[tuple[int, int, str]] | None = None,
) -> str:
    """Refine only raw apostrophes inside simple, complete, literal paragraphs.

    Rejoin styled text runs before testing the text. Markup, decoded entities,
    fields, instructions and command punctuation never establish ownership.
    All original text remains present in the independent finding views.
    """
    if len(content) > 1_000_000 or "wordprocessingml/" not in content[:8192]:
        return content
    parser = expat.ParserCreate(namespace_separator="}")
    namespace = ""
    stack: list[str] = []
    positions: list[int] = []
    pieces: list[tuple[int, bytes]] = []
    eligible = False
    in_paragraph = False
    paragraph_begin = 0
    collected: list[tuple[int, int, str]] = []
    events = 0
    source = content.encode("utf-8", "surrogatepass")

    def reject(*_: object) -> None:
        raise ValueError("unproved WordprocessingML")

    def start(name: str, attrs: dict[str, str]) -> None:
        nonlocal namespace, in_paragraph, eligible, events, paragraph_begin
        events += 1
        if events % 128 == 0:
            check_runtime()
        if events > 100_000 or len(stack) >= 128:
            reject()
        if not stack:
            namespace = next((ns for ns in (_WORD, _STRICT_WORD) if name == ns + "document"), "")
            if not namespace:
                reject()
        if name == namespace + "p":
            if in_paragraph:
                reject()
            in_paragraph = eligible = True
            paragraph_begin = parser.CurrentByteIndex
            pieces.clear()
        elif (
            in_paragraph
            and name not in {namespace + "r", namespace + "t"}
            and not any(
                item in (namespace + "pPr", namespace + "rPr") for item in (*stack[-4:], name)
            )
        ):
            eligible = False
        stack.append(name)

    def characters(text: str) -> None:
        nonlocal eligible
        if not eligible or not in_paragraph or not stack:
            return
        if stack[-1] != namespace + "t":
            if text.strip():
                eligible = False
            return
        if len(stack) < 3 or stack[-3:] != [namespace + "p", namespace + "r", namespace + "t"]:
            eligible = False
            return
        value = text.encode("utf-8")
        begin = parser.CurrentByteIndex
        if not source.startswith(value, begin):
            eligible = False
        else:
            pieces.append((begin, value))

    def end(name: str) -> None:
        nonlocal in_paragraph
        if name == namespace + "p":
            if eligible and pieces:
                value = b"".join(data for _, data in pieces)
                if paragraphs is not None:
                    collected.append(
                        (paragraph_begin, parser.CurrentByteIndex, value.decode("utf-8"))
                    )
                if b"'" in value and len(value) <= 4096 and value.isascii():
                    prose = value.decode("ascii")
                    if _PLAIN_PROSE.fullmatch(prose):
                        quotes = [match.start(1) for match in _CONTRACTION.finditer(prose)]
                        if quotes and len(quotes) == value.count(b"'"):
                            cursor = offset = 0
                            for begin, data in pieces:
                                while cursor < len(quotes) and quotes[cursor] < offset + len(data):
                                    positions.append(begin + quotes[cursor] - offset)
                                    cursor += 1
                                offset += len(data)
            in_paragraph = False
        stack.pop()

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = characters
    parser.StartDoctypeDeclHandler = reject
    parser.ExternalEntityRefHandler = reject
    parser.StartCdataSectionHandler = reject
    parser.ProcessingInstructionHandler = reject
    try:
        for offset in range(0, len(source), 32_768):
            check_runtime()
            parser.Parse(source[offset : offset + 32_768], False)
        parser.Parse(b"", True)
    except (ValueError, expat.ExpatError):
        return content
    if paragraphs is not None:
        previous_byte = previous_char = 0
        for begin, end_offset, value in collected:
            check_runtime()
            # Paragraphs are disjoint and ordered. Decode each interval once
            # to convert Expat's UTF-8 coordinates to source character offsets.
            start_char = previous_char + len(source[previous_byte:begin].decode("utf-8"))
            end_char = start_char + len(source[begin:end_offset].decode("utf-8"))
            paragraphs.append((start_char, end_char, value))
            previous_byte, previous_char = end_offset, end_char
    if not positions:
        return content
    result = bytearray(source)
    for position in positions:
        result[position] = ord("_")
    return result.decode("utf-8")
