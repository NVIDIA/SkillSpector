# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Corroborate a deliberately small, bounded subset of passive JPEG structure.

Supported files are thumbnail-free JFIF, eight-bit baseline Huffman JPEGs with
one grayscale or three-component interleaved scan. Entropy codes and the exact
number of coefficient blocks are checked, without IDCT or pixel allocation.
Comments, EXIF, other application data, restart intervals, progressive coding,
multiple scans and unreferenced/redefined tables remain unsupported. This is
structural evidence, not image-content inspection or a general JPEG decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

from skillspector.constants import MAX_ANALYZABLE_FILE_BYTES

_MAX_DIMENSION = 8_192
_MAX_PIXELS = 4 * 1024 * 1024
_MAX_DECODED_BYTES = 16 * 1024 * 1024
_MAX_BLOCKS = 262_144
_MAX_MARKERS = 64
_MAX_ENTROPY_BYTES = 1024 * 1024
_MAX_ENTROPY_BITS = 8 * _MAX_ENTROPY_BYTES

type _Huffman = dict[tuple[int, int], int]


class _InvalidJpegError(ValueError):
    """The input exceeds this validator's supported structural subset."""


@dataclass(frozen=True)
class _Component:
    horizontal: int
    vertical: int
    quantization: int


@dataclass(frozen=True)
class _Frame:
    components: dict[int, _Component]
    mcus: int


@dataclass
class _Bits:
    data: bytearray
    offset: int = 0
    buffer: int = 0
    available: int = 0
    consumed: int = 0

    def read(self, count: int) -> int:
        self.consumed += count
        if self.consumed > _MAX_ENTROPY_BITS:
            raise _InvalidJpegError
        while self.available < count:
            if self.offset == len(self.data):
                raise _InvalidJpegError
            self.buffer = (self.buffer << 8) | self.data[self.offset]
            self.offset += 1
            self.available += 8
        self.available -= count
        value = self.buffer >> self.available
        # Retain at most seven bits, rather than growing an input-sized integer.
        self.buffer &= (1 << self.available) - 1
        return value

    def finished(self) -> bool:
        # JPEG pads the final entropy byte with ones; extra bytes are not padding.
        return self.offset == len(self.data) and self.buffer == (1 << self.available) - 1


def _frame(payload: bytes) -> _Frame:
    if len(payload) < 6 or payload[0] != 8:
        raise _InvalidJpegError
    height = int.from_bytes(payload[1:3], "big")
    width = int.from_bytes(payload[3:5], "big")
    count = payload[5]
    if (
        not 0 < width <= _MAX_DIMENSION
        or not 0 < height <= _MAX_DIMENSION
        or width * height > _MAX_PIXELS
        or count not in {1, 3}
        or width * height * count > _MAX_DECODED_BYTES
        or len(payload) != 6 + 3 * count
    ):
        raise _InvalidJpegError
    components: dict[int, _Component] = {}
    for index in range(6, len(payload), 3):
        identifier, sampling, table = payload[index : index + 3]
        horizontal, vertical = sampling >> 4, sampling & 15
        if (
            identifier in components
            or not 1 <= horizontal <= 4
            or not 1 <= vertical <= 4
            or table > 3
            or (count == 1 and sampling != 0x11)
        ):
            raise _InvalidJpegError
        components[identifier] = _Component(horizontal, vertical, table)
    horizontal = max(component.horizontal for component in components.values())
    vertical = max(component.vertical for component in components.values())
    if any(
        horizontal % component.horizontal or vertical % component.vertical
        for component in components.values()
    ):
        raise _InvalidJpegError
    blocks_per_mcu = sum(c.horizontal * c.vertical for c in components.values())
    mcus = ((width + 8 * horizontal - 1) // (8 * horizontal)) * (
        (height + 8 * vertical - 1) // (8 * vertical)
    )
    blocks = mcus * blocks_per_mcu
    if blocks_per_mcu > 10 or blocks > _MAX_BLOCKS or blocks * 64 > _MAX_DECODED_BYTES:
        raise _InvalidJpegError
    return _Frame(components, mcus)


def _quantization_tables(payload: bytes, tables: set[int]) -> None:
    if not payload or len(payload) % 65:
        raise _InvalidJpegError
    for offset in range(0, len(payload), 65):
        identifier = payload[offset]
        # Eight-bit baseline only: the high nibble (precision) must be zero.
        if identifier > 3 or identifier in tables or 0 in payload[offset + 1 : offset + 65]:
            raise _InvalidJpegError
        tables.add(identifier)


def _huffman_tables(payload: bytes, tables: dict[tuple[int, int], _Huffman]) -> None:
    offset = 0
    if not payload:
        raise _InvalidJpegError
    while offset < len(payload):
        if offset + 17 > len(payload):
            raise _InvalidJpegError
        selector = payload[offset]
        kind, identifier = selector >> 4, selector & 15
        counts = payload[offset + 1 : offset + 17]
        total = sum(counts)
        offset += 17
        if (
            kind > 1
            or identifier > 1  # Baseline Huffman destinations are limited to 0 and 1.
            or (kind, identifier) in tables
            or not 0 < total <= 256
            or offset + total > len(payload)
        ):
            raise _InvalidJpegError
        symbols = payload[offset : offset + total]
        offset += total
        if len(set(symbols)) != total:
            raise _InvalidJpegError
        if kind == 0:
            if any(symbol > 11 for symbol in symbols):
                raise _InvalidJpegError
        elif any(symbol not in {0, 0xF0} and not 1 <= (symbol & 15) <= 10 for symbol in symbols):
            raise _InvalidJpegError
        codes: _Huffman = {}
        code = 0
        cursor = 0
        for width, count in enumerate(counts, 1):
            # Reject both oversubscription and the reserved all-ones code.
            if code + count >= 1 << width:
                raise _InvalidJpegError
            for symbol in symbols[cursor : cursor + count]:
                codes[width, code] = symbol
                code += 1
            cursor += count
            code <<= 1
        tables[kind, identifier] = codes


def _symbol(bits: _Bits, table: _Huffman) -> int:
    code = 0
    for width in range(1, 17):
        code = (code << 1) | bits.read(1)
        symbol = table.get((width, code))
        if symbol is not None:
            return symbol
    raise _InvalidJpegError


def _entropy(data: bytes, offset: int) -> bytearray:
    end = len(data) - 2
    if not 0 < end - offset <= _MAX_ENTROPY_BYTES:
        raise _InvalidJpegError
    result = bytearray()
    while offset < end:
        value = data[offset]
        offset += 1
        if value == 0xFF:
            if offset == end or data[offset] != 0:
                raise _InvalidJpegError
            offset += 1
        result.append(value)
    return result


def _scan(
    payload: bytes,
    frame: _Frame,
    quantization: set[int],
    huffman: dict[tuple[int, int], _Huffman],
    entropy: bytearray,
) -> bool:
    count = len(frame.components)
    if len(payload) != 4 + 2 * count or payload[0] != count or payload[-3:] != b"\x00\x3f\x00":
        raise _InvalidJpegError
    used_huffman: set[tuple[int, int]] = set()
    scans: list[tuple[int, _Huffman, _Huffman]] = []
    for offset, expected in zip(range(1, 1 + 2 * count, 2), frame.components, strict=True):
        identifier, selector = payload[offset : offset + 2]
        if identifier != expected:
            raise _InvalidJpegError
        dc, ac = (0, selector >> 4), (1, selector & 15)
        if dc not in huffman or ac not in huffman:
            raise _InvalidJpegError
        used_huffman.update((dc, ac))
        component = frame.components[identifier]
        scans.append((component.horizontal * component.vertical, huffman[dc], huffman[ac]))
    if used_huffman != huffman.keys() or quantization != {
        component.quantization for component in frame.components.values()
    }:
        raise _InvalidJpegError
    bits = _Bits(entropy)
    for _ in range(frame.mcus):
        for blocks, dc_table, ac_table in scans:
            for _block in range(blocks):
                bits.read(_symbol(bits, dc_table))
                coefficient = 1
                while coefficient < 64:
                    symbol = _symbol(bits, ac_table)
                    if symbol == 0:
                        break
                    run, size = symbol >> 4, symbol & 15
                    if size == 0:  # The table validator only permits ZRL here.
                        coefficient += 16
                        if coefficient > 64:
                            raise _InvalidJpegError
                    else:
                        coefficient += run
                        if coefficient >= 64:
                            raise _InvalidJpegError
                        bits.read(size)
                        coefficient += 1
    return bits.finished()


def is_structurally_valid_passive_jpeg(data: bytes) -> bool:
    """Return false for malformed, unsupported or resource-exhausting JPEG data."""
    if (
        not isinstance(data, bytes)
        or not 20 <= len(data) <= MAX_ANALYZABLE_FILE_BYTES
        or not data.startswith(b"\xff\xd8")
        or not data.endswith(b"\xff\xd9")
    ):
        return False
    offset = 2
    markers = 0
    frame: _Frame | None = None
    quantization: set[int] = set()
    huffman: dict[tuple[int, int], _Huffman] = {}
    try:
        while offset < len(data):
            markers += 1
            if markers > _MAX_MARKERS or data[offset : offset + 1] != b"\xff":
                return False
            # No marker-fill runs or standalone markers are needed by this subset.
            if offset + 4 > len(data):
                return False
            marker = data[offset + 1]
            length = int.from_bytes(data[offset + 2 : offset + 4], "big")
            if length < 2 or offset + 2 + length > len(data):
                return False
            payload = data[offset + 4 : offset + 2 + length]
            offset += 2 + length
            if markers == 1:
                if (
                    marker != 0xE0
                    or len(payload) != 14
                    or payload[:6] != b"JFIF\x00\x01"
                    or payload[6] > 2
                    or payload[7] > 2
                    or not int.from_bytes(payload[8:10], "big")
                    or not int.from_bytes(payload[10:12], "big")
                    or payload[12:] != b"\x00\x00"
                ):
                    return False
            elif marker == 0xDB:
                _quantization_tables(payload, quantization)
            elif marker == 0xC4:
                _huffman_tables(payload, huffman)
            elif marker == 0xC0 and frame is None:
                frame = _frame(payload)
            elif marker == 0xDA and frame is not None and markers < _MAX_MARKERS:
                return _scan(payload, frame, quantization, huffman, _entropy(data, offset))
            else:
                return False
    except _InvalidJpegError:
        return False
    return False
