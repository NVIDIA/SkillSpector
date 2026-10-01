# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stdlib-only text extraction from raster image containers.

Extracts human/tool-readable text layers without decoding pixels: PNG
``tEXt``/``iTXt``/``zTXt``/``eXIf`` chunks, JPEG ``COM`` segments and EXIF
``UserComment``, and GIF comment extensions. No OCR engine, no third-party
decoder, no Pillow, no network. Never raises: unparseable input yields ``""``.
"""

from __future__ import annotations

import struct
import zlib
from typing import Final

IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
)

MAX_IMAGE_TEXT_CHARS: Final = 256_000
_MAX_CHUNKS: Final = 256
_MAX_CHUNK_BYTES: Final = 1_000_000

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_IFD_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 7: 1}


def is_image_path(path: str) -> bool:
    """Return whether *path* names a supported raster image by suffix."""
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    index = name.rfind(".")
    return index >= 0 and name[index:].lower() in IMAGE_SUFFIXES


def _decode(payload: bytes) -> str:
    """Decode a text payload without ever raising."""
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("latin-1")


def _decompress(payload: bytes) -> bytes | None:
    """Bounded zlib inflate; ``None`` when the stream is not usable."""
    if not payload or len(payload) > _MAX_CHUNK_BYTES:
        return None
    try:
        decoder = zlib.decompressobj()
        result = decoder.decompress(payload, MAX_IMAGE_TEXT_CHARS + 1)
        if len(result) > MAX_IMAGE_TEXT_CHARS or decoder.unconsumed_tail:
            return None
        return result
    except zlib.error:
        return None


def _collect(parts: list[str], total: int, text: str) -> int:
    """Append *text* to *parts* within the global character budget."""
    if text and total < MAX_IMAGE_TEXT_CHARS:
        parts.append(text[: MAX_IMAGE_TEXT_CHARS - total])
        total += len(parts[-1])
    return total


def _png_text_chunk(chunk_type: bytes, payload: bytes) -> str:
    """Return the text carried by one PNG ancillary text chunk."""
    if chunk_type == b"tEXt":
        _, separator, text = payload.partition(b"\x00")
        return _decode(text) if separator else ""
    if chunk_type == b"zTXt":
        keyword, separator, rest = payload.partition(b"\x00")
        if not separator or len(rest) < 1 or rest[0] != 0:
            return ""
        decompressed = _decompress(rest[1:])
        return _decode(decompressed) if decompressed is not None else ""
    if chunk_type == b"iTXt":
        keyword, separator, rest = payload.partition(b"\x00")
        if not separator or len(rest) < 2:
            return ""
        compressed, method, rest = rest[0], rest[1], rest[2:]
        _, separator, rest = rest.partition(b"\x00")
        if not separator:
            return ""
        _, separator, text = rest.partition(b"\x00")
        if not separator:
            return ""
        if compressed:
            if method != 0:
                return ""
            decompressed = _decompress(text)
            return _decode(decompressed) if decompressed is not None else ""
        return _decode(text)
    return ""


def _png_text(data: bytes) -> str:
    parts: list[str] = []
    total = 0
    offset = len(_PNG_SIGNATURE)
    for _ in range(_MAX_CHUNKS):
        if offset + 8 > len(data) or total >= MAX_IMAGE_TEXT_CHARS:
            break
        length = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = data[offset + 4 : offset + 8]
        end = offset + 12 + length
        if length > _MAX_CHUNK_BYTES or end > len(data):
            break
        payload = data[offset + 8 : offset + 8 + length]
        if chunk_type in (b"tEXt", b"zTXt", b"iTXt"):
            total = _collect(parts, total, _png_text_chunk(chunk_type, payload))
        elif chunk_type == b"eXIf" and payload:
            # Unlike JPEG, the eXIf payload is a bare TIFF header.
            total = _collect(parts, total, _exif_user_comment(payload))
        if chunk_type == b"IEND":
            break
        offset = end
    return "\n".join(part for part in parts if part)


def _ifd_field(data: bytes, order: str, ifd_offset: int, tag: int) -> bytes | None:
    """Return the raw value bytes of one TIFF IFD entry, bounded."""
    if ifd_offset < 0 or ifd_offset + 2 > len(data):
        return None
    count = struct.unpack(order + "H", data[ifd_offset : ifd_offset + 2])[0]
    if count > 256:
        return None
    for index in range(count):
        entry = ifd_offset + 2 + index * 12
        if entry + 12 > len(data):
            return None
        entry_tag, field_type, field_count = struct.unpack(order + "HHI", data[entry : entry + 8])
        if entry_tag != tag:
            continue
        size = _IFD_TYPE_SIZES.get(field_type, 0)
        if not size or field_count > 65536 or field_count * size > _MAX_CHUNK_BYTES:
            return None
        total = field_count * size
        value = data[entry + 8 : entry + 12]
        if total <= 4:
            return value[:total]
        value_offset = struct.unpack(order + "I", value)[0]
        if value_offset + total > len(data):
            return None
        return data[value_offset : value_offset + total]
    return None


def _exif_user_comment(tiff: bytes) -> str:
    """Return the EXIF ``UserComment`` string from one TIFF header."""
    if len(tiff) < 8:
        return ""
    if tiff[:2] == b"II":
        order = "<"
    elif tiff[:2] == b"MM":
        order = ">"
    else:
        return ""
    if struct.unpack(order + "H", tiff[2:4])[0] != 42:
        return ""
    ifd_offset = struct.unpack(order + "I", tiff[4:8])[0]
    pointer = _ifd_field(tiff, order, ifd_offset, 0x8769)
    if pointer is None or len(pointer) != 4:
        return ""
    exif_offset = struct.unpack(order + "I", pointer)[0]
    comment = _ifd_field(tiff, order, exif_offset, 0x9286)
    if not comment or len(comment) < 8:
        return ""
    encoding, text = comment[:8], comment[8:]
    if encoding == b"UNICODE\0":
        try:
            return text.decode("utf-16", errors="replace").strip("\x00").strip()
        except (UnicodeDecodeError, ValueError):
            return ""
    return _decode(text).strip("\x00").strip()


def _jpeg_text(data: bytes) -> str:
    parts: list[str] = []
    total = 0
    offset = 2
    for _ in range(_MAX_CHUNKS):
        if offset + 4 > len(data) or total >= MAX_IMAGE_TEXT_CHARS:
            break
        if data[offset] != 0xFF:
            break
        marker = data[offset + 1]
        if marker == 0xD9:
            break
        if marker == 0xD8 or 0xD0 <= marker <= 0xD7:
            offset += 2
            continue
        if marker == 0xDA:
            break
        length = int.from_bytes(data[offset + 2 : offset + 4], "big")
        if length < 2 or offset + 2 + length > len(data):
            break
        payload = data[offset + 4 : offset + 2 + length]
        if marker == 0xFE and len(payload) <= _MAX_CHUNK_BYTES:
            total = _collect(parts, total, _decode(payload).strip("\x00").strip())
        elif (
            marker == 0xE1
            and payload.startswith(b"Exif\x00\x00")
            and len(payload) <= _MAX_CHUNK_BYTES
        ):
            total = _collect(parts, total, _exif_user_comment(payload[6:]))
        offset += 2 + length
    return "\n".join(part for part in parts if part)


def _read_sub_blocks(data: bytes, offset: int) -> tuple[bytes, int]:
    """Consume GIF data sub-blocks, returning their bytes and end offset."""
    collected = bytearray()
    while offset < len(data):
        size = data[offset]
        offset += 1
        if size == 0:
            break
        collected += data[offset : offset + size]
        offset += size
        if len(collected) > _MAX_CHUNK_BYTES:
            break
    return bytes(collected), offset


def _gif_text(data: bytes) -> str:
    parts: list[str] = []
    total = 0
    offset = 13
    if offset > len(data):
        return ""
    packed = data[10]
    if packed & 0x80:
        offset += 3 * (2 ** ((packed & 0x07) + 1))
    for _ in range(_MAX_CHUNKS):
        if offset >= len(data) or total >= MAX_IMAGE_TEXT_CHARS:
            break
        separator = data[offset]
        if separator == 0x3B:
            break
        if separator == 0x21:
            if offset + 2 > len(data):
                break
            label = data[offset + 1]
            offset += 2
            comment, offset = _read_sub_blocks(data, offset)
            if label == 0xFE:
                total = _collect(parts, total, _decode(comment).strip("\x00").strip())
        elif separator == 0x2C:
            if offset + 10 > len(data):
                break
            packed = data[offset + 9]
            offset += 10
            if packed & 0x80:
                offset += 3 * (2 ** ((packed & 0x07) + 1))
            if offset >= len(data):
                break
            offset += 1
            _, offset = _read_sub_blocks(data, offset)
        else:
            break
    return "\n".join(part for part in parts if part)


def extract_image_text(data: bytes) -> str:
    """Return embedded text from PNG/JPEG/GIF bytes, or ``""`` when none.

    Only metadata text chunks are read; pixels are never decoded (no OCR).
    BMP/WebP carry no stdlib-readable text layer and yield ``""``.
    """
    try:
        if data.startswith(_PNG_SIGNATURE):
            return _png_text(data)
        if data.startswith(b"\xff\xd8\xff"):
            return _jpeg_text(data)
        if data.startswith((b"GIF87a", b"GIF89a")):
            return _gif_text(data)
        return ""
    except Exception:
        return ""
