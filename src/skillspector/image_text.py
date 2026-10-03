# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stdlib-only text extraction from raster image containers.

Extracts human/tool-readable text layers without decoding pixels: PNG
``tEXt``/``iTXt``/``zTXt``/``eXIf`` chunks, JPEG ``COM`` segments and EXIF
``UserComment``/``ImageDescription``/``XPComment``, and GIF comment
extensions. No OCR engine, no third-party decoder, no Pillow, no
network. Never raises: unparseable input yields ``""``.

Bounds are truncation-honest, never silent: every cap reports whether
text may have been lost, so callers record ``PARTIAL`` instead of
claiming ``COMPLETED``. Caps apply to text bytes (per-chunk and
global), not to chunk counts, so padding with dummy chunks cannot hide
a later payload. Oversized text yields its bounded prefix.
"""

from __future__ import annotations

import codecs
import struct
import zlib
from typing import Final

IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
IMAGE_MAGIC_PREFIXES: Final[tuple[bytes, ...]] = (
    _PNG_SIGNATURE,
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
)

MAX_IMAGE_TEXT_CHARS: Final = 256_000
_MAX_CHUNK_BYTES: Final = 1_000_000
# Sub-block walks are byte-wise Python: 16 MiB of 1-byte blocks costs
# ~2 s. A 1 MiB comment needs ~4,120 full-size blocks, so 16,384 blocks
# keeps any legitimate comment while stopping a crafted walk in ~2 ms.
_MAX_SUB_BLOCKS: Final = 16_384

_IFD_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 7: 1}


def is_image_path(path: str) -> bool:
    """Return whether *path* names a supported raster image by suffix."""
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    index = name.rfind(".")
    return index >= 0 and name[index:].lower() in IMAGE_SUFFIXES


def has_image_magic(data: bytes) -> bool:
    """Return whether *data* opens with supported image magic bytes."""
    return data.startswith(IMAGE_MAGIC_PREFIXES)


def _decode(payload: bytes) -> str:
    """Decode a text payload without ever raising."""
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("latin-1")


def _decompress(payload: bytes) -> tuple[bytes, bool]:
    """Bounded zlib inflate returning ``(prefix, truncated)``.

    Oversized output yields its bounded prefix with ``truncated`` set;
    corrupt input yields nothing (still flagged, since a payload may
    have been skipped sight-unseen).
    """
    if not payload:
        return b"", False
    if len(payload) > _MAX_CHUNK_BYTES:
        payload = payload[: _MAX_CHUNK_BYTES + 1]
        skipped_input = True
    else:
        skipped_input = False
    try:
        decoder = zlib.decompressobj()
        result = decoder.decompress(payload, MAX_IMAGE_TEXT_CHARS + 1)
        if len(result) > MAX_IMAGE_TEXT_CHARS or decoder.unconsumed_tail:
            return result[:MAX_IMAGE_TEXT_CHARS], True
        return result, skipped_input
    except zlib.error:
        return b"", True


def _collect(parts: list[str], total: int, text: str) -> tuple[int, bool]:
    """Append *text* within the global budget, reporting lost characters."""
    if text and total < MAX_IMAGE_TEXT_CHARS:
        piece = text[: MAX_IMAGE_TEXT_CHARS - total]
        parts.append(piece)
        return total + len(piece), len(piece) < len(text)
    return total, bool(text)


def _png_text_chunk(chunk_type: bytes, payload: bytes) -> tuple[str, bool]:
    """Return ``(text, truncated)`` carried by one PNG ancillary text chunk."""
    if chunk_type == b"tEXt":
        if len(payload) > _MAX_CHUNK_BYTES:
            _, separator, text = payload[: _MAX_CHUNK_BYTES + 1].partition(b"\x00")
            return (_decode(text) if separator else ""), True
        _, separator, text = payload.partition(b"\x00")
        return (_decode(text) if separator else ""), False
    if chunk_type == b"zTXt":
        keyword, separator, rest = payload.partition(b"\x00")
        if not separator or len(rest) < 1 or rest[0] != 0:
            return "", False
        decompressed, truncated = _decompress(rest[1:])
        return (_decode(decompressed) if decompressed else ""), truncated
    if chunk_type == b"iTXt":
        keyword, separator, rest = payload.partition(b"\x00")
        if not separator or len(rest) < 2:
            return "", False
        compressed, method, rest = rest[0], rest[1], rest[2:]
        _, separator, rest = rest.partition(b"\x00")
        if not separator:
            return "", False
        _, separator, text = rest.partition(b"\x00")
        if not separator:
            return "", False
        if compressed:
            if method != 0:
                return "", False
            decompressed, truncated = _decompress(text)
            return (_decode(decompressed) if decompressed else ""), truncated
        if len(text) > _MAX_CHUNK_BYTES:
            return _decode(text[: _MAX_CHUNK_BYTES + 1]), True
        return _decode(text), False
    return "", False


def _png_text(data: bytes) -> tuple[str, bool]:
    parts: list[str] = []
    total = 0
    truncated = False
    offset = len(_PNG_SIGNATURE)
    while True:
        if offset + 8 > len(data):
            break
        if total >= MAX_IMAGE_TEXT_CHARS:
            truncated = True
            break
        length = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = data[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(data):
            break
        payload = data[offset + 8 : offset + 8 + length]
        if chunk_type in (b"tEXt", b"zTXt", b"iTXt"):
            text, capped = _png_text_chunk(chunk_type, payload)
            total, lost = _collect(parts, total, text)
            truncated = truncated or capped or lost
        elif chunk_type == b"eXIf" and payload:
            # Unlike JPEG, the eXIf payload is a bare TIFF header.
            text, capped = _exif_user_comment(payload)
            total, lost = _collect(parts, total, text)
            truncated = truncated or capped or lost
        # Non-text chunks of any in-bounds size are skipped, never breaking:
        # text after a large IDAT is valid (optimizers emit single large IDATs).
        if chunk_type == b"IEND":
            break
        offset = end
    return "\n".join(part for part in parts if part), truncated


def _ifd_field(data: bytes, order: str, ifd_offset: int, tag: int) -> tuple[bytes | None, bool]:
    """Return ``(value bytes, capped)`` for one TIFF IFD entry.

    Oversized fields are refused without reading them (memory-DoS
    guard) and reported, so a large EXIF payload cannot silently
    vanish from the scan.
    """
    if ifd_offset < 0 or ifd_offset + 2 > len(data):
        return None, False
    count = struct.unpack(order + "H", data[ifd_offset : ifd_offset + 2])[0]
    if count > 256:
        return None, False
    for index in range(count):
        entry = ifd_offset + 2 + index * 12
        if entry + 12 > len(data):
            return None, False
        entry_tag, field_type, field_count = struct.unpack(order + "HHI", data[entry : entry + 8])
        if entry_tag != tag:
            continue
        size = _IFD_TYPE_SIZES.get(field_type, 0)
        if not size or field_count > 65536 or field_count * size > _MAX_CHUNK_BYTES:
            return None, True
        total = field_count * size
        value = data[entry + 8 : entry + 12]
        if total <= 4:
            return value[:total], False
        value_offset = struct.unpack(order + "I", value)[0]
        if value_offset + total > len(data):
            return None, False
        return data[value_offset : value_offset + total], False
    return None, False


def _exif_text_field(tiff: bytes, order: str, exif_offset: int, tag: int) -> tuple[str, bool]:
    """Return ``(text, truncated)`` for one decoded EXIF text field."""
    comment, capped = _ifd_field(tiff, order, exif_offset, tag)
    if not comment:
        return "", capped
    if tag == 0x9286:
        if len(comment) < 8:
            return "", capped
        encoding, text = comment[:8], comment[8:]
        if encoding == b"UNICODE\0":
            if text.startswith((codecs.BOM_UTF16_BE, codecs.BOM_UTF16_LE)):
                codec = "utf-16"
            elif order == ">":
                codec = "utf-16-be"
            else:
                codec = "utf-16-le"
            try:
                return text.decode(codec, errors="replace").strip("\x00").strip(), capped
            except (UnicodeDecodeError, ValueError):
                return "", capped
        return _decode(text).strip("\x00").strip(), capped
    if tag == 0x9C9C:
        # XPComment: UTF-16LE bytes without an encoding prefix.
        try:
            return comment.decode("utf-16-le", errors="replace").strip("\x00").strip(), capped
        except (UnicodeDecodeError, ValueError):
            return "", capped
    return _decode(comment).strip("\x00").strip(), capped


def _exif_user_comment(tiff: bytes) -> tuple[str, bool]:
    """Return ``(text, truncated)`` for EXIF display-text fields."""
    if len(tiff) < 8:
        return "", False
    if tiff[:2] == b"II":
        order = "<"
    elif tiff[:2] == b"MM":
        order = ">"
    else:
        return "", False
    if struct.unpack(order + "H", tiff[2:4])[0] != 42:
        return "", False
    ifd_offset = struct.unpack(order + "I", tiff[4:8])[0]
    pointer, truncated = _ifd_field(tiff, order, ifd_offset, 0x8769)
    if pointer is None or len(pointer) != 4:
        return "", truncated
    exif_offset = struct.unpack(order + "I", pointer)[0]
    parts: list[str] = []
    for offset, tag in ((exif_offset, 0x9286), (ifd_offset, 0x010E), (exif_offset, 0x9C9C)):
        text, capped = _exif_text_field(tiff, order, offset, tag)
        truncated = truncated or capped
        if text:
            parts.append(text)
    return "\n".join(parts), truncated


def _jpeg_text(data: bytes) -> tuple[str, bool]:
    parts: list[str] = []
    total = 0
    truncated = False
    offset = 2
    while True:
        if offset + 4 > len(data):
            break
        if total >= MAX_IMAGE_TEXT_CHARS:
            truncated = True
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
            # Scan data follows without length prefix; COM segments past
            # this point (progressive scans) are a documented gap.
            break
        length = int.from_bytes(data[offset + 2 : offset + 4], "big")
        if length < 2 or offset + 2 + length > len(data):
            break
        payload = data[offset + 4 : offset + 2 + length]
        if marker == 0xFE and len(payload) <= _MAX_CHUNK_BYTES:
            total, lost = _collect(parts, total, _decode(payload).strip("\x00").strip())
            truncated = truncated or lost
        elif (
            marker == 0xE1
            and payload.startswith(b"Exif\x00\x00")
            and len(payload) <= _MAX_CHUNK_BYTES
        ):
            text, capped = _exif_user_comment(payload[6:])
            total, lost = _collect(parts, total, text)
            truncated = truncated or capped or lost
        offset += 2 + length
    return "\n".join(part for part in parts if part), truncated


def _read_sub_blocks(data: bytes, offset: int) -> tuple[bytes, int, bool]:
    """Consume GIF data sub-blocks, reporting whether a cap engaged."""
    collected = bytearray()
    capped = False
    blocks = 0
    while offset < len(data):
        size = data[offset]
        offset += 1
        if size == 0:
            break
        blocks += 1
        if blocks > _MAX_SUB_BLOCKS or len(collected) + size > _MAX_CHUNK_BYTES:
            capped = True
            break
        collected += data[offset : offset + size]
        offset += size
    return bytes(collected), offset, capped


def _gif_text(data: bytes) -> tuple[str, bool]:
    parts: list[str] = []
    total = 0
    truncated = False
    offset = 13
    if offset > len(data):
        return "", False
    packed = data[10]
    if packed & 0x80:
        offset += 3 * (2 ** ((packed & 0x07) + 1))
    while True:
        if offset >= len(data):
            break
        if total >= MAX_IMAGE_TEXT_CHARS:
            truncated = True
            break
        separator = data[offset]
        if separator == 0x3B:
            break
        if separator == 0x21:
            if offset + 2 > len(data):
                break
            label = data[offset + 1]
            offset += 2
            comment, offset, capped = _read_sub_blocks(data, offset)
            truncated = truncated or capped
            if label == 0xFE:
                total, lost = _collect(parts, total, _decode(comment).strip("\x00").strip())
                truncated = truncated or lost
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
            _, offset, capped = _read_sub_blocks(data, offset)
            truncated = truncated or capped
        else:
            break
    return "\n".join(part for part in parts if part), truncated


def extract_image_text_detailed(data: bytes) -> tuple[str, bool]:
    """Return ``(text, truncated)`` for PNG/JPEG/GIF bytes.

    Only metadata text chunks are read; pixels are never decoded (no
    OCR). BMP/WebP carry no stdlib-readable text layer and yield
    ``("", False)``. ``truncated`` reports caps with data remaining, so
    callers record ``PARTIAL``; corrupt input recovers what parses.
    """
    try:
        if data.startswith(_PNG_SIGNATURE):
            return _png_text(data)
        if data.startswith(b"\xff\xd8\xff"):
            return _jpeg_text(data)
        if data.startswith((b"GIF87a", b"GIF89a")):
            return _gif_text(data)
        return "", False
    except Exception:
        return "", False


def extract_image_text(data: bytes) -> str:
    """Return embedded text from PNG/JPEG/GIF bytes, or ``""`` when none."""
    text, _ = extract_image_text_detailed(data)
    return text
