# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Corroborate bounded, uncompressed Windows/OS2 bitmap storage.

Only CORE/INFO headers, RGB pixels and explicit palettes are supported. Other
headers, compression, profiles, gaps and trailers require separate inspection.
This proves storage structure, not the meaning of the rendered image.
"""

from __future__ import annotations

import struct

from skillspector.constants import MAX_ANALYZABLE_FILE_BYTES

_MAX_DIMENSION = 8_192
_MAX_PIXELS = 4 * 1024 * 1024


def is_structurally_valid_passive_bmp(data: bytes) -> bool:
    """Return false for malformed, unsupported or oversized bitmap storage."""
    if not 26 <= len(data) <= MAX_ANALYZABLE_FILE_BYTES or data[:2] != b"BM":
        return False
    size, reserved, pixel_offset, header_size = struct.unpack_from("<IIII", data, 2)
    if size != len(data) or reserved or header_size not in {12, 40}:
        return False
    if len(data) < 14 + header_size:
        return False
    if header_size == 12:
        width, height, planes, bits = struct.unpack_from("<HHHH", data, 18)
        palette_entry_size = 3
        colors = 1 << bits if bits in {1, 4, 8} else 0
        image_size = 0
        if bits not in {1, 4, 8, 24}:
            return False
    else:
        width, height, planes, bits, compression, image_size, xppm, yppm, colors, important = (
            struct.unpack_from("<iiHHIIiiII", data, 18)
        )
        if compression or bits not in {1, 4, 8, 16, 24, 32} or xppm < 0 or yppm < 0:
            return False
        if bits <= 8:
            colors = colors or 1 << bits
            if colors > 1 << bits or important > colors:
                return False
        elif colors or important:
            return False
        palette_entry_size = 4
    height = abs(height)
    if (
        planes != 1
        or not 0 < width <= _MAX_DIMENSION
        or not 0 < height <= _MAX_DIMENSION
        or width * height > _MAX_PIXELS
    ):
        return False
    row_bytes = (width * bits + 7) // 8
    stride = (row_bytes + 3) & ~3
    expected_size = stride * height
    if (
        pixel_offset != 14 + header_size + colors * palette_entry_size
        or pixel_offset + expected_size != len(data)
        or image_size not in {0, expected_size}
    ):
        return False
    if palette_entry_size == 4 and any(data[14 + header_size + 3 : pixel_offset : 4]):
        return False
    for row in range(pixel_offset, len(data), stride):
        # Unused storage must not carry a second, uninspected payload.
        if any(data[row + row_bytes : row + stride]):
            return False
        if bits == 32 and any(data[row + 3 : row + row_bytes : 4]):
            return False
        if bits == 16 and any(value & 0x80 for value in data[row + 1 : row + row_bytes : 2]):
            return False
        if bits <= 8:
            mask = (1 << bits) - 1
            for column in range(width):
                value = data[row + column * bits // 8]
                shift = 8 - bits - column * bits % 8
                if ((value >> shift) & mask) >= colors:
                    return False
            unused_bits = (-width * bits) % 8
            if unused_bits and data[row + row_bytes - 1] & ((1 << unused_bits) - 1):
                return False
    return True
