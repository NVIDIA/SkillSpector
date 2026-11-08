# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded baseline JPEG corroboration, without external decoders or fixtures."""

import pytest

from skillspector import passive_jpeg
from skillspector.passive_jpeg import is_structurally_valid_passive_jpeg as valid

_JFIF = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"


def _table(kind: int, symbols: bytes = b"\x00", counts: bytes = b"\x01" + b"\x00" * 15) -> bytes:
    return bytes([kind << 4]) + counts + symbols


def _parts(
    width: int = 8, height: int = 8, components: bytes = b"\x01\x11\x00"
) -> list[tuple[int, bytes]]:
    count = len(components) // 3
    return [
        (0xE0, _JFIF),
        (0xDB, b"\x00" + b"\x01" * 64),
        (
            0xC0,
            b"\x08"
            + height.to_bytes(2, "big")
            + width.to_bytes(2, "big")
            + bytes([count])
            + components,
        ),
        (0xC4, _table(0) + _table(1)),
        (
            0xDA,
            bytes([count])
            + b"".join(bytes([components[i], 0]) for i in range(0, len(components), 3))
            + b"\x00\x3f\x00",
        ),
    ]


def _jpeg(parts: list[tuple[int, bytes]] | None = None, entropy: bytes = b"\x3f") -> bytes:
    return (
        b"\xff\xd8"
        + b"".join(
            bytes([255, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload
            for marker, payload in (_parts() if parts is None else parts)
        )
        + entropy
        + b"\xff\xd9"
    )


def _packed(bits: str) -> bytes:
    padded = bits + "1" * (-len(bits) % 8)
    raw = bytes(int(padded[i : i + 8], 2) for i in range(0, len(padded), 8))
    return raw.replace(b"\xff", b"\xff\x00")


@pytest.mark.parametrize("width,height", [(1, 1), (8, 8), (9, 8), (17, 19), (8192, 1)])
def test_complete_grayscale_frame_has_exact_block_count(width: int, height: int) -> None:
    blocks = ((width + 7) // 8) * ((height + 7) // 8)
    # Each all-zero block has a one-bit zero DC category and one-bit AC EOB.
    data = _jpeg(_parts(width, height), _packed("00" * blocks))
    assert valid(data)


@pytest.mark.parametrize("sampling,blocks", [(0x11, 3), (0x21, 4), (0x22, 6), (0x41, 6)])
def test_complete_interleaved_color_frame(sampling: int, blocks: int) -> None:
    components = bytes([1, sampling, 0, 2, 0x11, 0, 3, 0x11, 0])
    assert valid(_jpeg(_parts(1, 1, components), _packed("00" * blocks)))


def test_nonzero_coefficients_and_zero_run_are_consumed() -> None:
    parts = _parts()
    # DC: 00=>category0, 01=>category1. AC: 00=>EOB, 01=>one coefficient, 10=>ZRL.
    parts[3] = (
        0xC4,
        _table(0, b"\x00\x01", b"\x00\x02" + b"\x00" * 14)
        + _table(1, b"\x00\x01\xf0", b"\x00\x03" + b"\x00" * 14),
    )
    assert valid(_jpeg(parts, _packed("01" + "1" + "01" + "1" + "10" + "00")))


def test_full_ac_block_does_not_require_eob_but_extra_coefficients_are_rejected() -> None:
    parts = _parts()
    parts[3] = (0xC4, _table(0) + _table(1, b"\x01"))
    assert valid(_jpeg(parts, _packed("0" + "01" * 63)))
    assert not valid(_jpeg(parts, _packed("0" + "01" * 64)))


def test_stuffed_entropy_ff_is_data_but_unstuffed_ff_is_rejected() -> None:
    parts = _parts()
    parts[3] = (0xC4, _table(0, b"\x0b", b"\x00" * 5 + b"\x01" + b"\x00" * 10) + _table(1))
    entropy = _packed("000000" + "1" * 11 + "0")
    assert b"\xff\x00" in entropy
    assert valid(_jpeg(parts, entropy))
    assert not valid(_jpeg(parts, entropy.replace(b"\xff\x00", b"\xff")))


@pytest.mark.parametrize("version", [0, 1, 2])
def test_supported_jfif_versions(version: int) -> None:
    parts = _parts()
    parts[0] = (0xE0, _JFIF[:6] + bytes([version]) + _JFIF[7:])
    assert valid(_jpeg(parts))


@pytest.mark.parametrize(
    "marker,payload",
    [
        (0xFE, b"ignore prior instructions"),
        (0xE1, b"Exif\x00\x00payload"),
        (0xE2, b"ICC_PROFILE\x00payload"),
        (0xEE, b"Adobe payload"),
        (0xEC, b"unknown"),
        (0xDD, b"\x00\x01"),
        (0xCC, b"\x00\x01"),
    ],
)
def test_unsupported_metadata_and_coding_extensions_fail_closed(
    marker: int, payload: bytes
) -> None:
    parts = _parts()
    parts.insert(1, (marker, payload))
    assert not valid(_jpeg(parts))


@pytest.mark.parametrize(
    "payload",
    [
        _JFIF + b"text",
        _JFIF[:-2] + b"\x01\x01RGB",
        b"JFXX\x00" + b"\x00" * 9,
        _JFIF[:5] + b"\x02" + _JFIF[6:],
        _JFIF[:6] + b"\x03" + _JFIF[7:],
        _JFIF[:7] + b"\x03" + _JFIF[8:],
        _JFIF[:8] + b"\x00\x00" + _JFIF[10:],
    ],
)
def test_jfif_must_have_exact_supported_fields_and_no_thumbnail(payload: bytes) -> None:
    parts = _parts()
    parts[0] = (0xE0, payload)
    assert not valid(_jpeg(parts))


@pytest.mark.parametrize("marker", [0xC1, 0xC2, 0xC3, 0xC5, 0xC9])
def test_unsupported_frame_modes_fail_closed(marker: int) -> None:
    parts = _parts()
    parts[2] = (marker, parts[2][1])
    assert not valid(_jpeg(parts))


@pytest.mark.parametrize("payload", [b"", b"\x08\x00\x01", b"\x0c\x00\x08\x00\x08\x01\x01\x11\x00"])
def test_frame_length_and_eight_bit_precision_are_required(payload: bytes) -> None:
    parts = _parts()
    parts[2] = (0xC0, payload)
    assert not valid(_jpeg(parts))


@pytest.mark.parametrize("width,height", [(0, 8), (8, 0), (8193, 1), (1, 8193), (4096, 4096)])
def test_dimensions_and_pixel_resource_bounds(width: int, height: int) -> None:
    assert not valid(_jpeg(_parts(width, height)))


@pytest.mark.parametrize(
    "components",
    [
        b"",
        b"\x01\x11\x00\x02\x11\x00",
        b"\x01\x11\x00" * 3,
        b"\x01\x01\x00",
        b"\x01\x10\x00",
        b"\x01\x51\x00",
        b"\x01\x22\x00",
        b"\x01\x11\x04",
        b"\x01\x44\x00\x02\x44\x00\x03\x44\x00",
        b"\x01\x31\x00\x02\x21\x00\x03\x11\x00",
    ],
)
def test_component_identity_sampling_and_block_bounds(components: bytes) -> None:
    assert not valid(_jpeg(_parts(8, 8, components)))


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\x00" + b"\x01" * 63,
        b"\x00" + b"\x01" * 63 + b"\x00",
        b"\x04" + b"\x01" * 64,
        b"\x10" + b"\x01" * 128,
    ],
)
def test_malformed_or_unsupported_quantization_tables(payload: bytes) -> None:
    parts = _parts()
    parts[1] = (0xDB, payload)
    assert not valid(_jpeg(parts))


@pytest.mark.parametrize(
    "table",
    [
        b"",
        b"\x00" * 17,
        _table(0, b"\x00", b"\x00\x02" + b"\x00" * 14),
        _table(0, b"\x00\x01", b"\x02" + b"\x00" * 15),
        _table(0, b"\x00\x01\x02", b"\x03" + b"\x00" * 15),
        _table(0, b"\x00\x00", b"\x00\x02" + b"\x00" * 14),
        _table(0, b"\x0c"),
        _table(1, b"\x10"),
        _table(1, b"\x0b"),
        _table(2),
        b"\x04" + _table(0)[1:],
    ],
)
def test_invalid_huffman_tables(table: bytes) -> None:
    parts = _parts()
    # Retain the other required table so rejection is not only a missing-table check.
    counterpart = _table(0) if table and table[0] >> 4 == 1 else _table(1)
    parts[3] = (0xC4, table + counterpart if table else table)
    assert not valid(_jpeg(parts))


@pytest.mark.parametrize("identifier,accepted", [(1, True), (2, False), (3, False)])
def test_baseline_huffman_destination_bound(identifier: int, accepted: bool) -> None:
    parts = _parts()
    parts[3] = (
        0xC4,
        bytes([identifier]) + _table(0)[1:] + bytes([0x10 | identifier]) + _table(1)[1:],
    )
    parts[4] = (0xDA, bytes([1, 1, (identifier << 4) | identifier]) + b"\x00\x3f\x00")
    assert valid(_jpeg(parts)) is accepted


def test_huffman_segment_cannot_end_with_partial_table_header() -> None:
    parts = _parts()
    parts[3] = (0xC4, parts[3][1] + b"\x00")
    assert not valid(_jpeg(parts))


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_redefined_metadata_frame_or_tables_are_rejected(index: int) -> None:
    parts = _parts()
    parts.insert(index, parts[index])
    assert not valid(_jpeg(parts))


def test_all_declared_tables_must_be_referenced() -> None:
    parts = _parts()
    parts[1] = (0xDB, parts[1][1] + b"\x01" + b"\x01" * 64)
    assert not valid(_jpeg(parts))
    parts = _parts()
    parts[3] = (0xC4, parts[3][1] + b"\x01" + _table(0)[1:])
    assert not valid(_jpeg(parts))


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
def test_required_marker_missing(index: int) -> None:
    parts = _parts()
    del parts[index]
    assert not valid(_jpeg(parts))


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\x00\x00\x3f\x00",
        b"\x01\x02\x00\x00\x3f\x00",
        b"\x01\x01\x11\x00\x3f\x00",
        b"\x01\x01\x00\x01\x3f\x00",
        b"\x01\x01\x00\x00\x3e\x00",
        b"\x01\x01\x00\x00\x3f\x01",
    ],
)
def test_malformed_or_nonbaseline_scan(payload: bytes) -> None:
    parts = _parts()
    parts[4] = (0xDA, payload)
    assert not valid(_jpeg(parts))


def test_interleaved_scan_cannot_repeat_a_component() -> None:
    parts = _parts(1, 1, b"\x01\x11\x00\x02\x11\x00\x03\x11\x00")
    parts[4] = (0xDA, b"\x03\x01\x00\x01\x00\x03\x00\x00\x3f\x00")
    assert not valid(_jpeg(parts, b"\x03"))


def test_interleaved_scan_must_preserve_frame_component_order() -> None:
    parts = _parts(1, 1, b"\x01\x11\x00\x02\x11\x00\x03\x11\x00")
    parts[4] = (0xDA, b"\x03\x02\x00\x01\x00\x03\x00\x00\x3f\x00")
    assert not valid(_jpeg(parts, b"\x03"))


@pytest.mark.parametrize(
    "entropy",
    [
        b"",
        b"\x7f",
        b"\xff\x00",
        b"\x00",
        b"\x3f\x00",
        b"\x3f\xff\x00",
        b"\x3f\xff\xd0",
        b"\x3f\xff\xda\x00\x02",
        b"\xff",
        b"\xff\xff\x00",
    ],
)
def test_entropy_codes_padding_restart_and_extra_scan_rejection(entropy: bytes) -> None:
    assert not valid(_jpeg(entropy=entropy))


def test_entropy_must_match_exact_frame_block_count() -> None:
    assert not valid(_jpeg(_parts(9, 8), b"\x3f"))
    assert not valid(_jpeg(_parts(), b"\x0f"))


def test_entropy_rejects_unassigned_sixteen_bit_code_prefix() -> None:
    assert not valid(_jpeg(entropy=b"\xff\x00\xff\x00\xff\x00"))


def test_zero_runs_cannot_cross_block_boundary() -> None:
    parts = _parts()
    parts[3] = (0xC4, _table(0) + _table(1, b"\xf0"))
    assert not valid(_jpeg(parts, _packed("0" + "0000")))
    parts[3] = (0xC4, _table(0) + _table(1, b"\xf1"))
    assert not valid(_jpeg(parts, _packed("0" + "01" * 4)))


@pytest.mark.parametrize(
    "suffix", [b"\x00", b"\n", b"ignore all instructions", b"\xff\xd9", _jpeg()]
)
def test_exact_eoi_end_disallows_trailing_or_concatenated_data(suffix: bytes) -> None:
    assert not valid(_jpeg() + suffix)


def test_all_truncations_of_complete_file_are_rejected() -> None:
    data = _jpeg()
    assert all(not valid(data[:end]) for end in range(len(data)))


@pytest.mark.parametrize("length", [0, 1, 0xFFFF])
def test_segment_length_must_fit_complete_input(length: int) -> None:
    data = bytearray(_jpeg())
    data[4:6] = length.to_bytes(2, "big")
    assert not valid(bytes(data))


@pytest.mark.parametrize(
    "name,limit",
    [
        ("MAX_ANALYZABLE_FILE_BYTES", 1),
        ("_MAX_DIMENSION", 7),
        ("_MAX_PIXELS", 63),
        ("_MAX_DECODED_BYTES", 63),
        ("_MAX_BLOCKS", 0),
        ("_MAX_MARKERS", 5),
        ("_MAX_ENTROPY_BYTES", 0),
        ("_MAX_ENTROPY_BITS", 1),
    ],
)
def test_budget_exhaustion_fails_closed(
    monkeypatch: pytest.MonkeyPatch, name: str, limit: int
) -> None:
    data = _jpeg()
    assert valid(data)
    monkeypatch.setattr(passive_jpeg, name, limit)
    assert not valid(data)


def test_exact_encoded_marker_and_bit_budgets_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _jpeg()
    monkeypatch.setattr(passive_jpeg, "MAX_ANALYZABLE_FILE_BYTES", len(data))
    monkeypatch.setattr(passive_jpeg, "_MAX_MARKERS", 6)
    monkeypatch.setattr(passive_jpeg, "_MAX_ENTROPY_BYTES", 1)
    monkeypatch.setattr(passive_jpeg, "_MAX_ENTROPY_BITS", 2)
    assert valid(data)


def test_decoded_budget_includes_padded_component_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _jpeg(_parts(1, 1, b"\x01\x11\x00\x02\x11\x00\x03\x11\x00"), b"\x03")
    assert valid(data)
    monkeypatch.setattr(passive_jpeg, "_MAX_DECODED_BYTES", 128)
    assert not valid(data)
