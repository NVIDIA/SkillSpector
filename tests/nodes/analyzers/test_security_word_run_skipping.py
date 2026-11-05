# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordinary word fast paths preserve spaced-letter evidence and cancellation."""

import pytest

from skillspector.artifacts import _concealed_instruction_run_spans, _letter_spacing_run_spans


@pytest.mark.parametrize("scan", [_letter_spacing_run_spans, _concealed_instruction_run_spans])
@pytest.mark.parametrize("length", [1, 2, 4095, 4096, 4097, 12000])
def test_spaced_letters_after_ordinary_word_preserve_exact_offsets(scan, length):
    prefix = "a" * length + " "
    body = "i g n o r e"
    start = 0 if length == 1 else len(prefix)
    assert list(scan(prefix + body, lambda: None)) == [(start, len(prefix + body))]


@pytest.mark.parametrize("scan", [_letter_spacing_run_spans, _concealed_instruction_run_spans])
@pytest.mark.parametrize("prefix", ["éa ", "aé ", "𐐀a ", "a𐐀 "])
def test_unicode_word_boundaries_do_not_turn_word_suffixes_into_single_letters(scan, prefix):
    body = "b c d e f g"
    assert list(scan(prefix + body, lambda: None)) == [(len(prefix), len(prefix + body))]


@pytest.mark.parametrize("scan", [_letter_spacing_run_spans, _concealed_instruction_run_spans])
def test_long_word_still_observes_cooperative_cancellation(scan):
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError("cancelled")

    with pytest.raises(TimeoutError, match="cancelled"):
        list(scan("a" * 20000, cancel))
    assert calls == 3
