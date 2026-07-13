# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Language detection via Unicode script ratio analysis.

Zero external dependencies — uses only the standard-library ``unicodedata``
module, the same one the main SkillSpector project already imports in
``mcp_tool_poisoning.py``.

Approach: count CJK / Hiragana / Katakana / Hangul characters against
total alphabetic content.  A configurable ratio threshold decides the
dominant language.  This avoids heavyweight ML-based detectors while
being accurate enough for the batch-scan use case.
"""

from __future__ import annotations

import unicodedata

# Unicode range constants — (start, end) inclusive.
_CJK_UNIFIED = (0x4E00, 0x9FFF)  # CJK Unified Ideographs
_CJK_EXT_A = (0x3400, 0x4DBF)    # CJK Unified Ideographs Extension A
_HIRAGANA = (0x3040, 0x309F)
_KATAKANA = (0x30A0, 0x30FF)
_HANGUL = (0xAC00, 0xD7AF)       # Hangul Syllables

# Thresholds — a skill file is classified as non-English when the ratio of
# CJK / kana / Hangul characters exceeds this proportion of total alpha chars.
_CJK_THRESHOLD = 0.10
_KANA_THRESHOLD = 0.05
_HANGUL_THRESHOLD = 0.10


def _in_range(cp: int, r: tuple[int, int]) -> bool:
    return r[0] <= cp <= r[1]


def detect_language(content: str) -> str:
    """Heuristic single-file language detection.

    Returns one of ``"zh"``, ``"ja"``, ``"ko"``, or ``"en"``.
    """
    cjk = kana = hangul = alpha = 0
    for ch in content:
        cp = ord(ch)
        if _in_range(cp, _CJK_UNIFIED) or _in_range(cp, _CJK_EXT_A):
            cjk += 1
        elif _in_range(cp, _HIRAGANA) or _in_range(cp, _KATAKANA):
            kana += 1
        elif _in_range(cp, _HANGUL):
            hangul += 1
        if unicodedata.category(ch).startswith("L"):
            alpha += 1

    if alpha == 0:
        return "en"

    if kana / alpha > _KANA_THRESHOLD:
        return "ja"
    if hangul / alpha > _HANGUL_THRESHOLD:
        return "ko"
    if cjk / alpha > _CJK_THRESHOLD:
        return "zh"
    return "en"


def detect_skill_language(file_cache: dict[str, str]) -> str:
    """Determine the dominant language across all files in a skill.

    Aggregates per-file :func:`detect_language` results via majority vote.
    When no non-English script is detected in any file, returns ``"en"``.
    """
    votes: dict[str, int] = {}
    for content in file_cache.values():
        lang = detect_language(content)
        votes[lang] = votes.get(lang, 0) + 1
    if not votes:
        return "en"
    return max(votes, key=lambda k: votes[k])  # type: ignore[no-any-return]
