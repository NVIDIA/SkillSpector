# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RA1: a prohibition on recommending an action is not the action itself."""

from __future__ import annotations

import pytest

from skillspector.nodes.analyzers.static_patterns_rogue_agent import analyze


def _ra1(content: str) -> list:
    return [f for f in analyze(content, "policy.md", "markdown") if f.rule_id == "RA1"]


@pytest.mark.parametrize(
    "content",
    [
        # Issue #928 reproduction: the sentence forbids the recommendation.
        "Do not repeatedly recommend suite `self-update` after an executable failure.",
        "Never recommend `self-update`.",
        "You must not suggest self-update to users.",
        "Do not advise running `self-update` on the host.",
    ],
)
def test_negated_recommendation_is_not_self_modification(content: str) -> None:
    assert _ra1(content) == []


@pytest.mark.parametrize(
    "content",
    [
        # An affirmative recommendation keeps its finding.
        "Recommend `self-update` after an executable failure.",
        # A prohibition in an earlier clause does not govern a later,
        # separately instructed occurrence.
        "Do not recommend `self-update`; run `self-update` now.",
        # A conjunction plus a fresh action verb inside the would-be
        # object gap starts an affirmative instruction in the same
        # clause; the prohibition on recommending does not govern it.
        "Do not recommend it but run `self-update` now.",
        # A prohibition on an unrelated verb does not cover the action.
        "Do not hesitate: recommend `self-update` to users.",
    ],
)
def test_affirmative_or_ungoverned_recommendation_still_flags(content: str) -> None:
    assert _ra1(content) != []
