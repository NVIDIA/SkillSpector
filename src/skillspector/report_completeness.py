# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded report-content omissions, independent of inspection and risk policy."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from itertools import chain, islice
from typing import TypedDict

MAX_OMISSION_REASONS = 256
MAX_OMISSION_REASON_CHARS = 96
MAX_OMITTED_COUNT = (1 << 63) - 1
_REASON_CODE = re.compile(r"[a-z][a-z0-9_]{0,95}\Z")


class ReportCompleteness(TypedDict):
    """Counts describe omitted content when known; zero need not mean none."""

    is_complete: bool
    omission_reasons: list[str]
    omitted_records: int
    omitted_skills: int


def _count(value: object) -> int:
    return min(MAX_OMITTED_COUNT, max(0, value)) if type(value) is int else 0


def build_report_completeness(
    *,
    omission_reasons: Iterable[str] = (),
    omitted_records: int = 0,
    omitted_skills: int = 0,
    is_complete: bool = True,
) -> ReportCompleteness:
    """Build a fresh, payload-free projection with bounded work and storage.

    Reasons are short machine identifiers, never source/provider text. An
    explicit incomplete status survives even when omission counts are unknown.
    """
    reasons: list[str] = []
    seen: set[str] = set()
    source = (omission_reasons,) if isinstance(omission_reasons, str) else omission_reasons
    for index, raw in enumerate(islice(source, MAX_OMISSION_REASONS + 1)):
        if index == MAX_OMISSION_REASONS:
            reasons = reasons[: MAX_OMISSION_REASONS - 1]
            if "omission_reason_limit" not in reasons:
                reasons.append("omission_reason_limit")
            break
        reason = (
            raw
            if isinstance(raw, str)
            and len(raw) <= MAX_OMISSION_REASON_CHARS
            and _REASON_CODE.fullmatch(raw)
            else "unknown_omission"
        )
        if reason not in seen:
            reasons.append(reason)
            seen.add(reason)
    records, skills = _count(omitted_records), _count(omitted_skills)
    return {
        "is_complete": is_complete is True and not reasons and not records and not skills,
        "omission_reasons": reasons,
        "omitted_records": records,
        "omitted_skills": skills,
    }


def merge_report_completeness(
    *projections: Mapping[str, object],
    omission_reasons: Iterable[str] = (),
    omitted_records: int = 0,
    omitted_skills: int = 0,
) -> ReportCompleteness:
    """Merge disjoint child/retention/local omissions into a private projection.

    Counts are summed (and saturated), not deduplicated. Callers must not pass
    both an aggregate and the same children again. No input is mutated.
    """
    reasons: list[Iterable[str]] = []
    records, skills = _count(omitted_records), _count(omitted_skills)
    complete = True
    for projection in projections:
        complete = complete and projection.get("is_complete", True) is True
        raw = projection.get("omission_reasons", ())
        if isinstance(raw, (list, tuple)):
            reasons.append(raw)
        elif isinstance(raw, str):
            reasons.append((raw,))
        records = min(MAX_OMITTED_COUNT, records + _count(projection.get("omitted_records")))
        skills = min(MAX_OMITTED_COUNT, skills + _count(projection.get("omitted_skills")))
    reasons.append(
        (omission_reasons,) if isinstance(omission_reasons, str) else omission_reasons
    )
    return build_report_completeness(
        omission_reasons=chain.from_iterable(reasons),
        omitted_records=records,
        omitted_skills=skills,
        is_complete=complete,
    )


def report_completeness_notice(projection: Mapping[str, object]) -> str:
    """A literal, bounded notice for terminal/Markdown and recursive fallbacks."""
    bounded = merge_report_completeness(projection)
    status = "complete" if bounded["is_complete"] else "incomplete"
    reasons = ", ".join(bounded["omission_reasons"]) or "none"
    return (
        f"Report completeness: {status}; omission reasons: {reasons}; "
        f"omitted records: {bounded['omitted_records']}; "
        f"omitted skills: {bounded['omitted_skills']}."
    )
