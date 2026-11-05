# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report declarations and ordering, without filesystem checks or emission."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class FormatChoice(StrEnum):
    """Output format choices shared by primary and additional CLI reports."""

    terminal = "terminal"
    json = "json"
    markdown = "markdown"
    sarif = "sarif"


@dataclass(frozen=True, slots=True)
class ReportOutput:
    """One output request; only the primary may have a stdout (None) path.

    Additional path strings retain their spelling, including embedded equals
    signs. Filesystem normalization and validation belong to later preflight.
    """

    format: FormatChoice
    path: str | None
    is_primary: bool = False


def parse_report_declarations(declarations: Sequence[str] | None) -> list[ReportOutput]:
    """Parse file-only FORMAT=PATH requests in order, or raise ValueError.

    This validates syntax only: it does not inspect paths, enforce scan-mode
    restrictions, deduplicate destinations, or write any output.
    """
    reports: list[ReportOutput] = []
    for declaration in declarations or ():
        format_name, separator, path = declaration.partition("=")
        if not separator:
            raise ValueError("Expected FORMAT=PATH for an additional report.")
        try:
            format_choice = FormatChoice(format_name.lower())
        except ValueError as exc:
            choices = ", ".join(choice.value for choice in FormatChoice)
            raise ValueError(f"Unsupported report format; choose one of: {choices}.") from exc
        if not path:
            raise ValueError("An additional report requires a non-empty file path.")
        if path == "-":
            raise ValueError("Additional reports are file-only; PATH=- is not supported.")
        reports.append(ReportOutput(format=format_choice, path=path))
    return reports


def build_report_outputs(
    primary_format: FormatChoice,
    primary_output: Path | None,
    reports: Sequence[ReportOutput],
) -> list[ReportOutput]:
    """Return the legacy primary first, then the parsed additional requests."""
    primary = ReportOutput(
        format=primary_format,
        path=str(primary_output) if primary_output is not None else None,
        is_primary=True,
    )
    return [primary, *reports]
