# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered, best-effort emission from an already validated report plan."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from skillspector.file_output import write_text_no_follow
from skillspector.report_completeness import merge_report_completeness
from skillspector.report_outputs import FormatChoice, ReportOutput


@dataclass(frozen=True, slots=True)
class ReportEmissionResult:
    """Policy inputs, not exit decisions; no report bodies or exceptions retained."""

    output_failed: bool = False
    report_incomplete: bool = False


def _stderr_diagnostic(message: str) -> None:
    print(message, file=sys.stderr)


def _destination(entry: ReportOutput) -> str:
    """Escape controls and bound path text; callers must disable markup too."""
    if entry.path is None:
        return "stdout"
    limit = 256
    return ascii(entry.path[:limit]) + (" (truncated)" if len(entry.path) > limit else "")


def emit_report_outputs(
    entries: Iterable[ReportOutput],
    renderer: Callable[[FormatChoice], dict[str, Any]],
    primary_writer: Callable[[dict[str, Any], ReportOutput], None],
    additional_writer: Callable[[str, str], None] = write_text_no_follow,
    diagnostic: Callable[[str], None] = _stderr_diagnostic,
) -> ReportEmissionResult:
    """Attempt each entry in plan order, rendering fully before any write.

    The caller MUST successfully validate the whole plan against known and
    newly discovered sources before calling, even when primary output is stdout.
    ``build_report_outputs`` supplies primary-first/argument order. The renderer
    serializes a completed snapshot, never runs analysis, and owns format bounds.
    Repeated formats are regenerated, not cached.

    Primary output and its legacy success messages belong to ``primary_writer``.
    Diagnostics receive bounded plain text identifying format/destination, never
    exception/provider text or rendered content. CLI callers should pass their
    literal-line stderr adapter. Ordinary diagnostic failures also flag failure
    without preventing later attempts. Interrupts/cancellation always propagate.
    The caller applies output-error precedence and optional strict incomplete
    gating afterwards and owns snapshot/resource cleanup.
    """
    output_failed = False
    report_incomplete = False
    for entry in entries:
        rendered = None
        stage = "render"
        failed = False
        try:
            rendered = renderer(entry.format)
            report_incomplete |= not merge_report_completeness(
                rendered.get("report_completeness", {})
            )["is_complete"]
            # Validate the complete body before delegating any output operation.
            body = rendered["report_body"]
            if not isinstance(body, str):
                raise TypeError("A rendered report body must be text.")
            stage = "write"
            if entry.is_primary:
                primary_writer(rendered, entry)
            else:
                if entry.path is None:
                    raise ValueError("An additional report requires a file destination.")
                additional_writer(entry.path, body)
        except Exception:
            output_failed = failed = True
        finally:
            # Do not keep the previous body alive while the next renderer runs.
            rendered = None
            body = None

        if failed or not entry.is_primary:
            status = f"Failed to {stage}" if failed else "Saved"
            message = f"{status} {entry.format.value} report to {_destination(entry)}."
            try:
                diagnostic(message)
            except Exception:
                output_failed = True

    return ReportEmissionResult(output_failed=output_failed, report_incomplete=report_incomplete)
