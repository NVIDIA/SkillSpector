# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse additional file reports without changing scan execution."""

from pathlib import Path

from skillspector.input_handler import _normalize_root_owned_alias


def parse_reports(
    declarations: list[str] | None,
    formats: tuple[str, ...],
    output: Path | None = None,
) -> list[tuple[str, Path]]:
    """Normalize writer destinations before checking aliases or scan inputs."""
    reports: list[tuple[str, Path]] = []
    paths = [_normalize_root_owned_alias(output)] if output is not None else []
    for declaration in declarations or []:
        name, separator, filename = declaration.partition("=")
        name = name.lower()
        if not separator or name not in formats or not filename or filename == "-":
            raise ValueError(
                f"--report expects FORMAT=PATH ({', '.join(formats)}); "
                "PATH must be a non-empty file path, not '-'."
            )
        # Match open_output_parent(), including lexical collapse of missing/..
        # and trusted root-level system aliases. Retain this destination for
        # input-preservation checks and writes, not just duplicate comparisons.
        path = _normalize_root_owned_alias(Path(filename))
        for previous in paths:
            if path.resolve() == previous.resolve() or (
                path.exists() and previous.exists() and path.samefile(previous)
            ):
                raise ValueError("Each report must have a distinct output path.")
        paths.append(path)
        reports.append((name, path))
    return reports
