# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse additional file reports without changing scan execution."""

from pathlib import Path


def parse_reports(
    declarations: list[str] | None,
    formats: tuple[str, ...],
    output: Path | None = None,
) -> list[tuple[str, Path]]:
    """Accept repeatable FORMAT=PATH entries, rejecting duplicate destinations."""
    reports: list[tuple[str, Path]] = []
    paths = [output] if output is not None else []
    for declaration in declarations or []:
        name, separator, filename = declaration.partition("=")
        name = name.lower()
        if not separator or name not in formats or not filename or filename == "-":
            raise ValueError(
                f"--report expects FORMAT=PATH ({', '.join(formats)}); "
                "PATH must be a non-empty file path, not '-'."
            )
        path = Path(filename)
        for previous in paths:
            if path.resolve() == previous.resolve() or (
                path.exists() and previous.exists() and path.samefile(previous)
            ):
                raise ValueError("Each report must have a distinct output path.")
        paths.append(path)
        reports.append((name, path))
    return reports
