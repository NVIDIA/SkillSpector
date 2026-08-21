# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the bounded ASCII skeleton table used by security text views.

The input is the versioned ``confusables.txt`` published with Unicode UTS #39.
Only single-code-point sources whose skeleton is made entirely of ASCII letters
or digits are retained.  This is the complete UTS #39 subset relevant to the
ASCII security tokens matched by SkillSpector's deterministic analyzers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_line(line: str) -> tuple[int, str] | None:
    data = line.split("#", 1)[0].strip()
    if not data:
        return None
    fields = [field.strip() for field in data.split(";")]
    if len(fields) < 2:
        return None
    source_points = fields[0].split()
    if len(source_points) != 1:
        return None
    target = "".join(chr(int(point, 16)) for point in fields[1].split())
    if not target or not all(char.isascii() and char.isalnum() for char in target):
        return None
    source = int(source_points[0], 16)
    # Raw ASCII is already scanned directly.  Retaining ASCII-to-ASCII skeleton
    # rewrites (for example ``m`` -> ``rn``) would mutate otherwise ordinary
    # detector tokens after a neighboring non-ASCII character triggered this
    # derived view.
    if source <= 0x7F:
        return None
    return source, target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    mappings = dict(
        parsed
        for line in args.source.read_text(encoding="utf-8").splitlines()
        if (parsed := _parse_line(line)) is not None
    )
    lines = [
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        "",
        '"""Generated ASCII skeleton subset from Unicode UTS #39 confusables data."""',
        "",
        "from __future__ import annotations",
        "",
        f'UNICODE_CONFUSABLES_VERSION = "{args.version}"',
        f"# Source: https://www.unicode.org/Public/{args.version}/security/confusables.txt",
        "# The source data is governed by https://www.unicode.org/license.txt.",
        "ASCII_CONFUSABLE_SKELETON: dict[int, str] = {",
    ]
    lines.extend(
        f"    0x{codepoint:04X}: {json.dumps(target)},"
        for codepoint, target in sorted(mappings.items())
    )
    lines.extend(["}", ""])
    args.destination.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
