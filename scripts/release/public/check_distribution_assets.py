#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the release wheel and source distribution for bundled runtime assets."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

SOURCE_ROOT = Path("src")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    args = parser.parse_args()

    package = SOURCE_ROOT / "skillspector"
    rules = {path for path in (package / "yara_rules").rglob("*") if path.is_file()}
    registries = set((package / "providers").rglob("model_registry.yaml"))
    if not rules or not registries:
        parser.error("expected both YARA rules and provider model registries in src/skillspector")
    expected = {path.relative_to(SOURCE_ROOT).as_posix() for path in rules | registries}

    with zipfile.ZipFile(args.wheel) as archive:
        wheel_assets = {entry.filename for entry in archive.infolist() if not entry.is_dir()}
    with tarfile.open(args.sdist) as archive:
        sdist_assets = {
            member.name.partition("/")[2].removeprefix("src/")
            for member in archive
            if member.isfile() and member.name.partition("/")[2].startswith("src/")
        }

    for distribution, assets in ((args.wheel, wheel_assets), (args.sdist, sdist_assets)):
        missing = expected - assets
        if missing:
            parser.error(
                f"{distribution.name} is missing bundled assets: {', '.join(sorted(missing))}"
            )
    print(f"Verified {len(expected)} bundled runtime assets in both distributions.")


if __name__ == "__main__":
    main()
