#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download the two pinned distributions from a published GitHub release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

_GITHUB_GET = ["gh", "api", "--hostname", "github.com", "--method", "GET"]
_REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+")
_STABLE_TAG = re.compile(r"v(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){2}")
_SHA256 = re.compile(r"[0-9A-Fa-f]{64}")


def _published_assets(release: object, tag: str, pins: dict[str, str]) -> dict[str, int]:
    """Validate release metadata and resolve asset IDs before downloading bytes."""
    if (
        not isinstance(release, dict)
        or release.get("tag_name") != tag
        or release.get("draft") is not False
        or release.get("prerelease") is not False
    ):
        raise ValueError(f"expected a published stable release for {tag}")

    published_at = release.get("published_at")
    try:
        published = (
            isinstance(published_at, str)
            and datetime.fromisoformat(published_at).tzinfo is not None
        )
    except ValueError:
        published = False
    if not published:
        raise ValueError(f"release {tag} has no valid publication timestamp")

    assets = release.get("assets")
    if not isinstance(assets, list):
        raise ValueError(f"release {tag} has invalid asset metadata")

    asset_ids: dict[str, int] = {}
    for name, pin in pins.items():
        matches = [
            asset for asset in assets if isinstance(asset, dict) and asset.get("name") == name
        ]
        if len(matches) != 1:
            raise ValueError(f"release {tag} must contain exactly one {name}")
        asset = matches[0]
        asset_id = asset.get("id")
        if type(asset_id) is not int or asset_id <= 0 or asset.get("state") != "uploaded":
            raise ValueError(f"{name} has invalid uploaded asset metadata")
        digest = asset.get("digest")
        if not isinstance(digest, str) or digest.lower() != f"sha256:{pin}":
            raise ValueError(f"{name} GitHub SHA-256 digest does not match the supplied pin")
        asset_ids[name] = asset_id
    return asset_ids


def _require_fresh_output(output: Path) -> None:
    """Check the output path without following its final component."""
    try:
        output.lstat()
    except FileNotFoundError:
        return
    raise ValueError(f"output path already exists: {output}")


def _download(repository: str, tag: str, pins: dict[str, str], output: Path) -> None:
    _require_fresh_output(output)
    endpoint = f"repos/{repository}/releases"
    result = subprocess.run(
        [*_GITHUB_GET, f"{endpoint}/tags/{tag}"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    asset_ids = _published_assets(json.loads(result.stdout), tag, pins)

    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{output.name}-") as temporary:
        staged = Path(temporary) / "verified"
        staged.mkdir()
        for name, asset_id in asset_ids.items():
            path = staged / name
            with path.open("wb") as distribution:
                subprocess.run(
                    [
                        *_GITHUB_GET,
                        f"{endpoint}/assets/{asset_id}",
                        "--header",
                        "Accept: application/octet-stream",
                    ],
                    stdout=distribution,
                    check=True,
                )
            with path.open("rb") as distribution:
                actual = hashlib.file_digest(distribution, "sha256").hexdigest()
            if actual != pins[name]:
                raise ValueError(
                    f"{name} downloaded SHA-256 digest does not match the supplied pin"
                )

        _require_fresh_output(output)
        staged.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, help="GitHub repository (OWNER/REPO)")
    parser.add_argument("--tag", required=True, help="Stable release tag (vX.Y.Z)")
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--sdist-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path, help="New distribution directory")
    args = parser.parse_args()

    if not _REPOSITORY.fullmatch(args.repository) or args.repository.split("/")[1] in {".", ".."}:
        parser.error("--repository must be a GitHub OWNER/REPO")
    if not _STABLE_TAG.fullmatch(args.tag):
        parser.error("--tag must be a stable vX.Y.Z release tag")
    for option, digest in (
        ("--wheel-sha256", args.wheel_sha256),
        ("--sdist-sha256", args.sdist_sha256),
    ):
        if not _SHA256.fullmatch(digest):
            parser.error(f"{option} must be a 64-character hexadecimal SHA-256 digest")

    version = args.tag.removeprefix("v")
    pins = {
        f"skillspector-{version}-py3-none-any.whl": args.wheel_sha256.lower(),
        f"skillspector-{version}.tar.gz": args.sdist_sha256.lower(),
    }
    try:
        _download(args.repository, args.tag, pins, args.output)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(f"Downloaded and verified both distributions for {args.tag}.")


if __name__ == "__main__":
    main()
