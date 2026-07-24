#!/usr/bin/env python3
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

"""Create a public GitHub release for the version in ``pyproject.toml``."""

from __future__ import annotations

import argparse
import json
import subprocess
import tomllib
from pathlib import Path
from urllib.parse import quote


def _project_version(path: Path) -> str:
    with path.open("rb") as pyproject:
        project = tomllib.load(pyproject)["project"]
    return str(project["version"])


def _github_api_json(endpoint: str) -> dict[str, object] | None:
    """Return a GitHub API object, or ``None`` when *endpoint* is absent."""
    result = subprocess.run(
        ["gh", "api", endpoint],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if "HTTP 404" in result.stderr:
            return None
        result.check_returncode()

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"GitHub API returned invalid JSON for {endpoint}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"GitHub API returned an unexpected response for {endpoint}")
    return payload


def _git_object(payload: dict[str, object], source: str) -> tuple[str, str]:
    """Extract a Git object type and SHA from a GitHub API response."""
    object_payload = payload.get("object")
    if not isinstance(object_payload, dict):
        raise RuntimeError(f"GitHub API returned no Git object for {source}")

    object_type = object_payload.get("type")
    object_sha = object_payload.get("sha")
    if not isinstance(object_type, str) or not isinstance(object_sha, str):
        raise RuntimeError(f"GitHub API returned an invalid Git object for {source}")
    return object_type, object_sha


def _resolve_tag_target(repository: str, tag: str) -> str | None:
    """Resolve *tag* to its commit SHA, recursively peeling annotated tags."""
    escaped_repository = quote(repository, safe="/")
    escaped_tag = quote(tag, safe="")
    reference = _github_api_json(f"repos/{escaped_repository}/git/ref/tags/{escaped_tag}")
    if reference is None:
        return None

    object_type, object_sha = _git_object(reference, f"tag {tag}")
    seen_tag_objects: set[str] = set()
    while object_type == "tag":
        if object_sha in seen_tag_objects:
            raise RuntimeError(f"GitHub tag {tag} contains an annotated-tag cycle")
        seen_tag_objects.add(object_sha)

        tag_object = _github_api_json(f"repos/{escaped_repository}/git/tags/{object_sha}")
        if tag_object is None:
            raise RuntimeError(f"GitHub tag object {object_sha} disappeared while resolving {tag}")
        object_type, object_sha = _git_object(tag_object, f"tag object {object_sha}")

    if object_type != "commit":
        raise RuntimeError(f"GitHub tag {tag} resolves to unsupported object type {object_type!r}")
    return object_sha


def _create_tag_ref(repository: str, tag: str, target: str) -> bool:
    """Atomically create *tag* at *target*, returning ``False`` on a collision."""
    escaped_repository = quote(repository, safe="/")
    result = subprocess.run(
        [
            "gh",
            "api",
            "--method",
            "POST",
            f"repos/{escaped_repository}/git/refs",
            "-f",
            f"ref=refs/tags/{tag}",
            "-f",
            f"sha={target}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True
    if "HTTP 422" in result.stderr:
        return False
    result.check_returncode()
    raise AssertionError("unreachable")


def _ensure_tag_at_target(repository: str, tag: str, target: str) -> None:
    """Ensure *tag* exists at *target* before a release can use it."""
    tag_target = _resolve_tag_target(repository, tag)
    if tag_target is None:
        _create_tag_ref(repository, tag, target)
        tag_target = _resolve_tag_target(repository, tag)
        if tag_target is None:
            raise RuntimeError(f"GitHub tag {tag} was not found after its creation attempt")

    if tag_target != target:
        raise RuntimeError(
            f"Refusing to create GitHub release {tag}: existing tag resolves to "
            f"{tag_target}, not requested target {target}"
        )


def _release_exists(repository: str, tag: str) -> bool:
    """Report whether GitHub has a release for *tag*."""
    escaped_repository = quote(repository, safe="/")
    escaped_tag = quote(tag, safe="")
    return _github_api_json(f"repos/{escaped_repository}/releases/tags/{escaped_tag}") is not None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, help="GitHub repository (OWNER/REPO)")
    parser.add_argument("--target", required=True, help="Commit SHA for the release tag")
    parser.add_argument("--dry-run", action="store_true", help="Report without creating a release")
    args = parser.parse_args()

    version = _project_version(Path("pyproject.toml"))
    tag = f"v{version}"

    if args.dry_run:
        print(f"Would create GitHub release {tag} in {args.repository} at {args.target}")
        return

    _ensure_tag_at_target(args.repository, tag, args.target)
    if _release_exists(args.repository, tag):
        print(f"GitHub release {tag} already exists at {args.target}; nothing to do.")
        return

    subprocess.run(
        [
            "gh",
            "release",
            "create",
            tag,
            "--repo",
            args.repository,
            "--verify-tag",
            "--title",
            f"SkillSpector {tag}",
            "--generate-notes",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
