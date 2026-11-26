# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the release helper exposes only the two pinned GitHub assets."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts/release/public/download_release_distributions.py"
)
WHEEL_NAME = "skillspector-2.13.7-py3-none-any.whl"
SDIST_NAME = "skillspector-2.13.7.tar.gz"
FILES = {WHEEL_NAME: b"example wheel", SDIST_NAME: b"example sdist"}
PINS = {name: hashlib.sha256(content).hexdigest() for name, content in FILES.items()}
RELEASE_ENDPOINT = "repos/NVIDIA/SkillSpector/releases/tags/v2.13.7"
ASSET_ENDPOINT = "repos/NVIDIA/SkillSpector/releases/assets/"


def _release() -> dict:
    return {
        "tag_name": "v2.13.7",
        "draft": False,
        "prerelease": False,
        "published_at": "2026-11-24T05:58:41Z",
        "assets": [
            {"id": asset_id, "name": name, "state": "uploaded", "digest": f"sha256:{digest}"}
            for asset_id, (name, digest) in enumerate(PINS.items(), start=1)
        ],
    }


def _run(
    tmp_path: Path,
    release: dict,
    *,
    repository: str = "NVIDIA/SkillSpector",
    tag: str = "v2.13.7",
    wheel_sha256: str = PINS[WHEEL_NAME],
    sdist_bytes: bytes = FILES[SDIST_NAME],
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    (tmp_path / "release.json").write_text(json.dumps(release), encoding="utf-8")
    (tmp_path / "wheel").write_bytes(FILES[WHEEL_NAME])
    (tmp_path / "sdist").write_bytes(sdist_bytes)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gh = fake_bin / "gh"
    gh.write_text(
        r"""#!/usr/bin/env python3
import sys
from pathlib import Path

args = sys.argv[1:]
assert args[:5] == ["api", "--hostname", "github.com", "--method", "GET"], args
endpoint = args[5]
with Path("api-calls").open("a", encoding="utf-8") as recorded:
    recorded.write(endpoint + "\n")
if endpoint == "repos/NVIDIA/SkillSpector/releases/tags/v2.13.7":
    sys.stdout.buffer.write(Path("release.json").read_bytes())
else:
    assert args[6:] == ["--header", "Accept: application/octet-stream"], args
    asset = {
        "repos/NVIDIA/SkillSpector/releases/assets/1": "wheel",
        "repos/NVIDIA/SkillSpector/releases/assets/2": "sdist",
    }[endpoint]
    sys.stdout.buffer.write(Path(asset).read_bytes())
""",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}
    invocation = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repository",
            repository,
            "--tag",
            tag,
            "--wheel-sha256",
            wheel_sha256,
            "--sdist-sha256",
            PINS[SDIST_NAME],
            "--output",
            "dist",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    recorded = tmp_path / "api-calls"
    calls = recorded.read_text(encoding="utf-8").splitlines() if recorded.exists() else []
    return invocation, calls


def test_downloads_exactly_the_two_pinned_files(tmp_path: Path) -> None:
    release = _release()
    release["assets"].append({"id": 99, "name": "../unrelated.whl", "state": "uploaded"})
    result, calls = _run(tmp_path, release, wheel_sha256=PINS[WHEEL_NAME].upper())

    assert result.returncode == 0, result.stderr
    assert calls == [RELEASE_ENDPOINT, f"{ASSET_ENDPOINT}1", f"{ASSET_ENDPOINT}2"]
    assert {path.name: path.read_bytes() for path in (tmp_path / "dist").iterdir()} == FILES
    assert not list(tmp_path.glob(".dist-*"))


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"repository": "NVIDIA/.."}, "--repository"),
        ({"repository": "NVIDIA/SkillSpector?per_page=1"}, "--repository"),
        ({"tag": "v2.13.7-rc.1"}, "--tag"),
        ({"tag": "v02.13.7"}, "--tag"),
        ({"wheel_sha256": "not-a-hash"}, "--wheel-sha256"),
    ],
)
def test_rejects_unsafe_input_before_fetching(tmp_path: Path, arguments: dict, error: str) -> None:
    result, calls = _run(tmp_path, _release(), **arguments)

    assert result.returncode != 0
    assert error in result.stderr
    assert calls == []
    assert not (tmp_path / "dist").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tag_name", "v2.13.6"),
        ("draft", True),
        ("prerelease", True),
        ("published_at", None),
        ("published_at", "not-a-time"),
    ],
)
def test_rejects_unexpected_release_before_downloading(
    tmp_path: Path, field: str, value: object
) -> None:
    release = _release()
    release[field] = value
    result, calls = _run(tmp_path, release)

    assert result.returncode != 0
    assert "release" in result.stderr
    assert calls == [RELEASE_ENDPOINT]
    assert not (tmp_path / "dist").exists()


@pytest.mark.parametrize("case", ["missing", "duplicate", "digest", "id", "state"])
def test_validates_both_asset_metadata_records_before_download(tmp_path: Path, case: str) -> None:
    release = _release()
    assets = release["assets"]
    match case:
        case "missing":
            assets.pop()
        case "duplicate":
            assets.append(assets[1].copy())
        case "digest":
            assets[1]["digest"] = f"sha256:{'a' * 64}"
        case "id":
            assets[1]["id"] = True
        case "state":
            assets[1]["state"] = "starter"
    result, calls = _run(tmp_path, release)

    assert result.returncode != 0
    assert SDIST_NAME in result.stderr
    assert calls == [RELEASE_ENDPOINT]
    assert not (tmp_path / "dist").exists()


def test_bad_second_download_leaves_no_upload_directory(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, _release(), sdist_bytes=b"different source distribution")

    assert result.returncode != 0
    assert SDIST_NAME in result.stderr
    assert "SHA-256" in result.stderr
    assert calls == [RELEASE_ENDPOINT, f"{ASSET_ENDPOINT}1", f"{ASSET_ENDPOINT}2"]
    assert not (tmp_path / "dist").exists()
    assert not list(tmp_path.glob(".dist-*"))


@pytest.mark.parametrize("kind", ["empty", "nonempty", "symlink"])
def test_refuses_an_existing_output_path(tmp_path: Path, kind: str) -> None:
    output = tmp_path / "dist"
    if kind == "symlink":
        output.symlink_to(tmp_path / "missing")
    else:
        output.mkdir()
        if kind == "nonempty":
            (output / "unexpected.whl").write_bytes(b"unverified")
    result, calls = _run(tmp_path, _release())

    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert calls == []
    if kind == "symlink":
        assert output.is_symlink()
    elif kind == "nonempty":
        assert (output / "unexpected.whl").read_bytes() == b"unverified"
    else:
        assert not list(output.iterdir())
