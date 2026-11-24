# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release archives must include the runtime's packaged data."""

from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/release/public/check_distribution_assets.py"


@pytest.mark.parametrize("missing_archive", [None, "wheel", "sdist"])
def test_release_distribution_requires_assets_in_both_formats(
    tmp_path: Path, missing_archive: str | None
) -> None:
    rule = Path("skillspector/yara_rules/agent_skills.yar")
    registry = Path("skillspector/providers/openai/model_registry.yaml")
    for resource in (rule, registry):
        source = tmp_path / "src" / resource
        source.parent.mkdir(parents=True)
        source.write_text("test asset", encoding="utf-8")

    wheel = tmp_path / "skillspector-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.write(tmp_path / "src" / rule, arcname=rule)
        if missing_archive != "wheel":
            archive.write(tmp_path / "src" / registry, arcname=registry)

    sdist = tmp_path / "skillspector-1.0.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(tmp_path / "src" / registry, arcname=f"skillspector-1.0.0/src/{registry}")
        if missing_archive != "sdist":
            archive.add(tmp_path / "src" / rule, arcname=f"skillspector-1.0.0/src/{rule}")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(wheel), str(sdist)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if missing_archive is None:
        assert result.returncode == 0, result.stderr
    else:
        distribution, asset = (wheel, registry) if missing_archive == "wheel" else (sdist, rule)
        assert result.returncode != 0
        assert distribution.name in result.stderr
        assert asset.as_posix() in result.stderr
