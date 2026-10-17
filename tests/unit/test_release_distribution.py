# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the release backend with a populated, non-VCS build cache."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

from hatchling.builders.sdist import SdistBuilder


def test_source_distribution_excludes_release_uv_cache(tmp_path: Path) -> None:
    """A release from an extracted archive must not ship dependency cache files."""
    root = Path(__file__).resolve().parents[2]
    project = tmp_path / "project"
    project.mkdir()
    shutil.copyfile(root / "pyproject.toml", project / "pyproject.toml")
    (project / "README.md").write_text("# Synthetic package fixture\n", encoding="utf-8")
    source = project / "src" / "skillspector"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("", encoding="utf-8")
    cache = project / ".uv-cache" / "archive-v0" / "fixture"
    cache.mkdir(parents=True)
    (cache / "private-build-input.txt").write_text("synthetic cache sentinel", encoding="utf-8")

    distribution = next(SdistBuilder(str(project)).build(directory=str(tmp_path / "dist")))

    with tarfile.open(distribution, "r:gz") as archive:
        names = [Path(name) for name in archive.getnames()]
    assert any(path.parts[1:] == ("src", "skillspector", "__init__.py") for path in names)
    assert all(".uv-cache" not in path.parts for path in names)
