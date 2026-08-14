# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that wheels contain the non-Python resources used at runtime."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from hatchling.build import build_wheel

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "skillspector"


def test_wheel_contains_runtime_resources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Package all built-in YARA rules and provider model registries."""
    monkeypatch.chdir(REPO_ROOT)
    wheel_path = tmp_path / build_wheel(str(tmp_path))

    yara_rules = {
        path.relative_to(SOURCE_ROOT).as_posix()
        for path in (PACKAGE_ROOT / "yara_rules").rglob("*")
        if path.is_file()
    }
    model_registries = {
        path.relative_to(SOURCE_ROOT).as_posix()
        for path in (PACKAGE_ROOT / "providers").glob("*/model_registry.yaml")
    }
    expected_resources = yara_rules | model_registries

    assert yara_rules
    assert model_registries
    with zipfile.ZipFile(wheel_path) as wheel:
        assert expected_resources <= set(wheel.namelist())
