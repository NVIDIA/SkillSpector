# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Input identity survives materialization without changing finding coordinates."""

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from skillspector.input_origin import input_origin
from skillspector.nodes.report import report
from skillspector.nodes.resolve_input import resolve_input
from skillspector.sarif_models import validate_sarif_report


@pytest.mark.parametrize(
    ("raw", "kind", "expected"),
    [
        (" ./literal name ", "directory", " ./literal name "),
        ("./literal?token=local#name.md", "file", "./literal?token=local#name.md"),
        (
            " https://user:secret@example.org:8443/a.md?custom=secret#secret ",
            "url",
            "https://example.org:8443/a.md",
        ),
        ("https://user:secret@[::1]:8443/a.md?secret", "url", "https://[::1]:8443/a.md"),
        ("https://example.org:bad/a.md?secret", "url", "<redacted-url>"),
        ("https://[bad/a.md", "url", "<redacted-url>"),
        ("git@example.org:team/repo.git", "git", "ssh://example.org/team/repo.git"),
        (
            "https://user:secret@github.com/team/repo/tree/main/sub?secret#secret",
            "git",
            "https://github.com/team/repo/tree/main/sub",
        ),
    ],
)
def test_origin_identity_and_redaction(raw, kind, expected):
    origin = input_origin(raw, kind)
    assert origin == {"source": expected, "type": kind}
    assert input_origin(origin["source"], kind) == origin


@pytest.mark.parametrize("kind", ["directory", "file", "zip"])
def test_real_materialization_retains_origin_and_cleanup(tmp_path, monkeypatch, kind):
    monkeypatch.chdir(tmp_path)
    skill = Path("skill ")
    skill.mkdir()
    (skill / "SKILL.md").write_text("# Hello")
    target = skill
    if kind == "file":
        target = skill / "SKILL.md"
    if kind == "zip":
        target = Path("skill.zip")
        with zipfile.ZipFile(target, "w") as z:
            z.write(skill / "SKILL.md", "SKILL.md")
    update = resolve_input({"input_path": str(target)})
    assert update["input_origin"] == {"source": str(target), "type": kind}
    assert Path(update["skill_path"]).is_dir()
    if update["temp_dir_for_cleanup"]:
        shutil.rmtree(update["temp_dir_for_cleanup"])
        assert not Path(update["skill_path"]).exists()
    assert target.exists()


@pytest.mark.parametrize("fmt", ["json", "sarif", "markdown", "terminal"])
def test_all_renderers_keep_origin_separate_and_redacted(fmt):
    state = {
        "output_format": fmt,
        "use_llm": False,
        "skill_path": "/tmp/materialized",
        "input_origin": {
            "type": "url",
            "source": "https://user:secret@example.org/SKILL.md?api_key=secret#secret",
        },
    }
    result = report(state)
    body = result["report_body"]
    assert "secret" not in body and "user:" not in body
    expected = {"type": "url", "source": "https://example.org/SKILL.md"}
    if fmt == "json":
        assert json.loads(body)["skill"]["origin"] == expected
        assert json.loads(body)["skill"]["source"] == "/tmp/materialized"
    elif fmt == "sarif":
        sarif = json.loads(body)
        validate_sarif_report(sarif)
        assert sarif["runs"][0]["properties"]["inputOrigin"] == expected
    else:
        assert "https://example.org/SKILL.md" in body and "/tmp/materialized" in body


@pytest.mark.parametrize(
    "raw", [None, {}, {"type": [], "source": "x"}, {"type": "url", "source": None}]
)
def test_legacy_or_invalid_state_omits_origin(raw):
    result = report({"output_format": "json", "use_llm": False, "input_origin": raw})
    assert "origin" not in json.loads(result["report_body"])["skill"]
