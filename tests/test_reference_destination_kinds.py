# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete Markdown destinations retain conservative reference-use provenance."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.mcp_server import run_scan
from skillspector.references import resolve_bundle_references_with_metadata
from tests.nodes.analyzers.test_documentation_reconstruction import (
    successful_llm_transport as successful_llm_transport,
)


@pytest.mark.parametrize("present", [True, False])
@pytest.mark.parametrize(
    ("source", "target", "kind"),
    [
        ("[Guide](docs/user%20guide.md)", "docs/user guide.md", "markdown_link"),
        ("[Guide](<docs/user guide.md>)", "docs/user guide.md", "markdown_link"),
        ("[Guide](docs/guide(v1).md)", "docs/guide(v1).md", "markdown_link"),
        (r"[Guide](docs/guide\(v1\).md)", "docs/guide(v1).md", "markdown_link"),
        (
            "[Guide][manual]\n\n[manual]: docs/user%20guide.md",
            "docs/user guide.md",
            "markdown_link",
        ),
        ("![Chart](assets/chart%20one.png)", "assets/chart one.png", "markdown_image"),
        ("![](assets/chart%20one.png)", "assets/chart one.png", "markdown_image"),
        (
            "![Chart][image]\n\n[image]: assets/chart%20one.png",
            "assets/chart one.png",
            "markdown_link",
        ),
        ("![Chart](assets/chart(v1).png)", "assets/chart(v1).png", "quoted_or_code"),
        (r"![Chart](assets/chart.png\))", "assets/chart.png)", "quoted_or_code"),
        (
            '![Chart](assets/chart%20one.png "see [sample](missing.md)")',
            "assets/chart one.png",
            "quoted_or_code",
        ),
        (
            r'![Chart](assets/chart%20one.png "A \"quoted\" title")',
            "assets/chart one.png",
            "quoted_or_code",
        ),
    ],
)
def test_complete_destinations_preserve_reference_kind(
    tmp_path: Path, source: str, target: str, kind: str, present: bool
) -> None:
    result = resolve_bundle_references_with_metadata(
        tmp_path,
        source_path="SKILL.md",
        source_text=source,
        known_paths=["SKILL.md", target] if present else ["SKILL.md"],
    )

    assert result.complete is True
    assert len(result.records) == 1
    record = result.records[0]
    assert record["target_path"] == (target if present else None)
    assert record["status"] == ("resolved" if present else "missing")
    assert record["reference_kind"] == kind


@pytest.mark.parametrize(
    ("source", "kind"),
    [
        ("`![Chart](assets/chart%20one.png)`", "quoted_or_code"),
        ("```markdown\n![Chart](assets/chart%20one.png)\n```", "quoted_or_code"),
        ("    ![Chart](assets/chart%20one.png)", "quoted_or_code"),
        ("<!--\n![Chart](assets/chart%20one.png)\n-->", "quoted_or_code"),
        (r"\![Chart](assets/chart%20one.png)", "plain_path"),
    ],
)
def test_encoded_images_in_literal_context_remain_non_passive(
    tmp_path: Path, source: str, kind: str
) -> None:
    result = resolve_bundle_references_with_metadata(
        tmp_path,
        source_path="SKILL.md",
        source_text=source + "\n\n![Rendered](assets/rendered%20chart.png)",
        known_paths=["SKILL.md", "assets/chart one.png", "assets/rendered chart.png"],
    )

    assert result.complete is True
    assert len(result.records) == 2
    assert all(record["status"] == "resolved" for record in result.records)
    assert {(record["target_path"], record["reference_kind"]) for record in result.records} == {
        ("assets/chart one.png", kind),
        ("assets/rendered chart.png", "markdown_image"),
    }


@pytest.mark.parametrize("marker", ["", "!"])
def test_complete_destination_keeps_distinct_visible_label_reference(
    tmp_path: Path, marker: str
) -> None:
    result = resolve_bundle_references_with_metadata(
        tmp_path,
        source_path="SKILL.md",
        source_text=f"{marker}[Inspect docs/extra.md](assets/chart%20one.png)",
        known_paths=["SKILL.md", "docs/extra.md", "assets/chart one.png"],
    )

    assert result.complete is True
    assert len(result.records) == 2
    assert {(record["target_path"], record["reference_kind"]) for record in result.records} == {
        ("docs/extra.md", "plain_path"),
        ("assets/chart one.png", "markdown_image" if marker else "markdown_link"),
    }


def _png_payload() -> bytes:
    def chunk(kind: bytes, content: bytes) -> bytes:
        return (
            struct.pack(">I", len(content))
            + kind
            + content
            + struct.pack(">I", zlib.crc32(kind + content))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\x40\x80\xc0\xff"))
        + chunk(b"IEND", b"")
    )


@pytest.mark.parametrize("channel", ["cli", "mcp"])
@pytest.mark.parametrize(
    ("marker", "present", "use_llm"),
    [
        ("!", True, False),
        ("!", True, True),
        ("", True, False),
        ("", True, True),
        ("!", False, False),
        ("!", False, True),
    ],
    ids=[
        "passive-image-no-llm",
        "passive-image-successful-mocked-llm",
        "ordinary-link-no-llm",
        "ordinary-link-successful-mocked-llm",
        "missing-image-no-llm",
        "missing-image-successful-mocked-llm",
    ],
)
async def test_encoded_destination_reporting_preserves_opaque_coverage(
    tmp_path: Path,
    successful_llm_transport: list[str],
    channel: str,
    use_llm: bool,
    marker: str,
    present: bool,
) -> None:
    (tmp_path / "SKILL.md").write_text(
        "---\nname: chart-guide\ndescription: Explain the chart colors.\n---\n"
        f"# Chart guide\n\n{marker}[Chart](assets/chart%20one.png)\n",
        encoding="utf-8",
    )
    target = "assets/chart one.png"
    if present:
        (tmp_path / "assets").mkdir()
        (tmp_path / target).write_bytes(_png_payload())

    if channel == "cli":
        args = ["scan", str(tmp_path), "--format", "json", "--fail-on-incomplete"]
        if not use_llm:
            args.append("--no-llm")
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 1, result.output
        report = json.loads(result.stdout)
    else:
        verdict = await run_scan(str(tmp_path), use_llm=use_llm, output_format="json")
        assert verdict["safe_to_install"] is (not present)
        assert verdict["llm_used"] is use_llm
        report = json.loads(verdict["report"])

    assert report["execution_successful"] is True
    completeness = report["analysis_completeness"]
    assert completeness["is_complete"] is False
    assert len(completeness["references"]) == 1
    reference = completeness["references"][0]
    assert reference["status"] == ("resolved" if present else "missing")
    assert reference["target_path"] == (target if present else None)
    assert reference["reference_kind"] == ("markdown_image" if marker else "markdown_link")
    assert any(issue["id"] == "AE1" for issue in report["issues"]) is (present and not marker)
    if marker:
        assert report["risk_assessment"]["recommendation"] == "CAUTION"
    if present:
        assert completeness["entirely_uninspected_files"] == 1
        assert any(item["path"] == target for item in completeness["ledger_exceptions"])
    if use_llm:
        # AE1 concerns bytes excluded from the LLM cache. Its locally retained
        # meta finding keeps the existing conservative runtime caveat, even
        # though all three semantic analyzers below completed successfully.
        if marker:
            assert report["metadata"].get("llm_degraded", False) is False
        else:
            assert report["metadata"]["llm_degraded"] is True
            assert "semantic runtime telemetry was incomplete" in report["metadata"]["llm_error"]
        assert report["metadata"]["llm_calls_succeeded"] >= 3
        assert (
            report["metadata"]["llm_calls_succeeded"] == report["metadata"]["llm_calls_attempted"]
        )
    else:
        assert successful_llm_transport == []
    semantic_statuses = {
        item["analyzer_id"]: item["status"]
        for item in completeness["analyzer_statuses"]
        if item["analyzer_id"]
        in {
            "semantic_security_discovery",
            "semantic_developer_intent",
            "semantic_quality_policy",
        }
    }
    assert len(semantic_statuses) == 3
    assert set(semantic_statuses.values()) == {"completed" if use_llm else "disabled"}
