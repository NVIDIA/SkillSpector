# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A passive declaration is proven syntax, never a generic quoted-path waiver."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import skillspector.nodes.finalize_inspection_ledger as finalizer
from skillspector.inspection_ledger import (
    LedgerOutcome,
    LedgerReason,
    LedgerRecordType,
    ledger_event,
)
from skillspector.references import resolve_bundle_references_with_metadata
from tests.nodes.test_finalize_inspection_ledger import _VALID_PASSIVE_PNG

TARGET = "assets/diagram.png"


def _resolve(source: str):
    return resolve_bundle_references_with_metadata(
        Path("."), source_path="SKILL.md", source_text=source, known_paths=["SKILL.md", TARGET]
    )


@pytest.mark.parametrize(
    "label",
    ["Asset", "Image", "Illustration", "Diagram", "Figure", "Screenshot", "Photo", "Picture"],
)
@pytest.mark.parametrize("ending", ["", ".", ".\n", ".\r\n"])
def test_standalone_caption_has_exact_source_provenance(label: str, ending: str) -> None:
    source = f"# Guide\n\n{label}: `{TARGET}`{ending}"
    result = _resolve(source)
    assert result.complete
    assert len(result.records) == 1
    record = result.records[0]
    assert record["reference_kind"] == "documentation_image"
    assert record["line"] == 3
    assert record["column"] == len(label) + 4
    assert record["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()


@pytest.mark.parametrize(
    "source",
    [
        "Run `assets/diagram.png`.",
        "Asset: 'assets/diagram.png'.",
        'Asset: "assets/diagram.png".',
        "Asset: `assets/diagram.png'.",
        'Asset: "assets/diagram.png`.',
        "Asset: ``assets/diagram.png``.",
        "Asset: `assets/diagram.png``.",
        "Asset: `assets/diagram.png` && sh assets/diagram.png",
        "sh -c 'Asset: `assets/diagram.png`.'",
        "Asset: `assets/diagram.png` `other.png`.",
        "```\nAsset: `assets/diagram.png`.\n```",
        "~~~\n\nAsset: `assets/diagram.png`.\n\n~~~",
        "    Asset: `assets/diagram.png`.",
        "  Asset: `assets/diagram.png`.",
        "<div>\n\nAsset: `assets/diagram.png`.\n</div>",
        "<!--\n\nAsset: `assets/diagram.png`.\n-->",
        "`open\n\nAsset: `assets/diagram.png`.\n",
        "``open\n\nAsset: `assets/diagram.png`.\n",
        "- item\n\n  Asset: `assets/diagram.png`.",
        "> command\nAsset: `assets/diagram.png`.",
        "Execute the following:\nAsset: `assets/diagram.png`.",
        "Asset: `assets/diagram.png`.\nThen execute it.",
        "---\n\nAsset: `assets/diagram.png`.\n\n---",
        "\ufeff---\n\nAsset: `assets/diagram.png`.\n\n---",
        "--- \n\nAsset: `assets/diagram.png`.\n\n---",
        "Asset: `assets/diagram.png`.;",
    ],
)
def test_ambiguous_or_executable_syntax_never_acquires_passive_role(source: str) -> None:
    result = _resolve(source)
    assert result.complete
    assert result.records
    assert all(row["reference_kind"] != "documentation_image" for row in result.records)


def _state(source: str = f"Asset: `{TARGET}`.\n") -> dict:
    references = _resolve(source).records
    for reference in references:
        reference["disposition"] = "out_of_scope"
    return {
        "components": ["SKILL.md", TARGET],
        "findings": [],
        "effective_finding_ids": [],
        "raw_file_cache": {"SKILL.md": source.encode(), TARGET: _VALID_PASSIVE_PNG},
        "artifact_inventory": [
            {
                "path": "SKILL.md",
                "content_kind": "text",
                "disposition": "analyzed",
                "size_bytes": len(source.encode()),
                "referenced": False,
            },
            {
                "path": TARGET,
                "content_kind": "binary",
                "disposition": "out_of_scope",
                "size_bytes": len(_VALID_PASSIVE_PNG),
                "referenced": True,
            },
        ],
        "artifact_references": references,
        "reference_resolution": {"complete": True, "limitations": []},
        "inspection_ledger": [
            ledger_event(
                outcome=LedgerOutcome.OUT_OF_SCOPE,
                record_type=LedgerRecordType.SYSTEM,
                phase="static",
                path=TARGET,
                reason=LedgerReason.OPAQUE_CONTENT,
            )
        ],
        "analyzer_status_events": [],
    }


def test_verified_caption_removes_only_format_ae1_and_preserves_partial_coverage() -> None:
    result = finalizer.finalize_inspection_ledger(_state())
    assert result["findings"] == []
    assert result["analysis_completeness"]["is_complete"] is False
    assert result["analysis_completeness"]["entirely_uninspected_files"] > 0
    assert result["analysis_completeness"]["scope_exclusions"]


@pytest.mark.parametrize(
    "mutation",
    [
        "line",
        "column",
        "digest",
        "kind",
        "source_bytes",
        "missing_bytes",
        "source_size",
        "source_binary",
        "missing_resolution",
        "resolution_limit",
        "missing_ledger",
        "fatal",
        "mixed_limit",
        "target_size",
        "target_trailing",
        "duplicate_inventory",
        "unhashable_column",
    ],
)
def test_forged_or_incomplete_evidence_keeps_ae1(mutation: str) -> None:
    state = _state()
    row = state["artifact_references"][0]
    if mutation in {"line", "column"}:
        row[mutation] += 1
    elif mutation == "digest":
        row["source_sha256"] = "0" * 64
    elif mutation == "kind":
        row["reference_kind"] = "quoted_or_code"
    elif mutation == "source_bytes":
        state["raw_file_cache"]["SKILL.md"] = b"Run:   `assets/diagram.png`.\n"
    elif mutation == "missing_bytes":
        del state["raw_file_cache"]["SKILL.md"]
    elif mutation == "source_size":
        state["artifact_inventory"][0]["size_bytes"] += 1
    elif mutation == "source_binary":
        state["artifact_inventory"][0]["content_kind"] = "binary"
    elif mutation == "missing_resolution":
        del state["reference_resolution"]
    elif mutation == "resolution_limit":
        state["reference_resolution"] = {"complete": False, "limitations": ["output_records"]}
    elif mutation == "missing_ledger":
        state["inspection_ledger"] = []
    elif mutation == "fatal":
        state["inspection_ledger"][0]["fatal"] = True
    elif mutation == "mixed_limit":
        state["inspection_ledger"].append(
            ledger_event(
                outcome=LedgerOutcome.PARTIAL,
                record_type=LedgerRecordType.SYSTEM,
                phase="static",
                path=TARGET,
                reason=LedgerReason.STATIC_PARSE_LIMIT,
            )
        )
    elif mutation == "target_size":
        state["artifact_inventory"][1]["size_bytes"] += 1
    elif mutation == "target_trailing":
        state["raw_file_cache"][TARGET] += b"uninspected trailing content"
        state["artifact_inventory"][1]["size_bytes"] = len(state["raw_file_cache"][TARGET])
    elif mutation == "duplicate_inventory":
        state["artifact_inventory"].append(dict(state["artifact_inventory"][0]))
    elif mutation == "unhashable_column":
        row["column"] = []
    assert [finding.rule_id for finding in finalizer._reference_coverage_findings(state)] == ["AE1"]


def test_an_independent_normalized_command_use_still_blocks() -> None:
    state = _state(f"Asset: `{TARGET}`.\n\nRun `bash ./assets/diagram.png`.\n")
    kinds = [row["reference_kind"] for row in state["artifact_references"]]
    assert "documentation_image" in kinds and "inline_command" in kinds
    assert [finding.rule_id for finding in finalizer._reference_coverage_findings(state)] == [
        "AE1",
        "AE1",
    ]


@pytest.mark.parametrize("active", ["Run `bash ./assets/diagram.png`.", "Run `diagram.png`."])
def test_reparse_retains_mixed_use_even_if_its_public_record_is_missing(active: str) -> None:
    state = _state(f"Asset: `{TARGET}`.\n\n{active}\n")
    state["artifact_references"] = [state["artifact_references"][0]]
    assert [finding.rule_id for finding in finalizer._reference_coverage_findings(state)] == ["AE1"]


def test_source_corroboration_is_cached_and_budget_failure_is_conservative(monkeypatch) -> None:
    state = _state((f"Asset: `{TARGET}`.\n\n") * 200)
    calls = []
    original = finalizer.documentation_image_provenance

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(finalizer, "documentation_image_provenance", counted)
    assert finalizer._reference_coverage_findings(state) == []
    assert calls == ["SKILL.md"]
    monkeypatch.setattr(finalizer, "MAX_REFERENCE_SOURCE_BYTES", 1)
    assert finalizer._reference_coverage_findings(state)
    assert calls == ["SKILL.md"]


def test_expired_source_proof_does_not_accept_a_role(monkeypatch) -> None:
    state = _state()
    monkeypatch.setattr(finalizer, "MAX_REFERENCE_RUNTIME_SECONDS", 0)
    assert finalizer._reference_coverage_findings(state)


def test_repeated_references_verify_asset_bytes_only_once(monkeypatch) -> None:
    state = _state((f"Asset: `{TARGET}`.\n\n") * 200)
    calls = []
    original = finalizer._has_verified_passive_png

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(finalizer, "_has_verified_passive_png", counted)
    assert finalizer._reference_coverage_findings(state) == []
    assert calls == [TARGET]


@pytest.mark.parametrize(
    "budget",
    [
        "_MAX_PASSIVE_RASTER_TOTAL_BYTES",
        "_MAX_PASSIVE_RASTER_PROOFS",
        "_MAX_PASSIVE_RASTER_SECONDS",
    ],
)
def test_exhausted_format_proof_budget_retains_ae1(monkeypatch, budget: str) -> None:
    monkeypatch.setattr(finalizer, budget, 0)
    assert finalizer._reference_coverage_findings(_state())


def _jpeg_state(source: str) -> dict:
    from tests.test_passive_jpeg import _jpeg

    target = "assets/diagram.jpg"
    state = _state(source.replace(target, TARGET))
    state["components"] = ["SKILL.md", target]
    state["raw_file_cache"] = {"SKILL.md": source.encode(), target: _jpeg()}
    state["artifact_inventory"][0]["size_bytes"] = len(source.encode())
    state["artifact_inventory"][1].update(path=target, size_bytes=len(_jpeg()))
    state["inspection_ledger"] = [
        ledger_event(
            outcome=LedgerOutcome.OUT_OF_SCOPE,
            record_type=LedgerRecordType.SYSTEM,
            phase="static",
            path=target,
            reason=LedgerReason.OPAQUE_CONTENT,
        )
    ]
    records = resolve_bundle_references_with_metadata(
        Path("."), source_path="SKILL.md", source_text=source, known_paths=["SKILL.md", target]
    ).records
    for row in records:
        row["disposition"] = "out_of_scope"
    state["artifact_references"] = records
    return state


@pytest.mark.parametrize(
    "source", ["![Diagram](assets/diagram.jpg)", "Asset: `assets/diagram.jpg`."]
)
def test_new_jpeg_roles_require_canonical_proof(source: str) -> None:
    state = _jpeg_state(source)
    assert (
        state["artifact_references"][0]["source_sha256"]
        == hashlib.sha256(source.encode()).hexdigest()
    )
    assert finalizer._reference_coverage_findings(state) == []
    state["artifact_references"][0]["column"] += 1
    assert finalizer._reference_coverage_findings(state)


@pytest.mark.parametrize(
    "mutation", ["quoted", "frontmatter", "missing_mixed", "missing_digest", "incomplete"]
)
def test_rendered_jpeg_cannot_gain_proof_from_forged_or_incomplete_records(mutation: str) -> None:
    source = "![Diagram](assets/diagram.jpg)"
    if mutation == "quoted":
        source = "Read `assets/diagram.jpg`."
    elif mutation == "frontmatter":
        source = "---\nname: example\ndescription: |\n  ![Diagram](assets/diagram.jpg)\n---\n"
    elif mutation == "missing_mixed":
        source += "\n\nRun `bash assets/diagram.jpg`."
    state = _jpeg_state(source)
    state["artifact_references"] = [state["artifact_references"][0]]
    row = state["artifact_references"][0]
    row["reference_kind"] = "markdown_image"
    row["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    if mutation == "missing_digest":
        row.pop("source_sha256")
    elif mutation == "incomplete":
        state["reference_resolution"]["complete"] = False
    assert [finding.rule_id for finding in finalizer._reference_coverage_findings(state)] == ["AE1"]
