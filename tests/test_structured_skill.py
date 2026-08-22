# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resource-bound tests for structured AISOP/AISP extraction."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from skillspector import structured_skill as module


def _bundle(name: str, *, functions: object | None = None) -> bytes:
    payload = [
        {
            "role": "system",
            "content": {"protocol": "AISOP V1", "format": "workflow"},
        },
        {
            "role": "user",
            "content": {
                "aisop": {"main": "graph TD"},
                "functions": functions if functions is not None else {name: {}},
            },
        },
    ]
    return json.dumps(payload).encode()


def _limitation(result: module.StructuredSkillExtractionResult, resource: str):
    return next(item for item in result.limitations if item.resource == resource)


def test_cache_api_never_traverses_or_rereads_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Supplied bytes are authoritative; the cache API performs no filesystem I/O."""

    def unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("cache extraction attempted filesystem I/O")

    monkeypatch.setattr(os, "scandir", unexpected)
    monkeypatch.setattr(Path, "rglob", unexpected)
    monkeypatch.setattr(Path, "read_text", unexpected)
    monkeypatch.setattr(Path, "open", unexpected)

    result = module.extract_structured_skill_context_from_cache(
        tmp_path,
        ["workflow.aisop.json"],
        raw_file_cache={"workflow.aisop.json": _bundle("cached")},
    )

    assert result.complete is True
    assert result.context is not None
    assert result.context["workflow_nodes"] == ["cached"]
    assert result.context["bundle_path"] == str(tmp_path / "workflow.aisop.json")


def test_cache_api_accepts_bounded_text_cache(tmp_path: Path) -> None:
    content = _bundle("text-cache").decode()
    result = module.extract_structured_skill_context_from_cache(
        tmp_path,
        file_cache={"workflow.aisop.json": content},
    )

    assert result.context is not None
    assert result.context["workflow_nodes"] == ["text-cache"]


def test_candidates_are_processed_in_deterministic_lexical_order(tmp_path: Path) -> None:
    result = module.extract_structured_skill_context_from_cache(
        tmp_path,
        ["z.aisop.json", "a.aisop.json"],
        raw_file_cache={
            "z.aisop.json": _bundle("last"),
            "a.aisop.json": _bundle("first"),
        },
    )

    assert result.context is not None
    assert result.context["workflow_nodes"] == ["first"]
    assert result.candidates_examined == 1


def test_oversized_document_is_partial_without_json_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_loads(*args: object, **kwargs: object) -> object:
        raise AssertionError("oversized structured input reached json.loads")

    monkeypatch.setattr(module.json, "loads", unexpected_loads)
    result = module.extract_structured_skill_context_from_cache(
        tmp_path,
        ["huge.aisop.json"],
        raw_file_cache={"huge.aisop.json": b"x" * (module.MAX_STRUCTURED_DOCUMENT_BYTES + 1)},
    )

    assert result.context is None
    assert result.complete is False
    limitation = _limitation(result, "structured_document_bytes")
    assert limitation.reason_code == "size_limit"
    assert limitation.observed_bytes == module.MAX_STRUCTURED_DOCUMENT_BYTES + 1
    assert limitation.limit_bytes == module.MAX_STRUCTURED_DOCUMENT_BYTES
    assert limitation.as_ledger_metadata()["outcome"] == "partial"


def test_later_valid_context_retains_earlier_partial_limitation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "MAX_STRUCTURED_DOCUMENT_BYTES", 512)
    valid = _bundle("bounded-valid")
    assert len(valid) < 512

    result = module.extract_structured_skill_context_from_cache(
        tmp_path,
        ["b.aisop.json", "a.aisop.json"],
        raw_file_cache={
            "a.aisop.json": b"x" * 513,
            "b.aisop.json": valid,
        },
    )

    assert result.context is not None
    assert result.context["workflow_nodes"] == ["bounded-valid"]
    assert result.complete is False
    assert _limitation(result, "structured_document_bytes").path == "a.aisop.json"


def test_total_candidate_input_bytes_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "MAX_STRUCTURED_TOTAL_INPUT_BYTES", 5)
    raw = {"a.aisop.json": b"{}", "b.aisop.json": b"null"}

    result = module.extract_structured_skill_context_from_cache(
        tmp_path,
        raw,
        raw_file_cache=raw,
    )

    assert result.context is None
    limitation = _limitation(result, "structured_total_input_bytes")
    assert limitation.reason_code == "total_bytes_limit"
    assert limitation.observed_bytes == 6
    assert limitation.limit_bytes == 5


def test_candidate_count_overflow_does_not_choose_arbitrary_subset(
    tmp_path: Path,
) -> None:
    raw = {
        f"candidate-{index:03d}.aisop.json": _bundle(f"node-{index}")
        for index in range(module.MAX_STRUCTURED_CANDIDATES + 1)
    }

    result = module.extract_structured_skill_context_from_cache(
        tmp_path,
        raw,
        raw_file_cache=raw,
    )

    assert result.context is None
    assert result.candidates_examined == 0
    limitation = _limitation(result, "structured_candidates")
    assert limitation.reason_code == "artifact_count_limit"
    assert limitation.observed_artifacts == module.MAX_STRUCTURED_CANDIDATES + 1
    assert limitation.limit_artifacts == module.MAX_STRUCTURED_CANDIDATES


def test_deep_json_reports_nesting_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "MAX_STRUCTURED_NESTING", 8)
    functions: dict[str, object] = {"leaf": {}}
    for index in range(12):
        functions = {f"node-{index}": {"functions": functions}}

    result = module.extract_structured_skill_context_from_cache(
        tmp_path,
        ["deep.aisop.json"],
        raw_file_cache={"deep.aisop.json": _bundle("unused", functions=functions)},
    )

    assert result.context is None
    limitation = _limitation(result, "structured_nesting")
    assert limitation.reason_code == "traversal_depth_limit"
    assert limitation.observed_depth == 9
    assert limitation.limit_depth == 8


def test_json_node_work_is_bounded(tmp_path: Path) -> None:
    payload = json.loads(_bundle("node-limit"))
    payload[0]["content"]["padding"] = list(range(module.MAX_STRUCTURED_NODES + 1))

    result = module.extract_structured_skill_context_from_cache(
        tmp_path,
        ["nodes.aisop.json"],
        raw_file_cache={"nodes.aisop.json": json.dumps(payload).encode()},
    )

    assert result.context is None
    limitation = _limitation(result, "structured_nodes")
    assert limitation.reason_code == "artifact_count_limit"
    assert limitation.observed_artifacts is not None
    assert limitation.observed_artifacts > module.MAX_STRUCTURED_NODES
    assert limitation.limit_artifacts == module.MAX_STRUCTURED_NODES


def test_oversized_numeric_scalar_is_partial_instead_of_crashing(tmp_path: Path) -> None:
    payload = b'{"value": ' + b"9" * 10_000 + b"}"

    result = module.extract_structured_skill_context_from_cache(
        tmp_path,
        ["numeric.aisop.json"],
        raw_file_cache={"numeric.aisop.json": payload},
    )

    assert result.context is None
    limitation = _limitation(result, "structured_scalar_conversion")
    assert limitation.reason_code == "output_limit"
    assert limitation.observed_records == 1
    assert limitation.limit_records == 0


def test_structured_output_records_are_bounded(
    tmp_path: Path,
) -> None:
    functions = {
        f"node-{index:03d}": {} for index in range(module.MAX_STRUCTURED_OUTPUT_RECORDS + 1)
    }

    result = module.extract_structured_skill_context_from_cache(
        tmp_path,
        ["outputs.aisop.json"],
        raw_file_cache={"outputs.aisop.json": _bundle("unused", functions=functions)},
    )

    assert result.context is None
    limitation = _limitation(result, "structured_output_records")
    assert limitation.reason_code == "output_limit"
    assert limitation.path == "outputs.aisop.json"
    assert limitation.observed_records == module.MAX_STRUCTURED_OUTPUT_RECORDS + 1
    assert limitation.limit_records == module.MAX_STRUCTURED_OUTPUT_RECORDS


def test_runtime_limit_is_reported_after_parser_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "MAX_STRUCTURED_RUNTIME_SECONDS", 1.0)
    timestamps = iter((0.0, 0.0, 0.0, 2.0))

    result = module.extract_structured_skill_context_from_cache(
        tmp_path,
        ["runtime.aisop.json"],
        raw_file_cache={"runtime.aisop.json": _bundle("runtime")},
        clock=lambda: next(timestamps),
    )

    assert result.context is None
    limitation = _limitation(result, "structured_runtime")
    assert limitation.reason_code == "runtime_limit"
    assert limitation.observed_seconds == 2.0
    assert limitation.limit_seconds == 1.0


def test_cache_api_honors_tighter_caller_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundle-wide caller deadline takes precedence over the local ceiling."""
    monkeypatch.setattr(module, "MAX_STRUCTURED_RUNTIME_SECONDS", 10.0)
    timestamps = iter((10.0, 10.1, 10.2, 10.6))

    result = module.extract_structured_skill_context_from_cache(
        tmp_path,
        ["runtime.aisop.json"],
        raw_file_cache={"runtime.aisop.json": _bundle("runtime")},
        clock=lambda: next(timestamps),
        deadline=10.5,
    )

    assert result.context is None
    limitation = _limitation(result, "structured_runtime")
    assert limitation.observed_seconds == pytest.approx(0.6)
    assert limitation.limit_seconds == pytest.approx(0.5)


def test_compatibility_wrapper_is_bounded_and_avoids_rglob_read_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "workflow.aisop.json"
    path.write_bytes(_bundle("compatibility"))

    def unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("legacy unbounded filesystem API was used")

    monkeypatch.setattr(Path, "rglob", unexpected)
    monkeypatch.setattr(Path, "read_text", unexpected)

    context = module.extract_structured_skill_context(tmp_path)

    assert context is not None
    assert context["workflow_nodes"] == ["compatibility"]
    assert context["bundle_path"] == str(path)


def test_compatibility_wrapper_fails_closed_on_directory_entry_overflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "MAX_STRUCTURED_DIRECTORY_ENTRIES", 2)
    (tmp_path / "workflow.aisop.json").write_bytes(_bundle("ignored"))
    (tmp_path / "one.txt").write_text("one", encoding="utf-8")
    (tmp_path / "two.txt").write_text("two", encoding="utf-8")

    assert module.extract_structured_skill_context(tmp_path) is None
