# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configured TP4 capacity changes throughput without hiding unfinished work."""

from __future__ import annotations

import asyncio
import importlib
import re
from collections import deque
from collections.abc import Callable, Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.llm_analyzer_base import LLMAnalyzerBase
from skillspector.nodes.analyzers import mcp_tool_poisoning as tp4

_ENVIRONMENT_KEY = "SKILLSPECTOR_TP4_MAX_BATCHES"
_CLEAN = {"is_mismatch": False}


class _StructuredModel:
    def __init__(self, responses: list[object]) -> None:
        self.responses = deque(responses)
        self.prompts: list[str] = []
        self.schema: type[BaseModel] | None = None

    def invoke_with_usage(self, prompt: str, collector: object) -> object:
        self.prompts.append(prompt)
        response = self.responses.popleft()
        if isinstance(response, BaseException):
            raise response
        collector.mark_response_received()  # type: ignore[attr-defined]
        assert self.schema is not None
        return self.schema.model_validate(response)

    async def ainvoke_with_usage(self, prompt: str, collector: object) -> object:
        return self.invoke_with_usage(prompt, collector)


class _ChatModel:
    def __init__(self, structured: _StructuredModel) -> None:
        self.structured = structured

    def with_structured_output(self, schema: type[BaseModel]) -> _StructuredModel:
        self.structured.schema = schema
        return self.structured


@pytest.fixture
def configure_capacity() -> Iterator[Callable[[str | None], None]]:
    """Exercise environment resolution at import, then restore the caller's module."""
    with pytest.MonkeyPatch.context() as environment:
        # Keep queue-based failure fixtures deterministic; the concurrency case below
        # uses keyed responses and explicitly exercises the shared async limiter.
        environment.setenv("SKILLSPECTOR_MAX_LLM_CONCURRENCY", "1")

        def configure(value: str | None) -> None:
            if value is None:
                environment.delenv(_ENVIRONMENT_KEY, raising=False)
            else:
                environment.setenv(_ENVIRONMENT_KEY, value)
            importlib.reload(tp4)

        yield configure
    importlib.reload(tp4)


def _model(monkeypatch: pytest.MonkeyPatch, responses: list[object]) -> _StructuredModel:
    structured = _StructuredModel(responses)
    monkeypatch.setattr(
        "skillspector.llm_analyzer_base.get_chat_model",
        lambda **_kwargs: _ChatModel(structured),
    )
    monkeypatch.setattr(LLMAnalyzerBase, "_asleep_before_retry", AsyncMock())
    return structured


def _fences(count: int) -> str:
    return "".join(f"```python\nprint('fence-{index:04d}')\n```\n" for index in range(count))


def _state(files: dict[str, str]) -> dict:
    return {
        "manifest": {"name": "capacity-fixture", "description": "Runs local calculations."},
        "file_cache": files,
        "component_metadata": [
            {"path": path, "type": "markdown" if path.endswith(".md") else "python"}
            for path in files
        ],
        "use_llm": True,
        "model_config": {"default": "test-model"},
    }


def _high(index: int) -> dict:
    return {
        "is_mismatch": True,
        "confidence": 0.9,
        "mismatched_capabilities": [f"undeclared network access {index}"],
    }


def _completed_ranges(result: object) -> list[tuple[str, int, int]]:
    return [
        (event["path"], event["start_line"], event["end_line"])
        for event in result.ledger  # type: ignore[attr-defined]
        if event["outcome"] is LedgerOutcome.COMPLETED
    ]


@pytest.mark.parametrize("value", [None, "", " \t"])
def test_default_stays_at_128_and_accounts_for_uninspected_fence(
    monkeypatch, configure_capacity, caplog, value
):
    configure_capacity(value)
    structured = _model(monkeypatch, [_CLEAN] * 128)

    result = tp4._check_tp4(_state({"guide.md": _fences(129)}))

    assert len(structured.prompts) == 128
    assert len(_completed_ranges(result)) == 128
    incomplete = [event for event in result.ledger if event["outcome"] is LedgerOutcome.PARTIAL]
    assert len(incomplete) == 1
    assert incomplete[0]["reason_code"] is LedgerReason.OUTPUT_LIMIT
    assert incomplete[0]["observed_records"] == 129
    assert incomplete[0]["limit_records"] == 128
    assert not any(record.levelname == "WARNING" for record in caplog.records)


@pytest.mark.parametrize(
    ("value", "fence_count"),
    [("512", 245), ("512", 300), ("512", 512), (" 0512 ", 129), ("4096", 129)],
)
def test_configured_capacity_inspects_late_code_and_preserves_high_finding_ranges(
    monkeypatch, configure_capacity, value, fence_count
):
    configure_capacity(value)
    structured = _model(monkeypatch, [_CLEAN] * (fence_count - 1) + [_high(fence_count - 1)])

    result = tp4._check_tp4(_state({"guide.md": _fences(fence_count)}))

    assert len(structured.prompts) == fence_count
    for index, prompt in enumerate(structured.prompts):
        assert f"print('fence-{index:04d}')" in prompt
        assert prompt.count("### guide.md (python)") == 1
        assert prompt.count("print('fence-") == 1
    assert _completed_ranges(result) == [
        ("guide.md", index * 3 + 2, index * 3 + 2) for index in range(fence_count)
    ]
    assert all(event["outcome"] is LedgerOutcome.COMPLETED for event in result.ledger)
    assert result.record["ok"] is True
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.rule_id == "TP4" and finding.severity == "HIGH"
    assert finding.evidence["code_path"] == "guide.md"
    assert finding.evidence["code_start_line"] == (fence_count - 1) * 3 + 2
    assert finding.evidence["code_end_line"] == (fence_count - 1) * 3 + 2
    assert result.ledger[-1]["emitted_finding_ids"] == [finding.finding_id]


@pytest.mark.parametrize("capacity", [1, 512])
def test_configured_capacity_plus_one_remains_incomplete(monkeypatch, configure_capacity, capacity):
    configure_capacity(str(capacity))
    structured = _model(monkeypatch, [_CLEAN] * capacity)

    result = tp4._check_tp4(_state({"guide.md": _fences(capacity + 1)}))

    assert len(structured.prompts) == capacity
    assert len(_completed_ranges(result)) == capacity
    incomplete = [event for event in result.ledger if event["outcome"] is LedgerOutcome.PARTIAL]
    assert len(incomplete) == 1
    assert incomplete[0]["path"] == "guide.md"
    assert incomplete[0]["reason_code"] is LedgerReason.OUTPUT_LIMIT
    assert incomplete[0]["observed_records"] == capacity + 1
    assert incomplete[0]["limit_records"] == capacity


@pytest.mark.parametrize("concurrency", [1, 2])
def test_configured_capacity_keeps_async_limit_and_late_finding_attribution(
    monkeypatch, configure_capacity, concurrency
):
    configure_capacity("512")
    monkeypatch.setenv("SKILLSPECTOR_MAX_LLM_CONCURRENCY", str(concurrency))

    class KeyedModel(_StructuredModel):
        def __init__(self) -> None:
            super().__init__([])
            self.in_flight = 0
            self.peak = 0
            self.completion_order: list[int] = []
            self.pair_events: dict[int, asyncio.Event] = {}

        def invoke_with_usage(self, prompt: str, collector: object) -> object:
            raise AssertionError("TP4 must retain the upstream asynchronous runner")

        async def ainvoke_with_usage(self, prompt: str, collector: object) -> object:
            match = re.search(r"print\('fence-(\d+)'\)", prompt)
            assert match is not None
            index = int(match.group(1))
            self.prompts.append(prompt)
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            try:
                if concurrency == 2:
                    event = self.pair_events.setdefault(index // 2, asyncio.Event())
                    if index % 2 == 0:
                        await asyncio.wait_for(event.wait(), timeout=1.0)
                    else:
                        event.set()
                else:
                    await asyncio.sleep(0)
                self.completion_order.append(index)
                collector.mark_response_received()  # type: ignore[attr-defined]
                assert self.schema is not None
                return self.schema.model_validate(_high(index) if index == 129 else _CLEAN)
            finally:
                self.in_flight -= 1

    structured = KeyedModel()
    monkeypatch.setattr(
        "skillspector.llm_analyzer_base.get_chat_model",
        lambda **_kwargs: _ChatModel(structured),
    )

    result = tp4._check_tp4(_state({"guide.md": _fences(130)}))

    assert structured.peak == concurrency
    assert sorted(structured.completion_order) == list(range(130))
    if concurrency == 2:
        assert structured.completion_order.index(129) < structured.completion_order.index(128)
    else:
        assert structured.completion_order == list(range(130))
    assert _completed_ranges(result) == [
        ("guide.md", index * 3 + 2, index * 3 + 2) for index in range(130)
    ]
    assert all(event["outcome"] is LedgerOutcome.COMPLETED for event in result.ledger)
    assert result.record["ok"] is True
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.rule_id == "TP4" and finding.severity == "HIGH"
    assert finding.evidence["code_path"] == "guide.md"
    assert finding.evidence["code_start_line"] == finding.evidence["code_end_line"] == 389
    assert result.ledger[-1]["emitted_finding_ids"] == [finding.finding_id]


@pytest.mark.parametrize(
    "value",
    ["invalid", "-1", "+512", "0", "1.5", "nan", "inf", "4097", "５１２", "9" * 5000],
    ids=[
        "text",
        "negative",
        "signed",
        "zero",
        "fraction",
        "nan",
        "infinite",
        "over-max",
        "unicode",
        "huge",
    ],
)
def test_invalid_configuration_warns_and_preserves_default_incomplete_accounting(
    monkeypatch, configure_capacity, caplog, value
):
    configure_capacity(value)
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert warnings and any(_ENVIRONMENT_KEY in record.getMessage() for record in warnings)
    if len(value) > 100:
        assert value not in caplog.text
    structured = _model(monkeypatch, [_CLEAN] * 128)

    result = tp4._check_tp4(_state({"guide.md": _fences(129)}))

    assert len(structured.prompts) == 128
    assert len(_completed_ranges(result)) == 128
    incomplete = [event for event in result.ledger if event["outcome"] is LedgerOutcome.PARTIAL]
    assert len(incomplete) == 1
    assert incomplete[0]["reason_code"] is LedgerReason.OUTPUT_LIMIT
    assert incomplete[0]["observed_records"] == 129
    assert incomplete[0]["limit_records"] == 128


@pytest.mark.parametrize("failure", ["provider", "malformed", "deadline"])
def test_late_repeated_failures_never_become_completed_inspection(
    monkeypatch, configure_capacity, failure
):
    configure_capacity("512")
    responses: list[object] = [_CLEAN] * 128
    if failure == "provider":
        responses += [RuntimeError("provider failure"), RuntimeError("provider failure")]
        reason = LedgerReason.LLM_BATCH_FAILED
        expected_calls = 130
    elif failure == "malformed":
        responses += [{}] * 8
        reason = LedgerReason.LLM_STRUCTURED_RESPONSE_INVALID
        expected_calls = 136
    else:
        reason = LedgerReason.RUNTIME_LIMIT
        expected_calls = 128
    structured = _model(monkeypatch, responses)
    if failure == "deadline":
        monkeypatch.setattr(
            tp4,
            "transitive_remaining_seconds",
            lambda _state: 60.0 if len(structured.prompts) < 128 else 0.0,
        )

    result = tp4._check_tp4(_state({"guide.md": _fences(130)}))

    assert len(structured.prompts) == expected_calls
    assert len(_completed_ranges(result)) == 128
    incomplete = [
        event for event in result.ledger if event["outcome"] is not LedgerOutcome.COMPLETED
    ]
    assert len(incomplete) == 2
    assert all(event["reason_code"] is reason for event in incomplete)
    assert [(event["path"], event["start_line"], event["end_line"]) for event in incomplete] == [
        ("guide.md", 386, 386),
        ("guide.md", 389, 389),
    ]
    assert result.record["ok"] is False


def test_configured_capacity_keeps_64_finding_cap(monkeypatch, configure_capacity):
    configure_capacity("512")
    structured = _model(monkeypatch, [_high(index) for index in range(65)])

    result = tp4._check_tp4(_state({"guide.md": _fences(65)}))

    assert len(structured.prompts) == 65
    assert len(result.findings) == 64
    incomplete = [event for event in result.ledger if event["outcome"] is LedgerOutcome.PARTIAL]
    assert len(incomplete) == 1
    assert incomplete[0]["reason_code"] is LedgerReason.OUTPUT_LIMIT
    assert incomplete[0]["limit_findings"] == 64
    assert incomplete[0]["observed_findings"] == 65
    assert incomplete[0]["start_line"] == incomplete[0]["end_line"] == 194


def test_configured_capacity_keeps_128_source_file_cap(monkeypatch, configure_capacity):
    configure_capacity("512")
    structured = _model(monkeypatch, [_CLEAN] * 128)
    files = {f"guide-{index:03d}.md": _fences(1) for index in range(129)}

    result = tp4._check_tp4(_state(files))

    assert len(structured.prompts) == 128
    incomplete = [event for event in result.ledger if event["outcome"] is LedgerOutcome.PARTIAL]
    assert len(incomplete) == 1
    assert incomplete[0]["path"] == "guide-128.md"
    assert incomplete[0]["reason_code"] is LedgerReason.ARTIFACT_COUNT_LIMIT
    assert incomplete[0]["limit_artifacts"] == 128
    assert incomplete[0]["observed_artifacts"] == 129


@pytest.mark.parametrize(
    "bound", ["file-bytes", "total-code-bytes", "single-line-tokens", "model-tokens"]
)
def test_configured_capacity_keeps_other_input_limits_without_provider_calls(
    monkeypatch, configure_capacity, bound
):
    configure_capacity("512")
    if bound == "file-bytes":
        files = {"guide.md": " " * (1024 * 1024 + 1)}
        reason, metric, limit = LedgerReason.SIZE_LIMIT, "limit_bytes", 1024 * 1024
    elif bound == "total-code-bytes":
        files = {f"blank-{index}.py": " " * (1024 * 1024) for index in range(4)}
        files["tail.py"] = "print('late')\n"
        reason, metric, limit = LedgerReason.TOTAL_BYTES_LIMIT, "limit_bytes", 4 * 1024 * 1024
    elif bound == "single-line-tokens":
        files = {"guide.md": "```python\n" + "x" * 128_001 + "\n```\n"}
        reason, metric, limit = LedgerReason.SIZE_LIMIT, "limit_characters", None
    else:
        monkeypatch.setattr(tp4, "get_max_input_tokens", lambda _model: 1)
        files = {"guide.md": _fences(1)}
        reason, metric, limit = LedgerReason.SIZE_LIMIT, "limit_characters", 4
    factory = MagicMock(side_effect=AssertionError("Input limits must prevent a model call"))
    monkeypatch.setattr("skillspector.llm_analyzer_base.get_chat_model", factory)

    result = tp4._check_tp4(_state(files))

    factory.assert_not_called()
    assert result.findings == []
    incomplete = [event for event in result.ledger if event["outcome"] is LedgerOutcome.PARTIAL]
    assert incomplete and all(event["reason_code"] is reason for event in incomplete)
    if limit is None:
        assert 0 < incomplete[0][metric] < 32_000 * 4
        assert incomplete[0]["start_line"] == incomplete[0]["end_line"] == 2
    else:
        assert incomplete[0][metric] == limit


def test_configured_capacity_keeps_aggregate_prompt_byte_limit(monkeypatch, configure_capacity):
    configure_capacity("512")
    fence = "```python\n" + "x" * 9000 + "\n```\n"
    files = {f"guide-{index}.md": fence * 64 for index in range(8)}
    structured = _model(monkeypatch, [_CLEAN] * 512)

    result = tp4._check_tp4(_state(files))

    assert 128 < len(structured.prompts) < 512
    assert sum(len(prompt.encode("utf-8")) for prompt in structured.prompts) <= 4 * 1024 * 1024
    incomplete = [event for event in result.ledger if event["outcome"] is LedgerOutcome.PARTIAL]
    assert incomplete and all(
        event["reason_code"] is LedgerReason.TOTAL_BYTES_LIMIT for event in incomplete
    )
    assert all(event["limit_bytes"] == 4 * 1024 * 1024 for event in incomplete)
    assert any(event["observed_bytes"] > 4 * 1024 * 1024 for event in incomplete)
