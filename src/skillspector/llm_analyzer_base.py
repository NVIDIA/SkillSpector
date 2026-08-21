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

"""Base LLM Analyzer with per-file / per-chunk batching (truncation-safe).

Provides ``LLMAnalyzerBase`` — a reusable run-loop that splits work into one
LLM call per file (or per chunk when a file exceeds the model's input budget),
using token budgets from ``constants.py`` so no single prompt is truncated.

The default ``response_schema`` is :class:`LLMAnalysisResult` (a list of
:class:`LLMFinding`), suitable for discovery-mode analyzers.  Subclasses may
override :attr:`response_schema` with a different Pydantic model, or set it
to ``None`` for raw-string mode.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from openai import APIStatusError
from pydantic import BaseModel, Field, ValidationError, field_validator

from skillspector.inspection_ledger import (
    AnalyzerStatusEvent,
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    analyzer_status_for_events,
    ledger_event,
    outcome_for_llm_batch_failure,
)
from skillspector.llm_utils import (
    StructuredOutputParseError,
    _AgentCLIMessage,
    _ainvoke_with_usage,
    _invoke_with_usage,
    get_chat_model,
    new_inference_usage_collector,
)
from skillspector.logging_config import get_logger
from skillspector.model_info import get_max_input_tokens
from skillspector.models import Finding

logger = get_logger(__name__)

DEFAULT_MAX_LLM_CONCURRENCY = 10
API_CONNECTION_MAX_RETRIES = 3
API_CONNECTION_RETRY_DELAYS_SECONDS = (0.5, 1.0, 2.0)
STRUCTURED_RESPONSE_MAX_RETRIES = 3
STRUCTURED_RESPONSE_MAX_ATTEMPTS = STRUCTURED_RESPONSE_MAX_RETRIES + 1
STRUCTURED_RESPONSE_RETRY_DELAYS_SECONDS = API_CONNECTION_RETRY_DELAYS_SECONDS
RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_RETRY_DELAYS_SECONDS = (5.0, 15.0, 30.0, 60.0, 60.0)
RATE_LIMIT_MAX_DELAY_SECONDS = 120.0
HTTP_TOO_MANY_REQUESTS = 429
LLM_BATCH_MAX_ATTEMPTS = (
    STRUCTURED_RESPONSE_MAX_ATTEMPTS + API_CONNECTION_MAX_RETRIES + RATE_LIMIT_MAX_RETRIES
)


class _StructuredResponseValidationError(Exception):
    """Signal that provider output failed structured-response validation."""


def _is_retryable_api_connection_error(exc: BaseException) -> bool:
    """Return whether *exc* is the narrowly supported transient provider failure."""
    return type(exc).__name__ == "APIConnectionError"


def _is_retryable_rate_limit_error(exc: BaseException) -> bool:
    """Return whether *exc* is a provider rate-limit (HTTP 429) rejection.

    Provider SDKs name this exception ``RateLimitError`` (OpenAI, Anthropic and
    the OpenAI-compatible endpoints reached through ``ChatOpenAI``); raw HTTP
    clients surface it as a 429 status error instead.
    """
    if type(exc).__name__ == "RateLimitError":
        return True
    if isinstance(exc, (APIStatusError, httpx.HTTPStatusError)):
        return exc.response.status_code == HTTP_TOO_MANY_REQUESTS
    return False


def _rate_limit_retry_after_seconds(exc: BaseException) -> float | None:
    """Return the provider-requested ``Retry-After`` delay in seconds, if usable."""
    if not isinstance(exc, (APIStatusError, httpx.HTTPStatusError)):
        return None
    header = exc.response.headers.get("retry-after")
    if header is None:
        return None
    try:
        seconds = float(header)
    except ValueError:
        # Retry-After may also be an HTTP-date; the schedule below covers it.
        return None
    if seconds <= 0:
        return None
    return min(seconds, RATE_LIMIT_MAX_DELAY_SECONDS)


def _rate_limit_retry_delay(exc: BaseException, retries_used: int) -> float:
    """Return the wait before the next attempt after a rate-limit rejection."""
    scheduled = RATE_LIMIT_RETRY_DELAYS_SECONDS[retries_used]
    requested = _rate_limit_retry_after_seconds(exc)
    return max(scheduled, requested) if requested is not None else scheduled


def _resolve_retry_delay(
    exc: BaseException,
    *,
    attempt: int,
    connection_retries: int,
    rate_limit_retries: int,
    uses_native_connection_retries: bool,
) -> tuple[Literal["rate_limit", "connection"], float] | None:
    """Return the retry kind and delay for *exc*, or ``None`` to fail the batch.

    Rate-limit rejections are retried even when the chat model has native retry
    support: native budgets back off in milliseconds, which a provider quota
    ceiling outlasts.
    """
    if attempt == LLM_BATCH_MAX_ATTEMPTS:
        return None
    if _is_retryable_rate_limit_error(exc):
        if rate_limit_retries >= len(RATE_LIMIT_RETRY_DELAYS_SECONDS):
            return None
        return "rate_limit", _rate_limit_retry_delay(exc, rate_limit_retries)
    if (
        not _is_retryable_api_connection_error(exc)
        or uses_native_connection_retries
        or connection_retries >= len(API_CONNECTION_RETRY_DELAYS_SECONDS)
    ):
        return None
    return "connection", API_CONNECTION_RETRY_DELAYS_SECONDS[connection_retries]


def _reason_for_batch_exception(exc: BaseException) -> LedgerReason:
    """Return the ledger reason recorded for an unrecovered batch failure."""
    if _is_retryable_rate_limit_error(exc):
        return LedgerReason.LLM_RATE_LIMIT_RETRIES_EXHAUSTED
    if _is_retryable_api_connection_error(exc):
        return LedgerReason.LLM_CONNECTION_RETRIES_EXHAUSTED
    return LedgerReason.LLM_BATCH_FAILED


def _uses_native_connection_retries(chat_model: object) -> bool:
    """Set the common native retry budget and report whether it is available."""
    if isinstance(chat_model, ChatOpenAI):
        for client in (chat_model.root_client, chat_model.root_async_client):
            if client is not None:
                client.max_retries = API_CONNECTION_MAX_RETRIES
        return True
    if isinstance(chat_model, ChatAnthropic):
        chat_model.max_retries = API_CONNECTION_MAX_RETRIES
        return True
    return False


def resolve_max_concurrency() -> int:
    """Resolve the LLM fan-out concurrency from ``SKILLSPECTOR_MAX_LLM_CONCURRENCY``.

    Defaults to :data:`DEFAULT_MAX_LLM_CONCURRENCY`. Users on rate-limited
    providers (free tiers with a low RPM) can set it to ``1`` to serialize
    requests instead of bursting up to 10 in parallel — a burst that otherwise
    guarantees 429s. Rate-limited batches are retried with the bounded
    schedule in the analyzer fan-out below, so serializing trades wall time
    for fewer retries rather than deciding whether a batch is analyzed at all.
    Invalid values fall back to the default; values below 1 are clamped to 1.
    """
    raw = os.environ.get("SKILLSPECTOR_MAX_LLM_CONCURRENCY", "").strip()
    if not raw:
        return DEFAULT_MAX_LLM_CONCURRENCY
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid SKILLSPECTOR_MAX_LLM_CONCURRENCY=%r (not an int); using %d",
            raw,
            DEFAULT_MAX_LLM_CONCURRENCY,
        )
        return DEFAULT_MAX_LLM_CONCURRENCY
    if value < 1:
        logger.warning("SKILLSPECTOR_MAX_LLM_CONCURRENCY=%d < 1; clamping to 1", value)
        return 1
    return value


# OpenAI suggests ~4 chars per token for English text with BPE tokenizers.
CHARS_PER_TOKEN = 4
CHUNK_OVERLAP_LINES = 50


# ---------------------------------------------------------------------------
# Default structured-output schemas (discovery mode)
# ---------------------------------------------------------------------------


class LLMFinding(BaseModel):
    """A single finding discovered by an LLM analyzer.

    Field names intentionally mirror :class:`~skillspector.models.Finding` so
    that :meth:`to_finding` can produce a graph-state ``Finding`` directly.
    """

    rule_id: str = Field(description="Identifier for the type of finding")
    message: str = Field(description="Short description of the finding")
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(description="Severity level")
    # start_line and confidence carry no ge/le Field bounds on purpose. Pydantic
    # bounds emit JSON-schema minimum/maximum, which some OpenAI-compatible
    # structured-output / tool-calling endpoints reject when they validate the
    # response schema, failing the whole call. The ranges are enforced by the
    # validators below instead, so the guarantee holds without those keywords in
    # the emitted schema. start_line stays required (no default), so a finding
    # with no location is still rejected rather than materialised at line 1;
    # only the numeric bound is removed, not the requiredness.
    start_line: int = Field(description="Starting line number (>= 1)")
    end_line: int | None = Field(default=None, description="Ending line number (optional)")
    confidence: float = Field(default=0.5, description="Confidence score between 0.0 and 1.0")
    explanation: str = Field(default="", description="Why this is a finding (2-3 sentences)")
    remediation: str = Field(default="", description="Actionable steps to fix the issue")

    @field_validator("start_line")
    @classmethod
    def _clamp_start_line(cls, v: int) -> int:
        # Clamp rather than raise: an LLM occasionally returns 0 for a
        # whole-file finding, and normalising to the first line is better than
        # dropping the finding over an off-by-one.
        return v if v >= 1 else 1

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v: object) -> float:
        # Accept 0-100 scale values from some models, then clamp into [0, 1].
        value = float(cast(Any, v))
        if value > 2.0:
            value = value / 100.0
        return min(1.0, max(0.0, value))

    def to_finding(self, file: str) -> Finding:
        """Convert to a :class:`Finding` for the graph state."""
        return Finding(
            rule_id=self.rule_id,
            message=self.message,
            severity=self.severity,
            confidence=self.confidence,
            file=file,
            start_line=self.start_line,
            end_line=self.end_line,
            explanation=self.explanation,
            remediation=self.remediation,
        )


class LLMAnalysisResult(BaseModel):
    """Structured LLM response containing discovered findings."""

    findings: list[LLMFinding] = Field(default_factory=list)


def estimate_tokens(text: str) -> int:
    """Approximate token count from character length."""
    return len(text) // CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Batch dataclass
# ---------------------------------------------------------------------------


@dataclass
class Batch:
    """One unit of work for an LLM call (single file or file-chunk)."""

    file_path: str
    content: str
    start_line: int = 1
    end_line: int | None = None
    findings: list[Finding] = field(default_factory=list)

    @property
    def is_chunk(self) -> bool:
        return self.end_line is not None

    @property
    def file_label(self) -> str:
        label = f"File: {self.file_path}"
        if self.is_chunk:
            label += f" (lines {self.start_line}\u2013{self.end_line})"
        return label


@dataclass(frozen=True)
class BatchFailure:
    """Sanitized failure outcome for one submitted LLM batch."""

    batch: Batch
    error_class: str
    reason: LedgerReason = LedgerReason.LLM_BATCH_FAILED


@dataclass
class BatchExecutionResult:
    """Detailed LLM batch outcome while preserving successful parsed values."""

    successful: list[tuple[Batch, list]] = field(default_factory=list)
    failures: list[BatchFailure] = field(default_factory=list)


def _batch_interval(batch: Batch) -> tuple[int | None, int | None]:
    """Return the canonical ledger range for a submitted batch."""
    if batch.end_line is not None:
        return batch.start_line, batch.end_line
    return None, None


def _uncovered_intervals(
    failed_interval: tuple[int | None, int | None],
    successful_intervals: list[tuple[int | None, int | None]],
) -> list[tuple[int | None, int | None]]:
    """Subtract successful chunk coverage from one failed batch interval."""
    failed_start, failed_end = failed_interval
    if failed_start is None:
        return [] if failed_interval in successful_intervals else [failed_interval]

    assert failed_end is not None
    covered_intervals: list[tuple[int, int]] = []
    for start_line, end_line in successful_intervals:
        if start_line is not None and end_line is not None:
            covered_intervals.append((start_line, end_line))

    uncovered: list[tuple[int | None, int | None]] = []
    next_uncovered_line = failed_start
    for covered_start, covered_end in sorted(covered_intervals):
        if covered_end < next_uncovered_line:
            continue
        if covered_start > failed_end:
            break
        if covered_start > next_uncovered_line:
            uncovered.append((next_uncovered_line, min(failed_end, covered_start - 1)))
        next_uncovered_line = max(next_uncovered_line, covered_end + 1)
        if next_uncovered_line > failed_end:
            break
    if next_uncovered_line <= failed_end:
        uncovered.append((next_uncovered_line, failed_end))
    return uncovered


def ledger_events_for_batches(
    analyzer_id: str,
    outcome: BatchExecutionResult,
) -> tuple[list[InspectionLedgerEvent], AnalyzerStatusEvent]:
    """Project detailed LLM batch execution into terminal ledger evidence."""
    events: list[InspectionLedgerEvent] = []
    successful_ranges: dict[str, list[tuple[int | None, int | None]]] = defaultdict(list)
    for batch, findings in outcome.successful:
        if not isinstance(batch, Batch):
            logger.debug("Skipping ledger projection for malformed successful batch: %r", batch)
            continue
        start_line, end_line = _batch_interval(batch)
        successful_ranges[batch.file_path].append((start_line, end_line))
        events.append(
            ledger_event(
                analyzer_id=analyzer_id,
                outcome=LedgerOutcome.COMPLETED,
                phase="semantic",
                path=batch.file_path,
                start_line=start_line,
                end_line=end_line,
                emitted_finding_ids=[finding.finding_id for finding in findings],
            )
        )

    failed_ranges: dict[str, list[tuple[BatchFailure, tuple[int | None, int | None]]]] = (
        defaultdict(list)
    )
    for failure in outcome.failures:
        if not isinstance(failure.batch, Batch):
            logger.debug("Skipping ledger projection for malformed failed batch: %r", failure.batch)
            continue
        failed_ranges[failure.batch.file_path].append((failure, _batch_interval(failure.batch)))

    for path, failures in failed_ranges.items():
        for failure, failed_range in failures:
            for start_line, end_line in _uncovered_intervals(failed_range, successful_ranges[path]):
                events.append(
                    ledger_event(
                        analyzer_id=analyzer_id,
                        outcome=outcome_for_llm_batch_failure(failure.reason),
                        phase="semantic",
                        path=path,
                        start_line=start_line,
                        end_line=end_line,
                        reason=failure.reason,
                        error_class=failure.error_class,
                    )
                )

    return events, analyzer_status_for_events(analyzer_id, events)


# ---------------------------------------------------------------------------
# Chunking utilities
# ---------------------------------------------------------------------------


def chunk_file_by_lines(
    content: str,
    max_tokens: int,
    overlap_lines: int = CHUNK_OVERLAP_LINES,
) -> list[tuple[str, int, int]]:
    """Split *content* into line-range chunks that each fit within *max_tokens*.

    Returns a list of ``(chunk_text, start_line, end_line)`` tuples where lines
    are 1-indexed.  Consecutive chunks share *overlap_lines* lines of context so
    findings near chunk boundaries still have surrounding code.
    """
    lines = content.splitlines(keepends=True)
    if not lines:
        return [("", 1, 1)]

    chunks: list[tuple[str, int, int]] = []
    start_idx = 0

    while start_idx < len(lines):
        token_count = 0
        end_idx = start_idx

        while end_idx < len(lines):
            line_tokens = estimate_tokens(lines[end_idx])
            if token_count + line_tokens > max_tokens and end_idx > start_idx:
                break
            token_count += line_tokens
            end_idx += 1

        chunk_text = "".join(lines[start_idx:end_idx])
        chunks.append((chunk_text, start_idx + 1, end_idx))  # 1-indexed

        if end_idx >= len(lines):
            break

        next_start = end_idx - overlap_lines
        if next_start <= start_idx:
            next_start = end_idx
        start_idx = next_start

    return chunks


def findings_in_range(
    findings: list[Finding],
    start_line: int,
    end_line: int,
) -> list[Finding]:
    """Return findings whose ``start_line`` falls within [start_line, end_line]."""
    return [f for f in findings if start_line <= f.start_line <= end_line]


def number_lines(content: str, start_line: int = 1) -> str:
    """Prefix each line with its 1-indexed line number (e.g. ``L1:``, ``L2:``).

    For chunks, *start_line* offsets the numbering so the LLM sees real file
    line numbers it can reference in :attr:`LLMFinding.start_line`.
    """
    lines = content.splitlines()
    if not lines:
        return ""
    end = start_line + len(lines) - 1
    width = len(str(end))
    return "\n".join(f"L{start_line + i:0>{width}}: {line}" for i, line in enumerate(lines))


def _message_text(response: object) -> str:
    """Extract provider-normalized text from a LangChain chat response."""
    if not isinstance(response, BaseMessage):
        raise TypeError(f"Expected BaseMessage from chat model, got {type(response).__name__}")
    return str(response.text)


def _raw_response_text(response: object) -> str:
    """Extract raw analyzer text from LangChain and CLI adapter messages."""
    if isinstance(response, _AgentCLIMessage):
        return str(response.content)
    return _message_text(response)


BASE_ANALYSIS_PROMPT = """\
{analyzer_prompt}

Analyze the following skill file for security issues matching the criteria above.
Reference line numbers (shown as L-prefixes) when reporting findings.

## {file_label}
```
{numbered_content}
```

## Output guidelines

- Most files are clean — an empty findings list is expected and correct when
  no genuine issues exist.  Do not manufacture findings to fill the response.
- Precision over recall: only report issues you are confident about.  It is
  far better to miss an edge case than to report a false positive.
- Be precise: report only genuine issues, not speculative ones."""


# ---------------------------------------------------------------------------
# Base LLM Analyzer
# ---------------------------------------------------------------------------


class LLMAnalyzerBase:
    """Per-file / per-chunk LLM analyzer.

    Subclass, supply an ``analyzer_prompt`` string, and optionally override
    :meth:`build_prompt` / :meth:`parse_response`.  The defaults produce a
    prompt with line-numbered file content and parse :class:`LLMAnalysisResult`
    (a list of :class:`LLMFinding`).

    Override :attr:`response_schema` with a different Pydantic model for
    custom structured output, or set it to ``None`` for raw-string mode.

    **Precision-over-recall default**: ``BASE_ANALYSIS_PROMPT`` appends
    output guidelines that instruct the LLM to prefer empty findings over
    false positives.  This applies to all analyzers that use the default
    :meth:`build_prompt`.  Subclasses that override :meth:`build_prompt`
    (e.g. the meta-analyzer) control their own output instructions.
    """

    response_schema: type | None = LLMAnalysisResult

    def __init__(
        self,
        base_prompt: str,
        model: str,
        *,
        node: str = "llm_analyzer",
        timeout: float | None | Callable[[], float | None] = None,
    ):
        self.base_prompt = base_prompt
        self.model = model
        self._timeout = timeout
        self._dynamic_timeout = callable(timeout)
        self._input_budget = get_max_input_tokens(model)
        self._llm = get_chat_model(model=model, timeout=self._remaining_timeout())
        self._uses_native_connection_retries = _uses_native_connection_retries(self._llm)
        self._structured_llm = (
            self._llm.with_structured_output(self.response_schema) if self.response_schema else None
        )
        self._usage_collector = new_inference_usage_collector(
            node=node,
            request_kind="structured_output" if self.response_schema else "chat_completion",
            model=model,
            chat_model=self._llm,
        )

    def _remaining_timeout(self) -> float | None:
        if callable(self._timeout):
            return self._timeout()
        return self._timeout

    def _model_for_call(self) -> tuple[object, object | None]:
        if not self._dynamic_timeout:
            return self._llm, self._structured_llm
        llm = get_chat_model(model=self.model, timeout=self._remaining_timeout())
        structured = (
            llm.with_structured_output(self.response_schema) if self.response_schema else None
        )
        return llm, structured

    @property
    def inference_usage(self) -> list[dict[str, object]]:
        """Provider-reported usage captured for this analyzer instance."""
        return list(self._usage_collector.snapshot())

    @property
    def response_received(self) -> bool:
        """Whether any analyzer call received a provider response."""
        return self._usage_collector.response_received

    # -- Batching -----------------------------------------------------------

    def _estimate_extra_overhead(self, findings: list[Finding]) -> int:
        """Token overhead beyond the base prompt (e.g. formatted findings).

        Override in subclasses that add findings text to the prompt.
        """
        return 0

    def get_batches(
        self,
        file_paths: list[str],
        file_cache: dict[str, str],
        findings: list[Finding] | None = None,
    ) -> list[Batch]:
        """Create one :class:`Batch` per file, splitting oversized files into chunks."""
        base_overhead = estimate_tokens(self.base_prompt)

        findings_by_file: dict[str, list[Finding]] = defaultdict(list)
        if findings:
            for f in findings:
                findings_by_file[f.file].append(f)

        batches: list[Batch] = []
        for path in file_paths:
            content = file_cache.get(path) or "No content available for this file."
            file_findings = findings_by_file.get(path, [])

            extra = self._estimate_extra_overhead(file_findings)
            content_budget = max(self._input_budget - base_overhead - extra, 1024)

            content_tokens = estimate_tokens(content)
            if content_tokens <= content_budget:
                batches.append(
                    Batch(
                        file_path=path,
                        content=content,
                        findings=file_findings,
                    )
                )
            else:
                chunk_budget = max(int(content_budget), 1024)
                for chunk_text, s_line, e_line in chunk_file_by_lines(content, chunk_budget):
                    chunk_findings = findings_in_range(file_findings, s_line, e_line)
                    batches.append(
                        Batch(
                            file_path=path,
                            content=chunk_text,
                            start_line=s_line,
                            end_line=e_line,
                            findings=chunk_findings,
                        )
                    )

        return batches

    # -- Prompt / parse -----------------------------------------------------

    def build_prompt(self, batch: Batch, **kwargs: object) -> str:
        """Build the LLM prompt for a single batch.

        The default wraps :attr:`base_prompt` with line-numbered file content
        so the LLM can reference exact line numbers in its findings.
        Override in subclasses that need a custom prompt layout.
        """
        numbered = number_lines(batch.content, batch.start_line)
        return BASE_ANALYSIS_PROMPT.format(
            analyzer_prompt=self.base_prompt,
            file_label=batch.file_label,
            numbered_content=numbered,
        )

    def parse_response(self, response: object, batch: Batch) -> list[Finding]:
        """Parse the LLM response for a single batch.

        The default converts each :class:`LLMFinding` to a :class:`Finding`
        via :meth:`LLMFinding.to_finding`.  Override in subclasses that use a
        different ``response_schema`` or raw-string mode.
        """
        if isinstance(response, LLMAnalysisResult):
            return [f.to_finding(batch.file_path) for f in response.findings]
        raise NotImplementedError(
            "Override parse_response for custom response_schema or raw-string mode"
        )

    # -- Run loop -----------------------------------------------------------

    def _invoke_batch(self, batch: Batch, prompt: str) -> tuple[Batch, list]:
        """Invoke and parse one batch synchronously."""
        logger.debug(
            "LLM call for %s (tokens~%d, findings=%d)",
            batch.file_label,
            estimate_tokens(prompt),
            len(batch.findings),
        )
        llm, structured_llm = self._model_for_call()
        if structured_llm:
            try:
                response = _invoke_with_usage(structured_llm, prompt, self._usage_collector)
            except (StructuredOutputParseError, ValidationError) as exc:
                raise _StructuredResponseValidationError from exc
        else:
            response = _raw_response_text(_invoke_with_usage(llm, prompt, self._usage_collector))
        logger.debug("LLM response for %s", batch.file_label)
        return batch, self.parse_response(response, batch)

    def _invoke_batch_with_retries(self, batch: Batch, prompt: str) -> tuple[Batch, list]:
        """Run one batch with bounded retries for malformed output, 429s and connection failures."""
        structured_retries = 0
        connection_retries = 0
        rate_limit_retries = 0
        for attempt in range(1, LLM_BATCH_MAX_ATTEMPTS + 1):
            try:
                return self._invoke_batch(batch, prompt)
            except _StructuredResponseValidationError:
                if (
                    structured_retries >= STRUCTURED_RESPONSE_MAX_ATTEMPTS - 1
                    or attempt == LLM_BATCH_MAX_ATTEMPTS
                ):
                    raise
                delay = STRUCTURED_RESPONSE_RETRY_DELAYS_SECONDS[structured_retries]
                structured_retries += 1
                logger.warning(
                    "LLM structured response validation failed for %s; retrying in %.2fs (%d/%d)",
                    batch.file_label,
                    delay,
                    structured_retries,
                    STRUCTURED_RESPONSE_MAX_RETRIES,
                )
                time.sleep(delay)
            except Exception as exc:
                decision = _resolve_retry_delay(
                    exc,
                    attempt=attempt,
                    connection_retries=connection_retries,
                    rate_limit_retries=rate_limit_retries,
                    uses_native_connection_retries=self._uses_native_connection_retries,
                )
                if decision is None:
                    raise
                kind, delay = decision
                if kind == "rate_limit":
                    rate_limit_retries += 1
                    logger.warning(
                        "LLM provider rate limited %s; retrying in %.2fs (%d/%d)",
                        batch.file_label,
                        delay,
                        rate_limit_retries,
                        RATE_LIMIT_MAX_RETRIES,
                    )
                else:
                    connection_retries += 1
                    logger.warning(
                        "LLM connection failed for %s; retrying in %.2fs (%d/%d)",
                        batch.file_label,
                        delay,
                        connection_retries,
                        API_CONNECTION_MAX_RETRIES,
                    )
                time.sleep(delay)

        raise AssertionError("bounded retry loop must return or raise")

    async def _ainvoke_batch(self, batch: Batch, prompt: str) -> tuple[Batch, list]:
        """Invoke and parse one batch asynchronously."""
        logger.debug(
            "LLM call for %s (tokens~%d, findings=%d)",
            batch.file_label,
            estimate_tokens(prompt),
            len(batch.findings),
        )
        llm, structured_llm = self._model_for_call()
        if structured_llm:
            try:
                response = await _ainvoke_with_usage(structured_llm, prompt, self._usage_collector)
            except (StructuredOutputParseError, ValidationError) as exc:
                raise _StructuredResponseValidationError from exc
        else:
            response = _raw_response_text(
                await _ainvoke_with_usage(llm, prompt, self._usage_collector)
            )
        logger.debug("LLM response for %s", batch.file_label)
        return batch, self.parse_response(response, batch)

    async def _ainvoke_batch_with_retries(self, batch: Batch, prompt: str) -> tuple[Batch, list]:
        """Asynchronously run one batch with bounded malformed-output, 429 and connection retries."""
        structured_retries = 0
        connection_retries = 0
        rate_limit_retries = 0
        for attempt in range(1, LLM_BATCH_MAX_ATTEMPTS + 1):
            try:
                return await self._ainvoke_batch(batch, prompt)
            except _StructuredResponseValidationError:
                if (
                    structured_retries >= STRUCTURED_RESPONSE_MAX_ATTEMPTS - 1
                    or attempt == LLM_BATCH_MAX_ATTEMPTS
                ):
                    raise
                delay = STRUCTURED_RESPONSE_RETRY_DELAYS_SECONDS[structured_retries]
                structured_retries += 1
                logger.warning(
                    "LLM structured response validation failed for %s; retrying in %.2fs (%d/%d)",
                    batch.file_label,
                    delay,
                    structured_retries,
                    STRUCTURED_RESPONSE_MAX_RETRIES,
                )
                await asyncio.sleep(delay)
            except Exception as exc:
                decision = _resolve_retry_delay(
                    exc,
                    attempt=attempt,
                    connection_retries=connection_retries,
                    rate_limit_retries=rate_limit_retries,
                    uses_native_connection_retries=self._uses_native_connection_retries,
                )
                if decision is None:
                    raise
                kind, delay = decision
                if kind == "rate_limit":
                    rate_limit_retries += 1
                    logger.warning(
                        "LLM provider rate limited %s; retrying in %.2fs (%d/%d)",
                        batch.file_label,
                        delay,
                        rate_limit_retries,
                        RATE_LIMIT_MAX_RETRIES,
                    )
                else:
                    connection_retries += 1
                    logger.warning(
                        "LLM connection failed for %s; retrying in %.2fs (%d/%d)",
                        batch.file_label,
                        delay,
                        connection_retries,
                        API_CONNECTION_MAX_RETRIES,
                    )
                await asyncio.sleep(delay)

        raise AssertionError("bounded retry loop must return or raise")

    def run_batches(
        self,
        batches: list[Batch],
        **kwargs: object,
    ) -> list[tuple[Batch, list]]:
        """Execute LLM calls for all *batches*, returning per-batch parsed results.

        The element type of the inner list depends on the subclass: the default
        :meth:`parse_response` returns :class:`Finding` objects; subclasses may
        return dicts or other types.
        """
        outcome = self.run_batches_detailed(batches, **kwargs)
        self._last_batch_outcome = outcome
        return outcome.successful

    def run_batches_detailed(
        self,
        batches: list[Batch],
        **kwargs: object,
    ) -> BatchExecutionResult:
        """Execute batches and retain each sanitized failure alongside successes."""
        outcome = BatchExecutionResult()
        for batch in batches:
            try:
                prompt = self.build_prompt(batch, **kwargs)
                result = self._invoke_batch_with_retries(batch, prompt)
                outcome.successful.append(result)
            except _StructuredResponseValidationError:
                logger.warning(
                    "LLM structured response validation failed for %s after %d attempts",
                    batch.file_label,
                    STRUCTURED_RESPONSE_MAX_ATTEMPTS,
                )
                outcome.failures.append(
                    BatchFailure(
                        batch=batch,
                        error_class=ValidationError.__name__,
                        reason=LedgerReason.LLM_STRUCTURED_RESPONSE_INVALID,
                    )
                )
            except (ValueError, NotImplementedError):
                raise
            except Exception as exc:
                logger.warning("LLM batch failed for %s: %s", batch.file_label, exc)
                outcome.failures.append(
                    BatchFailure(
                        batch=batch,
                        error_class=type(exc).__name__,
                        reason=_reason_for_batch_exception(exc),
                    )
                )
        return outcome

    async def arun_batches(
        self,
        batches: list[Batch],
        *,
        max_concurrency: int | None = None,
        **kwargs: object,
    ) -> list[tuple[Batch, list]]:
        """Execute LLM calls for all *batches* concurrently.

        Uses ``asyncio.gather`` with a semaphore to run up to
        *max_concurrency* LLM requests in parallel.  Both cross-file and
        cross-chunk batches are parallelized in a single gather call.

        When *max_concurrency* is ``None`` (the default) it is resolved from
        ``SKILLSPECTOR_MAX_LLM_CONCURRENCY`` via :func:`resolve_max_concurrency`,
        so users on rate-limited providers can serialize the fan-out; an
        explicit argument still wins.

        Failures are isolated per batch: a provider ``APIConnectionError``
        receives three bounded exponential-backoff retries (500ms, then 1s,
        then 2s) when the chat model has no native retry support. OpenAI and
        Anthropic chat models use their native three-retry policy instead;
        native retry timing remains provider-managed. Unrecovered errors cost
        only their own batch and are omitted from the result.
        Rate-limit rejections (HTTP 429) receive five bounded retries (5s, 15s,
        30s, then 60s twice, or the provider's ``Retry-After`` when it asks for
        longer, capped at 120s) regardless of native retry support, because a
        provider quota ceiling outlasts the sub-second native budget.
        Malformed structured responses (Pydantic ``ValidationError`` or CLI
        JSON parse failures) receive three bounded exponential-backoff retries
        and are then isolated to their batch. A batch makes at most twelve
        outer chat-model invocations even when all retry policies apply; native
        provider retries can make additional HTTP requests within one
        invocation.
        Callers can detect partial results by comparing the returned batches
        against the submitted ones.  Other ``ValueError`` instances and
        ``NotImplementedError`` signal misconfiguration rather than infra trouble
        and keep propagating.

        The return type mirrors :meth:`run_batches`.
        """
        outcome = await self.arun_batches_detailed(
            batches, max_concurrency=max_concurrency, **kwargs
        )
        self._last_batch_outcome = outcome
        return outcome.successful

    async def arun_batches_detailed(
        self,
        batches: list[Batch],
        *,
        max_concurrency: int | None = None,
        **kwargs: object,
    ) -> BatchExecutionResult:
        """Execute batches concurrently and retain sanitized per-batch failures."""
        if max_concurrency is None:
            max_concurrency = resolve_max_concurrency()
        sem = asyncio.Semaphore(max_concurrency)

        async def _process(batch: Batch) -> tuple[Batch, list]:
            async with sem:
                prompt = self.build_prompt(batch, **kwargs)
                return await self._ainvoke_batch_with_retries(batch, prompt)

        results = await asyncio.gather(*[_process(b) for b in batches], return_exceptions=True)
        outcome = BatchExecutionResult()
        for batch, result in zip(batches, results, strict=True):
            if isinstance(result, _StructuredResponseValidationError):
                logger.warning(
                    "LLM structured response validation failed for %s after %d attempts",
                    batch.file_label,
                    STRUCTURED_RESPONSE_MAX_ATTEMPTS,
                )
                outcome.failures.append(
                    BatchFailure(
                        batch=batch,
                        error_class=ValidationError.__name__,
                        reason=LedgerReason.LLM_STRUCTURED_RESPONSE_INVALID,
                    )
                )
                continue
            if isinstance(result, (ValueError, NotImplementedError)):
                raise result
            if isinstance(result, BaseException):
                logger.warning("LLM batch failed for %s: %s", batch.file_label, result)
                outcome.failures.append(
                    BatchFailure(
                        batch=batch,
                        error_class=type(result).__name__,
                        reason=_reason_for_batch_exception(result),
                    )
                )
                continue
            outcome.successful.append(result)
        return outcome

    # -- Convenience --------------------------------------------------------

    def collect_findings(
        self,
        batch_results: list[tuple[Batch, list]],
    ) -> list[Finding]:
        """Flatten per-batch results into a single :class:`Finding` list.

        Intended for discovery-mode analyzers where :meth:`parse_response`
        returns :class:`Finding` objects.  A typical node can do::

            batches = analyzer.get_batches(files, file_cache)
            results = analyzer.run_batches(batches)
            return {"findings": analyzer.collect_findings(results)}
        """
        return [f for _, items in batch_results for f in items]
