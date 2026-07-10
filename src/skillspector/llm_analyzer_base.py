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
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, field_validator

from skillspector.llm_utils import get_chat_model
from skillspector.logging_config import get_logger
from skillspector.model_info import get_max_input_tokens
from skillspector.models import Finding

logger = get_logger(__name__)

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
        v = float(v)
        if v > 2.0:
            v = v / 100.0
        return min(1.0, max(0.0, v))

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

    def __init__(self, base_prompt: str, model: str):
        self.base_prompt = base_prompt
        self.model = model
        self._input_budget = get_max_input_tokens(model)
        self._llm = get_chat_model(model=model)
        self._structured_llm = (
            self._llm.with_structured_output(self.response_schema) if self.response_schema else None
        )

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
        results: list[tuple[Batch, list]] = []
        for batch in batches:
            prompt = self.build_prompt(batch, **kwargs)
            logger.debug(
                "LLM call for %s (tokens~%d, findings=%d)",
                batch.file_label,
                estimate_tokens(prompt),
                len(batch.findings),
            )
            if self._structured_llm:
                response = self._structured_llm.invoke(prompt)
            else:
                response = _message_text(self._llm.invoke(prompt))
            logger.debug("LLM response for %s", batch.file_label)
            parsed = self.parse_response(response, batch)
            results.append((batch, parsed))
        return results

    async def arun_batches(
        self,
        batches: list[Batch],
        *,
        max_concurrency: int = 10,
        **kwargs: object,
    ) -> list[tuple[Batch, list]]:
        """Execute LLM calls for all *batches* concurrently.

        Uses ``asyncio.gather`` with a semaphore to run up to
        *max_concurrency* LLM requests in parallel.  Both cross-file and
        cross-chunk batches are parallelized in a single gather call.

        Failures are isolated per batch: a transient error (timeout, 429,
        oversized-chunk 400, ...) costs only its own batch, which is logged
        and omitted from the result, so one bad call cannot cancel the rest
        of the fan-out.  Callers can detect partial results by comparing the
        returned batches against the submitted ones.  ``ValueError`` and
        ``NotImplementedError`` signal misconfiguration rather than infra
        trouble and keep propagating.

        The return type mirrors :meth:`run_batches`.
        """
        sem = asyncio.Semaphore(max_concurrency)

        async def _process(batch: Batch) -> tuple[Batch, list]:
            async with sem:
                prompt = self.build_prompt(batch, **kwargs)
                logger.debug(
                    "LLM call for %s (tokens~%d, findings=%d)",
                    batch.file_label,
                    estimate_tokens(prompt),
                    len(batch.findings),
                )
                if self._structured_llm:
                    response = await self._structured_llm.ainvoke(prompt)
                else:
                    response = _message_text(await self._llm.ainvoke(prompt))
                logger.debug("LLM response for %s", batch.file_label)
                return (batch, self.parse_response(response, batch))

        results = await asyncio.gather(*[_process(b) for b in batches], return_exceptions=True)
        successful: list[tuple[Batch, list]] = []
        for batch, result in zip(batches, results, strict=True):
            if isinstance(result, (ValueError, NotImplementedError)):
                raise result
            if isinstance(result, BaseException):
                logger.warning("LLM batch failed for %s: %s", batch.file_label, result)
                continue
            successful.append(result)
        return successful

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
