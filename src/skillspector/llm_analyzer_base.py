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
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, field_validator

from skillspector.cache import get_cached_findings, initialize_cache_db, set_cached_findings
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

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        # Clamp into [0.0, 1.0] so a slightly out-of-range model value
        # normalises instead of failing the structured-output parse.
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


def is_relevant_for_llm(path: str) -> bool:
    """Check if the file is a static asset or configuration file that should be skipped for LLM scanning."""
    ext = Path(path).suffix.lower()
    if ext in (
        ".css", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".icns",
        ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3", ".wav", ".zip", ".tar", ".gz"
    ):
        return False
    filename = Path(path).name.lower()
    if ext == ".json" and filename not in ("skill.json", "manifest.json"):
        if filename in ("tsconfig.json", "package-lock.json", "tauri.conf.json", "jsconfig.json", "package.json"):
            return False
    if ext in (".yaml", ".yml"):
        if filename in ("pnpm-lock.yaml", "pnpm-lock.yml"):
            return False
    return True


def find_split_index(lines: list[str], start: int, end: int) -> int:
    """Find a natural splitting boundary (blank line, bracket, or code keyword) in the second half of the chunk."""
    safe_start = start + (end - start) // 2
    if safe_start >= end - 1:
        return end

    block_keywords = ("def ", "class ", "fn ", "function ", "pub fn ", "impl ", "struct ", "pub struct ")
    for idx in range(end - 1, safe_start, -1):
        line = lines[idx].strip()
        if any(line.startswith(kw) for kw in block_keywords):
            return idx
    for idx in range(end - 1, safe_start, -1):
        if not lines[idx].strip():
            return idx
    for idx in range(end - 1, safe_start, -1):
        line = lines[idx].strip()
        if line == "}" or line == "};" or line.startswith("}"):
            return idx + 1
    return end


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
                best_split = find_split_index(lines, start_idx, end_idx)
                end_idx = best_split
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
            if not is_relevant_for_llm(path):
                logger.debug("Skipping static file for LLM scan: %s", path)
                continue
            content = file_cache.get(path)
            if content is None:
                content = "No content available for this file."
            elif not content.strip():
                continue
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

    def _parse_raw_json(self, text: str) -> object:
        """Robustly extract and parse JSON from a raw model completion string."""
        import json
        import re

        # Quick check for common text indicator of empty findings
        if not text.strip() or any(phrase in text.lower() for phrase in ("no findings", "clean", "no vulnerabilities", "none", "[]")):
            if self.response_schema:
                return self.response_schema()

        logger.warning("RAW RESPONSE FROM LLM: %s", text[:1000])

        # 1. Clean the string (remove thinking tags if present)
        cleaned = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.DOTALL).strip()

        # 2. Extract JSON from markdown code block if present
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            json_str = cleaned

        # 3. Fallback: find the first '{' and last '}' if not starting with { or [
        if not (json_str.startswith("{") or json_str.startswith("[")):
            first_brace = json_str.find("{")
            last_brace = json_str.rfind("}")
            if first_brace != -1 and last_brace != -1:
                json_str = json_str[first_brace:last_brace + 1]

        # 4. Parse and validate
        if self.response_schema:
            try:
                parsed_json = json.loads(json_str)
                return self.response_schema.model_validate(parsed_json)
            except Exception as e:
                logger.warning("Failed parsing fallback raw JSON: %s. Defaulting to empty structured schema.", e)
                return self.response_schema()
        else:
            return json_str

    # -- Run loop -----------------------------------------------------------

    def _parse_reset_seconds(self, err_msg: str) -> float:
        import re
        match = re.search(r"reset after\s+(\d+(?:\.\d+)?)\s*s", err_msg, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 45.0

    def _call_llm_with_retry(self, fn, *args, **kwargs):
        import time
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                exc_str = str(exc)
                if "403" in exc_str or "rate limit" in exc_str.lower() or "429" in exc_str:
                    sleep_secs = self._parse_reset_seconds(exc_str) + 2.0
                    logger.warning(
                        "Rate limit or 403 hit. Sleeping for %.2f seconds before retry (attempt %d/%d). Error: %s",
                        sleep_secs, attempt + 1, max_retries, exc_str
                    )
                    time.sleep(sleep_secs)
                else:
                    raise exc
        raise RuntimeError(f"Failed after {max_retries} retries due to rate limits.")

    async def _acall_llm_with_retry(self, fn, *args, **kwargs):
        import asyncio
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                exc_str = str(exc)
                if "403" in exc_str or "rate limit" in exc_str.lower() or "429" in exc_str:
                    sleep_secs = self._parse_reset_seconds(exc_str) + 2.0
                    logger.warning(
                        "Rate limit or 403 hit. Sleeping for %.2f seconds before retry (attempt %d/%d). Error: %s",
                        sleep_secs, attempt + 1, max_retries, exc_str
                    )
                    await asyncio.sleep(sleep_secs)
                else:
                    raise exc
        raise RuntimeError(f"Failed after {max_retries} retries due to rate limits.")

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
        initialize_cache_db()
        results: list[tuple[Batch, list]] = []
        for batch in batches:
            # Check cache
            hasher = hashlib.sha256()
            hasher.update(self.base_prompt.encode("utf-8"))
            hasher.update(batch.content.encode("utf-8"))
            hasher.update(self.model.encode("utf-8"))
            if batch.findings:
                sorted_findings = sorted(batch.findings, key=lambda f: (f.file or "", f.rule_id or "", f.start_line or 0, f.message or ""))
                findings_str = str([dataclasses.asdict(f) for f in sorted_findings])
                hasher.update(findings_str.encode("utf-8"))
            ckey = hasher.hexdigest()

            cached_str = get_cached_findings(ckey)
            if cached_str is not None:
                try:
                    data = json.loads(cached_str)
                    if data and isinstance(data[0], dict) and "_file" in data[0]:
                        parsed = data
                    else:
                        parsed = [Finding(**d) for d in data]
                    results.append((batch, parsed))
                    continue
                except Exception as e:
                    logger.debug("Failed to deserialize cache for %s: %s", batch.file_path, e)

            prompt = self.build_prompt(batch, **kwargs)
            logger.debug(
                "LLM call for %s (tokens~%d, findings=%d)",
                batch.file_label,
                estimate_tokens(prompt),
                len(batch.findings),
            )
            if self._structured_llm:
                try:
                    response = self._call_llm_with_retry(self._structured_llm.invoke, prompt)
                except Exception as exc:
                    if "403" in str(exc) or "rate limit" in str(exc).lower() or "429" in str(exc):
                        raise exc
                    logger.warning("Structured output invocation failed: %s. Falling back to raw completion + manual parsing.", exc)
                    raw_response_msg = self._call_llm_with_retry(self._llm.invoke, prompt)
                    raw_response = _message_text(raw_response_msg)
                    response = self._parse_raw_json(raw_response)
            else:
                raw_response_msg = self._call_llm_with_retry(self._llm.invoke, prompt)
                response = _message_text(raw_response_msg)
            logger.debug("LLM response for %s", batch.file_label)
            parsed = self.parse_response(response, batch)

            # Store cache
            try:
                if parsed and isinstance(parsed[0], Finding):
                    serialized = json.dumps([dataclasses.asdict(f) for f in parsed])
                else:
                    serialized = json.dumps(parsed)
                set_cached_findings(
                    ckey,
                    serialized,
                    self.__class__.__name__,
                    hashlib.sha256(batch.content.encode("utf-8")).hexdigest(),
                    self.model
                )
            except Exception as e:
                logger.debug("Failed to cache findings for %s: %s", batch.file_path, e)

            results.append((batch, parsed))
        return results

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

        The return type mirrors :meth:`run_batches`.
        """
        import os
        if max_concurrency is None:
            env_concurrency = os.environ.get("SKILLSPECTOR_CONCURRENCY")
            if env_concurrency:
                try:
                    max_concurrency = int(env_concurrency)
                except ValueError:
                    max_concurrency = 5
            else:
                max_concurrency = 5

        initialize_cache_db()

        uncached_batches: list[Batch] = []
        cached_results: list[tuple[Batch, list]] = []
        cache_keys: dict[int, str] = {}

        for batch in batches:
            hasher = hashlib.sha256()
            hasher.update(self.base_prompt.encode("utf-8"))
            hasher.update(batch.content.encode("utf-8"))
            hasher.update(self.model.encode("utf-8"))
            if batch.findings:
                sorted_findings = sorted(batch.findings, key=lambda f: (f.file or "", f.rule_id or "", f.start_line or 0, f.message or ""))
                findings_str = str([dataclasses.asdict(f) for f in sorted_findings])
                hasher.update(findings_str.encode("utf-8"))
            ckey = hasher.hexdigest()
            cache_keys[id(batch)] = ckey

            cached_str = get_cached_findings(ckey)
            if cached_str is not None:
                try:
                    data = json.loads(cached_str)
                    if data and isinstance(data[0], dict) and "_file" in data[0]:
                        parsed = data
                    else:
                        parsed = [Finding(**d) for d in data]
                    cached_results.append((batch, parsed))
                    continue
                except Exception as e:
                    logger.debug("Failed to deserialize cache for %s: %s", batch.file_path, e)

            uncached_batches.append(batch)

        if uncached_batches:
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
                        try:
                            response = await self._acall_llm_with_retry(self._structured_llm.ainvoke, prompt)
                        except Exception as exc:
                            if "403" in str(exc) or "rate limit" in str(exc).lower() or "429" in str(exc):
                                raise exc
                            logger.warning("Structured output ainvoke failed: %s. Falling back to raw completion + manual parsing.", exc)
                            raw_response_msg = await self._acall_llm_with_retry(self._llm.ainvoke, prompt)
                            raw_response = _message_text(raw_response_msg)
                            response = self._parse_raw_json(raw_response)
                    else:
                        raw_response_msg = await self._acall_llm_with_retry(self._llm.ainvoke, prompt)
                        response = _message_text(raw_response_msg)
                    logger.debug("LLM response for %s", batch.file_label)
                    parsed = self.parse_response(response, batch)

                    # Store cache
                    ckey = cache_keys[id(batch)]
                    try:
                        if parsed and isinstance(parsed[0], Finding):
                            serialized = json.dumps([dataclasses.asdict(f) for f in parsed])
                        else:
                            serialized = json.dumps(parsed)
                        set_cached_findings(
                            ckey,
                            serialized,
                            self.__class__.__name__,
                            hashlib.sha256(batch.content.encode("utf-8")).hexdigest(),
                            self.model
                        )
                    except Exception as e:
                        logger.debug("Failed to cache findings for %s: %s", batch.file_path, e)

                    return (batch, parsed)

            uncached_results = list(await asyncio.gather(*[_process(b) for b in uncached_batches]))
        else:
            uncached_results = []

        return cached_results + uncached_results

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


LLMAnalyzerBase._original_run_batches = LLMAnalyzerBase.run_batches

