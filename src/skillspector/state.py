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

"""State schema for the Skillspector LangGraph workflow."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from time import monotonic
from typing import Annotated, NotRequired

from typing_extensions import TypedDict

from skillspector.artifacts import ArtifactRecord, BundleReference
from skillspector.inference_usage import InferenceUsageRecord
from skillspector.inspection_ledger import (
    MAX_INSPECTION_LEDGER_EVENTS,
    AnalysisCompleteness,
    AnalyzerStatusEvent,
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    LedgerRecordType,
    ledger_event,
)
from skillspector.models import Finding

MAX_WORKFLOW_SECONDS = 60.0
MAX_WORKFLOW_BYTES = 64 * 1024 * 1024
MAX_WORKFLOW_ARTIFACTS = 10_000
MAX_WORKFLOW_LIMITATION_RECORDS = 256


@dataclass(slots=True)
class WorkflowResourceBudget:
    """One resource budget shared by every node in a graph invocation.

    The normal graph entry points do not create the CLI's transitive traversal
    object.  Keeping this smaller contract in graph state gives direct, API,
    and MCP scans the same aggregate deadline and byte/artifact ceilings.  A
    supplied transitive traversal remains authoritative because it may carry a
    stricter allowance shared by several child graph invocations.
    """

    max_seconds: float = MAX_WORKFLOW_SECONDS
    max_bytes: int = MAX_WORKFLOW_BYTES
    max_artifacts: int = MAX_WORKFLOW_ARTIFACTS
    started_at: float | None = None
    scanned_bytes: int = 0
    scanned_artifacts: int = 0
    truncation_reasons: list[str] = field(default_factory=list)
    budget_exhausted: bool = False

    def start(self) -> None:
        """Start the aggregate deadline exactly once."""
        if self.started_at is None:
            self.started_at = monotonic()

    def remaining_seconds(self) -> float:
        """Return the non-negative aggregate workflow time allowance."""
        self.start()
        assert self.started_at is not None
        return max(0.0, self.max_seconds - (monotonic() - self.started_at))

    def remaining_bytes(self) -> int:
        """Return the exact remaining canonical-byte allowance."""
        return max(0, self.max_bytes - self.scanned_bytes)

    def remaining_artifacts(self) -> int:
        """Return the remaining discovery/nested-artifact allowance."""
        return max(0, self.max_artifacts - self.scanned_artifacts)

    def record_bytes(self, count: int) -> None:
        """Charge retained canonical bytes to the workflow allowance."""
        self.start()
        self.scanned_bytes += max(0, count)
        if self.scanned_bytes > self.max_bytes:
            self.note_truncation(f"byte budget {self.max_bytes} exceeded")

    def record_artifacts(self, count: int) -> None:
        """Charge discovered or expanded artifacts to the workflow allowance."""
        self.start()
        self.scanned_artifacts += max(0, count)
        if self.scanned_artifacts > self.max_artifacts:
            self.note_truncation(f"artifact budget {self.max_artifacts} exceeded")

    def note_truncation(self, reason: str) -> None:
        """Retain a bounded, content-free explanation of resource exhaustion."""
        if len(self.truncation_reasons) >= MAX_WORKFLOW_LIMITATION_RECORDS:
            sentinel = "additional workflow limitations omitted"
            if self.truncation_reasons[-1] != sentinel:
                self.truncation_reasons[-1] = sentinel
            self.budget_exhausted = True
            return
        if reason not in self.truncation_reasons:
            self.truncation_reasons.append(reason)
        if "budget" in reason:
            self.budget_exhausted = True


def ensure_workflow_resource_budget(state: SkillspectorState) -> object:
    """Return and start the strictest resource budget supplied to a graph scan."""
    transitive = state.get("transitive_traversal_state")
    existing = state.get("workflow_resource_budget")
    budget = transitive if _has_resource_budget_contract(transitive) else existing
    if not _has_resource_budget_contract(budget):
        budget = WorkflowResourceBudget()

    start = getattr(budget, "start", None)
    if callable(start):
        start()
    else:
        # The CLI traversal starts lazily through remaining_seconds().  Invoke
        # it before build-context work so child scans cannot restart the clock.
        remaining = getattr(budget, "remaining_seconds", None)
        if callable(remaining):
            remaining()
    return budget


def _has_resource_budget_contract(candidate: object | None) -> bool:
    """Recognize a complete time/byte/artifact workflow-budget contract."""
    return candidate is not None and all(
        callable(getattr(candidate, method, None))
        for method in ("remaining_seconds", "remaining_bytes", "remaining_artifacts")
    )


def merge_findings_by_id(existing: list[Finding], updates: list[Finding]) -> list[Finding]:
    """Merge findings by opaque ID, replacing enriched instances in place."""
    merged = list(existing)
    positions = {finding.finding_id: index for index, finding in enumerate(merged)}
    for finding in updates:
        position = positions.get(finding.finding_id)
        if position is None:
            positions[finding.finding_id] = len(merged)
            merged.append(finding)
        else:
            merged[position] = finding
    return merged


def merge_inspection_ledger(
    existing: list[InspectionLedgerEvent],
    updates: list[InspectionLedgerEvent],
) -> list[InspectionLedgerEvent]:
    """Concatenate ledger rows under one workflow-wide deterministic ceiling."""
    limit = max(1, MAX_INSPECTION_LEDGER_EVENTS)
    if existing and existing[-1].get("phase") == "ledger_output":
        prior = existing[-1]
        try:
            prior_observed = int(prior.get("observed_records", len(existing)))
        except (TypeError, ValueError):
            prior_observed = len(existing)
        observed = max(len(existing), prior_observed) + len(updates)
        return [
            *existing[:-1][: limit - 1],
            ledger_event(
                outcome=LedgerOutcome.PARTIAL,
                record_type=LedgerRecordType.SYSTEM,
                phase="ledger_output",
                path=str(prior.get("path", "SKILL.md")),
                reason=LedgerReason.OUTPUT_LIMIT,
                observed_records=observed,
                limit_records=limit,
            ),
        ]

    combined = [*existing, *updates]
    if len(combined) <= limit:
        return combined
    overflow = combined[limit - 1]
    return [
        *combined[: limit - 1],
        ledger_event(
            outcome=LedgerOutcome.PARTIAL,
            record_type=LedgerRecordType.SYSTEM,
            phase="ledger_output",
            path=str(overflow.get("path", "SKILL.md")),
            reason=LedgerReason.OUTPUT_LIMIT,
            observed_records=len(combined),
            limit_records=limit,
        ),
    ]


class SkillspectorState(TypedDict, total=False):
    """Graph state shared by all nodes."""

    # Input: resolve_input node consumes input_path or skill_path, sets skill_path
    input_path: str | None
    skill_path: str | None
    # Set by resolve_input when a temp dir was created (git/url/zip/file); caller should clean up
    temp_dir_for_cleanup: str | None
    zip_bytes: bytes | None
    mode: str

    # build_context node populates these
    components: list[str]
    # Visible authored text that may be submitted to external LLM providers.
    llm_components: list[str]
    file_cache: dict[str, str]
    # Full local-only deterministic view, including hidden and nested content.
    local_file_cache: dict[str, str]
    # Raw bytes remain the canonical source for YARA and content classification.
    raw_file_cache: dict[str, bytes]
    # External-model consumers use the redacted projection for sensitive local files.
    llm_file_cache: dict[str, str]
    # Visible artifacts omitted because bounded provider redaction did not complete.
    llm_redaction_incomplete_paths: list[str]
    artifact_inventory: list[ArtifactRecord]
    artifact_references: list[BundleReference]
    reference_resolution: dict[str, object]
    # Retained for compatibility with the persisted workflow-state schema.
    ast_cache: dict[str, str]
    # Key for the process-local parsed-AST cache.  The ASTs themselves stay
    # outside state because they are not checkpoint-serializable.
    python_ast_cache_key: str | None
    manifest: dict[str, object]
    previous_manifest: dict[str, object] | None

    # Accumulated canonical findings. Same-ID meta updates replace in place.
    findings: Annotated[list[Finding], merge_findings_by_id]
    inspection_ledger: Annotated[list[InspectionLedgerEvent], merge_inspection_ledger]
    analyzer_status_events: Annotated[list[AnalyzerStatusEvent], operator.add]
    effective_finding_ids: list[str]
    analysis_completeness: AnalysisCompleteness
    execution_successful: bool

    # Compatibility projection emitted only by the report after effective-ID
    # selection. Meta analysis never stores a second filtered collection.
    filtered_findings: list[Finding]

    # LLM runtime telemetry: each LLM-backed node appends one record (built with
    # ``llm_call_record``) so the report can detect a *silent degradation* — the
    # case where use_llm was requested but every LLM call failed at runtime
    # (transport/parse/auth error). Without this, such a failure would quietly
    # turn a requested deep scan into a static-only one while still reporting
    # llm_available=true. Reducer is operator.add so records concatenate across
    # the parallel analyzer nodes (same pattern as ``findings``).
    llm_call_log: Annotated[list[LLMCallRecord], operator.add]

    # Exact provider-response token counters. Each LLM-backed node appends its
    # per-call records; the report exposes the sanitized projection under
    # metadata.inference_usage. Missing records mean "not observable", never
    # an estimated zero.
    inference_usage: Annotated[list[InferenceUsageRecord], operator.add]

    # Baseline / false-positive suppression. `baseline` is a loaded
    # skillspector.suppression.Baseline (set by CLI/API); the report node drops
    # matching findings before scoring. `show_suppressed` keeps them in the
    # report (marked) for review; `suppressed_findings` is the report output.
    baseline: object | None
    # Absolute path selected by `scan --baseline` or targeted by `baseline -o`.
    # When it is inside the scan target, build_context excludes only that file
    # so waiver text cannot scan itself or enter regenerated fingerprints.
    baseline_path: str | None
    show_suppressed: bool
    suppressed_findings: list[object]

    # Model IDs per LLM-using node: e.g. {"default": "...", "meta_analyzer": "..."}
    model_config: dict[str, str]

    # Component metadata for reporting and risk scoring (from build_context)
    component_metadata: list[dict[str, object]]
    has_executable_scripts: bool
    # Structured workflow context for phase-1 AISOP/AISP summaries
    structured_skill_context: dict[str, object]
    # Report-only structured skill summaries emitted outside the finding pipeline
    structured_summaries: Annotated[list[dict[str, object]], operator.add]

    # Output: report node writes formatted string here
    output_format: str
    report_body: str

    # LLM: when False, LLM-based nodes (meta_analyzer, mcp_tool_poisoning's TP4,
    # and the semantic_* analyzers) return immediately without calling the LLM.
    # Each such node checks use_llm itself; there is no graph-level routing.
    use_llm: bool

    # Risk: report node sets these from risk_score
    risk_severity: str
    risk_recommendation: str

    sarif_report: dict[str, object]
    risk_score: int

    # Transitive traversal metadata for report output and CLI summaries.
    transitive_finding_count: int
    transitive_sources: list[str]
    transitive_targets_scanned: int
    transitive_bytes_scanned: int
    transitive_truncated: bool
    transitive_truncation_reasons: list[str]
    transitive_traversal_state: object
    # Present for every graph invocation. Transitive child scans point this at
    # their already-shared traversal object instead of starting a second clock.
    workflow_resource_budget: object

    # Additional YARA rules directory (user-specified via --yara-rules-dir)
    yara_rules_dir: str | None


class LLMCallRecord(TypedDict):
    """One LLM-stage telemetry record (an entry in ``llm_call_log``)."""

    node: str
    ok: bool
    error: str | None


def llm_call_record(node_id: str, *, ok: bool, error: str | None = None) -> LLMCallRecord:
    """Build one telemetry record for ``SkillspectorState['llm_call_log']``.

    LLM-backed nodes append a record on each run so the report can tell whether
    the LLM stage actually produced results. ``ok=False`` marks a runtime
    failure where the node fell back to empty/static findings (so the failure is
    not mistaken for "the LLM ran and found nothing").
    """
    return {"node": node_id, "ok": ok, "error": error}


def transitive_traversal_state(state: SkillspectorState) -> object | None:
    """Return the active shared workflow resource object, when one is present."""
    traversal = state.get("transitive_traversal_state")
    if traversal is not None:
        return traversal
    budget = state.get("workflow_resource_budget")
    return budget if budget is not None else None


def transitive_remaining_seconds(state: SkillspectorState) -> float | None:
    """Return the remaining transitive deadline in seconds, when available."""
    traversal = transitive_traversal_state(state)
    remaining = getattr(traversal, "remaining_seconds", None)
    if callable(remaining):
        try:
            return float(remaining())
        except (TypeError, ValueError):
            return None
    return None


def transitive_remaining_bytes(state: SkillspectorState) -> int | None:
    """Return the remaining transitive byte allowance, when available."""
    traversal = transitive_traversal_state(state)
    remaining = getattr(traversal, "remaining_bytes", None)
    if callable(remaining):
        try:
            return int(remaining())
        except (TypeError, ValueError):
            return None
    return None


def transitive_remaining_artifacts(state: SkillspectorState) -> int | None:
    """Return the remaining shared transitive artifact allowance, when available."""
    traversal = transitive_traversal_state(state)
    remaining = getattr(traversal, "remaining_artifacts", None)
    if callable(remaining):
        try:
            return int(remaining())
        except (TypeError, ValueError):
            return None
    return None


def transitive_record_artifacts(state: SkillspectorState, count: int) -> None:
    """Charge discovered or expanded artifacts to the shared traversal budget."""
    traversal = transitive_traversal_state(state)
    record = getattr(traversal, "record_artifacts", None)
    if callable(record):
        record(max(0, count))


def transitive_note_truncation(state: SkillspectorState, reason: str) -> None:
    """Record a transitive truncation reason on the shared traversal object."""
    traversal = transitive_traversal_state(state)
    note = getattr(traversal, "note_truncation", None)
    if callable(note):
        note(reason)


class AnalyzerNodeResponse(TypedDict):
    """Strict analyzer update payload for graph state."""

    findings: list[Finding]
    inspection_ledger: NotRequired[list[InspectionLedgerEvent]]
    analyzer_status_events: NotRequired[list[AnalyzerStatusEvent]]
    structured_summaries: NotRequired[list[dict[str, object]]]
    # LLM-backed analyzers also report one telemetry record; static analyzers
    # omit it (NotRequired keeps the key optional for them).
    llm_call_log: NotRequired[list[LLMCallRecord]]
    inference_usage: NotRequired[list[InferenceUsageRecord]]


class MetaAnalyzerResponse(TypedDict):
    """Meta-analyzer payload with canonical findings and ID selection."""

    findings: NotRequired[list[Finding]]
    effective_finding_ids: NotRequired[list[str]]
    inspection_ledger: NotRequired[list[InspectionLedgerEvent]]
    analyzer_status_events: NotRequired[list[AnalyzerStatusEvent]]
    llm_call_log: NotRequired[list[LLMCallRecord]]
    inference_usage: NotRequired[list[InferenceUsageRecord]]
