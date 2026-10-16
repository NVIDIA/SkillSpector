# SPDX-License-Identifier: Apache-2.0
"""Report-only AISOP/AISP source roles; never part of detection or scoring.

Only the scanner's admitted local bytes and raw-text coordinates are supported.
Transitive sources and transformed views fail closed to an explicit unknown.
The caller passes expanded occurrences AFTER suppression, scoring and compaction,
then sanitizes the returned copies before serialization. Canonical findings and
all caches remain untouched. Suppressed findings are outside this increment.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

from skillspector.models import Finding
from skillspector.structured_source import (
    RoleAnnotation,
    StructuredSourceMap,
    index_structured_source,
)

# Independently bounded auxiliary work. These are report annotation limits, not
# evidence that a detection pass was complete or incomplete.
MAX_ROLE_DOCUMENTS = 64
MAX_ROLE_INPUT_BYTES = 1024 * 1024
MAX_ROLE_RECORDS = 512
MAX_ROLE_OUTPUT_CHARS = 64 * 1024
MAX_ROLE_SECONDS = 2.0
MAX_ROLE_LEDGER_ROWS = 10_000
MAX_ROLE_LEDGER_IDS = 20_000
ROLE_EVIDENCE_KEY = "structured_source_role"
_DERIVED_VIEW_TAGS = frozenset({"normalized-view", "declared-marker-view"})


def _static_ids(events: object) -> set[str]:
    """Use scanner-owned ledger attribution, not an input file's claimed role."""
    if not isinstance(events, list) or len(events) > MAX_ROLE_LEDGER_ROWS:
        return set()
    result: set[str] = set()
    examined = 0
    for event in events:
        if not isinstance(event, Mapping) or event.get("phase") != "static":
            continue
        emitted = event.get("emitted_finding_ids", ())
        if not isinstance(emitted, (list, tuple)):
            continue
        examined += len(emitted)
        if examined > MAX_ROLE_LEDGER_IDS:
            return set()
        result.update(value for value in emitted if isinstance(value, str))
    return result


def _local_component(path: object, admitted: object) -> bool:
    if not isinstance(path, str) or not path.lower().endswith(".aisop.json"):
        return False
    # Never normalize a path into another cache entry or search by basename.
    if "\\" in path or "\x00" in path or path.startswith("/"):
        return False
    if any(part in {"", ".", ".."} for part in path.split("/")):
        return False
    if len(path) > 1 and path[1] == ":":
        return False
    return isinstance(admitted, (list, tuple, set, frozenset)) and path in admitted


class _RoleReport:
    """A single report's bounded, ephemeral cache. Never stored in graph state."""

    def __init__(
        self,
        state: Mapping[str, object],
        *,
        deadline: float | None,
        clock: Callable[[], float],
    ) -> None:
        self.state = state
        self.clock = clock
        stop = clock() + MAX_ROLE_SECONDS
        self.deadline = min(stop, deadline) if deadline is not None else stop
        self.static_ids = _static_ids(state.get("inspection_ledger"))
        self.indices: dict[str, StructuredSourceMap] = {}
        self.input_bytes = 0
        self.records = 0
        self.output_chars = 0
        self.reason: str | None = None
        self.exact = 0
        self.unknown = 0
        components = state.get("components")
        # The normal scanner already bounds these. Do not copy an unbounded
        # compatibility caller's input into a new set.
        self.components = (
            frozenset(item for item in components if isinstance(item, str))
            if isinstance(components, (list, tuple)) and len(components) <= 10_000
            else frozenset()
        )
        raw = state.get("raw_file_cache")
        text = state.get("local_file_cache") or state.get("file_cache")
        self.raw = raw if isinstance(raw, Mapping) else {}
        self.text = text if isinstance(text, Mapping) else {}

    def annotation(self, finding: Finding) -> RoleAnnotation:
        if self.clock() >= self.deadline:
            self.reason = "runtime_limit"
            return RoleAnnotation("unavailable", "runtime_limit")
        if finding.finding_id not in self.static_ids:
            return RoleAnnotation("unknown", "unverified_analyzer_origin")
        # Cache paths alone cannot identify a transitive tree. Do not use a
        # same-named local file for a finding carrying any external provenance.
        if (
            finding.source_identity
            or finding.source_digest
            or finding.source_url
            or finding.transitive_depth
            or finding.file.startswith("external/")
        ):
            return RoleAnnotation("unknown", "unsupported_source_scope")
        if finding.occurrences:
            return RoleAnnotation("unknown", "occurrence_not_expanded")
        if _DERIVED_VIEW_TAGS.intersection(finding.tags):
            return RoleAnnotation("unknown", "transformed_view_not_supported")
        if finding.start_column is None or finding.end_column is None:
            return RoleAnnotation("unknown", "missing_columns")
        path = finding.file
        index = self.indices.get(path)
        if index is None:
            if len(self.indices) >= MAX_ROLE_DOCUMENTS:
                self.reason = "document_limit"
                return RoleAnnotation("unavailable", "document_limit")
            raw = self.raw.get(path)
            text = self.text.get(path)
            if not isinstance(raw, bytes) or not isinstance(text, str):
                index = StructuredSourceMap("unavailable", "missing_source_snapshot")
            elif self.input_bytes + len(raw) > MAX_ROLE_INPUT_BYTES:
                self.reason = "total_bytes_limit"
                index = StructuredSourceMap("unavailable", "total_bytes_limit")
            else:
                self.input_bytes += len(raw)
                # Avoid decoding a file larger than the source index accepts.
                if len(raw) > 256 * 1024:
                    index = StructuredSourceMap("unavailable", "size_limit")
                else:
                    try:
                        original = raw.decode("utf-8", errors="strict")
                    except UnicodeDecodeError:
                        index = StructuredSourceMap("unavailable", "invalid_utf8")
                    else:
                        if original != text:
                            index = StructuredSourceMap("unavailable", "cache_view_mismatch")
                        else:
                            index = index_structured_source(
                                raw, deadline=self.deadline, clock=self.clock
                            )
            self.indices[path] = index
        if index.status != "ready":
            return RoleAnnotation("unavailable", index.reason)
        assert index.content_sha256 is not None
        return index.locate_lines(
            finding.start_line,
            finding.end_line,
            finding.start_column,
            finding.end_column,
            content_sha256=index.content_sha256,
            coordinate_space="original_unicode",
        )


def annotate_structured_report_findings(
    findings: Sequence[Finding],
    state: Mapping[str, object],
    *,
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[list[Finding], dict[str, object]]:
    """Return unsuppressed display copies plus auxiliary coverage accounting.

    The caller has already expanded each occurrence and computed risk. This
    function never modifies its inputs and never reads files, follows resources,
    invokes a provider, or changes a finding's score/confidence/fingerprint.
    It does not annotate ordinary source files or make conformance claims.
    """
    if deadline is not None and (type(deadline) not in (float, int) or not math.isfinite(deadline)):
        raise ValueError("deadline must be a finite monotonic timestamp")
    mapper = _RoleReport(state, deadline=deadline, clock=clock)
    rendered: list[Finding] = []
    eligible = 0
    omitted = 0
    for finding in findings:
        if not _local_component(finding.file, mapper.components):
            rendered.append(finding)
            continue
        eligible += 1
        if mapper.records >= MAX_ROLE_RECORDS or mapper.reason == "output_limit":
            omitted += 1
            mapper.reason = mapper.reason or "record_limit"
            rendered.append(finding)
            continue
        annotation = mapper.annotation(finding).to_dict()
        # Account for JSON escaping rather than Python repr: non-ASCII keys
        # and quotes can expand in JSON/SARIF even when the source is small.
        # This bounds annotation objects before report indentation/containers.
        size = len(json.dumps(annotation, ensure_ascii=True))
        if mapper.output_chars + size > MAX_ROLE_OUTPUT_CHARS:
            omitted += 1
            mapper.reason = "output_limit"
            rendered.append(finding)
            continue
        # Never overwrite analysis evidence, including a same-named extension.
        if ROLE_EVIDENCE_KEY in finding.evidence:
            omitted += 1
            mapper.reason = "evidence_key_collision"
            rendered.append(finding)
            continue
        mapper.records += 1
        mapper.output_chars += size
        mapper.exact += annotation["mapping_status"] == "exact"
        mapper.unknown += annotation["mapping_status"] != "exact"
        rendered.append(
            replace(finding, evidence={**finding.evidence, ROLE_EVIDENCE_KEY: annotation})
        )
    return rendered, {
        "scope": "unsuppressed_local_aisop_raw_source",
        "eligible_occurrences": eligible,
        "annotated_occurrences": mapper.records,
        "exact_occurrences": mapper.exact,
        "unknown_or_unavailable_occurrences": mapper.unknown,
        "omitted_occurrences": omitted,
        "documents_examined": len(mapper.indices),
        "documents_indexed": sum(index.status == "ready" for index in mapper.indices.values()),
        "input_bytes": mapper.input_bytes,
        "limitation": mapper.reason,
        "affects_detection_or_scoring": False,
    }
