# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Format-neutral recursive retention and compact post-analysis report inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass

from pydantic import BaseModel

from skillspector.multi_skill import SkillDirectory
from skillspector.nodes.report import PreparedReport


@dataclass(slots=True)
class RecursiveChild:
    """A retained prepared snapshot and bounded policy projection, never a graph."""

    skill: SkillDirectory
    prepared: PreparedReport | None
    policy: dict[str, object]
    error: str | None = None


@dataclass(slots=True)
class RecursiveReportCollection:
    """One completed collection reusable by independent format serializers."""

    children: list[RecursiveChild] = field(default_factory=list)
    skill_count: int = 0
    scanned_skill_count: int = 0
    unscanned_skill_count: int = 0
    omitted_symlink_entry_count: int = 0
    output_omitted_skill_count: int = 0
    retained_public_records: int = 0
    retained_prepared_bytes: int = 0
    retained_metadata_records: int = 0
    max_score: int = 0
    execution_failed: bool = False
    coverage_failed: bool = False
    has_findings: bool = False
    transitive_finding_count: int = 0
    transitive_sources: list[str] = field(default_factory=list)
    analysis_completeness: dict[str, object] = field(default_factory=dict)
    risk_assessment: dict[str, object] = field(default_factory=dict)
    retention_completeness: dict[str, object] = field(default_factory=dict)

    def release(self) -> None:
        """Release snapshots after all output attempts, including exceptional exits."""
        self.children.clear()


def prepared_payload_usage(
    prepared: PreparedReport, *, max_bytes: int, max_records: int
) -> tuple[int, int]:
    """Count canonical UTF-8 payload and collection records without rendering.

    Derived public/display fields share canonical objects. Visit each container
    or model once; scalar values are charged at every retained field. Fixed
    container/field overhead also bounds empty records. Stop at the first bound
    rather than constructing an intermediate JSON document or model dump.
    """
    seen: set[int] = set()
    byte_count = 0
    record_count = 0

    def visit(value: object, depth: int = 0) -> None:
        nonlocal byte_count, record_count
        if byte_count > max_bytes or record_count > max_records:
            return
        if depth > 128:
            # Fail closed on non-canonical/cyclic metadata rather than retaining
            # a payload whose size cannot be established with bounded work.
            byte_count = max_bytes + 1
            return
        if isinstance(value, str):
            byte_count += len(value.encode("utf-8"))
        elif value is None or isinstance(value, (bool, int, float)):
            byte_count += 8
        elif isinstance(value, bytes):
            byte_count += len(value)
        else:
            identity = id(value)
            if identity in seen:
                return
            seen.add(identity)
            byte_count += 16
            if isinstance(value, Mapping):
                record_count += len(value)
                for key, item in value.items():
                    visit(key, depth + 1)
                    visit(item, depth + 1)
                    if byte_count > max_bytes or record_count > max_records:
                        break
            elif isinstance(value, (list, tuple)):
                record_count += len(value)
                for item in value:
                    visit(item, depth + 1)
                    if byte_count > max_bytes or record_count > max_records:
                        break
            elif isinstance(value, BaseModel):
                for name in type(value).model_fields:
                    byte_count += 8
                    visit(getattr(value, name), depth + 1)
            elif is_dataclass(value) and not isinstance(value, type):
                for item in fields(value):
                    byte_count += 8
                    visit(getattr(value, item.name), depth + 1)
            else:
                # Preparation should only expose canonical data types.
                byte_count = max_bytes + 1

    visit(prepared)
    return byte_count, record_count


