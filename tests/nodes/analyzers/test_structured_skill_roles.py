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

"""Tests for the structured_skill_roles analyzer."""

from __future__ import annotations

from pathlib import Path

from skillspector.nodes.analyzers import structured_skill_roles as module
from skillspector.structured_skill import extract_structured_skill_context


def _write_aisop_bundle(path: Path) -> None:
    path.write_text(
        """
[
  {
    "role": "system",
    "content": {
      "protocol": "AISOP V1",
      "format": "workflow"
    }
  },
  {
    "role": "user",
    "content": {
      "aisop": {
        "main": "graph TD"
      },
      "functions": {
        "lookup": {"constraints": ["query"]}
      },
      "declared_tools": ["search", "calendar"]
    }
  }
]
""",
        encoding="utf-8",
    )


def test_no_structured_context_returns_no_findings() -> None:
    """A skill without structured-skill context yields no SSR-1 summary."""
    assert module.node({})["findings"] == []


def test_structured_bundle_emits_single_report_only_ssr1(tmp_path: Path) -> None:
    """Valid structured bundle context produces one report-only SSR-1 summary."""
    path = tmp_path / "bundle.aisop.json"
    _write_aisop_bundle(path)
    context = extract_structured_skill_context(tmp_path)
    assert context is not None

    result = module.node({"structured_skill_context": context})
    assert result["findings"] == []
    assert len(result["structured_summaries"]) == 1

    summary = result["structured_summaries"][0]
    assert summary["id"] == "SSR-1"
    assert summary["message"] == f"Structured {summary['layout_kind']} bundle detected (AISOP V1)"
    assert summary["file"] == str(path.resolve())
    assert summary["protocol"] == "AISOP V1"
    assert summary["layout_kind"] == context["layout_kind"]
    assert summary["declared_tools"] == ["calendar", "search"]
    assert summary["workflow_nodes"] == context["workflow_nodes"]
    assert summary["constraints"] == context["constraint_anchors"]
    assert summary["resources"] == context["resource_anchors"]
    assert summary["tags"] == ["AISOP", "AISP", "structured-skill"]
    assert "severity" not in summary
    assert "confidence" not in summary


def test_malformed_context_does_not_raise_no_findings(tmp_path: Path) -> None:
    """Malformed bundle parsing failure surfaces as no structured context and no finding."""
    (tmp_path / "bundle.aisop.json").write_text("{bad", encoding="utf-8")
    context = extract_structured_skill_context(tmp_path)
    assert context is None

    result = module.node({"structured_skill_context": context})
    assert result["findings"] == []


def test_analyzer_does_not_require_llm_credentials(tmp_path: Path) -> None:
    """Structured-skill analyzer is static and works without any LLM credentials."""
    path = tmp_path / "bundle.aisop.json"
    _write_aisop_bundle(path)
    context = extract_structured_skill_context(tmp_path)
    assert context is not None
    result = module.node({"structured_skill_context": context})
    assert result["findings"] == []
    assert result["structured_summaries"][0]["id"] == "SSR-1"
