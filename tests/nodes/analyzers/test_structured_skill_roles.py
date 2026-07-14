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
        "declared_tools": ["search", "calendar"],
        "functions": {
          "lookup": {"constraints": [{"anchor": "query"}]}
        }
      }
    }
  }
]
""",
        encoding="utf-8",
    )


def test_no_structured_context_returns_no_findings() -> None:
    """A skill without structured-skill context yields no SSR-1 findings."""
    assert module.node({})["findings"] == []


def test_structured_bundle_emits_single_low_ssr1(tmp_path: Path) -> None:
    """Valid structured bundle context produces one LOW SSR-1 finding."""
    path = tmp_path / "bundle.aisop.json"
    _write_aisop_bundle(path)
    context = extract_structured_skill_context(tmp_path)
    assert context is not None

    result = module.node({"structured_skill_context": context})
    assert len(result["findings"]) == 1

    finding = result["findings"][0]
    assert finding.rule_id == "SSR-1"
    assert finding.severity == "LOW"
    assert finding.file == str(path.resolve())
    assert finding.matched_text is not None
    assert finding.context is not None
    assert "declared_tools" in finding.context


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
    assert result["findings"][0].rule_id == "SSR-1"
