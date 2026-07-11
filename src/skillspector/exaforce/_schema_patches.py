# SPDX-License-Identifier: Apache-2.0
"""Prune unused keys from the LLM structured-output schemas (fork behavior).

Reproduces the schema pruning that would otherwise live in upstream
``llm_analyzer_base`` / ``meta_analyzer``, keeping those upstream files at
parity with NVIDIA/SkillSpector.
"""

from __future__ import annotations

import skillspector.llm_analyzer_base as llm_base
import skillspector.nodes.meta_analyzer as meta
from skillspector.models import Finding

from ._patchlib import pop_field_validator, remove_model_fields


def _pruned_to_finding(self: "llm_base.LLMFinding", file: str) -> Finding:
    """``LLMFinding.to_finding`` without the removed explanation/remediation."""
    return Finding(
        rule_id=self.rule_id,
        message=self.message,
        severity=self.severity,
        confidence=self.confidence,
        file=file,
        start_line=self.start_line,
        end_line=self.end_line,
    )


def apply() -> None:
    # LLMFinding: drop explanation + remediation, and stop forwarding them.
    remove_model_fields(llm_base.LLMFinding, ["explanation", "remediation"])
    llm_base.LLMFinding.to_finding = _pruned_to_finding
    llm_base.LLMFinding.model_rebuild(force=True)
    llm_base.LLMAnalysisResult.model_rebuild(force=True)

    # MetaAnalyzerFinding: drop intent + impact.
    remove_model_fields(meta.MetaAnalyzerFinding, ["intent", "impact"])
    meta.MetaAnalyzerFinding.model_rebuild(force=True)

    # MetaAnalyzerResult: drop overall_assessment field + its validator.
    remove_model_fields(meta.MetaAnalyzerResult, ["overall_assessment"])
    pop_field_validator(meta.MetaAnalyzerResult, "_parse_stringified_assessment")
    meta.MetaAnalyzerResult.model_rebuild(force=True)
