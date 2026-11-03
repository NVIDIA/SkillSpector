# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Oversized physical source lines retain exact provenance and honest coverage."""

from unittest.mock import MagicMock

import pytest

import skillspector.llm_analyzer_base as lab
import skillspector.nodes.meta_analyzer as meta
from skillspector.dependency_sources import redact_text
from skillspector.inspection_ledger import finalize_ledger
from skillspector.models import Finding


def planner():
    value = meta.LLMMetaAnalyzer.__new__(meta.LLMMetaAnalyzer)
    value.base_prompt = meta.PER_FILE_ANALYSIS_PROMPT
    value._input_budget = 16000
    value._token_counter = lambda text: len(text.encode())
    value._timeout = None
    value._pack_files = False
    value._usage_collector = MagicMock()
    value._usage_collector.snapshot.return_value = []
    return value


def finding(rule="SC4", path="synthetic.js"):
    return Finding(
        rule_id=rule,
        message="Synthetic security concern retained",
        severity="HIGH",
        file=path,
        start_line=1,
    )


def members(batches):
    return batches


@pytest.mark.parametrize("missing", [False, True])
def test_actual_meta_planner_keeps_incomplete_fragment_line_failed(missing):
    a = planner()
    f = finding()
    units = members(a.get_batches(["synthetic.js"], {"synthetic.js": "x" * 100000}, [f]))
    assert len(units) > 2
    dropped = units[len(units) // 2]
    outcome = lab.BatchExecutionResult(
        successful=[(b, []) for b in units if b is not dropped],
        failures=[] if missing else [lab.BatchFailure(dropped, "SyntheticFragmentFailure")],
    )
    events, status = meta._meta_ledger_response(units, outcome, [f])
    assert status["status"] == "failed"
    assert any(
        e["outcome"] == "failed" and f.finding_id in e["emitted_finding_ids"] for e in events
    )


def test_all_meta_fragment_results_missing_is_failed_and_preserves_finding():
    a = planner()
    f = finding()
    units = members(a.get_batches(["synthetic.js"], {"synthetic.js": "x" * 100000}, [f]))
    events, status = meta._meta_ledger_response(units, lab.BatchExecutionResult(), [f])
    assert status["status"] == "failed"
    assert any(f.finding_id in e["emitted_finding_ids"] for e in events)


def test_sc10_credential_url_crossing_proposed_fragment_boundary_never_leaks():
    a = planner()
    ordinary = finding()
    plain = "x" * 100000
    first = members(a.get_batches(["synthetic.js"], {"synthetic.js": plain}, [ordinary]))
    boundary = first[0].source_end_column
    marker = "SYNTHETIC_FAKE_SECRET_"
    url = "https://user:" + marker * 100 + "@example.invalid/path"
    content = plain[: boundary - 500] + " " + url + " " + plain[boundary - 500 :]
    assert marker not in redact_text(content)
    f = finding("SC10")
    batches = a.get_batches(["synthetic.js"], {"synthetic.js": content}, [f])
    units = members(batches)
    assert len(units) == 1 and units[0].source_start_column is None, (
        "sensitive source remains whole for normal redaction"
    )
    assert marker not in a.build_prompt(units[0])
    with pytest.raises(lab.LLMInputBudgetError):
        a._check_prompt_budget(a.build_prompt(units[0]))
    events, status = meta._meta_ledger_response(
        units,
        lab.BatchExecutionResult(failures=[lab.BatchFailure(units[0], "LLMInputBudgetError")]),
        [f],
    )
    assert status["status"] == "failed" and f.finding_id in events[0]["emitted_finding_ids"]


def test_redacted_file_keeps_other_lines_out_of_original_column_provenance():
    a = planner()
    secret = finding("SC10")
    other = finding()
    other.start_line = 2
    marker = "SYNTHETIC_FAKE_SECRET"
    text = "credential declared here\n" + "x" * 100000 + f" https://user:{marker}@example.invalid\n"
    cache = {"synthetic.js": text}
    batches = a.get_batches(["synthetic.js"], cache, [secret, other])
    assert cache["synthetic.js"] == text
    assert all(batch.source_start_column is None for batch in batches)
    assert all(marker not in a.build_prompt(batch) for batch in batches)
    oversized = [batch for batch in batches if batch.start_line == batch.end_line == 2]
    assert oversized
    with pytest.raises(lab.LLMInputBudgetError):
        a._check_prompt_budget(a.build_prompt(oversized[0]))


@pytest.mark.parametrize("mode", ["complete", "failure", "missing", "unknown"])
def test_real_meta_node_reconciles_every_fragment_and_retains_original_findings(monkeypatch, mode):
    a = planner()
    f = finding()
    text = "x" * 100000
    preview = members(a.get_batches(["synthetic.js"], {"synthetic.js": text}, [f]))
    target = preview[len(preview) // 2].source_start_column
    seen = []

    async def invoke(batch, prompt, **kwargs):
        a._check_prompt_budget(prompt)
        seen.append((batch.source_start_column, batch.source_end_column))
        if mode == "failure" and batch.source_start_column == target:
            raise RuntimeError("SyntheticFragmentFailure")
        item = {
            "finding_id": f.finding_id,
            "pattern_id": f.rule_id,
            "start_line": 1,
            "is_vulnerability": True,
            "confidence": 0.9,
            "intent": "negligent",
            "impact": "high",
            "_file": f.file,
        }
        return batch, [item]

    monkeypatch.setattr(a, "_ainvoke_batch_with_retries", invoke)
    original_arun = a.arun_batches
    if mode in ("missing", "unknown"):

        async def omit_one(requests, **kwargs):
            result = await original_arun(requests, **kwargs)
            result = [(b, v) for b, v in result if b.source_start_column != target]
            if mode == "unknown":
                result.append((lab.Batch(file_path="synthetic.js", content=text, findings=[f]), []))
            a._last_batch_outcome = lab.BatchExecutionResult(successful=result, failures=[])
            return result

        monkeypatch.setattr(a, "arun_batches", omit_one)
    monkeypatch.setattr(meta, "LLMMetaAnalyzer", lambda **kwargs: a)
    response = meta.meta_analyzer(
        {
            "findings": [f],
            "components": ["synthetic.js"],
            "file_cache": {"synthetic.js": text},
            "manifest": {},
            "model_config": {"default": "synthetic-offline"},
        }
    )
    assert len(seen) == len(preview) == len(set(seen)), (
        "each planned fragment ran once through normal aggregation"
    )
    assert f.finding_id in response["effective_finding_ids"]
    retained = next(value for value in response["findings"] if value.finding_id == f.finding_id)
    completeness, _ = finalize_ledger(
        {
            "components": ["synthetic.js"],
            "findings": response["findings"],
            "inspection_ledger": response["inspection_ledger"],
            "analyzer_status_events": response["analyzer_status_events"],
            "meta_review_required": True,
        }
    )
    if mode == "complete":
        assert response["analyzer_status_events"][0]["status"] == "completed"
        assert completeness["execution_successful"] is True
    else:
        assert response["analyzer_status_events"][0]["status"] == "failed"
        assert retained.evidence["llm_review_outcome"] == "failed"
        assert completeness["execution_successful"] is False
        assert response["llm_call_log"][0]["ok"] is False
