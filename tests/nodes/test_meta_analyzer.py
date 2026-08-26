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

"""Tests for the meta_analyzer node.

Covers ``LLMMetaAnalyzer`` filtering and partial-batch-failure handling, plus
the LLM-call telemetry and fail-closed construction that drive the report's
degradation signal.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason, finalize_ledger
from skillspector.llm_analyzer_base import (
    Batch,
    BatchExecutionResult,
    BatchFailure,
    LLMAnalyzerBase,
)
from skillspector.llm_utils import run_async
from skillspector.models import Finding
from skillspector.nodes.analyzers import static_patterns_anti_refusal
from skillspector.nodes.analyzers.static_runner import analyzer_finding_to_finding
from skillspector.nodes.meta_analyzer import (
    LLMMetaAnalyzer,
    MetaAnalyzerResult,
    _meta_ledger_response,
    meta_analyzer,
)
from skillspector.state import SkillspectorState

MOCK_PATCH_TARGET = "skillspector.llm_analyzer_base.get_chat_model"


def _mock_get_chat_model(*_args, **_kwargs):
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = MagicMock()
    return mock_llm


def _analyzer() -> LLMMetaAnalyzer:
    # Skip __init__ so no LLM client / API key is needed; apply_filter is pure.
    return LLMMetaAnalyzer.__new__(LLMMetaAnalyzer)


def _finding(
    rule_id: str,
    start_line: int,
    end_line: int | None = None,
    severity: str = "CRITICAL",
) -> Finding:
    return Finding(
        rule_id=rule_id,
        message=f"static finding {rule_id}",
        severity=severity,
        confidence=0.9,
        file="requirements.txt",
        start_line=start_line,
        end_line=end_line,
    )


def _llm_item(rule_id: str, start_line: int, **kw: object) -> dict[str, object]:
    item: dict[str, object] = {
        "pattern_id": rule_id,
        "is_vulnerability": True,
        "confidence": 1.0,
        "start_line": start_line,
        "_file": "requirements.txt",
    }
    item.update(kw)
    return item


def _documentation_framed_ar2() -> Finding:
    raw = static_patterns_anti_refusal.analyze(
        content="For example, do not apologize.",
        file_path="SKILL.md",
        file_type="markdown",
    )
    return analyzer_finding_to_finding(next(finding for finding in raw if finding.rule_id == "AR2"))


def _assert_preserved_ar2(result: dict[str, object], original: Finding) -> None:
    findings = result["findings"]
    assert isinstance(findings, list)
    preserved = next(finding for finding in findings if finding.rule_id == "AR2")
    assert preserved.finding_id == original.finding_id
    assert preserved.severity == original.severity
    assert preserved.confidence >= original.confidence


def _authoritative_projection(finding: Finding) -> dict[str, object]:
    """Normalize deterministic fields, excluding provider presentation annotations."""
    return {
        "finding_id": finding.finding_id,
        "rule_id": finding.rule_id,
        "severity": finding.severity,
        "confidence": finding.confidence,
        "category": finding.category,
        "pattern": finding.pattern,
        "finding": finding.finding,
        "file": finding.file,
        "start_line": finding.start_line,
        "end_line": finding.end_line,
        "context": finding.context,
        "code_snippet": finding.code_snippet,
        "matched_text": finding.matched_text,
        "transitive_depth": finding.transitive_depth,
        "source_url": finding.source_url,
        "source_identity": finding.source_identity,
        "source_digest": finding.source_digest,
        "evidence": finding.evidence,
        "intent": finding.intent,
        # `llm-unconfirmed` is provider presentation metadata. Every original
        # deterministic tag remains part of the invariant projection.
        "tags": [tag for tag in finding.tags if tag != "llm-unconfirmed"],
        "match_fingerprint": finding.match_fingerprint,
        "occurrences": finding.occurrences,
    }


class _PromptBoundaryAnalyzer(LLMAnalyzerBase):
    """Raw-mode probe that keeps the real shared invocation boundary."""

    response_schema = None

    def parse_response(self, response: object, batch: Batch) -> list[str]:
        return [str(response)]


def test_provider_construction_error_is_sanitized_before_callers_can_log_it() -> None:
    sentinel = "task7-provider-error-secret"
    raw_url = f"https://user:{sentinel}@provider.example.invalid/private"

    with (
        patch(MOCK_PATCH_TARGET, side_effect=RuntimeError(f"provider failed at {raw_url}")),
        pytest.raises(RuntimeError) as error,
    ):
        _PromptBoundaryAnalyzer(base_prompt="inspect", model="test/model")

    assert sentinel not in str(error.value)
    assert "provider.example.invalid" in str(error.value)


def test_documentation_framed_finding_survives_provider_outcome_matrix() -> None:
    original = _documentation_framed_ar2()
    state: SkillspectorState = {
        "findings": [original],
        "use_llm": False,
        "file_cache": {"SKILL.md": "For example, do not apologize."},
        "llm_file_cache": {"SKILL.md": "For example, do not apologize."},
        "manifest": {},
        "model_config": {},
    }

    _assert_preserved_ar2(meta_analyzer(state), original)

    batch = Batch(
        file_path="SKILL.md", content=state["file_cache"]["SKILL.md"], findings=[original]
    )
    denial = {
        "pattern_id": "AR2",
        "is_vulnerability": False,
        "confidence": 0.1,
        "start_line": original.start_line,
        "_file": "SKILL.md",
    }
    confirmed = {
        "pattern_id": "AR2",
        "is_vulnerability": True,
        "confidence": 0.95,
        "start_line": original.start_line,
        "_file": "SKILL.md",
        "explanation": "confirmed",
        "remediation": "review",
    }
    for response in ([denial], [confirmed]):
        filtered = _analyzer().apply_filter([original], [(batch, response)])
        _assert_preserved_ar2({"findings": filtered}, original)

    llm_state = dict(state)
    llm_state["use_llm"] = True
    for error in (TimeoutError("provider timeout"), RuntimeError("provider failure")):
        with patch("skillspector.nodes.meta_analyzer.LLMMetaAnalyzer") as mock_cls:
            mock_cls.return_value.get_batches.return_value = [batch]
            mock_cls.return_value.arun_batches = AsyncMock(side_effect=error)
            mock_cls.return_value.response_received = False
            mock_cls.return_value.inference_usage = []
            _assert_preserved_ar2(meta_analyzer(llm_state), original)

    with patch("skillspector.nodes.meta_analyzer.LLMMetaAnalyzer") as mock_cls:
        mock_cls.return_value.get_batches.return_value = [batch]
        mock_cls.return_value.arun_batches = AsyncMock(
            side_effect=ValueError("malformed structured response")
        )
        mock_cls.return_value.response_received = True
        mock_cls.return_value.inference_usage = []
        _assert_preserved_ar2(meta_analyzer(llm_state), original)


def test_authoritative_projection_is_invariant_across_every_provider_outcome() -> None:
    original = Finding(
        rule_id="SC10",
        message="deterministic dependency source replacement",
        finding_id="task7-authoritative-finding",
        severity="HIGH",
        confidence=0.91,
        file="pip.conf",
        start_line=3,
        end_line=3,
        category="supply_chain",
        pattern="dependency-source",
        finding="pip source replacement",
        context="sanitized deterministic context",
        code_snippet="sanitized deterministic snippet",
        matched_text="index-url = redacted destination",
        transitive_depth=1,
        source_url="https://packages.example.invalid/REDACTED_PATH",
        source_identity="sha256:task9-source",
        source_digest="sha256:task9-digest",
        evidence={"surface": "pip.conf", "operation": "replace", "scope": "global"},
        intent="negligent",
        match_fingerprint="sha256:task9-match",
        occurrences=[{"file": "pip.conf", "start_line": 3, "end_line": 3}],
    )
    batch = Batch(file_path=original.file, content="safe provider copy", findings=[original])
    expected = {
        "finding_id": "task7-authoritative-finding",
        "rule_id": "SC10",
        "severity": "HIGH",
        "confidence": 0.91,
        "category": "supply_chain",
        "pattern": "dependency-source",
        "finding": "pip source replacement",
        "file": "pip.conf",
        "start_line": 3,
        "end_line": 3,
        "context": "sanitized deterministic context",
        "code_snippet": "sanitized deterministic snippet",
        "matched_text": "index-url = redacted destination",
        "transitive_depth": 1,
        "source_url": "https://packages.example.invalid/REDACTED_PATH",
        "source_identity": "sha256:task9-source",
        "source_digest": "sha256:task9-digest",
        "evidence": {"surface": "pip.conf", "operation": "replace", "scope": "global"},
        "intent": "negligent",
        "tags": [],
        "match_fingerprint": "sha256:task9-match",
        "occurrences": [{"file": "pip.conf", "start_line": 3, "end_line": 3}],
    }
    state: SkillspectorState = {
        "findings": [original],
        "use_llm": False,
        "file_cache": {original.file: "safe canonical content"},
        "llm_file_cache": {original.file: "safe provider copy"},
        "manifest": {},
        "model_config": {},
    }
    projected: dict[str, Finding] = {}

    disabled_result = meta_analyzer(state)
    [projected["disabled"]] = disabled_result["findings"]

    failed_state = dict(state)
    failed_state["use_llm"] = True
    with patch("skillspector.nodes.meta_analyzer.LLMMetaAnalyzer") as mock_cls:
        mock_cls.return_value.get_batches.return_value = [batch]
        mock_cls.return_value.arun_batches = AsyncMock(side_effect=TimeoutError("provider timeout"))
        mock_cls.return_value.response_received = False
        mock_cls.return_value.inference_usage = []
        failed_result = meta_analyzer(failed_state)
    [projected["failed"]] = failed_result["findings"]

    with patch("skillspector.nodes.meta_analyzer.LLMMetaAnalyzer") as mock_cls:
        mock_cls.return_value.get_batches.return_value = [batch]
        mock_cls.return_value.arun_batches = AsyncMock(
            side_effect=ValueError("malformed structured response")
        )
        mock_cls.return_value.response_received = True
        mock_cls.return_value.inference_usage = []
        malformed_result = meta_analyzer(failed_state)
    [projected["malformed"]] = malformed_result["findings"]

    provider_outcomes: dict[str, list[dict[str, object]]] = {
        "empty": [],
        "confirming": [
            {
                "pattern_id": "SC10",
                "is_vulnerability": True,
                "confidence": 0.99,
                "start_line": 3,
                "_file": "pip.conf",
                "explanation": "useful provider presentation context",
                "remediation": "use the canonical registry",
            }
        ],
        "downgrading": [
            {
                "pattern_id": "SC10",
                "is_vulnerability": True,
                "confidence": 0.0,
                "start_line": 3,
                "_file": "pip.conf",
                "severity": "LOW",
            }
        ],
        "suppressing": [
            {
                "pattern_id": "SC10",
                "is_vulnerability": False,
                "confidence": 0.0,
                "start_line": 3,
                "_file": "pip.conf",
                "severity": "LOW",
                "category": "benign",
                "file": "other.conf",
                "matched_text": "rewritten",
                "evidence": {},
            }
        ],
        "rewriting": [
            {
                "pattern_id": "SC10",
                "is_vulnerability": True,
                "confidence": 0.99,
                "start_line": 3,
                "_file": "pip.conf",
                "explanation": "useful provider presentation context",
                "remediation": "use the canonical registry",
                "finding_id": "hostile-rewrite",
                "severity": "LOW",
                "evidence": {"surface": "hostile"},
            }
        ],
        "hostile": [
            {
                "pattern_id": "HOSTILE",
                "is_vulnerability": True,
                "confidence": 1.0,
                "start_line": 3,
                "_file": "pip.conf",
            }
        ],
    }

    for outcome, provider_items in provider_outcomes.items():
        [result] = _analyzer().apply_filter([original], [(batch, provider_items)])
        projected[outcome] = result

    assert set(projected) == {
        "disabled",
        "failed",
        "malformed",
        "empty",
        "confirming",
        "downgrading",
        "suppressing",
        "rewriting",
        "hostile",
    }
    for outcome, result in projected.items():
        assert _authoritative_projection(result) == expected, outcome


@patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
def test_final_meta_provider_request_redacts_every_unique_credential_carrier(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinels = {
        "https": "task9-provider-https-secret",
        "ssh": "task9-provider-ssh-secret",
        "scp": "task9-provider-scp-secret",
        "query": "task9-provider-query-secret",
        "fragment": "task9-provider-fragment-secret",
        "assignment": "task9-provider-assignment-secret",
        "heredoc": "task9-provider-heredoc-secret",
        "generated_config": "task9-provider-generated-secret",
        "finding": "task9-provider-finding-secret",
        "provider_batch": "task9-provider-batch-secret",
    }
    finding_url = f"https://finding:{sentinels['finding']}@finding.example.invalid/private"
    finding = Finding(
        rule_id="SC10",
        message=f"dependency source {finding_url}",
        file="scripts/setup.sh",
        start_line=1,
        matched_text=f"registry={finding_url}",
        context=f"generated registry {finding_url}",
    )
    content = "\n".join(
        (
            f"https://user:{sentinels['https']}@https.example.invalid/private",
            f"ssh://git:{sentinels['ssh']}@ssh.example.invalid/org/repo.git",
            f"{sentinels['scp']}@scp.example.invalid:org/repo.git",
            f"https://query.example.invalid/private?token={sentinels['query']}",
            f"https://fragment.example.invalid/private#{sentinels['fragment']}",
            f"REGISTRY=https://user:{sentinels['assignment']}@assignment.example.invalid/private",
            "cat <<'EOF'",
            f"registry=https://user:{sentinels['heredoc']}@heredoc.example.invalid/private",
            "EOF",
            "cat >.npmrc <<'EOF'",
            "registry=https://user:"
            f"{sentinels['generated_config']}@generated.example.invalid/private",
            "EOF",
            f"provider=https://user:{sentinels['provider_batch']}@provider.example.invalid/private",
        )
    )
    analyzer = LLMMetaAnalyzer(model="test/model")
    batch = Batch(file_path="scripts/setup.sh", content=content, findings=[finding])
    submitted: list[str] = []

    def capture(_llm: object, prompt: str, _collector: object) -> MetaAnalyzerResult:
        submitted.append(prompt)
        return MetaAnalyzerResult()

    with (
        caplog.at_level(logging.DEBUG, logger="skillspector"),
        patch("skillspector.llm_analyzer_base._invoke_with_usage", side_effect=capture),
    ):
        outcome = analyzer.run_batches_detailed([batch], metadata_text="No metadata available")

    assert len(outcome.successful) == 1
    assert outcome.failures == []
    assert len(submitted) == 1
    serialized_request = submitted[0]
    for carrier, sentinel in sentinels.items():
        assert sentinel not in serialized_request, carrier
        assert sentinel not in caplog.text, carrier


@patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
def test_sync_provider_prompt_is_redacted_immediately_before_invocation() -> None:
    sentinel = "task7-sync-prompt-secret"
    analyzer = _PromptBoundaryAnalyzer(base_prompt="inspect", model="test/model")
    batch = Batch(
        file_path="pip.conf",
        content=f"index-url = https://user:{sentinel}@packages.example.invalid/private",
    )
    submitted: list[str] = []

    def capture(_llm: object, prompt: str, _collector: object) -> AIMessage:
        submitted.append(prompt)
        return AIMessage(content="ok")

    with patch("skillspector.llm_analyzer_base._invoke_with_usage", side_effect=capture):
        outcome = analyzer.run_batches_detailed([batch])

    assert len(outcome.successful) == 1
    assert len(submitted) == 1
    assert sentinel not in submitted[0]
    assert "packages.example.invalid" in submitted[0]


@patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
def test_sync_embedded_scheme_relative_prompt_never_reaches_provider_raw() -> None:
    sentinel = "task9-sync-scheme-relative-secret"
    analyzer = _PromptBoundaryAnalyzer(base_prompt="inspect", model="test/model")
    batch = Batch(
        file_path="pip.conf",
        content=(f"index-url=//user:{sentinel}@packages.example.invalid/private?token={sentinel}"),
    )
    submitted: list[str] = []

    def capture(_llm: object, prompt: str, _collector: object) -> AIMessage:
        submitted.append(prompt)
        return AIMessage(content="ok")

    with patch("skillspector.llm_analyzer_base._invoke_with_usage", side_effect=capture):
        outcome = analyzer.run_batches_detailed([batch])

    assert len(outcome.successful) == 1
    assert len(submitted) == 1
    assert sentinel not in submitted[0]
    assert "[REDACTED_URL]" in submitted[0]


@patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
def test_async_embedded_scheme_relative_prompt_never_reaches_provider_raw() -> None:
    sentinel = "task9-async-scheme-relative-secret"
    analyzer = _PromptBoundaryAnalyzer(base_prompt="inspect", model="test/model")
    batch = Batch(
        file_path="pip.conf",
        content=(f"index-url=//user:{sentinel}@packages.example.invalid/private?token={sentinel}"),
    )
    submitted: list[str] = []

    async def capture(_llm: object, prompt: str, _collector: object) -> AIMessage:
        submitted.append(prompt)
        return AIMessage(content="ok")

    with patch(
        "skillspector.llm_analyzer_base._ainvoke_with_usage",
        new_callable=AsyncMock,
        side_effect=capture,
    ):
        outcome = run_async(analyzer.arun_batches_detailed([batch], max_concurrency=1))

    assert len(outcome.successful) == 1
    assert len(submitted) == 1
    assert sentinel not in submitted[0]
    assert "[REDACTED_URL]" in submitted[0]


@pytest.mark.parametrize(
    "template",
    [
        "<url>//user:{sentinel}@packages.example.invalid/{private_path}</url>",
        "url(//user:{sentinel}@packages.example.invalid/{private_path})",
    ],
    ids=("element-markup", "functional-markup"),
)
@patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
def test_sync_markup_embedded_scheme_relative_prompt_never_reaches_provider_raw(
    template: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "round2-sync-scheme-relative-secret"
    private_path = "round2-sync-private-path"
    analyzer = _PromptBoundaryAnalyzer(base_prompt="inspect", model="test/model")
    batch = Batch(
        file_path="pip.conf",
        content=template.format(sentinel=sentinel, private_path=private_path),
    )
    submitted: list[str] = []

    def capture(_llm: object, prompt: str, _collector: object) -> AIMessage:
        submitted.append(prompt)
        return AIMessage(content="ok")

    with (
        caplog.at_level(logging.DEBUG, logger="skillspector"),
        patch("skillspector.llm_analyzer_base._invoke_with_usage", side_effect=capture),
    ):
        outcome = analyzer.run_batches_detailed([batch])

    assert len(outcome.successful) == 1
    assert outcome.failures == []
    assert len(submitted) == 1
    assert sentinel not in submitted[0]
    assert private_path not in submitted[0]
    assert "[REDACTED_URL]" in submitted[0]
    assert sentinel not in caplog.text
    assert private_path not in caplog.text


@pytest.mark.parametrize(
    "template",
    [
        "<url>//user:{sentinel}@packages.example.invalid/{private_path}</url>",
        "url(//user:{sentinel}@packages.example.invalid/{private_path})",
    ],
    ids=("element-markup", "functional-markup"),
)
@patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
def test_async_markup_embedded_scheme_relative_prompt_never_reaches_provider_raw(
    template: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "round2-async-scheme-relative-secret"
    private_path = "round2-async-private-path"
    analyzer = _PromptBoundaryAnalyzer(base_prompt="inspect", model="test/model")
    batch = Batch(
        file_path="pip.conf",
        content=template.format(sentinel=sentinel, private_path=private_path),
    )
    submitted: list[str] = []

    async def capture(_llm: object, prompt: str, _collector: object) -> AIMessage:
        submitted.append(prompt)
        return AIMessage(content="ok")

    with (
        caplog.at_level(logging.DEBUG, logger="skillspector"),
        patch(
            "skillspector.llm_analyzer_base._ainvoke_with_usage",
            new_callable=AsyncMock,
            side_effect=capture,
        ),
    ):
        outcome = run_async(analyzer.arun_batches_detailed([batch], max_concurrency=1))

    assert len(outcome.successful) == 1
    assert outcome.failures == []
    assert len(submitted) == 1
    assert sentinel not in submitted[0]
    assert private_path not in submitted[0]
    assert "[REDACTED_URL]" in submitted[0]
    assert sentinel not in caplog.text
    assert private_path not in caplog.text


@patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
def test_sync_incomplete_prompt_redaction_makes_zero_calls_and_zero_retries() -> None:
    sentinel = "task7-sync-incomplete-secret"
    analyzer = _PromptBoundaryAnalyzer(base_prompt="inspect", model="test/model")
    candidates = " ".join(
        f"https://user:{sentinel}@host{index}.example.invalid/private" for index in range(1_025)
    )
    batch = Batch(file_path="pip.conf", content=candidates)

    with patch("skillspector.llm_analyzer_base._invoke_with_usage") as invoke:
        outcome = analyzer.run_batches_detailed([batch])

    invoke.assert_not_called()
    assert outcome.successful == []
    assert len(outcome.failures) == 1
    assert outcome.failures[0].error_class == "PromptRedactionIncomplete"
    assert outcome.failures[0].reason is LedgerReason.LLM_BATCH_FAILED


@patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
def test_async_incomplete_prompt_redaction_makes_zero_calls_and_zero_retries() -> None:
    sentinel = "task7-async-incomplete-secret"
    analyzer = _PromptBoundaryAnalyzer(base_prompt="inspect", model="test/model")
    candidates = " ".join(
        f"https://user:{sentinel}@host{index}.example.invalid/private" for index in range(1_025)
    )
    batch = Batch(file_path="pip.conf", content=candidates)

    with patch(
        "skillspector.llm_analyzer_base._ainvoke_with_usage", new_callable=AsyncMock
    ) as invoke:
        outcome = run_async(analyzer.arun_batches_detailed([batch], max_concurrency=1))

    invoke.assert_not_awaited()
    assert outcome.successful == []
    assert len(outcome.failures) == 1
    assert outcome.failures[0].error_class == "PromptRedactionIncomplete"
    assert outcome.failures[0].reason is LedgerReason.LLM_BATCH_FAILED


@patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
def test_prompt_redaction_failure_never_persists_or_logs_prompt_content(caplog) -> None:
    sentinel = "task7-prompt-log-secret"
    finding = Finding(rule_id="SC10", message="static", file="pip.conf", start_line=1)
    repeated_metadata = " ".join(
        f"https://user:{sentinel}@host{index}.example.invalid/private" for index in range(1_025)
    )
    state: SkillspectorState = {
        "findings": [finding],
        "use_llm": True,
        "llm_file_cache": {"pip.conf": "safe provider artifact"},
        "manifest": {"description": repeated_metadata},
        "model_config": {"meta_analyzer": "test/model"},
    }

    with caplog.at_level(logging.DEBUG):
        result = meta_analyzer(state)

    assert result["llm_call_log"] == [{"node": "meta_analyzer", "ok": False, "error": None}]
    assert sentinel not in caplog.text
    assert sentinel not in str(result["llm_call_log"])
    assert sentinel not in str(result["inspection_ledger"])


def test_confirmed_finding_kept_when_model_returns_end_line() -> None:
    """Regression: a static finding with end_line=None must still match a
    confirmation whose end_line is populated (e.g. end_line == start_line, as
    some models return). Previously these confirmed findings were silently
    dropped. See issue #67."""
    findings = [_finding("SC4", 4), _finding("SC4", 5)]
    items = [_llm_item("SC4", 4, end_line=4), _llm_item("SC4", 5, end_line=5)]
    batch = Batch(file_path="requirements.txt", content="", findings=findings)

    kept = _analyzer().apply_filter(findings, [(batch, items)])

    assert {f.start_line for f in kept} == {4, 5}
    assert len(kept) == 2


def test_rejected_finding_is_retained_as_unconfirmed() -> None:
    """LLM disagreement may annotate but cannot erase deterministic evidence."""
    findings = [_finding("SC4", 4, severity="MEDIUM")]
    items = [_llm_item("SC4", 4, end_line=4, is_vulnerability=False)]
    batch = Batch(file_path="requirements.txt", content="", findings=findings)

    kept = _analyzer().apply_filter(findings, [(batch, items)])

    assert len(kept) == 1
    assert "llm-unconfirmed" in kept[0].tags


def test_low_confidence_finding_is_retained_as_unconfirmed() -> None:
    """A low-confidence LLM verdict cannot erase deterministic evidence."""
    findings = [_finding("SC4", 4, severity="MEDIUM")]
    items = [_llm_item("SC4", 4, end_line=4, confidence=0.3)]
    batch = Batch(file_path="requirements.txt", content="", findings=findings)

    kept = _analyzer().apply_filter(findings, [(batch, items)])

    assert len(kept) == 1
    assert "llm-unconfirmed" in kept[0].tags


def test_finding_clones_preserve_security_metadata_and_confidence() -> None:
    """Confirmed and unconfirmed clones retain deterministic finding identity."""
    original = Finding(
        rule_id="SC4",
        message="static finding SC4",
        severity="MEDIUM",
        confidence=0.9,
        file="requirements.txt",
        start_line=4,
        intent="malicious",
        evidence={"source": "static", "nested": {"kind": "deterministic"}},
        match_fingerprint="sha256:deterministic",
        occurrences=[{"file": "requirements.txt", "start_line": 4, "end_line": 4}],
    )
    batch = Batch(file_path="requirements.txt", content="", findings=[original])
    outcomes = (
        [_llm_item("SC4", 4, end_line=4, confidence=0.7)],
        [_llm_item("SC4", 4, end_line=4, is_vulnerability=False)],
    )

    for items in outcomes:
        [returned] = _analyzer().apply_filter([original], [(batch, items)])
        assert returned.confidence == original.confidence
        assert returned.intent == original.intent
        assert returned.evidence == original.evidence
        assert returned.match_fingerprint == original.match_fingerprint
        assert returned.occurrences == original.occurrences


def test_exact_end_line_match_still_works() -> None:
    """Existing behaviour: when both sides carry the same concrete end_line,
    the finding is kept (no regression from the new fallback)."""
    findings = [_finding("AST1", 21, end_line=21)]
    items = [_llm_item("AST1", 21, end_line=21)]
    batch = Batch(file_path="requirements.txt", content="", findings=findings)

    kept = _analyzer().apply_filter(findings, [(batch, items)])

    assert len(kept) == 1
    assert kept[0].rule_id == "AST1"


def _confirm(pattern_id: str, file: str, start_line: int) -> dict[str, object]:
    """LLM item confirming a finding, as parse_response would emit it."""
    return {
        "pattern_id": pattern_id,
        "is_vulnerability": True,
        "confidence": 0.9,
        "explanation": "confirmed by llm",
        "remediation": "fix it",
        "_file": file,
        "start_line": start_line,
        "end_line": None,
    }


def _lineage_finding(finding_id: str, file: str, start_line: int) -> Finding:
    """Build a finding with an explicit ID for ledger-lineage assertions."""
    return Finding(
        rule_id=finding_id.upper(),
        message=f"static finding {finding_id}",
        finding_id=finding_id,
        severity="MEDIUM",
        confidence=0.9,
        file=file,
        start_line=start_line,
    )


class TestMetaLedgerResponse:
    """Direct contract tests for meta-analysis finding lineage."""

    def test_mixed_batches_distinguish_retained_and_filtered_findings(self) -> None:
        retained = _lineage_finding("retained", "complete.py", 1)
        filtered = _lineage_finding("filtered", "complete.py", 2)
        passed_through = _lineage_finding("passed-through", "failed.py", 3)
        completed_batch = Batch(
            file_path="complete.py",
            content="complete",
            findings=[retained, filtered],
        )
        failed_batch = Batch(
            file_path="failed.py",
            content="failed",
            findings=[passed_through],
        )

        events, status = _meta_ledger_response(
            [completed_batch, failed_batch],
            BatchExecutionResult(
                successful=[(completed_batch, [])],
                failures=[BatchFailure(batch=failed_batch, error_class="TimeoutError")],
            ),
            [retained, passed_through],
        )

        completed, failed = events
        assert completed["outcome"] == "completed"
        assert completed["input_finding_ids"] == ["retained", "filtered"]
        assert completed["emitted_finding_ids"] == ["retained"]
        assert failed["outcome"] == "failed"
        assert failed["input_finding_ids"] == ["passed-through"]
        assert failed["emitted_finding_ids"] == ["passed-through"]
        assert failed["reason_code"] == "llm_batch_failed"
        assert failed["error_class"] == "TimeoutError"
        assert status["status"] == "failed"
        assert status["planned_work"] == [
            {
                "work_id": event["work_id"],
                "path": event["path"],
                "start_line": event["start_line"],
                "end_line": event["end_line"],
            }
            for event in events
        ]

    def test_connection_failure_remains_fatal(self) -> None:
        failed = _lineage_finding("failed", "failed.py", 3)
        failed_batch = Batch(file_path="failed.py", content="failed", findings=[failed])

        events, status = _meta_ledger_response(
            [failed_batch],
            BatchExecutionResult(
                failures=[
                    BatchFailure(
                        batch=failed_batch,
                        error_class="APIConnectionError",
                        reason=LedgerReason.LLM_CONNECTION_RETRIES_EXHAUSTED,
                    )
                ]
            ),
            [failed],
        )

        assert events[0]["outcome"] is LedgerOutcome.FAILED
        assert events[0]["reason_code"] == LedgerReason.LLM_CONNECTION_RETRIES_EXHAUSTED
        assert events[0]["message"] == "LLM connection failed after bounded retries."
        assert status["status"] == "failed"

        completeness, _ = finalize_ledger(
            {
                "components": ["failed.py"],
                "findings": [failed],
                "effective_finding_ids": [failed.finding_id],
                "inspection_ledger": events,
                "analyzer_status_events": [status],
            }
        )

        assert completeness["execution_successful"] is False
        assert completeness["ledger_exceptions"][0]["fatal"] is True

    def test_structured_response_failure_is_nonfatal_and_degraded(self) -> None:
        failed = _lineage_finding("failed", "failed.py", 3)
        failed_batch = Batch(file_path="failed.py", content="failed", findings=[failed])

        events, status = _meta_ledger_response(
            [failed_batch],
            BatchExecutionResult(
                failures=[
                    BatchFailure(
                        batch=failed_batch,
                        error_class="ValidationError",
                        reason=LedgerReason.LLM_STRUCTURED_RESPONSE_INVALID,
                    )
                ]
            ),
            [failed],
        )

        assert events[0]["outcome"] is LedgerOutcome.SKIPPED
        assert events[0]["input_finding_ids"] == [failed.finding_id]
        assert events[0]["emitted_finding_ids"] == [failed.finding_id]
        assert status["status"] == "degraded"

        completeness, _ = finalize_ledger(
            {
                "components": ["failed.py"],
                "findings": [failed],
                "effective_finding_ids": [failed.finding_id],
                "inspection_ledger": events,
                "analyzer_status_events": [status],
            }
        )

        assert completeness["execution_successful"] is True
        assert completeness["is_complete"] is False
        assert completeness["ledger_exceptions"][0]["fatal"] is False

    def test_overlapping_batches_do_not_reaccount_completed_finding(self) -> None:
        shared = _lineage_finding("shared", "complete.py", 1)
        failed_only = _lineage_finding("failed-only", "failed.py", 2)
        completed_batch = Batch(file_path="complete.py", content="complete", findings=[shared])
        failed_batch = Batch(
            file_path="failed.py",
            content="failed",
            findings=[shared, failed_only],
        )

        events, _ = _meta_ledger_response(
            [completed_batch, failed_batch],
            BatchExecutionResult(
                successful=[(completed_batch, [])],
                failures=[BatchFailure(batch=failed_batch, error_class="ProviderError")],
            ),
            [shared, failed_only],
        )

        completed, failed = events
        assert completed["input_finding_ids"] == ["shared"]
        assert completed["emitted_finding_ids"] == ["shared"]
        assert failed["input_finding_ids"] == ["failed-only"]
        assert failed["emitted_finding_ids"] == ["failed-only"]

    def test_fully_accounted_failed_batch_does_not_degrade_meta_status(self) -> None:
        shared = _lineage_finding("shared", "complete.py", 1)
        completed_batch = Batch(file_path="complete.py", content="complete", findings=[shared])
        failed_batch = Batch(file_path="complete.py", content="retry", findings=[shared])

        events, status = _meta_ledger_response(
            [completed_batch, failed_batch],
            BatchExecutionResult(
                successful=[(completed_batch, [])],
                failures=[BatchFailure(batch=failed_batch, error_class="ProviderError")],
            ),
            [shared],
        )

        assert len(events) == 1
        assert events[0]["outcome"] == "completed"
        assert status["status"] == "completed"

    def test_empty_failed_batch_does_not_degrade_meta_status(self) -> None:
        empty_batch = Batch(file_path="empty.py", content="empty", findings=[])

        events, status = _meta_ledger_response(
            [empty_batch],
            BatchExecutionResult(
                failures=[BatchFailure(batch=empty_batch, error_class="ProviderError")]
            ),
            [],
        )

        assert events == []
        assert status["status"] == "completed"

    def test_failed_batch_passes_all_findings_through_when_none_are_retained(self) -> None:
        first = _lineage_finding("first", "failed.py", 1)
        second = _lineage_finding("second", "failed.py", 2)
        failed_batch = Batch(
            file_path="failed.py",
            content="failed",
            findings=[first, second],
        )

        events, status = _meta_ledger_response(
            [failed_batch],
            BatchExecutionResult(
                failures=[BatchFailure(batch=failed_batch, error_class="ConnectionError")]
            ),
            [],
        )

        assert events[0]["input_finding_ids"] == ["first", "second"]
        assert events[0]["emitted_finding_ids"] == ["first", "second"]
        assert status["status"] == "failed"


@patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
class TestMetaAnalyzerPartialBatchFailure:
    def _state(self, findings: list[Finding]) -> dict[str, object]:
        return {
            "findings": findings,
            "use_llm": True,
            "file_cache": {"a.py": "code a", "b.py": "code b"},
            "manifest": {},
            "model_config": {},
        }

    def test_unanalysed_findings_survive_a_failed_batch(self) -> None:
        """Findings whose batch failed are kept (no verdict != rejection)."""
        f_confirmed = Finding(rule_id="R1", message="m", file="a.py", start_line=1)
        f_rejected = Finding(rule_id="R2", message="m", file="a.py", start_line=5)
        f_unseen = Finding(rule_id="R1", message="m", file="b.py", start_line=3)

        batch_a = Batch(file_path="a.py", content="code a", findings=[f_confirmed, f_rejected])
        batch_b = Batch(file_path="b.py", content="code b", findings=[f_unseen])

        # batch_b never returned (timeout/429): only batch_a's verdicts exist,
        # and the LLM confirmed R1 but stayed silent on R2 (= rejection).
        partial_results = [(batch_a, [_confirm("R1", "a.py", 1)])]

        with (
            patch.object(LLMMetaAnalyzer, "get_batches", return_value=[batch_a, batch_b]),
            patch.object(
                LLMMetaAnalyzer,
                "arun_batches",
                new_callable=AsyncMock,
                return_value=partial_results,
            ),
        ):
            result = meta_analyzer(self._state([f_confirmed, f_rejected, f_unseen]))

        filtered = result["findings"]
        kept = {(f.file, f.rule_id) for f in filtered}

        # LLM analysis can enrich findings but cannot remove either returned
        # or unseen deterministic results.
        assert ("a.py", "R1") in kept
        assert ("a.py", "R2") in kept
        assert ("b.py", "R1") in kept
        assert result["effective_finding_ids"] == [
            f_confirmed.finding_id,
            f_rejected.finding_id,
            f_unseen.finding_id,
        ]
        assert result["analyzer_status_events"][0]["status"] == "failed"
        assert "filtered_findings" not in result

        confirmed = next(f for f in filtered if f.file == "a.py")
        assert confirmed.explanation == "confirmed by llm"

    def test_selection_does_not_persist_filtered_findings(self) -> None:
        finding = _lineage_finding("retained", "a.py", 1)
        state = self._state([finding])
        state["use_llm"] = False

        result = meta_analyzer(state)

        assert [returned.finding_id for returned in result["findings"]] == [finding.finding_id]
        assert result["effective_finding_ids"] == [finding.finding_id]
        assert "filtered_findings" not in result

    def test_all_batches_failed_keeps_everything_via_fallback(self) -> None:
        f1 = Finding(rule_id="R1", message="m", file="a.py", start_line=1)
        f2 = Finding(rule_id="R2", message="m", file="b.py", start_line=2)
        batch_a = Batch(file_path="a.py", content="code a", findings=[f1])
        batch_b = Batch(file_path="b.py", content="code b", findings=[f2])

        with (
            patch.object(LLMMetaAnalyzer, "get_batches", return_value=[batch_a, batch_b]),
            patch.object(
                LLMMetaAnalyzer,
                "arun_batches",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = meta_analyzer(self._state([f1, f2]))

        kept = {(f.file, f.rule_id) for f in result["findings"]}
        assert kept == {("a.py", "R1"), ("b.py", "R2")}
        assert "filtered_findings" not in result

    def test_reconstructed_partial_result_uses_canonical_batch_and_finding_ids(self) -> None:
        rejected = _lineage_finding("rejected", "a.py", 1)
        unseen = _lineage_finding("unseen", "b.py", 2)
        submitted_a = Batch(file_path="a.py", content="code a", findings=[rejected])
        submitted_b = Batch(file_path="b.py", content="code b", findings=[unseen])
        returned_a = Batch(
            file_path="a.py",
            content="code a",
            findings=[_lineage_finding("rejected", "a.py", 1)],
        )

        with (
            patch.object(LLMMetaAnalyzer, "get_batches", return_value=[submitted_a, submitted_b]),
            patch.object(
                LLMMetaAnalyzer,
                "arun_batches",
                new_callable=AsyncMock,
                return_value=[(returned_a, [])],
            ),
        ):
            result = meta_analyzer(self._state([rejected, unseen]))

        assert result["effective_finding_ids"] == [rejected.finding_id, unseen.finding_id]
        assert [finding.finding_id for finding in result["findings"]] == [
            rejected.finding_id,
            unseen.finding_id,
        ]
        assert result["analyzer_status_events"][0]["status"] == "failed"

    def test_duplicate_return_does_not_account_for_a_missing_batch(self) -> None:
        confirmed = _lineage_finding("confirmed", "a.py", 1)
        unseen = _lineage_finding("unseen", "b.py", 2)
        submitted_a = Batch(file_path="a.py", content="code a", findings=[confirmed])
        submitted_b = Batch(file_path="b.py", content="code b", findings=[unseen])
        returned_a = Batch(
            file_path="a.py",
            content="code a",
            findings=[_lineage_finding("confirmed", "a.py", 1)],
        )

        # A malformed/custom executor can return the same batch twice while
        # omitting another submitted batch. The missing batch must still use
        # fallback filtering; matching result-list lengths is insufficient.
        with (
            patch.object(LLMMetaAnalyzer, "get_batches", return_value=[submitted_a, submitted_b]),
            patch.object(
                LLMMetaAnalyzer,
                "arun_batches",
                new_callable=AsyncMock,
                return_value=[
                    (returned_a, [_confirm("CONFIRMED", "a.py", 1)]),
                    (returned_a, [_confirm("CONFIRMED", "a.py", 1)]),
                ],
            ),
        ):
            result = meta_analyzer(self._state([confirmed, unseen]))

        assert [finding.finding_id for finding in result["findings"]] == [
            confirmed.finding_id,
            unseen.finding_id,
        ]
        assert result["analyzer_status_events"][0]["status"] == "failed"

    def test_empty_meta_batches_are_not_submitted(self) -> None:
        finding = _lineage_finding("retained", "a.py", 1)
        empty_batch = Batch(file_path="a.py", content="context", findings=[])
        finding_batch = Batch(file_path="a.py", content="finding", findings=[finding])

        with (
            patch.object(
                LLMMetaAnalyzer,
                "get_batches",
                return_value=[empty_batch, finding_batch],
            ),
            patch.object(
                LLMMetaAnalyzer,
                "arun_batches",
                new_callable=AsyncMock,
                return_value=[(finding_batch, [_confirm("RETAINED", "a.py", 1)])],
            ) as arun_batches,
        ):
            result = meta_analyzer(self._state([finding]))

        assert arun_batches.await_args.args[0] == [finding_batch]
        assert result["effective_finding_ids"] == [finding.finding_id]

    def test_no_failures_still_preserve_deterministic_findings(self) -> None:
        """Even complete LLM coverage cannot suppress deterministic findings."""
        f_confirmed = Finding(rule_id="R1", message="m", file="a.py", start_line=1)
        f_rejected = Finding(rule_id="R2", message="m", file="b.py", start_line=2)
        batch_a = Batch(file_path="a.py", content="code a", findings=[f_confirmed])
        batch_b = Batch(file_path="b.py", content="code b", findings=[f_rejected])

        full_results = [
            (batch_a, [_confirm("R1", "a.py", 1)]),
            (batch_b, []),
        ]

        with (
            patch.object(LLMMetaAnalyzer, "get_batches", return_value=[batch_a, batch_b]),
            patch.object(
                LLMMetaAnalyzer,
                "arun_batches",
                new_callable=AsyncMock,
                return_value=full_results,
            ),
        ):
            result = meta_analyzer(self._state([f_confirmed, f_rejected]))

        kept = {(f.file, f.rule_id) for f in result["findings"]}
        assert kept == {("a.py", "R1"), ("b.py", "R2")}

    def test_effective_ids_follow_final_finding_order(self) -> None:
        a_pattern = _lineage_finding("pattern-a", "a.py", 1)
        b_pattern = _lineage_finding("pattern-b", "b.py", 2)
        a_entity = _lineage_finding("entity-a", "a.py", 3)
        b_entity = _lineage_finding("entity-b", "b.py", 4)
        batch_a = Batch(file_path="a.py", content="code a", findings=[a_pattern, a_entity])
        batch_b = Batch(file_path="b.py", content="code b", findings=[b_pattern, b_entity])
        findings = [a_pattern, b_pattern, a_entity, b_entity]

        with (
            patch.object(LLMMetaAnalyzer, "get_batches", return_value=[batch_a, batch_b]),
            patch.object(
                LLMMetaAnalyzer,
                "arun_batches",
                new_callable=AsyncMock,
                return_value=[
                    (batch_a, [_confirm("PATTERN-A", "a.py", 1), _confirm("ENTITY-A", "a.py", 3)]),
                    (batch_b, [_confirm("PATTERN-B", "b.py", 2), _confirm("ENTITY-B", "b.py", 4)]),
                ],
            ),
        ):
            result = meta_analyzer(self._state(findings))

        assert result["effective_finding_ids"] == [
            "pattern-a",
            "pattern-b",
            "entity-a",
            "entity-b",
        ]
        completeness, effective_ids = finalize_ledger(
            {
                "components": ["a.py", "b.py"],
                "findings": result["findings"],
                "effective_finding_ids": result["effective_finding_ids"],
                "inspection_ledger": result["inspection_ledger"],
                "analyzer_status_events": result["analyzer_status_events"],
            }
        )

        assert effective_ids == ["pattern-a", "pattern-b", "entity-a", "entity-b"]
        assert completeness["execution_successful"] is True
        assert completeness["ledger_exceptions"] == []


def test_local_only_high_finding_never_constructs_llm_analyzer() -> None:
    finding = Finding(
        rule_id="SC9",
        message="concealed executable",
        finding_id="local-finding",
        severity="HIGH",
        file=".hidden.docx!/payload.sh",
        tags=[],
        evidence={"outer_path": ".hidden.docx"},
    )
    state = {
        "findings": [finding],
        "file_cache": {"SKILL.md": "sentinel-visible-content"},
        "component_metadata": [{"path": ".hidden.docx!/payload.sh", "local_only": True}],
        "use_llm": True,
    }

    with patch("skillspector.nodes.meta_analyzer.LLMMetaAnalyzer") as analyzer_cls:
        result = meta_analyzer(state)

    analyzer_cls.assert_not_called()
    assert result["effective_finding_ids"] == ["local-finding"]
    assert result["findings"][0].severity == "HIGH"
    assert result["inspection_ledger"][0]["path"] == ".hidden.docx!/payload.sh"
    assert result["inspection_ledger"][0]["emitted_finding_ids"] == ["local-finding"]


@patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
def test_provider_receives_only_cache_safe_non_local_findings() -> None:
    safe = _lineage_finding("safe", "safe.py", 1)
    missing = _lineage_finding("missing", "missing.py", 2)
    metadata_local = _lineage_finding("metadata-local", "metadata.py", 3)
    tagged_local = _lineage_finding("tagged-local", "tagged.py", 4)
    tagged_local.tags.append("local-only")
    evidence_local = _lineage_finding("evidence-local", "evidence.py", 5)
    evidence_local.evidence["local_only"] = True
    findings = [safe, missing, metadata_local, tagged_local, evidence_local]
    safe_batch = Batch(file_path="safe.py", content="safe", findings=[safe])
    llm_file_cache = {
        "safe.py": "safe",
        "metadata.py": "must stay local",
        "tagged.py": "must stay local",
        "evidence.py": "must stay local",
    }
    state: SkillspectorState = {
        "findings": findings,
        "use_llm": True,
        "file_cache": {**llm_file_cache, "missing.py": "not provider safe"},
        "llm_file_cache": llm_file_cache,
        "component_metadata": [{"path": "metadata.py", "local_only": True}],
        "manifest": {},
        "model_config": {},
    }

    with (
        patch.object(LLMMetaAnalyzer, "get_batches", return_value=[safe_batch]) as get_batches,
        patch.object(
            LLMMetaAnalyzer,
            "arun_batches",
            new_callable=AsyncMock,
            return_value=[(safe_batch, [])],
        ),
    ):
        result = meta_analyzer(state)

    files, provider_cache, submitted_findings = get_batches.call_args.args
    assert files == ["safe.py"]
    assert provider_cache is llm_file_cache
    assert submitted_findings == [safe]
    assert {finding.finding_id for finding in result["findings"]} == {
        finding.finding_id for finding in findings
    }
    assert result["effective_finding_ids"] == [finding.finding_id for finding in result["findings"]]
    assert {event["path"] for event in result["inspection_ledger"]} == {
        "safe.py",
        "missing.py",
        "metadata.py",
        "tagged.py",
        "evidence.py",
    }


@patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
def test_local_only_event_survives_provider_failure() -> None:
    eligible = _lineage_finding("eligible", "safe.py", 1)
    local = _lineage_finding("local", "local.py", 2)
    local.tags.append("local-only")
    batch = Batch(file_path="safe.py", content="safe", findings=[eligible])
    state: SkillspectorState = {
        "findings": [eligible, local],
        "use_llm": True,
        "llm_file_cache": {"safe.py": "safe", "local.py": "must stay local"},
        "manifest": {},
        "model_config": {},
    }

    with (
        patch.object(LLMMetaAnalyzer, "get_batches", return_value=[batch]),
        patch.object(
            LLMMetaAnalyzer,
            "arun_batches",
            new_callable=AsyncMock,
            side_effect=RuntimeError("provider unavailable"),
        ),
    ):
        result = meta_analyzer(state)

    assert [finding.finding_id for finding in result["findings"]] == ["eligible", "local"]
    assert result["effective_finding_ids"] == ["eligible", "local"]
    assert [event["path"] for event in result["inspection_ledger"]] == ["local.py"]
    assert result["inspection_ledger"][0]["emitted_finding_ids"] == ["local"]
    assert result["analyzer_status_events"][0]["status"] == "unavailable"


# ---------------------------------------------------------------------------
# LLM-call telemetry + fail-closed construction (drives the report's
# degradation signal).
# ---------------------------------------------------------------------------


def _degr_finding(rule_id: str = "P1", severity: str = "HIGH") -> Finding:
    return Finding(
        rule_id=rule_id,
        message="test",
        severity=severity,
        confidence=0.8,
        file="SKILL.md",
        start_line=1,
    )


def _degr_state(**overrides: object) -> SkillspectorState:
    state: SkillspectorState = {
        "findings": [_degr_finding()],
        "use_llm": True,
        "file_cache": {"SKILL.md": "# Skill"},
        "manifest": {},
        "model_config": {},
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def test_records_ok_true_on_success() -> None:
    finding = _degr_finding()
    batch = Batch(file_path="SKILL.md", content="# Skill", findings=[finding])
    with (
        patch("skillspector.llm_analyzer_base.get_chat_model", return_value=MagicMock()),
        patch.object(LLMMetaAnalyzer, "get_batches", return_value=[batch]),
        patch(
            "skillspector.nodes.meta_analyzer.LLMMetaAnalyzer.arun_batches",
            new_callable=AsyncMock,
            return_value=[(batch, [])],
        ),
    ):
        result = meta_analyzer(_degr_state(findings=[finding]))
    assert result["llm_call_log"] == [{"node": "meta_analyzer", "ok": True, "error": None}]
    assert "filtered_findings" not in result


def test_construction_failure_is_caught_not_raised() -> None:
    """Regression: the chat model is constructed INSIDE the try, so a construction
    failure degrades (records ok=False, preserves findings) instead of crashing
    the whole graph."""
    with patch(
        "skillspector.llm_analyzer_base.get_chat_model",
        side_effect=RuntimeError("provider construction failed"),
    ):
        result = meta_analyzer(_degr_state())  # must not raise
    # Findings are preserved via the fallback path...
    assert len(result["findings"]) == 1
    assert "filtered_findings" not in result
    # ...and the failure is recorded so the report can flag degradation.
    log = result["llm_call_log"]
    assert log[0]["node"] == "meta_analyzer"
    assert log[0]["ok"] is False
    assert "provider construction failed" in log[0]["error"]
    status = result["analyzer_status_events"][0]
    assert status["status"] == "unavailable"
    assert "reason_code" not in status


def test_credential_error_propagates_instead_of_being_labelled_unavailable() -> None:
    """Only actual credential failures propagate; provider failures have no guessed cause."""
    with patch(
        "skillspector.llm_analyzer_base.get_chat_model",
        side_effect=ValueError("No LLM API key configured."),
    ):
        try:
            meta_analyzer(_degr_state())
        except ValueError as error:
            assert "API key" in str(error)
        else:
            raise AssertionError("credential failure must not be reported as unavailable")


def test_use_llm_false_records_nothing() -> None:
    result = meta_analyzer(_degr_state(use_llm=False))
    assert "llm_call_log" not in result
    assert "filtered_findings" not in result


def test_no_findings_records_nothing() -> None:
    result = meta_analyzer(_degr_state(findings=[]))
    assert "llm_call_log" not in result
    assert "filtered_findings" not in result


def test_no_findings_projects_incomplete_visible_artifact_as_failed_meta_work() -> None:
    """An omitted provider artifact remains failed planned work even without findings."""
    result = meta_analyzer(
        _degr_state(
            findings=[],
            llm_file_cache={},
            llm_redaction_incomplete_paths=["pip.conf"],
        )
    )

    assert result["findings"] == []
    assert result["effective_finding_ids"] == []
    assert result["llm_call_log"] == [{"node": "meta_analyzer", "ok": False, "error": None}]
    assert len(result["inspection_ledger"]) == 1
    event = result["inspection_ledger"][0]
    assert event["path"] == "pip.conf"
    assert event["outcome"] == LedgerOutcome.FAILED
    assert event["reason_code"] == LedgerReason.LLM_BATCH_FAILED
    assert event["input_finding_ids"] == []
    assert event["emitted_finding_ids"] == []
    [status] = result["analyzer_status_events"]
    assert status["status"] == "failed"
    assert [work["path"] for work in status["planned_work"]] == ["pip.conf"]
