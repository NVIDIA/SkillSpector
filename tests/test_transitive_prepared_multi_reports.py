# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance coverage for task 4.1's prepared transitive report boundary."""

from __future__ import annotations

import importlib
import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
from markdown_it import MarkdownIt
from typer.testing import CliRunner

from skillspector import __version__, cli
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason, ledger_event
from skillspector.models import OCCURRENCE_FINDING_ID_KEY, Finding
from skillspector.nodes.report import (
    PreparedReport,
    report_rendering_deferred,
    serialize_report,
)
from skillspector.sarif_models import validate_sarif_report
from skillspector.state import SkillspectorState, llm_call_record
from skillspector.suppression import Baseline, SuppressionRule, finding_fingerprint

report_module = importlib.import_module("skillspector.nodes.report")
FORMATS = ("terminal", "json", "markdown", "sarif")
TARGET = "https://github.com/acceptance/prepared-child"
MEMBER = "outer.zip!/inner.zip!/run.py"
ARCHIVE = {
    "outer_path": "outer.zip",
    "nested_path": "inner.zip!/run.py",
    "container_type": "zip",
    "container_ancestry": ["zip", "zip"],
    "container_depth": 2,
}


def _unexpected_work(*args: object, **kwargs: object) -> None:
    pytest.fail("Rendering repeated analysis, preparation, suppression, or provider work")


@pytest.fixture
def transitive_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep real graph invocation/orchestration/report nodes; control analysis only."""
    root = tmp_path / "skill"
    root.mkdir()
    (root / "SKILL.md").write_text(f"# Root\n{TARGET}\n", encoding="utf-8")
    child_root = tmp_path / "resolved-child"
    child_root.mkdir()
    (child_root / "outer.zip").write_bytes(b"controlled archive source")
    calls: list[str] = []
    preparations: list[tuple[dict[str, object], PreparedReport]] = []
    provider = Mock(side_effect=_unexpected_work)
    monkeypatch.setattr(report_module, "is_llm_available", provider)
    clock = Mock(wraps=datetime)
    clock.now.return_value = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(report_module, "datetime", clock)
    # Avoid TTY-dependent streaming so every graph state can be inspected.
    monkeypatch.setattr(cli.err_console, "_force_terminal", False)
    real_prepare = report_module.prepare_report
    real_serialize = report_module.serialize_report

    def capture_prepare(state: SkillspectorState) -> PreparedReport:
        prepared = real_prepare(state)
        preparations.append((deepcopy(dict(state)), prepared))
        return prepared

    monkeypatch.setattr(report_module, "prepare_report", capture_prepare)

    def install(
        *, show_suppressed: bool = False, incomplete: bool = False, llm_enabled: bool = False
    ) -> None:
        def invoke(state: dict[str, object], config=None) -> dict[str, object]:
            assert state["defer_report_rendering"] is True
            assert state["transitive_traversal_state"] is not None
            source = str(state["input_path"])
            calls.append(source)
            child = source == TARGET
            assert source in {str(root), TARGET}
            if child:
                # The root baseline must not suppress an unscoped child.
                assert state.get("baseline") is None
            path = MEMBER if child else "SKILL.md"
            first = Finding(
                rule_id="P1",
                finding_id="shared-first",
                message="Instruction override",
                severity="HIGH",
                confidence=1.0,
                file=path,
                start_line=2,
                end_line=2,
                start_column=0,
                end_column=8,
                occurrences=[
                    {
                        OCCURRENCE_FINDING_ID_KEY: "shared-first",
                        "file": path,
                        "start_line": 2,
                        "end_line": 2,
                        "start_column": 0,
                        "end_column": 8,
                    }
                ],
            )
            findings = [first]
            if child:
                findings.append(
                    Finding(
                        rule_id="P1",
                        finding_id="shared-second",
                        message="Instruction override",
                        severity="HIGH",
                        confidence=1.0,
                        file=path,
                        start_line=4,
                        end_line=4,
                        start_column=3,
                        end_column=11,
                        occurrences=[
                            {
                                OCCURRENCE_FINDING_ID_KEY: "shared-second",
                                "file": path,
                                "start_line": 4,
                                "end_line": 4,
                                "start_column": 3,
                                "end_column": 11,
                            }
                        ],
                    )
                )
            findings.append(
                Finding(
                    rule_id="BH2",
                    finding_id="shared-suppressed",
                    message="Reviewed blocked behavior",
                    severity="CRITICAL",
                    confidence=1.0,
                    file=path,
                    start_line=6,
                )
            )
            events = [
                dict(
                    ledger_event(
                        outcome=LedgerOutcome.COMPLETED,
                        phase="static_analysis",
                        path=path,
                        analyzer_id="static_patterns",
                        emitted_finding_ids=[f.finding_id for f in findings],
                    )
                )
            ]
            components = [path]
            metadata = [
                {
                    "path": path,
                    "type": "python" if child else "markdown",
                    "lines": 8,
                    "size_bytes": 64,
                    "executable": child,
                    **(ARCHIVE if child else {}),
                }
            ]
            if incomplete and child:
                broken = "outer.zip!/broken.zip"
                components.append(broken)
                events.append(
                    dict(
                        ledger_event(
                            outcome=LedgerOutcome.PARTIAL,
                            phase="nested_artifact_inspection",
                            path=broken,
                            reason=LedgerReason.ARCHIVE_TRUNCATED,
                        )
                    )
                )
                metadata.append({"path": broken, "type": "archive", "executable": False})
            cache = {path: "print('child')\n" if child else f"# Root\n{TARGET}\n"}
            usage = {
                "node": "meta_analyzer",
                "request_kind": "structured_output",
                "provider": "anthropic",
                "model": "fixture-model",
                "model_source": "provider_response",
                "usage_source": "provider_response",
                "prompt_tokens": 10 if child else 20,
                "completion_tokens": 2,
                "total_tokens": 12 if child else 22,
                "secret": "private-usage-sentinel",
            }
            canonical = {
                **state,
                "findings": findings,
                # Simulate a reduced meta view, not the canonical evidence.
                "filtered_findings": [first],
                "effective_finding_ids": [f.finding_id for f in findings],
                "components": components,
                "skill_path": str(child_root if child else root),
                "component_metadata": metadata,
                "file_cache": cache,
                "local_file_cache": cache,
                "inspection_ledger": events,
                "has_executable_scripts": child,
                "manifest": {"name": "prepared-child" if child else "prepared-root"},
                "use_llm": llm_enabled,
                "llm_requested": llm_enabled,
                "llm_call_log": [llm_call_record("meta_analyzer", ok=True)],
                "inference_usage": [usage],
                "llm_provenance": {
                    "provider": {
                        "configured_adapter": "anthropic",
                        "resolved_adapter": "anthropic",
                    },
                    "analyzers": [{"analyzer_id": "meta_analyzer", "model": "fixture-model"}],
                    "endpoint": "https://private-endpoint-sentinel.invalid",
                },
                "structured_summaries": [
                    {
                        "id": "SSR-child" if child else "SSR-root",
                        "message": "Structured analysis retained",
                        "file": path,
                        "protocol": "AISOP V1",
                    }
                ],
            }
            if not child:
                child_digest = cli._source_content_digest({}, {MEMBER: "print('child')\n"})
                child_identity = cli._source_identity(TARGET, child_digest)
                accepted_child = replace(
                    findings[-1],
                    file=MEMBER,
                    source_url=TARGET,
                    source_identity=child_identity,
                    source_digest=child_digest,
                    transitive_depth=1,
                )
                canonical["baseline"] = Baseline(
                    rules=[SuppressionRule(rule_id="BH2", reason="reviewed")],
                    fingerprints={
                        finding_fingerprint(
                            accepted_child,
                            file_content="print('child')\n",
                            scanner_version=__version__,
                        ): "reviewed"
                    },
                    scanner_version=__version__,
                )
                canonical["show_suppressed"] = show_suppressed
            result = report_module.report(cast(SkillspectorState, canonical))
            assert isinstance(result["prepared_report"], PreparedReport)
            assert "report_body" not in result
            assert "sarif_report" not in result
            return {**canonical, **result}

        monkeypatch.setattr(cli.graph, "invoke", invoke)

    return root, calls, preparations, provider, install, real_serialize


def _collect(root: Path, primary: str, *, no_llm: bool = True) -> dict[str, object]:
    with report_rendering_deferred():
        return cli._scan_skill(
            input_path=str(root),
            format=cli.FormatChoice(primary),
            no_llm=no_llm,
            baseline=None,
            yara_rules_dir=None,
            verbose=False,
            show_suppressed=False,
            transitive_enabled=True,
            transitive_depth=1,
            transitive_allow_prefix=(),
            transitive_deny_prefix=(),
        )


def _assert_canonical(prepared: PreparedReport, *, incomplete: bool, show_suppressed: bool) -> None:
    active = prepared.public_fields["active_findings"]
    assert len(active) == 3
    assert {f.rule_id for f in active} == {"P1"}
    assert len({f.finding_id for f in active}) == 3
    assert len(prepared.display_findings) == 3
    children = [f for f in prepared.display_findings if f.source_url]
    assert len(children) == 2
    assert {f.source_url for f in children} == {TARGET}
    identity = children[0].source_identity
    digest = children[0].source_digest
    assert identity and digest and digest.startswith("sha256:") and len(digest) == 71
    assert {f.source_identity for f in children} == {identity}
    assert {f.source_digest for f in children} == {digest}
    assert {f.transitive_depth for f in children} == {1}
    assert [(f.start_line, f.start_column, f.end_column) for f in children] == [
        (2, 0, 8),
        (4, 3, 11),
    ]
    assert len(prepared.suppressed) == 2
    assert {s.finding.rule_id for s in prepared.suppressed} == {"BH2"}
    assert {s.reason for s in prepared.suppressed} == {"reviewed"}
    assert prepared.show_suppressed is show_suppressed
    # Root: 25; executable child occurrences: 25*1.3*(0.5+0.25), truncated to int.
    assert prepared.risk_score == 49
    assert prepared.analysis_completeness["is_complete"] is not incomplete
    assert prepared.execution_successful is True
    assert prepared.report_completeness["is_complete"] is True
    assert prepared.transitive_targets_scanned == 1
    member = next(c for c in prepared.component_metadata if c["path"] == MEMBER)
    assert {key: member[key] for key in ARCHIVE} == ARCHIVE
    assert member["source_identity"] == identity
    assert member["source_digest"] == digest
    assert member["source_url"] == TARGET
    assert len(prepared.inference_usage) == len(prepared.llm_call_log) == 2
    assert [record["total_tokens"] for record in prepared.inference_usage] == [22, 12]
    child_call = next(record for record in prepared.llm_call_log if record.get("source_url"))
    assert child_call["source_identity"] == identity
    assert child_call["source_digest"] == digest
    assert child_call["source_url"] == TARGET
    if incomplete:
        exception = next(
            e
            for e in prepared.analysis_completeness["ledger_exceptions"]
            if e["reason_code"] == "archive_truncated"
        )
        assert exception["source_identity"] == identity
        assert exception["path"].startswith(identity + "/")


def _assert_projection(rendered: dict[str, object], fmt: str, prepared: PreparedReport) -> None:
    assert rendered["risk_score"] == prepared.risk_score
    assert rendered["analysis_completeness"] == prepared.analysis_completeness
    assert rendered["report_completeness"]["is_complete"] is True
    body = str(rendered["report_body"])
    assert "private-usage-sentinel" not in body
    assert "private-endpoint-sentinel" not in body
    if fmt == "json":
        data = json.loads(body)
        assert len(data["issues"]) == 3
        assert {issue["id"] for issue in data["issues"]} == {"P1"}
        assert data["suppressed_count"] == len(data["suppressed"]) == 2
        assert data["analysis_completeness"] == prepared.analysis_completeness
        assert data["risk_assessment"]["score"] == prepared.risk_score
        child_issues = [issue for issue in data["issues"] if issue.get("source_url")]
        assert [issue["location"] for issue in child_issues] == [
            f.to_dict()["location"] for f in prepared.display_findings if f.source_url
        ]
        for issue in child_issues:
            assert issue["source_identity"] == prepared.display_findings[-1].source_identity
            assert issue["source_digest"] == prepared.display_findings[-1].source_digest
            assert issue["transitive_depth"] == 1
        usage = data["metadata"]["inference_usage"]
        assert len(usage) == 2
        assert sum(record["total_tokens"] for record in usage) == 34
        provenance = data["metadata"]["llm_provenance"]["provider"]
        assert provenance["configured_adapter"] == provenance["resolved_adapter"] == "anthropic"
        assert provenance["effective_adapter"] == (
            "anthropic" if prepared.use_llm else "not_applicable"
        )
        member = next(c for c in data["components"] if c["path"] == MEMBER)
        assert {key: member[key] for key in ARCHIVE} == ARCHIVE
        assert member["source_identity"] == child_issues[0]["source_identity"]
        assert data["structured_summaries"][0]["id"] == "SSR-root"
    elif fmt == "sarif":
        data = json.loads(body)
        validate_sarif_report(data)
        run = data["runs"][0]
        active = [r for r in run["results"] if not r.get("suppressions")]
        suppressed = [r for r in run["results"] if r.get("suppressions")]
        assert len(active) == 3
        assert len(suppressed) == 2
        assert {r["ruleId"] for r in active} == {"P1"}
        assert all(r["suppressions"][0]["justification"] == "reviewed" for r in suppressed)
        children = [r for r in active if r["properties"].get("sourceUrl") == TARGET]
        assert len(children) == 2
        assert [r["locations"][0]["physicalLocation"]["region"] for r in children] == [
            {"startLine": 2, "endLine": 2, "startColumn": 1, "endColumn": 9},
            {"startLine": 4, "endLine": 4, "startColumn": 4, "endColumn": 12},
        ]
        identity = prepared.display_findings[-1].source_identity
        assert all(
            r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"].startswith(
                identity + "/"
            )
            for r in children
        )
        invocation = run["invocations"][0]
        assert invocation["executionSuccessful"] is True
        assert (
            invocation["properties"]["analysisCompleteness"]["isComplete"]
            == (prepared.analysis_completeness["is_complete"])
        )
        assert invocation["properties"]["reportCompleteness"]["is_complete"] is True
    else:
        visible = MarkdownIt().enable("table").render(body) if fmt == "markdown" else body
        assert "Instruction override" in visible
        assert "SSR-root" in visible
        assert "run.py" in visible
        assert ("BH2" in visible) is prepared.show_suppressed


@pytest.mark.parametrize("primary", FORMATS)
@pytest.mark.parametrize("show_suppressed", [False, True])
@pytest.mark.parametrize("incomplete", [False, True])
def test_transitive_prepares_root_child_and_final_canonical_view_once(
    transitive_case,
    monkeypatch: pytest.MonkeyPatch,
    primary: str,
    show_suppressed: bool,
    incomplete: bool,
) -> None:
    root, calls, preparations, provider, install, real_serialize = transitive_case
    install(show_suppressed=show_suppressed, incomplete=incomplete)
    # Preparation, including the final merge, must not invoke ANY serializer.
    monkeypatch.setattr(report_module, "serialize_report", _unexpected_work)
    result = _collect(root, primary)
    assert calls == [str(root), TARGET]
    assert len(preparations) == 3
    prepared = result["prepared_report"]
    assert prepared is preparations[-1][1]
    assert prepared is not preparations[0][1]
    assert prepared is not preparations[1][1]
    assert "report_body" not in result and "sarif_report" not in result
    _assert_canonical(prepared, incomplete=incomplete, show_suppressed=show_suppressed)
    before = deepcopy(prepared)
    for name in ("prepare_report", "partition_findings", "_compute_risk_score"):
        monkeypatch.setattr(report_module, name, _unexpected_work)
    monkeypatch.setattr(cli.graph, "invoke", _unexpected_work)
    monkeypatch.setattr(cli, "_run_graph_scan_for_source", _unexpected_work)
    for fmt in (*FORMATS, *reversed(FORMATS), primary):
        _assert_projection(real_serialize(prepared, fmt), fmt, prepared)
    assert prepared == before
    provider.assert_not_called()
    assert calls == [str(root), TARGET]


@pytest.mark.parametrize("primary", FORMATS)
def test_transitive_cli_emits_all_formats_from_the_final_prepared_view(
    transitive_case, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, primary: str
) -> None:
    root, calls, preparations, provider, install, real_serialize = transitive_case
    install(show_suppressed=True, incomplete=True)
    attempts: list[tuple[str, PreparedReport]] = []

    def serialize(prepared: PreparedReport, fmt: str) -> dict[str, object]:
        assert calls == [str(root), TARGET]
        assert len(preparations) == 3
        assert prepared is preparations[-1][1]
        attempts.append((fmt, prepared))
        return real_serialize(prepared, fmt)

    monkeypatch.setattr(cli, "serialize_report", serialize)
    monkeypatch.setattr(report_module, "serialize_report", serialize)
    destinations = {fmt: tmp_path / f"extra.{fmt}" for fmt in FORMATS}
    duplicate = tmp_path / "again.json"
    args = ["scan", str(root), "--no-llm", "--transitive", "--format", primary]
    for fmt, path in destinations.items():
        args.extend(["--report", f"{fmt}={path}"])
    args.extend(["--report", f"json={duplicate}", "--fail-on-incomplete"])
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 1, result.output
    assert [fmt for fmt, _ in attempts] == [primary, *FORMATS, "json"]
    prepared = preparations[-1][1]
    _assert_canonical(prepared, incomplete=True, show_suppressed=True)
    for fmt, path in destinations.items():
        assert path.is_file(), result.output
        rendered = real_serialize(prepared, fmt)
        assert path.read_text(encoding="utf-8") == rendered["report_body"]
        _assert_projection(rendered, fmt, prepared)
    assert json.loads(duplicate.read_text()) == json.loads(destinations["json"].read_text())
    if primary in {"json", "sarif"}:
        json.loads(result.stdout)  # Additional diagnostics must not pollute primary stdout.
    provider.assert_not_called()
    assert calls == [str(root), TARGET]


def test_transitive_sarif_failure_leaves_final_snapshot_renderable(
    transitive_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, calls, preparations, provider, install, real_serialize = transitive_case
    install()
    result = _collect(root, "json")
    prepared = result["prepared_report"]
    before = deepcopy(prepared)
    monkeypatch.setattr(
        report_module, "_build_sarif", Mock(side_effect=RuntimeError("SARIF fixture failure"))
    )
    with pytest.raises(RuntimeError, match="SARIF fixture failure"):
        serialize_report(prepared, "sarif")
    for fmt in ("json", "markdown", "terminal", "json"):
        _assert_projection(real_serialize(prepared, fmt), fmt, prepared)
    assert prepared == before
    assert len(preparations) == 3
    assert calls == [str(root), TARGET]
    provider.assert_not_called()


def test_transitive_provider_resolution_is_not_repeated_for_additional_formats(
    transitive_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, calls, preparations, provider, install, real_serialize = transitive_case
    provider.side_effect = None
    provider.return_value = (True, None)
    install(llm_enabled=True)
    result = _collect(root, "json", no_llm=False)
    prepared = result["prepared_report"]
    assert prepared.use_llm is prepared.provider_available is True
    assert provider.call_count == len(preparations) == 3
    before = deepcopy(prepared)
    provider.side_effect = _unexpected_work
    monkeypatch.setattr(cli.graph, "invoke", _unexpected_work)
    monkeypatch.setattr(report_module, "prepare_report", _unexpected_work)
    for fmt in (*FORMATS, "json", "sarif"):
        _assert_projection(real_serialize(prepared, fmt), fmt, prepared)
    assert prepared == before
    assert provider.call_count == 3
    assert calls == [str(root), TARGET]


def test_transitive_final_view_retains_child_structured_summaries(transitive_case) -> None:
    """The design requires full canonical inputs, not only the root summary."""
    root, calls, preparations, provider, install, real_serialize = transitive_case
    install()
    prepared = _collect(root, "json")["prepared_report"]
    assert {s["id"] for s in preparations[1][1].structured_summaries} == {"SSR-child"}
    assert {s["id"] for s in prepared.structured_summaries} == {"SSR-root", "SSR-child"}
    child_summary = next(s for s in prepared.structured_summaries if s["id"] == "SSR-child")
    assert child_summary["source_url"] == TARGET
    assert child_summary["file"].startswith(child_summary["source_identity"] + "/")
    assert child_summary["file"].endswith("outer.zip!/inner.zip!/run.py")
    assert child_summary["source_digest"].startswith("sha256:")
    assert child_summary["transitive_depth"] == 1
    payload = json.loads(real_serialize(prepared, "json")["report_body"])
    assert {s["id"] for s in payload["structured_summaries"]} == {"SSR-root", "SSR-child"}
    for fmt in FORMATS:
        body = str(real_serialize(prepared, fmt)["report_body"])
        visible = MarkdownIt().enable("table").render(body) if fmt == "markdown" else body
        assert "SSR-root" in visible
        assert "SSR-child" in visible
    assert calls == [str(root), TARGET]
    provider.assert_not_called()
