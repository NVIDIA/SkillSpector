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

"""Report node for Skillspector workflow.

Generates SARIF, computes risk score, and produces report_body in terminal/json/markdown/sarif
based on state["output_format"]. Single place for formatting (CLI and REST API reuse).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from io import StringIO
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from skillspector import __version__ as skillspector_version
from skillspector.llm_utils import is_llm_available
from skillspector.logging_config import get_logger
from skillspector.models import Finding
from skillspector.nodes.deduplicate import deduplicate
from skillspector.sarif_models import (
    SARIF_SCHEMA_URI,
    SarifArtifactLocation,
    SarifDriver,
    SarifLocation,
    SarifLog,
    SarifMessage,
    SarifPhysicalLocation,
    SarifRegion,
    SarifReportingDescriptor,
    SarifResult,
    SarifRun,
    SarifSuppression,
    SarifTool,
)
from skillspector.state import SkillspectorState
from skillspector.suppression import Baseline, SuppressedFinding, partition_findings

logger = get_logger(__name__)

# Risk bands (v1 alignment)
_RISK_SEVERITY_BANDS = [(81, "CRITICAL"), (51, "HIGH"), (21, "MEDIUM"), (0, "LOW")]
_RISK_RECOMMENDATION: dict[str, str] = {
    "LOW": "SAFE",
    "MEDIUM": "CAUTION",
    "HIGH": "DO_NOT_INSTALL",
    "CRITICAL": "DO_NOT_INSTALL",
}


def _severity_to_sarif_level(severity: str) -> Literal["error", "warning", "note"]:
    """Map Finding.severity to SARIF result level."""
    return {
        "CRITICAL": "error",
        "HIGH": "error",
        "MEDIUM": "warning",
        "LOW": "note",
    }.get(severity.upper(), "note")  # type: ignore[return-value]


_SEVERITY_POINTS: dict[str, int] = {
    "CRITICAL": 50,
    "HIGH": 25,
    "MEDIUM": 10,
    "LOW": 5,
}

_MAX_OCCURRENCES_PER_RULE = 3
_DIMINISHING_WEIGHTS = (1.0, 0.5, 0.25)


def _compute_risk_score(
    findings: list[Finding], has_executable_scripts: bool
) -> tuple[int, str, str]:
    """
    Compute risk score (0-100), severity band, and recommendation.

    Scoring uses per-rule diminishing returns: the first occurrence of a rule_id
    contributes full points, the second contributes half, and the third contributes
    a quarter. Occurrences beyond the third are ignored for scoring purposes.
    This prevents repeated pattern matches from inflating the score unboundedly.

    Each finding's contribution is also scaled by its confidence value (clamped
    to [0, 1]). Findings with confidence <= 0 are skipped entirely — they do not
    contribute to the score but remain in the reported findings list.

    Within each rule_id bucket, findings are processed in severity-descending
    order so that the highest-severity occurrence always receives the full weight.

    Base points per severity: CRITICAL=50, HIGH=25, MEDIUM=10, LOW=5.
    Multiplier: 1.3x if has_executable_scripts.
    """
    severity_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    sorted_findings = sorted(
        findings,
        key=lambda f: (f.rule_id or "UNKNOWN", severity_rank.get((f.severity or "LOW").upper(), 4)),
    )

    rule_occurrence_count: dict[str, int] = {}
    score = 0.0

    for f in sorted_findings:
        confidence = max(0.0, min(1.0, f.confidence))
        if confidence <= 0.0:
            continue

        sev = (f.severity or "LOW").upper()
        base_points = _SEVERITY_POINTS.get(sev, 5)

        rule_id = f.rule_id or "UNKNOWN"
        count = rule_occurrence_count.get(rule_id, 0)
        rule_occurrence_count[rule_id] = count + 1

        if count >= _MAX_OCCURRENCES_PER_RULE:
            continue

        weight = _DIMINISHING_WEIGHTS[count]
        score += base_points * weight * confidence

    if has_executable_scripts:
        score *= 1.3

    final_score = min(100, max(0, int(score)))

    severity_band = "LOW"
    for threshold, band in _RISK_SEVERITY_BANDS:
        if final_score >= threshold:
            severity_band = band
            break
    recommendation = _RISK_RECOMMENDATION.get(severity_band, "CAUTION")
    return final_score, severity_band, recommendation


def _build_sarif(
    findings: list[Finding],
    suppressed: list[SuppressedFinding] | None = None,
) -> dict[str, object]:
    """Build SARIF 2.1.0 log from findings.

    Filters out empty/malformed findings (missing rule_id or message) and
    builds the required tool.driver.rules[] array from referenced rule IDs.
    """
    results: list[SarifResult] = []
    seen_rule_ids: dict[str, str] = {}

    for finding in findings:
        if not finding.rule_id or not finding.message:
            continue
        start_line = finding.start_line
        end_line = finding.end_line
        region = SarifRegion(start_line=start_line, end_line=end_line)
        results.append(
            SarifResult(
                rule_id=finding.rule_id,
                message=SarifMessage(text=finding.message),
                level=_severity_to_sarif_level(finding.severity),
                locations=[
                    SarifLocation(
                        physical_location=SarifPhysicalLocation(
                            artifact_location=SarifArtifactLocation(uri=finding.file),
                            region=region,
                        )
                    )
                ],
            )
        )
        if finding.rule_id not in seen_rule_ids:
            seen_rule_ids[finding.rule_id] = finding.message

    # Baseline-suppressed findings are kept in the SARIF for an audit trail, but
    # marked with the `suppressions` property so consumers exclude them from counts.
    for sf in suppressed or []:
        finding = sf.finding
        if not finding.rule_id or not finding.message:
            continue
        results.append(
            SarifResult(
                rule_id=finding.rule_id,
                message=SarifMessage(text=finding.message),
                level=_severity_to_sarif_level(finding.severity),
                locations=[
                    SarifLocation(
                        physical_location=SarifPhysicalLocation(
                            artifact_location=SarifArtifactLocation(uri=finding.file),
                            region=SarifRegion(
                                start_line=finding.start_line, end_line=finding.end_line
                            ),
                        )
                    )
                ],
                suppressions=[SarifSuppression(kind="external", justification=sf.reason)],
            )
        )
        if finding.rule_id not in seen_rule_ids:
            seen_rule_ids[finding.rule_id] = finding.message

    rules = [
        SarifReportingDescriptor(
            id=rule_id,
            short_description=SarifMessage(text=description),
        )
        for rule_id, description in sorted(seen_rule_ids.items())
    ]

    sarif_log = SarifLog(
        schema_=SARIF_SCHEMA_URI,
        runs=[
            SarifRun(
                tool=SarifTool(
                    driver=SarifDriver(
                        name="skillspector",
                        version=skillspector_version,
                        rules=rules if rules else None,
                    )
                ),
                results=results,
            )
        ],
    )
    return sarif_log.model_dump(mode="json", by_alias=True, exclude_none=True)


def _format_terminal(
    findings: list[Finding],
    component_metadata: list[dict[str, object]],
    manifest: dict[str, object],
    skill_path: str | None,
    risk_score: int,
    risk_severity: str,
    risk_recommendation: str,
    has_executable_scripts: bool,
    suppressed: list[SuppressedFinding] | None = None,
    show_suppressed: bool = False,
) -> str:
    """Generate Rich terminal output and export as string."""
    suppressed = suppressed or []
    console = Console(record=True, force_terminal=True, width=80, file=StringIO())
    skill_name = (manifest.get("name") or "unknown") if manifest else "unknown"
    source = skill_path or ""

    console.print()
    console.print(
        Panel(
            "[bold]SkillSpector Security Report[/bold]",
            subtitle=f"v{skillspector_version}",
        )
    )
    console.print(f"\n[bold]Skill:[/bold] {skill_name}")
    console.print(f"[bold]Source:[/bold] {source}")
    console.print(f"[bold]Scanned:[/bold] {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    severity_colors = {
        "LOW": "green",
        "MEDIUM": "yellow",
        "HIGH": "red",
        "CRITICAL": "bold red",
    }
    color = severity_colors.get(risk_severity, "yellow")
    console.print("\n")
    risk_table = Table(title="Risk Assessment", show_header=False, box=None)
    risk_table.add_column("Metric", style="bold")
    risk_table.add_column("Value")
    risk_table.add_row("Score", f"[{color}]{risk_score}/100[/{color}]")
    risk_table.add_row("Severity", f"[{color}]{risk_severity}[/{color}]")
    risk_table.add_row(
        "Recommendation", f"[{color}]{risk_recommendation.replace('_', ' ')}[/{color}]"
    )
    console.print(risk_table)

    console.print("\n")
    comp_table = Table(title=f"Components ({len(component_metadata)})")
    comp_table.add_column("File", style="cyan")
    comp_table.add_column("Type")
    comp_table.add_column("Lines", justify="right")
    comp_table.add_column("Executable")
    for comp in component_metadata[:15]:
        path = comp.get("path", "")
        typ = comp.get("type", "")
        lines = comp.get("lines", 0)
        exec_flag = comp.get("executable", False)
        exec_marker = "[yellow]Yes[/yellow]" if exec_flag else "No"
        comp_table.add_row(path, typ, str(lines), exec_marker)
    if len(component_metadata) > 15:
        comp_table.add_row(f"... and {len(component_metadata) - 15} more", "", "", "")
    console.print(comp_table)

    if findings:
        console.print("\n")
        console.print(f"[bold]Issues ({len(findings)})[/bold]\n")
        severity_icons = {
            "LOW": "[green]LOW[/green]",
            "MEDIUM": "[yellow]MEDIUM[/yellow]",
            "HIGH": "[red]HIGH[/red]",
            "CRITICAL": "[bold red]CRITICAL[/bold red]",
        }
        for f in findings:
            icon = severity_icons.get((f.severity or "LOW").upper(), f.severity)
            console.print(f"  {icon}: {f.rule_id} - {f.message[:60]}...")
            end = f"–{f.end_line}" if f.end_line and f.end_line != f.start_line else ""
            console.print(f"    [dim]Location:[/dim] {f.file}:{f.start_line}{end}")
            console.print(f"    [dim]Confidence:[/dim] {f.confidence:.0%}")
            if f.remediation:
                console.print(f"    [dim]Remediation:[/dim] {(f.remediation or '')[:150]}...")
            console.print()
    else:
        console.print("\n[green]No security issues detected.[/green]\n")

    if suppressed:
        console.print(
            f"[dim]Suppressed by baseline: {len(suppressed)} (not counted toward risk score)[/dim]"
        )
        if show_suppressed:
            for sf in suppressed:
                f = sf.finding
                console.print(
                    f"  [dim]- {f.rule_id} {f.file}:{f.start_line} (reason: {sf.reason})[/dim]"
                )
        else:
            console.print("[dim]Use --show-suppressed to list them.[/dim]")
        console.print()

    console.print(f"[dim]Executable scripts: {'Yes' if has_executable_scripts else 'No'}[/dim]")
    return console.export_text()


def _build_metadata(has_executable_scripts: bool, use_llm: bool) -> dict[str, object]:
    """Build the metadata section shared by all output formats."""
    llm_available, llm_error = is_llm_available()
    meta_analysis_applied = use_llm and llm_available
    meta: dict[str, object] = {
        "has_executable_scripts": has_executable_scripts,
        "skillspector_version": skillspector_version,
        "llm_requested": use_llm,
        "llm_available": llm_available,
        "meta_analysis_applied": meta_analysis_applied,
    }
    if not meta_analysis_applied:
        meta["filtering_mode"] = "heuristic"
    if use_llm and not llm_available:
        meta["llm_error"] = llm_error
    return meta


def _build_analysis_completeness(
    components: list[str],
    file_cache: dict[str, str],
    use_llm: bool,
    findings_pre_filter: list[Finding],
    findings_post_filter: list[Finding],
) -> dict[str, object]:
    """Build analysis_completeness section indicating scan coverage and limitations.

    Helps consumers understand what was NOT analyzed and whether findings
    can be trusted as comprehensive.
    """
    total_components = len(components)
    scanned_components = sum(1 for c in components if c in file_cache)

    llm_available, llm_error = is_llm_available()
    llm_used = use_llm and llm_available

    limitations: list[str] = []
    if scanned_components < total_components:
        skipped = total_components - scanned_components
        limitations.append(f"{skipped} component(s) had no content in file_cache (skipped)")
    if use_llm and not llm_available:
        limitations.append(f"LLM meta-analysis unavailable: {llm_error or 'unknown reason'}")
    if not use_llm:
        limitations.append("LLM meta-analysis was disabled (--no-llm)")

    findings_dropped = len(findings_pre_filter) - len(findings_post_filter)
    if findings_dropped > 0:
        limitations.append(f"{findings_dropped} finding(s) filtered by meta-analyzer or heuristics")

    completeness: dict[str, object] = {
        "total_components": total_components,
        "scanned_components": scanned_components,
        "coverage_percent": round(scanned_components / total_components * 100, 1)
        if total_components > 0
        else 100.0,
        "llm_analysis": "applied" if llm_used else "skipped",
        "findings_before_filtering": len(findings_pre_filter),
        "findings_after_filtering": len(findings_post_filter),
        "limitations": limitations if limitations else None,
        "is_complete": len(limitations) == 0,
    }
    return completeness


def _format_json(
    findings: list[Finding],
    component_metadata: list[dict[str, object]],
    manifest: dict[str, object],
    skill_path: str | None,
    risk_score: int,
    risk_severity: str,
    risk_recommendation: str,
    has_executable_scripts: bool,
    use_llm: bool = True,
    analysis_completeness: dict[str, object] | None = None,
    suppressed: list[SuppressedFinding] | None = None,
) -> str:
    """Generate JSON report string."""
    suppressed = suppressed or []
    skill_name = (manifest.get("name") or "unknown") if manifest else "unknown"
    data: dict[str, object] = {
        "skill": {
            "name": skill_name,
            "source": skill_path or "",
            "scanned_at": datetime.now(UTC).isoformat(),
        },
        "risk_assessment": {
            "score": risk_score,
            "severity": risk_severity,
            "recommendation": risk_recommendation,
        },
        "components": [
            {
                "path": c.get("path"),
                "type": c.get("type"),
                "lines": c.get("lines"),
                "executable": c.get("executable"),
                "size_bytes": c.get("size_bytes"),
            }
            for c in component_metadata
        ],
        "issues": [f.to_dict() for f in findings],
        "suppressed_count": len(suppressed),
        "suppressed": [sf.to_dict() for sf in suppressed],
        "metadata": _build_metadata(has_executable_scripts, use_llm),
    }
    if analysis_completeness is not None:
        data["analysis_completeness"] = analysis_completeness
    return json.dumps(data, indent=2)


def _format_markdown(
    findings: list[Finding],
    component_metadata: list[dict[str, object]],
    manifest: dict[str, object],
    skill_path: str | None,
    risk_score: int,
    risk_severity: str,
    risk_recommendation: str,
    has_executable_scripts: bool,
    suppressed: list[SuppressedFinding] | None = None,
    show_suppressed: bool = False,
) -> str:
    """Generate Markdown report string."""
    suppressed = suppressed or []
    lines: list[str] = []
    skill_name = (manifest.get("name") or "unknown") if manifest else "unknown"
    source = skill_path or ""

    lines.append("# SkillSpector Security Report\n")
    lines.append(f"**Skill:** {skill_name}  ")
    lines.append(f"**Source:** `{source}`  ")
    lines.append(f"**Scanned:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}  ")
    lines.append("")

    lines.append("## Risk Assessment\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Score | {risk_score}/100 |")
    lines.append(f"| Severity | {risk_severity} |")
    lines.append(f"| Recommendation | {risk_recommendation.replace('_', ' ')} |")
    lines.append("")

    lines.append(f"## Components ({len(component_metadata)})\n")
    lines.append("| File | Type | Lines | Executable |")
    lines.append("|------|------|-------|------------|")
    for comp in component_metadata:
        path = comp.get("path", "")
        typ = comp.get("type", "")
        line_count = comp.get("lines", 0)
        exec_flag = comp.get("executable", False)
        exec_marker = "Yes" if exec_flag else "No"
        lines.append(f"| `{path}` | {typ} | {line_count} | {exec_marker} |")
    lines.append("")

    lines.append(f"## Issues ({len(findings)})\n")
    if not findings:
        lines.append("No security issues detected.\n")
    else:
        severity_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🔴"}
        for f in findings:
            sev = (f.severity or "LOW").upper()
            emoji = severity_emoji.get(sev, "")
            lines.append(f"### {emoji} {sev}: {f.rule_id}\n")
            end = f"–{f.end_line}" if f.end_line and f.end_line != f.start_line else ""
            lines.append(f"**Location:** `{f.file}:{f.start_line}{end}`  ")
            lines.append(f"**Confidence:** {f.confidence:.0%}  ")
            lines.append("")
            lines.append(f"**Message:** {f.message}")
            lines.append("")
            if f.remediation:
                lines.append(f"**Remediation:** {f.remediation}")
                lines.append("")
            lines.append("---\n")

    if suppressed:
        lines.append(f"## Suppressed ({len(suppressed)})\n")
        lines.append(
            "These findings matched the baseline and are **not** counted toward the risk score.\n"
        )
        if show_suppressed:
            lines.append("| Rule | Location | Reason |")
            lines.append("|------|----------|--------|")
            for sf in suppressed:
                f = sf.finding
                reason = sf.reason.replace("|", "\\|")
                lines.append(f"| {f.rule_id} | `{f.file}:{f.start_line}` | {reason} |")
            lines.append("")
        else:
            lines.append("_Run with `--show-suppressed` to list them._\n")

    lines.append("## Metadata\n")
    lines.append(f"- **Executable Scripts:** {'Yes' if has_executable_scripts else 'No'}")
    lines.append(f"\n*Generated by SkillSpector v{skillspector_version}*")
    return "\n".join(lines)


def report(state: SkillspectorState) -> dict[str, object]:
    """Generate SARIF, compute risk score, and set report_body from output_format.

    A baseline (state["baseline"]) suppresses matching findings: they never count
    toward the risk score and are excluded from SARIF. They are shown in the
    human-readable report only when state["show_suppressed"] is True.
    """
    raw_findings = state.get("findings", [])
    filtered_findings = state.get("filtered_findings", raw_findings)
    component_metadata = state.get("component_metadata") or []
    components = state.get("components") or []
    file_cache = state.get("file_cache") or {}
    has_executable_scripts = state.get("has_executable_scripts", False)
    manifest = state.get("manifest") or {}
    skill_path = state.get("skill_path")
    output_format = state.get("output_format") or "sarif"
    use_llm = state.get("use_llm", True)

    baseline = state.get("baseline")
    show_suppressed = state.get("show_suppressed", False)
    active_findings, suppressed = partition_findings(
        filtered_findings, baseline if isinstance(baseline, Baseline) else None
    )

    # Risk and SARIF reflect only the active (non-suppressed) findings; scoring
    # additionally de-duplicates so the same issue is not counted twice.
    findings_for_scoring = deduplicate(active_findings)
    risk_score, risk_severity, risk_recommendation = _compute_risk_score(
        findings_for_scoring, has_executable_scripts
    )
    sarif_report = _build_sarif(active_findings, suppressed)
    analysis_completeness = _build_analysis_completeness(
        components, file_cache, use_llm, raw_findings, filtered_findings
    )

    if output_format == "terminal":
        report_body = _format_terminal(
            active_findings,
            component_metadata,
            manifest,
            skill_path,
            risk_score,
            risk_severity,
            risk_recommendation,
            has_executable_scripts,
            suppressed=suppressed,
            show_suppressed=show_suppressed,
        )
    elif output_format == "json":
        report_body = _format_json(
            active_findings,
            component_metadata,
            manifest,
            skill_path,
            risk_score,
            risk_severity,
            risk_recommendation,
            has_executable_scripts,
            use_llm=use_llm,
            analysis_completeness=analysis_completeness,
            suppressed=suppressed,
        )
    elif output_format == "markdown":
        report_body = _format_markdown(
            active_findings,
            component_metadata,
            manifest,
            skill_path,
            risk_score,
            risk_severity,
            risk_recommendation,
            has_executable_scripts,
            suppressed=suppressed,
            show_suppressed=show_suppressed,
        )
    else:
        report_body = json.dumps(sarif_report, indent=2)

    logger.debug(
        "Report generated: format=%s, findings_count=%d, suppressed_count=%d",
        output_format,
        len(active_findings),
        len(suppressed),
    )

    out: dict[str, object] = {
        "sarif_report": sarif_report,
        "risk_score": risk_score,
        "risk_severity": risk_severity,
        "risk_recommendation": risk_recommendation,
        "report_body": report_body,
        "filtered_findings": filtered_findings,
        "suppressed_findings": suppressed,
    }
    return out
