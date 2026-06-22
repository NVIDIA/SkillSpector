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
from skillspector.sarif_models import (
    SARIF_SCHEMA_URI,
    SarifArtifactLocation,
    SarifDriver,
    SarifLocation,
    SarifLog,
    SarifMessage,
    SarifPhysicalLocation,
    SarifRegion,
    SarifResult,
    SarifRun,
    SarifTool,
)
from skillspector.state import SkillspectorState

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


def _compute_risk_score(
    findings: list[Finding], has_executable_scripts: bool
) -> tuple[int, str, str]:
    """
    Compute risk score (0-100), severity band, and recommendation.
    v1 rules: CRITICAL +50, HIGH +25, MEDIUM +10, LOW +5; 1.3x if has_executable_scripts.
    """
    score = 0
    for f in findings:
        sev = (f.severity or "LOW").upper()
        if sev == "CRITICAL":
            score += 50
        elif sev == "HIGH":
            score += 25
        elif sev == "MEDIUM":
            score += 10
        elif sev == "LOW":
            score += 5
    if has_executable_scripts:
        score = int(score * 1.3)
    score = min(100, max(0, score))

    severity_band = "LOW"
    for threshold, band in _RISK_SEVERITY_BANDS:
        if score >= threshold:
            severity_band = band
            break
    recommendation = _RISK_RECOMMENDATION.get(severity_band, "CAUTION")
    return score, severity_band, recommendation


def _build_sarif(findings: list[Finding]) -> dict[str, object]:
    """Build SARIF 2.1.0 log from findings."""
    results: list[SarifResult] = []
    for finding in findings:
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
    sarif_log = SarifLog(
        schema_=SARIF_SCHEMA_URI,
        runs=[
            SarifRun(
                tool=SarifTool(
                    driver=SarifDriver(name="skillspector", version=skillspector_version)
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
) -> str:
    """Generate Rich terminal output and export as string."""
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

    console.print(f"[dim]Executable scripts: {'Yes' if has_executable_scripts else 'No'}[/dim]")
    return console.export_text()


def _build_metadata(has_executable_scripts: bool, use_llm: bool) -> dict[str, object]:
    """Build the metadata section shared by all output formats."""
    llm_available, llm_error = is_llm_available()
    meta: dict[str, object] = {
        "has_executable_scripts": has_executable_scripts,
        "skillspector_version": skillspector_version,
        "llm_requested": use_llm,
        "llm_available": llm_available,
    }
    if use_llm and not llm_available:
        meta["llm_error"] = llm_error
    return meta


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
) -> str:
    """Generate JSON report string."""
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
        "metadata": _build_metadata(has_executable_scripts, use_llm),
    }
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
) -> str:
    """Generate Markdown report string."""
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

    lines.append("## Metadata\n")
    lines.append(f"- **Executable Scripts:** {'Yes' if has_executable_scripts else 'No'}")
    lines.append(f"\n*Generated by SkillSpector v{skillspector_version}*")
    return "\n".join(lines)


def report(state: SkillspectorState) -> dict[str, object]:
    """Generate SARIF, compute risk score, and set report_body from output_format."""
    findings = state.get("filtered_findings", state.get("findings", []))
    # When use_llm is False, meta_analyzer is skipped; ensure final state has filtered_findings
    if "filtered_findings" not in state:
        filtered_findings = state.get("findings", [])
    else:
        filtered_findings = findings
    component_metadata = state.get("component_metadata") or []
    has_executable_scripts = state.get("has_executable_scripts", False)
    manifest = state.get("manifest") or {}
    skill_path = state.get("skill_path")
    output_format = state.get("output_format") or "sarif"
    use_llm = state.get("use_llm", True)

    risk_score, risk_severity, risk_recommendation = _compute_risk_score(
        findings, has_executable_scripts
    )
    sarif_report = _build_sarif(findings)

    if output_format == "terminal":
        report_body = _format_terminal(
            findings,
            component_metadata,
            manifest,
            skill_path,
            risk_score,
            risk_severity,
            risk_recommendation,
            has_executable_scripts,
        )
    elif output_format == "json":
        report_body = _format_json(
            findings,
            component_metadata,
            manifest,
            skill_path,
            risk_score,
            risk_severity,
            risk_recommendation,
            has_executable_scripts,
            use_llm=use_llm,
        )
    elif output_format == "markdown":
        report_body = _format_markdown(
            findings,
            component_metadata,
            manifest,
            skill_path,
            risk_score,
            risk_severity,
            risk_recommendation,
            has_executable_scripts,
        )
    else:
        report_body = json.dumps(sarif_report, indent=2)

    logger.debug(
        "Report generated: format=%s, findings_count=%d",
        output_format,
        len(filtered_findings),
    )

    out: dict[str, object] = {
        "sarif_report": sarif_report,
        "risk_score": risk_score,
        "risk_severity": risk_severity,
        "risk_recommendation": risk_recommendation,
        "report_body": report_body,
        "filtered_findings": filtered_findings,
    }
    return out
