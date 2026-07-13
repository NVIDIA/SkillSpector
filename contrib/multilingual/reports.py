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

"""Batch report formatters — terminal (Rich), JSON, and Markdown.

All three formatters accept the same ``list[dict]`` result list and
produce a string.  The entry shape is defined by
:func:`~contrib.multilingual.runner.entry_from_result`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from io import StringIO

from skillspector import __version__ as _skillspector_version


def sorted_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return *results* sorted by risk score descending."""
    return sorted(
        results,
        key=lambda x: x.get("risk_assessment", {}).get("score", 0),  # type: ignore[no-any-return]
        reverse=True,
    )


# ═══════════════════════════════════════════════════════════════════
#  Terminal (Rich)
# ═══════════════════════════════════════════════════════════════════


def _format_terminal(results: list[dict[str, object]]) -> str:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        return _format_terminal_plain(results)

    capture = Console(record=True, force_terminal=True, width=80, file=StringIO())
    total = len(results)

    critical = _count_sev(results, "CRITICAL")
    high = _count_sev(results, "HIGH")
    medium = _count_sev(results, "MEDIUM")
    low_count = _count_sev(results, "LOW")
    errs = sum(1 for r in results if r.get("error"))
    completed = total - errs

    # ── Enhancement summary (for multilingual-enhanced mode) ────
    non_en = sum(1 for r in results if r.get("skill", {}).get("language", "en") != "en")
    gap_fill_total = sum(
        r.get("enhancements", {}).get("gap_fill_findings", 0) for r in results
    )
    gap_fill_skills = sum(
        1 for r in results if r.get("enhancements", {}).get("gap_fill_applied")
    )

    capture.print()
    capture.print(
        Panel(
            "[bold]SkillSpector Batch Scan Report[/bold]",
            subtitle=(
                f"v{_skillspector_version}  |  "
                "[green]Multilingual Enhanced[/green]"
            ),
        )
    )
    capture.print()
    capture.print(f"[bold]Total:[/bold] {total} skill(s) scanned")
    if errs:
        capture.print(f"[red]Errors:[/red] {errs}")
    if non_en:
        capture.print(
            f"[bold]Multilingual:[/bold] {non_en} non-English skill(s) "
            f"({gap_fill_skills} gap-fill applied, "
            f"{gap_fill_total} gap-fill finding(s))"
        )
    capture.print(
        "[dim]Compare with standard scan: "
        "skillspector scan <skill> -f json[/dim]"
    )
    capture.print()

    # ── Source breakdown ─────────────────────────────────────────
    _print_source_breakdown(capture, results)
    # ── Language breakdown ───────────────────────────────────────
    _print_language_breakdown(capture, results)

    severity_colors: dict[str, str] = {
        "LOW": "green",
        "MEDIUM": "yellow",
        "HIGH": "red",
        "CRITICAL": "bold red",
        "ERROR": "red",
    }

    table = Table(title=f"Skills by Risk Score ({completed} completed)")
    table.add_column("Skill", style="cyan")
    table.add_column("LR")
    table.add_column("Score", justify="right")
    table.add_column("Severity")
    table.add_column("Issues", justify="right")
    table.add_column("Lang")

    for r in sorted_results(results):
        skill = r.get("skill", {})
        risk = r.get("risk_assessment", {})
        name = skill.get("name", "?")
        score = risk.get("score", 0)
        sev = risk.get("severity", "LOW")
        color = severity_colors.get(sev, "")
        issues = len(r.get("issues", []))
        lang = skill.get("language", "en")
        lr = _lr_icon(sev, lang)

        if r.get("error"):
            table.add_row(str(name), "-", "ERR", "[red]ERROR[/red]", "—", lang)
        else:
            table.add_row(
                str(name),
                lr,
                f"[{color}]{score}/100[/{color}]",
                f"[{color}]{sev}[/{color}]",
                str(issues),
                lang,
            )
    capture.print(table)
    capture.print()

    if critical + high > 0:
        capture.print(
            f"[bold red]{critical + high} skill(s)[/bold red] "
            "with HIGH or CRITICAL risk — review immediately"
        )
    if medium > 0:
        capture.print(
            f"[yellow]{medium} skill(s)[/yellow] "
            "with MEDIUM risk — review before installing"
        )
    if low_count > 0:
        capture.print(
            f"[green]{low_count} skill(s)[/green] with LOW risk — likely safe"
        )
    capture.print()

    return capture.export_text()


def _count_sev(results: list[dict[str, object]], severity: str) -> int:
    return sum(
        1
        for r in results
        if r.get("risk_assessment", {}).get("severity") == severity
    )


def _lr_icon(severity: str, language: str) -> str:
    """Language Reliability indicator for the LR column."""
    if language == "en":
        return "[green]✓[/green]"  # ✓
    return "[yellow]⚠[/yellow]"  # ⚠


def _print_source_breakdown(c, results: list[dict[str, object]]) -> None:
    group_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    )
    for r in results:
        group = r.get("skill", {}).get("source_group", ".")
        sev = r.get("risk_assessment", {}).get("severity", "LOW")
        group_stats[group]["total"] += 1
        if sev in group_stats[group]:
            group_stats[group][sev] += 1

    if len(group_stats) > 1:
        c.print("[bold]Source Breakdown:[/bold]")
        for group in sorted(group_stats):
            st = group_stats[group]
            parts = [f"  {group:<30s} {st['total']:>4d} skills"]
            if st["CRITICAL"]:
                parts.append(f"[bold red]{st['CRITICAL']} CRITICAL[/bold red]")
            if st["HIGH"]:
                parts.append(f"[red]{st['HIGH']} HIGH[/red]")
            if st["MEDIUM"]:
                parts.append(f"[yellow]{st['MEDIUM']} MEDIUM[/yellow]")
            c.print(", ".join(parts))
        c.print()


def _print_language_breakdown(c, results: list[dict[str, object]]) -> None:
    lang_stats: dict[str, int] = defaultdict(int)
    lang_non_en: set[str] = set()
    for r in results:
        lang = r.get("skill", {}).get("language", "en")
        lang_stats[lang] = lang_stats.get(lang, 0) + 1
        if lang != "en":
            lang_non_en.add(lang)

    if len(lang_stats) > 1:
        c.print("[bold]Language Breakdown:[/bold]")
        for lang in sorted(lang_stats):
            count = lang_stats[lang]
            if lang == "en":
                c.print(f"  {lang:<6s} {count:>4d} skills  (static + LLM coverage: full)")
            else:
                c.print(
                    f"  {lang:<6s} {count:>4d} skills  "
                    f"[yellow](static: partial, LLM: full)[/yellow]"
                )
        c.print()


def _format_terminal_plain(results: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for r in sorted_results(results):
        risk = r.get("risk_assessment", {})
        skill = r.get("skill", {})
        lines.append(
            f"  {skill.get('name', '?'):40s} "
            f"{risk.get('score', 0):>3}/100 {risk.get('severity', 'LOW'):<8s}"
        )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  JSON
# ═══════════════════════════════════════════════════════════════════


def _format_json(results: list[dict[str, object]]) -> str:
    entries: list[dict[str, object]] = []
    for r in sorted_results(results):
        skill = r.get("skill", {})
        entry: dict[str, object] = {
            "skill": {
                "name": skill.get("name"),
                "source": skill.get("source"),
                "source_group": skill.get("source_group"),
                "language": skill.get("language"),
                "scanned_at": skill.get("scanned_at"),
            },
            "risk_assessment": r.get("risk_assessment", {}),
            "components": r.get("components", []),
            "issues": r.get("issues", []),
            "scan_mode": r.get("scan_mode", "multilingual-enhanced"),
            "enhancements": r.get("enhancements", {}),
        }
        if r.get("error"):
            entry["error"] = r["error"]
        entries.append(entry)

    # Aggregate enhancement stats for the batch envelope
    non_en_langs: set[str] = set()
    gap_fill_total = 0
    gap_fill_skills = 0
    for r in results:
        lang = r.get("skill", {}).get("language", "en")
        if lang != "en":
            non_en_langs.add(lang)
        enhancements = r.get("enhancements", {})
        gap_fill_total += enhancements.get("gap_fill_findings", 0)
        if enhancements.get("gap_fill_applied"):
            gap_fill_skills += 1

    data: dict[str, object] = {
        "batch": {
            "scanned_at": datetime.now(UTC).isoformat(),
            "total_skills": len(results),
            "scan_mode": "multilingual-enhanced",
            "enhancements": {
                "language_detection": "unicode-script-ratio",
                "languages_detected": {lang: sum(
                    1 for r in results
                    if r.get("skill", {}).get("language") == lang
                ) for lang in sorted(non_en_langs)},
                "gap_fill_applied": gap_fill_skills,
                "gap_fill_findings": gap_fill_total,
            },
        },
        "skills": entries,
        "metadata": {
            "skillspector_version": _skillspector_version,
        },
    }
    return json.dumps(data, indent=2)


# ═══════════════════════════════════════════════════════════════════
#  Markdown
# ═══════════════════════════════════════════════════════════════════


def _format_markdown(results: list[dict[str, object]]) -> str:
    lines: list[str] = []
    total = len(results)

    # ── Enhancement summary ─────────────────────────────────────
    non_en = sum(1 for r in results if r.get("skill", {}).get("language", "en") != "en")
    gap_fill_total = sum(
        r.get("enhancements", {}).get("gap_fill_findings", 0) for r in results
    )
    gap_fill_skills = sum(
        1 for r in results if r.get("enhancements", {}).get("gap_fill_applied")
    )

    lines.append("# SkillSpector Batch Scan Report\n")
    lines.append(
        f"**Scan mode:** Multilingual Enhanced  \n"
        f"**Version:** v{_skillspector_version}  \n"
    )
    if non_en:
        lines.append(
            f"**Enhancements:** {non_en} non-English skill(s) — "
            f"{gap_fill_skills} gap-fill applied, "
            f"{gap_fill_total} gap-fill finding(s)  \n"
        )
    lines.append(
        "**Compare with:** `skillspector scan <skill> -f json` "
        "for standard single-skill output  \n"
    )
    lines.append(f"**Skills scanned:** {total}  ")
    lines.append(
        f"**Scanned at:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}  \n"
    )

    critical = _count_sev(results, "CRITICAL")
    high = _count_sev(results, "HIGH")
    medium = _count_sev(results, "MEDIUM")
    low_count = _count_sev(results, "LOW")

    lines.append("## Summary\n")
    lines.append("| Severity | Count |")
    lines.append("|----------|-------|")
    lines.append(f"| 🔴 CRITICAL | {critical} |")
    lines.append(f"| 🔴 HIGH | {high} |")
    lines.append(f"| 🟡 MEDIUM | {medium} |")
    lines.append(f"| 🟢 LOW | {low_count} |")
    lines.append("")

    lines.append("## Skills by Risk Score\n")
    lines.append("| Skill | Score | Severity | Issues | Lang |")
    lines.append("|-------|-------|----------|--------|------|")
    for r in sorted_results(results):
        skill = r.get("skill", {})
        risk = r.get("risk_assessment", {})
        name = skill.get("name", "?")
        score = risk.get("score", 0)
        sev = risk.get("severity", "LOW")
        issues = len(r.get("issues", []))
        lang = skill.get("language", "en")

        if r.get("error"):
            lines.append(f"| `{name}` | ERR | ERROR | — | {lang} |")
        else:
            lines.append(f"| `{name}` | {score}/100 | {sev} | {issues} | {lang} |")
    lines.append("")

    # ── Issue details for HIGH / CRITICAL ────────────────────────
    high_critical = [
        r
        for r in sorted_results(results)
        if r.get("risk_assessment", {}).get("severity") in ("HIGH", "CRITICAL")
        and not r.get("error")
    ]
    if high_critical:
        severity_emoji = {"HIGH": "\U0001f534", "CRITICAL": "\U0001f534"}
        lines.append("## 🔴 HIGH / CRITICAL Issue Details\n")
        for r in high_critical:
            skill = r.get("skill", {})
            risk = r.get("risk_assessment", {})
            name = skill.get("name", "?")
            lines.append(
                f"### {name} — {risk.get('score', 0)}/100 "
                f"{risk.get('severity', 'HIGH')}\n"
            )
            for issue in r.get("issues", []):
                sev = str(issue.get("severity", "LOW")).upper()
                emoji = severity_emoji.get(sev, "")
                loc = issue.get("location", {})
                loc_start = loc.get("start_line", "?") if isinstance(loc, dict) else "?"
                loc_file = loc.get("file", "") if isinstance(loc, dict) else ""
                rule_id = issue.get("id", "?")
                explanation = issue.get("explanation", issue.get("message", ""))
                lines.append(f"- **{emoji} {rule_id}**: {explanation}")
                if loc_file:
                    lines.append(f"  - Location: `{loc_file}:{loc_start}`")
                conf = issue.get("confidence", 0)
                lines.append(f"  - Confidence: {float(conf):.0%}")
                rem = issue.get("remediation")
                if rem:
                    lines.append(f"  - Remediation: {rem}")
                lines.append("")
        lines.append("")

    lines.append(f"\n*Generated by SkillSpector v{_skillspector_version}*")
    return "\n".join(lines)
