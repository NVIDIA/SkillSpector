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

"""CLI for Skillspector — thin wrapper over the LangGraph workflow.

Maps CLI args to initial state, invokes the graph, then maps result to output and exit code.
No business logic; workflow lives in the graph.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from langchain_core.runnables import RunnableConfig
from rich.console import Console

from skillspector import __version__
from skillspector.graph import graph
from skillspector.logging_config import get_logger, set_level
from skillspector.multi_skill import MultiSkillDetectionResult, detect_skills
from skillspector.suppression import build_baseline_dict, dump_baseline, load_baseline

logger = get_logger(__name__)


def _ensure_utf8_streams() -> None:
    """Reconfigure stdout/stderr to UTF-8 so Unicode report output does not crash.

    On Windows the default console encoding (e.g. cp1252) cannot encode the
    box-drawing characters and icons used in the terminal report, which raises
    UnicodeEncodeError. Reconfiguring with errors="replace" makes output robust
    across platforms without crashing.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                logger.debug("Could not reconfigure %s to UTF-8", stream)


_ensure_utf8_streams()

app = typer.Typer(
    name="skillspector",
    help="Security scanner for AI agent skills (LangGraph). Detect vulnerabilities before installation.",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


class FormatChoice(StrEnum):
    """Output format choices for the CLI."""

    terminal = "terminal"
    json = "json"
    markdown = "markdown"
    sarif = "sarif"


class TransportChoice(StrEnum):
    """Transport choices for the MCP server."""

    stdio = "stdio"
    http = "http"


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"SkillSpector v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """
    SkillSpector - Security scanner for AI agent skills (LangGraph).

    Analyze skill bundles to detect vulnerabilities and security risks.
    Supports: Git URL, file URL, .zip file, .md file, or directory.
    """
    pass


def _auto_discover_baseline(input_path: str) -> Path | None:
    """Return the auto-discovered baseline path, or None if not found.

    Looks for ``.skillspector-baseline.yaml`` in the resolved directory
    when *input_path* points to a local directory.
    """
    candidate = Path(input_path)
    if candidate.is_dir():
        bl = candidate.resolve() / ".skillspector-baseline.yaml"
        if bl.exists():
            return bl
    return None


def _scan_state(
    input_path: str,
    format: FormatChoice,
    no_llm: bool,
    yara_rules_dir: str | None = None,
    baseline: Path | None = None,
    show_suppressed: bool = False,
    include_test_fixtures: bool = False,
    skip_meta: bool = False,
    trust_skill_classification: bool = False,
) -> dict[str, object]:
    """Build initial graph state from scan CLI args."""
    state: dict[str, object] = {
        "input_path": input_path,
        "output_format": format.value,
        "use_llm": not no_llm,
        "trust_skill_classification": trust_skill_classification,
    }
    if yara_rules_dir is not None:
        state["yara_rules_dir"] = yara_rules_dir
    if baseline is not None:
        # Loading may raise FileNotFoundError/ValueError, mapped to exit code 2 by scan().
        state["baseline"] = load_baseline(baseline)
        state["show_suppressed"] = show_suppressed
    if include_test_fixtures:
        state["include_test_fixtures"] = True
    if skip_meta:
        state["skip_meta"] = True
    return state


def _result_body(result: dict) -> str:
    report_body = result.get("report_body") or ""
    if not report_body and result.get("sarif_report") is not None:
        report_body = json.dumps(result["sarif_report"], indent=2)
    return report_body


def _write_result(
    result: dict[str, object],
    output: Path | None,
    format: FormatChoice,
) -> None:
    """Write report_body to file or stdout. Uses sarif_report if report_body missing."""
    report_body = _result_body(result)
    if output:
        Path(output).write_text(report_body, encoding="utf-8")
        if format == FormatChoice.terminal:
            console.print(f"\n[green]Report saved to:[/green] {output}")
        else:
            console.print(f"Report saved to: {output}")
    else:
        if format == FormatChoice.terminal:
            console.print(report_body)
        else:
            print(report_body)


def _cleanup_result(result: dict[str, object]) -> None:
    """Remove temp dir from graph result if set."""
    temp_dir = result.get("temp_dir_for_cleanup")
    if temp_dir and isinstance(temp_dir, str):
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.command()
def scan(
    input_path: Annotated[
        str,
        typer.Argument(
            help="Path or URL to scan. Supports: Git URL, file URL, zip file, .md file, or directory.",
        ),
    ],
    format: Annotated[
        FormatChoice,
        typer.Option(
            "--format",
            "-f",
            help="Output format.",
            case_sensitive=False,
        ),
    ] = FormatChoice.terminal,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file path. If not specified, prints to stdout.",
        ),
    ] = None,
    no_llm: Annotated[
        bool,
        typer.Option(
            "--no-llm",
            help="Skip LLM analysis (faster, less accurate). Uses static analysis only.",
        ),
    ] = False,
    yara_rules_dir: Annotated[
        Path | None,
        typer.Option(
            "--yara-rules-dir",
            help="Directory containing additional YARA rule files (.yar/.yara) to load alongside built-in rules.",
        ),
    ] = None,
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive",
            "-r",
            help="Scan immediate subdirectories that each contain a SKILL.md as independent skills.",
        ),
    ] = False,
    depth: Annotated[
        int,
        typer.Option(
            "--depth",
            help="Directory depth to search for sub-skills with --recursive. Default: 1.",
        ),
    ] = 1,
    baseline: Annotated[
        Path | None,
        typer.Option(
            "--baseline",
            "-b",
            help="Baseline file (YAML/JSON) of suppressed findings. Matching findings "
            "are dropped before scoring. Generate one with 'skillspector baseline'.",
        ),
    ] = None,
    show_suppressed: Annotated[
        bool,
        typer.Option(
            "--show-suppressed",
            help="List findings suppressed by the baseline in the report (they still "
            "do not count toward the risk score).",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-V",
            help="Show detailed progress.",
        ),
    ] = False,
    include_test_fixtures: Annotated[
        bool,
        typer.Option(
            "--include-test-fixtures",
            help="Include AST4/PE3 findings that are likely test-harness patterns (shell=False + "
            "sys.executable, /etc/passwd in test assertion). Default: downgrade these to INFO.",
        ),
    ] = False,
    skip_meta: Annotated[
        bool,
        typer.Option(
            "--skip-meta",
            help="Skip the meta-analyzer LLM pass. Reduces token cost (~40-60%) at the cost of "
            "more false positives. Use for rapid iterative scanning; omit for final/CI runs.",
        ),
    ] = False,
    auto_baseline: Annotated[
        bool,
        typer.Option(
            "--auto-baseline",
            help="Auto-discover and apply .skillspector-baseline.yaml in the scanned "
            "directory. Off by default: the scanned directory may be untrusted, and a "
            "malicious skill could ship a baseline that suppresses findings about itself.",
        ),
    ] = False,
    detail: Annotated[
        bool,
        typer.Option(
            "--detail",
            help="Include full finding details (issues[]) in recursive JSON output.",
        ),
    ] = False,
    trust_skill_classification: Annotated[
        bool,
        typer.Option(
            "--trust-skill-classification",
            help="Trust the scanned skill's own self-declared 'offensive_security' "
            "classification (from its manifest) to override the risk recommendation. "
            "Off by default: the manifest is attacker-controlled, and a malicious "
            "skill could label itself this way to suppress a DO_NOT_INSTALL verdict. "
            "The self-declared classification is always shown in JSON output "
            "(skill_declared_classification) regardless of this flag.",
        ),
    ] = False,
) -> None:
    """
    Scan a skill for security vulnerabilities.

    Examples:

        skillspector scan ./my-skill/
        skillspector scan ./my-skill/ --format json --output report.json
        skillspector scan https://github.com/user/my-skill --no-llm
        skillspector scan ./skill-collection/ --recursive
        skillspector scan ./skill-collection/ --recursive --depth 2
        skillspector scan ./my-skill/ --include-test-fixtures

    Flags:

        --include-test-fixtures: Include AST4/PE3 findings that are likely test-harness
                                 patterns (shell=False + sys.executable, /etc/passwd in
                                 test assertion). Default: downgrade these to INFO.

    Environment variables:

        SKILLSPECTOR_PROVIDER  Active LLM provider: openai | anthropic |
                               anthropic_proxy | bedrock | nv_build |
                               nv_inference | subprocess. Defaults to the
                               NVIDIA path (nv_inference, falling back to
                               nv_build in OSS builds).
        SKILLSPECTOR_MODEL     Override the active provider's default
                               model (applies to every analyzer slot).
        SKILLSPECTOR_LOG_LEVEL DEBUG | INFO | WARNING | ERROR (default WARNING).

    Provider credentials (one of):

        OPENAI_API_KEY [+ OPENAI_BASE_URL]   for SKILLSPECTOR_PROVIDER=openai
        ANTHROPIC_API_KEY                    for SKILLSPECTOR_PROVIDER=anthropic
        AWS_PROFILE (optional) + AWS_REGION  for SKILLSPECTOR_PROVIDER=bedrock
                                             (AWS_PROFILE: standard boto3 credential
                                             chain when unset; AWS_REGION default: us-west-2)
        NVIDIA_INFERENCE_KEY                 for the NVIDIA providers
        SKILLSPECTOR_LLM_COMMAND             for SKILLSPECTOR_PROVIDER=subprocess
                                             (shell command; prompt via stdin —
                                             e.g. "claude -p", "antigravity ask")
    """
    if verbose:
        set_level("DEBUG")

    resolved_path = Path(input_path).resolve()
    if recursive and resolved_path.is_dir():
        detection = detect_skills(resolved_path, depth=depth)
        if detection.is_multi_skill:
            _scan_multi_skill(detection, format, output, no_llm, yara_rules_dir, verbose, detail)
            return
        if not detection.has_root_skill and len(detection.skills) == 0:
            console.print(
                f"[yellow]Warning:[/yellow] no sub-skills found at depth {depth} under {input_path}.\n"
                f"If skills are nested deeper, try --depth {depth + 1} or --depth {depth + 2}.\n"
                "Falling back to flat scan of the entire directory."
            )
    elif resolved_path.is_dir():
        detection = detect_skills(resolved_path)
        if detection.is_multi_skill:
            console.print(
                f"[yellow]Warning:[/yellow] Found {len(detection.skills)} skills in "
                f"this directory. Use --recursive to scan each independently."
            )

    result = None
    try:
        yara_dir = str(yara_rules_dir.resolve()) if yara_rules_dir else None

        # Auto-discover baseline if not explicitly given
        effective_baseline = baseline
        if effective_baseline is None and auto_baseline:
            auto_bl = _auto_discover_baseline(input_path)
            if auto_bl is not None:
                effective_baseline = auto_bl
                try:
                    _loaded = load_baseline(auto_bl)
                    n = len(_loaded.fingerprints or {}) + len(_loaded.rules or [])
                except Exception:  # noqa: BLE001
                    n = "?"
                console.print(f"Baseline: applying {auto_bl.name} ({n} suppression(s))")

        state = _scan_state(
            input_path,
            format,
            no_llm,
            yara_rules_dir=yara_dir,
            baseline=effective_baseline,
            show_suppressed=show_suppressed,
            include_test_fixtures=include_test_fixtures,
            skip_meta=skip_meta,
            trust_skill_classification=trust_skill_classification,
        )
        if verbose:
            console.print("[dim]Running scan...[/dim]")
        logger.debug(
            "Scan started: input_path=%s, format=%s, use_llm=%s",
            input_path,
            format,
            not no_llm,
        )
        trace_config = _build_trace_config(input_path, format, no_llm)
        result = graph.invoke(state, config=trace_config)

        _write_result(result, output, format)

        if (result.get("risk_score") or 0) > 50:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2) from e
    except Exception as e:
        if verbose:
            console.print_exception()
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2) from e
    finally:
        if result is not None:
            _cleanup_result(result)


def _build_trace_config(input_path: str, format: FormatChoice, no_llm: bool) -> RunnableConfig:
    """Build LangSmith trace config for a scan invocation."""
    env = os.environ.get("ENV", "dev")
    tags = ["skillspector", f"environment:{env}"]
    extra_tags = os.environ.get("LANGCHAIN_TAGS_EXTRA", "")
    tags.extend(t.strip() for t in extra_tags.split(",") if t.strip())
    return {
        "run_name": "skillspector-scan",
        "tags": tags,
        "metadata": {
            "input_path": input_path,
            "use_llm": not no_llm,
            "output_format": format.value,
            "version": __version__,
        },
    }


def _scan_multi_skill(
    detection: MultiSkillDetectionResult,
    format: FormatChoice,
    output: Path | None,
    no_llm: bool,
    yara_rules_dir: Path | None,
    verbose: bool,
    detail: bool = False,
) -> None:
    """Scan each detected sub-skill independently and produce a combined report."""
    skills = detection.skills
    console.print(f"[bold]Multi-skill directory detected:[/bold] {len(skills)} skills found\n")

    results: list[dict[str, object]] = []
    max_score = 0

    for i, skill in enumerate(skills, 1):
        console.print(
            f"  [{i}/{len(skills)}] Scanning [bold]{skill.name}[/bold] ({skill.relative_path}/)"
        )
        yara_dir = str(yara_rules_dir.resolve()) if yara_rules_dir else None
        state = _scan_state(str(skill.path), format, no_llm, yara_rules_dir=yara_dir)
        trace_config = _build_trace_config(str(skill.path), format, no_llm)

        try:
            result = graph.invoke(state, config=trace_config)
            results.append(result)
            score = result.get("risk_score") or 0
            if isinstance(score, int) and score > max_score:
                max_score = score
            severity = result.get("risk_severity") or "LOW"
            console.print(f"         Score: {score}/100 ({severity})\n")
        except Exception as e:
            console.print(f"         [red]Error:[/red] {e}\n")
            results.append({"skill_name": skill.name, "error": str(e)})

    console.print("\n[bold]═══ Multi-Skill Summary ═══[/bold]\n")
    console.print(f"  {'Skill':<30} {'Score':<8} {'Severity':<12} {'Findings':<10}")
    console.print(f"  {'─' * 30} {'─' * 8} {'─' * 12} {'─' * 10}")

    for skill, result in zip(skills, results, strict=True):
        if "error" in result:
            console.print(f"  {skill.name:<30} {'ERROR':<8} {'—':<12} {'—':<10}")
            continue
        score = result.get("risk_score", 0)
        severity = result.get("risk_severity", "LOW")
        filtered = result.get("filtered_findings") or result.get("findings")
        finding_count = len(filtered) if isinstance(filtered, list) else 0
        console.print(f"  {skill.name:<30} {score:<8} {severity:<12} {finding_count:<10}")

    console.print("")

    if output and format == FormatChoice.json:
        # Count by severity across all skills for the summary.
        sev_counts: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        skills_list: list[dict[str, object]] = []
        for skill, result in zip(skills, results, strict=True):
            if "error" in result:
                skills_list.append(
                    {"name": skill.name, "path": skill.relative_path, "error": result["error"]}
                )
                continue
            findings_list = result.get("filtered_findings") or result.get("findings") or []
            for f in findings_list:
                sev = (f.severity if isinstance(f.severity, str) else str(f.severity)).lower()
                if sev in sev_counts:
                    sev_counts[sev] += 1
            entry: dict[str, object] = {
                "name": skill.name,
                "path": skill.relative_path,
                "risk_score": result.get("risk_score", 0),
                "risk_severity": result.get("risk_severity", "LOW"),
                "finding_count": len(findings_list),
            }
            if detail:
                entry["issues"] = [f.to_dict() for f in findings_list if hasattr(f, "to_dict")]
            skills_list.append(entry)

        # `multi_skill`/`skill_count`/`max_risk_score`/`skills` (list) are the
        # original contract — preserved as-is. `summary` and per-skill `issues`
        # are additive so existing consumers keep working unchanged.
        combined: dict[str, object] = {
            "multi_skill": True,
            "skill_count": len(skills),
            "max_risk_score": max_score,
            "summary": {
                "total_skills": len(skills),
                **sev_counts,
            },
            "skills": skills_list,
        }
        Path(output).write_text(json.dumps(combined, indent=2), encoding="utf-8")
        console.print(f"[green]Combined report saved to:[/green] {output}")
    elif output:
        # concatenated non-JSON output: not merged SARIF
        sections = []
        for skill, result in zip(skills, results, strict=True):
            if "error" not in result:
                sections.append(f"--- {skill.relative_path} ---\n\n{_result_body(result)}")
        Path(output).write_text("\n\n".join(sections), encoding="utf-8")
        console.print(f"[green]Combined report saved to:[/green] {output}")

    if max_score > 50:
        raise typer.Exit(code=1)


@app.command()
def mcp(
    transport: Annotated[
        TransportChoice,
        typer.Option(
            "--transport",
            "-t",
            help="Transport: FastMCP stdio for local CLI agents, http for remote/A2A callers.",
            case_sensitive=False,
        ),
    ] = TransportChoice.stdio,
    host: Annotated[
        str,
        typer.Option("--host", help="Host to bind (http transport only)."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", help="Port to bind (http transport only)."),
    ] = 8000,
) -> None:
    """
    Run SkillSpector as an MCP server.

    Exposes a single tool, ``scan_skill``, so any MCP-capable agent (Claude Code,
    Codex CLI, Gemini CLI) or remote runtime can scan a skill and gate installs
    on the verdict.

    Requires the optional mcp extra. Reinstall the GitHub tool package with
    that extra enabled, as shown in the README Quick Start section.

    Examples:

        skillspector mcp                      # FastMCP stdio for local CLI agents
        skillspector mcp --transport http --port 8000
    """
    try:
        from skillspector.mcp_server import run as run_mcp

        run_mcp(transport=transport.value, host=host, port=port)
    except ModuleNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2) from e


def _resolve_baseline_output(input_path: str, explicit_output: Path | None) -> Path:
    """Return the path where the baseline file should be written.

    Priority:
    1. Explicit --output path (always honoured).
    2. <input_path>/.skillspector-baseline.yaml when input_path is a local directory.
    3. CWD/.skillspector-baseline.yaml as a last resort (remote / archive inputs).
    """
    if explicit_output is not None:
        return explicit_output
    candidate = Path(input_path)
    if candidate.is_dir():
        return candidate.resolve() / ".skillspector-baseline.yaml"
    return Path(".skillspector-baseline.yaml")


def _warn_if_overwriting(output: Path) -> None:
    """Print a warning if a baseline file already exists at *output*."""
    if not output.exists():
        return
    try:
        import yaml as _yaml  # noqa: PLC0415

        data = _yaml.safe_load(output.read_text(encoding="utf-8")) or {}
        prior = len(data.get("fingerprints") or []) + len(data.get("rules") or [])
    except Exception:  # noqa: BLE001
        prior = "unknown"
    console.print(
        f"[yellow]Warning:[/yellow] overwriting existing baseline at {output} "
        f"({prior} prior suppression(s))"
    )


@app.command()
def baseline(
    input_path: Annotated[
        str,
        typer.Argument(
            help="Path or URL to scan. Supports: Git URL, file URL, zip file, .md file, or directory.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Where to write the baseline file (YAML; .json extension writes JSON). "
                "Defaults to <target-dir>/.skillspector-baseline.yaml."
            ),
        ),
    ] = None,
    no_llm: Annotated[
        bool,
        typer.Option(
            "--no-llm",
            help="Skip LLM analysis when generating the baseline (static analysis only).",
        ),
    ] = False,
    reason: Annotated[
        str,
        typer.Option(
            "--reason",
            help="Reason recorded for every suppressed finding in the baseline.",
        ),
    ] = "Accepted finding (auto-generated baseline)",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-V", help="Show detailed progress."),
    ] = False,
) -> None:
    """
    Generate a baseline file that suppresses every finding in the current scan.

    Run this once to accept all existing findings, then commit the file and pass
    it to future scans with --baseline so only NEW findings are reported.

    Examples:

        skillspector baseline ./my-skill/
        skillspector baseline ./my-skill/ -o team-baseline.yaml --no-llm
        skillspector scan ./my-skill/ --baseline .skillspector-baseline.yaml
    """
    result = None
    try:
        if verbose:
            set_level("DEBUG")
            console.print("[dim]Scanning to build baseline...[/dim]")
        # output_format is irrelevant here; we consume findings, not report_body.
        state = _scan_state(input_path, FormatChoice.json, no_llm)
        result = graph.invoke(state)
        findings = result.get("filtered_findings") or result.get("findings") or []
        data = build_baseline_dict(findings, reason=reason)
        resolved_output = _resolve_baseline_output(input_path, output)
        _warn_if_overwriting(resolved_output)
        dump_baseline(data, resolved_output)
        console.print(
            f"[green]Wrote baseline with {len(findings)} suppressed finding(s) to:[/green] {resolved_output}"
        )
    except typer.Exit:
        raise
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2) from e
    except Exception as e:
        if verbose:
            console.print_exception()
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2) from e
    finally:
        if result is not None:
            _cleanup_result(result)


if __name__ == "__main__":
    app()
