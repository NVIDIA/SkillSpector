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
import warnings
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from langchain_core.runnables import RunnableConfig
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.tree import Tree

from skillspector import __version__
from skillspector.graph import graph
from skillspector.logging_config import get_logger, set_level
from skillspector.multi_skill import MultiSkillDetectionResult, detect_skills
from skillspector.nodes.analyzers import ANALYZER_NODE_IDS
from skillspector.suppression import build_baseline_dict, dump_baseline, load_baseline

logger = get_logger(__name__)

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


def _scan_state(
    input_path: str,
    format: FormatChoice,
    no_llm: bool,
    yara_rules_dir: str | None = None,
    baseline: Path | None = None,
    show_suppressed: bool = False,
) -> dict[str, object]:
    """Build initial graph state from scan CLI args."""
    state: dict[str, object] = {
        "input_path": input_path,
        "output_format": format.value,
        "use_llm": not no_llm,
    }
    if yara_rules_dir is not None:
        state["yara_rules_dir"] = yara_rules_dir
    if baseline is not None:
        # Loading may raise FileNotFoundError/ValueError, mapped to exit code 2 by scan().
        state["baseline"] = load_baseline(baseline)
        state["show_suppressed"] = show_suppressed
    return state


def _write_result(
    result: dict[str, object],
    output: Path | None,
    format: FormatChoice,
) -> None:
    """Write report_body to file or stdout. Uses sarif_report if report_body missing."""
    report_body = result.get("report_body") or ""
    if not report_body and result.get("sarif_report") is not None:
        report_body = json.dumps(result["sarif_report"], indent=2)
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
            help="Scan directories containing multiple skills (immediate subdirectories with SKILL.md) independently.",
        ),
    ] = False,
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
) -> None:
    """
    Scan a skill for security vulnerabilities.

    Examples:

        skillspector scan ./my-skill/
        skillspector scan ./my-skill/ --format json --output report.json
        skillspector scan https://github.com/user/my-skill --no-llm
        skillspector scan ./skill-collection/ --recursive

    Environment variables:

        SKILLSPECTOR_PROVIDER  Active LLM provider: openai | anthropic |
                               nv_build | nv_inference. Defaults to the
                               NVIDIA path (nv_inference, falling back to
                               nv_build in OSS builds).
        SKILLSPECTOR_MODEL     Override the active provider's default
                               model (applies to every analyzer slot).
        SKILLSPECTOR_LOG_LEVEL DEBUG | INFO | WARNING | ERROR (default WARNING).

    Provider credentials (one of):

        OPENAI_API_KEY [+ OPENAI_BASE_URL]   for SKILLSPECTOR_PROVIDER=openai
        ANTHROPIC_API_KEY                    for SKILLSPECTOR_PROVIDER=anthropic
        NVIDIA_INFERENCE_KEY                 for the NVIDIA providers
    """
    if verbose:
        set_level("DEBUG")

    resolved_path = Path(input_path).resolve()
    if recursive and resolved_path.is_dir():
        detection = detect_skills(resolved_path)
        if detection.is_multi_skill:
            _scan_multi_skill(detection, format, output, no_llm, yara_rules_dir, verbose)
            return
        if not detection.has_root_skill and len(detection.skills) == 0:
            console.print(
                "[yellow]Warning:[/yellow] --recursive specified but no sub-skills "
                "detected. Scanning as single skill."
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
        state = _scan_state(
            input_path,
            format,
            no_llm,
            yara_rules_dir=yara_dir,
            baseline=baseline,
            show_suppressed=show_suppressed,
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
        if verbose:
            result = graph.invoke(state, config=trace_config)
        else:
            total_steps = 4 + len(ANALYZER_NODE_IDS)
            result = dict(state)

            # Use stderr for progress so stdout remains clean for structured outputs
            err_console = Console(stderr=True)

            # Suppress noisy Pydantic serialization warnings scoped to the graph run
            with warnings.catch_warnings(), Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=err_console,
                transient=True,
            ) as progress:
                warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
                task_id = progress.add_task("Resolving input...", total=total_steps)

                num_files = 0
                analyzers_done = 0
                total_analyzers = len(ANALYZER_NODE_IDS)

                for update in graph.stream(state, config=trace_config, stream_mode="updates"):
                    for node_name, node_output in update.items():
                        progress.advance(task_id)

                        # Accumulate scalar outputs needed by the CLI (report_body, risk_score, temp_dir, sarif_report)
                        if "temp_dir_for_cleanup" in node_output:
                            result["temp_dir_for_cleanup"] = node_output["temp_dir_for_cleanup"]
                        if "report_body" in node_output:
                            result["report_body"] = node_output["report_body"]
                        if "sarif_report" in node_output:
                            result["sarif_report"] = node_output["sarif_report"]
                        if "risk_score" in node_output:
                            result["risk_score"] = node_output["risk_score"]

                        # Update UI text based on graph progression
                        if node_name == "resolve_input":
                            progress.update(task_id, description="Building context...")
                        elif node_name == "build_context":
                            components = node_output.get("components", [])
                            num_files = len(components)
                            progress.update(task_id, description=f"Analyzing {num_files} files (0/{total_analyzers} rules applied)...")

                            # Print a proper report of the files and directories being scanned
                            tree = Tree("[bold blue]Discovered Files to Scan[/bold blue]")
                            nodes = {"": tree}
                            for path in sorted(components):
                                parts = Path(path).parts
                                current = ""
                                for part in parts:
                                    parent = current
                                    current = f"{current}/{part}" if current else part
                                    if current not in nodes:
                                        is_file = current == path
                                        icon = "📄 " if is_file else "📁 "
                                        style = "green" if is_file else "cyan"
                                        nodes[current] = nodes[parent].add(f"[{style}]{icon}{part}[/{style}]")

                            err_console.print(tree)
                            err_console.print()

                        elif node_name in ANALYZER_NODE_IDS:
                            analyzers_done += 1
                            progress.update(task_id, description=f"Analyzing {num_files} files ({analyzers_done}/{total_analyzers} rules applied)...")
                            # Print which rule just finished above the progress bar
                            err_console.print(f"[dim]✔ Rule completed: {node_name}[/dim]")
                        elif node_name == "meta_analyzer":
                            progress.update(task_id, description="Generating report...")
                            err_console.print("[dim]✔ Rule completed: meta_analyzer (filtering findings)[/dim]")

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
        combined = {
            "multi_skill": True,
            "skill_count": len(skills),
            "max_risk_score": max_score,
            "skills": [],
        }
        for skill, result in zip(skills, results, strict=True):
            if "error" in result:
                combined["skills"].append({"name": skill.name, "error": result["error"]})
            else:
                combined["skills"].append(
                    {
                        "name": skill.name,
                        "path": skill.relative_path,
                        "risk_score": result.get("risk_score", 0),
                        "risk_severity": result.get("risk_severity", "LOW"),
                        "finding_count": len(
                            result.get("filtered_findings") or result.get("findings") or []
                        ),
                    }
                )
        Path(output).write_text(json.dumps(combined, indent=2), encoding="utf-8")
        console.print(f"[green]Combined report saved to:[/green] {output}")
    elif output:
        for _skill, result in zip(skills, results, strict=True):
            if "error" not in result:
                _write_result(result, None, format)

    if max_score > 50:
        raise typer.Exit(code=1)


@app.command()
def mcp(
    transport: Annotated[
        TransportChoice,
        typer.Option(
            "--transport",
            "-t",
            help="Transport: stdio for local CLI agents, http for remote/A2A callers.",
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

    Examples:

        skillspector mcp                      # stdio (local agents)
        skillspector mcp --transport http --port 8000

    Requires the optional ``mcp`` dependency: pip install "skillspector[mcp]".
    """
    try:
        from skillspector.mcp_server import run as run_mcp

        run_mcp(transport=transport.value, host=host, port=port)
    except ModuleNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2) from e


@app.command()
def baseline(
    input_path: Annotated[
        str,
        typer.Argument(
            help="Path or URL to scan. Supports: Git URL, file URL, zip file, .md file, or directory.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Where to write the baseline file (YAML; .json extension writes JSON).",
        ),
    ] = Path(".skillspector-baseline.yaml"),
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
        dump_baseline(data, output)
        console.print(
            f"[green]Wrote baseline with {len(findings)} suppressed finding(s) to:[/green] {output}"
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
