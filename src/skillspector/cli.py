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
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from langchain_core.runnables import RunnableConfig
from rich.console import Console

from skillspector import __version__
from skillspector.graph import graph
from skillspector.logging_config import get_logger, set_level

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
) -> dict[str, object]:
    """Build initial graph state from scan CLI args."""
    state: dict[str, object] = {
        "input_path": input_path,
        "output_format": format.value,
        "use_llm": not no_llm,
    }
    if yara_rules_dir is not None:
        state["yara_rules_dir"] = yara_rules_dir
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
    result = None
    try:
        yara_dir = str(yara_rules_dir.resolve()) if yara_rules_dir else None
        state = _scan_state(input_path, format, no_llm, yara_rules_dir=yara_dir)
        if verbose:
            set_level("DEBUG")
            console.print("[dim]Running scan...[/dim]")
        logger.debug(
            "Scan started: input_path=%s, format=%s, use_llm=%s",
            input_path,
            format,
            not no_llm,
        )
        env = os.environ.get("ENV", "dev")
        tags = ["skillspector", f"environment:{env}"]
        extra_tags = os.environ.get("LANGCHAIN_TAGS_EXTRA", "")
        tags.extend(t.strip() for t in extra_tags.split(",") if t.strip())
        trace_config: RunnableConfig = {
            "run_name": "skillspector-scan",
            "tags": tags,
            "metadata": {
                "input_path": input_path,
                "use_llm": not no_llm,
                "output_format": format.value,
                "version": __version__,
            },
        }
        if verbose:
            result = graph.invoke(state, config=trace_config)
        else:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
            from rich.console import Console
            from skillspector.nodes.analyzers import ANALYZER_NODE_IDS
            import warnings

            # Suppress noisy Pydantic serialization warnings during structured LLM output
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

            total_steps = 4 + len(ANALYZER_NODE_IDS)
            result = dict(state)
            
            # Use stderr for progress so stdout remains clean for structured outputs
            err_console = Console(stderr=True)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=err_console,
                transient=True,
            ) as progress:
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
                            from rich.tree import Tree
                            from pathlib import Path
                            
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


if __name__ == "__main__":
    app()
