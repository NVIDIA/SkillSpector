"""
SkillSpector CLI - Command-line interface for scanning agent skills.

Usage:
    skillspector scan <input> [--format FORMAT] [--output FILE] [--no-llm]
    skillspector --version
    skillspector --help
"""

import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from skillspector import __version__
from skillspector.scanner import SkillScanner
from skillspector.report import ReportGenerator, OutputFormat

# Initialize Typer app
app = typer.Typer(
    name="skillspector",
    help="Security scanner for AI agent skills. Detect vulnerabilities before installation.",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


class FormatChoice(str, Enum):
    """Output format choices for the CLI."""
    terminal = "terminal"
    json = "json"
    markdown = "markdown"


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"SkillSpector v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
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
    SkillSpector - Security scanner for AI agent skills.

    Analyze skill bundles (prompts, code, configurations) to detect
    vulnerabilities, malicious patterns, and security risks before installation.

    Based on research finding that 26.1% of skills contain vulnerabilities.
    """
    pass


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
        Optional[Path],
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
            help="Skip LLM analysis (faster, but less accurate). Uses static analysis only.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-V",
            help="Show detailed progress and debug information.",
        ),
    ] = False,
) -> None:
    """
    Scan a skill for security vulnerabilities.

    Examples:

        # Scan a local directory
        skillspector scan ./my-skill/

        # Scan a single SKILL.md file
        skillspector scan ./SKILL.md

        # Scan a zip file
        skillspector scan ./my-skill.zip

        # Scan a Git repository
        skillspector scan https://github.com/user/my-skill

        # Output as JSON
        skillspector scan ./my-skill/ --format json --output report.json

        # Quick scan without LLM (static analysis only)
        skillspector scan ./my-skill/ --no-llm
    """
    try:
        # Initialize scanner
        scanner = SkillScanner(use_llm=not no_llm, verbose=verbose)

        # Show progress
        if format == FormatChoice.terminal:
            with console.status("[bold blue]Scanning skill...[/bold blue]"):
                result = scanner.scan(input_path)
        else:
            result = scanner.scan(input_path)

        # Generate report
        report_gen = ReportGenerator()
        output_format = OutputFormat(format.value)

        if output:
            # Write to file
            report_gen.write_to_file(result, output, output_format)
            if format == FormatChoice.terminal:
                console.print(f"\n[green]Report saved to:[/green] {output}")
        else:
            # Print to stdout
            report_content = report_gen.generate(result, output_format)
            if format == FormatChoice.terminal:
                console.print(report_content)
            else:
                print(report_content)

        # Exit with appropriate code
        if result.risk_assessment.score > 50:
            raise typer.Exit(code=1)  # High risk

    except typer.Exit:
        raise  # Re-raise typer.Exit without catching it
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2)
    except Exception as e:
        if verbose:
            console.print_exception()
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2)


@app.command()
def patterns() -> None:
    """
    List all vulnerability patterns that SkillSpector detects.
    """
    from rich.table import Table

    table = Table(title="SkillSpector Vulnerability Patterns (15 total)")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Category", style="magenta")
    table.add_column("Pattern", style="green")
    table.add_column("Severity", style="red")
    table.add_column("Description")

    patterns_list = [
        # Prompt Injection (5 patterns)
        ("P1", "Prompt Injection", "Instruction Override", "HIGH",
         "Explicit commands to ignore user/system constraints"),
        ("P2", "Prompt Injection", "Hidden Instructions", "HIGH",
         "Malicious directives in comments or invisible text"),
        ("P3", "Prompt Injection", "Exfiltration Commands", "HIGH",
         "Instructions directing agent to transmit context externally"),
        ("P4", "Prompt Injection", "Behavior Manipulation", "MEDIUM",
         "Subtle instructions altering agent decision-making"),
        ("P5", "Prompt Injection", "Harmful Content Injection", "CRITICAL",
         "Instructions that could cause physical harm"),

        # Data Exfiltration (4 patterns)
        ("E1", "Data Exfiltration", "External Transmission", "MEDIUM",
         "Sending data to hardcoded external URLs"),
        ("E2", "Data Exfiltration", "Env Variable Harvesting", "HIGH",
         "Collecting environment variables (API keys, secrets)"),
        ("E3", "Data Exfiltration", "File System Enumeration", "MEDIUM",
         "Scanning directories for sensitive files"),
        ("E4", "Data Exfiltration", "Context Leakage", "HIGH",
         "Transmitting agent conversation context externally"),

        # Privilege Escalation (3 patterns)
        ("PE1", "Privilege Escalation", "Excessive Permissions", "LOW",
         "Requesting access scope beyond stated functionality"),
        ("PE2", "Privilege Escalation", "Sudo/Root Execution", "MEDIUM",
         "Invoking elevated system privileges"),
        ("PE3", "Privilege Escalation", "Credential Access", "HIGH",
         "Reading SSH keys, tokens, password files"),

        # Supply Chain (3 patterns)
        ("SC1", "Supply Chain", "Unpinned Dependencies", "LOW",
         "No version constraints allowing malicious updates"),
        ("SC2", "Supply Chain", "External Script Fetching", "HIGH",
         "curl | bash and similar remote code execution"),
        ("SC3", "Supply Chain", "Obfuscated Code", "HIGH",
         "Base64/hex encoded execution hiding functionality"),
    ]

    for p in patterns_list:
        table.add_row(*p)

    console.print(table)


if __name__ == "__main__":
    app()
