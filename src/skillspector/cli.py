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
import sys
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Annotated, cast

import typer
from langchain_core.runnables import RunnableConfig
from rich.console import Console

from skillspector import __version__, transitive
from skillspector.cleanup import cleanup_result
from skillspector.constants import RISK_THRESHOLD
from skillspector.graph import graph
from skillspector.input_handler import validate_local_input_path
from skillspector.inspection_ledger import (
    MAX_INSPECTION_LEDGER_EVENTS,
    LedgerOutcome,
    LedgerReason,
    LedgerRecordType,
    finalize_ledger,
    inspection_work_id,
    ledger_event,
)
from skillspector.logging_config import get_logger, set_level
from skillspector.mcp_registry import scan_registry
from skillspector.models import Finding
from skillspector.multi_skill import MultiSkillDetectionResult, SkillDirectory, detect_skills
from skillspector.nodes.report import report
from skillspector.sarif_models import SARIF_SCHEMA_URI, validate_sarif_report
from skillspector.state import MAX_WORKFLOW_BYTES
from skillspector.suppression import (
    Baseline,
    build_baseline_dict,
    discover_baseline,
    dump_baseline,
    effective_findings,
    load_baseline,
)

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
err_console = Console(stderr=True)

_TRANSITIVE_MAX_TARGETS = 32
_TRANSITIVE_MAX_BYTES = 10 * 1024 * 1024
_TRANSITIVE_MAX_SECONDS = 60.0
_TRANSITIVE_MAX_ARTIFACTS = 10_000
_TRANSITIVE_MAX_FINDINGS = 10_000
_TRANSITIVE_MAX_COMPONENTS = 10_000
_TRANSITIVE_MAX_STATUS_EVENTS = 10_000
_TRANSITIVE_MAX_REFERENCES = 10_000
_MULTI_SKILL_MAX_SKILLS = 32
_MULTI_SKILL_MAX_PUBLIC_RECORDS = 10_000
_MULTI_SKILL_MAX_REPORT_CHARACTERS = 4 * 1024 * 1024


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


@dataclass(slots=True)
class _TransitiveBudget:
    max_targets: int = _TRANSITIVE_MAX_TARGETS
    max_bytes: int = _TRANSITIVE_MAX_BYTES
    max_seconds: float = _TRANSITIVE_MAX_SECONDS
    max_artifacts: int = _TRANSITIVE_MAX_ARTIFACTS
    max_findings: int = _TRANSITIVE_MAX_FINDINGS
    max_components: int = _TRANSITIVE_MAX_COMPONENTS
    max_ledger_events: int = MAX_INSPECTION_LEDGER_EVENTS
    max_status_events: int = _TRANSITIVE_MAX_STATUS_EVENTS
    max_references: int = _TRANSITIVE_MAX_REFERENCES


@dataclass(slots=True)
class _CachedTransitiveResult:
    source_url: str
    source_identity: str
    source_digest: str
    filtered_findings: list[Finding]
    findings: list[Finding]
    effective_finding_ids: list[str]
    inspection_ledger: list[dict[str, object]]
    analyzer_status_events: list[dict[str, object]]
    llm_call_log: list[dict[str, object]]
    inference_usage: list[dict[str, object]]
    components: list[str]
    component_metadata: list[dict[str, object]]
    file_cache: dict[str, str]
    local_file_cache: dict[str, str]
    artifact_inventory: list[dict[str, object]]
    artifact_references: list[dict[str, object]]
    has_executable_scripts: bool
    refs: list[str]


@dataclass(slots=True)
class _TransitiveTraversalState:
    cache: dict[str, _CachedTransitiveResult] = field(default_factory=dict)
    budget: _TransitiveBudget = field(default_factory=_TransitiveBudget)
    started_at: float | None = None
    scanned_targets: int = 0
    scanned_bytes: int = 0
    scanned_artifacts: int = 0
    truncation_reasons: list[str] = field(default_factory=list)
    budget_exhausted: bool = False
    paused_at: float | None = None

    def note_truncation(self, reason: str) -> None:
        if len(self.truncation_reasons) >= 256:
            sentinel = "additional transitive limitations omitted"
            if self.truncation_reasons[-1] != sentinel:
                self.truncation_reasons[-1] = sentinel
            self.budget_exhausted = True
            return
        if reason not in self.truncation_reasons:
            self.truncation_reasons.append(reason)
        if "budget" in reason or "time budget" in reason:
            self.budget_exhausted = True

    def note_child_scan_failure(self, target: str) -> None:
        self.note_truncation(f"transitive child scan failed for {target}")

    def _ensure_started(self) -> None:
        if self.started_at is None:
            self.started_at = monotonic()

    def can_scan_more(self) -> bool:
        self._ensure_started()
        if self.budget_exhausted:
            return False
        if self.scanned_targets >= self.budget.max_targets:
            self.note_truncation(f"target budget {self.budget.max_targets} reached")
            return False
        if self.remaining_bytes() <= 0:
            self.note_truncation(f"byte budget {self.budget.max_bytes} reached")
            return False
        if self.remaining_artifacts() <= 0:
            self.note_truncation(f"artifact budget {self.budget.max_artifacts} reached")
            return False
        if self.remaining_seconds() <= 0:
            self.note_truncation(f"time budget {self.budget.max_seconds:.0f}s reached")
            return False
        return True

    def record_scan(self) -> None:
        self._ensure_started()
        self.scanned_targets += 1
        if self.remaining_bytes() <= 0:
            self.note_truncation(f"byte budget {self.budget.max_bytes} reached")
        if self.remaining_seconds() <= 0:
            self.note_truncation(f"time budget {self.budget.max_seconds:.0f}s reached")

    def record_bytes(self, bytes_scanned: int) -> None:
        self._ensure_started()
        self.scanned_bytes += max(0, bytes_scanned)

    def remaining_seconds(self) -> float:
        self._ensure_started()
        assert self.started_at is not None
        end = self.paused_at if self.paused_at is not None else monotonic()
        return max(0.0, self.budget.max_seconds - (end - self.started_at))

    def remaining_bytes(self) -> int:
        return max(0, self.budget.max_bytes - self.scanned_bytes)

    def remaining_artifacts(self) -> int:
        return max(0, self.budget.max_artifacts - self.scanned_artifacts)

    def record_artifacts(self, artifacts: int) -> None:
        self._ensure_started()
        self.scanned_artifacts += max(0, artifacts)
        if self.scanned_artifacts > self.budget.max_artifacts:
            self.note_truncation(f"artifact budget {self.budget.max_artifacts} reached")

    def pause_deadline(self) -> None:
        if self.started_at is not None and self.paused_at is None:
            self.paused_at = monotonic()

    def resume_deadline(self) -> None:
        if self.paused_at is not None:
            assert self.started_at is not None
            self.started_at += monotonic() - self.paused_at
            self.paused_at = None


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
        state["baseline_path"] = os.path.abspath(baseline.expanduser())
        state["show_suppressed"] = show_suppressed
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


def _recursive_json_payload(result: dict[str, object]) -> dict[str, object] | None:
    """Return parsed report_body when it is valid JSON object text."""
    raw_report_body = result.get("report_body")
    if not isinstance(raw_report_body, str):
        return None

    try:
        parsed = json.loads(raw_report_body)
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _multi_skill_limitation_events(
    detection: MultiSkillDetectionResult,
) -> list[dict[str, object]]:
    """Project bounded pre-scan discovery limits into the canonical ledger."""
    events: list[dict[str, object]] = []
    for limitation in detection.limitations[:MAX_INSPECTION_LEDGER_EVENTS]:
        raw_reason = str(limitation.reason_code)
        try:
            reason = LedgerReason(raw_reason)
        except ValueError:
            reason = LedgerReason.READ_ERROR
        events.append(
            dict(
                ledger_event(
                    outcome=LedgerOutcome.PARTIAL,
                    record_type=LedgerRecordType.SYSTEM,
                    phase="multi_skill_discovery",
                    path="SKILL.md",
                    reason=reason,
                    stage=str(limitation.resource),
                    observed_characters=limitation.observed_characters,
                    limit_characters=limitation.limit_characters,
                    observed_bytes=limitation.observed_bytes,
                    limit_bytes=limitation.limit_bytes,
                    observed_artifacts=limitation.observed_artifacts,
                    limit_artifacts=limitation.limit_artifacts,
                    observed_depth=limitation.observed_depth,
                    limit_depth=limitation.limit_depth,
                    observed_seconds=limitation.observed_seconds,
                    limit_seconds=limitation.limit_seconds,
                )
            )
        )
    return events


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
    use_shipped_baseline: Annotated[
        bool,
        typer.Option(
            "--use-shipped-baseline",
            help="Apply a baseline shipped at the top level of the scanned skill "
            "directory (.skillspector-baseline.yaml). Off by default: a skill "
            "author's baseline can suppress findings in your scan, so a discovered "
            "baseline is only reported until you opt in. Ignored when --baseline "
            "is given.",
        ),
    ] = False,
    transitive_enabled: Annotated[
        bool,
        typer.Option(
            "--transitive",
            help="Follow transitive external references after the initial scan.",
        ),
    ] = False,
    transitive_depth: Annotated[
        int,
        typer.Option(
            "--transitive-depth",
            help="Maximum transitive depth to scan for external references.",
        ),
    ] = 1,
    transitive_allow_prefix: Annotated[
        list[str] | None,
        typer.Option(
            "--transitive-allow-prefix",
            help=(
                "Only scan transitive targets matching at least one canonical prefix. Repeatable."
            ),
        ),
    ] = None,
    transitive_deny_prefix: Annotated[
        list[str] | None,
        typer.Option(
            "--transitive-deny-prefix",
            help=("Skip transitive targets matching any canonical prefix. Repeatable."),
        ),
    ] = None,
    fail_on_incomplete: Annotated[
        bool,
        typer.Option(
            "--fail-on-incomplete",
            help="Exit 1 when relevant analysis is partial or incomplete.",
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
    mcp_registry: Annotated[
        bool,
        typer.Option(
            "--mcp-registry",
            help="Scan an MCP Registry payload or URL instead of a skill.",
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
                               anthropic_proxy | bedrock | nv_build |
                               nv_inference. Defaults to the NVIDIA path
                               (nv_inference, falling back to nv_build in
                               OSS builds).
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
    """
    if mcp_registry:
        if recursive or baseline is not None or show_suppressed or yara_rules_dir is not None:
            err_console.print(
                "[red]Error:[/red] --mcp-registry cannot be combined with "
                "--recursive, --baseline, --show-suppressed, or --yara-rules-dir"
            )
            raise typer.Exit(code=2)
        if format != FormatChoice.json:
            err_console.print(
                "[red]Error:[/red] --mcp-registry currently supports only --format json"
            )
            raise typer.Exit(code=2)
        try:
            result = scan_registry(input_path)
            report = json.dumps(result, indent=2)
            if output:
                output.write_text(report, encoding="utf-8")
                console.print(f"Report saved to: {output}")
            else:
                print(report)
            if result["risk_score"] > RISK_THRESHOLD:
                raise typer.Exit(code=1)
        except typer.Exit:
            raise
        except Exception as e:
            err_console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=2) from e
        return

    if verbose:
        set_level("DEBUG")

    resolved_path = Path(input_path)
    if not input_path.startswith(("http://", "https://", "git@")):
        try:
            resolved_path = validate_local_input_path(resolved_path)
        except ValueError as e:
            err_console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=2) from e
    try:
        transitive_allow_prefix, transitive_deny_prefix = transitive.normalize_prefixes(
            transitive_allow_prefix, transitive_deny_prefix
        )
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] invalid transitive prefix: {exc}")
        raise typer.Exit(code=2) from exc
    yara_dir = str(yara_rules_dir.resolve()) if yara_rules_dir else None
    pre_scan_ledger_events: list[dict[str, object]] = []
    if recursive and resolved_path.is_dir():
        detection = detect_skills(resolved_path)
        if not detection.complete:
            pre_scan_ledger_events = _multi_skill_limitation_events(detection)
            err_console.print(
                "[yellow]Warning:[/yellow] Recursive skill discovery was incomplete; "
                "continuing with a bounded scan and reporting partial coverage."
            )
        if detection.is_multi_skill:
            if baseline is not None:
                err_console.print(
                    "[red]Error:[/red] --baseline is not supported for recursive "
                    "multi-skill scans; scan each sub-skill with its own baseline"
                )
                raise typer.Exit(code=2)
            _scan_multi_skill(
                detection,
                format=format,
                output=output,
                no_llm=no_llm,
                baseline=baseline,
                show_suppressed=show_suppressed,
                transitive_enabled=transitive_enabled,
                transitive_depth=transitive_depth,
                transitive_allow_prefix=transitive_allow_prefix,
                transitive_deny_prefix=transitive_deny_prefix,
                yara_dir=yara_dir,
                verbose=verbose,
                fail_on_incomplete=fail_on_incomplete,
            )
            return
        if detection.complete and not detection.has_root_skill and len(detection.skills) == 0:
            console.print(
                "[yellow]Warning:[/yellow] --recursive specified but no sub-skills "
                "detected. Scanning as single skill."
            )
    elif resolved_path.is_dir():
        detection = detect_skills(resolved_path)
        if not detection.complete:
            pre_scan_ledger_events = _multi_skill_limitation_events(detection)
            err_console.print(
                "[yellow]Warning:[/yellow] Skill discovery was incomplete; continuing "
                "with a bounded scan and reporting partial coverage."
            )
        if detection.is_multi_skill:
            console.print(
                f"[yellow]Warning:[/yellow] Found {len(detection.skills)} skills in "
                f"this directory. Use --recursive to scan each independently."
            )

    shipped: Path | None = None
    if baseline is None and resolved_path.is_dir():
        shipped = discover_baseline(resolved_path)
    if shipped is not None:
        if use_shipped_baseline:
            baseline = shipped
            err_console.print(f"[yellow]Applying author-shipped baseline:[/yellow] {shipped}")
            err_console.print(
                "[dim]Suppressed findings do not count toward the risk score; "
                "use --show-suppressed to list them.[/dim]"
            )
        else:
            err_console.print(
                f"[yellow]Shipped baseline detected (not applied):[/yellow] {shipped}"
            )
            err_console.print(
                "[dim]Review it, then re-run with --use-shipped-baseline to apply its "
                "suppressions. Findings and risk score are unaffected until you opt in.[/dim]"
            )
    elif use_shipped_baseline and baseline is None:
        err_console.print(
            f"[dim]--use-shipped-baseline: no shipped baseline found in {resolved_path}; "
            "scanning without a baseline.[/dim]"
        )

    result = None
    try:
        result = _scan_skill(
            input_path=input_path,
            format=format,
            no_llm=no_llm,
            baseline=baseline,
            yara_rules_dir=Path(yara_dir) if yara_dir else None,
            verbose=verbose,
            show_suppressed=show_suppressed,
            transitive_enabled=transitive_enabled,
            transitive_depth=transitive_depth,
            transitive_allow_prefix=transitive_allow_prefix,
            transitive_deny_prefix=transitive_deny_prefix,
            pre_scan_ledger_events=pre_scan_ledger_events,
        )
        _write_result(result, output, format)

        if result.get("execution_successful") is False:
            raise typer.Exit(code=2)
        completeness_value = result.get("analysis_completeness")
        is_complete = (
            bool(completeness_value.get("is_complete", True))
            if isinstance(completeness_value, dict)
            else True
        )
        if fail_on_incomplete and not is_complete:
            raise typer.Exit(code=1)
        if (result.get("risk_score") or 0) > RISK_THRESHOLD:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except (FileNotFoundError, ValueError) as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2) from e
    except Exception as e:
        if verbose:
            err_console.print_exception()
        else:
            err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2) from e
    finally:
        if result is not None:
            cleanup_result(result)


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


def _coerce_str_path_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _coerce_findings_list(value: object) -> list[Finding]:
    if not isinstance(value, list):
        return []
    return [finding for finding in value if isinstance(finding, Finding)]


def _coerce_finding_ids(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _effective_finding_ids(result: dict[str, object]) -> list[str]:
    effective_ids = _coerce_finding_ids(result.get("effective_finding_ids"))
    if effective_ids:
        return effective_ids
    return [
        finding.finding_id for finding in _coerce_findings_list(result.get("filtered_findings"))
    ]


def _coerce_llm_call_log(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _coerce_dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _coerce_raw_file_cache(value: object) -> dict[str, bytes]:
    if not isinstance(value, dict):
        return {}
    return {
        str(path): content
        for path, content in value.items()
        if isinstance(path, str) and isinstance(content, bytes)
    }


def _source_content_digest(
    raw_file_cache: dict[str, bytes], local_file_cache: dict[str, str]
) -> str:
    """Hash the exact bounded child snapshot without constructing a combined payload."""
    digest = sha256()
    digest.update(b"skillspector-transitive-source-v1\0")
    if raw_file_cache:
        records = ((path, content) for path, content in sorted(raw_file_cache.items()))
    else:
        records = (
            (path, content.encode("utf-8", errors="replace"))
            for path, content in sorted(local_file_cache.items())
        )
    for path, content in records:
        encoded_path = path.encode("utf-8", errors="replace")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return f"sha256:{digest.hexdigest()}"


def _source_identity(target: str, source_digest: str) -> str:
    digest = sha256()
    digest.update(b"skillspector-transitive-identity-v1\0")
    digest.update(target.encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(source_digest.encode())
    return f"external/{digest.hexdigest()}"


def _scoped_finding_id(source_identity: str, finding_id: str) -> str:
    digest = sha256(f"{source_identity}\x1f{finding_id}".encode()).hexdigest()
    return f"finding-{digest}"


def _ledger_work_identity(entry: dict[str, object]) -> str:
    analyzer_id = entry.get("analyzer_id")
    if isinstance(analyzer_id, str) and analyzer_id:
        return analyzer_id
    record_type = entry.get("record_type", LedgerRecordType.WORK_ITEM)
    record_value = getattr(record_type, "value", record_type)
    return f"{record_value}:{entry.get('phase', '')}"


def _source_aware_ledger(
    value: object,
    *,
    source_url: str,
    source_identity: str,
    source_digest: str,
    finding_id_map: dict[str, str],
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for event in _coerce_dict_list(value):
        entry = dict(event)
        path = entry.get("path")
        if isinstance(path, str) and path:
            entry["path"] = _transitive_component_key(source_identity, path)
        entry["source_url"] = source_url
        entry["source_identity"] = source_identity
        entry["source_digest"] = source_digest
        for id_field in ("input_finding_ids", "emitted_finding_ids"):
            ids = entry.get(id_field)
            if isinstance(ids, list):
                entry[id_field] = [
                    finding_id_map.get(str(item), _scoped_finding_id(source_identity, str(item)))
                    for item in ids
                    if isinstance(item, str)
                ]
        scoped_path = str(entry.get("path", "SKILL.md"))
        start_line = entry.get("start_line")
        end_line = entry.get("end_line")
        entry["work_id"] = inspection_work_id(
            _ledger_work_identity(entry),
            scoped_path,
            start_line if isinstance(start_line, int) else None,
            end_line if isinstance(end_line, int) else None,
        )
        events.append(entry)
    return events


def _source_aware_status_events(
    value: object,
    *,
    source_url: str,
    source_identity: str,
    source_digest: str,
    retained_work_ids: set[str],
    max_planned_work: int,
) -> list[dict[str, object]]:
    statuses: list[dict[str, object]] = []
    planned_retained = 0
    for status in _coerce_dict_list(value):
        if len(statuses) >= _TRANSITIVE_MAX_STATUS_EVENTS:
            break
        entry = dict(status)
        analyzer_id = str(entry.get("analyzer_id", ""))
        entry["source_url"] = source_url
        entry["source_identity"] = source_identity
        entry["source_digest"] = source_digest
        planned_work = entry.get("planned_work")
        if isinstance(planned_work, list):
            scoped_work: list[dict[str, object]] = []
            for target in planned_work:
                if planned_retained >= max(0, max_planned_work):
                    break
                if not isinstance(target, dict):
                    continue
                scoped_target = dict(target)
                path = scoped_target.get("path")
                if isinstance(path, str) and path:
                    scoped_target["path"] = _transitive_component_key(source_identity, path)
                start_line = scoped_target.get("start_line")
                end_line = scoped_target.get("end_line")
                scoped_target["work_id"] = inspection_work_id(
                    analyzer_id,
                    str(scoped_target.get("path", "SKILL.md")),
                    start_line if isinstance(start_line, int) else None,
                    end_line if isinstance(end_line, int) else None,
                )
                if scoped_target["work_id"] not in retained_work_ids:
                    continue
                scoped_work.append(scoped_target)
                planned_retained += 1
            entry["planned_work"] = scoped_work
        statuses.append(entry)
    return statuses


def _coerce_file_cache(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(path): content
        for path, content in value.items()
        if isinstance(path, str) and isinstance(content, str)
    }


def _transitive_component_key(source_identity: str | None, path: str) -> str:
    if not source_identity:
        return path
    normalized = path.replace("\\", "/").lstrip("/") or "SKILL.md"
    return f"{source_identity}/{normalized}"


def _decorate_component_metadata(
    metadata: list[dict[str, object]],
    source_identity: str | None,
    *,
    source_url: str | None = None,
    source_digest: str | None = None,
) -> list[dict[str, object]]:
    decorated: list[dict[str, object]] = []
    for item in metadata:
        path = str(item.get("path", ""))
        entry = {**item, "coverage_key": _transitive_component_key(source_identity, path)}
        if source_url:
            entry["source_url"] = source_url
        if source_identity:
            entry["source_identity"] = source_identity
        if source_digest:
            entry["source_digest"] = source_digest
        decorated.append(entry)
    return decorated


def _source_aware_components(paths: list[str], source_identity: str | None) -> list[str]:
    return [_transitive_component_key(source_identity, path) for path in paths]


def _source_aware_file_cache(
    file_cache: dict[str, str], source_identity: str | None
) -> dict[str, str]:
    return {
        _transitive_component_key(source_identity, path): content
        for path, content in file_cache.items()
    }


def _source_aware_inventory(
    value: object,
    *,
    source_url: str,
    source_identity: str,
    source_digest: str,
) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for item in _coerce_dict_list(value)[:_TRANSITIVE_MAX_COMPONENTS]:
        entry = dict(item)
        entry["path"] = _transitive_component_key(
            source_identity, str(entry.get("path", "SKILL.md"))
        )
        entry["source_url"] = source_url
        entry["source_identity"] = source_identity
        entry["source_digest"] = source_digest
        inventory.append(entry)
    return inventory


def _source_aware_references(
    value: object,
    *,
    source_url: str,
    source_identity: str,
    source_digest: str,
) -> list[dict[str, object]]:
    references: list[dict[str, object]] = []
    for item in _coerce_dict_list(value)[:_TRANSITIVE_MAX_REFERENCES]:
        entry = dict(item)
        for key in ("source_path", "target_path"):
            path = entry.get(key)
            if isinstance(path, str) and path:
                entry[key] = _transitive_component_key(source_identity, path)
        entry["source_url"] = source_url
        entry["source_identity"] = source_identity
        entry["source_digest"] = source_digest
        references.append(entry)
    return references


def _component_identity(item: dict[str, object]) -> str:
    coverage_key = item.get("coverage_key")
    if isinstance(coverage_key, str) and coverage_key:
        return coverage_key
    path = str(item.get("path", ""))
    source_identity = item.get("source_identity")
    return _transitive_component_key(
        source_identity if isinstance(source_identity, str) else None, path
    )


def _merge_unique_component_metadata(items: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in items:
        identity = _component_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(item)
    return merged


def _transitive_limit_event(
    limitation: transitive.TransitiveResourceLimitation,
    *,
    phase: str,
    path: str,
) -> dict[str, object]:
    observed_bytes: int | None = None
    limit_bytes: int | None = None
    observed_records: int | None = None
    limit_records: int | None = None
    observed_artifacts: int | None = None
    limit_artifacts: int | None = None
    observed_seconds: float | None = None
    limit_seconds: float | None = None
    if limitation.resource in {"source_bytes"}:
        observed_bytes = int(limitation.observed)
        limit_bytes = int(limitation.limit)
    elif limitation.resource in {"runtime"}:
        observed_seconds = float(limitation.observed)
        limit_seconds = float(limitation.limit)
    elif limitation.resource in {
        "output_records",
        "frontier_references",
        "frontier_waves",
    }:
        observed_records = int(limitation.observed)
        limit_records = int(limitation.limit)
    else:
        observed_artifacts = int(limitation.observed)
        limit_artifacts = int(limitation.limit)
    return dict(
        ledger_event(
            outcome=LedgerOutcome.PARTIAL,
            record_type=LedgerRecordType.SYSTEM,
            phase=phase,
            path=path,
            reason=LedgerReason.REFERENCE_EXTRACTION_LIMIT,
            stage=limitation.resource,
            observed_bytes=observed_bytes,
            limit_bytes=limit_bytes,
            observed_records=observed_records,
            limit_records=limit_records,
            observed_artifacts=observed_artifacts,
            limit_artifacts=limit_artifacts,
            observed_seconds=observed_seconds,
            limit_seconds=limit_seconds,
        )
    )


def _cache_transitive_result(
    target: str,
    child_result: dict[str, object],
    traversal: _TransitiveTraversalState,
) -> _CachedTransitiveResult:
    child_local_cache = _coerce_file_cache(
        child_result.get("local_file_cache") or child_result.get("file_cache")
    )
    child_file_cache = _coerce_file_cache(child_result.get("file_cache"))
    child_raw_cache = _coerce_raw_file_cache(child_result.get("raw_file_cache"))
    source_digest = _source_content_digest(child_raw_cache, child_local_cache)
    source_identity = _source_identity(target, source_digest)

    child_filtered = _coerce_findings_list(child_result.get("filtered_findings"))
    child_findings = _coerce_findings_list(child_result.get("findings"))
    all_ids = {finding.finding_id for finding in [*child_filtered, *child_findings]}
    all_ids.update(_effective_finding_ids(child_result))
    finding_id_map = {
        finding_id: _scoped_finding_id(source_identity, finding_id) for finding_id in all_ids
    }

    def _scope_finding(finding: Finding) -> Finding:
        return replace(
            finding,
            finding_id=finding_id_map[finding.finding_id],
            source_url=target,
            source_identity=source_identity,
            source_digest=source_digest,
        )

    scoped_filtered = [_scope_finding(item) for item in child_filtered[:_TRANSITIVE_MAX_FINDINGS]]
    scoped_findings = [_scope_finding(item) for item in child_findings[:_TRANSITIVE_MAX_FINDINGS]]
    scoped_finding_ids = {item.finding_id for item in scoped_findings}
    scoped_findings.extend(
        item for item in scoped_filtered if item.finding_id not in scoped_finding_ids
    )
    scoped_findings = scoped_findings[:_TRANSITIVE_MAX_FINDINGS]
    if (
        len(child_filtered) > _TRANSITIVE_MAX_FINDINGS
        or len(child_findings) > _TRANSITIVE_MAX_FINDINGS
    ):
        traversal.note_truncation(
            f"finding budget {_TRANSITIVE_MAX_FINDINGS} reached for {source_identity}"
        )

    scoped_ledger = _source_aware_ledger(
        child_result.get("inspection_ledger"),
        source_url=target,
        source_identity=source_identity,
        source_digest=source_digest,
        finding_id_map=finding_id_map,
    )
    retained_finding_ids = {item.finding_id for item in scoped_findings}
    for event in scoped_ledger:
        for id_field in ("input_finding_ids", "emitted_finding_ids"):
            ids = event.get(id_field)
            if isinstance(ids, list):
                event[id_field] = [
                    item for item in ids if isinstance(item, str) and item in retained_finding_ids
                ]
    extraction_deadline = monotonic() + traversal.remaining_seconds()
    extraction = transitive.extract_external_refs_with_metadata(
        child_local_cache,
        deadline=extraction_deadline,
    )
    for limitation in extraction.limitations:
        traversal.note_truncation(
            f"transitive reference {limitation.resource} limit at "
            f"{limitation.source_scope or source_identity}"
        )
        scoped_ledger.append(
            _transitive_limit_event(
                limitation,
                phase="transitive_reference_extraction",
                path=(limitation.source_scope or source_identity) + "/SKILL.md",
            )
        )
    scoped_ledger = _merge_bounded_ledger(
        [],
        scoped_ledger,
        limit=traversal.budget.max_ledger_events,
        traversal=traversal,
    )
    retained_work_ids = {
        str(event.get("work_id", "")) for event in scoped_ledger if event.get("work_id")
    }
    scoped_statuses = _source_aware_status_events(
        child_result.get("analyzer_status_events"),
        source_url=target,
        source_identity=source_identity,
        source_digest=source_digest,
        retained_work_ids=retained_work_ids,
        max_planned_work=len(retained_work_ids),
    )
    child_metadata = _decorate_component_metadata(
        _coerce_component_metadata(child_result.get("component_metadata")),
        source_identity,
        source_url=target,
        source_digest=source_digest,
    )
    child_components = _coerce_str_path_list(child_result.get("components"))
    if len(child_components) > _TRANSITIVE_MAX_COMPONENTS:
        traversal.note_truncation(
            f"component budget {_TRANSITIVE_MAX_COMPONENTS} reached for {source_identity}"
        )
        child_components = child_components[:_TRANSITIVE_MAX_COMPONENTS]
    return _CachedTransitiveResult(
        source_url=target,
        source_identity=source_identity,
        source_digest=source_digest,
        filtered_findings=scoped_filtered,
        findings=scoped_findings,
        effective_finding_ids=[
            finding_id_map.get(item, _scoped_finding_id(source_identity, item))
            for item in _effective_finding_ids(child_result)[:_TRANSITIVE_MAX_FINDINGS]
        ],
        inspection_ledger=scoped_ledger,
        analyzer_status_events=scoped_statuses,
        llm_call_log=_coerce_llm_call_log(child_result.get("llm_call_log")),
        inference_usage=_coerce_dict_list(child_result.get("inference_usage")),
        components=_source_aware_components(child_components, source_identity),
        component_metadata=child_metadata,
        file_cache=_source_aware_file_cache(child_file_cache, source_identity),
        local_file_cache=_source_aware_file_cache(child_local_cache, source_identity),
        artifact_inventory=_source_aware_inventory(
            child_result.get("artifact_inventory"),
            source_url=target,
            source_identity=source_identity,
            source_digest=source_digest,
        ),
        artifact_references=_source_aware_references(
            child_result.get("artifact_references"),
            source_url=target,
            source_identity=source_identity,
            source_digest=source_digest,
        ),
        has_executable_scripts=bool(child_result.get("has_executable_scripts", False))
        or any(bool(entry.get("executable", False)) for entry in child_metadata),
        refs=extraction.references,
    )


def _run_graph_scan(
    input_path: str,
    format: FormatChoice,
    no_llm: bool,
    yara_dir: str | None = None,
    baseline: Path | None = None,
    show_suppressed: bool = False,
    transitive_traversal: _TransitiveTraversalState | None = None,
    initial_inspection_ledger: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    state = _scan_state(
        input_path=input_path,
        format=format,
        no_llm=no_llm,
        yara_rules_dir=yara_dir,
        baseline=baseline,
        show_suppressed=show_suppressed,
    )
    if transitive_traversal is not None:
        state["transitive_traversal_state"] = transitive_traversal
    if initial_inspection_ledger:
        state["inspection_ledger"] = initial_inspection_ledger
    trace_config = _build_trace_config(input_path, format, no_llm)
    return cast(dict[str, object], graph.invoke(state, config=trace_config))


def _annotate_transitive_findings(
    findings: list[Finding],
    *,
    source_url: str,
    source_identity: str,
    source_digest: str,
    transitive_depth: int,
) -> list[Finding]:
    annotated: list[Finding] = []
    for finding in findings:
        base_occurrences = finding.occurrences or [
            {
                "file": finding.file,
                "start_line": finding.start_line,
                "end_line": finding.end_line,
            }
        ]
        occurrences = [
            {
                **occurrence,
                "source_url": source_url,
                "source_identity": source_identity,
                "source_digest": source_digest,
                "transitive_depth": transitive_depth,
            }
            for occurrence in base_occurrences
        ]
        annotated.append(
            replace(
                finding,
                transitive_depth=transitive_depth,
                source_url=source_url,
                source_identity=source_identity,
                source_digest=source_digest,
                occurrences=occurrences,
            )
        )
    return annotated


def _bounded_extend[T](
    destination: list[T],
    values: list[T],
    *,
    limit: int,
    traversal: _TransitiveTraversalState,
    resource: str,
) -> None:
    remaining = max(0, limit - len(destination))
    destination.extend(values[:remaining])
    if len(values) > remaining:
        traversal.note_truncation(f"{resource} budget {limit} reached")


def _bounded_cache_update(
    destination: dict[str, str],
    values: dict[str, str],
    *,
    limit: int,
    traversal: _TransitiveTraversalState,
    resource: str,
) -> None:
    for path in sorted(values):
        if path in destination:
            destination[path] = values[path]
            continue
        if len(destination) >= limit:
            traversal.note_truncation(f"{resource} budget {limit} reached")
            break
        destination[path] = values[path]


def _bounded_root_status_events(
    value: object,
    *,
    retained_work_ids: set[str],
    limit: int,
) -> list[dict[str, object]]:
    statuses: list[dict[str, object]] = []
    planned_retained = 0
    for status in _coerce_dict_list(value):
        if len(statuses) >= limit:
            break
        entry = dict(status)
        planned = entry.get("planned_work")
        if isinstance(planned, list):
            bounded_planned: list[dict[str, object]] = []
            for target in planned:
                if planned_retained >= limit:
                    break
                if not isinstance(target, dict):
                    continue
                work_id = str(target.get("work_id", ""))
                if work_id not in retained_work_ids:
                    continue
                bounded_planned.append(dict(target))
                planned_retained += 1
            entry["planned_work"] = bounded_planned
        statuses.append(entry)
    return statuses


def _status_planned_work_count(statuses: list[dict[str, object]]) -> int:
    total = 0
    for status in statuses:
        planned = status.get("planned_work")
        if isinstance(planned, list):
            total += len(planned)
    return total


def _merge_bounded_ledger(
    existing: list[dict[str, object]],
    updates: list[dict[str, object]],
    *,
    limit: int,
    traversal: _TransitiveTraversalState | None = None,
) -> list[dict[str, object]]:
    """Merge traversal ledger rows under the caller's shared record ceiling."""
    effective_limit = max(1, limit)
    if existing and existing[-1].get("phase") == "ledger_output":
        if updates and traversal is not None:
            traversal.note_truncation(f"inspection ledger budget {effective_limit} reached")
        prior = existing[-1]
        observed_value = prior.get("observed_records")
        prior_observed = observed_value if isinstance(observed_value, int) else len(existing)
        return [
            *existing[:-1][: effective_limit - 1],
            dict(
                ledger_event(
                    outcome=LedgerOutcome.PARTIAL,
                    record_type=LedgerRecordType.SYSTEM,
                    phase="ledger_output",
                    path=str(prior.get("path", "SKILL.md")),
                    reason=LedgerReason.OUTPUT_LIMIT,
                    observed_records=max(prior_observed, len(existing)) + len(updates),
                    limit_records=effective_limit,
                )
            ),
        ]
    combined = [*existing, *updates]
    if len(combined) <= effective_limit:
        return combined
    if traversal is not None:
        traversal.note_truncation(f"inspection ledger budget {effective_limit} reached")
    overflow = combined[effective_limit - 1]
    return [
        *combined[: effective_limit - 1],
        dict(
            ledger_event(
                outcome=LedgerOutcome.PARTIAL,
                record_type=LedgerRecordType.SYSTEM,
                phase="ledger_output",
                path=str(overflow.get("path", "SKILL.md")),
                reason=LedgerReason.OUTPUT_LIMIT,
                observed_records=len(combined),
                limit_records=effective_limit,
            )
        ),
    ]


def _scan_transitive(
    initial_result: dict[str, object],
    format: FormatChoice,
    no_llm: bool,
    max_depth: int,
    transitive_allow_prefix: tuple[str, ...] | list[str] | None,
    transitive_deny_prefix: tuple[str, ...] | list[str] | None,
    baseline: Path | None,
    show_suppressed: bool,
    visited: set[str],
    scan_cache: dict[str, _CachedTransitiveResult] | None = None,
    budget: _TransitiveBudget | None = None,
    yara_dir: str | None = None,
    traversal: _TransitiveTraversalState | None = None,
) -> dict[str, object]:
    if max_depth <= 0:
        report_result = cast(dict[str, object], report(initial_result))
        report_result["temp_dir_for_cleanup"] = initial_result.get("temp_dir_for_cleanup")
        report_result["transitive_finding_count"] = 0
        report_result["transitive_sources"] = []
        report_result["transitive_targets_scanned"] = 0
        report_result["transitive_bytes_scanned"] = 0
        report_result["transitive_artifacts_scanned"] = 0
        report_result["transitive_truncated"] = False
        report_result["transitive_truncation_reasons"] = []
        report_result["analysis_completeness"] = initial_result.get("analysis_completeness", {})
        return report_result

    if traversal is None:
        traversal = _TransitiveTraversalState(
            cache=scan_cache if scan_cache is not None else {},
            budget=budget if budget is not None else _TransitiveBudget(),
        )
    elif scan_cache is not None and traversal.cache is not scan_cache:
        traversal.cache = scan_cache
    transitive_sources: set[str] = set()
    merged_filtered_findings = _coerce_findings_list(initial_result.get("filtered_findings"))[
        : traversal.budget.max_findings
    ]
    merged_findings = _coerce_findings_list(initial_result.get("findings"))[
        : traversal.budget.max_findings
    ]
    merged_llm_call_log = _coerce_llm_call_log(initial_result.get("llm_call_log"))[
        :MAX_INSPECTION_LEDGER_EVENTS
    ]
    merged_inference_usage = _coerce_dict_list(initial_result.get("inference_usage"))[
        :MAX_INSPECTION_LEDGER_EVENTS
    ]
    merged_effective_finding_ids = _effective_finding_ids(initial_result)[
        : traversal.budget.max_findings
    ]
    merged_inspection_ledger = _merge_bounded_ledger(
        [],
        _coerce_dict_list(initial_result.get("inspection_ledger")),
        limit=traversal.budget.max_ledger_events,
        traversal=traversal,
    )
    retained_work_ids = {
        str(event.get("work_id", "")) for event in merged_inspection_ledger if event.get("work_id")
    }
    merged_analyzer_status_events = _bounded_root_status_events(
        initial_result.get("analyzer_status_events"),
        retained_work_ids=retained_work_ids,
        limit=traversal.budget.max_status_events,
    )
    merged_components = _source_aware_components(
        _coerce_str_path_list(initial_result.get("components"))[: traversal.budget.max_components],
        None,
    )
    file_cache = _coerce_file_cache(initial_result.get("file_cache"))
    merged_file_cache = _source_aware_file_cache(file_cache, None)
    local_file_cache = _coerce_file_cache(
        initial_result.get("local_file_cache") or initial_result.get("file_cache")
    )
    merged_local_file_cache = _source_aware_file_cache(local_file_cache, None)
    merged_artifact_inventory = _coerce_dict_list(initial_result.get("artifact_inventory"))[
        : traversal.budget.max_components
    ]
    merged_artifact_references = _coerce_dict_list(initial_result.get("artifact_references"))[
        : traversal.budget.max_references
    ]
    component_metadata = _decorate_component_metadata(
        _coerce_component_metadata(initial_result.get("component_metadata")), None
    )[: traversal.budget.max_components]
    has_executable_scripts = bool(initial_result.get("has_executable_scripts", False))

    root_extraction = transitive.extract_external_refs_with_metadata(
        local_file_cache,
        deadline=monotonic() + traversal.remaining_seconds(),
    )
    for limitation in root_extraction.limitations:
        traversal.note_truncation(
            f"transitive reference {limitation.resource} limit at "
            f"{limitation.source_scope or 'root'}"
        )
        merged_inspection_ledger = _merge_bounded_ledger(
            merged_inspection_ledger,
            [
                _transitive_limit_event(
                    limitation,
                    phase="transitive_reference_extraction",
                    path=(limitation.source_scope or "SKILL.md"),
                )
            ],
            limit=traversal.budget.max_ledger_events,
            traversal=traversal,
        )
    frontier = transitive.BoundedTransitiveFrontier(
        deadline=monotonic() + traversal.remaining_seconds(),
        max_waves=min(max(1, max_depth), transitive.MAX_TRANSITIVE_FRONTIER_WAVES),
        max_references=min(
            traversal.budget.max_references,
            transitive.MAX_TRANSITIVE_FRONTIER_REFERENCES,
        ),
    )
    frontier.append(1, root_extraction.references)
    recorded_frontier_limitations = 0

    while frontier:
        if not traversal.can_scan_more():
            break
        wave = frontier.popleft()
        if wave is None:
            break
        current_depth, refs = wave.depth, wave.references
        plan = transitive.plan_transitive_targets_with_metadata(
            refs=refs,
            visited=visited,
            current_depth=current_depth,
            max_depth=max_depth,
            allow_prefixes=transitive_allow_prefix,
            deny_prefixes=transitive_deny_prefix,
            deadline=monotonic() + traversal.remaining_seconds(),
        )
        for limitation in plan.limitations:
            traversal.note_truncation(f"transitive plan {limitation.resource} limit")
            merged_inspection_ledger = _merge_bounded_ledger(
                merged_inspection_ledger,
                [
                    _transitive_limit_event(
                        limitation,
                        phase="transitive_target_planning",
                        path="SKILL.md",
                    )
                ],
                limit=traversal.budget.max_ledger_events,
                traversal=traversal,
            )
        for target in plan.targets:
            if not traversal.can_scan_more():
                break
            child_result: dict[str, object] | None = None
            try:
                cached = traversal.cache.get(target)
                if cached is None:
                    child_result = _run_graph_scan(
                        input_path=target,
                        format=format,
                        no_llm=no_llm,
                        yara_dir=yara_dir,
                        # A root baseline cannot pre-suppress dependency findings
                        # before source provenance is attached. Suppression is
                        # applied exactly once to the merged, source-bound set.
                        baseline=None,
                        show_suppressed=False,
                        transitive_traversal=traversal,
                    )
                    cached = _cache_transitive_result(target, child_result, traversal)
                    traversal.cache[target] = cached
                    traversal.record_scan()
                    if child_result.get("execution_successful") is False:
                        traversal.note_child_scan_failure(target)
                    child_completeness = child_result.get("analysis_completeness")
                    if (
                        isinstance(child_completeness, dict)
                        and child_completeness.get("is_complete") is False
                    ):
                        traversal.note_truncation(f"transitive child scan incomplete for {target}")
                transitive_sources.add(target)
                merged_inspection_ledger = _merge_bounded_ledger(
                    merged_inspection_ledger,
                    cached.inspection_ledger,
                    limit=traversal.budget.max_ledger_events,
                    traversal=traversal,
                )
                global_work_ids = {
                    str(event.get("work_id", ""))
                    for event in merged_inspection_ledger
                    if event.get("work_id")
                }
                bounded_statuses: list[dict[str, object]] = []
                for status in cached.analyzer_status_events:
                    entry = dict(status)
                    planned = entry.get("planned_work")
                    if isinstance(planned, list):
                        entry["planned_work"] = [
                            item
                            for item in planned
                            if isinstance(item, dict)
                            and str(item.get("work_id", "")) in global_work_ids
                        ][
                            : max(
                                0,
                                traversal.budget.max_ledger_events
                                - _status_planned_work_count(merged_analyzer_status_events),
                            )
                        ]
                    bounded_statuses.append(entry)
                _bounded_extend(
                    merged_analyzer_status_events,
                    bounded_statuses,
                    limit=traversal.budget.max_status_events,
                    traversal=traversal,
                    resource="analyzer status",
                )
                _bounded_extend(
                    merged_llm_call_log,
                    cached.llm_call_log,
                    limit=MAX_INSPECTION_LEDGER_EVENTS,
                    traversal=traversal,
                    resource="LLM call log",
                )
                _bounded_extend(
                    merged_inference_usage,
                    cached.inference_usage,
                    limit=MAX_INSPECTION_LEDGER_EVENTS,
                    traversal=traversal,
                    resource="inference usage",
                )
                annotated_filtered = _annotate_transitive_findings(
                    cached.filtered_findings,
                    source_url=cached.source_url,
                    source_identity=cached.source_identity,
                    source_digest=cached.source_digest,
                    transitive_depth=current_depth,
                )
                annotated_findings = _annotate_transitive_findings(
                    cached.findings,
                    source_url=cached.source_url,
                    source_identity=cached.source_identity,
                    source_digest=cached.source_digest,
                    transitive_depth=current_depth,
                )
                _bounded_extend(
                    merged_filtered_findings,
                    annotated_filtered,
                    limit=traversal.budget.max_findings,
                    traversal=traversal,
                    resource="finding",
                )
                _bounded_extend(
                    merged_findings,
                    annotated_findings,
                    limit=traversal.budget.max_findings,
                    traversal=traversal,
                    resource="finding",
                )
                _bounded_extend(
                    merged_effective_finding_ids,
                    [
                        item
                        for item in cached.effective_finding_ids
                        if item
                        in {
                            finding.finding_id
                            for finding in [*annotated_findings, *annotated_filtered]
                        }
                    ],
                    limit=traversal.budget.max_findings,
                    traversal=traversal,
                    resource="effective finding",
                )

                _bounded_extend(
                    component_metadata,
                    cached.component_metadata,
                    limit=traversal.budget.max_components,
                    traversal=traversal,
                    resource="component metadata",
                )
                if cached.has_executable_scripts:
                    has_executable_scripts = True
                _bounded_extend(
                    merged_components,
                    cached.components,
                    limit=traversal.budget.max_components,
                    traversal=traversal,
                    resource="component",
                )
                _bounded_cache_update(
                    merged_file_cache,
                    cached.file_cache,
                    limit=traversal.budget.max_components,
                    traversal=traversal,
                    resource="provider cache",
                )
                _bounded_cache_update(
                    merged_local_file_cache,
                    cached.local_file_cache,
                    limit=traversal.budget.max_components,
                    traversal=traversal,
                    resource="local cache",
                )
                _bounded_extend(
                    merged_artifact_inventory,
                    cached.artifact_inventory,
                    limit=traversal.budget.max_components,
                    traversal=traversal,
                    resource="artifact inventory",
                )
                _bounded_extend(
                    merged_artifact_references,
                    cached.artifact_references,
                    limit=traversal.budget.max_references,
                    traversal=traversal,
                    resource="artifact reference",
                )

                if current_depth < max_depth:
                    frontier.append(current_depth + 1, cached.refs)
                new_frontier_limitations = frontier.limitations[recorded_frontier_limitations:]
                recorded_frontier_limitations += len(new_frontier_limitations)
                for limitation in new_frontier_limitations:
                    traversal.note_truncation(f"transitive frontier {limitation.resource} limit")
                    merged_inspection_ledger = _merge_bounded_ledger(
                        merged_inspection_ledger,
                        [
                            _transitive_limit_event(
                                limitation,
                                phase="transitive_frontier",
                                path=cached.source_identity + "/SKILL.md",
                            )
                        ],
                        limit=traversal.budget.max_ledger_events,
                        traversal=traversal,
                    )
            except Exception:
                transitive_sources.add(target)
                traversal.note_child_scan_failure(target)
                if format == FormatChoice.json:
                    logger.warning("Transitive scan failed for %s", target)
                else:
                    console.print(f"[yellow]Warning:[/yellow] Transitive scan failed for {target}")
            finally:
                if child_result is not None:
                    cleanup_result(child_result)

    for limitation in frontier.limitations[recorded_frontier_limitations:]:
        traversal.note_truncation(f"transitive frontier {limitation.resource} limit")
        merged_inspection_ledger = _merge_bounded_ledger(
            merged_inspection_ledger,
            [
                _transitive_limit_event(
                    limitation,
                    phase="transitive_frontier",
                    path="SKILL.md",
                )
            ],
            limit=traversal.budget.max_ledger_events,
            traversal=traversal,
        )

    if traversal.truncation_reasons:
        traversal_event = ledger_event(
            outcome=LedgerOutcome.PARTIAL,
            record_type=LedgerRecordType.SYSTEM,
            phase="transitive_traversal",
            path="SKILL.md",
            reason=LedgerReason.OUTPUT_LIMIT,
        )
        merged_inspection_ledger = _merge_bounded_ledger(
            merged_inspection_ledger,
            [dict(traversal_event)],
            limit=traversal.budget.max_ledger_events,
            traversal=traversal,
        )

    merged_result: dict[str, object] = {
        **initial_result,
        "filtered_findings": merged_filtered_findings,
        "findings": merged_findings,
        "components": merged_components,
        "component_metadata": _merge_unique_component_metadata(component_metadata),
        "file_cache": merged_file_cache,
        "local_file_cache": merged_local_file_cache,
        "artifact_inventory": merged_artifact_inventory,
        "artifact_references": merged_artifact_references,
        "has_executable_scripts": has_executable_scripts,
        "llm_call_log": merged_llm_call_log,
        "inference_usage": merged_inference_usage,
        "effective_finding_ids": list(dict.fromkeys(merged_effective_finding_ids)),
        "inspection_ledger": merged_inspection_ledger,
        "analyzer_status_events": merged_analyzer_status_events,
        "baseline": initial_result.get(
            "baseline", baseline if isinstance(baseline, Baseline) else None
        ),
        "show_suppressed": initial_result.get("show_suppressed", show_suppressed),
        "transitive_targets_scanned": traversal.scanned_targets,
        "transitive_bytes_scanned": traversal.scanned_bytes,
        "transitive_artifacts_scanned": traversal.scanned_artifacts,
        "transitive_truncated": bool(traversal.truncation_reasons),
        "transitive_truncation_reasons": traversal.truncation_reasons,
    }
    if merged_inspection_ledger or merged_analyzer_status_events:
        completeness, effective_ids = finalize_ledger(merged_result)
        merged_result["analysis_completeness"] = completeness
        merged_result["execution_successful"] = completeness["execution_successful"]
        merged_result["effective_finding_ids"] = effective_ids
    report_result = cast(dict[str, object], report(merged_result))
    report_result["analysis_completeness"] = merged_result.get("analysis_completeness", {})
    report_result["temp_dir_for_cleanup"] = initial_result.get("temp_dir_for_cleanup")
    active_findings = _coerce_findings_list(report_result.get("filtered_findings"))
    report_result["transitive_finding_count"] = sum(
        1
        for finding in active_findings
        if isinstance(finding, Finding) and finding.source_url is not None
    )
    report_result["transitive_sources"] = sorted(transitive_sources)
    report_result["transitive_targets_scanned"] = traversal.scanned_targets
    report_result["transitive_bytes_scanned"] = traversal.scanned_bytes
    report_result["transitive_artifacts_scanned"] = traversal.scanned_artifacts
    report_result["transitive_truncated"] = bool(traversal.truncation_reasons)
    report_result["transitive_truncation_reasons"] = traversal.truncation_reasons
    return report_result


def _coerce_component_metadata(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _scan_skill(
    input_path: str,
    format: FormatChoice,
    no_llm: bool,
    baseline: Path | None,
    yara_rules_dir: Path | None,
    verbose: bool,
    show_suppressed: bool,
    transitive_enabled: bool,
    transitive_depth: int,
    transitive_allow_prefix: tuple[str, ...] | list[str] | None,
    transitive_deny_prefix: tuple[str, ...] | list[str] | None,
    transitive_cache: dict[str, _CachedTransitiveResult] | None = None,
    transitive_traversal: _TransitiveTraversalState | None = None,
    pre_scan_ledger_events: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    yara_dir = str(yara_rules_dir.resolve()) if yara_rules_dir else None
    active_visited: set[str] = set()
    if verbose:
        console.print("[dim]Running scan...[/dim]")
    logger.debug(
        "Scan started: input_path=%s, format=%s, use_llm=%s, transitive=%s",
        input_path,
        format,
        not no_llm,
        transitive_enabled,
    )
    if transitive_enabled and transitive_traversal is None:
        transitive_traversal = _TransitiveTraversalState(cache=transitive_cache or {})
    if pre_scan_ledger_events:
        result = _run_graph_scan(
            input_path=input_path,
            format=format,
            no_llm=no_llm,
            yara_dir=yara_dir,
            baseline=baseline,
            show_suppressed=show_suppressed,
            transitive_traversal=transitive_traversal,
            initial_inspection_ledger=pre_scan_ledger_events,
        )
    else:
        result = _run_graph_scan(
            input_path=input_path,
            format=format,
            no_llm=no_llm,
            yara_dir=yara_dir,
            baseline=baseline,
            show_suppressed=show_suppressed,
            transitive_traversal=transitive_traversal,
        )
    if not transitive_enabled:
        return result
    if transitive_traversal is None:  # Defensive: transitive scans initialize before root work.
        transitive_traversal = _TransitiveTraversalState(cache=transitive_cache or {})
    transitive_allow_prefix, transitive_deny_prefix = transitive.normalize_prefixes(
        transitive_allow_prefix, transitive_deny_prefix
    )
    try:
        active_visited.add(transitive.canonicalize_source_identity(input_path))
    except ValueError:
        pass
    return _scan_transitive(
        initial_result=result,
        format=format,
        no_llm=no_llm,
        max_depth=transitive_depth,
        transitive_allow_prefix=transitive_allow_prefix,
        transitive_deny_prefix=transitive_deny_prefix,
        baseline=baseline,
        show_suppressed=show_suppressed,
        visited=active_visited,
        scan_cache=transitive_cache,
        yara_dir=yara_dir,
        traversal=transitive_traversal,
    )


def _multi_skill_public_record_count(result: dict[str, object]) -> int:
    """Count bounded active and suppressed occurrence records in one child report."""
    count = 0
    active = effective_findings(result)
    suppressed = result.get("suppressed_findings")
    candidates: list[object] = [*active]
    if isinstance(suppressed, list):
        candidates.extend(
            finding
            for item in suppressed
            if (finding := getattr(item, "finding", None)) is not None
        )
    for finding in candidates:
        occurrences = getattr(finding, "occurrences", None)
        count += max(1, len(occurrences)) if isinstance(occurrences, list) else 1
        if count > _MULTI_SKILL_MAX_PUBLIC_RECORDS:
            return count
    return count


def _multi_skill_analysis_completeness(
    *,
    total_skills: int,
    complete_skills: int,
    partial_skills: int,
    failed_skills: int,
    omitted_skills: int,
    limitations: list[str],
) -> dict[str, object]:
    """Build one conservative machine-readable completeness summary for recursion."""
    is_complete = (
        not limitations and not partial_skills and not failed_skills and not omitted_skills
    )
    execution_successful = failed_skills == 0
    status = "failed" if not execution_successful else "complete" if is_complete else "partial"
    denominator = max(1, total_skills)
    return {
        "is_complete": is_complete,
        "execution_successful": execution_successful,
        "status": status,
        "coverage_percent": round(100.0 * complete_skills / denominator, 2),
        "fully_inspected_files": complete_skills,
        "partially_inspected_files": partial_skills,
        "entirely_uninspected_files": failed_skills + omitted_skills,
        "total_files": total_skills,
        "limitations": limitations,
        "scope": "recursive_skills",
    }


def _multi_skill_sarif_report(
    processed_skills: list[SkillDirectory],
    results: list[dict[str, object]],
    completeness: dict[str, object],
) -> dict[str, object]:
    """Merge bounded child SARIF runs and append one aggregate invocation run."""
    runs: list[dict[str, object]] = []
    for skill, result in zip(processed_skills, results, strict=True):
        sarif = result.get("sarif_report")
        if not isinstance(sarif, dict):
            parsed = _recursive_json_payload(result)
            sarif = parsed if isinstance(parsed, dict) and "runs" in parsed else None
        if not isinstance(sarif, dict):
            continue
        child_runs = sarif.get("runs")
        if not isinstance(child_runs, list):
            continue
        for raw_run in child_runs:
            if not isinstance(raw_run, dict):
                continue
            run = deepcopy(raw_run)
            properties = run.get("properties")
            run_properties = dict(properties) if isinstance(properties, dict) else {}
            run_properties["recursiveSkill"] = {
                "name": skill.name,
                "path": skill.relative_path,
            }
            run["properties"] = run_properties
            runs.append(run)

    aggregate_invocation: dict[str, object] = {
        "executionSuccessful": bool(completeness.get("execution_successful", False)),
        "properties": {"analysisCompleteness": completeness},
    }
    if not bool(completeness.get("is_complete", False)):
        aggregate_invocation["toolExecutionNotifications"] = [
            {
                "message": {
                    "text": "Recursive analysis was incomplete after an aggregate safety limit."
                },
                "level": "warning",
                "properties": {
                    "kind": "inspection_limitation",
                    "reasonCode": "output_limit",
                },
            }
        ]
    runs.append(
        {
            "tool": {"driver": {"name": "skillspector", "version": __version__}},
            "results": [],
            "invocations": [aggregate_invocation],
            "properties": {"kind": "recursiveAggregate"},
        }
    )
    merged: dict[str, object] = {
        "$schema": SARIF_SCHEMA_URI,
        "version": "2.1.0",
        "runs": runs,
    }
    validate_sarif_report(merged)
    return merged


def _mark_recursive_output_limited(
    completeness: dict[str, object],
) -> tuple[dict[str, object], list[str]]:
    """Return a fail-closed aggregate state after serialized output overflow."""
    reason = (
        f"recursive serialized report character budget {_MULTI_SKILL_MAX_REPORT_CHARACTERS} reached"
    )
    # Once the output itself is over budget, retain one content-free sentinel
    # rather than copying a potentially large list of earlier limitations into
    # the fallback document.
    bounded_limitations = [reason]
    limited = dict(completeness)
    limited["is_complete"] = False
    if limited.get("status") != "failed":
        limited["status"] = "partial"
    limited["limitations"] = bounded_limitations
    return limited, bounded_limitations


def _ensure_recursive_output_bound(rendered: str) -> None:
    """Refuse to write a recursive report that exceeds its public ceiling."""
    if len(rendered) > _MULTI_SKILL_MAX_REPORT_CHARACTERS:
        raise RuntimeError("recursive report could not fit the configured output budget")


def _scan_multi_skill(
    detection: MultiSkillDetectionResult,
    format: FormatChoice,
    output: Path | None,
    no_llm: bool,
    baseline: Path | None = None,
    show_suppressed: bool = False,
    transitive_enabled: bool = False,
    transitive_depth: int = 1,
    transitive_allow_prefix: tuple[str, ...] | list[str] | None = None,
    transitive_deny_prefix: tuple[str, ...] | list[str] | None = None,
    yara_dir: str | None = None,
    verbose: bool = False,
    fail_on_incomplete: bool = False,
    **legacy_kwargs: object,
) -> None:
    """Scan each detected sub-skill independently and produce a combined report."""
    if yara_dir is None and isinstance(legacy_kwargs.get("yara_rules_dir"), Path):
        yara_dir = str(legacy_kwargs["yara_rules_dir"])
    skills = detection.skills
    console.print(f"[bold]Multi-skill directory detected:[/bold] {len(skills)} skills found\n")

    shared_transitive_cache: dict[str, _CachedTransitiveResult] = {}
    shared_transitive_traversal = _TransitiveTraversalState(
        cache=shared_transitive_cache,
        budget=_TransitiveBudget(
            max_bytes=_TRANSITIVE_MAX_BYTES if transitive_enabled else MAX_WORKFLOW_BYTES
        ),
    )
    results: list[dict[str, object]] = []
    processed_skills: list[SkillDirectory] = []
    max_score = 0
    execution_failed = False
    transitive_finding_count = 0
    transitive_sources: set[str] = set()
    analysis_incomplete = not detection.complete
    aggregate_limitations = [
        f"recursive discovery {limitation.resource} limit reached"
        for limitation in detection.limitations[:256]
    ]
    retained_public_records = 0
    retained_report_characters = 0
    complete_skill_count = 0
    partial_skill_count = 0
    failed_skill_count = 0

    for i, skill in enumerate(skills, 1):
        if i > _MULTI_SKILL_MAX_SKILLS:
            analysis_incomplete = True
            aggregate_limitations.append(
                f"recursive skill count budget {_MULTI_SKILL_MAX_SKILLS} reached"
            )
            break
        if retained_public_records >= _MULTI_SKILL_MAX_PUBLIC_RECORDS:
            analysis_incomplete = True
            aggregate_limitations.append(
                f"recursive public finding record budget {_MULTI_SKILL_MAX_PUBLIC_RECORDS} reached"
            )
            break
        if retained_report_characters >= _MULTI_SKILL_MAX_REPORT_CHARACTERS:
            analysis_incomplete = True
            aggregate_limitations.append(
                f"recursive report character budget {_MULTI_SKILL_MAX_REPORT_CHARACTERS} reached"
            )
            break
        if not shared_transitive_traversal.can_scan_more():
            analysis_incomplete = True
            aggregate_limitations.extend(shared_transitive_traversal.truncation_reasons)
            break
        console.print(
            f"  [{i}/{len(skills)}] Scanning [bold]{skill.name}[/bold] ({skill.relative_path}/)"
        )
        try:
            result = _scan_skill(
                input_path=str(skill.path),
                format=format,
                no_llm=no_llm,
                baseline=baseline,
                yara_rules_dir=Path(yara_dir) if yara_dir else None,
                verbose=verbose,
                show_suppressed=show_suppressed,
                transitive_enabled=transitive_enabled,
                transitive_depth=transitive_depth,
                transitive_allow_prefix=transitive_allow_prefix,
                transitive_deny_prefix=transitive_deny_prefix,
                transitive_cache=shared_transitive_cache,
                transitive_traversal=shared_transitive_traversal,
            )
            result_body = _result_body(result)
            result_characters = len(result_body)
            result_records = _multi_skill_public_record_count(result)
            if (
                retained_public_records + result_records > _MULTI_SKILL_MAX_PUBLIC_RECORDS
                or retained_report_characters + result_characters
                > _MULTI_SKILL_MAX_REPORT_CHARACTERS
            ):
                analysis_incomplete = True
                if retained_public_records + result_records > _MULTI_SKILL_MAX_PUBLIC_RECORDS:
                    aggregate_limitations.append(
                        "recursive public finding record budget "
                        f"{_MULTI_SKILL_MAX_PUBLIC_RECORDS} reached"
                    )
                if (
                    retained_report_characters + result_characters
                    > _MULTI_SKILL_MAX_REPORT_CHARACTERS
                ):
                    aggregate_limitations.append(
                        "recursive report character budget "
                        f"{_MULTI_SKILL_MAX_REPORT_CHARACTERS} reached"
                    )
                cleanup_result(result)
                break
            results.append(result)
            processed_skills.append(skill)
            retained_public_records += result_records
            retained_report_characters += result_characters
            child_failed = result.get("execution_successful") is False
            if child_failed:
                execution_failed = True
                failed_skill_count += 1
            completeness_value = result.get("analysis_completeness")
            if (
                not child_failed
                and isinstance(completeness_value, dict)
                and not bool(completeness_value.get("is_complete", True))
            ):
                analysis_incomplete = True
                partial_skill_count += 1
            elif not child_failed:
                complete_skill_count += 1
            score = result.get("risk_score") or 0
            try:
                score = int(score)
            except (TypeError, ValueError):
                score = 0
            if score > max_score:
                max_score = score
            child_transitive_count = result.get("transitive_finding_count")
            if isinstance(child_transitive_count, int):
                transitive_finding_count += child_transitive_count
            for source in _coerce_str_path_list(result.get("transitive_sources")):
                transitive_sources.add(source)
            severity = result.get("risk_severity") or "LOW"
            console.print(f"         Score: {score}/100 ({severity})\n")
        except Exception as e:
            error_message = str(e)[:1_024]
            err_console.print(f"         [red]Error:[/red] {error_message}\n")
            execution_failed = True
            failed_skill_count += 1
            results.append({"skill_name": skill.name, "error": error_message})
            processed_skills.append(skill)

    omitted_skill_count = len(skills) - len(processed_skills)
    if omitted_skill_count:
        analysis_incomplete = True
        aggregate_limitations.append(
            f"{omitted_skill_count} recursive skill(s) omitted after an aggregate limit"
        )
    aggregate_limitations = list(dict.fromkeys(aggregate_limitations))[:256]
    aggregate_completeness = _multi_skill_analysis_completeness(
        total_skills=len(skills),
        complete_skills=complete_skill_count,
        partial_skills=partial_skill_count,
        failed_skills=failed_skill_count,
        omitted_skills=omitted_skill_count,
        limitations=aggregate_limitations,
    )
    analysis_incomplete = not bool(aggregate_completeness["is_complete"])

    console.print("\n[bold]═══ Multi-Skill Summary ═══[/bold]\n")
    console.print(
        f"  {'Skill':<30} {'Score':<8} {'Severity':<12} {'Findings':<10} {'Execution':<10}"
    )
    console.print(f"  {'─' * 30} {'─' * 8} {'─' * 12} {'─' * 10} {'─' * 10}")

    for skill, result in zip(processed_skills, results, strict=True):
        if "error" in result:
            console.print(f"  {skill.name:<30} {'ERROR':<8} {'—':<12} {'—':<10} {'error':<10}")
            continue
        score = result.get("risk_score", 0)
        severity = result.get("risk_severity", "LOW")
        finding_count = len(effective_findings(result))
        execution = "failed" if result.get("execution_successful") is False else "successful"
        console.print(
            f"  {skill.name:<30} {score:<8} {severity:<12} {finding_count:<10} {execution:<10}"
        )
    if omitted_skill_count:
        console.print(
            f"  {'<omitted>':<30} {'—':<8} {'—':<12} {omitted_skill_count:<10} {'partial':<10}"
        )
        console.print(
            "[yellow]Recursive scan incomplete:[/yellow] one or more skills were omitted "
            "after an aggregate safety limit."
        )

    if output and format == FormatChoice.json:
        combined: dict[str, object] = {
            "multi_skill": True,
            "skill_count": len(skills),
            "max_risk_score": max_score,
            "execution_successful": not execution_failed,
            "risk_recommendation": (
                "DO_NOT_INSTALL"
                if execution_failed or max_score > RISK_THRESHOLD
                else "CAUTION"
                if analysis_incomplete
                else "SAFE"
            ),
            "analysis_completeness": aggregate_completeness,
            "skills_scanned": len(processed_skills),
            "skills_omitted": omitted_skill_count,
            "public_finding_records": retained_public_records,
            "report_characters": retained_report_characters,
            "transitive_finding_count": transitive_finding_count,
            "transitive_sources": sorted(transitive_sources),
            "skills": [],
        }
        combined_skills = cast(list[dict[str, object]], combined["skills"])
        for skill, result in zip(processed_skills, results, strict=True):
            if "error" in result:
                combined_skills.append({"name": skill.name, "error": result["error"]})
            else:
                payload = _recursive_json_payload(result) or {}
                finding_count = len(effective_findings(result))
                entry = {
                    "name": skill.name,
                    "path": skill.relative_path,
                    "risk_score": result.get("risk_score", 0),
                    "risk_severity": result.get("risk_severity", "LOW"),
                    "finding_count": finding_count,
                    "execution_successful": result.get("execution_successful", True),
                    "transitive_finding_count": result.get("transitive_finding_count", 0),
                    "transitive_sources": result.get("transitive_sources", []),
                }
                entry.update(payload)
                entry["name"] = skill.name
                entry["path"] = skill.relative_path
                entry["risk_score"] = result.get("risk_score", 0)
                entry["risk_severity"] = result.get("risk_severity", "LOW")
                entry["finding_count"] = finding_count
                entry["execution_successful"] = result.get("execution_successful", True)
                combined_skills.append(entry)
                entry["transitive_finding_count"] = result.get("transitive_finding_count", 0)
                entry["transitive_sources"] = result.get("transitive_sources", [])
        if omitted_skill_count:
            combined_skills.append(
                {
                    "omitted": True,
                    "omitted_count": omitted_skill_count,
                    "reason": "aggregate_scan_limit",
                }
            )
        rendered = json.dumps(combined, indent=2)
        if len(rendered) > _MULTI_SKILL_MAX_REPORT_CHARACTERS:
            analysis_incomplete = True
            aggregate_completeness, aggregate_limitations = _mark_recursive_output_limited(
                aggregate_completeness,
            )
            combined = {
                "multi_skill": True,
                "skill_count": len(skills),
                "max_risk_score": max_score,
                "execution_successful": not execution_failed,
                "risk_recommendation": (
                    "DO_NOT_INSTALL"
                    if execution_failed or max_score > RISK_THRESHOLD
                    else "CAUTION"
                ),
                "analysis_completeness": aggregate_completeness,
                "skills_scanned": len(processed_skills),
                "skills_omitted": omitted_skill_count,
                "skills_output_omitted": len(processed_skills),
                "public_finding_records": 0,
                "transitive_finding_count": transitive_finding_count,
                "transitive_sources": [],
                "skills": [
                    {
                        "omitted": True,
                        "omitted_count": len(processed_skills),
                        "reason": "aggregate_output_limit",
                    }
                ],
            }
            rendered = json.dumps(combined, indent=2)
        _ensure_recursive_output_bound(rendered)
        Path(output).write_text(rendered, encoding="utf-8")
        console.print(f"[green]Combined report saved to:[/green] {output}")
    elif output and format == FormatChoice.sarif:
        merged_sarif = _multi_skill_sarif_report(
            processed_skills,
            results,
            aggregate_completeness,
        )
        rendered = json.dumps(merged_sarif, indent=2)
        if len(rendered) > _MULTI_SKILL_MAX_REPORT_CHARACTERS:
            analysis_incomplete = True
            aggregate_completeness, aggregate_limitations = _mark_recursive_output_limited(
                aggregate_completeness,
            )
            merged_sarif = _multi_skill_sarif_report([], [], aggregate_completeness)
            rendered = json.dumps(merged_sarif, indent=2)
        _ensure_recursive_output_bound(rendered)
        Path(output).write_text(rendered, encoding="utf-8")
        console.print(f"[green]Combined report saved to:[/green] {output}")
    elif output:
        sections: list[str] = []
        for skill, result in zip(processed_skills, results, strict=True):
            if "error" not in result:
                sections.append(f"--- {skill.relative_path} ---\n\n{_result_body(result)}")
        if analysis_incomplete:
            sections.append(
                "--- Recursive Inspection Completeness ---\n\n"
                "Status: partial\n\n" + "\n".join(f"- {item}" for item in aggregate_limitations)
            )
        rendered = "\n\n".join(sections)
        if len(rendered) > _MULTI_SKILL_MAX_REPORT_CHARACTERS:
            analysis_incomplete = True
            aggregate_completeness, aggregate_limitations = _mark_recursive_output_limited(
                aggregate_completeness,
            )
            rendered = (
                "--- Recursive Inspection Completeness ---\n\n"
                "Status: partial\n\n" + "\n".join(f"- {item}" for item in aggregate_limitations)
            )
        _ensure_recursive_output_bound(rendered)
        Path(output).write_text(rendered, encoding="utf-8")
        console.print(f"[green]Combined report saved to:[/green] {output}")

    for result in results:
        cleanup_result(result)

    if execution_failed:
        raise typer.Exit(code=2)
    if fail_on_incomplete and analysis_incomplete:
        raise typer.Exit(code=1)
    if max_score > RISK_THRESHOLD:
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
        err_console.print(f"[red]Error:[/red] {e}")
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
        state["baseline_path"] = os.path.abspath(output.expanduser())
        result = graph.invoke(state)
        findings = effective_findings(result)
        data = build_baseline_dict(
            findings,
            reason=reason,
            # Exact fingerprints must use the same local-only cache that fed
            # deterministic analyzers. The provider-safe cache intentionally
            # omits hidden, binary, and nested content.
            file_cache=result.get("local_file_cache") or result.get("file_cache") or {},
            scanner_version=__version__,
        )
        dump_baseline(data, output)
        console.print(
            f"[green]Wrote baseline with {len(findings)} suppressed finding(s) to:[/green] {output}"
        )
    except typer.Exit:
        raise
    except (FileNotFoundError, ValueError) as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2) from e
    except Exception as e:
        if verbose:
            err_console.print_exception()
        else:
            err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2) from e
    finally:
        if result is not None:
            cleanup_result(result)


if __name__ == "__main__":
    app()
