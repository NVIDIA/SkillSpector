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
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import Annotated

import typer
from langchain_core.runnables import RunnableConfig
from rich.console import Console

from skillspector import __version__, transitive
from skillspector import transitive as transitive_module
from skillspector.cleanup import cleanup_result
from skillspector.constants import RISK_THRESHOLD
from skillspector.graph import graph
from skillspector.logging_config import get_logger, set_level
from skillspector.models import Finding
from skillspector.multi_skill import MultiSkillDetectionResult, detect_skills
from skillspector.nodes.report import report
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

_TRANSITIVE_MAX_TARGETS = 32
_TRANSITIVE_MAX_BYTES = 10 * 1024 * 1024
_TRANSITIVE_MAX_SECONDS = 60.0


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


@dataclass(slots=True)
class _CachedTransitiveResult:
    filtered_findings: list[Finding]
    findings: list[Finding]
    llm_call_log: list[dict[str, object]]
    components: list[str]
    component_metadata: list[dict[str, object]]
    file_cache: dict[str, str]
    has_executable_scripts: bool
    refs: list[str]


@dataclass(slots=True)
class _TransitiveTraversalState:
    cache: dict[str, _CachedTransitiveResult] = field(default_factory=dict)
    budget: _TransitiveBudget = field(default_factory=_TransitiveBudget)
    started_at: float | None = None
    scanned_targets: int = 0
    scanned_bytes: int = 0
    truncation_reasons: list[str] = field(default_factory=list)

    def note_truncation(self, reason: str) -> None:
        if reason not in self.truncation_reasons:
            self.truncation_reasons.append(reason)

    def _ensure_started(self) -> None:
        if self.started_at is None:
            self.started_at = monotonic()

    def can_scan_more(self) -> bool:
        self._ensure_started()
        if self.truncation_reasons:
            return False
        if self.scanned_targets >= self.budget.max_targets:
            self.note_truncation(f"target budget {self.budget.max_targets} reached")
            return False
        if self.remaining_bytes() <= 0:
            self.note_truncation(f"byte budget {self.budget.max_bytes} reached")
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
        return max(0.0, self.budget.max_seconds - (monotonic() - self.started_at))

    def remaining_bytes(self) -> int:
        return max(0, self.budget.max_bytes - self.scanned_bytes)


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
    transitive: Annotated[
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
    if verbose:
        set_level("DEBUG")

    resolved_path = Path(input_path).resolve()
    transitive_allow_prefix, transitive_deny_prefix = transitive_module.normalize_prefixes(
        transitive_allow_prefix, transitive_deny_prefix
    )
    yara_dir = str(yara_rules_dir.resolve()) if yara_rules_dir else None
    if recursive and resolved_path.is_dir():
        detection = detect_skills(resolved_path)
        if detection.is_multi_skill:
            _scan_multi_skill(
                detection=detection,
                format=format,
                output=output,
                no_llm=no_llm,
                baseline=baseline,
                show_suppressed=show_suppressed,
                transitive_enabled=transitive,
                transitive_depth=transitive_depth,
                transitive_allow_prefix=transitive_allow_prefix,
                transitive_deny_prefix=transitive_deny_prefix,
                yara_dir=yara_dir,
                verbose=verbose,
            )
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
        result = _scan_skill(
            input_path=input_path,
            format=format,
            no_llm=no_llm,
            baseline=baseline,
            yara_rules_dir=Path(yara_dir) if yara_dir else None,
            verbose=verbose,
            show_suppressed=show_suppressed,
            transitive_enabled=transitive,
            transitive_depth=transitive_depth,
            transitive_allow_prefix=transitive_allow_prefix,
            transitive_deny_prefix=transitive_deny_prefix,
        )
        _write_result(result, output, format)

        if (result.get("risk_score") or 0) > RISK_THRESHOLD:
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


def _coerce_llm_call_log(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _coerce_file_cache(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(path): content
        for path, content in value.items()
        if isinstance(path, str) and isinstance(content, str)
    }


def _transitive_component_key(source_url: str | None, path: str) -> str:
    return f"{source_url}::{path}" if source_url else path


def _decorate_component_metadata(
    metadata: list[dict[str, object]], source_url: str | None
) -> list[dict[str, object]]:
    decorated: list[dict[str, object]] = []
    for item in metadata:
        path = str(item.get("path", ""))
        entry = {**item, "coverage_key": _transitive_component_key(source_url, path)}
        if source_url:
            entry["source_url"] = source_url
        decorated.append(entry)
    return decorated


def _source_aware_components(paths: list[str], source_url: str | None) -> list[str]:
    return [_transitive_component_key(source_url, path) for path in paths]


def _source_aware_file_cache(file_cache: dict[str, str], source_url: str | None) -> dict[str, str]:
    return {
        _transitive_component_key(source_url, path): content for path, content in file_cache.items()
    }


def _component_identity(item: dict[str, object]) -> str:
    coverage_key = item.get("coverage_key")
    if isinstance(coverage_key, str) and coverage_key:
        return coverage_key
    path = str(item.get("path", ""))
    source_url = item.get("source_url")
    return _transitive_component_key(source_url if isinstance(source_url, str) else None, path)


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


def _cache_transitive_result(
    target: str, child_result: dict[str, object]
) -> _CachedTransitiveResult:
    child_file_cache = _coerce_file_cache(child_result.get("file_cache"))
    child_metadata = _decorate_component_metadata(
        _coerce_component_metadata(child_result.get("component_metadata")), target
    )
    return _CachedTransitiveResult(
        filtered_findings=_coerce_findings_list(child_result.get("filtered_findings")),
        findings=_coerce_findings_list(child_result.get("findings")),
        llm_call_log=_coerce_llm_call_log(child_result.get("llm_call_log")),
        components=_source_aware_components(
            _coerce_str_path_list(child_result.get("components")), target
        ),
        component_metadata=child_metadata,
        file_cache=_source_aware_file_cache(child_file_cache, target),
        has_executable_scripts=bool(child_result.get("has_executable_scripts", False))
        or any(bool(entry.get("executable", False)) for entry in child_metadata),
        refs=transitive.extract_external_refs(child_file_cache),
    )


def _run_graph_scan(
    input_path: str,
    format: FormatChoice,
    no_llm: bool,
    yara_dir: str | None = None,
    baseline: Path | None = None,
    show_suppressed: bool = False,
    transitive_traversal: _TransitiveTraversalState | None = None,
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
    trace_config = _build_trace_config(input_path, format, no_llm)
    return graph.invoke(state, config=trace_config)


def _annotate_transitive_findings(
    findings: list[Finding],
    source_url: str,
    transitive_depth: int,
) -> list[Finding]:
    return [
        replace(finding, transitive_depth=transitive_depth, source_url=source_url)
        for finding in findings
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
        report_result = report(initial_result)
        report_result["temp_dir_for_cleanup"] = initial_result.get("temp_dir_for_cleanup")
        report_result["transitive_finding_count"] = 0
        report_result["transitive_sources"] = []
        report_result["transitive_targets_scanned"] = 0
        report_result["transitive_bytes_scanned"] = 0
        report_result["transitive_truncated"] = False
        report_result["transitive_truncation_reasons"] = []
        return report_result

    if traversal is None:
        traversal = _TransitiveTraversalState(
            cache=scan_cache if scan_cache is not None else {},
            budget=budget if budget is not None else _TransitiveBudget(),
        )
    elif scan_cache is not None and traversal.cache is not scan_cache:
        traversal.cache = scan_cache
    transitive_sources: set[str] = set()
    merged_filtered_findings: list[Finding] = _coerce_findings_list(
        initial_result.get("filtered_findings")
    )
    merged_findings: list[Finding] = _coerce_findings_list(initial_result.get("findings"))
    merged_llm_call_log: list[dict[str, object]] = _coerce_llm_call_log(
        initial_result.get("llm_call_log")
    )
    merged_components = _source_aware_components(
        _coerce_str_path_list(initial_result.get("components")), None
    )
    file_cache = _coerce_file_cache(initial_result.get("file_cache"))
    merged_file_cache = _source_aware_file_cache(file_cache, None)
    component_metadata = _decorate_component_metadata(
        _coerce_component_metadata(initial_result.get("component_metadata")), None
    )
    has_executable_scripts = bool(initial_result.get("has_executable_scripts", False))

    frontier: list[tuple[int, list[str]]] = [(1, transitive.extract_external_refs(file_cache))]

    while frontier:
        if not traversal.can_scan_more():
            break
        current_depth, refs = frontier.pop(0)
        targets = transitive.plan_transitive_targets(
            refs=refs,
            visited=visited,
            current_depth=current_depth,
            max_depth=max_depth,
            allow_prefixes=transitive_allow_prefix,
            deny_prefixes=transitive_deny_prefix,
        )
        for target in targets:
            if not traversal.can_scan_more():
                break
            try:
                cached = traversal.cache.get(target)
                child_result: dict[str, object] | None = None
                if cached is None:
                    child_result = _run_graph_scan(
                        input_path=target,
                        format=format,
                        no_llm=no_llm,
                        yara_dir=yara_dir,
                        baseline=baseline,
                        show_suppressed=show_suppressed,
                        transitive_traversal=traversal,
                    )
                    cached = _cache_transitive_result(target, child_result)
                    traversal.cache[target] = cached
                    traversal.record_scan()
                transitive_sources.add(target)
                merged_llm_call_log.extend(cached.llm_call_log)
                merged_filtered_findings.extend(
                    _annotate_transitive_findings(
                        cached.filtered_findings, source_url=target, transitive_depth=current_depth
                    )
                )
                merged_findings.extend(
                    _annotate_transitive_findings(
                        cached.findings, source_url=target, transitive_depth=current_depth
                    )
                )

                component_metadata.extend(cached.component_metadata)
                if cached.has_executable_scripts:
                    has_executable_scripts = True
                merged_components.extend(cached.components)
                merged_file_cache.update(cached.file_cache)

                if current_depth < max_depth:
                    frontier.append((current_depth + 1, cached.refs))
            except Exception as e:
                if format == FormatChoice.json:
                    logger.warning("Transitive scan failed for %s: %s", target, e)
                else:
                    console.print(
                        f"[yellow]Warning:[/yellow] Transitive scan failed for {target}: {e}"
                    )
            finally:
                if child_result is not None:
                    cleanup_result(child_result)

    merged_result: dict[str, object] = {
        **initial_result,
        "filtered_findings": merged_filtered_findings,
        "findings": merged_findings,
        "components": merged_components,
        "component_metadata": _merge_unique_component_metadata(component_metadata),
        "file_cache": merged_file_cache,
        "has_executable_scripts": has_executable_scripts,
        "llm_call_log": merged_llm_call_log,
        "baseline": baseline,
        "show_suppressed": show_suppressed,
        "transitive_targets_scanned": traversal.scanned_targets,
        "transitive_bytes_scanned": traversal.scanned_bytes,
        "transitive_truncated": bool(traversal.truncation_reasons),
        "transitive_truncation_reasons": traversal.truncation_reasons,
    }
    report_result = report(merged_result)
    report_result["temp_dir_for_cleanup"] = initial_result.get("temp_dir_for_cleanup")
    active_findings = report_result.get("active_findings") or []
    report_result["transitive_finding_count"] = sum(
        1
        for finding in active_findings
        if isinstance(finding, Finding) and finding.source_url is not None
    )
    report_result["transitive_sources"] = sorted(transitive_sources)
    report_result["transitive_targets_scanned"] = traversal.scanned_targets
    report_result["transitive_bytes_scanned"] = traversal.scanned_bytes
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
    result = _run_graph_scan(
        input_path=input_path,
        format=format,
        no_llm=no_llm,
        yara_dir=yara_dir,
        baseline=baseline,
        show_suppressed=show_suppressed,
        transitive_traversal=None,
    )
    if not transitive_enabled:
        return result
    if transitive_traversal is None:
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


def _scan_multi_skill(
    detection: MultiSkillDetectionResult,
    format: FormatChoice,
    output: Path | None,
    no_llm: bool,
    baseline: Path | None,
    show_suppressed: bool,
    transitive_enabled: bool,
    transitive_depth: int,
    transitive_allow_prefix: tuple[str, ...] | list[str] | None,
    transitive_deny_prefix: tuple[str, ...] | list[str] | None,
    yara_dir: str | None,
    verbose: bool,
) -> None:
    """Scan each detected sub-skill independently and produce a combined report."""
    skills = detection.skills
    console.print(f"[bold]Multi-skill directory detected:[/bold] {len(skills)} skills found\n")

    shared_transitive_cache: dict[str, _CachedTransitiveResult] = {}
    shared_transitive_traversal = _TransitiveTraversalState(cache=shared_transitive_cache)
    results: list[dict[str, object]] = []
    max_score = 0
    transitive_finding_count = 0
    transitive_sources: set[str] = set()

    for i, skill in enumerate(skills, 1):
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
            results.append(result)
            score = result.get("risk_score") or 0
            if isinstance(score, int) and score > max_score:
                max_score = score
            transitive_finding_count += int(result.get("transitive_finding_count") or 0)
            for source in _coerce_str_path_list(result.get("transitive_sources")):
                transitive_sources.add(source)
            severity = result.get("risk_severity") or "LOW"
            console.print(f"         Score: {score}/100 ({severity})\n")
        except Exception as e:
            console.print(f"         [red]Error:[/red] {e}\n")
            results.append({"skill_name": skill.name, "error": str(e)})

    # Existing direct output behavior remains, but shared traversal and visited state
    # are now handled by _scan_skill, including transitive helper path.
    _print_multi_summary(skills, results)

    if output and format == FormatChoice.json:
        combined = {
            "multi_skill": True,
            "skill_count": len(skills),
            "max_risk_score": max_score,
            "transitive_finding_count": transitive_finding_count,
            "transitive_sources": sorted(transitive_sources),
            "skills": [],
        }
        for skill, result in zip(skills, results, strict=True):
            if "error" in result:
                combined["skills"].append({"name": skill.name, "error": result["error"]})
            else:
                payload = _recursive_json_payload(result) or {}
                entry = {
                    "name": skill.name,
                    "path": skill.relative_path,
                    "risk_score": result.get("risk_score", 0),
                    "risk_severity": result.get("risk_severity", "LOW"),
                    "finding_count": len(
                        result.get("filtered_findings") or result.get("findings") or []
                    ),
                    "transitive_finding_count": result.get("transitive_finding_count", 0),
                    "transitive_sources": result.get("transitive_sources", []),
                }
                entry.update(payload)
                entry["name"] = skill.name
                entry["path"] = skill.relative_path
                entry["risk_score"] = result.get("risk_score", 0)
                entry["risk_severity"] = result.get("risk_severity", "LOW")
                entry["finding_count"] = len(
                    result.get("filtered_findings") or result.get("findings") or []
                )
                entry["transitive_finding_count"] = result.get("transitive_finding_count", 0)
                entry["transitive_sources"] = result.get("transitive_sources", [])
                combined["skills"].append(entry)
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

    if max_score > RISK_THRESHOLD:
        raise typer.Exit(code=1)


def _print_multi_summary(skills: list, results: list[dict[str, object]]) -> None:
    console.print("\n[bold]=== Multi-Skill Summary ===[/bold]\n")
    console.print(f"  {'Skill':<30} {'Score':<8} {'Severity':<12} {'Findings':<10}")
    console.print(f"  {'-' * 30} {'-' * 8} {'-' * 12} {'-' * 10}")

    for skill, result in zip(skills, results, strict=True):
        if "error" in result:
            console.print(f"  {skill.name:<30} {'ERROR':<8} {'n/a':<12} {'n/a':<10}")
            continue
        score = result.get("risk_score", 0)
        severity = result.get("risk_severity", "LOW")
        filtered = result.get("filtered_findings") or result.get("findings")
        finding_count = len(filtered) if isinstance(filtered, list) else 0
        console.print(f"  {skill.name:<30} {score:<8} {severity:<12} {finding_count:<10}")


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
            cleanup_result(result)


if __name__ == "__main__":
    app()
