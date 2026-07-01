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

"""Batch scanner for SkillSpector with multilingual enhancement and concurrent execution.

Scans a directory of AI agent skills in parallel (configurable worker pool)
and produces a single aggregated report (terminal / JSON / Markdown).  For
non-English skills, runs a targeted LLM gap-fill pass covering 8 vulnerability
categories that have no semantic-analyzer equivalent.

Concurrency model
-----------------
Each skill runs the full ``graph.invoke(state)`` pipeline in a dedicated
thread via :class:`~concurrent.futures.ThreadPoolExecutor`.  The number of
parallel workers is controlled by ``--workers`` (default 4).  A 90-second
per-skill timeout prevents stalled workers from blocking the batch.  This
sits on top of two built-in parallelism layers:

* **Layer 1** — 20 analyzers fan-out inside the LangGraph (per-skill)
* **Layer 2** — :meth:`~skillspector.llm_analyzer_base.LLMAnalyzerBase.arun_batches`
  with ``Semaphore(10)`` (per-analyzer)
* **Layer 3** — ``ThreadPoolExecutor(max_workers)`` across skills (this module)

API rate-limit protection is provided by the :class:`~.api_pool.ApiKeyPool`
for **all** LLM calls — graph-internal analyzers, meta-analyzer, and gap-fill
alike.  The pool is wired in via :func:`~.runner.set_api_pool` (monkey-patches
:func:`~skillspector.llm_utils.get_chat_model`) before any scan work starts.

Usage::

    python -m contrib.multilingual.batch_scan ./skills/ --no-llm
    python -m contrib.multilingual.batch_scan ./skills/ -f json -o report.json
    python -m contrib.multilingual.batch_scan ./skills/ --lang zh --workers 8
"""

from __future__ import annotations

# -- .env must load BEFORE any skillspector imports, because constants.py
#    reads SKILLSPECTOR_MODEL / SKILLSPECTOR_PROVIDER at import time.
try:
    import dotenv as _dotenv  # noqa: I001
except ImportError:
    pass
else:
    _dotenv.load_dotenv(_dotenv.find_dotenv(usecwd=True), override=True)

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from pathlib import Path
from skillspector.constants import MODEL_CONFIG
from skillspector.logging_config import set_level

from .annotation import annotate_findings
from .api_pool import create_api_key_pool_from_env
from .detection import detect_skill_language
from .discovery import discover_skills
from .gap_fill import run_gap_fill
from .reports import _format_json as format_json
from .reports import _format_markdown as format_markdown
from .reports import _format_terminal as format_terminal
from .runner import run_one

# Directories skipped during file reads (same set as build_context._SKIP_DIRS).
_SKIP_DIRS: frozenset[str] = frozenset(
    {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".pytest_cache"}
)

# Progress-print lock — Rich consoles are not thread-safe; serialize output
# from the main thread via this lock.
_print_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_skill_files(skill_dir: Path) -> dict[str, str]:
    """Lightweight file read for language detection and gap-fill.

    Mirrors the file-walk rules in
    :func:`skillspector.nodes.build_context._walk_skill_files`.
    """
    file_cache: dict[str, str] = {}
    for item in skill_dir.rglob("*"):
        if not item.is_file():
            continue
        if any(skip in item.parts for skip in _SKIP_DIRS):
            continue
        if item.name.startswith(".") and not item.name.startswith(".claude"):
            continue
        try:
            file_cache[str(item.relative_to(skill_dir))] = item.read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
    return file_cache


def _resolve_language(skill_dir: Path, cli_lang: str) -> str:
    """Determine the language for a skill directory.

    When *cli_lang* is ``"auto"``, reads files and runs heuristic
    detection.  Otherwise returns *cli_lang* as-is.
    """
    if cli_lang != "auto":
        return cli_lang
    fc = _read_skill_files(skill_dir)
    if not fc:
        return "en"
    return detect_skill_language(fc)


def _scan_skill(
    skill_dir: Path,
    root: Path,
    *,
    use_llm: bool,
    lang: str,
    require_llm: bool,
    api_pool=None,
) -> tuple[dict[str, object], str | None, str]:
    """Scan a single skill through the full pipeline.

    Returns
    -------
    (entry, error_message_or_None, relative_name)
    """
    try:
        rel_name = str(skill_dir.relative_to(root))
    except ValueError:
        rel_name = skill_dir.name

    # Core scan via the LangGraph graph
    entry, error_msg = run_one(
        skill_dir,
        root,
        use_llm=use_llm,
        detected_language=lang,
    )

    # Gap-fill for non-English skills (post-graph, appends to issues)
    if lang != "en" and use_llm and not error_msg:
        fc = _read_skill_files(skill_dir)
        gap_findings = run_gap_fill(
            fc, lang, model=MODEL_CONFIG.get("default"), api_pool=api_pool
        )
        if gap_findings:
            existing = list(entry.get("issues", []))
            new_issues = annotate_findings(
                [f.to_dict() for f in gap_findings], lang
            )
            entry["issues"] = existing + new_issues  # type: ignore[operator]
        # Patch enhancements so reports can show what was applied
        entry["enhancements"]["gap_fill_applied"] = True
        entry["enhancements"]["gap_fill_findings"] = len(gap_findings)

    return entry, error_msg, rel_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the batch scanner CLI."""
    # -- DeepSeek compatibility patches (scoped context manager) --------------
    # Patches are active for the entire scan and restored on exit — even if
    # an exception occurs.  Pattern: Save → Patch → Yield → Restore (finally).
    from .runner import deepseek_compat

    with deepseek_compat():
        _main_impl()


def _main_impl() -> None:
    """Body of main(), wrapped by deepseek_compat context manager."""
    # -- Windows Unicode support ---------------------------------------------
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    # -- Rich detection -------------------------------------------------------
    try:
        from rich.console import Console
    except ImportError:
        Console = None  # type: ignore[assignment]  # noqa: N806

    c = Console() if Console is not None else None

    def _print(*args: object, **kwargs: object) -> None:
        """Print through Rich when available, falling back to plain text."""
        if c:
            c.print(*args, **{k: v for k, v in kwargs.items() if k != "file"})
        else:
            msg = " ".join(str(a) for a in args)
            file = kwargs.get("file")
            if file:
                print(msg, file=file)  # type: ignore[arg-type]
            else:
                print(msg)

    # -- CLI arguments -------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Batch-scan a directory of AI agent skills with SkillSpector.",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing skill subdirectories (each with a SKILL.md).",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("terminal", "json", "markdown"),
        default="terminal",
        help="Output format (default: terminal).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write report to FILE (default: stdout).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        default=False,
        help="Skip LLM analysis — static patterns only.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel scan workers (default: 4).  "
        "Reduce to 1 for free-tier API keys, increase for enterprise tiers.  "
        "Skills that time out (90s) are skipped; other workers continue.",
    )
    parser.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--lang",
        choices=("auto", "en", "zh", "ja", "ko"),
        default="auto",
        help="Expected skill language (default: auto-detect).",
    )
    parser.add_argument(
        "--require-llm",
        action="store_true",
        default=True,
        help="Require LLM for non-English skills (default).",
    )
    parser.add_argument(
        "--no-require-llm",
        action="store_false",
        dest="require_llm",
        help="Allow non-English scans without LLM (results will be incomplete).",
    )
    args = parser.parse_args()

    if args.verbose:
        set_level("DEBUG")

    # -- Validation ----------------------------------------------------------
    root = args.input_dir.resolve()
    if not root.is_dir():
        _print(f"[red]Error:[/red] {root} is not a directory", file=sys.stderr)
        sys.exit(2)

    skill_dirs = discover_skills(root)
    if not skill_dirs:
        _print(
            "[yellow]No skills found.[/yellow] Each skill must be a subdirectory "
            "containing a SKILL.md file.",
            file=sys.stderr,
        )
        sys.exit(2)

    # -- API Pool (optional — returns None if single-key) --------------------
    api_pool = create_api_key_pool_from_env()
    if api_pool:
        from .runner import set_api_pool
        set_api_pool(api_pool)
    use_llm = not args.no_llm

    # -- Header --------------------------------------------------------------
    pool_note = (
        f", [green]{api_pool.keys_configured} keys "
        f"({api_pool.total_capacity} slots)[/green]"
        if api_pool
        else ""
    )
    _print(
        f"\n[bold]SkillSpector Batch Scan[/bold] — "
        f"{len(skill_dirs)} skill(s) in [dim]{root}[/dim]"
        f"  ([cyan]{args.workers} workers[/cyan]{pool_note})\n"
    )

    # -- Scan (parallel) -----------------------------------------------------
    results: list[dict[str, object]] = []
    errors = 0
    has_high_risk = False

    _sev_colors: dict[str, str] = {
        "LOW": "green",
        "MEDIUM": "yellow",
        "HIGH": "red",
        "CRITICAL": "bold red",
        "ERROR": "red",
    }

    # Pre-resolve languages so worker threads don't contend on file I/O
    lang_map: dict[Path, str] = {}
    for skill_dir in skill_dirs:
        lang_map[skill_dir] = _resolve_language(skill_dir, args.lang)

    total = len(skill_dirs)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                _scan_skill,
                skill_dir,
                root,
                use_llm=use_llm,
                lang=lang_map[skill_dir],
                require_llm=args.require_llm,
                api_pool=api_pool,
            ): idx
            for idx, skill_dir in enumerate(skill_dirs, 1)
        }

        for future in as_completed(future_map):
            idx = future_map[future]
            rel_name = str(skill_dirs[idx - 1].relative_to(root)) if idx <= len(skill_dirs) else "?"
            try:
                entry, error_msg, rel_name = future.result(timeout=90)
            except TimeoutError:
                errors += 1
                with _print_lock:
                    _print(
                        f"  [{idx}/{total}] [cyan]{rel_name}[/cyan] → "
                        f"[red]TIMEOUT (90s)[/red]"
                    )
                # Don't retry — the worker thread is still stuck and a
                # retry would consume another slot.  HTTP-level timeouts
                # (runner.py Patch 6) prevent most hangs from happening.
                continue
            except Exception:
                # Unexpected crash (e.g. asyncio event-loop failure).
                # Don't retry — log and continue.
                errors += 1
                with _print_lock:
                    _print(
                        f"  [{idx}/{total}] [cyan]{rel_name}[/cyan] → "
                        f"[red]CRASH[/red]"
                    )
                continue
            lang = lang_map[skill_dirs[idx - 1]]
            results.append(entry)

            # -- Progress (main thread via lock — safe for Rich) ---------
            with _print_lock:
                # Non-English LLM guard warning
                if lang != "en" and not use_llm and args.require_llm:
                    _print(
                        f"[yellow]WARNING:[/yellow] non-English skill "
                        f"'{rel_name}' ({lang}) scanned with --no-llm. "
                        f"Static pattern recall is reduced for this language. "
                        f"Re-run without --no-llm for full coverage, or use "
                        f"--no-require-llm to suppress this warning.",
                        file=sys.stderr,
                    )

                if error_msg:
                    errors += 1
                    _print(
                        f"  [{idx}/{total}] [cyan]{rel_name}[/cyan] → "
                        f"[red]ERROR: {error_msg}[/red]"
                    )
                else:
                    risk = entry.get("risk_assessment", {})
                    score = risk.get("score", 0)
                    severity = risk.get("severity", "LOW")
                    n_issues = len(entry.get("issues", []))
                    if score > 50:
                        has_high_risk = True
                    color = _sev_colors.get(severity, "")
                    _print(
                        f"  [{idx}/{total}] [cyan]{rel_name}[/cyan] → "
                        f"[{color}]{score}/100 {severity}[/{color}] "
                        f"({n_issues} issue(s))"
                    )

    # -- Sort results by risk score descending -------------------------------
    results.sort(
        key=lambda x: x.get("risk_assessment", {}).get("score", 0),  # type: ignore[no-any-return]
        reverse=True,
    )

    # -- API Pool summary (if active) ----------------------------------------
    if api_pool:
        snap = api_pool.snapshot()
        _parts = [
            f"{snap['total_requests_served']} requests served",
        ]
        if snap.get("peak_active_requests", 0) > 0:
            _parts.append(
                f"peak {snap['peak_active_requests']}/{snap['total_capacity']} slots"
            )
        if snap.get("rate_limits_hit", 0) > 0:
            _parts.append(
                f"{snap['rate_limits_hit']} rate-limit(s), "
                f"{snap['retry_successes']} retried"
            )
        _parts.append(f"{snap['keys_configured']} keys")
        _print(f"\n[dim]API Pool: {', '.join(_parts)}[/dim]")

    # -- Output --------------------------------------------------------------
    fmt = args.format
    if fmt == "terminal":
        report_body = format_terminal(results)
    elif fmt == "json":
        report_body = format_json(results)
    else:
        report_body = format_markdown(results)

    if args.output:
        args.output.write_text(report_body, encoding="utf-8")
        _print(f"\n[green]Batch report saved to:[/green] {args.output}")
    else:
        if fmt == "terminal":
            _print(report_body)
        else:
            sys.stdout.write(report_body + "\n")

    # -- Exit codes ----------------------------------------------------------
    if errors:
        sys.exit(2)
    if has_high_risk:
        sys.exit(1)
    # else: exit 0


if __name__ == "__main__":
    main()
