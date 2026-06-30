"""The classifier seam: prepare a unit, run SkillSpector, capture the result.

The SkillSpector import is intentionally local (inside ``run_skillspector``) so
the module loads fast in every spawned worker and can be repurposed for other
classifiers (write a sibling ``run_<other>`` and dispatch).
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import shutil
import signal
import tempfile
from datetime import UTC, datetime

from .auth import AuthAbortedError, token_manager
from .models import ScanResult, Unit
from .utils import materialized_path, quiet_logging


class _ScanTimeoutError(Exception):
    """Raised by the SIGALRM handler when a single scan exceeds its budget."""


def _alarm(_signum, _frame):
    raise _ScanTimeoutError


def run_skillspector(target: pathlib.Path, use_llm: bool) -> dict:
    """Classify one prepared directory IN-PROCESS; return the JSON report dict."""
    quiet_logging()
    # langchain_core re-enables its (Pending)DeprecationWarnings at import via
    # surface_langchain_deprecation_warnings(), clobbering any warnings filter
    # (PYTHONWARNINGS, -W, filterwarnings all fail). Swallowing stderr for the
    # one-time, per-process import is the only reliable silence. Scoped to the
    # import alone so real runtime errors during the scan still reach stderr.
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        from skillspector.graph import graph

    state = {"input_path": str(target), "output_format": "json", "use_llm": use_llm}
    result = graph.invoke(state)
    temp_dir = result.get("temp_dir_for_cleanup")
    if temp_dir and isinstance(temp_dir, str):
        shutil.rmtree(temp_dir, ignore_errors=True)
    return json.loads(result.get("report_body") or "{}")


def _classify_unit(unit: Unit, use_llm: bool) -> dict:
    """Materialize the unit if needed, then classify the directory."""
    if not unit.materialize:
        return run_skillspector(pathlib.Path(unit.source_path), use_llm)
    with tempfile.TemporaryDirectory(prefix="msb_") as tmp:
        root = pathlib.Path(tmp)
        for rel, content in unit.materialize.items():
            dest = materialized_path(root, rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
        return run_skillspector(root, use_llm)


def _report_to_result(unit: Unit, report: dict, elapsed: float) -> ScanResult:
    risk = report.get("risk_assessment", {})
    meta = report.get("metadata", {})
    issues = report.get("issues", []) or []
    return ScanResult(
        unit=unit,
        scan_status="ok",
        scan_seconds=elapsed,
        risk_score=risk.get("score"),
        risk_severity=risk.get("severity"),
        risk_recommendation=risk.get("recommendation"),
        num_issues=len(issues),
        llm_requested=bool(meta.get("llm_requested")),
        llm_available=bool(meta.get("llm_available")),
        issues=issues,
        components=report.get("components", []) or [],
    )


def scan_worker(unit: Unit, cfg: dict) -> ScanResult:
    """Worker entrypoint: set up creds, enforce a timeout, classify, return."""
    start = datetime.now(UTC)
    timeout = cfg.get("timeout") or 0
    # setitimer (not alarm) so a sub-second --timeout isn't truncated to 0 by
    # alarm(int(timeout)), which would CANCEL the timer instead of arming it.
    use_alarm = timeout and hasattr(signal, "setitimer")
    try:
        if cfg["use_llm"] and cfg["mint_bedrock"]:
            # Token fetch is BEFORE the alarm so a pause-for-`aws sso login`
            # isn't killed by the per-scan timeout.
            os.environ["OPENAI_API_KEY"] = token_manager(cfg["region"]).token(wait=True)
        if use_alarm:
            signal.signal(signal.SIGALRM, _alarm)
            signal.setitimer(signal.ITIMER_REAL, timeout)
        report = _classify_unit(unit, cfg["use_llm"])
    except AuthAbortedError as e:
        return ScanResult(unit=unit, scan_status="auth_failed", error_message=str(e))
    except _ScanTimeoutError:
        return ScanResult(
            unit=unit,
            scan_status="timeout",
            scan_seconds=float(timeout),
            error_message=f"scan exceeded {timeout}s",
        )
    except Exception as e:  # noqa: BLE001 - record, never crash the worker
        return ScanResult(
            unit=unit,
            scan_status="error",
            scan_seconds=(datetime.now(UTC) - start).total_seconds(),
            error_message=f"{type(e).__name__}: {e}"[:2000],
        )
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
    return _report_to_result(unit, report, (datetime.now(UTC) - start).total_seconds())
