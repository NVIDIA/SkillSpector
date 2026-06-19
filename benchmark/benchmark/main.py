"""Run SkillSpector over a MalSkillBench Dataset tree and record results to DuckDB.

Each *unit* is scanned in-process via SkillSpector's LangGraph workflow (one
fresh worker process per scan) and the scan -- plus its resolved ground-truth
label and SkillSpector's verdict, issues and components -- is written to DuckDB
so you can measure how well SkillSpector classifies.

Usage:
    benchmark /path/to/MalSkillBench/Dataset
    benchmark .../Dataset/Prompts/indirect-injection -o out.duckdb
    benchmark .../Dataset/Skills/malware --limit 50 --workers 8
    benchmark .../Dataset --no-llm        # static analysis only

(From the repo root without cd: ``uv run --directory benchmark benchmark ...``.)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime

from tqdm import tqdm

from .config import SCAN_TIMEOUT_SECONDS, configure_run
from .dataset_handler import discover
from .db import open_db, print_summary, purge_incomplete, record_result
from .models import ScanResult
from .runner import scan_worker
from .utils import quiet_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dataset_path", help="MalSkillBench Dataset dir or any subtree of it")
    parser.add_argument("-o", "--output", help="DuckDB output file (default benchmark_<id>.duckdb)")
    parser.add_argument("--no-llm", action="store_true", help="static analysis only (no LLM)")
    parser.add_argument(
        "--categories",
        default="skill,code",
        help="comma-separated unit categories to scan (skill, code, prompt). Prompts "
        "are EXCLUDED by default -- they are raw prompt-injection samples, not the "
        "repo config files this tool classifies. Pass --categories skill,code,prompt "
        "to include them.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="cap units PER GROUP (a unit's parent directory, e.g. Skills/malware "
        "vs Skills/benign); 0 = no cap",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="concurrent scan processes (default 8)"
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=1,
        help="scans per worker process before it is recycled (default 1 = fresh "
        "process per scan, avoids any SkillSpector state leak; raise to amortize "
        "the ~0.8s import cost on fast --no-llm runs)",
    )
    parser.add_argument(
        "--timeout", type=float, default=SCAN_TIMEOUT_SECONDS, help="per-scan timeout seconds"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="start fresh even if the output DB exists (default: resume it)",
    )
    parser.add_argument(
        "--auth-wait-seconds",
        type=float,
        default=1800,
        help="on a mid-run SSO expiry, how long to pause for `aws sso login` "
        "before aborting to a resumable DB (default 1800)",
    )
    args = parser.parse_args(argv)

    root = pathlib.Path(args.dataset_path).resolve()
    if not root.exists():
        print(f"ERROR: {root} does not exist")
        return 2

    print(f"discovering units under {root} ...")
    units = discover(root)

    selected = {c.strip() for c in args.categories.split(",") if c.strip()}
    available = {u.category for u in units}
    unknown = selected - available
    if unknown:
        print(
            f"note: --categories names {sorted(unknown)} not present here (found {sorted(available)})"
        )
    dropped = sum(u.category not in selected for u in units)
    units = [u for u in units if u.category in selected]
    if dropped:
        print(f"excluded {dropped} unit(s) outside --categories={args.categories}")

    if args.limit:
        per_group: dict[str, int] = {}
        capped = []
        for u in units:
            if per_group.get(u.group, 0) < args.limit:
                per_group[u.group] = per_group.get(u.group, 0) + 1
                capped.append(u)
        print(
            f"limiting to {args.limit}/group: kept {len(capped)} of {len(units)} units "
            f"across {len(per_group)} group(s)"
        )
        units = capped
    if not units:
        print(
            "No units found: empty/unsupported directory, or everything was filtered "
            f"out by --categories={args.categories}."
        )
        return 1

    by_cat: dict[str, int] = {}
    mal = sum(u.is_malicious for u in units)
    resolved = sum(u.label is not None for u in units)
    for u in units:
        by_cat[u.category] = by_cat.get(u.category, 0) + 1
    print(
        f"found {len(units)} units: {by_cat}  (malicious={mal}, benign={len(units) - mal}, "
        f"fine-label resolved={resolved})"
    )

    gen_id = uuid.uuid4().hex[:12]
    out_path = pathlib.Path(args.output or f"benchmark_{gen_id}.duckdb").resolve()
    resume = out_path.exists() and not args.overwrite
    if out_path.exists() and args.overwrite:
        out_path.unlink()
        # Drop the sibling write-ahead log too; a stale <db>.wal left behind
        # would otherwise be replayed into the fresh DB on connect.
        (out_path.parent / (out_path.name + ".wal")).unlink(missing_ok=True)

    # Inherited by spawned workers: how long a worker pauses for `aws sso login`.
    os.environ["SKILLSPECTOR_BENCH_AUTH_WAIT"] = str(args.auth_wait_seconds)
    cfg, provider, model, region = configure_run(args.no_llm, args.timeout)

    con, run_id, done, resuming = open_db(out_path, resume)
    if resuming:
        purge_incomplete(con, run_id)
        units_to_scan = [u for u in units if u.unit_path not in done]
        print(
            f"resuming run {run_id}: {len(done)} already classified, {len(units_to_scan)} to scan"
        )
    else:
        run_id = gen_id
        con.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                run_id,
                datetime.now(UTC),
                None,
                str(root),
                provider,
                model,
                region,
                not args.no_llm,
                args.workers,
                args.limit or None,
                len(units),
            ],
        )
        units_to_scan = units

    if not units_to_scan:
        print("nothing to scan: every unit is already classified in this DB.")
        print_summary(con, run_id)
        con.close()
        return 0

    print(
        f"scanning {len(units_to_scan)} units: {args.workers} worker procs, "
        f"{args.max_tasks_per_child} task(s)/child, llm={'off' if args.no_llm else 'on'}"
    )
    aborted = False
    with ProcessPoolExecutor(
        max_workers=args.workers,
        max_tasks_per_child=args.max_tasks_per_child,
        initializer=quiet_logging,
    ) as pool:
        futures = {pool.submit(scan_worker, u, cfg): u for u in units_to_scan}
        for fut in tqdm(as_completed(futures), total=len(units_to_scan), desc="scan", unit="u"):
            u = futures[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001 - record, never abort the run
                res = ScanResult(unit=u, scan_status="error", error_message=repr(e))
            res.unit = u
            record_result(con, run_id, res, args.no_llm)
            if res.scan_status == "auth_failed" and not aborted:
                aborted = True
                pool.shutdown(wait=False, cancel_futures=True)
                break

    con.execute("UPDATE runs SET finished_at = ? WHERE run_id = ?", [datetime.now(UTC), run_id])
    if aborted:
        con.close()
        print(
            "\nABORTED: AWS credentials were not restored in time; progress is saved.\n"
            "  Run `aws sso login`, then re-run the SAME command to resume:\n"
            f"    {pathlib.Path(sys.argv[0]).name} {args.dataset_path} -o {out_path}",
            file=sys.stderr,
        )
        return 3

    print_summary(con, run_id)
    con.close()
    print(f"\nwrote {out_path}")
    print(
        "tables: runs, units, classifications, issues, components | view: evaluation\n"
        f'  e.g.  duckdb {out_path.name} "SELECT outcome, count(*) FROM evaluation GROUP BY 1"'
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
