# SkillSpector Benchmark

A standalone harness that runs SkillSpector over a benchmark dataset
([MalSkillBench](https://github.com/NVIDIA/MalSkillBench) today) and scores its
classifications into DuckDB, so you can measure precision/recall/accuracy.

## Why this is a separate project

This is a **dev/eval tool**, not part of the shipped `skillspector` package. It
lives in its own uv project with its own `pyproject.toml`, lockfile, and venv,
and depends on `skillspector` as an **editable path dependency** (`../`). That
means:

- Nothing here touches the root `pyproject.toml` / lockfile, so upstream merges
  from `NVIDIA/skillspector` stay conflict-free.
- The harness-only dependencies (`duckdb`, `filelock`, `tqdm`,
  `aws-bedrock-token-generator`) never ship to `skillspector` end users.
- Edits to `src/skillspector` are picked up immediately (editable install), so
  you always benchmark your working tree.

## Running

From the repo root (no `cd` needed):

```bash
uv run --directory benchmark benchmark /path/to/MalSkillBench/Dataset
```

Or from inside this directory:

```bash
cd benchmark
uv run benchmark /path/to/MalSkillBench/Dataset
uv run benchmark .../Dataset/Prompts/indirect-injection -o out.duckdb
uv run benchmark .../Dataset/Skills/malware --limit 50 --workers 8
uv run benchmark .../Dataset --no-llm        # static analysis only
```

> Note: the console-script name (`benchmark`) intentionally matches the package
> directory, so there is **no** `__main__.py` — adding one would make `uv run
> benchmark` execute the directory instead of the installed entry point. Invoke
> it via the `benchmark` command (as above) or `.venv/bin/benchmark` directly.

### Key flags

| Flag | Meaning |
|------|---------|
| `-o, --output` | DuckDB output file (default `benchmark_<id>.duckdb`) |
| `--no-llm` | static analysis only (no LLM calls) |
| `--categories` | comma-separated unit categories to scan; default `skill,code`. **Prompts are excluded by default** — they're raw prompt-injection samples, not the repo config files this tool classifies (PI embedded in a skill is still covered via the `skill` units). Pass `--categories skill,code,prompt` to include them. |
| `--limit N` | cap units **per group** — a unit's parent directory (e.g. `--limit 20` on `Dataset/Skills` keeps 20 from `Skills/malware` *and* 20 from `Skills/benign`); 0 = no cap |
| `--workers N` | concurrent scan processes (default 8) |
| `--overwrite` | start fresh instead of resuming an existing DB |
| `--auth-wait-seconds` | how long to pause for `aws sso login` on a mid-run SSO expiry |

A run is **resumable**: re-running the same command against an existing output
DB skips already-classified units and re-scans only failures.

## Output

DuckDB tables `runs`, `units` (ground truth), `classifications` (SkillSpector
verdict), `issues`, `components`, plus an `evaluation` view labeling every scan
`TP`/`FP`/`TN`/`FN`/`ERROR`:

```bash
duckdb benchmark_<id>.duckdb "SELECT outcome, count(*) FROM evaluation GROUP BY 1"
```

## Inspecting classifier performance

The `queries/` directory holds ready-made DuckDB queries for evaluating how well
SkillSpector classified — precision/recall/F1 breakdowns, the false-negative and
false-positive lists worth eyeballing, run-health checks, and tuning aids. Run
one against an output DB with `.read`:

```bash
duckdb -readonly benchmark_<id>.duckdb ".read queries/01_overview.sql"
```

Each file is self-documenting (header comment explains what it shows) and, except
`01_overview.sql`, defaults to the **most recent run** in the DB — edit the `run`
CTE at the top to target a different run or span all of them.

| Query | What it answers |
|-------|-----------------|
| `01_overview.sql` | Per-run config + confusion matrix + precision/recall/F1/accuracy (compare runs) |
| `02_metrics_by_category.sql` | Metrics split by skill / code / prompt |
| `03_metrics_by_attack_vector.sql` | Metrics split by CI / PI / MIXED |
| `04_metrics_by_corpus.sql` | Metrics split by source corpus |
| `05_recall_by_behavior.sql` | Which attack behaviors (B1..B15) evade detection, worst first |
| `06_false_negatives.sql` | Malware that was cleared — the critical misses |
| `07_false_positives.sql` | Benign units flagged, with the rules that fired |
| `08_errors.sql` | Scans that failed (invalidate metrics if high) |
| `09_classification_status.sql` | LLM vs static vs **fallback** (catches "LLM didn't actually run") |
| `10_risk_score_distribution.sql` | Score histogram by ground truth — does the score separate classes? |
| `11_threshold_sweep.sql` | P/R/F1 across hypothetical `risk_score` cutoffs (tune the verdict) |
| `12_scan_timing.sql` | Timing percentiles + slowest scans |
| `13_label_coverage.sql` | How ground-truth labels were resolved (bounds trust in 03/05) |
| `14_top_rules.sql` | Which rules fire, on malicious vs benign (false-positive drivers) |

## Layout

```
benchmark/
  main.py                  # argparse + run orchestration
  config.py                # provider/run constants + parent-process env wiring
  models.py                # Unit, ScanResult
  auth.py                  # Bedrock bearer-token manager (cross-process cache)
  db.py                    # DuckDB schema + result persistence
  runner.py                # the classifier seam: prepare a unit, run SkillSpector
  utils.py                 # cross-cutting helpers
  dataset_handler/
    base.py                # DatasetHandler abstraction
    malskillbench.py       # MalSkillBench implementation
    __init__.py            # handler registry + discover() dispatch
```

## Adding another dataset

Discovery is behind an interface so the harness isn't locked to MalSkillBench:

1. Subclass `DatasetHandler` (in `dataset_handler/base.py`), implementing
   `matches(root)` and `discover(root) -> list[Unit]`.
2. Register it in `dataset_handler/__init__.py`'s `_HANDLERS` list (most
   specific first; MalSkillBench is the default fallback).

Everything downstream (runner, DB, scoring) works in terms of `Unit` /
`ScanResult`, so a new handler is the only code a new benchmark needs.
