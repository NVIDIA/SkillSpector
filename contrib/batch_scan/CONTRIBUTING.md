# Contributing — Multilingual Batch Scanner

> For developers who want to set up, test, and extend this module.

---

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp contrib/batch_scan/.env.example .env   # edit with your API keys
```

Verify everything works:
```bash
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ -f terminal --workers 8
```

---

## Project Map

```
contrib/batch_scan/
├── batch_scan.py          # CLI entry + ThreadPoolExecutor (start here)
├── runner.py              # graph.invoke() wrapper + 7 patches + pool wiring (core)
├── gap_fill.py            # GapFillAnalyzer — LLM pass for 8 uncovered rules
├── api_pool.py            # ApiKeyPool — multi-key scheduler + 429 backoff
├── detection.py           # Unicode script-ratio language detection
├── annotation.py          # Finding language-compatibility labels
├── discovery.py           # Recursive SKILL.md finder
├── reports.py             # Terminal / JSON / Markdown formatters
├── CONTRIBUTING.md        # this file
│
├── docs/
│   ├── README.md          # user guide — all commands, test commands, reviewer index
│   ├── DESIGN.md          # architecture — concurrency, patches, dual-patch mechanism
│   ├── REVIEW_RESPONSE.md # PR #100 review response
│   └── archive/           # deep dives, history, future work, pitfalls
│
└── tests/
    ├── test_pool_wiring.py            # smoke — 3-path pool verification
    ├── test_monkeypatch_invasiveness.py # thread isolation, scoping (14 tests)
    ├── test_monkeypatch_fragility.py    # guard verification, deep deps (26 tests)
    ├── docs/
    │   ├── TEST_DESIGN.md             # WHY each suite was designed
    │   ├── TEST_GUIDE.md              # WHAT each file covers + run commands
    │   └── BUGS_FOUND.md              # 16 bugs found & fixed
    └── tests-pro/
        ├── test_api_pool.py           # 45 tests — acquire/release/backoff
        ├── test_gap_fill.py           # 41 tests — JSON parsing, prompt building
        ├── test_runner_patches.py     # 24 tests — context manager, patches
        ├── test_annotation.py         # 10 tests — language compatibility
        ├── random_numbered.py         # main entry point (seed=42)
        └── mutation_max.py            # 30-bug injection framework
```

---

## Running Tests

```bash
# All 164 tests
python contrib/batch_scan/tests/tests-pro/random_numbered.py       # 120 unit (seed=42)
python contrib/batch_scan/tests/test_pool_wiring.py                 # 4 smoke checks
python contrib/batch_scan/tests/test_monkeypatch_invasiveness.py    # 14 thematic
python contrib/batch_scan/tests/test_monkeypatch_fragility.py       # 26 thematic

# Review-themed only
python -m unittest \
  contrib.batch_scan.tests.test_monkeypatch_invasiveness \
  contrib.batch_scan.tests.test_monkeypatch_fragility -v
python contrib/batch_scan/tests/test_pool_wiring.py

# Mutation test
python contrib/batch_scan/tests/tests-pro/mutation_max.py

# End-to-end (fixture suite)
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ -f terminal --workers 8
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ -f terminal --workers 8 --no-llm
```

**Three commands catch most regressions:**
```bash
python contrib/batch_scan/tests/tests-pro/random_numbered.py
python contrib/batch_scan/tests/test_pool_wiring.py
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ -f terminal --workers 8
```

---

## Code Conventions

Match SkillSpector upstream exactly:

- **SPDX header** on every `.py` file
- `from __future__ import annotations` as first import
- Imports: stdlib → third-party → `skillspector.*` → relative (`.`)
- `| None` syntax (not `Optional[X]`)
- `frozenset` / `Final` for module-level constants (`UPPER_SNAKE_CASE`)
- Private helpers: `_lower_snake_case`
- `logger = get_logger(__name__)` in every module
- Comments explain **why**, not what
- Docstrings on all public functions and classes

---

## Commit Style

```
fix: wire ApiKeyPool into llm_analyzer_base graph path
feat: add multilingual batch scanner with parallel execution
docs: document dual-patch pool wiring fix
```

- Present-tense, imperative mood
- `Signed-off-by` trailer required (NVIDIA DCO)
- `Co-authored-by` trailer for joint work

---

## Key Design Points

Before modifying code, understand these three:

1. **Dual-patch pool wiring.** `set_api_pool()` patches both `llm_utils.get_chat_model` AND `llm_analyzer_base.get_chat_model`. The latter is necessary because `llm_analyzer_base` imports via `from ... import`, creating a local reference that single-module patching misses. See `docs/archive/PITFALLS.md`.

2. **Instance-attribute injection (not class-attribute).** Patch 1 writes `self.response_schema = None` to instance `__dict__`, not class `__dict__`. Python MRO finds instance attributes first. This is what makes patches thread-safe. Mutating the class attribute causes cross-thread races (this killed V1).

3. **Guard before apply.** `_verify_patch_targets()` checks all 7 patch assumptions before `_apply_patches()` runs. If upstream changes a signature or removes a dependency, the guard raises immediately — patches fail closed, never silently.

Full architecture: `docs/DESIGN.md`.
All pitfalls: `docs/archive/PITFALLS.md`.

---

## Where to Contribute

See `docs/archive/FUTURE_WORK.md` for 12 future directions with effort estimates. High-impact items:
- Checkpoint/resume (prevents data loss on large scans)
- Language detection expansion (9+ languages)
- SARIF output format
- Non-English ground-truth fixtures

---

**Next:** [docs/README.md](docs/README.md) — user guide · [docs/DESIGN.md](docs/DESIGN.md) — architecture · [docs/REVIEW_RESPONSE.md](docs/REVIEW_RESPONSE.md) — PR #100 review response
