# Test Directory Guide

> Overview of every file under `tests/` — what it tests, how to run it,
> and whether it belongs in the PR or the internal archive.

---

## Directory Structure

```
tests/
├── test_pool_wiring.py              ← smoke test: pool wiring verification
├── TEST_DESIGN.md                   ← test suite architecture design
│
├── docs/                            ← test guidance documents
│   ├── BUGS_FOUND.md                ← production code bugs found during testing
│   ├── LINE_COVERAGE_ACQUIRE.md     ← line coverage: acquire()
│   ├── LINE_COVERAGE_GAPFILL.md     ← line coverage: gap_fill
│   ├── LINE_COVERAGE_INDEX.md       ← line coverage master index
│   ├── LINE_COVERAGE_PATCHES.md     ← line coverage: runner patches
│   ├── LINE_COVERAGE_RELEASE_TRY.md ← line coverage: try_acquire() + release()
│   ├── MUTATION_PLAN.md             ← mutation test design
│   ├── PATCH_FRAGILITY_AUDIT.md     ← patch fragility audit
│   ├── RISK_TABLE.md                ← concurrency risk checklist
│   ├── TEST_QUALITY_AUDIT.md        ← test quality master audit
│   └── TEST_SELF_AUDIT.md           ← self-audit registry
│
└── tests-pro/                       ← formal test code
    ├── test_api_pool.py             ← 10 classes, 45 tests — pool core logic
    ├── test_gap_fill.py             ← 11 classes, 41 tests — gap-fill parsing
    ├── test_runner_patches.py       ← 16 classes, 24 tests — patch context managers
    ├── test_annotation.py           ← 10 tests — annotation module
    ├── mutation_max.py              ← 30 mutation injection framework
    ├── random_numbered.py           ← main test entry point (120 tests, seed=42)
    └── __init__.py                  ← package marker
```

### Already Moved (archived in `contrib/lib/`)

| Moved File | Reason |
|-----------|------|
| `tests/test_api_pool.py` | early slim version (4 classes), fully superseded by tests-pro equivalent (10 classes) |
| `tests/test_gap_fill.py` | early slim version (6 classes), fully superseded by tests-pro equivalent (11 classes) |
| `tests/test_runner_patches.py` | early slim version (4 classes), fully superseded by tests-pro equivalent (16 classes) |
| `tests/TEST_FIRST_AAA_CHECKLIST.md` | internal AAA audit checklist, not a deliverable |
| `tests/TEST_REPORT.txt` | legacy test output snapshot |
| `tests-pro/mutation_test.py` | small variant, mutation_max covers it |
| `tests-pro/random_only.py` | random-only variant, random_numbered covers it |
| `tests-pro/run_random_bench.py` | one-off benchmark tool |
| `tests-pro/show_order.py` | one-off tool |
| `tests-pro/find_slow.py` | one-off tool |
| `tests-pro/debug_*.py` (7 files) | hang debugging scripts |
| `tests-pro/isolate_*.py` (2 files) | network isolation debugging scripts |
| `tests-pro/DIAGNOSIS_HANG.md` | random-order hang diagnosis |

---

## PR Test Files

### `tests-pro/test_api_pool.py` — 45 tests (10 classes)

| Class | Tests | Covers |
|-------|-------|--------|
| `TestCreateApiKeyPoolFromEnv` | 3 | Pool creation from env vars, single key, no keys |
| `TestAcquireRelease` | 6 | `acquire()`, `release()`, `try_acquire()`, `active_requests` tracking |
| `TestEdgeCases` | 4 | Empty key list, least-loaded scheduling, retry counter, capacity properties |
| `TestSnapshot` | 2 | Snapshot before/after usage |
| `TestRecoveredKeyScheduling` | 2 | Recovered key re-acquisition |
| `TestRateLimitBackoff` | 6 | Backoff `30s × 2^n`, recovery, consecutive 429 tracking |
| `TestAcquireTimeout` | 1 | Timeout raises `RuntimeError` when pool full |
| `TestConcurrentAcquireRelease` | 1 | No deadlock, `active_requests` returns to zero |
| `TestResourceLeakRecovery` | 2 | Exception between acquire/release does not leak slot |
| `TestIsRateLimit` | 5 | Detects 429 in string message, OpenAI error type, keyword match |
| `TestSetApiPoolRestore` | 1 | `set_api_pool(None)` restores original factory |
| Other | 12 | Retry success counter, backoff timestamp, key properties |

### `tests-pro/test_gap_fill.py` — 41 tests (11 classes)

| Class | Tests | Covers |
|-------|-------|--------|
| `TestParseResponseValidJSON` | 4 | Valid single/multiple/empty findings, default values |
| `TestParseResponseInvalidInput` | 9 | Non-JSON, integer, list, missing keys, null bytes, BOM, invalid severity |
| `TestParseResponseMarkdownFences` | 4 | Fenced JSON with/without language tag, jsonp suffix |
| `TestParseResponseFiltering` | 5 | Confidence threshold, unknown rule IDs, mixed valid/invalid |
| `TestParseResponseLargeFindings` | 1 | 100 findings parsed within 1 second |
| `TestParseResponsePydanticModel` | 1 | Pydantic model path delegation |
| `TestStripMarkdownFences` | 4 | Language tag, no tag, trailing whitespace, no closing fence |
| `TestBuildPrompt` | 2 | Language tag + file label, numbered content |
| `TestGetBatchesAndCollectFindings` | 2 | One batch per file, flattening |
| `TestRunGapFill` | 3 | English shortcut, empty file cache, full flow |
| Other | 6 | Language injection, finding conversion, scan state, entry construction |

### `tests-pro/test_runner_patches.py` — 24 tests (16 classes)

| Class | Tests | Covers |
|-------|-------|--------|
| `TestSetupFunction` | 2 | `setup_deepseek_compat()` applies patches, idempotent on double call |
| `TestSetupContextInteraction` | 1 | Context manager after setup does not restore on exit |
| `TestImportNoSideEffect` | 1 | Importing `runner` does NOT apply patches |
| `TestContextManagerApplyRestore` | 12 | All 7 patches applied/restored, exception safety, functional verification |
| `TestContextManagerNesting` | 2 | Double/triple nested context — only outermost exit restores |
| `TestVerifyPatchTargets` | 2 | Guard passes current upstream, triggers on context enter |
| `TestCheckSignature` | 2 | Raises on missing/renamed parameter |
| `TestPatch2OriginalCapture` | 1 | Original `ChatOpenAI.__init__` captured at import time |
| `TestPatch6ChatOpenAITimeout` | 1 | Timeout injection via Pydantic alias |

### `test_pool_wiring.py` — smoke test

Verifies `PooledChatModel` is wired into all LLM call paths. Single test that confirms the pool is actually used, not just instantiated.

---

## Test Guidance Documents (`tests/docs/`, 11 files)

These `.md` files document the design, audit, and quality assessment of the test system, so reviewers can understand the breadth and depth of test coverage.

| File | Content |
|------|------|
| `BUGS_FOUND.md` | production code bugs found during testing, mapped to the test that catches each one |
| `LINE_COVERAGE_ACQUIRE.md` | line coverage: every branch of `ApiKeyPool.acquire()` |
| `LINE_COVERAGE_GAPFILL.md` | line coverage: every branch of `GapFillAnalyzer.parse_response()` |
| `LINE_COVERAGE_INDEX.md` | line coverage master index — summary of 29 findings across 5 audit rounds |
| `LINE_COVERAGE_PATCHES.md` | line coverage: `_apply_patches` / `_restore_patches` / `deepseek_compat` |
| `LINE_COVERAGE_RELEASE_TRY.md` | line coverage: every branch of `try_acquire()` + `release()` |
| `MUTATION_PLAN.md` | 30 mutation injection design — which bugs are injected into 4 risk zones, and which tests are expected to catch them |
| `PATCH_FRAGILITY_AUDIT.md` | risk assessment for each of 7 monkey-patches — which is the most fragile, what upstream details it depends on |
| `RISK_TABLE.md` | concurrency danger zones + high-risk code checklist — must read before modifying these modules |
| `TEST_QUALITY_AUDIT.md` | final quality audit of the test suite — coverage gaps, weak points, improvement directions |
| `TEST_SELF_AUDIT.md` | self-audit registry — what each audit round found and fixed |

---

## Quick Reference

```bash
# Smoke test — verify pool is wired (PR #100 Issue 1)
python contrib/multilingual/tests/test_pool_wiring.py

# Unit tests — random order (seed=42, 120 tests)
cd contrib/multilingual/tests/tests-pro && python random_numbered.py

# Unit tests — sequential pytest
pytest contrib/multilingual/tests/tests-pro/ -v

# Mutation test — 30 injected bugs
python contrib/multilingual/tests/tests-pro/mutation_max.py

# Batch scan (end-to-end)
python -m contrib.multilingual.batch_scan ./tests/fixtures/ -f terminal --workers 8
```
