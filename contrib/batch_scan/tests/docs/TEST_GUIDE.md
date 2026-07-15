# Test Guide — contrib/batch_scan

> **WHAT & WHERE.** Coverage map and quick reference. For design rationale
> — why each suite exists and how it was designed — see `TEST_DESIGN.md`.
> For bugs found, see `BUGS_FOUND.md`.

---

## Quick Reference

```bash
# All 164 tests
python contrib/batch_scan/tests/tests-pro/random_numbered.py    # 120 unit (seed=42)
python contrib/batch_scan/tests/test_pool_wiring.py              # 4 smoke checks
python contrib/batch_scan/tests/test_monkeypatch_invasiveness.py # 14 thematic
python contrib/batch_scan/tests/test_monkeypatch_fragility.py    # 26 thematic

# Review-themed only (44 total)
python -m unittest \
  contrib.batch_scan.tests.test_monkeypatch_invasiveness \
  contrib.batch_scan.tests.test_monkeypatch_fragility -v
python contrib/batch_scan/tests/test_pool_wiring.py
```

---

## Directory Structure

```
tests/
├── test_pool_wiring.py                  ← Issue #1 — pool wiring smoke
├── test_monkeypatch_invasiveness.py     ← Issue #2 — thread isolation, scoping
├── test_monkeypatch_fragility.py        ← Issue #2 — guard verification
│
├── docs/
│   ├── TEST_DESIGN.md                   ← why each suite was designed
│   ├── TEST_GUIDE.md                    ← this file — what's covered
│   └── BUGS_FOUND.md                    ← 16 production bugs found
│
└── tests-pro/
    ├── test_api_pool.py                 ← 45 tests — pool acquire/release/backoff
    ├── test_gap_fill.py                 ← 41 tests — JSON parsing, prompt building
    ├── test_runner_patches.py           ← 24 tests — context manager, patches
    ├── test_annotation.py               ← 10 tests — language compatibility
    ├── random_numbered.py               ← main entry point (seed=42)
    ├── mutation_max.py                  ← 30-bug injection framework
    └── __init__.py
```

---

## Review-Themed Test Files — What Each Covers

### `test_pool_wiring.py` — Pool Wiring Smoke (4 checks)

Answers reviewer: *"The API key pool is built but never actually used."*

| Check | What it covers |
|-------|---------------|
| `llm_utils.get_chat_model()` → PooledChatModel | Direct module call path |
| `LLMAnalyzerBase._llm` → PooledChatModel | **Graph path** (20 analyzers per skill, 95% LLM calls) |
| `GapFillAnalyzer.chat_model` → PooledChatModel | Gap-fill path |
| `set_api_pool(None)` restores originals on both modules | Cleanup path |

---

### `test_monkeypatch_invasiveness.py` — Invasiveness (14 tests)

Answers reviewer: *"Import-time global monkey-patching is invasive."*

| Class | Tests | What it covers |
|-------|-------|---------------|
| `TestImportNoSideEffect` | 1 | Subprocess: `import runner` leaves `__init__` untouched |
| `TestThreadIsolation` | 4 | 50 concurrent instances → all `response_schema=None`; class attr intact; Thread B outside context sees original; instance attrs don't cross-contaminate |
| `TestContextManagerScoping` | 4 | All 5 methods replaced inside context; all 5 restored after exit; exception-safe restore; asyncio.run scoped |
| `TestContextManagerNesting` | 2 | Double nesting → inner exit doesn't restore; triple nesting → only outermost restores |
| `TestSetupFunction` | 3 | `setup_deepseek_compat()` applies patches; idempotent on repeat; setup then context → inner exit doesn't restore |

---

### `test_monkeypatch_fragility.py` — Fragility (26 tests)

Answers reviewer: *"Several patches depend on internal details that can break on upstream updates."*

| Class | Tests | What it covers |
|-------|-------|---------------|
| `TestCheckSignature` | 3 | Missing parameter → RuntimeError; parameter becomes keyword-only → RuntimeError; all params present → passes |
| `TestGuardPassesCurrentUpstream` | 4 | Guard passes against current upstream; context enter triggers guard; guard passes after apply+restore cycle; guard passes after setup+restore cycle |
| `TestGuardPatch1Init` | 3 | `base_prompt` missing → caught; `model` missing → caught; `response_schema` class attr removed → caught |
| `TestGuardPatch2ParseResponse` | 4 | `batch` missing → caught; `model_validate` removed → caught; `to_finding` removed → caught; `Batch.file_path` removed → caught |
| `TestGuardPatch3MetaParse` | 3 | `batch` missing → caught; `model_validate` removed → caught; `MetaAnalyzerResult.findings` removed → caught |
| `TestGuardPatch4BaseBuildPrompt` | 2 | `batch` missing → caught; `**kwargs` removed → caught |
| `TestGuardPatch5MetaBuildPrompt` | 1 | `batch` missing → caught |
| `TestGuardPatch7Asyncio` | 2 | `main` parameter present; `asyncio.new_event_loop` removed → caught |
| `TestGuardAtomicity` | 1 | Guard fails → ZERO patches applied |
| `TestOriginalCapturedAtImportTime` | 3 | Base init captured at import; ChatOpenAI init not None; asyncio.run is true stdlib |

---

## Unit Tests (tests-pro/) — What Each Covers

### `test_api_pool.py` — 45 tests, 10 classes

| Class | Tests | Covers |
|-------|-------|--------|
| `TestCreateApiKeyPoolFromEnv` | 3 | Multi-key env → pool; single key → None; no keys → None |
| `TestAcquireRelease` | 6 | `acquire()` least-loaded key; `release()` marks idle; `try_acquire()` fast path; `active_requests` tracking; slots exhausted → None; release after success resets 429 counter |
| `TestEdgeCases` | 4 | Empty key list → ValueError; released slot returns least-loaded; `retry_successes` counter; `keys_configured` / `total_capacity` |
| `TestSnapshot` | 2 | Initial state has all fields; peak/total update after usage |
| `TestRecoveredKeyScheduling` | 2 | Re-acquire after expire; `try_acquire` on recovered |
| `TestRateLimitBackoff` | 6 | Backoff 30s×2ⁿ (cap 300s); consecutive_429 increments; `recover_expired_keys()` restores; release(failure) marks rate-limited; failure marks unavailable; backoff computed from real release failure |
| `TestAcquireTimeout` | 1 | `acquire(timeout)` raises `RuntimeError` when pool full |
| `TestConcurrentAcquireRelease` | 1 | No deadlock; `active_requests` returns to zero |
| `TestResourceLeakRecovery` | 2 | Exception between acquire/release doesn't leak slot; release(failure) doesn't leak |
| `TestIsRateLimit` | 5 | 429 in string message; OpenAI `RateLimitError` type; `rate_limit` keyword; false for `ValueError`; false for ordinary `Exception` |

### `test_gap_fill.py` — 41 tests, 11 classes

| Class | Tests | Covers |
|-------|-------|--------|
| `TestParseResponseValidJSON` | 4 | Single finding; multiple findings; empty findings; default values |
| `TestParseResponseInvalidInput` | 9 | Non-JSON; integer; list; missing `rule_id`; null bytes; BOM prefix; missing `findings` key; illegal severity → defaults |
| `TestParseResponseMarkdownFences` | 4 | Fenced with language tag; no tag; trailing whitespace; unclosed fence |
| `TestParseResponseFiltering` | 5 | Confidence below threshold; unknown rule_id; mixed valid/invalid; all below threshold; all unknown |
| `TestParseResponsePydanticModel` | 1 | Delegate to Pydantic model path |
| `TestParseResponseLargeFindings` | 1 | 100 findings < 1s |
| `TestStripMarkdownFences` | 4 | Language tag; no tag; trailing whitespace; only opening fence |
| `TestBuildPrompt` | 2 | Language tag + file label; numbered content |
| `TestGetBatchesAndCollectFindings` | 2 | One batch per file; collect flattens |
| `TestRunGapFill` | 3 | English skill shortcuts early; empty file cache → `[]`; full flow |
| Other (language injection, conversion, state, entry) | 7 | Language injected into prompt; `to_finding()` preserves 9 fields; `scan_state()` keys; `entry_from_result()` edges |

### `test_runner_patches.py` — 24 tests, 16 classes

| Class | Tests | Covers |
|-------|-------|--------|
| `TestContextManagerApplyRestore` | 8 | All 5 methods replaced; all 5 restored; exception-safe; Patch 1/2/3/4/5 functional verification |
| `TestContextManagerNesting` | 2 | Double/triple nesting |
| `TestSetupFunction` | 2 | `setup_deepseek_compat()` applies; idempotent |
| `TestSetupContextInteraction` | 1 | setup then context → no restore on inner exit |
| `TestImportNoSideEffect` | 1 | Subprocess import isolation |
| `TestVerifyPatchTargets` | 2 | Guard passes; triggers on context enter |
| `TestCheckSignature` | 3 | Missing param; keyword-only; all present |
| `TestPatch2OriginalCapture` | 1 | `_original_chatopenai_init` captured at import |
| `TestPatch6ChatOpenAITimeout` | 1 | Both `timeout` + `request_timeout` set |
| `TestPatch7AsyncioQuietLoop` | 3 | asyncio replaced/restored; suppresses "Event loop is closed"; other exceptions propagate |
| `TestSanitizeMetaFinding` | 4 | null→""; "none"→"low"; invalid→"low"; valid unchanged |
| `TestStripMarkdownFences` | 5 | JSON fence; no tag; plain text; trailing ws; unclosed |
| `TestSetApiPoolRestore` | 1 | `set_api_pool(None)` restores |
| `TestScanState` | 2 | LLM enabled/disabled |
| `TestRelName` | 2 | Relative path; fallback to name |
| `TestEntryFromResult` | 9 | Required keys; default risk; explicit risk; gap_fill mark; skipped rules count; manifest name; directory fallback; different drives |

### `test_annotation.py` — 10 tests, 1 class

| Class | Tests | Covers |
|-------|-------|--------|
| `TestAnnotateFindings` | 10 | `is_language_compatible` for English→English, Chinese→LLM rules, Chinese→code rules, Chinese→English keyword rules; `annotate_findings` empty list, missing rule_id, mixed compatibility, all compatible |

---

## Adding New Tests

1. **Unit tests** → `tests-pro/` + add module to `random_numbered.py`
2. **Reviewer-concern thematic** → top-level `tests/test_<theme>.py`
3. Must pass `random_numbered.py` before committing
4. Use `_force_restore()` in `tearDownClass` if touching monkey-patches
5. Update this file and `TEST_DESIGN.md` when adding significant coverage

---

**Next:** [TEST_DESIGN.md](TEST_DESIGN.md) — why each suite was designed · [Main README](../../docs/README.md) — user guide · [CONTRIBUTING.md](../../CONTRIBUTING.md) — dev setup
