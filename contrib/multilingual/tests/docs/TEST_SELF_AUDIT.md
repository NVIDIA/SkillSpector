# Test Self-Audit — Complete Issues Register (Master)

> 2026-06-24 | Four rounds of self-audit + DeepSeek architecture review + Round 6 fine-tuning | 46 items discovered
> Round 5: Concurrency race conditions/exception safety/resource leaks | Round 6: C8/C9 implementation method correction + statistics correction
> Round 1-2: Code review | Round 3: Per-function cross-reference | Round 4: FIRST+AAA standards compliance
>
> ⚠️ **This document is a historical audit record.** Most Critical/Medium items were fixed before 2026-06-25.
> For current status see `BUGS_FOUND.md` (fixed bugs) and `TEST_QUALITY_AUDIT.md` (final quality audit).
>
> Companion file: `TEST_DESIGN.md` (test design document)

---

## 🔴 Critical

### #1 `test_setup_applies_patches` — Assertion Always Passes

**File:** `test_runner_patches.py:98-101`

```python
# Current — assertion is always True, regardless of whether patch was actually applied
self.assertIsNot(LLMAnalyzerBase.__init__,
    LLMAnalyzerBase.__init__ if False else True)  # ← always True
```

**Fix:** Save `orig_init` reference → call setup → assert reference changed + functional effect (response_schema=None)

### #2 No Test for `_verify_patch_targets()` — 17-Point Guard Has Zero Coverage

**File:** `runner.py` `_verify_patch_targets()` — no corresponding test

This function runs automatically on every `deepseek_compat()` entry, verifying 17 upstream API dependency points. If it silently breaks (e.g., signature check fails after upstream update), patches may silently deactivate.

**Fix:** Add test — verify guard passes under current upstream version inside context manager; construct fake incompatible scenario to verify guard raises RuntimeError.

---

## 🟡 Medium — Tests Wrong Behavior

### #3 `test_exponential_backoff_values` — Tests Math, Not Pool

**File:** `test_api_pool.py:79-84`

```python
# Current — directly computes formula, never calls pool.release(key, success=False)
self.assertEqual(min(30.0 * (2 ** 0), 300.0), 30.0)
```

**Fix:** Trigger real backoff via `release(success=False)`, check `rate_limited_until` timestamp

### #4 `_make_key()` — Dead Code

**File:** `test_api_pool.py:14-18`

Defined but never called. **Fix:** Remove

### #5 `_VALID_FINDING` — Mutable Module-Level Shared Dict

**File:** `test_gap_fill.py:21-28`

All tests share the same dict reference. If any test accidentally modifies it, other tests are affected.

**Fix:** Change to `_valid_finding(**overrides)` factory function

### #6 Patch 6 & 7 — Zero Direct Test Coverage

**File:** `runner.py` Patch 6 (ChatOpenAI timeout), Patch 7 (asyncio.run quiet loop)

These are the two patches Max explicitly marked as "high risk" — depending on Pydantic alias priority and CPython internal error messages. Currently 0 direct tests.

**Fix:** Patch 6 — verify `ChatOpenAI.__init__` is called with both `kwargs["timeout"]` and `kwargs["request_timeout"]` set. Patch 7 — verify `asyncio.run` is replaced inside context manager, event loop exception handler correctly installed.

### #7 `acquire(timeout=...)` — Timeout Path Untested

**File:** `api_pool.py` `ApiKeyPool.acquire(timeout=...)`

`acquire()`'s `timeout` parameter is never used in tests. The timeout-raises-`RuntimeError` logic has zero coverage.

**Fix:** Use 1-key 1-slot pool — fill the only slot → `acquire(timeout=0.1)` → assert raises `RuntimeError`

---

## 🟢 Minor — Coverage Gaps

### #9 `test_release_success_resets_consecutive_429` — Bypasses Real Flow

**File:** `test_api_pool.py:59`

Manually sets `key.consecutive_429 = 3` — skips the real `release(success=False)` accumulation path.

**Fix:** Three `release(key, success=False)` → assert count=3 → `release(key, success=True)` → assert count=0

### #10 `test_consecutive_429_increments` — Only Tests n=1

**File:** `test_api_pool.py:73-77`

Single 429. Does not verify that two consecutive failures push the counter to 2.

**Fix:** Two `release(success=False)` → assert count=2

### #13 `test_patches_restored_after_context` — Reference Check Only, No Functional Verification

**File:** `test_runner_patches.py:26-41`

Only verifies method references return to original. Does not verify that class **behavior** is also restored after exiting context.

**Fix:** After exiting context, create `LLMAnalyzerBase` instance, assert `response_schema` is not None

### #14 `test_patches_applied_inside_context` — Only 2/5 Methods Checked

**File:** `test_runner_patches.py:18-24`

Only checks `__init__` and `parse_response` are replaced. Does not check `build_prompt` and `LLMMetaAnalyzer` methods.

**Fix:** Save original references for all 5 methods and assert all are replaced

### #19 Subprocess Test Takes ~10s

**File:** `test_runner_patches.py:112-138`

Subprocess verification is the only reliable import isolation method. Cost: 44/45 tests < 2s, this one ~10s.

**Disposition:** Accept. Document honestly, do not modify code.

### #20 test_gap_fill setUp Creates Unnecessary ChatOpenAI Instances

**File:** `test_gap_fill.py:32-33`

`GapFillAnalyzer(language="zh")` calls `LLMAnalyzerBase.__init__` → `get_chat_model()` → creates `ChatOpenAI`. `parse_response` does not need LLM. 22 tests = 22 discarded ChatOpenAIs.

**Disposition:** Accept. Constructor behavior is upstream design. ~50ms each, total < 2s — acceptable.

### #21 Pool Wiring Test Doesn't Make Real LLM Call

**File:** `test_pool_wiring.py`

Only verifies type — `get_chat_model()` returns `PooledChatModel`. Does not verify actual LLM call through the pool (requires real API key).

**Disposition:** Accept. Real LLM calls belong to integration testing, not suitable for unit test suite.

---

## 🟡 Medium — Third Pass: Untested Functions (Zero Coverage)

The following discovered via per-function cross-reference — each callable object had zero direct tests at time of audit. **All have since been fixed.** See `BUGS_FOUND.md` for resolution details.

| # | Function | Current Status |
|---|----------|---------------|
| #22 | `create_api_key_pool_from_env()` | ✅ Tested (TestCreateApiKeyPoolFromEnv, 3 tests) |
| #23 | `_is_rate_limit()` | ✅ Tested (TestIsRateLimit, 5 tests) |
| #24 | `set_api_pool(None)` restore | ✅ Tested (TestSetApiPoolRestore) |
| #25 | `_sanitize_meta_finding()` | ✅ Tested (TestSanitizeMetaFinding, 3 tests) |
| #26 | `_strip_markdown_fences()` | ✅ Tested (TestStripMarkdownFences, 4 tests) |
| #27 | `annotate_findings()` / `is_language_compatible()` | ✅ Tested (TestAnnotateFindings, 10 tests) |
| #28 | `GapFillAnalyzer.build_prompt()` | ✅ Tested (TestBuildPrompt, 2 tests) |
| #29 | `GapFillAnalyzer.get_batches()` + `collect_findings()` | ✅ Tested (TestGetBatchesAndCollectFindings, 2 tests) |

---

## 🔴 Critical — Round 5: DeepSeek Architecture Review

### #C7 Multi-Threaded Race Condition — ✅ Fixed
Added `TestConcurrentAcquireRelease` — 10 threads via `threading.Barrier(10)` simultaneously contend for 1 key, 1 slot. Verifies zero deadlock, zero lost wakeups, `active_requests == 0` after completion.

### #C8 Patch 7 Behavioral Verification — ✅ Fixed
Added `TestPatch7AsyncioQuietLoop` — verifies that replaced `asyncio.run` correctly silences "Event loop is closed" and passes through other exceptions.

### #C9 Resource Leak Recovery — ✅ Fixed
Added `TestResourceLeakRecovery` — verifies that exceptions between acquire/release do not permanently leak slots, and pool can recover.

---

## Statistics (Historical — as of 2026-06-24 audit)

| Severity | Count at Audit Time | Current Status |
|--------|----------|---------|
| 🔴 Critical | 5 | ✅ All fixed (#1-#5, see BUGS_FOUND.md) |
| 🟡 Medium | 19 | ✅ Mostly fixed, remainder are known blind spots/edge risks |
| 🟢 Minor | 19 | ✅ Mostly fixed |
| 🔵 Info | 5 | ✅ Accepted |

---

## Actual Test Count After Fixes (2026-06-25)

| File | At Audit Time | Actually Achieved |
|------|--------|---------|
| test_api_pool.py | 12 | **45** |
| test_gap_fill.py | 22 | **35** |
| test_runner_patches.py | 10 | **48** |
| test_pool_wiring.py | 1 | 1 |
| test_annotation.py | 0 | **10** |
| **Total** | **45** | **120** |
