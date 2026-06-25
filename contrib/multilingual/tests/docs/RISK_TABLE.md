# Concurrency-Heavy & Failure-Prone Code — Full Inventory

> Max's words: *"the concurrency-heavy / failure-prone pieces"*
> Per-function enumeration, annotated with mutation test coverage status

---

## ApiKeyPool — Concurrent Pool Scheduler

| Function | Lines | Risk Type | Why Dangerous | Mutation | Test |
|------|----|---------|-----------|------|------|
| `acquire()` | 165-238 | 🔴 Concurrency | `threading.Condition.wait()` blocking, `while True` potential infinite loop, least-load `min()` logic, peak tracking, timeout branch | 1a (increment), 1c (load balance) | TestAcquireRelease, TestConcurrentAcquireRelease |
| `try_acquire()` | 239-259 | 🔴 Concurrency | `threading.Lock` non-blocking acquisition, `_recover_expired_keys` call, peak tracking | 1d (recovery broken) | TestAcquireRelease |
| `release()` | 260-301 | 🔴 Concurrency + 🔴 Fault tolerance | `notify_all()` wakes waiting threads, `success=True/False` two paths, backoff formula calculation, `max(0,active-1)` guard | 1b (decrement), 2a (backoff) | TestAcquireRelease, TestRateLimitBackoff, TestResourceLeakRecovery |
| `_recover_expired_keys()` | 358-367 | 🟡 Fault tolerance | State change — rate-limited→available. Depended on by `acquire()` and `try_acquire()` | 2b (never recovers) | TestRateLimitBackoff |
| `_next_available_in()` | 368-375 | 🟡 Fault tolerance | Computes earliest recovery time, affects blocking decision in `acquire()` | 5a (always None) — blind spot Q16 | ⚠ Indirect coverage |
| `snapshot()` | 339-357 | 🟡 Fault tolerance | Previously had deadlock bug (`self._lock` not reentrant). Multiple counter aggregations | ✅ tested | TestSnapshot |
| `record_retry_success()` | 302-309 | 🟢 Simple | Counter increment — only increments on retry success (attempt>0 and call succeeded) | ❌ Low value | TestEdgeCases |
| `_capacity_summary()` | 376-384 | 🟢 Simple | String formatting | ❌ Low value | ⚠ Indirect coverage via Timeout error message |
| `PooledChatModel._invoke_with_retry()` | 443-474 | 🔴 Fault tolerance | Synchronous retry loop, 429 detection, key switching, max 5 retries | ❌ Needs mock LLM | ⚠ Integration test coverage |
| `PooledChatModel._ainvoke_with_retry()` | 475-529 | 🔴 Fault tolerance | Async retry, `try_acquire()` fast path + `acquire()` blocking fallback | ❌ Needs mock LLM | ⚠ Integration test coverage |
| `PooledChatModel._is_rate_limit()` | 530-551 | 🟡 Fault tolerance | Dual-path detection — `isinstance(openai.RateLimitError)` + string matching | 6e (always False) | TestIsRateLimit (5 tests) |
| `create_api_key_pool_from_env()` | 552-619 | 🟡 Fault tolerance | Environment variable parsing, multi-key format, single-key fallback | 6f (always None) | TestCreateApiKeyPoolFromEnv (3 tests) |

---

## Runner — Monkey-Patch System

| Function | Lines | Risk Type | Why Dangerous | Mutation | Test |
|------|----|---------|-----------|------|------|
| `_apply_patches()` | 474-507 | 🔴 Global state | Replaces 5 class methods + `asyncio.run`. `_patches_depth` counter. ImportError path zero coverage | 3a (Patch 1 skipped) | TestContextManagerApplyRestore |
| `_restore_patches()` | 508-550 | 🔴 Global state | Nested exit logic — depth counter decrement. Restores 7 patches. | 5b (skips Patch 6+7) | TestContextManagerNesting, TestContextManagerApplyRestore |
| `_verify_patch_targets()` | 300-439 | 🟡 Fault tolerance | **17 signature verifications** — any single failure should raise RuntimeError. Raise path zero coverage | 5c (no-op) — blind spot Q13 | TestVerifyPatchTargets |
| `_patched_base_init` (Patch 1) | 120-134 | 🟡 Fault tolerance | MRO instance-dict injection — sets `response_schema=None` before `__init__` | 3a | TestContextManagerApplyRestore |
| `_patched_base_parse` (Patch 2) | 135-174 | 🟡 Fault tolerance | Manual JSON parsing — `json.loads` → `LLMAnalysisResult.model_validate`. Two levels of except handled independently | 3c (always empty) | TestContextManagerApplyRestore |
| `_patched_meta_parse` (Patch 3) | 175-218 | 🟡 Fault tolerance | Same as above + `_sanitize_meta_finding` cleans null/"none" | 3e (sanitize broken) | TestSanitizeMetaFinding |
| `_patched_base_build_prompt` (Patch 4) | 219-241 | 🟢 Simple | String append JSON instruction | 3f (prompt missing) | TestContextManagerApplyRestore ✅ Functional test |
| `_patched_meta_build_prompt` (Patch 5) | 242-256 | 🟢 Simple | Same as above | 3g (meta prompt missing) | TestContextManagerApplyRestore ✅ Functional test |
| `_patched_chatopenai_init` (Patch 6) | 257-276 | 🔴 Fault tolerance | **Pydantic alias priority** — sets both `timeout` + `request_timeout` | 3b (no timeout) | TestPatch6ChatOpenAITimeout |
| `_patched_asyncio_run` (Patch 7) | 277-299 | 🔴 Global state | Replaces `asyncio.run` — creates quiet event loop. Handler only silences "Event loop is closed" | 3d (not patched) | TestPatch7AsyncioQuietLoop |
| `deepseek_compat()` | 551-590 | 🟡 Fault tolerance | Context manager — `finally` guarantees restoration. Nesting-safe (depth counter) | 6g (no restore on exc) | TestContextManagerNesting, TestContextManagerApplyRestore |
| `set_api_pool()` | 58-112 | 🟡 Global state | Monkey-patch `get_chat_model`. `set_api_pool(None)` restore logic | 5e (broken fallback) | TestSetApiPoolRestore |
| `_check_signature()` | 440-473 | 🟡 Fault tolerance | `inspect.signature` may raise exceptions for certain objects. Raise path zero coverage | 5d (no-op) + direct test | TestCheckSignature (3 tests: pass, missing, keyword-only) |

---

## GapFill — LLM Parser

| Function | Lines | Risk Type | Why Dangerous | Mutation | Test |
|------|----|---------|-----------|------|------|
| `parse_response()` | 206-257 | 🔴 Fault tolerance | **4 layers of exception protection**: JSON parse → Pydantic validation → confidence filter → rule_id filter | 4a-4e (5 mutations) | TestParseResponse* (35 tests) |
| `build_prompt()` | 195-202 | 🟢 Simple | String template injection | 6a (missing content) | TestBuildPrompt (2 tests) |
| `get_batches()` | (inherited from LLMAnalyzerBase) | 🟢 Simple | Token budget calculation, file chunking | 6b (always empty) | TestGetBatchesAndCollectFindings |
| `collect_findings()` | (inherited from LLMAnalyzerBase) | 🟢 Simple | List flattening | 6c (always empty) | TestGetBatchesAndCollectFindings |
| `run_gap_fill()` | 265-305 | 🟡 Fault tolerance | Full pipeline call — create analyzer → get_batches → run_batches → collect_findings. Exceptions swallowed by try/except | 6d (always empty) | TestRunGapFill |

---

## Annotation — Rule Classification

| Function | Lines | Risk Type | Why Dangerous | Mutation | Test |
|------|----|---------|-----------|------|------|
| `annotate_findings()` | 86-100 | 🟢 Simple | Reads `issue["id"]` — field name convention | 5f (always incompatible) | TestAnnotateFindings (10 tests) |
| `is_language_compatible()` | 73-83 | 🟢 Simple | OR logic — union of three rule sets | 5g (always True) | TestAnnotateFindings |

---

## Coverage Summary

| Risk Level | Total Functions | With Mutation | Without Mutation (Reason) |
|----------|--------|--------|-------------|
| 🔴 High risk | 12 | 23 mutations covering 11 | 1 needs mock LLM |
| 🟡 Medium risk | 13 | 13 mutations covering 13 | 0 |
| 🟢 Low risk | 7 | 4 mutations covering 4 | 3 low value (counter/formatting/annotation) |
| **Total** | **32** | **40 mutations covering 28 functions** | **4 without mutation (1 mock, 3 low value)** |
