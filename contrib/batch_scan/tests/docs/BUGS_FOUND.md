# Production Code Bugs Found & Fixed

> Covers three phases: 6/23 (API pool refactor) + 6/24-25 (test architecture) + 6/26 (upstream merge + review hardening)
> All discovered by tests or test-driven audits

---

## 🔴 Production Code Bugs (15)

### 6/23 — Discovered During API Pool Refactor

| # | Location | Bug | Symptom | Fix | Discovery Method |
|---|------|-----|------|------|---------|
| B1 | `api_pool.py:snapshot()` | **Deadlock** — `self._lock` is not reentrant. `snapshot()` calls `self.active_requests` property while holding the lock → property internally acquires the same lock again | Process hangs | Read fields directly within the locked region, do not call property | Integration test |
| B2 | `api_pool.py:_capacity_summary()` | **Deadlock** — Same as above. `acquire()` calls `self.total_capacity` property while holding the lock | Same as above | Same as above | Integration test |
| B3 | `api_pool.py:PooledChatModel._ainvoke_with_retry()` | **Async event loop blocking** — `acquire()` synchronously blocks on `Condition.wait()`, asyncio event loop stalls | Concurrent performance degradation | Added `try_acquire()` non-blocking fast path | Integration test |
| B4 | `api_pool.py:record_retry_success()` | **Counting error** — Increments on retry **attempt**, not retry **success** | Report data is misleading | Moved to after `llm.invoke()` succeeds, inside `if attempt > 0` condition | Code review |
| B5 | `api_pool.py:set_api_pool(None)` | **Does not restore original function** — After calling `set_api_pool(None)`, the patched wrapper remains in memory | Subsequent calls still use the old path | Save `_original_get_chat_model`, restore when None | Integration test |
| B6 | `runner.py:Patch 6` | **Pydantic alias dependency** — Only sets `kwargs["timeout"]`, relying on Pydantic v2 alias to cover the canonical name | May break on upstream Pydantic version upgrade | Set both `kwargs["timeout"]` + `kwargs["request_timeout"]` | Audit discovery |
| B7 | `runner.py:cleanup_result()` | **Unreachable code** — `shutil.rmtree(ignore_errors=True)` never raises, subprocess `rm -rf` fallback never executes | Dead code | Removed fallback branch + unused import | Code review |
| B8 | `runner.py:Patch 2/3` | **Overly broad exception handling** — `except (json.JSONDecodeError, Exception)` makes `JSONDecodeError` redundant under `Exception`, and masks the difference between Pydantic validation errors and JSON parse errors | Masks real bugs | Split into separate `except json.JSONDecodeError` (LLM output quality issue) and `except Exception` (upstream schema change), with logs distinguishing "invalid JSON" vs "schema validation failed" | Code review |
| B9 | `batch_scan.py:main()` | **Report delay** — `with ThreadPoolExecutor` calls `shutdown(wait=True)` on exit, waiting for stuck worker threads. Timed-out skipped skills are still running, blocking report output | Report waits 80-100s | Changed to `executor.shutdown(wait=False)`, do not wait for dead threads | Integration test |

### 6/24-25 — Discovered During Test Architecture Audit

| # | Location | Bug | Symptom | Fix | Discovery Method |
|---|------|-----|------|------|---------|
| B10 | `runner.py:_apply_patches()` | **Nested premature restore** — `_patches_active: bool` flag. Inner `__exit__` removes patches that the outer block is still using | Patches silently deactivate | Changed to `_patches_depth: int` nesting counter | Code review + nesting test |
| B11 | `test_runner_patches.py:TestSetupFunction.tearDownClass` | **Infinite loop** — `from runner import _patches_depth` copies the int value. `while _patches_depth > 0:` reads the local copy, which is never 0 | Test process hangs permanently | Changed to `import runner as _r; while _r._patches_depth > 0:` | Random-order test |
| B12 | `test_runner_patches.py:test_setup_applies_patches` | **False assertion** — `assertIsNot(init, LLMAnalyzerBase.__init__ if False else True)` is always True | Test always passes, cannot detect patch failure | Changed to save `orig_init` reference then `assertIsNot(init, orig_init)` | Audit discovery |
| B13 | `runner.py:_check_signature()` | **Does not detect parameter kind** — Only checks parameter name existence, not whether it is keyword-only. If upstream changes to `def __init__(self, *, base_prompt, model)`, the check still passes | Patch may crash on newer Python 3 versions | Added `KEYWORD_ONLY` detection, raises RuntimeError when found | Audit discovery |
| B14 | `runner.py:_original_chatopenai_init` | **Capture timing depends on import order** — Captured when `_apply_patches()` runs. If another module pre-modifies `ChatOpenAI.__init__`, the wrong version is captured | Test environment may be incorrect | Moved to module load time (captured on `import runner.py`) | Audit discovery |
| B15 | `test_runner_patches.py:Patch 4/5` | **Missing functional verification** — Only checks that method references are replaced, does not verify that the replacement actually appends JSON instructions | Patch 4/5 failure is undetectable | Added 2 functional tests: `assertIn("Respond with ONLY a JSON object", prompt)` | Mutation testing |

### 6/26 — Discovered During Upstream Merge + Reviewer Response

| # | Location | Bug | Symptom | Fix | Discovery Method |
|---|------|-----|------|------|---------|
| B16 | `runner.py:set_api_pool()` | **Pool bypass: graph path** — Only patched `llm_utils.get_chat_model`. `llm_analyzer_base` imports via `from ... import`, creating a local reference. Graph analyzers (95% LLM calls) called the unpatched local reference. `snapshot()['rate_limits_hit']` always 0. | Pool appears wired but graph path bypasses it entirely | Added `_llm_analyzer_base.get_chat_model = _pooled_get_chat_model`; `test_pool_wiring.py` now verifies `LLMAnalyzerBase._llm is PooledChatModel` | PR re-review after upstream merge |

---

## 🟡 Test Code Bugs (3)

| # | Location | Bug | Fix |
|---|------|-----|------|
| T1 | `test_api_pool.py:test_exponential_backoff_values` | Tests the math formula `min(30*2^(n-1), 300)`, not the pool's actual `release(success=False)` behavior | Changed to go through the real release path |
| T2 | `test_api_pool.py:_make_key()` | Dead code — defined but never called | Removed |
| T3 | `test_gap_fill.py:_VALID_FINDING` | Module-level mutable dict — shared state risk | Changed to `_valid_finding(**overrides)` factory function |

---

## 📊 Statistics

| Category | Count |
|------|------|
| Production code bugs (fixed) | 16 |
| Test code bugs (fixed) | 3 |
| Known blind spots (accepted) | 4 (Q13, Q16, Q17, Q18) |
| Mutation MISSED (not production bugs) | 9 |
