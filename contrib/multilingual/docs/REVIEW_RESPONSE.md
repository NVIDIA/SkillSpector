# Response to PR #100 Review

> This document tracks how each issue raised in the PR #100 review was addressed.
> See `DESIGN.md` and `archive/FUTURE_WORK.md` for architecture details and roadmap.

---

## Issue 1 — API Key Pool Was Dead Code

**Review feedback:** `ApiKeyPool` was implemented but never wired into actual LLM
call paths. The pool existed on disk but no code path used it.

**Resolution:** `set_api_pool()` now replaces the global `get_chat_model` factory
with a pooled version. Every LLM call — both graph-internal analyzers (SSD, SDI,
SQP, meta, 20 per skill) and the gap-fill pass — draws from the shared key pool.

| Before | After |
|--------|-------|
| Pool instantiated but unused | `set_api_pool(pool)` injects at module level |
| gap-fill used single-key path | gap-fill + all analyzers share the pool |
| No key failover for graph-internal calls | 429 → automatic failover for every LLM call |

See: `api_pool.py` (`set_api_pool`, `PooledChatModel`), `runner.py` (pool integration)

---

## Issue 2 — Import-Time Monkey-Patches Were Invasive

**Review feedback:** Seven monkey-patches fired at module import (`runner.py`),
mutating upstream class attributes before any thread started. This was fragile
(import order dependent) and invasive (no opt-out).

**Resolution:** Replaced import-time auto-patching with explicit `setup_deepseek_compat()`
and a context manager that tracks nesting depth.

| Before | After |
|--------|-------|
| `import runner` → patches fire immediately | Call `setup_deepseek_compat()` explicitly |
| No way to skip patches | Don't call it → patches never apply |
| Class-attribute mutation (race risk) | Instance-attribute injection (thread-safe) |
| No nesting guard | Depth counter — only outermost exit restores originals |
| 7 separate `_patch_*` / `_restore_*` functions | Single context manager, apply-all / restore-all |

Additional hardening:
- **`_verify_patch_targets` guard** — verifies upstream signatures at context-enter
  time. If upstream changes a patched method's signature, the guard raises
  immediately with a clear error rather than silently breaking at runtime.
- **`test_pool_wiring.py`** — smoke test verifying `PooledChatModel` routes
  through every LLM call path.

See: `runner.py` (`setup_deepseek_compat`, `_verify_patch_targets`),
`CONTRIBUTING.md` (patch architecture)

---

## Issue 3 — Risky Code Lacked Tests

**Review feedback:** The four riskiest areas — pool acquire/release, 429 backoff,
monkey-patches, and gap-fill parsing — had zero automated tests.

**Resolution:** 120 unit tests across 4 modules, plus mutation testing.

| Module | Tests | Covers |
|--------|-------|--------|
| `test_api_pool.py` | 45 | acquire/release, rate-limit backoff, concurrency, edge cases, `try_acquire` |
| `test_gap_fill.py` | 41 | `parse_response` JSON recovery, markdown fence stripping, prompt building, batch/collect |
| `test_runner_patches.py` | 24 | `setup_deepseek_compat()`, context manager nesting, isolation, `_verify_patch_targets` |
| `test_annotation.py` | 10 | `is_language_compatible`, `annotate_findings` edge cases |

**Mutation testing:** 30 bugs injected across the 4 risk areas. Tests catch 21/30.
The 9 misses are documented in `archive/FUTURE_WORK.md` §5.

See: `tests/` directory

---

## Minor Issues

### M1 — `_strip_markdown_fences` duplicated in `runner.py` and `gap_fill.py`

Acknowledged. Listed in `archive/FUTURE_WORK.md` as a low-priority cleanup. The
duplication is deliberate for now — `gap_fill.py` is designed to work standalone
without importing `runner.py` and its side effects.

### M2 — `graph.invoke` call count mismatch in docstring

Fixed. Docstrings and comments updated to reflect the actual graph topology.

---

## Additional Improvements Beyond Review Scope

### Performance
- **7 failed optimization attempts evaluated and reverted.** Async pooling, global
  semaphore, slot-count-based scheduling, and 4 other approaches were tested
  and rejected. The current implementation represents the most stable
  configuration. Details in internal record `PERFORMANCE_OPT_FAILURES.md`.
- **99s baseline for 23-skill LLM scan** with 10 keys / 8 workers.

### Robustness
- `cleanup_result` subprocess fallback for stale file descriptors.
- `httpx.Timeout(connect=8s, read=30s)` prevents hung worker threads.
- `asyncio.run` exception handler suppresses harmless cleanup noise.
- Per-skill 90s timeout with skip-and-continue semantics.

### Documentation
- `DESIGN.md` — architecture, concurrency model, patch rationale, rejected alternatives.
- `CONTRIBUTING.md` — code map, design decisions, contribution guide.
- `archive/ARCHITECTURE_DEEP_DIVE.md` — statelessness proof, three-layer parallelism, bug history.
- `archive/FLOW_DIAGRAM.md` — visual pipeline diagrams.
- `archive/FUTURE_WORK.md` — 12-item roadmap with status and suggested directions.

---

## Summary

| Issue | Status |
|-------|--------|
| #1 — Pool dead code | ✅ Wired into all LLM paths via `set_api_pool()` |
| #2 — Invasive patches | ✅ Replaced with explicit `setup_deepseek_compat()` + context manager |
| #3 — No tests | ✅ 120 unit tests + 30-mutation suite |
| M1 — Duplicated utility | Known, deferred to cleanup |
| M2 — Docstring mismatch | Fixed |
