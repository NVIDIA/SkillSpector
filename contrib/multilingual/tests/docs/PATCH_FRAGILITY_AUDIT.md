# Monkey-Patch Fragility — Deep Audit

> 2026-06-25 | Per-Patch Review: Verified fixes + remaining fragility points

---

## ✅ Fixed

| Risk | Fix |
|------|------|
| Silent global mutation on import | `deepseek_compat()` context manager + `setup_deepseek_compat()` explicit call |
| Nested premature restore | `_patches_active: bool` → `_patches_depth: int` counter |
| Pydantic alias priority (Patch 6) | Set both `kwargs["timeout"]` + `kwargs["request_timeout"]` |
| MRO instance-dict injection (Patch 1) | Python language guarantee, not a library internal detail |
| `except (JSONDecodeError, Exception)` masks error types | Split into separate `except json.JSONDecodeError` (LLM output quality) + `except Exception` (upstream schema change), with logs distinguishing "invalid JSON" vs "schema validation failed" |
| `tearDownClass` infinite loop | `from import _patches_depth` → `import runner as _r; while _r._patches_depth > 0` |
| P1: _check_signature does not check parameter kind | Added `KEYWORD_ONLY` detection — raises RuntimeError when upstream changes to keyword-only |
| P2: _original_chatopenai_init capture timing | Moved to module load time (captured on `runner.py` import), not dependent on `_apply_patches` runtime |
| P4: Patch 4/5 reference-only check | Added 2 functional tests — verify build_prompt output contains JSON instruction |

---

## 🔴 Remaining Fragility Points (1 item)

### #P3 `_verify_patch_targets()` failure path zero coverage (known Q13)

**Location:** `runner.py:_verify_patch_targets()`

**Problem:** 17 signature checks — any single failure should raise `RuntimeError`. But no test verifies that this raise path actually works.

**Breakage scenario:** `_verify_patch_targets` has a bug (e.g., index error, attribute check omission), silently skips all checks, patches are still applied under an incompatible upstream environment.

**Fix:** Construct a fake incompatible upstream environment (or mock `inspect.signature`), verify that the guard raises `RuntimeError`. **High complexity, accepted as a known blind spot.**

---

## 🟡 Edge Risks (3 items)

| # | Risk | Severity |
|---|------|--------|
| P5 | Reference leak after multiple apply/restore cycles | Very low — production environment cycles only once |
| P6 | `_restore_patches()` overwrites independent patches from other modules | Very low — no other module modifies these classes |
| P7 | `import httpx` failure (Patch 6) silently skipped | Already handled — `except ImportError` |

---

## Mutation Coverage Status

| Patch | Mutation | Status |
|-------|------|------|
| 1 (init) | Skip replacement | ✅ Added |
| 2 (parse) | Always return empty | ✅ Added |
| 3 (meta parse) | Skip sanitize | ✅ Added |
| 4 (base prompt) | Do not append JSON instruction | ✅ Added |
| 5 (meta prompt) | Do not append JSON instruction | ✅ Added |
| 6 (timeout) | Do not inject timeout | ✅ Added |
| 7 (asyncio) | Degrade to original run | ✅ Added |

**All 7 Patches have mutation tests.** ✅

---

## Summary

| Category | Count |
|------|------|
| Fixed | 9 |
| Remaining fragility points | 1 (P3: `_verify_patch_targets` failure path zero coverage, known Q13) |
| Edge risks | 3 (P5-P7) |
| Mutation coverage | 7/7 Patch |
