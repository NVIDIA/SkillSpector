# Line Coverage Analysis — Master Index

> Fifth-round audit: source-level branch trace for all 4 modules.
> Code/test ratio: 0.88:1 (2,532 test lines / 2,892 production lines).

---

## Files & Findings

| # | File | Lines Analyzed | New Findings |
|---|------|---------------|--------------|
| 1 | [LINE_COVERAGE_ACQUIRE.md](LINE_COVERAGE_ACQUIRE.md) | acquire() 73 lines | Q16, Q17, Q18 |
| 2 | [LINE_COVERAGE_RELEASE_TRY.md](LINE_COVERAGE_RELEASE_TRY.md) | try_acquire() 20 lines, release() 40 lines | Q22, Q23, Q24, Q25 |
| 3 | [LINE_COVERAGE_GAPFILL.md](LINE_COVERAGE_GAPFILL.md) | parse_response() 52 lines | Q26, Q27 |
| 4 | [LINE_COVERAGE_PATCHES.md](LINE_COVERAGE_PATCHES.md) | _apply_patches, _restore_patches, deepseek_compat, _check_signature 76 lines | Q28, Q29 |

---

## All 29 Findings (Rounds 1-5)

| # | Sev | Where | What |
|---|-----|-------|------|
| Q1 | 🔴 | test_api_pool | 429 test uses guard, not real flow |
| Q2 | 🔴 | test_api_pool | Backoff test same guard dependency |
| Q3 | 🔴 | test_api_pool | isinstance path for 429 detection uncovered |
| Q4 | 🔴 | test_runner | Patch 7 handler never triggered in test |
| Q5 | 🔴 | test_runner | Patch 7 "other exceptions" tests Python, not patch |
| Q10 | 🔴 | test_runner | Test order fragility — global state leak |
| Q16 | 🔴 | api_pool | acquire() wait-for-recovery branch zero coverage |
| Q22 | 🔴 | api_pool | try_acquire recovery path untested (parallel to #C1) |
| Q23 | 🔴 | api_pool | Backoff formula n=3,4,5 never exercised |
| Q26 | 🔴 | gap_fill | Fence with no newline — uncovered branch |
| Q27 | 🔴 | gap_fill | Fence with no closing ``` — uncovered branch |
| Q28 | 🔴 | runner | Patch 6 ImportError skip path zero coverage |
| Q29 | 🔴 | runner | _check_signature except path zero coverage |
| Q6 | 🟡 | test_api_pool | Unused import |
| Q7 | 🟡 | test_gap_fill | BOM test too weak |
| Q8 | 🟡 | test_runner | Patch 6 test mutates global ChatOpenAI |
| Q12 | 🟡 | test_api_pool | Consecutive 429 test same guard as Q1 |
| Q13 | 🟡 | test_runner | Guard test doesn't assert guard ran |
| Q17 | 🟡 | api_pool | _next_available_in() zero direct coverage |
| Q18 | 🟡 | api_pool | _capacity_summary() zero direct coverage |
| Q19 | 🟡 | test_api_pool | Can't distinguish success vs failure decrement |
| Q24 | 🟡 | test_api_pool | rate_limits_hit counter never directly asserted |
| Q25 | ✅ | api_pool | notify_all behavior implicit, removable — accepted limitation |
| Q9 | 🟢 | test_gap_fill | Misleading docstring (Pydantic model) |
| Q11 | 🟢 | test_gap_fill | Misleading test name (English shortcut) |
| Q14 | 🟢 | test_annotation | Default behavior undocumented |
| Q15 | 🟢 | test_annotation | OR-blindness: rule misclassification |
| Q20 | 🟢 | test_pool_wiring | test_pool_wiring.py outside tests-pro/ |
| Q21 | 🟢 | test_gap_fill | setUpClass shared state undocumented |

**13 genuine issues. 8 design weaknesses (Q25 accepted). 7 cosmetic. 28 active.**
