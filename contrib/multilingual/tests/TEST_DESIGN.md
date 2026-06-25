# Test Design Document — contrib/multilingual

> Following FIRST principles & AAA pattern | 2026-06-25
> Corresponding to PR #100 Issue 3 — high-risk code lacks tests

---

## 1. Test Strategy Overview

| Layer | File | Test Count | Coverage Target |
|------|------|--------|---------|
| Unit | `tests/tests-pro/test_api_pool.py` | 27 | `ApiKeyPool` acquire/release/backoff/recovery |
| Unit | `tests/tests-pro/test_gap_fill.py` | 35 | `GapFillAnalyzer.parse_response` JSON parsing |
| Unit | `tests/tests-pro/test_runner_patches.py` | 48 | `setup_deepseek_compat()` context manager |
| Unit | `tests/tests-pro/test_annotation.py` | 10 | `is_language_compatible` / `annotate_findings` |
| Integration | `tests/test_pool_wiring.py` | 1 | End-to-end pool wiring verification |

**Total: 121 tests (120 unit + 1 smoke), all passing.**
**Random order seed=42, uniformly driven by `tests/tests-pro/random_numbered.py`.**

---

## 2. Design Principles (FIRST + AAA)

### 2.1 Fast

All 120 tests complete in ~34s (including cross-process import isolation tests + network-related tests). No external service dependencies.

### 2.2 Independent

Each test method independently creates its own `ApiKeyPool` / `GapFillAnalyzer` instance. No mutable state is shared between tests. The `setUp` method runs before each test.

### 2.3 Repeatable

Fixed seed=42 random order, no real-time dependencies (`time.monotonic()` used for backoff tests, values manually overridden). Consistent results in any environment, at any time.

### 2.4 Self-validating

All use standard `unittest` assertions. Zero human judgment. Outputs `OK` or `FAIL` + specific failure reason.

### 2.5 Timely

Written synchronously with production code. `_verify_patch_targets()` signature checks ensure tests immediately catch incompatible upstream patches.

### 2.6 AAA Pattern

```python
def test_slots_exhausted_try_acquire_returns_none(self):
    # Arrange — create a pool with 1 key, 2 slots
    pool = _make_pool(n=1, max_concurrent=2)
    a = pool.acquire()
    b = pool.acquire()

    # Act — third acquire attempt
    c = pool.try_acquire()

    # Assert — should return None (slots exhausted)
    self.assertIsNone(c)
```

---

## 3. Detailed Test Coverage Analysis

### 3.1 ApiKeyPool Scheduler (27 tests, 10 classes)

Covers PR review requirements: **pool acquire/release/backoff/recovery mechanisms**

| Test Class | Test Count | Coverage |
|--------|--------|---------|
| `TestCreateApiKeyPoolFromEnv` | 3 | Create pool from env vars, single key, no key |
| `TestAcquireRelease` | 6 | acquire/release/try_acquire, active_requests tracking |
| `TestEdgeCases` | 4 | Empty key list, minimum load scheduling, retry counter, capacity property |
| `TestSnapshot` | 2 | Initial state snapshot, state after usage |
| `TestRecoveredKeyScheduling` | 2 | Re-acquire/try_acquire on recovered keys |
| `TestRateLimitBackoff` | 6 | Exponential backoff 30s×2^n, recovery, consecutive_429 tracking |
| `TestAcquireTimeout` | 1 | acquire(timeout) raises RuntimeError when pool is full |
| `TestConcurrentAcquireRelease` | 1 | No deadlock, active_requests returns to zero |
| `TestResourceLeakRecovery` | 2 | Exceptions between acquire/release do not leak slots |
| `TestIsRateLimit` | 5 | Detect 429 in strings/OpenAI type/keywords |
| `TestSetApiPoolRestore` | 1 | `set_api_pool(None)` restores original factory |

---

### 3.2 GapFillAnalyzer.parse_response (35 tests, 11 classes)

Covers PR review requirements: **GapFillAnalyzer.parse_response**

| Test Class | Test Count | Coverage |
|--------|--------|---------|
| `TestParseResponseValidJSON` | 4 | Single/multiple/empty findings, default values |
| `TestParseResponseInvalidInput` | 9 | Non-JSON, integers, lists, missing fields, null bytes, BOM, illegal severity |
| `TestParseResponseMarkdownFences` | 4 | Fences with/without language tag, jsonp suffix |
| `TestParseResponseFiltering` | 5 | Confidence threshold, unknown rule_id, mixed valid/invalid |
| `TestParseResponseLargeFindings` | 1 | 100 findings parsed in under 1 second |
| `TestParseResponsePydanticModel` | 1 | Pydantic model path delegation |
| `TestStripMarkdownFences` | 4 | Language tag, no tag, trailing whitespace, unclosed fence |
| `TestBuildPrompt` | 2 | Language tag + file tag, numbered content |
| `TestGetBatchesAndCollectFindings` | 2 | One batch per file, flatten |
| `TestRunGapFill` | 3 | English shortcut, empty file cache, full flow |
| Other | 6 | Language injection, finding conversion, scan state, entry construction |

---

### 3.3 Monkey-Patch Context Manager (48 tests, 16 classes)

Covers PR review requirements: **monkey-patching**

| Test Class | Test Count | Coverage |
|--------|--------|---------|
| `TestContextManagerApplyRestore` | 12 | All 7 patches apply/restore, exception safety, functional verification |
| `TestContextManagerNesting` | 2 | Double/triple nesting — only restores on outermost exit |
| `TestSetupFunction` | 2 | `setup_deepseek_compat()` applies patches, repeated calls are idempotent |
| `TestSetupContextInteraction` | 1 | After setup, context manager does not restore on exit |
| `TestImportNoSideEffect` | 1 | **Subprocess verification**: importing runner does not trigger patches (addresses reviewer's import-time side-effects concern) |
| `TestVerifyPatchTargets` | 2 | Guard passes current upstream, triggers check on context enter |
| `TestCheckSignature` | 2 | Raises exception on missing/renamed parameters |
| `TestPatch2OriginalCapture` | 1 | Original `ChatOpenAI.__init__` captured at import time |
| `TestPatch6ChatOpenAITimeout` | 1 | Injects timeout via Pydantic alias |
| `TestPatch7AsyncioQuietLoop` | 3 | asyncio.run replacement, event loop suppression, other exception propagation |
| `TestSanitizeMetaFinding` | 3 | null→"", illegal impact→"low", valid values unchanged |
| `TestStripMarkdownFences` | 4 | Standalone fence stripping tests |
| `TestSetApiPoolRestore` | 1 | `set_api_pool(None)` restores outside context |
| `TestScanState` | 2 | State keys when LLM is enabled/disabled |
| `TestRelName` | 2 | Relative path resolution, fallback to skill name |
| `TestEntryFromResult` | 8 | Various edge cases for entry construction |

**Why subprocess?** Once a patch is applied, it cannot be fully restored within the current process. A subprocess provides a clean Python environment, the only reliable way to verify. This directly addresses the reviewer's "import-time side-effects" concern.

---

### 3.4 Annotation Module (10 tests, 1 class)

| Test Class | Test Count | Coverage |
|--------|--------|---------|
| `TestAnnotateFindings` | 10 | `is_language_compatible` for various language/rule combinations, `annotate_findings` edge cases |

---

### 3.5 Wiring Smoke Test (1 test)

`tests/test_pool_wiring.py` — end-to-end verification:

1. `create_api_key_pool_from_env()` builds a multi-key pool from environment variables
2. `setup_deepseek_compat()` context manager internally calls `set_api_pool()`
3. `get_chat_model()` returns `PooledChatModel` (verifies graph path wiring)
4. `GapFillAnalyzer` also uses `PooledChatModel` (verifies gap-fill path wiring)
5. Patches are automatically restored after context manager exits

---

## 4. Mock and Isolation Strategy

### 4.1 No External Dependencies

The 120 tests **do not make any real network requests**, do not read/write the filesystem, and do not depend on environment variables (except `SKILLSPECTOR_API_KEYS` explicitly set by the wiring test).

### 4.2 ApiKeyPool Test Isolation

- Each test creates an isolated pool instance via the `_make_pool(n, max_concurrent)` factory
- `time.monotonic()` is used for backoff calculation; recovery tests manually override `rate_limited_until`
- Uses fake key strings `"sk-test-a"`, `"sk-test-b"`

### 4.3 GapFillAnalyzer Test Isolation

- `parse_response` receives raw strings — simulating various LLM return formats
- No real LLM calls needed — strings are passed directly
- Instantiating `GapFillAnalyzer` does not trigger network requests

### 4.4 Context Manager Test Isolation

- Each test saves references to original methods; context manager automatically restores on exit
- Cross-process tests use `subprocess.run()` to create a clean Python process, passing the project path via `PYTHONPATH`

---

## 5. How to Run

```bash
# Random order (recommended, seed=42, 120 tests)
cd contrib/multilingual/tests/tests-pro && python random_numbered.py

# pytest sequential execution
pytest contrib/multilingual/tests/tests-pro/ -v

# Smoke test — verify pool wiring (PR #100 Issue 1)
python contrib/multilingual/tests/test_pool_wiring.py

# Mutation test — 30 injected bugs
python contrib/multilingual/tests/tests-pro/mutation_max.py
```

---

## 6. Coverage Blind Spots (Honest Statement)

| Blind Spot | Reason | Mitigation |
|------|------|---------|
| Concurrent race conditions | Requires multi-threaded stress testing | Verified in real 20-worker scans |
| Real 429 response handling | Requires a controllable API server | Indirectly covered by backoff formula unit tests |
| `run_batches` full call chain | Requires mocking LangChain/LangGraph | Indirectly covered by `test_pool_wiring.py` wiring test |
| 9 mutation test escapes | Non-production code paths | All confirmed as non-production bugs, see `docs/MUTATION_PLAN.md` |

---

## 7. Mapping to FIRST Principles

| Principle | Implementation |
|------|------|
| **F**ast | 120 tests ~34s (including ~3s cross-process + ~20s network-related), pure logic tests < 2s |
| **I**ndependent | setUp isolation + factory functions + no shared state |
| **R**epeatable | No network/file/random dependencies (seed=42 fixed random order) |
| **S**elf-validating | unittest assertions, outputs OK/FAIL |
| **T**imely | Written synchronously with production code, `_verify_patch_targets` signature checks |
