# Test Design Document — contrib/multilingual

> **WHY & HOW.** The design rationale behind every test suite — how each
> answers a specific concern from the PR #100 review. For coverage maps
> and run commands, see `TEST_GUIDE.md`.

---

## 1. Design Motivation — Three Reviewer Concerns

rng1995's PR #100 review identified three critical gaps. Each test suite was
designed to address one gap, not just to hit a coverage number.

### 1.1 Issue #1 — "The API key pool is built but never actually used"

**The problem:** `create_api_key_pool_from_env()` was called in `batch_scan.main()`,
but `PooledChatModel` was never instantiated anywhere. Graph analyzers went through
`LLMAnalyzerBase.__init__` → `get_chat_model()` directly, bypassing the pool.
The 590-line pool was dead code.

**Design response:** `set_api_pool()` monkey-patches `get_chat_model` at the module
level so every `ChatOpenAI` instance draws from the shared key ring.

**Why dual-patch?** `llm_analyzer_base` imports `get_chat_model` via
`from skillspector.llm_utils import get_chat_model` at module level. This creates
a local reference in `llm_analyzer_base`'s namespace. Patching only
`llm_utils.get_chat_model` leaves the local reference pointing to the original
function — graph analyzers (95% of LLM calls) bypass the pool entirely.

The fix patches **both** `llm_utils.get_chat_model` and
`llm_analyzer_base.get_chat_model`. `test_pool_wiring.py` verifies all three
paths: `llm_utils` module call, `LLMAnalyzerBase._llm` instance attribute, and
`GapFillAnalyzer.chat_model`.

**Why standalone script, not unittest?** The pool wiring test runs as a
standalone script so it can set `SKILLSPECTOR_API_KEYS` before any imports
and verify the full `create_api_key_pool_from_env` → `set_api_pool` →
`get_chat_model` chain end-to-end. It also verifies `set_api_pool(None)`
restores originals on both modules.

---

### 1.2 Issue #2 — "Import-time global monkey-patching is invasive and fragile"

This concern has two halves: **invasiveness** (patches leak where they shouldn't)
and **fragility** (patches break silently on upstream changes). We designed
separate test suites for each.

---

#### Invasiveness Design (`test_monkeypatch_invasiveness.py`)

**The V1 story (why this matters):** V1 mutated `LLMAnalyzerBase.response_schema`
(class attribute, shared by all threads). Thread A restored the original value
while Thread B was still creating instances → `with_structured_output()` fired
→ HTTP 400. This bug killed V1.

**V2 fix:** `self.response_schema = None` writes to the instance `__dict__`.
Python MRO finds instance attributes before class attributes. Each analyzer
instance gets its own `None` — zero shared state, zero races.

**Design of each test category:**

| Test | Design rationale |
|------|-----------------|
| **Subprocess import isolation** | Once a monkey-patch is applied process-wide, no amount of `tearDown` can prove the import itself is clean. A subprocess provides a pristine Python environment — the only reliable way to verify `import runner` has no side effects. |
| **Thread isolation (50 concurrent instances)** | Creates enough concurrency pressure to surface class-attribute races. If any thread mutates the class instead of the instance, at least one instance will have non-None `response_schema`. Uses `threading.Event` + `start.set()` to fire all threads simultaneously. |
| **Two independent contexts** | Uses `threading.Barrier` to synchronize two threads, each in its own `deepseek_compat()`. Thread A exits first — Thread B must still see patches active (nesting counter, not boolean flag). |
| **Instance-attr isolation** | Verifies `response_schema` is in `instance.__dict__`, not class `__dict__`, and class attribute is untouched. After context exit, new instances get class attribute back. |
| **Exception-safe restore** | `try/except` inside context — verifies `__exit__` always fires, even on exception path. |
| **Nesting** | Double/triple nested contexts — depth counter prevents inner `__exit__` from restoring. Only outermost restores. |

**Why `_force_restore()` in every tearDownClass?** `setup_deepseek_compat()` is
a one-way door — patches persist for the process lifetime. Random-order test
runners shuffle test classes; a class that calls `setup_deepseek_compat()` leaks
patches into the next class. `_force_restore()` loops `_restore_patches()` until
depth reaches zero, guaranteeing a clean slate regardless of test order.

---

#### Fragility Design (`test_monkeypatch_fragility.py`)

**The problem:** Seven monkey-patches depend on internal upstream details:
Pydantic alias precedence, MRO instance-attribute injection, method signatures,
dataclass fields, Pydantic model fields. If upstream changes any of these,
the patches could break silently — no crash, just incorrect behavior.

**Design response:** `_verify_patch_targets()` guard runs BEFORE `_apply_patches()`.
It checks every assumption our patches depend on. If anything changed, it raises
`RuntimeError` immediately with the specific patch number and what broke.

**Design of each test category:**

| Test | Design rationale |
|------|-----------------|
| **Guard passes current upstream** | Verifies no false positive. Tested against NVIDIA/SkillSpector@ab0431f (130+ commits, 89 files) — guard must not raise on the currently-installed upstream. Also tested after apply+restore cycle (state corruption check). |
| **Each of 7 patches individually verified** | For each patch, we temporarily break its specific target and verify the guard catches it with the correct patch number in the error message. This proves every guard check is unique and distinguishable — an operator seeing "Patch 3" in the error knows exactly what broke. |
| **Deep dependency detection** | Beyond function signatures, our patches call `model_validate()`, `to_finding()`, `Batch.file_path`, `MetaAnalyzerResult.findings`, `asyncio.new_event_loop`. These are inside `try/except` blocks — if they silently disappear, the patch catches the exception and returns `[]`, masking the problem. The guard checks these BEFORE patching. |
| **Keyword-only migration** | Python 3.x can change positional params to keyword-only. `_check_signature` detects `Parameter.KEYWORD_ONLY` kind and raises — our call sites pass these positionally. |
| **Atomicity** | Guard failure must leave the process in its original state. We break a target, call `_apply_patches()`, and verify all 5 methods are still originals — the guard raised before any assignment happened. |

**Why `builtins.hasattr` mock for Pydantic deps?** `model_validate` is a
Pydantic metaclass-injected classmethod — `delattr` cannot remove it. We
temporarily replace `builtins.hasattr` to return `False` for the specific
`(obj, name)` pair, simulating its absence without destructive changes.

---

### 1.3 Issue #3 — "The riskiest code is untested"

**The problem:** Pool acquire/release/backoff, monkey-patches, and gap-fill
parsing had zero automated tests. These are concurrency-heavy, failure-prone
pieces where bugs are most likely.

**Design response:** 120 unit tests across 4 modules covering the four risk
areas rng1995 named:

| Reviewer's risk area | Test file | Design approach |
|---------------------|-----------|----------------|
| Pool acquire/release/backoff/recovery | `test_api_pool.py` (45) | Fake keys + `_make_pool()` factory. `time.monotonic()` for backoff math; override `rate_limited_until` for recovery tests. No real HTTP. |
| Gap-fill parsing | `test_gap_fill.py` (41) | Raw string injection simulating LLM output variants: valid JSON, markdown-fenced, malformed, BOM, null bytes, Pydantic model delegation. |
| Monkey-patches | `test_runner_patches.py` (24) | Save originals at module load; context manager scoping; guard verification; signature mutation. |
| Annotation | `test_annotation.py` (10) | All language/rule combination matrices. |

**Why mutation testing?** 30 bugs injected across the 4 risk areas to verify
tests actually catch real defects, not just line coverage. Tests catch 21/30.
The 9 misses are documented as non-production code paths.

---

## 2. Design Principles (FIRST + AAA)

We apply FIRST because rng1995's concern was about **concurrency-heavy, failure-prone**
code — tests must be fast enough to run frequently, independent enough to run in
any order, and repeatable enough to trust.

| Principle | Why it matters here |
|-----------|-------------------|
| **F**ast | 164 tests < 15s. No network calls. Pool tests use fake keys. Parse tests use raw strings. If tests were slow, devs wouldn't run them before pushing. |
| **I**ndependent | Random-order runners (seed=42) shuffle test classes. `_force_restore()` prevents patch leakage. `_make_pool()` factory isolates pool state. No test reads another test's pool. |
| **R**epeatable | `time.monotonic()` for backoff; `rate_limited_until` overridden in recovery tests. No clock deps. No file deps (except subprocess import test). Same result every time. |
| **S**elf-validating | `unittest` assertions. `OK` or `FAIL` + specific reason. Zero human judgment needed. |
| **T**imely | Written with production code. `_verify_patch_targets` guard means tests catch upstream breaks immediately — the guard IS a test that runs at patch-application time. |

AAA pattern keeps tests readable and debuggable:
```python
def test_slots_exhausted_try_acquire_returns_none(self):
    # Arrange — create pool with known state
    pool = _make_pool(n=1, max_concurrent=2)
    pool.acquire(); pool.acquire()
    # Act — the operation under test
    result = pool.try_acquire()
    # Assert — single clear expectation
    self.assertIsNone(result)
```

---

## 3. Isolation Strategy

Each test design decision follows from a specific constraint:

| Strategy | Constraint it solves |
|----------|---------------------|
| No real network requests | Tests must pass offline, in CI, behind firewalls |
| Fake keys (`sk-test-a`) | Real keys would make tests environment-dependent |
| `_make_pool()` factory | Each test owns its pool; no shared state |
| `_force_restore()` in tearDownClass | Random-order test runners; patches are process-global |
| `threading.Barrier` for concurrent tests | Need deterministic thread interleaving, not `time.sleep` |
| `builtins.hasattr` mock for Pydantic deps | `model_validate` is metaclass-injected, cannot `delattr` |
| `_TempAttributeOverride` context manager | Non-destructive guard tests: break → verify → restore |
| Subprocess for import isolation | Once patched, can't fully un-patch in-process |

---

## 4. Coverage Blind Spots (Honest)

| Blind Spot | Why we accept it |
|------------|-----------------|
| Real 429 response handling | Requires a controllable API server. Backoff formula verified through `TestRateLimitBackoff` (6 tests). Real 429 behavior validated in production scans. |
| `run_batches` full LangChain chain | Requires mocking LangChain/LangGraph internals. Wired path verified via `test_pool_wiring.py` 3-path smoke. |
| 9 mutation test escapes | All confirmed non-production code paths (dead branches, type-narrowing guards). |
| Pool-level concurrent races (snapshot-vs-acquire, key-recovery-vs-new-acquire) | `TestThreadIsolation` covers the V1 killer bug (class-attr race). Remaining pool races verified in 20-worker production scans. |

---

**Next:** [TEST_GUIDE.md](TEST_GUIDE.md) — coverage maps & run commands · [BUGS_FOUND.md](BUGS_FOUND.md) — 16 bugs found · [Main README](../../docs/README.md) — user guide
