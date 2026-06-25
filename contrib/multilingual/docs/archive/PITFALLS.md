# Pitfalls & Lessons Learned

> Hard-won lessons from building this module.  If you're extending the batch
> scanner, read this before touching the concurrency or patch code.

---

## Thread Safety

### Class attributes are shared across threads — instance attributes are not

The original approach saved, mutated, and restored `LLMAnalyzerBase.response_schema`
as a class attribute.  With 4 threads running `graph.invoke()` concurrently,
Thread A restored the original value while Thread B's meta-analyzer was still
creating instances — sporadic 400 errors.

**Lesson:** `self.response_schema = None` writes to `self.__dict__`.  Python MRO
finds the instance attribute before the class attribute.  Each analyzer gets its
own copy.  Zero shared state, zero races.

### asyncio.Semaphore instances are independent per graph invocation

Upstream uses `asyncio.Semaphore(10)` per analyzer.  When N skills run in parallel
via `ThreadPoolExecutor`, each skill creates independent semaphore instances —
theoretical peak is `N × 40` concurrent requests.  The `--workers` knob is the
only practical throttle without modifying upstream.

**Lesson:** Count layers of concurrency before adding more.  This system already
has three (`ThreadPoolExecutor` → `asyncio.Semaphore` → 20-analyzer fan-out).

---

## DeepSeek Compatibility

### `response_format` → HTTP 400, silently corrupts the connection pool

DeepSeek's API does not support structured output.  Sending `response_format`
returns 400, which httpx does not clean up properly.  Subsequent requests on the
same connection pool fail with obscure errors.

**Lesson:** Patch 1 (`response_schema = None`) must be applied before **any**
`LLMAnalyzerBase` instantiation.  The `setup_deepseek_compat()` context manager
guarantees this.

### Pydantic v2 alias precedence: `timeout` beats `request_timeout`

`ChatOpenAI.__init__` accepts both `timeout` (alias) and `request_timeout`
(canonical).  When both are present in `**kwargs`, Pydantic v2 prefers the alias.
The client is cached eagerly — patching after `__init__` returns is too late.

**Lesson:** Overwrite `kwargs["timeout"]` (alias) before the original constructor
runs.  `kwargs["request_timeout"] = value` is silently ignored.

### Account-level rate limiting cannot be bypassed with multiple keys

10 API keys under one DeepSeek account share a single concurrency budget.
The pool provides key-level failover but cannot increase throughput beyond the
account limit.  API speed also varies 2–3× by time of day (99s at 6am, 160s at 4pm).

**Lesson:** The pool helps with per-key 429s.  It cannot fix account-level throttling.

---

## Performance Optimization Pitfalls

Seven optimization attempts were evaluated and reverted.  Each made things worse.

| Attempt | What happened | Why it failed |
|---------|--------------|---------------|
| Async pool (re-entrant `asyncio.run`) | Deadlocks | `asyncio.run()` cannot be nested; `graph.invoke()` already calls it |
| Global shared semaphore | Slower than baseline | Cross-thread lock contention outweighed any request smoothing |
| Slot-count-based scheduling | Workers starved | Available slots ≠ available concurrency budget |
| `ChatOpenAI` instance caching | Slower than baseline | Internal `AsyncClient` is event-loop-bound; cached instances cross loops |
| Batch-level pool wrapping | Lost key isolation | One bad key blocked all workers |
| Connection-pool reuse | 400 contamination spread | Corrupted connections propagated across requests |
| Immediate retry on 429 | Thundering herd | Retry without backoff multiplied load on the rate limiter |

**Lesson:** The baseline (ThreadPoolExecutor + ApiKeyPool + 30s exponential backoff)
is the most stable configuration found after 13 iterations.  Any optimization
that changes the concurrency model should be benchmarked against the 23-skill
fixture suite with both `--no-llm` and LLM modes.

---

## Cross-Platform Gotchas

### `shutil.rmtree` hangs on macOS with dangling file descriptors

When httpx connections are corrupted (e.g., after a 400 response), the temp
directory may contain files with dangling fd.  `shutil.rmtree` blocks indefinitely
on macOS.  `ignore_errors=True` handles this on all tested platforms.

### `ProcessPoolExecutor` + macOS `spawn` = 30s timeouts

macOS Python 3.13 uses `spawn` as the default multiprocessing start method.
Each child process reimports LangGraph + LangChain, causing 30+ second startup
times.  `fork` mode is unavailable on macOS since Python 3.8.

**Lesson:** `ThreadPoolExecutor` is the only viable option for cross-platform
parallel skill scanning without modifying upstream.

---

## Patch Design

### Narrow exception handlers

Catching `Exception` in a parse-response path masks the difference between
"the LLM returned bad JSON" (recoverable, log and return `[]`) and "the schema
changed upstream" (needs a code fix).  Split into:

```python
try:
    data = json.loads(text)
except json.JSONDecodeError:
    # LLM output malformed — recoverable
    return []
try:
    result = Model.model_validate(data)
except Exception:
    # Schema mismatch or unexpected error — log and surface
    return []
```

**Lesson:** The second `except Exception` is a safety net for upstream changes.
The first `except JSONDecodeError` is narrowly scoped to LLM output quality.

### Verify upstream signatures at patch time

Monkey-patches depend on upstream method signatures.  If upstream changes a
patched method's parameters, the patch can break silently (wrong number of
arguments passed through `*args`/`**kwargs`).

`_verify_patch_targets()` checks signatures at context-enter time and raises
immediately with a clear error message naming the mismatched method.

**Lesson:** Defensive guards catch drift before it becomes a runtime mystery.

---

## Development Workflow

### Always test with a real API key before claiming "it works"

The `--no-llm` path is fast and deterministic.  The LLM path adds network
latency, rate limiting, and JSON output variance.  Many bugs only manifest
under concurrent LLM load.  Run at least one `--workers 4` LLM scan before
declaring a change complete.

### The fixture suite is your safety net

```bash
python -m contrib.multilingual.batch_scan ./tests/fixtures/ -f terminal --workers 8
cd contrib/multilingual/tests/tests-pro && python random_numbered.py
python contrib/multilingual/tests/tests-pro/mutation_max.py
```

Three commands catch most regressions: batch scan → unit tests → mutation tests.
Run all three after any change to `api_pool.py`, `runner.py`, or `gap_fill.py`.
