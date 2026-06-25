# SkillSpector Architecture Deep Dive — Concurrency, Safety, and the Contrib Layer

> Audience: Upstream NVIDIA maintainers, new contributors
> Date: 2026-06-19
> Covers: upstream architecture, three-layer parallelism, thread safety, API rate limiting, provider system, contrib integration

---

## 1. The Core Insight: `graph.invoke()` Is a Pure Function

SkillSpector models "scan one skill" as a stateless pure function:

```python
state → graph.invoke(state) → result
```

If you accept this, "scan N skills" is just `map`:

```python
results = map(graph.invoke, states)
```

And parallel map:

```python
with ThreadPoolExecutor(max_workers=4) as pool:
    results = pool.map(graph.invoke, states)
```

The entire contrib design is: **add language detection, API pooling, and comparison markers around the map — never touch the function.**

---

## 2. Statelessness Proof: Layer by Layer

### State layer
```python
class SkillspectorState(TypedDict, total=False):
    input_path: str | None
    file_cache: dict[str, str]
    findings: Annotated[list[Finding], operator.add]
    ...
```
- `total=False` — all fields optional, no init constraints
- `findings` uses `operator.add` reducer — but only within one `invoke()` call
- Each `invoke()` creates a new dict; no cross-invocation references

### Provider layer
```python
def create_openai_compatible_chat_model(*, model, credentials, max_tokens, timeout):
    return ChatOpenAI(model=model, api_key=SecretStr(...), timeout=timeout)
```
- New `ChatOpenAI` instance per call — no connection pool caching
- Credentials from parameters, not global state

### Analyzer layer
```python
class LLMAnalyzerBase:
    def __init__(self, base_prompt, model):
        self._llm = get_chat_model(model=model)     # fresh instance
        self._structured_llm = ...                   # fresh instance
```
- Constructor takes only prompt + model — no external state
- `_llm` is instance-local, not shared

### Graph layer
```python
graph = create_graph()   # compiled once at module load
# Each invoke creates a new state; graph is a read-only execution plan
```
- `graph` = topology blueprint (read-only, stateless)
- `state` = material fed into the pipeline (per-invocation)

### Thread-safety check
```
Thread-1: graph.invoke(state_1) → reads/writes state_1 only
Thread-2: graph.invoke(state_2) → reads/writes state_2 only
Thread-3: graph.invoke(state_3) → reads/writes state_3 only
```
**Safe.** No shared mutable state between threads. The only shared object (`graph`) is a read-only compiled execution plan.

---

## 3. The Three-Layer Parallelism Pyramid

```
Layer 3 — batch_scan.py:        ThreadPoolExecutor(max_workers=N)   across skills  [CONTRIB]
Layer 2 — llm_analyzer_base:    asyncio.Semaphore(10)                per-analyzer   [UPSTREAM]
Layer 1 — graph.py:            20 analyzers fan-out                  per-skill      [UPSTREAM]
```

Each layer is **unaware** of the others:
- Graph doesn't know it's being called concurrently by multiple workers
- Worker doesn't know graph fans out 20 analyzers internally
- LLMAnalyzerBase doesn't know which worker calls it

### Layer 1: Graph fan-out (upstream)

LangGraph semantics: when one node has multiple outgoing edges, target nodes run in parallel. 20 analyzers fan out from `build_context`:
- 15 static analyzers (CPU, milliseconds) — patterns, AST, YARA, supply chain
- 5 LLM analyzers (network, seconds) — SSD, SDI, SQP, TP4, meta

### Layer 2: per-analyzer batching (upstream)

```python
# llm_analyzer_base.py:387
sem = asyncio.Semaphore(max_concurrency=10)

async def _process(batch):
    async with sem:
        response = await self._structured_llm.ainvoke(prompt)
        return self.parse_response(response, batch)

return list(await asyncio.gather(*[_process(b) for b in batches]))
```

Token-budget-aware chunking: files exceeding the model's context window are split by lines with 50-line overlap to prevent boundary misses.

### Layer 3: cross-skill parallelism (contrib)

```python
# batch_scan.py
with ThreadPoolExecutor(max_workers=args.workers) as executor:
    futures = {executor.submit(_scan_skill, dir, root, ...): idx
               for idx, dir in enumerate(skill_dirs)}
    for future in as_completed(futures):
        entry, error, name = future.result(timeout=90)
```

Configurable worker count, per-skill timeout, crash recovery.

---

## 4. Concurrency & Rate Limiting

### Upstream: asyncio.Semaphore(10) only

The sole concurrency control in upstream is a per-analyzer `Semaphore(10)`. No retry, no backoff, no 429 handling — LangChain's `ChatOpenAI` provides default 2 retries for network errors.

### The batch scaling problem

When 4 skills run in parallel via ThreadPoolExecutor, each creates independent `Semaphore(10)` instances. Theoretical peak: `4 × 40 = 160` simultaneous requests to one endpoint.

### Contrib solution: horizontal throttling via `--workers`

Rather than adding a global semaphore (which would require modifying upstream code), the contrib layer controls **how many skills run simultaneously**:

```
ThreadPoolExecutor(max_workers=N)
  ├─ skill_1 → graph.invoke() (upstream untouched)
  ├─ skill_2 → graph.invoke() (upstream untouched)
  └─ ...
```

`--workers` maps to API tier:
| Tier | Workers | Peak concurrent requests |
|------|---------|------------------------|
| Free tier | 1 | 10-15 |
| Paid basic | 4 (default) | 25-40 |
| Enterprise | 8 | 50-80 |

### ApiKeyPool for all LLM calls

All LLM calls — both graph-internal analyzers (SSD/SDI/SQP/meta, 20 per skill)
and the gap-fill pass — route through a shared K8s-scheduler-style key pool via
``set_api_pool()``.  The pool replaces the global ``get_chat_model`` factory,
so every ``ChatOpenAI`` instance draws from the same key ring.

- **Acquire**: least-loaded idle key
- **Rate-limit recovery**: exponential backoff `30s × 2^n`, capped at 300s
- **Automatic failover**: 429 → mark key rate-limited → next acquire picks different key
- **Retry**: `PooledChatModel` wraps LangChain `BaseChatModel` with transparent retry up to 5 attempts

---

## 5. Thread Safety: The 7 Compatibility Patches

Call ``setup_deepseek_compat()`` to apply seven targeted monkey-patches.  The
patches are applied explicitly via a context manager that tracks nesting depth —
only the outermost exit restores originals.  Each addresses a specific DeepSeek
compatibility constraint without modifying upstream source.

### Why patches are needed

DeepSeek's API does not support `response_format` (structured output). The upstream `LLMAnalyzerBase` unconditionally calls `with_structured_output(response_schema)` when `response_schema is not None`. Sending `response_format` to DeepSeek returns HTTP 400, corrupting the httpx connection pool.

### Patch design principle

All patches follow the same pattern: **inject via `__init__` wrapper before the original constructor runs.** This guarantees thread isolation because each instance gets its own value in `self.__dict__`.

| # | Target | What | Why |
|---|--------|------|-----|
| 1 | `LLMAnalyzerBase.__init__` | `self.response_schema = None` (instance attr) | Disable structured output; instance-isolated, no race |
| 2 | `LLMAnalyzerBase.parse_response` | Manual JSON parse + Pydantic validate | Handle raw string responses (no `response_format`) |
| 3 | `LLMMetaAnalyzer.parse_response` | Same + sanitize null→`""`, `"none"`→`"low"` | Handle LLM output quirks |
| 4 | `LLMAnalyzerBase.build_prompt` | Append JSON output instruction | Model needs explicit JSON format without `response_format` |
| 5 | `LLMMetaAnalyzer.build_prompt` | Same for meta-analyzer | Same |
| 6 | `ChatOpenAI.__init__` | `httpx.Timeout(connect=8s, read=30s)` | Prevent hung connections from blocking workers forever |
| 7 | `asyncio.run` | Silent exception handler for `Event loop is closed` | Suppress harmless httpx cleanup noise |

### Patch 1: instance attribute, not class attribute

This is the key insight that resolved the race condition. The original approach mutated `LLMAnalyzerBase.response_schema` (a class attribute shared by all threads). The fix sets `self.response_schema = None` on each instance's `__dict__` — Python MRO finds the instance attribute before the class attribute, so each analyzer instance is independently configured.

### Patch 6: Pydantic alias pipelaying

`ChatOpenAI.timeout` is the alias for `request_timeout`. The OpenAI client is cached eagerly in `__init__`. Pydantic v2 prefers alias values over canonical names when both are present. The patch overwrites `kwargs["timeout"]` (alias) before `__init__` runs, ensuring the timeout flows into every `root_client` / `async_client` from creation.

---

## 6. Bug History: Critical Race Condition Debugging

### Timeline

1. **Symptom:** `--no-llm` works perfectly; LLM path sporadically returns 400 errors or hangs in `cleanup_result`.
2. **Root cause:** Four threads concurrently reading/writing `LLMAnalyzerBase.response_schema` (class attribute). Thread A restores the original value while Thread B's meta-analyzer is still creating instances.
3. **Why meta-analyzer specifically:** It runs late in the graph (after fan-out). By the time its instance is created, another thread may have already restored the schema.
4. **Why 400 causes cleanup hang:** DeepSeek returns 400 for `response_format`. httpx connection pool isn't properly cleaned up after partial 400 responses. `shutil.rmtree` blocks on macOS when the temp directory contains files with dangling fd.
5. **Fix:** Patch 1 (instance attributes) + Patch 6 (httpx timeouts) + `cleanup_result` subprocess fallback.

---

## 7. Provider System

### Three abstraction layers

```
Protocol (base.py)              Implementation (per-provider)
─────────────────               ────────────────────────────
ModelMetadataProvider           openai / anthropic / nv_build
  ├─ get_context_length()         ├─ provider.py
  ├─ get_max_output_tokens()      └─ model_registry.yaml
  └─ resolve_model(slot)

CredentialsProvider
  └─ resolve_credentials()

ChatModelProvider
  └─ create_chat_model()
```

Protocols are structural subtypes — no ABC inheritance. Any object satisfying the method signatures works as a provider.

### Selection chain

```
SKILLSPECTOR_PROVIDER env var
  ├─ "openai"     → OpenAIProvider      → OPENAI_API_KEY
  ├─ "anthropic"  → AnthropicProvider   → ANTHROPIC_API_KEY
  ├─ "nv_build"   → NvBuildProvider     → NVIDIA key
  └─ unset        → NvInferenceProvider (→ NvBuildProvider fallback)
```

---

## 8. Contrib Integration: "Grown On, Not Pushed In"

### Zero files modified in src/skillspector/

The contrib layer sits entirely outside upstream. It imports upstream classes as parents and wraps upstream functions:

```
contrib/multilingual/
├── batch_scan.py      ← CLI + ThreadPoolExecutor
├── runner.py          ← graph.invoke() wrapper + 7 safety patches
├── gap_fill.py        ← GapFillAnalyzer(LLMAnalyzerBase)
├── api_pool.py        ← ApiKeyPool + PooledChatModel
├── detection.py       ← Unicode script-ratio language detection
├── annotation.py      ← finding language-compatibility labeling
├── discovery.py       ← recursive SKILL.md finder
└── reports.py         ← Terminal / JSON / Markdown formatters
```

### Design principles

1. **Subclass, don't rewrite.** GapFill extends `LLMAnalyzerBase` — inherits token budgeting, batching, concurrency.
2. **Wrap, don't drill.** API Pool wraps `ChatOpenAI` rather than modifying its construction.
3. **Tag, don't restructure.** Adds `language_compatible`, `scan_mode`, `enhancements` fields — doesn't change Finding structure.
4. **Compare, don't hide.** `skillspector scan` vs `batch_scan` produce diffable output. `scan_mode` label tracks provenance.

### When to upstream

If batch scanning, multilingual support, and API pooling prove broadly useful:

1. ApiKeyPool → `src/skillspector/providers/pool.py`
2. Language detection → `build_context` node
3. GapFill → register as 21st analyzer node
4. Batch scan → merge into CLI `scan` command

Until then: **prove value first, discuss merging later.**

---

## Appendix: Key File Index

| File | Role |
|------|------|
| `src/skillspector/graph.py` | Graph topology (7 nodes, 20 analyzer fan-out) |
| `src/skillspector/state.py` | State schema (TypedDict) |
| `src/skillspector/llm_analyzer_base.py` | LLM analyzer base (token budget + batching + concurrency) |
| `src/skillspector/providers/__init__.py` | Provider factory + credential fallback chain |
| `src/skillspector/providers/chat_models.py` | ChatOpenAI constructor |
| `src/skillspector/llm_utils.py` | LLM utilities (get_chat_model, chat_completion) |
| `src/skillspector/cli.py` | CLI entry (`scan` command) |
| `src/skillspector/nodes/analyzers/` | 20 analyzer implementations |
| `src/skillspector/nodes/meta_analyzer.py` | Meta-analyzer (LLM verification) |

## Appendix: Glossary

| Term | Meaning |
|------|---------|
| Skill | AI agent skill package (directory or zip) |
| Finding | One security finding (rule_id + severity + line + ...) |
| Batch | One LLM call unit (one file or one chunk) |
| State | Complete input/output of one `graph.invoke()` |
| Provider | LLM backend abstraction (OpenAI / Anthropic / NVIDIA) |
| Meta-analyzer | LLM verification/filtering node |
| Fan-out | One node → multiple parallel nodes |
| Fan-in | Multiple nodes → one aggregation node |
| Chunk | Oversized file split by lines with overlap |
| Semaphore | asyncio concurrency gate |
| API Pool | Multi-key resource scheduler |
