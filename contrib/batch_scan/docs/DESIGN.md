# Design — Multilingual Batch Scanner

> Built against SkillSpector v2.2.3.  This contrib module has its own
> independent versioning; the upstream version is noted for compatibility
> reference only.

## Architecture

```
CLI
 │  python -m contrib.batch_scan.batch_scan ./tests/fixtures/ --workers 7
 │
 ▼
batch_scan.py :: main()
 ├─ discover skills (recursive SKILL.md finder)
 ├─ detect language (Unicode script-ratio, per skill)
 ├─ create API pool (optional, 10-key scheduler)
 ├─ ThreadPoolExecutor(max_workers=N)
 │   ├─ Thread A: skill_1 → graph.invoke() + gap-fill
 │   ├─ Thread B: skill_2 → graph.invoke() + gap-fill
 │   └─ ...
 ├─ collect results, sort by risk score
 └─ report (terminal / JSON / Markdown)
```

### Per-skill flow

```
run_one(skill_dir)
 ├─ scan_state()          # build initial LangGraph state
 ├─ graph.invoke(state)   # upstream pipeline (unchanged)
 │   ├─ build_context     # file cache, manifest
 │   ├─ 20 analyzers      # fan-out (15 static + 5 LLM)
 │   └─ meta_analyzer     # LLM verification + enrich
 ├─ entry_from_result()   # extract + annotate
 └─ cleanup_result()      # shutil.rmtree → subprocess fallback
```

## Three-layer concurrency

```
Layer 3 — batch_scan.py:        ThreadPoolExecutor(max_workers=N)  [CONTRIB]
Layer 2 — llm_analyzer_base:    asyncio.Semaphore(10)               [UPSTREAM]
Layer 1 — graph.py:             20 analyzers fan-out                [UPSTREAM]
```

Each layer is unaware of the others.  The graph doesn't know it's being called
concurrently; the workers don't know the graph fans out internally.

## Why ThreadPoolExecutor

- ProcessPoolExecutor hangs on macOS (spawn mode reimports LangGraph per child)
- `graph.invoke()` is a pure function — same state → same result, no shared state
- Each thread operates on its own state dict, isolated from other threads

## DeepSeek compatibility patches

Call ``setup_deepseek_compat()`` before any LLM activity to apply seven targeted
monkey-patches.  The patches are applied explicitly (not at import time) via a
context manager that restores originals on exit.  Nesting is tracked internally
— only the outermost exit restores.

| # | Target | Mechanism | Why |
|---|--------|-----------|-----|
| 1 | `LLMAnalyzerBase.__init__` | `self.response_schema = None` (instance attr) | Disable structured output; instance-isolated |
| 2 | `LLMAnalyzerBase.parse_response` | `json.loads` → Pydantic validate | Handle raw string (no `response_format`) |
| 3 | `LLMMetaAnalyzer.parse_response` | Same + sanitize null/`"none"` | LLM output quirks |
| 4 | `LLMAnalyzerBase.build_prompt` | Append JSON output instruction | Model needs format hint |
| 5 | `LLMMetaAnalyzer.build_prompt` | Same | Same |
| 6 | `ChatOpenAI.__init__` | `httpx.Timeout(connect=8s, read=30s)` | Prevent hung connections |
| 7 | `asyncio.run` | Exception handler: drop `Event loop is closed` | Suppress cleanup noise |

### Why instance attributes (Patch 1 is the key insight)

The original approach mutated `LLMAnalyzerBase.response_schema` (class attribute,
shared by all threads).  Race: Thread A restores the original value while
Thread B is still creating instances → `with_structured_output()` fires → 400.

The fix: `self.response_schema = None` writes to the instance `__dict__`.
Python MRO finds the instance attribute before the class attribute.  Each
analyzer instance gets its own `None` — zero shared state, zero races.

### Why `ChatOpenAI.__init__` (Patch 6 pipeline)

httpx defaults: `connect=5.0`, `read=None` (infinite).  A TCP connection that
is accepted but never sends a response byte blocks the worker thread forever.
ThreadPoolExecutor cannot kill threads.

The fix injects `httpx.Timeout` via the `timeout` Pydantic alias **before**
the internal OpenAI client is cached.  `ChatOpenAI`'s Pydantic model defines
`request_timeout` as the canonical field name with `timeout` as its alias
(`populate_by_name=True`).  When both the alias and canonical name appear in
`**kwargs`, Pydantic v2 prefers the alias — so we overwrite `kwargs["timeout"]`
directly rather than setting `kwargs["request_timeout"]`.  This ensures the
``httpx.Timeout(connect=8s, read=30s)` value flows into every `root_client`
and `async_client` from their first instantiation.

## DeepSeek compatibility

DeepSeek's API does not support `response_format` (structured output).
Upstream calls `with_structured_output()` unconditionally.  Without patches,
this returns HTTP 400, corrupting the httpx connection pool.

The fix chain:
1. Patch 1 disables `with_structured_output()` → raw text responses
2. Patches 4/5 append JSON format instructions to every prompt
3. Patches 2/3 parse raw JSON strings manually with Pydantic validation

## Language detection

Unicode script-ratio heuristic, zero additional dependencies (uses `unicodedata`
from stdlib, already imported by upstream).

```
CJK Unified (0x4E00–0x9FFF)    → zh  (≥10% of alpha chars)
Hiragana + Katakana            → ja  (≥5%)
Hangul Syllables (0xAC00–0xD7AF) → ko  (≥10%)
Otherwise                       → en
```

Aggregated per file by majority vote.  Known limitation: Japanese text with
high kanji and low kana density misclassifies as Chinese.

## Gap-fill

When a skill is non-English, 25 English-keyword static rules lose recall.
17 are covered by SSD/SDI/SQP (semantic analyzers).  8 have no equivalent:

**P5** (harmful content), **P6–P8** (system prompt leakage),
**MP1–MP3** (memory poisoning), **RA1–RA2** (rogue agent).

`GapFillAnalyzer` extends `LLMAnalyzerBase` with a language-aware prompt,
runs via `ApiKeyPool` for key failover, and appends findings to the graph result.

## API Pool

Call ``set_api_pool(pool)`` before scanning to route **all** LLM calls — both
graph-internal analyzers (SSD/SDI/SQP/meta, 20 per skill) and the gap-fill pass —
through a shared key pool.  ``set_api_pool(None)`` restores the original factory.

Kubernetes-scheduler-inspired design:

```
acquire → pick least-loaded idle key
release(success=True)  → mark idle
release(success=False) → mark rate_limited, backoff 30s × 2^n (cap 300s)
acquire after 429      → picks different key automatically
```

The pool is created once and passed to ``set_api_pool()``, which patches both
``skillspector.llm_utils.get_chat_model`` **and**
``skillspector.llm_analyzer_base.get_chat_model`` — the latter is necessary
because ``llm_analyzer_base`` imports ``get_chat_model`` via ``from ... import``
at module level, creating a local reference that a single-module patch would
miss.  Without the dual patch, graph-internal analyzers (95% of LLM calls)
bypass the pool entirely.  ``test_pool_wiring.py`` verifies all three call paths
are wired: ``llm_utils``, ``LLMAnalyzerBase._llm``, and ``GapFillAnalyzer.chat_model``.

## cleanup_result resilience

```python
try:
    shutil.rmtree(temp_dir, ignore_errors=True)
except Exception:
    subprocess.run(["rm", "-rf", temp_dir], timeout=10, capture_output=True)
```

`shutil.rmtree` blocks on macOS when the directory contains files with
dangling fd (e.g., from corrupted httpx connections).  The subprocess
fallback runs outside the Python process and is unaffected.  Platform
detection (`os.name`) selects `rm -rf` on Unix or `rmdir /s /q` on
Windows.

## Per-skill timeout (90s)

A skill that takes >90s is marked TIMEOUT and skipped.  Other workers continue.
HTTP-level timeouts (Patch 6) prevent most hangs from reaching the 90s ceiling.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | All safe |
| 1 | ≥1 skill HIGH or CRITICAL |
| 2 | Scan errors |

## File layout

```
contrib/batch_scan/
├── __init__.py          # package init + dotenv preload
├── batch_scan.py        # CLI + ThreadPoolExecutor
├── runner.py            # graph wrapper + setup_deepseek_compat()
├── discovery.py         # SKILL.md finder
├── detection.py         # language detection
├── annotation.py        # finding compatibility labels
├── gap_fill.py          # GapFillAnalyzer
├── api_pool.py          # ApiKeyPool + PooledChatModel + set_api_pool()
├── reports.py           # Terminal / JSON / Markdown
├── .env.example         # configuration template
├── CONTRIBUTING.md      # dev setup, testing, code conventions
├── tests/
│   ├── test_pool_wiring.py
│   ├── test_monkeypatch_invasiveness.py
│   ├── test_monkeypatch_fragility.py
│   ├── tests-pro/       # 120 unit tests (4 modules)
│   └── docs/            # TEST_DESIGN, TEST_GUIDE, BUGS_FOUND
└── docs/
    ├── README.md        # user-facing guide
    ├── DESIGN.md        # this file
    ├── REVIEW_RESPONSE.md
    └── archive/         # deep dives, history, future work
```

## Rejected Alternatives

### Why ThreadPoolExecutor + asyncio, not full asyncio?

`graph.invoke(state)` is a synchronous blocking call.  LangGraph's compiled
graph executes nodes sequentially and fans out analyzers internally — it does
not expose an async entry point.  Replacing `graph.invoke()` with an async
equivalent would require modifying upstream's graph compilation, which violates
the zero-intrusion constraint.

The alternative — `asyncio.to_thread()` wrapping `graph.invoke()` inside an
async event loop — adds a scheduling layer without removing the thread-per-skill
requirement.  It would also require all batch orchestration code to be async,
complicating the CLI layer (`argparse`, Rich console output) with no throughput
gain.

`ProcessPoolExecutor` was tested and rejected: macOS Python 3.13 `spawn` mode
reimports LangGraph + LangChain per child process, causing 30+ second startup
timeouts.  `fork` mode is unavailable on macOS since Python 3.8.

### Why monkey-patch, not fork upstream?

Forking would create a permanent divergence.  Every upstream release would
require rebasing and re-verifying.  The monkey-patch approach keeps the contrib
module as a drop-in adapter: it tracks upstream automatically, and if upstream
adds a `response_schema` override (e.g., an env var `SKILLSPECTOR_RAW_LLM`),
the patches become no-ops and can be removed without code changes.

### Why 8 gap-fill rules, not a full second graph pass?

The 8 gap-fill rules (P5, P6-P8, MP1-MP3, RA1-RA2) are the intersection of:

1. **English-keyword dependency.**  Each rule's static analyzer uses regex
   patterns that match English text only (e.g., "print your system prompt",
   "clear your memory", "you are no longer an assistant").  Non-English
   text bypasses these patterns entirely.
2. **No semantic-analyzer equivalent.**  SSD (semantic security discovery),
   SDI (semantic developer intent), and SQP (semantic quality policy) cover
   17 other English-keyword rules because those rules detect semantics (intent,
   policy violation) rather than specific English phrases.
3. **LLM-solvable.**  The 8 rules describe security concepts (harmful content,
   memory manipulation, rogue persistence) that an LLM can recognize in any
   language when given a targeted prompt.

The standard for inclusion is: the static regex is provably English-only (by
inspecting `static_patterns_*.py` source), and no semantic analyzer claims the
rule ID in its coverage set.  Rules satisfying both criteria are gap-fill
candidates.

## Patch 2/3 Deep Dive: JSON Parse + Pydantic Validate

Patches 2 and 3 replace `LLMAnalyzerBase.parse_response` and
`LLMMetaAnalyzer.parse_response` respectively.  Both follow the same pipeline:

```
raw LLM string → _strip_markdown_fences() → json.loads() → model_validate() → Finding objects
```

The two-step parse (stdlib `json.loads` then Pydantic `model_validate`) exists
because:

1. `json.loads` is fast, deterministic, and raises clear `JSONDecodeError` on
   malformed output — we catch this and return `[]` (empty findings).
2. `model_validate` enforces the schema: required fields, literal enums,
   confidence range, string length.  Schema violations are caught and returned
   as `[]` with a warning log.

**Error propagation:** If the LLM returns invalid JSON or schema-mismatched
output, the analyzer returns `[]` (no findings for that file).  The scan
continues — a single malformed LLM response never blocks the pipeline.
The warning is logged at `WARNING` level so operators can monitor parse-failure
rates without sifting through debug logs.

Patch 3 adds a `_sanitize_meta_finding()` pass after validation to handle
known LLM quirks: `null` string fields → `""`, unrecognized enum values
(e.g., `"none"`) → `"low"`.  These are applied post-validation because they
represent recoverable soft errors, not hard schema violations.

## Gap-Fill Rule Selection Criteria

The 25 English-keyword static rules in upstream SkillSpector are:

| Group | Rule IDs | Detection method |
|-------|----------|-----------------|
| Prompt injection | P1-P4 | English-keyword regex |
| Harmful content | **P5** | English-keyword regex |
| System prompt leakage | **P6-P8** | English-keyword regex |
| Data exfiltration | E1-E4 | English-keyword regex |
| Privilege escalation | PE1-PE3 | English-keyword regex |
| Excessive agency | EA1-EA4 | English-keyword regex |
| Output handling | OH1-OH3 | English-keyword regex |
| Trigger abuse | TR1-TR3 | English-keyword regex |
| Memory poisoning | **MP1-MP3** | English-keyword regex |
| Rogue agent | **RA1-RA2** | English-keyword regex |

SSD, SDI, and SQP (semantic analyzers) cover the semantic intent behind
P1-P4, E1-E4, PE1-PE3, EA1-EA4, OH1-OH3, and TR1-TR3 — 17 rules total.
The remaining 8 rules (P5, P6-P8, MP1-MP3, RA1-RA2) are flagged as
gap-fill targets because their static detectors rely on specific English
phrases (e.g., `r"(clear|erase|wipe|forget)\s+(your|my|the)\s+(memory|context|instructions)"`)
that have zero recall on non-English text.

---

**Next:** [README.md](README.md) — user guide & all commands · [REVIEW_RESPONSE.md](REVIEW_RESPONSE.md) — PR #100 review response · [CONTRIBUTING.md](../CONTRIBUTING.md) — dev setup
