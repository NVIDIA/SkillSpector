# Design History — From Concept to Implementation

> Tracks the evolution of the multilingual batch scanner from initial planning through five design phases to the final shipped implementation.

---

## Phase 1: Problem Statement (early 2026-06-18)

**Upstream limitation:** `skillspector scan` handles exactly one skill per invocation. Scanning a repository with hundreds of skills requires an external loop.

**Multilingual gap:** 25 of SkillSpector's 64 rules are English-keyword regex patterns. For non-English skills (zh/ja/ko), these rules lose ~60% recall. 17 rules have equivalent semantic-analyzer coverage (SSD/SDI/SQP). 8 rules — P5 (harmful content), P6-P8 (system prompt leakage), MP1-MP3 (memory poisoning), RA1-RA2 (rogue agent) — have no equivalent.

**Design principles established:**
1. Zero changes to `src/skillspector/`
2. Subclass and wrap, don't rewrite
3. Output comparable with standard single-skill scan
4. All extensions in `contrib/batch_scan/`

---

## Phase 2: Architecture Design (see `docs/DESIGN.md`)

### Four-layer model

```
CLI layer          python -m contrib.batch_scan.batch_scan
Scheduling layer   ThreadPoolExecutor(max_workers=N)
API Pool layer     ApiKeyPool (multi-key scheduler)
Graph layer        graph.invoke() per skill (upstream, untouched)
```

### Component plan (25 tasks, 5 phases)

1. **Foundation** — discovery, language detection, worker pool
2. **API Pool** — multi-key scheduler with rate-limit backoff
3. **Gap-fill** — LLM analyzer covering 8 uncovered rules
4. **Reports** — aggregated terminal/JSON/Markdown output
5. **Integration** — end-to-end pipeline, comparison with upstream

---

## Phase 3: Key Design Decisions

### ThreadPoolExecutor vs ProcessPoolExecutor

macOS Python 3.13 `spawn` mode reimports LangGraph/LangChain in each child process, causing timeouts. Switched to `ThreadPoolExecutor`.

**Implication:** Threads share memory; requires strict thread safety for all shared state.

### Horizontal throttling vs global semaphore

Chose `--workers` (horizontal, per-skill) over a global shared semaphore (vertical, per-request). Rationale: zero intrusion on upstream's `arun_batches(sem=10)`, user-visible knob, conceptually simple.

### Raw JSON mode for DeepSeek

DeepSeek's API does not support `response_format` (structured output). Rather than building a separate provider, chose to patch `LLMAnalyzerBase.__init__` to inject `response_schema = None` as an instance attribute, then handle JSON parsing manually in `parse_response`.

### Unicode script-ratio language detection

Chose stdlib `unicodedata` over ML-based detectors (e.g., `langdetect`, `fasttext`). Zero additional dependencies, already imported by upstream's `mcp_tool_poisoning.py`. Thresholds: CJK ≥10% → zh, kana ≥5% → ja, Hangul ≥10% → ko.

---

## Phase 4: Critical Bug Discovery & Resolution

### Bug 1: Race condition in response_schema monkey-patch (BLOCKER)
- **Original approach:** Save → set class attr to None → run → restore class attr
- **Failure mode:** Four threads race on `LLMAnalyzerBase.response_schema`; Thread A restores before Thread B's meta-analyzer instantiates
- **Fix:** Replace class-attribute mutation with `__init__` wrapper that sets `self.response_schema = None` as instance attribute (Patch 1)

### Bug 2: LLM returned natural language instead of JSON (BLOCKER)
- **Cause:** Without `with_structured_output()`, prompts lacked JSON format instructions
- **Fix:** Append explicit JSON schema to all analyzer prompts (Patches 4 & 5)

### Bug 3: Worker threads hung on TCP connections (BLOCKER)
- **Cause:** httpx default `read=None` (infinite wait for first response byte)
- **Fix:** Inject `httpx.Timeout(connect=8s, read=30s)` via `ChatOpenAI.__init__` before client caching (Patch 6)
- **Complication:** Pydantic v2 alias resolution — `timeout` (alias) wins over `request_timeout` (canonical) when both present

### Bug 4: cleanup_result hung on stale file descriptors
- **Cause:** `shutil.rmtree` blocks on macOS with dangling fd from corrupted httpx connections
- **Fix:** Primary `shutil.rmtree` → fallback `subprocess.run(["rm", "-rf"], timeout=10)`

### Bug 5: asyncio "Event loop is closed" noise (COSMETIC)
- **Cause:** httpx background cleanup tasks fire after `asyncio.run()` tears down the event loop
- **Fix:** `asyncio.run` wrapper with exception handler that drops only `Event loop is closed` (Patch 7)

### Bug 6: LLM output quirk sanitization (COSMETIC)
- **Cause:** LLM occasionally returned `null` for string fields, `"none"` for enum
- **Fix:** `_sanitize_meta_finding` — null→`""`, `"none"`→`"low"` + prompt updated (Patch 3)

---

## Phase 5: Implementation Summary

### Files created (9 source + tests + docs)

```
contrib/batch_scan/
├── __init__.py                       # Package init + dotenv pre-loading
├── discovery.py                      # Recursive SKILL.md finder
├── detection.py                      # Unicode script-ratio detection
├── annotation.py                     # Finding language-compatibility
├── api_pool.py                       # ApiKeyPool + PooledChatModel + set_api_pool()
├── gap_fill.py                       # GapFillAnalyzer(LLMAnalyzerBase)
├── batch_scan.py                     # CLI + ThreadPoolExecutor
├── runner.py                         # Graph wrapper + setup_deepseek_compat()
├── reports.py                        # Terminal / JSON / Markdown
├── tests/
│   ├── test_api_pool.py
│   ├── test_gap_fill.py
│   ├── test_pool_wiring.py
│   └── test_runner_patches.py
├── docs/
│   ├── README.md
│   ├── DESIGN.md
│   ├── CONTRIBUTING.md
│   └── archive/
│       ├── ARCHITECTURE_DEEP_DIVE.md
│       ├── DESIGN_HISTORY.md         # This file
│       ├── FLOW_DIAGRAM.md
│       ├── QUICKSTART.md
│       └── FUTURE_WORK.md
```

### Performance (23-skill test suite, Mac Mini M4)

| Mode | Workers | Time | vs upstream |
|------|---------|------|-------------|
| Upstream (serial loop) | 1 | 5.97s | 1× |
| Batch `--no-llm` | 4 | 0.84s | 7.1× |
| Batch `--no-llm` | 7 | ~0.7s | 8.5× |
| Batch LLM | 7 | ~3 min | N/A (upstream has no LLM batch) |

---

## Design Principles (Recap)

1. **Zero intrusion** — not a single line changed in `src/skillspector/`
2. **Subclass, don't rewrite** — GapFillAnalyzer extends LLMAnalyzerBase
3. **Wrap, don't drill** — ApiKeyPool wraps ChatOpenAI
4. **Tag, don't restructure** — metadata fields on existing output shape
5. **Compare, don't hide** — `scan_mode` label enables upstream diff
6. **Prove first, merge later** — contrib stays independent until value is proven
