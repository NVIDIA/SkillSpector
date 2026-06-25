# Future Work — Known Limitations & Suggested Directions

> Honest assessment of what the current version does not yet cover,
> and where a motivated contributor could take it next.

---

## 1. API Key Pool Coverage ✅

**Current state:** All LLM calls — both graph-internal analyzers (SSD, SDI, SQP,
meta, 20 per skill) and the gap-fill pass — route through a shared key pool via
``set_api_pool()``.  The pool replaces the global ``get_chat_model`` factory so
every ``ChatOpenAI`` instance draws from the same key ring.

**Remaining gap:** ``set_api_pool`` uses a module-level global for pool reference.
A cleaner approach would be to thread the pool through the graph state or use a
context variable, but the current design is adequate for batch workloads where
the pool is set once before scanning and not changed mid-run.

---

## 2. Checkpoint / Resume

**Current state:** A batch scan that fails at skill 847 of 1000 loses all progress. There is no intermediate state written to disk.

**Impact:** Large repositories require restarting from scratch after any failure.

**Suggested direction:** Write per-skill results to a `_batch_checkpoint.jsonl` as each skill completes (before the aggregated report). On restart, skip skills already in the checkpoint. The file doubles as a progress log.

---

## 3. Language Detection Coverage

**Current state:** Unicode script-ratio detection supports four languages (en, zh, ja, ko). Japanese text with high kanji density and low kana frequency can be misclassified as Chinese. Mixed-language skills take a majority vote with no confidence score.

**Impact:** Non-CJK languages (Arabic, Hindi, Cyrillic, Latin-extended) are classified as English and lose non-English gap-fill coverage.

**Candidate languages (ranked by AI adoption density):**

| Script | Language | Unicode range | Difficulty |
|--------|----------|--------------|------------|
| Cyrillic | Russian (ru) | 0x0400–0x04FF | Low |
| Arabic | Arabic (ar) | 0x0600–0x06FF | Medium — RTL |
| Latin extended | French (fr), German (de), Spanish (es) | 0x00C0–0x024F | Low — diacritics |
| Devanagari | Hindi (hi) | 0x0900–0x097F | Medium |
| Thai | Thai (th) | 0x0E00–0x0E7F | Low |

**Suggested direction (three phases):**

1. **Phase 1 — detection.py extension:** Add Unicode ranges + thresholds. The architecture separates language detection from analysis, so adding a language is adding constants.

2. **Phase 2 — prompt optimization per script family:** Languages in the same script family (e.g., Latin-extended) can share validated prompt templates, reducing maintenance cost.

3. **Phase 3 — standalone contrib module:** If the module grows past 10+ languages, split `detection.py` into an independent multilingual detection layer with gap-fill prompts grouped by script family.

Also: return confidence scores alongside language tags for mixed-content skills, and consider a `--confidence-threshold` flag to control when gap-fill is applied.

---

## 4. Output Formats

**Current state:** Terminal (Rich), JSON, and Markdown. Upstream SkillSpector also supports SARIF.

**Impact:** Teams using SARIF-based CI tooling (GitHub Code Scanning, Azure DevOps) cannot ingest batch results directly.

**Suggested direction:** Add `-f sarif` output. SARIF's `runs[].results[].locations[].physicalLocation` maps cleanly to SkillSpector's `Finding.location` / `file` / `start_line` model. Batch-level metadata can live in `runs[].properties`.

Additionally, a **diff mode** (`--diff report1.json report2.json`) that shows which skills changed score between two scans would help teams track security drift over time.

---

## 5. Automated Testing ✅ (partial)

**Current state:** 120 unit tests across 4 modules (`test_api_pool.py`,
`test_gap_fill.py`, `test_runner_patches.py`, `test_annotation.py`), covering
pool acquire/release, JSON parsing, patch application, and language compatibility.
Mutation testing catches 21/30 injected bugs.

**Remaining gaps:**
- **Language detection** has no unit tests (`detect_language()`, script-ratio thresholds)
- **Integration tests** against `tests/fixtures/` are still manual
- **Non-English ground-truth** fixtures don't exist yet
- **`test_pool_wiring.py`** is a smoke test only — needs expansion

---

## 6. Non-English Gap-Fill Quality Baseline

**Current state:** Gap-fill correctness has been verified by manual inspection of LLM output during development. No systematic ground-truth comparison exists for non-English skills.

**Impact:** We know gap-fill *produces findings*, but we have not measured false-positive rate or recall against known vulnerabilities in non-English skills.

**Suggested direction:** Build a small non-English fixture set (zh/ja/ko skills with known vulnerabilities across the 8 gap-fill rules). Run gap-fill against this set and measure precision/recall. Publish the results as a confidence baseline for users.

---

## 7. Worker Scheduling

**Current state:** Workers are dispatched via `ThreadPoolExecutor(max_workers=N)` with no awareness of API pool capacity. When workers exceed the effective API concurrency limit, excess workers queue and waste resources.

**Empirical finding:** 10–15 workers provides the best observed throughput. Below 10, skills queue unnecessarily. Above 15–20, thread overhead and API contention offset gains. The exact optimal value depends on API provider behavior (account-level concurrency limits, per-request latency variance).

**Suggested direction:** Adaptive worker count based on pool slot availability. If all slots are full, pause skill submission. If slots are idle, ramp up. An `--auto-workers` flag could derive N from pool capacity.

---

## 8. ChatOpenAI Per-Call Instantiation

**Current state:** `_build_llm()` creates a new `ChatOpenAI` instance for every LLM call. With ~800 calls per 23-skill scan, this adds measurable overhead.

**Failed attempt:** Pool-level instance caching was tried but made things slower — `ChatOpenAI`'s internal `AsyncClient` is event-loop-bound.

**Suggested direction:** Per-event-loop caching, or leveraging LangChain's built-in connection pooling more effectively. Estimated ~15–20% speed improvement.

---

## 9. Pool Observability

**Current state:** `try_acquire()` (non-blocking fast path) and `acquire()` (blocking fallback) are both implemented, but we don't track how often each succeeds.

**Suggested direction:** Expose `try_acquire_hits / try_acquire_misses` in `snapshot()` to help operators determine whether the pool has enough capacity.

---

## 10. DeepSeek-Specific Constraints

- **No `response_format` support:** Patch 1 (`response_schema = None`) is required. Any attempt to use `with_structured_output()` returns HTTP 400.
- **Account-level rate limiting:** Multiple API keys under the same DeepSeek account share one concurrency budget. A 10-key pool cannot bypass this limit.
- **API speed variance:** Observed per-skill time varies 2–3× depending on time of day (API server load). The pool provides retry/backoff stability but cannot increase throughput beyond the account rate limit.

---

## 11. Custom Pool vs. Established Libraries

The current `ApiKeyPool` was built from scratch. This works but the problem space is well-traveled territory:

| Library | Pitch |
|---------|-------|
| `rotapool` | Resource pool with health-check-per-call, `CooldownResource` lifecycle — closest to our design |
| `apirotater` | Lightweight key rotation with per-key rate windows |
| `llm-keypool` | Full-featured: multi-provider, capability tags, 429 cooldown, built-in proxy |
| `envrotate` | Minimal: reads keys from env vars, random / round-robin |
| `pyrate-limiter` | General-purpose rate limiter (token bucket, sliding window) — complementary |

**Why not now:** The custom pool is battle-tested, fully understood, and integrated. Replacing it adds a dependency and migration risk. Revisit if maintenance burden grows or a library gains community trust with a benchmark showing clear improvement.

---

## 12. Additional Directions

### MetaAnalyzer Parallelization
The MetaAnalyzer runs after all analyzers complete (graph topology: `analyzers → meta_analyzer → report`). Its LLM calls are inherently sequential to the fan-out phase, accounting for 20–30% of per-skill wall time. Parallelizing the meta-analyzer would require modifying upstream graph topology.

### Local Model Compatibility
The pool and DeepSeek compat patches are designed for OpenAI-compatible endpoints. Ollama and llama.cpp expose similar endpoints — verifying and documenting compatibility would expand deployment options for air-gapped or cost-sensitive environments.

### Cross-File Dataflow Analysis
Gap-fill batches files by token budget; related files may land in different batches. Introducing file-level import dependency analysis during batch construction could improve finding quality for multi-file skills.

### File Cache Optimization
`_read_skill_files()` reads disk twice (language detection + gap-fill) with no cache. Per-skill file I/O is negligible (<5ms) at current scale, but a process-internal dict cache could eliminate redundant reads for large skill directories. Low priority — the bottleneck is LLM calls (seconds), not disk I/O (milliseconds).

| # | Area | Status | Next Step |
|---|------|--------|-----------|
| 1 | Pool coverage | ✅ All LLM paths | Refine global-state approach (context var) |
| 2 | Checkpoint | None | JSONL progress log + skip-on-restart |
| 3 | Language detection | 4 languages, no confidence | Expand to 9+ languages; return confidence scores |
| 4 | Output formats | Terminal/JSON/Markdown | Add SARIF + diff mode |
| 5 | Testing | ✅ 120 tests, 21/30 mutation | Language detection tests + integration tests |
| 6 | Gap-fill baseline | Not measured | Non-English fixture set + precision/recall |
| 7 | Worker scheduling | Naive ThreadPoolExecutor | Adaptive scheduling based on pool capacity |
| 8 | ChatOpenAI caching | New instance per call | Per-event-loop caching |
| 9 | Pool observability | No hit/miss counters | Expose try_acquire metrics in snapshot |
| 10 | DeepSeek constraints | Documented | Upstream `response_format` opt-out would remove Patches 1–5 |
| 11 | Pool vs. libraries | Custom, battle-tested | Revisit if maintenance burden grows |
| 12 | Additional directions | Not started | MetaAnalyzer parallelization, local model compat, cross-file dataflow, file cache |

All items are additive — none require breaking changes to the current API. A contributor can pick one area and ship independently.
