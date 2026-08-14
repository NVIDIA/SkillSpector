# Future Work — Known Limitations & Suggested Directions

> Honest assessment of what the current version does not yet cover,
> and where a motivated contributor could take it next.
> Last updated: 2026-06-26 (post PR #100 review resolution).

---

## 1. API Key Pool Coverage ✅

**Status:** All LLM calls — graph-internal analyzers (20 per skill) and the
gap-fill pass — route through a shared key pool via `set_api_pool()`, which
dual-patches both `llm_utils` and `llm_analyzer_base` to close the `from-import`
local-reference bypass.  `test_pool_wiring.py` verifies all three paths.

**Remaining gap:** `set_api_pool` uses a module-level global for the pool
reference.  A context variable or graph-state threading would be cleaner,
but the current design is adequate for batch workloads where the pool is
set once before scanning.

---

## 2. Checkpoint / Resume

**Current state:** A batch scan that fails at skill 847 of 1000 loses all
progress.  No intermediate state written to disk.

**Impact:** Large repositories require restarting from scratch after any failure.

**Suggested direction:** Write per-skill results to `_batch_checkpoint.jsonl`
as each skill completes.  On restart, skip skills already in the checkpoint.
The file doubles as a progress log.  ~50-line change to `batch_scan.py`.

---

## 3. Language Detection Coverage

**Current state:** Unicode script-ratio detection supports four languages
(en, zh, ja, ko).  Japanese text with high kanji density and low kana
frequency can misclassify as Chinese.  Mixed-language skills use majority
vote with no confidence score.

**Candidate languages (ranked by AI adoption density):**

| Script | Language | Unicode range | Difficulty |
|--------|----------|--------------|------------|
| Cyrillic | Russian (ru) | 0x0400–0x04FF | Low |
| Arabic | Arabic (ar) | 0x0600–0x06FF | Medium — RTL |
| Latin extended | French (fr), German (de), Spanish (es) | 0x00C0–0x024F | Low |
| Devanagari | Hindi (hi) | 0x0900–0x097F | Medium |
| Thai | Thai (th) | 0x0E00–0x0E7F | Low |

**Suggested direction:** Add Unicode ranges + threshold constants to
`detection.py`.  Return confidence scores alongside language tags.
Consider a `--confidence-threshold` flag.

---

## 4. Output Formats

**Current state:** Terminal (Rich), JSON, Markdown.  Upstream also supports SARIF.

**Suggested direction:** Add `-f sarif`.  SARIF's
`runs[].results[].locations[].physicalLocation` maps cleanly to
`Finding.location` / `file` / `start_line`.  Also: a `--diff report1.json report2.json`
mode to track security drift over time.

---

## 5. Automated Testing ✅ (partial)

**Current state:** 164 tests (120 unit + 44 review-themed), covering pool
acquire/release/backoff, gap-fill parsing, monkey-patch invasiveness (thread
isolation, import safety), monkey-patch fragility (per-patch guard verification,
deep dependency detection), and annotation. 30-bug mutation suite catches 21/30.

**Remaining gaps:**
- **Language detection** has no unit tests (`detect_language()`, script-ratio thresholds)
- **Integration tests** against `tests/fixtures/` are still manual
- **Non-English ground-truth** fixtures don't exist yet
- **Pool-level concurrent races** (snapshot-vs-acquire, key-recovery-vs-new-acquire) not yet covered by automated tests

---

## 6. Non-English Gap-Fill Quality Baseline

**Current state:** Gap-fill correctness verified by manual inspection.  No
systematic ground-truth comparison exists for non-English skills.

**Suggested direction:** Build non-English fixtures (zh/ja/ko skills with
known vulnerabilities across the 8 gap-fill rules).  Run gap-fill, measure
precision/recall.  Publish baseline.

---

## 7. Worker Scheduling

**Current state:** `ThreadPoolExecutor(max_workers=N)` with no awareness of
API pool capacity.  When workers exceed effective API concurrency, excess
workers queue and waste resources.

**Suggested direction:** Adaptive worker count based on pool slot availability.
`--auto-workers` flag deriving N from pool capacity.

---

## 8. ChatOpenAI Per-Call Instantiation

**Current state:** `_build_llm()` creates a new `ChatOpenAI` for every LLM call.
~800 calls per 23-skill scan adds measurable overhead.

**Failed attempt:** Pool-level instance caching was tried but made things
slower — `ChatOpenAI`'s internal `AsyncClient` is event-loop-bound.

**Suggested direction:** Per-event-loop caching.  Estimated ~15–20% speed
improvement.

---

## 9. Pool Observability

**Current state:** `try_acquire()` (non-blocking) and `acquire()` (blocking)
both implemented, but hit/miss ratio not tracked.

**Suggested direction:** Expose `try_acquire_hits / try_acquire_misses` in
`snapshot()`.

---

## 10. DeepSeek-Specific Constraints

- **No `response_format` support:** Patch 1 (`response_schema = None`) required.
  Upstream `response_format` opt-out would remove Patches 1–5.
- **Account-level rate limiting:** Multiple keys under one DeepSeek account
  share a concurrency budget.  A 10-key pool cannot bypass this.
- **API speed variance:** Per-skill time varies 2–3× by time of day.

---

## 11. Custom Pool vs. Established Libraries

The `ApiKeyPool` was built from scratch.  Established alternatives:

| Library | Pitch |
|---------|-------|
| `rotapool` | Resource pool with `CooldownResource` lifecycle — closest to our design |
| `apirotater` | Lightweight key rotation with per-key rate windows |
| `llm-keypool` | Multi-provider, capability tags, 429 cooldown, built-in proxy |
| `envrotate` | Minimal: reads keys from env, random / round-robin |
| `pyrate-limiter` | General-purpose rate limiter — complementary |

**Why not now:** The custom pool is battle-tested, fully understood, and
integrated.  Revisit if maintenance burden grows or a library proves itself.

---

## 12. Additional Directions

- **MetaAnalyzer parallelization** — LLM calls account for 20–30% of per-skill
  wall time.  Would require modifying upstream graph topology.
- **Local model compatibility** — Verify/document Ollama/llama.cpp compatibility.
- **Cross-file dataflow analysis** — File-level import dependency analysis
  during batch construction.
- **File cache optimization** — Eliminate redundant disk reads.  Low priority
  (bottleneck is LLM, not I/O).

---

## Summary

| # | Area | Status | Next Step |
|---|------|--------|-----------|
| 1 | Pool coverage | ✅ Dual-patch (llm_utils + llm_analyzer_base) | Context-variable refinement |
| 2 | Checkpoint | None | JSONL progress log + skip-on-restart |
| 3 | Language detection | 4 languages, no confidence | Expand to 9+ languages; return confidence scores |
| 4 | Output formats | Terminal/JSON/Markdown | SARIF + diff mode |
| 5 | Testing | ✅ 164 tests (120 unit + 44 thematic) | Language detection tests + integration tests |
| 6 | Gap-fill baseline | Not measured | Non-English fixture set + precision/recall |
| 7 | Worker scheduling | Naive ThreadPoolExecutor | Adaptive scheduling |
| 8 | ChatOpenAI caching | New instance per call | Per-event-loop caching |
| 9 | Pool observability | No hit/miss counters | Expose try_acquire metrics |
| 10 | DeepSeek constraints | Documented | Upstream `response_format` opt-out |
| 11 | Pool vs. libraries | Custom, battle-tested | Revisit if maintenance burden grows |
| 12 | Additional directions | Not started | MetaAnalyzer, local models, dataflow, cache |

---

For code conventions and commit style, see `../CONTRIBUTING.md`.
