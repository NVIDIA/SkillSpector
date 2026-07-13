# Contrib Architecture Flow Diagram

## Batch Entry Point

```
CLI
 │  python -m contrib.multilingual.batch_scan ./tests/fixtures/ --workers 4 [--no-llm]
 │
 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ batch_scan.py :: main()                                              │
│                                                                      │
│  ① discovery.discover_skills(root)                                   │
│     └─ rglob("SKILL.md") → [Path, Path, ...]  sorted                │
│                                                                      │
│  ② detection.detect_skill_language(file_cache)  per skill            │
│     └─ main thread pre-reads → Unicode script ratio → zh/ja/ko/en   │
│                                                                      │
│  ③ api_pool.create_api_key_pool_from_env()  optional                 │
│     └─ SKILLSPECTOR_API_KEYS → ApiKeyPool(10 keys)                  │
│                                                                      │
│  ④ ThreadPoolExecutor(max_workers=4)                                 │
│     ┌─────────────┬─────────────┬─────────────┬─────────────┐       │
│     │  Thread A   │  Thread B   │  Thread C   │  Thread D   │       │
│     │  skill_1    │  skill_2    │  skill_3    │  skill_4    │       │
│     │     │       │     │       │     │       │     │       │       │
│     │     ▼       │     ▼       │     ▼       │     ▼       │       │
│     │  _scan_skill()  parallel, 90s timeout per skill          │       │
│     └─────────────┴─────────────┴─────────────┴─────────────┘       │
│                                                                      │
│  ⑤ Collect results, sort by risk_score descending                    │
│  ⑥ reports._format_terminal / _format_json / _format_markdown       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Per-Skill Scan Flow (`_scan_skill`)

```
_scan_skill(skill_dir, root, use_llm, lang)
│
│  ┌─── ① runner.run_one(skill_dir, root, use_llm, lang) ────────────┐
│  │                                                                   │
│  │   graph.invoke(state)  ←── synchronous, blocks thread             │
│  │   │                                                               │
│  │   │  ┌──────────────────────────────────────────────────────┐     │
│  │   │  │              LangGraph Pipeline                      │     │
│  │   │  │                                                     │     │
│  │   │  │  build_context                                      │     │
│  │   │  │    └─ download/extract/build file cache              │     │
│  │   │  │       temp_dir_for_cleanup ← temporary directory     │     │
│  │   │  │                                                     │     │
│  │   │  │  ┌─── 20 Analyzers parallel fan-out ────────────┐   │     │
│  │   │  │  │                                               │   │     │
│  │   │  │  │  Static rules (no LLM):                      │   │     │
│  │   │  │  │  AST1-8   code injection                     │   │     │
│  │   │  │  │  TT1-5    tool usage                         │   │     │
│  │   │  │  │  YR1-4    YARA rules                         │   │     │
│  │   │  │  │  SC1-6    supply chain                       │   │     │
│  │   │  │  │  LP1-4    loop/recursion                     │   │     │
│  │   │  │  │  TP1-3    tool poisoning                     │   │     │
│  │   │  │  │  TM1-3    tool misuse                        │   │     │
│  │   │  │  │                                               │   │     │
│  │   │  │  │  LLM semantic rules (call LLM):              │   │     │
│  │   │  │  │  SSD1-4   sensitive data disclosure          │   │     │
│  │   │  │  │  SDI1-4   direct injection                   │   │     │
│  │   │  │  │  SQP1-3   suspicious privilege escalation    │   │     │
│  │   │  │  │                                               │   │     │
│  │   │  │  │  Each Analyzer instantiation:                │   │     │
│  │   │  │  │    LLMAnalyzerBase.__init__()                │   │     │
│  │   │  │  │      │                                       │   │     │
│  │   │  │  │      ▼                                       │   │     │
│  │   │  │  │  Patch 1: self.response_schema = None        │   │     │
│  │   │  │  │    → instance attribute, thread-isolated     │   │     │
│  │   │  │  │    → _structured_llm = None                  │   │     │
│  │   │  │  │    → raw text mode                           │   │     │
│  │   │  │  │                                               │   │     │
│  │   │  │  │  Patch 2: parse_response → JSON parse         │   │     │
│  │   │  │  │  Patch 4: build_prompt → JSON instruction     │   │     │
│  │   │  │  │  Patch 6: ChatOpenAI → httpx.Timeout          │   │     │
│  │   │  │  └───────────────────────────────────────────┘   │     │
│  │   │  │                                                     │     │
│  │   │  │  meta_analyzer (after fan-out fan-in)               │     │
│  │   │  │    └─ LLMMetaAnalyzer.__init__()                    │     │
│  │   │  │         Patch 1 ensures instance isolation          │     │
│  │   │  │         Patch 3: parse_response → JSON + sanitize   │     │
│  │   │  │         Patch 5: build_prompt → JSON instruction    │     │
│  │   │  │                                                     │     │
│  │   │  │  Results → filter → risk_score                      │     │
│  │   │  └─────────────────────────────────────────────────────┘     │
│  │   │                                                               │
│  │   result = {                                                      │
│  │     findings, filtered_findings, risk_score, risk_severity,      │
│  │     manifest, component_metadata, temp_dir_for_cleanup            │
│  │   }                                                               │
│  │                                                                   │
│  │   entry_from_result(result)                                       │
│  │     └─ extract fields → annotation.annotate_findings              │
│  │                                                                   │
│  └── ② return (entry, error_msg, rel_name) ─────────────────────────┘
│
│  ┌─── ③ non-English + use_llm → gap_fill ───────────────────────┐
│  │                                                                 │
│  │   run_gap_fill(file_cache, lang, model)                        │
│  │     └─ GapFillAnalyzer(language, model)                        │
│  │          ├─ response_schema = None  (class attr, by design)    │
│  │          ├─ parse_response()  manual JSON + Pydantic           │
│  │          └─ runs through ApiKeyPool for key failover           │
│  │     │                                                          │
│  │     ▼                                                          │
│  │   8 rules: P5, P6-P8, MP1-MP3, RA1-RA2                        │
│  │   (the 8 English-keyword static rules with no semantic         │
│  │    analyzer equivalent)                                        │
│  │                                                                 │
│  │   entry["issues"] += annotate_findings(gap_findings)           │
│  └─────────────────────────────────────────────────────────────────┘
│
│  Return entry (one record in batch results)
```

---

## Three Execution Paths (Post-Fix)

```
Path 1 — --no-llm (fast, deterministic):
────────────────────────────────────────
  use_llm=False → graph skips SSD/SDI/SQP/meta
  → Patches 1-7 still active but irrelevant (no LLM calls)
  → Static-only, matches upstream exactly
  → cleanup_result normal ✅


Path 2 — use_llm=True, all threads fine:
─────────────────────────────────────────
  Patch 1: each analyzer instance gets self.response_schema=None
    → instance dict isolation, no shared state, no race
  Patch 6: httpx.Timeout(connect=8s, read=30s)
    → hung connections fail fast as clean exceptions
  Patch 7: asyncio.run exception handler
    → "Event loop is closed" noise suppressed
  Patch 2/3: parse_response handles raw JSON
    → findings populated correctly ✅


Path 3 — use_llm=True, connection error:
─────────────────────────────────────────
  httpx connect/read timeout fires → exception
  → propagate through asyncio → graph catches
  → skill returns error entry (not findings)
  → cleanup_result: shutil.rmtree → subprocess fallback
  → other workers continue unaffected ✅
```

---

## The 7 Safety Patches (Explicit context manager)

```
setup_deepseek_compat() context manager
│
├─ Patch 1: LLMAnalyzerBase.__init__
│   self.response_schema = None  (instance attr, thread-isolated)
│
├─ Patch 2: LLMAnalyzerBase.parse_response
│   raw JSON string → json.loads → LLMAnalysisResult → Findings
│
├─ Patch 3: LLMMetaAnalyzer.parse_response
│   raw JSON string → json.loads → MetaAnalyzerResult → dicts
│   + sanitize: null→"", "none"→"low"
│
├─ Patch 4: LLMAnalyzerBase.build_prompt
│   append JSON output format instruction
│
├─ Patch 5: LLMMetaAnalyzer.build_prompt
│   append JSON output format instruction
│
├─ Patch 6: ChatOpenAI.__init__
│   inject httpx.Timeout(connect=8s, read=30s) before client caching
│
└─ Patch 7: asyncio.run
    suppress "Event loop is closed" from httpx cleanup
```

**Key insight:** Patch 1 uses instance attributes (`self.__dict__`), not class
attributes. Each analyzer instance gets its own `None` — zero shared state, zero
race conditions.  Nesting depth is tracked: only the outermost ``setup_deepseek_compat()``
exit restores the originals.
