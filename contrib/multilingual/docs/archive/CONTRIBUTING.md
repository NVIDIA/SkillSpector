# Contributing — Multilingual Batch Scanner

For developers who want to understand, extend, or fix this module.

## Quick Orientation

```
contrib/multilingual/
├── batch_scan.py      # CLI entry + ThreadPoolExecutor (start here)
├── runner.py          # graph.invoke() wrapper + 7 safety patches (core)
├── gap_fill.py        # GapFillAnalyzer — LLM pass for 8 uncovered rules
├── api_pool.py        # ApiKeyPool — multi-key scheduler
├── detection.py       # Unicode script-ratio language detection
├── annotation.py      # Finding language-compatibility labels
├── discovery.py       # Recursive SKILL.md finder
├── reports.py         # Terminal / JSON / Markdown formatters
└── docs/              # All documentation
```

**Read order for new developers:**
1. `README.md` — what this module does
2. `DESIGN.md` — architecture, concurrency model, patch rationale
3. Then the source files in the order above

## How It Works (Two-Minute Version)

The module wraps SkillSpector's single-skill pipeline inside a parallel map:

```python
# What upstream does:
state → graph.invoke(state) → result    # one skill at a time

# What we do:
ThreadPoolExecutor.map(graph.invoke, [state_1, state_2, ...])   # N skills in parallel
```

The complication: DeepSeek's API doesn't support `response_format` (structured
output).  Upstream's `LLMAnalyzerBase` calls `with_structured_output()`
unconditionally.  Sending `response_format` to DeepSeek returns HTTP 400,
corrupting the connection pool.

Our 7 safety patches (`runner.py`) work around this by:
1. Disabling structured output (instance-level `response_schema = None`)
2. Adding JSON format instructions to every prompt
3. Parsing raw JSON strings manually
4. Enforcing HTTP timeouts to prevent hung connections
5. Silencing harmless asyncio cleanup noise

Call ``setup_deepseek_compat()`` before any LLM activity to apply them.
The function uses a context manager that tracks nesting depth — only the
outermost exit restores originals.

## Mapping to Upstream SkillSpector

| Upstream concept | Our equivalent | File |
|-----------------|----------------|------|
| `graph.invoke(state)` | `run_one(skill_dir, root, use_llm, lang)` | `runner.py` |
| `LLMAnalyzerBase` | `GapFillAnalyzer(LLMAnalyzerBase)` subclass | `gap_fill.py` |
| `get_chat_model(model)` | `create_api_key_pool_from_env()` → `PooledChatModel` | `api_pool.py` |
| `build_context` node | `_read_skill_files()` | `batch_scan.py` |
| `report.py:_format_json()` | `_format_json(results)` (batch envelope added) | `reports.py` |
| `cli.py scan` command | `batch_scan.py main()` | `batch_scan.py` |
| `ARG1 + env vars` | `argparse` CLI + `.env` dotenv | `batch_scan.py` + `__init__.py` |
| `ANALYZER_NODE_IDS` registry | `_ENGLISH_KEYWORD_RULES` frozenset | `annotation.py` |
| `state["findings"]` with `operator.add` | `annotate_findings()` wrapper | `annotation.py` |

## Key Design Decisions (And Why)

### Zero intrusion on `src/skillspector/`

We subclass, wrap, and monkey-patch — never modify upstream source.  Reason:
upstream releases can be pulled without merge conflicts.  If upstream adds a
native `response_schema=None` mode (e.g., via env var), our patches become
no-ops and can be removed.

### Instance attributes for thread safety

The original approach mutated `LLMAnalyzerBase.response_schema` (class
attribute, shared across all threads).  Race: Thread A restores the original
value while Thread B's meta-analyzer is still creating instances → 400 error.

Fix: `self.response_schema = None` writes to `self.__dict__`.  Python MRO finds
the instance attribute before the class attribute.  Each analyzer gets its own
`None` — zero shared state, zero races.

### httpx.Timeout injection before client caching

`ChatOpenAI.__init__` caches the OpenAI client eagerly.  Patching `timeout`
after construction is too late — the cached client keeps the old value.
Our patch intercepts `__init__` kwargs and overwrites `timeout` (the Pydantic
alias, which v2 prefers over the canonical `request_timeout`) before the
original constructor runs.

## Where to Contribute

### High-impact, moderate-effort

1. **Add checkpoint/resume.**  Write per-skill results to
   `_batch_checkpoint.jsonl` as each skill completes.  On restart, skip skills
   already in the checkpoint.  A 50-line change to `batch_scan.py`.

2. **Add language-detection unit tests.**  Create `tests/test_detection.py`
   with known zh/ja/ko/en file content and verify `detect_language()` output.
   Low complexity, high confidence payoff.

### Moderate-impact, moderate-effort

4. **Expand language detection.**  Add Cyrillic (U+0400–U+04FF → `ru`/`uk`),
   Arabic (U+0600–U+06FF → `ar`), Devanagari (U+0900–U+097F → `hi`).  Each
   is a 3-line change to `detection.py` with threshold constants.

5. **Add SARIF output format.**  Model after upstream's SARIF formatter.
   `Finding` objects map cleanly to SARIF's `result.locations[].physicalLocation`.

6. **Build non-English ground-truth fixtures.**  Create zh/ja/ko skills with
   known vulnerabilities across the 8 gap-fill rules.  Run gap-fill and measure
   precision/recall.  Publish as `tests/fixtures/multilingual/`.

### Lower-priority

7. **Add `--diff` mode.**  Compare two batch JSON reports and show skills that
   changed score.
8. **Deduplicate `_strip_markdown_fences`.**  Currently lives in both
   `runner.py` and `gap_fill.py`.  Move to a shared utility.
9. **Reduce `report.py` Rich StringIO fragility.**  Use `Console(record=True)`
   without `file=` parameter.

## Code Conventions

This module follows SkillSpector upstream conventions exactly:

- **SPDX header** on every `.py` file
- `from __future__ import annotations` as first import
- Imports: stdlib → third-party → internal (`skillspector.*`) → relative (`.`)
- `| None` syntax for optional types (not `Optional[X]`)
- `frozenset` / `Final` for module-level constants (`UPPER_SNAKE_CASE`)
- Private helpers: `_lower_snake_case` functions
- `logger = get_logger(__name__)` in every module with log calls
- Comments explain **why**, not what (the code shows what)
- Docstrings on all public functions and classes

## Testing

### Automated tests (120 tests, 4 modules)

```bash
# Run all tests in randomized order (seed=42)
cd contrib/multilingual/tests/tests-pro && python random_numbered.py

# Or with pytest
pytest contrib/multilingual/tests/ -v
```

| Module | Tests | Covers |
|--------|-------|--------|
| `test_api_pool.py` | 45 | acquire/release, rate-limit backoff, concurrency, edge cases |
| `test_gap_fill.py` | 41 | parse_response, JSON recovery, prompt building, batch/collect |
| `test_runner_patches.py` | 24 | `setup_deepseek_compat()`, context manager, nesting, isolation |
| `test_annotation.py` | 10 | `is_language_compatible`, `annotate_findings` |

### Mutation testing

```bash
python contrib/multilingual/tests/tests-pro/mutation_max.py
```

Injects 30 deliberate bugs across Max's 4 risk areas, verifies tests catch them.
Current score: 21/30 caught.

### Manual verification

```bash
# Static mode (sub-second)
python -m contrib.multilingual.batch_scan ./tests/fixtures/ -f terminal --workers 8 --no-llm

# LLM mode (~2 min)
python -m contrib.multilingual.batch_scan ./tests/fixtures/ -f terminal --workers 8
```

Verify: 23/23 skills scanned, exit code 1 (HIGH/CRITICAL skills present),
`safe_skill` and `ssd_clean` both 0/100.

## Commit Style

Follow upstream conventions:
- Present-tense, imperative mood: `fix:`, `feat:`, `docs:`
- Reference upstream issue/PR numbers when relevant
- Co-authored-by trailer for joint work
