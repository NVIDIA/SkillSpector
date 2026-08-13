# Multilingual Batch Scanner for SkillSpector

[![Tests](https://img.shields.io/badge/tests-164%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Upstream](https://img.shields.io/badge/upstream-NVIDIA%2FSkillSpector-ab0431f-orange)](https://github.com/NVIDIA/SkillSpector)
[![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey)]()

SkillSpector is a static+LLM security analyzer for AI agent skill definitions.
This module extends it to scan **directories** of skills in parallel, with
automatic language detection and targeted LLM gap-fill for non-English skills.
Zero changes to upstream `src/skillspector/`.

**Contents:** [What it does](#what-it-does) · [Quickstart](#quickstart) · [All Commands](#all-commands) · [Running Tests](#running-tests) · [For PR Reviewers](#for-pr-reviewers)

## What it does

```
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ -f terminal --workers 7
```

1. Finds all `SKILL.md`-containing directories under the input root
2. Detects language per skill (en / zh / ja / ko)
3. Runs the full SkillSpector graph pipeline per skill in parallel
4. For non-English skills, applies LLM gap-fill for 8 vulnerability rules
   that English-keyword static patterns cannot detect
5. Produces an aggregated report sorted by risk score

## Quickstart

### Prerequisites

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install SkillSpector in development mode
pip install -e .

# Copy and edit the environment template
cp contrib/batch_scan/.env.example .env
```

The `.env` file needs these keys (see `.env.example` for the full template):

| Variable | Required | Purpose |
|----------|----------|---------|
| `SKILLSPECTOR_PROVIDER` | Yes | `openai` for DeepSeek/OpenAI-compatible |
| `SKILLSPECTOR_MODEL` | Yes | e.g. `deepseek-v4-flash` |
| `OPENAI_API_KEY` | For single-key | Standard OpenAI-compatible key |
| `OPENAI_BASE_URL` | For single-key | e.g. `https://api.deepseek.com/v1` |
| `SKILLSPECTOR_API_KEYS` | For multi-key | Pipe-delimited: `key\|base_url\|model`, one per line |

> **⚠️ Parallel LLM scanning requires multiple API keys.** With `--workers 4`
> and 1 key, you hit rate limits immediately.  Configure at least as many keys
> as workers — 10 keys for `--workers 8` is safe.  The ApiKeyPool handles
> automatic failover when a key is rate-limited.  If you only have 1 key, use
> `--workers 1` or `--no-llm`.

### Static-only (fast, no API keys needed)

```bash
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ --no-llm
```

### Full LLM scan

```bash
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ -f terminal --workers 7
```

### Test with built-in fixtures

```bash
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ -f terminal --workers 8
```

23 skills designed to exercise every detection rule.

## Output formats

| Format | Flag | Use case |
|--------|------|----------|
| Terminal (Rich) | `-f terminal` (default) | Human review |
| JSON | `-f json -o report.json` | CI pipelines |
| Markdown | `-f markdown -o report.md` | PR comments |

### Example: terminal output (23 fixtures, 8 workers)

```
SkillSpector Batch Scan — 23 skill(s) in ./tests/fixtures  (8 workers, 10 API keys)

  [1/23] malicious_skill → 100/100 CRITICAL (14 issue(s))
  [8/23] sdi/sdi1_mismatch → 97/100 CRITICAL (6 issue(s))
  [11/23] sdi/sdi4_divergence → 100/100 CRITICAL (8 issue(s))
  [19/23] ssd/ssd1_semantic_injection → 100/100 CRITICAL (4 issue(s))
  [5/23] mcp_poisoned_tool → 100/100 CRITICAL (16 issue(s))

╭──────────────────────────────────────────────────────────────────╮
│ SkillSpector Batch Scan Report                                   │
╰────────────────── v2.2.3  |  Multilingual Enhanced ──────────────╯

Total: 23 skill(s) scanned

                Skills by Risk Score (23 completed)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━┓
┃ Skill              ┃ LR ┃   Score ┃ Severity ┃ Issues ┃ Lang ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━┩
│ chef-assistant     │ ✓  │ 100/100 │ CRITICAL │     14 │ en   │
│ reаd_data          │ ✓  │ 100/100 │ CRITICAL │     16 │ en   │
│ ...                │    │         │          │        │      │
│ safe-greeting      │ ✓  │   0/100 │ LOW      │      0 │ en   │
│ code-reviewer      │ ✓  │   0/100 │ LOW      │      0 │ en   │
└────────────────────┴────┴─────────┴──────────┴────────┴──────┘

15 skill(s) with HIGH or CRITICAL risk — review immediately
6 skill(s) with LOW risk — likely safe
```

**LR column:** Language Reliability. ✓ = English (full static + LLM coverage).
⚠ = non-English (gap-fill applied, 8 extra rules covered).

### Example: JSON output (excerpt)

```json
{
  "batch": {
    "scanned_at": "2026-06-19T01:20:00+00:00",
    "total_skills": 23,
    "scan_mode": "multilingual-enhanced",
    "enhancements": {
      "language_detection": "unicode-script-ratio",
      "gap_fill_applied": 0,
      "gap_fill_findings": 0
    }
  },
  "skills": [
    {
      "skill": {
        "name": "malicious_skill",
        "source": "malicious_skill",
        "source_group": ".",
        "language": "en",
        "scanned_at": "2026-06-19T01:20:05+00:00"
      },
      "risk_assessment": {
        "score": 100,
        "severity": "CRITICAL",
        "recommendation": "DO NOT INSTALL"
      },
      "issues": [
        {
          "id": "E1",
          "message": "Skill executes shell commands without user consent",
          "severity": "CRITICAL",
          "confidence": 1.0,
          "language_compatible": true
        }
      ],
      "scan_mode": "multilingual-enhanced",
      "enhancements": {
        "gap_fill_applied": false,
        "gap_fill_findings": 0,
        "english_keyword_rules_skipped": 0
      }
    }
  ]
}
```

### LLM vs static comparison (same 23 fixtures, 8 workers)

| Skill | `--no-llm` | LLM mode | What LLM caught |
|-------|-----------|----------|-----------------|
| `ssd1_semantic_injection` | 0/100 (0) | **100/100** (4) | Semantic injection invisible to static |
| `ssd2_novel_phrasing` | 0/100 (0) | **100/100** (3) | Novel phrasing bypasses keyword match |
| `ssd3_nl_exfiltration` | 0/100 (0) | **60/100** (3) | NL-veiled data exfiltration |
| `ssd4_narrative_deception` | 10/100 (1) | **100/100** (9) | Deceptive narrative framing |
| `sdi4_divergence` | 13/100 (2) | **100/100** (8) | Intent-behavior mismatch |
| `sdi1_mismatch` | 52/100 (4) | **97/100** (6) | +2 additional LLM findings |
| `sdi3_scope_creep` | 71/100 (3) | **100/100** (9) | Hidden scope expansion |
| `sqp2_missing_warnings` | 26/100 (2) | **58/100** (3) | Missing safety guardrails |
| `malicious_skill` | 100/100 (6) | 100/100 **(14)** | +8 additional LLM findings |
| `mcp_poisoned_tool` | 100/100 (8) | 100/100 **(16)** | +8 additional LLM findings |
| `safe_skill` | 0/100 (0) | **0/100** (0) | Clean stays clean ✓ |
| `ssd_clean` | 0/100 (0) | **0/100** (0) | Clean stays clean ✓ |

**Key insight:** LLM semantic analyzers (SSD/SDI/SQP) catch entire vulnerability
categories that English-keyword static patterns miss completely.  Clean skills
remain clean — no false-positive inflation.  For skills already flagged by
static rules, LLM finds 2–8 additional issues per skill.

### Quick comparison: upstream vs batch

```bash
# Upstream — scan one skill
skillspector scan ./tests/fixtures/malicious_skill/ -f json -o upstream.json

# Batch — scan all skills
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ -f json -o batch.json
```

Key differences in batch output:
- `scan_mode: "multilingual-enhanced"` — provenance marker
- `enhancements.gap_fill_applied` — true if LLM gap-fill was used
- `enhancements.english_keyword_rules_skipped` — count of static rules bypassed
- `skill.language` — detected language tag

## All Commands

### Scan (LLM mode)

```bash
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ -f terminal --workers 7    # default
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ -f terminal --workers 1    # sequential, easy to read
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ -f terminal --workers 20   # high throughput
```

### Scan (static-only, no API keys)

```bash
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ --no-llm
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ --no-require-llm --no-llm  # skip LLM even for non-English
```

### Output formats

```bash
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ -f terminal                # default (Rich)
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ -f json -o report.json
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ -f markdown -o report.md
```

### Fixture test (built-in 23 skills)

```bash
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ -f terminal --workers 8
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ -f terminal --workers 8 --no-llm
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ -f json -o report.json --workers 8
```

### Language override

```bash
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ --lang auto --workers 4    # detect (default)
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ --lang zh -f terminal --workers 4
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ --lang ja -f terminal --workers 4
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ --lang ko -f terminal --workers 4
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ --lang en -f terminal --workers 4   # skip gap-fill
```

### Debugging

```bash
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ --workers 1 -V             # single worker + verbose
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ --workers 4 -V
skillspector scan ./tests/fixtures/malicious_skill/ --no-llm                   # verify upstream works
```

### Compare upstream vs batch

```bash
skillspector scan ./tests/fixtures/malicious_skill/ -f json -o upstream.json
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ -f json -o batch.json --workers 4
```

### CI

```bash
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ -f json -o report.json --workers 8
if [ $? -eq 0 ]; then echo "All clean"; fi
```

## Tuning `--workers`

| Scenario | Workers | Peak concurrent LLM requests |
|----------|---------|------------------------------|
| Free-tier API key | 1 | 10–15 |
| Paid basic | 4 (default) | 25–40 |
| Enterprise / multi-key | 7–10 | 50–80 |
| Debugging | 1 + `-V` | Sequential, easy to read |

## Language options

```bash
--lang auto    # Unicode script-ratio detection (default)
--lang zh      # Force Chinese
--lang ja      # Force Japanese
--lang ko      # Force Korean
--lang en      # Force English (skip gap-fill)
```

## Debugging

```bash
# Single worker + verbose output — easiest to read
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ --workers 1 -V

# Verify upstream still works
skillspector scan ./tests/fixtures/malicious_skill/ --no-llm
```

## Edge cases

```bash
# Static-only + skip LLM requirement even for non-English skills
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ --no-require-llm --no-llm
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | All safe (no HIGH/CRITICAL) |
| 1 | ≥1 skill has HIGH or CRITICAL risk |
| 2 | Scan errors occurred |

CI usage:

```bash
python -m contrib.batch_scan.batch_scan ./tests/fixtures/ -f json -o report.json
if [ $? -eq 0 ]; then
    echo "All clean"
fi
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| "No LLM API key configured" | Set up `.env` or use `--no-llm` |
| Connection errors / 429 | Reduce `--workers` |
| Skills timing out (90s) | Check network; the scanner skips and continues |
| "Event loop is closed" | Harmless, suppressed |
| model_info token limit warning | Harmless, 128K default used |

## Known Limitations

1. **No checkpoint/resume.**  A failure at skill 847 of 1000 loses all progress.
2. **Language detection covers 4 scripts.**  Arabic, Hindi, Cyrillic are
   classified as English and lose gap-fill coverage.
3. **No SARIF output.**  Upstream supports it; this contrib adds terminal/JSON/Markdown.
4. **Gap-fill quality not benchmarked for non-English.**  No ground-truth comparison exists.
5. **`parse_response` JSON recovery is best-effort.**  When the LLM returns
   malformed JSON, the analyzer returns empty findings (no crash).  This is a
   graceful-degradation choice: a single malformed response won't block the
   pipeline, but the user won't know which findings were lost.

See `DESIGN.md` for architecture details and `docs/archive/FUTURE_WORK.md` for suggested directions.

## Running Tests

```bash
# === All 164 tests ===

# Unit tests — random order (seed=42, 120 tests)
python contrib/batch_scan/tests/tests-pro/random_numbered.py

# Pool wiring smoke test (4 checks)
python contrib/batch_scan/tests/test_pool_wiring.py

# Monkey-patch invasiveness (14 tests)
python contrib/batch_scan/tests/test_monkeypatch_invasiveness.py

# Monkey-patch fragility (26 tests)
python contrib/batch_scan/tests/test_monkeypatch_fragility.py

# === Convenience ===

# All review-themed tests in one command
python -m unittest \
  contrib.batch_scan.tests.test_monkeypatch_invasiveness \
  contrib.batch_scan.tests.test_monkeypatch_fragility -v
python contrib/batch_scan/tests/test_pool_wiring.py

# Mutation test — 30 injected bugs across 4 risk areas
python contrib/batch_scan/tests/tests-pro/mutation_max.py

# Sequential pytest (if pytest installed)
pytest contrib/batch_scan/tests/tests-pro/ -v
```

## For PR Reviewers

> Since last review: pool is now fully wired (dual-patch closes `from-import` bypass),
> 44 new thematic tests answer Issues #1–#2 directly, and all 164 tests pass
> against upstream NVIDIA/SkillSpector@ab0431f (130+ commits, zero patch conflicts).

### What changed in production code (1 file)

[`runner.py#L70-L91`](../runner.py#L70-L91) — `set_api_pool()` now patches **both**
`llm_utils.get_chat_model` **and** `llm_analyzer_base.get_chat_model`.  Previously only
the former was patched; `llm_analyzer_base`'s `from ... import` created a local
reference that bypassed the pool entirely.  Graph analyzers (95% of LLM calls)
now go through `PooledChatModel`.  `set_api_pool(None)` restores both modules.

### How each review concern was addressed

| Issue | Answer | Proof |
|-------|--------|-------|
| **#1 — Pool dead code** | `set_api_pool()` dual-patch | `test_pool_wiring.py`: 3 paths verified → PooledChatModel |
| **#2 — Patches invasive** | Context manager + explicit `setup_deepseek_compat()` | `test_monkeypatch_invasiveness.py`: 14 tests — import isolation, thread isolation, 50-instance concurrency |
| **#2 — Patches fragile** | `_verify_patch_targets()` guard before apply | `test_monkeypatch_fragility.py`: 26 tests — each of 7 patches individually verified, deep deps checked, atomicity proven |
| **#3 — Risky code untested** | 120 unit tests across 4 risk areas | `tests/tests-pro/` — pool (45), gap-fill (41), patches (24), annotation (10) |

Full response with before/after tables: [`REVIEW_RESPONSE.md`](REVIEW_RESPONSE.md)

### Test suite at a glance (164 total)

```
tests/
├── test_pool_wiring.py               ← Issue #1: 4 smoke checks
├── test_monkeypatch_invasiveness.py   ← Issue #2: 14 tests (thread isolation)
├── test_monkeypatch_fragility.py     ← Issue #2: 26 tests (guard verification)
├── tests-pro/
│   ├── test_api_pool.py              ← Issue #3: 45 tests (acquire/backoff)
│   ├── test_gap_fill.py              ← Issue #3: 41 tests (JSON parsing)
│   ├── test_runner_patches.py        ← Issue #3: 24 tests (context manager)
│   └── test_annotation.py            ← Issue #3: 10 tests (language compat)
└── docs/
    ├── TEST_DESIGN.md                ← WHY each suite was designed
    ├── TEST_GUIDE.md                 ← WHAT each file covers (run commands)
    └── BUGS_FOUND.md                 ← 16 bugs found, 3 test bugs fixed
```

### Design context
- [`DESIGN.md`](DESIGN.md) — architecture, concurrency model, dual-patch mechanism
- [`archive/PITFALLS.md`](archive/PITFALLS.md) — thread safety, `from-import` pitfall, DeepSeek constraints
- [`archive/FUTURE_WORK.md`](archive/FUTURE_WORK.md) — future direction + code conventions

---

**Next:** [DESIGN.md](DESIGN.md) — architecture & concurrency model · [REVIEW_RESPONSE.md](REVIEW_RESPONSE.md) — PR #100 review response · [CONTRIBUTING.md](../CONTRIBUTING.md) — dev setup & code conventions
