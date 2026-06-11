# Detect Whitespace Padding Injection (P9)

Implements GitHub issue #20 — "Detect large whitespace padding used to hide prompt-injection instructions from review".

## Overview

- Attackers can pad a skill file (e.g. `SKILL.md`) with a large block of whitespace — dozens of blank lines, or a long horizontal run of spaces — so that injected instructions sit below or to the right of anything a human reviewer sees, while the agent reads the whole file and acts on them. The text-file equivalent of white-on-white text in a PDF.
- Existing patterns miss this: `P2` (Hidden Instructions) keys off zero-width chars/comments/base64, `TP2` (Unicode Deception) off homoglyphs/RTL, and `MP2` (Context Window Stuffing) off character *repetition* — its regex anchors on `\S`, so runs of blank lines slip through.
- This plan adds a new combined rule **P9 "Whitespace Padding"** (category: Prompt Injection) covering three signals, plus the same detection over MCP manifest description fields.
- **Critical requirement from the issue:** "whitespace" must mean Unicode whitespace (categories `Zs`, `Zl`, `Zp`, control chars `\t \n \r \v \f`, and the zero-width family U+200B/U+200C/U+200D/U+2060/U+FEFF) — not just ASCII space/tab. The zero-width set must be **one shared definition with P2**, not a drifting copy.

### Decisions made during planning

| Question (from issue #20) | Decision |
|---|---|
| Placement | `P`-series, inside `static_patterns_prompt_injection.py` next to P2 |
| Rule id | **P9** — one combined id for all three signals (P6–P8 are already taken by System Prompt Leakage); per-signal confidence carries the weighting |
| Severity | Vertical/horizontal: MEDIUM (HIGH when non-blank content follows a very large vertical gap); ratio: LOW |
| MCP manifest fields | **In scope** — wire the shared detector into `mcp_tool_poisoning.py` description checks, reporting the same P9 id |
| Testing approach | Regular (code first, then tests, within each task) |

## Context (from discovery)

- Files/components involved:
  - `src/skillspector/nodes/analyzers/static_patterns_prompt_injection.py` — P1–P4 live here; P9 joins them (module is already registered as analyzer node `static_patterns_prompt_injection`, so **no `ANALYZER_NODE_IDS`/`ANALYZER_NODES` changes needed**)
  - `src/skillspector/nodes/analyzers/mcp_tool_poisoning.py` — `_extract_metadata_texts()` (line ~78) yields `(text, source_field, is_identifier)` tuples; `node()` (line ~807) dispatches to `_check_tp1/2/3/4`; has its own `_ZERO_WIDTH_RE` (line ~134) that should converge on the shared definition
  - `src/skillspector/nodes/analyzers/pattern_defaults.py` — needs P9 entries in `DEFAULT_EXPLANATIONS`, `RULE_ID_TO_CATEGORY`, `PATTERN_NAMES`, `DEFAULT_REMEDIATIONS`
  - `src/skillspector/nodes/analyzers/common.py` — shared helpers (`get_line_number`, `get_context`)
  - `README.md` — Prompt Injection pattern table (5 → 6 patterns)
- Related patterns found: pattern modules expose `analyze(content, file_path, file_type) -> list[AnalyzerFinding]`; findings carry `rule_id`, `message`, `severity`, `location`, `confidence`, `tags`, `context`, `matched_text` (`models.py:46-62`). `static_runner.run_static_patterns` already skips eval datasets and files > 1 MB.
- Dependencies identified: none new — stdlib `unicodedata` for category classification.

## Development Approach

- **Testing approach**: Regular (implement, then tests, within each task)
- Complete each task fully before moving to the next
- Make small, focused changes
- **CRITICAL: every task MUST include new/updated tests** for code changes in that task
  - Tests are not optional — they are a required part of the checklist
  - Unit tests for new and modified functions; success and error/edge scenarios
- **CRITICAL: all tests must pass before starting next task** — `make test-unit`
- **CRITICAL: update this plan file when scope changes during implementation**
- Maintain backward compatibility — P2/TP1 behavior must not change except via the shared zero-width definition (which is character-for-character identical to today's sets)

## Testing Strategy

- **Unit tests**: required for every task (see above). Pattern tests live in `tests/nodes/analyzers/test_static_patterns.py` (analyzer-level) and `tests/unit/test_patterns_new.py`; MCP tests in `tests/test_mcp_tool_poisoning.py`.
- No UI, no e2e suite — `make test-unit` is the gate; `make test` (incl. integration) at the end.

## Progress Tracking

- Mark completed items with `[x]` immediately when done
- Add newly discovered tasks with ➕ prefix
- Document issues/blockers with ⚠️ prefix
- Update plan if implementation deviates from original scope

## Solution Overview

A new pure-function helper module `whitespace_padding.py` owns the Unicode whitespace character sets and a `detect_whitespace_padding()` scanner that returns structured "padding run" records. Two consumers build findings from those records:

1. `static_patterns_prompt_injection.analyze()` — emits `AnalyzerFinding(rule_id="P9", ...)` for file bodies (all text files the runner feeds it).
2. `mcp_tool_poisoning.node()` — a new `_check_p9_padding(text, source_field)` emits `Finding(rule_id="P9", ...)` for tool/parameter description fields (horizontal + ratio signals only; vertical gaps are meaningless in single-field descriptions at manifest granularity but blank-line runs inside a description still count as a contiguous-block signal).

`P2_PATTERNS`' zero-width character class is rebuilt from the shared `ZERO_WIDTH_CHARS` constant so the two patterns cannot drift (issue requirement). `mcp_tool_poisoning._ZERO_WIDTH_RE` likewise.

### The three signals (rule id P9 for all)

| # | Signal | Trigger | Severity | Confidence |
|---|---|---|---|---|
| 1 | Vertical blank-line run | ≥ 20 consecutive blank/whitespace-only lines | MEDIUM; **HIGH** when non-blank content follows a gap ≥ 40 lines | 0.8 when non-blank content follows the gap; 0.6 when the gap trails the file |
| 2 | Horizontal whitespace run | ≥ 80 consecutive whitespace chars within a line (incl. leading indentation); fires on the run itself regardless of what follows on the line | MEDIUM | 0.7 |
| 3 | Oversized whitespace ratio | a single contiguous whitespace block > 2 KB, **or** whitespace > 90% of a file that is > 4 KB | LOW | 0.4 |

Thresholds are module-level named constants (`VERTICAL_BLANK_LINES = 20`, `VERTICAL_HIGH_SEVERITY_LINES = 40`, `HORIZONTAL_RUN_CHARS = 80`, `BLOCK_BYTE_BUDGET = 2048`, `RATIO_THRESHOLD = 0.90`, `RATIO_MIN_FILE_BYTES = 4096`) so tuning is a one-line change.

### Whitespace classification (shared definitions)

```python
# whitespace_padding.py
ZERO_WIDTH_CHARS = frozenset("​‌‍⁠﻿")  # shared with P2 and mcp_tool_poisoning

def is_padding_char(ch: str) -> bool:
    # True for: ASCII controls \t \n \r \v \f; Unicode categories Zs, Zl, Zp
    # (covers U+00A0, U+2028, U+2029, U+3000, etc.); and ZERO_WIDTH_CHARS.
```

Covers every evasion candidate enumerated in the issue: U+00A0, U+2028, U+2029, U+000C, U+000B, U+3000, U+200B/C/D, U+2060, U+FEFF.

### Detector output

```python
@dataclass
class PaddingRun:
    kind: str            # "vertical" | "horizontal" | "block" | "ratio"
    start_offset: int    # char offset where the run starts
    start_line: int      # 1-based
    length: int          # chars (or lines for "vertical")
    followed_by_content: bool
    summary: str         # visible-ized snippet, e.g. "U+00A0 x82" or "\\n x82"
```

`summarize_run()` renders the run as counts of `U+XXXX xN` segments (collapsing mixed runs to the top few char codes) so the reviewer can *see* what was hidden — per the issue's reporting requirement.

### False-positive guards

- **Fenced code blocks**: the horizontal signal skips runs whose line falls inside a Markdown ``` fence region (line-based fence toggle scan; only applied when `file_type == "markdown"`). Vertical and ratio signals are unaffected (20 blank lines inside a fence is still suspicious).
- **Generated/vendored files**: skip detection entirely for filenames matching `*.min.js`, `*.min.css`, `*.lock`, `package-lock.json`, `yarn.lock`, `*.svg`, `*.map`.
- **Binary-ish content**: skip when content contains U+FFFD (the `errors="replace"` marker from `build_context`) — the repo has no other binary classification.
- **Ratio signal stays LOW/0.4** so it informs rather than dominates the score.
- Eval-dataset prose and > 1 MB files are already skipped upstream by `static_runner`.

## Technical Details

- **Processing flow (file bodies):** `static_runner` → `static_patterns_prompt_injection.analyze()` → after the P4 loop, call `detect_whitespace_padding(content, file_path=..., file_type=...)` → map each `PaddingRun` to an `AnalyzerFinding` with `rule_id="P9"`, `message="Whitespace Padding"`, `tags=[PatternCategory.PROMPT_INJECTION.value]`, `matched_text=run.summary`, `context=get_context(content, run.start_offset)`. P9 runs for **all** file types fed by the runner (unlike P2's markdown/other restriction), minus the vendored/binary guards above — padding in a `.py` or `.txt` body is the same attack.
- **Processing flow (MCP manifests):** in `mcp_tool_poisoning.node()`, alongside the `_check_tp1/_check_tp2` dispatch over `_extract_metadata_texts()`, call `_check_p9_padding(text, source_field)` for non-identifier fields. Manifest fields use the same `HORIZONTAL_RUN_CHARS` and `BLOCK_BYTE_BUDGET` thresholds for the first cut; the per-file ratio signal is skipped (fields are too short for a 4 KB floor to ever apply). Findings use the module's existing `Finding` construction style with `rule_id="P9"` and the MCP `_FRAMEWORK_TAGS`.
- **Dedup within a file:** signals 1/2 report each distinct run; signal 3 ("block"/"ratio") reports at most one finding per file. A vertical run that also exceeds the 2 KB block budget reports only the vertical finding (higher-signal id wins; suppress the block record when its span equals a vertical run's span).
- **Line/offset reporting:** finding points at the line where the padding **starts** (`get_line_number(content, run.start_offset)`); for a newline-free horizontal run the same line number plus the summary's char codes locate it.
- **Shared zero-width definition:** `P2_PATTERNS[2]` regex is built as `"[" + "".join(ZERO_WIDTH_CHARS) + "]"` (import from `whitespace_padding`); `mcp_tool_poisoning._ZERO_WIDTH_RE` rebuilt from the same constant (note: its current set lacks U+2060/U+FEFF — converging is a strict coverage improvement, and U+2060 is also independently flagged by TP2's invisible-chars check, which is fine: different rule, different meaning).

## What Goes Where

- **Implementation Steps** (`[ ]` checkboxes): code, tests, README/docs — all in this repo.
- **Post-Completion** (no checkboxes): threshold tuning against real-world skill corpora, issue/PR follow-ups.

## Implementation Steps

### Task 1: Whitespace padding detector helper module

**Files:**
- Create: `src/skillspector/nodes/analyzers/whitespace_padding.py`
- Create: `tests/nodes/analyzers/test_whitespace_padding.py`

- [x] create `whitespace_padding.py` with `ZERO_WIDTH_CHARS`, `is_padding_char()` (unicodedata category `Z*` + controls + zero-width), threshold constants, `PaddingRun` dataclass, and `summarize_run()` visible-izer (`U+00A0 x82` / `\n x82` rendering)
- [x] implement `detect_whitespace_padding(content, *, file_type="other") -> list[PaddingRun]` covering all three signals: vertical blank-line runs (with `followed_by_content`), horizontal in-line runs, contiguous block > 2 KB and > 90%-of-file ratio
- [x] implement false-positive guards: Markdown fence-region skip for the horizontal signal, U+FFFD (binary-ish) bail-out, vertical-run/block dedup
- [x] write tests: each signal fires at its threshold and not below it (19 blank lines no, 20 yes; 79 ws chars no, 80 yes; 2 KB block boundary)
- [x] write tests: Unicode evasion cases — padding made of U+00A0, U+2028/U+2029, U+000B/U+000C, U+3000, and zero-width chars all detected; `summarize_run` renders `U+00A0 x82`-style output
- [x] write tests: guards — horizontal run inside a ``` fence not reported for markdown, reported for non-markdown; content with U+FFFD returns no runs; `followed_by_content` true/false distinguished
- [x] run `make test-unit` — must pass before task 2

### Task 2: P9 findings in the prompt-injection analyzer + shared zero-width definition

**Files:**
- Modify: `src/skillspector/nodes/analyzers/static_patterns_prompt_injection.py`
- Modify: `src/skillspector/nodes/analyzers/pattern_defaults.py`
- Modify: `tests/nodes/analyzers/test_static_patterns.py`

- [x] in `static_patterns_prompt_injection.py`: import the detector, add a P9 block in `analyze()` mapping `PaddingRun` → `AnalyzerFinding` (severity/confidence per the signal table above; `matched_text=run.summary`); add the vendored-filename skip; update module docstring "(P1–P4)" → "(P1–P4, P9)"
- [x] rebuild `P2_PATTERNS`' zero-width regex from the shared `ZERO_WIDTH_CHARS` constant (no behavior change — same five chars)
- [x] add P9 to `pattern_defaults.py`: `DEFAULT_EXPLANATIONS`, `RULE_ID_TO_CATEGORY` (→ `PatternCategory.PROMPT_INJECTION`), `PATTERN_NAMES` ("Whitespace Padding"), `DEFAULT_REMEDIATIONS`
- [x] write tests in `test_static_patterns.py`: a SKILL.md body with 80 blank lines then an injected instruction yields a P9 finding with HIGH severity, correct `start_line` (start of the gap), and a visible-ized `matched_text`; trailing-gap variant yields MEDIUM/0.6; horizontal and ratio variants yield their severities; `*.min.js` path yields no P9
- [x] write test: existing P2 zero-width detection still fires identically after the shared-constant refactor
- [x] run `make test-unit` — must pass before task 3 (ran `uv run pytest -m "not integration" tests/` — 634 passed)

### Task 3: P9 over MCP manifest description fields

**Files:**
- Modify: `src/skillspector/nodes/analyzers/mcp_tool_poisoning.py`
- Modify: `tests/test_mcp_tool_poisoning.py`

- [x] add `_check_p9_padding(text, source_field) -> list[Finding]` using the shared detector (horizontal + contiguous-block signals; skip per-file ratio); wire it into `node()`'s metadata-text loop for non-identifier fields
- [x] rebuild `_ZERO_WIDTH_RE` from the shared `ZERO_WIDTH_CHARS` constant (adds U+2060/U+FEFF coverage to TP1's hidden-text check — strict improvement; note it in the docstring)
- [x] write tests: a tool description padded with 100 spaces before an instruction yields a P9 finding naming the source field; a normal multi-sentence description yields none; identifier fields are not scanned
- [x] write test: TP1 zero-width behavior unchanged for the original three chars, and now also fires on U+2060/U+FEFF
- [x] run `make test-unit` — must pass before task 4 (ran `uv run pytest -m "not integration" tests/` — 641 passed)

### Task 4: Verify acceptance criteria

- [x] verify all issue #20 requirements: three signals implemented, Unicode-category-based classification, shared zero-width definition with P2, visible-ized snippets, line/offset reporting, fenced-code + vendored-file + ratio-confidence FP guards, MCP manifest coverage — all PASS, no gaps found (see progress log)
- [x] adversarial self-check: craft a SKILL.md using each padding char from the issue's evasion list (U+00A0, U+2028, U+2029, U+000C, U+000B, U+3000, zero-width family) and confirm P9 fires on every one via the test suite — added `TestIssue20AdversarialEvasionCoverage` (33 parametrized cases, all 11 chars x inline/vertical/analyzer); all pass
- [x] run full suite: `make test` (unit + integration) — unit gate `uv run pytest -m "not integration" tests/` = 674 passed, 11 skipped (optional NVIDIA OSS providers); integration 12 failed only due to missing LLM API key (no network/creds), unrelated to our changes
- [x] run a real scan over a fixture skill (`uv run skillspector scan <fixture>` or project equivalent) and eyeball the P9 finding rendering in the report output — `uv run skillspector scan /tmp/p9-fixture --no-llm` rendered "HIGH: P9 - Whitespace Padding" at SKILL.md:10, confidence 80%, matched_text `\n x80` (see progress log)

### Task 5: [Final] Update documentation

- [x] add P9 row to the README Prompt Injection table and bump its "(5 patterns)" count and the total pattern count (5→6 patterns; total 64→65 in both the Features bullet and the Vulnerability Patterns intro)
- [x] CLAUDE.md — no suitable section / not applicable (no CLAUDE.md exists in the repo; nothing to update)
- [x] comment on issue #20 — drafted at docs/plans/p9-issue20-comment-draft.md (NOT posted; deferred to user)
- [x] plan move deferred to harness (post-run)

## Post-Completion

**Manual verification:**
- Tune thresholds against a corpus of real, benign skills (templates with spacer lines, ASCII-art-heavy READMEs) before relying on the ratio signal at anything above LOW confidence; issue #20 explicitly expects tuning.
- Security review of the regex/scan performance on pathological inputs (the detector is a linear scan, no backtracking regex, but confirm on a 1 MB all-whitespace file).

**External follow-ups:**
- Open PR against `NVIDIA/SkillSpector` referencing issue #20; the open questions answered here (single P9 id, MEDIUM/MEDIUM/LOW severities, MCP manifests in scope) should be re-stated in the PR description for maintainer sign-off, since the issue author offered to align on signals/thresholds before a PR.
