# Design: Move PR #8 schema-pruning into an isolated `exaforce` monkeypatch

- **Date:** 2026-07-10
- **Branch:** `prune-unused-keys-in-prompt-schemas` (PR #8)
- **Author:** wbeasley

## Goal

Reproduce the exact runtime behavior of PR #8 — pruning rarely-used keys out of
the LLM structured-output schemas and prompts to shrink requests and reduce LLM
timeouts — **without editing upstream source files**. The forked change must live
in the fork-only `src/skillspector/exaforce/` package so upstream syncs
(`NVIDIA/SkillSpector`) merge cleanly.

## What PR #8 changes (the behavior to reproduce)

| Upstream file | Runtime effect to reproduce |
|---|---|
| `llm_analyzer_base.py` | Remove `explanation`, `remediation` fields from `LLMFinding`; stop passing them in `LLMFinding.to_finding()` |
| `nodes/meta_analyzer.py` | Remove `intent`, `impact` fields from `MetaAnalyzerFinding`; remove `overall_assessment` field + its `_parse_stringified_assessment` validator from `MetaAnalyzerResult` (the `OverallAssessment` class becomes unused/unreferenced); trim the `PER_FILE_ANALYSIS_PROMPT` task list from 4 items to 2 (drop the intent + impact questions) |
| `nodes/analyzers/semantic_developer_intent.py` | Remove the sentence `"Reference the L-prefixed line numbers when reporting findings."` from `ANALYZER_PROMPT` |
| `nodes/analyzers/semantic_quality_policy.py` | Same sentence removal from `ANALYZER_PROMPT` |

## Approach: in-place monkeypatch, guarded, activated once at import

### Why in-place mutation (not class replacement)

Mutating the existing model classes in place preserves object identity, so:
- `LLMAnalyzerBase.response_schema` / `LLMMetaAnalyzer.response_schema` keep
  pointing at the same (now-pruned) class objects — no rebinding needed.
- `isinstance(response, LLMAnalysisResult)` in `parse_response` still holds.
- Nothing that imported these classes earlier ends up with a stale reference.

Verified behavior of the mutation recipe (Pydantic v2):
1. `del Model.model_fields[name]` for each removed field.
2. Pop any dangling `field_validator` decorator for a removed field via
   `Model.__pydantic_decorators__.field_validators.pop(<name>, None)`.
3. `Model.model_rebuild(force=True)` on the leaf model **and** on every container
   that nests it (e.g. rebuild `LLMFinding` then `LLMAnalysisResult`).

Confirmed: after this, removed keys disappear from `model_json_schema()`
(including nested `$defs`), from validation, and from `model_dump()`.

### Prompts

Prompt bodies are read from module globals at `node()` / analyzer `__init__`
time, so reassigning the module global before those run takes effect. Use a
**guarded targeted `str.replace()`** (assert the target substring is present,
then replace) rather than copying the full prompt text — this avoids duplicating
~100 lines and detects upstream drift.

### Guards (the key safety property)

A source-file fork *conflicts loudly* on upstream sync; a monkeypatch instead
*silently goes stale* if upstream renames a field or rewrites a prompt. To
recover the "loud failure" signal:

- Before removing a field, assert it is present in `model_fields`.
- Before replacing a prompt substring, assert the substring is present.
- On any failed assertion, raise a clear error naming the drifted symbol.

This turns silent divergence into a fail-fast at process startup.

### Idempotency

`apply_patches()` is guarded by a module-level `_PATCHED` flag. The first call
performs all mutations (and runs the drift guards); subsequent calls are no-ops.
This makes repeated imports / explicit test calls safe.

## Module layout

```
src/skillspector/exaforce/
  __init__.py          # exposes apply_patches(); holds _PATCHED flag
  _schema_patches.py   # LLMFinding / MetaAnalyzerFinding / MetaAnalyzerResult field pruning + to_finding
  _prompt_patches.py   # the three prompt-string edits
```
(Final file split can collapse to fewer files if trivial; boundary is
schema-patches vs prompt-patches.)

## Activation

Append to the end of `src/skillspector/__init__.py` (after the existing
`from skillspector.graph import ...` at line 35, so target modules are loaded):

```python
from skillspector import exaforce as _exaforce  # noqa: E402
_exaforce.apply_patches()
```

This is the **only** upstream-tracked file edited (one idempotent import +
call). It covers CLI, MCP server, the benchmark, and the test suite uniformly,
because any `import skillspector` runs it.

## Source files: restore to upstream parity

Revert these 4 files to their pre-PR-#8 (upstream) content, since their pruning
now happens at runtime via `apply_patches()`:
- `src/skillspector/llm_analyzer_base.py`
- `src/skillspector/nodes/meta_analyzer.py`
- `src/skillspector/nodes/analyzers/semantic_developer_intent.py`
- `src/skillspector/nodes/analyzers/semantic_quality_policy.py`

## Tests

- **Keep** PR #8's edits to `tests/nodes/test_llm_analyzer_base.py` and
  `tests/nodes/test_semantic_quality_policy.py`. Because `__init__.py` patches
  globally on import, the test process sees pruned behavior, so these
  pruned-expecting assertions pass. (Reverting them would make them fail against
  the patched runtime.) These two test files remain the only forked test
  artifacts.
- **Add** `tests/exaforce/test_patches.py` (fork-only) that:
  - asserts `apply_patches()` removed the expected keys from each model's JSON
    schema and from `model_dump()`;
  - asserts the three prompt strings no longer contain the removed sentences;
  - asserts idempotency (calling `apply_patches()` twice is safe);
  - asserts the drift guards raise when a target field/substring is absent
    (simulate by monkeypatching a target away, then expecting the guard error).

## Verification

- `uv run pytest` green (full suite, incl. kept pruned-expecting tests + new
  exaforce tests).
- A one-off runtime check: import `skillspector`, then dump
  `LLMAnalysisResult.model_json_schema()` and `MetaAnalyzerResult.model_json_schema()`
  and confirm the pruned keys are absent and the prompts lack the removed
  sentences — i.e. behavior identical to the PR #8 diff.
- `git diff upstream/main -- src/skillspector/llm_analyzer_base.py src/skillspector/nodes/meta_analyzer.py src/skillspector/nodes/analyzers/semantic_developer_intent.py src/skillspector/nodes/analyzers/semantic_quality_policy.py`
  is empty (source files at parity).

## Out of scope / non-goals

- No plugin system, no config flag to toggle pruning — pruning is always on
  (it is the desired product behavior).
- No reversible/per-test-scoped patching machinery; patches are process-global,
  matching the fact that the product always wants the pruned schema.

## Risks

1. **Silent drift** if upstream changes a targeted field/prompt — mitigated by
   fail-fast guards (raise at startup, don't silently no-op).
2. **Pydantic internal reliance** (`model_fields`, `__pydantic_decorators__`,
   `model_rebuild`) could break on a major Pydantic upgrade — mitigated by the
   exaforce test suite catching it in CI.
3. **Two test files remain forked** — accepted; their assertions track real
   (pruned) behavior and conflict far less than the source files did.
