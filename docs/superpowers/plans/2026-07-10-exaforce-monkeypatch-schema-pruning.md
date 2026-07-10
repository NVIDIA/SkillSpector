# ExaForce Monkeypatch Schema-Pruning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce PR #8's LLM schema/prompt pruning at runtime from the fork-only `src/skillspector/exaforce/` package, and restore every upstream-tracked file it edited to upstream parity.

**Architecture:** A guarded monkeypatch. On `import skillspector`, `exaforce.apply_patches()` mutates the existing Pydantic model classes in place (`del model_fields[...]` + `model_rebuild(force=True)` on leaf and container, preserving class identity) and rewrites three prompt module-globals via targeted `str.replace()`. Every mutation asserts its upstream target still exists and raises `PatchDriftError` if not, so an upstream rename fails loudly instead of silently going stale.

**Tech Stack:** Python 3.x, Pydantic v2, pytest, `uv` for running everything.

## Global Constraints

- Only `src/skillspector/exaforce/**` and **one** appended block in `src/skillspector/__init__.py` may diverge from upstream. Every other upstream-tracked file (source and tests) must end at upstream (pre-PR-#8) parity. Pre-PR-#8 tree = commit `bec96da` (`326848f^`).
- All guards must **raise** (`PatchDriftError`) on drift — never silently no-op a mutation.
- `apply_patches()` must be idempotent (safe to call repeatedly).
- Do **not** touch unrelated working-tree changes already present (`benchmark/**`, `.claude/**`, `.superpowers/**`, `docs/superpowers/**`, `*.db`, `*_report.md`, etc.). Stage only the exact files each task names.
- Run all commands via `uv run`. Test suite command: `uv run pytest`.
- The scanned skill content is adversarial input; do not change any security invariant — only reproduce PR #8's key/prompt pruning.

---

### Task 1: Revert PR #8 files to upstream parity

Undo PR #8's edits so the four source files are un-pruned again (the patch in later tasks needs those fields/prompts present to remove), and the two test files return to upstream. After this task the tree equals pre-PR-#8 and the suite is green.

**Files:**
- Modify (revert): `src/skillspector/llm_analyzer_base.py`
- Modify (revert): `src/skillspector/nodes/meta_analyzer.py`
- Modify (revert): `src/skillspector/nodes/analyzers/semantic_developer_intent.py`
- Modify (revert): `src/skillspector/nodes/analyzers/semantic_quality_policy.py`
- Modify (revert): `tests/nodes/test_llm_analyzer_base.py`
- Modify (revert): `tests/nodes/test_semantic_quality_policy.py`

**Interfaces:**
- Produces (restored upstream symbols later tasks patch): `llm_analyzer_base.LLMFinding` (with fields `explanation`, `remediation` and a `to_finding` that forwards them), `llm_analyzer_base.LLMAnalysisResult`; `meta_analyzer.MetaAnalyzerFinding` (with `intent`, `impact`), `meta_analyzer.MetaAnalyzerResult` (with `overall_assessment` field + `_parse_stringified_assessment` field_validator), `meta_analyzer.PER_FILE_ANALYSIS_PROMPT`; `semantic_developer_intent.ANALYZER_PROMPT`; `semantic_quality_policy.ANALYZER_PROMPT`.

- [ ] **Step 1: Restore the six files from the pre-PR-#8 commit**

```bash
git checkout 326848f^ -- \
  src/skillspector/llm_analyzer_base.py \
  src/skillspector/nodes/meta_analyzer.py \
  src/skillspector/nodes/analyzers/semantic_developer_intent.py \
  src/skillspector/nodes/analyzers/semantic_quality_policy.py \
  tests/nodes/test_llm_analyzer_base.py \
  tests/nodes/test_semantic_quality_policy.py
```

- [ ] **Step 2: Verify the four source files now contain the upstream (un-pruned) symbols**

Run:
```bash
git diff 326848f^ -- \
  src/skillspector/llm_analyzer_base.py \
  src/skillspector/nodes/meta_analyzer.py \
  src/skillspector/nodes/analyzers/semantic_developer_intent.py \
  src/skillspector/nodes/analyzers/semantic_quality_policy.py \
  tests/nodes/test_llm_analyzer_base.py \
  tests/nodes/test_semantic_quality_policy.py
```
Expected: **no output** (working tree matches pre-PR-#8 for these files).

Also confirm the fields are back:
```bash
grep -n "explanation\|remediation" src/skillspector/llm_analyzer_base.py
grep -n "intent\|impact\|overall_assessment\|OverallAssessment" src/skillspector/nodes/meta_analyzer.py
```
Expected: matches present (fields restored).

- [ ] **Step 3: Run the suite to confirm the reverted baseline is green**

Run: `uv run pytest -q`
Expected: PASS (source and tests are both un-pruned, so consistent). Note the passing count for comparison in Task 4.

- [ ] **Step 4: Commit**

```bash
git add \
  src/skillspector/llm_analyzer_base.py \
  src/skillspector/nodes/meta_analyzer.py \
  src/skillspector/nodes/analyzers/semantic_developer_intent.py \
  src/skillspector/nodes/analyzers/semantic_quality_policy.py \
  tests/nodes/test_llm_analyzer_base.py \
  tests/nodes/test_semantic_quality_policy.py
git commit -m "revert: restore PR #8 files to upstream parity

Pruning moves to the fork-only exaforce runtime patch (subsequent tasks)."
```

---

### Task 2: exaforce guard primitives + unit tests

Add the small, dependency-free patch primitives used by every later patch, with the drift guards. No real models are touched yet — these are tested against throwaway models.

**Files:**
- Create: `src/skillspector/exaforce/_patchlib.py`
- Create: `tests/exaforce/__init__.py` (empty)
- Test: `tests/exaforce/test_patchlib.py`

Note: `src/skillspector/exaforce/__init__.py` already exists (empty); leave it for Task 3.

**Interfaces:**
- Produces:
  - `class PatchDriftError(RuntimeError)`
  - `remove_model_fields(model: type[BaseModel], field_names: list[str]) -> None` — deletes each name from `model.model_fields`; raises `PatchDriftError` if a name is absent. Caller must `model_rebuild(force=True)` afterward.
  - `pop_field_validator(model: type[BaseModel], validator_name: str) -> None` — removes a field_validator decorator by function name; no-op if absent.
  - `replace_module_str(module: ModuleType, attr: str, old: str, new: str) -> None` — sets `module.attr = module.attr.replace(old, new)`; raises `PatchDriftError` if `old` not present.

- [ ] **Step 1: Write the failing tests**

Create `tests/exaforce/__init__.py` (empty file), then `tests/exaforce/test_patchlib.py`:

```python
from types import ModuleType

import pytest
from pydantic import BaseModel

from skillspector.exaforce._patchlib import (
    PatchDriftError,
    pop_field_validator,
    remove_model_fields,
    replace_module_str,
)


def test_remove_model_fields_removes_then_rebuild_drops_key():
    class M(BaseModel):
        a: str
        b: str = ""

    remove_model_fields(M, ["b"])
    M.model_rebuild(force=True)
    assert "b" not in M.model_json_schema()["properties"]
    assert "a" in M.model_json_schema()["properties"]


def test_remove_model_fields_raises_on_missing_field():
    class M(BaseModel):
        a: str

    with pytest.raises(PatchDriftError):
        remove_model_fields(M, ["nope"])


def test_pop_field_validator_is_noop_when_absent():
    class M(BaseModel):
        a: str

    pop_field_validator(M, "nonexistent")  # must not raise


def test_replace_module_str_replaces_substring():
    mod = ModuleType("dummy")
    mod.P = "hello world"
    replace_module_str(mod, "P", "world", "there")
    assert mod.P == "hello there"


def test_replace_module_str_raises_on_missing_substring():
    mod = ModuleType("dummy")
    mod.P = "hello world"
    with pytest.raises(PatchDriftError):
        replace_module_str(mod, "P", "absent", "x")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/exaforce/test_patchlib.py -q`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` (`_patchlib` does not exist yet).

- [ ] **Step 3: Implement `_patchlib.py`**

Create `src/skillspector/exaforce/_patchlib.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Guarded monkeypatch primitives for the ExaForce fork.

Each primitive asserts its upstream target still exists before mutating it, so
an upstream rename/rewrite fails loudly (``PatchDriftError``) at import time
instead of silently leaving the fork's runtime behavior stale.
"""

from __future__ import annotations

from types import ModuleType

from pydantic import BaseModel


class PatchDriftError(RuntimeError):
    """An upstream target a fork patch depends on has changed or is missing."""


def remove_model_fields(model: type[BaseModel], field_names: list[str]) -> None:
    """Delete ``field_names`` from ``model.model_fields``.

    The caller must run ``model.model_rebuild(force=True)`` afterward (and on any
    container model that nests ``model``) for the change to reach the emitted
    JSON schema. Raises ``PatchDriftError`` if a field is absent.
    """
    for name in field_names:
        if name not in model.model_fields:
            raise PatchDriftError(
                f"{model.__module__}.{model.__qualname__} has no field {name!r}; "
                "upstream changed — update the exaforce patch."
            )
        del model.model_fields[name]


def pop_field_validator(model: type[BaseModel], validator_name: str) -> None:
    """Remove a ``field_validator`` decorator by its function name (no-op if absent)."""
    model.__pydantic_decorators__.field_validators.pop(validator_name, None)


def replace_module_str(module: ModuleType, attr: str, old: str, new: str) -> None:
    """Replace substring ``old`` with ``new`` in module-global ``attr``.

    Raises ``PatchDriftError`` if ``old`` is not present in the current value.
    """
    current = getattr(module, attr)
    if old not in current:
        raise PatchDriftError(
            f"{module.__name__}.{attr} does not contain the expected text; "
            "upstream changed — update the exaforce patch."
        )
    setattr(module, attr, current.replace(old, new))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/exaforce/test_patchlib.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/skillspector/exaforce/_patchlib.py tests/exaforce/__init__.py tests/exaforce/test_patchlib.py
git commit -m "feat(exaforce): guarded monkeypatch primitives"
```

---

### Task 3: Schema + prompt patches and `apply_patches()`

Implement the actual pruning patches and the idempotent `apply_patches()` orchestrator. **No activation from `skillspector/__init__.py` yet** — this task tests via a subprocess that imports skillspector and calls `apply_patches()` explicitly, so the in-process suite (and the reverted upstream tests) stay green.

**Files:**
- Create: `src/skillspector/exaforce/_schema_patches.py`
- Create: `src/skillspector/exaforce/_prompt_patches.py`
- Modify: `src/skillspector/exaforce/__init__.py` (currently empty)
- Test: `tests/exaforce/test_patches.py`

**Interfaces:**
- Consumes: `PatchDriftError`, `remove_model_fields`, `pop_field_validator`, `replace_module_str` from `_patchlib` (Task 2); the restored upstream symbols from Task 1.
- Produces:
  - `skillspector.exaforce.apply_patches() -> None` — idempotent; runs `_schema_patches.apply()` then `_prompt_patches.apply()`.
  - `_schema_patches.apply() -> None`, `_prompt_patches.apply() -> None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/exaforce/test_patches.py`:

```python
import subprocess
import sys
import textwrap


def _run(code: str) -> str:
    """Run *code* in a fresh interpreter; return stdout, assert clean exit."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_apply_patches_prunes_model_schemas():
    out = _run(
        """
        import skillspector
        from skillspector.exaforce import apply_patches
        apply_patches()
        from skillspector.llm_analyzer_base import LLMFinding, LLMAnalysisResult
        from skillspector.nodes.meta_analyzer import (
            MetaAnalyzerFinding,
            MetaAnalyzerResult,
        )
        lf = set(LLMFinding.model_json_schema()["properties"])
        assert "explanation" not in lf and "remediation" not in lf, lf
        mf = set(MetaAnalyzerFinding.model_json_schema()["properties"])
        assert "intent" not in mf and "impact" not in mf, mf
        mr = set(MetaAnalyzerResult.model_json_schema()["properties"])
        assert "overall_assessment" not in mr, mr
        # Container schema (what the LLM actually receives) is pruned too:
        nested = LLMAnalysisResult.model_json_schema()["$defs"]["LLMFinding"]["properties"]
        assert "explanation" not in nested, nested
        print("OK")
        """
    )
    assert "OK" in out


def test_apply_patches_prunes_to_finding_and_dump():
    out = _run(
        """
        import skillspector
        from skillspector.exaforce import apply_patches
        apply_patches()
        from skillspector.llm_analyzer_base import LLMFinding
        f = LLMFinding(rule_id="R", message="m", severity="LOW", start_line=3)
        assert "explanation" not in f.model_dump()
        assert "remediation" not in f.model_dump()
        fin = f.to_finding("x.py")
        assert fin.explanation is None
        assert fin.rule_id == "R" and fin.start_line == 3
        print("OK")
        """
    )
    assert "OK" in out


def test_apply_patches_trims_prompts():
    out = _run(
        """
        import skillspector
        from skillspector.exaforce import apply_patches
        apply_patches()
        from skillspector.nodes.analyzers import semantic_developer_intent as d
        from skillspector.nodes.analyzers import semantic_quality_policy as q
        from skillspector.nodes import meta_analyzer as m
        assert "Reference the L-prefixed line numbers" not in d.ANALYZER_PROMPT
        assert "Reference the L-prefixed line numbers" not in q.ANALYZER_PROMPT
        assert "What is the likely intent" not in m.PER_FILE_ANALYSIS_PROMPT
        assert "What is the potential impact" not in m.PER_FILE_ANALYSIS_PROMPT
        assert "Use the rule IDs exactly as listed." in d.ANALYZER_PROMPT
        print("OK")
        """
    )
    assert "OK" in out


def test_apply_patches_is_idempotent():
    out = _run(
        """
        import skillspector
        from skillspector.exaforce import apply_patches
        apply_patches()
        apply_patches()
        apply_patches()
        print("OK")
        """
    )
    assert "OK" in out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/exaforce/test_patches.py -q`
Expected: FAIL — subprocess exits non-zero because `apply_patches` is not importable yet (empty `exaforce/__init__.py`), so `_run` hits its `assert result.returncode == 0`.

- [ ] **Step 3: Implement `_schema_patches.py`**

Create `src/skillspector/exaforce/_schema_patches.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Prune unused keys from the LLM structured-output schemas (fork behavior).

Reproduces the schema pruning that would otherwise live in upstream
``llm_analyzer_base`` / ``meta_analyzer``, keeping those upstream files at
parity with NVIDIA/SkillSpector.
"""

from __future__ import annotations

import skillspector.llm_analyzer_base as llm_base
import skillspector.nodes.meta_analyzer as meta
from skillspector.models import Finding

from ._patchlib import pop_field_validator, remove_model_fields


def _pruned_to_finding(self: "llm_base.LLMFinding", file: str) -> Finding:
    """``LLMFinding.to_finding`` without the removed explanation/remediation."""
    return Finding(
        rule_id=self.rule_id,
        message=self.message,
        severity=self.severity,
        confidence=self.confidence,
        file=file,
        start_line=self.start_line,
        end_line=self.end_line,
    )


def apply() -> None:
    # LLMFinding: drop explanation + remediation, and stop forwarding them.
    remove_model_fields(llm_base.LLMFinding, ["explanation", "remediation"])
    llm_base.LLMFinding.to_finding = _pruned_to_finding
    llm_base.LLMFinding.model_rebuild(force=True)
    llm_base.LLMAnalysisResult.model_rebuild(force=True)

    # MetaAnalyzerFinding: drop intent + impact.
    remove_model_fields(meta.MetaAnalyzerFinding, ["intent", "impact"])
    meta.MetaAnalyzerFinding.model_rebuild(force=True)

    # MetaAnalyzerResult: drop overall_assessment field + its validator.
    remove_model_fields(meta.MetaAnalyzerResult, ["overall_assessment"])
    pop_field_validator(meta.MetaAnalyzerResult, "_parse_stringified_assessment")
    meta.MetaAnalyzerResult.model_rebuild(force=True)
```

- [ ] **Step 4: Implement `_prompt_patches.py`**

Create `src/skillspector/exaforce/_prompt_patches.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Trim prompt text the pruned schema no longer needs (fork behavior)."""

from __future__ import annotations

import skillspector.nodes.analyzers.semantic_developer_intent as dev_intent
import skillspector.nodes.analyzers.semantic_quality_policy as quality
import skillspector.nodes.meta_analyzer as meta

from ._patchlib import replace_module_str

# Present verbatim in both semantic analyzers' ANALYZER_PROMPT (note the two
# spaces after "listed." and the mid-sentence newline).
_LINE_NUMBER_OLD = (
    "Use the rule IDs exactly as listed.  Reference the L-prefixed line numbers\n"
    "when reporting findings."
)
_LINE_NUMBER_NEW = "Use the rule IDs exactly as listed."

# The meta-analyzer's "Your Task" list: drop the intent (2) and impact (3) items.
_META_TASK_OLD = (
    "1. Is this a true vulnerability or a false positive?\n"
    "2. What is the likely intent (malicious, negligent, or benign)?\n"
    "3. What is the potential impact if exploited?\n"
    "4. Does the skill context make this more or less dangerous?"
)
_META_TASK_NEW = (
    "1. Is this a true vulnerability or a false positive?\n"
    "2. Does the skill context make this more or less dangerous?"
)


def apply() -> None:
    replace_module_str(dev_intent, "ANALYZER_PROMPT", _LINE_NUMBER_OLD, _LINE_NUMBER_NEW)
    replace_module_str(quality, "ANALYZER_PROMPT", _LINE_NUMBER_OLD, _LINE_NUMBER_NEW)
    replace_module_str(meta, "PER_FILE_ANALYSIS_PROMPT", _META_TASK_OLD, _META_TASK_NEW)
```

- [ ] **Step 5: Implement `exaforce/__init__.py`**

Overwrite `src/skillspector/exaforce/__init__.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""ExaForce fork-local runtime patches.

Keeps fork behavior — pruning unused LLM structured-output keys and prompt text
to shrink requests and reduce LLM timeouts — out of upstream-tracked source
files. All mutations are guarded: an upstream rename/rewrite raises
``PatchDriftError`` at import time rather than silently going stale.
"""

from __future__ import annotations

from . import _prompt_patches, _schema_patches

_PATCHED = False


def apply_patches() -> None:
    """Idempotently apply all fork-local runtime patches."""
    global _PATCHED
    if _PATCHED:
        return
    _schema_patches.apply()
    _prompt_patches.apply()
    _PATCHED = True
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/exaforce/test_patches.py -q`
Expected: PASS (4 passed).

- [ ] **Step 7: Confirm the in-process suite is still green (no activation yet)**

Run: `uv run pytest -q`
Expected: PASS — same count as Task 1 Step 3 plus the new exaforce tests. The reverted upstream tests still pass because `apply_patches()` has not been wired into `import skillspector` yet (it only runs inside the subprocess tests).

- [ ] **Step 8: Commit**

```bash
git add src/skillspector/exaforce/_schema_patches.py \
        src/skillspector/exaforce/_prompt_patches.py \
        src/skillspector/exaforce/__init__.py \
        tests/exaforce/test_patches.py
git commit -m "feat(exaforce): prune LLM schema keys + prompt text via guarded patch"
```

---

### Task 4: Activate on import + document expected upstream-test failures

Wire `apply_patches()` into `skillspector/__init__.py` so the pruning is the default everywhere (CLI, MCP, benchmark, tests). This makes the runtime match PR #8 exactly. As an accepted consequence, the two reverted upstream test files now fail (they assert the un-pruned schema); capture that exact set.

**Files:**
- Modify: `src/skillspector/__init__.py` (append activation block after `__all__`)
- Create: `tests/exaforce/test_activation.py`
- Create: `docs/superpowers/EXPECTED_TEST_FAILURES.md`

**Interfaces:**
- Consumes: `skillspector.exaforce.apply_patches` (Task 3).
- Produces: bare `import skillspector` yields pruned schemas/prompts.

- [ ] **Step 1: Write the failing activation test**

Create `tests/exaforce/test_activation.py`:

```python
import subprocess
import sys
import textwrap


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_bare_import_activates_patches():
    # No explicit apply_patches() call — importing skillspector must patch.
    out = _run(
        """
        import skillspector
        from skillspector.llm_analyzer_base import LLMFinding
        from skillspector.nodes.meta_analyzer import (
            MetaAnalyzerFinding,
            MetaAnalyzerResult,
        )
        from skillspector.nodes import meta_analyzer as m
        assert "explanation" not in LLMFinding.model_json_schema()["properties"]
        assert "intent" not in MetaAnalyzerFinding.model_json_schema()["properties"]
        assert "overall_assessment" not in MetaAnalyzerResult.model_json_schema()["properties"]
        assert "What is the likely intent" not in m.PER_FILE_ANALYSIS_PROMPT
        print("OK")
        """
    )
    assert "OK" in out
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/exaforce/test_activation.py -q`
Expected: FAIL — the subprocess assertion trips (bare import does not patch yet), so `_run` sees a non-zero exit.

- [ ] **Step 3: Append the activation block to `skillspector/__init__.py`**

Add these two lines at the **end** of `src/skillspector/__init__.py` (after the existing `__all__ = [...]` line — by this point `from skillspector.graph import ...` has already imported the target modules):

```python

# ExaForce fork: apply runtime schema/prompt patches (kept out of upstream files).
from skillspector import exaforce as _exaforce  # noqa: E402
_exaforce.apply_patches()
```

- [ ] **Step 4: Run the activation test to verify it passes**

Run: `uv run pytest tests/exaforce/test_activation.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Capture the expected upstream-test failures**

Run (the `- true` keeps going despite pytest's non-zero exit on failures):
```bash
uv run pytest tests/nodes/test_llm_analyzer_base.py tests/nodes/test_semantic_quality_policy.py -q || true
```
Expected: some FAILED tests in exactly these two files (assertions expecting `explanation`/`remediation`/`intent`/`impact`/`overall_assessment`). Confirm each failure is an **assertion** about a pruned key — **not** an import/collection error (which would indicate a real bug).

- [ ] **Step 6: Record the failure set**

Create `docs/superpowers/EXPECTED_TEST_FAILURES.md` listing the failing node IDs from Step 5, e.g.:

```markdown
# Expected test failures (fork: exaforce schema pruning)

These upstream tests are kept at upstream parity on purpose and therefore
assert the *un-pruned* schema, which the exaforce runtime patch removes. They
are expected to FAIL. A failure here is only a problem if the failure is NOT an
assertion about a pruned key (e.g. an import/collection error).

<!-- paste the exact FAILED node IDs from:
     uv run pytest tests/nodes/test_llm_analyzer_base.py tests/nodes/test_semantic_quality_policy.py -q -->
- tests/nodes/test_llm_analyzer_base.py::...
- tests/nodes/test_semantic_quality_policy.py::...
```

- [ ] **Step 7: Confirm the failure set is bounded to those two files**

Run: `uv run pytest -q -rf || true`
Expected: the only FAILED tests are inside `tests/nodes/test_llm_analyzer_base.py` and `tests/nodes/test_semantic_quality_policy.py`. The `tests/exaforce/**` suite passes. If any *other* file fails, investigate before finishing (it means the monkeypatch changed behavior beyond PR #8's scope).

- [ ] **Step 8: Confirm upstream-file parity (the whole point of the fork)**

Run:
```bash
git diff 326848f^ -- \
  src/skillspector/llm_analyzer_base.py \
  src/skillspector/nodes/meta_analyzer.py \
  src/skillspector/nodes/analyzers/semantic_developer_intent.py \
  src/skillspector/nodes/analyzers/semantic_quality_policy.py \
  tests/nodes/test_llm_analyzer_base.py \
  tests/nodes/test_semantic_quality_policy.py
```
Expected: **no output**. And `git diff 326848f^ -- src/skillspector/__init__.py` shows only the two-line activation block added.

- [ ] **Step 9: Commit**

```bash
git add src/skillspector/__init__.py tests/exaforce/test_activation.py docs/superpowers/EXPECTED_TEST_FAILURES.md
git commit -m "feat(exaforce): activate schema-pruning patch on import

Runtime now matches PR #8. The two reverted upstream test files assert the
un-pruned schema and fail by design (see docs/superpowers/EXPECTED_TEST_FAILURES.md)."
```

---

## Post-plan verification

- [ ] `uv run pytest tests/exaforce -q` → all pass.
- [ ] `uv run pytest -q -rf` → the only failures are in the two reverted upstream test files, matching `docs/superpowers/EXPECTED_TEST_FAILURES.md`.
- [ ] Behavioral equivalence to PR #8: in a fresh interpreter, `import skillspector` then dumping `LLMAnalysisResult.model_json_schema()` and `MetaAnalyzerResult.model_json_schema()` shows the same pruned key set PR #8 produced, and the three prompts lack the removed sentences.
- [ ] `git diff 326848f^` touches only `src/skillspector/exaforce/**`, the two-line block in `src/skillspector/__init__.py`, `tests/exaforce/**`, and `docs/superpowers/**` — no other upstream-tracked file diverges.
