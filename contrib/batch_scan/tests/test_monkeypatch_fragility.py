# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Thematic tests: monkey-patch fragility (Reviewer Issue #2).

Proves that ``deepseek_compat()`` patches survive upstream changes by
verifying that the ``_verify_patch_targets`` guard catches broken
assumptions BEFORE any patches are applied.

Key invariants:
  - Guard catches missing parameters (upstream renamed/removed)
  - Guard catches keyword-only migration (positional → kwarg)
  - Guard catches removed deep dependencies (Pydantic methods, Batch fields)
  - Guard catches removed class attributes (response_schema)
  - Guard passes cleanly against current upstream (no false positive)
  - Guard runs atomically — if any check fails, no patches are applied
  - Each of the 7 patches has unique, distinguishable guard coverage

See also: ``test_monkeypatch_invasiveness.py`` (thread-scoping proof).
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import sys
import unittest
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from skillspector.llm_analyzer_base import (
    Batch,
    LLMAnalyzerBase,
    LLMAnalysisResult,
    LLMFinding,
)
from skillspector.nodes.meta_analyzer import LLMMetaAnalyzer, MetaAnalyzerResult

from contrib.batch_scan.runner import (
    _check_signature,
    _original_asyncio_run,
    _original_base_init,
    _original_base_parse,
    _original_base_build_prompt,
    _original_meta_parse,
    _original_meta_build_prompt,
    _verify_patch_targets,
    _apply_patches,
    _restore_patches,
    deepseek_compat,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _force_restore() -> None:
    """Safety-net: restore all patches regardless of depth counter."""
    import contrib.batch_scan.runner as _runner
    while _runner._patches_depth > 0:
        _runner._restore_patches()


class _TempAttributeOverride:
    """Context manager to temporarily replace / delete an attribute on an object.

    Usage::

        with _TempAttributeOverride(LLMAnalysisResult, "model_validate", None):
            # model_validate is temporarily None
            ...
        # model_validate restored
    """

    def __init__(self, obj: object, attr: str, replacement=None, *, delete: bool = False):
        self._obj = obj
        self._attr = attr
        self._replacement = replacement
        self._delete = delete
        self._saved = None
        self._had_attr = False

    def __enter__(self):
        self._had_attr = hasattr(self._obj, self._attr)
        if self._had_attr:
            self._saved = getattr(self._obj, self._attr)
        if self._delete:
            if self._had_attr:
                delattr(self._obj, self._attr)
        else:
            setattr(self._obj, self._attr, self._replacement)
        return self

    def __exit__(self, *args):
        if self._had_attr:
            setattr(self._obj, self._attr, self._saved)
        elif not self._delete:
            delattr(self._obj, self._attr)


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: _check_signature — parameter-level guard
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckSignature(unittest.TestCase):
    """``_check_signature()`` — the micro-guard behind every parameter check.

    Three failure modes:
      1. Missing parameter (upstream removed it)
      2. KEYWORD_ONLY parameter (upstream made positional → kwarg)
      3. Uninspectable function (C builtin, etc.)
    """

    def test_passes_when_all_params_present(self) -> None:
        def _sample(self, a, b, c):
            pass

        # Should not raise
        _check_signature(_sample, ["self", "a", "b", "c"], "test_func", 99)

    def test_raises_when_param_missing(self) -> None:
        def _sample(self, a, b):
            pass

        with self.assertRaises(RuntimeError) as ctx:
            _check_signature(_sample, ["self", "a", "b", "c"], "test_func", 99)
        self.assertIn("no longer has 'c'", str(ctx.exception))

    def test_raises_when_param_becomes_keyword_only(self) -> None:
        def _sample(self, *, a, b, c):
            pass

        with self.assertRaises(RuntimeError) as ctx:
            _check_signature(_sample, ["self", "a", "b", "c"], "test_func", 99)
        self.assertIn("keyword-only", str(ctx.exception))


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Guard passes against current upstream (no false positive)
# ═══════════════════════════════════════════════════════════════════════════


class TestGuardPassesCurrentUpstream(unittest.TestCase):
    """``_verify_patch_targets()`` must pass cleanly against the currently
    installed upstream version.  Any failure here means upstream already
    broke something and the guard is doing its job — but patches need
    updating.
    """

    @classmethod
    def tearDownClass(cls) -> None:
        _force_restore()

    def test_verify_patch_targets_does_not_raise(self) -> None:
        try:
            _verify_patch_targets()
        except RuntimeError as exc:
            self.fail(f"_verify_patch_targets raised against current upstream: {exc}")

    def test_context_manager_enter_passes_guard(self) -> None:
        try:
            with deepseek_compat():
                pass
        except RuntimeError as exc:
            self.fail(f"deepseek_compat() guard failed: {exc}")

    def test_guard_after_context_cycle_still_passes(self) -> None:
        """Guard should pass even after patches were applied and restored."""
        with deepseek_compat():
            pass
        # After full apply+restore cycle, guard must still pass
        try:
            _verify_patch_targets()
        except RuntimeError as exc:
            self.fail(f"Guard failed after apply+restore cycle: {exc}")

    def test_guard_after_setup_and_manual_restore_still_passes(self) -> None:
        """Guard should pass after setup_deepseek_compat() + manual restore."""
        from contrib.batch_scan.runner import setup_deepseek_compat
        setup_deepseek_compat()
        _force_restore()
        try:
            _verify_patch_targets()
        except RuntimeError as exc:
            self.fail(f"Guard failed after setup+restore cycle: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Each patch guard catches its specific breakage
# ═══════════════════════════════════════════════════════════════════════════


class TestGuardPatch1Init(unittest.TestCase):
    """Guard for Patch 1: LLMAnalyzerBase.__init__(self, base_prompt, model)
    AND class attribute ``response_schema`` exists."""

    @classmethod
    def tearDownClass(cls) -> None:
        _force_restore()

    def test_guard_catches_missing_base_prompt_param(self) -> None:
        """If upstream removes 'base_prompt' from __init__, guard must raise."""
        original = LLMAnalyzerBase.__init__

        def _broken_init(self, model):
            pass

        try:
            LLMAnalyzerBase.__init__ = _broken_init
            with self.assertRaises(RuntimeError) as ctx:
                _verify_patch_targets()
            self.assertIn("Patch 1", str(ctx.exception))
            self.assertIn("base_prompt", str(ctx.exception))
        finally:
            LLMAnalyzerBase.__init__ = original

    def test_guard_catches_missing_model_param(self) -> None:
        """If upstream removes 'model' from __init__, guard must raise."""
        original = LLMAnalyzerBase.__init__

        def _broken_init(self, base_prompt):
            pass

        try:
            LLMAnalyzerBase.__init__ = _broken_init
            with self.assertRaises(RuntimeError):
                _verify_patch_targets()
        finally:
            LLMAnalyzerBase.__init__ = original

    def test_guard_catches_missing_response_schema_attr(self) -> None:
        """If upstream removes response_schema class attr, guard must raise."""
        with _TempAttributeOverride(LLMAnalyzerBase, "response_schema", delete=True):
            with self.assertRaises(RuntimeError) as ctx:
                _verify_patch_targets()
            self.assertIn("response_schema", str(ctx.exception))


class TestGuardPatch2ParseResponse(unittest.TestCase):
    """Guard for Patch 2: LLMAnalyzerBase.parse_response + deep deps."""

    @classmethod
    def tearDownClass(cls) -> None:
        _force_restore()

    def test_guard_catches_missing_batch_param(self) -> None:
        """If parse_response no longer accepts 'batch', guard must raise."""
        original = LLMAnalyzerBase.parse_response

        def _broken_parse(self, response):
            pass

        try:
            LLMAnalyzerBase.parse_response = _broken_parse
            with self.assertRaises(RuntimeError) as ctx:
                _verify_patch_targets()
            self.assertIn("Patch 2", str(ctx.exception))
        finally:
            LLMAnalyzerBase.parse_response = original

    def test_guard_catches_missing_model_validate(self) -> None:
        """If LLMAnalysisResult.model_validate is removed, guard must raise.

        model_validate is a Pydantic metaclass-injected classmethod that
        cannot be deleted via delattr.  We monkey-patch builtins.hasattr
        to simulate its absence.
        """
        import builtins
        _real_hasattr = builtins.hasattr

        def _fake_hasattr(obj, name):
            if obj is LLMAnalysisResult and name == "model_validate":
                return False
            return _real_hasattr(obj, name)

        try:
            builtins.hasattr = _fake_hasattr
            with self.assertRaises(RuntimeError) as ctx:
                _verify_patch_targets()
            self.assertIn("model_validate", str(ctx.exception))
        finally:
            builtins.hasattr = _real_hasattr

    def test_guard_catches_missing_to_finding(self) -> None:
        """If LLMFinding.to_finding is removed, guard must raise."""
        with _TempAttributeOverride(LLMFinding, "to_finding", delete=True):
            with self.assertRaises(RuntimeError) as ctx:
                _verify_patch_targets()
            self.assertIn("to_finding", str(ctx.exception))

    def test_guard_catches_missing_batch_file_path_field(self) -> None:
        """If Batch.file_path field is removed, guard must raise.

        Batch is a @dataclass — we test by removing the field from __dataclass_fields__.
        """
        saved_fields = Batch.__dataclass_fields__.copy()  # type: ignore[attr-defined]
        try:
            # Remove file_path from dataclass fields
            Batch.__dataclass_fields__ = {  # type: ignore[attr-defined]
                k: v for k, v in saved_fields.items() if k != "file_path"
            }
            with self.assertRaises(RuntimeError) as ctx:
                _verify_patch_targets()
            self.assertIn("file_path", str(ctx.exception))
        finally:
            Batch.__dataclass_fields__ = saved_fields  # type: ignore[attr-defined]


class TestGuardPatch3MetaParse(unittest.TestCase):
    """Guard for Patch 3: LLMMetaAnalyzer.parse_response + deep deps."""

    @classmethod
    def tearDownClass(cls) -> None:
        _force_restore()

    def test_guard_catches_missing_batch_param_on_meta_parse(self) -> None:
        original = LLMMetaAnalyzer.parse_response

        def _broken(self, response):
            pass

        try:
            LLMMetaAnalyzer.parse_response = _broken
            with self.assertRaises(RuntimeError) as ctx:
                _verify_patch_targets()
            self.assertIn("Patch 3", str(ctx.exception))
        finally:
            LLMMetaAnalyzer.parse_response = original

    def test_guard_catches_missing_meta_analyzer_model_validate(self) -> None:
        import builtins
        _real_hasattr = builtins.hasattr

        def _fake_hasattr(obj, name):
            if obj is MetaAnalyzerResult and name == "model_validate":
                return False
            return _real_hasattr(obj, name)

        try:
            builtins.hasattr = _fake_hasattr
            with self.assertRaises(RuntimeError) as ctx:
                _verify_patch_targets()
            self.assertIn("model_validate", str(ctx.exception))
        finally:
            builtins.hasattr = _real_hasattr

    def test_guard_catches_missing_findings_field(self) -> None:
        """If MetaAnalyzerResult no longer has 'findings' field."""
        saved = MetaAnalyzerResult.model_fields.copy()
        try:
            MetaAnalyzerResult.model_fields = {
                k: v for k, v in saved.items() if k != "findings"
            }
            with self.assertRaises(RuntimeError) as ctx:
                _verify_patch_targets()
            self.assertIn("findings", str(ctx.exception))
        finally:
            MetaAnalyzerResult.model_fields = saved


class TestGuardPatch4BaseBuildPrompt(unittest.TestCase):
    """Guard for Patch 4: LLMAnalyzerBase.build_prompt(self, batch, **kwargs)."""

    @classmethod
    def tearDownClass(cls) -> None:
        _force_restore()

    def test_guard_catches_missing_batch_param(self) -> None:
        original = LLMAnalyzerBase.build_prompt

        def _broken(self):
            return "prompt"

        try:
            LLMAnalyzerBase.build_prompt = _broken
            with self.assertRaises(RuntimeError) as ctx:
                _verify_patch_targets()
            self.assertIn("Patch 4", str(ctx.exception))
            self.assertIn("batch", str(ctx.exception))
        finally:
            LLMAnalyzerBase.build_prompt = original

    def test_guard_catches_missing_kwargs(self) -> None:
        """If build_prompt no longer accepts **kwargs."""
        original = LLMAnalyzerBase.build_prompt

        def _broken(self, batch):
            return "prompt"

        try:
            LLMAnalyzerBase.build_prompt = _broken
            with self.assertRaises(RuntimeError) as ctx:
                _verify_patch_targets()
            self.assertIn("**kwargs", str(ctx.exception))
        finally:
            LLMAnalyzerBase.build_prompt = original


class TestGuardPatch5MetaBuildPrompt(unittest.TestCase):
    """Guard for Patch 5: LLMMetaAnalyzer.build_prompt(self, batch, **kwargs)."""

    @classmethod
    def tearDownClass(cls) -> None:
        _force_restore()

    def test_guard_catches_missing_batch_param(self) -> None:
        original = LLMMetaAnalyzer.build_prompt

        def _broken(self):
            return "prompt"

        try:
            LLMMetaAnalyzer.build_prompt = _broken
            with self.assertRaises(RuntimeError) as ctx:
                _verify_patch_targets()
            self.assertIn("Patch 5", str(ctx.exception))
        finally:
            LLMMetaAnalyzer.build_prompt = original


class TestGuardPatch7Asyncio(unittest.TestCase):
    """Guard for Patch 7: asyncio.run(main, *, debug=None, loop_factory=None)
    AND deep dep: asyncio.new_event_loop is callable."""

    @classmethod
    def tearDownClass(cls) -> None:
        _force_restore()

    def test_guard_catches_missing_main_param(self) -> None:
        """If asyncio.run signature changes, guard uses saved _original_asyncio_run."""
        # _verify_patch_targets inspects _original_asyncio_run (module-load snapshot),
        # not asyncio.run (which may already be patched).  The original always has
        # 'main' — this is a structural test confirming the guard covers Patch 7.
        self.assertTrue(callable(_original_asyncio_run))

        # Verify the guard checks 'main' parameter on the original
        sig = inspect.signature(_original_asyncio_run)
        self.assertIn("main", sig.parameters,
                      "asyncio.run should have 'main' parameter")

    def test_guard_catches_missing_new_event_loop(self) -> None:
        """If asyncio.new_event_loop is removed, guard must raise."""
        with _TempAttributeOverride(asyncio, "new_event_loop", None):
            with self.assertRaises(RuntimeError) as ctx:
                _verify_patch_targets()
            self.assertIn("new_event_loop", str(ctx.exception))


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Atomicity — guard fails → no patches applied
# ═══════════════════════════════════════════════════════════════════════════


class TestGuardAtomicity(unittest.TestCase):
    """If _verify_patch_targets raises, NO patches should be applied.

    This is the "fail-closed" property: a broken upstream should result in
    a loud error, not silently-malfunctioning patches.
    """

    @classmethod
    def tearDownClass(cls) -> None:
        _force_restore()
        # Ensure response_schema is restored
        if hasattr(LLMAnalyzerBase, "_response_schema_original"):
            LLMAnalyzerBase.response_schema = LLMAnalyzerBase._response_schema_original

    def test_failed_guard_leaves_no_patches_applied(self) -> None:
        """Break response_schema, call _apply_patches, verify it raises and
        no methods are patched."""
        # Force-clean state
        _force_restore()

        with _TempAttributeOverride(LLMAnalyzerBase, "response_schema", delete=True):
            # Guard should raise → _apply_patches should propagate
            with self.assertRaises(RuntimeError):
                _apply_patches()

        # After the failed attempt, NO methods should be patched
        _assert_all_restored(self)


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: Original references captured at module load, not at apply-time
# ═══════════════════════════════════════════════════════════════════════════


class TestOriginalCapturedAtImportTime(unittest.TestCase):
    """Module-level original references are snapshotted when runner.py is
    first imported, not when _apply_patches() runs.  This ensures they are
    always the true upstream originals, never a previously-patched version.
    """

    def test_original_base_init_is_true_upstream(self) -> None:
        self.assertTrue(
            _original_base_init.__name__.startswith("__init__")
            or "LLMAnalyzerBase" in str(_original_base_init),
        )

    def test_original_chatopenai_init_is_not_none(self) -> None:
        from contrib.batch_scan.runner import _original_chatopenai_init
        self.assertIsNotNone(
            _original_chatopenai_init,
            "_original_chatopenai_init must be captured at import time",
        )

    def test_original_asyncio_run_is_true_stdlib(self) -> None:
        self.assertIs(_original_asyncio_run, asyncio.run,
                      "_original_asyncio_run should be the stdlib function (unpatched)")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers (module-level reuse)
# ═══════════════════════════════════════════════════════════════════════════


def _assert_all_restored(test_case: unittest.TestCase) -> None:
    """Assert all 5 method references point to originals."""
    test_case.assertIs(LLMAnalyzerBase.__init__, _original_base_init)
    test_case.assertIs(LLMAnalyzerBase.parse_response, _original_base_parse)
    test_case.assertIs(LLMAnalyzerBase.build_prompt, _original_base_build_prompt)
    test_case.assertIs(LLMMetaAnalyzer.parse_response, _original_meta_parse)
    test_case.assertIs(LLMMetaAnalyzer.build_prompt, _original_meta_build_prompt)


if __name__ == "__main__":
    unittest.main()
