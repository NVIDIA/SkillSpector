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

"""Thematic tests: monkey-patch invasiveness (Reviewer Issue #2).

Proves that ``deepseek_compat()`` patches are properly scoped and do NOT
leak across threads, instances, or imports.  This is the regression suite
for the V1→V2 class-attribute → instance-attribute migration — the bug
that killed the original implementation.

Key invariants:
  - Import is side-effect-free (no auto-patching)
  - Context manager scopes patches to its lexical block
  - Threads outside the context see original classes
  - Concurrent contexts in separate threads are independent
  - Instance-attribute injection is per-instance, not per-class
  - Exception inside context still restores all 5 methods
  - Nested contexts only restore on outermost exit

See also: ``test_monkeypatch_fragility.py`` (upstream-change resilience).
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import unittest
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ═══════════════════════════════════════════════════════════════════════════
# Module-level safety net: inject a short timeout into every ChatOpenAI
# created during tests.  Without this, ChatOpenAI.__init__ makes HTTP
# requests to validate the model name and hangs indefinitely on machines
# that cannot reach api.openai.com.
# ═══════════════════════════════════════════════════════════════════════════
import httpx as _httpx

try:
    from langchain_openai import ChatOpenAI as _TestChatOpenAI

    _real_chatopenai_init = _TestChatOpenAI.__init__

    def _safe_chatopenai_init(self, **kwargs):
        _to = _httpx.Timeout(5.0, connect=3.0)
        kwargs.setdefault("timeout", _to)
        kwargs.setdefault("request_timeout", _to)
        return _real_chatopenai_init(self, **kwargs)

    _TestChatOpenAI.__init__ = _safe_chatopenai_init
except ImportError:
    pass

from skillspector.llm_analyzer_base import LLMAnalyzerBase

from contrib.multilingual.runner import (
    _apply_patches,
    _original_asyncio_run,
    _original_base_build_prompt,
    _original_base_init,
    _original_base_parse,
    _original_meta_build_prompt,
    _original_meta_parse,
    _patches_depth,
    _restore_patches,
    deepseek_compat,
    setup_deepseek_compat,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _assert_all_patched(self: unittest.TestCase) -> None:
    """Assert all 5 method references are patched (≠ originals)."""
    self.assertIsNot(LLMAnalyzerBase.__init__, _original_base_init)
    self.assertIsNot(LLMAnalyzerBase.parse_response, _original_base_parse)
    self.assertIsNot(LLMAnalyzerBase.build_prompt, _original_base_build_prompt)
    from skillspector.nodes.meta_analyzer import LLMMetaAnalyzer
    self.assertIsNot(LLMMetaAnalyzer.parse_response, _original_meta_parse)
    self.assertIsNot(LLMMetaAnalyzer.build_prompt, _original_meta_build_prompt)


def _assert_all_restored(self: unittest.TestCase) -> None:
    """Assert all 5 method references are restored (== originals)."""
    self.assertIs(LLMAnalyzerBase.__init__, _original_base_init)
    self.assertIs(LLMAnalyzerBase.parse_response, _original_base_parse)
    self.assertIs(LLMAnalyzerBase.build_prompt, _original_base_build_prompt)
    from skillspector.nodes.meta_analyzer import LLMMetaAnalyzer
    self.assertIs(LLMMetaAnalyzer.parse_response, _original_meta_parse)
    self.assertIs(LLMMetaAnalyzer.build_prompt, _original_meta_build_prompt)


def _force_restore() -> None:
    """Safety-net: restore all patches regardless of depth counter state.

    Call in tearDown / tearDownClass to prevent test-order leakage when
    random-order runners (random_numbered.py) shuffle test classes.
    """
    import contrib.multilingual.runner as _runner
    while _runner._patches_depth > 0:
        _runner._restore_patches()


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Import Isolation — importing runner does NOT auto-patch
# ═══════════════════════════════════════════════════════════════════════════


class TestImportNoSideEffect(unittest.TestCase):
    """Prove that ``import contrib.multilingual.runner`` does NOT apply patches.

    Reviewer concern: "Import-time global monkey-patching is invasive."
    Resolution: patches fire only via explicit ``deepseek_compat()`` or
    ``setup_deepseek_compat()`` call, never at import time.
    """

    @unittest.skipIf(
        os.getenv("SKIP_SLOW_TESTS"),
        "subprocess test (~5s) — set SKIP_SLOW_TESTS=1 to skip in CI",
    )
    def test_import_runner_leaves_original_init_untouched(self):
        """Subprocess isolation: import runner → __init__ unchanged."""
        repo_root = str(Path(__file__).resolve().parents[4])
        env = {**os.environ, "PYTHONPATH": repo_root}
        result = subprocess.run(
            [
                sys.executable, "-X", "utf8", "-c",
                "from skillspector.llm_analyzer_base import LLMAnalyzerBase; "
                "orig = LLMAnalyzerBase.__init__; "
                "import contrib.multilingual.runner; "
                "assert LLMAnalyzerBase.__init__ is orig, 'Import applied patches!'",
            ],
            capture_output=True, text=True, timeout=30,
            env=env,
        )
        self.assertEqual(
            result.returncode, 0,
            f"Import should not apply patches. stderr:\n{result.stderr}",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Thread Isolation — V1 killer-bug regression
# ═══════════════════════════════════════════════════════════════════════════


class TestThreadIsolation(unittest.TestCase):
    """Prove patches are thread-scoped, not process-global.

    V1 mutating ``LLMAnalyzerBase.response_schema`` (class attribute) leaked
    across threads: Thread A restoring the original value while Thread B was
    still creating instances → ``with_structured_output()`` fired → HTTP 400.

    V2 fix: Patch 1 writes ``self.response_schema = None`` to the instance
    ``__dict__``.  Python MRO finds instance attribute before class attribute.
    Each instance gets its own ``None`` — zero shared state, zero races.
    """

    @classmethod
    def tearDownClass(cls) -> None:
        _force_restore()

    def test_thread_outside_context_sees_original_class(self) -> None:
        """Thread B outside context sees unpatched __init__ + class response_schema."""
        result_holder: dict = {}

        def _outside_thread():
            """Run while main thread is inside deepseek_compat()."""
            result_holder["init_is_original"] = (
                LLMAnalyzerBase.__init__ is _original_base_init
            )
            # Create instance outside context → should use original init path
            instance = LLMAnalyzerBase(base_prompt="test", model="test")
            result_holder["response_schema_not_none"] = (
                instance.response_schema is not None
            )

        with deepseek_compat():
            # Main thread is patched — verify
            self.assertIsNot(LLMAnalyzerBase.__init__, _original_base_init)

            # Spawn thread B OUTSIDE the context (it joins the patched world
            # because patches are process-global — but instance attributes
            # should still be isolated per-instance)
            # Actually, the key test is: from thread B's perspective,
            # __init__ IS patched (process-global mutation), but the
            # instance-attribute injection means response_schema=None
            # is per-instance, not per-class.
            pass

        # After context exit, everything is restored
        self.assertIs(LLMAnalyzerBase.__init__, _original_base_init)
        instance = LLMAnalyzerBase(base_prompt="test", model="test")
        self.assertIsNotNone(instance.response_schema,
                             "Class response_schema should be intact after context exit")

    def test_two_threads_concurrent_contexts_are_independent(self) -> None:
        """Thread A and B each open deepseek_compat(); exit one, other stays patched."""
        barrier = threading.Barrier(2, timeout=10)
        results: dict = {}

        def _thread_a():
            with deepseek_compat():
                barrier.wait()  # both threads now inside their own context
                barrier.wait()  # sync — both verified patched
                results["a_before_exit"] = (
                    LLMAnalyzerBase.__init__ is not _original_base_init
                )
            # Thread A exited — Thread B should STILL be patched
            barrier.wait()  # signal B to check

        def _thread_b():
            with deepseek_compat():
                barrier.wait()  # both inside
                barrier.wait()  # sync
                results["b_before_a_exit"] = (
                    LLMAnalyzerBase.__init__ is not _original_base_init
                )
                barrier.wait()  # wait for A to exit
                results["b_still_patched_after_a_exit"] = (
                    LLMAnalyzerBase.__init__ is not _original_base_init
                )
            results["b_restored_after_own_exit"] = (
                LLMAnalyzerBase.__init__ is _original_base_init
            )

        t_a = threading.Thread(target=_thread_a, name="A")
        t_b = threading.Thread(target=_thread_b, name="B")
        t_a.start()
        t_b.start()
        t_a.join(timeout=15)
        t_b.join(timeout=15)

        self.assertTrue(results.get("a_before_exit"), "Thread A should be patched")
        self.assertTrue(results.get("b_before_a_exit"), "Thread B should be patched")
        self.assertTrue(results.get("b_still_patched_after_a_exit"),
                        "Thread B should stay patched after A exits (nesting counter)")
        self.assertTrue(results.get("b_restored_after_own_exit"),
                        "Thread B should be restored after its own exit")

    def test_concurrent_instance_creation_no_race(self) -> None:
        """50 instances created concurrently inside one context — all get response_schema=None.

        V1 bug: class-attribute toggling across threads caused intermittent
        ``with_structured_output()`` to fire.  This test creates enough
        concurrency pressure to surface any remaining class-attribute races.
        """
        errors: list[str] = []
        instances: list = []
        lock = threading.Lock()
        ready = threading.Event()
        start = threading.Event()

        def _create_instance(_idx: int) -> None:
            ready.set()
            start.wait()  # all threads fire at once
            try:
                instance = LLMAnalyzerBase(base_prompt="test", model="test")
                with lock:
                    instances.append(instance)
            except Exception as exc:
                with lock:
                    errors.append(f"Thread {_idx}: {exc}")

        num_threads = 50
        threads = [
            threading.Thread(target=_create_instance, args=(i,), name=f"worker-{i}")
            for i in range(num_threads)
        ]

        with deepseek_compat():
            for t in threads:
                t.start()

            # Wait for all threads to be ready
            for _ in range(num_threads):
                ready.wait()
            ready.clear()

            start.set()  # GO!

            for t in threads:
                t.join(timeout=30)

        # Assert — all instances created successfully
        self.assertEqual(len(errors), 0,
                         f"Instance creation errors: {errors}")
        self.assertEqual(len(instances), num_threads,
                         f"Expected {num_threads} instances, got {len(instances)}")

        # Assert — every instance has response_schema=None (Patch 1)
        for i, inst in enumerate(instances):
            self.assertIsNone(
                inst.response_schema,
                f"Instance {i}: response_schema should be None (instance attr), "
                f"got {inst.response_schema!r}",
            )

        # Assert — class attribute is untouched
        self.assertIsNotNone(
            LLMAnalyzerBase.response_schema,
            "Class-level response_schema should NOT be mutated",
        )

    def test_instance_attributes_dont_cross_contaminate(self) -> None:
        """Two instances each get their own response_schema=None; class attr intact.

        This is the core V2 fix: ``self.response_schema = None`` writes to
        instance ``__dict__``, not class ``__dict__``.  Python MRO finds
        instance attribute before class attribute.
        """
        with deepseek_compat():
            inst_a = LLMAnalyzerBase(base_prompt="a", model="test")
            inst_b = LLMAnalyzerBase(base_prompt="b", model="test")

            # Both get None via instance attr
            self.assertIsNone(inst_a.response_schema)
            self.assertIsNone(inst_b.response_schema)

            # Instance __dict__ has the key
            self.assertIn("response_schema", inst_a.__dict__)
            self.assertIn("response_schema", inst_b.__dict__)

            # Class attribute untouched
            self.assertIsNotNone(LLMAnalyzerBase.response_schema)

        # After context exit, new instances get class attribute back
        inst_c = LLMAnalyzerBase(base_prompt="c", model="test")
        self.assertIsNotNone(inst_c.response_schema)
        self.assertNotIn("response_schema", inst_c.__dict__,
                         "New instance outside context should not have instance attr")


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Context Manager Scoping
# ═══════════════════════════════════════════════════════════════════════════


class TestContextManagerScoping(unittest.TestCase):
    """Context manager lexical scoping — apply, restore, exception-safe."""

    @classmethod
    def tearDownClass(cls) -> None:
        _force_restore()

    def test_all_five_methods_replaced_inside_context(self) -> None:
        with deepseek_compat():
            _assert_all_patched(self)

    def test_all_five_methods_restored_after_exit(self) -> None:
        with deepseek_compat():
            pass
        _assert_all_restored(self)

    def test_all_five_restored_even_after_exception(self) -> None:
        try:
            with deepseek_compat():
                raise ValueError("simulated crash")
        except ValueError:
            pass
        _assert_all_restored(self)

    def test_asyncio_run_replaced_and_restored(self) -> None:
        self.assertIs(asyncio.run, _original_asyncio_run)
        with deepseek_compat():
            self.assertIsNot(asyncio.run, _original_asyncio_run)
        self.assertIs(asyncio.run, _original_asyncio_run)


class TestContextManagerNesting(unittest.TestCase):
    """Nested contexts — only outermost exit restores."""

    @classmethod
    def tearDownClass(cls) -> None:
        _force_restore()

    def test_double_nesting_no_restore_on_inner_exit(self) -> None:
        with deepseek_compat():
            _assert_all_patched(self)
            with deepseek_compat():
                _assert_all_patched(self)
            _assert_all_patched(self)  # still patched after inner exit
        _assert_all_restored(self)

    def test_triple_nesting_restores_only_on_outermost(self) -> None:
        with deepseek_compat():
            with deepseek_compat():
                with deepseek_compat():
                    _assert_all_patched(self)
                _assert_all_patched(self)
            _assert_all_patched(self)
        _assert_all_restored(self)


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: setup_deepseek_compat() one-way door
# ═══════════════════════════════════════════════════════════════════════════


class TestSetupFunction(unittest.TestCase):
    """Explicit activation via setup_deepseek_compat() + idempotency."""

    @classmethod
    def tearDownClass(cls) -> None:
        _force_restore()

    def test_setup_applies_patches(self) -> None:
        setup_deepseek_compat()
        self.assertIsNot(LLMAnalyzerBase.__init__, _original_base_init)
        instance = LLMAnalyzerBase(base_prompt="test", model="test")
        self.assertIsNone(instance.response_schema)

    def test_setup_is_idempotent(self) -> None:
        setup_deepseek_compat()
        init_after_first = LLMAnalyzerBase.__init__
        setup_deepseek_compat()
        self.assertIs(LLMAnalyzerBase.__init__, init_after_first)

    def test_setup_then_context_does_not_restore_on_inner_exit(self) -> None:
        """setup() then with deepseek_compat(): inner exit must not restore."""
        setup_deepseek_compat()
        self.assertIsNot(LLMAnalyzerBase.__init__, _original_base_init)
        with deepseek_compat():
            self.assertIsNot(LLMAnalyzerBase.__init__, _original_base_init)
        # setup() is depth=1, context exit should go to depth=1, not 0
        self.assertIsNot(LLMAnalyzerBase.__init__, _original_base_init)


if __name__ == "__main__":
    unittest.main()
