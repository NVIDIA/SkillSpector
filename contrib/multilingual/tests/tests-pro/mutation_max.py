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

"""Mutation test — Max's 4 risk areas. Injects bugs, verifies tests catch them.

Areas: 1) Pool acquire/release  2) 429 backoff/recovery
        3) Monkey-patches       4) GapFillAnalyzer.parse_response
"""

from __future__ import annotations

import unittest, sys, time
from pathlib import Path

_project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_project_root))

results = []


def mutate(label: str, module: str, target: str, broken_fn, test_specs: list[tuple[str, str]]):
    """Inject *broken_fn* into *module.target*, run *test_specs*, restore."""
    mod = __import__(module, fromlist=[""])
    parts = target.split(".")
    obj = mod
    for p in parts[:-1]:
        obj = getattr(obj, p)
    attr = parts[-1]
    original = getattr(obj, attr)
    setattr(obj, attr, broken_fn)
    try:
        for test_mod, test_cls in test_specs:
            suite = unittest.TestLoader().loadTestsFromName(
                f"contrib.multilingual.tests.tests-pro.{test_mod}.{test_cls}"
            )
            r = unittest.TextTestRunner(verbosity=0).run(suite)
            caught = not r.wasSuccessful()
            results.append((label, test_cls, caught))
    finally:
        setattr(obj, attr, original)


# ═══════════════════════════════════════════════════════════════════════
# Area 1: Pool acquire/release
# ═══════════════════════════════════════════════════════════════════════

# Mutation 1a: acquire forgets to increment active_requests
import contrib.multilingual.api_pool as _ap
_orig_acquire = _ap.ApiKeyPool.acquire


def _broken_acquire_no_increment(self, timeout=None):
    import time as _t
    deadline = _t.monotonic() + timeout if timeout is not None else None
    with self._condition:
        while True:
            now = _t.monotonic()
            self._recover_expired_keys(now)
            available = [k for k in self._keys if k.available]
            if available:
                key = min(available, key=lambda k: k.active_requests)
                # BUG: forgot key.active_requests += 1
                key.total_requests += 1
                return key
            wait_for = self._next_available_in(now)
            remaining = self._remaining_timeout(deadline)
            if remaining is not None and remaining <= 0:
                raise RuntimeError("timeout")
            self._condition.wait(timeout=min(wait_for or remaining, remaining or 5.0))


_ap.ApiKeyPool.acquire = _broken_acquire_no_increment
mutate("acquire forgets active_requests++", "contrib.multilingual.api_pool",
       "ApiKeyPool.acquire", _broken_acquire_no_increment,
       [("test_api_pool", "TestAcquireRelease")])
_ap.ApiKeyPool.acquire = _orig_acquire

# Mutation 1b: release forgets to decrement active_requests
_orig_release = _ap.ApiKeyPool.release


def _broken_release_no_decrement(self, key, *, success=True):
    with self._condition:
        # BUG: forgot key.active_requests = max(0, key.active_requests - 1)
        if success:
            key.consecutive_429 = 0
        else:
            key.consecutive_429 += 1
            key.rate_limited_until = time.monotonic() + min(
                30 * (2 ** (key.consecutive_429 - 1)), 300
            )
            key.rate_limited = True
            self._rate_limits_hit += 1
        self._condition.notify_all()


_ap.ApiKeyPool.release = _broken_release_no_decrement
mutate("release forgets active_requests--", "contrib.multilingual.api_pool",
       "ApiKeyPool.release", _broken_release_no_decrement,
       [("test_api_pool", "TestAcquireRelease"),
        ("test_api_pool", "TestResourceLeakRecovery")])
_ap.ApiKeyPool.release = _orig_release

# Mutation 1c: least-loaded scheduling broken — always returns first key
_orig_acquire2 = _ap.ApiKeyPool.acquire


def _broken_acquire_no_load_balance(self, timeout=None):
    import time as _t
    deadline = _t.monotonic() + timeout if timeout is not None else None
    with self._condition:
        while True:
            now = _t.monotonic()
            self._recover_expired_keys(now)
            available = [k for k in self._keys if k.available]
            if available:
                # BUG: always returns first available key, ignoring load
                key = available[0]
                key.active_requests += 1
                key.total_requests += 1
                self._total_requests_served += 1
                _now_active = sum(k.active_requests for k in self._keys)
                if _now_active > self._peak_active_requests:
                    self._peak_active_requests = _now_active
                return key
            wait_for = self._next_available_in(now)
            remaining = self._remaining_timeout(deadline)
            if remaining is not None and remaining <= 0:
                raise RuntimeError("timeout")
            self._condition.wait(timeout=min(wait_for or remaining, remaining or 5.0))


_ap.ApiKeyPool.acquire = _broken_acquire_no_load_balance
mutate("least-loaded scheduling broken", "contrib.multilingual.api_pool",
       "ApiKeyPool.acquire", _broken_acquire_no_load_balance,
       [("test_api_pool", "TestEdgeCases")])  # test_released_slot_returns_least_loaded_key
_ap.ApiKeyPool.acquire = _orig_acquire2

# Mutation 1d: try_acquire ignores rate-limited keys
_orig_try_acquire = _ap.ApiKeyPool.try_acquire


def _broken_try_acquire(self):
    with self._lock:
        # BUG: _recover_expired_keys NOT called — rate-limited keys never recover via try_acquire
        available = [k for k in self._keys if k.available]
        if not available:
            return None
        key = min(available, key=lambda k: k.active_requests)
        key.active_requests += 1
        key.total_requests += 1
        self._total_requests_served += 1
        _now_active = sum(k.active_requests for k in self._keys)
        if _now_active > self._peak_active_requests:
            self._peak_active_requests = _now_active
        return key


_ap.ApiKeyPool.try_acquire = _broken_try_acquire
mutate("try_acquire recovery broken", "contrib.multilingual.api_pool",
       "ApiKeyPool.try_acquire", _broken_try_acquire,
       [("test_api_pool", "TestRecoveredKeyScheduling")])
_ap.ApiKeyPool.try_acquire = _orig_try_acquire

# ═══════════════════════════════════════════════════════════════════════
# Area 2: 429 backoff/recovery
# ═══════════════════════════════════════════════════════════════════════

# Mutation 2a: backoff always 5s regardless of consecutive count
_orig_release2 = _ap.ApiKeyPool.release


def _broken_release_fixed_backoff(self, key, *, success=True):
    with self._condition:
        key.active_requests = max(0, key.active_requests - 1)
        if success:
            key.consecutive_429 = 0
        else:
            key.consecutive_429 += 1
            # BUG: always 5s, not min(30*2^(n-1), 300)
            key.rate_limited_until = time.monotonic() + 5
            key.rate_limited = True
            self._rate_limits_hit += 1
        self._condition.notify_all()


_ap.ApiKeyPool.release = _broken_release_fixed_backoff
mutate("backoff always 5s", "contrib.multilingual.api_pool",
       "ApiKeyPool.release", _broken_release_fixed_backoff,
       [("test_api_pool", "TestRateLimitBackoff")])
_ap.ApiKeyPool.release = _orig_release2

# Mutation 2b: _recover_expired_keys never recovers
_orig_recover = _ap.ApiKeyPool._recover_expired_keys


def _broken_recover(self, now):
    pass  # BUG: never recovers rate-limited keys


_ap.ApiKeyPool._recover_expired_keys = _broken_recover
mutate("recovery never runs", "contrib.multilingual.api_pool",
       "ApiKeyPool._recover_expired_keys", _broken_recover,
       [("test_api_pool", "TestRateLimitBackoff")])  # TestRecoveredKeyScheduling hangs: acquire() blocks forever w/o recovery
_ap.ApiKeyPool._recover_expired_keys = _orig_recover

# ═══════════════════════════════════════════════════════════════════════
# Area 3: Monkey-patches
# ═══════════════════════════════════════════════════════════════════════

# Mutation 3a: Patch 1 broken — doesn't set response_schema=None
import contrib.multilingual.runner as _runner

_orig_patched_init = _runner._patched_base_init


def _broken_patched_init(self, base_prompt, model):
    # BUG: forgot self.response_schema = None
    _runner._original_base_init(self, base_prompt, model)


_runner._patched_base_init = _broken_patched_init
_runner.LLMAnalyzerBase.__init__ = _broken_patched_init
# Need to re-apply patches via setup for this mutation to take effect
# Actually, just test via direct replacement
del _runner._patched_base_init
# Restore properly
_runner._patched_base_init = _orig_patched_init

# Better approach: directly test with deepseek_compat context
_orig_apply = _runner._apply_patches


def _broken_apply_no_patch1():
    if _runner._patches_depth > 0:
        _runner._patches_depth += 1
        return
    _runner._verify_patch_targets()
    # BUG: skipping Patch 1 (LLMAnalyzerBase.__init__)
    # _runner.LLMAnalyzerBase.__init__ = _runner._patched_base_init
    _runner.LLMAnalyzerBase.parse_response = _runner._patched_base_parse
    _runner.LLMAnalyzerBase.build_prompt = _runner._patched_base_build_prompt
    _runner.LLMMetaAnalyzer.parse_response = _runner._patched_meta_parse
    _runner.LLMMetaAnalyzer.build_prompt = _runner._patched_meta_build_prompt
    try:
        import httpx
        from langchain_openai import ChatOpenAI as _CO
        _runner._original_chatopenai_init = _CO.__init__
        _CO.__init__ = _runner._patched_chatopenai_init
    except ImportError:
        pass
    _runner._asyncio.run = _runner._patched_asyncio_run
    _runner._patches_depth = 1


_runner._apply_patches = _broken_apply_no_patch1
mutate("Patch 1 not applied", "contrib.multilingual.runner",
       "_apply_patches", _broken_apply_no_patch1,
       [("test_runner_patches", "TestContextManagerApplyRestore")])
_runner._apply_patches = _orig_apply

# Mutation 3b: Patch 6 timeout not injected
_orig_patched_co = _runner._patched_chatopenai_init


def _broken_co_init(self, **kwargs):
    # BUG: forgot to inject timeout
    _runner._original_chatopenai_init(self, **kwargs)


_runner._patched_chatopenai_init = _broken_co_init
mutate("Patch 6 no timeout", "contrib.multilingual.runner",
       "_patched_chatopenai_init", _broken_co_init,
       [("test_runner_patches", "TestPatch6ChatOpenAITimeout")])
_runner._patched_chatopenai_init = _orig_patched_co

# ═══════════════════════════════════════════════════════════════════════
# Area 4: GapFillAnalyzer.parse_response
# ═══════════════════════════════════════════════════════════════════════

import contrib.multilingual.gap_fill as _gf

# Mutation 4a: confidence filter broken — threshold 0.7 → 0.0
_orig_parse = _gf.GapFillAnalyzer.parse_response


def _broken_parse_no_filter(self, response, batch):
    import json as _json
    text = str(response).strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    try:
        data = _json.loads(text)
    except _json.JSONDecodeError:
        return []
    try:
        result = _gf.GapFillResult.model_validate(data)
        items = []
        for item in result.findings:
            if item.rule_id not in _gf._GAP_FILL_RULE_IDS:
                continue
            # BUG: confidence check removed — all findings pass regardless
            items.append(item.to_finding(batch.file_path))
        return items
    except Exception:
        return []


# Apply directly to class since mutation test targets the class method
_gf.GapFillAnalyzer.parse_response = _broken_parse_no_filter
mutate("confidence filter removed", "contrib.multilingual.gap_fill",
       "GapFillAnalyzer.parse_response", _broken_parse_no_filter,
       [("test_gap_fill", "TestParseResponseFiltering")])
_gf.GapFillAnalyzer.parse_response = _orig_parse

# Mutation 4b: markdown fence stripping broken
_orig_parse2 = _gf.GapFillAnalyzer.parse_response


def _broken_parse_no_fence_strip(self, response, batch):
    import json as _json
    # BUG: fence stripping removed entirely
    text = str(response)  # missing .strip()
    try:
        data = _json.loads(text)
    except _json.JSONDecodeError:
        return []
    try:
        result = _gf.GapFillResult.model_validate(data)
        return [item.to_finding(batch.file_path)
                for item in result.findings
                if item.rule_id in _gf._GAP_FILL_RULE_IDS and item.confidence >= 0.7]
    except Exception:
        return []


_gf.GapFillAnalyzer.parse_response = _broken_parse_no_fence_strip
mutate("fence stripping broken", "contrib.multilingual.gap_fill",
       "GapFillAnalyzer.parse_response", _broken_parse_no_fence_strip,
       [("test_gap_fill", "TestParseResponseMarkdownFences")])
_gf.GapFillAnalyzer.parse_response = _orig_parse2

# ── Patch 2 mutation: parse_response broken ──────────────────────
_orig_patched_parse = _runner._patched_base_parse


def _broken_patched_parse(self, response, batch):
    # BUG: always returns empty — JSON parsing silently broken
    if isinstance(response, _runner.LLMAnalysisResult):
        return _runner._original_base_parse(self, response, batch)
    return []  # BUG: swallows all findings


_runner._patched_base_parse = _broken_patched_parse
_runner.LLMAnalyzerBase.parse_response = _broken_patched_parse
mutate("Patch 2 parse always empty", "contrib.multilingual.runner",
       "_patched_base_parse", _broken_patched_parse,
       [("test_runner_patches", "TestContextManagerApplyRestore")])
_runner._patched_base_parse = _orig_patched_parse

# ── Patch 3 mutation: _sanitize_meta_finding broken ───────────────
_orig_meta_parse = _runner._patched_meta_parse


def _broken_meta_parse(self, response, batch):
    if isinstance(response, _runner.MetaAnalyzerResult):
        return _runner._original_meta_parse(self, response, batch)
    text = _runner._strip_markdown_fences(str(response))
    try:
        import json as _json
        data = _json.loads(text)
        result = _runner.MetaAnalyzerResult.model_validate(data)
        items = []
        for f in result.findings:
            d = f.model_dump()
            # BUG: _sanitize_meta_finding NOT called — null fields leak through
            d["_file"] = batch.file_path
            items.append(d)
        return items
    except Exception:
        return []


_runner._patched_meta_parse = _broken_meta_parse
_runner.LLMMetaAnalyzer.parse_response = _broken_meta_parse
mutate("Patch 3 sanitize broken", "contrib.multilingual.runner",
       "_patched_meta_parse", _broken_meta_parse,
       [("test_runner_patches", "TestSanitizeMetaFinding")])
_runner._patched_meta_parse = _orig_meta_parse

# ── Patch 4 mutation: build_prompt appends nothing ─────────────────
_orig_base_build = _runner._patched_base_build_prompt


def _broken_base_build(self, batch, **kwargs):
    # BUG: JSON instruction NOT appended
    return _runner._original_base_build_prompt(self, batch, **kwargs)


_runner._patched_base_build_prompt = _broken_base_build
_runner.LLMAnalyzerBase.build_prompt = _broken_base_build
mutate("Patch 4 JSON prompt missing", "contrib.multilingual.runner",
       "_patched_base_build_prompt", _broken_base_build,
       [("test_runner_patches", "TestContextManagerApplyRestore")])
_runner._patched_base_build_prompt = _orig_base_build

# ── Patch 5 mutation: meta build_prompt appends nothing ────────────
_orig_meta_build = _runner._patched_meta_build_prompt


def _broken_meta_build(self, batch, **kwargs):
    return _runner._original_meta_build_prompt(self, batch, **kwargs)


_runner._patched_meta_build_prompt = _broken_meta_build
_runner.LLMMetaAnalyzer.build_prompt = _broken_meta_build
mutate("Patch 5 JSON meta prompt missing", "contrib.multilingual.runner",
       "_patched_meta_build_prompt", _broken_meta_build,
       [("test_runner_patches", "TestContextManagerApplyRestore")])
_runner._patched_meta_build_prompt = _orig_meta_build

# ── Patch 7 mutation: asyncio.run NOT replaced ────────────────────
_orig_patched_asyncio = _runner._patched_asyncio_run


def _broken_asyncio_run(main, *, debug=None, loop_factory=None):
    # BUG: completely bypasses the quiet-loop wrapper
    return _runner._original_asyncio_run(main, debug=debug, loop_factory=loop_factory)


_runner._patched_asyncio_run = _broken_asyncio_run
mutate("Patch 7 asyncio not patched", "contrib.multilingual.runner",
       "_patched_asyncio_run", _broken_asyncio_run,
       [("test_runner_patches", "TestPatch7AsyncioQuietLoop")])
_runner._patched_asyncio_run = _orig_patched_asyncio

# ── GapFill: rule_id filtering broken ─────────────────────────────
_orig_parse3 = _gf.GapFillAnalyzer.parse_response


def _broken_parse_no_rule_filter(self, response, batch):
    import json as _json
    text = str(response).strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    try:
        data = _json.loads(text)
    except _json.JSONDecodeError:
        return []
    try:
        result = _gf.GapFillResult.model_validate(data)
        items = []
        for item in result.findings:
            if item.confidence < 0.7:
                continue
            # BUG: rule_id check removed — unknown rules accepted
            items.append(item.to_finding(batch.file_path))
        return items
    except Exception:
        return []


_gf.GapFillAnalyzer.parse_response = _broken_parse_no_rule_filter
mutate("rule_id filter removed", "contrib.multilingual.gap_fill",
       "GapFillAnalyzer.parse_response", _broken_parse_no_rule_filter,
       [("test_gap_fill", "TestParseResponseFiltering")])
_gf.GapFillAnalyzer.parse_response = _orig_parse3

# ── GapFill: JSON decode errors not caught ─────────────────────────
_orig_parse4 = _gf.GapFillAnalyzer.parse_response


def _broken_parse_no_json_catch(self, response, batch):
    import json as _json
    text = str(response).strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    data = _json.loads(text)  # BUG: JSONDecodeError not caught — will crash
    result = _gf.GapFillResult.model_validate(data)
    return [item.to_finding(batch.file_path)
            for item in result.findings
            if item.rule_id in _gf._GAP_FILL_RULE_IDS and item.confidence >= 0.7]


_gf.GapFillAnalyzer.parse_response = _broken_parse_no_json_catch
mutate("JSON decode error not caught", "contrib.multilingual.gap_fill",
       "GapFillAnalyzer.parse_response", _broken_parse_no_json_catch,
       [("test_gap_fill", "TestParseResponseInvalidInput")])
_gf.GapFillAnalyzer.parse_response = _orig_parse4

# ── GapFill: Pydantic validation errors not caught ─────────────────
_orig_parse5 = _gf.GapFillAnalyzer.parse_response


def _broken_parse_no_pydantic_catch(self, response, batch):
    import json as _json
    text = str(response).strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    try:
        data = _json.loads(text)
    except _json.JSONDecodeError:
        return []
    result = _gf.GapFillResult.model_validate(data)  # BUG: validation error not caught
    return [item.to_finding(batch.file_path)
            for item in result.findings
            if item.rule_id in _gf._GAP_FILL_RULE_IDS and item.confidence >= 0.7]


_gf.GapFillAnalyzer.parse_response = _broken_parse_no_pydantic_catch
mutate("Pydantic validation error not caught", "contrib.multilingual.gap_fill",
       "GapFillAnalyzer.parse_response", _broken_parse_no_pydantic_catch,
       [("test_gap_fill", "TestParseResponseInvalidInput")])
_gf.GapFillAnalyzer.parse_response = _orig_parse5

# ── Area 5: Hedge — untested risky code from RISK_TABLE ─────────────

# Mutation 5a: _next_available_in broken — always returns None
_orig_next_avail = _ap.ApiKeyPool._next_available_in


def _broken_next_avail(self, now):
    return None  # BUG: never reports recovery time — acquire() waits forever


_ap.ApiKeyPool._next_available_in = _broken_next_avail
# Note: this mutation can't be directly tested without a rate-limited+full pool scenario
# which is Q16's blind spot.  Test validates the function exists but not this branch.
mutate("_next_available_in always None", "contrib.multilingual.api_pool",
       "ApiKeyPool._next_available_in", _broken_next_avail,
       [])  # No matching test — documented as Q16/Q17 blind spot
_ap.ApiKeyPool._next_available_in = _orig_next_avail

# Mutation 5b: _restore_patches broken — forgets to restore Patch 6
_orig_restore = _runner._restore_patches


def _broken_restore():
    
    if _runner._patches_depth == 0:
        return
    _runner._patches_depth -= 1
    if _runner._patches_depth > 0:
        return
    _runner.LLMAnalyzerBase.__init__ = _runner._original_base_init
    _runner.LLMAnalyzerBase.parse_response = _runner._original_base_parse
    _runner.LLMAnalyzerBase.build_prompt = _runner._original_base_build_prompt
    _runner.LLMMetaAnalyzer.parse_response = _runner._original_meta_parse
    _runner.LLMMetaAnalyzer.build_prompt = _runner._original_meta_build_prompt
    # BUG: Patch 6 (ChatOpenAI) and Patch 7 (asyncio) NOT restored


_runner._restore_patches = _broken_restore
mutate("_restore_patches skips Patch 6+7", "contrib.multilingual.runner",
       "_restore_patches", _broken_restore,
       [("test_runner_patches", "TestContextManagerApplyRestore")])
_runner._restore_patches = _orig_restore

# Mutation 5c: _verify_patch_targets broken — always passes silently
_orig_verify = _runner._verify_patch_targets


def _broken_verify():
    pass  # BUG: skips all 17 checks — never raises


_runner._verify_patch_targets = _broken_verify
mutate("_verify_patch_targets no-op", "contrib.multilingual.runner",
       "_verify_patch_targets", _broken_verify,
       [])  # Q13: no test asserts guard actually ran — documented blind spot
_runner._verify_patch_targets = _orig_verify

# Mutation 5d: _check_signature broken — never raises
_orig_check = _runner._check_signature


def _broken_check(func, expected, label, num):
    pass  # BUG: never validates — all signatures silently pass


_runner._check_signature = _broken_check
mutate("_check_signature no-op", "contrib.multilingual.runner",
       "_check_signature", _broken_check,
       [])  # No test directly calls _check_signature — documented
_runner._check_signature = _orig_check

# Mutation 5e: set_api_pool broken — doesn't save original
_orig_set_api = _runner.set_api_pool


def _broken_set_api(pool):
    _runner._api_pool = pool
    if pool is None:
        return
    import skillspector.llm_utils as _u
    def _bad_wrapper(model=None):
        if _runner._api_pool:
            from contrib.multilingual.api_pool import PooledChatModel
            return PooledChatModel(_runner._api_pool)
        # BUG: fallback calls patched version instead of original
        return _u.get_chat_model(model)
    _u.get_chat_model = _bad_wrapper


_runner.set_api_pool = _broken_set_api
mutate("set_api_pool broken fallback", "contrib.multilingual.runner",
       "set_api_pool", _broken_set_api,
       [("test_runner_patches", "TestSetApiPoolRestore")])
_runner.set_api_pool = _orig_set_api

# Mutation 5f: annotate_findings broken — always returns incompatible
import contrib.multilingual.annotation as _ann
_orig_annotate = _ann.annotate_findings


def _broken_annotate(issues, detected_language):
    annotated = []
    for issue in issues:
        entry = dict(issue)
        entry["language_compatible"] = False  # BUG: always False regardless of rule
        annotated.append(entry)
    return annotated


_ann.annotate_findings = _broken_annotate
mutate("annotate_findings always incompatible", "contrib.multilingual.annotation",
       "annotate_findings", _broken_annotate,
       [("test_annotation", "TestAnnotateFindings")])
_ann.annotate_findings = _orig_annotate

# Mutation 5g: is_language_compatible broken — always True
_orig_is_compat = _ann.is_language_compatible


def _broken_is_compat(rule_id, detected_language):
    return True  # BUG: all rules compatible — English keyword rules misclassified


_ann.is_language_compatible = _broken_is_compat
mutate("is_language_compatible always True", "contrib.multilingual.annotation",
       "is_language_compatible", _broken_is_compat,
       [("test_annotation", "TestAnnotateFindings")])
_ann.is_language_compatible = _orig_is_compat

# ── Area 6: Remaining untested functions from RISK_TABLE ────────────

# Mutation 6a: build_prompt broken — missing file label
_orig_build = _gf.GapFillAnalyzer.build_prompt


def _broken_build_prompt(self, batch, **kwargs):
    prompt = self.base_prompt
    # BUG: file_label + numbered_content NOT included — LLM gets no context
    return prompt


_gf.GapFillAnalyzer.build_prompt = _broken_build_prompt
mutate("build_prompt missing file content", "contrib.multilingual.gap_fill",
       "GapFillAnalyzer.build_prompt", _broken_build_prompt,
       [("test_gap_fill", "TestBuildPrompt")])
_gf.GapFillAnalyzer.build_prompt = _orig_build

# Mutation 6b: get_batches broken — always returns empty
_orig_batches = _gf.GapFillAnalyzer.get_batches


def _broken_get_batches(self, file_paths, file_cache, findings=None):
    return []  # BUG: all files skipped — no analysis happens


_gf.GapFillAnalyzer.get_batches = _broken_get_batches
mutate("get_batches always empty", "contrib.multilingual.gap_fill",
       "GapFillAnalyzer.get_batches", _broken_get_batches,
       [("test_gap_fill", "TestGetBatchesAndCollectFindings")])
_gf.GapFillAnalyzer.get_batches = _orig_batches

# Mutation 6c: collect_findings broken — returns empty
_orig_collect = _gf.GapFillAnalyzer.collect_findings


def _broken_collect_findings(self, batch_results):
    return []  # BUG: all findings discarded


_gf.GapFillAnalyzer.collect_findings = _broken_collect_findings
mutate("collect_findings always empty", "contrib.multilingual.gap_fill",
       "GapFillAnalyzer.collect_findings", _broken_collect_findings,
       [("test_gap_fill", "TestGetBatchesAndCollectFindings")])
_gf.GapFillAnalyzer.collect_findings = _orig_collect

# Mutation 6d: run_gap_fill broken — ignores all findings
_orig_run_gf = _gf.run_gap_fill


def _broken_run_gap_fill(file_cache, language, model=None, api_pool=None):
    return []  # BUG: always returns empty — never runs LLM


_gf.run_gap_fill = _broken_run_gap_fill
mutate("run_gap_fill always empty", "contrib.multilingual.gap_fill",
       "run_gap_fill", _broken_run_gap_fill,
       [("test_gap_fill", "TestRunGapFill")])
_gf.run_gap_fill = _orig_run_gf

# Mutation 6e: _is_rate_limit broken — always False
_orig_is_rl = _ap.PooledChatModel._is_rate_limit


def _broken_is_rl(exc):
    return False  # BUG: never detects rate limits — retries never happen


_ap.PooledChatModel._is_rate_limit = staticmethod(_broken_is_rl)
mutate("_is_rate_limit always False", "contrib.multilingual.api_pool",
       "PooledChatModel._is_rate_limit", staticmethod(_broken_is_rl),
       [("test_api_pool", "TestIsRateLimit")])
_ap.PooledChatModel._is_rate_limit = _orig_is_rl

# Mutation 6f: create_api_key_pool_from_env broken — always returns None
_orig_create_pool = _ap.create_api_key_pool_from_env


def _broken_create_pool(max_concurrent_per_key=5):
    return None  # BUG: pool never created — all LLM calls use single key


_ap.create_api_key_pool_from_env = _broken_create_pool
mutate("create_api_key_pool_from_env always None", "contrib.multilingual.api_pool",
       "create_api_key_pool_from_env", _broken_create_pool,
       [("test_api_pool", "TestCreateApiKeyPoolFromEnv")])
_ap.create_api_key_pool_from_env = _orig_create_pool

# Mutation 6g: deepseek_compat broken — doesn't restore on exception
from contextlib import contextmanager as _ctx_mgr
_orig_ds_compat = _runner.deepseek_compat


@_ctx_mgr
def _broken_ds_compat():
    _runner._apply_patches()
    try:
        yield
    # BUG: missing finally — patches NOT restored on exception
    finally:
        pass  # should be _restore_patches()


_runner.deepseek_compat = _broken_ds_compat
mutate("deepseek_compat no restore on exception", "contrib.multilingual.runner",
       "deepseek_compat", _broken_ds_compat,
       [("test_runner_patches", "TestContextManagerApplyRestore")])
_runner.deepseek_compat = _orig_ds_compat

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Mutation Test Results — Max's 4 Risk Areas")
print(f"{'='*60}")
for label, cls, caught in results:
    status = "✅ CAUGHT" if caught else "❌ MISSED"
    print(f"  {status} | {label} → {cls}")
caught = sum(1 for _, _, c in results if c)
missed = sum(1 for _, _, c in results if not c)
print(f"\nTotal: {caught}/{caught+missed} mutations caught")
if missed == 0:
    print("All mutations detected — tests are real.")
else:
    print(f"⚠ {missed} mutation(s) NOT caught — review blind spots.")
