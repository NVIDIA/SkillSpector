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

"""Graph invocation helpers for batch scanning.

Thin wrappers over ``skillspector.graph.graph`` — build initial state,
invoke the graph, and transform the raw result dict into a structured
batch entry suitable for downstream reporting.

Compatibility patches (DeepSeek / non-OpenAI providers)
-------------------------------------------------------
Call :func:`setup_deepseek_compat` before any LLM activity to apply
seven targeted monkey-patches that make the core analyzers work with
providers that lack structured-output (``response_format``) support.
The patches must be applied exactly once, before the first
``graph.invoke`` call.  Importing this module does NOT apply them
automatically — the caller controls when they take effect.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from skillspector.graph import graph
from skillspector.llm_analyzer_base import LLMAnalyzerBase, LLMAnalysisResult
from skillspector.logging_config import get_logger
from skillspector.nodes.meta_analyzer import LLMMetaAnalyzer, MetaAnalyzerResult

from .annotation import annotate_findings

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# API Key Pool — shared across graph-internal and gap-fill LLM calls
# ═══════════════════════════════════════════════════════════════════════════

_api_pool: "ApiKeyPool | None" = None

_original_get_chat_model = None  # saved on first set_api_pool call


def set_api_pool(pool: "ApiKeyPool | None") -> None:
    """Replace the LLM chat-model factory with a pooled version.

    When *pool* is set, every call to :func:`skillspector.llm_utils.get_chat_model`
    returns a :class:`~.api_pool.PooledChatModel` instance backed by the shared
    key pool.  This covers both graph-internal analyzers (20 per skill) and the
    gap-fill pass — every LLM call in the batch scan goes through the pool.

    Call ``set_api_pool(None)`` to restore the original factory.
    """
    global _api_pool, _original_get_chat_model

    import skillspector.llm_utils as _llm_utils

    if pool is None:
        _api_pool = None
        if _original_get_chat_model is not None:
            _llm_utils.get_chat_model = _original_get_chat_model
            _original_get_chat_model = None
            logger.info("API key pool removed — restored original get_chat_model")
        return

    _api_pool = pool
    if _original_get_chat_model is None:
        _original_get_chat_model = _llm_utils.get_chat_model

    def _pooled_get_chat_model(model=None):
        if _api_pool:
            from .api_pool import PooledChatModel
            return PooledChatModel(_api_pool)
        return _original_get_chat_model(model)

    _llm_utils.get_chat_model = _pooled_get_chat_model
    logger.info("API key pool wired — all LLM calls will use PooledChatModel")

# ═══════════════════════════════════════════════════════════════════════════
# HTTP timeout — stop hung connections from blocking workers forever
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_REQUEST_TIMEOUT = 30.0  # total request ceiling
_DEFAULT_CONNECT_TIMEOUT = 8.0   # TCP / TLS handshake

# ═══════════════════════════════════════════════════════════════════════════
# Compatibility patches (DeepSeek / non-OpenAI providers)
# ═══════════════════════════════════════════════════════════════════════════
#
# These patches are NOT applied at import time.  Call :func:`setup_deepseek_compat`
# before any LLM activity to activate them.  Each patch can only be applied once;
# subsequent calls are no-ops.

_patches_depth: int = 0  # nesting counter — safe for re-entrant context managers

# -- Patch 1: inject response_schema=None as instance attribute ------------
# We set response_schema=None on the *instance* dict before the original
# __init__ runs.  Python MRO always checks instance.__dict__ before
# class.__dict__ — this is a language-level guarantee (not a library
# internal).  The instance dict takes precedence regardless of how the
# upstream class hierarchy evolves, so this patch is safe against
# upstream refactors.
_original_base_init = LLMAnalyzerBase.__init__


def _patched_base_init(self, base_prompt, model):
    """Set response_schema=None on the instance dict BEFORE original init.

    Relies on Python MRO guarantee: instance.__dict__ is always checked
    before any class-level attribute.  This is language semantics, not
    a library internal.
    """
    self.response_schema = None
    _original_base_init(self, base_prompt, model)


# -- Patch 2: LLMAnalyzerBase.parse_response handles raw JSON --------------
_original_base_parse = LLMAnalyzerBase.parse_response


def _patched_base_parse(self, response, batch):
    """Parse raw LLM text into Findings via manual JSON + Pydantic."""
    if isinstance(response, LLMAnalysisResult):
        return _original_base_parse(self, response, batch)
    text = _strip_markdown_fences(str(response))
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "LLMAnalyzerBase.parse_response: invalid JSON for %s: %s",
            batch.file_label,
            exc,
        )
        return []
    try:
        result = LLMAnalysisResult.model_validate(data)
        return [f.to_finding(batch.file_path) for f in result.findings]
    except Exception as exc:
        logger.warning(
            "LLMAnalyzerBase.parse_response: schema validation failed for %s: %s",
            batch.file_label,
            exc,
        )
        return []


# -- Patch 3: LLMMetaAnalyzer.parse_response handles raw JSON ---------------
_original_meta_parse = LLMMetaAnalyzer.parse_response


def _sanitize_meta_finding(d: dict) -> dict:
    """Fix common LLM output quirks that break downstream consumers."""
    for key in ("remediation", "explanation"):
        if d.get(key) is None:
            d[key] = ""
    if d.get("impact") not in ("critical", "high", "medium", "low"):
        d["impact"] = "low"
    return d


def _patched_meta_parse(self, response, batch):
    """Parse raw LLM text into meta-analyzer dicts via manual JSON + Pydantic."""
    if isinstance(response, MetaAnalyzerResult):
        return _original_meta_parse(self, response, batch)
    text = _strip_markdown_fences(str(response))
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "LLMMetaAnalyzer.parse_response: invalid JSON for %s: %s",
            batch.file_label,
            exc,
        )
        return []
    try:
        result = MetaAnalyzerResult.model_validate(data)
        items = []
        for f in result.findings:
            d = _sanitize_meta_finding(f.model_dump())
            d["_file"] = batch.file_path
            items.append(d)
        return items
    except Exception as exc:
        logger.warning(
            "LLMMetaAnalyzer.parse_response: schema validation failed for %s: %s",
            batch.file_label,
            exc,
        )
        return []


# -- Patch 4: append JSON output format to base prompt ---------------------
_JSON_OUTPUT_INSTRUCTION = (
    "\n\nRespond with ONLY a JSON object (no markdown, no explanation):\n"
    '{"findings": [{"rule_id": "...", "message": "...", '
    '"severity": "LOW|MEDIUM|HIGH|CRITICAL", "start_line": 1, '
    '"end_line": null, "confidence": 0.0-1.0, '
    '"explanation": "...", "remediation": "..."}]}\n'
    "If no issues found, return: {\"findings\": []}"
)

_original_base_build_prompt = LLMAnalyzerBase.build_prompt


def _patched_base_build_prompt(self, batch, **kwargs):
    prompt = _original_base_build_prompt(self, batch, **kwargs)
    return prompt + _JSON_OUTPUT_INSTRUCTION


# -- Patch 5: append JSON format to meta-analyzer prompt -------------------
_original_meta_build_prompt = LLMMetaAnalyzer.build_prompt

_META_JSON_PROMPT = (
    "\n\nRespond with ONLY a JSON object (no markdown):\n"
    '{"findings": [{"pattern_id": "...", "is_vulnerability": true|false, '
    '"confidence": 0.0-1.0, "intent": "malicious|negligent|benign", '
    '"impact": "critical|high|medium|low", '
    '"explanation": "...", "remediation": "..."}], '
    '"overall_assessment": {"risk_level": "LOW|MEDIUM|HIGH|CRITICAL", '
    '"summary": "..."}}\n'
    'Rules: never use null — use "" for empty strings. '
    'Never use "none" for impact — use "low" for negligible. '
    'If no findings: {"findings": [], '
    '"overall_assessment": {"risk_level": "LOW", "summary": "No issues found"}}'
)


def _patched_meta_build_prompt(self, batch, **kwargs):
    prompt = _original_meta_build_prompt(self, batch, **kwargs)
    return prompt + _META_JSON_PROMPT


# -- Patch 6: enforce HTTP-level timeouts on all ChatOpenAI instances ------
# Capture at module-load time to avoid order-dependency (any prior import that
# patches ChatOpenAI would corrupt the capture inside _apply_patches).
try:
    from langchain_openai import ChatOpenAI as _CO_for_original
    _original_chatopenai_init = _CO_for_original.__init__
except ImportError:
    _original_chatopenai_init = None


def _patched_chatopenai_init(self, **kwargs):
    import httpx

    _to = httpx.Timeout(
        _DEFAULT_REQUEST_TIMEOUT,
        connect=_DEFAULT_CONNECT_TIMEOUT,
    )
    # Set both the Pydantic alias AND the canonical field name so we don't
    # depend on alias-precedence behaviour (which is a Pydantic v2 internal).
    kwargs["timeout"] = _to
    kwargs["request_timeout"] = _to
    _original_chatopenai_init(self, **kwargs)


# -- Patch 7: silence "Event loop is closed" noise from httpx cleanup ------
import asyncio as _asyncio

_original_asyncio_run = _asyncio.run


def _patched_asyncio_run(main, *, debug=None, loop_factory=None):
    def _make_quiet_loop():
        loop = (loop_factory or _asyncio.new_event_loop)()
        def _handler(loop, context):
            exc = context.get("exception")
            if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
                return
            loop.default_exception_handler(context)
        loop.set_exception_handler(_handler)
        return loop
    return _original_asyncio_run(main, debug=debug, loop_factory=_make_quiet_loop)


def setup_deepseek_compat() -> None:
    """Apply DeepSeek compatibility patches permanently (convenience wrapper).

    Prefer :func:`deepseek_compat` context manager for scoped, reversible
    patching.  This function is a one-way door — patches stay for the
    process lifetime.
    """
    _apply_patches()


def _verify_patch_targets() -> None:
    """Verify that all patch targets have expected signatures / attributes.

    Raises :class:`RuntimeError` with a specific message if an upstream
    change has broken one of the assumptions our patches depend on.
    This turns a silent, hard-to-debug failure into an immediate, clear
    error at patch-application time.

    Covers both surface-level (function signatures) and deep dependencies
    (methods called inside try/except that could silently degrade).
    """
    import dataclasses
    import inspect

    from skillspector.llm_analyzer_base import Batch, LLMFinding

    # -- Patch 1: LLMAnalyzerBase.__init__(self, base_prompt, model) ---------
    _check_signature(
        LLMAnalyzerBase.__init__,
        ["self", "base_prompt", "model"],
        "LLMAnalyzerBase.__init__",
        1,
    )
    if not hasattr(LLMAnalyzerBase, "response_schema"):
        raise RuntimeError(
            "Patch 1 target lost: LLMAnalyzerBase no longer has "
            "'response_schema' class attribute.  Upstream may have renamed "
            "or removed it."
        )

    # -- Patch 2: LLMAnalyzerBase.parse_response(self, response, batch) ------
    _check_signature(
        LLMAnalyzerBase.parse_response,
        ["self", "response", "batch"],
        "LLMAnalyzerBase.parse_response",
        2,
    )
    # Deep deps (called inside try/except — silent degradation if broken):
    if not hasattr(LLMAnalysisResult, "model_validate"):
        raise RuntimeError(
            "Patch 2 deep dependency lost: LLMAnalysisResult.model_validate "
            "no longer exists.  Upstream may have switched from Pydantic v2 "
            "to a different validation library."
        )
    if not hasattr(LLMFinding, "to_finding"):
        raise RuntimeError(
            "Patch 2 deep dependency lost: LLMFinding.to_finding method "
            "no longer exists.  Upstream may have renamed or removed it."
        )
    # Batch is a @dataclass — file_path is a field, file_label is a @property
    _batch_field_names = {f.name for f in dataclasses.fields(Batch)}
    if "file_path" not in _batch_field_names:
        raise RuntimeError(
            "Patch 2 deep dependency lost: Batch dataclass no longer has "
            "'file_path' field.  Upstream may have changed the Batch dataclass."
        )
    if "file_label" not in {n for n in dir(Batch) if isinstance(getattr(Batch, n, None), property)}:
        raise RuntimeError(
            "Patch 2 deep dependency lost: Batch no longer has 'file_label' "
            "property.  Upstream may have renamed or removed it."
        )

    # -- Patch 3: LLMMetaAnalyzer.parse_response(self, response, batch) ------
    _check_signature(
        LLMMetaAnalyzer.parse_response,
        ["self", "response", "batch"],
        "LLMMetaAnalyzer.parse_response",
        3,
    )
    if not hasattr(MetaAnalyzerResult, "model_validate"):
        raise RuntimeError(
            "Patch 3 deep dependency lost: MetaAnalyzerResult.model_validate "
            "no longer exists.  Upstream may have switched from Pydantic v2."
        )
    # Pydantic models don't expose fields as class attributes — use
    # model_fields (v2) or __fields__ (v1 fallback).
    _mr_fields = getattr(MetaAnalyzerResult, "model_fields", None) or getattr(
        MetaAnalyzerResult, "__fields__", {}
    )
    if "findings" not in _mr_fields:
        raise RuntimeError(
            "Patch 3 deep dependency lost: MetaAnalyzerResult no longer has "
            "'findings' field.  Upstream may have changed the Pydantic schema."
        )

    # -- Patch 4: LLMAnalyzerBase.build_prompt(self, batch, **kwargs) --------
    sig4 = inspect.signature(LLMAnalyzerBase.build_prompt)
    if "batch" not in sig4.parameters:
        raise RuntimeError(
            "Patch 4 target changed: LLMAnalyzerBase.build_prompt no longer "
            "accepts 'batch' parameter.  Upstream may have changed the API."
        )
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig4.parameters.values()):
        raise RuntimeError(
            "Patch 4 target changed: LLMAnalyzerBase.build_prompt no longer "
            "accepts **kwargs.  Upstream may have changed the API."
        )

    # -- Patch 5: LLMMetaAnalyzer.build_prompt(self, batch, **kwargs) --------
    sig5 = inspect.signature(LLMMetaAnalyzer.build_prompt)
    if "batch" not in sig5.parameters:
        raise RuntimeError(
            "Patch 5 target changed: LLMMetaAnalyzer.build_prompt no longer "
            "accepts 'batch' parameter.  Upstream may have changed the API."
        )

    # -- Patch 6: ChatOpenAI.__init__ — must accept **kwargs -----------------
    try:
        from langchain_openai import ChatOpenAI as _ChatOpenAI

        sig6 = inspect.signature(_ChatOpenAI.__init__)
        if not any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig6.parameters.values()
        ):
            raise RuntimeError(
                "Patch 6 target changed: ChatOpenAI.__init__ no longer "
                "accepts **kwargs.  Upstream may have removed the Pydantic "
                "alias or switched to a non-Pydantic model."
            )
    except ImportError:
        pass  # langchain_openai not available — Patch 6 is skipped anyway

    # -- Patch 7: asyncio.run(main, *, debug=None, loop_factory=None) --------
    # Only 'main' is positional; debug/loop_factory are keyword-only by design.
    _check_signature(
        _original_asyncio_run,
        ["main"],
        "asyncio.run",
        7,
    )
    # Deep dep: new_event_loop() is used inside _make_quiet_loop
    if not callable(getattr(_asyncio, "new_event_loop", None)):
        raise RuntimeError(
            "Patch 7 deep dependency lost: asyncio.new_event_loop is no "
            "longer available.  Python version may have changed the API."
        )

    logger.debug("All 7 patch targets verified — upstream API matches expectations")


def _check_signature(
    func: object,
    expected_params: list[str],
    label: str,
    patch_num: int,
) -> None:
    """Raise :class:`RuntimeError` if *func* doesn't accept *expected_params*."""
    import inspect

    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Patch {patch_num} target unavailable: cannot inspect {label} "
            f"signature.  Upstream may have changed the API.  ({exc})"
        ) from exc

    for param in expected_params:
        if param not in sig.parameters:
            raise RuntimeError(
                f"Patch {patch_num} target changed: {label} no longer has "
                f"'{param}' parameter.  Upstream may have changed the API."
            )
        # Guard against keyword-only migration: if a parameter we pass
        # positionally becomes keyword-only, our call sites break.
        _kind = sig.parameters[param].kind
        if _kind == inspect.Parameter.KEYWORD_ONLY:
            raise RuntimeError(
                f"Patch {patch_num} target changed: {label} parameter "
                f"'{param}' is now keyword-only (was positional).  Upstream "
                f"may have changed the API."
            )


def _apply_patches() -> None:
    """Apply all 7 compatibility patches (idempotent — safe to nest).

    Uses a nesting counter instead of a boolean flag so that nested
    ``with deepseek_compat()`` blocks don't restore on the inner exit.
    """
    global _patches_depth
    if _patches_depth > 0:
        _patches_depth += 1
        return

    _verify_patch_targets()

    LLMAnalyzerBase.__init__ = _patched_base_init
    LLMAnalyzerBase.parse_response = _patched_base_parse
    LLMAnalyzerBase.build_prompt = _patched_base_build_prompt

    LLMMetaAnalyzer.parse_response = _patched_meta_parse
    LLMMetaAnalyzer.build_prompt = _patched_meta_build_prompt

    try:
        import httpx
        from langchain_openai import ChatOpenAI as _ChatOpenAI

        _ChatOpenAI.__init__ = _patched_chatopenai_init
    except ImportError:
        logger.debug("httpx not available — skipping ChatOpenAI timeout patch")

    _asyncio.run = _patched_asyncio_run

    _patches_depth = 1
    logger.debug("DeepSeek compatibility patches applied (7 patches)")


def _restore_patches() -> None:
    """Restore all original class methods / functions (nesting-aware).

    Only actually restores when the outermost context manager exits
    (_patches_depth reaches 0).
    """
    global _patches_depth
    if _patches_depth == 0:
        return  # not active
    _patches_depth -= 1
    if _patches_depth > 0:
        return  # still nested — don't restore yet

    LLMAnalyzerBase.__init__ = _original_base_init
    LLMAnalyzerBase.parse_response = _original_base_parse
    LLMAnalyzerBase.build_prompt = _original_base_build_prompt

    LLMMetaAnalyzer.parse_response = _original_meta_parse
    LLMMetaAnalyzer.build_prompt = _original_meta_build_prompt

    if _original_chatopenai_init is not None:
        try:
            from langchain_openai import ChatOpenAI as _ChatOpenAI
            _ChatOpenAI.__init__ = _original_chatopenai_init
        except ImportError:
            pass

    _asyncio.run = _original_asyncio_run

    logger.debug("DeepSeek compatibility patches restored to originals")


# ---------------------------------------------------------------------------
# Context manager — scoped, reversible patching (Python best practice)
# ---------------------------------------------------------------------------
# Pattern: Save → Patch → Yield → Restore (finally-guaranteed)
# Reference: unittest.mock.patch, pytest.monkeypatch.context(), gevent.monkey


from contextlib import contextmanager


@contextmanager
def deepseek_compat():
    """Context manager that applies DeepSeek compatibility patches and
    restores original state on exit — even if an exception occurs.

    Usage::

        with deepseek_compat():
            # All 7 patches active inside this block
            batch_scan(tests/fixtures)

        # Outside the block: everything restored to original

    Patches applied (same 7 as :func:`setup_deepseek_compat`):
    1. ``LLMAnalyzerBase.__init__`` — inject ``response_schema=None``
    2. ``LLMAnalyzerBase.parse_response`` — manual JSON parsing
    3. ``LLMMetaAnalyzer.parse_response`` — manual JSON + field sanitize
    4. ``LLMAnalyzerBase.build_prompt`` — append JSON output instruction
    5. ``LLMMetaAnalyzer.build_prompt`` — append JSON output instruction
    6. ``ChatOpenAI.__init__`` — enforce HTTP-level timeouts
    7. ``asyncio.run`` — suppress "Event loop is closed" noise
    """
    _apply_patches()
    try:
        yield
    finally:
        _restore_patches()


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    return text.strip()


def scan_state(skill_dir: Path, use_llm: bool) -> dict[str, object]:
    """Build the initial LangGraph state for a single skill directory."""
    return {
        "input_path": str(skill_dir),
        "output_format": "json",
        "use_llm": use_llm,
    }


def cleanup_result(result: dict[str, object]) -> None:
    """Remove the temporary directory created by the graph, if any."""
    temp_dir = result.get("temp_dir_for_cleanup")
    if not temp_dir or not isinstance(temp_dir, str):
        return
    shutil.rmtree(temp_dir, ignore_errors=True)


# Number of English-keyword static rules that lose recall for non-English skills.
# These 25 rules are documented in annotation._ENGLISH_KEYWORD_RULES.
_ENGLISH_KEYWORD_RULE_COUNT = 25


def entry_from_result(
    result: dict[str, object],
    skill_dir: Path,
    root: Path,
    *,
    detected_language: str = "en",
    gap_fill_applied: bool = False,
    gap_fill_findings: int = 0,
) -> dict[str, object]:
    """Convert a raw ``graph.invoke()`` result into a batch-report entry.

    Extracts findings, manifest metadata, component metadata, and builds
    the canonical ``skill / risk_assessment / components / issues`` shape
    used by report formatters.  Adds ``source_group``, ``language``,
    ``scan_mode``, and ``enhancements`` fields for provenance tracking
    and comparability with the standard single-skill scan.

    Parameters
    ----------
    result :
        Raw dict returned by ``graph.invoke(state)``.
    skill_dir :
        The skill directory that was scanned.
    root :
        Root directory for relative-path computation.
    detected_language :
        Language detected for this skill (``"en"``, ``"zh"``, etc.).
    gap_fill_applied :
        ``True`` when the gap-fill LLM pass has been applied.
    gap_fill_findings :
        Number of gap-fill findings appended to the issues list.
    """
    findings = result.get("filtered_findings", result.get("findings", []))
    manifest = result.get("manifest") or {}
    component_metadata = result.get("component_metadata") or []
    skill_name = (
        (manifest.get("name") or skill_dir.name) if manifest else skill_dir.name
    )

    try:
        rel_path = str(skill_dir.relative_to(root))
    except ValueError:
        rel_path = str(skill_dir)

    source_group = rel_path.split("/")[0] if "/" in rel_path else "."

    raw_issues: list[dict[str, object]]
    if findings and hasattr(findings[0], "to_dict"):
        raw_issues = [f.to_dict() for f in findings]  # type: ignore[union-attr]
    elif findings:
        raw_issues = list(findings)  # type: ignore[assignment]
    else:
        raw_issues = []

    issues = annotate_findings(raw_issues, detected_language)
    is_non_en = detected_language != "en"

    return {
        "skill": {
            "name": skill_name,
            "source": rel_path,
            "source_group": source_group,
            "language": detected_language,
            "scanned_at": datetime.now(UTC).isoformat(),
        },
        "risk_assessment": {
            "score": result.get("risk_score", 0),
            "severity": result.get("risk_severity", "LOW"),
            "recommendation": (result.get("risk_recommendation") or "SAFE").replace(
                "_", " "
            ),
        },
        "components": [
            {
                "path": c.get("path"),
                "type": c.get("type"),
                "lines": c.get("lines"),
                "executable": c.get("executable"),
                "size_bytes": c.get("size_bytes"),
            }
            for c in component_metadata  # type: ignore[union-attr]
        ],
        "issues": issues,
        "scan_mode": "multilingual-enhanced",
        "enhancements": {
            "gap_fill_applied": gap_fill_applied,
            "gap_fill_findings": gap_fill_findings,
            "english_keyword_rules_skipped": (
                _ENGLISH_KEYWORD_RULE_COUNT if is_non_en else 0
            ),
        },
    }


def run_one(
    skill_dir: Path,
    root: Path,
    *,
    use_llm: bool,
    detected_language: str = "en",
    gap_fill_applied: bool = False,
    gap_fill_findings: int = 0,
) -> tuple[dict[str, object], str | None]:
    """Scan a single skill through the full graph pipeline.

    Parameters
    ----------
    skill_dir :
        Path to the skill directory.
    root :
        Root directory for relative-path computation in reports.
    use_llm :
        Passed through to the graph as ``state["use_llm"]``.
    detected_language :
        Language tag for annotation and reporting.
    gap_fill_applied :
        ``True`` when the caller has applied gap-fill (set by
        :func:`~.batch_scan._scan_skill` after the graph returns).
    gap_fill_findings :
        Number of gap-fill findings appended post-graph.

    Returns
    -------
    ``(entry, error_message_or_None)`` — on success *error_message*
    is ``None``; on failure *entry* is a stub error entry and
    *error_message* carries the exception text.
    """
    result = None
    try:
        state = scan_state(skill_dir, use_llm=use_llm)
        result = graph.invoke(state)
        entry = entry_from_result(
            result,
            skill_dir,
            root,
            detected_language=detected_language,
            gap_fill_applied=gap_fill_applied,
            gap_fill_findings=gap_fill_findings,
        )
        return entry, None
    except Exception as exc:
        rel_name = _rel_name(skill_dir, root)
        error_entry: dict[str, object] = {
            "skill": {
                "name": rel_name,
                "source": str(skill_dir),
                "source_group": rel_name.split("/")[0] if "/" in rel_name else ".",
                "language": detected_language,
                "scanned_at": datetime.now(UTC).isoformat(),
            },
            "risk_assessment": {
                "score": 0,
                "severity": "ERROR",
                "recommendation": "ERROR",
            },
            "components": [],
            "issues": [],
            "scan_mode": "multilingual-enhanced",
            "enhancements": {
                "gap_fill_applied": False,
                "gap_fill_findings": 0,
                "english_keyword_rules_skipped": 0,
            },
            "error": str(exc),
        }
        return error_entry, str(exc)
    finally:
        if result is not None:
            cleanup_result(result)


def _rel_name(skill_dir: Path, root: Path) -> str:
    """Best-effort relative name for display in progress lines."""
    try:
        return str(skill_dir.relative_to(root))
    except ValueError:
        return skill_dir.name
