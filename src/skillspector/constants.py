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

"""Shared constants for skillspector (env-driven where applicable)."""

import logging
import os

from skillspector.providers import get_metadata_provider

logger = logging.getLogger(__name__)

# % of model's max tokens used for input. 1-MAX_INPUT_TOKENS_PCT is used for output.
MAX_INPUT_TOKENS_PCT = 0.75
# Fallback context length when no metadata API or registry entry is available.
DEFAULT_CONTEXT_LENGTH = 128_000

# Default-model selection lives on each provider (see providers/<name>/provider.py
# for ``DEFAULT_MODEL`` and ``SLOT_DEFAULTS``).  The active provider's
# ``resolve_model`` runs the waterfall: ``SKILLSPECTOR_MODEL`` env > slot
# default > general default.  OSS users pointing at build.nvidia.com or
# stock OpenAI inherit ``NvBuildProvider``'s default model automatically.
_provider = get_metadata_provider()

# Exposed for analyzers that need a final fallback symbol (e.g.,
# ``model = state.model or MODEL_CONFIG[ANALYZER_ID] or _SKILLSPECTOR_DEFAULT_MODEL``).
_SKILLSPECTOR_DEFAULT_MODEL = _provider.DEFAULT_MODEL

_MODEL_SLOTS: tuple[str, ...] = (
    "default",
    "mcp_least_privilege",
    "mcp_rug_pull",
    "mcp_tool_poisoning",
    "semantic_developer_intent",
    "semantic_quality_policy",
    "semantic_security_discovery",
    "meta_analyzer",
)


def _resolve_slot_model(slot: str) -> str:
    """Resolve the model for *slot* with per-slot env var override support.

    Precedence: ``SKILLSPECTOR_MODEL_{SLOT}`` env var > provider
    ``resolve_model(slot)`` (which itself runs ``SKILLSPECTOR_MODEL`` env >
    provider slot default > provider ``DEFAULT_MODEL``).
    """
    env_key = f"SKILLSPECTOR_MODEL_{slot.upper()}"
    env_val = os.environ.get(env_key, "").strip()
    if env_val:
        return env_val
    return _provider.resolve_model(slot)


MODEL_CONFIG: dict[str, str] = {slot: _resolve_slot_model(slot) for slot in _MODEL_SLOTS}


def _validate_model_config() -> None:
    """Warn about models not found in the provider's model registry.

    When ``SKILLSPECTOR_STRICT_MODEL_VALIDATION=true``, raises
    ``ValueError`` instead of logging warnings.
    """
    unknown: list[str] = []
    for slot, model in MODEL_CONFIG.items():
        ctx = _provider.get_context_length(model)  # type: ignore[attr-defined]
        if ctx is None:
            unknown.append(f"  {slot}: {model}")
            logger.warning(
                "Model '%s' (slot: %s) not found in model_registry.yaml. "
                "Using fallback context length (%d). Token budgeting may be "
                "inaccurate — add the model to the registry or verify the "
                "model ID.",
                model,
                slot,
                DEFAULT_CONTEXT_LENGTH,
            )

    strict = os.environ.get("SKILLSPECTOR_STRICT_MODEL_VALIDATION", "").lower() == "true"
    if strict and unknown:
        raise ValueError(
            "Strict model validation enabled. Unknown models:\n"
            + "\n".join(unknown)
            + "\nAdd them to model_registry.yaml or disable "
            "SKILLSPECTOR_STRICT_MODEL_VALIDATION."
        )


_validate_model_config()

# Log level: from env or fallback (DEBUG, INFO, WARNING, ERROR).
SKILLSPECTOR_LOG_LEVEL = os.environ.get("SKILLSPECTOR_LOG_LEVEL", "WARNING")
