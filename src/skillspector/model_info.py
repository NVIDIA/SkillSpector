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

"""Model metadata helpers — token-budget resolution.

Layered resolution lives in :mod:`skillspector.providers`; this module is
the public façade that callers (e.g. ``llm_utils``, ``llm_analyzer_base``)
import from.  Test fixtures patch :func:`_resolve_context_length` here.
"""

from __future__ import annotations

from skillspector.constants import DEFAULT_CONTEXT_LENGTH, MAX_INPUT_TOKENS_PCT
from skillspector.logging_config import get_logger
from skillspector.providers import get_metadata_provider

logger = get_logger(__name__)


def _resolve_context_length(model_label: str) -> int:
    """Return the context window size for *model_label*.

    Delegates to the configured provider chain; falls back to
    :data:`DEFAULT_CONTEXT_LENGTH` with a warning when no provider knows
    about the model.
    """
    ctx = get_metadata_provider().get_context_length(model_label)
    if ctx is not None:
        logger.debug("Resolved %r context length: %d", model_label, ctx)
        return ctx

    logger.warning(
        "No token-limit info for model %r — using %d-token default. "
        "Add the model to model_registry.yaml.",
        model_label,
        DEFAULT_CONTEXT_LENGTH,
    )
    return DEFAULT_CONTEXT_LENGTH


def get_max_input_tokens(model: str) -> int:
    """Input token budget for *model* (75 %% of context window)."""
    return int(_resolve_context_length(model) * MAX_INPUT_TOKENS_PCT)


def get_max_output_tokens(model: str) -> int:
    """Output token budget for *model*.

    Uses the smaller of the percentage-based budget and any explicit
    ``max_output_tokens`` cap exposed by the provider chain (today: only
    the YAML registry surfaces this).
    """
    ctx = _resolve_context_length(model)
    pct_budget = int(ctx * (1 - MAX_INPUT_TOKENS_PCT))

    cap = get_metadata_provider().get_max_output_tokens(model)
    if cap is not None:
        return min(pct_budget, cap)
    return pct_budget
