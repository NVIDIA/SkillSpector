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

import os

from skillspector.providers import get_metadata_provider

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
_SKILLSPECTOR_DEFAULT_MODEL = _provider.DEFAULT_MODEL  # type: ignore[attr-defined]

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

MODEL_CONFIG: dict[str, str] = {slot: _provider.resolve_model(slot) for slot in _MODEL_SLOTS}

# Log level: from env or fallback (DEBUG, INFO, WARNING, ERROR).
SKILLSPECTOR_LOG_LEVEL = os.environ.get("SKILLSPECTOR_LOG_LEVEL", "WARNING")

# Maximum number of findings per meta-analyzer LLM call group.
# Keeps individual calls within context limits for large skill directories.
META_BATCH_SIZE: int = int(os.environ.get("SKILLSPECTOR_META_BATCH_SIZE", "20"))
