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

"""Gemini CLI provider — Stage-2 LLM analysis via the local ``gemini`` binary.

Activated by ``SKILLSPECTOR_PROVIDER=gemini_cli``. The gemini-specific flags
(``gemini -p "" -o json --approval-mode plan --skip-trust``; see
``_build_gemini_argv``) are verified end-to-end against gemini 0.46.0. No model
is pinned: gemini runs with the user's own default model; set
``SKILLSPECTOR_MODEL`` to override. All behaviour is inherited from
:class:`skillspector.providers._agent_cli_base.AgentCLIProviderBase`.
"""

from __future__ import annotations

from skillspector.providers._agent_cli_base import AgentCLIProviderBase

BINARY_NAME = "gemini"


class GeminiCLIProvider(AgentCLIProviderBase):
    """Gemini CLI provider (no API key; uses the local ``gemini`` login)."""

    BINARY_NAME = "gemini"
