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

"""GitHub Copilot CLI provider — Stage-2 LLM analysis via the local ``copilot`` binary.

Activated by ``SKILLSPECTOR_PROVIDER=copilot_cli``. Authentication is handled by
the ``copilot`` CLI's own login session (``copilot login``) or one of its
token environment variables; no OpenAI/Anthropic API key is read or required.

All behaviour is inherited from
:class:`skillspector.providers._agent_cli_base.AgentCLIProviderBase`; the
"copilot"-specific argv (``copilot -s --no-ask-user`` with an implausible
``--available-tools`` allowlist plus ``--deny-tool shell,write``; never
``--allow-all*``), output parsing, and auth probe live in the
:mod:`skillspector.providers._agent_cli` registry.
"""

from __future__ import annotations

from skillspector.providers._agent_cli_base import AgentCLIProviderBase

BINARY_NAME = "copilot"


class CopilotCLIProvider(AgentCLIProviderBase):
    """GitHub Copilot CLI provider (no API key; uses the local ``copilot`` login).

    No model is pinned: ``copilot`` runs with the CLI default model
    (pinning is fragile across providers, so the default is used). Set
    ``SKILLSPECTOR_MODEL`` to override.
    """

    BINARY_NAME = "copilot"
