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

"""Claude CLI provider — Stage-2 LLM analysis via the local ``claude`` binary.

Activated by ``SKILLSPECTOR_PROVIDER=claude_cli``. Authentication is handled by
the ``claude`` CLI's own OAuth/keychain session (``claude auth login``); no API
key is read or required.

All behaviour is inherited from
:class:`skillspector.providers._agent_cli_base.AgentCLIProviderBase`; the
"claude"-specific argv, output parsing, and auth probe live in the
:mod:`skillspector.providers._agent_cli` registry. Security comes from the
hardened, fail-closed ``run_agent_cli`` chokepoint.
"""

from __future__ import annotations

from skillspector.providers._agent_cli_base import AgentCLIProviderBase

BINARY_NAME = "claude"


class ClaudeCLIProvider(AgentCLIProviderBase):
    """Claude CLI provider (no API key; uses the local ``claude`` login).

    No model is pinned: ``claude`` falls back to its remaining runtime default
    after SkillSpector strips user, project, and local settings. Set
    ``SKILLSPECTOR_MODEL`` to override.
    """

    BINARY_NAME = "claude"
