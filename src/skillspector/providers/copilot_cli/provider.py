# SPDX-License-Identifier: Apache-2.0
"""GitHub Copilot CLI provider — Stage-2 LLM analysis via the local ``copilot`` binary.

Activated by ``SKILLSPECTOR_PROVIDER=copilot_cli``. Authentication is handled by
the Copilot CLI's own session (``copilot login``) or a ``COPILOT_GITHUB_TOKEN`` /
``GH_TOKEN`` / ``GITHUB_TOKEN`` environment token; no provider API key is read.

All behaviour is inherited from
:class:`skillspector.providers._agent_cli_base.AgentCLIProviderBase`; the
"copilot"-specific argv, output parsing, and auth probe live in the
:mod:`skillspector.providers._agent_cli` registry.
"""

from __future__ import annotations

from skillspector.providers._agent_cli_base import AgentCLIProviderBase

BINARY_NAME = "copilot"


class CopilotCLIProvider(AgentCLIProviderBase):
    """GitHub Copilot CLI provider (no API key; uses the local ``copilot`` login).

    No model is pinned: ``copilot`` runs with the CLI default model. Set
    ``SKILLSPECTOR_MODEL`` to override, e.g. ``claude-sonnet-5``.
    """

    BINARY_NAME = "copilot"
