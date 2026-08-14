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

"""Shared base for local agent-CLI providers (claude_cli, codex_cli, gemini_cli).

A concrete provider is just four class attributes (see how thin
``ClaudeCLIProvider`` / ``CodexCLIProvider`` / ``GeminiCLIProvider`` are). All
behaviour — availability/auth, the CLI transport, token-budget metadata, and
model resolution — is inherited from here. The per-CLI specifics (argv, output
parsing, auth probe) live in the :mod:`skillspector.providers._agent_cli`
registry, keyed by ``BINARY_NAME``.
"""

from __future__ import annotations

import os

from skillspector.providers import _agent_cli, registry


class AgentCLIProviderBase:
    """Base for providers that drive a local agent CLI (no API key needed)."""

    #: CLI name; must be a key in ``_agent_cli._REGISTRY``.
    BINARY_NAME: str = ""
    #: Always "" for CLI providers. Present only so constants.py's
    #: ``_provider.DEFAULT_MODEL`` lookup has an attribute; never pins a version.
    DEFAULT_MODEL: str = ""
    #: Optional path to a bundled ``model_registry.yaml`` for token budgets. CLI
    #: providers leave this empty and fall back to package-wide default budgets.
    REGISTRY_PATH: str = ""

    # -- Credentials ---------------------------------------------------------

    def resolve_credentials(self) -> tuple[str, str | None] | None:
        """No HTTP credentials needed — the CLI handles auth itself."""
        return None

    # -- Availability --------------------------------------------------------

    def is_available(self) -> tuple[bool, str | None]:
        """Binary on PATH AND authenticated (delegates to the registry probe)."""
        return _agent_cli.is_available(self.BINARY_NAME)

    # -- Transport -----------------------------------------------------------

    def complete(self, prompt: str, *, model: str, max_output_tokens: int = 8192) -> str:
        """Invoke the CLI via the hardened runner and return the assistant text.

        The prompt is passed through unchanged (parity with the HTTP path).
        Security comes from the capability-stripped, fail-closed invocation in
        :func:`skillspector.providers._agent_cli.run_agent_cli`.
        """
        return _agent_cli.run_agent_cli(
            self.BINARY_NAME, prompt, model=model, max_output_tokens=max_output_tokens
        )

    # -- Metadata ------------------------------------------------------------

    def get_context_length(self, model: str) -> int | None:
        if not self.REGISTRY_PATH:
            return None  # no registry -> caller uses the package-wide default budget
        return registry.lookup_context_length(self.REGISTRY_PATH, model)

    def get_max_output_tokens(self, model: str) -> int | None:
        if not self.REGISTRY_PATH:
            return None
        return registry.lookup_max_output_tokens(self.REGISTRY_PATH, model)

    def resolve_model(self, slot: str = "default") -> str:
        """Return the model to forward to the CLI.

        CLI providers omit ``--model`` unless ``SKILLSPECTOR_MODEL`` is set, so
        this returns ``""`` by default. Each CLI then applies whatever fallback
        model behaviour its own settings and runtime contract define.
        """
        return os.environ.get("SKILLSPECTOR_MODEL", "").strip()
