# SPDX-License-Identifier: Apache-2.0
"""GitHub Copilot CLI provider — uses the locally-installed ``copilot`` binary.

No provider API key required. Authentication is managed by the Copilot CLI's
own session (``copilot login``). Set ``SKILLSPECTOR_PROVIDER=copilot_cli`` to
activate.
"""

from .provider import CopilotCLIProvider

__all__ = ["CopilotCLIProvider"]
