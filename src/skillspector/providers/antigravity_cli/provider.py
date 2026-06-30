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

"""Antigravity CLI provider — Stage-2 LLM analysis via the local ``agy`` binary.

Activated by ``SKILLSPECTOR_PROVIDER=antigravity_cli``. **Registered but
DISABLED.** Tested end-to-end against the real ``agy``, it cannot be driven
programmatically: its ``--print`` mode renders to the TTY and returns empty
stdout on a pipe (how SkillSpector must capture it), and it takes the prompt as
an argv value rather than stdin. So it fails closed — ``is_available()`` reports
unavailable and any invocation raises. Its backend is Gemini, so for that
capability use ``SKILLSPECTOR_PROVIDER=gemini_cli`` (clean JSON over stdin). See
the antigravity note in :mod:`skillspector.providers._agent_cli` for the full
findings and what would be needed to enable it.

All behaviour is inherited from
:class:`skillspector.providers._agent_cli_base.AgentCLIProviderBase`.
"""

from __future__ import annotations

from skillspector.providers._agent_cli_base import AgentCLIProviderBase

BINARY_NAME = "agy"


class AntigravityCLIProvider(AgentCLIProviderBase):
    """Antigravity CLI provider (registered but disabled; fail-closed).

    ``agy`` cannot be captured from a pipe (TTY-only print mode) — see the module
    docstring and the antigravity note in
    :mod:`skillspector.providers._agent_cli`. Use ``gemini_cli`` for the same
    (Gemini) backend.
    """

    BINARY_NAME = "agy"
