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

"""Codex CLI provider — uses the locally-installed ``codex`` binary.

No API key required. Authentication is managed by the ``codex`` CLI's
own session (``codex login``). Set ``SKILLSPECTOR_PROVIDER=codex_cli``
to activate.

NOTE: codex_cli support is implemented using the same hardened subprocess
helper as claude_cli (``_agent_cli.run_agent_cli``).  See provider.py for
sandbox flags and limitations.
"""

from .provider import CodexCLIProvider

__all__ = ["CodexCLIProvider"]
