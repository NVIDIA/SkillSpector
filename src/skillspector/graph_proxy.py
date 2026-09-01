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

"""Lightweight lazy access to the compiled SkillSpector workflow graph."""

from __future__ import annotations

import sys
from threading import Lock
from typing import Any


class LazyGraph:
    """Load the compiled workflow only when a caller first uses it."""

    def __init__(self) -> None:
        self._compiled: Any | None = None
        self._lock = Lock()

    def _get_compiled(self) -> Any:
        if self._compiled is None:
            with self._lock:
                if self._compiled is None:
                    from skillspector.graph import graph as compiled_graph

                    self._compiled = compiled_graph
                    # Importing the submodule assigns it to the parent package.
                    # Restore the documented package-level lazy export before
                    # another caller imports it.
                    setattr(sys.modules["skillspector"], "graph", graph)
        return self._compiled

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_compiled(), name)


graph = LazyGraph()
