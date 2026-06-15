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

"""Analyzer node registry for Skillspector v2 stub workflow."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from skillspector.logging_config import get_logger

logger = get_logger(__name__)

ANALYZER_NODE_IDS: list[str] = []
ANALYZER_NODES: dict[str, Any] = {}
ANALYZER_MODULES: dict[str, Any] = {}

def _discover_analyzers() -> None:
    """Dynamically discover and register analyzer modules in this package."""
    if ANALYZER_NODE_IDS:
        return

    for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
        if is_pkg:
            continue

        full_module_name = f"{__name__}.{module_name}"
        try:
            mod = importlib.import_module(full_module_name)
        except Exception as exc:
            logger.debug("Skipping module %s during discovery: %s", module_name, exc)
            continue

        analyzer_id = getattr(mod, "ANALYZER_ID", None)
        node_func = getattr(mod, "node", None)

        if analyzer_id and callable(node_func):
            ANALYZER_NODE_IDS.append(analyzer_id)
            ANALYZER_NODES[analyzer_id] = node_func
            ANALYZER_MODULES[analyzer_id] = mod

_discover_analyzers()

__all__ = ["ANALYZER_NODE_IDS", "ANALYZER_NODES", "ANALYZER_MODULES"]
