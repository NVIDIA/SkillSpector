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

"""Skillspector v2 LangGraph workflow package."""

import warnings
from importlib.metadata import version as _pkg_version

__version__ = _pkg_version("skillspector")

# ponytail: langgraph deserializes with langchain's allowed_objects default,
# which warns. langchain_core's import re-enables that warning via
# surface_langchain_deprecation_warnings(), so import it first, then prepend our
# ignore filter so it wins. Drop this once langgraph pins an explicit default.
import langchain_core  # noqa: F401  (force its warning-filter setup before ours)

warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change",
    category=Warning,
)

from skillspector.graph import create_graph, graph  # noqa: E402 (after filter setup)

__all__ = ["create_graph", "graph", "__version__"]
