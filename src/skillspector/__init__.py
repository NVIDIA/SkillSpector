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

from importlib.metadata import version as _pkg_version
from typing import Any

from skillspector.graph_proxy import graph, restore_package_graph_export

__version__ = _pkg_version("skillspector")


def create_graph() -> Any:
    """Build and return a new SkillSpector workflow graph."""
    from skillspector.graph import create_graph as build_graph

    try:
        return build_graph()
    finally:
        # Importing the submodule shadows the lazy proxy on the package;
        # restore the documented export even when this factory is the first
        # graph access.
        restore_package_graph_export()


__all__ = ["create_graph", "graph", "__version__"]
