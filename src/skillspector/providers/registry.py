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

"""YAML model-registry utilities.

Each provider ships an accompanying ``<provider>.yaml`` that lists the
token-budget metadata for the models it serves.  Providers call
:func:`lookup_context_length` and :func:`lookup_max_output_tokens` with
their bundled YAML path; results are cached per path for the lifetime of
the process.

A user-supplied ``SKILLSPECTOR_MODEL_REGISTRY`` env var, when set,
overrides the bundled path globally — useful for adding models without
editing the package.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml

from skillspector.logging_config import get_logger

logger = get_logger(__name__)


@functools.cache
def _load(yaml_path: str) -> dict[str, dict[str, Any]]:
    """Read *yaml_path* and return its ``models`` map (cached).

    Returns an empty dict if the path is empty, missing, or unreadable —
    callers fall through to the default token-budget logic in that case.
    """
    if not yaml_path:
        return {}

    try:
        raw = Path(yaml_path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}
        return data.get("models") or {}
    except Exception:
        logger.warning("Could not load model registry at %s", yaml_path, exc_info=True)
        return {}


def _resolve_path(default_yaml_path: str) -> str:
    """Return the user override path if set, otherwise *default_yaml_path*."""
    override = os.environ.get("SKILLSPECTOR_MODEL_REGISTRY", "").strip()
    return override or default_yaml_path


def lookup_context_length(default_yaml_path: str, model: str) -> int | None:
    """Return ``context_length`` for *model* from the resolved YAML registry."""
    entry = _load(_resolve_path(default_yaml_path)).get(model)
    if entry and entry.get("context_length"):
        return int(entry["context_length"])
    return None


def lookup_max_output_tokens(default_yaml_path: str, model: str) -> int | None:
    """Return ``max_output_tokens`` for *model* from the resolved YAML registry."""
    entry = _load(_resolve_path(default_yaml_path)).get(model)
    if entry and entry.get("max_output_tokens"):
        return int(entry["max_output_tokens"])
    return None


# Back-compat alias for tests that previously called ``_load_registry``.
def lookup_structured_output_method(default_yaml_path: str, model: str) -> str | None:
    """Return the registry's ``structured_output`` method for *model*, if declared."""
    entry = _load(_resolve_path(default_yaml_path)).get(model)
    if not entry:
        return None
    value = entry.get("structured_output")
    return str(value) if value else None


_load_registry = _load
