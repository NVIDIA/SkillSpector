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
        models = data.get("models")
        if models is None:
            return {}
        if not isinstance(models, dict):
            raise ValueError("model registry 'models' must be a mapping")
        return models
    except Exception:
        logger.warning("Could not load model registry at %s", yaml_path, exc_info=True)
        return {}


def _resolve_path(default_yaml_path: str) -> str:
    """Return the user override path if set, otherwise *default_yaml_path*."""
    override = os.environ.get("SKILLSPECTOR_MODEL_REGISTRY", "").strip()
    return override or default_yaml_path


def lookup_context_length(default_yaml_path: str, model: str) -> int | None:
    """Return ``context_length`` for *model* from the resolved YAML registry."""
    return _lookup_token_limit(default_yaml_path, model, "context_length")


def lookup_max_output_tokens(default_yaml_path: str, model: str) -> int | None:
    """Return ``max_output_tokens`` for *model* from the resolved YAML registry."""
    return _lookup_token_limit(default_yaml_path, model, "max_output_tokens")


def _lookup_token_limit(default_yaml_path: str, model: str, field: str) -> int | None:
    """Read a positive budget, falling back for malformed entries or values."""
    yaml_path = _resolve_path(default_yaml_path)
    entry = _load(yaml_path).get(model)
    if entry is None:
        return None
    try:
        if not isinstance(entry, dict):
            raise ValueError("model registry entry must be a mapping")
        value = entry.get(field)
        if value is None:
            return None
        limit = int(value)
        if limit <= 0:
            raise ValueError("model token budget must be positive")
        return limit
    except (TypeError, ValueError, OverflowError):
        logger.warning("Invalid %s for model %s in registry %s", field, model, yaml_path)
        return None


# Back-compat alias for tests that previously called ``_load_registry``.
_load_registry = _load
