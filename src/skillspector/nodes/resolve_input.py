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

"""Resolve-input node: normalizes input_path or skill_path to a local directory (skill_path).

Runs as the first graph node. When input_path is a URL, zip, or single file,
resolves to a temp directory and sets temp_dir_for_cleanup so the caller (CLI/API)
can clean up after the graph completes.
"""

from __future__ import annotations

from pathlib import Path

from skillspector.input_handler import InputHandler
from skillspector.logging_config import get_logger
from skillspector.state import SkillspectorState

logger = get_logger(__name__)


def resolve_input(state: SkillspectorState) -> dict[str, object]:
    """
    Resolve input to a scannable directory.

    - If state has non-empty input_path: resolve it (Git URL, file URL, zip, file, or directory)
      and set skill_path. If resolution created a temp dir, set temp_dir_for_cleanup.
    - Else if state has skill_path: normalize to absolute path and set skill_path.
    - Else: set skill_path to None (build_context will raise a user-facing error).
    """
    input_path = state.get("input_path")
    skill_path = state.get("skill_path")

    if input_path and isinstance(input_path, str) and input_path.strip():
        handler = InputHandler()
        try:
            resolved, source_type = handler.resolve(input_path.strip())
            update: dict[str, object] = {"skill_path": str(resolved)}
            temp_dir = handler.temp_dir_for_cleanup()
            if temp_dir is not None:
                update["temp_dir_for_cleanup"] = str(temp_dir)
            else:
                update["temp_dir_for_cleanup"] = None
            return update
        except (ValueError, FileNotFoundError):
            raise

    if skill_path and isinstance(skill_path, str) and skill_path.strip():
        try:
            resolved = Path(skill_path).resolve()
            return {
                "skill_path": str(resolved),
                "temp_dir_for_cleanup": None,
            }
        except (OSError, RuntimeError) as e:
            logger.warning("Could not resolve skill_path %s: %s", skill_path, e)
            return {"skill_path": None, "temp_dir_for_cleanup": None}

    return {"skill_path": None, "temp_dir_for_cleanup": None}
