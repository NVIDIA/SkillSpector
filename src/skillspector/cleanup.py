# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared cleanup helpers for SkillSpector."""

import shutil

from skillspector.python_ast import clear_python_ast_cache


def cleanup_result(result: dict[str, object]) -> None:
    """Release scan-local resources and remove a temp dir if set."""
    python_ast_cache_key = result.get("python_ast_cache_key")
    clear_python_ast_cache(python_ast_cache_key if isinstance(python_ast_cache_key, str) else None)
    temp_dir = result.get("temp_dir_for_cleanup")
    if temp_dir and isinstance(temp_dir, str):
        shutil.rmtree(temp_dir, ignore_errors=True)
