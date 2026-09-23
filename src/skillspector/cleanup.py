# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared cleanup helpers for SkillSpector."""

import shutil

from skillspector.artifacts import clear_security_text_caches
from skillspector.python_ast import clear_python_ast_cache


def cleanup_result(result: dict[str, object] | None) -> None:
    """Release scan-local resources and remove a temp dir if set.

    ``result`` is ``None`` when the scan raised or was cancelled. The security
    text caches are keyed by content rather than by scan, so they can still be
    released -- and must be, or a long-lived server keeps the derived views and
    the scanned text itself alive until some later scan happens to succeed. The
    rest of the teardown needs the returned state and has nothing to act on.
    """
    clear_security_text_caches()
    if result is None:
        return
    python_ast_cache_key = result.get("python_ast_cache_key")
    clear_python_ast_cache(python_ast_cache_key if isinstance(python_ast_cache_key, str) else None)
    temp_dir = result.get("temp_dir_for_cleanup")
    if temp_dir and isinstance(temp_dir, str):
        shutil.rmtree(temp_dir, ignore_errors=True)
