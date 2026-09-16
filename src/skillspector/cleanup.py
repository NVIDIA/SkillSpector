# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared cleanup helpers for SkillSpector."""

import os
import shutil
import stat
from collections.abc import Callable
from pathlib import Path

from skillspector.python_ast import clear_python_ast_cache


def _retry_writable(function: Callable[[str], object], path: str, _error: BaseException) -> None:
    """Clear a read-only bit and retry once; Windows refuses to delete read-only files."""
    try:
        # chmod follows links, so never touch whatever a link points at.
        if not (os.path.islink(path) or os.path.isjunction(path)):
            os.chmod(path, stat.S_IWRITE)
            function(path)
    except OSError:
        pass


def remove_temp_tree(path: str | Path) -> None:
    """Best-effort removal of a scan temp directory, including read-only files.

    ``git clone`` writes its pack files read-only, so ``ignore_errors=True``
    alone leaves every cloned repository behind on Windows.
    """
    shutil.rmtree(path, onexc=_retry_writable)


def cleanup_result(result: dict[str, object]) -> None:
    """Release scan-local resources and remove a temp dir if set."""
    python_ast_cache_key = result.get("python_ast_cache_key")
    clear_python_ast_cache(python_ast_cache_key if isinstance(python_ast_cache_key, str) else None)
    temp_dir = result.get("temp_dir_for_cleanup")
    if temp_dir and isinstance(temp_dir, str):
        remove_temp_tree(temp_dir)
