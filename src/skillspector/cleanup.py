# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared cleanup helpers for SkillSpector."""

import os
import shutil
import stat
from collections.abc import Callable
from pathlib import Path

from skillspector.python_ast import clear_python_ast_cache


def _retry_writable(function: Callable[..., object], path: str, error: BaseException) -> None:
    """Clear a read-only bit and retry a refused removal once; Windows refuses to delete read-only files.

    Only a permission error from the removal calls is retried. Any other failure,
    such as a directory that is not empty yet, and any other callback ``rmtree``
    reports, such as ``os.open``, are left to its best-effort pass, and nothing
    raised here may reach the caller.
    """
    if function not in (os.unlink, os.rmdir) or not isinstance(error, PermissionError):
        return
    try:
        # chmod follows links, so never touch whatever a link points at.
        if os.path.islink(path) or os.path.isjunction(path):
            return
        # Add the owner-write bit only; replacing the mode would strip read and
        # search permission on POSIX and leave the entry harder to remove.
        os.chmod(path, stat.S_IMODE(os.lstat(path).st_mode) | stat.S_IWRITE)
        function(path)
    except Exception:
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
