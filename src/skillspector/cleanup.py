# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared cleanup helpers for SkillSpector."""

import os
import shutil
import stat
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

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


class TempDirTracker(BaseCallbackHandler):
    """Remember the temp directory a graph run materializes, as soon as it exists.

    ``resolve_input`` reports the directory in its output as ``temp_dir_for_cleanup``,
    but a run that raises or is interrupted returns no result for
    :func:`cleanup_result`. Pass the tracker in the run's ``callbacks`` and use
    :meth:`removing_on_error` around the run.
    """

    run_inline = True

    def __init__(self) -> None:
        super().__init__()
        self.temp_dir: str | None = None

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        """Record ``temp_dir_for_cleanup`` from a node or graph output."""
        if isinstance(outputs, Mapping):
            temp_dir = outputs.get("temp_dir_for_cleanup")
            if isinstance(temp_dir, str) and temp_dir:
                self.temp_dir = temp_dir

    def remove(self) -> None:
        """Remove the recorded temp directory, if any."""
        if self.temp_dir:
            remove_temp_tree(self.temp_dir)

    @contextmanager
    def removing_on_error(self) -> Iterator[None]:
        """Remove the recorded temp directory if the wrapped run raises or is interrupted."""
        try:
            yield
        except BaseException:
            self.remove()
            raise


def cleanup_result(result: dict[str, object]) -> None:
    """Release scan-local resources and remove a temp dir if set."""
    python_ast_cache_key = result.get("python_ast_cache_key")
    clear_python_ast_cache(python_ast_cache_key if isinstance(python_ast_cache_key, str) else None)
    temp_dir = result.get("temp_dir_for_cleanup")
    if temp_dir and isinstance(temp_dir, str):
        remove_temp_tree(temp_dir)
