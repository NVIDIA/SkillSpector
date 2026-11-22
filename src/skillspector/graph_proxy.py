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

"""Lightweight lazy access to the compiled SkillSpector workflow graph."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class BlockedNativeImport:
    """A native extension import the operating system refused to load."""

    module: str
    policy_blocked: bool


# Substrings marking an ImportError as the OS refusing to load a compiled
# extension module, rather than a module that is simply not installed.
_NATIVE_LOAD_FAILURE_MARKERS = (
    "dll load failed",
    "not a valid win32 application",
    "bad image",
)
_POLICY_BLOCK_MARKERS = (
    "blocked",
    "application control",
    "wdac",
    "code integrity",
)

_IMPORT_MODULE_RE = re.compile(r"while importing ([\w.]+)")


def blocked_native_import(exc: BaseException) -> BlockedNativeImport | None:
    """Return the blocked native module behind ``exc``, or None.

    Only an import failure that names a module and reads like a refused
    native load counts. A missing module (``No module named ...``) and
    unrelated errors return None, so they keep their existing handling.
    """
    if isinstance(exc, GraphLoadError):
        return BlockedNativeImport(module=exc.module, policy_blocked=exc.policy_blocked)
    if not isinstance(exc, (ImportError, OSError)):
        return None
    message = str(exc)
    lowered = message.lower()
    if not any(
        marker in lowered for marker in _NATIVE_LOAD_FAILURE_MARKERS + _POLICY_BLOCK_MARKERS
    ):
        return None
    module = getattr(exc, "name", None)
    if not isinstance(module, str) or not module:
        match = _IMPORT_MODULE_RE.search(message)
        module = match.group(1) if match else None
    if not module:
        return None
    return BlockedNativeImport(
        module=module,
        policy_blocked=any(marker in lowered for marker in _POLICY_BLOCK_MARKERS),
    )


def format_blocked_native_import_diagnostic(info: BlockedNativeImport) -> str:
    """Build the fail-fast diagnostic for a blocked native module import.

    The text is constructed from the module name only. The raw exception
    text is never copied, so secrets or local paths in it cannot leak into
    reports.
    """
    if info.policy_blocked:
        cause = (
            f"Windows Application Control blocked the unsigned native module "
            f"'{info.module}' at import"
        )
    else:
        cause = f"the operating system refused to load the native module '{info.module}' at import"
    return (
        f"SkillSpector could not load the scan workflow: {cause}. "
        f"Every scan fails while this module cannot load. Workarounds: run "
        f"SkillSpector in WSL or the provided Docker image, or ask your "
        f"administrator to allowlist the SkillSpector install directory in "
        f"the application control policy. Signed wheels are tracked in "
        f"NVIDIA/SkillSpector#1166."
    )


class GraphLoadError(ImportError):
    """The scan workflow could not load because a native module import failed."""

    def __init__(self, info: BlockedNativeImport) -> None:
        self.module = info.module
        self.policy_blocked = info.policy_blocked
        super().__init__(format_blocked_native_import_diagnostic(info))


def translate_graph_load_error(exc: BaseException) -> BaseException:
    """Wrap a blocked native import in a GraphLoadError.

    Any other error is returned unchanged, so existing error handling for
    missing modules and unrelated failures is preserved.
    """
    if isinstance(exc, GraphLoadError):
        return exc
    info = blocked_native_import(exc)
    if info is None:
        return exc
    return GraphLoadError(info)


class LazyGraph:
    """Load the compiled workflow only when a caller first uses it."""

    def __init__(self) -> None:
        self._compiled: Any | None = None
        self._lock = Lock()

    def _get_compiled(self) -> Any:
        if self._compiled is None:
            with self._lock:
                if self._compiled is None:
                    try:
                        from skillspector.graph import graph as compiled_graph
                    except (ImportError, OSError) as exc:
                        raise translate_graph_load_error(exc) from exc

                    self._compiled = compiled_graph
                    # Importing the submodule assigns it to the parent package.
                    # Restore the documented package-level lazy export before
                    # another caller imports it.
                    import skillspector as _skillspector_pkg

                    _skillspector_pkg.graph = graph
                    sys.modules["skillspector"].graph = graph
        return self._compiled

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_compiled(), name)


def restore_package_graph_export() -> None:
    """Re-assert the lazy proxy as the package-level ``graph`` export.

    Importing the ``skillspector.graph`` submodule assigns the module object
    to the parent package's ``graph`` attribute, shadowing this proxy. Call
    this after any eager submodule import (the ``create_graph()`` factory
    wrapper, ``mcp_server``) so ``skillspector.graph`` stays invokable
    regardless of import order.
    """
    import skillspector as _pkg

    _pkg.graph = graph


graph = LazyGraph()
