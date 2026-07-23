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

"""Skill Watch Mode: monitor directories for changes and auto-rescan.

Provides a ``skillspector watch`` CLI command that watches a directory for
file changes (using polling on all platforms) and automatically re-scans
when SKILL.md or executable files are modified.

Features:
- Configurable poll interval
- Debounce rapid changes
- Per-scan output formatting
- Baseline support for incremental scanning
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from skillspector.logging_config import get_logger

logger = get_logger(__name__)

_WATCH_EXTENSIONS = frozenset({
    ".md", ".markdown", ".py", ".sh", ".bash", ".zsh",
    ".js", ".ts", ".json", ".yaml", ".yml", ".toml",
    ".rb", ".go", ".rs",
})

_WATCH_PATTERNS = frozenset({
    "SKILL.md", "skill.md", "requirements.txt", "pyproject.toml",
    "package.json", "Gemfile", "go.mod", "Cargo.toml",
})


def _compute_directory_hash(directory: Path) -> str:
    """Compute a hash of all watchable files in a directory tree."""
    hasher = hashlib.md5(usedforsecurity=False)
    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue
        if file_path.suffix.lower() in _WATCH_EXTENSIONS or file_path.name in _WATCH_PATTERNS:
            try:
                content = file_path.read_bytes()
                hasher.update(file_path.relative_to(directory).as_posix().encode())
                hasher.update(content)
            except OSError:
                continue
    return hasher.hexdigest()


def watch_directory(
    directory: Path,
    callback,
    poll_interval: float = 2.0,
    debounce: float = 5.0,
    **callback_kwargs,
) -> None:
    """Watch a directory for changes and invoke callback on modification.

    Args:
        directory: Directory to watch.
        callback: Function to call when changes are detected. Receives directory path.
        poll_interval: Seconds between polls.
        debounce: Seconds to wait after a change before triggering a scan (to batch rapid edits).
        **callback_kwargs: Extra kwargs passed to callback.
    """
    logger.info("Watching %s (poll=%ss, debounce=%ss)", directory, poll_interval, debounce)

    last_hash = _compute_directory_hash(directory)
    last_change_time: float | None = None

    while True:
        time.sleep(poll_interval)
        current_hash = _compute_directory_hash(directory)

        if current_hash != last_hash:
            now = time.time()
            if last_change_time is None:
                last_change_time = now

            if now - last_change_time >= debounce:
                logger.info("Changes detected in %s, triggering scan...", directory)
                try:
                    callback(str(directory), **callback_kwargs)
                except Exception:
                    logger.exception("Error during watch scan callback")
                last_hash = _compute_directory_hash(directory)
                last_change_time = None
        else:
            last_change_time = None
