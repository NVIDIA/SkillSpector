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

"""Write reports without following attacker-replaced paths."""

from __future__ import annotations

import os
from pathlib import Path
from secrets import token_hex
from stat import S_ISREG

from skillspector.input_handler import _normalize_root_owned_alias

_SECURE_OUTPUT_SUPPORTED = (
    hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_DIRECTORY")
    and {os.open, os.stat, os.rename, os.unlink} <= os.supports_dir_fd
    and os.stat in os.supports_follow_symlinks
)


def write_text_no_follow(path: str | Path, text: str) -> None:
    """Atomically replace a regular output through an anchored parent descriptor.

    Files are created privately, including replacements. A final-component swap
    cannot redirect the write; a parent swap cannot change the opened directory.
    Platforms without these guarantees must use explicitly managed stdout.
    """
    if not _SECURE_OUTPUT_SUPPORTED:
        raise ValueError("Safe file output is unsupported on this platform; use stdout redirection.")
    absolute = _normalize_root_owned_alias(Path(path))
    flags = os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_PATH", os.O_RDONLY)
    directory_fd = os.open(absolute.anchor, flags)
    temporary_name = None
    try:
        for part in absolute.parts[1:-1]:
            next_fd = os.open(part, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        try:
            existing = os.stat(absolute.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if not S_ISREG(existing.st_mode):
                raise ValueError("Refusing to overwrite a non-regular output file.")
        candidate = f".skillspector-output-{token_hex(16)}"
        fd = os.open(
            candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600, dir_fd=directory_fd,
        )
        temporary_name = candidate
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(
            temporary_name, absolute.name,
            src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
        )
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)
