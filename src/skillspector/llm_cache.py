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

"""SQLite-backed LLM response cache for SkillSpector.

Caches LLM responses keyed by (file_content_hash, prompt_template_hash, schema_version).
Unchanged files do not make repeated LLM calls across scan runs.

Cache location: a trusted, per-skill directory under the OS application-cache
root (see `default_cache_dir`), never inside the scanned skill directory.
Disable entirely: set SKILLSPECTOR_NO_LLM_CACHE=1.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from skillspector.logging_config import get_logger

logger = get_logger(__name__)

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS llm_responses (
    content_hash  TEXT NOT NULL,
    prompt_hash   TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (content_hash, prompt_hash, schema_version)
);
"""


@dataclass(frozen=True)
class CacheKey:
    """Immutable cache key: hashes for content, prompt template, and schema version."""

    content_hash: str
    prompt_hash: str
    schema_version: str


def make_cache_key(content: str, prompt_template: str, schema_version: str) -> CacheKey:
    """Build a CacheKey from raw strings (SHA-256, truncated to 16 hex chars)."""
    return CacheKey(
        content_hash=hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16],
        prompt_hash=hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()[:16],
        schema_version=schema_version,
    )


def default_cache_dir(skill_dir: Path) -> Path:
    """Trusted application cache dir for *skill_dir*, always outside scanned content."""
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    key = hashlib.sha256(str(skill_dir.resolve()).encode("utf-8")).hexdigest()[:16]
    return root / "skillspector" / "llm-cache" / key


class LLMResponseCache:
    """SQLite-backed cache for LLM responses.

    Stores responses keyed by (content_hash, prompt_hash, schema_version) so that
    repeated scans of unchanged files skip LLM calls entirely.

    Thread-safety: one connection per instance; not safe for concurrent writes from
    multiple processes to the same database file (SQLite WAL mode is not enabled here
    by design — the cache is per-skill-directory, single-writer).
    """

    def __init__(self, cache_dir: Path) -> None:
        """Initialise the cache at *cache_dir*/llm_responses.db.

        The directory (and the SQLite file) are created lazily on the first
        ``put`` call.  Set ``SKILLSPECTOR_NO_LLM_CACHE=1`` in the environment
        to disable all caching without changing code.
        """
        self._db_path = Path(cache_dir) / "llm_responses.db"
        self._enabled = os.environ.get("SKILLSPECTOR_NO_LLM_CACHE", "").strip() not in (
            "1",
            "true",
            "yes",
        )
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        """Open (or reuse) the SQLite connection, creating the schema if needed."""
        if self._conn is None:
            if self._db_path.parent.is_symlink() or self._db_path.is_symlink():
                raise RuntimeError(f"Refusing to use symlinked cache path: {self._db_path}")
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path))
            conn.execute(_SCHEMA_DDL)
            conn.commit()
            self._conn = conn
        return self._conn

    def get(self, key: CacheKey) -> str | None:
        """Return cached response JSON, or None on miss."""
        if not self._enabled:
            return None
        try:
            conn = self._connect()
            row = conn.execute(
                "SELECT response_json FROM llm_responses "
                "WHERE content_hash=? AND prompt_hash=? AND schema_version=?",
                (key.content_hash, key.prompt_hash, key.schema_version),
            ).fetchone()
            return row[0] if row else None
        except Exception as exc:  # noqa: BLE001
            logger.debug("LLM cache read error: %s", exc)
            return None

    def put(self, key: CacheKey, response_json: str) -> None:
        """Store a response in the cache (insert or replace)."""
        if not self._enabled:
            return
        try:
            conn = self._connect()
            conn.execute(
                "INSERT OR REPLACE INTO llm_responses "
                "(content_hash, prompt_hash, schema_version, response_json) VALUES (?,?,?,?)",
                (key.content_hash, key.prompt_hash, key.schema_version, response_json),
            )
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.debug("LLM cache write error: %s", exc)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        """Close the database connection when the object is garbage collected."""
        self.close()
