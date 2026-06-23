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

"""SQLite-based persistent cache for SkillSpector LLM scans."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from skillspector.logging_config import get_logger

logger = get_logger(__name__)


def get_cache_db_path() -> Path:
    """Resolve cache database file path, creating parent directories if needed."""
    cache_dir = Path(os.path.expanduser("~/.cache/skillspector"))
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("Could not create cache directory %s: %s", cache_dir, e)
    return cache_dir / "scan_cache.db"


def initialize_cache_db() -> None:
    """Create the scan_cache table if it does not already exist."""
    db_path = get_cache_db_path()
    try:
        with sqlite3.connect(db_path, timeout=10.0) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scan_cache (
                    cache_key TEXT PRIMARY KEY,
                    analyzer_id TEXT,
                    content_hash TEXT,
                    model TEXT,
                    findings_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
    except sqlite3.Error as e:
        logger.warning("Failed to initialize SQLite cache database: %s", e)


def get_cached_findings(cache_key: str) -> str | None:
    """Retrieve cached findings JSON for the given cache key.

    Returns None if not found or on database error.
    """
    db_path = get_cache_db_path()
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT findings_json FROM scan_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cursor.fetchone()
            if row:
                return str(row[0])
    except sqlite3.Error as e:
        logger.debug("Failed to read from SQLite cache: %s", e)
    return None


def set_cached_findings(
    cache_key: str,
    findings_json: str,
    analyzer_id: str,
    content_hash: str,
    model: str,
) -> None:
    """Insert or replace findings in the persistent SQLite cache."""
    db_path = get_cache_db_path()
    # Ensure directory and table exist
    initialize_cache_db()
    try:
        with sqlite3.connect(db_path, timeout=10.0) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scan_cache
                (cache_key, analyzer_id, content_hash, model, findings_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cache_key, analyzer_id, content_hash, model, findings_json),
            )
            conn.commit()
    except sqlite3.Error as e:
        logger.warning("Failed to write to SQLite cache: %s", e)
