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

"""Tests for LLM response cache."""
import json
import sqlite3
from pathlib import Path

import pytest

from skillspector.llm_cache import CacheKey, LLMResponseCache, default_cache_dir


def test_cache_miss_returns_none(tmp_path):
    cache = LLMResponseCache(tmp_path)
    key = CacheKey(content_hash="abc123", prompt_hash="def456", schema_version="1")
    assert cache.get(key) is None


def test_cache_put_then_get(tmp_path):
    cache = LLMResponseCache(tmp_path)
    key = CacheKey(content_hash="abc123", prompt_hash="def456", schema_version="1")
    payload = json.dumps({"findings": []})
    cache.put(key, payload)
    assert cache.get(key) == payload


def test_cache_different_schema_version_is_miss(tmp_path):
    cache = LLMResponseCache(tmp_path)
    key_v1 = CacheKey(content_hash="abc", prompt_hash="def", schema_version="1")
    key_v2 = CacheKey(content_hash="abc", prompt_hash="def", schema_version="2")
    cache.put(key_v1, '{"findings": []}')
    assert cache.get(key_v2) is None


def test_cache_creates_db_on_first_use(tmp_path):
    cache_dir = tmp_path / "mycache"
    # Directory doesn't exist yet
    cache = LLMResponseCache(cache_dir)
    key = CacheKey(content_hash="x", prompt_hash="y", schema_version="1")
    cache.put(key, "test")
    assert (cache_dir / "llm_responses.db").exists()


def test_cache_key_from_content_and_prompt():
    from skillspector.llm_cache import make_cache_key
    key = make_cache_key(content="hello world", prompt_template="analyze: {}", schema_version="1")
    assert len(key.content_hash) == 16
    assert len(key.prompt_hash) == 16
    # Same inputs → same key
    key2 = make_cache_key(content="hello world", prompt_template="analyze: {}", schema_version="1")
    assert key == key2
    # Different content → different key
    key3 = make_cache_key(content="different", prompt_template="analyze: {}", schema_version="1")
    assert key3.content_hash != key.content_hash


def test_default_cache_dir_never_under_skill_dir(tmp_path):
    """The cache dir must always live outside the (untrusted) scanned skill directory."""
    skill_dir = tmp_path / "some-skill"
    skill_dir.mkdir()
    cache_dir = default_cache_dir(skill_dir)
    resolved_skill_dir = skill_dir.resolve()
    resolved_cache_dir = cache_dir.resolve()
    assert resolved_skill_dir not in resolved_cache_dir.parents
    assert resolved_cache_dir != resolved_skill_dir


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known, accepted gap outside default_cache_dir()'s threat model: if skill_dir "
        "IS the OS cache root itself (e.g. skillspector is pointed directly at "
        "%LOCALAPPDATA%/~/.cache), the hashed cache dir is necessarily nested under "
        "skill_dir, so containment is defeated for this self-targeting degenerate case. "
        "The real threat model is untrusted/malicious skill directories being scanned, "
        "not the user pointing the tool at their own cache root. Not fixed by design; "
        "this test documents the gap and must fail loudly (via xfail-strict) if someone "
        "changes default_cache_dir() such that this scenario starts passing without "
        "updating this test."
    ),
)
def test_default_cache_dir_never_under_skill_dir_when_skill_dir_is_cache_root(tmp_path, monkeypatch):
    """Known gap: if skill_dir IS the OS cache root itself (not merely a subdirectory
    of it), the derived cache dir (hashed, under skillspector/llm-cache/<hash>) is
    necessarily nested under skill_dir, so containment is broken for this degenerate
    self-targeting case. This is outside default_cache_dir()'s threat model (malicious
    skill directories being scanned) and is intentionally not handled.
    """
    fake_cache_root = tmp_path / "AppData" / "Local"
    fake_cache_root.mkdir(parents=True)
    monkeypatch.setenv("LOCALAPPDATA", str(fake_cache_root))
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_cache_root))

    # skill_dir literally IS the cache root, not merely a subdirectory of it
    skill_dir = fake_cache_root

    cache_dir = default_cache_dir(skill_dir)
    resolved_skill_dir = skill_dir.resolve()
    resolved_cache_dir = cache_dir.resolve()
    assert resolved_skill_dir not in resolved_cache_dir.parents
    assert resolved_cache_dir != resolved_skill_dir


def test_default_cache_dir_is_stable_and_differs_per_skill_dir(tmp_path):
    """Same skill_dir -> same cache dir; different skill_dir -> different cache dir."""
    skill_dir_a = tmp_path / "skill-a"
    skill_dir_b = tmp_path / "skill-b"
    skill_dir_a.mkdir()
    skill_dir_b.mkdir()

    dir_a1 = default_cache_dir(skill_dir_a)
    dir_a2 = default_cache_dir(skill_dir_a)
    dir_b = default_cache_dir(skill_dir_b)

    assert dir_a1 == dir_a2
    assert dir_a1 != dir_b


def test_llm_response_cache_refuses_symlinked_cache_dir(tmp_path, monkeypatch):
    """LLMResponseCache._connect() must refuse when the cache dir itself is a symlink."""
    real_target = tmp_path / "real_target"
    real_target.mkdir()
    cache_dir = tmp_path / "cache_link"

    # Prefer a real symlink; fall back to mocking Path.is_symlink if unsupported
    # (e.g. no admin/dev-mode privileges on Windows).
    try:
        cache_dir.symlink_to(real_target, target_is_directory=True)
        used_real_symlink = True
    except OSError:
        used_real_symlink = False

    if used_real_symlink:
        cache = LLMResponseCache(cache_dir)
        with pytest.raises(RuntimeError, match="symlink"):
            cache._connect()
    else:
        cache_dir.mkdir()
        cache = LLMResponseCache(cache_dir)
        original_is_symlink = Path.is_symlink

        def fake_is_symlink(self):
            if self == cache._db_path.parent:
                return True
            return original_is_symlink(self)

        monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
        with pytest.raises(RuntimeError, match="symlink"):
            cache._connect()


def test_llm_response_cache_refuses_symlinked_db_file(tmp_path, monkeypatch):
    """get()/put() must not read/write through a symlinked db file."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Pre-seed a fake db file elsewhere and symlink llm_responses.db to it.
    fake_db = tmp_path / "attacker_controlled.db"
    conn = sqlite3.connect(str(fake_db))
    conn.execute(
        "CREATE TABLE llm_responses ("
        "content_hash TEXT, prompt_hash TEXT, schema_version TEXT, response_json TEXT,"
        " created_at TEXT)"
    )
    conn.execute(
        "INSERT INTO llm_responses VALUES ('abc123', 'def456', '1', '{\"evil\": true}', 'now')"
    )
    conn.commit()
    conn.close()

    db_link = cache_dir / "llm_responses.db"

    try:
        db_link.symlink_to(fake_db)
        used_real_symlink = True
    except OSError:
        used_real_symlink = False

    key = CacheKey(content_hash="abc123", prompt_hash="def456", schema_version="1")

    if used_real_symlink:
        cache = LLMResponseCache(cache_dir)
        assert cache.get(key) is None
        cache.put(key, '{"trusted": true}')
        # Verify put() did not write through the symlink into the attacker's db.
        conn = sqlite3.connect(str(fake_db))
        rows = conn.execute("SELECT response_json FROM llm_responses").fetchall()
        conn.close()
        assert rows == [('{"evil": true}',)]
    else:
        cache = LLMResponseCache(cache_dir)
        original_is_symlink = Path.is_symlink

        def fake_is_symlink(self):
            if self == cache._db_path:
                return True
            return original_is_symlink(self)

        monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
        assert cache.get(key) is None
        cache.put(key, '{"trusted": true}')
        # The fake db file must remain untouched.
        conn = sqlite3.connect(str(fake_db))
        rows = conn.execute("SELECT response_json FROM llm_responses").fetchall()
        conn.close()
        assert rows == [('{"evil": true}',)]
