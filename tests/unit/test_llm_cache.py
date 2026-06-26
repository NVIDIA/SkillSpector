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
from pathlib import Path
import pytest
from skillspector.llm_cache import LLMResponseCache, CacheKey


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
