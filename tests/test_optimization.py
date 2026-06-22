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

"""Unit tests for SkillSpector caching, smart chunking, and pre-filtering optimizations."""

from __future__ import annotations

import os
import shutil
import sqlite3
import unittest.mock
from pathlib import Path

import pytest

from skillspector import cache
from skillspector.llm_analyzer_base import Batch, LLMAnalysisResult, LLMAnalyzerBase, chunk_file_by_lines, is_relevant_for_llm
from skillspector.models import Finding


@pytest.fixture(autouse=True)
def clean_cache_dir():
    """Ensure a clean cache database directory for each test."""
    db_path = cache.get_cache_db_path()
    if db_path.exists():
        try:
            db_path.unlink()
        except OSError:
            pass
    yield
    if db_path.exists():
        try:
            db_path.unlink()
        except OSError:
            pass


def test_is_relevant_for_llm():
    """Verify that is_relevant_for_llm correctly flags static/config files."""
    # Should skip static assets
    assert not is_relevant_for_llm("src/index.css")
    assert not is_relevant_for_llm("assets/logo.svg")
    assert not is_relevant_for_llm("src-tauri/icons/icon.icns")
    assert not is_relevant_for_llm("src-tauri/icons/icon.ico")

    # Should skip common JSON config files
    assert not is_relevant_for_llm("tsconfig.json")
    assert not is_relevant_for_llm("package-lock.json")
    assert not is_relevant_for_llm("src-tauri/tauri.conf.json")

    # Should keep source files and documentation/manifests
    assert is_relevant_for_llm("src/App.tsx")
    assert is_relevant_for_llm("src-tauri/src/main.rs")
    assert is_relevant_for_llm("README.md")
    assert is_relevant_for_llm("SKILL.md")
    assert is_relevant_for_llm("skill.json")


def test_smart_block_chunking():
    """Verify chunk_file_by_lines splits along block or empty line boundaries."""
    # A mocked code content with lines
    content = (
        "import sys\n"  # 11 chars ~ 2 tokens
        "import os\n"   # 10 chars ~ 2 tokens
        "\n"            # 1 char ~ 0 tokens
        "def foo():\n"  # 11 chars ~ 2 tokens
        "    print('hello')\n"
        "    print('world')\n"
        "\n"
        "def bar():\n"
        "    print('test')\n"
    )

    # Let's check splitting when budget is tight.
    # If we split strictly, it might cut def foo() in half.
    # With smart block splitting, it should split at blank lines or class/fn keywords.
    # Estimate of max_tokens is based on CHARS_PER_TOKEN = 4.
    # total len = ~90 chars, ~22 tokens. Let's set max_tokens to 10.
    chunks = chunk_file_by_lines(content, max_tokens=10, overlap_lines=0)
    assert len(chunks) >= 2
    # Check that first chunk ends before def bar() or at a blank line
    first_chunk_text = chunks[0][0]
    assert "def bar():" not in first_chunk_text
    # Should split before "def foo():" to keep the block intact
    assert "def foo():" not in first_chunk_text
    assert "def foo():" in chunks[1][0]


def test_persistent_sqlite_cache():
    """Verify set_cached_findings and get_cached_findings write/read database."""
    cache.initialize_cache_db()
    db_path = cache.get_cache_db_path()
    assert db_path.exists()

    cache_key = "test_key_123"
    findings_json = '{"findings": []}'

    # Read from empty cache
    assert cache.get_cached_findings(cache_key) is None

    # Write and read back
    cache.set_cached_findings(cache_key, findings_json, "test_analyzer", "hash_abc", "gpt-4")
    assert cache.get_cached_findings(cache_key) == findings_json


@pytest.mark.asyncio
async def test_llm_analyzer_cache_integration():
    """Verify that run_batches and arun_batches hit the cache and skip LLM calls."""
    # Create an analyzer instance
    analyzer = LLMAnalyzerBase("base prompt", "gpt-4")

    # Mock the LLM invocation using object.__setattr__ to bypass Pydantic restrictions
    mock_invoke = unittest.mock.MagicMock(return_value=LLMAnalysisResult(findings=[]))
    mock_llm = unittest.mock.MagicMock()
    mock_structured = unittest.mock.MagicMock()
    object.__setattr__(mock_llm, 'invoke', mock_invoke)
    object.__setattr__(mock_structured, 'invoke', mock_invoke)
    
    object.__setattr__(analyzer, '_llm', mock_llm)
    object.__setattr__(analyzer, '_structured_llm', mock_structured)

    batches = [Batch("test.py", "print('hello')")]

    # First run: cache miss, invokes LLM
    results = analyzer.run_batches(batches)
    assert len(results) == 1
    assert mock_invoke.call_count == 1

    # Second run: cache hit, does NOT invoke LLM
    mock_invoke.reset_mock()
    results_cached = analyzer.run_batches(batches)
    assert len(results_cached) == 1
    assert mock_invoke.call_count == 0

    # Test arun_batches async caching
    mock_ainvoke = unittest.mock.AsyncMock(return_value=LLMAnalysisResult(findings=[]))
    mock_llm_async = unittest.mock.MagicMock()
    mock_structured_async = unittest.mock.MagicMock()
    object.__setattr__(mock_llm_async, 'ainvoke', mock_ainvoke)
    object.__setattr__(mock_structured_async, 'ainvoke', mock_ainvoke)
    
    object.__setattr__(analyzer, '_llm', mock_llm_async)
    object.__setattr__(analyzer, '_structured_llm', mock_structured_async)

    # For async, we run with a new file content to force cache miss
    batches_async = [Batch("test2.py", "print('hello async')")]
    
    # First async run: cache miss
    results_async = await analyzer.arun_batches(batches_async)
    assert len(results_async) == 1
    assert mock_ainvoke.call_count == 1

    # Second async run: cache hit
    mock_ainvoke.reset_mock()
    results_async_cached = await analyzer.arun_batches(batches_async)
    assert len(results_async_cached) == 1
    assert mock_ainvoke.call_count == 0
