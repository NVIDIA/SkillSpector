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

"""Tests for the MCP server wrapper (run_scan core + scan_skill tool)."""

import asyncio
import os
import sys
from pathlib import Path

import pytest

from skillspector import mcp_server
from skillspector.mcp_server import run_scan
from skillspector.providers import reset_provider, use_provider


def _write_skill(tmp_path: Path, body: str = "# Safe skill") -> Path:
    (tmp_path / "SKILL.md").write_text(f"---\nname: mcp-test\n---\n{body}", encoding="utf-8")
    return tmp_path


async def test_run_scan_returns_structured_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_scan returns a JSON-serialisable verdict with the expected shape."""
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (False, "no llm"))
    _write_skill(tmp_path)

    result = await run_scan(str(tmp_path), use_llm=True, output_format="json")

    assert result["target"] == str(tmp_path)
    assert isinstance(result["risk_score"], int)
    assert 0 <= result["risk_score"] <= 100
    assert isinstance(result["findings"], list)
    assert isinstance(result["safe_to_install"], bool)
    assert result["safe_to_install"] == (result["risk_score"] <= 50)
    assert result["report"]  # non-empty rendered report


async def test_run_scan_llm_accounting_is_honest_without_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requesting the LLM with no credentials must report it as not used."""
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (False, "no llm"))
    _write_skill(tmp_path)

    result = await run_scan(str(tmp_path), use_llm=True, output_format="json")

    assert result["llm_requested"] is True
    assert result["llm_available"] is False
    assert result["llm_used"] is False
    assert result["scan_mode"] == "static-only"


async def test_run_scan_reports_llm_available_with_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Credentials present but use_llm=False: available, but honestly not used."""
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (True, None))
    _write_skill(tmp_path)

    result = await run_scan(str(tmp_path), use_llm=False, output_format="json")

    assert result["llm_available"] is True
    assert result["llm_requested"] is False
    assert result["llm_used"] is False
    assert result["scan_mode"] == "static-only"


async def test_run_scan_uses_bound_provider_without_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An injected provider can own the LLM client without exposing raw credentials."""

    class _InjectedProvider:
        DEFAULT_MODEL = "injected-default"
        SLOT_DEFAULTS = {"meta_analyzer": "injected-default"}

        def get_context_length(self, model: str) -> int | None:
            return 4096 if model == "injected-default" else None

        def get_max_output_tokens(self, model: str) -> int | None:
            return 128 if model == "injected-default" else None

        def resolve_model(self, slot: str = "default") -> str:
            return "injected-default"

        def resolve_credentials(self) -> tuple[str, str | None] | None:
            return None

        def create_chat_model(
            self,
            model: str,
            *,
            max_tokens: int,
            timeout: float | None = 120,
        ) -> object:
            return object()

    class _Graph:
        async def ainvoke(self, state, config):
            assert state["use_llm"] is True
            return {
                "filtered_findings": [],
                "risk_score": 0,
                "risk_severity": "LOW",
                "risk_recommendation": "OK",
                "report_body": "report",
            }

    token = use_provider(_InjectedProvider())
    monkeypatch.setattr(mcp_server, "graph", _Graph())
    _write_skill(tmp_path)

    try:
        result = await run_scan(str(tmp_path), use_llm=True, output_format="json")
    finally:
        reset_provider(token)

    assert result["llm_available"] is True
    assert result["llm_requested"] is True
    assert result["llm_used"] is True
    assert result["scan_mode"] == "static+llm"


async def test_run_scan_disables_llm_for_unavailable_bound_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bound provider that cannot build a chat model must stay static-only."""

    class _UnavailableInjectedProvider:
        DEFAULT_MODEL = "injected-default"
        SLOT_DEFAULTS = {"meta_analyzer": "injected-default"}

        def get_context_length(self, model: str) -> int | None:
            return 4096 if model == "injected-default" else None

        def get_max_output_tokens(self, model: str) -> int | None:
            return 128 if model == "injected-default" else None

        def resolve_model(self, slot: str = "default") -> str:
            return "injected-default"

        def resolve_credentials(self) -> tuple[str, str | None] | None:
            return None

        def create_chat_model(
            self,
            model: str,
            *,
            max_tokens: int,
            timeout: float | None = 120,
        ) -> object | None:
            return None

    class _Graph:
        async def ainvoke(self, state, config):
            assert state["use_llm"] is False
            return {
                "filtered_findings": [],
                "risk_score": 0,
                "risk_severity": "LOW",
                "risk_recommendation": "OK",
                "report_body": "report",
            }

    token = use_provider(_UnavailableInjectedProvider())
    monkeypatch.setattr(mcp_server, "graph", _Graph())
    _write_skill(tmp_path)

    try:
        result = await run_scan(str(tmp_path), use_llm=True, output_format="json")
    finally:
        reset_provider(token)

    assert result["llm_available"] is False
    assert result["llm_requested"] is True
    assert result["llm_used"] is False
    assert result["scan_mode"] == "static-only"


async def test_run_scan_rejects_invalid_format(tmp_path: Path) -> None:
    """An unsupported output_format is rejected before any scan runs."""
    with pytest.raises(ValueError):
        await run_scan(str(tmp_path), output_format="xml")


async def test_build_server_registers_scan_skill() -> None:
    """build_server wires up the scan_skill tool (requires the mcp extra)."""
    pytest.importorskip("mcp")

    server = mcp_server.build_server()
    tools = await server.list_tools()
    assert "scan_skill" in {tool.name for tool in tools}


async def test_mcp_stdio_initialize_registers_scan_skill() -> None:
    """The real stdio CLI must initialize and expose the scan_skill tool."""
    pytest.importorskip("mcp")

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    repo_root = Path(__file__).resolve().parents[2]
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "skillspector.cli", "mcp"],
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=15)
            tools = await asyncio.wait_for(session.list_tools(), timeout=15)

    assert "scan_skill" in {tool.name for tool in tools.tools}
