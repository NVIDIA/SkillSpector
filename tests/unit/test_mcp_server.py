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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from skillspector import mcp_server
from skillspector.mcp_server import run_scan


def _write_skill(tmp_path: Path, body: str = "# Safe skill") -> Path:
    (tmp_path / "SKILL.md").write_text(f"---\nname: mcp-test\n---\n{body}", encoding="utf-8")
    return tmp_path


async def test_run_scan_returns_structured_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_scan returns a JSON-serialisable verdict with the expected shape."""
    # No credentials: the LLM pass cannot run regardless of what is requested.
    monkeypatch.setattr(mcp_server, "resolve_provider_credentials", lambda: None)
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
    monkeypatch.setattr(mcp_server, "resolve_provider_credentials", lambda: None)
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
    monkeypatch.setattr(mcp_server, "resolve_provider_credentials", lambda: ("key", None))
    _write_skill(tmp_path)

    result = await run_scan(str(tmp_path), use_llm=False, output_format="json")

    assert result["llm_available"] is True
    assert result["llm_requested"] is False
    assert result["llm_used"] is False
    assert result["scan_mode"] == "static-only"


async def test_run_scan_rejects_invalid_format(tmp_path: Path) -> None:
    """An unsupported output_format is rejected before any scan runs."""
    with pytest.raises(ValueError):
        await run_scan(str(tmp_path), output_format="xml")


async def test_run_scan_rejects_local_target_when_disallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTP-style scans reject local targets before the graph is invoked."""
    graph_ainvoke = AsyncMock()
    monkeypatch.setattr(mcp_server.graph, "ainvoke", graph_ainvoke)
    monkeypatch.setattr(mcp_server, "resolve_provider_credentials", lambda: None)

    with pytest.raises(ValueError, match="local targets are disabled"):
        await run_scan(str(tmp_path), allow_local_targets=False)

    assert graph_ainvoke.await_count == 0


async def test_run_scan_rejects_file_url_when_local_targets_disallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same HTTP guard rejects file:// targets before any scan runs."""
    graph_ainvoke = AsyncMock()
    monkeypatch.setattr(mcp_server.graph, "ainvoke", graph_ainvoke)
    monkeypatch.setattr(mcp_server, "resolve_provider_credentials", lambda: None)

    with pytest.raises(ValueError, match="local targets are disabled"):
        await run_scan(tmp_path.as_uri(), allow_local_targets=False)

    assert graph_ainvoke.await_count == 0


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (r"\\server\share\skill", True),
        ("//server/share/skill", True),
        ("git@github.com:NVIDIA/SkillSpector.git", False),
        ("ssh://git@github.com/NVIDIA/SkillSpector.git", False),
        ("git+ssh://git@github.com/NVIDIA/SkillSpector.git", False),
        ("custom://example/skill", False),
    ],
)
def test_is_local_target_classifies_protocol_edges(target: str, expected: bool) -> None:
    """Classifier treats UNC-style paths as local and known remote schemes as remote."""
    assert mcp_server._is_local_target(target) is expected


def test_is_local_target_checks_relative_paths_from_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing relative paths are local; missing relative paths stay unresolved."""
    (tmp_path / "skill").mkdir()
    monkeypatch.chdir(tmp_path)

    assert mcp_server._is_local_target("skill") is True
    assert mcp_server._is_local_target("missing-skill") is False


async def test_run_scan_allows_remote_target_when_local_targets_disallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote HTTP targets still reach the resolver path when local targets are blocked."""
    graph_ainvoke = AsyncMock(
        return_value={
            "risk_score": 0,
            "risk_severity": "low",
            "risk_recommendation": "safe",
            "filtered_findings": [],
            "report_body": "ok",
        }
    )
    monkeypatch.setattr(mcp_server.graph, "ainvoke", graph_ainvoke)
    monkeypatch.setattr(mcp_server, "resolve_provider_credentials", lambda: None)

    target = "https://example.com/skills/safe.git"
    result = await run_scan(target, allow_local_targets=False)

    assert result["target"] == target
    assert graph_ainvoke.await_count == 1
    assert graph_ainvoke.await_args.args[0]["input_path"] == target


async def test_run_scan_keeps_default_local_target_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default run_scan path still accepts local targets."""
    graph_ainvoke = AsyncMock(
        return_value={
            "risk_score": 0,
            "risk_severity": "low",
            "risk_recommendation": "safe",
            "filtered_findings": [],
            "report_body": "ok",
        }
    )
    monkeypatch.setattr(mcp_server.graph, "ainvoke", graph_ainvoke)
    monkeypatch.setattr(mcp_server, "resolve_provider_credentials", lambda: None)

    result = await run_scan(str(tmp_path))

    assert result["target"] == str(tmp_path)
    assert graph_ainvoke.await_count == 1
    assert graph_ainvoke.await_args.args[0]["input_path"] == str(tmp_path)


@pytest.mark.parametrize(
    ("transport", "expected_allow_local_targets"),
    [("stdio", True), ("http", False)],
)
def test_run_passes_transport_local_target_policy(
    transport: str,
    expected_allow_local_targets: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run() keeps stdio local scans available and disables them for HTTP."""
    captured: dict[str, bool] = {}
    server = SimpleNamespace(
        settings=SimpleNamespace(host=None, port=None),
        run=MagicMock(),
    )

    def fake_build_server(*, allow_local_targets: bool = True):
        captured["allow_local_targets"] = allow_local_targets
        return server

    monkeypatch.setattr(mcp_server, "build_server", fake_build_server)

    mcp_server.run(transport=transport, host="0.0.0.0", port=9000)

    assert captured["allow_local_targets"] is expected_allow_local_targets
    if transport == "http":
        assert server.settings.host == "0.0.0.0"
        assert server.settings.port == 9000
        server.run.assert_called_once_with(transport="streamable-http")
    else:
        server.run.assert_called_once_with(transport="stdio")


async def test_build_server_registers_scan_skill() -> None:
    """build_server wires up the scan_skill tool (requires the mcp extra)."""
    pytest.importorskip("mcp")

    server = mcp_server.build_server()
    tools = await server.list_tools()
    assert "scan_skill" in {tool.name for tool in tools}
