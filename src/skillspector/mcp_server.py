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

"""MCP server exposing SkillSpector scanning as an agent-callable tool.

This lets any MCP-capable agent (Claude Code, Codex CLI, Gemini CLI) or remote
runtime call ``scan_skill`` and gate skill/MCP installs on the verdict, turning
SkillSpector from an out-of-band audit tool into a runtime guardrail.

The scan core (:func:`run_scan`) is deliberately independent of the ``mcp`` SDK
so it can be unit-tested without the optional dependency; :func:`build_server`
wraps it in a FastMCP tool and is only reachable once ``skillspector[mcp]`` is
installed.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Any

from skillspector import __version__
from skillspector.graph import graph
from skillspector.logging_config import get_logger
from skillspector.providers import resolve_provider_credentials

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = get_logger(__name__)

VALID_FORMATS = ("json", "markdown", "sarif", "terminal")

# Mirrors the CLI: a scan scoring above this is treated as unsafe (CLI exits 1).
RISK_THRESHOLD = 50


async def run_scan(
    target: str,
    *,
    use_llm: bool = True,
    output_format: str = "json",
    yara_rules_dir: str | None = None,
) -> dict[str, Any]:
    """Invoke the SkillSpector graph and return a structured verdict.

    Args:
        target: Git URL, file URL, ``.zip``, ``.md`` file, or local directory.
        use_llm: Whether to request the optional LLM semantic pass on top of
            static analysis. Honoured only when provider credentials resolve;
            the returned payload reports what actually happened.
        output_format: Format of the embedded ``report`` string. One of
            :data:`VALID_FORMATS`.
        yara_rules_dir: Optional directory of additional YARA rules.

    Returns:
        A JSON-serialisable verdict with ``risk_score`` (0-100), ``severity``,
        ``recommendation``, ``safe_to_install``, ``findings``, the rendered
        ``report``, and an honest LLM accounting (``llm_requested``,
        ``llm_available``, ``llm_used``, ``scan_mode``) so a caller is never
        misled into thinking a full semantic scan ran when it silently did not.
    """
    if output_format not in VALID_FORMATS:
        raise ValueError(f"output_format must be one of {VALID_FORMATS}, got {output_format!r}")

    llm_available = resolve_provider_credentials() is not None
    llm_used = use_llm and llm_available

    state: dict[str, Any] = {
        "input_path": target,
        "output_format": output_format,
        "use_llm": llm_used,
    }
    if yara_rules_dir:
        state["yara_rules_dir"] = yara_rules_dir

    logger.debug(
        "MCP scan started: target=%s, format=%s, llm_used=%s",
        target,
        output_format,
        llm_used,
    )

    result: dict[str, Any] | None = None
    try:
        result = await graph.ainvoke(
            state,
            config={
                "run_name": "skillspector-mcp-scan",
                "tags": ["skillspector", "mcp"],
                "metadata": {
                    "input_path": target,
                    "use_llm": llm_used,
                    "output_format": output_format,
                    "version": __version__,
                },
            },
        )
        findings = result.get("filtered_findings") or result.get("findings") or []
        risk_score = int(result.get("risk_score") or 0)
        return {
            "target": target,
            "risk_score": risk_score,
            "severity": result.get("risk_severity"),
            "recommendation": result.get("risk_recommendation"),
            "safe_to_install": risk_score <= RISK_THRESHOLD,
            "findings": [f.to_dict() for f in findings],
            "report": result.get("report_body") or "",
            # Honest LLM accounting — never silently imply a full semantic scan.
            "llm_requested": use_llm,
            "llm_available": llm_available,
            "llm_used": llm_used,
            "scan_mode": "static+llm" if llm_used else "static-only",
            "version": __version__,
        }
    finally:
        if result is not None:
            temp_dir = result.get("temp_dir_for_cleanup")
            if temp_dir and isinstance(temp_dir, str):
                shutil.rmtree(temp_dir, ignore_errors=True)


def build_server(name: str = "skillspector") -> FastMCP:
    """Construct the FastMCP server exposing the ``scan_skill`` tool.

    Requires the optional ``mcp`` dependency (``pip install 'skillspector[mcp]'``).
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The MCP server requires the optional 'mcp' dependency. "
            "Install it with: pip install 'skillspector[mcp]'"
        ) from exc

    server = FastMCP(name)

    @server.tool()
    async def scan_skill(
        target: str,
        use_llm: bool = True,
        output_format: str = "json",
    ) -> dict[str, Any]:
        """Scan an AI agent skill for security risks before installing it.

        Use this before installing or loading any skill or MCP server to decide
        whether it is safe. ``target`` accepts a Git URL, file URL, ``.zip``,
        ``.md`` file, or local directory.

        Returns a verdict with ``risk_score`` (0-100), ``severity``,
        ``recommendation``, ``safe_to_install``, and ``findings``. The
        ``llm_used`` / ``scan_mode`` fields report whether the semantic LLM pass
        actually ran, so a low score from a static-only scan is not mistaken for
        a clean full scan.
        """
        return await run_scan(target, use_llm=use_llm, output_format=output_format)

    return server


def run(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the MCP server over ``stdio`` (local agents) or ``http`` (remote/A2A)."""
    server = build_server()
    if transport == "stdio":
        server.run(transport="stdio")
    elif transport == "http":
        server.settings.host = host
        server.settings.port = port
        server.run(transport="streamable-http")
    else:
        raise ValueError(f"transport must be 'stdio' or 'http', got {transport!r}")
