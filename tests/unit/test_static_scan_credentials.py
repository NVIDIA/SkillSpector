# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static scans never consult provider credentials, including AWS credential processes."""

from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Iterator
from pathlib import Path

import boto3
import pytest
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.mcp_server import run_scan
from skillspector.nodes.report import _build_metadata
from skillspector.providers import reset_provider, use_provider
from skillspector.providers.bedrock import BedrockProvider


@pytest.fixture(params=[False, True], ids=["unbound", "bound"])
def aws_credential_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> Iterator[tuple[list[object], Path]]:
    """Use a real AWS credential process and record any credential-chain lookup."""
    tmp_path = tmp_path_factory.mktemp("aws-process")
    marker = tmp_path / "credential-process-ran"
    script = tmp_path / "credentials.py"
    payload = json.dumps(
        {
            "Version": 1,
            "AccessKeyId": "test-access-key",
            "SecretAccessKey": "test-secret-key",
            "SessionToken": "test-session-token",
        }
    )
    script.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        f"print({payload!r})\n",
        encoding="utf-8",
    )
    config = tmp_path / "aws-config"
    config.write_text(
        "[default]\nregion = us-west-2\ncredential_process = "
        + shlex.join([sys.executable, str(script)])
        + "\n",
        encoding="utf-8",
    )
    credentials = tmp_path / "aws-credentials"
    credentials.write_text("", encoding="utf-8")
    for name in (
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_ROLE_ARN",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "SKILLSPECTOR_MODEL",
        "SKILLSPECTOR_TEMPERATURE",
        "SKILLSPECTOR_REASONING_EFFORT",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "bedrock")
    monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "leftover-test-key")
    lookups: list[object] = []
    original = boto3.Session.get_credentials

    def record_lookup(session: boto3.Session):
        lookups.append(session)
        return original(session)

    monkeypatch.setattr(boto3.Session, "get_credentials", record_lookup)
    token = use_provider(BedrockProvider()) if request.param else None
    try:
        yield lookups, marker
    finally:
        if token is not None:
            reset_provider(token)


@pytest.mark.parametrize("output_format", ["json", "markdown", "sarif", "terminal"])
def test_cli_static_scan_never_looks_up_or_executes_aws_credentials(
    tmp_path: Path,
    safe_skill_dir: Path,
    aws_credential_process: tuple[list[object], Path],
    output_format: str,
) -> None:
    lookups, marker = aws_credential_process
    output = tmp_path / "report.txt"

    result = CliRunner().invoke(
        app,
        ["scan", str(safe_skill_dir), "--no-llm", "--format", output_format, "-o", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert output.read_text(encoding="utf-8")
    assert lookups == []
    assert not marker.exists()
    if output_format == "json":
        metadata = json.loads(output.read_text(encoding="utf-8"))["metadata"]
        assert metadata["llm_requested"] is False
        assert metadata["llm_available"] is False


@pytest.mark.parametrize("output_format", ["json", "markdown", "sarif", "terminal"])
async def test_mcp_static_scan_never_looks_up_or_executes_aws_credentials(
    safe_skill_dir: Path,
    aws_credential_process: tuple[list[object], Path],
    output_format: str,
) -> None:
    lookups, marker = aws_credential_process

    result = await run_scan(str(safe_skill_dir), use_llm=False, output_format=output_format)

    assert result["report"]
    assert result["llm_requested"] is False
    assert result["llm_available"] is False
    assert result["scan_mode"] == "static-only"
    assert lookups == []
    assert not marker.exists()


def test_static_report_metadata_never_looks_up_or_executes_aws_credentials(
    aws_credential_process: tuple[list[object], Path],
) -> None:
    lookups, marker = aws_credential_process

    metadata = _build_metadata(has_executable_scripts=False, use_llm=False)

    assert metadata["llm_requested"] is False
    assert metadata["llm_available"] is False
    assert lookups == []
    assert not marker.exists()
