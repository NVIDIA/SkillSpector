# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end coverage for byte-derived Python execution surfaces."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from skillspector.graph import graph


def _scan(root: Path) -> dict:
    return graph.invoke(
        {
            "input_path": str(root),
            "output_format": "json",
            "use_llm": False,
        }
    )


def _write_bundle(root: Path, files: dict[str, str | bytes]) -> None:
    for relative_path, content in files.items():
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")


def _tm1_paths(result: dict) -> set[str]:
    return {finding.file for finding in result["filtered_findings"] if finding.rule_id == "TM1"}


@pytest.mark.parametrize(
    ("filename", "prefix"),
    [
        pytest.param("run.pyw", "", id="python-window"),
        pytest.param("run", "#!/usr/bin/env python3\n", id="extensionless-shebang"),
    ],
)
def test_python_execution_surfaces_reach_static_and_behavioral_analyzers(
    tmp_path: Path,
    filename: str,
    prefix: str,
) -> None:
    _write_bundle(
        tmp_path,
        {
            "SKILL.md": "# Python helper",
            filename: (
                prefix
                + "import subprocess\n"
                + "enabled = True\n"
                + "subprocess.run(command, shell=enabled)\n"
            ),
        },
    )

    result = _scan(tmp_path)
    metadata = next(row for row in result["component_metadata"] if row["path"] == filename)

    assert filename in _tm1_paths(result)
    assert metadata["type"] == "python"
    assert result["analysis_completeness"]["is_complete"] is True


@pytest.mark.parametrize("selector", ["-s", "--script", "--gui-script"])
def test_uv_script_launcher_reaches_static_analyzers(tmp_path: Path, selector: str) -> None:
    filename = "runner"
    _write_bundle(
        tmp_path,
        {
            "SKILL.md": "# uv Python helper",
            filename: (
                f"#!/usr/bin/env -S uv run {selector}\n"
                "import subprocess\n"
                "enabled = True\n"
                "subprocess.run(command, shell=enabled)\n"
            ),
        },
    )
    (tmp_path / filename).chmod(0o755)

    result = _scan(tmp_path)
    metadata = next(row for row in result["component_metadata"] if row["path"] == filename)

    assert filename in _tm1_paths(result)
    assert metadata["type"] == "python"
    assert metadata["executable"] is True
    assert result["analysis_completeness"]["is_complete"] is True


def test_uv_run_without_script_selector_is_analyzed_fail_closed(tmp_path: Path) -> None:
    filename = "runner"
    _write_bundle(
        tmp_path,
        {
            "SKILL.md": "# Ambiguous uv helper",
            filename: (
                "#!/usr/bin/env -S uv run\n"
                "import subprocess\n"
                "enabled = True\n"
                "subprocess.run(command, shell=enabled)\n"
            ),
        },
    )
    (tmp_path / filename).chmod(0o755)

    result = _scan(tmp_path)
    exceptions = result["analysis_completeness"]["ledger_exceptions"]

    assert filename in _tm1_paths(result)
    assert result["analysis_completeness"]["is_complete"] is False
    assert any(
        row["path"] == filename and row["reason_code"] == "python_source_ambiguous"
        for row in exceptions
    )


def test_uv_filesystem_alias_is_analyzed_fail_closed(tmp_path: Path) -> None:
    filename = "runner"
    _write_bundle(
        tmp_path,
        {
            "SKILL.md": "# Platform-dependent uv helper",
            filename: (
                "#!/usr/bin/env -S UV run --script\n"
                "import subprocess\n"
                "enabled = True\n"
                "subprocess.run(command, shell=enabled)\n"
            ),
        },
    )
    (tmp_path / filename).chmod(0o755)

    result = _scan(tmp_path)
    exceptions = result["analysis_completeness"]["ledger_exceptions"]

    assert filename in _tm1_paths(result)
    assert result["analysis_completeness"]["is_complete"] is False
    assert any(
        row["path"] == filename and row["reason_code"] == "python_source_ambiguous"
        for row in exceptions
    )


def test_uv_global_option_script_launcher_is_analyzed_fail_closed(tmp_path: Path) -> None:
    filename = "runner"
    _write_bundle(
        tmp_path,
        {
            "SKILL.md": "# uv global-option helper",
            filename: (
                "#!/usr/bin/env -S uv --offline run --script\n"
                "import subprocess\n"
                "enabled = True\n"
                "subprocess.run(command, shell=enabled)\n"
            ),
        },
    )
    (tmp_path / filename).chmod(0o755)

    result = _scan(tmp_path)
    exceptions = result["analysis_completeness"]["ledger_exceptions"]

    assert filename in _tm1_paths(result)
    assert result["analysis_completeness"]["is_complete"] is False
    assert any(
        row["path"] == filename and row["reason_code"] == "python_source_ambiguous"
        for row in exceptions
    )


def test_ambiguous_python_surface_is_analyzed_fail_closed(tmp_path: Path) -> None:
    filename = "runner"
    _write_bundle(
        tmp_path,
        {
            "SKILL.md": "# Platform-dependent helper",
            filename: (
                "#!/usr/bin/env -i python3\n"
                "import subprocess\n"
                "enabled = True\n"
                "subprocess.run(command, shell=enabled)\n"
            ),
        },
    )

    result = _scan(tmp_path)
    exceptions = result["analysis_completeness"]["ledger_exceptions"]

    assert filename in _tm1_paths(result)
    assert result["analysis_completeness"]["is_complete"] is False
    assert any(
        row["path"] == filename and row["reason_code"] == "python_source_ambiguous"
        for row in exceptions
    )


def test_pep263_python_source_is_decoded_before_analysis(tmp_path: Path) -> None:
    filename = "run.py"
    source = (
        "# coding: latin-1\n# café\nimport subprocess\nsubprocess.run(command, shell=True)\n"
    ).encode("latin-1")
    _write_bundle(tmp_path, {"SKILL.md": "# Encoded helper", filename: source})

    result = _scan(tmp_path)

    assert filename in _tm1_paths(result)
    assert "café" in result["local_file_cache"][filename]
    assert result["analysis_completeness"]["is_complete"] is True


def test_nested_extensionless_python_surface_is_analyzed(tmp_path: Path) -> None:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr(
            "runner",
            "#!/usr/bin/env python3\n"
            "import subprocess\n"
            "enabled = True\n"
            "subprocess.run(command, shell=enabled)\n",
        )
    _write_bundle(
        tmp_path,
        {"SKILL.md": "# Nested helper", "bundle.zip": archive_bytes.getvalue()},
    )

    result = _scan(tmp_path)
    virtual_path = "bundle.zip!/runner"

    assert virtual_path in _tm1_paths(result)
    assert result["analysis_completeness"]["is_complete"] is True


def test_python_declared_encoding_failure_is_partial(tmp_path: Path) -> None:
    filename = "broken.py"
    _write_bundle(
        tmp_path,
        {
            "SKILL.md": "# Broken helper",
            filename: b"# coding: ascii\nname = '\xff'\n",
        },
    )

    result = _scan(tmp_path)
    artifact = next(row for row in result["artifact_inventory"] if row["path"] == filename)

    assert filename not in result["local_file_cache"]
    assert artifact["disposition"] == "partial"
    assert artifact["reason"] == "python_source_decode_error"
    assert result["analysis_completeness"]["is_complete"] is False
