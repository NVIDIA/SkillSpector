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

"""Tests for SARIF 2.1.0 Pydantic model and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from skillspector.sarif_models import (
    SARIF_SCHEMA_URI,
    SarifArtifactLocation,
    SarifDriver,
    SarifLocation,
    SarifLog,
    SarifMessage,
    SarifPhysicalLocation,
    SarifRegion,
    SarifResult,
    SarifRun,
    SarifTool,
    validate_sarif_report,
)


def test_sarif_log_minimal_roundtrip() -> None:
    """Building a minimal SarifLog and dumping to dict yields valid SARIF shape."""
    log = SarifLog(
        schema_=SARIF_SCHEMA_URI,
        runs=[
            SarifRun(
                tool=SarifTool(driver=SarifDriver(name="test-tool")),
                results=[
                    SarifResult(
                        ruleId="R001",
                        message=SarifMessage(text="Finding message"),
                        level="warning",
                        locations=[
                            SarifLocation(
                                physicalLocation=SarifPhysicalLocation(
                                    artifactLocation=SarifArtifactLocation(uri="file.py"),
                                    region=SarifRegion(startLine=1),
                                )
                            )
                        ],
                    )
                ],
            )
        ],
    )
    data = log.model_dump(mode="json", by_alias=True)
    assert data["version"] == "2.1.0"
    assert data["$schema"] == SARIF_SCHEMA_URI
    assert len(data["runs"]) == 1
    assert data["runs"][0]["tool"]["driver"]["name"] == "test-tool"
    assert len(data["runs"][0]["results"]) == 1
    assert data["runs"][0]["results"][0]["message"]["text"] == "Finding message"
    assert (
        data["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["region"]["startLine"]
        == 1
    )


def test_validate_sarif_report_accepts_valid() -> None:
    """validate_sarif_report accepts a valid minimal SARIF dict."""
    valid = {
        "version": "2.1.0",
        "runs": [
            {
                "tool": {"driver": {"name": "Tool"}},
                "results": [
                    {
                        "message": {"text": "msg"},
                        "ruleId": "R1",
                        "level": "warning",
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": "file.py"},
                                    "region": {"startLine": 1},
                                }
                            }
                        ],
                    }
                ],
            }
        ],
    }
    validate_sarif_report(valid)


def test_validate_sarif_report_rejects_not_dict() -> None:
    """validate_sarif_report rejects non-dict."""
    with pytest.raises(ValidationError):
        validate_sarif_report([])
    with pytest.raises(ValidationError):
        validate_sarif_report("2.1.0")


def test_validate_sarif_report_rejects_wrong_version() -> None:
    """validate_sarif_report rejects wrong version."""
    with pytest.raises(ValidationError):
        validate_sarif_report(
            {"version": "1.0", "runs": [{"tool": {"driver": {"name": "T"}}, "results": []}]}
        )


def test_validate_sarif_report_rejects_empty_runs() -> None:
    """validate_sarif_report rejects empty runs array."""
    with pytest.raises(ValidationError):
        validate_sarif_report({"version": "2.1.0", "runs": []})


def test_validate_sarif_report_rejects_result_without_message_text() -> None:
    """validate_sarif_report rejects result without message.text."""
    with pytest.raises(ValidationError):
        validate_sarif_report(
            {
                "version": "2.1.0",
                "runs": [
                    {
                        "tool": {"driver": {"name": "T"}},
                        "results": [{"ruleId": "R1"}],
                    }
                ],
            }
        )


def test_sarif_artifact_location_uri_base_id_roundtrip() -> None:
    """uriBaseId survives a model round trip through the SARIF 2.1.0 alias."""
    location = SarifArtifactLocation(uri="SKILL.md", uriBaseId="SKILLROOT")
    data = location.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert data == {"uri": "SKILL.md", "uriBaseId": "SKILLROOT"}
    assert SarifArtifactLocation.model_validate(data).uri_base_id == "SKILLROOT"


def test_sarif_run_original_uri_base_ids_roundtrip() -> None:
    """originalUriBaseIds survives a run round trip and validates raw dicts."""
    run = SarifRun(
        tool=SarifTool(driver=SarifDriver(name="skillspector")),
        originalUriBaseIds={"SKILLROOT": {"uri": "malicious_skill/"}},
    )
    data = run.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert data["originalUriBaseIds"] == {"SKILLROOT": {"uri": "malicious_skill/"}}
    validate_sarif_report({"version": "2.1.0", "runs": [data]})


def test_sarif_run_without_base_ids_omits_them() -> None:
    """Runs that do not set base IDs keep the previous wire shape."""
    run = SarifRun(
        tool=SarifTool(driver=SarifDriver(name="skillspector")),
        results=[
            SarifResult(
                ruleId="R1",
                message=SarifMessage(text="msg"),
                locations=[
                    SarifLocation(
                        physicalLocation=SarifPhysicalLocation(
                            artifactLocation=SarifArtifactLocation(uri="SKILL.md"),
                        )
                    )
                ],
            )
        ],
    )
    data = run.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert "originalUriBaseIds" not in data
    artifact = data["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"]
    assert artifact == {"uri": "SKILL.md"}
