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

"""SARIF 2.1.0 Pydantic models for report output (OASIS spec)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

SARIF_SCHEMA_URI = "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0-rtm.4.json"


class SarifRegion(BaseModel):
    """Region within an artifact (line/column range)."""

    model_config = {"populate_by_name": True}

    start_line: int = Field(alias="startLine")
    start_column: int | None = Field(default=None, alias="startColumn")
    end_line: int | None = Field(default=None, alias="endLine")
    end_column: int | None = Field(default=None, alias="endColumn")


class SarifArtifactLocation(BaseModel):
    """Reference to an artifact (file) in the run."""

    model_config = {"populate_by_name": True}

    uri: str
    # SARIF 2.1.0 section 3.4.4: base URI identifier resolved through the
    # run's originalUriBaseIds. Recursive scans scope each child run's
    # skill-relative URIs to its own skill directory with this.
    uri_base_id: str | None = Field(default=None, alias="uriBaseId")
    index: int | None = None
    properties: dict[str, object] | None = None


class SarifPhysicalLocation(BaseModel):
    """Physical location (artifact + optional region)."""

    model_config = {"populate_by_name": True}

    artifact_location: SarifArtifactLocation = Field(alias="artifactLocation")
    region: SarifRegion | None = None


class SarifLocation(BaseModel):
    """Result location (physical and/or logical)."""

    model_config = {"populate_by_name": True}

    physical_location: SarifPhysicalLocation = Field(alias="physicalLocation")


class SarifMessage(BaseModel):
    """SARIF message object (required: text)."""

    text: str


class SarifSuppression(BaseModel):
    """SARIF suppression object — marks a result as suppressed (e.g. via a baseline)."""

    kind: Literal["inSource", "external"] = "external"
    justification: str | None = None


class SarifResult(BaseModel):
    """A single analysis result (finding)."""

    model_config = {"populate_by_name": True}

    rule_id: str = Field(alias="ruleId")
    message: SarifMessage
    level: Literal["error", "warning", "note"] = "warning"
    locations: list[SarifLocation]
    # When present, the result is suppressed; SARIF consumers (e.g. GitHub code
    # scanning) exclude suppressed results from counts but keep them for audit.
    suppressions: list[SarifSuppression] | None = None
    properties: dict[str, object] | None = None


class SarifReportingDescriptor(BaseModel):
    """Rule metadata (SARIF reportingDescriptor)."""

    model_config = {"populate_by_name": True}

    id: str
    short_description: SarifMessage | None = Field(default=None, alias="shortDescription")
    default_configuration: dict[str, object] | None = Field(
        default=None, alias="defaultConfiguration"
    )
    # Rule-level metadata such as security-severity, emitted only when the
    # rule has a single severity across every reported finding.
    properties: dict[str, object] | None = None


class SarifDriver(BaseModel):
    """Tool driver (required: name; optional: version, rules)."""

    name: str
    version: str | None = None
    rules: list[SarifReportingDescriptor] | None = None


class SarifTool(BaseModel):
    """Tool that produced the run."""

    driver: SarifDriver


class SarifArtifact(BaseModel):
    """Artifact (file) analyzed in the run."""

    location: SarifArtifactLocation


class SarifNotification(BaseModel):
    """A notification about a condition encountered during tool execution."""

    text: SarifMessage = Field(alias="message")
    level: Literal["error", "warning", "note"] = "warning"
    locations: list[SarifLocation] | None = None
    properties: dict[str, object] | None = None

    model_config = {"populate_by_name": True}


class SarifInvocation(BaseModel):
    """Describes a single tool invocation (SARIF ``run.invocations[]``).

    ``executionSuccessful`` is derived from the canonical inspection ledger.
    """

    model_config = {"populate_by_name": True}

    execution_successful: bool = Field(alias="executionSuccessful")
    tool_execution_notifications: list[SarifNotification] | None = Field(
        default=None, alias="toolExecutionNotifications"
    )
    properties: dict[str, object] | None = None


class SarifRun(BaseModel):
    """A single run (one tool invocation)."""

    model_config = {"populate_by_name": True}

    tool: SarifTool
    results: list[SarifResult] = Field(default_factory=list)
    column_kind: Literal["utf16CodeUnits", "unicodeCodePoints"] | None = Field(
        default=None, alias="columnKind"
    )
    artifacts: list[SarifArtifact] | None = None
    invocations: list[SarifInvocation] | None = None
    # SARIF 2.1.0 section 3.14.14: resolves artifactLocation.uriBaseId
    # values used by this run's results.
    original_uri_base_ids: dict[str, SarifArtifactLocation] | None = Field(
        default=None, alias="originalUriBaseIds"
    )


class SarifLog(BaseModel):
    """Top-level SARIF log (SARIF 2.1.0)."""

    model_config = {"populate_by_name": True}

    version: Literal["2.1.0"] = "2.1.0"
    schema_: str | None = Field(default=None, alias="$schema")
    runs: list[SarifRun]

    @model_validator(mode="after")
    def runs_non_empty(self) -> SarifLog:
        if len(self.runs) == 0:
            raise ValueError("runs must be a non-empty array")
        return self


SarifReport = SarifLog


def validate_sarif_report(data: object) -> None:
    """Validate that data has the minimal SARIF 2.1.0 structure. Raises ValidationError if invalid."""
    SarifLog.model_validate(data)
