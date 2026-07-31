# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for batch-scan report serialization."""

from __future__ import annotations

import json

from contrib.batch_scan.reports import _format_json


def test_json_marks_error_entries_as_unsuccessful() -> None:
    entry = {
        "skill": {"name": "crashed-skill", "language": "en"},
        "risk_assessment": {"score": 0, "severity": "ERROR", "recommendation": "ERROR"},
        "components": [],
        "issues": [],
        "error": "scan crashed",
    }

    payload = json.loads(_format_json([entry]))

    assert payload["skills"][0]["error"] == "scan crashed"
    assert payload["skills"][0]["execution_successful"] is False
