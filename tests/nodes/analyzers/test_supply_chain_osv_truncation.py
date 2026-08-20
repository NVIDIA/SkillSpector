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

"""A capped advisory lookup must reach the report, not just the log."""

from __future__ import annotations

from skillspector.inspection_ledger import LedgerReason
from skillspector.nodes.analyzers import static_patterns_supply_chain as supply_chain
from skillspector.nodes.analyzers.osv_client import VulnResult

_TOTAL = 153
_EXAMINED = 10


def _ten_advisories() -> list[VulnResult]:
    return [
        VulnResult(vuln_id=f"GHSA-{i:04d}", summary="", severity="LOW", aliases=())
        for i in range(_EXAMINED)
    ]


def _patch_truncated_osv(monkeypatch) -> None:
    """OSV reports 153 advisories, the client examines 10 — the shape of the real defect."""
    monkeypatch.setattr(supply_chain, "query_batch", lambda pkgs, eco: [_ten_advisories()])
    monkeypatch.setattr(
        supply_chain,
        "last_truncations",
        lambda: [("jinja2", "3.1.2", _EXAMINED, _TOTAL)],
    )


def test_finding_message_states_the_true_advisory_count(monkeypatch) -> None:
    """Reporting "10 advisory(ies)" for a package that has 153 is a false statement.

    It is also the number a reader would use to judge the package, and it understates the risk
    by an order of magnitude while looking precise.
    """
    _patch_truncated_osv(monkeypatch)

    findings = supply_chain._analyze_dependencies("jinja2==3.1.2\n", "requirements.txt", {})

    sc4 = [f for f in findings if f.rule_id == "SC4"]
    assert sc4, "the vulnerable package must still be reported"
    message = sc4[0].message
    assert str(_TOTAL) in message, f"message hides the real advisory count: {message}"
    assert "10 advisory(ies)" not in message


def test_truncation_surfaces_as_a_ledger_exception(monkeypatch) -> None:
    """`analysis_completeness` is where a consumer looks for what was not examined."""
    _patch_truncated_osv(monkeypatch)

    state = {
        "components": ["requirements.txt"],
        "file_cache": {"requirements.txt": "jinja2==3.1.2\n"},
        "manifest": {},
        "skill_path": "",
        "component_metadata": [{"path": "requirements.txt", "executable": False}],
    }
    response = supply_chain.node(state)  # type: ignore[arg-type]

    capped = [
        event
        for event in response["inspection_ledger"]
        if event.get("reason_code") is LedgerReason.RESULT_LIMIT
    ]
    assert capped, "the cap left no trace in the ledger"
    assert capped[0]["observed_count"] == _TOTAL
    assert capped[0]["limit_count"] == _EXAMINED
    assert capped[0]["path"] == "requirements.txt"


def test_result_limit_reaches_analysis_completeness(monkeypatch) -> None:
    """The event must survive the projection into the public report, not just the ledger.

    `_exception_from_event` deliberately keeps the public row narrow — it carries what, where and
    why, and no numeric evidence: real reports show `size_limit` and `binary_content` rows without
    any byte counts. So the assertion here is the one that matters to a consumer: a capped
    advisory lookup shows up in `analysis_completeness.ledger_exceptions` naming the file and the
    analyzer. The quantity travels in the SC4 message instead, which is where a reader looking at
    the package will find it.
    """
    from skillspector.inspection_ledger import LedgerOutcome, finalize_ledger, ledger_event

    completeness, _limitations = finalize_ledger(
        {
            "components": ["requirements.txt"],
            "inspection_ledger": [
                ledger_event(
                    analyzer_id="static_patterns_supply_chain",
                    outcome=LedgerOutcome.SKIPPED,
                    phase="static",
                    path="requirements.txt",
                    reason=LedgerReason.RESULT_LIMIT,
                    observed_count=_TOTAL,
                    limit_count=_EXAMINED,
                )
            ],
        }
    )

    capped = [
        row
        for row in completeness["ledger_exceptions"]
        if row["reason_code"] is LedgerReason.RESULT_LIMIT
    ]
    assert capped, "the capped lookup never reaches the public report"
    assert capped[0]["path"] == "requirements.txt"
    assert "static_patterns_supply_chain" in capped[0]["analyzers"]
    assert capped[0]["message"]
