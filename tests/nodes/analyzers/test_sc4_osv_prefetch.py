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

"""SC4 packs every manifest's OSV lookups into shared batches (#1174)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from skillspector.inspection_ledger import LedgerReason
from skillspector.nodes.analyzers import osv_client
from skillspector.nodes.analyzers import static_patterns_supply_chain as supply_chain

# Synthetic packages: no real advisory data and no live OSV lookup.
FILES = {
    "package-lock.json": json.dumps(
        {
            "lockfileVersion": 3,
            "packages": {
                "": {"name": "fixture", "version": "1.0.0"},
                "node_modules/synthetic-a": {"version": "1.0.0"},
                "node_modules/synthetic-b": {"version": "1.0.0"},
                "node_modules/synthetic-c": {"version": "1.0.0"},
            },
        }
    ),
    "requirements.txt": "synthetic-py==1.0.0\n",
}


@pytest.fixture(autouse=True)
def _one_shared_batch(monkeypatch: pytest.MonkeyPatch):
    osv_client.clear_cache()
    monkeypatch.setattr(osv_client, "MAX_OSV_QUERIES_PER_BATCH", 4)
    monkeypatch.setattr(osv_client, "MAX_OSV_QUERY_BATCHES", 1)
    monkeypatch.setattr(
        supply_chain.static_runner,
        "run_static_patterns_with_ledger",
        lambda *_args: {"findings": [], "inspection_ledger": [], "analyzer_status_events": []},
    )
    yield
    osv_client.clear_cache()


def _osv_client(results: int) -> MagicMock:
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    response = MagicMock()
    response.json.return_value = {"results": [{"vulns": []}] * results}
    response.raise_for_status = MagicMock()
    client.post.side_effect = [response]
    return client


def _paths_with(response, reason: LedgerReason) -> set[str]:
    return {
        event["path"]
        for event in response["inspection_ledger"]
        if event.get("reason_code") in (reason, reason.value)
    }


def test_later_manifest_shares_the_lockfile_batch() -> None:
    client = _osv_client(results=4)

    with patch("skillspector.nodes.analyzers.osv_client.httpx.Client", return_value=client):
        response = supply_chain.node({"components": list(FILES), "file_cache": dict(FILES)})

    assert client.post.call_count == 1
    sent = client.post.call_args.kwargs["json"]["queries"]
    assert sorted(q["package"]["name"] for q in sent) == [
        "synthetic-a",
        "synthetic-b",
        "synthetic-c",
        "synthetic-py",
    ]
    assert "requirements.txt" not in _paths_with(response, LedgerReason.OUTPUT_LIMIT)


def test_replaced_query_batch_skips_the_prefetch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def adapter(packages, ecosystem, **_kwargs):
        calls.append(ecosystem)
        return [[] for _ in packages]

    monkeypatch.setattr(supply_chain, "query_batch", adapter)
    prefetch = MagicMock()
    monkeypatch.setattr(osv_client, "prefetch", prefetch)

    supply_chain.node({"components": list(FILES), "file_cache": dict(FILES)})

    prefetch.assert_not_called()
    assert set(calls) == {osv_client.ECOSYSTEM_NPM, osv_client.ECOSYSTEM_PYPI}


def test_runtime_limit_keeps_findings_the_prefetch_already_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deadline passes after the first batch: the lockfile keeps its finding,
    and requirements.txt reports the omission."""
    monkeypatch.setattr(osv_client, "MAX_OSV_QUERIES_PER_BATCH", 3)
    monkeypatch.setattr(osv_client, "MAX_OSV_QUERY_BATCHES", 2)
    clock = [1000.0]
    monkeypatch.setattr(osv_client.time, "monotonic", lambda: clock[0])
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    batch = MagicMock()
    batch.raise_for_status = MagicMock()
    batch.json.return_value = {
        "results": [{"vulns": [{"id": "OSV-1"}]}, {"vulns": []}, {"vulns": []}]
    }
    client.post.side_effect = [batch]
    detail = MagicMock()
    detail.raise_for_status = MagicMock()
    detail.json.return_value = {"id": "OSV-1", "summary": "synthetic advisory"}

    def get_detail(*_args, **_kwargs):
        clock[0] += 1_000_000.0  # the scan's runtime runs out after this request
        return detail

    client.get.side_effect = get_detail

    with patch("skillspector.nodes.analyzers.osv_client.httpx.Client", return_value=client):
        response = supply_chain.node({"components": list(FILES), "file_cache": dict(FILES)})

    assert client.post.call_count == 1
    assert [
        (finding.file, finding.rule_id)
        for finding in response["findings"]
        if "OSV-1" in str(finding)
    ] == [("package-lock.json", "SC4")]
    runtime_paths = _paths_with(response, LedgerReason.RUNTIME_LIMIT)
    assert "requirements.txt" in runtime_paths
    assert "package-lock.json" not in runtime_paths
