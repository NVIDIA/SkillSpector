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

"""Tests for the OSV.dev API client (osv_client.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from skillspector.nodes.analyzers.osv_client import (
    ECOSYSTEM_NPM,
    ECOSYSTEM_PYPI,
    VulnResult,
    _cache,
    _estimate_cvss_severity,
    _severity_from_vuln,
    clear_cache,
    query_batch,
    was_osv_reachable,
)


@pytest.fixture(autouse=True)
def _clear_osv_cache():
    """Ensure cache is empty before and after each test."""
    clear_cache()
    yield
    clear_cache()


class TestEstimateCvssSeverity:
    """Tests for CVSS v3 and v4 vector string parsing."""

    def test_v3_all_high_metrics_is_critical(self) -> None:
        vector = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"
        assert _estimate_cvss_severity(vector) == "CRITICAL"

    def test_v3_mostly_high_is_critical(self) -> None:
        vector = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
        assert _estimate_cvss_severity(vector) == "CRITICAL"

    def test_v3_mixed_high_is_high(self) -> None:
        vector = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:L/A:N"
        assert _estimate_cvss_severity(vector) == "HIGH"

    def test_v3_few_high_is_medium(self) -> None:
        vector = "CVSS:3.1/AV:N/AC:H/PR:N/UI:R/S:U/C:L/I:L/A:N"
        assert _estimate_cvss_severity(vector) == "MEDIUM"

    def test_v3_no_high_metrics_is_low(self) -> None:
        vector = "CVSS:3.1/AV:P/AC:H/PR:H/UI:R/S:U/C:L/I:N/A:N"
        assert _estimate_cvss_severity(vector) == "LOW"

    def test_v4_all_high_is_critical(self) -> None:
        vector = "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:H/SI:H/SA:H"
        assert _estimate_cvss_severity(vector) == "CRITICAL"

    def test_v4_low_severity(self) -> None:
        vector = "CVSS:4.0/AV:L/AC:H/AT:P/PR:H/UI:A/VC:L/VI:N/VA:N/SC:N/SI:N/SA:N"
        assert _estimate_cvss_severity(vector) == "LOW"

    def test_v4_mixed_is_high(self) -> None:
        vector = "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:N/VI:N/VA:H/SC:N/SI:N/SA:N"
        assert _estimate_cvss_severity(vector) == "HIGH"

    def test_invalid_vector_returns_none(self) -> None:
        assert _estimate_cvss_severity("not-a-vector") is None

    def test_empty_string_returns_none(self) -> None:
        assert _estimate_cvss_severity("") is None

    def test_bare_numeric_score_returns_none(self) -> None:
        assert _estimate_cvss_severity("7.5") is None


class TestSeverityFromVuln:
    """Tests for the full severity extraction pipeline."""

    def test_database_specific_is_primary(self) -> None:
        vuln = {
            "database_specific": {"severity": "CRITICAL"},
            "affected": [{"ecosystem_specific": {"severity": "LOW"}}],
            "severity": [{"score": "CVSS:3.1/AV:P/AC:H/PR:H/UI:R/S:U/C:N/I:N/A:N"}],
        }
        assert _severity_from_vuln(vuln) == "CRITICAL"

    def test_ecosystem_specific_when_no_db(self) -> None:
        vuln = {"affected": [{"ecosystem_specific": {"severity": "MEDIUM"}}]}
        assert _severity_from_vuln(vuln) == "MEDIUM"

    def test_cvss_vector_when_no_other_sources(self) -> None:
        vuln = {"severity": [{"score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"}]}
        assert _severity_from_vuln(vuln) == "CRITICAL"

    def test_cvss_vector_low_severity(self) -> None:
        vuln = {"severity": [{"score": "CVSS:3.1/AV:P/AC:H/PR:H/UI:R/S:U/C:L/I:N/A:N"}]}
        assert _severity_from_vuln(vuln) == "LOW"

    def test_database_specific_case_insensitive(self) -> None:
        vuln = {"database_specific": {"severity": "high"}}
        assert _severity_from_vuln(vuln) == "HIGH"

    def test_no_severity_defaults_high(self) -> None:
        assert _severity_from_vuln({}) == "HIGH"


class TestQueryBatch:
    def test_empty_packages_returns_empty(self) -> None:
        assert query_batch([], ECOSYSTEM_PYPI) == []

    def test_successful_batch_query(self) -> None:
        mock_batch_response = {
            "results": [
                {"vulns": [{"id": "GHSA-462w", "modified": "2024-01-01T00:00:00Z"}]},
                {"vulns": []},
            ]
        }
        mock_detail_response = {
            "id": "GHSA-462w",
            "summary": "XSS in Jinja2",
            "severity": [
                {"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N"}
            ],
            "aliases": ["CVE-2024-22195"],
        }

        mock_client = MagicMock()
        mock_post_resp = MagicMock()
        mock_post_resp.json.return_value = mock_batch_response
        mock_post_resp.raise_for_status = MagicMock()

        mock_get_resp = MagicMock()
        mock_get_resp.json.return_value = mock_detail_response
        mock_get_resp.raise_for_status = MagicMock()

        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_post_resp
        mock_client.get.return_value = mock_get_resp

        with patch(
            "skillspector.nodes.analyzers.osv_client.httpx.Client", return_value=mock_client
        ):
            results = query_batch(
                [("jinja2", "2.4.1"), ("requests", "2.31.0")],
                ECOSYSTEM_PYPI,
            )

        assert len(results) == 2
        assert len(results[0]) == 1
        assert results[0][0].vuln_id == "GHSA-462w"
        assert results[0][0].severity == "HIGH"
        assert "CVE-2024-22195" in results[0][0].aliases
        assert len(results[1]) == 0

    def test_network_failure_returns_empty(self) -> None:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")

        with patch(
            "skillspector.nodes.analyzers.osv_client.httpx.Client", return_value=mock_client
        ):
            results = query_batch([("jinja2", "2.4.1")], ECOSYSTEM_PYPI)

        assert results == [[]]

    def test_timeout_returns_empty(self) -> None:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.TimeoutException("Timed out")

        with patch(
            "skillspector.nodes.analyzers.osv_client.httpx.Client", return_value=mock_client
        ):
            results = query_batch([("lodash", "4.17.20")], ECOSYSTEM_NPM)

        assert results == [[]]

    def test_cache_hit_avoids_api_call(self) -> None:
        cached_vuln = VulnResult(vuln_id="CACHED-1", summary="cached", severity="HIGH", aliases=())
        mock_batch_response = {
            "results": [
                {"vulns": [{"id": "GHSA-new", "modified": "2024-01-01T00:00:00Z"}]},
            ]
        }
        mock_detail_response = {
            "id": "GHSA-new",
            "summary": "new vuln",
            "severity": [
                {"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}
            ],
            "aliases": [],
        }

        mock_client = MagicMock()
        mock_post_resp = MagicMock()
        mock_post_resp.json.return_value = mock_batch_response
        mock_post_resp.raise_for_status = MagicMock()
        mock_get_resp = MagicMock()
        mock_get_resp.json.return_value = mock_detail_response
        mock_get_resp.raise_for_status = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_post_resp
        mock_client.get.return_value = mock_get_resp

        import time

        _cache[("jinja2", "2.4.1", "PyPI")] = (time.monotonic(), [cached_vuln])

        with patch(
            "skillspector.nodes.analyzers.osv_client.httpx.Client", return_value=mock_client
        ):
            results = query_batch(
                [("jinja2", "2.4.1"), ("requests", "2.31.0")],
                ECOSYSTEM_PYPI,
            )

        assert len(results) == 2
        assert results[0] == [cached_vuln]
        assert len(results[1]) == 1
        assert results[1][0].vuln_id == "GHSA-new"

    def test_npm_ecosystem(self) -> None:
        mock_batch_response = {
            "results": [
                {"vulns": [{"id": "GHSA-npm1", "modified": "2024-01-01T00:00:00Z"}]},
            ]
        }
        mock_detail_response = {
            "id": "GHSA-npm1",
            "summary": "prototype pollution",
            "severity": [
                {"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:H/UI:N/S:U/C:H/I:H/A:H"}
            ],
            "aliases": ["CVE-2021-23337"],
        }

        mock_client = MagicMock()
        mock_post_resp = MagicMock()
        mock_post_resp.json.return_value = mock_batch_response
        mock_post_resp.raise_for_status = MagicMock()
        mock_get_resp = MagicMock()
        mock_get_resp.json.return_value = mock_detail_response
        mock_get_resp.raise_for_status = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_post_resp
        mock_client.get.return_value = mock_get_resp

        with patch(
            "skillspector.nodes.analyzers.osv_client.httpx.Client", return_value=mock_client
        ):
            results = query_batch([("lodash", "4.17.20")], ECOSYSTEM_NPM)

        assert len(results) == 1
        assert results[0][0].vuln_id == "GHSA-npm1"

    def test_was_osv_reachable_after_success(self) -> None:
        """After a successful query, was_osv_reachable() returns True."""
        mock_batch_response = {
            "results": [
                {"vulns": []},
            ]
        }

        mock_client = MagicMock()
        mock_post_resp = MagicMock()
        mock_post_resp.json.return_value = mock_batch_response
        mock_post_resp.raise_for_status = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_post_resp

        with patch(
            "skillspector.nodes.analyzers.osv_client.httpx.Client", return_value=mock_client
        ):
            query_batch([("requests", "2.31.0")], ECOSYSTEM_PYPI)

        assert was_osv_reachable() is True

    def test_was_osv_reachable_after_failure(self) -> None:
        """After a failed query, was_osv_reachable() returns False."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")

        with patch(
            "skillspector.nodes.analyzers.osv_client.httpx.Client", return_value=mock_client
        ):
            query_batch([("jinja2", "2.4.1")], ECOSYSTEM_PYPI)

        assert was_osv_reachable() is False
