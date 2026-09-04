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

"""Tests for transitive source extraction and traversal planning."""

import subprocess
import time
from pathlib import Path

import httpx
import pytest

from skillspector import input_handler as input_handler_module
from skillspector import transitive
from skillspector.input_handler import InputHandler, TransitiveIngestTruncatedError


def _limitation(
    result: transitive.ExternalReferenceExtractionResult | transitive.TransitiveTargetPlan,
    resource: str,
) -> transitive.TransitiveResourceLimitation:
    return next(item for item in result.limitations if item.resource == resource)


def test_plan_blocks_circular_reference() -> None:
    """Visited identities block repeated canonical targets before second resolution."""
    refs = [
        "https://github.com/org/dup.git",
        "git@github.com:org/dup.git",
        "https://github.com/org/dup",
    ]
    visited: set[str] = set()
    first = transitive.plan_transitive_targets(
        refs, visited=visited, current_depth=1, max_depth=3, allow_prefixes=(), deny_prefixes=()
    )
    second = transitive.plan_transitive_targets(
        refs, visited=visited, current_depth=1, max_depth=3, allow_prefixes=(), deny_prefixes=()
    )

    assert first == ["https://github.com/org/dup"]
    assert second == []
    assert visited == {"https://github.com/org/dup"}


def test_extract_excludes_badges_docs_and_issue_urls() -> None:
    """Non-scan URLs should be filtered out, even when they look URL-like."""
    file_cache = {
        "SKILL.md": (
            "badge https://img.shields.io/github/stars/user/repo?style=flat-square, "
            "issue https://github.com/NVIDIA/SkillSpector/issues/12, "
            "insecure http://github.com/NVIDIA/SkillSpector, "
            "docs https://github.com/NVIDIA/SkillSpector/wiki, "
            "ci https://github.com/NVIDIA/SkillSpector/actions, "
            "src https://raw.githubusercontent.com/NVIDIA/SkillSpector/main/tool.py, "
            "zip https://huggingface.co/abc/archive/main.zip"
        ),
    }

    refs = transitive.extract_external_refs(file_cache)
    assert refs == [
        "https://raw.githubusercontent.com/NVIDIA/SkillSpector/main/tool.py",
        "https://huggingface.co/abc/archive/main.zip",
    ]


def test_extract_keeps_repos_with_reserved_word_names() -> None:
    """Reserved UI words in org or repo names should not block valid repository targets."""
    file_cache = {
        "SKILL.md": (
            "https://github.com/wiki-tools/skill.git "
            "https://github.com/org/actions.git "
            "https://github.com/badger/skill.git "
            "https://github.com/mrdoob/three.js"
        ),
    }

    refs = transitive.extract_external_refs(file_cache)
    assert refs == [
        "https://github.com/wiki-tools/skill",
        "https://github.com/org/actions",
        "https://github.com/badger/skill",
        "https://github.com/mrdoob/three.js",
    ]


def test_extract_metadata_sorts_hidden_and_nested_cache_paths_deterministically() -> None:
    """Local-only hidden and nested cache entries are handled in stable path order."""
    entries = [
        ("visible.md", "https://github.com/org/visible.git"),
        ("bundle.zip!/nested/SKILL.md", "https://github.com/org/nested.git"),
        (".hidden/references.md", "https://github.com/org/hidden.git"),
    ]

    forward = transitive.extract_external_refs_with_metadata(dict(entries))
    reverse = transitive.extract_external_refs_with_metadata(dict(reversed(entries)))

    expected = [
        "https://github.com/org/hidden",
        "https://github.com/org/nested",
        "https://github.com/org/visible",
    ]
    assert forward.references == reverse.references == expected
    assert [record.source_scope for record in forward.records] == [
        transitive.report_safe_source_scope_key(".hidden/references.md"),
        transitive.report_safe_source_scope_key("bundle.zip!/nested/SKILL.md"),
        transitive.report_safe_source_scope_key("visible.md"),
    ]
    assert all(
        scope.startswith("transitive-reference-source/") and ".." not in scope and ":" not in scope
        for scope in (record.source_scope for record in forward.records)
    )
    assert forward.complete is True

    source_limited = transitive.extract_external_refs_with_metadata(
        dict(entries), limits=transitive.ExternalReferenceLimits(max_sources=2)
    )
    assert source_limited.references == expected[:2]
    assert source_limited.sources_observed == 3
    assert _limitation(source_limited, "sources").limit == 2


def test_extract_metadata_enforces_shared_source_byte_limit() -> None:
    kept = "https://github.com/org/kept.git "
    content = kept + "https://github.com/org/beyond.git"
    result = transitive.extract_external_refs_with_metadata(
        {"SKILL.md": content},
        limits=transitive.ExternalReferenceLimits(max_source_bytes=len(kept.encode("utf-8"))),
    )

    assert result.references == ["https://github.com/org/kept"]
    assert result.source_bytes_examined == result.source_bytes_limit
    assert result.source_bytes_observed == result.source_bytes_limit + 1
    limitation = _limitation(result, "source_bytes")
    assert limitation.observed == limitation.limit + 1
    assert limitation.source_scope == transitive.report_safe_source_scope_key("SKILL.md")
    assert result.complete is False


def test_extract_metadata_stops_at_raw_candidate_limit() -> None:
    content = " ".join(f"http://github.com/org/repo-{index}" for index in range(100))
    result = transitive.extract_external_refs_with_metadata(
        {"dense.md": content},
        limits=transitive.ExternalReferenceLimits(max_raw_candidates=3),
    )

    assert result.references == []
    assert result.raw_candidates_observed == 4
    assert _limitation(result, "raw_candidates").limit == 3


def test_extract_metadata_separates_accepted_and_output_record_limits() -> None:
    unique = " ".join(f"https://github.com/org/repo-{index}.git" for index in range(10))
    accepted_result = transitive.extract_external_refs_with_metadata(
        {"unique.md": unique},
        limits=transitive.ExternalReferenceLimits(
            max_accepted_references=2,
            max_output_records=20,
        ),
    )

    assert accepted_result.references == [
        "https://github.com/org/repo-0",
        "https://github.com/org/repo-1",
    ]
    assert accepted_result.accepted_references_observed == 3
    assert _limitation(accepted_result, "accepted_references").limit == 2

    duplicate = " ".join(["https://github.com/org/shared.git"] * 10)
    output_result = transitive.extract_external_refs_with_metadata(
        {"duplicates.md": duplicate},
        limits=transitive.ExternalReferenceLimits(
            max_accepted_references=10,
            max_output_records=2,
        ),
    )

    assert output_result.references == ["https://github.com/org/shared"]
    assert len(output_result.records) == 2
    assert output_result.accepted_references_observed == 1
    assert output_result.output_records_observed == 3
    assert _limitation(output_result, "output_records").limit == 2


def test_extract_dense_input_stops_without_materializing_all_matches() -> None:
    """A realistic dense file stops near the candidate cap, well before its tail."""
    content = " ".join(
        f"https://github.com/dense/repository-{index}.git" for index in range(20_000)
    )
    started_at = time.monotonic()
    result = transitive.extract_external_refs_with_metadata(
        {"dense.md": content},
        limits=transitive.ExternalReferenceLimits(
            max_raw_candidates=64,
            max_accepted_references=128,
            max_output_records=128,
        ),
    )
    elapsed = time.monotonic() - started_at

    assert result.raw_candidates_observed == 65
    assert len(result.references) == 64
    assert _limitation(result, "raw_candidates").limit == 64
    assert elapsed < 1.0


def test_extract_honors_caller_absolute_deadline() -> None:
    class StepClock:
        def __init__(self) -> None:
            self.value = 10.0

        def __call__(self) -> float:
            self.value += 0.01
            return self.value

    clock = StepClock()
    dense = " ".join(["https://github.com/org/repo.git"] * 1000)
    result = transitive.extract_external_refs_with_metadata(
        {"SKILL.md": dense},
        clock=clock,
        deadline=10.08,
    )

    assert result.complete is False
    assert result.raw_candidates_observed < 1000
    assert _limitation(result, "runtime").limit <= 0.071


def test_extract_does_not_accept_prefix_of_overlong_token() -> None:
    overlong = "https://github.com/org/" + (
        "a" * (transitive.MAX_EXTERNAL_REFERENCE_TOKEN_CHARACTERS + 1)
    )
    result = transitive.extract_external_refs_with_metadata(
        {"SKILL.md": f"{overlong} https://github.com/org/valid.git"}
    )

    assert result.references == ["https://github.com/org/valid"]
    assert result.raw_candidates_observed == 1


def test_input_handler_treats_github_archive_zip_as_file_url() -> None:
    """GitHub archive ZIP links should download as files, not route through git clone."""
    handler = InputHandler()
    url = "https://github.com/org/repo/archive/refs/heads/main.zip"

    assert handler._is_git_url(url) is False
    assert handler._is_file_url(url) is True


def test_input_handler_keeps_extension_named_github_repositories_as_git() -> None:
    handler = InputHandler()

    assert handler._is_git_url("https://github.com/mrdoob/three.js") is True
    assert handler._is_git_url("https://github.com/robfig/cron.go") is True
    assert handler._is_git_url("https://github.com/org/skills.zip") is True


def test_input_handler_resolves_github_archive_zip_via_validated_redirect(
    tmp_path: Path, monkeypatch
) -> None:
    """GitHub archive ZIP redirects should still resolve as downloadable archives."""

    class FakeResponse:
        def __init__(
            self,
            status_code: int,
            *,
            headers: dict[str, str] | None = None,
            content: bytes = b"",
        ) -> None:
            self.status_code = status_code
            self.headers = headers or {}
            self.content = content

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                request = httpx.Request("GET", "https://example.invalid")
                response = httpx.Response(
                    self.status_code,
                    headers=self.headers,
                    content=self.content,
                    request=request,
                )
                raise httpx.HTTPStatusError(
                    f"HTTP error {self.status_code}", request=request, response=response
                )

        def iter_bytes(self, chunk_size: int | None = None):
            assert chunk_size is not None
            yield self.content

    class FakeClient:
        def __init__(self, responses: list[FakeResponse], **kwargs) -> None:
            self._responses = responses

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def stream(self, method: str, url: str):
            response = self._responses.pop(0)

            class _StreamContext:
                def __enter__(self) -> FakeResponse:
                    return response

                def __exit__(self, exc_type, exc, tb) -> bool:
                    return False

            return _StreamContext()

    class Budget:
        def remaining_seconds(self) -> float:
            return 60.0

        def remaining_bytes(self) -> int:
            return 1024 * 1024

        def record_bytes(self, bytes_scanned: int) -> None:
            return None

        def note_truncation(self, reason: str) -> None:
            return None

    archive_url = "https://github.com/org/repo/archive/refs/heads/main.zip"
    redirected_url = "https://codeload.github.com/org/repo/zip/refs/heads/main"
    responses = [
        FakeResponse(302, headers={"location": redirected_url}),
        FakeResponse(200, headers={"content-type": "application/zip"}, content=b"zip-bytes"),
    ]
    handler = InputHandler(transitive_budget=Budget())

    monkeypatch.setattr(input_handler_module, "_is_private_ip", lambda host: False)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: FakeClient(responses, **kwargs))
    monkeypatch.setattr(handler, "_extract_zip", lambda zip_path: tmp_path / Path(zip_path).stem)

    resolved_path, source_type = handler.resolve(archive_url)

    assert source_type == "url"
    assert resolved_path == tmp_path / "download"
    assert responses == []


def test_input_handler_download_budget_exhaustion_is_typed(tmp_path: Path, monkeypatch) -> None:
    """A streamed transitive download over budget cannot look like a clean empty scan."""

    class Budget:
        def __init__(self) -> None:
            self.reasons: list[str] = []

        def remaining_seconds(self) -> float:
            return 60.0

        def remaining_bytes(self) -> int:
            return 4

        def note_truncation(self, reason: str) -> None:
            self.reasons.append(reason)

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/plain"}

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, chunk_size: int | None = None):
            assert chunk_size is not None
            yield b"12345"

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def stream(self, method: str, url: str):
            class _StreamContext:
                def __enter__(self) -> FakeResponse:
                    return FakeResponse()

                def __exit__(self, exc_type, exc, tb) -> bool:
                    return False

            return _StreamContext()

    budget = Budget()
    handler = InputHandler(transitive_budget=budget)
    monkeypatch.setattr(input_handler_module, "_is_private_ip", lambda host: False)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: FakeClient(**kwargs))
    monkeypatch.setattr(handler, "_get_temp_dir", lambda: tmp_path)

    with pytest.raises(TransitiveIngestTruncatedError) as raised:
        handler._download_file("https://raw.githubusercontent.com/org/repo/main/SKILL.md")

    assert raised.value.truncation.code == "byte_budget_exhausted"
    assert raised.value.truncation.source_type == "download"
    assert not (tmp_path / "download").exists()
    assert budget.reasons == [raised.value.truncation.message]


def test_input_handler_download_deadline_exhaustion_is_typed(tmp_path: Path, monkeypatch) -> None:
    """An exhausted transitive deadline produces a typed incomplete-input signal."""

    class Budget:
        def __init__(self) -> None:
            self.reasons: list[str] = []
            self.calls = 0

        def remaining_seconds(self) -> float:
            self.calls += 1
            return 60.0 if self.calls == 1 else 0.0

        def remaining_bytes(self) -> int:
            return 10

        def note_truncation(self, reason: str) -> None:
            self.reasons.append(reason)

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/plain"}

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, chunk_size: int | None = None):
            assert chunk_size is not None
            yield b"1"

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def stream(self, method: str, url: str):
            class _StreamContext:
                def __enter__(self) -> FakeResponse:
                    return FakeResponse()

                def __exit__(self, exc_type, exc, tb) -> bool:
                    return False

            return _StreamContext()

    budget = Budget()
    handler = InputHandler(transitive_budget=budget)
    monkeypatch.setattr(input_handler_module, "_is_private_ip", lambda host: False)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: FakeClient(**kwargs))
    monkeypatch.setattr(handler, "_get_temp_dir", lambda: tmp_path)

    with pytest.raises(TransitiveIngestTruncatedError) as raised:
        handler._download_file("https://raw.githubusercontent.com/org/repo/main/SKILL.md")

    assert raised.value.truncation.code == "time_budget_exhausted"
    assert raised.value.truncation.source_type == "download"
    assert not (tmp_path / "download").exists()
    assert budget.reasons == [raised.value.truncation.message]


def test_input_handler_rejects_oversized_transitive_git_clone(tmp_path: Path, monkeypatch) -> None:
    """A transitive clone over budget is typed and its partial tree is removed."""

    class Budget:
        def __init__(self) -> None:
            self.reasons: list[str] = []

        def remaining_seconds(self) -> float:
            return 60.0

        def remaining_bytes(self) -> int:
            return 5

        def note_truncation(self, reason: str) -> None:
            self.reasons.append(reason)

    budget = Budget()
    handler = InputHandler(transitive_budget=budget)
    commands: list[list[str]] = []
    monkeypatch.setattr(handler, "_get_temp_dir", lambda: tmp_path)
    monkeypatch.setattr(handler, "_validate_url_host", lambda url, allowed: "github.com")

    class CompletedProcess:
        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    def fake_popen(cmd, **kwargs):
        commands.append(cmd)
        clone_dir = Path(cmd[-1])
        clone_dir.mkdir(parents=True)
        (clone_dir / "large.py").write_text("0123456789", encoding="utf-8")
        return CompletedProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    with pytest.raises(TransitiveIngestTruncatedError) as raised:
        handler._clone_git("https://github.com/org/large.git")

    assert commands == [
        [
            "git",
            "-c",
            "core.symlinks=false",
            "clone",
            "--depth",
            "1",
            "--filter=blob:limit=5",
            "https://github.com/org/large.git",
            str(tmp_path / "repo"),
        ]
    ]
    assert raised.value.truncation.code == "byte_budget_exhausted"
    assert raised.value.truncation.source_type == "git"
    assert not (tmp_path / "repo").exists()
    assert budget.reasons == [raised.value.truncation.message]


def test_plan_depth_limit_prevents_next_wave() -> None:
    """When current depth exceeds max depth, no targets are returned."""
    refs = ["https://github.com/org/repo.git"]
    visited: set[str] = set()
    result = transitive.plan_transitive_targets(
        refs=refs,
        visited=visited,
        current_depth=4,
        max_depth=3,
        allow_prefixes=(),
        deny_prefixes=(),
    )

    assert result == []
    assert visited == set()


def test_plan_applies_allow_prefix() -> None:
    """Only identities matching allow prefixes are returned."""
    refs = [
        "https://github.com/ok/repo.git",
        "https://github.com/skip/repo.git",
    ]
    visited: set[str] = set()
    allowed = ("https://github.com/ok/",)

    result = transitive.plan_transitive_targets(
        refs=refs,
        visited=visited,
        current_depth=1,
        max_depth=2,
        allow_prefixes=allowed,
        deny_prefixes=(),
    )

    assert result == ["https://github.com/ok/repo"]


def test_plan_allow_prefix_respects_path_boundaries() -> None:
    """Allow prefixes should not match sibling org names sharing a string prefix."""
    refs = [
        "https://github.com/trusted/repo.git",
        "https://github.com/trusted-malicious/repo.git",
    ]
    visited: set[str] = set()

    result = transitive.plan_transitive_targets(
        refs=refs,
        visited=visited,
        current_depth=1,
        max_depth=2,
        allow_prefixes=("https://github.com/trusted/",),
        deny_prefixes=(),
    )

    assert result == ["https://github.com/trusted/repo"]


def test_plan_allow_prefix_normalizes_dot_segment_escapes() -> None:
    """Allow-prefix checks should run on normalized paths, not raw URL text."""
    refs = ["https://github.com/trusted/%2e%2e/evil/repo.git"]

    result = transitive.plan_transitive_targets(
        refs=refs,
        visited=set(),
        current_depth=1,
        max_depth=2,
        allow_prefixes=("https://github.com/trusted/",),
        deny_prefixes=(),
    )

    assert result == []


def test_plan_applies_deny_prefix() -> None:
    """Deny prefixes skip matching identities even if they are otherwise valid."""
    refs = [
        "https://github.com/ok/repo.git",
        "https://github.com/skip/repo.git",
    ]
    visited: set[str] = set()
    denied = ("https://github.com/skip/",)

    result = transitive.plan_transitive_targets(
        refs=refs,
        visited=visited,
        current_depth=1,
        max_depth=2,
        allow_prefixes=(),
        deny_prefixes=denied,
    )

    assert result == ["https://github.com/ok/repo"]


def test_plan_deny_prefix_respects_path_boundaries() -> None:
    """Deny prefixes should not block sibling org names that only share a string prefix."""
    refs = [
        "https://github.com/trusted/repo.git",
        "https://github.com/trusted-malicious/repo.git",
    ]
    visited: set[str] = set()

    result = transitive.plan_transitive_targets(
        refs=refs,
        visited=visited,
        current_depth=1,
        max_depth=2,
        allow_prefixes=(),
        deny_prefixes=("https://github.com/trusted/",),
    )

    assert result == ["https://github.com/trusted-malicious/repo"]


def test_plan_deny_prefix_blocks_normalized_dot_segment_escapes() -> None:
    """Deny-prefix checks should block refs that normalize into the denied path."""
    refs = ["https://github.com/trusted/%2e%2e/evil/repo.git"]

    result = transitive.plan_transitive_targets(
        refs=refs,
        visited=set(),
        current_depth=1,
        max_depth=2,
        allow_prefixes=(),
        deny_prefixes=("https://github.com/evil/",),
    )

    assert result == []


def test_plan_metadata_bounds_input_and_target_records() -> None:
    refs = [f"https://github.com/org/repo-{index}.git" for index in range(10)]
    input_limited = transitive.plan_transitive_targets_with_metadata(
        refs=refs,
        visited=set(),
        current_depth=1,
        max_depth=3,
        allow_prefixes=(),
        deny_prefixes=(),
        limits=transitive.TransitivePlanLimits(max_input_references=3, max_targets=10),
    )

    assert input_limited.targets == [
        "https://github.com/org/repo-0",
        "https://github.com/org/repo-1",
        "https://github.com/org/repo-2",
    ]
    assert input_limited.input_references_observed == 4
    assert _limitation(input_limited, "input_references").limit == 3

    visited: set[str] = set()
    output_limited = transitive.plan_transitive_targets_with_metadata(
        refs=refs,
        visited=visited,
        current_depth=1,
        max_depth=3,
        allow_prefixes=(),
        deny_prefixes=(),
        limits=transitive.TransitivePlanLimits(max_input_references=10, max_targets=2),
    )

    assert output_limited.targets == [
        "https://github.com/org/repo-0",
        "https://github.com/org/repo-1",
    ]
    assert output_limited.targets_observed == 3
    assert _limitation(output_limited, "output_records").limit == 2
    assert visited == set(output_limited.targets)


def test_plan_metadata_fails_closed_when_prefix_set_is_incomplete() -> None:
    result = transitive.plan_transitive_targets_with_metadata(
        refs=["https://github.com/blocked/repo.git"],
        visited=set(),
        current_depth=1,
        max_depth=2,
        allow_prefixes=(),
        deny_prefixes=(
            "https://github.com/first/",
            "https://github.com/blocked/",
        ),
        limits=transitive.TransitivePlanLimits(max_prefixes=1),
    )

    assert result.targets == []
    assert result.input_references_observed == 0
    assert _limitation(result, "prefixes").observed == 2


def test_plan_metadata_honors_absolute_deadline_without_mutating_visited() -> None:
    visited: set[str] = set()
    result = transitive.plan_transitive_targets_with_metadata(
        refs=["https://github.com/org/repo.git"],
        visited=visited,
        current_depth=1,
        max_depth=2,
        allow_prefixes=(),
        deny_prefixes=(),
        clock=lambda: 5.0,
        deadline=4.0,
    )

    assert result.targets == []
    assert visited == set()
    assert _limitation(result, "runtime").limit == 0.0


def test_bounded_frontier_caps_retained_waves_and_references() -> None:
    frontier = transitive.BoundedTransitiveFrontier(
        deadline=100.0,
        clock=lambda: 1.0,
        max_waves=2,
        max_references=3,
    )

    assert frontier.append(1, ["a", "b"]) is True
    assert frontier.append(2, ["c", "d", "e"]) is True
    assert frontier.append(3, ["f"]) is False
    assert len(frontier) == 2
    assert frontier.references_observed == 4
    assert {item.resource for item in frontier.limitations} == {
        "frontier_references",
        "frontier_waves",
    }

    first = frontier.popleft()
    second = frontier.popleft()
    assert first == transitive.TransitiveFrontierWave(depth=1, references=("a", "b"))
    assert second == transitive.TransitiveFrontierWave(depth=2, references=("c",))
