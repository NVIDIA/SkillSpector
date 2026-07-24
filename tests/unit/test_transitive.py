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
from pathlib import Path

import httpx

from skillspector import input_handler as input_handler_module
from skillspector import transitive
from skillspector.input_handler import InputHandler


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
            "https://github.com/badger/skill.git"
        ),
    }

    refs = transitive.extract_external_refs(file_cache)
    assert refs == [
        "https://github.com/wiki-tools/skill",
        "https://github.com/org/actions",
        "https://github.com/badger/skill",
    ]


def test_input_handler_treats_github_archive_zip_as_file_url() -> None:
    """GitHub archive ZIP links should download as files, not route through git clone."""
    handler = InputHandler()
    url = "https://github.com/org/repo/archive/refs/heads/main.zip"

    assert handler._is_git_url(url) is False
    assert handler._is_file_url(url) is True


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

        def iter_bytes(self):
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

    archive_url = "https://github.com/org/repo/archive/refs/heads/main.zip"
    redirected_url = "https://codeload.github.com/org/repo/zip/refs/heads/main"
    responses = [
        FakeResponse(302, headers={"location": redirected_url}),
        FakeResponse(200, headers={"content-type": "application/zip"}, content=b"zip-bytes"),
    ]
    handler = InputHandler()

    monkeypatch.setattr(input_handler_module, "_is_private_ip", lambda host: False)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: FakeClient(responses, **kwargs))
    monkeypatch.setattr(handler, "_extract_zip", lambda zip_path: tmp_path / Path(zip_path).stem)

    resolved_path, source_type = handler.resolve(archive_url)

    assert source_type == "url"
    assert resolved_path == tmp_path / "download"
    assert responses == []


def test_input_handler_download_budget_exhaustion_returns_empty_dir(
    tmp_path: Path, monkeypatch
) -> None:
    """A streamed transitive download over budget truncates instead of failing the child scan."""

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

        def iter_bytes(self):
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

    resolved = handler._download_file("https://raw.githubusercontent.com/org/repo/main/SKILL.md")

    assert resolved == tmp_path / "download"
    assert list(resolved.iterdir()) == []
    assert budget.reasons == ["Transitive byte budget exceeded by downloaded file"]


def test_input_handler_download_deadline_exhaustion_returns_empty_dir(
    tmp_path: Path, monkeypatch
) -> None:
    """A streamed transitive download that runs out of time truncates instead of hard-failing."""

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

        def iter_bytes(self):
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

    resolved = handler._download_file("https://raw.githubusercontent.com/org/repo/main/SKILL.md")

    assert resolved == tmp_path / "download"
    assert list(resolved.iterdir()) == []
    assert budget.reasons == ["Transitive time budget exhausted during download"]


def test_input_handler_rejects_oversized_transitive_git_clone(tmp_path: Path, monkeypatch) -> None:
    """A transitive git clone over the remaining byte budget returns an empty scan root."""

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

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        clone_dir = Path(cmd[-1])
        clone_dir.mkdir(parents=True)
        (clone_dir / "large.py").write_text("0123456789", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    resolved = handler._clone_git("https://github.com/org/large.git")

    assert commands == [
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:limit=5",
            "https://github.com/org/large.git",
            str(tmp_path / "repo"),
        ]
    ]
    assert resolved == tmp_path / "repo"
    assert list(resolved.iterdir()) == []
    assert budget.reasons == ["Transitive byte budget exceeded by cloned repository"]


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
