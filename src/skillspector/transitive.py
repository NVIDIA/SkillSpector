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

"""Shared helpers for opt-in transitive external-source traversal."""

from __future__ import annotations

import re
from urllib.parse import unquote, urlparse, urlunparse

from skillspector.input_handler import ALLOWED_DOWNLOAD_HOSTS, ALLOWED_GIT_HOSTS

_URL_PATTERN = re.compile(r"(?:https?://|git@)[^\s\]{}'\"<>`!?,;.)}]+")
_LEADING_PUNCTUATION = "([{\"'<"
_TRAILING_PUNCTUATION = "),.!?;:>\"'`]}"

_NON_GIT_FILE_EXTENSIONS = frozenset(
    {".md", ".py", ".sh", ".bash", ".zsh", ".js", ".ts", ".rb", ".go", ".rs", ".pl"}
)
_SUPPORTED_FILE_EXTENSIONS = frozenset(
    {
        ".md",
        ".py",
        ".sh",
        ".bash",
        ".zsh",
        ".js",
        ".ts",
        ".rb",
        ".go",
        ".rs",
        ".pl",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".txt",
        ".zip",
    }
)

_EXTERNAL_REF_PATTERN = re.compile(r"(?:https?://|git@)[^\s\"'<>`]+")

_EXCLUDED_HOSTS = frozenset(
    {
        "img.shields.io",
        "badge.fury.io",
        "travis-ci.com",
        "travis-ci.org",
    }
)

_EXCLUDED_PATH_MARKERS = frozenset(
    {
        "/badge",
        "/badges",
        "/blob/",
        "/issues/",
        "/pull/",
        "/pulls/",
        "/actions/",
        "/workflows/",
        "/checks/",
        "/wiki",
        "/ci/",
    }
)


def canonicalize_source_identity(url: str) -> str:
    """Return canonical URL identity used for dedupe and visited-state control."""
    token = _clean_token(url).strip()
    if not token:
        raise ValueError(f"Unsupported URL: {url}")

    parsed = _parse_url(token)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"

    path = (parsed.path or "/").rstrip("/")
    path = path.removesuffix(".git")
    return urlunparse(("https", netloc, path if path else "/", "", "", ""))


def extract_external_refs(file_cache: dict[str, str]) -> list[str]:
    """Extract candidate external references from a file cache."""
    refs: list[str] = []
    seen: set[str] = set()
    for raw_content in file_cache.values():
        if not isinstance(raw_content, str):
            continue
        for match in _EXTERNAL_REF_PATTERN.finditer(raw_content):
            token = match.group(0)
            try:
                identity = canonicalize_source_identity(token)
            except ValueError:
                continue
            if identity in seen:
                continue
            if not _is_source_reference(identity):
                continue
            refs.append(identity)
            seen.add(identity)
    return refs


def plan_transitive_targets(
    refs: list[str],
    visited: set[str],
    current_depth: int,
    max_depth: int,
    allow_prefixes: tuple[str, ...],
    deny_prefixes: tuple[str, ...],
) -> list[str]:
    """Plan the next transitive scan wave and mutate visited for approved targets."""
    if current_depth > max_depth or max_depth <= 0:
        return []
    if current_depth < 1:
        current_depth = 1

    normalized_allow_prefixes = tuple(_normalize_prefix(p) for p in allow_prefixes)
    normalized_deny_prefixes = tuple(_normalize_prefix(p) for p in deny_prefixes)

    targets: list[str] = []
    for ref in refs:
        try:
            identity = canonicalize_source_identity(ref)
        except ValueError:
            continue
        if not _is_source_reference(identity):
            continue
        if identity in visited:
            continue
        if normalized_allow_prefixes and not _matches_any_prefix(
            identity, normalized_allow_prefixes
        ):
            continue
        if normalized_deny_prefixes and _matches_any_prefix(identity, normalized_deny_prefixes):
            continue
        visited.add(identity)
        targets.append(identity)
    return targets


def _parse_url(url: str) -> object:
    token = _clean_token(url)
    if token.startswith("git@"):
        return _parse_git_ssh_url(token)
    parsed = urlparse(token)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Unsupported URL: {url}")
    return parsed


def _parse_git_ssh_url(url: str):
    match = re.fullmatch(r"git@([^:]+):(.+)", url)
    if not match:
        raise ValueError(f"Unsupported git URL format: {url}")
    host = match.group(1).strip()
    path = match.group(2).strip().lstrip("/")
    return urlparse(f"https://{host}/{path}")


def _clean_token(token: str) -> str:
    cleaned = token.strip()
    while cleaned and cleaned[0] in _LEADING_PUNCTUATION:
        cleaned = cleaned[1:]
    while cleaned and cleaned[-1] in _TRAILING_PUNCTUATION:
        cleaned = cleaned[:-1]
    return cleaned.strip()


def _normalize_prefix(prefix: str) -> str:
    if not prefix:
        return ""
    return canonicalize_source_identity(prefix)


def _matches_any_prefix(url: str, prefixes: tuple[str, ...]) -> bool:
    return any(_matches_prefix(url, prefix) for prefix in prefixes if prefix)


def _matches_prefix(url: str, prefix: str) -> bool:
    if url == prefix:
        return True
    if prefix.endswith("/"):
        return url.startswith(prefix)
    return url.startswith(prefix + "/")


def _is_source_reference(identity: str) -> bool:
    parsed = urlparse(identity)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return False
    if host in _EXCLUDED_HOSTS:
        return False
    if not _is_allowed_host(host):
        return False

    lower_path = unquote(parsed.path).lower()
    if _has_excluded_path_marker(lower_path):
        return False

    if _looks_like_git_reference(host, lower_path):
        return True
    return _looks_like_file_reference(host, lower_path, parsed.path)


def _has_excluded_path_marker(path: str) -> bool:
    if path.endswith(".svg"):
        return True
    segments = [segment for segment in path.split("/") if segment]
    if len(segments) < 3:
        return False
    ui_segment = segments[2]
    return ui_segment in {
        "actions",
        "badge",
        "badges",
        "blob",
        "checks",
        "ci",
        "issues",
        "pull",
        "pulls",
        "tree",
        "wiki",
        "workflows",
    }


def _looks_like_git_reference(host: str, path: str) -> bool:
    if not _host_in_allowed_git_hosts(host):
        return False
    if not path or path == "/":
        return False
    if path.startswith("/raw/"):
        return False
    if path.startswith("/blob/"):
        return False
    if "/tree/" in path:
        return False

    segments = [segment for segment in path.split("/") if segment]
    if len(segments) < 2:
        return False
    if len(segments) >= 3 and segments[2] == "actions":
        return False

    lower = path.lower()
    return not any(lower.endswith(ext) for ext in _NON_GIT_FILE_EXTENSIONS)


def _looks_like_file_reference(host: str, lower_path: str, raw_path: str) -> bool:
    if not _is_allowed_host(host):
        return False
    if raw_path.endswith("/"):
        return False
    extension = _split_extension(lower_path)
    if not extension:
        return False
    return extension in _SUPPORTED_FILE_EXTENSIONS


def _is_allowed_host(host: str) -> bool:
    return (
        host in ALLOWED_GIT_HOSTS
        or host in {f"www.{entry}" for entry in ALLOWED_GIT_HOSTS}
        or host in ALLOWED_DOWNLOAD_HOSTS
        or host in {f"www.{entry}" for entry in ALLOWED_DOWNLOAD_HOSTS}
    )


def _host_in_allowed_git_hosts(host: str) -> bool:
    return host in ALLOWED_GIT_HOSTS or host in {f"www.{entry}" for entry in ALLOWED_GIT_HOSTS}


def _split_extension(path: str) -> str:
    return (
        "." + path.rsplit("/", 1)[-1].rsplit(".", 1)[-1] if "." in path.rsplit("/", 1)[-1] else ""
    )
