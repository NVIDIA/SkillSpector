# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, bounded credential-redaction boundary for dependency-source evidence."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from ipaddress import IPv6Address
from typing import Final
from urllib.parse import SplitResult, urlsplit

REDACTED_URL: Final = "[REDACTED_URL]"
REDACTED_REMAINDER: Final = "[REDACTED_REMAINDER]"
REDACTED_VALUE: Final = "[REDACTED_VALUE]"
REDACTED_PATH: Final = "REDACTED_PATH"

MAX_REDACTION_CHARACTERS: Final = 16 * 1024 * 1024
MAX_REDACTION_CANDIDATES: Final = 1_024
MAX_REDACTION_DEPTH: Final = 16
MAX_REDACTION_NODES: Final = 10_000
MAX_REDACTION_MAPPING_KEY_CHARACTERS: Final = 128

_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*$")
_HIERARCHICAL_MARKER = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")
_SAFE_PATH = re.compile(r"^/[A-Za-z0-9._~!$&'()*+,;=:@/-]*$")
_SAFE_SCP_PATH = re.compile(r"^[A-Za-z0-9._~!$&'()*+,;=:@/-]+$")
_CODE_OWNED_MAPPING_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_SCP_URL = re.compile(r"^(?P<user>[^@\s]+)@(?P<host>\[[^\]\s]+\]|[^@/:\\\s]+):(?P<path>.+)$")
_PROSE_OPENERS: Final = frozenset("([{<\"'`")
_PAIRED_CLOSERS: Final = {
    ")": "(",
    "]": "[",
    "}": "{",
    ">": "<",
    '"': '"',
    "'": "'",
    "`": "`",
}
_SENTENCE_PUNCTUATION: Final = frozenset(".,")
_SCHEME_RELATIVE_MARKER = re.compile(
    r"(?:^[\(\[\{<\"'`]?|\s[\(\[\{<\"'`]?|=[\(\[\{<\"'`]*|:[\(\[\{<\"'`]+|[>(])//"
)


@dataclass(frozen=True, slots=True, init=False)
class CodeOwnedMapping(Mapping[object, object]):
    """Immutable provenance marker for mappings assembled by trusted caller code."""

    _entries: tuple[tuple[object, object], ...]

    def __init__(self, values: Mapping[object, object]) -> None:
        if not isinstance(values, Mapping):
            raise ValueError("code-owned mapping values must be a mapping")
        object.__setattr__(self, "_entries", tuple(values.items()))

    def __getitem__(self, key: object) -> object:
        for candidate, value in self._entries:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[object]:
        return (key for key, _value in self._entries)

    def __len__(self) -> int:
        return len(self._entries)


def _valid_bound(value: object) -> bool:
    return type(value) is int and value >= 0


def _valid_dns_host(host: str) -> bool:
    if not host or len(host) > 253 or not host.isascii():
        return False
    labels = host.split(".")
    return all(
        label
        and len(label) <= 63
        and label[0].isalnum()
        and label[-1].isalnum()
        and all(character.isalnum() or character == "-" for character in label)
        for label in labels
    )


def _valid_bracketed_ipv6(host: str) -> bool:
    if not (host.startswith("[") and host.endswith("]")):
        return False
    try:
        IPv6Address(host[1:-1])
    except ValueError:
        return False
    return True


def _safe_authority(parsed: SplitResult) -> str | None:
    authority = parsed.netloc
    if (
        not authority
        or not authority.isascii()
        or "\\" in authority
        or _CONTROL_CHARACTER.search(authority)
        or any(character.isspace() for character in authority)
        or authority.count("@") > 1
    ):
        return None

    if "@" in authority:
        userinfo, host_port = authority.rsplit("@", 1)
        if not userinfo:
            return None
    else:
        host_port = authority

    try:
        hostname = parsed.hostname
        port = parsed.port
    except (UnicodeError, ValueError):
        return None
    if hostname is None:
        return None

    if host_port.startswith("["):
        close = host_port.find("]")
        if close < 0 or not _valid_bracketed_ipv6(host_port[: close + 1]):
            return None
        suffix = host_port[close + 1 :]
        if suffix and (not suffix.startswith(":") or not suffix[1:].isdigit()):
            return None
    else:
        if host_port.count(":") > 1:
            return None
        raw_host, separator, raw_port = host_port.partition(":")
        if not _valid_dns_host(raw_host):
            return None
        if separator and not raw_port.isdigit():
            return None
    if port is not None and not 0 <= port <= 65_535:
        return None
    return host_port


def _safe_path(path: str) -> str | None:
    if not path:
        return ""
    if path == "/":
        return "/"
    if not _SAFE_PATH.fullmatch(path) or "//" in path:
        return None
    return f"/{REDACTED_PATH}"


def _marker_count(value: str, *, max_count: int) -> int:
    if value == "//":
        return 0
    stop_after = max_count + 1
    count = 0
    hierarchical_count = 0
    scheme_relative_count = 0
    first_authority_start: int | None = None

    for match in _HIERARCHICAL_MARKER.finditer(value):
        hierarchical_count += 1
        count += 1
        if first_authority_start is None:
            first_authority_start = match.end()
        if count >= stop_after:
            return stop_after

    for match in _SCHEME_RELATIVE_MARKER.finditer(value):
        scheme_relative_count += 1
        count += 1
        if first_authority_start is None:
            first_authority_start = match.end()
        if count >= stop_after:
            return stop_after

    if _has_encoded_url_marker(value):
        count += 1
        if count >= stop_after:
            return stop_after

    if hierarchical_count or scheme_relative_count:
        raw_slashes = value.count("//")
        structural_slashes = hierarchical_count + scheme_relative_count
        count += max(0, raw_slashes - structural_slashes)
        if count >= stop_after:
            return stop_after
        if first_authority_start is not None and _has_nested_scp_marker(
            value, first_authority_start
        ):
            count += 1
    elif _has_encoded_scp_structure(value):
        count += 1
    elif _has_scp_structure(value):
        count += max(1, value.count("@"))
    return min(count, stop_after)


def _has_encoded_url_marker(value: str) -> bool:
    return "%" in value and "%2f%2f" in value.casefold()


def _has_scp_structure(value: str) -> bool:
    at_sign = value.find("@")
    return at_sign > 0 and value.find(":", at_sign + 1) > at_sign + 1


def _has_encoded_scp_structure(value: str) -> bool:
    if "%" not in value:
        return False
    folded = value.casefold()
    at_sign = folded.find("%40")
    return at_sign > 0 and folded.find(":", at_sign + 3) > at_sign + 3


def _has_nested_scp_marker(value: str, authority_start: int) -> bool:
    boundary = len(value)
    for delimiter in "/?#":
        position = value.find(delimiter, authority_start)
        if position >= 0:
            boundary = min(boundary, position)
    suffix = value[boundary:]
    at_sign = suffix.find("@")
    if at_sign < 0:
        return False
    colon = suffix.find(":", at_sign + 1)
    if colon < 0:
        return False
    path = suffix[colon + 1 :].split("?", 1)[0].split("#", 1)[0]
    return bool(path)


def _redact_hierarchical(value: str, *, scheme_relative: bool) -> str:
    try:
        parsed = urlsplit(value)
    except (UnicodeError, ValueError):
        return REDACTED_URL
    if scheme_relative:
        if parsed.scheme:
            return REDACTED_URL
        prefix = "//"
    else:
        if not _SCHEME.fullmatch(parsed.scheme):
            return REDACTED_URL
        prefix = f"{parsed.scheme}://"
    authority = _safe_authority(parsed)
    path = _safe_path(parsed.path)
    if authority is None or path is None:
        return REDACTED_URL
    return f"{prefix}{authority}{path}"


def _looks_like_scp_git(value: str) -> bool:
    base = re.split(r"[?#]", value, maxsplit=1)[0]
    return _SCP_URL.fullmatch(base) is not None


def _redact_scp(value: str) -> str:
    base = re.split(r"[?#]", value, maxsplit=1)[0]
    match = _SCP_URL.fullmatch(base)
    if match is None or value.count("@") != 1:
        return REDACTED_URL
    host = match.group("host")
    if not (_valid_bracketed_ipv6(host) if host.startswith("[") else _valid_dns_host(host)):
        return REDACTED_URL
    path = match.group("path")
    if not _SAFE_SCP_PATH.fullmatch(path) or "//" in path:
        return REDACTED_URL
    return f"REDACTED@{host}:{REDACTED_PATH}"


def redact_url(value: str, *, max_characters: int = MAX_REDACTION_CHARACTERS) -> str:
    """Sanitize one exact URL candidate or fail closed with a fixed placeholder."""
    if not isinstance(value, str) or not _valid_bound(max_characters):
        return REDACTED_URL
    if len(value) > max_characters:
        return REDACTED_URL

    try:
        markers = _marker_count(value, max_count=1)
    except Exception:
        return REDACTED_URL
    if markers == 0:
        return value if value == REDACTED_URL else REDACTED_URL
    if (
        markers != 1
        or "%" in value
        or _CONTROL_CHARACTER.search(value)
        or any(character.isspace() for character in value)
    ):
        return REDACTED_URL
    try:
        if value.startswith("//"):
            return _redact_hierarchical(value, scheme_relative=True)
        marker = _HIERARCHICAL_MARKER.match(value)
        if marker is not None:
            return _redact_hierarchical(value, scheme_relative=False)
        if _looks_like_scp_git(value):
            return _redact_scp(value)
        return REDACTED_URL
    except Exception:
        return REDACTED_URL


class TextRedactionIncompleteReason(StrEnum):
    """Content-free reason that bounded text redaction did not complete."""

    CHARACTER_LIMIT = "character_limit"
    CANDIDATE_LIMIT = "candidate_limit"
    INVALID_INPUT = "invalid_input"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True, slots=True)
class TextRedactionResult:
    """Sanitized text plus truthful completion and candidate-usage metadata."""

    value: str
    complete: bool
    candidates: int
    reason: TextRedactionIncompleteReason | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.value, str)
            or type(self.complete) is not bool
            or not _valid_bound(self.candidates)
            or (
                self.reason is not None
                and not isinstance(self.reason, TextRedactionIncompleteReason)
            )
            or self.complete is (self.reason is not None)
        ):
            raise ValueError("invalid text redaction result")


def _token_parts(token: str) -> tuple[str, str, str, str]:
    split_at = len(token)
    while split_at and token[split_at - 1] in _SENTENCE_PUNCTUATION:
        split_at -= 1
    punctuation = token[split_at:]
    core = token[:split_at]
    if len(core) >= 2 and core[0] in _PROSE_OPENERS:
        closer = core[-1]
        if _PAIRED_CLOSERS.get(closer) == core[0]:
            return core[0], core[1:-1], closer, punctuation
    return "", core, "", punctuation


def _might_contain_candidate(value: str) -> bool:
    return bool(
        "://" in value
        or _has_encoded_url_marker(value)
        or _has_encoded_scp_structure(value)
        or ("//" in value and _SCHEME_RELATIVE_MARKER.search(value))
        or ("@" in value and ":" in value)
    )


def _redact_text(value: str, *, max_candidates: int) -> TextRedactionResult:
    pieces: list[str] = []
    cursor = 0
    candidates = 0
    try:
        for match in re.finditer(r"\S+", value):
            token = match.group()
            opener, candidate, closer, punctuation = _token_parts(token)
            signals = _marker_count(
                candidate,
                max_count=max_candidates - candidates,
            )
            if signals == 0:
                continue
            if signals > max_candidates - candidates:
                return TextRedactionResult(
                    REDACTED_REMAINDER,
                    False,
                    candidates,
                    TextRedactionIncompleteReason.CANDIDATE_LIMIT,
                )
            pieces.append(value[cursor : match.start()])
            sanitized = redact_url(candidate, max_characters=len(candidate))
            if sanitized == REDACTED_URL:
                pieces.append(f"{REDACTED_URL}{punctuation}")
            else:
                pieces.append(f"{opener}{sanitized}{closer}{punctuation}")
            cursor = match.end()
            candidates += signals
        pieces.append(value[cursor:])
        return TextRedactionResult("".join(pieces), True, candidates, None)
    except Exception:
        return TextRedactionResult(
            REDACTED_REMAINDER,
            False,
            0,
            TextRedactionIncompleteReason.INTERNAL_ERROR,
        )


def redact_text_result(
    value: str,
    *,
    max_characters: int = MAX_REDACTION_CHARACTERS,
    max_candidates: int = MAX_REDACTION_CANDIDATES,
) -> TextRedactionResult:
    """Return bounded sanitized text with content-free completion metadata."""
    if (
        not isinstance(value, str)
        or not _valid_bound(max_characters)
        or not _valid_bound(max_candidates)
    ):
        return TextRedactionResult(
            REDACTED_REMAINDER,
            False,
            0,
            TextRedactionIncompleteReason.INVALID_INPUT,
        )
    if len(value) > max_characters:
        return TextRedactionResult(
            REDACTED_REMAINDER,
            False,
            0,
            TextRedactionIncompleteReason.CHARACTER_LIMIT,
        )
    try:
        if not _might_contain_candidate(value):
            return TextRedactionResult(value, True, 0, None)
        return _redact_text(value, max_candidates=max_candidates)
    except Exception:
        return TextRedactionResult(
            REDACTED_REMAINDER,
            False,
            0,
            TextRedactionIncompleteReason.INTERNAL_ERROR,
        )


def redact_text(
    value: str,
    *,
    max_characters: int = MAX_REDACTION_CHARACTERS,
    max_candidates: int = MAX_REDACTION_CANDIDATES,
) -> str:
    """Return the sanitized value from :func:`redact_text_result`."""
    return redact_text_result(
        value,
        max_characters=max_characters,
        max_candidates=max_candidates,
    ).value


class _AggregateRedactionExhaustedError(Exception):
    """Internal control flow for any recursive redaction failure."""


@dataclass(slots=True)
class _ValueWalk:
    remaining_nodes: int
    max_depth: int
    remaining_text_characters: int
    remaining_text_candidates: int
    active: set[int] = field(default_factory=set)

    def visit(self, value: object, depth: int) -> object:
        if depth > self.max_depth or self.remaining_nodes <= 0:
            raise _AggregateRedactionExhaustedError
        self.remaining_nodes -= 1

        if isinstance(value, str):
            if len(value) > self.remaining_text_characters:
                raise _AggregateRedactionExhaustedError
            text_result = redact_text_result(
                value,
                max_characters=self.remaining_text_characters,
                max_candidates=self.remaining_text_candidates,
            )
            if not text_result.complete:
                raise _AggregateRedactionExhaustedError
            self.remaining_text_characters -= len(value)
            self.remaining_text_candidates -= text_result.candidates
            return text_result.value
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, (CodeOwnedMapping, list, tuple)):
            identity = id(value)
            if identity in self.active or len(value) > self.remaining_nodes:
                raise _AggregateRedactionExhaustedError
            self.active.add(identity)
            try:
                if isinstance(value, CodeOwnedMapping):
                    mapping_result: dict[object, object] = {}
                    for key, nested in value._entries:
                        if (
                            not isinstance(key, str)
                            or len(key) > MAX_REDACTION_MAPPING_KEY_CHARACTERS
                            or _CODE_OWNED_MAPPING_KEY.fullmatch(key) is None
                            or redact_text(key) != key
                            or len(key) > self.remaining_text_characters
                        ):
                            raise _AggregateRedactionExhaustedError
                        self.remaining_text_characters -= len(key)
                        mapping_result[key] = self.visit(nested, depth + 1)
                    return CodeOwnedMapping(mapping_result)
                items = [self.visit(nested, depth + 1) for nested in value]
                return tuple(items) if isinstance(value, tuple) else items
            finally:
                self.active.remove(identity)
        if isinstance(value, Mapping):
            raise _AggregateRedactionExhaustedError
        return REDACTED_VALUE


def redact_value(
    value: object,
    *,
    max_depth: int = MAX_REDACTION_DEPTH,
    max_nodes: int = MAX_REDACTION_NODES,
    max_text_characters: int = MAX_REDACTION_CHARACTERS,
    max_text_candidates: int = MAX_REDACTION_CANDIDATES,
) -> object:
    """Recursively sanitize evidence values under one aggregate bounded walk."""
    if not all(
        _valid_bound(bound)
        for bound in (max_depth, max_nodes, max_text_characters, max_text_candidates)
    ):
        return REDACTED_VALUE
    try:
        return _ValueWalk(
            remaining_nodes=max_nodes,
            max_depth=max_depth,
            remaining_text_characters=max_text_characters,
            remaining_text_candidates=max_text_candidates,
        ).visit(value, 0)
    except Exception:
        return REDACTED_VALUE
