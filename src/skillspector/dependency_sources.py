# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, local-only analysis of direct dependency-source configuration files."""

from __future__ import annotations

import configparser
import json
import re
import tomllib
import xml.etree.ElementTree as ET
from bisect import bisect_left
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, cast
from urllib.parse import urlsplit

import yaml  # type: ignore[import-untyped]
from yaml.events import (  # type: ignore[import-untyped]
    AliasEvent,
    CollectionEndEvent,
    CollectionStartEvent,
    MappingEndEvent,
    MappingStartEvent,
    ScalarEvent,
    SequenceEndEvent,
    SequenceStartEvent,
)
from yaml.parser import ParserError  # type: ignore[import-untyped]
from yaml.scanner import ScannerError  # type: ignore[import-untyped]

from skillspector.artifacts import ArtifactDisposition, ArtifactRecord, ContentKind
from skillspector.dependency_source_types import (
    DependencyEcosystem,
    DependencyFileBudget,
    DependencySourceAnalysis,
    DependencySourceLimitation,
    DependencySourceLimitationReason,
    DependencySourceOperation,
    DependencySourceParseResult,
    DependencySourceScope,
    DependencySourceSpan,
    DependencySourceSurface,
    DependencyWorkBudget,
    DependencyWorkExhaustion,
    DestinationStatus,
    SourceChange,
    SourceSpan,
    finding_from_source_change,
)
from skillspector.url_redaction import redact_url

_NPM_BASENAMES: Final = frozenset({".npmrc", "npmrc"})
_PIP_BASENAMES: Final = frozenset({"pip.conf", "pip.ini"})
_YARN_V1_BASENAMES: Final = frozenset({".yarnrc"})
_YARN_YAML_BASENAMES: Final = frozenset({".yarnrc.yml", ".yarnrc.yaml"})
_PYTHON_PROJECT_BASENAMES: Final = frozenset({"pyproject.toml", "uv.toml"})
_MAVEN_BASENAMES: Final = frozenset({"settings.xml", "pom.xml"})
_CARGO_FILENAMES: Final = frozenset({"config", "config.toml"})
_RECOGNIZED_BASENAMES: Final = (
    _NPM_BASENAMES
    | _PIP_BASENAMES
    | _YARN_V1_BASENAMES
    | _YARN_YAML_BASENAMES
    | _PYTHON_PROJECT_BASENAMES
    | _MAVEN_BASENAMES
)
_NPM_SCOPED_REGISTRY: Final = re.compile(r"^@[^:\s]+:registry$", re.IGNORECASE)
_YARN_SCOPED_REGISTRY: Final = re.compile(r"^@[^:\s]+:registry$")
_NPM_INTERPOLATION: Final = re.compile(r"\$\{[^{}]+\}")
_PIP_INTERPOLATION: Final = re.compile(r"%\([^)]+\)s")
_PDM_INTERPOLATION: Final = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
_MAVEN_INTERPOLATION: Final = re.compile(r"\$\{[^{}]+\}")
_PIP_ASSIGNMENT: Final = re.compile(r"^\s*([^:=\s][^:=]*?)\s*([=:])\s*(.*)$")
_PIP_SECTION: Final = re.compile(r"^\s*\[([^]]+)]\s*(?:[#;].*)?$")
_PIP_OPTIONS: Final = ("index-url", "extra-index-url")
_SHELL_SUFFIXES: Final = frozenset({".sh", ".bash", ".zsh", ".ksh", ".envrc"})
_SHELL_NAMES: Final = frozenset({"sh", "bash", "dash", "zsh", "ksh"})
_MARKDOWN_SHELL_INFO: Final = _SHELL_NAMES | frozenset(
    {"shell", "console", "terminal", "shell-session"}
)
_SHELL_SHEBANG: Final = re.compile(
    r"^#!(?:/[^\s]*/(?:sh|bash|dash|zsh|ksh)|/usr/bin/env(?:[ \t]+-S)?[ \t]+(?:sh|bash|dash|zsh|ksh))(?:[ \t]|$)"
)
_MARKDOWN_FENCE_OPEN: Final = re.compile(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)$")
_CANONICAL_DEFAULTS: Final[dict[DependencyEcosystem, frozenset[str]]] = {
    DependencyEcosystem.NPM: frozenset({"https://registry.npmjs.org/"}),
    DependencyEcosystem.YARN: frozenset({"https://registry.yarnpkg.com/"}),
    DependencyEcosystem.PIP: frozenset({"https://pypi.org/simple/"}),
    DependencyEcosystem.POETRY: frozenset({"https://pypi.org/simple/"}),
    DependencyEcosystem.PDM: frozenset({"https://pypi.org/simple/"}),
    DependencyEcosystem.UV: frozenset({"https://pypi.org/simple/"}),
    DependencyEcosystem.CARGO: frozenset(
        {
            "https://github.com/rust-lang/crates.io-index",
            "sparse+https://index.crates.io/",
        }
    ),
    DependencyEcosystem.MAVEN: frozenset({"https://repo.maven.apache.org/maven2/"}),
}
_MISSING: Final = object()
_WRONG_SHAPE: Final = object()


@dataclass(frozen=True, slots=True)
class _Candidate:
    ecosystem: DependencyEcosystem
    surface: DependencySourceSurface
    operation: DependencySourceOperation
    scope: DependencySourceScope
    span: SourceSpan
    destination: str | None = None


@dataclass(frozen=True, slots=True)
class _ValueFragment:
    line: int
    start_byte: int
    end_byte: int


@dataclass(slots=True)
class _YamlNode:
    kind: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    value: str | None = None
    tag: str | None = None
    anchor: str | None = None
    items: list[_YamlNode | tuple[_YamlNode, _YamlNode]] = field(default_factory=list)


@dataclass(slots=True)
class _YamlFrame:
    node: _YamlNode
    pending_key: _YamlNode | None = None


@dataclass(slots=True)
class _TomlTableCursor:
    path: tuple[str, ...]
    url_span: SourceSpan | None = None


@dataclass(frozen=True, slots=True)
class _XmlSemanticRecord:
    parent_path: tuple[str, ...]
    destination: str
    operation: DependencySourceOperation
    scope: DependencySourceScope


@dataclass(slots=True)
class _XmlFrame:
    name: str
    element: ET.Element
    accepted: bool
    urls: list[tuple[str | None, bool]] = field(default_factory=list)
    had_child: bool = False


@dataclass(slots=True)
class _XmlLexicalFrame:
    name: str
    inner_start: int
    has_markup: bool = False


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _is_cargo_path(path: str) -> bool:
    parts = path.split("/")
    return len(parts) >= 2 and parts[-2] == ".cargo" and parts[-1] in _CARGO_FILENAMES


def _is_recognized_path(path: str) -> bool:
    return _basename(path) in _RECOGNIZED_BASENAMES or _is_cargo_path(path)


def _line_count(raw: bytes | None) -> int:
    return max(1, raw.count(b"\n") + 1) if raw is not None else 1


def _limitation(
    path: str,
    raw: bytes | None,
    exhaustion: DependencyWorkExhaustion | None = None,
) -> DependencySourceLimitation:
    metrics = exhaustion.ledger_metrics() if exhaustion is not None else {}
    return DependencySourceLimitation(
        reason=DependencySourceLimitationReason.PARSE_INCOMPLETE,
        path=path,
        start_line=1,
        end_line=_line_count(raw),
        **metrics,
    )


def _is_complete_text_record(record: ArtifactRecord, raw_size: int) -> bool:
    try:
        return (
            record.get("content_kind") == ContentKind.TEXT
            and record.get("disposition") == ArtifactDisposition.ANALYZED
            and record.get("decodable") is True
            and record.get("contains_nul") is False
            and type(record.get("size_bytes")) is int
            and record["size_bytes"] == raw_size
        )
    except (KeyError, TypeError):
        return False


def _inventory_size(record: ArtifactRecord | None) -> int:
    if record is None:
        return 0
    size = record.get("size_bytes")
    return size if type(size) is int and size >= 0 else 0


def _physical_lines(text: str) -> list[str]:
    """Split only on LF while removing the CR that belongs to a CRLF boundary."""
    return [part[:-1] if part.endswith("\r") else part for part in text.split("\n")]


def _whole_file_span(path: str, raw: bytes | None) -> DependencySourceSpan:
    return DependencySourceSpan(path=path, start_line=1, end_line=_line_count(raw))


def _is_shell_shebang(line: str) -> bool:
    return _SHELL_SHEBANG.match(line) is not None


def _markdown_executable_ranges(lines: list[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        opener = _MARKDOWN_FENCE_OPEN.match(lines[index])
        if opener is None:
            index += 1
            continue
        fence = opener.group(1)
        info = opener.group(2).strip()
        token = info.split(maxsplit=1)[0].casefold() if info else ""
        closer = re.compile(rf"^ {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$")
        end_index = index + 1
        while end_index < len(lines) and closer.match(lines[end_index]) is None:
            end_index += 1
        content_end = min(end_index, len(lines))
        relevant = token in _MARKDOWN_SHELL_INFO
        if not info:
            first_content = next(
                (line for line in lines[index + 1 : content_end] if line.strip()),
                "",
            )
            relevant = (
                _is_shell_shebang(first_content)
                or first_content.startswith("$ ")
                or first_content.startswith("# ")
            )
        if relevant:
            ranges.append((index + 1, min(end_index + 1, len(lines))))
        index = end_index + 1 if end_index < len(lines) else len(lines)
    return ranges


def _make_recipe_ranges(lines: list[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        if not lines[index].startswith("\t"):
            index += 1
            continue
        start = index
        index += 1
        while index < len(lines) and (
            lines[index].startswith("\t") or lines[index - 1].rstrip().endswith("\\")
        ):
            index += 1
        ranges.append((start + 1, index))
    return ranges


def _executable_surface_ranges(
    path: str,
    text: str,
    raw: bytes,
    executable_paths: frozenset[str],
) -> list[DependencySourceSpan]:
    """Identify bounded executable shapes without interpreting command semantics."""
    lines = _physical_lines(text)
    basename = _basename(path)
    lower_path = path.casefold()
    whole_file = (1, _line_count(raw))
    ranges: list[tuple[int, int]] = []

    if (
        any(lower_path.endswith(suffix) for suffix in _SHELL_SUFFIXES)
        or (lines and _is_shell_shebang(lines[0]))
        or path in executable_paths
    ):
        ranges.append(whole_file)
    elif (basename == "Dockerfile" or basename.startswith("Dockerfile.")) and any(
        re.match(r"^[ \t]*RUN(?:[ \t]|$)", line, re.IGNORECASE) for line in lines
    ):
        ranges.append(whole_file)
    elif basename in {"Makefile", "makefile", "GNUmakefile"} or basename.endswith(".mk"):
        ranges.extend(_make_recipe_ranges(lines))
    elif lower_path.endswith((".md", ".markdown", ".mdown", ".mkd")):
        ranges.extend(_markdown_executable_ranges(lines))

    return [
        DependencySourceSpan(path=path, start_line=start_line, end_line=end_line)
        for start_line, end_line in dict.fromkeys(ranges)
    ]


def _line_offsets(text: str) -> list[int]:
    offsets: list[int] = []
    current = 0
    parts = text.split("\n")
    for index, line in enumerate(parts):
        offsets.append(current)
        current += len(line.encode("utf-8"))
        if index < len(parts) - 1:
            current += 1
    return offsets


def _byte_range(line: str, line_offset: int, start: int, end: int) -> tuple[int, int]:
    return (
        line_offset + len(line[:start].encode("utf-8")),
        line_offset + len(line[:end].encode("utf-8")),
    )


def _strip_comment(value: str) -> str:
    quote: str | None = None
    for index, character in enumerate(value):
        if character in {'"', "'"}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            continue
        if quote is None and character in {"#", ";"} and index > 0 and value[index - 1].isspace():
            return value[:index].rstrip()
    return value.rstrip()


def _normalize_literal(value: str) -> tuple[str, int, int] | None:
    left_trimmed = value.lstrip()
    left = len(value) - len(left_trimmed)
    without_comment = _strip_comment(left_trimmed)
    trimmed = without_comment.rstrip()
    if not trimmed:
        return None
    if trimmed[0] in {'"', "'"}:
        quote = trimmed[0]
        if len(trimmed) < 2 or trimmed[-1] != quote:
            return None
        literal = trimmed[1:-1]
        if not literal:
            return None
        return literal, left + 1, left + len(trimmed) - 1
    if trimmed[-1] in {'"', "'"}:
        return None
    return trimmed, left, left + len(trimmed)


def _normalize_pip_option(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith("--") and not normalized.startswith("---"):
        normalized = normalized[2:]
    return normalized.casefold().replace("_", "-")


class _PipConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return _normalize_pip_option(optionstr)


def _canonical_destination(ecosystem: DependencyEcosystem, value: str) -> bool:
    if "?" in value or "#" in value:
        return False
    try:
        parsed = urlsplit(value)
        if (
            parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
            or parsed.hostname is None
            or parsed.netloc.casefold() != parsed.hostname.casefold()
            or parsed.query
            or parsed.fragment
        ):
            return False
    except (TypeError, ValueError):
        return False
    for literal in _CANONICAL_DEFAULTS.get(ecosystem, frozenset()):
        canonical = urlsplit(literal)
        canonical_hostname = canonical.hostname
        if canonical_hostname is None:
            continue
        if (
            parsed.scheme.casefold() == canonical.scheme.casefold()
            and parsed.hostname.casefold() == canonical_hostname.casefold()
            and parsed.path.removesuffix("/") == canonical.path.removesuffix("/")
        ):
            return True
    return False


def _destination(
    ecosystem: DependencyEcosystem,
    raw_destination: str,
) -> tuple[str, DestinationStatus] | None:
    if _canonical_destination(ecosystem, raw_destination):
        return None
    interpolation = {
        DependencyEcosystem.NPM: _NPM_INTERPOLATION,
        DependencyEcosystem.PIP: _PIP_INTERPOLATION,
        DependencyEcosystem.PDM: _PDM_INTERPOLATION,
        DependencyEcosystem.MAVEN: _MAVEN_INTERPOLATION,
    }.get(ecosystem)
    if interpolation is not None and interpolation.search(raw_destination):
        return "unresolved", DestinationStatus.UNRESOLVED
    return redact_url(raw_destination), DestinationStatus.RESOLVED


def _candidate_change(
    candidate: _Candidate,
    raw: bytes,
    budget: DependencyFileBudget,
) -> tuple[SourceChange | None, DependencyWorkExhaustion | None]:
    if exhaustion := budget.charge_source_records(1):
        return None, exhaustion
    raw_destination = candidate.destination
    if raw_destination is None:
        raw_destination = raw[candidate.span.start_byte : candidate.span.end_byte].decode("utf-8")
    literal_bytes = len(raw_destination.encode("utf-8"))
    if exhaustion := budget.charge_retained_literal_bytes(literal_bytes):
        return None, exhaustion
    normalized = _destination(candidate.ecosystem, raw_destination)
    if normalized is None:
        return None, None
    if exhaustion := budget.reserve_source_changes():
        return None, exhaustion
    destination, status = normalized
    return (
        SourceChange(
            ecosystem=candidate.ecosystem,
            surface=candidate.surface,
            operation=candidate.operation,
            scope=candidate.scope,
            destination=destination,
            destination_status=status,
            span=candidate.span,
        ),
        None,
    )


def _changes_from_candidates(
    candidates: Sequence[_Candidate],
    *,
    path: str,
    raw: bytes,
    budget: DependencyFileBudget,
    atomic: bool = False,
) -> DependencySourceParseResult:
    if atomic:
        prepared: list[tuple[_Candidate, str, DestinationStatus]] = []
        retained_literal_bytes = 0
        for candidate in candidates:
            raw_destination = candidate.destination
            if raw_destination is None:
                raw_destination = raw[candidate.span.start_byte : candidate.span.end_byte].decode(
                    "utf-8"
                )
            retained_literal_bytes += len(raw_destination.encode("utf-8"))
            normalized = _destination(candidate.ecosystem, raw_destination)
            if normalized is not None:
                prepared.append((candidate, *normalized))
        exhaustion = budget.reserve_source_batch(
            source_records=len(candidates),
            retained_literal_bytes=retained_literal_bytes,
            emitted_changes=len(prepared),
        )
        if exhaustion is not None:
            return DependencySourceParseResult(
                limitations=(_limitation(path, raw, exhaustion),),
            )
        return DependencySourceParseResult(
            changes=tuple(
                SourceChange(
                    ecosystem=candidate.ecosystem,
                    surface=candidate.surface,
                    operation=candidate.operation,
                    scope=candidate.scope,
                    destination=destination,
                    destination_status=status,
                    span=candidate.span,
                )
                for candidate, destination, status in prepared
            )
        )

    changes: list[SourceChange] = []
    for candidate in candidates:
        change, exhaustion = _candidate_change(candidate, raw, budget)
        if exhaustion is not None:
            return DependencySourceParseResult(
                changes=() if atomic else tuple(changes),
                limitations=(_limitation(path, raw, exhaustion),),
            )
        if change is not None:
            changes.append(change)
    return DependencySourceParseResult(changes=tuple(changes))


def _parse_npm(
    path: str,
    text: str,
    raw: bytes,
    budget: DependencyFileBudget,
) -> DependencySourceParseResult:
    effective: dict[str, _Candidate] = {}
    offsets = _line_offsets(text)
    for line_number, line in enumerate(_physical_lines(text), start=1):
        stripped = line.lstrip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        if "=" not in line:
            possible_key = stripped.split(None, 1)[0].lower()
            if possible_key == "registry" or _NPM_SCOPED_REGISTRY.fullmatch(possible_key):
                return DependencySourceParseResult(limitations=(_limitation(path, raw),))
            continue
        key_part, value_part = line.split("=", 1)
        key = key_part.strip().lower()
        if key != "registry" and _NPM_SCOPED_REGISTRY.fullmatch(key) is None:
            continue
        if exhaustion := budget.charge_config_nodes(1):
            return DependencySourceParseResult(limitations=(_limitation(path, raw, exhaustion),))
        normalized = _normalize_literal(value_part)
        if normalized is None:
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        _literal, relative_start, relative_end = normalized
        value_column = line.index("=") + 1
        start = value_column + relative_start
        end = value_column + relative_end
        start_byte, end_byte = _byte_range(line, offsets[line_number - 1], start, end)
        effective[key] = _Candidate(
            ecosystem=DependencyEcosystem.NPM,
            surface=DependencySourceSurface.NPMRC,
            operation=DependencySourceOperation.REPLACE,
            scope=(
                DependencySourceScope.GLOBAL if key == "registry" else DependencySourceScope.SCOPED
            ),
            span=SourceSpan(path, start_byte, end_byte, line_number, line_number),
        )
    return _changes_from_candidates(
        tuple(sorted(effective.values(), key=lambda candidate: candidate.span.start_byte)),
        path=path,
        raw=raw,
        budget=budget,
    )


def _pip_fragments(
    value: str,
    *,
    line: str,
    line_number: int,
    line_offset: int,
    value_column: int,
) -> list[_ValueFragment] | None:
    normalized = _normalize_literal(value)
    if normalized is None:
        return None
    literal, relative_start, relative_end = normalized
    absolute_start = value_column + relative_start
    fragments: list[_ValueFragment] = []
    for match in re.finditer(r"\S+", literal):
        token_start = absolute_start + match.start()
        token_end = absolute_start + match.end()
        start_byte, end_byte = _byte_range(line, line_offset, token_start, token_end)
        fragments.append(_ValueFragment(line_number, start_byte, end_byte))
    if not fragments or relative_end < relative_start:
        return None
    return fragments


def _pip_fragments_match_value(
    fragments: Sequence[_ValueFragment],
    configured_value: str,
    raw: bytes,
) -> bool:
    normalized = _normalize_literal(configured_value)
    if normalized is None:
        return False
    literal, _start, _end = normalized
    configured_tokens = re.findall(r"\S+", literal)
    occurrence_tokens = [
        raw[fragment.start_byte : fragment.end_byte].decode("utf-8") for fragment in fragments
    ]
    return configured_tokens == occurrence_tokens


def _parse_pip(
    path: str,
    text: str,
    raw: bytes,
    budget: DependencyFileBudget,
) -> DependencySourceParseResult:
    lines = _physical_lines(text)
    offsets = _line_offsets(text)
    section: str | None = None
    section_seen = False
    current_key: tuple[str | None, str] | None = None
    current_fragments: list[_ValueFragment] | None = None
    current_indent: int | None = None
    occurrences: dict[tuple[str | None, str], list[_ValueFragment]] = {}

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        section_match = _PIP_SECTION.fullmatch(line)
        if section_match is not None:
            if exhaustion := budget.charge_config_nodes(1):
                return DependencySourceParseResult(
                    limitations=(_limitation(path, raw, exhaustion),)
                )
            raw_section = section_match.group(1)
            section = None if raw_section == configparser.DEFAULTSECT else raw_section
            section_seen = True
            current_key = None
            current_fragments = None
            current_indent = None
            continue
        indent = len(line) - len(line.lstrip())
        if (
            current_key is not None
            and current_fragments is not None
            and current_indent is not None
            and indent > current_indent
        ):
            fragments = _pip_fragments(
                line,
                line=line,
                line_number=line_number,
                line_offset=offsets[line_number - 1],
                value_column=0,
            )
            if fragments is None:
                return DependencySourceParseResult(limitations=(_limitation(path, raw),))
            current_fragments.extend(fragments)
            occurrences[current_key] = current_fragments
            continue
        assignment = _PIP_ASSIGNMENT.fullmatch(line)
        current_key = None
        current_fragments = None
        current_indent = None
        if assignment is None:
            continue
        normalized_key = _normalize_pip_option(assignment.group(1))
        if normalized_key not in _PIP_OPTIONS:
            continue
        if not section_seen:
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        if exhaustion := budget.charge_config_nodes(1):
            return DependencySourceParseResult(limitations=(_limitation(path, raw, exhaustion),))
        value = assignment.group(3)
        fragments = _pip_fragments(
            value,
            line=line,
            line_number=line_number,
            line_offset=offsets[line_number - 1],
            value_column=assignment.start(3),
        )
        if fragments is None and value.strip():
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        current_key = (section, normalized_key)
        current_fragments = fragments or []
        current_indent = indent
        occurrences[current_key] = current_fragments

    if any(not fragments for fragments in occurrences.values()):
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))

    parser = _PipConfigParser(
        interpolation=None,
        strict=False,
        delimiters=("=", ":"),
    )
    try:
        parser.read_string(text)
    except configparser.Error:
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))

    candidates: list[_Candidate] = []
    for concrete_section in parser.sections():
        for normalized_key in _PIP_OPTIONS:
            configured_value = parser.get(
                concrete_section,
                normalized_key,
                raw=True,
                fallback=None,
            )
            if configured_value is None:
                continue
            fragments = occurrences.get((concrete_section, normalized_key))
            if fragments is None:
                fragments = occurrences.get((None, normalized_key))
            if fragments is None or not _pip_fragments_match_value(
                fragments,
                configured_value,
                raw,
            ):
                return DependencySourceParseResult(limitations=(_limitation(path, raw),))
            for fragment in fragments:
                candidates.append(
                    _Candidate(
                        ecosystem=DependencyEcosystem.PIP,
                        surface=DependencySourceSurface.PIP_CONFIG,
                        operation=(
                            DependencySourceOperation.REPLACE
                            if normalized_key == "index-url"
                            else DependencySourceOperation.ADD
                        ),
                        scope=(
                            DependencySourceScope.GLOBAL
                            if concrete_section == "global"
                            else DependencySourceScope.COMMAND
                        ),
                        span=SourceSpan(
                            path,
                            fragment.start_byte,
                            fragment.end_byte,
                            fragment.line,
                            fragment.line,
                        ),
                    )
                )
    candidates.sort(key=lambda candidate: candidate.span.start_byte)
    return _changes_from_candidates(candidates, path=path, raw=raw, budget=budget)


def _yarn_v1_tokens(line: str) -> tuple[list[tuple[str, int, int]], bool]:
    tokens: list[tuple[str, int, int]] = []
    index = 0
    while index < len(line):
        whitespace_start = index
        while index < len(line) and line[index].isspace():
            index += 1
        if index >= len(line):
            break
        if line[index] in {"#", ";"}:
            if not tokens or index > whitespace_start:
                break
            return tokens, True
        if len(tokens) == 2:
            return tokens, True
        if line[index] in {'"', "'"}:
            quote = line[index]
            start = index + 1
            index += 1
            escaped = False
            value: list[str] = []
            while index < len(line):
                character = line[index]
                if escaped:
                    value.append(character)
                    escaped = False
                elif character == "\\" and quote == '"':
                    escaped = True
                elif character == quote:
                    tokens.append(("".join(value), start, index))
                    index += 1
                    break
                else:
                    value.append(character)
                index += 1
            else:
                return tokens, True
        else:
            start = index
            while index < len(line) and not line[index].isspace():
                index += 1
            tokens.append((line[start:index], start, index))
    return tokens, False


def _parse_yarn_v1(
    path: str,
    text: str,
    raw: bytes,
    budget: DependencyFileBudget,
) -> DependencySourceParseResult:
    effective: dict[str, _Candidate] = {}
    offsets = _line_offsets(text)
    for line_number, line in enumerate(_physical_lines(text), start=1):
        stripped = line.lstrip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        tokens, malformed = _yarn_v1_tokens(line)
        if not tokens:
            continue
        key = tokens[0][0]
        relevant = key == "registry" or _YARN_SCOPED_REGISTRY.fullmatch(key) is not None
        if not relevant:
            continue
        if malformed or len(tokens) != 2 or not tokens[1][0]:
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        if exhaustion := budget.charge_config_nodes(1):
            return DependencySourceParseResult(limitations=(_limitation(path, raw, exhaustion),))
        value, start, end = tokens[1]
        start_byte, end_byte = _byte_range(line, offsets[line_number - 1], start, end)
        effective[key] = _Candidate(
            ecosystem=DependencyEcosystem.YARN,
            surface=DependencySourceSurface.YARN_CONFIG,
            operation=DependencySourceOperation.REPLACE,
            scope=(
                DependencySourceScope.GLOBAL if key == "registry" else DependencySourceScope.SCOPED
            ),
            span=SourceSpan(path, start_byte, end_byte, line_number, line_number),
            destination=value,
        )
    return _changes_from_candidates(
        tuple(sorted(effective.values(), key=lambda item: item.span.start_byte)),
        path=path,
        raw=raw,
        budget=budget,
    )


def _char_to_byte_offsets(text: str) -> list[int]:
    offsets = [0]
    current = 0
    for character in text:
        current += len(character.encode("utf-8"))
        offsets.append(current)
    return offsets


def _newline_offsets(value: str | bytes) -> tuple[int, ...]:
    """Index LF boundaries once for bounded source-span correlation."""
    if isinstance(value, bytes):
        return tuple(index for index, character in enumerate(value) if character == ord("\n"))
    return tuple(index for index, character in enumerate(value) if character == "\n")


def _line_number_at(newline_offsets: tuple[int, ...], offset: int) -> int:
    """Return the one-based physical line containing a half-open source offset."""
    return bisect_left(newline_offsets, offset) + 1


def _yaml_attach_node(
    node: _YamlNode,
    stack: list[_YamlFrame],
) -> None:
    if not stack:
        return
    frame = stack[-1]
    if frame.node.kind == "sequence":
        frame.node.items.append(node)
    elif frame.pending_key is None:
        frame.pending_key = node
    else:
        frame.node.items.append((frame.pending_key, node))
        frame.pending_key = None


def _yaml_event_tree(
    path: str,
    text: str,
    raw: bytes,
    budget: DependencyFileBudget,
) -> tuple[_YamlNode | None, dict[str, _YamlNode], DependencySourceParseResult | None]:
    root: _YamlNode | None = None
    anchors: dict[str, _YamlNode] = {}
    stack: list[_YamlFrame] = []
    try:
        events = yaml.parse(text, Loader=yaml.SafeLoader)
        for event in events:
            if isinstance(event, AliasEvent):
                if exhaustion := budget.charge_yaml_aliases(1):
                    return (
                        None,
                        {},
                        DependencySourceParseResult(
                            limitations=(_limitation(path, raw, exhaustion),)
                        ),
                    )
                if exhaustion := budget.charge_config_nodes(1):
                    return (
                        None,
                        {},
                        DependencySourceParseResult(
                            limitations=(_limitation(path, raw, exhaustion),)
                        ),
                    )
                node = _YamlNode(
                    "alias",
                    event.start_mark.index,
                    event.end_mark.index,
                    event.start_mark.line + 1,
                    max(
                        event.start_mark.line + 1,
                        event.end_mark.line
                        if event.end_mark.column == 0
                        and event.end_mark.index > event.start_mark.index
                        else event.end_mark.line + 1,
                    ),
                    value=event.anchor,
                )
                if root is None:
                    root = node
                _yaml_attach_node(node, stack)
                continue
            if isinstance(event, (ScalarEvent, MappingStartEvent, SequenceStartEvent)):
                if exhaustion := budget.charge_config_nodes(1):
                    return (
                        None,
                        {},
                        DependencySourceParseResult(
                            limitations=(_limitation(path, raw, exhaustion),)
                        ),
                    )
                kind = (
                    "scalar"
                    if isinstance(event, ScalarEvent)
                    else "mapping"
                    if isinstance(event, MappingStartEvent)
                    else "sequence"
                )
                node = _YamlNode(
                    kind,
                    event.start_mark.index,
                    event.end_mark.index,
                    event.start_mark.line + 1,
                    max(
                        event.start_mark.line + 1,
                        event.end_mark.line
                        if event.end_mark.column == 0
                        and event.end_mark.index > event.start_mark.index
                        else event.end_mark.line + 1,
                    ),
                    value=event.value if isinstance(event, ScalarEvent) else None,
                    tag=event.tag,
                    anchor=event.anchor,
                )
                if root is None:
                    root = node
                _yaml_attach_node(node, stack)
                if event.anchor is not None:
                    anchors[event.anchor] = node
                if isinstance(event, CollectionStartEvent):
                    depth = len(stack) + 1
                    if exhaustion := budget.observe_depth(depth):
                        return (
                            None,
                            {},
                            DependencySourceParseResult(
                                limitations=(_limitation(path, raw, exhaustion),)
                            ),
                        )
                    stack.append(_YamlFrame(node))
                continue
            if isinstance(event, (MappingEndEvent, SequenceEndEvent, CollectionEndEvent)):
                if not stack:
                    return (
                        None,
                        {},
                        DependencySourceParseResult(limitations=(_limitation(path, raw),)),
                    )
                frame = stack.pop()
                if frame.pending_key is not None:
                    return (
                        None,
                        {},
                        DependencySourceParseResult(limitations=(_limitation(path, raw),)),
                    )
    except (
        ScannerError,
        ParserError,
        yaml.YAMLError,
        ValueError,
        OverflowError,
        RecursionError,
    ):
        return None, {}, DependencySourceParseResult(limitations=(_limitation(path, raw),))
    if stack:
        return None, {}, DependencySourceParseResult(limitations=(_limitation(path, raw),))
    return root, anchors, None


def _bounded_loaded_object(
    value: object,
    budget: DependencyFileBudget,
) -> DependencyWorkExhaustion | bool | None:
    stack: list[tuple[object, int, frozenset[int]]] = [(value, 1, frozenset())]
    seen: set[int] = set()
    while stack:
        current, depth, ancestors = stack.pop()
        if not isinstance(current, (dict, list)):
            continue
        identity = id(current)
        if identity in ancestors:
            return True
        if identity in seen:
            continue
        seen.add(identity)
        if exhaustion := budget.observe_depth(depth):
            return exhaustion
        nested_ancestors = ancestors | {identity}
        if isinstance(current, dict):
            for key, nested in current.items():
                stack.append((key, depth + 1, nested_ancestors))
                stack.append((nested, depth + 1, nested_ancestors))
        else:
            for nested in current:
                stack.append((nested, depth + 1, nested_ancestors))
    return None


def _yaml_resolve(node: _YamlNode, anchors: Mapping[str, _YamlNode]) -> _YamlNode | None:
    seen: set[str] = set()
    current = node
    while current.kind == "alias":
        name = current.value
        if name is None or name in seen:
            return None
        seen.add(name)
        target = anchors.get(name)
        if target is None:
            return None
        current = target
    return current


def _yaml_key(node: _YamlNode, anchors: Mapping[str, _YamlNode]) -> str | None:
    resolved = _yaml_resolve(node, anchors)
    return resolved.value if resolved is not None and resolved.kind == "scalar" else None


def _yaml_has_explicit_tag(
    node: _YamlNode,
    anchors: Mapping[str, _YamlNode],
) -> bool:
    stack = [node]
    seen: set[int] = set()
    while stack:
        resolved = _yaml_resolve(stack.pop(), anchors)
        if resolved is None:
            return True
        identity = id(resolved)
        if identity in seen:
            continue
        seen.add(identity)
        if resolved.tag is not None:
            return True
        for item in resolved.items:
            if isinstance(item, tuple):
                stack.extend(item)
            elif isinstance(item, _YamlNode):
                stack.append(item)
    return False


def _yaml_contains_scalar(node: _YamlNode, value: str) -> bool:
    if node.kind == "scalar" and node.value == value:
        return True
    for item in node.items:
        if isinstance(item, tuple):
            if _yaml_contains_scalar(item[0], value) or _yaml_contains_scalar(item[1], value):
                return True
        elif isinstance(item, _YamlNode) and _yaml_contains_scalar(item, value):
            return True
    return False


def _yaml_pairs(node: _YamlNode) -> list[tuple[_YamlNode, _YamlNode]] | None:
    if node.kind != "mapping" or not all(isinstance(item, tuple) for item in node.items):
        return None
    return [item for item in node.items if isinstance(item, tuple)]


def _yaml_span(
    path: str,
    node: _YamlNode,
    byte_offsets: Sequence[int],
) -> SourceSpan:
    return SourceSpan(
        path,
        byte_offsets[node.start_char],
        byte_offsets[node.end_char],
        node.start_line,
        node.end_line,
    )


def _yaml_candidate(
    *,
    path: str,
    node: _YamlNode,
    evidence_node: _YamlNode,
    anchors: Mapping[str, _YamlNode],
    byte_offsets: Sequence[int],
    scope: DependencySourceScope,
    semantic_value: object,
) -> _Candidate | None:
    resolved = _yaml_resolve(node, anchors)
    if (
        resolved is None
        or resolved.kind != "scalar"
        or resolved.tag is not None
        or not isinstance(semantic_value, str)
        or not semantic_value
    ):
        return None
    return _Candidate(
        ecosystem=DependencyEcosystem.YARN,
        surface=DependencySourceSurface.YARN_CONFIG,
        operation=DependencySourceOperation.REPLACE,
        scope=scope,
        span=_yaml_span(path, evidence_node, byte_offsets),
        destination=semantic_value,
    )


def _parse_yarn_yaml(
    path: str,
    text: str,
    raw: bytes,
    budget: DependencyFileBudget,
) -> DependencySourceParseResult:
    root, anchors, failure = _yaml_event_tree(path, text, raw, budget)
    if failure is not None:
        return failure
    if root is None or root.kind != "mapping":
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))
    root_pairs = _yaml_pairs(root)
    if root_pairs is None:
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))

    has_relevant_root_key = any(
        _yaml_key(key, anchors) in {"npmRegistryServer", "npmScopes"} for key, _value in root_pairs
    )
    if root.tag is not None and has_relevant_root_key:
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))

    if any(
        _yaml_key(key, anchors) is None
        and any(
            _yaml_contains_scalar(key, relevant) for relevant in ("npmRegistryServer", "npmScopes")
        )
        for key, _value in root_pairs
    ):
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))

    try:
        loaded = yaml.safe_load(text)
    except (yaml.YAMLError, ValueError, OverflowError, RecursionError):
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))
    loaded_check = _bounded_loaded_object(loaded, budget)
    if loaded_check is True:
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))
    if isinstance(loaded_check, DependencyWorkExhaustion):
        return DependencySourceParseResult(limitations=(_limitation(path, raw, loaded_check),))
    if not isinstance(loaded, dict):
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))
    if any(_yaml_key(key, anchors) == "<<" for key, _value in root_pairs) and any(
        relevant in loaded for relevant in ("npmRegistryServer", "npmScopes")
    ):
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))

    byte_offsets = _char_to_byte_offsets(text)
    candidates: list[_Candidate] = []
    top_seen: set[str] = set()
    for key_node, value_node in root_pairs:
        key = _yaml_key(key_node, anchors)
        if key not in {"npmRegistryServer", "npmScopes"}:
            continue
        if key in top_seen or _yaml_has_explicit_tag(key_node, anchors):
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        top_seen.add(key)
        if key == "npmRegistryServer":
            candidate = _yaml_candidate(
                path=path,
                node=value_node,
                evidence_node=value_node,
                anchors=anchors,
                byte_offsets=byte_offsets,
                scope=DependencySourceScope.GLOBAL,
                semantic_value=loaded.get("npmRegistryServer", _MISSING),
            )
            if candidate is None:
                return DependencySourceParseResult(limitations=(_limitation(path, raw),))
            candidates.append(candidate)
            continue
        scopes = _yaml_resolve(value_node, anchors)
        if (
            scopes is None
            or scopes.kind != "mapping"
            or _yaml_has_explicit_tag(value_node, anchors)
            or _yaml_has_explicit_tag(scopes, anchors)
        ):
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        scope_pairs = _yaml_pairs(scopes)
        if scope_pairs is None:
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        loaded_scopes = loaded.get("npmScopes", _MISSING)
        if not isinstance(loaded_scopes, dict):
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        scope_seen: set[str] = set()
        for scope_key_node, scope_value_node in scope_pairs:
            scope_name = _yaml_key(scope_key_node, anchors)
            if scope_name is None or scope_name == "<<" or scope_name in scope_seen:
                return DependencySourceParseResult(limitations=(_limitation(path, raw),))
            if scope_name not in loaded_scopes:
                return DependencySourceParseResult(limitations=(_limitation(path, raw),))
            scope_seen.add(scope_name)
            scope_mapping = _yaml_resolve(scope_value_node, anchors)
            if (
                scope_mapping is None
                or scope_mapping.kind != "mapping"
                or _yaml_has_explicit_tag(scope_key_node, anchors)
                or _yaml_has_explicit_tag(scope_value_node, anchors)
            ):
                return DependencySourceParseResult(limitations=(_limitation(path, raw),))
            field_pairs = _yaml_pairs(scope_mapping)
            if field_pairs is None:
                return DependencySourceParseResult(limitations=(_limitation(path, raw),))
            registry_nodes: list[_YamlNode] = []
            for field_key_node, field_value_node in field_pairs:
                field_name = _yaml_key(field_key_node, anchors)
                if field_name == "<<" or field_name is None:
                    return DependencySourceParseResult(limitations=(_limitation(path, raw),))
                if field_name == "npmRegistryServer":
                    if registry_nodes or _yaml_has_explicit_tag(field_key_node, anchors):
                        return DependencySourceParseResult(limitations=(_limitation(path, raw),))
                    registry_nodes.append(field_value_node)
            if registry_nodes:
                evidence = (
                    scope_value_node if scope_value_node.kind == "alias" else registry_nodes[0]
                )
                candidate = _yaml_candidate(
                    path=path,
                    node=registry_nodes[0],
                    evidence_node=evidence,
                    anchors=anchors,
                    byte_offsets=byte_offsets,
                    scope=DependencySourceScope.SCOPED,
                    semantic_value=(
                        loaded_scopes[scope_name].get("npmRegistryServer", _MISSING)
                        if isinstance(loaded_scopes[scope_name], dict)
                        else _MISSING
                    ),
                )
                if candidate is None:
                    return DependencySourceParseResult(limitations=(_limitation(path, raw),))
                candidates.append(candidate)
    candidates.sort(key=lambda item: item.span.start_byte)
    return _changes_from_candidates(
        candidates,
        path=path,
        raw=raw,
        budget=budget,
        atomic=True,
    )


def _toml_key_parts(raw_key: str) -> tuple[str, ...] | None:
    parts: list[str] = []
    index = 0
    while index < len(raw_key):
        while index < len(raw_key) and raw_key[index].isspace():
            index += 1
        if index >= len(raw_key):
            return None
        if raw_key[index] in {'"', "'"}:
            quote = raw_key[index]
            start = index
            index += 1
            escaped = False
            while index < len(raw_key):
                character = raw_key[index]
                if escaped:
                    escaped = False
                elif character == "\\" and quote == '"':
                    escaped = True
                elif character == quote:
                    break
                index += 1
            if index >= len(raw_key):
                return None
            token = raw_key[start : index + 1]
            try:
                value = json.loads(token) if quote == '"' else token[1:-1]
            except (TypeError, ValueError):
                return None
            index += 1
        else:
            match = re.match(r"[A-Za-z0-9_-]+", raw_key[index:])
            if match is None:
                return None
            value = match.group(0)
            index += len(value)
        parts.append(value)
        while index < len(raw_key) and raw_key[index].isspace():
            index += 1
        if index == len(raw_key):
            break
        if raw_key[index] != ".":
            return None
        index += 1
    return tuple(parts)


def _toml_find_unquoted(text: str, target: str) -> int | None:
    quote: str | None = None
    escaped = False
    for index, character in enumerate(text):
        if quote == '"' and escaped:
            escaped = False
            continue
        if quote == '"' and character == "\\":
            escaped = True
            continue
        if quote is not None:
            if character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character == target:
            return index
    return None


def _toml_value_extent(text: str, start: int) -> int:
    if text.startswith(('"""', "'''"), start):
        delimiter = text[start : start + 3]
        index = start + 3
        while index < len(text):
            if text.startswith(delimiter, index):
                return index + 3
            if delimiter == '"""' and text[index] == "\\":
                index += 2
            else:
                index += 1
        return len(text)
    quote: str | None = None
    escaped = False
    index = start
    end = start
    while index < len(text) and text[index] not in "\r\n":
        character = text[index]
        if quote == '"' and escaped:
            escaped = False
        elif quote == '"' and character == "\\":
            escaped = True
        elif quote is not None:
            if character == quote:
                quote = None
        elif character in {'"', "'"}:
            quote = character
        elif character == "#":
            break
        if not character.isspace() or quote is not None:
            end = index + 1
        index += 1
    return end


def _toml_multiline_string_state(line: str, delimiter: str | None) -> str | None:
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(line):
        if delimiter is not None:
            if delimiter == '"""' and line[index] == "\\":
                index += 2
                continue
            if line.startswith(delimiter, index):
                delimiter = None
                index += 3
                continue
            index += 1
            continue
        if quote == '"' and escaped:
            escaped = False
            index += 1
            continue
        if quote == '"' and line[index] == "\\":
            escaped = True
            index += 1
            continue
        if quote is not None:
            if line[index] == quote:
                quote = None
            index += 1
            continue
        if line[index] == "#":
            break
        if line.startswith(('"""', "'''"), index):
            delimiter = line[index : index + 3]
            index += 3
            continue
        if line[index] in {'"', "'"}:
            quote = line[index]
        index += 1
    return delimiter


def _toml_url_cursors(
    path: str,
    text: str,
    relevant_paths: frozenset[tuple[str, ...]],
) -> dict[tuple[str, ...], list[_TomlTableCursor]] | None:
    cursors: dict[tuple[str, ...], list[_TomlTableCursor]] = {
        table_path: [] for table_path in relevant_paths
    }
    current: _TomlTableCursor | None = None
    byte_offsets = _char_to_byte_offsets(text)
    newline_offsets = _newline_offsets(text)
    position = 0
    multiline_delimiter: str | None = None
    while position < len(text):
        line_end = text.find("\n", position)
        if line_end < 0:
            line_end = len(text)
        physical_end = (
            line_end - 1 if line_end > position and text[line_end - 1] == "\r" else line_end
        )
        line = text[position:physical_end]
        stripped = line.lstrip()
        leading = len(line) - len(stripped)
        starts_in_multiline_string = multiline_delimiter is not None
        if not starts_in_multiline_string and stripped.startswith("[["):
            close = stripped.find("]]", 2)
            if close < 0:
                return None
            table_path = _toml_key_parts(stripped[2:close])
            current = None
            if table_path in relevant_paths:
                current = _TomlTableCursor(table_path)
                cursors[table_path].append(current)
        elif not starts_in_multiline_string and stripped.startswith("["):
            current = None
        elif (
            not starts_in_multiline_string
            and current is not None
            and stripped
            and not stripped.startswith("#")
        ):
            equals = _toml_find_unquoted(stripped, "=")
            if equals is not None and _toml_key_parts(stripped[:equals]) == ("url",):
                value_start = position + leading + equals + 1
                while value_start < len(text) and text[value_start] in " \t":
                    value_start += 1
                value_end = _toml_value_extent(text, value_start)
                start_line = _line_number_at(newline_offsets, value_start)
                end_line = _line_number_at(newline_offsets, value_end)
                if current.url_span is not None or value_end <= value_start:
                    return None
                current.url_span = SourceSpan(
                    path,
                    byte_offsets[value_start],
                    byte_offsets[value_end],
                    start_line,
                    end_line,
                )
                position = value_end
                next_newline = text.find("\n", position)
                position = len(text) if next_newline < 0 else next_newline + 1
                continue
        multiline_delimiter = _toml_multiline_string_state(line, multiline_delimiter)
        position = len(text) if line_end == len(text) else line_end + 1
    return cursors


def _toml_lookup(value: object, path: tuple[str, ...]) -> object:
    current = value
    for part in path:
        if not isinstance(current, dict):
            return _WRONG_SHAPE
        if part not in current:
            return _MISSING
        current = current[part]
    return current


def _toml_structural_check(
    value: object,
    budget: DependencyFileBudget,
) -> DependencyWorkExhaustion | None:
    stack: list[tuple[object, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        if exhaustion := budget.charge_config_nodes(1):
            return exhaustion
        if isinstance(current, (dict, list)):
            if exhaustion := budget.observe_depth(depth):
                return exhaustion
        if isinstance(current, dict):
            for key, nested in current.items():
                stack.append((nested, depth + 1))
                stack.append((key, depth + 1))
        elif isinstance(current, list):
            for nested in current:
                stack.append((nested, depth + 1))
    return None


def _python_candidate(
    *,
    path: str,
    ecosystem: DependencyEcosystem,
    operation: DependencySourceOperation,
    url: object,
    span: SourceSpan | None,
) -> _Candidate | None:
    if not isinstance(url, str) or not url or span is None:
        return None
    return _Candidate(
        ecosystem=ecosystem,
        surface=DependencySourceSurface.PYTHON_PROJECT_CONFIG,
        operation=operation,
        scope=DependencySourceScope.PROJECT,
        span=span,
        destination=url,
    )


def _parse_python_project(
    path: str,
    text: str,
    raw: bytes,
    budget: DependencyFileBudget,
    *,
    skip_pyproject_uv: bool,
) -> DependencySourceParseResult:
    try:
        document = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, ValueError, OverflowError, RecursionError):
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))
    if exhaustion := _toml_structural_check(document, budget):
        return DependencySourceParseResult(limitations=(_limitation(path, raw, exhaustion),))

    table_specs: list[tuple[tuple[str, ...], DependencyEcosystem]]
    if _basename(path) == "uv.toml":
        table_specs = [(("index",), DependencyEcosystem.UV)]
    else:
        table_specs = [
            (("tool", "poetry", "source"), DependencyEcosystem.POETRY),
            (("tool", "pdm", "source"), DependencyEcosystem.PDM),
        ]
        if not skip_pyproject_uv:
            table_specs.append((("tool", "uv", "index"), DependencyEcosystem.UV))
    relevant_paths = frozenset(path_parts for path_parts, _ecosystem in table_specs)
    cursors = _toml_url_cursors(path, text, relevant_paths)
    if cursors is None:
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))

    candidates: list[_Candidate] = []
    for table_path, ecosystem in table_specs:
        records = _toml_lookup(document, table_path)
        if records is _MISSING:
            continue
        if records is _WRONG_SHAPE:
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        if not isinstance(records, list):
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        locations = cursors[table_path]
        if not records and not locations:
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        if len(locations) != len(records):
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        for record, cursor in zip(records, locations, strict=True):
            if not isinstance(record, dict):
                return DependencySourceParseResult(limitations=(_limitation(path, raw),))
            url = record.get("url", _MISSING)
            if not isinstance(url, str) or not url:
                return DependencySourceParseResult(limitations=(_limitation(path, raw),))
            if ecosystem in {DependencyEcosystem.POETRY, DependencyEcosystem.PDM}:
                name = record.get("name", _MISSING)
                if not isinstance(name, str) or not name:
                    return DependencySourceParseResult(limitations=(_limitation(path, raw),))
            if ecosystem is DependencyEcosystem.POETRY:
                priority = record.get("priority", "primary")
                if priority not in {"primary", "supplemental", "explicit"}:
                    return DependencySourceParseResult(limitations=(_limitation(path, raw),))
                operation = (
                    DependencySourceOperation.REPLACE
                    if priority == "primary"
                    else DependencySourceOperation.ADD
                )
            elif ecosystem is DependencyEcosystem.PDM:
                operation = (
                    DependencySourceOperation.REPLACE
                    if record["name"] == "pypi"
                    else DependencySourceOperation.ADD
                )
            else:
                name = record.get("name", _MISSING)
                if name is not _MISSING and (not isinstance(name, str) or not name):
                    return DependencySourceParseResult(limitations=(_limitation(path, raw),))
                default = record.get("default", False)
                if type(default) is not bool:
                    return DependencySourceParseResult(limitations=(_limitation(path, raw),))
                operation = (
                    DependencySourceOperation.REPLACE if default else DependencySourceOperation.ADD
                )
            candidate = _python_candidate(
                path=path,
                ecosystem=ecosystem,
                operation=operation,
                url=url,
                span=cursor.url_span,
            )
            if candidate is None:
                return DependencySourceParseResult(limitations=(_limitation(path, raw),))
            candidates.append(candidate)
    candidates.sort(key=lambda item: item.span.start_byte)
    return _changes_from_candidates(
        candidates,
        path=path,
        raw=raw,
        budget=budget,
        atomic=True,
    )


def _toml_direct_value_cursors(
    path: str,
    text: str,
    relevant_roots: frozenset[str],
    relevant_keys: frozenset[str],
) -> dict[tuple[tuple[str, ...], str], SourceSpan] | None:
    cursors: dict[tuple[tuple[str, ...], str], SourceSpan] = {}
    current_table: tuple[str, ...] | None = None
    byte_offsets = _char_to_byte_offsets(text)
    newline_offsets = _newline_offsets(text)
    position = 0
    multiline_delimiter: str | None = None
    while position < len(text):
        line_end = text.find("\n", position)
        if line_end < 0:
            line_end = len(text)
        physical_end = (
            line_end - 1 if line_end > position and text[line_end - 1] == "\r" else line_end
        )
        line = text[position:physical_end]
        stripped = line.lstrip()
        leading = len(line) - len(stripped)
        starts_in_multiline_string = multiline_delimiter is not None
        if not starts_in_multiline_string and stripped.startswith("[["):
            current_table = None
        elif not starts_in_multiline_string and stripped.startswith("["):
            close = _toml_find_unquoted(stripped, "]")
            current_table = None
            if close is None:
                return None
            table_path = _toml_key_parts(stripped[1:close])
            if table_path is not None and len(table_path) == 2 and table_path[0] in relevant_roots:
                current_table = table_path
        elif (
            not starts_in_multiline_string
            and current_table is not None
            and stripped
            and not stripped.startswith("#")
        ):
            equals = _toml_find_unquoted(stripped, "=")
            if equals is not None:
                key_parts = _toml_key_parts(stripped[:equals])
                if key_parts is not None and len(key_parts) == 1 and key_parts[0] in relevant_keys:
                    value_start = position + leading + equals + 1
                    while value_start < len(text) and text[value_start] in " \t":
                        value_start += 1
                    value_end = _toml_value_extent(text, value_start)
                    cursor_key = (current_table, key_parts[0])
                    if cursor_key in cursors or value_end <= value_start:
                        return None
                    cursors[cursor_key] = SourceSpan(
                        path,
                        byte_offsets[value_start],
                        byte_offsets[value_end],
                        _line_number_at(newline_offsets, value_start),
                        _line_number_at(newline_offsets, value_end),
                    )
                    position = value_end
                    next_newline = text.find("\n", position)
                    position = len(text) if next_newline < 0 else next_newline + 1
                    continue
        multiline_delimiter = _toml_multiline_string_state(line, multiline_delimiter)
        position = len(text) if line_end == len(text) else line_end + 1
    return cursors


def _resolve_cargo_replacements(
    sources: Mapping[str, tuple[str, str, SourceSpan]],
    registries: Mapping[str, tuple[str, SourceSpan]],
) -> dict[str, str | None] | None:
    """Resolve every Cargo replacement once, memoizing shared chain suffixes."""
    memo: dict[str, str | None] = {}
    resolved_sources: dict[str, str | None] = {}
    for source_name, (kind, target_name, _span) in sources.items():
        if kind != "replace-with":
            continue
        seen = {source_name}
        traversed: list[str] = []
        current = target_name
        while True:
            if current in seen:
                return None
            seen.add(current)

            source = sources.get(current, _MISSING)
            registry = registries.get(current, _MISSING)
            if source is not _MISSING and registry is not _MISSING:
                return None
            if current in memo:
                destination = memo[current]
                break

            traversed.append(current)
            if source is not _MISSING:
                target_kind, target_value, _target_span = cast(tuple[str, str, SourceSpan], source)
                if target_kind == "replace-with":
                    current = target_value
                    continue
                destination = target_value if target_kind == "registry" else None
                break
            if registry is not _MISSING:
                destination = cast(tuple[str, SourceSpan], registry)[0]
                break
            return None

        for traversed_name in traversed:
            memo[traversed_name] = destination
        memo[source_name] = destination
        resolved_sources[source_name] = destination
    return resolved_sources


def _parse_cargo(
    path: str,
    text: str,
    raw: bytes,
    budget: DependencyFileBudget,
) -> DependencySourceParseResult:
    try:
        document = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, ValueError, OverflowError, RecursionError):
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))
    if exhaustion := _toml_structural_check(document, budget):
        return DependencySourceParseResult(limitations=(_limitation(path, raw, exhaustion),))
    cursors = _toml_direct_value_cursors(
        path,
        text,
        frozenset({"source", "registries"}),
        frozenset({"replace-with", "registry", "directory", "local-registry", "git", "index"}),
    )
    if cursors is None:
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))

    source_root = document.get("source", _MISSING)
    registry_root = document.get("registries", _MISSING)
    if source_root is not _MISSING and not isinstance(source_root, dict):
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))
    if registry_root is not _MISSING and not isinstance(registry_root, dict):
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))
    sources: dict[str, tuple[str, str, SourceSpan]] = {}
    registries: dict[str, tuple[str, SourceSpan]] = {}
    candidates: list[_Candidate] = []
    source_kinds = ("replace-with", "registry", "directory", "local-registry", "git")

    for name, record in source_root.items() if isinstance(source_root, dict) else ():
        if not name or not isinstance(record, dict):
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        configured = [kind for kind in source_kinds if kind in record]
        if len(configured) > 1:
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        if not configured:
            continue
        kind = configured[0]
        value = record[kind]
        span = cursors.get((("source", name), kind))
        if not isinstance(value, str) or not value.strip() or span is None:
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        sources[name] = (kind, value, span)
        if kind == "registry":
            candidates.append(
                _Candidate(
                    ecosystem=DependencyEcosystem.CARGO,
                    surface=DependencySourceSurface.CARGO_CONFIG,
                    operation=DependencySourceOperation.ADD,
                    scope=DependencySourceScope.REGISTRY,
                    span=span,
                    destination=value,
                )
            )

    for name, record in registry_root.items() if isinstance(registry_root, dict) else ():
        if not name or not isinstance(record, dict):
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        if "index" not in record:
            continue
        value = record["index"]
        span = cursors.get((("registries", name), "index"))
        if not isinstance(value, str) or not value.strip() or span is None:
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        registries[name] = (value, span)
        candidates.append(
            _Candidate(
                ecosystem=DependencyEcosystem.CARGO,
                surface=DependencySourceSurface.CARGO_CONFIG,
                operation=DependencySourceOperation.ADD,
                scope=DependencySourceScope.REGISTRY,
                span=span,
                destination=value,
            )
        )

    resolved_sources = _resolve_cargo_replacements(sources, registries)
    if resolved_sources is None:
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))
    for source_name, (kind, _target_name, replace_span) in sources.items():
        if kind != "replace-with":
            continue
        destination = resolved_sources[source_name]
        if destination is not None:
            candidates.append(
                _Candidate(
                    ecosystem=DependencyEcosystem.CARGO,
                    surface=DependencySourceSurface.CARGO_CONFIG,
                    operation=DependencySourceOperation.REPLACE,
                    scope=DependencySourceScope.SOURCE,
                    span=replace_span,
                    destination=destination,
                )
            )

    candidates.sort(key=lambda item: item.span.start_byte)
    return _changes_from_candidates(
        candidates,
        path=path,
        raw=raw,
        budget=budget,
        atomic=True,
    )


_MAVEN_PARENT_SEMANTICS: Final[
    dict[tuple[str, ...], tuple[DependencySourceOperation, DependencySourceScope]]
] = {
    ("settings", "mirrors", "mirror"): (
        DependencySourceOperation.REPLACE,
        DependencySourceScope.MIRROR,
    ),
    ("settings", "profiles", "profile", "repositories", "repository"): (
        DependencySourceOperation.ADD,
        DependencySourceScope.REPOSITORY,
    ),
    ("settings", "profiles", "profile", "pluginRepositories", "pluginRepository"): (
        DependencySourceOperation.ADD,
        DependencySourceScope.REPOSITORY,
    ),
    ("project", "repositories", "repository"): (
        DependencySourceOperation.ADD,
        DependencySourceScope.REPOSITORY,
    ),
    ("project", "pluginRepositories", "pluginRepository"): (
        DependencySourceOperation.ADD,
        DependencySourceScope.REPOSITORY,
    ),
}


def _xml_local_name(tag: object) -> str | None:
    if not isinstance(tag, str):
        return None
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _xml_semantic_records(
    path: str,
    text: str,
    raw: bytes,
    budget: DependencyFileBudget,
) -> tuple[list[_XmlSemanticRecord] | None, bool, DependencySourceParseResult | None]:
    parser = ET.XMLPullParser(events=("start", "end"))
    frames: list[_XmlFrame] = []
    records: list[_XmlSemanticRecord] = []
    root_name: str | None = None
    invalid_relevant = False

    def consume_events() -> DependencyWorkExhaustion | bool | None:
        nonlocal root_name, invalid_relevant
        for raw_event in parser.read_events():
            event, element = cast(tuple[str, ET.Element], raw_event)
            if event == "start":
                name = _xml_local_name(element.tag)
                if name is None:
                    return True
                if exhaustion := budget.charge_config_nodes(1):
                    return exhaustion
                depth = len(frames) + 1
                if exhaustion := budget.observe_depth(depth):
                    return exhaustion
                if frames:
                    frames[-1].had_child = True
                else:
                    root_name = name
                parent_path = tuple(frame.name for frame in frames) + (name,)
                frames.append(
                    _XmlFrame(
                        name=name,
                        element=element,
                        accepted=parent_path in _MAVEN_PARENT_SEMANTICS,
                    )
                )
                continue
            if not frames or frames[-1].element is not element:
                return True
            current_path = tuple(frame.name for frame in frames)
            frame = frames[-1]
            if frame.name == "url" and len(frames) >= 2 and frames[-2].accepted:
                frames[-2].urls.append((element.text, frame.had_child or bool(element.attrib)))
            if frame.accepted:
                if len(frame.urls) != 1:
                    invalid_relevant = True
                else:
                    value, unsupported = frame.urls[0]
                    normalized = value.strip() if isinstance(value, str) else ""
                    if unsupported or not normalized:
                        invalid_relevant = True
                    else:
                        operation, scope = _MAVEN_PARENT_SEMANTICS[current_path]
                        records.append(
                            _XmlSemanticRecord(current_path, normalized, operation, scope)
                        )
            frames.pop()
            if frames:
                try:
                    frames[-1].element.remove(element)
                except ValueError:
                    return True
            element.clear()
        return None

    try:
        for position in range(0, len(text), 4096):
            parser.feed(text[position : position + 4096])
            failure = consume_events()
            if failure is not None:
                if isinstance(failure, DependencyWorkExhaustion):
                    return (
                        None,
                        False,
                        DependencySourceParseResult(limitations=(_limitation(path, raw, failure),)),
                    )
                return (
                    None,
                    False,
                    DependencySourceParseResult(limitations=(_limitation(path, raw),)),
                )
        parser.close()
        failure = consume_events()
        if failure is not None:
            if isinstance(failure, DependencyWorkExhaustion):
                return (
                    None,
                    False,
                    DependencySourceParseResult(limitations=(_limitation(path, raw, failure),)),
                )
            return None, False, DependencySourceParseResult(limitations=(_limitation(path, raw),))
    except (ET.ParseError, ValueError, OverflowError, RecursionError):
        return None, False, DependencySourceParseResult(limitations=(_limitation(path, raw),))
    if frames:
        return None, False, DependencySourceParseResult(limitations=(_limitation(path, raw),))
    expected_root = "settings" if _basename(path) == "settings.xml" else "project"
    applicable = root_name == expected_root
    if not applicable:
        return [], False, None
    if invalid_relevant:
        return None, True, DependencySourceParseResult(limitations=(_limitation(path, raw),))
    return records, True, None


def _xml_tag_end(raw: bytes, start: int) -> int | None:
    quote: int | None = None
    index = start
    while index < len(raw):
        character = raw[index]
        if quote is not None:
            if character == quote:
                quote = None
        elif character in {ord('"'), ord("'")}:
            quote = character
        elif character == ord(">"):
            return index + 1
        index += 1
    return None


def _xml_raw_local_name(token: bytes) -> str | None:
    raw_name = token.strip().split(None, 1)[0].rstrip(b"/") if token.strip() else b""
    if not raw_name:
        return None
    try:
        return raw_name.rsplit(b":", 1)[-1].decode("utf-8")
    except UnicodeDecodeError:
        return None


def _xml_url_spans(path: str, raw: bytes) -> list[tuple[tuple[str, ...], SourceSpan, bool]] | None:
    stack: list[_XmlLexicalFrame] = []
    spans: list[tuple[tuple[str, ...], SourceSpan, bool]] = []
    newline_offsets = _newline_offsets(raw)
    index = 0
    while index < len(raw):
        marker = raw.find(b"<", index)
        if marker < 0:
            break
        if raw.startswith(b"<!--", marker):
            end = raw.find(b"-->", marker + 4)
            if end < 0:
                return None
            if stack and stack[-1].name == "url":
                stack[-1].has_markup = True
            index = end + 3
            continue
        if raw.startswith(b"<![CDATA[", marker):
            end = raw.find(b"]]>", marker + 9)
            if end < 0:
                return None
            if stack and stack[-1].name == "url":
                stack[-1].has_markup = True
            index = end + 3
            continue
        if raw.startswith(b"<?", marker):
            end = raw.find(b"?>", marker + 2)
            if end < 0:
                return None
            if stack and stack[-1].name == "url":
                stack[-1].has_markup = True
            index = end + 2
            continue
        tag_end = _xml_tag_end(raw, marker + 1)
        if tag_end is None:
            return None
        token = raw[marker + 1 : tag_end - 1]
        if token.startswith(b"/"):
            name = _xml_raw_local_name(token[1:])
            if name is None or not stack or stack[-1].name != name:
                return None
            frame = stack.pop()
            parent_path = tuple(item.name for item in stack)
            if name == "url" and parent_path in _MAVEN_PARENT_SEMANTICS:
                span_start = frame.inner_start
                span_end = marker
                while span_start < span_end and raw[span_start] in b" \t\r\n":
                    span_start += 1
                while span_end > span_start and raw[span_end - 1] in b" \t\r\n":
                    span_end -= 1
                spans.append(
                    (
                        parent_path,
                        SourceSpan(
                            path,
                            span_start,
                            span_end,
                            _line_number_at(newline_offsets, span_start),
                            _line_number_at(newline_offsets, span_end),
                        ),
                        frame.has_markup,
                    )
                )
        elif token.startswith(b"!"):
            return None
        else:
            name = _xml_raw_local_name(token)
            if name is None:
                return None
            if stack and stack[-1].name == "url":
                stack[-1].has_markup = True
            self_closing = token.rstrip().endswith(b"/")
            if not self_closing:
                stack.append(_XmlLexicalFrame(name=name, inner_start=tag_end))
        index = tag_end
    return spans if not stack else None


def _parse_maven(
    path: str,
    text: str,
    raw: bytes,
    budget: DependencyFileBudget,
) -> DependencySourceParseResult:
    if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))
    records, applicable, failure = _xml_semantic_records(path, text, raw, budget)
    if failure is not None:
        return failure
    if not applicable:
        return DependencySourceParseResult()
    if records is None:
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))
    lexical = _xml_url_spans(path, raw)
    if lexical is None or len(lexical) != len(records):
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))
    candidates: list[_Candidate] = []
    for record, (parent_path, span, has_markup) in zip(records, lexical, strict=True):
        if has_markup or parent_path != record.parent_path:
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        candidates.append(
            _Candidate(
                ecosystem=DependencyEcosystem.MAVEN,
                surface=DependencySourceSurface.MAVEN_CONFIG,
                operation=record.operation,
                scope=record.scope,
                span=span,
                destination=record.destination,
            )
        )
    return _changes_from_candidates(
        candidates,
        path=path,
        raw=raw,
        budget=budget,
        atomic=True,
    )


def _parse_file(
    path: str,
    text: str,
    raw: bytes,
    budget: DependencyFileBudget,
    *,
    skip_pyproject_uv: bool = False,
) -> DependencySourceParseResult:
    basename = _basename(path)
    if basename in _NPM_BASENAMES:
        return _parse_npm(path, text, raw, budget)
    if basename in _PIP_BASENAMES:
        return _parse_pip(path, text, raw, budget)
    if basename in _YARN_V1_BASENAMES:
        return _parse_yarn_v1(path, text, raw, budget)
    if basename in _YARN_YAML_BASENAMES:
        return _parse_yarn_yaml(path, text, raw, budget)
    if _is_cargo_path(path):
        return _parse_cargo(path, text, raw, budget)
    if basename in _MAVEN_BASENAMES:
        return _parse_maven(path, text, raw, budget)
    return _parse_python_project(
        path,
        text,
        raw,
        budget,
        skip_pyproject_uv=skip_pyproject_uv,
    )


def analyze_dependency_sources(
    *,
    components: Iterable[str],
    local_file_cache: Mapping[str, str],
    raw_file_cache: Mapping[str, bytes],
    artifact_inventory: Iterable[ArtifactRecord],
    budget: DependencyWorkBudget,
    executable_paths: frozenset[str] = frozenset(),
) -> DependencySourceAnalysis:
    """Analyze direct configs and disclose structurally executable unscanned surfaces."""
    if not isinstance(executable_paths, frozenset):
        raise ValueError("executable_paths must be an immutable set")
    normalized_executable_paths = frozenset(
        DependencySourceSpan(path=path, start_line=1, end_line=1).path for path in executable_paths
    )
    inventory_by_path: dict[str, list[ArtifactRecord]] = {}
    for record in artifact_inventory:
        path = record.get("path")
        if isinstance(path, str):
            inventory_by_path.setdefault(path, []).append(record)

    component_paths = {path for path in components if isinstance(path, str)}
    uv_directories = {
        path.rpartition("/")[0] for path in component_paths if _basename(path) == "uv.toml"
    }
    applicable_spans = tuple(
        _whole_file_span(
            path,
            raw_file_cache.get(path) if isinstance(raw_file_cache.get(path), bytes) else None,
        )
        for path in sorted(component_paths)
        if _is_recognized_path(path)
    )
    coverage_limitations: list[DependencySourceLimitation] = []
    for path in sorted(component_paths):
        raw = raw_file_cache.get(path)
        if not isinstance(raw, bytes):
            continue
        records = inventory_by_path.get(path, [])
        if len(records) != 1 or not _is_complete_text_record(records[0], len(raw)):
            continue
        try:
            decoded = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        cached = local_file_cache.get(path)
        if not isinstance(cached, str) or cached != decoded:
            continue
        for span in _executable_surface_ranges(
            path,
            decoded,
            raw,
            normalized_executable_paths,
        ):
            coverage_limitations.append(
                DependencySourceLimitation(
                    reason=DependencySourceLimitationReason.UNSCANNED_EXECUTABLE_CONTENT,
                    path=span.path,
                    start_line=span.start_line,
                    end_line=span.end_line,
                )
            )
    coverage_limitations = list(
        {
            (item.reason, item.path, item.start_line, item.end_line): item
            for item in coverage_limitations
        }.values()
    )
    required_ledger_rows = len(applicable_spans) + len(coverage_limitations)
    if exhaustion := budget.charge_ledger_events(required_ledger_rows):
        budget.claim_reserved_truncation_event()
        return DependencySourceAnalysis(
            limitations=tuple(coverage_limitations),
            applicable_spans=applicable_spans,
            ledger_exhaustion=exhaustion,
        )

    changes: list[SourceChange] = []
    limitations: list[DependencySourceLimitation] = list(coverage_limitations)
    inspected_spans: list[DependencySourceSpan] = []
    for path in sorted(component_paths):
        if not isinstance(path, str) or not _is_recognized_path(path):
            continue
        raw = raw_file_cache.get(path)
        safe_raw = raw if isinstance(raw, bytes) else None
        records = inventory_by_path.get(path, [])
        matched_record = records[0] if len(records) == 1 else None
        observed_size = max(len(safe_raw or b""), _inventory_size(matched_record))
        file_budget = budget.for_file(path)
        if exhaustion := file_budget.charge_physical_bytes(observed_size):
            limitations.append(_limitation(path, safe_raw, exhaustion))
            continue
        if (
            safe_raw is None
            or matched_record is None
            or not _is_complete_text_record(matched_record, len(safe_raw))
        ):
            limitations.append(_limitation(path, safe_raw))
            continue
        try:
            decoded = safe_raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            limitations.append(_limitation(path, safe_raw))
            continue
        cached = local_file_cache.get(path)
        if not isinstance(cached, str) or cached != decoded:
            limitations.append(_limitation(path, safe_raw))
            continue
        parsed = _parse_file(
            path,
            decoded,
            safe_raw,
            file_budget,
            skip_pyproject_uv=(
                _basename(path) == "pyproject.toml" and path.rpartition("/")[0] in uv_directories
            ),
        )
        changes.extend(parsed.changes)
        limitations.extend(parsed.limitations)
        if not parsed.limitations:
            inspected_spans.append(_whole_file_span(path, safe_raw))

    return DependencySourceAnalysis(
        findings=tuple(finding_from_source_change(change) for change in changes),
        limitations=tuple(limitations),
        applicable_spans=applicable_spans,
        inspected_spans=tuple(inspected_spans),
    )
