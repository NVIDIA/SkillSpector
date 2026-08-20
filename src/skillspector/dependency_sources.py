# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic dependency-source redirection analysis.

The analyzer models package-manager configuration locally. It does not contact
registries, infer ownership/reputation, or trust explanatory prose.
"""

from __future__ import annotations

import configparser
import re
import tomllib
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import PurePosixPath

from skillspector.models import Finding

_URL_RE = re.compile(
    r"(?:https?|ssh|git\+https?|git\+ssh|sparse\+https)://[^\s'\"<>]+",
    re.IGNORECASE,
)
_VARIABLE_RE = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)
_ASSIGNMENT_WORD_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)="
    r"(?P<value>'[^']*'|\"(?:\\.|[^\"\\])*\"|[^\s]*)"
)
_FUNCTION_DECLARATION_RE = re.compile(
    r"^\s*(?:function\s+(?P<bash>[A-Za-z_][A-Za-z0-9_]*)(?:\s*\(\s*\))?"
    r"|(?P<posix>[A-Za-z_][A-Za-z0-9_]*)\s*\(\s*\))(?P<rest>.*)$"
)
_SENSITIVE_QUERY_KEY = re.compile(r"(?:auth|credential|key|pass|secret|signature|token)", re.I)
_SHELL_SUFFIXES = frozenset({".sh", ".bash", ".zsh"})
_SHELL_SHEBANG_RE = re.compile(r"^#![^\n]*(?:^|/|\s)(?:ba|z|da|k)?sh(?:\s|$)", re.I)

Assignments = dict[str, list[tuple[int, str | None]]]

_CANONICAL_DESTINATIONS: dict[str, frozenset[str]] = {
    "npm": frozenset({"https://registry.npmjs.org/"}),
    "yarn": frozenset(
        {
            "https://registry.npmjs.org/",
            "https://registry.yarnpkg.com/",
        }
    ),
    "pip": frozenset({"https://pypi.org/simple/"}),
    "poetry": frozenset({"https://pypi.org/simple/"}),
    "maven": frozenset(
        {
            "https://repo.maven.apache.org/maven2/",
            "https://repo1.maven.org/maven2/",
        }
    ),
    "cargo": frozenset(
        {
            "sparse+https://index.crates.io/",
            "https://github.com/rust-lang/crates.io-index/",
        }
    ),
}


@dataclass(frozen=True)
class SourceChange:
    """One dependency-source trust-boundary change."""

    ecosystem: str
    operation: str
    surface: str
    scope: str | None
    destination: str
    file: str
    line: int
    matched_text: str


@dataclass(frozen=True)
class _HeredocRegion:
    target: str
    body: str
    start_line: int
    end_line: int
    expand_variables: bool
    complete: bool


_HEREDOC_TARGET = r'(?P<target>"[^"]+"|\'[^\']+\'|[^\s;]+)'
_HEREDOC_BARE_CHARACTERS = frozenset(
    "-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.,:+/@%"
)
_HEREDOC_WORD = (
    r"(?P<word>(?:\\[^\n]|'[^'\n]*'|\"(?:\\.|[^\"\\\n])*\"|[^\s;|&<>()'\"\\])+)(?=$|[\s;|&<>()])"
)
_HEREDOC_DELIMITER = _HEREDOC_WORD
_SHELL_HEREDOC_OPERATOR = re.compile(rf"<<(?P<strip_tabs>-?)(?!<)\s*{_HEREDOC_DELIMITER}")
_HEREDOC_HEADERS = (
    re.compile(
        rf"^\s*cat\b.*?(?<![<>])>(?!>)\s*{_HEREDOC_TARGET}\s*"
        rf"<<(?P<strip_tabs>-?)\s*{_HEREDOC_DELIMITER}"
    ),
    re.compile(
        rf"^\s*cat\b.*?<<(?P<strip_tabs>-?)\s*{_HEREDOC_DELIMITER}\s*"
        rf"(?<![<>])>(?!>)\s*{_HEREDOC_TARGET}"
    ),
)


def _strip_shell_comment(value: str) -> str:
    """Remove an unquoted shell comment without interpreting the command."""
    quote: str | None = None
    for index, character in enumerate(value):
        if character in {'"', "'"}:
            quote = None if quote == character else character if quote is None else quote
        elif character == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value.strip()


def _brace_delta(value: str) -> int:
    """Count shell grouping braces while ignoring quotes and parameter expansion."""
    quote: str | None = None
    escaped = False
    parameter_depth = 0
    delta = 0
    index = 0
    while index < len(value):
        character = value[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if character == "\\" and quote != "'":
            escaped = True
            index += 1
            continue
        if character in {'"', "'"}:
            quote = None if quote == character else character if quote is None else quote
            index += 1
            continue
        if quote is not None:
            index += 1
            continue
        if character == "#" and (index == 0 or value[index - 1].isspace()):
            break
        if character == "$" and value[index : index + 2] == "${":
            parameter_depth += 1
            index += 2
            continue
        if character == "}" and parameter_depth:
            parameter_depth -= 1
        elif character == "{":
            delta += 1
        elif character == "}":
            delta -= 1
        index += 1
    return delta


def _command_segment_body(segment: str, *, allow_case_arm: bool = False) -> str:
    """Remove bounded shell-control wrappers around one simple command."""
    candidate = segment.strip().removesuffix("}").strip()
    candidate = candidate.lstrip("{").lstrip()
    keyword = re.match(r"^(?:then|do|else)\b\s*(?P<rest>.*)$", candidate)
    if keyword:
        candidate = keyword.group("rest")
    if allow_case_arm:
        case_arm = re.match(r"^[^)]*\)\s*(?P<rest>.+)$", candidate)
        if case_arm:
            candidate = case_arm.group("rest")
    return candidate.strip()


def _leading_assignments(segment: str) -> tuple[list[tuple[str, str]], str]:
    """Return leading assignment words and the remaining simple command."""
    candidate = segment.strip()
    position = 0
    export = re.match(r"export\b\s*", candidate)
    if export and _ASSIGNMENT_WORD_RE.match(candidate, export.end()):
        position = export.end()

    assignments: list[tuple[str, str]] = []
    while match := _ASSIGNMENT_WORD_RE.match(candidate, position):
        assignments.append((match.group("name"), match.group("value")))
        position = match.end()
        if position >= len(candidate):
            break
        if not candidate[position].isspace():
            return [], candidate
        position += len(candidate[position:]) - len(candidate[position:].lstrip())
    remainder = candidate[position:].strip()
    if (
        export
        and assignments
        and all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in remainder.split())
    ):
        remainder = ""
    return assignments, remainder


def _normalize_heredoc_word(word: str) -> tuple[str, bool] | None:
    """Apply bounded shell quote removal to one static heredoc word."""
    delimiter: list[str] = []
    quoted = False
    index = 0
    while index < len(word):
        character = word[index]
        if character == "'":
            end = word.find("'", index + 1)
            if end < 0:
                return None
            delimiter.append(word[index + 1 : end])
            quoted = True
            index = end + 1
            continue
        if character == '"':
            quoted = True
            index += 1
            while index < len(word) and word[index] != '"':
                if word[index] == "\\":
                    if index + 1 >= len(word):
                        return None
                    escaped = word[index + 1]
                    if escaped in {"$", "`", '"', "\\"}:
                        delimiter.append(escaped)
                    else:
                        delimiter.extend(("\\", escaped))
                    index += 2
                else:
                    delimiter.append(word[index])
                    index += 1
            if index >= len(word):
                return None
            index += 1
            continue
        if character == "\\":
            if index + 1 >= len(word):
                return None
            delimiter.append(word[index + 1])
            quoted = True
            index += 2
            continue
        if character not in _HEREDOC_BARE_CHARACTERS:
            return None
        delimiter.append(character)
        index += 1
    normalized = "".join(delimiter)
    return (normalized, quoted) if normalized else None


def _function_context(content: str, data_lines: set[int]) -> tuple[set[int], dict[str, set[str]]]:
    """Locate function definitions and variables they may assign, without executing them."""
    lines = content.splitlines()
    function_lines: set[int] = set()
    assigned_by_function: dict[str, set[str]] = {}
    index = 0
    while index < len(lines):
        line_number = index + 1
        if line_number in data_lines:
            index += 1
            continue
        declaration = _FUNCTION_DECLARATION_RE.match(_strip_shell_comment(lines[index]))
        if not declaration:
            index += 1
            continue
        name = declaration.group("bash") or declaration.group("posix") or ""
        rest = declaration.group("rest")
        opening_index = index if rest.lstrip().startswith("{") else None
        if opening_index is None:
            candidate = index + 1
            while candidate < len(lines) and not _strip_shell_comment(lines[candidate]).strip():
                candidate += 1
            if candidate >= len(lines) or not _strip_shell_comment(
                lines[candidate]
            ).lstrip().startswith("{"):
                index += 1
                continue
            opening_index = candidate

        function_lines.add(line_number)
        depth = 0
        cursor = opening_index
        assigned_names: set[str] = set()
        while cursor < len(lines):
            function_lines.add(cursor + 1)
            fragment = rest if cursor == index else lines[cursor]
            depth += _brace_delta(fragment)
            for _, segment in _shell_parts(fragment):
                assignment_words, remainder = _leading_assignments(
                    _command_segment_body(segment, allow_case_arm=True)
                )
                if not remainder:
                    assigned_names.update(name for name, _ in assignment_words)
            cursor += 1
            if depth <= 0:
                break
        assigned_by_function.setdefault(name, set()).update(assigned_names)
        index = max(index + 1, cursor)
    return function_lines, assigned_by_function


def _literal_assignments(content: str) -> Assignments:
    """Collect definite top-level assignments without evaluating shell syntax.

    Heredoc data and function bodies are inert at their physical location, so
    their assignment-shaped text is ignored. Assignments in conditional or
    iterative control flow are recorded as ambiguous so they cannot silently
    make an earlier possible destination appear canonical.
    """
    assignments: Assignments = {}
    heredoc_data_lines = _heredoc_data_lines(content)
    function_lines, assigned_by_function = _function_context(content, heredoc_data_lines)
    control_depth = 0
    for line_number, line in enumerate(content.splitlines(), 1):
        if line_number in heredoc_data_lines or line_number in function_lines:
            continue
        for separator, segment in _shell_parts(line):
            stripped = _command_segment_body(segment, allow_case_arm=bool(control_depth))
            if re.match(r"^(?:fi|done|esac)\b", stripped):
                control_depth = max(0, control_depth - 1)
                continue
            control = re.match(
                r"^(?P<keyword>if|elif|case|for|while|until|select)\b\s*(?P<rest>.*)$",
                stripped,
            )
            if control:
                keyword = control.group("keyword")
                if keyword != "elif":
                    control_depth += 1
                if keyword == "case":
                    case_arm = re.match(r"^[^)]*\)\s*(?P<rest>.+)$", control.group("rest"))
                    if not case_arm:
                        continue
                    stripped = case_arm.group("rest")
                elif keyword in {"if", "elif", "while", "until"}:
                    stripped = control.group("rest")
                else:
                    continue

            assignment_words, call_candidate = _leading_assignments(stripped)
            if assignment_words and not call_candidate:
                for name, raw_value in assignment_words:
                    value = _strip_shell_comment(raw_value).strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                        value = value[1:-1]
                    resolved_value: str | None = value
                    if (
                        control_depth
                        or separator in {"&&", "||", "|", "|&"}
                        or not value
                        or "$" in value
                        or "`" in value
                    ):
                        resolved_value = None
                    assignments.setdefault(name, []).append((line_number, resolved_value))
                continue

            call = re.match(r"^(?:command\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b", call_candidate)
            if call and call.group("name") in assigned_by_function:
                for name in assigned_by_function[call.group("name")]:
                    assignments.setdefault(name, []).append((line_number, None))
    return assignments


def _resolve_value(value: str, assignments: Assignments, use_line: int) -> tuple[str, bool]:
    """Resolve simple variables from the latest literal assignment before use."""
    resolved = _strip_shell_comment(value).strip().strip(";,)")
    if len(resolved) >= 2 and resolved[0] == resolved[-1] and resolved[0] in {'"', "'"}:
        resolved = resolved[1:-1]

    def replacement(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("plain") or ""
        prior = [assigned for line, assigned in assignments.get(name, []) if line < use_line]
        return prior[-1] if prior and prior[-1] is not None else match.group(0)

    resolved = _VARIABLE_RE.sub(replacement, resolved).strip().strip("\"'")
    dynamic = bool("$" in resolved or "`" in resolved)
    return ("unresolved" if dynamic or not resolved else resolved, not dynamic and bool(resolved))


def _normalize_destination(destination: str) -> str:
    """Normalize a URL for comparison with built-in canonical endpoints."""
    if destination == "unresolved":
        return destination
    try:
        parsed = urllib.parse.urlsplit(destination)
    except ValueError:
        return destination.rstrip("/") + "/"
    if not parsed.scheme or not parsed.hostname:
        return destination.rstrip("/") + "/"
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname.lower().rstrip(".")
    try:
        port = parsed.port
    except ValueError:
        return destination.rstrip("/") + "/"
    if port and not (
        (scheme in {"https", "sparse+https", "git+https"} and port == 443)
        or (scheme == "http" and port == 80)
    ):
        hostname = f"{hostname}:{port}"
    path = re.sub(r"/+", "/", parsed.path or "/")
    if not path.endswith("/"):
        path += "/"
    return urllib.parse.urlunsplit((scheme, hostname, path, "", ""))


def redact_url(destination: str) -> str:
    """Remove URL credentials and sensitive query values from report evidence."""
    if destination == "unresolved":
        return destination
    try:
        parsed = urllib.parse.urlsplit(destination)
    except ValueError:
        return "<redacted-url>"
    if not parsed.scheme or not parsed.hostname:
        if "@" in destination or _SENSITIVE_QUERY_KEY.search(destination.partition("?")[2]):
            return "<redacted-url>"
        return destination
    hostname = parsed.hostname
    try:
        port = parsed.port
    except ValueError:
        return "<redacted-url>"
    if port:
        hostname = f"{hostname}:{port}"
    if parsed.username is not None or parsed.password is not None:
        hostname = f"***@{hostname}"
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    safe_query = [
        (key, "***" if _SENSITIVE_QUERY_KEY.search(key) else value) for key, value in query
    ]
    return urllib.parse.urlunsplit(
        (parsed.scheme, hostname, parsed.path, urllib.parse.urlencode(safe_query), "")
    )


def redact_text(text: str) -> str:
    """Redact every URL-like token in source evidence."""

    def replacement(match: re.Match[str]) -> str:
        raw = match.group(0)
        suffix = ""
        while raw and raw[-1] in ".,;)]}":
            suffix = raw[-1] + suffix
            raw = raw[:-1]
        return redact_url(raw) + suffix

    return _URL_RE.sub(replacement, text)


def _is_canonical(ecosystem: str, destination: str) -> bool:
    normalized = _normalize_destination(destination)
    return normalized in _CANONICAL_DESTINATIONS[ecosystem]


def _line_for(content: str, needle: str, default: int = 1) -> int:
    for index, line in enumerate(content.splitlines(), 1):
        if needle and needle in line:
            return index
    return default


def _add_change(
    changes: list[SourceChange],
    *,
    ecosystem: str,
    operation: str,
    surface: str,
    scope: str | None,
    raw_destination: str,
    file: str,
    line: int,
    matched_text: str,
    assignments: Assignments,
) -> None:
    destination, resolved = _resolve_value(raw_destination, assignments, line)
    if resolved and _is_canonical(ecosystem, destination):
        return
    changes.append(
        SourceChange(
            ecosystem=ecosystem,
            operation=operation,
            surface=surface,
            scope=scope,
            destination=destination,
            file=file,
            line=line,
            matched_text=matched_text,
        )
    )


def _parse_npmrc(
    content: str, file: str, start_line: int, assignments: Assignments
) -> list[SourceChange]:
    changes: list[SourceChange] = []
    for offset, line in enumerate(content.splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        match = re.match(r"(?P<key>(?:@[\w.-]+:)?registry)\s*=\s*(?P<value>.+)$", stripped, re.I)
        if not match:
            continue
        scope = match.group("key").split(":", 1)[0] if match.group("key").startswith("@") else None
        _add_change(
            changes,
            ecosystem="npm",
            operation="replace",
            surface=".npmrc",
            scope=scope,
            raw_destination=match.group("value"),
            file=file,
            line=start_line + offset,
            matched_text=line,
            assignments=assignments,
        )
    return changes


def _parse_yarnrc(
    content: str, file: str, start_line: int, assignments: Assignments
) -> list[SourceChange]:
    changes: list[SourceChange] = []
    current_scope: str | None = None
    scope_indent = -1
    for offset, line in enumerate(content.splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        indent = len(line) - len(line.lstrip())
        scope_match = re.match(r"(?P<scope>[\w.-]+):\s*$", stripped)
        if scope_match and "npmScopes" not in stripped and indent > 0:
            current_scope = scope_match.group("scope")
            scope_indent = indent
            continue
        if current_scope and indent <= scope_indent:
            current_scope = None
        match = re.match(
            r"(?P<key>registry|npmRegistryServer)\s*(?::|\s)\s*(?P<value>.+)$",
            stripped,
            re.I,
        )
        if not match:
            continue
        _add_change(
            changes,
            ecosystem="yarn",
            operation="replace",
            surface=".yarnrc.yml" if file.lower().endswith((".yml", ".yaml")) else ".yarnrc",
            scope=current_scope,
            raw_destination=match.group("value"),
            file=file,
            line=start_line + offset,
            matched_text=line,
            assignments=assignments,
        )
    return changes


def _parse_pip_config(
    content: str, file: str, start_line: int, assignments: Assignments
) -> list[SourceChange]:
    changes: list[SourceChange] = []
    section: str | None = None
    for offset, line in enumerate(content.splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
            continue
        match = re.match(r"(?P<key>index-url|extra-index-url)\s*=\s*(?P<value>.+)$", stripped, re.I)
        if not match:
            continue
        key = match.group("key").lower()
        _add_change(
            changes,
            ecosystem="pip",
            operation="add" if key == "extra-index-url" else "replace",
            surface="pip config",
            scope=section,
            raw_destination=match.group("value"),
            file=file,
            line=start_line + offset,
            matched_text=line,
            assignments=assignments,
        )
    return changes


def _parse_poetry(content: str, file: str, assignments: Assignments) -> list[SourceChange]:
    changes: list[SourceChange] = []
    try:
        parsed = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return changes
    poetry = parsed.get("tool", {}).get("poetry", {})
    if not isinstance(poetry, dict):
        return changes
    sources = poetry.get("source", [])
    if isinstance(sources, dict):
        sources = [sources]
    if not isinstance(sources, list):
        return changes
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("url"), str):
            continue
        destination = str(source["url"])
        _add_change(
            changes,
            ecosystem="poetry",
            operation="add",
            surface="pyproject.toml source",
            scope=str(source.get("name")) if source.get("name") is not None else None,
            raw_destination=destination,
            file=file,
            line=_line_for(content, destination),
            matched_text=next(
                (line for line in content.splitlines() if destination in line), destination
            ),
            assignments=assignments,
        )
    return changes


def _parse_maven(content: str, file: str, assignments: Assignments) -> list[SourceChange]:
    changes: list[SourceChange] = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return changes

    def local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    for element in root.iter():
        if local_name(element.tag) not in {"mirror", "repository", "pluginRepository"}:
            continue
        values = {local_name(child.tag): (child.text or "").strip() for child in element}
        destination = values.get("url")
        if not destination:
            continue
        is_mirror = local_name(element.tag) == "mirror"
        _add_change(
            changes,
            ecosystem="maven",
            operation="replace" if is_mirror else "add",
            surface="settings.xml mirror" if is_mirror else "Maven repository",
            scope=values.get("mirrorOf") or values.get("id"),
            raw_destination=destination,
            file=file,
            line=_line_for(content, destination),
            matched_text=next(
                (line for line in content.splitlines() if destination in line), destination
            ),
            assignments=assignments,
        )
    return changes


def _parse_cargo(content: str, file: str, assignments: Assignments) -> list[SourceChange]:
    changes: list[SourceChange] = []
    try:
        parsed = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return changes
    sources = parsed.get("source", {})
    if isinstance(sources, dict):
        for name, source in sources.items():
            if not isinstance(source, dict):
                continue
            replacement = source.get("replace-with")
            if isinstance(replacement, str):
                target = sources.get(replacement, {})
                destination = target.get("registry") if isinstance(target, dict) else None
                raw_destination = str(destination) if destination else "unresolved"
                _add_change(
                    changes,
                    ecosystem="cargo",
                    operation="replace",
                    surface="Cargo source.replace-with",
                    scope=str(name),
                    raw_destination=raw_destination,
                    file=file,
                    line=_line_for(content, "replace-with"),
                    matched_text=next(
                        (line for line in content.splitlines() if "replace-with" in line),
                        "replace-with",
                    ),
                    assignments=assignments,
                )
            elif isinstance(source.get("registry"), str):
                destination = str(source["registry"])
                _add_change(
                    changes,
                    ecosystem="cargo",
                    operation="add" if name != "crates-io" else "replace",
                    surface="Cargo source registry",
                    scope=str(name),
                    raw_destination=destination,
                    file=file,
                    line=_line_for(content, destination),
                    matched_text=next(
                        (line for line in content.splitlines() if destination in line), destination
                    ),
                    assignments=assignments,
                )
    registries = parsed.get("registries", {})
    if isinstance(registries, dict):
        for name, registry in registries.items():
            if not isinstance(registry, dict) or not isinstance(registry.get("index"), str):
                continue
            destination = str(registry["index"])
            _add_change(
                changes,
                ecosystem="cargo",
                operation="add",
                surface="Cargo registry index",
                scope=str(name),
                raw_destination=destination,
                file=file,
                line=_line_for(content, destination),
                matched_text=next(
                    (line for line in content.splitlines() if destination in line), destination
                ),
                assignments=assignments,
            )
    return changes


def _heredocs(content: str) -> list[_HeredocRegion]:
    """Return bounded, linearly parsed generated-configuration heredocs."""
    lines = content.splitlines()
    regions: list[_HeredocRegion] = []
    index = 0
    while index < len(lines):
        match = next(
            (
                candidate
                for pattern in _HEREDOC_HEADERS
                if (candidate := pattern.search(lines[index])) is not None
            ),
            None,
        )
        if not match:
            index += 1
            continue
        normalized_word = _normalize_heredoc_word(match.group("word"))
        if normalized_word is None:
            index += 1
            continue
        delimiter, quoted = normalized_word
        strip_tabs = match.group("strip_tabs") == "-"
        end = index + 1
        while end < len(lines):
            terminator = lines[end].lstrip("\t") if strip_tabs else lines[end]
            if terminator == delimiter:
                break
            end += 1
        complete = end < len(lines)
        body_lines = lines[index + 1 : end]
        if strip_tabs:
            body_lines = [line.lstrip("\t") for line in body_lines]
        regions.append(
            _HeredocRegion(
                target=match.group("target").strip("'\""),
                body="\n".join(body_lines),
                start_line=index + 2,
                end_line=end + 1 if complete else len(lines),
                expand_variables=not quoted,
                complete=complete,
            )
        )
        if not complete:
            # An unmatched heredoc consumes the remaining shell input. Stopping
            # here both reflects that ambiguity and prevents repeated O(n) scans.
            break
        index = end + 1
    return regions


def _shell_heredoc_specs(line: str) -> list[tuple[str, bool]]:
    """Return unquoted heredoc delimiters declared by one shell command line."""
    specs: list[tuple[str, bool]] = []
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(line):
        character = line[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if character == "\\" and quote != "'":
            escaped = True
            index += 1
            continue
        if character in {'"', "'"}:
            quote = None if quote == character else character if quote is None else quote
            index += 1
            continue
        if quote is None and character == "#" and (index == 0 or line[index - 1].isspace()):
            break
        if (
            quote is None
            and line[index : index + 2] == "<<"
            and (index == 0 or line[index - 1] != "<")
        ):
            match = _SHELL_HEREDOC_OPERATOR.match(line, index)
            if match:
                normalized_word = _normalize_heredoc_word(match.group("word"))
                if normalized_word is not None:
                    delimiter, _ = normalized_word
                    specs.append((delimiter, match.group("strip_tabs") == "-"))
                index = match.end()
                continue
        index += 1
    return specs


def _heredoc_data_lines(content: str) -> set[int]:
    """Return all shell heredoc body and terminator lines in one bounded pass."""
    lines = content.splitlines()
    data_lines: set[int] = set()
    index = 0
    while index < len(lines):
        specs = _shell_heredoc_specs(lines[index])
        if not specs:
            index += 1
            continue
        body_index = index + 1
        for delimiter, strip_tabs in specs:
            end = body_index
            while end < len(lines):
                terminator = lines[end].lstrip("\t") if strip_tabs else lines[end]
                if terminator == delimiter:
                    break
                end += 1
            data_lines.update(range(body_index + 1, min(end + 2, len(lines) + 1)))
            if end >= len(lines):
                return data_lines
            body_index = end + 1
        index = body_index
    return data_lines


def _parse_generated_configs(
    content: str, file: str, assignments: Assignments
) -> list[SourceChange]:
    changes: list[SourceChange] = []
    heredoc_data_lines = _heredoc_data_lines(content)
    for region in _heredocs(content):
        if not region.complete or region.start_line - 1 in heredoc_data_lines:
            continue
        lower = region.target.lower()
        region_assignments = assignments if region.expand_variables else {}
        if lower.endswith(".npmrc"):
            changes.extend(_parse_npmrc(region.body, file, region.start_line, region_assignments))
        elif lower.endswith(".yarnrc") or lower.endswith((".yarnrc.yml", ".yarnrc.yaml")):
            changes.extend(_parse_yarnrc(region.body, file, region.start_line, region_assignments))
        elif lower.endswith(("pip.conf", "pip.ini")):
            changes.extend(
                _parse_pip_config(region.body, file, region.start_line, region_assignments)
            )
        elif lower.endswith(("settings.xml", "pom.xml")):
            generated = _parse_maven(region.body, file, region_assignments)
            changes.extend(
                SourceChange(
                    ecosystem=change.ecosystem,
                    operation=change.operation,
                    surface=f"generated {change.surface}",
                    scope=change.scope,
                    destination=change.destination,
                    file=change.file,
                    line=region.start_line + change.line - 1,
                    matched_text=change.matched_text,
                )
                for change in generated
            )
        elif lower.endswith("pyproject.toml"):
            generated = _parse_poetry(region.body, file, region_assignments)
            changes.extend(
                SourceChange(
                    ecosystem=change.ecosystem,
                    operation=change.operation,
                    surface=f"generated {change.surface}",
                    scope=change.scope,
                    destination=change.destination,
                    file=change.file,
                    line=region.start_line + change.line - 1,
                    matched_text=change.matched_text,
                )
                for change in generated
            )
        elif ".cargo/" in lower and lower.endswith(("/config", "/config.toml")):
            generated = _parse_cargo(region.body, file, region_assignments)
            changes.extend(
                SourceChange(
                    ecosystem=change.ecosystem,
                    operation=change.operation,
                    surface=f"generated {change.surface}",
                    scope=change.scope,
                    destination=change.destination,
                    file=change.file,
                    line=region.start_line + change.line - 1,
                    matched_text=change.matched_text,
                )
                for change in generated
            )
    return changes


def _shell_parts(line: str) -> list[tuple[str | None, str]]:
    """Split shell command lists while retaining the preceding control operator."""
    parts: list[tuple[str | None, str]] = []
    current: list[str] = []
    separator: str | None = None
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(line):
        character = line[index]
        if escaped:
            current.append(character)
            escaped = False
            index += 1
            continue
        if character == "\\" and quote != "'":
            current.append(character)
            escaped = True
            index += 1
            continue
        if character in {'"', "'"}:
            quote = None if quote == character else character if quote is None else quote
            current.append(character)
            index += 1
            continue
        if quote is None and character == "#" and (not current or current[-1].isspace()):
            break
        pair = line[index : index + 2]
        delimiter = pair if pair in {"&&", "||", "|&"} else character
        if quote is None and (character in {";", "|"} or pair in {"&&", "||", "|&"}):
            segment = "".join(current).strip()
            if segment:
                parts.append((separator, segment))
            current = []
            separator = delimiter
            index += 2 if pair in {"&&", "||", "|&"} else 1
            continue
        current.append(character)
        index += 1
    segment = "".join(current).strip()
    if segment:
        parts.append((separator, segment))
    return parts


def _shell_segments(line: str) -> list[str]:
    """Split executable shell command lists without evaluating shell syntax."""
    return [segment for _, segment in _shell_parts(line)]


def _parse_commands(content: str, file: str, assignments: Assignments) -> list[SourceChange]:
    changes: list[SourceChange] = []
    heredoc_data_lines = _heredoc_data_lines(content)
    command_prefix = r"^\s*(?:[$>]\s+)?(?:(?:command|sudo)\s+)?"
    patterns: tuple[tuple[str, str, str, str, re.Pattern[str]], ...] = (
        (
            "npm",
            "replace",
            "npm config set",
            "scope",
            re.compile(
                command_prefix
                + r"npm\s+config\s+set\s+(?P<scope>@[\w.-]+:)?registry\s+(?P<dest>\S+)",
                re.I,
            ),
        ),
        (
            "yarn",
            "replace",
            "yarn config set",
            "scope",
            re.compile(
                command_prefix
                + r"yarn\s+config\s+set\s+(?:registry|npmRegistryServer)\s+(?P<dest>\S+)",
                re.I,
            ),
        ),
        (
            "pip",
            "replace",
            "pip --index-url",
            "none",
            re.compile(
                command_prefix
                + r"(?:python(?:3)?\s+-m\s+)?pip(?:3)?\b[^\n]*?"
                + r"(?:--index-url|-i)(?:=|\s+)(?P<dest>\S+)",
                re.I,
            ),
        ),
        (
            "pip",
            "add",
            "pip --extra-index-url",
            "none",
            re.compile(
                command_prefix
                + r"(?:python(?:3)?\s+-m\s+)?pip(?:3)?\b[^\n]*?"
                + r"--extra-index-url(?:=|\s+)(?P<dest>\S+)",
                re.I,
            ),
        ),
        (
            "pip",
            "replace",
            "pip config set",
            "none",
            re.compile(
                command_prefix
                + r"pip(?:3)?\s+config\s+set\s+(?:global\.)?index-url\s+(?P<dest>\S+)",
                re.I,
            ),
        ),
        (
            "pip",
            "add",
            "pip config set",
            "none",
            re.compile(
                command_prefix
                + r"pip(?:3)?\s+config\s+set\s+(?:global\.)?extra-index-url\s+(?P<dest>\S+)",
                re.I,
            ),
        ),
        (
            "poetry",
            "add",
            "poetry source add",
            "poetry",
            re.compile(
                command_prefix
                + r"poetry\s+source\s+add(?:\s+--\S+)*\s+"
                + r"(?P<scope>[\w.-]+)\s+(?P<dest>\S+)",
                re.I,
            ),
        ),
        (
            "poetry",
            "add",
            "poetry config repositories",
            "poetry",
            re.compile(
                command_prefix
                + r"poetry\s+config\s+repositories\."
                + r"(?P<scope>[\w.-]+)\s+(?P<dest>\S+)",
                re.I,
            ),
        ),
        (
            "maven",
            "replace",
            "Maven CLI repository",
            "none",
            re.compile(
                command_prefix + r"mvn\b[^\n]*?-Dmaven\.repo\.remote=(?P<dest>\S+)",
                re.I,
            ),
        ),
    )
    for line_number, line in enumerate(content.splitlines(), 1):
        if line_number in heredoc_data_lines:
            continue
        for segment in _shell_segments(line):
            command_candidate = _command_segment_body(segment, allow_case_arm=True)
            command_candidate = re.sub(r"^(?:[$>]\s+)", "", command_candidate)
            _, command_candidate = _leading_assignments(command_candidate)
            wrapper = re.match(r"^(?:command|sudo)\b\s*(?P<rest>.*)$", command_candidate)
            if wrapper:
                _, command_candidate = _leading_assignments(wrapper.group("rest"))
            for ecosystem, operation, surface, scope_mode, pattern in patterns:
                match = pattern.search(command_candidate)
                if not match:
                    continue
                scope = match.groupdict().get("scope") if scope_mode != "none" else None
                if scope:
                    scope = scope.rstrip(":")
                _add_change(
                    changes,
                    ecosystem=ecosystem,
                    operation=operation,
                    surface=surface,
                    scope=scope,
                    raw_destination=match.group("dest"),
                    file=file,
                    line=line_number,
                    matched_text=line,
                    assignments=assignments,
                )

            env_match = re.match(
                r"\s*(?:export\s+)?(?P<name>NPM_CONFIG_REGISTRY|PIP_INDEX_URL|PIP_EXTRA_INDEX_URL|CARGO_REGISTRIES_[A-Za-z0-9_]+_INDEX)\s*=\s*(?P<dest>.+)$",
                _command_segment_body(segment, allow_case_arm=True),
                re.I,
            )
            if env_match:
                name = env_match.group("name").upper()
                if name == "NPM_CONFIG_REGISTRY":
                    ecosystem, operation, scope = "npm", "replace", None
                elif name == "PIP_INDEX_URL":
                    ecosystem, operation, scope = "pip", "replace", None
                elif name == "PIP_EXTRA_INDEX_URL":
                    ecosystem, operation, scope = "pip", "add", None
                else:
                    ecosystem, operation = "cargo", "add"
                    scope = name.removeprefix("CARGO_REGISTRIES_").removesuffix("_INDEX").lower()
                _add_change(
                    changes,
                    ecosystem=ecosystem,
                    operation=operation,
                    surface="environment variable",
                    scope=scope,
                    raw_destination=env_match.group("dest"),
                    file=file,
                    line=line_number,
                    matched_text=line,
                    assignments=assignments,
                )
    return changes


def _markdown_shell_content(content: str) -> str:
    """Keep actionable shell fences while blanking prose and preserving lines."""
    output: list[str] = []
    in_shell = False
    for line in content.splitlines():
        fence = re.match(r"^\s*```\s*([\w+-]*)", line)
        if fence:
            language = fence.group(1).lower()
            if in_shell:
                in_shell = False
            else:
                in_shell = language in {"bash", "sh", "shell", "zsh", "console"}
            output.append("")
        else:
            output.append(line if in_shell else "")
    return "\n".join(output)


def _changes_for_file(content: str, file: str, *, executable: bool = False) -> list[SourceChange]:
    normalized = file.replace("\\", "/")
    lower = normalized.lower()
    name = PurePosixPath(normalized).name.lower()
    assignments = _literal_assignments(content)
    changes: list[SourceChange] = []
    if name == ".npmrc":
        changes.extend(_parse_npmrc(content, file, 1, assignments))
    elif name == ".yarnrc":
        changes.extend(_parse_yarnrc(content, file, 1, assignments))
    elif name in {".yarnrc.yml", ".yarnrc.yaml"}:
        changes.extend(_parse_yarnrc(content, file, 1, assignments))
    elif name in {"pip.conf", "pip.ini"}:
        # ConfigParser validates basic INI structure without executing interpolation.
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(content)
        except configparser.Error:
            pass
        changes.extend(_parse_pip_config(content, file, 1, assignments))
    elif name == "pyproject.toml":
        changes.extend(_parse_poetry(content, file, assignments))
    elif name in {"settings.xml", "pom.xml"}:
        changes.extend(_parse_maven(content, file, assignments))
    elif name in {"config", "config.toml"} and "/.cargo/" in f"/{lower}":
        changes.extend(_parse_cargo(content, file, assignments))

    suffix = PurePosixPath(normalized).suffix.lower()
    is_script = suffix in _SHELL_SUFFIXES or (
        not suffix and executable and bool(_SHELL_SHEBANG_RE.search(content[:256]))
    )
    actionable = _markdown_shell_content(content) if name in {"skill.md", "readme.md"} else content
    if is_script or actionable != content:
        command_assignments = _literal_assignments(actionable) or assignments
        changes.extend(_parse_generated_configs(actionable, file, command_assignments))
        changes.extend(_parse_commands(actionable, file, command_assignments))
    return changes


def _finding(change: SourceChange, *, local_only: bool) -> Finding:
    destination = redact_url(change.destination)
    matched_text = redact_text(change.matched_text)
    resolved = change.destination != "unresolved"
    scope = change.scope or "global"
    tags = ["supply-chain", "dependency-source"]
    evidence: dict[str, object] = {
        "ecosystem": change.ecosystem,
        "operation": change.operation,
        "surface": change.surface,
        "scope": scope,
        "destination": destination,
        "destination_status": "resolved" if resolved else "unresolved",
    }
    if local_only:
        tags.append("local-only")
        evidence["local_only"] = True
    return Finding(
        rule_id="SC10",
        message=(
            f"{change.ecosystem} dependency source {change.operation} changes the "
            f"trust boundary to {destination}."
        ),
        severity="HIGH",
        confidence=1.0,
        file=change.file,
        start_line=change.line,
        category="Supply Chain",
        pattern="Dependency Source Redirection",
        finding=matched_text[:200],
        explanation=(
            "Dependency resolution is redirected away from a canonical default, adds another "
            "source, or uses a destination that cannot be resolved statically."
        ),
        remediation=(
            "Review the destination and configuration scope as a dependency trust-boundary "
            "change, and keep the intended source explicit and reviewable."
        ),
        tags=tags,
        context=matched_text,
        matched_text=matched_text[:200],
        evidence=evidence,
    )


def analyze_dependency_sources(
    components: list[str],
    file_cache: dict[str, str],
    component_metadata: list[dict[str, object]] | None = None,
) -> list[Finding]:
    """Return deterministic HIGH findings for dependency-source trust changes."""
    local_only_paths = {
        str(metadata.get("path", ""))
        for metadata in component_metadata or []
        if metadata.get("local_only") is True
    }
    executable_paths = {
        str(metadata.get("path", ""))
        for metadata in component_metadata or []
        if metadata.get("executable") is True
    }
    changes: list[SourceChange] = []
    for file in components:
        content = file_cache.get(file)
        if content is None or "\x00" in content[:8192]:
            continue
        changes.extend(_changes_for_file(content, file, executable=file in executable_paths))

    findings: list[Finding] = []
    seen: set[tuple[object, ...]] = set()
    for change in changes:
        key = (
            change.ecosystem,
            change.operation,
            change.surface,
            change.scope,
            change.destination,
            change.file,
            change.line,
        )
        if key in seen:
            continue
        seen.add(key)
        findings.append(_finding(change, local_only=change.file in local_only_paths))
    return findings
