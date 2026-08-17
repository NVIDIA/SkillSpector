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

_URL_RE = re.compile(r"(?:sparse\+)?(?:https?|git\+https?)://[^\s'\"<>]+", re.IGNORECASE)
_VARIABLE_RE = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)
_ASSIGNMENT_RE = re.compile(
    r"^\s*(?:export\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.+?)\s*$"
)
_SENSITIVE_QUERY_KEY = re.compile(r"(?:auth|credential|key|pass|secret|signature|token)", re.I)
_EXECUTABLE_SUFFIXES = frozenset({".sh", ".bash", ".zsh", ".py", ".js", ".ts", ".rb"})

_CANONICAL_DESTINATIONS: dict[str, frozenset[str]] = {
    "npm": frozenset({"https://registry.npmjs.org/"}),
    "yarn": frozenset({"https://registry.npmjs.org/"}),
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


def _strip_shell_comment(value: str) -> str:
    """Remove an unquoted shell comment without interpreting the command."""
    quote: str | None = None
    for index, character in enumerate(value):
        if character in {'"', "'"}:
            quote = None if quote == character else character if quote is None else quote
        elif character == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value.strip()


def _literal_assignments(content: str) -> dict[str, str]:
    """Collect simple literal local assignments; never evaluate shell syntax."""
    assignments: dict[str, str] = {}
    for line in content.splitlines():
        match = _ASSIGNMENT_RE.match(line)
        if not match:
            continue
        value = _strip_shell_comment(match.group("value")).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not value or any(token in value for token in ("`", "$(")):
            continue
        if _VARIABLE_RE.search(value):
            continue
        assignments[match.group("name")] = value
    return assignments


def _resolve_value(value: str, assignments: dict[str, str]) -> tuple[str, bool]:
    """Resolve simple variable references from the same file."""
    resolved = _strip_shell_comment(value).strip().strip(";,)")
    if len(resolved) >= 2 and resolved[0] == resolved[-1] and resolved[0] in {'"', "'"}:
        resolved = resolved[1:-1]

    def replacement(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("plain") or ""
        return assignments.get(name, match.group(0))

    resolved = _VARIABLE_RE.sub(replacement, resolved).strip().strip("\"'")
    dynamic = bool(_VARIABLE_RE.search(resolved) or "$(" in resolved or "`" in resolved)
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
    assignments: dict[str, str],
) -> None:
    destination, resolved = _resolve_value(raw_destination, assignments)
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
    content: str, file: str, start_line: int, assignments: dict[str, str]
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
    content: str, file: str, start_line: int, assignments: dict[str, str]
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
    content: str, file: str, start_line: int, assignments: dict[str, str]
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


def _parse_poetry(content: str, file: str, assignments: dict[str, str]) -> list[SourceChange]:
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


def _parse_maven(content: str, file: str, assignments: dict[str, str]) -> list[SourceChange]:
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


def _parse_cargo(content: str, file: str, assignments: dict[str, str]) -> list[SourceChange]:
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


def _heredocs(content: str) -> list[tuple[str, str, int]]:
    """Return generated target, body, and first body line for simple heredocs."""
    lines = content.splitlines()
    regions: list[tuple[str, str, int]] = []
    header = re.compile(
        r">\s*(?P<target>\"[^\"]+\"|'[^']+'|\S+)\s*<<-?\s*['\"]?(?P<delimiter>[A-Za-z_][A-Za-z0-9_]*)"
    )
    index = 0
    while index < len(lines):
        match = header.search(lines[index])
        if not match:
            index += 1
            continue
        delimiter = match.group("delimiter")
        end = index + 1
        while end < len(lines) and lines[end].strip() != delimiter:
            end += 1
        if end >= len(lines):
            index += 1
            continue
        regions.append(
            (match.group("target").strip("'\""), "\n".join(lines[index + 1 : end]), index + 2)
        )
        index = end + 1
    return regions


def _parse_generated_configs(
    content: str, file: str, assignments: dict[str, str]
) -> list[SourceChange]:
    changes: list[SourceChange] = []
    for target, body, start_line in _heredocs(content):
        lower = target.lower()
        if lower.endswith(".npmrc"):
            changes.extend(_parse_npmrc(body, file, start_line, assignments))
        elif lower.endswith(".yarnrc") or lower.endswith((".yarnrc.yml", ".yarnrc.yaml")):
            changes.extend(_parse_yarnrc(body, file, start_line, assignments))
        elif lower.endswith(("pip.conf", "pip.ini")):
            changes.extend(_parse_pip_config(body, file, start_line, assignments))
        elif lower.endswith(("settings.xml", "pom.xml")):
            generated = _parse_maven(body, file, assignments)
            changes.extend(
                SourceChange(
                    ecosystem=change.ecosystem,
                    operation=change.operation,
                    surface=f"generated {change.surface}",
                    scope=change.scope,
                    destination=change.destination,
                    file=change.file,
                    line=start_line + change.line - 1,
                    matched_text=change.matched_text,
                )
                for change in generated
            )
        elif lower.endswith("pyproject.toml"):
            generated = _parse_poetry(body, file, assignments)
            changes.extend(
                SourceChange(
                    ecosystem=change.ecosystem,
                    operation=change.operation,
                    surface=f"generated {change.surface}",
                    scope=change.scope,
                    destination=change.destination,
                    file=change.file,
                    line=start_line + change.line - 1,
                    matched_text=change.matched_text,
                )
                for change in generated
            )
        elif ".cargo/" in lower and lower.endswith(("/config", "/config.toml")):
            generated = _parse_cargo(body, file, assignments)
            changes.extend(
                SourceChange(
                    ecosystem=change.ecosystem,
                    operation=change.operation,
                    surface=f"generated {change.surface}",
                    scope=change.scope,
                    destination=change.destination,
                    file=change.file,
                    line=start_line + change.line - 1,
                    matched_text=change.matched_text,
                )
                for change in generated
            )
    return changes


def _parse_commands(content: str, file: str, assignments: dict[str, str]) -> list[SourceChange]:
    changes: list[SourceChange] = []
    patterns: tuple[tuple[str, str, str, str, re.Pattern[str]], ...] = (
        (
            "npm",
            "replace",
            "npm config set",
            "scope",
            re.compile(
                r"\bnpm\s+config\s+set\s+(?P<scope>@[\w.-]+:)?registry\s+(?P<dest>\S+)", re.I
            ),
        ),
        (
            "yarn",
            "replace",
            "yarn config set",
            "scope",
            re.compile(
                r"\byarn\s+config\s+set\s+(?:registry|npmRegistryServer)\s+(?P<dest>\S+)", re.I
            ),
        ),
        (
            "pip",
            "replace",
            "pip --index-url",
            "none",
            re.compile(r"\bpip(?:3)?\b[^\n]*?--index-url(?:=|\s+)(?P<dest>\S+)", re.I),
        ),
        (
            "pip",
            "add",
            "pip --extra-index-url",
            "none",
            re.compile(r"\bpip(?:3)?\b[^\n]*?--extra-index-url(?:=|\s+)(?P<dest>\S+)", re.I),
        ),
        (
            "pip",
            "replace",
            "pip config set",
            "none",
            re.compile(
                r"\bpip(?:3)?\s+config\s+set\s+(?:global\.)?index-url\s+(?P<dest>\S+)", re.I
            ),
        ),
        (
            "pip",
            "add",
            "pip config set",
            "none",
            re.compile(
                r"\bpip(?:3)?\s+config\s+set\s+(?:global\.)?extra-index-url\s+(?P<dest>\S+)", re.I
            ),
        ),
        (
            "poetry",
            "add",
            "poetry source add",
            "poetry",
            re.compile(
                r"\bpoetry\s+source\s+add(?:\s+--\S+)*\s+(?P<scope>[\w.-]+)\s+(?P<dest>\S+)", re.I
            ),
        ),
        (
            "poetry",
            "add",
            "poetry config repositories",
            "poetry",
            re.compile(
                r"\bpoetry\s+config\s+repositories\.(?P<scope>[\w.-]+)\s+(?P<dest>\S+)", re.I
            ),
        ),
        (
            "maven",
            "replace",
            "Maven CLI repository",
            "none",
            re.compile(r"-Dmaven\.repo\.remote=(?P<dest>\S+)", re.I),
        ),
    )
    for line_number, line in enumerate(content.splitlines(), 1):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        for ecosystem, operation, surface, scope_mode, pattern in patterns:
            for match in pattern.finditer(line):
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
            line,
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


def _changes_for_file(content: str, file: str) -> list[SourceChange]:
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

    is_script = PurePosixPath(normalized).suffix.lower() in _EXECUTABLE_SUFFIXES
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
    changes: list[SourceChange] = []
    for file in components:
        content = file_cache.get(file)
        if content is None or "\x00" in content[:8192]:
            continue
        changes.extend(_changes_for_file(content, file))

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
