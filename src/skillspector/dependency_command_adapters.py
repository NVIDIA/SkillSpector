# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded package-manager grammars over public typed shell command facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

from skillspector.dependency_source_types import (
    AssignmentSite,
    CommandProducerReachability,
    CommandResolutionKind,
    CommandSite,
    DependencyEcosystem,
    DependencySourceOperation,
    DependencySourceScope,
    DependencySourceSurface,
    DependencyWorkBudget,
    ShellIssue,
    ShellIssueReason,
    ShellTruncationClaimStatus,
    ShellWorkOutcome,
    SourceSpan,
    StaticValue,
    StaticValueState,
)


@dataclass(frozen=True, slots=True)
class DependencyCommandCandidate:
    """One policy-free recognized dependency-source command sink."""

    ecosystem: DependencyEcosystem
    surface: DependencySourceSurface
    operation: DependencySourceOperation
    scope: DependencySourceScope
    destination: StaticValue
    span: SourceSpan

    def __post_init__(self) -> None:
        if not isinstance(self.ecosystem, DependencyEcosystem):
            raise ValueError("ecosystem must be code-owned")
        if not isinstance(self.surface, DependencySourceSurface):
            raise ValueError("surface must be code-owned")
        if not isinstance(self.operation, DependencySourceOperation):
            raise ValueError("operation must be code-owned")
        if not isinstance(self.scope, DependencySourceScope):
            raise ValueError("scope must be code-owned")
        if not isinstance(self.destination, StaticValue):
            raise ValueError("destination must be a StaticValue")
        if not isinstance(self.span, SourceSpan):
            raise ValueError("span must be a SourceSpan")


@dataclass(frozen=True, slots=True)
class MavenSettingsReference:
    """One typed Maven settings-file operand for bundle-local resolution."""

    path: StaticValue
    span: SourceSpan

    def __post_init__(self) -> None:
        if not isinstance(self.path, StaticValue):
            raise ValueError("path must be a StaticValue")
        if not isinstance(self.span, SourceSpan):
            raise ValueError("span must be a SourceSpan")


@dataclass(frozen=True, slots=True)
class AdapterResult:
    """Bounded typed adapter output with no finding or ledger reservation."""

    candidates: tuple[DependencyCommandCandidate, ...] = ()
    issues: tuple[ShellIssue, ...] = ()
    maven_settings: tuple[MavenSettingsReference, ...] = ()

    def __post_init__(self) -> None:
        normalized = (
            ("candidates", self.candidates, DependencyCommandCandidate),
            ("issues", self.issues, ShellIssue),
            ("maven_settings", self.maven_settings, MavenSettingsReference),
        )
        for name, values, value_type in normalized:
            items = tuple(values)
            if not all(isinstance(value, value_type) for value in items):
                raise ValueError(f"{name} contains an invalid value")
            object.__setattr__(self, name, items)


@dataclass(frozen=True, slots=True)
class _Token:
    value: StaticValue
    span: SourceSpan


@dataclass(frozen=True, slots=True)
class _NormalizedCommand:
    manager: bytes | None
    tokens: tuple[_Token, ...]
    environment: tuple[AssignmentSite, ...]
    issue_span: SourceSpan | None = None
    inert: bool = False


_MANAGERS: Final = frozenset({b"npm", b"yarn", b"pnpm", b"pip", b"poetry", b"cargo", b"uv", b"mvn"})
_UV_NAMED_INDEX_ENVIRONMENTS: Final = frozenset(
    {"UV_INDEX", "UV_EXTRA_INDEX_URL", "uv_index", "uv_extra_index_url"}
)
_WRAPPERS: Final = frozenset(
    {
        b"env",
        b"sudo",
        b"command",
        b"exec",
        b"nohup",
        b"nice",
        b"timeout",
        b"setsid",
        b"stdbuf",
        b"corepack",
        b"npx",
        b"xargs",
    }
)


def _exact(token: _Token) -> bytes | None:
    return (
        cast(bytes, token.value.exact_bytes)
        if token.value.state is StaticValueState.EXACT
        else None
    )


def _versioned(name: bytes, stem: bytes) -> bool:
    if not name.startswith(stem):
        return False
    suffix = name[len(stem) :]
    return (
        bool(suffix)
        and suffix[:1].isdigit()
        and all(byte == ord(".") or ord("0") <= byte <= ord("9") for byte in suffix)
    )


def _executable(value: bytes) -> bytes | None:
    basename = value.rsplit(b"/", 1)[-1]
    if basename in {*_MANAGERS, *_WRAPPERS}:
        return basename
    if _versioned(basename, b"pip"):
        return b"pip"
    if basename in {b"mvnw", b"mvnw.cmd"}:
        return b"mvn"
    return None


def _python_executable(value: bytes) -> bool:
    basename = value.rsplit(b"/", 1)[-1]
    return basename == b"python" or _versioned(basename, b"python")


def _assignment_token(token: _Token, site: CommandSite) -> AssignmentSite | None:
    raw = _exact(token)
    if raw is None or b"=" not in raw:
        return None
    name_bytes, value = raw.split(b"=", 1)
    try:
        name = name_bytes.decode("ascii")
    except UnicodeDecodeError:
        return None
    if (
        not name
        or not (name[0].isalpha() or name[0] == "_")
        or any(not (character.isalnum() or character == "_") for character in name[1:])
    ):
        return None
    return AssignmentSite(
        site.unit_id,
        site.provenance,
        token.span,
        name,
        StaticValue.exact(value),
    )


def _unknown_result_span(site: CommandSite, span: SourceSpan) -> _NormalizedCommand:
    return _NormalizedCommand(None, (), (), issue_span=span)


def _take_operand(tokens: tuple[_Token, ...], cursor: int) -> tuple[int, SourceSpan | None]:
    if cursor + 1 >= len(tokens):
        return cursor, tokens[cursor].span
    return cursor + 2, None


def _normalize_command(site: CommandSite) -> _NormalizedCommand:
    tokens = tuple(
        _Token(value, span) for value, span in zip(site.argv, site.argument_spans, strict=True)
    )
    environment = list(site.prefix_assignments)
    cursor = 0
    while cursor < len(tokens):
        raw = _exact(tokens[cursor])
        if raw is None:
            return _unknown_result_span(site, tokens[cursor].span)
        executable = _executable(raw)
        if executable in _MANAGERS:
            manager_tokens = tokens[cursor:]
            if executable == b"pip" and raw.rsplit(b"/", 1)[-1] != b"pip":
                manager_tokens = (
                    _Token(StaticValue.exact(b"pip"), tokens[cursor].span),
                    *manager_tokens[1:],
                )
            return _NormalizedCommand(executable, manager_tokens, tuple(environment))
        if _python_executable(raw):
            if (
                cursor + 2 < len(tokens)
                and _exact(tokens[cursor + 1]) == b"-m"
                and (_pip_name := _exact(tokens[cursor + 2])) is not None
                and _executable(_pip_name) == b"pip"
            ):
                return _NormalizedCommand(
                    b"pip",
                    (
                        _Token(StaticValue.exact(b"pip"), tokens[cursor + 2].span),
                        *tokens[cursor + 3 :],
                    ),
                    tuple(environment),
                )
            return _NormalizedCommand(None, (), ())
        if executable is None:
            return _NormalizedCommand(None, (), ())
        if executable == b"xargs":
            return _unknown_result_span(site, tokens[cursor].span)

        cursor += 1
        if executable == b"env":
            while cursor < len(tokens):
                option = _exact(tokens[cursor])
                if option is None:
                    return _unknown_result_span(site, tokens[cursor].span)
                if option == b"--":
                    cursor += 1
                    while cursor < len(tokens):
                        assignment = _assignment_token(tokens[cursor], site)
                        if assignment is None:
                            break
                        environment = [item for item in environment if item.name != assignment.name]
                        environment.append(assignment)
                        cursor += 1
                    break
                assignment = _assignment_token(tokens[cursor], site)
                if assignment is not None:
                    environment = [item for item in environment if item.name != assignment.name]
                    environment.append(assignment)
                    cursor += 1
                    continue
                if option in {b"-i", b"--ignore-environment"}:
                    environment.clear()
                    cursor += 1
                    continue
                if option in {b"-u", b"--unset", b"-C", b"--chdir"}:
                    next_cursor, error = _take_operand(tokens, cursor)
                    if error is not None:
                        return _unknown_result_span(site, error)
                    if option in {b"-u", b"--unset"}:
                        name = _exact(tokens[cursor + 1])
                        if name is None:
                            return _unknown_result_span(site, tokens[cursor + 1].span)
                        environment = [item for item in environment if item.name.encode() != name]
                    cursor = next_cursor
                    continue
                if option.startswith(b"--unset="):
                    name = option.partition(b"=")[2]
                    if not name:
                        return _unknown_result_span(site, tokens[cursor].span)
                    environment = [item for item in environment if item.name.encode() != name]
                    cursor += 1
                    continue
                if option.startswith(b"--chdir="):
                    if not option.partition(b"=")[2]:
                        return _unknown_result_span(site, tokens[cursor].span)
                    cursor += 1
                    continue
                if option.startswith(b"-"):
                    return _unknown_result_span(site, tokens[cursor].span)
                break
            continue

        if executable == b"sudo":
            inherited = list(environment)
            environment = []
            preserve_all = False
            preserve_names: set[str] | None = None
            while cursor < len(tokens):
                option = _exact(tokens[cursor])
                if option is None:
                    return _unknown_result_span(site, tokens[cursor].span)
                if option == b"--":
                    cursor += 1
                    break
                assignment = _assignment_token(tokens[cursor], site)
                if assignment is not None:
                    environment = [item for item in environment if item.name != assignment.name]
                    environment.append(assignment)
                    cursor += 1
                    continue
                if option in {b"-E", b"--preserve-env"}:
                    preserve_all = True
                    cursor += 1
                    continue
                if option.startswith(b"--preserve-env="):
                    preserve_names = {
                        name
                        for name in option.partition(b"=")[2].decode("ascii", "ignore").split(",")
                        if name
                    }
                    cursor += 1
                    continue
                if option in {b"-k", b"-i", b"--login", b"-n", b"-S", b"-H", b"-b"}:
                    cursor += 1
                    continue
                if option in {
                    b"-u",
                    b"--user",
                    b"-g",
                    b"--group",
                    b"-h",
                    b"--host",
                    b"-p",
                    b"--prompt",
                    b"-C",
                    b"--close-from",
                    b"-R",
                    b"--chroot",
                    b"-T",
                    b"--command-timeout",
                }:
                    next_cursor, error = _take_operand(tokens, cursor)
                    if error is not None:
                        return _unknown_result_span(site, error)
                    cursor = next_cursor
                    continue
                if any(
                    option.startswith(prefix)
                    for prefix in (
                        b"--user=",
                        b"--group=",
                        b"--host=",
                        b"--prompt=",
                        b"--close-from=",
                        b"--chroot=",
                        b"--command-timeout=",
                    )
                ):
                    if not option.partition(b"=")[2]:
                        return _unknown_result_span(site, tokens[cursor].span)
                    cursor += 1
                    continue
                if option.startswith(b"-"):
                    return _unknown_result_span(site, tokens[cursor].span)
                break
            if preserve_all:
                preserved = inherited
            elif preserve_names is not None:
                preserved = [item for item in inherited if item.name in preserve_names]
            else:
                preserved = []
            explicit_names = {item.name for item in environment}
            environment = [
                *[item for item in preserved if item.name not in explicit_names],
                *environment,
            ]
            continue

        if executable == b"command":
            while cursor < len(tokens):
                option = _exact(tokens[cursor])
                if option == b"--":
                    cursor += 1
                    break
                if option == b"-p":
                    cursor += 1
                    continue
                if option in {b"-v", b"-V"}:
                    return _NormalizedCommand(None, (), (), inert=True)
                if option is None or option.startswith(b"-"):
                    return _unknown_result_span(site, tokens[cursor].span)
                break
            continue

        if executable == b"exec":
            while cursor < len(tokens):
                option = _exact(tokens[cursor])
                if option == b"--":
                    cursor += 1
                    break
                if option == b"-a":
                    next_cursor, error = _take_operand(tokens, cursor)
                    if error is not None:
                        return _unknown_result_span(site, error)
                    cursor = next_cursor
                    continue
                if option == b"-c":
                    environment.clear()
                    cursor += 1
                    continue
                if option == b"-l":
                    cursor += 1
                    continue
                if option is None or option.startswith(b"-"):
                    return _unknown_result_span(site, tokens[cursor].span)
                break
            continue

        if executable == b"nohup":
            if cursor < len(tokens) and _exact(tokens[cursor]) == b"--":
                cursor += 1
            elif cursor < len(tokens) and (_exact(tokens[cursor]) or b"").startswith(b"-"):
                if _exact(tokens[cursor]) in {b"--help", b"--version"}:
                    return _NormalizedCommand(None, (), (), inert=True)
                return _unknown_result_span(site, tokens[cursor].span)
            continue

        if executable == b"nice":
            while cursor < len(tokens):
                option = _exact(tokens[cursor])
                if option == b"--":
                    cursor += 1
                    break
                if option in {b"-n", b"--adjustment"}:
                    next_cursor, error = _take_operand(tokens, cursor)
                    if error is not None:
                        return _unknown_result_span(site, error)
                    cursor = next_cursor
                    continue
                if option is not None and (
                    option.startswith(b"--adjustment=")
                    or (option.startswith(b"-") and option[1:].isdigit())
                ):
                    if option == b"--adjustment=":
                        return _unknown_result_span(site, tokens[cursor].span)
                    cursor += 1
                    continue
                if option in {b"--help", b"--version"}:
                    return _NormalizedCommand(None, (), (), inert=True)
                if option is None or option.startswith(b"-"):
                    return _unknown_result_span(site, tokens[cursor].span)
                break
            continue

        if executable == b"timeout":
            while cursor < len(tokens):
                option = _exact(tokens[cursor])
                if option == b"--":
                    cursor += 1
                    break
                if option in {b"-k", b"--kill-after", b"-s", b"--signal"}:
                    next_cursor, error = _take_operand(tokens, cursor)
                    if error is not None:
                        return _unknown_result_span(site, error)
                    cursor = next_cursor
                    continue
                if option is not None and option.startswith((b"--kill-after=", b"--signal=")):
                    if option in {b"--kill-after=", b"--signal="}:
                        return _unknown_result_span(site, tokens[cursor].span)
                    cursor += 1
                    continue
                if option in {b"--foreground", b"--preserve-status", b"--verbose"}:
                    cursor += 1
                    continue
                if option in {b"--help", b"--version"}:
                    return _NormalizedCommand(None, (), (), inert=True)
                if option is None or option.startswith(b"-"):
                    return _unknown_result_span(site, tokens[cursor].span)
                break
            if cursor >= len(tokens):
                return _unknown_result_span(site, tokens[-1].span)
            cursor += 1
            continue

        if executable == b"setsid":
            while cursor < len(tokens):
                option = _exact(tokens[cursor])
                if option == b"--":
                    cursor += 1
                    break
                if option in {b"-c", b"--ctty", b"-f", b"--fork", b"-w", b"--wait"}:
                    cursor += 1
                    continue
                if option in {b"--help", b"--version"}:
                    return _NormalizedCommand(None, (), (), inert=True)
                if option is None or option.startswith(b"-"):
                    return _unknown_result_span(site, tokens[cursor].span)
                break
            continue

        if executable == b"stdbuf":
            while cursor < len(tokens):
                option = _exact(tokens[cursor])
                if option == b"--":
                    cursor += 1
                    break
                if option in {b"-i", b"-o", b"-e", b"--input", b"--output", b"--error"}:
                    next_cursor, error = _take_operand(tokens, cursor)
                    if error is not None:
                        return _unknown_result_span(site, error)
                    cursor = next_cursor
                    continue
                if option is not None and (
                    option.startswith((b"--input=", b"--output=", b"--error="))
                    or (len(option) > 2 and option[:2] in {b"-i", b"-o", b"-e"})
                ):
                    if option in {b"--input=", b"--output=", b"--error="}:
                        return _unknown_result_span(site, tokens[cursor].span)
                    cursor += 1
                    continue
                if option in {b"--help", b"--version"}:
                    return _NormalizedCommand(None, (), (), inert=True)
                if option is None or option.startswith(b"-"):
                    return _unknown_result_span(site, tokens[cursor].span)
                break
            continue

        if executable in {b"corepack", b"npx"}:
            while cursor < len(tokens):
                option = _exact(tokens[cursor])
                if option == b"--":
                    cursor += 1
                    break
                if executable == b"npx" and option in {
                    b"-y",
                    b"--yes",
                    b"--no",
                    b"--ignore-existing",
                }:
                    cursor += 1
                    continue
                if executable == b"npx" and option in {
                    b"-p",
                    b"--package",
                    b"--cache",
                    b"--userconfig",
                }:
                    next_cursor, error = _take_operand(tokens, cursor)
                    if error is not None:
                        return _unknown_result_span(site, error)
                    cursor = next_cursor
                    continue
                if (
                    executable == b"npx"
                    and option is not None
                    and option.startswith((b"--package=", b"--cache=", b"--userconfig="))
                ):
                    if option in {b"--package=", b"--cache=", b"--userconfig="}:
                        return _unknown_result_span(site, tokens[cursor].span)
                    cursor += 1
                    continue
                if option is None or option.startswith(b"-"):
                    return _unknown_result_span(site, tokens[cursor].span)
                break
            if cursor >= len(tokens):
                return _unknown_result_span(site, tokens[-1].span)
            downstream = _exact(tokens[cursor])
            if downstream is None or _executable(downstream) not in {b"npm", b"yarn", b"pnpm"}:
                return _unknown_result_span(site, tokens[cursor].span)
            continue

    return _NormalizedCommand(None, (), ())


def _candidate(
    ecosystem: DependencyEcosystem,
    surface: DependencySourceSurface,
    operation: DependencySourceOperation,
    scope: DependencySourceScope,
    token: _Token,
) -> DependencyCommandCandidate:
    return DependencyCommandCandidate(ecosystem, surface, operation, scope, token.value, token.span)


def _issue(
    site: CommandSite,
    span: SourceSpan,
    budget: DependencyWorkBudget,
) -> ShellIssue | None:
    exhaustion = budget.charge_shell_issues(1)
    if exhaustion is None:
        return ShellIssue(
            ShellIssueReason.UNSUPPORTED_SEMANTICS,
            ShellWorkOutcome.PARTIAL,
            span,
            unit_id=site.unit_id,
        )
    if budget.claim_reserved_shell_truncation_issue() is not ShellTruncationClaimStatus.CLAIMED:
        return None
    return ShellIssue(
        ShellIssueReason.RESOURCE_LIMIT,
        ShellWorkOutcome.PARTIAL,
        span,
        unit_id=site.unit_id,
        exhaustion=exhaustion,
    )


def _issues(issue: ShellIssue | None) -> tuple[ShellIssue, ...]:
    return () if issue is None else (issue,)


def _environment_ecosystem(name: str) -> DependencyEcosystem | None:
    if name in {"NPM_CONFIG_REGISTRY", "npm_config_registry"}:
        return DependencyEcosystem.NPM
    if name in {"YARN_NPM_REGISTRY_SERVER", "yarn_npm_registry_server"}:
        return DependencyEcosystem.YARN
    if name in {"PNPM_CONFIG_REGISTRY", "pnpm_config_registry"}:
        return DependencyEcosystem.PNPM
    if name in {"PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "pip_index_url", "pip_extra_index_url"}:
        return DependencyEcosystem.PIP
    if name in {
        "UV_INDEX",
        "UV_INDEX_URL",
        "UV_DEFAULT_INDEX",
        "UV_EXTRA_INDEX_URL",
        "uv_index",
        "uv_index_url",
        "uv_default_index",
        "uv_extra_index_url",
    }:
        return DependencyEcosystem.UV
    if name.startswith("CARGO_REGISTRIES_") and name.endswith("_INDEX"):
        return DependencyEcosystem.CARGO
    if name.startswith("POETRY_REPOSITORIES_") and name.endswith("_URL"):
        return DependencyEcosystem.POETRY
    return None


def _environment_candidates(
    assignments: tuple[AssignmentSite, ...],
    *,
    manager: DependencyEcosystem | None,
) -> tuple[list[DependencyCommandCandidate], SourceSpan | None]:
    candidates: list[DependencyCommandCandidate] = []
    for assignment in assignments:
        ecosystem = _environment_ecosystem(assignment.name)
        if ecosystem is None or (manager is not None and ecosystem is not manager):
            continue
        destination = assignment.value
        if assignment.name in _UV_NAMED_INDEX_ENVIRONMENTS:
            normalized = _uv_index_environment_destination(assignment.value)
            if normalized is None:
                return [], assignment.span
            destination = normalized
        candidates.append(
            DependencyCommandCandidate(
                ecosystem,
                DependencySourceSurface.ENVIRONMENT,
                DependencySourceOperation.SET,
                DependencySourceScope.ENVIRONMENT,
                destination,
                assignment.span,
            )
        )
    return candidates, None


def _option_value(
    tokens: tuple[_Token, ...], index: int, names: tuple[bytes, ...]
) -> tuple[_Token | None, int]:
    raw = _exact(tokens[index])
    if raw is None:
        return None, index + 1
    for name in names:
        if raw == name:
            return (tokens[index + 1], index + 2) if index + 1 < len(tokens) else (None, index + 1)
        prefix = name + b"="
        if raw.startswith(prefix):
            attached = raw[len(prefix) :]
            return (
                _Token(StaticValue.exact(attached), tokens[index].span) if attached else None,
                index + 1,
            )
    return None, index


def _uv_index_destination(value: _Token) -> _Token:
    if value.value.state is not StaticValueState.EXACT:
        return value
    raw = cast(bytes, value.value.exact_bytes)
    name, separator, destination = raw.partition(b"=")
    if (
        not separator
        or not name
        or not destination
        or any(
            character not in b"-_."
            and not 48 <= character <= 57
            and not 65 <= character <= 90
            and not 97 <= character <= 122
            for character in name
        )
    ):
        return value
    return _Token(StaticValue.exact(destination), value.span)


def _uv_index_environment_destination(value: StaticValue) -> StaticValue | None:
    if value.state is not StaticValueState.EXACT:
        return value
    raw = cast(bytes, value.exact_bytes)
    if not raw or any(character in b" \t\n\r\v\f" for character in raw):
        return None
    name, separator, destination = raw.partition(b"=")
    if not separator or b"://" in name:
        return value
    if (
        not name
        or not destination
        or any(
            character not in b"-_."
            and not 48 <= character <= 57
            and not 65 <= character <= 90
            and not 97 <= character <= 122
            for character in name
        )
    ):
        return None
    return StaticValue.exact(destination)


def _terminator_index(tokens: tuple[_Token, ...], start: int) -> int:
    return next(
        (index for index in range(start, len(tokens)) if _exact(tokens[index]) == b"--"),
        len(tokens),
    )


def _unknown_option_beside_source(
    tokens: tuple[_Token, ...],
    *,
    start: int,
    source_options: frozenset[bytes],
    no_operand_options: frozenset[bytes],
    operand_options: frozenset[bytes],
    attached_short_sources: tuple[bytes, ...] = (),
    attached_short_operands: tuple[bytes, ...] = (),
) -> SourceSpan | None:
    """Reject one unknown option only when a recognized sink shares its argv segment."""
    end = _terminator_index(tokens, start)
    source_present = False
    for token in tokens[start:end]:
        raw = _exact(token)
        if raw is None:
            continue
        if raw in source_options or any(raw.startswith(option + b"=") for option in source_options):
            source_present = True
            break
        if any(raw.startswith(option) and raw != option for option in attached_short_sources):
            source_present = True
            break
    if not source_present:
        return None
    cursor = start
    while cursor < end:
        raw = _exact(tokens[cursor])
        if raw is None or not raw.startswith(b"-") or raw == b"-":
            cursor += 1
            continue
        if raw in source_options or raw in operand_options:
            if cursor + 1 >= end:
                return tokens[cursor].span
            cursor += 2
            continue
        if raw in no_operand_options:
            cursor += 1
            continue
        if any(raw.startswith(option + b"=") for option in (*source_options, *operand_options)):
            cursor += 1
            continue
        if any(
            raw.startswith(option) and raw != option
            for option in (*attached_short_sources, *attached_short_operands)
        ):
            cursor += 1
            continue
        return tokens[cursor].span
    return None


def _npm_family(
    tokens: tuple[_Token, ...], ecosystem: DependencyEcosystem
) -> tuple[list[DependencyCommandCandidate], SourceSpan | None]:
    config_names = {b"config", b"c"}
    scope = DependencySourceScope.GLOBAL
    cursor = 1
    config_seen = False
    action: bytes | None = None
    operands: list[tuple[_Token, DependencySourceScope]] = []
    while cursor < len(tokens):
        raw = _exact(tokens[cursor])
        if raw is None:
            if action is not None:
                operands.append((tokens[cursor], scope))
                cursor += 1
                continue
            return [], tokens[cursor].span
        if raw == b"--":
            operands.extend((token, scope) for token in tokens[cursor + 1 :])
            break
        if raw in {b"-g", b"--global"}:
            scope = DependencySourceScope.GLOBAL
            cursor += 1
            continue
        if raw.startswith(b"--location="):
            location = raw.partition(b"=")[2]
            scope = (
                DependencySourceScope.PROJECT
                if location == b"project"
                else DependencySourceScope.GLOBAL
            )
            cursor += 1
            continue
        if raw == b"--location":
            if cursor + 1 >= len(tokens) or _exact(tokens[cursor + 1]) is None:
                return [], tokens[cursor].span
            location_operand = cast(bytes, _exact(tokens[cursor + 1]))
            scope = (
                DependencySourceScope.PROJECT
                if location_operand == b"project"
                else DependencySourceScope.GLOBAL
            )
            cursor += 2
            continue
        if action is not None and raw == b"--userconfig":
            if cursor + 1 >= len(tokens) or _exact(tokens[cursor + 1]) is None:
                return [], tokens[cursor].span
            cursor += 2
            continue
        if raw.startswith(b"-"):
            if config_seen or action is not None:
                return [], tokens[cursor].span
            if raw in {b"--silent", b"--json", b"--workspaces", b"--include-workspace-root"}:
                cursor += 1
                continue
            if raw in {b"--userconfig", b"--prefix", b"--workspace", b"-w"}:
                if cursor + 1 >= len(tokens):
                    return [], tokens[cursor].span
                cursor += 2
                continue
            if any(
                raw.startswith(prefix)
                for prefix in (b"--userconfig=", b"--prefix=", b"--workspace=")
            ):
                if not raw.partition(b"=")[2]:
                    return [], tokens[cursor].span
                cursor += 1
                continue
            return [], tokens[cursor].span
        if not config_seen:
            if raw in config_names:
                config_seen = True
                cursor += 1
                continue
            break
        if action is None:
            if raw in {b"set", b"delete", b"del", b"rm", b"unset"}:
                action = raw
                cursor += 1
                continue
            if raw in {b"get", b"list", b"ls", b"edit", b"fix", b"help"}:
                return [], None
            return [], tokens[cursor].span
        operands.append((tokens[cursor], scope))
        cursor += 1

    candidates: list[DependencyCommandCandidate] = []
    if config_seen and action is not None:
        remove = action in {b"delete", b"del", b"rm", b"unset"}
        if remove:
            if any(_exact(token) is None for token, _scope in operands):
                return [], operands[0][0].span
            return [], None
        operand_index = 0
        while operand_index < len(operands):
            key_token, key_pair_scope = operands[operand_index]
            key = _exact(key_token)
            if key is None:
                return [], key_token.span
            if b"=" in key:
                key, destination = key.split(b"=", 1)
                value = _Token(StaticValue.exact(destination), key_token.span)
                pair_scope = key_pair_scope
                operand_index += 1
            else:
                if operand_index + 1 >= len(operands):
                    return [], key_token.span
                value, pair_scope = operands[operand_index + 1]
                operand_index += 2
            key_scope = (
                DependencySourceScope.SCOPED
                if key.startswith(b"@") and key.endswith(b":registry")
                else pair_scope
            )
            if key != b"registry" and key_scope is not DependencySourceScope.SCOPED:
                continue
            candidates.append(
                _candidate(
                    ecosystem,
                    DependencySourceSurface.COMMAND,
                    DependencySourceOperation.SET,
                    key_scope,
                    value,
                )
            )
        return candidates, None

    inert = {_exact(token) for token in tokens[1:2]} & {
        b"view",
        b"ping",
        b"run",
        b"publish",
        b"help",
    }
    if inert:
        return [], None
    if issue_span := _unknown_option_beside_source(
        tokens,
        start=2,
        source_options=frozenset({b"--registry"}),
        no_operand_options=frozenset(
            {
                b"-g",
                b"--global",
                b"--dry-run",
                b"--ignore-scripts",
                b"--no-audit",
                b"--no-fund",
                b"--package-lock-only",
                b"--legacy-peer-deps",
                b"--force",
                b"-f",
                b"--silent",
                b"--verbose",
            }
        ),
        operand_options=frozenset(
            {
                b"--workspace",
                b"-w",
                b"--prefix",
                b"--cache",
                b"--userconfig",
                b"--tag",
                b"--omit",
                b"--include",
                b"--loglevel",
            }
        ),
    ):
        return [], issue_span
    cursor = 1
    end = _terminator_index(tokens, cursor)
    while cursor < end:
        invocation_value, next_cursor = _option_value(tokens, cursor, (b"--registry",))
        if next_cursor != cursor:
            if invocation_value is None:
                return [], tokens[cursor].span
            candidates.append(
                _candidate(
                    ecosystem,
                    DependencySourceSurface.INVOCATION,
                    DependencySourceOperation.USE,
                    DependencySourceScope.INVOCATION,
                    invocation_value,
                )
            )
            cursor = next_cursor
        else:
            cursor += 1
    return candidates, None


def _yarn(tokens: tuple[_Token, ...]) -> tuple[list[DependencyCommandCandidate], SourceSpan | None]:
    cursor = 1
    if cursor >= len(tokens) or _exact(tokens[cursor]) != b"config":
        if cursor < len(tokens) and _exact(tokens[cursor]) in {
            b"why",
            b"publish",
            b"run",
            b"help",
        }:
            return [], None
        return _npm_family(tokens, DependencyEcosystem.YARN)
    cursor += 1
    action: bytes | None = None
    scope = DependencySourceScope.GLOBAL
    operands: list[_Token] = []
    while cursor < len(tokens):
        raw = _exact(tokens[cursor])
        if raw is None:
            if action is not None:
                operands.append(tokens[cursor])
                cursor += 1
                continue
            return [], tokens[cursor].span
        if raw == b"--":
            operands.extend(tokens[cursor + 1 :])
            break
        if raw in {b"-H", b"--home"}:
            scope = DependencySourceScope.GLOBAL
            cursor += 1
            continue
        if raw in {b"--json", b"--no-defaults"}:
            cursor += 1
            continue
        if raw.startswith(b"-"):
            return [], tokens[cursor].span
        if action is None:
            if raw not in {b"set", b"unset", b"delete"}:
                if raw in {b"get", b"list", b"help"}:
                    return [], None
                return [], tokens[cursor].span
            action = raw
        else:
            operands.append(tokens[cursor])
        cursor += 1
    if action is None or not operands:
        return [], None
    key = _exact(operands[0])
    key_scope = (
        DependencySourceScope.SCOPED
        if key is not None and key.startswith(b"npmScopes.") and key.endswith(b".npmRegistryServer")
        else scope
    )
    if (
        key not in {b"registry", b"npmRegistryServer"}
        and key_scope is not DependencySourceScope.SCOPED
    ):
        return [], None
    if action == b"set":
        if len(operands) != 2:
            return [], operands[-1].span
        value = operands[1]
    else:
        if len(operands) != 1:
            return [], operands[-1].span
        return [], None
    return [
        _candidate(
            DependencyEcosystem.YARN,
            DependencySourceSurface.COMMAND,
            DependencySourceOperation.SET,
            key_scope,
            value,
        )
    ], None


def _pip(tokens: tuple[_Token, ...]) -> tuple[list[DependencyCommandCandidate], SourceSpan | None]:
    cursor = 1
    no_operand = {
        b"--isolated",
        b"--require-virtualenv",
        b"--no-input",
        b"--disable-pip-version-check",
    }
    with_operand = {
        b"--python",
        b"--proxy",
        b"--timeout",
        b"--retries",
        b"--cert",
        b"--client-cert",
        b"--cache-dir",
        b"--log",
    }
    while cursor < len(tokens) and (_exact(tokens[cursor]) or b"").startswith(b"-"):
        raw = cast(bytes, _exact(tokens[cursor]))
        if raw in no_operand:
            cursor += 1
        elif raw in with_operand:
            if cursor + 1 >= len(tokens):
                return [], tokens[cursor].span
            cursor += 2
        elif any(raw.startswith(option + b"=") for option in with_operand):
            if not raw.partition(b"=")[2]:
                return [], tokens[cursor].span
            cursor += 1
        else:
            return [], tokens[cursor].span
    if cursor < len(tokens) and _exact(tokens[cursor]) == b"help":
        return [], None
    if cursor < len(tokens) and _exact(tokens[cursor]) == b"config":
        cursor += 1
        scope_flag: DependencySourceScope | None = None
        action: bytes | None = None
        operands: list[_Token] = []
        while cursor < len(tokens):
            config_raw = _exact(tokens[cursor])
            if config_raw is None:
                if action is not None:
                    operands.append(tokens[cursor])
                    cursor += 1
                    continue
                return [], tokens[cursor].span
            if config_raw in {b"--global", b"--user", b"--site"}:
                scope_flag = DependencySourceScope.GLOBAL
            elif config_raw == b"--":
                operands.extend(tokens[cursor + 1 :])
                break
            elif config_raw.startswith(b"-"):
                return [], tokens[cursor].span
            elif action is None:
                if config_raw not in {b"set", b"unset"}:
                    if config_raw in {b"get", b"list", b"debug"}:
                        return [], None
                    return [], tokens[cursor].span
                action = config_raw
            else:
                operands.append(tokens[cursor])
            cursor += 1
        if action is None or not operands:
            return [], None
        key = _exact(operands[0])
        if key is None:
            return [], operands[0].span
        key_tail = key.rsplit(b".", 1)[-1]
        if key_tail not in {b"index-url", b"extra-index-url"}:
            return [], None
        scope = scope_flag or (
            DependencySourceScope.GLOBAL
            if key.startswith(b"global.")
            else DependencySourceScope.COMMAND
        )
        if action == b"set":
            if len(operands) != 2:
                return [], operands[-1].span
            config_value = operands[1]
        else:
            if len(operands) != 1:
                return [], operands[-1].span
            return [], None
        return [
            _candidate(
                DependencyEcosystem.PIP,
                DependencySourceSurface.COMMAND,
                DependencySourceOperation.SET,
                scope,
                config_value,
            )
        ], None
    if issue_span := _unknown_option_beside_source(
        tokens,
        start=min(cursor + 1, len(tokens)),
        source_options=frozenset({b"-i", b"--index-url", b"--index-u", b"--extra-index-url"}),
        no_operand_options=frozenset(
            {
                b"-q",
                b"--quiet",
                b"-v",
                b"--verbose",
                b"-U",
                b"--upgrade",
                b"--pre",
                b"--no-deps",
                b"--no-cache-dir",
                b"--require-hashes",
                b"--ignore-installed",
                b"--force-reinstall",
            }
        ),
        operand_options=frozenset(
            {
                b"-r",
                b"--requirement",
                b"-c",
                b"--constraint",
                b"-f",
                b"--find-links",
                b"--trusted-host",
                b"--timeout",
                b"--retries",
                b"--cert",
                b"--client-cert",
                b"--proxy",
            }
        ),
        attached_short_sources=(b"-i",),
        attached_short_operands=(b"-r", b"-c", b"-f"),
    ):
        return [], issue_span
    candidates: list[DependencyCommandCandidate] = []
    end = _terminator_index(tokens, cursor)
    while cursor < end:
        invocation_raw = _exact(tokens[cursor])
        if invocation_raw is None:
            cursor += 1
            continue
        invocation_value: _Token | None = None
        next_cursor = cursor
        if invocation_raw.startswith(b"-i") and invocation_raw != b"-i":
            invocation_value = _Token(StaticValue.exact(invocation_raw[2:]), tokens[cursor].span)
            next_cursor = cursor + 1
        else:
            invocation_value, next_cursor = _option_value(
                tokens,
                cursor,
                (b"-i", b"--index-url", b"--index-u", b"--extra-index-url"),
            )
        if next_cursor != cursor:
            if invocation_value is None:
                return [], tokens[cursor].span
            candidates.append(
                _candidate(
                    DependencyEcosystem.PIP,
                    DependencySourceSurface.INVOCATION,
                    DependencySourceOperation.USE,
                    DependencySourceScope.INVOCATION,
                    invocation_value,
                )
            )
            cursor = next_cursor
        else:
            cursor += 1
    return candidates, None


def _poetry(
    tokens: tuple[_Token, ...],
) -> tuple[list[DependencyCommandCandidate], SourceSpan | None]:
    cursor = 1
    while cursor < len(tokens):
        raw = _exact(tokens[cursor])
        if raw in {b"--help", b"--version", b"-h", b"-V"}:
            return [], None
        if raw in {b"-C", b"--directory", b"--project"}:
            if cursor + 1 >= len(tokens):
                return [], tokens[cursor].span
            cursor += 2
        elif raw in {
            b"-n",
            b"--no-interaction",
            b"--no-cache",
            b"--ansi",
            b"--no-ansi",
            b"-q",
            b"-v",
            b"-vv",
            b"-vvv",
        }:
            cursor += 1
        elif raw is not None and raw.startswith((b"--directory=", b"--project=")):
            if not raw.partition(b"=")[2]:
                return [], tokens[cursor].span
            cursor += 1
        elif raw is not None and raw.startswith(b"-"):
            return [], tokens[cursor].span
        else:
            break
    if cursor >= len(tokens):
        return [], None
    command = _exact(tokens[cursor])
    cursor += 1
    if command == b"source" and cursor < len(tokens):
        action = _exact(tokens[cursor])
        cursor += 1
        if action not in {b"add", b"remove"}:
            return [], None
        source_operands: list[_Token] = []
        while cursor < len(tokens):
            raw = _exact(tokens[cursor])
            if raw == b"--":
                source_operands.extend(tokens[cursor + 1 :])
                break
            if raw in {b"--priority"}:
                if cursor + 1 >= len(tokens):
                    return [], tokens[cursor].span
                cursor += 2
            elif raw in {b"--default", b"--secondary", b"-n", b"--no-interaction"}:
                cursor += 1
            elif raw is not None and raw.startswith(b"--priority="):
                cursor += 1
            elif raw is not None and raw.startswith(b"-"):
                return [], tokens[cursor].span
            else:
                source_operands.append(tokens[cursor])
                cursor += 1
        if action == b"add" and len(source_operands) == 2:
            return [
                _candidate(
                    DependencyEcosystem.POETRY,
                    DependencySourceSurface.COMMAND,
                    DependencySourceOperation.ADD,
                    DependencySourceScope.SOURCE,
                    source_operands[1],
                )
            ], None
        if action == b"remove" and len(source_operands) == 1:
            return [], None
        return ([], source_operands[-1].span) if source_operands else ([], tokens[cursor - 1].span)
    if command == b"config":
        local = False
        unset = False
        config_operands: list[_Token] = []
        while cursor < len(tokens):
            raw = _exact(tokens[cursor])
            if raw == b"--local":
                local = True
            elif raw == b"--unset":
                unset = True
            elif raw in {b"--list", b"-l"}:
                return [], None
            elif raw is not None and raw.startswith(b"-"):
                return [], tokens[cursor].span
            else:
                config_operands.append(tokens[cursor])
            cursor += 1
        key = _exact(config_operands[0]) if config_operands else None
        if key is None or not key.startswith(b"repositories."):
            return [], None
        key_parts = key.split(b".")
        if len(key_parts) not in {2, 3} or (len(key_parts) == 3 and key_parts[2] != b"url"):
            return [], None
        if unset and len(config_operands) == 1:
            return [], None
        if not unset and len(config_operands) == 2:
            return [
                _candidate(
                    DependencyEcosystem.POETRY,
                    DependencySourceSurface.COMMAND,
                    DependencySourceOperation.SET,
                    DependencySourceScope.PROJECT if local else DependencySourceScope.REPOSITORY,
                    config_operands[1],
                )
            ], None
        return [], config_operands[-1].span
    return [], None


def _cargo(
    tokens: tuple[_Token, ...],
) -> tuple[list[DependencyCommandCandidate], SourceSpan | None]:
    if issue_span := _unknown_option_beside_source(
        tokens,
        start=1,
        source_options=frozenset({b"--config"}),
        no_operand_options=frozenset(
            {b"-v", b"--verbose", b"-q", b"--quiet", b"--locked", b"--offline", b"--frozen"}
        ),
        operand_options=frozenset(
            {
                b"--color",
                b"--manifest-path",
                b"--target-dir",
                b"--target",
                b"-j",
                b"--jobs",
                b"--features",
                b"-p",
                b"--package",
            }
        ),
    ):
        return [], issue_span
    candidates: list[DependencyCommandCandidate] = []
    cursor = 1
    end = _terminator_index(tokens, cursor)
    while cursor < end:
        value, next_cursor = _option_value(tokens, cursor, (b"--config",))
        if next_cursor == cursor:
            cursor += 1
            continue
        if value is None or value.value.state is not StaticValueState.EXACT:
            return [], tokens[cursor].span
        raw = cast(bytes, value.value.exact_bytes)
        key, separator, destination = raw.partition(b"=")
        if separator and key.startswith(b"registries.") and key.endswith(b".index"):
            if (
                len(destination) >= 2
                and destination[:1] == destination[-1:]
                and destination[:1] in {b"'", b'"'}
            ):
                destination = destination[1:-1]
            candidates.append(
                _candidate(
                    DependencyEcosystem.CARGO,
                    DependencySourceSurface.INVOCATION,
                    DependencySourceOperation.USE,
                    DependencySourceScope.REGISTRY,
                    _Token(StaticValue.exact(destination), value.span),
                )
            )
        cursor = next_cursor
    return candidates, None


def _uv(tokens: tuple[_Token, ...]) -> tuple[list[DependencyCommandCandidate], SourceSpan | None]:
    cursor = 1
    if cursor < len(tokens) and _exact(tokens[cursor]) in {
        b"help",
        b"--help",
        b"--version",
        b"-h",
        b"-V",
    }:
        return [], None
    if cursor < len(tokens) and _exact(tokens[cursor]) == b"index":
        cursor += 1
        action = _exact(tokens[cursor]) if cursor < len(tokens) else None
        if action == b"add" and cursor + 2 < len(tokens):
            return [
                _candidate(
                    DependencyEcosystem.UV,
                    DependencySourceSurface.COMMAND,
                    DependencySourceOperation.ADD,
                    DependencySourceScope.SOURCE,
                    tokens[cursor + 2],
                )
            ], None
        return [], None
    if issue_span := _unknown_option_beside_source(
        tokens,
        start=cursor,
        source_options=frozenset(
            {b"--index-url", b"--default-index", b"--index", b"--extra-index-url"}
        ),
        no_operand_options=frozenset(
            {
                b"-q",
                b"--quiet",
                b"-v",
                b"--verbose",
                b"--offline",
                b"--no-cache",
                b"--refresh",
                b"--native-tls",
                b"--no-progress",
            }
        ),
        operand_options=frozenset(
            {
                b"--python",
                b"--project",
                b"--directory",
                b"--config-file",
                b"--keyring-provider",
                b"--resolution",
                b"--prerelease",
            }
        ),
    ):
        return [], issue_span
    candidates: list[DependencyCommandCandidate] = []
    end = _terminator_index(tokens, cursor)
    while cursor < end:
        option = _exact(tokens[cursor])
        value, next_cursor = _option_value(
            tokens,
            cursor,
            (b"--index-url", b"--default-index", b"--index", b"--extra-index-url"),
        )
        if next_cursor != cursor:
            if value is None:
                return [], tokens[cursor].span
            if option == b"--index" or (option or b"").startswith(b"--index="):
                value = _uv_index_destination(value)
            candidates.append(
                _candidate(
                    DependencyEcosystem.UV,
                    DependencySourceSurface.INVOCATION,
                    DependencySourceOperation.USE,
                    DependencySourceScope.INVOCATION,
                    value,
                )
            )
            cursor = next_cursor
        else:
            cursor += 1
    return candidates, None


def _maven(
    tokens: tuple[_Token, ...],
) -> tuple[tuple[MavenSettingsReference, ...], SourceSpan | None]:
    terminator = _terminator_index(tokens, 1)
    visible = tokens[1:terminator]
    settings_positions = [
        index
        for index, token in enumerate(visible, start=1)
        if (_exact(token) or b"") in {b"-s", b"--settings"}
        or (_exact(token) or b"").startswith((b"-s", b"--settings="))
    ]
    if not settings_positions:
        return (), None
    first_position = settings_positions[0]
    first_raw = _exact(tokens[first_position])
    if (
        first_position != 1
        or len(settings_positions) != 1
        or first_raw not in {b"-s", b"--settings"}
        or first_position + 1 >= terminator
    ):
        return (), tokens[first_position].span
    path = tokens[first_position + 1]
    if path.value.state is not StaticValueState.EXACT or not cast(bytes, path.value.exact_bytes):
        return (), path.span
    known_no_operand = {
        b"-q",
        b"--quiet",
        b"-B",
        b"--batch-mode",
        b"-U",
        b"--update-snapshots",
        b"-o",
        b"--offline",
        b"-e",
        b"--errors",
        b"-X",
        b"--debug",
        b"-N",
        b"--non-recursive",
        b"-V",
        b"--show-version",
    }
    known_operand = {
        b"-f",
        b"--file",
        b"-P",
        b"--activate-profiles",
        b"-pl",
        b"--projects",
        b"-T",
        b"--threads",
    }
    cursor = first_position + 2
    while cursor < terminator:
        raw = _exact(tokens[cursor])
        if raw is None or not raw.startswith(b"-"):
            cursor += 1
            continue
        if raw in known_no_operand or raw in {b"-am", b"-amd"} or raw.startswith(b"-D"):
            cursor += 1
            continue
        if raw in known_operand:
            if cursor + 1 >= terminator:
                return (), tokens[cursor].span
            cursor += 2
            continue
        if any(raw.startswith(option + b"=") for option in known_operand):
            cursor += 1
            continue
        return (), tokens[cursor].span
    return (MavenSettingsReference(path.value, path.span),), None


def adapt_command(site: CommandSite, *, budget: DependencyWorkBudget) -> AdapterResult:
    """Adapt one public typed command site through fixed wrapper/manager grammars."""
    if not isinstance(site, CommandSite):
        raise ValueError("site must be a CommandSite")
    if not isinstance(budget, DependencyWorkBudget):
        raise ValueError("budget must be a DependencyWorkBudget")
    if site.producer is CommandProducerReachability.INERT:
        return AdapterResult()
    first = (
        cast(bytes, site.argv[0].exact_bytes)
        if site.argv[0].state is StaticValueState.EXACT
        else None
    )
    relevant = bool(site.exported_assignments) or (
        first is not None and _executable(first) in {*_MANAGERS, *_WRAPPERS}
    )
    if site.resolution is CommandResolutionKind.FUNCTION:
        return AdapterResult()
    if relevant and (
        site.resolution is CommandResolutionKind.AMBIGUOUS
        or site.producer is CommandProducerReachability.AMBIGUOUS
    ):
        return AdapterResult(issues=_issues(_issue(site, site.span, budget)))

    candidates, environment_issue_span = _environment_candidates(
        site.exported_assignments,
        manager=None,
    )
    if environment_issue_span is not None:
        return AdapterResult(issues=_issues(_issue(site, environment_issue_span, budget)))
    normalized = _normalize_command(site)
    if normalized.inert:
        return AdapterResult(candidates=tuple(candidates))
    if normalized.issue_span is not None:
        return AdapterResult(
            candidates=tuple(candidates),
            issues=_issues(_issue(site, normalized.issue_span, budget)),
        )
    if normalized.manager is None:
        return AdapterResult(candidates=tuple(candidates))

    ecosystem = (
        DependencyEcosystem.MAVEN
        if normalized.manager == b"mvn"
        else DependencyEcosystem(normalized.manager.decode("ascii"))
    )
    environment_candidates, environment_issue_span = _environment_candidates(
        normalized.environment,
        manager=ecosystem,
    )
    if environment_issue_span is not None:
        return AdapterResult(issues=_issues(_issue(site, environment_issue_span, budget)))
    candidates.extend(environment_candidates)
    parsed: list[DependencyCommandCandidate]
    issue_span: SourceSpan | None
    maven_settings: tuple[MavenSettingsReference, ...] = ()
    if normalized.manager in {b"npm", b"pnpm"}:
        parsed, issue_span = _npm_family(normalized.tokens, ecosystem)
    elif normalized.manager == b"yarn":
        parsed, issue_span = _yarn(normalized.tokens)
    elif normalized.manager == b"pip":
        parsed, issue_span = _pip(normalized.tokens)
    elif normalized.manager == b"poetry":
        parsed, issue_span = _poetry(normalized.tokens)
    elif normalized.manager == b"cargo":
        parsed, issue_span = _cargo(normalized.tokens)
    elif normalized.manager == b"uv":
        parsed, issue_span = _uv(normalized.tokens)
    else:
        parsed = []
        maven_settings, issue_span = _maven(normalized.tokens)
    if issue_span is not None:
        return AdapterResult(
            candidates=tuple(candidates),
            issues=_issues(_issue(site, issue_span, budget)),
        )
    candidates.extend(parsed)
    return AdapterResult(candidates=tuple(candidates), maven_settings=maven_settings)
