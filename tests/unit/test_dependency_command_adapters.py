# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct contracts plus end-to-end red gates for dependency-command adapters."""

from __future__ import annotations

import dataclasses
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from skillspector.artifacts import classify_artifact
from skillspector.dependency_command_adapters import adapt_command
from skillspector.dependency_source_types import (
    MAX_DEPENDENCY_SHELL_LOCALIZED_ISSUES,
    AssignmentSite,
    CommandProducerReachability,
    CommandResolutionKind,
    CommandSite,
    DependencyEcosystem,
    DependencySourceOperation,
    DependencySourceScope,
    DependencySourceSpan,
    DependencySourceSurface,
    DependencyWorkBudget,
    DependencyWorkResource,
    ShellIssueReason,
    SiteProvenance,
    SourceSpan,
    StaticValue,
)
from skillspector.dependency_sources import analyze_dependency_sources
from skillspector.nested_artifacts import is_executable_content

REPRESENTATIVE_IDS = (
    "npm-flags-before-operands",
    "yarn-flags-before-operands",
    "pnpm-unmodeled",
    "pip-short-option-bundling",
    "poetry-global-options-before-subcommand",
    "cargo-has-no-command-branch",
    "uv-entirely-uncovered",
    "maven-settings-file-flag-and-unrecognised-filename",
)


def _finding_rows() -> dict[str, dict[str, Any]]:
    path = Path(__file__).parents[1] / "nodes/analyzers/data/sc10_findings.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    return {row["id"]: row for row in rows}


def _gap_marks(row: dict[str, Any]) -> list[pytest.MarkDecorator]:
    marks = [pytest.mark.sc10_pr2]
    if row["status"] != "fixed" and os.getenv("SKILLSPECTOR_SC10_GAPS") != "enforce":
        marks.append(pytest.mark.xfail(strict=True, reason=f"SC10 gap: {row['id']}"))
    return marks


def _normalized_finding(finding: Any) -> dict[str, Any]:
    evidence = finding.evidence
    result = {
        "severity": finding.severity,
        "ecosystem": evidence["ecosystem"],
        "surface": evidence["surface"],
        "operation": evidence["operation"],
        "scope": evidence["scope"],
        "destination": evidence["destination"],
        "destination_status": evidence["destination_status"],
        "file": finding.file,
        "start_line": finding.start_line,
    }
    if finding.end_line is not None and finding.end_line != finding.start_line:
        result["end_line"] = finding.end_line
    return result


ROWS_BY_ID = _finding_rows()
MISSING_REPRESENTATIVES = set(REPRESENTATIVE_IDS) - set(ROWS_BY_ID)
assert not MISSING_REPRESENTATIVES

ADAPTER_PARAMETERS = [
    pytest.param(ROWS_BY_ID[identifier], id=identifier, marks=_gap_marks(ROWS_BY_ID[identifier]))
    for identifier in REPRESENTATIVE_IDS
]


@pytest.mark.parametrize("row", ADAPTER_PARAMETERS)
def test_dependency_command_adapter_contracts(row: dict[str, Any]) -> None:
    files = row["files"]
    raw_files = {path: content.encode("utf-8") for path, content in files.items()}
    executable_paths = frozenset(
        DependencySourceSpan(path=path, start_line=1, end_line=1).path
        for path in sorted(raw_files)
        if is_executable_content(path, raw_files[path])
    )
    analysis = analyze_dependency_sources(
        components=sorted(files),
        local_file_cache=files,
        raw_file_cache=raw_files,
        artifact_inventory=[classify_artifact(path, raw_files[path]) for path in sorted(raw_files)],
        budget=DependencyWorkBudget(),
        executable_paths=executable_paths,
    )

    actual = [
        _normalized_finding(finding) for finding in analysis.findings if finding.rule_id == "SC10"
    ]
    assert Counter(json.dumps(item, sort_keys=True) for item in actual) == Counter(
        json.dumps(item, sort_keys=True) for item in row["expected_sc10"]
    )
    assert analysis.limitations == ()
    assert row["status"] == "fixed", "unimplemented adapter contracts remain explicit red gates"


UNIT_ID = "0" * 32


def _span(index: int = 0) -> SourceSpan:
    return SourceSpan("scripts/setup.sh", index, index + 1, 1, 1)


def _assignment(name: str, value: bytes | None) -> AssignmentSite:
    return AssignmentSite(
        UNIT_ID,
        SiteProvenance.FILE_SUFFIX,
        _span(),
        name,
        StaticValue.exact(value) if value is not None else StaticValue.unknown(),
    )


def _site(
    argv: list[bytes | None],
    *,
    prefix: tuple[AssignmentSite, ...] = (),
    exported: tuple[AssignmentSite, ...] = (),
    resolution: CommandResolutionKind = CommandResolutionKind.EXTERNAL,
    producer: CommandProducerReachability = CommandProducerReachability.ACTIVE,
) -> CommandSite:
    values = tuple(
        StaticValue.exact(value) if value is not None else StaticValue.unknown() for value in argv
    )
    return CommandSite(
        UNIT_ID,
        SiteProvenance.FILE_SUFFIX,
        SourceSpan("scripts/setup.sh", 0, max(1, len(values)), 1, 1),
        values,
        argument_spans=tuple(_span(index) for index in range(len(values))),
        prefix_assignments=prefix,
        exported_assignments=exported,
        resolution=resolution,
        producer=producer,
    )


def _projection(result: Any) -> list[tuple[object, ...]]:
    return [
        (
            candidate.ecosystem,
            candidate.surface,
            candidate.operation,
            candidate.scope,
            candidate.destination.state.value,
            candidate.destination.exact_bytes,
        )
        for candidate in result.candidates
    ]


@pytest.mark.parametrize(
    "trailing",
    [
        (b"--location=project",),
        (b"--location", b"project"),
        (b"--userconfig", b"./.npmrc"),
    ],
)
def test_npm_completed_pairs_snapshot_scope_before_known_trailing_options(
    trailing: tuple[bytes, ...],
) -> None:
    result = adapt_command(
        _site(
            [
                b"npm",
                b"config",
                b"set",
                b"registry",
                b"https://npm.invalid",
                *trailing,
            ]
        ),
        budget=DependencyWorkBudget(),
    )

    assert _projection(result) == [
        (
            DependencyEcosystem.NPM,
            DependencySourceSurface.COMMAND,
            DependencySourceOperation.SET,
            DependencySourceScope.GLOBAL,
            "exact",
            b"https://npm.invalid",
        )
    ]
    assert result.issues == ()


@pytest.mark.parametrize("trailing", [(b"--location",), (b"--userconfig",)])
def test_npm_missing_trailing_option_operands_remain_limitations(
    trailing: tuple[bytes, ...],
) -> None:
    result = adapt_command(
        _site(
            [
                b"npm",
                b"config",
                b"set",
                b"registry",
                b"https://npm.invalid",
                *trailing,
            ]
        ),
        budget=DependencyWorkBudget(),
    )

    assert result.candidates == ()
    assert len(result.issues) == 1


@pytest.mark.parametrize(
    ("argv", "semantic"),
    [
        (
            [b"npm", b"config", b"set", b"--location=project", b"registry", b"https://npm.invalid"],
            (
                DependencyEcosystem.NPM,
                DependencySourceSurface.COMMAND,
                DependencySourceOperation.SET,
                DependencySourceScope.PROJECT,
                b"https://npm.invalid",
            ),
        ),
        (
            [
                b"yarn",
                b"config",
                b"set",
                b"-H",
                b"npmScopes.team.npmRegistryServer",
                b"https://yarn.invalid",
            ],
            (
                DependencyEcosystem.YARN,
                DependencySourceSurface.COMMAND,
                DependencySourceOperation.SET,
                DependencySourceScope.SCOPED,
                b"https://yarn.invalid",
            ),
        ),
        (
            [b"pnpm", b"config", b"set", b"registry", b"https://pnpm.invalid"],
            (
                DependencyEcosystem.PNPM,
                DependencySourceSurface.COMMAND,
                DependencySourceOperation.SET,
                DependencySourceScope.GLOBAL,
                b"https://pnpm.invalid",
            ),
        ),
        (
            [
                b"pip3.12",
                b"--isolated",
                b"config",
                b"--user",
                b"set",
                b"global.index-url",
                b"https://pip.invalid/simple",
            ],
            (
                DependencyEcosystem.PIP,
                DependencySourceSurface.COMMAND,
                DependencySourceOperation.SET,
                DependencySourceScope.GLOBAL,
                b"https://pip.invalid/simple",
            ),
        ),
        (
            [
                b"poetry",
                b"-C",
                b".",
                b"source",
                b"add",
                b"--priority",
                b"explicit",
                b"private",
                b"https://poetry.invalid/simple",
            ],
            (
                DependencyEcosystem.POETRY,
                DependencySourceSurface.COMMAND,
                DependencySourceOperation.ADD,
                DependencySourceScope.SOURCE,
                b"https://poetry.invalid/simple",
            ),
        ),
        (
            [
                b"cargo",
                b"--config",
                b'registries.private.index="https://cargo.invalid/index"',
                b"build",
            ],
            (
                DependencyEcosystem.CARGO,
                DependencySourceSurface.INVOCATION,
                DependencySourceOperation.USE,
                DependencySourceScope.REGISTRY,
                b"https://cargo.invalid/index",
            ),
        ),
        (
            [b"uv", b"pip", b"install", b"--index-url", b"https://uv.invalid/simple", b"thing"],
            (
                DependencyEcosystem.UV,
                DependencySourceSurface.INVOCATION,
                DependencySourceOperation.USE,
                DependencySourceScope.INVOCATION,
                b"https://uv.invalid/simple",
            ),
        ),
        (
            [b"python3", b"-m", b"pip", b"install", b"-ihttps://python.invalid/simple", b"thing"],
            (
                DependencyEcosystem.PIP,
                DependencySourceSurface.INVOCATION,
                DependencySourceOperation.USE,
                DependencySourceScope.INVOCATION,
                b"https://python.invalid/simple",
            ),
        ),
    ],
)
def test_direct_manager_grammars_are_typed_and_bounded(
    argv: list[bytes], semantic: tuple[object, ...]
) -> None:
    result = adapt_command(_site(argv), budget=DependencyWorkBudget())

    assert _projection(result) == [(*semantic[:-1], "exact", semantic[-1])]
    assert result.issues == ()


def test_npm_multi_pair_and_scoped_keys_preserve_each_destination() -> None:
    result = adapt_command(
        _site(
            [
                b"npm",
                b"c",
                b"set",
                b"registry",
                b"https://one.invalid",
                b"@team:registry",
                b"https://two.invalid",
            ]
        ),
        budget=DependencyWorkBudget(),
    )

    assert [candidate.scope for candidate in result.candidates] == [
        DependencySourceScope.GLOBAL,
        DependencySourceScope.SCOPED,
    ]
    assert [candidate.destination.exact_bytes for candidate in result.candidates] == [
        b"https://one.invalid",
        b"https://two.invalid",
    ]
    assert result.candidates[0].span.start_line == result.candidates[1].span.start_line == 1
    assert result.candidates[0].span.start_byte != result.candidates[1].span.start_byte


def test_npm_config_set_accepts_key_equals_value_multi_pair_form() -> None:
    result = adapt_command(
        _site(
            [
                b"npm",
                b"config",
                b"set",
                b"registry=https://one.invalid",
                b"@team:registry=https://two.invalid",
            ]
        ),
        budget=DependencyWorkBudget(),
    )

    assert [candidate.scope for candidate in result.candidates] == [
        DependencySourceScope.GLOBAL,
        DependencySourceScope.SCOPED,
    ]
    assert [candidate.destination.exact_bytes for candidate in result.candidates] == [
        b"https://one.invalid",
        b"https://two.invalid",
    ]


@pytest.mark.parametrize(
    "argv",
    [
        [b"/usr/local/bin/npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [
            b"env",
            b"-u",
            b"IGNORED",
            b"--",
            b"npm",
            b"config",
            b"set",
            b"registry",
            b"https://x.invalid",
        ],
        [b"sudo", b"-k", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"command", b"-p", b"--", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"exec", b"-a", b"argv0", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"nohup", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"nice", b"-n", b"4", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [
            b"timeout",
            b"-k",
            b"1s",
            b"10s",
            b"npm",
            b"config",
            b"set",
            b"registry",
            b"https://x.invalid",
        ],
        [b"setsid", b"--", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"stdbuf", b"-o", b"L", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"corepack", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"npx", b"--yes", b"pnpm", b"config", b"set", b"registry", b"https://x.invalid"],
    ],
)
def test_supported_wrapper_options_preserve_one_literal_manager_sink(argv: list[bytes]) -> None:
    result = adapt_command(_site(argv), budget=DependencyWorkBudget())

    assert [candidate.destination.exact_bytes for candidate in result.candidates] == [
        b"https://x.invalid"
    ]
    assert result.issues == ()


@pytest.mark.parametrize(
    "argv",
    [
        [b"env", b"-S", b"npm config set registry https://x.invalid"],
        [b"sudo", b"--mystery", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"xargs", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"corepack", b"unknown", b"https://x.invalid"],
        [b"npx", b"--mystery", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
    ],
)
def test_unknown_wrapper_arity_is_one_localized_limitation(argv: list[bytes]) -> None:
    result = adapt_command(_site(argv), budget=DependencyWorkBudget())

    assert result.candidates == ()
    assert [issue.reason for issue in result.issues] == [ShellIssueReason.UNSUPPORTED_SEMANTICS]


@pytest.mark.parametrize(
    "argv",
    [
        [b"env", b"--", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"sudo", b"--", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"command", b"--", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"exec", b"--", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"nohup", b"--", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"nice", b"--", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"timeout", b"--", b"5s", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"setsid", b"--", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"stdbuf", b"--", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"corepack", b"--", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"npx", b"--", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
    ],
)
def test_each_transparent_wrapper_has_an_explicit_terminator_case(argv: list[bytes]) -> None:
    result = adapt_command(_site(argv), budget=DependencyWorkBudget())

    assert len(result.candidates) == 1
    assert result.issues == ()


@pytest.mark.parametrize(
    "argv",
    [
        [b"env", b"-u"],
        [b"sudo", b"-u"],
        [b"exec", b"-a"],
        [b"nice", b"-n"],
        [b"timeout", b"-k"],
        [b"stdbuf", b"-o"],
        [b"corepack"],
        [b"npx"],
    ],
)
def test_each_operand_wrapper_rejects_a_missing_operand(argv: list[bytes]) -> None:
    result = adapt_command(_site(argv), budget=DependencyWorkBudget())

    assert result.candidates == ()
    assert len(result.issues) == 1


@pytest.mark.parametrize(
    "argv",
    [
        [b"env", b"--unset=", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"env", b"--chdir=", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"sudo", b"--user=", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [
            b"nice",
            b"--adjustment=",
            b"npm",
            b"config",
            b"set",
            b"registry",
            b"https://x.invalid",
        ],
        [
            b"timeout",
            b"--kill-after=",
            b"5s",
            b"npm",
            b"config",
            b"set",
            b"registry",
            b"https://x.invalid",
        ],
        [b"stdbuf", b"--output=", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"npx", b"--package=", b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
    ],
)
def test_attached_empty_wrapper_operands_stop_before_the_manager_sink(
    argv: list[bytes],
) -> None:
    result = adapt_command(_site(argv), budget=DependencyWorkBudget())

    assert result.candidates == ()
    assert [issue.reason for issue in result.issues] == [ShellIssueReason.UNSUPPORTED_SEMANTICS]


@pytest.mark.parametrize(
    "wrapper",
    [
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
    ],
)
def test_each_transparent_wrapper_rejects_unknown_option_arity(wrapper: bytes) -> None:
    result = adapt_command(
        _site(
            [
                wrapper,
                b"--mystery",
                b"npm",
                b"config",
                b"set",
                b"registry",
                b"https://x.invalid",
            ]
        ),
        budget=DependencyWorkBudget(),
    )

    assert result.candidates == ()
    assert len(result.issues) == 1


@pytest.mark.parametrize(
    "argv",
    [
        [b"npm", b"config", b"get", b"registry"],
        [b"npm", b"config", b"delete", b"registry"],
        [b"yarn", b"why", b"https://x.invalid"],
        [b"yarn", b"config", b"unset", b"npmRegistryServer"],
        [b"pip", b"config", b"list"],
        [b"pip", b"config", b"unset", b"global.index-url"],
        [b"poetry", b"publish", b"--repository", b"https://x.invalid"],
        [b"mvn", b"-DaltDeploymentRepository=x::default::https://x.invalid", b"deploy"],
        [b"echo", b"https://x.invalid"],
    ],
)
def test_inert_actions_and_unrecognized_url_tokens_are_clean(argv: list[bytes]) -> None:
    result = adapt_command(_site(argv), budget=DependencyWorkBudget())

    assert result.candidates == result.issues == result.maven_settings == ()


@pytest.mark.parametrize(
    "argv",
    [
        [b"npm", b"run", b"build", b"--", b"--registry=https://x.invalid"],
        [b"npm", b"config", b"set", b"//x.invalid/:_authToken", b"secret"],
        [b"yarn", b"publish", b"--registry", b"https://x.invalid"],
        [b"mvn", b"-Dmaven.repo.remote=https://x.invalid", b"install"],
    ],
)
def test_manager_url_decoys_outside_recognized_resolution_sinks_are_inert(
    argv: list[bytes],
) -> None:
    result = adapt_command(_site(argv), budget=DependencyWorkBudget())

    assert result.candidates == result.issues == result.maven_settings == ()


def test_export_and_reaching_prefix_facts_are_named_environment_sinks() -> None:
    exported = adapt_command(
        _site(
            [b"export", b"NPM_CONFIG_REGISTRY=value"],
            exported=(_assignment("NPM_CONFIG_REGISTRY", b"https://export.invalid"),),
        ),
        budget=DependencyWorkBudget(),
    )
    prefixed = adapt_command(
        _site(
            [b"cargo", b"build"],
            prefix=(_assignment("CARGO_REGISTRIES_PRIVATE_INDEX", b"https://cargo.invalid"),),
        ),
        budget=DependencyWorkBudget(),
    )

    assert [candidate.ecosystem for candidate in exported.candidates] == [DependencyEcosystem.NPM]
    assert [candidate.ecosystem for candidate in prefixed.candidates] == [DependencyEcosystem.CARGO]
    assert all(
        candidate.surface is DependencySourceSurface.ENVIRONMENT
        for candidate in (*exported.candidates, *prefixed.candidates)
    )


def test_env_clear_and_sudo_reset_environment_conservatively() -> None:
    prefix = (_assignment("NPM_CONFIG_REGISTRY", b"https://prefix.invalid"),)
    env = adapt_command(
        _site([b"env", b"-i", b"npm", b"install"], prefix=prefix),
        budget=DependencyWorkBudget(),
    )
    sudo = adapt_command(
        _site([b"sudo", b"npm", b"install"], prefix=prefix),
        budget=DependencyWorkBudget(),
    )
    preserved = adapt_command(
        _site([b"sudo", b"-E", b"npm", b"install"], prefix=prefix),
        budget=DependencyWorkBudget(),
    )

    assert env.candidates == sudo.candidates == ()
    assert len(preserved.candidates) == 1


def test_env_terminator_still_consumes_command_environment_assignments() -> None:
    result = adapt_command(
        _site(
            [
                b"env",
                b"--",
                b"NPM_CONFIG_REGISTRY=https://inline.invalid",
                b"npm",
                b"install",
            ]
        ),
        budget=DependencyWorkBudget(),
    )

    assert [candidate.destination.exact_bytes for candidate in result.candidates] == [
        b"https://inline.invalid"
    ]


def test_sudo_explicit_assignment_overrides_a_preserved_prefix_value() -> None:
    result = adapt_command(
        _site(
            [
                b"sudo",
                b"-E",
                b"NPM_CONFIG_REGISTRY=https://inline.invalid",
                b"npm",
                b"install",
            ],
            prefix=(_assignment("NPM_CONFIG_REGISTRY", b"https://stale.invalid"),),
        ),
        budget=DependencyWorkBudget(),
    )

    assert [candidate.destination.exact_bytes for candidate in result.candidates] == [
        b"https://inline.invalid"
    ]


@pytest.mark.parametrize(
    "argv",
    [
        [b"npm", b"install", b"--", b"--registry=https://forwarded.invalid"],
        [b"pip", b"install", b"--", b"--index-url", b"https://forwarded.invalid"],
        [b"uv", b"pip", b"install", b"--", b"--index-url", b"https://forwarded.invalid"],
        [b"cargo", b"build", b"--", b"--config", b"registries.x.index=https://forwarded.invalid"],
        [b"mvn", b"install", b"--", b"--settings", b"forwarded-settings.xml"],
    ],
)
def test_manager_terminator_prevents_forwarded_flags_becoming_source_sinks(
    argv: list[bytes],
) -> None:
    result = adapt_command(_site(argv), budget=DependencyWorkBudget())

    assert result.candidates == result.issues == result.maven_settings == ()


@pytest.mark.parametrize(
    "argv",
    [
        [b"npm", b"install", b"--mystery", b"operand", b"--registry", b"https://x.invalid"],
        [b"pip", b"install", b"--mystery", b"operand", b"--index-url", b"https://x.invalid"],
        [b"uv", b"pip", b"install", b"--mystery", b"operand", b"--index-url", b"https://x.invalid"],
        [
            b"cargo",
            b"--mystery",
            b"operand",
            b"--config",
            b"registries.x.index=https://x.invalid",
            b"build",
        ],
        [b"mvn", b"--mystery", b"operand", b"--settings", b"ci-settings.xml", b"install"],
    ],
)
def test_unknown_manager_option_arity_beside_a_source_flag_is_one_limitation(
    argv: list[bytes],
) -> None:
    result = adapt_command(_site(argv), budget=DependencyWorkBudget())

    assert result.candidates == result.maven_settings == ()
    assert len(result.issues) == 1


def test_manager_source_flag_before_terminator_is_still_recognized_once() -> None:
    result = adapt_command(
        _site(
            [
                b"npm",
                b"install",
                b"--registry=https://kept.invalid",
                b"--",
                b"--registry=https://forwarded.invalid",
            ]
        ),
        budget=DependencyWorkBudget(),
    )

    assert [candidate.destination.exact_bytes for candidate in result.candidates] == [
        b"https://kept.invalid"
    ]
    assert result.issues == ()


def test_dynamic_sink_is_unresolved_but_unknown_relevant_option_never_guesses() -> None:
    dynamic = adapt_command(
        _site([b"pip", b"install", b"--index-url", None, b"thing"]),
        budget=DependencyWorkBudget(),
    )
    ambiguous = adapt_command(
        _site([b"npm", b"config", b"set", b"--mystery", b"registry", b"https://x.invalid"]),
        budget=DependencyWorkBudget(),
    )

    assert dynamic.candidates[0].destination == StaticValue.unknown()
    assert ambiguous.candidates == ()
    assert len(ambiguous.issues) == 1


@pytest.mark.parametrize(
    "argv",
    [
        [b"npm", b"config", b"set", b"registry", None],
        [b"yarn", b"config", b"set", b"npmRegistryServer", None],
        [b"pnpm", b"config", b"set", b"registry", None],
        [b"pip", b"config", b"set", b"global.index-url", None],
    ],
)
def test_static_config_key_with_dynamic_destination_is_one_unknown_candidate(
    argv: list[bytes | None],
) -> None:
    result = adapt_command(_site(argv), budget=DependencyWorkBudget())

    assert len(result.candidates) == 1
    assert result.candidates[0].destination == StaticValue.unknown()
    assert result.issues == ()


def test_poetry_source_add_terminator_preserves_positional_name_and_destination() -> None:
    result = adapt_command(
        _site([b"poetry", b"source", b"add", b"--", b"private", b"https://x.invalid"]),
        budget=DependencyWorkBudget(),
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].operation is DependencySourceOperation.ADD
    assert result.candidates[0].destination.exact_bytes == b"https://x.invalid"
    assert result.issues == ()


@pytest.mark.parametrize("assignment_field", ["prefix", "exported"])
@pytest.mark.parametrize("name", ["UV_INDEX", "UV_EXTRA_INDEX_URL"])
@pytest.mark.parametrize(
    ("value", "expected_destination", "expect_issue"),
    [
        (b"https://uv.invalid/simple", b"https://uv.invalid/simple", False),
        (b"private=https://uv.invalid/simple", b"https://uv.invalid/simple", False),
        (None, None, False),
        (b"https://one.invalid https://two.invalid", None, True),
        (b"", None, True),
        (b"=https://uv.invalid/simple", None, True),
        (b"private=", None, True),
    ],
)
def test_uv_documented_index_environment_surfaces_are_bounded_typed_candidates(
    assignment_field: str,
    name: str,
    value: bytes | None,
    expected_destination: bytes | None,
    expect_issue: bool,
) -> None:
    assignment = _assignment(name, value)
    site = (
        _site([b"uv", b"pip", b"install"], prefix=(assignment,))
        if assignment_field == "prefix"
        else _site([b"export"], exported=(assignment,))
    )

    result = adapt_command(site, budget=DependencyWorkBudget())

    if expect_issue:
        assert result.candidates == ()
        assert [issue.reason for issue in result.issues] == [ShellIssueReason.UNSUPPORTED_SEMANTICS]
        return
    assert len(result.candidates) == 1
    assert result.candidates[0].ecosystem is DependencyEcosystem.UV
    assert result.candidates[0].surface is DependencySourceSurface.ENVIRONMENT
    if expected_destination is None:
        assert result.candidates[0].destination == StaticValue.unknown()
    else:
        assert result.candidates[0].destination.exact_bytes == expected_destination
    assert result.issues == ()


def test_uv_named_index_operand_extracts_only_the_destination() -> None:
    result = adapt_command(
        _site(
            [
                b"uv",
                b"lock",
                b"--index",
                b"private=https://uv.invalid/simple",
            ]
        ),
        budget=DependencyWorkBudget(),
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].destination.exact_bytes == b"https://uv.invalid/simple"


def test_uv_url_only_and_unresolved_index_operands_preserve_typed_values() -> None:
    exact = adapt_command(
        _site([b"uv", b"lock", b"--index", b"https://uv.invalid/simple"]),
        budget=DependencyWorkBudget(),
    )
    unresolved = adapt_command(
        _site([b"uv", b"lock", b"--index", None]),
        budget=DependencyWorkBudget(),
    )

    assert exact.candidates[0].destination.exact_bytes == b"https://uv.invalid/simple"
    assert unresolved.candidates[0].destination == StaticValue.unknown()
    assert exact.issues == unresolved.issues == ()


def test_uv_similar_non_source_environment_and_option_names_are_inert() -> None:
    environment = adapt_command(
        _site(
            [b"export"],
            exported=(_assignment("UV_INDEX_STRATEGY", b"unsafe-best-match"),),
        ),
        budget=DependencyWorkBudget(),
    )
    option = adapt_command(
        _site([b"uv", b"lock", b"--index-strategy", b"unsafe-best-match"]),
        budget=DependencyWorkBudget(),
    )

    assert environment.candidates == environment.issues == ()
    assert option.candidates == option.issues == ()


@pytest.mark.parametrize(
    "argv",
    [
        [b"npm", b"--userconfig=", b"config", b"set", b"registry", b"https://x.invalid"],
        [b"npm", b"install", b"--registry="],
        [b"pip", b"--python=", b"install", b"--index-url", b"https://x.invalid"],
        [b"pip", b"install", b"--index-url="],
        [b"poetry", b"--directory=", b"source", b"add", b"private", b"https://x.invalid"],
        [b"cargo", b"--config=", b"build"],
        [b"uv", b"pip", b"install", b"--index-url="],
        [b"mvn", b"--settings=", b"install"],
    ],
)
def test_empty_attached_option_operands_are_missing_operand_limitations(
    argv: list[bytes],
) -> None:
    result = adapt_command(_site(argv), budget=DependencyWorkBudget())

    assert result.candidates == result.maven_settings == ()
    assert len(result.issues) == 1


@pytest.mark.parametrize(
    "argv",
    [
        [b"npm", b"help", b"install", b"--registry=https://x.invalid"],
        [b"yarn", b"help", b"install", b"--registry=https://x.invalid"],
        [b"pnpm", b"help", b"install", b"--registry=https://x.invalid"],
        [b"pip", b"help", b"install", b"--index-url", b"https://x.invalid"],
        [b"uv", b"help", b"pip", b"install", b"--index-url", b"https://x.invalid"],
        [b"poetry", b"config", b"repositories.private.username", b"https://x.invalid"],
    ],
)
def test_help_and_credential_value_decoys_are_inert(argv: list[bytes]) -> None:
    result = adapt_command(_site(argv), budget=DependencyWorkBudget())

    assert result.candidates == result.issues == result.maven_settings == ()


@pytest.mark.parametrize(
    "argv",
    [
        [b"npm", b"config", b"mystery", b"registry", b"https://x.invalid"],
        [b"yarn", b"config", b"mystery", b"registry", b"https://x.invalid"],
        [b"pip", b"config", b"mystery", b"global.index-url", b"https://x.invalid"],
    ],
)
def test_unknown_manager_actions_are_limitations_not_guessed_sinks(argv: list[bytes]) -> None:
    result = adapt_command(_site(argv), budget=DependencyWorkBudget())

    assert result.candidates == ()
    assert len(result.issues) == 1


def test_resolution_and_producer_facts_prevent_shadowed_or_inert_findings() -> None:
    for site in (
        _site(
            [b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
            resolution=CommandResolutionKind.FUNCTION,
        ),
        _site(
            [b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
            producer=CommandProducerReachability.INERT,
        ),
    ):
        result = adapt_command(site, budget=DependencyWorkBudget())
        assert result.candidates == result.issues == ()

    for site in (
        _site(
            [b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
            resolution=CommandResolutionKind.AMBIGUOUS,
        ),
        _site(
            [b"npm", b"config", b"set", b"registry", b"https://x.invalid"],
            producer=CommandProducerReachability.AMBIGUOUS,
        ),
    ):
        result = adapt_command(site, budget=DependencyWorkBudget())
        assert result.candidates == ()
        assert len(result.issues) == 1


def test_adapter_drops_resource_issue_when_reserved_issue_was_already_claimed() -> None:
    budget = DependencyWorkBudget()
    assert budget.charge_shell_issues(MAX_DEPENDENCY_SHELL_LOCALIZED_ISSUES - 1) is None
    site = _site([b"sudo", b"--mystery", b"npm", b"install"])

    first = adapt_command(site, budget=budget)
    second = adapt_command(site, budget=budget)

    assert [issue.reason for issue in first.issues] == [ShellIssueReason.RESOURCE_LIMIT]
    assert second.issues == ()
    assert (
        budget.used(DependencyWorkResource.SHELL_LOCALIZED_ISSUES)
        == MAX_DEPENDENCY_SHELL_LOCALIZED_ISSUES
    )


def test_maven_adapter_returns_only_literal_settings_references() -> None:
    exact = adapt_command(
        _site([b"mvn", b"-s", b"ci-settings.xml", b"install"]),
        budget=DependencyWorkBudget(),
    )
    dynamic = adapt_command(
        _site([b"mvn", b"--settings", None, b"install"]),
        budget=DependencyWorkBudget(),
    )

    assert [reference.path.exact_bytes for reference in exact.maven_settings] == [
        b"ci-settings.xml"
    ]
    assert exact.candidates == exact.issues == ()
    assert dynamic.maven_settings == ()
    assert len(dynamic.issues) == 1


@pytest.mark.parametrize(
    "argv",
    [
        [b"mvn", b"install", b"-s", b"ci-settings.xml"],
        [b"mvn", b"--unknown", b"-s", b"ci-settings.xml", b"install"],
        [b"mvn", b"-sci-settings.xml", b"install"],
        [b"mvn", b"--settings=ci-settings.xml", b"install"],
        [b"mvn", b"-s", b"one.xml", b"-s", b"two.xml", b"install"],
    ],
)
def test_maven_settings_reference_requires_one_leading_separate_operand(
    argv: list[bytes],
) -> None:
    result = adapt_command(_site(argv), budget=DependencyWorkBudget())

    assert result.maven_settings == result.candidates == ()
    assert len(result.issues) == 1


def test_adapter_result_repr_never_retains_exact_destination_bytes() -> None:
    secret = b"https://user:secret@packages.invalid/path?token=abc"
    result = adapt_command(
        _site([b"npm", b"config", b"set", b"registry", secret]),
        budget=DependencyWorkBudget(),
    )

    assert secret.decode() not in repr(result)
    assert not hasattr(result, "command")


def test_destination_candidate_keeps_the_typed_argument_span() -> None:
    site = _site([b"yarn", b"config", b"set", b"registry", b"https://x.invalid"])
    destination_span = SourceSpan("scripts/setup.sh", 40, 58, 4, 4)
    site = dataclasses.replace(
        site,
        argument_spans=(*site.argument_spans[:-1], destination_span),
    )

    result = adapt_command(site, budget=DependencyWorkBudget())

    assert result.candidates[0].span == destination_span
