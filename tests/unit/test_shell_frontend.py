# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for shell semantics that stay explicit limitations."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import Any

import pytest

import skillspector.dependency_source_types as dependency_types
import skillspector.shell_frontend as shell_frontend
from skillspector.artifacts import classify_artifact
from skillspector.dependency_source_types import (
    DependencySourceLimitationReason,
    DependencyWorkBudget,
)
from skillspector.dependency_sources import analyze_dependency_sources


def _extract(
    path: str,
    raw: bytes,
    *,
    executable_paths: frozenset[str] = frozenset(),
    budget: DependencyWorkBudget | None = None,
) -> Any:
    return shell_frontend.extract_shell_units(
        path,
        raw,
        executable_paths=executable_paths,
        budget=budget or DependencyWorkBudget(),
    )


def _analyze(
    raw: bytes,
    *,
    path: str = "scripts/lower.sh",
    budget: DependencyWorkBudget | None = None,
) -> tuple[Any, DependencyWorkBudget, Any]:
    active_budget = budget or DependencyWorkBudget()
    extraction = _extract(path, raw, budget=active_budget)
    assert len(extraction.units) == 1
    unit = extraction.units[0]
    return (
        shell_frontend.analyze_shell_unit(unit, budget=active_budget),
        active_budget,
        unit,
    )


def _analyze_private(
    raw: bytes,
    *,
    path: str = "scripts/state.sh",
    budget: DependencyWorkBudget | None = None,
) -> tuple[Any, DependencyWorkBudget, Any]:
    active_budget = budget or DependencyWorkBudget()
    extraction = _extract(path, raw, budget=active_budget)
    assert len(extraction.units) == 1
    unit = extraction.units[0]
    return (
        shell_frontend._analyze_shell_unit(unit, budget=active_budget),
        active_budget,
        unit,
    )


def _argv_bytes(command: Any) -> tuple[bytes | None, ...]:
    return tuple(
        value.exact_bytes if value.state is dependency_types.StaticValueState.EXACT else None
        for value in command.argv
    )


class _ObservedExecutablePaths(frozenset[str]):
    """Immutable inventory that records accidental whole-set iteration."""

    iteration_count: int
    membership_count: int

    def __new__(cls, values: Iterable[str]) -> _ObservedExecutablePaths:
        instance = super().__new__(cls, values)
        instance.iteration_count = 0
        instance.membership_count = 0
        return instance

    def __iter__(self) -> Iterator[str]:
        self.iteration_count += 1
        return super().__iter__()

    def __contains__(self, value: object) -> bool:
        self.membership_count += 1
        return super().__contains__(value)


class _ObservedDraftList(list[Any]):
    """Counts draft iteration without relying on a wall-clock threshold."""

    iterated_items: int = 0

    def __iter__(self) -> Iterator[Any]:
        for item in super().__iter__():
            self.iterated_items += 1
            yield item


def _gap_marks(case: dict[str, Any]) -> list[pytest.MarkDecorator]:
    marks = [pytest.mark.sc10_pr2]
    if case["status"] != "fixed" and os.getenv("SKILLSPECTOR_SC10_GAPS") != "enforce":
        marks.append(pytest.mark.xfail(strict=True, reason=f"SC10 gap: {case['id']}"))
    return marks


UNSUPPORTED_CASES = [
    {
        "id": "xargs-manager-construction-limitation",
        "status": "unfixed",
        "source": (
            "#!/bin/bash\n"
            "printf '%s\\n' 'config set registry https://packages.example.invalid' "
            "| xargs npm\n"
        ),
    },
    {
        "id": "env-s-split-string-limitation",
        "status": "unfixed",
        "source": (
            "#!/bin/bash\nenv -S 'npm config set registry https://packages.example.invalid'\n"
        ),
    },
    {
        "id": "data-to-shell-pipeline-limitation",
        "status": "unfixed",
        "source": (
            "#!/bin/bash\n"
            "printf '%s\\n' 'npm config set registry https://packages.example.invalid' "
            "| sh\n"
        ),
    },
]

UNSUPPORTED_SEMANTICS = [
    pytest.param(case, id=case["id"], marks=_gap_marks(case)) for case in UNSUPPORTED_CASES
]


@pytest.mark.parametrize("case", UNSUPPORTED_SEMANTICS)
def test_unsupported_shell_semantics_are_localized_limitations(case: dict[str, Any]) -> None:
    path = "scripts/setup.sh"
    source = case["source"]
    raw = source.encode("utf-8")
    analysis = analyze_dependency_sources(
        components=[path],
        local_file_cache={path: source},
        raw_file_cache={path: raw},
        artifact_inventory=[classify_artifact(path, raw)],
        budget=DependencyWorkBudget(),
        executable_paths=frozenset({path}),
    )

    assert [finding for finding in analysis.findings if finding.rule_id == "SC10"] == []
    assert [
        (
            limitation.reason,
            limitation.path,
            limitation.start_line,
            limitation.end_line,
        )
        for limitation in analysis.limitations
    ] == [(DependencySourceLimitationReason.PARSE_INCOMPLETE, path, 2, 2)]
    assert case["status"] == "fixed", "unimplemented shell contracts remain explicit red gates"


@pytest.mark.parametrize(
    ("path", "raw", "executable_paths", "dialect", "provenance"),
    [
        (
            "scripts/setup.bash",
            b"printf ok\n",
            frozenset(),
            "bash",
            "file_suffix",
        ),
        (
            "scripts/setup.sh",
            b"printf ok\n",
            frozenset(),
            "sh",
            "file_suffix",
        ),
        (
            "scripts/setup.txt",
            b"#!/usr/bin/env dash\nprintf ok\n",
            frozenset(),
            "dash",
            "shebang",
        ),
        (
            "bundle.zip!/bin/setup",
            b"#!/bin/bash\nprintf ok\n",
            frozenset({"bundle.zip!/bin/setup"}),
            "bash",
            "shebang",
        ),
        (
            "bin/setup",
            b"#!/usr/bin/env -S bash -eu\nprintf ok\n",
            frozenset({"bin/setup"}),
            "bash",
            "shebang",
        ),
    ],
)
def test_standalone_unit_extraction_uses_only_supported_suffixes_and_shebangs(
    path: str,
    raw: bytes,
    executable_paths: frozenset[str],
    dialect: str,
    provenance: str,
) -> None:
    result = _extract(path, raw, executable_paths=executable_paths)

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.raw_bytes == raw
    assert unit.dialect.value == dialect
    assert unit.kind is dependency_types.ShellUnitKind.STANDALONE
    assert unit.provenance.value == provenance
    last_line_start = raw.rfind(b"\n", 0, max(0, len(raw) - 1)) + 1
    assert unit.origin_span == dependency_types.SourceSpan(
        path,
        0,
        len(raw),
        1,
        max(1, raw.count(b"\n") + (0 if raw.endswith(b"\n") else 1)),
        start_column=0,
        end_column=len(raw) - last_line_start,
    )
    assert result.issues == ()


def test_executable_only_without_supported_dialect_is_applicable_but_not_bash() -> None:
    path = "bundle.zip!/bin/setup"
    result = _extract(
        path,
        b"printf ok\n",
        executable_paths=frozenset({path}),
    )

    assert result.units == ()
    assert [(issue.reason, issue.outcome, issue.span.path) for issue in result.issues] == [
        (
            dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS,
            dependency_types.ShellWorkOutcome.PARTIAL,
            path,
        )
    ]


@pytest.mark.parametrize(
    ("path", "raw"),
    [
        ("setup.zsh", b"printf ok\n"),
        ("setup.envrc", b"printf ok\n"),
        ("setup.ksh", b"printf ok\n"),
        ("Dockerfile", b"FROM scratch\nRUN printf ok\n"),
        ("build/Makefile", b"all:\n\tprintf ok\n"),
    ],
)
def test_out_of_gate_executable_dialects_remain_typed_limitations(
    path: str,
    raw: bytes,
) -> None:
    result = _extract(path, raw)

    assert result.units == ()
    assert [issue.reason for issue in result.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]


@pytest.mark.parametrize(
    ("path", "raw"),
    [
        ("Dockerfile", b"FROM scratch\nCOPY . /src\n"),
        ("build/Makefile", b"all: generated.txt\n"),
    ],
)
def test_out_of_gate_container_and_make_files_without_executable_units_are_inert(
    path: str,
    raw: bytes,
) -> None:
    result = _extract(path, raw)

    assert result.units == ()
    assert result.issues == ()


def test_markdown_fences_honor_delimiter_length_indentation_and_info_token() -> None:
    raw = (
        b"heading\n"
        b"  ````BASH linenums\r\n"
        b"printf first\r\n"
        b"```\r\n"
        b"  ````\r\n"
        b"~~~shell-script\n"
        b"printf second\n"
        b"~~~~\n"
        b"```console\n"
        b"printf third\n"
        b"```\n"
    )

    result = _extract("docs/guide.md", raw)

    assert [unit.raw_bytes for unit in result.units] == [
        b"printf first\r\n```\r\n",
        b"printf second\n",
        b"printf third\n",
    ]
    assert [unit.dialect for unit in result.units] == [
        dependency_types.ShellDialect.BASH,
        dependency_types.ShellDialect.SH,
        dependency_types.ShellDialect.SH,
    ]
    assert all(
        unit.provenance is dependency_types.SiteProvenance.MARKDOWN_FENCE for unit in result.units
    )
    assert result.issues == ()


def test_markdown_fence_map_preserves_multibyte_crlf_physical_byte_columns() -> None:
    raw = "intro\r\n```bash\r\né npm\r\n```\r\n".encode()
    result = _extract("docs/guide.md", raw)
    unit = result.units[0]
    command_start = unit.raw_bytes.index(b"npm")

    mapped = unit.source_map.map_range(command_start, command_start + 3)

    physical_start = raw.index(b"npm")
    assert mapped == dependency_types.SourceSpan(
        "docs/guide.md",
        physical_start,
        physical_start + 3,
        3,
        3,
        start_column=3,
        end_column=6,
    )
    assert len(unit.source_map.entries) == 1


def test_repeated_extraction_produces_equal_opaque_unit_identities() -> None:
    raw = b"```bash\nprintf ok\n```\n"

    first = _extract("docs/guide.md", raw)
    second = _extract("docs/guide.md", raw)

    assert [unit.unit_id for unit in first.units] == [unit.unit_id for unit in second.units]


def test_markdown_does_not_infer_shell_inside_untagged_indented_or_non_shell_fences() -> None:
    raw = (
        b"````python\n"
        b"```bash\n"
        b"printf hidden\n"
        b"```\n"
        b"````\n"
        b"    ```bash\n"
        b"    printf indented\n"
        b"    ```\n"
        b"```text\n"
        b"printf text\n"
        b"```\n"
        b"```\n"
        b"#!/bin/bash\n"
        b"printf untagged\n"
        b"```\n"
    )

    result = _extract("docs/guide.md", raw)

    assert result.units == ()
    assert [issue.reason for issue in result.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]


def test_unclosed_relevant_markdown_fence_is_bounded_and_localized() -> None:
    raw = b"before\n~~~Sh\nprintf ok\n"

    result = _extract("docs/guide.md", raw)

    assert [unit.raw_bytes for unit in result.units] == [b"printf ok\n"]
    assert [
        (issue.reason, issue.span.start_line, issue.span.end_line) for issue in result.issues
    ] == [(dependency_types.ShellIssueReason.SYNTAX_ERROR, 2, 3)]


def test_invalid_utf8_in_relevant_shell_input_yields_only_a_sanitized_typed_issue() -> None:
    raw = b"#!/bin/bash\nprintf token-51e2\xff\n"

    result = _extract("scripts/setup", raw, executable_paths=frozenset({"scripts/setup"}))

    assert result.units == ()
    assert [issue.reason for issue in result.issues] == [
        dependency_types.ShellIssueReason.SYNTAX_ERROR
    ]
    assert "token-51e2" not in repr(result)


def test_malformed_markdown_units_consume_capacity_before_the_limit_issue() -> None:
    raw = b"".join(b"```bash\n\xff\n```\n" for _ in range(257))
    budget = DependencyWorkBudget()

    result = _extract("docs/malformed.md", raw, budget=budget)

    assert result.units == ()
    assert [issue.reason for issue in result.issues[:256]] == [
        dependency_types.ShellIssueReason.SYNTAX_ERROR
    ] * 256
    assert result.issues[256].reason is dependency_types.ShellIssueReason.RESOURCE_LIMIT
    assert result.issues[256].outcome is dependency_types.ShellWorkOutcome.PARTIAL
    assert result.issues[256].exhaustion == dependency_types.DependencyWorkExhaustion(
        dependency_types.DependencyWorkResource.SHELL_UNITS,
        257,
        256,
    )
    assert (
        budget.for_file("docs/malformed.md").used(
            dependency_types.DependencyWorkResource.SHELL_UNITS
        )
        == 256
    )


@pytest.mark.parametrize(
    "raw",
    [b"printf a\x00b\n", b"printf first\rprintf second\fvalue"],
)
def test_nul_form_feed_and_lone_cr_are_preserved_without_invented_line_boundaries(
    raw: bytes,
) -> None:
    result = _extract("scripts/setup.sh", raw)

    assert [unit.raw_bytes for unit in result.units] == [raw]
    assert result.issues == ()


@pytest.mark.timeout(10)
def test_thousands_of_shell_fences_retain_exact_capacity_and_one_resource_issue() -> None:
    raw = b"".join(b"```bash\nprintf ok\n```\n" for _ in range(4_096))
    budget = DependencyWorkBudget()

    result = _extract("docs/many.md", raw, budget=budget)

    assert len(result.units) == 256
    assert [issue.reason for issue in result.issues] == [
        dependency_types.ShellIssueReason.RESOURCE_LIMIT
    ]
    assert (
        budget.for_file("docs/many.md").used(dependency_types.DependencyWorkResource.SHELL_UNITS)
        == 256
    )


@pytest.mark.timeout(10)
def test_operator_dense_many_short_commands_stop_at_ir_limit_with_truthful_issue() -> None:
    raw = b"a&&b||c|d;" * 15_000 + b"\n"
    budget = DependencyWorkBudget()

    result, _, unit = _analyze(raw, path="scripts/operator-dense.sh", budget=budget)

    assert result.commands == ()
    assert 1 <= len(result.issues) <= 2
    expected_exhaustion = dependency_types.DependencyWorkExhaustion(
        dependency_types.DependencyWorkResource.RETAINED_SHELL_IR,
        dependency_types.MAX_DEPENDENCY_RETAINED_SHELL_IR + 1,
        dependency_types.MAX_DEPENDENCY_RETAINED_SHELL_IR,
    )
    assert {issue.reason for issue in result.issues} == {
        dependency_types.ShellIssueReason.RESOURCE_LIMIT
    }
    assert {issue.exhaustion for issue in result.issues} == {expected_exhaustion}
    assert (
        budget.for_file(unit.origin_span.path).used_for_unit(
            unit, dependency_types.DependencyWorkResource.RETAINED_SHELL_IR
        )
        == dependency_types.MAX_DEPENDENCY_RETAINED_SHELL_IR
    )
    assert [item.outcome for item in result.work_items] == [
        dependency_types.ShellWorkOutcome.PARTIAL
    ]


def test_extraction_requires_normalized_paths_immutable_inventory_and_canonical_bytes() -> None:
    with pytest.raises(ValueError):
        _extract("./scripts/setup.sh", b"printf ok\n")
    with pytest.raises(ValueError):
        shell_frontend.extract_shell_units(
            "scripts/setup.sh",
            b"printf ok\n",
            executable_paths={"scripts/setup.sh"},
            budget=DependencyWorkBudget(),
        )
    with pytest.raises(TypeError):
        shell_frontend.extract_shell_units(
            "scripts/setup.sh",
            bytearray(b"printf ok\n"),
            executable_paths=frozenset(),
            budget=DependencyWorkBudget(),
        )


def test_extraction_uses_large_normalized_executable_inventory_without_iteration() -> None:
    executable_path = "bundle.zip!/bin/setup"
    executable_paths = _ObservedExecutablePaths(
        executable_path if index == 0 else f"bin/tool-{index}" for index in range(50_000)
    )

    applicable = _extract(
        executable_path,
        b"printf ok\n",
        executable_paths=executable_paths,
    )
    inert = _extract(
        "docs/readme.txt",
        b"plain text\n",
        executable_paths=executable_paths,
    )

    assert [issue.reason for issue in applicable.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]
    assert inert.units == ()
    assert inert.issues == ()
    assert executable_paths.iteration_count == 0
    assert executable_paths.membership_count == 2


def test_shell_lowering_parses_once_and_returns_one_completed_work_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"printf 'two words' plain\n"
    real_parse = shell_frontend.parse_bash_source
    parse_calls = 0

    def recording_parse(source: bytes, **kwargs: Any) -> Any:
        nonlocal parse_calls
        parse_calls += 1
        return real_parse(source, **kwargs)

    monkeypatch.setattr(shell_frontend, "parse_bash_source", recording_parse)

    result, _budget, unit = _analyze(raw)

    assert parse_calls == 1
    assert [_argv_bytes(command) for command in result.commands] == [
        (b"printf", b"two words", b"plain")
    ]
    assert result.assignments == ()
    assert result.generated_configs == ()
    assert result.issues == ()
    assert [(work.unit_id, work.outcome) for work in result.work_items] == [
        (unit.unit_id, dependency_types.ShellWorkOutcome.COMPLETED)
    ]


@pytest.mark.parametrize(
    ("raw", "node_type", "expected_fields"),
    [
        (b"cmd\n", "program", frozenset()),
        (b">out cmd arg\n", "command", frozenset({"redirect", "name", "argument"})),
        (b"A=one\n", "variable_assignment", frozenset({"name", "value"})),
        (b"export A=one\n", "declaration_command", frozenset()),
        (b"cmd >out\n", "redirected_statement", frozenset({"body", "redirect"})),
        (b"cmd 2>out\n", "file_redirect", frozenset({"descriptor", "destination"})),
        (b"f() { cmd; }\n", "function_definition", frozenset({"name", "body"})),
        (b"if ok; then yes; fi\n", "if_statement", frozenset({"condition"})),
        (
            b"for x in a; do yes; done\n",
            "for_statement",
            frozenset({"variable", "value", "body"}),
        ),
        (
            b"while ok; do yes; done\n",
            "while_statement",
            frozenset({"condition", "body"}),
        ),
        (b"case x in a) yes;; esac\n", "case_statement", frozenset({"value"})),
        (
            b"case x in a|b) first; second;; *) other;; esac\n",
            "case_item",
            frozenset({"value", "termination"}),
        ),
        (b"one | two\n", "pipeline", frozenset()),
        (b"one && two\n", "list", frozenset()),
        (b"! one\n", "negated_command", frozenset()),
        (b"( one )\n", "subshell", frozenset()),
        (b"{ one; }\n", "compound_statement", frozenset()),
        (b"outer $(inner)\n", "command_substitution", frozenset()),
        (b"outer <(inner)\n", "process_substitution", frozenset()),
        (b"if one; then two;\n", "fi", frozenset()),
        (b"good\nif broken\nlast $(\n", "ERROR", frozenset()),
    ],
)
def test_pinned_bash_cst_node_and_field_contract(
    raw: bytes,
    node_type: str,
    expected_fields: frozenset[str],
) -> None:
    root = shell_frontend.parse_bash_source(raw).root_node
    pending = [root]
    selected = None
    while pending:
        node = pending.pop()
        if node.type == node_type and (node_type != "fi" or node.is_missing):
            selected = node
            break
        pending.extend(reversed(node.children))

    assert selected is not None
    fields = frozenset(
        field_name
        for index in range(len(selected.children))
        if (field_name := selected.field_name_for_child(index)) is not None
    )
    contract_key = "MISSING" if selected.is_missing else node_type
    assert shell_frontend._PINNED_CST_FIELDS[contract_key] == expected_fields
    assert fields == expected_fields


def test_shell_lowering_visits_all_structural_regions_and_pipeline_stages() -> None:
    raw = (
        b"first && second || third; fourth &\n"
        b"( sub )\n"
        b"{ grouped; }\n"
        b"if cond; then yes; else no; fi\n"
        b"for item in a; do loop; done\n"
        b"while check; do body; done\n"
        b'case "$x" in a) arm;; esac\n'
        b"f() { inside; }\n"
        b'outer "$(inner arg)" <(producer) >(consumer)\n'
        b"one | two | three\n"
        b"! negated\n"
    )

    result, _budget, _unit = _analyze(raw)

    assert [command.argv[0].exact_bytes for command in result.commands] == [
        b"first",
        b"second",
        b"third",
        b"fourth",
        b"sub",
        b"grouped",
        b"cond",
        b"yes",
        b"no",
        b"loop",
        b"check",
        b"body",
        b"arm",
        b"inside",
        b"outer",
        b"inner",
        b"producer",
        b"consumer",
        b"one",
        b"two",
        b"three",
        b"negated",
    ]
    assert result.issues == ()
    assert result.work_items[0].outcome is dependency_types.ShellWorkOutcome.COMPLETED


def test_cst_substitution_depth_is_bounded_by_node_budget_not_nested_literal_budget() -> None:
    result, _budget, _unit = _analyze(
        b"outer $(middle $(inner $(deep)))\n",
        path="scripts/substitutions.sh",
    )

    assert [command.argv[0].exact_bytes for command in result.commands] == [
        b"outer",
        b"middle",
        b"inner",
        b"deep",
    ]
    assert result.issues == ()


def test_shell_lowering_emits_assignments_and_joins_only_line_continuations() -> None:
    raw = b"NA\\\nME=va\\\nlue com\\\nmand ar\\\ng\nexport C=see BARE\nD=$dynamic next\n"

    result, _budget, _unit = _analyze(raw)

    assert [
        (
            assignment.name,
            assignment.value.state,
            assignment.value.exact_bytes,
        )
        for assignment in result.assignments
    ] == [
        ("NAME", dependency_types.StaticValueState.EXACT, b"value"),
        ("C", dependency_types.StaticValueState.EXACT, b"see"),
        ("D", dependency_types.StaticValueState.UNKNOWN, None),
    ]
    assert [_argv_bytes(command) for command in result.commands] == [
        (b"command", b"arg"),
        (b"export", b"C=see", b"BARE"),
        (b"next",),
    ]


def test_only_structurally_bare_time_is_timing_syntax() -> None:
    raw = b'time -p timed arg;\n\\time escaped;\n"time" quoted;\n/bin/time path;\n$timer dynamic\n'

    result, _budget, _unit = _analyze(raw)

    assert [_argv_bytes(command) for command in result.commands] == [
        (b"timed", b"arg"),
        (b"time", b"escaped"),
        (b"time", b"quoted"),
        (b"/bin/time", b"path"),
        (None, b"dynamic"),
    ]


def test_redirect_destinations_do_not_hide_later_argv_and_exact_ampersand_redirect() -> None:
    raw = (
        b">lead first arg\n"
        b"second >middle arg\n"
        b"third arg >trail\n"
        b"fourth > one two three\n"
        b"fifth > dest\\\nination arg\n"
        b"sixth &>both arg\n"
    )

    analysis, _budget, unit = _analyze(raw)
    private = shell_frontend._analyze_shell_unit(unit, budget=DependencyWorkBudget())

    assert [_argv_bytes(command) for command in analysis.commands] == [
        (b"first", b"arg"),
        (b"second", b"arg"),
        (b"third", b"arg"),
        (b"fourth", b"two", b"three"),
        (b"fifth", b"arg"),
        (b"sixth", b"arg"),
    ]
    assert analysis.issues == ()
    assert [
        fact.kind.value for command in private.program.commands for fact in command.redirects
    ] == [
        "stdout_truncate",
        "stdout_truncate",
        "stdout_truncate",
        "stdout_truncate",
        "stdout_truncate",
        "stdout_stderr_truncate",
    ]


@pytest.mark.parametrize(
    ("raw", "syntax_start"),
    [
        (b"cmd &> | next\n", 7),
        (b"cmd >out <\n", 9),
    ],
)
def test_command_owned_malformed_redirect_regions_never_retain_supported_facts(
    raw: bytes,
    syntax_start: int,
) -> None:
    budget = DependencyWorkBudget()
    unit = _extract("scripts/malformed-redirect.sh", raw, budget=budget).units[0]

    private = shell_frontend._analyze_shell_unit(unit, budget=budget)

    assert [command.site.argv[0].exact_bytes for command in private.program.commands] == [b"cmd"]
    assert [fact for command in private.program.commands for fact in command.redirects] == []
    assert [
        (issue.reason, issue.span.start_byte, issue.span.end_byte)
        for issue in private.public.issues
    ] == [
        (
            dependency_types.ShellIssueReason.SYNTAX_ERROR,
            syntax_start,
            syntax_start + 1,
        )
    ]


@pytest.mark.parametrize(
    "malformed",
    [
        b"cmd &> | next\n",
        b"cmd >out <\n",
    ],
)
def test_malformed_redirect_suppression_does_not_poison_unrelated_commands(
    malformed: bytes,
) -> None:
    raw = b"safe >ok\n" + malformed
    budget = DependencyWorkBudget()
    unit = _extract("scripts/local-malformed-redirect.sh", raw, budget=budget).units[0]

    private = shell_frontend._analyze_shell_unit(unit, budget=budget)

    assert [command.site.argv[0].exact_bytes for command in private.program.commands] == [
        b"safe",
        b"cmd",
    ]
    assert [
        [fact.kind.value for fact in command.redirects] for command in private.program.commands
    ] == [["stdout_truncate"], []]
    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.SYNTAX_ERROR
    ]


def test_private_same_parse_ir_retains_bounded_structure_without_repr_content() -> None:
    budget = DependencyWorkBudget()
    unit = _extract(
        "scripts/private-ir.sh",
        b"secret_function() { A=secretvalue command secretliteral >secrettarget; }\n"
        b"eval 'probe child'\n",
        budget=budget,
    ).units[0]

    private = shell_frontend._analyze_shell_unit(unit, budget=budget)

    assert len(private.program.functions) == 1
    function = private.program.functions[0]
    command = private.program.commands[0]
    assignment = private.program.assignments[0]
    assert command.function_id == function.function_id
    assert assignment.function_id == function.function_id
    assert assignment.prefix_for_command_start_byte is not None
    assert [site.name for site in command.prefix_assignments] == ["A"]
    assert all(argument.fragments for argument in command.arguments)
    assert [fact.kind.value for fact in command.redirects] == ["stdout_truncate"]
    assert private.program.regions
    assert private.program.nested_programs[0].program.initial_functions
    rendered = repr(private)
    assert "secret_function" not in rendered
    assert "secretliteral" not in rendered
    assert "secrettarget" not in rendered
    assert "secretvalue" not in rendered


def test_ordered_fd0_fd1_redirect_facts_retain_only_the_final_effective_values() -> None:
    result, _budget, _unit = _analyze(b"command >first >|second 1>>.npmrc <<<first <<<final\n")

    assert [
        (config.target.exact_bytes, config.content.exact_bytes)
        for config in result.generated_configs
    ] == [(b".npmrc", b"final\n")]
    assert result.issues == ()


@pytest.mark.parametrize(
    "raw",
    [
        b"command >first 2>&1 <<EOF\nbody\nEOF\n",
        b"command <>readwrite >.npmrc <<EOF\nbody\nEOF\n",
        b"command &>>.npmrc <<EOF\nbody\nEOF\n",
        b"command 1>&2 >.npmrc <<EOF\nbody\nEOF\n",
        b"command <&0 >.npmrc <<EOF\nbody\nEOF\n",
        b"command 1<<EOF >.npmrc\nbody\nEOF\n",
        b"command 3<<EOF >.npmrc\nbody\nEOF\n",
        b"command <<EOF <input >.npmrc\nbody\nEOF\n",
    ],
)
def test_unsupported_descriptors_and_nonmodeled_fd_operations_fail_closed(
    raw: bytes,
) -> None:
    result, _budget, _unit = _analyze(raw)

    assert result.generated_configs == ()
    assert result.issues
    assert result.work_items[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL


def test_heredoc_wrapper_argument_field_remains_structurally_proven_argv() -> None:
    result, _budget, _unit = _analyze(b"command <<EOF arg\nbody\nEOF\n")

    assert [_argv_bytes(command) for command in result.commands] == [(b"command", b"arg")]
    assert result.issues == ()


def test_completed_heredoc_and_here_string_emit_policy_free_generated_configs() -> None:
    heredoc, _budget, _unit = _analyze(
        b"writer > .npmrc <<EOF\nregistry=https://packages.example.invalid\nEOF\n"
    )
    here_string, _budget, _unit = _analyze(
        b"writer > .npmrc <<< 'registry=https://packages.example.invalid'\n"
    )

    assert [
        (config.target.exact_bytes, config.content.exact_bytes)
        for config in heredoc.generated_configs
    ] == [(b".npmrc", b"registry=https://packages.example.invalid\n")]
    assert [
        (config.target.exact_bytes, config.content.exact_bytes)
        for config in here_string.generated_configs
    ] == [(b".npmrc", b"registry=https://packages.example.invalid\n")]
    assert heredoc.generated_configs[0].source_map is not None
    mapped = heredoc.generated_configs[0].source_map.map_range(9, 41)
    assert mapped is not None
    assert (mapped.path, mapped.start_line, mapped.end_line) == (
        "scripts/lower.sh",
        2,
        2,
    )
    assert heredoc.issues == ()
    assert here_string.issues == ()


def test_generated_target_uses_only_the_modeled_binding_at_its_write_site() -> None:
    result, _budget, _unit = _analyze(
        b"CFG=.npmrc\n"
        b'writer >"$CFG" <<EOF\nregistry=https://packages.example.invalid\nEOF\n'
        b"CFG=ignored.txt\n"
    )

    assert [config.target.exact_bytes for config in result.generated_configs] == [b".npmrc"]


def test_public_command_sites_lower_typed_adapter_reachability_facts() -> None:
    result, _budget, _unit = _analyze(
        b"export NPM_CONFIG_REGISTRY=https://packages.example.invalid\n"
        b"PIP_INDEX_URL=https://bare.example.invalid/simple\n"
        b"export PIP_INDEX_URL\n"
        b"PIP_INDEX_URL=https://pip.example.invalid/simple pip install thing\n"
        b"dead() { npm config set registry https://dead.example.invalid; }\n"
        b"live() { yarn config set registry https://live.example.invalid; }\n"
        b"live\n"
        b"npm() { :; }\n"
        b"npm config set registry https://shadowed.example.invalid\n"
        b"/usr/bin/npm config set registry https://external.example.invalid\n"
    )

    assert result.commands
    assert all(len(command.argument_spans) == len(command.argv) for command in result.commands)
    export = next(command for command in result.commands if command.exported_assignments)
    assert [(site.name, site.value.exact_bytes) for site in export.exported_assignments] == [
        ("NPM_CONFIG_REGISTRY", b"https://packages.example.invalid")
    ]
    bare_export = next(
        site
        for command in result.commands
        for site in command.exported_assignments
        if site.name == "PIP_INDEX_URL"
    )
    assert bare_export.value.exact_bytes == b"https://bare.example.invalid/simple"
    assert bare_export.span.start_line == 3
    pip = next(command for command in result.commands if command.argv[0].exact_bytes == b"pip")
    assert [site.name for site in pip.prefix_assignments] == ["PIP_INDEX_URL"]
    dead = next(
        command
        for command in result.commands
        if command.argv[0].exact_bytes == b"npm"
        and command.argv[-1].exact_bytes == b"https://dead.example.invalid"
    )
    live = next(command for command in result.commands if command.argv[0].exact_bytes == b"yarn")
    shadowed = next(
        command
        for command in result.commands
        if command.argv[0].exact_bytes == b"npm"
        and command.argv[-1].exact_bytes == b"https://shadowed.example.invalid"
    )
    assert dead.producer is dependency_types.CommandProducerReachability.INERT
    assert live.producer is dependency_types.CommandProducerReachability.ACTIVE
    assert shadowed.resolution is dependency_types.CommandResolutionKind.FUNCTION
    external = next(
        command for command in result.commands if command.argv[0].exact_bytes == b"/usr/bin/npm"
    )
    assert external.resolution is dependency_types.CommandResolutionKind.EXTERNAL


@pytest.mark.parametrize("value", [b"foo bar", b"*"])
def test_unquoted_output_target_expansion_never_proves_one_filename(value: bytes) -> None:
    prefix = b"DIR='" + value + b"'\n"
    suffix = b" <<EOF\nregistry=https://packages.example.invalid\nEOF\n"
    unquoted, _budget, _unit = _analyze(prefix + b"writer >$DIR/.npmrc" + suffix)
    quoted, _budget, _unit = _analyze(prefix + b'writer >"$DIR/.npmrc"' + suffix)

    assert len(unquoted.generated_configs) == 1
    assert unquoted.generated_configs[0].target.state is (dependency_types.StaticValueState.UNKNOWN)
    assert unquoted.issues
    assert [config.target.exact_bytes for config in quoted.generated_configs] == [
        value + b"/.npmrc"
    ]
    assert quoted.issues == ()


@pytest.mark.timeout(3)
def test_dynamic_function_calls_without_generated_configs_have_bounded_activation_work() -> None:
    count = 10_000
    unit_id = "0" * 32
    span = dependency_types.SourceSpan("scripts/bounded.sh", 0, 1, 1, 1)
    functions = tuple(
        shell_frontend._FunctionContext(
            index,
            dependency_types.StaticValue.exact(b"f" + str(index).encode("ascii")),
            span,
            (),
            None,
            None,
            index,
            index + 1,
        )
        for index in range(count)
    )
    commands = tuple(
        shell_frontend._CommandIR(
            dependency_types.CommandSite(
                unit_id,
                dependency_types.SiteProvenance.FILE_SUFFIX,
                span,
                (dependency_types.StaticValue.unknown(),),
            ),
            index,
            None,
            None,
            (),
            (),
            (),
            program_id=unit_id,
            resolution=shell_frontend._CommandResolution(
                shell_frontend._CommandResolutionKind.AMBIGUOUS
            ),
        )
        for index in range(count)
    )
    program = shell_frontend._ShellProgramIR(
        program_id=unit_id,
        functions=functions,
        commands=commands,
    )

    class IssueSink:
        def __init__(self) -> None:
            self.count = 0

        def _issue(self, *_args: Any, **_kwargs: Any) -> None:
            self.count += 1

    issue_sink = IssueSink()
    retained = shell_frontend._retain_typed_function_generated_configs(
        issue_sink,
        program,
        (),
        commands,
        [],
    )

    assert retained == []
    assert issue_sink.count == 0

    imported_count = 20_000
    same_name_functions = tuple(
        shell_frontend._FunctionContext(
            index,
            dependency_types.StaticValue.exact(b"f"),
            span,
            (),
            None,
            None,
            index,
            index + 1,
        )
        for index in range(imported_count)
    )
    imported_calls = tuple(
        shell_frontend._CommandIR(
            dependency_types.CommandSite(
                unit_id,
                dependency_types.SiteProvenance.FILE_SUFFIX,
                span,
                (dependency_types.StaticValue.exact(b"f"),),
            ),
            index,
            None,
            None,
            (),
            (),
            (),
            program_id=unit_id,
            resolution=shell_frontend._CommandResolution(
                shell_frontend._CommandResolutionKind.FUNCTION,
                shell_frontend._IMPORTED_FUNCTION_ID,
            ),
        )
        for index in range(imported_count)
    )
    imported_program = shell_frontend._ShellProgramIR(
        program_id=unit_id,
        functions=same_name_functions,
        commands=imported_calls,
    )
    config = shell_frontend._ProvenGeneratedConfig(
        unit_id=unit_id,
        provenance=dependency_types.SiteProvenance.GENERATED_CONFIG,
        span=span,
        target=dependency_types.StaticValue.exact(b".npmrc"),
        content=dependency_types.StaticValue.exact(b"registry=https://x.invalid\n"),
        physical_size_bytes=1,
        physical_line_starts=(0,),
        producer_program_id=unit_id,
        producer_function_id=0,
    )

    retained = shell_frontend._retain_typed_function_generated_configs(
        issue_sink,
        imported_program,
        (),
        imported_calls,
        [config],
    )

    assert retained == []
    assert issue_sink.count == imported_count


def test_dynamic_command_producer_reachability_is_linear_at_normative_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = 10_000
    unit_id = "0" * 32
    span = dependency_types.SourceSpan("scripts/bounded-producers.sh", 0, 1, 1, 1)
    functions = tuple(
        shell_frontend._FunctionContext(
            index,
            dependency_types.StaticValue.exact(b"f" + str(index).encode("ascii")),
            span,
            (),
            None,
            None,
            index,
            index + 1,
        )
        for index in range(count)
    )

    def command(
        index: int,
        *,
        dynamic: bool,
        function_id: int | None,
    ) -> shell_frontend._CommandIR:
        value = (
            dependency_types.StaticValue.unknown()
            if dynamic
            else dependency_types.StaticValue.exact(b"external")
        )
        return shell_frontend._CommandIR(
            dependency_types.CommandSite(
                unit_id,
                dependency_types.SiteProvenance.FILE_SUFFIX,
                span,
                (value,),
            ),
            index,
            None,
            function_id,
            (),
            (),
            (),
            program_id=unit_id,
            resolution=shell_frontend._CommandResolution(
                shell_frontend._CommandResolutionKind.AMBIGUOUS
                if dynamic
                else shell_frontend._CommandResolutionKind.EXTERNAL
            ),
        )

    root_calls = tuple(command(index, dynamic=True, function_id=None) for index in range(count))
    function_bodies = tuple(
        command(count + index, dynamic=False, function_id=index) for index in range(count)
    )
    commands = root_calls + function_bodies
    program = shell_frontend._ShellProgramIR(
        program_id=unit_id,
        functions=functions,
        commands=commands,
    )

    class BoundedSet(set[Any]):
        updated_items = 0

        def update(self, *others: Any) -> None:
            type(self).updated_items += sum(len(other) for other in others)
            if type(self).updated_items > count * 4:
                raise AssertionError("producer reachability materialized a quadratic edge set")
            super().update(*others)

    monkeypatch.setattr(shell_frontend, "set", BoundedSet, raising=False)

    annotated = shell_frontend._annotate_command_producer_reachability(program, (), commands)

    assert all(
        command.site.producer is dependency_types.CommandProducerReachability.ACTIVE
        for command in annotated[:count]
    )
    assert all(
        command.site.producer is dependency_types.CommandProducerReachability.AMBIGUOUS
        for command in annotated[count:]
    )
    assert BoundedSet.updated_items <= count * 4


def test_exact_name_ambiguity_never_materializes_caller_target_cartesian_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = 1_500
    unit_id = "0" * 32
    span = dependency_types.SourceSpan("scripts/same-name-producers.sh", 0, 1, 1, 1)
    functions = tuple(
        shell_frontend._FunctionContext(
            index,
            dependency_types.StaticValue.exact(b"f"),
            span,
            (),
            None,
            None,
            index,
            index + 1,
        )
        for index in range(count)
    )

    def ambiguous_call(index: int, function_id: int | None) -> shell_frontend._CommandIR:
        return shell_frontend._CommandIR(
            dependency_types.CommandSite(
                unit_id,
                dependency_types.SiteProvenance.FILE_SUFFIX,
                span,
                (dependency_types.StaticValue.exact(b"f"),),
            ),
            index,
            None,
            function_id,
            (),
            (),
            (),
            program_id=unit_id,
            resolution=shell_frontend._CommandResolution(
                shell_frontend._CommandResolutionKind.AMBIGUOUS
            ),
        )

    commands = (ambiguous_call(0, None),) + tuple(
        ambiguous_call(index + 1, index) for index in range(count)
    )
    program = shell_frontend._ShellProgramIR(
        program_id=unit_id,
        functions=functions,
        commands=commands,
    )

    class BoundedSet(set[Any]):
        updated_items = 0

        def update(self, *others: Any) -> None:
            type(self).updated_items += sum(len(other) for other in others)
            if type(self).updated_items > count * 4:
                raise AssertionError("producer reachability materialized a Cartesian edge set")
            super().update(*others)

    monkeypatch.setattr(shell_frontend, "set", BoundedSet, raising=False)

    annotated = shell_frontend._annotate_command_producer_reachability(program, (), commands)

    assert annotated[0].site.producer is dependency_types.CommandProducerReachability.ACTIVE
    assert all(
        command.site.producer is dependency_types.CommandProducerReachability.AMBIGUOUS
        for command in annotated[1:]
    )
    assert BoundedSet.updated_items <= count * 4


def test_second_on_line_recovery_emits_only_redirect_evidence() -> None:
    private, _budget, _unit = _analyze_private(
        b"cat > /dev/null <<A; cat > .npmrc <<B\n"
        b"first\n"
        b"A\n"
        b"registry=https://packages.example.invalid\n"
        b"B\n"
    )
    result = private.public

    assert [_argv_bytes(command) for command in result.commands] == [(b"cat",)]
    assert [
        (config.target.exact_bytes, config.content.exact_bytes)
        for config in result.generated_configs
        if config.target.exact_bytes == b".npmrc"
    ] == [(b".npmrc", b"registry=https://packages.example.invalid\n")]
    assert [fact.target.value.exact_bytes for fact in private.program.commands[0].redirects] == [
        b"/dev/null",
        b"first\n",
    ]
    assert result.issues == ()


@pytest.mark.parametrize("separator", [b";", b" &"])
def test_preceding_pending_heredoc_error_cannot_relabel_the_fifo_body(
    separator: bytes,
) -> None:
    result, _budget, _unit = _analyze(
        b"cat <<A >/dev/null" + separator + b" cat >.npmrc <<B\n"
        b"first\n"
        b"A\n"
        b"registry=https://packages.example.invalid\n"
        b"B\n"
    )

    assert result.generated_configs == ()
    assert [_argv_bytes(command) for command in result.commands] == [(b"cat",), (b"cat",)]
    assert result.issues
    assert result.work_items[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL


@pytest.mark.parametrize("separator", [b";", b"&"])
def test_pending_heredoc_guard_follows_one_exact_line_continuation(
    separator: bytes,
) -> None:
    result, _budget, _unit = _analyze(
        b"cat <<A >/dev/null " + separator + b" \\\ncat >.npmrc <<B\n"
        b"registry=https://first.example.invalid\n"
        b"A\n"
        b"# empty\n"
        b"B\n"
    )

    assert result.generated_configs == ()
    assert result.issues
    assert result.work_items[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL


@pytest.mark.parametrize("gap", [b" \\ \n", b" # not-a-continuation\n"])
def test_pending_heredoc_guard_does_not_expand_to_near_miss_gaps(gap: bytes) -> None:
    result, _budget, _unit = _analyze(
        b"cat <<A >/dev/null; " + gap + b"cat >.npmrc <<B\n"
        b"registry=https://first.example.invalid\n"
        b"A\n"
        b"# empty\n"
        b"B\n"
    )

    assert result.generated_configs == ()
    assert result.issues


@pytest.mark.parametrize(
    "header",
    [
        b'<<EN"D"',
        b'<<$"END"',
        b"<<$'END'",
        b"<<END\\\nMORE",
    ],
)
def test_audited_missing_end_shapes_use_physical_delimiter_quote_removal(
    header: bytes,
) -> None:
    delimiter = b"ENDMORE" if b"\\\n" in header else b"END"
    result, _budget, _unit = _analyze(
        b"writer >.npmrc "
        + header
        + b"\nregistry=https://packages.example.invalid\n"
        + delimiter
        + b"\n"
    )

    assert [config.content.exact_bytes for config in result.generated_configs] == [
        b"registry=https://packages.example.invalid\n"
    ]
    assert result.issues == ()


@pytest.mark.parametrize(
    "raw",
    [
        b"if true; then writer >.npmrc <<EOF\nregistry=x\nEOF\nfi\n",
        b"command writer >.npmrc <<EOF\nregistry=x\nEOF\n",
        b"mkdir -p . && writer >.npmrc <<EOF\nregistry=x\nEOF\n",
        b"writer >.npmrc \\\n  <<EOF\nregistry=x\nEOF\n",
        b"writer &>.npmrc <<EOF &\nregistry=x\nEOF\nwait\n",
    ],
)
def test_audited_control_and_header_shapes_keep_the_proven_write(raw: bytes) -> None:
    result, _budget, _unit = _analyze(raw)

    assert [
        (config.target.exact_bytes, config.content.exact_bytes)
        for config in result.generated_configs
    ] == [(b".npmrc", b"registry=x\n")]
    assert result.issues == ()


def test_tab_stripping_and_unquoted_body_line_continuation_preserve_exact_maps() -> None:
    stripped, _budget, _unit = _analyze(
        b"writer >.npmrc <<-EOF\n\tregistry=https://packages.example.invalid\n\tEOF\n"
    )
    continued, _budget, _unit = _analyze(
        b"writer >.npmrc <<EOF\nregistry=https://packages.example.inva\\\nlid\nEOF\n"
    )

    assert stripped.generated_configs[0].content.exact_bytes == (
        b"registry=https://packages.example.invalid\n"
    )
    assert continued.generated_configs[0].content.exact_bytes == (
        b"registry=https://packages.example.invalid\n"
    )
    assert stripped.generated_configs[0].source_map is not None
    assert continued.generated_configs[0].source_map is not None


def test_explicit_fd0_and_multiple_heredocs_are_recovered_fifo_and_masked() -> None:
    explicit, _budget, _unit = _analyze(
        b"writer 0<<EOF >.npmrc\nregistry=https://packages.example.invalid\nEOF\n"
    )
    multiple, _budget, _unit = _analyze(
        b"writer >.npmrc <<A <<B\n"
        b"npm config set registry https://ignored.example.invalid\n"
        b"A\n"
        b"registry=https://packages.example.invalid\n"
        b"B\n"
    )

    assert [config.content.exact_bytes for config in explicit.generated_configs] == [
        b"registry=https://packages.example.invalid\n"
    ]
    assert [config.content.exact_bytes for config in multiple.generated_configs] == [
        b"registry=https://packages.example.invalid\n"
    ]
    assert [_argv_bytes(command) for command in multiple.commands] == [(b"writer",)]
    assert explicit.issues == ()
    assert multiple.issues == ()


def test_false_clean_and_unterminated_extents_mask_through_eof_and_report_partial() -> None:
    false_clean, _budget, _unit = _analyze(
        b"writer >.npmrc <<EOF\n"
        b"registry=https://packages.example.invalid\n"
        b"EOF \n"
        b"npm config set registry https://not-a-command.example.invalid\n"
    )
    inert, _budget, _unit = _analyze(
        b"writer >/dev/null <<EOF\nnpm config set registry https://not-a-command.example.invalid\n"
    )

    assert [config.target.exact_bytes for config in false_clean.generated_configs] == [b".npmrc"]
    assert [_argv_bytes(command) for command in false_clean.commands] == [(b"writer",)]
    assert [_argv_bytes(command) for command in inert.commands] == [(b"writer",)]
    assert false_clean.work_items[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL
    assert inert.work_items[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL


def test_backslash_newline_delimiter_body_is_data_for_an_unrelated_target() -> None:
    result, _budget, _unit = _analyze(
        b"writer >/dev/null <<END\\\nMORE\n"
        b"npm config set registry https://not-a-command.example.invalid\n"
        b"ENDMORE\n"
    )

    assert [_argv_bytes(command) for command in result.commands] == [(b"writer",)]
    assert result.issues == ()


def test_unquoted_body_uses_bounded_modeled_expansion_but_quoted_body_is_literal() -> None:
    unquoted, _budget, _unit = _analyze(
        b"HOST=packages.example.invalid\n"
        b"writer >.npmrc <<EOF\nregistry=https://$HOST/private\nEOF\n"
    )
    quoted, _budget, _unit = _analyze(
        b"HOST=packages.example.invalid\n"
        b"writer >.npmrc <<'EOF'\nregistry=https://$HOST/private\nEOF\n"
    )
    dynamic, _budget, _unit = _analyze(
        b"writer >.npmrc <<EOF\nregistry=https://$(hostname)/private\nEOF\n"
    )

    assert unquoted.generated_configs[0].content.exact_bytes == (
        b"registry=https://packages.example.invalid/private\n"
    )
    assert quoted.generated_configs[0].content.exact_bytes == (b"registry=https://$HOST/private\n")
    assert dynamic.generated_configs[0].content.state is dependency_types.StaticValueState.UNKNOWN
    assert dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS in {
        issue.reason for issue in dynamic.issues
    }


def test_recovered_extent_dynamic_content_and_target_remain_limitations() -> None:
    dynamic_body, _budget, _unit = _analyze(
        b"writer 0<<EOF >.npmrc\nregistry=https://$(hostname)/private\nEOF\n"
    )
    dynamic_target, _budget, _unit = _analyze(
        b'cat >/dev/null <<A; cat >"$CFG" <<B\n'
        b"first\nA\nregistry=https://packages.example.invalid\nB\n"
    )

    assert dynamic_body.generated_configs[0].content.state is (
        dependency_types.StaticValueState.UNKNOWN
    )
    assert dynamic_target.generated_configs == ()
    assert dynamic_body.issues
    assert dynamic_target.issues


def test_here_string_expansion_retains_the_shell_added_newline() -> None:
    result, _budget, _unit = _analyze(
        b'HOST=packages.example.invalid\nwriter >.npmrc <<< "registry=https://$HOST/private"\n'
    )

    assert result.generated_configs[0].content.exact_bytes == (
        b"registry=https://packages.example.invalid/private\n"
    )


def test_narrow_external_tee_writer_and_pipeline_and_shadowing_boundaries() -> None:
    direct, _budget, _unit = _analyze(
        b"tee -a -- .npmrc <<EOF\nregistry=https://packages.example.invalid\nEOF\n"
    )
    pipeline, _budget, _unit = _analyze(
        b"cat <<EOF | tee .npmrc\nregistry=https://packages.example.invalid\nEOF\n"
    )
    shadowed, _budget, _unit = _analyze(
        b"tee() { :; }\ntee .npmrc <<EOF\nregistry=https://packages.example.invalid\nEOF\n"
    )

    assert [config.target.exact_bytes for config in direct.generated_configs] == [b".npmrc"]
    assert pipeline.generated_configs == ()
    assert shadowed.generated_configs == ()
    assert dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS in {
        issue.reason for issue in pipeline.issues
    }
    assert dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS in {
        issue.reason for issue in shadowed.issues
    }


def test_tee_file_operand_is_independent_of_its_stdout_redirection() -> None:
    result, _budget, _unit = _analyze(
        b"tee .npmrc >capture.log <<EOF\nregistry=https://packages.example.invalid\nEOF\n"
    )

    assert {config.target.exact_bytes for config in result.generated_configs} == {
        b".npmrc",
        b"capture.log",
    }


@pytest.mark.parametrize(
    "raw",
    [
        b"writer >.npmrc <<EOF )\nregistry=https://packages.example.invalid\nEOF\n",
        b"writer >.npmrc <<<< value\n",
        b"writer >.npmrc <<E\x00OF\nregistry=https://packages.example.invalid\nEOF\n",
        b"writer >.npmrc <<EOF\nregistry=https://packages.example.invalid\x00\nEOF\n",
        b"writer >.npmrc <<\nregistry=https://packages.example.invalid\n",
        b"cat >/dev/null <<A cat >.npmrc <<B\nfirst\nA\nregistry=x\nB\n",
        b"cat >/dev/null <<A; bogus x >.npmrc <<B\nfirst\nA\nregistry=x\nB\n",
        b"cat >/dev/null <<A; cat >.npmrc <<B )\nfirst\nA\nregistry=x\nB\n",
        b"cat >/dev/null <<A; cat >.npmrc <<B <y\nfirst\nA\nregistry=x\nB\n",
        b"cat >/dev/null <<A; cat >.npmrc <<B x<y\nfirst\nA\nregistry=x\nB\n",
    ],
)
def test_cst_recovery_near_misses_never_create_generated_evidence(raw: bytes) -> None:
    result, _budget, _unit = _analyze(raw)

    assert result.generated_configs == ()
    assert result.issues
    assert result.work_items[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL


@pytest.mark.parametrize(
    ("suffix", "target", "content"),
    [
        (b"<<<y", b".npmrc", b"y\n"),
        (b">.yarnrc", b".yarnrc", b"registry=x\n"),
        (b"1>>pip.conf", b"pip.conf", b"registry=x\n"),
        (b"&>.yarnrc", b".yarnrc", b"registry=x\n"),
    ],
)
def test_recovered_second_segment_reduces_its_own_final_fd0_fd1_facts(
    suffix: bytes,
    target: bytes,
    content: bytes,
) -> None:
    result, _budget, _unit = _analyze(
        b"cat >/dev/null <<A; cat >.npmrc <<B " + suffix + b"\nfirst\nA\nregistry=x\nB\n"
    )

    configs = [
        (config.target.exact_bytes, config.content.exact_bytes)
        for config in result.generated_configs
        if config.target.exact_bytes != b"/dev/null"
    ]
    assert configs == [(target, content)]


def test_recovery_and_generated_mapping_respect_exact_and_one_over_budgets() -> None:
    raw = (
        b"cat >/dev/null <<A; cat >.npmrc <<B\n"
        b"first\nA\nregistry=https://packages.example.invalid\nB\n"
    )
    baseline_budget = DependencyWorkBudget()
    unit = _extract("scripts/recovery-budget.sh", raw, budget=baseline_budget).units[0]
    baseline = shell_frontend.analyze_shell_unit(unit, budget=baseline_budget)
    assert baseline.work_items[0].outcome is dependency_types.ShellWorkOutcome.COMPLETED
    baseline_file = baseline_budget.for_file(unit.origin_span.path)
    required = {
        dependency_types.DependencyWorkResource.SHELL_CST_VISITS: (
            baseline_file.used_for_unit(
                unit,
                dependency_types.DependencyWorkResource.SHELL_CST_VISITS,
            )
        ),
        dependency_types.DependencyWorkResource.RETAINED_SHELL_IR: baseline_budget.used(
            dependency_types.DependencyWorkResource.RETAINED_SHELL_IR
        ),
        dependency_types.DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES: (
            baseline_file.used(dependency_types.DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES)
        ),
        dependency_types.DependencyWorkResource.SHELL_SOURCE_MAP_ENTRIES: (
            baseline_file.used(dependency_types.DependencyWorkResource.SHELL_SOURCE_MAP_ENTRIES)
        ),
    }
    limits = {
        dependency_types.DependencyWorkResource.SHELL_CST_VISITS: (
            dependency_types.DEPENDENCY_SHELL_CST_VISIT_FACTOR * len(raw)
            + dependency_types.DEPENDENCY_SHELL_CST_VISIT_BASE
        ),
        dependency_types.DependencyWorkResource.RETAINED_SHELL_IR: (
            dependency_types.MAX_DEPENDENCY_RETAINED_SHELL_IR
        ),
        dependency_types.DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES: (
            dependency_types.MAX_DEPENDENCY_SHELL_VALUE_BYTES_PER_FILE
        ),
        dependency_types.DependencyWorkResource.SHELL_SOURCE_MAP_ENTRIES: (
            dependency_types.MAX_DEPENDENCY_SOURCE_MAP_ENTRIES_PER_FILE
        ),
    }

    for resource, required_count in required.items():
        assert required_count > 0
        for extra, expected in [(0, "completed"), (1, "partial")]:
            budget = DependencyWorkBudget()
            file_budget = budget.for_file(unit.origin_span.path)
            file_budget.register_shell_file_size(len(raw))
            precharge = limits[resource] - required_count + extra
            if resource is dependency_types.DependencyWorkResource.SHELL_CST_VISITS:
                exhaustion = file_budget.charge_shell_cst_visits(unit, precharge)
            elif resource is dependency_types.DependencyWorkResource.RETAINED_SHELL_IR:
                exhaustion = file_budget.charge_retained_shell_ir(unit, precharge)
            elif resource is dependency_types.DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES:
                exhaustion = file_budget.reserve_shell_value_bytes(precharge)
            else:
                exhaustion = file_budget.charge_source_map_entries(precharge)
            assert exhaustion is None

            result = shell_frontend.analyze_shell_unit(unit, budget=budget)

            assert result.work_items[0].outcome.value == expected
            if extra:
                assert dependency_types.ShellIssueReason.RESOURCE_LIMIT in {
                    issue.reason for issue in result.issues
                }


def test_assignment_value_fragments_are_charged_as_retained_private_ir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    charges: list[int] = []
    original = dependency_types.DependencyFileBudget.charge_retained_shell_ir

    def recording_charge(
        file_budget: dependency_types.DependencyFileBudget,
        unit: dependency_types.ShellUnit,
        count: int,
    ) -> dependency_types.DependencyWorkExhaustion | None:
        charges.append(count)
        return original(file_budget, unit, count)

    monkeypatch.setattr(
        dependency_types.DependencyFileBudget,
        "charge_retained_shell_ir",
        recording_charge,
    )

    budget = DependencyWorkBudget()
    unit = _extract("scripts/assignment-ir.sh", b"A=value\n", budget=budget).units[0]
    private = shell_frontend._analyze_shell_unit(unit, budget=budget)

    fragment_count = len(private.program.assignments[0].value_fragments)
    assert fragment_count == 1
    assert 2 + fragment_count in charges


def test_redirect_association_does_not_rescan_full_command_inventory() -> None:
    command_count = 200
    raw = b"".join(f"command{index} >target\n".encode() for index in range(command_count))
    budget = DependencyWorkBudget()
    unit = _extract("scripts/many-redirects.sh", raw, budget=budget).units[0]
    file_budget = budget.for_file(unit.origin_span.path)
    lowerer = shell_frontend._ShellLowerer(unit, budget, file_budget)
    lowerer.walk(shell_frontend.parse_bash_source(raw).root_node)
    observed = _ObservedDraftList(lowerer.command_drafts)
    lowerer.command_drafts = observed

    program = lowerer.lower()

    assert len(program.commands) == command_count
    assert observed.iterated_items <= command_count * 3


def test_unretained_issue_still_marks_terminal_work_partial() -> None:
    budget = DependencyWorkBudget()
    assert (
        budget.charge_shell_issues(dependency_types.MAX_DEPENDENCY_SHELL_LOCALIZED_ISSUES - 1)
        is None
    )
    assert (
        budget.claim_reserved_shell_truncation_issue()
        is dependency_types.ShellTruncationClaimStatus.CLAIMED
    )

    result, _budget, _unit = _analyze(
        b"command bad\x00value\n",
        path="scripts/full-issue-budget.sh",
        budget=budget,
    )

    assert result.issues == ()
    assert result.work_items[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL


def test_unsupported_descriptor_redirects_preserve_provable_argv() -> None:
    raw = b"first &>>out arg\nsecond 2>&1 arg\nthird 1>&2 arg\nfourth <&0 arg\n"

    result, _budget, _unit = _analyze(raw)

    assert [_argv_bytes(command) for command in result.commands] == [
        (b"first", b"arg"),
        (b"second", b"arg"),
        (b"third", b"arg"),
        (b"fourth", b"arg"),
    ]
    assert [issue.reason for issue in result.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ] * 4
    assert result.work_items[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL


def test_markdown_lowering_composes_multibyte_crlf_locations() -> None:
    raw = "intro\r\n```bash\r\nécho 'x;y' # hidden\r\n```\r\n".encode()
    budget = DependencyWorkBudget()
    extraction = _extract("docs/guide.md", raw, budget=budget)

    result = shell_frontend.analyze_shell_unit(extraction.units[0], budget=budget)

    command_start = raw.index("écho".encode())
    command_end = command_start + len("écho 'x;y'".encode())
    assert result.commands[0].span == dependency_types.SourceSpan(
        "docs/guide.md",
        command_start,
        command_end,
        3,
        3,
        start_column=0,
        end_column=len("écho 'x;y'".encode()),
    )
    assert _argv_bytes(result.commands[0]) == ("écho".encode(), b"x;y")
    assert result.issues == ()


def test_comments_quotes_and_non_shell_separators_never_create_raw_commands() -> None:
    quoted, _budget, _unit = _analyze(b"printf 'a;#b' plain # ignored\r\n")
    separated, _budget, _unit = _analyze(
        b"printf first\fsecond\rthird\x00four\n",
        path="scripts/separators.sh",
    )

    assert [_argv_bytes(command) for command in quoted.commands] == [(b"printf", b"a;#b", b"plain")]
    assert quoted.issues == ()
    assert [_argv_bytes(command) for command in separated.commands] == [
        (b"printf", b"first", b"second", None)
    ]
    assert [issue.reason for issue in separated.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]


@pytest.mark.parametrize(
    "raw",
    [
        b"printf ~/file\n",
        b"printf file*\n",
        b"printf file?\n",
        b"printf file[ab]\n",
    ],
)
def test_unquoted_tilde_and_pathname_expansion_arguments_are_unknown(raw: bytes) -> None:
    result, _budget, _unit = _analyze(raw)

    assert result.commands[0].argv[0] == dependency_types.StaticValue.exact(b"printf")
    assert result.commands[0].argv[1].state is dependency_types.StaticValueState.UNKNOWN
    assert result.issues == ()


@pytest.mark.parametrize(
    "manager_name",
    [
        b"~/bin/npm",
        b"np*",
        b"np?",
        b"$manager",
        b"${manager}",
        b"npm-${channel}",
    ],
)
def test_unquoted_dynamic_manager_name_shapes_are_unknown(manager_name: bytes) -> None:
    result, _budget, _unit = _analyze(manager_name + b" install\n")

    assert result.commands[0].argv[0].state is dependency_types.StaticValueState.UNKNOWN
    assert result.commands[0].argv[1] == dependency_types.StaticValue.exact(b"install")
    assert result.issues == ()


def test_unquoted_tilde_assignment_value_is_unknown() -> None:
    result, _budget, _unit = _analyze(b"A=~/repo cmd\n")

    assert [(assignment.name, assignment.value.state) for assignment in result.assignments] == [
        ("A", dependency_types.StaticValueState.UNKNOWN)
    ]
    assert _argv_bytes(result.commands[0]) == (b"cmd",)
    assert result.issues == ()


def test_unquoted_tilde_after_colon_in_assignment_values_is_unknown() -> None:
    result, _budget, _unit = _analyze(b"A=/bin:~/bin cmd\nexport PATH=/bin:~/bin\n")

    assert [(assignment.name, assignment.value.state) for assignment in result.assignments] == [
        ("A", dependency_types.StaticValueState.UNKNOWN),
        ("PATH", dependency_types.StaticValueState.UNKNOWN),
    ]
    assert result.issues == ()


@pytest.mark.parametrize(
    "value",
    [
        b"~/repo",
        b"/bin:~/bin",
    ],
)
def test_continued_assignment_tilde_expansion_is_unknown(value: bytes) -> None:
    result, _budget, _unit = _analyze(b"A\\\n=" + value + b" cmd\n")

    assert [(assignment.name, assignment.value.state) for assignment in result.assignments] == [
        ("A", dependency_types.StaticValueState.UNKNOWN)
    ]
    assert _argv_bytes(result.commands[0]) == (b"cmd",)
    assert result.issues == ()


@pytest.mark.parametrize(
    ("raw", "expected_value"),
    [
        (b"A=foo\\\nbar cmd\n", dependency_types.StaticValue.exact(b"foobar")),
        (b"A=/bin:\\\n~/bin cmd\n", dependency_types.StaticValue.unknown()),
    ],
)
def test_prefix_assignment_continuation_absorbs_the_following_cst_name_fragment(
    raw: bytes,
    expected_value: dependency_types.StaticValue,
) -> None:
    result, _budget, _unit = _analyze(raw)

    assert [(assignment.name, assignment.value) for assignment in result.assignments] == [
        ("A", expected_value)
    ]
    assert [_argv_bytes(command) for command in result.commands] == [(b"cmd",)]
    assert result.issues == ()


def test_export_continued_assignment_is_emitted_exactly_once() -> None:
    result, _budget, _unit = _analyze(b"export A=foo\\\nbar\n")

    assert [(assignment.name, assignment.value) for assignment in result.assignments] == [
        ("A", dependency_types.StaticValue.exact(b"foobar"))
    ]
    assert [_argv_bytes(command) for command in result.commands] == [(b"export", b"A=foobar")]
    assert result.issues == ()


@pytest.mark.parametrize(
    "keyword",
    [b"declare", b"readonly", b"typeset", b"export"],
)
def test_declaration_continued_assignment_matches_full_argv_and_is_emitted_once(
    keyword: bytes,
) -> None:
    result, _budget, _unit = _analyze(keyword + b" A=foo\\\nbar\n")

    assert [_argv_bytes(command) for command in result.commands] == [(keyword, b"A=foobar")]
    assert [(assignment.name, assignment.value) for assignment in result.assignments] == [
        (
            "A",
            (
                dependency_types.StaticValue.unknown()
                if keyword == b"readonly"
                else dependency_types.StaticValue.exact(b"foobar")
            ),
        )
    ]
    assert [issue.reason for issue in result.issues] == (
        [dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS] if keyword == b"readonly" else []
    )


def test_local_continued_assignment_in_function_matches_full_argv_and_is_emitted_once() -> None:
    result, _budget, _unit = _analyze(b"f() { local A=foo\\\nbar; }\n")

    assert [_argv_bytes(command) for command in result.commands] == [(b"local", b"A=foobar")]
    assert [(assignment.name, assignment.value) for assignment in result.assignments] == [
        ("A", dependency_types.StaticValue.exact(b"foobar"))
    ]
    assert result.issues == ()


@pytest.mark.parametrize(
    ("raw", "expected_argv", "expected_assignments"),
    [
        (b'printf "x"~/file\n', (b"printf", b"x~/file"), ()),
        (
            b'A="x"~/repo cmd\n',
            (b"cmd",),
            (("A", dependency_types.StaticValue.exact(b"x~/repo")),),
        ),
        (b"np\\\n~/bin install\n", (b"np~/bin", b"install"), ()),
    ],
)
def test_cross_fragment_tilde_outside_expansion_position_remains_exact(
    raw: bytes,
    expected_argv: tuple[bytes, ...],
    expected_assignments: tuple[tuple[str, dependency_types.StaticValue], ...],
) -> None:
    result, _budget, _unit = _analyze(raw)

    assert [_argv_bytes(command) for command in result.commands] == [expected_argv]
    assert (
        tuple((assignment.name, assignment.value) for assignment in result.assignments)
        == expected_assignments
    )
    assert result.issues == ()


def test_tilde_after_a_non_assignment_equals_in_the_value_remains_exact() -> None:
    result, _budget, _unit = _analyze(b"A=foo=~/repo cmd\n")

    assert [(assignment.name, assignment.value) for assignment in result.assignments] == [
        ("A", dependency_types.StaticValue.exact(b"foo=~/repo"))
    ]
    assert [_argv_bytes(command) for command in result.commands] == [(b"cmd",)]
    assert result.issues == ()


@pytest.mark.parametrize(
    ("raw", "expected_argv", "expected_assignments"),
    [
        (b'printf ""~/file\n', (b"printf", b"~/file"), ()),
        (
            b'A=""~/repo cmd\n',
            (b"cmd",),
            (("A", dependency_types.StaticValue.exact(b"~/repo")),),
        ),
        (
            b'A=/bin:""~/repo cmd\n',
            (b"cmd",),
            (("A", dependency_types.StaticValue.exact(b"/bin:~/repo")),),
        ),
    ],
)
def test_zero_length_quoted_fragments_block_tilde_expansion(
    raw: bytes,
    expected_argv: tuple[bytes, ...],
    expected_assignments: tuple[tuple[str, dependency_types.StaticValue], ...],
) -> None:
    result, _budget, _unit = _analyze(raw)

    assert [_argv_bytes(command) for command in result.commands] == [expected_argv]
    assert (
        tuple((assignment.name, assignment.value) for assignment in result.assignments)
        == expected_assignments
    )
    assert result.issues == ()


def test_quoted_and_escaped_tilde_and_pathname_characters_remain_exact() -> None:
    result, _budget, _unit = _analyze(
        b"printf \"~/file\" 'file*' file\\? file\\[ab\\] \\~/file file\\*\n"
        b"A=\\~/repo cmd\n"
        b"B=/bin:\\~/bin cmd\n"
        b'C="/bin:~/bin" cmd\n'
    )

    assert [_argv_bytes(command) for command in result.commands] == [
        (b"printf", b"~/file", b"file*", b"file?", b"file[ab]", b"~/file", b"file*"),
        (b"cmd",),
        (b"cmd",),
        (b"cmd",),
    ]
    assert [
        (assignment.name, assignment.value.exact_bytes) for assignment in result.assignments
    ] == [
        ("A", b"~/repo"),
        ("B", b"/bin:~/bin"),
        ("C", b"/bin:~/bin"),
    ]
    assert result.issues == ()


def test_missing_syntax_is_localized_without_discarding_proven_commands() -> None:
    result, _budget, _unit = _analyze(b"if condition; then body;\n")

    assert [command.argv[0].exact_bytes for command in result.commands] == [
        b"condition",
        b"body",
    ]
    assert [(issue.reason, issue.span.start_line) for issue in result.issues] == [
        (dependency_types.ShellIssueReason.SYNTAX_ERROR, 1)
    ]
    assert result.work_items[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL


def test_preparse_resource_denial_is_skipped_without_calling_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = DependencyWorkBudget()
    extraction = _extract("scripts/limited.sh", b"printf ok\n", budget=budget)
    unit = extraction.units[0]
    file_budget = budget.for_file(unit.origin_span.path)
    assert file_budget.reserve_shell_parse(len(unit.raw_bytes)) is None
    assert file_budget.reserve_shell_parse(len(unit.raw_bytes)) is None

    def unexpected_parse(_source: bytes, **_kwargs: Any) -> Any:
        raise AssertionError("pre-parse denial must not invoke the parser")

    monkeypatch.setattr(shell_frontend, "parse_bash_source", unexpected_parse)

    result = shell_frontend.analyze_shell_unit(unit, budget=budget)

    assert result.commands == ()
    assert [issue.reason for issue in result.issues] == [
        dependency_types.ShellIssueReason.RESOURCE_LIMIT
    ]
    assert result.work_items[0].outcome is dependency_types.ShellWorkOutcome.SKIPPED


def test_exact_and_one_over_traversal_ir_and_value_budgets() -> None:
    raw = b"printf value\n"
    baseline_budget = DependencyWorkBudget()
    unit = _extract("scripts/bounds.sh", raw, budget=baseline_budget).units[0]
    baseline = shell_frontend.analyze_shell_unit(unit, budget=baseline_budget)
    assert baseline.work_items[0].outcome is dependency_types.ShellWorkOutcome.COMPLETED
    baseline_file = baseline_budget.for_file(unit.origin_span.path)
    required = {
        dependency_types.DependencyWorkResource.SHELL_CST_VISITS: baseline_file.used_for_unit(
            unit, dependency_types.DependencyWorkResource.SHELL_CST_VISITS
        ),
        dependency_types.DependencyWorkResource.RETAINED_SHELL_IR: baseline_budget.used(
            dependency_types.DependencyWorkResource.RETAINED_SHELL_IR
        ),
        dependency_types.DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES: baseline_file.used(
            dependency_types.DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES
        ),
    }
    limits = {
        dependency_types.DependencyWorkResource.SHELL_CST_VISITS: (
            dependency_types.DEPENDENCY_SHELL_CST_VISIT_FACTOR * len(raw)
            + dependency_types.DEPENDENCY_SHELL_CST_VISIT_BASE
        ),
        dependency_types.DependencyWorkResource.RETAINED_SHELL_IR: (
            dependency_types.MAX_DEPENDENCY_RETAINED_SHELL_IR
        ),
        dependency_types.DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES: (
            dependency_types.MAX_DEPENDENCY_SHELL_VALUE_BYTES_PER_FILE
        ),
    }

    for resource, required_count in required.items():
        exact_budget = DependencyWorkBudget()
        exact_file = exact_budget.for_file(unit.origin_span.path)
        exact_file.register_shell_file_size(len(raw))
        precharge = limits[resource] - required_count
        if resource is dependency_types.DependencyWorkResource.SHELL_CST_VISITS:
            assert exact_file.charge_shell_cst_visits(unit, precharge) is None
        elif resource is dependency_types.DependencyWorkResource.RETAINED_SHELL_IR:
            assert exact_file.charge_retained_shell_ir(unit, precharge) is None
        else:
            assert exact_file.reserve_shell_value_bytes(precharge) is None
        exact = shell_frontend.analyze_shell_unit(unit, budget=exact_budget)
        assert exact.work_items[0].outcome is dependency_types.ShellWorkOutcome.COMPLETED

        over_budget = DependencyWorkBudget()
        over_file = over_budget.for_file(unit.origin_span.path)
        over_file.register_shell_file_size(len(raw))
        precharge += 1
        if resource is dependency_types.DependencyWorkResource.SHELL_CST_VISITS:
            assert over_file.charge_shell_cst_visits(unit, precharge) is None
        elif resource is dependency_types.DependencyWorkResource.RETAINED_SHELL_IR:
            assert over_file.charge_retained_shell_ir(unit, precharge) is None
        else:
            assert over_file.reserve_shell_value_bytes(precharge) is None
        over = shell_frontend.analyze_shell_unit(unit, budget=over_budget)
        assert over.work_items[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL
        assert dependency_types.ShellIssueReason.RESOURCE_LIMIT in {
            issue.reason for issue in over.issues
        }


def test_exact_parse_revisit_ceiling_calls_parser_twice_then_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = DependencyWorkBudget()
    unit = _extract("scripts/revisit.sh", b"printf ok\n", budget=budget).units[0]
    real_parse = shell_frontend.parse_bash_source
    calls = 0

    def recording_parse(source: bytes, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return real_parse(source, **kwargs)

    monkeypatch.setattr(shell_frontend, "parse_bash_source", recording_parse)

    first = shell_frontend.analyze_shell_unit(unit, budget=budget)
    second = shell_frontend.analyze_shell_unit(unit, budget=budget)
    denied = shell_frontend.analyze_shell_unit(unit, budget=budget)

    assert [first.work_items[0].outcome, second.work_items[0].outcome] == [
        dependency_types.ShellWorkOutcome.COMPLETED,
        dependency_types.ShellWorkOutcome.COMPLETED,
    ]
    assert denied.work_items[0].outcome is dependency_types.ShellWorkOutcome.SKIPPED
    assert calls == 2


def test_runtime_parser_failure_is_one_local_partial_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = DependencyWorkBudget()
    unit = _extract("scripts/runtime.sh", b"printf ok\n", budget=budget).units[0]

    def cancelled(_source: bytes, **_kwargs: Any) -> Any:
        raise shell_frontend.ShellParserError(
            outcome=shell_frontend.ShellParserOutcome.PARTIAL,
            reason=shell_frontend.ShellParserFailureReason.RUNTIME_LIMIT,
            deadline_tripped=True,
        )

    monkeypatch.setattr(shell_frontend, "parse_bash_source", cancelled)

    result = shell_frontend.analyze_shell_unit(unit, budget=budget)

    assert result.commands == ()
    assert [issue.reason for issue in result.issues] == [
        dependency_types.ShellIssueReason.RUNTIME_LIMIT
    ]
    assert result.work_items[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL


@pytest.mark.parametrize(
    ("parser_outcome", "work_outcome"),
    [
        (
            shell_frontend.ShellParserOutcome.FAILED,
            dependency_types.ShellWorkOutcome.FAILED,
        ),
        (
            shell_frontend.ShellParserOutcome.PARTIAL,
            dependency_types.ShellWorkOutcome.PARTIAL,
        ),
    ],
)
def test_parser_unavailable_preserves_failed_or_preclassified_partial_outcome(
    monkeypatch: pytest.MonkeyPatch,
    parser_outcome: shell_frontend.ShellParserOutcome,
    work_outcome: dependency_types.ShellWorkOutcome,
) -> None:
    budget = DependencyWorkBudget()
    unit = _extract("scripts/unavailable.sh", b"printf ok\n", budget=budget).units[0]

    def unavailable(_source: bytes, **_kwargs: Any) -> Any:
        raise shell_frontend.ShellParserError(
            outcome=parser_outcome,
            reason=shell_frontend.ShellParserFailureReason.SHELL_PARSER_UNAVAILABLE,
            deadline_tripped=False,
        )

    monkeypatch.setattr(shell_frontend, "parse_bash_source", unavailable)

    result = shell_frontend.analyze_shell_unit(unit, budget=budget)

    assert result.commands == ()
    assert [(issue.reason, issue.outcome) for issue in result.issues] == [
        (dependency_types.ShellIssueReason.SHELL_PARSER_UNAVAILABLE, work_outcome)
    ]
    assert result.work_items[0].outcome is work_outcome


@pytest.mark.timeout(10)
def test_deep_shell_nesting_is_walked_without_python_recursion() -> None:
    depth = 2_000
    raw = b"(" * depth + b"printf ok" + b")" * depth + b"\n"

    result, _budget, _unit = _analyze(raw, path="scripts/deep.sh")

    assert len(result.work_items) == 1
    assert result.work_items[0].outcome in {
        dependency_types.ShellWorkOutcome.COMPLETED,
        dependency_types.ShellWorkOutcome.PARTIAL,
    }


def test_sequential_state_resolves_only_after_assignment_and_preserves_unbound() -> None:
    private, _budget, _unit = _analyze_private(
        b'probe "$A"\nA=one\nprobe "pre${A}/post"\nA=\nprobe "$A"\nunset A\nprobe "$A"\n'
    )

    assert [command.argv[1] for command in private.public.commands] == [
        dependency_types.StaticValue.unknown(),
        dependency_types.StaticValue.exact(b"preone/post"),
        dependency_types.StaticValue.exact(b""),
        dependency_types.StaticValue.unbound(),
    ]
    assert private.public.issues == ()


def test_prefix_overlay_is_site_local_and_export_updates_are_retained_privately() -> None:
    private, _budget, _unit = _analyze_private(
        b'A=base\nA=temp probe "$A"\nprobe "$A"\nexport E=child\nexport E\nexport -n E\n'
    )

    probes = [
        command
        for command in private.program.commands
        if command.site.argv[0] == dependency_types.StaticValue.exact(b"probe")
    ]
    assert [command.site.argv[1] for command in probes] == [
        dependency_types.StaticValue.exact(b"base"),
        dependency_types.StaticValue.exact(b"base"),
    ]
    assert [(binding.name, binding.value) for binding in probes[0].prefix_bindings] == [
        ("A", dependency_types.StaticValue.exact(b"temp"))
    ]
    assert probes[1].prefix_bindings == ()
    assert [
        update.binding.export_state.value
        for update in private.program.state_updates
        if update.binding.name == "E"
    ] == ["exported", "exported", "unexported"]
    frame_ids = {frame.frame_id for frame in private.program.state_frames}
    assert {command.state_frame_id for command in private.program.commands} <= frame_ids
    assert {update.frame_id for update in private.program.state_updates} <= frame_ids


@pytest.mark.parametrize(
    "raw",
    [
        b'A=base\nif cond; then A=branch; fi\nprobe "$A"\n',
        b'A=base\nA=branch && cond\nprobe "$A"\n',
        b'A=base\nwhile cond; do A=branch; done\nprobe "$A"\n',
        b'A=base\n{ A=branch; }\nprobe "$A"\n',
        b'A=base\nA=branch & wait\nprobe "$A"\n',
    ],
)
def test_uncertain_control_flow_widens_written_names(raw: bytes) -> None:
    private, _budget, _unit = _analyze_private(raw)

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1] == dependency_types.StaticValue.unknown()


@pytest.mark.parametrize(
    "isolated",
    [
        b"( A=inner )",
        b"A=inner | consume",
        b"capture $(A=inner; produce)",
        b"capture <(A=inner; produce)",
    ],
)
def test_child_process_writes_do_not_escape_to_parent_state(isolated: bytes) -> None:
    private, _budget, _unit = _analyze_private(b"A=outer\n" + isolated + b'\nprobe "$A"\n')

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1] == dependency_types.StaticValue.exact(b"outer")


def test_function_definition_order_scope_and_shadow_resolution_are_retained() -> None:
    private, _budget, _unit = _analyze_private(
        b"tool before\n"
        b'tool() { A=inner; tool inside "$A"; }\n'
        b"tool after\n"
        b"if cond; then tool() { neutral; }; fi\n"
        b"tool ambiguous\n"
        b"A=outer\n"
        b'probe "$A"\n'
    )

    tool_commands = [
        command
        for command in private.program.commands
        if command.site.argv[0] == dependency_types.StaticValue.exact(b"tool")
    ]
    assert [command.resolution.kind.value for command in tool_commands] == [
        "external",
        "function",
        "function",
        "ambiguous",
    ]
    assert tool_commands[1].site.argv[-1] == dependency_types.StaticValue.exact(b"inner")
    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1] == dependency_types.StaticValue.exact(b"outer")


def test_literal_shell_and_eval_programs_share_state_with_defined_propagation() -> None:
    private, _budget, _unit = _analyze_private(
        b"export A=exported\n"
        b"A=overlay sh -c 'probe \"$A\"'\n"
        b"A=outer\n"
        b"eval 'probe \"$A\"; A=evaluated'\n"
        b'probe "$A"\n'
    )

    probes = [
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    ]
    assert [command.argv[1] for command in probes] == [
        dependency_types.StaticValue.exact(b"overlay"),
        dependency_types.StaticValue.exact(b"outer"),
        dependency_types.StaticValue.exact(b"evaluated"),
    ]
    assert [command.provenance for command in probes] == [
        dependency_types.SiteProvenance.NESTED_LITERAL,
        dependency_types.SiteProvenance.NESTED_LITERAL,
        dependency_types.SiteProvenance.FILE_SUFFIX,
    ]


def test_nested_literal_depth_three_is_rejected_before_the_fourth_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"export A='probe ok'\nexport B='sh -c \"$A\"'\nexport C='sh -c \"$B\"'\nsh -c \"$C\"\n"
    real_parse = shell_frontend.parse_bash_source
    parsed: list[bytes] = []

    def recording_parse(source: bytes, **kwargs: Any) -> Any:
        parsed.append(source)
        return real_parse(source, **kwargs)

    monkeypatch.setattr(shell_frontend, "parse_bash_source", recording_parse)

    private, _budget, _unit = _analyze_private(raw)

    assert len(parsed) == 3
    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.RESOURCE_LIMIT
    ]
    assert private.public.issues[0].exhaustion == dependency_types.DependencyWorkExhaustion(
        dependency_types.DependencyWorkResource.SHELL_NESTED_DEPTH,
        3,
        dependency_types.MAX_DEPENDENCY_SHELL_NESTED_LITERAL_DEPTH,
    )


@pytest.mark.parametrize(
    "raw",
    [
        b"env -S 'probe hidden'\n",
        b"xargs probe\n",
        b"printf '%s' 'probe hidden' | sh\n",
        b'sh -c "$PROGRAM"\n',
        b'eval "$PROGRAM"\n',
        b"eval 'one' 'two'\n",
    ],
)
def test_data_constructed_or_dynamic_commands_are_explicit_limitations(raw: bytes) -> None:
    private, _budget, _unit = _analyze_private(raw)

    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]
    assert private.public.work_items[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL


def test_simple_expansion_state_is_exact_only_for_supported_site_time_shapes() -> None:
    private, _budget, _unit = _analyze_private(
        b'A=one\nB=two\nprobe "$A" "${A}" "pre${A}/${B}post" "${A:-fallback}" "$1"\n'
    )

    probe = private.public.commands[-1]
    assert probe.argv[1:] == (
        dependency_types.StaticValue.exact(b"one"),
        dependency_types.StaticValue.exact(b"one"),
        dependency_types.StaticValue.exact(b"preone/twopost"),
        dependency_types.StaticValue.unknown(),
        dependency_types.StaticValue.unknown(),
    )


def test_mixed_event_tape_preserves_substitution_order_and_control_roles() -> None:
    private, _budget, _unit = _analyze_private(
        b'A=before\nA=$(probe "$A")\nprobe "$A"\nB=one && C=two || D=three\nE=four & F=five\n'
    )

    probes = [
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    ]
    assert [probe.argv[1] for probe in probes] == [
        dependency_types.StaticValue.exact(b"before"),
        dependency_types.StaticValue.unknown(),
    ]
    roles_by_name = {
        event.name: event.role.value
        for event in private.program.state_events
        if event.kind.value == "assignment" and event.name is not None
    }
    assert roles_by_name == {
        "A": "straight",
        "B": "boolean_left",
        "C": "boolean_right",
        "D": "boolean_right",
        "E": "async",
        "F": "straight",
    }


def test_export_unset_shapes_and_later_definite_recovery_are_conservative() -> None:
    private, _budget, _unit = _analyze_private(
        b"export A=one\n"
        b"A=two\n"
        b"export -n A\n"
        b"export A\n"
        b"unset A\n"
        b'probe "$A"\n'
        b"unset B C\n"
        b"unset -v D\n"
        b"A=recovered\n"
        b'probe "$A"\n'
    )

    probes = [
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    ]
    assert [probe.argv[1] for probe in probes] == [
        dependency_types.StaticValue.unbound(),
        dependency_types.StaticValue.exact(b"recovered"),
    ]
    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS,
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS,
    ]
    assert [
        (update.binding.name, update.binding.export_state.value)
        for update in private.program.state_updates
        if update.binding.name == "A"
    ] == [
        ("A", "exported"),
        ("A", "exported"),
        ("A", "unexported"),
        ("A", "exported"),
        ("A", "unexported"),
        ("A", "unexported"),
    ]


def test_function_shadowing_prevents_nested_shell_and_eval_interpretation() -> None:
    private, _budget, _unit = _analyze_private(
        b"sh() { neutral; }\neval() { neutral; }\nsh -c 'probe hidden'\neval 'probe hidden'\n"
    )

    assert [
        command.resolution.kind.value
        for command in private.program.commands
        if command.site.argv[0]
        in {
            dependency_types.StaticValue.exact(b"sh"),
            dependency_types.StaticValue.exact(b"eval"),
        }
    ] == ["function", "function"]
    assert all(
        command.site.argv[0] != dependency_types.StaticValue.exact(b"probe")
        for command in private.program.commands
    )


def test_nested_literal_maps_compose_to_markdown_physical_bytes() -> None:
    raw = b"heading\n```bash\nsh -c 'probe child'\n```\n"
    budget = DependencyWorkBudget()
    unit = _extract("docs/state.md", raw, budget=budget).units[0]

    private = shell_frontend._analyze_shell_unit(unit, budget=budget)

    nested = private.program.nested_programs[0]
    assert nested.depth == 1
    assert nested.unit.raw_bytes == b"probe child"
    assert nested.unit.source_map is not None
    assert nested.unit.source_map.map_range(0, len(b"probe child")) == (
        dependency_types.SourceSpan(
            "docs/state.md",
            raw.index(b"probe child"),
            raw.index(b"probe child") + len(b"probe child"),
            3,
            3,
            start_column=7,
            end_column=18,
        )
    )
    probe = next(
        command
        for command in private.public.commands
        if command.provenance is dependency_types.SiteProvenance.NESTED_LITERAL
    )
    assert probe.span == nested.unit.source_map.map_range(0, len(b"probe child"))


def test_escape_folded_nested_map_uses_only_affine_surviving_runs() -> None:
    raw = b"sh -c probe\\ ok\n"
    private, _budget, _unit = _analyze_private(raw)

    nested = private.program.nested_programs[0]
    source_map = nested.unit.source_map
    assert nested.unit.raw_bytes == b"probe ok"
    assert source_map is not None
    assert all(
        entry.child_end_byte - entry.child_start_byte
        == entry.physical_end_byte - entry.physical_start_byte
        for entry in source_map.entries
    )
    assert (
        b"".join(
            raw[entry.physical_start_byte : entry.physical_end_byte] for entry in source_map.entries
        )
        == b"probe ok"
    )
    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]


@pytest.mark.parametrize(
    ("failure_reason", "issue_reason"),
    [
        (
            shell_frontend.ShellParserFailureReason.RUNTIME_LIMIT,
            dependency_types.ShellIssueReason.RUNTIME_LIMIT,
        ),
        (
            shell_frontend.ShellParserFailureReason.SHELL_PARSER_UNAVAILABLE,
            dependency_types.ShellIssueReason.SHELL_PARSER_UNAVAILABLE,
        ),
    ],
)
def test_nested_parser_failure_is_local_partial_and_preserves_other_sites(
    monkeypatch: pytest.MonkeyPatch,
    failure_reason: shell_frontend.ShellParserFailureReason,
    issue_reason: dependency_types.ShellIssueReason,
) -> None:
    real_parse = shell_frontend.parse_bash_source
    calls = 0

    def fail_child(source: bytes, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_parse(source, **kwargs)
        raise shell_frontend.ShellParserError(
            outcome=shell_frontend.ShellParserOutcome.FAILED,
            reason=failure_reason,
            deadline_tripped=(
                failure_reason is shell_frontend.ShellParserFailureReason.RUNTIME_LIMIT
            ),
        )

    monkeypatch.setattr(shell_frontend, "parse_bash_source", fail_child)

    private, _budget, _unit = _analyze_private(b"sh -c 'probe nested'\nprobe retained\n")

    assert calls == 2
    assert [issue.reason for issue in private.public.issues] == [issue_reason]
    assert private.public.issues[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL
    assert _argv_bytes(private.public.commands[-1]) == (b"probe", b"retained")


def test_nested_work_charges_root_unit_and_one_over_parser_budget_is_localized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"sh -c 'probe one'\nsh -c 'probe two'\n"
    budget = DependencyWorkBudget()
    unit = _extract("scripts/nested-budget.sh", raw, budget=budget).units[0]
    file_budget = budget.for_file(unit.origin_span.path)
    file_budget.register_shell_file_size(len(raw))
    for _ in range(dependency_types.MAX_DEPENDENCY_SHELL_PARSER_CALLS_PER_FILE - 2):
        assert file_budget.reserve_shell_parse(0) is None
    charged_unit_ids: list[str] = []
    real_charge = dependency_types.DependencyFileBudget.charge_shell_cst_visits

    def record_cst(self: Any, accounting_unit: Any, count: int) -> Any:
        charged_unit_ids.append(accounting_unit.unit_id)
        return real_charge(self, accounting_unit, count)

    monkeypatch.setattr(
        dependency_types.DependencyFileBudget,
        "charge_shell_cst_visits",
        record_cst,
    )

    private = shell_frontend._analyze_shell_unit(unit, budget=budget)

    assert set(charged_unit_ids) == {unit.unit_id}
    assert [
        item.outcome for item in private.public.work_items if item.kind.value == "nested_literal"
    ] == [
        dependency_types.ShellWorkOutcome.COMPLETED,
        dependency_types.ShellWorkOutcome.SKIPPED,
    ]
    assert private.public.issues[-1].exhaustion == dependency_types.DependencyWorkExhaustion(
        dependency_types.DependencyWorkResource.SHELL_PARSER_CALLS,
        dependency_types.MAX_DEPENDENCY_SHELL_PARSER_CALLS_PER_FILE + 1,
        dependency_types.MAX_DEPENDENCY_SHELL_PARSER_CALLS_PER_FILE,
    )


def test_branch_local_state_is_exact_then_widens_after_if() -> None:
    private, _budget, _unit = _analyze_private(
        b'A=base\nif cond; then A=branch; probe "$A"; fi\nprobe "$A"\n'
    )

    probes = [
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    ]
    assert [probe.argv[1] for probe in probes] == [
        dependency_types.StaticValue.exact(b"branch"),
        dependency_types.StaticValue.unknown(),
    ]


def test_conditional_export_widens_only_attribute_and_assignment_preserves_export() -> None:
    private, _budget, _unit = _analyze_private(
        b"A=base\n"
        b"if cond; then export A; fi\n"
        b'probe "$A"\n'
        b"export B=base\n"
        b"if cond; then B=branch; fi\n"
        b'probe "$B"\n'
    )

    probes = [
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    ]
    assert [probe.argv[1] for probe in probes] == [
        dependency_types.StaticValue.exact(b"base"),
        dependency_types.StaticValue.unknown(),
    ]
    last_by_name = {update.binding.name: update.binding for update in private.program.state_updates}
    assert last_by_name["A"].value == dependency_types.StaticValue.exact(b"base")
    assert last_by_name["A"].export_state.value == "unknown"
    assert last_by_name["B"].value == dependency_types.StaticValue.unknown()
    assert last_by_name["B"].export_state.value == "exported"


def test_conditional_function_is_visible_inside_branch_and_ambiguous_after() -> None:
    private, _budget, _unit = _analyze_private(
        b"if cond; then helper-name() { neutral; }; helper-name inside; fi\nhelper-name outside\n"
    )

    calls = [
        command
        for command in private.program.commands
        if command.site.argv[0] == dependency_types.StaticValue.exact(b"helper-name")
    ]
    assert [call.resolution.kind.value for call in calls] == ["function", "ambiguous"]


@pytest.mark.parametrize(
    "isolated",
    [
        b"(A=inner)",
        b"A=inner | consume",
        b"capture $(A=inner; produce)",
        b"capture <(A=inner; produce)",
    ],
)
def test_isolated_write_inside_function_does_not_escape(isolated: bytes) -> None:
    private, _budget, _unit = _analyze_private(b"f() { A=outer; " + isolated + b'; probe "$A"; }\n')

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1] == dependency_types.StaticValue.exact(b"outer")


def test_dynamic_unset_invalidates_stale_exact_state_then_definite_write_recovers() -> None:
    private, _budget, _unit = _analyze_private(
        b'TARGET=X\nX=one\nunset "$TARGET"\nprobe "$X"\nX=recovered\nprobe "$X"\n'
    )

    probes = [
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    ]
    assert [probe.argv[1] for probe in probes] == [
        dependency_types.StaticValue.unknown(),
        dependency_types.StaticValue.exact(b"recovered"),
    ]
    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]


@pytest.mark.parametrize("keyword", [b"export", b"declare", b"readonly", b"typeset"])
def test_declaration_argv_and_assignment_share_resolved_value(keyword: bytes) -> None:
    private, _budget, _unit = _analyze_private(b"B=one\n" + keyword + b" A=$B\n")

    declaration = private.public.commands[-1]
    assignment = next(
        assignment for assignment in private.public.assignments if assignment.name == "A"
    )
    assert declaration.argv[1] == dependency_types.StaticValue.exact(b"A=one")
    assert assignment.value == (
        dependency_types.StaticValue.unknown()
        if keyword == b"readonly"
        else dependency_types.StaticValue.exact(b"one")
    )


def test_mutually_exclusive_else_branch_restarts_from_pre_if_state() -> None:
    private, _budget, _unit = _analyze_private(
        b'A=base\nif cond; then A=then; else probe "$A"; A=else; fi\nprobe "$A"\n'
    )

    probes = [
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    ]
    assert [probe.argv[1] for probe in probes] == [
        dependency_types.StaticValue.exact(b"base"),
        dependency_types.StaticValue.unknown(),
    ]


def test_c_style_loop_postfix_write_does_not_leave_stale_exact_state() -> None:
    private, _budget, _unit = _analyze_private(
        b'A=base\nfor ((i=0; i<1; A++)); do neutral; done\nprobe "$A"\n'
    )

    probe = private.public.commands[-1]
    assert probe.argv[1] == dependency_types.StaticValue.unknown()
    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]


def test_export_n_assignment_is_not_interpreted_as_supported_export() -> None:
    private, _budget, _unit = _analyze_private(b"export -n A=one\n")

    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]
    last = private.program.state_updates[-1].binding
    assert last.name == "A"
    assert last.value == dependency_types.StaticValue.unknown()
    assert last.export_state.value == "unknown"


@pytest.mark.parametrize("shell", [b"bash", b"sh", b"dash"])
@pytest.mark.parametrize("option", [b"-c", b"-lc"])
def test_literal_shell_forms_receive_only_exports_plus_prefix(
    shell: bytes,
    option: bytes,
) -> None:
    private, _budget, _unit = _analyze_private(
        b"HIDDEN=private\n"
        b"export SHARED=exported\n"
        b"OVERLAY=persistent\n"
        + b"OVERLAY=temporary "
        + shell
        + b" "
        + option
        + b' \'probe "$HIDDEN" "$SHARED" "$OVERLAY"\' name arg\n'
        + b'probe "$OVERLAY"\n'
    )

    nested_probe = next(
        command
        for command in private.public.commands
        if command.provenance is dependency_types.SiteProvenance.NESTED_LITERAL
    )
    assert nested_probe.argv[1:] == (
        dependency_types.StaticValue.unknown(),
        dependency_types.StaticValue.exact(b"exported"),
        dependency_types.StaticValue.exact(b"temporary"),
    )
    assert private.public.commands[-1].argv[1] == dependency_types.StaticValue.exact(b"persistent")


def test_repeated_variable_payload_provenance_is_rejected_without_child_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_parse = shell_frontend.parse_bash_source
    parsed: list[bytes] = []

    def record_parse(source: bytes, **kwargs: Any) -> Any:
        parsed.append(source)
        return real_parse(source, **kwargs)

    monkeypatch.setattr(shell_frontend, "parse_bash_source", record_parse)

    private, _budget, _unit = _analyze_private(b"P='probe once'\neval \"$P$P\"\nprobe retained\n")

    assert parsed == [b"P='probe once'\neval \"$P$P\"\nprobe retained\n"]
    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]
    assert _argv_bytes(private.public.commands[-1]) == (b"probe", b"retained")


def test_root_and_nested_parses_receive_the_same_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_parse = shell_frontend.parse_bash_source
    deadlines: list[float | None] = []
    meaningful: list[bool] = []

    def record_parse(source: bytes, **kwargs: Any) -> Any:
        deadlines.append(kwargs.get("deadline_monotonic"))
        meaningful.append(kwargs.get("meaningful_work", False))
        return real_parse(source, **kwargs)

    monkeypatch.setattr(shell_frontend, "parse_bash_source", record_parse)
    budget = DependencyWorkBudget()
    unit = _extract("scripts/deadline.sh", b"sh -c 'probe child'\n", budget=budget).units[0]

    shell_frontend._analyze_shell_unit(unit, budget=budget, deadline_monotonic=9_999_999_999.0)

    assert deadlines == [9_999_999_999.0, 9_999_999_999.0]
    assert meaningful == [False, True]


def test_isolated_write_inside_conditional_does_not_pollute_branch_overlay() -> None:
    private, _budget, _unit = _analyze_private(
        b'A=base\nif cond; then A=branch; (A=isolated); probe "$A"; fi\nprobe "$A"\n'
    )

    probes = [
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    ]
    assert [probe.argv[1] for probe in probes] == [
        dependency_types.StaticValue.exact(b"branch"),
        dependency_types.StaticValue.unknown(),
    ]


def test_case_arm_restarts_from_pre_case_state() -> None:
    private, _budget, _unit = _analyze_private(
        b'A=base\ncase "$INPUT" in one) A=one ;; two) probe "$A"; A=two ;; esac\nprobe "$A"\n'
    )

    probes = [
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    ]
    assert [probe.argv[1] for probe in probes] == [
        dependency_types.StaticValue.exact(b"base"),
        dependency_types.StaticValue.unknown(),
    ]


def test_unsupported_multi_export_and_unset_f_do_not_retain_false_precision() -> None:
    private, _budget, _unit = _analyze_private(
        b"A=one\nB=two\nexport A B\nf() { neutral; }\nunset -f f\nf maybe\n"
    )

    last_by_name = {update.binding.name: update.binding for update in private.program.state_updates}
    assert last_by_name["A"].export_state.value == "unknown"
    assert last_by_name["B"].export_state.value == "unknown"
    f_call = next(
        command
        for command in private.program.commands
        if command.site.argv[0] == dependency_types.StaticValue.exact(b"f")
    )
    assert f_call.resolution.kind.value == "ambiguous"
    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS,
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS,
    ]


@pytest.mark.parametrize("interpreter", [b"eval", b"sh"])
def test_function_body_preserves_visible_shadow_for_nested_interpreters(
    interpreter: bytes,
) -> None:
    invocation = b"eval 'probe hidden'" if interpreter == b"eval" else b"sh -c 'probe hidden'"
    private, _budget, _unit = _analyze_private(
        interpreter + b"() { neutral; }\n" + b"f() { " + invocation + b"; }\n"
    )

    call = next(
        command
        for command in private.program.commands
        if command.site.argv[0] == dependency_types.StaticValue.exact(interpreter)
    )
    assert call.resolution.kind.value == "function"
    assert private.program.nested_programs == ()
    assert all(
        command.site.argv[0] != dependency_types.StaticValue.exact(b"probe")
        for command in private.program.commands
    )


def test_nested_command_context_retains_inherited_exports_and_parent_identity() -> None:
    private, _budget, _unit = _analyze_private(b"export TOKEN=one\nsh -c 'manager install'\n")

    nested = private.program.nested_programs[0]
    inherited = {binding.name: binding for binding in nested.program.initial_bindings}
    manager = nested.program.commands[0]
    assert inherited["TOKEN"].value == dependency_types.StaticValue.exact(b"one")
    assert inherited["TOKEN"].export_state.value == "exported"
    assert nested.parent_program_id == private.program.program_id
    assert nested.parent_command_start_byte == private.public.commands[1].span.start_byte
    assert manager.program_id == nested.program.program_id == nested.unit.unit_id


def test_eval_propagation_is_replayable_at_later_parent_command() -> None:
    private, _budget, _unit = _analyze_private(b"eval 'export TOKEN=one'\nmanager install\n")

    token_updates = [
        update for update in private.program.state_updates if update.binding.name == "TOKEN"
    ]
    manager = next(
        command
        for command in private.program.commands
        if command.site.argv[0] == dependency_types.StaticValue.exact(b"manager")
    )
    assert token_updates[-1].binding.value == dependency_types.StaticValue.exact(b"one")
    assert token_updates[-1].binding.export_state.value == "exported"
    assert manager.state_update_order > token_updates[-1].order
    assert manager.program_id == private.program.program_id


@pytest.mark.parametrize(
    "eval_command",
    [
        b'eval "$DYNAMIC"',
        b"eval 'A=two;' ':'",
    ],
)
def test_unsupported_eval_invalidates_state_and_external_resolution(
    eval_command: bytes,
) -> None:
    private, _budget, _unit = _analyze_private(
        b"A=one\n" + eval_command + b"\n" + b'probe "$A"\n' + b"sh -c 'probe hidden'\n"
    )

    visible_probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    shell = next(
        command
        for command in private.program.commands
        if command.site.argv[0] == dependency_types.StaticValue.exact(b"sh")
    )
    assert visible_probe.argv[1] == dependency_types.StaticValue.unknown()
    assert shell.resolution.kind.value == "ambiguous"
    assert private.program.nested_programs == ()


@pytest.mark.parametrize(
    "eval_command",
    [
        b"eval 'A=two'",
        b'eval "$DYNAMIC"',
    ],
)
def test_ambiguous_eval_applies_state_and_function_namespace_barrier(
    eval_command: bytes,
) -> None:
    private, _budget, _unit = _analyze_private(
        b"A=one\nif cond; then eval() { neutral; }; fi\n"
        + eval_command
        + b"\nprobe \"$A\"\nsh -c 'probe hidden'\n"
    )

    visible_probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    eval_site = next(
        command
        for command in private.program.commands
        if command.site.argv[0] == dependency_types.StaticValue.exact(b"eval")
        and command.function_id is None
    )
    shell = next(
        command
        for command in private.program.commands
        if command.site.argv[0] == dependency_types.StaticValue.exact(b"sh")
    )
    assert eval_site.resolution.kind.value == "ambiguous"
    assert visible_probe.argv[1] == dependency_types.StaticValue.unknown()
    assert shell.resolution.kind.value == "ambiguous"
    assert private.program.nested_programs == ()
    assert dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS in {
        issue.reason for issue in private.public.issues
    }


@pytest.mark.parametrize("update", [b"++A", b"A+=1"])
def test_c_style_non_postfix_update_does_not_leave_stale_exact_state(
    update: bytes,
) -> None:
    private, _budget, _unit = _analyze_private(
        b"A=one\nfor ((i=0; i<1; " + update + b')); do neutral; done\nprobe "$A"\n'
    )

    assert private.public.commands[-1].argv[1] == dependency_types.StaticValue.unknown()
    assert dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS in {
        issue.reason for issue in private.public.issues
    }


@pytest.mark.parametrize(
    "loop",
    [
        b"for ((++A; i<1; i++)); do neutral; done",
        b"for ((i=0; (A+=1)<2; i++)); do neutral; done",
        b"for ((i=0; A++<2; )); do neutral; done",
    ],
)
def test_c_style_initializer_or_condition_write_does_not_leave_stale_exact_state(
    loop: bytes,
) -> None:
    private, _budget, _unit = _analyze_private(b"A=one\n" + loop + b'\nprobe "$A"\n')

    assert private.public.commands[-1].argv[1] == dependency_types.StaticValue.unknown()
    assert dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS in {
        issue.reason for issue in private.public.issues
    }


@pytest.mark.parametrize(
    "loop",
    [
        b"for ((A=1; A<2; A++)); do neutral; done",
        b"for ((A=1; (A+=1)<3; )); do neutral; done",
        b"for ((A=1; A++<3; )); do neutral; done",
    ],
)
def test_c_style_header_barrier_cannot_be_overwritten_by_initializer_exact_recovery(
    loop: bytes,
) -> None:
    private, _budget, _unit = _analyze_private(b"A=1\n" + loop + b'\nprobe "$A"\n')

    assert private.public.commands[-1].argv[1] == dependency_types.StaticValue.unknown()


def test_eval_child_sees_prefix_overlay_without_leaking_prefix_binding() -> None:
    private, _budget, _unit = _analyze_private(b'A=one\nA=two eval \'B=$A\'\nprobe "$B" "$A"\n')

    nested = private.program.nested_programs[0]
    initial = {binding.name: binding for binding in nested.program.initial_bindings}
    assert initial["A"].value == dependency_types.StaticValue.exact(b"two")
    assert private.public.commands[-1].argv[1:] == (
        dependency_types.StaticValue.exact(b"two"),
        dependency_types.StaticValue.exact(b"one"),
    )


@pytest.mark.parametrize(
    "raw",
    [
        b"f() { eval 'probe hidden'; }\neval() { neutral; }\nf\n",
        b"eval() { neutral; }\nf() { eval 'probe visible'; }\nunset -f eval\nf\n",
    ],
)
def test_function_body_resolution_does_not_freeze_definition_time_global_shadow(
    raw: bytes,
) -> None:
    private, _budget, _unit = _analyze_private(raw)

    body_eval = next(
        command
        for command in private.program.commands
        if command.site.argv[0] == dependency_types.StaticValue.exact(b"eval")
        and command.function_id is not None
    )
    assert body_eval.resolution.kind.value == "ambiguous"
    assert private.program.nested_programs == ()
    assert dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS in {
        issue.reason for issue in private.public.issues
    }


def test_function_body_future_mutation_detects_state_resolved_eval_name() -> None:
    private, _budget, _unit = _analyze_private(
        b"f() { sh -c 'probe hidden'; }\nCMD=eval\n\"$CMD\" 'sh() { neutral; }'\nf\n"
    )

    body_shell = next(
        command
        for command in private.program.commands
        if command.site.argv[0] == dependency_types.StaticValue.exact(b"sh")
        and command.function_id is not None
    )
    assert body_shell.resolution.kind.value == "ambiguous"
    assert all(
        command.site.argv[0] != dependency_types.StaticValue.exact(b"probe")
        for command in private.program.commands
    )


@pytest.mark.parametrize(
    "raw",
    [
        b"outer() { f() { eval 'probe hidden'; }; eval() { neutral; }; f; }\nouter\n",
        b"outer() { eval() { neutral; }; f() { eval 'probe visible'; }; "
        b"unset -f eval; f; }\nouter\n",
    ],
)
def test_nested_function_body_does_not_freeze_enclosing_function_mutation_timing(
    raw: bytes,
) -> None:
    private, _budget, _unit = _analyze_private(raw)

    inner_eval = next(
        command
        for command in private.program.commands
        if command.site.argv[0] == dependency_types.StaticValue.exact(b"eval")
        and command.function_id is not None
    )
    assert inner_eval.resolution.kind.value == "ambiguous"
    assert private.program.nested_programs == ()


def test_deep_nested_function_body_checks_all_enclosing_function_mutations() -> None:
    private, _budget, _unit = _analyze_private(
        b"outer() { mid() { inner() { eval 'probe hidden'; }; inner; }; "
        b"eval() { neutral; }; mid; }\nouter\n"
    )

    inner_eval = next(
        command
        for command in private.program.commands
        if command.site.argv[0] == dependency_types.StaticValue.exact(b"eval")
        and command.function_id is not None
    )
    assert inner_eval.resolution.kind.value == "ambiguous"
    assert private.program.nested_programs == ()


def test_variable_backed_eval_flattening_preserves_execution_order() -> None:
    private, _budget, _unit = _analyze_private(
        b"P='probe nested'\nprobe before\neval \"$P\"\nprobe after\n"
    )

    assert [command.site.argv[0].exact_bytes for command in private.program.execution_commands] == [
        b"probe",
        b"eval",
        b"probe",
        b"probe",
    ]
    assert [command.site.argv[1].exact_bytes for command in private.program.execution_commands] == [
        b"before",
        b"probe nested",
        b"nested",
        b"after",
    ]
    assert [nested.depth for nested in private.program.nested_programs] == [1]


@pytest.mark.parametrize(
    "raw",
    [
        b"A=outer\nA=new B=$A C=literal neutral\n",
        b"A=outer\nleft | A=new B=$A C=literal neutral\n",
    ],
)
def test_prefix_rhs_referencing_sibling_prefix_is_unknown_in_every_context(
    raw: bytes,
) -> None:
    private, _budget, _unit = _analyze_private(raw)

    command = next(
        candidate
        for candidate in private.program.commands
        if candidate.site.argv[0] == dependency_types.StaticValue.exact(b"neutral")
    )
    bindings = {binding.name: binding.value for binding in command.prefix_bindings}
    assert bindings == {
        "A": dependency_types.StaticValue.exact(b"new"),
        "B": dependency_types.StaticValue.unknown(),
        "C": dependency_types.StaticValue.exact(b"literal"),
    }


def test_duplicate_prefix_name_counts_as_a_sibling_dependency() -> None:
    private, _budget, _unit = _analyze_private(b"A=outer\nA=one A=$A neutral\n")

    command = next(
        candidate
        for candidate in private.program.commands
        if candidate.site.argv[0] == dependency_types.StaticValue.exact(b"neutral")
    )
    assert [binding.value for binding in command.prefix_bindings] == [
        dependency_types.StaticValue.exact(b"one"),
        dependency_types.StaticValue.unknown(),
    ]


@pytest.mark.parametrize("keyword", [b"export", b"declare", b"readonly", b"typeset"])
def test_declaration_assignments_resolve_atomically_from_precommand_snapshot(
    keyword: bytes,
) -> None:
    private, _budget, _unit = _analyze_private(
        b"A=old\n" + keyword + b' A=new B=$A\nprobe "$A" "$B"\n'
    )

    assert private.public.commands[-1].argv[1:] == (
        (
            dependency_types.StaticValue.unknown()
            if keyword == b"readonly"
            else dependency_types.StaticValue.exact(b"new")
        ),
        (
            dependency_types.StaticValue.unknown()
            if keyword == b"readonly"
            else dependency_types.StaticValue.exact(b"old")
        ),
    )


def test_local_assignments_resolve_atomically_from_function_snapshot() -> None:
    private, _budget, _unit = _analyze_private(
        b'f() { A=old; local A=new B=$A; probe "$A" "$B"; }\n'
    )

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1:] == (
        dependency_types.StaticValue.exact(b"new"),
        dependency_types.StaticValue.exact(b"old"),
    )


@pytest.mark.parametrize(
    ("setup", "command", "expected_b_export", "expected_issues"),
    [
        (b"B=two", b"export A=one B", "exported", ()),
        (
            b"export B=two",
            b"export A=one -n B",
            "unknown",
            (dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS,),
        ),
    ],
)
def test_mixed_export_assignments_and_bare_operands_update_every_narrow_operand(
    setup: bytes,
    command: bytes,
    expected_b_export: str,
    expected_issues: tuple[dependency_types.ShellIssueReason, ...],
) -> None:
    private, _budget, _unit = _analyze_private(setup + b"\n" + command + b"\n")

    last_by_name = {update.binding.name: update.binding for update in private.program.state_updates}
    assert last_by_name["A"].value == dependency_types.StaticValue.exact(b"one")
    assert last_by_name["A"].export_state.value == "exported"
    assert last_by_name["B"].value == dependency_types.StaticValue.exact(b"two")
    assert last_by_name["B"].export_state.value == expected_b_export
    assert tuple(issue.reason for issue in private.public.issues) == expected_issues


def test_known_function_call_widens_its_body_variable_write_set() -> None:
    private, _budget, _unit = _analyze_private(b'A=old\nf() { A=new; }\nf\nprobe "$A"\n')

    function_call = next(
        command
        for command in private.program.commands
        if command.site.argv[0] == dependency_types.StaticValue.exact(b"f")
        and command.function_id is None
    )
    assert function_call.resolution.kind.value == "function"
    assert private.public.commands[-1].argv[1] == dependency_types.StaticValue.unknown()


def test_known_function_call_widens_its_body_function_write_set() -> None:
    private, _budget, _unit = _analyze_private(
        b"f() { sh() { neutral; }; }\nf\nsh -c 'probe hidden'\nprobe later\n"
    )

    shell = next(
        command
        for command in private.program.commands
        if command.site.argv[0] == dependency_types.StaticValue.exact(b"sh")
        and command.function_id is None
    )
    assert shell.resolution.kind.value == "ambiguous"
    assert all(
        command.site.argv[0] != dependency_types.StaticValue.exact(b"probe")
        or command.site.argv[1] == dependency_types.StaticValue.exact(b"later")
        for command in private.program.commands
    )


def test_known_function_call_widens_transitive_body_write_sets() -> None:
    private, _budget, _unit = _analyze_private(
        b"A=old\ng() { A=new; sh() { neutral; }; }\nf() { g; }\nf\n"
        b"probe \"$A\"\nsh -c 'probe hidden'\n"
    )

    shell = next(
        command
        for command in private.program.commands
        if command.site.argv[0] == dependency_types.StaticValue.exact(b"sh")
        and command.function_id is None
    )
    assert private.public.commands[-2].argv[1] == dependency_types.StaticValue.unknown()
    assert shell.resolution.kind.value == "ambiguous"
    assert private.program.nested_programs == ()


@pytest.mark.parametrize(
    "callee_definition",
    [
        b"g() { A=new; }",
        b"if cond; then g() { A=new; }; fi",
    ],
)
def test_known_function_summary_includes_future_or_conditional_ambiguous_callee(
    callee_definition: bytes,
) -> None:
    private, _budget, _unit = _analyze_private(
        b"A=old\nf() { g; }\n" + callee_definition + b'\nf\nprobe "$A"\n'
    )

    body_call = next(
        command
        for command in private.program.commands
        if command.function_id is not None
        and command.site.argv[0] == dependency_types.StaticValue.exact(b"g")
    )
    assert body_call.resolution.kind.value == "ambiguous"
    assert private.public.commands[-1].argv[1] == dependency_types.StaticValue.unknown()


def test_ambiguous_conditional_function_call_applies_possible_write_summary() -> None:
    private, _budget, _unit = _analyze_private(
        b'A=old\nif cond; then f() { A=new; }; fi\nf\nprobe "$A"\n'
    )

    call = next(
        command
        for command in private.program.commands
        if command.function_id is None
        and command.site.argv[0] == dependency_types.StaticValue.exact(b"f")
    )
    assert call.resolution.kind.value == "ambiguous"
    assert private.public.commands[-1].argv[1] == dependency_types.StaticValue.unknown()


def test_dynamic_ambiguous_body_call_widens_binding_and_function_namespaces() -> None:
    private, _budget, _unit = _analyze_private(
        b'A=old\ng() { A=new; }\nFN=g\nf() { "$FN"; }\nf\ng\nprobe "$A"\n'
    )

    body_call = next(
        command
        for command in private.program.commands
        if command.function_id is not None
        and command.site.argv[0].state is dependency_types.StaticValueState.UNKNOWN
    )
    root_g = next(
        command
        for command in private.program.commands
        if command.function_id is None
        and command.site.argv[0] == dependency_types.StaticValue.exact(b"g")
    )
    assert body_call.resolution.kind.value == "ambiguous"
    assert root_g.resolution.kind.value == "ambiguous"
    assert private.public.commands[-1].argv[1] == dependency_types.StaticValue.unknown()


def test_assignment_only_multi_bind_persists_every_assignment() -> None:
    private, _budget, _unit = _analyze_private(b'A=one B=two\nprobe "$A" "$B"\n')

    assert private.public.commands[-1].argv[1:] == (
        dependency_types.StaticValue.exact(b"one"),
        dependency_types.StaticValue.exact(b"two"),
    )


def test_indented_line_continuation_remains_a_prefix_overlay() -> None:
    private, _budget, _unit = _analyze_private(b'A=base\nA=temp \\\n  probe "$A"\nprobe "$A"\n')

    probes = [
        command
        for command in private.program.commands
        if command.site.argv[0] == dependency_types.StaticValue.exact(b"probe")
    ]
    assert [probe.site.argv[1] for probe in probes] == [
        dependency_types.StaticValue.exact(b"base"),
        dependency_types.StaticValue.exact(b"base"),
    ]
    assert [(binding.name, binding.value) for binding in probes[0].prefix_bindings] == [
        ("A", dependency_types.StaticValue.exact(b"temp"))
    ]
    assert probes[1].prefix_bindings == ()


def test_mixed_assignment_boundaries_partition_persistent_and_prefix_state() -> None:
    private, _budget, _unit = _analyze_private(
        b"A=base\nC=base\nB=base\n"
        b"A=one C=three\nB=two \\\n  "
        b'probe "$A" "$B" "$C"\nprobe "$A" "$B" "$C"\n'
    )

    probes = [
        command
        for command in private.program.commands
        if command.site.argv[0] == dependency_types.StaticValue.exact(b"probe")
    ]
    expected = (
        dependency_types.StaticValue.exact(b"one"),
        dependency_types.StaticValue.exact(b"base"),
        dependency_types.StaticValue.exact(b"three"),
    )
    assert [probe.site.argv[1:] for probe in probes] == [expected, expected]
    assert [(binding.name, binding.value) for binding in probes[0].prefix_bindings] == [
        ("B", dependency_types.StaticValue.exact(b"two"))
    ]
    assert probes[1].prefix_bindings == ()


def test_nested_loop_back_edge_survives_outer_conditional_scope() -> None:
    private, _budget, _unit = _analyze_private(
        b'A=old\nif outer; then while cond; do probe "$A"; A=new; done; fi\nprobe "$A"\n'
    )

    probes = [
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    ]
    assert [probe.argv[1] for probe in probes] == [
        dependency_types.StaticValue.unknown(),
        dependency_types.StaticValue.unknown(),
    ]


def test_nested_branch_joins_before_later_outer_branch_sites() -> None:
    private, _budget, _unit = _analyze_private(
        b'A=base\nif outer; then if inner; then A=one; else A=two; fi; probe "$A"; fi\nprobe "$A"\n'
    )

    probes = [
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    ]
    assert [probe.argv[1] for probe in probes] == [
        dependency_types.StaticValue.unknown(),
        dependency_types.StaticValue.unknown(),
    ]


def test_or_rhs_observes_boolean_short_circuit_join() -> None:
    private, _budget, _unit = _analyze_private(b'A=old\ncond && A=new || probe "$A"\n')

    assert private.public.commands[-1].argv[1] == dependency_types.StaticValue.unknown()


@pytest.mark.parametrize(
    "declaration",
    [
        b"declare -i A=1+2",
        b"typeset -n A=B",
    ],
)
def test_unsupported_declaration_modes_invalidate_assigned_values(
    declaration: bytes,
) -> None:
    private, _budget, _unit = _analyze_private(b"A=old\n" + declaration + b'\nprobe "$A"\n')

    assert private.public.commands[-1].argv[1] == dependency_types.StaticValue.unknown()
    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]


@pytest.mark.parametrize(
    "declaration",
    [
        b"readonly A=old",
        b"declare -r A=old",
    ],
)
def test_persistent_declaration_attributes_block_later_exact_recovery(
    declaration: bytes,
) -> None:
    private, _budget, _unit = _analyze_private(declaration + b'\nA=new\nprobe "$A"\n')

    assert private.public.commands[-1].argv[1] == dependency_types.StaticValue.unknown()
    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]


def test_readonly_persistent_barrier_remains_name_local() -> None:
    private, _budget, _unit = _analyze_private(b'readonly A=old\nB=new\nprobe "$A" "$B"\n')

    assert private.public.commands[-1].argv[1:] == (
        dependency_types.StaticValue.unknown(),
        dependency_types.StaticValue.exact(b"new"),
    )


def test_nameref_declaration_mode_blocks_alias_target_precision() -> None:
    private, _budget, _unit = _analyze_private(b'A=old\ntypeset -n R=A\nR=new\nprobe "$A" "$R"\n')

    assert private.public.commands[-1].argv[1:] == (
        dependency_types.StaticValue.unknown(),
        dependency_types.StaticValue.unknown(),
    )
    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]


def test_nameref_declaration_mode_blocks_future_alias_target_precision() -> None:
    private, _budget, _unit = _analyze_private(b'typeset -n R=A\nA=old\nR=new\nprobe "$A" "$R"\n')

    assert private.public.commands[-1].argv[1:] == (
        dependency_types.StaticValue.unknown(),
        dependency_types.StaticValue.unknown(),
    )
    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]


@pytest.mark.parametrize(
    "child",
    [
        b'if cond; then A=old; R=new; probe "$A" "$R"; fi',
        b'{ A=old; R=new; probe "$A" "$R"; }',
        b'( A=old; R=new; probe "$A" "$R" )',
        b'f() { A=old; R=new; probe "$A" "$R"; }\nf',
    ],
    ids=["conditional", "brace-group", "subshell", "called-function"],
)
def test_nameref_namespace_barrier_is_visible_in_child_state_frames(child: bytes) -> None:
    private, _budget, _unit = _analyze_private(b"typeset -n R=A\n" + child + b"\n")

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1:] == (
        dependency_types.StaticValue.unknown(),
        dependency_types.StaticValue.unknown(),
    )
    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]


def test_nameref_namespace_barrier_is_applied_at_later_function_call() -> None:
    private, _budget, _unit = _analyze_private(
        b'f() { A=old; R=new; probe "$A" "$R"; }\ntypeset -n R=A\nf\n'
    )

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1:] == (
        dependency_types.StaticValue.unknown(),
        dependency_types.StaticValue.unknown(),
    )
    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]


def test_nameref_barrier_call_excludes_superseded_unconditional_definition() -> None:
    private, _budget, _unit = _analyze_private(
        b'f() { A=old; R=oldr; probe_old "$A" "$R"; }\n'
        b"f\n"
        b'f() { A=new; R=newr; probe_new "$A" "$R"; }\n'
        b"typeset -n R=A\n"
        b"f\n"
    )

    probes = {
        command.argv[0].exact_bytes: command.argv[1:]
        for command in private.public.commands
        if command.argv[0].exact_bytes in {b"probe_old", b"probe_new"}
    }
    assert probes == {
        b"probe_old": (
            dependency_types.StaticValue.exact(b"old"),
            dependency_types.StaticValue.exact(b"oldr"),
        ),
        b"probe_new": (
            dependency_types.StaticValue.unknown(),
            dependency_types.StaticValue.unknown(),
        ),
    }


def test_nameref_barrier_call_keeps_conditional_redefinition_candidates() -> None:
    private, _budget, _unit = _analyze_private(
        b'f() { A=old; R=oldr; probe_old "$A" "$R"; }\n'
        b"f\n"
        b'if cond; then f() { A=new; R=newr; probe_new "$A" "$R"; }; fi\n'
        b"typeset -n R=A\n"
        b"f\n"
    )

    probes = {
        command.argv[0].exact_bytes: command.argv[1:]
        for command in private.public.commands
        if command.argv[0].exact_bytes in {b"probe_old", b"probe_new"}
    }
    assert probes == {
        b"probe_old": (
            dependency_types.StaticValue.unknown(),
            dependency_types.StaticValue.unknown(),
        ),
        b"probe_new": (
            dependency_types.StaticValue.unknown(),
            dependency_types.StaticValue.unknown(),
        ),
    }


def test_later_nameref_namespace_barrier_does_not_poison_earlier_function_call() -> None:
    private, _budget, _unit = _analyze_private(
        b'f() { A=old; R=new; probe "$A" "$R"; }\nf\ntypeset -n R=A\n'
    )

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1:] == (
        dependency_types.StaticValue.exact(b"old"),
        dependency_types.StaticValue.exact(b"new"),
    )


def test_nameref_namespace_barrier_reaches_transitive_called_function_frames() -> None:
    private, _budget, _unit = _analyze_private(
        b'g() { A=old; probe "$A"; }\nf() { g; }\ntypeset -n R=A\nf\n'
    )

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1] == dependency_types.StaticValue.unknown()


def test_nameref_namespace_barrier_reaches_dynamic_called_function_frames() -> None:
    private, _budget, _unit = _analyze_private(
        b'f() { A=old; probe "$A"; }\ntypeset -n R=A\nFN=f\n"$FN"\n'
    )

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1] == dependency_types.StaticValue.unknown()


def test_inherited_nameref_barrier_reaches_dynamic_transitive_function_frame() -> None:
    private, _budget, _unit = _analyze_private(
        b'g() { A=old; probe "$A"; }\nFN=g\nf() { "$FN"; }\ntypeset -n R=A\nf\n'
    )

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1] == dependency_types.StaticValue.unknown()


@pytest.mark.parametrize(
    ("body_call", "call_state"),
    [(b"g", b""), (b'"$FN"', b"FN=g; ")],
    ids=["exact-wrapper", "dynamic-wrapper"],
)
def test_outer_wrapper_barrier_resolves_call_site_local_function(
    body_call: bytes,
    call_state: bytes,
) -> None:
    private, _budget, _unit = _analyze_private(
        b"f() { "
        + body_call
        + b'; }\n( g() { A=old; probe "$A"; }; '
        + call_state
        + b"typeset -n R=A; f )\n"
    )

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1] == dependency_types.StaticValue.unknown()


@pytest.mark.parametrize(
    ("body_call", "call_state"),
    [(b"g", b""), (b'"$FN"', b"FN=g; ")],
    ids=["exact-wrapper", "dynamic-wrapper"],
)
def test_outer_wrapper_barrier_does_not_reach_sibling_subshell_function(
    body_call: bytes,
    call_state: bytes,
) -> None:
    private, _budget, _unit = _analyze_private(
        b"f() { "
        + body_call
        + b'; }\n( g() { A=old; probe "$A"; }; g )\n( '
        + call_state
        + b"typeset -n R=A; f )\n"
    )

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1] == dependency_types.StaticValue.exact(b"old")


@pytest.mark.parametrize(
    ("body_call", "call_state"),
    [(b"g", b""), (b'"$FN"', b"FN=g; ")],
    ids=["exact-wrapper", "dynamic-wrapper"],
)
def test_outer_wrapper_barrier_reaches_inherited_root_function(
    body_call: bytes,
    call_state: bytes,
) -> None:
    private, _budget, _unit = _analyze_private(
        b'g() { A=old; probe "$A"; }\nf() { '
        + body_call
        + b"; }\n( "
        + call_state
        + b"typeset -n R=A; f )\n"
    )

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1] == dependency_types.StaticValue.unknown()


@pytest.mark.parametrize(
    ("body_call", "call_state"),
    [(b"g", b""), (b'"$FN"', b"FN=g; ")],
    ids=["exact-wrapper", "dynamic-wrapper"],
)
def test_same_subshell_wrapper_barrier_reaches_local_function(
    body_call: bytes,
    call_state: bytes,
) -> None:
    private, _budget, _unit = _analyze_private(
        b'( g() { A=old; probe "$A"; }; f() { '
        + body_call
        + b"; }; "
        + call_state
        + b"typeset -n R=A; f )\n"
    )

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1] == dependency_types.StaticValue.unknown()


@pytest.mark.parametrize(
    ("barrier_call", "wrapper_definition", "call_state"),
    [
        (b"g", b"", b""),
        (b'"$FN"', b"", b"FN=g; "),
        (b"f", b"f() { g; }\n", b""),
        (b"f", b'f() { "$FN"; }\n', b"FN=g; "),
    ],
    ids=["direct-exact", "direct-dynamic", "transitive-exact", "transitive-dynamic"],
)
def test_function_origin_barrier_resolves_invocation_local_function(
    barrier_call: bytes,
    wrapper_definition: bytes,
    call_state: bytes,
) -> None:
    private, _budget, _unit = _analyze_private(
        b"h() { typeset -n R=A; "
        + barrier_call
        + b"; }\n"
        + wrapper_definition
        + b'( g() { A=old; R=new; probe "$A" "$R"; }; '
        + call_state
        + b"h )\n"
    )

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1:] == (
        dependency_types.StaticValue.unknown(),
        dependency_types.StaticValue.unknown(),
    )


@pytest.mark.parametrize(
    ("barrier_call", "wrapper_definition", "call_state"),
    [
        (b"g", b"", b""),
        (b'"$FN"', b"", b"FN=g; "),
        (b"f", b"f() { g; }\n", b""),
        (b"f", b'f() { "$FN"; }\n', b"FN=g; "),
    ],
    ids=["direct-exact", "direct-dynamic", "transitive-exact", "transitive-dynamic"],
)
def test_function_origin_barrier_does_not_reach_sibling_subshell_function(
    barrier_call: bytes,
    wrapper_definition: bytes,
    call_state: bytes,
) -> None:
    private, _budget, _unit = _analyze_private(
        b"h() { typeset -n R=A; "
        + barrier_call
        + b"; }\n"
        + wrapper_definition
        + b'( g() { A=old; R=new; probe "$A" "$R"; }; g )\n( '
        + call_state
        + b"h )\n"
    )

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1:] == (
        dependency_types.StaticValue.exact(b"old"),
        dependency_types.StaticValue.exact(b"new"),
    )


@pytest.mark.parametrize(
    ("barrier_call", "call_state"),
    [(b"g", b""), (b'"$FN"', b"FN=g; ")],
    ids=["exact", "dynamic"],
)
@pytest.mark.parametrize(
    "same_subshell",
    [False, True],
    ids=["inherited-root", "same-subshell"],
)
def test_function_origin_barrier_reaches_visible_function(
    barrier_call: bytes,
    call_state: bytes,
    same_subshell: bool,
) -> None:
    function_source = (
        b'g() { A=old; R=new; probe "$A" "$R"; }; h() { typeset -n R=A; ' + barrier_call + b"; }; "
    )
    raw = (
        b"( " + function_source + call_state + b"h )\n"
        if same_subshell
        else function_source + b"( " + call_state + b"h )\n"
    )
    private, _budget, _unit = _analyze_private(raw)

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1:] == (
        dependency_types.StaticValue.unknown(),
        dependency_types.StaticValue.unknown(),
    )


def test_isolated_dynamic_nameref_barrier_does_not_poison_future_function() -> None:
    private, _budget, _unit = _analyze_private(
        b'( typeset -n R=A; "$FN" )\nf() { A=old; R=new; probe "$A" "$R"; }\nf\n'
    )

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1:] == (
        dependency_types.StaticValue.exact(b"old"),
        dependency_types.StaticValue.exact(b"new"),
    )


@pytest.mark.parametrize(
    "call",
    [b"hidden", b'"$FN"'],
    ids=["exact-call", "dynamic-call"],
)
def test_nameref_barrier_ignores_unreachable_isolated_function(call: bytes) -> None:
    private, _budget, _unit = _analyze_private(
        b'( hidden() { A=old; probe "$A"; }; hidden )\ntypeset -n R=A\n' + call + b"\n"
    )

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1] == dependency_types.StaticValue.exact(b"old")


def test_isolated_nameref_barrier_marks_only_inner_shadow_definition() -> None:
    private, _budget, _unit = _analyze_private(
        b'f() { A=root; probe_root "$A"; }\n'
        b"f\n"
        b'( typeset -n R=A; f() { A=inner; probe_inner "$A"; }; f )\n'
        b"f\n"
    )

    probes = {
        command.argv[0].exact_bytes: command.argv[1]
        for command in private.public.commands
        if command.argv[0].exact_bytes in {b"probe_root", b"probe_inner"}
    }
    assert probes == {
        b"probe_root": dependency_types.StaticValue.exact(b"root"),
        b"probe_inner": dependency_types.StaticValue.unknown(),
    }


def test_subshell_nameref_namespace_barrier_does_not_escape_to_parent() -> None:
    private, _budget, _unit = _analyze_private(
        b'A=outer\n( typeset -n R=A; A=inner; R=changed; probe "$A" "$R" )\n'
        b'A=recovered\nprobe "$A"\n'
    )

    probes = [
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    ]
    assert [probe.argv[1:] for probe in probes] == [
        (
            dependency_types.StaticValue.unknown(),
            dependency_types.StaticValue.unknown(),
        ),
        (dependency_types.StaticValue.exact(b"recovered"),),
    ]


def test_nonpersistent_declaration_limitation_allows_definite_recovery() -> None:
    private, _budget, _unit = _analyze_private(b'declare -p A=old\nA=new\nprobe "$A"\n')

    assert private.public.commands[-1].argv[1] == dependency_types.StaticValue.exact(b"new")
    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]


@pytest.mark.parametrize("option", [b"-f", b"-p", b"-x"])
def test_unsupported_export_option_assignment_is_invalidated(option: bytes) -> None:
    private, _budget, _unit = _analyze_private(b"A=old\nexport " + option + b' A=new\nprobe "$A"\n')

    assert private.public.commands[-1].argv[1] == dependency_types.StaticValue.unknown()
    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]


def test_assignment_from_unset_expansion_does_not_propagate_unbound() -> None:
    private, _budget, _unit = _analyze_private(b'unset B\nA="$B"\nprobe "$A"\n')

    assert private.public.commands[-1].argv[1] == dependency_types.StaticValue.unknown()


def test_loop_back_edge_includes_eval_imported_function_effects() -> None:
    private, _budget, _unit = _analyze_private(
        b"A=old\neval 'foo() { A=new; }'\nwhile cond; do probe \"$A\"; foo; done\n"
    )

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1] == dependency_types.StaticValue.unknown()
    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]


def test_recursive_function_write_summary_terminates_conservatively() -> None:
    private, _budget, _unit = _analyze_private(b'A=old\nf() { f; A=new; }\nf\nprobe "$A"\n')

    assert private.public.commands[-1].argv[1] == dependency_types.StaticValue.unknown()


def test_declaration_named_function_still_contributes_transitive_write_effects() -> None:
    private, _budget, _unit = _analyze_private(
        b'A=old\ng() { A=new; }\nfunction export { g; }\nf() { export; }\nf\nprobe "$A"\n'
    )

    assert private.public.commands[-1].argv[1] == dependency_types.StaticValue.unknown()


def test_eval_imported_interpreter_function_never_parses_as_external() -> None:
    private, _budget, _unit = _analyze_private(
        b"eval 'sh() { neutral; }'\nsh -c 'probe hidden'\nprobe later\n"
    )

    shell = next(
        command
        for command in private.program.commands
        if command.site.argv[0] == dependency_types.StaticValue.exact(b"sh")
        and command.function_id is None
    )
    assert shell.resolution.kind.value in {"function", "ambiguous"}
    assert all(
        command.site.argv[0] != dependency_types.StaticValue.exact(b"probe")
        or command.site.argv[1] == dependency_types.StaticValue.exact(b"later")
        for command in private.program.commands
    )


def test_eval_imported_generic_function_applies_conservative_parent_barrier() -> None:
    private, _budget, _unit = _analyze_private(
        b"A=old\neval 'foo() { A=new; }'\nfoo\nprobe \"$A\"\n"
    )

    call = next(
        command
        for command in private.program.commands
        if command.site.argv[0] == dependency_types.StaticValue.exact(b"foo")
        and command.function_id is None
    )
    assert call.resolution.kind.value in {"function", "ambiguous"}
    assert private.public.commands[-1].argv[1] == dependency_types.StaticValue.unknown()
    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]
    assert private.public.issues[0].span == call.site.span
    assert private.public.work_items[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL


def test_known_function_call_does_not_widen_isolated_body_writes() -> None:
    private, _budget, _unit = _analyze_private(b'A=old\nf() { (A=new); }\nf\nprobe "$A"\n')

    assert private.public.commands[-1].argv[1] == dependency_types.StaticValue.exact(b"old")


@pytest.mark.parametrize(
    "control",
    [
        b"f() { A=new; }\nf\n",
        b"for item in one; do A=new; done\n",
    ],
)
def test_value_only_write_summaries_preserve_export_reachability(
    control: bytes,
) -> None:
    private, _budget, _unit = _analyze_private(
        b"export A=old\n" + control + b"sh -c 'probe \"$A\"'\n"
    )

    nested = private.program.nested_programs[0]
    inherited = {binding.name: binding for binding in nested.program.initial_bindings}
    assert inherited["A"].value == dependency_types.StaticValue.unknown()
    assert inherited["A"].export_state.value == "exported"


def test_loop_back_edge_includes_known_function_call_effects() -> None:
    private, _budget, _unit = _analyze_private(
        b'A=old\nf() { A=new; }\nfor item in one; do probe "$A"; f; done\n'
    )

    assert private.public.commands[-2].argv[1] == dependency_types.StaticValue.unknown()


@pytest.mark.parametrize("loop_keyword", [b"while", b"until", b"for"])
def test_loop_back_edge_widens_body_writes_before_every_body_site(
    loop_keyword: bytes,
) -> None:
    loop = (
        b'for item in one; do probe "$A"; A=new; done'
        if loop_keyword == b"for"
        else loop_keyword + b' cond; do probe "$A"; A=new; done'
    )
    private, _budget, _unit = _analyze_private(b"A=old\n" + loop + b'\nprobe "$A"\n')

    probes = [
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    ]
    assert [probe.argv[1] for probe in probes] == [
        dependency_types.StaticValue.unknown(),
        dependency_types.StaticValue.unknown(),
    ]


@pytest.mark.parametrize(
    ("loop", "expected"),
    [
        (
            b'for item in one; do A=new; probe "$A"; done',
            dependency_types.StaticValue.exact(b"new"),
        ),
        (
            b'while A=new; cond; do probe "$A"; done',
            dependency_types.StaticValue.exact(b"new"),
        ),
        (
            b'while cond; do (A=new); probe "$A"; done',
            dependency_types.StaticValue.exact(b"old"),
        ),
    ],
)
def test_loop_back_edge_preserves_dominating_and_isolated_values(
    loop: bytes,
    expected: dependency_types.StaticValue,
) -> None:
    private, _budget, _unit = _analyze_private(b"A=old\n" + loop + b"\n")

    assert private.public.commands[-1].argv[1] == expected


@pytest.mark.timeout(5)
def test_isolated_first_loop_event_never_cycles_or_leaks_into_loop_state() -> None:
    private, _budget, _unit = _analyze_private(
        b'A=old\ni=0\nwhile (A=new; test "$i" = 0); do probe "$A"; i=1; done\n'
    )

    probe = next(
        command
        for command in private.public.commands
        if command.argv[0] == dependency_types.StaticValue.exact(b"probe")
    )
    assert probe.argv[1] == dependency_types.StaticValue.exact(b"old")


@pytest.mark.parametrize("interpreter", [b"bash", b"sh", b"dash", b"eval"])
def test_ambiguous_nested_interpreter_is_one_localized_partial_limitation(
    interpreter: bytes,
) -> None:
    invocation = (
        b"eval 'probe hidden'" if interpreter == b"eval" else interpreter + b" -c 'probe hidden'"
    )
    private, _budget, _unit = _analyze_private(
        b"if cond; then " + interpreter + b"() { neutral; }; fi\n" + invocation + b"\nprobe later\n"
    )

    command = next(
        candidate
        for candidate in private.program.commands
        if candidate.site.argv[0] == dependency_types.StaticValue.exact(interpreter)
        and candidate.function_id is None
    )
    assert command.resolution.kind.value == "ambiguous"
    assert private.program.nested_programs == ()
    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]
    assert private.public.issues[0].span == command.site.span
    assert private.public.work_items[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL
    assert _argv_bytes(private.public.commands[-1]) == (b"probe", b"later")


def test_nested_commands_publish_once_in_execution_preorder() -> None:
    private, _budget, _unit = _analyze_private(b"P='probe child'\neval \"$P\"\nprobe later\n")

    expected = [(b"eval", b"probe child"), (b"probe", b"child"), (b"probe", b"later")]
    assert [_argv_bytes(command) for command in private.public.commands] == expected
    assert [_argv_bytes(command.site) for command in private.program.execution_commands] == expected
    assert private.program.commands is private.program.execution_commands
    nested_id = private.program.nested_programs[0].unit.unit_id
    assert sum(command.unit_id == nested_id for command in private.public.commands) == 1


def test_one_over_ir_publication_discards_unreserved_child_slice_coherently() -> None:
    raw = b"P='probe child-one; probe child-two'\neval \"$P\"\nprobe later\n"
    baseline_budget = DependencyWorkBudget()
    unit = _extract("scripts/transaction.sh", raw, budget=baseline_budget).units[0]
    baseline = shell_frontend._analyze_shell_unit(unit, budget=baseline_budget)
    assert baseline.public.work_items[0].outcome is dependency_types.ShellWorkOutcome.COMPLETED
    required_ir = baseline_budget.used(dependency_types.DependencyWorkResource.RETAINED_SHELL_IR)

    limited_budget = DependencyWorkBudget()
    limited_file = limited_budget.for_file(unit.origin_span.path)
    limited_file.register_shell_file_size(len(raw))
    assert (
        limited_file.charge_retained_shell_ir(
            unit,
            dependency_types.MAX_DEPENDENCY_RETAINED_SHELL_IR - required_ir + 1,
        )
        is None
    )
    limited = shell_frontend._analyze_shell_unit(unit, budget=limited_budget)

    public_keys = [
        (command.unit_id, command.span.start_byte, command.span.end_byte, command.argv)
        for command in limited.public.commands
    ]
    private_keys = [
        (
            command.site.unit_id,
            command.site.span.start_byte,
            command.site.span.end_byte,
            command.site.argv,
        )
        for command in limited.program.commands
    ]
    execution_keys = [
        (
            command.site.unit_id,
            command.site.span.start_byte,
            command.site.span.end_byte,
            command.site.argv,
        )
        for command in limited.program.execution_commands
    ]
    assert public_keys == private_keys == execution_keys
    assert len(public_keys) == len(set(public_keys))
    assert all(nested.program.commands == () for nested in limited.program.nested_programs)
    assert all(
        command.provenance is not dependency_types.SiteProvenance.NESTED_LITERAL
        for command in limited.public.commands
    )
    assert dependency_types.ShellIssueReason.RESOURCE_LIMIT in {
        issue.reason for issue in limited.public.issues
    }


def test_every_nested_ir_cutoff_keeps_all_command_views_transactional() -> None:
    raw = b"P='probe child-one; probe child-two'\neval \"$P\"\nprobe later\n"
    baseline_budget = DependencyWorkBudget()
    unit = _extract("scripts/cutoffs.sh", raw, budget=baseline_budget).units[0]
    shell_frontend._analyze_shell_unit(unit, budget=baseline_budget)
    required_ir = baseline_budget.used(dependency_types.DependencyWorkResource.RETAINED_SHELL_IR)

    for available_ir in range(required_ir + 1):
        budget = DependencyWorkBudget()
        file_budget = budget.for_file(unit.origin_span.path)
        file_budget.register_shell_file_size(len(raw))
        assert (
            file_budget.charge_retained_shell_ir(
                unit,
                dependency_types.MAX_DEPENDENCY_RETAINED_SHELL_IR - available_ir,
            )
            is None
        )
        result = shell_frontend._analyze_shell_unit(unit, budget=budget)
        public_keys = [
            (command.unit_id, command.span.start_byte, command.span.end_byte, command.argv)
            for command in result.public.commands
        ]
        private_keys = [
            (
                command.site.unit_id,
                command.site.span.start_byte,
                command.site.span.end_byte,
                command.site.argv,
            )
            for command in result.program.commands
        ]
        execution_keys = [
            (
                command.site.unit_id,
                command.site.span.start_byte,
                command.site.span.end_byte,
                command.site.argv,
            )
            for command in result.program.execution_commands
        ]
        assert public_keys == private_keys == execution_keys
        assert len(private_keys) == len(set(private_keys))
        assert all(command.program_id for command in result.program.commands)
        frame_ids = {frame.frame_id for frame in result.program.state_frames}
        assert all(command.state_frame_id in frame_ids for command in result.program.commands)
        assert all(update.frame_id in frame_ids for update in result.program.state_updates)
        assert all(
            (
                command.site.unit_id,
                command.site.span.start_byte,
                command.site.span.end_byte,
                command.site.argv,
            )
            in private_keys
            for nested in result.program.nested_programs
            for command in nested.program.commands
        )


def test_ir_discard_prunes_orphan_nested_records_and_work_items() -> None:
    raw = b"sh -c \"sh -c 'probe deep'\"\nprobe tail\n"
    baseline_budget = DependencyWorkBudget()
    unit = _extract("scripts/orphans.sh", raw, budget=baseline_budget).units[0]
    baseline_file = baseline_budget.for_file(unit.origin_span.path)
    baseline_file.register_shell_file_size(len(raw))
    for _ in range(dependency_types.MAX_DEPENDENCY_SHELL_PARSER_CALLS_PER_FILE - 2):
        assert baseline_file.reserve_shell_parse(0) is None
    shell_frontend._analyze_shell_unit(unit, budget=baseline_budget)
    required_ir = baseline_budget.used(dependency_types.DependencyWorkResource.RETAINED_SHELL_IR)

    limited_budget = DependencyWorkBudget()
    limited_file = limited_budget.for_file(unit.origin_span.path)
    limited_file.register_shell_file_size(len(raw))
    for _ in range(dependency_types.MAX_DEPENDENCY_SHELL_PARSER_CALLS_PER_FILE - 2):
        assert limited_file.reserve_shell_parse(0) is None
    assert (
        limited_file.charge_retained_shell_ir(
            unit,
            dependency_types.MAX_DEPENDENCY_RETAINED_SHELL_IR - required_ir + 1,
        )
        is None
    )
    result = shell_frontend._analyze_shell_unit(unit, budget=limited_budget)

    known_program_ids = {
        result.program.program_id,
        *(nested.program.program_id for nested in result.program.nested_programs),
    }
    assert all(
        nested.parent_program_id in known_program_ids for nested in result.program.nested_programs
    )
    nested_unit_ids = {nested.unit.unit_id for nested in result.program.nested_programs}
    assert {
        item.unit_id
        for item in result.public.work_items
        if item.kind is dependency_types.ShellUnitKind.NESTED_LITERAL
    } <= nested_unit_ids
