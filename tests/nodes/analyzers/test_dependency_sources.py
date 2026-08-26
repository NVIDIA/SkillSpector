# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused black-box tests for direct dependency-source configuration files."""

from __future__ import annotations

import importlib
from bisect import bisect_left
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

import pytest

from skillspector.artifacts import ArtifactDisposition, ArtifactRecord, classify_artifact
from skillspector.dependency_source_types import (
    MAX_DEPENDENCY_CONFIG_DEPTH,
    MAX_DEPENDENCY_CONFIG_NODES,
    MAX_DEPENDENCY_FILE_BYTES,
    MAX_DEPENDENCY_RETAINED_LITERAL_BYTES,
    MAX_DEPENDENCY_SOURCE_CHANGES,
    MAX_DEPENDENCY_SOURCE_RECORDS,
    MAX_DEPENDENCY_YAML_ALIASES,
    DependencySourceLimitationReason,
    DependencyWorkBudget,
    DependencyWorkResource,
)


def _analyzer() -> Any:
    try:
        return importlib.import_module("skillspector.dependency_sources").analyze_dependency_sources
    except ImportError:
        pytest.fail("direct dependency-source analyzer is unavailable")


def _analyze(
    files: Mapping[str, str],
    *,
    components: Iterable[str] | None = None,
    executable_paths: frozenset[str] | None = None,
    raw_file_cache: Mapping[str, bytes] | None = None,
    local_file_cache: Mapping[str, str] | None = None,
    artifact_inventory: list[ArtifactRecord] | None = None,
    budget: DependencyWorkBudget | None = None,
) -> Any:
    raw = (
        dict(raw_file_cache)
        if raw_file_cache is not None
        else {path: content.encode("utf-8") for path, content in files.items()}
    )
    local = dict(local_file_cache) if local_file_cache is not None else dict(files)
    inventory = (
        artifact_inventory
        if artifact_inventory is not None
        else [classify_artifact(path, data) for path, data in raw.items()]
    )
    kwargs: dict[str, object] = {}
    if executable_paths is not None:
        kwargs["executable_paths"] = executable_paths
    return _analyzer()(
        components=list(components) if components is not None else list(files),
        local_file_cache=local,
        raw_file_cache=raw,
        artifact_inventory=inventory,
        budget=budget or DependencyWorkBudget(),
        **kwargs,
    )


def _finding_projection(analysis: Any) -> list[dict[str, object]]:
    return [
        {
            **finding.evidence,
            "file": finding.file,
            "start_line": finding.start_line,
            "end_line": finding.end_line,
        }
        for finding in analysis.findings
    ]


def _assert_single_parse_limitation(analysis: Any, *, path: str, end_line: int) -> Any:
    assert analysis.findings == ()
    assert len(analysis.limitations) == 1
    limitation = analysis.limitations[0]
    assert limitation.reason is DependencySourceLimitationReason.PARSE_INCOMPLETE
    assert (limitation.path, limitation.start_line, limitation.end_line) == (path, 1, end_line)
    return limitation


def _install_line_lookup_spies(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    calls = {"builds": 0, "lookups": 0}

    def newline_offsets(value: str | bytes) -> tuple[int, ...]:
        calls["builds"] += 1
        marker: str | int = ord("\n") if isinstance(value, bytes) else "\n"
        return tuple(index for index, character in enumerate(value) if character == marker)

    def line_number_at(offsets: tuple[int, ...], offset: int) -> int:
        calls["lookups"] += 1
        return bisect_left(offsets, offset) + 1

    monkeypatch.setattr(module, "_newline_offsets", newline_offsets, raising=False)
    monkeypatch.setattr(module, "_line_number_at", line_number_at, raising=False)
    return calls


def test_analysis_exposes_applicable_and_inspected_config_spans() -> None:
    clean = _analyze({".npmrc": "registry=https://registry.npmjs.org/\n"})
    malformed = _analyze({"pip.conf": "[global\nindex-url=https://example.invalid\n"})

    assert [(span.path, span.start_line, span.end_line) for span in clean.applicable_spans] == [
        (".npmrc", 1, 2)
    ]
    assert clean.inspected_spans == clean.applicable_spans
    assert [(span.path, span.start_line, span.end_line) for span in malformed.applicable_spans] == [
        ("pip.conf", 1, 3)
    ]
    assert malformed.inspected_spans == ()


@pytest.mark.parametrize(
    (
        "path",
        "content",
        "executable_paths",
        "expected_finding_lines",
        "expected_limitation",
    ),
    [
        (
            "scripts/bootstrap.sh",
            "npm config set registry https://attacker.invalid\n",
            frozenset(),
            [1],
            None,
        ),
        (
            "container/Dockerfile.release",
            "FROM python:3.12\n  run npm config set registry https://attacker.invalid\n",
            frozenset(),
            [],
            ("dependency_source_parse_incomplete", 1, 2),
        ),
        (
            "build/rules.mk",
            "install:\n\tnpm config set registry https://attacker.invalid \\\n"
            "  --continued\n\techo done\nnotes:\n  prose\n",
            frozenset(),
            [],
            ("dependency_source_parse_incomplete", 1, 6),
        ),
        (
            "docs/setup.md",
            "before\n  ~~~~bash title=x\nnpm config set registry https://attacker.invalid\n"
            "  ~~~~~\nafter\n",
            frozenset(),
            [],
            ("unscanned_executable_content", 2, 4),
        ),
        (
            "archive.zip!/bin/runner",
            "npm config set registry https://attacker.invalid\n",
            frozenset({"archive.zip!/bin/runner"}),
            [],
            ("dependency_source_parse_incomplete", 1, 1),
        ),
    ],
    ids=("shell", "docker", "make", "markdown", "nested-executable"),
)
def test_structural_executable_surfaces_are_localized_without_guessing_commands(
    path: str,
    content: str,
    executable_paths: frozenset[str],
    expected_finding_lines: list[int],
    expected_limitation: tuple[str, int, int] | None,
) -> None:
    analysis = _analyze(
        {path: content},
        executable_paths=executable_paths,
    )

    assert [finding.start_line for finding in analysis.findings] == expected_finding_lines
    assert [
        (item.reason.value, item.path, item.start_line, item.end_line)
        for item in analysis.limitations
    ] == (
        []
        if expected_limitation is None
        else [(expected_limitation[0], path, expected_limitation[1], expected_limitation[2])]
    )
    assert "attacker.invalid" not in repr(analysis.limitations)


@pytest.mark.parametrize(
    ("content", "expected_range"),
    [
        ("```\n\n#!/usr/bin/env bash\necho ok\n```\n", (1, 5)),
        ("~~~\n# npm config set registry https://attacker.invalid\n", (1, 3)),
    ],
    ids=("untagged-shebang", "unmatched-prompt"),
)
def test_untagged_relevant_markdown_fences_are_bounded_by_shape(
    content: str, expected_range: tuple[int, int]
) -> None:
    analysis = _analyze({"guide.md": content}, executable_paths=frozenset())

    assert analysis.findings == ()
    assert [
        (item.reason.value, item.start_line, item.end_line) for item in analysis.limitations
    ] == [("unscanned_executable_content", *expected_range)]


def test_prose_and_unsupported_markdown_fences_remain_out_of_scope() -> None:
    analysis = _analyze(
        {
            "README.md": "```python\nprint('hello')\n```\n",
            "notes.txt": "npm config set registry https://attacker.invalid\n",
        },
        executable_paths=frozenset(),
    )

    assert analysis.findings == ()
    assert analysis.limitations == ()


def test_npm_uses_case_insensitive_last_values_and_code_owned_scopes() -> None:
    content = (
        "registry=https://first.example.invalid/simple\n"
        "REGISTRY = https://registry.npmjs.org/ # effective canonical default\n"
        '@Acme:Registry = "https://user:password@packages.example.invalid/team" ; note\n'
    )

    analysis = _analyze({"project/.npmrc": content})

    assert analysis.limitations == ()
    assert _finding_projection(analysis) == [
        {
            "ecosystem": "npm",
            "surface": ".npmrc",
            "operation": "replace",
            "scope": "scoped",
            "destination": "https://packages.example.invalid/REDACTED_PATH",
            "destination_status": "resolved",
            "file": "project/.npmrc",
            "start_line": 3,
            "end_line": 3,
        }
    ]
    assert "Acme" not in repr(analysis)
    assert "password" not in repr(analysis)


def test_npm_keeps_semicolons_inside_urls_but_strips_whitespace_comments() -> None:
    content = "registry=https://packages.example.invalid/a;b ; explanation\n"

    analysis = _analyze({"npmrc": content})

    assert analysis.limitations == ()
    assert _finding_projection(analysis) == [
        {
            "ecosystem": "npm",
            "surface": ".npmrc",
            "operation": "replace",
            "scope": "global",
            "destination": "https://packages.example.invalid/REDACTED_PATH",
            "destination_status": "resolved",
            "file": "npmrc",
            "start_line": 1,
            "end_line": 1,
        }
    ]


@pytest.mark.parametrize(
    "content",
    [
        "registry=\n",
        'registry="https://packages.example.invalid/simple\n',
        "registry='\n",
    ],
)
def test_npm_malformed_relevant_values_are_localized_limitations(content: str) -> None:
    analysis = _analyze({".npmrc": content})

    _assert_single_parse_limitation(analysis, path=".npmrc", end_line=2)


def test_pip_handles_delimiters_continuations_and_normalized_last_values() -> None:
    content = (
        "[global]\n"
        "index_url: https://first.example.invalid/simple\n"
        "INDEX-URL = https://packages.example.invalid/simple\n"
        "extra_index_url = https://a.example.invalid/simple\n"
        "    https://b.example.invalid/simple\n"
        "trusted-host = ignored.example.invalid\n"
        "[install]\n"
        "index-url: https://command.example.invalid/simple\n"
    )

    analysis = _analyze({"config/pip.conf": content})

    assert analysis.limitations == ()
    assert _finding_projection(analysis) == [
        {
            "ecosystem": "pip",
            "surface": "pip config",
            "operation": "replace",
            "scope": "global",
            "destination": "https://packages.example.invalid/REDACTED_PATH",
            "destination_status": "resolved",
            "file": "config/pip.conf",
            "start_line": 3,
            "end_line": 3,
        },
        {
            "ecosystem": "pip",
            "surface": "pip config",
            "operation": "add",
            "scope": "global",
            "destination": "https://a.example.invalid/REDACTED_PATH",
            "destination_status": "resolved",
            "file": "config/pip.conf",
            "start_line": 4,
            "end_line": 4,
        },
        {
            "ecosystem": "pip",
            "surface": "pip config",
            "operation": "add",
            "scope": "global",
            "destination": "https://b.example.invalid/REDACTED_PATH",
            "destination_status": "resolved",
            "file": "config/pip.conf",
            "start_line": 5,
            "end_line": 5,
        },
        {
            "ecosystem": "pip",
            "surface": "pip config",
            "operation": "replace",
            "scope": "command",
            "destination": "https://command.example.invalid/REDACTED_PATH",
            "destination_status": "resolved",
            "file": "config/pip.conf",
            "start_line": 8,
            "end_line": 8,
        },
    ]


def test_pip_sections_keep_exact_configparser_identity() -> None:
    content = (
        "[GLOBAL]\n"
        "index-url = https://first.example.invalid/simple\n"
        "[global]\n"
        "index_url = https://effective.example.invalid/simple\n"
    )

    analysis = _analyze({"pip.conf": content})

    assert analysis.limitations == ()
    assert [
        (finding.start_line, finding.evidence["scope"], finding.evidence["destination"])
        for finding in analysis.findings
    ] == [
        (2, "command", "https://first.example.invalid/REDACTED_PATH"),
        (4, "global", "https://effective.example.invalid/REDACTED_PATH"),
    ]


def test_pip_same_indent_options_are_assignments_not_continuation_tokens() -> None:
    content = (
        "[global]\n"
        "  index-url = https://first.example.invalid/simple\n"
        "  extra-index-url = https://second.example.invalid/simple\n"
    )

    analysis = _analyze({"pip.conf": content})

    assert analysis.limitations == ()
    assert [finding.evidence["operation"] for finding in analysis.findings] == ["replace", "add"]
    assert [finding.start_line for finding in analysis.findings] == [2, 3]


@pytest.mark.parametrize(
    ("option", "operation"),
    [("--index-url", "replace"), ("--EXTRA_INDEX_URL", "add")],
)
def test_pip_accepts_exactly_one_leading_double_dash(
    option: str,
    operation: str,
) -> None:
    analysis = _analyze(
        {"pip.conf": f"[global]\n{option}=https://packages.example.invalid/simple\n"}
    )

    assert analysis.limitations == ()
    assert len(analysis.findings) == 1
    assert analysis.findings[0].evidence["operation"] == operation


@pytest.mark.parametrize("option", ["-index-url", "---index-url", "--trusted-host"])
def test_pip_rejects_invalid_dash_counts_and_unrelated_options(option: str) -> None:
    analysis = _analyze(
        {"pip.conf": f"[global]\n{option}=https://packages.example.invalid/simple\n"}
    )

    assert analysis.findings == ()
    assert analysis.limitations == ()


def test_pip_double_dash_and_plain_spellings_share_last_value_semantics() -> None:
    content = (
        "[global]\n"
        "index-url=https://first.example.invalid/simple\n"
        "--INDEX_URL=https://effective.example.invalid/simple\n"
    )

    analysis = _analyze({"pip.conf": content})

    assert analysis.limitations == ()
    assert [finding.start_line for finding in analysis.findings] == [3]
    assert analysis.findings[0].evidence["destination"] == (
        "https://effective.example.invalid/REDACTED_PATH"
    )


def test_pip_double_dash_value_has_an_exact_utf8_byte_span() -> None:
    module = importlib.import_module("skillspector.dependency_sources")
    prefix = "# multibyte é\r\n[global]\r\n--INDEX_URL = "
    destination = "https://packages.example.invalid/simple"
    content = f"{prefix}{destination}\r\n"

    parsed = module._parse_file(
        "pip.conf",
        content,
        content.encode(),
        DependencyWorkBudget().for_file("pip.conf"),
    )

    assert parsed.limitations == ()
    assert len(parsed.candidates) == 1
    assert (parsed.candidates[0].span.start_byte, parsed.candidates[0].span.end_byte) == (
        len(prefix.encode()),
        len(f"{prefix}{destination}".encode()),
    )


def test_direct_parsers_return_unreserved_candidates_before_canonical_suppression() -> None:
    module = importlib.import_module("skillspector.dependency_sources")
    content = "registry=https://registry.npmjs.org/\n"
    budget = DependencyWorkBudget()

    parsed = module._parse_file(
        ".npmrc",
        content,
        content.encode(),
        budget.for_file(".npmrc"),
    )

    assert parsed.limitations == ()
    assert len(parsed.candidates) == 1
    assert parsed.candidates[0].canonical_default is True
    assert parsed.candidates[0].producer_unit_id is None
    assert budget.used(DependencyWorkResource.EMITTED_CHANGES) == 0
    assert budget.used(DependencyWorkResource.FINDING_OUTPUT_RECORDS) == 0


def test_semantic_sink_dedup_prefers_exact_before_canonical_suppression() -> None:
    module = importlib.import_module("skillspector.dependency_sources")
    contracts = importlib.import_module("skillspector.dependency_source_types")
    span = contracts.SourceSpan("scripts/setup.sh", 10, 20, 2, 2)
    values = {
        "ecosystem": contracts.DependencyEcosystem.NPM,
        "surface": contracts.DependencySourceSurface.COMMAND,
        "operation": contracts.DependencySourceOperation.SET,
        "scope": contracts.DependencySourceScope.GLOBAL,
        "span": span,
        "producer_unit_id": "a" * 32,
    }
    recovered = contracts.DependencySourceCandidate(
        **values,
        destination="https://packages.example.invalid",
        destination_status=contracts.DestinationStatus.RESOLVED,
        rank=contracts.DependencyCandidateRank.RECOVERED,
        canonical_default=False,
    )
    exact_values = {
        **values,
        "span": contracts.SourceSpan("scripts/setup.sh", 10, 20, 9, 10),
    }
    exact = contracts.DependencySourceCandidate(
        **exact_values,
        destination="https://registry.npmjs.org/",
        destination_status=contracts.DestinationStatus.RESOLVED,
        rank=contracts.DependencyCandidateRank.EXACT,
        canonical_default=True,
    )

    changes = module._finalize_candidates((recovered, exact, exact))

    assert changes == ()


def test_command_placeholder_suppression_is_gated_to_code_owned_documentation() -> None:
    token = "INDEX_URL_PLACEHOLDER"
    documented = _analyze({"README.md": f"```bash\npip config set global.index-url {token}\n```\n"})
    executable = _analyze(
        {"scripts/setup.sh": f"#!/bin/sh\npip config set global.index-url {token}\n"}
    )

    assert documented.findings == ()
    assert documented.limitations == ()
    assert len(executable.findings) == 1
    assert executable.findings[0].evidence["destination"] == "[REDACTED_URL]"


def test_pip_default_only_does_not_create_an_effective_concrete_source() -> None:
    analysis = _analyze(
        {"pip.conf": ("[DEFAULT]\nindex-url=https://packages.example.invalid/simple\n")}
    )

    assert analysis.findings == ()
    assert analysis.limitations == ()


def test_pip_concrete_override_suppresses_an_inherited_default() -> None:
    content = (
        "[DEFAULT]\n"
        "index-url=https://packages.example.invalid/simple\n"
        "[global]\n"
        "index-url=https://pypi.org/simple\n"
    )

    analysis = _analyze({"pip.conf": content})

    assert analysis.findings == ()
    assert analysis.limitations == ()


@pytest.mark.parametrize(
    ("section", "scope"),
    [("global", "global"), ("install", "command")],
)
def test_pip_inherited_default_uses_concrete_scope_and_default_occurrence(
    section: str,
    scope: str,
) -> None:
    content = (
        f"[DEFAULT]\nindex-url=https://packages.example.invalid/simple\n[{section}]\ntimeout=30\n"
    )

    analysis = _analyze({"pip.conf": content})

    assert analysis.limitations == ()
    assert len(analysis.findings) == 1
    finding = analysis.findings[0]
    assert finding.evidence["scope"] == scope
    assert finding.start_line == 2


def test_pip_default_inheritance_and_overrides_remain_independent_per_section() -> None:
    content = (
        "[DEFAULT]\n"
        "index-url=https://default.example.invalid/simple\n"
        "extra-index-url=https://extra.example.invalid/simple\n"
        "[global]\n"
        "index-url=https://pypi.org/simple\n"
        "[install]\n"
        "extra-index-url=https://install.example.invalid/simple\n"
        "[download]\n"
        "timeout=30\n"
    )

    analysis = _analyze({"pip.conf": content})

    assert analysis.limitations == ()
    assert [
        (finding.start_line, finding.evidence["operation"], finding.evidence["scope"])
        for finding in analysis.findings
    ] == [
        (2, "replace", "command"),
        (2, "replace", "command"),
        (3, "add", "global"),
        (3, "add", "command"),
        (7, "add", "command"),
    ]


def test_pip_queries_only_relevant_options_per_concrete_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillspector.dependency_sources")
    get_calls: list[tuple[str, str, bool]] = []
    items_calls: list[str] = []
    original_get = module._PipConfigParser.get
    original_items = module._PipConfigParser.items

    def counted_get(
        parser: Any,
        section: str,
        option: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        get_calls.append((section, option, kwargs.get("raw", False)))
        return original_get(parser, section, option, *args, **kwargs)

    def counted_items(parser: Any, section: str, *args: Any, **kwargs: Any) -> Any:
        items_calls.append(section)
        return original_items(parser, section, *args, **kwargs)

    monkeypatch.setattr(module._PipConfigParser, "get", counted_get)
    monkeypatch.setattr(module._PipConfigParser, "items", counted_items)
    irrelevant_defaults = "".join(f"setting-{index}=value-{index}\n" for index in range(64))
    content = (
        "[DEFAULT]\n"
        "index-url=https://default.example.invalid/simple\n"
        f"{irrelevant_defaults}"
        "[global]\n"
        "timeout=30\n"
        "[install]\n"
        "index-url=https://pypi.org/simple\n"
        "[download]\n"
        "extra-index-url=https://download.example.invalid/simple\n"
    )

    analysis = _analyze({"pip.conf": content})

    assert analysis.limitations == ()
    assert [
        (finding.evidence["operation"], finding.evidence["scope"], finding.start_line)
        for finding in analysis.findings
    ] == [
        ("replace", "global", 2),
        ("replace", "command", 2),
        ("add", "command", 72),
    ]
    assert items_calls == []
    assert get_calls == [
        (section, option, True)
        for section in ("global", "install", "download")
        for option in ("index-url", "extra-index-url")
    ]


@pytest.mark.parametrize(
    ("path", "content", "expected_start", "expected_end", "expected_line"),
    [
        (
            ".npmrc",
            "; multibyte é and lone carriage return \r stay on line one\r\n"
            "registry=https://packages.example.invalid/simple\r\n",
            len("; multibyte é and lone carriage return \r stay on line one\r\nregistry=".encode()),
            len(
                "; multibyte é and lone carriage return \r stay on line one\r\n"
                "registry=https://packages.example.invalid/simple".encode()
            ),
            2,
        ),
        (
            "pip.conf",
            "[global]\r\n"
            "# multibyte é and lone carriage return \r stay on line two\r\n"
            "extra-index-url = https://packages.example.invalid/simple\r\n",
            len(
                "[global]\r\n"
                "# multibyte é and lone carriage return \r stay on line two\r\n"
                "extra-index-url = ".encode()
            ),
            len(
                "[global]\r\n"
                "# multibyte é and lone carriage return \r stay on line two\r\n"
                "extra-index-url = https://packages.example.invalid/simple".encode()
            ),
            3,
        ),
    ],
)
def test_source_spans_use_utf8_bytes_and_only_lf_physical_line_boundaries(
    path: str,
    content: str,
    expected_start: int,
    expected_end: int,
    expected_line: int,
) -> None:
    module = importlib.import_module("skillspector.dependency_sources")
    raw = content.encode("utf-8")

    parsed = module._parse_file(
        path,
        content,
        raw,
        DependencyWorkBudget().for_file(path),
    )

    assert parsed.limitations == ()
    assert len(parsed.candidates) == 1
    span = parsed.candidates[0].span
    assert (span.start_byte, span.end_byte) == (expected_start, expected_end)
    assert (span.start_line, span.end_line) == (expected_line, expected_line)


@pytest.mark.parametrize(
    ("path", "content", "expected_status", "expected_destination"),
    [
        (".npmrc", "registry=${NPM_REGISTRY}\n", "unresolved", "unresolved"),
        (".npmrc", "registry=$NPM_REGISTRY\n", "resolved", "[REDACTED_URL]"),
        ("pip.ini", "[global]\nindex-url = %(mirror)s\n", "unresolved", "unresolved"),
        (
            "pip.ini",
            "[global]\nindex-url = https://packages.example.invalid/%2F\n",
            "resolved",
            "[REDACTED_URL]",
        ),
        ("pip.ini", "[global]\nindex-url = $PIP_INDEX_URL\n", "resolved", "[REDACTED_URL]"),
    ],
)
def test_interpolation_is_limited_to_manager_native_forms(
    path: str,
    content: str,
    expected_status: str,
    expected_destination: str,
) -> None:
    analysis = _analyze({path: content})

    assert analysis.limitations == ()
    assert len(analysis.findings) == 1
    assert analysis.findings[0].evidence["destination_status"] == expected_status
    assert analysis.findings[0].evidence["destination"] == expected_destination


@pytest.mark.parametrize(
    ("path", "content"),
    [
        (".npmrc", "registry=HTTPS://REGISTRY.NPMJS.ORG\n"),
        ("pip.conf", "[global]\nindex-url = HTTPS://PYPI.ORG/simple/\n"),
    ],
)
def test_exact_canonical_defaults_ignore_only_case_and_trailing_slash(
    path: str,
    content: str,
) -> None:
    analysis = _analyze({path: content})

    assert analysis.findings == ()
    assert analysis.limitations == ()


@pytest.mark.parametrize(
    "value",
    [
        "https://registry.npmjs.org:443/",
        "https://registry.npmjs.org:/",
        "https://registry.npmjs.org/path",
        "https://registry.npmjs.org/?",
        "https://registry.npmjs.org/?query=1",
        "https://registry.npmjs.org/#",
        "https://registry.npmjs.org/#fragment",
    ],
)
def test_npm_canonical_origin_variants_remain_noncanonical(value: str) -> None:
    analysis = _analyze({".npmrc": f"registry={value}\n"})

    assert len(analysis.findings) == 1
    assert analysis.limitations == ()


def test_dispatch_uses_only_deduplicated_component_exact_basenames() -> None:
    files = {
        "a/.npmrc": "registry=https://a.example.invalid/simple\n",
        "b/pip.ini": "[global]\nindex-url=https://b.example.invalid/simple\n",
        "ignored/.npmrc.backup": "registry=https://ignored.example.invalid/simple\n",
        "cache-only/pip.conf": "[global]\nindex-url=https://ignored.example.invalid/simple\n",
    }

    analysis = _analyze(
        files,
        components=["b/pip.ini", "a/.npmrc", "a/.npmrc", "ignored/.npmrc.backup"],
    )

    assert [finding.file for finding in analysis.findings] == ["a/.npmrc", "b/pip.ini"]
    assert analysis.limitations == ()


@pytest.mark.parametrize(
    ("path", "content", "expected_lines"),
    [
        (
            ".npmrc",
            "@scope:registry=https://first.example.invalid/simple\n"
            "registry=https://second.example.invalid/simple\n"
            "@SCOPE:REGISTRY=https://third.example.invalid/simple\n",
            [2, 3],
        ),
        (
            "pip.conf",
            "[global]\n"
            "index-url=https://first.example.invalid/simple\n"
            "extra-index-url=https://second.example.invalid/simple\n"
            "index_url=https://third.example.invalid/simple\n",
            [3, 4],
        ),
    ],
)
def test_effective_findings_are_ordered_by_occurrence_span(
    path: str,
    content: str,
    expected_lines: list[int],
) -> None:
    analysis = _analyze({path: content})

    assert [finding.start_line for finding in analysis.findings] == expected_lines
    assert analysis.limitations == ()


@pytest.mark.parametrize(
    "mutation",
    ["missing_inventory", "partial_inventory", "missing_raw", "missing_local", "cache_mismatch"],
)
def test_authoritative_input_failures_are_content_free_limitations(mutation: str) -> None:
    path = "pip.conf"
    content = "[global]\nindex-url=https://user:secret@packages.example.invalid/simple\n"
    raw = {path: content.encode()}
    local = {path: content}
    inventory = [classify_artifact(path, raw[path])]
    if mutation == "missing_inventory":
        inventory = []
    elif mutation == "partial_inventory":
        inventory[0]["disposition"] = ArtifactDisposition.PARTIAL
        inventory[0]["reason"] = "size_limit"
    elif mutation == "missing_raw":
        raw = {}
    elif mutation == "missing_local":
        local = {}
    else:
        local[path] = "[global]\nindex-url=https://different.example.invalid/simple\n"

    analysis = _analyze(
        {path: content},
        raw_file_cache=raw,
        local_file_cache=local,
        artifact_inventory=inventory,
    )

    _assert_single_parse_limitation(
        analysis,
        path=path,
        end_line=3 if path in raw else 1,
    )
    assert "secret" not in repr(analysis)


def test_invalid_utf8_is_not_analyzed_through_replacement_text() -> None:
    path = ".npmrc"
    raw = b"registry=https://packages.example.invalid/simple\xff\n"
    inventory = [classify_artifact(path, raw)]

    analysis = _analyze(
        {path: raw.decode("utf-8", errors="replace")},
        raw_file_cache={path: raw},
        artifact_inventory=inventory,
    )

    _assert_single_parse_limitation(analysis, path=path, end_line=2)


def test_inventory_size_proves_incomplete_physical_input_before_parsing() -> None:
    path = ".npmrc"
    content = "registry=https://packages.example.invalid/simple\n"
    raw = content.encode()
    inventory = classify_artifact(path, raw)
    inventory["size_bytes"] = 1_000_001

    analysis = _analyze({path: content}, artifact_inventory=[inventory])

    limitation = _assert_single_parse_limitation(analysis, path=path, end_line=2)
    assert limitation.ledger_metrics() == {
        "observed_bytes": 1_000_001,
        "limit_bytes": 1_000_000,
    }


def test_scan_wide_config_node_exhaustion_is_reported_without_a_partial_result() -> None:
    budget = DependencyWorkBudget()
    assert budget.charge_config_nodes(MAX_DEPENDENCY_CONFIG_NODES) is None

    analysis = _analyze(
        {".npmrc": "registry=https://packages.example.invalid/simple\n"},
        budget=budget,
    )

    limitation = _assert_single_parse_limitation(analysis, path=".npmrc", end_line=2)
    assert limitation.ledger_metrics() == {
        "observed_records": MAX_DEPENDENCY_CONFIG_NODES + 1,
        "limit_records": MAX_DEPENDENCY_CONFIG_NODES,
    }


_BUDGET_LITERAL = "https://packages.example.invalid/simple"


@pytest.mark.parametrize("resource", ["retained", "records"])
def test_candidate_budget_exact_limits_still_emit_the_finding(resource: str) -> None:
    budget = DependencyWorkBudget()
    if resource == "retained":
        assert (
            budget.charge_retained_literal_bytes(
                MAX_DEPENDENCY_RETAINED_LITERAL_BYTES - len(_BUDGET_LITERAL.encode())
            )
            is None
        )
    elif resource == "records":
        assert budget.charge_source_records(MAX_DEPENDENCY_SOURCE_RECORDS - 1) is None

    analysis = _analyze({".npmrc": f"registry={_BUDGET_LITERAL}\n"}, budget=budget)

    assert len(analysis.findings) == 1
    assert analysis.limitations == ()


@pytest.mark.parametrize("resource", ["retained", "records"])
def test_candidate_budget_one_over_preserves_prior_reserved_change_and_adds_limitation(
    resource: str,
) -> None:
    budget = DependencyWorkBudget()
    if resource == "retained":
        assert (
            budget.charge_retained_literal_bytes(
                MAX_DEPENDENCY_RETAINED_LITERAL_BYTES - len(_BUDGET_LITERAL.encode())
            )
            is None
        )
    elif resource == "records":
        assert budget.charge_source_records(MAX_DEPENDENCY_SOURCE_RECORDS - 1) is None
    content = f"registry={_BUDGET_LITERAL}\n@scope:registry={_BUDGET_LITERAL}\n"

    analysis = _analyze({".npmrc": content}, budget=budget)

    assert [finding.start_line for finding in analysis.findings] == [1]
    assert len(analysis.limitations) == 1
    limitation = analysis.limitations[0]
    assert limitation.reason is DependencySourceLimitationReason.PARSE_INCOMPLETE
    assert limitation.ledger_metrics()
    assert set(limitation.ledger_metrics()) in (
        {"observed_bytes", "limit_bytes"},
        {"observed_records", "limit_records"},
    )


@pytest.mark.parametrize(
    "content",
    [
        "index-url=https://packages.example.invalid/simple\n",
        "[global\nindex-url=https://packages.example.invalid/simple\n",
        "[global]\nindex-url=\n",
    ],
)
def test_malformed_pip_configs_are_localized_limitations(content: str) -> None:
    analysis = _analyze({"pip.conf": content})

    _assert_single_parse_limitation(
        analysis,
        path="pip.conf",
        end_line=max(1, content.encode().count(b"\n") + 1),
    )


def test_yarn_v1_uses_case_sensitive_independent_last_values_and_fixed_scopes() -> None:
    content = (
        "  # ignored\n"
        "registry https://first.example.invalid/a#fragment;data\n"
        "Registry https://ignored.example.invalid\n"
        '"@private:registry" "https://user:secret@packages.example.invalid/team" ; note\n'
        "registry https://registry.yarnpkg.com/ # effective default\n"
    )

    analysis = _analyze({"project/.yarnrc": content})

    assert analysis.limitations == ()
    assert _finding_projection(analysis) == [
        {
            "ecosystem": "yarn",
            "surface": "yarn-config",
            "operation": "replace",
            "scope": "scoped",
            "destination": "https://packages.example.invalid/REDACTED_PATH",
            "destination_status": "resolved",
            "file": "project/.yarnrc",
            "start_line": 4,
            "end_line": 4,
        }
    ]
    assert "private" not in repr(analysis)
    assert "secret" not in repr(analysis)


@pytest.mark.parametrize(
    "content",
    [
        "registry https://old.example.invalid\nregistry\n",
        'registry "https://old.example.invalid"\nregistry "https://broken.example.invalid\n',
        '"@private:registry"\n',
        'registry "https://packages.example.invalid"#not-a-comment\n',
    ],
)
def test_yarn_v1_malformed_final_relevant_assignment_does_not_revive_old_value(
    content: str,
) -> None:
    analysis = _analyze({".yarnrc": content})

    _assert_single_parse_limitation(
        analysis,
        path=".yarnrc",
        end_line=content.encode().count(b"\n") + 1,
    )


@pytest.mark.parametrize("path", [".yarnrc.yml", ".yarnrc.yaml"])
def test_yarn_yaml_accepts_flow_quoted_block_and_alias_values_with_exact_spans(
    path: str,
) -> None:
    content = (
        "note: café\r\n"
        'defaults: &registry "https://alias.example.invalid/simple"\r\n'
        '"npmRegistryServer": >-\r\n'
        "  https://global.example.invalid/simple\r\n"
        "npmScopes: {private: {npmRegistryServer: *registry}}\r\n"
    )

    analysis = _analyze({path: content})

    assert analysis.limitations == ()
    assert [
        (
            finding.evidence["scope"],
            finding.evidence["destination"],
            finding.start_line,
            finding.end_line,
        )
        for finding in analysis.findings
    ] == [
        ("global", "https://global.example.invalid/REDACTED_PATH", 3, 4),
        ("scoped", "https://alias.example.invalid/REDACTED_PATH", 5, 5),
    ]
    module = importlib.import_module("skillspector.dependency_sources")
    parsed = module._parse_file(
        path,
        content,
        content.encode(),
        DependencyWorkBudget().for_file(path),
    )
    assert content.encode()[
        parsed.candidates[0].span.start_byte : parsed.candidates[0].span.end_byte
    ].startswith(b">-")
    assert (
        content.encode()[parsed.candidates[1].span.start_byte : parsed.candidates[1].span.end_byte]
        == b"*registry"
    )


@pytest.mark.parametrize(
    "content",
    [
        "npmRegistryServer: https://one.example.invalid\nnpmRegistryServer: https://two.example.invalid\n",
        "npmScopes: 1\n",
        "npmScopes:\n  private: 1\n",
        "npmScopes:\n  private:\n    npmRegistryServer: 1\n",
        "npmScopes:\n  private: {}\n  private: {}\n",
        "npmScopes:\n  private:\n    npmRegistryServer: https://one.example.invalid\n    npmRegistryServer: https://two.example.invalid\n",
        "base: &base {npmRegistryServer: https://one.example.invalid}\nnpmScopes:\n  private:\n    <<: *base\n",
        "base: &base {npmRegistryServer: https://one.example.invalid}\n<<: *base\n",
        "npmRegistryServer: !mirror https://one.example.invalid\n",
        "? [npmRegistryServer]\n: https://one.example.invalid\n",
    ],
)
def test_yarn_yaml_rejects_ambiguous_relevant_shapes(content: str) -> None:
    analysis = _analyze({".yarnrc.yml": content})

    _assert_single_parse_limitation(
        analysis,
        path=".yarnrc.yml",
        end_line=content.encode().count(b"\n") + 1,
    )


def test_yarn_yaml_ignores_unrelated_registry_keys_even_when_duplicated() -> None:
    analysis = _analyze(
        {".yarnrc.yml": ("packageExtensions:\n  pkg:\n    registry: first\n    registry: second\n")}
    )

    assert analysis.findings == ()
    assert analysis.limitations == ()


def test_yarn_yaml_alias_limit_is_exact_and_one_over() -> None:
    exact = (
        "base: &base value\nitems: ["
        + ", ".join("*base" for _ in range(MAX_DEPENDENCY_YAML_ALIASES))
        + "]\n"
    )
    one_over = exact.replace("]\n", ", *base]\n")

    assert _analyze({".yarnrc.yml": exact}).limitations == ()
    limitation = _assert_single_parse_limitation(
        _analyze({".yarnrc.yml": one_over}),
        path=".yarnrc.yml",
        end_line=3,
    )
    assert limitation.ledger_metrics() == {
        "observed_records": MAX_DEPENDENCY_YAML_ALIASES + 1,
        "limit_records": MAX_DEPENDENCY_YAML_ALIASES,
    }


def test_yarn_yaml_recursive_alias_is_a_limitation() -> None:
    content = "npmScopes: &scopes\n  private: *scopes\n"

    analysis = _analyze({".yarnrc.yml": content})

    _assert_single_parse_limitation(analysis, path=".yarnrc.yml", end_line=3)


def test_yarn_yaml_rejects_explicitly_tagged_relevant_key_reached_through_alias() -> None:
    content = (
        "key: &relevant !!str npmRegistryServer\n"
        "*relevant: https://packages.example.invalid/simple\n"
    )

    analysis = _analyze({".yarnrc.yml": content})

    _assert_single_parse_limitation(analysis, path=".yarnrc.yml", end_line=3)


def test_yarn_yaml_rejects_tagged_relevant_root_but_keeps_unrelated_tagged_root_inert() -> None:
    relevant = _analyze(
        {".yarnrc.yml": "!!map {npmRegistryServer: https://packages.example.invalid/simple}\n"}
    )
    unrelated = _analyze({".yarnrc.yml": "!!map {unrelated: value}\n"})

    _assert_single_parse_limitation(relevant, path=".yarnrc.yml", end_line=2)
    assert unrelated.findings == ()
    assert unrelated.limitations == ()


def test_yarn_yaml_node_budget_is_charged_once_before_construction() -> None:
    # Root mapping, key scalar, and value scalar are the three node-producing events.
    exact_budget = DependencyWorkBudget()
    assert exact_budget.charge_config_nodes(MAX_DEPENDENCY_CONFIG_NODES - 3) is None
    assert _analyze({".yarnrc.yml": "unrelated: value\n"}, budget=exact_budget).limitations == ()

    over_budget = DependencyWorkBudget()
    assert over_budget.charge_config_nodes(MAX_DEPENDENCY_CONFIG_NODES - 2) is None
    limitation = _assert_single_parse_limitation(
        _analyze({".yarnrc.yml": "unrelated: value\n"}, budget=over_budget),
        path=".yarnrc.yml",
        end_line=2,
    )
    assert limitation.ledger_metrics() == {
        "observed_records": MAX_DEPENDENCY_CONFIG_NODES + 1,
        "limit_records": MAX_DEPENDENCY_CONFIG_NODES,
    }


def test_yarn_yaml_depth_limit_is_exact_and_one_over() -> None:
    def nested(depth: int) -> str:
        return "root: " + "[" * (depth - 1) + "value" + "]" * (depth - 1) + "\n"

    assert _analyze({".yarnrc.yml": nested(MAX_DEPENDENCY_CONFIG_DEPTH)}).limitations == ()
    limitation = _assert_single_parse_limitation(
        _analyze({".yarnrc.yml": nested(MAX_DEPENDENCY_CONFIG_DEPTH + 1)}),
        path=".yarnrc.yml",
        end_line=2,
    )
    assert limitation.ledger_metrics() == {
        "observed_depth": MAX_DEPENDENCY_CONFIG_DEPTH + 1,
        "limit_depth": MAX_DEPENDENCY_CONFIG_DEPTH,
    }


def test_python_project_sources_apply_manager_specific_operations_and_fixed_scope() -> None:
    content = (
        "[[tool.poetry.source]]\n"
        'name = "primary-name"\n'
        'url = "https://poetry-primary.example.invalid/simple"\n'
        "\n[[tool.poetry.source]]\n"
        'name = "supplement-name"\n'
        'url = "https://poetry-extra.example.invalid/simple"\n'
        'priority = "supplemental"\n'
        "\n[[tool.poetry.source]]\n"
        'name = "explicit-name"\n'
        'url = "https://poetry-explicit.example.invalid/simple"\n'
        'priority = "explicit"\n'
        "\n[[tool.pdm.source]]\n"
        'name = "pypi"\n'
        'url = "https://pdm-primary.example.invalid/simple"\n'
        "\n[[tool.pdm.source]]\n"
        'name = "extra-name"\n'
        'url = "https://pdm-extra.example.invalid/simple"\n'
        "\n[[tool.uv.index]]\n"
        'url = "https://uv-extra.example.invalid/simple"\n'
        "\n[[tool.uv.index]]\n"
        'name = "uv-primary-name"\n'
        'url = "https://uv-primary.example.invalid/simple"\n'
        "default = true\n"
    )

    analysis = _analyze({"pyproject.toml": content})

    assert analysis.limitations == ()
    assert [
        (finding.evidence["ecosystem"], finding.evidence["operation"], finding.start_line)
        for finding in analysis.findings
    ] == [
        ("poetry", "replace", 3),
        ("poetry", "add", 7),
        ("poetry", "add", 12),
        ("pdm", "replace", 17),
        ("pdm", "add", 21),
        ("uv", "add", 24),
        ("uv", "replace", 28),
    ]
    assert {finding.evidence["surface"] for finding in analysis.findings} == {
        "python-project-config"
    }
    assert {finding.evidence["scope"] for finding in analysis.findings} == {"project"}
    for raw_name in (
        "primary-name",
        "supplement-name",
        "explicit-name",
        "extra-name",
        "uv-primary-name",
    ):
        assert raw_name not in repr(analysis)


def test_pdm_alone_models_ascii_environment_substitution_without_environment_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIVATE_INDEX", "https://must-not-be-read.example.invalid")
    content = (
        "[[tool.pdm.source]]\n"
        'name = "private"\n'
        'url = "https://${PRIVATE_INDEX}/simple"\n'
        "[[tool.poetry.source]]\n"
        'name = "private"\n'
        'url = "https://${PRIVATE_INDEX}/simple"\n'
        "[[tool.uv.index]]\n"
        'url = "https://${PRIVATE_INDEX}/simple"\n'
    )

    analysis = _analyze({"pyproject.toml": content})

    assert analysis.limitations == ()
    assert [finding.evidence["destination_status"] for finding in analysis.findings] == [
        "unresolved",
        "resolved",
        "resolved",
    ]
    assert analysis.findings[0].evidence["destination"] == "unresolved"
    assert "must-not-be-read" not in repr(analysis)
    assert "PRIVATE_INDEX" not in repr(analysis)


def test_same_directory_uv_toml_precedes_only_pyproject_uv_tables() -> None:
    pyproject = (
        "[[tool.poetry.source]]\n"
        'name = "private"\n'
        'url = "https://poetry.example.invalid/simple"\n'
        "[[tool.pdm.source]]\n"
        'name = "private"\n'
        'url = "https://pdm.example.invalid/simple"\n'
        "[[tool.uv.index]]\n"
        'url = "https://ignored-uv.example.invalid/simple"\n'
    )
    uv = '[[index]]\nurl = "https://effective-uv.example.invalid/simple"\ndefault = true\n'

    analysis = _analyze({"nested/pyproject.toml": pyproject, "nested/uv.toml": uv})

    assert analysis.limitations == ()
    assert [finding.evidence["ecosystem"] for finding in analysis.findings] == [
        "poetry",
        "pdm",
        "uv",
    ]
    assert "ignored-uv" not in repr(analysis)


def test_uv_toml_does_not_precede_a_pyproject_in_another_directory() -> None:
    pyproject = '[[tool.uv.index]]\nurl = "https://project-uv.example.invalid/simple"\n'
    uv = '[[index]]\nurl = "https://standalone-uv.example.invalid/simple"\n'

    analysis = _analyze({"one/pyproject.toml": pyproject, "two/uv.toml": uv})

    assert analysis.limitations == ()
    assert [finding.file for finding in analysis.findings] == [
        "one/pyproject.toml",
        "two/uv.toml",
    ]


@pytest.mark.parametrize(
    ("path", "content"),
    [
        ("pyproject.toml", "[tool.poetry.source]\nname='x'\nurl='https://x.example.invalid'\n"),
        ("pyproject.toml", "[[tool.poetry.source]]\nurl='https://x.example.invalid'\n"),
        ("pyproject.toml", "[[tool.poetry.source]]\nname=''\nurl='https://x.example.invalid'\n"),
        ("pyproject.toml", "[[tool.poetry.source]]\nname='x'\nurl=''\n"),
        (
            "pyproject.toml",
            "[[tool.poetry.source]]\nname='x'\nurl='https://x.example.invalid'\npriority='secondary'\n",
        ),
        ("pyproject.toml", "[[tool.pdm.source]]\nname=1\nurl='https://x.example.invalid'\n"),
        ("pyproject.toml", "[[tool.uv.index]]\nname=''\nurl='https://x.example.invalid'\n"),
        ("pyproject.toml", "[[tool.uv.index]]\nurl='https://x.example.invalid'\ndefault='true'\n"),
        ("uv.toml", "[index]\nurl='https://x.example.invalid'\n"),
        ("uv.toml", "index=[]\n"),
        ("uv.toml", "[[index]]\nurl=1\n"),
        ("pyproject.toml", "[[tool.uv.index]\nurl='https://x.example.invalid'\n"),
    ],
)
def test_python_project_relevant_shape_and_field_errors_are_limitations(
    path: str,
    content: str,
) -> None:
    analysis = _analyze({path: content})

    _assert_single_parse_limitation(
        analysis,
        path=path,
        end_line=content.encode().count(b"\n") + 1,
    )


def test_python_project_accepts_quoted_dotted_keys_and_anchors_each_url_occurrence() -> None:
    prefix = "# café decoy https://same.example.invalid/simple\r\n"
    first = (
        '[["tool"."poetry"."source"]]\r\n'
        '"name" = "first"\r\n'
        '"url" = "https://same.example.invalid/simple"\r\n'
    )
    second = (
        "[[tool.poetry.source]]\r\n"
        'name = "second"\r\n'
        'url = "https://same.example.invalid/simple"\r\n'
        'priority = "explicit"\r\n'
    )
    content = prefix + first + second

    analysis = _analyze({"pyproject.toml": content})

    assert analysis.limitations == ()
    assert [finding.start_line for finding in analysis.findings] == [4, 7]
    module = importlib.import_module("skillspector.dependency_sources")
    parsed = module._parse_file(
        "pyproject.toml",
        content,
        content.encode(),
        DependencyWorkBudget().for_file("pyproject.toml"),
    )
    assert [
        content.encode()[change.span.start_byte : change.span.end_byte]
        for change in parsed.candidates
    ] == [
        b'"https://same.example.invalid/simple"',
        b'"https://same.example.invalid/simple"',
    ]


def test_python_project_multiline_url_span_covers_its_own_value_token() -> None:
    content = '[[index]]\r\nurl = """https://packages.example.invalid\r\n/simple""" # note\r\n'
    module = importlib.import_module("skillspector.dependency_sources")

    parsed = module._parse_file(
        "uv.toml",
        content,
        content.encode(),
        DependencyWorkBudget().for_file("uv.toml"),
    )

    assert parsed.limitations == ()
    assert len(parsed.candidates) == 1
    span = parsed.candidates[0].span
    assert (span.start_line, span.end_line) == (2, 3)
    assert content.encode()[span.start_byte : span.end_byte] == (
        b'"""https://packages.example.invalid\r\n/simple"""'
    )


def test_python_project_ignores_table_and_key_syntax_inside_multiline_string() -> None:
    content = (
        'description = """\n'
        "[[tool.poetry.source]]\n"
        'url = "https://decoy.example.invalid/simple"\n'
        '"""\n'
        "[[tool.poetry.source]]\n"
        'name = "real"\n'
        'url = "https://packages.example.invalid/simple"\n'
    )

    analysis = _analyze({"pyproject.toml": content})

    assert analysis.limitations == ()
    assert _finding_projection(analysis) == [
        {
            "ecosystem": "poetry",
            "surface": "python-project-config",
            "operation": "replace",
            "scope": "project",
            "destination": "https://packages.example.invalid/REDACTED_PATH",
            "destination_status": "resolved",
            "file": "pyproject.toml",
            "start_line": 7,
            "end_line": 7,
        }
    ]


def test_toml_config_node_budget_is_exact_and_one_over() -> None:
    content = '[[index]]\nurl="https://packages.example.invalid/simple"\n'
    exact_budget = DependencyWorkBudget()
    assert exact_budget.charge_config_nodes(MAX_DEPENDENCY_CONFIG_NODES - 6) is None
    assert _analyze({"uv.toml": content}, budget=exact_budget).limitations == ()

    over_budget = DependencyWorkBudget()
    assert over_budget.charge_config_nodes(MAX_DEPENDENCY_CONFIG_NODES - 5) is None
    limitation = _assert_single_parse_limitation(
        _analyze({"uv.toml": content}, budget=over_budget),
        path="uv.toml",
        end_line=3,
    )
    assert limitation.ledger_metrics() == {
        "observed_records": MAX_DEPENDENCY_CONFIG_NODES + 1,
        "limit_records": MAX_DEPENDENCY_CONFIG_NODES,
    }


def test_toml_depth_limit_is_exact_and_one_over() -> None:
    def nested(parts: int) -> str:
        return f"[{'.'.join(f'a{index}' for index in range(parts))}]\nvalue=1\n"

    assert _analyze({"pyproject.toml": nested(MAX_DEPENDENCY_CONFIG_DEPTH - 1)}).limitations == ()
    limitation = _assert_single_parse_limitation(
        _analyze({"pyproject.toml": nested(MAX_DEPENDENCY_CONFIG_DEPTH)}),
        path="pyproject.toml",
        end_line=3,
    )
    assert limitation.ledger_metrics() == {
        "observed_depth": MAX_DEPENDENCY_CONFIG_DEPTH + 1,
        "limit_depth": MAX_DEPENDENCY_CONFIG_DEPTH,
    }


@pytest.mark.parametrize(
    ("path", "content"),
    [
        (".yarnrc", "registry=https://packages.example.invalid\n"),
        (".yarnrc.yml", "npmRegistryServer: https://registry.yarnpkg.com/\n"),
        (
            "pyproject.toml",
            "[[tool.poetry.source]]\nname='custom'\nurl='https://pypi.org/simple/'\n",
        ),
        ("pyproject.toml", "[[tool.pdm.source]]\nname='custom'\nurl='HTTPS://PYPI.ORG/simple'\n"),
        ("uv.toml", "[[index]]\nurl='https://pypi.org/simple/'\n"),
    ],
)
def test_yarn_and_python_exact_canonical_destinations_are_inert(
    path: str,
    content: str,
) -> None:
    analysis = _analyze({path: content})

    assert analysis.findings == ()
    assert analysis.limitations == ()


@pytest.mark.parametrize(
    ("path", "content"),
    [
        (".yarnrc.yml", "npmRegistryServer: https://registry.yarnpkg.com///\n"),
        (
            "uv.toml",
            "[[index]]\nurl='https://pypi.org/simple///'\n",
        ),
    ],
)
def test_multiple_trailing_slashes_are_not_canonical_defaults(
    path: str,
    content: str,
) -> None:
    analysis = _analyze({path: content})

    assert len(analysis.findings) == 1
    assert analysis.limitations == ()


def test_toml_physical_limit_rejects_before_parser_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillspector.dependency_sources")
    calls: list[str] = []

    def unexpected_loads(text: str) -> object:
        calls.append(text)
        raise AssertionError("tomllib must not be called")

    monkeypatch.setattr(module.tomllib, "loads", unexpected_loads)
    content = "[[index]]\nurl='https://x.example.invalid'\n"
    inventory = classify_artifact("uv.toml", content.encode())
    inventory["size_bytes"] = 1_000_001

    analysis = _analyze({"uv.toml": content}, artifact_inventory=[inventory])

    _assert_single_parse_limitation(analysis, path="uv.toml", end_line=3)
    assert calls == []


@pytest.mark.parametrize("resource", ["retained", "records"])
def test_python_source_budget_one_over_discards_partial_file_results(resource: str) -> None:
    budget = DependencyWorkBudget()
    literal = "https://packages.example.invalid/simple"
    if resource == "retained":
        assert (
            budget.charge_retained_literal_bytes(
                MAX_DEPENDENCY_RETAINED_LITERAL_BYTES - len(literal.encode())
            )
            is None
        )
    elif resource == "records":
        assert budget.charge_source_records(MAX_DEPENDENCY_SOURCE_RECORDS - 1) is None
    content = f'[[index]]\nurl="{literal}"\n[[index]]\nurl="{literal}"\n'

    analysis = _analyze({"uv.toml": content}, budget=budget)

    assert analysis.findings == ()
    assert len(analysis.limitations) == 1
    assert analysis.limitations[0].ledger_metrics()


def test_transient_structured_candidates_do_not_reserve_public_output_capacity() -> None:
    budget = DependencyWorkBudget()
    prior = MAX_DEPENDENCY_SOURCE_CHANGES - 1
    assert budget.reserve_source_changes(prior) is None
    content = (
        '[[index]]\nurl="https://one.example.invalid/simple"\n'
        '[[index]]\nurl="https://two.example.invalid/simple"\n'
    )

    analysis = _analyze({"uv.toml": content}, budget=budget)

    assert [finding.start_line for finding in analysis.findings] == [2, 4]
    assert analysis.limitations == ()
    assert {
        resource: budget.used(resource)
        for resource in (
            DependencyWorkResource.SOURCE_RECORDS,
            DependencyWorkResource.RETAINED_LITERAL_BYTES,
            DependencyWorkResource.EMITTED_CHANGES,
            DependencyWorkResource.FINDING_OUTPUT_RECORDS,
        )
    } == {
        DependencyWorkResource.SOURCE_RECORDS: 2,
        DependencyWorkResource.RETAINED_LITERAL_BYTES: 68,
        DependencyWorkResource.EMITTED_CHANGES: prior,
        DependencyWorkResource.FINDING_OUTPUT_RECORDS: prior,
    }


@pytest.mark.parametrize(
    ("path", "content"),
    [
        (".yarnrc.yml", "unrelated: " + "1" * 5_000 + "\n"),
        ("pyproject.toml", "unrelated = " + "1" * 5_000 + "\n"),
    ],
    ids=("yaml", "toml"),
)
def test_structured_numeric_conversion_failure_is_a_localized_limitation(
    path: str,
    content: str,
) -> None:
    analysis = _analyze({path: content})

    _assert_single_parse_limitation(analysis, path=path, end_line=2)


@pytest.mark.parametrize("path", [".cargo/config", ".cargo/config.toml"])
def test_cargo_resolves_replacement_and_emits_each_exact_configured_occurrence(
    path: str,
) -> None:
    content = (
        "# decoy sparse+https://decoy.example.invalid/index/\n"
        "[source.crates-io]\n"
        'replace-with = "mirror"\n'
        "\n[registries.mirror]\n"
        'index = "sparse+https://packages.example.invalid/index/"\n'
    )

    analysis = _analyze({path: content})

    assert analysis.limitations == ()
    assert _finding_projection(analysis) == [
        {
            "ecosystem": "cargo",
            "surface": "cargo-config",
            "operation": "replace",
            "scope": "source",
            "destination": "sparse+https://packages.example.invalid/REDACTED_PATH",
            "destination_status": "resolved",
            "file": path,
            "start_line": 3,
            "end_line": 3,
        },
        {
            "ecosystem": "cargo",
            "surface": "cargo-config",
            "operation": "add",
            "scope": "registry",
            "destination": "sparse+https://packages.example.invalid/REDACTED_PATH",
            "destination_status": "resolved",
            "file": path,
            "start_line": 6,
            "end_line": 6,
        },
    ]

    module = importlib.import_module("skillspector.dependency_sources")
    parsed = module._parse_file(
        path,
        content,
        content.encode(),
        DependencyWorkBudget().for_file(path),
    )
    assert [
        content.encode()[change.span.start_byte : change.span.end_byte]
        for change in parsed.candidates
    ] == [
        b'"mirror"',
        b'"sparse+https://packages.example.invalid/index/"',
    ]


def test_maven_reports_only_direct_project_repositories_not_false_positive_decoys() -> None:
    content = (
        "<project>\n"
        "  <!-- <repositories><repository><url>https://comment.example.invalid/m2</url>"
        "</repository></repositories> -->\n"
        "  <distributionManagement>\n"
        "    <repository><url>https://release.example.invalid/m2</url></repository>\n"
        "    <snapshotRepository><url>https://snapshot.example.invalid/m2</url>"
        "</snapshotRepository>\n"
        "    <repository><pluginRepositories><pluginRepository>"
        "<url>https://nested-plugin.example.invalid/m2</url>"
        "</pluginRepository></pluginRepositories></repository>\n"
        "  </distributionManagement>\n"
        "  <pluginRepositories><pluginRepository>\n"
        "    <url>https://plugins.example.invalid/m2</url>\n"
        "  </pluginRepository></pluginRepositories>\n"
        "</project>\n"
    )

    analysis = _analyze({"pom.xml": content})

    assert analysis.limitations == ()
    assert _finding_projection(analysis) == [
        {
            "ecosystem": "maven",
            "surface": "maven-config",
            "operation": "add",
            "scope": "repository",
            "destination": "https://plugins.example.invalid/REDACTED_PATH",
            "destination_status": "resolved",
            "file": "pom.xml",
            "start_line": 9,
            "end_line": 9,
        }
    ]


def test_cargo_standalone_sources_and_registries_keep_distinct_equal_url_occurrences() -> None:
    content = (
        "# café\r\n"
        "[source.first-private-name]\r\n"
        'registry = "https://same.example.invalid/index"\r\n'
        "[registries.second-private-name]\r\n"
        'index = "https://same.example.invalid/index"\r\n'
    )

    analysis = _analyze({".cargo/config.toml": content})

    assert analysis.limitations == ()
    assert [finding.start_line for finding in analysis.findings] == [3, 5]
    assert {finding.evidence["surface"] for finding in analysis.findings} == {"cargo-config"}
    assert {finding.evidence["operation"] for finding in analysis.findings} == {"add"}
    assert {finding.evidence["scope"] for finding in analysis.findings} == {"registry"}
    assert len(analysis.findings) == 2
    assert "first-private-name" not in repr(analysis)
    assert "second-private-name" not in repr(analysis)


def test_cargo_two_hop_fan_in_emits_each_replacement_and_target_only_once() -> None:
    content = (
        "[source.first-private-name]\nreplace-with='middle-private-name'\n"
        "[source.second-private-name]\nreplace-with='middle-private-name'\n"
        "[source.middle-private-name]\nreplace-with='target-private-name'\n"
        "[registries.target-private-name]\n"
        "index='sparse+https://packages.example.invalid/index/'\n"
    )

    analysis = _analyze({".cargo/config.toml": content})

    assert analysis.limitations == ()
    assert [finding.evidence["operation"] for finding in analysis.findings] == [
        "replace",
        "replace",
        "replace",
        "add",
    ]
    assert [finding.start_line for finding in analysis.findings] == [2, 4, 6, 8]
    assert {finding.evidence["scope"] for finding in analysis.findings} == {
        "source",
        "registry",
    }
    for private_name in (
        "first-private-name",
        "second-private-name",
        "middle-private-name",
        "target-private-name",
    ):
        assert private_name not in repr(analysis)


class _LookupCountingDict(dict[str, object]):
    def __init__(self, values: Mapping[str, object]) -> None:
        super().__init__(values)
        self.lookups = 0

    def __contains__(self, key: object) -> bool:
        self.lookups += 1
        return super().__contains__(key)

    def get(self, key: str, default: object = None) -> object:
        self.lookups += 1
        return super().get(key, default)


def test_cargo_replacement_resolution_uses_linear_memoized_lookups_near_output_limit() -> None:
    module = importlib.import_module("skillspector.dependency_sources")
    chain_length = MAX_DEPENDENCY_SOURCE_CHANGES - 1
    destination = "sparse+https://packages.example.invalid/index/"
    span = module.SourceSpan(".cargo/config.toml", 0, 1, 1, 1)
    sources = _LookupCountingDict(
        {
            f"source-{index}": (
                "replace-with",
                f"source-{index + 1}" if index + 1 < chain_length else "target",
                span,
            )
            for index in range(chain_length)
        }
    )
    registries = _LookupCountingDict({"target": (destination, span)})
    resolver = getattr(module, "_resolve_cargo_replacements", None)

    assert callable(resolver), "Cargo replacement chains need one memoized resolver"
    resolved = resolver(sources, registries)

    assert resolved == {f"source-{index}": destination for index in range(chain_length)}
    assert sources.lookups + registries.lookups <= chain_length * 8


@pytest.mark.parametrize("family", ["python-toml", "cargo-toml", "maven-xml"])
def test_structured_source_span_line_lookups_are_precomputed_and_linear_near_change_limit(
    family: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillspector.dependency_sources")
    record_count = MAX_DEPENDENCY_SOURCE_CHANGES - 1
    calls = _install_line_lookup_spies(module, monkeypatch)

    if family == "python-toml":
        content = "".join(
            f"[[index]]\nurl='https://host-{index}.example.invalid/simple'\n"
            for index in range(record_count)
        )
        cursors = module._toml_url_cursors("uv.toml", content, frozenset({("index",)}))
        assert cursors is not None
        assert len(cursors[("index",)]) == record_count
    elif family == "cargo-toml":
        content = "".join(
            f"[registries.registry-{index}]\nindex='https://host-{index}.example.invalid/index'\n"
            for index in range(record_count)
        )
        cursors = module._toml_direct_value_cursors(
            ".cargo/config.toml",
            content,
            frozenset({"registries"}),
            frozenset({"index"}),
        )
        assert cursors is not None
        assert len(cursors) == record_count
    else:
        content = (
            "<project><repositories>"
            + "".join(
                f"<repository><url>https://host-{index}.example.invalid/m2</url></repository>"
                for index in range(record_count)
            )
            + "</repositories></project>"
        )
        cursors = module._xml_url_spans("pom.xml", content.encode("utf-8"))
        assert cursors is not None
        assert len(cursors) == record_count

    assert calls == {"builds": 1, "lookups": record_count * 2}


@pytest.mark.parametrize(
    ("source_target", "registry_url"),
    [
        (
            "registry='https://source.example.invalid/index'",
            "https://registry.example.invalid/index",
        ),
        (
            "registry='https://github.com/rust-lang/crates.io-index'",
            "sparse+https://index.crates.io/",
        ),
        (
            "directory='vendor'",
            "https://registry.example.invalid/index",
        ),
    ],
    ids=("configured-registry", "canonical-destinations", "inert-local-source"),
)
def test_cargo_replace_target_collision_between_source_and_registry_is_a_limitation(
    source_target: str,
    registry_url: str,
) -> None:
    content = (
        "[source.origin]\nreplace-with='collision'\n"
        f"[source.collision]\n{source_target}\n"
        f"[registries.collision]\nindex='{registry_url}'\n"
    )

    analysis = _analyze({".cargo/config.toml": content})

    _assert_single_parse_limitation(
        analysis,
        path=".cargo/config.toml",
        end_line=7,
    )


@pytest.mark.parametrize(
    "content",
    [
        "[source.a]\nreplace-with='missing'\n",
        "[source.a]\nreplace-with='b'\n[source.b]\nreplace-with='a'\n",
        "[source.a]\nreplace-with=''\n",
        "[source.a]\nreplace-with=1\n",
        "[source.a]\nregistry=''\n",
        "[source.a]\nregistry='   '\n",
        "[registries.a]\nindex=1\n",
        "[registries.a]\nindex='   '\n",
        "[source.a]\nregistry='https://one.example.invalid'\nregistry='https://two.example.invalid'\n",
        "[source.a]\nreplace-with='b'\nregistry='https://one.example.invalid'\n",
        "[source.a]\ndirectory='vendor'\ngit='https://git.example.invalid/repo'\n",
        "[source.a\nregistry='https://one.example.invalid'\n",
    ],
)
def test_cargo_ambiguous_or_malformed_relevant_configuration_is_a_limitation(
    content: str,
) -> None:
    analysis = _analyze({".cargo/config.toml": content})

    _assert_single_parse_limitation(
        analysis,
        path=".cargo/config.toml",
        end_line=content.encode().count(b"\n") + 1,
    )


@pytest.mark.parametrize(
    "content",
    [
        "[source.a]\ndirectory='vendor'\n",
        "[source.a]\nlocal-registry='vendor/index'\n",
        "[source.a]\ngit='https://git.example.invalid/repo'\n",
        (
            "[source.custom]\n"
            "registry='https://github.com/rust-lang/crates.io-index/'\n"
            "[registries.sparse]\nindex='SPARSE+HTTPS://INDEX.CRATES.IO'\n"
            "[source.origin]\nreplace-with='custom'\n"
        ),
    ],
)
def test_cargo_local_targets_and_exact_canonical_destinations_are_inert(content: str) -> None:
    analysis = _analyze({".cargo/config": content})

    assert analysis.findings == ()
    assert analysis.limitations == ()


def test_cargo_credentials_and_attacker_identifiers_never_cross_public_boundaries() -> None:
    identifier = "private-registry-identifier-7f3c"
    secret = "cargo-secret-4f387"
    content = (
        f"[registries.{identifier}]\n"
        f'index="sparse+https://alice:{secret}@packages.example.invalid/private?token={secret}"\n'
    )

    analysis = _analyze({".cargo/config.toml": content})

    assert len(analysis.findings) == 1
    assert analysis.limitations == ()
    assert secret not in repr(analysis)
    assert identifier not in repr(analysis)
    assert analysis.findings[0].evidence["destination"] == (
        "sparse+https://packages.example.invalid/REDACTED_PATH"
    )


def test_maven_settings_accepts_only_mirrors_and_profile_repository_paths() -> None:
    content = (
        '<settings xmlns="urn:test">\n'
        "  <mirrors><mirror><id>private-mirror-id</id><mirrorOf>private-pattern</mirrorOf>\n"
        "    <url>https://mirror.example.invalid/m2</url></mirror></mirrors>\n"
        "  <profiles><profile><id>private-profile-id</id>\n"
        "    <repositories><repository><url>https://repo.example.invalid/m2</url>"
        "</repository></repositories>\n"
        "    <pluginRepositories><pluginRepository>\n"
        "      <url>https://plugins.example.invalid/m2</url>\n"
        "    </pluginRepository></pluginRepositories>\n"
        "  </profile></profiles>\n"
        "  <repositories><repository><url>https://wrong-depth.example.invalid/m2</url>"
        "</repository></repositories>\n"
        "</settings>\n"
    )

    analysis = _analyze({"settings.xml": content})

    assert analysis.limitations == ()
    assert [
        (finding.evidence["operation"], finding.evidence["scope"], finding.start_line)
        for finding in analysis.findings
    ] == [
        ("replace", "mirror", 3),
        ("add", "repository", 5),
        ("add", "repository", 7),
    ]
    assert {finding.evidence["surface"] for finding in analysis.findings} == {"maven-config"}
    for private_value in ("private-mirror-id", "private-pattern", "private-profile-id"):
        assert private_value not in repr(analysis)


def test_proven_generated_npm_config_dispatch_remaps_to_physical_script_span() -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    raw = b"writer > .npmrc <<EOF\nregistry=https://packages.example.invalid/private\nEOF\n"
    budget = DependencyWorkBudget()
    extraction = shell_frontend.extract_shell_units(
        "scripts/setup.sh",
        raw,
        executable_paths=frozenset(),
        budget=budget,
    )
    frontend = shell_frontend.analyze_shell_unit(extraction.units[0], budget=budget)

    parsed = dependency_sources._parse_generated_configs(
        frontend.generated_configs,
        budget=budget,
    )

    assert parsed.limitations == ()
    assert [
        (
            change.ecosystem.value,
            change.surface.value,
            change.destination,
            change.span.path,
            change.span.start_line,
        )
        for change in parsed.candidates
    ] == [
        (
            "npm",
            "generated-config",
            "https://packages.example.invalid/REDACTED_PATH",
            "scripts/setup.sh",
            2,
        )
    ]


def test_async_pending_heredoc_cannot_create_a_generated_dependency_change() -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    raw = (
        b"cat <<A >/dev/null & cat >.npmrc <<B\n"
        b"registry=https://first.example.invalid\n"
        b"A\n"
        b"# empty\n"
        b"B\n"
    )
    budget = DependencyWorkBudget()
    unit = shell_frontend.extract_shell_units(
        "scripts/generated.sh",
        raw,
        executable_paths=frozenset(),
        budget=budget,
    ).units[0]
    analysis = shell_frontend._analyze_shell_unit(unit, budget=budget)

    parsed = dependency_sources._parse_generated_configs(
        analysis.program.generated_configs,
        budget=budget,
    )

    assert analysis.program.generated_configs == ()
    assert parsed.candidates == ()


def test_generated_configs_follow_typed_function_execution_context() -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")

    def destinations(script: bytes) -> tuple[list[str], Any]:
        budget = DependencyWorkBudget()
        unit = shell_frontend.extract_shell_units(
            "scripts/generated.sh",
            script,
            executable_paths=frozenset(),
            budget=budget,
        ).units[0]
        analysis = shell_frontend._analyze_shell_unit(unit, budget=budget)
        parsed = dependency_sources._parse_generated_configs(
            analysis.program.generated_configs,
            budget=budget,
        )
        return [change.destination for change in parsed.candidates], analysis

    root_definition = b"f(){ writer >.npmrc <<EOF\nregistry=https://root.example.invalid\nEOF\n}\n"
    assert destinations(root_definition + b":\n")[0] == []
    assert destinations(root_definition + b"f\n")[0] == ["https://root.example.invalid"]

    transitive = (
        b"leaf(){ writer >.npmrc <<EOF\nregistry=https://leaf.example.invalid\nEOF\n}\n"
        b"middle(){ leaf; }\n"
        b"top(){ middle; }\n"
    )
    assert destinations(transitive + b":\n")[0] == []
    transitive_destinations, transitive_exact_analysis = destinations(transitive + b"top\n")
    assert transitive_destinations == ["https://leaf.example.invalid"]
    assert transitive_exact_analysis.public.issues == ()

    superseded = (
        b"f(){ writer >.npmrc <<EOF\nregistry=https://old.example.invalid\nEOF\n}\n"
        b"f(){ writer >.npmrc <<EOF\nregistry=https://new.example.invalid\nEOF\n}\n"
        b"f\n"
    )
    assert destinations(superseded)[0] == ["https://new.example.invalid"]

    eval_definition = (
        b"eval 'f(){ writer >.npmrc <<EOF\nregistry=https://eval.example.invalid\nEOF\n}'\n"
    )
    assert destinations(eval_definition + b":\n")[0] == []
    assert destinations(eval_definition + b"f\n")[0] == ["https://eval.example.invalid"]

    containing_eval = (
        b"outer(){ eval 'writer >.npmrc <<EOF\nregistry=https://nested.example.invalid\nEOF\n'; }\n"
    )
    assert destinations(containing_eval + b":\n")[0] == []
    assert destinations(containing_eval + b"outer\n")[0] == ["https://nested.example.invalid"]

    ambiguous = (
        b"if cond; then f(){ writer >.npmrc <<EOF\n"
        b"registry=https://ambiguous.example.invalid\nEOF\n}; fi\n"
        b"f\n"
    )
    ambiguous_destinations, ambiguous_analysis = destinations(ambiguous)
    assert ambiguous_destinations == []
    assert ambiguous_analysis.public.issues

    dynamic_root_destinations, dynamic_root_analysis = destinations(root_definition + b'"$FN"\n')
    assert dynamic_root_destinations == []
    assert dynamic_root_analysis.public.issues

    dynamic_nested = root_definition + b'outer(){ "$FN"; }\n'
    inactive_destinations, inactive_analysis = destinations(dynamic_nested + b":\n")
    assert inactive_destinations == []
    assert inactive_analysis.public.issues == ()
    active_destinations, active_analysis = destinations(dynamic_nested + b"outer\n")
    assert active_destinations == []
    assert active_analysis.public.issues

    transitive_ambiguous = (
        b"leaf(){ writer >.npmrc <<EOF\n"
        b"registry=https://leaf.example.invalid\nEOF\n}\n"
        b"if other; then FN=leaf; fi\n"
        b'if cond; then mid(){ "$FN"; }; fi\n'
        b"mid\n"
    )
    transitive_destinations, transitive_analysis = destinations(transitive_ambiguous)
    assert transitive_destinations == []
    assert transitive_analysis.public.issues

    unrelated_ambiguous = (
        b"leaf(){ writer >.npmrc <<EOF\n"
        b"registry=https://leaf.example.invalid\nEOF\n}\n"
        b"if cond; then mid(){ :; }; fi\n"
        b"mid\n"
    )
    unrelated_destinations, unrelated_analysis = destinations(unrelated_ambiguous)
    assert unrelated_destinations == []
    assert unrelated_analysis.public.issues == ()

    invocation_dependent = (
        b"CFG=.npmrc\n"
        b'f(){ writer >"$CFG" <<EOF\n'
        b"registry=https://stale.example.invalid\nEOF\n}\n"
        b"f\n"
    )
    dependent_destinations, dependent_analysis = destinations(invocation_dependent)
    assert dependent_destinations == []
    assert dependent_analysis.public.issues


def test_eval_function_activation_uses_program_qualified_definition_order() -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")

    def destinations(payload: bytes) -> tuple[list[str], Any]:
        outer = b"f(){ writer >.npmrc <<EOF\nregistry=https://outer.example.invalid\nEOF\n}\n"
        budget = DependencyWorkBudget()
        unit = shell_frontend.extract_shell_units(
            "scripts/generated.sh",
            outer + b"eval '" + payload + b"'\n",
            executable_paths=frozenset(),
            budget=budget,
        ).units[0]
        analysis = shell_frontend._analyze_shell_unit(unit, budget=budget)
        parsed = dependency_sources._parse_generated_configs(
            analysis.program.generated_configs,
            budget=budget,
        )
        return [change.destination for change in parsed.candidates], analysis

    child_definition = (
        b"f(){ writer >.npmrc <<EOF\nregistry=https://child.example.invalid\nEOF\n}\n"
    )
    before_destinations, before = destinations(b"f\n" + child_definition)
    after_destinations, after = destinations(child_definition + b"f\n")
    both_destinations, both = destinations(b"f\n" + child_definition + b"f\n")
    ambiguous_destinations, ambiguous = destinations(
        b"if cond; then " + child_definition.rstrip(b"\n") + b"; fi\nf\n"
    )

    assert before_destinations == ["https://outer.example.invalid"]
    assert before.public.issues == ()
    assert after_destinations == ["https://child.example.invalid"]
    assert after.public.issues == ()
    assert both_destinations == [
        "https://outer.example.invalid",
        "https://child.example.invalid",
    ]
    assert both.public.issues == ()
    assert ambiguous_destinations == []
    assert ambiguous.public.issues


@pytest.mark.parametrize("setup", [b"CFG=.npmrc\n", b"unset CFG\n"])
def test_unquoted_function_target_limitation_follows_typed_activation(
    setup: bytes,
) -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")

    def analyze(tail: bytes) -> tuple[Any, Any]:
        raw = (
            b"f(){ " + setup + b"writer >$CFG <<EOF\n"
            b"registry=https://packages.example.invalid/private\nEOF\n}\n" + tail
        )
        budget = DependencyWorkBudget()
        unit = shell_frontend.extract_shell_units(
            "scripts/generated.sh",
            raw,
            executable_paths=frozenset(),
            budget=budget,
        ).units[0]
        analysis = shell_frontend._analyze_shell_unit(unit, budget=budget)
        parsed = dependency_sources._parse_generated_configs(
            analysis.program.generated_configs,
            budget=budget,
        )
        return analysis, parsed

    inactive, inactive_parsed = analyze(b":\n")
    active, active_parsed = analyze(b"f\n")

    assert inactive.public.issues == ()
    assert inactive.program.generated_configs == ()
    assert inactive_parsed.candidates == ()
    assert inactive_parsed.limitations == ()
    assert active.public.issues
    assert active_parsed.candidates == ()
    assert active_parsed.limitations

    ambiguous_raw = (
        b"if cond; then f(){ " + setup + b"writer >$CFG <<EOF\n"
        b"registry=https://packages.example.invalid/private\nEOF\n}; fi\n"
        b"f\n"
    )
    ambiguous_budget = DependencyWorkBudget()
    ambiguous_unit = shell_frontend.extract_shell_units(
        "scripts/generated.sh",
        ambiguous_raw,
        executable_paths=frozenset(),
        budget=ambiguous_budget,
    ).units[0]
    ambiguous = shell_frontend._analyze_shell_unit(
        ambiguous_unit,
        budget=ambiguous_budget,
    )

    assert ambiguous.program.generated_configs == ()
    assert ambiguous.public.issues


@pytest.mark.parametrize("target", [b"foo bar/.npmrc", b"*/.npmrc"])
def test_generated_config_unquoted_target_expansion_is_never_resolved(
    target: bytes,
) -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")

    def parse(raw: bytes) -> tuple[Any, Any]:
        budget = DependencyWorkBudget()
        unit = shell_frontend.extract_shell_units(
            "scripts/generated.sh",
            raw,
            executable_paths=frozenset(),
            budget=budget,
        ).units[0]
        analysis = shell_frontend._analyze_shell_unit(unit, budget=budget)
        return (
            dependency_sources._parse_generated_configs(
                analysis.program.generated_configs,
                budget=budget,
            ),
            analysis,
        )

    prefix = b"CFG='" + target + b"'\n"
    body = b" <<EOF\nregistry=https://packages.example.invalid/private\nEOF\n"
    unquoted, unquoted_analysis = parse(prefix + b"writer >$CFG" + body)
    quoted, quoted_analysis = parse(prefix + b'writer >"$CFG"' + body)

    assert unquoted.candidates == ()
    assert unquoted.limitations
    assert unquoted_analysis.public.issues
    assert [candidate.destination for candidate in quoted.candidates] == [
        "https://packages.example.invalid/REDACTED_PATH"
    ]
    assert quoted.limitations == ()
    assert quoted_analysis.public.issues == ()


@pytest.mark.parametrize(
    "raw",
    [
        (
            b"HOST=packages.example.invalid\n"
            b"writer >.npmrc <<EOF\nregistry=https://$HOST/private\nEOF\n"
        ),
        (b'HOST=packages.example.invalid\nwriter >.npmrc <<< "registry=https://$HOST/private"\n'),
        (
            b"REGISTRY=https://packages.example.invalid/private\n"
            b"writer >.npmrc <<EOF\nregistry=$REGISTRY\nEOF\n"
        ),
        (b"writer >.npmrc <<EOF\nregistry=https://packages.example.inva\\\nlid/private\nEOF\n"),
    ],
)
def test_generated_config_dispatch_encloses_proven_transformed_value_spans(raw: bytes) -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    budget = DependencyWorkBudget()
    unit = shell_frontend.extract_shell_units(
        "scripts/generated.sh",
        raw,
        executable_paths=frozenset(),
        budget=budget,
    ).units[0]
    frontend = shell_frontend.analyze_shell_unit(unit, budget=budget)

    parsed = dependency_sources._parse_generated_configs(
        frontend.generated_configs,
        budget=budget,
    )

    assert parsed.limitations == ()
    assert len(parsed.candidates) == 1
    change = parsed.candidates[0]
    assert change.destination == "https://packages.example.invalid/REDACTED_PATH"
    assert change.destination_status.value == "resolved"
    assert change.span.path == "scripts/generated.sh"


@pytest.mark.parametrize(
    "target",
    [b'"$HOME/.npmrc"', b"~/.npmrc"],
    ids=("quoted-home", "leading-tilde"),
)
def test_generated_home_npmrc_selector_proof_does_not_taint_exact_content(
    target: bytes,
) -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    raw = (
        b"#!/bin/sh\ncat > " + target + b" <<EOF\nregistry=https://packages.example.invalid\nEOF\n"
    )
    budget = DependencyWorkBudget()
    unit = shell_frontend.extract_shell_units(
        "scripts/generated.sh",
        raw,
        executable_paths=frozenset(),
        budget=budget,
    ).units[0]
    frontend = shell_frontend.analyze_shell_unit(unit, budget=budget)

    assert len(frontend.generated_configs) == 1
    config = frontend.generated_configs[0]
    assert config.target.state.value == "unknown"
    assert config.home_relative_target == b".npmrc"
    parsed = dependency_sources._parse_generated_configs((config,), budget=budget)
    assert parsed.limitations == ()
    assert len(parsed.candidates) == 1
    assert parsed.candidates[0].destination == "https://packages.example.invalid"
    assert parsed.candidates[0].destination_status.value == "resolved"


@pytest.mark.parametrize(
    "raw",
    [
        (b'writer >"$DIR/.npmrc" <<EOF\nregistry=https://packages.example.invalid/private\nEOF\n'),
        (b"writer >.npmrc <<EOF\nregistry=https://$HOST/private\nEOF\n"),
        b"writer >.npmrc <<EOF\nregistry=$REGISTRY\nEOF\n",
    ],
)
def test_generated_config_sink_confined_uncertainty_is_an_unresolved_change(raw: bytes) -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    budget = DependencyWorkBudget()
    unit = shell_frontend.extract_shell_units(
        "scripts/generated.sh",
        raw,
        executable_paths=frozenset(),
        budget=budget,
    ).units[0]
    analysis = shell_frontend._analyze_shell_unit(unit, budget=budget)

    parsed = dependency_sources._parse_generated_configs(
        analysis.program.generated_configs,
        budget=budget,
    )

    assert parsed.limitations == ()
    assert len(parsed.candidates) == 1
    change = parsed.candidates[0]
    assert change.destination == "unresolved"
    assert change.destination_status.value == "unresolved"
    assert change.span.path == "scripts/generated.sh"


@pytest.mark.parametrize(
    "raw",
    [
        b'writer >"$TARGET" <<EOF\nregistry=https://packages.example.invalid\nEOF\n',
        b"writer >.npmrc <<EOF\n$UNKNOWN\nEOF\n",
    ],
)
def test_generated_config_structure_changing_uncertainty_is_a_limitation(raw: bytes) -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    budget = DependencyWorkBudget()
    unit = shell_frontend.extract_shell_units(
        "scripts/generated.sh",
        raw,
        executable_paths=frozenset(),
        budget=budget,
    ).units[0]
    analysis = shell_frontend._analyze_shell_unit(unit, budget=budget)

    parsed = dependency_sources._parse_generated_configs(
        analysis.program.generated_configs,
        budget=budget,
    )

    assert parsed.candidates == ()
    assert len(parsed.limitations) == 1


def test_generated_config_unknown_state_requires_code_owned_uncertainty_markers() -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    raw = b"writer >.npmrc <<EOF\nregistry=https://$HOST/private\nEOF\n"
    budget = DependencyWorkBudget()
    unit = shell_frontend.extract_shell_units(
        "scripts/generated.sh",
        raw,
        executable_paths=frozenset(),
        budget=budget,
    ).units[0]
    config = shell_frontend._analyze_shell_unit(unit, budget=budget).program.generated_configs[0]
    assert config.content_proof is not None
    forged_raw = b"registry=https://packages.example.invalid/private\n"
    base_forged_proof = replace(
        config.content_proof,
        raw_bytes=forged_raw,
        entries=(
            shell_frontend._GeneratedProofEntry(
                0,
                len(forged_raw),
                config.span.start_byte,
                config.span.end_byte,
            ),
        ),
    )
    fake_start = forged_raw.index(b"packages")
    for forged_proof in (
        replace(base_forged_proof, unknown_ranges=()),
        replace(
            base_forged_proof,
            unknown_ranges=((fake_start, fake_start + len(b"packages")),),
        ),
    ):
        parsed = dependency_sources._parse_generated_configs(
            [replace(config, content_proof=forged_proof)],
            budget=budget,
        )

        assert parsed.candidates == ()
        assert len(parsed.limitations) == 1


def test_generated_config_unknown_state_requires_every_uncertainty_marker() -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    raw = (
        b'HOST="$A"\nOTHER="$B"\nwriter >.npmrc <<EOF\n'
        b"registry=https://$HOST/private\n"
        b"@scope:registry=https://$OTHER/private\nEOF\n"
    )
    budget = DependencyWorkBudget()
    unit = shell_frontend.extract_shell_units(
        "scripts/generated.sh",
        raw,
        executable_paths=frozenset(),
        budget=budget,
    ).units[0]
    config = shell_frontend._analyze_shell_unit(unit, budget=budget).program.generated_configs[0]
    assert config.content_proof is not None
    assert len(config.content_proof.unknown_ranges) == 2

    parsed = dependency_sources._parse_generated_configs(
        [
            replace(
                config,
                content_proof=replace(
                    config.content_proof,
                    unknown_ranges=config.content_proof.unknown_ranges[1:],
                ),
            )
        ],
        budget=budget,
    )

    assert parsed.candidates == ()
    assert len(parsed.limitations) == 1


def test_generated_config_rejects_tampered_interior_physical_line_metadata() -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    raw = b'HOST="$DYNAMIC"\nwriter >.npmrc <<EOF\nregistry=https://$HOST/private\nEOF\n'
    budget = DependencyWorkBudget()
    unit = shell_frontend.extract_shell_units(
        "scripts/generated.sh",
        raw,
        executable_paths=frozenset(),
        budget=budget,
    ).units[0]
    config = shell_frontend._analyze_shell_unit(unit, budget=budget).program.generated_configs[0]
    assert config.content_proof is not None
    starts = config.content_proof.physical_line_starts
    assert len(starts) >= 4
    tampered_starts = (*starts[:2], starts[2] - 1, *starts[3:])

    parsed = dependency_sources._parse_generated_configs(
        [
            replace(
                config,
                content_proof=replace(
                    config.content_proof,
                    physical_line_starts=tampered_starts,
                ),
            )
        ],
        budget=budget,
    )

    assert parsed.candidates == ()
    assert len(parsed.limitations) == 1


def test_generated_config_uncovered_uncertainty_fails_before_output_reservations() -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    budget = DependencyWorkBudget()
    raw = (
        b"writer >.npmrc <<EOF\nregistry=https://packages.example.invalid/private\n$UNKNOWN\nEOF\n"
    )
    unit = shell_frontend.extract_shell_units(
        "scripts/generated.sh",
        raw,
        executable_paths=frozenset(),
        budget=budget,
    ).units[0]
    analysis = shell_frontend._analyze_shell_unit(unit, budget=budget)

    parsed = dependency_sources._parse_generated_configs(
        analysis.program.generated_configs,
        budget=budget,
    )

    assert parsed.candidates == ()
    assert len(parsed.limitations) == 1
    assert budget.used(DependencyWorkResource.SOURCE_RECORDS) == 0
    assert budget.used(DependencyWorkResource.EMITTED_CHANGES) == 0


@pytest.mark.parametrize(
    ("target", "content", "ecosystem"),
    [
        ("/opt/homebrew/etc/npmrc", "registry=https://packages.example.invalid\n", "npm"),
        ("pip.conf", "[global]\nindex-url=https://packages.example.invalid\n", "pip"),
        (".yarnrc", 'registry "https://packages.example.invalid"\n', "yarn"),
        (".yarnrc.yml", 'npmRegistryServer: "https://packages.example.invalid"\n', "yarn"),
        (
            ".cargo/config.toml",
            '[registries.internal]\nindex = "https://packages.example.invalid"\n',
            "cargo",
        ),
        (
            "pyproject.toml",
            '[[tool.poetry.source]]\nname = "internal"\nurl = "https://packages.example.invalid"\n',
            "poetry",
        ),
        (
            "uv.toml",
            '[[index]]\nurl = "https://packages.example.invalid"\ndefault = true\n',
            "uv",
        ),
        (
            "settings.xml",
            "<settings><mirrors><mirror><url>https://packages.example.invalid</url>"
            "</mirror></mirrors></settings>\n",
            "maven",
        ),
    ],
)
def test_generated_config_dispatch_is_generic_but_evidence_stays_in_script(
    target: str,
    content: str,
    ecosystem: str,
) -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    raw = f"writer > {target} <<'EOF'\n{content}EOF\n".encode()
    budget = DependencyWorkBudget()
    unit = shell_frontend.extract_shell_units(
        "scripts/generated.sh",
        raw,
        executable_paths=frozenset(),
        budget=budget,
    ).units[0]
    frontend = shell_frontend.analyze_shell_unit(unit, budget=budget)

    parsed = dependency_sources._parse_generated_configs(
        frontend.generated_configs,
        budget=budget,
    )

    assert parsed.limitations == ()
    assert [(change.ecosystem.value, change.surface.value) for change in parsed.candidates] == [
        (ecosystem, "generated-config")
    ]
    assert parsed.candidates[0].span.path == "scripts/generated.sh"


def test_generated_config_invalid_values_and_mapping_gaps_fail_closed_before_output() -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    dependency_types = importlib.import_module("skillspector.dependency_source_types")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    raw = b"writer > .npmrc <<'EOF'\nregistry=https://packages.example.invalid\nEOF\n"

    def config() -> Any:
        budget = DependencyWorkBudget()
        unit = shell_frontend.extract_shell_units(
            "scripts/generated.sh",
            raw,
            executable_paths=frozenset(),
            budget=budget,
        ).units[0]
        return shell_frontend.analyze_shell_unit(unit, budget=budget).generated_configs[0]

    base = config()
    assert base.source_map is not None
    body_entry = base.source_map.entries[0]
    gap = len(base.content.exact_bytes) // 2
    cases = [
        replace(base, target=dependency_types.StaticValue.unknown()),
        replace(base, content=dependency_types.StaticValue.unknown(), source_map=None),
        replace(base, target=dependency_types.StaticValue.exact(b"bad\x00.npmrc")),
        replace(base, content=dependency_types.StaticValue.exact(b"registry=\xff\n")),
        replace(
            base,
            source_map=dependency_types.SourceMap(
                path=base.span.path,
                entries=(),
                child_size_bytes=len(base.content.exact_bytes),
                physical_size_bytes=len(raw),
                physical_line_starts=(
                    0,
                    *(index + 1 for index, byte in enumerate(raw) if byte == 10),
                ),
            ),
        ),
        replace(
            base,
            source_map=dependency_types.SourceMap(
                path=base.span.path,
                entries=(
                    dependency_types.SourceMapEntry(
                        0,
                        gap,
                        body_entry.physical_start_byte,
                        body_entry.physical_start_byte + gap,
                    ),
                    dependency_types.SourceMapEntry(
                        gap + 1,
                        len(base.content.exact_bytes),
                        body_entry.physical_start_byte + gap + 1,
                        body_entry.physical_end_byte,
                    ),
                ),
                child_size_bytes=len(base.content.exact_bytes),
                physical_size_bytes=len(raw),
                physical_line_starts=base.source_map.physical_line_starts,
            ),
        ),
    ]
    for candidate in cases:
        budget = DependencyWorkBudget()
        parsed = dependency_sources._parse_generated_configs([candidate], budget=budget)
        assert parsed.candidates == ()
        assert len(parsed.limitations) == 1
        assert budget.used(DependencyWorkResource.SOURCE_RECORDS) == 0
        assert budget.used(DependencyWorkResource.EMITTED_CHANGES) == 0

    ignored = dependency_sources._parse_generated_configs(
        [replace(base, target=dependency_types.StaticValue.exact(b"notes.txt"))],
        budget=DependencyWorkBudget(),
    )
    assert ignored.candidates == ()
    assert ignored.limitations == ()


def test_generated_config_private_proof_retained_collections_are_bounded() -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    dependency_types = importlib.import_module("skillspector.dependency_source_types")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    limit = dependency_types.MAX_DEPENDENCY_SOURCE_MAP_ENTRIES_PER_FILE

    def proof_config(
        raw: bytes,
        entries: tuple[Any, ...],
        unknown_ranges: tuple[tuple[int, int], ...],
    ) -> Any:
        span = dependency_types.SourceSpan(
            "scripts/generated.sh",
            0,
            len(raw),
            1,
            1,
        )
        proof = shell_frontend._GeneratedValueProof(
            raw,
            entries,
            unknown_ranges,
            True,
            span.path,
            len(raw),
            (0,),
        )
        return shell_frontend._ProvenGeneratedConfig(
            unit_id="0" * 32,
            provenance=dependency_types.SiteProvenance.GENERATED_CONFIG,
            span=span,
            target=dependency_types.StaticValue.exact(b".npmrc"),
            content=dependency_types.StaticValue.unknown(),
            source_map=None,
            content_proof=proof,
            physical_size_bytes=len(raw),
            physical_line_starts=(0,),
        )

    entry_raw = b"x" * (limit + 1)
    too_many_entries = tuple(
        shell_frontend._GeneratedProofEntry(index, index + 1, index, index + 1)
        for index in range(len(entry_raw))
    )
    entry_config = proof_config(entry_raw, too_many_entries, ())

    unknown_raw = b"x" * ((limit + 1) * 2)
    one_entry = (shell_frontend._GeneratedProofEntry(0, len(unknown_raw), 0, len(unknown_raw)),)
    too_many_unknowns = tuple((index * 2, index * 2 + 1) for index in range(limit + 1))
    unknown_config = proof_config(unknown_raw, one_entry, too_many_unknowns)

    assert dependency_sources._generated_proof_view(entry_config, "content_proof") is None
    assert dependency_sources._generated_proof_view(unknown_config, "content_proof") is None


@pytest.mark.parametrize("transformed", [False, True])
def test_generated_config_rejects_tampered_physical_line_metadata(
    transformed: bool,
) -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    raw = (
        b"HOST=packages.example.invalid\n"
        b"writer >.npmrc <<EOF\nregistry=https://$HOST/private\nEOF\n"
        if transformed
        else b"writer >.npmrc <<'EOF'\nregistry=https://packages.example.invalid/private\nEOF\n"
    )
    budget = DependencyWorkBudget()
    unit = shell_frontend.extract_shell_units(
        "scripts/generated.sh",
        raw,
        executable_paths=frozenset(),
        budget=budget,
    ).units[0]
    config = shell_frontend._analyze_shell_unit(unit, budget=budget).program.generated_configs[0]
    assert config.source_map is not None
    if transformed:
        assert config.content_proof is not None
        candidate = replace(
            config,
            content_proof=replace(config.content_proof, physical_line_starts=(0,)),
        )
    else:
        candidate = replace(
            config,
            source_map=replace(config.source_map, physical_line_starts=(0,)),
        )

    parsed = dependency_sources._parse_generated_configs([candidate], budget=budget)

    assert parsed.candidates == ()
    assert len(parsed.limitations) == 1


def test_literal_maven_settings_reference_dispatches_nonstandard_bundle_file() -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    script_path = "scripts/setup.sh"
    settings_path = "ci-settings.xml"
    script = b"mvn -s ci-settings.xml install\n"
    settings = (
        b"<settings>\n"
        b"  <mirrors><mirror>\n"
        b"    <url>https://packages.example.invalid/repository</url>\n"
        b"  </mirror></mirrors>\n"
        b"</settings>\n"
    )
    budget = DependencyWorkBudget()
    extraction = shell_frontend.extract_shell_units(
        script_path,
        script,
        executable_paths=frozenset(),
        budget=budget,
    )
    frontend = shell_frontend._analyze_shell_unit(extraction.units[0], budget=budget)

    parsed = dependency_sources._parse_maven_settings_references(
        frontend.program.execution_commands,
        components=[script_path, settings_path],
        local_file_cache={
            script_path: script.decode(),
            settings_path: settings.decode(),
        },
        raw_file_cache={script_path: script, settings_path: settings},
        artifact_inventory=[
            classify_artifact(script_path, script),
            classify_artifact(settings_path, settings),
        ],
        budget=budget,
    )

    assert parsed.limitations == ()
    assert [
        (
            change.ecosystem.value,
            change.surface.value,
            change.destination,
            change.span.path,
            change.span.start_line,
        )
        for change in parsed.candidates
    ] == [
        (
            "maven",
            "maven-config",
            "https://packages.example.invalid/REDACTED_PATH",
            settings_path,
            3,
        )
    ]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (b"mvn --settings ci-settings.xml install\n", "finding"),
        (b"mvn -s missing.xml install\n", "limitation"),
        (b'mvn -s "$CFG" install\n', "limitation"),
        (b'mvn "$OPT" -s ci-settings.xml install\n', "limitation"),
        (b"mvn --file -s ci-settings.xml install\n", "limitation"),
        (b"mvn --unknown -s ci-settings.xml install\n", "limitation"),
        (b"mvn install -s ci-settings.xml\n", "limitation"),
        (b"mvn -s /ci-settings.xml install\n", "limitation"),
        (b"mvn -s ../ci-settings.xml install\n", "limitation"),
        (b"mvn -s ci-settings.xml --settings ci-settings.xml install\n", "limitation"),
        (b"mvn -- -s ci-settings.xml install\n", "inert"),
        (b"mvn deploy -DaltDeploymentRepository=x::default::https://publish.invalid\n", "inert"),
        (b"other -s ci-settings.xml\n", "inert"),
    ],
)
def test_maven_settings_reference_is_literal_unique_bundle_root_only(
    command: bytes,
    expected: str,
) -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    settings_path = "ci-settings.xml"
    settings = (
        b"<settings><mirrors><mirror>\n"
        b"<url>https://packages.example.invalid/repository</url>\n"
        b"</mirror></mirrors></settings>\n"
    )
    budget = DependencyWorkBudget()
    unit = shell_frontend.extract_shell_units(
        "scripts/setup.sh",
        command,
        executable_paths=frozenset(),
        budget=budget,
    ).units[0]
    frontend = shell_frontend._analyze_shell_unit(unit, budget=budget)
    record = classify_artifact(settings_path, settings)

    parsed = dependency_sources._parse_maven_settings_references(
        frontend.program.execution_commands,
        components=["scripts/setup.sh", settings_path],
        local_file_cache={settings_path: settings.decode()},
        raw_file_cache={settings_path: settings},
        artifact_inventory=[record],
        budget=budget,
    )

    assert len(parsed.candidates) == (1 if expected == "finding" else 0)
    assert len(parsed.limitations) == (1 if expected == "limitation" else 0)


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate",
        "raw-missing",
        "decoded-mismatch",
        "partial-record",
        "size-mismatch",
        "invalid-utf8",
        "wrong-root",
    ],
)
def test_maven_settings_reference_rejects_inconsistent_supplied_bundle_maps(
    defect: str,
) -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    path = "ci-settings.xml"
    settings = (
        b"<settings><mirrors><mirror><url>https://packages.example.invalid</url>"
        b"</mirror></mirrors></settings>\n"
    )
    command = b"mvn -s ci-settings.xml install\n"
    budget = DependencyWorkBudget()
    unit = shell_frontend.extract_shell_units(
        "scripts/setup.sh",
        command,
        executable_paths=frozenset(),
        budget=budget,
    ).units[0]
    commands = shell_frontend._analyze_shell_unit(unit, budget=budget).program.execution_commands
    record = classify_artifact(path, settings)
    active_settings = settings
    if defect == "invalid-utf8":
        active_settings = b"\xff"
        record = classify_artifact(path, active_settings)
    elif defect == "wrong-root":
        active_settings = b"<project/>\n"
        record = classify_artifact(path, active_settings)
    elif defect == "partial-record":
        record = {**record, "decodable": False}
    elif defect == "size-mismatch":
        record = {**record, "size_bytes": len(settings) + 1}
    inventory = [record, record] if defect == "duplicate" else [record]
    raw_cache = {} if defect == "raw-missing" else {path: active_settings}
    local_cache = {
        path: (
            "different"
            if defect == "decoded-mismatch"
            else active_settings.decode("utf-8", errors="replace")
        )
    }

    parsed = dependency_sources._parse_maven_settings_references(
        commands,
        components=["scripts/setup.sh", path],
        local_file_cache=local_cache,
        raw_file_cache=raw_cache,
        artifact_inventory=inventory,
        budget=budget,
    )

    assert parsed.candidates == ()
    assert len(parsed.limitations) == 1


def test_maven_settings_reference_rejects_normalized_bundle_aliases() -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    path = "ci-settings.xml"
    alias = "./ci-settings.xml"
    settings = (
        b"<settings><mirrors><mirror><url>https://packages.example.invalid</url>"
        b"</mirror></mirrors></settings>\n"
    )
    conflicting = settings.replace(b"packages", b"conflict")
    budget = DependencyWorkBudget()
    unit = shell_frontend.extract_shell_units(
        "scripts/setup.sh",
        b"mvn -s ci-settings.xml install\n",
        executable_paths=frozenset(),
        budget=budget,
    ).units[0]
    commands = shell_frontend._analyze_shell_unit(
        unit,
        budget=budget,
    ).program.execution_commands

    parsed = dependency_sources._parse_maven_settings_references(
        commands,
        components=["scripts/setup.sh", path, alias],
        local_file_cache={path: settings.decode(), alias: conflicting.decode()},
        raw_file_cache={path: settings, alias: conflicting},
        artifact_inventory=[
            classify_artifact(path, settings),
            classify_artifact(alias, conflicting),
        ],
        budget=budget,
    )

    assert parsed.candidates == ()
    assert len(parsed.limitations) == 1


def test_maven_settings_reference_rejects_function_shadowed_mvn() -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    path = "ci-settings.xml"
    settings = (
        b"<settings><mirrors><mirror><url>https://packages.example.invalid</url>"
        b"</mirror></mirrors></settings>\n"
    )
    script = b"mvn() { :; }\nmvn -s ci-settings.xml install\n"
    budget = DependencyWorkBudget()
    unit = shell_frontend.extract_shell_units(
        "scripts/setup.sh",
        script,
        executable_paths=frozenset(),
        budget=budget,
    ).units[0]
    commands = shell_frontend._analyze_shell_unit(
        unit,
        budget=budget,
    ).program.execution_commands

    parsed = dependency_sources._parse_maven_settings_references(
        commands,
        components=["scripts/setup.sh", path],
        local_file_cache={path: settings.decode()},
        raw_file_cache={path: settings},
        artifact_inventory=[classify_artifact(path, settings)],
        budget=budget,
    )

    assert parsed.candidates == ()
    assert len(parsed.limitations) == 1


@pytest.mark.parametrize(
    ("script", "expected_limitations"),
    [
        (b"f() { mvn -s ci-settings.xml install; }\n", 0),
        (b". ./defs.sh\nmvn -s ci-settings.xml install\n", 1),
        (b"source ./defs.sh\nmvn -s ci-settings.xml install\n", 1),
    ],
)
def test_maven_settings_reference_honors_private_execution_and_source_barriers(
    script: bytes,
    expected_limitations: int,
) -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    path = "ci-settings.xml"
    settings = (
        b"<settings><mirrors><mirror><url>https://packages.example.invalid</url>"
        b"</mirror></mirrors></settings>\n"
    )
    budget = DependencyWorkBudget()
    unit = shell_frontend.extract_shell_units(
        "scripts/setup.sh",
        script,
        executable_paths=frozenset(),
        budget=budget,
    ).units[0]
    commands = shell_frontend._analyze_shell_unit(
        unit,
        budget=budget,
    ).program.execution_commands

    parsed = dependency_sources._parse_maven_settings_references(
        commands,
        components=["scripts/setup.sh", path],
        local_file_cache={path: settings.decode()},
        raw_file_cache={path: settings},
        artifact_inventory=[classify_artifact(path, settings)],
        budget=budget,
    )

    assert parsed.candidates == ()
    assert len(parsed.limitations) == expected_limitations


@pytest.mark.parametrize(
    ("script", "expected_limitations"),
    [
        (b"run() { mvn -s ci-settings.xml install; }\nrun\n", 1),
        (b"run() { mvn -s ci-settings.xml install; }\neval 'run'\n", 1),
        (b"run() { eval 'mvn -s ci-settings.xml install'; }\nrun\n", 1),
        (b"run() { sh -c 'mvn -s ci-settings.xml install'; }\nrun\n", 1),
        (
            b"run() { mvn -s ci-settings.xml install; }\neval 'other(){ :; }; run'\n",
            1,
        ),
        (
            b"inner() { mvn -s ci-settings.xml install; }\nouter() { inner; }\neval 'outer'\n",
            1,
        ),
        (
            b"danger() { mvn -s ci-settings.xml install; }\n"
            b"outer() { eval 'other(){ :; }; danger'; }\nouter\n",
            1,
        ),
        (
            b"danger() { mvn -s ci-settings.xml install; }\nouter() { eval 'danger'; }\nouter\n",
            1,
        ),
        (
            b"inner() { mvn -s ci-settings.xml install; }\n"
            b"mid() { inner; }\nouter() { eval 'mid'; }\nouter\n",
            1,
        ),
        (b"run() { mvn -s ci-settings.xml install; }\n:\n", 0),
        (b"load() { . ./defs.sh; }\nload\nmvn -s ci-settings.xml install\n", 1),
        (b". ./defs.sh\neval 'mvn -s ci-settings.xml install'\n", 1),
        (
            b"load() { . ./defs.sh; }\n"
            b"eval 'other(){ :; }; load'\n"
            b"mvn -s ci-settings.xml install\n",
            1,
        ),
    ],
)
def test_maven_settings_reference_fails_closed_across_typed_execution_boundaries(
    script: bytes,
    expected_limitations: int,
) -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    path = "ci-settings.xml"
    settings = (
        b"<settings><mirrors><mirror><url>https://packages.example.invalid</url>"
        b"</mirror></mirrors></settings>\n"
    )
    budget = DependencyWorkBudget()
    unit = shell_frontend.extract_shell_units(
        "scripts/setup.sh",
        script,
        executable_paths=frozenset(),
        budget=budget,
    ).units[0]
    commands = shell_frontend._analyze_shell_unit(
        unit,
        budget=budget,
    ).program.execution_commands

    parsed = dependency_sources._parse_maven_settings_references(
        commands,
        components=["scripts/setup.sh", path],
        local_file_cache={path: settings.decode()},
        raw_file_cache={path: settings},
        artifact_inventory=[classify_artifact(path, settings)],
        budget=budget,
    )

    assert parsed.candidates == ()
    assert len(parsed.limitations) == expected_limitations


@pytest.mark.parametrize(
    ("script", "expected_changes"),
    [
        (b"run() { eval 'mvn -s ci-settings.xml install'; }\n:\n", 0),
        (b"run() { sh -c 'mvn -s ci-settings.xml install'; }\n:\n", 0),
        (
            b"load() { eval '. ./defs.sh'; }\nmvn -s ci-settings.xml install\n",
            1,
        ),
    ],
)
def test_maven_settings_reference_preserves_uncalled_nested_function_context(
    script: bytes,
    expected_changes: int,
) -> None:
    dependency_sources = importlib.import_module("skillspector.dependency_sources")
    shell_frontend = importlib.import_module("skillspector.shell_frontend")
    path = "ci-settings.xml"
    settings = (
        b"<settings><mirrors><mirror><url>https://packages.example.invalid</url>"
        b"</mirror></mirrors></settings>\n"
    )
    budget = DependencyWorkBudget()
    unit = shell_frontend.extract_shell_units(
        "scripts/setup.sh",
        script,
        executable_paths=frozenset(),
        budget=budget,
    ).units[0]
    commands = shell_frontend._analyze_shell_unit(
        unit,
        budget=budget,
    ).program.execution_commands

    parsed = dependency_sources._parse_maven_settings_references(
        commands,
        components=["scripts/setup.sh", path],
        local_file_cache={path: settings.decode()},
        raw_file_cache={path: settings},
        artifact_inventory=[classify_artifact(path, settings)],
        budget=budget,
    )

    assert len(parsed.candidates) == expected_changes
    assert parsed.limitations == ()


def test_maven_project_accepts_both_direct_repository_container_types() -> None:
    content = (
        "<project>\n"
        "  <repositories><repository><url>https://repo.example.invalid/m2</url>"
        "</repository></repositories>\n"
        "  <pluginRepositories><pluginRepository><url>https://plugins.example.invalid/m2"
        "</url></pluginRepository></pluginRepositories>\n"
        "</project>\n"
    )

    analysis = _analyze({"pom.xml": content})

    assert analysis.limitations == ()
    assert [finding.start_line for finding in analysis.findings] == [2, 3]
    assert {finding.evidence["scope"] for finding in analysis.findings} == {"repository"}


@pytest.mark.parametrize(
    ("path", "content"),
    [
        (
            "settings.xml",
            "<project><mirrors><mirror><url>https://wrong.example.invalid/m2</url>"
            "</mirror></mirrors></project>\n",
        ),
        (
            "pom.xml",
            "<settings><repositories><repository><url>https://wrong.example.invalid/m2</url>"
            "</repository></repositories></settings>\n",
        ),
        (
            "pom.xml",
            "<project><profiles><profile><repositories><repository>"
            "<url>https://nested.example.invalid/m2</url></repository></repositories>"
            "</profile></profiles></project>\n",
        ),
        (
            "pom.xml",
            "<project><!-- <pluginRepositories><pluginRepository><url>"
            "https://comment.example.invalid/m2</url></pluginRepository>"
            "</pluginRepositories> --></project>\n",
        ),
    ],
)
def test_maven_wrong_roots_nested_paths_and_comments_are_inert(
    path: str,
    content: str,
) -> None:
    analysis = _analyze({path: content})

    assert analysis.findings == ()
    assert analysis.limitations == ()


@pytest.mark.parametrize(
    ("path", "content"),
    [
        (
            "settings.xml",
            "<settings><mirrors><mirror></mirror></mirrors></settings>\n",
        ),
        (
            "settings.xml",
            "<settings><mirrors><mirror><url> </url></mirror></mirrors></settings>\n",
        ),
        (
            "settings.xml",
            "<settings><mirrors><mirror><url>https://one.example.invalid</url>"
            "<url>https://two.example.invalid</url></mirror></mirrors></settings>\n",
        ),
        (
            "pom.xml",
            "<project><repositories><repository><url><value>https://x.example.invalid"
            "</value></url></repository></repositories></project>\n",
        ),
        (
            "pom.xml",
            "<project><repositories><repository><url>https://x.example.invalid<!--x-->"
            "</url></repository></repositories></project>\n",
        ),
        ("pom.xml", "<project><repositories>\n"),
    ],
)
def test_maven_missing_empty_duplicate_unsupported_or_malformed_urls_are_limitations(
    path: str,
    content: str,
) -> None:
    analysis = _analyze({path: content})

    _assert_single_parse_limitation(
        analysis,
        path=path,
        end_line=content.encode().count(b"\n") + 1,
    )


@pytest.mark.parametrize(
    "attribute",
    [
        'unexpected="value"',
        'xmlns:private="urn:test" private:unexpected="value"',
    ],
    ids=("plain", "namespaced"),
)
def test_maven_rejects_attributes_on_accepted_url(attribute: str) -> None:
    content = (
        f"<settings><mirrors><mirror><url {attribute}>"
        "https://packages.example.invalid/m2"
        "</url></mirror></mirrors></settings>\n"
    )

    analysis = _analyze({"settings.xml": content})

    _assert_single_parse_limitation(analysis, path="settings.xml", end_line=2)


@pytest.mark.parametrize(
    "marker",
    [
        "<!DOCTYPE x>",
        "<!ENTITY x 'value'>",
        "<!-- <!DOCTYPE x> -->",
        "<!-- <!ENTITY x 'value'> -->",
        "<![CDATA[<!DOCTYPE x>]]>",
        "<![CDATA[<!ENTITY x 'value'>]]>",
    ],
)
def test_maven_rejects_raw_dtd_and_entity_markers_everywhere_before_parser_construction(
    marker: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillspector.dependency_sources")
    calls: list[object] = []

    def unexpected_parser(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        raise AssertionError("XMLPullParser must not be constructed")

    monkeypatch.setattr(module.ET, "XMLPullParser", unexpected_parser)
    content = f"<settings><value>{marker}</value></settings>\n"

    analysis = _analyze({"settings.xml": content})

    _assert_single_parse_limitation(analysis, path="settings.xml", end_line=2)
    assert calls == []


def test_maven_xml_decoding_canonicality_interpolation_redaction_and_spans() -> None:
    secret = "maven-secret-4f387"
    content = (
        "<settings>\n"
        "  <mirrors>\n"
        "    <mirror><url>https://repo.maven.apache.org/maven2&#x2F;</url></mirror>\n"
        f"    <mirror><url>https://alice:{secret[:5]}&#x2D;{secret[6:]}@packages.example.invalid/private</url></mirror>\n"
        "    <mirror><url>https://${private.repository}/m2</url></mirror>\n"
        "  </mirrors>\n"
        "</settings>\n"
    )

    analysis = _analyze({"settings.xml": content})

    assert analysis.limitations == ()
    assert len(analysis.findings) == 2
    assert [finding.start_line for finding in analysis.findings] == [4, 5]
    assert analysis.findings[0].evidence["destination"] == (
        "https://packages.example.invalid/REDACTED_PATH"
    )
    assert analysis.findings[1].evidence == {
        "ecosystem": "maven",
        "surface": "maven-config",
        "operation": "replace",
        "scope": "mirror",
        "destination": "unresolved",
        "destination_status": "unresolved",
    }
    assert secret not in repr(analysis)
    assert "private.repository" not in repr(analysis)

    module = importlib.import_module("skillspector.dependency_sources")
    parsed = module._parse_file(
        "settings.xml",
        content,
        content.encode(),
        DependencyWorkBudget().for_file("settings.xml"),
    )
    redirected = next(
        candidate for candidate in parsed.candidates if not candidate.canonical_default
    )
    assert content.encode()[redirected.span.start_byte : redirected.span.end_byte].startswith(
        b"https://alice:"
    )


def test_maven_repeated_url_text_uses_accepted_parent_and_utf8_byte_correlation() -> None:
    content = (
        "<project>\r\n"
        "  <name>café https://same.example.invalid/m2</name>\r\n"
        "  <!-- https://same.example.invalid/m2 -->\r\n"
        "  <repositories><repository>\r\n"
        "    <url>https://same.example.invalid/m2</url>\r\n"
        "  </repository></repositories>\r\n"
        "</project>\r\n"
    )

    analysis = _analyze({"pom.xml": content})

    assert analysis.limitations == ()
    assert len(analysis.findings) == 1
    assert analysis.findings[0].start_line == 5
    module = importlib.import_module("skillspector.dependency_sources")
    parsed = module._parse_file(
        "pom.xml",
        content,
        content.encode(),
        DependencyWorkBudget().for_file("pom.xml"),
    )
    span = parsed.candidates[0].span
    assert content.encode()[span.start_byte : span.end_byte] == (b"https://same.example.invalid/m2")


def test_maven_url_span_excludes_surrounding_xml_whitespace() -> None:
    content = (
        "<settings><mirrors><mirror><url> \r\n"
        "  https://packages.example.invalid/m2\t </url></mirror></mirrors></settings>\n"
    )
    module = importlib.import_module("skillspector.dependency_sources")

    parsed = module._parse_file(
        "settings.xml",
        content,
        content.encode(),
        DependencyWorkBudget().for_file("settings.xml"),
    )

    assert parsed.limitations == ()
    assert len(parsed.candidates) == 1
    span = parsed.candidates[0].span
    assert (span.start_line, span.end_line) == (2, 2)
    assert content.encode()[span.start_byte : span.end_byte] == (
        b"https://packages.example.invalid/m2"
    )


@pytest.mark.parametrize(
    ("path", "content"),
    [
        (
            ".cargo/config.toml",
            "[registries.x]\nindex='https://github.com/rust-lang/crates.io-index?query=1'\n",
        ),
        (
            ".cargo/config.toml",
            "[registries.x]\nindex='sparse+https://index.crates.io/#fragment'\n",
        ),
        (
            "settings.xml",
            "<settings><mirrors><mirror><url>https://repo.maven.apache.org:443/maven2/"
            "</url></mirror></mirrors></settings>\n",
        ),
        (
            "settings.xml",
            "<settings><mirrors><mirror><url>https://repo.maven.apache.org/MAVEN2/"
            "</url></mirror></mirrors></settings>\n",
        ),
    ],
)
def test_cargo_and_maven_canonical_origin_variants_remain_noncanonical(
    path: str,
    content: str,
) -> None:
    analysis = _analyze({path: content})

    assert len(analysis.findings) == 1
    assert analysis.limitations == ()


@pytest.mark.parametrize(
    ("path", "raw"),
    [
        (".cargo/config.toml", b"[registries.x]\nindex='https://x.invalid'\xff\n"),
        ("settings.xml", b"<settings>\xff</settings>\n"),
    ],
)
def test_cargo_and_maven_invalid_utf8_are_content_free_limitations(
    path: str,
    raw: bytes,
) -> None:
    analysis = _analyze(
        {},
        components=[path],
        raw_file_cache={path: raw},
        local_file_cache={path: raw.decode("utf-8", errors="replace")},
        artifact_inventory=[classify_artifact(path, raw)],
    )

    limitation = _assert_single_parse_limitation(
        analysis,
        path=path,
        end_line=raw.count(b"\n") + 1,
    )
    assert "x.invalid" not in repr(limitation)


@pytest.mark.parametrize("family", ["cargo", "maven"])
def test_cargo_and_maven_physical_byte_limit_is_exact_and_one_over(family: str) -> None:
    if family == "cargo":
        prefix = "[registries.x]\nindex='https://packages.example.invalid/index'\n#"
        suffix = "\n"
        path = ".cargo/config.toml"
    else:
        prefix = "<settings><!--"
        suffix = "--></settings>"
        path = "settings.xml"
    exact = (
        prefix
        + "x" * (MAX_DEPENDENCY_FILE_BYTES - len(prefix.encode()) - len(suffix.encode()))
        + suffix
    )
    one_over = exact + ("#" if family == "cargo" else " ")

    exact_analysis = _analyze({path: exact})
    over_analysis = _analyze({path: one_over})

    assert exact_analysis.limitations == ()
    limitation = _assert_single_parse_limitation(
        over_analysis,
        path=path,
        end_line=1 if family == "maven" else 4,
    )
    assert limitation.ledger_metrics() == {
        "observed_bytes": MAX_DEPENDENCY_FILE_BYTES + 1,
        "limit_bytes": MAX_DEPENDENCY_FILE_BYTES,
    }


def test_maven_physical_limit_rejects_before_parser_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillspector.dependency_sources")
    calls: list[object] = []

    def unexpected_parser(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        raise AssertionError("XMLPullParser must not be constructed")

    monkeypatch.setattr(module.ET, "XMLPullParser", unexpected_parser)
    content = "<settings/>\n"
    inventory = classify_artifact("settings.xml", content.encode())
    inventory["size_bytes"] = MAX_DEPENDENCY_FILE_BYTES + 1

    analysis = _analyze({"settings.xml": content}, artifact_inventory=[inventory])

    _assert_single_parse_limitation(analysis, path="settings.xml", end_line=2)
    assert calls == []


def test_cargo_physical_limit_rejects_before_toml_parser_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillspector.dependency_sources")
    calls: list[str] = []

    def unexpected_loads(text: str) -> object:
        calls.append(text)
        raise AssertionError("tomllib must not be called")

    monkeypatch.setattr(module.tomllib, "loads", unexpected_loads)
    content = "[registries.x]\nindex='https://x.example.invalid'\n"
    inventory = classify_artifact(".cargo/config.toml", content.encode())
    inventory["size_bytes"] = MAX_DEPENDENCY_FILE_BYTES + 1

    analysis = _analyze({".cargo/config.toml": content}, artifact_inventory=[inventory])

    _assert_single_parse_limitation(analysis, path=".cargo/config.toml", end_line=3)
    assert calls == []


@pytest.mark.parametrize(
    ("family", "path", "content", "nodes"),
    [
        (
            "cargo",
            ".cargo/config.toml",
            "[registries.x]\nindex='https://packages.example.invalid/index'\n",
            7,
        ),
        (
            "maven",
            "settings.xml",
            "<settings><mirrors><mirror><url>https://packages.example.invalid/m2</url>"
            "</mirror></mirrors></settings>\n",
            4,
        ),
    ],
)
def test_cargo_and_maven_node_budget_is_exact_and_one_over(
    family: str,
    path: str,
    content: str,
    nodes: int,
) -> None:
    exact_budget = DependencyWorkBudget()
    assert exact_budget.charge_config_nodes(MAX_DEPENDENCY_CONFIG_NODES - nodes) is None
    assert _analyze({path: content}, budget=exact_budget).limitations == ()

    over_budget = DependencyWorkBudget()
    assert over_budget.charge_config_nodes(MAX_DEPENDENCY_CONFIG_NODES - nodes + 1) is None
    limitation = _assert_single_parse_limitation(
        _analyze({path: content}, budget=over_budget),
        path=path,
        end_line=content.encode().count(b"\n") + 1,
    )
    assert limitation.ledger_metrics() == {
        "observed_records": MAX_DEPENDENCY_CONFIG_NODES + 1,
        "limit_records": MAX_DEPENDENCY_CONFIG_NODES,
    }


@pytest.mark.parametrize("family", ["cargo", "maven"])
def test_cargo_and_maven_depth_limit_is_exact_and_one_over(family: str) -> None:
    if family == "cargo":
        path = ".cargo/config.toml"

        def nested(depth: int) -> str:
            return f"[{'.'.join(f'a{index}' for index in range(depth - 1))}]\nvalue=1\n"
    else:
        path = "settings.xml"

        def nested(depth: int) -> str:
            inner = "<value/>"
            for index in range(depth - 2):
                inner = f"<n{index}>{inner}</n{index}>"
            return f"<settings>{inner}</settings>\n"

    assert _analyze({path: nested(MAX_DEPENDENCY_CONFIG_DEPTH)}).limitations == ()
    limitation = _assert_single_parse_limitation(
        _analyze({path: nested(MAX_DEPENDENCY_CONFIG_DEPTH + 1)}),
        path=path,
        end_line=2 if family == "maven" else 3,
    )
    assert limitation.ledger_metrics() == {
        "observed_depth": MAX_DEPENDENCY_CONFIG_DEPTH + 1,
        "limit_depth": MAX_DEPENDENCY_CONFIG_DEPTH,
    }


@pytest.mark.parametrize(
    ("resource", "family"),
    [
        ("records", "cargo"),
        ("retained", "cargo"),
        ("records", "maven"),
        ("retained", "maven"),
    ],
)
def test_cargo_and_maven_semantic_budget_is_exact_and_one_over(
    resource: str,
    family: str,
) -> None:
    literal = "https://packages.example.invalid/m2"
    if family == "cargo":
        path = ".cargo/config.toml"
        content = f"[registries.x]\nindex='{literal}'\n"
    else:
        path = "settings.xml"
        content = f"<settings><mirrors><mirror><url>{literal}</url></mirror></mirrors></settings>\n"

    def budget_with_remaining(remaining: int) -> DependencyWorkBudget:
        budget = DependencyWorkBudget()
        if resource == "records":
            assert budget.charge_source_records(MAX_DEPENDENCY_SOURCE_RECORDS - remaining) is None
        elif resource == "retained":
            assert (
                budget.charge_retained_literal_bytes(
                    MAX_DEPENDENCY_RETAINED_LITERAL_BYTES - len(literal.encode()) + (1 - remaining)
                )
                is None
            )
        return budget

    assert _analyze({path: content}, budget=budget_with_remaining(1)).limitations == ()
    over = _analyze({path: content}, budget=budget_with_remaining(0))
    assert over.findings == ()
    assert len(over.limitations) == 1
    assert over.limitations[0].ledger_metrics()
