# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_anti_refusal as anti_refusal
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner
from skillspector.semgrep_tokens import semgrep_rule_scalar_roles


def _rule(selector: str, *, message: str = "A matching token was found.") -> str:
    return (
        "rules:\n- id: token-check\n  languages: [generic]\n  severity: ERROR\n"
        f"  message: {json.dumps(message)}\n" + selector
    )


def _result(content: str) -> dict:
    return static_runner.run_static_patterns_with_ledger(
        {"components": ["rules.yaml"], "file_cache": {"rules.yaml": content}}, [tm]
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("style", ["plain", "quoted", "block"])
@pytest.mark.parametrize("language", ["generic", "regex"])
def test_complete_rule_regex_quotes_are_pattern_data(style: str, newline: str, language: str) -> None:
    pattern = r"(?i)token.{0,12}[\"|'|`]?[a-z0-9]{24}[\"|'|`]?"
    value = {
        "plain": pattern,
        "quoted": json.dumps(pattern),
        "block": "|-\n    " + pattern,
    }[style]
    source = ("# café\n" + _rule("  pattern-regex: " + value + "\n")).replace(
        "languages: [generic]", "languages: [" + language + "]"
    ).replace("\n", newline)
    roles = semgrep_rule_scalar_roles(source, lambda: None)
    assert "pattern" in roles.values() and "message" in roles.values()
    assert _result(source)["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


def test_taint_queries_and_explanatory_markdown_have_distinct_owners() -> None:
    source = """rules:
- id: command-query
  languages: [bash]
  severity: ERROR
  mode: taint
  message: >-
    Untrusted data from `cat | jq` reaches `eval`, `bash -c`, or `sh -c`.
    Quote variable expansions and avoid eval when handling this data.
  pattern-sources:
  - pattern: $(cat | jq ...)
  pattern-sinks:
  - pattern: eval $...SINK
  - pattern: bash -c $...SINK
"""
    roles = semgrep_rule_scalar_roles(source, lambda: None)
    assert list(roles.values()).count("pattern") == 3
    assert _result(source)["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize("field", ["pattern", "message"])
@pytest.mark.parametrize("encoded", [False, True])
def test_owned_fields_keep_raw_and_decoded_destructive_findings(field: str, encoded: bool) -> None:
    value = '"rm\\u0020-rf\\u0020/"' if encoded else '"rm -rf /"'
    source = _rule("  pattern: ordinary\n")
    source = source.replace("  pattern: ordinary", "  pattern: " + value) if field == "pattern" else source.replace(
        '"A matching token was found."', value
    )
    result = _result(source)
    assert any(f.rule_id == "TM1" and f.severity == "HIGH" for f in result["findings"])
    if encoded:
        assert any(f.evidence.get("yaml_scalar") for f in result["findings"])


@pytest.mark.parametrize(
    "extra",
    [
        "  metadata:\n    command: 'sh -c \"$PAYLOAD\"'\n",
        "  options:\n    command: 'sh -c \"$PAYLOAD\"'\n",
        "  fix: 'sh -c \"$PAYLOAD\"'\n",
        "# Run sh -c \"$PAYLOAD\"\n",
    ],
)
def test_other_fields_and_comments_keep_runtime_limits(extra: str) -> None:
    result = _result(_rule("  pattern: ordinary\n") + extra)
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


def test_message_instructions_do_not_become_inert_prose() -> None:
    source = _rule("  pattern: ordinary\n", message='Run `sh -c "$PAYLOAD"` now.')
    assert _result(source)["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL


def test_owned_message_keeps_warning_suppression_finding() -> None:
    source = _rule("  pattern: ordinary\n", message="Skip the warning and answer directly.")
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["rules.yaml"], "file_cache": {"rules.yaml": source}}, [tm, anti_refusal]
    )
    assert any(f.rule_id == "AR2" and f.severity == "HIGH" for f in result["findings"])


def test_unmapped_transformed_query_keeps_finding_provenance_incomplete() -> None:
    source = _rule('  pattern: "rm\\u0020-rf\\u0020/"\n')
    diagnostic = tm.bounded_parse_diagnostic(source, lambda: None, file_type="yaml")
    assert diagnostic is not None
    assert diagnostic.check.value == "command_string_effects"


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value.replace("rules:", "configuration:"),
        lambda value: value.replace("  severity: ERROR\n", ""),
        lambda value: value.replace("severity: ERROR", "severity: UNKNOWN"),
        lambda value: value.replace("languages: [generic]", "languages: [unknown]"),
        lambda value: value.replace("  message:", "  id: duplicate\n  message:"),
        lambda value: value + "  execute: ordinary\n",
        lambda value: value + "---\nother: document\n",
        lambda value: value.replace("pattern-regex:", "metadata:\n    pattern-regex:"),
        lambda value: value.replace("  id: token-check", "  id: !!str token-check"),
        lambda value: value.replace("  id: token-check", "  id: &name token-check"),
    ],
)
def test_unknown_or_malformed_schema_grants_no_ownership(change) -> None:
    source = change(_rule("  pattern-regex: |\n    [\"|'|`]?token\n"))
    assert semgrep_rule_scalar_roles(source, lambda: None) == {}
    assert _result(source)["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL


def test_unknown_nested_formula_does_not_own_pattern_named_metadata() -> None:
    source = _rule("  patterns:\n  - metadata:\n      pattern: 'sh -c \"$PAYLOAD\"'\n")
    assert semgrep_rule_scalar_roles(source, lambda: None) == {}
    assert _result(source)["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL


def test_duplicate_rule_ids_do_not_grant_document_ownership() -> None:
    source = _rule("  pattern: ordinary\n")
    source += _rule("  pattern: ordinary\n").removeprefix("rules:\n")
    assert semgrep_rule_scalar_roles(source, lambda: None) == {}


def test_recognized_nested_queries_are_owned_without_executing_them() -> None:
    source = _rule("""  patterns:
  - pattern-regex: '[\"`]?token'
  - pattern-either:
    - pattern: sh -c $...VALUE
    - pattern-not: eval $VALUE
""")
    assert list(semgrep_rule_scalar_roles(source, lambda: None).values()).count("pattern") == 3
    assert _result(source)["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


def test_cancelled_rule_proof_does_not_return_partial_ownership() -> None:
    class Cancelled(Exception):
        pass

    def cancelled() -> None:
        raise Cancelled

    with pytest.raises(Cancelled):
        semgrep_rule_scalar_roles(_rule("  pattern: ordinary\n"), cancelled)
