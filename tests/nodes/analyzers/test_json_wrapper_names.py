# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete JSON metadata must not look like an unfinished shell wrapper."""

from __future__ import annotations

import json
from array import array

import pytest

from skillspector.artifacts import SecurityTextView
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.json_tokens import prepare_json_source
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner

_WRAPPERS = ("nice", "sudo", "xargs", "timeout")
_MANIFEST = ".claude-plugin/plugin.json"


def _scan(source: str) -> dict:
    return static_runner.run_static_patterns_with_ledger(
        {"components": [_MANIFEST], "file_cache": {_MANIFEST: source}}, [tm]
    )


@pytest.mark.parametrize("wrapper", _WRAPPERS)
@pytest.mark.parametrize(
    "template",
    [
        '{"name": %s}',
        '{"keywords": [%s]}',
        '{"metadata": {"name": %s}}',
        "%s",
        "[%s]",
        '{\n  "name": %s\n}\n',
    ],
    ids=["name", "keywords", "nested", "scalar", "array", "pretty"],
)
def test_json_wrapper_name_is_fully_inspected(wrapper: str, template: str) -> None:
    result = _scan(template % json.dumps(wrapper))

    assert result["findings"] == []
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert "reason_code" not in result["inspection_ledger"][0]


@pytest.mark.parametrize("wrapper", _WRAPPERS)
@pytest.mark.parametrize("spelling", ["uppercase", "absolute", "unicode"])
def test_json_wrapper_name_spelling_is_fully_inspected(wrapper: str, spelling: str) -> None:
    if spelling == "uppercase":
        source = json.dumps({"name": wrapper.upper()})
    elif spelling == "absolute":
        source = json.dumps({"name": "/usr/bin/" + wrapper})
    else:
        source = f'{{"name": "\\u{ord(wrapper[0]):04x}{wrapper[1:]}"}}'

    result = _scan(source)

    assert result["findings"] == []
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize("wrapper", _WRAPPERS)
def test_json_wrapper_name_keeps_destructive_command_findings(wrapper: str) -> None:
    prefix = "timeout 5" if wrapper == "timeout" else wrapper
    source = json.dumps({"name": wrapper, "command": prefix + " rm -rf /"})

    result = _scan(source)

    assert any(
        finding.rule_id == "TM1" and finding.severity == "HIGH" for finding in result["findings"]
    )


@pytest.mark.parametrize("wrapper", _WRAPPERS)
def test_json_wrapper_name_keeps_unresolved_command_partial(wrapper: str) -> None:
    prefix = "timeout 5" if wrapper == "timeout" else wrapper
    source = json.dumps({"name": wrapper, "command": prefix + ' sh -c "$CMD -rf /"'})

    result = _scan(source)

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
    assert result["inspection_ledger"][0]["parser_diagnostic"]["cause"] == "runtime_dependent"


@pytest.mark.parametrize(
    "arguments",
    [
        ["sudo", "sh", "-c", "$CMD -rf /"],
        ["nice", "sh", "-c", "$CMD -rf /"],
        ["xargs", "sh", "-c", "$CMD -rf /"],
        ["timeout", "5", "sh", "-c", "$CMD -rf /"],
        ["sudo", "eval", "$CMD -rf /"],
    ],
)
def test_json_wrapper_name_keeps_argv_consumer_partial(arguments: list[str]) -> None:
    result = _scan(json.dumps(arguments))

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["parser_diagnostic"]["cause"] == "runtime_dependent"


@pytest.mark.parametrize(
    "arguments",
    [["sudo", "-n"], ["nice", "-n"], ["xargs", "-n"], ["timeout", "-s"]],
)
def test_json_wrapper_with_consumed_options_stays_partial(arguments: list[str]) -> None:
    result = _scan(json.dumps(arguments))

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize("wrapper", _WRAPPERS)
def test_runtime_selected_json_wrapper_stays_partial(wrapper: str) -> None:
    result = _scan(json.dumps({"name": "$TOOL/" + wrapper}))

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize("wrapper", _WRAPPERS)
def test_json_wrapper_fragment_has_no_complete_scalar_proof(wrapper: str) -> None:
    assert (
        tm.bounded_parse_diagnostic(
            json.dumps(wrapper), lambda: None, file_type="json", complete_context=False
        )
        is not None
    )


def test_clipped_json_argv_cannot_claim_its_native_wrappers_eof() -> None:
    source = json.dumps(["sudo", "sh", "-c", "$CMD -rf /"])
    proof = prepare_json_source(source, lambda: None)
    assert proof is not None
    end = source.index('", "sh"') + 1
    view = SecurityTextView("window", source[:end], array("I", range(end)))

    assert (
        tm.bounded_parse_diagnostic(
            view.text,
            lambda: None,
            file_type="json",
            complete_context=False,
            prepared_analysis=proof,
            source_view=view,
        )
        is not None
    )
