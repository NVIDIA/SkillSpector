"""Static Markdown examples retain JSON syntax without hiding command values."""

import json

import pytest

from skillspector.inspection_ledger import LedgerOutcome, ParserCheck
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner
from skillspector.security_reconstruction import validated_json_string_spans


def _example(body: str) -> str:
    return f"A configuration example.\n\n```json\n{body}\n```\n"


@pytest.mark.parametrize(
    "body",
    [
        '{"timeout": 5}',
        '{"settings": {\n    "timeout"\n    : 5,\n    "nice": true\n}}',
        '{"timeout": "five seconds"}',
    ],
)
def test_complete_json_key_does_not_start_a_shell_command(body: str) -> None:
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["guide.md"], "file_cache": {"guide.md": _example(body)}}, [tm]
    )
    assert result["findings"] == []
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize("command", ['sh -c "$message"', "eval $message", "`printf hello`"])
def test_possible_expansion_inside_valid_json_is_still_partial(command: str) -> None:
    body = json.dumps({"timeout": 5, "command": command})
    source = _example(body)
    diagnostic = tm.bounded_parse_diagnostic(source, lambda: None, file_type="markdown")
    assert diagnostic is not None
    assert diagnostic.check is ParserCheck.COMMAND_WORD_BOUNDARY


@pytest.mark.parametrize(
    "source",
    [
        _example('{"timeout": 5,}'),
        'A configuration example.\n\n```json\n{"timeout": 5}\n',
        _example('{"timeout": 5}').replace("```json", "```sh"),
        _example(json.dumps({"timeout": 5, "value": "a\nb"})),
    ],
)
def test_unproven_key_never_acquires_json_ownership(source: str) -> None:
    keys: list[tuple[int, int]] = []
    validated_json_string_spans(source, lambda: None, literal_key_spans=keys)
    assert keys == []


def test_following_runtime_command_remains_partial() -> None:
    source = _example('{"timeout": 5}') + '\n```sh\neval "$message"\n```\n'
    diagnostic = tm.bounded_parse_diagnostic(source, lambda: None, file_type="markdown")
    assert diagnostic is not None
    assert diagnostic.check is ParserCheck.COMMAND_STRING_RUNTIME
    assert diagnostic.start > source.index("eval")


def test_finding_from_a_json_value_stays_visible() -> None:
    body = json.dumps({"timeout": 5, "example": "mode=0o777"})
    findings = tm.analyze(_example(body), "guide.md", "markdown")
    assert [finding.rule_id for finding in findings] == ["TM3"]
