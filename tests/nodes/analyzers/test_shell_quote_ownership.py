# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from itertools import product
from pathlib import Path

import pytest
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm


@pytest.mark.parametrize("padding", [0, 5000, 9000])
@pytest.mark.parametrize(
    "source",
    [
        'echo "hello $(printf x) ${DELAY}s"\n',
        'mkdir -p "$(dirname "$STORE")" || fail "cannot create $(dirname "$STORE")"\n',
        'echo "count $(get_count "$FILE") open ($MODE) -> enriching"\n',
        'echo "waiting ($PATH) -- waited ${DELAY}s"\nrm -rf "$CACHE"\n',
    ],
)
def test_closed_data_quotes_do_not_consume_later_padding(source: str, padding: int) -> None:
    source += "\n# " + "." * padding + "\n"
    assert not tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell")


@pytest.mark.parametrize(
    "source",
    [
        'echo "x: $($TOOL -rf /)"',
        'echo "x: $(echo "a b"; $TOOL -rf /)"',
        'echo "x: $(printf %s r m) -rf /"',
        '"$CMD $(date)" -rf /',
        'sh -c "$CMD -rf /"',
        'echo "unclosed',
        'echo "$(unterminated',
        pytest.param('echo "$(' + "r" * 4097 + ' -rf /)"', id="over-budget-nested-command"),
        # Public #780 review controls. All remain on the conservative path
        # where ownership is unsupported, or preserve the original frontier.
        'cat <<E\n"$(date)$()""\nE\n$CMD"" -rf / # "\n',
        'echo x\'a "z\'; $(echo rm)"" -rf / --no-preserve-root # "\n',
        'CMD="${TOOL:-rm}"\necho \'X="$(:)\'; $CMD "" -rf / --no-preserve-root #"\n',
        '# x "$(date)\n$CMD"" -rf / # "\n',
    ],
)
@pytest.mark.parametrize("padding", [0, 6000])
def test_runtime_and_unresolved_controls_remain_partial(source: str, padding: int) -> None:
    # Short unclosed literals are not necessarily over the existing budget.
    if not padding and source in {'echo "unclosed', 'echo "$(unterminated'}:
        assert not tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell")
        return
    source += "\n# " + "." * padding + "\n"
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell")


@pytest.mark.parametrize("file_type", ["markdown", "python", "powershell"])
def test_ownership_does_not_apply_to_other_languages(file_type: str) -> None:
    source = 'echo "hello $(printf x) ${DELAY}s"\n' + "#" * 6000
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type=file_type)


def test_fragment_cannot_prove_initial_quote_state() -> None:
    source = 'echo "hello $(printf x) ${DELAY}s"\n' + "#" * 6000
    assert tm.has_bounded_parse_exhaustion(
        source, lambda: None, file_type="shell", complete_context=False
    )


@pytest.mark.parametrize(
    "source",
    [
        'echo "prefix $(date) suffix"\ncat <<E\n"unowned\nE\n',
        'echo "prefix $(date) suffix"\necho `date`\n',
        'echo "prefix $(date) suffix"\necho "unfinished\n',
    ],
)
def test_unsupported_or_unresolved_source_discards_all_ownership(source: str) -> None:
    assert tm._shell_quoted_candidate_limits(source, lambda: None) == {}


def test_ownership_walk_obeys_runtime_budget() -> None:
    def cancel() -> None:
        raise TimeoutError("fixture budget")

    with pytest.raises(TimeoutError, match="fixture budget"):
        tm._shell_quoted_candidate_limits('echo "$(date)"\n', cancel)


def test_quoted_helper_is_completely_inspected(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text(
        "---\nname: repro\ndescription: Minimal shell fixture.\n---\nRun `scripts/run.sh`.\n"
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/run.sh").write_text(
        '#!/bin/bash\necho "hello $(printf x) ${DELAY}s"\n# ' + "." * 6000 + "\n"
    )
    result = CliRunner().invoke(app, ["scan", str(tmp_path), "--format", "json", "--no-llm"])
    assert result.exit_code in {0, 1}, result.output
    report = json.loads(result.output)
    assert report["analysis_completeness"]["status"] == "complete"
    assert report["analysis_completeness"]["ledger_exceptions"] == []
    assert not any(issue["id"] == "AE1" for issue in report["issues"])


def test_compound_adversarial_inputs_retain_security_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefixes = [
        "",
        'echo "hello $(printf x) ${DELAY}s"\n',
        'mkdir -p "$(dirname "$STORE")" || fail "cannot create $(dirname "$STORE")"\n',
        '# x "$(date)\n',
        "echo 'X=\"$(:)'; ",
        'cat <<E\n"$(date)$()""\nE\n',
        "echo x'a \"z'; ",
        'value="quoted"; ',
    ]
    commands = [
        "$CMD -rf /",
        '$CMD"" -rf /',
        '"$CMD" -rf /',
        '$(echo rm)"" -rf / --no-preserve-root',
        "$(printf %s r m) -rf /",
        'sh -c "$CMD -rf /"',
        'eval "$CMD -rf /"',
        'echo "$($CMD -rf /)"',
    ]
    suffixes = ["\n", ' # "\n', "\n# " + "." * 6000 + "\n"]
    for prefix, command, suffix in product(prefixes, commands, suffixes):
        source = prefix + command + suffix
        with monkeypatch.context() as baseline:
            baseline.setattr(tm, "_shell_quoted_candidate_limits", lambda *_: {})
            was_partial = tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell")
        is_partial = tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell")
        if was_partial and not is_partial:
            # A fully reconstructed literal rm can be completely inspected,
            # but must retain its blocking TM1 finding. Unknown runtime-selected
            # commands must retain the original partial result instead.
            assert command == "$(printf %s r m) -rf /", source
            assert any(
                finding.rule_id == "TM1" and finding.severity.value == "HIGH"
                for finding in tm.analyze(source, "run.sh", "shell")
            ), source
