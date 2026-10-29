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


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_nested_path_quote_after_here_string_does_not_consume_later_commands(newline: str) -> None:
    source = '''#!/bin/bash
mapfile -t matches < <(awk 'NF {print $1}' <<<"$listing")
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
root="$work/bundle"
cp "$(dirname "${SOURCE[0]}")/helper.sh" \\
  "$root/helper.sh"
chmod 700 "$root/helper.sh"
tool="$(resolve_one "$SPEC")"
archive="$work/archive.tbz2"
fetch "$tool" > "$archive"
tar -xf "$archive" -C "$root"
files=("$tool")
for file in first.py second.py; do
  item="$(resolve_one "$BASE/$file")"
  fetch "$item" > "$root/$file"
done
[[ ${#files[@]} -eq 1 ]] || {
  echo "Expected one nonempty tool in the downloaded archive" >&2
  exit 1
}
chmod 700 "${files[0]}"
printf 'Created %s\\n' "$root"
'''.replace("\n", newline)

    assert not tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell")


@pytest.mark.parametrize("operator", ["<<<", "<<< "])
def test_here_string_keeps_following_quote_bounds(operator: str) -> None:
    source = f'cat {operator}"$VALUE"\necho "waiting $(get_count) ${{DELAY}}s"\n'
    source += "# " + "." * 6000 + "\n"

    assert not tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell")


@pytest.mark.parametrize(
    "command",
    [
        '$CMD -rf /',
        '"$CMD" -rf /',
        'echo "x: $($CMD -rf /)"',
        'echo "x: $(echo "a b"; $CMD -rf /)"',
        'sh -c "$CMD -rf /"',
    ],
)
def test_here_string_and_nested_quote_closers_do_not_hide_runtime_commands(command: str) -> None:
    source = 'cat <<<"$VALUE"\ncp "$(dirname "$SOURCE")/file" "$DEST"\n'
    source += command + "\n# " + "." * 6000 + "\n"

    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell")


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("header", ["<<END", "<<'END'", '<<"END"', "<<-END"])
def test_complete_heredoc_keeps_later_nested_quote_ownership(header: str, newline: str) -> None:
    source = f"cat {header}\nordinary help text\nEND\n"
    source += '''state="$(inspect -f '{{value}}' "$NAME")" || return
[[ "$state" == true\\|true\\|* &&
   " ${state#*|*|} " == *' /dev '* &&
   " ${state#*|*|} " == *' /tools '* ]] || {
  echo "Required mounts are missing" >&2
  return 1
}
output="$(detect_devices)" || return
printf '%s\\n' "$output"
trap 'rm -rf "$CACHE"' EXIT
'''

    assert not tm.has_bounded_parse_exhaustion(
        source.replace("\n", newline), lambda: None, file_type="shell"
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_multiple_heredocs_are_consumed_in_declaration_order(newline: str) -> None:
    source = "cat <<'FIRST' <<-SECOND\nfirst body\nFIRST\n\tsecond body\n\tSECOND\n"
    source += 'echo "waiting $(get_count) ${DELAY}s"\n# ' + "." * 6000 + "\n"

    assert not tm.has_bounded_parse_exhaustion(
        source.replace("\n", newline), lambda: None, file_type="shell"
    )


@pytest.mark.parametrize("indent", [" ", " \t", "\t "])
def test_tab_stripping_heredoc_does_not_strip_spaces(indent: str) -> None:
    source = f"$(cat <<-END\nbody\n{indent}END\n)"
    end_cache: dict[int, int | None] = {}

    assert tm._skip_command_substitution(source, 0, len(source), end_cache=end_cache) is None
    assert end_cache == {0: None}


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize(
    ("body", "strip_tabs", "quoted"),
    [
        ("ordinary \\\ncontinued\nEND\n", False, False),
        ("E\\\nN\\\nD\n", False, False),
        ("body\\\\\nEND\n", False, False),
        ("E\\\\\\\nND\nEND\n", False, False),
        ("\tE\\\nND\n", True, False),
        ("E\\\n\tND\nEND\n", True, False),
        ("E\\\nND\nEND\n", False, True),
    ],
)
def test_heredoc_continuations_compare_logical_delimiter_lines(
    body: str, strip_tabs: bool, quoted: bool, newline: str
) -> None:
    body = body.replace("\n", newline)
    source = body + "$CMD -rf /\n"

    assert tm._skip_static_heredoc_bodies(
        source, 0, len(source), (("END", strip_tabs, quoted),), lambda: None
    ) == len(body)


@pytest.mark.parametrize("body", ["END", "E\\\nND", "\tE\\\nND"])
def test_heredoc_delimiter_can_end_at_eof(body: str) -> None:
    assert tm._skip_static_heredoc_bodies(
        body, 0, len(body), (("END", True, False),), lambda: None
    ) == len(body)


def test_large_continued_heredoc_checks_runtime_and_keeps_raw_commands() -> None:
    body = ("x" * 5000 + "\\\n") * 40 + "END\n$CMD -rf /\nEND\n"
    checks = 0

    def check_runtime() -> None:
        nonlocal checks
        checks += 1

    assert tm._skip_static_heredoc_bodies(
        body, 0, len(body), (("END", False, False),), check_runtime
    ) == len(body)
    assert checks >= 40
    assert tm.has_bounded_parse_exhaustion(
        "cat <<END\n" + body, lambda: None, file_type="shell"
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_heredoc_body_parentheses_cannot_close_an_enclosing_substitution(newline: str) -> None:
    source = "$(cat <<'END'\n) and an unmatched quote: \"\nEND\nprintf ok\n)"
    source = source.replace("\n", newline)

    assert tm._skip_command_substitution(source, 0, len(source), lambda: None) == len(source)


@pytest.mark.parametrize("delimiter", ["END", "'END'", '"END"'])
@pytest.mark.parametrize("command", ["$CMD -rf /", 'echo "$($CMD -rf /)"'])
def test_heredoc_ownership_keeps_runtime_payloads_partial(delimiter: str, command: str) -> None:
    source = f"cat <<{delimiter}\n{command}\nEND\n"
    source += 'echo "waiting $(get_count) ${DELAY}s"\n'

    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell")


@pytest.mark.parametrize("delimiter", ["END", "'END'"])
def test_heredoc_ownership_keeps_literal_destructive_findings(delimiter: str) -> None:
    source = f"cat <<{delimiter}\nrm -rf /\nEND\necho ready\n"

    assert any(finding.rule_id == "TM1" for finding in tm.analyze(source, "run.sh", "shell"))


@pytest.mark.parametrize("header", ["<<'END-MARK'", "<<\\END", "<<$END", "<<'END", "<<"])
def test_unsupported_heredoc_delimiter_never_proves_a_substitution_end(header: str) -> None:
    source = f"$(cat {header}\n)\nEND\n)"
    end_cache: dict[int, int | None] = {}

    assert tm._skip_command_substitution(source, 0, len(source), end_cache=end_cache) is None
    assert end_cache == {0: None}


@pytest.mark.parametrize("separator", [" ", "\t"])
@pytest.mark.parametrize("strip_tabs", [False, True])
def test_heredoc_header_whitespace_accepts_exact_span_boundary(
    separator: str, strip_tabs: bool
) -> None:
    prefix = "cat "
    source = prefix + ("<<-" if strip_tabs else "<<")
    source += separator * tm._SHELL_COMMAND_WORD_CHARS + "'END'"

    assert tm._static_heredoc_header(source, len(prefix), len(source)) == (
        ("END", strip_tabs, True), len(source)
    )


@pytest.mark.parametrize("separator", [" ", "\t"])
def test_heredoc_header_whitespace_over_span_boundary_fails_closed(separator: str) -> None:
    source = "$(cat <<" + separator * (tm._SHELL_COMMAND_WORD_CHARS + 1) + "END\n)\nEND\n)"
    end_cache: dict[int, int | None] = {}

    assert tm._static_heredoc_header(source, source.index("<<"), len(source)) is None
    assert tm._skip_command_substitution(source, 0, len(source), end_cache=end_cache) is None
    assert end_cache == {0: None}


@pytest.mark.parametrize("suffix", ["", "\nEND\n)"])
def test_pending_heredoc_cannot_close_before_its_body(suffix: str) -> None:
    source = "$(cat <<END)" + suffix

    assert tm._skip_command_substitution(source, 0, len(source), lambda: None) is None


def test_pending_heredoc_cannot_claim_a_plain_subshell_body() -> None:
    source = "$(cat <<END; (printf ok\n) body\nEND\n))"
    end_cache: dict[int, int | None] = {}
    closers: set[int] = set()

    assert tm._skip_command_substitution(source, 0, len(source), end_cache=end_cache) is None
    assert end_cache == {0: None}
    assert tm._shell_quoted_candidate_limits(source, lambda: None, quote_closers=closers) == {}
    assert closers == set()


def test_pending_heredoc_keeps_independent_command_substitution_arguments() -> None:
    source = '$(cat <<END "$(printf ok)"\nbody\nEND\n)'

    assert tm._skip_command_substitution(source, 0, len(source), lambda: None) == len(source)


def test_arithmetic_shift_is_not_a_heredoc_declaration() -> None:
    for source in ("$((1 << count))", "$(echo $((1 << count)))", "$( ((1 << count)) )"):
        assert tm._skip_command_substitution(source, 0, len(source), lambda: None) == len(source)


@pytest.mark.parametrize("operator", ["#", "##", "%", "%%"])
@pytest.mark.parametrize("variable", ["$ROOT", "${ROOT}"])
def test_simple_quoted_trim_pattern_preserves_outer_quote_ownership(
    operator: str, variable: str
) -> None:
    source = 'printf "%s" "${path' + operator + '"' + variable + '"/}"\n'
    source += '''state="$(inspect -f '{{value}}' "$NAME")" || return
[[ "$state" == true\\|true\\|* && " ${state#*|*|} " == *' /dev '* ]] || return
trap 'rm -rf "$CACHE"' EXIT
'''

    assert not tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell")
    assert tm.has_bounded_parse_exhaustion(
        source + "$CMD -rf /\n", lambda: None, file_type="shell"
    )


@pytest.mark.parametrize(
    "pattern",
    ['${path#"$($CMD -rf /)"}', '${path#"$ROOT"$"suffix"}', '${path#"${ROOT:-other}"}'],
)
def test_complex_trim_patterns_cannot_claim_simple_quote_ownership(pattern: str) -> None:
    source = 'printf "%s" "' + pattern + '"\n'
    closers: set[int] = set()

    assert tm._shell_quoted_candidate_limits(source, lambda: None, quote_closers=closers) == {}
    assert closers == set()
    assert tm.has_bounded_parse_exhaustion(
        source + "$CMD -rf /\n", lambda: None, file_type="shell"
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_literal_shell_command_has_its_own_quote_ownership(newline: str) -> None:
    source = """MODE=test bash -c '
if [[ "$1 $2" == "docker exec" &&
      "$*" != *" /bin/bash -c "* ]]; then
  return 127
fi
if [[ "$1 $2" == "docker exec" && "$*" == *" ldd "* ]]; then
  printf "%s\\n" "library found"
fi
command "$@"
'
trap 'rm -rf "$CACHE"' EXIT
"""

    assert not tm.has_bounded_parse_exhaustion(
        source.replace("\n", newline), lambda: None, file_type="shell"
    )


@pytest.mark.parametrize("command", ["$CMD -rf /", 'echo "$($CMD -rf /)"', 'sh -c "$CODE"'])
def test_literal_shell_command_ownership_keeps_unknown_commands_partial(command: str) -> None:
    source = f"bash -c '\necho ready\n{command}\n'\n"

    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell")


@pytest.mark.parametrize("argument", ['"$SCRIPT"', "'echo '\"ready\"", "$'echo ready'"])
def test_transformed_shell_command_cannot_claim_original_literal_offsets(argument: str) -> None:
    source = f"bash -c {argument}\n"

    assert tm._literal_shell_command_spans(source, 0, len(source), lambda: None) == {}


@pytest.mark.parametrize("shape", ["parameter", "alternating", "comment", "whitespace"])
def test_literal_command_before_hostile_nesting_keeps_proof_work_bounded(
    shape: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    repetitions = 2000
    if shape == "parameter":
        tail = '$(env "X=${x:- ' * repetitions + "A" + '}" echo printf)' * repetitions + " -rf *"
    elif shape == "alternating":
        tail = "$(env X=" + "${x:-$(echo " * repetitions + "z" + ")}" * repetitions + " echo printf) -rf /"
    elif shape == "comment":
        tail = "$(env X=x $(echo " * repetitions + "# no newline" + ")" * repetitions
    else:
        tail = "\n" * 40000 + "$CMD -rf /\n"
    source = "bash -c 'echo ready'\n" + tail
    scanned = 0
    checks = 0
    original = tm._skip_shell_delimited_expansion

    def counted(content: str, start: int, limit: int, *args, **kwargs):
        nonlocal scanned
        result = original(content, start, limit, *args, **kwargs)
        scanned += (limit if result is None else result) - start
        return result

    def check_runtime() -> None:
        nonlocal checks
        checks += 1

    monkeypatch.setattr(tm, "_skip_shell_delimited_expansion", counted)
    closers: set[int] = set()
    tm._shell_quoted_candidate_limits(source, check_runtime, quote_closers=closers)
    assert scanned <= 8 * len(source)
    assert checks > 0
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell")


@pytest.mark.parametrize("wrapper", ["bash", "env MODE=test bash", "sudo -n bash", "timeout 5 bash"])
def test_literal_command_proof_selects_only_relevant_clauses(
    wrapper: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = "echo ready\n" * 1000
    source = prefix + wrapper + " -c 'echo ready'\n"
    selected: list[int] = []
    original = tm._command_string_from_clause

    def recorded(content, start, check_runtime, **kwargs):
        selected.append(start)
        return original(content, start, check_runtime, **kwargs)

    monkeypatch.setattr(tm, "_command_string_from_clause", recorded)
    spans = tm._literal_shell_command_spans(source, 0, len(source), lambda: None)

    assert selected == [len(prefix)]
    assert spans is not None and len(spans) == 1
    assert source[next(iter(spans)) : next(iter(spans.values()))] == "echo ready"


@pytest.mark.parametrize(
    "source",
    [
        "# bash -c 'echo ready'\n",
        "echo \"bash -c 'echo ready'\"\n",
        "echo \"$(bash -c 'echo ready')\"\n",
        "echo -" + "c" * 100000 + " /tmp/value'\n",
    ],
)
def test_literal_command_option_lookalikes_cannot_claim_a_body(source: str) -> None:
    assert tm._literal_shell_command_spans(source, 0, len(source), lambda: None) == {}


def test_selected_wrapper_budget_discards_previously_collected_literal_proof() -> None:
    source = "bash -c 'echo ready'\n"
    source += "eval " + "'part' " * 24 + '\"-c \'echo ready\'\" \"$CMD -rf /\"\n'
    closers: set[int] = set()

    assert tm._literal_shell_command_spans(source, 0, len(source), lambda: None) is None
    assert tm._shell_quoted_candidate_limits(source, lambda: None, quote_closers=closers) == {}
    assert closers == set()
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell")


@pytest.mark.parametrize("escaped", ["\\\n", "\\\r\n", "\\x"])
def test_invocation_runtime_checks_survive_escaped_offset_jumps(escaped: str) -> None:
    source = " " + escaped * 12000 + "value"
    checks = 0

    def cancelled() -> None:
        nonlocal checks
        checks += 1
        if checks == 3:
            raise TimeoutError("escaped word deadline")

    with pytest.raises(TimeoutError, match="escaped word deadline"):
        tm._next_shell_invocation_word(source, 1, cancelled)


def test_literal_command_proof_propagates_real_runtime_exceptions() -> None:
    source = "bash -c 'echo ready'\n"

    def cancelled() -> None:
        raise TimeoutError("caller deadline")

    with pytest.raises(TimeoutError, match="caller deadline"):
        tm._literal_shell_command_spans(source, 0, len(source), cancelled)


@pytest.mark.parametrize(
    "tail",
    [
        'cat <<END\n"unowned\n',
        'cat <<END\ncontinued\\\nEND\n',
        "cat <<'END-MARK'\nbody\nEND-MARK\n",
        'echo `date`\n',
        'echo "unfinished\n',
    ],
)
def test_unproven_source_cannot_publish_quote_closers(tail: str) -> None:
    closers: set[int] = set()
    source = 'cat <<<"$VALUE"\ncp "$(dirname "$SOURCE")/file" "$DEST"\n' + tail

    assert tm._shell_quoted_candidate_limits(source, lambda: None, quote_closers=closers) == {}
    assert closers == set()


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
        'echo "prefix $(date) suffix"\ncat <<E\n"unowned\n',
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
            baseline.setattr(tm, "_shell_quoted_candidate_limits", lambda *_, **__: {})
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
