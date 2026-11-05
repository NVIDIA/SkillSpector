# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinator ordering, independent failures, policy flags and secure writes."""

import json
import weakref
from pathlib import Path
from unittest.mock import Mock

import pytest

from skillspector import file_output
from skillspector.file_output import write_text_no_follow
from skillspector.report_completeness import build_report_completeness
from skillspector.report_emission import ReportEmissionResult, emit_report_outputs
from skillspector.report_outputs import (
    FormatChoice,
    ReportOutput,
    build_report_outputs,
    parse_report_declarations,
)


def rendered(format):
    return {
        "report_body": f"{format.value} body",
        "report_completeness": build_report_completeness(),
    }


@pytest.mark.parametrize("primary_file", [False, True])
def test_plan_order_and_primary_callback_owns_output(primary_file):
    entries = build_report_outputs(
        FormatChoice.json,
        Path("primary.json") if primary_file else None,
        parse_report_declarations(["sarif=first.sarif", "json=second.json", "markdown=last.md"]),
    )
    events = []
    diagnostics = []

    def render(format):
        events.append(("render", format))
        return rendered(format)

    def primary(report, entry):
        assert entry is entries[0]
        events.append(("primary", entry.path, report["report_body"]))

    def write(path, body):
        events.append(("additional", path, body))

    result = emit_report_outputs(entries, render, primary, write, diagnostics.append)

    assert result == ReportEmissionResult()
    assert events == [
        ("render", FormatChoice.json),
        ("primary", entries[0].path, "json body"),
        ("render", FormatChoice.sarif),
        ("additional", "first.sarif", "sarif body"),
        ("render", FormatChoice.json),
        ("additional", "second.json", "json body"),
        ("render", FormatChoice.markdown),
        ("additional", "last.md", "markdown body"),
    ]
    assert len(diagnostics) == 3  # No replacement of legacy primary success messages.
    for message, entry in zip(diagnostics, entries[1:], strict=True):
        assert entry.format.value in message
        assert entry.path in message


@pytest.mark.skipif(
    not file_output._SECURE_OUTPUT_SUPPORTED, reason="secure file output unavailable"
)
@pytest.mark.parametrize("existing", [False, True])
def test_sarif_exception_leaves_destination_untouched_and_other_formats_succeed(tmp_path, existing):
    paths = [tmp_path / name for name in ("primary.json", "bad.sarif", "later.json", "last.md")]
    if existing:
        paths[1].write_text("previous SARIF", encoding="utf-8")
    before = paths[1].stat() if existing else None
    entries = build_report_outputs(
        FormatChoice.json,
        paths[0],
        parse_report_declarations(
            [f"sarif={paths[1]}", f"json={paths[2]}", f"markdown={paths[3]}"]
        ),
    )
    calls = []
    diagnostics = []

    def render(format):
        calls.append(format)
        if format == FormatChoice.sarif:
            raise RuntimeError("SECRET provider data\n" + "x" * 10000)
        assert not paths[1].exists() or paths[1].read_text() == "previous SARIF"
        return rendered(format)

    result = emit_report_outputs(
        entries,
        render,
        lambda report, entry: write_text_no_follow(entry.path, report["report_body"]),
        diagnostic=diagnostics.append,
    )

    assert result == ReportEmissionResult(output_failed=True)
    assert calls == [
        FormatChoice.json,
        FormatChoice.sarif,
        FormatChoice.json,
        FormatChoice.markdown,
    ]
    assert paths[0].read_text() == paths[2].read_text() == "json body"
    assert paths[3].read_text() == "markdown body"
    if existing:
        assert paths[1].stat() == before
        assert paths[1].read_text() == "previous SARIF"
    else:
        assert not paths[1].exists()
    assert "Failed to render sarif" in diagnostics[0]
    assert all("SECRET" not in message for message in diagnostics)
    assert not list(tmp_path.glob(".skillspector-output-*"))


@pytest.mark.parametrize("failure_stage", ["render", "primary", "additional"])
def test_ordinary_failures_attempt_all_later_entries(failure_stage):
    entries = build_report_outputs(
        FormatChoice.sarif,
        None,
        parse_report_declarations(["json=first.json", "markdown=last.md"]),
    )
    render = Mock(side_effect=rendered)
    primary = Mock()
    write = Mock()
    diagnostic = Mock()
    if failure_stage == "render":
        render.side_effect = [
            RuntimeError("render failure"),
            rendered(FormatChoice.json),
            rendered(FormatChoice.markdown),
        ]
    elif failure_stage == "primary":
        primary.side_effect = BrokenPipeError("stdout failure")
    else:
        write.side_effect = [OSError("write failure"), None]

    result = emit_report_outputs(entries, render, primary, write, diagnostic)

    assert result.output_failed
    assert not result.report_incomplete
    assert render.call_count == 3
    assert primary.call_count == (0 if failure_stage == "render" else 1)
    assert write.call_count == 2
    assert write.call_args.args == ("last.md", "markdown body")
    assert diagnostic.call_count == (2 if failure_stage == "additional" else 3)


@pytest.mark.skipif(
    not file_output._SECURE_OUTPUT_SUPPORTED, reason="secure file output unavailable"
)
def test_real_write_failure_preserves_earlier_and_existing_files_and_continues(
    tmp_path, monkeypatch
):
    paths = [tmp_path / name for name in ("first.json", "bad.json", "last.md")]
    for path in paths:
        path.write_text("previous", encoding="utf-8")
    before = paths[1].stat()
    original_replace = file_output.os.replace

    def replace(source, destination, **kwargs):
        if destination == paths[1].name:
            raise PermissionError("replacement failed")
        return original_replace(source, destination, **kwargs)

    monkeypatch.setattr(file_output.os, "replace", replace)
    entries = parse_report_declarations(
        [f"json={paths[0]}", f"json={paths[1]}", f"markdown={paths[2]}"]
    )
    result = emit_report_outputs(entries, rendered, Mock(), diagnostic=Mock())

    assert result == ReportEmissionResult(output_failed=True)
    assert paths[0].read_text() == "json body"
    assert paths[1].read_text() == "previous"
    assert paths[1].stat() == before
    assert paths[2].read_text() == "markdown body"
    assert not list(tmp_path.glob(".skillspector-output-*"))


@pytest.mark.parametrize("omitted_index", [0, 1, 2])
@pytest.mark.parametrize(
    "projection",
    [
        {"is_complete": False},
        {"omission_reasons": ["output_limit"]},
        {"omitted_records": 1},
        {"omitted_skills": 1},
    ],
)
@pytest.mark.parametrize("write_failure", [False, True])
def test_flags_include_any_report_omission_even_when_write_fails(
    omitted_index, projection, write_failure
):
    entries = build_report_outputs(
        FormatChoice.json, None, parse_report_declarations(["sarif=extra", "markdown=last"])
    )
    reports = [rendered(entry.format) for entry in entries]
    reports[omitted_index]["report_completeness"] = projection.copy()
    reports[0]["analysis_completeness"] = {"is_complete": True}
    render = Mock(side_effect=reports)
    write = Mock(side_effect=[OSError("failed"), None] if write_failure else None)
    result = emit_report_outputs(entries, render, Mock(), write, Mock())

    assert result == ReportEmissionResult(output_failed=write_failure, report_incomplete=True)
    assert render.call_count == 3
    assert write.call_count == 2
    assert reports[omitted_index]["report_completeness"] == projection
    assert reports[0]["analysis_completeness"] == {"is_complete": True}


def test_analysis_incompleteness_is_not_report_incompleteness():
    report = rendered(FormatChoice.json)
    report["analysis_completeness"] = {"is_complete": False}
    result = emit_report_outputs(
        build_report_outputs(FormatChoice.json, None, []),
        Mock(return_value=report),
        Mock(),
        diagnostic=Mock(),
    )
    assert result == ReportEmissionResult()


def test_repeated_formats_use_snapshot_without_retaining_rendered_bodies():
    class Body(str):
        pass

    analyze = Mock(return_value={"findings": ("one",)})
    snapshot = analyze()
    previous = None
    count = 0

    def render(format):
        nonlocal previous, count
        assert previous is None or previous() is None
        assert snapshot["findings"] == ("one",)
        body = Body(f"{format.value} {count}")
        previous = weakref.ref(body)
        count += 1
        return {"report_body": body, "report_completeness": build_report_completeness()}

    entries = build_report_outputs(
        FormatChoice.json,
        None,
        parse_report_declarations(["json=first", "json=second"]),
    )
    writes = []
    result = emit_report_outputs(
        entries,
        render,
        lambda report, entry: None,
        lambda path, body: writes.append((path, str(body))),
        Mock(),
    )

    assert result == ReportEmissionResult()
    assert count == 3
    analyze.assert_called_once()
    assert writes == [("first", "json 1"), ("second", "json 2")]
    assert previous() is None


@pytest.mark.parametrize("stage", ["render", "primary", "additional", "diagnostic"])
@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, BaseException])
def test_interrupts_and_other_base_exceptions_propagate(stage, exception):
    entries = build_report_outputs(
        FormatChoice.json, None, parse_report_declarations(["sarif=first", "markdown=last"])
    )
    render = Mock(side_effect=rendered)
    primary, write, diagnostic = Mock(), Mock(), Mock()
    callbacks = {
        "render": render,
        "primary": primary,
        "additional": write,
        "diagnostic": diagnostic,
    }
    interruption = exception("stop")
    callbacks[stage].side_effect = interruption

    with pytest.raises(exception) as caught:
        emit_report_outputs(entries, render, primary, write, diagnostic)

    assert caught.value is interruption
    assert render.call_count == (1 if stage in {"render", "primary"} else 2)
    assert write.call_count == (0 if stage in {"render", "primary"} else 1)


def test_default_diagnostics_are_bounded_literal_stderr_without_body_or_exception(capsys):
    hostile_path = "[red]path\n\x1b[31m\r\u202e" + "x" * 5000
    entries = build_report_outputs(
        FormatChoice.json, None, [ReportOutput(FormatChoice.sarif, hostile_path)]
    )

    def primary(report, entry):
        print(report["report_body"])

    def render(format):
        if format == FormatChoice.sarif:
            raise ValueError("SECRET body or provider error")
        return {"report_body": json.dumps({"ok": True})}

    result = emit_report_outputs(entries, render, primary)

    streams = capsys.readouterr()
    assert json.loads(streams.out) == {"ok": True}
    assert result.output_failed
    assert streams.err.count("\n") == 1
    assert "\\n" in streams.err and "\\x1b" in streams.err and "\\u202e" in streams.err
    assert "\x1b" not in streams.err and "\r" not in streams.err
    assert "sarif" in streams.err and "[red]path" in streams.err
    assert "SECRET" not in streams.err
    assert len(streams.err) < 1200


def test_diagnostic_exception_does_not_block_remaining_entries():
    render = Mock(side_effect=rendered)
    write = Mock()
    result = emit_report_outputs(
        parse_report_declarations(["json=first", "markdown=last"]),
        render,
        Mock(),
        write,
        Mock(side_effect=RuntimeError("diagnostic failed")),
    )
    assert result == ReportEmissionResult(output_failed=True)
    assert render.call_count == write.call_count == 2


@pytest.mark.parametrize(
    "bad_report", [{}, {"report_body": None}, {"report_body": iter(["partial"])}]
)
def test_invalid_rendered_body_never_reaches_any_writer(bad_report):
    entries = build_report_outputs(
        FormatChoice.json, None, parse_report_declarations(["json=bad", "markdown=last"])
    )
    primary, write = Mock(), Mock()
    result = emit_report_outputs(
        entries,
        Mock(side_effect=[bad_report, bad_report, rendered(FormatChoice.markdown)]),
        primary,
        write,
        Mock(),
    )
    assert result.output_failed
    primary.assert_not_called()
    write.assert_called_once_with("last", "markdown body")


def test_empty_plan_has_no_side_effects():
    render, primary, write, diagnostic = Mock(), Mock(), Mock(), Mock()
    assert emit_report_outputs([], render, primary, write, diagnostic) == ReportEmissionResult()
    for callback in (render, primary, write, diagnostic):
        callback.assert_not_called()
