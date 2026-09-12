# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the shared AST cache across analyzer consumers."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

import skillspector.nodes.build_context as build_context_module
import skillspector.python_ast as python_ast
from skillspector.artifacts import ArtifactDisposition, ContentKind, decode_text
from skillspector.graph import graph
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import (
    behavioral_ast,
    behavioral_taint_tracking,
    static_patterns_data_exfiltration,
    static_patterns_output_handling,
    static_patterns_tool_misuse,
)
from skillspector.nodes.build_context import build_context
from skillspector.python_ast import (
    ParsedPythonFile,
    PythonSourceClassification,
    classify_python_source,
    get_python_ast,
)


@pytest.mark.parametrize(
    ("filename", "prefix"),
    [
        pytest.param("script.py", "", id="py"),
        pytest.param("script.pyw", "", id="pyw"),
        pytest.param("script", "#!/usr/bin/env python3\n", id="env-shebang"),
    ],
)
def test_preparsed_python_is_reused_by_all_ast_analyzers(
    tmp_path, monkeypatch, filename: str, prefix: str
) -> None:
    """One scan parses each eligible Python file once before analyzer fan-out."""
    (tmp_path / filename).write_text(
        prefix + "import subprocess\n"
        "use_shell = True\n"
        "subprocess.run(output, shell=use_shell)\n"
        "import os\n"
        "payload = input()\n"
        "environment = os.environ.copy()\n"
        "exec(payload)\n",
        encoding="utf-8",
    )
    original_parse = python_ast.ast.parse
    parse_calls = 0

    def count_parse(*args, **kwargs):
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(*args, **kwargs)

    monkeypatch.setattr(python_ast.ast, "parse", count_parse)
    state = build_context({"skill_path": str(tmp_path)})

    metadata = next(item for item in state["component_metadata"] if item["path"] == filename)
    assert metadata["type"] == "python"
    assert metadata["executable"] is True

    python_ast_cache_key = state["python_ast_cache_key"]
    assert isinstance(python_ast_cache_key, str)
    parsed = get_python_ast(
        python_ast_cache_key,
        state["file_cache"][filename],
        filename,
    )
    assert isinstance(parsed, ParsedPythonFile)
    assert parsed.is_parseable
    assert parse_calls == 1

    data_findings = static_patterns_data_exfiltration.node(state)["findings"]
    output_findings = static_patterns_output_handling.node(state)["findings"]
    tool_misuse_findings = static_patterns_tool_misuse.node(state)["findings"]
    ast_findings = behavioral_ast.node(state)["findings"]
    taint_findings = behavioral_taint_tracking.node(state)["findings"]

    assert any(finding.rule_id == "E2" for finding in data_findings)
    assert any(finding.rule_id == "OH1" for finding in output_findings)
    assert any(finding.rule_id == "TM1" for finding in tool_misuse_findings)
    assert any(finding.rule_id == "AST1" for finding in ast_findings)
    assert any(finding.rule_id == "TT5" for finding in taint_findings)
    assert parse_calls == 1


def test_analyzers_reuse_build_context_python_classification(tmp_path, monkeypatch) -> None:
    """Analyzer fan-out does not repeat bounded shebang parsing per family."""
    filename = "runner"
    (tmp_path / filename).write_text(
        "#!/usr/bin/env python3\n"
        "import subprocess\n"
        "enabled = True\n"
        "subprocess.run(command, shell=enabled)\n"
        "payload = input()\n"
        "exec(payload)\n",
        encoding="utf-8",
    )
    state = build_context({"skill_path": str(tmp_path)})

    def forbidden_reclassification(*_args, **_kwargs):
        raise AssertionError("downstream analyzers must reuse build-context classification")

    monkeypatch.setattr(python_ast, "classify_python_source", forbidden_reclassification)

    assert any(
        finding.rule_id == "TM1" for finding in static_patterns_tool_misuse.node(state)["findings"]
    )
    assert any(finding.rule_id == "AST1" for finding in behavioral_ast.node(state)["findings"])
    assert any(
        finding.rule_id == "TT5" for finding in behavioral_taint_tracking.node(state)["findings"]
    )


def test_ambiguous_python_source_is_analyzed_once_and_marks_coverage_partial(
    tmp_path, monkeypatch
) -> None:
    """Runtime-dependent Python intent is scanned without claiming definitive coverage."""
    filename = "runner"
    (tmp_path / filename).write_text(
        "#!/usr/bin/env -S ${SKILLSPECTOR_INTERPRETER}\n"
        "import subprocess\n"
        "use_shell = True\n"
        "subprocess.run(output, shell=use_shell)\n"
        "import os\n"
        "payload = input()\n"
        "environment = os.environ.copy()\n"
        "exec(payload)\n",
        encoding="utf-8",
    )
    original_parse = python_ast.ast.parse
    parse_calls = 0

    def count_parse(*args, **kwargs):
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(*args, **kwargs)

    monkeypatch.setattr(python_ast.ast, "parse", count_parse)
    state = build_context({"skill_path": str(tmp_path)})

    metadata = next(item for item in state["component_metadata"] if item["path"] == filename)
    artifact = next(item for item in state["artifact_inventory"] if item["path"] == filename)
    classification_event = next(
        item
        for item in state["inspection_ledger"]
        if item.get("reason_code") is LedgerReason.PYTHON_SOURCE_AMBIGUOUS
    )
    assert metadata["type"] == "other"
    assert metadata["executable"] is True
    assert artifact["disposition"] is ArtifactDisposition.PARTIAL
    assert artifact["reason"] == LedgerReason.PYTHON_SOURCE_AMBIGUOUS.value
    assert classification_event["outcome"] is LedgerOutcome.PARTIAL

    tool_response = static_patterns_tool_misuse.node(state)
    ast_response = behavioral_ast.node(state)
    taint_response = behavioral_taint_tracking.node(state)
    assert any(finding.rule_id == "TM1" for finding in tool_response["findings"])
    assert any(finding.rule_id == "AST1" for finding in ast_response["findings"])
    assert any(finding.rule_id == "TT5" for finding in taint_response["findings"])
    for response in (tool_response, ast_response, taint_response):
        event = next(item for item in response["inspection_ledger"] if item["path"] == filename)
        assert event["outcome"] is LedgerOutcome.PARTIAL
        assert event["reason_code"] is LedgerReason.PYTHON_SOURCE_AMBIGUOUS
    assert parse_calls == 1


def test_raw_shebang_bytes_control_shared_python_classification(tmp_path) -> None:
    """Exact PEP 263 decoding retains raw ambiguity while enabling analysis."""
    filename = "runner"
    raw_line = b"#!/usr/bin/env -S X=" + b"\xff" + b"a" * 96 + b" python3 --"
    raw = (
        raw_line
        + b"\n# coding: latin-1\n"
        + b"import subprocess\n"
        + b"enabled = True\n"
        + b"subprocess.run(command, shell=enabled)\n"
    )
    (tmp_path / filename).write_bytes(raw)

    assert len(raw_line) == 128
    assert raw_line.find(b"python3") == 118
    assert classify_python_source(filename, raw) is PythonSourceClassification.AMBIGUOUS
    assert classify_python_source(filename, decode_text(raw)) is PythonSourceClassification.PYTHON

    state = build_context({"skill_path": str(tmp_path)})
    artifact = next(item for item in state["artifact_inventory"] if item["path"] == filename)
    assert state["raw_file_cache"][filename] == raw
    assert state["python_source_classifications"][filename] == "ambiguous"
    assert artifact["disposition"] is ArtifactDisposition.PARTIAL
    assert artifact["reason"] == LedgerReason.PYTHON_SOURCE_AMBIGUOUS.value
    assert "ÿ" in state["local_file_cache"][filename]
    assert "\ufffd" not in state["local_file_cache"][filename]

    tool_response = static_patterns_tool_misuse.node(state)
    assert any(finding.rule_id == "TM1" for finding in tool_response["findings"])
    event = next(item for item in tool_response["inspection_ledger"] if item["path"] == filename)
    assert event["outcome"] is LedgerOutcome.PARTIAL
    assert event["reason_code"] is LedgerReason.PYTHON_SOURCE_AMBIGUOUS


def test_pep263_python_source_is_strictly_decoded_and_analyzed(tmp_path) -> None:
    filename = "runner"
    raw = (
        b"#!/usr/bin/env python3\n"
        b"# coding: latin-1\n"
        b"# " + b"\xff" * 1_000 + b"\nimport subprocess\n"
        b"enabled = True\n"
        b"subprocess.run(command, shell=enabled)\n"
    )
    (tmp_path / filename).write_bytes(raw)

    state = build_context({"skill_path": str(tmp_path)})
    artifact = next(item for item in state["artifact_inventory"] if item["path"] == filename)

    assert state["raw_file_cache"][filename] == raw
    assert "ÿ" * 1_000 in state["local_file_cache"][filename]
    assert "\ufffd" not in state["local_file_cache"][filename]
    assert state["python_source_classifications"][filename] == "python"
    assert artifact["content_kind"] is ContentKind.TEXT
    assert artifact["disposition"] is ArtifactDisposition.ANALYZED
    assert artifact["decodable"] is True
    assert any(
        finding.rule_id == "TM1" for finding in static_patterns_tool_misuse.node(state)["findings"]
    )


@pytest.mark.parametrize(
    ("filename", "raw", "decoded_marker"),
    [
        pytest.param(
            "runner",
            b"#!/usr/bin/env python3\n"
            b"# coding: latin-1\n"
            b"import subprocess\n"
            b"activ\xe9 = True\n"
            b"subprocess.run(command, shell=activ\xe9)\n",
            "activé",
            id="heuristic-text-latin1",
        ),
        pytest.param(
            "runner.py",
            b"\xef\xbb\xbfimport subprocess\n"
            b"enabled = True\n"
            b"subprocess.run(command, shell=enabled)\n",
            "import subprocess",
            id="utf8-bom",
        ),
    ],
)
def test_python_exact_decode_replaces_every_lossy_text_projection(
    tmp_path, filename: str, raw: bytes, decoded_marker: str
) -> None:
    (tmp_path / filename).write_bytes(raw)

    state = build_context({"skill_path": str(tmp_path)})
    content = state["local_file_cache"][filename]
    artifact = next(item for item in state["artifact_inventory"] if item["path"] == filename)

    assert decoded_marker in content
    assert "\ufffd" not in content
    assert not content.startswith("\ufeff")
    assert state["file_cache"][filename] == content
    assert artifact["content_kind"] is ContentKind.TEXT
    assert artifact["decodable"] is True
    assert any(
        finding.rule_id == "TM1" for finding in static_patterns_tool_misuse.node(state)["findings"]
    )


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(
            b"#!/usr/bin/env python3\n"
            b"# coding: definitely-unknown\n"
            b"import subprocess\n"
            b"enabled = True\n"
            b"subprocess.run(command, shell=enabled)\n",
            id="unknown-cookie",
        ),
        pytest.param(
            b"#!/usr/bin/env python3\n# coding: utf-8\n# \xff\nvalue = 1\n",
            id="invalid-declared-utf8",
        ),
        pytest.param(
            b"#!/usr/bin/env python3\nvalue = 'before\x00after'\n",
            id="nul-byte",
        ),
        pytest.param(
            b"#!/usr/bin/env python3\n# coding: raw_unicode_escape\nvalue = '\\ud800'\n",
            id="lone-surrogate-escape",
        ),
        pytest.param(
            b"#!/usr/bin/env python3\n# coding: utf-7\n# +2AA-\nvalue = 1\n",
            id="lone-surrogate-utf7",
        ),
        pytest.param(
            b"#!/usr/bin/env python3\n# coding: unicode_escape\nvalue = '\\x00'\n",
            id="decoder-produced-nul",
        ),
    ],
)
def test_python_decode_failure_is_never_out_of_scope_or_complete(tmp_path, raw: bytes) -> None:
    filename = "runner"
    (tmp_path / filename).write_bytes(raw)

    state = build_context({"skill_path": str(tmp_path)})
    artifact = next(item for item in state["artifact_inventory"] if item["path"] == filename)
    event = next(
        item
        for item in state["inspection_ledger"]
        if item.get("path") == filename and item.get("reason_code") == "python_source_decode_error"
    )

    assert state["raw_file_cache"][filename] == raw
    assert state["python_source_classifications"][filename] == "python"
    assert artifact["disposition"] is ArtifactDisposition.PARTIAL
    assert artifact["reason"] == "python_source_decode_error"
    assert event["outcome"] is LedgerOutcome.PARTIAL
    assert filename not in state["local_file_cache"]
    assert filename not in state["file_cache"]


def test_mixed_newline_python_decode_failure_is_partial(tmp_path) -> None:
    filename = "mixed.py"
    raw = b"\r\n\t#coding:utf_16be\rx=1\n"
    (tmp_path / filename).write_bytes(raw)

    state = build_context({"skill_path": str(tmp_path)})
    artifact = next(item for item in state["artifact_inventory"] if item["path"] == filename)

    assert state["raw_file_cache"][filename] == raw
    assert artifact["disposition"] is ArtifactDisposition.PARTIAL
    assert artifact["reason"] == LedgerReason.PYTHON_SOURCE_DECODE_ERROR.value
    assert filename not in state["local_file_cache"]
    assert filename not in state["file_cache"]
    assert any(
        item.get("path") == filename
        and item.get("outcome") is LedgerOutcome.PARTIAL
        and item.get("reason_code") is LedgerReason.PYTHON_SOURCE_DECODE_ERROR
        for item in state["inspection_ledger"]
    )


def test_crlf_python_source_is_normalized_before_exact_decode(tmp_path) -> None:
    filename = "encoded.py"
    raw = b"\t#coding=utf_16be\r\nx=1\r\n"
    expected = b"\t#coding=utf_16be\nx=1\n".decode("utf_16be")
    (tmp_path / filename).write_bytes(raw)

    state = build_context({"skill_path": str(tmp_path)})
    artifact = next(item for item in state["artifact_inventory"] if item["path"] == filename)
    parsed = get_python_ast(
        state["python_ast_cache_key"],
        state["file_cache"][filename],
        filename,
    )

    assert state["raw_file_cache"][filename] == raw
    assert state["local_file_cache"][filename] == expected
    assert state["file_cache"][filename] == expected
    assert artifact["disposition"] is ArtifactDisposition.ANALYZED
    assert artifact["content_kind"] is ContentKind.TEXT
    assert parsed is not None and parsed.is_parseable
    assert not any(
        item.get("path") == filename
        and item.get("reason_code") is LedgerReason.PYTHON_SOURCE_DECODE_ERROR
        for item in state["inspection_ledger"]
    )


def test_classification_deadline_withholds_unclassified_python_from_analyzers(
    tmp_path, monkeypatch
) -> None:
    """A deadline suffix cannot fall back to the lossy generic text cache."""

    class FakeClock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

    fake_clock = FakeClock()
    (tmp_path / "SKILL.md").write_text("# Skill\n", encoding="utf-8")
    (tmp_path / "middle").write_text(
        "#!/usr/bin/env python3\npass\n",
        encoding="utf-8",
    )
    invalid_path = "zbad.py"
    (tmp_path / invalid_path).write_bytes(
        b"# coding: definitely-unknown\n"
        b"import subprocess\n"
        b"enabled = True\n"
        b"subprocess.run(command, shell=enabled)\n"
    )
    original_classify = build_context_module.classify_python_source

    def expiring_classification(path: str, content: str | bytes | None) -> object:
        result = original_classify(path, content)
        if path == "middle":
            fake_clock.now = 1.0
        return result

    monkeypatch.setattr(build_context_module, "MAX_BUNDLE_CACHE_SECONDS", 1.0)
    monkeypatch.setattr(build_context_module, "monotonic", fake_clock)
    monkeypatch.setattr(
        build_context_module,
        "classify_python_source",
        expiring_classification,
    )

    state = build_context({"skill_path": str(tmp_path)})
    responses = (
        static_patterns_tool_misuse.node(state),
        behavioral_ast.node(state),
        behavioral_taint_tracking.node(state),
    )

    assert invalid_path not in state["python_source_classifications"]
    assert state["python_source_classification_limitations"][invalid_path] == "runtime_limit"
    assert invalid_path in state["components"]
    assert invalid_path in state["raw_file_cache"]
    assert invalid_path not in state["local_file_cache"]
    assert invalid_path not in state["file_cache"]
    assert invalid_path not in state["llm_file_cache"]
    assert invalid_path not in state["llm_components"]
    assert not any(
        finding.file == invalid_path for response in responses for finding in response["findings"]
    )
    for response in responses:
        event = next(item for item in response["inspection_ledger"] if item["path"] == invalid_path)
        assert event["outcome"] is LedgerOutcome.PARTIAL
        assert event["reason_code"] is LedgerReason.RUNTIME_LIMIT


def test_uppercase_python_path_reuses_preparsed_ast_for_static_analyzers(
    tmp_path, monkeypatch
) -> None:
    """Static Python inference and cache eligibility use the same case handling."""
    (tmp_path / "script.PY").write_text(
        "import os\nimport subprocess\nos.environ.copy()\nsubprocess.run(output)\n",
        encoding="utf-8",
    )
    original_parse = python_ast.ast.parse
    parse_calls = 0

    def count_parse(*args, **kwargs):
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(*args, **kwargs)

    monkeypatch.setattr(python_ast.ast, "parse", count_parse)
    state = build_context({"skill_path": str(tmp_path)})

    data_findings = static_patterns_data_exfiltration.node(state)["findings"]
    output_findings = static_patterns_output_handling.node(state)["findings"]

    assert any(finding.rule_id == "E2" for finding in data_findings)
    assert any(finding.rule_id == "OH1" for finding in output_findings)
    assert parse_calls == 1


def test_graph_scan_parses_python_once_before_parallel_analyzers(tmp_path, monkeypatch) -> None:
    """The runtime cache shares one parse across the graph's analyzer fan-out."""
    (tmp_path / "script.py").write_text(
        "import os\n"
        "import subprocess\n"
        "payload = input()\n"
        "environment = os.environ.copy()\n"
        "subprocess.run(output)\n"
        "exec(payload)\n",
        encoding="utf-8",
    )
    original_parse = python_ast.ast.parse
    parse_calls = 0

    def count_parse(*args, **kwargs):
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(*args, **kwargs)

    monkeypatch.setattr(python_ast.ast, "parse", count_parse)

    result = graph.invoke({"skill_path": str(tmp_path), "use_llm": False})

    assert {"E2", "OH1", "AST1", "TT5"} <= {finding.rule_id for finding in result["findings"]}
    assert parse_calls == 1
    assert JsonPlusSerializer().dumps_typed(result)
