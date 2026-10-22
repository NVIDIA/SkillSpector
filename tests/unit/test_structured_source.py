# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the isolated phase-2 source-ownership core.

These are NOT the upstream SkillSpector graph/report/CLI regression suite.
No fixture string is executed, and no network or API credentials are used.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import FrozenInstanceError
from hashlib import sha256

import pytest

from skillspector.structured_source import (
    SourceLimits,
    _SourceParser,
    index_structured_source,
)


def program(protocol="AISP V1.0.0"):
    return [
        {
            "role": "system",
            "content": {
                "protocol": protocol,
                "axiom_0": "Human_Sovereignty_and_Wellbeing",
                "id": "mapping_fixture_aisp",
                "name": "Synthetic source mapping fixture",
                "version": "1.0.0",
                "license": "Apache-2.0",
                "flow_format": "mermaid",
                "loading_mode": "node",
                "tools": ["shell"],
                "system_prompt": "FIXTURE_PROMPT",
            },
        },
        {
            "role": "user",
            "content": {
                "instruction": "RUN aisop.main",
                "aisp_contract": {
                    "profile": "aisp.skill.v1",
                    "invocation": {
                        "mode": "manual_only",
                        "when_to_use": ["Synthetic source-only mapping tests."],
                        "when_not_to_use": ["Running this fixture."],
                    },
                    "non_negotiable": [
                        {
                            "rule": "The fixture is source text; no test executes it.",
                            "enforced_by": "aisop.main",
                        }
                    ],
                    "risk_level": "medium",
                    "resources": [
                        {
                            "id": "fixture_reference",
                            "path": "references/example.txt",
                            "kind": "reference",
                            "mode": "read_only",
                            "scope": "skill",
                            "requires_tools": ["filesystem"],
                        }
                    ],
                },
                "aisop": {"main": "graph TD\n    inspect[Inspect] --> end((End))"},
                "functions": {
                    "inspect": {
                        "step1": "FIXTURE_EXECUTION",
                        "constraints": ["FIXTURE_CONSTRAINT"],
                        "execute_mode": "inline",
                    },
                    "end": {
                        "step1": "Return without executing external operations.",
                        "execute_mode": "inline",
                    },
                },
            },
        },
    ]


def raw_program(value=None, **kwargs):
    return json.dumps(program() if value is None else value, ensure_ascii=False, **kwargs).encode(
        "utf-8"
    )


def mapping(raw, needle, nth=0, **kwargs):
    text = raw.decode("utf-8")
    positions = [m.start() for m in re.finditer(re.escape(needle), text)]
    start = positions[nth]
    index = index_structured_source(raw, **kwargs)
    assert index.status == "ready", index.reason
    annotation = index.locate(
        start,
        start + len(needle),
        content_sha256=index.content_sha256,
        coordinate_space="original_unicode",
    )
    return annotation, index


@pytest.mark.parametrize("indent", [None, 0, 2, 4, "\t"])
@pytest.mark.parametrize("protocol", ["AISOP V1.0.0", "AISP V1.0.0"])
def test_execution_ownership_across_serializations(indent, protocol):
    raw = raw_program(program(protocol), indent=indent)
    a, index = mapping(raw, "FIXTURE_EXECUTION")
    assert a.text_role == "executable_step"
    assert a.structured_source == "/1/content/functions/inspect/step1"
    assert a.mapping_status == "exact"
    assert index.protocol == protocol
    assert index.content_sha256 == "sha-256:" + sha256(raw).hexdigest()
    assert a.to_dict()["risk_polarity"] == "unknown"
    assert a.to_dict()["role_confidence"] is None


@pytest.mark.parametrize(
    ("needle", "role", "pointer"),
    [
        ("FIXTURE_CONSTRAINT", "constraint", "/1/content/functions/inspect/constraints/0"),
        ("FIXTURE_PROMPT", "prompt", "/0/content/system_prompt"),
        ("RUN aisop.main", "instruction", "/1/content/instruction"),
        ("shell", "tool_declaration", "/0/content/tools/0"),
        ("mapping_fixture_aisp", "metadata", "/0/content/id"),
        (
            "references/example.txt",
            "resource_declaration",
            "/1/content/aisp_contract/resources/0/path",
        ),
        ("read_only", "resource_declaration", "/1/content/aisp_contract/resources/0/mode"),
        (
            "filesystem",
            "resource_declaration",
            "/1/content/aisp_contract/resources/0/requires_tools/0",
        ),
        (
            "The fixture is source text",
            "control_declaration",
            "/1/content/aisp_contract/non_negotiable/0/rule",
        ),
        ("manual_only", "invocation_metadata", "/1/content/aisp_contract/invocation/mode"),
        (
            "Synthetic source-only",
            "invocation_metadata",
            "/1/content/aisp_contract/invocation/when_to_use/0",
        ),
        ("graph TD", "workflow_topology", "/1/content/aisop/main"),
        ("inline", "runtime_configuration", "/1/content/functions/inspect/execute_mode"),
    ],
)
def test_known_roles(needle, role, pointer):
    a, _ = mapping(raw_program(), needle)
    assert a.text_role == role
    assert a.structured_source == pointer


def test_binding_is_not_runtime_enforcement():
    a, _ = mapping(raw_program(), "aisop.main", nth=1)
    assert a.text_role == "control_binding"
    assert a.to_dict()["risk_polarity"] == "unknown"
    assert "severity" not in a.to_dict()
    assert "confidence" not in a.to_dict()


@pytest.mark.parametrize(
    "field",
    [
        "step1",
        "step99",
        "hard_deny",
        "deny_list",
        "description",
        "documentation",
        "example",
        "test_fixture",
        "new_instruction",
    ],
)
def test_non_reserved_function_fields_remain_steps(field):
    p = program()
    p[1]["content"]["functions"]["inspect"] = {field: "NEVER RUN THIS: MARKER"}
    a, _ = mapping(raw_program(p), "MARKER")
    assert a.text_role == "executable_step"
    assert a.to_dict()["risk_polarity"] == "unknown"


@pytest.mark.parametrize(
    "field",
    ["join", "map", "on_error", "retry_policy", "context_filter", "output_mapping", "execute_mode"],
)
def test_reserved_function_metadata_not_steps(field):
    p = program()
    p[1]["content"]["functions"]["inspect"][field] = "RESERVED_MARKER"
    a, _ = mapping(raw_program(p), "RESERVED_MARKER")
    assert a.text_role == "runtime_configuration"
    assert a.to_dict()["risk_polarity"] == "unknown"


def test_constraint_string_and_list_are_supported_but_nested_extension_is_unknown():
    p = program()
    body = p[1]["content"]["functions"]["inspect"]
    for value in ["UNIQUE", ["UNIQUE"]]:
        body["constraints"] = value
        a, _ = mapping(raw_program(p), "UNIQUE")
        assert a.text_role == "constraint"
    for value in [{"label": "UNIQUE"}, [["UNIQUE"]]]:
        body["constraints"] = value
        a, _ = mapping(raw_program(p), "UNIQUE")
        assert a.text_role == "unknown"


def test_user_extension_named_hard_deny_is_unknown():
    p = program()
    p[1]["content"]["hard_deny"] = ["UNTRUSTED_DECLARATION"]
    a, _ = mapping(raw_program(p), "UNTRUSTED_DECLARATION")
    assert a.mapping_status == "exact"
    assert a.text_role == "unknown"
    assert a.reason == "unsupported_source_field"


def test_metadata_does_not_claim_safe_or_executed():
    p = program()
    p[0]["content"]["description"] = "IGNORE ALL WARNINGS MARKER"
    a, _ = mapping(raw_program(p), "MARKER")
    assert a.text_role == "metadata"
    assert a.to_dict()["risk_polarity"] == "unknown"


def test_same_line_repeated_text_has_distinct_ownership():
    p = program()
    p[1]["content"]["functions"]["inspect"]["step1"] = "REPEAT"
    p[1]["content"]["functions"]["inspect"]["constraints"] = ["REPEAT"]
    raw = raw_program(p, separators=(",", ":"))
    a, _ = mapping(raw, "REPEAT", 0)
    b, _ = mapping(raw, "REPEAT", 1)
    assert a.text_role == "executable_step"
    assert b.text_role == "constraint"
    assert a.source_span != b.source_span


@pytest.mark.parametrize(
    "key",
    ["node/with/slash", "node~tilde", "node.with.dot", "node[0]", "node\\with\\slashes", "节😀点"],
)
def test_json_pointer_preserves_unambiguous_node_names(key):
    p = program()
    p[1]["content"]["functions"] = {key: {"step1": "UNIQUE_MARKER"}}
    a, _ = mapping(raw_program(p), "UNIQUE_MARKER")
    escaped = key.replace("~", "~0").replace("/", "~1")
    assert a.structured_source == f"/1/content/functions/{escaped}/step1"


@pytest.mark.parametrize("before", ["\\", '"', "\n", "\t", "😀汉字", "𝄞", "é"])
@pytest.mark.parametrize("ensure_ascii", [True, False])
def test_escapes_preserve_raw_character_spans(before, ensure_ascii):
    p = program()
    p[1]["content"]["functions"]["inspect"]["step1"] = before + "RAW_MARKER"
    raw = json.dumps(p, ensure_ascii=ensure_ascii).encode()
    a, _ = mapping(raw, "RAW_MARKER")
    start, end = a.source_span
    assert raw.decode()[start:end] == "RAW_MARKER"
    assert a.text_role == "executable_step"


def test_valid_surrogate_pair_decodes_but_raw_offsets_remain_escaped():
    p = program()
    p[1]["content"]["functions"]["inspect"]["step1"] = "😀"
    raw = json.dumps(p, ensure_ascii=True).encode()
    a, _ = mapping(raw, "\\ud83d\\ude00")
    assert a.text_role == "executable_step"


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_original_line_columns(newline):
    raw = raw_program(indent=2).replace(b"\n", newline.encode())
    a, index = mapping(raw, "FIXTURE_EXECUTION")
    start, end = a.source_span
    logical = re.compile(r"\r\n|[\r\n\v\f\x1c-\x1e\x85\u2028\u2029]")
    text = raw.decode()
    starts = [0, *(m.end() for m in logical.finditer(text))]
    line = max(i for i, x in enumerate(starts) if x <= start)
    b = index.locate_lines(
        line + 1,
        line + 1,
        start - starts[line],
        end - starts[line],
        content_sha256=index.content_sha256,
        coordinate_space="original_unicode",
    )
    assert b == a


@pytest.mark.parametrize("separator", ["\x85", "\u2028", "\u2029"])
def test_logical_line_breaks_inside_json_strings_follow_scanner_convention(separator):
    p = program()
    p[1]["content"]["functions"]["inspect"]["step1"] = "BEFORE" + separator + "AFTER"
    raw = raw_program(p)
    a, index = mapping(raw, "AFTER")
    assert len(index.line_starts) == 2
    b = index.locate_lines(
        2, 2, 0, 5, content_sha256=index.content_sha256, coordinate_space="original_unicode"
    )
    assert a == b
    # A span can cross a logical newline without leaving its JSON string.
    c, _ = mapping(raw, "BEFORE" + separator + "AFTER")
    assert c.mapping_status == "exact"


@pytest.mark.parametrize("coordinate_space", ["normalized", "decoded_json", "utf8_bytes", ""])
def test_transformed_coordinates_never_borrow_original_roles(coordinate_space):
    _, index = mapping(raw_program(), "FIXTURE_EXECUTION")
    a = index.locate(1, 2, content_sha256=index.content_sha256, coordinate_space=coordinate_space)
    assert a.mapping_status == "unknown"
    assert a.reason == "unbound_coordinates"


def test_same_readable_id_different_bytes_cannot_reuse_map():
    raw = raw_program()
    a, index = mapping(raw, "FIXTURE_EXECUTION")
    wrong_digest = "sha-256:" + sha256(raw + b" ").hexdigest()
    b = index.locate(
        *a.source_span, content_sha256=wrong_digest, coordinate_space="original_unicode"
    )
    assert b.reason == "source_mismatch"


@pytest.mark.parametrize(
    ("start", "end"),
    [(0, 0), (-1, 1), (5, 4), (0, 999999), (True, 5), (1.0, 5), ("1", 5), (1, None)],
)
def test_invalid_offsets_fail_closed(start, end):
    index = index_structured_source(raw_program())
    a = index.locate(
        start, end, content_sha256=index.content_sha256, coordinate_space="original_unicode"
    )
    assert a.reason == "invalid_span"


@pytest.mark.parametrize(
    ("sl", "el", "sc", "ec", "reason"),
    [
        (1, 1, None, None, "missing_columns"),
        (1, None, 0, None, "missing_columns"),
        (0, 1, 0, 5, "invalid_location"),
        (1, 999, 0, 5, "invalid_location"),
        (1, 1, -1, 5, "invalid_location"),
        (1, 1, 0, 999999, "invalid_location"),
        (True, 1, 0, 5, "invalid_location"),
        (1, 1, 0.0, 5, "invalid_location"),
    ],
)
def test_incomplete_or_invalid_locations_not_guessed(sl, el, sc, ec, reason):
    index = index_structured_source(raw_program())
    a = index.locate_lines(
        sl, el, sc, ec, content_sha256=index.content_sha256, coordinate_space="original_unicode"
    )
    assert a.reason == reason


def test_object_keys_and_between_values_not_assigned_a_role():
    raw = raw_program()
    text = raw.decode()
    index = index_structured_source(raw)
    starts = [text.index("step1"), text.index('"FIXTURE_EXECUTION"')]
    for start in starts:
        a = index.locate(
            start,
            start + 5,
            content_sha256=index.content_sha256,
            coordinate_space="original_unicode",
        )
        assert a.reason == "not_one_string_value"
    a = text.index("FIXTURE_EXECUTION")
    b = text.index("FIXTURE_CONSTRAINT") + len("FIXTURE_CONSTRAINT")
    r = index.locate(a, b, content_sha256=index.content_sha256, coordinate_space="original_unicode")
    assert r.mapping_status == "unknown"


@pytest.mark.parametrize("extra", [b",", b" garbage", b"{}", b"[]"])
def test_trailing_data_discards_all_previously_indexed_strings(extra):
    result = index_structured_source(raw_program() + extra)
    assert result.status == "unavailable"
    assert result.strings == ()
    assert result.reason == "trailing_data"


@pytest.mark.parametrize(
    "bad",
    [
        b"",
        b" ",
        b"[",
        b"{",
        b'{"x":}',
        b'{"x":1,}',
        b"[1,]",
        b"[1 2]",
        b'{"x" 1}',
        b"{'x':1}",
        b"{x:1}",
        b'{"x":01}',
        b'{"x":"bad\\q"}',
        b'{"x":"literal\nnewline"}',
        b"\xef\xbb\xbf[]",
    ],
)
def test_malformed_json_has_no_role_index(bad):
    result = index_structured_source(bad)
    assert result.status == "unavailable"
    assert result.strings == ()


@pytest.mark.parametrize(
    "snippet", [b'{"x":1,"x":2}', b'{"x":1,"\\u0078":2}', b'{"nested":{"x":"a","x":"b"}}']
)
def test_duplicate_decoded_keys_reject_whole_document(snippet):
    p = raw_program()
    raw = p[:-1] + b"," + snippet + b"]"
    result = index_structured_source(raw)
    assert result.reason == "duplicate_key"
    assert result.strings == ()


@pytest.mark.parametrize("bad", [b'{"x":"\\ud800"}', b'{"\\udfff":1}', b'{"x":"\\ud800x"}'])
def test_lone_surrogates_reject(bad):
    assert index_structured_source(bad).reason == "invalid_unicode"


@pytest.mark.parametrize("bad", [b"NaN", b"Infinity", b"-Infinity", b"1e9999"])
def test_nonfinite_number_rejected(bad):
    assert index_structured_source(b'{"x":' + bad + b"}").reason == "non_finite_number"


def test_invalid_utf8_and_input_type_are_unavailable():
    assert index_structured_source(b'"\xff"').reason == "invalid_utf8"
    assert index_structured_source("not bytes").reason == "expected_bytes"


@pytest.mark.parametrize(
    "kind", ["protocol", "roles", "root", "functions", "main", "instruction", "contract"]
)
def test_unsupported_layout_not_mistaken_for_validated_program(kind):
    p = program()
    if kind == "protocol":
        p[0]["content"]["protocol"] = "AISP V2.0.0"
    if kind == "roles":
        p.reverse()
    if kind == "root":
        p = {"program": p}
    if kind == "functions":
        p[1]["content"]["functions"] = []
    if kind == "main":
        p[1]["content"]["aisop"]["main"] = []
    if kind == "instruction":
        p[1]["content"]["instruction"] = 1
    if kind == "contract":
        del p[1]["content"]["aisp_contract"]
    r = index_structured_source(raw_program(p))
    assert r.reason == "unsupported_layout"
    assert not r.strings


def test_open_world_extensions_are_not_silently_classified():
    p = program()
    p[0]["content"]["future"] = {"label": "FUTURE_MARKER"}
    a, _ = mapping(raw_program(p), "FUTURE_MARKER")
    assert a.text_role == "unknown"


def test_aisop_does_not_inherit_aisp_contract_extension_semantics():
    raw = raw_program(program("AISOP V1.0.0"))
    a, _ = mapping(raw, "references/example.txt")
    assert a.text_role == "unknown"


def test_input_is_not_mutated_and_output_is_immutable():
    raw = raw_program()
    before = bytes(raw)
    a, index = mapping(raw, "FIXTURE_EXECUTION")
    with pytest.raises(FrozenInstanceError):
        index.reason = "changed"
    output = a.to_dict()
    output["risk_polarity"] = "safe"
    assert a.to_dict()["risk_polarity"] == "unknown"
    assert raw == before


@pytest.mark.parametrize(
    ("limits", "payload", "reason"),
    [
        (SourceLimits(max_bytes=10), b" " * 11, "size_limit"),
        (SourceLimits(max_depth=4), b"[[[[[0]]]]]", "depth_limit"),
        (SourceLimits(max_values=4), b"[1,2,3,4]", "value_limit"),
        (SourceLimits(max_strings=2), b'["a","b","c"]', "string_limit"),
        (SourceLimits(max_pointer_chars=4), b'{"too_long":"a"}', "pointer_limit"),
    ],
)
def test_budget_exhaustion_is_atomic(limits, payload, reason):
    result = index_structured_source(payload, limits=limits)
    assert result.reason == reason
    assert not result.strings
    assert (
        result.locate(0, 1, content_sha256="", coordinate_space="original_unicode").mapping_status
        == "unavailable"
    )


def test_extremely_deep_input_does_not_recurse_to_python_limit():
    raw = b"[" * 20000 + b"0" + b"]" * 20000
    r = index_structured_source(raw)
    assert r.reason == "depth_limit"


def test_oversized_unicode_source_is_bounded_in_bytes():
    raw = ('"' + "😀" * 65536 + '"').encode()
    assert len(raw) > 256 * 1024
    assert index_structured_source(raw).reason == "size_limit"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_depth", 0),
        ("max_depth", 65),
        ("max_depth", True),
        ("max_bytes", 262145),
        ("max_strings", 513),
        ("max_values", -1),
        ("max_seconds", float("inf")),
        ("max_seconds", float("nan")),
        ("max_seconds", 0),
        ("max_seconds", True),
        ("max_seconds", 2.1),
    ],
)
def test_callers_cannot_raise_core_ceiling(field, value):
    with pytest.raises(ValueError):
        SourceLimits(**{field: value})


def test_expired_shared_deadline_wins():
    result = index_structured_source(raw_program(), deadline=4.0, clock=lambda: 5.0)
    assert result.reason == "runtime_limit"


def test_local_deadline_cannot_be_extended_by_caller():
    values = iter([0.0, 3.0])
    r = index_structured_source(raw_program(), deadline=500, clock=lambda: next(values))
    assert r.reason == "runtime_limit"


def test_late_deadline_exhaustion_discards_index():
    ticks = iter(i / 1000 for i in range(100000))
    result = index_structured_source(
        raw_program(), limits=SourceLimits(max_seconds=0.02), clock=lambda: next(ticks)
    )
    assert result.reason == "runtime_limit"
    assert not result.strings


@pytest.mark.parametrize("error", [TimeoutError, RuntimeError, ValueError, KeyboardInterrupt])
def test_host_cancellation_is_not_swallowed_as_malformed_source(error):
    def cancel():
        raise error("host cancellation")

    with pytest.raises(error, match="host cancellation"):
        index_structured_source(raw_program(), check_runtime=cancel)


def test_indexer_does_not_read_files_or_execute_fixture(monkeypatch):
    def forbid(*_args, **_kwargs):
        raise AssertionError("forbidden I/O or execution")

    import builtins
    import os
    import subprocess

    monkeypatch.setattr(builtins, "open", forbid)
    monkeypatch.setattr(os, "system", forbid)
    monkeypatch.setattr(subprocess, "run", forbid)
    p = program()
    p[1]["content"]["functions"]["inspect"]["step1"] = "sys.run('NEVER_EXECUTE')"
    a, _ = mapping(raw_program(p), "NEVER_EXECUTE")
    assert a.text_role == "executable_step"


def test_numeric_conversion_failure_not_crash():
    raw = b'{"x":' + b"9" * 10000 + b"}"
    r = index_structured_source(raw)
    # Python 3.12+ default integer-string digit cap.
    assert r.reason == "scalar_conversion"


def test_reordered_fields_rebuild_correct_positions():
    p = program()
    raw1 = raw_program(p)
    raw2 = json.dumps(p, sort_keys=True, indent=4).encode()
    a, i = mapping(raw1, "FIXTURE_EXECUTION")
    b, j = mapping(raw2, "FIXTURE_EXECUTION")
    assert a.structured_source == b.structured_source
    assert i.content_sha256 != j.content_sha256
    assert a.source_span != b.source_span


def test_differential_parser_matches_stdlib_on_generated_json():
    rng = random.Random(130)
    atoms = [None, False, True, 0, -1, 1.5, "", '"', "\\", "汉字", "😀", "~x/y", "\n"]

    def generate(depth):
        if depth == 0 or rng.random() < 0.45:
            return rng.choice(atoms)
        if rng.random() < 0.5:
            return [generate(depth - 1) for _ in range(rng.randrange(5))]
        return {f"{i}~/{rng.randrange(99)}": generate(depth - 1) for i in range(rng.randrange(5))}

    for _ in range(300):
        payload = generate(5)
        text = json.dumps(
            payload, ensure_ascii=rng.choice([True, False]), indent=rng.choice([None, 2])
        )
        parser = _SourceParser(text, SourceLimits(), lambda: 0.0, 2.0, None)
        result = parser.value((), 0)
        parser.whitespace()
        assert result == json.loads(text)
        assert type(result) is type(json.loads(text))
        assert parser.position == len(text)
        for start, end, _path in parser.strings:
            assert isinstance(json.loads('"' + text[start:end] + '"'), str)


def test_malformed_mutation_smoke_never_crashes():
    rng = random.Random(211)
    original = raw_program().decode()
    for _ in range(300):
        pos = rng.randrange(len(original))
        text = original[:pos] + rng.choice(["{", '"', "]", ",", "\x00"]) + original[pos + 1 :]
        result = index_structured_source(text.encode())
        assert result.status in {"ready", "unavailable"}
        if result.status == "unavailable":
            assert not result.strings
