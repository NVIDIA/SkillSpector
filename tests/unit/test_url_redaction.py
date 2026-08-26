# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Black-box contracts for bounded dependency-source URL redaction."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import pytest

from skillspector import url_redaction as api


def test_canonical_registry_url_has_pinned_safe_output() -> None:
    raw = (
        "https://alice:supersecret@packages.example.invalid/private"
        "?token=querysecret&channel=stable#fragmentsecret"
    )

    redacted = api.redact_url(raw)

    assert redacted == "https://packages.example.invalid/REDACTED_PATH"
    for sentinel in (
        "alice",
        "supersecret",
        "private",
        "querysecret",
        "fragmentsecret",
        "channel",
    ):
        assert sentinel not in redacted


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "http://user:secret@packages.example.invalid:8080/simple?channel=stable#part",
            "http://packages.example.invalid:8080/REDACTED_PATH",
        ),
        (
            "ssh://git:secret@git.example.invalid/org/repo.git?ref=main#readme",
            "ssh://git.example.invalid/REDACTED_PATH",
        ),
        (
            "git+https://git:secret@git.example.invalid/org/repo.git?ref=main",
            "git+https://git.example.invalid/REDACTED_PATH",
        ),
        (
            "sparse+https://user:secret@index.example.invalid/crates#metadata",
            "sparse+https://index.example.invalid/REDACTED_PATH",
        ),
        (
            "//user:secret@packages.example.invalid/private?channel=stable#part",
            "//packages.example.invalid/REDACTED_PATH",
        ),
        (
            "https://user:secret@[2001:db8::1]:8443/private?channel=stable",
            "https://[2001:db8::1]:8443/REDACTED_PATH",
        ),
        (
            "git-user@git.example.invalid:org/repo.git?ref=main#readme",
            "REDACTED@git.example.invalid:REDACTED_PATH",
        ),
    ],
)
def test_simple_urls_drop_query_fragment_and_userinfo_but_keep_safe_origin_path(
    raw: str,
    expected: str,
) -> None:
    assert api.redact_url(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://packages.example.invalid", "https://packages.example.invalid"),
        ("https://packages.example.invalid/", "https://packages.example.invalid/"),
        ("//packages.example.invalid/", "//packages.example.invalid/"),
    ],
)
def test_empty_or_root_paths_are_the_only_path_contents_retained(
    raw: str,
    expected: str,
) -> None:
    assert api.redact_url(raw) == expected


@pytest.mark.parametrize(
    "value",
    ["ordinary", "token=plain-secret", "src/a//b.py"],
)
def test_exact_value_redaction_rejects_non_url_destinations(value: str) -> None:
    assert api.redact_url(value) == api.REDACTED_URL


def test_exact_value_redaction_preserves_its_fixed_placeholder() -> None:
    assert api.redact_url(api.REDACTED_URL) == api.REDACTED_URL


@pytest.mark.parametrize(
    "raw",
    [
        "https://user%3Asecret%40packages.example.invalid/private",
        "https://packages.example.invalid/private%2Fsecret",
        "https%3A%2F%2Fuser%3Asecret%40packages.example.invalid%2Fprivate",
        "https://user:secret@packages.example.invalid/private?next=https://evil.invalid/x",
        "https://packages.example.invalid/private?next=user@evil.invalid:org/repo.git",
        "https://packages.example.invalid/private?next=marker@evil.invalid:repo",
        "https://packages.example.invalid/private#next=marker@evil.invalid:repo",
        "https://packages.example.invalid/private?next=marker%40evil.invalid:repo",
        "https://packages.example.invalid/private#next=marker%40evil.invalid:repo",
        "credential-marker%40host.invalid:repo",
        "https://one.invalid/x,https://two.invalid/y",
        "https://first:secret@second@packages.example.invalid/private",
        "https://packages.example.invalid:bad/private",
        "https://[not-an-ipv6-address]/private",
        "([https://user:secret@packages.example.invalid/private])",
        '{"url":"https://user:secret@packages.example.invalid/private"}',
    ],
)
def test_encoded_malformed_nested_or_mixed_candidates_are_whole_masked(raw: str) -> None:
    assert api.redact_url(raw) == api.REDACTED_URL
    assert "secret" not in api.redact_url(raw)


@pytest.mark.parametrize(
    "text",
    [
        "a//b",
        "// comment",
        "ordinary // comment",
        "path a//b remains ordinary",
        "email dev@example.invalid remains ordinary",
        "Unicode ☃ and punctuation stay byte-for-byte unchanged.",
    ],
)
def test_ordinary_no_match_text_is_unchanged(text: str) -> None:
    assert api.redact_text_result(text) == api.TextRedactionResult(
        value=text,
        complete=True,
        candidates=0,
        reason=None,
    )


def test_text_scanner_supports_only_one_simple_paired_prose_wrapper() -> None:
    raw = (
        "Use (https://user:secret@packages.example.invalid/private?channel=stable#part), "
        "not ([https://nested:secret@other.example.invalid/x])."
    )

    assert api.redact_text(raw) == (
        "Use (https://packages.example.invalid/REDACTED_PATH), not [REDACTED_URL]."
    )


def test_separate_whitespace_tokens_are_sanitized_independently() -> None:
    raw = (
        "mirror https://user:first-secret@one.example.invalid/x?token=one "
        "then git-user@two.example.invalid:org/repo.git#second-secret"
    )

    assert api.redact_text(raw) == (
        "mirror https://one.example.invalid/REDACTED_PATH "
        "then REDACTED@two.example.invalid:REDACTED_PATH"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "source credential-marker@host.invalid:repo",
            "source REDACTED@host.invalid:REDACTED_PATH",
        ),
        (
            "source credential-marker%40host.invalid:repo",
            "source [REDACTED_URL]",
        ),
    ],
)
def test_text_discovers_single_component_and_encoded_scp_candidates(
    raw: str,
    expected: str,
) -> None:
    assert api.redact_text(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "registry=//user:scheme-relative-secret@host.invalid/private?token=hidden",
            api.REDACTED_URL,
        ),
        (
            '{"registry":"//user:scheme-relative-secret@host.invalid/private?token=hidden"}',
            api.REDACTED_URL,
        ),
        (
            '{"registry": "//user:scheme-relative-secret@host.invalid/private?token=hidden"}',
            f'{{"registry": {api.REDACTED_URL}',
        ),
        (
            'src="//user:scheme-relative-secret@host.invalid/private?token=hidden"',
            api.REDACTED_URL,
        ),
    ],
    ids=("assignment", "compact-json", "spaced-json", "source-markup"),
)
def test_embedded_scheme_relative_candidates_are_whole_masked(raw: str, expected: str) -> None:
    result = api.redact_text_result(raw)

    assert result == api.TextRedactionResult(
        value=expected,
        complete=True,
        candidates=1,
        reason=None,
    )
    assert "scheme-relative-secret" not in result.value


@pytest.mark.parametrize(
    "raw",
    [
        "<url>//user:round2-scheme-relative-secret@host.invalid/round2-private-path</url>",
        "url(//user:round2-scheme-relative-secret@host.invalid/round2-private-path)",
    ],
    ids=("element-markup", "functional-markup"),
)
def test_markup_embedded_scheme_relative_candidates_are_whole_masked(raw: str) -> None:
    result = api.redact_text_result(raw)

    assert result == api.TextRedactionResult(
        value=api.REDACTED_URL,
        complete=True,
        candidates=1,
        reason=None,
    )
    assert "round2-scheme-relative-secret" not in result.value
    assert "round2-private-path" not in result.value


class _SyntheticMatch:
    def end(self) -> int:
        return len("https://")


class _CountingMarkerPattern:
    def __init__(self) -> None:
        self.visits = 0

    def finditer(self, _value: str) -> Iterator[_SyntheticMatch]:
        for _index in range(10_000):
            self.visits += 1
            yield _SyntheticMatch()


def test_dense_marker_count_stops_at_remaining_candidate_budget_plus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pattern = _CountingMarkerPattern()
    monkeypatch.setattr(api, "_HIERARCHICAL_MARKER", pattern)

    result = api.redact_text_result("https://host.invalid/path", max_candidates=1)

    assert result.complete is False
    assert result.reason is api.TextRedactionIncompleteReason.CANDIDATE_LIMIT
    assert result.candidates == 0
    assert pattern.visits == 2


def test_multiple_candidates_in_one_token_are_whole_masked_or_exhaust_the_remainder() -> None:
    raw = "https://one.invalid/x,https://two.invalid/y"

    assert api.redact_text_result(raw, max_candidates=2) == api.TextRedactionResult(
        value=api.REDACTED_URL,
        complete=True,
        candidates=2,
        reason=None,
    )
    assert api.redact_text_result(raw, max_candidates=1) == api.TextRedactionResult(
        value=api.REDACTED_REMAINDER,
        complete=False,
        candidates=0,
        reason=api.TextRedactionIncompleteReason.CANDIDATE_LIMIT,
    )


def test_candidate_budget_is_aggregate_and_exact_limit_succeeds() -> None:
    raw = "https://user:first-secret@one.invalid/x https://user:second-secret@two.invalid/y"

    assert api.redact_text_result(raw, max_candidates=2) == api.TextRedactionResult(
        value="https://one.invalid/REDACTED_PATH https://two.invalid/REDACTED_PATH",
        complete=True,
        candidates=2,
        reason=None,
    )
    one_over = api.redact_text_result(raw, max_candidates=1)
    assert one_over.value == api.REDACTED_REMAINDER
    assert one_over.complete is False
    assert one_over.candidates == 1
    assert one_over.reason is api.TextRedactionIncompleteReason.CANDIDATE_LIMIT


def test_structured_result_distinguishes_literal_placeholder_from_real_exhaustion() -> None:
    literal = api.redact_text_result(api.REDACTED_REMAINDER)
    exhausted = api.redact_text_result(
        "prefix https://user:secret@host.invalid/path",
        max_candidates=0,
    )

    assert literal.value == exhausted.value
    assert literal.complete is True
    assert literal.reason is None
    assert exhausted.complete is False
    assert exhausted.reason is api.TextRedactionIncompleteReason.CANDIDATE_LIMIT


def test_character_overflow_masks_the_whole_input_without_parsing_a_prefix() -> None:
    raw = "https://user:prefix-secret@packages.example.invalid/private"

    assert api.redact_text_result(raw, max_characters=len(raw) - 1) == api.TextRedactionResult(
        value=api.REDACTED_REMAINDER,
        complete=False,
        candidates=0,
        reason=api.TextRedactionIncompleteReason.CHARACTER_LIMIT,
    )
    assert api.redact_url(raw, max_characters=len(raw) - 1) == api.REDACTED_URL


@pytest.mark.parametrize("invalid", [-1, True, False])
def test_text_redaction_rejects_negative_and_boolean_bounds(invalid: int | bool) -> None:
    result = api.redact_text_result("https://host.invalid/path", max_candidates=invalid)

    assert result.complete is False
    assert result.reason is api.TextRedactionIncompleteReason.INVALID_INPUT


def test_default_text_bound_covers_the_full_artifact_cache_contract() -> None:
    text = "x" * api.MAX_REDACTION_CHARACTERS

    result = api.redact_text_result(text)

    assert api.MAX_REDACTION_CHARACTERS == 16 * 1024 * 1024
    assert result.value is text
    assert result.complete is True


def test_unexpected_url_parser_errors_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_parser(_value: str) -> object:
        raise RuntimeError("attacker-controlled parser failure")

    monkeypatch.setattr(api, "urlsplit", broken_parser)

    assert api.redact_url("https://user:secret@host.invalid/path") == api.REDACTED_URL


class _BrokenCasefoldString(str):
    def casefold(self) -> str:
        raise RuntimeError("attacker-controlled string failure")


def test_internal_text_probe_errors_mask_the_whole_input_without_throwing() -> None:
    raw = _BrokenCasefoldString("https%3A%2F%2Fuser%40host.invalid%2Fpath")

    assert api.redact_text_result(raw) == api.TextRedactionResult(
        value=api.REDACTED_REMAINDER,
        complete=False,
        candidates=0,
        reason=api.TextRedactionIncompleteReason.INTERNAL_ERROR,
    )


def test_nested_values_preserve_code_owned_keys_and_container_types() -> None:
    value = api.CodeOwnedMapping(
        {
            "registry_url": "https://user:https-secret@packages.example.invalid/private?token=x",
            "details": [
                "ssh://user:ssh-secret@git.example.invalid/org/repo.git#part",
                ("ordinary", 7),
            ],
            "enabled": True,
        }
    )

    redacted = api.redact_value(value)

    assert redacted == api.CodeOwnedMapping(
        {
            "registry_url": "https://packages.example.invalid/REDACTED_PATH",
            "details": ["ssh://git.example.invalid/REDACTED_PATH", ("ordinary", 7)],
            "enabled": True,
        }
    )


@pytest.mark.parametrize(
    "value",
    [
        {"credential_marker": "safe"},
        {"ordinary_field": "safe"},
    ],
)
def test_plain_mappings_fail_closed_even_when_keys_have_identifier_shape(
    value: dict[str, str],
) -> None:
    assert api.redact_value(value) == api.REDACTED_VALUE


@pytest.mark.parametrize(
    "key",
    [
        "https://user:secret@host.invalid/path",
        "token=plain-secret",
        "not code owned",
        "évidence",
        7,
        "a" * 256,
    ],
)
def test_wrapped_mapping_rejects_invalid_or_oversized_keys(key: object) -> None:
    assert api.redact_value(api.CodeOwnedMapping({key: "ordinary"})) == api.REDACTED_VALUE


def test_mapping_keys_share_the_aggregate_character_budget_with_values() -> None:
    value = api.CodeOwnedMapping({"field": "x"})

    assert api.redact_value(value, max_text_characters=6) == value
    assert api.redact_value(value, max_text_characters=5) == api.REDACTED_VALUE


def test_recursive_text_character_and_candidate_budgets_are_aggregate() -> None:
    benign = api.CodeOwnedMapping({"first": "abcd", "second": "efgh"})
    candidates = api.CodeOwnedMapping(
        {
            "first": "https://user:first-secret@one.invalid/x",
            "second": "https://user:second-secret@two.invalid/y",
        }
    )

    assert api.redact_value(benign, max_text_characters=19) == benign
    assert api.redact_value(benign, max_text_characters=18) == api.REDACTED_VALUE
    assert api.redact_value(candidates, max_text_candidates=1) == api.REDACTED_VALUE
    exact = api.redact_value(candidates, max_text_candidates=2)
    assert exact == api.CodeOwnedMapping(
        {
            "first": "https://one.invalid/REDACTED_PATH",
            "second": "https://two.invalid/REDACTED_PATH",
        }
    )


def test_recursive_depth_and_node_bounds_are_exact_and_fail_closed_one_over() -> None:
    depth_value = api.CodeOwnedMapping({"outer": api.CodeOwnedMapping({"leaf": "ordinary"})})
    node_value = api.CodeOwnedMapping({"leaf": "ordinary"})

    assert api.redact_value(depth_value, max_depth=2) == depth_value
    assert api.redact_value(depth_value, max_depth=1) == api.REDACTED_VALUE
    assert api.redact_value(node_value, max_nodes=2) == node_value
    assert api.redact_value(node_value, max_nodes=1) == api.REDACTED_VALUE


class _CountingMapping(Mapping[str, str]):
    def __init__(self) -> None:
        self.iterations = 0

    def __getitem__(self, key: str) -> str:
        return {"first": "one", "second": "two", "third": "three"}[key]

    def __iter__(self) -> Iterator[str]:
        for key in ("first", "second", "third"):
            self.iterations += 1
            yield key

    def __len__(self) -> int:
        return 3


def test_recursive_node_exhaustion_stops_before_iterating_an_oversized_mapping() -> None:
    value = _CountingMapping()

    assert api.redact_value(value, max_nodes=2) == api.REDACTED_VALUE
    assert value.iterations == 0


def test_recursive_self_reference_terminates_fail_closed() -> None:
    value: list[object] = []
    value.append(value)

    assert api.redact_value(value) == api.REDACTED_VALUE


@pytest.mark.parametrize("function_name", ["redact_url", "redact_text"])
def test_string_redactors_are_deterministic_and_idempotent(function_name: str) -> None:
    function = getattr(api, function_name)
    raw = (
        "Prefix " if function_name == "redact_text" else ""
    ) + "https://user:secret@packages.example.invalid/repo?token=query#part"

    first = function(raw)

    assert function(raw) == first
    assert function(first) == first


def test_recursive_value_redaction_is_idempotent() -> None:
    value = api.CodeOwnedMapping(
        {
            "url": "https://user:secret@packages.example.invalid/repo?token=x",
            "items": ("plain",),
        }
    )

    first = api.redact_value(value)

    assert api.redact_value(value) == first
    assert api.redact_value(first) == first
