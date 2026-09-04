# SkillSpector v2.9.6

Released: 2026-08-18

## Summary

SkillSpector 2.9.6 improves PE3 credential-access accuracy for OAuth documentation. It now distinguishes compound-noun uses of `access token` and `access tokens` from instructions that access credentials, removing HIGH-severity false positives without weakening detection of actionable credential access.

## Highlights

- Stop flagging ordinary OAuth glossary entries, return-value descriptions, revocation behavior, storage guidance, and supported-token tables as credential-access findings.
- Preserve PE3 findings for imperative and modal access instructions, sensitive credential sources, and read, copy, send, post, leak, and exfiltration actions.

## Added

- Add benign and adversarial regression coverage for OAuth terminology in Markdown tables, headings, later clauses, URL navigation, action inflections, and HTTP POST token flows.

## Changed

- Classify noun-shaped `access token` terminology in documentation using bounded grammatical context instead of a narrow OAuth lifecycle allowlist.

## Fixed

- Prevent ordinary OAuth terminology in documentation from producing HIGH-severity PE3 findings while retaining findings for credential-access actions embedded in or adjacent to benign prose.

## Security

- Keep the OAuth documentation exception fail-closed when bounded context contains credential actions or sensitive sources.

## Breaking Changes and Migration

- None.

## Deprecations

- None.

## Validation

- `uv run pytest -q tests/unit/test_patterns.py tests/nodes/analyzers/test_binary_and_pe3_filtering.py tests/nodes/analyzers/test_static_runner_filtering.py` — 204 passed.
- `uv run pytest -q --ignore=tests/unit/test_input_handler.py --ignore=tests/unit/test_input_handler_ssrf.py` — 2,158 passed, 13 skipped, and 4 expected failures.
- `uv run ruff check src/ tests/` — passed.
- `uv run ruff format --check src/ tests/` — passed.

## Known Limitations

- The documentation exception is intentionally limited to Markdown and text files under recognized documentation paths; other file types and locations continue to use the stricter PE3 rule.

## References

- [GitHub PR #392](https://github.com/NVIDIA/SkillSpector/pull/392)
