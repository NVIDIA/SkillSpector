# SkillSpector v2.9.1

Released: 2026-08-10

## Summary

This patch release improves resilience to transient LLM-provider connection failures during analysis. It uses bounded retries and records clearer batch-failure reasons when retries cannot recover.

## Highlights

- Adds bounded retries for transient LLM provider connection failures while preserving per-batch failure reporting.

## Added

- The inspection ledger distinguishes malformed structured LLM responses from exhausted connection retries.

## Changed

- Supported OpenAI and Anthropic clients use a common bounded native retry budget, while other providers receive the same bounded fallback retry schedule when applicable.

## Fixed

- Transient LLM connection failures no longer terminate a batch before the configured retry budget is exhausted.

## Security

- None.

## Breaking Changes and Migration

- None.

## Deprecations

- None.

## Validation

- `uv run --locked --extra dev pytest tests/nodes/test_llm_analyzer_base.py tests/nodes/test_meta_analyzer.py` — passed.
- `GitLab main pipeline 61992428` — passed.
- `git diff --check release/2.9.0..68c7a026d4b2d574b63019ceacd8fe8d7caa35db` — passed.

## Known Limitations

- Retries are limited to transient LLM provider connection errors; other provider errors fail immediately.

## References

- `CHANGELOG.md`
