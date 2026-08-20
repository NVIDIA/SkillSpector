# SkillSpector v2.5.1

Released: 2026-07-30

## Summary

SkillSpector v2.5.1 lets users tune the concurrency of asynchronous LLM analyzer batches with an environment variable. This helps rate-limited providers avoid request bursts while retaining the existing default behavior and explicit per-call overrides. It also adds release-preparation scripts and documentation for publishing SkillSpector packages to PyPI.

## Highlights

- Set `SKILLSPECTOR_MAX_LLM_CONCURRENCY=1` to serialize asynchronous LLM analyzer requests for a rate-limited provider.

## Added

- `SKILLSPECTOR_MAX_LLM_CONCURRENCY` configures the default asynchronous LLM batch concurrency; blank or invalid values retain the default of 10, and values below 1 clamp to 1.

## Changed

- `LLMAnalyzerBase.arun_batches` now resolves its default concurrency from the environment while an explicit `max_concurrency` argument continues to take precedence.
- Release tooling now includes scripts and internal guidance to prepare and validate PyPI package releases.

## Fixed

- None.

## Security

- None.

## Breaking Changes and Migration

- None.

## Deprecations

- None.

## Validation

- `uv run pytest tests/nodes/test_llm_analyzer_base.py` — 121 passed.
- `uv run pytest -q -m 'not integration and not provider' tests` — completed with an empty failure cache.
- `uv run make lint` — passed.
- `uv run make format-check` — passed.
- `uv run python release.py --version patch --user keshavp@nvidia.com --release-notes-filepath docs/release/skillspector-2.5.1.md --dry-run` — validated the 2.5.1 release plan and notes.

## Known Limitations

- Provider-specific rate limits vary; choose a concurrency value appropriate for the configured provider.

## References

- `CHANGELOG.md`
- [GitHub PR #305](https://github.com/NVIDIA/SkillSpector/pull/305)
