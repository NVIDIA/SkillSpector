# SkillSpector v2.11.1

Released: 2026-09-07

## Summary

SkillSpector 2.11.1 raises the default aggregate scan deadline from 60 seconds to 600 seconds and makes it configurable, preventing larger valid scans from timing out under the previous one-minute workflow budget. This patch release also includes security-analysis correctness fixes merged since 2.11.0.

## Highlights

- Give direct, recursive, transitive, and multi-skill scans a 600-second aggregate workflow deadline by default.
- Allow operators to set a positive finite deadline with `SKILLSPECTOR_MAX_WORKFLOW_SECONDS` while retaining the safe default for invalid values.
- Preserve security finding classifications during scan-view and report deduplication, and strengthen detection of concealed instructions.

## Added

- Add `SKILLSPECTOR_MAX_WORKFLOW_SECONDS` as an optional environment setting for the aggregate workflow deadline.

## Changed

- Increase the default end-to-end workflow and transitive traversal deadline from 60 seconds to 600 seconds.
- Apply the configured deadline consistently across direct CLI, recursive, transitive, and multi-skill analysis paths.
- Enforce `SKILLSPECTOR_MAX_LLM_CONCURRENCY` across all concurrently running LLM analyzers instead of separately within each analyzer.

## Fixed

- Prevent premature workflow termination for scans that legitimately need more than one minute.
- Parse whitespace-separated `allowed-tools` declarations without producing least-privilege false positives.
- Normalize bounded concealed-instruction text and fail closed when inter-character obfuscation prevents complete interpretation.
- Keep safe and unsafe findings distinct when they share a rule fingerprint so deduplication cannot discard or misclassify security evidence.

## Security

- Retain classification and bounded evidence in finding identity across raw, normalized, continuity, report, JSON, and SARIF projections.
- Detect security-relevant instructions concealed with default-ignorable characters or bounded inter-character separators while preserving benign multilingual, punctuation, URL, email, table, and code controls.
- Bound total in-flight LLM requests across analyzers sharing an event loop and configured limit, improving behavior with rate-limited providers.

## Breaking Changes and Migration

- None. Existing users automatically receive the 600-second default.
- Deployments that require a different aggregate deadline can set `SKILLSPECTOR_MAX_WORKFLOW_SECONDS` to a positive finite number of seconds. Invalid, zero, negative, infinite, and NaN values retain the 600-second default.

## Deprecations

- None.

## Validation

- `uv lock --check` — passed.
- `uv run --no-sync pytest -q` — 3,985 passed, 14 skipped, 38 deselected, and 4 expected failures.
- `uv run --no-sync ruff check src/ tests/ scripts/` — passed.
- `uv run --no-sync ruff format --check src/ tests/ scripts/` — 201 files already formatted.
- Built wheel and source distributions; `twine check` passed for both artifacts.
- `skillspector --version` — reported `SkillSpector v2.11.1`.
- The GitHub release helper dry run resolved tag `v2.11.1` and the matching versioned release notes.
- `git diff --check` — passed.

## Known Limitations

- The deadline is an aggregate ceiling, not a per-analyzer allowance. All work in a direct or recursive scan shares the same configured budget.
- Changing `SKILLSPECTOR_MAX_WORKFLOW_SECONDS` requires starting a new SkillSpector process because the setting is resolved when the workflow state module is imported.
- Provider-specific request timeouts and deterministic byte, artifact, and analyzer ceilings remain independently enforced.

## References

- [GitHub PR #330](https://github.com/NVIDIA/SkillSpector/pull/330)
- [GitHub PR #401](https://github.com/NVIDIA/SkillSpector/pull/401)
- [GitHub PR #408](https://github.com/NVIDIA/SkillSpector/pull/408)
- [GitHub PR #462](https://github.com/NVIDIA/SkillSpector/pull/462)
- [GitHub PR #468](https://github.com/NVIDIA/SkillSpector/pull/468)
