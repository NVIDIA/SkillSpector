# SkillSpector v2.9.3

Released: 2026-08-11

## Summary

This patch makes malformed structured LLM responses non-fatal during analysis. Affected analysis work is now recorded as skipped so reports clearly show degraded, incomplete results while preserving the remaining analysis output.

## Highlights

- Improve resilience to malformed structured responses from LLM-backed analyzers without masking the affected analysis outcome.

## Added

- None.

## Changed

- Analysis ledger and analyzer status handling consistently represent malformed structured-response batches as skipped and degraded rather than failed.

## Fixed

- Preserve the input findings and incomplete-analysis provenance when a malformed structured response exhausts retry handling.

## Security

- None.

## Breaking Changes and Migration

- None.

## Deprecations

- None.

## Validation

- `uv run --locked --extra dev pytest tests/nodes/test_llm_analyzer_base.py tests/nodes/test_meta_analyzer.py tests/nodes/test_finalize_inspection_ledger.py tests/test_inspection_ledger.py tests/test_mcp_tool_poisoning.py` — passed.

## Known Limitations

- Malformed structured responses remain unavailable for analysis; this release surfaces their impact as incomplete rather than producing findings for the affected work.

## References

- `CHANGELOG.md`
