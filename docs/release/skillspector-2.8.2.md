# SkillSpector v2.8.2

Released: 2026-08-07

## Summary

This patch release improves the resilience of MCP tool-poisoning analysis when a model returns malformed structured output. Affected assessments are retried through the shared structured-output analyzer lifecycle rather than causing an immediate scan failure.

## Highlights

- MCP tool-poisoning TP4 checks now use the shared typed structured-output analyzer lifecycle.

## Added

- None.

## Changed

- Structured TP4 responses are validated through a typed schema with the same per-batch isolation behavior used by other LLM analyzers.

## Fixed

- Malformed structured model responses in MCP tool-poisoning checks are retried once and then isolated to the affected batch.

## Security

- None.

## Breaking Changes and Migration

- None.

## Deprecations

- None.

## Validation

- `git diff --check -- docs/release/skillspector-2.8.2.md` — passed.

## Known Limitations

- None.

## References

- `CHANGELOG.md`
