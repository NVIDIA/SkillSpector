# SkillSpector v2.9.2

Released: 2026-08-10

## Summary

This patch release makes structured LLM response handling more resilient to transient malformed payloads. It retries validation and structured-output parse failures with bounded backoff while retaining fail-closed batch isolation after the retry budget is exhausted.

## Highlights

- Structured LLM response failures now receive bounded retries before a batch is isolated.

## Added

- None.

## Changed

- Applied the structured-response retry policy centrally to semantic analyzers, the meta-analyzer, TP4, and gap-fill paths.

## Fixed

- Prevented transient malformed structured responses from immediately exhausting required LLM batches.

## Security

- None.

## Breaking Changes and Migration

- None.

## Deprecations

- None.

## Validation

- `git diff --check release/2.9.1..origin/main` — passed.
- Automated CI — lint, unit, integration, and Docker smoke checks passed; Sonar analysis succeeded.

## Known Limitations

- Requests still fail closed after the bounded retry budget is exhausted.

## References

- `CHANGELOG.md`
