# SkillSpector v2.5.0

Released: 2026-07-24

## Summary

SkillSpector 2.5.0 adds canonical inspection-ledger reporting so every scan can
show what was inspected, skipped, failed, or excluded. JSON-consuming security
automation can now distinguish normal policy findings from scans that did not
execute reliably.

## Highlights

- JSON and SARIF reports now include execution-completeness information,
  analyzer status, and safe explanations for skipped or failed work.
- JSON consumers can block incomplete or failed scans instead of treating a
  zero-finding report as a successful validation.

## Added

- Canonical inspection-ledger accounting across static and LLM analysis stages,
  including per-component coverage and explicit out-of-scope records.
- Execution-completeness fields in JSON and SARIF output so automation can
  distinguish a complete scan from a partial or failed one.
- The top-level `execution_successful` status and
  `analysis_completeness.ledger_exceptions` diagnostics in JSON output.

## Changed

- Recursive scans now return a failure when any child scan fails, and include
  the child status in the combined report.
- The CLI exits with code 2 for a fatal execution or accounting failure, even
  when a JSON report was produced.
- Baseline fingerprints use the version 2 format, binding accepted findings to
  the scanner version, source content, and full finding evidence.

## Fixed

- Tightened static-analysis filtering so documentation or code-example context
  cannot broadly suppress credential-access findings.
- Improved binary and large-file handling during analysis.
- CI validators can report the public completeness exceptions that explain a
  blocked execution failure.

## Security

- Version 1 baselines containing fingerprints are rejected instead of allowing
  stale or insufficiently specific suppressions. Regenerate and review a
  version 2 baseline after upgrading.
- Hardened analyzer and build-context processing against unsafe input handling
  while preserving auditable suppression records in SARIF output.

## Breaking Changes and Migration

- Baseline files with version 1 fingerprints are no longer accepted. Run
  `skillspector baseline <path>`, review the generated version 2 entries, and
  commit the replacement baseline; rules-only version 1 baselines remain
  supported with a warning.
- JSON integrations must treat invalid or missing output, a nonzero process
  failure, or `execution_successful: false` as a blocking validation error and
  surface `analysis_completeness.ledger_exceptions` for diagnosis. Continue to
  use HIGH or CRITICAL findings for ordinary security-policy failures.

## Deprecations

- None.

## Validation

- Required CI jobs: lint, test-unit, test-integration, docker-smoke, and
  sonar-scan — passed.

## Known Limitations

- None.

## References

- `CHANGELOG.md`
- `docs/SUPPRESSION.md`
