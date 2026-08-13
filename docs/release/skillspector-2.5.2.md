# SkillSpector v2.5.2

Released: 2026-08-04

## Summary

This patch strengthens input-ingestion limits, adds MCP registry posture scanning, and improves supply-chain advisory accuracy. It also reduces static-analysis false positives, makes SC4 reporting more precise, and improves Windows cleanup reliability.

## Highlights

- Added bounded handling for remote URLs, ZIP archives, and Git repositories before oversized content can be ingested.
- Added MCP registry posture scanning.
- Uses exact Python lockfile versions when resolving OSV advisories and reports SC4 vulnerabilities only when verified.

## Added

- MCP registry posture scanning.

## Changed

- Pinned dependencies used by workflows and Docker base images for more reproducible builds.
- Linked the Verified Skills pipeline and hosted documentation.

## Fixed

- Rejects oversized remote, archive, and repository inputs safely and cleans up temporary files when an ingest is rejected.
- Reduces false positives from benign instructional prose, Markdown tables, quote syntax, and valid OMS signatures.
- Resolves bundled metadata for supported NVIDIA Build endpoint IDs instead of using the generic token-budget fallback.
- Cleans up Windows temporary working directories without turning a successful batch into a failure.

## Security

- Enforces bounded URL download, archive extraction, and Git repository ingestion paths to reduce resource-exhaustion risk.
- Uses lockfile-resolved Python versions for OSV matching and avoids reporting unverified vulnerabilities in SC4 output.

## Breaking Changes and Migration

- None.

## Deprecations

- None.

## Validation

- `uv run --locked --extra dev pytest tests/unit/test_input_handler_bounds.py tests/unit/test_input_handler_ssrf.py` — 34 passed.
- `make test-unit` — passed for each merged import validation run.
- `uv run --locked --extra dev make test-ci` — passed on the corrected release source.

## Known Limitations

- NVIDIA Build metadata remains intentionally limited to owner-confirmed endpoint IDs; short-form aliases and unproven mappings continue to use the existing fallback behavior.

## References

- `CHANGELOG.md`
