# SkillSpector v2.11.2

Released: 2026-09-10

## Summary

SkillSpector 2.11.2 fixes fatal reference-accounting errors and several false static-parser limits triggered by ordinary documentation. This patch also preserves incomplete-analysis reporting when a runtime-selected executable prevents exact command reconstruction.

## Highlights

- Complete reference accounting when Markdown labels and destinations identify the same artifact, or when several referenced artifacts appear on one source line.
- Avoid false parser limits for simple runtime parameters, inline skill invocations, PowerShell member access, and long quoted prose.
- Keep runtime-selected `printf` and wrapper paths marked as partially inspected.

## Added

- None.

## Changed

- Record reference-coverage completion once per source line.

## Fixed

- Deduplicate reference-coverage records for the same source file, line, and target, preventing fatal `unaccounted_work` errors from duplicate Markdown references.
- Account for distinct reference targets on the same source line without creating conflicting completion records.
- Distinguish simple runtime parameters from command substitutions and complex parameter expansions in bounded shell reconstruction, including inline `$ARGUMENTS` documentation ([#464](https://github.com/NVIDIA/SkillSpector/issues/464)).
- Count unquoted characters separately from already-consumed quoted spans so long quoted prose does not cause a false command-word span limit.
- Preserve partial coverage when runtime parameters select a `printf`, `command`, `builtin`, or `env` executable path; a recognized basename alone cannot establish which executable will run.

## Security

- Fixed security findings.

## Breaking Changes and Migration

- None. No new configuration is required.

## Deprecations

- None.

## Validation

Validated locally with Python 3.12 and uv 0.10.10:

- `uv lock --check` — passed; third-party dependency versions are unchanged.
- `uv run --no-sync make test-ci` — 4,013 passed, 14 skipped, 38 deselected, and 4 expected failures.
- `uv run --no-sync make lint` and `uv run --no-sync make format-check` — passed.
- Built wheel and source distributions; `twine check` passed for both artifacts.
- `skillspector --version` — reported `SkillSpector v2.11.2`.
- The GitHub release helper dry run resolved `v2.11.2` and the matching versioned release notes.
- Docker image build and repository smoke tests passed on Linux/arm64, including the local safe fixture and public GitHub repository scans.
- `git diff --check` — passed.

## Known Limitations

- Full LLM analysis and downstream CI behavior require validation in the deployment that uses the release.

## References

- [GitHub PR #507](https://github.com/NVIDIA/SkillSpector/pull/507)
- [GitHub PR #508](https://github.com/NVIDIA/SkillSpector/pull/508)
