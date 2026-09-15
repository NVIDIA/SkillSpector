# SkillSpector v2.11.3

Released: 2026-09-15

## Summary

SkillSpector 2.11.3 fixes false AE1 incomplete-analysis results caused by ordinary Markdown and JSON documentation. It preserves incomplete coverage for unresolved runtime commands, adds a configurable static analysis time allowance, and corrects reference, manifest, package-name, and recursive JSON reporting behavior.

## Highlights

- Recognize complete JSON strings and Markdown code spans in their document context, avoiding false analysis limits while retaining analysis of their contents.
- Keep delimiter pairing within Markdown blocks and table cells so unrelated documentation cannot hide unresolved runtime commands.
- Scan JSON quote candidates in linear time and retain cancellation handling.

## Added

- `SKILLSPECTOR_MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT` configures the static pattern and YARA time allowance per artifact; its default increases from 30 to 300 seconds. The remaining workflow deadline still bounds both analyzers.

## Changed

- Align provider setup guidance and update research background documentation.

## Fixed

- Avoid false AE1 results from valid JSON placeholders, inline code, list and blockquote containers, indented JSON, Markdown tables, and literal Make syntax ([#516](https://github.com/NVIDIA/SkillSpector/pull/516)).
- Bound JSON quote traversal without repeatedly scanning overlapping suffixes ([#521](https://github.com/NVIDIA/SkillSpector/pull/521)).
- Emit recursive JSON reports to standard output when no output path is provided ([#467](https://github.com/NVIDIA/SkillSpector/pull/467)).
- Use the project manifest version for RP3 analysis ([#474](https://github.com/NVIDIA/SkillSpector/pull/474)).
- Prefer exact known-package matches when evaluating SC6 package-name similarity ([#530](https://github.com/NVIDIA/SkillSpector/pull/530)).
- Avoid treating slash-separated prose as local file references ([#451](https://github.com/NVIDIA/SkillSpector/pull/451)).
- Skip symlink test cases when the platform refuses symlink creation ([#501](https://github.com/NVIDIA/SkillSpector/pull/501)).

## Security

- Genuine removal instructions remain reportable. Unresolved runtime commands retain incomplete coverage and fail strict CLI/MCP installation gates, including when semantic analysis succeeds.
- JSON string ownership preserves source evidence and does not exempt string contents from analysis.

## Breaking Changes and Migration

- No required configuration changes. Static analysis can now run longer within the existing workflow deadline. Set `SKILLSPECTOR_MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT=30` to retain the previous per-artifact allowance, and restart the SkillSpector process after changing the setting ([#522](https://github.com/NVIDIA/SkillSpector/pull/522)).
- Third-party dependency versions are unchanged from 2.11.2.

## Deprecations

- None.

## Validation

Validated locally with Python 3.12 and uv 0.10.10:

- Locked dependency verification, Ruff lint, and formatting checks passed.
- `make test-ci` passed: 4,825 passed, 14 skipped, 38 deselected, and 4 expected failures; 89% coverage.
- Wheel and source distributions built successfully and passed `twine check`. All 97 Python source files in the wheel match the release candidate.
- `skillspector --version` reports `SkillSpector v2.11.3`; the GitHub release helper dry run resolves the matching tag and release notes.
- A fresh Linux/arm64 Docker image passed both repository smoke tests: the local safe fixture and a public GitHub repository scan completed with 100% coverage and no findings.
- Eight targeted CLI and MCP-helper scan scenarios passed, including paired static/live documentation checks and unresolved-runtime-command controls. All 20 recorded LLM attempts with the actual `codex_cli` provider succeeded. Benign documentation completed without AE1; unresolved runtime commands retained incomplete coverage and failed strict installation gates as expected.

The release PR records the full regression-suite results and additional integration validation.

## Known Limitations

- Local sanity checks cover the tested inputs and environment; live provider and deployment behavior depend on their configuration.

## References

- [Changes since v2.11.2](https://github.com/NVIDIA/SkillSpector/compare/v2.11.2...v2.11.3)
- [AE1 documentation fix #516](https://github.com/NVIDIA/SkillSpector/pull/516)
- [Static analysis time allowance #522](https://github.com/NVIDIA/SkillSpector/pull/522)

Prepared by Codex for Mohit Gupta.
