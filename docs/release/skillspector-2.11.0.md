# SkillSpector v2.11.0

Released: 2026-08-28

## Summary

SkillSpector 2.11.0 expands supply-chain coverage to installed npm dependency versions, adds bounded analysis of bundled lifecycle hooks and project permission grants, and introduces optional LLM sampling controls. It also improves provider guidance and fallback routing, supports secure file traversal in restricted Linux environments, and removes two reported MP3/P6 false positives without weakening directive detection.

## Highlights

- Resolve exact direct and transitive npm versions from `package-lock.json` and `npm-shrinkwrap.json` before vulnerability analysis.
- Add BH1–BH3 findings for bundled lifecycle hooks, directly proven remote transfer of sensitive content, and broad or ignored project permission modes.
- Add `SKILLSPECTOR_TEMPERATURE` and `SKILLSPECTOR_SEED` controls while preserving provider defaults when they are unset.
- Avoid MP3 and P6 findings for the reported nominal state-coverage and CSS print-rule descriptions while retaining actionable reset and disclosure directives.

## Added

- Parse npm lockfile versions 1, 2, and 3 under the existing dependency-analysis resource bounds, including nested installs and multiple installed versions of the same package.
- Analyze `hooks/hooks.json`, `.claude/settings.json`, and `.claude/settings.local.json` for bundled hook execution (BH1), directly proven remote transfer of sensitive event or file content (BH2), and broad permission surfaces (BH3).
- Add optional `SKILLSPECTOR_TEMPERATURE` validation for hosted providers and optional `SKILLSPECTOR_SEED` forwarding for OpenAI-compatible and Azure OpenAI endpoints.

## Changed

- Expand `skillspector scan --help` to list all supported hosted, local, compatible, and CLI-backed LLM providers with their authentication paths.
- Resolve configured model defaults from the provider that will actually build the chat model when OpenAI credentials satisfy the fallback path.

## Fixed

- Traverse intermediate path components with `O_PATH` where available so restricted Linux sandboxes do not require read access to every ancestor; final-file and no-symlink protections remain unchanged.
- Suppress only the bounded nominal grammar reported for `initial/reset state` and `descendant/compound/print rules are NOT evaluated`, including the reported line wrapping.
- Preserve MP3 and P6 detection for imperatives, agent-scoped instructions, anaphoric follow-ups, mixed benign/malicious content, and unrecognized surrounding grammar.

## Security

- Scan exact npm lockfile versions, including transitive and non-hoisted copies, instead of relying only on manifest ranges.
- Report conditional bundled hook reach, closed evidence of sensitive remote transfer, and declared project permission surfaces without executing bundled content.
- Preserve descriptor-relative, no-follow file opening while allowing safe traversal through search-only ancestor permissions on supported Linux systems.
- Keep the new nominal-phrase exclusions match-local and fail closed when directive framing or referential continuation is present.

## Breaking Changes and Migration

- No CLI command, option, provider, or report field was removed.
- Existing scans may report new BH1–BH3 findings for supported bundled hook and settings files. Review those findings and use the existing baseline mechanism only after validating the declared execution or permission surface.
- `SKILLSPECTOR_TEMPERATURE` accepts values from `0` through `1`, and `SKILLSPECTOR_SEED` accepts integers. Leave either variable unset or blank to preserve provider defaults.

## Deprecations

- None.

## Validation

- `uv lock --check` — passed.
- `.venv/bin/pytest -q tests/unit/test_create_github_release.py tests/unit/test_github_release_workflow.py tests/unit/test_wheel_contents.py` — 11 passed.
- Targeted regressions for bundled execution surfaces, npm lockfiles, provider routing and sampling, secure input traversal, CLI help, reporting, and MP3/P6 contextual handling — 1,070 passed and 10 skipped.
- `.venv/bin/ruff check src/ tests/ scripts/` — passed.
- `.venv/bin/ruff format --check src/ tests/ scripts/` — 199 files already formatted.
- Built `skillspector-2.11.0-py3-none-any.whl` and `skillspector-2.11.0.tar.gz`; `twine check` passed for both distributions.
- `skillspector --version` — reported `SkillSpector v2.11.0`.
- The GitHub release helper dry run resolved tag `v2.11.0` and the matching versioned release notes.
- `git diff --check` — passed.

## Known Limitations

- npm lockfile resolution covers `package-lock.json` and `npm-shrinkwrap.json`; Yarn and pnpm lockfiles are not included in this release.
- Bundled execution-surface analysis is limited to the supported exact configuration paths and does not execute hooks. Findings distinguish declarations that require conditional activation from permission modes ignored by the supported surface.
- Seed support is provider- and model-dependent and is forwarded only to OpenAI-compatible and Azure OpenAI endpoints.
- The MP3/P6 nominal exclusions intentionally recognize only the bounded reported grammar; other ambiguous prose remains fail closed for manual review.

## References

- [GitHub PR #325](https://github.com/NVIDIA/SkillSpector/pull/325)
- [GitHub PR #344](https://github.com/NVIDIA/SkillSpector/pull/344)
- [GitHub PR #427](https://github.com/NVIDIA/SkillSpector/pull/427)
- [GitHub PR #429](https://github.com/NVIDIA/SkillSpector/pull/429)
- [GitHub PR #432](https://github.com/NVIDIA/SkillSpector/pull/432)
- [GitHub PR #443](https://github.com/NVIDIA/SkillSpector/pull/443)
- [GitHub PR #453](https://github.com/NVIDIA/SkillSpector/pull/453)
