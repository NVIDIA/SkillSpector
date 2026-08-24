# SkillSpector v2.9.5

Released: 2026-08-14

## Summary

SkillSpector 2.9.5 expands provider support, adds opt-in author-shipped baselines, and strengthens static detection for insecure deserialization. It also improves analyzer accuracy and compatibility across safety-pattern, MCP dependency, and YARA scanning paths.

## Highlights

- Add Ollama, Azure OpenAI, and generic OpenAI-compatible providers.
- Add static detection coverage for insecure deserialization patterns.
- Improve scan accuracy with opt-in shipped baselines, lower false positives, and more precise YARA source locations.

## Added

- Add Ollama support for local OpenAI-compatible inference, Azure OpenAI deployment routing, and a configurable provider for other OpenAI-compatible endpoints.
- Add opt-in discovery of a top-level `.skillspector-baseline.yaml` while keeping explicitly supplied baselines authoritative.
- Add analyzer coverage for insecure deserialization patterns represented by AST10, TT6, and DS1–DS4 findings.

## Changed

- Use byte offsets when mapping YARA matches back to source lines so non-ASCII content is reported accurately.
- Scope the destructive-autonomy YARA post-filter to SkillSpector's built-in rule namespace and preserve deterministic built-in rule precedence.

## Fixed

- Reduce false positives when safety-sensitive language explicitly negates unsafe behavior.
- Bound the optional MCP dependency to the supported major version.

## Security

- Expand static analysis for insecure deserialization behavior and prevent custom YARA rules that reuse a built-in rule name from being incorrectly post-filtered.

## Breaking Changes and Migration

- None.

## Deprecations

- None.

## Validation

- Targeted and regression suites for provider selection, shipped baselines, deserialization analysis, safety-pattern controls, MCP packaging, and YARA analysis passed for the prepared imports.
- Ruff lint, Ruff formatting checks, and `git diff --check` passed for every prepared import.
- Required CI lint, unit, integration, Docker smoke, and Sonar checks passed for all seven imported changes.

## Known Limitations

- Ollama support requires a reachable local Ollama service; Azure OpenAI and generic OpenAI-compatible providers require their provider-specific endpoint and credential configuration.

## References

- [GitHub PR #246](https://github.com/NVIDIA/SkillSpector/pull/246)
- [GitHub PR #254](https://github.com/NVIDIA/SkillSpector/pull/254)
- [GitHub PR #339](https://github.com/NVIDIA/SkillSpector/pull/339)
- [GitHub PR #179](https://github.com/NVIDIA/SkillSpector/pull/179)
- [GitHub PR #286](https://github.com/NVIDIA/SkillSpector/pull/286)
- [GitHub PR #364](https://github.com/NVIDIA/SkillSpector/pull/364)
- [GitHub PR #365](https://github.com/NVIDIA/SkillSpector/pull/365)
