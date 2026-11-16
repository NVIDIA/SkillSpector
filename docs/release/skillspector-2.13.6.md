# SkillSpector 2.13.6

This maintenance release builds on v2.13.5 and corrects incomplete static inspection of complete JSON metadata containing bare shell-wrapper names.

## Fixes

- [#1123](https://github.com/NVIDIA/SkillSpector/pull/1123): Complete JSON values such as `{"name":"nice"}` and `{"keywords":["sudo"]}` no longer produce a false `static_parse_limit` / `command_word_boundary` diagnostic. Validated JSON value boundaries establish when a bare wrapper name has no operands. Command arguments, runtime-selected wrappers, consumed options, clipped inputs, and unresolved shell fragments retain their existing checks and incomplete-inspection diagnostics.

## Minor updates

- [#807](https://github.com/NVIDIA/SkillSpector/pull/807): Correct both development-guide links to the shared LLM analyzer base source module.
- [#423](https://github.com/NVIDIA/SkillSpector/pull/423): Document `--min-coverage` thresholds, recursive child coverage, and omitted-skill handling.
- [#685](https://github.com/NVIDIA/SkillSpector/pull/685): Correct least-privilege remediation guidance to require review of necessary capabilities and host enforcement before approving tools or changing permission declarations.
- [#816](https://github.com/NVIDIA/SkillSpector/pull/816): Document JSON archive-member provenance and how consumers should interpret virtual component locations.

## Scope and upgrading

Unsupported content, exhausted budgets, unprovable runtime behavior, and provider failures remain explicit inspection limitations. Existing consumers must continue to handle `complete_with_caveats` and review coverage diagnostics. Review version-bound baselines when upgrading.

GitHub release assets contain a wheel and source distribution. This workflow does not publish to PyPI.
