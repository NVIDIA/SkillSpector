# SkillSpector 2.13.6

This maintenance release builds on v2.13.5 and corrects incomplete static inspection of complete JSON metadata containing bare shell-wrapper names.

## Fixes

- [#1123](https://github.com/NVIDIA/SkillSpector/pull/1123): Complete JSON values such as `{"name":"nice"}` and `{"keywords":["sudo"]}` no longer produce a false `static_parse_limit` / `command_word_boundary` diagnostic. Validated JSON value boundaries establish when a bare wrapper name has no operands. Command arguments, runtime-selected wrappers, consumed options, clipped inputs, and unresolved shell fragments retain their existing checks and incomplete-inspection diagnostics.

## Previously released fixes retained

The v2.13.5 baseline already includes [#1087](https://github.com/NVIDIA/SkillSpector/pull/1087), grounding semantic intent findings in observed operations and declaration mismatches rather than domain labels alone. It also includes the requested fixes or equivalent backports from [#742](https://github.com/NVIDIA/SkillSpector/pull/742), [#773](https://github.com/NVIDIA/SkillSpector/pull/773), [#814](https://github.com/NVIDIA/SkillSpector/pull/814), [#963](https://github.com/NVIDIA/SkillSpector/pull/963), [#967](https://github.com/NVIDIA/SkillSpector/pull/967), [#855](https://github.com/NVIDIA/SkillSpector/pull/855), [#851](https://github.com/NVIDIA/SkillSpector/pull/851), [#899](https://github.com/NVIDIA/SkillSpector/pull/899), [#955](https://github.com/NVIDIA/SkillSpector/pull/955), [#965](https://github.com/NVIDIA/SkillSpector/pull/965), [#969](https://github.com/NVIDIA/SkillSpector/pull/969), [#752](https://github.com/NVIDIA/SkillSpector/pull/752), and [#662](https://github.com/NVIDIA/SkillSpector/pull/662).

The baseline retains the bounded multiline `printf` correction required with #814 and the actual formatting patch from the [#981](https://github.com/NVIDIA/SkillSpector/pull/981) PR head. The existing Bedrock Sonnet 4.6 profile identifier and output cap requested from [#806](https://github.com/NVIDIA/SkillSpector/pull/806) are also already present.

## Scope and upgrading

Unsupported content, exhausted budgets, unprovable runtime behavior, and provider failures remain explicit inspection limitations. Existing consumers must continue to handle `complete_with_caveats` and review coverage diagnostics. Review version-bound baselines when upgrading.

GitHub release assets contain a wheel and source distribution. This workflow does not publish to PyPI.
