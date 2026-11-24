# SkillSpector 2.13.7

This release comes from `main` at `2d0c6ddcb`. The previous release used the 2.13 maintenance branch. The changes below include earlier work on `main` and fixes merged after 2.13.6 shipped. [Published 2.13.6 release](https://github.com/NVIDIA/SkillSpector/releases/tag/v2.13.6)

## Scanning and reports

- Improve command and source boundaries in shell, Python, JavaScript, TypeScript, PowerShell, Markdown, and structured text. Scans retain source locations for unresolved command operands and other incomplete inspection. [#1151](https://github.com/NVIDIA/SkillSpector/pull/1151), [#1152](https://github.com/NVIDIA/SkillSpector/pull/1152), [#1153](https://github.com/NVIDIA/SkillSpector/pull/1153), [#1154](https://github.com/NVIDIA/SkillSpector/pull/1154), [#1127](https://github.com/NVIDIA/SkillSpector/pull/1127)
- Keep host Git templates out of Git URL scans. Reports include the sanitized acquisition origin and list built-in recursive root exclusions such as `.git/` and `.venv/` without counting those directories against coverage. [#1161](https://github.com/NVIDIA/SkillSpector/pull/1161), [#1118](https://github.com/NVIDIA/SkillSpector/pull/1118), [#1180](https://github.com/NVIDIA/SkillSpector/pull/1180)
- Point RP3 findings to the manifest's version field and include a bounded version snippet. Tag MCP tool-poisoning findings with MITRE ATLAS `AML.T0110`. [#1183](https://github.com/NVIDIA/SkillSpector/pull/1183), [#1134](https://github.com/NVIDIA/SkillSpector/pull/1134)
- Read dependency declarations from Poetry, PDM, Hatch, and uv tables in `pyproject.toml`. Support `--yara-rules-dir` during baseline generation so scans and baselines can use the same additional rules. [#190](https://github.com/NVIDIA/SkillSpector/pull/190), [#1117](https://github.com/NVIDIA/SkillSpector/pull/1117)
- Preserve handling for quoted YAML version metadata, including nested frontmatter. Real file references and quoted version-like text elsewhere in the document retain reference checks. [#525](https://github.com/NVIDIA/SkillSpector/pull/525)

## MCP and providers

- Use the stable official MCP Registry API and accept UTF-8 BOMs in local Registry JSON. MCP callers can request `include_report=false` to omit the rendered report from the tool response. [#1130](https://github.com/NVIDIA/SkillSpector/pull/1130), [#1136](https://github.com/NVIDIA/SkillSpector/pull/1136), [#1103](https://github.com/NVIDIA/SkillSpector/pull/1103)
- Add MiniMax regional provider support and update the NV Build default to GLM-5.3 Flash. [#366](https://github.com/NVIDIA/SkillSpector/pull/366), [#1009](https://github.com/NVIDIA/SkillSpector/pull/1009)
- Refresh the locked versions of pyasn1, cryptography, PyJWT, LangGraph, and LangGraph Checkpoint for dependency advisories. [#1148](https://github.com/NVIDIA/SkillSpector/pull/1148), [#1095](https://github.com/NVIDIA/SkillSpector/pull/1095)

## Documentation and release packaging

- Clarify Windows stdout, recursive skill discovery, Python API findings, and execution and completeness checks for install gates. [#1017](https://github.com/NVIDIA/SkillSpector/pull/1017), [#1172](https://github.com/NVIDIA/SkillSpector/pull/1172), [#1169](https://github.com/NVIDIA/SkillSpector/pull/1169), [#1182](https://github.com/NVIDIA/SkillSpector/pull/1182), [#1184](https://github.com/NVIDIA/SkillSpector/pull/1184)
- Move static analyzer test samples into text fixtures and scope the shared-AST tests to the scanned fixture, including CRLF source. [#1143](https://github.com/NVIDIA/SkillSpector/pull/1143), [#1181](https://github.com/NVIDIA/SkillSpector/pull/1181)
- Keep the source distribution limited to runtime source, build metadata, README, and license files. The approved release workflow builds and verifies both GitHub artifacts. PyPI and TestPyPI uploads use Trusted Publishing after the corresponding environment and index selector are configured. [#1179](https://github.com/NVIDIA/SkillSpector/pull/1179)

Review version-bound baselines when upgrading. Consumers should continue to handle `complete_with_caveats` and inspect coverage diagnostics according to their install policy.
