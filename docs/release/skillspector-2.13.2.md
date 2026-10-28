# SkillSpector 2.13.2

This patch release builds on the published v2.13.1 maintenance release and
backports three fixes for false incomplete-analysis reports. It preserves the
existing maintenance fixes, third-party dependency lock, and provider defaults.

## Fixes

- [#779](https://github.com/NVIDIA/SkillSpector/pull/779): Recognize command wrappers
  with no wrapped command at a proven clause boundary. Markdown table cells and
  ordinary source references to `timeout`, `sudo`, `nice`, and `xargs` no longer
  cause false parser-limit reports. Redirections, unfinished fragments, and
  commands in subsequent clauses retain their existing checks.
- [#917](https://github.com/NVIDIA/SkillSpector/pull/917): Treat `sudo -n` as a flag
  and parse descriptor redirections such as `2>&1` as complete operators. Align
  the MCP TP4 batch limit with its existing 128-file limit so supported inputs
  are not prematurely truncated after 64 batches. Byte, token, declaration,
  file, runtime, and provider-error limits remain enforced.
- [#921](https://github.com/NVIDIA/SkillSpector/pull/921): Recognize direct template
  messages on verified built-in Node assertion imports, avoiding false shell
  substitution limits in JavaScript helpers. Classify `.mjs` and `.cjs` files
  consistently, including archive members. Raw security findings and their
  source locations are preserved. Ambiguous syntax, assertion aliases or
  mutations, effectful interpolation, and exceeded bounds remain conservative;
  TypeScript retains its existing parser path.

## Scope and upgrading

The JavaScript correction is a bounded assertion-message proof, not a general
JavaScript parser or dataflow analysis. It assumes ordinary built-in assertion
semantics and does not prove the absence of arbitrary runtime mutation or
object getters. Existing findings and independent inspection limitations remain
visible; these fixes do not suppress malformed provider responses.

Regenerate and review exact version-bound finding baselines for 2.13.2 rather
than relabeling baselines from another scanner version. The GitHub release
provides a wheel and source distribution; the publication workflow does not
publish to PyPI.
