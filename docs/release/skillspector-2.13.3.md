# SkillSpector 2.13.3

This patch release builds on the published v2.13.2 maintenance release,
preserving its fixes, third-party dependency lock, and supported Python versions.

## Fixes

- [#945](https://github.com/NVIDIA/SkillSpector/pull/945): Make the tool-description
  comparison batch limit configurable with `SKILLSPECTOR_TP4_MAX_BATCHES`, default
  **256**, and raise the default workflow deadline to **1,080 seconds (18 minutes)**,
  configurable with `SKILLSPECTOR_MAX_WORKFLOW_SECONDS`. Other file, byte, token,
  concurrency, and provider limits remain enforced. The release also includes the
  required bounded parallel batch execution change from upstream commit
  [`9919c159`](https://github.com/NVIDIA/SkillSpector/commit/9919c159597e62016214a1d97ae1eb559e547b11),
  honoring `SKILLSPECTOR_MAX_LLM_CONCURRENCY` and preserving batch result order.
- [#946](https://github.com/NVIDIA/SkillSpector/pull/946): Validate whether AI
  responses actually finished and contain the required fields. Report refusals,
  truncated responses, missing fields, and invalid tool results as incomplete
  inspection. Permit bounded formatting recovery only when a completed response
  preserves the same values and structure. Use consistent structured-response
  configuration across analyzers, including the supported Opus 5 gateway route,
  and run semantic discovery through the shared asynchronous path.
- [#948](https://github.com/NVIDIA/SkillSpector/pull/948): Recognize additional
  bounded shell syntax, Python string ownership, and generated browser-template
  text without treating ordinary data as unfinished shell commands. Preserve
  findings and incomplete inspection when text can reach command execution,
  syntax is ambiguous, or analysis limits are exceeded.

## Scope and upgrading

The increased limits permit larger scans; completion still depends on input
size, provider latency, and the remaining analysis limits. The Pi and OpenCode
wrappers in this maintenance release retain their 120-second process timeout;
use the CLI directly for scans needing the longer workflow deadline.

These changes do not provide general analysis of commands generated and loaded
during execution, including generated helpers and archives. Such workflows may
still report incomplete inspection. Formatting recovery does not invent missing
content or turn an unfinished AI response into a completed review.

Regenerate and review version-bound finding baselines for 2.13.3 rather than
relabeling baselines from another scanner version. The GitHub release provides a
wheel and source distribution; the publication workflow does not publish to PyPI.
