# SkillSpector 2.13.3

This patch release builds on the published v2.13.2 maintenance release,
preserving its fixes and supported Python versions while adding selected merged
P0 security and scan-integrity fixes, including four dependency security updates.

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

## Security and scan integrity

- [#924](https://github.com/NVIDIA/SkillSpector/pull/924) and
  [#938](https://github.com/NVIDIA/SkillSpector/pull/938): Reject baseline and
  recursive report destinations that would overwrite the primary input or a
  discovered child skill's input, including file aliases.
- [#919](https://github.com/NVIDIA/SkillSpector/pull/919): Report incomplete
  inspection when a requested custom YARA rule directory is unavailable.
- [#772](https://github.com/NVIDIA/SkillSpector/pull/772): Include non-English
  gap-fill findings in contributed batch-scan scores, recommendations, and exit
  status instead of retaining a stale SAFE verdict.
- [#906](https://github.com/NVIDIA/SkillSpector/pull/906): Inspect extensionless
  Python scripts with CRLF shebangs while recording interpreter uncertainty.
- [#874](https://github.com/NVIDIA/SkillSpector/pull/874): Restore PowerShell
  coverage for SC3 obfuscation and P2 hidden-instruction checks.
- [#882](https://github.com/NVIDIA/SkillSpector/pull/882): Preserve shell-line
  boundaries across Markdown code fences during marker reconstruction, keeping
  separate command lines from being combined into a misleading directive.
  Long rejected backtick runs are scanned linearly within the current text bound.
- [#826](https://github.com/NVIDIA/SkillSpector/pull/826): Follow credential reads
  nested in `urllib` request headers into network sinks.
- [#776](https://github.com/NVIDIA/SkillSpector/pull/776) and
  [#881](https://github.com/NVIDIA/SkillSpector/pull/881): Inspect `npx` commands in
  JSON MCP configurations and recognize additional Claude, Cline, and Roo Code
  credential/configuration filenames. The required
  [#683](https://github.com/NVIDIA/SkillSpector/pull/683) dependency accepts version
  pins only on the relevant package operand, so an unrelated later pin cannot
  hide an unpinned package. JSON command findings retain correct source lines
  when keys use escapes or appear more than once.
- [#893](https://github.com/NVIDIA/SkillSpector/pull/893): Detect writes to known
  agent-memory files and directories from automatically run test carriers,
  including supported raw Windows paths. A release review correction prevents
  unterminated destinations from causing excessive regular-expression backtracking.
- [#860](https://github.com/NVIDIA/SkillSpector/pull/860): Add a bounded,
  lower-confidence YARA signal for nearby directory walking, encryption, and
  binary writes across lines. Existing stronger ransomware rules are retained;
  the new signature does not prove which files a write replaces.
- [#765](https://github.com/NVIDIA/SkillSpector/pull/765): Extract Pipenv
  dependencies and check locked versions, preserving incomplete coverage for
  malformed or unsupported dependency declarations.
- [#915](https://github.com/NVIDIA/SkillSpector/pull/915): Refresh advisory-affected
  lockfile pins to `anyio 4.14.2`, `certifi 2026.10.19`, `langgraph-sdk 0.4.4`, and
  `urllib3 2.8.0`, without expanding the supported Python range.
- [#949](https://github.com/NVIDIA/SkillSpector/pull/949): Contain artifact-integrity
  matcher timeouts so subsequent artifacts are still inspected and the timed-out
  work remains explicitly incomplete.

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
