# SkillSpector 2.14.0

This release candidate builds on the 2.13.3 maintenance candidate in
[#958](https://github.com/NVIDIA/SkillSpector/pull/958), pinned at
`df3d072cdf3e3410e6916c0642e06540d41dc5d1`; refresh the baseline pin
before publication if that candidate advances. The new scope contains 20 merged
PRs from the October 28–30 daily review reports: six P0 and fourteen P1 changes.
It also includes four required prerequisites and seven requested model PRs,
for 31 distinct upstream PR sources with new changes. Some sources contribute
only the portions missing from the baseline.

The changes below are additions to that baseline. Existing 2.13.3 work and
fixes from earlier releases are inherited and are not counted as new features.
The machine-readable selection and overlap evidence are recorded in
[the release manifest](skillspector-2.14.0-manifest.json).

## Merged PRs newly included

All entries below are already merged upstream. Changes inherited from #958 and
earlier releases are listed separately and are not counted again.

| PR | Group | Merged change included in v2.14.0 |
| --- | --- | --- |
| [#805](https://github.com/NVIDIA/SkillSpector/pull/805) | P0 | fix(report): keep missing references as non-blocking caveats |
| [#872](https://github.com/NVIDIA/SkillSpector/pull/872) | P0 | fix(mp2): ignore repetition inside proven Markdown tables |
| [#898](https://github.com/NVIDIA/SkillSpector/pull/898) | P0 | Fix deterministic layer failing closed on ordinary lists of verb names |
| [#907](https://github.com/NVIDIA/SkillSpector/pull/907) | P0 | perf(security): avoid Unicode folding for ASCII characters |
| [#934](https://github.com/NVIDIA/SkillSpector/pull/934) | P0 | fix(p2): keep reference destinations out of the benign-header exemption |
| [#939](https://github.com/NVIDIA/SkillSpector/pull/939) | P0 | fix(coverage): reconcile inspected ZIP containers and members |
| [#622](https://github.com/NVIDIA/SkillSpector/pull/622) | P1 | fix(analyzers): carve out structurally benign P2 header comments |
| [#713](https://github.com/NVIDIA/SkillSpector/pull/713) | P1 | feat(providers): accept OpenCode 1.18.31-1.19.4 through an explicit verified set |
| [#763](https://github.com/NVIDIA/SkillSpector/pull/763) | P1 | test(batch): cover pooled model cancellation and slot recovery |
| [#847](https://github.com/NVIDIA/SkillSpector/pull/847) | P1 | feat(cli): configure the recursive skill count limit |
| [#851](https://github.com/NVIDIA/SkillSpector/pull/851) | P1 | fix(report): preserve each compacted occurrence's finding text |
| [#855](https://github.com/NVIDIA/SkillSpector/pull/855) | P1 | fix(llm): keep CLI prompts under MAX_INPUT_BYTES when batching large files |
| [#864](https://github.com/NVIDIA/SkillSpector/pull/864) | P1 | feat: support Python 3.15 |
| [#876](https://github.com/NVIDIA/SkillSpector/pull/876) | P1 | perf(cli): defer scan imports for version output |
| [#899](https://github.com/NVIDIA/SkillSpector/pull/899) | P1 | fix(cli): render error text literally instead of as Rich markup |
| [#903](https://github.com/NVIDIA/SkillSpector/pull/903) | P1 | fix(tm1): abstain on uppercase HTTP DELETE verbs in REST route tables |
| [#904](https://github.com/NVIDIA/SkillSpector/pull/904) | P1 | perf(providers): defer chat client imports from metadata path |
| [#914](https://github.com/NVIDIA/SkillSpector/pull/914) | P1 | feat(providers): support custom OpenAI-compatible HTTP headers |
| [#929](https://github.com/NVIDIA/SkillSpector/pull/929) | P1 | fix(ra1): honor prohibitions on recommending a self-update |
| [#943](https://github.com/NVIDIA/SkillSpector/pull/943) | P1 | fix(pe3): ignore exact Python glab setting literals |
| [#529](https://github.com/NVIDIA/SkillSpector/pull/529) | Model update (partial) | fix(providers): request JSON-schema structured output for models that reject forced tool calls |
| [#598](https://github.com/NVIDIA/SkillSpector/pull/598) | Model update (partial) | feat(providers): register GPT-5.6 models (sol, terra, luna) |
| [#650](https://github.com/NVIDIA/SkillSpector/pull/650) | Model update | feat(providers): add GPT-6 Sol and Luna budgets to OpenAI and Azure registries |
| [#690](https://github.com/NVIDIA/SkillSpector/pull/690) | Model update | feat(providers): add GPT-6.1 Sol budget to OpenAI and Azure registries |
| [#708](https://github.com/NVIDIA/SkillSpector/pull/708) | Model update (partial) | fix(providers): honour tool_choice: auto for openai_compatible models |
| [#806](https://github.com/NVIDIA/SkillSpector/pull/806) | Model update | feat(providers): adopt GPT-6.1 Sol and Claude Opus 5.5 defaults with compatible structured output |
| [#824](https://github.com/NVIDIA/SkillSpector/pull/824) | Model update | fix: retain previous defaults and preserve MCP preflight errors |
| [#436](https://github.com/NVIDIA/SkillSpector/pull/436) | Prerequisite | fix(cli): defer graph initialization |
| [#662](https://github.com/NVIDIA/SkillSpector/pull/662) | Prerequisite | fix(cli): exit 2 when a recursive scan cannot write its report |
| [#810](https://github.com/NVIDIA/SkillSpector/pull/810) | Prerequisite | fix(providers): keep OpenCode's internal plugins so OAuth logins work |
| [#819](https://github.com/NVIDIA/SkillSpector/pull/819) | Prerequisite (partial) | fix(p2): bound Markdown reference-comment findings |

## New P0 changes

- Treat genuinely missing references as non-blocking caveats. Keep existing
  but uninspected targets, symlinks, ambiguous references, and independent
  inspection failures distinct so they still prevent complete inspection
  ([#805](https://github.com/NVIDIA/SkillSpector/pull/805)).
- Reconcile successfully inspected ZIP containers with their inspected members
  while retaining descendant failures, archive limits, and Office metadata
  safeguards ([#939](https://github.com/NVIDIA/SkillSpector/pull/939)).
- Avoid false parser limits for ordinary quoted lists of verb names
  ([#898](https://github.com/NVIDIA/SkillSpector/pull/898)). Skip redundant
  Unicode folding for ASCII characters while retaining the bounded character
  caches ([#907](https://github.com/NVIDIA/SkillSpector/pull/907)).
- Reduce repetition findings inside proven Markdown table layout while keeping
  substantive repeated content and escaped-pipe cases visible
  ([#872](https://github.com/NVIDIA/SkillSpector/pull/872)).
- Keep reference destinations outside the benign header-comment exemption,
  preventing a harmless title from hiding instruction-bearing destination text.
  This correction is included together with #622 below
  ([#934](https://github.com/NVIDIA/SkillSpector/pull/934)).

## New P1 changes

- Support Python 3.15 by extending the declared Python range to `<3.16` and
  upgrading `ormsgpack` from 1.12.2 to 1.13.0
  ([#864](https://github.com/NVIDIA/SkillSpector/pull/864)).
- Add optional `SKILLSPECTOR_COMPAT_HEADERS` JSON configuration for custom
  OpenAI-compatible gateway headers, including credential scrubbing for child
  processes ([#914](https://github.com/NVIDIA/SkillSpector/pull/914)). Defer
  chat-client imports until inference is requested on the provider metadata
  path ([#904](https://github.com/NVIDIA/SkillSpector/pull/904)).
- Accept the explicit verified OpenCode set: 1.18.31–1.18.35 and
  1.19.0–1.19.4; reject other versions and prereleases
  ([#713](https://github.com/NVIDIA/SkillSpector/pull/713)). Include its
  previously merged OAuth prerequisite: retain OpenCode's internal
  authentication plugins while preserving the deny-all tool policy,
  `--pure`, and external-plugin restrictions
  ([#810](https://github.com/NVIDIA/SkillSpector/pull/810)).
- Render CLI errors and output-destination messages literally, preserving
  brackets and making terminal control characters visible
  ([#899](https://github.com/NVIDIA/SkillSpector/pull/899)). Load the installed
  version-query entry point without initializing scan dependencies
  ([#876](https://github.com/NVIDIA/SkillSpector/pull/876)).
- Add `scan --recursive --max-skills N` for local multi-skill collections.
  The default remains 32; positive explicit limits retain the other aggregate
  discovery, byte, output, and runtime bounds
  ([#847](https://github.com/NVIDIA/SkillSpector/pull/847)).
- Preserve each compacted occurrence's own finding text when expanding JSON
  and SARIF reports ([#851](https://github.com/NVIDIA/SkillSpector/pull/851)).
- Size CLI-provider batches by UTF-8 bytes as well as tokens, measuring the
  complete prompt and structured-output instructions. Redact SC10 file content
  before splitting. Oversized individual lines remain visible and fail closed
  rather than being truncated
  ([#855](https://github.com/NVIDIA/SkillSpector/pull/855)).
- Reduce false positives for exact Python `GLAB_USE_KEYRING` setting literals
  and uppercase HTTP `DELETE` routes while retaining credential operations and
  destructive host-path commands
  ([#943](https://github.com/NVIDIA/SkillSpector/pull/943),
  [#903](https://github.com/NVIDIA/SkillSpector/pull/903)).
- Exempt only structurally proven benign license and metadata header comments;
  include the #934 destination-safety correction with this exemption
  ([#622](https://github.com/NVIDIA/SkillSpector/pull/622)). Recognize direct
  prohibitions on recommending a self-update, subject to the known limitation
  below ([#929](https://github.com/NVIDIA/SkillSpector/pull/929)).
- Add mocked regression tests for cancellation, slot reuse, and one-time pool
  cleanup. This is test coverage; the production cleanup fix from #796 is
  already inherited ([#763](https://github.com/NVIDIA/SkillSpector/pull/763)).

## Required prerequisites

- Restore OpenCode's internal authentication plugins and the associated
  isolation fixtures for the expanded verified runtime set
  ([#810](https://github.com/NVIDIA/SkillSpector/pull/810)).
- Initialize the package graph lazily so metadata and version-only imports can
  remain independent of scan and chat-client initialization
  ([#436](https://github.com/NVIDIA/SkillSpector/pull/436)).
- Include the reference-comment destination/title parser and its ownership,
  deadline, and boundary regressions required by #622/#934
  ([#819](https://github.com/NVIDIA/SkillSpector/pull/819)). The maintenance
  branch's already-shipped HTML-comment fixes from
  [#452](https://github.com/NVIDIA/SkillSpector/pull/452) remain inherited.
- Preserve recursive scan failures as CLI errors with exit code 2, including
  protected report-output failures, when applying the literal error-rendering
  correction ([#662](https://github.com/NVIDIA/SkillSpector/pull/662)).

## Model compatibility and defaults

Add the missing parts of the seven requested model PRs:
[#529](https://github.com/NVIDIA/SkillSpector/pull/529),
[#598](https://github.com/NVIDIA/SkillSpector/pull/598),
[#650](https://github.com/NVIDIA/SkillSpector/pull/650),
[#690](https://github.com/NVIDIA/SkillSpector/pull/690),
[#708](https://github.com/NVIDIA/SkillSpector/pull/708),
[#806](https://github.com/NVIDIA/SkillSpector/pull/806), and
[#824](https://github.com/NVIDIA/SkillSpector/pull/824).

The additions cover missing GPT-5.6 fallback, GPT-6 Sol/Luna, GPT-6.1 Sol,
Claude, and spark-x2.5 metadata, provider-specific structured-output policy,
model-name normalization, and unsupported-control preflight. Preserve #958's
completion validation, formatting recovery and binder support: their overlapping
parts of #529/#708 are not reapplied. Bundled OpenAI GPT-5.6 metadata from #598
is also already inherited; its missing root fallback entries are added.

The net defaults include #824's rollback:

| Provider | Default after this release |
| --- | --- |
| OpenAI and its credential fallback | `gpt-5.4` |
| Native Anthropic analyzers | `claude-opus-4-6` |
| Native Anthropic meta-analyzer | `claude-sonnet-4-6` |
| Bedrock | `us.anthropic.claude-sonnet-4-6`, with a 64,000-token output cap |

Newer registered models remain opt-in. Other provider defaults remain unchanged.
Bedrock's suffix-free profile replaces the invalid dated Sonnet 4.6 default;
custom registries must include the corrected key to supply its budgets.

GPT-6.1 Sol rejects explicit temperature and accepts reasoning effort only at
`low`, `medium`, `high`, `xhigh`, or `max`. Claude 5.5 control handling depends on
the route; native Anthropic accepts temperature `1.0`, while the other supported
routes reject explicit temperature. Unsupported settings fail before analysis.
MCP reports retain sanitized provider-preflight errors, and static-only scans
remain independent of provider credentials.

## Already included in the baseline

These items are deliberately excluded from the new-change count and bullets.
The daily reports' 38 P0/P1 records partition into 20 selected changes and 18
changes already represented by #958. No merged P2 PR is present in those reports.

| Existing change | Baseline representation |
| --- | --- |
| [#945](https://github.com/NVIDIA/SkillSpector/pull/945) | Configurable TP4 capacity and the 1,080-second workflow allowance are already in #958. |
| [#946](https://github.com/NVIDIA/SkillSpector/pull/946) and [#845](https://github.com/NVIDIA/SkillSpector/pull/845) | Completed-response validation and bounded formatting recovery already handle absent structured results. #845 is not added again. |
| [#948](https://github.com/NVIDIA/SkillSpector/pull/948) and [#912](https://github.com/NVIDIA/SkillSpector/pull/912) | Bounded shell, Python, and browser-template ownership already includes the lexical shell-ownership proof. |
| [#856](https://github.com/NVIDIA/SkillSpector/pull/856) | Asynchronous security discovery through the shared limiter is already present through #946; its production module matches main. |
| [#915](https://github.com/NVIDIA/SkillSpector/pull/915), [#919](https://github.com/NVIDIA/SkillSpector/pull/919), [#924](https://github.com/NVIDIA/SkillSpector/pull/924) | Dependency security pins, unavailable custom-YARA accounting, and baseline-input output protection are already backported. |
| [#906](https://github.com/NVIDIA/SkillSpector/pull/906), [#882](https://github.com/NVIDIA/SkillSpector/pull/882), [#874](https://github.com/NVIDIA/SkillSpector/pull/874) | CRLF Python shebangs, Markdown fence boundaries, and PowerShell SC3/P2 coverage are already backported. |
| [#881](https://github.com/NVIDIA/SkillSpector/pull/881), [#776](https://github.com/NVIDIA/SkillSpector/pull/776), [#683](https://github.com/NVIDIA/SkillSpector/pull/683) | MCP configuration filenames, JSON `npx` commands, and the package-operand pinning prerequisite are already included. |
| [#860](https://github.com/NVIDIA/SkillSpector/pull/860), [#826](https://github.com/NVIDIA/SkillSpector/pull/826), [#765](https://github.com/NVIDIA/SkillSpector/pull/765) | Multiline ransomware, nested urllib credential flows, and Pipenv dependency coverage are already backported. |
| [#893](https://github.com/NVIDIA/SkillSpector/pull/893), [#938](https://github.com/NVIDIA/SkillSpector/pull/938), [#772](https://github.com/NVIDIA/SkillSpector/pull/772), [#949](https://github.com/NVIDIA/SkillSpector/pull/949) | Agent-memory writes, recursive input protection, gap-fill scoring, and matcher-local timeout containment are already backported. The baseline also contains the follow-up memory-write regex bound. |
| [#452](https://github.com/NVIDIA/SkillSpector/pull/452) | Bounded HTML-comment closing and keyword ownership already shipped; the distinct reference-comment parser from #819 is a new prerequisite above. |
| Earlier releases | #958 descends from published v2.13.2 (#922), v2.13.1 (#852), and v2.13.0 (#833). Their maintenance fixes, including Python proofs, passive-image/Office handling, version metadata, and report ACL protection, are inherited. |

## Known limits and upgrading

- The merged #929 has a reported double-negative regression: a sentence such as
  `Never recommend skipping self-update` can lose its RA1 finding. This remains
  tracked in [#956](https://github.com/NVIDIA/SkillSpector/issues/956); the
  unmerged proposed correction [#957](https://github.com/NVIDIA/SkillSpector/pull/957)
  is outside this selection.
- The inherited #958 limitations still apply. Ambiguous or generated execution
  paths can remain incomplete; provider refusals, malformed responses, and
  exhausted budgets can prevent completed AI inspection. Increasing workflow
  limits does not extend Pi/OpenCode's separate wrapper timeouts. Earlier live
  results from #958 do not validate this new candidate.
- Regenerate and review version-bound finding baselines for 2.14.0. Do not
  relabel older exact fingerprints; changed findings and source evidence need
  a fresh review.
- The GitHub release distributes a wheel and source distribution. Its workflow
  does not publish to PyPI.

## Validation and publication

Validation is recorded on [release PR #959](https://github.com/NVIDIA/SkillSpector/pull/959).
Final isolated CI and baseline integration remain pending.

- Strict wheel/source-distribution build and metadata checks pass. All 128
  packaged runtime/resource files match the candidate source byte for byte.
- Twenty offline CLI scans across ten fixtures produce matching normalized
  source/wheel reports, including retained security findings. Installed-wheel
  model and registry checks pass all 145 cases.
- The local unit group recorded 2,932 passes, 14 skips, and three initial
  failures. All three isolated rechecks pass: one corrected API mock and two
  unchanged startup controls. This is not a clean full-suite run.
- Partial analyzer and other-source groups recorded 4,023 and 1,675 passes.
  Three analyzer child processes exceeded their startup limits before analyzer
  execution; one dense-directory timing check exceeded five seconds while
  retaining every file and coverage assertion. That timing failure also occurs
  on the #958 baseline. Limits and production code were not weakened.
- Availability test mocks now accept the supported optional timeout argument;
  test bodies and assertions are unchanged. Partial local runs and historical
  #958 live results do not establish final-candidate acceptance.

The release PR is stacked on `yashraj/release-2.13.3` so its diff contains the
new delta. Publication must wait for #958's integration and the final release
checks; the draft PR must not carry `release:publish` while those gates remain.
