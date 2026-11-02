# SkillSpector 2.14.0

This candidate includes **53 distinct merged PR sources with new deltas**: **41 priority changes (19 P0, 22 P1)**, **5 required prerequisites**, and **7 model update sources**. All 31 original selections are retained. This update adds 21 priority sources (13 P0, 8 P1) and the formatter prerequisite; partial sources count only for their new portion.

The integrated target is `yashraj/v2.13.0-base` at `8882c8f68a3157049d6a344480d5e881d6a0ddda`, containing final #958 head `6fd523afd26a8b0099f9ea465f4b9e157cdf044d` (merge `38a1dfe93325d2f8bfcd4f8d0241eed706334915`) and #976. The maintenance baseline integration is complete.

[Selection manifest](skillspector-2.14.0-manifest.json) records each source and overlap.

## Merged PR sources newly included

| PR | Group | Merged change included in v2.14.0 |
| --- | --- | --- |
| [#742](https://github.com/NVIDIA/SkillSpector/pull/742) | P0 | fix(analyzer): enforce deadlines for projected prompt matching |
| [#773](https://github.com/NVIDIA/SkillSpector/pull/773) | P0 | fix(analyzer): report malformed dependency manifests |
| [#801](https://github.com/NVIDIA/SkillSpector/pull/801) | P0 | fix(analyzer): recognize JavaScript template literals |
| [#805](https://github.com/NVIDIA/SkillSpector/pull/805) | P0 | fix(report): keep missing references as non-blocking caveats |
| [#814](https://github.com/NVIDIA/SkillSpector/pull/814) | P0 | fix(analyzer): keep checking command lines inside a word that spans lines |
| [#848](https://github.com/NVIDIA/SkillSpector/pull/848) | P0 | fix(analyzer): keep PowerShell host syntax from inventing shell parse limits |
| [#872](https://github.com/NVIDIA/SkillSpector/pull/872) | P0 | fix(mp2): ignore repetition inside proven Markdown tables |
| [#898](https://github.com/NVIDIA/SkillSpector/pull/898) | P0 | Fix deterministic layer failing closed on ordinary lists of verb names |
| [#902](https://github.com/NVIDIA/SkillSpector/pull/902) | P0 | fix(static): stop backticks/apostrophes in comments and prose from tripping static_parse_limit |
| [#907](https://github.com/NVIDIA/SkillSpector/pull/907) | P0 | perf(security): avoid Unicode folding for ASCII characters |
| [#916](https://github.com/NVIDIA/SkillSpector/pull/916) | P0 (partial) | fix(deps): bump uv.lock past four packages with published advisories (#910) |
| [#934](https://github.com/NVIDIA/SkillSpector/pull/934) | P0 | fix(p2): keep reference destinations out of the benign-header exemption |
| [#939](https://github.com/NVIDIA/SkillSpector/pull/939) | P0 | fix(coverage): reconcile inspected ZIP containers and members |
| [#947](https://github.com/NVIDIA/SkillSpector/pull/947) | P0 | fix(scanner): tighten bounded parser ownership |
| [#948](https://github.com/NVIDIA/SkillSpector/pull/948) | P0 (partial) | fix(static): preserve shell and browser template boundaries |
| [#950](https://github.com/NVIDIA/SkillSpector/pull/950) | P0 | fix(markdown): preserve bounded inline and list-fence ownership |
| [#952](https://github.com/NVIDIA/SkillSpector/pull/952) | P0 | feat(supply-chain): inspect locked npm dependency contents |
| [#957](https://github.com/NVIDIA/SkillSpector/pull/957) | P0 | fix(ra1): keep double negatives from exempting the action |
| [#963](https://github.com/NVIDIA/SkillSpector/pull/963) | P0 | fix(supply-chain): strip leading BOM before parsing requirements.txt |
| [#622](https://github.com/NVIDIA/SkillSpector/pull/622) | P1 | fix(analyzers): carve out structurally benign P2 header comments |
| [#703](https://github.com/NVIDIA/SkillSpector/pull/703) | P1 | feat(report): expose the file surface a finding landed on |
| [#713](https://github.com/NVIDIA/SkillSpector/pull/713) | P1 | feat(providers): accept OpenCode 1.18.31-1.19.4 through an explicit verified set |
| [#752](https://github.com/NVIDIA/SkillSpector/pull/752) | P1 | fix(yara): require a browser cookie-store path in info_stealer |
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
| [#955](https://github.com/NVIDIA/SkillSpector/pull/955) | P1 | fix(cli): reject non-regular report outputs before scanning |
| [#964](https://github.com/NVIDIA/SkillSpector/pull/964) | P1 | fix(providers): reject SKILLSPECTOR_TEMPERATURE for all Claude models that drop sampling |
| [#965](https://github.com/NVIDIA/SkillSpector/pull/965) | P1 | fix(p6): stop reporting Python returns of local names as prompt extraction |
| [#967](https://github.com/NVIDIA/SkillSpector/pull/967) | P1 | fix(osv): fall back to static data when no_proxy breaks httpx client setup |
| [#969](https://github.com/NVIDIA/SkillSpector/pull/969) | P1 | fix(ea5): stop reading option-listing help rows as model-switch commands |
| [#972](https://github.com/NVIDIA/SkillSpector/pull/972) | P1 | perf: defer workflow warning imports until graph initialization |
| [#529](https://github.com/NVIDIA/SkillSpector/pull/529) | Model update (partial) | Provider-specific structured-output policy and model metadata |
| [#598](https://github.com/NVIDIA/SkillSpector/pull/598) | Model update (partial) | GPT-5.6 model budget metadata |
| [#650](https://github.com/NVIDIA/SkillSpector/pull/650) | Model update | GPT-6 Sol and Luna model budgets |
| [#690](https://github.com/NVIDIA/SkillSpector/pull/690) | Model update | GPT-6.1 Sol model budgets |
| [#708](https://github.com/NVIDIA/SkillSpector/pull/708) | Model update (partial) | OpenAI-compatible automatic tool-choice policy |
| [#806](https://github.com/NVIDIA/SkillSpector/pull/806) | Model update | New-model compatibility, control preflight and Bedrock profile repair |
| [#824](https://github.com/NVIDIA/SkillSpector/pull/824) | Model update | Retain previous defaults and preserve MCP preflight errors |
| [#810](https://github.com/NVIDIA/SkillSpector/pull/810) | Prerequisite | Retain OpenCode internal authentication plugins |
| [#436](https://github.com/NVIDIA/SkillSpector/pull/436) | Prerequisite | Lazy package graph initialization |
| [#819](https://github.com/NVIDIA/SkillSpector/pull/819) | Prerequisite | Bounded reference-comment destination and title ownership |
| [#662](https://github.com/NVIDIA/SkillSpector/pull/662) | Prerequisite | Report recursive scan errors with exit code 2 |
| [#981](https://github.com/NVIDIA/SkillSpector/pull/981) | Prerequisite (partial) | style(analyzer): format the multiline word-shadow guard |

## Additional merged priority changes in this update

- [#742](https://github.com/NVIDIA/SkillSpector/pull/742): fix(analyzer): enforce deadlines for projected prompt matching.
- [#773](https://github.com/NVIDIA/SkillSpector/pull/773): fix(analyzer): report malformed dependency manifests.
- [#801](https://github.com/NVIDIA/SkillSpector/pull/801): fix(analyzer): recognize JavaScript template literals.
- [#814](https://github.com/NVIDIA/SkillSpector/pull/814): fix(analyzer): keep checking command lines inside a word that spans lines.
- [#848](https://github.com/NVIDIA/SkillSpector/pull/848): fix(analyzer): keep PowerShell host syntax from inventing shell parse limits.
- [#902](https://github.com/NVIDIA/SkillSpector/pull/902): fix(static): stop backticks/apostrophes in comments and prose from tripping static_parse_limit.
- [#916](https://github.com/NVIDIA/SkillSpector/pull/916): fix(deps): bump uv.lock past four packages with published advisories (#910).
- [#947](https://github.com/NVIDIA/SkillSpector/pull/947): fix(scanner): tighten bounded parser ownership.
- [#948](https://github.com/NVIDIA/SkillSpector/pull/948): fix(static): preserve shell and browser template boundaries.
- [#950](https://github.com/NVIDIA/SkillSpector/pull/950): fix(markdown): preserve bounded inline and list-fence ownership.
- [#952](https://github.com/NVIDIA/SkillSpector/pull/952): feat(supply-chain): inspect locked npm dependency contents.
- [#957](https://github.com/NVIDIA/SkillSpector/pull/957): fix(ra1): keep double negatives from exempting the action.
- [#963](https://github.com/NVIDIA/SkillSpector/pull/963): fix(supply-chain): strip leading BOM before parsing requirements.txt.
- [#703](https://github.com/NVIDIA/SkillSpector/pull/703): feat(report): expose the file surface a finding landed on.
- [#752](https://github.com/NVIDIA/SkillSpector/pull/752): fix(yara): require a browser cookie-store path in info_stealer.
- [#955](https://github.com/NVIDIA/SkillSpector/pull/955): fix(cli): reject non-regular report outputs before scanning.
- [#964](https://github.com/NVIDIA/SkillSpector/pull/964): fix(providers): reject SKILLSPECTOR_TEMPERATURE for all Claude models that drop sampling.
- [#965](https://github.com/NVIDIA/SkillSpector/pull/965): fix(p6): stop reporting Python returns of local names as prompt extraction.
- [#967](https://github.com/NVIDIA/SkillSpector/pull/967): fix(osv): fall back to static data when no_proxy breaks httpx client setup.
- [#969](https://github.com/NVIDIA/SkillSpector/pull/969): fix(ea5): stop reading option-listing help rows as model-switch commands.
- [#972](https://github.com/NVIDIA/SkillSpector/pull/972): perf: defer workflow warning imports until graph initialization.

## Previously selected P0 changes retained

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

## Previously selected P1 changes retained

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
  prohibitions on recommending a self-update, with the double-negative correction from #957 now included ([#929](https://github.com/NVIDIA/SkillSpector/pull/929)).
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

- Carry the required formatter-only correction from [#981](https://github.com/NVIDIA/SkillSpector/pull/981) using actual PR head `281850410f7a04838eb1a87cdf822572ca624e8a`; its merge commit is empty because the change already reached main through #916.

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
the route; only the official native Anthropic endpoint accepts temperature `1.0`, while the other supported
routes reject explicit temperature. Unsupported settings fail before analysis.
MCP reports retain sanitized provider-preflight errors, and static-only scans
remain independent of provider credentials.

The merged [#964](https://github.com/NVIDIA/SkillSpector/pull/964) broadens the normalized known-family guard to `claude-opus-4-7`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`, `claude-fable-5`, `claude-mythos-5`, `claude-haiku-5`. This does not claim support for unlisted future families.

## Already included in the baseline

19 priority sources are represented by inherited maintenance work and excluded from the new-delta count. The baseline also absorbs #976 separately. Earlier #948 portions remain inherited while its new residual correction is counted once.

| Existing change | Baseline representation |
| --- | --- |
| [#945](https://github.com/NVIDIA/SkillSpector/pull/945) | Configurable TP4 capacity and the 1,080-second workflow allowance are already in #958. |
| [#946](https://github.com/NVIDIA/SkillSpector/pull/946) and [#845](https://github.com/NVIDIA/SkillSpector/pull/845) | Completed-response validation and bounded formatting recovery already handle absent structured results. #845 is not added again. |
| [#948](https://github.com/NVIDIA/SkillSpector/pull/948) and [#912](https://github.com/NVIDIA/SkillSpector/pull/912) | Bounded shell, Python, and browser-template ownership retains the inherited lexical components; the new residual #948 correction is selected above. |
| [#856](https://github.com/NVIDIA/SkillSpector/pull/856) | Asynchronous security discovery through the shared limiter is already present through #946; its production module matches main. |
| [#915](https://github.com/NVIDIA/SkillSpector/pull/915), [#919](https://github.com/NVIDIA/SkillSpector/pull/919), [#924](https://github.com/NVIDIA/SkillSpector/pull/924) | Dependency security pins, unavailable custom-YARA accounting, and baseline-input output protection are already backported. |
| [#906](https://github.com/NVIDIA/SkillSpector/pull/906), [#882](https://github.com/NVIDIA/SkillSpector/pull/882), [#874](https://github.com/NVIDIA/SkillSpector/pull/874) | CRLF Python shebangs, Markdown fence boundaries, and PowerShell SC3/P2 coverage are already backported. |
| [#881](https://github.com/NVIDIA/SkillSpector/pull/881), [#776](https://github.com/NVIDIA/SkillSpector/pull/776), [#683](https://github.com/NVIDIA/SkillSpector/pull/683) | MCP configuration filenames, JSON `npx` commands, and the package-operand pinning prerequisite are already included. |
| [#860](https://github.com/NVIDIA/SkillSpector/pull/860), [#826](https://github.com/NVIDIA/SkillSpector/pull/826), [#765](https://github.com/NVIDIA/SkillSpector/pull/765) | Multiline ransomware, nested urllib credential flows, and Pipenv dependency coverage are already backported. |
| [#893](https://github.com/NVIDIA/SkillSpector/pull/893), [#938](https://github.com/NVIDIA/SkillSpector/pull/938), [#772](https://github.com/NVIDIA/SkillSpector/pull/772), [#949](https://github.com/NVIDIA/SkillSpector/pull/949) | Agent-memory writes, recursive input protection, gap-fill scoring, and matcher-local timeout containment are already backported. The baseline also contains the follow-up memory-write regex bound. |
| [#452](https://github.com/NVIDIA/SkillSpector/pull/452) | Bounded HTML-comment closing and keyword ownership already shipped; the distinct reference-comment parser from #819 is a new prerequisite above. |
| [#877](https://github.com/NVIDIA/SkillSpector/pull/877) | Fully already covered by the inherited passive-image production and regression behavior; no new delta is counted. |
| [#976](https://github.com/NVIDIA/SkillSpector/pull/976) (original [#768](https://github.com/NVIDIA/SkillSpector/pull/768)) | Scoped allowed-tools support is absorbed through the current maintenance target base, not counted as a new selected source. |
| Earlier releases | #958 descends from published v2.13.2 (#922), v2.13.1 (#852), and v2.13.0 (#833). Their maintenance fixes, including Python proofs, passive-image/Office handling, version metadata, and report ACL protection, are inherited. |

The former double-negative RA1 limitation tracked in [#956](https://github.com/NVIDIA/SkillSpector/issues/956) is addressed by the included merged correction and regression tests from [#957](https://github.com/NVIDIA/SkillSpector/pull/957).

## Known limits and upgrading

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

- Dependency-content inspection is opt-in and limited to supported public npm lockfiles and Linux seccomp workers. Unsupported or uninspected artifacts remain partial; integrity verification does not establish package safety.

## Validation and publication

**Current update — static checks only.** For integrated source f583e0b7 before documentation-only edits, aggregate AST/internal-import checking reports 123 production modules and 443 internal imports with no errors or dynamic imports; Ruff lint and format pass across 320 files. JSON/TOML metadata, the offline no-build lock check (159 packages), and whitespace checks pass. These are static checks only.
No local contributor tests, project imports, builds, source/wheel scans or live-provider runs were performed for this update. Carried regression tests were inspected as source; static checks do not establish runtime acceptance. Final integrated CI, runtime/package acceptance and independent release review remain publication gates.

**Historical initial-candidate evidence, not validation of this update:** production commit `774bd450cac531a49ccc68ff876c0535d4b0ab52` recorded strict wheel/sdist checks and 128 matching packaged files (wheel SHA-256 `3185a1b1391e054b3386797a4d04acbbd8875b822d043d2c45d5308990b8ba9d`); 20 offline scans over 10 fixtures matched source/wheel reports; 145 installed model/registry tests passed. The local unit group recorded 2,932 passes, 14 skips and three initial failures, with all three isolated rechecks passing. Partial analyzer and other-source groups recorded 4,023 passes / four skips / three startup timeouts, and 1,675 passes / one timing failure / 2,178 unexecuted cases. The unchanged five-second dense-directory control also failed on #958 while retaining all 256-file semantic assertions. These partial runs were not a clean full-suite acceptance. Availability mocks accepted the supported timeout argument with test bodies and assertions unchanged.

The maintenance baseline is now integrated. PR #959 remains open for review. Publication is gated on final validation and release-owner review before `release:publish` is applied. Use the existing label-gated GitHub wheel/sdist workflow, which does not publish to PyPI.
