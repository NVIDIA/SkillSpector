# SkillSpector v2.12.0

Release status: candidate; publication pending.

## Summary

SkillSpector 2.12.0 adds an opt-in CLI gate for any active finding, a configurable static-analysis allowance, OpenCode integrations, Gemini 3.5 Flash registry guidance, interactive scan progress, and TP4 analysis of executable Markdown fences. It also strengthens local input and report handling, expands detection of reflective Python access and shipped bytecode, retries transient provider failures, and distinguishes missing references from ambiguous MCP installation blockers. The release includes discovery, completeness, finding-identity, SARIF output, raw forge-file input, performance, and false-positive fixes described below.

The latest additions include sanitized LLM provenance, SC10 dependency-source analysis, GitHub tree-directory inputs, opt-in compact prompt numbering, expanded model-budget metadata, fail-closed recursive reporting, and occurrence-specific JSON/SARIF columns. Further detection and coverage fixes distinguish narrow passive-image cases from active or unknown opaque content without treating uninspected bytes as safe.

The candidate also includes the previously pending Markdown-reference and runtime-command completeness fixes, required-input and multiline-prompt accounting, and conservative Perl literal handling with clearer referenced-artifact diagnostics.

## Highlights

- Preserve fatal outcomes for recognized unsupported required input and incomplete coverage for bounded runtime-command and multiline-prompt reconstruction.
- Resolve complete Markdown reference destinations without losing URI or source ownership, and distinguish proven Perl print literals from ambiguous source.
- Record sanitized LLM configuration/execution provenance and preserve occurrence-specific locations in JSON and SARIF.
- Detect dependency-source redirection with SC10 and retain recursive failure, risk, and completeness evidence even when reports are bounded.
- Scan selected GitHub tree subdirectories, optionally compact prompt line labels, and use expanded OpenAI/Azure model-budget metadata.
- Extend TP4 semantic checks to executable Markdown fences, detect reflective Python module lookups, and find shipped bytecode inside Python environments.
- Retry transient provider failures within the workflow deadline and constrain local registry, baseline, and Pi extension inputs and report writes.
- Distinguish missing file references from ambiguous references in the MCP installation decision, while preserving completeness caveats in reports.
- Run semantic analysis through the new `opencode_cli` provider and invoke SkillSpector from OpenCode through a native `/skillspector` command and `skillspector_scan` tool.
- Use `gemini-3.5-flash` through the existing OpenAI-compatible provider, and see bounded live progress plus a control-safe discovered-file tree during non-verbose interactive scans.
- Emit merged recursive SARIF on standard output without status-text contamination, and scan GitHub/GitLab `/blob/` links as their raw file contents instead of forge HTML.
- Inventory executable and loadable content in normally excluded locations, preserving coverage evidence and failing closed when referenced or otherwise outside inspection coverage.
- Mark active hooks with unmodeled payload data flow as partial and opaque, retaining BH1 while preventing a misleading `SAFE` recommendation.
- Reduce security-view CPU and tail latency with ASCII token-gap fast paths and bounded memoization while preserving findings and coverage outcomes.
- Keep multilingual batch language detection and gap-fill on the graph's validated provider-eligible cache instead of rereading paths after inspection.
- Retarget per-call LLM deadlines on existing clients to avoid connection-pool churn and closed-event-loop cleanup failures.
- Honor model-registry overrides for CLI providers and accept valid Windows 8.3 aliases without weakening opened-handle validation.
- Recognize complete JSON strings and Markdown code spans in their document context, avoiding false analysis limits while retaining analysis of their contents.
- Keep delimiter pairing within Markdown blocks and table cells so unrelated documentation cannot hide unresolved runtime commands.
- Scan JSON quote candidates in linear time and retain cancellation handling.
- Preserve distinct full-evidence and rule identities, including findings with identical shortened previews, and retain precise locations for repeated occurrences.
- Keep requested but unavailable or incomplete semantic analysis visible in completeness metadata and strict CLI/MCP decisions.
- Classify selected OAuth, signed self-update, and warned installer documentation based on surrounding context, subject to the PE3 and RA1 gaps recorded under Known Limitations.
- Suppress AS3 only when a literal `skills/<name>/SKILL.md` path identifies the skill currently being scanned, while retaining peer-skill, transformed, and enumeration findings.

## Added

- JSON reports add `metadata.llm_provenance` (schema version 1), distinguishing configured, resolved, and effective providers and requested versus observed forwarded sampling/reasoning controls. It records bounded, sanitized model/package/source identity without endpoint URLs or credentials, and explicitly does not guarantee deterministic provider behavior. `SKILLSPECTOR_BUILD_REVISION` can supply source revision metadata when unavailable from the installed package ([#556](https://github.com/NVIDIA/SkillSpector/pull/556)).
- SC10 detects noncanonical or unresolved dependency sources in npm, Yarn, pip, Poetry, Maven, and Cargo configuration, supported shell scripts, generated configuration heredocs, and actionable shell fences in `SKILL.md`/`README.md`. Deterministic HIGH findings survive optional LLM filtering, subject to explicit runtime/finding budgets; intentional private registries may need reviewed baseline suppression. This is not a registry reputation or network-reachability check ([#383](https://github.com/NVIDIA/SkillSpector/pull/383)).
- HTTPS GitHub `/tree/` URLs can select an existing repository subdirectory. Resolution uses the longest advertised branch/tag name, including refs containing slashes, and rejects decoded traversal or out-of-checkout selections. Arbitrary commit-SHA tree URLs are not supported by this ref-resolution path ([#561](https://github.com/NVIDIA/SkillSpector/pull/561)).
- `SKILLSPECTOR_COMPACT_PROMPTS=1`, `true`, or `yes` enables unpadded LLM prompt line labels such as `L1:` instead of `L01:`. Values are trimmed and case-insensitive; unset and other values retain the existing format ([#542](https://github.com/NVIDIA/SkillSpector/pull/542)).
- `SKILLSPECTOR_PROVIDER=opencode_cli` runs semantic analysis through a local OpenCode login. The verified deny-all policy requires exactly OpenCode 1.18.31; authentication, version, policy, empty-output, or event-envelope failures fail closed. `SKILLSPECTOR_MODEL` remains optional ([#536](https://github.com/NVIDIA/SkillSpector/pull/536), [#575](https://github.com/NVIDIA/SkillSpector/pull/575)).
- TP4 semantic analysis includes non-empty, closed Markdown/text fences with recognized executable-language labels, using the validated provider-eligible cache and original document line numbers. Size, count, prompt, and runtime limits remain explicit coverage constraints ([#421](https://github.com/NVIDIA/SkillSpector/pull/421)).
- The repository-provided OpenCode extension adds a static-by-default `/skillspector` command and `skillspector_scan` tool. Copy `.opencode/` from a checkout to install it; the wheel does not install the extension. It resolves the binary from `SKILLSPECTOR_BIN`, a worktree `.venv`, or `PATH`, requests required host capabilities before launch, rejects symlinked target/output/binary paths, and uses a 120-second timeout with bounded, redacted output. Semantic analysis is opt-in through `noLlm=false` and the provider environment ([#537](https://github.com/NVIDIA/SkillSpector/pull/537)).
- `SKILLSPECTOR_MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT` configures the static pattern and YARA time allowance per artifact; its default increases from 30 to 300 seconds. The remaining workflow deadline still bounds both analyzers.
- `skillspector scan --fail-on-findings` exits with code 1 when a scan reports any active finding, including findings below the default risk-score threshold. It applies to single-skill, recursive, and MCP registry scans. Skill scans evaluate active findings after suppression ([#469](https://github.com/NVIDIA/SkillSpector/pull/469)).
- `gemini-3.5-flash` model budgets are registered for the root and OpenAI-compatible registries, with a Gemini setup example using the existing `openai` provider ([#7](https://github.com/NVIDIA/SkillSpector/pull/7)).

## Changed

- Expand GPT-family context/output-budget entries in the OpenAI and Azure registries, including correction of the OpenAI GPT-5.4 context entry. These are metadata changes only: default models and API adapters are unchanged, and Responses-only model entries do not establish compatibility with another API ([#539](https://github.com/NVIDIA/SkillSpector/pull/539)).
- JSON findings retain each occurrence's own zero-based columns; SARIF emits one-based Unicode-codepoint columns for active and suppressed findings. Unknown occurrence columns remain unknown rather than inheriting another occurrence's location ([#584](https://github.com/NVIDIA/SkillSpector/pull/584)).
- GHSA/OSV `MODERATE` severity maps to `MEDIUM`, correcting the previous `LOW` fallback and potentially increasing affected supply-chain risk scores ([#588](https://github.com/NVIDIA/SkillSpector/pull/588)).
- Missing local references now use `reference_missing`; ambiguous matches retain `reference_unresolved`. The MCP skill-scan installation decision can allow a scan whose only caveats are missing references, provided all discovered files were inspected and the other risk, execution, and requested-analysis checks pass. Completeness metadata and the report recommendation still retain the caveat ([#526](https://github.com/NVIDIA/SkillSpector/pull/526)).
- Remove the OpenSSF Scorecard and HVTrust README badges ([#582](https://github.com/NVIDIA/SkillSpector/pull/582)).
- Non-verbose scans attached to an interactive terminal now stream graph execution and render bounded progress, completed analyzer rules, and a control-safe discovered-file tree to standard error. Machine-readable standard output and exit-code behavior remain intact ([#7](https://github.com/NVIDIA/SkillSpector/pull/7)).
- Printable-ASCII token-gap scans use precomputed classification tables and a whole-text fast path; repeated pure security-view predicates use bounded memoization. The measured 901-skill corpus retained byte-identical findings and coverage outcomes while reducing p95, p99, and total CPU cost ([#569](https://github.com/NVIDIA/SkillSpector/pull/569), [#570](https://github.com/NVIDIA/SkillSpector/pull/570)).
- CLI-backed providers now honor `SKILLSPECTOR_MODEL_REGISTRY` for context and output-token limits. Missing entries retain the existing fallback behavior; malformed registry structures or invalid and non-positive budgets warn and fall back ([#463](https://github.com/NVIDIA/SkillSpector/pull/463)).
- The repository's `contrib/batch_scan` tool reuses the graph's validated `llm_file_cache` for multilingual language detection and gap-fill, avoiding a second raw filesystem read and retaining provider/local-only boundaries. `contrib` remains outside the wheel ([#558](https://github.com/NVIDIA/SkillSpector/pull/558)).
- Align provider setup guidance and update research background counts ([#434](https://github.com/NVIDIA/SkillSpector/pull/434), [#543](https://github.com/NVIDIA/SkillSpector/pull/543)). The HVTrust badge added in [#428](https://github.com/NVIDIA/SkillSpector/pull/428) was subsequently removed in #582.
- Reports retain requested LLM intent separately from runtime availability. Incomplete semantic execution remains visible through aggregate reports and CLI/MCP installation gates ([#410](https://github.com/NVIDIA/SkillSpector/pull/410)).

## Fixed

- Preserve incomplete coverage for runtime-selected `printf` and executable wrappers, bounded `eval`/shell `-c` strings, shell commands embedded in PowerShell, unsupported brace expansion, and commands crossing analysis windows. Parser and deadline uncertainty remains visible through `static_parse_limit` and CLI/MCP completeness gates even when semantic analysis succeeds. Literal/documentation controls remain distinct; this is bounded analysis, not general shell emulation ([#514](https://github.com/NVIDIA/SkillSpector/pull/514)).
- Tolerate transiently missing `.git` metadata only while a clone is active, then strictly remeasure the completed checkout. Permission errors, checkout failures, and final ingestion limits still fail closed ([#514](https://github.com/NVIDIA/SkillSpector/pull/514)).
- Record fatal `unsupported_primary_content` for recognized unsupported selected inputs and in-profile `SKILL.md`/`skill.md` instructions, including nested or renamed ZIP members. Preserve required-file identity, reject lossy required UTF-8 decoding, retain canonical bytes, and omit rejected primary text from provider input. Supported ZIP inspection and incidental-asset policy remain bounded and unchanged; a valid UTF-8 code point cut by a recorded byte limit remains partial rather than an unsupported-encoding failure ([#563](https://github.com/NVIDIA/SkillSpector/pull/563)).
- Retain failed artifact outcomes and excluded-executable evidence even when detailed ledger output is truncated or manifest parsing also fails. Fatal required-content failures remain execution failures rather than being downgraded to partial coverage ([#563](https://github.com/NVIDIA/SkillSpector/pull/563)).
- Account for covered pure or mixed singleton newline/horizontal spacing with source-preserving AE6 and `obfuscated_instruction_text` partial evidence. The projection preserves structural boundaries and uses interruptible, workflow-bounded matching; it does not manufacture a confirmed P3/P4 finding or claim universal deobfuscation ([#563](https://github.com/NVIDIA/SkillSpector/pull/563)).
- Parse bounded inline and reference-definition Markdown destinations with angle-wrapped spaces, balanced/escaped parentheses, and quoted titles. Split URI components before one-time decoding, preserve decoded containment and reference ownership, retain genuine missing targets, and report destination/title limit exhaustion as incomplete analysis. Title text is not reinterpreted as an extra reference ([#553](https://github.com/NVIDIA/SkillSpector/pull/553)).
- Classify `.pl` as Perl while retaining its existing security checks, and avoid false shell-parse-limit AE1 for narrowly proven standalone non-interpolated print literals. Printed content still receives security analysis; interpolation, quote-like syntax, heredocs, ambiguous fragments, and genuine limits remain conservative ([#615](https://github.com/NVIDIA/SkillSpector/pull/615)).
- Explain AE1 referenced-artifact failures with bounded, sanitized target disposition and canonical reason evidence, including truncation metadata and reason-specific remediation. Guidance preserves required references rather than suggesting their removal ([#615](https://github.com/NVIDIA/SkillSpector/pull/615)).
- Recursive and transitive scans retain failed-child, risk, completeness, and omission evidence when output or ledger budgets omit detailed bodies. Recursive Markdown stdout includes child report sections without requiring `--output`. Actual execution failures return exit 2; bounded static path postprocessing shares analysis budgets and preserves syntax/coverage limitations ([#576](https://github.com/NVIDIA/SkillSpector/pull/576)).
- Analyzer import/registry-load failures produce SYSTEM/PARTIAL ledger evidence with reason `analyzer_load_error`, so a missing analyzer cannot silently yield complete coverage. This remains a coverage gap rather than an execution crash ([#591](https://github.com/NVIDIA/SkillSpector/pull/591)).
- Suppress AE1 only for narrowly verified passive rendered Markdown-image references to structurally valid minimal non-interlaced PNG data when the only limitations are unsupported binary/opaque format. Incomplete coverage, `CAUTION`, strict incomplete gates, and MCP installation blocking remain. Other uses of resolved targets, malformed or non-PNG content, concealed payloads, and mixed/unknown limitations retain AE1. DEX and Lua bytecode signatures are also recognized as binary/executable content ([#597](https://github.com/NVIDIA/SkillSpector/pull/597)).
- Analyze bounded activation-intent clauses in descriptions when legacy `triggers` are absent or empty, covering broad trigger phrases, command interception, and catch-all activation while excluding ordinary capability prose. Excess signal-bearing clauses are recorded as incomplete coverage; explicit legacy triggers retain their behavior ([#541](https://github.com/NVIDIA/SkillSpector/pull/541)).
- Detect nearby literal `True` variables passed to subprocess `shell=`, with reassignment and Python-scope checks to avoid treating unrelated assignments as data flow ([#560](https://github.com/NVIDIA/SkillSpector/pull/560)).
- Recognize bounded literal XOR-decoded script-fetch commands from supported local Python helper patterns without general execution or deobfuscation, preserving other findings and logical source-line locations ([#546](https://github.com/NVIDIA/SkillSpector/pull/546)).
- Resolve bounded literal list/tuple joins used as reflective `getattr` names: dangerous names produce AST9, while unresolved or constructed benign names retain AST7. Prospective output length is checked before allocation ([#544](https://github.com/NVIDIA/SkillSpector/pull/544)).
- Require a word boundary for the YARA exploit-framework `ROP` token, retaining `ROP(elf)` controls without false HIGH findings for Rust `drop(&mut self)` ([#607](https://github.com/NVIDIA/SkillSpector/pull/607)).
- Redact credential-bearing URL userinfo and sensitive query fields throughout nested finding evidence and dependency-source meta-analysis inputs ([#383](https://github.com/NVIDIA/SkillSpector/pull/383)).
- Retry transient connection, timeout, selected HTTP status, and Bedrock service/throttling failures with bounded backoff and bounded `Retry-After` handling. Retry waits respect the workflow deadline; Bedrock SDK retries are disabled to avoid stacking retry budgets, and unrecovered failures retain incomplete analysis with sanitized diagnostics ([#555](https://github.com/NVIDIA/SkillSpector/pull/555)).
- Bound local MCP Registry JSON and suppression-baseline YAML/JSON before expansion, reject non-regular input files, and limit bytes, nesting, and records. The Pi extension resolves an explicit installed binary and stages report writes before atomically publishing within the workspace, preserving diagnostic reports where available ([#562](https://github.com/NVIDIA/SkillSpector/pull/562)).
- Constrain prose-oriented static patterns to a single paragraph so unrelated text across blank lines does not combine into a finding; executable and structured-code patterns retain multiline matching ([#491](https://github.com/NVIDIA/SkillSpector/pull/491)).
- Detect imported-module namespace access through `__dict__` or `vars(module)` using subscripts, `get`, `setdefault`, and `pop`: dynamic keys produce AST7 and dangerous literal names produce AST9 ([#517](https://github.com/NVIDIA/SkillSpector/pull/517)).
- Inspect `.venv`, `venv`, and `.tox` for shipped `.pyc`/`.pyo` files under existing traversal limits so Python environments cannot silently hide SC8 bytecode findings ([#571](https://github.com/NVIDIA/SkillSpector/pull/571)).
- Active hook declarations whose payload data flow remains unmodeled now retain the BH1 mechanism finding and record partial `opaque_content` coverage, making completeness false and preventing a `SAFE` recommendation ([#573](https://github.com/NVIDIA/SkillSpector/pull/573)).
- Letter-spacing reconstruction no longer treats a logical line break as the start of a token run, so CRLF or punctuation on the previous heading cannot turn identifier-adjacent text into false P3/P4 findings; AE6 still records the ambiguous form ([#564](https://github.com/NVIDIA/SkillSpector/pull/564)).
- Recursive `--format sarif` scans without `--output` now emit the merged SARIF log to standard output. SARIF advisories, progress, verbose status, and transitive warnings stay on standard error so the output remains parseable ([#565](https://github.com/NVIDIA/SkillSpector/pull/565)).
- GitHub and GitLab `/blob/` file URLs are rewritten to their raw-file forms before download, preventing scans of forge HTML in place of the requested file ([#566](https://github.com/NVIDIA/SkillSpector/pull/566)).
- Retarget dynamic workflow deadlines on existing OpenAI, Anthropic, and agent-CLI clients instead of constructing a new client per call, preventing connection-pool churn and closed-event-loop cleanup errors while preserving retry and concurrency behavior ([#520](https://github.com/NVIDIA/SkillSpector/pull/520)).
- Preserve whole-document Markdown ownership when recovering commands from validated JSON strings so fenced or literal content is not reinterpreted as standalone Markdown and unresolved commands cannot become a clean result through unrelated JSON adjacency ([#559](https://github.com/NVIDIA/SkillSpector/pull/559)).
- Inventory normally excluded executable and loadable content, retain root-coverage evidence, resolve explicit extensionless command paths, and inspect excluded ZIP-family containers within existing limits. Referenced executable exclusions or incomplete excluded-artifact inspection now emit HIGH SC9/incomplete evidence with the existing minimum score of 51 (`DO_NOT_INSTALL`); the narrow exception for direct, non-binary `.git/hooks/*.sample` files remains ([#548](https://github.com/NVIDIA/SkillSpector/pull/548)).
- Expand Windows 8.3 short-name components before comparing a requested path with its opened handle, allowing valid paths under spaced profile directories while retaining reparse-point and fail-closed checks ([#484](https://github.com/NVIDIA/SkillSpector/pull/484)).
- Avoid false AE1 results from valid JSON placeholders, inline code, list and blockquote containers, indented JSON, Markdown tables, and literal Make syntax ([#516](https://github.com/NVIDIA/SkillSpector/pull/516)).
- Bound JSON quote traversal without repeatedly scanning overlapping suffixes ([#521](https://github.com/NVIDIA/SkillSpector/pull/521)).
- Emit a HIGH AE7 analysis-evasion finding for per-file size limits that leave an artifact partially inspected, unless AE1 already covers that path. Supply a bounded text prefix and an explicit unreviewed-region marker to enabled LLM analysis; the unread region remains incomplete ([#509](https://github.com/NVIDIA/SkillSpector/pull/509)).
- Preserve full-evidence fingerprints, concrete YARA rule identity, and precise occurrence locations through projection, deduplication, and report compaction. Separate findings are retained while duplicate projections of the same occurrence are collapsed ([#409](https://github.com/NVIDIA/SkillSpector/pull/409)).
- Add bounded reconstruction for letter-spaced P3/P4 prompt instructions, retain source evidence, and use AE6 as an incomplete-coverage fallback for covered ambiguous reconstructions; the later multiline work extends this accounting without making it a universal decoder ([#470](https://github.com/NVIDIA/SkillSpector/pull/470), [#563](https://github.com/NVIDIA/SkillSpector/pull/563)).
- Discover skills inside dot-prefixed directories, retain inherited local-only restrictions for child skills, and partition transitive scan caching by those privacy restrictions ([#410](https://github.com/NVIDIA/SkillSpector/pull/410)).
- Preserve incomplete discovery and requested semantic-analysis failures, including unavailable providers and mixed success/failure telemetry, instead of allowing a complete scan result ([#410](https://github.com/NVIDIA/SkillSpector/pull/410)).
- Emit recursive JSON reports to standard output when no output path is provided ([#467](https://github.com/NVIDIA/SkillSpector/pull/467)).
- Use the project manifest version for RP3 analysis ([#474](https://github.com/NVIDIA/SkillSpector/pull/474)).
- Prefer exact known-package matches when evaluating SC6 package-name similarity ([#530](https://github.com/NVIDIA/SkillSpector/pull/530)).
- Avoid treating slash-separated prose as local file references ([#451](https://github.com/NVIDIA/SkillSpector/pull/451)).
- Reduce false-positive severity for companion CLI OAuth results and signed self-update documentation classified as benign, and provide contextual explanations for warned pipe-to-shell installers. The contextual classifiers have unresolved PE3 and RA1 fail-open paths (see Known Limitations) ([#547](https://github.com/NVIDIA/SkillSpector/pull/547)).
- Ignore literal AS3 references to the current skill only with trusted selected-source identity, preserving that identity across local, repository, and archive materialization. Manifest names corroborate that identity but do not independently authorize suppression; ambiguous sources, peer-skill paths, and obfuscated access remain reportable ([#506](https://github.com/NVIDIA/SkillSpector/pull/506), [#580](https://github.com/NVIDIA/SkillSpector/pull/580)).
- Skip symlink test cases when the platform refuses symlink creation ([#501](https://github.com/NVIDIA/SkillSpector/pull/501)).

## Testing and Portability

- Add acceptance regressions for self/existing references and distinct missing/ambiguous outcomes; these add test coverage rather than new runtime behavior ([#551](https://github.com/NVIDIA/SkillSpector/pull/551)).
- Make the Basic-auth test fixture explicitly synthetic ([#600](https://github.com/NVIDIA/SkillSpector/pull/600)).
- Make secure-open, FIFO, newline-sensitive build-context, nested OMS, and non-ASCII YARA fixtures deterministic across Windows and non-POSIX environments ([#503](https://github.com/NVIDIA/SkillSpector/pull/503), [#502](https://github.com/NVIDIA/SkillSpector/pull/502), [#505](https://github.com/NVIDIA/SkillSpector/pull/505), [#518](https://github.com/NVIDIA/SkillSpector/pull/518), [#504](https://github.com/NVIDIA/SkillSpector/pull/504)).
- Add dependency-free Node 22+ tests for the OpenCode tool helpers and an exact-head OpenCode TypeScript CI job ([#537](https://github.com/NVIDIA/SkillSpector/pull/537)).

## Security

- Recognized unsupported required bytes produce failed execution and CLI exit 2, while covered unresolved reconstruction produces partial coverage, an incomplete-report caveat, and MCP installation blocking. Successful semantic analysis or truncated ledger details do not erase those outcomes.
- Dependency-source SC10 findings remain authoritative through optional LLM filtering; intentional private endpoints are still policy-visible and require review. Nested report evidence and provenance use credential sanitization.
- Registry-load failures, recursive omissions/failures, and description-analysis budget exhaustion remain visible as incomplete coverage. The narrow passive-PNG AE1 exception does not claim semantic inspection or permit MCP installation of an incomplete scan.
- Local registry files are limited to 16 MiB, 64 nesting levels, and 10,000 combined server/package/remote records. Baselines are limited to 2 MiB, 64 nesting levels, and 10,000 rules/fingerprints, with additional bounds on YAML nodes, scalars, and alias expansion.
- Pi extension reports must remain within the workspace. Staged replacement avoids writing through existing hard links; symlink and non-file destinations are rejected. This changes the Pi extension's output-path and executable-discovery requirements.
- Agent CLI subprocesses also remove Anthropic proxy and SkillSpector API credentials from inherited environment variables ([#562](https://github.com/NVIDIA/SkillSpector/pull/562)).
- Active hook payloads whose data flow is not modeled remain visible through BH1 and now make analysis partial with `opaque_content`, preventing a complete or `SAFE` result for those hooks.
- Excluded executable or loadable content is inventoried before exclusion. Referenced or out-of-coverage bytes now produce SC9 and incomplete-analysis evidence that blocks strict installation gates; the narrow exception is limited to direct, non-binary `.git/hooks/*.sample` files.
- Multilingual batch analysis consumes only the validated provider-eligible snapshot, so language detection and gap-fill do not reread local-only or replaced path content after the core scan.
- OpenCode integration is static by default. The native tool redacts common secret forms and bounds output, while the semantic provider uses argv/stdin and treats missing authentication, empty output, or unsupported event streams as failures.
- Genuine removal instructions remain reportable. The runtime-command fixes in #514 retain incomplete coverage for covered unresolved forms and fail strict CLI/MCP installation gates, including when semantic analysis succeeds. Reconstruction remains bounded rather than a complete shell interpreter.
- JSON string ownership preserves source evidence and does not exempt string contents from analysis.
- Findings and exit status can change after upgrading: oversized files can add HIGH AE7 findings, letter-spaced instructions can produce P3/P4 or AE6 findings, and previously collapsed distinct matches can increase the retained finding count and risk score. Missing requested analysis remains incomplete even when static analysis finishes.
- Context-aware companion CLI classification can lower severity, scores, or recommendations for documentation classified as benign. Coverage is not fail-closed for every token-transfer or self-update phrasing: the known PE3 and RA1 exceptions below can be incorrectly downgraded or missed. Warned pipe-to-shell installer findings remain reportable.
- Literal current-skill references backed by trusted selected-source identity no longer produce AS3 findings; manifest text alone cannot exempt a peer-skill reference. Transformed or obfuscated paths, explicit enumeration, AS1, and AS2 remain reportable.

## Breaking Changes and Migration

- Previously accepted unsupported primary files or invalid required instruction encodings can now fail with `unsupported_primary_content` and CLI exit 2 regardless of strict flags. Use supported UTF-8 instructions or supported ZIP inputs; required identity also applies inside bounded nested archives.
- Runtime-selected commands, multiline prompt ambiguity, and over-limit Markdown references can now change a complete/`SAFE` result to partial/`CAUTION`, `--fail-on-incomplete` exit 1, and MCP `safe_to_install=false`. Default CLI exit 0 is not proof of completeness; AE6 marks unresolved interpretation rather than confirmed semantic wrongdoing.
- AE1 consumers should tolerate the additive `Incomplete referenced artifact analysis` pattern and bounded evidence fields such as `target_path`, `target_disposition`, `reasons`, and `reasons_truncated`. Proven literal Perl help text can lose false AE1 findings without exempting its payload or ambiguous source from analysis.
- Report consumers should tolerate additive `metadata.llm_provenance` fields and ledger reasons such as `analyzer_load_error` and `transitive_child_scan_failed`, preserve aggregate omission/failure evidence, and handle JSON zero-based versus SARIF one-based Unicode-codepoint columns.
- LLM seeds must fit a signed 64-bit integer. Provenance distinguishes requested settings from observed forwarded parameters and is not a reproducibility guarantee; source revision may be unknown unless packaged or supplied with `SKILLSPECTOR_BUILD_REVISION`.
- SC10 can flag intentional private registries as HIGH; review the source and use the existing baseline workflow for accepted cases. New trigger, shell-flag, reflective, XOR, and bytecode detection and corrected GHSA severity can increase findings or scores. False Rust ROP matches and narrowly verified passive-PNG AE1 findings can disappear without erasing coverage caveats.
- GitHub tree-directory selection performs advertised-ref lookup within the ingestion deadline. Compact prompt numbering is opt-in and does not change the default format or reported source line numbers.
- Interactive non-verbose scans now show progress and discovered files on standard error. Machine-readable JSON and SARIF remain clean on standard output; use `--verbose` to retain the non-streamed diagnostic path.
- Scanning GitHub or GitLab `/blob/` links now analyzes raw file bytes rather than the forge viewer page. Findings and recommendations can change because the intended content is finally scanned.
- Active hooks with unmodeled payload flows now produce incomplete coverage and cannot remain `SAFE`; identifier-adjacent letter spacing can lose false P3/P4 findings while retaining AE6.
- `opencode_cli` is opt-in and requires an authenticated OpenCode 1.18.31 executable. Users of earlier 2.12.0 candidates pinned to 1.18.30 must update OpenCode; all other versions fail closed. The OpenCode-native tool is installed by copying `.opencode/` from a checkout and defaults to static analysis; set `noLlm=false` and configure the provider environment to request semantic analysis.
- Pi extension users must install SkillSpector in the extension's `.venv` or set `SKILLSPECTOR_BIN` to an existing absolute executable path; ambient `PATH` lookup is no longer used. Move report outputs into the current workspace. These restrictions apply to the Pi extension, not the standalone CLI or OpenCode tool.
- Oversized or over-complex local registry and baseline files must be reduced to the documented bounds. Consumers of ledger reason codes should recognize `reference_missing` separately from `reference_unresolved`; `safe_to_install` may now be true when missing references are the only completeness caveat.
- TP4 checks can add semantic work and findings for accepted Markdown fences; AST7/AST9 and SC8 can add findings for reflective lookups and bytecode in Python environments. Paragraph-boundary fixes can remove false prose matches.
- Transient provider failures can now incur bounded additional requests and backoff within the workflow deadline; live-provider cost and latency depend on the configured service.
- Existing CLI-provider deployments that set `SKILLSPECTOR_MODEL_REGISTRY` now use its valid token budgets. Invalid or non-positive values warn and fall back instead of aborting.
- Scans that previously treated excluded executable content as clean can now become incomplete with SC9 and `DO_NOT_INSTALL`; consumers should retain completeness and exclusion evidence.
- The static-analysis allowance needs no new configuration. Static analysis can now run longer within the existing workflow deadline. Set `SKILLSPECTOR_MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT=30` to retain the previous per-artifact allowance, and restart the SkillSpector process after changing the setting ([#522](https://github.com/NVIDIA/SkillSpector/pull/522)).
- `--fail-on-findings` is opt-in. Combine it with `--fail-on-incomplete` when CI must reject either active findings or incomplete coverage. Code 2 still denotes an input or execution error; a nonzero exit alone does not identify which condition occurred.
- Consumers should retain completeness/degradation metadata and inspect findings as well as exit status. Explicitly use `--no-llm` for an intended static-only scan; requesting LLM analysis without an available provider is incomplete.
- Third-party dependency versions are unchanged from 2.11.2.

## Deprecations

- None.

## Validation

The release catalog covers 69 merged PRs since v2.11.2 and is synchronized through main commit `05119f4b868aafc7a6347043d3c3b8f9ac548091`. Local release checks use Python 3.12.13:

- `uv sync --locked --all-extras --python 3.12` installed the locked dependency set without changing the lockfile.
- Ruff lint and format checks passed for all source and test files.
- The CLI reported `SkillSpector v2.12.0`; the release helper dry run resolved `v2.12.0` and the matching versioned notes.
- All 65 OpenCode and Pi extension JavaScript/TypeScript tests passed.
- All 95 selected static integration cases for opaque-reference reporting and wrapped/canonical dependency-source behavior passed without live providers.
- Wheel and source distributions built successfully, and Twine validated both artifacts.
- `git diff --check` passed.

The release PR records the full Python test results and hosted checks for the current commit. Earlier test counts and hosted runs apply only to their recorded commits. Deployment/provider validation and the separate release gates below remain pending.

[Release PR #550](https://github.com/NVIDIA/SkillSpector/pull/550) records the candidate baseline, validation results, known gaps, and remaining release gates.

## Known Limitations

- Local sanity checks cover the tested inputs and environment; live provider and deployment behavior depend on their configuration.
- Incomplete inspection is a reportable result. Unsupported inputs, unavailable requested analysis, and resource limits must remain visible; these conditions cannot be treated as a clean scan.
- The release remains a candidate while outstanding review and release issues are assessed. Proposed fixes in unmerged PRs are not included in this candidate.
- Reconstruction remains bounded: the multiline projection deliberately preserves blank paragraphs, punctuation, ordinary multi-character tokens, and wider gaps. It is not a universal obfuscation decoder or full shell/Perl interpreter; recorded resource or parser limits remain incomplete coverage.
- Companion-context classification is not fail-closed. PE3 can downgrade imperative token-acquisition text and miss adjacent disclosure phrased with verbs such as `paste`; RA1 can accept protected agent/tool names with CLI suffixes and signed-release evidence from a different logical line. These variants can be incorrectly downgraded or missed.
- Required-input recognition is conservative rather than a universal encoding or file-format detector. Unrecognized representations whose bytes appear to be UTF-8 can remain text; supported/empty ZIP completeness does not prove a usable or harmless skill. The fixes in #514, #553, and #563 are included, but no broader language or format coverage is implied.
- `opencode_cli` currently reports no token-usage accounting, and model availability or rate limits remain external. Multilingual batch gap-fill was validated with mocked providers; live-provider qualification remains pending.

## References

- [Candidate changes since v2.11.2](https://github.com/NVIDIA/SkillSpector/compare/v2.11.2...05119f4b868aafc7a6347043d3c3b8f9ac548091)
- [AE1 documentation fix #516](https://github.com/NVIDIA/SkillSpector/pull/516)
- [Static analysis time allowance #522](https://github.com/NVIDIA/SkillSpector/pull/522)
- [OpenCode CLI provider #536](https://github.com/NVIDIA/SkillSpector/pull/536)
- [OpenCode-native integration #537](https://github.com/NVIDIA/SkillSpector/pull/537)
- [Excluded executable coverage #548](https://github.com/NVIDIA/SkillSpector/pull/548)
- [Gemini registry and CLI progress #7](https://github.com/NVIDIA/SkillSpector/pull/7)
- [Recursive SARIF stdout #565](https://github.com/NVIDIA/SkillSpector/pull/565)
- [Raw forge file URLs #566](https://github.com/NVIDIA/SkillSpector/pull/566)
- [Security-view performance #569](https://github.com/NVIDIA/SkillSpector/pull/569), [#570](https://github.com/NVIDIA/SkillSpector/pull/570)
- [Fail-closed hook payload coverage #573](https://github.com/NVIDIA/SkillSpector/pull/573)
- [Local input and Pi extension hardening #562](https://github.com/NVIDIA/SkillSpector/pull/562)
- [MCP missing-reference handling #526](https://github.com/NVIDIA/SkillSpector/pull/526)
- [Markdown fence analysis #421](https://github.com/NVIDIA/SkillSpector/pull/421)
- [Transient provider retries #555](https://github.com/NVIDIA/SkillSpector/pull/555)
- [Recursive failure reporting #576](https://github.com/NVIDIA/SkillSpector/pull/576)
- [LLM provenance #556](https://github.com/NVIDIA/SkillSpector/pull/556)
- [Dependency-source analysis #383](https://github.com/NVIDIA/SkillSpector/pull/383)
- [Occurrence columns #584](https://github.com/NVIDIA/SkillSpector/pull/584)
- [Passive-image coverage distinction #597](https://github.com/NVIDIA/SkillSpector/pull/597)
- [Runtime-command completeness #514](https://github.com/NVIDIA/SkillSpector/pull/514)
- [Markdown destinations #553](https://github.com/NVIDIA/SkillSpector/pull/553)
- [Required-input and multiline completeness #563](https://github.com/NVIDIA/SkillSpector/pull/563)
- [Perl ownership and referenced-artifact diagnostics #615](https://github.com/NVIDIA/SkillSpector/pull/615)

Prepared by Codex for Mohit Gupta.
