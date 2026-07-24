### 2.4.4 (Thursday, July 23, 2026)
### Features/Bug Fixes
* fix(anthropic): re-apply ANTHROPIC_BASE_URL override reverted by 2.4.3 snapshot (#301)
---
### 2.4.3 (Wednesday, July 22, 2026)
### Features/Bug Fixes
* Clarify CLI runtime model fallback in provider docs
* fix(provider): align Claude fallback contract with settings isolation (#295)
* fix(provider): isolate Claude settings hooks in spawned CLI (#295)
* fix(suppression): match reported finding text
* ci: disable optional provider test
* feat: publish a public-safe changelog
---
### 2.4.2 (Tuesday, July 21, 2026)
### Features/Bug Fixes
* fix(oss): keep internal provider references private
---
### 2.4.1 (Monday, July 20, 2026)
### Features/Bug Fixes
* fix(provider): keep reasoning effort pass-through consistent (#283)
* feat(provider): keep reasoning effort consistent across Anthropic paths (#283)
* feat(provider): forward reasoning effort to OpenAI-compatible models (#283)
* fix(analyzer): align file-size guard with character semantics (#284)
---
### 2.4.0 (Monday, July 20, 2026)
### Features/Bug Fixes
* fix(analyzer): reduce cupynumeric false positives
* fix(analyzer): reduce security-pattern false positives
* fix(analyzer): scope passwd mount and rm detection
---
### 2.3.13 (Tuesday, July 14, 2026)
### Features/Bug Fixes
* fix: mask release command failures
* ci: validate default branch pushes
* Fix Sonar finding in YARA rule materialization
* feat(provider): allow scoped LLM provider injection (#243)
* fix emoji zwj prompt injection false positive
* fix(analyzer): keep executable doc calls outside suppression (#251)
* fix(analyzer): keep inline block comments out of doc gating (#251)
* fix(analyzer): classify docs from the finding line (#251)
* fix(analyzer): keep config-file findings outside doc gating (#251)
* fix(analyzer): gate documentation false positives for PE3/RA1/TM1/AR2 (#251)
* fix(cli): preserve full per-skill JSON payload in recursive scans (#228)
* fix(yara): skip malformed unicode encoded rules (#236)
* fix(yara): reduce packaged malware-signature false positives (#236)
* fix(sc7): exclude --disable-content-trust=false to keep content-trust-enabled pulls clean
* fix(analyzer): rely on runner for SC7 example filtering to close executable bypass
* feat(analyzer): detect untrusted container image pull as SC7
* fix(report): preserve exact SARIF severity metadata (#229)
* fix(report): preserve remaining SARIF finding fields (#229)
* fix(report): preserve full finding metadata in SARIF output (#229)
* Format: ruff lint and format fixes
* Add unit tests for run_async utility function
* Fix: remove unused asyncio import from meta_analyzer.py
* Fix: Allow running in environments with existing event loop
---
### 2.3.12 (Monday, July 13, 2026)
### Features/Bug Fixes
* fix: mask release command secrets
* docs: correct MCP fixture expectations
* fix(mcp): prove stdio initialize compatibility (#199)
* fix: trim batch scan README command whitespace
* rename contrib/multilingual to contrib/batch_scan and update README usage
* ci: align GitHub CI with deterministic checks
---
### 2.3.11 (Monday, July 06, 2026)
### Features/Bug Fixes
---
### 2.3.10 (Monday, July 06, 2026)
### Features/Bug Fixes
* refactor: centralize cleanup and risk threshold
* docs: finalize PR #100 review — docs, tests, world-class polish
* fix: wire ApiKeyPool into llm_analyzer_base graph path
* fix: add SPDX headers, from __future__ annotations, conftest.py to all test files - Add SPDX license header to 8 test files - Add from __future__ import annotations to 8 test files - Fix Unicode stdout crash in test_pool_wiring.py on Windows - Add conftest.py with pytest markers registration - 120 tests passing Co-Authored-By: Claude <noreply@anthropic.com>
* docs: reorganize into core guides and process archive
* docs: add CONTRIBUTING guide, rejected alternatives, gap-fill selection criteria
* fix: add Windows Unicode stdout support for CJK output
* fix: add SPDX headers, cross-platform cleanup, and comprehensive documentation
* docs: organize documentation, translate to English, add NVIDIA convention audit
* fix: suppress asyncio noise, sanitize meta-analyzer output quirks
* fix: resolve LLM race condition, JSON parsing, and connection timeout
* add contrib multilingual batch scanner
---
### 2.3.9 (Tuesday, June 30, 2026)
### Features/Bug Fixes
* test: restore LLM-backed graph integration coverage
* test: keep graph integration scans offline
* style: format MCP least-privilege analyzer
* docs: correct stale analyzer status and dangling references
* feat(providers): local agent-CLI providers (claude/codex/gemini), no API key
* feat(ossf-scorecard): add ossf-scorecard github action integration
* fix(mcp): feed allowed-tools into LP1 under-declaration check
* fix(mcp): treat allowed-tools as a permission declaration for LP3
* test(input): add SSRF gate coverage for scp-extracted hosts
* fix(cli): preserve empty string from _result_body when sarif_report absent
* Support Python 3.14
* feat(analyzer): detect privileged Kubernetes workload deployment as TM4
* test(input): clarify scp_private_ip test covers allowlist gate
* fix(cli): write concatenated multi-skill report to --output for non-JSON formats
* fix(input): support scp-style SSH Git URLs in host validation
---
### 2.3.8 (Monday, June 29, 2026)
### Features/Bug Fixes
* style: fix merge-ref lint failures
* style: format chat model provider warning
* fix: address non-blocking reviewer nits from #178 and #179
* revert: restore provider CI failure policy
* ci: make live provider validation non-blocking
* style: complete GitHub PR 194 formatting for PR 125
* style: complete GitHub PR 194 formatting for PR 122
* style: apply GitHub PR 194 lint fix to PR 178 import
* style: apply GitHub PR 194 lint fix to PR 172 import
* style: apply GitHub PR 194 lint fix to PR 125 import
* style: apply GitHub PR 194 lint fix to PR 122 import
* feat: add AWS Bedrock provider for Claude via SigV4
* fix: address non-blocking reviewer nits from #140, #141, #143
* feat(analyzer): detect cloud-storage exfiltration as E5
* docs(mcp): clarify setup before users choose stdio
* feat(analyzer): detect privileged container execution and escape primitives as PE5
* docs(mcp): document HTTP transport trust model
* fix(report): strip ANSI/control bytes from report output
* fix(behavioral): detect builtins.* and importlib.import_module sink evasions
* feat: per-slot model env overrides and model validation
* fix(P2): narrow emoji tag carve-out to ISO-3166-2 codes (close smuggling bypass)
* fix(P2): detect Unicode Tag-block "ASCII smuggling" hidden instructions
* feat(analyzer): implement MCP rug-pull detection (RP1-RP3)
* fix(scoring): apply 1.3x multiplier only to findings from executable files
* feat(scripts): add PR review agent automation tooling
---
### 2.3.7 (Wednesday, June 24, 2026)
### Features/Bug Fixes
---
### 2.3.6 (Wednesday, June 24, 2026)
### Features/Bug Fixes
* feat(analyzer): detect SSRF (cloud metadata, internal-network, dynamic-host requests)
* feat(analyzer): add anti-refusal statement detection (AR1-AR3)
* address review feedback on #106
* feat(report): add baseline / false-positive suppression
* style: format meta analyzer regression test
* test: align meta analyzer drop cases with severity floor
* style: format static runner filtering changes
* style: format MP2 regex backtracking test
* Fix Windows path separators and console encoding
* fix(llm): isolate batch failures in Stage 2 and keep unanalysed findings
* test(scoring): add regression test for input-order-dependent severity sort
* fix(scoring): document confidence scaling, sort by severity within rule bucket
* fix(patterns): fix lint and whitespace-bearing stuffing false negative
* fix(patterns): skip single-char repetitions in MP2 to avoid separator false positives
* fix(patterns): anchor MP2 regex to prevent catastrophic backtracking
* ci: fix DCO check bypass and harden the CI workflow
* ci: add GitHub Actions CI/CD workflow
* fix(static-runner): remove .svg from binary extensions
* fix(static-runner): exempt SKILL.md from PE3 .env doc filter
* fix(static-runner): skip binary/PDF files and filter PE3 .env doc references
* fix(security)(skillspector): unsafe deserialization via yaml load
* fix(security)(skillspector): potential information disclosure via error message
* fix(analyzer): deduplicate PE4 findings per line to avoid double-reporting
* feat(analyzer): detect Docker socket access as PE4 privilege escalation
* feat(mcp): expose SkillSpector as an MCP server with scan_skill tool
* test(meta_analyzer): add regression tests for static findings with end_line=None
* fix(supply_chain): scan [build-system].requires in pyproject.toml
* security(meta_analyzer): add severity-gated floor to apply_filter
* chore(oss): exclude changelog from public snapshots
---
### 2.3.5 (Tuesday, June 23, 2026)
### Features/Bug Fixes
* test: align agent snooping same-line expectation
* test: pin nv_build provider default expectation
* style: format behavioral AST getattr detection
* style: format input handler SSRF changes
* test: remove unused sarif pytest import
* style: format meta analyzer fallback tests
* test: avoid duplicate agent snooping test class name
* feat(report): add analysis_completeness field to JSON output
* fix(schemas): normalize confidence from 0-100 scale before Pydantic validation
* chore: add perseus-ctx and mimir-mcp to popular PyPI packages
* feat(pi): add SkillSpector scan tool
* fix(static-patterns): restrict code-example hard-drop to non-executable files
* fix(multi-skill): address review nits - typing, dead code, help text, findings source
* fix(dedup): apply deduplication to score computation only, preserve all findings in report
* feat: support uv tool install and document in README
* fix(behavioral-ast): detect reflective exec via getattr() literal (AST9)
* fix(input-handler): disable HTTP redirect following to close SSRF bypass
* fix(report): filter empty LLM findings and add SARIF rules[] array
* fix(meta-analyzer): add severity floor, downweight instead of drop, fail-closed on LLM error
* fix(static-patterns): filter false positives from documentation and code examples
* feat(cli): add --recursive flag for multi-skill directory scanning
* fix(findings): deduplicate cross-analyzer findings before scoring
* fix(input-handler): validate git/download URLs against SSRF and add zip-slip protection
* fix(meta-analyzer): add heuristic fallback filter for --no-llm mode
* docs: document the integration contract and trust model
* fix(supply-chain): exclude pyproject metadata keys from dependency extraction
* feat: implement MCP rug pull analyzer and unit tests
* fix(sc4): pass version to OSV for all requirement operators, not just == and <=
* fix: use OpenAI default model for OpenAI fallback
* feat(analyzer): detect skills snooping on the agent ecosystem
* docs: correct stale analyzer status and dangling references
---
### 2.3.4 (Tuesday, June 23, 2026)
### Features/Bug Fixes
* Revert "Merge branch 'keshavp/codex/revert-mr-43' into 'main'"
---
### 2.3.3 (Tuesday, June 23, 2026)
### Features/Bug Fixes
* Revert "Merge branch 'github/pr-119' into 'main'"
---
### 2.3.2 (Monday, June 22, 2026)
### Features/Bug Fixes
* feat(release): auto-generate CHANGELOG.md on each release
* style: format lint fixes for PR 156
* fix(yara): use content hash for rule cache invalidation
---
### 2.3.1 (Monday, June 22, 2026)
### Features/Bug Fixes
* fix(scoring): prevent risk score saturation via per-rule diminishing returns
* fix(meta-analyzer): keep LLM-confirmed findings when model returns end_line
* add openai project header
* fix(yara): reduce remote bootstrap false positives
* feat(yara): add agent skill abuse signatures
---
### 2.3.0 (Monday, June 22, 2026)
### Features/Bug Fixes
* style: format OSV fallback changes
* style: format agent snooping analyzer
* style: format taint tracking tests
* style: format supply chain analyzer
* fix: avoid literal bidi controls in tests
* style: format anthropic proxy provider
* fix: reduce anthropic proxy sonar duplication
* feat: drop ge/le schema bounds on LLM finding confidence and start_line
* fix(build_context): use forward-slash component paths (cross-platform)
* fix(sc4): add global _last_query_ok declaration, validate env var, derive fallback count
* fix(sc4): surface OSV.dev fallback warnings and add configurable timeout
* fix(supply-chain): require relative edit distance for SC6 typosquat detection
* feat(analyzer): add agent snooping detector (AS1/AS2/AS3)
* fix(P2): add bidi control character detection (CVE-2021-42574 / Trojan Source)
* fix(meta_analyzer): parse stringified findings array from LLM
* fix(mcp): anchor TP3 loopback URL exemption to a host boundary
* fix(analyzers): resolve import aliases in AST and taint analyzers
* fix: validate trusted source hosts for SC2
* fix: restrict Python version to <3.14 due to jsonschema-rs/PyO3 incompatibility
* feat(provider): add anthropic_proxy provider for Vertex-style raw-predict endpoints
---
### 2.2.3 (Tuesday, June 16, 2026)
### Features/Bug Fixes
* chore: refresh uv lock for python 3.14
---
### 2.2.2 (Tuesday, June 16, 2026)
### Features/Bug Fixes
* chore: widen python range to <3.15 and bump version to 2.2.1
---
### 2.2.0 (Tuesday, June 16, 2026)
### Features/Bug Fixes
* Fixing â Release failed: uv.lock exists, but  is not installed or is not on PATH
* Create native LangChain chat models per provider
---
### 2.1.5 (Monday, June 15, 2026)
### Features/Bug Fixes
* Revert "test: preserve default graph invocation in PR 45 import"
* test: preserve default graph invocation in PR 45 import
* Reject invalid skill paths
* fix: add explicit returns in docker smoke test functions
* docs: fix model registry path
* ci: extract Docker smoke suite
* ci: add Docker GitHub URL smoke test
* fix(docker): install git for repository scans
---
### 2.1.4 (Saturday, June 13, 2026)
### Features/Bug Fixes
* ci: add Docker smoke test
* chore: add Docker build ignore file
* docs: simplify Docker usage examples
* fix: use official Python Docker base
* feat: adds dockerfile to run it without installing python
---
### 2.1.3 (Wednesday, June 10, 2026)
### Features/Bug Fixes
* Revert "chore: bump version to 2.1.3"
* Constrain supported Python versions
* Fix uv venv py-version
* fix: refresh uv lock during release
* Add contribution flow diagrams
* Make contribution sync flows explicit
* Remove copy-pr-bot references
* Clarify external PR import docs
* Reorganize GitHub release docs
* Add GitHub PR import skill
---
### 2.1.2 (Tuesday, June 09, 2026)
### Features/Bug Fixes
* Revert "chore: bump version to 2.1.2"
* fix(mcp): make TP3 (and parameter-scoped TP1/TP2) reachable on real scans
* Add SkillSpector GitHub release skill
---
### 2.1.1 (Thursday, June 04, 2026)
### Features/Bug Fixes
* Revert "chore: bump version to 2.1.1"
* Enforce non-mutating lint checks in CI
---
### 2.1.0 (Thursday, June 04, 2026)
### Features/Bug Fixes
* Skip eval dataset prose in static scans
* chore: add security policy
* chore: drop guardrail integration files
* chore(oss): strip OSS_RELEASE.md and the release script from snapshots
* chore(oss): switch release script to orphan branch
* Revert "docs(cli): drop nv_inference from scan --help"
* docs(cli): drop nv_inference from scan --help
* docs(oss): sanitize internal references from user-facing files
* chore(oss): drop broken make typecheck target
---
### 2.0.0 (Thursday, May 07, 2026)
### Features/Bug Fixes
* test(oss): mark SDI fixture tests as integration; fix nv_inference detection
* docs(oss): trim OSS_RELEASE.md to the how-to section only
* chore(oss): rename make-public.sh to create-oss-release.sh, auto-name + pull main
* chore(oss): split Makefile + consolidate internal-only files
* feat(providers): selectable provider + per-provider model defaults
* refactor(providers): per-package layout with bundled YAML registries
* chore: remove agent metadata from OSS config
* refactor(providers): isolate NVIDIA-specific code behind a single registration
* chore(oss): prepare branch for public OSS release
* feat(llm): generalize credential resolution for OSS-default endpoints
* refactor(metadata): introduce ModelMetadataProvider abstraction
* feat(tracing): support LANGCHAIN_TAGS_EXTRA env var for LangSmith tags
---
### 1.5.0 (Friday, May 01, 2026)
### Features/Bug Fixes
* feat(tracing): support LANGCHAIN_TAGS_EXTRA env var for LangSmith tags
---
### 1.4.0 (Tuesday, April 28, 2026)
### Features/Bug Fixes
* feat(mcp): MCP analyzers, Apache 2.0 license migration, and OSS compliance
---
### 1.3.0 (Friday, April 24, 2026)
### Features/Bug Fixes
* LangSmith Tracing + Integration Test Fixes
---
### 1.2.0 (Monday, April 06, 2026)
### Features/Bug Fixes
* docs(mcp): address review nitpicks on B.3.1 and B.3.2 docs
* docs(mcp): add detailed documentation for B.3.1 and B.3.2 analyzers
* fix(mcp): move noqa directive to correct line for ruff S603 suppression
* fix(mcp): address CodeRabbit review feedback
* test(mcp): add full-pipeline integration tests for SARIF and end-to-end
* feat(mcp): implement B.3.2 TP4 LLM description-behavior mismatch
* feat(mcp): implement B.3.2 TP1-TP3 static metadata poisoning detection
* feat(mcp): implement B.3.1 mcp_least_privilege (LP1-LP4)
* feat(mcp): add MCP pattern categories, LP/TP rule registry entries, and test fixtures
---
### 1.1.4 (Wednesday, March 25, 2026)
### Features/Bug Fixes
* Detects markdown code blocks (```), code-comment indicators (// â, // â, // GOOD:, // BAD:), and documentation keywords
---
### 1.1.3 (Tuesday, March 24, 2026)
### Features/Bug Fixes
* Reduce false positives for Dockerfile idioms and CI/CD docs
---
### 1.1.2 (Tuesday, March 24, 2026)
### Features/Bug Fixes
* Removed duplicate tests
---
### 1.1.1 (Tuesday, March 24, 2026)
### Features/Bug Fixes
* TM1 (Tool Parameter Abuse) - 19 false positives fixed:
---
### 1.1.0 (Tuesday, March 24, 2026)
### Features/Bug Fixes
* Move skillspector-specific safe patterns and LLM key checks from nv-base into skillspector
---
### 1.0.0 (Thursday, March 19, 2026)
### Features/Bug Fixes
* feat: added yara based analyzer
* feat: implement data-flow analyzer: sources -> sinks
* Implement `semantic_developer_intent` analyzer (SADD B.4.2)
* Replace hardcoded CVE lists with live OSV.dev vulnerability lookups (SC4)
* Implement semantic_security_discovery analyzer (SADD B.4.1)
* Implement `semantic_quality_policy` analyzer (SADD B.4.3) and fix meta_analyzer finding duplication bug
* Implement static analyzers (EA, OH, P6-P8, MP, TM, RA) and extend supply chain (SC4-SC6, TR1-TR3)
---
### 0.3.1 (Friday, March 13, 2026)
### Features/Bug Fixes
* Ignore .claude/
* Revert "chore: bump version to 0.3.1"
* Revert "chore: bump version to 0.3.2"
* feat: LLMAnalyzerBase — reusable base class for LLM-powered analyzer nodes
* feat: implemented analyzer for dangerous execution chains
* Restore dev changes: guardrails, typer compatibility, docs, and finding output shape
* Revert to state at d74cbf9: undo merge keshavp/dev, guardrail update, typer downgrade, docs, finding output
* Update guardrail version
* downgrade typer version for compatibility with nv-base
* docs: clarify venv setup and uv/pip fallback in Makefile and docs
* feat: full finding output shape and Finding model cleanup
* Revert "chore: bump version to 0.4.0"
* add Skillspector v2 LangGraph workflow scaffold
---
### 0.3.0 (Monday, February 09, 2026)
### Features/Bug Fixes
* Replace generic LLM unavailable message with pattern-specific explanations
---
### 0.2.0 (Friday, February 06, 2026)
### Features/Bug Fixes
* Unify LLM access via NVIDIA Inference Hub
* docs: condense RELEASE.md for clarity
* Integration with NV-BASE
---
### 0.1.3 (Friday, January 30, 2026)
### Features/Bug Fixes
* docs: update installation and release management instructions
* chore: add Makefile with development and build targets
* feat: add Poetry auth.toml credential support to release script
* feat: add release script for nv-shared-pypi publishing
* Update GitLab Issues link to new demos space
* Initial commit
* Add all 15 vulnerability patterns and author info
* Initial commit: SkillSpector security scanner for AI agent skills
* Initial commit
