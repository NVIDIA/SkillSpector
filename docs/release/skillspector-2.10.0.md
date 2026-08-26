# SkillSpector v2.10.0

Released: 2026-08-26

## Summary

SkillSpector 2.10.0 expands security coverage across concealed artifacts, referenced skills, structured skill bundles, and external model selection. It also makes incomplete analysis harder to mistake for a clean result, adds localized LLM finding text, and exposes the highest reported issue severity for downstream policy gates.

## Highlights

- Inspect hidden files and ZIP-compatible nested artifacts under cumulative safety bounds, with HIGH SC9 findings for concealed executables and provenance-preserving virtual paths.
- Add opt-in transitive reference scanning with bounded traversal, source provenance, shared budgets, and fail-closed completeness reporting.
- Recognize AISOP/AISP structured skill bundles and render report-only workflow summaries without affecting risk scores.
- Add EA5 detection for external model or provider selection, including silent coding-CLI account switches and top-level model pins.
- Add `SKILLSPECTOR_OUTPUT_LANGUAGE` for human-readable LLM finding text and `risk_assessment.max_issue_severity` for machine-readable policy gates.

## Added

- Add bounded local inspection of hidden and nested ZIP, DOCX, XLSX, and PPTX content without extracting or executing members.
- Add opt-in transitive scanning of supported skill references with `--transitive`, plus `--transitive-depth`, `--transitive-allow-prefix`, and `--transitive-deny-prefix` controls.
- Add structured skill summaries for valid AISOP/AISP bundles across terminal, Markdown, JSON, and SARIF output.
- Add dynamic analyzer discovery and validate risk-score inputs against the registered analyzer set.
- Add EA5 static findings for actionable external model or provider selection.
- Add configurable output-language instructions for discovery analyzers, the meta-analyzer, and MCP tool-poisoning analysis.
- Add `risk_assessment.max_issue_severity`, with `NONE` when no active issue is reported.

## Changed

- Move `langgraph-cli[inmem]` from the base installation to the `langgraph-dev` optional extra; the `dev` extra continues to include it.
- Update the NVIDIA Build default model to a currently served model and declare accurate limits for GLM-5.2.
- Tailor LP1 least-privilege remediation to the scanned manifest type.
- Automatically update eligible pull-request branches after changes land on `main`.

## Fixed

- Mark requested LLM analysis as degraded when any call fails or the configured provider is unavailable, flooring an otherwise `SAFE` recommendation to `CAUTION`.
- Report and baseline only the active findings that actually drove the risk score.
- Normalize serialized multi-skill risk scores before computing aggregate exit codes, with malformed values safely falling back to zero.
- Preserve eligible findings from `SKILL.md` instead of dropping them as code examples.
- Parse `package.json` as JSON for supply-chain analysis and route fatal CLI diagnostics to stderr.
- Detect whitespace-tolerant environment harvesting and all supported `os.environ` read forms.
- Reduce false positives across inactive Git hook samples, license boilerplate, wildcard tool grants, OAuth credential terminology, reference directories, and non-text artifact content.
- Require an operation tied to a keyring or keychain noun before reporting PE3 in Markdown and text prose, while preserving actionable credential-store findings.
- Preserve the original custom CLI-provider call contract for ordinary scans while forwarding explicit deadlines to providers used by bounded scan paths.

## Security

- Strengthen cumulative resource bounds, inspection-ledger completeness, finding provenance, Unicode normalization, and fail-closed behavior across scan paths.
- Keep hidden and nested artifact content local to deterministic analysis and exclude it from LLM prompts.
- Preserve deterministic security findings through filtering, suppression, recursive, transitive, MCP, and report-rendering paths.
- Surface partial provider execution and traversal truncation so incomplete deep scans cannot silently appear clean.

## Breaking Changes and Migration

- No existing CLI command, option, or report field was removed.
- LangGraph Studio users who install only the base package should install `skillspector[langgraph-dev]`; `make install-dev` and the `dev` extra continue to include this tooling.
- Custom CLI providers keep the original `complete(prompt, *, model, max_output_tokens)` contract for ordinary scans. Providers used with new deadline-bounded paths may additionally accept `timeout` as an optional keyword.

## Deprecations

- None.

## Validation

- `uv lock --check` — passed.
- `.venv/bin/pytest -q tests/unit/test_llm_utils.py tests/unit/test_create_github_release.py tests/unit/test_github_release_workflow.py tests/unit/test_wheel_contents.py` — 54 passed.
- `.venv/bin/pytest -m 'not integration and not provider' --cov=src/skillspector --cov-report=term --cov-report=xml tests/` — 2,937 passed, 13 skipped, 38 deselected, and 4 expected failures.
- `.venv/bin/ruff check src/ tests/ scripts/` — passed.
- `.venv/bin/ruff format --check src/ tests/ scripts/` — 196 files already formatted.
- Built `skillspector-2.10.0-py3-none-any.whl` and `skillspector-2.10.0.tar.gz`; `twine check` passed for both distributions.
- `skillspector --version` — reported `SkillSpector v2.10.0`.
- The GitHub release helper dry run resolved tag `v2.10.0` and the matching versioned release notes.
- `git diff --check` — passed.

## Known Limitations

- Transitive scanning remains opt-in and is limited to source types supported by the secure input handler; it is not a general-purpose web crawler and is not enabled for MCP scans.
- Nested inspection is limited to ZIP-compatible containers, enforces fixed cumulative bounds, and does not render, install, or execute nested content.
- `SKILLSPECTOR_OUTPUT_LANGUAGE` affects human-readable LLM-generated finding text only; deterministic findings and machine-readable schema values remain unchanged.
- Legacy custom CLI providers that do not accept `timeout` remain compatible with ordinary scans but cannot participate in a new path that requires an explicit provider deadline until they add that optional keyword.

## References

- [GitHub PR #74](https://github.com/NVIDIA/SkillSpector/pull/74)
- [GitHub PR #211](https://github.com/NVIDIA/SkillSpector/pull/211)
- [GitHub PR #225](https://github.com/NVIDIA/SkillSpector/pull/225)
- [GitHub PR #237](https://github.com/NVIDIA/SkillSpector/pull/237)
- [GitHub PR #291](https://github.com/NVIDIA/SkillSpector/pull/291)
- [GitHub PR #323](https://github.com/NVIDIA/SkillSpector/pull/323)
- [GitHub PR #328](https://github.com/NVIDIA/SkillSpector/pull/328)
- [GitHub commit 1d379dc](https://github.com/NVIDIA/SkillSpector/commit/1d379dca7f8e83f3785aef4954d53c56908d2ad0)
- [GitHub PR #362](https://github.com/NVIDIA/SkillSpector/pull/362)
- [GitHub PR #368](https://github.com/NVIDIA/SkillSpector/pull/368)
- [GitHub PR #375](https://github.com/NVIDIA/SkillSpector/pull/375)
- [GitHub PR #376](https://github.com/NVIDIA/SkillSpector/pull/376)
- [GitHub PR #381](https://github.com/NVIDIA/SkillSpector/pull/381)
- [GitHub PR #382](https://github.com/NVIDIA/SkillSpector/pull/382)
- [GitHub PR #390](https://github.com/NVIDIA/SkillSpector/pull/390)
- [GitHub PR #391](https://github.com/NVIDIA/SkillSpector/pull/391)
- [GitHub PR #393](https://github.com/NVIDIA/SkillSpector/pull/393)
- [GitHub PR #398](https://github.com/NVIDIA/SkillSpector/pull/398)
- [GitHub PR #402](https://github.com/NVIDIA/SkillSpector/pull/402)
- [GitHub PR #412](https://github.com/NVIDIA/SkillSpector/pull/412)
- [GitHub PR #415](https://github.com/NVIDIA/SkillSpector/pull/415)
- [GitHub PR #417](https://github.com/NVIDIA/SkillSpector/pull/417)
- [GitHub PR #422](https://github.com/NVIDIA/SkillSpector/pull/422)
- [GitHub PR #424](https://github.com/NVIDIA/SkillSpector/pull/424)
- [GitHub PR #425](https://github.com/NVIDIA/SkillSpector/pull/425)
- [GitHub PR #426](https://github.com/NVIDIA/SkillSpector/pull/426)
- [GitHub commit 550b9f0](https://github.com/NVIDIA/SkillSpector/commit/550b9f00ad1635f9b5066ac2b1c4cf399a631cfb)
