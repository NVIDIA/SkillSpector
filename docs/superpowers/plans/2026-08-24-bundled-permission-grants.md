# Bundled Permission Grant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete issue #399 by adding deterministic, trust-aware BH3 permission-grant analysis to
the existing bundled execution surface without overstating runtime activation.

**Architecture:** `bundled_execution_surface` keeps sole ownership of root discovery, duplicate-safe
JSON parsing, cache access, findings, ledger rows, and graph registration. A new pure
`bundled_permission_grants` module classifies one already-parsed settings mapping into frozen safe
records and builds at most one BH3 finding. The existing meta, report, suppression, and renderer
paths consume BH3 as an ordinary structural finding, with a boolean-gated score floor.

**Tech Stack:** Python 3.12+, frozen dataclasses, `json`, `hashlib`, LangGraph state reducers, pytest,
Typer CLI tests, Ruff, mypy, `uv`, and Claude Code 2.1.241 for isolated runtime probes.

---

## Working and review contract

- Work only in
  `/Users/christopherk/.config/superpowers/worktrees/Skillspector/issue-399-permission-surface` on
  `feat/christopherk/issue-399-permission-surface`.
- Keep the GitHub PR in draft and labeled `Part of #399`; it depends on draft PR #404.
- Start every production behavior with a focused failing test, observe the intended failure, then
  implement the smallest green change.
- Use `git commit -s` for every commit so each commit carries a DCO signoff.
- After each task, run a specification-conformance review and a code-quality/security review. Fix a
  review finding through a new failing regression test before continuing.
- Do not claim that a permission activated at runtime from parser output, unit tests, shaped graph
  state, or startup recognition alone.

## File responsibility map

**Create:**

- `src/skillspector/nodes/analyzers/bundled_permission_grants.py` — pure 2.1.241 permission grammar,
  severity, precedence, diagnostics, safe identity, and BH3 construction.
- `tests/nodes/analyzers/test_bundled_permission_grants.py` — pure classifier, evidence, and resource
  boundary matrix.
- `docs/superpowers/specs/2026-08-24-bundled-permission-grants-design.md` — approved normative
  contract.
- `docs/superpowers/plans/2026-08-24-bundled-permission-grants.md` — this executable plan.

**Modify:**

- `src/skillspector/nodes/analyzers/bundled_execution_surface.py:67-78,176-196,467-482,1008-1076,1078-2143`
  — exact settings-root work ownership, parse-once fan-out, sanitized source-line recovery, merged
  findings, and one terminal row.
- `tests/nodes/analyzers/test_bundled_execution_surface.py:210-290,500-590` — direct/archive roots,
  exclusions, mixed sections, manifest references, and merged ledger outcomes.
- `src/skillspector/nodes/analyzers/pattern_defaults.py:49-445` — BH3 explanation, category, name,
  and remediation.
- `tests/nodes/analyzers/test_static_patterns.py:312-320` — BH3 default metadata.
- `src/skillspector/nodes/meta_analyzer.py:243` and `tests/nodes/test_meta_analyzer.py:850-1070` —
  structural retention and provider isolation.
- `src/skillspector/nodes/report.py:420-513` and `tests/nodes/test_report.py:80-390` — strict boolean
  BH3 risk floor.
- `tests/integration/test_bundled_execution_surface.py` — real directory/ZIP graph, output, baseline,
  CLI-exit, and redaction coverage.
- `tests/nodes/analyzers/test_registry.py` — prove no second analyzer registration is added.
- `README.md:25-82,565-575,685-700` — rule count, BH3 contract, semantics, and exit behavior.

No production change is planned for `src/skillspector/nodes/analyzers/__init__.py`,
`src/skillspector/cli.py`, `src/skillspector/suppression.py`, or the generic report renderers. A test
that exposes a genuine generic defect must be reviewed before expanding that boundary.

### Task 1: Commit the approved contract before production work

**Files:**

- Create: `docs/superpowers/specs/2026-08-24-bundled-permission-grants-design.md`
- Create: `docs/superpowers/plans/2026-08-24-bundled-permission-grants.md`

- [ ] **Step 1: Sync locked development dependencies and verify the draft branch**

  Run:

  ```bash
  uv sync --locked --extra dev
  git branch --show-current
  gh pr view 429 --repo NVIDIA/SkillSpector --json isDraft,state,headRefName,body,url
  ```

  Expected: dependency synchronization exits zero; the branch is
  `feat/christopherk/issue-399-permission-surface`; PR #429 is open and draft; its body says
  `Depends on #404` and `Part of #399` without a closing keyword.

- [ ] **Step 2: Check the documents for forbidden placeholders and stale scope**

  Run:

  ```bash
  rg -n 'T[B]D|T[O]DO|implement la[t]er|fill in deta[i]ls|plugin-root sett[i]ngs.*permission source' \
    docs/superpowers/specs/2026-08-24-bundled-permission-grants-design.md \
    docs/superpowers/plans/2026-08-24-bundled-permission-grants.md
  ```

  Expected: no matches.

- [ ] **Step 3: Commit and push only the design documents**

  Run:

  ```bash
  git add docs/superpowers/specs/2026-08-24-bundled-permission-grants-design.md \
    docs/superpowers/plans/2026-08-24-bundled-permission-grants.md
  git commit -s -m "docs: design bundled permission grant analysis"
  git push fork HEAD
  ```

  Expected: one signed-off documentation commit appears on the still-draft PR; no source or test
  file is part of the commit.

### Task 2: Establish the frozen permission model and mode semantics

**Files:**

- Create: `src/skillspector/nodes/analyzers/bundled_permission_grants.py`
- Create: `tests/nodes/analyzers/test_bundled_permission_grants.py`

- [ ] **Step 1: Write the import, frozen-record, applicability, and mode tests**

  Begin the new test module with these exact helpers and assertions:

  ```python
  from __future__ import annotations

  from dataclasses import FrozenInstanceError

  import pytest

  from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
  from skillspector.nodes.analyzers.bundled_permission_grants import (
      PermissionAnalysis,
      PermissionSourceLines,
      analyze_permission_grants,
      build_bh3_finding,
  )


  def _analyze(permissions: object, *, source_kind: str = "project_settings") -> PermissionAnalysis:
      return analyze_permission_grants(
          {"permissions": permissions},
          source_kind=source_kind,
          content_digest="sha256:" + "1" * 64,
          source_identity_digest="sha256:" + "2" * 64,
          source_lines=PermissionSourceLines(permissions_line=2),
      )


  def test_mapping_without_permissions_is_not_applicable() -> None:
      result = analyze_permission_grants(
          {"env": {"SAFE": "1"}},
          source_kind="project_settings",
          content_digest="sha256:" + "1" * 64,
          source_identity_digest="sha256:" + "2" * 64,
          source_lines=PermissionSourceLines(),
      )
      assert result.applicable is False
      assert result.outcome is None
      assert result.grants == ()
      assert build_bh3_finding(result, source_path=".claude/settings.json") is None


  @pytest.mark.parametrize("mode", ["default", "manual", "plan", "dontAsk", "auto"])
  def test_non_grant_modes_are_silent(mode: str) -> None:
      result = _analyze({"defaultMode": mode})
      assert result.outcome is LedgerOutcome.COMPLETED
      assert result.grants == ()
      assert build_bh3_finding(result, source_path=".claude/settings.json") is None


  def test_delegate_is_not_a_valid_pinned_default_mode() -> None:
      result = _analyze({"defaultMode": "delegate"})
      assert result.outcome is LedgerOutcome.FAILED
      assert result.reason is LedgerReason.INVALID_CONFIGURATION
      assert result.grants == ()
      assert {item.diagnostic_kind for item in result.diagnostics} == {"unknown_mode"}


  def test_records_are_frozen() -> None:
      result = _analyze({"defaultMode": "acceptEdits"})
      with pytest.raises(FrozenInstanceError):
          result.applicable = False  # type: ignore[misc]
  ```

  Add an exact shared/local table. Shared allow/directory grants use activation `workspace_trust`,
  interface `claude_code_settings_consumers`, and tracking `not_applicable`; local allow/directory
  grants use `local_provenance_and_session_policy`, `claude_code_settings_consumers`, and `unknown`.
  Shared modes use `interface_and_external_policy`, `permission_mode_interface_dependent`, and
  `not_applicable`; local modes use the same first two tokens plus `unknown`. Assert no other token
  can enter evidence.

- [ ] **Step 2: Run the tests and verify RED**

  Run:

  ```bash
  uv run pytest -q tests/nodes/analyzers/test_bundled_permission_grants.py
  ```

  Expected: collection fails with `ModuleNotFoundError` for `bundled_permission_grants`.

- [ ] **Step 3: Add the complete model boundary and mode classifier**

  Define these frozen records and constants in the new module:

  ```python
  _EVIDENCE_SCHEMA: Final = "skillspector.bundled_permission.v1"
  _SEMANTICS_SNAPSHOT: Final = "2.1.241"
  MAX_PERMISSION_STRUCTURAL_ITEMS_PER_DOCUMENT: Final = 2048


  @dataclass(frozen=True)
  class PermissionGrant:
      grant_kind: str
      severity: str
      activation_requirement: str
      interface_applicability: str
      tracking_status: str
      blocking_critical: bool
      grant_digest: str
      source_line: int


  @dataclass(frozen=True)
  class PermissionDiagnostic:
      diagnostic_kind: str
      affects_completeness: bool
      diagnostic_digest: str
      source_line: int


  @dataclass(frozen=True)
  class PermissionSourceLines:
      permissions_line: int = 1
      permission_key_lines: tuple[int, ...] = ()
      allow_lines: tuple[int, ...] = ()
      ask_lines: tuple[int, ...] = ()
      deny_lines: tuple[int, ...] = ()
      additional_directory_lines: tuple[int, ...] = ()
      default_mode_line: int | None = None
      disable_bypass_line: int | None = None
      disable_auto_line: int | None = None
      skip_dangerous_prompt_line: int | None = None


  @dataclass(frozen=True)
  class PermissionAnalysis:
      applicable: bool
      outcome: LedgerOutcome | None
      reason: LedgerReason | None
      grants: tuple[PermissionGrant, ...]
      diagnostics: tuple[PermissionDiagnostic, ...]
      aggregate_digest: str | None
  ```

  Implement `analyze_permission_grants(raw, *, source_kind, content_digest,
  source_identity_digest, source_lines)` as a pure dispatcher. Validate both digests as full
  domain-tagged SHA-256 values. Entry classifiers take their positive line from the corresponding
  tuple index and fall back to `permissions_line`; unknown-key diagnostics use
  `permission_key_lines` in mapping iteration order. No raw value is retained with a line.
  Recognize the eight keys in the design. Implement mode outcomes exactly: bypass CRITICAL,
  acceptEdits MEDIUM, auto known-ignored, default/manual/plan/dontAsk silent, and every other value,
  including `delegate`, completeness-affecting. Populate shared/local activation and tracking
  classifications from
  `source_kind`; reject any other source kind with `ValueError` because discovery owns source scope.

- [ ] **Step 4: Add focused mode positives and mitigations**

  Add tests that assert bypass emits one CRITICAL grant with `blocking_critical is True`, acceptEdits
  emits one MEDIUM non-blocking grant, and this same-document pair emits no grant:

  ```python
  def test_same_document_disable_neutralizes_bypass() -> None:
      result = _analyze(
          {
              "defaultMode": "bypassPermissions",
              "disableBypassPermissionsMode": "disable",
          }
      )
      assert result.outcome is LedgerOutcome.COMPLETED
      assert result.grants == ()
      assert {item.diagnostic_kind for item in result.diagnostics} == {
          "bypass_disabled",
      }
  ```

  Add source-specific `skipDangerousModePermissionPrompt` tests: shared `true` returns
  `skip_dangerous_prompt_ignored` and no grant; local `true` alone returns
  `local_skip_dangerous_prompt_declared`, is applicable, and emits no BH3; local `true` alongside
  bypass remains CRITICAL/blocking, and its diagnostic and aggregate differ from bypass alone; and
  `false` is a recognized no-op in either source.

- [ ] **Step 5: Run GREEN, lint, and commit**

  Run:

  ```bash
  uv run pytest -q tests/nodes/analyzers/test_bundled_permission_grants.py
  uv run ruff check src/skillspector/nodes/analyzers/bundled_permission_grants.py \
    tests/nodes/analyzers/test_bundled_permission_grants.py
  uv run ruff format --check src/skillspector/nodes/analyzers/bundled_permission_grants.py \
    tests/nodes/analyzers/test_bundled_permission_grants.py
  git add src/skillspector/nodes/analyzers/bundled_permission_grants.py \
    tests/nodes/analyzers/test_bundled_permission_grants.py
  git commit -s -m "feat: model bundled permission modes"
  ```

  Expected: all focused tests pass; Ruff reports no errors or formatting changes; the commit is
  signed off.

### Task 3: Implement rule grammar, precedence, and severity

**Files:**

- Modify: `src/skillspector/nodes/analyzers/bundled_permission_grants.py`
- Modify: `tests/nodes/analyzers/test_bundled_permission_grants.py`

- [ ] **Step 1: Add a table-driven severity matrix before implementation**

  Add this matrix and assert the single returned grant's severity and blocking flag:

  ```python
  @pytest.mark.parametrize(
      ("rule", "severity", "blocking"),
      [
          ("Bash", "CRITICAL", True),
          ("Bash(*)", "CRITICAL", True),
          ("PowerShell(*)", "CRITICAL", True),
          ("Monitor", "CRITICAL", True),
          ("Read", "CRITICAL", True),
          ("Read(//**)", "CRITICAL", True),
          ("Edit(~/**)", "CRITICAL", True),
          ("Write", "CRITICAL", True),
          ("Read(~/.ssh/**)", "HIGH", False),
          ("NotebookEdit", "HIGH", False),
          ("MultiEdit", "HIGH", False),
          ("WebFetch", "HIGH", False),
          ("WebFetch(domain:*)", "HIGH", False),
          ("Edit(../shared/**)", "HIGH", False),
          ("Edit(//tmp/**)", "HIGH", False),
          ("mcp__billing", "HIGH", False),
          ("mcp__billing__*", "HIGH", False),
          ("Artifact", "HIGH", False),
          ("ShareOnboardingGuide", "HIGH", False),
          ("Workflow", "HIGH", False),
          ("EnterWorktree", "HIGH", False),
          ("Bash(npm test:*)", "MEDIUM", False),
          ("Monitor(npm test:*)", "MEDIUM", False),
          ("Glob", "MEDIUM", False),
          ("Grep", "MEDIUM", False),
          ("LSP", "MEDIUM", False),
          ("WebFetch(domain:docs.example)", "MEDIUM", False),
          ("WebFetch(domain:*.example.com)", "MEDIUM", False),
          ("WebFetch(domain:example.*)", "MEDIUM", False),
          ("mcp__billing__lookup", "MEDIUM", False),
          ("mcp__billing__get_*", "MEDIUM", False),
          ("Edit(../shared/config.json)", "MEDIUM", False),
          ("Edit(../shared/report-*.md)", "MEDIUM", False),
          ("Edit(./generated/**)", "MEDIUM", False),
          ("Edit(/tmp/**)", "MEDIUM", False),
          ("Read(../shared/report.md)", "MEDIUM", False),
          ("Skill", "MEDIUM", False),
          ("Skill(commit)", "MEDIUM", False),
          ("ExitPlanMode", "MEDIUM", False),
      ],
  )
  def test_allow_rule_severity(rule: str, severity: str, blocking: bool) -> None:
      result = _analyze({"allow": [rule]})
      assert [(grant.severity, grant.blocking_critical) for grant in result.grants] == [
          (severity, blocking)
      ]
  ```

  Add a silent control for narrow in-project Read. Assert exact `Bash(npx prettier:*)` and its pinned
  whitespace spelling are MEDIUM `scoped_execution`: `npx` can fetch packages and load executable
  configuration or plugins, so it is not a safe-wrapper exception. Assert `WebFetch(*)` is a
  completeness-neutral `unsupported_allow_specifier` diagnostic rather than an all-domain
  equivalent. Boundary-test WebFetch's 253-character total and 63-character label limits, valid
  ASCII/punycode and wildcard labels, terminal-dot normalization, and invalid schemes, user-info,
  ports, paths, whitespace, empty labels, `?`, and non-ASCII input.

- [ ] **Step 2: Verify RED on unimplemented grant rules**

  Run:

  ```bash
  uv run pytest -q tests/nodes/analyzers/test_bundled_permission_grants.py -k 'allow_rule or silent'
  ```

  Expected: assertions fail because allow rules are not yet classified.

- [ ] **Step 3: Implement closed grammar and path classes**

  Add pure private functions with these exact responsibilities:

  - `_parse_permission_rule(rule: str)` validates a bare tool or one parenthesized specifier.
  - `_classify_path_specifier(specifier: str)` returns only project, external, home, root,
    sensitive, or invalid classifications.
  - `_classify_allow_rule(rule: str, context)` returns a grant, known ignored diagnostic, or unknown
    diagnostic.
  - `_classify_additional_directory(value: str, context)` applies add-directory path semantics:
    filesystem-root/home CRITICAL, sensitive external/home HIGH, other parent/absolute external
    MEDIUM, project/current-directory silent, invalid path diagnostic, and static-unknown existence.

  Sensitive markers use ASCII case normalization and complete path-segment or
  ASCII-alphanumeric-token boundaries. Add negative tests proving `tokenizer.py`, `tokenization.md`,
  and `secretariat.md` remain ordinary paths while exact and punctuation-delimited secret/token
  material stays sensitive.

  Keep raw strings inside those call frames. Hash with a domain separator before constructing a
  returned frozen record. Do not use an unbounded regular expression; use one bounded split at the
  first `(` and require the last character to be `)`.

  Implement the complete design `grant_kind` allowlist as a frozen constant and map every mode,
  execution, read/edit/write, NotebookEdit/MultiEdit/Glob/Grep/LSP, WebFetch/WebSearch, MCP, and
  additional-directory class to exactly one token. Assert the constant equals the normative set,
  every table row maps to the expected token, `grant_count` counts retained grants, and
  `grant_kinds` later deduplicates and lexicographically joins tokens at document aggregation.

  Add a closed canonical-routing table from the pinned 2.1.241 tools snapshot. Route Monitor through
  the Bash execution classifier. Route Artifact/ShareOnboardingGuide to HIGH
  `external_content_upload`, Workflow to HIGH `autonomous_workflow`, EnterWorktree to HIGH
  `workspace_boundary_change`, Skill to MEDIUM `skill_invocation`, and ExitPlanMode to MEDIUM
  `approval_gate_transition`. Route bare Grep/Glob/LSP to their distinct MEDIUM filesystem tokens.
  Path-qualified Glob is a completeness-neutral `ignored_path_qualifier` diagnostic because 2.1.241
  uses `Read(...)` for that approval; path-qualified Grep/LSP are completeness-affecting
  `runtime_uncertain_rule` diagnostics because their pinned handling is not established. Only accept
  the documented scoped Skill form among the
  generic routes. Table-test every exact name and severity, and add an exhaustiveness assertion that
  each known name appears in exactly one route. Include feature-gated `SendUserMessage` and pinned
  `ReadMcpResourceDirTool` in the known-non-grant route and table-test their bare and scoped allow
  diagnostics plus valid ask/deny forms, so feature availability cannot turn a canonical tool into
  an unknown-rule failure.

  Add a dedicated additional-directory table proving that it does not reuse permission-rule anchor
  semantics: `/tmp` is absolute external MEDIUM, `//` and `~` are whole-root/home CRITICAL,
  `~/.ssh` is sensitive HIGH, `../docs` is external MEDIUM, and `./subdir` is within-project silent.
  Assert each lexically valid entry has a completeness-neutral
  `directory_existence_static_unknown` diagnostic because the pure helper does not call `stat`.
  Lexically normalize interior `.`/`..`: `child/../docs` stays project-local and
  `child/../../docs` becomes external. Only ASCII `[A-Za-z]:` prefixes are Windows drives; a drive
  root such as `C:\\` or `C:/` emits a
  conditional CRITICAL whole-root grant; a lexically sensitive Windows absolute path such as
  `C:\\Users\\x\\.ssh` emits a conditional HIGH sensitive-directory grant. Windows separator-only
  backslash forms resolve to the current drive root and are conditional CRITICAL. Other Windows
  absolute drive forms and complete UNC forms with non-empty server/share components emit a
  conditional MEDIUM external-directory grant unless sensitive. One-component UNC-like forms such
  as `\\server` or `//server` resolve drive-root-relative and are conditional MEDIUM unless
  sensitive. Recognize both separator spellings and attach the completeness-affecting
  `platform_dependent_path` diagnostic. Extended `\\?\\C:\\` drive roots are conditional CRITICAL;
  extended drive tails and `\\?\\UNC\\server\\share` are conditional MEDIUM unless sensitive. Bare
  `\\.\\` is a conditional CRITICAL current-drive-root form. Other device/reserved namespaces such
  as `\\.\\PIPE`, `\\?\\Volume{...}`, `\\?\\GLOBALROOT`, bare/incomplete `\\?\\...`, and
  `\\??\\...` emit only `platform_dependent_path`, with no existence diagnostic or guessed grant.
  Non-ASCII colon prefixes such as `é:/docs`, `中:/docs`, and `Ｃ:/docs` are ordinary
  project-relative paths with only the neutral existence diagnostic. Drive-relative or malformed
  UNC ambiguity with no provable resolved scope emits the platform diagnostic without guessing a
  grant. Empty, NUL, malformed-home, and environment-variable forms are `invalid_path`. Table-test
  `sensitive_additional_directory` in the grant-kind allowlist.

- [ ] **Step 4: Add and implement known ignored grammar tests**

  In `allow`, test `*`, `B*`, and `mcp__*` as ignored known diagnostics, not grants. Test
  path-qualified Write/NotebookEdit/MultiEdit/Glob, duplicate rules, and list permutation.
  They must produce stable semantic diagnostics or silent output without degrading completeness.
  Assert bare Write is CRITICAL, bare NotebookEdit/MultiEdit are HIGH, and bare Grep/Glob/LSP are
  MEDIUM. Assert path-qualified Grep/LSP instead produce completeness-affecting
  `runtime_uncertain_rule`. Lock bare MultiEdit with a pinned-2.1.241 fixture/probe assertion that it
  remains in the binary's canonical edit/write set. Test
  `UnknownTool(*)`, malformed delimiters, traversal, UNC, drive, NUL, and unknown MCP shapes as
  completeness-affecting.

  Table-test every exact known-non-grant name in the design, including `Agent`, `Cd`, task tools, and
  messaging/control tools: an allow returns a completeness-neutral
  `known_non_grant_tool` diagnostic and no grant. Assert `allow: ["Skill"]` is a completed MEDIUM
  grant, while `deny: ["Agent(Explore)"]`, `ask: ["Tool(param:value)"]`, and an unknown but
  syntactically valid generic deny rule are completed restrictions. Only a truly unknown/dynamic
  allow rule is completeness-affecting. For known tools that accept only a bare form, assert a
  syntactically valid generic `Tool(param:value)` allow is a completeness-neutral
  `unsupported_allow_specifier` diagnostic.

- [ ] **Step 5: Add RED precedence tests**

  Add exact allow/ask/deny, bare-tool coverage, and overlap-without-proof tests:

  ```python
  def test_bare_ask_neutralizes_scoped_allow_for_same_tool() -> None:
      result = _analyze({"allow": ["Bash(curl:*)"], "ask": ["Bash"]})
      assert result.grants == ()


  def test_nonidentical_overlap_is_not_credited_as_mitigation() -> None:
      result = _analyze(
          {"allow": ["Read(~/.ssh/**)"], "deny": ["Read(~/.ssh/id_rsa)"]}
      )
      assert [(grant.grant_kind, grant.severity) for grant in result.grants] == [
          ("sensitive_read", "HIGH")
      ]


  def test_dont_ask_does_not_remove_preapproved_allow() -> None:
      result = _analyze({"defaultMode": "dontAsk", "allow": ["Bash(npm test:*)"]})
      assert [grant.severity for grant in result.grants] == ["MEDIUM"]
  ```

  Add focused allow-versus-ask/deny cases for all proven equivalences and selectors:

  - `Bash(ls:*)` versus `Bash(ls *)`, and bare Bash versus `Bash(*)`;
  - scoped `Monitor(command)` versus equivalent Bash ask/deny command rules, while a bare Monitor
    allow retains its separate WebSocket-bearing capability under a Bash-only restriction;
  - PowerShell ASCII-case-insensitive tool/command spelling, with non-ASCII code points remaining
    exact rather than using Unicode full case folding;
  - WebFetch domain case and terminal-dot normalization, including an identical normalized wildcard
    domain pattern;
  - `deny: ["*"]`, `ask: ["B*"]`, and `deny: ["mcp__*"]` neutralizing matching allow candidates;
  - bare `mcp__billing` and `mcp__billing__*` as the same server-wide identity; and
  - conservative negative cases for different path globs, different WebFetch wildcard patterns,
    and every other unproven specifier overlap.

  Add bypass interaction tests proving `ask: ["*"]` and `deny: ["*"]` each remove the
  `permission_mode_bypass` grant and emit only the safe `bypass_global_restriction` diagnostic.
  Counter-test narrower `ask: ["Bash"]`, `deny: ["Read"]`, `ask: ["B*"]`, and
  `deny: ["mcp__*"]`: bypass remains CRITICAL and blocking because other tool calls still execute
  silently. Keep same-document `disableBypassPermissionsMode` as the independent exact disable.

  Assert allow-side `*`, `B*`, and `mcp__*` remain ignored diagnostics even though the same spellings
  are valid bounded precedence selectors in `ask` and `deny`.

  Run the three tests and confirm they fail before precedence is implemented.

- [ ] **Step 6: Implement conservative same-document precedence**

  Normalize proven runtime-equivalent identities in all three lists before mitigation: bare Bash
  equals `Bash(*)`; `Bash(ls:*)` equals `Bash(ls *)`; PowerShell matching is ASCII
  case-insensitive while non-ASCII remains exact;
  WebFetch domain patterns are case-insensitive with one trailing root dot removed; and bare MCP
  server equals its `__*` spelling. Apply deny before ask. Suppress an allow only for an identical
  normalized rule, a bare same-tool selector, or a valid bounded ask/deny tool-name glob that
  matches its normalized tool identifier. Do not perform path, domain-pattern, command-pattern, or
  other specifier-glob subsumption. After parsing the restrictive lists, neutralize bypass only when
  either list contains the exact valid global selector `*`; retain bypass for every narrower rule or
  glob. Emit `bypass_global_restriction` for that exact same-document mitigation.

  Index exact, bare, and MCP-server-wide restrictions before coverage and compile each distinct
  tool-name glob once. Before matching one distinct glob to one distinct candidate tool identifier,
  charge `len(glob) + len(tool_identifier)` against the exact 8,388,608-character document budget.
  Exceeding it is an atomic permission `COMPONENT_LIMIT` failure. Add a real near-1-MB unique-pattern
  cross-product regression that asserts the deterministic limit outcome, not a wall-clock threshold,
  plus a probe proving the bounded path no longer takes tens of seconds.

- [ ] **Step 7: Run GREEN and commit**

  Run:

  ```bash
  uv run pytest -q tests/nodes/analyzers/test_bundled_permission_grants.py
  uv run ruff check src/skillspector/nodes/analyzers/bundled_permission_grants.py \
    tests/nodes/analyzers/test_bundled_permission_grants.py
  git add src/skillspector/nodes/analyzers/bundled_permission_grants.py \
    tests/nodes/analyzers/test_bundled_permission_grants.py
  git commit -s -m "feat: classify bundled permission grants"
  ```

  Expected: the complete pure grammar matrix passes and lint is clean.

### Task 4: Make failure, cardinality, identity, and evidence contracts fail closed

**Files:**

- Modify: `src/skillspector/nodes/analyzers/bundled_permission_grants.py`
- Modify: `tests/nodes/analyzers/test_bundled_permission_grants.py`

- [ ] **Step 1: Add wrong-type, unknown-key, and mixed-validity tests**

  Assert that a valid reportable allow plus a wrong-type sibling returns PARTIAL with
  `LedgerReason.INVALID_CONFIGURATION` and preserves the grant. Assert a non-empty permissions
  object whose supplied fields are all unknown/invalid returns FAILED with no grant. Assert `{}`,
  recognized empty arrays, valid ask/deny-only, and auto-only objects return COMPLETED with no
  finding.

- [ ] **Step 2: Add exact resource-bound tests and verify RED**

  Count one structural item per permission-object key, including unknown keys, plus one per raw list
  entry in `allow`, `ask`, `deny`, and `additionalDirectories`. Test an object with `allow` and
  `defaultMode` keys plus 2,046 raw allow entries: exactly 2,048 items is accepted by the structural
  budget (use a shape that remains below the separate precedence matcher-work budget). Add one entry:
  2,049 returns FAILED with `LedgerReason.COMPONENT_LIMIT`, no grants, no diagnostics, and no
  aggregate digest. Also test 2,048 unique unknown keys (within the resource limit, at most one
  diagnostic per key) and a sub-megabyte object with 20,000 unique unknown keys (atomic
  COMPONENT_LIMIT before diagnostic construction). Nested values under an unknown key must never
  recursively expand the item or diagnostic count.
  Run:

  ```bash
  uv run pytest -q tests/nodes/analyzers/test_bundled_permission_grants.py -k 'limit or mixed or unknown'
  ```

  Expected: the boundary tests fail until total structural cardinality is counted before validation,
  diagnostics, and deduplication.

- [ ] **Step 3: Implement outcome reduction**

  Apply this exact order:

  1. reject non-object permissions;
  2. count all permission keys and raw entries of recognized list fields without constructing
     diagnostics;
  3. return atomic permission-subanalysis COMPONENT_LIMIT above 2,048;
  4. validate individual entries while preserving valid siblings;
  5. deduplicate and sort normalized effective grants/diagnostics;
  6. return PARTIAL when valid analysis and completeness-affecting diagnostics coexist;
  7. return FAILED when a non-empty permission section supplied values but no field or value can be
     safely analyzed; treat an empty object and recognized empty arrays as valid no-ops; and
  8. otherwise return COMPLETED.

- [ ] **Step 4: Add the exact safe-evidence test**

  Supply canaries containing a secret path, domain, MCP name, Markdown, control characters, and
  Unicode. Build BH3 and assert:

  ```python
  _ALLOWED_EVIDENCE = {
      "schema",
      "claude_semantics_snapshot",
      "source_kind",
      "declaration_status",
      "artifact_effect_status",
      "activation_requirement",
      "interface_applicability",
      "tracking_status",
      "runtime_status",
      "grant_count",
      "critical_grant_count",
      "high_grant_count",
      "medium_grant_count",
      "grant_kinds",
      "diagnostic_count",
      "diagnostic_kinds",
      "max_severity",
      "blocking_critical",
      "aggregate_digest",
  }

  finding = build_bh3_finding(result, source_path=".claude/settings.json")
  assert finding is not None
  assert set(finding.evidence) == _ALLOWED_EVIDENCE
  assert all(isinstance(value, str | int | bool) for value in finding.evidence.values())
  assert finding.evidence["schema"] == "skillspector.bundled_permission.v1"
  assert finding.evidence["claude_semantics_snapshot"] == "2.1.241"
  assert finding.evidence["runtime_status"] == "external_unknown"
  assert finding.finding == finding.matched_text == finding.evidence["aggregate_digest"]
  ```

  Serialize the whole finding and assert no canary occurs.

- [ ] **Step 5: Implement domain-separated aggregate identity and finding construction**

  Build the aggregate digest from schema, snapshot, source kind, full hashed physical
  `source_identity_digest`, full `content_digest`, sorted grant digests, sorted diagnostic digests,
  mitigation result, maximum severity, and the literal boolean. Add tests proving that reorder and
  duplicate variants have the same semantic grant/diagnostic/count projections but different
  aggregates when their physical content digests differ; whitespace-only byte mutation changes the
  aggregate; and identical bytes at two different source-identity digests have different
  aggregates. Supplying a malformed digest must raise `ValueError` without returning raw input.
  Set `start_line` to the minimum reportable grant source line, `confidence=1.0`, structural tags,
  fixed explanation/remediation, and one concise count-only message. Fall back to the enclosing
  permissions line and then line 1 only when no entry line was recoverable. Return no finding when
  no reportable grant remains. Add a test with silent line 2, HIGH line 7, and CRITICAL line 11 that
  asserts BH3 starts at line 7 without exposing either rule.

- [ ] **Step 6: Run GREEN, type-check, and commit**

  Run:

  ```bash
  uv run pytest -q tests/nodes/analyzers/test_bundled_permission_grants.py
  uv run mypy src/skillspector/nodes/analyzers/bundled_permission_grants.py
  uv run ruff check src/skillspector/nodes/analyzers/bundled_permission_grants.py \
    tests/nodes/analyzers/test_bundled_permission_grants.py
  git add src/skillspector/nodes/analyzers/bundled_permission_grants.py \
    tests/nodes/analyzers/test_bundled_permission_grants.py
  git commit -s -m "feat: emit safe bundled permission findings"
  ```

  Expected: classifier tests, mypy, and Ruff pass with no canary disclosure.

### Task 5: Integrate parse-once settings ownership and merged ledger outcomes

**Files:**

- Modify: `src/skillspector/nodes/analyzers/bundled_execution_surface.py`
- Modify: `tests/nodes/analyzers/test_bundled_execution_surface.py`

- [ ] **Step 1: Add exact-root and exclusion tests**

  Extend the existing settings tests with permissions-only fixtures for:

  - `.claude/settings.json` and `.claude/settings.local.json`;
  - `bundle.zip!/.claude/settings.json`;
  - `outer.zip!/inner.zip!/.claude/settings.local.json`;
  - excluded `settings.json`, `.claude-plugin/settings.json`,
    `example/.claude/settings.json`, `plugin/.claude/settings.json`, and archive equivalents.

  Assert only the four exact roots emit BH3 and each evidence `source_kind` is exact.

- [ ] **Step 2: Run the root tests and verify RED**

  Run:

  ```bash
  uv run pytest -q tests/nodes/analyzers/test_bundled_execution_surface.py \
    -k 'permission and (root or archive or exclusion)'
  ```

  Expected: no BH3 exists before surface integration.

- [ ] **Step 3: Introduce one parsed settings-work registry**

  Add a frozen surface-owned work record near `HookDocument`:

  ```python
  @dataclass(frozen=True)
  class _SettingsWork:
      source_path: str
      source_kind: str
      content_digest: str
      source_identity_digest: str
      raw: dict[str, object] | None
      parse_error: BaseException | None
      permission_analysis: PermissionAnalysis | None
      permission_source_lines: PermissionSourceLines
  ```

  Build `settings_work_by_path` before hook discovery by selecting paths whose namespace member parts
  equal `(".claude", "settings.json")` or `(".claude", "settings.local.json")`. Call `_load_json`
  once. Recover a `PermissionSourceLines` record from the already-cached JSON syntax tree using only
  key-order/list indexes and positive line numbers; unknown names and JSON values must not enter that
  record. Align `permission_key_lines` with parsed-mapping insertion order and known list-line tuples
  with their raw entry indexes. Compute the existing content digest once. Compute
  `source_identity_digest` as full SHA-256
  over `b"skillspector.bundled_permission.source.v1\0"` plus the UTF-8 encoding of the normalized
  cache-key path, including the complete `outer.zip!/inner.zip!/member` namespace for archives. Pass
  only the two full digests, mapping, and sanitized lines to `analyze_permission_grants`; never pass
  the raw source path into the analysis helper. The separate finding builder receives it only as
  `Finding.file`, never as evidence or aggregate input. Do not add permissions-only paths to
  `handled_paths`.

- [ ] **Step 4: Refactor hooks to consume the retained mapping**

  Split hook parsing into a mapping-based helper so root settings and later manifest references do
  not call `_load_json` again. Keep `_parse_hook_document` for non-settings hook JSON. When a
  settings path is manifest-referenced, evaluate the hook role even if permission analysis already
  ran; merge declaration roles into the existing HookDocument when hooks are valid.

- [ ] **Step 5: Add mixed-section and manifest-reference RED tests**

  Add cases for:

  - valid hooks plus valid permissions: one path-level COMPLETED row owning BH1/BH2/BH3;
  - valid hooks plus invalid permissions: one PARTIAL row owning BH1/BH2;
  - invalid hooks plus valid permissions: one PARTIAL row owning BH3;
  - permissions-only settings later referenced by a manifest: BH3 survives and the invalid hook
    role makes the same row PARTIAL;
  - missing/malformed/duplicate-key/binary/oversized settings: one FAILED row and no settings
    findings;
  - 2,049 permission structural items plus valid hooks: one COMPONENT_LIMIT PARTIAL row preserving
    BH1/BH2 and no BH3;
  - 2,049 permission structural items with no valid hook section: one COMPONENT_LIMIT FAILED row and
    no settings findings.

  For every case, assert exactly one event with `event["path"] == settings_path`, phase
  `bundled_settings`, and no duplicate producer origin.

  Add a multiline permission fixture and assert BH3 points to the earliest reportable grant rather
  than line 1 or an earlier silent grant. Add an unavailable-location control that falls back to the
  `permissions` key line without changing the semantic outcome.

- [ ] **Step 6: Implement document-outcome reduction and one producer row**

  Stage settings findings until hook and permission subanalyses are complete. Reduce outcomes as
  follows:

  - atomic shared parse/integrity error: FAILED and discard staged settings findings;
  - one valid subanalysis plus one invalid or component-limited subanalysis: PARTIAL and retain the
    valid subanalysis findings;
  - a component-limited permission subanalysis with no independently valid hook section: FAILED;
  - all applicable subanalyses valid: COMPLETED;
  - neither hooks nor permissions applicable: no row.

  Use `LedgerReason.INVALID_CONFIGURATION` for mixed semantic errors and
  `LedgerReason.COMPONENT_LIMIT` for permission cardinality. Attach every retained BH1/BH2/BH3 ID to
  the single path-level row. Preserve existing line-ranged flow rows for reachable payload work.

- [ ] **Step 7: Run all surface and flow regressions**

  Run:

  ```bash
  uv run pytest -q \
    tests/nodes/analyzers/test_bundled_permission_grants.py \
    tests/nodes/analyzers/test_bundled_execution_surface.py \
    tests/nodes/analyzers/test_bundled_execution_runtime.py \
    tests/nodes/analyzers/test_bundled_hook_flow.py
  uv run ruff check src/skillspector/nodes/analyzers/bundled_permission_grants.py \
    src/skillspector/nodes/analyzers/bundled_execution_surface.py \
    tests/nodes/analyzers/test_bundled_permission_grants.py \
    tests/nodes/analyzers/test_bundled_execution_surface.py
  ```

  Expected: every permission, hook-runtime, hook-flow, and surface test passes; Ruff is clean.

- [ ] **Step 8: Commit the surface integration**

  Run:

  ```bash
  git add src/skillspector/nodes/analyzers/bundled_execution_surface.py \
    tests/nodes/analyzers/test_bundled_execution_surface.py
  git commit -s -m "feat: analyze permissions in bundled settings"
  ```

### Task 6: Preserve BH3 structurally and add the strict score floor

**Files:**

- Modify: `src/skillspector/nodes/analyzers/pattern_defaults.py`
- Modify: `tests/nodes/analyzers/test_static_patterns.py`
- Modify: `src/skillspector/nodes/meta_analyzer.py`
- Modify: `tests/nodes/test_meta_analyzer.py`
- Modify: `src/skillspector/nodes/report.py`
- Modify: `tests/nodes/test_report.py`
- Modify: `tests/nodes/analyzers/test_registry.py`

- [ ] **Step 1: Write RED defaults and registry tests**

  Extend the existing BH defaults parametrization to `['BH1', 'BH2', 'BH3']`. Assert BH3 resolves to
  category `Bundled Execution Surface` and pattern `Bundled Permission Grant`. Add a registry test
  that the analyzer sequence still contains exactly one `bundled_execution_surface` and no
  `bundled_permission_grants` node.

- [ ] **Step 2: Write RED structural-retention tests**

  Clone the BH1 structural tests with a LOW-confidence BH3. Assert it bypasses provider batching,
  survives provider rejection, survives `use_llm=False`, and owns consistent meta lineage. Run:

  ```bash
  uv run pytest -q tests/nodes/analyzers/test_static_patterns.py \
    tests/nodes/test_meta_analyzer.py tests/nodes/analyzers/test_registry.py -k 'BH3 or bh3'
  ```

  Expected: defaults and retention fail because BH3 is not in the four maps or structural set.

- [ ] **Step 3: Implement defaults and structural registration**

  Add BH3 to `DEFAULT_EXPLANATIONS`, `RULE_ID_TO_CATEGORY`, `PATTERN_NAMES`, and
  `DEFAULT_REMEDIATIONS`. Add only `BH3` to `_STRUCTURAL_RULE_IDS`. Do not modify the dynamic analyzer
  registry.

- [ ] **Step 4: Write strict score-floor RED tests**

  Add tests for:

  ```python
  blocking = _finding("BH3", "CRITICAL", confidence=1.0, file=".claude/settings.json")
  blocking.evidence = {"blocking_critical": True}
  assert _compute_risk_score([blocking], False) == (51, "HIGH", "DO_NOT_INSTALL")

  nonblocking = _finding("BH3", "CRITICAL", confidence=1.0, file=".claude/settings.json")
  nonblocking.evidence = {"blocking_critical": False}
  assert _compute_risk_score([nonblocking], False)[0] == 50

  string_marked = _finding("BH3", "LOW", confidence=1.0, file=".claude/settings.json")
  string_marked.evidence = {"blocking_critical": "true"}
  assert _compute_risk_score([string_marked], False)[0] == 5
  ```

  Also test missing evidence and zero confidence. Run the focused tests and confirm the blocking case
  remains at ordinary score 50 before implementation.

- [ ] **Step 5: Implement a finding-aware floor function**

  Keep the static SC8/BH2 map. Add a private function that returns 51 for BH3 only when
  `finding.evidence.get("blocking_critical") is True`; return the existing static floor for other
  rules. Call it from the post-suppression score-floor reduction. Do not use truthiness or severity
  as a proxy.

- [ ] **Step 6: Run GREEN and commit**

  Run:

  ```bash
  uv run pytest -q tests/nodes/analyzers/test_static_patterns.py \
    tests/nodes/test_meta_analyzer.py tests/nodes/test_report.py \
    tests/nodes/analyzers/test_registry.py
  uv run ruff check src/skillspector/nodes/analyzers/pattern_defaults.py \
    src/skillspector/nodes/meta_analyzer.py src/skillspector/nodes/report.py \
    tests/nodes/analyzers/test_static_patterns.py tests/nodes/test_meta_analyzer.py \
    tests/nodes/test_report.py tests/nodes/analyzers/test_registry.py
  git add src/skillspector/nodes/analyzers/pattern_defaults.py \
    src/skillspector/nodes/meta_analyzer.py src/skillspector/nodes/report.py \
    tests/nodes/analyzers/test_static_patterns.py tests/nodes/test_meta_analyzer.py \
    tests/nodes/test_report.py tests/nodes/analyzers/test_registry.py
  git commit -s -m "feat: integrate BH3 reporting and risk policy"
  ```

  Expected: all touched suites pass, the registry remains unchanged, and lint is clean.

### Task 7: Verify graph, archives, outputs, baselines, and CLI exits

**Files:**

- Modify: `tests/integration/test_bundled_execution_surface.py`

- [ ] **Step 1: Add issue #399 Case B/C graph fixtures**

  Extend `_case_files` with permission-only Case B and mixed Case C. Parameterize direct directory
  and ZIP input. Add a small test-only nested-ZIP materializer—the existing helper covers direct
  directories and ZIPs only—and use it for a genuine outer-ZIP containing inner-ZIP bytes. Assert:

  - Case B emits one BH3;
  - Case C emits BH1, BH2, and BH3;
  - the blocking cases score at least 51 and recommend `DO_NOT_INSTALL`;
  - local settings evidence uses tracking `unknown`; and
  - no issue projection contains raw rules or canaries.

- [ ] **Step 2: Add all-format output tests before renderer changes**

  Extend the existing `json`, `markdown`, `sarif`, and `terminal` parametrization to a BH3 fixture.
  Give BH3 its permission evidence allowlist and require a full aggregate digest. Assert raw path,
  domain, command, MCP, Markdown, control, and Unicode canaries are absent from each serialized
  report.

- [ ] **Step 3: Add baseline mutation tests**

  Generate a real baseline from a ZIP BH3 fixture, rescan unchanged, and assert score zero with BH3
  suppressed. Then independently mutate one effective grant and
  `disableBypassPermissionsMode`; each mutation must reactivate BH3 because its aggregate digest
  changes. Also prove whitespace/reorder/duplicate byte mutations and moving identical bytes to a
  distinct normalized archive member identity reactivate BH3, while their semantic projections
  remain stable where applicable. Assert suppressed BH3 never applies the 51 floor.

- [ ] **Step 4: Add CLI outcome interaction tests**

  Through the real Typer subprocess helper, assert:

  - COMPLETED non-blocking BH3 exits 0 when score is at most 50;
  - COMPLETED blocking BH3 exits 1;
  - PARTIAL plus blocking BH3 exits 1 and reports completeness `partial`;
  - PARTIAL non-blocking exits 0 by default and 1 with `--fail-on-incomplete`; and
  - atomic malformed/duplicate JSON settings exits 2 and emits no settings finding;
  - permissions-only structural cardinality failure exits 2 and emits no BH3; and
  - the same permission cardinality failure alongside valid hooks is PARTIAL, preserves BH1/BH2,
    and follows the existing PARTIAL exit policy.

- [ ] **Step 5: Run integration RED, then make only necessary fixture/helper changes**

  Run:

  ```bash
  uv run pytest -q -m integration tests/integration/test_bundled_execution_surface.py
  ```

  Expected before the production tasks are complete: the new BH3 assertions fail. After Tasks 2-6,
  the integration module passes without changing generic renderer or CLI production code.

- [ ] **Step 6: Commit integration coverage**

  Run:

  ```bash
  git add tests/integration/test_bundled_execution_surface.py
  git commit -s -m "test: cover bundled permissions end to end"
  ```

### Task 8: Update user-facing documentation and run corpus calibration

**Files:**

- Modify: `README.md`

- [ ] **Step 1: Update the documented surface exactly**

  Change the pattern count from 72 to 73, rename the feature to bundled hook and permission
  analysis, add BH3 to the overview and pattern table, and replace the hooks-only/BH3 non-goal text.
  Document exact project/local roots, 2.1.241 permission snapshot, trust/provenance/interface
  conditions, boolean-gated score floor, PARTIAL/FAILED exits, safe evidence, and the exclusion of
  plugin-root/user/managed settings. Do not silently change the BH1/BH2 snapshot if their code still
  records a different version.

- [ ] **Step 2: Check README counts and stale claims**

  Run:

  ```bash
  rg -n '72 vulnerability|Bundled Execution Surface \(2 patterns\)|does \*\*not\*\* implement BH3|BH3' README.md
  ```

  Expected: no stale 72/two-pattern/hooks-only claim; BH3 appears in the feature overview and rule
  table.

- [ ] **Step 3: Locate and pin available corpora read-only**

  Run `rg --files` under the known local catalog roots, record `git rev-parse HEAD` for each Git
  checkout, and scan only exact pinned paths. Do not modify, clean, or update a corpus checkout.
  Record for each corpus: path, revision, total components, exact settings roots, BH1 count, BH2
  count, BH3 count, ledger exceptions, and command exit. Use these current discovery roots:

  ```bash
  for corpus_root in \
    /Users/christopherk/Work/skills/agent-skills \
    /Users/christopherk/.claude/plugins/cache \
    /Users/christopherk/.codex/plugins/cache; do
    if [ -d "$corpus_root" ]; then
      rg --files --hidden "$corpus_root" | wc -l
      git -C "$corpus_root" rev-parse HEAD 2>/dev/null || true
      git -C "$corpus_root" status --short 2>/dev/null || true
    else
      echo "unavailable: $corpus_root"
    fi
  done
  ```

  Create a temporary report directory with `BH3_CALIBRATION_DIR=$(mktemp -d)` and derive exact scan
  roots rather than passing a deep cache parent to `--recursive`. The recursive CLI intentionally
  detects only immediate child skills; a cache parent would miss deeper plugin versions, while a
  monolithic parent scan would make their settings paths nested and therefore inapplicable. Derive
  each skill root and each plugin/version root that owns an exact settings, hooks, or plugin-manifest
  sentinel, deduplicate the paths, and scan every root independently:

  ```bash
  BH3_CORPUS_ROOTS="$BH3_CALIBRATION_DIR/corpus-roots.txt"
  : > "$BH3_CORPUS_ROOTS"
  for bh3_catalog_root in \
    /Users/christopherk/Work/skills/agent-skills \
    /Users/christopherk/.claude/plugins/cache \
    /Users/christopherk/.codex/plugins/cache; do
    [ -d "$bh3_catalog_root" ] || continue
    rg --files --hidden "$bh3_catalog_root" | while IFS= read -r bh3_candidate; do
      case "$bh3_candidate" in
        */SKILL.md|*/skill.md)
          dirname "$bh3_candidate"
          ;;
        */.claude/settings.json)
          printf '%s\n' "${bh3_candidate%/.claude/settings.json}"
          ;;
        */.claude/settings.local.json)
          printf '%s\n' "${bh3_candidate%/.claude/settings.local.json}"
          ;;
        */hooks/hooks.json)
          printf '%s\n' "${bh3_candidate%/hooks/hooks.json}"
          ;;
        */.claude-plugin/plugin.json)
          printf '%s\n' "${bh3_candidate%/.claude-plugin/plugin.json}"
          ;;
      esac
    done
  done | LC_ALL=C sort -u > "$BH3_CORPUS_ROOTS"

  BH3_CORPUS_INDEX=0
  BH3_CORPUS_MANIFEST="$BH3_CALIBRATION_DIR/results.tsv"
  : > "$BH3_CORPUS_MANIFEST"
  while IFS= read -r bh3_scan_root; do
    BH3_CORPUS_INDEX=$((BH3_CORPUS_INDEX + 1))
    BH3_CORPUS_REPORT=$(printf '%s/%06d.json' "$BH3_CALIBRATION_DIR" "$BH3_CORPUS_INDEX")
    if uv run skillspector scan "$bh3_scan_root" --no-llm --format json \
      --output "$BH3_CORPUS_REPORT"; then
      BH3_CORPUS_EXIT=0
    else
      BH3_CORPUS_EXIT=$?
    fi
    printf '%s\t%s\t%s\n' "$BH3_CORPUS_EXIT" "$BH3_CORPUS_REPORT" "$bh3_scan_root" \
      >> "$BH3_CORPUS_MANIFEST"
  done < "$BH3_CORPUS_ROOTS"
  ```

  Preserve that directory until counts and exceptions have been copied to draft PR #429 verification
  notes. A 0, 1, or 2 exit is data for calibration; record it with the corresponding exact root and
  report, and investigate every 2 before deciding whether the corpus result is usable. Record roots
  that contain more than one immediate skill as monolithic bundle scans; do not substitute a
  recursive cache-parent scan.

- [ ] **Step 4: Run the benign and positive calibration sets**

  The benign set must include restrictive-only settings, auto in project/local scope, narrow project
  Read, tracked-looking local content without provenance claims, and settings-like nested/plugin
  files. The positive set must include the exact prettier rule as MEDIUM `scoped_execution`, every
  CRITICAL/HIGH/MEDIUM class, same-document mitigation, mixed validity, direct/ZIP/nested-ZIP, and
  Case B/C. Require zero unexpected BH3 on the benign set and the expected class on every positive
  fixture.

  If a named catalog is unavailable, write `unavailable` plus the checked path in the PR verification
  notes. Do not reuse issue #399's historical counts as a fresh result.

- [ ] **Step 5: Commit documentation**

  Run:

  ```bash
  git add README.md
  git commit -s -m "docs: document bundled permission analysis"
  ```

### Task 9: Real 2.1.241 probes, final review, and complete verification

**Files:**

- Modify: production/tests/docs only when a reproduced issue requires a regression fix
- Update: draft PR body and issue #399 comment after verification

- [ ] **Step 1: Record runtime versions and the pinned MultiEdit evidence**

  Run:

  ```bash
  claude --version
  npx -y @anthropic-ai/claude-code@2.1.241 --version
  BH3_PINNED_PACKAGE_DIR=$(mktemp -d)
  npm pack --silent --pack-destination "$BH3_PINNED_PACKAGE_DIR" \
    @anthropic-ai/claude-code@2.1.241
  tar -xzf "$BH3_PINNED_PACKAGE_DIR/anthropic-ai-claude-code-2.1.241.tgz" \
    -C "$BH3_PINNED_PACKAGE_DIR"
  shasum -a 256 "$BH3_PINNED_PACKAGE_DIR/anthropic-ai-claude-code-2.1.241.tgz"
  LC_ALL=C rg -a -o -m 5 '.{0,96}MultiEdit.{0,96}' \
    "$BH3_PINNED_PACKAGE_DIR/package/cli.js"
  ```

  Expected: record the installed local version separately; the pinned runner prints `2.1.241`; and
  the unpacked pinned executable contains MultiEdit in its canonical edit/write tool sets. Record
  the tarball SHA-256 and bounded matching context as evidence, without claiming tool activation.
  Preserve the temporary package until review evidence is copied to PR #429, then remove only that
  explicitly created directory.

- [ ] **Step 2: Run isolated startup/config E2E combinations**

  In disposable repositories containing no credentials or external endpoints, exercise:

  - shared allow plus additionalDirectories before trust;
  - untracked local allow;
  - the same local file added to the Git index;
  - project and local auto;
  - project and local acceptEdits;
  - bypass alone; and
  - bypass plus same-document disable;
  - bypass plus exact global ask/deny, and bypass plus narrower ask/deny controls;
  - bare MultiEdit, Monitor, Skill, Workflow, EnterWorktree, and ShareOnboardingGuide recognition;
  - feature-gated `SendUserMessage` recognition with `--brief` enabled;
  - bare/server-wide, exact-tool, and partial-tool MCP spellings;
  - bare WebFetch, `domain:*`, literal/wildcard domain, and unsupported `WebFetch(*)` spellings; and
  - `/tmp`, `//`, `~`, `~/.ssh`, `../docs`, `./subdir`, normalized interior-parent, Windows drive
    root, separator-only backslash root, sensitive absolute, ordinary absolute-drive, complete
    backslash/forward-slash UNC, one-component UNC-like, extended drive/UNC roots and tails, bare and
    unsupported device namespaces, non-ASCII colon-relative, malformed UNC, and drive-relative
    additional-directory spellings.

  Capture only safe debug/status lines. Never run a destructive command and never transmit a canary.
  If login/model access permits, add benign reads/writes inside a disposable directory to test actual
  authorization. Otherwise state that recognition/config loading was tested but model-driven tool
  authorization was not.

- [ ] **Step 3: Run an independent specification review**

  Give a fresh reviewer the design, plan, and complete diff. Require a requirement-by-requirement
  verdict on roots, parse-once ownership, grammar, trust/provenance/interface semantics, evidence,
  ledger, scoring, baseline, and exits. For every blocker, add a focused failing regression test,
  implement the fix, and repeat this review until no blocker remains.

- [ ] **Step 4: Run an independent code/security review**

  Require review of edge cases, option interactions, raw-data leakage, denial/ask coverage,
  cardinality/runtime bounds, archive namespace behavior, output correctness, and BH1/BH2
  regressions. Resolve every actionable finding through red-green testing and rerun the affected
  suites.

- [ ] **Step 5: Run fresh focused and repository-wide verification**

  Run these commands after the final code change:

  ```bash
  uv run pytest -q \
    tests/nodes/analyzers/test_bundled_permission_grants.py \
    tests/nodes/analyzers/test_bundled_execution_surface.py \
    tests/nodes/analyzers/test_bundled_execution_runtime.py \
    tests/nodes/analyzers/test_bundled_hook_flow.py \
    tests/nodes/test_meta_analyzer.py \
    tests/nodes/test_report.py
  uv run pytest -q -m integration tests/integration/test_bundled_execution_surface.py
  uv run pytest -q -m 'not integration and not provider' tests/
  uv run pytest -q -m integration tests/ --ignore=tests/integration/test_agent_cli_live.py
  uv run ruff check src/ tests/
  uv run ruff format --check src/ tests/
  uv run mypy \
    src/skillspector/nodes/analyzers/bundled_permission_grants.py \
    src/skillspector/nodes/analyzers/bundled_execution_surface.py \
    src/skillspector/nodes/report.py
  uv build
  git diff --check
  ```

  Expected: focused, non-provider, lint, format, mypy, build, and diff checks exit zero. Rerun every
  integration test even if the known current-base failure
  `tests/integration/test_graph.py::test_graph_surfaces_degraded_llm_stage` appears. Record exact
  pass/fail/skip/xfail counts; do not quote counts from an earlier commit. Any failure not reproduced
  identically on a clean current base blocks completion. At plan authoring, the verified failing
  `origin/main` commit is `d486d0a84d1ab2f90081fde5713638a7d3538e11`; refresh it rather than
  assuming that failure remains forever.

- [ ] **Step 6: Reproduce any full-suite failure on a clean current base**

  Refresh `origin/main`, create a detached temporary worktree, and run the exact failing node there:

  ```bash
  git fetch origin main
  BH3_BASE_VERIFY_PARENT=$(mktemp -d)
  BH3_BASE_VERIFY_PATH="$BH3_BASE_VERIFY_PARENT/skillspector-main"
  git worktree add --detach "$BH3_BASE_VERIFY_PATH" origin/main
  uv run --directory "$BH3_BASE_VERIFY_PATH" --extra dev pytest -q -m integration \
    tests/integration/test_graph.py::test_graph_surfaces_degraded_llm_stage
  test -n "$BH3_BASE_VERIFY_PARENT" && test -n "$BH3_BASE_VERIFY_PATH"
  case "$BH3_BASE_VERIFY_PATH" in
    "$BH3_BASE_VERIFY_PARENT"/*) ;;
    *) exit 1 ;;
  esac
  git worktree remove --force "$BH3_BASE_VERIFY_PATH"
  rmdir "$BH3_BASE_VERIFY_PARENT"
  ```

  Before removal, verify `BH3_BASE_VERIFY_PATH` is non-empty and its parent is exactly
  `BH3_BASE_VERIFY_PARENT`; `--force` is permitted only for this explicitly created detached
  worktree. Expected when the fetched base is still the recorded commit: the named degraded-LLM-stage
  integration test reproduces its existing failure. Record the fetched base commit and exact
  assertion/error. If the fetched base has fixed the test, the feature branch must pass it too. Do
  not waive a branch failure if the base passes, fails differently, or if any additional branch test
  fails.

- [ ] **Step 7: Attempt Docker and live-provider verification with explicit boundaries**

  Run `make docker-smoke` only when `docker info` succeeds. The smoke script builds an image, writes
  two reports, and performs a live GitHub fetch. Redirect its mounted report directory to an explicit
  temporary directory and record GitHub reachability as an external dependency:

  ```bash
  BH3_DOCKER_REPORT_PARENT=$(mktemp -d)
  SKILLSPECTOR_REPO_DIR="$BH3_DOCKER_REPORT_PARENT" make docker-smoke
  ```

  Preserve the reports until their JSON and expected components are checked, then remove only that
  explicitly created directory. Run live provider tests only when their named credentials and
  supported models are available. Record daemon, network, credential, model, authentication, cost,
  and UI blockers exactly; do not convert an unavailable check into a pass.

- [ ] **Step 8: Inspect the final Git and PR state**

  Run:

  ```bash
  git status --short
  BH3_DCO_BASE=$(git merge-base HEAD fork/feat/christopherk/issue-399-hook-surface)
  git log --format='%H' "$BH3_DCO_BASE"..HEAD | while read -r commit_sha; do
    git show -s --format='%B' "$commit_sha" | rg -q '^Signed-off-by:' || {
      echo "missing Signed-off-by: $commit_sha"
      exit 1
    }
  done
  gh pr view 429 --repo NVIDIA/SkillSpector --json isDraft,state,mergeable,headRefOid,statusCheckRollup,url
  ```

  Expected: only intentional files are changed or the tree is clean after the final signed-off
  commit; every commit in the dependent PR range has a `Signed-off-by:` trailer; PR #429 remains open
  and draft.

- [ ] **Step 9: Push and publish evidence without marking ready**

  Push the final signed-off commits to `fork`, update the draft PR body with exact test/runtime/corpus
  evidence and gaps, and add an issue #399 comment linking PR #404 plus the BH3 draft and summarizing
  which PR implements BH1/BH2 versus BH3. Keep the PR draft and keep issue #399 open until both
  slices are merged and acceptance criteria are rechecked.
