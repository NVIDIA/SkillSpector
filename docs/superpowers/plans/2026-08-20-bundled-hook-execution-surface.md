# Bundled Hook Execution Surface Implementation Plan

> **Required workflow:** Execute each task red-green-refactor. Preserve the user-owned working tree,
> keep implementation local for review, and run the deepest practical Claude Code runtime E2E before
> claiming parity.

**Goal:** Add deterministic BH1 hook inventory and fail-closed BH2 bundled-hook exfiltration analysis
for Claude Code runtime sources covered by the approved design.

**Architecture:** A source/runtime module discovers and normalizes root-aware hook declarations. A
flow module classifies shell versus exec handlers, correlates sensitive sources with outbound sinks,
and follows cache-contained entrypoints under hard limits. The analyzer emits ordinary findings and
ledger rows, so the existing graph, reports, suppression, and exit policy remain authoritative.

**Stack:** Python 3.12+, dataclasses, `json`, PyYAML, `shlex`, `ast`, LangGraph state reducers, pytest.

## Task 1: Add failure reasons and analyzer registry seam

**Files:**

- Modify: `src/skillspector/inspection_ledger.py`
- Modify: `src/skillspector/nodes/analyzers/__init__.py`
- Modify: `tests/test_inspection_ledger.py`
- Modify: `tests/nodes/analyzers/test_registry.py`

1. Add failing tests that construct payload-free ledger rows for `INVALID_CONFIGURATION`,
   `DEPTH_LIMIT`, `COMPONENT_LIMIT`, `AGGREGATE_BUDGET`, and `UNMODELED_PAYLOAD`, and assert
   `bundled_execution_surface` occurs immediately after `static_yara` in both registry collections.
2. Run:

   ```bash
   uv run pytest tests/test_inspection_ledger.py tests/nodes/analyzers/test_registry.py -q
   ```

   Confirm failure because the reasons/analyzer do not exist.
3. Add the enum values and non-sensitive messages. Add a temporary analyzer node only after its first
   functional test exists in Task 2; update registry in the same green step.
4. Re-run the targeted tests and keep the registry test red until Task 2 provides the node.

## Task 2: Discover and parse root-aware hook documents

**Files:**

- Create: `src/skillspector/nodes/analyzers/bundled_execution_surface.py`
- Create: `tests/nodes/analyzers/test_bundled_execution_surface.py`
- Modify: `src/skillspector/nodes/analyzers/__init__.py`

1. Add a small state fixture using ordered `components` plus `local_file_cache`. Add failing tests for:
   plugin default hooks, inline/reference/mixed-array manifest hooks, root project/local settings,
   `SKILL.md`, command frontmatter, project-agent frontmatter, marketplace strict semantics, and ZIP
   virtual paths. Assert one BH1 per concrete source document and exact `source_kind` evidence.
2. Add false-positive controls for generic JSON, docs/fixtures, nested manifestless hooks, nested
   project settings, lowercase `skill.md` runtime-unconfirmed behavior, and archive namespace escape.
3. Add duplicate-key, malformed, wrong-type, missing-cache, and valid-plus-invalid isolation tests.
   The invalid source must fail its own ledger work while the valid source still emits findings.
4. Run the test module and record the expected import/behavior failures.
5. Implement immutable `HookDocument`/`HookRegistration` records, duplicate-key JSON loading,
   frontmatter loading, path/namespace helpers, root discovery, manifest/marketplace effective-source
   expansion, and per-document ledger ownership. Never read from disk; use
   `local_file_cache or file_cache` only.
6. Emit an initial safe BH1 with a full domain-separated digest first in `matched_text`; do not retain
   raw commands, URLs, headers, or frontmatter values in findings/evidence.
7. Register the analyzer after `static_yara` and make all Task 1/2 tests green.

## Task 3: Normalize runtime semantics and BH1 severity

**Files:**

- Modify: `src/skillspector/nodes/analyzers/bundled_execution_surface.py`
- Modify: `tests/nodes/analyzers/test_bundled_execution_surface.py`

1. Add table-driven failing tests for every documented event, matcher support, handler type, and known
   event/type compatibility. Cover ignored matchers, `FileChanged`, unknown declarations, `once`,
   `async`, decision/input-rewrite events, and activation lifetime.
2. Add tests for non-tool `if` dormancy and tool-event `if` match, non-match, parse-failure fail-open,
   and dynamic fail-open. Add plugin shell-form `${user_config.*}` rejection and exec-form acceptance.
3. Add LOW/MEDIUM/HIGH BH1 severity tests. Remote/dynamic HTTP, known command transports, unresolved
   reachable entrypoints, and unmodeled known-event handlers must be HIGH.
4. Run the focused tests to observe failures, implement the versioned semantics tables and pure
   normalization functions, then rerun.

## Task 4: Implement command-flow correlation and safe chain identity

**Files:**

- Create: `src/skillspector/nodes/analyzers/bundled_hook_flow.py`
- Create: `tests/nodes/analyzers/test_bundled_hook_flow.py`
- Modify: `src/skillspector/nodes/analyzers/bundled_execution_surface.py`

1. Add failing shell/exec tests proving:
   shell form is parsed only when `args` is absent; exec form treats arguments literally; real
   `bash -c`/PowerShell/cmd wrappers re-enter a shell parser; `echo`/registry/comment/quoted-text cases
   remain negative.
2. Add same-handler source/sink tests for sensitive file operands, ambient credential environment
   sources including auth headers, event stdin, HTTP/SSH/file-transfer/netcat/mail/DNS/cloud sinks,
   dynamic destinations, and statically proven loopback. Separate handlers must never correlate.
3. Add HTTP-handler event matrix tests: a non-loopback HTTP hook over a payload-rich event emits BH2
   from the implicit POST body; metadata-only, dormant, unknown-event, and loopback cases do not.
4. Implement typed `SourceKind`, `SinkKind`, and `DestinationClass` results. Analyze exec argv
   structurally and shell simple commands with bounded tokenization. A concrete tainted send to
   `dynamic_unknown` is outbound-capable; only proven loopback is negative.
5. Build full `sha256:` chain digests from domain tag, ordered normalized component keys/full content
   hashes, and source/sink/destination semantics. Use the full digest at the beginning of
   `matched_text`.
6. Assert every emitted evidence value is a flat allowlisted scalar and no supplied canary leaks.

## Task 5: Follow bounded referenced shell, Python, and JavaScript payloads

**Files:**

- Modify: `src/skillspector/nodes/analyzers/bundled_hook_flow.py`
- Modify: `tests/nodes/analyzers/test_bundled_hook_flow.py`

1. Add failing tests for `${CLAUDE_PLUGIN_ROOT}` and project-setting
   `${CLAUDE_PROJECT_DIR}` entrypoints, interpreters, `source`, and
   `cd "$CLAUDE_PLUGIN_ROOT" && ./script`. Prove bare plugin-relative paths and plugin
   `${CLAUDE_PROJECT_DIR}` do not resolve into the bundle.
2. Add shell/Python/JavaScript direct and bounded-variable source-to-sink fixtures plus two-wrapper
   chains. Assert BH2 is located at the concrete sink component and every traversed component affects
   the digest.
3. Add exact-boundary and boundary-plus-one tests for hop depth, component count, per-component size,
   and aggregate budget. Add cycles, missing cache, NUL/traversal/absolute/UNC/drive paths, archive
   namespace escape, binary, dynamic imports/eval, and unsupported native payloads.
4. Implement normalized cache-only resolution and bounded supported-language analysis. For reachable
   work, every unresolved or unmodeled condition produces one FAILED terminal ledger row and cannot
   fall back to filesystem reads. Dormant/unreachable files remain nonfatal.
5. Add multi-chain and intermediate-only mutation tests. One component ledger work item may own
   multiple emitted findings without duplicate work IDs.

## Task 6: Preserve structural findings and integrate score/baseline/report contracts

**Files:**

- Modify: `src/skillspector/nodes/analyzers/pattern_defaults.py`
- Modify: `src/skillspector/nodes/meta_analyzer.py`
- Modify: `src/skillspector/nodes/report.py`
- Modify: `src/skillspector/cli.py`
- Modify: `tests/nodes/test_meta_analyzer.py`
- Modify: `tests/nodes/test_report.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_suppression.py`

1. Add failing tests for BH defaults, structural-rule partition before provider batching, LLM rejection
   bypass, no-LLM parity, and complete meta ledger lineage.
2. Add failing tests for BH2 floor 51, `DO_NOT_INSTALL`, CLI exit 1, suppressed score zero, and fatal
   analysis taking precedence as exit 2 while retaining BH2 output.
3. Add baseline tests using `local_file_cache` for hidden/ZIP components. Generate a baseline, rescan
   unchanged, then mutate activation, intermediate wrapper, payload, and destination semantics; every
   mutation must invalidate exact suppression.
4. Add terminal/JSON/Markdown/SARIF tests with control/Markdown/Unicode/URL/header/secret canaries.
   Assert flat allowlisted evidence and no raw value appears in any rendered format.
5. Implement deterministic BH defaults, structural partition/rejoin, score floor, local-cache baseline
   lookup, and any necessary safe scalar rendering fixes. Re-run all touched suites.

## Task 7: Full graph, ZIP, CLI, performance, and corpus verification

**Files:**

- Create: `tests/integration/test_bundled_execution_surface.py`
- Create: `tests/fixtures/bundled_hooks/` fixtures as needed via `apply_patch`
- Modify: `README.md`

1. Add full-graph directory and ZIP tests for issue #399 Case A, direct Case C, referenced-script Case
   C, remote `UserPromptSubmit` HTTP implicit POST, and combined BH2-plus-fatal-incomplete state.
2. Add CLI subprocess coverage for JSON, Markdown, SARIF, baseline generation/rescan, exit 1, and exit
   2. Use real temporary artifacts, not mocked analyzer returns.
3. Add a one-million-character adversarial input timing test with a generous deterministic upper
   bound. Run the benign calibration corpus and assert zero BH2.
4. Scan pinned local NVIDIA/third-party catalogs if available; record exact paths/revisions and BH1/BH2
   counts. Absence is a disclosed corpus gap, not a fabricated pass.
5. Document BH1/BH2 sources, snapshot, exit behavior, evidence safety, and explicit BH3/non-goals.

## Task 8: Real Claude runtime E2E and final Review Guru gate

**Files:**

- Create: `tests/e2e/fixtures/claude_hooks/` only if reusable runtime fixtures add value
- Modify: draft PR notes only after user authorizes a push

1. Record `claude --version` and validate disposable default, inline, and referenced plugin fixtures
   using `claude plugin validate`.
2. With a loopback-only capture server and synthetic canary data, run the actual local Claude CLI to
   observe `SessionStart`, `UserPromptSubmit`, and a tool event; matcher-ignore, non-tool-`if`
   dormancy, command stdin, HTTP POST body, and exec-argv literal behavior. Never use an external
   destination or a real secret.
3. Where safe automation cannot cross auth/trust/model/UI boundaries, record the exact command and
   blocker; label those cases validator-only or parser-only.
4. Run an independent specification-conformance review, then a code-quality/security review. Fix every
   blocker through a new failing regression test and rerun the focused suite.
5. Run fresh final verification:

   ```bash
   uv run make lint
   uv run make format-check
   uv run make test-ci
   uv run make test-integration
   uv run python -m build
   ```

   Run Docker smoke only when a local Docker daemon is available. Inspect the complete diff, check
   generated artifacts and git status, and report exact passed/failed/skipped boundaries.
6. Keep the branch local for the user's requested review. Do not push or mark the draft ready without
   fresh authorization.
