# Bundled Hook Execution Surface Analysis

**Status:** Approved for implementation; amended after adversarial design review

**Date:** 2026-08-20

**Issue:** [#399](https://github.com/NVIDIA/SkillSpector/issues/399)

**Draft PR:** [#404](https://github.com/NVIDIA/SkillSpector/pull/404)

## Outcome

Add a deterministic, runtime-aware `bundled_execution_surface` analyzer that makes bundled Claude
Code hook declarations visible as BH1 findings and blocks installation when it can prove a BH2
sensitive-data-to-transport chain.

This first PR is deliberately hooks-only. It does not implement BH3 permission analysis because the
current Claude Code contract does not apply `permissions` from plugin-root `settings.json`.
Plugin-root settings currently support only `agent` and `subagentStatusLine`; unknown keys are
ignored. Project `.claude/settings.json` is a separate runtime surface and its hook declarations are
in scope, but its permission policy is not.

The design also corrects two assumptions in issue #399:

- Installation or workspace trust is the relevant user trust action. Once a hook is enabled, it
  fires automatically without a separate approval for each event; the design does not claim that a
  user is never prompted at all.
- A command hook with `args` uses direct exec semantics. Its arguments are literal argv elements and
  must not be concatenated with `command` and reinterpreted as shell source.

Because BH3 remains unresolved, draft PR #404 references `Part of #399` rather than using a closing
keyword.

## Goals

1. Identify supported hook declarations by schema and runtime location rather than by searching all
   JSON/YAML files for the word `hooks`.
2. Report one concise BH1 inventory finding per concrete hook document, even when every handler
   appears benign.
3. Emit BH2 only for a correlated source-to-sink chain within one handler and its bounded referenced
   entrypoints.
4. Preserve BH1 and BH2 deterministically in both LLM and no-LLM scans.
5. Fail closed, visibly and per work item, when an applicable hook document or referenced payload
   cannot be inspected.
6. Preserve existing report formats, baseline behavior, ledger accounting, and CLI exit semantics.
7. Verify static behavior against real Claude Code hook execution before claiming runtime parity.

## Non-goals

- BH3 permission-grant analysis.
- Plugin-root `settings.json` permission analysis.
- Background monitor, plugin MCP-server autostart, LSP-server, channel, workflow, or general `bin/`
  inventory beyond an executable reached through a documented hook command path.
- User-level or managed settings outside the scanned artifact.
- Plugin-shipped agent frontmatter hooks, which the current plugin contract rejects. Project
  `.claude/agents/` frontmatter hooks are a separate, valid project-runtime source and are in scope.
- Complete interprocedural analysis of arbitrary shell, Python, JavaScript, or native programs.
- Emulation of every historical Claude Code release. Findings state the semantics snapshot they use.

## Normative runtime basis

The implementation is based on the current official Claude Code documentation and records a
`claude_semantics_snapshot` constant in evidence and tests. At design time, the official docs describe
behavior through Claude Code 2.1.238, while the locally installed CLI is 2.1.227.

Primary references:

- [Hooks reference](https://code.claude.com/docs/en/hooks)
- [Plugins reference](https://code.claude.com/docs/en/plugins-reference)
- [Create plugins](https://code.claude.com/docs/en/plugins)
- [Permissions](https://code.claude.com/docs/en/permissions)
- [Claude Code changelog](https://code.claude.com/docs/en/changelog)

Static parser compatibility and observed runtime compatibility are reported separately. A parser
test derived from current documentation is not evidence that an older local CLI executes that shape.

## Supported declaration sources

The analyzer recognizes only root-aware runtime locations:

| Source kind | Accepted shape | Activation model | First-PR treatment |
|---|---|---|---|
| Plugin default | `<plugin-root>/hooks/hooks.json` with optional `description` and a root `hooks` event map | While plugin is enabled | Canonical plugin hook source |
| Plugin manifest inline | `.claude-plugin/plugin.json` whose `hooks` field is an event-map object | While plugin is enabled | Parse direct event map; accept a wrapped compatibility shape only when structurally unambiguous |
| Plugin manifest reference | Manifest `hooks` string or mixed array of `./` paths and inline objects | While plugin is enabled | Resolve each path inside the same plugin root/cache namespace and deduplicate repeated targets |
| Marketplace plugin definition | `.claude-plugin/marketplace.json` entry whose effective plugin definition declares inline or referenced `hooks` | While that marketplace plugin is enabled | Apply documented `strict` merge/replacement semantics and retain each plugin root |
| Project settings | Root `.claude/settings.json` with a `hooks` object | Interactive after workspace trust; `-p`/SDK treats the folder as trusted | Classify as `project_settings`, never as plugin-installed settings |
| Local project settings | Root `.claude/settings.local.json` with a `hooks` object | Same project, local scope | Scan if the artifact contains it; retain local-scope evidence |
| Skill frontmatter | Root/project/plugin skills, including manifest-declared custom skill directories, whose `SKILL.md` YAML frontmatter has `hooks` | From invocation through the rest of the session, or once when configured | Parse the hook map and record invocation-gated lifetime; lowercase `skill.md` is parser compatibility only and is labeled runtime-unconfirmed |
| Command frontmatter | Project or plugin command Markdown, including manifest-declared custom command directories, whose YAML frontmatter has `hooks` | From command invocation through the rest of the session | Parse the same hook schema as skill frontmatter and record invocation-gated lifetime |
| Project agent frontmatter | Root `.claude/agents/*.md` whose YAML frontmatter has `hooks` | While the project subagent runs | Parse as project-runtime hooks; plugin-shipped agent hooks remain rejected/out of scope |

The analyzer does not treat a generic `package.json`, documentation fixture, or arbitrary nested file
as active merely because it has a `hooks` key.

### Root discovery

Plugin roots are derived as follows:

1. For each `<plugin-root>/.claude-plugin/plugin.json`, the plugin root is the parent of the
   `.claude-plugin` directory, not the manifest's immediate parent.
2. The scan root is allowed to be a manifestless plugin root when it contains root
   `hooks/hooks.json`; plugin manifests are optional.
3. A nested `hooks/hooks.json` requires a sibling `.claude-plugin/plugin.json`. This prevents
   examples, fixtures, and documentation trees from being promoted to active plugin roots.
4. Archive members retain their virtual `outer.zip!/member` namespace. A manifest and every file it
   activates must remain in the same archive namespace.
5. Project settings are recognized only at the scan root. A plugin repository's
   `.claude/settings.json` is a project setting that affects work performed in that repository; it is
   not installed as plugin configuration.
6. Skill and command frontmatter is inspected only at documented root/project/plugin locations and
   manifest-declared custom component paths. Project agent frontmatter is inspected only below root
   `.claude/agents/`. Generic nested Markdown remains dormant fixture/content.
7. Marketplace plugin definitions derive independent plugin roots and apply `strict: true` as a merge
   with that plugin's manifest, or `strict: false` as the complete definition. A declared runtime
   source that cannot be mapped to a cache-contained plugin root is a visible incomplete analysis.

When a manifest declares custom hook paths and default `hooks/hooks.json` is also present, the analyzer
inspects both declarations, deduplicates the same physical/cache component, and records conservative
activation evidence. Current documentation is not explicit enough about every default-versus-custom
precedence combination; live E2E determines whether a declaration is labeled runnable or merely
declared under the pinned runtime. It is never silently omitted.

Multiple inline hook objects in one manifest are aggregated into one manifest-backed
`HookDocument`; each distinct referenced configuration file is its own document. This keeps BH1
concise while retaining per-handler identity for BH2.

### Trust, enablement, and external policy

Findings describe the capability of the scanned artifact after the ordinary trust/enable action for
that source. They record whether a plugin defaults disabled, a skill requires invocation, or project
hooks require workspace trust. They do not claim that those conditions have already occurred.

User/managed settings, CLI overrides, `allowedHttpHookUrls`, `httpHookAllowedEnvVars`, and
`disableAllHooks` can change effective runtime behavior outside the artifact. Those external controls
are recorded as unknown policy and are not accepted as a mitigation for untrusted bundled code.
Handler-local semantics that intrinsically prevent spawning, such as an `if` on a non-tool event or
an unsupported event/type combination, do make that registration non-runnable for BH2.

## Normalized model

Parsing produces immutable internal records before classification:

```text
HookDocument
  source_kind
  source_path
  plugin_or_project_root
  activation_lifetime
  document_shape
  content_digest
  registrations[]

HookRegistration
  event
  event_status
  matcher
  matcher_kind
  matcher_effective
  handler_type
  handler_status
  if_rule_present
  runnable
  once
  async
  command_mode
  chain_digest
  referenced_components[]
```

Raw commands, URLs, headers, prompts, environment values, event payloads, and script excerpts do not
enter this normalized reporting model. Classifiers operate on raw content locally but return typed
enums, booleans, counts, line numbers, normalized paths, and full opaque SHA-256 chain digests. Short
digest prefixes are display-only and are never used for identity, deduplication, or suppression.

## Event, matcher, and handler semantics

The implementation owns a tested table of documented hook events, matcher behavior, input-data
classes, decision capabilities, and supported handler types.

### Matchers

- Omitted, empty, or `*` matchers are broad.
- Exact-list and JavaScript-regex matcher syntax is classified according to the documented event.
- `FileChanged` uses literal filename-watch behavior, not ordinary regex behavior.
- On events without matcher support, the matcher is ignored and the registration is broad. The
  current no-matcher set includes `UserPromptSubmit`, `PostToolBatch`, `Stop`, `TeammateIdle`,
  `TaskCreated`, `TaskCompleted`, `WorktreeCreate`, `WorktreeRemove`, `MessageDisplay`, and
  `CwdChanged`.
- An unknown event is retained as an unconfirmed declaration. BH1 reports it without claiming that
  the current runtime executes it, and BH2 is not emitted from it.

### `if`

- `if` is evaluated only for `PreToolUse`, `PostToolUse`, `PostToolUseFailure`,
  `PermissionRequest`, and `PermissionDenied`.
- On every non-tool event, a handler containing `if` is dormant under the current semantics snapshot.
- A dormant declaration remains in BH1 inventory with `runnable=false`; it cannot contribute BH2.
- On supported tool events, `if` is best-effort. A statically resolved non-match is dormant, a match is
  runnable, and a parse failure or dynamic/unresolved condition fails open and is classified broad.
- Historical pre-2.1.85 behavior is not emulated. The evidence identifies the current semantics
  snapshot so consumers do not mistake the result for an all-version claim.

### Handler compatibility

All five current handler types are inventoried: `command`, `http`, `mcp_tool`, `prompt`, and `agent`.
Known unsupported event/type combinations are marked non-runnable. Unknown handler types are retained
as unmodeled declarations and raise BH1 severity because SkillSpector cannot safely characterize a
future or malformed runtime surface; they do not produce BH2 without a proven sink.

The pinned compatibility table has three handler groups:

- all five types on `PermissionDenied`, `PermissionRequest`, `PostToolBatch`, `PostToolUse`,
  `PostToolUseFailure`, `PreToolUse`, `Stop`, `SubagentStop`, `TaskCompleted`, `TaskCreated`,
  `TeammateIdle`, `UserPromptExpansion`, and `UserPromptSubmit`;
- `command`, `http`, and `mcp_tool` on `ConfigChange`, `CwdChanged`, `DirectoryAdded`, `Elicitation`,
  `ElicitationResult`, `FileChanged`, `InstructionsLoaded`, `MessageDisplay`, `Notification`,
  `PostCompact`, `PreCompact`, `SessionEnd`, `StopFailure`, `SubagentStart`, `WorktreeCreate`, and
  `WorktreeRemove`;
- `command` and `mcp_tool` only on `SessionStart` and `Setup`.

The table is versioned with the semantics snapshot. A newly documented event remains an unconfirmed
BH1 declaration until the table and its input-data class are deliberately updated.

### Command execution modes

Command handlers have two distinct parsers:

- **Exec form:** `args` is present, including `args: []`. `command` is one executable and each
  argument is literal. `shell` is ignored. Shell metacharacters in an argument are data.
- **Shell form:** `args` is absent. The command is parsed as shell source with the documented
  platform/shell choice.

Under the pinned plugin contract, shell-form commands containing `${user_config.*}` are rejected and
are marked non-runnable; exec-form fields may use the documented substitution. This rule is source-
specific and must not be generalized to ordinary environment interpolation.

Exec form is never joined and reparsed as shell. Only a real shell-interpreter invocation such as
`bash -c`, `sh -c`, `zsh -c`, `pwsh -Command`, `powershell -Command`, or `cmd /c` causes the relevant
payload argument to enter a nested shell parser.

Examples that must stay negative:

- `echo` with literal argv that mentions `curl`, a URL, and `.env`.
- a package-manager `--registry=https://...` argument.
- comments or quoted documentation strings that merely name a transport.

## BH1 — bundled hook declaration

BH1 is a deterministic inventory finding, consolidated to one finding per concrete `HookDocument`.
It is emitted whenever the document declares at least one handler, including dormant or unmodeled
handlers, so structural visibility does not depend on a suspicious payload string.

The message reports counts and the highest effective risk class. Evidence contains only the safe
schema described below.

### BH1 severity

The document's severity is the maximum of its handler classifications:

| Severity | Conditions |
|---|---|
| LOW | All declarations are narrow, local, post-event/non-controlling handlers, one-shot handlers, currently dormant declarations with no transport, or unknown-event candidates with no proven runnable transport |
| MEDIUM | Any runnable ambient/broad local command, prompt, agent, or MCP hook; a local loopback HTTP hook; or a local handler on a decision/input/output-control event |
| HIGH | Any non-loopback or dynamic HTTP destination; a known command transport even without a proven sensitive source; an unresolved referenced entrypoint; an unknown handler type on a known event; or MCP input that forwards sensitive event fields to a destination that cannot be resolved |

BH1 alone does not force `DO_NOT_INSTALL`. It supplies reviewable execution-surface context and a
bounded risk contribution.

## BH2 — bundled hook exfiltration

BH2 is CRITICAL with confidence 1.0 and is emitted only for a proven correlated chain:

```text
runnable hook activation
  -> sensitive source
  -> concrete outbound sink
```

The source and sink must occur in the same handler or in a bounded entrypoint chain reachable from
that handler. SkillSpector never combines a source found in one registration with a sink found in
another.

### Sensitive sources

The first implementation recognizes:

1. Sensitive local file reads or upload operands, including credential stores, private keys, agent
   configuration, shell history, cloud credentials, and explicit secret files.
2. Sensitive environment values whenever they are placed into any outbound request field, including
   payloads, query parameters, uploaded files, or headers. An ambient credential such as a cloud,
   source-control, or signing token does not become safe merely because it is labeled an authorization
   header. The only negative exception is a plugin-owned setting declared as sensitive `userConfig`,
   used solely as authentication to one statically known service origin; runtime-controlled origins or
   mixed payload/header use remain outbound-capable.
3. Sensitive hook event data when the event schema carries user, assistant, tool, task, compacted, or
   elicitation content.

The event-data table is allowlisted and versioned. It includes prompt text, expanded prompt content,
tool inputs/results/errors, parallel batch results, displayed/assistant messages, task descriptions,
compaction content, and elicitation request/response content where documented. Common fields such as
`transcript_path`, `cwd`, IDs, and `permission_mode` are metadata; `transcript_path` is not treated as
the transcript's contents.

### Outbound sinks

Recognized sinks include concrete upload/send forms of:

- HTTP clients such as `curl`, `wget`, and supported Python/JavaScript send APIs.
- `ssh`, `scp`, `sftp`, and remote-form `rsync`.
- `nc`/`ncat`/`netcat`, `socat`, and `/dev/tcp`.
- mail senders and DNS payloads such as `dig` when data is encoded into the query.
- supported cloud/object-store upload APIs.

A URL literal is not a sink by itself. A local copy or local `rsync` is not outbound. Loopback HTTP is
not remote exfiltration. Private, link-local, and non-loopback internal destinations remain outbound
because they cross the local process/host trust boundary. Destination classification is three-valued:
statically proven loopback is negative; statically proven non-loopback is outbound; and dynamic or
runtime-controlled is outbound-capable when a concrete send operation receives tainted data. Unknown
destinations never turn a proven source-to-send flow into a BH2 bypass.

### Implicit event transport

- A non-loopback `http` handler always POSTs the complete event JSON. A runnable HTTP handler on a
  sensitive-data event therefore satisfies BH2 without a path literal in the configuration.
- Every command handler receives event JSON on stdin. A command or referenced script that forwards
  stdin using forms such as `curl --data-binary @-`, `wget --post-file=-`, `nc`, `ssh host cat`, or a
  mail body satisfies the source half when its event carries sensitive data.
- Merely receiving stdin is not a sink. The command chain must actually consume/forward it.

### Referenced payloads

BH2 follows only literal, bundle-resolvable entrypoints:

- `${CLAUDE_PLUGIN_ROOT}/...` for plugin hooks.
- `${CLAUDE_PROJECT_DIR}/...` for root project settings.
- interpreter argv that names one of those paths.
- documented shell forms such as a quoted placeholder path, `source`, or
  `cd "$CLAUDE_PLUGIN_ROOT" && ./script`.

Bare `./script` and bare `bin/tool` in a plugin hook are not assumed plugin-relative because hooks run in the session
working directory. `${CLAUDE_PROJECT_DIR}` in a plugin hook refers to the user's project, not bundled
plugin content. `${CLAUDE_PLUGIN_DATA}` is persistent runtime state, not shipped content.

Resolution uses `local_file_cache` only. The analyzer never calls `Path.open`, follows a symlink, or
re-reads the filesystem after discovery. It rejects NULs, absolute/UNC/drive paths, `..` segments,
namespace changes, and missing cache members. Archive paths cannot escape their existing `!/`
namespace.

Traversal is bounded to two literal wrapper hops, eight referenced components per handler, and a
two-million-character aggregate payload budget. Cycles are detected by normalized cache key. For a
runnable or reachable payload, `DEPTH_LIMIT`, `COMPONENT_LIMIT`, `AGGREGATE_BUDGET`, `SIZE_LIMIT`,
`BINARY_CONTENT`, `UNMODELED_PAYLOAD`, missing cache content, dynamic entrypoints, and unsupported
languages are terminal `FAILED` work items and force analysis-incomplete/CLI exit 2 while preserving
findings from other sources. The same limitation on a proven dormant declaration can be nonfatal.
No analysis limit may degrade to BH1/CAUTION with exit 0 for runnable work.

Within supported shell, Python, and JavaScript payloads, BH2 requires direct source-to-sink use or
bounded local variable propagation. The supported subset is explicit: shell simple commands,
assignments, pipelines, `source`, and documented interpreter wrappers; Python AST assignments and
supported call arguments; JavaScript/TypeScript literal imports/requires, local assignments, stdin or
environment sources, and supported send/upload call arguments. Dynamic evaluation, computed imports,
opaque subprocess construction, native executables, and flows outside that subset are
`UNMODELED_PAYLOAD` for reachable work rather than guessed safe. Python flow logic reuses or extracts
the existing behavioral taint primitives rather than implementing a competing unbounded engine.

## Stable finding and evidence contract

BH1 and BH2 evidence is flat and contains scalar values only. Allowed fields are:

```json
{
  "schema": "skillspector.bundled_hook.v1",
  "claude_semantics_snapshot": "2.1.238",
  "source_kind": "plugin_default",
  "declaration_roles": "plugin_default,plugin_manifest_reference",
  "activation_lifetime": "plugin_enabled",
  "runtime_status": "runnable",
  "handler_count": 2,
  "runnable_handler_count": 2,
  "ambient_handler_count": 1,
  "handler_types": "command,http",
  "events": "PostToolUse,UserPromptSubmit",
  "chain_digest": "sha256:<full-64-hex-digest>",
  "transport_kind": "http",
  "destination_class": "public_remote",
  "sensitive_source_kind": "user_prompt_event",
  "payload_component": "scripts/telemetry.js",
  "component_count": 2
}
```

Inapplicable fields are omitted. Raw command text, full URLs, URL userinfo/query strings, headers,
environment variable values, secret-bearing variable names, prompts, tool data, or script snippets
are forbidden in message, context, matched text, and evidence.

`matched_text` starts with one full, domain-separated `chain_digest` before any descriptive token. The
digest hashes the ordered normalized cache keys and full content hashes of the activation document and
every traversed wrapper/payload, plus normalized source kind, sink kind, and destination class. It is
used for identity and suppression; the report may separately display a prefix. A cross-file BH2 is
located at the concrete sink component. Exact baseline fingerprints therefore change when an
activation, intermediate wrapper, terminal payload, or source/sink/destination semantic changes.

When multiple declarations activate the same cache component, `source_kind` retains the canonical
primary role and `declaration_roles` lists every normalized role in lexical order. The component is
parsed once and owns one terminal ledger row; a declaration cycle is invalid configuration rather than
an invitation to re-run or silently discard an activation edge.

## Meta-analysis and reporting

BH1 and BH2 are structural facts, not LLM opinions. `meta_analyzer` partitions structural findings
before provider batching, never sends their IDs/content to an LLM, applies deterministic defaults, and
rejoins them unchanged in both LLM and no-LLM paths with complete ledger lineage. This is an explicit
structural-rule policy; it does not misuse the `local-only` tag.

No new report-only summary channel is introduced. BH1 is the visible inventory in terminal, JSON,
Markdown, and SARIF. Existing reports continue to render findings and flat sanitized evidence.
Tests verify control-character removal, stable JSON/SARIF properties, Markdown-safe scalar rendering,
and absence of raw commands/secrets in every format.

`pattern_defaults.py` supplies BH1/BH2 category, explanation, and remediation defaults so preserved
findings remain complete without LLM enrichment.

## Scoring and CLI gate

One confidence-1.0 CRITICAL finding currently contributes exactly 50 points, while the install gate
blocks only above 50. The report therefore adds `BH2: 51` to the existing severity-floor table.

For an unsuppressed BH2:

- risk score is at least 51;
- recommendation is `DO_NOT_INSTALL`;
- CLI scan exits 1;
- maximum issue severity remains `CRITICAL`, even if the normalized score band is `HIGH`.

Suppressed BH2 findings do not contribute score or a floor. Analyzer/accounting failure remains exit
2 and is not conflated with a security verdict.

## Ledger and failure contract

Every analyzer work item has exactly one terminal ledger event:

- `COMPLETED` for a parsed hook document or inspected referenced component, with every emitted
  finding ID listed once.
- `FAILED / SIZE_LIMIT` for an oversized runnable/reachable applicable file; a proven dormant file may
  be skipped without making the scan fatal.
- `FAILED / MISSING_FILE_CACHE` when an inventoried applicable file has no cache entry.
- `FAILED / INVALID_CONFIGURATION` for malformed JSON/YAML, duplicate keys, or a structurally invalid
  hook field.
- `FAILED / DEPTH_LIMIT`, `COMPONENT_LIMIT`, `AGGREGATE_BUDGET`, or `UNMODELED_PAYLOAD` when bounded
  analysis of runnable/reachable work cannot establish behavior.
- `FAILED / ANALYZER_RUNTIME_ERROR` for an unexpected isolated classifier failure.

The new reasons are allowlisted and payload-free. Unknown events and handler types are validly parsed
declarations, not parser failures, but a reachable unmodeled handler/payload remains incomplete.

One source failure does not discard findings from another source. `analyzer_status_for_events`
derives the analyzer status from exact planned work. Referenced components have one terminal event per
normalized cache key; that event may own multiple emitted BH2 IDs. The full chain digest binds the
activation document and every intermediate component without inventing duplicate ledger work IDs.

The analyzer consumes deterministic `components` order and `local_file_cache or file_cache`, matching
hidden and nested artifact policy. Baseline generation is updated to use the local cache so findings
on hidden hook sources can be fingerprinted without failing.

## Repository changes

The implementation is expected to touch these boundaries:

- `src/skillspector/nodes/analyzers/bundled_execution_surface.py`
  - source discovery, parser, normalization, runtime semantics table, BH1 classification, bounded
    orchestration, and analyzer node.
- `src/skillspector/nodes/analyzers/bundled_hook_flow.py`
  - shell/exec separation, transport and sensitive-source classification, supported script flows,
    cache-only reference resolution, and chain identity.
- `src/skillspector/nodes/analyzers/__init__.py`
  - register immediately after `static_yara`; the graph auto-wires registry entries.
- `src/skillspector/nodes/analyzers/pattern_defaults.py`
  - BH1/BH2 defaults.
- `src/skillspector/nodes/meta_analyzer.py`
  - deterministic structural-rule pass-through in LLM and fallback paths.
- `src/skillspector/inspection_ledger.py`
  - payload-free invalid-configuration reason.
- `src/skillspector/nodes/report.py`
  - BH2 risk floor and evidence-format regression coverage.
- `src/skillspector/cli.py`
  - baseline creation uses the local deterministic cache.
- `README.md` or a focused security-rule document
  - explain BH1/BH2, supported sources, semantics snapshot, and non-goals.

The analyzer uses two internal modules to keep schema/runtime normalization separate from payload-flow
analysis. Neither module is a public API. Pure boundaries are `HookDocument`, `HookRegistration`,
source discovery, parsing, activation classification, transport classification, sensitive-source
classification, safe reference resolution, chain identity, and finding construction.

## Test strategy

Implementation follows red-green-refactor. Tests are added before each behavior and are organized so
parser, semantics, correlation, graph, output, and live-runtime failures are distinguishable.

### Unit and property matrix

1. **Source discovery and parsing**
   - default plugin wrapper;
   - manifest direct inline map, wrapped compatibility map, string path, mixed array, duplicate refs;
   - project and local settings with unrelated keys;
   - recognized skill, command, and project-agent frontmatter;
   - marketplace strict merge/replacement and manifest custom skills/commands/hooks paths;
   - nested plugin roots and nested archive namespaces;
   - package/docs/fixture false-positive controls;
   - malformed JSON/YAML, duplicate keys, wrong types, missing cache, binary, and size limit.
2. **Runtime semantics**
   - every documented event and handler-type compatibility row;
   - unknown event/type retention without false runnable claims;
   - omitted/empty/`*`, exact, regex, ignored, and `FileChanged` matchers;
   - non-tool `if` dormancy and tool-event `if` match/non-match/fail-open behavior;
   - plugin shell-form `${user_config.*}` rejection and exec-form substitution;
   - `once`, async, decision-capable, and invocation-gated lifetime evidence.
3. **Shell versus exec**
   - absent `args`, empty `args`, literal metacharacters, interpreter `-c` forms, Windows shell forms;
   - real exec-form transport arguments;
   - `echo`/registry/comment/quoted-literal negatives.
4. **BH2 correlation**
   - inline sensitive path plus each transport family;
   - source and sink split across command/args while preserving exec field boundaries;
   - source and sink in different handlers stays negative;
   - remote HTTP event-payload matrix;
   - command stdin forwarding matrix;
   - ambient credential in auth header positive; declared sensitive `userConfig` to one static service
     origin negative; URL-only, path-only, loopback, local-rsync, and transcript-path negatives;
   - dynamic destination plus a concrete tainted send positive;
   - referenced shell/Python/JavaScript direct and bounded-variable flows;
   - wrapper depth, cycles, traversal, symlink absence, namespace escape, and aggregate budget.
5. **Identity and safety**
   - canonical matched-text prefixes do not deduplicate distinct sources/chains;
   - only flat allowlisted evidence is emitted;
   - control, Unicode, Markdown, URL userinfo/query, header, and secret-value injection cannot leak.
6. **Meta, ledger, scoring, and baseline**
   - LLM rejection and no-LLM fallback both preserve BH1/BH2 IDs and evidence;
   - exactly one producer origin per finding;
   - one malformed source does not erase another source's finding;
   - BH2 floor 51, `DO_NOT_INSTALL`, CLI exit 1;
   - parser/accounting failure exits 2;
   - suppressed BH2 scores zero;
   - baseline mutation invalidates when only activation, intermediate wrapper, or payload changes;
   - hidden/nested findings can generate a baseline from local cache.

### Graph and output verification

- Full graph scans for direct directories and ZIP inputs in `--no-llm` mode.
- A controlled fake-LLM integration that attempts to reject BH1/BH2.
- Terminal, JSON, Markdown, and SARIF snapshots/assertions for findings, severity, evidence,
  completeness, suppression, and exit behavior.
- Registry order and analyzer-status/completeness tests.

### Performance and corpus verification

- A one-million-character adversarial command/config input pins bounded runtime and guards against
  catastrophic regex behavior.
- Current pinned checkouts of NVIDIA's skills catalog and real third-party hook plugins measure BH1
  volume and require zero BH2 false positives before the implementation is pushed.
- The benign calibration set includes formatter hooks, release/auth headers, registry URLs, health
  checks, comments, `.env.example`, and literal argv examples.

### Live Claude Code E2E

The deepest practical verification uses disposable fixtures and local-only capture:

1. Run `claude plugin validate` on default, inline, and referenced hook fixtures.
2. Run an enabled plugin fixture and capture actual `SessionStart`, `UserPromptSubmit`, and tool-event
   firings.
3. Prove a matcher on `UserPromptSubmit` is ignored, a non-tool `if` handler is dormant, and exec
   `args` metacharacters remain literal.
4. Capture an HTTP hook body at a loopback test server and compare its fields with the event-data
   table. No external endpoint or real secret is used.
5. Exercise project-settings trust behavior in interactive and `-p` modes where automation permits.
6. Record exact CLI versions. Run the local 2.1.227 CLI and, if a safely isolated pinned 2.1.238
   runner is practical, repeat the version-sensitive cases there.

If authentication, model cost, interactive trust UI, or runtime availability prevents a case, the PR
must state exactly which cases were parser-only, validator-only, or live-executed. Unit tests and
shaped captures are not described as runtime E2E.

### Repository-wide verification

Before implementation completion:

- targeted analyzer/meta/report/CLI tests;
- `uv run make lint`;
- `uv run make format-check`;
- `uv run make test-ci`;
- integration tests that do not require unavailable provider credentials;
- Docker build/smoke when the local Docker service is available;
- a final edge-case review covering event/type interactions, shell/exec behavior, report leakage,
  score/suppression behavior, ledger completeness, and regressions.

## Acceptance criteria

The first implementation is ready to push to draft PR #404 only when all of the following are true:

1. Case A from issue #399 emits one deterministic BH1 finding instead of risk 0/SAFE.
2. Direct and referenced-script Case C variants emit BH2 and independently produce
   `DO_NOT_INSTALL`/exit 1.
3. Remote `UserPromptSubmit` HTTP exfiltration emits BH2 without requiring a sensitive path literal.
4. Shell and exec forms produce the documented positive and negative results.
5. Invalid/oversized/unresolved/unmodeled runnable inputs are visible, make analysis incomplete, and
   exit 2 rather than becoming an unqualified SAFE/CAUTION result.
6. The benign formatter/configured-service-auth-header/registry/comment corpus emits no BH2, while an
   ambient credential in an outbound header does emit BH2.
7. Findings and evidence contain no raw command, secret, header, prompt, tool payload, or full remote
   URL.
8. Unit, graph, output, CLI, performance, corpus, and deepest-practical live tests are reported with
   exact pass/fail/skip boundaries.
9. BH3 remains absent and issue #399 remains open or is explicitly tracked by a separately approved
   follow-up.

## Design review resolution

Three independent review tracks evaluated the threat model, Claude runtime semantics, and current
SkillSpector integration contracts. Their blocking findings are incorporated here:

- skill frontmatter, local settings, and manifest-array bypasses are covered;
- HTTP and command-stdin implicit event exfiltration are modeled;
- source/sink correlation is handler-local;
- script resolution is cache-only and namespace-contained;
- structural findings bypass LLM filtering;
- BH2 has an independent blocking score floor;
- evidence, deduplication, baselines, ledger failure, and PR-closing semantics are explicit.

With those changes and the user's written approval, the design is ready for production implementation.
