# Bundled Permission Grant Analysis

**Status:** Approved for implementation after design and architecture review

**Date:** 2026-08-24

**Issue:** [#399](https://github.com/NVIDIA/SkillSpector/issues/399)

**Depends on:** Draft PR [#404](https://github.com/NVIDIA/SkillSpector/pull/404), which introduces
the `bundled_execution_surface` analyzer and BH1/BH2 hook analysis

**Implementation PR:** Draft PR [#429](https://github.com/NVIDIA/SkillSpector/pull/429)

**Implementation branch:** `feat/christopherk/issue-399-permission-surface`

## Outcome and PR boundary

Add deterministic BH3 analysis for permission-bearing Claude Code project settings shipped in a
scanned artifact. BH3 makes structurally effective permission grants visible, distinguishes a
declared capability from proven runtime activation, and blocks installation for the narrow set of
grants that can remove the approval boundary or expose the filesystem root or home directory.

Issue #399 is implemented as two reviewable draft PRs:

1. PR #404 implements the hook surface: BH1 inventory and BH2 correlated hook exfiltration.
2. Draft PR #429 implements BH3 and completes the remaining settings scope.

Both PRs use `Part of #399`; neither PR closes the issue alone. The BH3 diff is reviewed against PR
#404's branch even if GitHub temporarily displays the dependent draft against `main`.

## Corrections to the issue proposal

The implementation intentionally differs from the issue's original sketch where later runtime
evidence established a narrower contract:

- A plugin-root `settings.json` is not an installed permission source. Only exact project settings
  roots are in scope.
- Shared-project `permissions.allow` and `permissions.additionalDirectories` do not become active in
  a never-trusted workspace merely because headless project hooks load.
- Project-local grants are provenance-dependent: an untracked local settings file can apply before
  trust, while a Git-tracked local settings file is treated like repository-controlled content.
- Since Claude Code 2.1.142, `defaultMode: "auto"` is ignored in project and local settings. Under
  the pinned 2.1.241 semantics it produces an internal diagnostic, not BH3.
- Permission mode support differs across CLI, IDE, Desktop, web, remote-control, cloud, and Agent
  SDK entrypoints. A static artifact scan reports the declared capability and the relevant
  activation condition; it never claims that a particular session activated it.

## Goals

1. Analyze `permissions` structurally only in exact settings roots that Claude Code can treat as
   project settings.
2. Parse each physical settings JSON document once, then analyze its `hooks` and `permissions`
   sections independently.
3. Emit at most one sanitized BH3 finding per physical settings document.
4. Preserve BH1/BH2 when a permission sibling is invalid, and preserve BH3 when a hook sibling is
   invalid.
5. Distinguish shared-project trust, local-file provenance, interface support, and external policy
   from the capability declared by the artifact.
6. Apply a score floor only to unsuppressed BH3 findings that explicitly carry a boolean
   `blocking_critical: true` evidence value.
7. Keep baseline, terminal, JSON, Markdown, SARIF, ledger, archive, and CLI exit contracts intact.
8. Fail closed, with a terminal ledger result, when an applicable permission document cannot be
   safely interpreted.

## Non-goals

- User settings at `~/.claude/settings.json`, managed settings, MDM, server-managed settings, or
  global state in `~/.claude.json`.
- Plugin-root `settings.json`, `.claude-plugin/settings.json`, or a settings-like JSON file in an
  ordinary nested directory.
- Proving that the scanned `.claude/settings.local.json` is tracked, ignored, untracked, symlinked,
  copied, or present on a future consumer's machine.
- Predicting higher-precedence CLI, user, local, managed, IDE, organization, or host-process policy.
- Reporting restrictive `deny`/`ask` policy as a vulnerability.
- Treating every narrow, intentional permission rule as a finding.
- Emulating permission grammars from every historical Claude Code release.
- Adding another graph analyzer node. BH3 extends the existing `bundled_execution_surface` node.

## Normative runtime basis

BH3 is pinned to Claude Code **2.1.241** and records that value in finding evidence. The normative
references are:

- [Claude Code settings](https://code.claude.com/docs/en/settings)
- [Configure permissions](https://code.claude.com/docs/en/permissions)
- [Permission modes](https://code.claude.com/docs/en/permission-modes)
- [Pre-trust behavior](https://code.claude.com/docs/en/permissions#what-runs-before-you-trust-a-folder)

Disposable 2.1.241 startup probes informed this design:

- a never-trusted shared settings file did not activate `permissions.allow` or
  `permissions.additionalDirectories`;
- an untracked local settings file was accepted without the shared trust warning, while the same
  file added to a Git index was trust-gated;
- project/local `defaultMode: "auto"` was ignored as repository-controllable configuration; and
- project `defaultMode: "bypassPermissions"` was recognized as a startup mode request, subject to
  the separate disable and interface controls.

These probes pin classification. They are not evidence that a final SkillSpector build performed a
real tool call under each mode. Final implementation verification repeats the safe startup probes
and reports any authentication, model, UI, or interface boundary that prevents deeper execution.

## Applicable settings roots

The analyzer uses the existing cache-key namespace parser. A settings file is applicable only when
the member path inside its current namespace has exactly two components:

| Exact member path | `source_kind` | Included |
|---|---|---|
| `.claude/settings.json` | `project_settings` | Yes |
| `.claude/settings.local.json` | `project_local_settings` | Yes |
| `settings.json` | none | No |
| `.claude-plugin/settings.json` | none | No |
| `plugin/settings.json` | none | No |
| `example/.claude/settings.json` | none | No |
| `plugin/.claude/settings.json` | none | No |

The same exact-root rule applies independently inside every real archive namespace:

- `bundle.zip!/.claude/settings.json` is applicable.
- `outer.zip!/inner.zip!/.claude/settings.local.json` is applicable at the nested archive root.
- `bundle.zip!/example/.claude/settings.json` is not applicable.
- A suffix lookalike such as `bundle.zip!/settings.json` is not applicable.

Archive namespace boundaries are identity boundaries. A document in one namespace cannot acquire a
role, mitigation, or grant from another namespace.

The applicability rule is independent of the JSON content. A root settings document with a
`permissions` key is applicable to BH3 whether or not it declares hooks. A root settings document
with only unrelated settings is not applicable and produces no bundled-settings ledger row.

## One parse and one physical owner

`bundled_execution_surface.node` remains the sole owner of discovery, cache access, JSON parsing,
finding attachment, ledger rows, ordering, and analyzer status. It performs duplicate-key-safe JSON
loading once for each applicable physical settings path and retains the parsed mapping in a
path-keyed settings work record.

The parsed mapping is passed independently to:

- existing hook normalization when the root contains `hooks` or when a plugin manifest later
  declares the same path as a hook reference; and
- the new pure permission helper when the root contains `permissions`.

The permission helper never opens files, reads Git state, resolves symlinks, parses JSON, mutates
graph state, or emits ledger rows. The existing dynamic analyzer registry is unchanged.

The surface also performs source-location recovery from the cached JSON syntax tree. This is not a
second semantic JSON load: it produces only a frozen, sanitized `PermissionSourceLines` record of
positive line numbers for permission-key positions and known list indexes. `permission_key_lines`
aligns with the parsed mapping's insertion order, so an unknown-key diagnostic can recover its line
without retaining that key. No JSON value or unknown key name crosses the record boundary. If
location recovery cannot identify an entry, its line falls back to the enclosing
`permissions` line, then line 1. Location recovery cannot turn otherwise valid JSON into a failed
permission analysis.

Before calling the pure helper, the surface also hashes the normalized physical cache path,
including every archive namespace, as SHA-256 over
`b"skillspector.bundled_permission.source.v1\0" + normalized_cache_path.encode("utf-8")`. It passes
only that full `sha256:` source-identity digest plus the full content digest. Identical settings
bytes at two physical/cache identities therefore produce distinct BH3 aggregate identities without
placing a raw path inside the helper records or evidence.

`handled_paths` is not permission ownership. In particular, a permissions-only settings path must
not be skipped if a manifest later references it as a hook document. Hook roles are evaluated from
the retained parsed mapping. This produces one combined terminal result for the physical settings
path rather than one permission row followed by a duplicate hook row.

For a physical settings document, BH1, BH2, and BH3 finding IDs are attached to one producer ledger
row at the path with phase `bundled_settings`. Non-settings hook documents and reachable payload
work retain phase `bundled_hook`. A line-ranged reachable payload remains its own work item; it does
not create a second path-level parse owner for the settings document.

## Pure normalized model

Create `src/skillspector/nodes/analyzers/bundled_permission_grants.py` with frozen, private data
records. Public graph state does not expose them.

```python
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

The boundary functions are:

```python
def analyze_permission_grants(
    raw: Mapping[str, object],
    *,
    source_kind: str,
    content_digest: str,
    source_identity_digest: str,
    source_lines: PermissionSourceLines,
) -> PermissionAnalysis: ...


def build_bh3_finding(
    analysis: PermissionAnalysis,
    *,
    source_path: str,
) -> Finding | None: ...
```

The implementation may add closed enums for the string values, but these signatures and field
meanings are stable. The helper returns `applicable=False`, `outcome=None`, and no digest when the
mapping has no `permissions` key. It returns at most one `Finding` through the builder.

The raw rule, command, permission target path/directory, domain, tool name, MCP server name, and MCP
tool name never leave function-local parsing state. Returned records contain only allowlisted
classifications, positive source lines, and full domain-separated SHA-256 digests. The finding
builder receives the ordinary scanned source path only to populate `Finding.file`; it never places
that path in evidence or aggregate input, which instead uses the precomputed source-identity digest.

## Permission object and resource bounds

`permissions` must be a JSON object. The recognized 2.1.241 keys are:

- `allow`, `ask`, and `deny`: arrays of strings;
- `additionalDirectories`: an array of strings;
- `defaultMode`: a string;
- `disableBypassPermissionsMode`: the literal string `"disable"` when set;
- `disableAutoMode`: the literal string `"disable"` when set; and
- `skipDangerousModePermissionPrompt`: a boolean. Shared-project `true` is ignored by 2.1.241;
  project-local `true` is a recognized, applicable prompt-control declaration but does not enable
  bypass by itself.

Unknown keys inside `permissions` are completeness-affecting diagnostics. Unrelated top-level
settings keys are ignored because the settings schema contains many non-permission features.

An empty `permissions` object and recognized empty rule/directory arrays are valid, completed no-op
configurations. A section fails for having no valid analyzable content only when it contains one or
more supplied values but every supplied permission field/value is unknown or invalid.

One permission object may contain at most **2,048 structural items**. Count one item for every
top-level key in `permissions`, including an unknown key, plus one item for every raw list entry in
`allow`, `ask`, `deny`, and `additionalDirectories`. Count keys and entries before type validation,
diagnostic construction, or duplicate removal. A total of 2,048 is accepted; 2,049 is an atomic
permission-subanalysis `COMPONENT_LIMIT` failure with no grants, diagnostics, or aggregate digest.
An unknown key produces at most one diagnostic for that key; its nested value is never recursively
expanded into diagnostics. This structural budget bounds validation and diagnostic fan-out even for
a sub-megabyte object containing thousands of unique unknown siblings. File size and binary bounds
remain the existing `MAX_FILE_CHARS` and NUL checks in the shared settings parser.

Validated exact, bare, and MCP-server-wide restrictions are indexed rather than crossed against
every allow. Each distinct ask/deny tool-name glob is compiled once. Glob coverage uses a separate
deterministic per-document budget of **8,388,608 charged characters**: before matching one distinct
glob against one distinct candidate tool identifier, charge the sum of their character lengths. If
the next match would exceed the budget, permission analysis fails atomically with
`COMPONENT_LIMIT`, no grants, diagnostics, or aggregate. A 2,048-item document is accepted by the
structural budget but remains subject to this matcher-work bound. This prevents a valid sub-megabyte
cross product from becoming a CPU denial of service without imposing wall-clock-dependent behavior.

Exact duplicate list entries are collapsed after validation. Semantic classifications, counts,
maximum severity, blocking status, and diagnostic ordering are stable under list reordering and
duplication. Aggregate identity is intentionally physical: it includes the full content digest, so
reordering, adding a duplicate, changing whitespace, or making any other byte-level document
mutation changes the aggregate digest even when those semantic projections stay equal.

## Permission rule grammar

BH3 models a closed 2.1.241 routing snapshot derived from the
[official tools reference](https://code.claude.com/docs/en/tools-reference) and locked by pinned
startup probes. A rule is either a bare tool name or a tool plus one parenthesized specifier.

### Canonical tool routing

The parser routes a syntactically valid allow rule by exact tool name before classifying its
specifier:

| Route | Exact 2.1.241 names | Allow treatment |
|---|---|---|
| Shell execution | `Bash`, `PowerShell`, `Monitor` | Specialized command grammar; Monitor uses the same execution classes as Bash |
| Filesystem | `Read`, `Edit`, `Write`, `NotebookEdit`, `MultiEdit`, `Glob`, `Grep`, `LSP` | Specialized bare/path behavior below |
| Network | `WebFetch`, `WebSearch` | Specialized domain/search behavior below |
| MCP | `mcp__<server>` and `mcp__<server>__<tool>` families | Specialized MCP behavior below |
| External upload | `Artifact`, `ShareOnboardingGuide` | HIGH `external_content_upload`; only the documented bare form is valid |
| Skill invocation | `Skill` | MEDIUM `skill_invocation`; accepts the documented scoped form |
| Dynamic workflow | `Workflow` | HIGH `autonomous_workflow`; only the documented bare form is valid |
| Workspace boundary | `EnterWorktree` | HIGH `workspace_boundary_change`; only the documented bare form is valid |
| Approval transition | `ExitPlanMode` | MEDIUM `approval_gate_transition`; only the documented bare form is valid |
| Known non-grant | names enumerated below | Completeness-neutral `known_non_grant_tool` diagnostic; no grant |
| Unknown/dynamic allow | every other non-MCP name | Completeness-affecting `unknown_rule`; never silently safe |

The exact known-non-grant set is `Agent`, `AskUserQuestion`, `Cd`, `CronCreate`, `CronDelete`,
`CronList`, `EndConversation`, `EnterPlanMode`, `ExitWorktree`, `ListAgents`,
`ListMcpResourcesTool`, `PushNotification`, `ReadMcpResourceDirTool`, `ReadMcpResourceTool`,
`RemoteTrigger`,
`ReportFindings`, `ScheduleWakeup`, `SendMessage`, `SendUserFile`, `SendUserMessage`, `Task`,
`TaskCreate`, `TaskGet`, `TaskList`, `TaskOutput`, `TaskStop`, `TaskUpdate`, `TodoWrite`, `ToolSearch`,
and `WaitForMcpServers`. These names are known non-grants because an allow declaration does not
remove an approval boundary in the pinned snapshot. `SendUserMessage` is feature-gated but remains
a canonical 2.1.241 tool name. `Read`, `Glob`, `Grep`, and `LSP` are deliberately not in this set:
their specialized broad/external behavior is classified below.

Any syntactically valid bare or `Tool(param:value)` entry in `ask` or `deny` is a restrictive rule,
not a grant. It is completeness-neutral even when the tool name is unknown or dynamically supplied;
known candidates can still use it for conservative precedence. Thus normal declarations such as
`allow: ["Skill"]` are reportable, while `deny: ["Agent(Explore)"]` is completed restrictive policy,
not a failed or partial scan. Malformed delimiters and non-string entries remain invalid.

For allow, a scoped form is accepted only on a route whose grammar documents it. A syntactically
valid `Tool(param:value)` for a known bare-only or known-non-grant tool is a completeness-neutral
`unsupported_allow_specifier` diagnostic, not a grant and not an unknown tool. Truly unknown or
dynamic allow names remain completeness-affecting. Routing tests enumerate every known name and
fail if a name belongs to zero or multiple routes.

### Tool-wide and malformed rules

- A bare `Bash` and `Bash(*)` are equivalent tool-wide grants.
- The equivalent `PowerShell` and `PowerShell(*)` forms are tool-wide execution grants.
- In `allow`, an unanchored tool-name glob is not a grant. `*`, `B*`, and `mcp__*` are ignored known
  diagnostics, not BH3 grants.
- The ignored allow-glob diagnostic is completeness-neutral because 2.1.241 deterministically skips
  those unanchored forms with a startup warning; the scanner is not guessing whether they grant
  access. A future or otherwise unknown allow grammar remains completeness-affecting.
- In `ask` and `deny`, tool-name globs are valid precedence selectors. They are matched with bounded,
  case-sensitive glob semantics against the normalized tool identifier. Thus `deny: ["*"]`
  neutralizes every ordinary allow candidate, `ask: ["B*"]` neutralizes Bash candidates, and
  `deny: ["mcp__*"]` neutralizes matching MCP candidates. The exact global selector `*` in either
  `ask` or `deny` also neutralizes the document's bypass grant; narrower selectors do not.
- `mcp__<literal-server>` and `mcp__<literal-server>__*` are equivalent server-wide MCP rules and
  are HIGH.
- `mcp__<literal-server>__<literal-tool>` is a valid exact MCP-tool rule and is MEDIUM.
- `mcp__<literal-server>__<partial-tool-glob>`, such as `mcp__files__get_*`, is a valid scoped MCP
  rule and is MEDIUM. The server segment must remain literal.
- An unknown allow tool or malformed delimiter is not guessed. It is a completeness-affecting
  unknown-rule diagnostic. If no valid analyzable permission entry remains, the document fails.
- `UnknownTool(*)` in allow is not silently accepted as safe merely because a runtime may retain it
  for a future or dynamically registered tool. The same syntactically valid rule in ask/deny is a
  completeness-neutral restriction.

### File rules

The initial classifier recognizes path-qualified `Read` and `Edit` rules. A bare `Read` or `Edit`
is tool-wide. Under the pinned runtime, path-qualified `Write`, `NotebookEdit`, `Glob`, and
`MultiEdit` rules are not consulted for path authorization; those forms are known ignored
diagnostics. Their bare forms are explicit grants: bare `Write` is CRITICAL, bare `NotebookEdit` and
bare `MultiEdit` are HIGH broad-write grants, and bare `Glob` is MEDIUM filesystem-enumeration
capability. No path specifier is inferred for those bare forms.

Bare `Grep` and bare `LSP` are MEDIUM broad filesystem search/intelligence grants. A path-qualified
`Glob` is a known ignored diagnostic in 2.1.241 because external path approval is expressed with
`Read(...)`; the scanner does not reinterpret its specifier. The pinned runtime and official
permission contract do not establish equivalent ignored behavior for path-qualified `Grep` or `LSP`,
so those forms produce a completeness-affecting `runtime_uncertain_rule` diagnostic and no grant.
Bare MultiEdit remains recognized because the pinned 2.1.241 executable includes it in its canonical
edit/write tool sets, despite its omission from the evolving public tools table. A pinned probe locks
this behavior.

Path scope is classified without exposing the path:

- `//` and `//...` are anchored at the filesystem root.
- `~`, `~/`, and `~/...` are anchored at the user's home directory.
- a single-leading-slash form is relative to the runtime's starting/project root associated with
  the settings source; it is not the filesystem-root `//` form.
- `./...` and other relative forms are project-relative.
- one or more leading `../` segments form a valid external path; an interior parent segment after a
  literal segment, NUL, drive-qualified, UNC, malformed home, and ambiguous separator forms are
  unknown rules and affect completeness.

Whole-root/home coverage is closed: `//`, `//**`, `//**/*`, `~`, `~/`, `~/**`, and `~/**/*` are
CRITICAL for Read/Edit. A root- or home-anchored specific path is not automatically CRITICAL; it is
classified as sensitive, broad external, or scoped external by the rules below. A single-leading
slash remains project-relative, so `Edit(/tmp/**)` means the project's `tmp` subtree and is MEDIUM,
not an absolute `/tmp` grant.

For Edit, a **broad external write** is HIGH only when the normalized target is outside the project
and its final path pattern is an all-entry wildcard (`*` or `**`) or it ends in `/**` or `/**/*`.
Examples: `Edit(../shared/**)` and `Edit(//tmp/**)` are HIGH. A specific external file or bounded
filename glob is MEDIUM: `Edit(../shared/config.json)` and `Edit(../shared/report-*.md)` are MEDIUM.
Project-relative writes remain MEDIUM even for a project subtree:
`Edit(./generated/**)` and `Edit(/tmp/**)` are MEDIUM. Non-sensitive external Read is MEDIUM;
narrow project Read is silent; sensitive Read/Edit is HIGH.

The sensitive-path classifier is a closed, tested set covering agent configuration, shell history,
SSH/private keys, cloud credentials, Kubernetes, Docker, package-manager credentials, Git
credentials, `.env`/secret/token material, and equivalent home-scoped credential stores. Closed
markers use ASCII case normalization and match complete path segments or ASCII-alphanumeric token
boundaries inside a filename; they do not match ordinary continuations such as `tokenizer.py`,
`tokenization.md`, or `secretariat.md`. Evidence reports `sensitive_path`, never the matched path.

### Additional-directory paths

`additionalDirectories` uses add-directory filesystem-path semantics, not permission-rule pattern
anchors. In particular, `/tmp` is an absolute external directory and is not the project-relative
`Edit(/tmp/**)` spelling. Classification is lexical and relative to the project/runtime starting
directory; it never resolves a symlink or exposes the path.

Pinned 2.1.241 first applies ECMAScript `String.trim()` to the entire entry. The exact trimmed set is
U+0009-U+000D, U+0020, U+00A0, U+1680, U+2000-U+200A, U+2028-U+2029, U+202F, U+205F, U+3000, and
U+FEFF. U+001C-U+001F, U+0085, and U+180E are not trimmed; Python's unrestricted `str.strip()` is
therefore forbidden. Empty or whitespace-only input resolves to the project/base directory: it emits
no grant, one completeness-neutral `directory_existence_static_unknown`, and canonical identity `.`.
Only exact `~` and `~/...` use home semantics. `~user`, `$HOME`, `${HOME}`, `$Env:USERPROFILE`,
`%USERPROFILE%`, `!TEMP!`, `$Recycle.Bin`, and `100%done` are literal paths because this runtime does
not interpolate environment variables. Canonical identity and deduplication use the trimmed,
lexically normalized value.

The general classification is:

| Shape | Treatment |
|---|---|
| `.`, `./`, a plain relative child, `./child`, or a lexically normalized relative path that remains inside the project | Already within the project boundary; silent |
| A relative path whose lexical normalization retains leading `../` segments | External directory, MEDIUM unless sensitive |
| `/absolute` other than the whole root | External directory, MEDIUM unless sensitive |
| `/`, `//`, or equivalent separator-only filesystem root | Whole-root CRITICAL |
| `~` or `~/` | Whole-home CRITICAL |
| `~/child` | External/home directory, MEDIUM unless sensitive |
| Any sensitive external/home/absolute directory such as `~/.ssh` | HIGH `sensitive_additional_directory` |
| Windows ASCII drive root such as `C:\\` or `C:/` | Conditional whole-root CRITICAL plus completeness-affecting `platform_dependent_path` |
| Windows separator-only backslash form such as `\` or `\\` | Conditional current-drive-root CRITICAL plus completeness-affecting `platform_dependent_path` |
| Lexically sensitive Windows absolute path such as `C:\\Users\\x\\.ssh` | Conditional sensitive-directory HIGH plus completeness-affecting `platform_dependent_path` |
| Valid complete ordinary Windows UNC using `\\server\\share` or `//server/share` | Conditional external MEDIUM unless sensitive, plus completeness-affecting `platform_dependent_path` |
| Valid one-component UNC-like form such as `\\server` or `//server` | Conditional drive-root-relative external MEDIUM unless sensitive, plus completeness-affecting `platform_dependent_path` |
| Extended ASCII-drive root `\\?\\C:\\` or `//?/C:/` | Conditional whole-root CRITICAL plus completeness-affecting `platform_dependent_path` |
| Extended ASCII-drive path `\\?\\C:\\docs` or `//?/C:/docs` | Conditional external MEDIUM unless sensitive, plus completeness-affecting `platform_dependent_path` |
| Extended UNC `\\?\\UNC\\server\\share` or `//?/UnC/server/share` | Conditional external MEDIUM unless sensitive, plus completeness-affecting `platform_dependent_path`; only the `UNC` namespace token is ASCII-case-insensitive |
| Bare device namespace `\\.\\` or `//./` | Conditional current-drive-root CRITICAL plus completeness-affecting `platform_dependent_path` |
| Other Windows absolute ASCII-drive form | Conditional external MEDIUM plus completeness-affecting `platform_dependent_path` |
| Other reserved/device namespace such as `\\.\\PIPE`, `\\?\\Volume{...}`, `\\?\\GLOBALROOT`, bare/incomplete `\\?\\...`, or `\\??\\...` | Completeness-affecting `platform_dependent_path`; no grant or existence diagnostic is guessed |
| Non-ASCII colon prefix such as `é:/docs`, `中:/docs`, or `Ｃ:/docs` | Ordinary project-relative path after lexical normalization; no grant plus completeness-neutral existence diagnostic |
| Drive-relative, malformed UNC, or otherwise platform-ambiguous form with no provable resolved scope | Completeness-affecting `platform_dependent_path`; no grant is guessed |
| NUL-bearing value | Completeness-affecting `invalid_path` |

The pure analyzer collapses lexical `.` and `..` segments but does not call `stat`, resolve a
symlink, or pick a target operating system. Thus `child/../docs` is project-local while
`child/../../docs` remains external. For ordinary ASCII-drive and ordinary UNC tails only, it also
models Win32 component normalization: exact `.`/`..` are applied; one terminal dot is removed from
an earlier component only when the terminal-dot run has length one; and the final component of an
input without a trailing separator loses trailing U+0020, then applies `.`/`..`, otherwise loses
trailing U+0020 and periods and drops an empty result. Traversal clamps at the drive or UNC-share
root. Extended `\\?\\` paths do not receive this trimming. Consequently ordinary `.ssh.` and
`.ssh ` normalize to sensitive `.ssh`, `C:/foo/.. ` becomes a CRITICAL drive root, `C:/...` becomes
a CRITICAL root, and `C:/.../` retains its literal component and remains MEDIUM.

Ordinary and extended UNC dispatch happens after reserved `//?/`, `//./`, `/??/`, and malformed
two-leading `//??/` namespaces are recognized; both NT-namespace spellings produce platform-only,
no-existence results. Exactly two leading separators are required for ordinary UNC. A server is a nonempty Unicode scalar
string excluding surrogates, U+0000-U+001F, U+007F, Unicode whitespace, `\\ / : * ? " < > | ,`, and
exact `.`/`..`. A share is 1-80 UTF-16 code units and excludes surrogates, U+0000-U+001F,
`" \\ / [ ] : | < > + = ; , * ?`, and exact `.`/`..`. `$`, `%`, periods, spaces where permitted,
and other Unicode remain literal, so `C$`, `ADMIN$`, and `IPC$` are valid shares. Server/share
anchors are never trimmed or removed by tail traversal. A malformed or reserved anchor emits only
`platform_dependent_path`, with no guessed grant or existence diagnostic.

Sensitive credential-store pairs such as `.config/gcloud`, `.config/gh`, and `.config/glab` match as
adjacent normalized component subsequences anywhere after the UNC server component (the share may
participate), but not
`.configuration/gcloud`, `.config/gclouding`, or `.config/x/gh`. Existence and directory type remain
`static_unknown`. Each otherwise valid or conditionally external entry gets at most one
completeness-neutral `directory_existence_static_unknown`; runtime absence does not turn a lexical
grant into a safe result. Reserved/device and malformed UNC forms intentionally receive no existence
diagnostic.

### Network, execution, and MCP rules

- Bare `WebFetch` and `WebFetch(domain:*)` are equivalent all-domain HIGH grants. In this snapshot,
  `WebFetch(*)` is a completeness-neutral `unsupported_allow_specifier` diagnostic, not an
  all-domain equivalent. Literal and valid wildcard domain scopes such as
  `WebFetch(domain:docs.example)`, `WebFetch(domain:*.example.com)`, and
  `WebFetch(domain:example.*)` are MEDIUM.
- A domain pattern is 1-253 ASCII characters after removing one terminal root dot. Dot-separated
  labels are 1-63 characters from ASCII letters, digits, hyphen, and `*`; a label cannot begin or
  end with hyphen. `*` is the sole all-domain pattern. Schemes, user info, ports, paths, whitespace,
  empty labels, `?`, and non-ASCII input are invalid. Punycode is treated as ordinary ASCII; the
  scanner performs no Unicode conversion.
- Bare `WebSearch` is MEDIUM because it permits network search but not an arbitrary caller-selected
  fetch destination.
- Bare or scoped `Bash`/`PowerShell`/`Monitor` command scope is classified according to the
  matrix below. Monitor inherits Bash command-pattern normalization and severity because it runs a
  shell command; its bare form also admits its WebSocket source and is CRITICAL.
- For precedence, a scoped Monitor command uses the Bash permission family: an equivalent scoped or
  bare Bash ask/deny rule covers the command portion. A bare Monitor allow remains distinct because
  it also admits the Monitor WebSocket source; a Bash restriction alone does not neutralize that
  separate capability.
- Every valid scoped `Bash`, `PowerShell`, or `Monitor` allow is MEDIUM `scoped_execution`,
  including `Bash(npx prettier:*)` and its 2.1.241 whitespace spelling. `npx` can fetch packages and
  load executable configuration or plugins, so there is no execution safe-list or substring-based
  "formatter" exception.
- A bare literal MCP server or literal-server wildcard is HIGH; a literal exact or partial-glob MCP
  tool is MEDIUM.

## Precedence and mitigation

Within the same parsed document, runtime rule precedence is `deny`, then `ask`, then `allow`.
SkillSpector suppresses an allow candidate only when coverage is statically proven:

1. an identical normalized `deny` or `ask` rule covers the candidate; or
2. a bare/tool-wide `deny` or `ask` rule covers every use admitted by a narrower rule for the same
   tool; or
3. a valid ask/deny tool-name glob matches the candidate's normalized tool identifier, including
   `*`, `B*`, and `mcp__*`.

The implementation does not attempt path/specifier glob-subsumption proofs. An overlapping but
non-identical path/specifier pattern is not credited as a mitigation. The
[official permission-mode contract](https://code.claude.com/docs/en/permission-modes) and pinned
2.1.241 evaluate deny rules in every mode and keep explicit ask rules interactive even in bypass
mode. Therefore an exact
global `deny: ["*"]` or `ask: ["*"]` removes the silent whole-tool capability and neutralizes the
document's `defaultMode: "bypassPermissions"` grant. A narrower deny or ask still leaves other tool
calls silently approved, so it does not mitigate the document-level bypass grant. This special case
is locked to the exact global selector and does not infer broader glob or specifier coverage.

Before exact mitigation comparison, normalize only proven 2.1.241 runtime-equivalent identities:

- bare `Bash` equals `Bash(*)`;
- the legacy Bash prefix separator and whitespace spelling are equivalent, so `Bash(ls:*)` equals
  `Bash(ls *)`;
- a scoped `Monitor(command)` allow compares against the equivalent Bash command identity for
  ask/deny precedence, while bare Monitor keeps its distinct WebSocket-bearing identity;
- PowerShell tool/command matching is ASCII case-insensitive; non-ASCII code points remain exact
  because the pinned runtime evidence does not establish Unicode full case folding; and
- WebFetch domain patterns are ASCII case-insensitive and wildcard-aware, and one terminal DNS root
  dot is removed, so
  `WebFetch(domain:EXAMPLE.com.)` equals `WebFetch(domain:example.com)`.

The same normalization applies to allow, ask, and deny before comparison. An identical normalized
WebFetch wildcard pattern can mitigate its matching allow, but the scanner does not prove
subsumption between distinct domain patterns. Normalization does not authorize Unicode hostname or
PowerShell full case folding, path case folding, MCP case folding, command parsing beyond the proven
Bash/PowerShell rules, or path/specifier wildcard-subsumption guesses.

`permissions.disableBypassPermissionsMode: "disable"` in the same physical document does
neutralize that document's `defaultMode: "bypassPermissions"`. Both remain part of aggregate
identity, no blocking critical is emitted for the neutralized mode, and a safe diagnostic records
the recognized control. A higher-precedence external disable may also mitigate at runtime, but the
scanner cannot prove it and does not remove the artifact finding.

The same no-grant result applies when that physical document combines bypass with the exact global
`ask: ["*"]` or `deny: ["*"]` selector. A safe `bypass_global_restriction` diagnostic records the
recognized same-document control. The aggregate retains a domain-hashed semantic identity for the
source list and selector, without exposing either raw value.

`defaultMode: "dontAsk"` is silent because it auto-denies actions that are not pre-approved. It does
not suppress a reportable `allow`: those pre-approved actions still execute in `dontAsk` mode.

`skipDangerousModePermissionPrompt: true` is source-sensitive. In shared project settings it is a
known ignored diagnostic. In local settings it is a recognized applicable diagnostic and is included
in physical/semantic identity. Alone it emits no BH3 because it neither enables nor selects bypass.
Alongside local `defaultMode: "bypassPermissions"`, the document remains the same blocking CRITICAL
BH3, and the applicable prompt-control diagnostic participates in the aggregate digest. A literal
`false` is a recognized no-op. This design does not infer an externally enabled bypass mode.

## Permission mode semantics

| `defaultMode` value | BH3 treatment in project/local settings | Reason |
|---|---|---|
| `bypassPermissions` | CRITICAL, blocking unless disabled or globally ask/deny restricted in the same document | Skips ordinary prompts and permission checks, subject to interface and external policy |
| `acceptEdits` | MEDIUM | Auto-accepts edits and a bounded set of filesystem commands |
| `auto` | No finding; known ignored diagnostic | Project/local values are ignored in 2.1.241 |
| `dontAsk` | No finding | Restrictive by itself; report any separate allow grants |
| `default` | No finding | Ordinary approval behavior |
| `manual` | No finding; legacy alias diagnostic | Treated as the default/manual approval posture |
| `plan` | No finding | Read-oriented planning posture |
| `delegate` or any other value | Completeness-affecting diagnostic | Not a valid 2.1.241 `permissions.defaultMode`; future/invalid modes are not guessed |

The table classifies only the artifact declaration. CLI flags, managed settings, user settings,
host-managed settings, account eligibility, UI opt-ins, and platform restrictions can override or
disable a mode.

## Trust, provenance, and interface contract

Each returned grant carries these safe classifications:

| Source/capability | `activation_requirement` | `interface_applicability` | `tracking_status` | Static interpretation |
|---|---|---|---|---|
| Shared `allow` or `additionalDirectories` | `workspace_trust` | `claude_code_settings_consumers` | `not_applicable` | Inactive before trust; capability after trust |
| Local `allow` or `additionalDirectories` | `local_provenance_and_session_policy` | `claude_code_settings_consumers` | `unknown` | Untracked can apply before trust; tracked is trust-gated; scanner cannot decide |
| Shared permission mode | `interface_and_external_policy` | `permission_mode_interface_dependent` | `not_applicable` | Mode support and higher-precedence policy remain external |
| Local permission mode | `interface_and_external_policy` | `permission_mode_interface_dependent` | `unknown` | Mode support, local provenance, and higher-precedence policy remain external |
| Shared `deny` or `ask` | `none_for_restriction` | `claude_code_settings_consumers` | `not_applicable` | Restriction can apply before shared trust; not a BH3 grant |
| Local `deny` or `ask` | `none_for_restriction` | `claude_code_settings_consumers` | `unknown` | Restriction participates in local precedence; not a BH3 grant |

The complete allowlists are `workspace_trust`, `local_provenance_and_session_policy`, and
`interface_and_external_policy` for emitted-grant activation; `claude_code_settings_consumers` and
`permission_mode_interface_dependent` for interface applicability; and `not_applicable` and
`unknown` for tracking. `none_for_restriction` is internal to ask/deny precedence and never appears
in BH3 evidence unless a future finding class explicitly reports restrictions. Document evidence
sorts and comma-joins distinct emitted-grant tokens when more than one class appears.

Cloud sessions load committed shared project settings but do not load a user's local settings file.
VS Code, Desktop, remote-control, web, and SDK hosts expose different mode sets and may inject policy.
Consequently, finding evidence uses `runtime_status: "external_unknown"`. It does not use
`active`, `enabled`, `installed`, `effective`, or `shipped` as a runtime fact.

## Severity and blocking matrix

The document severity is the maximum effective, non-mitigated grant severity after known ignored
forms and precedence are applied.

| Severity | Effective grant classes |
|---|---|
| CRITICAL | `bypassPermissions`; tool-wide Bash/PowerShell/Monitor; bare or filesystem-root/home-wide `Read` or `Edit`; bare tool-wide `Write`; filesystem-root/home `additionalDirectories` |
| HIGH | Sensitive-path read/edit/additional directory; bare `NotebookEdit`; bare `MultiEdit`; broad external write; all-domain fetch; broad literal MCP-server capability; external upload/publish; Workflow; EnterWorktree |
| MEDIUM | Scoped execution; scoped network; scoped write/edit; bare `Glob`/`Grep`/`LSP`; exact/partial MCP tool; external additional directory; `acceptEdits`; Skill; ExitPlanMode |
| Silent | Narrow in-project read; restrictive ask/deny; default/manual/plan/dontAsk; project/local auto |

`blocking_critical` is `True` if and only if at least one effective CRITICAL grant remains after
same-document mitigation. HIGH and MEDIUM BH3 findings do not set it. The boolean is independent of
the fact that activation remains conditional on trust, provenance, interface, or policy.

### Closed grant-kind vocabulary

Every effective grant maps to exactly one of these allowlisted `grant_kind` tokens; no tool, path,
command, domain, server, or mode value is copied into the token:

| Token | Reportable class |
|---|---|
| `permission_mode_bypass` | Unmitigated `bypassPermissions` mode |
| `permission_mode_accept_edits` | `acceptEdits` mode |
| `tool_wide_execution` | Bare or `(*)` Bash/PowerShell/Monitor |
| `scoped_execution` | Reportable scoped Bash/PowerShell/Monitor command |
| `tool_wide_read` | Bare Read |
| `root_or_home_wide_read` | Closed whole-root/home Read pattern |
| `sensitive_read` | Sensitive-path Read |
| `external_read` | Non-sensitive external Read |
| `tool_wide_edit` | Bare Edit |
| `root_or_home_wide_edit` | Closed whole-root/home Edit pattern |
| `sensitive_edit` | Sensitive-path Edit |
| `broad_external_edit` | External all-entry/subtree Edit |
| `scoped_edit` | Project or bounded external Edit |
| `tool_wide_write` | Bare Write |
| `broad_notebook_edit` | Bare NotebookEdit |
| `broad_multi_edit` | Bare MultiEdit |
| `filesystem_enumeration` | Bare Glob |
| `filesystem_search` | Bare Grep |
| `code_intelligence` | Bare LSP |
| `all_domain_fetch` | Bare WebFetch or `WebFetch(domain:*)` |
| `scoped_domain_fetch` | Literal or valid wildcard domain-scoped WebFetch |
| `network_search` | Bare WebSearch |
| `mcp_server_wide` | Bare literal MCP server or literal-server `__*` |
| `mcp_exact_tool` | Literal MCP server and literal tool |
| `mcp_partial_tool` | Literal MCP server and partial-tool glob |
| `root_or_home_additional_directory` | Root/home additional-directory grant |
| `sensitive_additional_directory` | Sensitive external/home additional-directory grant |
| `external_additional_directory` | External additional-directory grant |
| `external_content_upload` | Artifact or onboarding-guide upload/publish grant |
| `skill_invocation` | Skill invocation grant |
| `autonomous_workflow` | Workflow dynamic-orchestration grant |
| `workspace_boundary_change` | EnterWorktree external-cwd/write-boundary grant |
| `approval_gate_transition` | ExitPlanMode approval transition |

After precedence and exact-rule deduplication, `grant_count` counts retained grant records, not
distinct tokens. `grant_kinds` is the lexicographically sorted, comma-joined set of their tokens, so
two distinct scoped commands count twice but contribute `scoped_execution` once. Severity counts use
retained records. Silent controls and diagnostics never enter either grant projection.

## One BH3 finding per document

If at least one reportable grant remains, the helper creates exactly one deterministic BH3 finding:

- `rule_id`: `BH3`
- `category`: `Bundled Execution Surface`
- `pattern`: `Bundled Permission Grant`
- `severity`: maximum grant severity
- `confidence`: `1.0`
- `file`: physical settings cache path
- `start_line`: earliest reportable grant `source_line`, falling back to the enclosing permissions
  line and then line 1 only when structural line recovery is unavailable
- `matched_text` and `finding`: the full aggregate digest
- tags: `bundled-execution-surface`, `structural`

The message reports only the grant count and maximum class. It never contains a rule, path, command,
domain, MCP identifier, or mode value beyond the generic finding classification.

### Evidence schema

BH3 evidence is a flat mapping named `skillspector.bundled_permission.v1`. Its exact allowlist is:

```text
schema
claude_semantics_snapshot
source_kind
declaration_status
artifact_effect_status
activation_requirement
interface_applicability
tracking_status
runtime_status
grant_count
critical_grant_count
high_grant_count
medium_grant_count
grant_kinds
diagnostic_count
diagnostic_kinds
max_severity
blocking_critical
aggregate_digest
```

Values are only `str`, `int`, or `bool`. Multi-value classifications are sorted comma-separated
tokens from closed enums. The fixed values are:

- `schema`: `skillspector.bundled_permission.v1`
- `claude_semantics_snapshot`: `2.1.241`
- `declaration_status`: `declared`
- `artifact_effect_status`: `conditional`
- `runtime_status`: `external_unknown`

Diagnostic kinds are also closed and contain no user value. The initial set is
`auto_ignored`, `legacy_manual`, `bypass_disabled`,
`bypass_global_restriction`, `auto_disabled`,
`skip_dangerous_prompt_ignored`, `local_skip_dangerous_prompt_declared`, `ignored_allow_rule_glob`,
`ignored_path_qualifier`, `runtime_uncertain_rule`, `unsupported_allow_specifier`,
`known_non_grant_tool`, `restrictive_rule`, `mitigated_allow`, `platform_dependent_path`,
`directory_existence_static_unknown`, `unknown_permission_key`, `unknown_mode`, `unknown_rule`,
`wrong_type`, and `invalid_path`.

No list, object, null, raw rule, raw settings fragment, raw path, short digest, or nested evidence is
allowed. `aggregate_digest` is a full `sha256:` value and equals `matched_text`/`finding`.

Each retained grant/diagnostic digest is SHA-256 over a domain-separated canonical JSON object of
its safe classifications plus a separately domain-hashed normalized rule/key identity; `source_line`
is excluded. The raw normalized identity is discarded immediately after hashing. The aggregate is
SHA-256 over the byte prefix `b"skillspector.bundled_permission.aggregate.v1\0"` plus UTF-8 canonical JSON
(`sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=True`) with exactly these internal keys:
`schema`, `claude_semantics_snapshot`, `source_kind`, `source_identity_digest`, `content_digest`,
`grant_digests`, `diagnostic_digests`, `mitigated_allow_count`, `max_severity`, and
`blocking_critical`. Digest arrays are sorted full `sha256:` strings; the count is an integer and the
blocking value is a literal boolean.

Any byte-level physical mutation or source-identity change invalidates an exact baseline. Semantic
classification/count/severity projections remain deterministic under reordering/duplicates, but
the aggregate intentionally does not.

## Outcomes and ledger semantics

JSON integrity errors are atomic because hooks and permissions cannot safely share a root mapping:

| Condition | Outcome | Findings |
|---|---|---|
| Missing cache, NUL/binary, size overflow, malformed JSON, duplicate key, non-object root | `FAILED` | None from this physical settings document |
| Permission structural-item count above 2,048, with no independently valid hook section | `FAILED` / `COMPONENT_LIMIT` | No BH3 |
| Valid hook section plus permission structural-item count above 2,048 | `PARTIAL` / `COMPONENT_LIMIT` | Preserve BH1/BH2; no BH3 |
| Valid hook and valid permission sections | `COMPLETED` | Combined BH1/BH2/BH3 IDs |
| Valid analyzable permission content with no reportable grant | `COMPLETED` | No BH3; hook findings remain |
| Valid grant plus wrong-type/unknown permission sibling | `PARTIAL` / `INVALID_CONFIGURATION` | Preserve BH3 and any hook findings |
| Valid hook plus invalid permission section | `PARTIAL` / `INVALID_CONFIGURATION` | Preserve BH1/BH2 |
| Valid permission section plus invalid manifest-referenced hook role | `PARTIAL` / `INVALID_CONFIGURATION` | Preserve BH3 |
| Non-empty permission section whose supplied fields are all unknown/invalid, with no independently valid hook section | `FAILED` / `INVALID_CONFIGURATION` | No settings findings |
| Permission section has no valid analyzable entry but the hook section is valid | `PARTIAL` / `INVALID_CONFIGURATION` | Preserve BH1/BH2 |
| Empty permission object or recognized empty arrays | `COMPLETED` | No BH3; hook findings remain |
| Root settings with neither hooks nor permissions | Not applicable | No row |

A PARTIAL producer row may own emitted findings and makes analysis completeness partial, but it is
not an execution failure under the existing ledger contract. Therefore an unsuppressed blocking BH3
still produces score at least 51 and default CLI exit 1. `--fail-on-incomplete` also exits 1 for a
non-blocking PARTIAL scan. An atomic FAILED row sets `execution_successful=false` and exits 2; that
takes precedence over risk scoring.

## Meta-analysis, defaults, scoring, suppression, and output

BH3 is a deterministic structural rule:

- Add BH3 to `_STRUCTURAL_RULE_IDS` so neither an LLM response nor a no-LLM confidence threshold can
  remove it.
- Add BH3 to explanation, remediation, category, and pattern-name defaults.
- Do not add another analyzer to `ANALYZER_NODE_IDS` or `ANALYZER_NODES`.
- Keep ordinary severity scoring. In addition, return a 51 floor for BH3 only when
  `finding.evidence.get("blocking_critical") is True`. A truthy string, integer, missing evidence,
  or CRITICAL label alone does not activate the floor.
- Baseline suppression runs before scoring, so a suppressed BH3 contributes neither points nor the
  floor. It remains available only through the existing suppressed-finding surfaces.
- Terminal, JSON, Markdown, and SARIF use the existing generic finding renderers. Tests prove the
  schema allowlist and canary non-disclosure rather than adding BH3-specific rendering branches.

## Test strategy

Implementation follows red-green-refactor in the accompanying plan.

### Pure classifier matrix

- every permission mode, including auto ignored and bypass disabled;
- shared versus local activation/provenance labels;
- bare, `(*)`, scoped, sensitive, root, home, project-relative, network, and MCP grants;
- exact and bare ask/deny precedence without speculative glob subsumption;
- `dontAsk` plus allow;
- known ignored wildcard and path-qualified unsupported forms;
- malformed/unknown rules, keys, modes, and wrong types;
- duplicate list values and reorder-stable semantic projections while physical aggregate identity changes;
- exact 2,048 structural-item boundary, including permission keys and raw list entries, and
  2,049-item atomic permission-subanalysis failure;
- exact 8,388,608-character precedence matcher-work boundary, compiled-glob reuse, and an
  adversarial distinct-rule cross product that fails atomically without a wall-clock oracle;
- allow-only wildcard diagnostics versus ask/deny wildcard mitigation, including `deny: ["*"]`;
- bare/server-wide, exact-tool, and partial-tool-glob MCP forms;
- proven Bash, PowerShell, and WebFetch equivalence normalization with conservative negative cases;
- shared-ignored versus local-applicable `skipDangerousModePermissionPrompt` behavior;
- exact broad-external versus bounded-scoped Edit thresholds;
- physical content and hashed source-identity aggregate mutation;
- only flat safe scalar evidence and no supplied canary leakage.

### Surface and ledger matrix

- direct shared/local roots, root archive, nested archive, and all exclusion paths;
- permissions-only, hooks-only, and mixed settings;
- valid hook plus invalid permissions and valid permissions plus invalid referenced-hook role;
- settings discovered before a later manifest reference without `handled_paths` suppression;
- one path-level terminal producer row and combined finding IDs;
- direct directory, ZIP, and nested ZIP graph scans;
- no registry change and no duplicate analyzer execution.

### Report and CLI matrix

- structural retention with LLM enabled, provider rejection, and `--no-llm`;
- HIGH/MEDIUM ordinary scoring, blocking boolean floor, non-boolean negative controls, and suppressed
  floor removal;
- default exit 0/1/2 and `--fail-on-incomplete` interactions;
- terminal, JSON, Markdown, and SARIF evidence and redaction;
- baseline generation, unchanged rescan, permission mutation, mitigation mutation, and ZIP cache
  lookup.

## Deepest practical runtime and corpus verification

The final implementation must be tested beyond shaped unit artifacts.

1. Record local `claude --version` and use an isolated pinned
   `npx -y @anthropic-ai/claude-code@2.1.241` runner for version-sensitive startup checks.
2. In disposable repositories with no real secrets or external endpoints, verify shared allow and
   additional-directory rejection before trust, untracked-local acceptance, tracked-local trust
   gating, auto rejection, bypass recognition, same-document bypass disable behavior, exact global
   ask/deny restriction, and narrower ask/deny controls that leave bypass reportable.
3. Exercise multiple modes and both settings sources. Capture startup/config diagnostics and never
   infer a successful tool authorization from file parsing alone.
4. If login/model/API access permits, issue benign local-only tool calls to distinguish recognized
   configuration from actual authorization. If it does not, label the result startup/config E2E and
   state that tool-call authorization was not executed.
5. Record IDE/Desktop/cloud/Agent SDK cases as untested unless those real interfaces are available;
   CLI behavior does not prove parity for them.
6. Re-scan the pinned NVIDIA skills catalog and available third-party hook/settings bundles. Report
   exact checkout paths, revisions, file counts, BH1/BH2/BH3 counts, and every exception. If a corpus
   is unavailable, disclose the missing corpus rather than carrying forward issue #399's historical
   counts as a new result.
7. Run direct directory, ZIP, nested ZIP, all report formats, baseline mutation, full non-provider
   tests, non-live integrations, lint, format, type checking, package build, and Docker smoke when a
   daemon is available.

## Acceptance criteria

BH3 is ready to remain on the dependent draft PR only when all of the following are true:

1. Issue #399 Case B produces one BH3 that describes the grants without exposing their raw values.
2. Case C produces BH1/BH2/BH3 from the mixed artifact; the settings file has one producer owner.
3. `Bash(*)`, bare Bash, root/home read/edit, root/home additional directories, and unmitigated
   bypass mode set `blocking_critical: true` and score at least 51 when unsuppressed.
4. Project/local auto, dontAsk alone, restrictive rules, and narrow project read do not emit BH3;
   scoped execution such as the exact prettier rule emits MEDIUM BH3.
5. Shared trust, local provenance, and interface uncertainty are explicit; no report claims the
   permission was active, installed, or used.
6. Malformed siblings preserve independently valid analysis as PARTIAL. Atomic shared-JSON failures
   emit no settings findings and exit 2; a permission structural-limit failure is FAILED by itself
   but is PARTIAL and preserves an independently valid hook section in the same document.
7. Unknown runtime grammar cannot score SAFE through omission: it produces a visible partial/failed
   analysis outcome.
8. An unsuppressed blocking BH3 floors the score; a suppressed or non-boolean-marked BH3 does not.
9. All evidence remains flat, scalar, allowlisted, deterministic, and free of raw permission data.
10. The final PR reports exact unit, integration, runtime, corpus, build, and unavailable-E2E
    boundaries, and stays in draft for review.

## Design review resolution

The approved design and follow-up architecture audit required the following changes, all captured in
this specification:

- settings roots and archive namespaces are exact and exclusions are explicit;
- one JSON parse feeds independent hook and permission analysis;
- the current analyzer owns one physical settings row and the registry stays unchanged;
- shared trust, local tracking provenance, and interface policy are separate facts;
- project/local auto is an ignored diagnostic, not a finding;
- mode, rule, path, precedence, duplicate, and cardinality behavior is closed and testable;
- partial versus atomic failure behavior, score floor, baseline, and CLI exits are explicit; and
- runtime/corpus claims require fresh evidence with gaps reported.

With the user's approval, this specification is decision-complete for production implementation.
