# Analysis Resource Bounds

SkillSpector applies deterministic resource ceilings to untrusted skill bundles. A ceiling is a
safety boundary, not an allowlist and not evidence that the portion examined was clean. When a
relevant ceiling is reached, the scanner records the limitation and reports partial analysis.

Values below are implementation defaults. MiB means 1,048,576 bytes.

## Bundle discovery and materialization

| Resource | Ceiling | Scope |
|---|---:|---|
| Discovered entries | 10,000 | One skill bundle |
| Entries materialized from one directory | 10,000 | One directory |
| Filesystem traversal depth | 64 | One skill bundle |
| Discovery time | 30 seconds | One skill bundle |
| Canonical cached source bytes | 64 MiB | Ordinary files and expanded nested members combined |
| Cached bytes from one filesystem artifact | 16 MiB | One artifact |
| End-to-end workflow time | 600 seconds | One graph execution |
| Cache and context-processing time | 60 seconds | One skill bundle, within the workflow deadline |

The 64 MiB ceiling accounts for the canonical raw bytes retained for analysis. Local decoded text
and LLM-safe text are derived only from those bounded bytes; they do not authorize another source
read allowance. Exact raw bytes from readable nested members consume the remaining portion of the
same 64 MiB budget. The byte ceiling is an input-accounting limit, not a promise that Python object
overhead or decoded Unicode storage occupies exactly the same amount of memory.

Files larger than 16 MiB receive a bounded local projection and a `partial` inventory disposition.
They are not sent to an external model. Lexical static analysis uses 256,000-character windows with
an 8,192-character overlap. Unicode-derived security views are re-sliced to the same ceiling before
pattern modules receive them. Whole-file Python AST analysis is limited to 1,000,000 characters, and
the shared parsed-AST cache retains at most 8,000,000 source characters per scan.

## Nested containers

Archive inspection shares the bundle's remaining artifact, byte, and processing-time allowances.
Its additional ceilings are:

| Resource | Ceiling | Scope |
|---|---:|---|
| Recursion depth | 3 | One outer-to-inner provenance chain |
| Members | 1,000 | All outer and nested containers combined |
| Expanded bytes | 25 MiB | All outer and nested containers combined |
| Central-directory bytes | 4 MiB | One container, before ZIP metadata allocation |
| Materialized member | 1,000,000 bytes | One member |
| Compression ratio | 100:1 | One member |
| Archive-inspection time | 5 seconds | All outer and nested containers combined |

The effective member and expanded-byte ceilings are the smaller of these values and the enclosing
bundle's remaining allowances. EOCD and ZIP64 metadata, declared counts, central-directory bytes,
and actual central headers are checked before ZIP metadata objects are created. See
[Nested Artifact Inspection](NESTED_ARTIFACT_INSPECTION.md) for provenance and containment rules.

## Manifest YAML

Only a bounded primary `SKILL.md` frontmatter prefix is eligible for YAML parsing:

| Resource | Ceiling |
|---|---:|
| Frontmatter bytes | 256 KiB |
| YAML nodes | 10,000 |
| YAML nesting depth | 64 |
| Projected manifest records | 1,024 |
| Projected manifest characters | 256 KiB |
| Manifest parse time | 1 second |

The closing frontmatter delimiter must be present inside the bounded prefix. Node, depth, projected
output, and time limits are checked before the bounded document is accepted. YAML aliases are
charged for each projected occurrence, so a compact alias graph cannot amplify the returned
manifest past these limits. A malformed or incomplete claimed frontmatter leaves the manifest
empty, marks the primary artifact `partial`, and records an allowlisted parse error or limit reason.

## Intra-bundle references

Reference extraction from the primary instructions is independently bounded:

| Resource | Ceiling |
|---|---:|
| Source bytes examined | 1,000,000 |
| Raw candidates considered | 4,096 |
| Accepted references | 256 |
| Output records | 1,024 |
| Extraction time | 2 seconds |

Truncated extraction and missing or ambiguous local references are explicit partial-coverage
conditions. A referenced binary, opaque, or otherwise uninspected artifact is not treated as a
successfully analyzed reference.

## Structured skill data

AISOP/AISP structured extraction consumes the already-bounded cache and shares the enclosing
processing deadline. It does not start a second unbounded filesystem traversal.

| Resource | Ceiling | Scope |
|---|---:|---|
| Candidate documents | 64 | One extraction |
| Bytes per document | 256 KiB | One candidate |
| Total structured input | 1 MiB | One extraction |
| Parsed nesting depth | 64 | One extraction |
| Parsed nodes | 4,096 | One extraction |
| Output records | 512 | One extraction |
| Extraction time | 2 seconds | One extraction, constrained by the bundle deadline |

## Recursive and transitive scans

Pre-scan recursive discovery uses bounded `scandir` traversal and does not construct YAML merely to
obtain a display name.

| Resource | Ceiling | Scope |
|---|---:|---|
| Recursive discovery entries | 10,000 | One invocation |
| Recursive entries retained for sorting | 1,024 | One directory |
| Recursive structured candidates | 1,024 | One invocation |
| Recursive structured candidate bytes | 16 MiB | One invocation |
| Recursive discovery time | 2 seconds | One invocation |
| Recursive skills scanned | 32 | One invocation |
| Recursive public finding/occurrence records | 10,000 | One combined report |
| Recursive serialized report characters | 4 Mi characters | One combined report |

All recursively scanned roots share the same artifact, byte, and workflow deadline rather than
receiving a fresh allowance per child. If discovery or scanning reaches a ceiling, the arbitrary
partial skill list is discarded or the unscanned suffix is represented by one sanitized omitted-
scope record. The aggregate JSON, Markdown, and SARIF projections carry partial completeness.

Transitive external-reference scanning adds the following shared ceilings. Root and dependency
work consume the same allowance.

| Resource | Ceiling | Scope |
|---|---:|---|
| External targets | 32 | One traversal |
| Downloaded and cached source bytes | 10 MiB | Root and dependencies combined |
| Discovered and expanded artifacts | 10,000 | Root and dependencies combined |
| Traversal time | 600 seconds | Root and dependencies combined |
| Reference source records | 1,024 | One extraction |
| Reference source bytes | 1,000,000 | One extraction |
| Raw reference candidates | 4,096 | One extraction |
| Accepted references | 256 | One extraction |
| Frontier references | 4,096 | One traversal |

Reference extraction uses the bounded local deterministic cache, including locally inspected hidden
and nested content. Each dependency receives an opaque content-bound identity; display URLs remain
separate from finding, suppression, risk, and SARIF identity. A root baseline cannot glob-suppress a
dependency finding before that provenance is attached.

Remote Git materialization treats partial-clone filters as hints, not enforcement. While Git is
running, SkillSpector repeatedly measures the bounded clone tree, terminates the process when its
entry, byte, or deadline ceiling is crossed, discards subprocess output instead of buffering it, and
removes the rejected partial checkout.

## Ledger, analyzer, and finding output

| Resource | Ceiling | Scope |
|---|---:|---|
| Inspection-ledger events | 10,000 | One graph execution |
| Build-context ledger events | 10,000 | One bundle context |
| Static findings | 10,000 | One artifact |
| Static findings | 10,000 | One analyzer |
| Static-analysis time | 30 seconds | One artifact |
| YARA rule-directory entries | 10,000 | Built-in and optional directories combined |
| YARA rule files | 1,024 | One rule load |
| YARA rule source bytes | 1 MiB | One rule file |
| YARA rule source bytes | 16 MiB | One rule load |
| YARA rule active processing time | 5 seconds | One rule load, within the workflow wall-clock deadline |
| Retained YARA string instances | 4,096 | One matched rule |
| Shipped-bytecode discovery entries | 10,000 | One analyzer execution |
| Shipped-bytecode traversal depth | 64 | One analyzer execution |
| Shipped-bytecode analysis time | 5 seconds | One analyzer execution, within the workflow deadline |
| Dependency manifests | 64 | One analyzer execution |
| Dependency packages | 256 | One manifest |
| Dependency packages | 1,024 | One analyzer execution |
| Dependency findings | 2,048 | One analyzer execution |
| Dependency analysis time | 30 seconds | One analyzer execution, within the workflow deadline |
| OSV packages / query batches / detail requests | 256 / 4 / 64 | One dependency analysis budget |
| OSV response bytes / retained results | 4 MiB / 256 | One dependency analysis budget |
| TP4 source files / batches / findings | 128 / 64 / 64 | One analyzer execution |
| TP4 source and prompt input | 4 MiB each | One analyzer execution |
| TP4 model input | 32,000 tokens | One batch |
| Public finding and occurrence records | 10,000 | One report |

Ledger and public finding/occurrence truncation reserve an explicit `output_limit` record. The public
record cap is applied after severity-ordered deduplication; risk scoring still considers all retained
active findings before report output is bounded. Reaching an output ceiling therefore cannot silently
turn a truncated result into a complete result. Findings already produced by deterministic analyzers
remain primary evidence; optional semantic analysis may enrich them but does not select them out. If
the projection ceiling is reached, the severity-ordered bounded output and its explicit
`output_limit` record apply.

The shared static runner guards both findings constructed inside a pattern module and findings
emitted by returned iterables, so a module cannot first materialize an attacker-sized private list
and rely on later report truncation. Runtime is checked before and after trusted module calls and
during finding construction/emission. YARA uses its engine timeout and fast match mode, applies the
same per-artifact and per-analyzer finding ceilings, and bounds retained string instances per rule.
An overrun is nonfatal incomplete work rather than a clean scan or an execution crash.

## Fail-closed partial behavior

Resource-limit events carry an allowlisted reason and the applicable observed and limit values.
Affected inventory rows become `partial`, `failed`, or opaque as appropriate. Finalization exposes
the result through `analysis_completeness`, including coverage counts, ledger exceptions, analyzer
statuses, references, and limitations.

When relevant analysis is incomplete:

- A recommendation that would otherwise be `SAFE` is raised to at least `CAUTION`.
- Terminal, JSON, Markdown, and SARIF reports expose the incomplete status and its bounded reason.
- `skillspector scan --fail-on-incomplete` exits with status 1. Without this option, the CLI retains
  its compatibility behavior and still applies its ordinary risk-score exit policy. Execution
  failures exit with status 2.
- MCP responses set `safe_to_install` to `false` when analysis is incomplete, any relevant file is
  entirely uninspected, execution failed, or the risk score exceeds the installation threshold.

A low score or zero findings must not be interpreted as complete coverage when
`analysis_completeness.is_complete` is false.


## Configuring the aggregate workflow deadline

The scanner limits one complete workflow to 600 seconds by default. Set
`SKILLSPECTOR_MAX_WORKFLOW_SECONDS` to a positive finite number of seconds to
change that aggregate deadline. The setting applies to direct,
CLI, recursive, and multi-skill scans; byte and artifact ceilings remain in
effect. Invalid, zero, negative, infinite, or NaN values safely keep the
600-second default.
