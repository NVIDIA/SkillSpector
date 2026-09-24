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

## JSON quote ownership

The deterministic instruction parser can distinguish structural JSON quotes from instruction
delimiters only after validating a complete JSON value within **65,536 source characters**.
This is a structural capacity limit, independent of file-size and scan-time allowances. It is
unchanged by this diagnostic update; larger JSON values are not newly supported.

The inclusive ceiling counts characters, not UTF-8 bytes or decoded JSON string lengths:

- Standalone JSON includes surrounding whitespace. After bounded frontmatter, only the body
  following the closing delimiter is counted.
- An explicitly labeled, closed JSON fence counts its body, including newlines and raw list or
  blockquote prefixes. Opening and closing fence lines are excluded.
- Unicode characters count once; each source character of an escape such as `\u0061` counts.
  The frontmatter boundary search has its own 65,536-character prefix bound.

When marker reconstruction is already incomplete and its first unresolved directive lies in an
oversized JSON candidate, the ledger reports `json_quote_ownership_limit`. Its message identifies
the zero-based, end-exclusive source character span `[start, end)`, observed character count,
limit, and next step; the ledger also retains `observed_characters` and `limit_characters`.
The exception's `path` identifies the source artifact. It does not assert that the candidate is
valid JSON or that capacity is the only unresolved instruction issue. Invalid, truncated, and
oversized candidates never acquire quote ownership. Unrelated parser failures retain their
existing reason codes; an unclosed fence cannot establish a complete JSON body boundary.

To resolve a capacity limitation, split the input into smaller **complete** JSON values or
documents, preserve all required content and references, and rescan. Do not truncate the input
or remove a required reference. Raising the timeout cannot raise this structural ceiling.
Findings remain visible, and unresolved coverage remains incomplete: CLI `--fail-on-incomplete`
returns nonzero and the MCP installation gate rejects it, even after successful semantic analysis.
JSON with no unresolved instruction parsing may still complete without needing quote ownership.

A future increase requires an explicit supported-input decision and validation of dense strings,
deep nesting, escaped/Unicode text, malformed input, cancellation, and the interaction with analysis
windows. JSON decoding allocates synchronously between deadline checks; a larger benign example
passing in isolation does not validate that larger resource allowance. This patch retains the
existing bound and does not change expected-complete dataset contracts for oversized examples.

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

AE1 is suppressed only for a positively identified rendered Markdown image whose complete
bytes verify as a minimal, non-interlaced PNG and whose canonical inspection evidence has
only unsupported-format limitations. Active, ambiguous, escaped, code, and missing-kind
references retain AE1, as do non-PNG formats, executable or concealed content, other failures,
unknown reasons, and mixed limitations. Suppression leaves the coverage exception, coverage
metrics, incomplete-scan recommendation, and `--fail-on-incomplete` behavior unchanged. Such
an image-only limitation may report LOW severity while still recommending `CAUTION` because
the scan remains incomplete.

### Diagnosing incomplete referenced artifacts

AE1 uses the check name **Incomplete referenced artifact analysis**. Its location
is the reference in the source document. The affected file appears separately in
`evidence.target_path`, with its final `target_disposition` and up to 16 distinct
reason records. Each reason identifies the canonical `reason_code`, message,
phase, and analyzer; target line numbers and observed/limit values appear when
the ledger provides them. `reasons_truncated: true` means this finding contains
only a subset of the reasons. Review the target's entries in
`analysis_completeness.ledger_exceptions`, including any output-limit records,
before deciding how to resolve the failure.

Two references to the same partially inspected helper can produce two AE1
locations. They identify one affected artifact, rather than demonstrating two
independent vulnerabilities. Incomplete analysis alone also does not establish
malicious evasion. Keep required references and use the reason to choose the fix:

| Reason | Next step |
|---|---|
| `static_parse_limit` | Inspect the expression and analyzer. If valid source is misinterpreted, correct or update the scanner and rerun. |
| `json_quote_ownership_limit` | Use the reported source span and 65,536-character bound to split complete JSON values while retaining required content, then rescan. |
| `read_error`, `stat_error`, `file_disappeared`, `missing_file_cache` | Ensure the resolved target remains readable throughout the scan. |
| `size_limit`, `runtime_limit` | Review the reported bounds and input size; distinguish a scanner performance problem from a legitimate resource ceiling. |
| `binary_content`, `opaque_content` | Provide inspectable source or analysis support for the referenced format. |

The finding and incomplete-analysis gates remain active until the relevant
limitations are resolved. Evidence fields are diagnostic facts, not suppression
instructions.

### Perl literal help text

For `.pl` source, the tool-misuse analyzer recognizes a narrow form of standalone
`print`: an ordinary, single-line, non-interpolated quoted literal, optionally
with `STDOUT`/`STDERR` or parentheses. For example:

```perl
print "Use rm to remove a project\n";
```

LF and CRLF line endings, including a trailing comment, are supported while
source offsets are preserved. Line breaks inside the quoted literal remain
outside this recognized form.

When complete surrounding source proves those quote boundaries, the analyzer
keeps the literal's payload visible while distinguishing Perl delimiters from
shell delimiters. This prevents ordinary help text from creating a false shell
parse limit. Printed dangerous commands still receive security checks, and Perl
retains the existing prompt-injection and supply-chain checks.

This is bounded recognition, not a general Perl parser. Ambiguous quoting,
interpolation, quote operators such as `qx`, quote-like special variables,
legacy package separators, incomplete fragments, and real parser limits remain
on the conservative analysis path. A complex helper may therefore still need
its particular ledger reason and expression reviewed.

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
| Static-analysis time | 300 seconds | One artifact, within the workflow deadline |
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

## Configuring the static analysis deadline

Static pattern analysis and YARA matching allow up to 300 seconds per artifact by
default. Set `SKILLSPECTOR_MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT` to a positive
finite number of seconds to change that allowance. Invalid, zero, negative,
infinite, or NaN values log a warning and retain the 300-second default.

Each operation still uses the smaller of this allowance and the remaining workflow
time. Increasing it does not extend the aggregate workflow deadline. A limit
reached during analysis retains existing findings and reports partial work through
the inspection ledger. Both environment settings are read when their modules are
imported, so restart the SkillSpector process after changing them.
