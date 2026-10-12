# Nested Artifact Inspection

SkillSpector inventories hidden regular files and inspects ZIP-compatible content locally. The
container is recognized from its bytes and internal structure rather than its filename extension.
Supported document containers are DOCX, XLSX, and PPTX; generic ZIP and nested ZIP-compatible
members use the same traversal policy.

Nested members use a stable virtual path that retains their full provenance:

```text
outer-file!/nested.zip!/scripts/setup.sh
```

## JSON component locations

JSON component rows preserve this virtual path in `path`, matching finding locations and
inspection-ledger entries. Archive members also expose their existing inventory provenance:

```json
{
  "path": "bundle.whl!/package/module.py",
  "type": "python",
  "executable": true,
  "outer_path": "bundle.whl",
  "nested_path": "package/module.py",
  "container_type": "zip",
  "container_ancestry": ["zip"],
  "container_depth": 1
}
```

`outer_path` names the physical container relative to that component's source root. `nested_path`
retains the full member chain inside it, including any nested `!/` boundaries. Each member keeps
its own component row, executable flag, type, line count, and byte size. Ordinary files do not gain
archive fields merely because their names contain `!/`.

Consumers must distinguish a virtual member from an on-disk file: `executable: true` describes
the member's content, not the existence of its virtual path on disk. Use the source identity and
full `path` to join components with findings; do not collapse members by `outer_path`. When
validating a report, check archive provenance and containment within the corresponding source
root, and retain incomplete-inspection failures. These additive fields do not change existing
consumer validators automatically, prove that every member was inspected, or justify extracting
or executing archive content.

## Security invariants

- Members are read in memory. SkillSpector never extracts, renders, imports, installs, or executes
  archive content.
- Absolute paths, parent traversal, drive-qualified paths, and link members are not followed.
- Hidden files, recognized containers, and all nested content are local-only and are never included
  in an external LLM request.
- Deterministic HIGH findings survive optional LLM meta-analysis.
- A zero-finding result does not make opaque or uninspected content complete.

## Cumulative bounds

Archive inspection uses one shared budget for the whole skill bundle. Opening another outer
container does not reset the member, expanded-byte, or time budget. The bundle scanner may pass a
smaller remaining artifact or byte allowance, and its deadline always takes precedence.

| Bound | Limit | Scope |
|---|---:|---|
| Container depth | 3 | One outer-to-inner provenance chain |
| Members | 1,000 | All outer and recursively nested containers combined |
| Expanded member bytes | 25 MiB | All outer and recursively nested containers combined |
| Central directory | 4 MiB | Each container, checked before creating ZIP metadata objects |
| Materialized member | 1,000,000 bytes | Each member |
| Compression ratio | 100:1 | Each member |
| Inspection wall time | 5 seconds | All outer and recursively nested containers combined |

The 1,000-member and 25 MiB ceilings are also reduced to the bundle scanner's remaining
10,000-artifact and 64 MiB canonical-byte budgets. Nested members therefore cannot obtain a fresh
allowance after ordinary files have consumed part of the bundle budget.

Before Python's ZIP reader is invoked, SkillSpector validates the terminal EOCD or ZIP64 records,
the declared central-directory count and byte size, and the actual sequence of central-directory
headers. This preflight prevents a forged entry count from causing an unbounded metadata list.
Already-bounded outer bytes are reused from the bundle cache instead of being read a second time.

These are resource-safety limits, not trust configuration. They are intentionally not user-managed
allowlists. See [Analysis Resource Bounds](ANALYSIS_RESOURCE_BOUNDS.md) for the enclosing bundle,
parser, ledger, and finding ceilings.

## Failure and completeness behavior

Malformed, encrypted, truncated, unreadable, unsafe-path, link, unsupported, and over-budget
members are recorded as inspection-ledger exceptions. Readable members retain their exact raw
bytes and a local-only decoded view. Unreadable members retain an opaque inventory record with a
`partial` or `failed` disposition; they are never represented as successfully analyzed.

The scan continues safely when possible, but any relevant omitted or partially inspected content
makes the analysis incomplete. A result that would otherwise be `SAFE` is reported as at least
`CAUTION`; the MCP verdict sets `safe_to_install` to `false`. CLI users can make incomplete analysis
a failing gate with `--fail-on-incomplete`. Inspection exceptions and their affected outer or nested
paths are surfaced in terminal, JSON, Markdown, and SARIF output.

## SC9: Concealed Executable Artifact

SC9 is a deterministic HIGH finding when executable content is concealed inside an Office document
container or a hidden/disguised artifact. Executability is established from an executable suffix,
a shebang, or archive mode bits. A benign document without executable members does not produce SC9.

SC9 reports evidence and risk; it does not execute the member or prescribe an installation decision.
