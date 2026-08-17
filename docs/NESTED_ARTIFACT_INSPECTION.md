# Nested Artifact Inspection

SkillSpector inventories hidden regular files and inspects ZIP-compatible content locally. The
container is recognized from its bytes and internal structure rather than its filename extension.
Supported document containers are DOCX, XLSX, and PPTX; generic ZIP and nested ZIP-compatible
members use the same traversal policy.

Nested members use a stable virtual path that retains their full provenance:

```text
outer-file!/nested.zip!/scripts/setup.sh
```

## Security invariants

- Members are read in memory. SkillSpector never extracts, renders, imports, installs, or executes
  archive content.
- Absolute paths, parent traversal, drive-qualified paths, and link members are not followed.
- Hidden files, recognized containers, and all nested content are local-only and are never included
  in an external LLM request.
- Deterministic HIGH findings survive optional LLM meta-analysis.
- A zero-finding result does not make opaque or uninspected content complete.

## Cumulative bounds

The following fixed limits apply to one outer container and every nested container below it:

| Bound | Limit |
|---|---:|
| Container depth | 3 |
| Members | 1,000 |
| Declared/uncompressed content | 25 MiB |
| Materialized member | 1 MiB |
| Compression ratio | 100:1 |
| Inspection wall time | 5 seconds |

These are resource-safety limits, not trust configuration. They are intentionally not user-managed
allowlists.

## Failure and completeness behavior

Malformed, encrypted, truncated, unreadable, unsafe-path, link, unsupported, and over-budget
members are recorded as inspection-ledger exceptions. The scan continues safely when possible, but
the analysis is marked incomplete. Outer and nested paths remain visible in terminal, JSON,
Markdown, and SARIF output.

## SC9: Concealed Executable Artifact

SC9 is a deterministic HIGH finding when executable content is concealed inside an Office document
container or a hidden/disguised artifact. Executability is established from an executable suffix,
a shebang, or archive mode bits. A benign document without executable members does not produce SC9.

SC9 reports evidence and risk; it does not execute the member or prescribe an installation decision.
