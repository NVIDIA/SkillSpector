# Structured source roles (phase 2, bounded first increment)

This change adds auxiliary source-location context to unsuppressed AISOP/AISP
findings in the generated terminal, Markdown, JSON and SARIF reports. It follows
issue #130 and the phase-1 summaries in #211. It does not implement phase-3 risk
adjustment, reduce reported false positives, or certify a skill or runtime.

## Reading the annotations

A report finding may contain `evidence.structured_source_role`:

- `mapping_status`: `exact`, `unknown`, or `unavailable`.
- `text_role`: the owned string value's source role, such as `executable_step`,
  `constraint`, `resource_declaration`, or `unknown`.
- `structured_source`: a JSON Pointer shown through the existing report sanitizer.
- `source_span`: the matched original Unicode-character range, end exclusive.
- `content_sha256`: the algorithm-tagged digest of the exact cached file bytes.
- `risk_polarity`: always `unknown` in this increment.
- `role_confidence`: null; no calibrated probability is claimed.
- `reason`: a bounded machine-readable explanation of the mapping result.

An exact structural location is not evidence that its text is benign or that a
control was enforced. A non-reserved field in an AISOP function is an executable
step even when it is named `hard_deny` or `example`. No keyword-based safety
classification is performed. The recognized envelope is a supported layout, not
a complete AISP/AISOP conformance validation.

`structured_source` is sanitized diagnostic text and can differ from the original
pointer if it contains credentials or display controls. Use the content digest
and `source_span` for original-byte/coordinate correlation, not the sanitized
pointer alone. The digest does not identify a complete transitive repository tree
and does not establish execution, provenance trust, or safety.

## Isolation from safety decisions

The annotation pass runs after baseline suppression, risk calculation and finding
compaction. It uses expanded occurrence-specific display copies. The returned
canonical `findings`, `active_findings`, `filtered_findings`, suppressed findings,
fingerprints, severity, detection confidence and risk result are not enriched.
This prevents a subsequent report or transitive aggregation from using role
metadata as new classification evidence.

Each occurrence is mapped independently. Identical text in an executable step
and a constraint does not acquire one representative role. Existing report
sanitization/redaction and output escaping also apply to the added fields.

Phase-1 `structured_summaries` remain separate and unchanged. The MCP response's
embedded report includes the rendered annotations; its canonical `findings`
array does not. This is intentional for this bounded report-only increment.

## Supported evidence and conservative fallbacks

Only paths admitted in the scanner's `components` and present in both its raw
byte and deterministic text caches are mapped. The strict UTF-8 decode must equal
the text used by the deterministic scan. The scanner-owned inspection ledger
must attribute the finding to a static analyzer. Both end-exclusive columns are
required; there is no full-text search to guess a missing location.

This increment returns an explicit unknown for normalized/reconstructed views,
transitive provenance, unexpanded occurrences and missing columns. It does not
borrow a same-named local file for an external source. Suppressed findings and
sources outside the admitted local `.aisop.json` candidate set are not annotated.
Supporting additional scopes will require their own source-coordinate binding
and regression tests rather than changing these fallbacks to optimistic guesses.

Malformed JSON, decoded duplicate keys, invalid UTF-8, unsupported layout and
resource-limit failures produce no exact source role. The original finding is
retained in every case. Strings are located with a bounded structural walk using
Python's JSON decoder for scalar/string syntax. The existing reconstruction
helper identifies literal spans but not their object/array paths; it is not
silently treated as a JSON Pointer ownership map.

## Auxiliary coverage and budgets

The pass has per-report caps on documents (64), input bytes (1 MiB), annotation
records (512), ASCII-escaped annotation JSON characters (64 KiB), and elapsed
mapping time (2 seconds).
The output charge is `len(json.dumps(annotation, ensure_ascii=True))` for each
retained annotation, so Unicode and quotation-mark escaping are included. It
bounds the annotation objects before report indentation and enclosing fields; it
is not a byte or memory limit for the entire report.

A supplied workflow deadline can only shorten that time. The source index adds
per-file size, nesting, value, string and pointer-length limits. Source files are
never reopened and the pass never executes workflow steps or follows resources.

JSON reports expose `structured_role_coverage`; SARIF carries the same accounting
as invocation property `structuredRoleCoverage`. Terminal/Markdown reports show
exact, unknown/unavailable and omitted counts. A role limitation is auxiliary:
it does not overwrite the scanner's detection completeness or risk gate and it
must not be advertised as a clean security result.

## Verification before submission

With the project's dependencies installed, run the new tests and the affected
upstream tests, followed by the repository's lint/format and full offline suite:

```sh
pytest tests/unit/test_structured_source.py tests/unit/test_structured_role_report.py tests/nodes/test_structured_role_reporting.py
pytest tests/nodes/test_report.py tests/nodes/test_deduplicate.py tests/unit/test_suppression.py
make lint
make format-check
make test-unit
```

The development bundle's isolated-import runner is not an upstream test runner
and is not part of this proposed source change. Its results do not substitute
for installed-package, automatic-registry, graph, CLI or MCP tests.

References: [#130](https://github.com/NVIDIA/SkillSpector/issues/130),
[#211](https://github.com/NVIDIA/SkillSpector/pull/211),
[AISOP V1.0.0](https://github.com/AIXP-Labs/AISOP/blob/377ad24ffd88d76de9494f71f9311287877dc071/specification/aisop-spec.md),
[AISP V1.0.0](https://github.com/AIXP-Labs/AISP/blob/68777bf65b2229148d599a4abbe2fa931fb756a8/specification/AISP_Protocol.md).
