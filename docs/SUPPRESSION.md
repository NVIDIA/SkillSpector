# Baseline / False-Positive Suppression

SkillSpector's analyzers — especially the LLM semantic ones — can produce
findings that are correct in general but not actionable for *your* skills
(framework/architectural patterns, first-party tooling conventions, accepted
lab practices). A **baseline** lets you suppress those known findings so that:

- the risk score reflects only **un-triaged** issues,
- re-scans surface only **new** findings (incremental CI/CD), and
- every suppression carries an auditable **reason**.

Suppressed findings never count toward the risk score or active finding count.
They remain in SARIF marked with an external suppression for auditability. They
are shown in the terminal/Markdown report only when you pass `--show-suppressed`,
and are always listed (machine-readable) in the JSON report under `suppressed` /
`suppressed_count`.

> Addresses [issue #88](https://github.com/NVIDIA/SkillSpector/issues/88).

## Quick start

```bash
# 1. Accept all current findings into a baseline (run once).
skillspector baseline ./my-skill/ -o .skillspector-baseline.yaml

# 2. Commit the baseline, then scan against it. Only NEW findings are reported.
skillspector scan ./my-skill/ --baseline .skillspector-baseline.yaml

# Review what was suppressed.
skillspector scan ./my-skill/ --baseline .skillspector-baseline.yaml --show-suppressed
```

## CLI

| Command / option | Description |
|------------------|-------------|
| `skillspector baseline <path> [-o FILE] [--no-llm] [--reason TEXT]` | Scan and write a baseline that fingerprint-suppresses every current finding. Default output: `.skillspector-baseline.yaml`. |
| `skillspector scan <path> --baseline FILE` (`-b`) | Suppress findings matching the baseline before scoring/reporting. |
| `skillspector scan <path> --baseline FILE --show-suppressed` | Also list the suppressed findings (they still don't affect the score). |

A missing, malformed, or unsupported baseline file exits with code 2.
Generation also exits with code 2 without creating or replacing the output when
scan execution fails, including failed semantic analysis or unreadable primary
content. Resolve the failure and rerun the scan before accepting its findings.
Use `--no-llm` when you intentionally want to accept only static findings.

A successful but incomplete scan can generate a baseline for its observed
findings, with a warning on stderr. This does not accept uninspected content or
clear coverage gaps: subsequent reports retain their incomplete status and
recommendation even if every observed finding is suppressed.

When a selected baseline or baseline output is stored inside the scan target,
SkillSpector treats that exact file as an explicit scope exclusion. This
prevents sensitive rule text from creating a finding against itself or entering
regenerated fingerprints. Other baseline files and sibling YAML/JSON files
remain in normal scan scope unless they are selected with `--baseline` or `-o`.

## Baseline file format

YAML or JSON (the `.json` extension selects JSON output when generating). Two
complementary mechanisms:

```yaml
version: 2
scanner_version: "X.Y.Z" # generated automatically; do not edit

rules:                       # human-authored, glob-based, drift-tolerant
  - id: "SQP-1"              # glob over the finding's rule id
    reason: "Trigger-phrase breadth is a description nit, not a vuln"
  - id: "SSD-2"
    path: "example-skill/SKILL.md"       # glob over the finding's file
    message: "*example false-positive phrase*"   # glob over its description or matched text
    reason: "False positive: benign trigger phrase, not an instruction"

fingerprints:                # machine-generated, exact
  - hash: "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    rule_id: "SDI-2"         # informational (for humans reading the file)
    file: "example-skill/SKILL.md"
    reason: "Accepted — reads its own environment for context"
```

### `rules` — glob suppression

A finding is suppressed when **every** field a rule specifies matches it;
unspecified fields match anything. Use this for:

- **Global pattern suppression** — `id: "SQP-1"` (or `id: "SQP-*"`) drops a rule
  or rule family across all skills.
- **Skill/file-scoped suppression** — add `path:` (and optionally `message:`) to
  scope the suppression to a specific skill, file, or message.

Field reference:

| Field | Matches against | Notes |
|-------|-----------------|-------|
| `id` (or `rule_id`) | `Finding.rule_id` | glob |
| `path` (or `file`) | `Finding.file` | glob; `*` crosses `/`, `**` is an alias for `*` |
| `message` | `Finding.message`, plus the matched text shown as `finding` in reports | glob, case-insensitive; wrap a keyword in `*` for substring |
| `reason` | — | required; recorded in reports and audits |

Glob matching uses Python's [`fnmatch`](https://docs.python.org/3/library/fnmatch.html),
so `*` matches across path separators (`*SKILL.md` matches `a/b/SKILL.md`).
Rules are **drift-tolerant**: they keep working after line numbers shift or
content is reworded.

### `fingerprints` — exact suppression

Each entry is a full SHA-256 digest over canonical JSON that binds the finding
to the SkillSpector version, normalized component path, complete decoded text
presented to the scanner, and every risk/evidence field (including rule,
severity, confidence, location, matched text, context, intent, and tags).
Generated by `skillspector baseline`, it is intentionally exact:
editing the source or upgrading SkillSpector keeps the finding active until it
is reviewed and the baseline is regenerated.

LLM and meta-analysis can change explanations, remediation, confidence, or other
evidence between scans of unchanged files. Those results remain active when
their exact fingerprints differ. For repeatable static acceptance, use
`--no-llm` for both baseline generation and subsequent scans.

Generation fingerprints each original finding before report compaction, including
repeated matches on different lines and identical matches in different files.
Only identical exact fingerprints share an entry. A baseline generated by an
affected version may omit these occurrences; regenerate it to include them.
If the complete baseline exceeds the loader's size or record limits, generation
fails before replacing the output file instead of writing a partial baseline.
After validation, generation normally writes a complete temporary file beside the
destination and atomically replaces the output. A failed write or replacement
preserves the existing baseline. Existing destinations must be writable regular
files; symlinks and special files are rejected. Replacement requires a writable
parent directory. On POSIX, new files grant access only to their owner; existing
ordinary permission bits and ownership are preserved, including group-write access.
When a non-owner has write access to an existing file, or the owner cannot
assign the file's group because it is not a member, generation updates its
validated descriptor in place and preserves its ACLs. This shared-file fallback
serializes cooperating writers, but readers can see a partial write and an I/O
failure or interruption can leave a partial baseline. Use an owner-managed output
when atomic replacement is required.

Every v2 entry must be a mapping with a 64-hex-character `sha256:` hash and a
non-empty `reason`. `rule_id` and `file` are informational fields for reviewers.
If source content is unavailable or `scanner_version` does not match, exact
fingerprints fail closed and suppress nothing. Use `rules` only when you
intentionally want a reviewed suppression to survive source drift.

### Migrating version 1 baselines

Version 1 fingerprints omitted the matched evidence and source content, so a
benign and malicious finding could share a fingerprint when rule, file, line,
and generic message were unchanged. They cannot be upgraded safely without a
new scan and human review. SkillSpector rejects version 1 files that contain
fingerprints; rerun `skillspector baseline`, re-triage every generated entry,
and commit the v2 file. Legacy files containing only explicit rules remain
loadable with a warning so reviewed policy suppressions are preserved. Do not
copy old hashes into the new file.

Recursive multi-skill scans do not accept one shared baseline because exact
fingerprints are scoped to each independently scanned skill. Run each sub-skill
with its own baseline. A single-skill scan still supports `--recursive` together
with `--baseline`.

## How it fits the pipeline

Suppression is applied in the **report node** (`skillspector/nodes/report.py`),
the single place where findings are scored and formatted, so the CLI and any
future REST API behave identically. The CLI loads the baseline file into a
`skillspector.suppression.Baseline` and passes it via graph state
(`state["baseline"]`, `state["show_suppressed"]`); the report node partitions
findings into kept vs. suppressed via
`skillspector.suppression.partition_findings`.

## Recommended workflow

1. Triage the first scan and generate exact v2 fingerprints for individually
   accepted findings. Reserve drift-tolerant `rules` for deliberate,
   tightly-scoped policy suppressions: source changes do not invalidate them,
   so a broad rule can hide newly malicious content.
2. Commit the baseline file to the repo.
3. In CI, run `skillspector scan <path> --baseline <file>`; the build fails
   (exit 1) only when a **new** finding pushes the risk score above threshold.
4. Periodically review with `--show-suppressed` and prune stale entries.
