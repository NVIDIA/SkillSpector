# SkillSpector 2.13.1

This patch release is based on the published v2.13.0 maintenance branch. It
corrects false incomplete-analysis reports for ordinary Python report strings,
quoted script text, regular expressions, and possessive Markdown prose.

## Selected changes

- [#778](https://github.com/NVIDIA/SkillSpector/pull/778) establishes bounded
  Python and Perl host-language ownership for the shell parser.
- [#784](https://github.com/NVIDIA/SkillSpector/pull/784) shares validated Python
  quote ownership with marker reconstruction.
- [#849](https://github.com/NVIDIA/SkillSpector/pull/849) corrects prose and
  report-string parsing, using those prerequisites.
- Additional release hardening retains the previous incomplete-inspection
  result when a report-shaped string is later assembled into a shell command.
  The check covers recognized local execution and a bounded set of bundled
  helpers, including import-time initialization. It does not lower findings
  or suppress an incomplete scan.
- Source distributions exclude the `.uv-cache` directory used by the release
  workflow.

The release preserves v2.13.0's third-party dependency lock, provider and model
defaults, and existing publishing workflow. It does not take the remaining
unselected changes from `main`.

## Compatibility and remaining limits

Genuine incomplete inspection and independent security findings remain blocking
signals. A missing-reference advisory is distinct from an uninspected existing
artifact. This patch does not resolve every parser limitation: nested Markdown
table cases and references to otherwise inspected ZIP/XLSX containers retain
their existing limitations. Passive image handling is unchanged.

The bundled-helper check reads only materialized scan inputs. Unsupported,
ambiguous, missing, cyclic, or over-budget local evidence retains the previous
parser fallback. External imports and arbitrary interprocedural execution remain
outside this bounded proof; this release does not claim whole-program analysis.

Exact, version-bound finding baselines must be regenerated and reviewed for
2.13.1; do not relabel a baseline from another scanner version. Consumers should
validate the published wheel against their original inputs and baseline before
activating the upgrade.

Preparing or merging this source change does not itself publish an artifact.
Publication remains a separate maintainer action through the existing
label-gated release workflow.
