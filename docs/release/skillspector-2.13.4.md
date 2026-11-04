# SkillSpector 2.13.4

This maintenance release includes the v2.13.3 fixes and selected improvements to
provider compatibility, source inspection, and recovery of malformed AI responses.

## Fixes

- [#806](https://github.com/NVIDIA/SkillSpector/pull/806): provider compatibility
  and structured responses for GPT-6.1 Sol and Claude Opus 5.5 across supported routes.
- [#947](https://github.com/NVIDIA/SkillSpector/pull/947),
  [#950](https://github.com/NVIDIA/SkillSpector/pull/950), and
  [#973](https://github.com/NVIDIA/SkillSpector/pull/973): bounded parsing of shell,
  PowerShell, JavaScript templates, Python strings, and Markdown documentation.
- [#974](https://github.com/NVIDIA/SkillSpector/pull/974): inspect supported generated
  shell helpers and archived sources without executing uploaded code. Dependencies
  outside the uploaded skill and unprovable runtime behavior remain incomplete.
- [#997](https://github.com/NVIDIA/SkillSpector/pull/997) and
  [#999](https://github.com/NVIDIA/SkillSpector/pull/999): distinguish unsupported
  syntax, unresolved parser boundaries, and exhausted resource limits.
- [#1000](https://github.com/NVIDIA/SkillSpector/pull/1000): bounded TAR/GZIP
  inspection with explicit accounting for omitted archive members.
- [#979](https://github.com/NVIDIA/SkillSpector/pull/979),
  [#998](https://github.com/NVIDIA/SkillSpector/pull/998), and
  [#1007](https://github.com/NVIDIA/SkillSpector/pull/1007): preserve provider
  failure details, bound retries, and distinguish outages from security findings.
- [#1011](https://github.com/NVIDIA/SkillSpector/pull/1011): modern JavaScript parsing
  and narrower classification of passive presentation assets and formatting metadata.
- [#1014](https://github.com/NVIDIA/SkillSpector/pull/1014): shared recovery of
  malformed AI assessments, including plain-text and individual-field requests.
  Recovered findings remain visible. Unverifiable formatting-only responses carry
  an explicit warning and retain unverified coverage; refusals, provider failures,
  and unrelated inspection gaps retain their separate diagnostics.
- The bounded advisory-count correction from
  [#1010](https://github.com/NVIDIA/SkillSpector/pull/1010), plus required literal,
  documentation, reference, and text-pattern prerequisites. The broader Python
  docstring exemption from that PR is excluded.

Integration corrections preserve malformed single-object findings, avoid repeated
shell-comment scans, and limit decoded signature metadata ownership to parser
boundaries while retaining checks on unknown fields and decoded payloads. The
maintenance branch's scoped allowed-tools and output-file protections are retained.
Markdown list fences end with their containing list, so following prose does not
extend a shell string. Source distributions contain the runtime package and public
package metadata.

## Scope and upgrading

This release does not resolve every parser limitation or guarantee that every
large skill completes. File, archive, retained-content, report, and time limits
remain enforced. Additional container support can consume the shared byte budget
and displace previously inspected members in oversized inputs; those omissions
remain explicitly incomplete, and disappearing findings behind a limit are not
security fixes.

Downstream consumers need compatible handling of `complete_with_caveats` and the
format-recovery warning policy. Updating the scanner alone does not change an
evaluator's decision to block incomplete inspection. Some consumers also need
separate handling for passive BMP assets inside chart archives.

Regenerate and review version-bound baselines for 2.13.4. Release artifacts are
published on GitHub as a wheel and source distribution; this workflow does not
publish to PyPI.
