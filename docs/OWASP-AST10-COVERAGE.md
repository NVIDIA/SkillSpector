# OWASP Agentic Skills Top 10 (AST10) coverage matrix

Mapped against: OWASP Agentic Skills Top 10, version 1.0-2026, public review v1, [repo commit `0e5a4c0601e41f1f6eda14da1017034c0bd9cbfb`](https://github.com/OWASP/www-project-agentic-skills-top-10/tree/0e5a4c0601e41f1f6eda14da1017034c0bd9cbfb)
Retrieved: 2026-07-19
Status: Informational, documentation only

> Addresses https://github.com/NVIDIA/SkillSpector/issues/221.

## Scope

This page is a revision-pinned crosswalk between SkillSpector's current rule catalog and one concrete OWASP AST10 revision. It helps readers reason about where current SkillSpector rules align with AST10 categories, where the alignment is partial, and where the repo has a documented gap.

It is an informational crosswalk, not an assurance claim, regulatory attestation, or exhaustive assessment. AST10 is still evolving, and SkillSpector's own rule set will continue to change.

## Terminology note

OWASP uses `AST10` to name the Agentic Skills Top 10 project. SkillSpector also uses `AST1` through `AST9` as internal rule ids for the Behavioral AST analyzer family. Those names are unrelated. In this page, `AST01` through `AST10` refer to OWASP risk categories, while `AST1` through `AST9` refer to SkillSpector rule ids.

## Method

This matrix is anchored to:

- OWASP AST10 source pages at the pinned commit linked above
- SkillSpector's current rule catalog in [README.md](https://github.com/NVIDIA/SkillSpector/blob/8f534e2951e0b7d0b8fb8e84832cd3605f95c032/README.md#vulnerability-patterns)

Each row asks a narrow question: which current SkillSpector rules are directly relevant to this AST10 risk, and what remains outside the tool's current surface.

Coverage labels are intentionally conservative:

- `Related rules present` means the current rule catalog has direct signals for the risk.
- `Partially addressed` means the current rule catalog exposes some symptoms or related mechanisms, but not the whole risk surface.
- `Not currently addressed` means the current rule catalog does not directly model the category.

## Matrix

| Category | Related SkillSpector rules | Coverage level | Rationale |
|---|---|---|---|
| AST01 Malicious Skills | `P1`-`P5`, `AR1`-`AR3`, `E1`-`E4`, `PE3`, `SC2`, `SC3`, `MP1`-`MP3`, `RA1`, `RA2`, `AST1`-`AST9`, `TT3`-`TT5`, `YR1`-`YR4`, `TP1`-`TP3` | Related rules present | Current rules detect malicious instructions, secret theft, persistence, dangerous execution chains, known malware signatures, and poisoned metadata commonly used by malicious skills. |
| AST02 Supply Chain Compromise | `SC1`-`SC6` | Related rules present | The supply-chain family covers unpinned dependencies, remote script fetching, obfuscated execution, known vulnerable packages, abandoned packages, and typosquatting. |
| AST03 Over-Privileged Skills | `PE1`-`PE3`, `EA1`-`EA4`, `LP1`-`LP4` | Related rules present | Current rules flag excessive permissions, unrestricted tool or resource access, scope creep, and mismatches between declared and observed MCP capabilities. |
| AST04 Insecure Metadata | `P2`, `LP1`-`LP4`, `TP1`-`TP4`, `TR1`-`TR3` | Related rules present | Current rules detect hidden instructions, poisoned MCP metadata, trigger abuse, and permission declaration mismatches that make skill metadata deceptive or unsafe. |
| AST05 Untrusted External Instructions | `P1`-`P4`, `SC2`, `TP1`-`TP3`, `TT5` | Partially addressed | Current rules can detect dangerous instructions and remote execution patterns once the content is present in the scan input, but SkillSpector does not inventory or pin every mutable external instruction source by itself. |
| AST06 Weak Isolation | `PE2`, `EA1`, `EA4`, `TM3`, `AST1`, `AST4`, `AST5`, `TT5` | Partially addressed | Current rules highlight behaviors that become more dangerous when a skill runs with weak process, filesystem, shell, or network isolation, but they do not prove the deployed sandbox or runtime boundary. |
| AST07 Update Drift | `SC1`, `SC4`, `SC5` | Partially addressed | Dependency pinning, live vulnerability checks, and abandoned-package detection expose some update-drift risk, but the tool does not track installed package state, rollout history, or patch lag in a live environment. |
| AST08 Poor Scanning | Static patterns, Behavioral AST, taint tracking, YARA, MCP least privilege, MCP tool poisoning, optional LLM semantic pass | Partially addressed | SkillSpector exists to improve scanning of agentic-skill specific risks, but it does not execute skills at runtime, fetch every external surface automatically, or settle every evasion path on its own. |
| AST09 No Governance | none directly | Not currently addressed | Reports, baselines, and SARIF output can feed governance workflows, but the current rule catalog does not directly model approval workflows, ownership, audit policy, or revocation state. |
| AST10 Cross-Platform Reuse | `LP1`-`LP4`, `TP1`-`TP4`, `TR1`-`TR3`, `PE1`, `EA3` | Partially addressed | Current rules can expose permission drift, metadata deception, trigger mismatch, and scope creep after a cross-platform port, but they do not compare source and target manifests for semantic equivalence. |

## Coverage gaps and unknowns

- AST05 remains partial because SkillSpector scans what it is given; it does not recursively fetch, pin, or monitor every external instruction document that a skill may reference.
- AST06 remains partial because local code and metadata inspection are not the same thing as proving container, sandbox, namespace, localhost-auth, or egress policy enforcement.
- AST07 remains partial because current rules reason about dependency hygiene and known package risk, not the live patch level or update history of an installed deployment.
- AST08 remains partial because the scanner itself has bounded visibility. It does not provide runtime execution tracing, binary unpacking for every format, or exhaustive coverage of every attacker-controlled external surface.
- AST09 is not currently addressed as a direct rule surface. Governance needs inventories, approval controls, action logging, and revocation workflows that sit outside the current scanner.
- AST10 remains partial because cross-platform translation can drop or reinterpret security metadata in ways that require source-to-target manifest comparison, not only single-manifest analysis.

## What stays out of scope here

- No rule metadata fields are added.
- No SARIF or JSON taxonomy fields are added.
- No current rule ids or analyzer behaviors change.

Those follow-ups can be revisited after the AST10 taxonomy settles further.

## SkillSpector-specific limits that matter here

This mapping should be read alongside the repo's documented limits:

- SkillSpector is a static and optional LLM-assisted scanner, not a runtime sandbox.
- Coverage depends on the content being present in the scan input.
- The repo's own [trust model and data egress](https://github.com/NVIDIA/SkillSpector/blob/8f534e2951e0b7d0b8fb8e84832cd3605f95c032/README.md#trust-model-and-data-egress) and [limitations](https://github.com/NVIDIA/SkillSpector/blob/8f534e2951e0b7d0b8fb8e84832cd3605f95c032/README.md#limitations) sections still define what the tool can and cannot prove.

## Updating this page

When the OWASP AST10 project publishes a new revision, update this page by:

1. pinning the new revision explicitly
2. rechecking the exact AST01-AST10 names
3. rerunning the mapping against the current SkillSpector rule catalog
4. rewriting any rows whose rationale changed

## References

- OWASP AST10 home page: `index.md` at the pinned commit
- OWASP AST10 visual overview: `top10.md` at the pinned commit
- OWASP AST10 category pages: `ast01.md` through `ast10.md` at the pinned commit
- SkillSpector rule catalog: https://github.com/NVIDIA/SkillSpector/blob/8f534e2951e0b7d0b8fb8e84832cd3605f95c032/README.md#vulnerability-patterns
- Maintainer scope for issue #221: https://github.com/NVIDIA/SkillSpector/issues/221#issuecomment-5008664101
- OWASP project license: https://creativecommons.org/licenses/by-sa/4.0/
