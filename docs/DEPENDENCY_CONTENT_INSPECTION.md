# Locked npm dependency content inspection

Advisory checks identify known vulnerable releases. They do not automatically
inspect the source shipped in an external package. Opt in to content inspection:

```bash
skillspector scan ./my-skill --inspect-dependencies --no-llm \
  --fail-on-incomplete --format json
```

The flag also applies to each child of a recursive scan and to external sources
followed by `--transitive`. The npm dependency graph comes from the selected
lockfiles; `--transitive` is not required to inspect their recorded transitive
packages. MCP Registry mode rejects the flag. Direct graph callers can set
`inspect_dependencies: True`; the MCP tool does not expose a new argument.

## Supported artifacts

- `package-lock.json` and `npm-shrinkwrap.json`, lockfile versions 2 and 3.
  A sibling shrinkwrap takes precedence over a package lock.
- Exact HTTPS `.tgz` URLs on `registry.npmjs.org`, with valid SHA-512 SRI.
  Registry redirects, credentials, query strings, alternate hosts, HTTP, local
  paths, Git dependencies, workspace links, and bundled lock entries are
  unsupported and produce incomplete inspection.
- Every recorded package entry, including development and optional packages and
  nested transitive installations. Aliases retain the artifact's actual package
  name and all lockfile installation locations. Identical locked artifacts are
  inspected once within an invocation.

The scanner uses cached lockfile bytes from bundle discovery. It rejects malformed
or duplicate-key JSON, unsupported lockfile versions, and disagreement between a
sibling package manifest's dependency declarations and the lockfile root. A
manifest in another directory cannot borrow a project's lockfile. Requesting
inspection without an available lockfile records incomplete coverage.

The downloader does not invoke npm, resolve versions, read `.npmrc`, use a
package-manager cache, or inherit proxy and credential configuration. DNS lookup
has its own process timeout; acquisition pins a validated public address while
retaining registry TLS certificate verification and SNI. Redirects and HTTP
content encodings are rejected. Downloaded bytes must match a recorded SHA-512
digest before the artifact reaches the inspection worker. The package's own
`package.json` name and version must also match its locked identity.

HTTP acquisition runs in a disposable downloader process. The parent enforces
the remaining absolute dependency/workflow deadline across process startup,
connection, TLS, response headers, body streaming, and client cleanup, killing
and reaping the downloader on timeout. Slow streams cannot extend the deadline
by sending bytes within each read timeout. Failed acquisition conservatively
charges its granted download-byte allowance when exact consumption is unavailable.

## Static analysis and isolation

Package code is never installed or executed. Each verified tarball is passed to
a fresh isolated Python process with a minimal environment. After trusted scanner
imports, the worker closes additional file descriptors and installs a Linux
seccomp syscall allowlist synchronized across its threads. The allowlist denies
filesystem access, networking, new processes, and execution. No archive member is
written to disk. If this isolation cannot be established, inspection fails closed
with `dependency_sandbox_unavailable`.

This worker currently requires Linux on x86-64 or AArch64 and a kernel that permits
seccomp filter synchronization. Other platforms retain ordinary SkillSpector
scanning; requested dependency inspection remains incomplete.

Supported UTF-8 text extensions are `.js`, `.mjs`, `.cjs`, `.ts`, `.json`, `.py`,
`.sh`, `.bash`, `.zsh`, `.ps1`, `.md`, `.markdown`, `.txt`, `.yaml`, `.yml`, and
`.toml`; conventional license and notice files are also inspected. The worker
uses the existing data-exfiltration, privilege-escalation, and supply-chain
pattern analyzers. It decodes every npm script command, including installation
hooks and commands reached through `npm run`, while retaining the script name
and original `package.json` property location in findings.

Dependency contents receive local static analysis only. This pass does not send
them to an LLM provider, perform additional advisory requests, or run YARA.
Ordinary analysis of the skill still follows its existing LLM, YARA, and OSV
settings. Use `--no-llm` to keep ordinary source analysis local as well.

Unsafe paths, duplicate members, links, special files, malformed archives,
unexamined trailing payloads, unsupported encodings, native modules, binaries,
and other unsupported formats remain visible as inspection gaps. The scanner does
not fetch additional payloads named in installation scripts or generated at runtime.

## Limits

| Resource | Default ceiling |
|---|---:|
| Locked entries considered | 128 per invocation |
| Lockfiles considered | 128 per invocation |
| Lockfile or sibling manifest bytes | 1 MiB each |
| Downloaded bytes | 32 MiB total; 8 MiB per artifact |
| Decompressed tar bytes | 64 MiB total; 16 MiB per artifact |
| Archive headers considered | 4,096 total; 1,024 per artifact |
| Source bytes per member | 1 MiB |
| Compression ratio | 200:1 |
| Decoded package scripts | 128 per package |
| Findings | 1,000 per invocation |
| Dependency pass wall time | 120 seconds |
| Worker wall time | 30 seconds, including startup |
| Worker CPU time | 30 seconds |
| Worker address space | 1 GiB |
| Worker output | 4 MiB |

The remaining shared workflow time, byte, and artifact budgets can reduce these
allowances. Downloads and decompressed tar bytes both consume shared workflow
bytes. A failed worker conservatively consumes its granted expansion allowance;
it cannot obtain a fresh budget through repeated failures. Byte accounting bounds
input work rather than all Python object overhead.

Four environment settings adjust aggregate dependency limits. Each is read at scan
time; invalid, non-positive, and non-finite values retain the default:

| Setting | Default |
|---|---:|
| `SKILLSPECTOR_MAX_DEPENDENCY_INSPECTION_PACKAGES` | `128` |
| `SKILLSPECTOR_MAX_DEPENDENCY_INSPECTION_SECONDS` | `120` |
| `SKILLSPECTOR_MAX_DEPENDENCY_INSPECTION_DOWNLOAD_BYTES` | `33554432` |
| `SKILLSPECTOR_MAX_DEPENDENCY_INSPECTION_EXPANDED_BYTES` | `67108864` |

The lockfile count follows the package-entry limit. Raising aggregate limits does
not raise per-artifact, worker, report, or shared workflow ceilings. Python callers
of `inspect_dependencies` may supply a `DependencyInspectionLimits` instance for
stricter per-call limits.

## Reports and completeness

Dependency findings appear in terminal, JSON, Markdown, and SARIF reports. Each
carries the package name, version, exact artifact SHA-256 digest, lockfile
references, and source location. Artifact identity also scopes finding
fingerprints. Script projections identify their original package manifest
property and the corresponding line within the decoded command.

`analysis_completeness.dependency_inspection` reports `not_requested`, `completed`,
or `partial`, with counts and bounded artifact records. Failed and unsupported
work also appears in the canonical completeness ledger and coverage inventory.
Uninspected content cannot count as fully inspected. Existing findings survive a
later artifact failure. Diagnostic byte counters ending in `_charged` can include
conservative budget reservations after failures.

Incomplete requested inspection prevents a `SAFE` recommendation and keeps
`safe_to_install` false where that verdict is exposed. `--fail-on-incomplete`
additionally causes a failing CLI exit status. With the flag disabled, reports
explicitly show that dependency content inspection was not requested.

Integrity verification establishes correspondence with the lockfile, not publisher
trust or package safety. Coverage is limited to recorded lock entries and the
supported static analysis described here. It does not certify lockfile graph
completeness or that a future installation will use only the inspected artifacts.
