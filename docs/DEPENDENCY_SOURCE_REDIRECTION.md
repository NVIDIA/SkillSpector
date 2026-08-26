# Dependency Source Redirection (SC10)

SC10 reports a deterministic `HIGH` finding when a supported direct configuration file changes
dependency resolution away from that ecosystem's built-in canonical default. The analysis is
local, static-only, and advisory: it reports evidence for review but does not decide whether a
skill should be installed.

## Direct configuration coverage

SC10 inspects only the following direct configuration surfaces:

| Ecosystem | Files | Inspected declarations |
|---|---|---|
| npm | `.npmrc`, `npmrc` | `registry` and scoped `@scope:registry` assignments |
| pip | `pip.conf`, `pip.ini` | `index-url` and `extra-index-url` assignments in configuration sections |
| Yarn | `.yarnrc`, `.yarnrc.yml`, `.yarnrc.yaml` | Yarn v1 `registry` and scoped registry entries; Yarn YAML `npmRegistryServer` and `npmScopes.*.npmRegistryServer` entries |
| Poetry | `pyproject.toml` | `[[tool.poetry.source]]` entries |
| PDM | `pyproject.toml` | `[[tool.pdm.source]]` entries |
| uv | `pyproject.toml`, `uv.toml` | `[[tool.uv.index]]` or `[[index]]` entries; a same-directory `uv.toml` takes precedence over the `pyproject.toml` uv table |
| Cargo | `.cargo/config`, `.cargo/config.toml` | `[source.*].registry`, resolvable `[source.*].replace-with` chains, and `[registries.*].index` |
| Maven | `settings.xml`, `pom.xml` | Settings mirrors and profile repositories/plugin repositories; direct project repositories/plugin repositories |

For Maven, `distributionManagement` descendants are outside this rule's direct-source scope.
For Cargo, directory, local-registry, and Git source targets are outside SC10's reporting scope.
A `replace-with` chain is reported only when it resolves to a `[source.*].registry` or
`[registries.*].index` destination.

The analyzer suppresses an unchanged canonical public default. It compares only the exact
built-in ecosystem defaults, with scheme and host case normalization and an optional trailing
slash. A port, query, fragment, or different path remains noncanonical. These fixed protocol
defaults are not a user-managed allowlist or trust list.

## Findings and incomplete direct parses

Each finding carries code-owned ecosystem, surface, operation, and scope values; a sanitized
destination; and the physical source range. URL credentials, queries, fragments, and non-root
paths are removed or replaced before evidence reaches a finding or public output. Supported
interpolation forms that cannot be resolved from the direct file are reported with the fixed
destination status `unresolved`; SC10 does not read environment variables or neighboring files.

Recognized direct files are accepted only from complete, strictly decoded cached artifacts. A
missing or inconsistent cache/inventory record, malformed or ambiguous relevant syntax,
unsupported relevant structure, invalid UTF-8, truncation, or resource exhaustion produces a
localized `dependency_source_parse_incomplete` limitation instead of a clean result.

Direct parser limits are shared across the scan where applicable:

| Resource | Limit |
|---|---:|
| Physical bytes per direct configuration file | 1,000,000 |
| Parsed configuration nodes | 50,000 |
| YAML aliases | 256 |
| Configuration depth | 64 |
| Retained source records | 50,000 |
| Retained literal bytes | 2,000,000 |
| Emitted source changes | 10,000 |

## Executable and generated configuration boundary

This implementation does not parse commands or generated configuration. It structurally
recognizes executable shell files, executable inventory entries, Dockerfiles containing `RUN`,
Make recipes, and shell-like Markdown fences only to report their affected ranges as
`unscanned_executable_content`. Those ranges are incomplete coverage pending the syntax-aware
parser follow-up; their contents do not produce SC10 findings in this implementation.

The coverage notice does not guess whether a dependency-source command is present. It prevents a
recognized executable surface from being represented as fully analyzed and can raise an otherwise
`SAFE` report to `CAUTION` through the existing completeness policy. It does not change risk
scoring or recommendation policy.

## Security and product boundary

SC10 does not execute project content, commands, package managers, or generated files. It makes no
network, DNS, or reputation requests; maintains no user-managed allow/block/trust lists; and adds
no telemetry, service, or worker. Optional provider analysis may add presentation context, but it
cannot suppress or downgrade the deterministic SC10 evidence.

The result remains advisory. A `HIGH` finding or incomplete-coverage notice is evidence for the
user's review, not an installation decision or certification.
