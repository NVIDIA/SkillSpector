# Dependency Source Redirection (SC10)

SC10 reports a deterministic `HIGH` finding when a supported direct configuration or
syntax-proven executable surface changes dependency resolution away from that ecosystem's built-in
canonical default. The analysis is local, static-only, and advisory: it reports evidence for review
by a human.

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

## Syntax-aware executable coverage

SC10 parses canonical raw bytes from these executable units:

- standalone `.sh` and `.bash` files and files with a `bash`, `sh`, or `dash` shebang;
- `bash`, `sh`, `shell-script`, and `console` fenced blocks in `README` Markdown files and
  `SKILL.md`; and
- statically literal `bash`/`sh`/`dash` `-c` or `-lc` programs and one-literal-operand `eval`
  programs reached from a supported unit.

Parsing uses `tree-sitter==0.25.2` with `tree-sitter-bash==0.25.1`. Each unit is parsed from
immutable bytes through a bounded callable reader. The syntax frontend extracts commands from
lists, conditionals, loops, `case` statements, groups, subshells, pipelines, functions, and
command/process substitutions. It models only bounded literal assignments, exports, command-local
prefix assignments, and conservative control-flow joins. Tree-sitter supplies syntax, not
execution semantics: SC10 does not run shell, expand the host environment, or fall back to a regex,
raw-text, `shlex`, or custom shell parser.

Fixed command adapters recognize these dependency-source sinks:

| Ecosystem | Supported executable forms |
|---|---|
| npm, Yarn, pnpm | Registry/scoped-registry configuration changes and per-invocation registry options |
| pip | `config` index changes and per-invocation index options, including versioned `pip` and `python -m pip` |
| Poetry | Source add and repository configuration forms |
| Cargo | Literal `--config registries.<name>.index=...` overrides |
| uv | Index add, per-invocation index options, and named index values |
| Maven | One literal, unique, bundle-local `-s`/`--settings` reference, parsed with the direct Maven settings parser |

Path-qualified manager names and fixed transparent forms of `env`, `sudo`, `command`, `exec`,
`nohup`, `nice`, `timeout`, `setsid`, and `stdbuf` are supported. `corepack` and `npx` are supported
only when they name a literal recognized downstream manager. Named source environment variables for
the ecosystems above are findings only when an `export` or command-local assignment is proven to
reach the matching manager; a plain persistent shell assignment is state, not an environment
observation.

## Generated configuration

A completed heredoc or here-string can be parsed as generated direct configuration only when its
effective standard input and final standard-output write are structurally proven. Supported writes
use `>`, `>|`, or `>>` to a recognized direct-configuration path, or the fixed literal-output
`tee` form. Unquoted heredoc expansion is limited to modeled literal bindings and preserves a map
back to physical script bytes. A literal Maven settings reference may resolve one uniquely named
bundle-local XML file even when that file has a nonstandard basename.

Dynamic structure, an ambiguous target or wrapper, unsupported option arity, function shadowing,
malformed syntax, data piped into a shell, `xargs`, command-wrapper `env -S`, a heredoc piped to a
downstream writer, unsupported file-descriptor behavior, or an unproven generated-file write
produces a localized `dependency_source_parse_incomplete` limitation rather than a guessed finding
or clean result.
Dockerfile `RUN`, Make recipes, `.zsh`, `.ksh`, `.envrc`, unsupported shebangs, executable-only files
without a supported dialect, indented Markdown code, and shell fences outside `README`/`SKILL.md`
remain explicit `unscanned_executable_content` coverage limitations when recognized.

Shell analysis shares the direct-parser and output budgets and adds these ceilings:

| Resource | Limit |
|---|---:|
| Shell units per file | 256 |
| Parser calls per file | 512 |
| Parsed-byte revisits per file | 2 times the file size |
| Parsed shell bytes across the scan | 6,000,000 |
| CST visits per unit | 12 times unit bytes plus 1,024 |
| Nested literal-program depth | 2 |
| Retained shell IR across the scan | 50,000 |
| Source-map entries per file | 50,000 |
| Retained shell value bytes per file | 2,000,000 |
| Localized shell issues across the scan | 10,000 |

Parser cancellation, parser unavailability, or resource exhaustion is reported as localized
partial or failed work. It never silently converts an applicable unit into complete coverage. The
parser packages are required runtime dependencies; installation therefore requires compatible
distributions for the target Python and platform, and an ABI or semantic-version mismatch is
treated as parser unavailability.

## Security and product boundary

SC10 does not execute project content, commands, package managers, or generated files. It makes no
network, DNS, or reputation requests; maintains no user-managed allow/block/trust lists; and adds
no telemetry, service, or worker. It is not a general shell interpreter. Optional provider analysis
may add presentation context, but it cannot suppress, downgrade, rewrite, or remove deterministic
SC10 evidence.

The result remains advisory. A `HIGH` finding or incomplete-coverage notice is evidence for human
review.
