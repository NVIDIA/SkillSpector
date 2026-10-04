# Dependency Source Redirection

SkillSpector reports deterministic HIGH SC10 findings when skill content adds or replaces a
package-manager source, or when the destination cannot be resolved from simple local assignments.
This makes the dependency trust-boundary change explicit without making a reputation judgment
about the destination.

## Supported ecosystems and surfaces

| Ecosystem | Direct configuration | Commands and environment | Generated configuration |
|---|---|---|---|
| npm | `.npmrc` registry and scoped registry | `npm config set`, `NPM_CONFIG_REGISTRY` | `.npmrc` heredoc |
| Yarn | `.yarnrc`, `.yarnrc.yml` | `yarn config set` | Yarn config heredoc |
| pip | `pip.conf`, `pip.ini` | index flags, `pip config set`, `PIP_INDEX_URL`, `PIP_EXTRA_INDEX_URL` | pip config heredoc |
| Poetry | `pyproject.toml` sources | `poetry source add`, repository config | `pyproject.toml` heredoc |
| Maven | `settings.xml`, `pom.xml` repositories and mirrors | Maven CLI repository override | Maven XML heredoc |
| Cargo | `.cargo/config`, `.cargo/config.toml` sources and registries | Cargo registry-index environment variables | Cargo config heredoc |

Commands in executable scripts and shell-language Markdown fences are actionable scan surfaces.
Explanatory prose, comments, and non-shell fences do not create SC10 findings.

## Evidence

Each finding records the ecosystem, add/replace operation, configuration surface, scope,
destination, and whether that destination was resolved. Simple literal variables defined in the
same file are resolved without evaluating shell code. Dynamic destinations are reported as
`unresolved` rather than ignored.

Credentials and sensitive query values embedded in URLs are redacted from findings and every
report format. The analyzer never logs credentials, executes configuration, or contacts the
destination.

## Trust model

Canonical public defaults are built into the analyzer solely to avoid reporting an unchanged
default as a redirection. Every other resolved destination is reported uniformly: SkillSpector
does not maintain an organization allowlist, infer whether a host is public or private, perform
DNS resolution, or make network/reputation calls.

SC10 remains HIGH through optional LLM meta-analysis. An explicit, user-selected baseline retains
its existing ability to suppress reviewed findings.
