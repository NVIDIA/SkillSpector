# SkillSpector Pi Extension

SkillSpector can be installed into Pi as a local package. The extension registers a `skillspector_scan` tool that runs the existing SkillSpector CLI.

## Requirements

- Pi installed.
- Python `>=3.12,<3.15`.
- `uv` recommended.
- This repo checked out locally.

## Install

```bash
cd /path/to/SkillSpector
uv sync
pi install /path/to/SkillSpector
```

Then reload Pi:

```text
/reload
```

## Basic scan

Ask Pi:

```text
Use skillspector_scan on tests/fixtures/safe_skill/SKILL.md with noLlm=true.
```

Equivalent CLI:

```bash
.venv/bin/skillspector scan tests/fixtures/safe_skill/SKILL.md --no-llm
```

## Tool parameters

- `target`: path, URL, zip, Git repo, or `SKILL.md` to scan.
- `format`: `terminal`, `json`, `markdown`, or `sarif`. Default: `terminal`.
- `output`: optional report path.
- `noLlm`: default `true`.
- `provider`: optional `openai`, `anthropic`, `anthropic_proxy`, or `nv_build`.
- `model`: optional model override.
- `yaraRulesDir`: optional directory of extra YARA rules.
- `verbose`: optional detailed progress.

## LLM-backed analysis

Static scan is default. To use semantic LLM analysis, configure provider credentials in your shell before launching Pi, then call the tool with `noLlm=false` and a provider.

Static scans have a 120-second process limit. Explicit LLM scans have a
630-second limit, allowing the CLI's default 600-second workflow budget plus
startup and report writing. This tool limit stays fixed even if
`SKILLSPECTOR_MAX_WORKFLOW_SECONDS` is configured above 600.

Example:

```text
Use skillspector_scan on ./my-skill with noLlm=false and provider=anthropic.
```

The extension does not read `.env` and redacts secret-looking output.

## Remove

```bash
pi remove /path/to/SkillSpector
```
