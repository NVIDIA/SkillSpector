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
- `provider`: optional `openai`, `anthropic`, `anthropic_proxy`, `nv_build`, `nv_inference`, or `gemini`.
- `model`: optional model override.
- `yaraRulesDir`: optional directory of extra YARA rules.
- `verbose`: optional detailed progress.

Inputs inside the session's working directory run without a prompt. Remote targets
and external paths require confirmation; redirected aliases show their resolved
destination too. Print and JSON sessions reject these requests because they cannot
show a dialog. Use TUI or RPC mode, move the skill into the working directory, or
run the CLI directly. Local targets retain the CLI's refusal of symlinked paths.
Missing rule directories fail before scanning, and YARA `include` directives are
disabled: put self-contained rule files in the selected directory.

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
