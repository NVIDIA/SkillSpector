# SkillSpector OpenCode Extension

SkillSpector can be installed into OpenCode as a local extension. The extension registers a `skillspector_scan` tool and a `/skillspector` slash command that run the existing SkillSpector CLI.

## Requirements

- OpenCode installed.
- Python `>=3.12,<3.15`.
- `uv` recommended.
- This repo checked out locally.
- Node 22+ to run the extension unit tests (type stripping, no extra dependencies).

## Install

Copy this repo's `.opencode/` directory into your project (or `~/.config/opencode/` for global use):

```bash
cp -r /path/to/SkillSpector/.opencode /path/to/my-project/
```

Make sure `skillspector` is on PATH, or point `SKILLSPECTOR_BIN` at the binary:

```bash
export SKILLSPECTOR_BIN=/path/to/SkillSpector/.venv/bin/skillspector
```

Then reload OpenCode or start a new session; `/skillspector` is auto-discovered.

## Basic scan

In OpenCode:

```text
/skillspector ./my-skill
```

Equivalent CLI (static analysis only):

```bash
skillspector scan ./my-skill --no-llm
```

Before starting the CLI, the tool asks OpenCode for the capabilities used by
that invocation: target reads (and remote fetches), report writes, external
paths, and the CLI subprocess. A denied request stops the invocation before
the subprocess starts.

## Tool parameters

- `target`: path, URL, zip, Git repo, or `SKILL.md` to scan.
- `format`: `terminal`, `json`, `markdown`, or `sarif`. Default: `json`.
- `output`: optional report path.
- `noLlm`: default `true`.

Unlike the [Pi extension](PI_EXTENSION.md), this tool has no `provider`, `model`, `yaraRulesDir`, or `verbose` parameters: LLM-backed analysis is configured through the environment instead (see below).

## LLM-backed analysis

Static scan is default. To use semantic LLM analysis, configure a supported
provider before launching OpenCode, then call the tool with `noLlm` false. The
tool makes a separate permission request naming the provider, model, and
credential-free destination before analyzer-eligible skill content can leave
the host:

```text
Use skillspector_scan on ./my-skill with noLlm=false.
```

```bash
export SKILLSPECTOR_PROVIDER=nv_build
export NVIDIA_INFERENCE_KEY=nvapi-...
# Optional; omit to use nv_build's bundled default model.
# export SKILLSPECTOR_MODEL=z-ai/glm-5.2
```

Other valid providers and their credential variables are listed in the main
[LLM Analysis](../README.md#llm-analysis) table. The extension passes the
environment to the existing SkillSpector CLI, but never puts credentials in a
permission request. Model-visible output is bounded and redacts the supported
provider credential values and names.

## Unit tests

Pure tool helpers live in dependency-free `.opencode/tools/skillspector_scan_lib.ts`, covered by `tests/opencode/skillspector_scan_lib.test.ts` via stdlib `node --test` (zero new dependencies):

```bash
node --test tests/opencode/skillspector_scan_lib.test.ts
```

## Remove

Delete the copied `.opencode/tools/skillspector_scan.*` and `.opencode/commands/skillspector.md` files.
