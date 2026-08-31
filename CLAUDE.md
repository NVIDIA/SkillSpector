# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SkillSpector is a security scanner for AI agent skills (Claude Code, Cursor, etc.). It statically analyzes a
skill directory/zip/URL for prompt injection, data exfiltration, dangerous code, and other risk patterns, then
optionally runs an LLM semantic pass, and produces a risk score + SARIF/JSON/Markdown/terminal report. It never
executes the scanned skill — all analysis is static (regex, Python AST, YARA) plus optional LLM evaluation of
file contents.

The engine is a **LangGraph** workflow (`from skillspector import graph`). See `docs/DEVELOPMENT.md` for the
full architecture writeup — the summary below is enough to be productive; read that doc before adding an
analyzer or provider.

## Commands

All `make` targets assume a venv is already created and activated (Makefile picks `uv` if available, else `pip`).

```bash
uv venv .venv && source .venv/bin/activate
make install-dev          # editable install + dev/mcp/langgraph-dev extras

make test-unit             # unit tests only, no LLM calls (what you want most of the time)
make test-integration       # invokes the full graph, may call configured LLM providers
make test                   # test-unit + test-integration
make test-cov               # test-unit with HTML+terminal coverage report
make test-provider [openai|anthropic|nv_build]   # live provider tests, needs real API keys

pytest tests/unit/test_patterns.py -k some_test   # run a single test
pytest -m "not integration and not provider" tests/some_dir/  # scope to a subtree, respecting default markers

make lint                   # ruff check src/ tests/
make lint-fix                # ruff check --fix
make format                  # ruff check --fix + ruff format
make format-check            # ruff format --check (no changes)

skillspector scan ./my-skill/ --no-llm     # run the CLI directly against a skill under test
make langgraph-dev                          # LangGraph Studio dev server, for visually inspecting/running the graph
```

Tests are marked `integration` (full-graph, may call LLMs) and `provider` (live provider endpoint tests); the
default `pytest` addopts excludes both, so `make test-unit` == plain `pytest tests/` in practice.

## Architecture

**Data flow:** `resolve_input` (URL/zip/file/dir → local `skill_path`) → `build_context` (reads files into
`components`/`file_cache`/`ast_cache`/`manifest`) → ~22 analyzer nodes run in parallel (fan-out) → `meta_analyzer`
(fan-in; per-file LLM filter/enrich of findings when `use_llm`) → `report` (baseline suppression, SARIF build,
risk scoring, `report_body` formatting) → END.

```
resolve_input → build_context → [analyzers: static_* / behavioral_* / mcp_* / semantic_*] → meta_analyzer → report
```

- **State**: `SkillspectorState` (`state.py`, `TypedDict, total=False`) is threaded through every node. Findings
  accumulate via an `operator.add` reducer on `state["findings"]`. See the field table in
  `docs/DEVELOPMENT.md` §4 before touching state shape.
- **Findings**: analyzers emit `AnalyzerFinding` (`Location` + `Severity` enum), converted to the canonical
  `Finding` (`models.py`) via `static_runner.analyzer_finding_to_finding`. `Finding.to_dict()` is the schema
  boundary for JSON/SARIF output — treat it as the contract when changing finding shape.
- **Analyzers** (`nodes/analyzers/`): registered in `ANALYZER_NODE_IDS` / `ANALYZER_NODES`
  (`nodes/analyzers/__init__.py`); `graph.py` wires `build_context → each analyzer → meta_analyzer` in a loop, so
  adding a node needs **no** `graph.py` change. Categories: `static_patterns_*` (regex, one `analyze(content,
  file_path, file_type) -> list[AnalyzerFinding]` per module, built on `static_runner.run_static_patterns` +
  `pattern_defaults` for category/remediation), `static_yara.py` (YARA), `behavioral_ast.py` (AST1-9: exec/eval/
  subprocess/os.system/compile/dynamic-import/getattr), `behavioral_taint_tracking.py` (TT1-5: source→sink
  dataflow over Python AST), `mcp_least_privilege.py` / `mcp_tool_poisoning.py` / `mcp_rug_pull.py`, and
  `semantic_*.py` (LLM-only; return `{"findings": []}` when `use_llm` is False — this is also the pattern for a
  not-yet-implemented placeholder analyzer).
- **LLM plumbing**: `llm_utils.get_chat_model()` / `chat_completion()` dispatch on `SKILLSPECTOR_PROVIDER`.
  `providers/<name>/` is one subpackage per provider (own `provider.py` + bundled `model_registry.yaml`);
  `providers/registry.py` exposes context-length/max-output lookups. CLI providers (`claude_cli`, `codex_cli`)
  implement `AgentCLICapable` and shell out through the hardened `providers/_agent_cli.py` (no shell, stdin-only
  untrusted content, env scrubbed of API keys, tools/MCP disabled, per-call timeout). `nodes/llm_analyzer_base.py`
  (`LLMAnalyzerBase`, `LLMMetaAnalyzer`) provides shared per-file/per-chunk token-budget-aware batching used by
  `meta_analyzer` and the semantic analyzers.
- **Suppression** (`suppression.py`): baseline findings can be accepted via exact fingerprint or drift-tolerant
  glob rule; `report.py` partitions them out before scoring. See `docs/SUPPRESSION.md`.
- **Entry points**: CLI (`cli.py`, Typer app, `skillspector scan`), programmatic (`from skillspector import
  graph; graph.invoke({...})`), LangGraph Studio (`make langgraph-dev`, graph declared in `langgraph.json`), and
  an MCP server (`mcp_server.py`, `skillspector mcp`, requires the `mcp` extra) exposing a single `scan_skill`
  tool for gating installs at runtime.

## Adding an analyzer

1. Implement a node: input `state: SkillspectorState`, output `AnalyzerNodeResponse` (`{"findings":
   list[Finding]}`).
2. For a regex-pattern analyzer, write `analyze(content, file_path, file_type) -> list[AnalyzerFinding]` and run
   it through `static_runner.run_static_patterns`, sourcing category/remediation text from `pattern_defaults`.
3. Register the node id/callable in `nodes/analyzers/__init__.py`'s `ANALYZER_NODE_IDS` / `ANALYZER_NODES` —
   `graph.py` wires the edges automatically.
4. Add unit tests (and fixtures under `tests/fixtures/` if needed); see `tests/nodes/analyzers/` for the existing
   pattern.

## Conventions

- New source files need the SPDX license header (copy from any existing `.py` file).
- Commits require DCO sign-off (`git commit -s`).
- Ruff is the only linter/formatter in CI (line-length 100, `target-version py312`); mypy config exists in
  `pyproject.toml` but is not currently run in CI.
- Env config for local runs lives in `.env` (copy `.env.example`); `SKILLSPECTOR_PROVIDER` selects the LLM
  provider (default `nv_build`), credential var depends on provider (see README's provider table).
