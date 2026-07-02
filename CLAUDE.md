# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SkillSpector is a security scanner for AI agent skills (Claude Code, Codex CLI, Gemini CLI). It detects vulnerabilities, malicious patterns, and security risks using a LangGraph-based analysis pipeline. Developed by NVIDIA, Apache 2.0 license.

## Commands

All commands assume a Python 3.12+ venv is created and activated (`uv venv .venv && source .venv/bin/activate`). The Makefile prefers `uv` if available, otherwise falls back to `pip`.

```bash
make install-dev     # Install with dev dependencies
make test            # Unit + integration tests
make test-unit       # Unit tests only (no LLM calls)
make test-cov        # Unit tests with HTML coverage
make lint            # ruff check src/ tests/
make format          # ruff check --fix + ruff format
make format-check    # ruff format --check (used in CI)
make build           # Build wheel/sdist
make clean           # Remove build artifacts, caches, __pycache__
```

**Running the CLI:**
```bash
skillspector scan ./my-skill/                  # terminal output
skillspector scan ./my-skill/ --no-llm         # static analysis only
skillspector scan ./my-skill/ --format json -o report.json
skillspector scan https://github.com/user/repo # Git URL input
```

**Running a single test:**
```bash
pytest tests/unit/test_cli.py::test_specific_name
pytest tests/integration/test_graph.py -k "test_name"
```

Pytest markers: `integration` (full graph, may call LLMs), `provider` (live API endpoint tests). Default pytest config excludes both markers.

## Architecture

### LangGraph pipeline (graph.py)

```
START → resolve_input → build_context → [22 analyzers in parallel] → meta_analyzer → report → END
```

- **State**: `SkillspectorState` (TypedDict in `state.py`). Key fields: `input_path`, `skill_path`, `file_cache`, `findings` (reducer: `operator.add`), `filtered_findings`, `risk_score`, `report_body`, `sarif_report`.
- **No conditional edges** — the graph is a straight pipeline with a fan-out/fan-in at the analyzer layer.

### Nodes

| Node | Source | Role |
|------|--------|------|
| `resolve_input` | `nodes/resolve_input.py` | Normalizes URL/zip/file/dir input → `skill_path` |
| `build_context` | `nodes/build_context.py` | Walks skill dir, populates `file_cache`, `manifest`, `component_metadata` |
| Analyzers (22) | `nodes/analyzers/` | Each returns `{"findings": list[Finding]}`; state reducer appends |
| `meta_analyzer` | `nodes/meta_analyzer.py` | Per-file LLM filtering/enrichment → `filtered_findings` |
| `report` | `nodes/report.py` | SARIF 2.1.0 + risk score (0-100) + formatted output |

### Analyzer categories

- **14 static pattern analyzers** — regex-based detection via `static_runner.py`. Pattern modules must export `analyze(content, file_path, file_type) -> list[AnalyzerFinding]`.
- **1 static YARA analyzer** — YARA signature scanning.
- **2 behavioral analyzers** — Python AST analysis (`behavioral_ast.py`) and taint tracking (`behavioral_taint_tracking.py`).
- **3 MCP analyzers** — MCP least privilege, tool poisoning, rug pull detection.
- **3 semantic (LLM) analyzers** — Security discovery, developer intent, quality policy. Return `[]` when `use_llm` is False.

### Adding an analyzer

1. Implement a node returning `{"findings": list[Finding]}`.
2. Register in `nodes/analyzers/__init__.py` (`ANALYZER_NODE_IDS` + `ANALYZER_NODES`).
3. No changes needed in `graph.py` — edges are created in a loop over `ANALYZER_NODE_IDS`.

### Key data types

- `Finding` (`models.py`): `rule_id`, `message`, `severity`, `confidence`, `file`, `start_line`, `end_line`, `category`, `pattern`, `finding`, `explanation`, `remediation`, `code_snippet`, `tags`.
- `AnalyzerFinding`: Analyzer-internal type with `Location` and `Severity` enum; converted to `Finding` via `static_runner.analyzer_finding_to_finding()`.
- SARIF models (`sarif_models.py`): Pydantic models for SARIF 2.1.0.

### LLM providers

Pluggable providers in `src/skillspector/providers/`. Selected via `SKILLSPECTOR_PROVIDER` env var (default: `nv_build`):
- **HTTP providers**: `nv_build`, `openai`, `anthropic`, `bedrock`, `anthropic_proxy`
- **CLI providers** (no API key needed): `claude_cli`, `codex_cli`, `gemini_cli` — use local agent binaries with sandboxed subprocess calls

### Logging

Internal logging uses stdlib `logging` via `get_logger(__name__)` from `logging_config.py`. User-facing output uses Rich `console.print()`. Set `SKILLSPECTOR_LOG_LEVEL` (default: `WARNING`) or use `--verbose`/`-V` on the CLI.

## Key files

| File | Purpose |
|------|---------|
| `src/skillspector/cli.py` | Typer CLI entry point (`scan`, `mcp`, `baseline` commands) |
| `src/skillspector/graph.py` | LangGraph workflow definition |
| `src/skillspector/state.py` | State schema |
| `src/skillspector/models.py` | Data models |
| `src/skillspector/constants.py` | Model config, token budgets, env resolution |
| `src/skillspector/input_handler.py` | Input resolution (Git clone, download, zip, etc.) |
| `src/skillspector/suppression.py` | Baseline/false-positive suppression |
| `src/skillspector/nodes/analyzers/__init__.py` | Analyzer registry |
| `src/skillspector/nodes/analyzers/pattern_defaults.py` | Pattern ID → category/explanation/remediation mapping |
| `pyproject.toml` | Dependencies, ruff/mypy/pytest config |
| `docs/DEVELOPMENT.md` | Detailed architecture and extension guide |
