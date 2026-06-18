# SkillSpector

**Security scanner for AI agent skills.** Detect vulnerabilities, malicious patterns, and security risks before installing agent skills.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

## Overview

AI agent skills (used by Claude Code, Codex CLI, Gemini CLI, etc.) execute with implicit trust and minimal vetting. Research shows that **26.1% of skills contain vulnerabilities** and **5.2% show likely malicious intent**.

SkillSpector helps you answer: **"Is this skill safe to install?"**

## Documentation

- **[Development guide](docs/DEVELOPMENT.md)** — Architecture, package layout, and how to extend the analyzer pipeline.

## Features

- **Multi-format input**: Scan Git repos, URLs, zip files, directories, or single files
- **64 vulnerability patterns** across 16 categories: prompt injection, data exfiltration, privilege escalation, supply chain, excessive agency, output handling, system prompt leakage, memory poisoning, tool misuse, rogue agent, trigger abuse, dangerous code (AST), taint tracking, YARA signatures, MCP least privilege, and MCP tool poisoning
- **Two-stage analysis**: Fast static analysis + optional LLM semantic evaluation
- **Live vulnerability lookups**: SC4 queries [OSV.dev](https://osv.dev) for real-time CVE data with automatic offline fallback
- **Multiple output formats**: Terminal, JSON, Markdown, and SARIF reports
- **Risk scoring**: 0-100 score with severity labels and clear recommendations

## Quick Start

### Installation

Create and activate a virtual environment first (all `make` targets assume the venv is active). Use **uv** or **pip**; the Makefile uses `uv` if available, otherwise `pip`.

```bash
# Clone the repository
git clone https://github.com/NVIDIA/skillspector.git
cd skillspector

# Create and activate virtual environment
uv venv .venv && source .venv/bin/activate
# or: python3 -m venv .venv && source .venv/bin/activate

# Install for production use
make install

# Or install with development dependencies
make install-dev
```

### Docker (no Python required)

Run SkillSpector without installing Python by building it locally from the included [Dockerfile](Dockerfile). The image is based on the Docker Official Python `3.12-slim-bookworm` image.

**Build the image:**

```bash
make docker-build
# or: docker build -t skillspector .
```

**Scan a local directory** by mounting your current directory into `/scan`, the container's working directory:

```bash
docker run --rm -v "$PWD:/scan" skillspector scan ./my-skill/ --no-llm
```

**Scan with LLM analysis** by passing credentials with a local `.env` file:

```bash
cat > .env <<'EOF'
SKILLSPECTOR_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
EOF
```

```bash
docker run --rm \
  -v "$PWD:/scan" \
  --env-file .env \
  skillspector scan ./my-skill/
```

Or pass credentials directly from your shell environment:

```bash
docker run --rm \
  -v "$PWD:/scan" \
  -e SKILLSPECTOR_PROVIDER=anthropic \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  skillspector scan ./my-skill/
```

**Write a report to the host filesystem** by writing to the mounted directory:

```bash
docker run --rm \
  -v "$PWD:/scan" \
  skillspector scan ./my-skill/ --no-llm --format json --output report.json
```

**Optional alias** for repeated static scans:

```bash
alias skillspector-docker='docker run --rm -v "$PWD:/scan" skillspector'
skillspector-docker scan ./my-skill/ --no-llm
```

### Basic Usage

```bash
# Scan a local skill directory
skillspector scan ./my-skill/

# Scan a single SKILL.md file
skillspector scan ./SKILL.md

# Scan a Git repository
skillspector scan https://github.com/user/my-skill

# Scan a zip file
skillspector scan ./my-skill.zip
```

### Output Formats

```bash
# Terminal output (default) - pretty formatted
skillspector scan ./my-skill/

# JSON output - machine readable
skillspector scan ./my-skill/ --format json --output report.json

# Markdown output - for documentation
skillspector scan ./my-skill/ --format markdown --output report.md

# SARIF output - for CI/CD integration and IDE tooling
skillspector scan ./my-skill/ --format sarif --output report.sarif
```

### LLM Analysis

For the best results, configure an OpenAI-compatible LLM endpoint for
semantic analysis. Pick a provider with `SKILLSPECTOR_PROVIDER`; each
ships its own bundled default model. SkillSpector also works against
local OpenAI-compatible servers (Ollama, vLLM, llama.cpp) and managed
inference gateways.

| Provider (`SKILLSPECTOR_PROVIDER`) | Credential env var | Endpoint | Default model |
| ---------- | ---- | ---- | ---- |
| `openai` | `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`) | api.openai.com (or any OpenAI-compatible URL) | `gpt-5.4` |
| `anthropic` | `ANTHROPIC_API_KEY` | api.anthropic.com | `claude-opus-4-6` |
| `nv_build` | `NVIDIA_INFERENCE_KEY` | build.nvidia.com | `deepseek-ai/deepseek-v4-flash` |

```bash
# Stock OpenAI
export SKILLSPECTOR_PROVIDER=openai
export OPENAI_API_KEY=sk-...
skillspector scan ./my-skill/

# Anthropic
export SKILLSPECTOR_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
skillspector scan ./my-skill/

# NVIDIA build.nvidia.com
export SKILLSPECTOR_PROVIDER=nv_build
export NVIDIA_INFERENCE_KEY=nvapi-...
skillspector scan ./my-skill/

# Local Ollama or any OpenAI-compatible endpoint
export SKILLSPECTOR_PROVIDER=openai
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=http://localhost:11434/v1
export SKILLSPECTOR_MODEL=llama3.1:8b
skillspector scan ./my-skill/

# Override the provider's default model
export SKILLSPECTOR_MODEL=gpt-5.2
skillspector scan ./my-skill/

# Skip LLM analysis (faster, static analysis only)
skillspector scan ./my-skill/ --no-llm
```

## Vulnerability Patterns

SkillSpector detects **64 vulnerability patterns** across 16 categories:

### Prompt Injection (5 patterns)

| ID | Pattern | Severity | Description |
|----|---------|----------|-------------|
| P1 | Instruction Override | HIGH | Commands to ignore safety constraints |
| P2 | Hidden Instructions | HIGH | Malicious directives in comments/invisible text |
| P3 | Exfiltration Commands | HIGH | Instructions to transmit context externally |
| P4 | Behavior Manipulation | MEDIUM | Subtle instructions altering agent decisions |
| P5 | Harmful Content | CRITICAL | Instructions that could cause physical harm |

### Data Exfiltration (4 patterns)

| ID | Pattern | Severity | Description |
|----|---------|----------|-------------|
| E1 | External Transmission | MEDIUM | Sending data to external URLs |
| E2 | Env Variable Harvesting | HIGH | Collecting API keys and secrets |
| E3 | File System Enumeration | MEDIUM | Scanning directories for sensitive files |
| E4 | Context Leakage | HIGH | Transmitting conversation context externally |

### Privilege Escalation (3 patterns)

| ID | Pattern | Severity | Description |
|----|---------|----------|-------------|
| PE1 | Excessive Permissions | LOW | Requesting access beyond stated functionality |
| PE2 | Sudo/Root Execution | MEDIUM | Invoking elevated system privileges |
| PE3 | Credential Access | HIGH | Reading SSH keys, tokens, passwords |

### Supply Chain (6 patterns)

| ID | Pattern | Severity | Description |
|----|---------|----------|-------------|
| SC1 | Unpinned Dependencies | LOW | No version constraints on packages |
| SC2 | External Script Fetching | HIGH | curl \| bash and remote code execution |
| SC3 | Obfuscated Code | HIGH | Base64/hex encoded execution |
| SC4 | Known Vulnerable Dependencies | HIGH | Dependencies with known CVEs (live OSV.dev lookup) |
| SC5 | Abandoned Dependencies | MEDIUM | Unmaintained packages without security updates |
| SC6 | Typosquatting | HIGH | Package names similar to popular packages |

### Excessive Agency (4 patterns)

| ID | Pattern | Severity | Description |
|----|---------|----------|-------------|
| EA1 | Unrestricted Tool Access | HIGH | Unfettered tool access without constraints |
| EA2 | Autonomous Decision Making | HIGH | High-impact decisions without human-in-the-loop |
| EA3 | Scope Creep | MEDIUM | Capabilities extending beyond stated purpose |
| EA4 | Unbounded Resource Access | MEDIUM | No rate limits or quotas on resource consumption |

### Output Handling (3 patterns)

| ID | Pattern | Severity | Description |
|----|---------|----------|-------------|
| OH1 | Unvalidated Output Injection | HIGH | Model output used without sanitization |
| OH2 | Cross-Context Output | MEDIUM | Output flows across trust boundaries without validation |
| OH3 | Unbounded Output | MEDIUM | No limits on output size or generation rate |

### System Prompt Leakage (3 patterns)

| ID | Pattern | Severity | Description |
|----|---------|----------|-------------|
| P6 | Direct Leakage | HIGH | Instructions that expose system prompts or internal rules |
| P7 | Indirect Extraction | MEDIUM | Extraction via rephrasing, translation, or side-channels |
| P8 | Tool-Based Exfiltration | HIGH | System prompts exfiltrated via file writes or network requests |

### Memory Poisoning (3 patterns)

| ID | Pattern | Severity | Description |
|----|---------|----------|-------------|
| MP1 | Persistent Context Injection | HIGH | Content designed to persist across interactions |
| MP2 | Context Window Stuffing | MEDIUM | Filler content displacing safety constraints |
| MP3 | Memory Manipulation | HIGH | Tampering with agent memory or stored state |

### Tool Misuse (3 patterns)

| ID | Pattern | Severity | Description |
|----|---------|----------|-------------|
| TM1 | Tool Parameter Abuse | HIGH | Crafted parameters for unintended behavior (shell=True, --force) |
| TM2 | Chaining Abuse | HIGH | Tool chains that bypass individual safety checks |
| TM3 | Unsafe Defaults | MEDIUM | Overly permissive defaults (disabled TLS, no auth) |

### Rogue Agent (2 patterns)

| ID | Pattern | Severity | Description |
|----|---------|----------|-------------|
| RA1 | Self-Modification | CRITICAL | Modifying own code or configuration at runtime |
| RA2 | Session Persistence | HIGH | Unauthorized persistence via cron jobs or startup scripts |

### Trigger Abuse (3 patterns)

| ID | Pattern | Severity | Description |
|----|---------|----------|-------------|
| TR1 | Overly Broad Trigger | MEDIUM | Trigger patterns matching common words |
| TR2 | Shadow Command Trigger | HIGH | Triggers that shadow built-in commands or other skills |
| TR3 | Keyword Baiting Trigger | MEDIUM | Generic triggers designed to maximize activation |

### Behavioral AST (8 patterns)

| ID | Pattern | Severity | Description |
|----|---------|----------|-------------|
| AST1 | exec() Call | CRITICAL | Direct exec() enabling arbitrary code execution |
| AST2 | eval() Call | HIGH | Direct eval() evaluating arbitrary expressions |
| AST3 | Dynamic Import | HIGH | \_\_import\_\_() loading arbitrary modules at runtime |
| AST4 | subprocess Call | HIGH | External command execution via subprocess |
| AST5 | os.system / exec-family | HIGH | Shell commands via os module |
| AST6 | compile() Call | MEDIUM | Code object creation from strings |
| AST7 | Dynamic getattr() | MEDIUM | Arbitrary attribute access with non-literal names |
| AST8 | Dangerous Execution Chain | CRITICAL | exec/eval combined with dynamic source (network, encoded data) |

### Taint Tracking (5 patterns)

| ID | Pattern | Severity | Description |
|----|---------|----------|-------------|
| TT1 | Direct Taint Flow | HIGH | Data flows directly from a source to a sink without sanitization |
| TT2 | Variable-Mediated Taint Flow | MEDIUM | Data flows from source to sink through intermediate variables |
| TT3 | Credential Exfiltration Chain | CRITICAL | Credentials (env vars, secrets) flow to network output sinks |
| TT4 | File Read to Network Exfiltration | HIGH | File contents flow to network output sinks |
| TT5 | External Input to Code Execution | CRITICAL | Network or user input flows to exec/eval/subprocess sinks |

### YARA Signatures (4 patterns)

| ID | Pattern | Severity | Description |
|----|---------|----------|-------------|
| YR1 | Malware Match | CRITICAL | YARA rule match for known malware signatures |
| YR2 | Webshell Match | CRITICAL | YARA rule match for webshell patterns |
| YR3 | Cryptominer Match | HIGH | YARA rule match for crypto mining indicators |
| YR4 | Hack Tool / Exploit Match | HIGH | YARA rule match for hack tools or exploit code |

### MCP Least Privilege (4 patterns)

| ID | Pattern | Severity | Description |
|----|---------|----------|-------------|
| LP1 | Underdeclared Capability | HIGH | Code uses capabilities not listed in declared permissions |
| LP2 | Wildcard Permission | MEDIUM | Permission list contains wildcards (\*, all, full, any) |
| LP3 | Missing Permission Declaration | MEDIUM | No permissions field but code has detectable capabilities |
| LP4 | Overdeclared Permission | LOW | Permission declared but no corresponding code capability found |

### MCP Tool Poisoning (4 patterns)

| ID | Pattern | Severity | Description |
|----|---------|----------|-------------|
| TP1 | Hidden Instructions | HIGH | Hidden directives in metadata (HTML comments, zero-width chars, base64, data URIs) |
| TP2 | Unicode Deception | HIGH | Homoglyphs, RTL overrides, mixed-script identifiers in tool metadata |
| TP3 | Parameter Description Injection | MEDIUM | Injection patterns in parameter definitions (overrides, system tokens, malicious defaults) |
| TP4 | Description-Behavior Mismatch | MEDIUM | Declared tool description does not match actual code behavior (LLM-powered) |

All detected patterns are listed in the tables above.

## Risk Scoring

### Score Calculation

- **CRITICAL issues**: +50 points
- **HIGH issues**: +25 points
- **MEDIUM issues**: +10 points
- **LOW issues**: +5 points
- **Executable scripts**: 1.3x multiplier

### Severity Levels

| Score | Severity | Recommendation |
|-------|----------|----------------|
| 0-20 | LOW | SAFE |
| 21-50 | MEDIUM | CAUTION |
| 51-80 | HIGH | DO NOT INSTALL |
| 81-100 | CRITICAL | DO NOT INSTALL |

## Example Output

### Terminal Output

```
 SkillSpector Security Report  v2.0.0

Skill: suspicious-skill
Source: ./suspicious-skill/
Scanned: 2026-01-29 10:30:00 UTC

        Risk Assessment
 Metric          Value
 Score           78/100
 Severity        HIGH
 Recommendation  DO NOT INSTALL

        Components (3)
 File              Type      Lines  Executable
 SKILL.md          markdown    142  No
 scripts/sync.py   python       87  Yes
 requirements.txt  text          3  No

Issues (2)

  HIGH: Env Variable Harvesting (E2)
    Location: scripts/sync.py:23
    Finding: for key, val in os.environ.items():...
    Confidence: 94%
    Explanation: This code collects environment variables containing
    API keys and secrets, then sends them to an external server.

  HIGH: External Transmission (E1)
    Location: scripts/sync.py:45
    Finding: requests.post("https://api.skill.io/env"...
    Confidence: 89%
    Explanation: Data is being sent to an external server. Combined
    with env harvesting above, this indicates credential exfiltration.
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SKILLSPECTOR_PROVIDER` | Active LLM provider: `openai`, `anthropic`, or `nv_build`. Each provider has its own bundled `model_registry.yaml` and default model (see the LLM Analysis table above). Defaults to `nv_build`. | Optional |
| `NVIDIA_INFERENCE_KEY` | Credential for the `nv_build` provider (build.nvidia.com). | Required for LLM analysis when `SKILLSPECTOR_PROVIDER=nv_build` |
| `OPENAI_API_KEY` | Credential for the OpenAI provider (`SKILLSPECTOR_PROVIDER=openai`). Also serves as the tier-2 fallback in the credential waterfall when the active provider returns no credentials. | Required for LLM analysis when `SKILLSPECTOR_PROVIDER=openai` |
| `OPENAI_BASE_URL` | Override the OpenAI endpoint (e.g. point at Ollama). | Optional |
| `ANTHROPIC_API_KEY` | Credential for the Anthropic provider (`SKILLSPECTOR_PROVIDER=anthropic`). | Required for LLM analysis when `SKILLSPECTOR_PROVIDER=anthropic` |
| `SKILLSPECTOR_MODEL` | Override the active provider's default model. See the LLM Analysis table for each provider's default. | Optional |
| `SKILLSPECTOR_MODEL_REGISTRY` | Override the bundled per-provider YAML registry (`src/skillspector/providers/<provider>/model_registry.yaml`) with a custom path. | Optional |
| `SKILLSPECTOR_LOG_LEVEL` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `WARNING`). | Optional |

### CLI Options

```bash
skillspector scan --help

Options:
  -f, --format [terminal|json|markdown|sarif]  Output format [default: terminal]
  -o, --output PATH                            Output file path
  --no-llm                                     Skip LLM analysis (static only)
  -V, --verbose                                Show detailed progress
  --help                                       Show this message and exit
```

## Development

### Setup

All `make` targets assume a virtual environment is already created and activated. The Makefile uses **uv** if available, else **pip**.

```bash
# Clone, create venv, activate, install dev dependencies
git clone https://github.com/NVIDIA/skillspector.git
cd skillspector
uv venv .venv && source .venv/bin/activate
# or: python3 -m venv .venv && source .venv/bin/activate
make install-dev

# Run tests
make test

# Run tests with coverage
make test-cov

# Run linting
make lint

# Format code
make format
```

## How It Works

SkillSpector uses a two-stage detection pipeline:

### Stage 1: Static Analysis
- Fast regex-based pattern matching across 11 static analyzers
- AST-based behavioral analysis detecting dangerous calls (exec, eval, subprocess, etc.)
- Live vulnerability lookups via OSV.dev for known CVEs in dependencies
- Scans all files in the skill
- High recall (catches most issues)
- Moderate precision (some false positives)

### Stage 2: LLM Semantic Analysis (Optional)
- Evaluates context and intent
- Filters false positives
- Provides human-readable explanations
- Improves precision to ~87%

The LLM prompt includes anti-jailbreak protections to prevent malicious skills from manipulating the analysis.

## Live Vulnerability Lookups (SC4)

SC4 uses the [OSV.dev](https://osv.dev) API to check dependencies against the full Open Source Vulnerabilities database — covering tens of thousands of advisories across PyPI and npm.

- **No API key required** — OSV.dev is free and unauthenticated.
- **Batch queries** — all dependencies are checked in a single HTTP call.
- **Automatic fallback** — if OSV.dev is unreachable (air-gapped/offline), a small built-in fallback list is used.
- **Caching** — results are cached in-memory for 1 hour to avoid redundant API calls during a session.

The tool requires outbound HTTPS access to `api.osv.dev` for live vulnerability data. When that is not available, findings are limited to the static fallback list.

## Limitations

- **Non-English content**: May miss patterns in other languages
- **Image-based attacks**: Cannot analyze text in images
- **Encrypted/binary code**: Cannot analyze compiled or encrypted content
- **Runtime behavior**: Static analysis only, no dynamic execution
- **Offline SC4**: Without network access to `api.osv.dev`, SC4 uses a small static fallback list

## Research Background

Based on research from "Agent Skills in the Wild: An Empirical Study of Security Vulnerabilities at Scale" (Liu et al., 2026):

- **Dataset**: 42,447 skills from major marketplaces
- **Vulnerable**: 26.1% contain at least one vulnerability
- **High-severity**: 5.2% show likely malicious intent
- **Key finding**: Skills with executable scripts are 2.12x more likely to be vulnerable

## Python API Integration

```python
from skillspector import graph

# Invoke the LangGraph workflow
result = graph.invoke({
    "input_path": "/path/to/skill",
    "output_format": "json",   # terminal, json, markdown, or sarif
    "use_llm": True,           # False for static-only analysis
})

# Access results
print(f"Risk Score: {result['risk_score']}/100")
print(f"Severity: {result['risk_severity']}")
print(f"Recommendation: {result['risk_recommendation']}")

for finding in result["filtered_findings"]:
    print(f"[{finding['severity']}] {finding['rule_id']}: {finding['message']}")
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## Support

- **Issues**: [GitHub Issues](https://github.com/NVIDIA/skillspector/issues)
