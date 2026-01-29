# SkillSpector

**Security scanner for AI agent skills.** Detect vulnerabilities, malicious patterns, and security risks before installing agent skills.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

AI agent skills (used by Claude Code, Codex CLI, Gemini CLI, etc.) execute with implicit trust and minimal vetting. Research shows that **26.1% of skills contain vulnerabilities** and **5.2% show likely malicious intent**.

SkillSpector helps you answer: **"Is this skill safe to install?"**

## Features

- **Multi-format input**: Scan Git repos, URLs, zip files, directories, or single files
- **8 vulnerability patterns**: Detects prompt injection, data exfiltration, privilege escalation, and supply chain attacks
- **Two-stage analysis**: Fast static analysis + optional LLM semantic evaluation
- **Multiple output formats**: Terminal, JSON, and Markdown reports
- **Risk scoring**: 0-100 score with severity labels and clear recommendations

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/skillspector.git
cd skillspector

# Install with pip (recommended)
pip install -e .

# Or install with dependencies for development
pip install -e ".[dev]"
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
```

### LLM Analysis

For the best results, provide an LLM API key for semantic analysis:

```bash
# Using Anthropic Claude
export ANTHROPIC_API_KEY=sk-ant-...
skillspector scan ./my-skill/

# Using Google Gemini
export GOOGLE_API_KEY=...
skillspector scan ./my-skill/

# Skip LLM analysis (faster, static analysis only)
skillspector scan ./my-skill/ --no-llm
```

## Vulnerability Patterns

SkillSpector detects **15 vulnerability patterns** across 4 categories:

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

### Supply Chain (3 patterns)

| ID | Pattern | Severity | Description |
|----|---------|----------|-------------|
| SC1 | Unpinned Dependencies | LOW | No version constraints on packages |
| SC2 | External Script Fetching | HIGH | curl \| bash and remote code execution |
| SC3 | Obfuscated Code | HIGH | Base64/hex encoded execution |

View all patterns:
```bash
skillspector patterns
```

## Risk Scoring

### Score Calculation

- **CRITICAL issues**: +50 points
- **HIGH issues**: +25 points
- **MEDIUM issues**: +10 points
- **LOW issues**: +5 points
- **MEDIUM issues**: +10 points
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
 SkillSpector Security Report  v0.1.0

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
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | One of these |
| `GOOGLE_API_KEY` | Google Gemini API key | required for LLM |
| `SKILLSPECTOR_MODEL` | LLM model to use | Optional |

### CLI Options

```bash
skillspector scan --help

Options:
  -f, --format [terminal|json|markdown]  Output format [default: terminal]
  -o, --output PATH                      Output file path
  --no-llm                               Skip LLM analysis (static only)
  -V, --verbose                          Show detailed progress
  --help                                 Show this message and exit
```

## Development

### Setup

```bash
# Clone and install dev dependencies
git clone https://github.com/your-org/skillspector.git
cd skillspector
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=skillspector

# Run linting
ruff check src/

# Run type checking
mypy src/
```

### Project Structure

```
skillspector/
├── pyproject.toml          # Project configuration
├── README.md               # This file
├── src/
│   └── skillspector/
│       ├── __init__.py     # Package init
│       ├── cli.py          # CLI interface (Typer)
│       ├── scanner.py      # Main orchestration
│       ├── input_handler.py # URL/zip/file handling
│       ├── inventory.py    # Component discovery
│       ├── static_analyzer.py # Regex patterns
│       ├── llm_analyzer.py # LLM integration
│       ├── report.py       # Output formatting
│       ├── models.py       # Data models
│       └── patterns/       # Pattern modules
│           ├── prompt_injection.py
│           ├── data_exfiltration.py
│           ├── privilege_escalation.py
│           ├── supply_chain.py
│           └── harmful_content.py
└── tests/
    ├── test_scanner.py
    ├── test_patterns.py
    └── fixtures/
        ├── safe_skill/
        └── malicious_skill/
```

## How It Works

SkillSpector uses a two-stage detection pipeline:

### Stage 1: Static Analysis
- Fast regex-based pattern matching
- Scans all files in the skill
- High recall (catches most issues)
- Moderate precision (some false positives)

### Stage 2: LLM Semantic Analysis (Optional)
- Evaluates context and intent
- Filters false positives
- Provides human-readable explanations
- Improves precision to ~87%

The LLM prompt includes anti-jailbreak protections to prevent malicious skills from manipulating the analysis.

## Limitations

- **Non-English content**: May miss patterns in other languages
- **Image-based attacks**: Cannot analyze text in images
- **Encrypted/binary code**: Cannot analyze compiled or encrypted content
- **Runtime behavior**: Static analysis only, no dynamic execution

## Research Background

Based on research from "Agent Skills in the Wild: An Empirical Study of Security Vulnerabilities at Scale" (Liu et al., 2026):

- **Dataset**: 42,447 skills from major marketplaces
- **Vulnerable**: 26.1% contain at least one vulnerability
- **High-severity**: 5.2% show likely malicious intent
- **Key finding**: Skills with executable scripts are 2.12x more likely to be vulnerable

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## Support

- **Issues**: [GitLab Issues](https://gitlab-master.nvidia.com/prototypes/skillspector/-/issues)

---

## Author

**Nir Paz** @ NVIDIA
