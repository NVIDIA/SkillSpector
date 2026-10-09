# NVIDIA Skill Inspector — Security Inspection Plugin

**100% local, fully offline. No cloud services, no cloud databases, no external API calls, no telemetry. Air-gapped ready.**

Deterministic static analysis for AI agent skills. All data stays on the user's machine under `~/.skill-inspector/` (SQLite, SBOMs, reports).

## Features

| Feature | Tech |
|---|---|
| **Skill Dependency Graph** | `networkx` DiGraph → Cytoscape interactive viz, cycle detection |
| **Permission/Capability Manifest** | Declarative `skill.json` validation + AST/regex scan vs actual `open`/`requests`/`subprocess` usage |
| **Reputation & Provenance** | SHA256 skill hashing, author/origin/version history in SQLite, git config origin |
| **Secrets & Credential-Flow Analyzer** | 16+ regex patterns + Shannon entropy (≥4.2) for hardcoded keys, private keys, tokens; logs-secret detection |
| **Skill Diff Security** | Git plumbing (`git diff --no-color`) with `difflib` fallback; flags new network/subprocess/permission escalations |
| **SBOM for Agent Skills** | CycloneDX 1.5 JSON (per-skill + aggregate), hashes, purls |
| **Security Scorecard** | 0-100 deterministic scoring weighted by severity × category, grade A-F, explainable deductions |
| **Privacy/Data Classification** | PII/Financial/Health/Location/Biometric/Credentials detection + read/write/transmit flow risk |
| **Interactive Security Report** | `FastAPI` + `Jinja2` on `http://127.0.0.1:<port>` only (never 0.0.0.0), SQLite-backed |

All analysis is **deterministic, reproducible, without LLM**.

## Quick Start (Offline)

```bash
pip install -r requirements.txt
# Scan a skills directory (each subfolder is a skill)
python -m skill_inspector_security.cli scan ./tests/sample_skills --no-browser
# Or with auto-serve + browser
python -m skill_inspector_security.cli scan ./tests/sample_skills --port 8899
# Serve latest report
python -m skill_inspector_security.cli serve --port 8899
# Audit offline compliance
python -m skill_inspector_security.cli audit .
```

Data stored at:
- SQLite: `~/.skill-inspector/skill_inspector.db` (env `SKILL_INSPECTOR_DATA_DIR` to override)
- Reports: `~/.skill-inspector/reports/<scan_id>.json`
- SBOMs: `~/.skill-inspector/sbom/<scan_id>-*.cdx.json`

## Permission Manifest Example (`skill.json`)

```json
{
  "name": "my-skill",
  "version": "1.0.0",
  "author": "team@local",
  "permissions": {
    "file_access": "read_only",
    "network": "none",
    "subprocess": false
  },
  "dependencies": ["skill-b"]
}
```

Allowed values: `file_access`: `none|read_only|read_write|unrestricted`, `network`: `none|loopback|outbound|unrestricted`.

## Offline Guarantees

- No `requests` to external hosts (only localhost `127.0.0.1` for report)
- No cloud DB, no telemetry, no LLM calls
- `report_server.py` binds only to `127.0.0.1` (`ALLOWED_BIND`), `ALLOWED_HOSTS` restricted
- `uvicorn` host is hardcoded to `127.0.0.1`
- All hashing/entropy/regex is local CPU-bound

## Project Structure

```
skill_inspector_security/
  config.py, storage.py, models.py
  dependency_graph.py (networkx)
  permission_manifest.py
  provenance.py (hashes + SQLite history)
  secrets_analyzer.py (regex + entropy)
  privacy_classifier.py
  diff_security.py (git plumbing / difflib)
  sbom.py (CycloneDX)
  scorecard.py
  scanner.py (orchestrator)
  report_server.py (FastAPI+Jinja2, loopback only)
  cli.py
templates/report.html
static/css/style.css, static/js/graph.js
```

## Testing

```bash
pip install pytest httpx
pytest tests/ -v
```

## License

MIT — Local use only.
