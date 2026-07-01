# Subprocess Provider — Acceptance Test Plan

**Feature:** `SKILLSPECTOR_PROVIDER=subprocess` — routes LLM prompts through a
configurable shell command, enabling SkillSpector to run inside Claude Code,
OpenClaw, Antigravity, or any other AI-tool session without a separate API key.

**Scope:** These tests must be executed **outside** the development session that
built this feature — in a fresh shell where no prior environment is inherited.
They cover the full user-visible surface: CLI, env vars, error messages, and
scan quality.

**Prerequisites:**
- SkillSpector installed: `uv pip install -e .` (or the packaged wheel)
- At least one AI-tool CLI available: `claude`, `antigravity`, or `openclaw`
- `SKILLSPECTOR_PROVIDER` and any prior provider credentials **cleared** from
  environment before each test group

---

## Test Group 1 — Happy Path: scan with subprocess provider

### AT-01 — Basic scan with `claude -p`

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "subprocess"
$env:SKILLSPECTOR_LLM_COMMAND = "claude -p"
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:NVIDIA_INFERENCE_KEY -ErrorAction SilentlyContinue
```

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --format terminal
```

**Expected:**
- Exit code 1 (non-zero; malicious skill scores > 50)
- Report printed to terminal
- At least one finding with severity HIGH or CRITICAL
- No error mentioning "API key", "OPENAI", or "NVIDIA"
- LLM meta-analyzer runs (output does NOT say "LLM analysis skipped")

---

### AT-02 — Scan a safe skill produces low/no risk score

**Setup:** Same as AT-01.

**Steps:**
```powershell
skillspector scan tests/fixtures/safe_skill --format terminal
```

**Expected:**
- Exit code 0
- Risk score 0–20 / severity LOW or SAFE
- No false positives elevated to HIGH or CRITICAL by meta-analyzer

---

### AT-03 — JSON output format

**Setup:** Same as AT-01.

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --format json --output report.json
Get-Content report.json | python -m json.tool | Select-Object -First 5
```

**Expected:**
- `report.json` created
- Valid JSON (python json.tool exits 0)
- Top-level keys include `issues` (findings array), `risk_assessment` (contains `score` and `severity`), and `skill`

---

### AT-04 — Markdown output format

**Setup:** Same as AT-01.

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --format markdown --output report.md
Select-String "##" report.md | Select-Object -First 5
```

**Expected:**
- `report.md` created
- Contains markdown headings (`##`)

---

### AT-05 — SKILLSPECTOR_LLM_COMMAND with spaces in path (Windows)

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "subprocess"
$env:SKILLSPECTOR_LLM_COMMAND = '"C:\Program Files\Claude\claude.exe" -p'
```

**Steps:**
```powershell
skillspector scan tests/fixtures/safe_skill --format terminal
```

**Expected:**
- Subprocess launches correctly (path with spaces handled by shlex on Windows)
- No `FileNotFoundError` about the path

> Skip this test if Claude is not installed in `Program Files`.

---

## Test Group 2 — Error Handling

### AT-06 — Missing SKILLSPECTOR_LLM_COMMAND raises clear error

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "subprocess"
Remove-Item Env:SKILLSPECTOR_LLM_COMMAND -ErrorAction SilentlyContinue
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
```

**Steps:**
```powershell
skillspector scan tests/fixtures/safe_skill --format terminal
```

**Expected:**
- Exit code non-zero
- Error message contains `SKILLSPECTOR_LLM_COMMAND`
- Error message does NOT suggest setting `OPENAI_API_KEY` or `NVIDIA_INFERENCE_KEY`

---

### AT-07 — Invalid command surfaces meaningful error

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "subprocess"
$env:SKILLSPECTOR_LLM_COMMAND = "nonexistent-command-xyz"
```

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --format terminal
```

**Expected:**
- Exit code non-zero
- Error message mentions the command failed or was not found
- No unhandled Python traceback reaching the user (or traceback is readable)

---

### AT-08 — Command that exits non-zero surfaces meaningful error

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "subprocess"
$env:SKILLSPECTOR_LLM_COMMAND = "cmd /c exit 1"   # always fails
```

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --format terminal
```

**Expected:**
- Exit code non-zero
- Error message contains "LLM subprocess failed" and the exit code

---

### AT-09 — --no-llm bypasses subprocess entirely (no command needed)

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "subprocess"
Remove-Item Env:SKILLSPECTOR_LLM_COMMAND -ErrorAction SilentlyContinue
```

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --no-llm --format terminal
```

**Expected:**
- Exit code 1 (non-zero; malicious skill scores > 50 even with static analysis only)
- Scan completes with static findings only
- No error about missing `SKILLSPECTOR_LLM_COMMAND`

---

## Test Group 3 — Provider Isolation

### AT-10 — subprocess provider does not fall back to OpenAI

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "subprocess"
$env:SKILLSPECTOR_LLM_COMMAND = "nonexistent-xyz"
$env:OPENAI_API_KEY = "sk-fake-key-that-should-not-be-used"
```

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --format terminal 2>&1
```

**Expected:**
- Error is about the subprocess command failing, NOT an OpenAI API error
- The fake OpenAI key is never used (no OpenAI network call attempted)

---

### AT-11 — Switching back to a standard provider works after subprocess

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "openai"
$env:OPENAI_API_KEY = "sk-real-key-here"
Remove-Item Env:SKILLSPECTOR_LLM_COMMAND -ErrorAction SilentlyContinue
```

**Steps:**
```powershell
skillspector scan tests/fixtures/safe_skill --format terminal
```

**Expected:**
- Scans successfully using the OpenAI provider
- No subprocess-related error

> Skip if no real OpenAI key is available.

---

## Test Group 4 — Alternative AI Tools

### AT-12 — Scan with Antigravity

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "subprocess"
$env:SKILLSPECTOR_LLM_COMMAND = "antigravity ask"
```

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --format terminal
```

**Expected:** Same as AT-01. Report produced, no API key error.

> Skip if `antigravity` CLI is not installed.

---

### AT-13 — Scan with OpenClaw

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "subprocess"
$env:SKILLSPECTOR_LLM_COMMAND = "openclaw chat"
```

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --format terminal
```

**Expected:** Same as AT-01. Report produced, no API key error.

> Skip if `openclaw` CLI is not installed.

---

## Test Group 5 — CLI Help & Documentation

### AT-14 — --help output mentions subprocess provider

**Steps:**
```powershell
skillspector scan --help
```

**Expected:**
- Output contains the word `subprocess`
- Output contains `SKILLSPECTOR_LLM_COMMAND`

---

### AT-15 — README provider table is accurate

**Steps:** Open `README.md` and read the LLM Analysis provider table.

**Expected:**
- Row for `subprocess` is present
- Credential column shows `SKILLSPECTOR_LLM_COMMAND`
- Endpoint column shows a shell command example

---

## Pass/Fail Criteria — Subprocess Provider

| Group | Tests | Required to pass |
|-------|-------|-----------------|
| Happy path | AT-01 to AT-05 | AT-01, AT-02, AT-03 mandatory; AT-04/05 recommended |
| Error handling | AT-06 to AT-09 | All mandatory |
| Provider isolation | AT-10, AT-11 | AT-10 mandatory; AT-11 if key available |
| Alternative tools | AT-12, AT-13 | Each skippable if CLI not installed; run any available |
| Docs | AT-14, AT-15 | Both mandatory |

**Feature is accepted when:** All mandatory tests pass and no skipped test is
due to a code defect (only due to missing optional CLI tool).

---

---

# Classic Provider Acceptance Tests

Tests for the pre-existing provider paths: `--no-llm`, Anthropic, OpenAI /
ChatGPT, and both the API-key and CLI routes for OpenClaw and Antigravity.

**Run these in a clean shell.** Clear all provider env vars before each group:

```powershell
# Paste this block before every test group
Remove-Item Env:SKILLSPECTOR_PROVIDER      -ErrorAction SilentlyContinue
Remove-Item Env:SKILLSPECTOR_LLM_COMMAND   -ErrorAction SilentlyContinue
Remove-Item Env:SKILLSPECTOR_MODEL         -ErrorAction SilentlyContinue
Remove-Item Env:OPENAI_API_KEY             -ErrorAction SilentlyContinue
Remove-Item Env:OPENAI_BASE_URL            -ErrorAction SilentlyContinue
Remove-Item Env:ANTHROPIC_API_KEY          -ErrorAction SilentlyContinue
Remove-Item Env:NVIDIA_INFERENCE_KEY       -ErrorAction SilentlyContinue
```

---

## Test Group 6 — No-LLM (Static Analysis Only)

The `--no-llm` flag skips every LLM call and runs static analyzers only.
No provider, no credentials, no network access required.

### AT-16 — Static scan of malicious skill detects findings without LLM

**Setup:** Clean env (no provider vars set).

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --no-llm --format terminal
```

**Expected:**
- Exit code 1 (non-zero exit indicates findings with risk score > 50; this is intentional behavior)
- At least one finding reported (static analyzers fire on the malicious fixture)
- Report does NOT mention "meta-analyzer" or "LLM"
- Completes in under 10 seconds

---

### AT-17 — Static scan of safe skill reports clean

**Setup:** Clean env.

**Steps:**
```powershell
skillspector scan tests/fixtures/safe_skill --no-llm --format terminal
```

**Expected:**
- Exit code 0
- Risk score 0–10 / severity LOW or SAFE
- No findings with HIGH or CRITICAL severity

---

### AT-18 — --no-llm works with every output format

**Setup:** Clean env.

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --no-llm --format json    --output nlm-report.json
skillspector scan tests/fixtures/malicious_skill --no-llm --format markdown --output nlm-report.md
skillspector scan tests/fixtures/malicious_skill --no-llm --format sarif   --output nlm-report.sarif
```

**Expected (each):**
- Exit code 1 (non-zero; malicious skill scores > 50, which is the findings-present signal)
- Output file created and non-empty
- JSON: `python -m json.tool nlm-report.json` exits 0
- SARIF: file contains `"$schema"` and `"runs"`

---

### AT-19 — --no-llm ignores any provider env vars that happen to be set

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "anthropic"
$env:ANTHROPIC_API_KEY     = "sk-ant-fake-key"
```

**Steps:**
```powershell
skillspector scan tests/fixtures/safe_skill --no-llm --format terminal
```

**Expected:**
- Exit code 0
- No network call to Anthropic (scan finishes instantly, no auth error)
- No error mentioning the fake key

---

### AT-20 — Recursive scan with --no-llm processes multiple skills

**Setup:** Clean env.

**Steps:**
```powershell
skillspector scan tests/fixtures/ --recursive --no-llm --format terminal
```

**Expected:**
- Exit code 1 (non-zero; at least one skill in the fixture set scores > 50)
- More than one skill scanned (output shows multiple skill names or a summary line)
- Each skill gets its own report section

---

## Test Group 7 — Anthropic Provider

> **Prerequisite:** A valid `ANTHROPIC_API_KEY` (begins `sk-ant-`).
> All tests in this group are **skippable** if no key is available.

### AT-21 — Basic scan with Anthropic API key

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "anthropic"
$env:ANTHROPIC_API_KEY     = "sk-ant-<your-key>"
```

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --format terminal
```

**Expected:**
- Exit code 0
- At least one HIGH or CRITICAL finding
- LLM meta-analyzer runs (findings list is filtered/annotated)
- No mention of OpenAI or NVIDIA in output

---

### AT-22 — Anthropic with model override

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "anthropic"
$env:ANTHROPIC_API_KEY     = "sk-ant-<your-key>"
$env:SKILLSPECTOR_MODEL    = "claude-sonnet-4-6"
```

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --format terminal --verbose
```

**Expected:**
- Exit code 0
- Verbose output references `claude-sonnet-4-6` (or the override is silently accepted)
- Findings reported as in AT-21

---

### AT-23 — Anthropic with invalid key fails with auth error, not crash

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "anthropic"
$env:ANTHROPIC_API_KEY     = "sk-ant-INVALID"
```

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --format terminal
```

**Expected:**
- Exit code non-zero
- Error message references authentication or API error
- No unformatted Python traceback as the final output (error is user-readable)

---

### AT-24 — Anthropic provider does not accept OPENAI_API_KEY as fallback

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "anthropic"
Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
$env:OPENAI_API_KEY = "sk-fake-openai-key"
```

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --format terminal 2>&1
```

**Expected:**
- Exit code non-zero
- Error references missing Anthropic credentials, not OpenAI
- OpenAI key is NOT used for an Anthropic scan

---

## Test Group 8 — OpenAI Provider

> **Prerequisite:** A valid `OPENAI_API_KEY` (begins `sk-`).
> All tests in this group are **skippable** if no key is available.

### AT-25 — Basic scan with OpenAI API key

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "openai"
$env:OPENAI_API_KEY        = "sk-<your-key>"
```

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --format terminal
```

**Expected:**
- Exit code 0
- At least one HIGH or CRITICAL finding
- LLM meta-analyzer runs
- No mention of Anthropic or NVIDIA in output

---

### AT-26 — OpenAI with ChatGPT model (gpt-4o)

ChatGPT's API uses the same `openai` provider. This test verifies a specific
GPT-4 class model works end-to-end.

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "openai"
$env:OPENAI_API_KEY        = "sk-<your-key>"
$env:SKILLSPECTOR_MODEL    = "gpt-4o"
```

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --format terminal --verbose
```

**Expected:**
- Exit code 0
- Findings reported; model override accepted without error
- Verbose output confirms `gpt-4o` or the override is silently accepted

---

### AT-27 — OpenAI with invalid key fails gracefully

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "openai"
$env:OPENAI_API_KEY        = "sk-INVALID-KEY"
```

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --format terminal
```

**Expected:**
- Exit code non-zero
- Error message references authentication or API error
- No raw Python traceback as final output

---

### AT-28 — No provider set but OPENAI_API_KEY present triggers fallback

The tool's credential waterfall uses `OPENAI_API_KEY` as a tier-2 fallback
when the active provider returns no credentials.

**Setup:**
```powershell
Remove-Item Env:SKILLSPECTOR_PROVIDER -ErrorAction SilentlyContinue
$env:OPENAI_API_KEY = "sk-<your-key>"
```

**Steps:**
```powershell
skillspector scan tests/fixtures/safe_skill --format terminal
```

**Expected:**
- Exit code 0
- Scan completes using OpenAI (or the default NVIDIA provider with OpenAI fallback)
- No error about missing credentials

---

## Test Group 9 — OpenAI-Compatible Endpoints (OpenClaw, Antigravity, Local)

OpenClaw and Antigravity may expose an OpenAI-compatible REST API in addition
to their CLI interfaces. This group tests the `openai` provider pointed at a
custom `OPENAI_BASE_URL` — the same mechanism works for Ollama, vLLM, and any
other compatible server.

> **Prerequisite for each:** The target server must be running and reachable.
> Skip any test whose server is unavailable.

### AT-29 — Scan via OpenClaw API endpoint

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "openai"
$env:OPENAI_API_KEY        = "<openclaw-api-key>"
$env:OPENAI_BASE_URL       = "<openclaw-openai-compatible-base-url>"
$env:SKILLSPECTOR_MODEL    = "<openclaw-model-name>"
```

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --format terminal
```

**Expected:**
- Exit code 0
- At least one HIGH or CRITICAL finding
- No reference to OpenAI's api.openai.com in error output (request went to the custom URL)

---

### AT-30 — Scan via Antigravity API endpoint

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "openai"
$env:OPENAI_API_KEY        = "<antigravity-api-key>"
$env:OPENAI_BASE_URL       = "<antigravity-openai-compatible-base-url>"
$env:SKILLSPECTOR_MODEL    = "<antigravity-model-name>"
```

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --format terminal
```

**Expected:**
- Exit code 0
- At least one HIGH or CRITICAL finding
- LLM meta-analyzer runs (report shows filtered findings)

---

### AT-31 — Local Ollama endpoint (model-agnostic baseline)

Use this test when no cloud key is available. Confirms the `OPENAI_BASE_URL`
override works with any OpenAI-compatible server.

**Setup:**
```powershell
# Start Ollama first: ollama serve
$env:SKILLSPECTOR_PROVIDER = "openai"
$env:OPENAI_API_KEY        = "ollama"          # Ollama ignores the key value
$env:OPENAI_BASE_URL       = "http://localhost:11434/v1"
$env:SKILLSPECTOR_MODEL    = "llama3.1:8b"     # or whichever model is pulled
```

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --format terminal
```

**Expected:**
- Exit code 0
- Findings reported (quality may vary by local model)
- No cloud network calls

---

### AT-32 — Wrong base URL produces connection error, not silent failure

**Setup:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "openai"
$env:OPENAI_API_KEY        = "sk-fake"
$env:OPENAI_BASE_URL       = "http://localhost:19999/v1"   # nothing listening here
```

**Steps:**
```powershell
skillspector scan tests/fixtures/malicious_skill --format terminal
```

**Expected:**
- Exit code non-zero
- Error message references connection failure or unreachable host
- Not a silent hang (fails within the configured timeout)

---

## Test Group 10 — OpenClaw and Antigravity CLI Path (Cross-Reference)

OpenClaw and Antigravity can also be driven through the `subprocess` provider
without any API key. These tests confirm both paths are available and produce
consistent results.

### AT-33 — OpenClaw CLI path vs API path produce equivalent severity

> Requires OpenClaw CLI **and** OpenClaw API endpoint both available.

**Setup A — CLI path:**
```powershell
$env:SKILLSPECTOR_PROVIDER    = "subprocess"
$env:SKILLSPECTOR_LLM_COMMAND = "openclaw chat"
skillspector scan tests/fixtures/malicious_skill --format json --output oc-cli.json
```

**Setup B — API path:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "openai"
$env:OPENAI_API_KEY        = "<openclaw-api-key>"
$env:OPENAI_BASE_URL       = "<openclaw-base-url>"
skillspector scan tests/fixtures/malicious_skill --format json --output oc-api.json
```

**Expected:**
- Both produce exit code 0
- Both report severity HIGH or CRITICAL for the malicious fixture
- Specific finding counts may differ slightly (LLM non-determinism) but overall risk tier matches

---

### AT-34 — Antigravity CLI path vs API path produce equivalent severity

> Requires Antigravity CLI **and** Antigravity API endpoint both available.

**Setup A — CLI path:**
```powershell
$env:SKILLSPECTOR_PROVIDER    = "subprocess"
$env:SKILLSPECTOR_LLM_COMMAND = "antigravity ask"
skillspector scan tests/fixtures/malicious_skill --format json --output ag-cli.json
```

**Setup B — API path:**
```powershell
$env:SKILLSPECTOR_PROVIDER = "openai"
$env:OPENAI_API_KEY        = "<antigravity-api-key>"
$env:OPENAI_BASE_URL       = "<antigravity-base-url>"
skillspector scan tests/fixtures/malicious_skill --format json --output ag-api.json
```

**Expected:**
- Both produce exit code 0
- Both report severity HIGH or CRITICAL
- Overall risk tier matches between paths

---

## Pass/Fail Criteria — All Providers

| Group | Tests | Mandatory | Skip condition |
|-------|-------|-----------|----------------|
| No-LLM | AT-16 to AT-20 | All | None — no credentials required |
| Anthropic | AT-21 to AT-24 | AT-21, AT-23, AT-24 | Skip group if no `ANTHROPIC_API_KEY` |
| OpenAI | AT-25 to AT-28 | AT-25, AT-27, AT-28 | Skip AT-25/27 if no `OPENAI_API_KEY`; AT-28 requires key |
| OpenAI-compatible | AT-29 to AT-32 | AT-32 | Skip AT-29/30/31 if server unavailable |
| CLI vs API parity | AT-33, AT-34 | Neither (informational) | Skip if either path unavailable |

**Overall acceptance:** No-LLM group (AT-16–20) must pass unconditionally.
Each keyed group passes when mandatory tests in that group pass.
Skips are valid only when the prerequisite service/key is genuinely absent —
not when a test reveals a defect.
