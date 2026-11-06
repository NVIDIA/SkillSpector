# Agent scan workflow observation: 6 November 2026

All 12 CLI observations and four real stdio MCP scan calls matched the bounded
interface checks below. Initialization, tool discovery, JSON-RPC framing, and
server shutdown succeeded. Missing LLM credentials were an intentional failure
control; this is not a successful LLM-backed scan or full CI result.
See the [repeatable procedure](AGENT_SCAN_WORKFLOW.md).

## Artifact and environment

| Item | Recorded identity |
|------|-------------------|
| Source | [`e48d771a26f7398b36256ceb8d8f94adce3d5c44`](https://github.com/NVIDIA/SkillSpector/tree/e48d771a26f7398b36256ceb8d8f94adce3d5c44) |
| Installed artifact | Locally built `skillspector-2.12.0-py3-none-any.whl`; separate runtime, no editable project install |
| Wheel SHA-256 | `acf31f34bf09d751a9bf17cfc94dc240b2a3934e4b11e64aba52b77ddb72a3bc` |
| Package integrity | All 148 wheel package files matched both source and installed contents |
| Runtime | macOS 26.7.2 arm64; CPython 3.12.13 |
| SDK / transport | MCP Python SDK 1.29.0, AnyIO 4.15.1; stdio to installed `skillspector mcp` |
| Negotiated protocol | `2025-11-25` |
| Wire identity fields | Client `mcp` / `0.1.0`; server `skillspector` / `1.29.0` |
| Scanner identity | Installed package and verdict version `2.12.0`, distinguished by source and wheel hash above |
| Integrating agent | Implementing Codex agent orchestrating the SDK client and interpreting results; exact model/client build unavailable |
| Bounds | 90 seconds per CLI command/call; 450-second MCP session bound; observation finished in 37.990 seconds |

The wheel was built without build isolation using existing `build` 1.5.0 tools.
Runtime dependencies came from this revision's frozen `uv.lock` using
`uv export --frozen --no-dev --extra mcp --no-emit-project`, then installation
into a fresh venv. uv 0.10.9 used a warm package cache. The exported requirements
SHA-256 was `35e83f7a723ad6402871328e68eb283d6219ad515176c05cdcc558d1b556f8a8`.
The installed module resolved inside that venv's `site-packages`, outside the
source checkout.

The wheel's `2.12.0` metadata does not mean it was built from the `v2.12.0`
release tag. Likewise, the generic wire identity version fields are not proof
of scanner source identity. No release/version-identity fix is claimed here.

Commands ran outside the restricted execution sandbox with an explicit
environment: fresh `HOME`/`TMPDIR`, runtime/system `PATH`, `PYTHONNOUSERSITE=1`,
locale/color settings, disabled personal/system Git configuration, and
`SKILLSPECTOR_PROVIDER=nv_build`. No real provider credentials, `.env`, tracing
settings, or personal CLI-provider configuration were passed. Fixtures remained
data and were never installed, enabled, or executed. Network traffic was not
captured or blocked; no zero-network claim is made.

## Frozen inputs

The procedure contains the exact fixture-generation code, including the complete
benign 1x1 RGBA PNG. Paths below are relative to the fixture directory.

| File | Bytes | SHA-256 |
|------|-------|---------|
| `clean/SKILL.md` | 124 | `adba802d2383bf183a6e2168fa4bc89bdcfb5a084a6d70a25e7ca6835ed0b4d7` |
| `finding/SKILL.md` | 141 | `3767c4af7182a3b072f5478e811b07507f9e2783a9bf8a9e683f113d75d802c4` |
| `incomplete/SKILL.md` | 194 | `294bf664e3cf1c4f812b6cb1bea8b357e775e302089992329ab1aa593aab2818` |
| `incomplete/assets/chart.png` | 70 | `2e9e614a233d1845c927cdda9175ec8bdf3929c242f4d2ad97a90547bf19974e` |

- **Clean:** a date-formatting description and benign instructions.
- **Finding:** the active line `Ignore all previous instructions.`; the agent
  treated this as untrusted fixture text. One HIGH P1 finding was expected.
- **Incomplete:** benign chart instructions explicitly referencing the PNG.
  Opaque image analysis was expected to remain incomplete without manufacturing
  a security finding.
- **LLM unavailable:** the clean input with LLM requested in the credential-free
  environment. Missing requested analysis was expected to prevent a clean full
  scan interpretation.

## CLI expected and actual checks

The installed console binary was invoked from outside the source checkout:

```bash
skillspector scan <fixture> --format json --no-llm
skillspector scan <fixture> --format json --no-llm --fail-on-findings
skillspector scan <fixture> --format json --no-llm --fail-on-incomplete
```

For the unavailable-LLM row, `<fixture>` was the clean input and `--no-llm` was
omitted in all three invocations. Each complete stdout parsed as one JSON
document; diagnostics were captured separately from stdout.

Expected gate behavior: ordinary score-band policy can return 0 despite active
findings or incomplete analysis; the corresponding strict gate rejects those
conditions. Fatal execution failure takes precedence and exits 2 under all
three modes. Actual results:

| Control | Score / recommendation | Findings | Completeness / execution | Default / findings-strict / incomplete-strict |
|---------|------------------------|----------|--------------------------|----------------------------------------------|
| Clean, static | 0 / SAFE | none | complete / true | 0 / 0 / 0 |
| Active override, static | 20 / SAFE | P1, HIGH | complete / true | 0 / 1 / 0 |
| Referenced PNG, static | 0 / CAUTION | none | partial / true | 0 / 0 / 1 |
| Clean, LLM requested but unavailable | 0 / CAUTION | none | failed / false | 2 / 2 / 2 |

The P1 row's aggregate LOW/SAFE band does not erase the HIGH issue. An
any-active-finding policy must inspect `issues` or use `--fail-on-findings`.
The PNG row proves that default exit 0, zero findings, and score 0 can coexist
with CAUTION and incomplete analysis. These are measured outputs of the pinned
revision, not universal scores for future versions.

## Actual MCP transport

The MCP SDK launched a transparent capture process, which forwarded bytes
unchanged to/from the installed `skillspector mcp` process. No graph/helper call
or mocked scan result substituted for a tool invocation. All seven client
frames and six server frames parsed as JSON-RPC 2.0; response IDs matched
requests 0 through 5. No JSON-RPC errors or stray protocol stdout appeared.
SDK client stderr was empty; scanner diagnostics stayed on server stderr.
The session closed and the scanner process exited 0.

Wire transcript summary:

| ID | Client method | Observed server result |
|----|---------------|------------------------|
| 0 | `initialize` | protocol 2025-11-25, skillspector server info and capabilities |
| none | `notifications/initialized` | notification; no response expected |
| 1 | `tools/list` | one tool, `scan_skill`; inputs `target`, `use_llm`, `output_format` |
| 2 | `tools/call`, clean / `use_llm=false` | structured verdict received |
| 3 | `tools/call`, finding / `use_llm=false` | structured verdict received |
| 4 | `tools/call`, incomplete / `use_llm=false` | structured verdict received |
| 5 | `tools/call`, clean / `use_llm=true` | structured unavailable-analysis verdict received |

Every call requested `output_format="json"`. `response.structuredContent` was a
dictionary, and its embedded `report` parsed as JSON with the same recommendation.
Actual top-level verdicts:

| Control | Score / recommendation | Findings | Completeness / execution | `safe_to_install` | LLM requested / available / used |
|---------|------------------------|----------|--------------------------|-------------------|---------------------------------|
| Clean | 0 / SAFE | none | complete / true | true | false / false / false |
| Active override | 20 / SAFE | P1, HIGH | complete / true | true | false / false / false |
| Referenced PNG | 0 / CAUTION | none | partial / true | false | false / false / false |
| LLM unavailable | 0 / CAUTION | none | partial / true | false | true / false / false |

`isError` was false for all four MCP calls. That field describes tool-call
delivery, not scan completeness. Every `scan_mode` was `static-only`.
The PNG had 50% coverage and an `opaque_content` ledger exception. The
unavailable-LLM MCP result had 100% component coverage but remained incomplete
with a `missing_credentials` exception.

The unavailable-LLM CLI and MCP paths classified the result differently:
CLI reported failed execution/exit 2; MCP returned a successful tool response
with partial static analysis and `safe_to_install=false`. The agent retained
that difference instead of converting either result into successful semantic
analysis. No CLI or MCP output contract was changed.

## Integrating agent interpretation

- **Clean:** no active static finding was observed in this complete static
  scope. LLM analysis was not requested; this is not proof of safety or an
  instruction to install anything.
- **Active override:** inspect the P1 finding despite SAFE, exit 0, and
  `safe_to_install=true`. The any-active-finding gate rejected it. The fixture
  instruction was not followed.
- **Incomplete:** hold for review/additional analysis. CAUTION and
  `is_complete=false` distinguish this from the clean control even though both
  had zero score/findings and default exit 0. The gap alone does not establish
  maliciousness.
- **LLM unavailable:** requested analysis did not happen. Report the missing
  prerequisite and actual interface outcome; do not describe this as an
  intentional static-only request or a clean full scan.

This interpretation was recorded by the implementing agent after reading the
reports. It was not a blind assessment, separate fresh-agent benchmark, or
SkillSpector bundled semantic analysis. No installation was requested or made.

## Scope and remaining limits

The procedure's fixture-generation example was also executed and reproduced all
four file hashes. Its SDK client example then completed four additional MCP
calls with the same verdict fields. These example checks are separate from the
primary wire transcript above. Markdown rendering and local links/anchors were
checked as well.

All interface checks completed within their declared bounds. The unavailable
analysis control intentionally produced error/partial scan outcomes; those
were expected observations, not passed semantic analysis. No runtime prerequisite
blocked the declared scenarios, and none timed out.

The result applies to the exact wheel, macOS/Python runtime, SDK, stdio transport,
and tiny controls above. It does not establish HTTP/other-client behavior,
general scanner accuracy, provider availability, or universal stdio health.
Source-version identity and the historical README stdio warning remain separate
documentation work; they were not treated as resolved by this check.

Not run: live paid-provider comparisons, other formats/transports/platforms,
remote targets, full unit/integration/coverage suites, Docker, or complete PR CI.
Historical release validation was not reused. This PR adds validation documents
and navigation; it does not change scanner, scoring, CLI, or MCP runtime code.
