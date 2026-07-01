# B.3.1: MCP Least-Privilege Analysis (LP1 -- LP4)

**Author:** Nir Paz | **Date:** 2026-03-30 | **Status:** Implemented  
**Component:** `src/skillspector/nodes/analyzers/mcp_least_privilege.py`

---

## 1. Background

MCP (Model Context Protocol) skills declare their intended permissions in a
manifest (`SKILL.md`). A well-behaved skill should request only the capabilities
it actually uses -- the **principle of least privilege**. In practice, skills
frequently:

- Use capabilities they never declared (hiding true intent),
- Declare broad wildcards (`*`, `all`) instead of specific permissions,
- Omit the permissions field entirely while still performing sensitive operations,
- Declare permissions for capabilities no longer present in the code.

These gaps are invisible to users and to the AI agent that invokes the skill.
The B.3.1 analyzer bridges this gap by cross-referencing what the manifest
**declares** against what the code **actually does**.

---

## 2. Architecture

```text
                    ┌─────────────┐
                    │   SKILL.md  │
                    │  (manifest) │
                    └──────┬──────┘
                           │ permissions[]
                           ▼
           ┌───────────────────────────────┐
           │   mcp_least_privilege (node)  │
           │                               │
           │  1. Map permissions → caps    │
           │  2. Detect code capabilities  │
           │  3. Cross-reference & emit    │
           └───────────────────────────────┘
                           │
                   ┌───────┴───────┐
                   │   Findings    │
                   │  LP1 -- LP4   │
                   └───────────────┘
```

The analyzer runs as a LangGraph node. It requires three pieces of state:

| State key              | Type             | Description                                  |
|------------------------|------------------|----------------------------------------------|
| `manifest`             | `dict`           | Parsed SKILL.md frontmatter                  |
| `file_cache`           | `dict[str, str]` | File path -> content for all skill files     |
| `component_metadata`   | `list[dict]`     | Per-file metadata including `executable` flag |

The analyzer is **pure static analysis** -- no LLM calls, no network access.
It completes in milliseconds regardless of skill size.

---

## 3. Capability Detection

The analyzer scans executable files for regex patterns grouped into six
capability categories:

| Category      | Example patterns detected                                          |
|---------------|--------------------------------------------------------------------|
| `shell`       | `subprocess`, `Popen`, `curl`, `wget`, `chmod`                    |
| `network`     | `httpx`, `requests`, `urllib`, `aiohttp`, `socket.connect`        |
| `file_read`   | `open(..., "r")`, `.read_text()`, `os.listdir`, `os.walk`, `glob`|
| `file_write`  | `open(..., "w")`, `.write_text()`, `shutil.copy`, `os.rename`    |
| `env`         | `os.environ`, `os.getenv`, `process.env`, `dotenv`               |
| `mcp`         | `create_session`, `MCPClient`, `mcp.client`                      |

Each file is scanned once. If any pattern in a category matches, that category
is recorded for the file. Test files (`test_*.py`, `*_test.py`) are tracked
separately -- capabilities found only in test files receive lower confidence
scores.

---

## 4. Permission Mapping

Declared permission strings (from `manifest.permissions[]`) are mapped to the
same capability categories using keyword matching:

| Permission keyword(s)                       | Maps to category |
|---------------------------------------------|------------------|
| `bash`, `shell`, `terminal`, `command`      | `shell`          |
| `network`, `http`, `fetch`, `api`           | `network`        |
| `read`, `fs_read`, `file_read`              | `file_read`      |
| `write`, `fs_write`, `file_write`           | `file_write`     |
| `env`, `environment`                        | `env`            |
| `mcp`, `tools`, `tool_use`                  | `mcp`            |

Wildcard values (`*`, `all`, `full`, `any`) are detected separately and trigger
the LP2 rule.

---

## 5. Rules

### LP1 -- Underdeclared Capability

| Field       | Value                                                     |
|-------------|-----------------------------------------------------------|
| **Severity**   | HIGH                                                   |
| **Confidence** | 0.75 (code files) / 0.55 (test-only files)            |
| **Tags**       | ASI02                                                  |

**Triggers when:** A code capability category is detected in executable files
but no declared permission maps to that category.

**Why it matters:** A skill that uses network access but doesn't declare it is
hiding its true behavior. This is the strongest indicator of deceptive intent
among the LP rules -- the skill is actively performing operations it claims not
to need.

**Example:** A skill declares `permissions: [read]` but its code contains
`httpx.post(...)`. LP1 fires for the undeclared `network` capability.

**Remediation:** Add the missing permission to SKILL.md, or remove the code
that requires it.

---

### LP2 -- Wildcard Permission

| Field       | Value                                                     |
|-------------|-----------------------------------------------------------|
| **Severity**   | MEDIUM                                                 |
| **Confidence** | 0.90                                                   |
| **Tags**       | ASI02                                                  |

**Triggers when:** Any entry in the permissions list is `*`, `all`, `full`, or
`any`.

**Why it matters:** Wildcard permissions disable permission-based security
controls entirely. Any capability the skill uses is technically "declared," but
the user has no visibility into what the skill actually does. This is the
permission-system equivalent of `chmod 777`.

**Example:**
```yaml
permissions:
  - "*"
```

**Remediation:** Replace the wildcard with an explicit list of required
permissions. Request only the minimum access needed.

---

### LP3 -- Missing Permission Declaration

| Field       | Value                                                     |
|-------------|-----------------------------------------------------------|
| **Severity**   | MEDIUM                                                 |
| **Confidence** | 0.70                                                   |
| **Tags**       | ASI02                                                  |

**Triggers when:** The manifest has no `permissions` field (or it is an empty
list) **and** the analyzer detects code capabilities in executable files.

**Why it matters:** Without declared permissions, the skill's intent is
completely opaque. Users and agents cannot evaluate whether the skill's access
level is appropriate. This is less suspicious than LP1 (could be an oversight
rather than deception) but still a significant transparency gap.

**Example:** A SKILL.md with no `permissions:` key, but the code calls
`os.environ["API_KEY"]` and `subprocess.run(...)`.

**Remediation:** Add a `permissions` field to SKILL.md listing the capabilities
the skill requires.

---

### LP4 -- Overdeclared Permission

| Field       | Value                                                     |
|-------------|-----------------------------------------------------------|
| **Severity**   | LOW                                                    |
| **Confidence** | 0.65                                                   |
| **Tags**       | ASI02                                                  |

**Triggers when:** A permission is declared in the manifest but no
corresponding code capability is detected in any file.

**Why it matters:** Overdeclared permissions may indicate:
- Removed functionality that wasn't cleaned up (benign but sloppy),
- Pre-staging for future abuse (the permission is declared now so that
  malicious code added later won't trigger LP1),
- Copy-paste from another skill's manifest.

This is LOW severity because it doesn't represent active exploitation, but it
does violate least-privilege and warrants review.

**Example:** `permissions: [shell, network]` but the code only uses `httpx`
(network). The `shell` permission is overdeclared.

**Remediation:** Remove the declared permission if the corresponding capability
is no longer used.

---

## 6. Interaction Between Rules

The rules are designed to avoid redundant or contradictory findings:

| Condition                        | Rules that fire        | Rules suppressed |
|----------------------------------|------------------------|------------------|
| Wildcard + underdeclared caps    | LP2                    | LP1 (suppressed) |
| No permissions + capabilities    | LP3                    | LP1, LP4 (no list to compare) |
| Normal list + gap                | LP1 and/or LP4         | --               |
| Docs-only skill (no executables) | (none -- analyzer skips) | All            |
| No manifest                      | (none -- analyzer skips) | All            |

---

## 7. Test Fixtures

| Fixture directory                 | Expected findings | Purpose                          |
|-----------------------------------|-------------------|----------------------------------|
| `mcp_clean_skill/`               | None              | Negative test -- all caps declared |
| `mcp_underdeclared_skill/`        | LP1, LP3          | Missing permissions + undeclared caps |
| `mcp_overprivileged_skill/`      | LP2, LP4          | Wildcard + overdeclared permissions |

---

## 8. Limitations

- **Regex-based detection:** Capability detection uses pattern matching, not
  semantic analysis. A capability used inside a never-executed code path will
  still be detected. Conversely, capabilities invoked via dynamic dispatch
  (`getattr`, `importlib`) may be missed.
- **No transitive analysis:** If a skill calls a library function that
  internally uses `subprocess`, the analyzer won't detect `shell` capability
  unless the skill's own code mentions the patterns directly.
- **Permission keywords are English-only:** The keyword-to-category mapping
  assumes English permission names.
