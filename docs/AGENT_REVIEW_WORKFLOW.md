# Agent-led Skill Review Workflow

SkillSpector can be used as the static evidence layer in an agent-led review
workflow. This is useful when an agent runtime can read source files and reason
about whether a flagged behavior is expected for a skill's stated purpose.

The workflow is intentionally simple:

1. Run a static SkillSpector scan.
2. Read the JSON report.
3. Inspect the skill source around each important finding.
4. Decide whether the behavior is expected, suspicious, or unacceptable.

This keeps the fast, deterministic scanner in the loop while still requiring a
contextual review before installing a skill.

## Why use an agent-led review?

Risk scores are evidence, not a complete decision. Some skills legitimately need
sensitive capabilities. For example, a deployment skill may need shell access,
and a web-search skill may need network access. The reviewer still needs to ask:

- Does the skill description explain this capability?
- Is the destination or file access documented?
- Is the behavior necessary for the skill's purpose?
- Are permissions broader than the implementation needs?
- Is there hidden behavior such as credential harvesting, persistence, or
  unexpected external transmission?

## Recommended scan command

Use static analysis as the default evidence pass:

```bash
skillspector scan ./my-skill --no-llm --format json --output /tmp/skillspector-report.json
```

Then have the agent read `/tmp/skillspector-report.json` and inspect the files
referenced by HIGH, CRITICAL, and sensitive MEDIUM findings.

Sensitive MEDIUM findings usually include:

- network requests or uploads
- environment variables, tokens, passwords, or local config
- file writes or deletes
- shell execution, subprocesses, or dynamic code execution
- MCP permission mismatches or tool metadata issues
- persistence, self-modification, or downloaded code

## Suggested verdicts

A simple three-state verdict is usually enough:

| Verdict | Meaning |
|---|---|
| `APPROVE` | No meaningful security concern; behavior matches the skill's stated purpose. |
| `CAUTION` | Sensitive behavior exists, but it is documented, necessary, and bounded. |
| `REJECT` | Malicious, deceptive, unexplained, or overly broad behavior was found. |

## Score guidance

SkillSpector's risk score should guide review priority, not replace review:

| Score | Suggested posture |
|---|---|
| `0-20` | Usually acceptable after a quick source check. |
| `21-35` | Findings should be explained by the skill's purpose. |
| `36-50` | Manual review required; default to caution. |
| `51-80` | Default reject unless the source is trusted and every sensitive behavior is necessary. |
| `81-100` | Reject. |

## Output shape

Agents should summarize both evidence and judgment:

```markdown
## Skill Review: `<skill-name>`

**Verdict:** CAUTION
**Risk:** 33/100 · MEDIUM
**Install posture:** trusted workstation only

### Signal Overview
| Source | Result | Interpretation |
|---|---|---|
| SkillSpector static scan | 2 findings | network and env/config access |
| Agent source review | justified | behavior matches stated purpose |
| Sensitive surface | token cache, external API | requires trusted environment |

### Key Evidence
| Rule | Severity | Location | Review judgment |
|---|---|---|---|
| E1 | MEDIUM | scripts/main.sh:42 | expected API call for documented feature |

### Decision
Explain whether the static evidence is consistent with the skill's purpose and
which guardrails are required before installation.
```

## Optional skill wrapper

Agent runtimes that support skill-style workflows can encode the steps above in
a small companion skill. One community example is
[skill-inspector](https://github.com/Dxboy266/skill-inspector), which runs a
static SkillSpector pass and then asks the active agent to perform source-level
semantic review.
