# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Semantic quality-policy analyzer node.

Evaluates AI agent skill files against a quality and safety rubric using
LLM-based discovery.  Flags vague triggers, missing user warnings, and
natural-language policy violations that static/behavioral tools cannot detect.
"""

from __future__ import annotations

import asyncio

from skillspector.constants import _SKILLSPECTOR_DEFAULT_MODEL
from skillspector.llm_analyzer_base import LLMAnalyzerBase
from skillspector.logging_config import get_logger
from skillspector.state import AnalyzerNodeResponse, SkillspectorState, llm_call_record

ANALYZER_ID = "semantic_quality_policy"
logger = get_logger(__name__)


class _NoUsage:
    llm_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


ANALYZER_PROMPT = """\
You are a quality and safety auditor for AI agent skills.  Your job is to
review a single skill file and report findings that fall into the categories
below.  Each category specifies which file types it applies to — you MUST
skip categories that do not apply to the file you are reviewing.

IMPORTANT: The file is identified by its path and extension in the heading
that follows this prompt.  Only report findings from categories whose
file-type scope matches the current file.  If a category says "markdown and
manifest files only", do NOT report those findings for .py or .sh files.

Use the rule IDs exactly as listed.  Reference the L-prefixed line numbers
when reporting findings.

| Rule ID | Category | Applies to |
|---------|----------|------------|
| SQP-1 | Vague Triggers | markdown, plain text, manifest files only |
| SQP-2 | Missing User Warnings | code files AND markdown files |
| SQP-3 | Natural-Language Policy Violations | ALL file types |

---

### SQP-1  Vague Triggers
**Applies to: markdown (.md), plain text (.txt), and manifest files (.yaml, .yml, .json, .toml) only.**
Skip this category for code files.

Look for activation conditions, trigger phrases, or invocation descriptions
that are ambiguous or overly broad and could cause unintended skill
invocations.  Flag any of the following:
- Overly broad trigger phrase that overlaps with common everyday speech (e.g. "help me", "do this")
- Ambiguous activation condition — unclear when the skill activates vs. does not
- Missing specificity on trigger scope or constraints (no explicit list of trigger phrases, or no negative examples)

Do NOT flag if:
- The trigger phrase is domain-specific enough to avoid everyday collisions
  (e.g. "run terraform plan" is specific, not vague)
- The skill explicitly lists negative examples or exclusion conditions
- The manifest/description limits activation to a narrow context (e.g. only
  inside a specific IDE command palette)

Use rule ID **SQP-1** for all vague-trigger findings.

---

### SQP-2  Missing User Warnings
**Applies to: code files (.py, .sh, .js, .ts, .go, .rs, .rb, .pl, etc.) AND markdown files (.md), but with different criteria per type.**

**For code files:** flag safety-critical operations that lack ANY form of user
disclosure — no confirmation prompt, no logging/print statement, no docstring
or comment explaining the action, and no mention in the skill's README/SKILL.md.
Operations to check:
- File writes or deletions
- Network / HTTP calls that transmit user or system data
- Access to sensitive environment variables or credentials
- Subprocess or shell execution
- Destructive or irreversible operations

Do NOT flag an operation if:
- The code includes a visible confirmation prompt, user-facing log, or print
- The skill's markdown description explicitly warns about the operation
- The operation is clearly part of the skill's stated purpose (e.g. a "deploy"
  skill running shell commands is expected, not a missing warning)

**For markdown files:** flag when the skill description omits warnings about
behaviours that could affect user data, privacy, or system integrity.

Use rule ID **SQP-2** for all missing-warning findings.

---

### SQP-3  Natural-Language Policy Violations
**Applies to: ALL file types** (markdown, code, config, etc.).

Look for natural-language organizational policy violations.  These may appear
in markdown instructions, code string literals, comments, or config values.
Flag any of the following:
- Language or locale policy violation (e.g. skill forces a specific language without user opt-in)

Do NOT flag if:
- The skill explicitly offers the user a language/locale choice or opt-in
- The locale constraint is clearly documented and justified (e.g. a
  region-specific compliance tool)

Use rule ID **SQP-3** for all policy-violation findings.

---

### Output rules

- Do NOT report issues already covered by static security scanners (e.g. regex
  prompt-injection patterns, known exfiltration signatures).  Focus on semantic
  quality and policy concerns that require natural-language understanding.
"""


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Discover quality/policy findings via LLM analysis."""
    if not state.get("use_llm", True):
        return {"findings": []}

    file_cache: dict[str, str] = state.get("file_cache") or {}
    files = sorted(file_cache.keys())
    if not files:
        return {"findings": []}

    model_config: dict[str, str] = state.get("model_config") or {}
    model = (
        model_config.get(ANALYZER_ID) or model_config.get("default") or _SKILLSPECTOR_DEFAULT_MODEL
    )

    try:
        analyzer = LLMAnalyzerBase(base_prompt=ANALYZER_PROMPT, model=model)
        batches = analyzer.get_batches(files, file_cache)
        results = asyncio.run(analyzer.arun_batches(batches))
        findings = analyzer.collect_findings(results)
        logger.info("%s: %d findings", ANALYZER_ID, len(findings))
        return {
            "findings": findings,
            "llm_call_log": [llm_call_record(ANALYZER_ID, ok=True, **analyzer.llm_usage)],
        }
    except ValueError:
        raise
    except Exception as exc:
        logger.warning("%s failed: %s", ANALYZER_ID, exc)
        return {
            "findings": [],
            "llm_call_log": [
                llm_call_record(
                    ANALYZER_ID,
                    ok=False,
                    error=str(exc),
                    **locals().get("analyzer", _NoUsage()).llm_usage,
                )
            ],
        }
