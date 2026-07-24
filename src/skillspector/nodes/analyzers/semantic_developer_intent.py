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

"""Semantic developer-intent analyzer node.

Detects context-dependent risk and semantic description–behavior mismatches
by comparing the skill's manifest (name, description, permissions) against
its actual code behavior using LLM-based analysis.
"""

from __future__ import annotations

from skillspector.constants import _SKILLSPECTOR_DEFAULT_MODEL, MODEL_CONFIG
from skillspector.llm_analyzer_base import LLMAnalyzerBase
from skillspector.llm_utils import run_async
from skillspector.logging_config import get_logger
from skillspector.state import AnalyzerNodeResponse, SkillspectorState, llm_call_record

ANALYZER_ID = "semantic_developer_intent"
logger = get_logger(__name__)


class _NoUsage:
    llm_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


ANALYZER_PROMPT = """\
You are a developer-intent auditor for AI agent skills.  Your job is to
detect mismatches between what a skill *claims* to do (its manifest and
code documentation) and what it *actually* does in code, as well as
capabilities that are unjustified given the skill's stated purpose.

Skill manifest context:
{manifest_section}

Use the rule IDs exactly as listed.  Reference the L-prefixed line numbers
when reporting findings.

| Rule ID | Detection |
|---------|-----------|
| SDI-1   | Description-behavior mismatch: the skill's manifest description does not match actual code operations |
| SDI-2   | Context-inappropriate capability: code capability is unjustified given the skill's stated purpose |
| SDI-3   | Scope creep: code accesses/modifies more than declared manifest permissions |
| SDI-4   | Intent-code divergence: comments/docstrings actively contradict what the code does |

---

### SDI-1  Description-Behavior Mismatch
Skill-manifest-level semantic check: the natural-language description in the
manifest claims limited scope but the code does more.

Examples:
- Manifest says "summarize text" but code sends HTTP requests to external URLs
- Manifest says "local file reader" but code modifies remote resources
- Manifest says "read-only analytics" but code writes to databases

Do NOT flag if the behavior is an obviously expected implementation detail of
the described purpose (e.g. a "web search" skill making HTTP requests).

Use rule ID **SDI-1** for all description-behavior mismatch findings.

---

### SDI-2  Context-Inappropriate Capability
The code implements a capability that is not justified by the skill's stated
purpose in the manifest.

Examples:
- A "text formatter" skill that spawns subprocesses or executes shell commands
- A "calendar reminder" skill that reads environment variables for credentials
- A "document converter" skill that accesses the network

Do NOT flag if:
- The capability is a direct and obvious requirement of the stated purpose
- The manifest explicitly declares the capability as part of the skill's scope

Use rule ID **SDI-2** for all context-inappropriate-capability findings.

---

### SDI-3  Scope Creep Relative to Declared Permissions
The skill's manifest declares a specific set of permissions, but the code
accesses or modifies more than what those permissions cover.

Examples:
- Manifest permissions list only "read:files" but code writes files
- Manifest declares no network permissions but code makes HTTP calls
- Manifest says permissions: [] but code reads sensitive environment variables

Do NOT flag if:
- The code's actual behavior matches the declared permissions
- The manifest has no permissions section (no baseline to compare against)

Use rule ID **SDI-3** for all scope-creep findings.

---

### SDI-4  Intent-Code Divergence
Comments, docstrings, or inline documentation actively contradict what the
code does.

Examples:
- A function docstring says "returns None, no side effects" but the function
  writes to disk and returns a value
- A comment says "# read-only query" above a statement that deletes records
- A module docstring says "safe, sandboxed" but the code calls os.system()

Do NOT flag if:
- The comment/docstring is merely incomplete (missing information is not the
  same as contradictory)
- The difference is a minor implementation detail irrelevant to security or intent

Use rule ID **SDI-4** for all intent-code-divergence findings.

---

### Output rules
- Skip findings for behavior that is obviously expected given the skill's
  stated purpose.
- Focus on semantic and intent-level mismatches that require understanding of
  the skill's purpose — not low-level static code patterns.
- Do NOT report issues already covered by static or structural analyzers
  (e.g. MCP schema violations, regex-detected patterns).
"""


def _format_manifest(manifest: dict) -> str:
    """Format manifest dict into a readable string for the prompt."""
    if not manifest:
        return "(No manifest available — treat as unknown purpose skill.)"
    parts = []
    if name := manifest.get("name"):
        parts.append(f"Name: {name}")
    if description := manifest.get("description"):
        parts.append(f"Description: {description}")
    if triggers := manifest.get("triggers"):
        if isinstance(triggers, list):
            parts.append(f"Triggers: {', '.join(str(t) for t in triggers)}")
        else:
            parts.append(f"Triggers: {triggers}")
    if permissions := manifest.get("permissions"):
        if isinstance(permissions, list):
            parts.append(f"Permissions: {', '.join(str(p) for p in permissions)}")
        else:
            parts.append(f"Permissions: {permissions}")
    return "\n".join(parts) if parts else "(No manifest details available.)"


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Discover developer-intent findings via LLM analysis."""
    if not state.get("use_llm", True):
        return {"findings": []}

    file_cache: dict[str, str] = state.get("file_cache") or {}
    if not file_cache:
        return {"findings": []}

    manifest: dict = state.get("manifest") or {}
    model_config: dict[str, str] = state.get("model_config") or {}
    model = (
        model_config.get(ANALYZER_ID)
        or model_config.get("default")
        or MODEL_CONFIG.get(ANALYZER_ID)
        or _SKILLSPECTOR_DEFAULT_MODEL
    )

    try:
        prompt = ANALYZER_PROMPT.format(manifest_section=_format_manifest(manifest))
        analyzer = LLMAnalyzerBase(base_prompt=prompt, model=model)
        batches = analyzer.get_batches(sorted(file_cache), file_cache)
        results = run_async(analyzer.arun_batches(batches))
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
