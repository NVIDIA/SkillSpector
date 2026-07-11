# SPDX-License-Identifier: Apache-2.0
"""Trim prompt text the pruned schema no longer needs (fork behavior)."""

from __future__ import annotations

import skillspector.nodes.analyzers.semantic_developer_intent as dev_intent
import skillspector.nodes.analyzers.semantic_quality_policy as quality
import skillspector.nodes.meta_analyzer as meta

from ._patchlib import replace_module_str

# Present verbatim in both semantic analyzers' ANALYZER_PROMPT (note the two
# spaces after "listed." and the mid-sentence newline).
_LINE_NUMBER_OLD = (
    "Use the rule IDs exactly as listed.  Reference the L-prefixed line numbers\n"
    "when reporting findings."
)
_LINE_NUMBER_NEW = "Use the rule IDs exactly as listed."

# The meta-analyzer's "Your Task" list: drop the intent (2) and impact (3) items.
_META_TASK_OLD = (
    "1. Is this a true vulnerability or a false positive?\n"
    "2. What is the likely intent (malicious, negligent, or benign)?\n"
    "3. What is the potential impact if exploited?\n"
    "4. Does the skill context make this more or less dangerous?"
)
_META_TASK_NEW = (
    "1. Is this a true vulnerability or a false positive?\n"
    "2. Does the skill context make this more or less dangerous?"
)


def apply() -> None:
    replace_module_str(dev_intent, "ANALYZER_PROMPT", _LINE_NUMBER_OLD, _LINE_NUMBER_NEW)
    replace_module_str(quality, "ANALYZER_PROMPT", _LINE_NUMBER_OLD, _LINE_NUMBER_NEW)
    replace_module_str(meta, "PER_FILE_ANALYSIS_PROMPT", _META_TASK_OLD, _META_TASK_NEW)
