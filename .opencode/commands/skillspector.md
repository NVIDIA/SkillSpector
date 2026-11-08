---
description: Scan an AI agent skill for security risks with SkillSpector (static analysis by default); use when vetting a skill before install or auditing one in use.
---
Scan the skill at $ARGUMENTS with the skillspector_scan tool (static analysis only, no LLM calls).
To opt into LLM semantic analysis instead, call skillspector_scan with noLlm false and SKILLSPECTOR_PROVIDER/SKILLSPECTOR_MODEL set.
