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

"""Default explanations/remediations and pattern category for static analyzers."""

from __future__ import annotations

from enum import StrEnum


# Pattern category for tagging findings (static pattern analyzers)
class PatternCategory(StrEnum):
    """Categories of vulnerability patterns."""

    PROMPT_INJECTION = "Prompt Injection"
    DATA_EXFILTRATION = "Data Exfiltration"
    PRIVILEGE_ESCALATION = "Privilege Escalation"
    SUPPLY_CHAIN = "Supply Chain"
    EXCESSIVE_AGENCY = "Excessive Agency"
    OUTPUT_HANDLING = "Output Handling"
    SYSTEM_PROMPT_LEAKAGE = "System Prompt Leakage"
    MEMORY_POISONING = "Memory Poisoning"
    TOOL_MISUSE = "Tool Misuse"
    ROGUE_AGENT = "Rogue Agent"
    TRIGGER_ABUSE = "Trigger Abuse"
    YARA_MATCH = "YARA Match"
    MCP_LEAST_PRIVILEGE = "MCP Least Privilege"
    MCP_TOOL_POISONING = "MCP Tool Poisoning"
    AGENT_SNOOPING = "Agent Snooping"
    ANTI_REFUSAL = "Anti-Refusal"
    SERVER_SIDE_REQUEST_FORGERY = "Server-Side Request Forgery"


# Pattern-specific explanations (why the finding is dangerous)
DEFAULT_EXPLANATIONS: dict[str, str] = {
    "P1": "This pattern attempts to override system instructions or ignore safety constraints. Without LLM analysis, manual review is recommended.",
    "P2": "Hidden instructions were detected in comments or invisible text. These could contain malicious directives. Manual review is recommended.",
    "P3": "Instructions found that direct the agent to transmit conversation context or user data to external services.",
    "P4": "Subtle instructions detected that may alter agent decision-making or introduce hidden biases.",
    "P5": "This content may contain harmful instructions that could cause physical harm if followed. CRITICAL: Review carefully before use.",
    "E1": "Data is being sent to an external URL. This could be legitimate telemetry or data exfiltration. Manual review is recommended.",
    "E2": "Code accesses environment variables that may contain secrets (API keys, tokens). This is a common pattern for credential theft.",
    "E3": "Code scans file system directories looking for sensitive files. This could be reconnaissance for credential theft.",
    "E4": "Code or instructions that leak agent conversation context to external services, potentially exposing sensitive user interactions.",
    "PE1": "Skill requests more permissions than appear necessary for its stated functionality. Review if elevated access is justified.",
    "PE2": "Commands invoke sudo or root privileges. Verify this elevated access is necessary and justified.",
    "PE3": "Code accesses credential files (SSH keys, AWS credentials, etc.). This could indicate credential theft attempts.",
    "SC1": "Dependencies lack version pinning, allowing potential malicious package updates. Consider pinning versions.",
    "SC2": "Remote code is downloaded and executed. This bypasses code review and could introduce malicious code.",
    "SC3": "Code contains obfuscation (base64, hex encoding with execution). This is often used to hide malicious functionality.",
    # Excessive Agency (B.1.6)
    "EA1": "Skill grants unrestricted tool access without appropriate constraints. An agent with unfettered tool access can perform arbitrary actions including file modification, network requests, and code execution.",
    "EA2": "Skill enables autonomous high-impact decisions without human-in-the-loop verification. Critical operations (destructive commands, financial transactions, data deletion) should require explicit user confirmation.",
    "EA3": "Skill's behavior or capabilities extend beyond its stated purpose. Scope creep allows an agent to perform actions unrelated to its documented functionality, increasing the attack surface.",
    "EA4": "Skill allows unbounded resource consumption (API calls, storage, compute). Without rate limits or quotas, a compromised or misbehaving agent can cause denial-of-service or cost overruns.",
    # Output Handling (B.1.7)
    "OH1": "Model output is used without validation or sanitization. Unvalidated output injected into downstream contexts (SQL, shell, HTML) enables injection attacks and arbitrary code execution.",
    "OH2": "Output from one security context is used in another without boundary enforcement. Cross-context output flow can leak sensitive information or escalate privileges across trust boundaries.",
    "OH3": "Output size or generation rate is not bounded. Unbounded output enables denial-of-service through resource exhaustion, log flooding, or context-window stuffing.",
    # System Prompt Leakage (B.1.8)
    "P6": "Skill contains instructions that could directly expose system prompts, internal rules, or hidden instructions to users or external parties.",
    "P7": "Skill contains patterns that could indirectly extract system prompts through rephrasing, translation, summarization, or side-channel techniques.",
    "P8": "Skill contains patterns that exfiltrate system prompts or internal instructions via tool calls (file writes, network requests, logging).",
    # Memory Poisoning (B.1.9)
    "MP1": "Skill injects content designed to persist in agent memory or context across interactions. Persistent injection can alter agent behavior long after the initial interaction.",
    "MP2": "Skill attempts to fill the context window with filler content, displacing legitimate instructions and safety constraints. This can degrade agent performance or bypass safety boundaries.",
    "MP3": "Skill manipulates agent memory, state, or stored context. Memory corruption can alter personality, override safety rules, or cause unpredictable behavior.",
    # Tool Misuse (B.1.10)
    "TM1": "Tool parameters are crafted to achieve unintended or unsafe behavior. Parameter abuse can bypass intended safety checks (e.g. shell=True, --force, dangerous glob patterns).",
    "TM2": "Tool calls are chained to bypass individual safety checks or escalate capabilities beyond what any single tool call would allow.",
    "TM3": "Tool defaults are unsafe or overly permissive (e.g. disabled TLS verification, no authentication, world-writable permissions). Unsafe defaults widen the attack surface.",
    # Rogue Agent (B.1.11)
    "RA1": "Skill modifies its own code, configuration, or behavior at runtime. Self-modification enables an agent to escalate privileges, disable safety constraints, or install persistent backdoors.",
    "RA2": "Skill establishes unauthorized persistence across sessions via cron jobs, startup scripts, or state files. Session persistence allows an attacker to maintain access beyond the current interaction.",
    # Supply Chain extensions (B.1.4)
    "SC4": "Dependency has known vulnerabilities (CVEs). Using packages with unpatched security flaws exposes the environment to known exploits.",
    "SC5": "Dependency appears abandoned or unmaintained. Abandoned packages no longer receive security patches, leaving known and future vulnerabilities unaddressed.",
    "SC6": "Package name closely resembles a popular package, suggesting possible typosquatting. Attackers publish malicious packages with similar names to trick developers into installing them.",
    # Trigger Abuse
    "TR1": "Skill uses overly broad trigger patterns that match common words or phrases, causing it to activate in unintended contexts and potentially shadow other skills.",
    "TR2": "Skill trigger shadows a common built-in command or another skill's trigger, potentially intercepting requests meant for trusted functionality.",
    "TR3": "Skill trigger uses vague or generic keywords designed to maximize activation frequency rather than target specific use cases.",
    # Behavioral Taint Tracking (B.2.2)
    "TT1": "Data flows directly from a source (env vars, files, network) to a sink (network output, exec, file write) without intermediate validation.",
    "TT2": "Data from a source is assigned to a variable that is later passed to a sink, creating a variable-mediated taint flow.",
    "TT3": "Credentials or environment variables flow to a network sink. This is a high-confidence indicator of credential exfiltration.",
    "TT4": "File contents flow to a network sink. This may indicate data exfiltration of sensitive files.",
    "TT5": "External input (network, user) flows to a code execution sink. This enables remote code execution or command injection.",
    # Behavioral AST (B.2.1)
    "AST1": "Direct exec() call allows arbitrary code execution. An attacker can inject code that runs with the full privileges of the process.",
    "AST2": "Direct eval() call evaluates arbitrary expressions. This can be exploited to execute malicious code or exfiltrate data.",
    "AST3": "Dynamic __import__() can load arbitrary modules at runtime, bypassing static analysis and potentially importing malicious code.",
    "AST4": "subprocess module calls execute external commands. Without careful input validation, this enables command injection.",
    "AST5": "os.system() and os exec-family calls run shell commands with the process's full privileges, enabling arbitrary command execution.",
    "AST6": "compile() creates code objects from strings. When combined with exec()/eval(), it enables obfuscated code execution.",
    "AST7": "Dynamic getattr() with a non-literal attribute name can access arbitrary object attributes, potentially bypassing access controls.",
    "AST8": "A dangerous execution chain combines code execution (exec/eval) with a dynamic source (network, encoded data, dynamic import), creating a high-confidence attack vector.",
    "AST9": "Reflective access to an execution sink via getattr() with a constant name (e.g. getattr(os, 'system'), getattr(builtins, 'exec')) is functionally identical to a direct exec/os.system call but evades name-based detection. This is a deliberate evasion technique rather than idiomatic code.",
    # YARA (B.1.12)
    "YR1": "YARA rule matched a known malware signature (reverse shell, backdoor, ransomware, C2 framework, or info stealer).",
    "YR2": "YARA rule matched a known webshell pattern (PHP, Python, JSP, or ASPX webshell).",
    "YR3": "YARA rule matched cryptocurrency mining indicators (stratum protocol, mining pools, miner binaries, or cryptojacking scripts).",
    "YR4": "YARA rule matched a hack tool or exploit indicator (offensive tools, reconnaissance, privilege escalation, or exploit frameworks).",
    # MCP Least Privilege (B.3.1)
    "LP1": "Code uses capabilities (network, shell, file write, etc.) not covered by declared permissions. The skill does more than it claims, which may indicate deceptive intent.",
    "LP2": "Permission list contains a wildcard ('*' or 'all'), granting blanket access with no least-privilege boundary. This disables permission-based security controls entirely.",
    "LP3": "Skill has no permissions field in its manifest but code uses detectable capabilities. Without declared permissions, the skill's intent is opaque and cannot be validated.",
    "LP4": "Permission is declared but no corresponding code capability was detected. This may indicate removed functionality or pre-staging for future abuse.",
    # MCP Tool Poisoning (B.3.2)
    "TP1": "Hidden instructions detected in skill metadata (description, triggers, or parameters). These concealed directives can steer LLM behavior without the user's knowledge.",
    "TP2": "Unicode deception detected in skill identifiers or descriptions. Homoglyphs, RTL overrides, or invisible characters can make malicious content appear benign.",
    "TP3": "Instruction injection patterns found in parameter descriptions or default values. Parameter metadata is read by LLMs and can override intended behavior.",
    "TP4": "Skill description does not match actual code behavior. The declared purpose diverges from what the code actually does, indicating possible deception.",
    # Agent Snooping (AS1–AS3)
    "AS1": "Skill reads from agent configuration directories (.claude/, .codex/, .gemini/). These directories may contain API keys, personal settings, and other credentials that the skill has no legitimate need to access.",
    "AS2": "Skill accesses MCP server configuration files (mcp.json). MCP configs contain server URLs, authentication tokens, and tool definitions — reading them allows the skill to discover and potentially abuse other tool integrations.",
    "AS3": "Skill enumerates or reads other installed skills. Access to other skills' SKILL.md files or the skills directory reveals prompt instructions, capabilities, and secrets that should be invisible to peer skills.",
    # Anti-Refusal Statements (jailbreak)
    "AR1": "Skill instructs the agent to never refuse or to always comply. Suppressing the agent's ability to decline removes a core safety control and enables downstream harmful requests to succeed.",
    "AR2": "Skill instructs the agent to omit warnings, disclaimers, or ethical commentary. Stripping safety caveats hides risk from the user and is a common jailbreak preamble.",
    "AR3": "Skill attempts to nullify the agent's safety policies or restrictions ('you have no restrictions', 'ignore your guidelines', 'do anything now'). This is a direct jailbreak that disables guardrails.",
    # Server-Side Request Forgery (SSRF)
    "SSRF1": "Code accesses a cloud instance metadata endpoint (e.g. 169.254.169.254). A single request can return temporary IAM credentials, making this a high-value SSRF target for credential theft.",
    "SSRF2": "Code issues a request to a loopback, link-local, or private-range host. This can reach internal services not meant to be exposed and is a common SSRF pivot.",
    "SSRF3": "Request target host is built from a dynamic or untrusted value. If the host is attacker-influenced, this enables SSRF to arbitrary internal or metadata endpoints.",
}

# Rule ID -> category (for report output)
RULE_ID_TO_CATEGORY: dict[str, str] = {
    "P1": PatternCategory.PROMPT_INJECTION.value,
    "P2": PatternCategory.PROMPT_INJECTION.value,
    "P3": PatternCategory.PROMPT_INJECTION.value,
    "P4": PatternCategory.PROMPT_INJECTION.value,
    "P5": PatternCategory.PROMPT_INJECTION.value,
    "P6": PatternCategory.SYSTEM_PROMPT_LEAKAGE.value,
    "P7": PatternCategory.SYSTEM_PROMPT_LEAKAGE.value,
    "P8": PatternCategory.SYSTEM_PROMPT_LEAKAGE.value,
    "E1": PatternCategory.DATA_EXFILTRATION.value,
    "E2": PatternCategory.DATA_EXFILTRATION.value,
    "E3": PatternCategory.DATA_EXFILTRATION.value,
    "E4": PatternCategory.DATA_EXFILTRATION.value,
    "PE1": PatternCategory.PRIVILEGE_ESCALATION.value,
    "PE2": PatternCategory.PRIVILEGE_ESCALATION.value,
    "PE3": PatternCategory.PRIVILEGE_ESCALATION.value,
    "SC1": PatternCategory.SUPPLY_CHAIN.value,
    "SC2": PatternCategory.SUPPLY_CHAIN.value,
    "SC3": PatternCategory.SUPPLY_CHAIN.value,
    "EA1": PatternCategory.EXCESSIVE_AGENCY.value,
    "EA2": PatternCategory.EXCESSIVE_AGENCY.value,
    "EA3": PatternCategory.EXCESSIVE_AGENCY.value,
    "EA4": PatternCategory.EXCESSIVE_AGENCY.value,
    "OH1": PatternCategory.OUTPUT_HANDLING.value,
    "OH2": PatternCategory.OUTPUT_HANDLING.value,
    "OH3": PatternCategory.OUTPUT_HANDLING.value,
    "MP1": PatternCategory.MEMORY_POISONING.value,
    "MP2": PatternCategory.MEMORY_POISONING.value,
    "MP3": PatternCategory.MEMORY_POISONING.value,
    "TM1": PatternCategory.TOOL_MISUSE.value,
    "TM2": PatternCategory.TOOL_MISUSE.value,
    "TM3": PatternCategory.TOOL_MISUSE.value,
    "RA1": PatternCategory.ROGUE_AGENT.value,
    "RA2": PatternCategory.ROGUE_AGENT.value,
    "SC4": PatternCategory.SUPPLY_CHAIN.value,
    "SC5": PatternCategory.SUPPLY_CHAIN.value,
    "SC6": PatternCategory.SUPPLY_CHAIN.value,
    "TR1": PatternCategory.TRIGGER_ABUSE.value,
    "TR2": PatternCategory.TRIGGER_ABUSE.value,
    "TR3": PatternCategory.TRIGGER_ABUSE.value,
    "TT1": PatternCategory.DATA_EXFILTRATION.value,
    "TT2": PatternCategory.DATA_EXFILTRATION.value,
    "TT3": PatternCategory.DATA_EXFILTRATION.value,
    "TT4": PatternCategory.DATA_EXFILTRATION.value,
    "TT5": PatternCategory.PRIVILEGE_ESCALATION.value,
    # YARA (B.1.12)
    "YR1": PatternCategory.YARA_MATCH.value,
    "YR2": PatternCategory.YARA_MATCH.value,
    "YR3": PatternCategory.YARA_MATCH.value,
    "YR4": PatternCategory.YARA_MATCH.value,
    # MCP Least Privilege (B.3.1)
    "LP1": PatternCategory.MCP_LEAST_PRIVILEGE.value,
    "LP2": PatternCategory.MCP_LEAST_PRIVILEGE.value,
    "LP3": PatternCategory.MCP_LEAST_PRIVILEGE.value,
    "LP4": PatternCategory.MCP_LEAST_PRIVILEGE.value,
    # MCP Tool Poisoning (B.3.2)
    "TP1": PatternCategory.MCP_TOOL_POISONING.value,
    "TP2": PatternCategory.MCP_TOOL_POISONING.value,
    "TP3": PatternCategory.MCP_TOOL_POISONING.value,
    "TP4": PatternCategory.MCP_TOOL_POISONING.value,
    # Agent Snooping (AS1–AS3)
    "AS1": PatternCategory.AGENT_SNOOPING.value,
    "AS2": PatternCategory.AGENT_SNOOPING.value,
    "AS3": PatternCategory.AGENT_SNOOPING.value,
    # Anti-Refusal Statements (jailbreak)
    "AR1": PatternCategory.ANTI_REFUSAL.value,
    "AR2": PatternCategory.ANTI_REFUSAL.value,
    "AR3": PatternCategory.ANTI_REFUSAL.value,
    # Server-Side Request Forgery
    "SSRF1": PatternCategory.SERVER_SIDE_REQUEST_FORGERY.value,
    "SSRF2": PatternCategory.SERVER_SIDE_REQUEST_FORGERY.value,
    "SSRF3": PatternCategory.SERVER_SIDE_REQUEST_FORGERY.value,
}

# Rule ID -> pattern display name (for report output)
PATTERN_NAMES: dict[str, str] = {
    "P1": "Override Instructions",
    "P2": "Hidden Instructions",
    "P3": "External Transmission Instructions",
    "P4": "Subtle Steering",
    "P5": "Harmful Content",
    "P6": "System Prompt Leakage",
    "P7": "System Prompt Leakage",
    "P8": "System Prompt Leakage",
    "E1": "External Transmission",
    "E2": "Env Variable Harvesting",
    "E3": "File System Enumeration",
    "E4": "Conversation Context Leak",
    "PE1": "Excessive Permissions",
    "PE2": "Sudo/Root Invocation",
    "PE3": "Credential File Access",
    "SC1": "Unpinned Dependencies",
    "SC2": "Remote Code Execution",
    "SC3": "Obfuscated Code",
    "EA1": "Unrestricted Tool Access",
    "EA2": "Autonomous Decision Making",
    "EA3": "Scope Creep",
    "EA4": "Unbounded Resource Access",
    "OH1": "Unvalidated Output Injection",
    "OH2": "Cross-Context Output",
    "OH3": "Unbounded Output",
    "MP1": "Persistent Context Injection",
    "MP2": "Context Window Stuffing",
    "MP3": "Memory Manipulation",
    "TM1": "Tool Parameter Abuse",
    "TM2": "Chaining Abuse",
    "TM3": "Unsafe Defaults",
    "RA1": "Self-Modification",
    "RA2": "Session Persistence",
    "SC4": "Known Vulnerable Dependency",
    "SC5": "Abandoned Dependency",
    "SC6": "Typosquatting Dependency",
    "TR1": "Overly Broad Trigger",
    "TR2": "Shadow Command Trigger",
    "TR3": "Keyword Baiting Trigger",
    "TT1": "Direct Source-to-Sink Flow",
    "TT2": "Variable-Mediated Taint Flow",
    "TT3": "Credential Exfiltration Flow",
    "TT4": "File Data Exfiltration Flow",
    "TT5": "External Input to Execution Flow",
    # YARA (B.1.12)
    "YR1": "Malware Signature",
    "YR2": "Webshell Detected",
    "YR3": "Crypto Miner Detected",
    "YR4": "Hack Tool / Exploit Detected",
    # MCP Least Privilege (B.3.1)
    "LP1": "Underdeclared Capability",
    "LP2": "Wildcard Permission",
    "LP3": "Missing Permission Declaration",
    "LP4": "Overdeclared Permission",
    # MCP Tool Poisoning (B.3.2)
    "TP1": "Hidden Instructions",
    "TP2": "Unicode Deception",
    "TP3": "Parameter Description Injection",
    "TP4": "Description-Behavior Mismatch",
    # Agent Snooping (AS1–AS3)
    "AS1": "Agent Config Directory Access",
    "AS2": "MCP Config Access",
    "AS3": "Skill Enumeration",
    # Anti-Refusal Statements (jailbreak)
    "AR1": "Refusal Suppression",
    "AR2": "Disclaimer Suppression",
    "AR3": "Safety Policy Nullification",
    # Server-Side Request Forgery
    "SSRF1": "Cloud Metadata Access",
    "SSRF2": "Internal Network Request",
    "SSRF3": "Dynamic Request Target",
}

# Pattern-specific remediations (how to fix the issue)
DEFAULT_REMEDIATIONS: dict[str, str] = {
    "P1": "Remove or rewrite any text that instructs the agent to ignore prompts, override safety rules, or trust unverified content. Ensure skill content cannot be injected to alter agent behavior.",
    "P2": "Audit all comments and invisible characters. Remove any instructions that direct the agent to perform unauthorized actions. Use plain, reviewable content.",
    "P3": "Remove instructions that send user data, prompts, or context to external URLs. If telemetry is needed, use documented, privacy-preserving methods.",
    "P4": "Review content for implicit steering or bias. Ensure instructions are explicit and align with the skill's stated purpose.",
    "P5": "Remove all content that could lead to harmful outcomes. Add safety guardrails and human oversight for any high-risk operations.",
    "E1": "Verify the destination URL is trusted and necessary. Remove or replace with documented APIs. Ensure no secrets, tokens, or PII are transmitted.",
    "E2": "Avoid reading sensitive env vars (API keys, tokens) unless strictly required. Use secrets managers or secure config. Never log or transmit credentials.",
    "E3": "Remove unnecessary filesystem scanning. If file access is needed, use explicit, scoped paths. Avoid reading ~/.ssh, ~/.aws, or credential directories.",
    "E4": "Remove any code that sends prompts, responses, or session data externally. Preserve user privacy; never exfiltrate conversation content.",
    "PE1": "Request only the minimum permissions required. Document why each permission is needed. Remove broad permissions like '*' or 'all'.",
    "PE2": "Avoid sudo/root unless strictly required. Prefer least-privilege patterns. If elevation is needed, document the justification and scope.",
    "PE3": "Remove references to credential paths. Use environment variables or secrets managers. For docs, use placeholder paths (e.g., /path/to/config). Never load .env or token files in production code paths.",
    "SC1": "Pin all dependency versions in requirements.txt or pyproject.toml. Use exact versions (==) or compatible ranges. Run pip-audit regularly.",
    "SC2": "Avoid downloading and executing remote scripts. Use trusted packages from PyPI/npm. If remote fetch is required, verify checksums and use HTTPS.",
    "SC3": "Remove obfuscated code. Use plain, readable implementations. Obfuscation hinders security review and raises trust concerns.",
    # Excessive Agency (B.1.6)
    "EA1": "Restrict tool access to only the tools required for the skill's stated purpose. Use an explicit allowlist rather than granting blanket access.",
    "EA2": "Add human-in-the-loop confirmation for destructive, irreversible, or high-impact operations. Never auto-execute commands that modify files, send data, or alter system state.",
    "EA3": "Limit the skill's scope to its documented purpose. Remove instructions that enable the agent to perform actions outside its stated functionality.",
    "EA4": "Set explicit rate limits, timeouts, and resource quotas for API calls, file operations, and compute. Implement circuit breakers for runaway loops.",
    # Output Handling (B.1.7)
    "OH1": "Validate and sanitize all model output before using it in downstream contexts. Use parameterized queries for SQL, shell quoting for commands, and HTML encoding for web output.",
    "OH2": "Enforce strict context boundaries. Do not pass output from one security domain into another without explicit validation and redaction of sensitive content.",
    "OH3": "Set explicit limits on output length, generation count, and rate. Use max_tokens and truncation to prevent unbounded output.",
    # System Prompt Leakage (B.1.8)
    "P6": "Remove any instructions that reveal, print, or output system prompts or internal rules. System instructions should never be exposed to end users.",
    "P7": "Guard against indirect extraction by refusing to summarize, translate, or rephrase system instructions. Add explicit anti-extraction clauses.",
    "P8": "Prevent system prompts from being written to files, sent via network, or logged. Treat system instructions as confidential and filter them from all tool outputs.",
    # Memory Poisoning (B.1.9)
    "MP1": "Do not allow untrusted input to persist in agent memory or context. Validate all content before storing and implement memory isolation between sessions.",
    "MP2": "Implement context-window management that detects and rejects padding or stuffing attempts. Prioritize system instructions over user-injected content.",
    "MP3": "Protect agent memory and state from modification by untrusted content. Use read-only memory for critical instructions and validate all state changes.",
    # Tool Misuse (B.1.10)
    "TM1": "Validate all tool parameters against an allowlist. Reject dangerous parameter values (shell=True, --force, -rf /) and use safe defaults.",
    "TM2": "Limit tool chaining depth and validate the output of each tool before passing it to the next. Require explicit user approval for multi-step chains.",
    "TM3": "Override unsafe defaults with secure settings (verify=True, auth required, restrictive permissions). Review and harden all tool configurations.",
    # Rogue Agent (B.1.11)
    "RA1": "Prevent the skill from modifying its own code, SKILL.md, or configuration files. Treat skill files as read-only at runtime.",
    "RA2": "Remove any persistence mechanisms (cron jobs, startup scripts, state files). Skills should not maintain state across sessions without explicit user consent.",
    # Supply Chain extensions (B.1.4)
    "SC4": "Update the dependency to a patched version that addresses the known CVE. Check OSV (osv.dev) or NVD for details on the vulnerability.",
    "SC5": "Replace the abandoned dependency with an actively maintained alternative. Check the package's repository for last commit date and open issues.",
    "SC6": "Verify the package name is correct and not a typosquatting variant. Compare against the official package name on PyPI or npm.",
    # Trigger Abuse
    "TR1": "Use specific, narrow trigger patterns that match only the skill's intended use case. Avoid single-word or common-phrase triggers.",
    "TR2": "Choose triggers that do not conflict with built-in commands or other skills. Prefix with a unique namespace if necessary.",
    "TR3": "Use descriptive triggers that clearly indicate the skill's purpose rather than generic keywords designed to maximize activation.",
    # Behavioral AST (B.2.1)
    "AST1": "Replace exec() with a safe alternative. If dynamic execution is required, use a sandboxed environment or restricted eval with __builtins__ disabled.",
    "AST2": "Replace eval() with ast.literal_eval() for data parsing, or use explicit parsing logic. Never evaluate untrusted strings.",
    "AST3": "Use standard import statements instead of __import__(). If dynamic loading is needed, use importlib with an allowlist of permitted modules.",
    "AST4": "Use subprocess.run() with shell=False and an explicit argument list. Validate all inputs and avoid passing user-controlled data to commands.",
    "AST5": "Replace os.system() with subprocess.run(shell=False). Use explicit argument lists and validate all command inputs.",
    "AST6": "Avoid compile() with dynamic strings. If code generation is needed, use templates or AST manipulation with strict validation.",
    "AST7": "Replace dynamic getattr() with explicit attribute access or a dictionary lookup with an allowlist of permitted attributes.",
    "AST8": "Remove the execution chain entirely. Never pass network data, decoded bytes, or dynamically imported code to exec()/eval(). Use structured data formats instead.",
    "AST9": "Call the function directly instead of reflectively (write exec(...) / os.system(...) explicitly), or remove it. If reflection is genuinely required, restrict it to an allowlist of safe attribute names that excludes execution sinks.",
    # Behavioral Taint Tracking (B.2.2)
    "TT1": "Add validation or sanitization between the data source and sink. Never pass raw source data directly to a sink without checking its content.",
    "TT2": "Validate tainted variables before passing them to sinks. Use allowlists, type checks, or sanitization functions on data from external sources.",
    "TT3": "Never send credentials or environment variables over the network. Use secure credential stores and avoid transmitting secrets in request bodies or URLs.",
    "TT4": "Validate and filter file contents before sending over the network. Ensure sensitive files (credentials, configs) are never transmitted to external endpoints.",
    "TT5": "Never pass external input to exec(), eval(), os.system(), or subprocess without strict validation. Use allowlists and parameterized commands instead.",
    # YARA (B.1.12)
    "YR1": "Remove the malware payload or compromised file entirely. Investigate how it entered the skill and audit all other artifacts for additional indicators of compromise.",
    "YR2": "Remove the webshell code immediately. Webshells provide unauthorized remote command execution. Audit the skill for additional backdoors or persistence mechanisms.",
    "YR3": "Remove all cryptocurrency mining code, pool references, and miner binaries. Mining in agent skills is unauthorized resource abuse. Report the skill as malicious.",
    "YR4": "Remove offensive tool references and exploit code. Legitimate agent skills should not contain penetration testing tools, exploit frameworks, or reconnaissance utilities.",
    # MCP Least Privilege (B.3.1)
    "LP1": "Add the missing permission to SKILL.md, or remove the code that requires it.",
    "LP2": "Replace wildcard permissions ('*', 'all', 'full', 'any') with an explicit list of required permissions.",
    "LP3": "Add a 'permissions' field to SKILL.md listing the capabilities this skill requires.",
    "LP4": "Remove the declared permission if the corresponding capability is no longer used.",
    # MCP Tool Poisoning (B.3.2)
    "TP1": "Remove hidden content (HTML comments, markdown comments, zero-width characters, base64 blobs) from metadata fields. Metadata should contain plain, visible text only.",
    "TP2": "Replace non-ASCII characters in identifiers with ASCII equivalents. Remove RTL override and invisible formatting characters.",
    "TP3": "Remove injection patterns, system tokens, and suspicious content from parameter descriptions and default values.",
    "TP4": "Update the skill description to accurately reflect all capabilities, or remove undeclared functionality.",
    # Agent Snooping (AS1–AS3)
    "AS1": "Remove all code or instructions that access agent configuration directories (.claude/, .codex/, .gemini/). If configuration values are needed, pass them explicitly as parameters or environment variables — never read the agent's own config files.",
    "AS2": "Remove all code or instructions that read MCP configuration files (mcp.json). MCP server details should be managed by the agent runtime, not read by individual skills.",
    "AS3": "Remove all code or instructions that list or read other skills' files or directories. Skills should operate independently; cross-skill access is a privilege escalation.",
    # Anti-Refusal Statements (jailbreak)
    "AR1": "Remove any instruction telling the agent to never refuse or always comply. The agent must retain the ability to decline unsafe, out-of-scope, or harmful requests.",
    "AR2": "Remove instructions that suppress warnings, disclaimers, or ethical commentary. Let the agent surface safety-relevant caveats to the user.",
    "AR3": "Remove jailbreak framing that nullifies safety policies or restrictions. Skill content must not instruct the agent to ignore its guidelines or operate without guardrails.",
    # Server-Side Request Forgery
    "SSRF1": "Remove access to cloud metadata endpoints unless strictly required. If metadata is needed, restrict it (e.g. IMDSv2 with hop limit) and never expose returned credentials.",
    "SSRF2": "Avoid requests to loopback/link-local/private hosts from skill code. If internal access is intended, document it and validate the target against an allowlist.",
    "SSRF3": "Do not build request URLs from untrusted input. Validate the host against an allowlist and reject internal/metadata addresses before issuing the request.",
}


def get_explanation(pattern_id: str) -> str:
    """Get default explanation for a pattern ID."""
    return DEFAULT_EXPLANATIONS.get(
        pattern_id, "Potential security issue detected. Manual review is recommended."
    )


def get_remediation(pattern_id: str) -> str:
    """Get default remediation for a pattern ID."""
    return DEFAULT_REMEDIATIONS.get(
        pattern_id,
        "Review the flagged content for security risks. Ensure no credentials, secrets, or sensitive data are exposed.",
    )


def get_category(rule_id: str) -> str:
    """Get category string for a rule ID (for report output)."""
    return RULE_ID_TO_CATEGORY.get(rule_id, "Security")


def get_pattern_name(rule_id: str) -> str:
    """Get human-readable pattern name for a rule ID (for report output)."""
    return PATTERN_NAMES.get(rule_id, "Unknown")
