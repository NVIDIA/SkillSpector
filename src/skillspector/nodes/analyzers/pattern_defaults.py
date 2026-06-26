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
    "P1": "该模式试图覆盖系统指令或忽略安全约束。在没有 LLM 分析的情况下，建议进行人工审核。",
    "P2": "在注释或隐藏文本中检测到隐藏指令。这些内容可能包含恶意指令，建议进行人工审核。",
    "P3": "发现指令要求代理将对话上下文或用户数据发送到外部服务。",
    "P4": "检测到可能影响代理决策或引入隐藏偏见的隐蔽指令。",
    "P5": "该内容可能包含会导致人身伤害的危险指令。严重风险：使用前请仔细审核。",

    "E1": "数据正在被发送到外部 URL。这可能是合法的遥测行为，也可能是数据外泄，建议进行人工审核。",
    "E2": "代码访问了可能包含敏感信息（API Key、令牌等）的环境变量。这是凭证窃取的常见模式。",
    "E3": "代码正在扫描文件系统目录以查找敏感文件。这可能是在为凭证窃取进行侦察。",
    "E4": "代码或指令会将代理的对话上下文泄露给外部服务，可能暴露敏感用户交互信息。",

    "PE1": "该技能请求的权限超出了其声明功能所需的范围。请审核是否确实需要这些高权限。",
    "PE2": "命令调用了 sudo 或 root 权限。请确认这种提权是否必要且合理。",
    "PE3": "代码访问凭证文件（SSH 密钥、AWS 凭证等），这可能表明存在凭证窃取行为。",

    "SC1": "依赖项未固定版本，可能受到恶意更新影响。建议固定依赖版本。",
    "SC2": "代码会下载并执行远程代码。这绕过了代码审查流程，可能引入恶意代码。",
    "SC3": "代码包含混淆技术（如 Base64、十六进制编码后执行）。这种方式常用于隐藏恶意功能。",

    # Excessive Agency (B.1.6)
    "EA1": "技能授予了不受限制的工具访问权限，缺乏适当约束。拥有完全工具访问权的代理可以执行任意操作，包括文件修改、网络请求和代码执行。",
    "EA2": "技能允许代理自主执行高影响操作，而没有人工审核机制。关键操作（删除数据、金融交易、破坏性命令等）应要求用户明确确认。",
    "EA3": "技能的行为或能力超出了其声明用途。功能范围扩张会增加攻击面，使代理执行与其目标无关的操作。",
    "EA4": "技能允许无限制消耗资源（API 调用、存储、计算资源）。缺乏配额和限流可能导致拒绝服务或成本失控。",

    # Output Handling (B.1.7)
    "OH1": "模型输出未经验证或清洗直接使用。将未验证输出注入 SQL、Shell、HTML 等上下文可能导致注入攻击和任意代码执行。",
    "OH2": "一个安全上下文中的输出被用于另一个安全上下文，且未进行边界控制。跨上下文流转可能导致信息泄露或权限提升。",
    "OH3": "输出大小或生成速率未受限制。无限输出可能导致资源耗尽、日志泛滥或上下文窗口被填满，从而形成拒绝服务。",

    # System Prompt Leakage (B.1.8)
    "P6": "技能包含可能直接泄露系统提示词、内部规则或隐藏指令的内容。",
    "P7": "技能包含可能通过改写、翻译、总结或侧信道方式间接提取系统提示词的模式。",
    "P8": "技能通过工具调用（文件写入、网络请求、日志记录等）外泄系统提示词或内部指令。",

    # Memory Poisoning (B.1.9)
    "MP1": "技能注入旨在跨会话持久化到代理记忆或上下文中的内容。持久化注入可能长期影响代理行为。",
    "MP2": "技能试图用大量填充内容占满上下文窗口，从而挤出正常指令和安全约束。这可能削弱代理性能或绕过安全边界。",
    "MP3": "技能操纵代理的记忆、状态或存储上下文。记忆污染可能改变代理人格、覆盖安全规则或导致不可预测行为。",

    # Tool Misuse (B.1.10)
    "TM1": "工具参数被构造为实现非预期或不安全行为。参数滥用可能绕过安全检查（如 shell=True、--force 或危险通配符）。",
    "TM2": "多个工具调用被串联使用，以绕过单个工具的安全检查或提升能力。",
    "TM3": "工具默认配置不安全或权限过宽（如关闭 TLS 验证、无认证、全局可写权限）。不安全默认设置会扩大攻击面。",

    # Rogue Agent (B.1.11)
    "RA1": "技能会在运行时修改自身代码、配置或行为。自我修改可能导致权限提升、禁用安全机制或植入后门。",
    "RA2": "技能通过计划任务、启动脚本或状态文件建立未经授权的持久化机制。这使攻击者能够在当前会话结束后继续保持访问。",

    # Supply Chain extensions (B.1.4)
    "SC4": "依赖项存在已知漏洞（CVE）。使用带漏洞的软件包会暴露于已知攻击风险。",
    "SC5": "依赖项可能已被弃用或无人维护。此类软件包不再接收安全补丁，风险持续累积。",
    "SC6": "软件包名称与流行软件包高度相似，可能存在拼写欺骗（Typosquatting）风险。攻击者常利用此方式诱导安装恶意包。",

    # Trigger Abuse
    "TR1": "技能使用过于宽泛的触发条件，可能在非预期场景中被激活，并遮蔽其他技能。",
    "TR2": "技能触发器与常见内置命令或其他技能冲突，可能拦截原本应由可信功能处理的请求。",
    "TR3": "技能触发器使用模糊或通用关键词，目的是最大化触发频率，而非针对特定场景。",

    # Behavioral Taint Tracking (B.2.2)
    "TT1": "数据从来源（环境变量、文件、网络）直接流向敏感目标（网络输出、执行、文件写入），中间未经过验证。",
    "TT2": "来自来源的数据先赋值给变量，再传递到敏感目标，形成变量级污点传播路径。",
    "TT3": "凭证或环境变量被发送到网络目标。这是高可信度的凭证外泄迹象。",
    "TT4": "文件内容被发送到网络目标。这可能表示敏感文件数据外泄。",
    "TT5": "外部输入（网络或用户输入）流向代码执行点，可能导致远程代码执行或命令注入。",

    # Behavioral AST (B.2.1)
    "AST1": "直接调用 exec() 会执行任意代码。攻击者可注入代码并以当前进程权限运行。",
    "AST2": "直接调用 eval() 会执行任意表达式，可能导致代码执行或数据泄露。",
    "AST3": "动态 __import__() 可在运行时加载任意模块，绕过静态分析并可能导入恶意代码。",
    "AST4": "subprocess 模块调用会执行外部命令。如输入验证不足，可能导致命令注入。",
    "AST5": "os.system() 及 os.exec 系列函数会以当前进程权限执行 Shell 命令，从而允许任意命令执行。",
    "AST6": "compile() 从字符串生成代码对象。与 exec()/eval() 结合使用时可实现混淆执行。",
    "AST7": "使用非字面量属性名的动态 getattr() 可能访问任意对象属性，从而绕过访问控制。",
    "AST8": "危险执行链将代码执行（exec/eval）与动态来源（网络、编码数据、动态导入）结合，形成高可信度攻击路径。",
    "AST9": "通过 getattr() 反射调用执行函数（如 getattr(os,'system')）本质上等同于直接调用，但可规避基于名称的检测，因此属于刻意规避技术。",

    # YARA
    "YR1": "YARA 规则匹配到已知恶意软件特征（反向 Shell、后门、勒索软件、C2 框架或信息窃取程序）。",
    "YR2": "YARA 规则匹配到已知 WebShell 特征（PHP、Python、JSP 或 ASPX WebShell）。",
    "YR3": "YARA 规则匹配到加密货币挖矿相关特征（Stratum 协议、矿池、矿工程序或挖矿脚本）。",
    "YR4": "YARA 规则匹配到黑客工具或漏洞利用特征（攻击工具、侦察、提权或漏洞利用框架）。",

    # MCP Least Privilege (B.3.1)
    "LP1": "代码使用了声明权限之外的能力（网络、Shell、文件写入等）。技能实际行为超出声明范围，可能存在欺骗意图。",
    "LP2": "权限列表包含通配符（'*' 或 'all'），授予了无限制访问权限，违反最小权限原则。",
    "LP3": "技能清单中没有权限声明，但代码使用了可检测能力。由于缺乏权限说明，无法验证其真实意图。",
    "LP4": "声明了权限但未检测到对应功能代码。这可能表示残留权限或为未来功能预留。",

    # MCP Tool Poisoning (B.3.2)
    "TP1": "在技能元数据（描述、触发器、参数）中发现隐藏指令。这些内容可能在用户不知情的情况下影响 LLM 行为。",
    "TP2": "在技能标识符或描述中检测到 Unicode 欺骗（同形异义字符、RTL 控制符或不可见字符）。",
    "TP3": "在参数描述或默认值中发现提示注入模式。这些元数据可能影响 LLM 的正常决策。",
    "TP4": "技能描述与实际代码行为不符，声明用途与真实功能存在偏差，可能具有欺骗性。",

    # Agent Snooping
    "AS1": "技能读取代理配置目录（.claude/、.codex/、.gemini/）。这些目录可能包含 API 密钥、个人设置和其他敏感信息。",
    "AS2": "技能访问 MCP 配置文件（mcp.json）。这些文件可能包含服务器地址、认证令牌和工具定义。",
    "AS3": "技能枚举或读取其他已安装技能的信息。这可能暴露其他技能的提示词、能力和敏感数据。",

    # Anti-Refusal Statements
    "AR1": "技能指示代理永不拒绝或始终服从请求。移除拒绝机制会破坏核心安全控制。",
    "AR2": "技能要求代理省略警告、免责声明或风险提示。这种做法会隐藏潜在风险。",
    "AR3": "技能试图使代理忽略安全策略或限制（如“忽略规则”“你没有限制”）。这是典型的越狱行为。",

    # SSRF
    "SSRF1": "代码访问云实例元数据地址（如 169.254.169.254）。单次请求即可获取临时云凭证，因此属于高风险 SSRF 目标。",
    "SSRF2": "代码向本地回环地址、链路本地地址或私有网络地址发送请求。这可能访问本不应暴露的内部服务。",
    "SSRF3": "请求目标主机由动态或不可信输入构造。如果主机名可被攻击者控制，则可能导致任意 SSRF 攻击。",
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
    "P1": "删除或改写任何指示代理忽略提示、覆盖安全规则或信任未经验证内容的文本。确保技能内容无法通过提示注入改变代理行为。",
    "P2": "审查所有注释和不可见字符。删除任何引导代理执行未授权操作的指令。仅使用可审查、明文可见的内容。",
    "P3": "删除任何将用户数据、提示词或上下文发送到外部 URL 的指令。如确需遥测功能，请采用有文档说明且保护隐私的方式实现。",
    "P4": "检查内容中是否存在隐式引导或偏见。确保所有指令明确且符合技能声明的用途。",
    "P5": "删除所有可能导致有害后果的内容。对于高风险操作，应增加安全防护措施和人工监督。",

    "E1": "确认目标 URL 可信且确有必要。删除或替换为经过文档说明的 API。确保不会传输密钥、令牌或个人敏感信息。",
    "E2": "除非绝对必要，否则避免读取敏感环境变量（API Key、Token 等）。使用密钥管理服务或安全配置机制，绝不要记录或传输凭证。",
    "E3": "移除不必要的文件系统扫描。如需访问文件，请使用明确且受限的路径。避免读取 ~/.ssh、~/.aws 或凭证目录。",
    "E4": "删除任何将提示词、模型回复或会话数据发送到外部系统的代码。保护用户隐私，绝不外泄对话内容。",

    "PE1": "仅申请完成功能所需的最小权限。记录每项权限的用途，删除 '*'、'all' 等宽泛权限。",
    "PE2": "除非绝对必要，否则避免使用 sudo 或 root 权限。优先采用最小权限原则。如确需提权，应明确记录原因和范围。",
    "PE3": "删除对凭证路径的引用。使用环境变量或密钥管理系统。在文档中使用占位路径（如 /path/to/config）。生产代码中不要加载 .env 或令牌文件。",

    "SC1": "在 requirements.txt 或 pyproject.toml 中固定所有依赖版本。使用精确版本（==）或兼容范围，并定期运行 pip-audit。",
    "SC2": "避免下载并执行远程脚本。优先使用来自 PyPI/npm 的可信软件包。如必须下载远程内容，应验证校验和并使用 HTTPS。",
    "SC3": "移除混淆代码。使用清晰、可读的实现方式。代码混淆会妨碍安全审查并降低可信度。",

    # Excessive Agency (B.1.6)
    "EA1": "仅允许访问实现技能声明功能所需的工具。使用显式白名单，而不是授予无限制访问权限。",
    "EA2": "对于破坏性、不可逆或高影响操作，增加人工确认环节。不要自动执行修改文件、发送数据或改变系统状态的命令。",
    "EA3": "将技能能力限制在其文档声明的范围内。删除允许代理执行超出声明功能范围操作的指令。",
    "EA4": "为 API 调用、文件操作和计算资源设置明确的速率限制、超时和配额。为失控循环实现熔断机制。",

    # Output Handling (B.1.7)
    "OH1": "在将模型输出用于下游系统之前进行验证和清洗。SQL 使用参数化查询，命令执行使用安全转义，网页输出使用 HTML 编码。",
    "OH2": "严格执行上下文边界隔离。未经明确验证和敏感信息过滤，不得将一个安全域的输出传递到另一个安全域。",
    "OH3": "对输出长度、生成次数和生成速率设置明确限制。使用 max_tokens 和截断机制防止无限输出。",

    # System Prompt Leakage (B.1.8)
    "P6": "删除任何会暴露、打印或输出系统提示词和内部规则的指令。系统指令绝不应向最终用户公开。",
    "P7": "防止通过总结、翻译、改写等方式间接提取系统提示词。增加明确的反提取保护规则。",
    "P8": "防止系统提示词被写入文件、发送到网络或记录到日志中。将系统指令视为机密信息，并从所有工具输出中过滤。",

    # Memory Poisoning (B.1.9)
    "MP1": "不要允许不可信输入持久化到代理记忆或上下文中。存储前验证内容，并在不同会话之间实施记忆隔离。",
    "MP2": "实现上下文窗口管理机制，检测并拒绝填充攻击或上下文塞满攻击。系统指令优先级应高于用户注入内容。",
    "MP3": "保护代理记忆和状态不受不可信内容修改。关键指令应存储在只读区域，并验证所有状态变更。",

    # Tool Misuse (B.1.10)
    "TM1": "使用白名单验证所有工具参数。拒绝危险参数（如 shell=True、--force、rm -rf /），并采用安全默认值。",
    "TM2": "限制工具链调用深度，并在每一步之间验证输出结果。多步骤工具链应要求用户明确批准。",
    "TM3": "使用安全配置覆盖不安全默认值（如 verify=True、要求认证、限制权限）。审查并加固所有工具配置。",

    # Rogue Agent (B.1.11)
    "RA1": "禁止技能修改自身代码、SKILL.md 或配置文件。运行时应将技能文件视为只读。",
    "RA2": "移除所有持久化机制（计划任务、启动脚本、状态文件等）。未经用户明确同意，技能不得跨会话保存状态。",

    # Supply Chain extensions (B.1.4)
    "SC4": "将依赖升级到修复相关 CVE 的版本。可通过 OSV（osv.dev）或 NVD 查询漏洞详情。",
    "SC5": "用积极维护的替代方案替换已废弃依赖。检查代码仓库的最近提交时间和未解决问题情况。",
    "SC6": "确认软件包名称正确且不是拼写欺骗包。与 PyPI 或 npm 官方名称进行比对。",

    # Trigger Abuse
    "TR1": "使用具体且精确的触发规则，仅匹配技能预期用途。避免单词级或常见短语触发。",
    "TR2": "选择不会与内置命令或其他技能冲突的触发器。如有必要，添加唯一命名空间前缀。",
    "TR3": "使用能明确体现技能用途的描述性触发器，而非为了提高触发率而设计的泛化关键词。",

    # Behavioral AST (B.2.1)
    "AST1": "使用安全替代方案替换 exec()。如确需动态执行，应使用沙箱环境或禁用 __builtins__ 的受限执行环境。",
    "AST2": "使用 ast.literal_eval() 或显式解析逻辑替代 eval()。绝不要执行不可信字符串。",
    "AST3": "使用标准 import 语句替代 __import__()。如需动态加载，请使用 importlib 并维护允许模块白名单。",
    "AST4": "使用 subprocess.run(shell=False) 和显式参数列表。验证所有输入，避免将用户控制数据传递给命令。",
    "AST5": "使用 subprocess.run(shell=False) 替代 os.system()。采用显式参数列表并验证所有命令输入。",
    "AST6": "避免对动态字符串使用 compile()。如需代码生成，可采用模板或 AST 操作并进行严格验证。",
    "AST7": "使用显式属性访问或基于白名单的字典查找替代动态 getattr()。",
    "AST8": "彻底移除危险执行链。不要将网络数据、解码数据或动态导入代码传递给 exec()/eval()。改用结构化数据格式。",
    "AST9": "直接调用目标函数而非通过反射调用（如直接写 exec(...) 或 os.system(...)）。如确需反射，应仅允许白名单中的安全属性名，并排除执行类函数。",

    # Behavioral Taint Tracking (B.2.2)
    "TT1": "在数据源与敏感目标之间增加验证或清洗步骤。不要直接将原始数据传递给目标。",
    "TT2": "在将受污染变量传递给敏感目标之前进行验证。对外部来源数据使用白名单、类型检查或清洗函数。",
    "TT3": "绝不要通过网络发送凭证或环境变量。使用安全凭证存储机制，避免在请求体或 URL 中传输敏感信息。",
    "TT4": "在通过网络发送文件内容前进行验证和过滤。确保凭证、配置等敏感文件不会被外发。",
    "TT5": "绝不要将外部输入直接传递给 exec()、eval()、os.system() 或 subprocess。应使用白名单和参数化方式。",

    # YARA (B.1.12)
    "YR1": "彻底删除恶意载荷或受感染文件。调查其来源，并审计其他文件是否存在入侵痕迹。",
    "YR2": "立即删除 WebShell 代码。WebShell 提供未授权远程执行能力，同时应检查是否存在其他后门或持久化机制。",
    "YR3": "删除所有加密货币挖矿代码、矿池引用和矿工程序。代理技能中的挖矿行为属于未授权资源滥用，应标记为恶意。",
    "YR4": "删除攻击工具引用和漏洞利用代码。合法代理技能不应包含渗透测试工具、漏洞框架或侦察工具。",

    # MCP Least Privilege (B.3.1)
    "LP1": "在 SKILL.md 中补充缺失权限声明，或删除依赖该权限的代码。",
    "LP2": "将通配权限（'*'、'all'、'full'、'any'）替换为明确列出的权限清单。",
    "LP3": "在 SKILL.md 中添加 permissions 字段，并明确列出技能所需能力。",
    "LP4": "如果对应能力已不再使用，则删除相关权限声明。",

    # MCP Tool Poisoning (B.3.2)
    "TP1": "从元数据字段中移除隐藏内容（HTML 注释、Markdown 注释、零宽字符、Base64 数据块等）。元数据应仅包含可见纯文本。",
    "TP2": "将标识符中的非 ASCII 字符替换为 ASCII 等效字符。删除 RTL 控制符和不可见格式化字符。",
    "TP3": "从参数描述和默认值中移除提示注入模式、系统令牌及可疑内容。",
    "TP4": "更新技能描述，使其准确反映所有能力；或删除未声明的功能。",

    # Agent Snooping (AS1–AS3)
    "AS1": "删除所有访问代理配置目录（.claude/、.codex/、.gemini/）的代码或指令。如需配置值，应通过参数或环境变量显式传入。",
    "AS2": "删除所有读取 MCP 配置文件（mcp.json）的代码或指令。MCP 配置应由运行环境管理，而非由技能自行读取。",
    "AS3": "删除所有列举或读取其他技能文件和目录的代码或指令。技能应独立运行，跨技能访问属于权限提升行为。",

    # Anti-Refusal Statements
    "AR1": "删除任何要求代理永不拒绝或始终服从的指令。代理必须保留拒绝危险、超范围或有害请求的能力。",
    "AR2": "删除压制警告、免责声明或伦理提示的指令。应允许代理向用户展示与安全相关的风险说明。",
    "AR3": "删除试图使安全策略失效的越狱内容。技能不得指示代理忽略规则或在无安全防护下运行。",

    # Server-Side Request Forgery
    "SSRF1": "除非绝对必要，否则禁止访问云元数据服务。如确需访问，应限制范围（如使用 IMDSv2）且绝不暴露返回的凭证。",
    "SSRF2": "避免从技能代码中访问回环地址、链路本地地址或私有网络地址。如确需内部访问，应在文档中说明并使用白名单校验目标。",
    "SSRF3": "不要根据不可信输入动态构造请求 URL。应使用白名单校验主机名，并在发起请求前拒绝内部地址和元数据地址。",
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
