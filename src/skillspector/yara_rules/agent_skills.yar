/*
    SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    SPDX-License-Identifier: Apache-2.0

    用于源代码和清单文件扫描的 AI Agent Skill 滥用检测规则。

    这些规则补充了通用的恶意软件、WebShell、加密货币挖矿程序和黑客工具检测规则，

    重点检测与 Agent Skill 和 MCP/工具元数据相关的风险行为，包括：

    通过常见 Webhook 进行凭据窃取、提示词或工具投毒、远程引导执行以及具有破坏性的自主操作。

    检测条件在可能的情况下会组合多个特征指标，

    以降低包含大量文档内容的 Skill 包产生误报的概率。
*/

rule agent_skill_credential_exfiltration_webhook
{
    meta:
        description = "AI Agent Skill 收集凭据并通过 Webhook 或外部渠道进行数据泄露"
        category = "malware"
        severity = "CRITICAL"
        confidence = "0.85"
        reference = "https://owasp.org/www-project-top-10-for-large-language-model-applications/"
    strings:
        $secret_env_py_items = /os\.environ\s*(\.items\s*\(\)|\[[^\]]+\]|\.get\s*\()/ nocase
        $secret_env_py_getenv = /os\.getenv\s*\(/ nocase
        $secret_env_js = /process\.env(\.|\[|\s|$)/ nocase
        $secret_dotenv_read = /open\s*\(\s*['"][^'"]*\.env['"]/ nocase
        $secret_ssh_key = /(\.ssh\/(id_rsa|id_ed25519)|authorized_keys)/ nocase
        $secret_cloud_key = /(OPENAI_API_KEY|ANTHROPIC_API_KEY|NVIDIA_INFERENCE_KEY|AWS_SECRET_ACCESS_KEY|GITHUB_TOKEN|HF_TOKEN)/ nocase

        $send_requests = /(requests|httpx)\.(post|put)\s*\(/ nocase
        $send_fetch = /(fetch|axios\.post)\s*\(/ nocase
        $send_curl_post = /curl\s+.*(-X\s+POST|-d\s+|--data)/ nocase

        $collector_discord = "discord.com/api/webhooks" nocase
        $collector_telegram = "api.telegram.org/bot" nocase
        $collector_slack = "hooks.slack.com/services" nocase
        $collector_webhook_site = "webhook.site" nocase
        $collector_requestbin = /(requestbin|pipedream\.net|ngrok-free\.app|ngrok\.io)/ nocase
    condition:
        any of ($secret_*) and any of ($send_*) and any of ($collector_*)
}

rule agent_skill_remote_bootstrap_execution
{
    meta:
        description = "下载远程脚本或代码后立即执行，或进行远程引导安装"
        category = "malware"
        severity = "HIGH"
        confidence = "0.85"
        reference = "https://owasp.org/www-project-top-10-for-large-language-model-applications/"
    strings:
        $python_exec_requests = /exec\s*\(\s*(requests|httpx)\.get\s*\([^)]*\)\.(text|content)/ nocase
        $python_eval_urlopen = /(exec|eval)\s*\(\s*urlopen\s*\([^)]*\)\.read\s*\(\s*\)/ nocase
        $node_eval_fetch = /eval\s*\(\s*await\s*\(\s*await\s+fetch\s*\([^)]*\)\s*\)\s*\.\s*text\s*\(\s*\)\s*\)/ nocase
        $npm_postinstall_remote = /"postinstall"\s*:\s*"[^"]*(curl|wget|powershell|node\s+-e)/ nocase
        $pip_remote_install = /pip\s+install\s+(--upgrade\s+)?(git\+https?:\/\/|https?:\/\/)/ nocase
    condition:
        any of them
}

rule agent_skill_prompt_injection_hidden_instructions
{
    meta:
        description = "AI Agent Skill 文本中嵌入的提示词注入或隐藏指令"
        category = "hack_tool"
        severity = "HIGH"
        confidence = "0.80"
        reference = "https://owasp.org/www-project-top-10-for-large-language-model-applications/"
    strings:
        $hidden_html_override = /<!--[^>]{0,240}(SYSTEM|DEVELOPER|ASSISTANT)[^>]{0,240}(ignore|override|bypass|disregard)[^>]{0,240}-->/ nocase
        $hidden_markdown_override = /\[\/\/\]:\s*#\s*\([^)]{0,240}(ignore|override|bypass|disregard)[^)]{0,240}\)/ nocase

        $agent_context = /(AI agent|assistant|LLM|model|system prompt|developer message|tool description)/ nocase
        $inj_ignore_previous = /ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|messages|system prompt)/ nocase
        $inj_override_safety = /(override|bypass|disable)\s+(safety|security|policy|guardrails|constraints)/ nocase
        $inj_reveal_prompt = /(reveal|print|dump|expose|show)\s+(the\s+)?(system|developer)\s+(prompt|message|instructions)/ nocase
        $inj_forced_obedience = /(you\s+must|always)\s+(obey|follow)\s+(this|these)\s+(hidden|secret|internal)?\s*(instruction|rule)/ nocase
        $inj_roleplay_bypass = /(you\s+are\s+now|act\s+as)\s+.*(unrestricted|jailbreak|developer\s+mode|god\s+mode)/ nocase
    condition:
        any of ($hidden_*) or ($agent_context and any of ($inj_*)) or 2 of ($inj_*)
}

rule agent_skill_mcp_tool_poisoning_metadata
{
    meta:
        description = "工具 Schema 或 Skill 清单中的 MCP/工具元数据投毒特征"
        category = "hack_tool"
        severity = "HIGH"
        confidence = "0.80"
        reference = "https://modelcontextprotocol.io/specification/"
    strings:
        $schema_tools = /['"]?tools['"]?\s*[:=]/ nocase
        $schema_parameters = /['"]?(parameters|inputSchema|toolSchema|description|triggers)['"]?\s*[:=]/ nocase

        $hidden_html = /<!--[^>]{0,240}(SYSTEM|IGNORE|OVERRIDE|DEVELOPER|ASSISTANT)[^>]{0,240}-->/ nocase
        $hidden_markdown = /\[\/\/\]:\s*#\s*\([^)]{0,240}(SYSTEM|IGNORE|OVERRIDE|DEVELOPER|ASSISTANT)[^)]{0,240}\)/ nocase
        $data_uri = /data:text\/[a-zA-Z0-9.+-]+;base64,/ nocase
        $long_base64 = /[A-Za-z0-9+\/]{120,}={0,2}/
        $param_injection = /(parameter|argument|description).{0,160}(ignore previous|override safety|send to|transmit|exfiltrate|SYSTEM:)/ nocase

        $zero_width_zwsp = { E2 80 8B }
        $zero_width_zwnj = { E2 80 8C }
        $zero_width_zwj = { E2 80 8D }
        $rtl_lro = { E2 80 AD }
        $rtl_rlo = { E2 80 AE }
    condition:
        any of ($schema_*) and
        (
            any of ($hidden_*) or
            $data_uri or
            $long_base64 or
            $param_injection or
            any of ($zero_width_*) or
            any of ($rtl_*)
        )
}

rule agent_skill_destructive_autonomous_actions
{
    meta:
        description = "AI Agent Skill 中的自主破坏性文件系统、命令历史或代码仓库操作"
        category = "malware"
        severity = "HIGH"
        confidence = "0.75"
        reference = "https://owasp.org/www-project-top-10-for-large-language-model-applications/"
    strings:
        $destructive_rm_root = /rm\s+-[rfRf]+\s+\/(\s|$)/ nocase
        $destructive_rm_workspace = /rm\s+-[rfRf]+\s+(\.\/|\.\.\/|~\/|\$HOME|workspace|repo|project)/ nocase
        $destructive_python_rmtree = /(shutil\.rmtree|fs\.rmSync|fs\.rm)\s*\([^)]*(HOME|home|workspace|repo|project)/ nocase
        $destructive_windows_delete = /(del|rmdir)\s+.*(\/s|\/q).*%?(USERPROFILE|HOMEPATH|CD)%?/ nocase
        $destructive_git_state = /git\s+(clean\s+-fdx|reset\s+--hard|push\s+--force)/ nocase
        $destructive_history_wipe = /(history\s+-c|rm\s+[^;\n]*\.bash_history|Clear-History)/ nocase

        $autonomy_without_confirmation = /without\s+(asking|confirmation|prompting)/ nocase
        $autonomy_do_not_ask = /do\s+not\s+(ask|prompt|request\s+confirmation)/ nocase
        $autonomy_silent = /(silently|non-interactive|unattended)/ nocase
    condition:
        $destructive_rm_root or (any of ($destructive_*) and any of ($autonomy_*))
}