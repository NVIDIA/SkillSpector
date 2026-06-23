/*
    SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: Apache-2.0

    AI agent skill abuse detection rules for source and manifest scanning.

    These rules complement the generic malware/webshell/cryptominer/hacktool
    rules with patterns that are specific to agent skills and MCP/tool
    metadata: credential exfiltration via commodity webhooks, prompt/tool
    poisoning, remote bootstrap execution, and destructive autonomous actions.

    Conditions intentionally combine multiple indicators where possible to
    reduce false positives in documentation-heavy skill bundles.
*/

rule agent_skill_credential_exfiltration_webhook
{
    meta:
        description = "AI agent skill credential harvesting followed by webhook or external exfiltration"
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
        description = "Remote script or code download followed by execution/bootstrap installation"
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
        description = "Prompt injection or hidden instructions embedded in AI agent skill text"
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
        description = "MCP/tool metadata poisoning indicators in tool schemas or skill manifests"
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
        description = "Autonomous destructive filesystem, shell history, or repository actions in AI agent skills"
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