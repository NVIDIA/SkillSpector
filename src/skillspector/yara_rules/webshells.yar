/*
    用于源代码扫描的 WebShell 检测规则。
    基于 Neo23x0/signature-base（DRL-1.1）和社区研究中的模式。
    已适配 AI Agent Skill 工件扫描场景。
*/

rule php_webshell_generic
{
    meta:
        description = "通用 PHP WebShell —— 对用户可控输入执行 eval/assert"
        category = "webshell"
        severity = "CRITICAL"
        confidence = "0.85"
        reference = "https://github.com/Neo23x0/signature-base"
    strings:
        $eval_post     = /eval\s*\(\s*\$_(POST|GET|REQUEST|COOKIE)\s*\[/ nocase
        $assert_post   = /assert\s*\(\s*\$_(POST|GET|REQUEST|COOKIE)\s*\[/ nocase
        $system_post   = /system\s*\(\s*\$_(POST|GET|REQUEST)\s*\[/ nocase
        $passthru_post = /passthru\s*\(\s*\$_(POST|GET|REQUEST)\s*\[/ nocase
        $exec_post     = /shell_exec\s*\(\s*\$_(POST|GET|REQUEST)\s*\[/ nocase
        $popen_post    = /popen\s*\(\s*\$_(POST|GET|REQUEST)\s*\[/ nocase
        $proc_open     = /proc_open\s*\(\s*\$_(POST|GET|REQUEST)\s*\[/ nocase
    condition:
        any of them
}

rule php_webshell_obfuscated
{
    meta:
        description = "混淆型 PHP WebShell —— 使用 base64_decode、gzinflate 或 str_rot13 等方式隐藏恶意代码"
        category = "webshell"
        severity = "CRITICAL"
        confidence = "0.8"
        reference = "https://github.com/Neo23x0/signature-base"
    strings:
        $b64_eval       = /eval\s*\(\s*base64_decode\s*\(/ nocase
        $gz_eval        = /eval\s*\(\s*gzinflate\s*\(\s*base64_decode/ nocase
        $rot13_eval     = /eval\s*\(\s*str_rot13\s*\(/ nocase
        $gzuncompress   = /eval\s*\(\s*gzuncompress\s*\(/ nocase
        $preg_replace_e = /preg_replace\s*\(\s*['"]\/.*\/e['"]/ nocase
        $create_func    = /create_function\s*\(\s*['"][^'"]*['"]\s*,\s*\$/ nocase
    condition:
        any of them
}

rule php_webshell_known
{
    meta:
        description = "已知 PHP WebShell 家族（如 c99、r57、b374k、WSO 等）"
        category = "webshell"
        severity = "CRITICAL"
        confidence = "0.9"
        reference = "https://github.com/Neo23x0/signature-base"
    strings:
        $c99      = "c99shell" nocase
        $c99v2    = "c99_sess_put" nocase
        $r57      = "r57shell" nocase
        $wso      = "Web Shell by oRb" nocase
        $wso2     = "WSO " nocase
        $b374k    = "b374k" nocase
        $alfa     = "STARTER ALFA" nocase
        $weevely  = "weevely" nocase
        $p0wny    = "p0wny" nocase
        $antsword = "antSword" nocase
        $behinder = "behinder" nocase
        $godzilla = "GodzillaShell" nocase
        $china_chopper = "China Chopper" nocase
    condition:
        any of them
}

rule python_webshell
{
    meta:
        description = "Python WebShell —— 对用户请求中的输入执行 exec、eval 或系统命令"
        category = "webshell"
        severity = "HIGH"
        confidence = "0.75"
    strings:
        $exec_request      = /exec\s*\(\s*request\./ nocase
        $eval_request      = /eval\s*\(\s*request\./ nocase
        $os_popen_request  = /os\.popen\s*\(\s*request\./ nocase
        $subprocess_req    = /subprocess\.[a-zA-Z0-9_]+\s*\(\s*request\./ nocase
        $os_system_req     = /os\.system\s*\(\s*request\./ nocase
        $flask_cmd_exec    = /os\.(system|popen)\s*\(\s*request\.(args|form|data|json)/ nocase
    condition:
        any of them
}

rule jsp_webshell
{
    meta:
        description = "JSP WebShell —— 将请求参数传递给 Runtime.exec 执行系统命令"
        category = "webshell"
        severity = "HIGH"
        confidence = "0.8"
    strings:
        $runtime_exec    = /Runtime\.getRuntime\(\)\.exec\s*\(\s*request\.getParameter/ nocase
        $processbuilder  = /ProcessBuilder\s*\(.*request\.getParameter/ nocase
    condition:
        any of them
}

rule aspx_webshell
{
    meta:
        description = "ASPX WebShell —— 使用 Request 输入调用 Process.Start 执行系统命令"
        category = "webshell"
        severity = "HIGH"
        confidence = "0.8"
    strings:
        $process_start = /Process\.Start\s*\(.*Request\[/ nocase
        $cmd_request   = /cmd\.exe.*Request\./ nocase
    condition:
        any of them
}