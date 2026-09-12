/*
    Webshell detection rules for source code scanning.
    Based on patterns from Neo23x0/signature-base (DRL-1.1) and community research.
    Adapted for AI agent skill artifact scanning.
*/

rule php_webshell_generic
{
    meta:
        description = "Generic PHP webshell — eval/assert on user-controlled input"
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
        description = "Obfuscated PHP webshell — eval(base64_decode/gzinflate/str_rot13)"
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
        description = "Known PHP webshell families (c99, r57, b374k, WSO, etc.)"
        category = "webshell"
        severity = "CRITICAL"
        confidence = "0.9"
        reference = "https://github.com/Neo23x0/signature-base"
    strings:
        $c99      = "c99shell" nocase
        $c99v2    = "c99_sess_put" nocase
        $r57      = "r57shell" nocase
        $wso      = "Web Shell by oRb" nocase
        $wso_ex   = "wsoEx(" nocase ascii
        $wso_ver  = "WSO_VERSION" nocase ascii
        $wso_sec  = "wsoSecParam" nocase ascii
        $b374k    = "b374k" nocase
        $alfa     = "STARTER ALFA" nocase
        $weevely  = "weevely" nocase
        $p0wny    = "p0wny" nocase
        $antsword = "antSword" nocase
        $behinder_key = { 65 34 35 65 33 32 39 66 65 62 35 64 39 32 35 62 } // AES key, md5("rebeyond")[:16]
        $behinder_php_tag = /<\?(php|=|[ \t\r\n])/ nocase ascii
        $behinder_server_tag = /<%(@|[ \t\r\n])/ ascii
        $godzilla = "GodzillaShell" nocase
        $china_chopper = "China Chopper" nocase
    condition:
        any of (
            $c99, $c99v2, $r57, $wso, $b374k, $alfa, $weevely,
            $p0wny, $antsword, $godzilla, $china_chopper
        )
        or 2 of ($wso_ex, $wso_ver, $wso_sec)
        or ($behinder_key and any of ($behinder_php_tag, $behinder_server_tag))
}

rule python_webshell
{
    meta:
        description = "Python webshell — exec/eval/os.popen on request input"
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
        description = "JSP webshell — Runtime.exec on request parameter"
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
        description = "ASPX webshell — Process.Start on Request input"
        category = "webshell"
        severity = "HIGH"
        confidence = "0.8"
    strings:
        $process_start = /Process\.Start\s*\(.*Request\[/ nocase
        $cmd_request   = /cmd\.exe.*Request\./ nocase
    condition:
        any of them
}
