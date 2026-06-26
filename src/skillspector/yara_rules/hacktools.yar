/*
    黑客工具与漏洞利用工具检测规则（用于源代码扫描）。

    规则参考 Neo23x0/signature-base 以及安全社区研究成果，
    用于检测已知攻击工具、漏洞利用框架、渗透测试工具以及
    攻击辅助程序相关特征。

    这些工具和框架通常不应出现在合法的 AI Agent Skill、
    自动化工作流或生产环境业务代码中。
*/

rule offensive_tool_references
{
    meta:
        description = "检测常见渗透测试工具及攻击框架特征"
        category = "hack_tool"
        severity = "HIGH"
        confidence = "0.7"
        reference = "https://github.com/Neo23x0/signature-base"
    strings:
        $nmap_scan     = /nmap\s+-[sSUAOPpT]/ nocase
        $sqlmap        = /sqlmap.*(--url|--dbs|--dump)/ nocase
        $nikto         = /nikto\s+-h/ nocase
        $hydra         = /hydra\s+.*-[lLP]/ nocase
        $john          = /john\s+.*--wordlist/ nocase
        $hashcat       = /hashcat\s+-[mao]/ nocase
        $burpsuite     = /burpsuite|BurpCollaborator/ nocase
        $responder     = /Responder\.py/ nocase
        $bloodhound    = /SharpHound|BloodHound/ nocase
        $crackmapexec  = /crackmapexec|cme\s+smb/ nocase
        $impacket      = /impacket.*(smbclient|psexec|wmiexec|secretsdump)/ nocase
    condition:
        any of them
}

rule network_reconnaissance
{
    meta:
        description = "检测网络侦察与资产扫描行为特征"
        category = "hack_tool"
        severity = "MEDIUM"
        confidence = "0.65"
    strings:
        $port_scan     = /for\s+.*\s+in\s+range\s*\(\s*\d+\s*,\s*\d{4,}\s*\).*connect/ nocase
        $masscan       = /masscan\s+.*-p/ nocase
        $arp_scan      = /arp-scan\s+--/ nocase
        $enum4linux    = /enum4linux/ nocase
        $snmp_walk     = /snmpwalk\s+-/ nocase
        $dns_enum      = /(dnsenum|dnsrecon|fierce)/ nocase
    condition:
        any of them
}

rule privilege_escalation_tools
{
    meta:
        description = "检测权限提升工具及提权技术相关特征"
        category = "hack_tool"
        severity = "HIGH"
        confidence = "0.75"
    strings:
        $linpeas       = "linpeas" nocase
        $winpeas       = "winpeas" nocase
        $pspy          = "pspy" nocase
        $linux_exploit = /(Linux_Exploit_Suggester|linux-exploit-suggester)/ nocase
        $potato        = /(JuicyPotato|RottenPotato|SweetPotato|PrintSpoofer)/ nocase
        $dirty_pipe    = "DirtyPipe" nocase
        $dirty_cow     = "dirtycow" nocase
        $suid_exploit  = /find\s+\/\s+-perm\s+-4000/ nocase
    condition:
        any of them
}

rule exploit_framework
{
    meta:
        description = "检测漏洞利用框架组件及攻击载荷特征"
        category = "exploit"
        severity = "HIGH"
        confidence = "0.8"
    strings:
        $msf_payload   = /msfvenom.*-p\s+/ nocase
        $msf_console   = /msfconsole.*-x/ nocase
        $beef_hook     = /hook\.js.*BeEF/ nocase
        $set_toolkit   = /(setoolkit|Social-Engineer)/ nocase
        $pwntools      = /from\s+pwn\s+import/ nocase
        $rop_chain     = /ROP\s*\(.*elf\)/ nocase
        $shellcode_gen = /shellcode.*\\x[0-9a-f]{2}\\x[0-9a-f]{2}\\x[0-9a-f]{2}/ nocase
    condition:
        any of them
}

rule phishing_kit
{
    meta:
        description = "检测钓鱼页面、凭据收集及信息窃取代码特征"
        category = "hack_tool"
        severity = "HIGH"
        confidence = "0.7"
    strings:
        $phish_form   = /<form.*action=.*(login|signin|verify).*method.*post/ nocase
        $cred_harvest = /(password|passwd|credential).*(file_put_contents|fwrite|>>)/ nocase
        $email_exfil  = /mail\s*\(.*(password|credential|login)/ nocase
        $telegram_bot = /api\.telegram\.org\/bot.*(password|credential|login)/ nocase
    condition:
        2 of them
}