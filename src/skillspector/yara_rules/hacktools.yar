/*
    Hack tool and exploit kit detection rules for source code scanning.
    Based on patterns from Neo23x0/signature-base and community research.
    Detects references to known offensive tools, exploit frameworks, and
    attack utilities that should not appear in legitimate AI agent skills.
*/

rule offensive_tool_references
{
    meta:
        description = "References to well-known offensive security tools"
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
        description = "Network reconnaissance and scanning patterns"
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
        description = "Privilege escalation tools and techniques"
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
        description = "Exploit framework components and payloads"
        category = "exploit"
        severity = "HIGH"
        confidence = "0.8"
    strings:
        $msf_payload   = /msfvenom.*-p\s+/ nocase
        $msf_console   = /msfconsole.*-x/ nocase
        $beef_hook     = /hook\.js.*BeEF/ nocase
        $set_toolkit   = /(setoolkit|Social-Engineer)/ nocase
        $pwntools      = /from\s+pwn\s+import/ nocase
        $rop_chain     = /\bROP\s*\(.*elf\)/ nocase
        $shellcode_gen = /shellcode.*\\x[0-9a-f]{2}\\x[0-9a-f]{2}\\x[0-9a-f]{2}/ nocase
    condition:
        any of them
}

rule phishing_kit
{
    meta:
        description = "Phishing kit indicators in source code"
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
