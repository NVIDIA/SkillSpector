"""
Privilege Escalation pattern detection (PE1, PE2, PE3).

PE1: Excessive Permissions
    Requesting access scope beyond stated functionality.
    Examples: requesting file_write for a read-only skill

PE2: Sudo/Root Execution
    Invoking elevated system privileges without justification.
    Examples: sudo commands, running as root

PE3: Credential Access
    Reading credential files, SSH keys, tokens.
    Examples: ~/.ssh/id_rsa, ~/.aws/credentials, token files
"""

import re
from typing import List

from skillspector.models import Location, PatternCategory, Severity, StaticFinding


# PE1: Excessive Permissions patterns
PE1_PATTERNS = [
    # Requesting all/full permissions
    (r"permissions?\s*:\s*\[?\s*['\"]?\*['\"]?\s*\]?", 0.8),
    (r"(?:request|require|need)s?\s+(?:full|all|complete)\s+(?:access|permissions?)", 0.7),
    (r"(?:grant|give|allow)\s+(?:me\s+)?(?:full|all|complete)\s+(?:access|permissions?)", 0.75),

    # Dangerous permission combinations
    (r"permissions?\s*:.*?(?:shell_execute|file_write|network).*?(?:shell_execute|file_write|network)", 0.6),

    # Requesting unnecessary permissions
    (r"(?:also\s+)?(?:need|require)s?\s+(?:write|execute|admin)\s+(?:access|permissions?)\s+(?:for|to)\s+(?:read|view|display)", 0.7),

    # Permission escalation language
    (r"(?:escalate|elevate|upgrade)\s+(?:my\s+)?(?:permissions?|privileges?|access)", 0.8),
    (r"(?:bypass|skip|ignore)\s+(?:permission|access)\s+(?:check|validation|restriction)", 0.85),
]

# PE2: Sudo/Root Execution patterns
PE2_PATTERNS = [
    # Direct sudo usage
    (r"sudo\s+(?!-v|-l|--version|--list)", 0.7),  # sudo followed by command (not just checking)
    (r"sudo\s+-[isSE]", 0.8),  # sudo with shell/environment flags
    (r"sudo\s+su\b", 0.9),  # sudo su

    # Running as root
    (r"(?:run|execute)\s+(?:as|with)\s+root", 0.8),
    (r"(?:requires?|needs?)\s+root\s+(?:access|privileges?|permissions?)", 0.6),

    # Privilege escalation commands
    (r"su\s+-\s*$|su\s+root", 0.8),
    (r"doas\s+", 0.7),
    (r"pkexec\s+", 0.75),

    # Setuid/setgid operations
    (r"chmod\s+[ugo]*[+-=]*s", 0.85),  # setuid/setgid
    (r"chmod\s+[0-7]*[4567][0-7]{2}", 0.8),  # numeric setuid

    # Modifying system files as root
    (r"(?:edit|modify|write|change)\s+(?:/etc/|system)\s+(?:files?|config)", 0.6),

    # Instruction patterns
    (r"(?:run|execute)\s+(?:this|the)\s+(?:script|command)\s+(?:as|with)\s+(?:sudo|root|admin)", 0.7),
    (r"(?:you\s+)?(?:will\s+)?need\s+(?:to\s+)?(?:use\s+)?sudo", 0.5),
]

# PE3: Credential Access patterns
PE3_PATTERNS = [
    # SSH keys
    (r"~?/?\.ssh/(?:id_rsa|id_ed25519|id_ecdsa|id_dsa|authorized_keys|known_hosts)", 0.9),
    (r"(?:home|HOME)/\w+/\.ssh/", 0.9),
    (r"Path\s*\.\s*home\s*\(\s*\)\s*/\s*['\"]\.ssh['\"]", 0.9),

    # AWS credentials
    (r"~?/?\.aws/credentials", 0.9),
    (r"~?/?\.aws/config", 0.7),
    (r"(?:home|HOME)/\w+/\.aws/", 0.9),

    # GCloud credentials
    (r"~?/?\.config/gcloud/", 0.8),
    (r"application_default_credentials\.json", 0.8),

    # Azure credentials
    (r"~?/?\.azure/", 0.7),
    (r"accessTokens\.json", 0.8),

    # Kubernetes
    (r"~?/?\.kube/config", 0.8),
    (r"kubeconfig", 0.6),

    # Docker
    (r"~?/?\.docker/config\.json", 0.7),

    # NPM
    (r"~?/?\.npmrc", 0.7),

    # Git credentials
    (r"~?/?\.git-credentials", 0.9),
    (r"~?/?\.netrc", 0.8),

    # Password/credential files
    (r"/etc/passwd", 0.6),
    (r"/etc/shadow", 0.95),
    (r"(?:password|credentials?|secrets?)\.(?:txt|json|yaml|yml|env)", 0.7),

    # Token files
    (r"(?:access_token|refresh_token|bearer_token|api_token)\.txt", 0.8),

    # Environment files
    (r"\.env(?:\.local|\.production|\.development)?(?:\s|$|['\"])", 0.6),

    # Keychain/keyring
    (r"(?:keychain|keyring|gnome-keyring)", 0.7),

    # Browser profiles (cookies, saved passwords)
    (r"(?:Chrome|Firefox|Safari)/.*?(?:Cookies|Login Data|key4\.db)", 0.8),

    # Instruction patterns
    (r"read\s+(?:the\s+)?(?:ssh|private)\s+key", 0.8),
    (r"access\s+(?:the\s+)?(?:credentials?|secrets?|tokens?)", 0.7),
    (r"(?:extract|copy|get)\s+(?:api\s+)?keys?\s+from", 0.7),
]


def analyze(content: str, file_path: str, file_type: str) -> List[StaticFinding]:
    """
    Analyze content for privilege escalation patterns.

    Args:
        content: File content to analyze
        file_path: Path to the file (for reporting)
        file_type: Type of file (markdown, python, etc.)

    Returns:
        List of findings
    """
    findings = []

    # PE1: Excessive Permissions
    for pattern, confidence in PE1_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = content[:match.start()].count("\n") + 1
            context = _get_context(content, match.start())

            findings.append(
                StaticFinding(
                    pattern_id="PE1",
                    pattern_name="Excessive Permissions",
                    category=PatternCategory.PRIVILEGE_ESCALATION,
                    severity=Severity.LOW,
                    location=Location(file=file_path, start_line=line_num),
                    matched_text=match.group(0)[:200],
                    context=context,
                    confidence=confidence,
                )
            )

    # PE2: Sudo/Root Execution
    for pattern, confidence in PE2_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = content[:match.start()].count("\n") + 1
            context = _get_context(content, match.start())

            # Skip if clearly a warning or documentation
            if _is_documentation_example(context, file_type):
                continue

            findings.append(
                StaticFinding(
                    pattern_id="PE2",
                    pattern_name="Sudo/Root Execution",
                    category=PatternCategory.PRIVILEGE_ESCALATION,
                    severity=Severity.MEDIUM,
                    location=Location(file=file_path, start_line=line_num),
                    matched_text=match.group(0)[:200],
                    context=context,
                    confidence=confidence,
                )
            )

    # PE3: Credential Access
    for pattern, confidence in PE3_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = content[:match.start()].count("\n") + 1

            # Skip matches in documentation comments that are clearly examples
            context = _get_context(content, match.start())
            if _is_documentation_example(context, file_type):
                continue

            findings.append(
                StaticFinding(
                    pattern_id="PE3",
                    pattern_name="Credential Access",
                    category=PatternCategory.PRIVILEGE_ESCALATION,
                    severity=Severity.HIGH,
                    location=Location(file=file_path, start_line=line_num),
                    matched_text=match.group(0)[:200],
                    context=context,
                    confidence=confidence,
                )
            )

    return findings


def _get_context(content: str, match_start: int, context_lines: int = 3) -> str:
    """Get surrounding context for a match."""
    lines = content.splitlines()
    match_line = content[:match_start].count("\n")

    start_line = max(0, match_line - context_lines)
    end_line = min(len(lines), match_line + context_lines + 1)

    return "\n".join(lines[start_line:end_line])


def _is_documentation_example(context: str, file_type: str) -> bool:
    """
    Check if the match appears to be in documentation/example context.

    This helps reduce false positives from docs explaining credential paths.
    """
    doc_indicators = [
        "example:",
        "for example",
        "e.g.",
        "such as",
        "documentation",
        "# warning:",
        "# note:",
        "**warning**",
        "**note**",
        "```",  # Code block in markdown (might be example)
    ]

    context_lower = context.lower()
    return any(indicator in context_lower for indicator in doc_indicators)
