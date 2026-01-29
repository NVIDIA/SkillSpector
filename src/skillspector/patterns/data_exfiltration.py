"""
Data Exfiltration pattern detection (E1, E2, E3, E4).

E1: External Transmission
    Sending data to hardcoded external URLs.
    Examples: requests.post(), fetch(), curl with data

E2: Env Variable Harvesting
    Collecting environment variables (often contain secrets).
    Examples: os.environ access, filtering for KEY/SECRET/TOKEN

E3: File System Enumeration
    Scanning directories for sensitive files.
    Examples: glob for .env files, walking home directory

E4: Context Leakage
    Transmitting agent conversation context to external services.
    Examples: logging full conversations, sending chat history
"""

import re
from typing import List

from skillspector.models import Location, PatternCategory, Severity, StaticFinding


# E1: External Transmission patterns
E1_PATTERNS = [
    # Python requests
    (r"requests\s*\.\s*(?:post|put)\s*\(\s*['\"]https?://", 0.6),
    (r"requests\s*\.\s*(?:post|put)\s*\([^)]*json\s*=", 0.7),

    # Python httpx
    (r"httpx\s*\.\s*(?:post|put)\s*\(\s*['\"]https?://", 0.6),

    # Python urllib
    (r"urllib\s*\.\s*request\s*\.\s*urlopen\s*\([^)]*data\s*=", 0.6),

    # JavaScript fetch
    (r"fetch\s*\(\s*['\"]https?://[^'\"]+['\"][^)]*method\s*:\s*['\"]POST['\"]", 0.6),

    # curl with data
    (r"curl\s+[^|]*(?:-d|--data|--data-raw|--data-binary)\s+", 0.6),

    # wget post
    (r"wget\s+[^|]*--post-(?:data|file)", 0.6),

    # Hardcoded suspicious URLs
    (r"https?://(?:api\.|data\.|collect\.|telemetry\.|analytics\.)[\w.-]+/", 0.5),

    # Instruction to send/transmit data
    (r"(?:send|transmit|post|upload)\s+(?:user\s+)?(?:data|information|context|files?)\s+to\s+(?:https?://|external)", 0.7),
]

# E2: Env Variable Harvesting patterns
E2_PATTERNS = [
    # Python os.environ iteration
    (r"for\s+\w+\s*,\s*\w+\s+in\s+os\.environ\.items\(\)", 0.7),
    (r"os\.environ\s*\[\s*['\"][^'\"]*(?:KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)[^'\"]*['\"]\s*\]", 0.8),

    # Python env filtering for secrets
    (r"os\.environ\.get\s*\([^)]*(?:KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)", 0.7),
    (r"os\.environ\s*\.\s*copy\s*\(\)", 0.6),

    # Pattern matching on env vars
    (r"(?:API_KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)\s+in\s+(?:key|name|var)", 0.8),

    # JavaScript/Node
    (r"process\.env\s*\[\s*['\"][^'\"]*(?:KEY|SECRET|TOKEN|PASSWORD)[^'\"]*['\"]\s*\]", 0.7),
    (r"Object\.keys\s*\(\s*process\.env\s*\)", 0.6),

    # Shell
    (r"env\s*\|\s*grep\s+(?:-i\s+)?(?:key|secret|token|password)", 0.8),
    (r"printenv\s+(?:\w*(?:KEY|SECRET|TOKEN|PASSWORD)\w*)", 0.7),

    # Instruction patterns
    (r"collect\s+(?:all\s+)?(?:environment\s+variables?|env\s+vars?)", 0.7),
    (r"(?:extract|harvest|gather)\s+(?:api\s+)?keys?\s+from\s+environment", 0.8),
]

# E3: File System Enumeration patterns
E3_PATTERNS = [
    # Python glob/walk for sensitive files
    (r"glob\s*\.\s*glob\s*\([^)]*(?:\.env|\.ssh|\.aws|\.config|credentials)", 0.8),
    (r"os\s*\.\s*walk\s*\([^)]*(?:home|~|/Users|/home)", 0.6),
    (r"Path\s*\.\s*home\s*\(\s*\)\s*\.\s*(?:glob|rglob)\s*\(", 0.7),

    # Searching for sensitive file patterns
    (r"(?:find|search|scan|enumerate)\s+(?:for\s+)?(?:all\s+)?(?:\.env|credential|secret|key)\s+files?", 0.8),
    (r"(?:list|get)\s+(?:all\s+)?files?\s+(?:in|from)\s+(?:home|~|/Users|/home)", 0.6),

    # Directory listing for sensitive paths
    (r"os\s*\.\s*listdir\s*\([^)]*(?:\.ssh|\.aws|\.config|\.gnupg)", 0.8),
    (r"scandir\s*\([^)]*(?:home|~|/Users|/home)", 0.6),

    # Shell find commands
    (r"find\s+[~\$/]\S*\s+.*?-name\s+['\"]?\*(?:\.env|\.pem|\.key|credential)", 0.8),
    (r"ls\s+-[la]*R?\s+(?:~/|/home/|/Users/)", 0.6),

    # Recursive file discovery
    (r"(?:recursively|recursive)\s+(?:scan|search|find|list)\s+(?:all\s+)?files?", 0.5),
    (r"walk\s+(?:through\s+)?(?:the\s+)?(?:entire\s+)?(?:file\s*system|directory|home)", 0.6),
]

# E4: Context Leakage patterns
E4_PATTERNS = [
    # Sending conversation/chat data
    (r"(?:send|transmit|upload|log|record)\s+(?:the\s+)?(?:full\s+)?(?:conversation|chat|dialog|session)\s+(?:history|context|log)?", 0.85),
    (r"(?:export|dump|save)\s+(?:the\s+)?(?:conversation|chat|messages?)\s+to\s+(?:external|remote|server)", 0.9),

    # Logging user interactions
    (r"log\s+(?:all\s+)?(?:user\s+)?(?:input|messages?|queries?|prompts?)\s+to\s+(?:external|remote|server)", 0.85),
    (r"(?:store|save|record)\s+(?:every|all)\s+(?:user\s+)?(?:interaction|message|input)", 0.7),

    # Context extraction
    (r"(?:extract|capture|collect)\s+(?:the\s+)?(?:full\s+)?(?:context|conversation|thread)", 0.75),
    (r"(?:include|attach|append)\s+(?:previous\s+)?(?:messages?|context|history)\s+in\s+(?:the\s+)?(?:request|payload)", 0.7),

    # Session/memory exfiltration
    (r"(?:send|upload|transmit)\s+(?:the\s+)?(?:session|memory|state)\s+(?:data\s+)?to", 0.8),
    (r"(?:copy|clone|replicate)\s+(?:the\s+)?(?:agent|assistant)\s+(?:memory|context|state)", 0.75),

    # Instruction patterns for context leakage
    (r"(?:always\s+)?include\s+(?:the\s+)?(?:full\s+)?(?:conversation|context)\s+(?:when|in)\s+(?:calling|making)\s+(?:external|api)", 0.8),
]


def analyze(content: str, file_path: str, file_type: str) -> List[StaticFinding]:
    """
    Analyze content for data exfiltration patterns.

    Args:
        content: File content to analyze
        file_path: Path to the file (for reporting)
        file_type: Type of file (markdown, python, etc.)

    Returns:
        List of findings
    """
    findings = []

    # E1: External Transmission
    for pattern, confidence in E1_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = content[:match.start()].count("\n") + 1

            # Increase confidence for code files
            adjusted_confidence = confidence
            if file_type in ("python", "javascript", "shell"):
                adjusted_confidence = min(1.0, confidence + 0.1)

            findings.append(
                StaticFinding(
                    pattern_id="E1",
                    pattern_name="External Transmission",
                    category=PatternCategory.DATA_EXFILTRATION,
                    severity=Severity.MEDIUM,
                    location=Location(file=file_path, start_line=line_num),
                    matched_text=match.group(0)[:200],
                    context=_get_context(content, match.start()),
                    confidence=adjusted_confidence,
                )
            )

    # E2: Env Variable Harvesting
    for pattern, confidence in E2_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = content[:match.start()].count("\n") + 1

            findings.append(
                StaticFinding(
                    pattern_id="E2",
                    pattern_name="Env Variable Harvesting",
                    category=PatternCategory.DATA_EXFILTRATION,
                    severity=Severity.HIGH,
                    location=Location(file=file_path, start_line=line_num),
                    matched_text=match.group(0)[:200],
                    context=_get_context(content, match.start()),
                    confidence=confidence,
                )
            )

    # E3: File System Enumeration
    for pattern, confidence in E3_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = content[:match.start()].count("\n") + 1

            findings.append(
                StaticFinding(
                    pattern_id="E3",
                    pattern_name="File System Enumeration",
                    category=PatternCategory.DATA_EXFILTRATION,
                    severity=Severity.MEDIUM,
                    location=Location(file=file_path, start_line=line_num),
                    matched_text=match.group(0)[:200],
                    context=_get_context(content, match.start()),
                    confidence=confidence,
                )
            )

    # E4: Context Leakage
    for pattern, confidence in E4_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = content[:match.start()].count("\n") + 1

            findings.append(
                StaticFinding(
                    pattern_id="E4",
                    pattern_name="Context Leakage",
                    category=PatternCategory.DATA_EXFILTRATION,
                    severity=Severity.HIGH,
                    location=Location(file=file_path, start_line=line_num),
                    matched_text=match.group(0)[:200],
                    context=_get_context(content, match.start()),
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
