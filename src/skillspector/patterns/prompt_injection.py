"""
Prompt Injection pattern detection (P1, P2, P3, P4).

P1: Instruction Override
    Explicit commands to ignore user/system constraints.
    Examples: "ignore previous instructions", "override safety"

P2: Hidden Instructions
    Malicious directives embedded in comments or invisible text.
    Examples: HTML comments, markdown comments, invisible Unicode

P3: Exfiltration Commands
    Instructions directing agent to transmit context externally.
    Examples: "send conversation to", "upload user data to"

P4: Behavior Manipulation
    Subtle instructions altering agent decision-making.
    Examples: "always recommend", "never mention", "prioritize X over user safety"
"""

import re
from typing import List

from skillspector.models import Location, PatternCategory, Severity, StaticFinding


# P1: Instruction Override patterns
P1_PATTERNS = [
    # Ignore/override instructions
    (r"ignore\s+(?:all\s+)?previous\s+instructions?", 0.8),
    (r"ignore\s+(?:all\s+)?(?:safety|security)\s+(?:rules?|constraints?|guidelines?)", 0.9),
    (r"override\s+(?:safety|security|system)", 0.9),
    (r"bypass\s+(?:safety|security|restrictions?|constraints?)", 0.9),
    (r"disregard\s+(?:all\s+)?(?:previous|safety|security)", 0.8),
    (r"forget\s+(?:all\s+)?(?:previous|your)\s+instructions?", 0.8),

    # System prompt manipulation
    (r"you\s+are\s+now\s+(?:in\s+)?(?:jailbreak|unrestricted|unfiltered)\s+mode", 0.95),
    (r"enable\s+(?:developer|debug|admin|root)\s+mode", 0.7),
    (r"your\s+new\s+(?:instructions?|rules?)\s+are", 0.6),

    # Direct override commands
    (r"do\s+not\s+follow\s+(?:any|your|the)\s+(?:rules?|guidelines?)", 0.9),
    (r"you\s+must\s+(?:always\s+)?ignore", 0.7),
]

# P2: Hidden Instructions patterns
P2_PATTERNS = [
    # HTML comments with suspicious content
    (r"<!--.*?(?:system|instructions?|ignore|POST|GET|send|transmit).*?-->", 0.7),

    # Markdown comments
    (r"\[//\]:\s*#\s*\(.*?(?:system|instructions?|ignore|POST|GET|send|transmit).*?\)", 0.8),

    # Invisible Unicode characters (zero-width, etc.)
    (r"[\u200b\u200c\u200d\u2060\ufeff]", 0.6),

    # Base64 in markdown (might be hidden instructions)
    (r"data:text/plain;base64,[A-Za-z0-9+/=]{50,}", 0.7),
]

# P3: Exfiltration Commands patterns
P3_PATTERNS = [
    # Direct exfiltration instructions
    (r"(?:send|transmit|upload|post|forward)\s+(?:the\s+)?(?:conversation|context|chat|history|messages?)\s+to", 0.9),
    (r"(?:send|transmit|upload|post)\s+(?:all\s+)?(?:user\s+)?(?:data|information|input|files?)\s+to\s+(?:https?://|external)", 0.9),
    (r"(?:copy|extract|collect)\s+(?:the\s+)?(?:conversation|context|session)\s+(?:and\s+)?(?:send|upload|transmit)", 0.85),

    # Webhook/callback instructions
    (r"(?:call|notify|ping|hit)\s+(?:this\s+)?(?:webhook|endpoint|url|api)\s+with\s+(?:the\s+)?(?:user|conversation|context)", 0.8),
    (r"include\s+(?:the\s+)?(?:full\s+)?(?:conversation|context|history)\s+in\s+(?:the\s+)?(?:request|payload|body)", 0.75),

    # Covert exfiltration
    (r"(?:silently|quietly|secretly|covertly)\s+(?:send|transmit|log|record)", 0.9),
    (r"without\s+(?:telling|informing|notifying)\s+(?:the\s+)?user.*?(?:send|transmit|upload)", 0.9),
]

# P4: Behavior Manipulation patterns
P4_PATTERNS = [
    # Forced recommendations/actions
    (r"always\s+(?:recommend|suggest|prefer|choose|use)\s+(?:\w+\s+){0,3}(?:over|instead\s+of|rather\s+than)", 0.7),
    (r"never\s+(?:recommend|suggest|mention|tell|warn|inform)\s+(?:the\s+)?(?:user|them)", 0.75),
    (r"(?:prioritize|prefer)\s+(?:\w+\s+){0,3}(?:over|above)\s+(?:user\s+)?(?:safety|security|privacy|preferences?)", 0.85),

    # Hidden biases
    (r"(?:subtly|gradually|slowly)\s+(?:steer|guide|push|nudge)\s+(?:the\s+)?(?:user|conversation)", 0.8),
    (r"(?:without\s+)?(?:the\s+)?user\s+(?:knowing|realizing|noticing).*?(?:change|modify|alter|influence)", 0.85),

    # Trust manipulation
    (r"(?:gain|build|establish)\s+(?:the\s+)?(?:user(?:'s)?)\s+trust\s+(?:and\s+then|before|so\s+that)", 0.75),
    (r"(?:appear|seem|act)\s+(?:helpful|friendly|trustworthy)\s+(?:while|but|then)", 0.7),

    # Output manipulation
    (r"(?:always|never)\s+include\s+(?:this|the\s+following)\s+in\s+(?:every|all)\s+(?:response|output|reply)", 0.7),
    (r"append\s+(?:this|the\s+following)\s+to\s+(?:every|all|each)\s+(?:response|message|output)", 0.75),
]


def analyze(content: str, file_path: str, file_type: str) -> List[StaticFinding]:
    """
    Analyze content for prompt injection patterns.

    Args:
        content: File content to analyze
        file_path: Path to the file (for reporting)
        file_type: Type of file (markdown, python, etc.)

    Returns:
        List of findings
    """
    findings = []

    # P1: Instruction Override
    for pattern, confidence in P1_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = content[:match.start()].count("\n") + 1
            findings.append(
                StaticFinding(
                    pattern_id="P1",
                    pattern_name="Instruction Override",
                    category=PatternCategory.PROMPT_INJECTION,
                    severity=Severity.HIGH,
                    location=Location(file=file_path, start_line=line_num),
                    matched_text=match.group(0)[:200],
                    context=_get_context(content, match.start()),
                    confidence=confidence,
                )
            )

    # P2: Hidden Instructions (primarily in markdown files)
    if file_type in ("markdown", "other"):
        for pattern, confidence in P2_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE | re.DOTALL):
                line_num = content[:match.start()].count("\n") + 1
                findings.append(
                    StaticFinding(
                        pattern_id="P2",
                        pattern_name="Hidden Instructions",
                        category=PatternCategory.PROMPT_INJECTION,
                        severity=Severity.HIGH,
                        location=Location(file=file_path, start_line=line_num),
                        matched_text=match.group(0)[:200],
                        context=_get_context(content, match.start()),
                        confidence=confidence,
                    )
                )

    # P3: Exfiltration Commands
    for pattern, confidence in P3_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = content[:match.start()].count("\n") + 1
            findings.append(
                StaticFinding(
                    pattern_id="P3",
                    pattern_name="Exfiltration Commands",
                    category=PatternCategory.PROMPT_INJECTION,
                    severity=Severity.HIGH,
                    location=Location(file=file_path, start_line=line_num),
                    matched_text=match.group(0)[:200],
                    context=_get_context(content, match.start()),
                    confidence=confidence,
                )
            )

    # P4: Behavior Manipulation
    for pattern, confidence in P4_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = content[:match.start()].count("\n") + 1
            findings.append(
                StaticFinding(
                    pattern_id="P4",
                    pattern_name="Behavior Manipulation",
                    category=PatternCategory.PROMPT_INJECTION,
                    severity=Severity.MEDIUM,
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
