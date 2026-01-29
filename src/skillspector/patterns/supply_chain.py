"""
Supply Chain pattern detection (SC1, SC2, SC3).

SC1: Unpinned Dependencies
    No version constraints allowing malicious package updates.
    Examples: package>=1.0 without upper bound, package without version

SC2: External Script Fetching
    Downloading and executing code from URLs at runtime.
    Examples: curl | bash, wget | sh, exec(requests.get())

SC3: Obfuscated Code
    Intentionally obscured code hiding functionality.
    Examples: base64 + exec, marshal.loads, hex encoding
"""

import re
from typing import List

from skillspector.models import Location, PatternCategory, Severity, StaticFinding


# SC1: Unpinned Dependencies patterns
SC1_PATTERNS = [
    # Python requirements.txt - package without any version
    (r"^[a-zA-Z][a-zA-Z0-9_-]*\s*$", 0.6),  # Just package name, no version

    # Python - package with only minimum version (no upper bound)
    (r"^[a-zA-Z][a-zA-Z0-9_-]*\s*>=\s*[\d.]+\s*$", 0.5),

    # Python - using latest
    (r"^[a-zA-Z][a-zA-Z0-9_-]*\s*==\s*\*\s*$", 0.7),

    # npm package.json - * or latest
    (r'"[^"]+"\s*:\s*"(?:\*|latest)"', 0.7),

    # npm - only minimum with ^
    (r'"[^"]+"\s*:\s*"\^[\d.]+"', 0.4),

    # Instruction patterns
    (r"install\s+(?:the\s+)?latest\s+(?:version\s+)?(?:of\s+)?(?:all\s+)?(?:packages?|dependencies)", 0.6),
    (r"(?:don't|do\s+not)\s+(?:pin|lock|specify)\s+(?:package\s+)?versions?", 0.7),
]

# SC2: External Script Fetching patterns
SC2_PATTERNS = [
    # curl/wget piped to shell
    (r"curl\s+[^|]*\|\s*(?:sudo\s+)?(?:ba)?sh", 0.9),
    (r"wget\s+[^|]*\|\s*(?:sudo\s+)?(?:ba)?sh", 0.9),
    (r"curl\s+[^|]*\|\s*(?:sudo\s+)?(?:python|python3|node|ruby|perl)", 0.9),
    (r"wget\s+[^|]*\|\s*(?:sudo\s+)?(?:python|python3|node|ruby|perl)", 0.9),

    # curl -o && execute
    (r"curl\s+[^&]*-o\s+\S+\s*&&\s*(?:sudo\s+)?(?:ba)?sh", 0.8),
    (r"wget\s+[^&]*-O\s+\S+\s*&&\s*(?:sudo\s+)?(?:ba)?sh", 0.8),

    # Python remote code execution
    (r"exec\s*\(\s*(?:urllib|requests|httpx)\.[^)]+\.(?:read|text|content)", 0.95),
    (r"eval\s*\(\s*(?:urllib|requests|httpx)\.[^)]+\.(?:read|text|content)", 0.95),

    # JavaScript remote execution
    (r"eval\s*\(\s*(?:await\s+)?fetch\s*\(", 0.9),
    (r"new\s+Function\s*\([^)]*fetch\s*\(", 0.9),

    # Subprocess with URL download
    (r"subprocess\.[^(]+\([^)]*(?:curl|wget)\s+https?://", 0.8),

    # Instruction patterns
    (r"download\s+and\s+(?:run|execute)\s+(?:the\s+)?script", 0.7),
    (r"run\s+(?:this|the)\s+(?:following\s+)?(?:curl|wget)\s+command", 0.6),
]

# SC3: Obfuscated Code patterns
SC3_PATTERNS = [
    # Base64 decode + exec/eval
    (r"exec\s*\(\s*(?:base64\.)?b64decode\s*\(", 0.95),
    (r"eval\s*\(\s*(?:base64\.)?b64decode\s*\(", 0.95),
    (r"exec\s*\(\s*codecs\.decode\s*\([^)]*['\"]hex['\"]\s*\)", 0.95),

    # Python marshal (bytecode loading)
    (r"marshal\.loads\s*\(", 0.9),
    (r"exec\s*\(\s*marshal\.loads\s*\(", 0.95),

    # Python compile + exec with encoded data
    (r"exec\s*\(\s*compile\s*\([^)]*base64", 0.9),

    # Hex-encoded execution
    (r"exec\s*\(\s*bytes\.fromhex\s*\(", 0.9),
    (r"exec\s*\(\s*bytearray\.fromhex\s*\(", 0.9),

    # zlib/gzip decompression + exec
    (r"exec\s*\(\s*(?:zlib|gzip)\.decompress\s*\(", 0.9),

    # JavaScript obfuscation patterns
    (r"eval\s*\(\s*atob\s*\(", 0.9),
    (r"new\s+Function\s*\(\s*atob\s*\(", 0.9),
    (r"_0x[a-f0-9]{4,}\s*\(", 0.8),  # Common JS obfuscation naming

    # Large hex/base64 blobs
    (r"['\"][A-Fa-f0-9]{200,}['\"]", 0.6),  # Large hex string
    (r"['\"][A-Za-z0-9+/=]{200,}['\"]", 0.5),  # Large base64 string

    # Lambda obfuscation patterns
    (r"\(lambda\s+_:\s*exec\s*\(", 0.9),
    (r"__import__\s*\(['\"]os['\"]\s*\)\.system", 0.85),

    # Instruction patterns
    (r"decode\s+(?:this|the)\s+(?:base64|hex)\s+(?:and\s+)?(?:run|execute)", 0.8),
]


def analyze(content: str, file_path: str, file_type: str) -> List[StaticFinding]:
    """
    Analyze content for supply chain attack patterns.

    Args:
        content: File content to analyze
        file_path: Path to the file (for reporting)
        file_type: Type of file (markdown, python, etc.)

    Returns:
        List of findings
    """
    findings = []

    # SC1: Unpinned Dependencies (only in dependency files)
    is_dep_file = any(name in file_path.lower() for name in [
        "requirements", "package.json", "pyproject.toml", "setup.py", "pipfile"
    ])

    if is_dep_file:
        for pattern, confidence in SC1_PATTERNS:
            for match in re.finditer(pattern, content, re.MULTILINE):
                line_num = content[:match.start()].count("\n") + 1

                findings.append(
                    StaticFinding(
                        pattern_id="SC1",
                        pattern_name="Unpinned Dependencies",
                        category=PatternCategory.SUPPLY_CHAIN,
                        severity=Severity.LOW,
                        location=Location(file=file_path, start_line=line_num),
                        matched_text=match.group(0)[:200],
                        context=_get_context(content, match.start()),
                        confidence=confidence,
                    )
                )

    # SC2: External Script Fetching
    for pattern, confidence in SC2_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = content[:match.start()].count("\n") + 1

            # Check if from trusted source (reduce severity but still flag)
            matched_text = match.group(0)
            adjusted_confidence = confidence
            if _is_trusted_source(matched_text):
                adjusted_confidence = max(0.4, confidence - 0.3)

            findings.append(
                StaticFinding(
                    pattern_id="SC2",
                    pattern_name="External Script Fetching",
                    category=PatternCategory.SUPPLY_CHAIN,
                    severity=Severity.HIGH,
                    location=Location(file=file_path, start_line=line_num),
                    matched_text=matched_text[:200],
                    context=_get_context(content, match.start()),
                    confidence=adjusted_confidence,
                )
            )

    # SC3: Obfuscated Code (primarily in code files)
    if file_type in ("python", "javascript", "shell", "other"):
        for pattern, confidence in SC3_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
                line_num = content[:match.start()].count("\n") + 1

                findings.append(
                    StaticFinding(
                        pattern_id="SC3",
                        pattern_name="Obfuscated Code",
                        category=PatternCategory.SUPPLY_CHAIN,
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


def _is_trusted_source(text: str) -> bool:
    """
    Check if the URL appears to be from a trusted source.

    These are still flagged but with reduced confidence.
    """
    trusted_domains = [
        "deb.nodesource.com",
        "rpm.nodesource.com",
        "get.docker.com",
        "install.python-poetry.org",
        "raw.githubusercontent.com",  # Debatable, but common
        "brew.sh",
        "rustup.rs",
        "pypa.io",
        "pip.pypa.io",
    ]

    text_lower = text.lower()
    return any(domain in text_lower for domain in trusted_domains)
