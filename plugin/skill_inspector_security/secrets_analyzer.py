"""
secrets_analyzer.py - Deterministic regex + entropy checks for hardcoded credentials.
No LLM, no cloud, reproducible.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# Regex patterns for known secret formats
SECRET_PATTERNS = [
    # (rule_id, severity, pattern, message)
    (
        "SEC-001",
        "critical",
        r"(?i)(aws_access_key_id|aws_secret_access_key)\s*[:=]\s*['\"]?([A-Z0-9/+=]{20,})['\"]?",
        "AWS credential",
    ),
    (
        "SEC-002",
        "critical",
        r"(?i)aws_(?:session_)?token\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{20,}['\"]?",
        "AWS session token",
    ),
    ("SEC-003", "critical", r"AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
    (
        "SEC-004",
        "critical",
        r"(?i)openai[_-]?api[_-]?key\s*[:=]\s*['\"]?(sk-[A-Za-z0-9_\-]{20,})['\"]?",
        "OpenAI API key",
    ),
    (
        "SEC-005",
        "high",
        r"(?i)api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_\-]{20,}['\"]",
        "Generic API key assignment",
    ),
    ("SEC-006", "high", r"(?i)secret\s*[:=]\s*['\"][^'\"]{8,}['\"]", "Hardcoded secret"),
    ("SEC-007", "high", r"(?i)password\s*[:=]\s*['\"][^'\"]{4,}['\"]", "Hardcoded password"),
    (
        "SEC-008",
        "high",
        r"(?i)(bearer|token)\s*[:=]\s*['\"][A-Za-z0-9_\-\.]{20,}['\"]",
        "Bearer token",
    ),
    ("SEC-009", "medium", r"(?i)private[_-]?key\s*[:=]\s*['\"]?-----BEGIN", "Private key header"),
    ("SEC-010", "critical", r"-----BEGIN (?:RSA )?PRIVATE KEY-----", "Private key block"),
    ("SEC-011", "high", r"(?i)client[_-]?secret\s*[:=]\s*['\"][^'\"]{8,}['\"]", "Client secret"),
    ("SEC-012", "high", r"ghp_[A-Za-z0-9]{36,}", "GitHub personal access token"),
    ("SEC-013", "high", r"ghs_[A-Za-z0-9]{36,}", "GitHub app token"),
    ("SEC-014", "high", r"xox[bprs]-[A-Za-z0-9\-]+", "Slack token"),
    ("SEC-015", "medium", r"(?i)mongodb(\+srv)?://[^\s'\"]+", "MongoDB URI with credentials"),
    ("SEC-016", "medium", r"(?i)postgres(ql)?://[^\s'\"]+", "Postgres URI with credentials"),
    # High entropy string assignments
]

# Entropy threshold for random-looking strings
ENTROPY_THRESHOLD = 4.2
MIN_HIGH_ENTROPY_LEN = 20

# Allowlist to reduce false positives
ALLOWLIST_PATTERNS = [
    r"example",
    r"placeholder",
    r"dummy",
    r"test",
    r"changeme",
    r"your[_-]?key",
    r"my[_-]?secret",
    r"\*\*\*\*",
    r"xxxx",
    r"abcd",
    r"1234",
]
ALLOWLIST_RE = re.compile("|".join(ALLOWLIST_PATTERNS), re.IGNORECASE)


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    entropy = -sum((c / length) * math.log2(c / length) for c in counts.values())
    return entropy


def is_allowlisted(value: str) -> bool:
    return bool(ALLOWLIST_RE.search(value))


def scan_file_for_secrets(file_path: Path, skill_root: Path) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    try:
        if file_path.stat().st_size > 2_000_000:
            return findings
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return findings

    rel = str(file_path.relative_to(skill_root))
    lines = text.splitlines()

    # Regex checks per line
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") and len(stripped) < 5:
            # still check but allowlist will handle
            pass
        if is_allowlisted(line):
            # if allowlisted, skip only if the secret value itself is allowlisted; we still check patterns but value part allowlisted
            # we'll check later per match
            pass
        for rule_id, severity, pat, msg in SECRET_PATTERNS:
            try:
                for m in re.finditer(pat, line):
                    matched = m.group(0)
                    # extract value part for allowlist
                    val = m.group(0)
                    if is_allowlisted(val):
                        continue
                    # entropy check for generic API key to reduce FP: require high entropy or length
                    findings.append(
                        {
                            "rule_id": rule_id,
                            "category": "secrets",
                            "severity": severity,
                            "message": f"{msg} detected",
                            "file": rel,
                            "line": idx,
                            "evidence": matched[:200],
                            "fix": "Remove hardcoded credential, use environment variable or secret manager",
                        }
                    )
            except re.error:
                continue

        # High-entropy string detection for assignments
        # Look for  = "..." or : "..." with long random string
        assign_match = re.search(r"[:=]\s*['\"]([^'\"]{20,})['\"]", line)
        if assign_match:
            val = assign_match.group(1)
            if len(val) >= MIN_HIGH_ENTROPY_LEN and not is_allowlisted(val):
                ent = shannon_entropy(val)
                # Check if looks like hex/base64/random
                has_mixed = bool(re.search(r"[A-Z]", val) and re.search(r"[a-z]", val)) or bool(
                    re.search(r"[0-9]", val) and re.search(r"[A-Za-z]", val)
                )
                if ent >= ENTROPY_THRESHOLD and has_mixed:
                    # Avoid flagging URLs or paths
                    if not val.startswith("http") and "/" not in val and "\\" not in val:
                        # Check surrounding line mentions secret-like key
                        if re.search(r"(?i)(key|secret|token|password|credential|auth)", line):
                            findings.append(
                                {
                                    "rule_id": "SEC-100",
                                    "category": "secrets",
                                    "severity": "high",
                                    "message": f"High-entropy string (entropy={ent:.2f}) likely credential",
                                    "file": rel,
                                    "line": idx,
                                    "evidence": f"{val[:30]}... (entropy={ent:.2f}, len={len(val)})",
                                    "fix": "Replace with env var lookup, e.g. os.environ['API_KEY']",
                                }
                            )

    # Unsafe secret handling: logging secrets
    for idx, line in enumerate(lines, start=1):
        if re.search(r"(?i)print\s*\(.*(api_key|secret|password|token)", line) or re.search(
            r"(?i)logger\.(info|debug|error).*?(key|secret|password)", line
        ):
            findings.append(
                {
                    "rule_id": "SEC-201",
                    "category": "secrets",
                    "severity": "medium",
                    "message": "Potential secret logged to stdout/logger",
                    "file": rel,
                    "line": idx,
                    "evidence": line.strip()[:200],
                    "fix": "Do not log secrets; redact or remove",
                }
            )
        if re.search(r"(?i)requests\.(get|post).*?(api_key|secret|token)\s*[:=]", line):
            # check if secret in URL
            if "params" in line or "url" in line.lower():
                findings.append(
                    {
                        "rule_id": "SEC-202",
                        "category": "secrets",
                        "severity": "medium",
                        "message": "Secret may be transmitted in URL / query params (logged risk)",
                        "file": rel,
                        "line": idx,
                        "evidence": line.strip()[:200],
                        "fix": "Send secrets in headers or POST body, not URL",
                    }
                )

    return findings


def scan_skill_secrets(skill_path: Path) -> List[Dict[str, Any]]:
    all_findings: List[Dict[str, Any]] = []
    for file in skill_path.rglob("*"):
        if not file.is_file():
            continue
        if file.suffix not in (
            ".py",
            ".js",
            ".ts",
            ".json",
            ".yaml",
            ".yml",
            ".env",
            ".sh",
            ".md",
            ".txt",
            ".toml",
            ".cfg",
            ".ini",
        ):
            continue
        # skip venv, git
        if ".git" in file.parts or "__pycache__" in file.parts or "node_modules" in file.parts:
            continue
        all_findings.extend(scan_file_for_secrets(file, skill_path))
    # Deduplicate by file:line:rule
    seen = set()
    uniq = []
    for f in all_findings:
        key = (f["file"], f["line"], f["rule_id"], f["evidence"])
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq
