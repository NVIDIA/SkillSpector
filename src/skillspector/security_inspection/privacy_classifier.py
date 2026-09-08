"""
privacy_classifier.py - Classify data types each skill reads/writes/transmits.
Deterministic pattern matching, no LLM.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Any

DATA_CATEGORIES = {
    "PII": {
        "patterns": [
            r"(?i)\b(email|e-mail)\b", r"(?i)\bphone\b", r"(?i)\bssn\b", r"(?i)social[_-]?security",
            r"(?i)\baddress\b", r"(?i)\bname\b.*\buser", r"(?i)\bdate[_-]?of[_-]?birth\b", r"(?i)\bdob\b",
            r"(?i)\bpassport", r"(?i)\bnational[_-]?id"
        ],
        "severity": "high"
    },
    "Credentials": {
        "patterns": [r"(?i)password", r"(?i)api[_-]?key", r"(?i)secret", r"(?i)token", r"(?i)credential", r"(?i)private[_-]?key"],
        "severity": "critical"
    },
    "Financial": {
        "patterns": [r"(?i)credit[_-]?card", r"(?i)card[_-]?number", r"(?i)cvv", r"(?i)bank[_-]?account", r"(?i)iban", r"(?i)payment"],
        "severity": "high"
    },
    "Health": {
        "patterns": [r"(?i)medical", r"(?i)health", r"(?i)diagnosis", r"(?i)patient", r"(?i)hipaa"],
        "severity": "high"
    },
    "Location": {
        "patterns": [r"(?i)latitude", r"(?i)longitude", r"(?i)geolocation", r"(?i)gps", r"(?i)location"],
        "severity": "medium"
    },
    "Biometric": {
        "patterns": [r"(?i)face[_-]?id", r"(?i)fingerprint", r"(?i)biometric", r"(?i)voice[_-]?print"],
        "severity": "high"
    },
    "System": {
        "patterns": [r"(?i)os\.environ", r"(?i)hostname", r"(?i)ip[_-]?address", r"/etc/passwd", r"/etc/shadow", r"user[_-]?data"],
        "severity": "medium"
    }
}

TRANSMIT_PATTERNS = [
    r"\brequests\.(get|post|put|delete)", r"\burllib\.request", r"\bhttp\.client", r"\bsocket\.send", r"\baiohttp", r"\bhttpx", r"\bwebsocket", r"\bsmtp", r"\bftp\.", r"fetch\s*\("
]
WRITE_PATTERNS = [
    r"\bopen\s*\([^)]*['\"]w", r"write_text", r"write_bytes", r"\.write\s*\(", r"shutil\.copy", r"os\.remove", r"json\.dump"
]
READ_PATTERNS = [
    r"\bopen\s*\([^)]*['\"]r", r"read_text", r"read_bytes", r"\.read\s*\(", r"csv\.reader", r"json\.load", r"pickle\.load"
]

def classify_skill_data(skill_path: Path) -> Dict[str, Any]:
    """Returns {categories: [...], flows: {reads, writes, transmits}, details: [...]}"""
    text_combined = ""
    file_details: List[Dict[str, Any]] = []
    for file in skill_path.rglob("*"):
        if not file.is_file() or file.suffix not in (".py", ".js", ".ts", ".json", ".yaml", ".yml", ".md", ".txt", ".sh"):
            continue
        if ".git" in file.parts or "__pycache__" in file.parts:
            continue
        try:
            if file.stat().st_size > 2_000_000:
                continue
            t = file.read_text(encoding="utf-8", errors="ignore")
            text_combined += "\n" + t
            rel = str(file.relative_to(skill_path))
            # per-file classification
            for cat, cfg in DATA_CATEGORIES.items():
                for pat in cfg["patterns"]:
                    if re.search(pat, t):
                        file_details.append({"file": rel, "category": cat, "pattern": pat, "severity": cfg["severity"]})
                        break
        except Exception:
            continue

    detected_cats: List[str] = []
    cat_findings: List[Dict[str, Any]] = []
    for cat, cfg in DATA_CATEGORIES.items():
        for pat in cfg["patterns"]:
            if re.search(pat, text_combined):
                detected_cats.append(cat)
                cat_findings.append({
                    "rule_id": f"PRIV-{cat[:4].upper()}",
                    "category": "privacy",
                    "severity": cfg["severity"],
                    "message": f"Skill handles {cat} data",
                    "file": skill_path.name,
                    "line": None,
                    "evidence": f"pattern: {pat}"
                })
                break

    reads = bool(any(re.search(p, text_combined) for p in READ_PATTERNS))
    writes = bool(any(re.search(p, text_combined) for p in WRITE_PATTERNS))
    transmits = bool(any(re.search(p, text_combined, re.IGNORECASE) for p in TRANSMIT_PATTERNS))

    flows = {
        "reads": reads,
        "writes": writes,
        "transmits": transmits,
        "risk": "high" if transmits and any(c in detected_cats for c in ["PII", "Credentials", "Financial", "Health"]) else ("medium" if transmits or writes else "low")
    }

    # Generate privacy findings for high-risk flows
    findings: List[Dict[str, Any]] = list(cat_findings)
    if transmits and detected_cats:
        findings.append({
            "rule_id": "PRIV-001",
            "category": "privacy",
            "severity": "high" if any(c in ["PII", "Credentials", "Financial"] for c in detected_cats) else "medium",
            "message": f"Skill transmits sensitive data ({', '.join(detected_cats)}) over network - requires encryption & consent",
            "file": skill_path.name,
            "line": None,
            "evidence": f"categories={detected_cats}, transmits={transmits}",
            "fix": "Ensure TLS, minimize data, declare in privacy manifest"
        })
    if writes and "Credentials" in detected_cats:
        findings.append({
            "rule_id": "PRIV-002",
            "category": "privacy",
            "severity": "critical",
            "message": "Skill writes credentials to disk - risk of plaintext storage",
            "file": skill_path.name,
            "line": None,
            "evidence": "Credentials + file write detected",
            "fix": "Use secure credential store, avoid plaintext"
        })

    return {
        "categories": sorted(set(detected_cats)),
        "flows": flows,
        "details": file_details,
        "findings": findings,
        "summary": f"Reads={reads}, Writes={writes}, Transmits={transmits} | Categories={detected_cats or ['none']}"
    }
