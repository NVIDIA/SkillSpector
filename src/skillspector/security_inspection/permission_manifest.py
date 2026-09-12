"""
permission_manifest.py - Declare and validate skill capabilities.
Deterministic static analysis: regex/AST scan vs declared manifest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Patterns for capability detection (deterministic)
CAPABILITY_PATTERNS = {
    "file_access": {
        "read": [
            r"\bopen\s*\(",
            r"\bPath\s*\(",
            r"os\.path",
            r"pathlib",
            r"read_text",
            r"read_bytes",
            r"glob\.",
            r"rglob",
        ],
        "write": [
            r"\bopen\s*\([^)]*['\"]w['\"]",
            r"write_text",
            r"write_bytes",
            r"os\.remove",
            r"shutil\.",
            r"mkdir",
            r"unlink",
        ],
    },
    "network": {
        "patterns": [
            r"\brequests\.",
            r"\burllib\.",
            r"\bhttp\.client",
            r"\bsocket\.",
            r"\baiohttp",
            r"\bhttpx\.",
            r"\bwebsocket",
            r"fetch\s*\(",
            r"urllib\.request\.urlopen",
            r"socket\.create_connection",
        ]
    },
    "subprocess": {
        "patterns": [
            r"\bsubprocess\.",
            r"\bos\.system\s*\(",
            r"\bos\.popen\s*\(",
            r"\bexec\s*\(",
            r"\beval\s*\(",
            r"pty\.",
            r"multiprocessing",
        ]
    },
    "env": {"patterns": [r"os\.environ", r"os\.getenv", r"dotenv", r"load_dotenv"]},
    "secrets": {"patterns": [r"api_key", r"secret", r"token", r"password", r"credential"]},
}

# Allowed manifest schema
MANIFEST_SCHEMA_KEYS = {
    "name",
    "version",
    "description",
    "author",
    "permissions",
    "dependencies",
    "capabilities",
}


def load_manifest(skill_path: Path) -> tuple[dict[str, Any], str]:
    """Load manifest dict and source path string."""
    candidates = [
        skill_path / "skill.json",
        skill_path / "manifest.json",
        skill_path / "skill.yaml",
        skill_path / "skill.yml",
        skill_path / "manifest.yaml",
        skill_path / "manifest.yml",
    ]
    for p in candidates:
        if p.exists():
            try:
                if p.suffix == ".json":
                    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                    return data, str(p)
                else:
                    # yaml manual simple parse without dependency
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    # attempt json-like fallback
                    try:
                        import yaml

                        data = yaml.safe_load(text)
                        return data if isinstance(data, dict) else {}, str(p)
                    except ImportError:
                        # naive key:value parse
                        data = {}
                        for line in text.splitlines():
                            if ":" in line and not line.strip().startswith("#"):
                                k, v = line.split(":", 1)
                                data[k.strip()] = v.strip().strip("\"'")
                        return data, str(p)
            except Exception as e:
                return {"_parse_error": str(e)}, str(p)
    return {}, ""  # no manifest


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []
    if not manifest:
        findings.append(
            {
                "rule_id": "PERM-001",
                "category": "permission",
                "severity": "medium",
                "message": "Missing permission manifest (skill.json/manifest.json). Using defaults; declare explicit permissions.",
                "file": "manifest",
                "line": None,
                "evidence": "No manifest found",
            }
        )
        return findings
    perms = manifest.get("permissions", {})
    if not isinstance(perms, dict):
        findings.append(
            {
                "rule_id": "PERM-002",
                "category": "permission",
                "severity": "high",
                "message": "Malformed permissions field - must be object/dict",
                "file": "manifest",
                "line": None,
                "evidence": str(perms)[:200],
            }
        )
        return findings
    # check known keys
    allowed_network = {"none", "loopback", "outbound", "unrestricted"}
    if "network" in perms and perms["network"] not in allowed_network:
        findings.append(
            {
                "rule_id": "PERM-003",
                "category": "permission",
                "severity": "medium",
                "message": f"Invalid network permission value '{perms['network']}' - allowed: {allowed_network}",
                "file": "manifest",
                "line": None,
                "evidence": str(perms["network"]),
            }
        )
    # check for overly broad
    if perms.get("network") == "unrestricted":
        findings.append(
            {
                "rule_id": "PERM-004",
                "category": "permission",
                "severity": "medium",
                "message": "Network permission is 'unrestricted' - consider least privilege (loopback/outbound)",
                "file": "manifest",
                "line": None,
                "evidence": "network=unrestricted",
            }
        )
    if perms.get("file_access") == "unrestricted":
        findings.append(
            {
                "rule_id": "PERM-005",
                "category": "permission",
                "severity": "medium",
                "message": "File access is 'unrestricted' - consider read_only/read_write least privilege",
                "file": "manifest",
                "line": None,
                "evidence": "file_access=unrestricted",
            }
        )
    if perms.get("subprocess") is True:
        findings.append(
            {
                "rule_id": "PERM-006",
                "category": "permission",
                "severity": "info",
                "message": "Subprocess execution explicitly allowed - ensure justified",
                "file": "manifest",
                "line": None,
                "evidence": "subprocess=true",
            }
        )
    return findings


def scan_capabilities(skill_path: Path) -> dict[str, Any]:
    """Scan code for actual capabilities via deterministic regex/AST."""
    detected: dict[str, Any] = {
        "file_read": False,
        "file_write": False,
        "network": False,
        "subprocess": False,
        "env": False,
        "details": {},
    }
    evidences: dict[str, list[dict[str, Any]]] = {
        k: [] for k in ["file_read", "file_write", "network", "subprocess", "env"]
    }

    # collect all text files
    for file in skill_path.rglob("*"):
        if not file.is_file():
            continue
        if file.suffix not in (".py", ".js", ".ts", ".sh", ".json", ".yaml", ".yml", ".md", ".txt"):
            continue
        # skip large binaries
        try:
            if file.stat().st_size > 2_000_000:
                continue
            text = file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rel = str(file.relative_to(skill_path))

        # file read
        for pat in CAPABILITY_PATTERNS["file_access"]["read"]:
            if re.search(pat, text):
                detected["file_read"] = True
                evidences["file_read"].append({"file": rel, "pattern": pat})
                break
        for pat in CAPABILITY_PATTERNS["file_access"]["write"]:
            if re.search(pat, text):
                detected["file_write"] = True
                evidences["file_write"].append({"file": rel, "pattern": pat})
                break
        for cat in ["network", "subprocess", "env"]:
            for pat in CAPABILITY_PATTERNS[cat]["patterns"]:
                if re.search(pat, text, re.IGNORECASE):
                    detected[cat] = True
                    evidences[cat].append({"file": rel, "pattern": pat})
                    break
    detected["details"] = evidences
    return detected


def compare_manifest_vs_actual(
    manifest: dict[str, Any], actual: dict[str, Any], skill_path: Path
) -> list[dict[str, Any]]:
    """Flag undeclared capabilities (manifest says none but code does network etc)."""
    findings: list[dict[str, Any]] = []
    perms = manifest.get("permissions", {}) if manifest else {}
    # normalize
    declared_network = perms.get("network", "none") if perms else "none"
    declared_subprocess = perms.get("subprocess", False) if perms else False
    declared_file = perms.get("file_access", "none") if perms else "none"

    if actual.get("network") and declared_network == "none":
        ev = actual["details"].get("network", [{}])[0]
        findings.append(
            {
                "rule_id": "PERM-101",
                "category": "permission",
                "severity": "high",
                "message": "Undeclared network capability - code performs network calls but manifest declares network=none",
                "file": ev.get("file", "unknown"),
                "line": None,
                "evidence": ev.get("pattern", "network pattern"),
                "fix": "Declare permissions.network='outbound' or remove network code",
            }
        )
    if actual.get("subprocess") and not declared_subprocess:
        ev = actual["details"].get("subprocess", [{}])[0]
        findings.append(
            {
                "rule_id": "PERM-102",
                "category": "permission",
                "severity": "high",
                "message": "Undeclared subprocess capability - code uses subprocess/exec but manifest forbids it",
                "file": ev.get("file", "unknown"),
                "line": None,
                "evidence": ev.get("pattern", "subprocess pattern"),
                "fix": "Declare permissions.subprocess=true or remove subprocess usage",
            }
        )
    if actual.get("file_write") and declared_file in ("none", "read_only"):
        ev = actual["details"].get("file_write", [{}])[0]
        findings.append(
            {
                "rule_id": "PERM-103",
                "category": "permission",
                "severity": "medium",
                "message": f"Undeclared file write - manifest file_access='{declared_file}' but code writes files",
                "file": ev.get("file", "unknown"),
                "line": None,
                "evidence": ev.get("pattern", "write pattern"),
                "fix": "Declare file_access='read_write' or remove write operations",
            }
        )
    if actual.get("file_read") and declared_file == "none":
        ev = actual["details"].get("file_read", [{}])[0]
        findings.append(
            {
                "rule_id": "PERM-104",
                "category": "permission",
                "severity": "medium",
                "message": "Undeclared file read - manifest declares no file access but code reads files",
                "file": ev.get("file", "unknown"),
                "line": None,
                "evidence": ev.get("pattern", "read pattern"),
                "fix": "Declare file_access='read_only'",
            }
        )
    return findings
