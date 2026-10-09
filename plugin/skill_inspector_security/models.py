"""
models.py - Dataclasses for findings, skills, manifests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Category(str, Enum):
    SECRETS = "secrets"
    PERMISSION = "permission"
    NETWORK = "network"
    SUBPROCESS = "subprocess"
    FILE_ACCESS = "file_access"
    PRIVACY = "privacy"
    DEPENDENCY = "dependency"
    PROVENANCE = "provenance"
    SBOM = "sbom"
    DIFF = "diff"


@dataclass
class Finding:
    rule_id: str
    category: str
    severity: str
    message: str
    file: str
    line: int | None = None
    evidence: str | None = None
    fix: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SkillInfo:
    name: str
    version: str
    path: str
    author: str = "unknown"
    description: str = ""
    hash_sha256: str = ""
    manifest: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    scorecard: Dict[str, Any] = field(default_factory=dict)
    privacy: Dict[str, Any] = field(default_factory=dict)
    sbom_ref: str = ""


@dataclass
class PermissionManifest:
    """Declarative manifest of what skill is allowed to do."""

    name: str
    version: str = "1.0.0"
    permissions: Dict[str, Any] = field(default_factory=dict)
    # permissions keys: file_access, network, subprocess, env, secrets

    ALLOWED_KEYS = {"file_access", "network", "subprocess", "env", "secrets", "capabilities"}
    NETWORK_VALUES = {"none", "loopback", "outbound", "unrestricted"}
    FILE_VALUES = {"none", "read_only", "read_write", "unrestricted"}

    @staticmethod
    def default_manifest(skill_name: str) -> Dict[str, Any]:
        return {
            "name": skill_name,
            "version": "1.0.0",
            "permissions": {
                "file_access": "read_only",
                "network": "none",
                "subprocess": False,
                "env": [],
                "capabilities": [],
            },
        }
