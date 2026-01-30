"""
SkillSpector - Security Scanner for AI Agent Skills

Detect vulnerabilities, malicious patterns, and security risks in agent skills
before installation.
"""

__version__ = "0.1.2"
__author__ = "SkillSpector Team"

from skillspector.scanner import SkillScanner
from skillspector.models import (
    ScanResult,
    RiskAssessment,
    SecurityIssue,
    Component,
    Severity,
)

__all__ = [
    "SkillScanner",
    "ScanResult",
    "RiskAssessment",
    "SecurityIssue",
    "Component",
    "Severity",
    "__version__",
]
