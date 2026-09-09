"""Offline Security Inspection Plugin for NVIDIA Skill Inspector.

100% local, deterministic, no cloud, no LLM, no telemetry.
Aggregates plugin checks into Skillspector findings:
 - Secrets & credential-flow (regex + entropy)
 - Permission / capability manifest validation
 - Privacy / data classification (read/write/transmit)
 - Dependency graph (networkx) with cycle detection
 - Provenance & reputation (hashes, author/origin)
 - Diff security (git plumbing / difflib)
 - SBOM (CycloneDX) generation
 - Scorecard (0-100 deterministic)

All storage under ~/.skill-inspector/ (SQLite).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skillspector.logging_config import get_logger
from skillspector.models import Finding, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

logger = get_logger(__name__)

ANALYZER_ID = "offline_security_inspection"

# Map plugin severity -> Skillspector Severity enum string
_SEV_MAP = {
    "critical": Severity.CRITICAL,
    "high": Severity.HIGH,
    "medium": Severity.MEDIUM,
    "low": Severity.LOW,
    "info": Severity.LOW,
}

# Map plugin category -> skillspector category (keep original but ensure consistent)
_CATEGORY_MAP = {
    "secrets": "supply_chain",
    "permission": "privilege_escalation",
    "privacy": "data_exfiltration",
    "provenance": "supply_chain",
    "diff": "supply_chain",
    "network": "data_exfiltration",
    "subprocess": "privilege_escalation",
    "file_access": "privilege_escalation",
}


def _plugin_severity_to_enum(s: str) -> Severity:
    return _SEV_MAP.get(s.lower(), Severity.MEDIUM)


def _to_finding(pf: dict[str, Any], default_file: str = "SKILL.md") -> Finding:
    """Convert plugin finding dict to Skillspector Finding."""
    sev = _plugin_severity_to_enum(pf.get("severity", "medium"))
    rule = pf.get("rule_id", "OFFLINE-001")
    cat = pf.get("category", "supply_chain")
    # Normalize category for SARIF
    cat = _CATEGORY_MAP.get(cat, cat)
    fpath = pf.get("file", default_file) or default_file
    # ensure relative path
    if fpath.startswith("/") or fpath.startswith("\\"):
        fpath = Path(fpath).name
    # some plugin findings use skill name as file
    if not fpath or fpath == "manifest":
        fpath = "skill.json"
    if fpath == "diff":
        fpath = "SKILL.md"
    try:
        start_line = int(pf.get("line")) if pf.get("line") is not None else 1
    except Exception:
        start_line = 1
    if start_line < 1:
        start_line = 1
    msg = pf.get("message", "")
    evidence = pf.get("evidence", "")
    remediation = pf.get("fix")
    # Confidence: deterministic static => 0.85, critical/high higher
    confidence = 0.9 if sev in (Severity.CRITICAL, Severity.HIGH) else 0.85
    # Add tags for offline provenance
    tags = ["offline", "deterministic", "no-llm", pf.get("category", "offline")]
    return Finding(
        rule_id=rule,
        message=msg,
        severity=sev.value if isinstance(sev, Severity) else str(sev),
        confidence=confidence,
        file=fpath,
        start_line=start_line,
        end_line=None,
        category=cat,
        pattern=rule,
        finding=evidence[:500] if evidence else None,
        explanation=msg,
        remediation=remediation,
        code_snippet=evidence[:500] if evidence else None,
        tags=tags,
        matched_text=evidence[:300] if evidence else None,
        evidence={"offline_plugin": True, "raw_category": pf.get("category"), "fix": remediation},
    )


def analyze(state: SkillspectorState) -> list[Finding]:
    """Run offline security inspection plugin analysis.

    Uses deterministic local scans only; no network.
    """
    skill_path_raw = state.get("skill_path") or state.get("input_path") or "."
    skill_path = Path(str(skill_path_raw)).resolve()
    if not skill_path.exists():
        logger.warning("offline_security_inspection: skill_path %s does not exist", skill_path)
        return []

    # Import lazily to avoid circular deps and keep offline isolation
    try:
        from skillspector.security_inspection.scanner import SecurityScanner
    except ImportError as e:
        logger.error(
            "offline_security_inspection: failed to import security_inspection package: %s", e
        )
        return []

    # Run scanner (deterministic, offline). This populates ~/.skill-inspector/ SQLite + SBOM + report
    # Use a bounded scan: if skill_path contains many sub-skills, scanner handles dependency graph.
    try:
        scanner = SecurityScanner(skill_path)
        result = scanner.scan()
    except Exception as e:
        logger.error("offline_security_inspection scan failed: %s", e)
        return [
            Finding(
                rule_id="OFFLINE-ERR",
                message=f"Offline security scan failed: {e}",
                severity=Severity.MEDIUM.value,
                confidence=0.99,
                file="SKILL.md",
                start_line=1,
                category="supply_chain",
                pattern="OFFLINE-ERR",
                explanation=f"Scanner error: {e}",
                tags=["offline", "error"],
            )
        ]

    # Store summary in state for report node / security-report CLI
    # State is mutable dict; store under well-known key for later use
    try:
        state["offline_report"] = result  # type: ignore
        state["offline_graph"] = result.get("graph", {})
        state["offline_scorecard"] = {
            s["name"]: s.get("scorecard", {}) for s in result.get("skills", [])
        }
    except Exception:
        pass

    findings: list[Finding] = []
    for skill in result.get("skills", []):
        skill_name = skill.get("name", "unknown")
        # Convert each plugin finding
        for pf in skill.get("findings", []):
            # Skip info-only provenance findings that would be noisy, but keep them as LOW with tag
            f = _to_finding(
                pf,
                default_file=f"{skill_name}/SKILL.md"
                if len(result.get("skills", [])) > 1
                else "SKILL.md",
            )
            # If multi-skill, prefix file with skill name for clarity
            if len(result.get("skills", [])) > 1 and not f.file.startswith(skill_name):
                f.file = f"{skill_name}/{f.file}"
            findings.append(f)

        # Add scorecard aggregate as informational finding if score low
        sc = skill.get("scorecard", {})
        if sc and sc.get("score", 100) < 50:
            findings.append(
                Finding(
                    rule_id="OFFLINE-SCORE",
                    message=f"Skill '{skill_name}' security score {sc.get('score')}/100 (grade {sc.get('grade')}) - {sc.get('explanation', '')}",
                    severity=Severity.HIGH.value
                    if sc.get("score", 0) < 45
                    else Severity.MEDIUM.value,
                    confidence=0.95,
                    file=f"{skill_name}/skill.json"
                    if len(result.get("skills", [])) > 1
                    else "skill.json",
                    start_line=1,
                    category="supply_chain",
                    pattern="OFFLINE-SCORE",
                    explanation=sc.get("explanation", ""),
                    remediation="; ".join(sc.get("recommendations", [])[:2]),
                    tags=["offline", "scorecard", f"grade-{sc.get('grade', 'F')}"],
                    evidence={
                        "score": sc.get("score"),
                        "grade": sc.get("grade"),
                        "breakdown": sc.get("breakdown", [])[:3],
                    },
                )
            )

    # Add graph cycle finding if present
    metrics = result.get("graph_metrics", {})
    cycles = metrics.get("cycles", [])
    if cycles:
        findings.append(
            Finding(
                rule_id="OFFLINE-GRAPH-CYCLE",
                message=f"Dependency cycle detected: {cycles[:2]}",
                severity=Severity.MEDIUM.value,
                confidence=0.9,
                file="SKILL.md",
                start_line=1,
                category="supply_chain",
                pattern="OFFLINE-GRAPH-CYCLE",
                explanation="Circular dependency between skills may cause load-order or trust issues",
                tags=["offline", "dependency_graph", "cycle"],
                evidence={"cycles": cycles},
            )
        )

    # Cap findings to avoid ledger overflow (analyzer ceiling)
    if len(findings) > 5000:
        logger.warning("offline_security_inspection truncated %d findings to 5000", len(findings))
        findings = findings[:5000]

    logger.info(
        "%s: %d findings from %d skills (offline, deterministic)",
        ANALYZER_ID,
        len(findings),
        len(result.get("skills", [])),
    )
    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Analyzer node entry point for LangGraph."""
    findings = analyze(state)
    return {"findings": findings}
