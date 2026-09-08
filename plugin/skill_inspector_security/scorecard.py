"""
scorecard.py - Compute per-skill security rating from all findings.
Deterministic, explainable scoring.
"""
from __future__ import annotations
from typing import Dict, List, Any

SEVERITY_WEIGHTS = {
    "critical": 25,
    "high": 15,
    "medium": 7,
    "low": 3,
    "info": 1,
}

CATEGORY_WEIGHTS = {
    "secrets": 1.5,
    "permission": 1.2,
    "privacy": 1.3,
    "subprocess": 1.2,
    "network": 1.1,
    "provenance": 0.8,
    "diff": 1.0,
    "dependency": 0.9,
}

def grade_from_score(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 65:
        return "C"
    if score >= 45:
        return "D"
    return "F"

def compute_scorecard(findings: List[Dict[str, Any]], provenance: Dict[str, Any] = None, privacy: Dict[str, Any] = None, sbom_components: int = 0) -> Dict[str, Any]:
    """
    Findings each have severity/category.
    Score starts 100, deductions per finding weighted.
    """
    base = 100
    deductions = 0
    breakdown: List[Dict[str, Any]] = []
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}

    for f in findings:
        sev = f.get("severity", "medium").lower()
        cat = f.get("category", "info").lower()
        w = SEVERITY_WEIGHTS.get(sev, 5)
        cat_mult = CATEGORY_WEIGHTS.get(cat, 1.0)
        deduct = int(w * cat_mult)
        # cap individual deductions to avoid negative overflow; we sum then cap total
        deductions += deduct
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
        breakdown.append({
            "rule_id": f.get("rule_id"),
            "severity": sev,
            "category": cat,
            "message": f.get("message"),
            "deduction": deduct,
            "evidence": f.get("evidence", "")[:120],
        })

    # Additional penalties
    if provenance and provenance.get("author") == "unknown":
        deductions += 3
        breakdown.append({"rule_id": "SCORE-PROV", "severity": "low", "category": "provenance", "message": "Missing author", "deduction": 3})
    if privacy and privacy.get("flows", {}).get("risk") == "high":
        deductions += 10
        breakdown.append({"rule_id": "SCORE-PRIVACY", "severity": "high", "category": "privacy", "message": "High-risk data transmission", "deduction": 10})

    score = max(0, base - deductions)
    grade = grade_from_score(score)

    # Explanations
    if score >= 90:
        explanation = "Excellent posture. Minor or no findings. Safe to use."
    elif score >= 80:
        explanation = "Good posture with low-risk findings. Review infos before production."
    elif score >= 65:
        explanation = "Moderate risk. Medium findings require attention; restrict permissions."
    elif score >= 45:
        explanation = "High risk. Critical/high findings present. Isolate and remediate."
    else:
        explanation = "Critical risk. Multiple severe findings. Do not use without remediation."

    # Top risks sorted by deduction
    top_risks = sorted(breakdown, key=lambda x: x["deduction"], reverse=True)[:5]

    return {
        "score": score,
        "grade": grade,
        "base": base,
        "deductions": deductions,
        "severity_counts": severity_counts,
        "total_findings": len(findings),
        "breakdown": breakdown,
        "top_risks": top_risks,
        "explanation": explanation,
        "recommendations": generate_recommendations(findings, privacy, provenance),
    }

def generate_recommendations(findings: List[Dict[str, Any]], privacy: Dict[str, Any] = None, provenance: Dict[str, Any] = None) -> List[str]:
    recs: List[str] = []
    cats = set(f.get("category") for f in findings)
    sevs = set(f.get("severity") for f in findings)
    if "secrets" in cats:
        recs.append("Remove hardcoded credentials; use environment variables or vault")
    if "permission" in cats:
        recs.append("Declare least-privilege permissions in skill.json (file_access, network, subprocess)")
    if "privacy" in cats or (privacy and privacy.get("flows", {}).get("transmits")):
        recs.append("Minimize sensitive data handling; ensure TLS and audit data flows")
    if "provenance" in cats or (provenance and provenance.get("author") == "unknown"):
        recs.append("Add author/origin and signed hashes to manifest for provenance")
    if "critical" in sevs or "high" in sevs:
        recs.append("Prioritize critical/high findings before deploying skill")
    if not recs:
        recs.append("No major actions; maintain current hygiene and re-scan on updates")
    return recs
