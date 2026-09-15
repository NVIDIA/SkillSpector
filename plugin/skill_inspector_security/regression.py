"""regression.py - Security Regression Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def compare_reports(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two scanner reports (old vs new)."""

    # Build rule -> findings map
    def findings_set(report):
        s = set()
        for skill in report.get("skills", []):
            for f in skill.get("findings", []):
                s.add((f.get("rule_id"), f.get("file"), f.get("message")[:60]))
        return s

    old_set = findings_set(old)
    new_set = findings_set(new)

    added = new_set - old_set
    removed = old_set - new_set

    # Capabilities diff
    def caps(report):
        c = set()
        for skill in report.get("skills", []):
            caps = skill.get("capabilities", {})
            for k, v in caps.items():
                if v:
                    c.add(k)
            # also drift
            for f in skill.get("findings", []):
                if f.get("rule_id", "").startswith("DRIFT"):
                    c.add(f.get("capability", ""))
        return c

    old_caps = caps(old)
    new_caps = caps(new)

    added_caps = new_caps - old_caps
    removed_caps = old_caps - new_caps

    old_score = old.get("summary", {}).get("avg_score", 0)
    new_score = new.get("summary", {}).get("avg_score", 0)
    old_grade = _grade(old_score)
    new_grade = _grade(new_score)

    # Decision
    if new_score < old_score - 10 or new_grade > old_grade:  # grade worsened (F > A)
        # grade comparison: F is worse, need ordinal
        order = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
        if order.get(new_grade, 0) > order.get(old_grade, 0) or new_score < 45:
            decision = "BLOCK"
        else:
            decision = "WARN"
    elif added:
        decision = "WARN"
    else:
        decision = "PASS"

    return {
        "added": sorted(added),
        "removed": sorted(removed),
        "added_caps": sorted(added_caps),
        "removed_caps": sorted(removed_caps),
        "old_score": old_score,
        "new_score": new_score,
        "old_grade": old_grade,
        "new_grade": new_grade,
        "risk_delta": new_score - old_score,
        "decision": decision,
        "summary": f"NEW {len(added)} REMOVED {len(removed)} RISK {old_score}->{new_score} Grade {old_grade}->{new_grade} Decision {decision}",
    }


def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 65:
        return "C"
    if score >= 45:
        return "D"
    return "F"


def regression_to_findings(reg: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings = []
    for cap in reg["added_caps"]:
        findings.append(
            {
                "rule_id": "REG-NEW-CAP",
                "category": "regression",
                "severity": "high",
                "message": f"New capability added: {cap}",
                "file": "regression",
                "line": None,
                "evidence": f"old {reg['old_score']} -> new {reg['new_score']}",
            }
        )
    if reg["decision"] == "BLOCK":
        findings.append(
            {
                "rule_id": "REG-BLOCK",
                "category": "regression",
                "severity": "critical",
                "message": f"Security regression: {reg['summary']}",
                "file": "regression",
                "line": None,
                "evidence": reg["summary"],
            }
        )
    return findings
