"""correlation_engine.py - Attack-path correlation."""
from __future__ import annotations
from typing import List, Dict, Any
from .event_model import SecurityEvent, EventGraph
import re

# Define attack templates as ordered capability sequences
ATTACK_TEMPLATES = [
    {
        "id": "CRED_EXFIL",
        "name": "Credential exfiltration",
        "sequence": ["filesystem.read", "processes.execute", "network.outbound"],
        "indicators": [r"aws|credential|secret|token", r"subprocess|os\.system", r"requests|socket|urllib"],
        "severity": "critical",
    },
    {
        "id": "DATA_EXFIL",
        "name": "Data exfiltration via network",
        "sequence": ["filesystem.read", "network.outbound"],
        "severity": "high",
    },
    {
        "id": "ENV_EXFIL",
        "name": "Environment harvesting to network",
        "sequence": ["env.read", "network.outbound"],
        "severity": "high",
    },
    {
        "id": "EXEC_CHAIN",
        "name": "Executable chain",
        "sequence": ["filesystem.write", "executables.execute", "processes.execute"],
        "severity": "critical",
    },
    {
        "id": "PKG_SUPPLY",
        "name": "Package install + network",
        "sequence": ["package.install", "network.outbound"],
        "severity": "medium",
    },
]

def correlate(graph: EventGraph) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    # Group events by capability order
    caps = [e.capability for e in graph.events]
    # Also consider category+action as capability
    for tmpl in ATTACK_TEMPLATES:
        seq = tmpl["sequence"]
        # Check if sequence appears in order (subsequence)
        idx = 0
        matched: List[SecurityEvent] = []
        for ev in graph.events:
            if ev.capability == seq[idx] or ev.category in seq[idx] or seq[idx] in ev.capability:
                matched.append(ev)
                idx += 1
                if idx == len(seq):
                    break
        if idx == len(seq):
            # Validate indicators if present
            if "indicators" in tmpl:
                text = " ".join(e.evidence + " " + e.target for e in matched)
                ok = all(re.search(pat, text, re.I) for pat in tmpl["indicators"])
                if not ok:
                    continue
            # Build attack path finding
            path = " -> ".join(f"{e.category}/{e.action} ({e.target[:30]})" for e in matched)
            findings.append({
                "rule_id": f"COR-{tmpl['id']}",
                "category": "correlation",
                "severity": tmpl["severity"],
                "message": f"Attack path {tmpl['name']}: {path}",
                "file": matched[0].subject,
                "line": None,
                "evidence": path + " | " + " | ".join(e.evidence[:80] for e in matched),
                "fix": "Break chain: restrict file read or network egress",
                "attack_template": tmpl["id"],
                "matched_events": [e.to_dict() for e in matched],
            })
        # Also detect simple two-step: secret -> network
        # Check for secrets + network co-occurrence
    # Generic: if any secrets event + network event, emit
    has_secret = any(e.category == "secrets" or "secret" in e.target.lower() or "credential" in e.target.lower() for e in graph.events)
    has_network = any(e.category in ("network","outbound","dns") for e in graph.events)
    if has_secret and has_network:
        # Avoid duplicate if already CRED_EXFIL
        if not any(f["rule_id"] == "COR-CRED_EXFIL" for f in findings):
            findings.append({
                "rule_id": "COR-SECRET_NET",
                "category": "correlation",
                "severity": "high",
                "message": "Secret handling + network transmission observed -> potential exfiltration",
                "file": graph.events[0].subject if graph.events else "skill",
                "line": None,
                "evidence": f"secrets={has_secret} network={has_network}",
                "fix": "Ensure secrets not transmitted; use env vault",
            })
    return findings

def events_from_findings(findings: List[Dict[str, Any]], source: str, subject: str) -> List[SecurityEvent]:
    evs = []
    for f in findings:
        cat = f.get("category","correlation")
        evs.append(SecurityEvent(
            source=source,
            category=cat,
            action="correlate",
            subject=subject,
            target=f.get("file", subject),
            capability=f.get("rule_id", cat),
            evidence=f.get("message","") + " | " + f.get("evidence",""),
            severity=f.get("severity","medium"),
            confidence=0.9,
            rule_id=f.get("rule_id",""),
            metadata={"correlation": True},
        ))
    return evs
