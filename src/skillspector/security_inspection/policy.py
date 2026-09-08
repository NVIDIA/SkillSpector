"""policy.py - Policy-as-Code enforcement."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

DEFAULT_POLICY = {
    "network": {"allow": [], "deny": ["*"], "mode": "block"},  # default block outbound
    "subprocess": {"allowed": False},
    "secrets": {"allowed": False},
    "filesystem": {"write": ["./workspace/**"], "read": ["./**"]},
    "package": {"allow": []},
}

def load_policy(path: Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return DEFAULT_POLICY
    txt = Path(path).read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(txt) or {}
        # normalize
        return {**DEFAULT_POLICY, **data}
    except Exception:
        return DEFAULT_POLICY

def _match_glob(pattern: str, target: str) -> bool:
    # simple glob: * and **
    regex = pattern.replace(".", r"\.").replace("**", ".*").replace("*", "[^/]*")
    return re.match(f"^{regex}$", target) is not None

def evaluate_policy(events: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for e in events:
        # e is finding dict or event dict
        cat = e.get("category","")
        sev = e.get("severity","medium")
        tgt = e.get("target") or e.get("file") or ""
        rule = e.get("rule_id","")
        # network
        if cat in ("network","outbound","dns") or "network" in str(e.get("capability","")):
            allow = policy.get("network", {}).get("allow", [])
            deny = policy.get("network", {}).get("deny", [])
            # if allow list not empty, check
            if allow:
                if not any(_match_glob(p, tgt) or p in tgt for p in allow):
                    violations.append({"policy": "network.allow", "severity": "high", "message": f"Network to {tgt} not in allow list {allow}", "evidence": str(e), "rule_id": "POL-NET"})
            if tgt and any(_match_glob(p, tgt) for p in deny):
                violations.append({"policy": "network.deny", "severity": "critical", "message": f"Network to {tgt} denied by policy", "evidence": str(e), "rule_id": "POL-NET-DENY"})
        # subprocess
        if cat in ("processes","subprocess") or e.get("capability")=="processes.execute":
            if not policy.get("subprocess", {}).get("allowed", False):
                violations.append({"policy": "subprocess.allowed", "severity": "high", "message": f"Subprocess not allowed but observed: {tgt}", "evidence": str(e), "rule_id": "POL-PROC"})
        # secrets
        if cat == "secrets" or "secret" in rule.lower():
            if not policy.get("secrets", {}).get("allowed", False):
                violations.append({"policy": "secrets.allowed", "severity": "critical", "message": f"Secrets handling not allowed: {rule}", "evidence": str(e), "rule_id": "POL-SEC"})
        # filesystem write
        if cat == "filesystem" and ("write" in str(e.get("capability","")) or e.get("action")=="write"):
            allowed_writes = policy.get("filesystem", {}).get("write", ["./workspace/**"])
            if not any(_match_glob(p, tgt) for p in allowed_writes):
                violations.append({"policy": "filesystem.write", "severity": "medium", "message": f"Write to {tgt} not in allowed {allowed_writes}", "evidence": str(e), "rule_id": "POL-FS"})
    # decision
    has_critical = any(v["severity"]=="critical" for v in violations)
    has_high = any(v["severity"]=="high" for v in violations)
    if has_critical or has_high:
        decision = "BLOCK"
    elif violations:
        decision = "WARN"
    else:
        decision = "PASS"
    return {"decision": decision, "violations": violations, "count": len(violations)}

def policy_to_findings(policy_result: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []
    for v in policy_result["violations"]:
        findings.append({
            "rule_id": v["rule_id"],
            "category": "policy",
            "severity": v["severity"],
            "message": v["message"],
            "file": "policy",
            "line": None,
            "evidence": v["evidence"][:300],
            "fix": "Update policy or remove capability",
        })
    return findings
