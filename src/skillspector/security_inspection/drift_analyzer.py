"""drift_analyzer.py - Declared vs Static vs Runtime permission drift."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Set

from .event_model import SecurityEvent

DRIFT_CLASS = ["MATCH", "UNDER_DECLARED", "OVER_DECLARED", "UNUSED", "UNKNOWN"]

def normalize_cap(s: str) -> str:
    s = s.lower().strip()
    # map to canonical: filesystem.read, filesystem.write, network.outbound, network.dns, processes.execute, env.read, etc
    if "filesystem" in s or "file" in s:
        if "write" in s: return "filesystem.write"
        return "filesystem.read"
    if "network" in s or "outbound" in s or "dns" in s:
        return "network.outbound" if "outbound" in s or "network" in s else "network.dns"
    if "subprocess" in s or "process" in s:
        return "processes.execute"
    if "env" in s:
        return "env.read"
    if "execut" in s:
        return "executables.execute"
    if "package" in s:
        return "package.install"
    if "mcp" in s:
        return "mcp.call"
    return s

def capabilities_to_set(perms: Dict[str, Any], static_caps: Dict[str, Any], runtime_caps: Dict[str, bool]) -> tuple[Set[str], Set[str], Set[str]]:
    declared: Set[str] = set()
    # permissions manifest
    if perms.get("file_access") not in (None, "none"):
        # read always if file_access is read_only etc
        if perms["file_access"] in ("read_only", "read_write", "unrestricted"):
            declared.add("filesystem.read")
        if perms["file_access"] in ("read_write", "unrestricted"):
            declared.add("filesystem.write")
    if perms.get("network") not in (None, "none"):
        declared.add("network.outbound")
        declared.add("network.dns")
    if perms.get("subprocess"):
        declared.add("processes.execute")
    if perms.get("env"):
        if perms["env"]:
            declared.add("env.read")
    # static
    static: Set[str] = set()
    if static_caps.get("file_read"): static.add("filesystem.read")
    if static_caps.get("file_write"): static.add("filesystem.write")
    if static_caps.get("network"): static.add("network.outbound"); static.add("network.dns")
    if static_caps.get("subprocess"): static.add("processes.execute")
    if static_caps.get("env"): static.add("env.read")
    # runtime
    runtime: Set[str] = set()
    for k, v in runtime_caps.items():
        if not v: continue
        if k == "filesystem_read": runtime.add("filesystem.read")
        if k == "filesystem_write": runtime.add("filesystem.write")
        if k == "network": runtime.add("network.outbound")
        if k == "subprocess": runtime.add("processes.execute")
        if k == "env": runtime.add("env.read")
        if k == "executables": runtime.add("executables.execute")
        if k == "package": runtime.add("package.install")
        if k == "mcp": runtime.add("mcp.call")
    return declared, static, runtime

def classify_drift(declared: Set[str], static: Set[str], runtime: Set[str]) -> List[Dict[str, Any]]:
    all_caps = declared | static | runtime
    # also include known universe to detect UNUSED
    universe = {"filesystem.read","filesystem.write","network.outbound","network.dns","processes.execute","env.read","executables.execute","package.install","mcp.call"}
    all_caps |= universe
    results = []
    for cap in sorted(all_caps):
        in_decl = cap in declared
        in_static = cap in static
        in_runtime = cap in runtime
        if in_decl and in_static and in_runtime:
            drift = "MATCH"
            severity = "info"
        elif not in_decl and (in_static or in_runtime):
            drift = "UNDER_DECLARED"
            severity = "high"
        elif in_decl and not in_static and not in_runtime:
            drift = "OVER_DECLARED"
            severity = "low"
        elif in_decl and in_static and not in_runtime:
            drift = "UNUSED"
            severity = "low"
        elif not in_decl and not in_static and in_runtime:
            drift = "UNKNOWN"
            severity = "high"
        else:
            drift = "MATCH"
            severity = "info"
        # Determine evidence
        if drift != "MATCH":
            results.append({
                "capability": cap,
                "drift": drift,
                "severity": severity,
                "declared": in_decl, "static": in_static, "runtime": in_runtime,
                "message": f"{cap}: declared={in_decl} static={in_static} runtime={in_runtime} -> {drift}",
            })
    return results

def drift_to_findings(drifts: List[Dict[str, Any]], skill_name: str) -> List[Dict[str, Any]]:
    findings = []
    for d in drifts:
        sev = d["severity"]
        rule = f"DRIFT-{d['drift']}"
        findings.append({
            "rule_id": rule,
            "category": "permission",
            "severity": sev,
            "message": f"Permission drift {d['drift']} for {d['capability']} ({skill_name})",
            "file": "manifest",
            "line": None,
            "evidence": d["message"],
            "fix": "Align manifest.json permissions with observed static+runtime capabilities" if d["drift"]=="UNDER_DECLARED" else "Remove unused permission declaration",
            "drift": d["drift"],
            "capability": d["capability"],
        })
    return findings

def drift_to_events(drifts: List[Dict[str, Any]], subject: str) -> List[SecurityEvent]:
    evs = []
    for d in drifts:
        if d["drift"] == "MATCH": continue
        cat = d["capability"].split(".")[0]
        evs.append(SecurityEvent(
            source="drift",
            category=cat,
            action="drift",
            subject=subject,
            target=d["capability"],
            capability=d["capability"],
            evidence=d["message"],
            severity=d["severity"],
            confidence=0.9,
            rule_id=f"DRIFT-{d['drift']}",
            metadata={"drift": d["drift"], "declared": d["declared"], "static": d["static"], "runtime": d["runtime"]},
        ))
    return evs
