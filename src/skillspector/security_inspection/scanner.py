"""
scanner.py - Orchestrator: runs all analyzers deterministically, stores results.
"""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import get_data_dir, get_db_path, get_sbom_dir
from .correlation_engine import correlate
from .dependency_graph import build_dependency_graph, compute_metrics, graph_to_cytoscape
from .diff_security import diff_against_previous_version
from .drift_analyzer import capabilities_to_set, classify_drift, drift_to_events, drift_to_findings
from .event_model import EventGraph, SecurityEvent
from .permission_manifest import (
    compare_manifest_vs_actual,
    load_manifest,
    scan_capabilities,
    validate_manifest,
)
from .policy import evaluate_policy, load_policy, policy_to_findings
from .privacy_classifier import classify_skill_data
from .provenance import record_provenance
from .regression import compare_reports
from .sbom import generate_aggregate_sbom, generate_sbom, save_sbom
from .scorecard import compute_scorecard
from .secrets_analyzer import scan_skill_secrets
from .storage import init_db, save_scan


class SecurityScanner:
    def __init__(self, skills_root: Path, db_path: Path | None = None):
        self.skills_root = Path(skills_root).expanduser().resolve()
        self.db_path = db_path or get_db_path()
        init_db(self.db_path)
        self.data_dir = get_data_dir()

    def scan(self, previous_snapshot: Path | None = None, enable_runtime: bool = False, policy_path: Path | None = None, old_report: dict[str, Any] | None = None) -> dict[str, Any]:
        scan_id = f"scan-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        # Dependency graph
        G, discovered = build_dependency_graph(self.skills_root)
        graph_json = graph_to_cytoscape(G)
        metrics = compute_metrics(G)

        skills_results: list[dict[str, Any]] = []
        aggregate_sbom = generate_aggregate_sbom(self.skills_root, discovered)
        # Save aggregate SBOM
        save_sbom(aggregate_sbom, get_sbom_dir() / f"{scan_id}-aggregate.cdx.json")

        # Global event graph for correlation
        global_graph = EventGraph()
        policy = load_policy(policy_path) if policy_path else None

        for name, skill_path in sorted(discovered.items()):
            findings: list[dict[str, Any]] = []
            events = EventGraph()

            # Provenance
            prov = record_provenance(skill_path, self.skills_root, db_path=self.db_path)
            findings.extend(prov.get("findings", []))
            for f in prov.get("findings", []):
                events.add(SecurityEvent(source="provenance", category=f.get("category","provenance"), action="provenance", subject=name, target=f.get("file",""), capability=f.get("rule_id",""), evidence=f.get("message",""), severity=f.get("severity","low"), rule_id=f.get("rule_id","")))

            # Permission manifest
            manifest, manifest_src = load_manifest(skill_path)
            manifest_findings = validate_manifest(manifest)
            findings.extend(manifest_findings)
            actual_caps = scan_capabilities(skill_path)
            perm_mismatch = compare_manifest_vs_actual(manifest, actual_caps, skill_path)
            findings.extend(perm_mismatch)
            for f in manifest_findings + perm_mismatch:
                events.add(SecurityEvent(source="permission", category="permission", action="mismatch", subject=name, target=f.get("file",""), capability=f.get("rule_id",""), evidence=f.get("message",""), severity=f.get("severity","medium"), rule_id=f.get("rule_id","")))

            # Secrets
            secret_findings = scan_skill_secrets(skill_path)
            findings.extend(secret_findings)
            for f in secret_findings:
                events.add(SecurityEvent(source="static", category="secrets", action="detect", subject=f.get("file",""), target=f.get("evidence","")[:60], capability="secrets.detect", evidence=f.get("message",""), severity=f.get("severity","high"), rule_id=f.get("rule_id","")))

            # Privacy
            privacy = classify_skill_data(skill_path)
            findings.extend(privacy.get("findings", []))
            for f in privacy.get("findings", []):
                events.add(SecurityEvent(source="privacy", category="privacy", action="classify", subject=name, target=",".join(privacy.get("categories",[])), capability="privacy."+f.get("rule_id",""), evidence=f.get("message",""), severity=f.get("severity","medium"), rule_id=f.get("rule_id","")))

            # Diff (if previous snapshot or git)
            diff_res = diff_against_previous_version(skill_path, previous_snapshot / name if previous_snapshot else None)
            findings.extend(diff_res.get("findings", []))
            for f in diff_res.get("findings", []):
                events.add(SecurityEvent(source="diff", category="diff", action="diff", subject=name, target=f.get("file",""), capability=f.get("rule_id",""), evidence=f.get("message",""), severity=f.get("severity","medium"), rule_id=f.get("rule_id","")))

            # Runtime monitor (optional, isolated)
            runtime_caps: dict[str, bool] = {}
            runtime_graph = EventGraph()
            if enable_runtime:
                try:
                    from .runtime_monitor import collect_runtime_capabilities, run_isolated
                    runtime_graph = run_isolated(skill_path, timeout=8.0)
                    runtime_caps = collect_runtime_capabilities(runtime_graph)
                    # Add runtime events
                    for ev in runtime_graph.events:
                        events.add(ev)
                        global_graph.add(ev)
                    findings.extend([e.to_finding() for e in runtime_graph.events if e.severity in ("high","critical")])
                except Exception:
                    runtime_caps = {}

            # Drift analysis: declared vs static vs runtime
            perms = manifest.get("permissions", {}) if isinstance(manifest.get("permissions"), dict) else {}
            declared_set, static_set, runtime_set = capabilities_to_set(perms, actual_caps, runtime_caps)
            drifts = classify_drift(declared_set, static_set, runtime_set)
            drift_findings = drift_to_findings(drifts, name)
            findings.extend(drift_findings)
            for ev in drift_to_events(drifts, name):
                events.add(ev)
                global_graph.add(ev)

            # Correlation (per-skill)
            # Add all current findings as events for correlation
            for f in findings:
                # Avoid duplicating drift/secrets already added, but add generic
                pass
            corr_findings = correlate(events)
            findings.extend(corr_findings)
            for ev in corr_findings:
                # also add as event
                events.add(SecurityEvent(source="correlation", category="correlation", action="attack_path", subject=name, target=ev.get("rule_id",""), capability=ev.get("rule_id",""), evidence=ev.get("message",""), severity=ev.get("severity","high"), rule_id=ev.get("rule_id","")))

            # Policy enforcement
            policy_findings: list[dict[str, Any]] = []
            if policy:
                pol_res = evaluate_policy(findings + [e.to_dict() for e in events.events], policy)
                policy_findings = policy_to_findings(pol_res)
                findings.extend(policy_findings)
                # Store policy decision in skill
            # SBOM per skill
            sbom = generate_sbom(skill_path, self.skills_root)
            sbom_path = get_sbom_dir() / f"{scan_id}-{name}.cdx.json"
            save_sbom(sbom, sbom_path)

            # Scorecard (include drift/correlation in score)
            scorecard = compute_scorecard(findings, provenance=prov, privacy=privacy, sbom_components=len(sbom["components"]))

            # Privacy summary already

            skills_results.append({
                "name": name,
                "path": str(skill_path),
                "version": prov.get("version", "unknown"),
                "author": prov.get("author", "unknown"),
                "origin": prov.get("origin", "local"),
                "hash_sha256": prov.get("hash_sha256", ""),
                "manifest": manifest,
                "manifest_source": manifest_src,
                "capabilities": actual_caps,
                "runtime_capabilities": runtime_caps,
                "privacy": privacy,
                "findings": findings,
                "scorecard": scorecard,
                "sbom_path": str(sbom_path),
                "sbom_components": len(sbom["components"]),
                "diff_stats": diff_res.get("stats", {}),
                "events": [e.to_dict() for e in events.events],
                "drift": drifts,
                "correlation": corr_findings,
                "policy": policy_findings,
                "runtime_events": len(runtime_graph.events) if enable_runtime else 0,
            })

        # Global summary
        all_findings = [f for s in skills_results for f in s["findings"]]
        sev_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for f in all_findings:
            sev_counts[f.get("severity", "info")] = sev_counts.get(f.get("severity", "info"), 0) + 1

        avg_score = sum(s["scorecard"]["score"] for s in skills_results) / len(skills_results) if skills_results else 0

        # Global correlation across all skills
        global_attack_paths = correlate(global_graph)
        # Policy global decision
        policy_decision = None
        if policy:
            pol_global = evaluate_policy(all_findings, policy)
            policy_decision = pol_global
        # Regression vs old report
        regression = None
        if old_report:
            regression = compare_reports(old_report, {"skills": skills_results, "summary": {"avg_score": round(avg_score,1)}, "graph": graph_json})

        result: dict[str, Any] = {
            "scan_id": scan_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "skills_root": str(self.skills_root),
            "skills": skills_results,
            "graph": graph_json,
            "graph_metrics": metrics,
            "summary": {
                "total_skills": len(skills_results),
                "total_findings": len(all_findings),
                "severity_counts": sev_counts,
                "avg_score": round(avg_score, 1),
                "sbom_aggregate": str(get_sbom_dir() / f"{scan_id}-aggregate.cdx.json"),
            },
            "event_graph": global_graph.to_dict(),
            "attack_paths": global_attack_paths,
            "policy_decision": policy_decision,
            "regression": regression,
        }

        # Persist
        save_scan(scan_id, str(self.skills_root), result, self.db_path)

        # Also write JSON report to data_dir
        report_path = self.data_dir / "reports" / f"{scan_id}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["report_path"] = str(report_path)

        return result

    def scan_single(self, skill_path: Path) -> dict[str, Any]:
        # Convenience: scan containing directory as root
        root = skill_path.parent if skill_path.parent != self.skills_root else self.skills_root
        return self.scan()
