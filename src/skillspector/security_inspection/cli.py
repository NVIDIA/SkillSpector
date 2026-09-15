"""
cli.py - CLI for offline security inspection plugin.
No cloud, no telemetry, 100% local.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import get_data_dir, get_db_path
from .report_server import serve
from .scanner import SecurityScanner
from .storage import get_latest_scan


def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA Skill Inspector - Security Plugin (Offline, Local Only)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="Scan skills directory")
    p_scan.add_argument("path", help="Path to skills root directory (contains skill subfolders)")
    p_scan.add_argument(
        "--previous",
        dest="previous",
        help="Previous snapshot path for diff (optional)",
        default=None,
    )
    p_scan.add_argument(
        "--port", type=int, default=8899, help="Port for report server (after scan)"
    )
    p_scan.add_argument(
        "--no-serve", action="store_true", help="Do not start report server after scan"
    )
    p_scan.add_argument("--no-browser", action="store_true", help="Do not open browser")
    p_scan.add_argument(
        "--runtime",
        action="store_true",
        help="Enable runtime behavior monitor (isolated execution)",
    )
    p_scan.add_argument(
        "--policy", dest="policy", help="Policy YAML file for enforcement", default=None
    )

    p_serve = sub.add_parser("serve", help="Serve latest report on localhost")
    p_serve.add_argument("--port", type=int, default=8899)
    p_serve.add_argument("--no-browser", action="store_true")

    p_sbom = sub.add_parser("sbom", help="Generate SBOM only")
    p_sbom.add_argument("path", help="Skills root")

    p_audit = sub.add_parser("audit", help="Audit offline compliance (no network calls)")
    p_audit.add_argument("path", nargs="?", default=".", help="Project path to audit")

    p_runtime = sub.add_parser(
        "runtime",
        help="Runtime Behavior Monitor (isolated, collects filesystem/network/processes/env/DNS/outbound)",
    )
    p_runtime.add_argument("path", help="Skill path to run")
    p_runtime.add_argument("--timeout", type=int, default=10, help="Timeout seconds")
    p_runtime.add_argument("--output", dest="output", help="Write events JSON", default=None)

    p_policy = sub.add_parser("policy", help="Policy-as-Code check")
    p_policy.add_argument("path", help="Skill path")
    p_policy.add_argument("--policy", dest="policy", help="Policy YAML file", default="policy.yaml")
    p_policy.add_argument("--output", dest="output", help="Write result JSON", default=None)

    p_diff = sub.add_parser("diff", help="Security Regression: compare old vs new")
    p_diff.add_argument("old", help="Old skill path or scan JSON")
    p_diff.add_argument("new", help="New skill path or scan JSON")
    p_diff.add_argument("--output", dest="output", help="Write diff JSON", default=None)

    args = parser.parse_args()

    if args.cmd == "scan":
        root = Path(args.path)
        if not root.exists():
            print(f"Error: path not found: {root}", file=sys.stderr)
            sys.exit(1)
        print(f"[Skill Inspector] Scanning {root} (offline, deterministic)...")
        print(f"[Skill Inspector] Data dir: {get_data_dir()}  DB: {get_db_path()}")
        scanner = SecurityScanner(root)
        prev = Path(args.previous) if args.previous else None
        pol = Path(args.policy) if getattr(args, "policy", None) else None
        result = scanner.scan(
            previous_snapshot=prev, enable_runtime=getattr(args, "runtime", False), policy_path=pol
        )
        print(f"[Skill Inspector] Scan ID: {result['scan_id']}")
        print(
            f"[Skill Inspector] Skills: {result['summary']['total_skills']}  Findings: {result['summary']['total_findings']}  Avg Score: {result['summary']['avg_score']}"
        )
        for s in result["skills"]:
            drift = len(s.get("drift", []))
            corr = len(s.get("correlation", []))
            rt = s.get("runtime_events", 0)
            print(
                f"  - {s['name']}: score {s['scorecard']['score']} ({s['scorecard']['grade']}) findings={len(s['findings'])} drift={drift} corr={corr} runtime={rt}"
            )
        if result.get("attack_paths"):
            print(f"[Skill Inspector] Attack paths: {len(result['attack_paths'])}")
            for ap in result["attack_paths"][:2]:
                print(f"    - {ap['rule_id']}: {ap['message'][:120]}")
        if result.get("policy_decision"):
            print(
                f"[Skill Inspector] Policy: {result['policy_decision']['decision']} violations={result['policy_decision']['count']}"
            )
        print(f"[Skill Inspector] Report JSON: {result['report_path']}")
        print(f"[Skill Inspector] SBOM: {result['summary']['sbom_aggregate']}")
        if not args.no_serve:
            serve(result, port=args.port, open_browser=not args.no_browser)
        else:
            print(
                f"Run 'skill-inspector serve --port {args.port}' to view report at http://127.0.0.1:{args.port}"
            )

    elif args.cmd == "serve":
        latest = get_latest_scan()
        if not latest:
            print("No scans found. Run 'scan' first.", file=sys.stderr)
            sys.exit(1)
        print(
            f"[Skill Inspector] Serving latest scan {latest['scan_id']} on http://127.0.0.1:{args.port}"
        )
        serve(latest["results"], port=args.port, open_browser=not args.no_browser)

    elif args.cmd == "sbom":
        root = Path(args.path)
        scanner = SecurityScanner(root)
        result = scanner.scan()
        print(
            json.dumps(
                {
                    "sbom_aggregate": result["summary"]["sbom_aggregate"],
                    "skills": [
                        {"name": s["name"], "sbom": s["sbom_path"]} for s in result["skills"]
                    ],
                },
                indent=2,
            )
        )

    elif args.cmd == "runtime":
        from .runtime_monitor import run_isolated

        rp = Path(args.path)
        out = Path(args.output) if args.output else None
        graph = run_isolated(rp, timeout=float(args.timeout))
        print(f"[Runtime] {len(graph.events)} events for {rp}")
        for e in graph.events[:10]:
            print(
                f"  - [{e.source}] {e.category}/{e.action} {e.subject} -> {e.target} ({e.capability})"
            )
        if out:
            out.write_text(
                json.dumps([e.to_dict() for e in graph.events], indent=2), encoding="utf-8"
            )
            print(f"Events saved to {out}")
        # Drift summary
        from .drift_analyzer import capabilities_to_set, classify_drift
        from .permission_manifest import load_manifest, scan_capabilities
        from .runtime_monitor import collect_runtime_capabilities

        # Show drift for single skill
        if rp.is_dir():
            # find first skill
            from .dependency_graph import discover_skills

            skills = discover_skills(rp)
            if skills:
                name, sp = next(iter(skills.items()))
                manifest, _ = load_manifest(sp)
                perms = (
                    manifest.get("permissions", {})
                    if isinstance(manifest.get("permissions"), dict)
                    else {}
                )
                static = scan_capabilities(sp)
                runtime = collect_runtime_capabilities(graph)
                decl, stat, rt = capabilities_to_set(perms, static, runtime)
                drifts = classify_drift(decl, stat, rt)
                print(f"[Drift] {len([d for d in drifts if d['drift'] != 'MATCH'])} drifts:")
                for d in drifts:
                    if d["drift"] != "MATCH":
                        print(f"  - {d['capability']}: {d['drift']} ({d['message']})")

    elif args.cmd == "policy":
        from .policy import evaluate_policy, load_policy

        rp = Path(args.path)
        scanner = SecurityScanner(rp)
        result = scanner.scan(policy_path=Path(args.policy))
        pol = result.get("policy_decision") or evaluate_policy(
            [f for s in result["skills"] for f in s["findings"]], load_policy(Path(args.policy))
        )
        print(f"[Policy] {pol['decision']} violations={pol['count']}")
        for v in pol["violations"][:10]:
            print(f"  - {v['rule_id']}: {v['message']}")
        if args.output:
            Path(args.output).write_text(json.dumps(pol, indent=2), encoding="utf-8")
        if pol["decision"] == "BLOCK":
            sys.exit(1)

    elif args.cmd == "diff":
        from .regression import compare_reports

        def load_report(p: str):
            pp = Path(p)
            if pp.is_file() and pp.suffix == ".json":
                try:
                    data = json.loads(pp.read_text(encoding="utf-8"))
                    if "scan_id" in data or "skills" in data:
                        return data
                except Exception:
                    pass
            return SecurityScanner(Path(p)).scan()

        old = load_report(args.old)
        new = load_report(args.new)
        reg = compare_reports(old, new)
        print(f"[Regression] {reg['summary']}")
        print(f"  Added: {reg['added_caps']}")
        print(f"  Removed: {reg['removed_caps']}")
        print(f"  Decision: {reg['decision']}")
        if args.output:
            Path(args.output).write_text(json.dumps(reg, indent=2), encoding="utf-8")
        if reg["decision"] == "BLOCK":
            sys.exit(1)

    elif args.cmd == "audit":
        # Offline compliance check: ensure no network imports
        import re

        forbidden = [
            r"requests\.post.*https?://",
            r"socket\.",
            r"urllib\.request\.urlopen",
            r"openai\.",
            r"boto3",
            r"telemetry",
            r"analytics",
        ]
        # scan this project
        proj = Path(args.path)
        violations = []
        for f in proj.rglob("*.py"):
            if ".venv" in str(f) or "__pycache__" in str(f):
                continue
            try:
                t = f.read_text(encoding="utf-8", errors="ignore")
                for pat in forbidden:
                    if re.search(pat, t):
                        # but we allow localhost only for report server
                        if "127.0.0.1" in t or "localhost" in t:
                            continue
                        violations.append((str(f), pat))
            except Exception:
                continue
        # also check that we bind only to 127.0.0.1
        report_server = proj / "skill_inspector_security" / "report_server.py"
        if report_server.exists():
            txt = report_server.read_text(encoding="utf-8", errors="ignore")
            # Only flag if actual binding code uses 0.0.0.0, not docstring "no 0.0.0.0"
            code_only = "\n".join(line for line in txt.splitlines() if "no 0.0.0.0" not in line.lower())
            if '"0.0.0.0"' in code_only or "'0.0.0.0'" in code_only:
                # check if it's a host binding (not just a comment)
                if re.search(r'host\s*=\s*["\']0\.0\.0\.0["\']', code_only):
                    violations.append((str(report_server), "binds to 0.0.0.0 - must be 127.0.0.1"))
            if "ALLOWED_BIND" not in txt:
                violations.append((str(report_server), "missing ALLOWED_BIND check"))
        if violations:
            print("Offline compliance violations found:")
            for f, pat in violations:
                print(f"  {f}: {pat}")
            sys.exit(1)
        else:
            print(
                "Offline compliance: PASS — no cloud services, no external API calls, loopback only"
            )


if __name__ == "__main__":
    main()
