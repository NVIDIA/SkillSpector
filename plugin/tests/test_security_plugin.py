import tempfile, json, pathlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from skill_inspector_security.scanner import SecurityScanner
from skill_inspector_security.dependency_graph import build_dependency_graph
from skill_inspector_security.secrets_analyzer import scan_skill_secrets, shannon_entropy
from skill_inspector_security.permission_manifest import load_manifest, scan_capabilities, compare_manifest_vs_actual
from skill_inspector_security.privacy_classifier import classify_skill_data
from skill_inspector_security.sbom import generate_sbom
from skill_inspector_security.scorecard import compute_scorecard
from skill_inspector_security.config import get_data_dir

def test_entropy():
    assert shannon_entropy("aaaa") < 1.0
    assert shannon_entropy("sk-proj-1234567890abcdef") > 3.5

def test_dependency_graph():
    root = Path(__file__).parent / "sample_skills"
    G, discovered = build_dependency_graph(root)
    assert "skill-a" in discovered
    assert "skill-b" in discovered
    assert G.has_edge("skill-a", "skill-b")

def test_secrets():
    skill_a = Path(__file__).parent / "sample_skills" / "skill-a"
    findings = scan_skill_secrets(skill_a)
    rule_ids = [f["rule_id"] for f in findings]
    assert any(r.startswith("SEC-") for r in rule_ids)
    assert "SEC-201" in rule_ids or "SEC-005" in rule_ids or "SEC-100" in rule_ids

def test_permission_mismatch():
    skill_a = Path(__file__).parent / "sample_skills" / "skill-a"
    manifest, _ = load_manifest(skill_a)
    caps = scan_capabilities(skill_a)
    assert caps["network"] is True
    assert caps["subprocess"] is True
    mism = compare_manifest_vs_actual(manifest, caps, skill_a)
    assert any(m["rule_id"] == "PERM-101" for m in mism)
    assert any(m["rule_id"] == "PERM-102" for m in mism)

def test_privacy():
    skill_a = Path(__file__).parent / "sample_skills" / "skill-a"
    priv = classify_skill_data(skill_a)
    assert "Financial" in priv["categories"] or "PII" in priv["categories"] or "Credentials" in priv["categories"]
    assert priv["flows"]["transmits"] is True

def test_sbom():
    skill_a = Path(__file__).parent / "sample_skills" / "skill-a"
    bom = generate_sbom(skill_a, skill_a.parent)
    assert bom["bomFormat"] == "CycloneDX"
    assert bom["specVersion"] == "1.5"
    assert len(bom["components"]) > 0

def test_scorecard():
    findings = [
        {"rule_id": "SEC-001", "severity": "critical", "category": "secrets", "message": "x", "evidence": ""},
        {"rule_id": "PERM-101", "severity": "high", "category": "permission", "message": "y", "evidence": ""},
    ]
    sc = compute_scorecard(findings)
    assert sc["score"] < 100
    assert sc["grade"] in ("A","B","C","D","F")
    assert sc["explanation"]

def test_full_scan():
    with tempfile.TemporaryDirectory() as tmp:
        # Use temp data dir
        import os
        os.environ["SKILL_INSPECTOR_DATA_DIR"] = tmp
        from importlib import reload
        import skill_inspector_security.config as cfg
        import skill_inspector_security.storage as stor
        # Need to re-init but scanner will handle
        root = Path(__file__).parent / "sample_skills"
        scanner = SecurityScanner(root, db_path=Path(tmp)/"test.db")
        result = scanner.scan()
        assert result["summary"]["total_skills"] == 2
        assert result["summary"]["total_findings"] > 5
        assert len(result["skills"]) == 2
        # check scorecard exists
        for s in result["skills"]:
            assert "scorecard" in s
            assert "score" in s["scorecard"]
        # graph
        assert "graph" in result
        assert len(result["graph"]["nodes"]) >= 2
        # SBOM files exist
        for s in result["skills"]:
            assert Path(s["sbom_path"]).exists()
        # Report server app
        from skill_inspector_security.report_server import create_app_for_testing
        app = create_app_for_testing(result)
        from fastapi.testclient import TestClient
        client = TestClient(app)
        r = client.get("/")
        assert r.status_code == 200
        assert "Skill Inspector" in r.text
        r2 = client.get("/api/scan")
        assert r2.status_code == 200
        r3 = client.get("/health")
        assert r3.json()["bind"] == "127.0.0.1"
        # cleanup env
        del os.environ["SKILL_INSPECTOR_DATA_DIR"]

def test_offline_compliance():
    proj = Path(__file__).parent.parent
    txt = (proj / "skill_inspector_security" / "report_server.py").read_text(encoding="utf-8")
    assert "127.0.0.1" in txt
    # Ensure localhost-only binding is enforced via ALLOWED_BIND, not 0.0.0.0 host
    assert 'ALLOWED_BIND = "127.0.0.1"' in (proj / "skill_inspector_security" / "config.py").read_text(encoding="utf-8")
    assert 'host=ALLOWED_BIND' in txt or 'host="127.0.0.1"' in txt
    # docstring mentions 0.0.0.0 as "no 0.0.0.0" - check no actual binding to 0.0.0.0 outside docstring
    code_lines = [l for l in txt.splitlines() if "no 0.0.0.0" not in l.lower()]
    assert not any('"0.0.0.0"' in l and "host" in l for l in code_lines)
    # ensure no openai/boto3 import
    for p in proj.rglob("*.py"):
        if "tests" in str(p) or "__pycache__" in str(p):
            continue
        t = p.read_text(encoding="utf-8", errors="ignore")
        assert "import openai" not in t
        assert "import boto3" not in t
