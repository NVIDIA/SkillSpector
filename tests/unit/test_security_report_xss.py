from fastapi.testclient import TestClient

from skillspector.security_inspection.report_server import create_app_for_testing


def test_report_html_xss_escaping() -> None:
    hostile_scan = {
        "scan_id": "scan-test",
        "timestamp": "2026-10-07T00:00:00Z",
        "skills_root": "/tmp",
        "skills": [
            {
                "name": "</script><script>alert('skill')</script>",
                "version": "1.0.0",
                "author": "<script>alert('author')</script>",
                "origin": "https://evil.com/<script>",
                "hash_sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                "capabilities": {"network": False, "subprocess": False},
                "privacy": {"categories": [], "flows": {"risk": "low"}},
                "scorecard": {
                    "score": 85,
                    "grade": "B",
                    "severity_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0},
                    "recommendations": [],
                },
                "findings": [],
            }
        ],
        "summary": {
            "total_skills": 1,
            "total_findings": 0,
            "severity_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "avg_score": 85,
            "grade": "B",
        },
        "graph_metrics": {
            "total_nodes": 1,
            "total_edges": 1,
            "cyclomatic_complexity": 0,
            "is_dag": True,
            "cycles": [],
        },
        "graph": {
            "nodes": [
                {
                    "data": {
                        "id": "</script><script>alert(1)</script>",
                        "label": '<img src=x onerror="alert(2)">',
                    }
                }
            ],
            "edges": [
                {
                    "data": {
                        "source": "</script><script>alert('src')</script>",
                        "target": "<script>alert('tgt')</script>",
                        "type": "dependency",
                    }
                }
            ],
        },
    }

    app = create_app_for_testing(hostile_scan)
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    html = response.text

    # Assert that no raw unescaped script tag injection breaks out of the JSON script blocks
    assert "</script><script>" not in html
    assert "<script>alert" not in html
    assert '<img src=x onerror="alert(2)">' not in html
