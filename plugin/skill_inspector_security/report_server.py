"""
report_server.py - FastAPI + Jinja2 localhost-only interactive HTML report.
Binds only to 127.0.0.1, no 0.0.0.0, no external exposure.
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import ALLOWED_BIND, get_data_dir
from .storage import get_connection, get_latest_scan


def create_app(
    scan_data: Dict[str, Any] | None = None,
    templates_dir: Path | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    app = FastAPI(
        title="NVIDIA Skill Inspector - Security Report",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # Resolve dirs
    base = Path(__file__).parent.parent
    t_dir = templates_dir or (base / "templates")
    s_dir = static_dir or (base / "static")

    env = Environment(
        loader=FileSystemLoader(str(t_dir)), autoescape=select_autoescape(["html", "xml"])
    )

    # Mount static only if exists
    if s_dir.exists():
        app.mount("/static", StaticFiles(directory=str(s_dir)), name="static")

    # In-memory scan data (or load latest from DB)
    def get_scan() -> Dict[str, Any]:
        if scan_data is not None:
            return scan_data
        latest = get_latest_scan()
        if not latest:
            return {
                "skills": [],
                "summary": {
                    "total_skills": 0,
                    "total_findings": 0,
                    "severity_counts": {},
                    "avg_score": 0,
                },
                "graph": {"nodes": [], "edges": []},
                "graph_metrics": {},
                "scan_id": "none",
                "timestamp": "",
            }
        return latest["results"]

    @app.get("/", response_class=HTMLResponse)
    def index():
        data = get_scan()
        try:
            tmpl = env.get_template("report.html")
        except Exception as e:
            return HTMLResponse(
                f"<h1>Template missing</h1><pre>{e}</pre><pre>{json.dumps(data, indent=2)[:4000]}</pre>",
                status_code=500,
            )
        # Prepare JSON for JS
        graph_json = json.dumps(data.get("graph", {"nodes": [], "edges": []}))
        skills_json = json.dumps(data.get("skills", []))
        summary = data.get("summary", {})
        metrics = data.get("graph_metrics", {})
        html = tmpl.render(
            scan_id=data.get("scan_id", "unknown"),
            timestamp=data.get("timestamp", ""),
            skills_root=data.get("skills_root", ""),
            skills=data.get("skills", []),
            summary=summary,
            graph_json=graph_json,
            skills_json=skills_json,
            metrics=metrics,
            graph=data.get("graph", {}),
        )
        return HTMLResponse(html)

    @app.get("/api/scan")
    def api_scan():
        return JSONResponse(get_scan())

    @app.get("/api/skills")
    def api_skills():
        data = get_scan()
        return JSONResponse(data.get("skills", []))

    @app.get("/api/graph")
    def api_graph():
        data = get_scan()
        return JSONResponse(data.get("graph", {"nodes": [], "edges": []}))

    @app.get("/api/findings")
    def api_findings():
        data = get_scan()
        all_f = []
        for s in data.get("skills", []):
            for f in s.get("findings", []):
                all_f.append({**f, "skill": s.get("name")})
        return JSONResponse(all_f)

    @app.get("/sbom/{skill_name}")
    def get_sbom(skill_name: str):
        data = get_scan()
        for s in data.get("skills", []):
            if s.get("name") == skill_name:
                p = Path(s.get("sbom_path", ""))
                if p.exists():
                    return FileResponse(
                        str(p), media_type="application/json", filename=f"{skill_name}.cdx.json"
                    )
        raise HTTPException(status_code=404, detail="SBOM not found")

    @app.get("/health")
    def health():
        return {"status": "ok", "bind": ALLOWED_BIND, "offline": True}

    return app


def serve(
    scan_data: Dict[str, Any],
    port: int = 8899,
    open_browser: bool = True,
    templates_dir: Path | None = None,
    static_dir: Path | None = None,
):
    """
    Serve report on localhost only. Blocks.
    """
    app = create_app(scan_data, templates_dir, static_dir)
    url = f"http://127.0.0.1:{port}"
    print(f"[Skill Inspector] Serving security report at {url} (loopback only, offline)")
    print(f"[Skill Inspector] Data dir: {get_data_dir()}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    # Enforce loopback only: host must be 127.0.0.1
    uvicorn.run(app, host=ALLOWED_BIND, port=port, log_level="info", access_log=False)


def create_app_for_testing(scan_data: Dict[str, Any]) -> FastAPI:
    base = Path(__file__).parent.parent
    return create_app(scan_data, templates_dir=base / "templates", static_dir=base / "static")
