"""
storage.py - SQLite local storage (offline, no cloud).
Stores: scan results, provenance records, scorecard history.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import get_db_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT UNIQUE NOT NULL,
    timestamp TEXT NOT NULL,
    skills_root TEXT NOT NULL,
    total_skills INTEGER NOT NULL,
    total_findings INTEGER NOT NULL,
    results_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS provenance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_name TEXT NOT NULL,
    version TEXT NOT NULL,
    author TEXT,
    origin TEXT,
    hash_sha256 TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    manifest_json TEXT,
    UNIQUE(skill_name, version, hash_sha256)
);
CREATE TABLE IF NOT EXISTS scorecard_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_name TEXT NOT NULL,
    version TEXT NOT NULL,
    score INTEGER NOT NULL,
    grade TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    details_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT NOT NULL,
    skill_name TEXT NOT NULL,
    category TEXT NOT NULL,
    severity TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    message TEXT NOT NULL,
    file_path TEXT,
    line INTEGER,
    evidence TEXT,
    FOREIGN KEY(scan_id) REFERENCES scans(scan_id)
);
"""

def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_db(db_path: Path | None = None) -> Path:
    path = db_path or get_db_path()
    conn = get_connection(path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
    return path

def save_scan(scan_id: str, skills_root: str, results: dict[str, Any], db_path: Path | None = None):
    conn = get_connection(db_path)
    try:
        ts = datetime.now(UTC).isoformat()
        total_skills = len(results.get("skills", []))
        total_findings = sum(len(s.get("findings", [])) for s in results.get("skills", []))
        conn.execute(
            "INSERT OR REPLACE INTO scans (scan_id, timestamp, skills_root, total_skills, total_findings, results_json) VALUES (?,?,?,?,?,?)",
            (scan_id, ts, skills_root, total_skills, total_findings, json.dumps(results)),
        )
        # save findings
        conn.execute("DELETE FROM findings WHERE scan_id=?", (scan_id,))
        for skill in results.get("skills", []):
            for f in skill.get("findings", []):
                conn.execute(
                    "INSERT INTO findings (scan_id, skill_name, category, severity, rule_id, message, file_path, line, evidence) VALUES (?,?,?,?,?,?,?,?,?)",
                    (scan_id, skill.get("name"), f.get("category"), f.get("severity"), f.get("rule_id"), f.get("message"), f.get("file"), f.get("line"), f.get("evidence")),
                )
        # save scorecard history
        for skill in results.get("skills", []):
            sc = skill.get("scorecard", {})
            if sc:
                conn.execute(
                    "INSERT INTO scorecard_history (skill_name, version, score, grade, timestamp, details_json) VALUES (?,?,?,?,?,?)",
                    (skill.get("name"), skill.get("version", "unknown"), sc.get("score", 0), sc.get("grade", "F"), ts, json.dumps(sc)),
                )
        conn.commit()
    finally:
        conn.close()

def save_provenance(skill_name: str, version: str, author: str, origin: str, hash_sha256: str, manifest: dict[str, Any], db_path: Path | None = None):
    conn = get_connection(db_path)
    try:
        ts = datetime.now(UTC).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO provenance (skill_name, version, author, origin, hash_sha256, timestamp, manifest_json) VALUES (?,?,?,?,?,?,?)",
            (skill_name, version, author, origin, hash_sha256, ts, json.dumps(manifest)),
        )
        conn.commit()
    finally:
        conn.close()

def get_provenance_history(skill_name: str, db_path: Path | None = None) -> list[dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM provenance WHERE skill_name=? ORDER BY timestamp DESC", (skill_name,))
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

def get_scorecard_history(skill_name: str, db_path: Path | None = None) -> list[dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM scorecard_history WHERE skill_name=? ORDER BY timestamp DESC", (skill_name,))
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

def get_latest_scan(db_path: Path | None = None) -> dict[str, Any] | None:
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM scans ORDER BY timestamp DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        d["results"] = json.loads(d["results_json"])
        return d
    finally:
        conn.close()
