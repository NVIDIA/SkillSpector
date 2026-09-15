"""
provenance.py - Skill Reputation & Provenance (offline, local SQLite + hashing).
Records origin, author, version history, integrity hashes.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .storage import get_provenance_history, save_provenance


def hash_skill_directory(skill_path: Path) -> str:
    """Deterministic SHA256 over all files sorted."""
    h = hashlib.sha256()
    files = sorted(
        [p for p in skill_path.rglob("*") if p.is_file()],
        key=lambda p: str(p.relative_to(skill_path)),
    )
    for fp in files:
        try:
            # skip cache dirs
            if ".git" in fp.parts or "__pycache__" in fp.parts or ".venv" in fp.parts:
                continue
            rel = str(fp.relative_to(skill_path)).encode()
            h.update(rel + b"\x00")
            h.update(fp.read_bytes())
            h.update(b"\x00")
        except Exception:
            continue
    return h.hexdigest()


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_metadata(skill_path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "name": skill_path.name,
        "version": "0.0.0",
        "author": "unknown",
        "origin": "local",
        "description": "",
    }
    # Try manifests
    for cand in [skill_path / "skill.json", skill_path / "manifest.json"]:
        if cand.exists():
            try:
                data = json.loads(cand.read_text(encoding="utf-8", errors="ignore"))
                meta["name"] = data.get("name", meta["name"])
                meta["version"] = str(data.get("version", meta["version"]))
                meta["author"] = data.get("author", meta["author"])
                meta["origin"] = data.get("origin", data.get("repository", meta["origin"]))
                meta["description"] = data.get("description", meta["description"])
                break
            except Exception:
                continue
    # SKILL.md
    md = skill_path / "SKILL.md"
    if md.exists():
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")
            # try to extract author/version from header
            m = re.search(r"author:\s*(.+)", text, re.IGNORECASE)
            if m:
                meta["author"] = m.group(1).strip().strip("\"'")
            m = re.search(r"version:\s*(.+)", text, re.IGNORECASE)
            if m:
                meta["version"] = m.group(1).strip().strip("\"'")
        except Exception:
            pass
    # Git origin if available (local plumbing, no network)
    git_config = skill_path / ".git" / "config"
    if not git_config.exists():
        # try parent
        parent_git = skill_path.parent / ".git" / "config"
        if parent_git.exists():
            git_config = parent_git
    if git_config.exists():
        try:
            cfg = git_config.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"url\s*=\s*(.+)", cfg)
            if m:
                meta["origin"] = m.group(1).strip()
        except Exception:
            pass
    return meta


def record_provenance(
    skill_path: Path, skills_root: Path | None = None, db_path: Path | None = None
) -> dict[str, Any]:
    meta = extract_metadata(skill_path)
    sha = hash_skill_directory(skill_path)
    manifest_data = {}
    for cand in [skill_path / "skill.json", skill_path / "manifest.json"]:
        if cand.exists():
            try:
                manifest_data = json.loads(cand.read_text(encoding="utf-8", errors="ignore"))
                break
            except Exception:
                manifest_data = {}
    save_provenance(
        meta["name"],
        meta["version"],
        meta["author"],
        meta["origin"],
        sha,
        manifest_data,
        db_path=db_path,
    )
    # check history for reputation signals
    history = get_provenance_history(meta["name"], db_path=db_path)
    findings: list[dict[str, Any]] = []
    if len(history) > 1:
        hashes = set(r["hash_sha256"] for r in history)
        if len(hashes) > 1 and history[0]["hash_sha256"] != sha:
            # Actually current not yet in history? We just saved, so check prior
            pass
        # Check for author change - potential hijack
        authors = set(r["author"] for r in history)
        if len(authors) > 1:
            findings.append(
                {
                    "rule_id": "PROV-001",
                    "category": "provenance",
                    "severity": "medium",
                    "message": f"Author changed across versions: {authors} - verify legitimate ownership transfer",
                    "file": str(skill_path),
                    "line": None,
                    "evidence": f"authors={authors}",
                }
            )
        # Version regression?
        versions = [r["version"] for r in history]
        if len(versions) >= 2:
            # simple check: if current version duplicates old hash but version bumped incorrectly?
            pass
    # Integrity: if no author/origin
    if meta["author"] == "unknown":
        findings.append(
            {
                "rule_id": "PROV-002",
                "category": "provenance",
                "severity": "low",
                "message": "Missing author in manifest - provenance incomplete",
                "file": str(skill_path / "skill.json"),
                "line": None,
                "evidence": "author=unknown",
            }
        )
    if meta["origin"] == "local":
        findings.append(
            {
                "rule_id": "PROV-003",
                "category": "provenance",
                "severity": "info",
                "message": "Origin is local filesystem (no git remote) - manual provenance",
                "file": str(skill_path),
                "line": None,
                "evidence": "origin=local",
            }
        )
    return {
        "name": meta["name"],
        "version": meta["version"],
        "author": meta["author"],
        "origin": meta["origin"],
        "hash_sha256": sha,
        "history_count": len(history),
        "findings": findings,
    }
