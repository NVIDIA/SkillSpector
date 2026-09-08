"""
sbom.py - Generate CycloneDX-format SBOM for installed skills (offline, deterministic).
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _hash_file(p: Path) -> str:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        return ""

def _parse_requirements(skill_path: Path) -> list[dict[str, str]]:
    components: list[dict[str, str]] = []
    req_files = [skill_path / "requirements.txt", skill_path / "requirements.pip", skill_path / "pyproject.toml", skill_path / "setup.py", skill_path / "Pipfile", skill_path / "environment.yml"]
    for rf in req_files:
        if not rf.exists():
            continue
        try:
            text = rf.read_text(encoding="utf-8", errors="ignore")
            if rf.name == "requirements.txt":
                for line in text.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # parse name==version
                    m = re.match(r"([A-Za-z0-9_\-\.]+)([=<>!~]+.*)?", line)
                    if m:
                        name = m.group(1)
                        ver = (m.group(2) or "").strip(" =")
                        components.append({"name": name, "version": ver or "unknown", "source": str(rf.name)})
            elif rf.name == "pyproject.toml":
                # naive parse
                for m in re.finditer(r'"([A-Za-z0-9_\-\.]+)"\s*=\s*"([^"]+)"', text):
                    # dependencies = ["requests>=2.0"] etc - fallback
                    pass
                # look for dependencies = [
                deps_match = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, re.DOTALL)
                if deps_match:
                    for dep in re.findall(r'"([^"]+)"', deps_match.group(1)):
                        m2 = re.match(r"([A-Za-z0-9_\-\.]+)(.*)", dep)
                        if m2:
                            components.append({"name": m2.group(1), "version": m2.group(2).strip(" =<>~!") or "unknown", "source": "pyproject.toml"})
            elif rf.name == "setup.py":
                # search install_requires
                im = re.search(r"install_requires\s*=\s*\[(.*?)\]", text, re.DOTALL)
                if im:
                    for dep in re.findall(r'"([^"]+)"|\'([^\']+)\'', im.group(1)):
                        dep_str = dep[0] or dep[1]
                        m2 = re.match(r"([A-Za-z0-9_\-\.]+)(.*)", dep_str)
                        if m2:
                            components.append({"name": m2.group(1), "version": m2.group(2).strip(" =<>~!") or "unknown", "source": "setup.py"})
        except Exception:
            continue
    # Also scan imports to infer dependencies (stdlib vs external)
    # For now dedupe
    seen = set()
    uniq = []
    for c in components:
        key = (c["name"].lower(), c["version"])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq

def generate_sbom(skill_path: Path, skills_root: Path) -> dict[str, Any]:
    """Generate CycloneDX 1.5 SBOM for a single skill."""
    skill_name = skill_path.name
    # metadata
    meta_files = []
    for p in skill_path.rglob("*"):
        if p.is_file() and p.suffix in (".py", ".json", ".yaml", ".yml", ".txt", ".toml", ".cfg", ".md") and ".git" not in p.parts:
            meta_files.append(p)

    components: list[dict[str, Any]] = []
    # main skill component
    skill_hash = _hash_file(skill_path / "skill.json") or _hash_file(skill_path / "manifest.json") or hashlib.sha256(skill_name.encode()).hexdigest()[:16]
    # version from manifest
    version = "unknown"
    for cand in [skill_path / "skill.json", skill_path / "manifest.json"]:
        if cand.exists():
            try:
                data = json.loads(cand.read_text(encoding="utf-8", errors="ignore"))
                version = str(data.get("version", version))
                break
            except Exception:
                pass

    components.append({
        "type": "application",
        "name": skill_name,
        "version": version,
        "purl": f"pkg:skill/{skill_name}@{version}",
        "hashes": [{"alg": "SHA-256", "content": hashlib.sha256(skill_name.encode()).hexdigest()}],
        "description": f"Agent skill: {skill_name}",
    })

    # file components
    for fp in sorted(meta_files, key=lambda x: str(x))[:200]:  # cap for size
        rel = str(fp.relative_to(skill_path))
        h = _hash_file(fp)
        components.append({
            "type": "file",
            "name": rel,
            "version": "1.0",
            "hashes": [{"alg": "SHA-256", "content": h}],
            "purl": f"pkg:file/{skill_name}/{rel}",
        })

    # dependency components from requirements
    deps = _parse_requirements(skill_path)
    for dep in deps:
        components.append({
            "type": "library",
            "name": dep["name"],
            "version": dep["version"],
            "purl": f"pkg:pypi/{dep['name']}@{dep['version']}",
            "scope": "required",
            "evidence": {"occurrences": [{"location": dep["source"]}]},
        })

    bom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{hashlib.sha256((skill_name + version).encode()).hexdigest()[:32]}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "component": {
                "type": "application",
                "name": skill_name,
                "version": version,
            },
            "tools": [{"vendor": "NVIDIA", "name": "Skill Inspector Security Plugin", "version": "1.0.0"}],
        },
        "components": components,
    }
    return bom

def generate_aggregate_sbom(skills_root: Path, discovered: dict[str, Path]) -> dict[str, Any]:
    """Aggregate SBOM for all skills."""
    all_components = []
    for name, path in discovered.items():
        bom = generate_sbom(path, skills_root)
        # prefix component names to avoid collision
        for comp in bom["components"]:
            if comp["type"] == "application":
                all_components.append(comp)
            else:
                # keep file/library but namespace
                comp_copy = dict(comp)
                comp_copy["skill"] = name
                all_components.append(comp_copy)
    agg = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{hashlib.sha256(str(sorted(discovered.keys())).encode()).hexdigest()[:32]}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "component": {"type": "application", "name": "agent-skills-aggregate", "version": "1.0"},
            "tools": [{"vendor": "NVIDIA", "name": "Skill Inspector Security Plugin", "version": "1.0.0"}],
        },
        "components": all_components,
    }
    return agg

def save_sbom(bom: dict[str, Any], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bom, indent=2), encoding="utf-8")
