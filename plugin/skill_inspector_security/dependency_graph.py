"""
dependency_graph.py - Build skill dependency graph using networkx (offline, deterministic).
Parses:
 - manifest dependencies (skill.json, manifest.yaml/json, SKILL.md frontmatter)
 - Python imports referencing other skills
 - file includes
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx

SKILL_MANIFEST_NAMES = [
    "skill.json",
    "manifest.json",
    "manifest.yaml",
    "manifest.yml",
    "SKILL.md",
    "skill.yaml",
    "skill.yml",
]
IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)")


def discover_skills(root: Path) -> Dict[str, Path]:
    """Each immediate subdirectory with at least one manifest or .py is a skill."""
    skills: Dict[str, Path] = {}
    if not root.exists():
        return skills
    for child in root.iterdir():
        if child.is_dir():
            # check if looks like skill
            has_manifest = any((child / m).exists() for m in SKILL_MANIFEST_NAMES)
            has_py = any(child.rglob("*.py"))
            has_md = (child / "SKILL.md").exists() or (child / "README.md").exists()
            if has_manifest or has_py or has_md:
                skills[child.name] = child
    # if root itself is a skill (single skill repo)
    if not skills and root.is_dir():
        if any((root / m).exists() for m in SKILL_MANIFEST_NAMES) or any(root.glob("*.py")):
            skills[root.name] = root
    return skills


def parse_manifest_dependencies(skill_path: Path) -> List[str]:
    deps: List[str] = []
    candidates = [
        skill_path / "skill.json",
        skill_path / "manifest.json",
        skill_path / "skill.yaml",
        skill_path / "skill.yml",
        skill_path / "manifest.yaml",
        skill_path / "manifest.yml",
    ]
    for p in candidates:
        if p.exists():
            try:
                if p.suffix in (".yaml", ".yml"):
                    try:
                        import yaml  # optional, but we try regex fallback if not present

                        data = yaml.safe_load(p.read_text(encoding="utf-8", errors="ignore"))
                    except ImportError:
                        text = p.read_text(encoding="utf-8", errors="ignore")
                        # simple regex extraction
                        deps.extend(re.findall(r"depends_on:\s*\n(?:\s*-\s*(\S+)\n?)+", text))
                        deps.extend(re.findall(r'"dependencies"\s*:\s*\[([^\]]+)\]', text))
                        continue
                else:
                    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(data, dict):
                    for key in ["dependencies", "depends_on", "requires", "skills"]:
                        if key in data:
                            val = data[key]
                            if isinstance(val, list):
                                deps.extend([str(v).strip() for v in val])
                            elif isinstance(val, str):
                                deps.append(val)
                    # permissions dependencies not but capture
                # handle SKILL.md frontmatter? already
            except Exception:
                continue
    # SKILL.md parse
    skill_md = skill_path / "SKILL.md"
    if skill_md.exists():
        try:
            text = skill_md.read_text(encoding="utf-8", errors="ignore")
            # frontmatter between ---
            fm_match = re.search(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
            if fm_match:
                fm = fm_match.group(1)
                # naive yaml parse for dependencies
                for line in fm.splitlines():
                    if "dependencies" in line or "depends_on" in line:
                        # subsequent list items
                        pass
                deps.extend(re.findall(r"-\s*([a-zA-Z0-9_\-]+)", fm))
        except Exception:
            pass
    return [d.strip().strip("\"'") for d in deps if d.strip()]


def parse_python_imports(skill_path: Path, all_skill_names: Set[str]) -> List[str]:
    deps: Set[str] = set()
    for py_file in skill_path.rglob("*.py"):
        try:
            text = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(text, filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top in all_skill_names and top != skill_path.name:
                            deps.add(top)
                        # also check skill import pattern like "skills.<name>"
                        if alias.name.startswith("skills.") or alias.name.startswith("skill_"):
                            # heuristic
                            pass
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        top = node.module.split(".")[0]
                        if top in all_skill_names and top != skill_path.name:
                            deps.add(top)
        except Exception:
            # fallback regex
            try:
                text = py_file.read_text(encoding="utf-8", errors="ignore")
                for line in text.splitlines():
                    m = IMPORT_RE.match(line)
                    if m:
                        mod = m.group(1).split(".")[0]
                        if mod in all_skill_names and mod != skill_path.name:
                            deps.add(mod)
            except Exception:
                continue
    # Also check for string references like "skill://name" or "use_skill('name')"
    try:
        for f in skill_path.rglob("*"):
            if f.is_file() and f.suffix in (".py", ".md", ".json", ".yaml", ".yml"):
                t = f.read_text(encoding="utf-8", errors="ignore")
                for pat in [
                    r"skill://([a-zA-Z0-9_\-]+)",
                    r"use_skill\(\s*['\"]([^'\"]+)['\"]",
                    r"load_skill\(\s*['\"]([^'\"]+)['\"]",
                ]:
                    for m in re.findall(pat, t):
                        if m in all_skill_names and m != skill_path.name:
                            deps.add(m)
    except Exception:
        pass
    return sorted(deps)


def build_dependency_graph(root: Path) -> Tuple[nx.DiGraph, Dict[str, Path]]:
    skills = discover_skills(root)
    G = nx.DiGraph()
    for name, path in skills.items():
        G.add_node(name, path=str(path), label=name)
    all_names = set(skills.keys())
    for name, path in skills.items():
        manifest_deps = parse_manifest_dependencies(path)
        import_deps = parse_python_imports(path, all_names)
        combined = set(manifest_deps + import_deps)
        for dep in combined:
            if dep in all_names:
                G.add_edge(name, dep, type="depends_on")
            else:
                # external dep (still add as node for visibility)
                if dep and dep not in G:
                    G.add_node(dep, path="", label=dep, external=True)
                if dep:
                    G.add_edge(name, dep, type="external")
    return G, skills


def graph_to_cytoscape(G: nx.DiGraph) -> List[Dict[str, Any]]:
    """Convert to cytoscape/visjs friendly JSON."""
    nodes = []
    for n, data in G.nodes(data=True):
        nodes.append(
            {
                "data": {
                    "id": n,
                    "label": data.get("label", n),
                    "external": data.get("external", False),
                }
            }
        )
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({"data": {"source": u, "target": v, "type": data.get("type", "depends_on")}})
    return {"nodes": nodes, "edges": edges}


def detect_cycles(G: nx.DiGraph) -> List[List[str]]:
    try:
        cycles = list(nx.simple_cycles(G))
        return cycles
    except Exception:
        return []


def compute_metrics(G: nx.DiGraph) -> Dict[str, Any]:
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "is_dag": nx.is_directed_acyclic_graph(G),
        "cycles": detect_cycles(G),
        "isolated": list(nx.isolates(G)),
        "in_degree": dict(G.in_degree()),
        "out_degree": dict(G.out_degree()),
        "most_depended": sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:5]
        if G.number_of_nodes()
        else [],
    }
