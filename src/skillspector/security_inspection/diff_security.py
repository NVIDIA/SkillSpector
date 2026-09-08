"""
diff_security.py - Compare skill versions and flag risky changes.
Uses Git plumbing if available, else difflib (offline, deterministic).
"""
from __future__ import annotations

import difflib
import re
import subprocess
from pathlib import Path
from typing import Any

RISKY_CHANGE_PATTERNS = {
    "new_network": [r"requests\.", r"urllib\.", r"socket\.", r"aiohttp", r"httpx", r"http\.client"],
    "new_subprocess": [r"subprocess\.", r"os\.system", r"os\.popen", r"eval\s*\(", r"exec\s*\("],
    "new_permissions": [r"permissions", r"capabilities", r"file_access", r"network.*unrestricted"],
    "new_secrets": [r"api_key", r"secret", r"password", r"token"],
    "data_flow_change": [r"read_text", r"write_text", r"open\s*\(", r"json\.load", r"pickle\.load"],
}

def _git_diff(skill_path: Path, ref_a: str, ref_b: str) -> str | None:
    """Try git diff using plumbing, returns None if not git repo."""
    try:
        # find git root
        cur = skill_path.resolve()
        git_root = None
        for parent in [cur] + list(cur.parents):
            if (parent / ".git").exists():
                git_root = parent
                break
        if not git_root:
            return None
        # Use git plumbing: git diff --no-color ref_a..ref_b -- <path>
        # Only if refs are valid commits/tags
        rel = skill_path.relative_to(git_root) if skill_path != git_root else Path(".")
        cmd = ["git", "-C", str(git_root), "diff", "--no-color", f"{ref_a}..{ref_b}", "--", str(rel)]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout
        # fallback: git diff HEAD
        cmd2 = ["git", "-C", str(git_root), "diff", "--no-color", "--", str(rel)]
        out2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=10)
        if out2.returncode == 0 and out2.stdout.strip():
            return out2.stdout
        return None
    except Exception:
        return None

def _difflib_dir_compare(dir_a: Path, dir_b: Path) -> str:
    """Fallback difflib directory comparison."""
    # collect files
    files_a = {str(p.relative_to(dir_a)): p for p in dir_a.rglob("*") if p.is_file() and ".git" not in p.parts}
    files_b = {str(p.relative_to(dir_b)): p for p in dir_b.rglob("*") if p.is_file() and ".git" not in p.parts}
    all_keys = sorted(set(files_a.keys()) | set(files_b.keys()))
    diff_out = []
    for key in all_keys:
        if key not in files_a:
            diff_out.append(f"+++ Added file: {key}")
            try:
                diff_out.append(files_b[key].read_text(encoding="utf-8", errors="ignore")[:2000])
            except Exception:
                diff_out.append("<binary or unreadable>")
        elif key not in files_b:
            diff_out.append(f"--- Removed file: {key}")
        else:
            try:
                a_text = files_a[key].read_text(encoding="utf-8", errors="ignore").splitlines()
                b_text = files_b[key].read_text(encoding="utf-8", errors="ignore").splitlines()
                if a_text != b_text:
                    diff = difflib.unified_diff(a_text, b_text, fromfile=f"a/{key}", tofile=f"b/{key}", lineterm="")
                    diff_out.extend(list(diff))
            except Exception:
                continue
    return "\n".join(diff_out)

def analyze_diff_text(diff_text: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    added_lines = [l for l in diff_text.splitlines() if l.startswith("+") and not l.startswith("+++")]
    added_text = "\n".join(added_lines)

    for category, patterns in RISKY_CHANGE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, added_text, re.IGNORECASE):
                sev = "high" if category in ("new_network", "new_subprocess", "new_secrets") else "medium"
                findings.append({
                    "rule_id": f"DIFF-{category.upper()}",
                    "category": "diff",
                    "severity": sev,
                    "message": f"Risky change: new {category.replace('_', ' ')} detected in diff (pattern: {pat})",
                    "file": "diff",
                    "line": None,
                    "evidence": f"Added line matching {pat}: {next((l for l in added_lines if re.search(pat, l, re.IGNORECASE)), '')[:200]}",
                    "fix": "Review change for least privilege and security impact"
                })
                break  # one per category

    # Permission escalation: check manifest diff
    if re.search(r'"network"\s*:\s*"unrestricted"', added_text) or re.search(r"network:\s*unrestricted", added_text):
        findings.append({
            "rule_id": "DIFF-PERM-ESCALATE",
            "category": "diff",
            "severity": "critical",
            "message": "Permission escalation: network changed to unrestricted",
            "file": "manifest",
            "line": None,
            "evidence": "network=unrestricted in diff",
            "fix": "Require manual review for unrestricted network"
        })
    # New file with executable permission?
    if "Added file:" in diff_text and re.search(r"\.sh|\.py", diff_text):
        # count added files
        added_files = [l for l in diff_text.splitlines() if l.startswith("+++ Added file:")]
        if len(added_files) > 3:
            findings.append({
                "rule_id": "DIFF-NEW-FILES",
                "category": "diff",
                "severity": "medium",
                "message": f"Many new files added ({len(added_files)}) - review for supply chain risk",
                "file": "diff",
                "line": None,
                "evidence": ", ".join(added_files[:5])[:300],
            })

    return findings

def compare_skill_versions(skill_path_a: Path, skill_path_b: Path, ref_a: str = "a", ref_b: str = "b") -> dict[str, Any]:
    """
    Compare two skill directories (or git refs). If git available, prefer git plumbing.
    Returns {diff_text, findings, stats}
    """
    diff_text: str | None = None
    # if paths are same but refs differ, try git
    if skill_path_a == skill_path_b:
        diff_text = _git_diff(skill_path_a, ref_a, ref_b)
    if diff_text is None:
        # if different dirs, do difflib compare
        if skill_path_a != skill_path_b and skill_path_a.exists() and skill_path_b.exists():
            diff_text = _difflib_dir_compare(skill_path_a, skill_path_b)
        elif skill_path_a.exists() and skill_path_b == skill_path_a:
            # no diff
            diff_text = ""
        else:
            diff_text = _git_diff(skill_path_a, ref_a, ref_b) or ""

    if diff_text is None:
        diff_text = ""

    findings = analyze_diff_text(diff_text)
    stats = {
        "added_lines": len([l for l in diff_text.splitlines() if l.startswith("+") and not l.startswith("+++")]),
        "removed_lines": len([l for l in diff_text.splitlines() if l.startswith("-") and not l.startswith("---")]),
        "diff_size": len(diff_text),
    }
    return {"diff_text": diff_text, "findings": findings, "stats": stats}

def diff_against_previous_version(skill_path: Path, previous_snapshot: Path | None = None) -> dict[str, Any]:
    """Compare current skill against previous snapshot if provided; else try git diff."""
    if previous_snapshot and previous_snapshot.exists():
        return compare_skill_versions(previous_snapshot, skill_path)
    # Try git diff HEAD
    diff_text = _git_diff(skill_path, "HEAD", "HEAD")  # will fallback to working tree diff
    if not diff_text:
        # Try git diff vs last commit file list
        try:
            cur = skill_path.resolve()
            git_root = None
            for parent in [cur] + list(cur.parents):
                if (parent / ".git").exists():
                    git_root = parent
                    break
            if git_root:
                out = subprocess.run(["git", "-C", str(git_root), "diff", "--no-color", "--", str(skill_path.relative_to(git_root))], capture_output=True, text=True, timeout=10)
                diff_text = out.stdout
        except Exception:
            diff_text = ""
    if not diff_text:
        diff_text = ""
    findings = analyze_diff_text(diff_text) if diff_text else []
    return {"diff_text": diff_text, "findings": findings, "stats": {"added_lines": len([l for l in diff_text.splitlines() if l.startswith("+")]), "removed_lines": len([l for l in diff_text.splitlines() if l.startswith("-")]), "diff_size": len(diff_text)}}
