"""
diff_security.py - Compare skill versions and flag risky changes.
Uses in-process difflib only (offline, deterministic, safe against hostile .git/config).
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

RISKY_CHANGE_PATTERNS = {
    "new_network": [r"requests\.", r"urllib\.", r"socket\.", r"aiohttp", r"httpx", r"http\.client"],
    "new_subprocess": [r"subprocess\.", r"os\.system", r"os\.popen", r"eval\s*\(", r"exec\s*\("],
    "new_permissions": [r"permissions", r"capabilities", r"file_access", r"network.*unrestricted"],
    "new_secrets": [r"api_key", r"secret", r"password", r"token"],
    "data_flow_change": [r"read_text", r"write_text", r"open\s*\(", r"json\.load", r"pickle\.load"],
}


def _difflib_dir_compare(dir_a: Path, dir_b: Path) -> str:
    """In-process difflib directory comparison."""
    # collect files
    files_a = {
        str(p.relative_to(dir_a)): p
        for p in dir_a.rglob("*")
        if p.is_file() and ".git" not in p.parts
    }
    files_b = {
        str(p.relative_to(dir_b)): p
        for p in dir_b.rglob("*")
        if p.is_file() and ".git" not in p.parts
    }
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
                    diff = difflib.unified_diff(
                        a_text, b_text, fromfile=f"a/{key}", tofile=f"b/{key}", lineterm=""
                    )
                    diff_out.extend(list(diff))
            except Exception:
                continue
    return "\n".join(diff_out)


def analyze_diff_text(diff_text: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    added_lines = [
        line
        for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    added_text = "\n".join(added_lines)

    for category, patterns in RISKY_CHANGE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, added_text, re.IGNORECASE):
                sev = (
                    "high"
                    if category in ("new_network", "new_subprocess", "new_secrets")
                    else "medium"
                )
                findings.append(
                    {
                        "rule_id": f"DIFF-{category.upper()}",
                        "category": "diff",
                        "severity": sev,
                        "message": f"Risky change: new {category.replace('_', ' ')} detected in diff (pattern: {pat})",
                        "file": "diff",
                        "line": None,
                        "evidence": f"Added line matching {pat}: {next((line for line in added_lines if re.search(pat, line, re.IGNORECASE)), '')[:200]}",
                        "fix": "Review change for least privilege and security impact",
                    }
                )
                break  # one per category

    # Permission escalation: check manifest diff
    if re.search(r'"network"\s*:\s*"unrestricted"', added_text) or re.search(
        r"network:\s*unrestricted", added_text
    ):
        findings.append(
            {
                "rule_id": "DIFF-PERM-ESCALATE",
                "category": "diff",
                "severity": "critical",
                "message": "Permission escalation: network changed to unrestricted",
                "file": "manifest",
                "line": None,
                "evidence": "network=unrestricted in diff",
                "fix": "Require manual review for unrestricted network",
            }
        )
    # New file with executable permission?
    if "Added file:" in diff_text and re.search(r"\.sh|\.py", diff_text):
        # count added files
        added_files = [
            line for line in diff_text.splitlines() if line.startswith("+++ Added file:")
        ]
        if len(added_files) > 3:
            findings.append(
                {
                    "rule_id": "DIFF-NEW-FILES",
                    "category": "diff",
                    "severity": "medium",
                    "message": f"Many new files added ({len(added_files)}) - review for supply chain risk",
                    "file": "diff",
                    "line": None,
                    "evidence": ", ".join(added_files[:5])[:300],
                }
            )

    return findings


def compare_skill_versions(
    skill_path_a: Path, skill_path_b: Path, ref_a: str = "a", ref_b: str = "b"
) -> dict[str, Any]:
    """
    Compare two skill directories in-process using difflib.
    Never run git on scanned content to prevent executing repository-local hooks/config.
    Returns {diff_text, findings, stats}
    """
    diff_text = ""
    try:
        if (
            skill_path_a.exists()
            and skill_path_b.exists()
            and skill_path_a.resolve() != skill_path_b.resolve()
        ):
            diff_text = _difflib_dir_compare(skill_path_a, skill_path_b)
    except Exception:
        diff_text = ""

    findings = analyze_diff_text(diff_text)
    stats = {
        "added_lines": len(
            [
                line
                for line in diff_text.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            ]
        ),
        "removed_lines": len(
            [
                line
                for line in diff_text.splitlines()
                if line.startswith("-") and not line.startswith("---")
            ]
        ),
        "diff_size": len(diff_text),
    }
    return {"diff_text": diff_text, "findings": findings, "stats": stats}


def diff_against_previous_version(
    skill_path: Path, previous_snapshot: Path | None = None
) -> dict[str, Any]:
    """Compare current skill against an explicit snapshot in-process if provided.

    Never invokes git on scanned content.
    """
    if previous_snapshot and previous_snapshot.exists():
        return compare_skill_versions(previous_snapshot, skill_path)
    return {
        "diff_text": "",
        "findings": [],
        "stats": {
            "added_lines": 0,
            "removed_lines": 0,
            "diff_size": 0,
        },
    }
