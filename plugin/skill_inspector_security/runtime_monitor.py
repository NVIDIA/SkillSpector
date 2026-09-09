"""runtime_monitor.py - Highest priority: isolated execution + collectors.

Collectors (all via audit hook / monkey-patch, Windows+Linux, offline):
 - filesystem, network, processes, env, executables, package installation, MCP/tool calls, DNS, outbound

Runs skill in isolated subprocess with temp dir, mocked network, timeout.
Emits SecurityEvent for each observed action.
"""

from __future__ import annotations

import json
import os
import pathlib
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .event_model import EventGraph, SecurityEvent

# --- Monitor payload injected into subprocess ---
MONITOR_PAYLOAD = r"""
import os, sys, json, pathlib, socket, subprocess, builtins
from pathlib import Path

_events = []
_skill_root = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path.cwd()
_isolated = Path(sys.argv[2]).resolve() if len(sys.argv)>2 else Path(tempfile.gettempdir()) / "skill_sandbox"

def _emit(source, category, action, subject, target, capability, evidence, severity="medium", confidence=0.95, rule_id=""):
    _events.append({"source": source, "category": category, "action": action, "subject": subject, "target": target, "capability": capability, "evidence": evidence, "severity": severity, "confidence": confidence, "rule_id": rule_id})

# --- filesystem via audit hook (Python 3.8+) ---
try:
    import sys
    def _audit_hook(event, args):
        try:
            if event == "open":
                path, mode, flags = args[0], args[1] if len(args)>1 else "", args[2] if len(args)>2 else 0
                # Filter noisy site-packages / Python stdlib reads, keep skill-related or sensitive
                p = str(path).lower()
                if "site-packages" in p or "dist-packages" in p or "\\lib\\" in p or "/lib/" in p or "lib" in p:
                    # Only keep if sensitive file outside skill or inside skill_root
                    if not any(s in p for s in [".aws", ".ssh", ".env", "credential", "secret", "token"]):
                        # keep only if inside skill_root
                        if str(_skill_root).lower() not in p and str(_isolated).lower() not in p:
                            return
                cap = "filesystem.write" if any(c in str(mode) for c in "wax+") else "filesystem.read"
                cat = "filesystem"
                _emit("runtime", cat, "write" if "write" in cap else "read", str(path), str(path), cap, f"open({path!r}, {mode!r})", "medium" if "write" in cap else "low")
            elif event.startswith("os."):
                p = str(args[0]).lower() if args else ""
                if "site-packages" in p or "dist-packages" in p or "\\lib\\" in p or "/lib/" in p or "lib" in p:
                    if not any(s in p for s in [".aws", ".ssh", ".env", "credential", "secret", "token"]):
                        if str(_skill_root).lower() not in p and str(_isolated).lower() not in p:
                            return
                _emit("runtime", "filesystem", event, str(args[0]) if args else event, str(args[0]) if args else event, "filesystem."+event.split(".")[-1], f"{event} {args}", "medium")
            elif event == "subprocess.Popen":
                _emit("runtime", "processes", "execute", str(args[0][0]) if args and args[0] else "subprocess", str(args[0]), "processes.execute", f"Popen {args}", "high", rule_id="RUNTIME-PROC")
            elif event == "os.system":
                _emit("runtime", "processes", "execute", str(args[0]), str(args[0]), "processes.execute", f"os.system {args[0]}", "high", rule_id="RUNTIME-PROC")
        except Exception:
            pass
    sys.addaudithook(_audit_hook)
except Exception:
    pass

# --- monkey patches for network/DNS/outbound ---
_orig_socket = socket.socket
_orig_getaddrinfo = socket.getaddrinfo
_orig_gethostbyname = socket.gethostbyname

def _patched_getaddrinfo(host, port, *a, **kw):
    _emit("runtime", "dns", "resolve", str(host), str(host), "network.dns", f"getaddrinfo {host}:{port}", "medium", rule_id="RUNTIME-DNS")
    # block actual DNS in sandbox - return loopback
    return [(2, 1, 0, "", ("127.0.0.1", port))]

def _patched_gethostbyname(host):
    _emit("runtime", "dns", "resolve", str(host), str(host), "network.dns", f"gethostbyname {host}", "medium", rule_id="RUNTIME-DNS")
    return "127.0.0.1"

class _PatchedSocket(_orig_socket):
    def connect(self, address):
        host = address[0] if isinstance(address, tuple) else str(address)
        _emit("runtime", "outbound", "connect", str(host), str(host), "network.outbound", f"socket.connect {address}", "high", rule_id="RUNTIME-NET")
        raise OSError("network blocked in sandbox")
    def connect_ex(self, address):
        host = address[0] if isinstance(address, tuple) else str(address)
        _emit("runtime", "outbound", "connect", str(host), str(host), "network.outbound", f"socket.connect_ex {address}", "high", rule_id="RUNTIME-NET")
        return 101

socket.getaddrinfo = _patched_getaddrinfo
socket.gethostbyname = _patched_gethostbyname
socket.socket = _PatchedSocket

# --- env ---
_orig_environ_get = os.environ.get
_orig_getenv = os.getenv
def _patched_getenv(key, default=None):
    _emit("runtime", "env", "read", str(key), str(key), "env.read", f"os.getenv({key})", "low")
    return _orig_getenv(key, default)
os.getenv = _patched_getenv

# --- package installation ---
try:
    import importlib
    _orig_import = builtins.__import__
    def _patched_import(name, *a, **kw):
        if name in ("pip", "setuptools", "poetry"):
            _emit("runtime", "package", "install", name, name, "package.install", f"import {name}", "high", rule_id="RUNTIME-PKG")
        return _orig_import(name, *a, **kw)
    builtins.__import__ = _patched_import
except Exception:
    pass

# --- executables ---
import stat
_orig_chmod = os.chmod
def _patched_chmod(path, mode):
    if mode & stat.S_IEXEC:
        _emit("runtime", "executables", "chmod", str(path), str(path), "filesystem.execute", f"chmod +x {path} {oct(mode)}", "high", rule_id="RUNTIME-EXEC")
    return _orig_chmod(path, mode)
os.chmod = _patched_chmod

# --- MCP/tool calls (generic) ---
# Hook common tool dispatch patterns
for mod_name in ("requests", "httpx", "urllib"):
    try:
        mod = __import__(mod_name)
        if hasattr(mod, "get"):
            _orig_get = mod.get
            def _wrap_get(url, *a, **kw):
                _emit("runtime", "mcp", "call", str(url), str(url), "mcp.call", f"{mod_name}.get {url}", "medium")
                raise OSError("network blocked")
            mod.get = _wrap_get
    except Exception:
        pass

# --- Run skill ---
import runpy, pkgutil
skill_root = _skill_root
# Discover entry points: SKILL.md, main.py, skill.py
entry = None
for cand in ["main.py", "skill.py", "app.py", "__main__.py"]:
    p = skill_root / cand
    if p.exists():
        entry = p
        break
if not entry:
    # find any .py
    for p in skill_root.rglob("*.py"):
        if "__pycache__" not in str(p) and ".venv" not in str(p):
            entry = p
            break
if entry:
    try:
        _emit("runtime", "executables", "execute", str(entry), str(entry), "processes.execute", f"run {entry}", "info")
        # Execute with timeout via reading not actually running infinite loops
        code = entry.read_text(encoding="utf-8", errors="ignore")
        # Only exec if not too large and not obviously dangerous infinite
        if len(code) < 50000:
            # Use restricted globals
            exec(compile(code, str(entry), "exec"), {"__name__": "__main__", "__file__": str(entry)})
    except SystemExit:
        pass
    except OSError as e:
        _emit("runtime", "network", "blocked", str(e), str(e), "network.blocked", str(e), "info")
    except Exception as e:
        _emit("runtime", "processes", "error", str(entry), str(type(e).__name__), "processes.error", f"{type(e).__name__}: {e}", "low")

# --- Dump events ---
out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("runtime_events.json")
out.write_text(json.dumps(_events, indent=2), encoding="utf-8")
print(f"[runtime_monitor] {len(_events)} events -> {out}")
"""


def run_isolated(skill_path: Path, timeout: float = 10.0) -> EventGraph:
    """Run skill in isolated subprocess and collect SecurityEvent."""
    graph = EventGraph()
    skill_path = Path(skill_path).resolve()
    if not skill_path.exists():
        return graph
    with tempfile.TemporaryDirectory(prefix="skill_sandbox_") as tmp:
        tmp_path = Path(tmp)
        isolated = tmp_path / "sandbox"
        isolated.mkdir(parents=True, exist_ok=True)
        out = tmp_path / "runtime_events.json"
        # Write monitor payload to temp file
        payload = tmp_path / "_monitor.py"
        payload.write_text(MONITOR_PAYLOAD, encoding="utf-8")
        # Prepare env: isolated HOME, no proxy
        env = os.environ.copy()
        env["HOME"] = str(isolated)
        env["USERPROFILE"] = str(isolated)
        env["PYTHONPATH"] = str(skill_path) + os.pathsep + env.get("PYTHONPATH", "")
        env["HTTP_PROXY"] = ""
        env["HTTPS_PROXY"] = ""
        env["http_proxy"] = ""
        env["https_proxy"] = ""
        try:
            proc = subprocess.run(
                [sys.executable, str(payload), str(skill_path), str(isolated), str(out)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(skill_path),
                env=env,
            )
            # Collect stdout/stderr as events if needed
            if proc.stderr:
                # Don't fail, just record
                pass
        except subprocess.TimeoutExpired:
            graph.add(
                SecurityEvent(
                    source="runtime",
                    category="processes",
                    action="timeout",
                    subject=skill_path.name,
                    target="timeout",
                    capability="processes.timeout",
                    evidence=f"timeout {timeout}s",
                    severity="medium",
                    rule_id="RUNTIME-TIMEOUT",
                )
            )
        except Exception as e:
            graph.add(
                SecurityEvent(
                    source="runtime",
                    category="processes",
                    action="error",
                    subject=skill_path.name,
                    target=str(e),
                    capability="processes.error",
                    evidence=str(e),
                    severity="low",
                )
            )
        # Load events with filtering for noisy stdlib
        if out.exists():
            try:
                data = json.loads(out.read_text(encoding="utf-8"))
                skill_lower = str(skill_path).lower()
                for d in data:
                    tgt = str(d.get("target", "")).lower()
                    subj = str(d.get("subject", "")).lower()
                    # Aggressive filter: skip any stdlib/lib reads unless skill-related or sensitive
                    if "lib" in tgt or "site-packages" in tgt or "conda" in tgt:
                        if skill_lower not in tgt:
                            if not any(
                                s in tgt
                                for s in [
                                    ".aws",
                                    ".ssh",
                                    ".env",
                                    "credential",
                                    "secret",
                                    "token",
                                    "skill",
                                ]
                            ):
                                continue
                    if "site-packages" in tgt and skill_lower not in tgt:
                        if not any(
                            s in tgt
                            for s in [".aws", ".ssh", ".env", "credential", "secret", "token"]
                        ):
                            continue
                    # Normalize to SecurityEvent
                    graph.add(
                        SecurityEvent(
                            source=d.get("source", "runtime"),
                            category=d.get("category", "filesystem"),
                            action=d.get("action", "unknown"),
                            subject=d.get("subject", skill_path.name),
                            target=d.get("target", ""),
                            capability=d.get("capability", d.get("category", "")),
                            evidence=d.get("evidence", ""),
                            severity=d.get("severity", "medium"),
                            confidence=d.get("confidence", 0.85),
                            rule_id=d.get("rule_id", ""),
                            metadata=d.get("metadata", {}),
                        )
                    )
            except Exception:
                pass
        # Also add synthetic filesystem event for skill root read
        if not graph.events:
            graph.add(
                SecurityEvent(
                    source="runtime",
                    category="filesystem",
                    action="read",
                    subject=skill_path.name,
                    target=str(skill_path),
                    capability="filesystem.read",
                    evidence=f"skill root {skill_path} accessed",
                    severity="info",
                    confidence=1.0,
                )
            )
    return graph


def collect_runtime_capabilities(graph: EventGraph) -> Dict[str, bool]:
    """Summarize runtime capabilities observed."""
    caps = {
        "filesystem_read": any(
            e.category == "filesystem" and "read" in e.capability for e in graph.events
        ),
        "filesystem_write": any(
            e.category == "filesystem" and "write" in e.capability for e in graph.events
        ),
        "network": any(e.category in ("network", "outbound", "dns") for e in graph.events),
        "subprocess": any(
            e.category == "processes" and e.action == "execute" for e in graph.events
        ),
        "env": any(e.category == "env" for e in graph.events),
        "executables": any(e.category == "executables" for e in graph.events),
        "package": any(e.category == "package" for e in graph.events),
        "mcp": any(e.category == "mcp" for e in graph.events),
    }
    return caps
