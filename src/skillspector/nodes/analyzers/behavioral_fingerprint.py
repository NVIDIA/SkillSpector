# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Behavioral fingerprint analyzer: extract and hash behavioral signatures from skills.

Computes a behavioral fingerprint of each skill by extracting:
- Import statements (what modules it uses)
- Function calls (what APIs it invokes)
- File access patterns (what paths it reads/writes)
- Network access patterns (what URLs/domains it contacts)
- Environment variable access (what secrets it reads)

The fingerprint is a deterministic JSON hash that enables:
- Quick comparison against known-bad fingerprints
- Drift detection between skill versions
- Community threat intelligence sharing
"""

from __future__ import annotations

import ast
import hashlib
import json
import re

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Finding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from .common import build_import_aliases, resolve_call_name
from .static_runner import FILE_TYPES, MAX_FILE_CHARS, analyzer_finding_to_finding

ANALYZER_ID = "behavioral_fingerprint"
logger = get_logger(__name__)

_TAG = "Behavioral Fingerprint"

# Patterns for detecting network URLs in code/strings. The HTTP-verb form is
# matched case-sensitively and only when followed by a URL or a path, so prose
# like "get started" or "put the file" is not reported as a network endpoint.
_URL_PATTERN = re.compile(r"https?://[^\s\"']+|wss?://[^\s\"']+")
_HTTP_VERB_PATTERN = re.compile(
    r"(?:POST|GET|PUT|DELETE|PATCH)\s+(?:https?://[^\s\"']+|/[^\s\"']+)"
)

# Ubiquitous standard-library modules that are not risky on their own; they are
# deliberately excluded from the FP4 "dangerous combination" check so that a
# plain `import os, json` in nearly every Python file does not fire.
_EXECUTION_MODULES = frozenset({"subprocess", "pty", "popen", "shlex"})
_SERIALIZATION_MODULES = frozenset({"pickle", "marshal", "shelve"})
_NETWORK_MODULES = frozenset({"requests", "urllib", "httpx", "aiohttp", "socket", "websocket"})
_FILE_MODULES = frozenset({"shutil", "tempfile"})

_DANGEROUS_COMBOS: tuple[tuple[frozenset[str], frozenset[str], str], ...] = (
    (_EXECUTION_MODULES, _NETWORK_MODULES, "execution + network"),
    (_FILE_MODULES, _NETWORK_MODULES, "file_io + network"),
    (_SERIALIZATION_MODULES, _EXECUTION_MODULES, "serialization + execution"),
)

# Patterns for detecting file path access
_PATH_ACCESS_PATTERNS = [
    re.compile(r"(?:open|read|write|read_text|write_text)\s*\(\s*['\"]([^'\"]+)['\"]"),
    re.compile(r"(?:Path|PurePath)\s*\(\s*['\"]([^'\"]+)['\"]"),
    re.compile(r"(?:os\.path\.join|os\.path\.expanduser)\s*\(\s*['\"]([^'\"]+)['\"]"),
    re.compile(r"~/(?:\.ssh|\.aws|\.config|\.env|\.git|Library)"),
]

# Patterns for detecting env var access (both .get()/.pop() calls and []
# subscript reads such as os.environ["API_KEY"]).
_ENV_VAR_PATTERNS = [
    re.compile(r"os\.environ\s*\[\s*['\"]([A-Z_]+)['\"]"),
    re.compile(r"os\.environ(?:\.get|\.pop)\s*\(\s*['\"]([A-Z_]+)['\"]"),
    re.compile(r"os\.getenv\s*\(\s*['\"]([A-Z_]+)['\"]"),
    re.compile(r"ENV\s+([A-Z_]+)="),
]

# Dangerous file paths that indicate credential access
_SENSITIVE_PATHS = frozenset({
    "~/.ssh", "~/.aws", "~/.config", "~/.env", "~/.git",
    "/etc/passwd", "/etc/shadow", "/etc/hosts",
    "~/.bashrc", "~/.zshrc", "~/.profile",
})


def _extract_imports(tree: ast.Module) -> list[str]:
    """Extract all import names from a Python AST."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)
    return sorted(set(imports))


def _extract_function_calls(tree: ast.Module, aliases: dict[str, str]) -> list[str]:
    """Extract all function call names from a Python AST."""
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = resolve_call_name(node, aliases)
            if name:
                calls.append(name)
    return sorted(set(calls))


def _extract_string_literals(tree: ast.Module) -> list[str]:
    """Extract all string literals from a Python AST."""
    strings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) > 3:
                strings.append(node.value)
    return strings


def _extract_urls(text: str) -> list[str]:
    """Find URLs and HTTP endpoint references in *text*."""
    urls = set()
    for match in _URL_PATTERN.finditer(text):
        urls.add(match.group(0).strip())
    for match in _HTTP_VERB_PATTERN.finditer(text):
        urls.add(match.group(0).strip())
    return sorted(urls)


def _detect_urls_in_strings(strings: list[str]) -> list[str]:
    """Find URLs in string literals."""
    urls = set()
    for s in strings:
        urls.update(_extract_urls(s))
    return sorted(urls)


def _detect_file_paths_in_strings(strings: list[str]) -> list[str]:
    """Find file path references in string literals."""
    paths = set()
    for s in strings:
        for pattern in _PATH_ACCESS_PATTERNS:
            for match in pattern.finditer(s):
                paths.add(match.group(1) if match.lastindex else match.group(0))
        for sensitive in _SENSITIVE_PATHS:
            if sensitive in s:
                paths.add(sensitive)
    return sorted(paths)


def _detect_env_vars(content: str) -> list[str]:
    """Find environment variable accesses in code."""
    env_vars = set()
    for pattern in _ENV_VAR_PATTERNS:
        for match in pattern.finditer(content):
            env_vars.add(match.group(1))
    return sorted(env_vars)


def _detect_dangerous_combos(imports: list[str]) -> list[str]:
    """Return the dangerous import combinations present in *imports*."""
    roots = {imp.split(".")[0] for imp in imports}
    combos = []
    for exec_mods, other_mods, label in _DANGEROUS_COMBOS:
        if roots & exec_mods and roots & other_mods:
            combos.append(label)
    return combos


def _compute_fingerprint(
    imports: list[str],
    calls: list[str],
    urls: list[str],
    file_paths: list[str],
    env_vars: list[str],
) -> str:
    """Compute a deterministic SHA-256 hash of the behavioral fingerprint."""
    fingerprint_data = {
        "imports": imports,
        "calls": calls,
        "urls": urls,
        "file_paths": file_paths,
        "env_vars": env_vars,
    }
    canonical = json.dumps(fingerprint_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _analyze_python_fingerprint(
    content: str, file_path: str
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """Extract behavioral features from a Python file."""
    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        return [], [], [], [], []

    aliases = build_import_aliases(tree)
    imports = _extract_imports(tree)
    calls = _extract_function_calls(tree, aliases)
    strings = _extract_string_literals(tree)
    urls = _detect_urls_in_strings(strings)
    file_paths = _detect_file_paths_in_strings(strings)
    env_vars = _detect_env_vars(content)
    return imports, calls, urls, file_paths, env_vars


def _analyze_markdown_fingerprint(content: str) -> tuple[list[str], list[str], list[str]]:
    """Extract behavioral features from markdown/config files."""
    urls = _extract_urls(content)
    env_vars = set()
    for pattern in _ENV_VAR_PATTERNS:
        for match in pattern.finditer(content):
            env_vars.add(match.group(1))
    file_paths = set()
    for pattern in _PATH_ACCESS_PATTERNS:
        for match in pattern.finditer(content):
            file_paths.add(match.group(1) if match.lastindex else match.group(0))
    for sensitive in _SENSITIVE_PATHS:
        if sensitive in content:
            file_paths.add(sensitive)
    return urls, sorted(file_paths), sorted(env_vars)


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content and extract behavioral fingerprint features."""
    findings: list[AnalyzerFinding] = []

    if file_type == "python":
        imports, calls, urls, file_paths, env_vars = _analyze_python_fingerprint(content, file_path)
    elif file_type in ("markdown", "yaml", "json", "toml"):
        urls, file_paths, env_vars = _analyze_markdown_fingerprint(content)
        imports, calls = [], []
    else:
        return findings

    # FP1: Sensitive file path access
    sensitive_access = [p for p in file_paths if p in _SENSITIVE_PATHS]
    if sensitive_access:
        findings.append(
            AnalyzerFinding(
                rule_id="FP1",
                message=f"Sensitive file path access detected: {', '.join(sensitive_access)}",
                severity=Severity.HIGH,
                location=Location(file=file_path, start_line=1),
                confidence=0.8,
                tags=[_TAG],
                context=f"Accessed paths: {', '.join(sensitive_access)}",
                matched_text=", ".join(sensitive_access),
            )
        )

    # FP2: Credential-related env var access
    credential_envs = [v for v in env_vars if any(
        kw in v for kw in ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH")
    )]
    if credential_envs:
        findings.append(
            AnalyzerFinding(
                rule_id="FP2",
                message=f"Credential environment variable access: {', '.join(credential_envs)}",
                severity=Severity.MEDIUM,
                location=Location(file=file_path, start_line=1),
                confidence=0.7,
                tags=[_TAG],
                context=f"Env vars: {', '.join(credential_envs)}",
                matched_text=", ".join(credential_envs),
            )
        )

    # FP3: External network endpoints
    external_urls = [u for u in urls if not u.startswith(("http://localhost", "http://127.", "http://0."))]
    if external_urls:
        findings.append(
            AnalyzerFinding(
                rule_id="FP3",
                message=f"External network endpoints referenced: {len(external_urls)} URL(s)",
                severity=Severity.LOW,
                location=Location(file=file_path, start_line=1),
                confidence=0.5,
                tags=[_TAG],
                context=f"URLs: {', '.join(external_urls[:5])}",
                matched_text=", ".join(external_urls[:5]),
            )
        )

    # FP4: Dangerous import combination
    if imports:
        dangerous_combos = _detect_dangerous_combos(imports)
        if dangerous_combos:
            findings.append(
                AnalyzerFinding(
                    rule_id="FP4",
                    message=f"Dangerous import combination: {', '.join(dangerous_combos)}",
                    severity=Severity.MEDIUM,
                    location=Location(file=file_path, start_line=1),
                    confidence=0.65,
                    tags=[_TAG],
                    context=f"Modules: {', '.join(imports[:10])}",
                    matched_text=", ".join(dangerous_combos),
                )
            )

    return findings


def _infer_file_type(path: str) -> str:
    """Infer file type from path (extension)."""
    idx = path.rfind(".")
    suffix = path[idx:].lower() if idx >= 0 else ""
    return FILE_TYPES.get(suffix, "other")


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Compute behavioral fingerprints and detect risky behavioral patterns."""
    components: list[str] = state.get("components") or []
    file_cache: dict[str, str] = state.get("file_cache") or {}
    all_findings: list[Finding] = []

    all_imports: list[str] = []
    all_calls: list[str] = []
    all_urls: list[str] = []
    all_paths: list[str] = []
    all_envs: list[str] = []

    for path in components:
        content = file_cache.get(path)
        if content is None or len(content) > MAX_FILE_CHARS:
            continue
        file_type = _infer_file_type(path)
        if file_type == "other":
            continue
        raw = analyze(content, path, file_type)
        all_findings.extend(analyzer_finding_to_finding(af) for af in raw)

        # Aggregate fingerprint features in the same pass (single parse per file).
        if file_type == "python":
            i, c, u, p, e = _analyze_python_fingerprint(content, path)
        else:
            u, p, e = _analyze_markdown_fingerprint(content)
            i, c = [], []
        all_imports.extend(i)
        all_calls.extend(c)
        all_urls.extend(u)
        all_paths.extend(p)
        all_envs.extend(e)

    fingerprint = _compute_fingerprint(
        sorted(set(all_imports)),
        sorted(set(all_calls)),
        sorted(set(all_urls)),
        sorted(set(all_paths)),
        sorted(set(all_envs)),
    )
    logger.info("%s: %d findings, fingerprint=%s", ANALYZER_ID, len(all_findings), fingerprint[:12])
    return {"findings": all_findings}
