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

"""Static patterns: supply chain (SC1–SC6) and trigger analysis (TR1–TR3).

SC1–SC3: regex-based pattern matching (original implementation).
SC4: Known vulnerable dependencies — live OSV.dev lookup with static fallback.
SC5: Abandoned dependencies — flags known-abandoned or archived packages.
SC6: Typosquatting — flags package names similar to popular packages.
TR1–TR3: Trigger analysis — flags overly broad, shadowing, or baiting triggers.

Node and analyze() in one module.
"""

from __future__ import annotations

import re
import sys
import tomllib
from urllib.parse import urlparse

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Finding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import get_context, get_line_number
from .osv_client import ECOSYSTEM_NPM, ECOSYSTEM_PYPI, VulnResult, query_batch, was_osv_reachable
from .pattern_defaults import PatternCategory
from .static_runner import analyzer_finding_to_finding

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_supply_chain"

# ---------------------------------------------------------------------------
# SC1–SC3: Original regex-based patterns
# ---------------------------------------------------------------------------

SC1_PATTERNS = [
    (r"^[a-zA-Z][a-zA-Z0-9_-]*\s*$", 0.6),
    (r"^[a-zA-Z][a-zA-Z0-9_-]*\s*>=\s*[\d.]+\s*$", 0.5),
    (r"^[a-zA-Z][a-zA-Z0-9_-]*\s*==\s*\*\s*$", 0.7),
    (r'"[^"]+"\s*:\s*"(?:\*|latest)"', 0.7),
    (r'"[^"]+"\s*:\s*"\^[\d.]+"', 0.4),
    (
        r"install\s+(?:the\s+)?latest\s+(?:version\s+)?(?:of\s+)?(?:all\s+)?(?:packages?|dependencies)",
        0.6,
    ),
    (r"(?:don't|do\s+not)\s+(?:pin|lock|specify)\s+(?:package\s+)?versions?", 0.7),
]
SC2_PATTERNS = [
    (r"curl\s+[^|]*\|\s*(?:sudo\s+)?(?:ba)?sh", 0.9),
    (r"wget\s+[^|]*\|\s*(?:sudo\s+)?(?:ba)?sh", 0.9),
    (r"curl\s+[^|]*\|\s*(?:sudo\s+)?(?:python|python3|node|ruby|perl)", 0.9),
    (r"wget\s+[^|]*\|\s*(?:sudo\s+)?(?:python|python3|node|ruby|perl)", 0.9),
    (r"curl\s+[^&]*-o\s+\S+\s*&&\s*(?:sudo\s+)?(?:ba)?sh", 0.8),
    (r"wget\s+[^&]*-O\s+\S+\s*&&\s*(?:sudo\s+)?(?:ba)?sh", 0.8),
    (r"exec\s*\(\s*(?:urllib|requests|httpx)\.[^)]+\.(?:read|text|content)", 0.95),
    (r"eval\s*\(\s*(?:urllib|requests|httpx)\.[^)]+\.(?:read|text|content)", 0.95),
    (r"eval\s*\(\s*(?:await\s+)?fetch\s*\(", 0.9),
    (r"new\s+Function\s*\([^)]*fetch\s*\(", 0.9),
    (r"subprocess\.[^(]+\([^)]*(?:curl|wget)\s+https?://", 0.8),
    (r"download\s+and\s+(?:run|execute)\s+(?:the\s+)?script", 0.7),
    (r"run\s+(?:this|the)\s+(?:following\s+)?(?:curl|wget)\s+command", 0.6),
]
SC3_PATTERNS = [
    (r"exec\s*\(\s*(?:base64\.)?b64decode\s*\(", 0.95),
    (r"eval\s*\(\s*(?:base64\.)?b64decode\s*\(", 0.95),
    (r"exec\s*\(\s*codecs\.decode\s*\([^)]*['\"]hex['\"]\s*\)", 0.95),
    (r"marshal\.loads\s*\(", 0.9),
    (r"exec\s*\(\s*marshal\.loads\s*\(", 0.95),
    (r"exec\s*\(\s*compile\s*\([^)]*base64", 0.9),
    (r"exec\s*\(\s*bytes\.fromhex\s*\(", 0.9),
    (r"exec\s*\(\s*bytearray\.fromhex\s*\(", 0.9),
    (r"exec\s*\(\s*(?:zlib|gzip)\.decompress\s*\(", 0.9),
    (r"eval\s*\(\s*atob\s*\(", 0.9),
    (r"new\s+Function\s*\(\s*atob\s*\(", 0.9),
    (r"_0x[a-f0-9]{4,}\s*\(", 0.8),
    (r"['\"][A-Fa-f0-9]{200,}['\"]", 0.6),
    (r"['\"][A-Za-z0-9+/=]{200,}['\"]", 0.5),
    (r"\(lambda\s+_:\s*exec\s*\(", 0.9),
    (r"__import__\s*\(['\"]os['\"]\s*\)\.system", 0.85),
    (r"decode\s+(?:this|the)\s+(?:base64|hex)\s+(?:and\s+)?(?:run|execute)", 0.8),
]

# ---------------------------------------------------------------------------
# SC4: Known Vulnerable Dependencies
#
# Primary source: live OSV.dev API queries (see osv_client.py).
# Fallback lists below are used when the API is unreachable.
# ---------------------------------------------------------------------------

_FALLBACK_VULNERABLE_PYPI: list[tuple[str, str | None, str, float]] = [
    ("py", None, "CVE-2022-42969 (ReDoS)", 0.7),
    ("pycrypto", None, "CVE-2013-7459 (heap overflow, unmaintained)", 0.8),
    ("pyyaml", "5.4", "CVE-2020-14343 (arbitrary code execution via yaml.load)", 0.75),
    ("urllib3", "1.26.5", "CVE-2021-33503 (ReDoS)", 0.7),
    ("pillow", "9.0.0", "CVE-2022-22817 (arbitrary code execution)", 0.7),
    ("setuptools", "65.5.1", "CVE-2022-40897 (ReDoS)", 0.65),
    ("certifi", "2022.12.07", "CVE-2023-37920 (removed trust root)", 0.7),
    ("requests", "2.31.0", "CVE-2023-32681 (header leak on redirect)", 0.65),
    ("jinja2", "3.1.3", "CVE-2024-22195 (XSS)", 0.7),
    ("cryptography", "41.0.6", "CVE-2023-49083 (NULL dereference)", 0.7),
    ("django", "4.2.7", "CVE-2023-46695 (DoS)", 0.7),
    ("flask", "2.3.2", "CVE-2023-30861 (session cookie)", 0.65),
    ("tornado", "6.3.3", "CVE-2023-28370 (open redirect)", 0.65),
    ("aiohttp", "3.8.6", "CVE-2023-47627 (HTTP request smuggling)", 0.7),
    ("paramiko", "3.4.0", "CVE-2023-48795 (Terrapin SSH)", 0.75),
]

_FALLBACK_VULNERABLE_NPM: list[tuple[str, str | None, str, float]] = [
    ("event-stream", None, "Malicious package (credential theft)", 0.95),
    ("flatmap-stream", None, "Malicious package (cryptocurrency theft)", 0.95),
    ("ua-parser-js", "0.7.31", "Malicious versions (cryptominer)", 0.85),
    ("coa", "2.0.2", "Malicious versions (credential theft)", 0.85),
    ("rc", "1.2.8", "Malicious versions (credential theft)", 0.85),
    ("colors", "1.4.0", "Protestware (infinite loop)", 0.8),
    ("faker", "5.5.3", "Protestware (infinite loop)", 0.8),
    ("node-ipc", "10.1.0", "Protestware (destructive payload)", 0.9),
    ("lodash", "4.17.21", "CVE-2021-23337 (prototype pollution)", 0.65),
]

# ---------------------------------------------------------------------------
# SC5: Abandoned / Unmaintained Dependencies
# ---------------------------------------------------------------------------

_ABANDONED_PACKAGES: set[str] = {
    # Python
    "pycrypto",
    "nose",
    "optparse",
    "distribute",
    "mimetools",
    "multifile",
    "popen2",
    "rfc822",
    "sets",
    "sha",
    "md5",
    "commands",
    "dircache",
    "fpformat",
    "htmllib",
    "ihooks",
    "linuxaudiodev",
    "mhlib",
    "mimify",
    "mutex",
    "new",
    "posixfile",
    "pre",
    "regsub",
    "sgmllib",
    "stat",
    "statvfs",
    "stringold",
    "sunaudiodev",
    "sv",
    "timing",
    "toaiff",
    "user",
    "xmllib",
    # npm
    "request",
    "nomnom",
    "optimist",
    "dominion",
    "npm-conf",
}

# ---------------------------------------------------------------------------
# SC6: Typosquatting — popular packages and edit-distance check
# ---------------------------------------------------------------------------

_POPULAR_PYPI: set[str] = {
    "requests",
    "numpy",
    "pandas",
    "flask",
    "django",
    "boto3",
    "setuptools",
    "pip",
    "urllib3",
    "pyyaml",
    "cryptography",
    "pillow",
    "pydantic",
    "sqlalchemy",
    "pytest",
    "click",
    "jinja2",
    "httpx",
    "aiohttp",
    "fastapi",
    "celery",
    "paramiko",
    "beautifulsoup4",
    "lxml",
    "scrapy",
    "redis",
    "pymongo",
    "psycopg2",
    "matplotlib",
    "scipy",
    "scikit-learn",
    "tensorflow",
    "torch",
    "keras",
    "transformers",
    "openai",
    "langchain",
    "gunicorn",
    "uvicorn",
    "rich",
    "typer",
    "black",
    "ruff",
    "mypy",
    "pylint",
    "flake8",
    "isort",
    "perseus-ctx",
    "mimir-mcp",
}

_POPULAR_NPM: set[str] = {
    "express",
    "react",
    "react-dom",
    "next",
    "vue",
    "angular",
    "lodash",
    "axios",
    "moment",
    "chalk",
    "commander",
    "inquirer",
    "webpack",
    "babel",
    "eslint",
    "prettier",
    "typescript",
    "jest",
    "mocha",
    "chai",
    "puppeteer",
    "socket.io",
    "mongoose",
    "sequelize",
    "passport",
    "jsonwebtoken",
    "dotenv",
    "cors",
    "body-parser",
    "nodemon",
    "pm2",
}


def _edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr_row = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr_row.append(min(curr_row[j] + 1, prev_row[j + 1] + 1, prev_row[j] + cost))
        prev_row = curr_row
    return prev_row[-1]


def _is_typosquat(pkg_name: str, popular: set[str], max_distance: int = 2) -> str | None:
    """Return the popular package name if pkg_name is a close-but-not-exact match."""
    normalized = pkg_name.lower().replace("_", "-")
    for popular_name in popular:
        pop_norm = popular_name.lower().replace("_", "-")
        if normalized == pop_norm:
            return None
        if len(normalized) < 3 or len(pop_norm) < 3:
            continue
        dist = _edit_distance(normalized, pop_norm)
        if not 0 < dist <= max_distance:
            continue
        # Relative-distance guard: a genuine typosquat perturbs only a small
        # fraction of the name. Short, legitimate-but-distinct names collide
        # under an absolute distance of 2 (e.g. "task" is edit-distance 2 from
        # "flask" yet is a real package) and are not typosquats. Require
        # dist/len <= 1/3, so short names need an all-but-one-character match
        # while longer names may still differ by two (e.g. "reqeusts" vs
        # "requests").
        shorter = min(len(normalized), len(pop_norm))
        if dist * 3 > shorter:
            continue
        return popular_name
    return None


# ---------------------------------------------------------------------------
# Trigger analysis helpers
# ---------------------------------------------------------------------------

_BUILTIN_COMMANDS: set[str] = {
    "help",
    "search",
    "find",
    "run",
    "test",
    "build",
    "deploy",
    "install",
    "create",
    "delete",
    "update",
    "list",
    "show",
    "get",
    "set",
    "open",
    "close",
    "start",
    "stop",
    "restart",
    "status",
    "log",
    "debug",
    "commit",
    "push",
    "pull",
    "merge",
    "branch",
    "checkout",
    "rebase",
    "diff",
    "blame",
    "stash",
    "tag",
    "release",
    "version",
    "lint",
    "format",
    "fix",
    "refactor",
    "review",
    "explain",
    "chat",
    "ask",
    "edit",
    "write",
    "read",
    "save",
    "load",
    "copy",
    "move",
}

_OVERLY_BROAD_SINGLE_WORDS: set[str] = {
    "the",
    "a",
    "an",
    "is",
    "it",
    "do",
    "go",
    "make",
    "thing",
    "stuff",
    "code",
    "file",
    "data",
    "text",
    "work",
    "good",
    "bad",
    "yes",
    "no",
    "ok",
    "please",
    "thanks",
    "hi",
    "hello",
    "hey",
}


def _extract_packages_from_requirements(content: str) -> list[tuple[str, str | None, int]]:
    """Extract (package_name, version_or_None, line_number) from requirements.txt format."""
    results: list[tuple[str, str | None, int]] = []
    for i, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        m = re.match(r"^([a-zA-Z][a-zA-Z0-9._-]*)(?:\[.*?\])?\s*(?:([=<>!~]=?)\s*([\d.*]+))?", line)
        if m:
            name = m.group(1)
            version = m.group(3) if m.group(2) else None
            results.append((name, version, i))
    return results


def _extract_packages_from_package_json(content: str) -> list[tuple[str, str | None, int]]:
    """Extract (package_name, version_or_None, line_number) from package.json content."""
    results: list[tuple[str, str | None, int]] = []
    in_deps = False
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        if re.search(r'"(?:dependencies|devDependencies|peerDependencies)"', stripped):
            in_deps = True
            continue
        if in_deps and stripped.startswith("}"):
            in_deps = False
            continue
        if in_deps:
            m = re.match(r'"([^"]+)"\s*:\s*"([^"]*)"', stripped)
            if m:
                name = m.group(1)
                ver_str = m.group(2).lstrip("^~>=<")
                version = ver_str if re.match(r"^\d", ver_str) else None
                results.append((name, version, i))
    return results


def _extract_packages_from_pyproject(content: str) -> list[tuple[str, str | None, int]]:
    """Extract (package_name, version_or_None, line_number) from pyproject.toml.

    Reads PEP 621 ``[project]`` ``dependencies`` / ``optional-dependencies``,
    PEP 735 ``[dependency-groups]``, and ``[build-system].requires``. Standard
    metadata keys (``requires-python``, ``name``, ``version``, ...) are not
    dependencies and must not be looked up as packages.
    """
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return []

    specs: list[str] = []
    project = data.get("project")
    if isinstance(project, dict):
        deps = project.get("dependencies")
        if isinstance(deps, list):
            specs.extend(d for d in deps if isinstance(d, str))
        optional = project.get("optional-dependencies")
        if isinstance(optional, dict):
            for group in optional.values():
                if isinstance(group, list):
                    specs.extend(d for d in group if isinstance(d, str))
    groups = data.get("dependency-groups")
    if isinstance(groups, dict):
        for group in groups.values():
            if isinstance(group, list):
                specs.extend(d for d in group if isinstance(d, str))
    build_system = data.get("build-system")
    if isinstance(build_system, dict):
        requires = build_system.get("requires")
        if isinstance(requires, list):
            specs.extend(d for d in requires if isinstance(d, str))

    results: list[tuple[str, str | None, int]] = []
    for spec in specs:
        m = re.match(r"^([a-zA-Z][a-zA-Z0-9._-]*)(?:\[.*?\])?\s*(?:([=<>!~]=?)\s*([\d.*]+))?", spec)
        if not m:
            continue
        name = m.group(1)
        version = m.group(3) if m.group(2) in ("==", "<=") else None
        idx = content.find(spec)
        line_num = get_line_number(content, idx) if idx >= 0 else 1
        results.append((name, version, line_num))
    return results


def _version_lt(v1: str, v2: str) -> bool:
    """Simple version comparison: True if v1 < v2 (numeric tuple comparison)."""

    def parts(v: str) -> tuple[int, ...]:
        return tuple(int(x) for x in re.findall(r"\d+", v))

    try:
        return parts(v1) < parts(v2)
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Main analyze() — SC1–SC3 regex patterns
# ---------------------------------------------------------------------------


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for supply chain patterns (SC1–SC3)."""
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [PatternCategory.SUPPLY_CHAIN.value]

    is_dep_file = any(
        n in file_path.lower()
        for n in ["requirements", "package.json", "pyproject.toml", "setup.py", "pipfile"]
    )
    if is_dep_file:
        for pattern, confidence in SC1_PATTERNS:
            for match in re.finditer(pattern, content, re.MULTILINE):
                line_num = get_line_number(content, match.start())
                findings.append(
                    AnalyzerFinding(
                        rule_id="SC1",
                        message="Unpinned Dependencies",
                        severity=Severity.LOW,
                        location=loc(line_num),
                        confidence=confidence,
                        tags=tag,
                        context=ctx(match.start()),
                        matched_text=match.group(0)[:200],
                    )
                )
    for pattern, confidence in SC2_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            mt = match.group(0)
            if _is_safe_supply_chain_pattern(mt):
                adj = min(confidence, 0.15)
                sev = Severity.LOW
            else:
                adj = confidence
                sev = Severity.HIGH
            findings.append(
                AnalyzerFinding(
                    rule_id="SC2",
                    message="External Script Fetching",
                    severity=sev,
                    location=loc(line_num),
                    confidence=adj,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=mt[:200],
                )
            )
    if file_type in ("python", "javascript", "shell", "other"):
        for pattern, confidence in SC3_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
                line_num = get_line_number(content, match.start())
                findings.append(
                    AnalyzerFinding(
                        rule_id="SC3",
                        message="Obfuscated Code",
                        severity=Severity.HIGH,
                        location=loc(line_num),
                        confidence=confidence,
                        tags=tag,
                        context=ctx(match.start()),
                        matched_text=match.group(0)[:200],
                    )
                )
    return findings


_TRUSTED_DOMAINS: tuple[str, ...] = (
    "deb.nodesource.com",
    "rpm.nodesource.com",
    "get.docker.com",
    "install.python-poetry.org",
    "raw.githubusercontent.com",
    "brew.sh",
    "rustup.rs",
    "pypa.io",
    "pip.pypa.io",
    "astral.sh",
    "pypi.org",
    "npmjs.com",
    "github.com",
)

_SAFE_INSTALL_PATTERN = re.compile(r"(?:pip|npm)\s+install", re.IGNORECASE)
_URL_TOKEN_PATTERN = re.compile(
    r"https?://[^\s|;&)]+|(?<![?=&/])(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s|;&)]*)?",
    re.IGNORECASE,
)


def _is_trusted_source(text: str) -> bool:
    for match in _URL_TOKEN_PATTERN.finditer(text):
        token = match.group(0).strip("\"'`<>()[]{}")
        parsed = urlparse(token if "://" in token else f"//{token}")
        hostname = (parsed.hostname or "").rstrip(".").lower()
        if any(
            hostname == domain or hostname.endswith(f".{domain}") for domain in _TRUSTED_DOMAINS
        ):
            return True
    return False


def _is_safe_supply_chain_pattern(text: str) -> bool:
    """Return True when the matched text is a known-safe install or fetch pattern."""
    return _is_trusted_source(text) or bool(_SAFE_INSTALL_PATTERN.search(text))


# ---------------------------------------------------------------------------
# SC4–SC6: Dependency-level analysis (runs per dependency file)
# ---------------------------------------------------------------------------


_SEVERITY_CONFIDENCE: dict[str, float] = {
    "CRITICAL": 0.9,
    "HIGH": 0.8,
    "MEDIUM": 0.7,
    "LOW": 0.6,
}

_SEVERITY_ORDER: dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def _osv_severity_to_app(sev: str) -> Severity:
    upper = sev.upper()
    if upper == "CRITICAL":
        return Severity.CRITICAL
    if upper == "HIGH":
        return Severity.HIGH
    if upper == "MEDIUM":
        return Severity.MEDIUM
    return Severity.LOW


def _format_vuln_ids(vulns: list[VulnResult]) -> str:
    """Build a human-readable summary string from OSV results."""
    ids = []
    for v in vulns[:3]:
        label = v.vuln_id
        if v.aliases:
            cves = [a for a in v.aliases if a.startswith("CVE-")]
            if cves:
                label = cves[0]
        if v.summary:
            label = f"{label} ({v.summary[:80]})"
        ids.append(label)
    suffix = f" +{len(vulns) - 3} more" if len(vulns) > 3 else ""
    return "; ".join(ids) + suffix


def _sc4_from_osv(
    packages: list[tuple[str, str | None, int]],
    ecosystem: str,
    file_path: str,
    tag: list[str],
) -> tuple[list[AnalyzerFinding], set[str]]:
    """Query OSV.dev and emit SC4 findings for vulnerable packages.

    Returns:
        A tuple of (findings, covered_packages) where *covered_packages* is
        the set of normalised package names for which OSV returned at least
        one vulnerability.  Callers can use this to decide which packages
        still need a fallback lookup.
    """
    pkg_pairs = [(name, version) for name, version, _ in packages]
    osv_results = query_batch(pkg_pairs, ecosystem)

    findings: list[AnalyzerFinding] = []
    covered: set[str] = set()
    for (pkg_name, pkg_version, line_num), vulns in zip(packages, osv_results, strict=False):
        if not vulns:
            continue
        covered.add(pkg_name.lower().replace("_", "-"))
        worst_severity = "LOW"
        for v in vulns:
            if _SEVERITY_ORDER.get(v.severity.upper(), 0) > _SEVERITY_ORDER.get(
                worst_severity.upper(), 0
            ):
                worst_severity = v.severity
        severity = _osv_severity_to_app(worst_severity)
        confidence = _SEVERITY_CONFIDENCE.get(worst_severity.upper(), 0.75)
        version_str = f"=={pkg_version}" if pkg_version else ""
        vuln_desc = _format_vuln_ids(vulns)
        findings.append(
            AnalyzerFinding(
                rule_id="SC4",
                message=(
                    f"Known Vulnerable Dependency: {pkg_name}{version_str}"
                    f" — {len(vulns)} advisory(ies): {vuln_desc}"
                ),
                severity=severity,
                location=Location(file=file_path, start_line=line_num),
                confidence=confidence,
                tags=tag,
                matched_text=f"{pkg_name}{version_str}" if version_str else pkg_name,
            )
        )
    return findings, covered


def _sc4_from_fallback(
    packages: list[tuple[str, str | None, int]],
    fallback_db: list[tuple[str, str | None, str, float]],
    file_path: str,
    tag: list[str],
) -> list[AnalyzerFinding]:
    """Emit SC4 findings from the static fallback list (offline mode)."""
    findings: list[AnalyzerFinding] = []
    for pkg_name, pkg_version, line_num in packages:
        pkg_lower = pkg_name.lower().replace("_", "-")
        for vuln_name, max_safe, cve_info, confidence in fallback_db:
            if pkg_lower != vuln_name.lower().replace("_", "-"):
                continue
            if max_safe is None:
                findings.append(
                    AnalyzerFinding(
                        rule_id="SC4",
                        message=f"Known Vulnerable Dependency: {pkg_name} ({cve_info})",
                        severity=Severity.HIGH,
                        location=Location(file=file_path, start_line=line_num),
                        confidence=confidence,
                        tags=tag,
                        matched_text=pkg_name,
                    )
                )
            elif pkg_version and _version_lt(pkg_version, max_safe):
                findings.append(
                    AnalyzerFinding(
                        rule_id="SC4",
                        message=(
                            f"Known Vulnerable Dependency: {pkg_name}=={pkg_version}"
                            f" (fix: >={max_safe}, {cve_info})"
                        ),
                        severity=Severity.HIGH,
                        location=Location(file=file_path, start_line=line_num),
                        confidence=confidence,
                        tags=tag,
                        matched_text=f"{pkg_name}=={pkg_version}",
                    )
                )
    return findings


def _analyze_dependencies(
    content: str,
    file_path: str,
) -> list[AnalyzerFinding]:
    """Run SC4/SC5/SC6 checks on dependency files."""
    findings: list[AnalyzerFinding] = []
    tag = [PatternCategory.SUPPLY_CHAIN.value]

    lower_path = file_path.lower()
    is_python_dep = any(
        n in lower_path for n in ["requirements", "pyproject.toml", "setup.py", "pipfile"]
    )
    is_npm_dep = "package.json" in lower_path

    if not is_python_dep and not is_npm_dep:
        return findings

    if is_python_dep:
        if "pyproject.toml" in lower_path:
            packages = _extract_packages_from_pyproject(content)
        else:
            packages = _extract_packages_from_requirements(content)
        ecosystem = ECOSYSTEM_PYPI
        fallback_db = _FALLBACK_VULNERABLE_PYPI
        popular = _POPULAR_PYPI
    else:
        packages = _extract_packages_from_package_json(content)
        ecosystem = ECOSYSTEM_NPM
        fallback_db = _FALLBACK_VULNERABLE_NPM
        popular = _POPULAR_NPM

    # SC4: Live OSV.dev lookup, then static fallback for uncovered packages
    osv_findings, osv_covered = _sc4_from_osv(packages, ecosystem, file_path, tag)
    findings.extend(osv_findings)
    uncovered_packages = [p for p in packages if p[0].lower().replace("_", "-") not in osv_covered]
    fallback_findings = _sc4_from_fallback(uncovered_packages, fallback_db, file_path, tag)
    if fallback_findings:
        logger.debug(
            "SC4: using static fallback for %d uncovered packages", len(uncovered_packages)
        )
    elif uncovered_packages and not osv_findings and not was_osv_reachable():
        # OSV.dev was unreachable and fallback found nothing — surface the gap
        findings.append(
            AnalyzerFinding(
                rule_id="SC4",
                message=(
                    f"🟡 SC4: OSV.dev unreachable, using static fallback "
                    f"({len(fallback_db)} packages). "
                    "Results may be incomplete. Set SKILLSPECTOR_OSV_TIMEOUT to increase "
                    "timeout or check network connectivity to api.osv.dev."
                ),
                severity=Severity.LOW,
                location=Location(file=file_path, start_line=1),
                confidence=1.0,
                tags=tag,
                matched_text="SC4 fallback active",
            )
        )
    findings.extend(fallback_findings)

    for pkg_name, _pkg_version, line_num in packages:
        pkg_lower = pkg_name.lower().replace("_", "-")

        # SC5: Abandoned dependencies
        if pkg_lower in {a.lower().replace("_", "-") for a in _ABANDONED_PACKAGES}:
            findings.append(
                AnalyzerFinding(
                    rule_id="SC5",
                    message=f"Abandoned Dependency: {pkg_name} is unmaintained and no longer receives security updates",
                    severity=Severity.MEDIUM,
                    location=Location(file=file_path, start_line=line_num),
                    confidence=0.75,
                    tags=tag,
                    matched_text=pkg_name,
                )
            )

        # SC6: Typosquatting
        similar = _is_typosquat(pkg_name, popular)
        if similar:
            findings.append(
                AnalyzerFinding(
                    rule_id="SC6",
                    message=f"Possible Typosquatting: '{pkg_name}' resembles popular package '{similar}'",
                    severity=Severity.HIGH,
                    location=Location(file=file_path, start_line=line_num),
                    confidence=0.7,
                    tags=tag,
                    matched_text=pkg_name,
                )
            )

    return findings


# ---------------------------------------------------------------------------
# Trigger analysis (TR1–TR3): operates on manifest from state
# ---------------------------------------------------------------------------


def _analyze_triggers(manifest: dict[str, object], skill_path: str) -> list[Finding]:
    """Analyze the triggers field from SKILL.md manifest for abuse patterns."""
    triggers: list[str] = []
    raw = manifest.get("triggers", [])
    if isinstance(raw, list):
        triggers = [str(t).strip() for t in raw if str(t).strip()]
    if not triggers:
        return []

    findings: list[Finding] = []
    tag = [PatternCategory.TRIGGER_ABUSE.value]
    file_ref = "SKILL.md"

    for i, trigger in enumerate(triggers, 1):
        trigger_lower = trigger.lower().strip()
        words = trigger_lower.split()

        # TR1: Overly broad triggers (single common word or very short)
        if len(words) == 1 and trigger_lower in _OVERLY_BROAD_SINGLE_WORDS:
            findings.append(
                Finding(
                    rule_id="TR1",
                    message=f"Overly Broad Trigger: '{trigger}' is a common word that will activate in many unintended contexts",
                    severity="LOW",
                    confidence=0.75,
                    file=file_ref,
                    start_line=i,
                    tags=tag,
                    matched_text=trigger,
                    category=PatternCategory.TRIGGER_ABUSE.value,
                    pattern="Overly Broad Trigger",
                )
            )
        elif len(trigger_lower) <= 2:
            findings.append(
                Finding(
                    rule_id="TR1",
                    message=f"Overly Broad Trigger: '{trigger}' is too short and may match unintended inputs",
                    severity="LOW",
                    confidence=0.7,
                    file=file_ref,
                    start_line=i,
                    tags=tag,
                    matched_text=trigger,
                    category=PatternCategory.TRIGGER_ABUSE.value,
                    pattern="Overly Broad Trigger",
                )
            )

        # TR2: Shadow commands (conflicts with built-in commands)
        if trigger_lower in _BUILTIN_COMMANDS or (
            len(words) > 0 and words[0] in _BUILTIN_COMMANDS and len(words) <= 2
        ):
            findings.append(
                Finding(
                    rule_id="TR2",
                    message=f"Shadow Command Trigger: '{trigger}' conflicts with built-in command '{words[0]}'",
                    severity="MEDIUM",
                    confidence=0.7,
                    file=file_ref,
                    start_line=i,
                    tags=tag,
                    matched_text=trigger,
                    category=PatternCategory.TRIGGER_ABUSE.value,
                    pattern="Shadow Command Trigger",
                )
            )

        # TR3: Keyword baiting (trigger is generic/vague, designed to maximize activation)
        baiting_patterns = [
            r"^(?:anything|everything|whatever|always|any\s+(?:question|request|task|input))$",
            r"^(?:when(?:ever)?|if|every\s+time)\s+(?:the\s+)?user\s+(?:says?|asks?|types?|sends?)\s+(?:anything|something|a\s+message)$",
            r"^(?:all|any|every)\s+(?:messages?|inputs?|requests?|queries?|questions?)$",
        ]
        for bp in baiting_patterns:
            if re.search(bp, trigger_lower):
                findings.append(
                    Finding(
                        rule_id="TR3",
                        message=f"Keyword Baiting Trigger: '{trigger}' is designed to match all or most user inputs",
                        severity="MEDIUM",
                        confidence=0.8,
                        file=file_ref,
                        start_line=i,
                        tags=tag,
                        matched_text=trigger,
                        category=PatternCategory.TRIGGER_ABUSE.value,
                        pattern="Keyword Baiting Trigger",
                    )
                )
                break

    return findings


# ---------------------------------------------------------------------------
# Graph node
# ---------------------------------------------------------------------------


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run supply_chain patterns (SC1–SC6) and trigger analysis (TR1–TR3)."""
    # SC1–SC3 via static_runner
    findings = static_runner.run_static_patterns(state, [sys.modules[__name__]])

    # SC4–SC6: dependency-level analysis on dependency files
    components: list[str] = state.get("components") or []
    file_cache: dict[str, str] = state.get("file_cache") or {}
    for path in components:
        lower_path = path.lower()
        is_dep_file = any(
            n in lower_path
            for n in ["requirements", "package.json", "pyproject.toml", "setup.py", "pipfile"]
        )
        if not is_dep_file:
            continue
        content = file_cache.get(path)
        if not content:
            continue
        dep_findings = _analyze_dependencies(content, path)
        findings.extend(analyzer_finding_to_finding(af) for af in dep_findings)

    # TR1–TR3: trigger analysis from manifest
    manifest: dict[str, object] = state.get("manifest") or {}
    if manifest:
        skill_path = state.get("skill_path") or ""
        trigger_findings = _analyze_triggers(manifest, skill_path)
        findings.extend(trigger_findings)

    logger.info("%s: %d findings", ANALYZER_ID, len(findings))
    return {"findings": findings}
