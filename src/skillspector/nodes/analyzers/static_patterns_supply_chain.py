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

"""Static patterns: supply chain (SC1–SC10) and trigger analysis (TR1–TR3).

SC1–SC3: regex-based pattern matching (original implementation).
SC4: Known vulnerable dependencies — live OSV.dev lookup with static fallback.
SC5: Abandoned dependencies — flags known-abandoned or archived packages.
SC6: Typosquatting — flags package names similar to popular packages.
SC7: Untrusted container image — flags image signature / registry-verification bypass.
SC8: Shipped Python bytecode — flags __pycache__/ and *.pyc/*.pyo that discovery skips.
SC9: Concealed executable artifact — flags executables nested in document or hidden artifacts.
SC10: Dependency source redirection — flags noncanonical package registries and indexes.
TR1–TR3: Trigger analysis — flags overly broad, shadowing, or baiting triggers.

Node and analyze() in one module.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tomllib
from pathlib import Path
from urllib.parse import urlparse

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

from skillspector.dependency_sources import analyze_dependency_sources
from skillspector.inspection_ledger import LedgerOutcome, analyzer_status_for_events, ledger_event
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

# SC7: Untrusted Container Image — pulling images with signature/registry
# verification turned off. These flags disable image trust regardless of the
# registry, so they are a strong supply-chain signal with near-zero FP.
# (`--tls-verify=false` is intentionally omitted: TM3's `verify=False` already
# covers it; SC7 targets the image-specific bypasses TM3 does not see.)
SC7_PATTERNS = [
    (
        r"--disable-content-trust\b(?!=false)",
        0.85,
    ),  # Content Trust off (exclude =false, which keeps it on)
    (r"DOCKER_CONTENT_TRUST\s*=\s*0", 0.85),  # signature verification disabled via env
    (r"--insecure-registry", 0.8),  # registry TLS verification off
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


def _pinned_version(operator: str | None, version: str | None) -> str | None:
    """Return *version* only when the specifier pins one concrete release.

    A vulnerability lookup answers "is THIS release affected?". That question is only
    meaningful when the manifest admits exactly one release. Under PEP 440 that is ``==``
    with a fully concrete version: floors (``>=``, ``>``), caps (``<=``, ``<``), exclusions
    (``!=``), compatible releases (``~=``) and wildcard equality (``==1.*``) all admit more
    than one, so the installed version is unknown and must not be passed off as a pin.
    """
    if operator != "==" or not version or "*" in version:
        return None
    try:
        Version(version)
    except InvalidVersion:
        return None
    return version


def _extract_python_requirement(spec: str) -> tuple[str, str | None] | None:
    """Extract a package and a concrete PEP 440 pin from a PEP 508 requirement.

    ``packaging`` parses complete specifiers rather than accepting a numeric prefix.
    That keeps valid PEP 440 versions such as ``10.0.0rc1``, ``10.0.0.post1``,
    and ``1!10.0`` intact for OSV queries.
    """
    try:
        requirement = Requirement(spec)
    except InvalidRequirement:
        return None

    specifiers = list(requirement.specifier)
    if len(specifiers) != 1:
        return requirement.name, None
    specifier = specifiers[0]
    return requirement.name, _pinned_version(specifier.operator, specifier.version)


def _logical_requirement_lines(content: str) -> list[tuple[int, str]]:
    """Join pip-style continuations and retain each logical line's first line number."""
    logical_lines: list[tuple[int, str]] = []
    parts: list[str] = []
    start_line = 1

    for line_num, line in enumerate(content.splitlines(), 1):
        if not parts:
            start_line = line_num

        is_comment = line.lstrip().startswith("#")
        if line.endswith("\\") and not is_comment:
            parts.append(line.strip("\\"))
            continue

        if is_comment:
            # pip prefixes a comment that closes a continued line with a space,
            # allowing its later comment-stripping pass to recognize it.
            line = " " + line
        parts.append(line)
        logical_lines.append((start_line, "".join(parts)))
        parts = []

    if parts:
        logical_lines.append((start_line, "".join(parts)))
    return logical_lines


def _strip_pip_per_requirement_options(line: str) -> str:
    """Remove pip-only options while preserving the original PEP 508 prefix."""
    quote: str | None = None
    escaped = False
    token_start = True

    for index, char in enumerate(line):
        if escaped:
            escaped = False
            token_start = False
        elif char == "\\":
            escaped = True
            token_start = False
        elif quote:
            if char == quote:
                quote = None
        elif char in {"'", '"'}:
            quote = char
            token_start = False
        elif char.isspace():
            token_start = True
        elif token_start and char == "-":
            return line[:index].rstrip()
        else:
            token_start = False
    return line


def _pinned_npm_version(spec: str) -> str | None:
    """Return the pinned version of an npm dependency spec, or None for any range.

    npm defaults to caret ranges, so ``"^1.8.3"`` is *not* a pin: stripping the operator
    turns a range into a concrete release that the project may never install.
    """
    candidate = spec.strip()
    if re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", candidate):
        return candidate
    return None


def _extract_packages_from_requirements(content: str) -> list[tuple[str, str | None, int]]:
    """Extract (package_name, version_or_None, line_number) from requirements.txt format."""
    results: list[tuple[str, str | None, int]] = []
    for line_num, line in _logical_requirement_lines(content):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # pip treats a whitespace-prefixed ``#`` as an inline comment, while
        # PEP 508 parsing does not. Preserve normal requirements.txt behavior
        # before handing the complete requirement to ``packaging``.
        line = re.split(r"\s+#", line, maxsplit=1)[0]
        line = _strip_pip_per_requirement_options(line)
        requirement = _extract_python_requirement(line)
        if requirement:
            name, version = requirement
            results.append((name, version, line_num))
    return results


_NPM_DEPENDENCY_SECTIONS = ("dependencies", "devDependencies", "peerDependencies")


def _package_json_line(content: str, section: str, name: str) -> int:
    """Best-effort line for a dependency entry, so findings keep pointing somewhere useful.

    Parsing JSON loses positions, and the search starts at the section header so a name that
    also appears in ``scripts`` does not win.
    """
    header = re.search(rf'"{re.escape(section)}"\s*:', content)
    start = header.end() if header else 0
    entry = re.compile(rf'"{re.escape(name)}"\s*:').search(content, start)
    return get_line_number(content, entry.start()) if entry else 1


def _extract_packages_from_package_json_scan(content: str) -> list[tuple[str, str | None, int]]:
    """Line-oriented fallback, used only when the manifest is not valid JSON."""
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
                results.append((m.group(1), _pinned_npm_version(m.group(2)), i))
    return results


def _extract_packages_from_package_json(content: str) -> list[tuple[str, str | None, int]]:
    """Extract (package_name, version_or_None, line_number) from package.json content.

    package.json is JSON, so it is parsed as JSON. Scanning it line by line made the result
    depend on formatting: a manifest written on a single line — which is valid, and what many
    generators emit — never entered the dependency section at all and yielded *no* dependencies,
    silently. The line-oriented scan remains as a fallback for manifests that do not parse.
    """
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        return _extract_packages_from_package_json_scan(content)
    if not isinstance(data, dict):
        return []
    results: list[tuple[str, str | None, int]] = []
    for section in _NPM_DEPENDENCY_SECTIONS:
        deps = data.get(section)
        if not isinstance(deps, dict):
            continue
        for name, spec in deps.items():
            if not isinstance(name, str) or not isinstance(spec, str):
                continue
            line = _package_json_line(content, section, name)
            results.append((name, _pinned_npm_version(spec), line))
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
        requirement = _extract_python_requirement(spec)
        if not requirement:
            continue
        name, version = requirement
        idx = content.find(spec)
        line_num = get_line_number(content, idx) if idx >= 0 else 1
        results.append((name, version, line_num))
    return results


_LOCKFILE_PACKAGE_BLOCK_RE = re.compile(
    r"(?ms)^\s*\[\[package\]\]\s*$.*?(?=^\s*\[\[package\]\]\s*$|\Z)"
)


def _normalize_package_name(name: str) -> str:
    """Normalize package names the same way OSV/fallback coverage does."""
    return name.lower().replace("_", "-")


def _is_python_lockfile(file_path: str) -> bool:
    lower_path = file_path.lower()
    return "uv.lock" in lower_path or "poetry.lock" in lower_path


def _extract_packages_from_toml_lock(content: str) -> list[tuple[str, str | None, int]]:
    """Extract exact package versions from TOML lockfiles such as uv.lock and poetry.lock."""
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return []
    packages = data.get("package")
    if not isinstance(packages, list):
        return []
    blocks = list(_LOCKFILE_PACKAGE_BLOCK_RE.finditer(content))
    results: list[tuple[str, str | None, int]] = []
    for package, block in zip(packages, blocks, strict=False):
        if not isinstance(package, dict):
            continue
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not name.strip():
            continue
        version_value = version.strip() if isinstance(version, str) and version.strip() else None
        name_match = re.search(r"(?m)^\s*name\s*=", block.group(0))
        idx = block.start() + name_match.start() if name_match else block.start()
        line_num = get_line_number(content, idx)
        results.append((name, version_value, line_num))
    return results


def _apply_locked_versions(
    packages: list[tuple[str, str | None, int]],
    locked_versions: dict[str, str] | None,
) -> list[tuple[str, str | None, int]]:
    """Prefer lockfile versions for manifest dependencies without exact versions."""
    if not locked_versions:
        return packages
    resolved: list[tuple[str, str | None, int]] = []
    for name, version, line_num in packages:
        locked_version = locked_versions.get(_normalize_package_name(name))
        resolved.append((name, version or locked_version, line_num))
    return resolved


def _collect_locked_versions(
    file_cache: dict[str, str],
    components: list[str],
) -> dict[str, str]:
    """Build package -> exact version map from Python lockfiles in the project."""
    locked_versions: dict[str, str] = {}
    for path in components:
        if not _is_python_lockfile(path):
            continue
        content = file_cache.get(path)
        if not content:
            continue
        for name, version, _line_num in _extract_packages_from_toml_lock(content):
            if version:
                locked_versions[_normalize_package_name(name)] = version
    return locked_versions


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
    """Analyze content for supply chain patterns (SC1–SC3, SC7)."""
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
    # SC7: untrusted container image. Example filtering is delegated to the runner.
    for pattern, confidence in SC7_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="SC7",
                    message="Untrusted Container Image",
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
        vuln_desc = _format_vuln_ids(vulns)
        if pkg_version:
            message = (
                f"Known Vulnerable Dependency: {pkg_name}=={pkg_version}"
                f" — {len(vulns)} advisory(ies): {vuln_desc}"
            )
            matched_text = f"{pkg_name}=={pkg_version}"
        else:
            # No resolvable version: OSV was queried by name only, so these advisories are
            # NOT matched against the release that will actually be installed — they are the
            # package's history, and the worst of them may predate every version the range
            # admits. Reporting that as the finding's severity turns "setuptools>=61" into a
            # CRITICAL. The unpinned dependency itself is already reported by SC1, so what is
            # left to say here is "could not verify", and it must not outrank a real match.
            severity = Severity.LOW
            confidence = 0.4
            message = (
                f"Unverifiable Dependency: {pkg_name} has {len(vulns)} known advisory(ies)"
                f" ({vuln_desc}), but the manifest does not pin a version, so it is unknown"
                " whether the installed release is affected"
            )
            matched_text = pkg_name
        findings.append(
            AnalyzerFinding(
                rule_id="SC4",
                message=message,
                severity=severity,
                location=Location(file=file_path, start_line=line_num),
                confidence=confidence,
                tags=tag,
                matched_text=matched_text,
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
    locked_versions: dict[str, str] | None = None,
) -> list[AnalyzerFinding]:
    """Run SC4/SC5/SC6 checks on dependency files."""
    findings: list[AnalyzerFinding] = []
    tag = [PatternCategory.SUPPLY_CHAIN.value]

    lower_path = file_path.lower()
    is_lockfile = _is_python_lockfile(lower_path)
    is_python_dep = (
        any(n in lower_path for n in ["requirements", "pyproject.toml", "setup.py", "pipfile"])
        or is_lockfile
    )
    is_npm_dep = "package.json" in lower_path

    if not is_python_dep and not is_npm_dep:
        return findings

    if is_python_dep:
        if "pyproject.toml" in lower_path:
            packages = _extract_packages_from_pyproject(content)
        elif is_lockfile:
            packages = _extract_packages_from_toml_lock(content)
        else:
            packages = _extract_packages_from_requirements(content)
        if not is_lockfile:
            packages = _apply_locked_versions(packages, locked_versions)
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
# SC8: Shipped Python bytecode (closes silent __pycache__ / .pyc skip)
# ---------------------------------------------------------------------------

# Still skip heavy/vendor trees for SC8, but *do* descend into __pycache__.
_SC8_SKIP_DIRS = frozenset({".git", "node_modules", ".venv", "venv", ".tox", ".pytest_cache"})
_SC8_BYTECODE_SUFFIXES = (".pyc", ".pyo")


def _analyze_shipped_bytecode(skill_path: str) -> list[Finding]:
    """Emit SC8 when a skill ships __pycache__ dirs or .pyc/.pyo files.

    ``build_context`` excludes ``__pycache__`` from inventory and
    ``static_runner`` treats ``.pyc`` as binary, so malicious bytecode can
    otherwise score SAFE. Presence alone is a HIGH supply-chain signal;
    full disassembly can come later.
    """
    findings: list[Finding] = []
    if not skill_path or not isinstance(skill_path, str):
        return findings
    root = Path(skill_path)
    if not root.is_dir():
        return findings

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in _SC8_SKIP_DIRS)
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        if rel_dir == ".":
            rel_dir = ""

        for dirname in list(dirnames):
            if dirname != "__pycache__":
                continue
            rel = f"{rel_dir}/{dirname}/" if rel_dir else f"{dirname}/"
            af = AnalyzerFinding(
                rule_id="SC8",
                message="Skill ships a __pycache__ directory that normal discovery skips",
                severity=Severity.HIGH,
                location=Location(file=rel, start_line=1),
                confidence=0.95,
                tags=[PatternCategory.SUPPLY_CHAIN.value],
                matched_text=rel,
                context=(
                    "Python may load .pyc from this directory even when decoy "
                    ".py sources look clean (PEP 552 UNCHECKED_HASH)."
                ),
            )
            findings.append(analyzer_finding_to_finding(af))

        for filename in sorted(filenames):
            lower = filename.lower()
            if not lower.endswith(_SC8_BYTECODE_SUFFIXES):
                continue
            rel = f"{rel_dir}/{filename}" if rel_dir else filename
            af = AnalyzerFinding(
                rule_id="SC8",
                message="Skill ships Python bytecode (.pyc/.pyo) that normal analysis skips",
                severity=Severity.HIGH,
                location=Location(file=rel, start_line=1),
                confidence=0.95,
                tags=[PatternCategory.SUPPLY_CHAIN.value],
                matched_text=filename,
                context=(
                    "Bytecode is excluded from content analysis; a malicious "
                    ".pyc can execute while source decoys remain clean."
                ),
            )
            findings.append(analyzer_finding_to_finding(af))

    return findings


def _analyze_concealed_executables(
    component_metadata: list[dict[str, object]],
) -> list[Finding]:
    """Emit SC9 for executable content concealed in a local-only artifact."""
    findings: list[Finding] = []
    for metadata in component_metadata:
        if not metadata.get("concealed_executable"):
            continue
        path = str(metadata.get("path", ""))
        if not path:
            continue
        outer_path = str(metadata.get("outer_path", path.split("!/", 1)[0]))
        nested_path = str(
            metadata.get("nested_path", path.split("!/", 1)[1] if "!/" in path else path)
        )
        container_type = str(metadata.get("container_type", "zip"))
        raw_reasons = metadata.get("concealment_reasons", [])
        concealment_reasons = (
            [str(item) for item in raw_reasons] if isinstance(raw_reasons, list) else []
        )
        if not concealment_reasons:
            if container_type in {"docx", "xlsx", "pptx"}:
                concealment_reasons.append("document_container")
            elif metadata.get("outer_hidden") or metadata.get("hidden"):
                concealment_reasons.append("hidden_artifact")
            else:
                concealment_reasons.append("disguised_container")
        concealment = concealment_reasons[0]
        findings.append(
            Finding(
                rule_id="SC9",
                message=(
                    "Executable content is concealed inside a document, hidden, "
                    "or disguised artifact."
                ),
                severity="HIGH",
                confidence=1.0,
                file=path,
                start_line=1,
                category="Supply Chain",
                pattern="Concealed Executable Artifact",
                finding=nested_path,
                explanation=(
                    "An executable nested in a document or hidden/disguised artifact can "
                    "evade ordinary extension-based review while still being available to "
                    "the skill at runtime."
                ),
                remediation=(
                    "Review the artifact provenance and the reason executable content is "
                    "packaged in this location; keep executable files explicit and directly "
                    "reviewable."
                ),
                tags=["supply-chain", "concealed-executable", "local-only"],
                matched_text=path,
                evidence={
                    "outer_path": outer_path,
                    "nested_path": nested_path,
                    "container_type": container_type,
                    "container_ancestry": metadata.get("container_ancestry", [container_type]),
                    "container_depth": metadata.get("container_depth", 1),
                    "concealment": concealment,
                    "concealment_reasons": concealment_reasons,
                    "local_only": True,
                },
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Graph node
# ---------------------------------------------------------------------------


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run supply_chain patterns (SC1–SC10) and trigger analysis (TR1–TR3)."""
    # SC1–SC3 via static_runner
    response = static_runner.run_static_patterns_with_ledger(state, [sys.modules[__name__]])
    findings = response["findings"]

    def record_extra_findings(
        path: str,
        extra_findings: list[Finding],
        fallback_analyzer_id: str,
    ) -> None:
        """Attach supplemental findings to the matching completed work item."""
        if not extra_findings:
            return
        finding_ids = [finding.finding_id for finding in extra_findings]
        for event in response["inspection_ledger"]:
            if event["path"] == path and event["outcome"] is LedgerOutcome.COMPLETED:
                event["emitted_finding_ids"].extend(finding_ids)
                return
        response["inspection_ledger"].append(
            ledger_event(
                analyzer_id=fallback_analyzer_id,
                outcome=LedgerOutcome.COMPLETED,
                phase="static",
                path=path,
                emitted_finding_ids=finding_ids,
            )
        )

    # SC4–SC6: dependency-level analysis on dependency files
    components: list[str] = state.get("components") or []
    file_cache: dict[str, str] = state.get("local_file_cache") or state.get("file_cache") or {}
    locked_versions = _collect_locked_versions(file_cache, components)
    for path in components:
        lower_path = path.lower()
        is_dep_file = any(
            n in lower_path
            for n in [
                "requirements",
                "package.json",
                "pyproject.toml",
                "setup.py",
                "pipfile",
                "uv.lock",
                "poetry.lock",
            ]
        )
        if not is_dep_file:
            continue
        content = file_cache.get(path)
        if not content:
            continue
        dep_findings = _analyze_dependencies(content, path, locked_versions)
        dependency_findings = [analyzer_finding_to_finding(af) for af in dep_findings]
        findings.extend(dependency_findings)
        record_extra_findings(
            path,
            dependency_findings,
            f"{ANALYZER_ID}_dependencies",
        )

    # TR1–TR3: trigger analysis from manifest
    manifest: dict[str, object] = state.get("manifest") or {}
    if manifest:
        skill_path = state.get("skill_path") or ""
        trigger_findings = _analyze_triggers(manifest, skill_path)
        findings.extend(trigger_findings)
        record_extra_findings(
            "SKILL.md",
            trigger_findings,
            f"{ANALYZER_ID}_triggers",
        )

    # SC8: shipped bytecode / __pycache__ (discovery otherwise skips these)
    skill_path = state.get("skill_path") or ""
    if isinstance(skill_path, str) and skill_path.strip():
        bytecode_findings = _analyze_shipped_bytecode(skill_path)
        findings.extend(bytecode_findings)
        for finding_path in sorted({finding.file.rstrip("/") for finding in bytecode_findings}):
            record_extra_findings(
                finding_path,
                [
                    finding
                    for finding in bytecode_findings
                    if finding.file.rstrip("/") == finding_path
                ],
                f"{ANALYZER_ID}_bytecode",
            )

    # SC9: executables concealed in document containers or hidden/disguised artifacts.
    component_metadata: list[dict[str, object]] = state.get("component_metadata") or []
    concealed_findings = _analyze_concealed_executables(component_metadata)
    findings.extend(concealed_findings)
    for finding_path in sorted({finding.file for finding in concealed_findings}):
        record_extra_findings(
            finding_path,
            [finding for finding in concealed_findings if finding.file == finding_path],
            f"{ANALYZER_ID}_concealed_executable",
        )

    # SC10: deterministic dependency registry/source trust-boundary changes.
    dependency_source_findings = analyze_dependency_sources(
        components,
        file_cache,
        component_metadata,
    )
    findings.extend(dependency_source_findings)
    for finding_path in sorted({finding.file for finding in dependency_source_findings}):
        record_extra_findings(
            finding_path,
            [finding for finding in dependency_source_findings if finding.file == finding_path],
            f"{ANALYZER_ID}_dependency_source",
        )

    logger.info("%s: %d findings", ANALYZER_ID, len(findings))
    response["analyzer_status_events"] = [
        analyzer_status_for_events(ANALYZER_ID, response["inspection_ledger"])
    ]
    return response
