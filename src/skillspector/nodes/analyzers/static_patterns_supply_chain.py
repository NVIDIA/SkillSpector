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
SC10: Dependency source change — deterministic direct-config inspection.
TR1–TR3: Trigger analysis — flags overly broad, shadowing, or baiting triggers.

Node and analyze() in one module.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

from skillspector.dependency_source_types import (
    DependencySourceLimitation,
    DependencySourceLimitationReason,
    DependencyWorkBudget,
    ShellIssue,
    ShellIssueReason,
    ShellWorkItem,
    ShellWorkOutcome,
)
from skillspector.dependency_sources import analyze_dependency_sources
from skillspector.inspection_ledger import (
    MAX_FINDING_OUTPUT_RECORDS,
    AnalyzerStatusEvent,
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    LedgerRecordType,
    analyzer_status_event,
    analyzer_status_for_events,
    inspection_work_id,
    ledger_event,
)
from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Finding, Location, Severity
from skillspector.state import (
    AnalyzerNodeResponse,
    SkillspectorState,
    merge_inspection_ledger,
    transitive_note_truncation,
    transitive_remaining_seconds,
)

from . import static_runner
from .common import get_context, get_line_number
from .osv_client import (
    ECOSYSTEM_NPM,
    ECOSYSTEM_PYPI,
    OsvQueryBudget,
    OsvQueryLimitation,
    QueryBatchResults,
    VulnResult,
    query_batch,
    was_osv_reachable,
)
from .pattern_defaults import PatternCategory
from .static_runner import analyzer_finding_to_finding

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_supply_chain"
DEPENDENCY_SOURCE_ANALYZER_ID = "dependency_sources"
DEPENDENCY_SOURCE_SHELL_ANALYZER_ID = "dependency_source_shell"
DEPENDENCY_SOURCE_COVERAGE_ANALYZER_ID = "dependency_source_coverage"

_SHELL_OUTCOME_PRIORITY = {
    ShellWorkOutcome.COMPLETED: 0,
    ShellWorkOutcome.PARTIAL: 1,
    ShellWorkOutcome.SKIPPED: 2,
    ShellWorkOutcome.FAILED: 3,
}
_SHELL_REASON_PRIORITY = {
    ShellIssueReason.UNSUPPORTED_SEMANTICS: 0,
    ShellIssueReason.SYNTAX_ERROR: 1,
    ShellIssueReason.RUNTIME_LIMIT: 2,
    ShellIssueReason.RESOURCE_LIMIT: 3,
    ShellIssueReason.SHELL_PARSER_UNAVAILABLE: 4,
}
_SHELL_LEDGER_REASON = {
    ShellIssueReason.SYNTAX_ERROR: LedgerReason.DEPENDENCY_SOURCE_SHELL_SYNTAX_ERROR,
    ShellIssueReason.UNSUPPORTED_SEMANTICS: (LedgerReason.DEPENDENCY_SOURCE_UNSUPPORTED_SEMANTICS),
    ShellIssueReason.RUNTIME_LIMIT: LedgerReason.RUNTIME_LIMIT,
    ShellIssueReason.SHELL_PARSER_UNAVAILABLE: (
        LedgerReason.DEPENDENCY_SOURCE_SHELL_PARSER_UNAVAILABLE
    ),
    ShellIssueReason.RESOURCE_LIMIT: LedgerReason.DEPENDENCY_SOURCE_RESOURCE_LIMIT,
}


def _aggregate_shell_outcome(
    current: ShellWorkOutcome,
    incoming: ShellWorkOutcome,
) -> ShellWorkOutcome:
    """Retain productive partial work when a same-range sibling is skipped."""
    if {current, incoming} == {ShellWorkOutcome.PARTIAL, ShellWorkOutcome.SKIPPED}:
        return ShellWorkOutcome.PARTIAL
    return (
        incoming
        if _SHELL_OUTCOME_PRIORITY[incoming] > _SHELL_OUTCOME_PRIORITY[current]
        else current
    )


@dataclass(slots=True)
class _DependencyRowPlan:
    """One staged SC10 producer row before atomic public-output reservation."""

    analyzer_id: str
    outcome: LedgerOutcome
    path: str
    start_line: int
    end_line: int
    reason: LedgerReason | None = None
    emitted_finding_ids: list[str] | None = None
    unit_ids: set[str] | None = None
    metrics: dict[str, int] | None = None

    def event(self) -> InspectionLedgerEvent:
        metrics = self.metrics or {}
        return ledger_event(
            analyzer_id=self.analyzer_id,
            outcome=self.outcome,
            phase="static",
            path=self.path,
            start_line=self.start_line,
            end_line=self.end_line,
            reason=self.reason,
            emitted_finding_ids=self.emitted_finding_ids or (),
            observed_bytes=metrics.get("observed_bytes"),
            limit_bytes=metrics.get("limit_bytes"),
            observed_findings=metrics.get("observed_findings"),
            limit_findings=metrics.get("limit_findings"),
            observed_depth=metrics.get("observed_depth"),
            limit_depth=metrics.get("limit_depth"),
            observed_records=metrics.get("observed_records"),
            limit_records=metrics.get("limit_records"),
        )


def _clone_ledger_event(event: InspectionLedgerEvent) -> InspectionLedgerEvent:
    """Clone one row and its mutable finding-ID collections for staged mutation."""
    cloned = event.copy()
    cloned["input_finding_ids"] = list(event["input_finding_ids"])
    cloned["emitted_finding_ids"] = list(event["emitted_finding_ids"])
    return cloned


def _shell_dependency_row_plans(
    work_items: tuple[ShellWorkItem, ...],
    issues: tuple[ShellIssue, ...],
) -> tuple[list[_DependencyRowPlan], dict[str, _DependencyRowPlan]]:
    """Aggregate duplicate terminal identities and retain their worst outcome."""
    plans: dict[tuple[str, int, int], _DependencyRowPlan] = {}
    unit_keys: dict[str, set[tuple[str, int, int]]] = {}
    unit_item_counts: dict[str, int] = {}
    issues_by_key: dict[tuple[str, int, int], list[ShellIssue]] = {}

    for item in work_items:
        key = (item.span.path, item.span.start_line, item.span.end_line)
        unit_keys.setdefault(item.unit_id, set()).add(key)
        unit_item_counts[item.unit_id] = unit_item_counts.get(item.unit_id, 0) + 1
        outcome = LedgerOutcome(item.outcome.value)
        plan = plans.get(key)
        if plan is None:
            plans[key] = _DependencyRowPlan(
                analyzer_id=DEPENDENCY_SOURCE_SHELL_ANALYZER_ID,
                outcome=outcome,
                path=item.span.path,
                start_line=item.span.start_line,
                end_line=item.span.end_line,
                emitted_finding_ids=[],
                unit_ids={item.unit_id},
            )
        else:
            plan.outcome = LedgerOutcome(
                _aggregate_shell_outcome(
                    ShellWorkOutcome(plan.outcome.value),
                    item.outcome,
                ).value
            )
            if plan.unit_ids is None:
                plan.unit_ids = set()
            plan.unit_ids.add(item.unit_id)

    for issue in issues:
        issue_key: tuple[str, int, int]
        keys = unit_keys.get(issue.unit_id, set()) if issue.unit_id is not None else set()
        if len(keys) == 1:
            issue_key = next(iter(keys))
        else:
            issue_key = (issue.span.path, issue.span.start_line, issue.span.end_line)
        plan = plans.get(issue_key)
        if plan is None:
            plans[issue_key] = _DependencyRowPlan(
                analyzer_id=DEPENDENCY_SOURCE_SHELL_ANALYZER_ID,
                outcome=LedgerOutcome(issue.outcome.value),
                path=issue_key[0],
                start_line=issue_key[1],
                end_line=issue_key[2],
                emitted_finding_ids=[],
                unit_ids=({issue.unit_id} if issue.unit_id is not None else set()),
            )
        else:
            current = ShellWorkOutcome(plan.outcome.value)
            plan.outcome = LedgerOutcome(_aggregate_shell_outcome(current, issue.outcome).value)
        issues_by_key.setdefault(issue_key, []).append(issue)

    global_blocking_issues = [
        issue
        for issue in issues
        if issue.reason
        in {
            ShellIssueReason.RUNTIME_LIMIT,
            ShellIssueReason.RESOURCE_LIMIT,
            ShellIssueReason.SHELL_PARSER_UNAVAILABLE,
        }
    ]
    for key, plan in plans.items():
        if plan.outcome is LedgerOutcome.COMPLETED:
            continue
        candidates = issues_by_key.get(key, [])
        if not candidates and plan.outcome is LedgerOutcome.SKIPPED:
            candidates = global_blocking_issues
        selected = max(
            candidates,
            key=lambda issue: _SHELL_REASON_PRIORITY[issue.reason],
            default=None,
        )
        if selected is None:
            plan.reason = LedgerReason.DEPENDENCY_SOURCE_UNSUPPORTED_SEMANTICS
        else:
            plan.reason = _SHELL_LEDGER_REASON[selected.reason]
            if selected.exhaustion is not None:
                plan.metrics = selected.exhaustion.ledger_metrics()

    plan_by_unit_id: dict[str, _DependencyRowPlan] = {}
    for unit_id, keys in unit_keys.items():
        if len(keys) == 1 and unit_item_counts.get(unit_id) == 1:
            plan_by_unit_id[unit_id] = plans[next(iter(keys))]
    return list(plans.values()), plan_by_unit_id


# Dependency work is supplemental to the canonical text scan and therefore
# needs its own aggregate ceilings.  These apply across every manifest in a
# bundle, not independently per file.
MAX_DEPENDENCY_FILES_PER_SCAN = 64
MAX_DEPENDENCY_PACKAGES_PER_FILE = 256
MAX_DEPENDENCY_PACKAGES_PER_SCAN = 1_024
MAX_DEPENDENCY_FINDINGS_PER_FILE = 512
MAX_DEPENDENCY_FINDINGS_PER_SCAN = 2_048
MAX_DEPENDENCY_ANALYSIS_SECONDS = 30.0
MAX_DEPENDENCY_NAME_CHARS = 256
MAX_DEPENDENCY_VERSION_CHARS = 128
MAX_DEPENDENCY_SPEC_CHARS = 4_096

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
    for popular_name in sorted(popular):
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


def _logical_requirement_lines(content: str) -> Iterator[tuple[int, str]]:
    """Join pip-style continuations and retain each logical line's first line number."""
    parts: list[str] = []
    start_line = 1

    for line_num, raw_line in enumerate(io.StringIO(content), 1):
        line = raw_line.rstrip("\r\n")
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
        yield start_line, "".join(parts)
        parts = []

    if parts:
        yield start_line, "".join(parts)


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


def _extract_packages_from_requirements(
    content: str,
    *,
    limit: int | None = None,
) -> list[tuple[str, str | None, int]]:
    """Extract (package_name, version_or_None, line_number) from requirements.txt format."""
    results, _largest_omitted = _extract_packages_from_requirements_detailed(
        content,
        limit=limit,
    )
    return results


def _extract_packages_from_requirements_detailed(
    content: str,
    *,
    limit: int | None = None,
) -> tuple[list[tuple[str, str | None, int]], int | None]:
    """Extract bounded requirements and measure any oversized logical specifier."""
    results: list[tuple[str, str | None, int]] = []
    largest_omitted: int | None = None
    if limit is not None and limit <= 0:
        return results, largest_omitted
    for line_num, line in _logical_requirement_lines(content):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # pip treats a whitespace-prefixed ``#`` as an inline comment, while
        # PEP 508 parsing does not. Preserve normal requirements.txt behavior
        # before handing the complete requirement to ``packaging``.
        line = re.split(r"\s+#", line, maxsplit=1)[0]
        line = _strip_pip_per_requirement_options(line)
        if len(line) > MAX_DEPENDENCY_SPEC_CHARS:
            largest_omitted = max(largest_omitted or 0, len(line))
            continue
        requirement = _extract_python_requirement(line)
        if requirement:
            name, version = requirement
            results.append((name, version, line_num))
            if limit is not None and len(results) >= max(0, limit):
                break
    return results, largest_omitted


_NPM_DEPENDENCY_SECTIONS = ("dependencies", "devDependencies", "peerDependencies")


def _package_json_line(content: str, section: str, name: str) -> int:
    """Best-effort line for a dependency entry, so findings keep pointing somewhere useful.

    Parsing JSON loses positions, and the search starts at the section header so a name that
    also appears in ``scripts`` does not win.
    """
    return _package_json_lines(content, [(section, name)]).get((section, name), 1)


_JSON_OBJECT_KEY_RE = re.compile(r'"((?:\\.|[^"\\])*)"\s*:')


def _package_json_lines(
    content: str,
    requested: list[tuple[str, str]],
) -> dict[tuple[str, str], int]:
    """Locate dependency keys in one bounded pass instead of rescanning per package."""
    requested_by_name: dict[str, set[str]] = {}
    encoded_to_name: dict[str, str] = {}
    for section, name in requested:
        requested_by_name.setdefault(name, set()).add(section)
        encoded_to_name[json.dumps(name, ensure_ascii=True)[1:-1]] = name
    if not requested_by_name:
        return {}

    section_starts: dict[str, int] = {}
    for section in _NPM_DEPENDENCY_SECTIONS:
        header = re.search(rf'"{re.escape(section)}"\s*:', content)
        section_starts[section] = header.end() if header else 0

    positions: dict[tuple[str, str], int] = {}
    for match in _JSON_OBJECT_KEY_RE.finditer(content):
        matched_name = encoded_to_name.get(match.group(1))
        if matched_name is None:
            continue
        for section in requested_by_name[matched_name]:
            key = (section, matched_name)
            if key not in positions and match.start() >= section_starts.get(section, 0):
                positions[key] = match.start()
        if len(positions) >= len(requested):
            break

    ordered_positions = sorted((position, key) for key, position in positions.items())
    line_numbers: dict[tuple[str, str], int] = {}
    position_index = 0
    line_number = 1
    for newline in re.finditer("\n", content):
        while (
            position_index < len(ordered_positions)
            and ordered_positions[position_index][0] < newline.start()
        ):
            _position, key = ordered_positions[position_index]
            line_numbers[key] = line_number
            position_index += 1
        line_number += 1
    while position_index < len(ordered_positions):
        _position, key = ordered_positions[position_index]
        line_numbers[key] = line_number
        position_index += 1
    return line_numbers


def _extract_packages_from_package_json_scan(
    content: str,
    *,
    limit: int | None = None,
) -> list[tuple[str, str | None, int]]:
    """Line-oriented fallback, used only when the manifest is not valid JSON."""
    results: list[tuple[str, str | None, int]] = []
    if limit is not None and limit <= 0:
        return results
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
                if limit is not None and len(results) >= max(0, limit):
                    break
    return results


def _extract_packages_from_package_json(
    content: str,
    *,
    limit: int | None = None,
) -> list[tuple[str, str | None, int]]:
    """Extract (package_name, version_or_None, line_number) from package.json content.

    package.json is JSON, so it is parsed as JSON. Scanning it line by line made the result
    depend on formatting: a manifest written on a single line — which is valid, and what many
    generators emit — never entered the dependency section at all and yielded *no* dependencies,
    silently. The line-oriented scan remains as a fallback for manifests that do not parse.
    """
    if limit is not None and limit <= 0:
        return []
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        return _extract_packages_from_package_json_scan(content, limit=limit)
    if not isinstance(data, dict):
        return []
    dependencies: list[tuple[str, str, str]] = []
    for section in _NPM_DEPENDENCY_SECTIONS:
        deps = data.get(section)
        if not isinstance(deps, dict):
            continue
        for name, spec in deps.items():
            if not isinstance(name, str) or not isinstance(spec, str):
                continue
            dependencies.append((section, name, spec))
            if limit is not None and len(dependencies) >= max(0, limit):
                break
        if limit is not None and len(dependencies) >= max(0, limit):
            break
    line_numbers = _package_json_lines(
        content,
        [
            (section, name)
            for section, name, _spec in dependencies
            if len(name) <= MAX_DEPENDENCY_NAME_CHARS
        ],
    )
    return [
        (
            name,
            (_pinned_npm_version(spec) if len(spec) <= MAX_DEPENDENCY_SPEC_CHARS else None),
            line_numbers.get((section, name), 1),
        )
        for section, name, spec in dependencies
    ]


def _extract_packages_from_pyproject(
    content: str,
    *,
    limit: int | None = None,
) -> list[tuple[str, str | None, int]]:
    """Extract (package_name, version_or_None, line_number) from pyproject.toml.

    Reads PEP 621 ``[project]`` ``dependencies`` / ``optional-dependencies``,
    PEP 735 ``[dependency-groups]``, and ``[build-system].requires``. Standard
    metadata keys (``requires-python``, ``name``, ``version``, ...) are not
    dependencies and must not be looked up as packages.
    """
    if limit is not None and limit <= 0:
        return []
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return []

    specs: list[str] = []

    def extend_specs(values: object) -> None:
        if not isinstance(values, list):
            return
        for value in values:
            if limit is not None and len(specs) >= max(0, limit):
                return
            if isinstance(value, str):
                specs.append(value)
                if limit is not None and len(specs) >= max(0, limit):
                    return

    project = data.get("project")
    if isinstance(project, dict):
        extend_specs(project.get("dependencies"))
        optional = project.get("optional-dependencies")
        if isinstance(optional, dict):
            for group in optional.values():
                extend_specs(group)
                if limit is not None and len(specs) >= max(0, limit):
                    break
    groups = data.get("dependency-groups")
    if isinstance(groups, dict) and (limit is None or len(specs) < max(0, limit)):
        for group in groups.values():
            extend_specs(group)
            if limit is not None and len(specs) >= max(0, limit):
                break
    build_system = data.get("build-system")
    if isinstance(build_system, dict) and (limit is None or len(specs) < max(0, limit)):
        extend_specs(build_system.get("requires"))

    results: list[tuple[str, str | None, int]] = []
    for spec in specs:
        requirement = _extract_python_requirement(spec)
        if not requirement:
            continue
        name, version = requirement
        idx = content.find(spec)
        line_num = get_line_number(content, idx) if idx >= 0 else 1
        results.append((name, version, line_num))
        if limit is not None and len(results) >= max(0, limit):
            break
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


def _extract_packages_from_toml_lock(
    content: str,
    *,
    limit: int | None = None,
) -> list[tuple[str, str | None, int]]:
    """Extract exact package versions from TOML lockfiles such as uv.lock and poetry.lock."""
    if limit is not None and limit <= 0:
        return []
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return []
    packages = data.get("package")
    if not isinstance(packages, list):
        return []
    results: list[tuple[str, str | None, int]] = []
    blocks = _LOCKFILE_PACKAGE_BLOCK_RE.finditer(content)
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
        if limit is not None and len(results) >= max(0, limit):
            break
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
    *,
    limit: int = MAX_DEPENDENCY_PACKAGES_PER_SCAN,
) -> dict[str, str]:
    """Build package -> exact version map from Python lockfiles in the project."""
    locked_versions, _limitations = _collect_locked_versions_detailed(
        file_cache,
        components,
        limit=limit,
    )
    return locked_versions


def _collect_locked_versions_detailed(
    file_cache: dict[str, str],
    components: list[str],
    *,
    limit: int = MAX_DEPENDENCY_PACKAGES_PER_SCAN,
    max_files: int | None = None,
    timeout_seconds: float | None = None,
) -> tuple[dict[str, str], list[tuple[str, OsvQueryLimitation]]]:
    """Build a bounded lock map and identify any manifest whose tail was omitted."""
    locked_versions: dict[str, str] = {}
    limitations: list[tuple[str, OsvQueryLimitation]] = []
    packages_seen = 0
    lockfiles_seen = 0
    file_limit = MAX_DEPENDENCY_FILES_PER_SCAN if max_files is None else max(0, max_files)
    started_at = time.monotonic()
    runtime_limit = (
        MAX_DEPENDENCY_ANALYSIS_SECONDS
        if timeout_seconds is None
        else min(MAX_DEPENDENCY_ANALYSIS_SECONDS, max(0.0, timeout_seconds))
    )
    deadline = started_at + runtime_limit
    for path in components:
        if not _is_python_lockfile(path):
            continue
        lockfiles_seen += 1
        if lockfiles_seen > file_limit:
            limitations.append(
                (
                    path,
                    OsvQueryLimitation(
                        reason=LedgerReason.OUTPUT_LIMIT,
                        observed_records=lockfiles_seen,
                        limit_records=file_limit,
                    ),
                )
            )
            break
        now = time.monotonic()
        if now >= deadline:
            limitations.append(
                (
                    path,
                    OsvQueryLimitation(
                        reason=LedgerReason.RUNTIME_LIMIT,
                        observed_seconds=max(0.0, now - started_at),
                        limit_seconds=runtime_limit,
                    ),
                )
            )
            break
        content = file_cache.get(path)
        if not content:
            continue
        remaining = max(0, limit - packages_seen)
        if remaining <= 0:
            limitations.append(
                (
                    path,
                    OsvQueryLimitation(
                        reason=LedgerReason.OUTPUT_LIMIT,
                        observed_records=packages_seen + 1,
                        limit_records=max(0, limit),
                    ),
                )
            )
            break
        packages = _extract_packages_from_toml_lock(
            content,
            limit=remaining + 1,
        )
        if len(packages) > remaining:
            limitations.append(
                (
                    path,
                    OsvQueryLimitation(
                        reason=LedgerReason.OUTPUT_LIMIT,
                        observed_records=packages_seen + len(packages),
                        limit_records=max(0, limit),
                    ),
                )
            )
            packages = packages[:remaining]
        packages_seen += len(packages)
        for name, version, _line_num in packages:
            if version:
                locked_versions[_normalize_package_name(name)] = version
        if limitations:
            break
    return locked_versions, limitations


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
        return str(get_context(content, start))

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
    findings, covered, _limitations = _sc4_from_osv_detailed(
        packages,
        ecosystem,
        file_path,
        tag,
    )
    return findings, covered


def _sc4_from_osv_detailed(
    packages: list[tuple[str, str | None, int]],
    ecosystem: str,
    file_path: str,
    tag: list[str],
    *,
    timeout_seconds: float | None = None,
    budget: OsvQueryBudget | None = None,
) -> tuple[list[AnalyzerFinding], set[str], list[OsvQueryLimitation]]:
    """Run a bounded OSV lookup and retain its non-fatal limitation metadata."""
    pkg_pairs = [(name, version) for name, version, _ in packages]
    if budget is not None:
        osv_results = query_batch(pkg_pairs, ecosystem, budget=budget)
    elif timeout_seconds is not None:
        osv_results = query_batch(pkg_pairs, ecosystem, timeout_seconds=timeout_seconds)
    else:
        # Keep the two-argument call compatible with callers that replace the
        # OSV function with a small offline test/provider adapter.
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
    limitations = (
        list(osv_results.limitations) if isinstance(osv_results, QueryBatchResults) else []
    )
    return findings, covered, limitations


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
    findings, _limitations, _packages_seen = _analyze_dependencies_detailed(
        content,
        file_path,
        locked_versions,
    )
    return findings


def _analyze_dependencies_detailed(
    content: str,
    file_path: str,
    locked_versions: dict[str, str] | None = None,
    *,
    max_packages: int | None = None,
    max_findings: int | None = None,
    timeout_seconds: float | None = None,
    osv_budget: OsvQueryBudget | None = None,
) -> tuple[list[AnalyzerFinding], list[OsvQueryLimitation], int]:
    """Run bounded dependency checks and return sanitized omission metadata."""
    findings: list[AnalyzerFinding] = []
    limitations: list[OsvQueryLimitation] = []
    tag = [PatternCategory.SUPPLY_CHAIN.value]

    lower_path = file_path.lower()
    is_lockfile = _is_python_lockfile(lower_path)
    is_python_dep = (
        any(n in lower_path for n in ["requirements", "pyproject.toml", "setup.py", "pipfile"])
        or is_lockfile
    )
    is_npm_dep = "package.json" in lower_path

    if not is_python_dep and not is_npm_dep:
        return findings, limitations, 0

    requested_package_limit = (
        MAX_DEPENDENCY_PACKAGES_PER_FILE if max_packages is None else max_packages
    )
    package_limit = max(
        0,
        min(requested_package_limit, MAX_DEPENDENCY_PACKAGES_PER_FILE),
    )
    extraction_limit = package_limit + 1

    if is_python_dep:
        if "pyproject.toml" in lower_path:
            packages = _extract_packages_from_pyproject(content, limit=extraction_limit)
        elif is_lockfile:
            packages = _extract_packages_from_toml_lock(content, limit=extraction_limit)
        else:
            packages, oversized_spec = _extract_packages_from_requirements_detailed(
                content,
                limit=extraction_limit,
            )
            if oversized_spec is not None:
                limitations.append(
                    OsvQueryLimitation(
                        reason=LedgerReason.SIZE_LIMIT,
                        observed_characters=oversized_spec,
                        limit_characters=MAX_DEPENDENCY_SPEC_CHARS,
                    )
                )
        if not is_lockfile:
            packages = _apply_locked_versions(packages, locked_versions)
        ecosystem = ECOSYSTEM_PYPI
        fallback_db = _FALLBACK_VULNERABLE_PYPI
        popular = _POPULAR_PYPI
    else:
        packages = _extract_packages_from_package_json(content, limit=extraction_limit)
        ecosystem = ECOSYSTEM_NPM
        fallback_db = _FALLBACK_VULNERABLE_NPM
        popular = _POPULAR_NPM

    if len(packages) > package_limit:
        limitations.append(
            OsvQueryLimitation(
                reason=LedgerReason.OUTPUT_LIMIT,
                observed_records=len(packages),
                limit_records=package_limit,
            )
        )
        packages = packages[:package_limit]
    parsed_package_count = len(packages)

    bounded_packages: list[tuple[str, str | None, int]] = []
    for name, version, line_num in packages:
        if len(name) > MAX_DEPENDENCY_NAME_CHARS:
            limitations.append(
                OsvQueryLimitation(
                    reason=LedgerReason.SIZE_LIMIT,
                    observed_characters=len(name),
                    limit_characters=MAX_DEPENDENCY_NAME_CHARS,
                )
            )
            continue
        if version is not None and len(version) > MAX_DEPENDENCY_VERSION_CHARS:
            limitations.append(
                OsvQueryLimitation(
                    reason=LedgerReason.SIZE_LIMIT,
                    observed_characters=len(version),
                    limit_characters=MAX_DEPENDENCY_VERSION_CHARS,
                )
            )
            continue
        bounded_packages.append((name, version, line_num))
    packages = bounded_packages

    requested_finding_limit = (
        MAX_DEPENDENCY_FINDINGS_PER_FILE if max_findings is None else max_findings
    )
    finding_limit = max(
        0,
        min(requested_finding_limit, MAX_DEPENDENCY_FINDINGS_PER_FILE),
    )

    def retain(extra: list[AnalyzerFinding]) -> None:
        remaining = max(0, finding_limit - len(findings))
        if len(extra) > remaining:
            limitations.append(
                OsvQueryLimitation(
                    reason=LedgerReason.OUTPUT_LIMIT,
                    observed_records=len(findings) + len(extra),
                    limit_records=finding_limit,
                )
            )
        findings.extend(extra[:remaining])

    # SC4: Live OSV.dev lookup, then static fallback for uncovered packages
    osv_findings, osv_covered, osv_limitations = _sc4_from_osv_detailed(
        packages,
        ecosystem,
        file_path,
        tag,
        timeout_seconds=timeout_seconds,
        budget=osv_budget,
    )
    limitations.extend(osv_limitations)
    retain(osv_findings)
    uncovered_packages = [p for p in packages if p[0].lower().replace("_", "-") not in osv_covered]
    fallback_findings = _sc4_from_fallback(uncovered_packages, fallback_db, file_path, tag)
    if fallback_findings:
        logger.debug(
            "SC4: using static fallback for %d uncovered packages", len(uncovered_packages)
        )
    elif uncovered_packages and not osv_findings and not was_osv_reachable():
        # OSV.dev was unreachable and fallback found nothing — surface the gap
        retain(
            [
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
            ]
        )
    retain(fallback_findings)

    for pkg_name, _pkg_version, line_num in packages:
        pkg_lower = pkg_name.lower().replace("_", "-")

        # SC5: Abandoned dependencies
        if pkg_lower in {a.lower().replace("_", "-") for a in _ABANDONED_PACKAGES}:
            retain(
                [
                    AnalyzerFinding(
                        rule_id="SC5",
                        message=f"Abandoned Dependency: {pkg_name} is unmaintained and no longer receives security updates",
                        severity=Severity.MEDIUM,
                        location=Location(file=file_path, start_line=line_num),
                        confidence=0.75,
                        tags=tag,
                        matched_text=pkg_name,
                    )
                ]
            )

        # SC6: Typosquatting
        similar = _is_typosquat(pkg_name, popular)
        if similar:
            retain(
                [
                    AnalyzerFinding(
                        rule_id="SC6",
                        message=f"Possible Typosquatting: '{pkg_name}' resembles popular package '{similar}'",
                        severity=Severity.HIGH,
                        location=Location(file=file_path, start_line=line_num),
                        confidence=0.7,
                        tags=tag,
                        matched_text=pkg_name,
                    )
                ]
            )

    # Do not let repeated provider conditions create unbounded metadata.
    unique_limitations: list[OsvQueryLimitation] = []
    for limitation in limitations:
        if limitation not in unique_limitations:
            unique_limitations.append(limitation)
        if len(unique_limitations) >= 16:
            break
    return findings, unique_limitations, parsed_package_count


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
MAX_SC8_DISCOVERED_ENTRIES = 10_000
MAX_SC8_DIRECTORY_ENTRIES = 10_000
MAX_SC8_TRAVERSAL_DEPTH = 64
MAX_SC8_ANALYSIS_SECONDS = 5.0
MAX_SC8_FINDINGS = 10_000
MAX_SC8_LIMITATIONS = 256


@dataclass(frozen=True)
class _SupplementalLimitation:
    """One bounded, report-safe supplemental omission."""

    path: str
    reason: LedgerReason
    observed_artifacts: int | None = None
    limit_artifacts: int | None = None
    observed_depth: int | None = None
    limit_depth: int | None = None
    observed_findings: int | None = None
    limit_findings: int | None = None
    observed_seconds: float | None = None
    limit_seconds: float | None = None
    error_class: str | None = None


@dataclass(frozen=True)
class _ShippedBytecodeScanResult:
    findings: list[Finding]
    limitations: list[_SupplementalLimitation]


def _scan_shipped_bytecode(
    skill_path: str,
    *,
    timeout_seconds: float | None = None,
    max_findings: int | None = None,
) -> _ShippedBytecodeScanResult:
    """Discover shipped bytecode with deterministic aggregate resource bounds."""
    findings: list[Finding] = []
    limitations: list[_SupplementalLimitation] = []
    if not skill_path or not isinstance(skill_path, str):
        return _ShippedBytecodeScanResult(findings, limitations)
    root = Path(skill_path)
    if not root.is_dir():
        return _ShippedBytecodeScanResult(findings, limitations)

    started_at = time.monotonic()
    runtime_limit = max(0.0, MAX_SC8_ANALYSIS_SECONDS)
    if timeout_seconds is not None:
        runtime_limit = min(runtime_limit, max(0.0, timeout_seconds))
    deadline = started_at + runtime_limit
    requested_finding_limit = MAX_SC8_FINDINGS if max_findings is None else max_findings
    finding_limit = max(0, min(requested_finding_limit, MAX_SC8_FINDINGS))
    discovered_entries = 0
    stack: list[tuple[Path, str, int]] = [(root, "", 0)]
    stop_scan = False

    def scope_path(relative_directory: str) -> str:
        return relative_directory.rstrip("/") or "SKILL.md"

    def add_limitation(limitation: _SupplementalLimitation) -> None:
        if limitation in limitations:
            return
        if len(limitations) < max(1, MAX_SC8_LIMITATIONS):
            limitations.append(limitation)
            return
        limitations[-1] = _SupplementalLimitation(
            path=scope_path(""),
            reason=LedgerReason.OUTPUT_LIMIT,
            observed_findings=len(limitations) + 1,
            limit_findings=max(1, MAX_SC8_LIMITATIONS),
        )

    def runtime_exhausted(relative_directory: str) -> bool:
        now = time.monotonic()
        if now < deadline:
            return False
        add_limitation(
            _SupplementalLimitation(
                path=scope_path(relative_directory),
                reason=LedgerReason.RUNTIME_LIMIT,
                observed_seconds=max(0.0, now - started_at),
                limit_seconds=runtime_limit,
            )
        )
        return True

    def add_finding(relative_path: str, *, directory: bool) -> bool:
        nonlocal stop_scan
        if len(findings) >= finding_limit:
            add_limitation(
                _SupplementalLimitation(
                    path=relative_path.rstrip("/"),
                    reason=LedgerReason.OUTPUT_LIMIT,
                    observed_findings=len(findings) + 1,
                    limit_findings=finding_limit,
                )
            )
            stop_scan = True
            return False
        if directory:
            analyzer_finding = AnalyzerFinding(
                rule_id="SC8",
                message="Skill ships a __pycache__ directory that normal discovery skips",
                severity=Severity.HIGH,
                location=Location(file=relative_path, start_line=1),
                confidence=0.95,
                tags=[PatternCategory.SUPPLY_CHAIN.value],
                matched_text=relative_path,
                context=(
                    "Python may load .pyc from this directory even when decoy "
                    ".py sources look clean (PEP 552 UNCHECKED_HASH)."
                ),
            )
        else:
            analyzer_finding = AnalyzerFinding(
                rule_id="SC8",
                message="Skill ships Python bytecode (.pyc/.pyo) that normal analysis skips",
                severity=Severity.HIGH,
                location=Location(file=relative_path, start_line=1),
                confidence=0.95,
                tags=[PatternCategory.SUPPLY_CHAIN.value],
                matched_text=Path(relative_path).name,
                context=(
                    "Bytecode is excluded from content analysis; a malicious "
                    ".pyc can execute while source decoys remain clean."
                ),
            )
        findings.append(analyzer_finding_to_finding(analyzer_finding))
        return True

    while stack and not stop_scan:
        directory, relative_directory, depth = stack.pop()
        if runtime_exhausted(relative_directory):
            break
        remaining_entries = max(0, MAX_SC8_DISCOVERED_ENTRIES - discovered_entries)
        directory_limit = min(max(0, MAX_SC8_DIRECTORY_ENTRIES), remaining_entries)
        if directory_limit <= 0:
            add_limitation(
                _SupplementalLimitation(
                    path=scope_path(relative_directory),
                    reason=LedgerReason.ARTIFACT_COUNT_LIMIT,
                    observed_artifacts=discovered_entries + 1,
                    limit_artifacts=max(0, MAX_SC8_DISCOVERED_ENTRIES),
                )
            )
            break

        entries: list[tuple[str, bool, bool, bool, str | None]] = []
        directory_overflow = False
        try:
            with os.scandir(directory) as scanner:
                for entry in scanner:
                    if runtime_exhausted(relative_directory):
                        directory_overflow = True
                        break
                    if len(entries) >= directory_limit:
                        add_limitation(
                            _SupplementalLimitation(
                                path=scope_path(relative_directory),
                                reason=LedgerReason.ARTIFACT_COUNT_LIMIT,
                                observed_artifacts=discovered_entries + len(entries) + 1,
                                limit_artifacts=min(
                                    max(0, MAX_SC8_DISCOVERED_ENTRIES),
                                    discovered_entries + max(0, MAX_SC8_DIRECTORY_ENTRIES),
                                ),
                            )
                        )
                        directory_overflow = True
                        break
                    try:
                        is_link = entry.is_symlink()
                        is_directory = entry.is_dir(follow_symlinks=False)
                        is_file = entry.is_file(follow_symlinks=False)
                        error_class = None
                    except OSError as exc:
                        is_link = False
                        is_directory = False
                        is_file = False
                        error_class = type(exc).__name__
                    entries.append((entry.name, is_directory, is_file, is_link, error_class))
        except OSError as exc:
            add_limitation(
                _SupplementalLimitation(
                    path=scope_path(relative_directory),
                    reason=LedgerReason.READ_ERROR,
                    error_class=type(exc).__name__,
                )
            )
            continue
        if directory_overflow:
            break

        child_directories: list[tuple[Path, str, int]] = []
        for name, is_directory, is_file, is_link, error_class in sorted(
            entries, key=lambda item: item[0]
        ):
            if runtime_exhausted(relative_directory):
                stop_scan = True
                break
            discovered_entries += 1
            relative_path = f"{relative_directory}/{name}" if relative_directory else name
            if error_class is not None:
                add_limitation(
                    _SupplementalLimitation(
                        path=relative_path,
                        reason=LedgerReason.STAT_ERROR,
                        error_class=error_class,
                    )
                )
                continue
            if is_link:
                continue
            if is_directory:
                if name == "__pycache__" and not add_finding(f"{relative_path}/", directory=True):
                    break
                if name in _SC8_SKIP_DIRS:
                    continue
                child_depth = depth + 1
                if child_depth > max(0, MAX_SC8_TRAVERSAL_DEPTH):
                    add_limitation(
                        _SupplementalLimitation(
                            path=relative_path,
                            reason=LedgerReason.TRAVERSAL_DEPTH_LIMIT,
                            observed_depth=child_depth,
                            limit_depth=max(0, MAX_SC8_TRAVERSAL_DEPTH),
                        )
                    )
                    continue
                child_directories.append((directory / name, relative_path, child_depth))
                continue
            if is_file and name.lower().endswith(_SC8_BYTECODE_SUFFIXES):
                if not add_finding(relative_path, directory=False):
                    break
        stack.extend(reversed(child_directories))

    return _ShippedBytecodeScanResult(findings, limitations)


def _analyze_shipped_bytecode(skill_path: str) -> list[Finding]:
    """Emit SC8 when a skill ships __pycache__ dirs or .pyc/.pyo files.

    ``build_context`` excludes ``__pycache__`` from inventory and
    ``static_runner`` treats ``.pyc`` as binary, so malicious bytecode can
    otherwise score SAFE. Presence alone is a HIGH supply-chain signal;
    full disassembly can come later.
    """
    return _scan_shipped_bytecode(skill_path).findings


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
    completed_event_by_path = {
        event["path"]: event
        for event in response["inspection_ledger"]
        if event["outcome"] is LedgerOutcome.COMPLETED
    }
    recorded_limitations: set[tuple[str, str, LedgerReason]] = set()

    def record_extra_findings(
        path: str,
        extra_findings: list[Finding],
        fallback_analyzer_id: str,
    ) -> None:
        """Attach supplemental findings to the matching completed work item."""
        if not extra_findings:
            return
        finding_ids = [finding.finding_id for finding in extra_findings]
        event = completed_event_by_path.get(path)
        if event is not None:
            event["emitted_finding_ids"].extend(finding_ids)
            return
        event = ledger_event(
            analyzer_id=fallback_analyzer_id,
            outcome=LedgerOutcome.COMPLETED,
            phase="static",
            path=path,
            emitted_finding_ids=finding_ids,
        )
        response["inspection_ledger"].append(event)
        completed_event_by_path[path] = event

    def record_limitation(
        path: str,
        limitation: OsvQueryLimitation | _SupplementalLimitation,
        fallback_analyzer_id: str,
    ) -> None:
        """Project one supplemental omission into canonical partial accounting."""
        key = (path, fallback_analyzer_id, limitation.reason)
        if key in recorded_limitations:
            return
        recorded_limitations.add(key)
        response["inspection_ledger"].append(
            ledger_event(
                analyzer_id=f"{fallback_analyzer_id}_{limitation.reason.value}",
                outcome=LedgerOutcome.PARTIAL,
                record_type=LedgerRecordType.SYSTEM,
                phase="static",
                path=path,
                reason=limitation.reason,
                error_class=limitation.error_class,
                observed_records=getattr(limitation, "observed_records", None),
                limit_records=getattr(limitation, "limit_records", None),
                observed_characters=getattr(limitation, "observed_characters", None),
                limit_characters=getattr(limitation, "limit_characters", None),
                observed_bytes=getattr(limitation, "observed_bytes", None),
                limit_bytes=getattr(limitation, "limit_bytes", None),
                observed_artifacts=getattr(limitation, "observed_artifacts", None),
                limit_artifacts=getattr(limitation, "limit_artifacts", None),
                observed_depth=getattr(limitation, "observed_depth", None),
                limit_depth=getattr(limitation, "limit_depth", None),
                observed_findings=getattr(limitation, "observed_findings", None),
                limit_findings=getattr(limitation, "limit_findings", None),
                observed_seconds=limitation.observed_seconds,
                limit_seconds=limitation.limit_seconds,
            )
        )
        transitive_note_truncation(
            state,
            f"{fallback_analyzer_id} incomplete: {limitation.reason.value}",
        )

    # SC4–SC6: dependency-level analysis on dependency files
    components: list[str] = state.get("components") or []
    file_cache: dict[str, str] = state.get("local_file_cache") or state.get("file_cache") or {}
    dependency_started_at = time.monotonic()
    workflow_remaining = transitive_remaining_seconds(state)
    dependency_runtime_limit = max(0.0, MAX_DEPENDENCY_ANALYSIS_SECONDS)
    if workflow_remaining is not None:
        dependency_runtime_limit = min(
            dependency_runtime_limit,
            max(0.0, workflow_remaining),
        )
    dependency_deadline = dependency_started_at + dependency_runtime_limit

    def dependency_remaining_seconds() -> float:
        local_remaining = max(0.0, dependency_deadline - time.monotonic())
        shared_remaining = transitive_remaining_seconds(state)
        return (
            local_remaining
            if shared_remaining is None
            else min(local_remaining, max(0.0, shared_remaining))
        )

    locked_versions, lockfile_limitations = _collect_locked_versions_detailed(
        file_cache,
        components,
        limit=MAX_DEPENDENCY_PACKAGES_PER_SCAN,
        max_files=MAX_DEPENDENCY_FILES_PER_SCAN,
        timeout_seconds=dependency_remaining_seconds(),
    )
    for lockfile_path, limitation in lockfile_limitations:
        record_limitation(
            lockfile_path,
            limitation,
            f"{ANALYZER_ID}_dependencies",
        )
    dependency_files_seen = 0
    dependency_packages_seen = 0
    dependency_findings_seen = 0
    osv_budget = OsvQueryBudget.create(dependency_remaining_seconds())
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
        dependency_files_seen += 1
        if dependency_files_seen > max(0, MAX_DEPENDENCY_FILES_PER_SCAN):
            record_limitation(
                path,
                OsvQueryLimitation(
                    reason=LedgerReason.OUTPUT_LIMIT,
                    observed_records=dependency_files_seen,
                    limit_records=max(0, MAX_DEPENDENCY_FILES_PER_SCAN),
                ),
                f"{ANALYZER_ID}_dependencies",
            )
            break
        remaining_packages = max(
            0,
            MAX_DEPENDENCY_PACKAGES_PER_SCAN - dependency_packages_seen,
        )
        remaining_dependency_findings = max(
            0,
            min(
                MAX_DEPENDENCY_FINDINGS_PER_SCAN - dependency_findings_seen,
                MAX_FINDING_OUTPUT_RECORDS - len(findings),
            ),
        )
        if remaining_packages <= 0 or remaining_dependency_findings <= 0:
            record_limitation(
                path,
                OsvQueryLimitation(
                    reason=LedgerReason.OUTPUT_LIMIT,
                    observed_records=(
                        dependency_packages_seen + 1
                        if remaining_packages <= 0
                        else dependency_findings_seen + 1
                    ),
                    limit_records=(
                        MAX_DEPENDENCY_PACKAGES_PER_SCAN
                        if remaining_packages <= 0
                        else MAX_DEPENDENCY_FINDINGS_PER_SCAN
                    ),
                ),
                f"{ANALYZER_ID}_dependencies",
            )
            break
        shared_remaining = dependency_remaining_seconds()
        if shared_remaining <= 0:
            record_limitation(
                path,
                OsvQueryLimitation(
                    reason=LedgerReason.RUNTIME_LIMIT,
                    observed_seconds=max(0.0, time.monotonic() - dependency_started_at),
                    limit_seconds=dependency_runtime_limit,
                ),
                f"{ANALYZER_ID}_dependencies",
            )
            break
        content = file_cache.get(path)
        if not content:
            continue
        dep_findings, dependency_limitations, packages_seen = _analyze_dependencies_detailed(
            content,
            path,
            locked_versions,
            max_packages=min(MAX_DEPENDENCY_PACKAGES_PER_FILE, remaining_packages),
            max_findings=min(MAX_DEPENDENCY_FINDINGS_PER_FILE, remaining_dependency_findings),
            timeout_seconds=shared_remaining,
            osv_budget=osv_budget,
        )
        dependency_packages_seen += packages_seen
        dependency_findings = [analyzer_finding_to_finding(af) for af in dep_findings]
        dependency_findings_seen += len(dependency_findings)
        findings.extend(dependency_findings)
        record_extra_findings(
            path,
            dependency_findings,
            f"{ANALYZER_ID}_dependencies",
        )
        for limitation in dependency_limitations:
            record_limitation(
                path,
                limitation,
                f"{ANALYZER_ID}_dependencies",
            )

    # TR1–TR3: trigger analysis from manifest
    manifest: dict[str, object] = state.get("manifest") or {}
    if manifest:
        skill_path = state.get("skill_path") or ""
        trigger_findings = _analyze_triggers(manifest, skill_path)
        trigger_limit = max(0, MAX_FINDING_OUTPUT_RECORDS - len(findings))
        omitted_triggers = len(trigger_findings) > trigger_limit
        trigger_findings = trigger_findings[:trigger_limit]
        findings.extend(trigger_findings)
        record_extra_findings(
            "SKILL.md",
            trigger_findings,
            f"{ANALYZER_ID}_triggers",
        )
        if omitted_triggers:
            record_limitation(
                "SKILL.md",
                OsvQueryLimitation(
                    reason=LedgerReason.OUTPUT_LIMIT,
                    observed_records=len(trigger_findings) + 1,
                    limit_records=trigger_limit,
                ),
                f"{ANALYZER_ID}_triggers",
            )

    # SC8: shipped bytecode / __pycache__ (discovery otherwise skips these)
    skill_path = state.get("skill_path") or ""
    if isinstance(skill_path, str) and skill_path.strip():
        bytecode_scan = _scan_shipped_bytecode(
            skill_path,
            timeout_seconds=transitive_remaining_seconds(state),
            max_findings=max(0, MAX_FINDING_OUTPUT_RECORDS - len(findings)),
        )
        bytecode_findings = bytecode_scan.findings
        findings.extend(bytecode_findings)
        findings_by_path: dict[str, list[Finding]] = {}
        for finding in bytecode_findings:
            findings_by_path.setdefault(finding.file.rstrip("/"), []).append(finding)
        for finding_path in sorted(findings_by_path):
            record_extra_findings(
                finding_path,
                findings_by_path[finding_path],
                f"{ANALYZER_ID}_bytecode",
            )
        for sc8_limitation in bytecode_scan.limitations:
            record_limitation(
                sc8_limitation.path,
                sc8_limitation,
                f"{ANALYZER_ID}_bytecode",
            )

    # SC9: executables concealed in document containers or hidden/disguised artifacts.
    component_metadata: list[dict[str, object]] = state.get("component_metadata") or []
    concealed_findings = _analyze_concealed_executables(component_metadata)
    concealed_limit = max(0, MAX_FINDING_OUTPUT_RECORDS - len(findings))
    omitted_concealed = len(concealed_findings) > concealed_limit
    concealed_findings = concealed_findings[:concealed_limit]
    findings.extend(concealed_findings)
    concealed_by_path: dict[str, list[Finding]] = {}
    for finding in concealed_findings:
        concealed_by_path.setdefault(finding.file, []).append(finding)
    for finding_path in sorted(concealed_by_path):
        record_extra_findings(
            finding_path,
            concealed_by_path[finding_path],
            f"{ANALYZER_ID}_concealed_executable",
        )
    if omitted_concealed:
        record_limitation(
            "SKILL.md",
            OsvQueryLimitation(
                reason=LedgerReason.OUTPUT_LIMIT,
                observed_records=len(concealed_findings) + 1,
                limit_records=concealed_limit,
            ),
            f"{ANALYZER_ID}_concealed_executable",
        )

    # SC10: deterministic direct configuration plus explicit executable-surface gaps.
    # Only the normalized executable bit crosses the inventory boundary; no metadata
    # payload is passed to dependency-source parsing or projected into output.
    executable_path_values: set[str] = set()
    for metadata in component_metadata:
        metadata_path = metadata.get("path")
        if (
            metadata.get("executable") is True
            and isinstance(metadata_path, str)
            and metadata_path in components
        ):
            executable_path_values.add(metadata_path)
    executable_paths = frozenset(executable_path_values)
    dependency_source_budget = DependencyWorkBudget.from_existing(
        findings=findings,
        ledger_events=response["inspection_ledger"],
    )
    source_analysis = analyze_dependency_sources(
        components=components,
        local_file_cache=file_cache,
        raw_file_cache=state.get("raw_file_cache") or {},
        artifact_inventory=state.get("artifact_inventory") or [],
        budget=dependency_source_budget,
        executable_paths=executable_paths,
        deadline_monotonic=dependency_deadline,
    )
    parse_limitations_by_path: dict[str, list[DependencySourceLimitation]] = {}
    coverage_limitations: list[DependencySourceLimitation] = []
    for source_limitation in source_analysis.limitations:
        if (
            source_limitation.reason
            is DependencySourceLimitationReason.UNSCANNED_EXECUTABLE_CONTENT
        ):
            coverage_limitations.append(source_limitation)
        else:
            parse_limitations_by_path.setdefault(source_limitation.path, []).append(
                source_limitation
            )

    source_plans: list[_DependencyRowPlan] = []
    direct_plan_by_path: dict[str, _DependencyRowPlan] = {}
    for span in source_analysis.applicable_spans:
        path_limitations = parse_limitations_by_path.get(span.path, [])
        if path_limitations:
            source_limitation = path_limitations[0]
            plan = _DependencyRowPlan(
                analyzer_id=DEPENDENCY_SOURCE_ANALYZER_ID,
                outcome=LedgerOutcome.PARTIAL,
                path=span.path,
                start_line=min(item.start_line for item in path_limitations),
                end_line=max(item.end_line for item in path_limitations),
                reason=LedgerReason.DEPENDENCY_SOURCE_PARSE_INCOMPLETE,
                emitted_finding_ids=[],
                metrics=source_limitation.ledger_metrics(),
            )
        else:
            plan = _DependencyRowPlan(
                analyzer_id=DEPENDENCY_SOURCE_ANALYZER_ID,
                outcome=LedgerOutcome.COMPLETED,
                path=span.path,
                start_line=span.start_line,
                end_line=span.end_line,
                emitted_finding_ids=[],
            )
        if span.path in direct_plan_by_path:
            raise ValueError("dependency-source path has multiple direct producer rows")
        direct_plan_by_path[span.path] = plan
        source_plans.append(plan)

    shell_plans, shell_plan_by_unit_id = _shell_dependency_row_plans(
        source_analysis.shell_work_items,
        source_analysis.shell_issues,
    )
    coverage_plans: list[_DependencyRowPlan] = [
        _DependencyRowPlan(
            analyzer_id=DEPENDENCY_SOURCE_COVERAGE_ANALYZER_ID,
            outcome=LedgerOutcome.PARTIAL,
            path=source_limitation.path,
            start_line=source_limitation.start_line,
            end_line=source_limitation.end_line,
            reason=LedgerReason.UNSCANNED_EXECUTABLE_CONTENT,
            emitted_finding_ids=[],
        )
        for source_limitation in coverage_limitations
    ]

    for finding, producer_unit_id in zip(
        source_analysis.findings,
        source_analysis.finding_producer_unit_ids,
        strict=True,
    ):
        producer_plan = (
            direct_plan_by_path.get(finding.file)
            if producer_unit_id is None
            else shell_plan_by_unit_id.get(producer_unit_id)
        )
        if producer_plan is None:
            raise ValueError("dependency-source finding has no unique producer row")
        if producer_plan.outcome not in {LedgerOutcome.COMPLETED, LedgerOutcome.PARTIAL}:
            raise ValueError("non-producing dependency work cannot attach a finding")
        if producer_plan.emitted_finding_ids is None:
            producer_plan.emitted_finding_ids = []
        if finding.finding_id not in producer_plan.emitted_finding_ids:
            producer_plan.emitted_finding_ids.append(finding.finding_id)

    all_plans = [*source_plans, *shell_plans, *coverage_plans]
    planned_events: list[InspectionLedgerEvent] = [plan.event() for plan in all_plans]
    planned_work_ids = [event["work_id"] for event in planned_events]
    if len(planned_work_ids) != len(set(planned_work_ids)):
        raise ValueError("dependency-source work identities must be unique before commit")

    base_ledger: list[InspectionLedgerEvent] = list(response["inspection_ledger"])
    staged_base_ledger: list[InspectionLedgerEvent] = [
        _clone_ledger_event(event) for event in base_ledger
    ]
    base_positions: dict[str, int] = {}
    for index, event in enumerate(staged_base_ledger):
        work_id = event["work_id"]
        if work_id in base_positions:
            raise ValueError("base ledger contains a duplicate dependency producer identity")
        base_positions[work_id] = index

    new_rows: list[InspectionLedgerEvent] = []
    for event in planned_events:
        existing_index = base_positions.get(event["work_id"])
        if existing_index is None:
            new_rows.append(event)
            continue
        existing = staged_base_ledger[existing_index]
        if (
            existing.get("analyzer_id") != event.get("analyzer_id")
            or existing["path"] != event["path"]
            or existing["start_line"] != event["start_line"]
            or existing["end_line"] != event["end_line"]
            or existing["outcome"] is not event["outcome"]
        ):
            raise ValueError("existing dependency producer row conflicts with planned work")
        for finding_id in event["emitted_finding_ids"]:
            if finding_id not in existing["emitted_finding_ids"]:
                existing["emitted_finding_ids"].append(finding_id)

    staged_ledger: list[InspectionLedgerEvent] = [*staged_base_ledger, *new_rows]
    for finding in source_analysis.findings:
        attachment_count = sum(
            finding.finding_id in event["emitted_finding_ids"] for event in staged_ledger
        )
        if attachment_count != 1:
            raise ValueError("dependency-source finding must attach to exactly one producer")

    existing_finding_ids = {finding.finding_id for finding in findings}
    new_source_findings = [
        finding
        for finding in source_analysis.findings
        if finding.finding_id not in existing_finding_ids
    ]
    exhaustion = dependency_source_budget.reserve_dependency_outputs(
        emitted_changes=len(new_source_findings),
        finding_output_records=len(new_source_findings),
        new_ledger_events=len(new_rows),
    )

    def output_limited_status(
        analyzer_id: str,
        plans: list[_DependencyRowPlan],
    ) -> AnalyzerStatusEvent:
        if not plans:
            return analyzer_status_for_events(analyzer_id, [])
        return analyzer_status_event(
            analyzer_id=analyzer_id,
            status="degraded",
            reason=LedgerReason.OUTPUT_LIMIT,
            planned_work=[
                {
                    "work_id": inspection_work_id(
                        analyzer_id,
                        plan.path,
                        plan.start_line,
                        plan.end_line,
                    ),
                    "path": plan.path,
                    "start_line": plan.start_line,
                    "end_line": plan.end_line,
                }
                for plan in plans
            ],
        )

    if exhaustion is None:
        findings = [*findings, *new_source_findings]
        response["findings"] = findings
        response["inspection_ledger"] = staged_ledger
        committed_by_id = {event["work_id"]: event for event in staged_ledger}

        def committed_events(plans: list[_DependencyRowPlan]) -> list[InspectionLedgerEvent]:
            return [committed_by_id[plan.event()["work_id"]] for plan in plans]

        source_status = analyzer_status_for_events(
            DEPENDENCY_SOURCE_ANALYZER_ID,
            committed_events(source_plans),
        )
        shell_status = analyzer_status_for_events(
            DEPENDENCY_SOURCE_SHELL_ANALYZER_ID,
            committed_events(shell_plans),
        )
        coverage_status = analyzer_status_for_events(
            DEPENDENCY_SOURCE_COVERAGE_ANALYZER_ID,
            committed_events(coverage_plans),
        )
    else:
        dependency_source_budget.claim_reserved_truncation_event()
        omitted_path = all_plans[0].path if all_plans else "SKILL.md"
        marker = ledger_event(
            outcome=LedgerOutcome.PARTIAL,
            record_type=LedgerRecordType.SYSTEM,
            phase="ledger_output",
            path=omitted_path,
            reason=LedgerReason.OUTPUT_LIMIT,
            observed_records=exhaustion.observed,
            limit_records=exhaustion.limit,
        )
        response["inspection_ledger"] = merge_inspection_ledger(base_ledger, [marker])
        source_status = output_limited_status(
            DEPENDENCY_SOURCE_ANALYZER_ID,
            source_plans,
        )
        shell_status = output_limited_status(
            DEPENDENCY_SOURCE_SHELL_ANALYZER_ID,
            shell_plans,
        )
        coverage_status = output_limited_status(
            DEPENDENCY_SOURCE_COVERAGE_ANALYZER_ID,
            coverage_plans,
        )

    logger.info("%s: %d findings", ANALYZER_ID, len(findings))
    response["analyzer_status_events"] = [
        analyzer_status_for_events(ANALYZER_ID, base_ledger),
        source_status,
        shell_status,
        coverage_status,
    ]
    return response
