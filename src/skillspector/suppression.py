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

"""Baseline / false-positive suppression for SkillSpector.

A *baseline* is a YAML (or JSON) file that tells the report node which findings
to drop before scoring and reporting. It supports two complementary mechanisms:

* ``rules`` — human-authored, glob-based suppressions for the root scan. A finding is suppressed
  when every field a rule specifies (``id``, ``path``, ``message``) glob-matches
  the finding. ``message`` covers both the analyzer description and the matched
  text surfaced as ``finding`` in reports. Unspecified fields match anything.
  This covers both global pattern suppression (e.g. ``id: "SQP-1"``) and
  skill/file-scoped suppression (e.g. ``id: "SSD-2"`` +
  ``path: "deploy-topology-execute-scripts/SKILL.md"``).

* ``fingerprints`` — machine-generated exact suppressions. Each entry is the
  stable hash of one known finding, so re-scans only surface *new* findings.
  Generate these with ``skillspector baseline <path>`` for incremental CI use.

Example baseline::

    version: 2
    scanner_version: "X.Y.Z"
    rules:
      - id: "SQP-1"
        reason: "Trigger-phrase breadth is a description nit, not a vuln"
      - id: "SSD-2"
        path: "*deploy-topology*/SKILL.md"
        message: "*run the exploit*"
        reason: "False positive: 'run the exploit' is a lab test-workflow phrase"
    fingerprints:
      - hash: "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        rule_id: "SDI-2"
        file: "baas-build-analysis/SKILL.md"
        reason: "Accepted 2026-06-19 — first-party env detection"

Glob semantics use :func:`fnmatch.fnmatch`, where ``*`` matches across path
separators (so ``*SKILL.md`` matches ``a/b/SKILL.md``); ``**`` is accepted as a
friendly alias for ``*``. Message globs are matched case-insensitively, so wrap
a keyword in ``*`` (e.g. ``"*telemetry*"``) for substring matching.
"""

from __future__ import annotations

import errno
import fnmatch
import hashlib
import json
import os
import posixpath
import re
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from stat import S_IMODE, S_ISREG
from typing import Any

import yaml

from skillspector.logging_config import get_logger
from skillspector.models import Finding

logger = get_logger(__name__)

BASELINE_VERSION = 2
MAX_BASELINE_BYTES = 2 * 1024 * 1024
MAX_BASELINE_NODES = 100_000
MAX_BASELINE_DEPTH = 64
MAX_BASELINE_RECORDS = 10_000
MAX_BASELINE_SCALAR_CHARS = 64 * 1024
# Validation passes for an output path that concurrent writers replace.
_BASELINE_DESTINATION_ATTEMPTS = 2
_FINGERPRINT_SCHEMA = "skillspector-finding-fingerprint-v2"
_FINGERPRINT_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SOURCE_IDENTITY_RE = re.compile(r"external/[0-9a-f]{64}\Z")


class _BoundedBaselineLoader(yaml.SafeLoader):
    """Bound composition and alias expansion before YAML object construction."""

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._nodes = 0
        self._depth = 0

    def compose_node(self, parent: object, index: object) -> yaml.Node:
        self._nodes += 1
        if self._nodes > MAX_BASELINE_NODES:
            raise ValueError("baseline exceeds YAML node limit")
        if self._depth >= MAX_BASELINE_DEPTH:
            raise ValueError("baseline exceeds YAML depth limit")
        self._depth += 1
        try:
            return super().compose_node(parent, index)
        finally:
            self._depth -= 1

    def construct_document(self, node: yaml.Node) -> Any:
        nodes = 0
        characters = 0
        active: set[int] = set()

        def visit(item: yaml.Node, depth: int) -> None:
            nonlocal nodes, characters
            nodes += 1
            if nodes > MAX_BASELINE_NODES:
                raise ValueError("baseline exceeds expanded YAML node limit")
            if depth > MAX_BASELINE_DEPTH:
                raise ValueError("baseline exceeds expanded YAML depth limit")
            if id(item) in active:
                raise ValueError("baseline contains cyclic YAML aliases")
            if isinstance(item, yaml.ScalarNode):
                if len(item.value) > MAX_BASELINE_SCALAR_CHARS:
                    raise ValueError("baseline exceeds YAML scalar character limit")
                characters += len(item.value)
                if characters > MAX_BASELINE_BYTES:
                    raise ValueError("baseline exceeds expanded YAML character limit")
                return
            active.add(id(item))
            children = (
                (child for pair in item.value for child in pair)
                if isinstance(item, yaml.MappingNode)
                else iter(item.value)
            )
            for child in children:
                visit(child, depth + 1)
            active.remove(id(item))

        # Revisit each alias: counting only unique nodes misses merge amplification.
        visit(node, 1)
        return super().construct_document(node)


def _has_exact_source_provenance(finding: Finding) -> bool:
    """Return whether immutable transitive provenance has canonical form."""
    return bool(
        finding.source_identity
        and _SOURCE_IDENTITY_RE.fullmatch(finding.source_identity)
        and finding.source_digest
        and _FINGERPRINT_RE.fullmatch(finding.source_digest)
    )


def _match_glob(value: str, pattern: str) -> bool:
    """Case-insensitive glob match; ``**`` is treated as an alias for ``*``.

    Patterns use :func:`fnmatch.fnmatch` semantics, so ``*``, ``?`` and ``[...]``
    are treated as glob metacharacters. Rule ids and the messages we match are
    plain text in practice, but if you ever need to match one of those characters
    literally, escape it with :func:`fnmatch.translate` / ``[`` brackets rather
    than relying on literal matching here.
    """
    normalized = pattern.replace("**", "*")
    return fnmatch.fnmatch(value.lower(), normalized.lower())


def _normalize_component_path(path: str) -> str:
    """Return a stable slash-separated relative component path."""
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return posixpath.normpath(normalized)


def _component_content(
    file_cache: Mapping[str, str],
    file_path: str,
    *,
    source_identity: str | None = None,
    source_url: str | None = None,
) -> str | None:
    """Look up *file_path* without crossing source-scope boundaries."""
    source_scope = source_identity or source_url
    if source_scope:
        normalized_path = _normalize_component_path(file_path)
        scoped_candidates = (
            f"{source_scope}::{file_path}",
            f"{source_scope}::{normalized_path}",
            f"{source_scope.rstrip('/')}/{normalized_path}",
        )
        for source_key in scoped_candidates:
            if source_key in file_cache:
                return file_cache[source_key]
        # A transitive finding must never borrow a same-named root or sibling
        # component when its own immutable source cache entry is unavailable.
        return None
    if file_path in file_cache:
        return file_cache[file_path]
    normalized = _normalize_component_path(file_path)
    for candidate, content in file_cache.items():
        if _normalize_component_path(candidate) == normalized:
            return content
    return None


def finding_fingerprint(
    finding: Finding,
    *,
    file_content: str | None = None,
    scanner_version: str | None = None,
) -> str:
    """Return an evidence-bound v2 fingerprint for *finding*.

    Exact suppressions bind to the complete scanned component, scanner version,
    finding identity, severity, location, and emitted evidence.  Canonical JSON
    avoids delimiter ambiguity and the full SHA-256 digest avoids the legacy
    64-bit truncation.  Any source or scanner change therefore requires review
    and baseline regeneration.
    """
    if not isinstance(file_content, str):
        raise ValueError("file_content is required to create an exact baseline fingerprint")
    if not isinstance(scanner_version, str) or not scanner_version.strip():
        raise ValueError("scanner_version is required to create an exact baseline fingerprint")

    payload = {
        "schema": _FINGERPRINT_SCHEMA,
        "scanner_version": scanner_version.strip(),
        "component": {
            "path": _normalize_component_path(finding.file or ""),
            "sha256": hashlib.sha256(file_content.encode("utf-8")).hexdigest(),
        },
        "finding": {
            "rule_id": finding.rule_id or "",
            "severity": finding.severity or "",
            "confidence": finding.confidence,
            "start_line": finding.start_line,
            "end_line": finding.end_line,
            "category": (finding.category or "").strip(),
            "message": (finding.message or "").strip(),
            "pattern": (finding.pattern or "").strip(),
            "matched_text": (finding.matched_text or "").strip(),
            "finding": (finding.finding or "").strip(),
            "explanation": (finding.explanation or "").strip(),
            "remediation": (finding.remediation or "").strip(),
            "intent": (finding.intent or "").strip(),
            "tags": sorted(finding.tags),
            "context": finding.context or "",
            "code_snippet": finding.code_snippet or "",
        },
    }
    if (
        finding.source_identity
        or finding.source_digest
        or finding.source_url
        or finding.transitive_depth
    ):
        payload["source"] = {
            "identity": finding.source_identity or "",
            "digest": finding.source_digest or "",
            "url": (
                finding.source_url
                if not finding.source_identity and not finding.source_digest
                else ""
            )
            or "",
            "depth": finding.transitive_depth,
        }
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True)
class SuppressionRule:
    """A glob-based suppression rule. Empty rules (no field set) never match."""

    rule_id: str | None = None
    path: str | None = None
    message: str | None = None
    reason: str = ""

    def matches(self, finding: Finding) -> bool:
        """True when every field this rule specifies glob-matches *finding*."""
        if self.rule_id is None and self.path is None and self.message is None:
            return False  # guard: an all-wildcard rule would suppress everything
        if self.rule_id is not None and not _match_glob(finding.rule_id or "", self.rule_id):
            return False
        if self.path is not None and not _match_glob(finding.file or "", self.path):
            return False
        if self.message is not None:
            message_candidates = (
                finding.message or "",
                finding.finding or "",
                finding.matched_text or "",
            )
            if not any(_match_glob(candidate, self.message) for candidate in message_candidates):
                return False
        return True


@dataclass(frozen=True)
class SuppressedFinding:
    """A finding paired with the reason it was suppressed."""

    finding: Finding
    reason: str

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form: the full finding plus its suppression reason."""
        data = self.finding.to_dict()
        data["suppressed"] = True
        data["suppression_reason"] = self.reason
        return data


@dataclass
class Baseline:
    """Loaded baseline: glob rules plus exact fingerprint suppressions."""

    rules: list[SuppressionRule] = field(default_factory=list)
    fingerprints: dict[str, str] = field(default_factory=dict)  # hash -> reason
    scanner_version: str | None = None

    def reason_for(
        self,
        finding: Finding,
        *,
        file_content: str | None = None,
        scanner_version: str | None = None,
    ) -> str | None:
        """Return the suppression reason for *finding*, or None if not suppressed."""
        is_transitive = bool(
            finding.source_identity
            or finding.source_digest
            or finding.source_url
            or finding.transitive_depth
        )
        # Root-authored globs are intentionally never inherited by dependencies.
        # A transitive finding needs an exact fingerprint bound to both its opaque
        # source identity and the immutable digest of the inspected source.
        if not is_transitive:
            for rule in self.rules:
                if rule.matches(finding):
                    return rule.reason or "matched suppression rule"
        elif not _has_exact_source_provenance(finding):
            return None
        if (
            file_content is None
            or not scanner_version
            or not self.scanner_version
            or scanner_version != self.scanner_version
        ):
            return None
        fp = finding_fingerprint(
            finding,
            file_content=file_content,
            scanner_version=scanner_version,
        )
        if fp in self.fingerprints:
            return self.fingerprints[fp] or "matched baseline fingerprint"
        return None

    def is_empty(self) -> bool:
        """True when the baseline has no rules and no fingerprints."""
        return not self.rules and not self.fingerprints


def baseline_from_dict(data: dict[str, Any]) -> Baseline:
    """Build a :class:`Baseline` from a parsed mapping (YAML/JSON)."""
    if not isinstance(data, dict):
        raise ValueError(f"baseline must be a mapping (got {type(data).__name__})")

    version = data.get("version")
    raw_rules = data.get("rules")
    raw_fingerprints = data.get("fingerprints")
    if raw_rules is None:
        raw_rules = []
    if raw_fingerprints is None:
        raw_fingerprints = []
    if not isinstance(raw_rules, list) or not isinstance(raw_fingerprints, list):
        raise ValueError("baseline rules and fingerprints must be lists")
    if len(raw_rules) + len(raw_fingerprints) > MAX_BASELINE_RECORDS:
        raise ValueError(f"baseline exceeds record limit ({MAX_BASELINE_RECORDS})")
    is_legacy_rule_only = version in (None, 1) and not raw_fingerprints
    if version != BASELINE_VERSION and not is_legacy_rule_only:
        migration = (
            " Version 1 fingerprints cannot be trusted because they did not bind to finding "
            "evidence; rescan and re-triage with `skillspector baseline`."
            if version in (None, 1)
            else ""
        )
        raise ValueError(
            f"unsupported baseline version {version!r}; expected {BASELINE_VERSION}.{migration}"
        )
    if is_legacy_rule_only:
        logger.warning(
            "Loading legacy rule-only baseline version %r; regenerate it as version %s",
            version,
            BASELINE_VERSION,
        )

    rules: list[SuppressionRule] = []
    for raw in raw_rules:
        if not isinstance(raw, dict):
            raise ValueError(f"each baseline rule must be a mapping, got: {raw!r}")
        reason = raw.get("reason", "")
        if version == BASELINE_VERSION and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError("each v2 suppression rule must have a non-empty reason")
        rule = SuppressionRule(
            rule_id=raw.get("id") or raw.get("rule_id"),
            path=raw.get("path") or raw.get("file"),
            message=raw.get("message"),
            reason=reason.strip() if isinstance(reason, str) else "",
        )
        if rule.rule_id is None and rule.path is None and rule.message is None:
            raise ValueError(
                "a baseline rule must set at least one of: id, path, message "
                f"(offending rule: {raw!r})"
            )
        rules.append(rule)

    fingerprints: dict[str, str] = {}
    for raw in raw_fingerprints:
        if not isinstance(raw, dict) or not raw.get("hash"):
            raise ValueError(
                "each v2 fingerprint must be a mapping with 'hash' and non-empty 'reason'"
            )
        fingerprint = str(raw["hash"])
        if _FINGERPRINT_RE.fullmatch(fingerprint) is None:
            raise ValueError(f"invalid v2 fingerprint hash: {fingerprint!r}")
        reason = raw.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("each v2 fingerprint must have a non-empty reason")
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate baseline fingerprint: {fingerprint}")
        fingerprints[fingerprint] = reason.strip()

    scanner_version = data.get("scanner_version")
    if fingerprints and (not isinstance(scanner_version, str) or not scanner_version.strip()):
        raise ValueError("a v2 baseline with fingerprints must set scanner_version")

    return Baseline(
        rules=rules,
        fingerprints=fingerprints,
        scanner_version=scanner_version.strip() if isinstance(scanner_version, str) else None,
    )


def load_baseline(path: str | Path) -> Baseline:
    """Load a baseline file (YAML or JSON) into a :class:`Baseline`.

    Raises FileNotFoundError if *path* is missing, ValueError if it is malformed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Baseline file not found: {p}")
    with open(
        p,
        "rb",
        opener=lambda path, flags: os.open(path, flags | getattr(os, "O_NONBLOCK", 0)),
    ) as source:
        if not S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError(f"Baseline must be a regular file: {p}")
        content = source.read(MAX_BASELINE_BYTES + 1)
    if len(content) > MAX_BASELINE_BYTES:
        raise ValueError(f"Baseline file exceeds byte limit ({MAX_BASELINE_BYTES}): {p}")
    try:
        # SafeLoader parses JSON too, so a single path handles both formats.
        data = yaml.load(content.decode("utf-8"), Loader=_BoundedBaselineLoader) or {}
    except yaml.YAMLError as e:  # pragma: no cover - error path
        raise ValueError(f"Could not parse baseline file {p}: {e}") from e
    return baseline_from_dict(data)


SHIPPED_BASELINE_FILENAME = ".skillspector-baseline.yaml"


def discover_baseline(skill_dir: str | Path) -> Path | None:
    """Return the baseline shipped at the top level of *skill_dir*, or None.

    Pure existence check for the single canonical filename
    (``.skillspector-baseline.yaml``, the name ``skillspector baseline`` writes
    by default). The file is never read here, so an untrusted shipped baseline
    is not parsed until the caller decides to load it. Nested files are ignored;
    per-sub-skill discovery belongs to the recursive path and is out of scope.
    ``.yml`` / ``.json`` baselines stay usable through explicit ``--baseline``.
    """
    candidate = Path(skill_dir) / SHIPPED_BASELINE_FILENAME
    return candidate if candidate.is_file() else None


def partition_findings(
    findings: list[Finding],
    baseline: Baseline | None,
    *,
    file_cache: Mapping[str, str] | None = None,
    scanner_version: str | None = None,
) -> tuple[list[Finding], list[SuppressedFinding]]:
    """Split *findings* into (kept, suppressed) using *baseline*.

    With no baseline, everything is kept. Suppressed findings never count toward
    the risk score and are excluded from the SARIF results.
    """
    if baseline is None or baseline.is_empty():
        return list(findings), []
    kept: list[Finding] = []
    suppressed: list[SuppressedFinding] = []
    cache = file_cache or {}
    if baseline.fingerprints and baseline.scanner_version != scanner_version:
        logger.warning(
            "Baseline scanner version %r does not match current version %r; exact "
            "fingerprints will not suppress findings",
            baseline.scanner_version,
            scanner_version,
        )
    for finding in findings:
        reason = baseline.reason_for(
            finding,
            file_content=_component_content(
                cache,
                finding.file or "",
                source_identity=finding.source_identity,
                source_url=finding.source_url,
            ),
            scanner_version=scanner_version,
        )
        if reason is None:
            kept.append(finding)
        else:
            suppressed.append(SuppressedFinding(finding=finding, reason=reason))
    if suppressed:
        logger.debug("Suppressed %d finding(s) via baseline", len(suppressed))
    return kept, suppressed


def effective_findings(result: Mapping[str, object]) -> list[Finding]:
    """Return the findings from a graph *result* that actually drove its risk score.

    The report node returns ``filtered_findings`` as the full pre-partition set
    (kept plus baseline-suppressed) alongside ``suppressed_findings``, but scores
    and SARIF results from the kept subset alone. Consumers that want the numbers
    the report itself published must therefore subtract the suppressed partition.

    Two failure modes this exists to prevent, both of which over-report:

    * ``result.get("filtered_findings") or result.get("findings")`` treats an
      empty filtered list as absent and falls back to the raw pre-filter
      findings. An empty list is a real answer -- every finding was filtered out
      or suppressed -- not a missing one.
    * Using ``filtered_findings`` directly counts baseline-suppressed findings
      that the report excluded from the score, so a fully suppressed skill
      reports risk 0 alongside a non-zero finding count.

    Falls back to the raw ``findings`` list only when ``filtered_findings`` is
    absent or malformed, and does not subtract there: raw findings are not the
    population that produced ``suppressed_findings``.
    """
    filtered = result.get("filtered_findings")
    if not isinstance(filtered, list):
        raw = result.get("findings")
        return list(raw) if isinstance(raw, list) else []

    suppressed = result.get("suppressed_findings")
    if not isinstance(suppressed, list) or not suppressed:
        return list(filtered)

    suppressed_ids = {
        entry.finding.finding_id
        for entry in suppressed
        if isinstance(entry, SuppressedFinding) and entry.finding is not None
    }
    return [
        finding
        for finding in filtered
        if not isinstance(finding, Finding) or finding.finding_id not in suppressed_ids
    ]


def build_baseline_dict(
    findings: list[Finding],
    reason: str = "Accepted finding (auto-generated baseline)",
    *,
    file_cache: Mapping[str, str] | None = None,
    scanner_version: str | None = None,
) -> dict[str, object]:
    """Build a baseline mapping that fingerprint-suppresses every given finding."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("baseline fingerprint reason must be non-empty")
    if not isinstance(scanner_version, str) or not scanner_version.strip():
        raise ValueError("scanner_version is required to build a baseline")
    if file_cache is None:
        raise ValueError("file_cache is required to build an exact baseline")

    entries: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    for finding in findings:
        is_transitive = bool(
            finding.source_identity
            or finding.source_digest
            or finding.source_url
            or finding.transitive_depth
        )
        if is_transitive and not _has_exact_source_provenance(finding):
            raise ValueError(
                "cannot create an exact transitive fingerprint without canonical "
                "source_identity and source_digest"
            )
        content = _component_content(
            file_cache,
            finding.file or "",
            source_identity=finding.source_identity,
            source_url=finding.source_url,
        )
        if content is None:
            raise ValueError(
                f"cannot create an exact fingerprint: source content missing for {finding.file!r}"
            )
        fingerprint = finding_fingerprint(
            finding,
            file_content=content,
            scanner_version=scanner_version,
        )
        if fingerprint in seen_hashes:
            continue
        seen_hashes.add(fingerprint)
        entries.append(
            {
                "hash": fingerprint,
                "rule_id": finding.rule_id,
                "file": finding.file,
                "reason": reason.strip(),
                **({"source_identity": finding.source_identity} if finding.source_identity else {}),
                **({"source_digest": finding.source_digest} if finding.source_digest else {}),
            }
        )

    return {
        "version": BASELINE_VERSION,
        "scanner_version": scanner_version.strip(),
        "rules": [],
        "fingerprints": entries,
    }


def _restrict_baseline_temporary(descriptor: int) -> None:
    """Remove inherited access before a temporary file receives baseline data."""
    if os.name == "posix":
        # Also masks named-user/group ACL grants on POSIX ACL implementations.
        os.fchmod(descriptor, 0o600)
    if sys.platform != "darwin":
        return

    # macOS extended ACL grants are independent of permission bits. Use the
    # already-open descriptor so clearing them cannot follow a swapped path.
    import ctypes

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        init_acl = libc.acl_init
        set_acl = libc.acl_set_fd_np
        free_acl = libc.acl_free
    except AttributeError:
        # Filesystems/platforms without extended ACL support need only mode bits.
        return
    init_acl.argtypes = [ctypes.c_int]
    init_acl.restype = ctypes.c_void_p
    set_acl.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    set_acl.restype = ctypes.c_int
    free_acl.argtypes = [ctypes.c_void_p]
    free_acl.restype = ctypes.c_int
    empty_acl = init_acl(0)
    if not empty_acl:
        raise OSError(ctypes.get_errno(), "Could not initialize baseline ACL")
    try:
        if set_acl(descriptor, empty_acl, 0x100) != 0:  # ACL_TYPE_EXTENDED
            error = ctypes.get_errno()
            if error not in {errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise OSError(error, "Could not clear inherited baseline ACLs")
    finally:
        free_acl(empty_acl)


def _preserve_baseline_acl(source: int, destination: int) -> None:
    """Copy an existing access ACL through descriptors before publication."""
    if sys.platform == "darwin":
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        get_acl = libc.acl_get_fd_np
        set_acl = libc.acl_set_fd_np
        free_acl = libc.acl_free
        get_acl.argtypes = [ctypes.c_int, ctypes.c_int]
        get_acl.restype = ctypes.c_void_p
        set_acl.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        set_acl.restype = ctypes.c_int
        free_acl.argtypes = [ctypes.c_void_p]
        free_acl.restype = ctypes.c_int
        acl = get_acl(source, 0x100)
        if not acl:
            error = ctypes.get_errno()
            if error in {errno.ENOENT, errno.ENOTSUP, errno.EOPNOTSUPP}:
                return
            raise OSError(error, "Could not read existing baseline ACL")
        try:
            if set_acl(destination, acl, 0x100) != 0:
                raise OSError(ctypes.get_errno(), "Could not preserve existing baseline ACL")
        finally:
            free_acl(acl)
    elif sys.platform.startswith("linux"):
        try:
            acl = os.getxattr(source, "system.posix_acl_access")
        except OSError as error:
            if error.errno in {errno.ENODATA, errno.ENOTSUP, errno.EOPNOTSUPP}:
                return
            raise
        os.setxattr(destination, "system.posix_acl_access", acl)


def _write_baseline_in_place(
    descriptor: int, encoded: bytes, p: Path, opened: os.stat_result
) -> bool:
    """Rewrite a validated inode in place, keeping its owner, group, mode and ACLs.

    Returns False, without writing, when *p* no longer names *opened*.
    """
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_EX)
    current = p.lstat()
    if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
        return False
    remaining = memoryview(encoded)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError(errno.EIO, "Could not write shared baseline", str(p))
        remaining = remaining[written:]
    os.ftruncate(descriptor, len(encoded))
    os.fsync(descriptor)
    return True


def dump_baseline(data: dict[str, object], path: str | Path) -> None:
    """Validate and write a regular baseline (``.json`` -> JSON).

    On POSIX, new files have owner-only permissions. Replacements preserve
    ownership and ordinary permission bits; the old file must be writable.
    Non-owner writers, and owners that cannot assign the destination's group,
    update the validated descriptor in place, preserving its permissions and
    ACLs without requiring chown. That shared-file path is not atomic for
    readers or crash-safe. Symlinks and special files are rejected.
    """
    baseline_from_dict(data)
    p = Path(path)
    if p.suffix.lower() == ".json":
        # PyYAML does not combine JSON's escaped UTF-16 surrogate pairs. Emit
        # astral characters directly, escaping only genuine lone surrogates.
        content = (
            json.dumps(data, indent=2, ensure_ascii=False)
            .encode("utf-8", errors="backslashreplace")
            .decode("utf-8")
        )
        # JSON permits these raw characters, but YAML rejects most of them,
        # folds NEL into a space, and treats LS/PS as line breaks that strip
        # adjacent spaces or start a "---" document marker. The loader reads
        # both formats through PyYAML.
        content = re.sub(
            r"[\x7f-\x9f\u2028\u2029\ufffe\uffff]",
            lambda match: f"\\u{ord(match[0]):04x}",
            content,
        )
    else:
        header = (
            "# SkillSpector baseline — findings listed here are suppressed on future scans.\n"
            "# Edit 'reason' fields and add glob 'rules' as needed. See docs/SUPPRESSION.md.\n"
        )
        content = header + yaml.safe_dump(data, sort_keys=False)
    # A complete population can exceed the loader's limits even when a compact
    # report fits. Reject it before overwriting an existing, usable baseline.
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_BASELINE_BYTES:
        raise ValueError(f"Baseline file exceeds byte limit ({MAX_BASELINE_BYTES}): {p}")
    yaml.load(content, Loader=_BoundedBaselineLoader)

    destination = None
    access_descriptor = None
    # A cooperating writer can atomically replace the path between validating
    # and opening or locking it. Validate the replacement from the start, but
    # only a bounded number of times; a path that keeps changing fails closed.
    for attempt in range(1, _BASELINE_DESTINATION_ATTEMPTS + 1):
        final_attempt = attempt == _BASELINE_DESTINATION_ATTEMPTS
        try:
            destination = p.lstat()
        except FileNotFoundError:
            destination = None
            break
        if not S_ISREG(destination.st_mode):
            raise ValueError(f"Baseline output must be a regular file: {p}")
        if os.name == "posix" and os.geteuid() == 0 and not destination.st_mode & 0o222:
            raise PermissionError(errno.EACCES, "Baseline output is not writable", str(p))
        # The descriptor check handles ACL grants that mode bits omit. Root
        # still observes the explicit read-only mode guard above.
        flags = os.O_WRONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(p, flags)
        try:
            opened = os.fstat(descriptor)
            if not S_ISREG(opened.st_mode):
                raise ValueError(f"Baseline output must be a regular file: {p}")
            if (opened.st_dev, opened.st_ino) != (destination.st_dev, destination.st_ino):
                if final_attempt:
                    raise ValueError(f"Baseline output changed while opening: {p}")
                continue
            destination = opened
            if os.name == "posix" and os.geteuid() not in {0, destination.st_uid}:
                # Replacing somebody else's writable file would require chown
                # and would discard its ACLs. Serialize cooperating shared-file
                # writers and keep this already validated inode instead.
                if _write_baseline_in_place(descriptor, encoded, p, opened):
                    return
                if final_attempt:
                    raise ValueError(f"Baseline output changed before writing: {p}")
                continue
            # Retain the validated inode's access metadata while competing
            # atomic writers replace the path. There is no need to reopen it.
            access_descriptor = os.dup(descriptor)
            break
        finally:
            os.close(descriptor)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=p.parent, prefix=".skillspector-baseline.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            _restrict_baseline_temporary(temporary.fileno())
            if destination is not None:
                current = os.fstat(temporary.fileno())
                if (current.st_uid, current.st_gid) != (destination.st_uid, destination.st_gid):
                    try:
                        os.fchown(temporary.fileno(), destination.st_uid, destination.st_gid)
                    except PermissionError:
                        if access_descriptor is None:
                            raise
                        # A non-root owner cannot assign a group it is not a
                        # member of. Rewrite the validated inode, keeping its group.
                        if not _write_baseline_in_place(access_descriptor, encoded, p, destination):
                            raise ValueError(
                                f"Baseline output changed before writing: {p}"
                            ) from None
                        return
                # Keep existing group writers/readers. Newly generated files
                # remain private; replacing one does not revoke shared access.
                mode = S_IMODE(destination.st_mode) & 0o777
                if os.name == "posix":
                    os.fchmod(temporary.fileno(), mode)
                    # Clearing inherited ACLs must not remove a restrictive or
                    # shared access ACL from the existing destination.
                    assert access_descriptor is not None
                    _preserve_baseline_acl(access_descriptor, temporary.fileno())
                else:
                    os.chmod(temporary_path, mode)
            # Configure all destination access metadata while the file is empty.
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, p)
    finally:
        if access_descriptor is not None:
            os.close(access_descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
