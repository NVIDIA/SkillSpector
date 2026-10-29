# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disposable static inspection of one verified npm tarball, entirely in memory."""

from __future__ import annotations

import base64
import gzip
import io
import json
import math
import os
import re
import sys
import tarfile
from dataclasses import asdict
from pathlib import PurePosixPath

from skillspector.artifacts import ContentKind, classify_artifact
from skillspector.dependency_sandbox import restrict_worker
from skillspector.nodes.analyzers import (
    static_patterns_data_exfiltration,
    static_patterns_privilege_escalation,
    static_patterns_supply_chain,
)
from skillspector.nodes.analyzers.static_runner import run_static_patterns_with_ledger
from skillspector.state import WorkflowResourceBudget

MAX_INPUT_BYTES = 12 * 1024 * 1024
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
MODULES = (
    static_patterns_data_exfiltration,
    static_patterns_privilege_escalation,
    static_patterns_supply_chain,
)
SOURCE_SUFFIXES = frozenset(
    {
        ".js",
        ".mjs",
        ".cjs",
        ".ts",
        ".json",
        ".py",
        ".sh",
        ".bash",
        ".zsh",
        ".ps1",
        ".md",
        ".markdown",
        ".txt",
        ".yaml",
        ".yml",
        ".toml",
    }
)
MAX_PACKAGE_SCRIPTS = 128


class InspectionError(ValueError):
    """An allowlisted archive or coverage limitation."""


def _object_members(text: str, position: int):
    """Yield object properties and token offsets from already validated JSON."""
    decoder = json.JSONDecoder()
    whitespace = re.compile(r"[ \t\r\n]*")
    position += 1
    while True:
        position = whitespace.match(text, position).end()
        if text[position] == "}":
            return
        key_start = position
        key, position = decoder.raw_decode(text, position)
        position = whitespace.match(text, position).end() + 1  # colon
        value_start = whitespace.match(text, position).end()
        value, position = decoder.raw_decode(text, value_start)
        yield key, value, key_start, value_start
        position = whitespace.match(text, position).end()
        if text[position] == "}":
            return
        position += 1  # comma


def _script_lines(text: str) -> dict[str, int]:
    scripts_start = None
    for key, value, _, value_start in _object_members(text, text.index("{")):
        if key == "scripts":
            scripts_start = value_start if isinstance(value, dict) else None
    if scripts_start is None:
        return {}
    result = {}
    for index, (key, _, key_start, _) in enumerate(_object_members(text, scripts_start)):
        if index >= MAX_PACKAGE_SCRIPTS:
            break
        result[key] = text.count("\n", 0, key_start) + 1
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InspectionError("dependency_parse_error")
        result[key] = value
    return result


def _member_path(name: str) -> str:
    if (
        not name
        or len(name) > 1024
        or "\\" in name
        or ":" in name
        or any(ord(c) < 32 or ord(c) == 127 for c in name)
    ):
        raise InspectionError("archive_unsafe_member_path")
    parts = name.rstrip("/").split("/")
    if parts[0] != "package" or any(part in {"", ".", ".."} for part in parts):
        raise InspectionError("archive_unsafe_member_path")
    return "/".join(parts[1:])


def _scan(text: str, path: str, budget: WorkflowResourceBudget) -> tuple[list[dict], list[str]]:
    findings: list[dict] = []
    reasons: list[str] = []
    state = {
        "components": [path],
        "file_cache": {path: text},
        "local_file_cache": {path: text},
        "use_llm": False,
        "workflow_resource_budget": budget,
    }
    for module in MODULES:
        result = run_static_patterns_with_ledger(state, [module])
        findings.extend(asdict(item) for item in result.get("findings", []))
        reasons.extend(
            str(item["reason_code"])
            for item in result.get("inspection_ledger", [])
            if item.get("outcome") not in {"completed", "out_of_scope"}
        )
    return findings, reasons


def inspect_archive(request: dict) -> dict:
    """Read bounded gzip/tar bytes; never extract a file or run package code."""
    blob = base64.b64decode(request["artifact"], validate=True)
    limits = request["limits"]
    expanded_limit = limits["expanded_bytes"]
    file_limit = limits["file_bytes"]
    member_limit = limits["members"]
    finding_limit = limits["findings"]
    with gzip.GzipFile(fileobj=io.BytesIO(blob)) as compressed:
        raw_tar = compressed.read(expanded_limit + 1)
    if len(raw_tar) > expanded_limit:
        raise InspectionError("archive_size_limit")
    if len(raw_tar) > max(1, len(blob)) * limits["compression_ratio"]:
        raise InspectionError("archive_compression_ratio")

    members: dict[str, bytes] = {}
    sizes: dict[str, int] = {}
    oversized: set[str] = set()
    limitations: list[str] = []
    seen: set[str] = set()
    member_count = 0
    with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r:") as archive:
        for member in archive:
            member_count += 1
            if member_count > member_limit:
                member_count = member_limit
                limitations.append("archive_member_limit")
                break
            path = _member_path(member.name)
            if not path and member.isdir():
                continue
            if not path or path in seen:
                raise InspectionError("archive_ambiguous_member_path")
            seen.add(path)
            if member.isdir():
                continue
            if not member.isfile():
                raise InspectionError(
                    "archive_link_member"
                    if member.issym() or member.islnk()
                    else "not_regular_file"
                )
            sizes[path] = member.size
            if member.size > file_limit:
                limitations.append("archive_member_size_limit")
                # Keep an explicit inventory row for an oversized file.
                members[path] = b""
                oversized.add(path)
                continue
            source = archive.extractfile(member)
            if source is None:
                raise InspectionError("archive_malformed")
            content = source.read(file_limit + 1)
            if len(content) != member.size:
                raise InspectionError("archive_truncated")
            members[path] = content
        if "archive_member_limit" not in limitations and any(raw_tar[archive.offset :]):
            raise InspectionError("archive_malformed")

    if "package.json" not in members:
        raise InspectionError(limitations[0] if limitations else "dependency_parse_error")
    if "package.json" in oversized:
        raise InspectionError("archive_member_size_limit")
    try:
        manifest = json.loads(members.get("package.json", b""), object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise InspectionError("dependency_parse_error") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("name") != request["name"]
        or manifest.get("version") != request["version"]
    ):
        raise InspectionError("dependency_identity_mismatch")

    prefix = request["prefix"]
    budget = WorkflowResourceBudget(max_seconds=limits["seconds"])
    budget.start()
    files: list[dict] = []
    total_findings = 0
    for path, content in members.items():
        canonical = f"{prefix}/{path}"
        row = {
            **classify_artifact(canonical, content),
            "size_bytes": sizes[path],
            "disposition": "analyzed",
            "findings": [],
        }
        reasons: list[str] = []
        supported = PurePosixPath(path).suffix.lower() in SOURCE_SUFFIXES or PurePosixPath(
            path
        ).name.lower().startswith(("license", "notice"))
        if path in oversized:
            reasons.append("archive_member_size_limit")
        elif (
            not supported
            or row["content_kind"] is not ContentKind.TEXT
            or not row["decodable"]
            or row["contains_nul"]
        ):
            reasons.append("dependency_unsupported_content")
        else:
            try:
                text = content.decode("utf-8")
            except UnicodeError:
                text = ""
                reasons.append("dependency_unsupported_content")
            if not reasons:
                row["findings"], scan_reasons = _scan(text, canonical, budget)
                reasons.extend(scan_reasons)
                if path == "package.json":
                    scripts = manifest.get("scripts", {})
                    if not isinstance(scripts, dict):
                        reasons.append("dependency_parse_error")
                        scripts = {}
                    if len(scripts) > MAX_PACKAGE_SCRIPTS:
                        reasons.append("artifact_count_limit")
                    script_lines = _script_lines(text)
                    # Decode every script, including commands reached through
                    # `npm run`, without inferring that every command executes.
                    for script_index, hook in enumerate(list(scripts)[:MAX_PACKAGE_SCRIPTS]):
                        command = scripts[hook]
                        if not isinstance(command, str):
                            reasons.append("dependency_parse_error")
                            continue
                        projected, hook_reasons = _scan(
                            command, f"{prefix}/hooks/{script_index}.sh", budget
                        )
                        reasons.extend(hook_reasons)
                        manifest_line = script_lines.get(hook, 1)
                        for finding in projected:
                            script_line = finding["start_line"]
                            finding.update(
                                file=canonical,
                                start_line=manifest_line,
                                end_line=None,
                                start_column=None,
                                end_column=None,
                            )
                            finding["evidence"].update(
                                npm_script=hook,
                                script_line=script_line,
                                projection="decoded npm script command",
                            )
                        row["findings"].extend(projected)
        if total_findings + len(row["findings"]) > finding_limit:
            row["findings"] = row["findings"][: max(0, finding_limit - total_findings)]
            reasons.append("output_limit")
        total_findings += len(row["findings"])
        if reasons:
            row.update(disposition="partial", reason=reasons[0])
            limitations.extend(reasons)
        files.append(row)
        if total_findings >= finding_limit or budget.remaining_seconds() <= 0:
            limitations.append(
                "output_limit" if total_findings >= finding_limit else "runtime_limit"
            )
            break
    return {
        "files": files,
        "limitations": sorted(set(limitations)),
        "expanded_bytes": len(raw_tar),
        "members": member_count,
    }


def main() -> None:
    """Read a bounded JSON request and emit a bounded JSON response."""
    try:
        import resource

        descriptor_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        os.closerange(3, descriptor_limit if descriptor_limit > 0 else 1_048_576)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        # Load the codec and gzip machinery before denying filesystem access.
        b"".decode("utf-8")
        gzip.decompress(gzip.compress(b""))
        restrict_worker()
    except (ImportError, OSError, ValueError):
        print('{"error":"dependency_sandbox_unavailable"}')
        return
    try:
        raw_request = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(raw_request) > MAX_INPUT_BYTES:
            raise InspectionError("size_limit")
        request = json.loads(raw_request)
        # The parent enforces wall time independently of the worker CPU limit.
        if not math.isfinite(request["limits"]["seconds"]):
            raise InspectionError("runtime_limit")
        result = inspect_archive(request)
        output = json.dumps(result, ensure_ascii=True)
        if len(output) > MAX_OUTPUT_BYTES:
            raise InspectionError("output_limit")
    except InspectionError as exc:
        output = json.dumps({"error": str(exc)})
    except (
        ValueError,
        TypeError,
        KeyError,
        EOFError,
        RecursionError,
        MemoryError,
        OSError,
        tarfile.TarError,
    ):
        output = '{"error":"archive_malformed"}'
    sys.stdout.write(output)


if __name__ == "__main__":
    main()
