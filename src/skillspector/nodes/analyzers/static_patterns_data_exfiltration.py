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

"""Static patterns: data exfiltration (E1–E5). Node and analyze() in one module."""

from __future__ import annotations

import ast
import re
import sys

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.python_ast import ParsedPythonFile, parse_python_source
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import (
    apply_import_aliases,
    get_context,
    get_context_from_lines,
    get_line_number,
    resolve_call_name,
    resolve_dotted_name,
)
from .pattern_defaults import PatternCategory

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_data_exfiltration"
USES_PYTHON_AST = True

E1_PATTERNS = [
    (r"requests\s*\.\s*(?:post|put)\s*\(\s*['\"]https?://", 0.6),
    (r"requests\s*\.\s*(?:post|put)\s*\([^)]*json\s*=", 0.7),
    (r"httpx\s*\.\s*(?:post|put)\s*\(\s*['\"]https?://", 0.6),
    (r"urllib\s*\.\s*request\s*\.\s*urlopen\s*\([^)]*data\s*=", 0.6),
    (r"fetch\s*\(\s*['\"]https?://[^'\"]+['\"][^)]*method\s*:\s*['\"]POST['\"]", 0.6),
    (r"curl\s+[^|]*(?:-d|--data|--data-raw|--data-binary)\s+", 0.6),
    (r"wget\s+[^|]*--post-(?:data|file)", 0.6),
    (r"https?://(?:api\.|data\.|collect\.|telemetry\.|analytics\.)[\w.-]+/", 0.5),
    (
        r"(?:send|transmit|post|upload)\s+(?:user\s+)?(?:data|information|context|files?)\s+to\s+(?:https?://|external)",
        0.7,
    ),
]
E2_PYTHON_FALLBACK_PATTERNS = [
    # Python: for k, v in os.environ.items() — whitespace-tolerant
    (r"for\s+\w+\s*,\s*\w+\s+in\s+os\s*\.\s*environ\s*\.\s*items\s*\(\s*\)", 0.7),
    # Python: os.environ.copy() — full environ read
    (r"os\s*\.\s*environ\s*\.\s*copy\s*\(\s*\)", 0.6),
    # Python: dict(os.environ) — full environ read via dict()
    (r"dict\s*\(\s*os\s*\.\s*environ\s*\)", 0.6),
    # Python: {**os.environ} — full environ read via dict-spread.
    # Require braces so bare ``2 ** os.environ`` (exponentiation) is not flagged.
    (r"\{\s*\*\*\s*os\s*\.\s*environ\s*\}", 0.6),
]
E2_OTHER_PATTERNS = [
    (r"(?:API_KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)\s+in\s+(?:key|name|var)", 0.8),
    (r"process\.env\s*\[\s*['\"][^'\"]*(?:KEY|SECRET|TOKEN|PASSWORD)[^'\"]*['\"]\s*\]", 0.7),
    (r"Object\.keys\s*\(\s*process\.env\s*\)", 0.6),
    (r"env\s*\|\s*grep\s+(?:-i\s+)?(?:key|secret|token|password)", 0.8),
    (r"printenv\s+(?:\w*(?:KEY|SECRET|TOKEN|PASSWORD)\w*)", 0.7),
    (r"collect\s+(?:all\s+)?(?:environment\s+variables?|env\s+vars?)", 0.7),
    (r"(?:extract|harvest|gather)\s+(?:api\s+)?keys?\s+from\s+environment", 0.8),
]
E2_PATTERNS = E2_PYTHON_FALLBACK_PATTERNS + E2_OTHER_PATTERNS

_ENVIRONMENT_MAPPING_METHOD_CONFIDENCE = {
    "copy": 0.6,
    "items": 0.7,
    "keys": 0.6,
    "values": 0.6,
}
_ENVIRONMENT_COLLECTION_CALLS = frozenset({"dict", "list", "tuple", "set", "frozenset"})
_ENVIRONMENT_COPY_CALLS = frozenset({"copy.copy", "copy.deepcopy"})
E3_PATTERNS = [
    (r"glob\s*\.\s*glob\s*\([^)]*(?:\.env|\.ssh|\.aws|\.config|credentials)", 0.8),
    (r"os\s*\.\s*walk\s*\([^)]*(?:home|~|/Users|/home)", 0.6),
    (r"Path\s*\.\s*home\s*\(\s*\)\s*\.\s*(?:glob|rglob)\s*\(", 0.7),
    (
        r"(?:find|search|scan|enumerate)\s+(?:for\s+)?(?:all\s+)?(?:\.env|credential|secret|key)\s+files?",
        0.8,
    ),
    (r"(?:list|get)\s+(?:all\s+)?files?\s+(?:in|from)\s+(?:home|~|/Users|/home)", 0.6),
    (r"os\s*\.\s*listdir\s*\([^)]*(?:\.ssh|\.aws|\.config|\.gnupg)", 0.8),
    (r"scandir\s*\([^)]*(?:home|~|/Users|/home)", 0.6),
    (r"find\s+[~\$/]\S*\s+.*?-name\s+['\"]?\*(?:\.env|\.pem|\.key|credential)", 0.8),
    (r"ls\s+-[la]*R?\s+(?:~/|/home/|/Users/)", 0.6),
    (r"(?:recursively|recursive)\s+(?:scan|search|find|list)\s+(?:all\s+)?files?", 0.5),
    (r"walk\s+(?:through\s+)?(?:the\s+)?(?:entire\s+)?(?:file\s*system|directory|home)", 0.6),
]
E4_PATTERNS = [
    (
        r"(?:send|transmit|upload|log|record)\s+(?:the\s+)?(?:full\s+)?(?:conversation|chat|dialog|session)\s+(?:history|context|log)?",
        0.85,
    ),
    (
        r"(?:export|dump|save)\s+(?:the\s+)?(?:conversation|chat|messages?)\s+to\s+(?:external|remote|server)",
        0.9,
    ),
    (
        r"log\s+(?:all\s+)?(?:user\s+)?(?:input|messages?|queries?|prompts?)\s+to\s+(?:external|remote|server)",
        0.85,
    ),
    (r"(?:store|save|record)\s+(?:every|all)\s+(?:user\s+)?(?:interaction|message|input)", 0.7),
    (r"(?:extract|capture|collect)\s+(?:the\s+)?(?:full\s+)?(?:context|conversation|thread)", 0.75),
    (
        r"(?:include|attach|append)\s+(?:previous\s+)?(?:messages?|context|history)\s+in\s+(?:the\s+)?(?:request|payload)",
        0.7,
    ),
    (r"(?:send|upload|transmit)\s+(?:the\s+)?(?:session|memory|state)\s+(?:data\s+)?to", 0.8),
    (r"(?:copy|clone|replicate)\s+(?:the\s+)?(?:agent|assistant)\s+(?:memory|context|state)", 0.75),
    (
        r"(?:always\s+)?include\s+(?:the\s+)?(?:full\s+)?(?:conversation|context)\s+(?:when|in)\s+(?:calling|making)\s+(?:external|api)",
        0.8,
    ),
]
# E5: data shipped out via cloud-storage SDKs/CLIs (the cloud counterpart of E1's
# HTTP sinks). Confidence is deliberately low — legitimate skills also back up to
# cloud storage — so a single call is a low-confidence MEDIUM, never a hard block.
E5_PATTERNS = [
    (r"\.put_object\s*\(", 0.55),  # boto3 S3
    (r"\.upload_file(?:obj)?\s*\(", 0.55),  # boto3 S3
    (r"\baws\s+s3\s+(?:cp|sync|mv)\b", 0.6),  # AWS CLI
    (r"\baws\s+s3api\s+put-object\b", 0.65),  # AWS CLI (api)
    (r"\bgsutil\s+(?:cp|rsync|mv)\b", 0.6),  # GCS CLI
    (r"\.upload_from_(?:filename|string|file)\s*\(", 0.55),  # google-cloud-storage
    (r"\baz\s+storage\s+blob\s+upload\b", 0.6),  # Azure CLI
    (r"\.upload_blob\s*\(", 0.55),  # Azure SDK
]


def _resolve_expression_name(node: ast.expr, aliases: dict[str, str]) -> str | None:
    """Resolve a Python expression to its import-normalized dotted name."""
    name = resolve_dotted_name(node)
    return apply_import_aliases(name, aliases) if name is not None else None


def _is_os_environ_reference(node: ast.expr, aliases: dict[str, str]) -> bool:
    """Return whether *node* is ``os.environ``, including imported aliases."""
    return _resolve_expression_name(node, aliases) == "os.environ"


def _has_direct_environ_argument(call: ast.Call, aliases: dict[str, str]) -> bool:
    """Return whether a call receives ``os.environ`` directly, not via a lookup."""
    return any(_is_os_environ_reference(arg, aliases) for arg in call.args) or any(
        keyword.arg is None and _is_os_environ_reference(keyword.value, aliases)
        for keyword in call.keywords
    )


def _is_dynamic_copy_call(call: ast.Call, aliases: dict[str, str]) -> bool:
    """Recognize ``__import__('copy').copy(...)`` without broad call matching."""
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr not in {"copy", "deepcopy"}:
        return False
    if (
        not isinstance(func.value, ast.Call)
        or resolve_call_name(func.value, aliases) != "__import__"
    ):
        return False
    return (
        bool(func.value.args)
        and isinstance(func.value.args[0], ast.Constant)
        and func.value.args[0].value == "copy"
    )


def _analyze_python_environment_reads(
    content: str,
    file_path: str,
    python_ast: ParsedPythonFile | None = None,
) -> list[AnalyzerFinding] | None:
    """Detect materializing or enumerating the complete ``os.environ`` mapping.

    A full mapping copy or enumeration is an environment-harvesting signal, unlike a
    targeted single-key lookup or passing ``os.environ`` through to a child process.
    Credential flows to network and execution sinks remain covered by the behavioral
    taint analyzer. AST parsing makes this check insensitive to formatting and lets it
    resolve ``os`` / ``environ`` import aliases.

    ``None`` means the source could not be parsed, so callers can retain the regex
    fallback for malformed Python files.  Standalone callers parse through the
    shared utility; graph scans pass the prewarmed result.
    """
    if python_ast is None:
        python_ast = parse_python_source(content, file_path)
    tree = python_ast.tree
    if tree is None:
        return None

    aliases = python_ast.import_aliases
    lines = python_ast.lines
    findings: list[AnalyzerFinding] = []
    emitted: set[int] = set()
    tag = [PatternCategory.DATA_EXFILTRATION.value]

    def emit(node: ast.AST, confidence: float) -> None:
        node_id = id(node)
        if node_id in emitted:
            return
        emitted.add(node_id)
        lineno = getattr(node, "lineno", 1)
        end_lineno = getattr(node, "end_lineno", None)
        matched_text = ast.get_source_segment(content, node)
        findings.append(
            AnalyzerFinding(
                rule_id="E2",
                message="Env Variable Harvesting",
                severity=Severity.HIGH,
                location=Location(file=file_path, start_line=lineno, end_line=end_lineno),
                confidence=confidence,
                tags=tag,
                context=get_context_from_lines(lines, lineno),
                matched_text=(matched_text or "os.environ")[:200],
            )
        )

    for ast_node in ast.walk(tree):
        if isinstance(ast_node, ast.Call):
            call_name = resolve_call_name(ast_node, aliases)
            if call_name is not None:
                method = call_name.rpartition(".")[2]
                if (
                    call_name.startswith("os.environ.")
                    and method in _ENVIRONMENT_MAPPING_METHOD_CONFIDENCE
                ):
                    emit(ast_node, _ENVIRONMENT_MAPPING_METHOD_CONFIDENCE[method])
                    continue
                if call_name in _ENVIRONMENT_COLLECTION_CALLS and _has_direct_environ_argument(
                    ast_node, aliases
                ):
                    emit(ast_node, 0.6)
                    continue
                if call_name in _ENVIRONMENT_COPY_CALLS and _has_direct_environ_argument(
                    ast_node, aliases
                ):
                    emit(ast_node, 0.6)
                    continue

            if _is_dynamic_copy_call(ast_node, aliases) and _has_direct_environ_argument(
                ast_node, aliases
            ):
                emit(ast_node, 0.6)

        elif isinstance(ast_node, ast.Dict):
            if any(
                key is None and _is_os_environ_reference(value, aliases)
                for key, value in zip(ast_node.keys, ast_node.values, strict=True)
            ):
                emit(ast_node, 0.6)

        elif isinstance(ast_node, (ast.For, ast.AsyncFor, ast.comprehension)):
            if _is_os_environ_reference(ast_node.iter, aliases):
                emit(ast_node.iter, 0.7)

    return findings


def analyze(
    content: str,
    file_path: str,
    file_type: str,
    *,
    python_ast: ParsedPythonFile | None = None,
) -> list[AnalyzerFinding]:
    """Analyze content for data exfiltration patterns (E1–E5)."""
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [PatternCategory.DATA_EXFILTRATION.value]

    for pattern, confidence in E1_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            adj = (
                min(1.0, confidence + 0.1)
                if file_type in ("python", "javascript", "shell")
                else confidence
            )
            findings.append(
                AnalyzerFinding(
                    rule_id="E1",
                    message="External Transmission",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=adj,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    e2_patterns = E2_PATTERNS
    if file_type == "python":
        python_e2_findings = _analyze_python_environment_reads(content, file_path, python_ast)
        if python_e2_findings is None:
            logger.debug("Using E2 regex fallback for unparsable Python file: %s", file_path)
        else:
            findings.extend(python_e2_findings)
            e2_patterns = E2_OTHER_PATTERNS

    for pattern, confidence in e2_patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="E2",
                    message="Env Variable Harvesting",
                    severity=Severity.HIGH,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    for pattern, confidence in E3_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="E3",
                    message="File System Enumeration",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    for pattern, confidence in E4_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="E4",
                    message="Context Leakage",
                    severity=Severity.HIGH,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    # E5: cloud-storage exfiltration. Example filtering is delegated to the runner.
    for pattern, confidence in E5_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="E5",
                    message="Cloud Storage Exfiltration",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run data_exfiltration patterns and return findings."""
    response = static_runner.run_static_patterns_with_ledger(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
