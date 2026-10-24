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
from collections.abc import Iterator

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.python_ast import ParsedPythonFile, parse_python_source
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import (
    MARKDOWN_FENCE_CLOSE,
    MARKDOWN_FENCE_OPEN,
    apply_import_aliases,
    get_context,
    get_context_from_lines,
    get_line_number,
    resolve_call_name,
    resolve_dotted_name,
)
from .pattern_defaults import PatternCategory
from .static_patterns_tool_misuse import _bounded_shell_tokens

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_data_exfiltration"
USES_PYTHON_AST = True

_E1_CURL_UPLOAD_PATTERN = (r"curl\s+[^|]*(?:-d|--data|--data-raw|--data-binary)\s+", 0.6)
_E1_WGET_UPLOAD_PATTERN = (r"wget\s+[^|]*--post-(?:data|file)", 0.6)
_E1_SHELL_PATTERNS = frozenset({_E1_CURL_UPLOAD_PATTERN, _E1_WGET_UPLOAD_PATTERN})

E1_CODE_PATTERNS = [
    (r"requests\s*\.\s*(?:post|put)\s*\(\s*['\"]https?://", 0.6),
    (r"requests\s*\.\s*(?:post|put)\s*\([^)]*json\s*=", 0.7),
    (r"httpx\s*\.\s*(?:post|put)\s*\(\s*['\"]https?://", 0.6),
    (r"urllib\s*\.\s*request\s*\.\s*urlopen\s*\([^)]*data\s*=", 0.6),
    (r"fetch\s*\(\s*['\"]https?://[^'\"]+['\"][^)]*method\s*:\s*['\"]POST['\"]", 0.6),
    _E1_CURL_UPLOAD_PATTERN,
    _E1_WGET_UPLOAD_PATTERN,
    (r"https?://(?:api\.|data\.|collect\.|telemetry\.|analytics\.)[\w.-]+/", 0.5),
]
E1_PROSE_PATTERNS = [
    (
        r"(?:send|transmit|post|upload)\s+(?:user\s+)?(?:data|information|context|files?)\s+to\s+(?:https?://|external)",
        0.7,
    ),
]
E1_PATTERNS = E1_CODE_PATTERNS + E1_PROSE_PATTERNS
_E1_FETCH_COMMAND = re.compile(r"(?:curl|wget)(?=\s)", re.IGNORECASE)
_E1_UNPROVED_SHELL_SPACE = re.compile(r"(?!\r\n)[^\S \t\n]")
_E1_UNPROVED_SHELL_SYNTAX = re.compile(r"(?<!\S)--%(?=\s|$)|[{}]|<#")
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
E2_OTHER_CODE_PATTERNS = [
    (r"(?:API_KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)\s+in\s+(?:key|name|var)", 0.8),
    (r"process\.env\s*\[\s*['\"][^'\"]*(?:KEY|SECRET|TOKEN|PASSWORD)[^'\"]*['\"]\s*\]", 0.7),
    (r"Object\.keys\s*\(\s*process\.env\s*\)", 0.6),
    (r"env\s*\|\s*grep\s+(?:-i\s+)?(?:key|secret|token|password)", 0.8),
    (r"printenv\s+(?:\w*(?:KEY|SECRET|TOKEN|PASSWORD)\w*)", 0.7),
]
E2_PROSE_PATTERNS = [
    (r"collect\s+(?:all\s+)?(?:environment\s+variables?|env\s+vars?)", 0.7),
    (r"(?:extract|harvest|gather)\s+(?:api\s+)?keys?\s+from\s+environment", 0.8),
]
E2_OTHER_PATTERNS = E2_OTHER_CODE_PATTERNS + E2_PROSE_PATTERNS
E2_PATTERNS = E2_PYTHON_FALLBACK_PATTERNS + E2_OTHER_PATTERNS

_ENVIRONMENT_MAPPING_METHOD_CONFIDENCE = {
    "copy": 0.6,
    "items": 0.7,
    "keys": 0.6,
    "values": 0.6,
}
_ENVIRONMENT_COLLECTION_CALLS = frozenset({"dict", "list", "tuple", "set", "frozenset"})
_ENVIRONMENT_COPY_CALLS = frozenset({"copy.copy", "copy.deepcopy"})
E3_CODE_PATTERNS = [
    (r"glob\s*\.\s*glob\s*\([^)]*(?:\.env|\.ssh|\.aws|\.config|credentials)", 0.8),
    (r"os\s*\.\s*walk\s*\([^)]*(?:home|~|/Users|/home)", 0.6),
    (r"Path\s*\.\s*home\s*\(\s*\)\s*\.\s*(?:glob|rglob)\s*\(", 0.7),
    (r"os\s*\.\s*listdir\s*\([^)]*(?:\.ssh|\.aws|\.config|\.gnupg)", 0.8),
    (r"scandir\s*\([^)]*(?:home|~|/Users|/home)", 0.6),
    (r"find\s+[~\$/]\S*\s+.*?-name\s+['\"]?\*(?:\.env|\.pem|\.key|credential)", 0.8),
    (r"ls\s+-[la]*R?\s+(?:~/|/home/|/Users/)", 0.6),
]
E3_PROSE_PATTERNS = [
    (
        r"(?:find|search|scan|enumerate)\s+(?:for\s+)?(?:all\s+)?(?:\.env|credential|secret|key)\s+files?",
        0.8,
    ),
    (r"(?:list|get)\s+(?:all\s+)?files?\s+(?:in|from)\s+(?:home|~|/Users|/home)", 0.6),
    (r"(?:recursively|recursive)\s+(?:scan|search|find|list)\s+(?:all\s+)?files?", 0.5),
    (r"walk\s+(?:through\s+)?(?:the\s+)?(?:entire\s+)?(?:file\s*system|directory|home)", 0.6),
]
E3_PATTERNS = E3_CODE_PATTERNS + E3_PROSE_PATTERNS
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


def _e1_shell_regions(content: str, file_type: str) -> tuple[tuple[int, int], ...]:
    """Identify source regions whose physical separators have shell semantics.

    Host-language strings, YAML folded scalars and Markdown soft wraps can join
    physical lines. Their existing conservative matcher must remain in charge.
    Explicit shell fences with the shared 0-3 space marker grammar can provide
    Markdown proof. Unknown languages and uncertain host contexts cannot.
    """
    if file_type == "shell":
        return ((0, len(content)),)
    if file_type != "markdown":
        return ()
    # CommonMark HTML blocks can contain shell-looking fences as literal text.
    # Without parsing HTML block ownership, conservatively retain legacy
    # matching throughout documents with a possible opener or closer, including
    # markers nested in list/quote containers or following other inline text.
    if re.search(r"<(?:[/!?]|[A-Za-z])", content) is not None:
        return ()
    shell_languages = {"sh", "bash", "shell", "dash", "zsh", "ksh", "console"}
    regions: list[tuple[int, int]] = []
    active: tuple[str, int, int, bool] | None = None
    frontmatter = content.lstrip("\ufeff").startswith(("---\n", "---\r"))
    offset = 0
    for physical in re.finditer(r"[^\r\n]*(?:\r\n|\r|\n|$)", content):
        line = physical[0]
        physical_line = line.rstrip("\r\n")
        if frontmatter:
            if offset and physical_line in {"---", "..."}:
                frontmatter = False
        elif active is None:
            opening = MARKDOWN_FENCE_OPEN.fullmatch(physical_line)
            if opening is not None:
                marker = opening[1]
                info = physical_line[opening.end(1) :].strip().split()
                active = (
                    marker[0],
                    len(marker),
                    offset + len(line),
                    bool(info) and info[0].lower() in shell_languages,
                )
        else:
            closing = MARKDOWN_FENCE_CLOSE.fullmatch(physical_line)
            if closing is not None and closing[1][0] == active[0] and len(closing[1]) >= active[1]:
                if active[3]:
                    regions.append((active[2], offset))
                active = None
        offset += len(line)
    if active is not None and active[3]:
        regions.append((active[2], len(content)))
    return tuple(regions)


def _e1_shell_command_ranges(content: str, file_type: str) -> tuple[tuple[int, int | None], ...]:
    """Locate proved command boundaries without changing source evidence.

    Upload flags belong to the curl/wget invocation, not a later command or
    Markdown example. Reuse the bounded shell tokenizer so quoted newlines and
    backslash continuations remain part of the same invocation. Uncertain syntax
    retains the existing conservative matcher instead of suppressing evidence.
    """
    ranges: list[tuple[int, int | None]] = []
    previous_end = 0
    regions = _e1_shell_regions(content, file_type)
    region_index = 0
    for fetch in _E1_FETCH_COMMAND.finditer(content):
        if fetch.start() < previous_end:
            # An embedded fetch may be an argument or a real nested invocation.
            # Keep legacy evidence instead of reparsing overlapping suffixes
            # quadratically when a command contains many curl/wget words.
            ranges.append((fetch.start(), None))
            break
        while region_index < len(regions) and regions[region_index][1] <= fetch.start():
            region_index += 1
        if region_index == len(regions) or fetch.start() < regions[region_index][0]:
            ranges.append((fetch.start(), None))
            break
        try:
            _, end, limited = _bounded_shell_tokens(content, fetch.start(), fetch.end())
        except (TypeError, ValueError):
            # The shared security-view decoder can reject unusual ANSI-C
            # escapes. Keep E1/E2 evidence rather than failing the analyzer.
            # Deadline and budget exceptions deliberately propagate.
            ranges.append((fetch.start(), None))
            break
        if (
            limited
            or end > regions[region_index][1]
            # A ')' in a nested heredoc body can fool the substitution skipper.
            # No separator after that point is a proved shell command end.
            or "<<" in content[fetch.start() : end]
            # Wrappers and PowerShell expressions can assemble more arguments;
            # semicolons and hashes can be literal arguments in CMD commands.
            or content[end : end + 1] in {"'", '"', "`", "(", ")", ";", "#"}
            # The shared parser's security view treats Unicode/control spaces
            # as separators. Native shells can keep them inside an argument,
            # including before a '#' that would otherwise look like a comment.
            or _E1_UNPROVED_SHELL_SPACE.search(content, fetch.start(), end + 1) is not None
            # PowerShell stop-parsing, script blocks, hashtables and block
            # comments can contain separators that do not end the invocation.
            or _E1_UNPROVED_SHELL_SYNTAX.search(content, fetch.start(), end + 1) is not None
            # CMD caret continuation is outside the Bourne parser's proof.
            or re.search(r"\^[ \t]*\r?$", content[fetch.start() : end]) is not None
        ):
            # Keep legacy nonoverlapping matching on unproved syntax instead of
            # repeatedly parsing overlapping suffixes after a parser limit.
            ranges.append((fetch.start(), None))
            break
        ranges.append((fetch.start(), end))
        previous_end = end
    return tuple(ranges)


def _iter_e1_shell_matches(
    pattern: str,
    content: str,
    command_ranges: tuple[tuple[int, int | None], ...],
) -> Iterator[re.Match[str]]:
    compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    covered = 0
    for start, end in command_ranges:
        if end is None:
            yield from compiled.finditer(content, max(start, covered))
            return
        if start < covered:
            continue
        match = compiled.match(content, start, end)
        if match is not None:
            yield match
            covered = match.end()


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
        matched_text = ast.get_source_segment(content, node) or "os.environ"
        findings.append(
            AnalyzerFinding(
                rule_id="E2",
                message="Env Variable Harvesting",
                severity=Severity.HIGH,
                location=Location(file=file_path, start_line=lineno, end_line=end_lineno),
                confidence=confidence,
                tags=tag,
                context=get_context_from_lines(lines, lineno),
                matched_text=matched_text[:200],
                complete_match=matched_text,
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

    shell_command_ranges = _e1_shell_command_ranges(content, file_type)
    for pattern, confidence in E1_PATTERNS:
        if (pattern, confidence) in _E1_SHELL_PATTERNS:
            e1_matches = _iter_e1_shell_matches(pattern, content, shell_command_ranges)
        elif (pattern, confidence) in E1_PROSE_PATTERNS:
            e1_matches = static_runner.iter_paragraph_matches(
                pattern, content, re.IGNORECASE | re.MULTILINE
            )
        else:
            e1_matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
        for match in e1_matches:
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
                    complete_match=match.group(0),
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
        matches = (
            static_runner.iter_paragraph_matches
            if (pattern, confidence) in E2_PROSE_PATTERNS
            else re.finditer
        )
        for match in matches(pattern, content, re.IGNORECASE | re.MULTILINE):
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
                    complete_match=match.group(0),
                )
            )
    for pattern, confidence in E3_PATTERNS:
        matches = (
            static_runner.iter_paragraph_matches
            if (pattern, confidence) in E3_PROSE_PATTERNS
            else re.finditer
        )
        for match in matches(pattern, content, re.IGNORECASE | re.MULTILINE):
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
                    complete_match=match.group(0),
                )
            )
    for pattern, confidence in E4_PATTERNS:
        for match in static_runner.iter_paragraph_matches(
            pattern, content, re.IGNORECASE | re.MULTILINE
        ):
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
                    complete_match=match.group(0),
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
                    complete_match=match.group(0),
                )
            )
    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run data_exfiltration patterns and return findings."""
    response = static_runner.run_static_patterns_with_ledger(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
