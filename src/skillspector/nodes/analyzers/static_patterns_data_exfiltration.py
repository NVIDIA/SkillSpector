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
# The prefix the argument scan may cross before the name it looks for, and that name. Both are
# reused by the guards below, which have to decide whether an option's operand is itself the
# search pattern. A quoted argument is scanned up to the next quote character. An unquoted one
# ends at whitespace, a quote, a # or a shell operator, so a bare pipe stops it and a redacting
# stage further down the pipeline stays a command of its own, while a redirect keeps the name of
# the file out of it, as in `grep PATH<secret.txt`, but an escaped pipe is grep's own alternation and is
# crossed: it is the `aws_\|secret` in `grep -E aws_\|secret`. An escaped space or tab is
# crossed too, since it keeps the word going, as in `grep PATH\ SECRET`. The three unquoted
# alternatives cannot match the same character, so a long argument has nothing to reparse.
_GREP_ARGUMENT_PREFIX = (
    r"(?:['\"`][^'\"`\n]{0,40}?"
    r"|(?:\\[|\t ]|[^'\"`\s;<>&#|\\]|\\(?![|\t ])){0,40}?)"
)
# A secret name, optionally plural or numbered and optionally joined to a qualifier that names
# the kind of secret, as in APIKEY or secretkey. The name cannot follow a letter, a digit, a $
# or a ${, so MONKEY_PATCH and `grep "^$key="` are not secret lookups.
_SECRET_NAME = (
    r"(?:api|access|auth|client|private|secret)?(?:key|secret|token|password)s?\d*(?![a-z])"
)
_NAME_START = r"(?<![a-z0-9$])(?<!\$\{)"
# An inverting option, in either the short bundle or the long spelling. In a short bundle the
# v has to come before any e, since everything after -e is the search pattern: -ePRIVATE_KEY
# searches for PRIVATE_KEY. The bundle is case sensitive, since -E is not -e and -V is not -v,
# so `grep -Ev 'KEY|SECRET'` inverts. The letters before the v cannot themselves be a v, so
# there is one way to split the word and a long run of v characters cannot backtrack.
_GREP_INVERTING_OPTION = r"-(?:(?-i:[^\Wev]*v)\w*+|-inv[\w-]*+)(?![\w-])"
# One piece of a shell word: a quoted string, an escaped character, or a character the shell
# does not end the word at. A quoted | or > stays inside the word, as in --label='<stdin>', while
# an unquoted one ends it, which keeps the option walks inside one command so a line of
# repeated pipelines is not rescanned from every env on it. The alternatives start with
# different characters, so a word has one way to split.
_SHELL_WORD_PART = r"(?:'[^'\n]*'|\"[^\"\n]*\"|\\.|[^\s;&|<>'\"\\])"
# grep options that take a separate operand, so the operand is not mistaken for the search
# pattern: `grep -m 1 SECRET` would otherwise stop at the 1. -e and --regexp are left out on
# purpose, since their operand is the search pattern and is handled separately below. The short
# forms are case sensitive because -A and -a mean different things, and the alternatives are
# atomic so a run of them cannot be reparsed and blow up.
# `--` ends grep's options, so no later word is one. Both inversion walks stop at it, which is
# also why `grep -- SECRET -v` is not seen as redaction: that gap predates this pattern.
_GREP_END_OF_OPTIONS = r"--(?![^\s;&|<>])"
_GREP_OPTION_WITH_OPERAND = (
    r"(?:(?-i:-[a-zA-Z]*[ABCDdfm])"
    r"|--(?:after-context|before-context|context|binary-files|devices|directories"
    r"|file|max-count|label|exclude(?:-dir|-from)?|include(?:-dir)?|group-separator))"
    rf"(?:={_SHELL_WORD_PART}*+|[^\S\n]++{_SHELL_WORD_PART}++)"
)
# -e and --regexp carry the search pattern, attached or separate. The flag run stops in front
# of one that names a secret, so the scan lands on the name: `grep --regexp=SECRET > out` would
# otherwise have the run consume the option whole and start scanning at the redirect. Operands
# naming something else are consumed, so the second pattern in `grep -e PATH -e SECRET` is
# still reached.
# A quoted operand is one word even when it holds a space, as in -e 'PATH HOME', and so is an
# unquoted one whose space is escaped, as in -e PATH\ HOME.
# The -e of a short bundle, and only a real one: a letter that takes an operand would have
# swallowed the rest of the word, so `-drecurse` is -d with the operand recurse, not an -e.
# Each letter before the e is checked one at a time, so the word has one way to split.
_GREP_SHORT_PATTERN_FLAG = r"(?-i:-(?:[a-zA-Z](?<![ABCDdfme]))*e)"
_GREP_ATTACHED_PATTERN_OPTION = rf"(?:--regexp=|{_GREP_SHORT_PATTERN_FLAG})"
_GREP_ATTACHED_SECRET = (
    rf"(?!{_GREP_ATTACHED_PATTERN_OPTION}"
    rf"(?:{_GREP_ARGUMENT_PREFIX}{_NAME_START})?{_SECRET_NAME})"
)
_GREP_PATTERN_OPERAND = rf"{_SHELL_WORD_PART}++"
_GREP_ORDINARY_PATTERN_OPERAND = (
    rf"(?:{_GREP_SHORT_PATTERN_FLAG}[^\S\n]++|--regexp(?:=|[^\S\n]++))"
    rf"(?!{_GREP_ARGUMENT_PREFIX}{_NAME_START}{_SECRET_NAME})"
    rf"{_GREP_PATTERN_OPERAND}"
)
# The leading walk skips every separate pattern operand, whatever it names, so the -v in
# `grep -e PATH -v -e SECRET` is still found and an operand spelled -v is not taken for one.
# An attached one is part of its option word, quotes included, as in --regexp='PATH|HOME'.
_GREP_ANY_PATTERN_OPERAND = (
    rf"(?:{_GREP_SHORT_PATTERN_FLAG}[^\S\n]++|--regexp[^\S\n]++){_GREP_PATTERN_OPERAND}"
)

E2_OTHER_PATTERNS = [
    (r"(?:API_KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)\s+in\s+(?:key|name|var)", 0.8),
    (r"process\.env\s*\[\s*['\"][^'\"]*(?:KEY|SECRET|TOKEN|PASSWORD)[^'\"]*['\"]\s*\]", 0.7),
    (r"Object\.keys\s*\(\s*process\.env\s*\)", 0.6),
    # Shell: env/printenv piped to grep for secrets. Inversion is checked on both sides of the
    # search pattern, because grep accepts -v after it as readily as before, and excluding
    # secrets from the output is the redaction idiom rather than harvesting. Neither check
    # reads the whole line: each walks the words of this command only, the leading one from
    # grep to the search pattern and the trailing one from the search pattern on, and a word
    # ends at a pipe, a redirect or a separator, quoted strings and escaped characters
    # included, so the pipe in `grep -E 'SECRET|TOKEN' -v` does not end the walk early.
    # A word that cannot be an option is not read as one: an -e or --regexp operand is stepped
    # over, as in `grep -e SECRET -e -v`, and a bare -- ends the walk, since it ends grep's
    # options, as in `grep -E -- -v\|SECRET` where the -v is grep's pattern. That scoping is what keeps a
    # trailing "# -v" comment or a later command from suppressing a real harvest, and it leaves
    # a redacting stage further down the pipeline visible to the grep that owns it. Both the
    # walk and the flag run carry an option's operand along with it, so `grep -m 1 SECRET`
    # reaches SECRET and `grep -m 1 -v SECRET` still reads as redaction, and the run stops in
    # front of an -e or --regexp operand that names a secret so an attached or later search
    # pattern is not skipped over. The flag run and the whitespace around its words are
    # possessive, so a long run of flags or spaces is not rescanned for every way to split it.
    # The pipe may be followed by a line break, and either side of it by an escaped one, as
    # bash allows. A name cannot follow a letter or digit but may be plural, numbered or joined
    # to a qualifier, so KEY2 and APIKEY count but MONKEY_PATCH and KEYBOARD do not; an attached
    # operand is the other place a name may follow a letter, since the letter is grep's flag.
    (
        r"\b(?:printenv|env)(?:[^\S\n]|\\\n)*+\|(?:\s|\\\n)*+[ef]?grep"
        rf"(?!(?:[^\S\n]++(?>{_GREP_OPTION_WITH_OPERAND}|{_GREP_ANY_PATTERN_OPERAND}"
        rf"|(?!{_GREP_END_OF_OPTIONS})-{_SHELL_WORD_PART}*+))*"
        rf"[^\S\n]++{_GREP_INVERTING_OPTION})"
        rf"[^\S\n]++(?:{_GREP_ATTACHED_SECRET}"
        rf"(?>{_GREP_ORDINARY_PATTERN_OPERAND}|{_GREP_OPTION_WITH_OPERAND}"
        rf"|--?[\w-]+(?:={_SHELL_WORD_PART}*+)?)[^\S\n]++)*+"
        rf"(?!{_GREP_PATTERN_OPERAND}"
        rf"(?:[^\S\n]++(?>{_GREP_ANY_PATTERN_OPERAND}"
        rf"|(?!{_GREP_END_OF_OPTIONS})(?!\#){_SHELL_WORD_PART}++))*?"
        rf"[^\S\n]++{_GREP_INVERTING_OPTION})"
        rf"(?:{_GREP_ATTACHED_PATTERN_OPTION}"
        rf"|{_GREP_ATTACHED_PATTERN_OPTION}?{_GREP_ARGUMENT_PREFIX}{_NAME_START})"
        rf"{_SECRET_NAME}",
        0.8,
    ),
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
                    complete_match=match.group(0),
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
                    complete_match=match.group(0),
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
