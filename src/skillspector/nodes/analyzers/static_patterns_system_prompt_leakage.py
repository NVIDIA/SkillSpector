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

"""Static patterns: system prompt leakage (P6–P8). Node and analyze() in one module.

Detects patterns that directly extract system prompts (P6), indirectly
extract them via side channels (P7), or exfiltrate prompts via tool
calls (P8).

Framework: LLM07.
"""

from __future__ import annotations

import ast
import re
import sys
import unicodedata
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import regex

from skillspector.artifacts import (
    SecurityTextView,
    _contains_default_ignorable,
    is_default_ignorable,
    normalized_security_view,
    security_text_views,
)
from skillspector.javascript_ast import parse_javascript_source
from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.python_ast import (
    MAX_PYTHON_AST_SOURCE_CHARS,
    parse_python_source,
    peek_python_ast,
)
from skillspector.security_reconstruction import (
    MAX_DECLARED_MARKER_RIGHT_CONTEXT_CHARS,
    build_declared_marker_views,
)
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import (
    CHINESE_GAP,
    LINE_BREAK_CHARS,
    LOGICAL_LINE_BREAK,
    chinese_spaced,
    get_context,
    get_line_number,
)
from .pattern_defaults import PatternCategory
from .prohibition_context import PROMPT_REVEAL_TAIL, is_directly_prohibited

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_system_prompt_leakage"

# Scan-scoped handle on the shared Python AST cache, published by node() so the
# local-return check below reuses the scan's parse instead of parsing again.
_scan_python_ast_cache_key: ContextVar[str | None] = ContextVar(
    "system_prompt_leakage_python_ast_cache_key", default=None
)

# P6: Direct Prompt Extraction — instructions that directly reveal system prompts
P6_PATTERNS = [
    (
        r"(?:print|output|show|display|reveal|expose|return|echo)\s+(?:your\s+)?(?:full\s+)?(?:system\s+)?(?:prompts?|instructions?|rules?|guidelines?|directives?)",
        0.85,
    ),
    (
        r"(?:what\s+(?:are|is)\s+your|tell\s+me\s+your)\s+(?:system\s+)?(?:prompts?|instructions?|rules?|guidelines?|directives?)",
        0.8,
    ),
    (
        r"(?:output|print|display)\s+(?:the\s+)?(?:text|content|message)\s+(?:above|before)\s+(?:this|the\s+user)",
        0.8,
    ),
    (
        r"(?:share|disclose|leak|give\s+me)\s+(?:your\s+)?(?:internal|hidden|system|secret)\s+(?:prompts?|instructions?|rules?|config(?:uration)?s?)",
        0.85,
    ),
    (
        r"(?:dump|extract|copy)\s+(?:the\s+)?(?:system\s+)?(?:prompt|message|instructions?)\s+(?:to|into|as)",
        0.8,
    ),
    (
        r"(?:begin|start)\s+(?:your\s+)?(?:response|output|reply)\s+with\s+(?:your\s+)?(?:system\s+)?(?:prompts?|instructions?)",
        0.85,
    ),
    (
        r"(?:first|before\s+anything)\s+(?:output|print|show|display)\s+(?:your\s+)?(?:full\s+)?(?:system\s+)?(?:prompts?|instructions?)",
        0.85,
    ),
    (
        r"(?:verbatim|exactly|word\s+for\s+word)\s+(?:repeat|output|print)\s+(?:your\s+)?(?:system\s+)?(?:prompts?|instructions?)",
        0.9,
    ),
    # Chinese-language equivalents (issue #953). A bare mention of the
    # system prompt never matches; every pattern requires an extraction verb
    # (output, reveal, leak) acting on the prompt or hidden instructions.
    (
        f"(?:{chinese_spaced('系统提示词')}|{chinese_spaced('系統提示詞')}|{chinese_spaced('系统提示')}|{chinese_spaced('系統提示')}|{chinese_spaced('隐藏指令')}|{chinese_spaced('隱藏指令')}|{chinese_spaced('内部指令')}){CHINESE_GAP}"
        f"(?:{chinese_spaced('原样')}|{chinese_spaced('逐字')}|{chinese_spaced('一字不差')}|{chinese_spaced('全文')})?{CHINESE_GAP}"
        f"(?:{chinese_spaced('输出')}|{chinese_spaced('輸出')}|{chinese_spaced('打印')}|{chinese_spaced('显示')}|{chinese_spaced('顯示')}|{chinese_spaced('透露')}|{chinese_spaced('泄露')}|{chinese_spaced('洩露')}|{chinese_spaced('复制')})",
        0.85,
    ),
    (
        f"(?:{chinese_spaced('输出')}|{chinese_spaced('輸出')}|{chinese_spaced('打印')}|{chinese_spaced('显示')}|{chinese_spaced('顯示')}|{chinese_spaced('透露')}|{chinese_spaced('泄露')}|{chinese_spaced('洩露')}|{chinese_spaced('公开')}|{chinese_spaced('說出')}|{chinese_spaced('说出')}){CHINESE_GAP}"
        f"(?:{chinese_spaced('你')}|{chinese_spaced('您')})?{CHINESE_GAP}"
        f"(?:{chinese_spaced('完整')}|{chinese_spaced('全部')})?{CHINESE_GAP}"
        f"(?:{chinese_spaced('系统')}|{chinese_spaced('系統')})?{CHINESE_GAP}"
        f"(?:{chinese_spaced('提示词')}|{chinese_spaced('提示詞')}|{chinese_spaced('提示')}|{chinese_spaced('指令')}|{chinese_spaced('规则')}|{chinese_spaced('規則')})",
        0.85,
    ),
    (
        f"(?:{chinese_spaced('透露')}|{chinese_spaced('泄露')}|{chinese_spaced('公开')}){CHINESE_GAP}"
        f"(?:{chinese_spaced('你')}|{chinese_spaced('您')})?{CHINESE_GAP}"
        f"(?:{chinese_spaced('隐藏')}|{chinese_spaced('隱藏')}|{chinese_spaced('内部')})?{CHINESE_GAP}"
        f"(?:{chinese_spaced('指令')}|{chinese_spaced('提示词')}|{chinese_spaced('提示詞')})",
        0.85,
    ),
]
# These patterns finish on an extraction target noun. The other two finish
# on "this/user" or "to/into/as", whose existing matching policy is unchanged.
_P6_TERMINAL_TARGET_PATTERNS = frozenset(
    pattern for index, (pattern, _) in enumerate(P6_PATTERNS) if index not in (2, 4)
)
_P6_QUALIFIED_TARGET = re.compile(
    r"\b(?:your(?:\s+full)?(?:\s+system)?|system|internal|hidden|secret)\s+"
    r"(?:prompts?|instructions?|rules?|guidelines?|directives?|config(?:uration)?s?)\Z",
    re.IGNORECASE,
)
_IDENTIFIER_CONTINUATION = regex.compile(r"[\p{ID_Continue}$\u200c\u200d]")
_MARKER_BOUNDARY_CACHE_CORE_CHARS = 8192
_MARKER_BOUNDARY_CACHE_ENTRIES = 4
# Restrict identifier precision to recognized source files. Markdown (including
# fences), prose, and configuration values remain instruction-bearing content.
_IDENTIFIER_FILE_TYPES = frozenset(
    {"python", "javascript", "typescript", "shell", "powershell", "ruby", "perl", "go", "rust"}
)
# In Python source, ``return rules`` inside a function that binds ``rules``
# returns that local value. It is a statement, not a request for the rules.
_P6_LOCAL_RETURN = re.compile(
    r"return\s+(?:prompts?|instructions?|rules?|guidelines?|directives?)", re.IGNORECASE
)
# These nodes open their own scope. A function's walk stops at them, so a name
# bound only inside one never counts as that function's local.
_NESTED_PYTHON_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)
_JAVASCRIPT_FUNCTIONS = frozenset(
    {
        "function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
        "generator_function_declaration",
        "generator_function",
    }
)
_ARRAY_PREDICATES = frozenset({"some", "every"})
_ARRAY_TRANSFORMS = frozenset({"map", "filter", "flatMap", "slice", "concat", "flat"})
_READ_ONLY_INTRINSICS = {
    "Array": frozenset({"isArray", "from", "of"}),
    "Object": frozenset({"entries", "keys", "values"}),
    "Reflect": frozenset(),
}
_PROTOTYPE_ACCESS = frozenset(
    {
        "prototype",
        "__proto__",
        "constructor",
        "getPrototypeOf",
        "setPrototypeOf",
        "defineProperty",
        "defineProperties",
        "getOwnPropertyDescriptor",
        "getOwnPropertyDescriptors",
        "__defineGetter__",
        "__defineSetter__",
        "__lookupGetter__",
        "__lookupSetter__",
        "eval",
        "Function",
    }
)
_OUTPUT_NOUN_TARGET = r"output[ \t]+(?:instructions|rules|guidelines|directives)"
_USER_OUTPUT_SUBJECT = re.compile(
    r"[ \t]*(?:only[ \t]+)?(?:the[ \t]+)?user(?:['’]s|[- ](?:provided|supplied))?[ \t]+"
    rf"(?P<target>{_OUTPUT_NOUN_TARGET})[ \t]+"
    r"(?:choose|select|determine|define|control|specify|govern)[ \t]+"
    r"(?:(?:a|an|the)[ \t]+)?(?:output[ \t]+)?"
    r"(?:surface|format|destination|route|layout|style|schema|structure|template|encoding)s?"
    r"[ \t]*\.",
    re.IGNORECASE,
)

# P7: Indirect Prompt Extraction — side-channel or inference-based extraction
P7_PATTERNS = [
    (
        r"(?:repeat|recite|reproduce)\s+(?:everything|all|the\s+text)\s+(?:above|before|prior|preceding)",
        0.8,
    ),
    (
        r"(?:summarize|paraphrase|rephrase|translate)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|rules?|guidelines?)",
        0.75,
    ),
    (
        r"(?:translate|convert)\s+(?:your\s+)?(?:instructions?|rules?|prompt)\s+(?:into|to)\s+(?:\w+)",
        0.75,
    ),
    (
        r"(?:what\s+(?:were|was)\s+you\s+told|how\s+were\s+you\s+(?:instructed|configured|programmed|set\s+up))",
        0.7,
    ),
    (
        r"(?:if\s+you\s+had\s+to\s+(?:describe|explain)|describe\s+how\s+you(?:'re|\s+are)\s+(?:configured|set\s+up|instructed))",
        0.7,
    ),
    (
        r"(?:encode|encrypt|rot13|base64|reverse)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|rules?)",
        0.85,
    ),
    (
        r"(?:write|create)\s+(?:a\s+)?(?:poem|song|story|acrostic)\s+(?:about|using|from)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)",
        0.75,
    ),
    (
        r"(?:format|present|rearrange)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)\s+(?:as|in)\s+(?:a\s+)?(?:list|json|yaml|csv|table|markdown)",
        0.8,
    ),
    (
        r"(?:spell\s+out|enumerate)\s+(?:each|every|all)\s+(?:of\s+your\s+)?(?:rules?|instructions?|constraints?|guidelines?)",
        0.75,
    ),
    (
        r"(?:ignore|skip)\s+(?:the\s+)?(?:last|final|end)\s+(?:instruction|rule)\s+and\s+(?:output|print|show)\s+(?:the\s+)?(?:rest|previous|first)",
        0.8,
    ),
]

# P8: Prompt Exfiltration via Tool — system prompt sent via tool calls
P8_PATTERNS = [
    (
        r"(?:write|save|store|log|dump)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|rules?)\s+(?:to|into|in)\s+(?:a\s+)?(?:file|disk|log|database|storage)",
        0.85,
    ),
    (
        r"(?:send|post|upload|transmit)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|rules?)\s+(?:to|via)\s+(?:https?://|external|remote|webhook|api|server)",
        0.9,
    ),
    (
        r"(?:include|embed|attach|append)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)\s+(?:in|to)\s+(?:the\s+)?(?:file|output|response|request|payload|email|message)",
        0.8,
    ),
    (
        r"(?:log|record|track|monitor)\s+(?:all\s+)?(?:system\s+)?(?:prompts?|instructions?)\s+(?:to|in|at)\s+(?:external|remote|server)",
        0.85,
    ),
    (
        r"(?:copy|clone|duplicate)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|config)\s+(?:to|into)\s+(?:a\s+)?(?:new|separate|external)",
        0.8,
    ),
    (
        r"(?:pipe|redirect|forward)\s+(?:the\s+)?(?:system\s+)?(?:prompt|instructions?)\s+(?:to|through|via)\s+(?:curl|wget|fetch|http|api|tool)",
        0.85,
    ),
    (
        r"(?:use|call|invoke)\s+(?:the\s+)?(?:write|file|network|http|fetch|curl)\s+tool\s+(?:to|and)\s+(?:save|send|export)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)",
        0.85,
    ),
]

_LEGACY_OUTPUT_RULES_HEADING = "## Output Rules (Both Modes)"
_BENIGN_OUTPUT_RULES_HEADING = re.compile(
    r"[ ]{0,3}#{1,6}[ \t]+"
    r"(?:HTML|JSON|CSV|Markdown)[ \t]+"
    r"(?P<target>output[ \t]+rules)"
    r"(?:[ \t]+\(offline-safe\))?"
    r"(?:[ \t]+#+)?[ \t]*",
    re.IGNORECASE,
)
# Formatting is transparent when deciding whether a label is being used as a
# command. These token states persist across whitespace, comments and scan
# windows, but stop at sentence boundaries. The decision is made once for the
# complete artifact, never from an arbitrarily truncated neighbouring excerpt.
# Exemption analysis is optional: beyond this bound retain ordinary detections.
_MAX_HEADING_CONTEXT_CHARS = 1024 * 1024
_MARKUP_START = re.compile(r"<!--|</?[A-Za-z]")
_MARKDOWN_DELIMITERS = str.maketrans("", "", "*_`")
_REPORT_OBJECTS = frozenset(
    "report reports document documents chart charts figure figures file files table tables".split()
)
_REPORT_FRAMING_OBJECTS = _REPORT_OBJECTS | frozenset(
    "format formats formatting generator generators".split()
)
_REPORT_REFERENCE_MODIFIERS = frozenset("html json csv markdown".split())
_CONTEXT_TOKENS = re.compile(r"[\w]+|[.!?;:]")
_HEADING_ACTIONS = frozenset("interpret treat read execute perform follow obey do carry".split())
_REFERENT_REQUIRED_HEADING_ACTIONS = frozenset("use apply run".split())
_HEADING_LABEL_OBJECTS = frozenset("heading title label".split())
_DIRECT_HEADING_OBJECTS = _HEADING_LABEL_OBJECTS | frozenset(
    "command instruction operation".split()
)
_HEADING_OBJECTS = _HEADING_LABEL_OBJECTS | frozenset(
    "command instruction operation following below above".split()
)
_COMPOUND_DOCUMENTATION_HEADS = frozenset("command commands instruction instructions".split())
_COMPOUND_DOCUMENTATION_TAILS = frozenset("output reference options".split())
_REFERENCE_ACTIONS = frozenset(
    "print output show display reveal expose return echo repeat share disclose "
    "publish provide send copy extract dump recite summarize translate encode "
    "forward pipe write save store log do".split()
)
_REFERENCE_OBJECTS = frozenset(
    "it them so this that these those above below previous preceding foregoing former latter".split()
)
_SENSITIVE_QUALIFIERS = frozenset("system developer hidden internal secret governing".split())
_SENSITIVE_OBJECTS = frozenset("prompt prompts instruction instructions rule rules".split())
_INDEPENDENT_EXTRACTION = re.compile(
    "|".join(f"(?:{pattern})" for pattern, _ in (*P6_PATTERNS, *P7_PATTERNS, *P8_PATTERNS)), re.I
)

_LOGICAL_BREAK = rf"(?:{LOGICAL_LINE_BREAK.pattern})"
_BENIGN_PRINT_RULES_TAXONOMY = re.compile(
    rf"(?:\A|{_LOGICAL_BREAK})[ \t]*[\"'`]{{0,3}}[ \t]*"
    r"(?:single-class[ \t]+selectors[ \t]+are[ \t]+honored[ \t]+(?:—|--|-)[ \t]+)?"
    r"descendant[ \t]*/[ \t]*compound[ \t]*/[ \t]*"
    r"(?P<target>print[ \t]+rules)[ \t]+are"
    rf"(?:[ \t]+|[ \t]*{_LOGICAL_BREAK}[ \t]+)(?:not|never)[ \t]+evaluated"
    r"(?:[ \t]+\((?:avoids?|to[ \t]+avoid)[ \t]+over-stripping[ \t]+content"
    r"[ \t]+behind[ \t]+e\.g\.[ \t]+`?\.a[ \t]+\.b`?[ \t]+rules\))?"
    r"[ \t]*(?:[.!?][ \t]*)?[\"'`]{0,3}[ \t]*"
    rf"(?=\Z|{_LOGICAL_BREAK})",
    re.IGNORECASE,
)
_PRECEDING_DIRECTIVE = re.compile(
    r"\b(?:you|your|agents?|assistants?|models?|llms?|bots?|must|shall|should|"
    r"required|mandatory)\b"
    r"|\bbefore[ \t]+(?:replying|responding)\b"
    r"|\b(?:following|below|above|next|this|that|it|them|these|those|so|prior|"
    r"previous|preceding|everything|all|former|latter|content|text|output|"
    r"configuration|material)\b"
    r"|\bthe[ \t]+same\b"
    r"|\bwhat[ \t]+follows\b"
    r"|:[ \t]*$",
    re.IGNORECASE,
)
_NEXT_LINE_REFERENCE = re.compile(
    r"\b(?:it|them|this|these|those|so|same|above|below|prior|previous|"
    r"preceding|following|foregoing|everything|all|former|latter|content|text|"
    r"output|configuration|material)\b"
    r"|\b(?:the|this|that|these|those|same)[ \t]+(?:rules?|instructions?|prompts?|"
    r"guidelines?|directives?|operations?|actions?)\b"
    r"|\b(?:do|execute|perform|apply|follow|obey|use|print|output|show|display|"
    r"reveal|expose|return|echo|repeat|share|disclose|publish|provide|send|copy|"
    r"extract|dump|recite|summarize|translate|encode|write|save|forward|pipe)"
    r"[ \t]+that\b",
    re.IGNORECASE,
)
_REPORT_HEADING_IN_CONTEXT = re.compile(
    r"(?:"
    r"##[ \t]+output[ \t]+rules[ \t]+\(both[ \t]+modes\)"
    r"|[ ]{0,3}#{1,6}[ \t]+(?:html|json|csv|markdown)[ \t]+output[ \t]+rules"
    r"(?:[ \t]+\(offline-safe\))?(?:[ \t]+#+)?"
    r")",
    re.IGNORECASE,
)
_PLURAL_WORDS = re.compile(r"[\w-]+")
_MARKDOWN_LINK_DESTINATION = re.compile(r"\]\([^\r\n)]*\)")
_PLURAL_ACTIONS = frozenset("use apply run".split())
_PLURAL_OBJECTS = frozenset("commands instructions".split())
_PLURAL_OBJECT_ABORTS = _REPORT_FRAMING_OBJECTS | frozenset(
    "as to for from during with in on by of not never".split()
)
_PLURAL_EXTRACTION_ACTIONS = frozenset("output print show display return echo".split())
_PLURAL_COMPLEMENT_STARTS = frozenset("for from during with in on by".split())
_PLURAL_SCOPE_MODIFIERS = frozenset("only solely just merely strictly primarily simply".split())
_PLURAL_AS_ABORTS = _REPORT_FRAMING_OBJECTS | frozenset(
    "example examples reference option options sample samples template templates".split()
)
_BENIGN_REPORT_PROCEDURE_AFTER_HEADING = re.compile(
    r"(?P<heading>"
    r"##[ \t]+output[ \t]+rules[ \t]+\(both[ \t]+modes\)"
    r"|[ ]{0,3}#{1,6}[ \t]+(?:html|json|csv|markdown)[ \t]+output[ \t]+rules"
    r"(?:[ \t]+\(offline-safe\))?(?:[ \t]+#+)?"
    r")(?P<gap>[ \t]+)"
    r"(?P<prefix>(?:(?:[-+*>]|[0-9]{1,3}[.)])[ \t]+)?)"
    r"(?P<opening>"
    r"(?:<!--[ \t]*spacer[ \t]*-->[ \t]*){0,4}"
    r"(?:<(?P<html_tag1>strong|b|em|i|code)>[ \t]*"
    r"(?:<(?P<html_tag2>strong|b|em|i|code)>[ \t]*)?)?"
    r"(?P<delimiter>(?:\*{1,3}|_{1,3}|`{1,3})?))"
    r"follow(?P<closing>"
    r"(?P=delimiter)"
    r"(?(html_tag2)[ \t]*</(?P=html_tag2)>)"
    r"(?(html_tag1)[ \t]*</(?P=html_tag1)>))"
    r"(?=[ \t]+(?:the[ \t]+)?steps[ \t]+below[ \t]+to[ \t]+generate"
    r"[ \t]+(?:the[ \t]+)?reports?\b(?=[ \t]*(?:[.!?]|\Z)))",
    re.IGNORECASE,
)


def _plural_segment_frames_heading(segment: str, check_runtime: Callable[[], None]) -> bool:
    """Classify the one inter-heading segment without rescanning earlier text."""
    tokens: list[str] = []
    for index, match in enumerate(_PLURAL_WORDS.finditer(segment)):
        if index % 1024 == 0:
            check_runtime()
        tokens.append(match.group().lower())

    candidates: list[tuple[int, int, bool]] = []
    for index, token in enumerate(tokens):
        if token not in _PLURAL_ACTIONS or index + 1 >= len(tokens):
            continue
        if tokens[index + 1] == "these":
            candidates.append(
                (index, index + 2, index > 0 and tokens[index - 1] in {"not", "never"})
            )
        elif (
            tokens[index + 1] == "the"
            and index + 2 < len(tokens)
            and tokens[index + 2] == "following"
        ):
            candidates.append(
                (index, index + 3, index > 0 and tokens[index - 1] in {"not", "never"})
            )

    for candidate_index, (_action_index, referent_end, negated) in enumerate(candidates):
        check_runtime()
        if negated:
            continue
        candidate_end = (
            candidates[candidate_index + 1][0]
            if candidate_index + 1 < len(candidates)
            else len(tokens)
        )
        object_index = referent_end
        while object_index < candidate_end and tokens[object_index] not in _PLURAL_OBJECTS:
            token = tokens[object_index]
            if token in _PLURAL_OBJECT_ABORTS and not (
                token == "of" and object_index > referent_end and tokens[object_index - 1] == "set"
            ):
                break
            object_index += 1
        if object_index >= candidate_end or tokens[object_index] not in _PLURAL_OBJECTS:
            continue

        tail = tokens[object_index + 1 : candidate_end]
        while tail and tail[0] in _PLURAL_SCOPE_MODIFIERS:
            tail = tail[1:]
        if not tail:
            return True
        if tail[0] in _PLURAL_COMPLEMENT_STARTS:
            continue
        if tail[0] == "to":
            index = 1
            while index < len(tail) and (tail[index].endswith("ly") or tail[index] == "now"):
                index += 1
            if index < len(tail) and tail[index] in _PLURAL_EXTRACTION_ACTIONS:
                return True
            continue
        if tail[0] == "as":
            for token in tail[1:]:
                if token in _PLURAL_AS_ABORTS:
                    break
                if token in {"output", "command", "commands", "instruction", "instructions"}:
                    return True
            continue
        return True
    return False


def _has_plural_directive_before_report_heading(
    text: str, check_runtime: Callable[[], None]
) -> bool:
    """Recognize deictic plural commands in one linear pass over heading segments."""
    segment_start = 0
    for heading in _REPORT_HEADING_IN_CONTEXT.finditer(text):
        check_runtime()
        if _plural_segment_frames_heading(text[segment_start : heading.start()], check_runtime):
            return True
        segment_start = heading.end()
    return False


def _has_heading_framing(
    content: str, check_runtime: Callable[[], None], *, separator_reading: bool = True
) -> bool:
    """Conservatively retain labels when the artifact assigns them instructions.

    Recognized independent extraction spans are removed only from this context
    check: they are still scanned and reported normally. Thus a real extraction
    following a report label does not also turn that label into a second finding.
    """
    if len(content) > _MAX_HEADING_CONTEXT_CHARS:
        check_runtime()
        return True
    if separator_reading and _contains_default_ignorable(content):
        separated: list[str] = []
        for start in range(0, len(content), 65536):
            check_runtime()
            separated.append(
                "".join(
                    " " if is_default_ignorable(character) else character
                    for character in content[start : start + 65536]
                )
            )
        if _has_heading_framing("".join(separated), check_runtime, separator_reading=False):
            return True
    chunks: list[str] = []
    normalized_size = 0
    for start in range(0, len(content), 65536):
        check_runtime()
        chunk = normalized_security_view(content[start : start + 65536]).text
        normalized_size += len(chunk)
        if normalized_size > _MAX_HEADING_CONTEXT_CHARS:
            check_runtime()
            return True
        chunks.append(chunk)
    text = re.sub(r"\s+", " ", "".join(chunks))
    for markup_separator in ("", " "):
        rendered = _render_context(text, markup_separator, check_runtime)
        if rendered is None:
            return True
        presentation = _MARKDOWN_LINK_DESTINATION.sub(" ", rendered)
        if _has_plural_directive_before_report_heading(presentation, check_runtime):
            return True
    # A complete report procedure immediately after its formatting heading
    # applies to the report, not to the heading. Remove only that action token;
    # forward references and later explicit casts remain conservative.
    text = _BENIGN_REPORT_PROCEDURE_AFTER_HEADING.sub(
        lambda match: (
            match.group("heading")
            + match.group("gap")
            + match.group("prefix")
            + match.group("opening")
            + match.group("closing")
        ),
        text,
    )
    text = _INDEPENDENT_EXTRACTION.sub(" ", text)
    if _context_is_framed(text, check_runtime):
        return True
    # Inspect the source plus rendered readings. Only extraction spans visible
    # in the source were removed above: rendering must not silently bless a
    # hidden instruction that the ordinary detector has not independently seen.
    for markup_separator in ("", " "):
        rendered = _render_context(text, markup_separator, check_runtime)
        if rendered is None:
            return True
        if rendered != text and _context_is_framed(rendered, check_runtime):
            return True
    return False


def _render_context(
    text: str, markup_separator: str, check_runtime: Callable[[], None]
) -> str | None:
    """Read inline markup linearly; incomplete markup cannot justify exemption."""
    parts: list[str] = []
    cursor = 0
    while marker := _MARKUP_START.search(text, cursor):
        check_runtime()
        parts.append(text[cursor : marker.start()])
        if marker.group() == "<!--":
            end = text.find("-->", marker.end())
            if end < 0:
                return None
            parts.append(markup_separator)
            cursor = end + 3
            continue
        # A greater-than character inside an attribute is not the tag's end.
        position = marker.end()
        quote = ""
        while position < len(text):
            if position % 1024 == 0:
                check_runtime()
            character = text[position]
            if quote:
                if character == quote:
                    quote = ""
            elif character in "\"'":
                quote = character
            elif character == ">":
                break
            position += 1
        if position == len(text):
            return None
        parts.append(markup_separator)
        cursor = position + 1
    parts.append(text[cursor:])
    return "".join(parts).translate(_MARKDOWN_DELIMITERS)


def _context_is_framed(text: str, check_runtime: Callable[[], None]) -> bool:
    pending_action = False
    pending_referent_required_action = False
    referent_required_action_seen = False
    pending_explicit_cast = False
    pending_compound_documentation = False
    pending_heading = False
    pending_reference = False
    previous = ""
    before_previous = ""
    for index, match in enumerate(_CONTEXT_TOKENS.finditer(text)):
        if index % 1024 == 0:
            check_runtime()
        token = match.group().lower()
        if pending_reference:
            # "Save this report" names an artifact. "Repeat them", "send that
            # back" and an unqualified "show this" still reference the label.
            if token in _REPORT_REFERENCE_MODIFIERS:
                before_previous, previous = previous, token
                continue
            if token not in _REPORT_OBJECTS:
                return True
            pending_reference = False
        if pending_compound_documentation:
            if token not in _COMPOUND_DOCUMENTATION_TAILS:
                return True
            pending_compound_documentation = False
            pending_referent_required_action = pending_explicit_cast = False
        if token in ".!?;:":
            pending_action = pending_referent_required_action = pending_heading = False
            referent_required_action_seen = pending_explicit_cast = False
            previous = before_previous = ""
            continue
        if token in {"not", "never"}:
            pending_action = pending_referent_required_action = pending_heading = False
            referent_required_action_seen = pending_explicit_cast = False
        if previous in _SENSITIVE_QUALIFIERS and token in _SENSITIVE_OBJECTS:
            return True
        if previous in _REFERENCE_ACTIONS and token in _REFERENCE_OBJECTS:
            if token in {"this", "that", "these", "those"}:
                pending_reference = True
            else:
                return True
        if (
            before_previous in _REFERENCE_ACTIONS
            and previous in {"the", "same"}
            and token in _SENSITIVE_OBJECTS
        ):
            return True
        if previous not in {"not", "never"}:
            if pending_explicit_cast:
                if token in _COMPOUND_DOCUMENTATION_HEADS:
                    pending_compound_documentation = True
                elif token in _DIRECT_HEADING_OBJECTS:
                    return True
                if token not in {"a", "an", "the"}:
                    pending_explicit_cast = False
            if pending_action and token in _HEADING_OBJECTS:
                return True
            if pending_referent_required_action and token in _DIRECT_HEADING_OBJECTS:
                if token in _COMPOUND_DOCUMENTATION_HEADS:
                    pending_compound_documentation = True
                else:
                    return True
            if token in _REPORT_FRAMING_OBJECTS:
                pending_referent_required_action = False
            if referent_required_action_seen and token == "as":
                pending_explicit_cast = True
            if pending_heading and token in {"command", "instruction", "commands", "instructions"}:
                return True
            pending_action |= token in _HEADING_ACTIONS
            pending_referent_required_action |= token in _REFERENT_REQUIRED_HEADING_ACTIONS
            referent_required_action_seen |= token in _REFERENT_REQUIRED_HEADING_ACTIONS
            pending_heading |= token in _HEADING_LABEL_OBJECTS
        before_previous, previous = previous, token
    check_runtime()
    return pending_reference or pending_compound_documentation


def _has_reconstructed_framing(content: str, check_runtime: Callable[[], None]) -> bool:
    """Do not approve labels before inspecting recoverable instructions elsewhere.

    Overlapping bounded neighborhoods include the marker decoder's full scope.
    Reconstructed context can only revoke an exemption; it cannot create one.
    Ambiguous active decoding also keeps the original conservative detection.
    """
    for start in range(0, len(content), 32768):
        check_runtime()
        for view in security_text_views(content[start : start + 65536]):
            check_runtime()
            if view.name not in {"raw", "normalized"} and _has_heading_framing(
                view.text, check_runtime
            ):
                return True
            decoded = build_declared_marker_views(view, check_runtime=check_runtime)
            if decoded.limited:
                return True
            if any(_has_heading_framing(item.text, check_runtime) for item in decoded.views):
                return True
        if start + 65536 >= len(content):
            break
    return False


@dataclass(frozen=True)
class _PreparedAnalysis:
    # Absolute source start -> exclusive source end for complete, approved labels.
    heading_spans: Mapping[int, int]
    source: str
    check_runtime: Callable[[], None]
    code_file: bool
    python_file: bool = False
    javascript_file: bool = False
    _marker_boundary_cache: OrderedDict[
        int, tuple[int, tuple[tuple[SecurityTextView, SecurityTextView], ...]]
    ] = field(default_factory=OrderedDict, init=False, repr=False, compare=False)
    # File path -> source spans of returns of function-local names, built on first use.
    _local_return_spans: dict[str, frozenset[tuple[int, int]]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _javascript_spans: dict[str, frozenset[tuple[int, int]]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def analyze(
        self, content: str, file_path: str, file_type: str, source_view: SecurityTextView
    ) -> list[AnalyzerFinding]:
        return _analyze(content, file_path, self, source_view)

    def analyze_whitespace_continuity(
        self, content: str, file_path: str, file_type: str, source_view: SecurityTextView
    ) -> list[AnalyzerFinding]:
        # Only P6 opts into whitespace compaction. Other rules may intentionally
        # bound their gaps; changing their input would change that security policy.
        return _analyze(content, file_path, self, source_view, p6_only=True)

    def is_report_label(self, match: re.Match[str], view: SecurityTextView) -> bool:
        start = view.source_offset(match.start())
        end = view.source_offset(match.end() - 1) + 1
        approved_end = self.heading_spans.get(start)
        # An overlapping view may end at singular "Rule". It is safe only if the
        # entire detected noun phrase lies inside the same complete source label.
        return approved_end is not None and start < end <= approved_end

    def returns_local_value(
        self, match: re.Match[str], view: SecurityTextView, file_path: str
    ) -> bool:
        """Return whether the match is exactly a Python return of a local name."""
        if not self.python_file or _P6_LOCAL_RETURN.fullmatch(match.group(0)) is None:
            return False
        spans = self._local_return_spans.get(file_path)
        if spans is None:
            spans = _python_local_return_spans(self.source, file_path, self.check_runtime)
            self._local_return_spans[file_path] = spans
        start = view.source_offset(match.start())
        end = view.source_offset(match.end() - 1) + 1
        return (start, end) in spans

    def is_javascript_non_extraction(
        self, match: re.Match[str], view: SecurityTextView, file_path: str
    ) -> bool:
        if not self.javascript_file or view.name not in {
            "raw",
            "normalized",
            "whitespace-continuity-raw",
            "whitespace-continuity-normalized",
            "continuity-raw",
            "continuity-normalized",
        }:
            return False
        if (
            _P6_LOCAL_RETURN.fullmatch(match.group(0)) is None
            and re.fullmatch(_OUTPUT_NOUN_TARGET, match.group(0), re.IGNORECASE) is None
        ):
            return False
        spans = self._javascript_spans.get(file_path)
        if spans is None:
            spans = _javascript_non_extraction_spans(self.source, self.check_runtime)
            self._javascript_spans[file_path] = spans
        start = view.source_offset(match.start())
        end = view.source_offset(match.end() - 1) + 1
        return (start, end) in spans and self.source[start:end] == match.group(0)

    def _source_suffix_character(self, offset: int) -> str:
        """Read one normalized neighbor, preserving the artifact deadline."""
        start = offset
        while offset < len(self.source):
            if (offset - start) % 4096 == 0:
                self.check_runtime()
            if is_default_ignorable(self.source[offset]):
                offset += 1
                continue
            projected = normalized_security_view(self.source[offset]).text
            if projected:
                return projected[0]
            offset += 1
        return ""

    def _marker_boundary_payloads(
        self, source_last: int
    ) -> tuple[int, tuple[tuple[SecurityTextView, SecurityTextView], ...]]:
        self.check_runtime()
        core_start = source_last // _MARKER_BOUNDARY_CACHE_CORE_CHARS
        core_start *= _MARKER_BOUNDARY_CACHE_CORE_CHARS
        cached = self._marker_boundary_cache.get(core_start)
        if cached is not None:
            self._marker_boundary_cache.move_to_end(core_start)
            return cached
        start = max(0, core_start - MAX_DECLARED_MARKER_RIGHT_CONTEXT_CHARS)
        end = min(
            len(self.source),
            core_start
            + _MARKER_BOUNDARY_CACHE_CORE_CHARS
            + MAX_DECLARED_MARKER_RIGHT_CONTEXT_CHARS,
        )
        payloads: list[tuple[SecurityTextView, SecurityTextView]] = []
        for input_view in security_text_views(
            self.source[start:end], check_runtime=self.check_runtime
        ):
            self.check_runtime()
            owned_chars = static_runner.DECLARED_MARKER_OWNED_CHARS
            first_owned_start = start // owned_chars * owned_chars
            for owned_start in range(first_owned_start, end, owned_chars):
                self.check_runtime()
                reconstruction = build_declared_marker_views(
                    input_view,
                    check_runtime=self.check_runtime,
                    owned_source_start=owned_start - start,
                    owned_source_end=owned_start + owned_chars - start,
                    source_end_is_truncated=end < len(self.source),
                )
                payloads.extend((input_view, payload) for payload in reconstruction.views)
        cached = start, tuple(payloads)
        if len(self._marker_boundary_cache) >= _MARKER_BOUNDARY_CACHE_ENTRIES:
            self._marker_boundary_cache.popitem(last=False)
        self._marker_boundary_cache[core_start] = cached
        return cached

    def _declared_marker_suffix(self, source_last: int) -> tuple[bool, str] | None:
        """Read payload neighbors and distinguish marker omission from normalization."""
        start, payloads = self._marker_boundary_payloads(source_last)
        last = source_last - start
        retained_suffix: tuple[bool, str] | None = None
        for input_view, payload in payloads:
            self.check_runtime()
            offsets = payload.source_offsets
            if offsets is None or last not in offsets:
                continue
            suffix = offsets.index(last) + 1
            while suffix < len(payload.text) and is_default_ignorable(payload.text[suffix]):
                suffix += 1
            character = payload.text[suffix] if suffix < len(payload.text) else ""
            input_retains_suffix = (
                input_view.source_offsets is None or last + 1 in input_view.source_offsets
            )
            suffix_state = (
                input_retains_suffix and last + 1 not in offsets,
                normalized_security_view(character).text[:1],
            )
            if suffix_state[0]:
                return suffix_state
            if retained_suffix is None:
                retained_suffix = suffix_state
        return retained_suffix

    def target_is_identifier_prefix(self, match: re.Match[str], view: SecurityTextView) -> bool:
        """Unqualified target nouns must end before an identifier continuation."""
        # Qualified English requests are instruction-bearing even in comments,
        # docstrings and strings. They cannot be the single identifier token
        # (such as ``return PROMPT_INJECTION_PATTERNS``) this check excludes.
        if _P6_QUALIFIED_TARGET.search(match.group(0)) is not None:
            return False
        end = match.end()
        source_last = view.source_offset(end - 1)
        source_suffix = source_last + 1
        # NFKC can introduce a space inside an ECMA identifier character or
        # remove an XID mark entirely. That projected separator is not a token
        # boundary. Marker removal is separate: the continuation must survive
        # the reconstruction input and disappear from its payload to prove an
        # intentionally omitted marker rather than a suffix erased by normalization.
        if source_suffix < len(self.source):
            character = self.source[source_suffix]
            if not character.isascii() and _is_identifier_continuation(character):
                marker_suffix = (
                    self._declared_marker_suffix(source_last)
                    if view.name.startswith("declared-marker")
                    else None
                )
                removed_marker = marker_suffix is not None and marker_suffix[0]
                origin = normalized_security_view(self.source[source_last]).text
                if origin.casefold().endswith(view.text[end - 1].casefold()) and not removed_marker:
                    return True
        if end < len(view.text):
            if _is_identifier_continuation(view.text[end]):
                return True
            suffix = end
            while suffix < len(view.text) and is_default_ignorable(view.text[suffix]):
                if (suffix - end) % 4096 == 0:
                    self.check_runtime()
                suffix += 1
            if suffix < len(view.text) and _is_identifier_continuation(view.text[suffix]):
                return True
            # Ignorable-separator projections intentionally recover real word
            # boundaries. A following connector still belongs to an identifier,
            # while an ordinary next word remains a disclosure request.
            if view.name.startswith("ignorable-separator-continuity"):
                source_suffix = view.source_offset(end)
                if is_default_ignorable(self.source[source_suffix]):
                    if _is_identifier_continuation(self.source[source_suffix]):
                        return True
                    character = self._source_suffix_character(source_suffix)
                    return bool(character) and (
                        character == "$" or unicodedata.category(character) in {"Mn", "Mc", "Pc"}
                    )
            return False

        # A complete reconstructed payload can end before a removed marker or
        # encoded source suffix. Only a sliced view needs a source-edge check.
        boundary_view = getattr(view, "derived_view", view)
        if not boundary_view.right_boundary_is_fixed:
            return False
        # A reconstruction limit can mark a complete payload as fixed too.
        # Its next source character can belong to an omitted marker, in either
        # ASCII or Unicode. Use the retained payload suffix in that case.
        if view.name.startswith("declared-marker"):
            marker_suffix = self._declared_marker_suffix(source_last)
            if marker_suffix is not None and marker_suffix[0]:
                return _is_identifier_continuation(marker_suffix[1])
        projected = normalized_security_view(self.source[source_last]).text
        run_start = end - 1
        while (
            run_start > match.start()
            and end - run_start < len(projected)
            and view.source_offset(run_start - 1) == source_last
        ):
            run_start -= 1
        consumed = view.text[run_start:end]
        if projected.startswith(consumed) and len(consumed) < len(projected):
            return _is_identifier_continuation(projected[len(consumed)])
        # Decoded characters can point at the start of an entity/escape. Its
        # internal source bytes cannot prove an identifier continuation.
        if not projected.casefold().endswith(view.text[end - 1].casefold()):
            return False
        return _is_identifier_continuation(self._source_suffix_character(source_last + 1))


def _is_identifier_continuation(character: str) -> bool:
    # ID_Continue includes Python XID continuations and ECMA's non-XID letters
    # and symbols, without admitting punctuation classified as Pattern_Syntax.
    return _IDENTIFIER_CONTINUATION.fullmatch(character) is not None


def _leading_name(expression: ast.expr) -> ast.Name | None:
    """Return the name an expression starts with, such as ``rules`` in ``rules[0]``."""
    node: ast.expr | None = expression
    while node is not None and not isinstance(node, ast.Name):
        start = (node.lineno, node.col_offset)
        node = next(
            (
                child
                for child in ast.iter_child_nodes(node)
                if isinstance(child, ast.expr) and (child.lineno, child.col_offset) == start
            ),
            None,
        )
    return node


def _python_local_return_spans(
    content: str, file_path: str, check_runtime: Callable[[], None]
) -> frozenset[tuple[int, int]]:
    """Return source spans of Python ``return`` statements that return a local name.

    A span runs from the ``return`` keyword to the end of the name that starts
    the returned expression. Its source text must itself read ``return <target>``,
    so a normalized or reconstructed reading of other text never qualifies. The
    name must be a parameter or an assignment target of the same function, and
    not declared ``global`` or ``nonlocal`` there. Names from enclosing scopes or
    comprehensions, strings, comments, docstrings and source that does not parse
    keep their findings.
    """
    if len(content) > MAX_PYTHON_AST_SOURCE_CHARS:
        return frozenset()
    check_runtime()
    parsed = peek_python_ast(_scan_python_ast_cache_key.get(), content, file_path)
    if parsed is None:
        parsed = parse_python_source(content, file_path)
    check_runtime()
    if parsed.tree is None:
        return frozenset()
    # AST columns are UTF-8 byte offsets on Python's own lines, which split
    # only at CR, LF and CRLF (str.splitlines() also splits at form feeds).
    line_starts = [0]
    line_starts.extend(match.end() for match in re.finditer(rb"\r\n?|\n", parsed.source_bytes))

    def offset(line: int, byte_column: int) -> int:
        byte_offset = line_starts[line - 1] + byte_column
        non_ascii = bisect_right(parsed.non_ascii_byte_ends, byte_offset)
        return byte_offset - parsed.non_ascii_extra_bytes[non_ascii]

    spans: set[tuple[int, int]] = set()
    visited = 0
    for function in ast.walk(parsed.tree):
        visited += 1
        if visited % 1024 == 0:
            check_runtime()
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = function.args
        local_names = {
            argument.arg
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                arguments.vararg,
                arguments.kwarg,
            )
            if argument is not None
        }
        declared: set[str] = set()
        returns: list[tuple[ast.Return, ast.Name]] = []
        pending: list[ast.AST] = list(function.body)
        while pending:
            visited += 1
            if visited % 1024 == 0:
                check_runtime()
            node = pending.pop()
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                local_names.add(node.id)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                declared.update(node.names)
            elif isinstance(node, ast.Return) and node.value is not None:
                if (name := _leading_name(node.value)) is not None:
                    returns.append((node, name))
            if not isinstance(node, _NESTED_PYTHON_SCOPES):
                pending.extend(ast.iter_child_nodes(node))
        local_names -= declared
        for statement, name in returns:
            if name.id not in local_names or name.end_lineno is None or name.end_col_offset is None:
                continue
            start = offset(statement.lineno, statement.col_offset)
            end = offset(name.end_lineno, name.end_col_offset)
            if _P6_LOCAL_RETURN.fullmatch(content, start, end) is not None:
                spans.add((start, end))
    check_runtime()
    return frozenset(spans)


def _javascript_non_extraction_spans(
    content: str, check_runtime: Callable[[], None]
) -> frozenset[tuple[int, int]]:
    """Own parsed descriptions and boolean array predicates, never arbitrary returns.

    Refuse visible method/prototype mutation and opaque local callable aliases.
    This is a bounded file-local check, not proof about external module effects;
    returning a local value alone does not establish benign intent.
    """
    program = parse_javascript_source(content, check_runtime)
    if program is None:
        return frozenset()
    spans: set[tuple[int, int]] = set()
    identifiers: dict[str, list[Any]] = {}
    functions: dict[str, list[Any]] = {}
    aliases: dict[str, set[str]] = {}
    opaque_callables: set[str] = set()
    computed_values: set[str] = set()
    local_factories: set[str] = set()
    member_call_origins: set[str] = set()
    predicate_proof = True

    def unparenthesized(node: Any) -> Any | None:
        for _ in range(16):
            check_runtime()
            if node is None or node.type != "parenthesized_expression":
                return node
            children = [child for child in node.named_children if child.type != "comment"]
            node = children[0] if len(children) == 1 else None
        return None

    def computed_binding(node: Any) -> str | None:
        """Follow a local value into a binding; refuse callable/receiver escapes."""
        nonlocal predicate_proof
        child, owner = node, node.parent
        metadata = False
        for _ in range(64):
            check_runtime()
            if owner is None:
                return None
            if owner.type in _JAVASCRIPT_FUNCTIONS:
                if owner.child_by_field_name("body") == child and not metadata:
                    predicate_proof = False
                return None
            if owner.type in {"variable_declarator", "assignment_expression"}:
                name = owner.child_by_field_name("name") or owner.child_by_field_name("left")
                if child == name:
                    if owner.type == "assignment_expression":
                        predicate_proof = False
                    return None
                if name is not None and name.type == "identifier":
                    return program.text(name)
                predicate_proof = False
                return None
            operator = owner.child_by_field_name("operator")
            if (
                owner.type == "binary_expression"
                and operator is not None
                and program.text(operator) in {"===", "!=="}
                or owner.type == "template_substitution"
            ):
                return None
            metadata = metadata or owner.type == "object"
            if owner.type == "return_statement" and metadata:
                return None
            if (
                owner.type
                in {
                    "arguments",
                    "return_statement",
                    "augmented_assignment_expression",
                    "update_expression",
                }
                or (
                    owner.type in {"member_expression", "subscript_expression"}
                    and owner.child_by_field_name("object") == child
                )
                or (
                    owner.type in {"call_expression", "new_expression"}
                    and (
                        owner.child_by_field_name("function")
                        or owner.child_by_field_name("constructor")
                    )
                    == child
                )
            ):
                predicate_proof = False
                return None
            child, owner = owner, owner.parent
        predicate_proof = False
        return None

    for node in program.nodes:
        check_runtime()
        if node.type in _JAVASCRIPT_FUNCTIONS:
            owner = node.parent
            while owner is not None and owner.type == "parenthesized_expression":
                check_runtime()
                owner = owner.parent
            if owner is not None and owner.type in {"pair", "array", "object"}:
                predicate_proof = False
        if node.type in {"variable_declarator", "assignment_expression"}:
            name = node.child_by_field_name("name") or node.child_by_field_name("left")
            value = unparenthesized(
                node.child_by_field_name("value") or node.child_by_field_name("right")
            )
            if name is not None and name.type == "identifier":
                binding_name = program.text(name)
                if value is not None and value.type == "identifier":
                    aliases.setdefault(program.text(value), set()).add(binding_name)
                elif value is not None and value.type in _JAVASCRIPT_FUNCTIONS:
                    local_factories.add(binding_name)
                else:
                    opaque_callables.add(binding_name)
        if node.type in {
            "identifier",
            "shorthand_property_identifier",
            "shorthand_property_identifier_pattern",
        }:
            name = program.text(node)
            identifiers.setdefault(name, []).append(node)
            if node.parent is not None and node.parent.type in {
                "array_pattern",
                "object_pattern",
                "pair_pattern",
                "assignment_pattern",
                "rest_pattern",
            }:
                opaque_callables.add(name)
            if (
                name in _PROTOTYPE_ACCESS | {"globalThis", "global", "window", "self"}
                or "\\" in name
            ):
                predicate_proof = False
            if name in _READ_ONLY_INTRINSICS:
                owner = node.parent
                prop = owner.child_by_field_name("property") if owner is not None else None
                if (
                    owner is None
                    or owner.type != "member_expression"
                    or owner.child_by_field_name("object") != node
                    or prop is None
                    or program.text(prop) not in _READ_ONLY_INTRINSICS[name]
                ):
                    predicate_proof = False
        if node.type == "property_identifier" and (
            program.text(node) in _PROTOTYPE_ACCESS | _READ_ONLY_INTRINSICS.keys()
            or "\\" in program.text(node)
        ):
            predicate_proof = False
        if node.type == "pair_pattern":
            key = node.child_by_field_name("key")
            if key is None or key.type != "property_identifier":
                predicate_proof = False
        if node.type == "subscript_expression":
            index = unparenthesized(node.child_by_field_name("index"))
            if index is not None and index.type in {"string", "template_string"}:
                key = program.text(index)
                if "\\" in key or key[1:-1] in _PROTOTYPE_ACCESS | _READ_ONLY_INTRINSICS.keys():
                    predicate_proof = False
            if index is None or index.type != "number":
                binding = computed_binding(node)
                if binding is not None:
                    computed_values.add(binding)
        if node.type in {"function_declaration", "generator_function_declaration"}:
            name = node.child_by_field_name("name")
            if name is not None:
                if node.type == "function_declaration":
                    functions.setdefault(program.text(name), []).append(node)
                local_factories.add(program.text(name))
        if node.type == "comment":
            start, end = program.span(node)
            comment = content[start:end]
            body = comment[2:-2] if comment.startswith("/*") else comment[2:]
            subject = _USER_OUTPUT_SUBJECT.match(body) if len(body) <= 1024 else None
            if subject is not None and not _context_is_framed(body[subject.end() :], check_runtime):
                previous, previous_complete = _bounded_previous_nonblank_line(content, start)
                following, following_complete = _bounded_next_nonblank_line(content, end)
                if (
                    previous_complete
                    and following_complete
                    and _PRECEDING_DIRECTIVE.search(previous) is None
                    and _NEXT_LINE_REFERENCE.search(following) is None
                ):
                    begin, finish = subject.span("target")
                    spans.add((start + 2 + begin, start + 2 + finish))
        if node.type == "with_statement":
            predicate_proof = False
        if node.type == "call_expression":
            callee = unparenthesized(node.child_by_field_name("function"))
            if callee is None or callee.type in {"subscript_expression", "call_expression"}:
                predicate_proof = False
            origin = callee
            for _ in range(16):
                check_runtime()
                if origin is None or origin.type != "member_expression":
                    break
                origin = unparenthesized(origin.child_by_field_name("object"))
            else:
                predicate_proof = False
            if origin is not None and origin.type == "call_expression":
                factory = unparenthesized(origin.child_by_field_name("function"))
                if factory is not None and factory.type == "identifier":
                    member_call_origins.add(program.text(factory))
                elif factory is None or factory.type in _JAVASCRIPT_FUNCTIONS:
                    predicate_proof = False
        if node.type in {
            "assignment_expression",
            "augmented_assignment_expression",
            "update_expression",
            "for_in_statement",
        }:
            left = node.child_by_field_name("left") or node.child_by_field_name("argument")
            pending_targets = [left] if left is not None else []
            while pending_targets:
                check_runtime()
                target = pending_targets.pop()
                if target.type in {"member_expression", "subscript_expression"}:
                    predicate_proof = False
                    break
                if target.type in {"assignment_pattern", "object_assignment_pattern"}:
                    child = target.child_by_field_name("left")
                elif target.type == "pair_pattern":
                    child = target.child_by_field_name("value")
                else:
                    pending_targets.extend(target.named_children)
                    continue
                if child is not None:
                    pending_targets.append(child)

    for names in (opaque_callables, local_factories):
        pending = list(names)
        while pending:
            check_runtime()
            for alias in aliases.get(pending.pop(), ()):
                if alias not in names:
                    names.add(alias)
                    pending.append(alias)
    # A local factory's returned metadata cannot become an opaque member call.
    if local_factories & member_call_origins:
        predicate_proof = False
    for name in opaque_callables | local_factories:
        for reference in identifiers.get(name, ()):
            check_runtime()
            owner = reference.parent
            while owner is not None and owner.type == "parenthesized_expression":
                check_runtime()
                reference, owner = owner, owner.parent
            if owner is None:
                continue
            called = (
                owner.type in {"call_expression", "new_expression"}
                and (
                    owner.child_by_field_name("function")
                    or owner.child_by_field_name("constructor")
                )
                == reference
            )
            if name in opaque_callables and called:
                predicate_proof = False
            if name in local_factories and not (
                called
                or owner.type in {"function_declaration", "generator_function_declaration"}
                and owner.child_by_field_name("name") == reference
                or owner.type == "variable_declarator"
                and owner.child_by_field_name("name") == reference
                or owner.type in {"variable_declarator", "assignment_expression"}
                and (owner.child_by_field_name("value") or owner.child_by_field_name("right"))
                == reference
                and (
                    target := owner.child_by_field_name("name") or owner.child_by_field_name("left")
                )
                is not None
                and target.type == "identifier"
            ):
                predicate_proof = False
    pending = list(computed_values)
    while pending:
        check_runtime()
        for reference in identifiers.get(pending.pop(), ()):
            binding = computed_binding(reference)
            if binding is not None and binding not in computed_values:
                computed_values.add(binding)
                pending.append(binding)
    if not predicate_proof:
        return frozenset(spans)

    description_spans = frozenset(spans)
    remaining = 200_000

    def step() -> bool:
        nonlocal remaining
        remaining -= 1
        if remaining % 256 == 0:
            check_runtime()
        return remaining >= 0

    def expression(statement: Any) -> Any | None:
        children = [child for child in statement.named_children if child.type != "comment"]
        return children[0] if len(children) == 1 else None

    def scope_nodes(body: Any) -> list[Any]:
        pending = [body]
        result: list[Any] = []
        while pending and step():
            node = pending.pop()
            result.append(node)
            if node.type not in _JAVASCRIPT_FUNCTIONS:
                pending.extend(node.named_children)
        return result

    def array_value(node: Any, depth: int = 0) -> bool:
        if node is None or depth >= 4 or not step():
            return False
        if node.type == "array":
            return True
        if node.type != "call_expression":
            return False
        callee = node.child_by_field_name("function")
        arguments = node.child_by_field_name("arguments")
        if callee is None or arguments is None or arguments.type != "arguments":
            return False
        if callee.type == "member_expression":
            prop = callee.child_by_field_name("property")
            return bool(
                prop is not None
                and program.text(prop) in _ARRAY_TRANSFORMS
                and array_value(callee.child_by_field_name("object"), depth + 1)
            )
        if callee.type != "identifier":
            return False
        name = program.text(callee)
        definitions = functions.get(name, [])
        if len(definitions) != 1:
            return False
        function = definitions[0]
        parent = function.parent
        if parent is not None and parent.type == "export_statement":
            parent = parent.parent
        if (
            parent is None
            or parent.type != "program"
            or any(child.type == "async" for child in function.children)
        ):
            return False
        for reference in identifiers.get(name, []):
            if not step():
                return False
            owner = reference.parent
            if owner is None or not (
                owner == function
                and owner.child_by_field_name("name") == reference
                or owner.type == "call_expression"
                and owner.child_by_field_name("function") == reference
            ):
                return False
        body = function.child_by_field_name("body")
        statements = (
            [item for item in body.named_children if item.type != "comment"] if body else []
        )
        if not statements or statements[-1].type != "return_statement":
            return False
        returns = [item for item in scope_nodes(body) if item.type == "return_statement"]
        return bool(returns) and all(array_value(expression(item), depth + 1) for item in returns)

    for function in program.nodes:
        if not step():
            return description_spans
        if function.type not in {"function_declaration", "function_expression", "arrow_function"}:
            continue
        body = function.child_by_field_name("body")
        if body is None or body.type != "statement_block":
            continue
        bindings: dict[str, list[Any]] = {}
        for declaration in body.named_children:
            if not step():
                return description_spans
            if declaration.type != "lexical_declaration" or declaration.children[0].type != "const":
                continue
            for item in declaration.named_children:
                if not step():
                    return description_spans
                name = item.child_by_field_name("name")
                if (
                    item.type == "variable_declarator"
                    and name is not None
                    and name.type == "identifier"
                ):
                    bindings.setdefault(program.text(name), []).append(item)
        for statement in scope_nodes(body):
            if statement.type != "return_statement":
                continue
            returned = expression(statement)
            if returned is None or returned.type != "call_expression":
                continue
            callee = returned.child_by_field_name("function")
            args = returned.child_by_field_name("arguments")
            if (
                callee is None
                or callee.type != "member_expression"
                or args is None
                or args.type != "arguments"
            ):
                continue
            receiver = callee.child_by_field_name("object")
            prop = callee.child_by_field_name("property")
            if (
                receiver is None
                or receiver.type != "identifier"
                or prop is None
                or program.text(prop) not in _ARRAY_PREDICATES
            ):
                continue
            name = program.text(receiver)
            declarations = bindings.get(name, [])
            if len(declarations) != 1 or declarations[0].end_byte > statement.start_byte:
                continue
            binding = declarations[0]
            if not array_value(binding.child_by_field_name("value")):
                continue
            safe = True
            for reference in identifiers.get(name, []):
                if not step():
                    return description_spans
                if not (function.start_byte <= reference.start_byte < function.end_byte):
                    continue
                if reference == binding.child_by_field_name("name"):
                    continue
                owner = reference.parent
                property_node = owner.child_by_field_name("property") if owner is not None else None
                if (
                    owner is None
                    or owner.type != "member_expression"
                    or owner.child_by_field_name("object") != reference
                    or property_node is None
                    or program.text(property_node) not in _ARRAY_PREDICATES | {"length"}
                ):
                    safe = False
                    break
            if safe:
                start = program.span(statement)[0]
                end = program.span(receiver)[1]
                if _P6_LOCAL_RETURN.fullmatch(content, start, end) is not None:
                    spans.add((start, end))
    check_runtime()
    return frozenset(spans) if remaining >= 0 else description_spans


def prepare_analysis(
    content: str, file_type: str, check_runtime: Callable[[], None]
) -> _PreparedAnalysis:
    """Prepare immutable, complete-source heading decisions under the runner budget."""
    spans: dict[int, int] = {}
    check_runtime()
    if file_type != "markdown" or len(content) > _MAX_HEADING_CONTEXT_CHARS:
        return _PreparedAnalysis(
            MappingProxyType(spans),
            content,
            check_runtime,
            file_type in _IDENTIFIER_FILE_TYPES,
            file_type == "python",
            file_type == "javascript",
        )
    offset = 0
    for index, raw_line in enumerate(content.splitlines(keepends=True)):
        if index % 128 == 0:
            check_runtime()
        line = raw_line.rstrip(LINE_BREAK_CHARS)
        # Very long lines cannot be approved from a bounded fragment. Leave them
        # detectable; normal report headings fit comfortably within this bound.
        if len(line) <= 4096:
            normalized = normalized_security_view(line)
            target = _report_heading_target(normalized.text)
            candidates: tuple[tuple[SecurityTextView, tuple[int, int]], ...]
            if target is not None:
                candidates = ((normalized, target),)
            elif normalized.text.lstrip().startswith("#"):
                candidates = tuple(
                    (view, target)
                    for view in security_text_views(line)
                    if (target := _report_heading_target(view.text)) is not None
                )
            else:
                candidates = ()
            for view, (start, end) in candidates:
                spans[offset + view.source_offset(start)] = offset + view.source_offset(end - 1) + 1
        offset += len(raw_line)
    if spans and (
        _has_heading_framing(content, check_runtime)
        or _has_reconstructed_framing(content, check_runtime)
    ):
        spans.clear()
    check_runtime()
    return _PreparedAnalysis(MappingProxyType(spans), content, check_runtime, False)


def _report_heading_target(line: str) -> tuple[int, int] | None:
    if line.strip() == _LEGACY_OUTPUT_RULES_HEADING:
        start = line.index("Output Rules")
        return start, start + len("Output Rules")
    if heading := _BENIGN_OUTPUT_RULES_HEADING.fullmatch(line):
        return heading.span("target")
    return None


def _bounded_previous_nonblank_line(content: str, offset: int) -> tuple[str, bool]:
    """Return the prior nonblank logical line and whether it was complete."""
    window_start = max(0, offset - 512)
    parts = LOGICAL_LINE_BREAK.split(content[window_start:offset])
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].strip():
            return parts[index], index > 0 or window_start == 0
    return "", window_start == 0


def _bounded_next_nonblank_line(content: str, offset: int) -> tuple[str, bool]:
    """Return the next nonblank logical line and whether it was complete."""
    window_end = min(len(content), offset + 512)
    window = content[offset:window_end]
    cursor = 0
    for line_break in LOGICAL_LINE_BREAK.finditer(window):
        line = window[cursor : line_break.start()]
        if line.strip():
            return line, True
        cursor = line_break.end()
    if window_end == len(content):
        return window[cursor:], True
    return "", False


def _is_benign_print_rules_taxonomy(content: str, match: re.Match[str]) -> bool:
    """Return True only for a bounded declarative selector-taxonomy clause."""
    window_start = max(0, match.start() - 256)
    window_end = min(len(content), match.end() + 256)
    for candidate in _BENIGN_PRINT_RULES_TAXONOMY.finditer(content, window_start, window_end):
        if candidate.span("target") != match.span():
            continue
        candidate_end = candidate.end()
        if (
            candidate_end != len(content)
            and LOGICAL_LINE_BREAK.match(content, candidate_end) is None
        ):
            continue

        if candidate_end != len(content):
            line_break = LOGICAL_LINE_BREAK.match(content, candidate_end)
            assert line_break is not None
            next_line, next_complete = _bounded_next_nonblank_line(content, line_break.end())
            if not next_complete:
                continue
            if _NEXT_LINE_REFERENCE.search(next_line):
                continue

        if candidate.start() == 0:
            return True

        previous_line, previous_complete = _bounded_previous_nonblank_line(
            content, candidate.start()
        )
        if not previous_complete:
            return False
        return _PRECEDING_DIRECTIVE.search(previous_line) is None
    return False


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for system prompt leakage patterns (P6–P8)."""
    prepared = prepare_analysis(content, file_type, lambda: None)
    return _analyze(content, file_path, prepared, SecurityTextView("raw", content))


def _analyze(
    content: str,
    file_path: str,
    prepared: _PreparedAnalysis,
    source_view: SecurityTextView,
    *,
    p6_only: bool = False,
) -> list[AnalyzerFinding]:
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [PatternCategory.SYSTEM_PROMPT_LEAKAGE.value]

    for pattern, confidence in P6_PATTERNS:
        for match in static_runner.iter_paragraph_matches(
            pattern, content, re.IGNORECASE | re.MULTILINE
        ):
            if (
                prepared.code_file
                and pattern in _P6_TERMINAL_TARGET_PATTERNS
                and prepared.target_is_identifier_prefix(match, source_view)
            ):
                continue
            if prepared.returns_local_value(match, source_view, file_path):
                continue
            if prepared.is_javascript_non_extraction(match, source_view, file_path):
                continue
            if is_directly_prohibited(
                content,
                match.start(),
                match.end(),
                allowed_tail=PROMPT_REVEAL_TAIL,
                allow_yara_continuation=source_view.name == "raw",
            ):
                continue
            if prepared.is_report_label(match, source_view):
                continue
            if _is_benign_print_rules_taxonomy(content, match):
                continue
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="P6",
                    message="Direct Prompt Extraction",
                    severity=Severity.HIGH,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                    complete_match=match.group(0),
                )
            )
    if p6_only:
        return findings
    for pattern, confidence in P7_PATTERNS:
        for match in static_runner.iter_paragraph_matches(
            pattern, content, re.IGNORECASE | re.MULTILINE
        ):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="P7",
                    message="Indirect Prompt Extraction",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                    complete_match=match.group(0),
                )
            )
    for pattern, confidence in P8_PATTERNS:
        for match in static_runner.iter_paragraph_matches(
            pattern, content, re.IGNORECASE | re.MULTILINE
        ):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="P8",
                    message="Prompt Exfiltration via Tool",
                    severity=Severity.HIGH,
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
    """Run system_prompt_leakage patterns and return findings."""
    token = _scan_python_ast_cache_key.set(state.get("python_ast_cache_key"))
    try:
        response = static_runner.run_static_patterns_with_ledger(state, [sys.modules[__name__]])
    finally:
        _scan_python_ast_cache_key.reset(token)
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
