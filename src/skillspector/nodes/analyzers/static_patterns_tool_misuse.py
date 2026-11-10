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

"""Static patterns: tool misuse (TM1–TM4). Node and analyze() in one module.

Detects patterns where tool parameters are abused (TM1), tool chaining
is used to bypass safety (TM2), tool defaults are unsafe (TM3), or a
privileged Kubernetes workload is deployed (TM4).

Framework: ASI02.
"""

from __future__ import annotations

import ast
import heapq
import re
import sys
import time
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256

from skillspector.artifacts import (
    SecurityTextView,
    _has_derived_security_view,
    normalized_security_prefix,
    normalized_security_view,
    security_text_views,
)
from skillspector.inspection_ledger import (
    ParserCause,
    ParserCheck,
    ParserUncheckedScope,
    ResourceMeasurements,
    StaticParserDiagnostic,
)
from skillspector.javascript_tokens import JavaScriptTemplateSpans, assertion_template_shell_text
from skillspector.javascript_tokens import javascript_template_spans as _javascript_template_spans
from skillspector.json_tokens import JSONSourceOwnership, prepare_json_source
from skillspector.logging_config import get_logger
from skillspector.markdown_tokens import quoted_list_spans
from skillspector.models import AnalyzerFinding, Finding, Location, Severity, map_analyzer_findings
from skillspector.powershell_tokens import PowerShellSource, PowerShellString, powershell_source
from skillspector.python_ast import ParsedPythonFile, parse_python_source
from skillspector.python_string_constants import python_constant_string_fields
from skillspector.python_tokens import (
    PythonLiteralSpans,
    python_browser_template_shell_text,
    python_string_value_bounds_allowed,
)
from skillspector.python_tokens import python_literal_spans as _python_literal_spans
from skillspector.python_tokens import python_string_body as _python_string_body
from skillspector.security_reconstruction import validated_json_string_spans
from skillspector.state import AnalyzerNodeResponse, SkillspectorState
from skillspector.yaml_tokens import YamlScalar, YamlSourceSpans, yaml_source_spans

from . import static_runner
from .common import (
    LINE_BREAK_CHARS,
    LOGICAL_LINE_BREAK,
    MARKDOWN_FENCE_CLOSE,
    MARKDOWN_FENCE_OPEN,
    SourceLocationIndex,
    get_context,
    get_context_from_lines,
    is_reference_material,
)
from .pattern_defaults import PatternCategory

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_tool_misuse"
USES_RUNTIME_CHECK = True
USES_PYTHON_IMPORT_BOUNDS = True
USES_PREPARED_PARSER_ANALYSIS = True
ANALYZE_USES_POSTPROCESS = True
POSTPROCESS_USES_PYTHON_AST = True
POSTPROCESS_USES_RUNTIME_BUDGET = True
_VARIABLE_SHELL_FLAG_EVIDENCE = "_tm1_variable_shell_flag"
LEXICAL_DIRECT_SHELL_EVIDENCE = "_tm1_lexical_direct_shell"
LEXICAL_RAW_OWNER_EVIDENCE = "_tm1_lexical_raw_owner"
LEXICAL_NORMALIZED_MATCH_EVIDENCE = "_tm1_lexical_normalized_match"
LEXICAL_BOUND_IDENTITY_EVIDENCE = "_tm1_lexical_bound_identity"
LEXICAL_BOUND_IDENTITY_SOURCE_EVIDENCE = "_tm1_lexical_bound_identity_source"
LEXICAL_BOUND_CLASSIFICATION_EVIDENCE = "_tm1_lexical_bound_classification"
LEXICAL_REACH_TERMINATED_EVIDENCE = "_tm1_lexical_reach_terminated"
LEXICAL_BOUND_REACHABLE_EVIDENCE = "_tm1_lexical_bound_reachable"
_MAX_DIRECT_SHELL_CONTEXT_CHARS = 1024
_SHELL_FUNCTION_BODY_RE = re.compile(r"[ \t]*\([ \t]*\)[ \t]*(?:\r?\n[ \t]*)?\{")

_SHELL_COMMAND_WORD_START_RE = re.compile(r"[rRdDeE$'\"`\\]")
_SHELL_COMMAND_WORD_CHARS = 4096
_ROOT_GLOB_COMMAND_CHARS = 8192
_SHELL_HORIZONTAL_OR_CONTINUED_GAP = r"(?>(?:\\\r?\n)*[ \t](?:[ \t]|\\\r?\n)*)"
_BOUNDED_SHELL_CLAUSE_ATOM = r"(?:\\(?:\r?\n|[^\n])|[^\\\n|;&])"
_STATIC_BRACE_WORD_CHARS = 256
_PRINTF_STATIC_CHARS = 256
_PRINTF_STATIC_ARGUMENTS = 32
_PRINTF_STATIC_WORD_RE = re.compile(r"[-A-Za-z0-9_./*?%]{0,64}")
_PRINTF_QUOTED_WORD_RE = re.compile(r"[-A-Za-z0-9_./*?% \t\r\n]{0,64}")
# A bounded ``printf`` format operand may also carry POSIX backslash escapes.
_PRINTF_FORMAT_WORD_RE = re.compile(r"[-A-Za-z0-9_./*?%\\]{0,64}")
# The old deterministic alphabet plus the control/high-byte values needed by
# byte-oriented probes. Command substitutions are expanded after shell syntax
# has been recognized, so decoded angle brackets remain literal word data
# rather than redirections. Anything else outside this set is either a shell
# separator, a metacharacter the downstream glob model does not handle, or
# otherwise not provably inert, so the substitution stays undecidable.
_PRINTF_SAFE_WORD_RE = re.compile(r"[\x01-\x08\x0e-\x1f\x7f-\xffA-Za-z0-9_./*?%<>-]*")
_DESTRUCTIVE_COMMAND_BASENAMES = frozenset({"rm", "del", "erase"})
_QUOTED_GLOB_SENTINEL = "\ue000"
_DYNAMIC_SHELL_WORD_SENTINEL = "\ue001"
_RUNTIME_SHELL_PARAMETER_SENTINEL = "\ue002"
# A braced expansion head whose result depends only on parameter values: an
# optional length prefix, a name with an optional literal subscript or a
# numeric/special parameter, then the closing brace or a value operator.
# Indirection, arithmetic subscripts, substring offsets, ``@`` transformations,
# zsh flags and ``${ cmd; }`` can evaluate code, so they do not match. The
# documentation placeholder ``${...}`` is a bad substitution in every shell.
_VALUE_PARAMETER_EXPANSION_HEAD_RE = re.compile(
    r"\$\{(?:#?(?:[A-Za-z_][A-Za-z0-9_]*(?:\[(?:[@*]|[0-9]+)\])?|[0-9]+|[@*#?$!-])"
    r"(?:\}|:?[-=?+]|##?|%%?|/[/#%]?|\^\^?|,,?)|\.\.\.\})"
)
_VALUE_PARAMETER_EXPANSION_START_RE = re.compile(r"\$\{")
_EXECUTABLE_EXPANSION_MARKERS = ("`", "$(", "$[", "<(", ">(")
_SHELL_ASSIGNMENT_PREFIX_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\+?=")
_SHELL_ASSIGNMENT_NAME_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
)
_SHELL_CLAUSE_BOUNDARY_CHARACTERS = "\r\n;|&(){}`"
_SHELL_DELIMITER_WORD_RE = re.compile(r"[^$'\"`\\(){}<>#;|&!\s]++")
# A ``#`` that starts a word begins a shell comment. Quoting is not tracked, so
# ``echo "a # b"`` also matches; callers only use this to restrict ownership.
_SHELL_COMMENT_START_RE = re.compile(r"(?<![^\s;&|()<>])#")
_PERL_LITERAL_PRINT_RE = re.compile(
    r"^[ \t]*+print\b(?:[ \t]++(?:STDOUT|STDERR)\b)?[ \t]*+(?P<paren>\()?[ \t]*+"
    r"(?P<literal>\"(?:\\[\\\"'nrt]|[^\\\"$@`\r\n])*+\""
    r"|'(?:\\[\\\"'nrt]|[^\\'$@`\r\n])*+')"
    r"[ \t]*+(?(paren)\))[ \t]*+;[ \t]*+(?:\#[^\r\n]*+)?\r?$",
    re.MULTILINE,
)
_PERL_QUOTE_OPERATOR_RE = re.compile(r"\b(?:q[qwxr]?|m|s|tr|y)(?:\s+\S|[^\w\s])")
_PERL_AMBIGUOUS_SIGIL_RE = re.compile(r"[$@%&*]\s*+[{#'\"`]")
# PowerShell accepts typographic quote characters as string delimiters, and a
# doubled delimiter of the same class escapes one inside its string.
_POWERSHELL_SINGLE_QUOTES = "'\u2018\u2019\u201a\u201b"
_POWERSHELL_DOUBLE_QUOTES = '"\u201c\u201d\u201e'
_POWERSHELL_QUOTES = _POWERSHELL_SINGLE_QUOTES + _POWERSHELL_DOUBLE_QUOTES
_POWERSHELL_TOKEN_BREAKS = frozenset(" \t\f\v\r\n;|&(){},")
_POWERSHELL_MAX_NESTING = 32
# Commands whose string arguments only reach the host's display streams. Their
# arguments are messages, never code, unless a capture below redirects them.
_POWERSHELL_STREAM_ONLY_COMMANDS = frozenset(
    {
        "write-host",
        "write-verbose",
        "write-warning",
        "write-debug",
        "write-information",
        "write-progress",
    }
)
# Output-stream commands emit their arguments as values. Only a top-level
# statement's value leaves the script uncaptured by an expression in it.
_POWERSHELL_OUTPUT_COMMANDS = frozenset({"write-output", "echo", "write"})
# Common parameters that capture a stream into a variable. PowerShell accepts
# any unambiguous prefix, so every prefix of these names counts.
_POWERSHELL_CAPTURE_PARAMETERS = (
    "outvariable",
    "informationvariable",
    "warningvariable",
    "errorvariable",
    "pipelinevariable",
)
_POWERSHELL_CAPTURE_PARAMETER_ALIASES = frozenset({"ov", "iv", "wv", "ev", "pv"})
_POWERSHELL_COMMAND_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.\\:-]*")
_POWERSHELL_PARAMETER_RE = re.compile(r"-([A-Za-z]{1,64})")
_POWERSHELL_STATEMENT_END_RE = re.compile(r"[ \t]*+(?:[\r\n;)}#]|\Z)")
# Source that can rebind a command name, load code that may rebind one, or run
# a runtime-built command makes argument ownership unprovable. Matching is on
# the raw text, so a spelling inside a string also disables message ownership.
_POWERSHELL_COMMAND_REBINDING_RE = re.compile(
    r"alias|function:|\b(?:function|filter|workflow)\b[^\r\n{(]*\b(?:write|echo)\b"
    r"|\b(?:sal|nal|ipal|ipmo|ipsn|iex|icm|invoke-expression|invoke-command)\b"
    r"|import-module|import-pssession|\busing[ \t]+module\b|#requires"
    r"|scriptblock|psdefaultparametervalues|executioncontext|commandlookupaction"
    r"|commandnotfoundaction|system\.management\.automation"
    r"|&[ \t]*[$(@{'\"\u2018-\u201e]|(?:^|[\s;|&({])\.[ \t]+[^\s]",
    re.IGNORECASE | re.MULTILINE,
)
# Perl's ``eval BLOCK`` runs already-compiled statements and traps exceptions.
# Only ``eval EXPR`` (a string, variable, or quote operator) reparses source.
_PERL_EVAL_BLOCK_RE = re.compile(r"\s*+\{")
# Outside string and comment tokens, valid Python ends a shell word at
# whitespace or a shell control character. A backslash-newline is a line
# continuation in both languages, so it does not end the word.
_PYTHON_SHELL_WORD_BREAK_RE = re.compile(r"(?<!\\)(?<!\\\r)[\s;|&()<>]")
# Whitespace and line continuations between Python tokens. Comments are
# tokens of their own.
_PYTHON_TOKEN_GAP_RE = re.compile(r"(?:\s|\\\r?\n)*+")
# Python operators that can still extend or rewrite a string value.
_PYTHON_STRING_EXTENSION_OPERATORS = frozenset({"+", "%", ".", "*", "["})
# One brace opens an f-string replacement field; a doubled brace is literal.
_PYTHON_REPLACEMENT_FIELD_RE = re.compile(r"\{\{|(\{)")
_PRINTF_FORMAT_CONVERSION_RE = re.compile(r"%[-+ #0-9.*']*[A-Za-z%]")
_SHELL_ROOT_TARGET_ESCAPE_RE = re.compile(
    r"\\(?:[/~*?]|x(?:2[fF]|7[eE]|2[aA]|3[fF])|"
    r"u(?:002[fF]|007[eE]|002[aA]|003[fF])|"
    r"U(?:0000002[fF]|0000007[eE]|0000002[aA]|0000003[fF])|"
    r"(?:057|176|052|077)(?![0-7]))"
)
_POWERSHELL_REPLACE_EXPRESSION_RE = re.compile(
    r"\A\s*\$(?:[A-Za-z_][A-Za-z0-9_]*|_[A-Za-z0-9_.]*)"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*)*\s+-replace\b",
    re.IGNORECASE,
)
_SHELL_COMMAND_STRING_SHELLS = frozenset({"sh", "ash", "bash", "dash", "ksh", "yash", "zsh"})
_SHELL_COMMAND_STRING_ARGUMENTS = 32
_SHELL_CLAUSE_PREFIX_WORDS = frozenset({"do", "else", "elif", "then", "time", "!"})
_SHELL_ASSIGNMENT_PREFIX_WORDS = _SHELL_CLAUSE_PREFIX_WORDS | frozenset(
    {"if", "while", "until", "export", "local", "declare", "readonly", "typeset"}
)
_SHELL_REDIRECTION_PREFIX_RE = re.compile(r"(?:[0-9]+)?(?:&>>?|<>|>&|<&|>>?|<<-?|[<>])")
_RECURSIVE_OPTION_SOURCE_RE = re.compile(r"-(?:[A-Za-z]*[rR]|-recursive)")
_FORCE_OPTION_SOURCE_RE = re.compile(r"-(?:[A-Za-z]*f|-force)")
_MARKDOWN_CONTRACTION_PROSE_RE = re.compile(
    r"(?:The|This|That|These|Those|It|They|We|You|I|A|An)[ \t]+[A-Za-z \t,]*\b"
    r"(?:doesn|isn|aren|wasn|weren|don|didn|hasn|haven|hadn|can|couldn|shouldn|wouldn|won|mustn)"
    r"(?P<apostrophe>')t\b[A-Za-z \t,]*[.!?]",
    re.IGNORECASE,
)
# An English possessive or contraction inside one prose word, such as
# ``reference's`` or ``don't``. The suffix must end the word, so a quote that
# closes a shell word, as in ``r'm'``, never matches.
_MARKDOWN_WORD_APOSTROPHE_RE = re.compile(
    r"(?<![\w'/.~-])[A-Za-z]++(?P<apostrophe>')(?:s|t|d|m|re|ve|ll)(?![\w'/])",
    re.IGNORECASE,
)
# Shell syntax that opens a quote, an expansion, or a line continuation.
_MARKDOWN_PROSE_SHELL_SYNTAX_RE = re.compile(r"[$`\\\"]")
# Candidate punctuation may surround quoted prose; the final shell walk below
# still refuses apostrophes owned by an earlier quote or unresolved word.
_MARKDOWN_PROSE_EXPANSION_RE = re.compile(r"[$`\\]")
_ROOT_GLOB_DOCUMENTATION_LINE_RE = re.compile(
    r"[ \t]*(?:(?:[-*+]|#{1,6})[ \t]+)?"
    r"(?:(?:(?:documentation|note|example)[ \t]*:[ \t]*)"
    r"(?:(?:the|this|a|an)[ \t]+)?|(?:the|this|a|an)[ \t]+)"
    r"(?:(?:gnu|posix(?:\.[0-9]+)?|unix)[ \t]+)?rm[ \t]+(?:utility|command|tool)\b"
    r"[^;\n]{0,160}\b(?:accepts?|supports?)\b"
    r"[^;\n]{0,160}\b(?:denotes?|means?|represents?)\b"
    r"[^;\n]{0,80}\b(?:wildcard|glob|option|flag)\b[.!?]?[ \t]*",
    re.IGNORECASE,
)
_ROOT_GLOB_EXECUTION_PREFIX_RE = re.compile(
    r"\b(?P<action>run|execute|invoke|issue|launch|perform|call|eval|type|submit|"
    r"enter|paste|carry[ \t]+out)\b[^.;\n]{0,240}$",
    re.IGNORECASE,
)
_ROOT_GLOB_NEGATED_EXECUTION_RE = re.compile(
    r"(?:(?:\bdo[ \t]+not|\bdon't|\bnever|\bavoid|\bmust[ \t]+not)"
    r"(?:(?:[ \t]*,[ \t]*|[ \t]+)"
    r"(?!(?:so|but|yet|then|therefore|however|instead)\b)\w+){0,8}[ \t,]*|"
    r"\bnot[ \t,]*|"
    r"\bnot(?:[ \t]*,[ \t]*|[ \t]+)"
    r"(?:ever|directly|immediately|actually)[ \t,]*|"
    r"\bnot(?:[ \t]*,[ \t]*|[ \t]+)(?:under|in)"
    r"(?:[ \t]*,[ \t]*|[ \t]+)(?:any|all|these|those)"
    r"(?:(?:[ \t]*,[ \t]*|[ \t]+)\w+){0,4}[ \t,]*)$",
    re.IGNORECASE,
)
_ROOT_GLOB_SAFETY_WARNING_PREFIX_RE = re.compile(
    r"[ \t]*(?:please[ \t]+)?(?:"
    r"(?:do[ \t]+not|don't|never|must[ \t]+not)[ \t]+"
    r"(?:run|execute|invoke|issue|launch|perform|call|eval|type|submit|enter|paste|use)"
    r"|avoid[ \t]+(?:running|executing|invoking|issuing|launching|performing|calling|"
    r"evaluating|typing|submitting|entering|pasting|using)"
    r"|refrain[ \t]+from[ \t]+(?:running|executing|invoking|issuing|launching|"
    r"performing|calling|evaluating|typing|submitting|entering|pasting|using)"
    r")[ \t]+(?:the[ \t]+)?",
    re.IGNORECASE,
)
_ROOT_GLOB_AFFIRMATIVE_NEGATION_PREFIX_RE = re.compile(
    r".*\b(?:do[ \t]+not|don't|never)[ \t]+(?:"
    r"forget(?:[ \t]+to|[ \t]+that(?:[ \t]+you)?[ \t]+"
    r"(?:must|should|shall|will|need[ \t]+to)"
    r"|[ \t]*,[ \t]*(?:and[ \t]+)?(?:be|make)[ \t]+sure[ \t]+to)"
    r"|hesitate(?:[ \t]+at[ \t]+all)?[ \t]+to"
    r"|fail[ \t]+to)[ \t]*$",
    re.IGNORECASE,
)

# Match a literal True assigned to a local name shortly before it is passed as
# shell=<name> to a subprocess invocation.  The bounded newline gap and the
# AST binding-evidence guard keep this a local data-flow fact; Python scope
# visibility is enforced separately in analyze() via _VARIABLE_SHELL_FLAG_RE.
_VARIABLE_SHELL_ASSIGNMENT_PATTERN = r"(?m)^\s*([A-Za-z_]\w*)\s*=\s*True\s*$\n"
_VARIABLE_SHELL_FLAG_PATTERN = (
    _VARIABLE_SHELL_ASSIGNMENT_PATTERN + r"(?:[^\n]{0,240}\n){0,4}?[^\n]{0,240}"
    r"(?:subprocess\.\w+|Popen)\s*\([^)]*\bshell\s*=\s*\1\b"
)
_VARIABLE_SHELL_FLAG_RE = re.compile(_VARIABLE_SHELL_FLAG_PATTERN, re.IGNORECASE | re.MULTILINE)
_VARIABLE_SHELL_ASSIGNMENT_RE = re.compile(
    _VARIABLE_SHELL_ASSIGNMENT_PATTERN, re.IGNORECASE | re.MULTILINE
)
_SHELL_NAME_ARGUMENT_RE = re.compile(r"\bshell\s*=\s*\w+", re.IGNORECASE)

# Option searches are bounded to one command before applying their suffix.
# Whitespace may include an explicit Bash or PowerShell line continuation.
_OPTION_COMMAND_SPACE = r"(?:[ \t]|[\\`]\r?\n)+"
_OPTION_COMMAND_RULES = [
    (rf"(?:chmod|chown){_OPTION_COMMAND_SPACE}", r"a\+rwx", 0.8),
    (rf"git{_OPTION_COMMAND_SPACE}push{_OPTION_COMMAND_SPACE}", r"--force", 0.7),
    (rf"curl{_OPTION_COMMAND_SPACE}", r"-k\b", 0.6),
    (rf"curl{_OPTION_COMMAND_SPACE}", r"--insecure\b", 0.65),
    (rf"wget{_OPTION_COMMAND_SPACE}", r"--no-check-certificate", 0.65),
]
_OPTION_COMMAND_PREFIXES = {
    prefix + r"[\s\S]*" + option: re.compile(prefix, re.IGNORECASE)
    for prefix, option, _ in _OPTION_COMMAND_RULES
}
# Host values can fold physical newlines into spaces. Without a shell-region
# proof, retain the original conservative patterns rather than clipping them.
_OPTION_COMMAND_HOST_PATTERNS = {
    pattern: pattern.replace(_OPTION_COMMAND_SPACE, r"\s+").replace(r"[\s\S]*", r"[^|]*")
    for pattern in _OPTION_COMMAND_PREFIXES
}
_OPTION_SHELL_FENCE_LANGUAGES = frozenset({"sh", "bash", "shell", "powershell", "pwsh", "ps1"})

# TM1: Tool Parameter Abuse — dangerous parameter values
DIRECT_SHELL_TRUE_PATTERNS: tuple[tuple[str, float], ...] = (
    # shell=True is a classic command injection vector
    (r"subprocess\.\w+\s*\([^)]*shell\s*=\s*True\b", 0.8),
    (r"Popen\s*\([^)]*shell\s*=\s*True\b", 0.8),
)
_DIRECT_SHELL_TRUE_VALUE = re.compile(r"shell\s*=\s*True\b", re.IGNORECASE)
TM1_CODE_PATTERNS = [
    *DIRECT_SHELL_TRUE_PATTERNS,
    # Preserve the current-main fallback when the Python AST is unavailable.
    (_VARIABLE_SHELL_FLAG_PATTERN, 0.8),
    # Bound command names on both sides so prefixes such as rmm/ (RAPIDS
    # Memory Manager headers) are not interpreted as destructive commands.
    # Keep the scan within one bounded shell command.  The former ``[^|]*``
    # crossed newlines/``;``/``&`` and rescanned the remainder of a document
    # for every prose mention of ``rm``, making repeated documentation
    # quadratic.  Escaped newlines remain valid shell continuations.
    (
        rf"\b(?:rm|del|erase)\b{_SHELL_HORIZONTAL_OR_CONTINUED_GAP}"
        rf"{_BOUNDED_SHELL_CLAUSE_ATOM}{{0,{_ROOT_GLOB_COMMAND_CHARS}}}"
        rf"(?:\\)?-(?:rf|fr|r){_SHELL_HORIZONTAL_OR_CONTINUED_GAP}[/~]",
        0.9,
    ),
    (r"--force\s+(?:delete|remove|push|reset|clean)", 0.7),
    # A bare application-defined --no-verify flag is ambiguous. Match it only
    # for known Git hook bypasses below; retain the other explicit unsafe flags.
    (r"--no-?(?:check|validate|confirm|protect|safe)\b", 0.75),
    (r"--skip-?(?:validation|verification|checks?|auth|tests?)", 0.7),
    # --allow-empty is a benign git-commit option, unlike the bypass flags below.
    (r"--allow-?(?:root|unrelated|unsafe)\b", 0.65),
    # Dangerous globs and wildcards in destructive commands
    # Match a path in the actual rm argument token. Stop at whitespace and shell
    # redirection operators so `rm "$VAR" 2>/dev/null` does not borrow the slash
    # from the redirection target.
    (
        r"\brm\b\s+(?:-[A-Za-z]+\s+)*(?:--\s+)?"
        r"(?:['\"][^'\"]*/[^'\"]*['\"]|[^\s|;&>]*/[^\s|;&>]*)",
        0.85,
    ),
    (r"\bshutil\.rmtree\s*\(\s*['\"]\s*/", 0.85),
    # A numeric chmod mode is world-writable when the others-triple carries the
    # write bit, i.e. its last digit is 2, 3, 6 or 7.  Matching the mode argument
    # itself rather than a bare `777`/`666` substring also keeps a path or a
    # trailing comment from being read as the mode (chmod 755 /tmp/6660 is a
    # chmod 755).  This owns the numeric forms that TM1 already owned for
    # 777/666, plus the 646/757/0662 spellings the privilege-escalation rule used
    # to catch as a side effect of matching 4/5/6/7 anywhere in the mode.
    # Only `chmod` is matched here: `chown` takes an owner rather than a mode, so
    # a numeric owner ID whose last digit is 2/3/6/7 (Grafana's 472:472, Istio's
    # 1337:1337) is not a world-writable mode.  The optional [+=] accepts the
    # operator GNU chmod allows in front of the digits (chmod +777, chmod =666);
    # BSD chmod rejects those, so they are reported rather than assumed safe.
    # The option group takes a single leading dash for the reason given in the
    # privilege-escalation rule: `--?` made the match exponential on a line of
    # unmatched `-- ` tokens, and `[\w=-]*` already absorbs a long option's
    # second dash.
    (r"chmod\s+(?:-[\w=-]*[ \t]+)*[\"']?[+=]?0*[0-7]{2,3}[2367](?![0-9\w])[\"']?", 0.8),
    *[
        (prefix + r"[\s\S]*" + option, confidence)
        for prefix, option, confidence in _OPTION_COMMAND_RULES
    ],
    # Other Git force operations
    (r"git\s+reset\s+--hard", 0.65),
    (r"git\s+clean\s+-[fd]+x", 0.7),
    (r"\bgit\s+(?:am|commit|merge|push)\b[^\n|]*--no-verify\b", 0.75),
    # File deletion commands (delete/remove) with explicit path argument
    # Separate from rm pattern: requires \b and a short span to a "/" to avoid
    # matching "remove" in prose; HTTP verb DELETE in REST route tables is
    # abstained in _is_rest_route.
    (r"\b(?:delete|remove)\s+['\"]?/[^\s'\"]{1,100}", 0.80),
    # SQL injection via parameter construction
    (
        r"(?:execute|query)\s*\(\s*f?['\"].*?\{.*?\}.*?\b(?:DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE)\b",
        0.85,
    ),
]
TM1_PROSE_PATTERNS = [
    # Dangerous tool parameter patterns in instructions
    (
        r"(?:set|pass|use)\s+(?:the\s+)?(?:parameter|argument|flag|option)\s+(?:to\s+)?(?:shell\s*=\s*True|--force|-rf)\b",
        0.8,
    ),
]
TM1_PATTERNS = TM1_CODE_PATTERNS + TM1_PROSE_PATTERNS

# TM2: Chaining Abuse — chained commands to bypass safety
TM2_CODE_PATTERNS = [
    # Shell command chaining with dangerous commands (\b prevents substring matches)
    (r"(?:&&|;)\s*\b(?:rm\b|del\b|erase\b)\s+-", 0.75),
    (r"(?:&&|;)\s*(?:curl|wget)\s+[^|]*\|\s*(?:ba)?sh", 0.9),
    (r"(?:&&|;)\s*(?:sudo|su\s+)", 0.75),
    (r"(?:&&|;)\s*(?:chmod|chown)\s+(?:777|666|a\+rwx|-R)", 0.75),
    # Pipe chains with escalating danger
    (r"\|\s*(?:sudo|su)\s+", 0.75),
    (r"\|\s*(?:sh|bash|zsh|python|node|ruby|perl)\s*$", 0.7),
    (r"\|\s*(?:tee|xargs)\s+.*?\b(?:rm|del|sudo|curl)\b", 0.75),
]
TM2_PROSE_PATTERNS = [
    # Multi-step chains designed to circumvent individual checks
    # Bounded to ~500 chars per gap to avoid spanning entire documents
    (
        r"(?:first|step\s+1)[^\n]{0,500}(?:then|step\s+2)[^\n]{0,500}(?:finally|step\s+3)[^\n]{0,200}\b(?:delete|remove|wipe|destroy|exfiltrate|send)\b",
        0.7,
    ),
    (
        r"(?:chain|combine|sequence|pipe)\s+(?:these\s+)?(?:tools?|commands?|actions?)\s+to\s+(?:bypass|circumvent|avoid|skip)\s+(?:the\s+)?(?:safety|security|check|restriction|limit)",
        0.9,
    ),
    (
        r"(?:use|call)\s+(?:tool\s+)?(?:A|one|the\s+first)\s+(?:to|and)[^\n]{0,300}(?:then\s+)?(?:use|call|pass\s+(?:the\s+)?(?:output|result)\s+to)\s+(?:tool\s+)?(?:B|two|another)",
        0.6,
    ),
    # Natural language chaining instructions
    (
        r"(?:after|once)\s+(?:the\s+)?(?:first|initial)\s+(?:tool|command|action)\s+(?:succeeds|completes|runs)[^\n]{0,300}(?:immediately|then|next)\s+(?:run|execute|call|invoke)",
        0.6,
    ),
]
TM2_PATTERNS = TM2_CODE_PATTERNS + TM2_PROSE_PATTERNS

# TM3: Unsafe Defaults — overly permissive default settings
#
# A umask leaves new files world-writable when its world-write bit (0o002) is unset.
# In octal digits that is a last digit of 0, 1, 4 or 5.
_UMASK_OCTAL_WORLD_WRITE = r"[0-7]*[0145]\b"
# Python, JavaScript and C read a number as octal only with a 0o prefix or a leading
# 0, so a bare 14 is 0o016 (world-write masked) and a bare 8 is 0o010 (unmasked). A
# decimal N leaves the bit unset when N % 4 is 0 or 1, which its last two digits decide.
_UMASK_NUMBER_WORLD_WRITE = (
    rf"(?:0(?:o?{_UMASK_OCTAL_WORLD_WRITE})?\b"
    r"|[1-9]\d*+(?<=[02468][014589]|[13579][2367]|\D[14589])\b)"
)
# umask 000, or an assigned value such as umask = 0o000 or umask = 8.
_TM3_UMASK = (
    rf"umask(?:[ \t]+(?:0o)?{_UMASK_OCTAL_WORLD_WRITE}|\s*=\s*{_UMASK_NUMBER_WORLD_WRITE})",
    0.8,
)
# Shell reads every umask value as octal, including a variable such as UMASK=14 that
# is later passed to the umask command.
_TM3_SHELL_UMASK = (rf"umask(?:\s*=\s*|[ \t]+)(?:0o)?{_UMASK_OCTAL_WORLD_WRITE}", 0.8)
# umask(0) as a statement or inside an expression. `old = os.umask(0)` is how Python
# reads the umask before restoring it, so analyze() skips a call whose result is
# assigned (see _is_assigned_value).
_TM3_UMASK_CALL = (rf"(?<![\w.])(?:\w+\.)*umask\(\s*{_UMASK_NUMBER_WORLD_WRITE}\s*\)", 0.8)
TM3_CODE_PATTERNS = [
    # TLS/SSL verification disabled
    (r"verify\s*=\s*False", 0.75),
    (r"VERIFY_SSL\s*=\s*False", 0.8),
    (r"(?:ssl|tls)[_.]?verify\s*=\s*(?:False|false|0|off|no|disable)", 0.8),
    (r"(?:REQUESTS_CA_BUNDLE|CURL_CA_BUNDLE)\s*=\s*['\"]['\"]", 0.75),
    (r"NODE_TLS_REJECT_UNAUTHORIZED\s*=\s*['\"]?0['\"]?", 0.8),
    # Authentication disabled
    (r"(?:auth|authentication|authorization)\s*=\s*(?:None|False|false|disabled?|off|no)", 0.75),
    (r"(?:require[_-]?auth|auth[_-]?required|check[_-]?auth)\s*=\s*(?:False|false|0|no|off)", 0.8),
    (r"(?:allow[_-]?anonymous|anonymous[_-]?access)\s*=\s*(?:True|true|1|yes|on)", 0.75),
    # Overly permissive CORS / access
    (r"(?:CORS|cors)[^=]*=\s*['\"]?\*['\"]?", 0.65),
    (r"(?:allow|access)[_-]?(?:origin|hosts?)\s*=\s*['\"]?\*['\"]?", 0.7),
    # Unsafe permissions. A mode masked with the umask (0o666 & ~umask) respects it,
    # like a plain open(), when the mask ends the expression: a closing bracket, a
    # separator, or the end of a line (LF or CRLF, after an optional # comment) whose
    # next line does not continue it with an operator, `and` or `or`. Any other
    # operator, such as | 0o777 or the floor division // 512, can change the result
    # and is still reported. The comment is bounded so the check stays linear.
    (
        r"(?:mode|permission)\s*=\s*(?:0?o?777|0?o?666)"
        r"(?![ \t]*&[ \t]*~[ \t]*[\w.]*umask\b(?:\([^()\n]*\))?[ \t]*"
        r"(?:[),;\]}]|(?:#[^\n]{0,240}|\r)?$"
        r"(?!\n[ \t]*(?:[-+*/%|&^<>=?:.]|(?:and|or)\b))))",
        0.8,
    ),
    _TM3_UMASK,
    _TM3_UMASK_CALL,
    (r"world[_-]?(?:readable|writable|executable)", 0.7),
    # Debug/dev mode in production
    (r"(?:debug|dev|development)[_-]?mode\s*=\s*(?:True|true|1|on|yes|enable)", 0.6),
    (
        r"(?:FLASK_ENV|NODE_ENV|RAILS_ENV|DJANGO_DEBUG)\s*=\s*['\"]?(?:development|debug|true|1)['\"]?",
        0.6,
    ),
    # Disable security features
    (
        r"(?:disable|skip|ignore|bypass)[_-]?(?:security|auth|validation|sanitization|encoding|escaping)",
        0.8,
    ),
    (r"(?:safe[_-]?mode|secure[_-]?mode|sandbox)\s*=\s*(?:False|false|0|off|no|disable)", 0.8),
]
TM3_PROSE_PATTERNS = [
    (r"(?:allow|trust)\s+(?:all|any|every)\s+(?:origins?|hosts?|domains?|ips?)", 0.7),
    # Natural language unsafe defaults
    (r"(?:by\s+default|default\s+to)\s+(?:allow|accept|trust)\s+(?:all|any|everything)", 0.7),
    (
        r"(?:trust|accept|allow)\s+(?:all|any)\s+(?:input|connections?|certificates?|origins?)\s+(?:by\s+default)",
        0.7,
    ),
]
TM3_PATTERNS = TM3_CODE_PATTERNS + TM3_PROSE_PATTERNS
_TM3_SHELL_PATTERNS = [_TM3_SHELL_UMASK if entry == _TM3_UMASK else entry for entry in TM3_PATTERNS]

# TM4: Privileged Kubernetes Workload — manifest/CLI primitives that grant
# node/host takeover (the cluster-scale counterpart of a privileged container).
# Only isolation-breaking signals are matched, so a normal `kubectl apply` or a
# plain DaemonSet does not fire.
TM4_PATTERNS = [
    (r"privileged\s*:\s*true", 0.7),  # privileged container in a manifest
    (r"hostPath\s*:", 0.55),  # host filesystem mount
    (r"host(?:PID|Network|IPC)\s*:\s*true", 0.6),  # host namespace sharing
    (r"kubectl\s+run\b[^\n]*--privileged", 0.7),  # privileged ad-hoc pod
    (r"--set\b[^\n]*privileged\s*=\s*true", 0.6),  # helm privileged override
]


_SAFE_CONTAINER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"docker\s+run\s+.*--rm", re.IGNORECASE),
    re.compile(r"docker\s+run\s+.*-it", re.IGNORECASE),
    re.compile(r"docker\s+(?:build|compose|pull|push)\b", re.IGNORECASE),
    re.compile(r"podman\s+run\b", re.IGNORECASE),
)

# Standard Dockerfile RUN idioms that are best practice, not abuse
_SAFE_DOCKERFILE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # apt cleanup: rm -rf /var/lib/apt/lists/*
    re.compile(r"rm\s+-rf\s+/var/lib/apt/lists", re.IGNORECASE),
    re.compile(r"rm\s+-rf\s+/var/cache/apt", re.IGNORECASE),
    # Dockerfile user setup: chown -R user:group /path
    re.compile(r"chown\s+-R\s+\w+:\w+\s+/", re.IGNORECASE),
    # pip cache cleanup
    re.compile(r"rm\s+-rf\s+/root/\.cache", re.IGNORECASE),
)

_SAFE_CACHE_CLEANUP_RE = re.compile(
    r"\brm\s+-rf\s+(?P<quote>['\"]?)(?P<path>(?:\$?\{?HOME\}?|~)/\.cache/[^\s;&|'\"]+)(?P=quote)(?:\s*(?:$|[;&|]))",
    re.IGNORECASE,
)

# Dockerfile context indicators (nearby keywords that signal Dockerfile content)
_DOCKERFILE_CONTEXT_RE = re.compile(
    r"\b(?:FROM|RUN|WORKDIR|COPY|ADD|ENV|EXPOSE|ENTRYPOINT|CMD|USER|HEALTHCHECK|ARG)\s",
)

_HOST_SYSTEM_ROOTS: frozenset[str] = frozenset(
    s.lower()
    for s in {
        "bin",
        "boot",
        "dev",
        "etc",
        "home",
        "lib",
        "lib64",
        "media",
        "mnt",
        "opt",
        "private",
        "proc",
        "root",
        "run",
        "sbin",
        "sys",
        "tmp",
        "users",
        "usr",
        "var",
        "volumes",
    }
)

_REST_ROUTE_LINE_RE = re.compile(
    r"^[ \t]*(?:[-*#|>~`]+\s*)*`?DELETE\b",
)


def _is_rest_route(matched_text: str, matched_line: str) -> bool:
    """Return whether an uppercase DELETE match is a REST route rather than file deletion."""
    if not matched_text.startswith("DELETE"):
        return False
    parts = matched_text.split(None, 1)
    if len(parts) < 2:
        return False
    raw_path = parts[1].strip("'\"")
    if not raw_path.startswith("/"):
        return False
    path_segments = [s for s in raw_path.split("/") if s]
    if not path_segments:
        return False
    if path_segments[0].lower() in _HOST_SYSTEM_ROOTS:
        return False
    if _REST_ROUTE_LINE_RE.match(matched_line):
        return True
    if any(marker in raw_path for marker in (":", "{", "}", "?")):
        return True
    if path_segments[0].lower() in {
        "api",
        "v1",
        "v2",
        "v3",
        "v4",
        "rest",
        "graphql",
        "auth",
        "oauth",
    }:
        return True
    return False


def _is_safe_container_command(text: str) -> bool:
    """Return True for standard Docker/Podman commands that are not parameter abuse."""
    return any(p.search(text) for p in _SAFE_CONTAINER_PATTERNS)


def _is_safe_dockerfile_idiom(context: str, matched_text: str) -> bool:
    """Return True for standard Dockerfile cleanup/setup patterns."""
    if not _DOCKERFILE_CONTEXT_RE.search(context):
        return False
    return any(p.search(matched_text) or p.search(context) for p in _SAFE_DOCKERFILE_PATTERNS)


def _is_safe_cache_cleanup(matched_text: str) -> bool:
    """Return True for scoped cleanup of a tool-owned user cache path."""
    match = _SAFE_CACHE_CLEANUP_RE.fullmatch(matched_text)
    if not match:
        return False
    parts = match.group("path").split("/")
    lowered_parts = [part.lower() for part in parts]
    cache_index = lowered_parts.index(".cache")
    cache_parts = parts[cache_index + 1 :]
    return bool(cache_parts) and not any(
        part in ("", ".", "..") or "*" in part for part in cache_parts
    )


@dataclass(frozen=True)
class _ShellToken:
    text: str
    root_glob: bool
    glob_projection: str
    has_quoted_content: bool
    brace_expansion: bool
    leading_tilde_unquoted: bool


@dataclass(frozen=True)
class _ShellCommandWord:
    text: str
    end: int
    dynamic: bool
    limited: bool = False
    diagnostic: StaticParserDiagnostic | None = None


@dataclass(frozen=True)
class _ShellCommandString:
    content: str | None
    startup_context: list[bool]
    startup_diagnostics: list[StaticParserDiagnostic]
    start: int
    end: int
    exact_source: bool
    diagnostic: StaticParserDiagnostic | None
    xargs_contexts: tuple[tuple[str, list[bool | None]], ...] = ()


def _note_parser_diagnostic(
    diagnostics: list[StaticParserDiagnostic] | None,
    cause: ParserCause,
    check: ParserCheck,
    start: int,
    end: int,
    scope: ParserUncheckedScope,
    *,
    measurements: ResourceMeasurements | None = None,
    lower_bound: bool = False,
    full_input_checked: bool = False,
) -> None:
    """Keep the first cause from one parser attempt, without retaining its input."""
    if diagnostics is not None and not diagnostics:
        diagnostics.append(
            StaticParserDiagnostic(
                cause,
                check,
                start,
                max(start + 1, end),
                scope,
                measurements or {},
                observed_is_lower_bound=lower_bound,
                full_input_checked=full_input_checked,
            )
        )


@dataclass(frozen=True)
class _ParameterExpansionEnd:
    """Cached parameter endpoint plus its enclosing-quote transition."""

    end: int | None
    inherited_quote_closed: bool = False


@dataclass
class _ShellDelimiterFrame:
    """One iterative shell-expansion delimiter frame."""

    kind: str
    start: int | None
    quote: str | None = None
    ansi_c_quote: bool = False
    word_started: bool = False
    inherited_double_quote: bool = False
    inherited_quote_closed: bool = False
    pending_case_clauses: int = 0
    open_case_clauses: int = 0
    pending_heredocs: tuple[tuple[str, bool, bool], ...] = ()
    arithmetic: bool = False


def _is_shell_command_word_start(content: str, start: int) -> bool:
    if start == 0:
        return True
    previous = content[start - 1]
    if previous.isspace() or previous in ";|&()<>/":
        return True
    if previous in "'\"`":
        before_quote = start - 2
        return (
            before_quote < 0
            or content[before_quote].isspace()
            or content[before_quote] in ";|&()<>{}/"
        )
    return False


def _is_shell_command_position(content: str, start: int) -> bool:
    """Return whether ``start`` opens a line or follows a command separator."""
    cursor = start - 1
    while cursor >= 0 and content[cursor] in " \t":
        cursor -= 1
    return cursor < 0 or content[cursor] in "\n\r;|&("


def _opens_runtime_word(content: str, start: int) -> bool:
    """Return whether ``start`` opens a ``$`` or single-backtick word, bare or in ``"``."""
    cursor = start + 1 if content.startswith('"', start) else start
    if content.startswith("$", cursor):
        return True
    # A backtick run such as a Markdown fence never opens a substitution.
    return (
        content.startswith("`", cursor)
        and not content.startswith("``", cursor)
        and (cursor == 0 or content[cursor - 1] != "`")
    )


def _has_quoted_assignment_prefix(content: str, start: int) -> bool:
    """Return whether a quoted candidate starts with a shell assignment name."""
    if (
        start >= len(content)
        or content[start] not in "'\""
        or start > 0
        and content[start - 1] == "$"
    ):
        return False
    cursor = start + 1
    if cursor >= len(content) or not (content[cursor].isalpha() or content[cursor] == "_"):
        return False
    cursor += 1
    while cursor < len(content) and (content[cursor].isalnum() or content[cursor] == "_"):
        cursor += 1
    return cursor < len(content) and content[cursor] == "="


def _decode_ansi_c_escape(content: str, start: int, limit: int) -> tuple[str, int]:
    """Decode one bounded Bash ANSI-C escape without evaluating shell input."""
    if start + 1 >= limit:
        return "\\", start + 1
    escaped = content[start + 1]
    simple = {
        "a": "\a",
        "b": "\b",
        "e": "\x1b",
        "E": "\x1b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "v": "\v",
        "\\": "\\",
        "'": "'",
        '"': '"',
        "?": "?",
    }
    if escaped in simple:
        return simple[escaped], start + 2
    if escaped in "xXuU":
        maximum_digits = {"x": 2, "X": 2, "u": 4, "U": 8}[escaped]
        cursor = start + 2
        while (
            cursor < limit
            and cursor < start + 2 + maximum_digits
            and content[cursor] in "0123456789abcdefABCDEF"
        ):
            cursor += 1
        if cursor > start + 2:
            value = int(content[start + 2 : cursor], 16)
            if value <= sys.maxunicode:
                return chr(value), cursor
    if escaped in "01234567":
        cursor = start + 1
        while cursor < limit and cursor < start + 4 and content[cursor] in "01234567":
            cursor += 1
        return chr(int(content[start + 1 : cursor], 8)), cursor
    if escaped == "c" and start + 2 < limit:
        return chr(ord(content[start + 2].upper()) ^ 0x40), start + 3
    return "\\" + escaped, start + 2


def _skip_ansi_c_quote_tail(content: str, start: int, limit: int) -> int | None:
    """Find the real closing quote after a NUL, skipping escaped quotes."""
    cursor = start
    while cursor < limit:
        if content[cursor] == "\\" and cursor + 1 < limit:
            cursor += 2
            continue
        if content[cursor] == "'":
            return cursor
        cursor += 1
    return None


def _static_heredoc_header(
    content: str,
    start: int,
    limit: int,
    diagnostics: list[StaticParserDiagnostic] | None = None,
) -> tuple[tuple[str, bool, bool], int] | None:
    """Read one bounded literal heredoc delimiter, without evaluating it."""
    cursor = start + 2
    strip_tabs = content[cursor : cursor + 1] == "-"
    cursor += strip_tabs
    whitespace_limit = min(limit, cursor + _SHELL_COMMAND_WORD_CHARS)
    while cursor < whitespace_limit and content[cursor] in " \t":
        cursor += 1
    if cursor < limit and content[cursor] in " \t":
        _note_parser_diagnostic(
            diagnostics,
            ParserCause.RESOURCE_LIMIT,
            ParserCheck.HEREDOC_HEADER,
            start,
            cursor + 1,
            ParserUncheckedScope.EXPRESSION,
            measurements={
                "observed_characters": _SHELL_COMMAND_WORD_CHARS + 1,
                "limit_characters": _SHELL_COMMAND_WORD_CHARS,
            },
            lower_bound=True,
        )
        return None
    quote = content[cursor : cursor + 1]
    quoted = quote in {"'", '"'}
    cursor += quoted
    name = re.match(
        r"[A-Za-z_][A-Za-z0-9_]*",
        content[cursor : min(limit, cursor + _SHELL_COMMAND_WORD_CHARS + 1)],
    )
    if name is None or len(name[0]) > _SHELL_COMMAND_WORD_CHARS:
        _note_parser_diagnostic(
            diagnostics,
            ParserCause.RESOURCE_LIMIT if name is not None else ParserCause.UNSUPPORTED_SYNTAX,
            ParserCheck.HEREDOC_HEADER,
            start,
            min(limit, cursor + _SHELL_COMMAND_WORD_CHARS + 1),
            ParserUncheckedScope.EXPRESSION,
            measurements={
                "observed_characters": len(name[0]),
                "limit_characters": _SHELL_COMMAND_WORD_CHARS,
            }
            if name is not None
            else None,
            lower_bound=name is not None,
        )
        return None
    cursor += len(name[0])
    if quoted:
        if cursor >= limit or content[cursor] != quote:
            _note_parser_diagnostic(
                diagnostics,
                ParserCause.UNSUPPORTED_SYNTAX,
                ParserCheck.HEREDOC_HEADER,
                start,
                min(cursor + 1, limit),
                ParserUncheckedScope.EXPRESSION,
            )
            return None
        cursor += 1
    if cursor < limit and not (content[cursor].isspace() or content[cursor] in ";|&()<>"):
        _note_parser_diagnostic(
            diagnostics,
            ParserCause.UNSUPPORTED_SYNTAX,
            ParserCheck.HEREDOC_HEADER,
            start,
            cursor + 1,
            ParserUncheckedScope.EXPRESSION,
        )
        return None
    return (name[0], strip_tabs, quoted), cursor


def _skip_static_heredoc_bodies(
    content: str,
    start: int,
    limit: int,
    pending: tuple[tuple[str, bool, bool], ...],
    check_runtime: Callable[[], None] | None,
    diagnostics: list[StaticParserDiagnostic] | None = None,
) -> int | None:
    """Skip ordered bodies only after their declaring command's newline.

    This proves outer lexical state only. The raw scanner still inspects every
    body, including substitutions in unquoted heredocs. Delimiter comparison
    joins unquoted continuations while retaining at most one delimiter's width.
    """
    cursor = start
    expected_crlf = content[start - 2 : start] == "\r\n" if start > 0 else None
    for delimiter, strip_tabs, quoted in pending:
        logical_prefix = ""
        while cursor < limit:
            if check_runtime is not None:
                check_runtime()
            newline = content.find("\n", cursor, limit)
            line_end = limit if newline < 0 else newline
            crlf = newline >= 0 and line_end > cursor and content[line_end - 1] == "\r"
            if newline >= 0:
                if expected_crlf is None:
                    expected_crlf = crlf
                elif crlf != expected_crlf:
                    # Mixed physical endings cannot manufacture a delimiter.
                    # Body-only callers establish the style on their first
                    # complete line; uniformly CRLF bodies remain supported.
                    return None
            if crlf:
                line_end -= 1
            line = content[cursor:line_end]
            continued = (
                not quoted and newline >= 0 and (len(line) - len(line.rstrip("\\"))) % 2 == 1
            )
            if continued:
                line = line[:-1]
            if strip_tabs and not logical_prefix:
                line = line.lstrip("\t")
            # Once overlong, a logical line cannot become the delimiter. Keep
            # a capped prefix instead of accumulating arbitrarily long bodies.
            logical_prefix += line[: max(0, len(delimiter) + 1 - len(logical_prefix))]
            if not continued and logical_prefix == delimiter:
                cursor = limit if newline < 0 else newline + 1
                break
            if newline < 0:
                _note_parser_diagnostic(
                    diagnostics,
                    ParserCause.BOUNDARY_UNRESOLVED,
                    ParserCheck.HEREDOC_BODY,
                    start,
                    limit,
                    ParserUncheckedScope.EXPRESSION,
                    full_input_checked=limit == len(content),
                )
                return None
            cursor = newline + 1
            if not continued:
                logical_prefix = ""
        else:
            _note_parser_diagnostic(
                diagnostics,
                ParserCause.BOUNDARY_UNRESOLVED,
                ParserCheck.HEREDOC_BODY,
                start,
                limit,
                ParserUncheckedScope.EXPRESSION,
                full_input_checked=limit == len(content),
            )
            return None
    return cursor


def _skip_shell_delimited_expansion(
    content: str,
    start: int,
    limit: int,
    root_kind: str,
    check_runtime: Callable[[], None] | None,
    parameter_end_cache: dict[int, _ParameterExpansionEnd] | None,
    substitution_end_cache: dict[int, int | None] | None,
    backtick_end_cache: dict[int, int | None] | None,
    root_inherited_double_quote: bool = False,
    inherited_quote_closed: list[bool] | None = None,
    diagnostics: list[StaticParserDiagnostic] | None = None,
) -> int | None:
    """Skip nested shell delimiters iteratively and cache every typed endpoint."""
    opener_width = 1 if root_kind == "backtick" else 2
    frames = [
        _ShellDelimiterFrame(
            root_kind,
            start,
            inherited_double_quote=root_inherited_double_quote,
            arithmetic=root_kind == "command" and content.startswith("$((", start),
        )
    ]
    cursor = start + opener_width
    next_runtime_check = ((cursor + 4095) // 4096) * 4096

    def cache_frame(frame: _ShellDelimiterFrame, end: int | None) -> None:
        if frame.start is None:
            return
        if frame.kind == "parameter" and parameter_end_cache is not None:
            parameter_end_cache[frame.start] = _ParameterExpansionEnd(
                end,
                frame.inherited_quote_closed,
            )
        elif frame.kind == "command" and substitution_end_cache is not None:
            substitution_end_cache[frame.start] = end
        elif frame.kind == "backtick" and backtick_end_cache is not None:
            backtick_end_cache[frame.start] = end

    def cache_failure() -> None:
        for frame in frames:
            cache_frame(frame, None)

    def close_frame(end: int) -> int | None:
        frame = frames.pop()
        cache_frame(frame, end)
        return end if not frames else None

    def push(kind: str, frame_start: int | None, width: int) -> None:
        nonlocal cursor
        parent = frames[-1]
        if kind in {"parameter", "command", "backtick"} and parent.kind in {
            "command",
            "paren",
        }:
            parent.word_started = True
        inherited_double_quote = kind in {"parameter", "brace"} and (
            parent.quote == '"'
            or (
                parent.inherited_double_quote
                and parent.kind in {"parameter", "brace"}
                and parent.quote is None
            )
        )
        frames.append(
            _ShellDelimiterFrame(
                kind,
                frame_start,
                inherited_double_quote=inherited_double_quote,
                arithmetic=(
                    (kind == "paren" and (parent.arithmetic or content.startswith("((", cursor)))
                    or (frame_start is not None and content.startswith("$((", frame_start))
                ),
            )
        )
        cursor += width

    def at_shell_keyword(keyword: str) -> bool:
        end = cursor + len(keyword)
        if end > limit or content[cursor:end] != keyword:
            return False
        before = content[cursor - 1] if cursor else " "
        after = content[end] if end < limit else " "
        delimiters = ";|&(){}<>! \t\r\n"
        return before in delimiters and after in delimiters

    while cursor < limit:
        if check_runtime is not None and cursor >= next_runtime_check:
            check_runtime()
            next_runtime_check = cursor + 4096
        frame = frames[-1]
        character = content[cursor]

        if frame.kind == "backtick":
            if character == "\\" and cursor + 1 < limit:
                cursor += 2
                continue
            if character == "`":
                endpoint = close_frame(cursor + 1)
                cursor += 1
                if endpoint is not None:
                    return endpoint
                continue
            cursor += 1
            continue

        if frame.quote is not None:
            if frame.quote == '"' and content.startswith("${", cursor):
                push("parameter", cursor, 2)
                continue
            if frame.quote == '"' and content.startswith("$(", cursor):
                push("command", cursor, 2)
                continue
            if frame.quote == '"' and character == "`":
                push("backtick", cursor, 1)
                continue
            if character == frame.quote:
                frame.quote = None
                frame.ansi_c_quote = False
            elif character == "\\" and frame.ansi_c_quote:
                _, cursor = _decode_ansi_c_escape(content, cursor, limit)
                continue
            elif character == "\\" and frame.quote == '"' and cursor + 1 < limit:
                cursor += 2
                continue
            cursor += 1
            continue

        if frame.kind in {"command", "paren"} and not frame.arithmetic:
            if character == "\n" and frame.pending_heredocs:
                body_end = _skip_static_heredoc_bodies(
                    content, cursor + 1, limit, frame.pending_heredocs, check_runtime, diagnostics
                )
                if body_end is None:
                    cache_failure()
                    return None
                frame.pending_heredocs = ()
                frame.word_started = False
                cursor = body_end
                continue
            if content.startswith("<<<", cursor):
                cursor += 3
                frame.word_started = False
                continue
            if content.startswith("<<", cursor):
                header = _static_heredoc_header(content, cursor, limit, diagnostics)
                if header is None or len(frame.pending_heredocs) == _SHELL_COMMAND_STRING_ARGUMENTS:
                    if header is not None:
                        _note_parser_diagnostic(
                            diagnostics,
                            ParserCause.RESOURCE_LIMIT,
                            ParserCheck.HEREDOC_COUNT,
                            start,
                            cursor,
                            ParserUncheckedScope.EXPRESSION,
                            measurements={
                                "observed_records": len(frame.pending_heredocs) + 1,
                                "limit_records": _SHELL_COMMAND_STRING_ARGUMENTS,
                            },
                            lower_bound=True,
                        )
                    cache_failure()
                    return None
                spec, cursor = header
                frame.pending_heredocs += (spec,)
                frame.word_started = True
                continue

        # Ordinary words cannot change the delimiter stack. Consume them in
        # one bounded match rather than testing each character for every shell
        # opener, quote, and keyword. Nested static-evaluator windows otherwise
        # repeat those Python-level checks hundreds of thousands of times.
        word = _SHELL_DELIMITER_WORD_RE.match(content, cursor, min(limit, cursor + 4096))
        if word is not None:
            if frame.kind in {"command", "paren"}:
                if at_shell_keyword("case"):
                    frame.pending_case_clauses += 1
                elif frame.pending_case_clauses and at_shell_keyword("in"):
                    frame.pending_case_clauses -= 1
                    frame.open_case_clauses += 1
                elif frame.open_case_clauses and at_shell_keyword("esac"):
                    frame.open_case_clauses -= 1
                frame.word_started = True
            cursor = word.end()
            continue

        if content.startswith("${", cursor):
            push("parameter", cursor, 2)
            continue
        if content.startswith("$(", cursor):
            push("command", cursor, 2)
            continue
        if character in "<>" and cursor + 1 < limit and content[cursor + 1] == "(":
            push("command", cursor, 2)
            continue
        if character == "`":
            push("backtick", cursor, 1)
            continue
        if (
            character == "$"
            and cursor + 1 < limit
            and content[cursor + 1] in "'\""
            and not (
                frame.inherited_double_quote
                and frame.kind in {"parameter", "brace"}
                and content[cursor + 1] in "'\""
            )
        ):
            frame.quote = content[cursor + 1]
            frame.ansi_c_quote = frame.quote == "'"
            if frame.kind in {"command", "paren"}:
                frame.word_started = True
            cursor += 2
            continue
        if (
            character == '"'
            and frame.inherited_double_quote
            and frame.kind in {"parameter", "brace"}
        ):
            # In ksh, $" inside a parameter expansion inherited from a
            # double-quoted word is a literal '$' followed by the closing
            # quote.  Do not mistake that quote for the start of a new
            # locale-translation string, or the scanner can hide commands
            # that follow the parameter expansion.
            frame.inherited_double_quote = False
            frame.inherited_quote_closed = True
            if inherited_quote_closed is not None:
                inherited_quote_closed[0] = True
            for parent in reversed(frames[:-1]):
                if parent.kind in {"parameter", "brace"}:
                    parent.inherited_double_quote = False
                    parent.inherited_quote_closed = True
                if parent.quote == '"':
                    parent.quote = None
                    parent.ansi_c_quote = False
                    break
            cursor += 1
            continue
        if character in "'\"" and not (
            character == "'"
            and frame.inherited_double_quote
            and frame.kind in {"parameter", "brace"}
        ):
            frame.quote = character
            frame.ansi_c_quote = False
            if frame.kind in {"command", "paren"}:
                frame.word_started = True
            cursor += 1
            continue
        if character == "\\" and cursor + 1 < limit:
            if frame.kind in {"command", "paren"}:
                frame.word_started = True
            cursor += 2
            continue

        if frame.kind in {"parameter", "brace"}:
            if character == "{":
                push("brace", None, 1)
                continue
            if character == "}":
                endpoint = close_frame(cursor + 1)
                cursor += 1
                if endpoint is not None:
                    return endpoint
                continue
            cursor += 1
            continue

        if character == "(":
            if frame.pending_heredocs:
                # Plain subshell grammar can share pending heredocs with the
                # enclosing command. Do not assign that body to a new frame.
                _note_parser_diagnostic(
                    diagnostics,
                    ParserCause.UNSUPPORTED_SYNTAX,
                    ParserCheck.HEREDOC_GRAMMAR,
                    start,
                    cursor + 1,
                    ParserUncheckedScope.EXPRESSION,
                )
                cache_failure()
                return None
            push("paren", None, 1)
            continue
        if character == ")":
            if frame.open_case_clauses:
                # ``)`` terminates a case pattern, not the surrounding command
                # substitution. The corresponding ``esac`` above releases the
                # real substitution closer without requiring a full shell AST.
                frame.word_started = False
                cursor += 1
                continue
            if frame.pending_heredocs:
                _note_parser_diagnostic(
                    diagnostics,
                    ParserCause.UNSUPPORTED_SYNTAX,
                    ParserCheck.HEREDOC_GRAMMAR,
                    start,
                    cursor + 1,
                    ParserUncheckedScope.EXPRESSION,
                )
                cache_failure()
                return None
            endpoint = close_frame(cursor + 1)
            cursor += 1
            if endpoint is not None:
                return endpoint
            continue
        if character == "#" and not frame.word_started:
            newline = content.find("\n", cursor + 1, limit)
            if newline < 0:
                _note_parser_diagnostic(
                    diagnostics,
                    ParserCause.BOUNDARY_UNRESOLVED,
                    ParserCheck.EXPANSION_BOUNDARY,
                    start,
                    limit,
                    ParserUncheckedScope.EXPRESSION,
                    full_input_checked=limit == len(content),
                )
                cache_failure()
                return None
            cursor = newline
            continue
        if character.isspace() or character in ";|&":
            frame.word_started = False
        else:
            frame.word_started = True
        cursor += 1

    _note_parser_diagnostic(
        diagnostics,
        ParserCause.BOUNDARY_UNRESOLVED,
        ParserCheck.EXPANSION_BOUNDARY,
        start,
        limit,
        ParserUncheckedScope.EXPRESSION,
        full_input_checked=limit == len(content),
    )
    cache_failure()
    return None


def _skip_parameter_expansion(
    content: str,
    start: int,
    limit: int,
    check_runtime: Callable[[], None] | None = None,
    end_cache: dict[int, _ParameterExpansionEnd] | None = None,
    substitution_end_cache: dict[int, int | None] | None = None,
    backtick_end_cache: dict[int, int | None] | None = None,
    inherited_double_quote: bool = False,
    inherited_quote_closed: list[bool] | None = None,
    diagnostics: list[StaticParserDiagnostic] | None = None,
) -> int | None:
    if end_cache is not None and start in end_cache:
        cached = end_cache[start]
        if cached.end is None:
            _note_parser_diagnostic(
                diagnostics,
                ParserCause.BOUNDARY_UNRESOLVED,
                ParserCheck.EXPANSION_BOUNDARY,
                start,
                limit,
                ParserUncheckedScope.EXPRESSION,
            )
        if inherited_quote_closed is not None:
            inherited_quote_closed[0] = cached.inherited_quote_closed
        return cached.end
    if start + 1 >= limit or content[start] != "$":
        return None
    next_character = content[start + 1]
    if next_character == "{":
        return _skip_shell_delimited_expansion(
            content,
            start,
            limit,
            "parameter",
            check_runtime,
            end_cache,
            substitution_end_cache,
            backtick_end_cache,
            inherited_double_quote,
            inherited_quote_closed,
            diagnostics,
        )
    if next_character.isalpha() or next_character == "_":
        cursor = start + 2
        while cursor < limit and (content[cursor].isalnum() or content[cursor] == "_"):
            cursor += 1
        return cursor
    if next_character.isdigit() or next_character in "*@#?$!-":
        return start + 2
    return None


def _is_ifs_expansion(content: str, start: int, end: int) -> bool:
    return content[start:end] in {"$IFS", "${IFS}"}


def _consume_printf_invocation(
    next_word: Callable[[], str | None],
    *,
    runtime_command_context: bool = False,
    diagnostics: list[StaticParserDiagnostic] | None = None,
    source_end: int = 1,
) -> tuple[bool, bool]:
    """Resolve an allowlisted invocation; return ``(recognized, exact)``."""
    pending: str | None = None
    for _ in range(4):
        word = pending if pending is not None else next_word()
        pending = None
        if word is None:
            return False, False
        if _DYNAMIC_SHELL_WORD_SENTINEL in word:
            # A command substitution or complex expansion participates in the
            # invocation or wrapper word. Its basename is not deterministic.
            _note_parser_diagnostic(
                diagnostics,
                ParserCause.RUNTIME_DEPENDENT,
                ParserCheck.PRINTF_INVOCATION,
                0,
                source_end,
                ParserUncheckedScope.COMMAND_WORD,
            )
            return True, False
        command = word.casefold().rsplit("/", 1)[-1]
        if _RUNTIME_SHELL_PARAMETER_SENTINEL in word and command in {
            "printf",
            "command",
            "builtin",
            "env",
        }:
            # A known basename does not make a runtime-selected executable exact.
            _note_parser_diagnostic(
                diagnostics,
                ParserCause.RUNTIME_DEPENDENT,
                ParserCheck.PRINTF_INVOCATION,
                0,
                source_end,
                ParserUncheckedScope.COMMAND_WORD,
            )
            return True, False
        if _RUNTIME_SHELL_PARAMETER_SENTINEL in command:
            # An opaque basename can still participate in printf reconstruction.
            # Require bounded invocation evidence or destructive outer operands,
            # rather than reclassifying ordinary runtime-parameter notation.
            if runtime_command_context:
                _note_parser_diagnostic(
                    diagnostics,
                    ParserCause.RUNTIME_DEPENDENT,
                    ParserCheck.PRINTF_INVOCATION,
                    0,
                    source_end,
                    ParserUncheckedScope.COMMAND_WORD,
                )
                return True, False
            characters = 0
            for _ in range(_PRINTF_STATIC_ARGUMENTS):
                operand = next_word()
                if operand is None:
                    break
                characters += len(operand)
                if characters > _PRINTF_STATIC_CHARS:
                    _note_parser_diagnostic(
                        diagnostics,
                        ParserCause.RESOURCE_LIMIT,
                        ParserCheck.PRINTF_INPUT_CHARACTERS,
                        0,
                        source_end,
                        ParserUncheckedScope.COMMAND_WORD,
                        measurements={
                            "observed_characters": characters,
                            "limit_characters": _PRINTF_STATIC_CHARS,
                        },
                        lower_bound=True,
                    )
                    return True, False
                if (
                    operand.casefold().rsplit("/", 1)[-1] == "printf"
                    or _PRINTF_FORMAT_CONVERSION_RE.search(operand) is not None
                ):
                    _note_parser_diagnostic(
                        diagnostics,
                        ParserCause.RUNTIME_DEPENDENT,
                        ParserCheck.PRINTF_INVOCATION,
                        0,
                        source_end,
                        ParserUncheckedScope.COMMAND_WORD,
                    )
                    return True, False
            else:
                _note_parser_diagnostic(
                    diagnostics,
                    ParserCause.RESOURCE_LIMIT,
                    ParserCheck.PRINTF_ARGUMENT_COUNT,
                    0,
                    source_end,
                    ParserUncheckedScope.COMMAND_WORD,
                    measurements={
                        "observed_records": _PRINTF_STATIC_ARGUMENTS,
                        "limit_records": _PRINTF_STATIC_ARGUMENTS,
                    },
                    lower_bound=True,
                )
                return True, False
            return False, False
        if command == "printf":
            return True, True
        if command == "command":
            while True:
                word = next_word()
                if word == "-p":
                    continue
                if word == "--":
                    word = next_word()
                break
            if word is None or word.startswith("-"):
                return False, False
            pending = word
            continue
        if command == "builtin":
            word = next_word()
            if word == "--":
                word = next_word()
            if word is None or word.startswith("-"):
                return False, False
            pending = word
            continue
        if command != "env":
            return False, False
        while (word := next_word()) is not None:
            if word == "--":
                pending = next_word()
                break
            # ``env`` assignment values may contain quoted newlines and other
            # characters. Only the portable name and first ``=`` determine
            # whether this word is an assignment operand.
            if re.match(r"[A-Za-z_][A-Za-z0-9_]*=", word) is not None:
                continue
            if word in {"-i", "--ignore-environment"}:
                continue
            if word in {"-u", "--unset", "-C", "--chdir"}:
                if next_word() is None:
                    _note_parser_diagnostic(
                        diagnostics,
                        ParserCause.UNSUPPORTED_SYNTAX,
                        ParserCheck.PRINTF_INVOCATION,
                        0,
                        source_end,
                        ParserUncheckedScope.COMMAND_WORD,
                    )
                    return True, False
                continue
            if re.fullmatch(r"(?:--unset|--chdir)=.+", word) is not None:
                continue
            if word.startswith("-"):
                # Preserve the existing fail-closed contract for unsupported
                # env options even though the final command cannot be resolved.
                _note_parser_diagnostic(
                    diagnostics,
                    ParserCause.UNSUPPORTED_SYNTAX,
                    ParserCheck.PRINTF_INVOCATION,
                    0,
                    source_end,
                    ParserUncheckedScope.COMMAND_WORD,
                )
                return True, False
            pending = word
            break
        if pending is None:
            return False, False
    # The allowlisted wrapper chain exceeded the deterministic depth budget.
    # Treat it as recognized but inexact so callers record a coverage gap
    # instead of silently declaring a potentially reconstructed command safe.
    _note_parser_diagnostic(
        diagnostics,
        ParserCause.RESOURCE_LIMIT,
        ParserCheck.WRAPPER_DEPTH,
        0,
        source_end,
        ParserUncheckedScope.COMMAND_WORD,
        measurements={"observed_depth": 4, "limit_depth": 4},
        lower_bound=True,
    )
    return True, False


def _printf_invocation_arguments(
    inner: str,
    *,
    runtime_command_context: bool = False,
    check_runtime: Callable[[], None] | None = None,
    diagnostics: list[StaticParserDiagnostic] | None = None,
) -> tuple[bool, list[str]]:
    """Parse direct or allowlisted wrapper invocations of shell ``printf``."""
    runtime_check = check_runtime or (lambda: None)

    cursor = 0
    limited = False

    def next_word() -> str | None:
        nonlocal cursor, limited
        word, cursor, word_limited = _next_shell_invocation_word(
            inner,
            cursor,
            runtime_check,
            diagnostics=diagnostics,
        )
        limited = limited or word_limited or (word is None and cursor < len(inner))
        return word

    recognized, exact = _consume_printf_invocation(
        next_word,
        runtime_command_context=runtime_command_context,
        diagnostics=diagnostics,
        source_end=len(inner),
    )
    if not recognized or not exact or limited:
        return recognized, []

    arguments: list[str] = []
    while cursor < len(inner):
        word = next_word()
        if word is None:
            return (True, []) if limited or cursor < len(inner) else (True, arguments)
        arguments.append(word)
    return True, arguments


def _is_value_parameter_expansion(content: str, start: int, end: int) -> bool:
    """Return whether a braced expansion and its nested expansions substitute only values."""
    expansion = content[start:end]
    if any(marker in expansion for marker in _EXECUTABLE_EXPANSION_MARKERS):
        return False
    return all(
        _VALUE_PARAMETER_EXPANSION_HEAD_RE.match(expansion, nested.start()) is not None
        for nested in _VALUE_PARAMETER_EXPANSION_START_RE.finditer(expansion)
    )


def _invocation_expansion_marker(content: str, start: int, end: int) -> str:
    """Distinguish runtime-only parameters from possible command reconstruction."""
    if content.startswith("$(", start) or (
        content.startswith("${", start) and not _is_value_parameter_expansion(content, start, end)
    ):
        # Other parameter expansions may contain nested command substitutions
        # or evaluate a value as code.
        return _DYNAMIC_SHELL_WORD_SENTINEL
    # ``${NAME:-default}``, ``${ARR[@]}`` and similar forms select the same
    # runtime value as ``$NAME`` and need the same printf invocation evidence.
    return _RUNTIME_SHELL_PARAMETER_SENTINEL


def _next_shell_invocation_word(
    content: str,
    start: int,
    check_runtime: Callable[[], None],
    parameter_end_cache: dict[int, _ParameterExpansionEnd] | None = None,
    substitution_end_cache: dict[int, int | None] | None = None,
    backtick_end_cache: dict[int, int | None] | None = None,
    *,
    diagnostics: list[StaticParserDiagnostic] | None = None,
    source_start: list[int] | None = None,
    leading_process_substitution: bool = False,
) -> tuple[str | None, int, bool]:
    """Read one direct shell word without traversing nested substitutions."""
    cursor = start
    limit = len(content)

    def unresolved_boundary(position: int, *, full_input_checked: bool = False) -> None:
        _note_parser_diagnostic(
            diagnostics,
            ParserCause.BOUNDARY_UNRESOLVED,
            ParserCheck.COMMAND_WORD_BOUNDARY,
            position,
            limit,
            ParserUncheckedScope.COMMAND_WORD,
            full_input_checked=full_input_checked,
        )

    next_runtime_check = ((cursor + 4095) // 4096) * 4096
    while cursor < limit:
        if cursor >= next_runtime_check:
            check_runtime()
            next_runtime_check = cursor + 4096
        if content[cursor].isspace():
            cursor += 1
            continue
        if content.startswith("\\\r\n", cursor):
            cursor += 3
            continue
        if content.startswith("\\\n", cursor):
            cursor += 2
            continue
        break
    if source_start is not None:
        if source_start:
            source_start[0] = cursor
        else:
            source_start.append(cursor)
    start = cursor
    if cursor >= limit:
        return None, cursor, False
    if content[cursor] in ");|&()<>" and not (
        leading_process_substitution and content.startswith(("<(", ">("), cursor)
    ):
        return None, cursor, False

    output: list[str] = []
    quote: str | None = None
    ansi_c_quote = False
    word_started = False
    unquoted_characters = 0
    while cursor < limit:
        if cursor >= next_runtime_check:
            check_runtime()
            next_runtime_check = cursor + 4096
        character = content[cursor]
        if quote is not None:
            if character == quote:
                quote = None
                ansi_c_quote = False
            elif character == "\\" and ansi_c_quote:
                decoded, cursor = _decode_ansi_c_escape(content, cursor, limit)
                if "\x00" in decoded:
                    closing_quote = _skip_ansi_c_quote_tail(content, cursor, limit)
                    if closing_quote is None:
                        unresolved_boundary(start, full_input_checked=True)
                        return None, cursor, True
                    cursor = closing_quote
                    continue
                output.append(decoded)
                word_started = True
                continue
            elif quote == '"' and character == "\\" and cursor + 1 < limit:
                escaped = content[cursor + 1]
                if escaped == "\n":
                    cursor += 2
                    continue
                if escaped == "\r" and cursor + 2 < limit and content[cursor + 2] == "\n":
                    cursor += 3
                    continue
                if escaped in '$`"\\':
                    output.append(escaped)
                else:
                    output.extend(("\\", escaped))
                cursor += 2
                continue
            elif quote == '"' and character == "$":
                inherited_quote_closed = [False]
                if cursor + 1 < limit and content[cursor + 1] == "(":
                    parameter_end = _skip_command_substitution(
                        content,
                        cursor,
                        limit,
                        check_runtime,
                        substitution_end_cache,
                        parameter_end_cache,
                        backtick_end_cache,
                        diagnostics=diagnostics,
                    )
                else:
                    parameter_end = _skip_parameter_expansion(
                        content,
                        cursor,
                        limit,
                        check_runtime,
                        parameter_end_cache,
                        substitution_end_cache,
                        backtick_end_cache,
                        True,
                        inherited_quote_closed,
                        diagnostics=diagnostics,
                    )
                if parameter_end is None:
                    if cursor + 1 < limit and content[cursor + 1] in "({":
                        unresolved_boundary(cursor)
                        return None, cursor, True
                    output.append("$")
                    word_started = True
                    cursor += 1
                    continue
                output.append(_invocation_expansion_marker(content, cursor, parameter_end))
                word_started = True
                cursor = parameter_end
                if inherited_quote_closed[0]:
                    quote = None
                    ansi_c_quote = False
                continue
            elif quote == '"' and character == "`":
                substitution_end = _skip_backtick_substitution(
                    content,
                    cursor,
                    limit,
                    check_runtime,
                    backtick_end_cache,
                    parameter_end_cache,
                    substitution_end_cache,
                    diagnostics=diagnostics,
                )
                if substitution_end is None:
                    unresolved_boundary(cursor)
                    return None, cursor, True
                output.append(_DYNAMIC_SHELL_WORD_SENTINEL)
                word_started = True
                cursor = substitution_end
                continue
            else:
                output.append(character)
        elif character in "'\"":
            quote = character
            ansi_c_quote = False
            word_started = True
        elif character == "\\" and cursor + 1 < limit:
            escaped = content[cursor + 1]
            if escaped == "\n":
                cursor += 2
                continue
            if escaped == "\r" and cursor + 2 < limit and content[cursor + 2] == "\n":
                cursor += 3
                continue
            output.append(escaped)
            word_started = True
            cursor += 2
            continue
        elif character == "$":
            if cursor + 1 < limit and content[cursor + 1] in "'\"":
                quote = content[cursor + 1]
                ansi_c_quote = quote == "'"
                word_started = True
                cursor += 2
                continue
            if cursor + 1 < limit and content[cursor + 1] == "(":
                parameter_end = _skip_command_substitution(
                    content,
                    cursor,
                    limit,
                    check_runtime,
                    substitution_end_cache,
                    parameter_end_cache,
                    backtick_end_cache,
                    diagnostics=diagnostics,
                )
            else:
                parameter_end = _skip_parameter_expansion(
                    content,
                    cursor,
                    limit,
                    check_runtime,
                    parameter_end_cache,
                    substitution_end_cache,
                    backtick_end_cache,
                    diagnostics=diagnostics,
                )
            if parameter_end is None:
                if cursor + 1 < limit and content[cursor + 1] in "({":
                    unresolved_boundary(cursor)
                    return None, cursor, True
                output.append("$")
                word_started = True
                cursor += 1
                continue
            # A simple runtime parameter does not invoke the printf evaluator.
            output.append(_invocation_expansion_marker(content, cursor, parameter_end))
            word_started = True
            cursor = parameter_end
            continue
        elif character == "`":
            substitution_end = _skip_backtick_substitution(
                content,
                cursor,
                limit,
                check_runtime,
                backtick_end_cache,
                parameter_end_cache,
                substitution_end_cache,
                diagnostics=diagnostics,
            )
            if substitution_end is None:
                unresolved_boundary(cursor)
                return None, cursor, True
            output.append(_DYNAMIC_SHELL_WORD_SENTINEL)
            word_started = True
            cursor = substitution_end
            continue
        elif character in "<>" and cursor + 1 < limit and content[cursor + 1] == "(":
            substitution_end = _skip_command_substitution(
                content,
                cursor,
                limit,
                check_runtime,
                substitution_end_cache,
                parameter_end_cache,
                backtick_end_cache,
                diagnostics=diagnostics,
            )
            if substitution_end is None:
                unresolved_boundary(cursor)
                return None, cursor, True
            output.append(_DYNAMIC_SHELL_WORD_SENTINEL)
            word_started = True
            cursor = substitution_end
            continue
        elif character.isspace() or character in ";|&()<>":
            break
        elif character == "#" and not word_started:
            return None, cursor, False
        else:
            output.append(character)
            word_started = True
            unquoted_characters += 1
            if unquoted_characters > _SHELL_COMMAND_WORD_CHARS:
                _note_parser_diagnostic(
                    diagnostics,
                    ParserCause.RESOURCE_LIMIT,
                    ParserCheck.COMMAND_WORD_CHARACTERS,
                    start,
                    cursor + 1,
                    ParserUncheckedScope.COMMAND_WORD,
                    measurements={
                        "observed_characters": unquoted_characters,
                        "limit_characters": _SHELL_COMMAND_WORD_CHARS,
                    },
                    lower_bound=True,
                )
                return None, cursor, True
        cursor += 1
    if quote is not None:
        unresolved_boundary(start, full_input_checked=True)
        return None, cursor, True
    return "".join(output) if word_started else None, cursor, False


_PRINTF_HEX_ESCAPE_RE = re.compile(r"[0-9A-Fa-f]{1,2}")
_PRINTF_OCTAL_ESCAPE_RE = re.compile(r"[0-7]{1,3}")
_PRINTF_ESCAPE_SEQUENCES = {
    "a": "\a",
    "b": "\b",
    "e": "\x1b",
    "E": "\x1b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
    "\\": "\\",
    "'": "'",
    '"': '"',
}


def _decode_printf_escapes(text: str) -> str | None:
    """Decode bounded escapes as modeled by Bash builtin/coreutils ``printf``.

    This intentionally models the Bash/GNU-coreutils escape subset used by the
    static parser.  Extensions such as ``\\xHH``, ``\\e``/``\\E``, ``\\'`` and
    ``\\"`` are not portable to a dash ``printf`` and are not a general POSIX
    guarantee.  Unknown or trailing escapes make the substitution undecidable
    and return ``None`` so the caller keeps its fail-closed behaviour.
    """
    output: list[str] = []
    cursor = 0
    while cursor < len(text):
        character = text[cursor]
        if character != "\\":
            output.append(character)
            cursor += 1
            continue
        if cursor + 1 >= len(text):
            return None
        escaped = text[cursor + 1]
        simple = _PRINTF_ESCAPE_SEQUENCES.get(escaped)
        if simple is not None:
            output.append(simple)
            cursor += 2
            continue
        if escaped == "x":
            digits = _PRINTF_HEX_ESCAPE_RE.match(text, cursor + 2)
            if digits is None:
                return None
            output.append(chr(int(digits.group(0), 16)))
            cursor = digits.end()
            continue
        if escaped in "01234567":
            digits = _PRINTF_OCTAL_ESCAPE_RE.match(text, cursor + 1)
            if digits is None:
                return None
            value = int(digits.group(0), 8)
            if value > 0o377:
                return None
            output.append(chr(value))
            cursor = digits.end()
            continue
        return None
    return "".join(output)


def _bounded_printf_word(result: str) -> str | None:
    """Return a decoded ``printf`` result only when it stays one inert word."""
    if _PRINTF_SAFE_WORD_RE.fullmatch(result) is None:
        return None
    return result


def _has_literal_printf_arguments(inner: str) -> bool:
    """Prove literal arguments without treating shell expansions as text."""
    quote: str | None = None
    cursor = 0
    while cursor < len(inner):
        character = inner[cursor]
        if character == "\\" and quote != "'":
            if cursor + 1 >= len(inner):
                return False
            cursor += 3 if inner.startswith("\\\r\n", cursor) else 2
            continue
        if quote is not None:
            if character == quote:
                quote = None
            elif quote == '"' and character in "$`":
                return False
        elif character in "'\"":
            quote = character
        elif character in "$`;|&<>()\r\n*?[]{}~#":
            return False
        cursor += 1
    if quote is not None:
        return False
    command, _, limited = _next_shell_invocation_word(inner, 0, lambda: None)
    return command == "printf" and not limited


def _static_printf_substitution(
    content: str,
    start: int,
    end: int,
    *,
    backtick: bool = False,
    quoted_result: bool = False,
    diagnostics: list[StaticParserDiagnostic] | None = None,
) -> str | None:
    """Evaluate bounded literal ``printf`` output, preserving its quoting."""
    inner_start = start + (1 if backtick else 2)
    inner_end = end - 1
    inner = content[inner_start:inner_end]

    def failed(
        check: ParserCheck = ParserCheck.PRINTF_FORMAT,
        cause: ParserCause = ParserCause.UNSUPPORTED_SYNTAX,
        measurements: ResourceMeasurements | None = None,
        *,
        lower_bound: bool = False,
    ) -> None:
        _note_parser_diagnostic(
            diagnostics,
            cause,
            check,
            start,
            end,
            ParserUncheckedScope.COMMAND_WORD,
            measurements=measurements,
            lower_bound=lower_bound,
        )

    if len(inner) > _PRINTF_STATIC_CHARS:
        failed(
            ParserCheck.PRINTF_INPUT_CHARACTERS,
            ParserCause.RESOURCE_LIMIT,
            {"observed_characters": len(inner), "limit_characters": _PRINTF_STATIC_CHARS},
        )
        return None
    literal_result = quoted_result and not backtick and _has_literal_printf_arguments(inner)
    if quoted_result and not backtick and not literal_result:
        command, _, _ = _next_shell_invocation_word(inner, 0, lambda: None)
        if command == "printf":
            failed()
            return None
    invocation_diagnostics: list[StaticParserDiagnostic] = []
    recognized, arguments = _printf_invocation_arguments(inner, diagnostics=invocation_diagnostics)
    if not recognized or (not literal_result and re.search(r"[;|&<>\r\n]", inner) is not None):
        if diagnostics is not None and invocation_diagnostics:
            diagnostics.extend(item.shifted(inner_start) for item in invocation_diagnostics[:1])
        failed(ParserCheck.PRINTF_INVOCATION)
        return None
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    elif literal_result and arguments and arguments[0].startswith("-"):
        # In particular, ``-v`` assigns a variable instead of producing output.
        failed()
        return None
    if not arguments:
        if diagnostics is not None and invocation_diagnostics:
            diagnostics.extend(item.shifted(inner_start) for item in invocation_diagnostics[:1])
        failed(ParserCheck.PRINTF_INVOCATION)
        return None
    if len(arguments) > _PRINTF_STATIC_ARGUMENTS + 1:
        failed(
            ParserCheck.PRINTF_ARGUMENT_COUNT,
            ParserCause.RESOURCE_LIMIT,
            {"observed_records": len(arguments) - 1, "limit_records": _PRINTF_STATIC_ARGUMENTS},
        )
        return None
    format_word, *values = arguments
    value_pattern = _PRINTF_QUOTED_WORD_RE if literal_result else _PRINTF_STATIC_WORD_RE
    if _PRINTF_FORMAT_WORD_RE.fullmatch(format_word) is None or any(
        value_pattern.fullmatch(value) is None for value in values
    ):
        longest = max(map(len, arguments))
        if longest > 64:
            failed(
                ParserCheck.PRINTF_ARGUMENT_CHARACTERS,
                ParserCause.RESOURCE_LIMIT,
                {"observed_characters": longest, "limit_characters": 64},
            )
        elif any(
            marker in word
            for word in arguments
            for marker in (_DYNAMIC_SHELL_WORD_SENTINEL, _RUNTIME_SHELL_PARAMETER_SENTINEL)
        ):
            failed(ParserCheck.PRINTF_FORMAT, ParserCause.RUNTIME_DEPENDENT)
        else:
            failed()
        return None

    segments: list[str | None] = []
    literal: list[str] = []

    def flush_literal() -> bool:
        # POSIX ``printf`` decodes escapes in the format operand before any
        # conversion runs, so decode each literal segment in place. Decoding
        # after the conversion split keeps an escaped percent sign from
        # re-entering conversion parsing.
        text = "".join(literal)
        literal.clear()
        if "\\" in text:
            decoded = _decode_printf_escapes(text)
            if decoded is None:
                return False
            text = decoded
        segments.append(text)
        return True

    cursor = 0
    while cursor < len(format_word):
        character = format_word[cursor]
        if character != "%":
            literal.append(character)
            cursor += 1
            continue
        if cursor + 1 >= len(format_word) or format_word[cursor + 1] not in "s%":
            failed()
            return None
        if literal and not flush_literal():
            failed()
            return None
        conversion = format_word[cursor + 1]
        segments.append(None if conversion == "s" else "%")
        cursor += 2
    if literal and not flush_literal():
        failed()
        return None

    conversions = sum(segment is None for segment in segments)

    def bounded_result(result: str) -> str | None:
        if literal_result:
            # Quoted command substitution keeps one word and strips trailing
            # LF bytes only. Its internal whitespace never becomes shell syntax.
            if len(result) > _PRINTF_STATIC_CHARS:
                failed(
                    ParserCheck.PRINTF_OUTPUT_CHARACTERS,
                    ParserCause.RESOURCE_LIMIT,
                    {"observed_characters": len(result), "limit_characters": _PRINTF_STATIC_CHARS},
                )
                return None
            if "\x00" in result:
                failed()
                return None
            return result.rstrip("\n")
        bounded = _bounded_printf_word(result)
        if bounded is None:
            if len(result) > 64:
                failed(
                    ParserCheck.PRINTF_OUTPUT_CHARACTERS,
                    ParserCause.RESOURCE_LIMIT,
                    {"observed_characters": len(result), "limit_characters": 64},
                )
            else:
                failed()
        return bounded

    if conversions == 0:
        return bounded_result("".join(segments))
    remaining = iter(values)
    output: list[str] = []
    cycles = max(1, (len(values) + conversions - 1) // conversions)
    for _ in range(cycles):
        for segment in segments:
            output.append(next(remaining, "") if segment is None else segment)
        if sum(len(piece) for piece in output) > (_PRINTF_STATIC_CHARS if literal_result else 64):
            failed(
                ParserCheck.PRINTF_OUTPUT_CHARACTERS,
                ParserCause.RESOURCE_LIMIT,
                {
                    "observed_characters": sum(map(len, output)),
                    "limit_characters": _PRINTF_STATIC_CHARS if literal_result else 64,
                },
                lower_bound=True,
            )
            return None
    return bounded_result("".join(output))


def _may_have_destructive_outer_operands(content: str) -> bool:
    """Conservatively prove whether one view can contain destructive operands."""
    target_evidence = (
        any(marker in content for marker in "/~*?")
        or _SHELL_ROOT_TARGET_ESCAPE_RE.search(content) is not None
    )
    option_evidence = (
        _RECURSIVE_OPTION_SOURCE_RE.search(content) is not None
        and _FORCE_OPTION_SOURCE_RE.search(content) is not None
    ) or _has_constructed_recursive_force_option_source(content)
    return target_evidence and option_evidence


def _has_constructed_recursive_force_option_source(content: str) -> bool:
    """Return whether a bounded option word can construct both ``r`` and ``f``."""
    for hyphen in re.finditer(r"-(?=\S)", content):
        fragment = content[hyphen.start() : hyphen.start() + 128]
        command_substitution = fragment.startswith("-$(")
        if command_substitution:
            close = fragment.find(")")
            if close >= 0:
                fragment = fragment[: close + 1]
        else:
            boundary = re.search(r"[\s;|&]", fragment)
            if boundary is not None:
                fragment = fragment[: boundary.start()]
        if not any(marker in fragment for marker in ("\\", "'", '"', "{", "}", "$(")):
            continue
        lowered = fragment.casefold()
        if "r" in lowered and "f" in lowered:
            return True
    return False


def _is_powershell_replace_value_context(
    content: str,
    start: int,
    end: int,
    inner: str,
) -> bool:
    """Recognize a bounded PowerShell ``-replace`` value statement."""
    if (
        _POWERSHELL_REPLACE_EXPRESSION_RE.search(inner) is None
        or any(separator in inner for separator in ";|&\n")
        or any(marker in inner for marker in ("$(", "`", "<(", ">("))
    ):
        return False

    line_start = content.rfind("\n", 0, start) + 1
    line_end = content.find("\n", end)
    if line_end < 0:
        line_end = len(content)
    placeholder = "\ue003"
    statement = (content[line_start:start] + placeholder + content[end:line_end]).strip()

    if statement.startswith("|") and statement.endswith("|"):
        cells = [cell.strip() for cell in statement[1:-1].split("|")]
        value_cells = [cell for cell in cells if placeholder in cell]
        if len(value_cells) != 1:
            return False
        statement = value_cells[0]
    container = re.match(
        r"(?:(?:>[ \t]*)+|(?:[-*+]|[0-9]+[.)]|#{1,6})[ \t]+)",
        statement,
    )
    if container is not None:
        statement = statement[container.end() :].strip()

    # Raw Markdown inline code retains its delimiter; the normalized Markdown
    # view blanks it. Support the explicit documentary prefix in both views
    # without trusting a fenced-code language label as a security boundary.
    prefix = re.match(
        r"(?:The[ \t]+following[ \t]+is[ \t]+(?:a[ \t]+)?)?"
        r"PowerShell(?:[ \t]+(?:example|snippet|command))?[ \t]*:[ \t]*",
        statement,
        re.IGNORECASE,
    )
    if prefix is not None:
        statement = statement[prefix.end() :].strip()
        if statement.endswith("."):
            statement = statement[:-1].rstrip()
    inline = re.fullmatch(r"(?P<ticks>`+)(?P<body>.*)(?P=ticks)", statement)
    if inline is not None:
        statement = inline.group("body").strip()

    value = f'"{placeholder}"'
    shapes = (
        rf"Write-Output[ \t]+(?:{re.escape(value)}|\([ \t]*{re.escape(value)}[ \t]*\))",
        rf"Write-Host(?:[ \t]+-NoNewline)?[ \t]+"
        rf"(?:{re.escape(value)}|\([ \t]*{re.escape(value)}[ \t]*\))",
        rf"{re.escape(value)}[ \t]*\|[ \t]*Write-Output",
        rf"\$[A-Za-z_][A-Za-z0-9_]*[ \t]*=[ \t]*{re.escape(value)}",
    )
    return any(re.fullmatch(shape, statement, re.IGNORECASE) is not None for shape in shapes)


def _is_printf_substitution(
    content: str,
    start: int,
    end: int,
    *,
    backtick: bool = False,
    check_command_context: bool = True,
    check_runtime: Callable[[], None] | None = None,
    diagnostics: list[StaticParserDiagnostic] | None = None,
) -> bool:
    """Return whether a substitution invokes the bounded ``printf`` evaluator."""
    runtime_check = check_runtime or (lambda: None)
    inner_start = start + (1 if backtick else 2)
    inner_end = end - 1
    inner = content[inner_start:inner_end]
    if not backtick and _is_powershell_replace_value_context(content, start, end, inner):
        return False
    invocation_diagnostics: list[StaticParserDiagnostic] = []
    recognized, _ = _printf_invocation_arguments(
        inner, check_runtime=runtime_check, diagnostics=invocation_diagnostics
    )
    if recognized or not check_command_context:
        if recognized and diagnostics is not None:
            diagnostics.extend(item.shifted(inner_start) for item in invocation_diagnostics[:1])
        return recognized
    if "$" not in inner:
        return False
    command_start, body_start = start, end
    if start > 0 and content[start - 1] == '"' and end < len(content) and content[end] == '"':
        command_start -= 1
        body_start += 1
    tail = content[body_start : body_start + _ROOT_GLOB_COMMAND_CHARS]
    if "\\" not in tail and ("-" not in tail or not any(marker in tail for marker in "/~*?")):
        # Without option and target characters the bounded tokenizer cannot
        # produce a destructive root command. Keep repeated parameter notation
        # cheap; escapes still require tokenization because they can encode both.
        return False
    if (
        not any(marker in tail for marker in ("\\", "'", '"', "{", "}"))
        and "printf" not in tail.casefold()
        and (
            _RECURSIVE_OPTION_SOURCE_RE.search(tail) is None
            or _FORCE_OPTION_SOURCE_RE.search(tail) is None
        )
    ):
        # Plain options must contain recursive and force spelling in the source.
        # Quoting, escapes, braces, or printf can construct those spellings, so
        # keep those cases on the full tokenizer path.
        return False
    possible_runtime, _ = _printf_invocation_arguments(
        inner,
        runtime_command_context=True,
        check_runtime=runtime_check,
    )
    if not possible_runtime:
        return False
    argument_diagnostics: list[StaticParserDiagnostic] = []
    tokens, command_end, exhausted = _bounded_shell_tokens(
        content,
        command_start,
        body_start,
        check_runtime=runtime_check,
        diagnostics=argument_diagnostics,
    )
    if exhausted:
        if diagnostics is not None:
            diagnostics.extend(argument_diagnostics[:1])
        return True
    brace_diagnostic = _brace_expansion_diagnostic(tokens, command_start, command_end)
    if brace_diagnostic is not None:
        if diagnostics is not None:
            diagnostics.append(brace_diagnostic)
        return True
    destructive = _has_destructive_root_glob(tokens) or _has_destructive_root_path(tokens)
    if destructive:
        _note_parser_diagnostic(
            diagnostics,
            ParserCause.RUNTIME_DEPENDENT,
            ParserCheck.PRINTF_INVOCATION,
            start,
            end,
            ParserUncheckedScope.COMMAND_WORD,
        )
    return destructive


def _printf_substitution_result(
    content: str,
    start: int,
    end: int,
    check_runtime: Callable[[], None],
    *,
    backtick: bool = False,
    quoted_result: bool = False,
    recognize_limit: bool = True,
    check_command_context: bool = True,
) -> tuple[str | None, bool, StaticParserDiagnostic | None]:
    """Keep reconstruction failure evidence only when the caller flags its scope."""
    value_diagnostics: list[StaticParserDiagnostic] = []
    value = _static_printf_substitution(
        content,
        start,
        end,
        backtick=backtick,
        quoted_result=quoted_result,
        diagnostics=value_diagnostics,
    )
    invocation_diagnostics: list[StaticParserDiagnostic] = []
    limited = (
        value is None
        and recognize_limit
        and _is_printf_substitution(
            content,
            start,
            end,
            backtick=backtick,
            check_command_context=check_command_context,
            check_runtime=check_runtime,
            diagnostics=invocation_diagnostics,
        )
    )
    if not limited:
        return value, False, None
    diagnostics = invocation_diagnostics or value_diagnostics
    return value, True, diagnostics[0]


def _skip_backtick_substitution(
    content: str,
    start: int,
    limit: int,
    check_runtime: Callable[[], None] | None = None,
    end_cache: dict[int, int | None] | None = None,
    parameter_end_cache: dict[int, _ParameterExpansionEnd] | None = None,
    substitution_end_cache: dict[int, int | None] | None = None,
    *,
    diagnostics: list[StaticParserDiagnostic] | None = None,
) -> int | None:
    if end_cache is not None and start in end_cache:
        if end_cache[start] is None:
            _note_parser_diagnostic(
                diagnostics,
                ParserCause.BOUNDARY_UNRESOLVED,
                ParserCheck.EXPANSION_BOUNDARY,
                start,
                limit,
                ParserUncheckedScope.EXPRESSION,
            )
        return end_cache[start]
    return _skip_shell_delimited_expansion(
        content,
        start,
        limit,
        "backtick",
        check_runtime,
        parameter_end_cache,
        substitution_end_cache,
        end_cache,
        diagnostics=diagnostics,
    )


def _previous_word_end(content: str, start: int) -> int | None:
    """Return where the previous word in this clause ends, or ``None`` at a clause start."""
    cursor = start
    while cursor > 0 and content[cursor - 1] in " \t":
        cursor -= 1
    if cursor == 0 or content[cursor - 1] in _SHELL_CLAUSE_BOUNDARY_CHARACTERS:
        return None
    return cursor


def _is_assignment_word(content: str, start: int) -> bool:
    """Return whether ``start`` begins a prefix assignment word or its quoted value."""
    word_start = start
    # The exhaustion sweep already skips the checks for a quote after ``=``.
    # Recognizing it here keeps the parser consistent for any caller and avoids
    # evaluating printf reconstruction whose result would be discarded.
    if 0 < start < len(content) and content[start] in "'\"" and content[start - 1] == "=":
        word_start = start - 1
        if word_start > 0 and content[word_start - 1] == "+":
            word_start -= 1
        while word_start > 0 and content[word_start - 1] in _SHELL_ASSIGNMENT_NAME_CHARACTERS:
            word_start -= 1
    prefix = _SHELL_ASSIGNMENT_PREFIX_RE.match(content, word_start)
    if prefix is None or (word_start != start and prefix.end() != start):
        return False
    # The shell recognizes assignments only before the command name. After
    # ``alias``, ``echo`` or a path prefix, ``NAME=value`` is an ordinary word.
    previous_end = _previous_word_end(content, word_start)
    if previous_end is None:
        return True
    if previous_end == word_start:
        return False
    previous_start = previous_end
    while (
        previous_start > 0
        and content[previous_start - 1] not in " \t" + _SHELL_CLAUSE_BOUNDARY_CHARACTERS
    ):
        previous_start -= 1
    return (
        content[previous_start:previous_end] in _SHELL_ASSIGNMENT_PREFIX_WORDS
        and _previous_word_end(content, previous_start) is None
    )


def _parse_shell_command_word(
    content: str,
    start: int,
    parameter_end_cache: dict[int, _ParameterExpansionEnd] | None = None,
    substitution_end_cache: dict[int, int | None] | None = None,
    backtick_end_cache: dict[int, int | None] | None = None,
    *,
    check_runtime: Callable[[], None] | None = None,
    owned_word_positions: set[int] | None = None,
    enclosing_delimiter: str | None = None,
    quoted_printf_allowed: bool = True,
    diagnostics: list[StaticParserDiagnostic] | None = None,
) -> _ShellCommandWord | None:
    runtime_check = check_runtime or (lambda: None)
    # The shell recognizes ``NAME=value`` before expansion and never runs the
    # value as the command name. Its substitutions keep their own candidate
    # positions, and a later ``$NAME`` command word is still checked, so only
    # this word's printf reconstruction check is skipped.
    assignment = _is_assignment_word(content, start)

    output: list[str] = []
    wrapper_quote = enclosing_delimiter or (
        _command_wrapper_quote(content, start) if content[start] in "$`" else None
    )
    quote: str | None = None
    ansi_c_quote = False
    dynamic = False
    limited = False
    diagnostic: StaticParserDiagnostic | None = None
    unquoted_characters = 0
    cursor = start
    limit = len(content)
    while cursor < limit:
        if cursor > start and (cursor - start) % 4096 == 0:
            runtime_check()
        character = content[cursor]
        if quote is not None:
            if character == quote:
                if owned_word_positions is not None:
                    owned_word_positions.add(cursor)
                quote = None
                ansi_c_quote = False
            elif character == "\\" and ansi_c_quote:
                decoded, cursor = _decode_ansi_c_escape(content, cursor, limit)
                if "\x00" in decoded:
                    closing_quote = _skip_ansi_c_quote_tail(content, cursor, limit)
                    if closing_quote is None:
                        _note_parser_diagnostic(
                            diagnostics,
                            ParserCause.BOUNDARY_UNRESOLVED,
                            ParserCheck.COMMAND_WORD_BOUNDARY,
                            start,
                            limit,
                            ParserUncheckedScope.COMMAND_WORD,
                            full_input_checked=True,
                        )
                        return None
                    cursor = closing_quote
                    continue
                output.append(decoded)
                continue
            elif quote == '"' and character == "$" and cursor + 1 < limit:
                inherited_quote_closed = [False]
                if content[cursor + 1] == "(":
                    substitution_end = _skip_command_substitution(
                        content,
                        cursor,
                        limit,
                        runtime_check,
                        end_cache=substitution_end_cache,
                        parameter_end_cache=parameter_end_cache,
                        backtick_end_cache=backtick_end_cache,
                        diagnostics=diagnostics,
                    )
                    if substitution_end is None:
                        return None
                    static_value, substitution_limited, substitution_diagnostic = (
                        _printf_substitution_result(
                            content,
                            cursor,
                            substitution_end,
                            runtime_check,
                            quoted_result=quoted_printf_allowed,
                            recognize_limit=not limited and not assignment,
                        )
                    )
                    if static_value is None:
                        output.append("$()")
                        dynamic = True
                        limited = limited or substitution_limited
                        diagnostic = diagnostic or substitution_diagnostic
                    else:
                        output.append(static_value)
                    cursor = substitution_end
                    continue
                parameter_end = _skip_parameter_expansion(
                    content,
                    cursor,
                    limit,
                    runtime_check,
                    end_cache=parameter_end_cache,
                    substitution_end_cache=substitution_end_cache,
                    backtick_end_cache=backtick_end_cache,
                    inherited_double_quote=True,
                    inherited_quote_closed=inherited_quote_closed,
                    diagnostics=diagnostics,
                )
                if parameter_end is not None:
                    output.append("$PARAM")
                    dynamic = True
                    cursor = parameter_end
                    if inherited_quote_closed[0]:
                        quote = None
                        ansi_c_quote = False
                    continue
            elif quote == '"' and character == "`":
                substitution_end = _skip_backtick_substitution(
                    content,
                    cursor,
                    limit,
                    runtime_check,
                    end_cache=backtick_end_cache,
                    parameter_end_cache=parameter_end_cache,
                    substitution_end_cache=substitution_end_cache,
                    diagnostics=diagnostics,
                )
                if substitution_end is None:
                    return None
                static_value, substitution_limited, substitution_diagnostic = (
                    _printf_substitution_result(
                        content,
                        cursor,
                        substitution_end,
                        runtime_check,
                        backtick=True,
                        recognize_limit=not limited and not assignment,
                    )
                )
                if static_value is None:
                    output.append("$()")
                    dynamic = True
                    limited = limited or substitution_limited
                    diagnostic = diagnostic or substitution_diagnostic
                else:
                    output.append(static_value)
                cursor = substitution_end
                continue
            elif character == "\\" and quote == '"' and cursor + 1 < limit:
                escaped = content[cursor + 1]
                if escaped == "\n":
                    cursor += 2
                    continue
                if escaped == "\r" and cursor + 2 < limit and content[cursor + 2] == "\n":
                    cursor += 3
                    continue
                if escaped in '$`"\\':
                    output.append(escaped)
                else:
                    output.extend(("\\", escaped))
                cursor += 2
                continue
            else:
                output.append(character)
        elif character == "$" and cursor + 1 < limit and content[cursor + 1] in "'\"":
            quote = content[cursor + 1]
            ansi_c_quote = quote == "'"
            cursor += 2
            continue
        elif character == "$" and cursor + 1 < limit and content[cursor + 1] == "(":
            substitution_end = _skip_command_substitution(
                content,
                cursor,
                limit,
                runtime_check,
                end_cache=substitution_end_cache,
                parameter_end_cache=parameter_end_cache,
                backtick_end_cache=backtick_end_cache,
                diagnostics=diagnostics,
            )
            if substitution_end is None:
                return None
            static_value, substitution_limited, substitution_diagnostic = (
                _printf_substitution_result(
                    content,
                    cursor,
                    substitution_end,
                    runtime_check,
                    recognize_limit=not limited and not assignment,
                )
            )
            if static_value is None:
                output.append("$()")
                dynamic = True
                limited = limited or substitution_limited
                diagnostic = diagnostic or substitution_diagnostic
            else:
                output.append(static_value)
            cursor = substitution_end
            continue
        elif character == "$":
            parameter_end = _skip_parameter_expansion(
                content,
                cursor,
                limit,
                runtime_check,
                end_cache=parameter_end_cache,
                substitution_end_cache=substitution_end_cache,
                backtick_end_cache=backtick_end_cache,
                diagnostics=diagnostics,
            )
            if parameter_end is not None:
                if _is_ifs_expansion(content, cursor, parameter_end):
                    cursor = parameter_end
                    break
                output.append("$PARAM")
                dynamic = True
                cursor = parameter_end
                continue
        elif character == "`":
            if enclosing_delimiter == "`":
                # The enclosing substitution ends here; this is not an opener.
                if owned_word_positions is not None:
                    owned_word_positions.add(cursor)
                break
            substitution_end = _skip_backtick_substitution(
                content,
                cursor,
                limit,
                runtime_check,
                end_cache=backtick_end_cache,
                parameter_end_cache=parameter_end_cache,
                substitution_end_cache=substitution_end_cache,
                diagnostics=diagnostics,
            )
            if substitution_end is None:
                return None
            static_value, substitution_limited, substitution_diagnostic = (
                _printf_substitution_result(
                    content,
                    cursor,
                    substitution_end,
                    runtime_check,
                    backtick=True,
                    recognize_limit=not limited and not assignment,
                )
            )
            if static_value is None:
                output.append("$()")
                dynamic = True
                limited = limited or substitution_limited
                diagnostic = diagnostic or substitution_diagnostic
            else:
                output.append(static_value)
            cursor = substitution_end
            continue
        elif character in "'\"":
            if character == wrapper_quote:
                break
            quote = character
        elif character == "\\" and cursor + 1 < limit:
            if content[cursor + 1] == "\n":
                cursor += 2
                continue
            if content[cursor + 1] == "\r" and cursor + 2 < limit and content[cursor + 2] == "\n":
                cursor += 3
                continue
            cursor += 1
            output.append(content[cursor])
        elif character.isspace() or character in ";|&()<>":
            break
        else:
            output.append(character)
            # Quoted spans have already been consumed in full. Their decoded
            # length must not exhaust the budget for the following literal
            # suffix, which can resolve the candidate as an ordinary word.
            unquoted_characters += 1
            if unquoted_characters > _SHELL_COMMAND_WORD_CHARS:
                return _ShellCommandWord(
                    "".join(output),
                    cursor,
                    dynamic,
                    limited=True,
                    diagnostic=diagnostic
                    or StaticParserDiagnostic(
                        ParserCause.RESOURCE_LIMIT,
                        ParserCheck.COMMAND_WORD_CHARACTERS,
                        start,
                        cursor + 1,
                        ParserUncheckedScope.COMMAND_WORD,
                        {
                            "observed_characters": unquoted_characters,
                            "limit_characters": _SHELL_COMMAND_WORD_CHARS,
                        },
                        observed_is_lower_bound=True,
                    ),
                )
        cursor += 1
    if quote is not None:
        _note_parser_diagnostic(
            diagnostics,
            ParserCause.BOUNDARY_UNRESOLVED,
            ParserCheck.COMMAND_WORD_BOUNDARY,
            start,
            limit,
            ParserUncheckedScope.COMMAND_WORD,
            full_input_checked=True,
        )
        return None
    return _ShellCommandWord("".join(output), cursor, dynamic, limited, diagnostic)


def _plain_quoted_assignment_word(parsed: _ShellCommandWord) -> bool:
    """Return whether a static assignment argument cannot contain shell clauses."""
    return (
        not parsed.dynamic
        and re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*=[^\s;|&()<>]*",
            parsed.text,
        )
        is not None
    )


def _destructive_command_words(
    content: str, *, check_runtime: Callable[[], None] | None = None
) -> Iterator[tuple[int, int]]:
    if check_runtime is not None:
        check_runtime()
    runtime_kwargs = {"check_runtime": check_runtime} if check_runtime is not None else {}
    parsed_through = 0
    parameter_end_cache: dict[int, _ParameterExpansionEnd] = {}
    substitution_end_cache: dict[int, int | None] = {}
    backtick_end_cache: dict[int, int | None] = {}
    for candidate in _SHELL_COMMAND_WORD_START_RE.finditer(content):
        if check_runtime is not None:
            check_runtime()
        start = candidate.start()
        if start < parsed_through:
            continue
        if not _is_shell_command_word_start(content, start):
            continue
        if _has_quoted_assignment_prefix(content, start):
            # This candidate cannot name a destructive command. Claim a static
            # quoted word so a later quote cannot be misread as its opener.
            # Dynamic words remain unclaimed so nested substitutions are still
            # inspected independently by the outer scan.
            parsed = _parse_shell_command_word(
                content,
                start,
                parameter_end_cache,
                substitution_end_cache,
                backtick_end_cache,
                **runtime_kwargs,
            )
            if parsed is not None and _plain_quoted_assignment_word(parsed):
                parsed_through = max(parsed_through, parsed.end)
            continue
        if content.startswith("$(", start):
            substitution_end = _bounded_static_substitution_end(content, start)
            if (
                substitution_end is None
                or _static_printf_substitution(
                    content,
                    start,
                    substitution_end,
                )
                is None
            ):
                # Dynamic substitutions cannot deterministically name a
                # destructive command. Their inner literal commands remain
                # independent candidates; the bounded structural check avoids
                # reparsing nested suffixes quadratically.
                continue
        parsed = _parse_shell_command_word(
            content,
            start,
            parameter_end_cache,
            substitution_end_cache,
            backtick_end_cache,
            **runtime_kwargs,
        )
        if parsed is None:
            continue
        # A long concatenated quoted word can expose every closing quote as a
        # regex candidate. Skip the rest only when it cannot be an executed
        # whitespace-bearing wrapper or a dynamic substitution wrapper.
        candidate_character = content[start]
        if (
            candidate_character in "'\""
            and not parsed.dynamic
            and not any(character.isspace() for character in parsed.text)
        ):
            parsed_through = max(parsed_through, parsed.end)
        command_basename = parsed.text.casefold().rsplit("/", 1)[-1]
        if (
            not parsed.dynamic
            and not parsed.limited
            and command_basename in _DESTRUCTIVE_COMMAND_BASENAMES
        ):
            parsed_through = max(parsed_through, parsed.end)
            yield start, parsed.end


class _ShellOwnershipBudgetExceededError(Exception):
    """Stop optional quote proof without swallowing the caller's cancellation."""


def _literal_shell_command_spans(
    content: str, begin: int, end: int, check_runtime: Callable[[], None]
) -> dict[int, int] | None:
    """Locate exact single-quoted shell -c bodies once per lexical scope."""
    source = content[begin:end]
    if "'" not in source:
        return {}
    spans: list[tuple[int, int]] = []
    # Invocation readers report periodic progress across escaped sequences.
    # Bound repeated delimiter scans proportionally to this scope, so a valid
    # literal command before deeply nested unrelated input cannot make this
    # optional ownership pass quadratic. Exhaustion publishes no proof.
    remaining_checks = 4 + 4 * ((len(source) + 4095) // 4096)

    def check_proof_runtime() -> None:
        nonlocal remaining_checks
        check_runtime()
        remaining_checks -= 1
        if remaining_checks < 0:
            raise _ShellOwnershipBudgetExceededError

    clause_starts = _shell_clause_starts(source, check_runtime)
    next_clause_start = next(clause_starts, None)
    clause_start = 0
    last_selected_start = -1
    try:
        for option in re.finditer(r"(-[A-Za-z]+)\s+(?:\\\r?\n[ \t]*)*'", source):
            if "c" not in option[1]:
                continue
            check_runtime()
            # Select the nearest preceding clause for full validation of its
            # wrapper and literal body; ambiguous context retains no proof.
            # Both iterators move forward once; unrelated clauses need no parse.
            while next_clause_start is not None and next_clause_start <= option.start():
                clause_start = next_clause_start
                next_clause_start = next(clause_starts, None)
            if clause_start == last_selected_start:
                continue
            last_selected_start = clause_start
            _command_string_from_clause(
                source, clause_start, check_proof_runtime, literal_command_spans=spans
            )
    except _ShellOwnershipBudgetExceededError:
        return None
    return {begin + start: begin + stop for start, stop in spans}


def _destructive_command_effects(
    content: str, check_runtime: Callable[[], None] | None = None
) -> Iterator[tuple[int, int]]:
    """Yield commands whose existing operand parser proves destructive effects."""
    for command_start, body_start in _destructive_command_words(
        content, check_runtime=check_runtime
    ):
        if check_runtime is not None:
            check_runtime()
        tokens, _, _ = _bounded_shell_tokens(
            content, command_start, body_start, check_runtime=check_runtime
        )
        if _has_destructive_root_glob(tokens) or _has_destructive_root_path(tokens):
            yield command_start, body_start


def _has_destructive_command_string(
    content: str,
    *,
    diagnostics: list[StaticParserDiagnostic] | None = None,
    check_runtime: Callable[[], None] | None = None,
) -> bool:
    """Return whether an executable string contains a proven destructive command."""
    for command_start, _ in _destructive_command_effects(content, check_runtime):
        _note_parser_diagnostic(
            diagnostics,
            ParserCause.UNSUPPORTED_SYNTAX,
            ParserCheck.COMMAND_STRING_EFFECTS,
            command_start,
            len(content),
            ParserUncheckedScope.COMMAND_STRING,
        )
        return True
    return False


def _shell_quoted_candidate_limits(
    content: str,
    check_runtime: Callable[[], None],
    *,
    quote_closers: set[int] | None = None,
) -> dict[int, int]:
    """Prove literal quote bounds by walking shell source in order.

    This is only an ownership refinement, never an execution or finding parser.
    Comments and single quotes cannot open double-quoted spans. Substitution
    bodies have independent lexical state. Heredoc bodies cannot own surrounding
    quotes. Unsupported delimiters/backticks or
    unresolved syntax discard ALL ownership, retaining the conservative scan.
    """
    limits: dict[int, int] = {}
    closers: set[int] = set()
    parameter_cache: dict[int, _ParameterExpansionEnd] = {}
    substitution_cache: dict[int, int | None] = {}

    def walk(begin: int, end: int, depth: int = 0) -> bool:
        if depth > 32:
            return False
        cursor = begin
        word_started = False
        quote: str | None = None
        quote_start = begin
        ansi_c = False
        candidates: list[int] = []
        pending_heredocs: tuple[tuple[str, bool, bool], ...] = ()
        command_spans = _literal_shell_command_spans(content, begin, end, check_runtime)
        if command_spans is None:
            return False
        while cursor < end:
            if (cursor - begin) % 1024 == 0:
                check_runtime()
            character = content[cursor]
            if character == "\n" and pending_heredocs:
                if quote is not None:
                    return False
                body_end = _skip_static_heredoc_bodies(
                    content, cursor + 1, end, pending_heredocs, check_runtime
                )
                if body_end is None:
                    return False
                pending_heredocs = ()
                word_started = False
                cursor = body_end
                continue
            if quote is not None and character == quote:
                closers.add(cursor)
                if not ansi_c:
                    single_line = (
                        "\n" not in content[quote_start:cursor]
                        and "\r" not in content[quote_start:cursor]
                    )
                    for position in candidates:
                        if single_line or (
                            quote == '"'
                            and content.startswith("$(", position)
                            and substitution_cache.get(position) == cursor
                            and _static_printf_substitution(
                                content, position, cursor, quoted_result=True
                            )
                            is not None
                        ):
                            # Only a proved literal printf at the quote's end
                            # may refine a multiline substitution candidate.
                            limits[position] = cursor
                quote = None
                cursor += 1
                continue
            if quote == "'":
                if character == "\\" and ansi_c:
                    cursor += 2
                    continue
                if not ansi_c and _SHELL_COMMAND_WORD_START_RE.match(content, cursor):
                    candidates.append(cursor)
                cursor += 1
                continue
            if character == "\\":
                # In double quotes only these escapes consume the next byte.
                if quote is None or (cursor + 1 < end and content[cursor + 1] in '$`"\\\n\r'):
                    if quote is None and cursor + 1 < end and content[cursor + 1] not in "\r\n":
                        word_started = True
                    cursor += 2
                    continue
            if quote is None and content.startswith("<<<", cursor):
                # A here-string takes an ordinary shell word; unlike a
                # heredoc it does not change the ownership of later lines.
                cursor += 3
                word_started = False
                continue
            if quote is None and content.startswith("<<", cursor):
                header = _static_heredoc_header(content, cursor, end)
                if header is None or len(pending_heredocs) >= _SHELL_COMMAND_STRING_ARGUMENTS:
                    return False
                spec, cursor = header
                pending_heredocs += (spec,)
                word_started = True
                continue
            if quote is None and character == "(" and pending_heredocs:
                return False
            if character == "`":
                return False
            if quote is None and character == "#" and not word_started:
                while cursor < end and content[cursor] not in "\r\n":
                    cursor += 1
                continue
            if (
                quote is None
                and character == "$"
                and content[cursor + 1 : cursor + 2] in {"'", '"'}
            ):
                quote = content[cursor + 1]
                ansi_c = quote == "'"
                quote_start = cursor + 1
                candidates = []
                word_started = True
                cursor += 2
                continue
            if quote is None and character in "'\"":
                command_end = command_spans.get(cursor + 1) if character == "'" else None
                if command_end is not None:
                    if not walk(cursor + 1, command_end, depth + 1):
                        return False
                    closers.add(command_end)
                    cursor = command_end + 1
                    word_started = True
                    continue
                quote = character
                ansi_c = False
                quote_start = cursor
                candidates = []
                word_started = True
                cursor += 1
                continue
            if quote == '"' and _SHELL_COMMAND_WORD_START_RE.match(content, cursor):
                candidates.append(cursor)
            if content.startswith("$(", cursor) or (
                quote is None and content[cursor : cursor + 2] in {"<(", ">("}
            ):
                substitution_end = _skip_command_substitution(
                    content, cursor, end, check_runtime, end_cache=substitution_cache
                )
                if substitution_end is None or not walk(
                    cursor + 2, substitution_end - 1, depth + 1
                ):
                    return False
                cursor = substitution_end
                word_started = True
                continue
            if content.startswith("${", cursor):
                if quote == '"':
                    # A quoted simple variable used as a trim pattern owns its
                    # inner quotes. Do not apply this to the ksh-sensitive $"
                    # cases handled by the general parameter parser below.
                    pattern = re.match(
                        r"\$\{[A-Za-z_][A-Za-z0-9_]*(?:#{1,2}|%{1,2})"
                        r'"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[A-Za-z_][A-Za-z0-9_]*\})"'
                        r"""[^\\'"$`{}\r\n]*\}""",
                        content[cursor : min(end, cursor + _SHELL_COMMAND_WORD_CHARS + 1)],
                    )
                    if pattern is not None and len(pattern[0]) <= _SHELL_COMMAND_WORD_CHARS:
                        pattern_end = cursor + len(pattern[0])
                        closers.add(content.rfind('"', cursor, pattern_end))
                        cursor = pattern_end
                        word_started = True
                        continue
                inherited_closed = [False]
                parameter_end = _skip_parameter_expansion(
                    content,
                    cursor,
                    end,
                    check_runtime,
                    end_cache=parameter_cache,
                    inherited_double_quote=quote == '"',
                    inherited_quote_closed=inherited_closed,
                )
                if parameter_end is None or inherited_closed[0]:
                    return False
                # Keep parameter bodies opaque to ownership. Nested commands
                # remain independent candidates in the original scanner.
                body = content[cursor:parameter_end]
                if "`" in body or "<<" in body:
                    return False
                cursor = parameter_end
                word_started = True
                continue
            if quote is None:
                word_started = not (character.isspace() or character in ";|&()<>")
            cursor += 1
        return quote is None and not pending_heredocs and cursor <= end

    if not walk(0, len(content)):
        return {}
    if quote_closers is not None:
        quote_closers.update(
            position
            for position in closers
            if content[position + 1 : position + 2] not in ("$", "`")
        )
    return limits


def _quoted_candidate_is_complete(
    content: str,
    start: int,
    quote_limits: dict[int, int] | None,
    check_runtime: Callable[[], None],
    quoted_printf_allowed: bool = True,
) -> bool:
    """Refine a failure only inside a lexically proven shell quote.

    Do not change the original parser's frontier or its quote ownership. A
    shorter diagnostic parse must never hide candidates that the original
    scan would inspect (including after a comment's mis-paired quote).
    """
    end = quote_limits.get(start) if quote_limits is not None else None
    if end is None:
        return False
    if (
        quoted_printf_allowed
        and content.startswith("$(", start)
        and _skip_command_substitution(content, start, end, check_runtime) == end
        and _static_printf_substitution(content, start, end, quoted_result=True) is not None
    ):
        return True
    fragment = content[start:end]
    parsed = _parse_shell_command_word(
        fragment, 0, check_runtime=check_runtime, quoted_printf_allowed=quoted_printf_allowed
    )
    if parsed is None or parsed.limited:
        return False
    return not (
        parsed.dynamic
        and _has_runtime_command_operand_exhaustion(
            fragment, 0, parsed.end, check_runtime, quoted_printf_allowed=quoted_printf_allowed
        )
    )


def _shell_comment_start_before(content: str, start: int, *, ansi_c: bool = False) -> int | None:
    """Return a genuine shell comment ``#`` on the candidate's line, if any.

    This scans from the start of ``content`` so quote state is inherited
    across lines: a ``#`` on the candidate's line that sits inside a string
    opened on an earlier line (``echo "prefix\\n # literal"; ...``) is not a
    comment start, and the exemption is declined. A ``#`` that opens a
    comment on an earlier line ends at that line's newline, so quotes inside
    comment text never leak into later lines. When the initial quote state
    cannot be proven from the source handed to the analyzer, the scan still
    starts unquoted at the content boundary, which is the most the analyzer
    can prove; a genuinely unprovable fragment stays conservative because
    the caller only reaches this helper after the line-local comment idiom
    already matched on the candidate's own line.

    With ``ansi_c`` (shell only), a ``$'...'`` string follows the same
    ownership rule as the main word parser: a backslash escapes the next
    character, so ``\\'`` inside the string does not close it and a ``#``
    after it is still string content, not a comment start. Other
    hash-comment languages do not share Bash ANSI-C quoting (``$'`` is a
    variable in Perl and Ruby), so they keep the plain single-quote rule
    and never get the ANSI-C treatment.

    A ``${...}`` parameter expansion is skipped whole via the existing
    bounded parser, so a ``#`` inside it (for example the default value in
    ``${NOTE:- # literal}``) never reads as a comment start. When the
    expansion cannot be proven (the parser returns no end), the exemption
    is declined rather than guessed.
    """
    line_start = content.rfind("\n", 0, start) + 1
    stack: list[str | None] = [None]
    i = 0
    in_comment = False
    while i < start:
        ch = content[i]
        if in_comment:
            if ch == "\n":
                in_comment = False
            i += 1
            continue
        top = stack[-1]
        if top == "ansi":
            if ch == "\\":
                i += 2
                continue
            if ch == "'":
                stack.pop()
            i += 1
            continue
        if top == "'":
            if ch == "'":
                stack.pop()
            i += 1
            continue
        if top == '"':
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                stack.pop()
            elif ch == "`":
                stack.append("`")
            elif ch == "$" and content[i + 1 : i + 2] == "(":
                stack.append("sub")
                i += 1
            i += 1
            continue
        # Command context: the base text, a $( ) substitution, or backticks.
        if ch == "\\":
            i += 2
            continue
        if ch == "$" and content[i + 1 : i + 2] == "{":
            parameter_end = _skip_parameter_expansion(content, i, len(content), None)
            if parameter_end is None:
                return None
            i = parameter_end
            continue
        if ch == "$" and ansi_c and content[i + 1 : i + 2] == "'":
            stack.append("ansi")
            i += 1
        elif ch in ("'", '"'):
            stack.append(ch)
        elif ch == "`":
            if top == "`":
                stack.pop()
            else:
                stack.append("`")
        elif ch == "$" and content[i + 1 : i + 2] == "(":
            stack.append("sub")
            i += 1
        elif ch == ")" and top == "sub":
            stack.pop()
        elif ch == "#":
            if i == 0 or content[i - 1] in " \t\r\n;&|()<>":
                if i >= line_start:
                    return i
                in_comment = True
        i += 1
    return None


def _is_comment_candidate(
    content: str,
    start: int,
    python_source: _PythonSourceOwnership | None,
    *,
    file_type: str | None = None,
) -> bool:
    """Return whether a candidate position sits inside a ``#`` comment.

    The shell comment-start idiom is checked first so host Python source only
    pays for proven token ownership when a ``#`` actually precedes the
    candidate on its line. A ``#`` inside a string literal never counts: for
    host source the comment token must be proven, and without host source
    the candidate's line is scanned with quote state inherited from the
    start of the source, so a ``#`` inside an already closed string, or
    inside a string opened on an earlier line, cannot exempt a later command.
    """
    line_start = content.rfind("\n", 0, start) + 1
    if _SHELL_COMMENT_START_RE.search(content, line_start, start) is None:
        return False
    if python_source is None:
        return _shell_comment_start_before(content, start, ansi_c=file_type == "shell") is not None
    return python_source.comment(start) is not None


# File types where ``#`` opens a comment. Markdown and plain text are
# excluded: there ``#`` starts a heading or is prose, and the conservative
# partial result for fragmented headings is deliberate.
_HASH_COMMENT_FILE_TYPES = frozenset({"shell", "python", "perl", "ruby", "powershell"})


def _has_shell_command_word_exhaustion(
    content: str,
    check_runtime: Callable[[], None],
    *,
    structural_quote_closers: set[int] | None = None,
    structural_quote_openers: set[int] | None = None,
    python_source: _PythonSourceOwnership | None = None,
    javascript_source: _JavaScriptTemplateOwnership | None = None,
    powershell: PowerShellSource | None = None,
    perl_eval_blocks: bool = False,
    quote_limits: dict[int, int] | None = None,
    file_type: str = "shell",
    _command_string_depth: int = 0,
    _quoted_printf_allowed: bool = True,
    _startup_exposure: list[bool] | None = None,
    _startup_diagnostics: list[StaticParserDiagnostic] | None = None,
    _xargs_contexts: tuple[tuple[str, list[bool | None]], ...] = (),
    complete_context: bool = True,
) -> StaticParserDiagnostic | None:
    """Find candidate command words whose deterministic parse hit a safety bound.

    ``python_source`` supplies proven Python string and comment ownership, so
    host code is never charged to a shell word or command string.
    ``perl_eval_blocks`` recognizes Perl's exception-trapping ``eval BLOCK``,
    which reparses no string.
    """
    quoted_printf_allowed = _quoted_printf_allowed and not _shell_redefines_command(
        content, check_runtime
    )
    parsed_through = 0
    # A word that spans lines may pair a heredoc, comment or prose quote with
    # a later command line. Main skips the candidates inside such a word, so
    # they are still checked here, but they never change what later
    # candidates are skipped or owned.
    shadow_through = 0
    # Completed words own closing quotes, never executable expansion starts.
    # Inner commands remain independent candidates; never suppress their bodies.
    owned_word_positions: set[int] = set()
    parameter_end_cache: dict[int, _ParameterExpansionEnd] = {}
    substitution_end_cache: dict[int, int | None] = {}
    backtick_end_cache: dict[int, int | None] = {}
    powershell_strings = _powershell_string_operands(powershell)
    powershell_here_closers = (
        {
            string.end - 2
            for string in powershell.strings
            if string.invoked and content[string.start : string.start + 2] in {"@'", '@"'}
        }
        if powershell is not None
        else set()
    )
    may_have_destructive_outer_operands = _may_have_destructive_outer_operands(content) or any(
        string.decoded for string in powershell_strings.values()
    )
    json_openers = sorted(structural_quote_openers or ())
    complete_comments: set[tuple[int, int]] = set()
    word_diagnostics: list[StaticParserDiagnostic] = []
    # Short text without shell-expansion openers cannot trigger these shell-word
    # reconstruction or span limits. Keep its prior lazy ownership paths instead
    # of parsing an ordinary Python module for every candidate letter.
    python_word_source = (
        python_source
        if python_source is not None
        and (len(content) > _SHELL_COMMAND_WORD_CHARS or "$" in content or "`" in content)
        else None
    )
    for candidate in _SHELL_COMMAND_WORD_START_RE.finditer(content):
        check_runtime()
        start = candidate.start()
        if python_word_source is not None and python_word_source.is_string_delimiter(start):
            continue
        if start in owned_word_positions:
            continue
        if javascript_source is not None and javascript_source.owns(start):
            continue
        if powershell is not None and powershell.owns(start):
            continue
        if start in powershell_here_closers:
            # This quote closes a proven host literal. Its body remains visible,
            # and native invocation checks retain the decoded value and operands.
            continue
        if powershell is not None and file_type == "powershell" and content[start] == "$":
            line_start = content.rfind("\n", max(0, start - 512), start) + 1
            if (
                (line_start > 0 or start <= 512)
                and not content[line_start:start].strip(" \t")
                and re.match(
                    r"\$(?:[A-Za-z_][A-Za-z0-9_]*:)?[A-Za-z_][A-Za-z0-9_]*[ \t]*=(?!=)",
                    content[start : start + 512],
                )
            ):
                # A native assignment's left side is not a runtime command.
                # Skip only that candidate; executable subexpressions in its
                # value remain independently visited through lexical holes.
                continue
        if structural_quote_closers is not None and start in structural_quote_closers:
            continue
        json_string_start = structural_quote_openers is not None and (
            start in structural_quote_openers or start - 1 in structural_quote_openers
        )
        assignment_quote = start > 0 and content[start - 1] == "=" and content[start] in "'\""
        if start < parsed_through or (
            not json_string_start
            and not assignment_quote
            and not _is_shell_command_word_start(content, start)
            and not (
                python_word_source is not None and python_word_source.starts_string_body(start)
            )
        ):
            continue
        shadowed = start < shadow_through
        if _has_quoted_assignment_prefix(content, start):
            # A quoted NAME=value argument cannot be the command word. Claim
            # only a complete static word; dynamic substitutions must remain
            # visible as independent executable candidates.
            assignment_positions: set[int] = set()
            parsed = _parse_shell_command_word(
                content,
                start,
                parameter_end_cache,
                substitution_end_cache,
                backtick_end_cache,
                check_runtime=check_runtime,
                owned_word_positions=assignment_positions,
            )
            if (
                parsed is not None
                and not parsed.dynamic
                and not parsed.limited
                and not shadowed
                and "\n" not in content[start : parsed.end]
                and "\r" not in content[start : parsed.end]
            ):
                # Spaces or punctuation inside a quoted value do not turn
                # its closing quote into a new opener. Own only delimiters:
                # embedded command text remains independently inspected.
                owned_word_positions.update(assignment_positions)
            if parsed is not None and _plain_quoted_assignment_word(parsed) and not shadowed:
                parsed_through = max(parsed_through, parsed.end)
            continue
        # A runtime-selected command may follow any prefix word (``sudo -u
        # root``, ``then``, ``{``, ``env A=1``), so ``$``- and backtick-led
        # candidates keep main's word-start check; other words must open a
        # command position.
        if (
            shadowed
            and not _opens_runtime_word(content, start)
            and not _is_shell_command_position(content, start)
        ):
            continue
        candidate_word_positions: set[int] = set()
        word_diagnostics.clear()
        parsed = _parse_shell_command_word(
            content,
            start,
            parameter_end_cache,
            substitution_end_cache,
            backtick_end_cache,
            check_runtime=check_runtime,
            owned_word_positions=candidate_word_positions,
            enclosing_delimiter=(
                _assignment_value_wrapper(content, start) if assignment_quote else None
            ),
            quoted_printf_allowed=quoted_printf_allowed,
            diagnostics=word_diagnostics,
        )
        if parsed is None:
            substitution_start = (
                start + 1
                if content[start : start + 1] in {"'", '"'} and content.startswith("$(", start + 1)
                else start
            )
            if content.startswith("$(", substitution_start):
                substitution_end = _skip_command_substitution(
                    content,
                    substitution_start,
                    len(content),
                    check_runtime,
                    end_cache=substitution_end_cache,
                    parameter_end_cache=parameter_end_cache,
                    backtick_end_cache=backtick_end_cache,
                )
                if substitution_end is not None and _is_printf_substitution(
                    content,
                    substitution_start,
                    substitution_end,
                    check_runtime=check_runtime,
                ):
                    if not _quoted_candidate_is_complete(
                        content, start, quote_limits, check_runtime, quoted_printf_allowed
                    ):
                        return (
                            word_diagnostics[0]
                            if word_diagnostics
                            else StaticParserDiagnostic(
                                ParserCause.BOUNDARY_UNRESOLVED,
                                ParserCheck.COMMAND_WORD_BOUNDARY,
                                start,
                                len(content),
                                ParserUncheckedScope.COMMAND_WORD,
                                full_input_checked=complete_context,
                            )
                        )
                    continue
            if content[start] == "`":
                substitution_end = _skip_backtick_substitution(
                    content, start, len(content), check_runtime, end_cache=backtick_end_cache
                )
                if (
                    substitution_end is not None
                    and _is_printf_substitution(
                        content,
                        start,
                        substitution_end,
                        backtick=True,
                        check_runtime=check_runtime,
                    )
                    and not (
                        file_type in _HASH_COMMENT_FILE_TYPES
                        and _is_comment_candidate(
                            content, start, python_source, file_type=file_type
                        )
                    )
                ):
                    return (
                        word_diagnostics[0]
                        if word_diagnostics
                        else StaticParserDiagnostic(
                            ParserCause.BOUNDARY_UNRESOLVED,
                            ParserCheck.COMMAND_WORD_BOUNDARY,
                            start,
                            len(content),
                            ParserUncheckedScope.COMMAND_WORD,
                            full_input_checked=complete_context,
                        )
                    )
            if shadowed:
                # The spanning word already closed this region's quotes.
                continue
            # An unclosed expansion or quote can consume the rest of the
            # artifact. Once that unresolved span exceeds the command-word
            # budget, treating it as clean would turn malformed, deeply nested
            # runtime selection into a fail-open result.
            # A validated JSON value owns its bytes and is checked separately.
            # Do not charge that value to an earlier unmatched Markdown tick.
            next_json = bisect_right(json_openers, start)
            unresolved_end = (
                json_openers[next_json] if next_json < len(json_openers) else len(content)
            )
            if powershell is not None:
                unresolved_end = min(unresolved_end, powershell.next_boundary(start, len(content)))
            if unresolved_end - start > _SHELL_COMMAND_WORD_CHARS and python_source is not None:
                # Likewise, a host string or comment token owns its bytes: a
                # shell quote it opens cannot continue into later host code.
                unresolved_end = min(unresolved_end, python_source.word_end(start))
            elif (
                unresolved_end - start > _SHELL_COMMAND_WORD_CHARS and javascript_source is not None
            ):
                # Template literal delimiters and their host-language
                # expressions cannot continue a shell word into later code.
                unresolved_end = min(unresolved_end, javascript_source.word_end(start))
            if unresolved_end - start > _SHELL_COMMAND_WORD_CHARS:
                if not _quoted_candidate_is_complete(
                    content, start, quote_limits, check_runtime, quoted_printf_allowed
                ):
                    return (
                        word_diagnostics[0]
                        if word_diagnostics
                        else StaticParserDiagnostic(
                            ParserCause.BOUNDARY_UNRESOLVED,
                            ParserCheck.COMMAND_WORD_BOUNDARY,
                            start,
                            unresolved_end,
                            ParserUncheckedScope.COMMAND_WORD,
                            {
                                "observed_characters": unresolved_end - start,
                                "limit_characters": _SHELL_COMMAND_WORD_CHARS,
                            },
                            observed_is_lower_bound=True,
                            full_input_checked=complete_context,
                        )
                    )
            continue
        # A mis-paired quote can produce a completed word that crosses several
        # Python tokens. It cannot own quotes or skip candidates beyond its
        # proven host region, even when that lexical parse itself succeeded.
        ownership_boundary = min(
            python_word_source.word_end(start) if python_word_source is not None else len(content),
            powershell.next_boundary(start, len(content))
            if powershell is not None
            else len(content),
        )
        owned_end = min(parsed.end, ownership_boundary)
        # Comments end at a newline, and a quote after ``=`` in prose or host
        # source may be mis-paired. Such a word cannot own a later line.
        confined_word = "\n" in content[start : parsed.end] and (
            assignment_quote
            or _SHELL_COMMENT_START_RE.search(content, content.rfind("\n", 0, start) + 1, start)
            is not None
        )
        if not confined_word and not shadowed:
            # A quote that opens a runtime-selected word stays an independent
            # candidate, so a mis-paired claim cannot hide its operands.
            # An enclosing backtick may be owned at parsed.end itself.
            owned_word_positions.update(
                position
                for position in candidate_word_positions
                if position <= parsed.end
                and position < ownership_boundary
                and content[position + 1 : position + 2] not in ("$", "`")
            )
        # Only executable nested substitutions retain independent command
        # positions. A plain dynamic data argument still owns its inner bytes;
        # revisiting those as commands would turn quoted printf data into code.
        raw_word = content[start : parsed.end]
        simple_backtick_parameter = (
            re.fullmatch(
                r"`\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[A-Za-z_][A-Za-z0-9_]*\})`",
                raw_word,
            )
            is not None
        )
        if (
            (
                not parsed.dynamic
                or "$" not in raw_word
                or not any(marker in raw_word for marker in ("$(", "`"))
                or simple_backtick_parameter
            )
            and not (assignment_quote and confined_word)
            and not shadowed
        ):
            if python_source is None and ("\n" in raw_word or "\r" in raw_word):
                shadow_through = max(shadow_through, owned_end)
            else:
                parsed_through = max(parsed_through, owned_end)
        if assignment_quote:
            # An assignment value never names the command, and main never
            # parsed this position, so it only claims ownership.
            continue
        if parsed.limited:
            if file_type in _HASH_COMMENT_FILE_TYPES and _is_comment_candidate(
                content, start, python_source, file_type=file_type
            ):
                # A comment is never executed: a backtick there is literal
                # text, so it cannot open a command substitution that
                # reconstructs a printf invocation (issue #901). Unresolved
                # spans and runtime-selected operands keep their existing
                # conservative treatment.
                continue
            if not _quoted_candidate_is_complete(
                content, start, quote_limits, check_runtime, quoted_printf_allowed
            ):
                assert parsed.diagnostic is not None
                return parsed.diagnostic
        operand_diagnostic = (
            _has_runtime_command_operand_exhaustion(
                content,
                start,
                parsed.end,
                check_runtime,
                powershell_strings=powershell_strings,
                quoted_printf_allowed=quoted_printf_allowed,
            )
            if parsed.dynamic and may_have_destructive_outer_operands
            else None
        )
        if operand_diagnostic is not None:
            comment = python_source.comment(start) if python_source is not None else None
            if comment is None:
                if _quoted_candidate_is_complete(
                    content, start, quote_limits, check_runtime, quoted_printf_allowed
                ):
                    continue
                # Host code after a Python string's closing quote is not shell
                # text, so a command in the string takes operands only from
                # the rest of the string's value, including literals
                # concatenated with it. A value that code can still extend,
                # for example with ``%`` or ``.format``, stays partial.
                value_source_range: list[int] = []
                value = (
                    python_source.string_value(start, parsed.end, source_range=value_source_range)
                    if python_source is not None
                    else None
                )
                if value is None:
                    return operand_diagnostic
                value_diagnostic = _has_runtime_command_operand_exhaustion(
                    *value, check_runtime, quoted_printf_allowed=quoted_printf_allowed
                )
                if value_diagnostic is not None:
                    return value_diagnostic.enclosed(*value_source_range)
                continue
            # A Python comment is prose that no shell receives, so a command
            # named in it can take operands only from the same comment, never
            # from later host code. Recheck that comment as standalone shell
            # text.
            if comment not in complete_comments:
                diagnostic = _has_shell_command_word_exhaustion(
                    content[comment[0] : comment[1]],
                    check_runtime,
                    _command_string_depth=_command_string_depth,
                    _quoted_printf_allowed=quoted_printf_allowed,
                )
                if diagnostic is not None:
                    return diagnostic.shifted(comment[0])
                complete_comments.add(comment)
    for command in _shell_command_strings(
        content,
        check_runtime,
        perl_eval_blocks=perl_eval_blocks,
        python_source=python_source,
        powershell=powershell,
        shell_substitutions=file_type == "shell",
        startup_exposure=_startup_exposure,
        startup_diagnostics=_startup_diagnostics,
        xargs_contexts=_xargs_contexts,
    ):
        command_string = command.content
        if command_string is None:
            assert command.diagnostic is not None
            return command.diagnostic
        if _command_string_depth >= 8:
            return StaticParserDiagnostic(
                ParserCause.RESOURCE_LIMIT,
                ParserCheck.COMMAND_STRING_DEPTH,
                command.start,
                command.end,
                ParserUncheckedScope.COMMAND_STRING,
                {"observed_depth": _command_string_depth, "limit_depth": 8},
                observed_is_lower_bound=True,
            )
        command_effects: list[StaticParserDiagnostic] = []
        if _has_destructive_command_string(command_string, diagnostics=command_effects):
            diagnostic = command_effects[0]
            return (
                diagnostic.shifted(command.start)
                if command.exact_source
                else diagnostic.enclosed(command.start, command.end)
            )
        command_closers: set[int] = set()
        command_limits = _shell_quoted_candidate_limits(
            command_string, check_runtime, quote_closers=command_closers
        )
        nested_startup_diagnostics = [
            item.enclosed(0, len(command_string)) for item in command.startup_diagnostics[:1]
        ]
        inherited_startup_diagnostic = (
            nested_startup_diagnostics[0] if nested_startup_diagnostics else None
        )
        diagnostic = _has_shell_command_word_exhaustion(
            command_string,
            check_runtime,
            structural_quote_closers=command_closers,
            quote_limits=command_limits,
            _command_string_depth=_command_string_depth + 1,
            _quoted_printf_allowed=quoted_printf_allowed,
            _startup_exposure=command.startup_context,
            _startup_diagnostics=nested_startup_diagnostics,
            _xargs_contexts=command.xargs_contexts,
        )
        if nested_startup_diagnostics and not command.startup_diagnostics:
            # eval shares the exposure flag with its parent; retain its cause
            # too when no command in this body needed that proof yet. A child
            # shell's context has separate carriers and cannot affect siblings.
            command.startup_diagnostics.append(
                nested_startup_diagnostics[0].enclosed(command.start, command.end)
            )
        if diagnostic is not None:
            if diagnostic is inherited_startup_diagnostic:
                # This cause already belongs to the parent's original carrier.
                # Returning the same object at each level preserves that origin
                # through later eval consumers, including nested ones.
                return command.startup_diagnostics[0]
            return (
                diagnostic.shifted(command.start)
                if command.exact_source
                else diagnostic.enclosed(command.start, command.end)
            )
    return None


def _powershell_string_operands(
    source: PowerShellSource | None,
) -> dict[int, PowerShellString]:
    """Native string arguments retain their boundaries and known literal text.

    Embedded executable subexpressions split ownership and are excluded here.
    Ordinary variable interpolation stays opaque, just as in shell arguments;
    it never makes PowerShell backticks into Bash command substitution.
    """
    if source is None:
        return {}
    return {
        string.start: string
        for string in source.strings
        if not string.invoked
        and source.owns(string.start)
        and source.next_boundary(string.start, string.end) == string.end
    }


def _has_runtime_command_operand_exhaustion(
    content: str,
    start: int,
    word_end: int,
    check_runtime: Callable[[], None],
    *,
    powershell_strings: Mapping[int, PowerShellString] | None = None,
    quoted_printf_allowed: bool = True,
) -> StaticParserDiagnostic | None:
    """Return whether a runtime-selected command's operands are undecidable."""
    diagnostics: list[StaticParserDiagnostic] = []
    tokens, command_end, exhausted = _bounded_shell_tokens(
        content,
        start,
        word_end,
        check_runtime=check_runtime,
        powershell_strings=powershell_strings,
        quoted_printf_allowed=quoted_printf_allowed,
        diagnostics=diagnostics,
    )
    if exhausted and command_end - start < _ROOT_GLOB_COMMAND_CHARS:
        # Earlier prose (for example "row-first") cannot supply this
        # command's operands. Refine only this candidate's exhausted
        # span, including nested commands inside the executable word.
        # Hitting the span bound always remains partial, including when
        # real operands occur beyond that bound. This rescan is bounded
        # by the same constant as the tokenizer.
        check_runtime()
        exhausted = _may_have_destructive_outer_operands(content[start:command_end])
    if exhausted:
        return diagnostics[0]
    brace_diagnostic = _brace_expansion_diagnostic(tokens, start, command_end)
    if brace_diagnostic is not None:
        return brace_diagnostic
    if _has_destructive_root_glob(tokens) or _has_destructive_root_path(tokens):
        return StaticParserDiagnostic(
            ParserCause.RUNTIME_DEPENDENT,
            ParserCheck.COMMAND_IDENTITY,
            start,
            max(start + 1, command_end),
            ParserUncheckedScope.COMMAND_ARGUMENTS,
        )
    return None


def _shell_command_strings(
    content: str,
    check_runtime: Callable[[], None],
    *,
    perl_eval_blocks: bool = False,
    python_source: _PythonSourceOwnership | None = None,
    powershell: PowerShellSource | None = None,
    shell_substitutions: bool = True,
    startup_exposure: list[bool] | None = None,
    startup_diagnostics: list[StaticParserDiagnostic] | None = None,
    xargs_contexts: tuple[tuple[str, list[bool | None]], ...] = (),
) -> Iterator[_ShellCommandString]:
    """Yield bounded strings reparsed by ``eval`` or a shell ``-c`` wrapper."""
    nested_substitutions = startup_exposure is not None and ("$(" in content or "`" in content)
    if startup_exposure is None:
        startup_exposure = [False]
    if startup_diagnostics is None:
        startup_diagnostics = []
    substitution_scopes: list[int] = []
    backtick_starts: set[int] = set()
    scoped_startup: list[tuple[int, list[bool], list[StaticParserDiagnostic], int, str]] = []
    startup_work_limit = 4 * len(content) + _ROOT_GLOB_COMMAND_CHARS
    startup_work = [startup_work_limit]
    xargs_override_cache: list[bool | None] = [None]
    clause_starts = (
        powershell.clause_starts
        if powershell is not None
        else _nonempty_shell_clause_starts(
            content,
            check_runtime,
            shell_substitutions=shell_substitutions,
            substitution_scopes=substitution_scopes,
            backtick_starts=backtick_starts,
        )
    )
    for clause_start in clause_starts:
        check_runtime()
        # Substitutions inherit settings but cannot alter their parent shell.
        # Keep each live lexical scope's mutable context so eval still updates
        # subsequent commands in that same scope.
        while scoped_startup and (
            len(scoped_startup) > len(substitution_scopes)
            or scoped_startup[-1][0] != substitution_scopes[len(scoped_startup) - 1]
        ):
            scoped_startup.pop()
        for scope in substitution_scopes[len(scoped_startup) :]:
            inherited = scoped_startup[-1][1] if scoped_startup else startup_exposure
            inherited_diagnostics = scoped_startup[-1][2] if scoped_startup else startup_diagnostics
            # Legacy closers are not shell word delimiters. Cache this body's
            # bounded source once, so readers cannot consume its closing tick.
            body_start = clause_start if content[scope - 1] == "`" else 0
            body = content[body_start : scope - 1] if body_start else content
            scoped_startup.append(
                (scope, [inherited[0]], list(inherited_diagnostics), body_start, body)
            )
        startup_context = scoped_startup[-1][1] if scoped_startup else startup_exposure
        context_diagnostics = scoped_startup[-1][2] if scoped_startup else startup_diagnostics
        if clause_start in backtick_starts:
            # A legacy substitution may launch a child with startup code.
            # Only its opener enters this lexical marker set.
            if startup_context[0]:
                yield _ShellCommandString(
                    None,
                    startup_context,
                    context_diagnostics,
                    clause_start,
                    len(content),
                    False,
                    context_diagnostics[0]
                    if context_diagnostics
                    else StaticParserDiagnostic(
                        ParserCause.RUNTIME_DEPENDENT,
                        ParserCheck.COMMAND_STRING_RUNTIME,
                        clause_start,
                        len(content),
                        ParserUncheckedScope.REMAINING_INPUT,
                    ),
                )
                return
            continue
        body_start, body = scoped_startup[-1][3:5] if scoped_startup else (0, content)
        if not startup_context[0]:
            local_diagnostics: list[StaticParserDiagnostic] = []
            startup_context[0] = _shell_clause_sets_startup(
                body,
                clause_start - body_start,
                check_runtime,
                startup_work,
                local_diagnostics,
                work_limit=startup_work_limit,
            )
            if local_diagnostics and not context_diagnostics:
                context_diagnostics.append(local_diagnostics[0].shifted(body_start))
        if body_start and startup_context[0] and "`" in body:
            # Escaped nested legacy substitutions need another decoding pass;
            # this scope walk cannot prove their child startup dependencies.
            yield _ShellCommandString(
                None,
                startup_context,
                context_diagnostics,
                clause_start,
                body_start + len(body),
                False,
                context_diagnostics[0]
                if context_diagnostics
                else StaticParserDiagnostic(
                    ParserCause.RUNTIME_DEPENDENT,
                    ParserCheck.COMMAND_STRING_RUNTIME,
                    clause_start,
                    body_start + len(body),
                    ParserUncheckedScope.REMAINING_INPUT,
                ),
            )
            return
        starts_shell = [False]
        diagnostics: list[StaticParserDiagnostic] = []
        source_span = [clause_start - body_start, clause_start - body_start]
        literal_spans: list[tuple[int, int]] = []
        recognized, command_string = _command_string_from_clause(
            body,
            clause_start - body_start,
            check_runtime,
            perl_eval_blocks=perl_eval_blocks,
            shell_traps=shell_substitutions,
            startup_exposure=startup_context,
            startup_diagnostics=context_diagnostics,
            starts_shell=starts_shell,
            diagnostics=diagnostics,
            source_span=source_span,
            literal_command_spans=literal_spans,
            xargs_override_cache=xargs_override_cache,
            xargs_source=content,
            xargs_parent_contexts=xargs_contexts,
        )
        if nested_substitutions and startup_exposure[0]:
            # Quoted substitutions can launch shells outside this clause walk.
            yield _ShellCommandString(
                None,
                startup_exposure,
                startup_diagnostics,
                clause_start,
                len(content),
                False,
                startup_diagnostics[0]
                if startup_diagnostics
                else StaticParserDiagnostic(
                    ParserCause.RUNTIME_DEPENDENT,
                    ParserCheck.COMMAND_STRING_RUNTIME,
                    clause_start,
                    len(content),
                    ParserUncheckedScope.REMAINING_INPUT,
                ),
            )
            return
        if not recognized:
            continue
        if python_source is not None:
            word_start = clause_start
            while word_start < len(content) and content[word_start].isspace():
                word_start += 1
            if python_source.is_code(word_start):
                # A clause that begins in Python code is Python, not shell:
                # ``signal.alarm(timeout)`` names no ``timeout`` command.
                # Shell text can only be reparsed from a string literal.
                continue
        # eval shares this shell's settings; a child shell has its own scope.
        # Keep exposure monotonic across eval, including settings in its body.
        start, end = literal_spans[-1] if literal_spans else source_span
        diagnostic = diagnostics[0] if diagnostics else None
        if diagnostic is not None and not any(diagnostic is item for item in context_diagnostics):
            diagnostic = diagnostic.shifted(body_start)
        yield _ShellCommandString(
            command_string,
            [False] if starts_shell[0] else startup_context,
            [] if starts_shell[0] else context_diagnostics,
            start + body_start,
            max(start + 1, end) + body_start,
            bool(literal_spans),
            diagnostic,
            xargs_contexts + ((content, xargs_override_cache),),
        )


def _shell_clause_sets_startup(
    content: str,
    start: int,
    check_runtime: Callable[[], None],
    remaining_work: list[int],
    diagnostics: list[StaticParserDiagnostic] | None = None,
    *,
    work_limit: int | None = None,
    variable_names: tuple[str, ...] = ("BASH_ENV", "ENV"),
    allow_empty: bool = True,
) -> bool:
    """Find explicit startup settings in bounded command-prefix syntax."""
    source = content[start : start + _ROOT_GLOB_COMMAND_CHARS + 1]
    cursor = 0
    mode: str | None = None
    setter_mode: str | None = None
    skip_operand = False
    read_options = True
    read_array = False
    printf_options = True
    setting_namerefs = False
    word_diagnostics: list[StaticParserDiagnostic] = []
    variable_pattern = (
        "BASH_ENV|ENV"
        if variable_names == ("BASH_ENV", "ENV")
        else "|".join(re.escape(name) for name in variable_names)
    )

    def uncertain(
        cause: ParserCause = ParserCause.RUNTIME_DEPENDENT,
        check: ParserCheck = ParserCheck.COMMAND_STRING_RUNTIME,
        measurements: ResourceMeasurements | None = None,
        *,
        lower_bound: bool = False,
    ) -> bool:
        _note_parser_diagnostic(
            diagnostics,
            cause,
            check,
            start,
            start + cursor,
            ParserUncheckedScope.REMAINING_INPUT,
            measurements=measurements,
            lower_bound=lower_bound,
        )
        return True

    def startup_target(word: str) -> bool:
        # A setter can modify an already exported variable even without a
        # preceding export in this source. Dynamic destinations remain unknown.
        return bool(re.fullmatch(rf"(?:{variable_pattern})(?:\[.*\])?", word, re.DOTALL)) or any(
            marker in word
            for marker in (_DYNAMIC_SHELL_WORD_SENTINEL, _RUNTIME_SHELL_PARAMETER_SENTINEL)
        )

    for _ in range(_SHELL_COMMAND_STRING_ARGUMENTS):
        check_runtime()
        while cursor < len(source):
            if source.startswith("\\\r\n", cursor):
                cursor += 3
            elif source.startswith("\\\n", cursor):
                cursor += 2
            elif source[cursor] in " \t":
                cursor += 1
            else:
                break
        if cursor >= len(source) or source[cursor] in "\r\n;|&(){}#":
            if cursor > _ROOT_GLOB_COMMAND_CHARS:
                return uncertain(
                    ParserCause.RESOURCE_LIMIT,
                    ParserCheck.COMMAND_ARGUMENT_CHARACTERS,
                    {"observed_characters": cursor, "limit_characters": _ROOT_GLOB_COMMAND_CHARS},
                    lower_bound=True,
                )
            return (
                uncertain(ParserCause.BOUNDARY_UNRESOLVED, ParserCheck.COMMAND_WORD_BOUNDARY)
                if skip_operand
                else False
            )
        word_start = cursor
        word, cursor, limited = _next_shell_invocation_word(
            source, cursor, check_runtime, diagnostics=word_diagnostics
        )
        remaining_work[0] -= cursor - word_start
        if remaining_work[0] < 0 or limited or cursor > _ROOT_GLOB_COMMAND_CHARS:
            # Proof work is linear in the containing source, even when clause
            # starts overlap malformed quoted words or nested expansions.
            if diagnostics is not None and not diagnostics and word_diagnostics:
                diagnostics.append(
                    replace(word_diagnostics[0].shifted(start), full_input_checked=False)
                )
            if remaining_work[0] < 0:
                limit = (
                    work_limit
                    if work_limit is not None
                    else 4 * len(content) + _ROOT_GLOB_COMMAND_CHARS
                )
                return uncertain(
                    ParserCause.RESOURCE_LIMIT,
                    ParserCheck.STARTUP_PROOF_CHARACTERS,
                    {"observed_characters": limit - remaining_work[0], "limit_characters": limit},
                    lower_bound=True,
                )
            if limited:
                return uncertain(ParserCause.BOUNDARY_UNRESOLVED, ParserCheck.COMMAND_WORD_BOUNDARY)
            return uncertain(
                ParserCause.RESOURCE_LIMIT,
                ParserCheck.COMMAND_ARGUMENT_CHARACTERS,
                {"observed_characters": cursor, "limit_characters": _ROOT_GLOB_COMMAND_CHARS},
                lower_bound=True,
            )
        if word is None:
            return False
        if skip_operand:
            skip_operand = False
            continue
        if mode == "setter-target":
            if startup_target(word):
                return uncertain()
            mode = setter_mode
            continue
        if mode == "printf":
            if any(
                marker in word
                for marker in (_DYNAMIC_SHELL_WORD_SENTINEL, _RUNTIME_SHELL_PARAMETER_SENTINEL)
            ):
                return uncertain()
            if printf_options and word == "--":
                printf_options = False
                continue
            if printf_options and word == "-v":
                setter_mode = "printf"
                mode = "setter-target"
                continue
            if printf_options and word.startswith("-v"):
                if startup_target(word[2:]):
                    return uncertain()
                continue
            # Bash's %n writes to a variable named by its argument, even without
            # -v. Escaped percent pairs cannot introduce a conversion.
            return (
                uncertain()
                if re.search(r"%[-+ #0'.*0-9]*[hjlLtz]*n", word.replace("%%", "")) is not None
                else False
            )
        if mode == "read":
            if read_options and word == "--":
                read_options = False
                continue
            if read_options and word.startswith("-") and word != "-":
                for index, option in enumerate(word[1:], 2):
                    if option in "Eers":
                        continue
                    if option not in "adinNptu":
                        # Unsupported option arity cannot prove the target.
                        return uncertain(ParserCause.UNSUPPORTED_SYNTAX)
                    attached = word[index:]
                    if option == "a":
                        read_array = True
                        if attached:
                            if startup_target(attached):
                                return uncertain()
                        else:
                            setter_mode = "read"
                            mode = "setter-target"
                    else:
                        skip_operand = not attached
                    break
                continue
            read_options = False
            if not read_array and startup_target(word):
                return uncertain()
            continue
        assignment = re.fullmatch(rf"({variable_pattern})(\+?=)(.*)", word, re.DOTALL)
        if assignment is not None and (
            mode in {"env", "setting"} or _SHELL_ASSIGNMENT_PREFIX_RE.match(source, word_start)
        ):
            if assignment[2] == "+=" or assignment[3] or not allow_empty:
                return uncertain()
            continue
        if mode == "setting":
            destination = word.partition("=")[0].removesuffix("+")
            if startup_target(destination):
                return uncertain()  # Exporting an unknown inherited value is unresolved.
            if setting_namerefs and re.fullmatch(r"-[A-Za-z]*n[A-Za-z]*", word):
                # A name reference can redirect a later assignment to a startup
                # variable without spelling that variable at the write site.
                return uncertain()
            if word.startswith(("-", "+")) or re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*(?:=.*)?", word, re.DOTALL
            ):
                continue
            return False
        if mode == "env":
            if re.match(r"[A-Za-z_][A-Za-z0-9_]*=", word) is not None:
                continue
            if word in {"-i", "--ignore-environment", "--"} or re.fullmatch(
                r"(?:--unset|--chdir)=.+", word
            ):
                continue
            if word in {"-u", "--unset", "-C", "--chdir"}:
                skip_operand = True
                continue
            if word.startswith("-"):
                return uncertain(ParserCause.UNSUPPORTED_SYNTAX)
            mode = None
        elif mode in {"command", "builtin"}:
            if word == "--" or (mode == "command" and word == "-p"):
                continue
            if word.startswith("-"):
                return False
            mode = None
        command = word.casefold().rsplit("/", 1)[-1]
        if command in {"export", "declare", "typeset", "readonly", "local"}:
            mode = "setting"
            setting_namerefs = command in {"declare", "typeset", "local"}
        elif command in {"env", "command", "builtin", "read", "printf"}:
            mode = command
        elif command in _SHELL_CLAUSE_PREFIX_WORDS or _SHELL_ASSIGNMENT_PREFIX_RE.match(
            source, word_start
        ):
            continue
        else:
            return False
    return uncertain(
        ParserCause.RESOURCE_LIMIT,
        ParserCheck.WRAPPER_ARGUMENTS,
        {
            "observed_records": _SHELL_COMMAND_STRING_ARGUMENTS,
            "limit_records": _SHELL_COMMAND_STRING_ARGUMENTS,
        },
        lower_bound=True,
    )


def _shell_redefines_command(
    content: str,
    check_runtime: Callable[[], None],
    command: str = "printf",
    *,
    shell_substitutions: bool = False,
) -> bool:
    """Do not widen builtin evaluation across an explicit local replacement."""
    work = 4 * len(content) + _ROOT_GLOB_COMMAND_CHARS
    setting_work = [work]
    scope_ends: list[int] = []
    aliases: set[str] = set()
    invoked: set[str] = set()
    backtick_starts: set[int] = set()
    for start in _nonempty_shell_clause_starts(
        content,
        check_runtime,
        shell_substitutions=shell_substitutions,
        substitution_scopes=scope_ends,
        backtick_starts=backtick_starts if shell_substitutions else None,
    ):
        check_runtime()
        if start in backtick_starts:
            # The opener is a traversal marker; its body is visited separately.
            continue
        scope_end = scope_ends[-1] - 1 if scope_ends else len(content)
        if start >= scope_end:
            continue
        source = content[start : min(scope_end, start + _ROOT_GLOB_COMMAND_CHARS + 1)]
        if command == "xargs" and _shell_clause_sets_startup(
            source,
            0,
            check_runtime,
            setting_work,
            variable_names=("PATH",),
            allow_empty=False,
        ):
            return True
        cursor = 0
        wrapper: str | None = None
        for _ in range(8):
            word_start = cursor
            while word_start < len(source):
                if source.startswith("\\\r\n", word_start):
                    word_start += 3
                elif source.startswith("\\\n", word_start):
                    word_start += 2
                elif source[word_start] in " \t":
                    word_start += 1
                else:
                    break
            word, end, limited = _next_shell_invocation_word(source, cursor, check_runtime)
            work -= end - cursor
            cursor = end
            if work < 0 or limited or cursor > _ROOT_GLOB_COMMAND_CHARS:
                return True
            if command == "xargs" and re.fullmatch(r"PATH(?:\+)?=.*", word or "", re.DOTALL):
                return True
            if _SHELL_ASSIGNMENT_PREFIX_RE.match(source, word_start):
                continue
            if command == "xargs" and word is not None:
                invoked.add(word)
            if wrapper is not None:
                if word == "--" or (wrapper == "command" and word == "-p"):
                    continue
                if word is None or word.startswith("-"):
                    break
                wrapper = None
            if word in {"command", "builtin"}:
                wrapper = word
                continue
            if word in _SHELL_CLAUSE_PREFIX_WORDS or word in {"if", "while", "until"}:
                continue
            if command == "xargs":
                if any(
                    marker in (word or "")
                    for marker in (_DYNAMIC_SHELL_WORD_SENTINEL, _RUNTIME_SHELL_PARAMETER_SENTINEL)
                ):
                    return True
                if word == "hash":
                    return True
            if word in {"eval", "source", ".", "enable", "trap"}:
                # Imported/evaluated code, handlers and builtin configuration
                # can replace the command in this shell.
                # Retain the earlier narrow decoder across those unknown effects.
                return True
            if word == "function":
                name, end, limited = _next_shell_invocation_word(source, cursor, check_runtime)
                work -= end - cursor
                if limited or work < 0 or end > _ROOT_GLOB_COMMAND_CHARS or name == command:
                    return True
                break
            if word == "alias":
                for _ in range(_SHELL_COMMAND_STRING_ARGUMENTS):
                    while cursor < len(source) and source[cursor] in " \t":
                        cursor += 1
                    if cursor >= len(source) or source[cursor] in "\r\n;|&()#":
                        break
                    alias, end, limited = _next_shell_invocation_word(source, cursor, check_runtime)
                    work -= end - cursor
                    cursor = end
                    if limited or work < 0 or cursor > _ROOT_GLOB_COMMAND_CHARS:
                        return True
                    if alias is None:
                        break
                    if command == "xargs" and "=" in alias:
                        aliases.add(alias.partition("=")[0])
                    if alias.startswith(command + "=") or any(
                        marker in alias
                        for marker in (
                            _DYNAMIC_SHELL_WORD_SENTINEL,
                            _RUNTIME_SHELL_PARAMETER_SENTINEL,
                        )
                    ):
                        return True
                else:
                    return True
                break
            if word != command:
                break
            # Bash accepts line continuations around the empty parameter list.
            for expected in "()":
                while cursor < len(source):
                    if source.startswith("\\\r\n", cursor):
                        cursor += 3
                    elif source.startswith("\\\n", cursor):
                        cursor += 2
                    elif source[cursor] in " \t":
                        cursor += 1
                    else:
                        break
                if cursor > _ROOT_GLOB_COMMAND_CHARS:
                    return True
                if cursor == len(source) or source[cursor] != expected:
                    break
                cursor += 1
            else:
                return True
            break
        else:
            return True
    # A used alias can turn an ordinary name into eval/source or another alias.
    return bool(aliases & invoked)


def _nonempty_shell_clause_starts(
    content: str,
    check_runtime: Callable[[], None],
    *,
    shell_substitutions: bool = False,
    substitution_scopes: list[int] | None = None,
    backtick_starts: set[int] | None = None,
) -> Iterator[int]:
    """Avoid reparsing the same trailing command across empty physical lines."""
    for start in _shell_clause_starts(
        content,
        check_runtime,
        shell_substitutions=shell_substitutions,
        substitution_scopes=substitution_scopes,
        backtick_starts=backtick_starts,
    ):
        while start < len(content) and content[start] in " \t":
            if start % 4096 == 0:
                check_runtime()
            start += 1
        if start < len(content) and content[start] not in "\r\n;|&(){}#":
            yield start


def _shell_clause_starts(
    content: str,
    check_runtime: Callable[[], None],
    *,
    shell_substitutions: bool = False,
    substitution_scopes: list[int] | None = None,
    backtick_starts: set[int] | None = None,
) -> Iterator[int]:
    """Yield command-clause starts outside quotes and comments in one pass."""
    yield 0
    quote: str | None = None
    ansi_c_quote = False
    word_started = False
    cursor = 0
    limit = len(content)
    frames: list[tuple[int, int, str | None, bool, bool]] = []
    substitution_ends: dict[int, int | None] = {}
    backtick_ends: dict[int, int | None] = {}
    while cursor < limit or frames:
        if cursor >= limit:
            cursor, limit, quote, ansi_c_quote, word_started = frames.pop()
            if substitution_scopes is not None:
                substitution_scopes.pop()
            continue
        if cursor % 4096 == 0:
            check_runtime()
        character = content[cursor]
        if shell_substitutions and quote != "'" and content.startswith("$(", cursor):
            end = _skip_command_substitution(
                content, cursor, limit, check_runtime, end_cache=substitution_ends
            )
            if end is not None:
                # A command substitution executes even inside double quotes.
                # Give its body an independent quote state and then resume
                # the parent word; cached endpoints keep nested walks linear.
                frames.append((end, limit, quote, ansi_c_quote, True))
                if substitution_scopes is not None:
                    substitution_scopes.append(end)
                cursor += 2
                limit = end - 1
                quote = None
                ansi_c_quote = False
                word_started = False
                yield cursor
                continue
        if (
            shell_substitutions
            and backtick_starts is not None
            and quote != "'"
            and character == "`"
        ):
            end = _skip_backtick_substitution(
                content, cursor, limit, check_runtime, end_cache=backtick_ends
            )
            if end is not None:
                backtick_starts.add(cursor)
                yield cursor
                frames.append((end, limit, quote, ansi_c_quote, True))
                if substitution_scopes is not None:
                    substitution_scopes.append(end)
                cursor += 1
                limit = end - 1
                quote = None
                ansi_c_quote = False
                word_started = False
                if cursor < limit:
                    yield cursor
                continue
        if quote is not None:
            if character == quote:
                quote = None
                ansi_c_quote = False
            elif (quote == '"' or ansi_c_quote) and character == "\\" and cursor + 1 < limit:
                cursor += 2
                continue
        elif (
            shell_substitutions
            and character == "$"
            and content[cursor + 1 : cursor + 2] in {"'", '"'}
        ):
            quote = content[cursor + 1]
            ansi_c_quote = quote == "'"
            word_started = True
            cursor += 2
            continue
        elif character in "'\"":
            quote = character
            word_started = True
        elif character == "\\" and cursor + 1 < limit:
            cursor += 2
            word_started = True
            continue
        elif character == "#" and not word_started:
            newline = content.find("\n", cursor + 1, limit)
            if newline < 0:
                cursor = limit
                continue
            cursor = newline + 1
            word_started = False
            yield cursor
            continue
        elif character in ";|&(){}\n":
            word_started = False
            yield cursor + 1
        elif character.isspace():
            word_started = False
        else:
            word_started = True
        cursor += 1


def _eval_command_string(
    content: str,
    start: int,
    check_runtime: Callable[[], None],
    *,
    diagnostics: list[StaticParserDiagnostic] | None = None,
    source_span: list[int] | None = None,
) -> str | None:
    """Join bounded eval operands, excluding outer redirections and clauses."""
    # Include one lookahead character so reaching the limit cannot masquerade
    # as a complete final word. No individual word can scan beyond this slice.
    source = content[start : start + _ROOT_GLOB_COMMAND_CHARS + 1]
    cursor = 0
    operands: list[str] = []
    word_diagnostics: list[StaticParserDiagnostic] = []

    def failed(
        cause: ParserCause,
        check: ParserCheck,
        measurements: ResourceMeasurements | None = None,
        *,
        lower_bound: bool = False,
    ) -> None:
        if source_span is not None:
            source_span[1] = start + cursor
        _note_parser_diagnostic(
            diagnostics,
            cause,
            check,
            start,
            start + cursor,
            ParserUncheckedScope.COMMAND_STRING,
            measurements=measurements,
            lower_bound=lower_bound,
        )

    for count in range(_SHELL_COMMAND_STRING_ARGUMENTS + 1):
        check_runtime()
        while cursor < len(source):
            if source.startswith("\\\r\n", cursor):
                cursor += 3
            elif source.startswith("\\\n", cursor):
                cursor += 2
            elif source[cursor].isspace() and source[cursor] not in "\r\n":
                cursor += 1
            else:
                break
        if cursor > _ROOT_GLOB_COMMAND_CHARS:
            failed(
                ParserCause.RESOURCE_LIMIT,
                ParserCheck.COMMAND_ARGUMENT_CHARACTERS,
                {"observed_characters": cursor, "limit_characters": _ROOT_GLOB_COMMAND_CHARS},
                lower_bound=True,
            )
            return None
        redirection = _SHELL_REDIRECTION_PREFIX_RE.match(source, cursor)
        if redirection is None and (cursor == len(source) or source[cursor] in "\r\n;|&()#"):
            if source_span is not None:
                source_span[1] = start + cursor
            return " ".join(operands)
        if count == _SHELL_COMMAND_STRING_ARGUMENTS:
            failed(
                ParserCause.RESOURCE_LIMIT,
                ParserCheck.WRAPPER_ARGUMENTS,
                {"observed_records": count, "limit_records": _SHELL_COMMAND_STRING_ARGUMENTS},
                lower_bound=True,
            )
            return None
        if redirection is not None:
            # Here-documents/strings require a different grammar. Keep their
            # coverage partial instead of treating their delimiter as code.
            if "<<" in redirection.group():
                failed(ParserCause.UNSUPPORTED_SYNTAX, ParserCheck.COMMAND_STRING_RUNTIME)
                return None
            cursor = redirection.end()
            while cursor < len(source) and source[cursor] in " \t":
                cursor += 1
            if cursor == len(source) or source[cursor] in "\r\n;|&()#":
                failed(ParserCause.BOUNDARY_UNRESOLVED, ParserCheck.COMMAND_ARGUMENT_BOUNDARY)
                return None
        word, cursor, limited = _next_shell_invocation_word(
            source, cursor, check_runtime, diagnostics=word_diagnostics
        )
        if limited or word is None or cursor > _ROOT_GLOB_COMMAND_CHARS:
            if diagnostics is not None and word_diagnostics:
                diagnostic = word_diagnostics[0].shifted(start)
                if len(source) < len(content) - start:
                    diagnostic = replace(diagnostic, full_input_checked=False)
                diagnostics.append(diagnostic)
            if cursor > _ROOT_GLOB_COMMAND_CHARS:
                failed(
                    ParserCause.RESOURCE_LIMIT,
                    ParserCheck.COMMAND_ARGUMENT_CHARACTERS,
                    {"observed_characters": cursor, "limit_characters": _ROOT_GLOB_COMMAND_CHARS},
                    lower_bound=True,
                )
            else:
                failed(ParserCause.BOUNDARY_UNRESOLVED, ParserCheck.COMMAND_WORD_BOUNDARY)
            return None
        if redirection is not None:
            continue
        if any(
            marker in word
            for marker in (_DYNAMIC_SHELL_WORD_SENTINEL, _RUNTIME_SHELL_PARAMETER_SENTINEL)
        ):
            failed(ParserCause.RUNTIME_DEPENDENT, ParserCheck.COMMAND_STRING_RUNTIME)
            return None
        if not operands and word == "--":
            continue
        operands.append(word)
    failed(ParserCause.UNSUPPORTED_SYNTAX, ParserCheck.COMMAND_STRING_RUNTIME)
    return None


def _xargs_has_source_override(content: str, check_runtime: Callable[[], None]) -> bool:
    """Require the external utility before granting its redirected-data proof."""
    return _shell_redefines_command(content, check_runtime, "xargs", shell_substitutions=True)


def _command_string_from_clause(
    content: str,
    start: int,
    check_runtime: Callable[[], None],
    *,
    perl_eval_blocks: bool = False,
    shell_traps: bool = False,
    literal_command_spans: list[tuple[int, int]] | None = None,
    startup_exposure: list[bool] | None = None,
    startup_diagnostics: list[StaticParserDiagnostic] | None = None,
    starts_shell: list[bool] | None = None,
    diagnostics: list[StaticParserDiagnostic] | None = None,
    source_span: list[int] | None = None,
    xargs_override_cache: list[bool | None] | None = None,
    xargs_source: str | None = None,
    xargs_parent_contexts: tuple[tuple[str, list[bool | None]], ...] = (),
) -> tuple[bool, str | None]:
    """Resolve a bounded wrapper chain to an ``eval`` or shell command string."""
    cursor = start
    pending: str | None = None
    wrapper_seen = False
    last_word_start = start
    word_start = [start]
    xargs_replacements: set[str] = set()
    if xargs_override_cache is None:
        xargs_override_cache = [None]

    def external_xargs_proved() -> bool:
        if startup_exposure is not None and startup_exposure[0]:
            return False
        # Defer ancestor checks until xargs needs the printing exemption.
        for source, cached in xargs_parent_contexts:
            if cached[0] is None:
                cached[0] = _xargs_has_source_override(source, check_runtime)
            if cached[0]:
                return False
        if xargs_override_cache[0] is None:
            xargs_override_cache[0] = _xargs_has_source_override(
                content if xargs_source is None else xargs_source, check_runtime
            )
        return not xargs_override_cache[0]

    def failed(
        cause: ParserCause = ParserCause.UNSUPPORTED_SYNTAX,
        check: ParserCheck = ParserCheck.COMMAND_STRING_RUNTIME,
        measurements: ResourceMeasurements | None = None,
        *,
        lower_bound: bool = False,
    ) -> tuple[bool, None]:
        _note_parser_diagnostic(
            diagnostics,
            cause,
            check,
            start,
            cursor,
            ParserUncheckedScope.COMMAND_STRING,
            measurements=measurements,
            lower_bound=lower_bound,
        )
        return True, None

    def observe_assignment(word: str) -> None:
        if startup_exposure is not None and re.fullmatch(r"(?:BASH_ENV|ENV)=.+", word, re.DOTALL):
            startup_exposure[0] = True

    def next_word() -> tuple[str | None, bool]:
        nonlocal cursor, last_word_start
        word, cursor, limited = _next_shell_invocation_word(
            content, cursor, check_runtime, diagnostics=diagnostics, source_start=word_start
        )
        if word is not None and any(marker in word for marker in xargs_replacements):
            word = _DYNAMIC_SHELL_WORD_SENTINEL
        last_word_start = word_start[0]
        if source_span is not None:
            source_span[1] = cursor
        return word, limited

    def horizontal_gap() -> None:
        nonlocal cursor
        next_runtime_check = cursor
        while cursor < len(content):
            if cursor >= next_runtime_check:
                check_runtime()
                next_runtime_check = cursor + 4096
            if content[cursor] in " \t":
                cursor += 1
            elif content.startswith("\\\r\n", cursor):
                cursor += 3
            elif content.startswith("\\\n", cursor):
                cursor += 2
            else:
                break

    def later_here_string() -> bool | None:
        # Input modes need their own child-input analysis; do not extend the
        # redirected-data exemption to a redirection after the utility either.
        tail = content[cursor : cursor + _ROOT_GLOB_COMMAND_CHARS + 1]
        position = 0
        for _ in range(_SHELL_COMMAND_STRING_ARGUMENTS):
            check_runtime()
            while position < len(tail):
                if position % 4096 == 0:
                    check_runtime()
                if tail[position] in " \t":
                    position += 1
                elif tail.startswith("\\\r\n", position):
                    position += 3
                elif tail.startswith("\\\n", position):
                    position += 2
                else:
                    break
            if position > _ROOT_GLOB_COMMAND_CHARS:
                return None
            if position == len(tail):
                return False
            if re.match(r"(?:[0-9]+)?<<<", tail[position : position + 32]):
                return True
            redirection = _SHELL_REDIRECTION_PREFIX_RE.match(tail, position)
            if redirection is not None and not tail.startswith(("<(", ">("), position):
                position = redirection.end()
                continue
            if tail[position] in "\r\n;|&()#":
                return False
            word, end, limited = _next_shell_invocation_word(
                tail, position, check_runtime, leading_process_substitution=True
            )
            if limited or word is None or end <= position or end > _ROOT_GLOB_COMMAND_CHARS:
                return None
            position = end
        return None

    def resolved_command_string(word: str | None, limited: bool) -> str | None:
        if (
            limited
            or word is None
            or any(
                marker in word
                for marker in (_DYNAMIC_SHELL_WORD_SENTINEL, _RUNTIME_SHELL_PARAMETER_SENTINEL)
            )
        ):
            failed(
                ParserCause.RUNTIME_DEPENDENT
                if word is not None and not limited
                else ParserCause.BOUNDARY_UNRESOLVED
            )
            return None
        return word

    def commandless_wrapper() -> tuple[bool, None]:
        # env/command/builtin can end without invoking another command. A
        # redirection or opening parenthesis may still hide a following one.
        ended = (
            cursor >= len(content)
            or content[cursor] == "#"
            or _ends_wrapped_clause(content, cursor)
        )
        return (
            (False, None)
            if ended
            else failed(ParserCause.BOUNDARY_UNRESOLVED, ParserCheck.COMMAND_WORD_BOUNDARY)
        )

    for _ in range(32):
        while cursor < len(content) and content[cursor].isspace():
            if cursor % 4096 == 0:
                check_runtime()
            cursor += 1
        redirection = _SHELL_REDIRECTION_PREFIX_RE.match(content, cursor)
        if redirection is not None:
            cursor = redirection.end()
            target, limited = next_word()
            if limited or target is None:
                return (
                    failed(ParserCause.BOUNDARY_UNRESOLVED, ParserCheck.COMMAND_WORD_BOUNDARY)
                    if wrapper_seen
                    else (False, None)
                )
            continue

        word, limited = (pending, False) if pending is not None else next_word()
        pending = None
        if limited:
            return failed() if wrapper_seen else (False, None)
        if word is None:
            return False, None
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", word, re.DOTALL) is not None:
            observe_assignment(word)
            continue
        if xargs_replacements and _DYNAMIC_SHELL_WORD_SENTINEL in word:
            return failed(ParserCause.RUNTIME_DEPENDENT)
        command = word.casefold().rsplit("/", 1)[-1]
        if (
            not wrapper_seen
            and content[last_word_start:cursor] == word
            and _SHELL_FUNCTION_BODY_RE.match(content[cursor : cursor + 256])
        ):
            # A function declaration names no invocation; its body is scanned
            # independently when clause traversal reaches the opening brace.
            return False, None
        if command in _SHELL_CLAUSE_PREFIX_WORDS:
            continue
        if shell_traps and command == "trap":
            body, limited = next_word()
            if body == "--" and not limited:
                body, limited = next_word()
            if limited:
                return True, None
            # Reset/ignore/query forms install no executable command string.
            if body is None or body in {"", "-", "-p", "-l"}:
                return False, None
            while cursor < len(content) and content[cursor] in " \t":
                cursor += 1
            if cursor == len(content) or content[cursor] in "\r\n;|&()#":
                return False, None
            return True, resolved_command_string(body, False)
        if command == "eval":
            if (
                perl_eval_blocks
                and word == "eval"
                and _PERL_EVAL_BLOCK_RE.match(content, cursor) is not None
            ):
                # Perl ``eval BLOCK`` is exception handling, not a string
                # reparse. ``{`` starts a new clause, so the statements in the
                # block are still scanned as ordinary command candidates.
                return False, None
            return True, _eval_command_string(
                content, cursor, check_runtime, diagnostics=diagnostics, source_span=source_span
            )
        if command in _SHELL_COMMAND_STRING_SHELLS:
            for _ in range(16):
                option, limited = next_word()
                if limited:
                    return failed()
                if (
                    xargs_replacements
                    and option is not None
                    and any(
                        marker in option
                        for marker in (
                            _DYNAMIC_SHELL_WORD_SENTINEL,
                            _RUNTIME_SHELL_PARAMETER_SENTINEL,
                        )
                    )
                ):
                    return failed(ParserCause.RUNTIME_DEPENDENT)
                if option is None:
                    # Without -c there is no command string to reconstruct.
                    # This also keeps Markdown fence labels such as sh inert.
                    return False, None
                if option.startswith("-") and not option.startswith("--"):
                    if "c" in option[1:]:
                        if starts_shell is not None:
                            starts_shell[0] = True
                        if startup_exposure is not None and startup_exposure[0]:
                            # A startup file can replace the child's builtins.
                            if diagnostics is not None and not diagnostics and startup_diagnostics:
                                diagnostics.append(startup_diagnostics[0])
                            return failed(ParserCause.RUNTIME_DEPENDENT)
                        command_string, limited = next_word()
                        resolved = resolved_command_string(command_string, limited)
                        if (
                            literal_command_spans is not None
                            and resolved is not None
                            and content[last_word_start : last_word_start + 1] == "'"
                            and content[cursor - 1 : cursor] == "'"
                            and cursor - last_word_start == len(resolved) + 2
                            and content[last_word_start + 1 : cursor - 1] == resolved
                        ):
                            # One literal has an exact source-to-body mapping.
                            # Concatenation, expansion and ANSI-C escapes cannot
                            # lend their decoded offsets to the original text.
                            literal_command_spans.append((last_word_start + 1, cursor - 1))
                        return True, resolved
                    continue
                if option.startswith("+") or option.startswith("--"):
                    continue
                return False, None
            return failed(
                ParserCause.RESOURCE_LIMIT,
                ParserCheck.WRAPPER_ARGUMENTS,
                {"observed_records": 16, "limit_records": 16},
                lower_bound=True,
            )

        if command == "env":
            wrapper_seen = True
            for _ in range(_SHELL_COMMAND_STRING_ARGUMENTS):
                option, limited = next_word()
                if limited:
                    return failed()
                if option is None:
                    return commandless_wrapper()
                if option == "--":
                    pending, limited = next_word()
                    if limited:
                        return failed()
                    if pending is None:
                        return commandless_wrapper()
                    break
                if re.match(r"[A-Za-z_][A-Za-z0-9_]*=", option) is not None:
                    observe_assignment(option)
                    continue
                if option in {"-i", "--ignore-environment"} or re.fullmatch(
                    r"(?:--unset|--chdir)=.+", option
                ):
                    continue
                if option in {"-u", "--unset", "-C", "--chdir"}:
                    operand, limited = next_word()
                    if limited or operand is None:
                        return failed(
                            ParserCause.BOUNDARY_UNRESOLVED, ParserCheck.COMMAND_WORD_BOUNDARY
                        )
                    continue
                if option.startswith("-"):
                    return failed()
                pending = option
                break
            else:
                return failed(
                    ParserCause.RESOURCE_LIMIT,
                    ParserCheck.WRAPPER_ARGUMENTS,
                    {
                        "observed_records": _SHELL_COMMAND_STRING_ARGUMENTS,
                        "limit_records": _SHELL_COMMAND_STRING_ARGUMENTS,
                    },
                    lower_bound=True,
                )
            continue
        if command in {"command", "builtin"}:
            wrapper_seen = True
            for _ in range(_SHELL_COMMAND_STRING_ARGUMENTS):
                option, limited = next_word()
                if limited:
                    return failed()
                if option is None:
                    return commandless_wrapper()
                if option == "--":
                    pending, limited = next_word()
                    if limited:
                        return failed()
                    if pending is None:
                        return commandless_wrapper()
                    break
                if command == "command" and option == "-p":
                    continue
                if option.startswith("-"):
                    return False, None
                pending = option
                break
            else:
                return failed(
                    ParserCause.RESOURCE_LIMIT,
                    ParserCheck.WRAPPER_ARGUMENTS,
                    {
                        "observed_records": _SHELL_COMMAND_STRING_ARGUMENTS,
                        "limit_records": _SHELL_COMMAND_STRING_ARGUMENTS,
                    },
                    lower_bound=True,
                )
            continue
        if command == "nohup":
            wrapper_seen = True
            continue
        if command in {"sudo", "nice", "xargs"}:
            wrapper_seen = True
            xargs_options = True
            xargs_redirected = False
            xargs_input_mode = False
            for _ in range(_SHELL_COMMAND_STRING_ARGUMENTS):
                if command == "xargs":
                    horizontal_gap()
                    here_string = re.match(r"(?:[0-9]+)?<<<", content[cursor : cursor + 32])
                    if here_string is not None:
                        cursor += here_string.end()
                        horizontal_gap()
                        if cursor < len(content) and content[cursor] in "\r\n":
                            return failed(
                                ParserCause.BOUNDARY_UNRESOLVED, ParserCheck.COMMAND_WORD_BOUNDARY
                            )
                        operand, limited = next_word()
                        if limited or operand is None:
                            return failed(
                                ParserCause.BOUNDARY_UNRESOLVED, ParserCheck.COMMAND_WORD_BOUNDARY
                            )
                        if not external_xargs_proved():
                            return failed(ParserCause.RUNTIME_DEPENDENT)
                        if xargs_input_mode:
                            return failed()
                        xargs_redirected = True
                        continue
                    if xargs_redirected and cursor < len(content) and content[cursor] in "\r\n":
                        if not external_xargs_proved():
                            return failed(ParserCause.RUNTIME_DEPENDENT)
                        # Without an explicit utility xargs prints its input.
                        # A following physical command is a separate clause.
                        return False, None
                option, limited = next_word()
                if limited:
                    return failed()
                if option is None:
                    if command == "xargs":
                        if re.match(r"(?:[0-9]+)?<<<", content[cursor : cursor + 32]):
                            continue
                        if xargs_redirected and content[cursor : cursor + 1] == "|":
                            # A pipeline may load the printed input as code.
                            return failed(ParserCause.RUNTIME_DEPENDENT)
                        if xargs_redirected and not external_xargs_proved():
                            return failed(ParserCause.RUNTIME_DEPENDENT)
                    return (
                        (False, None)
                        if _ends_wrapped_clause(content, cursor)
                        else failed(
                            ParserCause.BOUNDARY_UNRESOLVED, ParserCheck.COMMAND_WORD_BOUNDARY
                        )
                    )
                if option == "--" and (command != "xargs" or xargs_options):
                    if command == "xargs" and xargs_options:
                        xargs_options = False
                        continue
                    pending, limited = next_word()
                    if limited:
                        return failed()
                    break
                if command == "xargs" and any(
                    marker in option
                    for marker in (_DYNAMIC_SHELL_WORD_SENTINEL, _RUNTIME_SHELL_PARAMETER_SENTINEL)
                ):
                    return failed(ParserCause.RUNTIME_DEPENDENT)
                if not option.startswith("-") or command == "xargs" and not xargs_options:
                    pending = option
                    break
                if command == "xargs" and (
                    option in {"-I", "-J", "-i", "--replace"}
                    or option.startswith(("-I", "-J", "--replace=", "-i"))
                ):
                    if option in {"-I", "-J"}:
                        marker, limited = next_word()
                    elif option in {"-i", "--replace"}:
                        marker, limited = "{}", False
                    else:
                        marker, limited = (
                            option.partition("=")[2] if option.startswith("--") else option[2:],
                            False,
                        )
                    if (
                        not marker
                        or limited
                        or any(
                            sentinel in marker
                            for sentinel in (
                                _DYNAMIC_SHELL_WORD_SENTINEL,
                                _RUNTIME_SHELL_PARAMETER_SENTINEL,
                            )
                        )
                    ):
                        return failed(ParserCause.RUNTIME_DEPENDENT)
                    xargs_replacements.add(marker)
                    continue
                if command == "xargs" and (
                    option in {"-a", "--arg-file", "--open-tty"}
                    or option.startswith(("-a", "--arg-file="))
                    or re.fullmatch(r"-[0rtpox]*o[0rtpox]*", option)
                ):
                    if not xargs_input_mode:
                        xargs_input_mode = True
                        if xargs_redirected or later_here_string() is not False:
                            return failed()
                if command == "xargs" and option in {
                    "-a",
                    "--arg-file",
                    "-d",
                    "--delimiter",
                    "-E",
                    "-J",
                    "--max-args",
                    "--max-procs",
                    "--max-chars",
                    "-n",
                    "-L",
                    "-P",
                    "-s",
                    "-R",
                    "-S",
                }:
                    operand, limited = next_word()
                    if limited or operand is None:
                        return failed(
                            ParserCause.BOUNDARY_UNRESOLVED, ParserCheck.COMMAND_WORD_BOUNDARY
                        )
                    continue
                if command == "xargs":
                    if (
                        option
                        in {
                            "--null",
                            "--no-run-if-empty",
                            "--verbose",
                            "--interactive",
                            "--exit",
                            "--open-tty",
                            "--max-lines",
                            "--eof",
                        }
                        or re.fullmatch(r"-(?:[0rtpox]+|[nLPRsS][0-9]+|[adE].+)", option)
                        or re.fullmatch(
                            r"--(?:arg-file|delimiter|max-args|max-procs|max-chars|max-lines|eof)=.+",
                            option,
                        )
                    ):
                        continue
                    # In particular, combined argument-taking flags can hide
                    # a replacement option. Unknown forms grant no proof.
                    return failed()
                # Unlike nice/xargs, sudo's -n is a flag, not an operand option.
                if command == "sudo" and option == "-n":
                    continue
                if option in {
                    "-u",
                    "-g",
                    "-h",
                    "-p",
                    "-C",
                    "-T",
                    "-R",
                    "-D",
                    "-n",
                    "-I",
                    "-L",
                    "-P",
                    "-s",
                }:
                    _, limited = next_word()
                    if limited:
                        return failed()
            else:
                return failed(
                    ParserCause.RESOURCE_LIMIT,
                    ParserCheck.WRAPPER_ARGUMENTS,
                    {
                        "observed_records": _SHELL_COMMAND_STRING_ARGUMENTS,
                        "limit_records": _SHELL_COMMAND_STRING_ARGUMENTS,
                    },
                    lower_bound=True,
                )
            continue
        if command == "timeout":
            wrapper_seen = True
            while True:
                option, limited = next_word()
                if limited:
                    return failed()
                if option is None:
                    return (
                        (False, None)
                        if _ends_wrapped_clause(content, cursor)
                        else failed(
                            ParserCause.BOUNDARY_UNRESOLVED, ParserCheck.COMMAND_WORD_BOUNDARY
                        )
                    )
                if option in {"-k", "--kill-after", "-s", "--signal"}:
                    _, limited = next_word()
                    if limited:
                        return failed()
                    continue
                if option.startswith("-"):
                    continue
                break
            pending, limited = next_word()
            if limited:
                return failed()
            continue
        return False, None
    return (
        failed(
            ParserCause.RESOURCE_LIMIT,
            ParserCheck.WRAPPER_CLAUSE_STEPS,
            {"observed_records": 32, "limit_records": 32},
            lower_bound=True,
        )
        if wrapper_seen
        else (False, None)
    )


def _ends_wrapped_clause(content: str, cursor: int) -> bool:
    """Return whether a wrapper's missing command is a proven end of its clause.

    When a control operator follows ``sudo``, ``nice``, ``xargs`` or
    ``timeout`` directly, the clause names no wrapped command, so there is no
    command string to reconstruct. A shell such a wrapper starts on its own
    (``sudo -s``) reads its input like ``sh`` in a pipeline, which this check
    does not model either. Commands after the operator start their own
    clause and are checked separately. A redirection may still precede the
    wrapped command, and a fragment may continue past its end, so both
    remain unresolved.
    """
    if cursor >= len(content):
        return False
    character = content[cursor]
    return character in ";|)" or (character == "&" and content[cursor + 1 : cursor + 2] != ">")


def _command_wrapper_quote(content: str, command_start: int) -> str | None:
    if command_start == 0 or content[command_start - 1] not in "'\"`":
        return None
    backslashes = 0
    cursor = command_start - 2
    while cursor >= 0 and content[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return content[command_start - 1] if backslashes % 2 == 0 else None


def _assignment_value_wrapper(content: str, value_start: int) -> str | None:
    """Return an unescaped quote or backtick directly enclosing an assignment."""
    name_start = value_start - 1
    if name_start > 0 and content[name_start - 1] == "+":
        name_start -= 1
    while name_start > 0 and (
        content[name_start - 1] == "_"
        or (content[name_start - 1].isascii() and content[name_start - 1].isalnum())
    ):
        name_start -= 1
    return _command_wrapper_quote(content, name_start)


def _perl_literal_print_shell_text(
    content: str,
    check_runtime: Callable[[], None],
) -> str:
    """Project only proven ordinary Perl print delimiters, retaining the payload.

    This deliberately recognizes a small source subset, not general Perl: a
    standalone print of one non-interpolated, single-line quoted literal, with
    an optional standard output handle or parentheses. Ordinary surrounding
    quotes and comments must also balance. Quote operators, heredocs, regexes,
    and interpolation make host ownership uncertain, so leave that source on
    the conservative path. Never use this to grant ownership to a fragment.

    Only the two host delimiters change, with offsets preserved. In particular,
    commands, prompt instructions and parser-limit payloads remain available to
    the security scanners; printed text is not an analysis exemption.
    """
    literals: set[tuple[int, int]] = set()
    check_runtime()
    for match in _PERL_LITERAL_PRINT_RE.finditer(content):
        check_runtime()
        literals.add(match.span("literal"))
    check_runtime()
    if not literals:
        return content
    owned: list[tuple[int, int]] = []
    cursor = 0
    next_runtime_check = 0
    while cursor < len(content):
        if cursor >= next_runtime_check:
            check_runtime()
            next_runtime_check = cursor + 256
        character = content[cursor]
        if character == "#" and (cursor == 0 or content[cursor - 1] != "$"):
            newline = content.find("\n", cursor)
            cursor = len(content) if newline < 0 else newline + 1
            continue
        if character in "$@%&*" and _PERL_AMBIGUOUS_SIGIL_RE.match(content, cursor) is not None:
            # Perl sigils can own quote punctuation (for example $' and *"),
            # with whitespace/comments or braces inside the variable spelling.
            # Leave those ambiguous forms outside this bounded source subset.
            return content
        if (
            character == "/"
            or content.startswith("<<", cursor)
            or character in "qmsty"
            and (cursor == 0 or content[cursor - 1] not in "$@%&")
            and _PERL_QUOTE_OPERATOR_RE.match(content, cursor) is not None
        ):
            return content
        if character not in "'\"`":
            cursor += 1
            continue
        if cursor > 0:
            previous = content[cursor - 1]
            if character == "'" and (
                previous.isalnum() or previous in "_:@%&*" or ord(previous) > 127
            ):
                # Apostrophes can separate legacy package names. Treat ambiguous adjacency
                # (including print'...') conservatively instead of inventing
                # a string that could hide an executable quote operator.
                return content
        start = cursor
        quote = character
        cursor += 1
        while cursor < len(content):
            if cursor >= next_runtime_check:
                check_runtime()
                next_runtime_check = cursor + 256
            character = content[cursor]
            if character == "\\":
                cursor += 2
                continue
            if character == quote:
                cursor += 1
                if (start, cursor) in literals:
                    owned.append((start, cursor))
                break
            if quote != "'" and character in "$@":
                # Interpolation can itself contain Perl expressions/quotes.
                return content
            cursor += 1
        else:
            return content
    if not owned:
        return content
    output = list(content)
    for start, end in owned:
        output[start] = output[end - 1] = " "
    return "".join(output)


class _PowerShellOwnershipUnprovenError(Exception):
    """PowerShell source left the bounded token subset this projection proves."""


@dataclass(eq=False)
class _PowerShellSegment:
    """One pipeline element: a command name and the strings it receives."""

    frame: str
    command: str | None = None
    started: bool = False
    captured: bool = False
    first_start: int = -1
    strings: list[tuple[int, int, list[tuple[int, int]]]] = field(default_factory=list)

    def owns_messages(self) -> bool:
        if self.captured or self.command is None:
            return False
        if self.command in _POWERSHELL_STREAM_ONLY_COMMANDS:
            return self.frame != "hash"
        return self.command in _POWERSHELL_OUTPUT_COMMANDS and self.frame == "top"


class _PowerShellProjection:
    """Lex the PowerShell token subset that proves comment and message ownership.

    The lexer tracks comments, quoted strings, here-strings, subexpressions,
    script blocks and pipeline boundaries. Any construct whose tokenization
    depends on parser mode (a ``#`` or ``<#`` inside a word, a word glued to
    a closing quote, stop-parsing ``--%``) or that does not close raises
    ``_PowerShellOwnershipUnprovenError``, so the caller keeps the raw source.
    """

    def __init__(self, content: str, check_runtime: Callable[[], None]) -> None:
        self.content = content
        self.check_runtime = check_runtime
        self.next_runtime_check = 0
        self.comments: list[tuple[int, int]] = []
        self.continuations: list[int] = []
        self.segments: list[_PowerShellSegment] = []

    def tick(self, position: int) -> None:
        if position >= self.next_runtime_check:
            self.check_runtime()
            self.next_runtime_check = position + 4096

    def code(self, position: int, closer: str | None, frame: str, depth: int) -> int:
        """Lex code until ``closer`` (or the end of input) and return its end."""
        if depth > _POWERSHELL_MAX_NESTING:
            raise _PowerShellOwnershipUnprovenError
        content = self.content
        length = len(content)
        segment = self.segment(frame)
        last_segment: _PowerShellSegment | None = None
        token_start = True
        # The last code character and token of this frame decide whether a
        # newline ends the statement or the statement continues past it.
        last_code_end = 0
        comma_end = -1
        last_token_start = -1
        while position < length:
            self.tick(position)
            character = content[position]
            following = content[position + 1 : position + 2]
            if character in "\r\n" and last_code_end:
                last_code = content[last_code_end - 1]
                if last_code_end == comma_end or (
                    segment.command is None
                    and segment.started
                    and (
                        last_code in "=+-*/%!<>.:"
                        or (last_token_start >= 0 and content[last_token_start] == "-")
                    )
                ):
                    # A trailing comma continues a command's arguments. An
                    # expression also continues after an operator; such an
                    # element names no command and never owns a message.
                    position += 1
                    token_start = True
                    continue
            if character in " \t\f\v,":
                if character == ",":
                    last_code_end = comma_end = position + 1
                position += 1
                token_start = True
                continue
            if character == "#" or (character == "<" and following == "#"):
                if not token_start:
                    raise _PowerShellOwnershipUnprovenError
                if character == "#":
                    end = position
                    while end < length and content[end] not in "\r\n":
                        end += 1
                else:
                    end = content.find("#>", position + 2)
                    if end < 0:
                        raise _PowerShellOwnershipUnprovenError
                    end += 2
                self.comments.append((position, end))
                position = end
                continue
            if (
                token_start
                and character not in "\r\n;|&)}"
                and not (character == "`" and following in "\r\n")
            ):
                last_token_start = position
            if character in "\r\n;" or (
                character == "&"
                and following != "&"
                and segment.started
                and _POWERSHELL_STATEMENT_END_RE.match(content, position + 1) is not None
            ):
                # A newline, semicolon or trailing background operator ends
                # the pipeline element without passing its output onward.
                if segment.started:
                    last_segment = segment
                    segment = self.segment(frame)
                position += 1
                token_start = True
            elif closer is not None and character == closer:
                return position + 1
            elif character in ")}":
                raise _PowerShellOwnershipUnprovenError
            elif character == "|" or character == "&":
                if following == character:
                    # ``&&`` and ``||`` chain statements by status. The chain
                    # shares one assignment or redirection target.
                    position += 2
                    if segment.started:
                        last_segment = segment
                        segment = self.segment(frame)
                        segment.captured = last_segment.captured
                elif character == "|":
                    # The previous element's output flows into the next one.
                    # PowerShell 7 also accepts a pipe that starts a new line.
                    upstream = segment if segment.started else last_segment
                    segment = self.segment(frame)
                    if upstream is not None:
                        # The last element's output goes wherever the
                        # pipeline's does, such as an assignment upstream.
                        segment.captured = upstream.captured
                        upstream.captured = True
                    position += 1
                else:
                    # A call operator runs a computed command.
                    segment.started = segment.captured = True
                    position += 1
                token_start = True
            elif character == "`":
                if position + 1 >= length:
                    raise _PowerShellOwnershipUnprovenError
                if following in " \t\f\v" or content.startswith("--%", position + 1):
                    # An escaped blank splits or joins words by parser mode,
                    # and an escaped stop-parsing token still stops parsing.
                    raise _PowerShellOwnershipUnprovenError
                if following in "\r\n":
                    if not token_start:
                        # Glued to a word, the escaped newline either ends a
                        # parameter or joins the word, depending on mode.
                        raise _PowerShellOwnershipUnprovenError
                    # A line continuation is whitespace within one statement.
                    self.continuations.append(position)
                    position += 3 if content.startswith("\r\n", position + 1) else 2
                    token_start = True
                    continue
                self.word(segment, position, token_start)
                position += 2
                token_start = False
            elif character in _POWERSHELL_QUOTES:
                standalone = token_start
                if not standalone:
                    if last_token_start == segment.first_start:
                        # A quote inside the command-name word builds the
                        # name itself; its parts are not message arguments.
                        raise _PowerShellOwnershipUnprovenError
                    if content[position - 1] in ".:[":
                        # A member name, parameter value or index is not a
                        # message argument.
                        segment.captured = True
                self.word(segment, position, token_start)
                position = self.string(segment, position, depth)
                if standalone:
                    token_start = self.string_suffix(segment, position)
                else:
                    token_start = False
                    if content[position : position + 1] in (".", "["):
                        segment.captured = True
            elif character == "@" and following in _POWERSHELL_QUOTES:
                if not token_start:
                    raise _PowerShellOwnershipUnprovenError
                self.word(segment, position, token_start)
                position = self.here_string(segment, position, depth)
                token_start = self.string_suffix(segment, position)
            elif character in "$@" and following == "(":
                self.word(segment, position, token_start)
                position = self.code(position + 2, ")", "paren", depth + 1)
                token_start = False
            elif character == "@" and following == "{":
                self.word(segment, position, token_start)
                position = self.code(position + 2, "}", "hash", depth + 1)
                token_start = False
            elif character == "(":
                self.word(segment, position, token_start)
                position = self.code(position + 1, ")", "paren", depth + 1)
                token_start = True
            elif character == "{":
                self.word(segment, position, token_start)
                position = self.code(position + 1, "}", "block", depth + 1)
                token_start = True
            elif character == "$" and following == "{":
                self.word(segment, position, token_start)
                position = self.braced_variable(position)
                token_start = False
            else:
                if content.startswith("--%", position):
                    # Stop-parsing passes the rest of the line verbatim.
                    raise _PowerShellOwnershipUnprovenError
                if character in "=>":
                    # Assignment and redirection can retain a message as data.
                    segment.captured = True
                elif token_start and character == "@" and following.isalnum():
                    # Splatting can supply a capture parameter.
                    segment.captured = True
                elif token_start and character == "-" and segment.command is not None:
                    match = _POWERSHELL_PARAMETER_RE.match(content, position)
                    if match is not None:
                        name = match.group(1).lower()
                        if name in _POWERSHELL_CAPTURE_PARAMETER_ALIASES or any(
                            parameter.startswith(name)
                            for parameter in _POWERSHELL_CAPTURE_PARAMETERS
                        ):
                            segment.captured = True
                self.word(segment, position, token_start)
                position += 1
                token_start = False
            last_code_end = position
        if closer is not None:
            raise _PowerShellOwnershipUnprovenError
        return position

    def string_suffix(self, segment: _PowerShellSegment, position: int) -> bool:
        """Classify what follows a standalone string; return whether a token starts."""
        content = self.content
        if (
            position >= len(content)
            or content[position] in _POWERSHELL_TOKEN_BREAKS
            or content[position] in _POWERSHELL_QUOTES
            or content[position] == "#"
            or content.startswith("<#", position)
        ):
            return True
        if content[position] not in ".[":
            # Whether a glued suffix continues this string's word depends on
            # parser mode; do not guess.
            raise _PowerShellOwnershipUnprovenError
        # Member access or indexing passes the string to an expression, which
        # can hand it to a script block. A later ``#`` is inside that word.
        segment.captured = True
        return False

    def segment(self, frame: str) -> _PowerShellSegment:
        segment = _PowerShellSegment(frame)
        self.segments.append(segment)
        return segment

    def word(self, segment: _PowerShellSegment, position: int, token_start: bool) -> None:
        """Record the first token of a pipeline element as its command name."""
        if not token_start or segment.started:
            return
        segment.started = True
        segment.first_start = position
        match = _POWERSHELL_COMMAND_NAME_RE.match(self.content, position)
        if match is None:
            return
        end = match.end()
        if end < len(self.content) and self.content[end] not in _POWERSHELL_TOKEN_BREAKS:
            return
        segment.command = match.group().lower()

    def braced_variable(self, position: int) -> int:
        content = self.content
        cursor = position + 2
        while cursor < len(content):
            self.tick(cursor)
            if content[cursor] == "`":
                cursor += 2
            elif content[cursor] == "}":
                return cursor + 1
            else:
                cursor += 1
        raise _PowerShellOwnershipUnprovenError

    def string(self, segment: _PowerShellSegment, position: int, depth: int) -> int:
        content = self.content
        expandable = content[position] in _POWERSHELL_DOUBLE_QUOTES
        quotes = _POWERSHELL_DOUBLE_QUOTES if expandable else _POWERSHELL_SINGLE_QUOTES
        holes: list[tuple[int, int]] = []
        cursor = position + 1
        while cursor < len(content):
            self.tick(cursor)
            character = content[cursor]
            if character in quotes:
                if content[cursor + 1 : cursor + 2] and content[cursor + 1] in quotes:
                    cursor += 2
                    continue
                segment.strings.append((position, cursor + 1, holes))
                return cursor + 1
            if expandable:
                cursor = self.expansion(cursor, holes, depth)
            else:
                cursor += 1
        raise _PowerShellOwnershipUnprovenError

    def expansion(self, cursor: int, holes: list[tuple[int, int]], depth: int) -> int:
        """Advance over one expandable-string character, lexing subexpressions."""
        content = self.content
        character = content[cursor]
        following = content[cursor + 1 : cursor + 2]
        if character == "`":
            return cursor + 2
        if character == "$" and following == "(":
            # Subexpressions are code: their commands run even when the
            # surrounding string is only a message.
            end = self.code(cursor + 2, ")", "paren", depth + 1)
            holes.append((cursor, end))
            return end
        if character == "$" and following == "{":
            return self.braced_variable(cursor)
        return cursor + 1

    def here_string(self, segment: _PowerShellSegment, position: int, depth: int) -> int:
        content = self.content
        quote = content[position + 1]
        if quote not in "'\"":
            raise _PowerShellOwnershipUnprovenError
        cursor = position + 2
        while cursor < len(content) and content[cursor] in " \t":
            cursor += 1
        if content.startswith("\r\n", cursor):
            cursor += 2
        elif content.startswith("\n", cursor):
            cursor += 1
        else:
            raise _PowerShellOwnershipUnprovenError
        holes: list[tuple[int, int]] = []
        line_start = True
        while cursor < len(content):
            self.tick(cursor)
            if line_start and content.startswith(quote + "@", cursor):
                segment.strings.append((position, cursor + 2, holes))
                return cursor + 2
            character = content[cursor]
            if quote == '"' and character != "\n":
                next_cursor = self.expansion(cursor, holes, depth)
            else:
                next_cursor = cursor + 1
            line_start = content[next_cursor - 1] == "\n"
            cursor = next_cursor
        raise _PowerShellOwnershipUnprovenError


def _powershell_shell_text(
    content: str,
    check_runtime: Callable[[], None],
    *,
    continuations: list[int] | None = None,
    allow_messages: bool = True,
) -> str:
    """Mask PowerShell comments and display-only messages, preserving offsets.

    PowerShell is not a POSIX shell: ``#`` comments, ``<# #>`` blocks,
    doubled-quote escapes, backtick escapes and here-strings otherwise look
    like unclosed shell quotes and substitutions. Comments never execute. A
    string is a message only when it is a direct argument of a host-display
    command (or a top-level ``Write-Output``) whose output is not piped,
    redirected, assigned or captured, and only when the file cannot rebind
    that command or run runtime-built code. Subexpressions inside a message
    remain code. All other strings, including payloads for ``bash -c``,
    ``Invoke-Expression``, script-block creation and call operators, keep their
    bytes for the bounded shell checks. Source this lexer cannot fully prove
    is returned unchanged, so every existing bound and partial status applies.
    """
    projection = _PowerShellProjection(content, check_runtime)
    try:
        projection.code(0, None, "top", 0)
    except _PowerShellOwnershipUnprovenError:
        return content
    masked: list[tuple[int, int]] = list(projection.comments)
    output = list(content)
    for position in projection.continuations:
        # A code-level backtick before a newline continues the statement, as
        # a backslash does in POSIX shell. Every other escape keeps its bytes,
        # so a string payload's escaped expansion still reaches the checks.
        output[position] = "\\"
        if continuations is not None:
            continuations.append(position)
    if allow_messages and _POWERSHELL_COMMAND_REBINDING_RE.search(content) is None:
        for segment in projection.segments:
            check_runtime()
            if not segment.owns_messages():
                continue
            for start, end, holes in segment.strings:
                if not holes:
                    masked.append((start, end))
                    continue
                # Keep each subexpression, including its ``$(`` and ``)``, and
                # the string's own delimiters, so the subexpression keeps the
                # quoted context its existing proofs (such as -replace) need.
                delimiter = 2 if content[start] == "@" else 1
                cursor = start + delimiter
                for hole_start, hole_end in holes:
                    masked.append((cursor, hole_start))
                    cursor = hole_end
                masked.append((cursor, end - delimiter))
    for start, end in masked:
        check_runtime()
        for index in range(start, end):
            if output[index] not in "\r\n":
                output[index] = " "
    return "".join(output)


def _python_literal_operands(
    content: str,
    token: tuple[int, int],
    start: int,
    end: int,
    constant_fields: tuple[tuple[int, int, str], ...] = (),
) -> str:
    """Return string literal body text from ``start`` that a shell reads as operands.

    Unknown replacement fields can hold recursive and force options. Preserve
    that uncertainty unless the complete field has a proven plain value.
    """
    text = content[start:end]
    body_start = _python_string_body(content, *token)[0]
    if not {"f", "t"} & set(content[token[0] : body_start].lower()):
        return text

    def unknown_fields(value: str) -> str:
        return _PYTHON_REPLACEMENT_FIELD_RE.sub(
            lambda field: field[0] if field[1] is None else " -rf {", value
        )

    pieces: list[str] = []
    cursor = start
    index = bisect_left(constant_fields, start, key=lambda field: field[0])
    while index < len(constant_fields):
        field_start, field_end, value = constant_fields[index]
        if field_start >= end:
            break
        if field_end <= end:
            pieces.append(unknown_fields(content[cursor:field_start]))
            pieces.append(value)
            cursor = field_end
        index += 1
    pieces.append(unknown_fields(content[cursor:end]))
    return "".join(pieces)


class _PythonSourceOwnership:
    """Lazily proven Python string and comment ownership for one source text.

    Python has no shell semantics. Shell text can occur only inside a Python
    string, which owns its bytes, or as prose in a comment that no shell ever
    receives. Python code between those tokens contains no shell quote or
    expansion opener, so an in-phase shell word ends at the first code-level
    whitespace or control character. A quote or substitution that is still
    open there was opened inside an earlier string or comment token of the
    same word, and that token owns the rest of it.

    The token spans are computed lazily, only for source that reaches one of
    these decisions. Invalid or ambiguous source keeps every conservative bound.
    """

    def __init__(
        self,
        content: str,
        check_runtime: Callable[[], None],
        python_import_check: Callable[[Callable[[], None]], bool] | None = None,
        python_stdlib_check: Callable[[str, Callable[[], None]], bool] | None = None,
    ) -> None:
        self._python_import_check = python_import_check
        self._python_stdlib_check = python_stdlib_check
        self._constant_fields: tuple[tuple[int, int, str], ...] | None = None
        self._content = content
        self._check_runtime = check_runtime
        self._spans: PythonLiteralSpans | None = None
        self._spans_computed = False
        self._cached_word_start = -1
        self._cached_word_end = -1

    def _owned_spans(self) -> PythonLiteralSpans | None:
        if not self._spans_computed:
            self._spans = _python_literal_spans(self._content, self._check_runtime)
            self._spans_computed = True
        return self._spans

    def _owner(self, position: int) -> tuple[int, int] | None:
        spans = self._owned_spans()
        if spans is None:
            return None
        starts, ends = spans
        index = bisect_right(starts, position) - 1
        if index >= 0 and position < ends[index]:
            return starts[index], ends[index]
        return None

    def is_code(self, position: int) -> bool:
        """Return whether a position is proven Python code, not a literal."""
        return self._owned_spans() is not None and self._owner(position) is None

    def comment(self, position: int) -> tuple[int, int] | None:
        """Return the span of a proven Python comment containing a position."""
        owner = self._owner(position)
        if owner is None or self._content[owner[0]] != "#":
            return None
        return owner

    def starts_string_body(self, position: int) -> bool:
        """Return whether a candidate starts the value of a proven string token."""
        owner = self._owner(position)
        return (
            owner is not None
            and self._content[owner[0]] != "#"
            and position == _python_string_body(self._content, *owner)[0]
        )

    def is_string_delimiter(self, position: int) -> bool:
        """Return whether Python owns a string prefix or opening/closing quote."""
        owner = self._owner(position)
        if owner is None or self._content[owner[0]] == "#":
            return False
        start, end = _python_string_body(self._content, *owner)
        return not start <= position < end

    def string_value(
        self, start: int, end: int, *, source_range: list[int] | None = None
    ) -> tuple[str, int, int] | None:
        """Return the proven string value that holds a shell word, and its span.

        A string literal's quotes are Python syntax: the shell receives only
        its body, so a word ends at the closing quote, and a word starting at
        that quote is not shell text at all. Implicit or ``+`` concatenation
        continues the value with later literal bodies. Return ``None`` for a
        word outside a string literal, or for a value that a format operator,
        method call, subscript, repetition or non-literal operand can still
        extend, also after a closing parenthesis.
        """
        spans = self._owned_spans()
        owner = self._owner(start)
        content = self._content
        if spans is None or owner is None or content[owner[0]] == "#":
            return None
        if source_range is not None:
            source_range[:] = owner
        if self._constant_fields is None:
            # Additional constant substitution requires the released execution
            # and import guards. Refusing that proof preserves lexical handling
            # and unknown-field uncertainty from the existing parser.
            constant_fields: tuple[tuple[int, int, str], ...] = ()
            if python_string_value_bounds_allowed(content, self._check_runtime) and (
                self._python_import_check is None or self._python_import_check(self._check_runtime)
            ):
                constant_fields = python_constant_string_fields(
                    content, self._check_runtime, stdlib_check=self._python_stdlib_check
                )
            self._constant_fields = constant_fields
        body_start, body_end = _python_string_body(content, *owner)
        if start >= body_end:
            return "", 0, 0
        word_start = max(start, body_start)
        word_end = max(min(end, body_end), word_start)
        # Keep the character before the word and its escaping backslashes:
        # they decide whether a quote in the body wraps the command.
        origin = word_start - 1 if word_start > body_start else word_start
        while origin > body_start and content[origin - 1] == "\\":
            origin -= 1
        # The operand tokenizer reads at most this much after the word.
        limit = word_end + _ROOT_GLOB_COMMAND_CHARS + 1
        pieces = [
            content[origin:word_end],
            _python_literal_operands(
                content, owner, word_end, min(body_end, limit), self._constant_fields
            ),
        ]
        remaining = limit - body_end
        starts, ends = spans
        index = bisect_right(starts, start) - 1

        def next_token(position: int) -> int:
            # Skip whitespace, line continuations and comment tokens.
            nonlocal index
            while True:
                gap = _PYTHON_TOKEN_GAP_RE.match(content, position)
                position = position if gap is None else gap.end()
                if (
                    index + 1 == len(starts)
                    or starts[index + 1] != position
                    or content[position] != "#"
                ):
                    return position
                index += 1
                position = ends[index]

        while remaining > 0:
            self._check_runtime()
            cursor = next_token(ends[index])
            operator = content[cursor : cursor + 1]
            if operator == "+":
                cursor = next_token(cursor + 1)
            if index + 1 < len(starts) and starts[index + 1] == cursor:
                index += 1
                literal = (starts[index], ends[index])
                if source_range is not None:
                    source_range[1] = literal[1]
                literal_start, literal_end = _python_string_body(content, *literal)
                pieces.append(
                    _python_literal_operands(
                        content, literal, literal_start, literal_end, self._constant_fields
                    )
                )
                remaining -= literal_end - literal_start
                continue
            # A parenthesized value, as in ``("$TOOL ") + "-rf /"``, can
            # still be extended after its closing parenthesis.
            while operator == ")":
                cursor = next_token(cursor + 1)
                operator = content[cursor : cursor + 1]
            if operator in _PYTHON_STRING_EXTENSION_OPERATORS:
                return None
            break
        if remaining <= 0 and self._constant_fields:
            # Constant replacement can shorten the projected text. Reaching
            # the source bound cannot prove that later operands are absent.
            return None
        return "".join(pieces), word_start - origin, word_end - origin

    def word_end(self, start: int) -> int:
        """Return the end of the host region a shell word at ``start`` can own."""
        spans = self._owned_spans()
        content = self._content
        if spans is None:
            return len(content)
        # Candidates arrive in source order. No code-level break exists
        # between a cached start and its result, so they share that result.
        if self._cached_word_start <= start <= self._cached_word_end:
            return self._cached_word_end
        starts, ends = spans
        cursor = start
        while True:
            self._check_runtime()
            index = bisect_right(starts, cursor) - 1
            if index >= 0 and cursor < ends[index]:
                cursor = ends[index]
            segment_end = starts[index + 1] if index + 1 < len(starts) else len(content)
            match = _PYTHON_SHELL_WORD_BREAK_RE.search(content, cursor, segment_end)
            if match is not None:
                result = match.start()
                break
            if segment_end == len(content):
                result = segment_end
                break
            cursor = segment_end
        self._cached_word_start, self._cached_word_end = start, result
        return result


class _JavaScriptTemplateOwnership:
    """Lazily proven ownership of template literal bytes in complete JavaScript."""

    def __init__(self, content: str, check_runtime: Callable[[], None]) -> None:
        self._content = content
        self._check_runtime = check_runtime
        self._spans: JavaScriptTemplateSpans | None = None
        self._spans_computed = False
        self._unresolved_executables: list[int] = []

    def _owned_spans(self) -> JavaScriptTemplateSpans | None:
        if not self._spans_computed:
            self._spans = _javascript_template_spans(
                self._content,
                self._check_runtime,
                unresolved_executables=self._unresolved_executables,
            )
            self._spans_computed = True
        return self._spans

    def unresolved_execution_diagnostic(self) -> StaticParserDiagnostic | None:
        """Keep dynamic execution incomplete even when raw shell text is short."""
        self._owned_spans()
        if not self._unresolved_executables:
            return None
        return StaticParserDiagnostic(
            ParserCause.RUNTIME_DEPENDENT,
            ParserCheck.COMMAND_STRING_RUNTIME,
            self._unresolved_executables[0],
            len(self._content),
            ParserUncheckedScope.REMAINING_INPUT,
        )

    def owns(self, position: int) -> bool:
        spans = self._owned_spans()
        if spans is None:
            return False
        starts, ends = spans
        index = bisect_right(starts, position) - 1
        return index >= 0 and position < ends[index]

    def word_end(self, start: int) -> int:
        """Return the end of a shell word while skipping proven template bytes."""
        spans = self._owned_spans()
        content = self._content
        if spans is None:
            return len(content)
        starts, ends = spans
        cursor = start
        while True:
            self._check_runtime()
            index = bisect_right(starts, cursor) - 1
            if index >= 0 and cursor < ends[index]:
                cursor = ends[index]
            segment_end = starts[index + 1] if index + 1 < len(starts) else len(content)
            match = _PYTHON_SHELL_WORD_BREAK_RE.search(content, cursor, segment_end)
            if match is not None:
                return match.start()
            if segment_end == len(content):
                return segment_end
            cursor = segment_end


def _skip_command_substitution(
    content: str,
    start: int,
    limit: int,
    check_runtime: Callable[[], None] | None = None,
    end_cache: dict[int, int | None] | None = None,
    parameter_end_cache: dict[int, _ParameterExpansionEnd] | None = None,
    backtick_end_cache: dict[int, int | None] | None = None,
    *,
    diagnostics: list[StaticParserDiagnostic] | None = None,
) -> int | None:
    if end_cache is not None and start in end_cache:
        if end_cache[start] is None:
            _note_parser_diagnostic(
                diagnostics,
                ParserCause.BOUNDARY_UNRESOLVED,
                ParserCheck.EXPANSION_BOUNDARY,
                start,
                limit,
                ParserUncheckedScope.EXPRESSION,
            )
        return end_cache[start]
    return _skip_shell_delimited_expansion(
        content,
        start,
        limit,
        "command",
        check_runtime,
        parameter_end_cache,
        end_cache,
        backtick_end_cache,
        diagnostics=diagnostics,
    )


def _bounded_static_substitution_end(content: str, start: int) -> int | None:
    """Return a structurally valid ``$()`` close within the static evaluator bound."""
    inner_start = start + 2
    scan_limit = min(len(content), inner_start + _PRINTF_STATIC_CHARS + 1)
    end = _skip_command_substitution(content, start, scan_limit)
    if end is None or end - 1 - inner_start > _PRINTF_STATIC_CHARS:
        return None
    return end


def _bounded_shell_tokens(
    content: str,
    command_start: int,
    body_start: int,
    *,
    check_runtime: Callable[[], None] | None = None,
    powershell_strings: Mapping[int, PowerShellString] | None = None,
    quoted_printf_allowed: bool = True,
    diagnostics: list[StaticParserDiagnostic] | None = None,
) -> tuple[tuple[_ShellToken, ...], int, bool]:
    """Return argument words from one security-view-bounded shell command."""
    runtime_check = check_runtime or (lambda: None)
    tokens: list[_ShellToken] = []
    current: list[str] = []
    current_glob_projection: list[str] = []
    word_started = False
    current_is_argument = True
    current_has_unquoted_star = False
    current_can_collapse_to_root_glob = True
    current_has_quoted_content = False
    current_unquoted_brace_open = False
    current_unquoted_brace_close = False
    current_unquoted_brace_comma = False
    current_leading_tilde_unquoted = False
    source_word_has_content = False
    expect_redirection_target = False
    quote: str | None = None
    ansi_c_quote = False
    boundary_incomplete = False
    parse_limited = False
    cursor = body_start
    limit = min(len(content), body_start + _ROOT_GLOB_COMMAND_CHARS)
    wrapper_quote = _command_wrapper_quote(content, command_start)

    def boundary_failure(
        *, full_input_checked: bool = False
    ) -> tuple[tuple[_ShellToken, ...], int, bool]:
        _note_parser_diagnostic(
            diagnostics,
            ParserCause.BOUNDARY_UNRESOLVED,
            ParserCheck.COMMAND_ARGUMENT_BOUNDARY,
            command_start,
            limit,
            ParserUncheckedScope.COMMAND_ARGUMENTS,
            measurements={
                "observed_characters": limit - body_start,
                "limit_characters": _ROOT_GLOB_COMMAND_CHARS,
            }
            if limit < len(content)
            else None,
            lower_bound=limit < len(content),
            full_input_checked=full_input_checked,
        )
        return tuple(tokens), limit, True

    def printf_value(
        start: int, end: int, *, backtick: bool = False, quoted: bool = False
    ) -> str | None:
        nonlocal parse_limited
        value, limited, diagnostic = _printf_substitution_result(
            content,
            start,
            end,
            runtime_check,
            backtick=backtick,
            quoted_result=quoted,
            recognize_limit=not parse_limited,
            check_command_context=False,
        )
        parse_limited = parse_limited or limited
        if diagnostic is not None and diagnostics is not None and not diagnostics:
            diagnostics.append(diagnostic)
        return value

    def start_word() -> None:
        nonlocal word_started, current_is_argument, expect_redirection_target
        if not word_started:
            word_started = True
            current_is_argument = not expect_redirection_target
            expect_redirection_target = False

    def mark_quoted_word() -> None:
        nonlocal current_has_quoted_content, source_word_has_content
        start_word()
        current_has_quoted_content = True
        source_word_has_content = True

    def append_piece(
        piece: str,
        *,
        unquoted: bool = False,
        dynamic_can_be_empty: bool = False,
        quoted: bool = False,
        unquoted_brace_syntax: bool = False,
    ) -> None:
        nonlocal current_has_unquoted_star
        nonlocal current_can_collapse_to_root_glob
        nonlocal current_has_quoted_content
        nonlocal current_unquoted_brace_open
        nonlocal current_unquoted_brace_close
        nonlocal current_unquoted_brace_comma
        nonlocal source_word_has_content
        start_word()
        source_word_has_content = True
        current.append(piece)
        current_glob_projection.append(
            piece
            if unquoted
            else piece.replace("*", _QUOTED_GLOB_SENTINEL).replace("?", _QUOTED_GLOB_SENTINEL)
        )
        current_has_quoted_content = current_has_quoted_content or quoted
        if unquoted_brace_syntax:
            current_unquoted_brace_open = current_unquoted_brace_open or "{" in piece
            current_unquoted_brace_close = current_unquoted_brace_close or "}" in piece
            current_unquoted_brace_comma = current_unquoted_brace_comma or "," in piece
        current_has_unquoted_star = current_has_unquoted_star or (unquoted and "*" in piece)
        if not dynamic_can_be_empty and piece:
            current_can_collapse_to_root_glob = current_can_collapse_to_root_glob and (
                unquoted and all(character in "*?" for character in piece)
            )

    def flush(*, complete: bool = True) -> None:
        nonlocal word_started
        nonlocal current_has_unquoted_star
        nonlocal current_can_collapse_to_root_glob
        nonlocal current_has_quoted_content
        nonlocal current_unquoted_brace_open
        nonlocal current_unquoted_brace_close
        nonlocal current_unquoted_brace_comma
        nonlocal current_leading_tilde_unquoted
        if word_started:
            if complete and current_is_argument:
                tokens.append(
                    _ShellToken(
                        "".join(current),
                        current_has_unquoted_star and current_can_collapse_to_root_glob,
                        "".join(current_glob_projection),
                        current_has_quoted_content,
                        current_unquoted_brace_open
                        and current_unquoted_brace_close
                        and current_unquoted_brace_comma,
                        current_leading_tilde_unquoted,
                    )
                )
            current.clear()
            current_glob_projection.clear()
            word_started = False
            current_has_unquoted_star = False
            current_can_collapse_to_root_glob = True
            current_has_quoted_content = False
            current_unquoted_brace_open = False
            current_unquoted_brace_close = False
            current_unquoted_brace_comma = False
            current_leading_tilde_unquoted = False

    while cursor < limit:
        if (cursor - body_start) % 256 == 0:
            runtime_check()
        character = content[cursor]
        native_string = powershell_strings.get(cursor) if powershell_strings is not None else None
        if quote is None and native_string is not None:
            if native_string.end > limit:
                _note_parser_diagnostic(
                    diagnostics,
                    ParserCause.RESOURCE_LIMIT,
                    ParserCheck.COMMAND_ARGUMENT_CHARACTERS,
                    command_start,
                    limit + 1,
                    ParserUncheckedScope.COMMAND_ARGUMENTS,
                    measurements={
                        "observed_characters": limit - body_start + 1,
                        "limit_characters": _ROOT_GLOB_COMMAND_CHARS,
                    },
                    lower_bound=True,
                )
                return tuple(tokens), limit, True
            # PowerShell's backtick escapes belong to one native argument.
            # They cannot open Bash command substitution or consume later
            # lines. Decoded flags remain visible to the operand checks.
            mark_quoted_word()
            append_piece(
                native_string.value,
                quoted=True,
                dynamic_can_be_empty="$" in native_string.value,
            )
            cursor = native_string.end
            continue
        if quote is not None:
            if character == quote:
                quote = None
                ansi_c_quote = False
            elif character == "\\" and ansi_c_quote:
                decoded, cursor = _decode_ansi_c_escape(content, cursor, limit)
                if "\x00" in decoded:
                    closing_quote = _skip_ansi_c_quote_tail(content, cursor, limit)
                    if closing_quote is None:
                        return boundary_failure()
                    cursor = closing_quote
                    continue
                append_piece(decoded, quoted=True)
                continue
            elif quote == '"' and character == "$" and cursor + 1 < limit:
                inherited_quote_closed = [False]
                if content[cursor + 1] == "(":
                    substitution_end = _skip_command_substitution(
                        content,
                        cursor,
                        limit,
                        runtime_check,
                        diagnostics=diagnostics,
                    )
                    if substitution_end is None:
                        return boundary_failure()
                    static_value = printf_value(
                        cursor,
                        substitution_end,
                        quoted=quoted_printf_allowed,
                    )
                    append_piece(
                        "$DYNAMIC" if static_value is None else static_value,
                        dynamic_can_be_empty=static_value is None,
                        quoted=True,
                    )
                    cursor = substitution_end
                    continue
                parameter_end = _skip_parameter_expansion(
                    content,
                    cursor,
                    limit,
                    runtime_check,
                    inherited_double_quote=True,
                    inherited_quote_closed=inherited_quote_closed,
                    diagnostics=diagnostics,
                )
                if parameter_end is not None:
                    append_piece(
                        "$DYNAMIC",
                        dynamic_can_be_empty=True,
                        quoted=True,
                    )
                    cursor = parameter_end
                    if inherited_quote_closed[0]:
                        quote = None
                        ansi_c_quote = False
                    continue
            elif quote == '"' and character == "`":
                substitution_end = _skip_backtick_substitution(
                    content,
                    cursor,
                    limit,
                    runtime_check,
                    diagnostics=diagnostics,
                )
                if substitution_end is None:
                    return boundary_failure()
                static_value = printf_value(
                    cursor,
                    substitution_end,
                    backtick=True,
                )
                append_piece(
                    "$DYNAMIC" if static_value is None else static_value,
                    dynamic_can_be_empty=static_value is None,
                    quoted=True,
                )
                cursor = substitution_end
                continue
            elif character == "\\" and quote == '"' and cursor + 1 < limit:
                escaped = content[cursor + 1]
                if escaped == "\n":
                    cursor += 2
                    continue
                if escaped == "\r" and cursor + 2 < limit and content[cursor + 2] == "\n":
                    cursor += 3
                    continue
                if escaped in '$`"\\':
                    append_piece(escaped, quoted=True)
                else:
                    append_piece("\\" + escaped, quoted=True)
                cursor += 2
                continue
            elif character == "\\" and quote == '"':
                boundary_incomplete = True
                cursor += 1
                continue
            else:
                append_piece(character, quoted=True)
            cursor += 1
            continue
        if wrapper_quote is not None and character == wrapper_quote:
            flush()
            return tuple(tokens), cursor, parse_limited
        if character == "$" and cursor + 1 < limit and content[cursor + 1] in "'\"":
            mark_quoted_word()
            quote = content[cursor + 1]
            ansi_c_quote = quote == "'"
            cursor += 2
            continue
        if character in "'\"":
            mark_quoted_word()
            quote = character
        elif character == "`":
            substitution_end = _skip_backtick_substitution(
                content,
                cursor,
                limit,
                runtime_check,
                diagnostics=diagnostics,
            )
            if substitution_end is None:
                return boundary_failure()
            static_value = printf_value(
                cursor,
                substitution_end,
                backtick=True,
            )
            append_piece(
                "$DYNAMIC" if static_value is None else static_value,
                unquoted=static_value is not None,
                dynamic_can_be_empty=static_value is None,
            )
            cursor = substitution_end
            continue
        elif character == "$" and cursor + 1 < limit and content[cursor + 1] == "(":
            substitution_end = _skip_command_substitution(
                content,
                cursor,
                limit,
                runtime_check,
                diagnostics=diagnostics,
            )
            if substitution_end is None:
                return boundary_failure()
            static_value = printf_value(cursor, substitution_end)
            append_piece(
                "$DYNAMIC" if static_value is None else static_value,
                unquoted=static_value is not None,
                dynamic_can_be_empty=static_value is None,
            )
            cursor = substitution_end
            continue
        elif character == "$":
            parameter_end = _skip_parameter_expansion(
                content,
                cursor,
                limit,
                runtime_check,
                diagnostics=diagnostics,
            )
            if parameter_end is not None:
                if _is_ifs_expansion(content, cursor, parameter_end):
                    source_word_has_content = True
                    flush()
                else:
                    append_piece("$DYNAMIC", dynamic_can_be_empty=True)
                cursor = parameter_end
                continue
            if cursor + 1 == limit and limit < len(content) and content[limit] == "(":
                boundary_incomplete = True
        elif character in "<>" and cursor + 1 < limit and content[cursor + 1] == "(":
            substitution_end = _skip_command_substitution(
                content,
                cursor,
                limit,
                runtime_check,
                diagnostics=diagnostics,
            )
            if substitution_end is None:
                return boundary_failure()
            append_piece(character + "$DYNAMIC", dynamic_can_be_empty=True)
            cursor = substitution_end
            continue
        elif character == "\\":
            if cursor + 1 >= limit:
                boundary_incomplete = True
                cursor += 1
                continue
            if content[cursor + 1] == "\n":
                cursor += 2
                continue
            if content[cursor + 1] == "\r" and cursor + 2 < limit and content[cursor + 2] == "\n":
                cursor += 3
                continue
            append_piece(content[cursor + 1])
            cursor += 1
        elif character == "\n":
            flush()
            return tuple(tokens), cursor, parse_limited
        elif character.isspace():
            flush()
            source_word_has_content = False
        elif character == "#" and not word_started:
            return tuple(tokens), cursor, parse_limited
        elif character in "<>":
            flush()
            source_word_has_content = False
            expect_redirection_target = True
            if cursor + 1 < limit and content[cursor + 1] == character:
                cursor += 1
            if cursor + 1 < limit and content[cursor + 1] in "&|":
                cursor += 1
        elif (
            character == "&"
            and cursor + 1 == limit
            and limit < len(content)
            and content[limit] == ">"
        ):
            boundary_incomplete = True
            cursor += 1
            continue
        elif character == "&" and cursor + 1 < limit and content[cursor + 1] == ">":
            flush()
            source_word_has_content = False
            expect_redirection_target = True
            cursor += 1
            if cursor + 1 < limit and content[cursor + 1] == ">":
                cursor += 1
        elif character in ";|&()":
            flush()
            return tuple(tokens), cursor, parse_limited
        else:
            if not source_word_has_content and character == "~":
                current_leading_tilde_unquoted = True
            append_piece(
                character,
                unquoted=character in "*?",
                unquoted_brace_syntax=character in "{},",
            )
        cursor += 1
    neutral_state = quote is None and not boundary_incomplete and not expect_redirection_target
    known_terminator = (
        limit < len(content)
        and neutral_state
        and (
            content[limit] in "\n;|&()"
            or content[limit] == wrapper_quote
            or (content[limit] == "#" and not word_started)
        )
    )
    complete = (
        limit == len(content) and neutral_state and wrapper_quote is None
    ) or known_terminator
    flush(complete=complete)
    if not complete:
        if limit < len(content) and neutral_state:
            _note_parser_diagnostic(
                diagnostics,
                ParserCause.RESOURCE_LIMIT,
                ParserCheck.COMMAND_ARGUMENT_CHARACTERS,
                command_start,
                limit + 1,
                ParserUncheckedScope.COMMAND_ARGUMENTS,
                measurements={
                    "observed_characters": limit - body_start + 1,
                    "limit_characters": _ROOT_GLOB_COMMAND_CHARS,
                },
                lower_bound=True,
            )
        return boundary_failure(full_input_checked=limit == len(content))
    return tuple(tokens), limit, not complete or parse_limited


def _is_root_glob_documentation(
    content: str,
    command_start: int,
    body_start: int,
) -> bool:
    """Return whether ``rm`` is the noun in a bounded prose description."""
    context_start = max(0, command_start - 256)
    previous_boundary = max(
        content.rfind("\n", context_start, command_start),
        content.rfind(";", context_start, command_start),
    )
    prefix_truncated = previous_boundary < 0 and context_start > 0
    clause_start = previous_boundary + 1 if previous_boundary >= 0 else context_start
    context_end = min(len(content), command_start + 512)
    ends = [
        boundary
        for boundary in (
            content.find("\n", command_start, context_end),
            content.find(";", command_start, context_end),
        )
        if boundary >= 0
    ]
    clause_end = min(ends, default=context_end)
    clause = content[clause_start:clause_end]
    if _ROOT_GLOB_DOCUMENTATION_LINE_RE.fullmatch(clause) is not None:
        return True

    relative_start = command_start - clause_start
    prefix = clause[:relative_start]
    suffix = clause[max(relative_start, body_start - clause_start) :]
    if not prefix.strip() or prefix_truncated:
        return False
    execution = _ROOT_GLOB_EXECUTION_PREFIX_RE.search(prefix)
    negated_execution = False
    if execution is not None:
        negation_prefix = prefix[: execution.start("action")]
        if _ROOT_GLOB_AFFIRMATIVE_NEGATION_PREFIX_RE.fullmatch(negation_prefix) is not None:
            return False
        if _ROOT_GLOB_NEGATED_EXECUTION_RE.search(negation_prefix) is None:
            return False
        negated_execution = True
    if _ROOT_GLOB_SAFETY_WARNING_PREFIX_RE.fullmatch(prefix) is not None:
        return True
    if (
        re.search(
            r"\b(?:sudo|env|xargs|command)\b[^.;\n]{0,64}$",
            prefix,
            re.IGNORECASE,
        )
        is not None
    ):
        return False
    article_prefix = re.fullmatch(
        r"[ \t]*(?:the|this|a|an)(?:[ \t]+(?:gnu|posix|unix))?[ \t]*",
        prefix,
        re.IGNORECASE,
    )
    documentary_prefix = re.search(
        r"(?:\b(?:explain|describe|document|note|state|say)(?:[ \t]+that)?|"
        r"(?:documentation|docs?|guide|manual|reference|note|example)[ \t]*:)"
        r"[ \t]*$",
        prefix,
        re.IGNORECASE,
    )
    documentary_leadin = re.fullmatch(
        r"[ \t]*(?:(?:in|under)[ \t]+(?:the[ \t]+)?(?:posix(?:\.[0-9]+)?|gnu|unix)"
        r"(?:[ \t]+(?:standard|specification))?|"
        r"according[ \t]+to[ \t]+(?:the[ \t]+)?(?:manual|documentation|reference|"
        r"(?:posix(?:\.[0-9]+)?|gnu|unix)(?:[ \t]+(?:standard|specification))?)|"
        r"for[ \t]+reference|"
        r"this[ \t]+(?:section|guide|documentation)[ \t]+"
        r"(?:explains?|describes?|documents?|notes?|states?|says?)(?:[ \t]+that)?)"
        r"[ \t]*,?[ \t]+(?:(?:the|this|a|an)[ \t]+)?"
        r"(?:(?:gnu|posix(?:\.[0-9]+)?|unix)[ \t]+)?",
        prefix,
        re.IGNORECASE,
    )
    if (
        article_prefix is None
        and documentary_prefix is None
        and documentary_leadin is None
        and not negated_execution
    ):
        return False
    describes_flags = re.search(
        r"\b(?:command|utility|tool)\b|\b(?:accepts?|supports?|uses?|takes?)\b",
        suffix,
        re.IGNORECASE,
    )
    describes_glob = re.search(
        r"\b(?:denotes?|means?|represents?|matches?|wildcards?|globs?)\b|"
        r"\b(?:all|every)[ \t]+files?\b",
        suffix,
        re.IGNORECASE,
    )
    return describes_flags is not None and describes_glob is not None and "*" in suffix


def _static_brace_expansions(
    word: str, diagnostics: list[StaticParserDiagnostic] | None = None
) -> tuple[str, ...] | None:
    """Expand a small, literal Bash brace word without executing shell code."""
    if len(word) > _STATIC_BRACE_WORD_CHARS:
        _note_parser_diagnostic(
            diagnostics,
            ParserCause.RESOURCE_LIMIT,
            ParserCheck.BRACE_CHARACTERS,
            0,
            len(word),
            ParserUncheckedScope.COMMAND_ARGUMENTS,
            measurements={
                "observed_characters": len(word),
                "limit_characters": _STATIC_BRACE_WORD_CHARS,
            },
        )
        return None
    expanded = [word]
    changed = False
    for _ in range(4):
        next_words: list[str] = []
        expanded_this_round = False
        for candidate in expanded:
            brace = re.search(r"\{(?P<body>[^{}]*)\}", candidate)
            if brace is None or "," not in brace.group("body"):
                next_words.append(candidate)
                continue
            alternatives = brace.group("body").split(",")
            if len(alternatives) > 8:
                _note_parser_diagnostic(
                    diagnostics,
                    ParserCause.RESOURCE_LIMIT,
                    ParserCheck.BRACE_ALTERNATIVES,
                    0,
                    len(word),
                    ParserUncheckedScope.COMMAND_ARGUMENTS,
                    measurements={"observed_records": len(alternatives), "limit_records": 8},
                )
                return None
            next_words.extend(
                candidate[: brace.start()] + alternative + candidate[brace.end() :]
                for alternative in alternatives
            )
            if len(next_words) > 32:
                _note_parser_diagnostic(
                    diagnostics,
                    ParserCause.RESOURCE_LIMIT,
                    ParserCheck.BRACE_PRODUCTS,
                    0,
                    len(word),
                    ParserUncheckedScope.COMMAND_ARGUMENTS,
                    measurements={"observed_records": len(next_words), "limit_records": 32},
                    lower_bound=True,
                )
                return None
            expanded_this_round = True
            changed = True
        expanded = next_words
        if not expanded_this_round:
            break
    if any("{" in candidate or "}" in candidate for candidate in expanded):
        _note_parser_diagnostic(
            diagnostics,
            ParserCause.RESOURCE_LIMIT if expanded_this_round else ParserCause.UNSUPPORTED_SYNTAX,
            ParserCheck.BRACE_ROUNDS if expanded_this_round else ParserCheck.BRACE_SYNTAX,
            0,
            len(word),
            ParserUncheckedScope.COMMAND_ARGUMENTS,
            measurements={"observed_depth": 4, "limit_depth": 4} if expanded_this_round else None,
            lower_bound=expanded_this_round,
        )
        return None
    if not changed:
        _note_parser_diagnostic(
            diagnostics,
            ParserCause.UNSUPPORTED_SYNTAX,
            ParserCheck.BRACE_SYNTAX,
            0,
            len(word),
            ParserUncheckedScope.COMMAND_ARGUMENTS,
        )
        return None
    return tuple(expanded)


def _has_recursive_force_options(tokens: tuple[_ShellToken, ...]) -> bool:
    """Interpret options using GNU/BSD ``rm`` syntax.

    Generic destructive-command paths also call this helper, but ``del`` and
    ``erase`` use slash options rather than one-dash clusters. Runtime-selected
    commands stay conservative because they may resolve to rm.
    """
    try:
        options_end = next(index for index, token in enumerate(tokens) if token.text == "--")
    except StopIteration:
        options_end = len(tokens)
    option_words: list[str] = []
    for token in tokens[:options_end]:
        if token.brace_expansion:
            expanded = _static_brace_expansions(token.text)
            if expanded is not None:
                option_words.extend(expanded)
                continue
        option_words.append(token.text)
    # A one-dash word is an rm option cluster only when every character is a
    # supported short option. For example, ``-force`` is not ``-f`` + ``-r``:
    # rm rejects its unsupported ``o``, ``c`` and ``e`` options before acting.
    rm_short_options = frozenset("fIiPRrWdvx")
    short_options = [
        word[1:]
        for word in option_words
        if re.fullmatch(r"-[A-Za-z]+", word) is not None and set(word[1:]) <= rm_short_options
    ]
    recursive = "--recursive" in option_words or any(
        "r" in option or "R" in option for option in short_options
    )
    force = "--force" in option_words or any("f" in option for option in short_options)
    return recursive and force


def _has_destructive_root_glob(tokens: tuple[_ShellToken, ...]) -> bool:
    def is_root_glob(token: _ShellToken) -> bool:
        if token.root_glob:
            return True
        if not token.brace_expansion:
            return False
        expanded = _static_brace_expansions(token.glob_projection)
        return expanded is not None and any(
            re.fullmatch(r"[*?]+", word) is not None for word in expanded
        )

    return any(is_root_glob(token) for token in tokens) and _has_recursive_force_options(tokens)


def _has_destructive_root_path(tokens: tuple[_ShellToken, ...]) -> bool:
    """Return whether recursive-force options target root or home expansion."""

    def is_root_path(token: _ShellToken) -> bool:
        candidates: tuple[str, ...] = (token.text,)
        if token.brace_expansion:
            expanded = _static_brace_expansions(token.text)
            if expanded is not None:
                candidates = expanded
        return any(candidate.startswith("/") for candidate in candidates) or (
            token.leading_tilde_unquoted
            and any(candidate.startswith("~") for candidate in candidates)
        )

    has_root_path = any(is_root_path(token) for token in tokens)
    return has_root_path and _has_recursive_force_options(tokens)


def _has_unsupported_brace_expansion(tokens: tuple[_ShellToken, ...]) -> bool:
    return _brace_expansion_diagnostic(tokens, 0, 1) is not None


def _brace_expansion_diagnostic(
    tokens: tuple[_ShellToken, ...], start: int, end: int
) -> StaticParserDiagnostic | None:
    for token in tokens:
        if token.brace_expansion:
            diagnostics: list[StaticParserDiagnostic] = []
            if _static_brace_expansions(token.text, diagnostics) is None:
                return diagnostics[0].enclosed(start, max(start + 1, end))
    return None


def _projected_subprocess_qualifier(
    content: str,
    popen_start: int,
    maximum_lookbehind: int,
) -> tuple[int, int, str, bool] | None:
    """Return a security-view ``subprocess.`` suffix and its raw source span."""
    qualifier = "subprocess."
    if popen_start <= 0:
        return None
    maximum = min(popen_start, max(0, maximum_lookbehind))
    lookbehind = min(maximum, 64)
    while lookbehind:
        prefix_start = popen_start - lookbehind
        # Include the first method character so contextual default-ignorable
        # handling sees the same right boundary as the full security view.
        needs_more_source = False
        source = content[prefix_start : popen_start + 1]
        for view in security_text_views(source):
            derived_popen_start = (
                popen_start - prefix_start
                if view.source_offsets is None
                else bisect_left(view.source_offsets, popen_start - prefix_start)
            )
            prefix = view.text[:derived_popen_start]
            if len(prefix) < len(qualifier):
                needs_more_source = True
                continue
            suffix_start = len(prefix) - len(qualifier)
            suffix = prefix[suffix_start:]
            if suffix.casefold() == qualifier:
                return (
                    prefix_start + view.source_offset(suffix_start),
                    prefix_start + view.source_offset(derived_popen_start - 1) + 1,
                    suffix,
                    view.name != "raw",
                )
        if not needs_more_source:
            return None
        if lookbehind == maximum:
            return None
        lookbehind = min(maximum, lookbehind * 2)
    return None


def _subprocess_qualifier_start(content: str, popen_start: int) -> int | None:
    """Find a bounded security-view ``subprocess.`` suffix before bare Popen."""
    qualifier = _projected_subprocess_qualifier(
        content,
        popen_start,
        static_runner._WINDOW_OVERLAP_CHARS + len("subprocess."),
    )
    return qualifier[0] if qualifier is not None else None


def _outer_shell_true_anchor(matched_text: str) -> int | None:
    """Locate the first outer-call shell literal, ignoring quoted lookalikes."""
    stack: list[str] = []
    quote: str | None = None
    triple = False
    in_comment = False
    cursor = 0
    pairs = {")": "(", "]": "[", "}": "{"}
    while cursor < len(matched_text):
        character = matched_text[cursor]
        if in_comment:
            if character in LINE_BREAK_CHARS:
                in_comment = False
            cursor += 1
            continue
        if quote is not None:
            if character == "\\":
                cursor = min(len(matched_text), cursor + 2)
                continue
            marker = quote * (3 if triple else 1)
            if matched_text.startswith(marker, cursor):
                cursor += len(marker)
                quote = None
                triple = False
            else:
                cursor += 1
            continue
        if character in "'\"":
            quote = character
            triple = matched_text.startswith(character * 3, cursor)
            cursor += 3 if triple else 1
            continue
        if character == "#":
            in_comment = True
            cursor += 1
            continue
        if character in "([{":
            stack.append(character)
            cursor += 1
            continue
        if character in pairs:
            if stack and stack[-1] == pairs[character]:
                stack.pop()
            cursor += 1
            continue
        if (
            stack == ["("]
            and (
                cursor == 0
                or not (matched_text[cursor - 1].isalnum() or matched_text[cursor - 1] == "_")
            )
            and _DIRECT_SHELL_TRUE_VALUE.match(matched_text, cursor) is not None
        ):
            return cursor
        cursor += 1
    return None


_SCOPE_NODE_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
_COMPREHENSION_SCOPE_TYPES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_NATIVE_SHELL_RECEIVERS = frozenset({"subprocess", "Popen"})


def _paired_store_values(target: ast.AST, value: ast.AST) -> Iterator[tuple[ast.AST, ast.AST]]:
    """Yield each store target with the RHS element it receives, when the shapes match."""
    pairs = [(target, value)]
    while pairs:
        destination, source = pairs.pop()
        if (
            isinstance(destination, (ast.Tuple, ast.List))
            and isinstance(source, (ast.Tuple, ast.List))
            and len(destination.elts) == len(source.elts)
            and not any(isinstance(item, ast.Starred) for item in (*destination.elts, *source.elts))
        ):
            pairs.extend(zip(destination.elts, source.elts, strict=True))
        else:
            yield destination, source


def _captures_native_receiver(value: ast.AST, receiver: str) -> bool:
    """Return whether *value* reads the native *receiver* callable or module."""
    if isinstance(value, ast.Name):
        return value.id == receiver
    return (
        receiver == "Popen"
        and isinstance(value, ast.Attribute)
        and value.attr == "Popen"
        and isinstance(value.value, ast.Name)
        and value.value.id == "subprocess"
    )


def _native_receiver_aliases(tree: ast.Module) -> dict[str, str]:
    """Map each name whose every binding captures one native receiver to it."""
    paired: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                paired.update(_paired_store_values(target, node.value))
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            paired[node.target] = node.value
    captures: dict[str, set[str | None]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    continue
                native = (
                    "subprocess"
                    if isinstance(node, ast.Import) and alias.name == "subprocess"
                    else "Popen"
                    if isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "subprocess"
                    and alias.name == "Popen"
                    else None
                )
                bound = alias.asname or alias.name.split(".", 1)[0]
                captures.setdefault(bound, set()).add(native)
            continue
        name: str | None = None
        native = None
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            name = node.id
            value = paired.get(node) if isinstance(node.ctx, ast.Store) else None
            native = next(
                (
                    receiver
                    for receiver in sorted(_NATIVE_SHELL_RECEIVERS)
                    if value is not None and _captures_native_receiver(value, receiver)
                ),
                None,
            )
        elif isinstance(node, ast.arg):
            name = node.arg
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            name = node.name
        elif isinstance(node, ast.MatchMapping):
            name = node.rest
        if name is not None:
            captures.setdefault(name, set()).add(native)
    return {
        name: receiver
        for name, values in captures.items()
        if len(values) == 1 and (receiver := next(iter(values))) is not None
    }


def _store_keeps_shell_signal(name: str, value: ast.AST, aliases: dict[str, str]) -> bool | None:
    """Classify one simple-name store for lexical TM1 reconciliation.

    ``None`` means an identity store that leaves the binding unchanged. ``True``
    means the store keeps the native receiver or a truthy shell flag. ``False``
    means the store may replace the binding.
    """
    if isinstance(value, ast.Name) and value.id == name:
        return None
    if name in _NATIVE_SHELL_RECEIVERS:
        return _captures_native_receiver(value, name) or (
            isinstance(value, ast.Name) and aliases.get(value.id) == name
        )
    if name.casefold() in {"subprocess", "popen"}:
        return False
    return isinstance(value, ast.Constant) and bool(value.value)


@dataclass(frozen=True)
class _ScopedShellNode:
    """One relevant AST node plus its source and lexical-scope identity."""

    node: ast.Assign | ast.Call
    scope_chain: tuple[ast.AST, ...]
    start: int


@dataclass(frozen=True)
class _VariableShellAstIndex:
    """Linear-time index used to reconcile every bounded regex candidate."""

    tree: ast.Module
    line_character_starts: tuple[int, ...]
    assignments: dict[tuple[int, str], tuple[_ScopedShellNode, ...]]
    calls: dict[str, tuple[_ScopedShellNode, ...]]
    call_starts: dict[str, tuple[int, ...]]
    bindings: dict[ast.AST, frozenset[str]]
    class_binding_starts: dict[ast.ClassDef, dict[str, tuple[int, ...]]]
    declarations: dict[ast.AST, dict[str, str]]
    binding_events: dict[tuple[ast.AST, ast.AST], dict[str, tuple[tuple[int, bool], ...]]]
    eager_effect_starts: dict[ast.AST, tuple[int, ...]]
    reference_starts: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class _VariableShellCandidate:
    """One raw variable-shell regex candidate resolved against the shared AST."""

    name: str
    call: ast.Call
    same_scope: bool
    visible: bool
    assignment_start: int
    assignment_scope: ast.AST
    call_scope_chain: tuple[ast.AST, ...]


def _resolved_name_scope(
    index: _VariableShellAstIndex,
    use_chain: tuple[ast.AST, ...],
    name: str,
    use_start: int,
) -> ast.AST:
    """Return the Python scope that resolves *name* at one use site."""
    crossed_function = False
    nonlocal_lookup = False
    seen_class_scope = False
    for scope in reversed(use_chain):
        if scope is index.tree:
            return index.tree
        declaration = index.declarations.get(scope, {}).get(name)
        if declaration == "global":
            return index.tree
        if declaration == "nonlocal":
            nonlocal_lookup = True
            crossed_function = True
            continue
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if name in index.bindings.get(scope, frozenset()):
                return scope
            crossed_function = True
            continue
        if isinstance(scope, _COMPREHENSION_SCOPE_TYPES):
            if name in index.bindings.get(scope, frozenset()):
                return scope
            # Comprehensions use an implicit function scope.  Free names skip a
            # surrounding class namespace just like names in a method body.
            crossed_function = True
            continue
        if isinstance(scope, ast.ClassDef):
            # A method does not close over its class namespace.  A call evaluated
            # directly in the class body sees assignments already executed, but
            # a later class assignment does not create a compile-time local.
            starts = index.class_binding_starts.get(scope, {}).get(name, ())
            if (
                not crossed_function
                and not nonlocal_lookup
                and not seen_class_scope
                and bisect_left(starts, use_start)
            ):
                return scope
            seen_class_scope = True
    return index.tree


def _node_character_span(parsed: ParsedPythonFile, node: ast.AST) -> tuple[int, int] | None:
    """Return one AST node's absolute character span."""
    line = getattr(node, "lineno", None)
    end_line = getattr(node, "end_lineno", None)
    byte_column = getattr(node, "col_offset", None)
    end_byte_column = getattr(node, "end_col_offset", None)
    if not all(isinstance(value, int) for value in (line, end_line, byte_column, end_byte_column)):
        return None
    assert isinstance(line, int)
    assert isinstance(end_line, int)
    assert isinstance(byte_column, int)
    assert isinstance(end_byte_column, int)
    start_column = parsed.character_column(line, byte_column)
    end_column = parsed.character_column(end_line, end_byte_column)
    if start_column is None or end_column is None:
        return None
    start = parsed.line_character_starts[line - 1] + start_column
    end = parsed.line_character_starts[end_line - 1] + end_column
    return start, end


def _parameter_names(arguments: ast.arguments) -> tuple[str, ...]:
    names = [
        argument.arg
        for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
    ]
    if arguments.vararg is not None:
        names.append(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.append(arguments.kwarg.arg)
    return tuple(names)


def _direct_shell_name(call: ast.Call) -> str | None:
    function = call.func
    direct = isinstance(function, ast.Name) and function.id == "Popen"
    direct = direct or (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "subprocess"
    )
    if not direct:
        return None
    for keyword in call.keywords:
        if keyword.arg == "shell" and isinstance(keyword.value, ast.Name):
            return keyword.value.id
    return None


def _passive_assignment_name_snapshots(tree: ast.AST) -> dict[ast.AST, frozenset[str]]:
    """Prove prior simple-name bindings in supported straight-line scopes."""
    from . import static_python_shell_truthiness

    snapshots: dict[ast.AST, frozenset[str]] = {}
    for scope in ast.walk(tree):
        if not isinstance(scope, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        bound_names = (
            set(_parameter_names(scope.args)) if not isinstance(scope, ast.Module) else set()
        )
        declarations = static_python_shell_truthiness._DirectBindingCollector(frozenset())
        for statement in scope.body:
            declarations.visit(statement)
        bound_names.update(declarations.nonlocal_names)
        safe_names: set[str] = set()
        trusted_names = set(static_python_shell_truthiness._DIRECT_CALL_NAMES).difference(
            bound_names
        )
        unknown_effects = [False]
        for statement in scope.body:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)) and not unknown_effects[0]:
                targets = (
                    statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                )
                value = statement.value
                if value is not None:
                    pairs = [
                        pair for target in targets for pair in _paired_store_values(target, value)
                    ]
                    if pairs and all(
                        isinstance(target, ast.Name)
                        and (target.id not in bound_names or target.id in safe_names)
                        for target, _ in pairs
                    ):
                        snapshots[statement] = frozenset(
                            item.id
                            for item in ast.walk(value)
                            if isinstance(item, ast.Name) and item.id in safe_names
                        )
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                native_import = (
                    isinstance(statement, ast.Import)
                    and all(
                        alias.name == "subprocess" and alias.asname in (None, "subprocess")
                        for alias in statement.names
                    )
                    or isinstance(statement, ast.ImportFrom)
                    and statement.level == 0
                    and statement.module == "subprocess"
                    and all(
                        alias.name == "Popen" and alias.asname in (None, "Popen")
                        for alias in statement.names
                    )
                )
                if not native_import:
                    # Arbitrary module initialization can change existing names.
                    unknown_effects[0] = True
                    safe_names.clear()
            static_python_shell_truthiness._advance_trusted_names(
                statement, trusted_names, bound_names, safe_names, unknown_effects
            )
            if isinstance(statement, (ast.Import, ast.ImportFrom)) and not unknown_effects[0]:
                # Only a fresh, proven native import adds release-safe names.
                safe_names.update(
                    trusted_names.intersection(
                        static_python_shell_truthiness._direct_bound_names(statement)
                    )
                )
    return snapshots


def _build_variable_shell_ast_index(parsed: ParsedPythonFile) -> _VariableShellAstIndex | None:
    """Index relevant nodes, bindings, and declarations in one AST traversal."""
    tree = parsed.tree
    if tree is None:
        return None
    assignments: dict[tuple[int, str], list[_ScopedShellNode]] = {}
    calls: dict[str, list[_ScopedShellNode]] = {}
    bindings: dict[ast.AST, set[str]] = {tree: set()}
    class_binding_starts: dict[ast.ClassDef, dict[str, list[int]]] = {}
    declarations: dict[ast.AST, dict[str, str]] = {}
    binding_events: dict[ast.AST, dict[str, list[tuple[int, bool]]]] = {}
    eager_effect_starts: dict[ast.AST, list[int]] = {}
    reference_starts: dict[str, list[int]] = {}
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    attribute_identity_stores: set[ast.Attribute] = set()
    # Simple-name stores that keep the current binding, including unpacked
    # self-stores. They neither replace nor restore a binding.
    identity_name_stores: set[ast.Name] = set()
    native_aliases = _native_receiver_aliases(tree)
    passive_assignment_snapshots = _passive_assignment_name_snapshots(tree)

    def mark_attribute_identity_stores(target: ast.AST, value: ast.AST) -> None:
        for destination, source in _paired_store_values(target, value):
            if (
                isinstance(destination, ast.Attribute)
                and isinstance(destination.value, ast.Name)
                and isinstance(source, ast.Attribute)
                and isinstance(source.value, ast.Name)
                and destination.attr == source.attr
                and destination.value.id == source.value.id
            ):
                attribute_identity_stores.add(destination)

    def record_name_store_values(
        scope: ast.AST, targets: list[ast.AST], value: ast.AST, end: int
    ) -> None:
        from . import static_python_shell_truthiness

        # Pair unpacked targets with their RHS elements so a native alias, a
        # truthy flag, or a self-store is not mistaken for a replacement.
        pairs = [
            (destination, source)
            for target in targets
            for destination, source in reversed(list(_paired_store_values(target, value)))
        ]
        stores = [
            (destination, source)
            for destination, source in pairs
            if isinstance(destination, ast.Name)
        ]
        # The paired iterator is stack-based. Restore Python's left-to-right
        # target order before deciding which occurrence owns the final value.
        final_values = {destination.id: source for destination, source in stores}
        safe_names = passive_assignment_snapshots.get(node)
        safe_stores = safe_names is not None
        safe_value = (
            safe_names is not None
            and static_python_shell_truthiness._is_finalizer_safe_value(value, set(safe_names))
        )
        # Unsupported RHS effects or target protocols cannot prove the final
        # flag. Preserve the existing conservative event policy for those shapes.
        false_values = (
            {
                source
                for source in set(final_values.values())
                if static_python_shell_truthiness._truth_value(source, {}) is False
            }
            if safe_stores and safe_value
            else set()
        )
        definitely_false = {name for name, source in final_values.items() if source in false_values}
        for destination, source in stores:
            keeps_signal = _store_keeps_shell_signal(destination.id, source, native_aliases)
            if keeps_signal is None:
                identity_name_stores.add(destination)
            elif keeps_signal and destination.id not in definitely_false:
                record_binding(scope, destination.id, end, True)

    def record_binding(scope: ast.AST, name: str, start: int, restores: bool = False) -> None:
        # A conditional store is not affirmative replacement evidence. Only a
        # literal branch whose execution is proven can revoke the lexical owner.
        child = node
        if child in identity_name_stores:
            return
        parent = parents.get(child)
        if isinstance(parent, ast.AnnAssign) and parent.value is None and child is parent.target:
            return
        while parent is not None and parent is not scope:
            if isinstance(parent, ast.If):
                if not isinstance(parent.test, ast.Constant):
                    return
                taken = parent.body if bool(parent.test.value) else parent.orelse
                if child not in taken:
                    return
            elif isinstance(parent, ast.BoolOp):
                # Later operands run only after every earlier operand lets the
                # short circuit continue. Only constant operands prove that.
                continues = isinstance(parent.op, ast.And)
                for operand in parent.values:
                    if operand is child:
                        break
                    if (
                        not isinstance(operand, ast.Constant)
                        or bool(operand.value) is not continues
                    ):
                        return
            elif isinstance(parent, ast.IfExp) and child is not parent.test:
                if not isinstance(parent.test, ast.Constant):
                    return
                if child is not (parent.body if bool(parent.test.value) else parent.orelse):
                    return
            elif isinstance(
                parent,
                (
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    ast.With,
                    ast.AsyncWith,
                    ast.Try,
                    ast.TryStar,
                    ast.Match,
                    ast.comprehension,
                    *_COMPREHENSION_SCOPE_TYPES,
                ),
            ):
                return
            child, parent = parent, parents.get(parent)
        binding_events.setdefault(scope, {}).setdefault(name, []).append((start, restores))

    def bind(scope: ast.AST, name: str, start: int) -> None:
        bindings.setdefault(scope, set()).add(name)
        record_binding(scope, name, start)
        if isinstance(scope, ast.ClassDef):
            class_binding_starts.setdefault(scope, {}).setdefault(name, []).append(start)

    stack: list[tuple[ast.AST, tuple[ast.AST, ...], int | None]] = [(tree, (tree,), None)]
    while stack:
        node, scope_chain, binding_start = stack.pop()
        scope = scope_chain[-1]
        span = _node_character_span(parsed, node)
        start = span[0] if span is not None else 0
        end = span[1] if span is not None else start
        # A direct observed slot write is useful only before another eager
        # effect. Cached-module lifetime across imports belongs to the companion.
        if isinstance(
            node,
            (
                ast.Call,
                ast.Attribute,
                ast.Subscript,
                ast.BinOp,
                ast.BoolOp,
                ast.Compare,
                ast.UnaryOp,
                ast.NamedExpr,
                ast.JoinedStr,
                ast.Import,
                ast.ImportFrom,
                ast.Delete,
                ast.AugAssign,
                ast.If,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.With,
                ast.AsyncWith,
                ast.Try,
                ast.TryStar,
                ast.Match,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
                *_COMPREHENSION_SCOPE_TYPES,
            ),
        ):
            execution = next(
                (
                    item
                    for item in reversed(scope_chain)
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
                ),
                tree,
            )
            eager_effect_starts.setdefault(execution, []).append(start)
        # Any later reference may invoke a deferred body, so keep every load.
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            reference_starts.setdefault(node.id, []).append(start)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            reference_starts.setdefault(node.attr, []).append(start)

        if isinstance(node, ast.Assign):
            for target in node.targets:
                mark_attribute_identity_stores(target, node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            mark_attribute_identity_stores(node.target, node.value)
        if isinstance(node, ast.Assign):
            # The RHS is captured before any target is assigned. A final
            # same-slot store restores that snapshot after earlier writes in
            # this statement, including chained and repeated unpacking targets.
            stores: dict[tuple[str, str], list[ast.Attribute]] = {}
            targets = list(reversed(node.targets))
            while targets:
                target = targets.pop()
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    stores.setdefault((target.value.id, target.attr), []).append(target)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    targets.extend(reversed(target.elts))
            for slot_stores in stores.values():
                if slot_stores[-1] in attribute_identity_stores:
                    attribute_identity_stores.update(slot_stores)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bind(scope, node.name, end)
            nested_chain = (*scope_chain, node)
            bindings.setdefault(node, set()).update(_parameter_names(node.args))
            header_nodes: list[ast.AST] = [*node.decorator_list, node.args]
            if node.returns is not None:
                header_nodes.append(node.returns)
            header_nodes.extend(getattr(node, "type_params", []))
            stack.extend((child, scope_chain, None) for child in header_nodes)
            stack.extend((child, nested_chain, None) for child in node.body)
            continue
        if isinstance(node, ast.Lambda):
            nested_chain = (*scope_chain, node)
            bindings.setdefault(node, set()).update(_parameter_names(node.args))
            stack.append((node.args, scope_chain, None))
            stack.append((node.body, nested_chain, None))
            continue
        if isinstance(node, ast.ClassDef):
            bind(scope, node.name, end)
            nested_chain = (*scope_chain, node)
            bindings.setdefault(node, set())
            header_nodes = [*node.decorator_list, *node.bases]
            header_nodes.extend(keyword.value for keyword in node.keywords)
            header_nodes.extend(getattr(node, "type_params", []))
            stack.extend((child, scope_chain, None) for child in header_nodes)
            stack.extend((child, nested_chain, None) for child in node.body)
            continue
        if isinstance(node, _COMPREHENSION_SCOPE_TYPES):
            nested_chain = (*scope_chain, node)
            bindings.setdefault(node, set())
            generators = node.generators
            if generators:
                first, *remaining = generators
                # The leftmost iterable is the only comprehension expression
                # evaluated in the enclosing scope.
                stack.append((first.iter, scope_chain, None))
                stack.append((first.target, nested_chain, None))
                stack.extend((condition, nested_chain, None) for condition in first.ifs)
                for generator in remaining:
                    stack.append((generator.iter, nested_chain, None))
                    stack.append((generator.target, nested_chain, None))
                    stack.extend((condition, nested_chain, None) for condition in generator.ifs)
            if isinstance(node, ast.DictComp):
                stack.append((node.key, nested_chain, None))
                stack.append((node.value, nested_chain, None))
            else:
                stack.append((node.elt, nested_chain, None))
            continue

        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
        ):
            assignment = parents.get(node)
            if (
                node not in attribute_identity_stores
                and isinstance(assignment, ast.Assign)
                and len(assignment.targets) == 1
                and assignment.targets[0] is node
                and tree.body
                and assignment is tree.body[0]
                and not any(
                    isinstance(value, ast.Name)
                    and isinstance(value.ctx, ast.Load)
                    and value.id in {node.value.id, "Popen"}
                    for value in ast.walk(assignment.value)
                )
            ):
                # Retain the legacy first-statement direct shadow control only.
                # An RHS using the receiver or Popen may preserve a native callable.
                # Prior stores/protocol effects need the companion's explicit proof.
                record_binding(
                    scope,
                    node.value.id + "." + node.attr,
                    binding_start if binding_start is not None else end,
                )

        if isinstance(node, ast.Global):
            for name in node.names:
                declarations.setdefault(scope, {})[name] = "global"
        elif isinstance(node, ast.Nonlocal):
            for name in node.names:
                declarations.setdefault(scope, {})[name] = "nonlocal"
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bind(scope, node.id, binding_start if binding_start is not None else start)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name != "*":
                    bound_name = alias.asname or alias.name.split(".", 1)[0]
                    bind(scope, bound_name, end)
                    restores = (
                        isinstance(node, ast.Import)
                        and alias.name == "subprocess"
                        and bound_name == "subprocess"
                    ) or (
                        isinstance(node, ast.ImportFrom)
                        and node.level == 0
                        and node.module == "subprocess"
                        and alias.name == "Popen"
                        and bound_name == "Popen"
                    )
                    if restores:
                        record_binding(scope, bound_name, end, True)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            type_span = _node_character_span(parsed, node.type) if node.type is not None else None
            bind(scope, node.name, type_span[1] if type_span is not None else start)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bind(scope, node.name, start)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bind(scope, node.rest, start)

        if (
            isinstance(node, ast.Assign)
            and span is not None
            and isinstance(node.value, ast.Constant)
            and node.value.value is True
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    scoped = _ScopedShellNode(node=node, scope_chain=scope_chain, start=start)
                    assignments.setdefault((node.lineno, target.id), []).append(scoped)
        elif isinstance(node, ast.Call) and span is not None:
            shell_name = _direct_shell_name(node)
            if shell_name is not None:
                scoped = _ScopedShellNode(node=node, scope_chain=scope_chain, start=start)
                calls.setdefault(shell_name, []).append(scoped)

        if isinstance(node, ast.Assign):
            stack.append((node.value, scope_chain, None))
            stack.extend((target, scope_chain, end) for target in node.targets)
            record_name_store_values(scope, list(node.targets), node.value, end)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if node.value is not None:
                stack.append((node.value, scope_chain, None))
            stack.append((node.target, scope_chain, end))
            if isinstance(node, ast.AnnAssign):
                stack.append((node.annotation, scope_chain, None))
                if node.value is not None:
                    record_name_store_values(scope, [node.target], node.value, end)
        elif isinstance(node, ast.NamedExpr):
            stack.append((node.value, scope_chain, None))
            # PEP 572 makes a walrus target inside one or more
            # comprehensions local to the nearest containing real scope.
            # Comprehension ``for`` targets remain local to their implicit
            # scope, but the named-expression target skips those scopes.
            target_chain = scope_chain
            while len(target_chain) > 1 and isinstance(
                target_chain[-1], _COMPREHENSION_SCOPE_TYPES
            ):
                target_chain = target_chain[:-1]
            stack.append((node.target, target_chain, end))
            record_name_store_values(target_chain[-1], [node.target], node.value, end)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            iteration_span = _node_character_span(parsed, node.iter)
            iteration_end = iteration_span[1] if iteration_span is not None else end
            stack.append((node.iter, scope_chain, None))
            stack.append((node.target, scope_chain, iteration_end))
            stack.extend((child, scope_chain, None) for child in (*node.body, *node.orelse))
        elif isinstance(node, ast.withitem):
            context_span = _node_character_span(parsed, node.context_expr)
            context_end = context_span[1] if context_span is not None else end
            stack.append((node.context_expr, scope_chain, None))
            if node.optional_vars is not None:
                stack.append((node.optional_vars, scope_chain, context_end))
        else:
            stack.extend(
                (child, scope_chain, binding_start) for child in ast.iter_child_nodes(node)
            )

    def execution_scope(scope: ast.AST) -> ast.AST:
        current: ast.AST | None = scope
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                return current
            current = parents.get(current)
        return tree

    # Keep the execution scope of every store. A global/nonlocal declaration
    # changes the binding namespace; it never proves a deferred body was run.
    scoped_events: dict[tuple[ast.AST, ast.AST], dict[str, list[tuple[int, bool]]]] = {}
    sorted_class_starts = {
        scope: {name: tuple(sorted(starts)) for name, starts in names.items()}
        for scope, names in class_binding_starts.items()
    }
    for scope, names in binding_events.items():
        for name, events in names.items():
            receiver_name = name.partition(".")[0]
            declaration = declarations.get(scope, {}).get(receiver_name)
            for event in events:
                target_scope = scope
                if "." in name and declaration is None:
                    # A class body can change where its receiver resolves
                    # between stores. Resolve each slot store at its own offset.
                    current: ast.AST | None = scope
                    crossed_function = False
                    while current is not None:
                        if current is tree:
                            target_scope = tree
                            break
                        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                            if receiver_name in bindings.get(current, set()):
                                target_scope = current
                                break
                            crossed_function = True
                        elif isinstance(current, ast.ClassDef) and not crossed_function:
                            starts = sorted_class_starts.get(current, {}).get(receiver_name, ())
                            if bisect_right(starts, event[0]):
                                target_scope = current
                                break
                        current = parents.get(current)
                if declaration == "global":
                    target_scope = tree
                elif declaration == "nonlocal":
                    outer = parents.get(scope)
                    while outer is not None:
                        if isinstance(
                            outer, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
                        ) and receiver_name in bindings.get(outer, set()):
                            target_scope = outer
                            break
                        outer = parents.get(outer)
                key = (target_scope, execution_scope(scope))
                scoped_events.setdefault(key, {}).setdefault(name, []).append(event)

    frozen_calls = {
        name: tuple(sorted(nodes, key=lambda item: item.start)) for name, nodes in calls.items()
    }
    return _VariableShellAstIndex(
        tree=tree,
        line_character_starts=parsed.line_character_starts,
        assignments={key: tuple(nodes) for key, nodes in assignments.items()},
        calls=frozen_calls,
        call_starts={
            name: tuple(item.start for item in nodes) for name, nodes in frozen_calls.items()
        },
        bindings={scope: frozenset(names) for scope, names in bindings.items()},
        class_binding_starts=sorted_class_starts,
        declarations=declarations,
        eager_effect_starts={
            scope: tuple(sorted(set(starts))) for scope, starts in eager_effect_starts.items()
        },
        reference_starts={name: tuple(sorted(starts)) for name, starts in reference_starts.items()},
        binding_events={
            scope: {name: tuple(sorted(set(events))) for name, events in names.items()}
            for scope, names in scoped_events.items()
        },
    )


def _assignment_name_scope(
    index: _VariableShellAstIndex,
    scope_chain: tuple[ast.AST, ...],
    name: str,
) -> ast.AST:
    """Return the scope mutated by a simple-name assignment."""
    scope = scope_chain[-1]
    declaration = index.declarations.get(scope, {}).get(name)
    if declaration == "global":
        return index.tree
    if declaration == "nonlocal":
        for outer in reversed(scope_chain[:-1]):
            if isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and name in (
                index.bindings.get(outer, frozenset())
            ):
                return outer
    return scope


def _variable_shell_call_start(match: re.Match[str]) -> int | None:
    """Return the final call selected by one bounded variable-shell match."""
    callees = tuple(re.finditer(r"(?:subprocess\.\w+|Popen)\s*\(", match.group(0), re.IGNORECASE))
    return match.start() + callees[-1].start() if callees else None


def _is_true_prefixed_name(name: str) -> bool:
    """Return whether a variable window keeps a call-anchored fallback for a name."""
    return name.casefold().startswith("true")


def _variable_shell_matches(content: str) -> list[re.Match[str]]:
    """Return bounded variable-shell windows, one per call for true-prefixed names.

    The non-overlapping scan pairs an assignment with one call and skips an
    assignment inside an earlier window. The ``\\b``-bounded direct pattern
    does not report ``shell=true_value`` and folds identical same-line
    ``shell=true`` calls into one candidate, so every assignment of a
    true-prefixed name is also paired with each call its bounded window
    reaches. The nearest assignment owns each call.
    """
    windows: dict[int, re.Match[str]] = {}

    def keep(window: re.Match[str]) -> None:
        previous = windows.get(window.end())
        if previous is None or window.start() > previous.start():
            windows[window.end()] = window

    for match in _VARIABLE_SHELL_FLAG_RE.finditer(content):
        keep(match)
    for assignment in _VARIABLE_SHELL_ASSIGNMENT_RE.finditer(content):
        if not _is_true_prefixed_name(assignment.group(1)):
            continue
        search_end = assignment.end() - 1
        for _ in range(5):
            search_end = content.find("\n", search_end + 1)
            if search_end == -1:
                search_end = len(content)
                break
        for argument in _SHELL_NAME_ARGUMENT_RE.finditer(content, assignment.end(), search_end):
            window = _VARIABLE_SHELL_FLAG_RE.fullmatch(content, assignment.start(), argument.end())
            if window is not None:
                keep(window)
    return sorted(windows.values(), key=lambda window: (window.start(), window.end()))


def _resolve_variable_shell_candidate(
    index: _VariableShellAstIndex,
    match: re.Match[str],
    assignment_line: int,
) -> _VariableShellCandidate | None:
    """Resolve a bounded regex match to its exact assignment, call, and binding."""
    name = match.group(1)
    assignment_nodes = [
        item
        for item in index.assignments.get((assignment_line, name), ())
        if match.start() <= item.start < match.end()
    ]
    call_nodes = index.calls.get(name, ())
    starts = index.call_starts.get(name, ())
    expected_call_start = _variable_shell_call_start(match)
    if expected_call_start is None:
        return None
    call_index = bisect_left(starts, expected_call_start)
    if (
        len(assignment_nodes) != 1
        or call_index >= len(call_nodes)
        or call_nodes[call_index].start != expected_call_start
    ):
        return None
    assignment = assignment_nodes[0]
    call = call_nodes[call_index]
    if call.start >= match.end():
        return None
    assignment_scope = _assignment_name_scope(index, assignment.scope_chain, name)
    resolved_scope = _resolved_name_scope(index, call.scope_chain, name, call.start)
    return _VariableShellCandidate(
        name=name,
        call=call.node,
        same_scope=assignment.scope_chain == call.scope_chain,
        visible=resolved_scope is assignment_scope,
        assignment_start=assignment.start,
        assignment_scope=assignment_scope,
        call_scope_chain=call.scope_chain,
    )


def _earliest_deferred_invocation(
    index: _VariableShellAstIndex,
    parsed: ParsedPythonFile,
    call_scope_chain: tuple[ast.AST, ...],
    owner_execution: ast.AST,
) -> float:
    """Return the earliest owner-scope offset at which a deferred call may run.

    A function, method, or lambda can run once it exists and something refers to
    it: a call in any statement (including ``if``/``try``/main guards), a callback
    registration, a decorator, or another body. Without a visible reference it
    runs only after the owner scope finishes.
    """
    owner_position = next(
        (position for position, scope in enumerate(call_scope_chain) if scope is owner_execution),
        None,
    )
    if owner_position is None or owner_position + 1 >= len(call_scope_chain):
        return 0
    entries = call_scope_chain[owner_position + 1 :]
    span = _node_character_span(parsed, entries[0])
    if span is None:
        return 0
    defined = span[1]
    if any(
        isinstance(entry, (ast.Lambda, ast.GeneratorExp)) or getattr(entry, "decorator_list", None)
        for entry in entries
    ):
        return defined
    names = {
        entry.name
        for entry in entries
        if isinstance(entry, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    references = [
        start
        for name in names
        for start in index.reference_starts.get(name, ())
        if not span[0] <= start < span[1]
    ]
    if not references:
        return float("inf")
    return max(defined, min(references))


def _deferred_receiver_is_replaced(
    index: _VariableShellAstIndex,
    parsed: ParsedPythonFile,
    candidate: _VariableShellCandidate,
    receiver: str,
    receiver_scope: ast.AST,
    receiver_execution: ast.AST,
    current_execution: ast.AST,
    call_start: int,
) -> bool:
    """Return whether every possible invocation of a deferred call sees a replacement."""
    local_events = index.binding_events.get((receiver_scope, current_execution), {}).get(
        receiver, ()
    )
    local_index = bisect_right(local_events, (call_start, True))
    if local_index:
        # A store earlier in the same invocation decides the receiver.
        return not local_events[local_index - 1][1]
    outer_events = index.binding_events.get((receiver_scope, receiver_execution), {}).get(
        receiver, ()
    )
    first_invocation = _earliest_deferred_invocation(
        index, parsed, candidate.call_scope_chain, receiver_execution
    )
    keeps_receiver: dict[int, bool] = {}
    for start, restores in outer_events:
        keeps_receiver[start] = keeps_receiver.get(start, False) or restores
    starts = sorted(keeps_receiver)
    earlier = bisect_left(starts, first_invocation)
    if not earlier:
        return False
    # The binding seen by the first possible invocation, then every later one.
    return not any(keeps_receiver[start] for start in starts[earlier - 1 :])


def _lexical_shell_has_counterevidence(
    index: _VariableShellAstIndex,
    parsed: ParsedPythonFile,
    candidate: _VariableShellCandidate,
    *,
    receiver_trusted: bool | None,
) -> bool:
    """Reject a lexical candidate only for an observed binding replacement.

    The companion's unknown receiver trust is an abstention. Generic calls,
    protocol effects, and unsupported statements do not prove a replacement.
    Stores/imports do; use their source order and Python binding scopes.
    """
    call_span = _node_character_span(parsed, candidate.call)
    if call_span is None:
        return False
    call_start = call_span[0]
    shell = next((item.value for item in candidate.call.keywords if item.arg == "shell"), None)
    shell_span = _node_character_span(parsed, shell) if shell is not None else None
    shell_start = shell_span[0] if shell_span is not None else call_start
    current_execution = next(
        (
            scope
            for scope in reversed(candidate.call_scope_chain)
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        ),
        index.tree,
    )

    def binding_events(scope: ast.AST, name: str) -> tuple[tuple[int, bool], ...]:
        owner_execution = (
            scope
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            else index.tree
        )
        events = index.binding_events.get((scope, owner_execution), {}).get(name, ())
        if current_execution is not owner_execution:
            local_events = index.binding_events.get((scope, current_execution), {}).get(name, ())
            return tuple(sorted((*events, *local_events)))
        return events

    flag_events = binding_events(candidate.assignment_scope, candidate.name)
    flag_index = bisect_right(flag_events, (shell_start, True))
    if flag_index:
        last_flag = flag_events[flag_index - 1]
        if last_flag[0] > candidate.assignment_start and not last_flag[1]:
            return True
    if receiver_trusted is True:
        return False
    function = candidate.call.func
    receiver = function.id if isinstance(function, ast.Name) else function.value.id
    receiver_scope = _resolved_name_scope(index, candidate.call_scope_chain, receiver, call_start)
    receiver_execution = (
        receiver_scope
        if isinstance(receiver_scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        else index.tree
    )
    # Called-module slot replacement is proved only by the companion cache
    # state, which discards stale proof after an unknown eager effect.
    receiver_events = binding_events(receiver_scope, receiver)
    receiver_index = bisect_right(receiver_events, (call_start, True))
    if receiver_execution is not current_execution:
        # Source order does not order a deferred body against outer stores.
        # Require a replacement at every invocation the outer scope can reach.
        if _deferred_receiver_is_replaced(
            index,
            parsed,
            candidate,
            receiver,
            receiver_scope,
            receiver_execution,
            current_execution,
            call_start,
        ):
            return True
    elif receiver_index:
        if not receiver_events[receiver_index - 1][1]:
            return True
    elif isinstance(receiver_scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        # Compile-time local bindings shadow a global receiver even before store.
        if receiver in index.bindings.get(receiver_scope, frozenset()):
            return True
    if isinstance(function, ast.Attribute):
        # Preserve a direct same-name replacement only in this execution scope,
        # after the completed store and before another effect or receiver binding.
        method_events = index.binding_events.get((receiver_scope, current_execution), {}).get(
            receiver + "." + function.attr, ()
        )
        method_index = bisect_right(method_events, (call_start, True))
        if method_index:
            method_start, restores = method_events[method_index - 1]
            effects = index.eager_effect_starts.get(current_execution, ())
            effect_index = bisect_right(effects, method_start)
            later_binding = receiver_index and receiver_events[receiver_index - 1][0] > method_start
            if (
                not restores
                and not later_binding
                and (effect_index == len(effects) or effects[effect_index] >= call_start)
            ):
                return True
    return False


def _destructive_command_candidates(
    candidate_content: str, *, check_runtime: Callable[[], None] | None
) -> Iterator[tuple[int, int, str, float, bool, int | None, int | None, str | None]]:
    seen_commands: set[tuple[int, int]] = set()
    covered_until = 0
    for command_start, body_start in _destructive_command_words(
        candidate_content, check_runtime=check_runtime
    ):
        if check_runtime is not None:
            check_runtime()
        if command_start < covered_until:
            continue
        command_key = (command_start, body_start)
        if command_key in seen_commands:
            continue
        seen_commands.add(command_key)
        # Documentation is excluded regardless of the shell parse result. Apply
        # that existing semantic gate first so large manuals do not pay for a
        # character-by-character shell parse for every explanatory ``rm`` noun.
        if _is_root_glob_documentation(candidate_content, command_start, body_start):
            continue
        tokens, command_end, exhausted = _bounded_shell_tokens(
            candidate_content,
            command_start,
            body_start,
            check_runtime=check_runtime,
        )
        # An unresolved quote cannot own and hide later executable candidates.
        if not exhausted:
            covered_until = max(covered_until, command_end)
        command = candidate_content[command_start:command_end]
        if _has_destructive_root_glob(tokens) or _has_destructive_root_path(tokens):
            yield command_start, command_end, command, 0.9, False, None, None, None


def _has_tm1_code_pattern_at(
    content: str, start: int, end: int, check_runtime: Callable[[], None]
) -> bool:
    """Check existing raw patterns at one bounded, proven command position."""
    for pattern, _ in TM1_CODE_PATTERNS:
        check_runtime()
        if re.compile(pattern, re.IGNORECASE | re.MULTILINE).match(content, start, end):
            return True
    return False


def _option_command_end(
    content: str,
    start: int,
    limit: int,
    check_runtime: Callable[[], None],
    *,
    powershell: bool = False,
) -> int:
    """Stop at an unquoted command separator, retaining balanced arguments."""
    quote: str | None = None
    ansi_c_quote = False
    closers: list[str] = []
    backtick_ends: dict[int, int | None] = {}
    i = start
    next_runtime_check = start
    while i < limit:
        if i >= next_runtime_check:
            check_runtime()
            next_runtime_check = i + 1024
        ch = content[i]
        if quote != "'" and content.startswith("${", i):
            inherited_quote_closed = [False]
            expansion_end = _skip_parameter_expansion(
                content,
                i,
                limit,
                check_runtime,
                backtick_end_cache=backtick_ends,
                inherited_double_quote=quote == '"',
                inherited_quote_closed=inherited_quote_closed,
            )
            if expansion_end is None:
                # An unresolved expansion cannot prove a shell separator.
                # The owning Markdown region still bounds this candidate.
                return limit
            if inherited_quote_closed[0]:
                quote = None
            i = expansion_end
            continue
        if quote != "'" and (
            content.startswith("$(", i) or quote is None and content.startswith(("<(", ">("), i)
        ):
            substitution_end = _skip_command_substitution(
                content, i, limit, check_runtime, backtick_end_cache=backtick_ends
            )
            if substitution_end is None:
                return limit
            i = substitution_end
            continue
        if quote is not None:
            if ch == quote:
                quote = None
                ansi_c_quote = False
            elif (quote != "'" or ansi_c_quote) and ch == "\\":
                i += 2 if content[i + 1 : i + 3] != "\r\n" else 3
                continue
            elif quote == '"' and ch == "`":
                substitution_end = _skip_backtick_substitution(
                    content, i, limit, check_runtime, end_cache=backtick_ends
                )
                if substitution_end is not None:
                    i = substitution_end
                else:
                    # A PowerShell backtick can escape a quote. A balanced
                    # Bash substitution above owns both of its delimiters.
                    i += 3 if content[i + 1 : i + 3] == "\r\n" else 2
                continue
        elif content.startswith("$'", i):
            quote = "'"
            ansi_c_quote = True
            i += 2
            continue
        elif ch in "\\`":
            # Backticks are also Markdown delimiters. Only a line-ending
            # backtick proves a continuation without relying on fence labels.
            if content[i + 1 : i + 3] == "\r\n":
                i += 3
                continue
            if content[i + 1 : i + 2] == "\n" or ch == "\\":
                i += 2
                continue
            substitution_end = _skip_backtick_substitution(
                content, i, limit, check_runtime, end_cache=backtick_ends
            )
            if substitution_end is not None:
                i = substitution_end
                continue
            # An unpaired PowerShell backtick escapes the next character,
            # including a separator that is literal data in this command.
            i += 2
            continue
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            closers.append(")")
        elif powershell and ch == "{":
            # Hashtable and script-block arguments own their inner separators.
            # Bash braces do not establish the same argument boundary.
            closers.append("}")
        elif closers and ch == closers[-1]:
            closers.pop()
        elif not closers and ch in "\r\n;|&":
            break
        i += 1
    return min(i, limit)


def _option_command_matches(
    pattern: str,
    content: str,
    check_runtime: Callable[[], None],
    markdown_fences: list[int],
    host_ranges: list[tuple[int, int]],
    powershell_ranges: list[tuple[int, int]],
) -> Iterator[re.Match[str]]:
    """Apply an unsafe-option pattern only within each candidate command."""
    compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    host_compiled = re.compile(_OPTION_COMMAND_HOST_PATTERNS[pattern], re.IGNORECASE | re.MULTILINE)
    host_starts = [start for start, _ in host_ranges]
    command_prefix = _OPTION_COMMAND_PREFIXES[pattern]
    search_prefix = (
        re.compile(
            command_prefix.pattern.replace(
                _OPTION_COMMAND_SPACE, rf"(?:{_OPTION_COMMAND_SPACE}|\s+)"
            ),
            re.IGNORECASE,
        )
        if host_ranges
        else command_prefix
    )
    powershell_starts = [start for start, _ in powershell_ranges]
    covered_until = 0
    for prefix in search_prefix.finditer(content):
        check_runtime()
        if prefix.start() < covered_until:
            continue
        host_index = bisect_right(host_starts, prefix.start()) - 1
        if host_index >= 0 and prefix.start() < host_ranges[host_index][1]:
            match = host_compiled.match(content, prefix.start(), host_ranges[host_index][1])
            if match is not None:
                covered_until = match.end()
                yield match
            continue
        fence_index = bisect_right(markdown_fences, prefix.start())
        limit = markdown_fences[fence_index] if fence_index < len(markdown_fences) else len(content)
        shell_prefix = command_prefix.match(content, prefix.start(), limit)
        if shell_prefix is None:
            continue
        powershell_index = bisect_right(powershell_starts, prefix.start()) - 1
        end = _option_command_end(
            content,
            shell_prefix.end(),
            limit,
            check_runtime,
            powershell=(
                powershell_index >= 0 and prefix.start() < powershell_ranges[powershell_index][1]
            ),
        )
        covered_until = end
        match = compiled.match(content, prefix.start(), end)
        if match is not None:
            yield match


def _tm1_candidates(
    content: str,
    *,
    direct_shell_only: bool = False,
    shell_content: str | None = None,
    check_runtime: Callable[[], None] | None,
    file_type: str = "",
) -> Iterator[tuple[int, int, str, float, bool, int | None, int | None, str | None]]:
    runtime_check = check_runtime or (lambda: None)

    def direct_matches(
        pattern_index: int,
        pattern: str,
        confidence: float,
    ) -> Iterator[tuple[int, int, re.Match[str], float]]:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            yield match.start(), pattern_index, match, confidence

    def direct_candidates() -> Iterator[
        tuple[int, int, str, float, bool, int | None, int | None, str | None]
    ]:
        direct_iterators = (
            direct_matches(pattern_index, pattern, confidence)
            for pattern_index, (pattern, confidence) in enumerate(DIRECT_SHELL_TRUE_PATTERNS)
        )
        qualified_popen_starts: set[int] = set()
        for _, pattern_index, match, confidence in heapq.merge(
            *direct_iterators,
            key=lambda candidate: (candidate[0], candidate[1]),
        ):
            alternate_start: int | None = None
            alternate_matched_text: str | None = None
            if pattern_index == 0:
                method = re.match(r"subprocess\.(?P<method>\w+)", match.group(0), re.IGNORECASE)
                if method is not None and method.group("method").casefold() == "popen":
                    alternate_start = match.start() + method.start("method")
                    alternate_matched_text = content[alternate_start : match.end()]
                    qualified_popen_starts.add(alternate_start)
            elif match.start() in qualified_popen_starts:
                # Emit one candidate for a qualified ``subprocess.Popen`` call,
                # but retain the bare-method coordinate as an ownership fallback
                # when the qualifier belongs to an adjacent scan window.
                qualified_popen_starts.remove(match.start())
                continue
            elif (
                qualifier_start := _subprocess_qualifier_start(content, match.start())
            ) is not None:
                alternate_start = qualifier_start
                alternate_matched_text = content[qualifier_start : match.end()]
            relative_anchor = _outer_shell_true_anchor(match.group(0))
            if relative_anchor is None:
                for shell_value in _DIRECT_SHELL_TRUE_VALUE.finditer(match.group(0)):
                    relative_anchor = shell_value.start()
            anchor = match.start() + relative_anchor if relative_anchor is not None else None
            yield (
                match.start(),
                match.end(),
                match.group(0),
                confidence,
                True,
                anchor,
                alternate_start,
                alternate_matched_text,
            )

    if direct_shell_only:
        yield from direct_candidates()
        return

    option_markdown_fences: list[int] = []
    option_fence_regions: list[tuple[int, int, str]] = []
    option_host_ranges: list[tuple[int, int]] = []
    if file_type == "markdown" and any(
        prefix.search(content)
        or re.search(prefix.pattern.replace(_OPTION_COMMAND_SPACE, r"\s+"), content, re.IGNORECASE)
        for prefix in _OPTION_COMMAND_PREFIXES.values()
    ):
        _markdown_shell_text(
            content,
            runtime_check,
            fence_boundaries=option_markdown_fences,
            fence_regions=option_fence_regions,
            inline_regions=option_host_ranges,
        )
    option_host_ranges.extend(
        (start, end)
        for start, end, language in option_fence_regions
        if language not in _OPTION_SHELL_FENCE_LANGUAGES
    )
    option_host_ranges.sort()
    option_powershell_ranges = (
        [(0, len(content))]
        if file_type == "powershell"
        else [
            (start, end)
            for start, end, language in option_fence_regions
            if language in {"powershell", "pwsh", "ps1"}
        ]
    )

    def ordinary_matches(
        pattern_index: int,
        pattern: str,
        confidence: float,
    ) -> Iterator[tuple[int, int, re.Match[str], float]]:
        if pattern in _OPTION_COMMAND_PREFIXES and file_type in {"markdown", "shell", "powershell"}:
            for match in _option_command_matches(
                pattern,
                content,
                runtime_check,
                option_markdown_fences,
                option_host_ranges,
                option_powershell_ranges,
            ):
                yield match.start(), pattern_index, match, confidence
            return
        pattern = _OPTION_COMMAND_HOST_PATTERNS.get(pattern, pattern)
        if pattern == _VARIABLE_SHELL_FLAG_PATTERN:
            for match in _variable_shell_matches(content):
                yield match.start(), pattern_index, match, confidence
            return
        matches = (
            static_runner.iter_paragraph_matches
            if (pattern, confidence) in TM1_PROSE_PATTERNS
            else re.finditer
        )
        for match in matches(pattern, content, re.IGNORECASE | re.MULTILINE):
            yield match.start(), pattern_index, match, confidence

    def ordinary_candidates() -> Iterator[
        tuple[int, int, str, float, bool, int | None, int | None, str | None]
    ]:
        ordinary_iterators = (
            ordinary_matches(pattern_index, pattern, confidence)
            for pattern_index, (pattern, confidence) in enumerate(
                TM1_PATTERNS[len(DIRECT_SHELL_TRUE_PATTERNS) :]
            )
        )
        for _, _, match, confidence in heapq.merge(
            *ordinary_iterators,
            key=lambda candidate: (candidate[0], candidate[1]),
        ):
            yield (
                match.start(),
                match.end(),
                match.group(0),
                confidence,
                False,
                None,
                None,
                None,
            )

    destructive_iterators = [_destructive_command_candidates(content, check_runtime=check_runtime)]
    if shell_content is not None and shell_content != content:
        destructive_iterators.append(
            _destructive_command_candidates(shell_content, check_runtime=check_runtime)
        )
    yield from heapq.merge(
        direct_candidates(),
        ordinary_candidates(),
        *destructive_iterators,
        key=lambda candidate: candidate[0],
    )


def _markdown_block_separator(line: str, check_runtime: Callable[[], None]) -> bool:
    """Recognize Setext underlines/thematic breaks with bounded, linear work."""
    line = line.strip(" \t")
    marker = line[:1]
    if marker not in {"=", "-", "*", "_"}:
        return False
    count = 0
    internal_gap = False
    for index, character in enumerate(line):
        if index % 256 == 0:
            check_runtime()
        if character == marker:
            count += 1
        elif character in " \t" and marker != "=":
            internal_gap = True
        else:
            return False
    return count >= (3 if internal_gap or marker in {"*", "_"} else 1)


def _markdown_table_cells(
    line: str, start: int, check_runtime: Callable[[], None]
) -> list[tuple[int, int]]:
    """Locate GFM cells without copying content or changing source offsets."""
    end = len(line)
    while start < end and line[start] in " \t":
        if start % 256 == 0:
            check_runtime()
        start += 1
    while end > start and line[end - 1] in " \t":
        if end % 256 == 0:
            check_runtime()
        end -= 1
    if start == end:
        return []
    # Edge pipes are optional. In cmark-gfm a backslash immediately before a
    # pipe escapes it even when that backslash follows another backslash.
    if line[start] == "|":
        start += 1
    if line[end - 1] == "|" and (end == 1 or line[end - 2] != "\\"):
        end -= 1
    if start > end:
        return []
    cells: list[tuple[int, int]] = []
    cell_start = start
    for cursor in range(start, end):
        if (cursor - start) % 256 == 0:
            check_runtime()
        if line[cursor] == "|" and (cursor == start or line[cursor - 1] != "\\"):
            cells.append((cell_start, cursor))
            cell_start = cursor + 1
    cells.append((cell_start, end))
    return cells


def _markdown_table_delimiter_columns(
    line: str, minimum_indent: int, check_runtime: Callable[[], None]
) -> int | None:
    """Prove a compatible delimiter row under the existing container scope."""
    line = line.rstrip(LINE_BREAK_CHARS)
    prefix = 0
    column = 0
    while prefix < len(line) and line[prefix] in " \t":
        if prefix % 256 == 0:
            check_runtime()
        column += 4 - column % 4 if line[prefix] == "\t" else 1
        prefix += 1
    # Indentation belongs to the proven container, not the absolute column.
    if column < minimum_indent or column - minimum_indent >= 4 or prefix == len(line):
        return None
    if line[prefix] not in "|:-":
        return None
    # A new list item or a bare Setext/thematic underline takes precedence
    # over table recognition, including a pipe-bearing preceding header.
    if (
        line[prefix] == "-" and prefix + 1 < len(line) and line[prefix + 1] in " \t"
    ) or _markdown_block_separator(line, check_runtime):
        return None
    cells = _markdown_table_cells(line, prefix, check_runtime)
    if not cells:
        return None
    for start, end in cells:
        check_runtime()
        while start < end and line[start] in " \t":
            if start % 256 == 0:
                check_runtime()
            start += 1
        while end > start and line[end - 1] in " \t":
            if end % 256 == 0:
                check_runtime()
            end -= 1
        if start < end and line[start] == ":":
            start += 1
        hyphen_start = start
        while start < end and line[start] == "-":
            if (start - hyphen_start) % 256 == 0:
                check_runtime()
            start += 1
        if start == hyphen_start:
            return None
        if start < end and line[start] == ":":
            start += 1
        if start != end:
            return None
    return len(cells)


def _blank_line_local_paired_backticks(line: str) -> str:
    """Blank backtick runs on one line that pair as inline-code delimiters.

    Mirrors the equal-length, right-to-left pairing of the paragraph-scope
    inline-code masking: a backtick that demonstrably delimits inline code on
    its own line is Markdown syntax, never a shell substitution opener. Runs
    that could only pair across lines stay visible, keeping the prose check
    conservative, and escaped runs stay visible exactly as the masking does.
    """
    runs = [(match.start(), match.end()) for match in re.finditer(r"`+", line)]
    next_by_length: dict[int, int] = {}
    closing: dict[int, int] = {}
    for index in range(len(runs) - 1, -1, -1):
        length = runs[index][1] - runs[index][0]
        if length in next_by_length:
            closing[index] = next_by_length[length]
        next_by_length[length] = index
    chars = list(line)
    index = 0
    while index < len(runs):
        start, end = runs[index]
        escape_start = start
        while escape_start > 0 and line[escape_start - 1] == "\\":
            escape_start -= 1
        close_index = closing.get(index)
        if (start - escape_start) % 2 or close_index is None:
            index += 1
            continue
        close_start, close_end = runs[close_index]
        chars[start:end] = [" "] * (end - start)
        chars[close_start:close_end] = [" "] * (close_end - close_start)
        index = close_index + 1
    return "".join(chars)


def _markdown_shell_text(
    content: str,
    check_runtime: Callable[[], None],
    *,
    complete_context: bool = True,
    word_apostrophes: bool = False,
    fence_boundaries: list[int] | None = None,
    fence_regions: list[tuple[int, int, str]] | None = None,
    inline_regions: list[tuple[int, int]] | None = None,
    shell_fence_ranges: list[tuple[int, int]] | None = None,
    hardware_apostrophe_ambiguity: list[bool] | None = None,
    hardware_apostrophe_locations: list[tuple[int, int]] | None = None,
    powershell_blocks: list[tuple[int, int]] | None = None,
) -> str:
    """Mask Markdown delimiters while retaining code and exact source offsets.

    Inline code delimiters are not legacy shell substitutions. Fenced and
    indented code stays literal; longer inline delimiters preserve backticks
    inside their bodies. Pair equal-length runs in linear time.
    ``word_apostrophes`` selects the prose reading, which also masks the
    possessives and contractions of plain prose lines.
    ``fence_boundaries`` receives structural opening and closing marker offsets;
    inline-code delimiters do not establish a fence boundary.
    ``fence_regions`` receives body bounds and language labels, including
    unlabelled and unclosed fences, for host-language option matching.
    ``inline_regions`` receives proven inline-code body bounds, whose physical
    newlines fold into spaces rather than ending a command.
    ``shell_fence_ranges`` receives closed, explicitly labelled shell bodies
    whose quote ownership can be checked independently of surrounding prose.
    """
    output = list(content)
    prose_apostrophes: set[int] = set()
    runs: list[tuple[int, int]] = []
    list_marker = re.compile(r"(?:[-+*]|[0-9]{1,9}[.)])(?=[ \t])")
    backtick_runs = re.compile(r"`+")

    def complete_inline_code_pairs(
        source: str,
        delimiter_runs: list[tuple[int, int]],
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        next_by_length: dict[int, int] = {}
        closing: dict[int, int] = {}
        for index in range(len(delimiter_runs) - 1, -1, -1):
            check_runtime()
            start, end = delimiter_runs[index]
            length = end - start
            if length in next_by_length:
                closing[index] = next_by_length[length]
            next_by_length[length] = index

        pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        index = 0
        while index < len(delimiter_runs):
            check_runtime()
            start, end = delimiter_runs[index]
            escape_start = start
            while escape_start > 0 and source[escape_start - 1] == "\\":
                escape_start -= 1
            close_index = closing.get(index)
            if (start - escape_start) % 2 or close_index is None:
                index += 1
                continue
            close_start, close_end = delimiter_runs[close_index]
            pairs.append(((start, end), (close_start, close_end)))
            index = close_index + 1
        return pairs

    def mask_inline_delimiters(*, only_complete: bool = False) -> None:
        if not complete_context:
            runs.clear()
            return
        pairs = complete_inline_code_pairs(content, runs)
        if only_complete:
            pair_index = 0
            for start, end in runs:
                check_runtime()
                while pair_index < len(pairs) and pairs[pair_index][1][1] <= start:
                    pair_index += 1
                if pair_index < len(pairs) and (
                    pairs[pair_index][0][0] <= start < end <= pairs[pair_index][1][1]
                ):
                    continue
                escape_start = start
                while escape_start > 0 and content[escape_start - 1] == "\\":
                    escape_start -= 1
                if (start - escape_start) % 2 == 0:
                    return
        for (start, end), (close_start, close_end) in pairs:
            if inline_regions is not None:
                inline_regions.append((end, close_start))
            output[start:end] = " " * (end - start)
            output[close_start:close_end] = " " * (close_end - close_start)
        runs.clear()

    def prose_without_complete_inline_code(
        text: str,
        source_start: int,
        *,
        list_marker_owns_initial_command: bool,
    ) -> tuple[str, bool]:
        """Blank complete same-line spans and retain their runtime-command guard."""
        line_runs = [(match.start(), match.end()) for match in backtick_runs.finditer(text)]
        projected = list(text)
        protects_runtime_command = False

        def starts_additional_clause(source: str) -> bool:
            return any(
                clause_start > 0 and source[clause_start - 1] in ";|&\n"
                for clause_start in _shell_clause_starts(source, check_runtime)
            )

        for (start, end), (close_start, close_end) in complete_inline_code_pairs(text, line_runs):
            body = text[end:close_start]
            remainder_start = source_start + start
            evidence_end = remainder_start + _ROOT_GLOB_COMMAND_CHARS
            list_marker_owns_span_command = (
                list_marker_owns_initial_command
                and not starts_additional_clause(text[:start])
                and not starts_additional_clause(body)
            )
            if (
                # A clipped window cannot prove that operands are absent.
                (
                    evidence_end < len(content)
                    or _may_have_destructive_outer_operands(content[remainder_start:evidence_end])
                )
                and (
                    _MARKDOWN_PROSE_SHELL_SYNTAX_RE.search(body) is not None
                    or next(_destructive_command_words(body), None) is not None
                )
                and not list_marker_owns_span_command
            ):
                protects_runtime_command = True
            projected[start:close_end] = " " * (close_end - start)
        return "".join(projected), protects_runtime_command

    # This is a conservative projection, not a general Markdown renderer.
    # Container/HTML bodies with uncertain inline ownership remain literal.
    fence: tuple[str, int, int, str] | None = None
    fence_body_start = 0
    fence_literal_apostrophes: list[int] = []
    fence_literal_ownership = True
    fence_has_literal_apostrophe = False
    fence_has_nonliteral_apostrophe = False
    shell_fence_start: int | None = None
    quoted_block = False
    html_end: str | None = None
    paragraph_open = False
    paragraph_in_list = False
    paragraph_list_indent = 0
    list_content_indents: list[int] = []
    table_columns: int | None = None
    table_minimum_indent = 0
    table_delimiter_line = -1
    offset = 0
    lines = content.splitlines(keepends=True)
    for line_index, line in enumerate(lines):
        check_runtime()
        stripped = line.rstrip(LINE_BREAK_CHARS)
        leading = stripped.lstrip(" \t")
        indentation = len(stripped[: len(stripped) - len(leading)].expandtabs(4))
        # List containers survive blank lines and restore their enclosing
        # context after a nested item. Literal fence/HTML/quote bodies cannot
        # establish or discard list ownership.
        list_context = fence is None and html_end is None and not quoted_block
        list_index = bisect_right(list_content_indents, indentation)
        list_base = list_content_indents[list_index - 1] if list_index else 0
        # Interpret list padding in columns, preserving the original offsets.
        # More than four columns after a marker can introduce indented code.
        prefix = len(stripped) - len(leading)
        column = indentation
        list_indented = False
        has_list_marker = False
        # The first marker is the shell word that owns the command position.
        # ``*`` is a glob, not an inert word, so it never owns it.
        first_marker_is_glob = False
        list_markers = 0
        new_list_indents: list[int] = []
        if 0 <= indentation - list_base < 4:
            while marker := list_marker.match(stripped, prefix):
                check_runtime()
                if (
                    not has_list_marker
                    and paragraph_open
                    and not paragraph_in_list
                    and marker[0][0].isdigit()
                    and int(marker[0][:-1]) != 1
                ):
                    # Only a list starting at 1 can interrupt a paragraph.
                    # Other numbers may be literal text within an inline span.
                    break
                if not has_list_marker:
                    first_marker_is_glob = marker[0][0] == "*"
                has_list_marker = True
                list_markers += 1
                column += marker.end() - prefix
                prefix = marker.end()
                padding_start = column
                while prefix < len(stripped) and stripped[prefix] in " \t":
                    check_runtime()
                    column += 4 - column % 4 if stripped[prefix] == "\t" else 1
                    prefix += 1
                if column - padding_start > 4:
                    list_indented = True
                    break
                new_list_indents.append(column)
            leading = stripped[prefix:]
        quote_start = indentation < 4 and leading.startswith(">")
        heading = indentation < 4 and re.match(r"#{1,6}(?:[ \t]|$)", leading) is not None
        separator = indentation < 4 and _markdown_block_separator(stripped, check_runtime)
        setext_only = separator and (leading.startswith("=") or leading.rstrip(" \t") == "--")
        empty_list_item = (
            not paragraph_open and re.fullmatch(r"(?:[-+*]|[0-9]{1,9}[.)])", leading) is not None
        )
        if table_columns is not None and setext_only:
            # A table row has no paragraph for a Setext underline to close.
            # Single '-' still starts an empty item; '---' remains thematic.
            separator = False
        html_open = re.match(r"<(?:[A-Za-z][A-Za-z0-9-]*(?=[\s/>]|$)|[!?/])", leading)
        lazy_list_paragraph = (
            paragraph_open
            and paragraph_in_list
            and not (has_list_marker or quote_start or heading or separator or html_open)
            and MARKDOWN_FENCE_OPEN.fullmatch(leading) is None
        )
        if list_context and leading and not lazy_list_paragraph:
            del list_content_indents[list_index:]
        if list_context and not quote_start and not html_open:
            list_content_indents.extend(new_list_indents)
        list_content_indent = list_content_indents[-1] if list_content_indents else None
        list_fence = (
            list_content_indent is not None
            and table_columns is None
            and (paragraph_in_list or line_index > 0 and not lines[line_index - 1].strip())
            and not has_list_marker
            and 0 <= indentation - list_content_indent <= 3
            and MARKDOWN_FENCE_OPEN.fullmatch(leading) is not None
        )
        nested_fence_opening = (
            MARKDOWN_FENCE_OPEN.fullmatch(stripped[prefix:]) if list_fence else None
        )
        continuing_paragraph = False
        if table_columns is not None and (
            html_end is not None
            or fence is not None
            or quote_start
            or quoted_block
            or html_open
            or not leading
            or indentation >= 4
            or indentation < table_minimum_indent
            or list_indented
            or has_list_marker
            or empty_list_item
            or heading
            or separator
        ):
            table_columns = None
        if fence is not None and leading and indentation < fence[2]:
            # A fenced block inside a list ends when its list container ends.
            # Process this dedented line again as Markdown, rather than letting
            # a prose apostrophe open a shell quote inside the preceding fence.
            if fence_boundaries is not None:
                fence_boundaries.append(offset)
            if fence_regions is not None:
                fence_regions.append((fence_body_start, offset, fence[3]))
            if (
                powershell_blocks is not None
                and complete_context
                and fence[3] in {"powershell", "pwsh", "ps1"}
            ):
                powershell_blocks.append((fence_body_start, offset))
            fence = None
        if html_end is not None:
            mask_inline_delimiters()
            if (html_end and html_end in leading.lower()) or (not html_end and not leading):
                html_end = None
        elif fence is not None:
            fence_body = stripped.lstrip(" \t")
            closing_fence = MARKDOWN_FENCE_CLOSE.fullmatch(fence_body)
            if (
                closing_fence
                and 0 <= indentation - fence[2] <= 3
                and closing_fence[1][0] == fence[0]
                and len(closing_fence[1]) >= fence[1]
            ):
                begin, end = closing_fence.span(1)
                body_start = offset + len(stripped) - len(fence_body)
                output[body_start + begin : body_start + end] = " " * (end - begin)
                if fence_boundaries is not None:
                    fence_boundaries.append(body_start + begin)
                if fence_regions is not None:
                    fence_regions.append((fence_body_start, offset, fence[3]))
                if (
                    powershell_blocks is not None
                    and complete_context
                    and fence[3] in {"powershell", "pwsh", "ps1"}
                ):
                    powershell_blocks.append((fence_body_start, offset))
                if shell_fence_start is not None and shell_fence_ranges is not None:
                    shell_fence_ranges.append((shell_fence_start, offset))
                shell_fence_start = None
                fence = None
                fence_literal_apostrophes.clear()
                fence_literal_ownership = True
                fence_has_literal_apostrophe = False
                fence_has_nonliteral_apostrophe = False
            elif fence[3] in {"tcl", "verilog", "systemverilog", "sv", "svh"}:
                # Sized hardware literals use an apostrophe as a base marker,
                # not a shell quote. Unsized forms are owned only in an HDL
                # expression context; a fence label alone cannot prove that
                # an apostrophe inside embedded shell text is a literal.
                literal_apostrophes: set[int] = set()
                for literal in re.finditer(
                    r"(?<![A-Za-z0-9_])(?:[0-9]+(?P<sized>')[sS]?[bBoOdDhH]"
                    r"[0-9a-fA-F_xXzZ?]+|(?P<unsized>')[01xXzZ])(?![A-Za-z0-9_])",
                    stripped,
                ):
                    if literal["unsized"]:
                        prefix = stripped[: literal.start()].rstrip(" \t")
                        if (
                            fence[3] == "tcl"
                            or not prefix
                            or prefix[-1] not in "=(:,{[!~&|?+-*/%"
                            or stripped[literal.end() :].startswith("'")
                        ):
                            continue
                    literal_apostrophes.add(
                        literal.start("sized" if literal["sized"] else "unsized")
                    )
                has_nonliteral_apostrophe = any(
                    character == "'" and index not in literal_apostrophes
                    for index, character in enumerate(stripped)
                )
                fence_has_literal_apostrophe = fence_has_literal_apostrophe or bool(
                    literal_apostrophes
                )
                fence_has_nonliteral_apostrophe = (
                    fence_has_nonliteral_apostrophe or has_nonliteral_apostrophe
                )
                if (
                    hardware_apostrophe_ambiguity is not None
                    and fence_has_literal_apostrophe
                    and fence_has_nonliteral_apostrophe
                ):
                    hardware_apostrophe_ambiguity[0] = True
                    if (
                        hardware_apostrophe_locations is not None
                        and not hardware_apostrophe_locations
                    ):
                        hardware_apostrophe_locations.append((offset, offset + len(line)))
                if has_nonliteral_apostrophe:
                    for apostrophe in fence_literal_apostrophes:
                        output[apostrophe] = "'"
                    fence_literal_apostrophes.clear()
                    fence_literal_ownership = False
                elif fence_literal_ownership:
                    for apostrophe in literal_apostrophes:
                        absolute_apostrophe = offset + apostrophe
                        output[absolute_apostrophe] = " "
                        fence_literal_apostrophes.append(absolute_apostrophe)
        elif quote_start or quoted_block:
            mask_inline_delimiters()
            quoted_block = bool(leading)
        elif html_open:
            mask_inline_delimiters()
            raw_tag = re.match(r"<(pre|script|style|textarea)(?=[\s/>]|$)", leading, re.I)
            terminator = (
                f"</{raw_tag[1].lower()}>"
                if raw_tag
                else "-->"
                if leading.startswith("<!--")
                else "?>"
                if leading.startswith("<?")
                else "]]>"
                if leading.startswith("<![CDATA[")
                else ">"
                if re.match(r"<![A-Z]", leading)
                else ""
            )
            html_end = None if terminator and terminator in leading.lower() else terminator
        elif nested_fence_opening:
            mask_inline_delimiters()
            begin, end = nested_fence_opening.span(1)
            if fence_boundaries is not None:
                fence_boundaries.append(offset + prefix + begin)
            info = stripped[prefix + end :].strip()
            fence = (
                nested_fence_opening[1][0],
                len(nested_fence_opening[1]),
                list_content_indent,
                info.split(maxsplit=1)[0].casefold() if info else "",
            )
            fence_body_start = offset + len(line)
            fence_literal_apostrophes.clear()
            fence_literal_ownership = True
            fence_has_literal_apostrophe = False
            fence_has_nonliteral_apostrophe = False
            output[offset + prefix + begin : offset + prefix + end] = " " * (end - begin)
        elif (
            not leading
            or list_indented
            or indentation >= 4
            and has_list_marker
            or indentation >= 4
            and not (
                paragraph_in_list
                and 0 <= indentation - paragraph_list_indent <= 3
                # A continuation uses the proven content margin. A nested
                # marker or a new block needs its own container proof.
                and list_marker.match(leading) is None
                and not leading.startswith((">", "|"))
                and re.match(r"#{1,6}(?:[ \t]|$)", leading) is None
                and not _markdown_block_separator(leading, check_runtime)
                and table_columns is None
                and not (
                    line_index + 1 < len(lines)
                    and (
                        columns := _markdown_table_delimiter_columns(
                            lines[line_index + 1], paragraph_list_indent, check_runtime
                        )
                    )
                    is not None
                    and len(_markdown_table_cells(stripped, prefix, check_runtime)) == columns
                )
            )
        ):
            mask_inline_delimiters()
        else:
            # Inline code may continue within a paragraph/list item, but cannot
            # pair with delimiters in a new item, heading, or following block.
            if has_list_marker or heading or separator:
                mask_inline_delimiters()
            # The body after any list markers can begin a fenced block.
            opening = MARKDOWN_FENCE_OPEN.fullmatch(stripped[prefix:])
            if opening:
                mask_inline_delimiters()
                begin, end = opening.span(1)
                if fence_boundaries is not None:
                    fence_boundaries.append(offset + prefix + begin)
                info = stripped[prefix + end :].strip()
                language = info.split(maxsplit=1)[0].casefold() if info else ""
                fence = (
                    opening[1][0],
                    len(opening[1]),
                    column if has_list_marker else 0,
                    language,
                )
                fence_body_start = offset + len(line)
                fence_literal_apostrophes.clear()
                fence_literal_ownership = True
                fence_has_literal_apostrophe = False
                fence_has_nonliteral_apostrophe = False
                output[offset + prefix + begin : offset + prefix + end] = " " * (end - begin)
                if (
                    complete_context
                    and shell_fence_ranges is not None
                    and not has_list_marker
                    and not paragraph_in_list
                    and stripped[prefix + end :].strip() in {"bash", "sh", "shell"}
                ):
                    shell_fence_start = offset + len(line)
            else:
                minimum_indent = (
                    column if has_list_marker else paragraph_list_indent if paragraph_in_list else 0
                )
                if (
                    table_columns is None
                    and not heading
                    and not empty_list_item
                    and (not separator or setext_only and not paragraph_open)
                    and list_markers <= 1
                    and (has_list_marker or indentation >= minimum_indent)
                    and line_index + 1 < len(lines)
                ):
                    columns = _markdown_table_delimiter_columns(
                        lines[line_index + 1], minimum_indent, check_runtime
                    )
                    if (
                        columns is not None
                        and len(_markdown_table_cells(stripped, prefix, check_runtime)) == columns
                    ):
                        # Establish the header boundary before pairing any
                        # opener from the preceding paragraph with this row.
                        mask_inline_delimiters()
                        # A Setext-looking line at a fresh block boundary is
                        # a header only after the next row proves that role.
                        separator = False
                        table_columns = columns
                        table_minimum_indent = minimum_indent
                        table_delimiter_line = line_index + 1
                if table_columns is not None:
                    mask_inline_delimiters()
                    if line_index != table_delimiter_line:
                        cells = _markdown_table_cells(stripped, prefix, check_runtime)
                        for cell_index, (cell_start, cell_end) in enumerate(cells):
                            if cell_index >= table_columns:
                                # GFM drops excess body cells: no rendered
                                # inline node can grant ownership to their ticks.
                                break
                            for match in backtick_runs.finditer(stripped, cell_start, cell_end):
                                check_runtime()
                                runs.append((offset + match.start(), offset + match.end()))
                            mask_inline_delimiters()
                        if not cells:
                            table_columns = None
                        elif (
                            word_apostrophes
                            and complete_context
                            and "'" in stripped
                            and _MARKDOWN_PROSE_EXPANSION_RE.search(stripped) is None
                        ):
                            apostrophes = [
                                offset + word.start("apostrophe")
                                for word in _MARKDOWN_WORD_APOSTROPHE_RE.finditer(stripped)
                            ]
                            if len(apostrophes) == stripped.count("'"):
                                prose_apostrophes.update(apostrophes)
                elif not separator and not empty_list_item:
                    prose_leading = leading
                    inline_runtime_guard = False
                    if (
                        complete_context
                        and not runs
                        and len(leading) <= 512
                        and 0 <= leading.find("`") < leading.find("'")
                    ):
                        prose_leading, inline_runtime_guard = prose_without_complete_inline_code(
                            leading,
                            offset + prefix,
                            list_marker_owns_initial_command=(
                                has_list_marker and not first_marker_is_glob
                            ),
                        )
                    if (
                        complete_context
                        and not runs
                        and not heading
                        and not inline_runtime_guard
                        and len(prose_leading) <= 512
                        and (prose := _MARKDOWN_CONTRACTION_PROSE_RE.fullmatch(prose_leading))
                    ):
                        # A whole plain-language sentence, including a parsed
                        # list-item body, owns its contraction;
                        # it is not an unclosed shell quote spanning later prose.
                        # The restrictive grammar excludes code delimiters and
                        # shell syntax. Fences, indented code, HTML, tables, and
                        # pending multiline inline spans never reach this rule.
                        prose_apostrophes.add(offset + prefix + prose.start("apostrophe"))
                    elif (
                        word_apostrophes
                        and complete_context
                        and not runs
                        and not heading
                        and not inline_runtime_guard
                        and "'" in prose_leading
                        # Inline-code delimiters are Markdown, not shell
                        # syntax: judge the line with its line-local pairs
                        # blanked, exactly as the masking pass will.
                        and _MARKDOWN_PROSE_EXPANSION_RE.search(
                            _blank_line_local_paired_backticks(prose_leading)
                        )
                        is None
                        and (
                            line_index == 0
                            or (
                                lines[line_index - 1].endswith("\n")
                                and not lines[line_index - 1].rstrip("\r\n").endswith("\\")
                            )
                        )
                    ):
                        # In the prose reading, a prose line owns every
                        # possessive and contraction in it. Every quote on the
                        # line must be one, the line has no expansion or line
                        # continuation, and it starts a new shell line, so no
                        # runtime-selected command can take later text as
                        # operands through it. The same block exclusions apply.
                        apostrophes: list[int] = []
                        for word in _MARKDOWN_WORD_APOSTROPHE_RE.finditer(prose_leading):
                            check_runtime()
                            apostrophes.append(offset + prefix + word.start("apostrophe"))
                        if len(apostrophes) == prose_leading.count("'"):
                            prose_apostrophes.update(apostrophes)
                    pending_inline = bool(runs)
                    for match in backtick_runs.finditer(line):
                        check_runtime()
                        runs.append((offset + match.start(), offset + match.end()))
                    if heading:
                        mask_inline_delimiters()
                    else:
                        if not pending_inline:
                            # A complete line cannot lend an opener to later prose.
                            # Unmatched runs retain the existing multiline proof.
                            mask_inline_delimiters(only_complete=True)
                        continuing_paragraph = True
                        paragraph_in_list = has_list_marker or paragraph_in_list
                        if has_list_marker:
                            paragraph_list_indent = column
        paragraph_open = continuing_paragraph
        if not paragraph_open:
            paragraph_in_list = False
            paragraph_list_indent = 0
        offset += len(line)
    if fence is not None and fence_regions is not None:
        fence_regions.append((fence_body_start, len(content), fence[3]))
    mask_inline_delimiters()
    projected = "".join(output)
    quoted_code: dict[int, tuple[int, int, int]] = {}
    if complete_context and (quoted := quoted_list_spans(content, check_runtime)) is not None:
        quoted_code = {start: (body, close, end) for start, body, close, end in quoted.code}
        if word_apostrophes:
            code_starts = sorted(quoted_code)
            for start, end in quoted.lines:
                check_runtime()
                line_start = content.rfind("\n", 0, start) + 1
                if line_start and content[max(0, line_start - 3) : line_start].rstrip(
                    "\r\n"
                ).endswith("\\"):
                    continue
                line = list(projected[start:end])
                for index in range(bisect_left(code_starts, start), bisect_left(code_starts, end)):
                    opening = code_starts[index]
                    body, close, finish = quoted_code[opening]
                    line[opening - start : body - start] = " " * (body - opening)
                    line[close - start : finish - start] = " " * (finish - close)
                prose = "".join(line)
                if "'" in prose and _MARKDOWN_PROSE_EXPANSION_RE.search(prose) is None:
                    apostrophes = [
                        start + word.start("apostrophe")
                        for word in _MARKDOWN_WORD_APOSTROPHE_RE.finditer(prose)
                    ]
                    if len(apostrophes) == prose.count("'"):
                        prose_apostrophes.update(apostrophes)
    if prose_apostrophes or quoted_code:
        # A prose-looking line can still be inside an earlier shell word, such
        # as a multiline bash -c string. Reuse the shell parser to preserve that
        # ownership before masking any apostrophe. Unresolved words retain all
        # remaining bytes; this pass never grants ownership past a parse limit.
        cursor = 0
        next_runtime_check = 0
        last_candidate = max(prose_apostrophes | quoted_code.keys())
        parameter_ends: dict[int, _ParameterExpansionEnd] = {}
        substitution_ends: dict[int, int | None] = {}
        backtick_ends: dict[int, int | None] = {}
        while cursor <= last_candidate:
            if cursor >= next_runtime_check:
                check_runtime()
                next_runtime_check = cursor + 4096
            if cursor in quoted_code:
                body, close, end = quoted_code[cursor]
                output[cursor:body] = " " * (body - cursor)
                output[close:end] = " " * (end - close)
                cursor = end
                continue
            if cursor in prose_apostrophes:
                output[cursor] = " "
            elif projected[cursor] in "'\"`$\\":
                parsed = _parse_shell_command_word(
                    projected,
                    cursor,
                    parameter_ends,
                    substitution_ends,
                    backtick_ends,
                    check_runtime=check_runtime,
                )
                if parsed is None or parsed.limited:
                    break
                cursor = max(cursor + 1, parsed.end)
                continue
            cursor += 1
        return "".join(output)
    return projected


def _fenced_powershell_source(
    content: str,
    blocks: list[tuple[int, int]],
    check_runtime: Callable[[], None],
    unresolved_executables: list[int],
) -> PowerShellSource | None:
    """Keep complete PowerShell fence ownership in document coordinates."""
    starts: list[int] = []
    ends: list[int] = []
    strings: list[PowerShellString] = []
    clause_starts: set[int] = set()
    continuations: list[tuple[int, int]] = []
    invocations: list[tuple[int, int]] = []
    for start, end in blocks:
        check_runtime()
        unresolved: list[int] = []
        source = powershell_source(
            content[start:end], check_runtime, unresolved_executables=unresolved
        )
        unresolved_executables.extend(start + position for position in unresolved)
        if source is None:
            continue
        starts.extend(start + position for position in source.starts)
        ends.extend(start + position for position in source.ends)
        strings.extend(
            PowerShellString(
                start + string.start,
                start + string.end,
                string.value,
                string.decoded,
                string.invoked,
                start + string.operand_start if string.operand_start is not None else None,
            )
            for string in source.strings
        )
        clause_starts.update(start + position for position in source.clause_starts)
        continuations.extend((start + begin, start + end) for begin, end in source.continuations)
        invocations.extend((start + begin, start + end) for begin, end in source.invocations)
    if not starts and not strings and not continuations and not invocations:
        return None
    source = PowerShellSource(tuple(starts), tuple(ends), tuple(strings), ())
    # Preserve the ordinary shell traversal outside the proven host spans.
    # Add code inside PowerShell $() that POSIX string rules would miss.
    # Host quotes cannot swallow a following Bash fence. This projection is
    # only for locating clauses; command strings still use the original bytes.
    projected = list(content)
    for start, end in zip(source.starts, source.ends, strict=True):
        check_runtime()
        projected[start:end] = [
            character if character in "\r\n" else " " for character in content[start:end]
        ]
    for start in _shell_clause_starts("".join(projected), check_runtime):
        check_runtime()
        position = start
        while position < len(content) and content[position] in " \t":
            if (position - start) % 1024 == 0:
                check_runtime()
            position += 1
        if not source.owns(position):
            clause_starts.add(start)
    return PowerShellSource(
        source.starts,
        source.ends,
        source.strings,
        tuple(sorted(clause_starts)),
        tuple(continuations),
        tuple(invocations),
    )


def _yaml_source_start(
    content: str,
    source_view: SecurityTextView | None,
    canonical: str,
    check_runtime: Callable[[], None],
) -> int | None:
    if (
        source_view is None
        or source_view.name != "raw"
        or source_view.text != content
        or not content
    ):
        return None
    start = source_view.source_offset(0)
    end = start + len(content)
    if not 0 <= start < end <= len(canonical) or canonical[start:end] != content:
        return None
    for offset in range(len(content)):
        if offset % 4096 == 0:
            check_runtime()
        if source_view.source_offset(offset) != start + offset:
            return None
    return start


@dataclass(frozen=True)
class _YamlParserAnalysis:
    content: str
    spans: YamlSourceSpans

    def can_emit(self, scalar: YamlScalar) -> bool:
        # A scalar no wider than the overlap fits the raw window owning its
        # opener. Larger scalars need the complete small-artifact window.
        return (
            len(self.content) <= static_runner.SECURITY_VIEW_WINDOW_CHARS
            or scalar.end - scalar.start <= static_runner._WINDOW_OVERLAP_CHARS
        )

    def owns(
        self, content: str, source_view: SecurityTextView | None, check_runtime: Callable[[], None]
    ) -> bool:
        return (
            len(content) == len(self.content)
            and _yaml_source_start(content, source_view, self.content, check_runtime) == 0
        )


def prepare_parser_analysis(
    content: str, file_type: str, check_runtime: Callable[[], None]
) -> JSONSourceOwnership | _YamlParserAnalysis | None:
    if file_type == "json":
        return prepare_json_source(content, check_runtime)
    if file_type != "yaml":
        return None
    spans = yaml_source_spans(content, check_runtime)
    return _YamlParserAnalysis(content, spans) if spans is not None else None


def _yaml_scalar_mapper(
    location: Location, context: str, start: int, scalar: YamlScalar
) -> Callable[[AnalyzerFinding], None]:
    occurrence = 0

    def enclose(finding: AnalyzerFinding) -> None:
        nonlocal occurrence
        occurrence += 1
        decoded_line = finding.location.start_line
        # Decoded coordinates must never escape through an interrupted
        # finding-budget prefix or Python-specific reconciliation.
        finding.location = replace(location)
        finding.context = context
        finding.evidence = {
            key: value for key, value in finding.evidence.items() if not key.startswith("_")
        }
        finding.evidence.update(
            {
                static_runner._VIEW_START_EVIDENCE: start,
                "yaml_scalar": {
                    "location_precision": "enclosing_expression",
                    "source_start_offset": scalar.start,
                    "source_end_offset": scalar.end,
                    "decoded_start_line": decoded_line,
                    "decoded_occurrence": occurrence,
                },
            }
        )

    return enclose


@dataclass(frozen=True)
class _PreparedYamlAnalysis:
    parser: _YamlParserAnalysis
    check_runtime: Callable[[], None]

    def analyze(
        self, content: str, file_path: str, file_type: str, source_view: SecurityTextView
    ) -> list[AnalyzerFinding]:
        findings = analyze(
            content,
            file_path,
            file_type,
            check_runtime=self.check_runtime,
            defer_variable_reconciliation=True,
        )
        if file_type != "yaml":
            return findings
        source_start = _yaml_source_start(
            content, source_view, self.parser.content, self.check_runtime
        )
        if source_start is None:
            return findings
        locations = SourceLocationIndex(content, file_path)
        for scalar in self.parser.spans.scalars:
            self.check_runtime()
            if not self.parser.can_emit(scalar):
                continue
            start, end = scalar.start - source_start, scalar.end - source_start
            if not 0 <= start < end <= len(content):
                continue
            if self.parser.content[scalar.body_start : scalar.body_end] == scalar.value:
                continue
            enclose = _yaml_scalar_mapper(
                locations.location(start, end), get_context(content, start), start, scalar
            )
            with map_analyzer_findings(enclose):
                findings.extend(
                    analyze(scalar.value, file_path, "shell", check_runtime=self.check_runtime)
                )
        return findings


def prepare_analysis(
    content: str, file_type: str, check_runtime: Callable[[], None]
) -> _PreparedYamlAnalysis | None:
    if file_type != "yaml":
        return None
    parser = prepare_parser_analysis(content, file_type, check_runtime)
    return _PreparedYamlAnalysis(parser, check_runtime) if parser is not None else None


def has_bounded_parse_exhaustion(
    content: str,
    check_runtime: Callable[[], None],
    *,
    file_type: str = "shell",
    complete_context: bool = True,
    prepared_analysis: object | None = None,
    source_view: SecurityTextView | None = None,
    python_import_check: Callable[[Callable[[], None]], bool] | None = None,
    python_stdlib_check: Callable[[str, Callable[[], None]], bool] | None = None,
) -> bool:
    """Return whether the bounded static parser leaves an expression unresolved."""
    return (
        bounded_parse_diagnostic(
            content,
            check_runtime,
            file_type=file_type,
            complete_context=complete_context,
            prepared_analysis=prepared_analysis,
            source_view=source_view,
            python_import_check=python_import_check,
            python_stdlib_check=python_stdlib_check,
        )
        is not None
    )


def bounded_parse_diagnostic(
    content: str,
    check_runtime: Callable[[], None],
    *,
    file_type: str = "shell",
    complete_context: bool = True,
    prepared_analysis: object | None = None,
    source_view: SecurityTextView | None = None,
    python_import_check: Callable[[Callable[[], None]], bool] | None = None,
    python_stdlib_check: Callable[[str, Callable[[], None]], bool] | None = None,
) -> StaticParserDiagnostic | None:
    """Explain the existing first incomplete parser scope without reparsing it."""
    exhausted = _has_bounded_parse_exhaustion(
        content,
        check_runtime,
        file_type=file_type,
        complete_context=complete_context,
        prepared_analysis=prepared_analysis,
        source_view=source_view,
        python_import_check=python_import_check,
        python_stdlib_check=python_stdlib_check,
    )
    if exhausted is not None and not complete_context:
        exhausted = replace(exhausted, full_input_checked=False)
    if file_type == "yaml" and complete_context:
        # A complete YAML grammar bounds scalars independently. Inspect decoded
        # text even when raw quoting concealed a runtime command or finding.
        yaml_spans = yaml_source_spans(content, check_runtime)
        if yaml_spans is None:
            return exhausted
        mapped_source = isinstance(
            prepared_analysis, _YamlParserAnalysis
        ) and prepared_analysis.owns(content, source_view, check_runtime)
        for scalar in yaml_spans.scalars:
            check_runtime()
            diagnostic = _has_bounded_parse_exhaustion(
                scalar.value, check_runtime, file_type="shell", complete_context=True
            )
            exact_source = content[scalar.body_start : scalar.body_end] == scalar.value
            if diagnostic is not None:
                return (
                    diagnostic.shifted(scalar.body_start)
                    if exact_source
                    else diagnostic.enclosed(scalar.start, scalar.end)
                )
            # Folding and escapes may introduce a finding without raw source
            # provenance. Keep that transformed form partial until a derived
            # finding view can carry it through the normal reporting path.
            mapped_scalar = (
                isinstance(prepared_analysis, _YamlParserAnalysis)
                and mapped_source
                and prepared_analysis.can_emit(scalar)
            )
            if (
                not exact_source
                and not mapped_scalar
                and analyze(scalar.value, "<yaml-scalar>", "shell", check_runtime=check_runtime)
            ):
                return StaticParserDiagnostic(
                    ParserCause.UNSUPPORTED_SYNTAX,
                    ParserCheck.COMMAND_STRING_EFFECTS,
                    scalar.start,
                    scalar.end,
                    ParserUncheckedScope.EXPRESSION,
                )
        for start, end in yaml_spans.comments:
            check_runtime()
            # Comments can instruct an agent too. Prose punctuation has its
            # own reading, while runtime-dependent commands remain incomplete.
            diagnostic = bounded_parse_diagnostic(
                content[start + 1 : end], check_runtime, file_type="markdown"
            )
            if diagnostic is not None:
                return diagnostic.shifted(start + 1)
        return None
    if not exhausted or file_type != "markdown" or not complete_context:
        return exhausted
    # A possessive such as "reference's" in prose is punctuation, yet the
    # lexical reading can leave it open as a shell quote to the end of the
    # file. Retry with the prose reading. Masking an apostrophe can also turn
    # a later closing quote into an opener, so the prose reading may only
    # resolve a limit of the lexical one, never introduce one.
    return _has_bounded_parse_exhaustion(
        content,
        check_runtime,
        file_type=file_type,
        word_apostrophes=True,
        source_view=source_view,
    )


def _has_bounded_parse_exhaustion(
    content: str,
    check_runtime: Callable[[], None],
    *,
    file_type: str,
    complete_context: bool = True,
    word_apostrophes: bool = False,
    prepared_analysis: object | None = None,
    source_view: SecurityTextView | None = None,
    python_import_check: Callable[[Callable[[], None]], bool] | None = None,
    python_stdlib_check: Callable[[str, Callable[[], None]], bool] | None = None,
) -> StaticParserDiagnostic | None:
    # PowerShell source can invoke another shell with an executable command
    # string. Apply the same bounded checks to those strings; benign PowerShell
    # -replace values have their own narrowly proven expression context.
    structural_quote_closers = None
    projected_continuations: list[int] = []
    if file_type == "powershell" and complete_context:
        # Only a complete script proves token ownership. A fragment may begin
        # inside a string or comment and would invert code and literal bytes.
        content = _powershell_shell_text(
            content, check_runtime, continuations=projected_continuations
        )
        continuation_diagnostics: list[StaticParserDiagnostic] = []
        if projected_continuations and _has_destructive_command_string(
            content, diagnostics=continuation_diagnostics
        ):
            # Projection can expose operands absent from the raw finding view.
            # Retain the native continuation contract before ownership is rebuilt.
            return continuation_diagnostics[0]
    structural_quote_openers = None
    shell_fence_ranges: list[tuple[int, int]] = []
    json_strings: list[tuple[int, int]] = []
    json_operand_spans: list[tuple[int, int]] = []
    python_source = None
    javascript_source = None
    unresolved_executables: list[int] = []
    powershell = (
        powershell_source(content, check_runtime, unresolved_executables=unresolved_executables)
        if file_type == "powershell" and complete_context
        else None
    )
    if file_type == "perl" and complete_context:
        content = _perl_literal_print_shell_text(content, check_runtime)
    if file_type == "python" and complete_context:
        # Only a complete module can prove Python token ownership. A fragment
        # may begin inside a string and would invert code and literal bytes.
        python_source = _PythonSourceOwnership(
            content, check_runtime, python_import_check, python_stdlib_check
        )
    if file_type == "javascript" and complete_context:
        # Modern host syntax owns literal delimiters only after a bounded,
        # complete parse. Unresolved executable interpolations keep their bounds.
        javascript_source = _JavaScriptTemplateOwnership(content, check_runtime)
        diagnostic = javascript_source.unresolved_execution_diagnostic()
        if diagnostic is not None:
            return diagnostic
    json_projection = None
    if file_type == "json":
        if prepared_analysis is None and complete_context:
            prepared_analysis = prepare_json_source(content, check_runtime)
        if isinstance(prepared_analysis, JSONSourceOwnership):
            view = source_view or SecurityTextView("raw", content)
            if view.text == content:
                json_projection = prepared_analysis.project(view, check_runtime)
    if json_projection is not None:
        json_strings = list(json_projection.strings)
        structural_quote_closers = json_projection.closers
        structural_quote_openers = json_projection.openers
    elif file_type in {"json", "markdown"} and complete_context:
        json_strings = validated_json_string_spans(content, check_runtime)
        structural_quote_closers = {end - 1 for _, end in json_strings}
        structural_quote_openers = {start for start, _ in json_strings}
    if json_strings and (source_view is None or source_view.name == "raw"):
        # The original finding producer is the authority for source provenance.
        # Both iterators are ordered; retain only one candidate, not a per-file
        # set or repeated scans of every suffix. Finding construction stays raw.
        reported_starts = (
            candidate[0]
            for candidate in _destructive_command_candidates(content, check_runtime=check_runtime)
        )
        reported_start: int | None = -1
        for start, end in json_strings:
            check_runtime()
            for command_start, body_start in _destructive_command_effects(
                content[start + 1 : end - 1], check_runtime
            ):
                absolute_start = start + 1 + command_start
                if _has_tm1_code_pattern_at(
                    content,
                    absolute_start,
                    min(end - 1, start + 1 + body_start + _ROOT_GLOB_COMMAND_CHARS),
                    check_runtime,
                ):
                    continue
                while reported_start is not None and reported_start < absolute_start:
                    reported_start = next(reported_starts, None)
                if reported_start == absolute_start:
                    continue
                # Structural ownership alone cannot establish provenance for
                # an embedded command the raw analyzer cannot report. Derived
                # marker views already carry their reconstructed findings.
                return StaticParserDiagnostic(
                    ParserCause.UNSUPPORTED_SYNTAX,
                    ParserCheck.COMMAND_STRING_EFFECTS,
                    start,
                    end,
                    ParserUncheckedScope.COMMAND_STRING,
                    location_precision="enclosing_expression",
                )
    if json_strings:
        # Raw substitution delimiters must survive Markdown's inline projection.
        json_operand_spans = [
            (start, end)
            for start, end in json_strings
            if "$(" not in content[start:end] and "`" not in content[start:end]
        ]
    if file_type == "markdown":
        hardware_apostrophe_ambiguity = [False]
        hardware_apostrophe_locations: list[tuple[int, int]] = []
        powershell_blocks: list[tuple[int, int]] = []
        content = _markdown_shell_text(
            content,
            check_runtime,
            complete_context=complete_context,
            word_apostrophes=word_apostrophes,
            shell_fence_ranges=shell_fence_ranges,
            hardware_apostrophe_ambiguity=hardware_apostrophe_ambiguity,
            hardware_apostrophe_locations=hardware_apostrophe_locations,
            powershell_blocks=powershell_blocks,
        )
        if hardware_apostrophe_ambiguity[0]:
            start, end = hardware_apostrophe_locations[0]
            return StaticParserDiagnostic(
                ParserCause.UNSUPPORTED_SYNTAX,
                ParserCheck.MARKDOWN_LITERAL_OWNERSHIP,
                start,
                end,
                ParserUncheckedScope.EXPRESSION,
            )
        if powershell_blocks:
            projected = list(content)
            # Separate code fences do not establish separate runtime scopes.
            allow_messages = _POWERSHELL_COMMAND_REBINDING_RE.search(content) is None
            for start, end in powershell_blocks:
                check_runtime()
                block_continuations: list[int] = []
                block = _powershell_shell_text(
                    content[start:end],
                    check_runtime,
                    continuations=block_continuations,
                    allow_messages=allow_messages,
                )
                block_diagnostics: list[StaticParserDiagnostic] = []
                if block_continuations and _has_destructive_command_string(
                    block, diagnostics=block_diagnostics
                ):
                    return block_diagnostics[0].shifted(start)
                projected[start:end] = block
            content = "".join(projected)
            powershell = _fenced_powershell_source(
                content, powershell_blocks, check_runtime, unresolved_executables
            )
    if unresolved_executables:
        return StaticParserDiagnostic(
            ParserCause.RUNTIME_DEPENDENT,
            ParserCheck.COMMAND_STRING_RUNTIME,
            unresolved_executables[0],
            len(content),
            ParserUncheckedScope.REMAINING_INPUT,
        )
    if powershell is not None and powershell.continuations:
        projected = list(content)
        for start, end in powershell.continuations:
            check_runtime()
            projected[start:end] = " " * (end - start)
        content = "".join(projected)
        # The continuation may have hidden destructive operands from the raw
        # finding parser. Keep coverage partial when the logical command now
        # exposes them; never claim those operands were fully inspected.
        continuation_diagnostics = []
        if _has_destructive_command_string(content, diagnostics=continuation_diagnostics):
            return continuation_diagnostics[0]
    # Diagnostic template delimiters belong to JavaScript only when the whole
    # source proves their assertion-message owner. Destructive checks stay raw.
    # TypeScript's postfix/type expressions need a separate lexical proof.
    shell_word_content = (
        assertion_template_shell_text(content, check_runtime)
        if file_type == "javascript" and complete_context
        else content
    )
    if file_type == "python" and complete_context:
        shell_word_content = python_browser_template_shell_text(content, check_runtime)
    if json_projection is not None:
        # Only validated host punctuation terminates command words. Every byte
        # inside strings stays in the parser and original finding views remain raw.
        shell_word_content = json_projection.text
    quote_limits = None
    if file_type == "shell" and complete_context:
        # A closing delimiter in proven shell source cannot open a new
        # command word. Nested substitutions retain their own lexical walk,
        # so their executable candidates remain independently inspected.
        structural_quote_closers = set()
        quote_limits = _shell_quoted_candidate_limits(
            content, check_runtime, quote_closers=structural_quote_closers
        )
    elif file_type == "markdown" and complete_context:
        quote_limits = {}
        for start, end in shell_fence_ranges:
            check_runtime()
            closers: set[int] = set()
            limits = _shell_quoted_candidate_limits(
                content[start:end], check_runtime, quote_closers=closers
            )
            quote_limits.update(
                {start + position: start + limit for position, limit in limits.items()}
            )
            assert structural_quote_closers is not None
            structural_quote_closers.update(start + position for position in closers)
    # Perl ``eval BLOCK`` is recognized from its clause-initial ``eval`` and the
    # adjacent brace alone, so unlike quote ownership it needs no full context.
    diagnostic = _has_shell_command_word_exhaustion(
        shell_word_content,
        check_runtime,
        structural_quote_closers=structural_quote_closers,
        structural_quote_openers=structural_quote_openers,
        python_source=python_source,
        javascript_source=javascript_source,
        powershell=powershell,
        perl_eval_blocks=file_type == "perl",
        quote_limits=quote_limits,
        file_type=file_type,
        complete_context=complete_context,
    )
    if diagnostic is not None:
        return diagnostic
    json_starts = [start for start, _ in json_operand_spans]
    covered_until = 0
    powershell_strings = _powershell_string_operands(powershell)
    decoded_starts = sorted(
        string.start for string in powershell_strings.values() if string.decoded
    )
    for command_start, body_start in _destructive_command_words(content):
        check_runtime()
        if command_start < covered_until:
            continue
        if powershell is not None and powershell.owns(command_start):
            continue
        # A recognized prose description cannot contribute parse exhaustion:
        # the same predicate below would discard it after parsing.  Short-circuit
        # it here to keep bounded-runtime accounting linear for documentation.
        if _is_root_glob_documentation(content, command_start, body_start):
            continue
        argument_diagnostics: list[StaticParserDiagnostic] = []
        tokens, command_end, exhausted = _bounded_shell_tokens(
            content,
            command_start,
            body_start,
            check_runtime=check_runtime,
            diagnostics=argument_diagnostics,
            powershell_strings=powershell_strings,
        )
        if exhausted:
            word = content[command_start:body_start]
            boundary = quote_limits.get(command_start) if quote_limits is not None else None
            index = bisect_right(json_starts, command_start) - 1
            if boundary is None and index >= 0:
                begin, end = json_operand_spans[index]
                if begin < command_start < body_start < end:
                    boundary = end - 1
            if (
                boundary is not None
                and not any(character in word for character in "$`'\"\\")
                and word.casefold().rsplit("/", 1)[-1] in _DESTRUCTIVE_COMMAND_BASENAMES
            ):
                # A proven enclosing closer terminates operands; reconstructed
                # command words retain their original uncertainty and bounds.
                tokens, command_end, exhausted = _bounded_shell_tokens(
                    content[command_start:boundary],
                    0,
                    body_start - command_start,
                    check_runtime=check_runtime,
                    diagnostics=(refined_diagnostics := []),
                )
                argument_diagnostics = [item.shifted(command_start) for item in refined_diagnostics]
                command_end += command_start
        covered_until = max(covered_until, command_end)
        if exhausted:
            diagnostic = argument_diagnostics[0]
            if (
                file_type == "shell"
                and complete_context
                and command_start == 0
                and diagnostic.cause is ParserCause.BOUNDARY_UNRESOLVED
                and diagnostic.check is ParserCheck.COMMAND_ARGUMENT_BOUNDARY
                and diagnostic.full_input_checked
                and not any(
                    character in content[command_start:body_start] for character in "$`'\"\\"
                )
                and content[command_start:body_start].casefold().rsplit("/", 1)[-1]
                in _DESTRUCTIVE_COMMAND_BASENAMES
            ):
                # A literal command at the beginning of complete shell input
                # has no unknown outer quote that could own its missing closer.
                diagnostic = replace(diagnostic, cause=ParserCause.MALFORMED_INPUT)
            return diagnostic
        brace_diagnostic = _brace_expansion_diagnostic(tokens, command_start, command_end)
        if brace_diagnostic is not None:
            return brace_diagnostic
        decoded_index = bisect_left(decoded_starts, body_start)
        if (
            decoded_index < len(decoded_starts)
            and decoded_starts[decoded_index] < command_end
            and (_has_destructive_root_glob(tokens) or _has_destructive_root_path(tokens))
        ):
            # Native escapes can assemble destructive flags the raw finding
            # parser did not see. Keep this reconstruction explicitly partial.
            return StaticParserDiagnostic(
                ParserCause.UNSUPPORTED_SYNTAX,
                ParserCheck.COMMAND_STRING_EFFECTS,
                command_start,
                command_end,
                ParserUncheckedScope.COMMAND_ARGUMENTS,
            )
    if powershell is not None:
        for start, end in powershell.invocations:
            check_runtime()
            diagnostic = _has_runtime_command_operand_exhaustion(
                content, start, end, check_runtime, powershell_strings=powershell_strings
            )
            if diagnostic is not None:
                return diagnostic
            tokens, command_end, _ = _bounded_shell_tokens(
                content,
                start,
                end,
                check_runtime=check_runtime,
                powershell_strings=powershell_strings,
            )
            skip_option_value = False
            for token in tokens:
                if skip_option_value:
                    skip_option_value = False
                    continue
                option = token.text.casefold()
                if not option.startswith("-") or option == "--":
                    break
                if (
                    option in {"-ec", "--command"}
                    or any(name.startswith(option[1:]) for name in ("command", "encodedcommand"))
                    or re.fullmatch(
                        r"-[abefhkmnoptuvxBCEHPTilrsO]*c[abefhkmnoptuvxBCEHPTilrsO]*", token.text
                    )
                ):
                    # Inspect actual code-selector options, not arbitrary
                    # native options containing 'c' such as -NonInteractive.
                    # Options after a script/subcommand operand are its data.
                    return StaticParserDiagnostic(
                        ParserCause.RUNTIME_DEPENDENT,
                        ParserCheck.COMMAND_STRING_RUNTIME,
                        start,
                        max(start + 1, command_end),
                        ParserUncheckedScope.COMMAND_STRING,
                    )
                skip_option_value = token.text in {"-o", "-O"} or option in {
                    "-executionpolicy",
                    "-ep",
                    "-inputformat",
                    "-if",
                    "-outputformat",
                    "-of",
                    "-windowstyle",
                    "-w",
                    "-workingdirectory",
                    "-wd",
                }
        for string in powershell.strings:
            check_runtime()
            if not string.invoked:
                # The projection already removed proven display-only messages.
                # Remaining values can be captured or reach rebound commands;
                # native delimiter ownership must not hide their shell limits.
                diagnostic = _has_bounded_parse_exhaustion(
                    string.value, check_runtime, file_type="powershell", complete_context=False
                )
                if diagnostic is not None:
                    return diagnostic.enclosed(string.start, string.end)
            # A PowerShell escape can assemble shell syntax absent from the
            # original bytes. Never turn such a command into a clean result
            # merely because the host string's boundaries are now understood.
            string_diagnostics: list[StaticParserDiagnostic] = []
            if string.decoded and _has_destructive_command_string(
                string.value, diagnostics=string_diagnostics
            ):
                return string_diagnostics[0].enclosed(string.start, string.end)
            if string.invoked:
                # A quoted command after & or dot-sourcing retains its actual
                # operands. Ordinary host strings are not shell programs.
                if string.operand_start is None:
                    return StaticParserDiagnostic(
                        ParserCause.BOUNDARY_UNRESOLVED,
                        ParserCheck.COMMAND_ARGUMENT_BOUNDARY,
                        string.start,
                        string.end,
                        ParserUncheckedScope.COMMAND_ARGUMENTS,
                    )
                end = min(len(content), string.operand_start + _ROOT_GLOB_COMMAND_CHARS)
                newline = content.find("\n", string.operand_start, end)
                if newline >= 0:
                    end = newline
                command = string.value + content[string.operand_start : end]
                command_diagnostics: list[StaticParserDiagnostic] = []
                if _has_destructive_command_string(command, diagnostics=command_diagnostics):
                    return command_diagnostics[0].enclosed(string.start, end)
                diagnostic = _has_bounded_parse_exhaustion(
                    command, check_runtime, file_type="shell", complete_context=False
                )
                if diagnostic is not None:
                    return diagnostic.enclosed(string.start, end)
    # A literal candidate outside a JSON string can consume it before the
    # whole-content parser reaches its structural opener. Recover each proven
    # string independently, without bypassing that parser's forward watermark
    # and reparsing overlapping suffixes. These raw spans are disjoint and
    # bounded by JSON validation. Reuse the whole-document projection so each
    # string retains its original fenced/literal or inline-code ownership.
    # Reinterpreting a string as standalone Markdown could mask shell backticks
    # that were literal inside its surrounding code fence. The projection keeps
    # source offsets, outer JSON quotes and escaped bytes unchanged.
    for start, end in json_strings:
        check_runtime()
        projected = content[start:end]
        diagnostic = _has_shell_command_word_exhaustion(
            projected,
            check_runtime,
            structural_quote_openers={0},
            structural_quote_closers={len(projected) - 1},
        )
        if diagnostic is not None:
            return diagnostic.shifted(start)
    return None


def _line_containing(content: str, start: int, end: int) -> str:
    """Return the full line containing a regex match."""
    line_start = content.rfind("\n", 0, start) + 1
    line_end = content.find("\n", end)
    if line_end == -1:
        line_end = len(content)
    return content[line_start:line_end]


def _is_assigned_value(content: str, start: int) -> bool:
    """Return whether the expression at *start* is the value of an assignment.

    Whitespace, line continuations and opening parentheses after the ``=`` are
    skipped, so ``old =  (os.umask(0))`` reads like ``old = os.umask(0)``.
    ``==``, ``!=``, ``<=`` and ``>=`` compare rather than assign.
    """
    index = start
    while index and content[index - 1] in " \t\r\n\\(":
        index -= 1
    if not index or content[index - 1] != "=":
        return False
    return index < 2 or content[index - 2] not in "=!<>"


def _bounded_context(content: str, match_start: int) -> str:
    """Return a fixed-size context centered on one direct shell call."""
    left = max(0, match_start - _MAX_DIRECT_SHELL_CONTEXT_CHARS // 2)
    right = min(len(content), left + _MAX_DIRECT_SHELL_CONTEXT_CHARS)
    left = max(0, right - _MAX_DIRECT_SHELL_CONTEXT_CHARS)
    return content[left:right].rstrip(LINE_BREAK_CHARS)


def _raw_classification_bounds(content: str, source_start: int) -> tuple[int, int]:
    """Return the raw scanner-window bounds that own *source_start*."""
    if len(content) <= static_runner.SECURITY_VIEW_WINDOW_CHARS:
        return 0, len(content)
    owned_start = (
        source_start // static_runner._RAW_WINDOW_OWNED_CHARS
    ) * static_runner._RAW_WINDOW_OWNED_CHARS
    owned_end = min(len(content), owned_start + static_runner._RAW_WINDOW_OWNED_CHARS)
    raw_start = max(0, owned_start - static_runner._WINDOW_OVERLAP_CHARS)
    raw_end = min(len(content), owned_end + static_runner._WINDOW_OVERLAP_CHARS)
    return raw_start, raw_end


def _classify_tm1(
    context: str,
    matched_text: str,
    matched_line: str,
    confidence: float,
    file_type: str,
    *,
    safe_context: bool | None = None,
    safe_matched_line: bool | None = None,
) -> tuple[Severity, float]:
    """Apply the existing TM1 contextual classification to one candidate."""
    if safe_context is None:
        safe_context = _is_safe_container_command(context) or _is_safe_dockerfile_idiom(
            context, matched_text
        )
    if safe_matched_line is None:
        safe_matched_line = _is_safe_cache_cleanup(matched_line)
    if safe_context or safe_matched_line:
        return Severity.LOW, min(confidence, 0.15)
    adjusted = (
        min(1.0, confidence + 0.1) if file_type in ("python", "shell", "javascript") else confidence
    )
    return Severity.HIGH, adjusted


def analyze(
    content: str,
    file_path: str,
    file_type: str,
    *,
    check_runtime: Callable[[], None] | None = None,
    _direct_shell_only: bool = False,
    defer_variable_reconciliation: bool = False,
) -> list[AnalyzerFinding]:
    """Analyze content for tool misuse patterns (TM1–TM3)."""
    runtime_check = check_runtime or (lambda: None)
    runtime_check()
    findings: list[AnalyzerFinding] = []
    locations = SourceLocationIndex(content, file_path)
    lines = content.splitlines()

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def line_number(start: int) -> int:
        return locations.line_and_column(start)[0]

    context_by_line: dict[int, str] = {}

    def ctx(start: int) -> str:
        line, column = locations.line_and_column(start)
        context = context_by_line.get(line)
        if context is None:
            context = get_context_from_lines(lines, line, column=column)
            context_by_line[line] = context
        return context

    tag = [PatternCategory.TOOL_MISUSE.value]
    tm1_findings_by_key: dict[tuple[int, str], AnalyzerFinding] = {}
    direct_safety_by_line: dict[int, tuple[bool, bool]] = {}

    variable_matches = (
        {}
        if _direct_shell_only
        else {
            (match.start(), match.end()): (match.group(1), match)
            for match in _variable_shell_matches(content)
        }
    )
    invisible_variable_matches: set[tuple[int, int]] = set()
    if file_type == "python" and variable_matches and not defer_variable_reconciliation:
        runtime_check()
        parsed = parse_python_source(content, file_path)
        runtime_check()
        ast_index = _build_variable_shell_ast_index(parsed)
        runtime_check()
        if ast_index is not None:
            for span, (_, match) in variable_matches.items():
                runtime_check()
                assignment_line = bisect_right(ast_index.line_character_starts, match.start(1))
                candidate = _resolve_variable_shell_candidate(ast_index, match, assignment_line)
                if candidate is not None and (
                    not candidate.visible
                    or _lexical_shell_has_counterevidence(
                        ast_index, parsed, candidate, receiver_trusted=None
                    )
                ):
                    invisible_variable_matches.add(span)

    shell_content = (
        _perl_literal_print_shell_text(content, runtime_check) if file_type == "perl" else None
    )
    for (
        match_start,
        match_end,
        matched_text,
        confidence,
        is_direct_shell_true,
        direct_shell_anchor,
        alternate_start,
        alternate_matched_text,
    ) in _tm1_candidates(
        content,
        direct_shell_only=_direct_shell_only,
        shell_content=shell_content,
        check_runtime=check_runtime,
        file_type=file_type,
    ):
        runtime_check()
        variable_match = variable_matches.get((match_start, match_end))
        if variable_match is not None and (match_start, match_end) in invisible_variable_matches:
            continue
        line_num = line_number(match_start)
        classification_context = ctx(match_start)
        context_text = (
            _bounded_context(content, match_start)
            if is_direct_shell_true
            else classification_context
        )
        matched = matched_text[:200]
        matched_line = _line_containing(content, match_start, match_end)

        if not is_direct_shell_true and _is_rest_route(matched_text, matched_line):
            continue

        if is_direct_shell_true:
            safety = direct_safety_by_line.get(line_num)
            if safety is None:
                safety = (
                    _is_safe_container_command(classification_context)
                    or _is_safe_dockerfile_idiom(classification_context, matched),
                    _is_safe_cache_cleanup(matched_line),
                )
                direct_safety_by_line[line_num] = safety
            sev, adj = _classify_tm1(
                classification_context,
                matched,
                matched_line,
                confidence,
                file_type,
                safe_context=safety[0],
                safe_matched_line=safety[1],
            )
        else:
            sev, adj = _classify_tm1(
                classification_context,
                matched,
                matched_line,
                confidence,
                file_type,
            )
        candidate_identity = (
            matched_text if is_direct_shell_true or variable_match is not None else matched
        )
        complete_identity_match = matched_text
        candidate_key = (
            line_num,
            sha256(" ".join(candidate_identity.strip().split()).encode()).hexdigest(),
        )
        existing = tm1_findings_by_key.get(candidate_key)
        if existing is not None:
            if adj > existing.confidence:
                existing.confidence = adj
                existing.severity = sev
            continue
        evidence: dict[str, object] = {static_runner._VIEW_START_EVIDENCE: match_start}
        if is_direct_shell_true:
            canonical_start = (
                alternate_start
                if alternate_start is not None and alternate_start < match_start
                else match_start
            )
            shell_match = (
                _DIRECT_SHELL_TRUE_VALUE.match(content, direct_shell_anchor)
                if direct_shell_anchor is not None
                else None
            )
            canonical_match = (
                content[canonical_start : shell_match.end()]
                if shell_match is not None
                else (
                    alternate_matched_text
                    if alternate_start is not None
                    and alternate_start < match_start
                    and alternate_matched_text is not None
                    else matched_text
                )
            )
            complete_identity_match = normalized_security_view(canonical_match).text
            reach_end = content.find(")", match_start)
            reach_terminated = reach_end != -1
            if not reach_terminated:
                reach_end = len(content)
            evidence.update(
                {
                    static_runner._PRESERVE_SOURCE_START_EVIDENCE: True,
                    static_runner._VIEW_REACH_END_EVIDENCE: reach_end,
                    static_runner._VIEW_REPLACEMENT_START_LIMIT_EVIDENCE: max(
                        0,
                        len(content) - len("True"),
                    ),
                    LEXICAL_REACH_TERMINATED_EVIDENCE: reach_terminated,
                    LEXICAL_DIRECT_SHELL_EVIDENCE: True,
                    LEXICAL_NORMALIZED_MATCH_EVIDENCE: normalized_security_prefix(
                        canonical_match,
                        200,
                    ),
                }
            )
            if direct_shell_anchor is not None:
                evidence[static_runner._VIEW_ANCHOR_EVIDENCE] = direct_shell_anchor
            if alternate_start is not None:
                evidence[static_runner._VIEW_ALTERNATE_START_EVIDENCE] = alternate_start
            if alternate_matched_text is not None:
                evidence[static_runner._ALTERNATE_MATCHED_TEXT_EVIDENCE] = alternate_matched_text[
                    :200
                ]
        if variable_match is not None and defer_variable_reconciliation:
            # Without Python bindings, keep only call-anchored windows for
            # true-prefixed names; coalescing drops one a direct owner reports.
            if file_type != "python" and not _is_true_prefixed_name(variable_match[0]):
                continue
            evidence[_VARIABLE_SHELL_FLAG_EVIDENCE] = variable_match[0]
            evidence[static_runner._PRESERVE_SOURCE_START_EVIDENCE] = True
            call_start = _variable_shell_call_start(variable_match[1])
            if call_start is not None:
                if variable_match[0].casefold().startswith("true"):
                    evidence[static_runner._VIEW_START_EVIDENCE] = call_start
                    # Like a direct owner, a call-anchored window keeps one
                    # identity across raw and normalized security views.
                    complete_identity_match = normalized_security_view(matched_text).text
                evidence[static_runner._VIEW_ANCHOR_EVIDENCE] = call_start
        finding = AnalyzerFinding(
            rule_id="TM1",
            message="Tool Parameter Abuse",
            severity=sev,
            location=loc(line_num),
            confidence=adj,
            tags=tag,
            context=context_text,
            matched_text=matched,
            complete_match=complete_identity_match,
            evidence=evidence,
        )
        tm1_findings_by_key[candidate_key] = finding
        findings.append(finding)

    if _direct_shell_only:
        return findings

    for pattern, confidence in TM2_PATTERNS:
        matches = (
            static_runner.iter_paragraph_matches
            if (pattern, confidence) in TM2_PROSE_PATTERNS
            else re.finditer
        )
        for match in matches(pattern, content, re.IGNORECASE | re.MULTILINE):
            runtime_check()
            line_num = line_number(match.start())
            context_text = ctx(match.start())
            matched = match.group(0)[:200]

            if _is_safe_dockerfile_idiom(context_text, matched):
                adj = min(confidence, 0.15)
                sev = Severity.LOW
            else:
                adj = confidence
                sev = Severity.HIGH
            findings.append(
                AnalyzerFinding(
                    rule_id="TM2",
                    message="Chaining Abuse",
                    severity=sev,
                    location=loc(line_num),
                    confidence=adj,
                    tags=tag,
                    context=context_text,
                    matched_text=matched,
                    complete_match=match.group(0),
                )
            )
    for pattern, confidence in _TM3_SHELL_PATTERNS if file_type == "shell" else TM3_PATTERNS:
        matches = (
            static_runner.iter_paragraph_matches
            if (pattern, confidence) in TM3_PROSE_PATTERNS
            else re.finditer
        )
        skip_assigned = (pattern, confidence) == _TM3_UMASK_CALL
        for match in matches(pattern, content, re.IGNORECASE | re.MULTILINE):
            runtime_check()
            if skip_assigned and _is_assigned_value(content, match.start()):
                continue
            line_num = line_number(match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="TM3",
                    message="Unsafe Defaults",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                    complete_match=match.group(0),
                )
            )
    # TM4: privileged K8s workload. Example filtering is delegated to the runner,
    # which keeps reference material because it is part of the skill. Findings in
    # top-level `references/` are tagged for triage only; the agent reads those
    # files as instructions, so confidence and score are left unchanged.
    reference_material = is_reference_material(file_path, file_type)
    tm4_tags = [*tag, "contextual-triage", "likely-benign-context"] if reference_material else tag
    for pattern, confidence in TM4_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            runtime_check()
            line_num = line_number(match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="TM4",
                    message="Privileged Kubernetes Workload",
                    severity=Severity.HIGH,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tm4_tags,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                    complete_match=match.group(0),
                )
            )
    return findings


def coalesce_path_findings(content: str, findings: list[Finding]) -> list[Finding]:
    """Select one owner per direct call without discarding private coordinates."""
    from . import static_python_shell_truthiness

    full_match_identities = {
        finding.finding_id: finding.match_fingerprint
        for finding in findings
        if finding.match_fingerprint is not None
    }
    bound_call_starts = {
        call_start
        for finding in findings
        if finding.evidence.get(static_python_shell_truthiness.BOUND_SHELL_EVIDENCE) is True
        and type(
            call_start := finding.evidence.get(
                static_python_shell_truthiness.BOUND_CALL_START_EVIDENCE
            )
        )
        is int
    }
    # The case-insensitive direct ``shell=True`` pattern also matches a name
    # spelled ``true``, so its lexical owner already reports that exact call
    # even when no Python AST can reconcile the variable window.
    true_name_call_starts = bound_call_starts | {
        coordinate
        for finding in findings
        if finding.evidence.get(LEXICAL_DIRECT_SHELL_EVIDENCE) is True
        for key in (
            static_runner._ABSOLUTE_START_EVIDENCE,
            static_runner._ABSOLUTE_ALTERNATE_START_EVIDENCE,
        )
        if type(coordinate := finding.evidence.get(key)) is int
    }
    findings = [
        finding
        for finding in findings
        if not (
            isinstance(variable_name := finding.evidence.get(_VARIABLE_SHELL_FLAG_EVIDENCE), str)
            and finding.evidence.get(static_runner._ABSOLUTE_ANCHOR_EVIDENCE)
            in (true_name_call_starts if variable_name.casefold() == "true" else bound_call_starts)
        )
    ]
    canonical_bound_calls: dict[tuple[str, int, str], set[int]] = {}
    for finding in findings:
        canonical = finding.evidence.get(
            static_python_shell_truthiness.BOUND_CANONICAL_FINGERPRINT_EVIDENCE
        )
        call_start = finding.evidence.get(static_python_shell_truthiness.BOUND_CALL_START_EVIDENCE)
        if isinstance(canonical, str) and type(call_start) is int:
            canonical_bound_calls.setdefault(
                (finding.file, finding.start_line, canonical),
                set(),
            ).add(call_start)
    colliding_bound_identities = {
        key for key, call_starts in canonical_bound_calls.items() if len(call_starts) > 1
    }

    lexical_records: list[tuple[Finding, int]] = []
    parents: dict[int, int] = {}

    def find(coordinate: int) -> int:
        parent = parents.setdefault(coordinate, coordinate)
        while parent != coordinate:
            grandparent = parents[parent]
            parents[coordinate] = grandparent
            coordinate = parent
            parent = grandparent
        return coordinate

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        canonical = min(first_root, second_root)
        parents[max(first_root, second_root)] = canonical

    for finding in findings:
        if finding.evidence.get(LEXICAL_DIRECT_SHELL_EVIDENCE) is not True:
            continue
        start = finding.evidence.get(static_runner._ABSOLUTE_START_EVIDENCE)
        if type(start) is int:
            find(start)
            alternate_start = finding.evidence.get(static_runner._ABSOLUTE_ALTERNATE_START_EVIDENCE)
            if type(alternate_start) is int:
                union(start, alternate_start)
            lexical_records.append((finding, start))

    lexical_groups: dict[int, list[Finding]] = {}
    for finding, start in lexical_records:
        lexical_groups.setdefault(find(start), []).append(finding)

    severity_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

    def lexical_owner_rank(
        finding: Finding,
        call_start: int,
    ) -> tuple[bool, bool, bool, int, float]:
        """Rank public classification donors independently of scan order."""
        return (
            finding.evidence.get(LEXICAL_RAW_OWNER_EVIDENCE) is True,
            "normalized-view" not in finding.tags,
            finding.evidence.get(static_runner._ABSOLUTE_START_EVIDENCE) == call_start,
            severity_rank.get(finding.severity, -1),
            finding.confidence,
        )

    def select_lexical_owner(group: list[Finding], call_start: int) -> Finding:
        return max(
            enumerate(group),
            key=lambda item: (*lexical_owner_rank(item[1], call_start), -item[0]),
        )[1]

    def fingerprint_source_rank(
        finding: Finding,
        call_start: int,
    ) -> tuple[bool, bool, int, bool, int]:
        """Rank immutable identity donors independently of scan order."""
        anchor = finding.evidence.get(static_runner._ABSOLUTE_ANCHOR_EVIDENCE)
        reach_end = finding.evidence.get(static_runner._ABSOLUTE_REACH_END_EVIDENCE)
        return (
            finding.evidence.get(static_runner._ABSOLUTE_START_EVIDENCE) == call_start,
            "normalized-view" in finding.tags,
            anchor if type(anchor) is int else -1,
            finding.evidence.get(LEXICAL_REACH_TERMINATED_EVIDENCE) is True,
            reach_end if type(reach_end) is int else -1,
        )

    def fingerprint_record_rank(
        record: dict[str, object],
        call_start: int,
    ) -> tuple[bool, bool, int, bool, int]:
        """Re-rank a retained identity donor at the current group coordinate."""
        anchor = record.get("anchor")
        reach_end = record.get("reach_end")
        return (
            record.get("start") == call_start,
            record.get("normalized") is True,
            anchor if type(anchor) is int else -1,
            record.get("reach_terminated") is True,
            reach_end if type(reach_end) is int else -1,
        )

    def classification_record(
        finding: Finding,
    ) -> dict[str, object]:
        """Snapshot the bounded public fields needed across cap finalizers."""
        return {
            "severity": finding.severity,
            "confidence": finding.confidence,
            "base_tags": tuple(finding.tags),
            "tags": tuple(finding.tags),
            "raw_owner": finding.evidence.get(LEXICAL_RAW_OWNER_EVIDENCE) is True,
            "raw_view": "normalized-view" not in finding.tags,
            "start": finding.evidence.get(static_runner._ABSOLUTE_START_EVIDENCE),
        }

    def classification_record_rank(
        record: dict[str, object],
        call_start: int,
    ) -> tuple[bool, bool, bool, int, float]:
        """Re-rank a retained public donor at the current group coordinate."""
        severity = record.get("severity")
        confidence = record.get("confidence")
        return (
            record.get("raw_owner") is True,
            record.get("raw_view") is True,
            record.get("start") == call_start,
            severity_rank.get(severity, -1) if isinstance(severity, str) else -1,
            (
                float(confidence)
                if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
                else -1.0
            ),
        )

    def select_fingerprint_source(group: list[Finding], call_start: int) -> Finding:
        """Prefer the canonical-coordinate normalized match, independent of scan order."""
        return max(
            enumerate(group),
            key=lambda item: (*fingerprint_source_rank(item[1], call_start), -item[0]),
        )[1]

    def lexical_fingerprint(finding: Finding) -> str | None:
        """Hash immutable lexical text instead of a prior coalescing override."""
        canonical_match = finding.evidence.get(LEXICAL_NORMALIZED_MATCH_EVIDENCE)
        if not isinstance(canonical_match, str) and finding.matched_text is None:
            return finding.fingerprint()
        if not isinstance(canonical_match, str):
            canonical_match = normalized_security_prefix(finding.matched_text or "", 200)
        normalized = " ".join(canonical_match.strip().split())
        return sha256(f"{finding.rule_id}\x1f{normalized}".encode()).hexdigest()

    def actual_shell_is_reachable(
        candidate: Finding,
        shell_value_start: int,
        direct_match_end: int,
    ) -> bool | None:
        """Compare an AST shell value with this candidate's exact regex reach."""
        reach_end = candidate.evidence.get(static_runner._ABSOLUTE_REACH_END_EVIDENCE)
        reach_terminated = candidate.evidence.get(LEXICAL_REACH_TERMINATED_EVIDENCE)
        if type(reach_end) is not int or type(reach_terminated) is not bool:
            return None
        replacement_start_limit = candidate.evidence.get(
            static_runner._ABSOLUTE_REPLACEMENT_START_LIMIT_EVIDENCE
        )
        if type(replacement_start_limit) is int and shell_value_start > replacement_start_limit:
            replacement_recovery_start = candidate.evidence.get(
                static_runner._ABSOLUTE_REPLACEMENT_RECOVERY_START_EVIDENCE
            )
            if type(replacement_recovery_start) is not int or not any(
                type(candidate_start) is int and candidate_start >= replacement_recovery_start
                for candidate_start in (
                    candidate.evidence.get(static_runner._ABSOLUTE_START_EVIDENCE),
                    candidate.evidence.get(static_runner._ABSOLUTE_ALTERNATE_START_EVIDENCE),
                )
            ):
                return False
        if reach_terminated:
            return reach_end > shell_value_start
        return reach_end >= direct_match_end

    lexical_owners: dict[int, Finding] = {}
    lexical_member_starts: dict[str, int] = {}
    for call_start, group in lexical_groups.items():
        owner = select_lexical_owner(group, call_start)
        canonical = select_fingerprint_source(group, call_start)
        if "normalized-view" not in owner.tags:
            owner.evidence[LEXICAL_RAW_OWNER_EVIDENCE] = True
        owner_start = owner.evidence.get(static_runner._ABSOLUTE_START_EVIDENCE)
        if owner is not canonical:
            # Coalescing can run repeatedly while a finding cap is being
            # finalized. Persist the canonical identity coordinates on the
            # retained public owner even when both candidates start together,
            # otherwise a later pass re-hashes the earlier lexical lookalike.
            canonical_match = canonical.evidence.get(LEXICAL_NORMALIZED_MATCH_EVIDENCE)
            if isinstance(canonical_match, str):
                owner.evidence[LEXICAL_NORMALIZED_MATCH_EVIDENCE] = canonical_match
            for evidence_key in (
                static_runner._ABSOLUTE_ANCHOR_EVIDENCE,
                static_runner._ABSOLUTE_REACH_END_EVIDENCE,
                static_runner._ABSOLUTE_REPLACEMENT_START_LIMIT_EVIDENCE,
                static_runner._ABSOLUTE_REPLACEMENT_RECOVERY_START_EVIDENCE,
                LEXICAL_REACH_TERMINATED_EVIDENCE,
            ):
                if evidence_key in canonical.evidence:
                    owner.evidence[evidence_key] = canonical.evidence[evidence_key]
                else:
                    owner.evidence.pop(evidence_key, None)
        if (
            owner is not canonical
            and type(owner_start) is int
            and owner_start != call_start
            and canonical.matched_text is not None
        ):
            # A raw bare-Popen owner can be the only classification context
            # that existed before a cross-window normalized qualifier was
            # reconstructed. Preserve that public owner while borrowing only
            # the canonical call identity from the derived candidate.
            owner.matched_text = canonical.matched_text
            owner.finding = canonical.finding
            owner.evidence[static_runner._ABSOLUTE_ALTERNATE_START_EVIDENCE] = call_start
            if "normalized-view" in canonical.tags and "normalized-view" not in owner.tags:
                owner.tags.append("normalized-view")
        lexical_canonical_fingerprint = lexical_fingerprint(canonical)
        complete_identity = full_match_identities.get(canonical.finding_id)
        owner.match_fingerprint = (
            complete_identity
            if isinstance(complete_identity, str)
            else lexical_canonical_fingerprint
        )
        lexical_owners[call_start] = owner
        lexical_member_starts.update((candidate.finding_id, call_start) for candidate in group)

    ast_call_starts: set[int] = set()
    ast_owners: dict[str, Finding] = {}
    for finding in findings:
        if finding.evidence.get(static_python_shell_truthiness.BOUND_SHELL_EVIDENCE) is not True:
            continue
        canonical_fingerprint = finding.evidence.get(
            static_python_shell_truthiness.BOUND_CANONICAL_FINGERPRINT_EVIDENCE
        )
        lexical_identity = finding.evidence.get(LEXICAL_BOUND_IDENTITY_EVIDENCE)
        previously_reachable = finding.evidence.get(LEXICAL_BOUND_REACHABLE_EVIDENCE) is True
        complete_identity = full_match_identities.get(finding.finding_id)
        canonical_collision = (
            isinstance(canonical_fingerprint, str)
            and (
                finding.file,
                finding.start_line,
                canonical_fingerprint,
            )
            in colliding_bound_identities
        )
        fingerprint = (
            lexical_identity
            if isinstance(lexical_identity, str)
            else complete_identity
            if canonical_collision and isinstance(complete_identity, str)
            else canonical_fingerprint
        )
        if canonical_collision and isinstance(fingerprint, str):
            finding.evidence[LEXICAL_BOUND_IDENTITY_EVIDENCE] = fingerprint
        if isinstance(fingerprint, str):
            finding.match_fingerprint = fingerprint
        ast_call_start = finding.evidence.get(
            static_python_shell_truthiness.BOUND_CALL_START_EVIDENCE
        )
        ast_call_end = finding.evidence.get(static_python_shell_truthiness.BOUND_CALL_END_EVIDENCE)
        direct_match_end = finding.evidence.get(
            static_python_shell_truthiness.BOUND_DIRECT_MATCH_END_EVIDENCE
        )
        popen_start = finding.evidence.get(
            static_python_shell_truthiness.BOUND_POPEN_START_EVIDENCE
        )
        shell_anchor = finding.evidence.get(
            static_python_shell_truthiness.BOUND_SHELL_ANCHOR_EVIDENCE
        )
        ast_owner: Finding | None = None
        if type(ast_call_start) is int:
            ast_call_starts.add(ast_call_start)
            lexical_coordinates = [ast_call_start]
            if type(popen_start) is int:
                # Parenthesized, spaced, and explicitly continued receivers
                # can only be recognized lexically from the method token. That
                # token still belongs to this parsed call even when a quoted
                # trailing ``shell=True`` produces a different shell anchor.
                ast_call_starts.add(popen_start)
                lexical_coordinates.append(popen_start)
            coordinate_groups = [
                (coordinate, lexical_groups[coordinate])
                for coordinate in lexical_coordinates
                if coordinate in lexical_groups
            ]
            exact_group = [
                candidate
                for _, group in coordinate_groups
                for candidate in group
                if type(shell_anchor) is int
                and candidate.evidence.get(static_runner._ABSOLUTE_ANCHOR_EVIDENCE) == shell_anchor
            ]
            if exact_group:
                ast_owner = select_lexical_owner(exact_group, ast_call_start)
                ast_owners[finding.finding_id] = ast_owner
                finding.evidence.pop(LEXICAL_BOUND_IDENTITY_EVIDENCE, None)
                finding.evidence.pop(LEXICAL_BOUND_IDENTITY_SOURCE_EVIDENCE, None)
                finding.evidence.pop(LEXICAL_BOUND_CLASSIFICATION_EVIDENCE, None)
            elif type(shell_anchor) is int and type(direct_match_end) is int:
                # The legacy regex stops at the first nested ``)``. If an
                # earlier parenthesized argument contains ``shell=True``, a
                # direct literal is therefore fingerprinted from that retained
                # same-call prefix in every file type. Keep a bound-name call
                # on that established identity without borrowing its public
                # text, tags, or classification. Exact coordinates exclude a
                # separate nested call that happens to appear in the arguments.
                same_call_candidates = [
                    candidate for _, group in coordinate_groups for candidate in group
                ]
                shell_value_start = direct_match_end - len("True")
                reachability = [
                    actual_shell_is_reachable(
                        candidate,
                        shell_value_start,
                        direct_match_end,
                    )
                    for candidate in same_call_candidates
                ]
                actual_shell_is_unreachable = bool(same_call_candidates) and all(
                    type(
                        candidate_anchor := candidate.evidence.get(
                            static_runner._ABSOLUTE_ANCHOR_EVIDENCE
                        )
                    )
                    is int
                    and ast_call_start <= candidate_anchor < shell_anchor
                    and (type(ast_call_end) is not int or candidate_anchor < ast_call_end)
                    and reachable is False
                    for candidate, reachable in zip(
                        same_call_candidates,
                        reachability,
                        strict=True,
                    )
                )
                if previously_reachable or any(reachable is True for reachable in reachability):
                    finding.evidence[LEXICAL_BOUND_REACHABLE_EVIDENCE] = True
                    finding.evidence.pop(LEXICAL_BOUND_IDENTITY_EVIDENCE, None)
                    finding.evidence.pop(LEXICAL_BOUND_IDENTITY_SOURCE_EVIDENCE, None)
                    finding.evidence.pop(LEXICAL_BOUND_CLASSIFICATION_EVIDENCE, None)
                    if isinstance(canonical_fingerprint, str):
                        finding.match_fingerprint = canonical_fingerprint
                elif actual_shell_is_unreachable:
                    identity_coordinate = min(coordinate for coordinate, _ in coordinate_groups)
                    identity_source = select_fingerprint_source(
                        same_call_candidates,
                        identity_coordinate,
                    )
                    public_owner = select_lexical_owner(
                        same_call_candidates,
                        identity_coordinate,
                    )
                    legacy_fingerprint = lexical_fingerprint(identity_source)
                    if isinstance(legacy_fingerprint, str):
                        identity_record: dict[str, object] = {
                            "start": identity_source.evidence.get(
                                static_runner._ABSOLUTE_START_EVIDENCE
                            ),
                            "normalized": "normalized-view" in identity_source.tags,
                            "anchor": identity_source.evidence.get(
                                static_runner._ABSOLUTE_ANCHOR_EVIDENCE
                            ),
                            "reach_terminated": identity_source.evidence.get(
                                LEXICAL_REACH_TERMINATED_EVIDENCE
                            ),
                            "reach_end": identity_source.evidence.get(
                                static_runner._ABSOLUTE_REACH_END_EVIDENCE
                            ),
                        }
                        prior_identity_record = finding.evidence.get(
                            LEXICAL_BOUND_IDENTITY_SOURCE_EVIDENCE
                        )
                        if not isinstance(prior_identity_record, dict) or (
                            fingerprint_record_rank(identity_record, identity_coordinate)
                            > fingerprint_record_rank(
                                prior_identity_record,
                                identity_coordinate,
                            )
                        ):
                            finding.evidence[LEXICAL_BOUND_IDENTITY_EVIDENCE] = legacy_fingerprint
                            finding.evidence[LEXICAL_BOUND_IDENTITY_SOURCE_EVIDENCE] = (
                                identity_record
                            )
                        persisted_identity = finding.evidence.get(LEXICAL_BOUND_IDENTITY_EVIDENCE)
                        if isinstance(persisted_identity, str):
                            finding.match_fingerprint = persisted_identity

                        prior_classification = finding.evidence.get(
                            LEXICAL_BOUND_CLASSIFICATION_EVIDENCE
                        )
                        current_classification = classification_record(public_owner)
                        if not isinstance(prior_classification, dict) or (
                            classification_record_rank(
                                current_classification,
                                identity_coordinate,
                            )
                            > classification_record_rank(
                                prior_classification,
                                identity_coordinate,
                            )
                        ):
                            prior_classification = current_classification

                        base_tags = prior_classification.get("base_tags")
                        projected_tags = (
                            list(base_tags)
                            if isinstance(base_tags, (list, tuple))
                            and all(isinstance(tag, str) for tag in base_tags)
                            else []
                        )
                        persisted_identity_record = finding.evidence.get(
                            LEXICAL_BOUND_IDENTITY_SOURCE_EVIDENCE
                        )
                        if (
                            isinstance(persisted_identity_record, dict)
                            and persisted_identity_record.get("normalized") is True
                            and persisted_identity_record.get("start") == identity_coordinate
                            and prior_classification.get("start") != identity_coordinate
                            and "normalized-view" not in projected_tags
                        ):
                            # This is the same projection performed above when
                            # a raw bare-Popen owner and its normalized qualified
                            # identity coexist in one pass. Re-derive it when
                            # those bounded donors arrive in separate cap passes.
                            projected_tags.append("normalized-view")
                        prior_classification["tags"] = tuple(projected_tags)
                        finding.evidence[LEXICAL_BOUND_CLASSIFICATION_EVIDENCE] = (
                            prior_classification
                        )
        if ast_owner is not None and isinstance(canonical_fingerprint, str):
            ast_owner.match_fingerprint = canonical_fingerprint

    reconciled: list[Finding] = []
    emitted_call_starts: set[int] = set()
    for finding in findings:
        is_ast = finding.evidence.get(static_python_shell_truthiness.BOUND_SHELL_EVIDENCE) is True
        ast_call_start = finding.evidence.get(
            static_python_shell_truthiness.BOUND_CALL_START_EVIDENCE
        )
        ast_owner = ast_owners.get(finding.finding_id)
        if is_ast:
            if type(ast_call_start) is int and ast_call_start in emitted_call_starts:
                continue
            if ast_owner is not None:
                reconciled.append(ast_owner)
            else:
                reconciled.append(finding)
            if type(ast_call_start) is int:
                emitted_call_starts.add(ast_call_start)
            continue
        lexical_start = lexical_member_starts.get(finding.finding_id)
        if lexical_start is not None:
            if lexical_start not in ast_call_starts and lexical_start not in emitted_call_starts:
                reconciled.append(lexical_owners[lexical_start])
                emitted_call_starts.add(lexical_start)
            continue
        reconciled.append(finding)

    source_line_starts = (0, *(match.end() for match in LOGICAL_LINE_BREAK.finditer(content)))
    tm1_slots: list[int] = []
    tm1_by_source: list[tuple[int, int, Finding]] = []
    for index, finding in enumerate(reconciled):
        if finding.rule_id != "TM1":
            continue
        bound_start = finding.evidence.get(static_python_shell_truthiness.BOUND_CALL_START_EVIDENCE)
        lexical_coordinate = finding.evidence.get(static_runner._ABSOLUTE_START_EVIDENCE)
        source_start = (
            bound_start
            if type(bound_start) is int
            else lexical_coordinate
            if type(lexical_coordinate) is int
            else None
        )
        if source_start is None:
            line_index = min(max(finding.start_line - 1, 0), len(source_line_starts) - 1)
            source_start = source_line_starts[line_index]
            if type(finding.start_column) is int and finding.start_column > 0:
                source_start += finding.start_column
        tm1_slots.append(index)
        tm1_by_source.append((source_start, index, finding))
    for slot, (_, _, finding) in zip(
        tm1_slots,
        sorted(tm1_by_source, key=lambda item: (item[0], item[1])),
        strict=True,
    ):
        reconciled[slot] = finding
    public: list[Finding] = []
    seen_direct_calls: set[tuple[str, int, str | None]] = set()
    for finding in reconciled:
        is_direct_shell = (
            finding.evidence.get(LEXICAL_DIRECT_SHELL_EVIDENCE) is True
            or finding.evidence.get(static_python_shell_truthiness.BOUND_SHELL_EVIDENCE) is True
        )
        if finding.rule_id != "TM1" or not is_direct_shell:
            public.append(finding)
            continue
        key = (finding.file, finding.start_line, finding.fingerprint())
        if key in seen_direct_calls:
            continue
        seen_direct_calls.add(key)
        public.append(finding)
    return public


def _owning_raw_window(
    content: str,
    source_start: int,
) -> tuple[int, int, int, int]:
    """Return the raw and owned bounds used by the ordinary window scanner."""
    if len(content) <= static_runner.SECURITY_VIEW_WINDOW_CHARS:
        return 0, len(content), 0, len(content)
    owned_start = (
        source_start // static_runner._RAW_WINDOW_OWNED_CHARS
    ) * static_runner._RAW_WINDOW_OWNED_CHARS
    owned_end = min(len(content), owned_start + static_runner._RAW_WINDOW_OWNED_CHARS)
    raw_start = max(0, owned_start - static_runner._WINDOW_OVERLAP_CHARS)
    raw_end = min(len(content), owned_end + static_runner._WINDOW_OVERLAP_CHARS)
    return raw_start, raw_end, owned_start, owned_end


def _window_popen_qualifiers(
    content: str,
    popen_starts: set[int],
    check_runtime: Callable[[], None],
    source_context: static_runner._WindowSourceContext,
) -> dict[int, tuple[int, int, str, str, bool, bool]]:
    """Replay each owning bounded window once for retained Popen anchors."""
    targets_by_window: dict[tuple[int, int, int, int], set[int]] = {}
    for popen_start in popen_starts:
        bounds = _owning_raw_window(content, popen_start)
        targets_by_window.setdefault(bounds, set()).add(popen_start)

    qualifiers: dict[int, tuple[int, int, str, str, bool, bool]] = {}
    qualified_pattern = DIRECT_SHELL_TRUE_PATTERNS[0][0]
    for (raw_start, raw_end, owned_start, owned_end), targets in targets_by_window.items():
        check_runtime()
        raw_window = content[raw_start:raw_end]
        context_prefix = static_runner._markdown_context_prefix(
            content,
            raw_start,
            raw_end,
            source_context.fence_states,
            source_context.fence_transitions,
        )
        for raw_view in security_text_views(context_prefix + raw_window):
            full_view = static_runner._window_view_with_markdown_context(
                raw_view,
                len(context_prefix),
            )
            check_runtime()
            for view in static_runner._bounded_view_slices(full_view):
                check_runtime()
                for match in re.finditer(
                    qualified_pattern, view.text, re.IGNORECASE | re.MULTILINE
                ):
                    method = re.match(
                        r"subprocess\.(?P<method>\w+)",
                        match.group(0),
                        re.IGNORECASE,
                    )
                    if method is None or method.group("method").casefold() != "popen":
                        continue
                    derived_popen_start = match.start() + method.start("method")
                    popen_start = raw_start + view.source_offset(derived_popen_start)
                    if popen_start not in targets:
                        continue
                    qualifier_start = raw_start + view.source_offset(match.start())
                    qualifier_end = raw_start + view.source_offset(derived_popen_start - 1) + 1
                    qualifier_text = view.text[match.start() : derived_popen_start]
                    raw_match_end = raw_start + view.source_offset(match.end() - 1) + 1
                    requires_derived_match = (
                        view.name != "raw"
                        and match.group(0) != content[qualifier_start:raw_match_end]
                    )
                    candidate = (
                        qualifier_start,
                        qualifier_end,
                        qualifier_text,
                        match.group(0),
                        requires_derived_match,
                        owned_start <= qualifier_start < owned_end,
                    )
                    previous = qualifiers.get(popen_start)
                    if previous is None or (candidate[4], candidate[5]) > (
                        previous[4],
                        previous[5],
                    ):
                        qualifiers[popen_start] = candidate
    return qualifiers


def _cross_window_subprocess_qualifiers(
    content: str,
    popen_anchors: dict[int, int],
    check_runtime: Callable[[], None],
) -> dict[int, tuple[int, int, str, str, bool]]:
    """Recover retained qualifiers while replaying each shared projection once."""
    qualifiers: dict[int, tuple[int, int, str, str, bool]] = {}
    qualified_pattern = DIRECT_SHELL_TRUE_PATTERNS[0][0]
    targets_by_run: dict[tuple[int, int], dict[int, int]] = {}
    runs_by_anchor = static_runner._continuity_runs_for_anchors(
        content,
        set(popen_anchors.values()),
        check_runtime,
    )
    for popen_start, anchor in popen_anchors.items():
        run = runs_by_anchor.get(anchor)
        if run is not None:
            targets_by_run.setdefault(run, {})[popen_start] = anchor

    for _, projection_target_anchors in sorted(targets_by_run.items()):
        projection_targets = set(projection_target_anchors)
        projection = static_runner._anchored_continuity_view(
            content,
            min(projection_target_anchors.values()),
            check_runtime,
        )
        if projection is None or not projection.text or projection.source_offsets is None:
            continue

        for full_view in security_text_views(projection.text):
            for view in static_runner._bounded_view_slices(full_view):
                check_runtime()
                for match in re.finditer(
                    qualified_pattern,
                    view.text,
                    re.IGNORECASE | re.MULTILINE,
                ):
                    method = re.match(
                        r"subprocess\.(?P<method>\w+)",
                        match.group(0),
                        re.IGNORECASE,
                    )
                    if method is None or method.group("method").casefold() != "popen":
                        continue
                    derived_popen_start = match.start() + method.start("method")
                    projected_popen_start = view.source_offset(derived_popen_start)
                    raw_popen_start = projection.source_offset(projected_popen_start)
                    if raw_popen_start not in projection_targets or raw_popen_start in qualifiers:
                        continue
                    projected_start = view.source_offset(match.start())
                    projected_end = view.source_offset(derived_popen_start - 1)
                    qualifiers[raw_popen_start] = (
                        projection.source_offset(projected_start),
                        projection.source_offset(projected_end) + 1,
                        view.text[match.start() : derived_popen_start],
                        match.group(0),
                        True,
                    )
    return qualifiers


def reconcile_retained_findings(
    content: str,
    findings: list[Finding],
    check_runtime: Callable[[], None],
    source_context: static_runner._WindowSourceContext,
) -> None:
    """Finalize retained bare-Popen identity without discovering new findings."""
    retained: list[tuple[Finding, int, int]] = []
    for finding in findings:
        if (
            finding.evidence.get(LEXICAL_DIRECT_SHELL_EVIDENCE) is not True
            or re.match(r"Popen\b", finding.matched_text or "", re.IGNORECASE) is None
        ):
            continue
        coordinates = [
            coordinate
            for key in (
                static_runner._ABSOLUTE_START_EVIDENCE,
                static_runner._ABSOLUTE_ALTERNATE_START_EVIDENCE,
            )
            if type(coordinate := finding.evidence.get(key)) is int
        ]
        if coordinates:
            popen_start = max(coordinates)
            shell_anchor = finding.evidence.get(static_runner._ABSOLUTE_ANCHOR_EVIDENCE)
            retained.append(
                (
                    finding,
                    popen_start,
                    shell_anchor if type(shell_anchor) is int else popen_start,
                )
            )

    window_qualifiers = _window_popen_qualifiers(
        content,
        {popen_start for _, popen_start, _ in retained},
        check_runtime,
        source_context,
    )
    cross_window_qualifiers = _cross_window_subprocess_qualifiers(
        content,
        {popen_start: continuity_anchor for _, popen_start, continuity_anchor in retained},
        check_runtime,
    )
    for finding, popen_start, _ in retained:
        check_runtime()
        window_qualifier = window_qualifiers.get(popen_start)
        cross_window_qualifier = cross_window_qualifiers.get(popen_start)
        if cross_window_qualifier is not None:
            qualifier = cross_window_qualifier
            update_public_match = True
        elif window_qualifier is not None:
            (
                qualifier_start,
                qualifier_end,
                qualifier_text,
                canonical_match,
                normalized_view,
                qualifier_owned,
            ) = window_qualifier
            update_public_match = normalized_view or qualifier_owned
            qualifier = (
                qualifier_start,
                qualifier_end,
                qualifier_text,
                canonical_match,
                normalized_view,
            )
        else:
            qualifier = None
            update_public_match = False
        if qualifier is None:
            continue
        qualifier_start, _, _, canonical_match, normalized_view = qualifier
        canonical_projection = normalized_security_view(canonical_match).text
        canonical_preview = canonical_projection[:200]
        finding.evidence[LEXICAL_NORMALIZED_MATCH_EVIDENCE] = canonical_preview
        if update_public_match:
            finding.matched_text = canonical_preview
            finding.finding = canonical_preview
        finding.evidence[static_runner._ABSOLUTE_ALTERNATE_START_EVIDENCE] = qualifier_start
        finding.match_fingerprint = sha256(
            f"{finding.rule_id}\x1f{' '.join(canonical_projection.strip().split())}".encode()
        ).hexdigest()
        if normalized_view and "normalized-view" not in finding.tags:
            finding.tags.append("normalized-view")


def _lexical_call_start(finding: Finding) -> int | None:
    """Return a lexical owner's canonical call coordinate."""
    coordinates = (
        coordinate
        for key in (
            static_runner._ABSOLUTE_START_EVIDENCE,
            static_runner._ABSOLUTE_ALTERNATE_START_EVIDENCE,
        )
        if type(coordinate := finding.evidence.get(key)) is int
    )
    return min(coordinates, default=None)


def _internal_evidence_keys() -> tuple[str, ...]:
    """Return private reconciliation keys that must not escape public findings."""
    from . import static_python_shell_truthiness

    return (
        _VARIABLE_SHELL_FLAG_EVIDENCE,
        LEXICAL_DIRECT_SHELL_EVIDENCE,
        LEXICAL_RAW_OWNER_EVIDENCE,
        LEXICAL_NORMALIZED_MATCH_EVIDENCE,
        LEXICAL_BOUND_IDENTITY_EVIDENCE,
        LEXICAL_BOUND_IDENTITY_SOURCE_EVIDENCE,
        LEXICAL_BOUND_CLASSIFICATION_EVIDENCE,
        LEXICAL_REACH_TERMINATED_EVIDENCE,
        LEXICAL_BOUND_REACHABLE_EVIDENCE,
        static_python_shell_truthiness.BOUND_SHELL_EVIDENCE,
        static_python_shell_truthiness.BOUND_CALL_START_EVIDENCE,
        static_python_shell_truthiness.BOUND_CALL_END_EVIDENCE,
        static_python_shell_truthiness.BOUND_SHELL_ANCHOR_EVIDENCE,
        static_python_shell_truthiness.BOUND_CANONICAL_FINGERPRINT_EVIDENCE,
        static_python_shell_truthiness.BOUND_NORMALIZED_VIEW_EVIDENCE,
        static_python_shell_truthiness.BOUND_CLASSIFICATION_MATCH_EVIDENCE,
        static_python_shell_truthiness.BOUND_DIRECT_MATCH_END_EVIDENCE,
        static_python_shell_truthiness.BOUND_POPEN_START_EVIDENCE,
        static_python_shell_truthiness.BOUND_DIRECT_OWNER_START_EVIDENCE,
        static_python_shell_truthiness.BOUND_SHELL_VALUE_START_EVIDENCE,
        static_python_shell_truthiness.BOUND_SHELL_VALUE_END_EVIDENCE,
        static_python_shell_truthiness.DIRECT_LITERAL_METADATA_EVIDENCE,
        static_runner._VIEW_START_EVIDENCE,
        static_runner._VIEW_ANCHOR_EVIDENCE,
        static_runner._VIEW_ALTERNATE_START_EVIDENCE,
        static_runner._VIEW_REACH_END_EVIDENCE,
        static_runner._VIEW_REPLACEMENT_START_LIMIT_EVIDENCE,
        static_runner._SOURCE_START_EVIDENCE,
        static_runner._SOURCE_ANCHOR_EVIDENCE,
        static_runner._SOURCE_ALTERNATE_START_EVIDENCE,
        static_runner._SOURCE_REACH_END_EVIDENCE,
        static_runner._SOURCE_REPLACEMENT_START_LIMIT_EVIDENCE,
        static_runner._SOURCE_REPLACEMENT_RECOVERY_START_EVIDENCE,
        static_runner._PRESERVE_SOURCE_START_EVIDENCE,
        static_runner._ABSOLUTE_START_EVIDENCE,
        static_runner._ABSOLUTE_ANCHOR_EVIDENCE,
        static_runner._ABSOLUTE_ALTERNATE_START_EVIDENCE,
        static_runner._ABSOLUTE_REACH_END_EVIDENCE,
        static_runner._ABSOLUTE_REPLACEMENT_START_LIMIT_EVIDENCE,
        static_runner._ABSOLUTE_REPLACEMENT_RECOVERY_START_EVIDENCE,
        static_runner._ALTERNATE_MATCHED_TEXT_EVIDENCE,
    )


def cleanup_path_findings(findings: list[Finding]) -> list[Finding]:
    """Strip private reconciliation state when the analysis deadline has expired."""
    internal_keys = _internal_evidence_keys()
    for finding in findings:
        for key in internal_keys:
            finding.evidence.pop(key, None)
    return findings


class _DirectShellReplayLexical:
    """Lexical-only facade used for direct-equivalent bounded windows."""

    USES_PYTHON_SOURCE_TYPE = True

    @staticmethod
    def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
        return analyze(
            content,
            file_path,
            file_type,
            _direct_shell_only=True,
        )


def _bound_direct_replays(
    content: str,
    findings: list[Finding],
    python_ast: ParsedPythonFile | None,
    *,
    started_at: float | None = None,
    timeout_seconds: float | None = None,
) -> dict[int, Finding]:
    """Replay only retained owners in one cap-independent virtual artifact."""
    from . import static_python_shell_truthiness

    if python_ast is None or python_ast.tree is None:
        return {}
    retained_call_starts: set[int] = set()
    file_path = "<unknown>"
    for finding in findings:
        if (
            finding.evidence.get(static_python_shell_truthiness.BOUND_SHELL_EVIDENCE) is not True
            or finding.evidence.get(static_python_shell_truthiness.DIRECT_LITERAL_METADATA_EVIDENCE)
            is True
        ):
            continue
        call_start = finding.evidence.get(static_python_shell_truthiness.BOUND_CALL_START_EVIDENCE)
        if type(call_start) is int:
            retained_call_starts.add(call_start)
            file_path = finding.file
    if not retained_call_starts:
        return {}

    budget_started_at = time.monotonic() if started_at is None else started_at
    runtime_limit = static_runner.MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT
    if timeout_seconds is not None:
        runtime_limit = min(runtime_limit, max(0.0, timeout_seconds))
    replay_budget = static_runner._FindingBudget(
        max_findings=static_runner.MAX_FILE_CHARS,
        started_at=budget_started_at,
        deadline=budget_started_at + runtime_limit,
        clock=time.monotonic,
    )
    try:
        specs = static_python_shell_truthiness.bound_shell_metadata(
            python_ast,
            file_path,
            check_runtime=replay_budget.check_runtime,
        )
    except static_runner._StaticResourceLimitError:
        return {}
    if not specs:
        return {}

    cursor = 0
    shift = 0
    virtual_parts: list[str] = []
    virtual_coordinates_by_call: dict[int, tuple[int, int | None, int]] = {}
    try:
        for spec in specs:
            if spec.value_start < cursor:
                # The supported straight-line grammar cannot overlap shell-name
                # spans. Stay conservative if malformed coordinates ever do.
                return {}
            replay_budget.check_runtime()
            virtual_parts.append(content[cursor : spec.value_start])
            virtual_call_start = spec.call_start + shift
            virtual_coordinates_by_call[spec.call_start] = (
                virtual_call_start,
                spec.popen_start + shift if spec.popen_start is not None else None,
                spec.shell_anchor + shift,
            )
            virtual_parts.append("True")
            cursor = spec.value_end
            shift += len("True") - (spec.value_end - spec.value_start)
        virtual_parts.append(content[cursor:])
        virtual_content = "".join(virtual_parts)
        replay_budget.check_runtime()
    except static_runner._StaticResourceLimitError:
        return {}
    replacement_starts = [spec.value_start for spec in specs]
    replacement_ends: list[int] = []
    cumulative_deltas: list[int] = [0]
    for spec in specs:
        replacement_ends.append(spec.value_end)
        cumulative_deltas.append(
            cumulative_deltas[-1] + len("True") - (spec.value_end - spec.value_start)
        )

    def source_to_virtual_boundary(source_offset: int) -> int:
        replacement_count = bisect_right(replacement_ends, source_offset)
        return source_offset + cumulative_deltas[replacement_count]

    def boundary_is_preserved(source_offset: int, virtual_offset: int) -> bool:
        """Return whether a raw-window edge has one unambiguous virtual edge."""
        replacement_index = bisect_right(replacement_starts, source_offset) - 1
        if (
            replacement_index >= 0
            and replacement_starts[replacement_index]
            < source_offset
            < replacement_ends[replacement_index]
        ):
            return False
        return source_to_virtual_boundary(source_offset) == virtual_offset

    metadata_by_call = {spec.call_start: spec for spec in specs}
    fixed_coordinates = {
        coordinate
        for finding in findings
        if finding.evidence.get(LEXICAL_DIRECT_SHELL_EVIDENCE) is True
        and type(finding.evidence.get(static_runner._ABSOLUTE_REPLACEMENT_START_LIMIT_EVIDENCE))
        is int
        for key in (
            static_runner._ABSOLUTE_START_EVIDENCE,
            static_runner._ABSOLUTE_ALTERNATE_START_EVIDENCE,
        )
        if type(coordinate := finding.evidence.get(key)) is int
    }
    retained_shell_anchors = {
        spec.shell_anchor for spec in specs if spec.call_start in retained_call_starts
    }
    try:
        continuity_runs = static_runner._continuity_runs_for_anchors(
            content,
            retained_shell_anchors,
            replay_budget.check_runtime,
        )
    except static_runner._StaticResourceLimitError:
        return {}
    continuity_calls = {
        spec.call_start
        for spec in specs
        if spec.call_start in retained_call_starts and spec.shell_anchor in continuity_runs
    }
    replay_targets: dict[int, int] = {}
    target_calls: set[int] = set()
    for call_start in retained_call_starts:
        target_spec = metadata_by_call.get(call_start)
        target_coordinates = virtual_coordinates_by_call.get(call_start)
        if target_spec is None or target_coordinates is None:
            continue
        virtual_call_start, virtual_popen_start, _ = target_coordinates
        related_fixed_boundary = bool(
            {call_start, target_spec.popen_start}.difference({None}).intersection(fixed_coordinates)
        )
        source_coordinates = (call_start, target_spec.popen_start)
        virtual_coordinates = (virtual_call_start, virtual_popen_start)
        changes_outer_window = any(
            not all(
                boundary_is_preserved(source_boundary, virtual_boundary)
                for source_boundary, virtual_boundary in zip(
                    _owning_raw_window(content, source_coordinate),
                    _owning_raw_window(virtual_content, virtual_coordinate),
                    strict=True,
                )
            )
            for source_coordinate, virtual_coordinate in zip(
                source_coordinates,
                virtual_coordinates,
                strict=True,
            )
            if source_coordinate is not None and virtual_coordinate is not None
        ) or (len(content) <= static_runner.SECURITY_VIEW_WINDOW_CHARS) != (
            len(virtual_content) <= static_runner.SECURITY_VIEW_WINDOW_CHARS
        )
        call_end = (
            target_spec.call_end if target_spec.call_end is not None else target_spec.value_end
        )
        try:
            replay_budget.check_runtime()
            call_text = content[target_spec.call_start : call_end]
            call_has_derived_view = _has_derived_security_view(call_text)
            call_has_direct_lookalike = _DIRECT_SHELL_TRUE_VALUE.search(call_text) is not None
            replay_budget.check_runtime()
        except static_runner._StaticResourceLimitError:
            return {}
        if not (
            target_spec.normalized_view
            or call_has_derived_view
            or call_has_direct_lookalike
            or related_fixed_boundary
            or changes_outer_window
            or call_start in continuity_calls
        ):
            continue
        target_calls.add(call_start)
        replay_targets[virtual_call_start] = call_start
        if virtual_popen_start is not None:
            replay_targets[virtual_popen_start] = call_start
    if not target_calls:
        return {}

    source_line_starts = (
        0,
        *(match.end() for match in LOGICAL_LINE_BREAK.finditer(virtual_content)),
    )
    window_targets: dict[tuple[int, int, int, int], set[int]] = {}
    for coordinate in replay_targets:
        bounds = _owning_raw_window(virtual_content, coordinate)
        window_targets.setdefault(bounds, set()).add(coordinate)

    replay_candidates: list[Finding] = []
    seen_candidates: set[static_runner._ViewFindingKey] = set()
    whole_artifact_window = len(virtual_content) <= static_runner.SECURITY_VIEW_WINDOW_CHARS
    try:
        for (raw_start, raw_end, owned_start, owned_end), _ in window_targets.items():
            replay_budget.check_runtime()
            raw_window = virtual_content[raw_start:raw_end]
            owned_source_start = owned_start - raw_start
            owned_source_end = owned_end - raw_start
            outer_right_boundary_is_fixed = raw_end < len(virtual_content) or (
                whole_artifact_window
                and len(virtual_content) == static_runner.SECURITY_VIEW_WINDOW_CHARS
            )
            right_boundary_recovery_start = (
                owned_start + static_runner._RAW_WINDOW_OWNED_CHARS - raw_start
                if raw_end < len(virtual_content)
                else static_runner._RAW_WINDOW_OWNED_CHARS
                if whole_artifact_window
                and len(virtual_content) > static_runner.SECURITY_VIEW_WINDOW_CHARS - len("True")
                else None
            )
            for full_view in security_text_views(raw_window):
                full_view = static_runner._with_fixed_right_boundary(
                    full_view,
                    outer_right_boundary_is_fixed,
                    right_boundary_recovery_start,
                )
                for view in static_runner._bounded_view_slices(full_view):
                    replay_budget.check_runtime()
                    view_findings, _ = static_runner._scan_view_windows(
                        file_path,
                        view,
                        [_DirectShellReplayLexical],
                        replay_budget,
                        None,
                        python_source=True,
                        source_text=raw_window,
                    )
                    owned_findings: list[Finding] = []
                    for replay in view_findings:
                        source_start = replay.evidence.get(static_runner._SOURCE_START_EVIDENCE)
                        if isinstance(source_start, int) and not (
                            owned_source_start <= source_start < owned_source_end
                        ):
                            alternate_start = replay.evidence.get(
                                static_runner._SOURCE_ALTERNATE_START_EVIDENCE
                            )
                            if not (
                                isinstance(alternate_start, int)
                                and owned_source_start <= alternate_start < owned_source_end
                            ):
                                continue
                            replay.evidence[static_runner._SOURCE_START_EVIDENCE] = alternate_start
                            replay.evidence[static_runner._SOURCE_ALTERNATE_START_EVIDENCE] = (
                                source_start
                            )
                        owned_findings.append(replay)
                    static_runner._restore_source_lines(
                        owned_findings,
                        raw_window=raw_window,
                        window_line=1,
                        view=view,
                        window_start=raw_start,
                        source_line_starts=source_line_starts,
                    )
                    for replay in owned_findings:
                        candidate_coordinates = {
                            coordinate
                            for key in (
                                static_runner._ABSOLUTE_START_EVIDENCE,
                                static_runner._ABSOLUTE_ALTERNATE_START_EVIDENCE,
                            )
                            if type(coordinate := replay.evidence.get(key)) is int
                        }
                        if not candidate_coordinates.intersection(replay_targets):
                            continue
                        key = static_runner._view_finding_key(replay)
                        if key not in seen_candidates:
                            seen_candidates.add(key)
                            replay_candidates.append(replay)

        for call_start in continuity_calls.intersection(target_calls):
            replay_budget.check_runtime()
            _, _, virtual_shell_anchor = virtual_coordinates_by_call[call_start]
            projection = static_runner._anchored_continuity_view(
                virtual_content,
                virtual_shell_anchor,
                replay_budget.check_runtime,
            )
            if projection is None or not projection.text or projection.source_offsets is None:
                continue
            projection_is_fixed = projection.source_offsets[-1] + 1 < len(virtual_content)
            for full_view in security_text_views(projection.text):
                full_view = SecurityTextView(
                    name=f"continuity-{full_view.name}",
                    text=full_view.text,
                    source_offsets=full_view.source_offsets,
                    right_boundary_is_fixed=projection_is_fixed,
                )
                for view in static_runner._bounded_view_slices(full_view):
                    replay_budget.check_runtime()
                    view_findings, _ = static_runner._scan_view_windows(
                        file_path,
                        view,
                        [_DirectShellReplayLexical],
                        replay_budget,
                        None,
                        python_source=True,
                        source_text=projection.text,
                    )
                    static_runner._restore_source_lines(
                        view_findings,
                        raw_window=projection.text,
                        window_line=1,
                        view=view,
                        start_source_offsets=projection.source_offsets,
                    )
                    for replay in view_findings:
                        candidate_coordinates = {
                            coordinate
                            for key in (
                                static_runner._ABSOLUTE_START_EVIDENCE,
                                static_runner._ABSOLUTE_ALTERNATE_START_EVIDENCE,
                            )
                            if type(coordinate := replay.evidence.get(key)) is int
                        }
                        if not candidate_coordinates.intersection(replay_targets):
                            continue
                        key = static_runner._view_finding_key(replay)
                        if key not in seen_candidates:
                            seen_candidates.add(key)
                            replay_candidates.append(replay)
    except static_runner._StaticResourceLimitError:
        return {}

    replay_findings = static_runner._deduplicate_view_findings(
        coalesce_path_findings(virtual_content, replay_candidates)
    )
    replay_by_call: dict[int, Finding] = {}
    for replay in replay_findings:
        replay_coordinates = {
            coordinate
            for key in (
                static_runner._ABSOLUTE_START_EVIDENCE,
                static_runner._ABSOLUTE_ALTERNATE_START_EVIDENCE,
            )
            if type(coordinate := replay.evidence.get(key)) is int
        }
        for coordinate in sorted(replay_coordinates):
            call_start = replay_targets.get(coordinate)
            if call_start is not None:
                spec = metadata_by_call[call_start]
                _, _, virtual_shell_anchor = virtual_coordinates_by_call[call_start]
                lexical_anchor = replay.evidence.get(static_runner._ABSOLUTE_ANCHOR_EVIDENCE)
                if lexical_anchor == virtual_shell_anchor:
                    replay.match_fingerprint = spec.canonical_fingerprint
                    if spec.normalized_view and "normalized-view" not in replay.tags:
                        replay.tags.append("normalized-view")
                replay_by_call.setdefault(call_start, replay)
                break
    return replay_by_call


def _bound_shell_call_key(call: ast.Call) -> tuple[int, int, int, int]:
    """Return the call key shared with the flow-sensitive companion."""
    return (
        getattr(call, "lineno", 1),
        getattr(call, "col_offset", 0),
        getattr(call, "end_lineno", getattr(call, "lineno", 1)),
        getattr(call, "end_col_offset", getattr(call, "col_offset", 0)),
    )


def _reconcile_variable_shell_findings(
    content: str,
    findings: list[Finding],
    python_ast: ParsedPythonFile | None,
) -> list[Finding]:
    """Reconcile the bounded variable regex with the shared Python AST."""
    marked = [
        finding
        for finding in findings
        if isinstance(finding.evidence.get(_VARIABLE_SHELL_FLAG_EVIDENCE), str)
    ]
    if not marked:
        return findings

    file_path = marked[0].file
    if python_ast is None or python_ast.tree is None:
        return findings

    from . import static_python_shell_truthiness

    ast_index = _build_variable_shell_ast_index(python_ast)
    if ast_index is None:
        return findings
    resolved: dict[tuple[int, str], list[_VariableShellCandidate]] = {}
    for match in _variable_shell_matches(content):
        assignment_line = bisect_right(ast_index.line_character_starts, match.start(1))
        candidate = _resolve_variable_shell_candidate(ast_index, match, assignment_line)
        if candidate is not None:
            call_span = _node_character_span(python_ast, candidate.call)
            if call_span is not None:
                resolved.setdefault((call_span[0], candidate.name), []).append(candidate)
    ownership, emitted, cached_replacements = (
        static_python_shell_truthiness.bound_shell_call_analysis(file_path, python_ast)
    )
    supported_calls = set(emitted)
    retained_companion_starts: set[int] = set()
    retained_companion_sources: set[tuple[int, str]] = set()
    for finding in findings:
        if finding.evidence.get(static_python_shell_truthiness.BOUND_SHELL_EVIDENCE) is not True:
            continue
        start = finding.evidence.get(static_python_shell_truthiness.BOUND_CALL_START_EVIDENCE)
        end = finding.evidence.get(static_python_shell_truthiness.BOUND_CALL_END_EVIDENCE)
        if type(start) is int and type(end) is int and 0 <= start < end <= len(content):
            retained_companion_starts.add(start)
            retained_companion_sources.add(
                (bisect_right(python_ast.line_character_starts, start), content[start:end])
            )
    retained_emitted: set[tuple[int, int, int, int]] = set()
    for key in emitted:
        start = python_ast.line_character_starts[key[0] - 1] + (
            python_ast.character_column(key[0], key[1]) or 0
        )
        end = python_ast.line_character_starts[key[2] - 1] + (
            python_ast.character_column(key[2], key[3]) or 0
        )
        # The companion shares one report owner for identical same-line calls.
        # Compare the complete raw source, since displayed previews can collide.
        if (
            start in retained_companion_starts
            or (key[0], content[start:end]) in retained_companion_sources
        ):
            retained_emitted.add(key)
    emitted = retained_emitted

    reconciled: list[Finding] = []
    for finding in findings:
        variable_name = finding.evidence.get(_VARIABLE_SHELL_FLAG_EVIDENCE)
        if not isinstance(variable_name, str):
            reconciled.append(finding)
            continue
        call_start = finding.evidence.get(static_runner._ABSOLUTE_ANCHOR_EVIDENCE)
        candidates = (
            resolved.get((call_start, variable_name), []) if type(call_start) is int else []
        )
        if len(candidates) != 1:
            reconciled.append(finding)
            continue
        candidate = candidates[0]
        if not candidate.visible:
            continue
        call_key = _bound_shell_call_key(candidate.call)
        if (
            call_key in cached_replacements
            or call_key in emitted
            or _lexical_shell_has_counterevidence(
                ast_index, python_ast, candidate, receiver_trusted=ownership.get(call_key)
            )
        ):
            continue
        if call_key in supported_calls:
            retained_owner = static_python_shell_truthiness.bound_shell_finding_for_call(
                file_path, python_ast, candidate.call
            )
            if retained_owner is not None:
                replacement = static_runner.analyzer_finding_to_finding(retained_owner)
                replacement.finding_id = finding.finding_id
                reconciled.append(replacement)
                continue
        if variable_name.casefold().startswith("true"):
            call = candidate.call
            finding.start_line = getattr(call, "lineno", finding.start_line)
            finding.end_line = getattr(call, "end_lineno", finding.end_line)
            finding.start_column = python_ast.character_column(
                finding.start_line, getattr(call, "col_offset", 0)
            )
            if finding.end_line is not None:
                finding.end_column = python_ast.character_column(
                    finding.end_line, getattr(call, "end_col_offset", 0)
                )
            call_source = python_ast.source_segment(call)
            if call_source is not None:
                finding.matched_text = call_source[:200]
                finding.finding = call_source[:200]
            if type(call_start) is int:
                call_context = get_context(content, call_start)
                finding.context = call_context
                finding.code_snippet = call_context
        reconciled.append(finding)
    return reconciled


def postprocess_path_findings(
    content: str,
    findings: list[Finding],
    *,
    python_ast: ParsedPythonFile | None = None,
    started_at: float | None = None,
    timeout_seconds: float | None = None,
) -> list[Finding]:
    """Finalize ownership, classification, and private TM1 coordinates."""
    from . import static_python_shell_truthiness

    findings = _reconcile_variable_shell_findings(content, findings, python_ast)
    direct_replays = _bound_direct_replays(
        content,
        findings,
        python_ast,
        started_at=started_at,
        timeout_seconds=timeout_seconds,
    )
    reconciled = coalesce_path_findings(content, findings)
    if python_ast is not None:
        direct_starts = {
            start
            for finding in reconciled
            if finding.evidence.get(LEXICAL_DIRECT_SHELL_EVIDENCE) is True
            and type(start := _lexical_call_start(finding)) is int
        }
        direct_metadata = static_python_shell_truthiness.direct_literal_metadata(
            python_ast,
            next((finding.file for finding in reconciled), "<unknown>"),
            direct_starts,
        )
        canonical_direct_calls: dict[tuple[str, int, str], set[int]] = {}
        for finding in reconciled:
            direct_call_start = _lexical_call_start(finding)
            metadata = (
                direct_metadata.get(direct_call_start) if type(direct_call_start) is int else None
            )
            if metadata is not None:
                canonical_direct_calls.setdefault(
                    (finding.file, finding.start_line, metadata.match_fingerprint),
                    set(),
                ).add(metadata.call_start)
        colliding_direct_identities = {
            key for key, call_starts in canonical_direct_calls.items() if len(call_starts) > 1
        }
        for finding in reconciled:
            direct_call_start = _lexical_call_start(finding)
            metadata = (
                direct_metadata.get(direct_call_start) if type(direct_call_start) is int else None
            )
            lexical_anchor = finding.evidence.get(static_runner._ABSOLUTE_ANCHOR_EVIDENCE)
            if metadata is None or lexical_anchor != metadata.shell_anchor:
                continue
            if (
                finding.file,
                finding.start_line,
                metadata.match_fingerprint,
            ) not in colliding_direct_identities:
                finding.match_fingerprint = metadata.match_fingerprint
            finding.evidence.update(
                {
                    static_python_shell_truthiness.BOUND_SHELL_EVIDENCE: True,
                    static_python_shell_truthiness.BOUND_CALL_START_EVIDENCE: metadata.call_start,
                    static_python_shell_truthiness.DIRECT_LITERAL_METADATA_EVIDENCE: True,
                }
            )
            if metadata.call_end is not None:
                finding.evidence[static_python_shell_truthiness.BOUND_CALL_END_EVIDENCE] = (
                    metadata.call_end
                )
            if metadata.normalized_view and "normalized-view" not in finding.tags:
                finding.tags.append("normalized-view")
            if metadata.normalized_view:
                finding.evidence[static_python_shell_truthiness.BOUND_NORMALIZED_VIEW_EVIDENCE] = (
                    True
                )
    safety_by_scope: dict[tuple[int, int, bool], tuple[bool, bool]] = {}
    raw_lines_by_start: dict[int, tuple[str, ...]] = {}
    source_line_starts = (0, *(match.end() for match in LOGICAL_LINE_BREAK.finditer(content)))

    for finding in reconciled:
        is_bound = finding.evidence.get(static_python_shell_truthiness.BOUND_SHELL_EVIDENCE) is True
        if (
            finding.evidence.get(LEXICAL_DIRECT_SHELL_EVIDENCE) is True
            and "normalized-view" in finding.tags
            and isinstance(
                canonical_match := finding.evidence.get(LEXICAL_NORMALIZED_MATCH_EVIDENCE),
                str,
            )
        ):
            finding.matched_text = canonical_match
            finding.finding = canonical_match
        is_direct_literal = (
            finding.evidence.get(static_python_shell_truthiness.DIRECT_LITERAL_METADATA_EVIDENCE)
            is True
        )
        if is_bound and not is_direct_literal:
            if (
                finding.evidence.get(static_python_shell_truthiness.BOUND_NORMALIZED_VIEW_EVIDENCE)
                is True
                and "normalized-view" not in finding.tags
            ):
                finding.tags.append("normalized-view")
            matched = (finding.matched_text or "")[:200]
            classification_match = finding.evidence.get(
                static_python_shell_truthiness.BOUND_CLASSIFICATION_MATCH_EVIDENCE,
                matched,
            )
            if not isinstance(classification_match, str):
                classification_match = matched
            call_start = finding.evidence.get(
                static_python_shell_truthiness.BOUND_CALL_START_EVIDENCE
            )
            call_end = finding.evidence.get(static_python_shell_truthiness.BOUND_CALL_END_EVIDENCE)
            classification_context = finding.context or ""
            matched_line = matched
            safe_context: bool | None = None
            safe_matched_line: bool | None = None
            if type(call_start) is int:
                preserves_lexical_location = (
                    finding.evidence.get(
                        static_python_shell_truthiness.DIRECT_LITERAL_METADATA_EVIDENCE
                    )
                    is True
                )
                if not preserves_lexical_location:
                    finding.start_line = bisect_right(source_line_starts, call_start)
                    if type(call_end) is int and call_end > call_start:
                        finding.end_line = bisect_right(source_line_starts, call_end - 1)
                retained_start = finding.evidence.get(static_runner._ABSOLUTE_START_EVIDENCE)
                direct_owner_start = finding.evidence.get(
                    static_python_shell_truthiness.BOUND_DIRECT_OWNER_START_EVIDENCE
                )
                classification_start = (
                    retained_start
                    if preserves_lexical_location and type(retained_start) is int
                    else direct_owner_start
                    if type(direct_owner_start) is int
                    else call_start
                )
                direct_match_end = finding.evidence.get(
                    static_python_shell_truthiness.BOUND_DIRECT_MATCH_END_EVIDENCE
                )
                popen_start = finding.evidence.get(
                    static_python_shell_truthiness.BOUND_POPEN_START_EVIDENCE
                )
                _, qualifier_window_end = _raw_classification_bounds(content, call_start)
                if (
                    not preserves_lexical_location
                    and classification_start == call_start
                    and type(direct_match_end) is int
                    and direct_match_end > qualifier_window_end
                    and type(popen_start) is int
                ):
                    classification_start = popen_start
                raw_start, raw_end = _raw_classification_bounds(content, classification_start)
                classification_line = bisect_right(
                    source_line_starts,
                    classification_start,
                )
                normalized_view = (
                    finding.evidence.get(
                        static_python_shell_truthiness.BOUND_NORMALIZED_VIEW_EVIDENCE
                    )
                    is True
                )
                scope_key = (raw_start, classification_line, normalized_view)
                safety = safety_by_scope.get(scope_key)
                if safety is None:
                    raw_lines = raw_lines_by_start.get(raw_start)
                    if raw_lines is None:
                        raw_lines = tuple(content[raw_start:raw_end].splitlines())
                        raw_lines_by_start[raw_start] = raw_lines
                    raw_start_line = bisect_right(source_line_starts, raw_start)
                    local_line = classification_line - raw_start_line
                    if 0 <= local_line < len(raw_lines):
                        context_start = max(0, local_line - 3)
                        context_end = min(len(raw_lines), local_line + 4)
                        classification_context = "\n".join(raw_lines[context_start:context_end])
                        matched_line = raw_lines[local_line]
                    if normalized_view:
                        classification_context = normalized_security_view(
                            classification_context
                        ).text
                        matched_line = normalized_security_view(matched_line).text
                        classification_match = normalized_security_view(classification_match).text
                    safety = (
                        _is_safe_container_command(classification_context)
                        or _is_safe_dockerfile_idiom(
                            classification_context,
                            classification_match,
                        ),
                        _is_safe_cache_cleanup(matched_line),
                    )
                    safety_by_scope[scope_key] = safety
                safe_context, safe_matched_line = safety
            severity, finding.confidence = _classify_tm1(
                classification_context,
                classification_match,
                matched_line,
                (
                    0.8
                    if finding.evidence.get(
                        static_python_shell_truthiness.DIRECT_LITERAL_METADATA_EVIDENCE
                    )
                    is True
                    else finding.confidence
                ),
                "python",
                safe_context=safe_context,
                safe_matched_line=safe_matched_line,
            )
            finding.severity = severity.value
        borrowed_classification = finding.evidence.get(LEXICAL_BOUND_CLASSIFICATION_EVIDENCE)
        if isinstance(borrowed_classification, dict):
            borrowed_severity = borrowed_classification.get("severity")
            borrowed_confidence = borrowed_classification.get("confidence")
            borrowed_tags = borrowed_classification.get("tags")
            if isinstance(borrowed_severity, str):
                finding.severity = borrowed_severity
            if isinstance(borrowed_confidence, (int, float)) and not isinstance(
                borrowed_confidence, bool
            ):
                finding.confidence = float(borrowed_confidence)
            if isinstance(borrowed_tags, (list, tuple)) and all(
                isinstance(tag, str) for tag in borrowed_tags
            ):
                finding.tags = list(borrowed_tags)
        replay_call_start = finding.evidence.get(
            static_python_shell_truthiness.BOUND_CALL_START_EVIDENCE
        )
        replay = (
            direct_replays.get(replay_call_start)
            if type(replay_call_start) is int and not is_direct_literal
            else None
        )
        if replay is not None:
            finding.match_fingerprint = replay.fingerprint()
            finding.severity = replay.severity
            finding.confidence = replay.confidence
            finding.tags = list(replay.tags)
    return cleanup_path_findings(reconciled)


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run tool_misuse patterns and return findings."""
    from . import static_python_shell_truthiness

    response = static_runner.run_static_patterns_with_ledger(
        state,
        [sys.modules[__name__], static_python_shell_truthiness],
    )
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
