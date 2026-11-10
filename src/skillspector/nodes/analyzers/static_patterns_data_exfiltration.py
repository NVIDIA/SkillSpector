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
import itertools
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

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
# Calls that hand an environment mapping to a child process. Materializing
# ``os.environ`` for one of these is process launching, not harvesting: the child
# receives the environment the skill already runs in, and no value leaves the host.
_CHILD_PROCESS_ENV_CALLS = frozenset(
    {
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.Popen",
        "asyncio.create_subprocess_exec",
        "asyncio.create_subprocess_shell",
    }
)
_LAUNCHER_MODULES = frozenset(name.partition(".")[0] for name in _CHILD_PROCESS_ENV_CALLS)
# A copy the flow pass traces only to a child process stays an E2 finding, but as a
# low-confidence LOW. The proof covers ordinary Python, not every introspection
# trick, so a read it misses lowers the finding instead of removing it.
_CHILD_PROCESS_PASSTHROUGH_CONFIDENCE = 0.2
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


# In-place edits of a mapping that keep the values on the host. ``update`` and
# ``clear`` return ``None``. The extraction methods hand a value back, so they
# only count as edits when that value is thrown away. A value kept from the copy,
# even by one literal key, is invisible to taint tracking, so it keeps the finding.
_ENVIRONMENT_MUTATION_METHODS = frozenset({"update", "clear"})
_ENVIRONMENT_EXTRACTION_METHODS = frozenset({"pop", "popitem", "setdefault"})

# Statement lists a raise, ``break`` or ``continue`` can leave before a later
# rebinding runs, while code that still sees the older binding goes on to run:
# a handler, a ``finally``, a context manager that swallows the error, or the
# code after the loop. Loop bodies are handled in ``_visit_loop``.
_PROTECTED_BODIES = frozenset(
    {
        (ast.Try, "body"),
        (ast.Try, "orelse"),
        (ast.TryStar, "body"),
        (ast.TryStar, "orelse"),
        (ast.ExceptHandler, "body"),
        (ast.With, "body"),
        (ast.AsyncWith, "body"),
    }
)

# Builtins, attributes and modules that read variables by a name only known at
# run time.
_DYNAMIC_NAMESPACE_NAMES = frozenset({"globals", "locals", "vars", "eval", "exec", "sys.modules"})
_DYNAMIC_NAMESPACE_ATTRIBUTES = frozenset({"f_locals", "f_globals", "__globals__"})

# Calls that can replace an attribute of the object passed first, and the
# ``sys.modules`` methods that can replace a module before it is imported.
_ATTRIBUTE_WRITE_CALLS = frozenset({"setattr", "delattr", "vars"})
_MODULE_TABLE_WRITE_METHODS = frozenset(
    {"update", "setdefault", "pop", "popitem", "clear", "__setitem__", "__delitem__"}
)

# Bindings of one name that may all still be live, past which the name is
# reported rather than followed. Keeps crafted files linear.
_MAX_OPEN_BINDINGS = 32


def _method_receiver(node: ast.AST, methods: frozenset[str]) -> ast.Name | None:
    """Return the bare name a ``name.method(...)`` call is made on, for *methods*."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in methods
        and isinstance(node.func.value, ast.Name)
    ):
        return node.func.value
    return None


def _edit_receiver(node: ast.AST) -> ast.Name | None:
    """Return the name a call edits in place without taking values off the mapping."""
    receiver = _method_receiver(node, _ENVIRONMENT_MUTATION_METHODS)
    if receiver is None and isinstance(node, ast.Expr):
        receiver = _method_receiver(node.value, _ENVIRONMENT_EXTRACTION_METHODS)
    return receiver


def _is_main_guard(test: ast.expr) -> bool:
    """Return whether ``test`` is ``__name__ == "__main__"``, either way round."""
    if not (
        isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
    ):
        return False
    sides = (test.left, test.comparators[0])
    return any(isinstance(side, ast.Name) and side.id == "__name__" for side in sides) and any(
        isinstance(side, ast.Constant) and side.value == "__main__" for side in sides
    )


def _reads_names_dynamically(node: ast.AST, aliases: dict[str, str]) -> bool:
    """Return whether ``node`` can read a variable by a name only known at run time.

    Any reference counts, not only a call, so ``namespace = locals`` is caught too.
    """
    if isinstance(node, ast.Attribute):
        if node.attr in _DYNAMIC_NAMESPACE_ATTRIBUTES:
            return True
        # Two parts at most (``sys.modules``, ``builtins.locals``) keeps this linear.
        if not isinstance(node.value, ast.Name):
            return False
        name = apply_import_aliases(f"{node.value.id}.{node.attr}", aliases)
    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        name = apply_import_aliases(node.id, aliases)
    else:
        return False
    return name in _DYNAMIC_NAMESPACE_NAMES


class _FileBindings:
    """How one file binds its names: by import or otherwise, and what it patches.

    The flow pass sees one file, and call names are matched by spelling, so a
    launcher or a ``__main__`` guard is only trusted when nothing in the file can
    have replaced it.
    """

    def __init__(self, tree: ast.AST, aliases: dict[str, str]) -> None:
        # Local name to the absolute import paths bound to it. ``""`` marks a
        # relative import, which may load a file the skill ships.
        self._imports: dict[str, set[str]] = {}
        # Names bound other than by an import, anywhere in the file.
        self._rebound: set[str] = set()
        # Top-level modules with an attribute written from this file.
        self._patched: set[str] = set()
        # A star import or a write to ``sys.modules`` can replace any import.
        self._imports_replaceable = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # ``import a.b`` binds ``a``; ``import a.b as c`` binds ``c`` to ``a.b``.
                    root = alias.name.partition(".")[0]
                    local, path = (alias.asname, alias.name) if alias.asname else (root, root)
                    self._imports.setdefault(local, set()).add(path)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        self._imports_replaceable = True
                        continue
                    path = f"{node.module}.{alias.name}" if node.module and not node.level else ""
                    self._imports.setdefault(alias.asname or alias.name, set()).add(path)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self._rebound.add(node.name)
            elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                self._rebound.add(node.id)
            elif isinstance(node, ast.arg):
                self._rebound.add(node.arg)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                self._rebound.update(node.names)
            elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
                self._rebound.add(node.name)
            elif isinstance(node, ast.MatchMapping) and node.rest:
                self._rebound.add(node.rest)
            else:
                self._record_patch(node, aliases)

    def _record_patch(self, node: ast.AST, aliases: dict[str, str]) -> None:
        """Note a module whose attributes ``node`` may replace."""
        written: ast.expr | None = None
        if isinstance(node, (ast.Attribute, ast.Subscript)) and not isinstance(node.ctx, ast.Load):
            written = node.value if isinstance(node, ast.Subscript) else node
        elif (
            isinstance(node, ast.Attribute)
            and node.attr == "__dict__"
            and isinstance(node.value, ast.Name)
        ):
            written = node.value
        elif isinstance(node, ast.Call) and node.args:
            call_name = resolve_call_name(node, aliases)
            if call_name in _ATTRIBUTE_WRITE_CALLS:
                written = node.args[0]
            elif call_name is not None and call_name.startswith("sys.modules."):
                if call_name.rpartition(".")[2] in _MODULE_TABLE_WRITE_METHODS:
                    written = node.func
        if written is None:
            return
        name = _resolve_expression_name(written, aliases)
        if name is None:
            return
        if name == "sys.modules" or name.startswith("sys.modules."):
            self._imports_replaceable = True
        self._patched.add(name.partition(".")[0])

    def launchers(self) -> set[str]:
        """Local names a child-process launcher can be called through.

        Each one is bound only by absolute imports of one path in a launcher
        module, and nothing in the file writes to that module.
        """
        if self._imports_replaceable:
            return set()
        trusted: set[str] = set()
        for name, paths in self._imports.items():
            if name in self._rebound or len(paths) != 1:
                continue
            module = next(iter(paths)).partition(".")[0]
            if module in _LAUNCHER_MODULES and module not in self._patched:
                trusted.add(name)
        return trusted

    def main_guard_holds(self) -> bool:
        """Whether ``__name__`` keeps its import-time value in this file."""
        return (
            not self._imports_replaceable
            and "__name__" not in self._rebound
            and "__name__" not in self._imports
        )


@dataclass
class _EnvironmentBinding:
    """One name bound to a candidate expression, tracked until a rebinding that surely runs."""

    value_id: int
    # The statement block the binding was made in, as a path of block numbers.
    block: tuple[int, ...]
    reached_child_process: bool = False
    escaped: bool = False


@dataclass
class _Loop:
    """Names read and bindings made in one loop, whose body may run again."""

    reads: set[tuple[_Scope, str]] = field(default_factory=set)
    bindings: list[tuple[_Scope, str, _EnvironmentBinding]] = field(default_factory=list)


# Compared by identity, so a scope can key the per-loop read set.
@dataclass(eq=False)
class _Scope:
    """Open bindings of one lexical scope: module, function, lambda, class or comprehension."""

    # ``None`` for the module, which owns every name no inner scope binds.
    local_names: set[str] | None
    declared_global: set[str] = field(default_factory=set)
    # The body runs when it is called or iterated, not where it is defined.
    deferred: bool = False
    is_class: bool = False
    is_comprehension: bool = False
    # Several bindings of one name stay open when a rebinding may not have run.
    open: dict[str, list[_EnvironmentBinding]] = field(default_factory=dict)
    read_later: set[str] = field(default_factory=set)

    def owns(self, name: str) -> bool:
        return self.local_names is None or name in self.local_names


def _scope_declarations(body: Sequence[ast.AST]) -> tuple[set[str], set[str]]:
    """Return the names a scope binds itself and the names it declares ``global``.

    Nested functions, lambdas and classes are scopes of their own, so only their
    names count here. Comprehension targets stay inside the comprehension, but a
    walrus inside one binds in this scope. ``nonlocal`` names belong to an
    enclosing function.
    """
    bound: set[str] = set()
    declared_global: set[str] = set()
    declared_nonlocal: set[str] = set()
    pending: list[ast.AST] = list(body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
            continue
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, ast.comprehension):
            pending.append(node.iter)
            pending.extend(node.ifs)
            continue
        if isinstance(node, ast.Global):
            declared_global.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            declared_nonlocal.update(node.names)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update(
                (alias.asname or alias.name).split(".")[0]
                for alias in node.names
                if alias.name != "*"
            )
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
        pending.extend(ast.iter_child_nodes(node))
    return bound - declared_global - declared_nonlocal, declared_global


def _argument_names(arguments: ast.arguments) -> set[str]:
    return {
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


def _annotation_expressions(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.expr | None]:
    arguments = node.args
    return [
        *(
            argument.annotation
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                arguments.vararg,
                arguments.kwarg,
            )
            if argument is not None
        ),
        node.returns,
    ]


def _type_param_expressions(type_params: Sequence[ast.type_param]) -> list[ast.expr | None]:
    # Type parameter defaults arrived in Python 3.13.
    return [
        expression
        for type_param in type_params
        for expression in (
            getattr(type_param, "bound", None),
            getattr(type_param, "default_value", None),
        )
    ]


class _EnvironmentFlowVisitor(ast.NodeVisitor):
    """Decide which environment mappings only ever reach a child-process ``env=``.

    Names are followed per lexical scope, in evaluation order (assignment values
    before their targets), until a rebinding that surely runs. A later ``env = {}``
    handed to a launcher therefore cannot vouch for an earlier
    ``env = os.environ.copy()``, and a nested scope's own ``env`` cannot close the
    outer one. A binding passes through only if every use is a child-process
    ``env=`` argument or an in-place edit of the mapping; any other use keeps the
    finding HIGH.

    The pass sees one file, so a binding only passes through by name where no
    other file can reach it: a function local, or a module global bound in a
    top-level ``if __name__ == "__main__":`` body, which importers never run. A
    launcher only counts when its root name is bound solely by an absolute import
    from ``subprocess`` or ``asyncio`` and nothing in the file writes to that
    module, so a local ``def run`` or ``class subprocess`` is never trusted.

    Control flow errs toward keeping the finding HIGH. A rebinding closes only the
    bindings made in its own block or in blocks nested inside it, so one inside an
    ``if``, ``try``, ``with``, ``match`` or loop body leaves the outer binding open
    next to the new one, and later uses count against both. A rebinding never
    closes anything where a raise, ``break`` or ``continue`` can skip it (``try``,
    ``except``, ``with`` and loop bodies), or as a walrus, which can sit in a
    short-circuit. A loop body may run again, so a read anywhere in a loop counts
    against every binding made in that loop. A walrus result, a class attribute
    and any name in a file that can read names dynamically (``globals``,
    ``eval``, frame locals, ``__globals__``, ``sys.modules``) keep the finding,
    as does a name with too many live bindings.

    A function or lambda body runs at some later time, so its reads of an
    enclosing name count against every binding of that name that is open when the
    body is defined or made after it. Its writes to an enclosing name through
    ``global`` or ``nonlocal`` cannot be ordered against the enclosing uses and
    leave those bindings as they are.
    """

    def __init__(self, tree: ast.AST, aliases: dict[str, str]) -> None:
        self._env_arguments: set[int] = set()
        self._mutated_names: set[int] = set()
        self._dynamic_reads = False
        bindings = _FileBindings(tree, aliases)
        launchers = bindings.launchers()
        # Importers never run a top-level ``if __name__ == "__main__":`` body.
        self._main_guards: set[int] = set()
        if isinstance(tree, ast.Module) and bindings.main_guard_holds():
            self._main_guards = {
                id(statement)
                for statement in tree.body
                if isinstance(statement, ast.If) and _is_main_guard(statement.test)
            }
        self._in_main_guard = False
        self._lazy = 0
        for node in ast.walk(tree):
            if _reads_names_dynamically(node, aliases):
                self._dynamic_reads = True
            receiver = _edit_receiver(node)
            if receiver is not None:
                self._mutated_names.add(id(receiver))
            elif isinstance(node, ast.Call):
                call_name = resolve_call_name(node, aliases)
                spelling = resolve_dotted_name(node.func)
                if (
                    call_name in _CHILD_PROCESS_ENV_CALLS
                    and spelling is not None
                    and spelling.partition(".")[0] in launchers
                ):
                    self._env_arguments.update(
                        id(keyword.value) for keyword in node.keywords if keyword.arg == "env"
                    )
            elif (
                isinstance(node, ast.Subscript)
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and isinstance(node.value, ast.Name)
            ):
                self._mutated_names.add(id(node.value))
            elif (
                isinstance(node, ast.AugAssign)
                and isinstance(node.op, ast.BitOr)
                and isinstance(node.target, ast.Name)
            ):
                # Only ``dict.__ior__`` exists. Any other operator hands the whole
                # mapping to the right operand's reflected method.
                self._mutated_names.add(id(node.target))
        self._scopes: list[_Scope] = [_Scope(local_names=None)]
        self._passed_through: set[int] = set()
        self._escaped: set[int] = set()
        self._block_numbers = itertools.count()
        self._block: tuple[int, ...] = ()
        # How many protected bodies of the current scope enclose this point.
        self._protected = 0
        self._loops: list[_Loop] = []

    def passthrough_ids(self) -> set[int]:
        """Ids of expressions whose only destination is a child-process ``env=``."""
        self._close_scope(self._scopes[0])
        if self._dynamic_reads:
            # Any name may be read back by a string, so only inline arguments are safe.
            return set(self._env_arguments)
        return (self._passed_through - self._escaped) | self._env_arguments

    def _close(self, scope: _Scope, name: str, block: tuple[int, ...] | None = None) -> None:
        """Close the bindings of ``name`` made in ``block`` or in blocks nested inside it.

        Those are the bindings a rebinding in ``block`` surely replaces; ``None``
        closes them all.
        """
        still_open: list[_EnvironmentBinding] = []
        for binding in scope.open.pop(name, []):
            if block is not None and binding.block[: len(block)] != block:
                still_open.append(binding)
            elif binding.escaped:
                self._escaped.add(binding.value_id)
            elif binding.reached_child_process:
                self._passed_through.add(binding.value_id)
        if still_open:
            scope.open[name] = still_open

    def _close_scope(self, scope: _Scope) -> None:
        for name in list(scope.open):
            self._close(scope, name)

    def _resolve(self, name: str, *, walrus: bool = False) -> tuple[_Scope, bool]:
        """Return the scope that owns ``name`` here, and whether this access runs later."""
        deferred = self._lazy > 0
        innermost = True
        for scope in reversed(self._scopes):
            if walrus and scope.is_comprehension:
                deferred = deferred or scope.deferred
                continue
            # A class body does not enclose the scopes nested inside it.
            if innermost or not scope.is_class:
                if scope.owns(name):
                    return scope, deferred
                if name in scope.declared_global:
                    return self._scopes[0], deferred or scope.deferred
            deferred = deferred or scope.deferred
            innermost = False
        return self._scopes[0], deferred

    def _store(self, name: str, value_id: int | None, *, walrus: bool = False) -> None:
        scope, deferred = self._resolve(name, walrus=walrus)
        if deferred:
            return
        if not walrus and not self._protected:
            self._close(scope, name, self._block)
        if value_id is None:
            return
        # A walrus also hands the value to the enclosing expression, a class
        # attribute outlives the class body, and another file can import a module
        # global this pass never sees.
        escaped = (
            walrus
            or scope.is_class
            or name in scope.read_later
            or (scope is self._scopes[0] and not self._in_main_guard)
        )
        binding = _EnvironmentBinding(value_id, self._block, escaped=escaped)
        bindings = scope.open.setdefault(name, [])
        bindings.append(binding)
        for loop in self._loops:
            loop.bindings.append((scope, name, binding))
        if len(bindings) > _MAX_OPEN_BINDINGS:
            self._escaped.update(open_binding.value_id for open_binding in bindings)
            del scope.open[name]

    def _bind(self, target: ast.expr, value_id: int, *, walrus: bool = False) -> None:
        if isinstance(target, ast.Name):
            self._store(target.id, value_id, walrus=walrus)
        else:
            self.visit(target)

    def _visit_block(self, nodes: Sequence[ast.AST], *, protected: bool = False) -> None:
        """Visit statements that may run zero, one or several times as one block."""
        outer = self._block
        self._block = (*outer, next(self._block_numbers))
        self._protected += protected
        for node in nodes:
            self.visit(node)
        self._protected -= protected
        self._block = outer

    def _visit_loop(self, header: Sequence[ast.AST], body: list[ast.stmt]) -> None:
        """Visit a loop's repeated part, then count its reads against its bindings."""
        loop = _Loop()
        self._loops.append(loop)
        # ``break`` and ``continue`` can skip any rebinding in the body.
        self._visit_block([*header, *body], protected=True)
        self._loops.pop()
        # The next pass reads what this pass bound, whatever the source order.
        for scope, name, binding in loop.bindings:
            if (scope, name) in loop.reads:
                self._escaped.add(binding.value_id)

    def generic_visit(self, node: ast.AST) -> None:
        """Visit children in order, giving every nested statement list its own block."""
        for field_name, value in ast.iter_fields(node):
            if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                protected = (type(node), field_name) in _PROTECTED_BODIES
                self._visit_block(value, protected=protected)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)

    def _visit_scope(self, scope: _Scope, body: list[ast.stmt] | list[ast.expr]) -> None:
        # A ``try`` around a definition cannot see the locals of its body.
        outer_protected, self._protected = self._protected, 0
        self._scopes.append(scope)
        self._visit_block(body)
        self._close_scope(self._scopes.pop())
        self._protected = outer_protected

    def _visit_lazy(self, expressions: Sequence[ast.expr | None]) -> None:
        """Visit expressions that run where they are defined, or only when read.

        Annotations run eagerly before Python 3.14 and lazily from it on, and type
        parameters and ``type`` alias values always run lazily, so a read here
        counts against the bindings open now and every later one.
        """
        self._lazy += 1
        for expression in expressions:
            if expression is not None:
                self.visit(expression)
        self._lazy -= 1

    def _visit_defaults(self, arguments: ast.arguments) -> None:
        for default in (*arguments.defaults, *arguments.kw_defaults):
            if default is not None:
                self.visit(default)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_defaults(node.args)
        self._visit_lazy(
            [*_type_param_expressions(node.type_params), *_annotation_expressions(node)]
        )
        bound, declared_global = _scope_declarations(node.body)
        scope = _Scope(bound | _argument_names(node.args), declared_global, deferred=True)
        self._visit_scope(scope, node.body)
        self._store(node.name, None)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_defaults(node.args)
        bound, _ = _scope_declarations([node.body])
        scope = _Scope(bound | _argument_names(node.args), deferred=True)
        self._visit_scope(scope, [node.body])

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_lazy(_type_param_expressions(node.type_params))
        for expression in (*node.decorator_list, *node.bases):
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)
        bound, declared_global = _scope_declarations(node.body)
        self._visit_scope(_Scope(bound, declared_global, is_class=True), node.body)
        self._store(node.name, None)

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        results: list[ast.expr],
    ) -> None:
        # The first iterable is evaluated in the enclosing scope.
        self.visit(node.generators[0].iter)
        targets = {
            name.id
            for generator in node.generators
            for name in ast.walk(generator.target)
            if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)
        }
        deferred = isinstance(node, ast.GeneratorExp)
        self._scopes.append(_Scope(targets, deferred=deferred, is_comprehension=True))
        repeated: list[ast.AST] = []
        for index, generator in enumerate(node.generators):
            if index:
                repeated.append(generator.iter)
            repeated.append(generator.target)
            repeated.extend(generator.ifs)
        self._visit_loop([*repeated, *results], [])
        self._close_scope(self._scopes.pop())

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, [node.elt])

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node, [node.elt])

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node, [node.elt])

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, [node.key, node.value])

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self._bind(target, id(node.value))

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            self._bind(node.target, id(node.value))
        elif not isinstance(node.target, ast.Name):
            # Without a value, an attribute or subscript target still evaluates its base.
            self.visit(node.target)
        # A function never evaluates its local annotations; counting them anyway is safe.
        self._visit_lazy([node.annotation])

    def visit_TypeAlias(self, node: ast.TypeAlias) -> None:
        self.visit(node.name)
        self._visit_lazy([*_type_param_expressions(node.type_params), node.value])

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._bind(node.target, id(node.value), walrus=True)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        target = node.target
        if isinstance(target, ast.Name) and id(target) not in self._mutated_names:
            # The operator reads the old value before the name is rebound.
            self.visit(ast.copy_location(ast.Name(target.id, ast.Load()), target))
        self.visit(target)

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        # The target is rebound on every pass, and not at all for an empty iterable.
        self._visit_loop([node.target], node.body)
        self._visit_block(node.orelse)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_If(self, node: ast.If) -> None:
        if id(node) not in self._main_guards:
            self.generic_visit(node)
            return
        self.visit(node.test)
        self._in_main_guard = True
        self._visit_block(node.body)
        self._in_main_guard = False
        self._visit_block(node.orelse)

    def visit_While(self, node: ast.While) -> None:
        # The test runs again before every pass.
        self._visit_loop([node.test], node.body)
        self._visit_block(node.orelse)

    def visit_Name(self, node: ast.Name) -> None:
        node_id = id(node)
        if not isinstance(node.ctx, ast.Load) and node_id not in self._mutated_names:
            self._store(node.id, None)
            return
        scope, deferred = self._resolve(node.id)
        bindings = scope.open.get(node.id, [])
        if node_id in self._env_arguments:
            for binding in bindings:
                binding.reached_child_process = True
        elif node_id not in self._mutated_names:
            if deferred:
                scope.read_later.add(node.id)
            if scope.is_class:
                # Before the class body binds the name, the read falls through to globals.
                bindings = [*bindings, *self._scopes[0].open.get(node.id, [])]
            for binding in bindings:
                binding.escaped = True
            for loop in self._loops:
                loop.reads.add((scope, node.id))


def _analyze_python_environment_reads(
    content: str,
    file_path: str,
    python_ast: ParsedPythonFile | None = None,
) -> list[AnalyzerFinding] | None:
    """Detect materializing or enumerating the complete ``os.environ`` mapping.

    A full mapping copy or enumeration is an environment-harvesting signal, unlike a
    targeted single-key lookup. A copy that the flow pass traces only to a child
    process is still reported, but as a low-confidence LOW pass-through.
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
    try:
        flow = _EnvironmentFlowVisitor(tree, aliases)
        flow.visit(tree)
        child_process_passthroughs = flow.passthrough_ids()
    except RecursionError:
        # Too deep for the recursive flow pass: report every full read as HIGH
        # instead of losing the whole file.
        child_process_passthroughs = set()
    findings: list[AnalyzerFinding] = []
    emitted: set[int] = set()
    tag = [PatternCategory.DATA_EXFILTRATION.value]

    def emit(node: ast.AST, confidence: float) -> None:
        node_id = id(node)
        if node_id in emitted:
            return
        emitted.add(node_id)
        passthrough = node_id in child_process_passthroughs
        lineno = getattr(node, "lineno", 1)
        end_lineno = getattr(node, "end_lineno", None)
        matched_text = ast.get_source_segment(content, node) or "os.environ"
        findings.append(
            AnalyzerFinding(
                rule_id="E2",
                message=(
                    "Environment Pass-Through to Child Process"
                    if passthrough
                    else "Env Variable Harvesting"
                ),
                severity=Severity.LOW if passthrough else Severity.HIGH,
                location=Location(file=file_path, start_line=lineno, end_line=end_lineno),
                confidence=(
                    min(confidence, _CHILD_PROCESS_PASSTHROUGH_CONFIDENCE)
                    if passthrough
                    else confidence
                ),
                explanation=(
                    "The copy of os.environ is only handed to a child process launched "
                    "through subprocess or asyncio, so the child receives the environment "
                    "the skill already runs in. No other read of the copy was found."
                    if passthrough
                    else None
                ),
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
