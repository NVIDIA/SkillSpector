# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer-verified string and comment ownership for complete Python modules.

Static analyzers read Python source as text. In a module that the Python parser
accepts, every quote character is a string delimiter, string content, or
comment text, and the tokenizer says which. Lexical scanners can borrow that
ownership instead of guessing quote pairs. Fragments, malformed source, and
polyglots prove nothing and keep each caller's conservative path.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
import warnings
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Callable
from threading import Lock

from skillspector.python_ast import MAX_PYTHON_AST_SOURCE_CHARS

# The tokenizer and the compiler disagree about a carriage return that does
# not start CRLF, so such source never acquires token ownership.
_LONE_CARRIAGE_RETURN_RE = re.compile(r"\r(?!\n)")
# A shebang that names another interpreter (for example a shell polyglot)
# makes the Python host ambiguous. Without a shebang, ``.py`` source is Python.
_PYTHON_SHEBANG_RE = re.compile(
    r"#![ \t]*+(?:\S*/)?(?:env[ \t]++(?:-\S*[ \t]++)*(?:\S*/)?)?"
    r"(?:python[0-9.]*|pypy[0-9.]*|uv)(?=[ \t\r\n]|$)"
)
# ``warnings.catch_warnings`` swaps the process-global filter list, and analyzer
# nodes run on concurrent graph worker threads. Two such blocks that exit out of
# order can leave their ``ignore`` filter installed for the whole process, so
# the ownership parse serializes its block.
_PYTHON_OWNERSHIP_PARSE_LOCK = Lock()
_PYTHON_TEMPLATE_STRING_STARTS = frozenset(
    {tokenize.FSTRING_START, getattr(tokenize, "TSTRING_START", tokenize.FSTRING_START)}
)
_PYTHON_TEMPLATE_STRING_ENDS = frozenset(
    {tokenize.FSTRING_END, getattr(tokenize, "TSTRING_END", tokenize.FSTRING_END)}
)

# Start offsets and matching end offsets of a module's outermost string and
# comment tokens, in source order. Results are shared, so they are immutable.
PythonLiteralSpans = tuple[tuple[int, ...], tuple[int, ...]]

# One static scan of a module asks for its spans twice: the declared-marker
# pass proves string closers, and the tool-misuse shell parser proves token
# ownership. The most recent results are remembered so both share one parse.
# Entries are keyed by length and hash and confirmed by string equality, which
# is an identity check when both callers hold the same file-cache string. A
# second entry absorbs a scan of another file on a concurrent analyzer thread.
# Oversized source is never stored, and eviction bounds how long a module's
# text outlives its scan. This lock guards only the entries, so a lookup never
# waits for another thread's parse.
_PYTHON_LITERAL_SPANS_CACHE_SIZE = 2
_PYTHON_LITERAL_SPANS_CACHE_LOCK = Lock()
_PYTHON_LITERAL_SPANS_CACHE: OrderedDict[tuple[int, int], tuple[str, PythonLiteralSpans | None]] = (
    OrderedDict()
)


def python_literal_spans(
    content: str,
    check_runtime: Callable[[], None],
) -> PythonLiteralSpans | None:
    """Return outermost string and comment token spans of valid Python source.

    Return ``None`` unless the whole text is accepted by the Python parser,
    so a fragment, malformed file, or shell-shebang polyglot cannot borrow
    host ownership. Every span is checked against the exact source text.
    Results, including ``None``, are memoized for the most recent sources.
    """
    if len(content) > MAX_PYTHON_AST_SOURCE_CHARS:
        return None
    key = (len(content), hash(content))
    with _PYTHON_LITERAL_SPANS_CACHE_LOCK:
        entry = _PYTHON_LITERAL_SPANS_CACHE.get(key)
        if entry is not None and entry[0] == content:
            _PYTHON_LITERAL_SPANS_CACHE.move_to_end(key)
        else:
            entry = None
    if entry is not None:
        # A shared result skips the parse, not the caller's runtime bound.
        check_runtime()
        return entry[1]
    # A runtime-limit exception propagates from here, so a proof that the
    # bound interrupted is never stored.
    spans = _python_literal_spans_uncached(content, check_runtime)
    with _PYTHON_LITERAL_SPANS_CACHE_LOCK:
        _PYTHON_LITERAL_SPANS_CACHE[key] = (content, spans)
        _PYTHON_LITERAL_SPANS_CACHE.move_to_end(key)
        while len(_PYTHON_LITERAL_SPANS_CACHE) > _PYTHON_LITERAL_SPANS_CACHE_SIZE:
            _PYTHON_LITERAL_SPANS_CACHE.popitem(last=False)
    return spans


def _python_literal_spans_uncached(
    content: str,
    check_runtime: Callable[[], None],
) -> PythonLiteralSpans | None:
    """Parse and tokenize ``content`` for :func:`python_literal_spans`."""
    # A leading U+FEFF byte-order mark fails the module parse below and keeps
    # every conservative bound. The file cache decodes with ``utf-8``, not
    # ``utf-8-sig``, so the AST analyzers already report such a file as a
    # syntax error; stripping the mark belongs with that decoding fix.
    if _LONE_CARRIAGE_RETURN_RE.search(content) is not None or (
        content.startswith("#!") and _PYTHON_SHEBANG_RE.match(content) is None
    ):
        return None
    check_runtime()
    try:
        # The lenient tokenizer accepts bytes such as ``$`` and a backtick as
        # operators. Requiring a module parse proves that every byte outside
        # the spans below is Python syntax, never a shell quote or expansion.
        # The scanned module's own parse already reports its compiler warnings
        # (for example an invalid ``"\d"`` escape). Repeating them here would
        # duplicate that output, and ``-W error`` would turn them into a
        # ``SyntaxError`` that silently drops ownership.
        with _PYTHON_OWNERSHIP_PARSE_LOCK, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(content)
    except (SyntaxError, ValueError, RecursionError):
        return None
    check_runtime()
    line_starts = [0]
    line_starts.extend(match.end() for match in re.finditer("\n", content))
    starts: list[int] = []
    ends: list[int] = []
    template_depth = 0
    template_start = 0

    def offset(position: tuple[int, int]) -> int:
        return line_starts[position[0] - 1] + position[1]

    try:
        for index, token in enumerate(tokenize.generate_tokens(io.StringIO(content).readline)):
            if index % 256 == 0:
                check_runtime()
            if token.type in _PYTHON_TEMPLATE_STRING_STARTS:
                if template_depth == 0:
                    template_start = offset(token.start)
                    if not content.startswith(token.string, template_start):
                        return None
                template_depth += 1
            elif token.type in _PYTHON_TEMPLATE_STRING_ENDS:
                template_depth -= 1
                end = offset(token.end)
                if template_depth < 0 or not content.endswith(token.string, 0, end):
                    return None
                if template_depth == 0:
                    starts.append(template_start)
                    ends.append(end)
            elif not template_depth and token.type in (tokenize.STRING, tokenize.COMMENT):
                # Inside a replacement field, nested strings and comments
                # belong to the enclosing f-string or t-string span.
                start, end = offset(token.start), offset(token.end)
                if content[start:end] != token.string:
                    return None
                starts.append(start)
                ends.append(end)
    except (tokenize.TokenError, SyntaxError, ValueError, IndexError):
        return None
    if template_depth or any(ends[index] > starts[index + 1] for index in range(len(starts) - 1)):
        return None
    return tuple(starts), tuple(ends)


class PythonStringClosers:
    """Lazily proven closing delimiters of one complete module's string literals.

    :meth:`closes_string` reports whether a source offset lies in the closing
    delimiter of an outermost string, f-string, or t-string token. Such a quote
    is Python syntax that ends a literal; it never opens text. The module is
    parsed and tokenized at most once, on the first query, so a caller that
    never asks pays nothing. Source that :func:`python_literal_spans` cannot
    prove answers ``False`` everywhere, which keeps the caller's lexical path.
    """

    def __init__(self, content: str, check_runtime: Callable[[], None]) -> None:
        self._content = content
        self._check_runtime = check_runtime
        self._spans: PythonLiteralSpans | None = None
        self._spans_computed = False

    def closes_string(self, offset: int) -> bool:
        """Return whether ``offset`` is in a proven string literal's closing delimiter."""
        if not self._spans_computed:
            self._spans = python_literal_spans(self._content, self._check_runtime)
            self._spans_computed = True
        if self._spans is None:
            return False
        starts, ends = self._spans
        index = bisect_right(starts, offset) - 1
        if index < 0 or offset >= ends[index]:
            return False
        content = self._content
        start, end = starts[index], ends[index]
        if content[start] == "#":
            return False
        return offset >= python_string_body(content, start, end)[1]


def python_string_body(content: str, start: int, end: int) -> tuple[int, int]:
    """Return the body offsets of the string token at ``content[start:end]``."""
    # Skip a string prefix such as ``rb``, ``f`` or ``t`` to the opening
    # quote. The closing delimiter repeats it, three times when tripled.
    opener = start
    while content[opener] not in "'\"":
        opener += 1
    width = 3 if content.startswith(content[opener] * 3, opener, end) else 1
    return opener + width, end - width


def python_string_value_bounds_allowed(
    content: str,
    check_runtime: Callable[[], None],
    *,
    allow_dynamic_subprocess: bool = True,
) -> bool:
    """Allow isolated string-value bounds only without a recognized shell path.

    This is a module-wide syntax guard, not a claim of whole-program inertness.
    Any recognized shell sink, escaping sink/module alias, reflection, dynamic
    import, or ambiguous subprocess shell mode keeps the original whole-source
    fallback. Direct subprocess calls with default/false shell mode are eligible
    unless their literal argv names a shell. Unknown application calls
    and nonliteral argv retain the scanner's existing interprocedural limits.
    """
    if len(content) > MAX_PYTHON_AST_SOURCE_CHARS:
        return False
    check_runtime()
    with _PYTHON_OWNERSHIP_PARSE_LOCK, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            tree = ast.parse(content)
        except (SyntaxError, ValueError, RecursionError):
            return False
    check_runtime()
    known_modules = {
        name: name for name in ("os", "sys", "subprocess", "asyncio", "builtins", "importlib")
    }
    known_modules.update(
        posix="os", nt="os", __builtins__="builtins", posixpath="os.path", ntpath="os.path"
    )
    modules = dict(known_modules)
    imported: dict[str, tuple[str, str]] = {}
    nodes = []
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        check_runtime()
        nodes.append(node)
        parents.update((child, node) for child in ast.iter_child_nodes(node))
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in known_modules:
                    modules[alias.asname or root] = (
                        "os.path"
                        if alias.asname and alias.name == "os.path"
                        else known_modules[root]
                    )
        elif isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                return False
            root = (node.module or "").split(".")[0]
            if root in known_modules:
                if node.module != root:
                    return False  # Module descendants may expose their shell-capable parent.
                for alias in node.names:
                    imported[alias.asname or alias.name] = (known_modules[root], alias.name)

    reflection = {
        "exec",
        "eval",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "globals",
        "locals",
        "vars",
    }
    subprocess_calls = {"run", "Popen", "call", "check_call", "check_output"}
    subprocess_values = {
        "CompletedProcess",
        "CalledProcessError",
        "SubprocessError",
        "TimeoutExpired",
        "PIPE",
        "STDOUT",
        "DEVNULL",
    }
    os_values = {
        "environ",
        "getenv",
        "name",
        "path",
        "fspath",
        "PathLike",
        "sep",
        "altsep",
        "pathsep",
        "linesep",
        "curdir",
        "pardir",
        "devnull",
        "getcwd",
        "getcwdb",
        "cpu_count",
        "listdir",
        "scandir",
        "walk",
        "stat",
        "lstat",
        "access",
        "F_OK",
        "R_OK",
        "W_OK",
        "X_OK",
    }
    shells = {
        "sh",
        "bash",
        "dash",
        "ksh",
        "zsh",
        "fish",
        "ash",
        "csh",
        "tcsh",
        "pwsh",
        "powershell",
        "cmd",
        "cmd.exe",
    }

    module_names_shell = any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.replace("\\", "/").rsplit("/", 1)[-1].lower().removesuffix(".exe") in shells
        for node in nodes
    )
    path_calls = {
        "abspath",
        "basename",
        "commonpath",
        "commonprefix",
        "dirname",
        "exists",
        "expanduser",
        "expandvars",
        "getatime",
        "getctime",
        "getmtime",
        "getsize",
        "isabs",
        "isdir",
        "isfile",
        "islink",
        "ismount",
        "join",
        "lexists",
        "normcase",
        "normpath",
        "realpath",
        "relpath",
        "samefile",
        "sameopenfile",
        "samestat",
        "split",
        "splitdrive",
        "splitext",
        "splitroot",
    }

    def safe_path_call(attribute: str, node: ast.AST) -> bool:
        parent = parents.get(node)
        return attribute in path_calls and isinstance(parent, ast.Call) and parent.func is node

    def safe_subprocess_call(node: ast.AST) -> bool:
        if module_names_shell:
            return False  # Indirect argv may refer to an explicit local shell literal.
        parent = parents.get(node)
        if not isinstance(parent, ast.Call) or parent.func is not node:
            return False  # An assigned/passed callable can acquire later operands.
        if len(parent.args) > 1 or any(isinstance(arg, ast.Starred) for arg in parent.args):
            return False
        for keyword in parent.keywords:
            if keyword.arg is None or keyword.arg == "executable":
                return False
            if keyword.arg == "shell" and not (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is False
            ):
                return False
        argv = (
            parent.args[0]
            if parent.args
            else next((keyword.value for keyword in parent.keywords if keyword.arg == "args"), None)
        )
        if not allow_dynamic_subprocess and not (
            isinstance(argv, (ast.List, ast.Tuple))
            and argv.elts
            and isinstance(argv.elts[0], ast.Constant)
            and isinstance(argv.elts[0].value, str)
        ):
            return False  # A bundled helper can receive shell argv from its caller.
        # A shell may accept -c after other flags or through argv concatenation.
        # Keep the entire module conservative whenever literal argv names one;
        # there is no need to emulate each shell's argument parser here.
        for item in ast.walk(argv) if argv is not None else ():
            check_runtime()
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                if (
                    item.value.replace("\\", "/").rsplit("/", 1)[-1].lower().removesuffix(".exe")
                    in shells
                ):
                    return False
                if re.search(
                    r"(?:^|\s)(?:\S*/)?(?:sh|bash|dash|ksh|zsh|fish|ash|csh|tcsh|pwsh|powershell|cmd)(?:\.exe)?(?:\s|$)",
                    item.value,
                    re.IGNORECASE,
                ):
                    return False
        return True

    def safe_attribute(module: str, attribute: str, node: ast.AST) -> bool:
        if module == "os":
            if attribute == "path":
                parent = parents.get(node)
                # os.path is a module, not an inert value: only audited terminal
                # path calls may use it. Unknown descendants/escapes keep fallback.
                return (
                    isinstance(parent, ast.Attribute)
                    and parent.value is node
                    and safe_path_call(parent.attr, parent)
                )
            return attribute in os_values
        if module == "os.path":
            return safe_path_call(attribute, node)
        if module == "sys":
            return attribute in {
                "argv",
                "executable",
                "exit",
                "path",
                "platform",
                "version",
                "version_info",
                "stdin",
                "stdout",
                "stderr",
                "maxsize",
                "byteorder",
            }
        if module == "subprocess":
            return attribute in subprocess_values or (
                attribute in subprocess_calls and safe_subprocess_call(node)
            )
        if module == "asyncio":
            # Loop methods can execute shell commands through chained accesses.
            return False
        # builtins/importlib expose dynamic execution and module descendants.
        return False

    for node in nodes:
        check_runtime()
        if isinstance(node, ast.Name):
            if node.id in reflection:
                return False
            if node.id in modules:
                parent = parents.get(node)
                if (
                    not isinstance(parent, ast.Attribute)
                    or parent.value is not node
                    or not safe_attribute(modules[node.id], parent.attr, parent)
                ):
                    return False
            if node.id in imported and not safe_attribute(*imported[node.id], node):
                return False
        elif isinstance(node, ast.Attribute):
            # A safe first attribute does not own the rest of a chain, e.g.
            # os.path.os.system or an event loop's subprocess_shell method.
            if (
                (node.attr.startswith("__") and node.attr.endswith("__"))
                or node.attr in reflection
                or node.attr
                in {
                    "system",
                    "popen",
                    "create_subprocess_shell",
                    "subprocess_shell",
                    "subprocess_exec",
                    "__dict__",
                    "execl",
                    "execle",
                    "execlp",
                    "execlpe",
                    "execv",
                    "execve",
                    "execvp",
                    "execvpe",
                    "spawnl",
                    "spawnle",
                    "spawnlp",
                    "spawnlpe",
                    "spawnv",
                    "spawnve",
                    "spawnvp",
                    "spawnvpe",
                    "posix_spawn",
                    "posix_spawnp",
                }
            ):
                return False
        elif isinstance(node, ast.ImportFrom) and node.module in known_modules:
            # Imported shell sinks are unsafe even if an alias is later rebound.
            if known_modules[node.module] == "os" and any(
                alias.name not in os_values for alias in node.names
            ):
                return False
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in modules or node.name in imported:
                return False
        elif isinstance(node, ast.arg) and (node.arg in modules or node.arg in imported):
            return False
    return True
