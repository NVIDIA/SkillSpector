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
from html.parser import HTMLParser
from threading import Lock

from skillspector.javascript_tokens import browser_escaped_list_shell_text
from skillspector.python_ast import MAX_PYTHON_AST_SOURCE_CHARS, parse_python_source

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

# Three consumers ask for a module's spans during one scan: the declared-marker
# pass proves string closers, the tool-misuse shell parser proves token
# ownership, and AR2 proves comment ownership. Recent results are remembered so
# they share one parse. Entries are keyed by length and hash and confirmed by
# string equality, which is an identity check when the callers hold the same
# file-cache string.
#
# The consumers run in different analyzer nodes (the marker pass in every
# static-pattern node). The graph starts all analyzer nodes together on worker
# threads; the CLI and MCP server set no ``max_concurrency``, so the pool has
# Python's default size, min(32, CPUs + 4). Every node walks the components in
# the same order but at its own pace, so requests for other modules arrive
# between a module's first and last consumer. An entry holds one module however
# many consumers ask, so eight entries keep a result while up to seven other
# modules are requested in that gap. A wider gap only repeats the parse.
# Oversized source is never stored, so eight entries hold at most eight times
# MAX_PYTHON_AST_SOURCE_CHARS, which equals one scan's AST cache budget
# (MAX_PYTHON_AST_CACHE_SOURCE_CHARS), and eviction bounds how long that text
# outlives its scan. This lock guards only the entries, so a lookup never waits
# for another thread's parse.
_PYTHON_LITERAL_SPANS_CACHE_SIZE = 8
_PYTHON_LITERAL_SPANS_CACHE_LOCK = Lock()
_PYTHON_LITERAL_SPANS_CACHE: OrderedDict[tuple[int, int], tuple[str, PythonLiteralSpans | None]] = (
    OrderedDict()
)

_HTML_SCRIPT_OPEN_RE = re.compile(r"<script[ \t\r\n]*>", re.IGNORECASE)
_HTML_SCRIPT_CLOSE_RE = re.compile(r"</script[ \t\r\n]*>", re.IGNORECASE)


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


def python_fstring_literal_spans(
    content: str,
    start: int,
    end: int,
    check_runtime: Callable[[], None],
) -> PythonLiteralSpans | None:
    """Return exact constant-text runs of one proven outer f-string token.

    Replacement expressions and their format specifications never gain literal
    ownership. Doubled braces keep both original source bytes in their run.
    """
    owners = python_literal_spans(content, check_runtime)
    if owners is None:
        return None
    owner_index = bisect_right(owners[0], start) - 1
    if owner_index < 0 or (owners[0][owner_index], owners[1][owner_index]) != (start, end):
        return None
    source = content[start:end]
    line_starts = [0, *(match.end() for match in re.finditer("\n", source))]
    starts: list[int] = []
    ends: list[int] = []
    template_depth = field_depth = 0
    complete = False

    def offset(position: tuple[int, int]) -> int:
        return start + line_starts[position[0] - 1] + position[1]

    try:
        for index, token in enumerate(tokenize.generate_tokens(io.StringIO(source).readline)):
            if index % 256 == 0:
                check_runtime()
            if token.type in _PYTHON_TEMPLATE_STRING_STARTS:
                if not template_depth and (
                    token.type != tokenize.FSTRING_START or offset(token.start) != start
                ):
                    return None
                template_depth += 1
            elif token.type in _PYTHON_TEMPLATE_STRING_ENDS:
                template_depth -= 1
                if template_depth == 0:
                    complete = offset(token.end) == end and field_depth == 0
            elif template_depth == 1 and token.type == tokenize.OP:
                if token.string == "{":
                    field_depth += 1
                elif token.string == "}":
                    field_depth -= 1
            elif template_depth == 1 and not field_depth and token.type == tokenize.FSTRING_MIDDLE:
                literal_start, literal_end = offset(token.start), offset(token.end)
                if (
                    token.string.endswith(("{", "}"))
                    and content[literal_end : literal_end + 1] == token.string[-1]
                ):
                    literal_end += 1
                raw = content[literal_start:literal_end]
                if raw.replace("{{", "{").replace("}}", "}") != token.string:
                    return None
                if ends and ends[-1] == literal_start:
                    ends[-1] = literal_end
                else:
                    starts.append(literal_start)
                    ends.append(literal_end)
    except (tokenize.TokenError, SyntaxError, ValueError, IndexError):
        return None
    if not complete or template_depth or field_depth:
        return None
    return tuple(starts), tuple(ends)


def _browser_html_fields(
    literal: ast.JoinedStr,
    field_kind: Callable[[ast.AST], str | None],
    check_runtime: Callable[[], None],
) -> bool:
    """Accept one ordinary HTML document with inert, escaped Python fields."""
    marker = "__skillspector_python_html_field_"
    fields: dict[str, str | None] = {}
    chunks: list[str] = []
    for index, value in enumerate(literal.values):
        check_runtime()
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            if marker in value.value:
                return False
            chunks.append(value.value)
        elif (
            isinstance(value, ast.FormattedValue)
            and value.conversion == -1
            and value.format_spec is None
        ):
            key = f"{marker}{index}__"
            fields[key] = field_kind(value.value)
            chunks.append(key)
        else:
            return False
    if any(kind is None for kind in fields.values()):
        return False

    class Document(HTMLParser):
        valid = True
        context = ""
        executable_scripts = 0

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            check_runtime()
            if tag not in {
                "html",
                "head",
                "title",
                "meta",
                "style",
                "body",
                "div",
                "span",
                "aside",
                "main",
                "section",
                "input",
                "button",
                "strong",
                "ul",
                "li",
                "h1",
                "h2",
                "h3",
                "p",
                "script",
            }:
                self.valid = False
            names = [name for name, _ in attrs]
            if len(names) != len(set(names)):
                self.valid = False
            for name, value in attrs:
                if name not in {
                    "id",
                    "class",
                    "title",
                    "lang",
                    "charset",
                    "name",
                    "content",
                    "type",
                    "placeholder",
                } and not name.startswith("data-"):
                    self.valid = False
                if value is not None and marker in value:
                    self.valid = False
            attributes = dict(attrs)
            if tag == "meta" and not (
                set(attributes) == {"charset"}
                or attributes.get("name") == "viewport"
                and set(attributes) == {"name", "content"}
            ):
                self.valid = False
            if tag == "script":
                if not attrs:
                    self.executable_scripts += 1
                    self.context = "script"
                elif attributes.get("type") == "application/json" and set(attributes) <= {
                    "id",
                    "type",
                }:
                    self.context = "json"
                else:
                    self.valid = False
            elif tag in {"style", "title"}:
                self.context = tag

        def handle_endtag(self, tag: str) -> None:
            if tag in {"script", "style", "title"}:
                self.context = ""

        def handle_data(self, data: str) -> None:
            check_runtime()
            if self.context == "style" and re.search(
                r"(?:@import|url\s*\(|expression\s*\(|\\)", data, re.IGNORECASE
            ):
                self.valid = False
            if marker not in data:
                return
            if self.context == "json":
                if fields.pop(data.strip(), None) != "json":
                    self.valid = False
            elif self.context in {"script", "style"}:
                self.valid = False
            else:
                for key in re.findall(marker + r"[0-9]+__", data):
                    if fields.pop(key, None) != "text":
                        self.valid = False

        def handle_comment(self, data: str) -> None:
            if marker in data:
                self.valid = False

    parser = Document(convert_charrefs=False)
    parser.feed("".join(chunks))
    parser.close()
    return parser.valid and parser.executable_scripts == 1 and not fields


def _python_browser_strings(
    content: str, check_runtime: Callable[[], None]
) -> set[tuple[int, int]]:
    """Prove explicit pathlib consumers and escaped HTML field boundaries.

    Only direct write_text arguments and direct returns from unaliased local
    functions used exclusively by those writes qualify. Unknown consumers,
    rebinding, mutation and unsupported scope forms retain the shell reading.
    Ordinary pathlib/JSON/text protocols are assumed, as with Node assertions;
    explicit local protocol overrides fall back. This is not a proof against
    arbitrary runtime getters or monkeypatching outside the inspected source.
    """
    with _PYTHON_OWNERSHIP_PARSE_LOCK, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = parse_python_source(content, "<browser ownership>")
    tree = parsed.tree
    if tree is None:
        return set()
    parents: dict[ast.AST, ast.AST] = {}
    scopes: dict[ast.AST, ast.AST] = {}
    bindings: dict[tuple[ast.AST, str], list[ast.AST]] = {}
    returns: dict[ast.AST, list[ast.Return]] = {}
    references: dict[str, list[ast.Name]] = {}
    nodes: list[ast.AST] = []
    pending: list[tuple[ast.AST, ast.AST]] = [(tree, tree)]
    while pending:
        node, scope = pending.pop()
        if len(nodes) >= 100_000:
            return set()
        if len(nodes) % 256 == 0:
            check_runtime()
        nodes.append(node)
        scopes[node] = scope
        if isinstance(node, ast.Return):
            returns.setdefault(scope, []).append(node)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            references.setdefault(node.id, []).append(node)
        names: list[str] = []
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names = [node.id]
        elif isinstance(node, ast.arg):
            names = [node.arg]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
            if node.name in {
                "replace",
                "__str__",
                "__format__",
                "__fspath__",
                "__html__",
                "__getattribute__",
                "__getattr__",
                "__getitem__",
            }:
                return set()
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.asname or alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names = [node.name]
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.NamedExpr, ast.Match)):
            return set()
        if (
            "*" in names
            or isinstance(node, ast.Attribute)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.attr
            in {
                "Path",
                "write_text",
                "resolve",
                "cwd",
                "home",
                "escape",
                "dumps",
                "replace",
                "__code__",
                "__defaults__",
                "__kwdefaults__",
            }
        ):
            return set()
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in {"eval", "exec", "setattr", "delattr", "globals", "locals", "vars"}
            or isinstance(node, ast.Attribute)
            and node.attr in {"setattr", "delattr", "__dict__", "__setattr__"}
        ):
            return set()
        for name in names:
            if name in {
                "replace",
                "__str__",
                "__format__",
                "__fspath__",
                "__html__",
                "__getattribute__",
                "__getattr__",
                "__getitem__",
            }:
                return set()
            bindings.setdefault((scope, name), []).append(node)
        child_scope = (
            node
            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.Lambda,
                    ast.ListComp,
                    ast.SetComp,
                    ast.DictComp,
                    ast.GeneratorExp,
                ),
            )
            else scope
        )
        for child in ast.iter_child_nodes(node):
            parents[child] = node
            pending.append((child, child_scope))

    budget = 100_000

    def binding(name: ast.Name) -> ast.AST | None:
        nonlocal budget
        budget -= 1
        if budget <= 0:
            return None
        if budget % 256 == 0:
            check_runtime()
        scope = scopes[name]
        first = True
        while True:
            # Methods do not close over class-local bindings. Function closures
            # and comprehension bindings do shadow a module import.
            candidates = (
                []
                if not first and isinstance(scope, ast.ClassDef)
                else bindings.get((scope, name.id), [])
            )
            if candidates:
                return candidates[0] if len(candidates) == 1 else None
            if scope is tree:
                return None
            scope = scopes[scope]
            first = False
            budget -= 1
            if budget <= 0:
                return None

    def imported(node: ast.AST, module: str, name: str | None = None) -> bool:
        if not isinstance(node, ast.Name):
            return False
        owner = binding(node)
        if isinstance(owner, ast.Import) and name is None:
            return any(
                alias.name == module and alias.asname is None and node.id == module
                for alias in owner.names
            )
        if isinstance(owner, ast.ImportFrom) and owner.module == module and owner.level == 0:
            return any(
                alias.name == name and alias.asname is None and node.id == name
                for alias in owner.names
            )
        return False

    def assigned(node: ast.Name) -> ast.AST | None:
        owner = binding(node)
        parent = parents.get(owner) if owner is not None else None
        if isinstance(parent, ast.Assign) and parent.targets == [owner]:
            return parent.value
        if isinstance(parent, ast.AnnAssign) and parent.target is owner:
            return parent.value
        return None

    def annotation(node: ast.AST) -> bool:
        parent = parents.get(node)
        return bool(
            isinstance(parent, ast.arg)
            and parent.annotation is node
            or isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
            and parent.returns is node
            or isinstance(parent, ast.AnnAssign)
            and parent.annotation is node
        )

    # Audit every explicit use of the trusted imports, including aliases that
    # could let another call replace a constructor or escaping implementation.
    for name in ("Path", "json", "escape"):
        for reference in references.get(name, []):
            check_runtime()
            if annotation(reference):
                continue
            module, member = {
                "Path": ("pathlib", "Path"),
                "json": ("json", None),
                "escape": ("html", "escape"),
            }[name]
            if not imported(reference, module, member):
                return set()
            parent = parents.get(reference)
            if (
                name in {"Path", "escape"}
                and isinstance(parent, ast.Call)
                and parent.func is reference
            ):
                continue
            if isinstance(parent, ast.Attribute) and isinstance(parent.ctx, ast.Load):
                allowed = {"Path": {"cwd", "home"}, "json": {"loads", "dumps"}, "escape": set()}[
                    name
                ]
                call = parents.get(parent)
                if parent.attr in allowed and isinstance(call, ast.Call) and call.func is parent:
                    continue
            return set()

    path_bindings: set[ast.AST] = set()
    path_functions: set[ast.FunctionDef] = set()

    def path(node: ast.AST, depth: int = 0) -> bool:
        if depth >= 16 or budget <= 0:
            return False
        if isinstance(node, ast.Name):
            value = assigned(node)
            owner = binding(node)
            if owner is not None:
                path_bindings.add(owner)
            return value is not None and path(value, depth + 1)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            # An arbitrary RHS can run __rtruediv__ and replace the receiver.
            native_text = isinstance(node.right, ast.JoinedStr) or (
                isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)
            )
            return native_text and path(node.left, depth + 1)
        if not isinstance(node, ast.Call):
            return False
        if imported(node.func, "pathlib", "Path"):
            return True
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in {"cwd", "home"} and imported(node.func.value, "pathlib", "Path"):
                return not node.args and not node.keywords
            if node.func.attr == "resolve":
                return path(node.func.value, depth + 1)
        if isinstance(node.func, ast.Name):
            function = binding(node.func)
            if (
                isinstance(function, ast.FunctionDef)
                and scopes[function] is tree
                and not function.decorator_list
            ):
                path_functions.add(function)
                values = returns.get(function, [])
                return bool(values) and all(
                    part.value is not None and path(part.value, depth + 1) for part in values
                )
        return False

    def written(node: ast.AST) -> bool:
        scope = scopes[node]
        if scope is not tree and (
            not isinstance(scope, ast.FunctionDef) or scopes[scope] is not tree
        ):
            return False
        parent = parents.get(node)
        return bool(
            isinstance(parent, ast.Call)
            and len(parent.args) == 1
            and parent.args[0] is node
            and len(parent.keywords) == 1
            and parent.keywords[0].arg == "encoding"
            and isinstance(parent.keywords[0].value, ast.Constant)
            and parent.keywords[0].value.value in {"utf-8", "utf8", "UTF-8", "UTF8"}
            and isinstance(parent.func, ast.Attribute)
            and parent.func.attr == "write_text"
            and path(parent.func.value)
        )

    def path_use(node: ast.AST, depth: int = 0) -> bool:
        """A selected Path receiver cannot escape through a bound method/property."""
        if depth >= 16:
            return False
        parent = parents.get(node)
        if isinstance(parent, ast.FormattedValue):
            return True
        if not isinstance(parent, ast.Attribute) or not isinstance(parent.ctx, ast.Load):
            return False
        if parent.attr == "parent":
            return path_use(parent, depth + 1)
        call = parents.get(parent)
        if not isinstance(call, ast.Call) or call.func is not parent:
            return False
        if parent.attr in {"write_text", "mkdir", "as_uri"}:
            # These return native scalar/None values, not the selected receiver.
            return True
        if parent.attr == "resolve" and not call.args and not call.keywords:
            return path_use(call, depth + 1)
        return False

    def field_kind(node: ast.AST, depth: int = 0) -> str | None:
        if depth >= 16 or budget <= 0:
            return None
        if isinstance(node, ast.Name):
            value = assigned(node)
            return None if value is None else field_kind(value, depth + 1)
        if not isinstance(node, ast.Call):
            return None
        if (
            imported(node.func, "html", "escape")
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], (ast.Constant, ast.Name, ast.Subscript))
        ):
            return "text"
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "replace"
            and len(node.args) == 2
            and not node.keywords
        ):
            before, after = node.args
            value = node.func.value
            if (
                isinstance(before, ast.Constant)
                and before.value == "</"
                and isinstance(after, ast.Constant)
                and after.value == "<\\/"
                and isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "dumps"
                and imported(value.func.value, "json")
                and len(value.args) == 1
                and not isinstance(value.args[0], ast.Starred)
                and all(
                    keyword.arg == "separators"
                    and isinstance(keyword.value, ast.Tuple)
                    and len(keyword.value.elts) == 2
                    and all(
                        isinstance(item, ast.Constant) and isinstance(item.value, str)
                        for item in keyword.value.elts
                    )
                    for keyword in value.keywords
                )
            ):
                return "json"
        return None

    result: set[tuple[int, int]] = set()
    for node in nodes:
        if (
            not isinstance(node, ast.JoinedStr)
            or not node.values
            or not isinstance(node.values[0], ast.Constant)
            or not isinstance(node.values[0].value, str)
            or not node.values[0].value.lstrip().lower().startswith("<!doctype html>")
        ):
            continue
        check_runtime()
        consumed = written(node)
        parent = parents.get(node)
        function = scopes[node]
        if (
            isinstance(parent, ast.Return)
            and isinstance(function, ast.FunctionDef)
            and scopes[function] is tree
            and not function.decorator_list
            and returns.get(function) == [parent]
        ):
            uses = references.get(function.name, [])
            consumed = bool(uses) and all(
                binding(part) is function
                and isinstance(parents[part], ast.Call)
                and parents[part].func is part
                and written(parents[part])
                for part in uses
            )
        if consumed and _browser_html_fields(node, field_kind, check_runtime):
            start_column = parsed.character_column(node.lineno, node.col_offset)
            end_column = parsed.character_column(node.end_lineno, node.end_col_offset)
            if start_column is not None and end_column is not None:
                result.add(
                    (
                        parsed.line_character_starts[node.lineno - 1] + start_column,
                        parsed.line_character_starts[node.end_lineno - 1] + end_column,
                    )
                )
    for reference_nodes in references.values():
        for reference in reference_nodes:
            check_runtime()
            owner = binding(reference)
            if budget <= 0:
                return set()
            parent = parents.get(reference)
            if owner in path_functions:
                if not isinstance(parent, ast.Call) or parent.func is not reference:
                    return set()
            if owner in path_bindings:
                if not path_use(reference):
                    return set()
    return result if budget > 0 else set()


def python_browser_template_shell_text(content: str, check_runtime: Callable[[], None]) -> str:
    """Project display delimiters in constant browser scripts of HTML f-strings.

    Neither Python replacement fields nor escape decoding may establish a
    script boundary. Original lengths and every non-delimiter byte survive.
    """
    if "`" not in content or "<!doctype html>" not in content.lower():
        return content
    owners = python_literal_spans(content, check_runtime)
    if owners is None:
        return content
    browser_strings = _python_browser_strings(content, check_runtime)
    if not browser_strings:
        return content
    output: list[str] | None = None
    for start, end in zip(*owners, strict=True):
        check_runtime()
        if (start, end) not in browser_strings:
            continue
        if content[start] == "#":
            continue
        body_start, body_end = python_string_body(content, start, end)
        if "f" not in content[start:body_start].lower():
            continue
        body = content[body_start:body_end]
        if not body.lstrip().lower().startswith(
            "<!doctype html>"
        ) or not body.rstrip().lower().endswith("</html>"):
            continue
        runs = python_fstring_literal_spans(content, start, end, check_runtime)
        if runs is None:
            continue
        for literal_start, literal_end in zip(*runs, strict=True):
            search_start = literal_start
            while (
                script := _HTML_SCRIPT_OPEN_RE.search(content, search_start, literal_end)
            ) is not None:
                check_runtime()
                closing = _HTML_SCRIPT_CLOSE_RE.search(content, script.end(), literal_end)
                if closing is None:
                    break
                script_start, script_end = script.end(), closing.start()
                search_start = closing.end()
                raw = content[script_start:script_end]
                if "\\" in raw:
                    continue
                decoded: list[str] = []
                source_spans: list[tuple[int, int]] = []
                cursor = script_start
                while cursor < script_end:
                    if len(decoded) % 4096 == 0:
                        check_runtime()
                    width = 2 if content[cursor : cursor + 2] in {"{{", "}}"} else 1
                    decoded.append(content[cursor])
                    source_spans.append((cursor, cursor + width))
                    cursor += width
                source = "".join(decoded)
                projected = browser_escaped_list_shell_text(source, check_runtime)
                if projected == source:
                    continue
                if output is None:
                    output = list(content)
                for position, (original, replacement) in enumerate(
                    zip(source, projected, strict=True)
                ):
                    if position % 4096 == 0:
                        check_runtime()
                    if original != replacement:
                        lower, upper = source_spans[position]
                        output[lower:upper] = " " * (upper - lower)
    check_runtime()
    return content if output is None else "".join(output)


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
