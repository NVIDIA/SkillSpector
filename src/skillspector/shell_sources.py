# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded values reconstructed from complete, already-loaded shell sources."""

import hashlib
import re
from bisect import bisect_right
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

from skillspector.shell_helper_argv import is_managed_shell_variable

MAX_SHELL_SOURCE_FILES = 128
MAX_SHELL_SOURCE_CHARS = 8_000_000
MAX_SHELL_VALUE_CHARS = 65_536
MAX_SHELL_BINDINGS = 256
MAX_SHELL_BINDING_DEFINITIONS = 8192
MAX_SHELL_VALUE_DEPTH = 16
SAFE_PATH_MARKER = "__skillspector_generated_path__"

ReadWord = Callable[[str, int, Callable[[], None]], tuple[str | None, int, bool]]
SubstitutionEnd = Callable[[str, int, int], int | None]
HeredocHeader = Callable[[str, int, int], tuple[tuple[str, bool, bool], int] | None]
HeredocBodies = Callable[
    [str, int, int, tuple[tuple[str, bool, bool], ...], Callable[[], None]], int | None
]


@dataclass(frozen=True, slots=True)
class ShellSourceContext:
    path: str
    sources: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ShellSourceOrigin:
    path: str
    start: int
    end: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ShellSourceValue:
    text: str
    kind: str = "literal"
    origins: tuple[ShellSourceOrigin, ...] = ()
    safe_spans: tuple[tuple[int, int], ...] = ()
    argv: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class ShellSourceCommand:
    start: int
    end: int
    words: tuple[tuple[str, int, int], ...]
    redirects: tuple[tuple[int, int], ...] = ()
    conditional: bool = False


@dataclass(frozen=True, slots=True)
class GeneratedShellFile:
    path: ShellSourceValue
    value: ShellSourceValue
    body_start: int
    body_end: int
    declaration_start: int
    declaration_end: int


def _canonical_path(path: str, *, allow_parent: bool = False) -> str | None:
    if not path or path.startswith("/") or any(item in path for item in ("\\", "\x00", ":", "!/")):
        return None
    parts: list[str] = []
    for part in path.split("/"):
        if part in {"", "."}:
            if not allow_parent:
                return None
        elif part == "..":
            if not allow_parent or not parts:
                return None
            parts.pop()
        else:
            parts.append(part)
    return "/".join(parts) or ("." if allow_parent else None)


class ShellSourceResolver:
    """Resolve ordered source bindings without executing shell or reading files.

    The context describes uploaded bytes. Explicit parent-scope mutation and
    unsupported source/eval effects refuse proof; it is not a claim that an
    arbitrary runtime filesystem or external environment cannot change.
    """

    def __init__(
        self,
        context: ShellSourceContext,
        read_word: ReadWord,
        substitution_end: SubstitutionEnd,
        heredoc_header: HeredocHeader,
        heredoc_bodies: HeredocBodies,
        check_runtime: Callable[[], None],
        *,
        content: str | None = None,
        inherited_values: Mapping[str, ShellSourceValue] | None = None,
        source_identity: bool = True,
        work_budget: list[int] | None = None,
    ) -> None:
        self.context = context
        self.read_word = read_word
        self.substitution_end = substitution_end
        self.heredoc_header = heredoc_header
        self.heredoc_bodies = heredoc_bodies
        self.check_runtime = check_runtime
        self.content = context.sources.get(context.path, "") if content is None else content
        self.source_identity = source_identity
        self.inherited_values = inherited_values or {}
        self.child_contexts: Callable[[int], Iterator[tuple[ShellSourceResolver, int]]] | None = (
            None
        )
        self.substitution_contexts: (
            Callable[[int], Iterator[tuple[ShellSourceResolver, int]]] | None
        ) = None
        self.effect_source: Callable[[ShellSourceValue, int], ShellSourceValue | None] | None = None
        self.executable_spans: list[tuple[int, int, int, bool]] = []
        self.executable_children_complete = True
        self.lexically_valid = False
        self.bindings: dict[str, list[tuple[int, int, bool, bool]]] = {}
        self.binding_definitions = 0
        self.values: dict[tuple[str, int], ShellSourceValue | None] = {}
        self.resolving: set[tuple[str, int]] = set()
        self.hashes: dict[str, str] = {}
        self.commands: list[ShellSourceCommand] = []
        self.child_commands: list[ShellSourceCommand] = []
        self.dynamic_callees: list[tuple[int, int]] = []
        self.literal_heredocs: list[tuple[int, int, int]] = []
        self.heredoc_counts: dict[int, int] = {}
        self.input_redirect_commands: set[int] = set()
        self.mapfile_process_commands: set[int] = set()
        self.definite_commands: set[int] = set()
        self.fail_stop_commands: set[int] = set()
        self.functions: set[str] = set()
        self.invalid_names: set[str] = set()
        self.errexit = False
        self.pipefail = False
        self.controls: list[str] = []
        self.loop_scopes: list[tuple[str, tuple[str, ...]] | None] = []
        self.loop_components: dict[int, dict[str, tuple[str, ...]]] = {}
        self.invalid_loop_names: set[str] = set()
        self.valid = (
            context.path in context.sources
            and len(context.sources) <= MAX_SHELL_SOURCE_FILES
            and sum(map(len, context.sources.values())) <= MAX_SHELL_SOURCE_CHARS
            and all(_canonical_path(path) == path for path in context.sources)
            and all(SAFE_PATH_MARKER not in source for source in context.sources.values())
        )
        self.work_budget = work_budget or [
            min(MAX_SHELL_SOURCE_CHARS, 16 * len(self.content) + MAX_SHELL_VALUE_CHARS)
        ]
        if self.valid:
            self._collect_bindings()

    def _spend(self, amount: int = 1) -> bool:
        self.check_runtime()
        self.work_budget[0] -= amount
        if self.work_budget[0] < 0:
            self.valid = False
        return self.valid

    def audit_spend(self, amount: int = 1) -> bool:
        """Charge a refusal audit without reviving semantic value proof."""
        self._spend(amount)
        return self.lexically_valid and self.work_budget[0] >= 0

    def audit_word(self, start: int, end: int) -> ShellSourceValue | None:
        return self.word(start, end, literal_only=not self.valid)

    def audit_path_values(
        self, start: int, end: int, command_start: int
    ) -> tuple[ShellSourceValue, ...] | None:
        """Resolve only finite literal loop filename components for write audits."""
        value = self.audit_word(start, end)
        if value is not None:
            return (value,)
        if not self.valid:
            return None
        raw = self.content[start:end]
        suffix = re.fullmatch(
            r'"(.+)/\$(?:([A-Za-z_][A-Za-z0-9_]*)|\{([A-Za-z_][A-Za-z0-9_]*)\})"', raw
        )
        if suffix is None:
            return None
        name = suffix[2] or suffix[3]
        names = self.loop_components.get(command_start, {}).get(name)
        if not names or name in self.invalid_loop_names:
            return None
        prefix = self.word(start + suffix.start(1), start + suffix.end(1), assignment_value=True)
        if prefix is None or prefix.kind != "safe_path" or ".." in prefix.text.split("/"):
            return None
        return tuple(
            ShellSourceValue(
                prefix.text + "/" + item, "safe_path", prefix.origins, prefix.safe_spans
            )
            for item in names
        )

    def literal_children(self, before_offset: int) -> Iterator[tuple["ShellSourceResolver", int]]:
        if self.child_contexts is not None:
            yield from self.child_contexts(before_offset)

    def executable_children(
        self, before_offset: int
    ) -> Iterator[tuple["ShellSourceResolver", int]]:
        if self.substitution_contexts is not None:
            yield from self.substitution_contexts(before_offset)

    def _read_source_value(
        self, target: ShellSourceValue, use_offset: int
    ) -> ShellSourceValue | None:
        """Read uploaded or reconstructed bytes only after their effect proof."""
        if target.kind not in {"bundle_path", "safe_path"}:
            return None
        if target.kind == "bundle_path" and target.text in self.context.sources:
            if source_is_unmodified(self, target.text, use_offset):
                source = self.context.sources[target.text]
                if not self._spend(len(source)):
                    return None
                digest = self.hashes.get(target.text)
                if digest is None:
                    digest = hashlib.sha256(source.encode()).hexdigest()
                    self.hashes[target.text] = digest
                return ShellSourceValue(
                    source, "code", (ShellSourceOrigin(target.text, 0, len(source), digest),)
                )
        if self.effect_source is None or not self.valid:
            return None
        value = self.effect_source(target, use_offset)
        if (
            value is None
            or value.kind != "code"
            or not value.origins
            or not self._spend(len(value.text))
        ):
            return None
        return value

    def _source_origins(
        self, target: ShellSourceValue, value: ShellSourceValue, start: int, end: int
    ) -> tuple[ShellSourceOrigin, ...]:
        if (
            target.kind == "bundle_path"
            and self.context.sources.get(target.text) == value.text
            and len(value.origins) == 1
            and value.origins[0].path == target.text
            and value.origins[0].start == 0
            and value.origins[0].end == len(value.text)
        ):
            return (ShellSourceOrigin(target.text, start, end, value.origins[0].sha256),)
        # Copies, transformations and archives retain the original provenance;
        # offsets in reconstructed bytes do not become uploaded-file offsets.
        return value.origins

    def _collect_executable_spans(
        self, start: int, end: int, *, heredoc: bool = False, child_scope: bool = False
    ) -> None:
        """Record complete command substitutions, without evaluating their data.

        Quoted heredocs never call this method. In expandable heredocs quotes
        are data, while backslash still protects dollar and backtick bytes.
        Unsupported executable forms refuse the source-write proof rather
        than disappearing because the outer word parser consumed them.
        """
        quote: str | None = None
        cursor = start
        while cursor < end:
            if not self._spend():
                return
            character = self.content[cursor]
            if not heredoc and quote == "'":
                if character == "'":
                    quote = None
            elif character == "\\":
                if cursor + 1 < end and (
                    not heredoc and quote is None or self.content[cursor + 1] in '$`"\\\n\r'
                ):
                    cursor += 2
                    continue
            elif not heredoc and character in "'\"" and (quote is None or character == quote):
                quote = None if quote == character else character
            elif character == "`":
                self.executable_children_complete = False
                return
            elif self.content.startswith(("$((", "$["), cursor):
                # Arithmetic may mutate variables and recursively interpret
                # array subscripts. This optional value pass does not model it.
                self.valid = False
                self.executable_children_complete = False
                return
            elif self.content.startswith("$(", cursor):
                stop = self.substitution_end(self.content, cursor, end)
                if stop is None or stop > end or len(self.executable_spans) >= 1024:
                    self.executable_children_complete = False
                    return
                self.executable_spans.append((cursor + 2, stop - 1, stop, child_scope))
                cursor = stop
                continue
            elif self.content.startswith("${", cursor):
                parameter = re.match(r"\$\{[A-Za-z_][A-Za-z0-9_]*(?::?=)", self.content[cursor:end])
                if parameter is not None:
                    self.valid = False
                    return
            cursor += 1

    def _collect_bindings(self) -> None:
        """Collect only commands with proven parent-shell lexical ownership.

        The generic scanner's clause candidates include quoted/generated code;
        they cannot establish assignment identity. Here words consume their
        complete substitutions, heredoc bodies are skipped at their newline,
        and ordinary subshells never publish parent bindings. Unsupported
        parent control/function syntax refuses this optional proof.
        """
        cursor = 0
        child_depth = 0
        function_depth = 0
        words: list[tuple[str, int, int]] = []
        redirects: list[tuple[int, int]] = []
        pending: list[tuple[str, bool, bool]] = []
        pending_owners: list[tuple[int, bool]] = []
        conditional = False

        def finish(end: int, follows_operator: bool = False) -> None:
            if words or redirects:
                if len(self.commands) + len(self.child_commands) >= 8192:
                    self.valid = False
                    return
                destination = (
                    self.child_commands if child_depth or function_depth else self.commands
                )
                destination.append(
                    ShellSourceCommand(
                        words[0][1] if words else redirects[0][0],
                        end,
                        tuple(words),
                        tuple(redirects),
                        conditional or follows_operator,
                    )
                )
            words.clear()
            redirects.clear()

        while cursor < len(self.content) and self.valid:
            if not self._spend():
                return
            character = self.content[cursor]
            if character in " \t\r":
                cursor += 1
                continue
            if self.content.startswith("\\\r\n", cursor) or self.content.startswith("\\\n", cursor):
                cursor += 3 if self.content.startswith("\\\r\n", cursor) else 2
                continue
            if character == "#" and (cursor == 0 or self.content[cursor - 1].isspace()):
                newline = self.content.find("\n", cursor)
                cursor = len(self.content) if newline < 0 else newline
                continue
            if character in "\n;|&":
                finish(cursor, character in "|&")
                conditional = character in "|&"
                cursor += 1
                if character == "\n" and pending:
                    for item, (owner, child_scope) in zip(pending, pending_owners, strict=True):
                        stop = self.heredoc_bodies(
                            self.content, cursor, len(self.content), (item,), self.check_runtime
                        )
                        if stop is None:
                            self.valid = False
                            return
                        if not item[2]:
                            self._collect_executable_spans(
                                cursor,
                                stop,
                                heredoc=True,
                                child_scope=bool(child_depth or function_depth),
                            )
                        elif not item[1] and not child_scope:
                            record_end = stop - int(self.content[stop - 1 : stop] == "\n")
                            body_end = self.content.rfind("\n", cursor, record_end) + 1
                            self.literal_heredocs.append((owner, cursor, max(cursor, body_end)))
                        cursor = stop
                    pending.clear()
                    pending_owners.clear()
                continue
            if character == "(":
                # A preceding command word makes this function/array grammar,
                # whose effects are not represented by this small source pass.
                if len(words) == 1 and not child_depth and not function_depth and not pending:
                    declaration = re.match(
                        r"\([ \t]*\)[ \t\r\n]*\{(?=\s|;|$)", self.content[cursor : cursor + 4096]
                    )
                    name = words[0][0]
                    if declaration is not None and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                        self.functions.add(name)
                        words.clear()
                        function_depth = 1
                        cursor += declaration.end()
                        continue
                if words or pending:
                    self.valid = False
                    return
                child_depth += 1
                cursor += 1
                continue
            if character == ")":
                finish(cursor)
                if not child_depth or pending:
                    self.valid = False
                    return
                child_depth -= 1
                cursor += 1
                continue
            if (
                character in "{}"
                and not child_depth
                and (
                    cursor + 1 == len(self.content)
                    or self.content[cursor + 1].isspace()
                    or self.content[cursor + 1] in ";|&()<>"
                )
            ):
                if not function_depth or words or pending:
                    self.valid = False
                    return
                function_depth += 1 if character == "{" else -1
                cursor += 1
                continue
            if character in "<>":
                if self.content.startswith("<<", cursor) and not self.content.startswith(
                    "<<<", cursor
                ):
                    header = self.heredoc_header(self.content, cursor, len(self.content))
                    if header is None or len(pending) >= 32:
                        self.valid = False
                        return
                    owner = words[0][1] if words else cursor
                    self.heredoc_counts[owner] = self.heredoc_counts.get(owner, 0) + 1
                    item, cursor = header
                    pending.append(item)
                    pending_owners.append((owner, bool(child_depth or function_depth)))
                    continue
                operator = re.match(
                    r"(?:<<<|>>|>&|<&|>\||<>|>|<)", self.content[cursor : cursor + 3]
                )
                if operator is None:
                    self.valid = False
                    return
                owner = words[0][1] if words else cursor
                prior_input = owner in self.input_redirect_commands
                if "<" in operator[0]:
                    self.input_redirect_commands.add(owner)
                cursor += len(operator[0])
                while cursor < len(self.content) and self.content[cursor] in " \t":
                    cursor += 1
                if cursor == len(self.content) or self.content[cursor] in "\r\n;|&":
                    self.valid = False
                    return
                if operator[0] == "<" and self.content.startswith("<(", cursor):
                    raw_words = [self.content[left:right] for _, left, right in words]
                    if (
                        child_depth
                        or function_depth
                        or prior_input
                        or redirects
                        or pending
                        or len(raw_words) != 3
                        or raw_words[:2] != ["mapfile", "-t"]
                        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw_words[2]) is None
                        or is_managed_shell_variable(raw_words[2])
                    ):
                        self.valid = False
                        return
                    stop = self.substitution_end(self.content, cursor, len(self.content))
                    if stop is None or not cursor < stop <= len(self.content):
                        self.valid = False
                        return
                    following = stop
                    while following < len(self.content) and self.content[following] in " \t":
                        following += 1
                    if (
                        following < len(self.content) and self.content[following] not in "\r\n;|&"
                    ) or not self._spend(following - cursor):
                        self.valid = False
                        return
                    self.mapfile_process_commands.add(owner)
                    # Lexical ownership is not an effects proof for this
                    # asynchronous body. Every cached read must use the ordered
                    # auditor, which checks the original producer and consumer.
                    self.executable_children_complete = False
                    cursor = stop
                    continue
                _, stop, limited = self.read_word(self.content, cursor, self.check_runtime)
                if limited or stop <= cursor or not self._spend(stop - cursor):
                    self.valid = False
                    return
                if ">" in operator[0]:
                    redirects.append((cursor, stop))
                self._collect_executable_spans(
                    cursor, stop, child_scope=bool(child_depth or function_depth)
                )
                cursor = stop
                continue
            word, stop, limited = self.read_word(self.content, cursor, self.check_runtime)
            if limited or word is None or stop <= cursor or not self._spend(stop - cursor):
                self.valid = False
                return
            if len(words) >= 256:
                self.valid = False
                return
            words.append((word, cursor, stop))
            self._collect_executable_spans(
                cursor, stop, child_scope=bool(child_depth or function_depth)
            )
            cursor = stop
        finish(cursor)
        if child_depth or function_depth or pending:
            self.valid = False
            return
        self.lexically_valid = self.valid
        for command in self.commands:
            self._record_command(command)
        if self.controls:
            self.valid = False
        if self.functions & {
            "sed",
            "dirname",
            "pwd",
            "cd",
            "mktemp",
            "printf",
            "set",
            "trap",
            "export",
        }:
            self.valid = False
        for start, end in self.dynamic_callees:
            # A later loop assignment can change an earlier lexical call on its
            # next iteration. Resolve only after all invalidations are known.
            executable = self.word(start, end)
            if (
                executable is None
                or "/" not in executable.text
                or executable.text in self.functions
            ):
                self.valid = False
                break

    def _record_command(self, command: ShellSourceCommand) -> None:
        if not self.valid:
            return
        if not self.controls and not command.conditional:
            self.definite_commands.add(command.start)
            if self.errexit:
                self.fail_stop_commands.add(command.start)
        active_loops = {
            name: values
            for scope in self.loop_scopes
            if scope is not None
            for name, values in (scope,)
        }
        if active_loops:
            self.loop_components[command.start] = active_loops
        assignments: list[tuple[str, int, int]] = []
        words = command.words
        cursor = 0
        while cursor < len(words):
            keyword = words[cursor][0]
            if keyword in {"if", "while", "until", "for"}:
                if len(self.controls) >= MAX_SHELL_VALUE_DEPTH:
                    self.valid = False
                    return
                self.controls.append("if" if keyword == "if" else "loop")
                self.loop_scopes.append(None)
                if keyword == "for":
                    if (
                        cursor + 1 >= len(words)
                        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", words[cursor + 1][0]) is None
                    ):
                        self.valid = False
                        return
                    name = words[cursor + 1][0]
                    self.invalid_names.add(name)
                    if name in active_loops:
                        self.invalid_loop_names.add(name)
                    candidates = words[cursor + 3 :]
                    if (
                        cursor + 2 < len(words)
                        and words[cursor + 2][0] == "in"
                        and 0 < len(candidates) <= 32
                    ):
                        components = []
                        for _, left, right in candidates:
                            literal = self.word(left, right, literal_only=True)
                            if (
                                literal is None
                                or literal.kind != "literal"
                                or literal.text in {".", ".."}
                                or re.fullmatch(r"[-A-Za-z0-9_.]+", literal.text) is None
                            ):
                                break
                            components.append(literal.text)
                        else:
                            self.loop_scopes[-1] = (name, tuple(components))
                    return
                cursor += 1
            elif keyword in {"then", "do", "else", "elif"}:
                if not self.controls:
                    self.valid = False
                    return
                cursor += 1
            elif keyword in {"fi", "done"}:
                expected = "if" if keyword == "fi" else "loop"
                if not self.controls or self.controls.pop() != expected:
                    self.valid = False
                    return
                self.loop_scopes.pop()
                cursor += 1
            else:
                break
        if cursor == len(words):
            return
        while cursor < len(words):
            _, start, end = words[cursor]
            assignment = re.match(r"([A-Za-z_][A-Za-z0-9_]*)=", self.content[start:end])
            if assignment is None:
                break
            assignments.append((assignment[1], start + assignment.end(), end))
            self.invalid_loop_names.add(assignment[1])
            cursor += 1
        # Explicit native settings can change a child's startup or executable
        # selection. PATH alone has a separate exact first-directory proof.
        if any(is_managed_shell_variable(name) and name != "PATH" for name, _, _ in assignments):
            self.valid = False
            return
        # Prefix assignments to an external invocation are child environment,
        # not parent bindings. Refuse special-builtin prefixes whose scope may
        # persist, rather than promoting them to a unique parent definition.
        if cursor == len(words):
            for name, start, end in assignments:
                if is_managed_shell_variable(name) or name == "LD_PRELOAD":
                    self.valid = False
                    return
                if len(self.bindings) >= MAX_SHELL_BINDINGS and name not in self.bindings:
                    self.valid = False
                    return
                self.binding_definitions += 1
                if self.binding_definitions > MAX_SHELL_BINDING_DEFINITIONS:
                    self.valid = False
                    return
                if command.conditional or self.controls:
                    # A loop's later lexical assignment can precede an earlier
                    # use on another iteration. Keep conditional writes globally
                    # unknown until control-flow-specific histories are proved.
                    self.invalid_names.add(name)
                self.bindings.setdefault(name, []).append(
                    (start, end, self.errexit and len(assignments) == 1, self.pipefail)
                )
            return
        self.invalid_names.update(name for name, _, _ in assignments)
        executable = words[cursor][0]
        _, start, end = words[cursor]
        raw_executable = self.content[start:end]
        if raw_executable in {"!", "time"}:
            # These reserved wrappers can invoke a mutating parent builtin.
            self.valid = False
            return
        literal_executable = self.word(start, end, literal_only=True)
        if literal_executable is None and raw_executable not in {"[", "[["}:
            self.dynamic_callees.append((start, end))
        elif literal_executable is not None:
            executable = literal_executable.text
        if executable in self.functions:
            # A declared function may mutate globals; command-substitution
            # calls were consumed as child words and cannot mutate the parent.
            self.valid = False
            return
        if executable == "set":
            named_option: str | None = None
            for option, _left, _right in words[cursor + 1 :]:
                if named_option is not None:
                    if option == "errexit":
                        self.errexit = (
                            named_option == "-" and not command.conditional and not self.controls
                        )
                    elif option == "pipefail":
                        self.pipefail = (
                            named_option == "-" and not command.conditional and not self.controls
                        )
                    elif option != "nounset":
                        self.errexit = False
                        self.pipefail = False
                    named_option = None
                    continue
                if not re.fullmatch(r"[-+][a-zA-Z]+", option):
                    self.errexit = False
                    self.pipefail = False
                    break
                if "e" in option[1:]:
                    self.errexit = (
                        not command.conditional and not self.controls and option[0] == "-"
                    )
                if "o" in option[1:]:
                    if not option.endswith("o"):
                        self.errexit = False
                        self.pipefail = False
                        break
                    named_option = option[0]
            if named_option is not None:
                self.errexit = False
                self.pipefail = False
        if executable in {
            "case",
            "esac",
            "function",
            "source",
            ".",
            "eval",
            "alias",
            "unalias",
            "enable",
            "hash",
        }:
            self.valid = False
            return
        if executable == "mapfile" and command.start in self.mapfile_process_commands:
            # The exact plain destination is known; its array contents are not.
            # Invalidate older bindings too, rather than retaining captured code.
            name = words[cursor + 2][0]
            self.invalid_names.add(name)
            self.invalid_loop_names.add(name)
            return
        if (
            executable
            in {
                "declare",
                "typeset",
                "local",
                "readonly",
                "unset",
                "read",
                "readarray",
                "mapfile",
                "let",
                "getopts",
            }
            or (executable == "printf" and any(word == "-v" for word, _, _ in words[cursor + 1 :]))
            or (
                executable == "wait"
                and any(word.startswith("-p") for word, _, _ in words[cursor + 1 :])
            )
        ):
            # Unknown nameref/array/destination grammar can affect any binding.
            self.valid = False
            return
        if executable == "printf":
            arguments = words[cursor + 1 :]
            if arguments and arguments[0][0] == "--":
                arguments = arguments[1:]
            if (
                not arguments
                or any(
                    marker in self.content[arguments[0][1] : arguments[0][2]]
                    for marker in ("$", "`")
                )
                or re.search(r"%[-+ #0'.*0-9]*[hjlLtz]*n", arguments[0][0].replace("%%", ""))
                is not None
                or arguments[0][0].startswith("-")
            ):
                self.valid = False
                return
        if executable == "export":
            for word, start, end in words[cursor + 1 :]:
                match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:=.*)?", word, re.DOTALL)
                if match is None or self.content[start:end] != word:
                    self.valid = False
                    return
                self.invalid_names.add(match[1])
                self.invalid_loop_names.add(match[1])
                if is_managed_shell_variable(match[1]) or match[1] == "LD_PRELOAD":
                    self.valid = False
                    return
        if any(
            re.match(r"[A-Za-z_][A-Za-z0-9_]*(?:\+=|\[)", self.content[start:end])
            for _, start, end in words
        ):
            self.valid = False
        if executable in {"command", "builtin"}:
            # The optional resolver does not mistake wrapped mutation for an
            # external command. The ordinary raw scanner still handles it.
            self.valid = False
        if executable == "trap":
            values = [word for word, _, _ in words[cursor + 1 :]]
            if (
                len(values) != 2
                or values[1] not in {"EXIT", "0"}
                or re.fullmatch(r'rm -rf "\$[A-Za-z_][A-Za-z0-9_]*"', values[0]) is None
            ):
                self.valid = False

    def _name(self, name: str, depth: int, use: int) -> ShellSourceValue | None:
        if (
            not self.valid
            or is_managed_shell_variable(name)
            or name in self.invalid_names
            or depth >= MAX_SHELL_VALUE_DEPTH
        ):
            return None
        spans = self.bindings.get(name, ())
        index = bisect_right(spans, use, key=lambda span: span[0]) - 1
        if index < 0:
            return self.inherited_values.get(name)
        span = spans[index]
        key = name, span[0]
        if span[1] > use or key in self.resolving:
            return None
        if key in self.values:
            return self.values[key]
        self.resolving.add(key)
        result = self.word(
            span[0],
            span[1],
            depth=depth + 1,
            assignment_value=True,
            fail_stop=span[2],
            pipefail=span[3],
        )
        self.resolving.remove(key)
        self.values[key] = result
        return result

    def generated_files(self) -> Iterator[GeneratedShellFile]:
        """Expose exact bytes from unconditional literal cat-to-file heredocs."""
        if not self.valid or not self.source_identity:
            return
        commands = {command.start: command for command in self.commands}
        counts: dict[int, int] = {}
        for owner, _, _ in self.literal_heredocs:
            counts[owner] = counts.get(owner, 0) + 1
        for owner, left, right in self.literal_heredocs:
            command = commands.get(owner)
            if (
                command is None
                or owner not in self.fail_stop_commands
                or counts[owner] != 1
                or self.heredoc_counts.get(owner) != 1
                or owner in self.input_redirect_commands
                or len(command.words) != 1
                or len(command.redirects) != 1
            ):
                continue
            head = self.word(command.words[0][1], command.words[0][2])
            if head is None or head.kind != "literal" or head.text != "cat":
                continue
            target_start = command.redirects[0][0]
            if (
                re.search(
                    r"(?<![A-Za-z0-9_>&])>[ \t]*$", self.content[command.words[0][2] : target_start]
                )
                is None
            ):
                continue
            destination = self.word(*command.redirects[0])
            if (
                destination is None
                or destination.kind != "safe_path"
                or ".." in destination.text.split("/")
                or not re.fullmatch(r"[-A-Za-z0-9_.]+", destination.text.rpartition("/")[2])
            ):
                continue
            if right - left > MAX_SHELL_VALUE_CHARS or not self._spend(right - left):
                return
            digest = self.hashes.get(self.context.path)
            if digest is None:
                digest = hashlib.sha256(self.content.encode()).hexdigest()
                self.hashes[self.context.path] = digest
            value = ShellSourceValue(
                self.content[left:right],
                "code",
                (ShellSourceOrigin(self.context.path, left, right, digest),),
            )
            yield GeneratedShellFile(destination, value, left, right, command.start, command.end)

    def path_prefix(self, start: int, end: int) -> ShellSourceValue | None:
        """Prove only a first PATH entry before the inherited search path."""
        raw = self.content[start:end]
        match = re.fullmatch(r'PATH="(.+):\$(?:PATH|\{PATH\})"', raw, re.DOTALL)
        if match is None:
            return None
        value = self.word(start + match.start(1), start + match.end(1), assignment_value=True)
        return (
            value
            if value is not None
            and value.kind == "safe_path"
            and ":" not in value.text
            and ".." not in value.text.split("/")
            else None
        )

    @staticmethod
    def _join(parts: list[ShellSourceValue]) -> ShellSourceValue | None:
        text = "".join(part.text for part in parts)
        if len(text) > MAX_SHELL_VALUE_CHARS:
            return None
        origins: list[ShellSourceOrigin] = []
        seen: set[ShellSourceOrigin] = set()
        spans: list[tuple[int, int]] = []
        offset = 0
        kinds = {part.kind for part in parts}
        for part in parts:
            for origin in part.origins:
                if origin not in seen:
                    seen.add(origin)
                    origins.append(origin)
            if len(origins) > 1024:
                return None
            spans.extend((offset + start, offset + end) for start, end in part.safe_spans)
            offset += len(part.text)
        kind = next(
            (kind for kind in ("code", "safe_path", "bundle_path") if kind in kinds), "literal"
        )
        if kind == "bundle_path":
            normalized = _canonical_path(text, allow_parent=True)
            if normalized is None:
                return None
            text = normalized
        return ShellSourceValue(text, kind, tuple(origins), tuple(spans))

    def word(
        self,
        start: int,
        end: int,
        *,
        depth: int = 0,
        assignment_value: bool = False,
        fail_stop: bool = False,
        literal_only: bool = False,
        pipefail: bool = False,
    ) -> ShellSourceValue | None:
        spend = self.audit_spend if literal_only else self._spend
        if (
            (not self.valid and not (literal_only and self.lexically_valid))
            or depth >= MAX_SHELL_VALUE_DEPTH
            or end - start > MAX_SHELL_VALUE_CHARS
            or not spend(end - start)
        ):
            return None
        parts: list[ShellSourceValue] = []
        literal: list[str] = []
        quote: str | None = None
        cursor = start

        def flush() -> None:
            if literal:
                parts.append(ShellSourceValue("".join(literal)))
                literal.clear()

        while cursor < end:
            character = self.content[cursor]
            if character == quote:
                quote = None
            elif quote == "'":
                literal.append(character)
            elif character in "'\"" and quote is None:
                quote = character
            elif character == "\\":
                if self.content.startswith("\\\r\n", cursor):
                    cursor += 3
                    continue
                if self.content.startswith("\\\n", cursor):
                    cursor += 2
                    continue
                if cursor + 1 >= end:
                    return None
                escaped = self.content[cursor + 1]
                if quote == '"' and escaped not in '$`"\\':
                    literal.append("\\")
                literal.append(escaped)
                cursor += 2
                continue
            elif character == "$":
                # A dollar that cannot begin a shell expansion is literal,
                # including the end-anchor in a quoted sed regex ("$/p").
                following = self.content[cursor + 1 : cursor + 2]
                if (
                    not following
                    or following
                    not in "([{ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_@*#?$!-\"'"
                    or quote == '"'
                    and following in "\"'"
                ):
                    literal.append("$")
                    cursor += 1
                    continue
                if literal_only:
                    return None
                flush()
                if self.content.startswith("$(", cursor):
                    stop = self.substitution_end(self.content, cursor, end)
                    if stop is None:
                        return None
                    if not fail_stop:
                        return None
                    value = self._substitution(cursor + 2, stop - 1, depth + 1, pipefail=pipefail)
                else:
                    name = re.match(
                        r"\$(?:([A-Za-z_][A-Za-z0-9_]*)|\{([A-Za-z_][A-Za-z0-9_]*)\})",
                        self.content[cursor:end],
                    )
                    if name is None:
                        return None
                    stop = cursor + name.end()
                    value = self._name(name[1] or name[2], depth + 1, cursor)
                if value is None:
                    return None
                # Unquoted expansion would require field-splitting/glob state.
                if quote is None and not assignment_value:
                    return None
                parts.append(value)
                cursor = stop
                continue
            elif character == "`" or (
                quote is None and (character.isspace() or character in "*?[]{}~;|&()<>")
            ):
                return None
            else:
                literal.append(character)
            cursor += 1
        if quote is not None:
            return None
        flush()
        return self._join(parts)

    def _words(self, start: int, end: int) -> list[tuple[str, int, int]] | None:
        words: list[tuple[str, int, int]] = []
        source = self.content[start:end]
        cursor = 0
        for _ in range(32):
            while cursor < len(source):
                if source.startswith("\\\r\n", cursor):
                    cursor += 3
                elif source.startswith("\\\n", cursor):
                    cursor += 2
                elif source[cursor] in " \t":
                    cursor += 1
                else:
                    break
            if cursor >= len(source):
                return words
            if source[cursor] in "\r\n;|&<>()":
                return None
            word, stop, limited = self.read_word(source, cursor, self.check_runtime)
            if word is None or limited or stop <= cursor or not self._spend(stop - cursor):
                return None
            words.append((word, start + cursor, start + stop))
            cursor = stop
        return None

    def _substitution(
        self, start: int, end: int, depth: int, *, pipefail: bool = False
    ) -> ShellSourceValue | None:
        if depth >= MAX_SHELL_VALUE_DEPTH or not self._spend(end - start):
            return None
        raw = self.content[start:end]
        if "|" in raw:
            return _heredoc_sed_substitution(self, start, end, depth) if pipefail else None
        if self.source_identity and re.fullmatch(r'dirname\s+"\$\{BASH_SOURCE\[0\]\}"', raw):
            parent = self.context.path.rpartition("/")[0] or "."
            return ShellSourceValue(parent, "bundle_path")
        directory = re.fullmatch(r"cd\s+(.+?)\s*&&\s*pwd", raw, re.DOTALL)
        if directory is not None:
            value = self.word(
                start + directory.start(1),
                start + directory.end(1),
                depth=depth + 1,
                fail_stop=True,
            )
            return value if value is not None and value.kind == "bundle_path" else None
        words = self._words(start, end)
        if not words:
            return None
        values = [word for word, _, _ in words]
        # Command/options/range literals must really be literal argv. Decoded
        # sentinels or unquoted glob syntax are not executable identities.
        for expected, left, right in words[:-1]:
            literal = self.word(left, right, depth=depth + 1)
            if literal is None or literal.kind != "literal" or literal.text != expected:
                return None
        if values[:2] == ["mktemp", "-d"] and len(words) == 3:
            template = values[2]
            literal = self.word(words[2][1], words[2][2], depth=depth + 1)
            if literal is None or literal.kind != "literal" or literal.text != template:
                return None
            if re.fullmatch(r"/tmp/[-A-Za-z0-9_.]+X{3,}", template) is None:
                return None
            value = re.sub(r"X+$", SAFE_PATH_MARKER, template)
            return ShellSourceValue(value, "safe_path", safe_spans=((0, len(value)),))
        if values[:1] == ["cat"] and len(words) == 2 and "cat" not in self.functions:
            target = self.word(words[1][1], words[1][2], depth=depth + 1)
            value = self._read_source_value(target, start) if target is not None else None
            if value is None or len(value.text) > MAX_SHELL_VALUE_CHARS:
                return None
            return ShellSourceValue(value.text.rstrip("\n"), "code", value.origins)
        if values[:2] != ["sed", "-n"] or len(words) < 4 or "sed" in self.functions:
            return None
        expressions: list[tuple[str, str]] = []
        cursor = 2
        while cursor < len(words) - 1:
            if values[cursor] == "-e":
                cursor += 1
            elif expressions:
                return None
            if cursor >= len(words) - 1:
                return None
            expression = re.fullmatch(
                r"/\^([A-Za-z_][A-Za-z0-9_]*\(\) \{)/,/\^\}/p", values[cursor]
            )
            if expression is None:
                return None
            expressions.append((expression[1], "}"))
            cursor += 1
        target = self.word(words[-1][1], words[-1][2], depth=depth + 1)
        value = self._read_source_value(target, start) if target is not None else None
        if value is None:
            return None
        source = value.text
        active = [False] * len(expressions)
        output: list[str] = []
        origins: list[ShellSourceOrigin] = []
        offset = size = 0
        while offset < len(source):
            newline = source.find("\n", offset)
            stop = len(source) if newline < 0 else newline + 1
            line = source[offset:stop]
            if not self._spend(len(line)):
                return None
            for index, (first, last) in enumerate(expressions):
                started = not active[index] and line.startswith(first)
                if started:
                    active[index] = True
                if active[index]:
                    output.append(line)
                    origins.extend(self._source_origins(target, value, offset, offset + len(line)))
                    size += len(line)
                    if size > MAX_SHELL_VALUE_CHARS or len(origins) > 1024:
                        return None
                    # sed tests the second regex only on following lines.
                    if not started and line.startswith(last):
                        active[index] = False
            offset = stop
        if not output or any(active):
            return None
        return ShellSourceValue("".join(output).rstrip("\n"), "code", tuple(dict.fromkeys(origins)))


def readonly_sed_program(program: str) -> bool:
    """Recognize the bounded sed grammar shared by source effect proofs."""
    # A substitution's w/e flags and standalone w/e commands can change
    # files or execute code without -i. Accept only a small literal grammar.
    if re.fullmatch(r"(?:[0-9$]+(?:,[0-9$]+)?)?[pdq](?:;[0-9$]*[pdq])*", program):
        return True
    if re.fullmatch(r"/\^[A-Za-z_][A-Za-z0-9_]*\(\) \{/,/\^\}/p", program):
        return True
    if re.fullmatch(
        r"/[A-Za-z_][A-Za-z0-9_]*=\$\(cat <<'[A-Za-z_][A-Za-z0-9_]*'/,/\^[A-Za-z_][A-Za-z0-9_]*\$/p",
        program,
    ):
        return True
    if len(program) < 4 or program[0] != "s":
        return False
    delimiter = program[1]
    if delimiter.isalnum() or delimiter.isspace() or delimiter == "\\":
        return False
    cursor = 2
    for _ in range(2):
        while cursor < len(program):
            if program[cursor] == "\\":
                cursor += 2
            elif program[cursor] == delimiter:
                cursor += 1
                break
            elif program[cursor] in "\r\n":
                return False
            else:
                cursor += 1
        else:
            return False
    return re.fullmatch(r"[gIp0-9]*", program[cursor:]) is not None


def readonly_file_arguments(
    arguments: tuple[str | None, ...], check_runtime: Callable[[], None]
) -> bool:
    """Prove a finite file inspection mode without compiling a magic database."""
    if len(arguments) > 128:
        return False
    options = True
    cursor = 0
    while cursor < len(arguments):
        check_runtime()
        value = arguments[cursor]
        if value is None or len(value) > MAX_SHELL_VALUE_CHARS:
            return False
        cursor += 1
        if not options or value == "-" or not value.startswith("-"):
            continue
        if value == "--":
            options = False
        elif re.fullmatch(r"-[bihLk]+", value) or value in {
            "--brief",
            "--mime",
            "--mime-type",
            "--mime-encoding",
            "--dereference",
            "--no-dereference",
            "--keep-going",
        }:
            continue
        elif value in {"-m", "--magic-file", "-F", "--separator"}:
            if cursor == len(arguments):
                return False
            check_runtime()
            operand = arguments[cursor]
            if operand is None or len(operand) > MAX_SHELL_VALUE_CHARS:
                return False
            cursor += 1
        elif value.startswith(("--magic-file=", "--separator=")):
            continue
        else:
            # Unknown options, abbreviations and clusters cannot hide -C or
            # --compile. A known option's operand and words after -- are data.
            return False
    return True


def source_is_unmodified(resolver: ShellSourceResolver, target_path: str, use_offset: int) -> bool:
    """Refuse a cached read after an explicit possible write to its source.

    This is an audit of inspected commands, not a promise that external programs
    or an unrelated process cannot change the filesystem. Uploaded script calls
    still require an effect proof; their presence in the cache is not one.
    Child commands share filesystem effects even when their variables are local.
    """
    readers = frozenset(
        {
            ":",
            "true",
            "false",
            "echo",
            "printf",
            "cat",
            "head",
            "tail",
            "grep",
            "egrep",
            "fgrep",
            "cut",
            "wc",
            "stat",
            "test",
            "[",
            "[[",
            "sha256sum",
            "sha1sum",
            "md5sum",
            "cmp",
            "diff",
            "readlink",
            "dirname",
            "ls",
            "set",
            "export",
            "exit",
            "return",
            "break",
            "continue",
            "sleep",
            "wait",
            "shift",
            "pwd",
            "cd",
            "mktemp",
            "local",
        }
    )
    writers = frozenset(
        {"rm", "unlink", "rmdir", "cp", "mv", "install", "ln", "tee", "truncate", "touch", "dd"}
    )
    controls = frozenset({"if", "then", "elif", "else", "fi", "while", "until", "do", "done", "!"})

    def touches(value: ShellSourceValue | None, *, subtree: bool = False) -> bool:
        if value is None:
            return True
        if value.kind == "audit_pid_file":
            if subtree:
                return True
            prefix, suffix = value.text.rpartition("/")[2].split("$$")
            # $$ contributes only decimal digits. This fact proves a filename
            # disjoint, never a directory ancestor or executable code value.
            return (
                re.fullmatch(
                    re.escape(prefix) + r"[0-9]+" + re.escape(suffix),
                    target_path.rpartition("/")[2],
                )
                is not None
            )
        if value.kind == "safe_path":
            # Do not let a path containing a proved temporary component escape
            # that directory again through appended parent traversals.
            return not (
                value.text.startswith("/tmp/")
                and SAFE_PATH_MARKER in value.text.split("/")[2]
                and ".." not in value.text.split("/")
            )
        path = _canonical_path(value.text, allow_parent=True)
        if value.kind == "bundle_path":
            return (
                path is None
                or path == "."
                or path == target_path
                or target_path.startswith(path + "/")
            )
        if value.kind != "literal":
            return True
        if subtree:
            # Without a bundle/fresh-root identity, a relative directory may
            # contain the source after a cwd change. Let the ordered effect
            # audit establish disjointness instead of comparing spellings.
            return True
        if path == target_path:
            return True
        # A literal filename without source identity can still name the source
        # from a different working directory. Do not assume the current cwd.
        return value.text.rstrip("/").rpartition("/")[2] == target_path.rpartition("/")[2]

    def visit(current: ShellSourceResolver, before: int, depth: int) -> bool:
        if depth > 8 or not current.lexically_valid or not current.audit_spend():
            return False
        literal_children = tuple(current.literal_children(before))
        executable_children = tuple(current.executable_children(before))
        if not current.executable_children_complete:
            return False
        children = (*literal_children, *executable_children)
        child_consumers = {consumer_offset for _, consumer_offset in literal_children}
        child_ids = {id(command) for command in current.child_commands}
        for command in (*current.commands, *current.child_commands):
            if not current.audit_spend():
                return False
            if command.start >= before:
                continue
            child_scope = id(command) in child_ids

            def resolve(
                left: int, right: int, *, child_scope: bool = child_scope
            ) -> ShellSourceValue | None:
                raw = current.content[left:right]
                pid_file = re.fullmatch(r'"(/tmp/[-A-Za-z0-9_.]*\$\$[-A-Za-z0-9_.]*)"', raw)
                if pid_file is not None:
                    return ShellSourceValue(pid_file[1], "audit_pid_file")
                # Parent bindings cannot prove a variable's value inside an
                # independently parsed subshell/function with local assignments.
                if child_scope and ("$" in raw or "`" in raw):
                    return None
                return current.audit_word(left, right)

            for left, right in command.redirects:
                value = resolve(left, right)
                prefix = current.content[max(command.start, left - 32) : left]
                if (
                    re.search(r"(?:>&|<&)[ \t]*$", prefix)
                    and value is not None
                    and re.fullmatch(r"[0-9]+|-", value.text)
                ):
                    continue
                destinations = (
                    (value,)
                    if value is not None
                    else current.audit_path_values(left, right, command.start)
                    if not child_scope
                    else None
                )
                if destinations is None or any(
                    touches(destination) for destination in destinations
                ):
                    return False
            words = command.words
            cursor = 0
            while cursor < len(words):
                token, left, right = words[cursor]
                assignment = re.match(r"([A-Za-z_][A-Za-z0-9_]*)=", current.content[left:right])
                if assignment is not None and is_managed_shell_variable(assignment[1]):
                    # Even a single-invocation PATH override can replace a
                    # reader or the shell which owns an inspected -c string.
                    return False
                if token in controls or assignment is not None:
                    cursor += 1
                    continue
                break
            if cursor == len(words):
                continue
            head, head_start, head_end = words[cursor]
            if head == "for" and current.content[head_start:head_end] == "for":
                # The iteration list is data. Its executable substitutions and
                # any redirections still belong to the surrounding audit.
                header = words[cursor + 1 :]
                if not header or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", header[0][0]) is None:
                    return False
                if len(header) > 1 and header[1][0] != "in":
                    return False
                continue
            executable = (
                ShellSourceValue(head)
                if head in {"[", "[["} and current.content[head_start:head_end] == head
                else resolve(head_start, head_end)
            )
            if executable is None:
                return False
            name = executable.text.rpartition("/")[2]
            if (
                executable.kind != "literal"
                or name in current.functions
                or executable.text not in {name, "/bin/" + name, "/usr/bin/" + name}
            ):
                # A bundled program or /tmp/cat may have arbitrary effects.
                # Native semantics belong to a proved identity, not basename.
                return False
            arguments = words[cursor + 1 :]
            if name == "env":
                invocation = decode_env_unset_command(current, words, cursor)
                if invocation is None:
                    return False
                index, _, _ = invocation
                target = resolve(words[index][1], words[index][2])
                if target is None:
                    return False
                if (
                    command.start in child_consumers
                    and target.kind == "literal"
                    and target.text
                    in {
                        prefix + shell
                        for prefix in ("", "/bin/", "/usr/bin/")
                        for shell in ("bash", "sh", "dash", "zsh", "ksh")
                    }
                ):
                    continue
                return False
            if name in {
                "command",
                "builtin",
                "sudo",
                "nohup",
                "timeout",
                "nice",
                "exec",
                "eval",
                "source",
                ".",
            }:
                # A wrapper may conceal a writer; its name is not evidence of
                # a read-only operation. Extend only through understood argv.
                return False
            if name == "trap":
                # Deferred handlers use bindings at invocation, including early
                # exits. Only the ordered effect audit can establish those.
                return False
            if name in {"bash", "sh", "dash", "zsh", "ksh"}:
                if command.start not in child_consumers:
                    return False
                # Only fully inspected child programs supply an effect audit.
                continue
            mutation = name in writers
            if name == "sed":
                programs: list[str] = []
                index = 0
                while index < len(arguments):
                    _, left, right = arguments[index]
                    value = resolve(left, right)
                    if value is None:
                        return False
                    if value.text == "-n" and not programs:
                        index += 1
                        continue
                    if value.text == "-e":
                        index += 1
                        if index == len(arguments):
                            return False
                        _, left, right = arguments[index]
                        value = resolve(left, right)
                        if value is None or value.kind != "literal":
                            return False
                        programs.append(value.text)
                    elif value.text == "--":
                        index += 1
                        if programs:
                            break
                        if index == len(arguments):
                            return False
                        _, left, right = arguments[index]
                        value = resolve(left, right)
                        if value is None or value.kind != "literal":
                            return False
                        programs.append(value.text)
                        index += 1
                        break
                    elif value.text.startswith("-"):
                        return False
                    elif not programs and value.kind == "literal":
                        programs.append(value.text)
                    else:
                        break
                    index += 1
                if not programs or any(
                    not current.audit_spend(len(program)) or not readonly_sed_program(program)
                    for program in programs
                ):
                    return False
                # GNU sed accepts options after input filenames. Every
                # remaining word must therefore be a literal filename, not an
                # unchecked -e/-f/-i option that changes the effect proof.
                for _, left, right in arguments[index:]:
                    value = resolve(left, right)
                    if value is None or value.text.startswith("-"):
                        return False
            if mutation:
                nonrecursive_pid_removal = name == "unlink" or (
                    name == "rm"
                    and all(
                        not word.startswith("-") or word in {"-f", "--force", "--"}
                        for word, _, _ in arguments
                    )
                )
                options = True
                for _, left, right in arguments:
                    if not current.audit_spend():
                        return False
                    value = resolve(left, right)
                    if options and value is not None and value.text == "--":
                        options = False
                        continue
                    if options and value is not None and value.text.startswith("-"):
                        if value.kind != "literal" or not re.fullmatch(r"--?[A-Za-z]+", value.text):
                            return False
                        continue
                    if name == "dd":
                        if value is None:
                            return False
                        if not value.text.startswith("of="):
                            continue
                        value = ShellSourceValue(value.text[3:], value.kind)
                    subtree = name in {"rm", "rmdir", "cp", "mv", "install", "ln"}
                    if (
                        value is not None
                        and value.kind == "audit_pid_file"
                        and nonrecursive_pid_removal
                    ):
                        subtree = False
                    if touches(value, subtree=subtree):
                        return False
            elif name == "file":
                values: list[str | None] = []
                for _, left, right in arguments:
                    if not current.audit_spend():
                        return False
                    value = resolve(left, right)
                    values.append(None if value is None else value.text)
                if not readonly_file_arguments(tuple(values), current.check_runtime):
                    return False
            elif name not in readers and name != "sed":
                # Opaque code and archive operations can write implicit paths.
                # Only the ordered effects audit may prove their preservation.
                return False
        for child, consumer_offset in children:
            if consumer_offset < before and not visit(child, len(child.content), depth + 1):
                return False
        return current.audit_spend()

    if _canonical_path(target_path) != target_path or not 0 <= use_offset <= len(resolver.content):
        return False
    return visit(resolver, use_offset, 0)


def decode_env_unset_command(
    resolver: ShellSourceResolver,
    words: tuple[tuple[str, int, int], ...],
    start: int,
) -> tuple[int, tuple[str, ...], dict[str, ShellSourceValue]] | None:
    """Decode only env -u NAME pairs and known, non-startup assignments.

    Returns the executable word index, removed names and explicit bindings.
    The caller still owns executable identity and child-code inspection.
    """
    if not 0 <= start < len(words) or len(words) - start > 128 or not resolver.audit_spend():
        return None
    executable = resolver.audit_word(words[start][1], words[start][2])
    if (
        executable is None
        or executable.kind != "literal"
        or executable.text not in {"env", "/bin/env", "/usr/bin/env"}
    ):
        return None
    cursor = start + 1
    removed: list[str] = []
    while cursor < len(words):
        flag = resolver.audit_word(words[cursor][1], words[cursor][2])
        if flag is None or flag.kind != "literal" or flag.text != "-u":
            break
        if cursor + 1 >= len(words) or len(removed) >= 32:
            return None
        name = resolver.audit_word(words[cursor + 1][1], words[cursor + 1][2])
        if (
            name is None
            or name.kind != "literal"
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name.text) is None
        ):
            return None
        removed.append(name.text)
        cursor += 2
    if not removed:
        return None
    environment: dict[str, ShellSourceValue] = {}
    execution_settings = {"LD_LIBRARY_PATH", "PS4", "PROMPT_COMMAND"}
    while cursor < len(words):
        _, left, right = words[cursor]
        assignment = re.match(r"([A-Za-z_][A-Za-z0-9_]*)=", resolver.content[left:right])
        if assignment is None:
            break
        name = assignment[1]
        if (
            is_managed_shell_variable(name)
            or name in execution_settings
            or name in environment
            or len(environment) >= 32
        ):
            return None
        value = resolver.word(
            left + assignment.end(), right, assignment_value=True, literal_only=not resolver.valid
        )
        if value is None or value.kind not in {"literal", "safe_path", "bundle_path"}:
            return None
        environment[name] = value
        cursor += 1
    if cursor == len(words):
        return None
    target = resolver.audit_word(words[cursor][1], words[cursor][2])
    if target is None or target.text.startswith("-") or "=" in target.text:
        return None
    return cursor, tuple(removed), environment


def _heredoc_sed_substitution(
    resolver: ShellSourceResolver, start: int, end: int, depth: int
) -> ShellSourceValue | None:
    """Read a literal heredoc range through an exact two-stage sed pipeline.

    The caller must already prove assignment failure stops execution and that
    pipefail is active. Without both, failure of the first sed can be hidden by
    the second stage. This function preserves sed's LF records and range order;
    it does not execute either command or infer arbitrary pipeline semantics.
    """
    if (
        depth >= MAX_SHELL_VALUE_DEPTH
        or not 0 <= start < end <= len(resolver.content)
        or end - start > MAX_SHELL_VALUE_CHARS
        or not resolver._spend(end - start)
    ):
        return None
    source = resolver.content[start:end]
    stages: list[list[tuple[str, int, int]]] = [[]]
    cursor = 0
    while cursor < len(source):
        if not resolver._spend():
            return None
        if source.startswith("\\\r\n", cursor):
            cursor += 3
            continue
        if source.startswith("\\\n", cursor):
            cursor += 2
            continue
        if source[cursor] in " \t":
            cursor += 1
            continue
        if source[cursor] == "|":
            if len(stages) != 1 or len(stages[0]) != 4:
                return None
            stages.append([])
            cursor += 1
            continue
        if source[cursor] in "\r\n;&<>()" or len(stages[-1]) >= 4:
            return None
        _, stop, limited = resolver.read_word(source, cursor, resolver.check_runtime)
        if limited or stop <= cursor or not resolver._spend(stop - cursor):
            return None
        value = resolver.word(start + cursor, start + stop, depth=depth + 1)
        if value is None:
            return None
        stages[-1].append((value.text, start + cursor, start + stop))
        cursor = stop
    if len(stages) != 2 or len(stages[0]) != 4 or len(stages[1]) != 2:
        return None
    first, second = stages
    if [word[0] for word in first[:2]] != ["sed", "-n"] or [word[0] for word in second] != [
        "sed",
        "1d;$d",
    ]:
        return None
    for _, left, right in (*first[:3], *second):
        value = resolver.word(left, right, depth=depth + 1)
        if value is None or value.kind != "literal":
            return None
    expression = re.fullmatch(
        r"/([A-Za-z_][A-Za-z0-9_]*=\$\(cat <<'([A-Za-z_][A-Za-z0-9_]*)')/,/\^([A-Za-z_][A-Za-z0-9_]*)\$/p",
        first[2][0],
    )
    if expression is None or expression[2] != expression[3]:
        return None
    target = resolver.word(first[3][1], first[3][2], depth=depth + 1)
    if "sed" in resolver.functions:
        return None
    value = resolver._read_source_value(target, start) if target is not None else None
    if value is None:
        return None
    cached = value.text
    selected: list[tuple[int, int]] = []
    active = False
    offset = size = 0
    while offset < len(cached):
        newline = cached.find("\n", offset)
        stop = len(cached) if newline < 0 else newline + 1
        if not resolver._spend(stop - offset):
            return None
        record = cached[offset:stop]
        started = not active and expression[1] in record
        if started:
            active = True
        if active:
            selected.append((offset, stop))
            size += stop - offset
            if size > MAX_SHELL_VALUE_CHARS or len(selected) > 1024:
                return None
            # sed starts checking its second regex on the following record.
            if not started and record.removesuffix("\n") == expression[2]:
                active = False
        offset = stop
    if active or len(selected) <= 2:
        return None
    # The second sed removes the first and last records of its entire input,
    # including when the first stage emits several disjoint matching ranges.
    selected = selected[1:-1]
    while selected:
        left, right = selected[-1]
        trimmed = right
        while trimmed > left and cached[trimmed - 1] == "\n":
            trimmed -= 1
        if trimmed == left:
            selected.pop()
        else:
            selected[-1] = (left, trimmed)
            break
    if not selected:
        return None
    return ShellSourceValue(
        "".join(cached[left:right] for left, right in selected),
        "code",
        tuple(
            dict.fromkeys(
                origin
                for left, right in selected
                for origin in resolver._source_origins(target, value, left, right)
            )
        ),
    )


def generated_file_is_unmodified(
    resolver: ShellSourceResolver,
    generated: GeneratedShellFile,
    callers: tuple[ShellSourceCommand, ...],
) -> bool:
    """Bind generated bytes to the complete, explicit PATH consumer set.

    The truncating cat establishes the inspected file contents. Later writes,
    aliases and directory escapes must be understood; a successful executable
    chmod must precede every caller. Like cached-source ownership, this checks
    inspected effects under ordinary built-ins, not unrelated filesystem races.
    No generated pathname becomes generally trusted executable input.
    """
    if (
        not resolver.valid
        or not resolver.source_identity
        or not callers
        or len(callers) > 128
        or generated.path.kind != "safe_path"
        or generated.declaration_start not in resolver.fail_stop_commands
        or not resolver.audit_spend()
    ):
        return False
    target = generated.path.text
    directory, _, basename = target.rpartition("/")
    if (
        not directory.startswith("/tmp/")
        or SAFE_PATH_MARKER not in directory
        or ".." in target.split("/")
    ):
        return False
    parent_commands = {command.start: command for command in resolver.commands}
    caller_starts = {command.start for command in callers}
    if len(caller_starts) != len(callers) or any(
        parent_commands.get(command.start) != command or command.start <= generated.declaration_end
        for command in callers
    ):
        return False
    last_use = max(caller_starts)
    chmod_offsets: list[int] = []
    controls = {"if", "then", "elif", "else", "fi", "while", "until", "do", "done", "!"}
    writers = {
        "rm",
        "unlink",
        "rmdir",
        "cp",
        "mv",
        "install",
        "ln",
        "tee",
        "truncate",
        "touch",
        "dd",
        "chmod",
        "chown",
        "chgrp",
    }

    def related(value: ShellSourceValue | None, *, ancestor: bool = False) -> bool:
        if value is None:
            return False
        if value.kind == "code":
            return directory in value.text
        path = value.text.rstrip("/")
        if path == target or path == directory or path.startswith(directory + "/"):
            return True
        if ancestor and path and directory.startswith(path + "/"):
            return True
        # Literal relative paths can name the file from an inspected cwd.
        return (
            value.kind == "literal"
            and re.fullmatch(r"[-A-Za-z0-9_./]+", path) is not None
            and path.rpartition("/")[2] in {basename, directory.rpartition("/")[2]}
        )

    def visit(current: ShellSourceResolver, anchor: int, depth: int) -> bool:
        if depth > 8 or not current.lexically_valid or not current.audit_spend():
            return False
        is_parent = current is resolver
        commands = (*current.commands, *current.child_commands)
        child_ids = {id(command) for command in current.child_commands}
        for command in commands:
            if not current.audit_spend():
                return False
            event = command.start if is_parent else anchor
            # The literal declaration replaces earlier bytes. The fresh file
            # is not borrowed as executable code before that declaration.
            if event <= generated.declaration_end:
                continue
            before_use = event <= last_use
            child_scope = id(command) in child_ids

            def word(
                left: int, right: int, *, assignment: bool = False, child_scope: bool = child_scope
            ) -> ShellSourceValue | None:
                if child_scope and any(
                    marker in current.content[left:right] for marker in ("$", "`")
                ):
                    return None
                return current.word(
                    left, right, assignment_value=assignment, literal_only=not current.valid
                )

            for left, right in command.redirects:
                value = word(left, right)
                prefix = current.content[max(command.start, left - 32) : left]
                if (
                    re.search(r"(?:>&|<&)[ \t]*$", prefix)
                    and value is not None
                    and re.fullmatch(r"[0-9]+|-", value.text)
                ):
                    continue
                destinations = (
                    (value,)
                    if value is not None
                    else current.audit_path_values(left, right, command.start)
                    if not child_scope
                    else None
                )
                if destinations is None:
                    if before_use:
                        return False
                elif any(related(item, ancestor=True) for item in destinations):
                    return False
            cursor = 0
            path_count = 0
            while cursor < len(command.words):
                _, left, right = command.words[cursor]
                raw = current.content[left:right]
                if raw in controls:
                    cursor += 1
                    continue
                assignment = re.match(r"([A-Za-z_][A-Za-z0-9_]*)=", raw)
                if assignment is None:
                    break
                if assignment[1] == "PATH":
                    path = current.path_prefix(left, right)
                    if related(path):
                        if (
                            not is_parent
                            or command.start not in caller_starts
                            or path is None
                            or path.text != directory
                        ):
                            return False
                        path_count += 1
                    elif path is None:
                        return False
                elif is_managed_shell_variable(assignment[1]):
                    return False
                elif related(word(left + assignment.end(), right, assignment=True), ancestor=True):
                    # Assignment aliases and environment escapes would permit
                    # additional, unenumerated writes or helper invocations.
                    return False
                cursor += 1
            if is_parent and command.start in caller_starts:
                if path_count != 1:
                    return False
                # These exact programs are separately parsed by the helper
                # argv proof; their generated-path substitutions are refused.
                continue
            if cursor == len(command.words):
                continue
            raw_head, left, right = command.words[cursor]
            head = word(left, right)
            if raw_head in {"[", "[[", "for"} and current.content[left:right] == raw_head:
                head = ShellSourceValue(raw_head)
            if head is None:
                if before_use:
                    return False
                continue
            if related(head, ancestor=True):
                return False
            name = head.text.rpartition("/")[2]
            arguments = command.words[cursor + 1 :]
            values = []
            for _, left, right in arguments:
                assignment = re.match(r"[A-Za-z_][A-Za-z0-9_]*=", current.content[left:right])
                if name == "env" and assignment is not None:
                    variable = assignment[0][:-1]
                    if is_managed_shell_variable(variable):
                        # A later env PATH caller is still a consumer of the
                        # generated helper. It cannot disappear merely because
                        # the selected callers use direct prefix assignments.
                        # Until that invocation is included in the complete
                        # caller set, retain the conservative partial result.
                        return False
                values.append(
                    word(left + assignment.end(), right, assignment=True)
                    if assignment is not None
                    else word(left, right)
                )
            if name == "umask":
                return False
            if name == "chmod" and is_parent and not child_scope:
                if (
                    head.kind == "literal"
                    and head.text in {"chmod", "/bin/chmod", "/usr/bin/chmod"}
                    and name not in current.functions
                    and command.start in resolver.fail_stop_commands
                    and len(values) == 2
                    and values[0] is not None
                    and values[0].kind == "literal"
                    and values[0].text in {"+x", "u+x", "a+x", "700", "755", "0700", "0755"}
                    and values[1] is not None
                    and values[1].kind == "safe_path"
                    and values[1].text == target
                ):
                    chmod_offsets.append(command.start)
                    continue
            if name == "mkdir":
                if all(value is not None for value in values) and all(
                    value.text in {"-p", "--"}
                    or (value.kind == "safe_path" and value.text != target)
                    for value in values
                    if value is not None
                ):
                    continue
            if before_use and name in {
                "eval",
                "exec",
                "source",
                ".",
                "sed",
                "env",
                "command",
                "builtin",
                "sudo",
                "nohup",
                "timeout",
                "nice",
                "bash",
                "sh",
                "dash",
                "zsh",
                "ksh",
            }:
                return False
            for value in values:
                if related(value, ancestor=True):
                    return False
                if before_use and name in writers and value is None:
                    return False
            # A literal child containing an unknown write is checked below;
            # arbitrary opaque external side effects are outside this contract.
        literal_children = tuple(current.literal_children(len(current.content)))
        executable_children = tuple(current.executable_children(len(current.content)))
        if not current.executable_children_complete:
            return False
        for child, consumer_offset in (*literal_children, *executable_children):
            event = consumer_offset if is_parent else anchor
            if event in caller_starts and is_parent:
                continue
            if event > generated.declaration_end and not visit(child, event, depth + 1):
                return False
        return current.audit_spend()

    # Explicit permission-mode changes before creation also affect chmod +x.
    for command in resolver.commands:
        if not resolver.audit_spend():
            return False
        if command.words and command.words[0][0] == "umask":
            return False
    if not visit(resolver, 0, 0):
        return False
    return (
        all(
            any(generated.declaration_end < offset < use for offset in chmod_offsets)
            for use in caller_starts
        )
        and resolver.audit_spend()
    )
