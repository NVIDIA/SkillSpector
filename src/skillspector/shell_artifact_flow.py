# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed effects of known shell file producers, without executing their input.

The caller owns lexical command identity, control-flow coverage and argument
expansion. This layer applies only explicit successful operations to a virtual
file inventory. Unknown output and unsupported argv invalidate their destination.
"""

import hashlib
import re
import shlex
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

from skillspector.shell_artifacts import ShellArtifact, ShellArtifactStore
from skillspector.shell_helper_argv import (
    ShellHelperOutput,
    _Parser,
    _State,
    _Token,
    _UnresolvedError,
    _word,
    is_managed_shell_variable,
    reconstruct_helper_output,
)
from skillspector.shell_sources import (
    ReadWord,
    ShellSourceContext,
    ShellSourceOrigin,
    ShellSourceValue,
)

# Specialized word, mapfile, manifest and tar handlers assume these native
# identities too. A function definition must not silently replace any of them.
_PRODUCER_NATIVE_NAMES = frozenset(
    {
        "printf",
        "cp",
        "tar",
        "mkdir",
        "chmod",
        "cat",
        "p4",
        "local",
        "mapfile",
        "set",
        "command",
        "umask",
        "trap",
        "mktemp",
        "cd",
        "return",
        "exit",
        "echo",
        "dirname",
        "awk",
        "find",
        "sort",
        "xargs",
        "sha256sum",
        "gzip",
        "pigz",
        "bzip2",
        "lbzip2",
        "xz",
        "rm",
    }
)


def helper_stdout_artifacts(
    variants: tuple[tuple[ShellHelperOutput, ...], ...],
    store: ShellArtifactStore,
    origins: tuple[ShellSourceOrigin, ...],
    check_runtime: Callable[[], None],
) -> tuple[ShellArtifact, ...] | None:
    """Retain archive identity only for an unchanged complete stdout stream."""
    output: list[ShellArtifact] = []
    if not variants or len(variants) > 128:
        return None
    total = 0
    for pieces in variants:
        check_runtime()
        if len(pieces) > 128:
            return None
        text: list[str] = []
        provenance = list(origins)
        archive: ShellArtifact | None = None
        for piece in pieces:
            check_runtime()
            if (piece.text is None) == (piece.path is None):
                return None
            value = piece.text
            if piece.path is not None:
                artifact = store.read(piece.path)
                if artifact is None:
                    return None
                if artifact.is_archive:
                    if archive is not None or any(text):
                        return None
                    archive = artifact
                    continue
                if artifact.value is None:
                    return None
                value = artifact.value.text
                provenance.extend(artifact.value.origins)
            assert value is not None
            if archive is not None and value:
                return None
            total += len(value)
            if total > 1_000_000 or len(provenance) > 1024:
                return None
            text.append(value)
        result = archive or ShellArtifact(
            ShellSourceValue("".join(text), "code", tuple(dict.fromkeys(provenance)))
        )
        if result not in output:
            output.append(result)
    return tuple(output)


class ShellArtifactEffects:
    """Apply finite native argv after the caller establishes their identity."""

    def __init__(self, store: ShellArtifactStore, check_runtime: Callable[[], None]) -> None:
        self.store = store
        self.check_runtime = check_runtime
        self.steps = 0

    def _tick(self, argv: Iterable[str]) -> bool:
        self.check_runtime()
        self.steps += 1
        arguments = tuple(argv)
        return self.steps <= 8192 and len(arguments) <= 128 and sum(map(len, arguments)) <= 131_072

    def copy_file(self, argv: tuple[str, ...]) -> bool:
        if not self._tick(argv):
            self.store.invalidate(None)
            return False
        arguments = argv[1:]
        if arguments[:1] == ("--",):
            arguments = arguments[1:]
        if (
            argv[:1] != ("cp",)
            or len(arguments) != 2
            or any(value.startswith("-") for value in arguments)
        ):
            self.store.invalidate(None)
            return False
        source, destination = arguments
        artifact = self.store.read(source)
        if artifact is None:
            self.store.invalidate(destination)
            return False
        previous = self.store.read(destination)
        if previous is not None:
            artifact = replace(artifact, executable=previous.executable)
        elif self.store.exists(destination) is not False:
            artifact = replace(artifact, executable=None)
        return self.store.write(destination, artifact)

    def redirect_stdout(
        self, destination: str, variants: tuple[ShellArtifact, ...], *, append: bool = False
    ) -> bool:
        if not self._tick((destination,)) or len(variants) != 1:
            self.store.invalidate(destination)
            return False
        artifact = variants[0]
        previous = self.store.read(destination)
        executable = (
            previous.executable
            if previous is not None
            else False
            if self.store.exists(destination) is False
            else None
        )
        if append:
            if previous is None or previous.value is None or artifact.value is None:
                self.store.invalidate(destination)
                return False
            artifact = ShellArtifact(
                ShellSourceValue(
                    previous.value.text + artifact.value.text,
                    "code",
                    tuple(dict.fromkeys((*previous.value.origins, *artifact.value.origins))),
                )
            )
        # Redirection changes bytes, not an existing file's permissions. A new
        # regular file starts without execute bits, independently of stdout.
        return self.store.write(destination, replace(artifact, executable=executable))

    def tar(self, argv: tuple[str, ...]) -> bool:
        """Recognize only one native compressor and one explicit file/tree."""
        if not self._tick(argv):
            self.store.invalidate(None)
            return False
        arguments = argv[1:]
        if arguments[:1] == ("-I",):
            if len(arguments) < 2 or arguments[1] not in {"gzip", "pigz", "bzip2", "lbzip2", "xz"}:
                self.store.invalidate(None)
                return False
            arguments = arguments[2:]
        if (
            argv[:1] != ("tar",)
            or len(arguments) not in {4, 5}
            or arguments[0] not in {"-cf", "-xf"}
            or arguments[2] != "-C"
            or any(value.startswith("-") for value in (arguments[1], arguments[3]))
        ):
            self.store.invalidate(None)
            return False
        operation, archive, _, directory, *members = arguments
        if operation == "-xf":
            if members:
                self.store.invalidate(directory)
                return False
            return self.store.extract(archive, directory)
        if (
            len(members) != 1
            or not members[0]
            or members[0].startswith("/")
            or any(part in {"", ".", ".."} for part in members[0].split("/"))
        ):
            self.store.invalidate(archive)
            return False
        # A tar -C DIR NAME stores NAME itself in each member name. Archiving
        # DIR/NAME would silently lose that prefix on later extraction.
        return self.store.archive(directory + "/" + members[0], archive, member_prefix=members[0])


@dataclass(frozen=True, slots=True)
class _ProducerNode:
    kind: str
    words: tuple[_Token, ...] = ()
    body: tuple["_ProducerNode", ...] = ()


class _ProducerParser:
    """Parse finite producer statements, loops, definitions and failure guards."""

    def __init__(self, text: str, read_word: ReadWord, tick: Callable[[], None]) -> None:
        self.tokens = _Parser(text, read_word, tick, output_redirects=True).tokens
        self.position = 0
        self.tick = tick

    def peek(self, offset: int = 0) -> str:
        position = self.position + offset
        return self.tokens[position].raw if position < len(self.tokens) else ""

    def take(self, expected: str | None = None) -> _Token:
        self.tick()
        if not self.peek() or expected is not None and self.peek() != expected:
            raise _UnresolvedError
        token = self.tokens[self.position]
        self.position += 1
        return token

    def sequence(self, stop: str = "", depth: int = 0) -> tuple[_ProducerNode, ...]:
        if depth > 12:
            raise _UnresolvedError
        nodes = []
        while self.peek() and self.peek() != stop:
            if self.peek() in {"\n", ";"}:
                self.take()
                continue
            if self.peek() == "(":
                self.take()
                nodes.append(_ProducerNode("subshell", body=self.sequence(")", depth + 1)))
                self.take(")")
                continue
            if self.peek(1) == "(" and self.peek(2) == ")" and self.peek(3) == "{":
                name = self.take()
                self.take("(")
                self.take(")")
                self.take("{")
                nodes.append(_ProducerNode("function", (name,), self.sequence("}", depth + 1)))
                self.take("}")
                continue
            if self.peek() == "for":
                self.take()
                header = []
                while self.peek() and self.peek() not in {";", "\n"}:
                    header.append(self.take())
                while self.peek() in {";", "\n"}:
                    self.take()
                self.take("do")
                nodes.append(_ProducerNode("for", tuple(header), self.sequence("done", depth + 1)))
                self.take("done")
                continue
            words = []
            nesting = 0
            while self.peek():
                token = self.peek()
                if not nesting and token in {";", "\n", "}", ")", "||"}:
                    if token == "\n" and words and words[-1].raw == "|":
                        self.take()
                        continue
                    break
                if token in {"(", "[["}:
                    nesting += 1
                elif token in {")", "]]"}:
                    nesting -= 1
                    if nesting < 0:
                        raise _UnresolvedError
                if token == "{" or not nesting and token in {"&", "&&"}:
                    raise _UnresolvedError
                words.append(self.take())
            if not words or nesting:
                raise _UnresolvedError
            if self.peek() == "||":
                self.take()
                self.take("{")
                nodes.append(_ProducerNode("guard", tuple(words), self.sequence("}", depth + 1)))
                self.take("}")
            else:
                nodes.append(_ProducerNode("command", tuple(words)))
        if stop and not self.peek():
            raise _UnresolvedError
        return tuple(nodes)


@dataclass(slots=True)
class _ProducerState:
    values: dict[str, str | tuple[str, ...] | None]
    argv: tuple[str, ...]
    cwd: str | None = None
    output: tuple[ShellArtifact, ...] = ()
    stopped: bool = False
    local_names: frozenset[str] = frozenset()

    def copy(self) -> "_ProducerState":
        return _ProducerState(
            self.values.copy(), self.argv, self.cwd, self.output, self.stopped, self.local_names
        )


@dataclass(frozen=True, slots=True)
class ShellProducerResult:
    """Known final outputs and the complete set of explicit write locations."""

    outputs: tuple[tuple[str, ShellArtifact], ...]
    writes: tuple[str, ...]


class ShellProducer:
    """Interpret a bounded file-producing subset against an in-memory store.

    Commands outside this grammar, unknown destinations, unsupported functions
    and unknown native identities refuse the whole producer proof. A helper
    callback supplies already-proved stdout variants and its separately audited
    effects; it must not invoke a process or read a host file.
    """

    def __init__(
        self,
        store: ShellArtifactStore,
        context: ShellSourceContext,
        read_word: ReadWord,
        check_runtime: Callable[[], None],
        helper_output: Callable[
            [tuple[str, ...], dict[str, str | None]], tuple[ShellArtifact, ...] | None
        ]
        | None,
    ) -> None:
        self.store = store
        self.context = context
        self.read_word = read_word
        self.check_runtime = check_runtime
        self.helper_output = helper_output
        self.effects = ShellArtifactEffects(store, check_runtime)
        self.functions: dict[str, tuple[_ProducerNode, ...]] = {}
        self.writes: set[str] = set()
        self.cleanup: str | None = None
        self.steps = 0
        self.depth = 0
        self.capture_depth = 0
        self.subshell_depth = 0
        self.temporary = 0
        self.exported: frozenset[str] = frozenset()

    def tick(self) -> None:
        self.check_runtime()
        self.steps += 1
        if self.steps > 16384:
            raise _UnresolvedError

    def parse(self, text: str) -> tuple[_ProducerNode, ...]:
        if len(text) > 131072 or "\x00" in text:
            raise _UnresolvedError
        parser = _ProducerParser(text, self.read_word, self.tick)
        program = parser.sequence()
        if parser.peek():
            raise _UnresolvedError
        return program

    def word(self, raw: str, state: _ProducerState) -> str:
        self.tick()
        length = re.fullmatch(r'("?)\$\{#([A-Za-z_][A-Za-z0-9_]*)\[@\]\}\1', raw)
        if length:
            value = state.values.get(length[2])
            if not isinstance(value, tuple):
                raise _UnresolvedError
            return str(len(value))
        if raw == "$#":
            return str(len(state.argv))
        array = re.fullmatch(r'"\$\{([A-Za-z_][A-Za-z0-9_]*)\[(0|[1-9][0-9]{0,2})\]\}"', raw)
        if array:
            value = state.values.get(array[1])
            if not isinstance(value, tuple):
                raise _UnresolvedError
            if int(array[2]) < len(value):
                return value[int(array[2])]
            raise _UnresolvedError
        directory = re.fullmatch(
            r'"\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)(/[-A-Za-z0-9_./]+)"', raw
        )
        if directory:
            suffix = directory[1]
            if ".." in suffix.split("/"):
                raise _UnresolvedError
            return (self.context.path.rpartition("/")[0] or ".") + suffix
        values = {
            name: value if isinstance(value, str) else None for name, value in state.values.items()
        }
        value = _word(raw, _State(values, [], state.argv, nounset=True), self.tick)
        if not isinstance(value, str):
            raise _UnresolvedError
        return value

    def arguments(self, words: tuple[_Token, ...], state: _ProducerState) -> tuple[str, ...]:
        output = []
        for token in words:
            array = re.fullmatch(r'"\$\{([A-Za-z_][A-Za-z0-9_]*)\[@\]\}"', token.raw)
            if array:
                value = state.values.get(array[1])
                if not isinstance(value, tuple):
                    raise _UnresolvedError
                output.extend(value)
            else:
                output.append(self.word(token.raw, state))
        if len(output) > 128 or sum(map(len, output)) > 131072:
            raise _UnresolvedError
        return tuple(output)

    def assignment(self, name: str, raw: str, state: _ProducerState) -> list[_ProducerState]:
        if is_managed_shell_variable(name) or self.depth and name not in state.local_names:
            raise _UnresolvedError
        substitution = re.fullmatch(r'"?\$\((.*)\)"?', raw, re.DOTALL)
        if not substitution:
            state.values[name] = self.word(raw, state)
            return [state]
        self.capture_depth += 1
        try:
            captured = self.run(self.parse(substitution[1]), [state.copy()], capture=True)
        finally:
            self.capture_depth -= 1
        output = []
        for branch in captured:
            if branch.stopped:
                continue
            if any(item.value is None for item in branch.output):
                raise _UnresolvedError
            value = "".join(item.value.text for item in branch.output).rstrip("\n")
            if len(value) > 65536:
                raise _UnresolvedError
            child = state.copy()
            child.values[name] = value
            output.append(child)
        return output

    def run(
        self,
        nodes: tuple[_ProducerNode, ...],
        states: list[_ProducerState],
        *,
        capture: bool = False,
    ) -> list[_ProducerState]:
        if capture:
            for state in states:
                state.output = ()
        for node in nodes:
            output = []
            for state in states:
                self.tick()
                output.extend([state] if state.stopped else self.node(node, state))
            if len(output) > 128 or self.depth == 0 and len(output) > 1:
                raise _UnresolvedError
            states = output
        return states

    def node(self, node: _ProducerNode, state: _ProducerState) -> list[_ProducerState]:
        if node.kind == "function":
            name = node.words[0].raw
            if (
                self.depth
                or self.capture_depth
                or self.subshell_depth
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
                or name in self.functions
                or name in _PRODUCER_NATIVE_NAMES
            ):
                raise _UnresolvedError
            self.functions[name] = node.body
            return [state]
        if node.kind == "subshell":
            self.subshell_depth += 1
            try:
                result = self.run(node.body, [state.copy()])
            finally:
                self.subshell_depth -= 1
            if len(result) != 1 or result[0].stopped:
                raise _UnresolvedError
            # The child shares stdout, while its variable and cwd changes stay
            # local. Its copied output already includes the preceding prefix.
            state.output = result[0].output
            return [state]
        if node.kind == "for":
            if (
                len(node.words) < 3
                or node.words[1].raw != "in"
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", node.words[0].raw)
            ):
                raise _UnresolvedError
            name = node.words[0].raw
            if is_managed_shell_variable(name) or self.depth and name not in state.local_names:
                raise _UnresolvedError
            values = self.arguments(node.words[2:], state)
            if len(values) > 32:
                raise _UnresolvedError
            states = [state]
            for value in values:
                for branch in states:
                    branch.values[name] = value
                states = self.run(node.body, states)
            return states
        if node.kind == "guard":
            # Failure branches may report/terminate only. Their status cannot
            # hide writes, definitions or a successful continuation.
            if not node.body or node.body[-1].kind != "command":
                raise _UnresolvedError
            terminal = node.body[-1].words
            if (
                len(terminal) != 2
                or terminal[0].raw not in {"exit", "return"}
                or terminal[1].raw not in {"1", "2"}
            ):
                raise _UnresolvedError
            for item in node.body[:-1]:
                if item.kind != "command" or not item.words or item.words[0].raw != "echo":
                    raise _UnresolvedError
                self.command(item.words, state.copy(), diagnostics=True)
            if not self.predicate(node.words, state):
                state.stopped = True
            return [state]
        return self.command(node.words, state)

    def predicate(self, words: tuple[_Token, ...], state: _ProducerState) -> bool:
        raw = tuple(token.raw for token in words)
        if raw[:1] != ("[[",) or raw[-1:] != ("]]",):
            raise _UnresolvedError
        terms = []
        current = []
        for token in words[1:-1]:
            if token.raw == "&&":
                terms.append(tuple(current))
                current = []
            else:
                current.append(token)
        terms.append(tuple(current))
        result = True
        for term in terms:
            if len(term) == 3 and term[0].raw == "!" and term[1].raw == "-e":
                exists = self.store.exists(self.word(term[2].raw, state))
                if exists is None:
                    raise _UnresolvedError
                result &= not exists
            elif len(term) == 2 and term[0].raw == "-s":
                artifact = self.store.read(self.word(term[1].raw, state))
                result &= (
                    artifact is not None
                    and artifact.value is not None
                    and bool(artifact.value.text)
                )
            elif len(term) == 3 and term[1].raw == "-eq":
                left, right = self.word(term[0].raw, state), self.word(term[2].raw, state)
                if not re.fullmatch(r"0|[1-9][0-9]{0,8}", left) or not re.fullmatch(
                    r"0|[1-9][0-9]{0,8}", right
                ):
                    raise _UnresolvedError
                result &= int(left) == int(right)
            elif (
                len(term) == 3
                and term[1].raw == "=="
                and re.fullmatch(r"\*\.[A-Za-z0-9]+", term[2].raw)
            ):
                result &= self.word(term[0].raw, state).endswith(term[2].raw[1:])
            else:
                raise _UnresolvedError
        return result

    def command(
        self, words: tuple[_Token, ...], state: _ProducerState, *, diagnostics: bool = False
    ) -> list[_ProducerState]:
        self.tick()
        raw = tuple(token.raw for token in words)
        if not words:
            raise _UnresolvedError
        if diagnostics:
            if raw[0] != "echo" or raw[-2:] != (">&", "2"):
                raise _UnresolvedError
            for token in words[1:-2]:
                inert = re.sub(
                    r"\$(?:[0-9]|[A-Za-z_][A-Za-z0-9_]*|\{#?[A-Za-z_][A-Za-z0-9_]*(?:\[@\])?\})",
                    "x",
                    token.raw,
                )
                if not isinstance(_word(inert, _State({}, [], ()), self.tick), str):
                    raise _UnresolvedError
            return [state]
        array = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(\+?)=", raw[0])
        if array and raw[1:2] == ("(",) and raw[-1:] == (")",):
            if (
                is_managed_shell_variable(array[1])
                or self.depth
                and array[1] not in state.local_names
            ):
                raise _UnresolvedError
            value = self.arguments(words[2:-1], state)
            if array[2]:
                previous = state.values.get(array[1])
                if not isinstance(previous, tuple):
                    raise _UnresolvedError
                value = previous + value
            if len(value) > 128:
                raise _UnresolvedError
            state.values[array[1]] = value
            return [state]
        assignment = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", raw[0], re.DOTALL)
        if assignment and len(words) == 1:
            return self.assignment(assignment[1], assignment[2], state)
        if raw[0] == "local":
            if not self.depth:
                raise _UnresolvedError
            if (
                len(words) == 5
                and raw[1] == "-a"
                and raw[3:] == ("(", ")")
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=", raw[2])
            ):
                if is_managed_shell_variable(raw[2][:-1]):
                    raise _UnresolvedError
                state.values[raw[2][:-1]] = ()
                state.local_names |= {raw[2][:-1]}
                return [state]
            for token in words[1:]:
                assignment = re.fullmatch(
                    r"([A-Za-z_][A-Za-z0-9_]*)(?:=(.*))?", token.raw, re.DOTALL
                )
                if not assignment or is_managed_shell_variable(assignment[1]):
                    raise _UnresolvedError
                state.values[assignment[1]] = (
                    None if assignment[2] is None else self.word(assignment[2], state)
                )
                state.local_names |= {assignment[1]}
            return [state]
        if raw[0] == "mapfile":
            return self.mapfile(words, state)
        if "|" in raw:
            if self.depth:
                raise _UnresolvedError
            return self.manifest(words, state)
        command_words = []
        redirects = []
        position = 0
        while position < len(words):
            token = words[position]
            if token.raw in {">", ">>", ">&"}:
                descriptor = "1"
                if (
                    command_words
                    and command_words[-1].raw.isdecimal()
                    and command_words[-1].end == token.start
                ):
                    descriptor = command_words.pop().raw
                if position + 1 >= len(words):
                    raise _UnresolvedError
                redirects.append(
                    (descriptor + token.raw, self.word(words[position + 1].raw, state))
                )
                position += 2
            else:
                if token.raw in {"<", "<<", "(", ")", "[[", "]]"}:
                    raise _UnresolvedError
                command_words.append(token)
                position += 1
        argv = self.arguments(tuple(command_words), state)
        if not argv:
            raise _UnresolvedError
        name, *arguments = argv
        destination = None
        append = False
        discard_stdout = False
        stdout_redirects = 0
        for redirect, target in redirects:
            if redirect == "2>&" and target == "1":
                # Helper summaries prove stdout only. Merged stderr can add
                # executable bytes whose contents and ordering are unknown.
                raise _UnresolvedError
            if redirect == "1>&" and target == "2" and diagnostics:
                continue
            if redirect in {"1>", "1>>"}:
                stdout_redirects += 1
                if stdout_redirects > 1:
                    raise _UnresolvedError
            if target == "/dev/null" and redirect in {"1>", "1>>", "2>", "2>>"}:
                discard_stdout |= redirect.startswith("1")
                continue
            if redirect not in {"1>", "1>>"} or destination is not None:
                raise _UnresolvedError
            destination = self.path(target, state)
            append = redirect == "1>>"
        if self.depth and (
            destination is not None
            or name not in {*self.functions, "p4", "printf", "return", "exit"}
        ):
            # Capture functions may branch over helper output. They are pure
            # apart from separately audited helper logs; file effects cannot
            # leak from a failed alternative into a surviving state.
            raise _UnresolvedError
        if name in self.functions:
            if self.depth or not self.capture_depth or destination is not None:
                raise _UnresolvedError
            self.depth += 1
            child = _ProducerState(state.values.copy(), tuple(arguments), state.cwd)
            branches = self.run(self.functions[name], [child])
            self.depth -= 1
            output = []
            for branch in branches:
                if branch.stopped:
                    continue
                parent = state.copy()
                if not discard_stdout:
                    parent.output += branch.output
                output.append(parent)
            return output
        produced = None
        if (
            name == "set"
            and tuple(arguments) in {("-euo", "pipefail"), ("-Eeuo", "pipefail")}
            and not redirects
        ):
            pass
        elif name == "umask" and arguments == ["077"] and not redirects:
            pass
        elif name == "trap" and len(arguments) == 2 and arguments[1] == "EXIT" and not redirects:
            if self.capture_depth or self.subshell_depth:
                raise _UnresolvedError
            cleanup = re.fullmatch(r'rm -rf "\$([A-Za-z_][A-Za-z0-9_]*)"', arguments[0])
            value = state.values.get(cleanup[1]) if cleanup else None
            if not isinstance(value, str) or self.store.inventory(value) is None:
                raise _UnresolvedError
            self.writes.add(value)
            self.cleanup = cleanup[1]
        elif (
            name == "command"
            and len(arguments) == 2
            and arguments[0] == "-v"
            and arguments[1] in {"p4", "pigz", "lbzip2"}
        ):
            if arguments[1] == "p4" and self.helper_output is None:
                raise _UnresolvedError
        elif name == "mktemp" and len(arguments) == 2 and arguments[0] == "-d" and not redirects:
            if re.fullmatch(r"/tmp/[-A-Za-z0-9_.]+X{3,}", arguments[1]) is None:
                raise _UnresolvedError
            self.temporary += 1
            value = re.sub(r"X+$", f"__skillspector_producer_{self.temporary}__", arguments[1])
            if not self.store.create_empty_directory(value):
                raise _UnresolvedError
            self.writes.add(value)
            produced = (ShellArtifact(ShellSourceValue(value + "\n")),)
        elif name == "mkdir" and arguments[:1] == ["-p"] and len(arguments) >= 2 and not redirects:
            for argument in arguments[1:]:
                path = self.path(argument, state)
                if self.store.inventory(path) is None:
                    raise _UnresolvedError
                if self.store.exists(path) is False and not self.store.create_empty_directory(path):
                    raise _UnresolvedError
                self.writes.add(path)
        elif name == "cp" and not redirects:
            resolved = (
                name,
                *(
                    "--"
                    if index == 0 and argument == "--"
                    else self.path(argument, state)
                    for index, argument in enumerate(arguments)
                ),
            )
            if not self.effects.copy_file(resolved):
                raise _UnresolvedError
            self.writes.add(resolved[-1])
        elif name == "tar" and not redirects:
            if not self.effects.tar(argv):
                raise _UnresolvedError
            mode = argv.index("-cf") if "-cf" in argv else argv.index("-xf")
            self.writes.add(argv[mode + 1] if argv[mode] == "-cf" else argv[argv.index("-C") + 1])
        elif (
            name == "chmod"
            and len(arguments) == 2
            and arguments[0] in {"600", "700", "+x"}
            and not redirects
        ):
            path = self.path(arguments[1], state)
            artifact = self.store.read(path)
            if artifact is None or not self.store.write(
                path, replace(artifact, executable=arguments[0] != "600")
            ):
                raise _UnresolvedError
            self.writes.add(path)
        elif name == "cd" and len(arguments) == 1 and not redirects:
            path = self.path(arguments[0], state)
            if self.store.inventory(path) is None:
                raise _UnresolvedError
            state.cwd = path
        elif name == "printf":
            text = " ".join(shlex.quote(argument) for argument in argv)
            outputs = reconstruct_helper_output(
                text, (), {}, frozenset(), self.read_word, self.check_runtime
            )
            produced = (
                helper_stdout_artifacts(outputs, self.store, (), self.check_runtime)
                if outputs is not None
                else None
            )
            if produced is None:
                raise _UnresolvedError
        elif name == "p4" and self.helper_output is not None:
            # Inherited names keep their export attribute after assignment,
            # including inside the bounded capture-function scope. Ordinary
            # locals are not implicitly exported to a helper process.
            environment = {
                name: value if isinstance(value := state.values.get(name), str) else None
                for name in self.exported
            }
            produced = self.helper_output(argv, environment)
            if not produced:
                raise _UnresolvedError
        elif (
            name in {"return", "exit"}
            and len(arguments) == 1
            and arguments[0] in {"1", "2"}
            and not redirects
        ):
            state.stopped = True
        else:
            raise _UnresolvedError
        if produced is None:
            if destination is not None:
                raise _UnresolvedError
            return [state]
        if destination is not None:
            if not self.effects.redirect_stdout(destination, produced, append=append):
                raise _UnresolvedError
            self.writes.add(destination)
            return [state]
        if discard_stdout:
            return [state]
        output = []
        for artifact in produced:
            branch = state.copy()
            branch.output += (artifact,)
            output.append(branch)
        return output

    def path(self, value: str, state: _ProducerState) -> str:
        if value.startswith("/"):
            return value
        if state.cwd is None or any(part in {"", ".", ".."} for part in value.split("/")):
            raise _UnresolvedError
        return value if state.cwd == "." else state.cwd + "/" + value

    def mapfile(self, words: tuple[_Token, ...], state: _ProducerState) -> list[_ProducerState]:
        raw = tuple(token.raw for token in words)
        if (
            len(raw) < 10
            or raw[:2] != ("mapfile", "-t")
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw[2])
            or raw[3:6] != ("<", "<", "(")
            or raw[-1] != ")"
        ):
            raise _UnresolvedError
        name = raw[2]
        if is_managed_shell_variable(name) or self.depth and name not in state.local_names:
            raise _UnresolvedError
        inner = words[6:-1]
        if (
            len(inner) == 5
            and tuple(token.raw for token in inner[2:4]) == ("<<", "<")
            and self.arguments(inner[:2], state) == ("awk", "NF {print $1}")
        ):
            value = self.word(inner[4].raw, state)
            # This exact AWK program uses LF records and ASCII blank fields.
            # Locale-sensitive characters/control bytes need their own proof;
            # Python's Unicode whitespace rules would silently change them.
            if any(character not in "\t\n" and not " " <= character <= "~" for character in value):
                raise _UnresolvedError
            rows = []
            for line in value.split("\n"):
                self.tick()
                line = line.strip(" \t")
                if line:
                    rows.append(re.split(r"[ \t]+", line, maxsplit=1)[0])
            state.values[name] = tuple(rows)
        else:
            argv = self.arguments(inner, state)
            if (
                len(argv) != 7
                or argv[0] != "find"
                or argv[2:5] != ("-type", "f", "-name")
                or argv[-1] != "-print"
                or re.fullmatch(r"[-A-Za-z0-9_.]+", argv[5]) is None
            ):
                raise _UnresolvedError
            inventory = self.store.inventory(argv[1])
            if inventory is None:
                raise _UnresolvedError
            state.values[name] = tuple(
                argv[1] + "/" + path for path, _ in inventory if path.rpartition("/")[2] == argv[5]
            )
        if len(state.values[name]) > 128:
            raise _UnresolvedError
        return [state]

    def manifest(self, words: tuple[_Token, ...], state: _ProducerState) -> list[_ProducerState]:
        argv = self.arguments(words, state)
        if argv != (
            "find",
            ".",
            "-type",
            "f",
            "!",
            "-name",
            "MANIFEST.sha256",
            "-print0",
            "|",
            "sort",
            "-z",
            "|",
            "xargs",
            "-0",
            "sha256sum",
            ">",
            "MANIFEST.sha256",
        ):
            raise _UnresolvedError
        inventory = self.store.inventory(state.cwd) if state.cwd is not None else None
        if inventory is None:
            raise _UnresolvedError
        rows = []
        for path, artifact in inventory:
            self.tick()
            if path == "MANIFEST.sha256":
                continue
            if artifact.value is None or any(character in path for character in "\\\r\n\x00"):
                raise _UnresolvedError
            rows.append(
                hashlib.sha256(artifact.value.text.encode()).hexdigest() + "  ./" + path + "\n"
            )
        destination = state.cwd + "/MANIFEST.sha256"
        # sort's order depends on the runtime locale. Keep the finite file
        # inventory and byte bound, but never publish invented manifest bytes
        # as executable source or as a subsequent hash/cat input.
        if not self.store.write(
            destination, ShellArtifact(opaque_size=len("".join(rows).encode()))
        ):
            raise _UnresolvedError
        self.writes.add(destination)
        return [state]

    def inspect(
        self,
        argv: tuple[str, ...],
        environment: dict[str, str],
        outputs: tuple[str, ...],
        *,
        initial_cwd: str | None = None,
    ) -> ShellProducerResult | None:
        """Return a complete successful effect proof, or forget every output."""
        try:
            if (
                len(argv) > 128
                or len(environment) > 128
                or not outputs
                or len(outputs) > 128
                or any(is_managed_shell_variable(name) for name in environment)
                or any(
                    "__skillspector_producer_" in source for source in self.context.sources.values()
                )
            ):
                raise _UnresolvedError
            source = self.context.sources.get(self.context.path)
            if source is None:
                raise _UnresolvedError
            self.exported = frozenset(environment)
            program = self.parse(source)
            if (
                not program
                or program[0].kind != "command"
                or tuple(token.raw for token in program[0].words)
                not in {("set", "-euo", "pipefail"), ("set", "-Eeuo", "pipefail")}
            ):
                raise _UnresolvedError
            states = self.run(program, [_ProducerState(environment.copy(), argv, initial_cwd)])
            successful = [state for state in states if not state.stopped]
            if not successful:
                raise _UnresolvedError
            if self.cleanup is not None:
                for state in successful:
                    # The quoted trap expands its variable when the shell
                    # exits, after any later assignment or replacement trap.
                    path = state.values.get(self.cleanup)
                    if not isinstance(path, str) or self.store.inventory(path) is None:
                        raise _UnresolvedError
                    self.writes.add(path)
                    self.store.invalidate(path)
            result = []
            for path in outputs:
                value = self.store.read(path)
                if value is None:
                    raise _UnresolvedError
                result.append((path, value))
            return ShellProducerResult(tuple(result), tuple(sorted(self.writes)))
        except _UnresolvedError:
            self.store.invalidate(None)
            return None
