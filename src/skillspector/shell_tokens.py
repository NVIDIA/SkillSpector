# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded lexical ownership of shell string closing delimiters.

No shell program is executed. Unsupported words and heredocs prove no quote
ownership; literal contents stay available to every security-text check.
"""

from collections.abc import Callable


class ShellStringClosers:
    """Lazily identify closing quotes using the scanner's shell word grammar."""

    def __init__(self, content, read_word, heredoc_header, heredoc_bodies, check_runtime):
        self.content = content
        self.read_word = read_word
        self.heredoc_header = heredoc_header
        self.heredoc_bodies = heredoc_bodies
        self.check_runtime: Callable[[], None] = check_runtime
        self.closers: frozenset[int] | None = None

    def closes_string(self, offset: int) -> bool:
        if self.closers is None:
            self.closers = self._collect()
        self.check_runtime()
        return offset in self.closers

    def _word_closers(self, start: int, end: int) -> set[int]:
        content = self.content
        quote = None
        ansi_c = False
        result: set[int] = set()
        cursor = start
        while cursor < end:
            if cursor % 1024 == 0:
                self.check_runtime()
            character = content[cursor]
            if quote == "'" and not ansi_c:
                if character == quote:
                    result.add(cursor)
                    quote = None
            elif character == "\\":
                if (
                    quote is None
                    or ansi_c
                    or content[cursor + 1 : cursor + 2] in {"$", "`", '"', "\\", "\n"}
                ):
                    cursor += 2
                    continue
            elif character == "`" or content.startswith(("$(", "${", "$["), cursor):
                # Nested expansion grammar owns its own quotes. This small
                # proof declines the entire word instead of guessing them.
                return set()
            elif quote is None and content.startswith("$'", cursor):
                quote, ansi_c = "'", True
                cursor += 2
                continue
            elif character in "'\"" and (quote is None or quote == character):
                if quote is None:
                    quote = character
                else:
                    result.add(cursor)
                    quote, ansi_c = None, False
            cursor += 1
        return result if quote is None else set()

    def _collect(self) -> frozenset[int]:
        content = self.content
        if len(content) > 1_000_000 or "\x00" in content:
            return frozenset()
        result: set[int] = set()
        pending = []
        cursor = words = 0
        while cursor < len(content):
            self.check_runtime()
            character = content[cursor]
            if character == "\n":
                cursor += 1
                if pending:
                    stop = self.heredoc_bodies(
                        content, cursor, len(content), tuple(pending), self.check_runtime
                    )
                    if stop is None:
                        return frozenset()
                    cursor = stop
                    pending.clear()
                continue
            if character in " \t\r;|&(){}":
                cursor += 1
                continue
            if character == "#":
                end = content.find("\n", cursor)
                cursor = len(content) if end < 0 else end
                continue
            if content.startswith("<<", cursor) and not content.startswith("<<<", cursor):
                header = self.heredoc_header(content, cursor, len(content))
                if header is None or len(pending) >= 32:
                    return frozenset()
                item, cursor = header
                pending.append(item)
                continue
            if character in "<>":
                cursor += 3 if content.startswith("<<<", cursor) else 1
                continue
            _, end, limited = self.read_word(content, cursor, self.check_runtime)
            words += 1
            if limited or end <= cursor or words > 100_000:
                return frozenset()
            result.update(self._word_closers(cursor, end))
            cursor = end
        return frozenset() if pending else frozenset(result)
