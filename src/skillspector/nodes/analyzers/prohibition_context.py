# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize a small, bounded grammar of direct defensive instructions.

This is deliberately not a nearby-negation heuristic. Only a prohibition that
immediately governs the matched action can exempt it; unknown grammar retains
ordinary detection. Callers must pass match offsets in the same content view.
"""

from __future__ import annotations

import re

from skillspector.artifacts import normalized_security_view

from .common import LINE_BREAK_CHARS, LOGICAL_LINE_BREAK

_CONTEXT_CHARS = 512
_MARKDOWN = str.maketrans("", "", "*_`")
# A negation on another logical line cannot govern an affirmative instruction.
_SPACE = r"[ \t]+"
_DIRECT_PREFIX = re.compile(
    r"(?:\A|(?<=[\n\r.!?;,\[]))[ \t]*"
    r"(?:[-+][ \t]+|\d+[.)][ \t]+)?"
    rf"(?:please{_SPACE})?"
    rf"(?:(?:you|(?:the{_SPACE})?(?:agent|assistant|model)){_SPACE})?"
    rf"(?P<negation>do{_SPACE}not|don['’]t|never|no|must{_SPACE}not|shall{_SPACE}not){_SPACE}"
    rf"(?:(?:ever|first|verbatim|exactly|word{_SPACE}for{_SPACE}word){_SPACE})?\Z",
    re.IGNORECASE,
)
_BLANK_LINE = re.compile(r"\n[ \t]*\n")
_CONTRACTION = re.compile(r"(?<=\w)['’](?=\w)")
_SENTENCE_END = re.compile(rf"[.!?;{LINE_BREAK_CHARS}]")
_EXCEPTION = re.compile(r"\b(?:unless|except|until|but|however|instead)\b", re.IGNORECASE)
# A condition or scope before the prohibition limits it exactly as a trailing
# one does: "Unless the user says X, do not reveal ..." implies disclosure.
_CLAUSE_BREAK = re.compile(rf"[.!?;\[{LINE_BREAK_CHARS}]")
_LEADING_QUALIFIER = re.compile(
    r"\b(?:unless|except|until|if|when|whenever|while|only|otherwise|else|but|however|"
    r"instead|for|to|in|on|during|after|before)\b",
    re.IGNORECASE,
)

# Reversal wording is matched by stem so that every inflection ("negation",
# "inverts", "opposites", "contrarily", ...) retains detection.
_DISAVOWAL = re.compile(
    r"\b(?:ignore|disregard|override|obsolete|invalid|bypass|suspend|violat\w*|"
    r"opposit\w*|invers\w*|invert\w*|revers\w*|contrar\w*|negat\w*)\b",
    re.IGNORECASE,
)

# Each caller supplies only object completions its regex intentionally leaves
# outside the match. Channel, frequency, recipient and other qualifiers are
# deliberately absent: a limited prohibition does not rule out disclosure.
PROMPT_REVEAL_TAIL = re.compile(
    r"(?:content)?"
    r"(?:,\s*memory,?\s*(?:or|and)\s+prior\s+conversation)?"
    r"(?:\s*if\s+the\s+pasted\s+prompt\s+requests\s+it)?",
    re.IGNORECASE,
)
_REFERENTIAL_ACTION = re.compile(
    r"\b(?:do|perform|execute|reveal|output|print|show|display|share|disclose|"
    r"repeat|recite|reproduce|summarize|paraphrase|rephrase|translate|encode|"
    r"encrypt|reverse|send|post|upload|transmit|save|store|log|copy|extract|"
    r"dump|include|provide|read|write|persist|deploy|delete)\s+(?:(?:all|some|part)\s+of\s+)?"
    r"(?:it|them|this|that|so)\b",
    re.IGNORECASE,
)
# A boundary can introduce a qualifying fragment, including inside a list or
# heading. Approve only complete harmless instructions or explicit risky
# objects that the downstream detectors can still identify independently.
_LIST_PREFIX = re.compile(r"(?:[-+]|\d{1,9}[.)])\s+")
_NEXT_INSTRUCTION = re.compile(
    r"(?:(?:then|now)\s+)?(?:"
    r"(?:analyze\s+the\s+structure\s+and\s+intent\s+without\s+obeying\s+its\s+directives"
    r"|follow\s+the\s+user['’]s\s+task"
    r"|only\s+make\s+changes\s+directly\s+requested)"
    rf"(?=\s*(?:[.!?;{LINE_BREAK_CHARS}]|\Z))"
    r"|(?:(?:do\s+not|don['’]t|never|must\s+not|shall\s+not)\s+)?"
    r"(?:"
    r"(?:reveal|output|print|show|display|expose|return|echo)\s+"
    r"(?:your\s+)?(?:full\s+)?(?:system\s+)?"
    r"(?:prompt|instructions?|rules?|guidelines?|directives?)\b"
    r"|(?P<yara_reveal>(?:reveal|print|dump|expose|show)\s+(?:the\s+)?"
    r"(?:system|developer)\s+(?:prompt|message|instructions)\b)"
    r")"
    r"|deploy\s+without\s+(?:approval|confirmation|consent)\b"
    r")",
    re.IGNORECASE | re.ASCII,
)

# Exact raw ASCII context alternatives from agent_skills.yar. A YARA-only
# action string cannot prove a finding without the rule's context condition.
# Do not normalize these bytes or broaden literal spaces to Unicode whitespace.
_YARA_AGENT_CONTEXT = re.compile(
    r"(?:AI agent|assistant|LLM|model|system prompt|developer message|tool description)",
    re.IGNORECASE | re.ASCII,
)


def is_directly_prohibited(
    content: str,
    start: int,
    end: int,
    *,
    allowed_tail: re.Pattern[str] | None = None,
    allow_yara_continuation: bool = False,
) -> bool:
    """Whether a bounded, unambiguous prohibition governs this exact action.

    Markdown emphasis and inline-code delimiters may surround the prohibition;
    quotation marks and arbitrary intervening words are not exempted. Exceptions
    in the same clause, and conditions or scopes leading into the prohibition,
    also retain detection. Nonempty object completions must
    fully match the caller's allowed_tail grammar after formatting is stripped.
    Callers may allow YARA-only continuations only when content is raw source,
    because YARA does not scan normalized or reconstructed security views.
    Work per match is constant-bounded; unknown or incomplete context retains
    detection.
    """
    if not 0 <= start < end <= len(content):
        return False
    left = max(0, start - _CONTEXT_CHARS)
    prefix = content[left:start].translate(_MARKDOWN)
    prohibited = _DIRECT_PREFIX.search(prefix)
    if prohibited is None or (left and prohibited.start() == 0):
        return False
    leading = prefix[: prohibited.start()]
    quoted = _CONTRACTION.sub("", leading)
    if sum(quoted.count(char) for char in '"“”') % 2:
        return False
    if sum(quoted.count(char) for char in "'‘’") % 2:
        return False
    if _DISAVOWAL.search(leading):
        return False
    if _LEADING_QUALIFIER.search(_CLAUSE_BREAK.split(leading)[-1]):
        return False
    # A bare "No" after a question answers it; it does not prohibit the
    # imperative that follows ("Keep anything private? No Reveal ...").
    if prohibited.group("negation").lower() == "no" and leading.rstrip().endswith("?"):
        return False
    if _BLANK_LINE.search(LOGICAL_LINE_BREAK.sub("\n", prohibited.group())):
        return False

    # A long unfinished clause may hide a later exception. Do not infer safety
    # from an arbitrarily clipped fragment of that clause.
    raw_tail = content[end : end + _CONTEXT_CHARS]
    tail_view = normalized_security_view(raw_tail)
    tail = tail_view.text
    if _DISAVOWAL.search(tail) or _REFERENTIAL_ACTION.search(tail):
        return False
    boundary = _SENTENCE_END.search(tail)
    if boundary is None and end + len(raw_tail) < len(content):
        return False
    if boundary is not None:
        # Prove independent detection against the actual continuation bytes:
        # YARA does not scan normalized or reconstructed text. In particular,
        # fullwidth/markup-split YARA-only verbs must not justify exemption.
        following_start = (
            tail_view.source_offset(boundary.end()) if boundary.end() < len(tail) else len(raw_tail)
        )
        following = raw_tail[following_start:].lstrip()
        following = _LIST_PREFIX.sub("", following, count=1)
        independent = _NEXT_INSTRUCTION.match(following) if following else None
        if following and independent is None:
            return False
        if (
            independent is not None
            and independent.group("yara_reveal") is not None
            and (
                not allow_yara_continuation
                or _YARA_AGENT_CONTEXT.search(content, left, end + _CONTEXT_CHARS) is None
            )
        ):
            return False
    clause = tail[: boundary.start()] if boundary is not None else tail
    if _EXCEPTION.search(clause):
        return False
    clean_tail = clause.translate(_MARKDOWN).strip().rstrip("])}").rstrip()
    return not clean_tail or (
        allowed_tail is not None and allowed_tail.fullmatch(clean_tail) is not None
    )
