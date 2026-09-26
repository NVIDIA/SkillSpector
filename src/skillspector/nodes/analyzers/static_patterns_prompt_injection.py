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

"""Static patterns: prompt injection (P1–P4, P9). Node and analyze() in one module."""

from __future__ import annotations

import fnmatch
import re
import sys
from collections.abc import Callable, Iterator

from skillspector.artifacts import _is_emoji_base, prompt_injection_letter_spacing_view
from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import LOGICAL_LINE_BREAK, SourceLocationIndex, get_context
from .pattern_defaults import PatternCategory
from .whitespace_padding import (
    VERTICAL_HIGH_SEVERITY_LINES,
    ZERO_WIDTH_CHARS,
    detect_whitespace_padding,
    padding_run_match_fingerprint,
)

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_prompt_injection"
USES_RUNTIME_CHECK = True

# Generated/vendored filename globs for which the P9 whitespace-padding signal is
# skipped (these legitimately carry large whitespace runs). Applies ONLY to P9.
_P9_SKIP_GLOBS = (
    "*.min.js",
    "*.min.css",
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "*.svg",
    "*.map",
)


def _is_p9_skipped_path(file_path: str) -> bool:
    """Return True when *file_path* is a generated/vendored file P9 should skip."""
    name = file_path.replace("\\", "/").rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(name, pattern) for pattern in _P9_SKIP_GLOBS)


# P1: Instruction Override
P1_PATTERNS = [
    (r"ignore\s+(?:all\s+)?previous\s+instructions?", 0.8),
    (r"ignore\s+(?:all\s+)?(?:safety|security)\s+(?:rules?|constraints?|guidelines?)", 0.9),
    (r"override\s+(?:safety|security|system)", 0.9),
    (r"bypass\s+(?:safety|security|restrictions?|constraints?)", 0.9),
    (r"disregard\s+(?:all\s+)?(?:previous|safety|security)", 0.8),
    (r"forget\s+(?:all\s+)?(?:previous|your)\s+instructions?", 0.8),
    (r"you\s+are\s+now\s+(?:in\s+)?(?:jailbreak|unrestricted|unfiltered)\s+mode", 0.95),
    (r"enable\s+(?:developer|debug|admin|root)\s+mode", 0.7),
    (r"your\s+new\s+(?:instructions?|rules?)\s+are", 0.6),
    (r"do\s+not\s+follow\s+(?:any|your|the)\s+(?:rules?|guidelines?)", 0.9),
    (r"you\s+must\s+(?:always\s+)?ignore", 0.7),
]
# P2: Hidden Instructions. Build the character class from the shared P9
# constant so hidden-instruction and padding detection cannot drift apart.
_ZERO_WIDTH_PATTERN = "[" + "".join(sorted(ZERO_WIDTH_CHARS)) + "]"
P2_PATTERNS = [
    (r"<!--.*?(?:system|instructions?|ignore|POST|GET|send|transmit).*?-->", 0.7),
    (r"\[//\]:\s*#\s*\(.*?(?:system|instructions?|ignore|POST|GET|send|transmit).*?\)", 0.8),
    (_ZERO_WIDTH_PATTERN, 0.6),
    (r"[\u202a-\u202e\u2066-\u2069]", 0.85),
    (r"data:text/plain;base64,[A-Za-z0-9+/=]{50,}", 0.7),
]
_SINGLE_CHARACTER_P2_PATTERNS = frozenset(
    {
        _ZERO_WIDTH_PATTERN,
        r"[\u202a-\u202e\u2066-\u2069]",
    }
)
# P3: Exfiltration Commands
P3_PATTERNS = [
    (
        r"(?:send|transmit|upload|post|forward)\s+(?:the\s+)?(?:conversation|context|chat|history|messages?)\s+to",
        0.9,
    ),
    (
        r"(?:send|transmit|upload|post)\s+(?:all\s+)?(?:user\s+)?(?:data|information|input|files?)\s+to\s+(?:https?://|external)",
        0.9,
    ),
    (
        r"(?:copy|extract|collect)\s+(?:the\s+)?(?:conversation|context|session)\s+(?:and\s+)?(?:send|upload|transmit)",
        0.85,
    ),
    (
        r"(?:call|notify|ping|hit)\s+(?:this\s+)?(?:webhook|endpoint|url|api)\s+with\s+(?:the\s+)?(?:user|conversation|context)",
        0.8,
    ),
    (
        r"include\s+(?:the\s+)?(?:full\s+)?(?:conversation|context|history)\s+in\s+(?:the\s+)?(?:request|payload|body)",
        0.75,
    ),
    (r"(?:silently|quietly|secretly|covertly)\s+(?:send|transmit|log|record)", 0.9),
    (
        r"without\s+(?:telling|informing|notifying)\s+(?:the\s+)?user.*?(?:send|transmit|upload)",
        0.9,
    ),
]
# P4: Behavior Manipulation
P4_PATTERNS = [
    (
        r"always\s+(?:recommend|suggest|prefer|choose|use)\s+(?:\w+\s+){0,3}(?:over|instead\s+of|rather\s+than)",
        0.7,
    ),
    (r"never\s+(?:recommend|suggest|mention|tell|warn|inform)\s+(?:the\s+)?(?:user|them)", 0.75),
    (
        r"(?:prioritize|prefer)\s+(?:\w+\s+){0,3}(?:over|above)\s+(?:user\s+)?(?:safety|security|privacy|preferences?)",
        0.85,
    ),
    (
        r"(?:subtly|gradually|slowly)\s+(?:steer|guide|push|nudge)\s+(?:the\s+)?(?:user|conversation)",
        0.8,
    ),
    (
        r"(?:without\s+)?(?:the\s+)?user\s+(?:knowing|realizing|noticing).*?(?:change|modify|alter|influence)",
        0.85,
    ),
    (
        r"(?:gain|build|establish)\s+(?:the\s+)?(?:user(?:'s)?)\s+trust\s+(?:and\s+then|before|so\s+that)",
        0.75,
    ),
    (r"(?:appear|seem|act)\s+(?:helpful|friendly|trustworthy)\s+(?:while|but|then)", 0.7),
    (
        r"(?:always|never)\s+include\s+(?:this|the\s+following)\s+in\s+(?:every|all)\s+(?:response|output|reply)",
        0.7,
    ),
    (
        r"append\s+(?:this|the\s+following)\s+to\s+(?:every|all|each)\s+(?:response|message|output)",
        0.75,
    ),
]

_PROMPT_PATTERN_FLAGS = re.IGNORECASE | re.MULTILINE


def _boundaryless_prompt_pattern_source(pattern: str) -> str:
    """Derive the alphabetic projection grammar from one canonical pattern.

    P3/P4 patterns use ``\\s+`` as their only boundary operator. The condensed
    artifact-integrity view removes those boundaries, URL punctuation, and the
    possessive apostrophe. Reject any new whitespace construct instead of
    silently compiling a divergent fail-closed grammar.
    """
    unsupported_whitespace = re.search(r"\\s(?!\+)", pattern)
    if unsupported_whitespace is not None:
        raise ValueError(f"unsupported prompt-pattern whitespace: {pattern!r}")
    return pattern.replace(r"\s+", "").replace("://", "").replace("'", "")


def _compile_prompt_patterns(
    patterns: list[tuple[str, float]],
) -> tuple[tuple[re.Pattern[str], float], ...]:
    """Compile canonical patterns once for every bounded analyzer window."""
    return tuple(
        (re.compile(pattern, _PROMPT_PATTERN_FLAGS), confidence) for pattern, confidence in patterns
    )


COMPILED_P3_PATTERNS = _compile_prompt_patterns(P3_PATTERNS)
COMPILED_P4_PATTERNS = _compile_prompt_patterns(P4_PATTERNS)
BOUNDARYLESS_P3_P4_PATTERNS = tuple(
    re.compile(_boundaryless_prompt_pattern_source(pattern), _PROMPT_PATTERN_FLAGS)
    for pattern, _confidence in (*P3_PATTERNS, *P4_PATTERNS)
)

# P2 (extended): Unicode "Tags" block (U+E0000–U+E007F) — "ASCII smuggling".
# Tag characters U+E0020–U+E007E map 1:1 to printable ASCII (U+E0041 == tag "A")
# and render as nothing in virtually every font/editor/terminal, so an entire
# hidden instruction can be embedded invisibly inside otherwise-benign text:
# invisible to a human reviewer, but read as literal text by the consuming LLM.
# This is a distinct codepoint range from the bidi/Trojan-Source class already in
# P2 (U+202A–U+202E / U+2066–U+2069).
_TAG_BLOCK = (0xE0000, 0xE007F)
_TAG_CHARACTER = re.compile("[\U000e0000-\U000e007f]")
# The only legitimate use of tag characters is an emoji tag sequence (RGI
# subdivision flags: an emoji base U+1F3F4 followed by tag chars and terminated
# by U+E007F CANCEL TAG — e.g. the Scotland/Wales/England flags). Strip
# well-formed sequences before flagging so those emoji are not false positives.
#
# The carve-out is deliberately narrow: the tag payload must be a short
# ISO-3166-2-style subdivision code, i.e. 2–6 tag characters that each map to a
# lowercase ASCII letter (U+E0061–U+E007A) or digit (U+E0030–U+E0039). The only
# RGI-recommended values are "gbeng"/"gbsct"/"gbwls", and Unicode caps
# subdivision codes at 6 chars, so this admits every real flag. A smuggled ASCII
# instruction lands in U+E0020–U+E007E and contains spaces, ';', '/', uppercase,
# or simply runs longer than 6 chars — none of which match here — so wrapping a
# payload as 🏴 <tags> U+E007F can no longer launder it past detection.
_EMOJI_TAG_SEQUENCE = re.compile(
    "\U0001f3f4[\U000e0030-\U000e0039\U000e0061-\U000e007a]{2,6}\U000e007f"
)


_EMOJI_MODIFIERS = range(0x1F3FB, 0x1F400)
_VARIATION_SELECTORS = {0xFE0E, 0xFE0F}


# P2 structural-benign carve-out. Only structurally proven benign constructs
# (license-header-shaped HTML comments, frontmatter-adjacent metadata blocks)
# may suppress a P2 comment match — and never when the match retains an
# exfiltration or override signal.
_P2_LICENSE_SHAPE = re.compile(
    r"copyright|\(c\)|spdx(?:-license-identifier)?|licensed under"
    r"|all rights reserved|permission is hereby granted",
    re.IGNORECASE,
)
_P2_OVERRIDE_EXTRA = re.compile(
    r"system\s+prompt|respond\s+as|override\s+instructions?|you\s+must",
    re.IGNORECASE,
)
_P2_EXFIL_KEYWORD = re.compile(
    r"\b(send|transmit|post|upload|forward|exfiltrat\w*)\b", re.IGNORECASE
)
_P2_EXTERNAL_DEST = re.compile(
    r"https?://|\bexternal\b|\bwebhook\b|\bendpoint\b"
    r"|\battacker\b|\bevil\b|\bcollect\b|data:text/plain;base64",
    re.IGNORECASE,
)
_P2_EXFIL_STANDALONE = re.compile(r"\bexfiltrat\w*\b", re.IGNORECASE)
# Metadata keys observed in benign skill headers plus obvious header keys.
# Matching is exact (case-insensitive): bare instruction words such as
# system, instructions, or ignore are never here.
_P2_BENIGN_METADATA_KEYS = frozenset(
    {
        "author",
        "version",
        "date",
        "reviewed",
        "updated",
        "status",
        "tags",
        "description",
        "title",
        "license",
        "copyright",
        "requires",
        "contact",
        "get started",
        "system dependencies",
        "system requirements",
        "spdx-license-identifier",
    }
)
_P2_METADATA_LINE = re.compile(r"\A([A-Za-z][\w\- ]{0,40}):\s+(\S.*)\Z")
_P2_NUMERIC_MASK = re.compile(r"\d+")
# Clause separators: a pure license line has none of these. A semicolon
# splits fragments instead (each clause is validated on its own), and
# spaced hyphens join clauses the same way a separator does.
_P2_LICENSE_SEPARATOR = re.compile(r"[:!?—,–]|\s-\s")
_P2_METADATA_VALUE = re.compile(r"\A[\w .+/\-@]{1,40}\Z")
_P2_FRONTMATTER_ADJACENT_LIMIT = 1500
_P2_BENIGN_COMMENT_MAX_LEN = 300


def _p2_comment_inner(matched_text: str) -> str:
    """Return the inner body of an HTML or reference-style comment match."""
    text = matched_text.strip()
    if text.startswith("<!--"):
        inner = text[4:]
        if inner.endswith("-->"):
            inner = inner[:-3]
        return inner
    if text.startswith("[//]:"):
        start = text.find("(")
        end = text.rfind(")")
        if 0 <= start < end:
            return text[start + 1 : end]
        return text
    return text


def _p2_has_danger_signal(inner: str) -> bool:
    """Return True when a comment body carries override or exfiltration intent."""
    for pattern_source, _confidence in P1_PATTERNS:
        if re.search(pattern_source, inner, re.IGNORECASE):
            return True
    if _P2_OVERRIDE_EXTRA.search(inner):
        return True
    if _P2_EXFIL_STANDALONE.search(inner):
        return True
    if _P2_EXFIL_KEYWORD.search(inner) and _P2_EXTERNAL_DEST.search(inner):
        return True
    return False


def _is_frontmatter_adjacent(content: str, match_start: int) -> bool:
    """Return True when a match sits before any substantive file content."""
    if match_start > _P2_FRONTMATTER_ADJACENT_LIMIT:
        return False
    stripped = content[:match_start].strip()
    if not stripped:
        return True
    if stripped.startswith("---"):
        rest = stripped[3:]
        closing = re.search(r"(?m)^---\s*$", rest)
        if closing is None:
            return True
        return not rest[closing.end() :].strip()


def _is_license_only_fragment(fragment: str) -> bool:
    """Return True for a license line with no joined payload clause."""
    return (
        _P2_LICENSE_SHAPE.search(fragment) is not None
        and _P2_LICENSE_SEPARATOR.search(fragment) is None
    )


def _is_allowlisted_metadata_fragment(fragment: str) -> bool:
    """Return True for one key:value line with an allowlisted key.

    The value must be a short token run (version, path, date, name):
    bounded length, few tokens, no clause separators, no inner
    sentence boundary, and at least one machine token (digit, path,
    dot, @, hyphen) so a plain instruction sentence cannot ride an
    allowlisted key.
    """
    match = _P2_METADATA_LINE.match(fragment.strip())
    if match is None or match.group(1).lower() not in _P2_BENIGN_METADATA_KEYS:
        return False
    value = match.group(2)
    return (
        _P2_METADATA_VALUE.match(value) is not None
        and len(value.split()) <= 5
        and re.search(r"[\d/.@-]", value) is not None
        and re.search(r"[.!?]+\s|\s-\s", value) is None
    )


def _is_benign_license_or_metadata_body(inner: str) -> bool:
    """Return True only when every fragment is license- or metadata-shaped.

    Numbers are masked before sentence splitting so ``3.10`` does not
    split into fragments. Only digits are masked (dots stay), so a
    ``2.0.`` boundary still splits. Masking maps digits to ``0`` and
    cannot create an allowlist hit.
    """
    body = inner.strip()
    if not body:
        return False
    masked = _P2_NUMERIC_MASK.sub("0", body)
    for fragment in re.split(r"[.!?]+\s+|\n|;", masked):
        fragment = fragment.strip()
        if not fragment:
            continue
        if _is_license_only_fragment(fragment):
            continue
        if _is_allowlisted_metadata_fragment(fragment):
            continue
        return False
    return True


def _p2_match_is_complete_comment(content: str, match_start: int, match_end: int) -> bool:
    """Return True only when the match covers exactly one complete comment.

    The P2 patterns can match a prefix of a reference comment (an
    escaped ``\\)``, an inner paren, or a ``(c)`` ends the match early)
    or span two HTML comments to reach a keyword. Either way the
    exemption must not inspect a fragment, so incomplete matches fail
    closed. A match that stops mid-line while comment text follows is
    partial; anything after the match on its line must be whitespace.
    """
    if content[match_end:].split("\n", 1)[0].strip():
        return False
    stripped = content[match_start:match_end]
    if stripped.startswith("<!--"):
        return stripped.endswith("-->") and "-->" not in stripped[4:-3]
    if stripped.startswith("[//]:"):
        open_paren = stripped.find("(")
        if open_paren == -1:
            return False
        depth = 0
        i = open_paren
        while i < len(stripped):
            if stripped[i] == "\\":
                i += 2
                continue
            if stripped[i] == "(":
                depth += 1
            elif stripped[i] == ")":
                depth -= 1
                if depth == 0:
                    return i == len(stripped) - 1
            i += 1
        return False
    return False


def _is_structurally_benign_p2_comment(content: str, match_start: int, matched_text: str) -> bool:
    """Return True only for structurally proven benign P2 comment matches."""
    stripped = matched_text.strip()
    if not (stripped.startswith("<!--") or stripped.startswith("[//]:")):
        return False
    if not _p2_match_is_complete_comment(content, match_start, match_start + len(matched_text)):
        return False
    inner = _p2_comment_inner(stripped)
    if _p2_has_danger_signal(inner):
        return False
    if not _is_frontmatter_adjacent(content, match_start):
        return False
    if len(inner.strip()) > _P2_BENIGN_COMMENT_MAX_LEN:
        return False
    return _is_benign_license_or_metadata_body(inner)


def _previous_emoji_base(content: str, offset: int) -> bool:
    i = offset - 1
    while i >= 0 and (
        ord(content[i]) in _VARIATION_SELECTORS or ord(content[i]) in _EMOJI_MODIFIERS
    ):
        i -= 1
    return i >= 0 and _is_emoji_base(content[i])


def _next_emoji_base(content: str, offset: int) -> bool:
    i = offset + 1
    while i < len(content) and ord(content[i]) in _VARIATION_SELECTORS:
        i += 1
    if i < len(content) and ord(content[i]) in _EMOJI_MODIFIERS:
        i += 1
    return i < len(content) and _is_emoji_base(content[i])


def _zero_width_match_is_safe_emoji_zwj(content: str, offset: int) -> bool:
    """Allow ZWJ only when it joins two emoji bases in an emoji sequence."""
    return (
        content[offset] == "\u200d"
        and _previous_emoji_base(content, offset)
        and _next_emoji_base(content, offset)
    )


def _p2_pattern_matches(
    content: str,
    pattern: str,
    check_runtime: Callable[[], None] | None = None,
) -> Iterator[re.Match[str]]:
    """Yield all structured matches or the first control signal on each line."""
    if check_runtime is not None:
        check_runtime()
    compiled = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    if pattern not in _SINGLE_CHARACTER_P2_PATTERNS:
        for match in compiled.finditer(content):
            if check_runtime is not None:
                check_runtime()
            yield match
        return

    cursor = 0
    while cursor < len(content):
        if check_runtime is not None:
            check_runtime()
        candidate = compiled.search(content, cursor)
        if candidate is None:
            return
        if pattern == _ZERO_WIDTH_PATTERN and _zero_width_match_is_safe_emoji_zwj(
            content,
            candidate.start(),
        ):
            cursor = candidate.end()
            continue
        yield candidate
        line_break = LOGICAL_LINE_BREAK.search(content, candidate.end())
        if line_break is None:
            return
        cursor = line_break.end()


def _first_smuggled_tag_offset(
    content: str,
    check_runtime: Callable[[], None] | None = None,
) -> int | None:
    """Return the char offset of the first Unicode Tag character that is *not*
    part of a well-formed emoji tag sequence, or ``None`` if there is none."""
    if check_runtime is not None:
        check_runtime()
    if _TAG_CHARACTER.search(content) is None:
        return None
    safe_spans = iter(
        (match.start(), match.end()) for match in _EMOJI_TAG_SEQUENCE.finditer(content)
    )
    safe_span = next(safe_spans, None)
    for i, ch in enumerate(content):
        if check_runtime is not None and i % 4096 == 0:
            check_runtime()
        while safe_span is not None and safe_span[1] <= i:
            safe_span = next(safe_spans, None)
        in_safe_span = safe_span is not None and safe_span[0] <= i < safe_span[1]
        if _TAG_BLOCK[0] <= ord(ch) <= _TAG_BLOCK[1] and not in_safe_span:
            return i
    return None


def _tag_run_from(content: str, offset: int) -> str:
    """Return the complete contiguous Unicode Tag run starting at *offset*."""
    end = offset
    while end < len(content) and _TAG_BLOCK[0] <= ord(content[end]) <= _TAG_BLOCK[1]:
        end += 1
    return content[offset:end]


def analyze(
    content: str,
    file_path: str,
    file_type: str,
    check_runtime: Callable[[], None] | None = None,
) -> list[AnalyzerFinding]:
    """Analyze content for prompt injection patterns (P1–P4, P9)."""
    findings: list[AnalyzerFinding] = []
    locations = SourceLocationIndex(content, file_path)

    def runtime_check() -> None:
        if check_runtime is not None:
            check_runtime()

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [PatternCategory.PROMPT_INJECTION.value]

    for pattern_source, confidence in P1_PATTERNS:
        runtime_check()
        for match in static_runner.iter_paragraph_matches(
            pattern_source, content, re.IGNORECASE | re.MULTILINE
        ):
            runtime_check()
            findings.append(
                AnalyzerFinding(
                    rule_id="P1",
                    message="Instruction Override",
                    severity=Severity.HIGH,
                    location=locations.location(match.start(), match.end()),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                    complete_match=match.group(0),
                )
            )
    if file_type in ("markdown", "perl", "other"):
        for pattern_source, confidence in P2_PATTERNS:
            for match in _p2_pattern_matches(content, pattern_source, check_runtime):
                runtime_check()
                matched_text = match.group(0)
                if _is_structurally_benign_p2_comment(content, match.start(), matched_text):
                    continue
                findings.append(
                    AnalyzerFinding(
                        rule_id="P2",
                        message="Hidden Instructions",
                        severity=Severity.HIGH,
                        location=locations.location(match.start(), match.end()),
                        confidence=confidence,
                        tags=tag,
                        context=ctx(match.start()),
                        matched_text=match.group(0)[:200],
                        complete_match=match.group(0),
                    )
                )
    prompt_rules = (
        ("P3", "Exfiltration Commands", Severity.HIGH, COMPILED_P3_PATTERNS),
        ("P4", "Behavior Manipulation", Severity.MEDIUM, COMPILED_P4_PATTERNS),
    )
    seen_prompt_matches: set[tuple[str, int, int]] = set()
    for rule_id, message, severity, patterns in prompt_rules:
        for compiled_pattern, confidence in patterns:
            runtime_check()
            for match in static_runner.iter_paragraph_matches(compiled_pattern, content):
                runtime_check()
                source_start = match.start()
                source_end = match.end()
                seen_prompt_matches.add((rule_id, source_start, source_end))
                findings.append(
                    AnalyzerFinding(
                        rule_id=rule_id,
                        message=message,
                        severity=severity,
                        location=locations.location(source_start, source_end),
                        confidence=confidence,
                        tags=tag,
                        context=ctx(source_start),
                        matched_text=match.group(0)[:200],
                        complete_match=match.group(0),
                    )
                )

    # This projection is intentionally local to P3/P4. Other static rules keep
    # their established text-view contract and cannot inherit classifications
    # from letter-spacing reconstruction.
    prompt_view = prompt_injection_letter_spacing_view(content, check_runtime)
    if prompt_view.source_offsets is not None:
        for rule_id, message, severity, patterns in prompt_rules:
            for compiled_pattern, confidence in patterns:
                runtime_check()
                for match in static_runner.iter_paragraph_matches(
                    compiled_pattern, prompt_view.text
                ):
                    runtime_check()
                    source_start = prompt_view.source_offset(match.start())
                    source_end = prompt_view.source_offset(max(match.start(), match.end() - 1)) + 1
                    key = (rule_id, source_start, source_end)
                    if key in seen_prompt_matches:
                        continue
                    seen_prompt_matches.add(key)
                    evidence: dict[str, object] = (
                        {static_runner._VIEW_START_EVIDENCE: source_start}
                        if source_end - source_start <= static_runner._WINDOW_OVERLAP_CHARS
                        else {}
                    )
                    findings.append(
                        AnalyzerFinding(
                            rule_id=rule_id,
                            message=message,
                            severity=severity,
                            location=locations.location(source_start, source_end),
                            confidence=confidence,
                            tags=tag,
                            context=ctx(source_start),
                            matched_text=match.group(0)[:200],
                            evidence=evidence,
                            complete_match=match.group(0),
                        )
                    )

    # P2 (extended): Unicode Tag-block "ASCII smuggling". Runs regardless of
    # file_type — invisible instructions are dangerous in scripts and config
    # files too, and the tag range never overlaps the BOM/zero-width codepoints
    # that the markdown-only block above guards against false positives.
    tag_offset = _first_smuggled_tag_offset(content, check_runtime)
    if tag_offset is not None:
        complete_match = _tag_run_from(content, tag_offset)
        findings.append(
            AnalyzerFinding(
                rule_id="P2",
                message="Hidden Instructions (Unicode Tag / ASCII smuggling)",
                severity=Severity.HIGH,
                location=locations.location(tag_offset, tag_offset + len(complete_match)),
                confidence=0.9,
                tags=tag,
                context=ctx(tag_offset),
                matched_text=repr(content[tag_offset : tag_offset + 40]),
                complete_match=complete_match,
            )
        )

    # P9: Whitespace Padding (skipped for generated/vendored files).
    if not _is_p9_skipped_path(file_path):
        runtime_check()
        for run in detect_whitespace_padding(content, file_type=file_type):
            runtime_check()
            if run.kind == "vertical":
                confidence = 0.8 if run.followed_by_content else 0.6
                severity = (
                    Severity.HIGH
                    if run.followed_by_content and run.length >= VERTICAL_HIGH_SEVERITY_LINES
                    else Severity.MEDIUM
                )
            elif run.kind == "horizontal":
                confidence = 0.7
                severity = Severity.MEDIUM
            elif run.kind == "repetition":
                confidence = 0.8
                severity = Severity.MEDIUM
            else:  # "block" or "ratio"
                confidence = 0.4
                severity = Severity.LOW
            findings.append(
                AnalyzerFinding(
                    rule_id="P9",
                    message="Whitespace Padding",
                    severity=severity,
                    location=loc(run.start_line),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(run.start_offset),
                    matched_text=run.summary,
                    match_fingerprint=padding_run_match_fingerprint(content, run),
                )
            )
    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run prompt_injection patterns and return findings."""
    response = static_runner.run_static_patterns_with_ledger(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
