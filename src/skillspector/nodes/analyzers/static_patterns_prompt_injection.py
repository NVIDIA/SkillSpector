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
# Bidirectional control characters (Trojan Source, CVE-2021-42574). Evaluated
# unconditionally below, NOT in the markdown-gated P2_PATTERNS list -- see the
# comment at that check for why.
# The class is the 9 paired controls only. The direction marks U+200E/U+200F/
# U+061C are deliberately excluded: they cannot reorder a span the way the
# paired controls can, which is the same line GCC's -Wbidi-chars draws by
# default. Do not add them here without revisiting that tradeoff.
_BIDI_CONTROL_PATTERN = r"[\u202a-\u202e\u2066-\u2069]"
P2_PATTERNS = [
    (r"<!--.*?(?:system|instructions?|ignore|POST|GET|send|transmit).*?-->", 0.7),
    (r"\[//\]:\s*#\s*\(.*?(?:system|instructions?|ignore|POST|GET|send|transmit).*?\)", 0.8),
    (_ZERO_WIDTH_PATTERN, 0.6),
    (r"data:text/plain;base64,[A-Za-z0-9+/=]{50,}", 0.7),
]
_SINGLE_CHARACTER_P2_PATTERNS = frozenset(
    {
        _ZERO_WIDTH_PATTERN,
        _BIDI_CONTROL_PATTERN,
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
# (anchored license-line forms, per-key metadata lines in complete comments
# near the top of the file or directly after closed frontmatter) may
# suppress a P2 comment match — and never when the match retains an
# exfiltration or override signal.
#
# Each license fragment must FULLY match one anchored form below. The
# substring check these replace let space-joined payloads through
# ("Copyright ... <arbitrary instruction>" with no separators). The
# holder is capped at six name-like tokens and about 60 characters
# ("NVIDIA CORPORATION & AFFILIATES" is three); fragments carrying a
# standalone trigger word never exempt. One trailing period is
# allowed ("All rights reserved."). Residual: a very short
# triggerless instruction fits the holder cap ("Copyright 2026 Acme
# delete everything") — accepted per the anchored-forms spec; the
# surrounding gates (danger signal first, top-of-file, complete
# comment, 300 chars) still apply.
# Shared grammar cores: the year-range + holder cap and the SPDX
# expression each appear in both a license-line form and a metadata
# value form. One core per shape so the next cap change edits one place.
# Name-like tokens carry letters, digits, underscore, and header
# punctuation only (no colon, slash, tilde, or inner dot — the path
# grammar needs the dot for the file extension) with at most one
# inner separator ([-,_]) and an optional trailing dot or comma
# (Corp., Acme,), so a dot-, comma-, hyphen-, colon-, or
# slash-joined sentence never counts as tokens.
_P2_NAME_TOKEN = r"[A-Za-z0-9&'()+_]+(?:[-,_][A-Za-z0-9&'()+_]+)?[.,]?"
_P2_COPYRIGHT_CORE = (
    r"(?=.{1,60}\Z)(?:\(c\)\s+|©\s+)?\d{1,4}(?:\s*-\s*\d{1,4})?"
    r"\s+(?:" + _P2_NAME_TOKEN + r"\s+){0,5}" + _P2_NAME_TOKEN + r"\.?"
)
# SPDX ids allow at most two hyphens, plus a third only in a
# digit-bearing id (versioned compounds such as GPL-3.0-or-later and
# CC-BY-SA-4.0; all-alpha 3-hyphen runs stay refused). Dots sit only
# between digits, so dot-joined prose never fits. Length is scoped to
# the id run so chains and trailing periods never shrink it; chains
# cap at three ids.
_P2_SPDX_ID = (
    r"(?=[A-Za-z0-9.+\-]{1,24}(?![A-Za-z0-9.+\-]))"
    r"(?:[A-Za-z0-9+]+(?:\.[0-9]+)?(?:-[A-Za-z0-9+]+(?:\.[0-9]+)?){0,2}"
    r"|(?=[A-Za-z0-9.+\-]*[0-9])[A-Za-z0-9+]+(?:\.[0-9]+)?"
    r"(?:-[A-Za-z0-9+]+(?:\.[0-9]+)?){0,3})"
)
_P2_SPDX_ATOM = _P2_SPDX_ID + r"(?:\s+(?:OR|AND|WITH)\s+" + _P2_SPDX_ID + r"){0,2}"
_P2_LICENSE_LINE_RES = (
    re.compile(
        r"\ACopyright\s+" + _P2_COPYRIGHT_CORE + r"\Z",
        re.IGNORECASE,
    ),
    re.compile(r"\AAll rights reserved\.?\Z", re.IGNORECASE),
    re.compile(
        r"\ASPDX-License-Identifier:\s*" + _P2_SPDX_ATOM + r"\.?\Z",
        re.IGNORECASE,
    ),
    re.compile(
        r"\ALicensed under the\s+(?=.{1,40}\s+License)"
        r"(?:" + _P2_NAME_TOKEN + r"\s+){0,2}" + _P2_NAME_TOKEN + r"\s+License"
        r"(?:, Version \d+\.\d+)?\.?\Z",
        re.IGNORECASE,
    ),
)
_P2_COPYRIGHT_LINE_INDEX = 0
_P2_SPDX_LINE_INDEX = 2
_P2_LICENSED_LINE_INDEX = 3
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
# Instruction-ish words refuse the benign exemption when they appear
# anywhere in a value or license fragment, even inside a longer word —
# that is how main's P2 regex matches. Keys are never scanned (system
# requirements, get started). "get" is deliberately excluded: as a
# substring it fires on ordinary prose (getting_started, target,
# forget), while GET-plus-URL exfiltration still trips the danger
# scans. This costs one plain-prose shape (PostgreSQL carries "post"):
# it reports P2, as it does on main, as does Systems ("system").
# Applied to values and license fragments only, never keys.
_P2_TRIGGER_SUBSTRING = re.compile(
    r"(system|instructions?|ignore|post|send|transmit)", re.IGNORECASE
)


def _p2_trigger_scan_text(text: str) -> str:
    """Return a de-obfuscated copy of ``text`` for the danger scans.

    camelCase joints, underscores, and digits become spaces, so
    IgnorePriorInstructions scans as Ignore Prior Instructions.
    Only the danger scans use the copy; the exemption trigger check
    is a raw substring, which needs no de-obfuscation.
    """
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    text = re.sub(r"[0-9_]+", " ", text)
    return re.sub(r"\s+", " ", text)


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
# Per-key metadata value grammars. Token content is constrained as
# well as token count: versions are semver-like (with a trailing + or *
# for "3.10+"-style floors), SPDX atoms are bounded-length ids, names
# are 1-4 capitalized tokens or an email, requires entries each carry
# a version unless the value is one bare name (operators may follow
# the name directly: "python>=3.10"), free-text keys take one short
# token with at most two inner dots, and get-started paths need a
# dotted final segment so slash-joined prose no longer fits. A
# trigger word anywhere inside a value (system, instruction(s),
# ignore, post, send, transmit — never "get", see above) refuses
# exemption; keys are excluded since "system requirements" and "get
# started" are keys. Known residuals: a bare triggerless package name
# ("requires: numpy") parses as a package; short triggerless SPDX
# OR-chains fit the atom cap; a trigger broken inside itself
# ("ig_nore") carries no contiguous trigger run, so neither this
# check nor main's regex sees it.
# Requirement names take a package-name shape: optional @scope/
# (hyphens allowed in the scope), at most 40 characters, at most two
# -/_/. separators — a separator-joined sentence never counts as one
# bare name. Length is scoped to the name run so comma-separated
# lists never shrink it.
_P2_REQ_NAME = (
    r"(?=[A-Za-z0-9_.@/\-]{1,40}(?![A-Za-z0-9_.@/\-]))"
    r"(?:@[A-Za-z0-9_.\-]+/)?[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+){0,2}"
)
_P2_REQ_VER = r"v?\d+(?:\.\d+){0,3}(?:[-+][0-9A-Za-z.]{1,20})?[+*]?"
_P2_REQ_OP = r"(?:>=|<=|==|!=|~=|\^|>|<|=|~)"
_P2_REQ_VER_SUFFIX = r"(?:\s*" + _P2_REQ_OP + r"\s*" + _P2_REQ_VER + r"|\s+" + _P2_REQ_VER + r")"
_P2_VERSION_RE = re.compile(r"\A" + _P2_REQ_VER + r"\Z")
_P2_DATE_RE = re.compile(
    r"\A(?:\d{4}-\d{2}-\d{2}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?"
    r"|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?"
    r"|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4})\Z",
    re.IGNORECASE,
)
_P2_NAME_RE = re.compile(r"\A[A-Z][A-Za-z']*(?:\s+[A-Z][A-Za-z']*){0,3}\Z")
_P2_EMAIL_RE = re.compile(r"\A[\w.+\-]+@[\w.\-]+\.[A-Za-z]{2,}\Z")
_P2_SPDX_EXPR_RE = re.compile(r"\A" + _P2_SPDX_ATOM + r"\Z")
_P2_REQUIREMENTS_RE = re.compile(
    r"\A" + _P2_REQ_NAME + _P2_REQ_VER_SUFFIX + r"?"
    r"(?:\s*[,;]\s*" + _P2_REQ_NAME + _P2_REQ_VER_SUFFIX + r")*\Z"
)
_P2_PATH_VALUE_RE = re.compile(
    r"\A(?=[\w\-./#:?]{1,80}\Z)(?:https?://[\w\-.~]+(?::\d+)?/)?"
    r"(?:" + _P2_NAME_TOKEN + r"/){0,3}" + _P2_NAME_TOKEN + r"\.[\w]{1,10}(?:[#?]\S*)?\Z"
)
_P2_SINGLE_TOKEN_RE = re.compile(r"\A(?=[A-Za-z0-9.]{1,24}\Z)[A-Za-z0-9]+(?:\.[A-Za-z0-9]+){0,2}\Z")
_P2_COPYRIGHT_VALUE_RE = re.compile(
    r"\A" + _P2_COPYRIGHT_CORE + r"\Z",
    re.IGNORECASE,
)
# Every allowlisted key must appear here or in _P2_FREE_TEXT_KEYS
# (pinned by test_metadata_keys_all_have_grammars).
_P2_METADATA_VALUE_RES = {
    "version": (_P2_VERSION_RE,),
    "date": (_P2_DATE_RE,),
    "updated": (_P2_DATE_RE,),
    "reviewed": (_P2_DATE_RE,),
    "author": (_P2_NAME_RE, _P2_EMAIL_RE),
    "contact": (_P2_NAME_RE, _P2_EMAIL_RE),
    "license": (_P2_SPDX_EXPR_RE,),
    "spdx-license-identifier": (_P2_SPDX_EXPR_RE,),
    "requires": (_P2_REQUIREMENTS_RE,),
    "system dependencies": (_P2_REQUIREMENTS_RE,),
    "system requirements": (_P2_REQUIREMENTS_RE,),
    "get started": (_P2_PATH_VALUE_RE,),
    "copyright": (_P2_COPYRIGHT_VALUE_RE,),
}
_P2_FREE_TEXT_KEYS = frozenset({"description", "title", "status", "tags"})
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
    """Return True when a comment body carries override or exfiltration intent.

    Both the raw body and a de-obfuscated copy are scanned: the copy
    catches camelCase/underscore-joined intent, the raw body keeps
    digit-bearing patterns exact. Either firing refuses the exemption.
    """
    for text in {inner, _p2_trigger_scan_text(inner)}:
        for pattern_source, _confidence in P1_PATTERNS:
            if re.search(pattern_source, text, re.IGNORECASE):
                return True
        if _P2_OVERRIDE_EXTRA.search(text):
            return True
        if _P2_EXFIL_STANDALONE.search(text):
            return True
        if _P2_EXFIL_KEYWORD.search(text) and _P2_EXTERNAL_DEST.search(text):
            return True
    return False


_P2_FRONTMATTER_OPEN = re.compile(r"\A---[ \t]*\r?\n")
_P2_FRONTMATTER_CLOSE = re.compile(r"(?m)^---\s*$")


def _is_frontmatter_adjacent(content: str, match_start: int) -> bool:
    """Return True when a match sits before any substantive file content."""
    if match_start > _P2_FRONTMATTER_ADJACENT_LIMIT:
        return False
    stripped = content[:match_start].strip()
    if not stripped:
        return True
    open_match = _P2_FRONTMATTER_OPEN.match(stripped)
    if open_match is None:
        return False
    closing = _P2_FRONTMATTER_CLOSE.search(stripped, open_match.end())
    if closing is None:
        return False
    return not stripped[closing.end() :].strip()


def _is_license_only_fragment(fragment: str) -> bool:
    """Return True when the fragment fully matches an anchored license-line form."""
    if _P2_TRIGGER_SUBSTRING.search(fragment) is not None:
        return False
    return any(pattern.match(fragment) is not None for pattern in _P2_LICENSE_LINE_RES)


def _is_allowlisted_metadata_fragment(fragment: str) -> bool:
    """Return True for one key:value line whose value matches its key grammar.

    The key must be allowlisted and the value must fully match that key's
    narrow form (version token, date, name-or-email, SPDX expression,
    package[version] list, single path/URL, or single token for
    free-text keys) — never the old generic token-run shape that let an
    imperative instruction through on a single dot or digit.
    """
    match = _P2_METADATA_LINE.match(fragment.strip())
    if match is None:
        return False
    key = match.group(1).lower()
    value = match.group(2).strip()
    if len(value) > 1 and value.endswith("."):
        value = value[:-1]
    if _P2_TRIGGER_SUBSTRING.search(value) is not None:
        return False
    if key in _P2_FREE_TEXT_KEYS:
        return _P2_SINGLE_TOKEN_RE.match(value) is not None
    validators = _P2_METADATA_VALUE_RES.get(key)
    if not validators:
        return False
    return any(pattern.match(value) is not None for pattern in validators)


def _is_benign_license_or_metadata_body(inner: str) -> bool:
    """Return True only when every fragment is license- or metadata-shaped.

    Fragments are split and validated on the raw text: digit-masking
    never affected split points (digits are not split characters), and
    validating masked text broke real dates. An allowlisted key may
    appear only once, so an instruction split across repeated keys
    still fires.
    """
    body = inner.strip()
    if not body:
        return False
    seen_keys: set[str] = set()
    copyright_lines = 0
    spdx_lines = 0
    licensed_lines = 0
    for fragment in re.split(r"[.!?]+\s+|\n|;", body):
        fragment = fragment.strip()
        if not fragment:
            continue
        if _is_license_only_fragment(fragment):
            if _P2_LICENSE_LINE_RES[_P2_COPYRIGHT_LINE_INDEX].match(fragment) is not None:
                copyright_lines += 1
                if copyright_lines > 2:
                    return False
            elif _P2_LICENSE_LINE_RES[_P2_SPDX_LINE_INDEX].match(fragment) is not None:
                spdx_lines += 1
                if spdx_lines > 1:
                    return False
            elif _P2_LICENSE_LINE_RES[_P2_LICENSED_LINE_INDEX].match(fragment) is not None:
                licensed_lines += 1
                if licensed_lines > 1:
                    return False
            continue
        key_match = _P2_METADATA_LINE.match(fragment)
        if key_match is not None:
            key = key_match.group(1).lower()
            if key in _P2_BENIGN_METADATA_KEYS:
                if key in seen_keys:
                    return False
                seen_keys.add(key)
                if key == "copyright":
                    copyright_lines += 1
                    if copyright_lines > 2:
                        return False
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
    line_end = content.find("\n", match_end)
    tail = content[match_end:] if line_end == -1 else content[match_end:line_end]
    if tail.strip():
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
    # Cheap positional gate first: only matches in the head of the file
    # can ever qualify, so skip all string work for later matches.
    if match_start > _P2_FRONTMATTER_ADJACENT_LIMIT:
        return False
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
        if pattern == _ZERO_WIDTH_PATTERN and (
            # A leading U+FEFF is a byte-order mark, not hidden text.
            (candidate.start() == 0 and content[0] == "\ufeff")
            or _zero_width_match_is_safe_emoji_zwj(content, candidate.start())
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

    # P2 (extended): Bidirectional control characters (Trojan Source,
    # CVE-2021-42574). Runs regardless of file_type, like the Tag-block check
    # above — bidi overrides are exploitable in scripts and config files too
    # (issue #39). Zero-width detection stays markdown-gated because
    # ZERO_WIDTH_CHARS includes U+FEFF (BOM), which would false-positive on
    # every BOM-prefixed source file; the bidi range never overlaps it.
    for match in _p2_pattern_matches(content, _BIDI_CONTROL_PATTERN, check_runtime):
        runtime_check()
        findings.append(
            AnalyzerFinding(
                rule_id="P2",
                message="Hidden Instructions",
                severity=Severity.HIGH,
                location=locations.location(match.start(), match.end()),
                confidence=0.85,
                tags=tag,
                context=ctx(match.start()),
                matched_text=match.group(0)[:200],
                complete_match=match.group(0),
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
