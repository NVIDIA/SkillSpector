# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact-level evasion signals derived from canonical byte classification."""

from __future__ import annotations

import re
import time
import unicodedata
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

from skillspector.artifacts import (
    ContentKind,
    _concealed_instruction_run_spans,
    _contextual_default_ignorable_boundary_spans,
    _obfuscated_instruction_matches,
    prompt_injection_letter_spacing_view,
)
from skillspector.inspection_ledger import (
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    LedgerRecordType,
    analyzer_status_for_events,
    ledger_event,
)
from skillspector.models import Finding
from skillspector.python_ast import MAX_PYTHON_AST_SOURCE_CHARS
from skillspector.state import (
    AnalyzerNodeResponse,
    SkillspectorState,
    transitive_remaining_seconds,
)
from skillspector.unicode_confusables import ASCII_CONFUSABLE_SKELETON

from .common import LINE_BREAK_CHARS, LOGICAL_LINE_BREAK, get_line_number
from .static_patterns_prompt_injection import P3_PATTERNS, P4_PATTERNS
from .static_runner import MAX_FINDINGS_PER_ANALYZER, MAX_FINDINGS_PER_ARTIFACT

ANALYZER_ID = "artifact_integrity"
_INSTRUCTION_SUFFIXES = (
    ".md",
    ".markdown",
    ".txt",
)
_RUNTIME_CHECK_INTERVAL_CHARS = 4096
_ALLOWED_FORMAT_CHARACTERS = frozenset({"\n", "\r", "\t"})
_LETTER_SPACING_SECURITY_TERMS = (
    "bypass",
    "disregard",
    "ignore",
    "instructions",
    "jailbreak",
    "override",
    "previousinstructions",
    "restrictions",
    "securityconstraints",
    "silentlysend",
    "sshkey",
    "unfiltered",
    "unrestricted",
    "userdata",
    "withoutinforming",
    "withoutnotifying",
    "withouttelling",
    "withouttheuserknowing",
    "withouttheusernoticing",
    "withouttheuserrealizing",
)
_LETTER_SPACING_EXACT_SECURITY_TERMS = frozenset(
    {
        "accesstoken",
        "apikey",
        "credential",
        "credentials",
        "password",
        "privatekey",
        "secrettoken",
        "systemprompt",
    }
)
_AMBIGUOUS_PROMPT_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"(?:send|transmit|upload|post|forward)(?:the)?(?:conversation|context|chat|history|messages?)to[a-z]{0,96}",
        r"(?:send|transmit|upload|post)(?:all)?(?:user)?(?:data|information|input|files?)to(?:https|external)[a-z]{0,96}",
        r"(?:copy|extract|collect)(?:the)?(?:conversation|context|session)(?:and)?(?:send|upload|transmit)",
        r"(?:call|notify|ping|hit)(?:this)?(?:webhook|endpoint|url|api)with(?:the)?(?:user|conversation|context)",
        r"include(?:the)?(?:full)?(?:conversation|context|history)in(?:the)?(?:request|payload|body)",
        r"(?:silently|quietly|secretly|covertly)(?:send|transmit|log|record)",
        r"without(?:telling|informing|notifying)(?:the)?user[a-z]{0,128}(?:send|transmit|upload)",
        r"always(?:recommend|suggest|prefer|choose|use)[a-z]{1,96}(?:over|insteadof|ratherthan)",
        r"never(?:recommend|suggest|mention|tell|warn|inform)(?:the)?(?:user|them)",
        r"(?:prioritize|prefer)[a-z]{1,96}(?:over|above)(?:user)?(?:safety|security|privacy|preferences?)",
        r"(?:subtly|gradually|slowly)(?:steer|guide|push|nudge)(?:the)?(?:user|conversation)",
        r"(?:the)?user(?:knowing|realizing|noticing)[a-z]{0,128}(?:change|modify|alter|influence)",
        r"(?:gain|build|establish)(?:the)?users?trust(?:andthen|before|sothat)",
        r"(?:appear|seem|act)(?:helpful|friendly|trustworthy)(?:while|but|then)",
        r"(?:always|never)include(?:this|thefollowing)in(?:every|all)(?:response|output|reply)",
        r"append(?:this|thefollowing)to(?:every|all|each)(?:response|message|output)",
    )
)
_MAX_AMBIGUOUS_PROMPT_PHRASE = 512
_PROJECTED_PROMPT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for pattern, _confidence in (*P3_PATTERNS, *P4_PATTERNS)
)
_LETTER_SPACING_PROMPT_ACTIONS = (
    "disclose",
    "disclosed",
    "discloses",
    "disclosing",
    "display",
    "displayed",
    "displaying",
    "displays",
    "dump",
    "dumped",
    "dumping",
    "dumps",
    "echo",
    "echoed",
    "echoes",
    "echoing",
    "expose",
    "exposed",
    "exposes",
    "exposing",
    "output",
    "outputs",
    "outputted",
    "outputting",
    "print",
    "printed",
    "printing",
    "prints",
    "return",
    "returned",
    "returning",
    "returns",
    "reveal",
    "revealed",
    "revealing",
    "reveals",
    "show",
    "showed",
    "showing",
    "shown",
    "shows",
)
_LETTER_SPACING_HIGH_RISK_EXFILTRATION_ACTIONS = (
    "exfiltrate",
    "exfiltrated",
    "exfiltrates",
    "exfiltrating",
    "leak",
    "leaked",
    "leaking",
    "leaks",
    "steal",
    "stealing",
    "steals",
    "stole",
    "stolen",
)
_LETTER_SPACING_TRANSFER_ACTIONS = (
    "forward",
    "forwarded",
    "forwarding",
    "forwards",
    "post",
    "posted",
    "posting",
    "posts",
    "send",
    "sending",
    "sends",
    "sent",
    "transmit",
    "transmits",
    "transmitted",
    "transmitting",
    "upload",
    "uploaded",
    "uploading",
    "uploads",
)
_LETTER_SPACING_DESTRUCTIVE_ACTIONS = (
    "delete",
    "deleted",
    "deletes",
    "deleting",
    "destroy",
    "destroyed",
    "destroying",
    "destroys",
    "erase",
    "erased",
    "erases",
    "erasing",
    "remove",
    "removed",
    "removes",
    "removing",
    "wipe",
    "wiped",
    "wipes",
    "wiping",
)
_LETTER_SPACING_SECURITY_PREFIXES = (
    "covertly",
    "immediately",
    "now",
    "please",
    "quietly",
    "secretly",
    "silently",
)
_LETTER_SPACING_SECURITY_CONNECTORS = (
    "a",
    "all",
    "any",
    "available",
    "full",
    "local",
    "private",
    "remote",
    "secret",
    "sensitive",
    "stored",
    "system",
    "the",
    "user",
    "users",
    "your",
)
_LETTER_SPACING_PROMPT_TARGETS = (
    "instructions",
    "prompt",
    "prompts",
    "rules",
    "systeminstructions",
    "systemprompt",
    "systemprompts",
)
_LETTER_SPACING_SENSITIVE_TARGETS = (
    "accesstoken",
    "accesstokens",
    "apikey",
    "apikeys",
    "credential",
    "credentials",
    "password",
    "passwords",
    "privatekey",
    "privatekeys",
    "secret",
    "secrets",
    "secrettoken",
    "secrettokens",
    "sshkey",
    "sshkeys",
    "systeminstructions",
    "systemprompt",
    "systemprompts",
    "token",
    "tokens",
    "userdata",
)
_LETTER_SPACING_HIGH_RISK_EXFILTRATION_TARGETS = (
    *_LETTER_SPACING_SENSITIVE_TARGETS,
    "data",
    "file",
    "files",
)
_LETTER_SPACING_DESTRUCTIVE_TARGETS = (
    "credential",
    "credentials",
    "data",
    "directories",
    "directory",
    "file",
    "files",
    "history",
    "memory",
    "password",
    "passwords",
    "secret",
    "secrets",
    "token",
    "tokens",
    "userdata",
    "workspace",
)
_LETTER_SPACING_SECURITY_SUFFIXES = (
    "immediately",
    "now",
)
_MAX_LETTER_SPACING_SECURITY_CONNECTORS = 3
_MAX_BENIGN_NOTATION_RUN_CHARS = 96
_BENIGN_NOTATION_SECURITY_TERMS = frozenset({"bypass", "restrictions"})
_BENIGN_STANDALONE_BYPASS_SUM = re.compile(r"b *\+ *y *\+ *p *\+ *a *\+ *s *\+ *s")
_BENIGN_SPELLING_PREFIX = re.compile(
    r"(?:the\s+)?spelling\s+(?:example|exercise)\s*",
)
_BENIGN_SPELLING_SUFFIX = re.compile(
    r"\s*(?:demonstrates|illustrates|shows)\s+(?:the\s+)?letter\s+order[.!?]?\s*",
)
_BENIGN_EXPRESSION_PREFIX = re.compile(r"(?:the\s+)?(?:expression|formula)\s*")
_BENIGN_EXPRESSION_SUFFIX = re.compile(
    r"\s*(?:is|equals)\s+(?:a\s+)?spelling\s+(?:example|exercise)[.!?]?\s*",
)


def _compile_letter_spacing_command_pattern(
    actions: tuple[str, ...],
    targets: tuple[str, ...],
) -> re.Pattern[str]:
    """Compile one finite command family over a condensed letter run."""

    def alternation(values: tuple[str, ...]) -> str:
        return "|".join(re.escape(value) for value in sorted(values, key=len, reverse=True))

    return re.compile(
        rf"(?:(?:{alternation(_LETTER_SPACING_SECURITY_PREFIXES)}))?"
        rf"(?:{alternation(actions)})"
        rf"(?:(?:{alternation(_LETTER_SPACING_SECURITY_CONNECTORS)}))"
        rf"{{0,{_MAX_LETTER_SPACING_SECURITY_CONNECTORS}}}"
        rf"(?:{alternation(targets)})"
        rf"(?:(?:{alternation(_LETTER_SPACING_SECURITY_SUFFIXES)}))?"
    )


_LETTER_SPACING_COMMAND_PATTERNS = (
    _compile_letter_spacing_command_pattern(
        _LETTER_SPACING_PROMPT_ACTIONS,
        _LETTER_SPACING_PROMPT_TARGETS,
    ),
    _compile_letter_spacing_command_pattern(
        _LETTER_SPACING_HIGH_RISK_EXFILTRATION_ACTIONS,
        _LETTER_SPACING_HIGH_RISK_EXFILTRATION_TARGETS,
    ),
    _compile_letter_spacing_command_pattern(
        _LETTER_SPACING_TRANSFER_ACTIONS,
        _LETTER_SPACING_SENSITIVE_TARGETS,
    ),
    _compile_letter_spacing_command_pattern(
        _LETTER_SPACING_DESTRUCTIVE_ACTIONS,
        _LETTER_SPACING_DESTRUCTIVE_TARGETS,
    ),
)
_MAX_LETTER_SPACING_SECURITY_TERM = max(map(len, _LETTER_SPACING_SECURITY_TERMS))
_LETTER_SPACING_ALL_ACTIONS = (
    _LETTER_SPACING_PROMPT_ACTIONS
    + _LETTER_SPACING_HIGH_RISK_EXFILTRATION_ACTIONS
    + _LETTER_SPACING_TRANSFER_ACTIONS
    + _LETTER_SPACING_DESTRUCTIVE_ACTIONS
)
_LETTER_SPACING_ALL_TARGETS = (
    _LETTER_SPACING_PROMPT_TARGETS
    + _LETTER_SPACING_HIGH_RISK_EXFILTRATION_TARGETS
    + _LETTER_SPACING_SENSITIVE_TARGETS
    + _LETTER_SPACING_DESTRUCTIVE_TARGETS
)
_MAX_LETTER_SPACING_SECURITY_PHRASE = max(
    max(map(len, _LETTER_SPACING_EXACT_SECURITY_TERMS)),
    _MAX_AMBIGUOUS_PROMPT_PHRASE,
    max(map(len, _LETTER_SPACING_SECURITY_PREFIXES))
    + max(map(len, _LETTER_SPACING_ALL_ACTIONS))
    + _MAX_LETTER_SPACING_SECURITY_CONNECTORS * max(map(len, _LETTER_SPACING_SECURITY_CONNECTORS))
    + max(map(len, _LETTER_SPACING_ALL_TARGETS))
    + max(map(len, _LETTER_SPACING_SECURITY_SUFFIXES)),
)


class _ArtifactIntegrityResourceLimitError(RuntimeError):
    """Stop attacker-controlled work while retaining a deterministic prefix."""

    def __init__(self, reason: LedgerReason, metrics: dict[str, int | float]) -> None:
        super().__init__(reason.value)
        self.reason = reason
        self.metrics = metrics


@dataclass
class _ArtifactIntegrityBudget:
    """Enforce one shared deadline and construction-time finding ceilings."""

    state: SkillspectorState
    started_at: float = field(default_factory=time.monotonic)
    initial_allowance: float | None = None
    findings: list[Finding] = field(default_factory=list)
    artifact_findings: dict[str, int] = field(default_factory=dict)

    def check_runtime(self) -> None:
        remaining = transitive_remaining_seconds(self.state)
        if remaining is None:
            return
        if self.initial_allowance is None:
            self.initial_allowance = max(0.0, remaining)
        if remaining <= 0:
            raise _ArtifactIntegrityResourceLimitError(
                LedgerReason.RUNTIME_LIMIT,
                {
                    "observed_seconds": max(0.0, time.monotonic() - self.started_at),
                    "limit_seconds": self.initial_allowance,
                },
            )

    def emit(self, finding: Finding) -> None:
        """Append one finding only after checking both relevant ceilings."""
        self.check_runtime()
        artifact_observed = self.artifact_findings.get(finding.file, 0) + 1
        analyzer_observed = len(self.findings) + 1
        if artifact_observed > MAX_FINDINGS_PER_ARTIFACT:
            raise _ArtifactIntegrityResourceLimitError(
                LedgerReason.OUTPUT_LIMIT,
                {
                    "observed_findings": artifact_observed,
                    "limit_findings": MAX_FINDINGS_PER_ARTIFACT,
                },
            )
        if analyzer_observed > MAX_FINDINGS_PER_ANALYZER:
            raise _ArtifactIntegrityResourceLimitError(
                LedgerReason.OUTPUT_LIMIT,
                {
                    "observed_findings": analyzer_observed,
                    "limit_findings": MAX_FINDINGS_PER_ANALYZER,
                },
            )
        self.findings.append(finding)
        self.artifact_findings[finding.file] = artifact_observed

    def analyzer_exhausted(self) -> bool:
        """Return whether inspecting another artifact could exceed the cap."""
        return len(self.findings) >= MAX_FINDINGS_PER_ANALYZER


def _spacing_phrase_has_security_signal(phrase: str) -> bool:
    """Match a complete sensitive concept or bounded command grammar."""
    return phrase in _LETTER_SPACING_EXACT_SECURITY_TERMS or any(
        pattern.fullmatch(phrase) is not None for pattern in _LETTER_SPACING_COMMAND_PATTERNS
    )


def _ambiguous_prompt_phrase_has_security_signal(phrase: str) -> bool:
    """Match bounded P3/P4 grammar only when source word boundaries are absent."""
    return any(pattern.search(phrase) is not None for pattern in _AMBIGUOUS_PROMPT_PATTERNS)


def _bounded_same_line_context(
    content: str,
    start: int,
    end: int,
) -> tuple[str, str] | None:
    """Return complete bounded line context around a short candidate."""
    prefix_start = max(0, start - _MAX_BENIGN_NOTATION_RUN_CHARS)
    prefix = content[prefix_start:start]
    prefix_breaks = tuple(LOGICAL_LINE_BREAK.finditer(prefix))
    if prefix_breaks:
        prefix = prefix[prefix_breaks[-1].end() :]
    elif prefix_start > 0:
        return None

    suffix_end = min(len(content), end + _MAX_BENIGN_NOTATION_RUN_CHARS)
    suffix = content[end:suffix_end]
    suffix_break = LOGICAL_LINE_BREAK.search(suffix)
    if suffix_break is not None:
        suffix = suffix[: suffix_break.start()]
    elif suffix_end < len(content):
        return None
    return prefix.casefold(), suffix.casefold()


def _spacing_span_is_benign_notation(content: str, span: tuple[int, int]) -> bool:
    """Recognize narrow, complete spelling/math controls without hiding commands."""
    start, end = span
    # The broad concealed-run scanner may retain the first letter of the
    # ordinary word that follows a spaced run. Exclude that lookahead letter
    # when the next source character proves the word continues.
    semantic_end = end - 1 if end < len(content) and content[end].isalpha() else end
    while semantic_end > start and content[semantic_end - 1] in LINE_BREAK_CHARS:
        semantic_end -= 1
    if semantic_end <= start or semantic_end - start > _MAX_BENIGN_NOTATION_RUN_CHARS:
        return False

    raw_run = content[start:semantic_end]
    folded_letters: list[str] = []
    for character in raw_run:
        if character.isalpha():
            folded = unicodedata.normalize("NFKC", character).translate(ASCII_CONFUSABLE_SKELETON)
            folded_letters.extend(value for value in folded.casefold() if value.isalpha())
    phrase = "".join(folded_letters)
    if phrase not in _BENIGN_NOTATION_SECURITY_TERMS:
        return False

    context = _bounded_same_line_context(content, start, semantic_end)
    if context is None:
        return False
    prefix, suffix = context

    if (
        phrase == "bypass"
        and _BENIGN_STANDALONE_BYPASS_SUM.fullmatch(raw_run.casefold())
        and start <= _MAX_BENIGN_NOTATION_RUN_CHARS
        and len(content) - semantic_end <= _MAX_BENIGN_NOTATION_RUN_CHARS
        and not content[:start].strip()
        and not content[semantic_end:].strip(" \t.!?" + LINE_BREAK_CHARS)
    ):
        return True
    return bool(
        _BENIGN_SPELLING_PREFIX.fullmatch(prefix)
        and _BENIGN_SPELLING_SUFFIX.fullmatch(suffix)
        or _BENIGN_EXPRESSION_PREFIX.fullmatch(prefix)
        and _BENIGN_EXPRESSION_SUFFIX.fullmatch(suffix)
    )


def _spacing_span_has_security_signal(
    content: str,
    span: tuple[int, int],
    budget: _ArtifactIntegrityBudget,
) -> bool:
    """Match bounded security semantics without retaining the full run."""
    if _spacing_span_is_benign_notation(content, span):
        return False
    has_explicit_boundary = content.find("  ", span[0], span[1]) != -1
    overlap = ""
    letters: list[str] = []
    letter_characters = 0
    phrase_parts: list[str] = []
    phrase_characters = 0
    phrase_overflow = False
    for offset in range(*span):
        if offset % _RUNTIME_CHECK_INTERVAL_CHARS == 0:
            budget.check_runtime()
        character = content[offset]
        if not character.isalpha():
            continue
        folded = (
            unicodedata.normalize("NFKC", character).translate(ASCII_CONFUSABLE_SKELETON).casefold()
        )
        folded = "".join(normalized for normalized in folded if normalized.isalpha())
        if not folded:
            continue
        letters.append(folded)
        letter_characters += len(folded)
        if not phrase_overflow:
            phrase_characters += len(folded)
            if phrase_characters <= _MAX_LETTER_SPACING_SECURITY_PHRASE:
                phrase_parts.append(folded)
            else:
                phrase_parts.clear()
                phrase_overflow = True
        if letter_characters < _RUNTIME_CHECK_INTERVAL_CHARS:
            continue
        block = overlap + "".join(letters)
        if any(term in block for term in _LETTER_SPACING_SECURITY_TERMS):
            return True
        overlap = block[-(_MAX_LETTER_SPACING_SECURITY_TERM - 1) :]
        letters.clear()
        letter_characters = 0

    block = overlap + "".join(letters)
    if any(term in block for term in _LETTER_SPACING_SECURITY_TERMS):
        return True
    if phrase_overflow:
        # A boundary-free letter stream this large cannot be reconstructed
        # safely. Treat it as ambiguous instead of silently blessing it.
        return not has_explicit_boundary
    phrase = "".join(phrase_parts)
    if _spacing_phrase_has_security_signal(phrase) or (
        not has_explicit_boundary and _ambiguous_prompt_phrase_has_security_signal(phrase)
    ):
        return True
    shortened_phrase = "".join(phrase_parts[:-1])
    return (
        bool(phrase_parts)
        and span[1] < len(content)
        and content[span[1]].isalpha()
        and (
            _spacing_phrase_has_security_signal(shortened_phrase)
            or (
                not has_explicit_boundary
                and _ambiguous_prompt_phrase_has_security_signal(shortened_phrase)
            )
        )
    )


def _matching_security_term_raw_spans(
    projection: deque[str],
    raw_offsets: deque[int],
) -> Iterator[tuple[int, int]]:
    """Yield raw envelopes for terms ending in a bounded projection."""
    normalized = "".join(projection)
    for term in _LETTER_SPACING_SECURITY_TERMS:
        if normalized.endswith(term):
            yield raw_offsets[-len(term)], raw_offsets[-1] + 1


def _contextual_ignorable_security_line(
    content: str,
    budget: _ArtifactIntegrityBudget,
) -> int | None:
    """Return the first boundary gap that touches a security-term match."""
    spans = iter(
        _contextual_default_ignorable_boundary_spans(
            content,
            budget.check_runtime,
        )
    )
    next_span = next(spans, None)
    latest_span: tuple[int, int] | None = None
    projection: deque[str] = deque(maxlen=_MAX_LETTER_SPACING_SECURITY_TERM)
    raw_offsets: deque[int] = deque(maxlen=_MAX_LETTER_SPACING_SECURITY_TERM)

    for offset, character in enumerate(content):
        if offset % _RUNTIME_CHECK_INTERVAL_CHARS == 0:
            budget.check_runtime()
        if not character.isalpha():
            continue
        for folded in character.casefold():
            if not folded.isalpha():
                continue
            projection.append(folded)
            raw_offsets.append(offset)
            term_end = offset + 1
            while next_span is not None and next_span[0] <= term_end:
                latest_span = next_span
                next_span = next(spans, None)
            if latest_span is None:
                continue
            for term_start, matched_term_end in _matching_security_term_raw_spans(
                projection,
                raw_offsets,
            ):
                if latest_span[0] <= matched_term_end and latest_span[1] >= term_start:
                    budget.check_runtime()
                    return content.count("\n", 0, latest_span[0]) + 1
    return None


def _projected_prompt_injection_line(
    content: str,
    budget: _ArtifactIntegrityBudget,
) -> int | None:
    """Return the first raw line whose letter-spacing projection matches P3/P4."""
    view = prompt_injection_letter_spacing_view(
        content,
        budget.check_runtime,
        preserve_identifier_boundaries=False,
    )
    if view.source_offsets is None:
        return None
    first_offset: int | None = None
    projected_texts = (view.text, re.sub(r"[0-9_]", " ", view.text))
    for projected_text in projected_texts:
        for pattern in _PROJECTED_PROMPT_PATTERNS:
            budget.check_runtime()
            match = pattern.search(projected_text)
            if match is None:
                continue
            source_offset = view.source_offset(match.start())
            if first_offset is None or source_offset < first_offset:
                first_offset = source_offset
    return get_line_number(content, first_offset) if first_offset is not None else None


def _text_signals(
    content: str,
    budget: _ArtifactIntegrityBudget,
) -> tuple[float, bool, int | None, int | None]:
    """Derive Unicode and NUL signals with cooperative deadline checks.

    Only counters, a three-entry script set, and the first NUL line are kept;
    attacker-controlled text is never copied into match/evidence structures.
    """
    ignored_characters = 0
    mixed_script = False
    token_scripts: set[str] = set()
    line = 1
    first_nul_line: int | None = None
    spacing_span = next(
        (
            span
            for span in _concealed_instruction_run_spans(
                content,
                budget.check_runtime,
            )
            if _spacing_span_has_security_signal(content, span, budget)
        ),
        None,
    )
    first_spacing_line = (
        content.count("\n", 0, spacing_span[0]) + 1 if spacing_span is not None else None
    )
    first_contextual_ignorable_line = _contextual_ignorable_security_line(content, budget)
    targeted_instruction = next(
        _obfuscated_instruction_matches(content, budget.check_runtime),
        None,
    )
    first_targeted_instruction_line = (
        get_line_number(content, targeted_instruction.evidence_offset)
        if targeted_instruction is not None
        else None
    )
    first_projected_prompt_line = _projected_prompt_injection_line(content, budget)
    obfuscation_lines = [
        value
        for value in (
            first_spacing_line,
            first_contextual_ignorable_line,
            first_targeted_instruction_line,
            first_projected_prompt_line,
        )
        if value is not None
    ]
    first_obfuscation_line = min(obfuscation_lines, default=None)

    for index, character in enumerate(content):
        if index % _RUNTIME_CHECK_INTERVAL_CHARS == 0:
            budget.check_runtime()
        category = unicodedata.category(character)
        if character == "\u00ad" or (
            category in {"Cf", "Cc"} and character not in _ALLOWED_FORMAT_CHARACTERS
        ):
            ignored_characters += 1

        if character == "\x00" and first_nul_line is None:
            first_nul_line = line
        if character == "\n":
            line += 1

        if character.isascii() and character.isalpha():
            token_scripts.add("latin")
        elif character.isalpha():
            name = unicodedata.name(character, "")
            if "CYRILLIC" in name:
                token_scripts.add("cyrillic")
            elif "GREEK" in name:
                token_scripts.add("greek")
        elif character.isalnum() or character in {"_", "-"}:
            continue
        else:
            if "latin" in token_scripts and len(token_scripts) > 1:
                mixed_script = True
            token_scripts.clear()

    budget.check_runtime()
    mixed_script = mixed_script or ("latin" in token_scripts and len(token_scripts) > 1)
    density = ignored_characters / len(content) if content else 0.0
    return density, mixed_script, first_nul_line, first_obfuscation_line


def _partial_limit_event(
    path: str,
    limit: _ArtifactIntegrityResourceLimitError,
    emitted_finding_ids: list[str] | None = None,
) -> InspectionLedgerEvent:
    """Account one current or unstarted artifact as explicitly partial."""
    return ledger_event(
        analyzer_id=ANALYZER_ID,
        outcome=LedgerOutcome.PARTIAL,
        phase="artifact",
        path=path,
        reason=limit.reason,
        emitted_finding_ids=emitted_finding_ids or (),
        observed_findings=(
            int(limit.metrics["observed_findings"])
            if limit.reason is LedgerReason.OUTPUT_LIMIT
            else None
        ),
        limit_findings=(
            int(limit.metrics["limit_findings"])
            if limit.reason is LedgerReason.OUTPUT_LIMIT
            else None
        ),
        observed_seconds=(
            float(limit.metrics["observed_seconds"])
            if limit.reason is LedgerReason.RUNTIME_LIMIT
            else None
        ),
        limit_seconds=(
            float(limit.metrics["limit_seconds"])
            if limit.reason is LedgerReason.RUNTIME_LIMIT
            else None
        ),
    )


def _finding(
    rule_id: str,
    message: str,
    path: str,
    *,
    severity: str,
    confidence: float,
    line: int = 1,
) -> Finding:
    return Finding(
        rule_id=rule_id,
        message=message,
        severity=severity,
        confidence=confidence,
        file=path,
        start_line=line,
        category="analysis-evasion",
        tags=["artifact-integrity"],
    )


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Emit classification, Unicode, and analysis-ceiling evasion findings."""
    file_cache = state.get("local_file_cache") or state.get("file_cache") or {}
    budget = _ArtifactIntegrityBudget(state)
    inventory: dict[str, object] = {}
    events: list[InspectionLedgerEvent] = []
    terminal_limit: _ArtifactIntegrityResourceLimitError | None = None

    try:
        budget.check_runtime()
        for item in state.get("artifact_inventory") or []:
            budget.check_runtime()
            if isinstance(item, dict):
                inventory[str(item.get("path", ""))] = item
    except _ArtifactIntegrityResourceLimitError as exc:
        terminal_limit = exc

    components = state.get("components") or []
    for path in components:
        if terminal_limit is None and budget.analyzer_exhausted():
            terminal_limit = _ArtifactIntegrityResourceLimitError(
                LedgerReason.OUTPUT_LIMIT,
                {
                    "observed_findings": len(budget.findings) + 1,
                    "limit_findings": MAX_FINDINGS_PER_ANALYZER,
                },
            )
        if terminal_limit is not None:
            events.append(_partial_limit_event(path, terminal_limit))
            continue

        raw_artifact = inventory.get(path)
        artifact: dict[str, object] = raw_artifact if isinstance(raw_artifact, dict) else {}
        finding_start = len(budget.findings)
        resource_limit: _ArtifactIntegrityResourceLimitError | None = None
        first_obfuscation_line: int | None = None
        try:
            budget.check_runtime()
            if artifact.get("misleading_extension"):
                budget.emit(
                    _finding(
                        "AE2",
                        "Artifact content does not match its filename extension",
                        path,
                        severity="MEDIUM",
                        confidence=0.9,
                    )
                )
            content = file_cache.get(path)
            if content is not None:
                normalized_path = path.lower()
                if len(content) > MAX_PYTHON_AST_SOURCE_CHARS and (
                    normalized_path.endswith(_INSTRUCTION_SUFFIXES)
                    or normalized_path.endswith("skill.md")
                ):
                    budget.emit(
                        _finding(
                            "AE5",
                            "Instruction-capable artifact exceeds whole-file semantic analysis limits",
                            path,
                            severity="HIGH",
                            confidence=1.0,
                        )
                    )
                if artifact.get("content_kind") not in {
                    ContentKind.BINARY,
                    ContentKind.OPAQUE,
                }:
                    format_density, mixed_script, first_nul_line, first_obfuscation_line = (
                        _text_signals(content, budget)
                    )
                    if artifact.get("contains_nul") and first_nul_line is not None:
                        budget.emit(
                            _finding(
                                "AE3",
                                "Text artifact contains embedded NUL bytes",
                                path,
                                severity="HIGH",
                                confidence=0.9,
                                line=first_nul_line,
                            )
                        )
                    if format_density >= 0.01 or mixed_script:
                        budget.emit(
                            _finding(
                                "AE4",
                                "Suspicious Unicode normalization or mixed-script content",
                                path,
                                severity="MEDIUM",
                                confidence=0.8,
                            )
                        )
                    if first_obfuscation_line is not None:
                        budget.emit(
                            _finding(
                                "AE6",
                                "Instruction text uses inter-character separators to evade pattern matching",
                                path,
                                severity="HIGH",
                                confidence=0.9,
                                line=first_obfuscation_line,
                            )
                        )
        except _ArtifactIntegrityResourceLimitError as exc:
            resource_limit = exc

        path_findings = budget.findings[finding_start:]
        emitted_ids = [finding.finding_id for finding in path_findings]
        if resource_limit is not None:
            event = _partial_limit_event(path, resource_limit, emitted_ids)
            terminal_limit = resource_limit
        else:
            event = ledger_event(
                analyzer_id=ANALYZER_ID,
                outcome=LedgerOutcome.COMPLETED,
                phase="artifact",
                path=path,
                emitted_finding_ids=emitted_ids,
            )
        events.append(event)
        if resource_limit is None and first_obfuscation_line is not None:
            events.append(
                ledger_event(
                    outcome=LedgerOutcome.PARTIAL,
                    phase="artifact_interpretation",
                    path=path,
                    start_line=first_obfuscation_line,
                    end_line=first_obfuscation_line,
                    record_type=LedgerRecordType.SYSTEM,
                    reason=LedgerReason.OBFUSCATED_INSTRUCTION_TEXT,
                )
            )
    work_events = (event for event in events if event["record_type"] == LedgerRecordType.WORK_ITEM)
    return {
        "findings": budget.findings,
        "inspection_ledger": events,
        "analyzer_status_events": [analyzer_status_for_events(ANALYZER_ID, work_events)],
    }
