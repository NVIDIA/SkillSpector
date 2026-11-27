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

"""Adaptive structural de-contextualization for Markdown skill documents.

Splits pure semantic/documentation files along Markdown block boundaries so
concealed channels (HTML comments, hidden Unicode tags, risky frontmatter) and
long sections can be evaluated with minimal narrative dilution while keeping
source line numbers strictly aligned to the original file.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Final

from skillspector.resource_limits import bounded_capacity

DEFAULT_DECONTEXT_MAX_CHARS: Final[int] = 4_000
MAX_DECONTEXT_MAX_CHARS: Final[int] = 64_000
MIN_DECONTEXT_MAX_CHARS: Final[int] = 400
MAX_SLICER_INPUT_CHARS: Final[int] = 262_144
SEMANTIC_MARKDOWN_EXTS: Final[frozenset[str]] = frozenset(
    {".md", ".mdc", ".markdown", ".txt", ".rst"}
)

_TRUTHY_ENV_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})

# Keep identical to whitespace_padding.ZERO_WIDTH_CHARS without importing
# skillspector.nodes.analyzers (which eagerly loads all analyzer nodes).
_ZERO_WIDTH_CHARS: Final[frozenset[str]] = frozenset("\u200b\u200c\u200d\u2060\ufeff")
_BIDI_CONTROL_CHARS: Final[frozenset[str]] = frozenset(
    {
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }
)

_ATX_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})(?:\s+\S|\s*$)")
_SETEXT_UNDERLINE_RE = re.compile(r"^\s{0,3}(=+|-+)\s*$")
_THEMATIC_BREAK_RE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_FENCE_OPEN_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_LIST_ITEM_RE = re.compile(r"^(\s*)(?:[-*+]|\d{1,3}[.)])\s+")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_HTML_COMMENT_OPEN_RE = re.compile(r"^\s{0,3}<!--")
_MALFORMED_COMMENT_CLOSE_RE = re.compile(r"^\s*-*>\s*$")

_URL_RE = re.compile(r"https?://[^\s`\"')<>]+", re.IGNORECASE)
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b")
_SENSITIVE_PATH_RE = re.compile(
    r"(?:~/\.[^\s`\"')]+|\b(?:\.env(?:\.[A-Za-z0-9_.-]+)?|\.ssh(?:/[^\s`\"')]+)?|"
    r"\.aws(?:/[^\s`\"')]+)?|id_rsa|id_ed25519|authorized_keys|MEMORY\.md|SOUL\.md|"
    r"/etc/passwd|/tmp/[A-Za-z0-9_.-]+|"
    r"[A-Z][A-Z0-9_]{2,}_(?:API_KEY|SECRET|SECRET_KEY|TOKEN|PASSWORD))\b)"
)
_EXEC_TOKEN_RE = re.compile(
    r"(?:!\s*`[^`]{1,100}`|\|\s*(?:ba|z)?sh\b|"
    r"\b(?:curl|wget|base64|powershell|pwsh|iex|crontab|launchctl|systemctl)\b|"
    r"^\s*(?:hooks|command|UserPromptSubmit|PreToolUse|PostToolUse)\s*:|"
    r"extract\s+(?:with|using)\s+pass\b)",
    re.IGNORECASE | re.MULTILINE,
)
_BACKTICK_IDENT_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_.-]{1,48})`")
_SUSPICIOUS_IDENT_HINT_RE = re.compile(
    r"(?:^_|ctx|diag|bundle|state|token|secret|key|env|telemetry|header|X-[A-Za-z0-9-]+)",
    re.IGNORECASE,
)


def is_decontext_slicing_enabled() -> bool:
    """Return True when opt-in de-contextualized SSD slicing is enabled."""
    raw = os.environ.get("SKILLSPECTOR_DECONTEXT_SLICING", "").strip().lower()
    return raw in _TRUTHY_ENV_VALUES


def resolve_decontext_max_chars() -> int:
    """Resolve bounded per-slice character budget from ``SKILLSPECTOR_DECONTEXT_MAX_CHARS``."""
    configured = bounded_capacity(
        "SKILLSPECTOR_DECONTEXT_MAX_CHARS",
        DEFAULT_DECONTEXT_MAX_CHARS,
        MAX_DECONTEXT_MAX_CHARS,
    )
    return max(MIN_DECONTEXT_MAX_CHARS, configured)


def is_semantic_markdown_path(path: str) -> bool:
    """Return True iff *path* names a Markdown or plain-text skill document."""
    normalized = path.replace("\\", "/").strip()
    if not normalized:
        return False
    pure = PurePosixPath(normalized)
    if pure.name.lower() == "skill.md":
        return True
    return pure.suffix.lower() in SEMANTIC_MARKDOWN_EXTS


@dataclass(frozen=True, slots=True)
class CapabilityBaseline:
    """Minimal declared capability baseline anchored from manifest or SKILL.md."""

    name: str = ""
    description: str = ""
    source_file: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.name.strip() or self.description.strip())

    def format_anchor(self) -> str:
        """Render the declared baseline as a single-line untrusted anchor."""
        if self.is_empty:
            return "(no declared baseline)"
        parts = []
        if self.name:
            parts.append(f"name={self.name}")
        if self.description:
            parts.append(f"description={self.description}")
        return "; ".join(parts)


@dataclass(frozen=True, slots=True)
class SemanticSlice:
    """A verbatim structural slice of a semantic document with 1-based line bounds."""

    file_path: str
    slice_index: int
    line_start: int
    line_end: int
    kind: str
    text: str
    heading: str = ""
    hidden_decoded_text: str = ""
    objective_literals: dict[str, list[str]] = field(default_factory=dict)
    is_isolated: bool = False


@dataclass(frozen=True, slots=True)
class _Block:
    kind: str
    text: str
    line_start: int
    line_end: int
    heading_level: int = 0


def extract_hidden_unicode_text(text: str) -> str:
    """Decode Unicode Tag steganography or strip zero-width/Bidi obfuscation."""
    if not text or text.isascii():
        return ""

    recovered_parts: list[str] = []
    tag_chars: list[str] = []
    for ch in text:
        cp = ord(ch)
        if 0xE0020 <= cp <= 0xE007E:
            tag_chars.append(chr(cp - 0xE0000))
        elif cp == 0xE000A:
            tag_chars.append("\n")
        elif cp == 0xE0009:
            tag_chars.append("\t")
    decoded_tags = "".join(tag_chars).strip()
    if len(decoded_tags) >= 4:
        recovered_parts.append(f"[Unicode Tag steganography]: {decoded_tags}")

    bidi_count = sum(1 for ch in text if ch in _BIDI_CONTROL_CHARS)
    zw_sus = 0
    for idx, ch in enumerate(text):
        if ch not in _ZERO_WIDTH_CHARS:
            continue
        # Ignore ZWJ inside emoji sequences.
        if ch == "\u200d" and 0 < idx < len(text) - 1:
            if ord(text[idx - 1]) >= 0x2600 and ord(text[idx + 1]) >= 0x2600:
                continue
        zw_sus += 1

    if bidi_count >= 1 or zw_sus >= 2:
        stripped = "".join(
            ch
            for ch in text
            if ch not in _BIDI_CONTROL_CHARS
            and ch not in _ZERO_WIDTH_CHARS
            and not (0xE0000 <= ord(ch) <= 0xE007F)
        ).strip()
        if stripped and stripped != decoded_tags:
            recovered_parts.append(f"[Zero-width/Bidi stripped text]: {stripped}")

    return "\n".join(recovered_parts)


def extract_objective_literals(
    text: str,
    *,
    kind: str = "",
    hidden_decoded_text: str = "",
) -> dict[str, list[str]]:
    """Deterministically extract concrete literal tokens without prose framing."""
    combined = f"{text}\n{hidden_decoded_text}" if hidden_decoded_text else text

    urls = list(dict.fromkeys(_URL_RE.findall(combined)))[:6]
    ips = [ip for ip in dict.fromkeys(_IP_RE.findall(combined)) if not ip.startswith("127.0.0.")][
        :4
    ]
    sens = list(dict.fromkeys(_SENSITIVE_PATH_RE.findall(combined)))[:8]
    execs = list(dict.fromkeys(m.strip() for m in _EXEC_TOKEN_RE.findall(combined) if m.strip()))[
        :6
    ]

    idents: list[str] = []
    for raw_id in _BACKTICK_IDENT_RE.findall(combined):
        if _SUSPICIOUS_IDENT_HINT_RE.search(raw_id) and raw_id not in idents:
            idents.append(raw_id)
        if len(idents) >= 6:
            break

    concealed: list[str] = []
    if hidden_decoded_text:
        concealed.append("hidden_unicode:" + re.sub(r"\s+", " ", hidden_decoded_text).strip()[:160])
    if kind == "html_comment":
        concealed.append("html_comment:" + re.sub(r"\s+", " ", text).strip()[:160])

    return {
        "urls": urls,
        "ips": ips,
        "sensitive_tokens": sens,
        "exec_tokens": execs,
        "state_idents": idents,
        "concealed": concealed,
    }


def _sanitize_baseline_field(value: str, max_chars: int) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", value or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_chars:
        return cleaned[: max_chars - 3].rstrip() + "..."
    return cleaned


def _parse_frontmatter_metadata(frontmatter_text: str) -> tuple[str, str]:
    lines = frontmatter_text.splitlines()
    if lines and lines[0].strip() == "---":
        lines = lines[1:]
    if lines and lines[-1].strip() in {"---", "..."}:
        lines = lines[:-1]

    fields: dict[str, str] = {}
    i = 0
    while i < len(lines):
        match = re.match(r"^([A-Za-z0-9_-]+)\s*:\s*(.*)$", lines[i])
        if not match:
            i += 1
            continue
        key = match.group(1).strip().lower()
        val = match.group(2).strip()
        if val in {"|", "|-", "|+", ">", ">-", ">+"}:
            i += 1
            block_lines: list[str] = []
            while i < len(lines) and (lines[i].startswith((" ", "\t")) or not lines[i].strip()):
                block_lines.append(lines[i].strip())
                i += 1
            val = " ".join(part for part in block_lines if part)
        else:
            if len(val) >= 2 and (
                (val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")
            ):
                val = val[1:-1]
            i += 1
        if key in {"name", "description"} and val:
            fields[key] = val
    return fields.get("name", ""), fields.get("description", "")


def _find_frontmatter_end(lines: list[str]) -> int | None:
    if not lines or lines[0].strip() != "---":
        return None
    for idx in range(1, len(lines)):
        if lines[idx].strip() in {"---", "..."}:
            return idx
    return None


def _parse_blocks(lines: list[str]) -> list[_Block]:
    """Parse text lines into structural blocks with 1-based line coordinates."""
    blocks: list[_Block] = []
    total = len(lines)
    idx = 0

    fm_end = _find_frontmatter_end(lines)
    if fm_end is not None:
        blocks.append(
            _Block(
                kind="frontmatter",
                text="\n".join(lines[: fm_end + 1]),
                line_start=1,
                line_end=fm_end + 1,
            )
        )
        idx = fm_end + 1

    while idx < total:
        line = lines[idx]
        if not line.strip():
            idx += 1
            continue

        fence_match = _FENCE_OPEN_RE.match(line)
        if fence_match:
            fence = fence_match.group(1)
            close_re = re.compile(rf"^\s{{0,3}}{re.escape(fence[0])}{{{len(fence)},}}\s*$")
            start = idx
            idx += 1
            while idx < total and not close_re.match(lines[idx]):
                idx += 1
            if idx < total:
                idx += 1
            blocks.append(
                _Block(
                    kind="fenced_block",
                    text="\n".join(lines[start:idx]),
                    line_start=start + 1,
                    line_end=idx,
                )
            )
            continue

        atx_match = _ATX_HEADING_RE.match(line)
        if atx_match:
            blocks.append(
                _Block(
                    kind="heading",
                    text=line,
                    line_start=idx + 1,
                    line_end=idx + 1,
                    heading_level=len(atx_match.group(1)),
                )
            )
            idx += 1
            continue

        if idx + 1 < total and _SETEXT_UNDERLINE_RE.match(lines[idx + 1]):
            underline_char = lines[idx + 1].lstrip()[0]
            blocks.append(
                _Block(
                    kind="heading",
                    text="\n".join(lines[idx : idx + 2]),
                    line_start=idx + 1,
                    line_end=idx + 2,
                    heading_level=1 if underline_char == "=" else 2,
                )
            )
            idx += 2
            continue

        if _HTML_COMMENT_OPEN_RE.match(line):
            start = idx
            if "-->" not in line and not _MALFORMED_COMMENT_CLOSE_RE.match(line):
                idx += 1
                while idx < total:
                    if "-->" in lines[idx] or _MALFORMED_COMMENT_CLOSE_RE.match(lines[idx]):
                        idx += 1
                        break
                    idx += 1
            else:
                idx += 1
            blocks.append(
                _Block(
                    kind="html_comment",
                    text="\n".join(lines[start:idx]),
                    line_start=start + 1,
                    line_end=idx,
                )
            )
            continue

        if _THEMATIC_BREAK_RE.match(line):
            blocks.append(
                _Block(kind="thematic_break", text=line, line_start=idx + 1, line_end=idx + 1)
            )
            idx += 1
            continue

        if re.match(r"^\s{0,3}>", line):
            start = idx
            idx += 1
            while idx < total:
                cur = lines[idx]
                if not cur.strip():
                    break
                if (
                    _FENCE_OPEN_RE.match(cur)
                    or _ATX_HEADING_RE.match(cur)
                    or _THEMATIC_BREAK_RE.match(cur)
                ):
                    break
                if re.match(r"^\s{0,3}>", cur) or not _LIST_ITEM_RE.match(cur):
                    idx += 1
                else:
                    break
            blocks.append(
                _Block(
                    kind="blockquote",
                    text="\n".join(lines[start:idx]),
                    line_start=start + 1,
                    line_end=idx,
                )
            )
            continue

        if line.lstrip().startswith("|") or (
            idx + 1 < total and "|" in line and _TABLE_SEP_RE.match(lines[idx + 1])
        ):
            start = idx
            idx += 1
            while idx < total and lines[idx].strip() and "|" in lines[idx]:
                idx += 1
            blocks.append(
                _Block(
                    kind="table",
                    text="\n".join(lines[start:idx]),
                    line_start=start + 1,
                    line_end=idx,
                )
            )
            continue

        list_match = _LIST_ITEM_RE.match(line)
        if list_match:
            base_indent = len(list_match.group(1))
            start = idx
            idx += 1
            while idx < total:
                cur = lines[idx]
                if not cur.strip():
                    break
                if (
                    _FENCE_OPEN_RE.match(cur)
                    or _ATX_HEADING_RE.match(cur)
                    or _THEMATIC_BREAK_RE.match(cur)
                ):
                    break
                next_list = _LIST_ITEM_RE.match(cur)
                if next_list and len(next_list.group(1)) <= base_indent:
                    break
                idx += 1
            blocks.append(
                _Block(
                    kind="list_item",
                    text="\n".join(lines[start:idx]),
                    line_start=start + 1,
                    line_end=idx,
                )
            )
            continue

        start = idx
        idx += 1
        while idx < total:
            cur = lines[idx]
            if not cur.strip():
                break
            if (
                _FENCE_OPEN_RE.match(cur)
                or _ATX_HEADING_RE.match(cur)
                or _THEMATIC_BREAK_RE.match(cur)
                or _LIST_ITEM_RE.match(cur)
                or re.match(r"^\s{0,3}>", cur)
            ):
                break
            if idx + 1 < total and _SETEXT_UNDERLINE_RE.match(lines[idx + 1]):
                break
            idx += 1
        blocks.append(
            _Block(
                kind="paragraph",
                text="\n".join(lines[start:idx]),
                line_start=start + 1,
                line_end=idx,
            )
        )

    return blocks


def baseline_from_manifest(
    manifest: Mapping[str, object] | None,
    *,
    fallback_text: str = "",
    fallback_path: str = "SKILL.md",
) -> CapabilityBaseline:
    """Build a sanitized capability baseline from manifest fields or fallback text."""
    name = ""
    description = ""
    source_file = fallback_path

    if manifest:
        raw_name = manifest.get("name")
        raw_desc = manifest.get("description")
        if isinstance(raw_name, str):
            name = raw_name
        if isinstance(raw_desc, str):
            description = raw_desc

    if (not name or not description) and fallback_text and fallback_text.strip():
        blocks = _parse_blocks(fallback_text.splitlines()[:400])
        for block in blocks:
            if block.kind == "frontmatter":
                fm_name, fm_desc = _parse_frontmatter_metadata(block.text)
                name = name or fm_name
                description = description or fm_desc
                break
        if not name:
            for block in blocks:
                if block.kind == "heading":
                    name = re.sub(r"^\s{0,3}#{1,6}\s*", "", block.text.splitlines()[0]).strip()
                    break
        if not description:
            for block in blocks:
                if block.kind == "paragraph":
                    description = block.text
                    break

    s_name = _sanitize_baseline_field(name, 80)
    s_desc = _sanitize_baseline_field(description, 300)
    if not s_name and not s_desc:
        return CapabilityBaseline()
    return CapabilityBaseline(name=s_name, description=s_desc, source_file=source_file)


def _is_isolated_block(block: _Block, hidden_decoded: str, literals: dict[str, list[str]]) -> bool:
    if block.kind == "html_comment" or bool(hidden_decoded):
        return True
    if block.kind == "frontmatter":
        return any(literals[key] for key in ("exec_tokens", "urls", "ips", "sensitive_tokens"))
    return False


def _split_line_range_by_chars(
    raw_lines: list[str],
    line_start: int,
    line_end: int,
    max_chars: int,
) -> list[tuple[int, int]]:
    """Split an inclusive 1-based line span into contiguous chunks ``<= max_chars``."""
    ranges: list[tuple[int, int]] = []
    cur_start = line_start
    cur_len = 0
    for line_no in range(line_start, line_end + 1):
        line_len = len(raw_lines[line_no - 1])
        added = line_len + (1 if line_no > cur_start else 0)
        if line_no > cur_start and cur_len + added > max_chars:
            ranges.append((cur_start, line_no - 1))
            cur_start = line_no
            cur_len = line_len
        else:
            cur_len += added
    ranges.append((cur_start, line_end))
    return ranges


def slice_markdown_adaptive(
    text: str,
    file_path: str,
    *,
    max_slice_chars: int = DEFAULT_DECONTEXT_MAX_CHARS,
) -> list[SemanticSlice]:
    """Adaptively slice Markdown into verbatim, line-aligned semantic slices.

    - Concealed or risky structural blocks (HTML comments, hidden Unicode
      steganography, and frontmatter with URL/exec/sensitive-path cues) are
      always isolated into dedicated slices regardless of file size.
    - Small non-isolated files (``<= max_slice_chars``) remain a single whole
      document slice so per-file call count and multi-step narrative coherence
      are preserved.
    - Oversized files are packed across contiguous line spans up to
      ``max_slice_chars`` while respecting major heading boundaries.
    """
    if not text or not text.strip():
        return []
    if len(text) > MAX_SLICER_INPUT_CHARS:
        return []

    raw_lines = text.splitlines()
    blocks = _parse_blocks(raw_lines)
    if not blocks:
        return []

    first_heading = ""
    annotated: list[tuple[_Block, str, dict[str, list[str]], bool]] = []
    any_isolated = False
    for block in blocks:
        if not first_heading and block.kind == "heading":
            first_heading = block.text.strip()
        hidden = extract_hidden_unicode_text(block.text)
        lits = extract_objective_literals(
            block.text,
            kind=block.kind,
            hidden_decoded_text=hidden,
        )
        isolated = _is_isolated_block(block, hidden, lits)
        if isolated:
            any_isolated = True
        annotated.append((block, hidden, lits, isolated))

    if not any_isolated and len(text) <= max_slice_chars:
        return [
            SemanticSlice(
                file_path=file_path,
                slice_index=1,
                line_start=1,
                line_end=len(raw_lines),
                kind="document",
                text=text,
                heading=first_heading,
                hidden_decoded_text="",
                objective_literals=extract_objective_literals(text),
                is_isolated=False,
            )
        ]

    slices: list[SemanticSlice] = []
    buf_start: int | None = None
    buf_end: int | None = None
    buf_heading = ""
    buf_kind = ""
    current_heading = ""

    def emit_span(
        start_line: int,
        end_line: int,
        kind: str,
        heading: str,
        *,
        is_isolated: bool,
        hidden_override: str | None = None,
        literals_override: dict[str, list[str]] | None = None,
    ) -> None:
        verbatim = "\n".join(raw_lines[start_line - 1 : end_line])
        if not verbatim.strip():
            return
        hidden = (
            extract_hidden_unicode_text(verbatim) if hidden_override is None else hidden_override
        )
        literals = (
            extract_objective_literals(verbatim, kind=kind, hidden_decoded_text=hidden)
            if literals_override is None
            else literals_override
        )
        slices.append(
            SemanticSlice(
                file_path=file_path,
                slice_index=len(slices) + 1,
                line_start=start_line,
                line_end=end_line,
                kind=kind,
                text=verbatim,
                heading=heading,
                hidden_decoded_text=hidden,
                objective_literals=literals,
                is_isolated=is_isolated,
            )
        )

    def flush_buf() -> None:
        nonlocal buf_start, buf_end, buf_heading, buf_kind
        if buf_start is not None and buf_end is not None:
            emit_span(
                buf_start,
                buf_end,
                buf_kind,
                buf_heading,
                is_isolated=False,
                hidden_override="",
            )
        buf_start = None
        buf_end = None
        buf_heading = ""
        buf_kind = ""

    for block, hidden, lits, isolated in annotated:
        if block.kind == "thematic_break":
            flush_buf()
            continue

        if isolated:
            flush_buf()
            if len(block.text) <= max_slice_chars:
                emit_span(
                    block.line_start,
                    block.line_end,
                    block.kind,
                    current_heading,
                    is_isolated=True,
                    hidden_override=hidden,
                    literals_override=lits,
                )
            else:
                for sub_start, sub_end in _split_line_range_by_chars(
                    raw_lines, block.line_start, block.line_end, max_slice_chars
                ):
                    emit_span(
                        sub_start,
                        sub_end,
                        block.kind,
                        current_heading,
                        is_isolated=True,
                    )
            continue

        if len(block.text) > max_slice_chars:
            flush_buf()
            if block.kind == "heading":
                current_heading = block.text.strip()
            for sub_start, sub_end in _split_line_range_by_chars(
                raw_lines, block.line_start, block.line_end, max_slice_chars
            ):
                emit_span(
                    sub_start,
                    sub_end,
                    block.kind,
                    current_heading,
                    is_isolated=False,
                    hidden_override="",
                )
            continue

        if buf_start is None:
            if block.kind == "heading":
                current_heading = block.text.strip()
            buf_start = block.line_start
            buf_end = block.line_end
            buf_heading = current_heading
            buf_kind = block.kind
            continue

        candidate_span = "\n".join(raw_lines[buf_start - 1 : block.line_end])
        current_len = len("\n".join(raw_lines[buf_start - 1 : buf_end]))
        major_heading_break = (
            block.kind == "heading"
            and 1 <= block.heading_level <= 2
            and current_len >= max_slice_chars // 2
        )
        if len(candidate_span) > max_slice_chars or major_heading_break:
            flush_buf()
            if block.kind == "heading":
                current_heading = block.text.strip()
            buf_start = block.line_start
            buf_end = block.line_end
            buf_heading = current_heading
            buf_kind = block.kind
        else:
            if block.kind == "heading":
                current_heading = block.text.strip()
                if not buf_heading:
                    buf_heading = current_heading
            buf_end = block.line_end
            if buf_kind != block.kind:
                buf_kind = "section_group"

    flush_buf()
    return slices
