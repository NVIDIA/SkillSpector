# SPDX-License-Identifier: Apache-2.0
"""Bounded, source-owned role annotations for AISOP/AISP (phase-2 prototype).

Pure source inspection: no filesystem access, discovery, execution, findings
mutation, risk scoring or natural-language polarity classification. Offsets and
columns are Unicode character offsets in the original strict UTF-8 text, NOT
UTF-8 byte offsets, decoded JSON-string offsets or normalized scanner offsets.

The surrounding scanner must retain its existing discovery/caching/resource
policy. This module indexes ONE admitted artifact; it is not a replacement for
structured_skill.py, a full protocol validator or an execution-trust check.
"""

from __future__ import annotations

import json
import math
import re
import time
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

# Pinned grammar: AISOP V1.0.0 specification section 5.2. Non-reserved
# function-body fields are execution steps, even when named "hard_deny".
_RESERVED_FUNCTION_KEYS = frozenset(
    {
        "join",
        "map",
        "on_error",
        "retry_policy",
        "context_filter",
        "output_mapping",
        "constraints",
        "execute_mode",
    }
)
_PROTOCOLS = frozenset({"AISOP V1.0.0", "AISP V1.0.0"})
_WHITESPACE = re.compile(r"[ \t\r\n]*")
# Mirrors the pinned common.SourceLocationIndex logical-line convention.
_LOGICAL_LINE_BREAK = re.compile(r"\r\n|[\r\n\v\f\x1c-\x1e\x85\u2028\u2029]")
_RESOURCE_FIELDS = frozenset(
    {"id", "path", "kind", "mode", "scope", "sha256", "when", "requires_tools"}
)
_METADATA_FIELDS = frozenset(
    {
        "protocol",
        "axiom_0",
        "id",
        "name",
        "version",
        "license",
        "summary",
        "description",
        "flow_format",
        "loading_mode",
    }
)
PathPart = str | int
JsonPath = tuple[PathPart, ...]


@dataclass(frozen=True, slots=True)
class SourceLimits:
    """One-document ceilings; the caller must also enforce a scan-wide budget."""

    max_bytes: int = 256 * 1024
    max_depth: int = 64
    max_values: int = 4096
    max_strings: int = 512
    max_pointer_chars: int = 4096
    max_seconds: float = 2.0

    def __post_init__(self) -> None:
        for name, ceiling in (
            ("max_bytes", 256 * 1024),
            ("max_depth", 64),
            ("max_values", 4096),
            ("max_strings", 512),
            ("max_pointer_chars", 4096),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= ceiling:
                raise ValueError(f"{name} must be an integer in [1, {ceiling}]")
        if (
            type(self.max_seconds) not in (float, int)
            or not math.isfinite(self.max_seconds)
            or not 0 < self.max_seconds <= 2.0
        ):
            raise ValueError("max_seconds must be finite and in (0, 2]")


@dataclass(frozen=True, slots=True)
class SourceString:
    """A string VALUE's original-text span, excluding its surrounding quotes."""

    start: int
    end: int
    pointer: str
    text_role: str


@dataclass(frozen=True, slots=True)
class RoleAnnotation:
    """Report-only metadata; no risk or detection-confidence inference."""

    mapping_status: Literal["exact", "unknown", "unavailable"]
    reason: str
    text_role: str = "unknown"
    structured_source: str | None = None
    source_span: tuple[int, int] | None = None
    content_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "mapping_status": self.mapping_status,
            "reason": self.reason,
            "text_role": self.text_role,
            # A structure alone establishes neither benign nor dangerous intent.
            "risk_polarity": "unknown",
            # Deliberately uncalibrated, not a fabricated probability such as .94.
            "role_confidence": None,
            "structured_source": self.structured_source,
            "source_span": list(self.source_span) if self.source_span is not None else None,
            "content_sha256": self.content_sha256,
            "coordinate_space": "original_unicode",
        }


@dataclass(frozen=True, slots=True)
class StructuredSourceMap:
    """Complete index or an explicit inability to map; never a partial success."""

    status: Literal["ready", "unavailable"]
    reason: str
    content_sha256: str | None = None
    protocol: str | None = None
    strings: tuple[SourceString, ...] = ()
    line_starts: tuple[int, ...] = ()
    line_ends: tuple[int, ...] = ()
    text_length: int = 0

    def locate(
        self,
        start: int,
        end: int,
        *,
        content_sha256: str,
        coordinate_space: str,
    ) -> RoleAnnotation:
        """Attribute a nonempty original-source match to ONE complete string value.

        No substring search, line-only guessing or decoded/normalized-offset
        fallback. The caller must establish the source snapshot and coordinates.
        Matching digests do not attest the execution or trust the source.
        """
        if self.status != "ready":
            return RoleAnnotation("unavailable", self.reason)
        if content_sha256 != self.content_sha256:
            return RoleAnnotation("unknown", "source_mismatch")
        if coordinate_space != "original_unicode":
            return RoleAnnotation("unknown", "unbound_coordinates")
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= self.text_length
        ):
            return RoleAnnotation("unknown", "invalid_span")
        # Strings are disjoint and sorted. At most 512 records per artifact.
        starts = [item.start for item in self.strings]
        pos = bisect_right(starts, start) - 1
        if pos < 0 or end > self.strings[pos].end:
            return RoleAnnotation("unknown", "not_one_string_value")
        item = self.strings[pos]
        return RoleAnnotation(
            "exact",
            "source_string_owned" if item.text_role != "unknown" else "unsupported_source_field",
            item.text_role,
            item.pointer,
            (start, end),
            self.content_sha256,
        )

    def locate_lines(
        self,
        start_line: int,
        end_line: int | None,
        start_column: int | None,
        end_column: int | None,
        *,
        content_sha256: str,
        coordinate_space: str,
    ) -> RoleAnnotation:
        """Use 1-based lines / 0-based end-exclusive character columns."""
        if self.status != "ready":
            return RoleAnnotation("unavailable", self.reason)
        if start_column is None or end_column is None:
            return RoleAnnotation("unknown", "missing_columns")
        if end_line is None:
            end_line = start_line
        values = (start_line, end_line, start_column, end_column)
        if any(type(value) is not int for value in values):
            return RoleAnnotation("unknown", "invalid_location")
        if not 1 <= start_line <= end_line <= len(self.line_starts):
            return RoleAnnotation("unknown", "invalid_location")
        for line, column in ((start_line, start_column), (end_line, end_column)):
            width = self.line_ends[line - 1] - self.line_starts[line - 1]
            if not 0 <= column <= width:
                return RoleAnnotation("unknown", "invalid_location")
        return self.locate(
            self.line_starts[start_line - 1] + start_column,
            self.line_starts[end_line - 1] + end_column,
            content_sha256=content_sha256,
            coordinate_space=coordinate_space,
        )


class _CannotIndexError(ValueError):
    """Content-free limitation code; never echo attacker-controlled payloads."""


def _pointer(path: JsonPath) -> str:
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in path)


def _reject_constant(_text: str) -> None:
    raise _CannotIndexError("non_finite_number")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise _CannotIndexError("non_finite_number")
    return value


class _SourceParser:
    """Bounded structural walk; stdlib decoder owns string/scalar semantics.

    Bounds are checked BEFORE nesting descent. Duplicate decoded object keys,
    lone surrogates, nonfinite numbers and trailing data invalidate the ENTIRE
    annotation index. No previously collected role survives a parse failure.
    """

    def __init__(
        self,
        text: str,
        limits: SourceLimits,
        clock: Callable[[], float],
        deadline: float,
        check_runtime: Callable[[], None] | None,
    ) -> None:
        self.text = text
        self.limits = limits
        self.clock = clock
        self.deadline = deadline
        self.check_runtime = check_runtime
        self.position = 0
        self.values = 0
        self.strings: list[tuple[int, int, JsonPath]] = []
        self.decoder = json.JSONDecoder(parse_constant=_reject_constant, parse_float=_finite_float)

    def check(self) -> None:
        if self.check_runtime is not None:
            # The host owns cancellation. Do not swallow its exception as JSON.
            self.check_runtime()
        if self.clock() >= self.deadline:
            raise _CannotIndexError("runtime_limit")

    def whitespace(self) -> None:
        match = _WHITESPACE.match(self.text, self.position)
        assert match is not None
        self.position = match.end()

    def scalar(self) -> object:
        try:
            value, self.position = self.decoder.raw_decode(self.text, self.position)
            return value
        except _CannotIndexError:
            raise
        except json.JSONDecodeError:
            raise _CannotIndexError("invalid_json") from None
        except (ValueError, OverflowError):
            raise _CannotIndexError("scalar_conversion") from None

    def string(self) -> str:
        if self.position >= len(self.text) or self.text[self.position] != '"':
            raise _CannotIndexError("invalid_json")
        value = self.scalar()
        assert isinstance(value, str)
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise _CannotIndexError("invalid_unicode")
        self.check()
        return value

    def value(self, path: JsonPath, depth: int) -> object:
        self.check()
        if depth > self.limits.max_depth:
            raise _CannotIndexError("depth_limit")
        self.values += 1
        if self.values > self.limits.max_values:
            raise _CannotIndexError("value_limit")
        self.whitespace()
        if self.position >= len(self.text):
            raise _CannotIndexError("invalid_json")
        char = self.text[self.position]
        if char == "{":
            return self.object(path, depth)
        if char == "[":
            return self.array(path, depth)
        if char == '"':
            start = self.position + 1
            value = self.string()
            if len(self.strings) >= self.limits.max_strings:
                raise _CannotIndexError("string_limit")
            if len(_pointer(path)) > self.limits.max_pointer_chars:
                raise _CannotIndexError("pointer_limit")
            self.strings.append((start, self.position - 1, path))
            return value
        value = self.scalar()
        self.check()
        return value

    def object(self, path: JsonPath, depth: int) -> dict[str, object]:
        self.position += 1
        self.whitespace()
        result: dict[str, object] = {}
        if self.position < len(self.text) and self.text[self.position] == "}":
            self.position += 1
            return result
        while True:
            self.check()
            key = self.string()
            if key in result:
                raise _CannotIndexError("duplicate_key")
            # Limit parent/key expansion before retaining a long repeated path.
            next_path = (*path, key)
            if len(_pointer(next_path)) > self.limits.max_pointer_chars:
                raise _CannotIndexError("pointer_limit")
            self.whitespace()
            if self.position >= len(self.text) or self.text[self.position] != ":":
                raise _CannotIndexError("invalid_json")
            self.position += 1
            result[key] = self.value(next_path, depth + 1)
            self.whitespace()
            if self.position >= len(self.text):
                raise _CannotIndexError("invalid_json")
            char = self.text[self.position]
            self.position += 1
            if char == "}":
                return result
            if char != ",":
                raise _CannotIndexError("invalid_json")
            self.whitespace()

    def array(self, path: JsonPath, depth: int) -> list[object]:
        self.position += 1
        self.whitespace()
        result: list[object] = []
        if self.position < len(self.text) and self.text[self.position] == "]":
            self.position += 1
            return result
        while True:
            result.append(self.value((*path, len(result)), depth + 1))
            self.whitespace()
            if self.position >= len(self.text):
                raise _CannotIndexError("invalid_json")
            char = self.text[self.position]
            self.position += 1
            if char == "]":
                return result
            if char != ",":
                raise _CannotIndexError("invalid_json")


def _supported_envelope(payload: object) -> str | None:
    """Recognize the pinned layout, not full AISP/AISOP conformance or safety.

    Phase-1's broader discovery behavior is intentionally NOT changed here.
    Unknown keys are retained and may remain unclassified. Future versions
    receive no v1 role semantics automatically.
    """
    if not isinstance(payload, list) or len(payload) != 2:
        return None
    system, user = payload
    if not isinstance(system, dict) or not isinstance(user, dict):
        return None
    if system.get("role") != "system" or user.get("role") != "user":
        return None
    metadata, content = system.get("content"), user.get("content")
    if not isinstance(metadata, dict) or not isinstance(content, dict):
        return None
    protocol = metadata.get("protocol")
    if not isinstance(protocol, str) or protocol not in _PROTOCOLS:
        return None
    if not isinstance(content.get("instruction"), str):
        return None
    flow, functions = content.get("aisop"), content.get("functions")
    if (
        not isinstance(flow, dict)
        or not isinstance(flow.get("main"), (str, dict))
        or not isinstance(functions, dict)
        or not all(isinstance(value, dict) for value in functions.values())
    ):
        return None
    if protocol == "AISP V1.0.0" and not isinstance(content.get("aisp_contract"), dict):
        return None
    return protocol


def _role(path: JsonPath, protocol: str) -> str:
    """Roles describe structure only: a constraint is NOT an enforced denial."""
    if path[:2] == (0, "content"):
        if len(path) == 3 and path[2] in _METADATA_FIELDS:
            return "metadata"
        if path == (0, "content", "system_prompt"):
            return "prompt"
        if len(path) == 4 and path[2] == "tools" and type(path[3]) is int:
            return "tool_declaration"
        return "unknown"
    if path[:2] != (1, "content") or len(path) < 3:
        return "unknown"
    if path == (1, "content", "instruction"):
        return "instruction"
    if path == (1, "content", "user_input"):
        return "input"
    if len(path) >= 4 and path[2] == "aisop":
        return "workflow_topology"
    if len(path) >= 5 and path[2] == "functions" and isinstance(path[3], str):
        field = path[4]
        if field == "constraints":
            if len(path) == 5 or len(path) == 6 and type(path[5]) is int:
                return "constraint"
            return "unknown"
        if field in _RESERVED_FUNCTION_KEYS:
            return "runtime_configuration"
        if len(path) == 5:
            return "executable_step"
        return "unknown"
    if path[:3] != (1, "content", "aisp_contract") or len(path) < 4:
        return "unknown"
    # Contract paths have meaning only for an AISP v1 layout.
    if protocol != "AISP V1.0.0":
        return "unknown"
    if path[3] == "resources" and len(path) >= 6 and type(path[4]) is int:
        if path[5] in _RESOURCE_FIELDS:
            if len(path) == 6 or (
                path[5] == "requires_tools" and len(path) == 7 and type(path[6]) is int
            ):
                return "resource_declaration"
        return "unknown"
    if path[3] == "non_negotiable" and len(path) == 6 and type(path[4]) is int:
        return {"rule": "control_declaration", "enforced_by": "control_binding"}.get(
            path[5], "unknown"
        )
    if path[3] == "invocation":
        if len(path) == 5 and path[4] == "mode":
            return "invocation_metadata"
        if (
            len(path) == 6
            and path[4] in {"when_to_use", "when_not_to_use"}
            and type(path[5]) is int
        ):
            return "invocation_metadata"
    if path in {
        (1, "content", "aisp_contract", "profile"),
        (1, "content", "aisp_contract", "risk_level"),
    }:
        return "metadata"
    return "unknown"


def index_structured_source(
    raw: bytes,
    *,
    limits: SourceLimits | None = None,
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    check_runtime: Callable[[], None] | None = None,
) -> StructuredSourceMap:
    """Index one admitted source snapshot; any failure returns no annotations.

    Strict decoding deliberately differs from phase-1's summary-only tolerant
    decoding. It must not change discovery or suppress the original findings.
    A caller-supplied deadline may only tighten the local two-second ceiling.
    """
    limits = limits or SourceLimits()
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("deadline must be a finite monotonic timestamp")
    started = clock()
    stop = started + limits.max_seconds
    if deadline is not None:
        stop = min(stop, deadline)
    if type(raw) is not bytes:
        return StructuredSourceMap("unavailable", "expected_bytes")
    if len(raw) > limits.max_bytes:
        return StructuredSourceMap("unavailable", "size_limit")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return StructuredSourceMap("unavailable", "invalid_utf8")
    digest = "sha-256:" + sha256(raw).hexdigest()
    parser = _SourceParser(text, limits, clock, stop, check_runtime)
    try:
        payload = parser.value((), 0)
        parser.whitespace()
        if parser.position != len(text):
            raise _CannotIndexError("trailing_data")
        parser.check()
    except _CannotIndexError as error:
        return StructuredSourceMap("unavailable", str(error), digest)
    except json.JSONDecodeError:
        return StructuredSourceMap("unavailable", "invalid_json", digest)
    except RecursionError:
        return StructuredSourceMap("unavailable", "depth_limit", digest)
    protocol = _supported_envelope(payload)
    if protocol is None:
        return StructuredSourceMap("unavailable", "unsupported_layout", digest)
    strings = tuple(
        SourceString(start, end, _pointer(path), _role(path, protocol))
        for start, end, path in parser.strings
    )
    line_starts = [0]
    line_ends = []
    # Use the scanner's logical line convention, including U+2028/U+2029.
    for match in _LOGICAL_LINE_BREAK.finditer(text):
        line_ends.append(match.start())
        line_starts.append(match.end())
    line_ends.append(len(text))
    try:
        parser.check()
    except _CannotIndexError as error:
        return StructuredSourceMap("unavailable", str(error), digest)
    return StructuredSourceMap(
        "ready",
        "source_indexed",
        digest,
        protocol,
        strings,
        tuple(line_starts),
        tuple(line_ends),
        len(text),
    )
