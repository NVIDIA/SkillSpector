# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove a bounded SVG drawing subset without rendering or fetching resources.

Unsupported XML, CSS and SVG features retain executable classification. Text
security checks still inspect the original source, including titles and text;
this proof does not establish the meaning of the rendered drawing.
"""

from __future__ import annotations

import math
import re
import time
from xml.parsers import expat

_SVG = "http://www.w3.org/2000/svg"
_XML = "http://www.w3.org/XML/1998/namespace"
_XLINK = "http://www.w3.org/1999/xlink"
_MAX_BYTES = 1_000_000
_MAX_NODES = 4_096
_MAX_DEPTH = 64
_MAX_ATTRIBUTE_CHARS = 65_536
_MAX_SECONDS = 0.25
_ELEMENTS = frozenset(
    {
        "svg",
        "g",
        "defs",
        "path",
        "rect",
        "circle",
        "ellipse",
        "line",
        "polyline",
        "polygon",
        "text",
        "tspan",
        "title",
        "desc",
        "style",
    }
)
_TEXT_ELEMENTS = frozenset({"text", "tspan", "title", "desc"})
_NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
_NUMBERS = re.compile(rf"{_NUMBER}(?:(?:\s*,\s*|\s+){_NUMBER})*")
_LENGTH = re.compile(rf"({_NUMBER})(?:%|px|pt|pc|mm|cm|in|em|ex)?")
_IDENTIFIERS = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]*(?:\s+[A-Za-z_][A-Za-z0-9_.:-]*)*")
_COLOR = re.compile(r"#(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{4}|[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})")
_COLOR_KEYWORDS = frozenset(
    {
        "none",
        "currentColor",
        "inherit",
        "transparent",
        "black",
        "white",
        "red",
        "green",
        "blue",
        "gray",
        "grey",
    }
)
_SELECTOR = re.compile(r"[.#]?[A-Za-z_][A-Za-z0-9_.-]*(?:\s*,\s*[.#]?[A-Za-z_][A-Za-z0-9_.-]*)*")
_PATH_TOKEN = re.compile(rf"[MmZzLlHhVvCcSsQqTtAa]|{_NUMBER}")
_PATH_ARITY = {"m": 2, "z": 0, "l": 2, "h": 1, "v": 1, "c": 6, "s": 4, "q": 4, "t": 2, "a": 7}
_TRANSFORM = re.compile(
    rf"\s*(matrix|translate|scale|rotate|skewX|skewY)\(\s*({_NUMBER}(?:[\s,]+{_NUMBER})*)\s*\)\s*"
)
_TRANSFORM_ARITY = {
    "matrix": {6},
    "translate": {1, 2},
    "scale": {1, 2},
    "rotate": {1, 3},
    "skewX": {1},
    "skewY": {1},
}
_LENGTH_ATTRIBUTES = frozenset(
    {
        "x",
        "y",
        "x1",
        "y1",
        "x2",
        "y2",
        "cx",
        "cy",
        "r",
        "rx",
        "ry",
        "width",
        "height",
        "dx",
        "dy",
        "stroke-width",
        "stroke-miterlimit",
        "stroke-dashoffset",
        "font-size",
        "letter-spacing",
        "word-spacing",
        "opacity",
        "fill-opacity",
        "stroke-opacity",
    }
)
_ENUM_ATTRIBUTES = {
    "fill-rule": {"nonzero", "evenodd"},
    "clip-rule": {"nonzero", "evenodd"},
    "stroke-linecap": {"butt", "round", "square"},
    "stroke-linejoin": {"miter", "round", "bevel"},
    "overflow": {"visible", "hidden", "scroll", "auto"},
    "display": {"none", "inline"},
    "visibility": {"visible", "hidden", "collapse"},
    "text-anchor": {"start", "middle", "end"},
    "font-style": {"normal", "italic", "oblique"},
    "font-weight": {"normal", "bold", "bolder", "lighter", *map(str, range(100, 1000, 100))},
    "vector-effect": {"none", "non-scaling-stroke"},
}
_STYLE_PROPERTIES = (
    _LENGTH_ATTRIBUTES
    | _ENUM_ATTRIBUTES.keys()
    | {
        "fill",
        "stroke",
        "color",
        "stroke-dasharray",
        "enable-background",
    }
)


class _UnsupportedSvgError(ValueError):
    pass


def _numbers(value: str) -> list[float] | None:
    if _NUMBERS.fullmatch(value) is None:
        return None
    numbers = [float(number) for number in re.split(r"[\s,]+", value)]
    return numbers if all(math.isfinite(number) for number in numbers) else None


def _path(value: str) -> bool:
    """Accept complete command groups; unsupported compact arc flags stay unproven."""
    if not value:
        return True
    command = ""
    values: list[str] = []
    cursor = 0

    def complete() -> bool:
        arity = _PATH_ARITY[command]
        if not arity:
            return not values
        if not values or len(values) % arity:
            return False
        if command == "a":
            return all(
                float(values[index]) >= 0
                and float(values[index + 1]) >= 0
                and values[index + 3] in {"0", "1"}
                and values[index + 4] in {"0", "1"}
                for index in range(0, len(values), arity)
            )
        return True

    for match in _PATH_TOKEN.finditer(value):
        separator = value[cursor : match.start()]
        token = match.group()
        is_command = token.lower() in _PATH_ARITY
        if separator.strip() not in {"", ","} or ("," in separator and (not values or is_command)):
            return False
        if is_command:
            if not command and token.lower() != "m" or command and not complete():
                return False
            command = token.lower()
            values = []
        else:
            if not command or command == "z" or not math.isfinite(float(token)):
                return False
            values.append(token)
        cursor = match.end()
    return bool(command) and not value[cursor:].strip() and complete()


def _attribute(name: str, value: str) -> bool:
    value = value.strip()
    if name in _LENGTH_ATTRIBUTES:
        match = _LENGTH.fullmatch(value)
        return bool(
            match is not None
            and math.isfinite(float(match[1]))
            and (
                name not in {"width", "height", "r", "rx", "ry", "stroke-width", "font-size"}
                or float(match[1]) >= 0
            )
        )
    if name in _ENUM_ATTRIBUTES:
        return value in _ENUM_ATTRIBUTES[name]
    if name in {"fill", "stroke", "color"}:
        return value in _COLOR_KEYWORDS or _COLOR.fullmatch(value) is not None
    if name in {"id", "class", "aria-labelledby", "aria-describedby", "role", "data-name"}:
        return _IDENTIFIERS.fullmatch(value) is not None
    if name == "viewBox":
        numbers = _numbers(value)
        return numbers is not None and len(numbers) == 4 and numbers[2] >= 0 and numbers[3] >= 0
    if name == "points":
        numbers = _numbers(value)
        return numbers is not None and len(numbers) % 2 == 0
    if name == "stroke-dasharray":
        numbers = _numbers(value)
        return value == "none" or numbers is not None and all(number >= 0 for number in numbers)
    if name == "d":
        return _path(value)
    if name == "transform":
        cursor = 0
        while cursor < len(value):
            match = _TRANSFORM.match(value, cursor)
            numbers = _numbers(match[2]) if match is not None else None
            if numbers is None or len(numbers) not in _TRANSFORM_ARITY[match[1]]:
                return False
            cursor = match.end()
            if cursor < len(value) and value[cursor] == ",":
                cursor += 1
                if not value[cursor:].strip():
                    return False
        return bool(value)
    if name == "enable-background":
        return value == "accumulate" or (
            value.startswith("new ") and _numbers(value[4:]) is not None
        )
    if name == "preserveAspectRatio":
        return (
            re.fullmatch(r"(?:none|x(?:Min|Mid|Max)Y(?:Min|Mid|Max))(?:\s+(?:meet|slice))?", value)
            is not None
        )
    if name == "version":
        return value in {"1.0", "1.1", "2.0"}
    if name == f"{_XML}}}space":
        return value in {"default", "preserve"}
    if name == "style":
        return _declarations(value)
    return False


def _declarations(content: str) -> bool:
    for declaration in content.split(";"):
        if not declaration.strip():
            continue
        name, separator, value = declaration.partition(":")
        name = name.strip()
        if not separator or name not in _STYLE_PROPERTIES or not _attribute(name, value):
            return False
    return True


def _stylesheet(content: str) -> bool:
    if not content.strip():
        return True
    blocks = content.split("}")
    if blocks.pop().strip():
        return False
    for block in blocks:
        selector, separator, declarations = block.partition("{")
        if (
            not separator
            or not _SELECTOR.fullmatch(selector.strip())
            or not _declarations(declarations)
        ):
            return False
    return True


def is_structurally_valid_passive_svg(data: bytes) -> bool:
    """Recognize only complete UTF-8 drawings with bounded, passive XML/CSS."""
    if not data or len(data) > _MAX_BYTES:
        return False
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    deadline = time.monotonic() + _MAX_SECONDS
    parser = expat.ParserCreate(namespace_separator="}")
    stack: list[str] = []
    styles: list[str] = []
    nodes = 0

    def reject(*_args: object) -> None:
        raise _UnsupportedSvgError

    def start(name: str, attributes: dict[str, str]) -> None:
        nonlocal nodes
        nodes += 1
        if (
            nodes > _MAX_NODES
            or len(stack) >= _MAX_DEPTH
            or len(attributes) > 64
            or time.monotonic() > deadline
        ):
            reject()
        namespace, separator, local = name.rpartition("}")
        if not separator or namespace != _SVG or local not in _ELEMENTS:
            reject()
        if (not stack and local != "svg") or (stack and stack[-1] in {"style", "title", "desc"}):
            reject()
        for key, value in attributes.items():
            if len(value) > _MAX_ATTRIBUTE_CHARS:
                reject()
            if local == "style" and key == "type" and value == "text/css":
                continue
            if not _attribute(key, value):
                reject()
        stack.append(local)

    def end(_name: str) -> None:
        if stack.pop() == "style":
            if not _stylesheet("".join(styles)):
                reject()
            styles.clear()

    def text(value: str) -> None:
        if time.monotonic() > deadline:
            reject()
        if stack and stack[-1] == "style":
            styles.append(value)
        elif value.strip() and (not stack or stack[-1] not in _TEXT_ELEMENTS):
            reject()

    def namespace(_prefix: str | None, uri: str | None) -> None:
        if uri not in {_SVG, _XML, _XLINK}:
            reject()

    def declaration(version: str, encoding: str | None, _standalone: int) -> None:
        if version != "1.0" or encoding is not None and encoding.lower() not in {"utf-8", "utf8"}:
            reject()

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = text
    parser.StartNamespaceDeclHandler = namespace
    parser.XmlDeclHandler = declaration
    parser.StartDoctypeDeclHandler = reject
    parser.EntityDeclHandler = reject
    parser.ProcessingInstructionHandler = reject
    parser.CommentHandler = reject
    parser.ExternalEntityRefHandler = lambda *_args: 0
    try:
        parser.Parse(content, True)
    except (expat.ExpatError, _UnsupportedSvgError, ValueError):
        return False
    return nodes > 0 and not stack and time.monotonic() <= deadline
