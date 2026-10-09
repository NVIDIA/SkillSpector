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

"""Expose the file surface a finding landed on.

The label is *advisory reporting metadata*. It never filters, ranks, or
suppresses a finding: severity, confidence, the risk score, and the finding
count are unchanged by it. Consumers sort or group on it themselves -- read the
``code`` hits first, diff two scans per surface, or write a local suppression
policy and take on that risk explicitly -- instead of the scanner deciding.

Labels: ``code`` (executable source, and anything unclassified),
``instructions`` (``SKILL.md``, which the agent executes rather than prose
*about* a command), ``docs``, ``tests``, ``comments`` (a comment line inside a
``code``/``config`` file), and ``config`` (configuration and manifests).

``comments`` outranks only ``code`` and ``config``: a comment inside a test file
stays ``tests``. Every other label comes from the path, so a caller without the
matched line still gets a stable one.
"""

from __future__ import annotations

CODE = "code"
INSTRUCTIONS = "instructions"
DOCS = "docs"
TESTS = "tests"
COMMENTS = "comments"
CONFIG = "config"

SURFACES: frozenset[str] = frozenset({CODE, INSTRUCTIONS, DOCS, TESTS, COMMENTS, CONFIG})

_INSTRUCTION_BASENAMES = frozenset({"skill.md"})
_TEST_DIRS = frozenset("test tests __tests__ spec specs".split())
_TEST_BASENAMES = frozenset("test tests".split())
_TEST_PREFIXES = ("test_", "test-")
_TEST_SUFFIXES = ("_test", ".test", ".spec")
_DOC_DIRS = frozenset("doc docs documentation manual man".split())
_DOC_EXTENSIONS = frozenset(".md .markdown .rst .adoc .txt".split())
_DOC_STEMS = frozenset("readme changelog changes contributing license notice faq".split())
_CONFIG_EXTENSIONS = frozenset(
    ".json .jsonc .yaml .yml .toml .ini .cfg .conf .properties .lock .mod .sum .env".split()
)
_CONFIG_BASENAMES = frozenset(
    "requirements.txt constraints.txt gemfile gemfile.lock pipfile pipfile.lock justfile .env".split()
)
_CONFIG_DIRS = frozenset("config configs .github requirements".split())
_EXECUTABLE_EXTENSIONS = frozenset(
    ".py .pyi .sh .bash .zsh .ps1 .js .mjs .cjs .jsx .ts .tsx .go .rs .java .kt "
    ".c .h .cc .cpp .hpp .cs .php .swift .rb .pl .tf .lua .sql .hs".split()
)

# ``//`` is a complete line comment; ``/*`` is only a comment when it has
# no same-line closing delimiter.  A bare ``*/`` or ``*`` continuation is not
# enough to prove a line is comment-only.
_BLOCK_MARKERS = ("//", "/*")
_COMMENT_MARKERS: dict[str, tuple[str, ...]] = {
    **dict.fromkeys(
        ".py .pyi .sh .bash .zsh .ps1 .yaml .yml .toml .rb .pl .tf .cfg .conf .ini .env".split(),
        ("#",),
    ),
    **dict.fromkeys(
        ".js .mjs .cjs .jsx .ts .tsx .go .rs .java .kt .c .h .cc .cpp .hpp .cs .php .swift".split(),
        _BLOCK_MARKERS,
    ),
    **dict.fromkeys(".sql .lua .hs".split(), ("--",)),
    **dict.fromkeys(".html .htm .xml .svg".split(), ("<!--",)),
}


def _parts(path: str) -> list[str]:
    """Split a path into case-folded segments, treating ``\\`` as a separator."""
    return [part.casefold() for part in path.replace("\\", "/").split("/") if part]


def _stem(basename: str) -> str:
    """Return a basename without its final extension (dotfiles keep their name)."""
    index = basename.rfind(".")
    return basename[:index] if index > 0 else basename


def _extension(basename: str) -> str:
    """Return the lower-cased final extension, or ``""`` for dotfiles."""
    index = basename.rfind(".")
    return basename[index:] if index > 0 else ""


def _is_env(basename: str) -> bool:
    """Return whether a basename is a dotenv file such as ``.env.local``."""
    return basename == ".env" or basename.startswith(".env.")


def _is_test(parts: list[str], basename: str) -> bool:
    """Return whether a path lives in a test tree or uses a test basename."""
    if any(part in _TEST_DIRS for part in parts[:-1]):
        return True
    if basename.startswith(_TEST_PREFIXES):
        return True
    stem = _stem(basename)
    return stem in _TEST_BASENAMES or stem.endswith(_TEST_SUFFIXES)


def _is_config_basename(basename: str) -> bool:
    """Return whether a basename is a configuration or manifest name."""
    return (
        _is_env(basename)
        or basename in _CONFIG_BASENAMES
        or (basename.startswith("requirements") and basename.endswith(".txt"))
    )


def _is_docs(parts: list[str], basename: str) -> bool:
    """Return whether a path is documentation prose."""
    extension = _extension(basename)
    if extension in _DOC_EXTENSIONS:
        return True
    # Conventional doc stems only count for extensionless files, so a script
    # such as ``install.sh`` is not read as prose for its name.
    if not extension and _stem(basename) in _DOC_STEMS:
        return True
    return any(part in _DOC_DIRS for part in parts[:-1])


def _classify(parts: list[str], basename: str) -> str:
    """Return the path-derived surface label."""
    if _is_test(parts, basename):
        return TESTS
    if basename in _INSTRUCTION_BASENAMES:
        return INSTRUCTIONS
    # Manifest basenames win over documentation, so ``requirements.txt`` is not
    # reported as prose merely for its extension.
    if _is_config_basename(basename):
        return CONFIG
    # Executable source extensions must not be contradicted by a docs/config
    # directory name (for example ``docs/install.sh`` or ``config/hook.py``).
    extension = _extension(basename)
    if extension in _EXECUTABLE_EXTENSIONS:
        return CODE
    if any(part == "requirements" for part in parts[:-1]) and extension == ".txt":
        return CONFIG
    if _is_docs(parts, basename):
        return DOCS
    if extension in _CONFIG_EXTENSIONS or any(part in _CONFIG_DIRS for part in parts[:-1]):
        return CONFIG
    return CODE


def _is_comment_line(line_text: str, basename: str) -> bool:
    """Return whether a line is provably comment-only for the file's syntax."""
    extension = _extension(basename)
    markers = _COMMENT_MARKERS.get(extension, ("#",) if _is_env(basename) else ())
    stripped = line_text.strip()
    if not markers or not stripped:
        return False
    if "#" in markers and stripped.startswith("#"):
        # In PowerShell, ``#>`` closes a block comment wherever it appears;
        # non-whitespace after any closer is executable and must not be
        # labelled as a comment.
        if extension == ".ps1":
            closer = stripped.find("#>")
            if closer >= 0 and stripped[closer + 2 :].strip():
                return False
        return True
    if "//" in markers and stripped.startswith("//"):
        # PHP's ``?>`` exits PHP mode even when it appears inside a ``//``
        # comment, so code after the closer executes.
        return not (extension == ".php" and "?>" in stripped)
    if "--" in markers and stripped.startswith("--"):
        if stripped.startswith("--["):
            level_end = 3
            while level_end < len(stripped) and stripped[level_end] == "=":
                level_end += 1
            if level_end < len(stripped) and stripped[level_end] == "[":
                closer = "]" + "=" * (level_end - 3) + "]"
                return closer not in stripped[level_end + 1 :]
        return True
    if "/*" in markers and stripped.startswith("/*"):
        return "*/" not in stripped[2:]
    if "<!--" in markers and stripped.startswith("<!--"):
        comment_body = stripped[2:]
        return "-->" not in comment_body and "--!>" not in comment_body
    return False


def infer_surface(file_path: str, line_text: str | None = None) -> str:
    """Return the reporting surface for a finding located at *file_path*.

    ``line_text`` is the matched source line when the caller has it. It only
    refines ``code``/``config`` paths into ``comments``; every other label is
    path-derived, so callers without the line still get a stable classification.
    """
    parts = _parts(file_path)
    basename = parts[-1] if parts else ""
    surface = _classify(parts, basename)
    if (
        line_text is not None
        and surface in {CODE, CONFIG}
        and _is_comment_line(line_text, basename)
    ):
        return COMMENTS
    return surface
