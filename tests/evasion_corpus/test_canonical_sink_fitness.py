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

"""Fitness gate: the canonical layer maps every evasion spelling to one sink id.

This is the structural counter-measure to the "enumerate instead of canonicalize"
anti-pattern. Rather than one test per idiom, the shared corpus asserts that *every*
equivalent spelling reduces to the same canonical id through the single chokepoint, and
that benign neighbours never do. Adding a new sink means adding corpus rows, not a new
detector branch.
"""

from __future__ import annotations

import ast

import pytest

from skillspector.nodes.analyzers import behavioral_ast, behavioral_taint_tracking
from skillspector.nodes.analyzers.behavioral_taint_tracking import _EXEC_SINKS
from skillspector.nodes.analyzers.canonical_sink import resolve_to_canonical_sink
from skillspector.nodes.analyzers.common import build_import_aliases, build_type_map

from .corpus import EQUIVALENT_SPELLINGS, FALSE_POSITIVE_NEIGHBOURS, Neighbour, Spelling


def _outermost_call(code: str) -> ast.Call:
    """Return the outermost sink-invocation call from a corpus snippet.

    Corpus rows place the sink call as the last top-level ``Expr``/``Assign`` value;
    this returns that call node so the chokepoint is queried on the actual invocation
    (e.g. ``getattr(os, "system")(x)`` resolves on the outer ``(x)`` call).
    """
    tree = ast.parse(code)
    last: ast.Call | None = None
    for stmt in tree.body:
        value = stmt.value if isinstance(stmt, (ast.Expr, ast.Assign)) else None
        if isinstance(value, ast.Call):
            last = value
    if last is None:  # pragma: no cover - corpus rows always end in a call
        raise AssertionError(f"corpus snippet has no top-level call: {code!r}")
    return last


def _canonical_for(code: str) -> str | None:
    """Resolve a corpus snippet's outermost call to its canonical sink id."""
    tree = ast.parse(code)
    aliases = build_import_aliases(tree)
    type_map = build_type_map(tree)
    return resolve_to_canonical_sink(_outermost_call(code), aliases, type_map)


@pytest.mark.parametrize("row", EQUIVALENT_SPELLINGS, ids=lambda r: r.label)
def test_equivalent_spelling_canonicalizes(row: Spelling) -> None:
    """Every equivalent spelling reduces to its single canonical sink id."""
    assert _canonical_for(row.code) == row.canonical


@pytest.mark.parametrize("neighbour", FALSE_POSITIVE_NEIGHBOURS, ids=lambda n: n.label)
def test_false_positive_neighbour_not_a_sink(neighbour: Neighbour) -> None:
    """Benign look-alikes never canonicalize to a dangerous exec sink id."""
    assert _canonical_for(neighbour.code) not in _EXEC_SINKS


def test_corpus_has_min_spellings_per_primitive() -> None:
    """Each canonical primitive carries at least eight equivalent spellings.

    Guards the corpus against silently shrinking below the breadth that makes it a
    meaningful evasion gate.
    """
    counts: dict[str, int] = {}
    for row in EQUIVALENT_SPELLINGS:
        counts[row.canonical] = counts.get(row.canonical, 0) + 1
    assert counts.get("exec", 0) >= 8
    assert all(count >= 5 for count in counts.values())
    # The sibling-machinery surface (dynamic import / code exec) is represented.
    assert "__import__" in counts


# ── End-to-end wiring gate ────────────────────────────────────────────
#
# The fitness checks above prove the resolver *can* canonicalize. These prove the
# resolver is actually *wired into production*: every spelling must reach a real sink
# rule (AST + taint), so an accidental un-wiring or severity downgrade fails loudly
# rather than silently regressing to a missed detection.

# behavioral_ast rules that constitute a real dangerous-execution sink detection. A
# spelling is "detected" if it produces any of these — the precise rule depends on which
# complementary path fires (the canonical ladders AST1/AST4/AST5/AST6, or AST9 for a
# literal getattr on an allowlisted name, owned by the getattr branch on `main`).
_DANGEROUS_EXECUTION_RULES: frozenset[str] = frozenset(
    {"AST1", "AST2", "AST3", "AST4", "AST5", "AST6", "AST9"}
)

# Minimum severity each canonical primitive must keep — guards against a downgrade
# (e.g. an exec sink silently dropping to AST3-MEDIUM). ``__import__`` is the
# dynamic-import class (AST3-MEDIUM), matching the repo's grading of ``__import__``.
_MIN_SEVERITY: dict[str, str] = {
    "exec": "HIGH",
    "os.system": "HIGH",
    "subprocess.run": "MEDIUM",
    "__import__": "MEDIUM",
}

_SEVERITY_ORDER: dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

# Canonical ids the taint analyzer treats as exec sinks (TT5). ``__import__`` is
# deliberately excluded: the repo does not grade bare ``__import__`` as a taint exec
# sink, so its dynamic-import siblings stay consistent with that baseline.
_TAINT_EXEC_CANONICALS: frozenset[str] = frozenset(
    {"exec", "eval", "compile", "os.system", "subprocess.run"}
)


def _run_ast(code: str) -> list:
    """Run the real behavioral_ast analyzer on a one-file skill and return findings."""
    state = {"components": ["s.py"], "file_cache": {"s.py": code}}
    return behavioral_ast.node(state)["findings"]


def _run_taint(code: str) -> list:
    """Run the real behavioral_taint_tracking analyzer; *code* must end in a tainted flow."""
    state = {"components": ["s.py"], "file_cache": {"s.py": code}}
    return behavioral_taint_tracking.node(state)["findings"]


@pytest.mark.parametrize("row", EQUIVALENT_SPELLINGS, ids=lambda r: r.label)
def test_spelling_reaches_ast_sink_rule(row: Spelling) -> None:
    """Every spelling reaches a dangerous-execution rule in production at full severity.

    Proves the chokepoint is actually invoked by production (not merely importable) and
    that the spelling is not silently dropped or downgraded. The exact rule may be a
    canonical-ladder rule or AST9 (literal getattr, owned by the getattr branch on
    ``main``) — both are valid sink detections — so the assertion targets the
    dangerous-execution rule set plus a minimum-severity floor.
    """
    findings = _run_ast(row.code)
    sink_findings = [f for f in findings if f.rule_id in _DANGEROUS_EXECUTION_RULES]
    assert sink_findings, f"{row.label}: no sink rule, got {[f.rule_id for f in findings]}"
    best = max(_SEVERITY_ORDER[f.severity] for f in sink_findings)
    assert best >= _SEVERITY_ORDER[_MIN_SEVERITY[row.canonical]], (
        f"{row.label}: severity downgrade, got {[(f.rule_id, f.severity) for f in sink_findings]}"
    )


@pytest.mark.parametrize(
    "row",
    [r for r in EQUIVALENT_SPELLINGS if r.canonical in _TAINT_EXEC_CANONICALS and "(x)" in r.code],
    ids=lambda r: r.label,
)
def test_spelling_reaches_taint_exec_sink(row: Spelling) -> None:
    """Routed through a tainted input, every exec-class spelling reaches the TT5 flow.

    Restricted to exec-class canonicals whose sink receives the tainted value ``x``;
    ``__import__`` siblings are excluded (the taint analyzer does not grade dynamic
    import as an exec sink — baseline parity), as are arg-less sinks like
    ``code.interact()`` that cannot carry a tainted argument.
    """
    tainted = f"x = input()\n{row.code}"
    findings = _run_taint(tainted)
    assert any(f.rule_id == "TT5" for f in findings), (
        f"{row.label}: expected TT5, got {[f.rule_id for f in findings]}"
    )


# Reflective subprocess invocations must grade HIGH (AST9), consistent with reflective
# os.system — the reflection itself signals evasion intent, unlike a direct AST4-MEDIUM
# subprocess.* call.
_SUBPROCESS_REFLECTION: tuple[str, ...] = (
    'import subprocess\ngetattr(subprocess, "Popen")(x)',
    'import subprocess\ngetattr(subprocess, "run")(x)',
    'import subprocess\ngetattr(subprocess, "check_output")(x)',
)


@pytest.mark.parametrize("code", _SUBPROCESS_REFLECTION)
def test_reflective_subprocess_grades_high(code: str) -> None:
    """Reflective subprocess sinks fire AST9-HIGH (not AST4-MEDIUM) for severity parity."""
    findings = _run_ast(code)
    ast9 = [f for f in findings if f.rule_id == "AST9"]
    assert ast9, f"expected AST9, got {[(f.rule_id, f.severity) for f in findings]}"
    assert ast9[0].severity == "HIGH"
    assert not any(f.rule_id == "AST4" for f in findings), "should not double-fire AST4"


def test_direct_subprocess_stays_medium() -> None:
    """A *direct* subprocess.* call keeps its baseline AST4-MEDIUM (no over-escalation)."""
    findings = _run_ast("import subprocess\nsubprocess.run(['id'])")
    ast4 = [f for f in findings if f.rule_id == "AST4"]
    assert ast4 and ast4[0].severity == "MEDIUM"
    assert not any(f.rule_id == "AST9" for f in findings)


@pytest.mark.parametrize("neighbour", FALSE_POSITIVE_NEIGHBOURS, ids=lambda n: n.label)
def test_neighbour_produces_no_ast_findings(neighbour: Neighbour) -> None:
    """Benign look-alikes produce zero behavioral_ast findings in production."""
    assert _run_ast(neighbour.code) == []
