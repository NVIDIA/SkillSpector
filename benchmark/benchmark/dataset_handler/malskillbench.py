"""MalSkillBench dataset handler.

Point this at a local MalSkillBench checkout's ``Dataset`` directory (or any
subtree of it, e.g. ``Dataset/Prompts/indirect-injection``). The four Dataset
categories have different on-disk shapes:

* ``Skills/``  -- proper skill directories under ``malware/`` and ``benign/``,
                 scanned in place. Ground truth comes from the parent dir; the
                 fine taxonomy (vector CI/PI/MIXED, behavior B1-B15, insertion
                 strategy) is resolved from ``Skills/malware/_source_inventory.txt``
                 -- which covers every malware skill, including the ~2400 whose
                 directory name carries no ``__PI_B4`` suffix.
* ``Codes/``   -- flat ``*.json`` files, each a list of
                 ``{"pyfile", "malicious_code_snippets"}``. All malicious (CI).
                 Each snippet is materialized into a temp dir; the behavior id
                 is joined from ``Codes/.../malware_classified/_classified.json``.
* ``Prompts/`` -- flat ``*.json`` files with heterogeneous schemas. A prompt
                 text and (where available) a 0/1 label are extracted and
                 materialized as a ``SKILL.md`` (PI). The behavior id is joined
                 from ``Prompts/classified/_classified.json`` when present.
* ``MCPs/``    -- currently empty.
"""

from __future__ import annotations

import json
import pathlib
import re

from ..models import Unit
from ..utils import canonical_label, rel_id
from .base import DatasetHandler

# Dataset category roots (top-level dirs inside MalSkillBench/Dataset).
_CATEGORIES = ("Skills", "Codes", "Prompts", "MCPs")

# JSON files that are corpus metadata / indexes rather than samples.
_SKIP_JSON = frozenset({"knowledge_base.json", "_classified.json", "_deduped.json"})

# Keys we try, in order, to pull a prompt body out of a heterogeneous record.
_PROMPT_TEXT_KEYS = (
    "text",
    "adversarial",
    "prompt",
    "user_input",
    "vanilla",
    "completion",
    "model_output",
    "act",
)
# Keys that carry a malicious/benign label (truthy => malicious).
_PROMPT_LABEL_KEYS = ("label", "jailbreaking", "jailbreak")

_STRATEGY_TS = re.compile(r"_\d{6,}.*$")  # strip trailing _<timestamp>[_hash]


# --------------------------------------------------------------------------- #
# Path / category helpers
# --------------------------------------------------------------------------- #
def _dataset_anchor(root: pathlib.Path) -> pathlib.Path:
    """Find the enclosing 'Dataset' dir so unit ids/labels resolve consistently."""
    for p in (root, *root.parents):
        if p.name == "Dataset":
            return p
    return root


def _detect_category(root: pathlib.Path) -> str | None:
    """Return the Dataset category a path falls under, or None for the root.

    None means `root` is (or looks like) the Dataset directory itself and every
    present category should be dispatched.
    """
    parts = root.parts
    for cat in _CATEGORIES:
        if cat in parts:
            return cat
    if any((root / c).is_dir() for c in _CATEGORIES):
        return None
    return "Skills"


# --------------------------------------------------------------------------- #
# Authoritative label resolution (loaded once per run)
# --------------------------------------------------------------------------- #
class LabelStore:
    """Loads MalSkillBench's authoritative label sources, keyed for joining."""

    def __init__(self, anchor: pathlib.Path) -> None:
        self.inventory = _load_inventory(anchor)  # skill_dir_name -> taxonomy dict
        self.code_behavior = _load_code_classified(anchor)  # (json_name, pyfile) -> Bxx
        self.prompt_behavior = _load_prompt_classified(anchor)  # (source, text) -> Bxx


def _load_inventory(anchor: pathlib.Path) -> dict[str, dict]:
    """Parse Skills/malware/_source_inventory.txt -> {skill_dir: taxonomy}.

    Plain ``GENERATED/WILD/TEST`` lines give the dir name + source path (split
    on ``<-`` so dir names containing spaces survive). Generated source paths
    encode ``generated_malicious/<VECTOR>/<BEHAVIOR>/<strategy>_<ts>``. The
    ``*_LABEL`` lines supply vector/behavior for wild/test samples whose source
    path doesn't encode them; they share the source path as the join key.
    """
    path = anchor / "Skills" / "malware" / "_source_inventory.txt"
    if not path.is_file():
        return {}
    dir_by_src: dict[str, tuple[str, str]] = {}  # src -> (sample_type, dir)
    tax_by_src: dict[str, dict] = {}  # src -> {attack_vector, behavior}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "<-" not in line:
            continue
        left, src = line.split("<-", 1)
        src = src.strip()
        toks = left.split()
        if not toks:
            continue
        typ = toks[0]
        if typ in ("GENERATED", "WILD", "TEST"):
            dir_by_src[src] = (typ, left[len(typ) :].strip())
        elif typ == "GENERATED_LABEL" and len(toks) >= 3:
            tax_by_src[src] = {"attack_vector": toks[1], "behavior": toks[2]}
        elif typ in ("WILD_LABEL", "TEST_LABEL") and len(toks) >= 2:
            tax_by_src[src] = {"attack_vector": None, "behavior": toks[1]}

    result: dict[str, dict] = {}
    for src, (sample_type, skill_dir) in dir_by_src.items():
        tax = tax_by_src.get(src, {})
        parts = src.split("/")
        vector = tax.get("attack_vector")
        behavior = tax.get("behavior")
        strategy = None
        if parts and parts[0] == "generated_malicious":
            if len(parts) >= 2 and not vector:
                vector = parts[1]
            if len(parts) >= 3 and not behavior:
                behavior = parts[2]
            if len(parts) >= 4:
                strategy = _STRATEGY_TS.sub("", parts[3]) or None
        result[skill_dir] = {
            "sample_type": sample_type,
            "attack_vector": vector,
            "behavior": behavior,
            "insertion_strategy": strategy,
        }
    return result


def _load_json_list(path: pathlib.Path) -> list:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def _load_code_classified(anchor: pathlib.Path) -> dict[tuple[str, str], str]:
    """(json_filename, pyfile) -> behavior_id, from the CI classified corpora."""
    out: dict[tuple[str, str], str] = {}
    for rel in (
        "Codes/Python/malware_classified/_classified.json",
        "Codes/IntelliGraph/classified/_classified.json",
    ):
        for e in _load_json_list(anchor / rel):
            if not isinstance(e, dict):
                continue
            bid = e.get("behavior_id")
            pyf = e.get("pyfile")
            if not bid or not pyf:
                continue
            fn = e.get("file")
            pkg = e.get("package_name")
            if fn:
                out[(fn, pyf)] = bid
            if pkg:  # fallback join key: stem + .json
                out.setdefault((f"{pkg}.json", pyf), bid)
    return out


def _load_prompt_classified(anchor: pathlib.Path) -> dict[tuple[str, str], str]:
    """(source, text) -> behavior_id, from the PI classified corpus."""
    out: dict[tuple[str, str], str] = {}
    for e in _load_json_list(anchor / "Prompts" / "classified" / "_classified.json"):
        if not isinstance(e, dict):
            continue
        src = e.get("source")
        txt = e.get("text")
        bid = e.get("behavior_id")
        if src and isinstance(txt, str) and bid:
            out[(src, txt.strip())] = bid
    return out


# --------------------------------------------------------------------------- #
# Unit discovery (one collector per category)
# --------------------------------------------------------------------------- #
def _parse_skill_labels(name: str) -> dict[str, str | None]:
    """Parse attack labels out of a malware skill dir name (fallback path).

    Names look like ``credential-guard__PI_B2__Full_Camouflage__...`` or just
    ``inngest__CI_B1``. Most malware should resolve via the inventory; this only
    catches anything the inventory misses.
    """
    out: dict[str, str | None] = {
        "attack_vector": None,
        "behavior": None,
        "insertion_strategy": None,
    }
    for tok in name.split("__")[1:]:
        head = tok.split("_")[0]
        if head in ("CI", "PI", "MIXED"):
            out["attack_vector"] = head
            rest = tok[len(head) + 1 :]
            if rest.startswith("B") and rest.split("_")[0][1:].isdigit():
                out["behavior"] = "B" + rest.split("_")[0][1:]
        elif tok.startswith("B") and tok[1:].split("_")[0].isdigit():
            out["behavior"] = "B" + tok[1:].split("_")[0]
        elif (
            out["insertion_strategy"] is None
            and any(c.isalpha() for c in tok)
            and not _looks_like_runid(tok)
        ):
            out["insertion_strategy"] = tok
    return out


def _looks_like_runid(tok: str) -> bool:
    """Heuristic: trailing tokens like full_camouflage_20260426_065726_be0d2b."""
    return any(part.isdigit() and len(part) >= 6 for part in tok.split("_"))


def collect_skills(root: pathlib.Path, anchor: pathlib.Path, labels: LabelStore) -> list[Unit]:
    units: list[Unit] = []
    if root.parent.name in ("malware", "benign"):
        candidates = [root]
    else:
        candidates = [
            d for d in root.rglob("*") if d.is_dir() and d.parent.name in ("malware", "benign")
        ]
    for d in sorted(candidates):
        cls = d.parent.name
        malicious = cls == "malware"
        unit = Unit(
            unit_path=rel_id(d, anchor),
            category="skill",
            source_path=str(d),
            display_name=d.name.split("__")[0],
            is_malicious=malicious,
            corpus=cls,
        )
        if malicious:
            _label_malware_skill(unit, d.name, labels)
        else:
            unit.label_source = "dir"  # benign status resolved from the directory
        units.append(unit)
    return units


def _label_malware_skill(unit: Unit, dir_name: str, labels: LabelStore) -> None:
    """Resolve a malware skill's taxonomy: inventory first, then name parsing."""
    inv = labels.inventory.get(dir_name)
    if inv:
        unit.attack_vector = inv["attack_vector"]
        unit.behavior = inv["behavior"]
        unit.insertion_strategy = inv["insertion_strategy"]
        unit.sample_type = inv["sample_type"]
        unit.label = canonical_label(inv["attack_vector"], inv["behavior"])
        unit.label_source = "inventory"
        return
    parsed = _parse_skill_labels(dir_name)
    unit.attack_vector = parsed["attack_vector"]
    unit.behavior = parsed["behavior"]
    unit.insertion_strategy = parsed["insertion_strategy"]
    guess = canonical_label(parsed["attack_vector"], parsed["behavior"])
    if guess:
        unit.label = guess  # parsed from an explicit __VECTOR_Bxx suffix
        unit.label_source = "name"
    else:
        unit.label_source = "dir"  # malicious is known; fine taxonomy is not


def _iter_json_records(path: pathlib.Path):
    for i, rec in enumerate(_load_json_list(path)):
        if isinstance(rec, dict):
            yield i, rec


def collect_codes(root: pathlib.Path, anchor: pathlib.Path, labels: LabelStore) -> list[Unit]:
    units: list[Unit] = []
    files = [root] if root.is_file() else sorted(root.rglob("*.json"))
    for f in files:
        if f.name in _SKIP_JSON or "classified" in f.parts:
            continue
        rel = rel_id(f, anchor)
        for i, rec in _iter_json_records(f):
            snippet = rec.get("malicious_code_snippets")
            if not isinstance(snippet, str) or not snippet.strip():
                continue
            pyfile = rec.get("pyfile") or "snippet.py"
            behavior = labels.code_behavior.get((f.name, pyfile))
            units.append(
                Unit(
                    unit_path=f"{rel}#{i}",
                    category="code",
                    source_path=f"{f}#{i}",
                    display_name=f"{f.stem}:{pyfile}",
                    is_malicious=True,
                    attack_vector="CI",
                    behavior=behavior,
                    label=canonical_label("CI", behavior),
                    label_source="classified" if behavior else "corpus",
                    corpus=f.parent.name,
                    materialize={pyfile: snippet},
                )
            )
    return units


def _prompt_text(rec: dict) -> str | None:
    for k in _PROMPT_TEXT_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _prompt_field_label(rec: dict) -> bool | None:
    """Read an explicit malicious/benign label from a record, or None."""
    for k in _PROMPT_LABEL_KEYS:
        if k in rec:
            v = rec[k]
            if isinstance(v, bool):
                return v
            try:
                # float (not int): a 0/1 label parses, but so does a
                # probability like 0.9 or "0.7" -- int() would floor those to 0
                # (benign) or raise. truthy => malicious.
                return float(v) != 0
            except (TypeError, ValueError):
                continue
    return None


def collect_prompts(root: pathlib.Path, anchor: pathlib.Path, labels: LabelStore) -> list[Unit]:
    units: list[Unit] = []
    files = [root] if root.is_file() else sorted(root.rglob("*.json"))
    for f in files:
        if f.name in _SKIP_JSON or "classified" in f.parts:
            continue
        rel = rel_id(f, anchor)
        corpus = f.parent.name
        source = f"{corpus}/{f.name}"  # matches the classified corpus 'source'
        for i, rec in _iter_json_records(f):
            text = _prompt_text(rec)
            if text is None:
                continue
            units.append(
                _build_prompt_unit(
                    f"{rel}#{i}", f"{f}#{i}", f.stem, i, text, rec, source, corpus, labels
                )
            )
    return units


def _build_prompt_unit(unit_path, source_path, stem, idx, text, rec, source, corpus, labels):
    behavior = labels.prompt_behavior.get((source, text.strip()))
    field_label = _prompt_field_label(rec)
    if field_label is not None:  # explicit 0/1 label wins
        malicious, label_source = field_label, "field"
    elif behavior is not None:  # present in the curated malicious PI corpus
        malicious, label_source = True, "classified"
    else:  # attack corpus with no per-record label -> best guess
        malicious, label_source = True, "corpus"
    label = canonical_label("PI", behavior) if malicious else None
    unit = Unit(
        unit_path=unit_path,
        category="prompt",
        source_path=source_path,
        display_name=f"{stem}#{idx}",
        is_malicious=malicious,
        attack_vector="PI",
        behavior=behavior if malicious else None,
        label_source=label_source,
        corpus=corpus,
        materialize={"SKILL.md": _as_skill_md(stem, text)},
    )
    if label_source == "corpus":
        unit.best_guess_label = canonical_label("PI", behavior) or "PI"
    else:
        unit.label = label
    return unit


def _as_skill_md(name: str, body: str) -> str:
    """Wrap a raw prompt as a minimal SKILL.md so PI analyzers see it."""
    safe = name.replace("\n", " ")[:60]
    return f"---\nname: {safe}\ndescription: benchmark prompt sample\n---\n\n{body}\n"


# --------------------------------------------------------------------------- #
# Handler
# --------------------------------------------------------------------------- #
class MalSkillBenchHandler(DatasetHandler):
    """Discover scannable units from a MalSkillBench ``Dataset`` tree."""

    name = "malskillbench"

    _COLLECTORS = {
        "Skills": collect_skills,
        "Codes": collect_codes,
        "Prompts": collect_prompts,
    }

    def matches(self, root: pathlib.Path) -> bool:
        if _dataset_anchor(root) != root:
            return True  # somewhere inside a .../Dataset/... tree
        if any((root / c).is_dir() for c in _CATEGORIES):
            return True  # the Dataset dir itself
        return root.parent.name in ("malware", "benign")

    def discover(self, root: pathlib.Path) -> list[Unit]:
        """Collect all units under `root`, dispatching by Dataset category."""
        anchor = _dataset_anchor(root)
        labels = LabelStore(anchor)
        category = _detect_category(root)
        if category is None:  # pointed at the Dataset dir: do every present category
            units: list[Unit] = []
            for cat, fn in self._COLLECTORS.items():
                sub = root / cat
                if sub.is_dir():
                    units.extend(fn(sub, anchor, labels))
            return units
        if category == "MCPs":
            return []
        return self._COLLECTORS.get(category, collect_skills)(root, anchor, labels)
