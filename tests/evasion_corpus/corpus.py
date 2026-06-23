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

"""Shared evasion corpus: equivalent spellings of dangerous sinks + FP neighbours.

A single source of truth that *every* sink detector must satisfy. ``EQUIVALENT_SPELLINGS``
lists, per canonical sink id, ≥8 semantically equivalent ways to invoke it (bare name,
import alias, ``from``/``as`` rebinding, ``builtins.*``, ``importlib.import_module``,
``getattr``, subscript reflection). ``FALSE_POSITIVE_NEIGHBOURS`` lists benign code that
merely *resembles* a sink and must never canonicalize to one. New detectors import these
and assert the invariant, so a future regression (a missed spelling) fails a shared gate
rather than shipping as a silent blind spot.
"""

from __future__ import annotations

from typing import NamedTuple


class Spelling(NamedTuple):
    """One concrete way to write a call, with the canonical sink id it must reduce to."""

    label: str
    code: str
    canonical: str


class Neighbour(NamedTuple):
    """A benign call that resembles a sink and must NOT reduce to any dangerous sink id."""

    label: str
    code: str


# Each row's ``code`` ends with the sink invocation as the final top-level statement.
EQUIVALENT_SPELLINGS: tuple[Spelling, ...] = (
    # ── exec (bare builtin) ───────────────────────────────────────────
    Spelling("exec_bare", "exec(x)", "exec"),
    Spelling("exec_from_builtins", "from builtins import exec\nexec(x)", "exec"),
    Spelling("exec_builtins_attr", "import builtins\nbuiltins.exec(x)", "exec"),
    Spelling("exec_builtins_alias", "import builtins as b\nb.exec(x)", "exec"),
    Spelling("exec_from_builtins_as", "from builtins import exec as e\ne(x)", "exec"),
    Spelling("exec_getattr", 'import builtins\ngetattr(builtins, "exec")(x)', "exec"),
    Spelling("exec_subscript_dunder", '__builtins__["exec"](x)', "exec"),
    Spelling("exec_subscript_vars", 'import builtins\nvars(builtins)["exec"](x)', "exec"),
    Spelling("exec_builtins_dunder_dict", 'import builtins\nbuiltins.__dict__["exec"](x)', "exec"),
    # ── os.system (dotted module sink) ────────────────────────────────
    Spelling("os_system_direct", "import os\nos.system(x)", "os.system"),
    Spelling("os_system_alias", "import os as o\no.system(x)", "os.system"),
    Spelling("os_system_from", "from os import system\nsystem(x)", "os.system"),
    Spelling("os_system_getattr", 'import os\ngetattr(os, "system")(x)', "os.system"),
    Spelling("os_system_dunder_dict", 'import os\nos.__dict__["system"](x)', "os.system"),
    Spelling(
        "os_system_importlib",
        'import importlib\nimportlib.import_module("os").system(x)',
        "os.system",
    ),
    Spelling(
        "os_system_import_module_bare",
        'from importlib import import_module\nimport_module("os").system(x)',
        "os.system",
    ),
    Spelling(
        "os_system_importlib_alias",
        'import importlib as il\nil.import_module("os").system(x)',
        "os.system",
    ),
    # ── subprocess.run (dotted module sink) ───────────────────────────
    Spelling("subprocess_run_direct", "import subprocess\nsubprocess.run(x)", "subprocess.run"),
    Spelling("subprocess_run_alias", "import subprocess as sp\nsp.run(x)", "subprocess.run"),
    Spelling("subprocess_run_from", "from subprocess import run\nrun(x)", "subprocess.run"),
    Spelling(
        "subprocess_run_getattr",
        'import subprocess\ngetattr(subprocess, "run")(x)',
        "subprocess.run",
    ),
    Spelling(
        "subprocess_run_importlib",
        'import importlib\nimportlib.import_module("subprocess").run(x)',
        "subprocess.run",
    ),
    # ── dynamic-import siblings (→ __import__ class) ──────────────────
    Spelling(
        "importlib_dunder_import",
        'import importlib\nimportlib.__import__("os")',
        "__import__",
    ),
    Spelling(
        "importlib_find_spec",
        'import importlib.util\nimportlib.util.find_spec("os")',
        "__import__",
    ),
    Spelling(
        "importlib_module_from_spec",
        "import importlib.util\nimportlib.util.module_from_spec(spec)",
        "__import__",
    ),
    Spelling("runpy_run_module", 'import runpy\nrunpy.run_module("os")', "__import__"),
    Spelling(
        "spec_loader_exec_module",
        "import importlib.util\n"
        "spec = importlib.util.module_from_spec(s)\n"
        "spec.loader.exec_module(mod)",
        "__import__",
    ),
    # ── code-exec siblings (→ exec class) ─────────────────────────────
    Spelling("code_interact", "import code\ncode.interact()", "exec"),
    Spelling(
        "code_runsource",
        "import code\ncode.InteractiveInterpreter().runsource(x)",
        "exec",
    ),
)

# Benign code that resembles a sink; must canonicalize to something OUTSIDE the sink sets.
FALSE_POSITIVE_NEIGHBOURS: tuple[Neighbour, ...] = (
    Neighbour("user_exec_helper", "from mymod import exec_helper\nexec_helper()"),
    Neighbour("getattr_benign_attr", 'val = getattr(cfg, "timeout")'),
    Neighbour(
        "importlib_benign_loads",
        'import importlib\nimportlib.import_module("json").loads(x)',
    ),
    Neighbour("subscript_user_dict", 'handlers = {}\nhandlers["exec"](x)'),
    Neighbour("getattr_non_dangerous_module", 'import os\ngetattr(os, "getcwd")()'),
    Neighbour("dunder_dict_benign_attr", 'import os\nos.__dict__["getcwd"]()'),
    Neighbour(
        "importlib_spec_from_file_benign",
        'import importlib.util\nimportlib.util.spec_from_file_location("m", "/x.py")',
    ),
    Neighbour("user_object_dunder_dict", 'obj = Thing()\nobj.__dict__["handler"](x)'),
    Neighbour(
        "user_class_exec_module",
        "class L:\n    def exec_module(self, m):\n        pass\n\n\nL().exec_module(1)",
    ),
    Neighbour("user_runsource_method", "ci = MyRepl()\nci.runsource(x)"),
    Neighbour("user_loader_unrelated", "loader = MyLoader()\nloader.exec_module(m)"),
)
