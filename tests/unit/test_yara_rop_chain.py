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
"""Regression tests for exploit_framework $rop_chain false positives.

Issue #605: the pwntools-oriented /ROP\\s*\\(.*elf\\)/ nocase string matched
the tail of Rust's `fn drop(&mut self)` (rop( + &mut s + elf)), flagging every
`impl Drop` as an exploit framework at HIGH severity. A leading word boundary
restricts the match to standalone ROP tokens.
"""

from __future__ import annotations

import pathlib

import yara

RULES_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "skillspector"
    / "yara_rules"
    / "hacktools.yar"
)

RUST_DROP_REPRO = b"""pub struct TempDir {
    path: std::path::PathBuf,
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
"""


def _compile() -> yara.Rules:
    return yara.compile(filepath=str(RULES_PATH))


def _matched_identifiers(rules: yara.Rules, data: bytes) -> set[str]:
    identifiers: set[str] = set()
    for match in rules.match(data=data):
        for string in match.strings:
            identifiers.add(string.identifier)
    return identifiers


def test_rop_chain_ignores_rust_drop_repro(tmp_path: pathlib.Path) -> None:
    """The issue #605 reproduction must not fire exploit_framework."""
    target = tmp_path / "drop.rs"
    target.write_bytes(RUST_DROP_REPRO)
    matched = _matched_identifiers(_compile(), target.read_bytes())
    assert "$rop_chain" not in matched


def test_rop_chain_still_matches_pwntools_usage() -> None:
    """True positives: standalone ROP(elf) tokens still fire $rop_chain."""
    rules = _compile()
    for sample in (
        b"from pwn import *\nrop = ROP(elf)\n",
        b"chain = rop (elf)\n",
        b"ROP  (elf)\n",
    ):
        assert "$rop_chain" in _matched_identifiers(rules, sample), sample


def test_plain_drop_impl_triggers_no_rop_chain_string(tmp_path: pathlib.Path) -> None:
    """A plain impl Drop file triggers no string of exploit_framework."""
    target = tmp_path / "drop.rs"
    target.write_bytes(RUST_DROP_REPRO)
    rules = _compile()
    assert not [m for m in rules.match(data=target.read_bytes()) if m.rule == "exploit_framework"]
