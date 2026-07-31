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

"""Tests for the watcher module (skillspector watch)."""

from __future__ import annotations

import itertools
from pathlib import Path
from unittest.mock import patch

import pytest

from skillspector import watcher as watcher_mod
from skillspector.watcher import _compute_directory_hash, watch_directory


class TestDirectoryHash:
    def test_deterministic(self, tmp_path):
        (tmp_path / "SKILL.md").write_text("# hi\n", encoding="utf-8")
        assert _compute_directory_hash(tmp_path) == _compute_directory_hash(tmp_path)

    def test_changes_with_content(self, tmp_path):
        md = tmp_path / "SKILL.md"
        md.write_text("# hi\n", encoding="utf-8")
        before = _compute_directory_hash(tmp_path)
        md.write_text("# bye\n", encoding="utf-8")
        assert _compute_directory_hash(tmp_path) != before

    def test_ignores_untracked_files(self, tmp_path):
        (tmp_path / "SKILL.md").write_text("# hi\n", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("not watched\n", encoding="utf-8")
        tracked = _compute_directory_hash(tmp_path)
        (tmp_path / "notes.txt").write_text("still not watched\n", encoding="utf-8")
        assert _compute_directory_hash(tmp_path) == tracked


class TestWatchDirectory:
    def _run_watch(self, hashes, debounce) -> list[str]:
        """Patch time so each poll advances the clock by 0.5s and drive the hash
        from an injected sequence. Returns the collected callback arguments."""
        fake_time = [0.0]

        def fake_sleep(_: float) -> None:
            fake_time[0] += 0.5

        def fake_now() -> float:
            return fake_time[0]

        calls: list[str] = []

        def fake_hash(_directory: Path) -> str:
            return next(hashes)

        def cb(directory: str, **kwargs) -> None:
            calls.append(directory)
            raise KeyboardInterrupt  # ends the infinite watch loop

        with patch.object(watcher_mod.time, "sleep", side_effect=fake_sleep), patch.object(
            watcher_mod.time, "time", side_effect=fake_now
        ), patch.object(watcher_mod, "_compute_directory_hash", side_effect=fake_hash):
            with pytest.raises(KeyboardInterrupt):
                watch_directory(Path("."), cb, poll_interval=2.0, debounce=debounce)
        return calls

    def test_fires_after_debounce_of_stability(self):
        # initial hash, then one change, then stable forever
        hashes = iter(itertools.chain(["h0"], itertools.repeat("h1")))
        assert self._run_watch(hashes, debounce=1.0) == ["."]

    def test_debounce_restarts_on_continued_changes(self):
        """A scan must not fire debounce seconds after the FIRST change while
        edits are still landing; it waits for a quiet period after the LAST one."""
        # h0 (initial), h1 (change 1), h2 (change 2), then stable forever
        hashes = itertools.chain(["h0", "h1", "h2"], itertools.repeat("h2"))
        # With a 0.5s poll and 1.0s debounce, the second change lands at t=1.0,
        # so the scan fires only once the tree is stable for 1.0s after t=1.0.
        assert self._run_watch(hashes, debounce=1.0) == ["."]
