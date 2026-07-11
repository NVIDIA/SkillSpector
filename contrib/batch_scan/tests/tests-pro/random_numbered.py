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

"""Random order with numbered progress."""

from __future__ import annotations

import unittest, sys, time, random, os
from pathlib import Path

_project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_project_root))

loader = unittest.TestLoader()
all_tests = []


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            flatten(item)
        else:
            all_tests.append(item)


for mod in [
    "test_api_pool",
    "test_gap_fill",
    "test_runner_patches",
    "test_annotation",
]:
    flatten(
        loader.loadTestsFromName(
            f"contrib.batch_scan.tests.tests-pro.{mod}"
        )
    )

random.seed(42)
random.shuffle(all_tests)

total = len(all_tests)
print(f"Total: {total} tests")

t0 = time.perf_counter()
count = 0


class _NumberedResult(unittest.TestResult):
    def startTest(self, test):
        global count
        count += 1
        short = test.id().split(".")[-2] + "." + test.id().split(".")[-1]
        print(f"[{count}/{total}] {short}", flush=True)
        super().startTest(test)


r = unittest.TextTestRunner(verbosity=0, resultclass=_NumberedResult).run(
    unittest.TestSuite(all_tests)
)
dt = time.perf_counter() - t0
print(f"Time: {dt:.0f}s | {r.testsRun} run | {len(r.failures)} fail |", "PASS" if r.wasSuccessful() else "FAIL")
