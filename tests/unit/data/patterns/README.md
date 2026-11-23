<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Static pattern samples

These `.txt` files contain deliberately suspicious Python and shell snippets
used by `tests/unit/test_patterns.py`. They are test data: load them as UTF-8
text and pass them to the static analyzers. Do not execute or import them.

Keep their whitespace, including leading blank lines, intact because finding
locations depend on it. The tests still supply the original simulated filename
and language to the analyzer; the `.txt` suffix is only for storing the sample.

Keeping these samples outside the Python test module addresses the source-layout
concern in [#1088](https://github.com/NVIDIA/SkillSpector/issues/1088), but does not
guarantee antivirus software will stop flagging the corpus. See the
[contributor guidance](../../../../CONTRIBUTING.md#antivirus-alerts-on-test-fixtures).
