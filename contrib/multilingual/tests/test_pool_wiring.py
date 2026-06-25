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

"""Smoke test: verify PooledChatModel is wired into ALL LLM call paths.

Uses the deepseek_compat() context manager to apply patches only for
the duration of the test, then restore original state on exit.
"""

from __future__ import annotations

import sys
from pathlib import Path

# -- Windows Unicode support (emoji in print statements) --------------------
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# Ensure project root is on sys.path (test lives under contrib/multilingual/tests/)
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
import os

# -- Simulate multi-key env ------------------------------------------------
os.environ["SKILLSPECTOR_API_KEYS"] = (
    "sk-test1|https://api.openai.com/v1|gpt-5.4;"
    "sk-test2|https://api.openai.com/v1|gpt-5.4"
)

# -- Build pool ------------------------------------------------------------
from contrib.multilingual.api_pool import create_api_key_pool_from_env
pool = create_api_key_pool_from_env()
assert pool is not None, "2 keys should produce a pool"
print(f"✅ Pool created: {pool.keys_configured} keys")

# -- Scoped patches + pool wiring -----------------------------------------
from contrib.multilingual.runner import set_api_pool, deepseek_compat

with deepseek_compat():
    set_api_pool(pool)

    import skillspector.llm_utils as _llm_utils
    model = _llm_utils.get_chat_model(model="gpt-5.4")
    assert type(model).__name__ == "PooledChatModel", \
        f"get_chat_model should return PooledChatModel, got {type(model).__name__}"
    print(f"✅ get_chat_model → {type(model).__name__} (graph path)")

    from contrib.multilingual.gap_fill import GapFillAnalyzer
    analyzer = GapFillAnalyzer(language="zh", api_pool=pool)
    assert type(analyzer.chat_model).__name__ == "PooledChatModel"
    print(f"✅ GapFillAnalyzer → {type(analyzer.chat_model).__name__} (gap-fill path)")

# Patches restored here (context manager __exit__)

# -- Verify patches are actually restored ----------------------------------
import skillspector.llm_analyzer_base as _base
assert _base.LLMAnalyzerBase.__init__.__name__ != "_patched_base_init", \
    "Patches should be restored after context manager exit"
print("✅ Patches restored to originals (context manager exited)")

print("\n\U0001F389 All LLM paths go through ApiKeyPool now.")
