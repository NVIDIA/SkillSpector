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

"""Multilingual batch scan for SkillSpector.

Community-contributed tool for scanning directories of AI agent skills
in non-English languages.  Extends SkillSpector's built-in analyzers
with targeted LLM gap-fill for vulnerability categories that static
English-keyword regex rules cannot detect.

Public API
----------
- :func:`~.discovery.discover_skills`
- :func:`~.detection.detect_language`
- :func:`~.detection.detect_skill_language`
- :func:`~.annotation.is_language_compatible`
- :func:`~.annotation.annotate_findings`
- :func:`~.gap_fill.run_gap_fill`
- :func:`~.runner.run_one`
"""

from __future__ import annotations

# -- .env MUST load before any skillspector import.  Python imports
#    this __init__.py before executing the batch_scan module body;
#    without this early load, constants.py resolves the provider
#    with stale env vars.
try:
    import dotenv as _dotenv
except ImportError:
    pass
else:
    _dotenv.load_dotenv(_dotenv.find_dotenv(usecwd=True), override=True)

from .annotation import annotate_findings, is_language_compatible
from .api_pool import ApiKey, ApiKeyPool, PooledChatModel, create_api_key_pool_from_env
from .detection import detect_language, detect_skill_language
from .discovery import discover_skills
from .gap_fill import GapFillAnalyzer, GapFillFinding, GapFillResult, run_gap_fill
from .runner import run_one

__all__ = [
    "annotate_findings",
    "ApiKey",
    "ApiKeyPool",
    "create_api_key_pool_from_env",
    "detect_language",
    "detect_skill_language",
    "discover_skills",
    "GapFillAnalyzer",
    "GapFillFinding",
    "GapFillResult",
    "is_language_compatible",
    "PooledChatModel",
    "run_gap_fill",
    "run_one",
]
