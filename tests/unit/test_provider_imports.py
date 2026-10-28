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

"""Fresh-process regressions for lightweight provider metadata imports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_python(code: str, *, provider: str) -> subprocess.CompletedProcess[str]:
    env = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP")
        if key in os.environ
    }
    env.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"),
        SKILLSPECTOR_PROVIDER=provider,
        LANGSMITH_TRACING="false",
        LANGCHAIN_TRACING_V2="false",
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        env=env,
        check=False,
    )


@pytest.mark.parametrize("provider", ["nv_build", ""])
def test_metadata_import_does_not_load_chat_client_stack(provider: str) -> None:
    result = _run_python(
        "import sys\n"
        "from skillspector.constants import MODEL_CONFIG\n"
        "from skillspector.providers import get_metadata_provider\n"
        "metadata_provider = get_metadata_provider()\n"
        "assert MODEL_CONFIG\n"
        "assert metadata_provider.resolve_model()\n"
        "unexpected = [\n"
        "    name for name in (\n"
        "        'skillspector.providers.chat_models',\n"
        "        'langchain_openai',\n"
        "        'langchain_anthropic',\n"
        "        'openai',\n"
        "        'anthropic',\n"
        "    )\n"
        "    if name in sys.modules\n"
        "]\n"
        "assert not unexpected, f'unexpected eager imports: {unexpected}'\n",
        provider=provider,
    )

    assert result.returncode == 0, result.stderr
