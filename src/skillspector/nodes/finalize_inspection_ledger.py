# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-node adapter for canonical inspection-ledger finalization."""

from __future__ import annotations

from skillspector.inspection_ledger import finalize_ledger
from skillspector.state import SkillspectorState


def finalize_inspection_ledger(state: SkillspectorState) -> dict[str, object]:
    """Validate full internal facts and derive the public completeness projection."""
    completeness, effective_finding_ids = finalize_ledger(state)
    return {
        "analysis_completeness": completeness,
        "execution_successful": completeness["execution_successful"],
        "effective_finding_ids": effective_finding_ids,
    }
