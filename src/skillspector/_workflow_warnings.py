# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configure workflow warnings without loading LangChain for static imports."""

import warnings

import langchain_core  # noqa: F401  (initialize its warning filters before ours)

# LangGraph imports serializers with an allowed_objects default. Keep this
# targeted filter after LangChain installs its warning filters.
warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change",
    category=Warning,
)
