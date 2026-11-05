# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded operator settings for retained inspection work."""

import logging
import os

logger = logging.getLogger(__name__)


def bounded_capacity(name: str, default: int, maximum: int) -> int:
    """Read a positive decimal capacity without logging untrusted environment text."""
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    digits = value.lstrip("0")
    if (
        0 < len(digits) <= len(str(maximum))
        and digits.isascii()
        and digits.isdecimal()
    ):
        capacity = int(digits)
        if capacity <= maximum:
            return capacity
    logger.warning(
        "%s must be a decimal integer from 1 to %d; using default %d", name, maximum, default
    )
    return default
