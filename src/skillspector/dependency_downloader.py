# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disposable HTTP acquisition process, timed and reaped by its parent.

Only verified artifact bytes subsequently reach the restricted analysis worker.
This process performs no archive parsing or package code execution.
"""

from __future__ import annotations

import json
import sys

import httpx

from skillspector.dependency_inspection import DependencyInspectionError, _stream_artifact


def main() -> int:
    """Return raw bytes on success, otherwise an allowlisted failure exit code."""
    try:
        request = json.loads(sys.stdin.buffer.read(8192))
        blob = _stream_artifact(**request, usage={"bytes": 0})
        sys.stdout.buffer.write(blob)
        sys.stdout.buffer.flush()
    except DependencyInspectionError as exc:
        return {"runtime_limit": 3, "size_limit": 4}.get(exc.reason, 2)
    except (httpx.HTTPError, OSError, ValueError, TypeError):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
