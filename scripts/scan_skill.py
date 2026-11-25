#!/usr/bin/env python3
"""Run the SkillSpector CLI bundled with this skill from any working directory."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    """Pass all arguments to the locked, production-only SkillSpector environment."""
    uv = shutil.which("uv")
    if uv is None:
        print(
            "Error: skill-scanner requires uv. Install uv and retry; "
            "the bundled project requires Python 3.12 through 3.14.",
            file=sys.stderr,
        )
        return 2

    command = [
        uv,
        "run",
        "--project",
        str(SKILL_ROOT),
        "--frozen",
        "--no-dev",
        "skillspector",
        *sys.argv[1:],
    ]

    try:
        completed = subprocess.run(command, check=False)
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print(f"Error: unable to launch SkillSpector: {exc}", file=sys.stderr)
        return 2
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
