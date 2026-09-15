"""
config.py - Offline-local configuration.
All data under ~/.skill-inspector/ (or SKILL_INSPECTOR_DATA_DIR env).
No cloud, no telemetry, no external calls.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_DIR_NAME = ".skill-inspector"
DEFAULT_DB_NAME = "skill_inspector.db"
DEFAULT_PORT = 8899


def get_data_dir() -> Path:
    custom = os.environ.get("SKILL_INSPECTOR_DATA_DIR")
    if custom:
        p = Path(custom).expanduser().resolve()
    else:
        p = Path.home() / DEFAULT_DATA_DIR_NAME
    p.mkdir(parents=True, exist_ok=True)
    # subdirectories
    for sub in ["scans", "reports", "sbom", "provenance"]:
        (p / sub).mkdir(parents=True, exist_ok=True)
    return p


def get_db_path() -> Path:
    return get_data_dir() / DEFAULT_DB_NAME


def get_reports_dir() -> Path:
    return get_data_dir() / "reports"


def get_sbom_dir() -> Path:
    return get_data_dir() / "sbom"


# Validation: ensure loopback only
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}
ALLOWED_BIND = "127.0.0.1"
