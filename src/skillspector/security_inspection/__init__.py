"""
NVIDIA Skill Inspector - Security Inspection Plugin
100% local, offline, deterministic static analysis.
"""
__version__ = "1.0.0"
__author__ = "Skill Inspector Security Plugin"

from .config import get_data_dir, get_db_path
from .scanner import SecurityScanner

__all__ = ["SecurityScanner", "get_data_dir", "get_db_path"]
