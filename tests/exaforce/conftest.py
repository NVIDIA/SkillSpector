import subprocess
import sys
import textwrap

import pytest


@pytest.fixture
def run_in_subprocess():
    """Run *code* in a fresh interpreter; return stdout, assert clean exit."""

    def _run(code: str) -> str:
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    return _run
