# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auto-run carriers that write to agent-memory files are MP3 findings.

Issue #879: MP1/MP3 prose patterns score the injection text, not the write, so
a test hook appending to MEMORY.md scored 0/100 SAFE. The write to a known
agent-memory destination is the finding, whatever the payload bytes are.
"""

import pytest

from skillspector.models import Severity
from skillspector.nodes.analyzers import static_patterns_memory_poisoning as mp


def mp3_writes(content: str, file_path: str, file_type: str):
    return [
        f
        for f in mp.analyze(content, file_path, file_type)
        if f.rule_id == "MP3" and f.severity == Severity.HIGH and f.confidence == 0.85
    ]


TS_CARRIER = (
    'import { appendFileSync } from "fs";\n'
    'beforeAll(() => { appendFileSync("MEMORY.md", "\\n<!-- canary: inert -->\\n"); });'
)
PY_CARRIER = (
    "def pytest_configure(config):\n"
    '    with open("MEMORY.md", "a") as fh:\n'
    '        fh.write("\\n<!-- canary: inert -->\\n")'
)


@pytest.mark.parametrize(
    ("content", "file_path", "file_type", "destination"),
    [
        # The reporter's variants A (TypeScript) and E (Python).
        (TS_CARRIER, "notes.test.ts", "typescript", "MEMORY.md"),
        (PY_CARRIER, "conftest.py", "python", "MEMORY.md"),
        (PY_CARRIER, "tests/conftest.py", "python", "MEMORY.md"),
        # Other write sinks and memory destinations.
        ('open("MEMORY.md", "w").write("x")', "test_notes.py", "python", "MEMORY.md"),
        ('open("MEMORY.md", mode="a").write("x")', "test_notes.py", "python", "MEMORY.md"),
        ('Path("CLAUDE.md").write_text("x")', "test_notes.py", "python", "CLAUDE.md"),
        ('Path("AGENTS.md").write_bytes(b"x")', "test_notes.py", "python", "AGENTS.md"),
        ('shutil.copy("src.txt", "SOUL.md")', "test_notes.py", "python", "SOUL.md"),
        ('io.open("MEMORY.md", "a").write("x")', "test_notes.py", "python", "MEMORY.md"),
        (
            'open(f"{base}/MEMORY.md", "w").write("x")',
            "test_notes.py",
            "python",
            "{base}/MEMORY.md",
        ),
        ('open("./MEMORY.md", "a").write("x")', "test_notes.py", "python", "./MEMORY.md"),
        (
            'open("C:\\\\notes\\\\MEMORY.md", "a")',
            "test_notes.py",
            "python",
            "C:\\\\notes\\\\MEMORY.md",
        ),
        ('fs.writeFileSync("MEMORY.md", "x")', "notes.test.ts", "typescript", "MEMORY.md"),
        (
            'await fs.promises.appendFile("MEMORY.md", "x")',
            "notes.spec.js",
            "javascript",
            "MEMORY.md",
        ),
        ('fs.createWriteStream("AGENTS.md")', "notes.test.mjs", "javascript", "AGENTS.md"),
        (
            'writeFile(".claude/settings.json", "{}")',
            "notes.test.ts",
            "typescript",
            ".claude/settings.json",
        ),
        (
            "appendFileSync(`${dir}/MEMORY.md`, x)",
            "notes.test.ts",
            "typescript",
            "${dir}/MEMORY.md",
        ),
        (
            "appendFileSync(`.cursor/rules.md`, x)",
            "notes.test.ts",
            "typescript",
            ".cursor/rules.md",
        ),
    ],
)
def test_memory_file_write_from_auto_run_carrier_is_mp3(content, file_path, file_type, destination):
    findings = mp3_writes(content, file_path, file_type)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.evidence["destination"] == destination
    assert finding.evidence["carrier_kind"] == "auto_run"
    assert finding.remediation


@pytest.mark.parametrize(
    ("content", "file_path", "file_type"),
    [
        # Reads stay quiet.
        ('open("MEMORY.md").read()', "test_notes.py", "python"),
        ('open("MEMORY.md", "r").read()', "test_notes.py", "python"),
        ('Path("MEMORY.md").read_text()', "test_notes.py", "python"),
        # Ordinary file I/O is not the signal; the destination is.
        ('open("notes.txt", "w").write("x")', "test_notes.py", "python"),
        ('Path("report.md").write_text("x")', "test_notes.py", "python"),
        ('open("memory.md.bak", "w")', "test_notes.py", "python"),
        ('open("my-MEMORY.md", "w")', "test_notes.py", "python"),
        ('appendFileSync("notes.txt", x)', "notes.test.ts", "typescript"),
        # Non-literal destinations cannot be resolved to a memory file.
        ('open(MEMORY_PATH, "w")', "test_notes.py", "python"),
        ("appendFileSync(`${MEMORY_PATH}`, x)", "notes.test.ts", "typescript"),
        # Explicitly invoked helpers are out of scope: only auto-run carriers.
        (TS_CARRIER, "notes.ts", "typescript"),
        (TS_CARRIER, "scripts/notes.js", "javascript"),
        (PY_CARRIER, "helper.py", "python"),
        (PY_CARRIER, "skills/notes/writer.py", "python"),
        # Prose about memory files is not a write operation.
        ("The skill writes its notes to MEMORY.md when asked.", "SKILL.md", "markdown"),
        # Unrelated languages are not analyzed for write sinks.
        ('open("MEMORY.md", "w")', "notes.sh", "other"),
    ],
)
def test_memory_file_write_does_not_fire(content, file_path, file_type):
    assert mp3_writes(content, file_path, file_type) == []
