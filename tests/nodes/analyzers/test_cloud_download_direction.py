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

"""Only proven, non-destructive cloud downloads can lower E5 evidence."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.nodes.analyzers import static_patterns_data_exfiltration as data_exfiltration
from skillspector.nodes.analyzers import static_runner


@pytest.mark.parametrize(
    "command",
    [
        "aws s3 cp s3://example-bucket/public.csv ./public.csv",
        "aws s3 sync s3://example-bucket/public ./public",
        "gsutil cp gs://example-bucket/public.csv ./public.csv",
        "gsutil rsync gs://example-bucket/public ./public",
        "aws s3 cp s3://example-bucket/public.csv public.csv",
        "aws s3 cp s3://example-bucket/public.csv /tmp/public.csv",
        "aws s3 cp --region us-west-2 s3://example-bucket/public.csv ./public.csv",
        "aws s3 cp s3://example-bucket/public.csv ./public.csv --only-show-errors",
        "aws s3 cp --profile public --no-sign-request s3://example-bucket/public.csv ./public.csv",
        "aws s3 cp --recursive s3://example-bucket/public ./public",
        "gsutil cp -r gs://example-bucket/public ./public",
        "gsutil rsync -r gs://example-bucket/public ./public",
        'aws s3 cp "s3://example-bucket/public data.csv" "./public data.csv"',
        "aws s3 cp \\\n  s3://example-bucket/public.csv \\\n  ./public.csv",
        "aws s3 cp \\\r\n  s3://example-bucket/public.csv \\\r\n  ./public.csv",
        "  aws s3 cp s3://example-bucket/public.csv ./public.csv\n",
    ],
)
def test_literal_cloud_download_does_not_report_exfiltration(command: str) -> None:
    findings = data_exfiltration.analyze(command, "download.sh", "shell")
    assert not any(finding.rule_id == "E5" for finding in findings)


@pytest.mark.parametrize(
    "command",
    [
        "aws s3 cp ./public.csv s3://example-bucket/public.csv",
        "aws s3 sync ./public s3://example-bucket/public",
        "gsutil cp ./public.csv gs://example-bucket/public.csv",
        "gsutil rsync ./public gs://example-bucket/public",
        "aws s3 cp - s3://example-bucket/public.csv",
        "aws s3 cp s3://one-bucket/public.csv s3://another-bucket/public.csv",
        "gsutil cp gs://one-bucket/public.csv gs://another-bucket/public.csv",
        "aws s3 mv s3://example-bucket/public.csv ./public.csv",
        "gsutil mv gs://example-bucket/public.csv ./public.csv",
        "aws s3 sync --delete s3://example-bucket/public ./public",
        "gsutil rsync -d gs://example-bucket/public ./public",
        "aws s3 cp s3://example-bucket/public.csv -",
        'aws s3 cp "$source" "$destination"',
        'aws s3 cp s3://example-bucket/public.csv "$destination"',
        "aws s3 cp s3://example-bucket/public.csv ./public.csv --unknown-option",
        "aws s3 cp s3://example-bucket/public.csv ./public.csv --region",
        "aws s3 cp s3://example-bucket/public.csv ./public.csv --region=us-west-2",
        "aws s3 cp s3://example-bucket/public.csv ./public.csv extra.csv",
        "aws s3 cp s3://example-bucket/public.csv './public.csv",
        "aws s3 cp s3://example-bucket/public.csv ./public.csv; echo done",
        "aws s3 cp s3://example-bucket/public.csv ./public.csv | cat",
        "aws s3 cp s3://example-bucket/public.csv ./public.csv && echo done",
        "aws s3 cp s3://example-bucket/public.csv ./public.csv > ./log",
        "(aws s3 cp s3://example-bucket/public.csv ./public.csv)",
        "{ aws s3 cp s3://example-bucket/public.csv ./public.csv; }",
        'echo "$(aws s3 cp s3://example-bucket/public.csv ./public.csv)"',
        'aws s3 cp s3://example-bucket/public.csv "./public$(echo name).csv"',
        "aws s3 cp s3://example-bucket/public.csv ./public`echo name`.csv",
        "AWS S3 CP s3://example-bucket/public.csv ./public.csv",
        "aws s3 cp s3:///public.csv ./public.csv",
        "aws s3 cp s3://example-bucket/public.csv https://example.invalid/public.csv",
        "aws s3 cp s3://example-bucket/public.csv ./public.csv # trailing comment",
        "aws s3 cp s3://example-bucket/public.csv " + "x" * 4200,
        "aws s3 cp s3://example-bucket/public.csv //server/share/public.csv",
        r"aws s3 cp s3://example-bucket/public.csv \\server\share\public.csv",
        "aws s3 cp s3://example-bucket/public.csv /dev/stdout",
        "aws s3 cp s3://example-bucket/public.csv /tmp/../dev/stdout",
        "aws s3 cp s3://example-bucket/public.csv /proc/self/fd/1",
        "aws s3 cp s3://example-bucket/public.csv %DEST%",
        "aws s3 cp s3://example-bucket/public.csv !DEST!",
        "aws s3 cp s3://example-bucket/public.csv ~/public.csv",
        "(\naws s3 cp s3://example-bucket/public.csv ./public.csv\n)",
        "{\naws s3 cp s3://example-bucket/public.csv ./public.csv\n}",
        'echo "$(\naws s3 cp s3://example-bucket/public.csv ./public.csv\n)"',
    ],
)
def test_upload_or_unproved_direction_keeps_cloud_finding(command: str) -> None:
    findings = data_exfiltration.analyze(command, "transfer.sh", "shell")
    assert any(finding.rule_id == "E5" for finding in findings)


@pytest.mark.parametrize(
    "content, expected_lines",
    [
        (
            "aws s3 cp s3://example-bucket/public.csv ./public.csv\n"
            "aws s3 cp ./public.csv s3://destination-bucket/public.csv\n",
            [2],
        ),
        (
            "aws s3 cp s3://example-bucket/public.csv ./public.csv; "
            "aws s3 cp ./public.csv s3://destination-bucket/public.csv",
            [1, 1],
        ),
        (
            "aws s3 cp s3://example-bucket/public.csv - | "
            "aws s3 cp - s3://destination-bucket/public.csv",
            [1, 1],
        ),
        (
            "aws s3 cp s3://example-bucket/public.csv "
            "./public$(aws s3 cp ./other.csv s3://destination-bucket/other.csv).csv",
            [1, 1],
        ),
    ],
)
def test_each_cloud_occurrence_retains_its_own_direction_and_location(
    content: str, expected_lines: list[int]
) -> None:
    findings = data_exfiltration.analyze(content, "transfer.sh", "shell")
    e5 = [finding for finding in findings if finding.rule_id == "E5"]
    assert [finding.location.start_line for finding in e5] == expected_lines


def test_runner_keeps_upload_after_a_download_without_changing_source_offsets() -> None:
    content = (
        "#!/bin/sh\n"
        "aws s3 cp s3://example-bucket/public.csv ./public.csv\n"
        "aws s3 cp ./public.csv s3://destination-bucket/public.csv\n"
    )
    findings = static_runner.run_static_patterns(
        {"components": ["transfer.sh"], "file_cache": {"transfer.sh": content}},
        [data_exfiltration],
    )
    e5 = [finding for finding in findings if finding.rule_id == "E5"]
    assert len(e5) == 1
    assert e5[0].start_line == 3
    assert e5[0].start_column == 0
    assert e5[0].matched_text == "aws s3 cp"


@pytest.mark.parametrize(
    "path, file_type",
    [
        ("download.ps1", "shell"),
        ("download.bat", "shell"),
        ("download.cmd", "shell"),
        ("helper.py", "python"),
        ("helper.js", "javascript"),
    ],
)
def test_other_host_language_syntax_keeps_legacy_cloud_evidence(path: str, file_type: str) -> None:
    findings = data_exfiltration.analyze(
        "aws s3 cp s3://example-bucket/public.csv ./public.csv", path, file_type
    )
    assert any(finding.rule_id == "E5" for finding in findings)


@pytest.mark.parametrize(
    "prefix",
    [
        "alias aws=custom\n",
        "alias 'aws=custom'\n",
        "al\\ias aws=custom\n",
        "al''ias aws=custom\n",
        'aws() { custom "$@"; }\n',
        'aws ()\n{\n custom "$@"\n}\n',
        'function aws { custom "$@"; }\n',
        'a\\ws() { custom "$@"; }\n',
        "source ./plugin.sh\n",
        ". ./plugin.sh\n",
        "\\. ./plugin.sh\n",
        "so''urce ./plugin.sh\n",
        'eval "$setup"\n',
        'trap "$setup" DEBUG\n',
        "PATH=./tools\n",
        "hash -p ./custom-aws aws\n",
        "builtin hash -p ./custom-aws aws\n",
        "hash aws=./custom-aws\n",
        "commands[aws]=./custom-aws\n",
        "commands=(aws ./custom-aws)\n",
        "path=./tools\n",
        "path+=./tools\n",
        "path[1]=./tools\n",
        "set -A path ./tools\n",
        "printf -v PATH ./tools\n",
        "read PATH\n",
    ],
)
def test_visible_command_redefinition_keeps_cloud_finding(prefix: str) -> None:
    findings = data_exfiltration.analyze(
        prefix + "aws s3 cp s3://example-bucket/public.csv ./public.csv",
        "download.sh",
        "shell",
    )
    assert any(finding.rule_id == "E5" for finding in findings)


@pytest.mark.parametrize("setup", ["cd /dev", "chdir /dev", "pushd /dev", "popd"])
@pytest.mark.parametrize(
    "transfer",
    [
        "aws s3 cp s3://example-bucket/public.csv",
        "aws s3 sync s3://example-bucket/public",
        "gsutil cp gs://example-bucket/public.csv",
        "gsutil rsync gs://example-bucket/public",
    ],
)
@pytest.mark.parametrize("destination", ["stdout", "./stdout", "../dev/stdout"])
def test_directory_changes_keep_relative_destination_unproved(
    setup: str, transfer: str, destination: str
) -> None:
    findings = data_exfiltration.analyze(
        f"{setup}\n{transfer} {destination}\n", "download.sh", "shell"
    )
    e5 = [finding for finding in findings if finding.rule_id == "E5"]
    assert len(e5) == 1
    assert e5[0].location.start_line == 2


@pytest.mark.parametrize(
    "transfer", ["aws s3 cp s3://example-bucket/a", "gsutil cp gs://example-bucket/a"]
)
@pytest.mark.parametrize("setup", ["cd /dev", "set -eu", "read filename"])
def test_proved_file_destination_survives_unrelated_setup(transfer: str, setup: str) -> None:
    findings = data_exfiltration.analyze(
        f"{setup}\n{transfer} /tmp/public.csv\n", "download.sh", "shell"
    )
    assert not any(finding.rule_id == "E5" for finding in findings)


def test_large_prefix_cannot_forget_an_earlier_alias() -> None:
    content = (
        "alias aws=custom\n# " + "x" * 4500 + "\n"
        "aws s3 cp s3://example-bucket/public.csv ./public.csv\n"
    )
    findings = static_runner.run_static_patterns(
        {"components": ["transfer.sh"], "file_cache": {"transfer.sh": content}},
        [data_exfiltration],
    )
    assert any(finding.rule_id == "E5" for finding in findings)


@pytest.mark.parametrize(
    "commands, expected_lines",
    [
        ("aws s3 cp s3://example-bucket/public.csv ./public.csv\n", []),
        ("set -eu\naws s3 cp s3://example-bucket/public.csv ./public.csv\n", []),
        (
            "hash -p ./custom-aws aws\naws s3 cp s3://example-bucket/public.csv ./public.csv\n",
            [3],
        ),
        ("cd /dev\naws s3 cp s3://example-bucket/public.csv stdout\n", [3]),
        ("cd /dev\ngsutil cp gs://example-bucket/public.csv stdout\n", [3]),
        ("cd /dev\naws s3 cp s3://example-bucket/public.csv /tmp/public.csv\n", []),
    ],
)
def test_real_cli_download_strict_static_policy(
    tmp_path: Path, commands: str, expected_lines: list[int]
) -> None:
    (tmp_path / "SKILL.md").write_text(
        "---\nname: public-data-helper\ndescription: Read a selected public data file.\n---\n"
        "# Public data\nRead [the helper](download.sh).\n"
    )
    (tmp_path / "download.sh").write_text("#!/bin/sh\n" + commands)
    result = CliRunner().invoke(
        app,
        [
            "scan",
            str(tmp_path),
            "--no-llm",
            "--format",
            "json",
            "--fail-on-findings",
            "--fail-on-incomplete",
            "--min-coverage",
            "100",
        ],
    )
    assert result.exit_code == bool(expected_lines), result.output
    report = json.loads(result.stdout)
    assert report["execution_successful"] is True
    assert report["analysis_completeness"]["is_complete"] is True
    assert [issue["id"] for issue in report["issues"]] == ["E5"] * len(expected_lines)
    assert [issue["location"]["start_line"] for issue in report["issues"]] == expected_lines
    assert all(issue["location"]["file"] == "download.sh" for issue in report["issues"])
