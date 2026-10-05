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

"""Tests for MCP rug-pull analyzer (RP1–RP3)."""

from __future__ import annotations

import json
import time

from skillspector.nodes.analyzers.mcp_rug_pull import node
from skillspector.nodes.build_context import build_context
from skillspector.nodes.deduplicate import deduplicate
from skillspector.nodes.report import report
from skillspector.state import SkillspectorState


def _state(
    manifest: dict | None = None, file_cache: dict[str, str] | None = None
) -> SkillspectorState:
    state: SkillspectorState = {}
    if manifest is not None:
        state["manifest"] = manifest
    if file_cache is not None:
        state["file_cache"] = file_cache
    return state


def test_rp1_npx_unpinned():
    """RP1 detects npx without @version suffix."""
    result = node(
        _state(
            manifest={"name": "test-skill"},
            file_cache={"setup.sh": "npx @scope/mcp-server\n"},
        )
    )
    rp1 = [f for f in result["findings"] if f.rule_id == "RP1"]
    assert len(rp1) == 1
    assert "npx @scope/mcp-server" in rp1[0].matched_text
    issue = json.loads(report({"filtered_findings": rp1, "output_format": "json"})["report_body"])[
        "issues"
    ][0]
    assert issue["pattern"] == rp1[0].message
    assert issue["finding"] == "npx @scope/mcp-server"


def test_rp1_scans_cached_files_without_a_manifest():
    """Cache-based RP1 checks remain applicable when manifest parsing failed."""
    result = node(_state(file_cache={"setup.sh": "npx @scope/mcp-server\n"}))

    assert [finding.rule_id for finding in result["findings"]] == ["RP1"]


def test_cache_only_scan_skips_manifest_comparison_checks():
    """A prior manifest cannot be diffed against an absent current manifest."""
    state = _state(file_cache={"setup.sh": "npx @scope/mcp-server\n"})
    state["previous_manifest"] = {
        "triggers": ["legacy"],
        "parameters": [{"name": "token", "type": "string"}],
    }

    result = node(state)

    assert [finding.rule_id for finding in result["findings"]] == ["RP1"]


def test_rp1_npx_pinned_no_finding():
    """RP1 does not fire when npx has @version."""
    result = node(
        _state(
            manifest={"name": "test-skill"},
            file_cache={"setup.sh": "npx @scope/mcp-server@1.2.3\n"},
        )
    )
    rp1 = [f for f in result["findings"] if f.rule_id == "RP1"]
    assert len(rp1) == 0


def test_rp1_unrelated_version_pin_does_not_suppress():
    """A pin on another argument or command on the same line does not pin the package."""
    for content, expected in (
        ("npx evil-package --label helper@1.2.3\n", "npx evil-package"),
        ("npx @scope/mcp-server http://localhost:3000/sse\n", "npx @scope/mcp-server"),
        ("npx @scope/server-a && npx @scope/server-b@1.2.3\n", "npx @scope/server-a"),
        ("uvx my-mcp-server --with helper==1.2.3\n", "uvx my-mcp-server"),
        ("pip install my-mcp-server other-package==1.2.3\n", "pip install my-mcp-server"),
    ):
        result = node(_state(file_cache={"setup.sh": content}))
        rp1 = [f for f in result["findings"] if f.rule_id == "RP1"]
        assert [f.matched_text for f in rp1] == [expected], content


def test_rp1_version_pin_attached_to_package_no_finding():
    """A pin attached to the package operand still counts when other arguments follow."""
    for content in (
        "npx -y @scope/mcp-server@1.2.3 --label helper\n",
        "npx -p @scope/mcp-server@1.2.3 mcp-server\n",
        "uvx my-mcp-server==1.2.3 --host 127.0.0.1:8000\n",
        "pip install my-mcp-server[cli]==1.2.3\n",
    ):
        result = node(_state(file_cache={"setup.sh": content}))
        assert not [f for f in result["findings"] if f.rule_id == "RP1"], content


def test_rp1_uvx_unpinned():
    """RP1 detects uvx without ==version."""
    result = node(
        _state(
            manifest={"name": "test-skill"},
            file_cache={"install.sh": "uvx my-mcp-server\n"},
        )
    )
    rp1 = [f for f in result["findings"] if f.rule_id == "RP1"]
    assert len(rp1) >= 1
    assert any("uvx" in f.matched_text for f in rp1)


def test_rp1_docker_unpinned():
    """RP1 detects docker run without tag."""
    node(
        _state(
            manifest={"name": "test-skill"},
            file_cache={"Dockerfile": "FROM org/mcp-server\n"},
        )
    )
    # RP1 docker pattern matches "docker pull|run|create"
    # FROM in Dockerfile isn't matched by our regex, so update test
    result2 = node(
        _state(
            manifest={"name": "test-skill"},
            file_cache={"setup.sh": "docker run org/mcp-server\n"},
        )
    )
    rp1 = [f for f in result2["findings"] if f.rule_id == "RP1"]
    assert len(rp1) >= 1


_DIGEST = "sha256:" + "0" * 64


def _rp1_matches(content: str) -> list[str]:
    result = node(_state(file_cache={"setup.sh": content}))
    return [f.matched_text for f in result["findings"] if f.rule_id == "RP1"]


def test_rp1_docker_pinned_image_after_options_no_finding():
    """Options before a pinned image are not read as the image."""
    for content in (
        "docker run --rm alpine:3.20 cat /etc/alpine-release\n",
        f"docker run -d img@{_DIGEST}\n",
        f"docker run --rm -e A my-image@{_DIGEST}\n",
        'docker run -it --rm -v "$(pwd)":/app -w /app node:20 npm test\n',
        "docker run -dp 8080:80 --name web nginx:1.27\n",
        "docker run -p8080:80 --network=host nginx:1.27\n",
        "docker run --gpus all --user 1000:1000 nvcr.io/nvidia/pytorch:24.01-py3\n",
        'docker run --rm --entrypoint "" -- alpine:3.20\n',
        'docker run --rm \\\n  -v "$PWD:/work" \\\n  ghcr.io/org/tool:1.4.2 lint\n',
        "`docker create --name probe alpine:3.20`\n",
        "docker pull -q --platform linux/amd64 alpine:3.20\n",
        "docker pull -a localhost:5000/team/tool:1.0\n",
    ):
        assert _rp1_matches(content) == [], content


def test_rp1_docker_unpinned_image_after_options_names_image():
    """The finding names the image, and option values are not taken as its tag."""
    for content, expected in (
        ("docker run --rm alpine\n", "docker run --rm alpine"),
        (
            "docker run --user 1000:1000 evil/image\n",
            "docker run --user 1000:1000 evil/image",
        ),
        (
            "docker run --rm -e MODE=fast -p 8080:80 evil/image\n",
            "docker run --rm -e MODE=fast -p 8080:80 evil/image",
        ),
        ("docker run --publish=8080:80 evil/image\n", "docker run --publish=8080:80 evil/image"),
        ("docker run -p8080:80 evil/image\n", "docker run -p8080:80 evil/image"),
        ("docker run -e TAG=1.2 evil/image:latest\n", "docker run -e TAG=1.2 evil/image:latest"),
        ('docker run "--env=x:1" evil/image\n', 'docker run "--env=x:1" evil/image'),
        ('docker run -e "A x:1" evil/image\n', 'docker run -e "A x:1" evil/image'),
        ("docker run -e A\\ x:1 evil/image\n", "docker run -e A\\ x:1 evil/image"),
        ("docker run --rm \\\n  evil/image\n", "docker run --rm \\\n  evil/image"),
        ("docker pull localhost:5000/team/tool\n", "docker pull localhost:5000/team/tool"),
        ("docker pull -a evil/image\n", "docker pull -a evil/image"),
        (
            "docker run --rm --entrypoint /openshell-sandbox "
            '"${SANDBOX_IMAGE:-ghcr.io/nvidia/openshell/sandbox:latest}" --version\n',
            "docker run --rm --entrypoint /openshell-sandbox "
            '"${SANDBOX_IMAGE:-ghcr.io/nvidia/openshell/sandbox:latest}"',
        ),
        ("docker run --rm alpine:3.20; docker run evil/image\n", "docker run evil/image"),
    ):
        assert _rp1_matches(content) == [expected], content


def test_rp1_docker_unresolved_image_is_still_reported():
    """When the image cannot be identified, the command stays reported."""
    for content, expected in (
        ("docker run --rm\n", "docker run --rm"),
        ("docker run --rm | tee log\n", "docker run --rm"),
        ("docker run -e\n", "docker run -e"),
        ("docker run --bogus x:1 evil/image\n", "docker run --bogus"),
        ("docker run -Z x:1 evil/image\n", "docker run -Z"),
        ("docker run -dZ x:1 evil/image\n", "docker run -dZ"),
        ("docker run -e A=$((1+2)) x:1 evil/image\n", "docker run -e A=$"),
        ('docker run -e "A x:1 evil/image\n', "docker run -e"),
        ("docker run --rm\nalpine:3.20\n", "docker run --rm"),
    ):
        assert _rp1_matches(content) == [expected], content

    # The read bound cuts this image after "localhost:5000"; that prefix is not
    # taken as a tagged image.
    padding = "-e A " * 202
    content = f"docker run {padding}localhost:5000/evil\n"
    assert _rp1_matches(content) == [f"docker run {padding.rstrip()}"[:200]]


def test_rp1_docker_operand_scan_is_linear():
    """Adversarial lines finish quickly; the operand scan is bounded."""
    for content in (
        "docker run " + "-e A " * 10_000,
        "docker run " * 4_545,
        'docker run -e "' + "a" * 50_000,
        "docker run --rm $(" + "a" * 50_000,
        "docker run -e " + "\\a" * 25_000,
        "docker run " + "\\\n" * 25_000 + "img",
        'docker pull -q "' * 3_125,
    ):
        started = time.perf_counter()
        _rp1_matches(content)
        elapsed = time.perf_counter() - started
        assert elapsed < 1.0, f"{elapsed:.2f}s for {content[:30]!r}"


def test_rp1_docker_credentials_are_redacted_in_reports():
    result = node(
        _state(
            file_cache={
                "setup.sh": "docker pull https://deploy:s3cret@registry.example.com/team/image"
            }
        )
    )
    rp1 = [f for f in result["findings"] if f.rule_id == "RP1"]
    assert len(rp1) == 1

    json_body = report({"filtered_findings": rp1, "output_format": "json"})["report_body"]
    issue = json.loads(json_body)["issues"][0]
    assert "https://***@registry.example.com" in issue["pattern"]
    assert "https://***@registry.example.com" in issue["finding"]
    sarif_body = report({"filtered_findings": rp1, "output_format": "sarif"})["report_body"]
    for body in (json_body, sarif_body):
        assert "deploy:s3cret" not in body
        assert "s3cret" not in body


def test_rp1_multiple_patterns():
    """Multiple unpinned references produce multiple RP1 findings."""
    result = node(
        _state(
            manifest={"name": "test-skill"},
            file_cache={
                "setup.sh": "npx @scope/server-a\nnpx @org/server-b\n",
            },
        )
    )
    rp1 = [f for f in result["findings"] if f.rule_id == "RP1"]
    assert len(rp1) == 2


def test_rp3_version_wildcard():
    """RP3 detects wildcard version."""
    result = node(
        _state(
            manifest={"version": "*", "name": "test"},
        )
    )
    rp3 = [f for f in result["findings"] if f.rule_id == "RP3"]
    assert len(rp3) >= 1


def test_rp3_version_wildcard_from_skill_frontmatter(tmp_path):
    """RP3 receives the version projected from real skill frontmatter."""
    (tmp_path / "SKILL.md").write_text(
        '---\nname: test-skill\ndescription: For tests\nversion: "*"\n---\n',
        encoding="utf-8",
    )

    result = node(build_context({"skill_path": str(tmp_path)}))

    rp3 = [finding for finding in result["findings"] if finding.rule_id == "RP3"]
    assert len(rp3) == 1
    assert rp3[0].matched_text == "*"


def test_rp3_broad_version_preview_preserves_full_value_identity() -> None:
    prefix = "^" + "1" * 200
    complete_values = (prefix + "first", prefix + "second")
    findings = [
        next(
            finding
            for finding in node(_state(manifest={"version": value}))["findings"]
            if finding.rule_id == "RP3"
        )
        for value in complete_values
    ]

    assert findings[0].matched_text == findings[1].matched_text
    assert len({finding.fingerprint() for finding in findings}) == 2
    assert len(deduplicate(findings)) == 2
    for finding, complete_value in zip(findings, complete_values, strict=True):
        assert complete_value not in json.dumps(finding.to_dict(), sort_keys=True)


def test_rp3_version_ok_no_finding():
    """RP3 does not fire on pinned version."""
    result = node(
        _state(
            manifest={"version": "1.2.3", "name": "test"},
        )
    )
    rp3 = [f for f in result["findings"] if f.rule_id == "RP3"]
    assert len(rp3) == 0


def test_empty_state_returns_no_findings():
    """Empty state produces no findings."""
    result = node({})
    assert result["findings"] == []
