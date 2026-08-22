# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for marketplace-backed bundled hook source discovery."""

from __future__ import annotations

import json

import pytest

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers.bundled_execution_surface import node
from skillspector.state import SkillspectorState


def _hook_map(command: str) -> dict[str, object]:
    return {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": command}]}]}


def _state(cache: dict[str, str], components: list[str] | None = None) -> SkillspectorState:
    return {
        "components": components if components is not None else list(cache),
        "local_file_cache": cache,
        "file_cache": {},
    }


def _marketplace(plugins: list[object], *, metadata: dict[str, object] | None = None) -> str:
    payload: dict[str, object] = {
        "name": "catalog",
        "owner": {"name": "NVIDIA"},
        "plugins": plugins,
    }
    if metadata is not None:
        payload["metadata"] = metadata
    return json.dumps(payload)


def _plugin_entry(
    name: str = "demo", source: object = "./plugins/demo", **fields: object
) -> dict[str, object]:
    return {"name": name, "source": source, **fields}


def _frontmatter(command: str) -> str:
    return (
        "---\nhooks:\n  PreToolUse:\n    - hooks:\n        - type: command\n          command: "
        + command
        + "\n---\n# Hook\n"
    )


@pytest.mark.parametrize(
    "marketplace_path",
    [
        "fake.claude-plugin/marketplace.json",
        "docs/fake.claude-plugin/marketplace.json",
        "bundle.zip!/fake.claude-plugin/marketplace.json",
        "bundle.zip!/docs/fake.claude-plugin/marketplace.json",
    ],
)
def test_marketplace_discovery_requires_exact_metadata_directory_segment(
    marketplace_path: str,
) -> None:
    """Suffix lookalikes cannot activate inline remote-plugin declarations."""
    remote_source = {"source": "github", "repo": "NVIDIA/demo"}
    content = _marketplace([_plugin_entry(source=remote_source, hooks=_hook_map("echo dormant"))])

    result = node(_state({marketplace_path: content}))

    assert result["findings"] == []
    assert result["inspection_ledger"] == []


@pytest.mark.parametrize(
    "marketplace_path",
    [
        ".claude-plugin/marketplace.json",
        "catalog/.claude-plugin/marketplace.json",
        "bundle.zip!/.claude-plugin/marketplace.json",
        "bundle.zip!/catalog/.claude-plugin/marketplace.json",
    ],
)
def test_exact_marketplace_metadata_paths_remain_active(marketplace_path: str) -> None:
    """Root and nested marketplaces remain active in project and archive namespaces."""
    content = _marketplace(
        [_plugin_entry(source=".", strict=False, hooks=_hook_map("echo active"))]
    )

    result = node(_state({marketplace_path: content}))

    assert [(finding.file, finding.evidence["source_kind"]) for finding in result["findings"]] == [
        (marketplace_path, "marketplace_plugin_inline")
    ]
    assert [(event["path"], event["outcome"]) for event in result["inspection_ledger"]] == [
        (marketplace_path, LedgerOutcome.COMPLETED)
    ]


@pytest.mark.parametrize(
    ("marketplace", "metadata", "source_root", "manifest_root"),
    [
        (
            "catalog/.claude-plugin/marketplace.json",
            None,
            "./plugins/demo",
            "catalog/plugins/demo",
        ),
        (
            "catalog/.claude-plugin/marketplace.json",
            {"pluginRoot": "./plugins"},
            ".",
            "catalog/plugins",
        ),
        (
            "bundle.zip!/catalog/.claude-plugin/marketplace.json",
            {"pluginRoot": "./plugins"},
            ".",
            "bundle.zip!/catalog/plugins",
        ),
    ],
)
def test_local_marketplace_sources_resolve_from_catalog_root_and_preserve_archives(
    marketplace: str,
    metadata: dict[str, object] | None,
    source_root: str,
    manifest_root: str,
) -> None:
    """Local sources use marketplace-root metadata and never cross a ZIP namespace."""
    manifest = f"{manifest_root}/.claude-plugin/plugin.json"
    default_hooks = f"{manifest_root}/hooks/hooks.json"
    outside_hooks = "plugins/demo/hooks/hooks.json"
    cache = {
        marketplace: _marketplace([_plugin_entry(source=source_root)], metadata=metadata),
        manifest: json.dumps({"name": "demo"}),
        default_hooks: json.dumps({"hooks": _hook_map("echo marketplace-default")}),
        outside_hooks: json.dumps({"hooks": _hook_map("echo outside")}),
    }

    result = node(_state(cache, components=[marketplace, manifest]))

    assert [finding.file for finding in result["findings"]] == [default_hooks]
    assert all(finding.file != outside_hooks for finding in result["findings"])
    assert result["findings"][0].evidence["source_kind"] == "plugin_default"


def test_marketplace_rejects_unsafe_local_sources_without_looking_up_escaped_paths() -> None:
    """Absolute, traversal, backslash, and cross-archive sources fail their entry only."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    valid_manifest = "catalog/plugins/valid/.claude-plugin/plugin.json"
    valid_hooks = "catalog/plugins/valid/hooks/hooks.json"
    unsafe_entries = [
        _plugin_entry(name="absolute", source="/tmp/plugin"),
        _plugin_entry(name="traversal", source="../outside"),
        _plugin_entry(name="windows", source=".\\outside"),
        _plugin_entry(name="archive", source="./other.zip!/plugin"),
    ]
    cache = {
        marketplace: _marketplace(
            [*unsafe_entries, _plugin_entry(name="valid", source="./plugins/valid")]
        ),
        valid_manifest: json.dumps({"name": "valid"}),
        valid_hooks: json.dumps({"hooks": _hook_map("echo valid")}),
        "outside/hooks/hooks.json": json.dumps({"hooks": _hook_map("echo escaped")}),
    }

    result = node(_state(cache, components=[marketplace, valid_manifest]))

    assert [finding.file for finding in result["findings"]] == [valid_hooks]
    failed = [
        event for event in result["inspection_ledger"] if event["outcome"] is LedgerOutcome.FAILED
    ]
    assert len(failed) == len(unsafe_entries)
    assert all(event["reason_code"] is LedgerReason.INVALID_CONFIGURATION for event in failed)


def test_strict_true_merges_marketplace_manifest_and_plugin_default_hooks() -> None:
    """Strict marketplace entries add hooks to the plugin manifest and default document."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    manifest = "catalog/plugins/demo/.claude-plugin/plugin.json"
    default_hooks = "catalog/plugins/demo/hooks/hooks.json"
    cache = {
        marketplace: _marketplace(
            [
                _plugin_entry(
                    strict=True,
                    hooks=_hook_map("echo marketplace"),
                )
            ]
        ),
        manifest: json.dumps({"name": "demo", "hooks": _hook_map("echo manifest")}),
        default_hooks: json.dumps({"hooks": _hook_map("echo default")}),
    }

    result = node(_state(cache, components=[marketplace, manifest, default_hooks]))

    assert {finding.file for finding in result["findings"]} == {
        marketplace,
        manifest,
        default_hooks,
    }
    assert any(
        finding.evidence["source_kind"] == "marketplace_plugin_inline"
        for finding in result["findings"]
    )


def test_strict_false_is_complete_and_conflicts_with_manifest_components() -> None:
    """A strict-false complete definition cannot be merged with manifest components."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    manifest = "catalog/plugins/demo/.claude-plugin/plugin.json"
    default_hooks = "catalog/plugins/demo/hooks/hooks.json"
    cache = {
        marketplace: _marketplace(
            [
                _plugin_entry(
                    strict=False,
                    hooks=_hook_map("echo complete"),
                    skills=["./skills"],
                )
            ]
        ),
        manifest: json.dumps(
            {"name": "demo", "hooks": _hook_map("echo manifest"), "skills": ["./other-skills"]}
        ),
        default_hooks: json.dumps({"hooks": _hook_map("echo default")}),
    }

    result = node(_state(cache, components=[marketplace, manifest, default_hooks]))

    assert result["findings"] == []
    assert any(
        event["outcome"] is LedgerOutcome.FAILED
        and event["reason_code"] is LedgerReason.INVALID_CONFIGURATION
        for event in result["inspection_ledger"]
    )


@pytest.mark.parametrize(
    ("component_field", "component_value"),
    [
        ("agents", "./agents"),
        ("mcpServers", {}),
        ("lspServers", {}),
        ("outputStyles", "./styles"),
        ("workflows", "./workflows"),
        ("experimental", {"themes": "./themes"}),
    ],
)
def test_strict_false_conflicts_with_every_manifest_component_family(
    component_field: str, component_value: object
) -> None:
    """A strict-false marketplace entry cannot coexist with any manifest component."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    manifest = "catalog/plugins/demo/.claude-plugin/plugin.json"
    cache = {
        marketplace: _marketplace([_plugin_entry(strict=False, hooks=_hook_map("marketplace"))]),
        manifest: json.dumps({"name": "demo", component_field: component_value}),
    }

    result = node(_state(cache, components=[marketplace, manifest]))

    assert result["findings"] == []
    assert any(
        event["outcome"] is LedgerOutcome.FAILED
        and event["reason_code"] is LedgerReason.INVALID_CONFIGURATION
        for event in result["inspection_ledger"]
    )


def test_strict_true_malformed_manifest_does_not_activate_plugin_defaults() -> None:
    """An invalid authority manifest makes that plugin incomplete rather than runnable."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    manifest = "catalog/plugins/demo/.claude-plugin/plugin.json"
    default_hooks = "catalog/plugins/demo/hooks/hooks.json"
    cache = {
        marketplace: _marketplace([_plugin_entry(strict=True)]),
        manifest: "{not-json",
        default_hooks: json.dumps({"hooks": _hook_map("must-not-activate")}),
    }

    result = node(_state(cache))

    assert result["findings"] == []
    assert any(
        event["path"] == manifest
        and event["outcome"] is LedgerOutcome.FAILED
        and event["reason_code"] is LedgerReason.INVALID_CONFIGURATION
        for event in result["inspection_ledger"]
    )


@pytest.mark.parametrize("manifest_content", ["{not-json", None])
def test_invalid_authority_manifest_suppresses_marketplace_hook_supplements(
    manifest_content: str | None,
) -> None:
    """Inline and referenced marketplace hooks cannot bypass an invalid manifest."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    manifest = "catalog/plugins/demo/.claude-plugin/plugin.json"
    referenced_hooks = "catalog/plugins/demo/hooks/custom.json"
    cache: dict[str, str | None] = {
        marketplace: _marketplace(
            [
                _plugin_entry(
                    strict=True,
                    hooks=[_hook_map("inline-must-not-run"), "./hooks/custom.json"],
                )
            ]
        ),
        manifest: manifest_content,
        referenced_hooks: json.dumps({"hooks": _hook_map("reference-must-not-run")}),
    }

    result = node(_state(cache))  # type: ignore[arg-type]

    assert result["findings"] == []
    failed = [
        event for event in result["inspection_ledger"] if event["outcome"] is LedgerOutcome.FAILED
    ]
    assert [(event["path"], event["reason_code"]) for event in failed] == [
        (
            manifest,
            LedgerReason.MISSING_FILE_CACHE
            if manifest_content is None
            else LedgerReason.INVALID_CONFIGURATION,
        )
    ]


def test_cached_invalid_manifest_is_authoritative_even_when_omitted_from_components() -> None:
    """Discovery cannot bypass a cached manifest merely through a sparse component list."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    manifest = "catalog/plugins/demo/.claude-plugin/plugin.json"
    cache = {
        marketplace: _marketplace(
            [_plugin_entry(strict=True, hooks=_hook_map("must-not-activate"))]
        ),
        manifest: "{not-json",
    }

    result = node(_state(cache, components=[marketplace]))

    assert result["findings"] == []
    assert [(event["path"], event["reason_code"]) for event in result["inspection_ledger"]] == [
        (manifest, LedgerReason.INVALID_CONFIGURATION)
    ]


def test_plugin_root_metadata_allows_bare_sources_relative_to_that_root() -> None:
    """metadata.pluginRoot permits the documented short source form without `./`."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    hooks = "catalog/plugins/demo/hooks/custom.json"
    cache = {
        marketplace: _marketplace(
            [
                _plugin_entry(
                    source="demo",
                    strict=False,
                    hooks="./hooks/custom.json",
                )
            ],
            metadata={"pluginRoot": "./plugins"},
        ),
        hooks: json.dumps({"hooks": _hook_map("bare-source")}),
    }

    result = node(_state(cache))

    assert [finding.file for finding in result["findings"]] == [hooks]


@pytest.mark.parametrize(
    ("marketplace", "manifest"),
    [
        (".claude-plugin/marketplace.json", ".claude-plugin/plugin.json"),
        (
            "bundle.zip!/.claude-plugin/marketplace.json",
            "bundle.zip!/.claude-plugin/plugin.json",
        ),
    ],
)
def test_strict_false_conflict_uses_canonical_root_and_archive_manifest_paths(
    marketplace: str, manifest: str
) -> None:
    """Root and archive namespaces must not gain leading or doubled separators."""
    cache = {
        marketplace: _marketplace(
            [_plugin_entry(source="./", strict=False, hooks=_hook_map("marketplace"))]
        ),
        manifest: json.dumps({"name": "demo", "hooks": _hook_map("manifest")}),
    }

    result = node(_state(cache))

    assert result["findings"] == []
    assert any(
        event["outcome"] is LedgerOutcome.FAILED
        and event["reason_code"] is LedgerReason.INVALID_CONFIGURATION
        for event in result["inspection_ledger"]
    )


def test_metadata_only_plugin_manifest_is_allowed_when_marketplace_declares_hooks() -> None:
    """A metadata-only manifest remains valid when the marketplace supplies the hook map."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    manifest = "catalog/plugins/demo/.claude-plugin/plugin.json"
    cache = {
        marketplace: _marketplace(
            [_plugin_entry(strict=False, hooks=_hook_map("echo marketplace-only"))]
        ),
        manifest: json.dumps({"name": "demo", "description": "metadata only"}),
    }

    result = node(_state(cache, components=[marketplace, manifest]))

    assert [finding.file for finding in result["findings"]] == [marketplace]
    assert result["findings"][0].evidence["source_kind"] == "marketplace_plugin_inline"
    assert not any(
        event["path"] == manifest and event["outcome"] is LedgerOutcome.FAILED
        for event in result["inspection_ledger"]
    )


def test_remote_marketplace_source_is_incomplete_but_retains_inline_marketplace_hooks() -> None:
    """An unmappable remote source is visible as incomplete without dropping inline hooks."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    cache = {
        marketplace: _marketplace(
            [
                _plugin_entry(
                    name="remote",
                    source={"source": "github", "repo": "example/remote"},
                    hooks=_hook_map("echo inline-retained"),
                )
            ]
        )
    }

    result = node(_state(cache, components=[marketplace]))

    assert [finding.file for finding in result["findings"]] == [marketplace]
    assert result["findings"][0].evidence["source_kind"] == "marketplace_plugin_inline"
    assert any(
        event["outcome"] is LedgerOutcome.FAILED
        and event["reason_code"] is LedgerReason.MISSING_FILE_CACHE
        for event in result["inspection_ledger"]
    )


def test_missing_local_marketplace_source_is_a_visible_incomplete_analysis() -> None:
    """An unresolved local plugin root must not produce a not-applicable false SAFE."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    cache = {marketplace: _marketplace([_plugin_entry(source="./missing")])}

    result = node(_state(cache, components=[marketplace]))

    assert result["findings"] == []
    assert len(result["inspection_ledger"]) == 1
    event = result["inspection_ledger"][0]
    assert event["path"] == f"{marketplace}#plugin[0]"
    assert event["outcome"] is LedgerOutcome.FAILED
    assert event["reason_code"] is LedgerReason.MISSING_FILE_CACHE
    assert result["analyzer_status_events"][0]["status"] == "failed"


def test_multiple_inline_entries_share_one_document_without_losing_handlers() -> None:
    """Physical-document dedupe aggregates, rather than drops, per-entry declarations."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    cache = {
        marketplace: _marketplace(
            [
                _plugin_entry(name="first", source="./plugins/first", hooks=_hook_map("one")),
                _plugin_entry(name="second", source="./plugins/second", hooks=_hook_map("two")),
            ]
        ),
        "catalog/plugins/first/.claude-plugin/plugin.json": json.dumps({"name": "first"}),
        "catalog/plugins/second/.claude-plugin/plugin.json": json.dumps({"name": "second"}),
    }

    result = node(_state(cache))

    assert [finding.file for finding in result["findings"]] == [marketplace]
    assert result["findings"][0].evidence["handler_count"] == 2
    assert [event["path"] for event in result["inspection_ledger"]].count(marketplace) == 1


def test_inline_entries_exceeding_shared_document_cap_fail_without_partial_bh1() -> None:
    """The marketplace physical document commits all inline entries or none of them."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    handlers = [{"type": "command", "command": "echo safe"} for _ in range(1_025)]
    hook_map = {"PostToolUse": [{"matcher": "Bash", "hooks": handlers}]}
    cache = {
        marketplace: _marketplace(
            [
                _plugin_entry(name="first", source="./plugins/first", strict=False, hooks=hook_map),
                _plugin_entry(
                    name="second", source="./plugins/second", strict=False, hooks=hook_map
                ),
            ]
        ),
        "catalog/plugins/first/README.md": "first plugin\n",
        "catalog/plugins/second/README.md": "second plugin\n",
    }

    result = node(_state(cache, components=[marketplace]))

    assert result["findings"] == []
    assert [
        (event["path"], event["outcome"], event.get("reason_code"))
        for event in result["inspection_ledger"]
    ] == [(marketplace, LedgerOutcome.FAILED, LedgerReason.COMPONENT_LIMIT)]


def test_marketplace_inline_and_manifest_reference_roles_share_physical_document() -> None:
    """Top-level referenced hooks and plugin-entry hooks both remain inventoried."""
    parent_manifest = ".claude-plugin/plugin.json"
    marketplace = "catalog/.claude-plugin/marketplace.json"
    marketplace_payload = json.loads(
        _marketplace(
            [
                _plugin_entry(
                    source="./plugins/demo",
                    strict=False,
                    hooks=_hook_map("echo marketplace inline"),
                )
            ]
        )
    )
    marketplace_payload["hooks"] = _hook_map("echo referenced top-level")
    cache = {
        parent_manifest: json.dumps(
            {"name": "parent", "hooks": "./catalog/.claude-plugin/marketplace.json"}
        ),
        marketplace: json.dumps(marketplace_payload),
        "catalog/plugins/demo/README.md": "plugin exists\n",
    }

    result = node(_state(cache))

    assert [finding.file for finding in result["findings"]] == [marketplace]
    finding = result["findings"][0]
    assert finding.evidence["handler_count"] == 2
    assert finding.evidence["declaration_roles"] == (
        "marketplace_plugin_inline,plugin_manifest_reference"
    )
    assert [(event["path"], event["outcome"]) for event in result["inspection_ledger"]] == [
        (marketplace, LedgerOutcome.COMPLETED)
    ]


def test_cross_role_marketplace_cap_fails_physical_document_transactionally() -> None:
    """Referenced and inline roles share one cap and cannot leave a partial BH1."""
    parent_manifest = ".claude-plugin/plugin.json"
    marketplace = "catalog/.claude-plugin/marketplace.json"
    handlers = [{"type": "command", "command": "echo safe"} for _ in range(1_025)]
    hook_map = {"PostToolUse": [{"matcher": "Bash", "hooks": handlers}]}
    marketplace_payload = json.loads(
        _marketplace(
            [
                _plugin_entry(
                    source="./plugins/demo",
                    strict=False,
                    hooks=hook_map,
                )
            ]
        )
    )
    marketplace_payload["hooks"] = hook_map
    cache = {
        parent_manifest: json.dumps(
            {"name": "parent", "hooks": "./catalog/.claude-plugin/marketplace.json"}
        ),
        marketplace: json.dumps(marketplace_payload),
        "catalog/plugins/demo/README.md": "plugin exists\n",
    }

    result = node(_state(cache))

    assert result["findings"] == []
    assert [
        (event["path"], event["outcome"], event.get("reason_code"))
        for event in result["inspection_ledger"]
    ] == [(marketplace, LedgerOutcome.FAILED, LedgerReason.COMPONENT_LIMIT)]


def test_remote_inline_overflow_retains_each_entry_incomplete_row() -> None:
    """A shared inline cap cannot conceal independent remote-source incompleteness."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    handlers = [{"type": "command", "command": "echo safe"} for _ in range(1_025)]
    hook_map = {"PostToolUse": [{"matcher": "Bash", "hooks": handlers}]}
    cache = {
        marketplace: _marketplace(
            [
                _plugin_entry(
                    name="first",
                    source={"source": "github", "repo": "example/first"},
                    hooks=hook_map,
                ),
                _plugin_entry(
                    name="second",
                    source={"source": "github", "repo": "example/second"},
                    hooks=hook_map,
                ),
            ]
        )
    }

    result = node(_state(cache, components=[marketplace]))

    assert result["findings"] == []
    terminal_rows = {
        (event["path"], event["outcome"], event.get("reason_code"))
        for event in result["inspection_ledger"]
    }
    assert terminal_rows == {
        (marketplace, LedgerOutcome.FAILED, LedgerReason.COMPONENT_LIMIT),
        (
            f"{marketplace}#plugin[0]",
            LedgerOutcome.FAILED,
            LedgerReason.MISSING_FILE_CACHE,
        ),
        (
            f"{marketplace}#plugin[1]",
            LedgerOutcome.FAILED,
            LedgerReason.MISSING_FILE_CACHE,
        ),
    }
    work_ids = [event["work_id"] for event in result["inspection_ledger"]]
    assert len(work_ids) == len(set(work_ids)) == 3


def test_failed_referenced_marketplace_role_suppresses_valid_inline_sibling_role() -> None:
    """A physical-path failure dominates later roles and keeps one terminal work row."""
    parent_manifest = ".claude-plugin/plugin.json"
    marketplace = "catalog/.claude-plugin/marketplace.json"
    marketplace_payload = json.loads(
        _marketplace(
            [
                _plugin_entry(
                    source="./plugins/demo",
                    strict=False,
                    hooks=_hook_map("echo must-not-run"),
                )
            ]
        )
    )
    marketplace_payload["hooks"] = 7
    cache = {
        parent_manifest: json.dumps(
            {"name": "parent", "hooks": "./catalog/.claude-plugin/marketplace.json"}
        ),
        marketplace: json.dumps(marketplace_payload),
        "catalog/plugins/demo/README.md": "plugin exists\n",
    }

    result = node(_state(cache))

    assert result["findings"] == []
    assert [
        (event["path"], event["outcome"], event.get("reason_code"))
        for event in result["inspection_ledger"]
    ] == [(marketplace, LedgerOutcome.FAILED, LedgerReason.INVALID_CONFIGURATION)]
    assert len({event["work_id"] for event in result["inspection_ledger"]}) == 1


def test_many_marketplace_roots_use_indexed_set_membership() -> None:
    """Marketplace-owned defaults and components avoid cross-root list scans."""

    class _CountingPath(str):
        comparisons = 0

        def __eq__(self, other: object) -> bool:
            type(self).comparisons += 1
            return super().__eq__(other)

        __hash__ = str.__hash__

    archive_count = 32
    cache: dict[str, str] = {}
    for index in range(archive_count):
        marketplace = _CountingPath(f"bundle-{index}.zip!/.claude-plugin/marketplace.json")
        default_hooks = _CountingPath(f"bundle-{index}.zip!/hooks/hooks.json")
        skill = _CountingPath(f"bundle-{index}.zip!/skills/review/SKILL.md")
        cache[marketplace] = _marketplace(
            [
                _plugin_entry(
                    name=f"demo-{index}",
                    source="./",
                    strict=False,
                    skills="./skills",
                )
            ]
        )
        cache[default_hooks] = json.dumps({"hooks": _hook_map("echo excluded default")})
        cache[skill] = _frontmatter(f"echo archive-{index}")

    _CountingPath.comparisons = 0
    result = node(_state(cache))
    comparisons = _CountingPath.comparisons

    assert len(result["findings"]) == archive_count
    assert comparisons < archive_count * 20


def test_marketplace_self_reference_keeps_inline_and_top_level_scopes() -> None:
    """Equal roots do not deduplicate distinct inline and top-level declarations."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    marketplace_payload = json.loads(
        _marketplace(
            [
                _plugin_entry(
                    source=".",
                    strict=False,
                    hooks=[
                        _hook_map("echo inline"),
                        "./.claude-plugin/marketplace.json",
                    ],
                )
            ],
            metadata={"pluginRoot": "."},
        )
    )
    marketplace_payload["hooks"] = _hook_map("echo top-level")

    result = node(_state({marketplace: json.dumps(marketplace_payload)}))

    assert [finding.file for finding in result["findings"]] == [marketplace]
    finding = result["findings"][0]
    assert finding.evidence["handler_count"] == 2
    assert finding.evidence["declaration_roles"] == (
        "marketplace_plugin_inline,marketplace_plugin_reference"
    )
    assert [(event["path"], event["outcome"]) for event in result["inspection_ledger"]] == [
        (marketplace, LedgerOutcome.COMPLETED)
    ]


def test_remote_mixed_inline_and_reference_retains_inline_with_one_incomplete_row() -> None:
    """An unmappable remote reference cannot discard a valid sibling inline declaration."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    cache = {
        marketplace: _marketplace(
            [
                _plugin_entry(
                    name="remote",
                    source={"source": "github", "repo": "example/remote"},
                    hooks=[_hook_map("inline"), "./hooks/remote.json"],
                )
            ]
        )
    }

    result = node(_state(cache, components=[marketplace]))

    assert [finding.file for finding in result["findings"]] == [marketplace]
    assert result["findings"][0].evidence["handler_count"] == 1
    entry_events = [
        event
        for event in result["inspection_ledger"]
        if event["path"] == f"{marketplace}#plugin[0]"
    ]
    assert len(entry_events) == 1
    assert entry_events[0]["outcome"] is LedgerOutcome.FAILED
    assert entry_events[0]["reason_code"] is LedgerReason.MISSING_FILE_CACHE


def test_invalid_marketplace_entry_does_not_suppress_valid_entry() -> None:
    """One malformed plugin entry has one failure while a sibling still produces BH1."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    valid_manifest = "catalog/plugins/valid/.claude-plugin/plugin.json"
    valid_hooks = "catalog/plugins/valid/hooks/hooks.json"
    cache = {
        marketplace: _marketplace(
            [
                {"name": "invalid", "source": 7},
                _plugin_entry(name="valid", source="./plugins/valid"),
            ]
        ),
        valid_manifest: json.dumps({"name": "valid"}),
        valid_hooks: json.dumps({"hooks": _hook_map("echo valid")}),
    }

    result = node(_state(cache, components=[marketplace, valid_manifest]))

    assert [finding.file for finding in result["findings"]] == [valid_hooks]
    assert (
        sum(
            event["outcome"] is LedgerOutcome.FAILED
            and event["reason_code"] is LedgerReason.INVALID_CONFIGURATION
            for event in result["inspection_ledger"]
        )
        == 1
    )


def test_invalid_entry_and_valid_inline_entry_have_distinct_terminal_work_ids() -> None:
    """Per-entry failures cannot collide with the marketplace document's completed row."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    manifest = "catalog/plugins/valid/.claude-plugin/plugin.json"
    cache = {
        marketplace: _marketplace(
            [
                {"name": "invalid", "source": 7},
                _plugin_entry(name="valid", source="./plugins/valid", hooks=_hook_map("valid")),
            ]
        ),
        manifest: json.dumps({"name": "valid"}),
    }

    result = node(_state(cache))

    assert [finding.file for finding in result["findings"]] == [marketplace]
    work_ids = [event["work_id"] for event in result["inspection_ledger"]]
    assert len(work_ids) == len(set(work_ids))
    assert any(event["path"] == f"{marketplace}#plugin[0]" for event in result["inspection_ledger"])


def test_synthetic_marketplace_entry_path_cannot_collide_with_cached_document() -> None:
    """Synthetic entry work identities disambiguate real cache keys deterministically."""
    parent_manifest = ".claude-plugin/plugin.json"
    marketplace = "catalog/.claude-plugin/marketplace.json"
    real_hook_path = f"{marketplace}#plugin[0]"
    cache = {
        parent_manifest: json.dumps({"name": "parent", "hooks": f"./{real_hook_path}"}),
        marketplace: _marketplace([{"name": "invalid", "source": 7}]),
        real_hook_path: json.dumps({"hooks": _hook_map("echo real cache document")}),
    }

    result = node(_state(cache))

    assert [finding.file for finding in result["findings"]] == [real_hook_path]
    assert [(event["path"], event["outcome"]) for event in result["inspection_ledger"]] == [
        (f"{real_hook_path}#ledger[1]", LedgerOutcome.FAILED),
        (real_hook_path, LedgerOutcome.COMPLETED),
    ]
    work_ids = [event["work_id"] for event in result["inspection_ledger"]]
    assert len(work_ids) == len(set(work_ids)) == 2


def test_synthetic_marketplace_entry_path_cannot_collide_with_missing_reference() -> None:
    """Synthetic identities also reserve uncached physical paths discovered by references."""
    parent_manifest = ".claude-plugin/plugin.json"
    marketplace = "catalog/.claude-plugin/marketplace.json"
    missing_hook_path = f"{marketplace}#plugin[0]"
    cache = {
        parent_manifest: json.dumps({"name": "parent", "hooks": f"./{missing_hook_path}"}),
        marketplace: _marketplace([{"name": "invalid", "source": 7}]),
    }

    result = node(_state(cache))

    assert result["findings"] == []
    assert [
        (event["path"], event["outcome"], event["reason_code"])
        for event in result["inspection_ledger"]
    ] == [
        (
            f"{missing_hook_path}#ledger[1]",
            LedgerOutcome.FAILED,
            LedgerReason.INVALID_CONFIGURATION,
        ),
        (missing_hook_path, LedgerOutcome.FAILED, LedgerReason.MISSING_FILE_CACHE),
    ]
    work_ids = [event["work_id"] for event in result["inspection_ledger"]]
    assert len(work_ids) == len(set(work_ids)) == 2


def test_marketplace_hook_references_are_deduplicated_by_physical_cache_path() -> None:
    """Repeated marketplace references produce one finding and one terminal work item."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    shared = "catalog/plugins/demo/hooks/shared.json"
    cache = {
        marketplace: _marketplace(
            [
                _plugin_entry(
                    strict=False,
                    hooks=["./hooks/shared.json", "./hooks/shared.json"],
                )
            ]
        ),
        shared: json.dumps({"hooks": _hook_map("echo shared")}),
    }

    result = node(_state(cache, components=[marketplace]))

    assert [finding.file for finding in result["findings"]] == [shared]
    assert [event["path"] for event in result["inspection_ledger"]].count(shared) == 1


def test_marketplace_components_add_skills_and_replace_default_commands() -> None:
    """Marketplace skills add to defaults while declared commands replace defaults."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    manifest = "catalog/plugins/demo/.claude-plugin/plugin.json"
    default_skill = "catalog/plugins/demo/skills/default/SKILL.md"
    default_command = "catalog/plugins/demo/commands/default.md"
    custom_skill = "catalog/plugins/demo/custom-skills/review/SKILL.md"
    custom_command = "catalog/plugins/demo/custom-commands/release.md"
    cache = {
        marketplace: _marketplace(
            [
                _plugin_entry(
                    strict=True,
                    skills="./custom-skills",
                    commands="./custom-commands",
                )
            ]
        ),
        manifest: json.dumps({"name": "demo"}),
        default_skill: _frontmatter("echo default-skill"),
        default_command: _frontmatter("echo default-command"),
        custom_skill: _frontmatter("echo custom-skill"),
        custom_command: _frontmatter("echo custom-command"),
    }

    result = node(_state(cache, components=[marketplace, manifest]))

    assert {finding.file for finding in result["findings"]} == {
        default_skill,
        custom_skill,
        custom_command,
    }
    assert default_command not in {finding.file for finding in result["findings"]}


def test_lowercase_skill_reached_by_marketplace_path_is_runtime_unconfirmed() -> None:
    """Marketplace overrides do not promote unsupported lowercase skill.md files."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    lowercase_skill = "catalog/plugins/demo/custom/skill.md"
    cache = {
        marketplace: _marketplace([_plugin_entry(strict=False, skills="./custom/skill.md")]),
        lowercase_skill: _frontmatter("echo lowercase"),
    }

    result = node(_state(cache, components=[marketplace]))

    assert [finding.file for finding in result["findings"]] == [lowercase_skill]
    finding = result["findings"][0]
    assert finding.evidence["source_kind"] == "marketplace_plugin_skill"
    assert finding.evidence["runtime_status"] == "runtime_unconfirmed"
    assert finding.evidence["runnable_handler_count"] == 0
    assert finding.evidence["ambient_handler_count"] == 0


def test_marketplace_root_source_with_specific_skills_replaces_shared_default_scan() -> None:
    """Specific skill paths isolate entries whose plugin source is the marketplace root."""
    marketplace = ".claude-plugin/marketplace.json"
    manifest = ".claude-plugin/plugin.json"
    shared_skill = "skills/shared/SKILL.md"
    selected_skill = "skills/demo/SKILL.md"
    cache = {
        marketplace: _marketplace(
            [
                _plugin_entry(
                    source="./",
                    strict=True,
                    skills="./skills/demo",
                )
            ]
        ),
        manifest: json.dumps({"name": "demo"}),
        shared_skill: _frontmatter("echo shared-must-not-load"),
        selected_skill: _frontmatter("echo selected"),
    }

    result = node(_state(cache))

    assert [finding.file for finding in result["findings"]] == [selected_skill]


def test_marketplace_root_skill_is_a_fallback_when_no_plugin_skill_directory_exists() -> None:
    """A marketplace plugin root SKILL.md is discovered when no skill directory is present."""
    marketplace = ".claude-plugin/marketplace.json"
    manifest = ".claude-plugin/plugin.json"
    root_skill = "SKILL.md"
    cache = {
        marketplace: _marketplace([_plugin_entry(source="./", strict=True)]),
        manifest: json.dumps({"name": "demo"}),
        root_skill: _frontmatter("echo marketplace-root-skill"),
    }

    result = node(_state(cache))

    assert [finding.file for finding in result["findings"]] == [root_skill]
    assert result["findings"][0].evidence["source_kind"] == "plugin_root_skill"


def test_nested_manifestless_marketplace_root_skill_is_a_plugin_fallback() -> None:
    """A strict local marketplace source can expose its root SKILL without a manifest."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    root_skill = "catalog/plugins/demo/SKILL.md"
    cache = {
        marketplace: _marketplace([_plugin_entry(strict=True)]),
        root_skill: _frontmatter("echo nested-marketplace-root-skill"),
    }

    result = node(_state(cache))

    assert [finding.file for finding in result["findings"]] == [root_skill]
    assert result["findings"][0].evidence["source_kind"] == "plugin_root_skill"


def test_marketplace_default_commands_are_used_when_commands_are_not_declared() -> None:
    """Strict marketplace entries without commands retain their plugin default commands."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    manifest = "catalog/plugins/demo/.claude-plugin/plugin.json"
    default_command = "catalog/plugins/demo/commands/release.md"
    cache = {
        marketplace: _marketplace([_plugin_entry(strict=True)]),
        manifest: json.dumps({"name": "demo"}),
        default_command: _frontmatter("echo marketplace-default-command"),
    }

    result = node(_state(cache))

    assert [finding.file for finding in result["findings"]] == [default_command]


def test_marketplace_specific_skills_fall_back_to_shared_defaults_when_all_are_missing() -> None:
    """An all-missing explicit skill selection retains the documented shared default fallback."""
    marketplace = ".claude-plugin/marketplace.json"
    manifest = ".claude-plugin/plugin.json"
    shared_skill = "skills/shared/SKILL.md"
    cache = {
        marketplace: _marketplace(
            [_plugin_entry(source="./", strict=True, skills="./skills/missing")]
        ),
        manifest: json.dumps({"name": "demo"}),
        shared_skill: _frontmatter("echo shared-fallback"),
    }

    result = node(_state(cache))

    assert [finding.file for finding in result["findings"]] == [shared_skill]


@pytest.mark.parametrize("skills_path", [".", "./"])
def test_marketplace_skills_accepts_documented_plugin_root_paths(skills_path: str) -> None:
    """The skills field may explicitly name the plugin root itself."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    root_skill = "catalog/plugins/demo/SKILL.md"
    cache = {
        marketplace: _marketplace(
            [
                _plugin_entry(
                    strict=False,
                    skills=skills_path,
                )
            ]
        ),
        root_skill: _frontmatter("echo root-skill"),
    }

    result = node(_state(cache))

    assert [finding.file for finding in result["findings"]] == [root_skill]


def test_marketplace_commands_accepts_dot_slash_but_rejects_bare_dot() -> None:
    """Only skills have the documented bare-dot root exception."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    root_command = "catalog/plugins/demo/release.md"
    base_cache = {root_command: _frontmatter("echo release")}

    accepted = node(
        _state(
            {
                marketplace: _marketplace([_plugin_entry(strict=False, commands="./")]),
                **base_cache,
            }
        )
    )
    rejected = node(
        _state(
            {
                marketplace: _marketplace([_plugin_entry(strict=False, commands=".")]),
                **base_cache,
            }
        )
    )

    assert [finding.file for finding in accepted["findings"]] == [root_command]
    assert rejected["findings"] == []
    assert any(
        event["reason_code"] is LedgerReason.INVALID_CONFIGURATION
        for event in rejected["inspection_ledger"]
    )


def test_strict_false_marketplace_components_are_complete_without_plugin_defaults() -> None:
    """Strict-false entries retain only their explicitly declared Markdown components."""
    marketplace = "catalog/.claude-plugin/marketplace.json"
    manifest = "catalog/plugins/demo/.claude-plugin/plugin.json"
    default_skill = "catalog/plugins/demo/skills/default/SKILL.md"
    default_command = "catalog/plugins/demo/commands/default.md"
    custom_skill = "catalog/plugins/demo/custom-skills/review/SKILL.md"
    custom_command = "catalog/plugins/demo/custom-commands/release.md"
    cache = {
        marketplace: _marketplace(
            [
                _plugin_entry(
                    strict=False,
                    skills="./custom-skills",
                    commands="./custom-commands",
                )
            ]
        ),
        manifest: json.dumps({"name": "demo"}),
        default_skill: _frontmatter("echo default-skill"),
        default_command: _frontmatter("echo default-command"),
        custom_skill: _frontmatter("echo custom-skill"),
        custom_command: _frontmatter("echo custom-command"),
    }

    result = node(_state(cache, components=[marketplace, manifest]))

    assert {finding.file for finding in result["findings"]} == {custom_skill, custom_command}
    assert default_skill not in {finding.file for finding in result["findings"]}
    assert default_command not in {finding.file for finding in result["findings"]}


@pytest.mark.parametrize(
    ("payload", "entry_index"),
    [
        ({"name": "catalog", "owner": {"name": "NVIDIA"}, "plugins": {}}, None),
        (
            {"name": "catalog", "owner": {"name": "NVIDIA"}, "plugins": [{"name": "demo"}]},
            0,
        ),
        (
            {
                "name": "catalog",
                "owner": {"name": "NVIDIA"},
                "plugins": [{"name": "demo", "source": ["./demo"]}],
            },
            0,
        ),
        (
            {
                "name": "catalog",
                "owner": {"name": "NVIDIA"},
                "metadata": {"pluginRoot": 7},
                "plugins": [_plugin_entry()],
            },
            None,
        ),
        (
            {
                "name": "catalog",
                "owner": {"name": "NVIDIA"},
                "plugins": [{"name": "demo", "strict": "yes", "source": "./demo"}],
            },
            0,
        ),
        (
            {
                "name": "catalog",
                "owner": {"name": "NVIDIA"},
                "plugins": [
                    {
                        "name": "demo",
                        "source": {"source": 7, "repo": "example/demo"},
                    }
                ],
            },
            0,
        ),
    ],
)
def test_malformed_marketplace_schema_fails_as_one_invalid_configuration(
    payload: dict[str, object],
    entry_index: int | None,
) -> None:
    """Malformed marketplace metadata does not silently activate arbitrary cache paths."""
    marketplace = "catalog/.claude-plugin/marketplace.json"

    result = node(_state({marketplace: json.dumps(payload)}, components=[marketplace]))

    assert result["findings"] == []
    expected_path = marketplace if entry_index is None else f"{marketplace}#plugin[{entry_index}]"
    assert [(event["path"], event["reason_code"]) for event in result["inspection_ledger"]] == [
        (expected_path, LedgerReason.INVALID_CONFIGURATION)
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"owner": {"name": "NVIDIA"}, "plugins": []},
        {"name": "catalog", "plugins": []},
        {"name": "catalog", "owner": "NVIDIA", "plugins": []},
        {"name": "catalog", "owner": {}, "plugins": []},
        {"name": "catalog", "owner": {"name": ""}, "plugins": []},
    ],
)
def test_marketplace_required_identity_fields_validate_before_activation(
    payload: dict[str, object],
) -> None:
    marketplace = "catalog/.claude-plugin/marketplace.json"

    result = node(_state({marketplace: json.dumps(payload)}, components=[marketplace]))

    assert result["findings"] == []
    assert [(event["path"], event["reason_code"]) for event in result["inspection_ledger"]] == [
        (marketplace, LedgerReason.INVALID_CONFIGURATION)
    ]


@pytest.mark.parametrize(
    "entry",
    [
        {"source": "./plugins/demo"},
        {"name": "", "source": "./plugins/demo"},
        {"name": "demo", "source": {"source": "github"}},
        {"name": "demo", "source": {"source": "github", "repo": 7}},
        {"name": "demo", "source": {"source": "url"}},
        {"name": "demo", "source": {"source": "git-subdir", "url": "https://x", "path": 7}},
        {"name": "demo", "source": {"source": "npm"}},
        {"name": "demo", "source": {"source": "future", "repo": "owner/repo"}},
    ],
)
def test_marketplace_entry_name_and_remote_source_union_are_required(
    entry: dict[str, object],
) -> None:
    marketplace = "catalog/.claude-plugin/marketplace.json"
    payload = {
        "name": "catalog",
        "owner": {"name": "NVIDIA"},
        "plugins": [entry],
    }

    result = node(_state({marketplace: json.dumps(payload)}, components=[marketplace]))

    assert result["findings"] == []
    assert [(event["path"], event["reason_code"]) for event in result["inspection_ledger"]] == [
        (f"{marketplace}#plugin[0]", LedgerReason.INVALID_CONFIGURATION)
    ]


@pytest.mark.parametrize(
    "source",
    [
        {"source": "github", "repo": "owner/repo"},
        {"source": "url", "url": "https://example.invalid/plugin.git"},
        {
            "source": "git-subdir",
            "url": "https://example.invalid/plugins.git",
            "path": "plugins/demo",
        },
        {"source": "npm", "package": "@example/demo"},
        {"source": "archive", "url": "https://example.invalid/demo.zip"},
        {"source": "command", "command": "example-plugin-path"},
    ],
)
def test_documented_remote_source_union_is_accepted_as_cache_incomplete(
    source: dict[str, object],
) -> None:
    marketplace = "catalog/.claude-plugin/marketplace.json"

    result = node(
        _state(
            {marketplace: _marketplace([_plugin_entry(source=source)])},
            components=[marketplace],
        )
    )

    assert result["findings"] == []
    assert [(event["path"], event["reason_code"]) for event in result["inspection_ledger"]] == [
        (f"{marketplace}#plugin[0]", LedgerReason.MISSING_FILE_CACHE)
    ]


@pytest.mark.parametrize(
    ("marketplace", "plugin_root"),
    [
        ("catalog/.claude-plugin/marketplace.json", "catalog/plugins/demo"),
        ("bundle.zip!/catalog/.claude-plugin/marketplace.json", "bundle.zip!/catalog/plugins/demo"),
    ],
)
def test_manifestless_marketplace_inline_entrypoint_uses_its_explicit_root(
    marketplace: str, plugin_root: str
) -> None:
    payload = f"{plugin_root}/scripts/hook.js"
    hooks = {
        "PostToolUse": [
            {
                "matcher": "Bash",
                "hooks": [
                    {
                        "type": "command",
                        "command": "node",
                        "args": ["${CLAUDE_PLUGIN_ROOT}/scripts/hook.js"],
                    }
                ],
            }
        ]
    }
    cache = {
        marketplace: _marketplace([_plugin_entry(strict=False, hooks=hooks)]),
        payload: "console.log('safe')\n",
    }

    result = node(_state(cache, components=[marketplace]))

    assert len(result["findings"]) == 1
    assert result["findings"][0].severity == "LOW"


def test_marketplace_handler_line_ignores_earlier_metadata_type_fields() -> None:
    marketplace = "catalog/.claude-plugin/marketplace.json"
    content = """{
  "name": "catalog",
  "owner": {"name": "NVIDIA"},
  "metadata": {"type": "catalog", "pluginRoot": "./plugins"},
  "plugins": [{
    "name": "demo",
    "source": "demo",
    "strict": false,
    "hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{
      "type": "command",
      "command": "echo safe"
    }]}]}
  }]
}
"""
    expected_line = next(
        index
        for index, line in enumerate(content.splitlines(), start=1)
        if '"type": "command"' in line
    )
    cache = {
        marketplace: content,
        "catalog/plugins/demo/README.md": "plugin exists\n",
    }

    result = node(_state(cache, components=[marketplace]))

    assert len(result["findings"]) == 1
    assert result["findings"][0].start_line == expected_line


def test_marketplace_inline_registrations_keep_two_entry_roots_isolated() -> None:
    marketplace = "catalog/.claude-plugin/marketplace.json"
    hooks = {
        "PostToolUse": [
            {
                "matcher": "Bash",
                "hooks": [
                    {
                        "type": "command",
                        "command": "node",
                        "args": ["${CLAUDE_PLUGIN_ROOT}/scripts/hook.js"],
                    }
                ],
            }
        ]
    }
    cache = {
        marketplace: _marketplace(
            [
                _plugin_entry(name="alpha", source="./plugins/alpha", strict=False, hooks=hooks),
                _plugin_entry(name="beta", source="./plugins/beta", strict=False, hooks=hooks),
            ]
        ),
        "catalog/plugins/alpha/scripts/hook.js": "console.log('alpha')\n",
        "catalog/plugins/beta/README.md": "beta exists but its hook does not\n",
    }

    result = node(_state(cache, components=[marketplace]))

    assert len(result["findings"]) == 1
    assert result["findings"][0].severity == "HIGH"
    assert result["findings"][0].evidence["handler_count"] == 2


def test_marketplace_reference_and_component_entrypoints_keep_plugin_root() -> None:
    marketplace = "catalog/.claude-plugin/marketplace.json"
    referenced = "catalog/plugins/demo/hooks/custom.json"
    command = "catalog/plugins/demo/commands/release.md"
    payload = "catalog/plugins/demo/scripts/hook.js"
    hook_map = {
        "PostToolUse": [
            {
                "matcher": "Bash",
                "hooks": [
                    {
                        "type": "command",
                        "command": "node",
                        "args": ["${CLAUDE_PLUGIN_ROOT}/scripts/hook.js"],
                    }
                ],
            }
        ]
    }
    cache = {
        marketplace: _marketplace(
            [
                _plugin_entry(
                    strict=False,
                    hooks="./hooks/custom.json",
                    commands="./commands/release.md",
                )
            ]
        ),
        referenced: json.dumps({"hooks": hook_map}),
        command: """---
hooks:
  PostToolUse:
    - matcher: Bash
      hooks:
        - type: command
          command: node ${CLAUDE_PLUGIN_ROOT}/scripts/hook.js
---
""",
        payload: "console.log('safe')\n",
    }

    result = node(_state(cache, components=[marketplace]))

    assert {finding.file for finding in result["findings"]} == {referenced, command}
    assert {finding.severity for finding in result["findings"]} == {"LOW"}


def test_plugin_project_dir_never_resolves_to_bundled_plugin_content() -> None:
    marketplace = "catalog/.claude-plugin/marketplace.json"
    payload = "catalog/plugins/demo/scripts/hook.js"
    hooks = {
        "PostToolUse": [
            {
                "matcher": "Bash",
                "hooks": [
                    {
                        "type": "command",
                        "command": "node",
                        "args": ["${CLAUDE_PROJECT_DIR}/scripts/hook.js"],
                    }
                ],
            }
        ]
    }
    cache = {
        marketplace: _marketplace([_plugin_entry(strict=False, hooks=hooks)]),
        payload: "console.log('bundled, not project content')\n",
    }

    result = node(_state(cache, components=[marketplace]))

    assert len(result["findings"]) == 1
    assert result["findings"][0].severity == "HIGH"
