# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-skill local image inventory.

Local markdown image targets (derived from resolved reference records)
plus loose image files are inventoried; remote URLs are cited as
scheme-plus-host and never fetched over the network.
"""

from __future__ import annotations

import socket
import time

from skillspector.artifacts import ArtifactDisposition, ReferenceKind
from skillspector.image_text import IMAGE_SUFFIXES
from skillspector.inspection_ledger import (
    LedgerOutcome,
    LedgerReason,
    image_inventory_public_view,
)
from skillspector.references import collect_image_inventory, image_inventory_ledger_events


def _record(target: str, kind: object = ReferenceKind.MARKDOWN_IMAGE) -> dict:
    return {
        "source_path": "SKILL.md",
        "line": 1,
        "column": 1,
        "evidence": f"![x]({target})",
        "target_path": target,
        "status": "resolved",
        "disposition": ArtifactDisposition.ANALYZED,
        "reference_kind": kind,
    }


def _inventory(
    source: str, path: str, known: list[str], records: list[dict] | None = None
) -> tuple[dict[str, list[str]], bool]:
    return collect_image_inventory(
        source, path, known, records or [], clock=time.monotonic, deadline=float("inf")
    )


def test_local_markdown_image_targets_are_inventoried() -> None:
    inventory, truncated = _inventory(
        "See ![arch](assets/arch.png) and ![logo](logo.jpg).",
        "SKILL.md",
        ["SKILL.md", "assets/arch.png", "logo.jpg"],
        [_record("assets/arch.png"), _record("logo.jpg")],
    )
    assert inventory["local_images"] == ["assets/arch.png", "logo.jpg"]
    assert inventory["remote_images"] == []
    assert truncated is False


def test_loose_image_files_are_inventoried_without_references() -> None:
    inventory, _ = _inventory(
        "No images referenced here.",
        "SKILL.md",
        ["SKILL.md", "assets/loose.webp", "notes.md"],
    )
    assert inventory["local_images"] == ["assets/loose.webp"]
    assert inventory["remote_images"] == []


def test_supported_suffixes_are_all_inventoried() -> None:
    known = ["SKILL.md", *(f"img/a{i}{suffix}" for i, suffix in enumerate(sorted(IMAGE_SUFFIXES)))]
    inventory, _ = _inventory("No references.", "SKILL.md", known)
    assert inventory["local_images"] == sorted(known[1:])
    assert inventory["remote_images"] == []


def test_fenced_image_to_missing_file_is_not_inventoried() -> None:
    # No record, not a real file: fence text must not conjure an entry.
    source = "```\n![arch](assets/arch.png)\n```\n"
    inventory, _ = _inventory(source, "SKILL.md", ["SKILL.md"], [])
    assert inventory["local_images"] == []


def test_remote_image_urls_are_cited_but_unfetched(monkeypatch) -> None:
    def _no_network(*args: object, **kwargs: object) -> object:
        raise AssertionError("network fetch attempted during image inventory")

    monkeypatch.setattr(socket, "socket", _no_network)
    source = "![a](https://example.com/a.png?token=secret) ![b](data:image/png;base64,iVBOR)"
    inventory, _ = _inventory(source, "SKILL.md", ["SKILL.md"])
    assert inventory["local_images"] == []
    assert inventory["remote_images"] == ["https://example.com"]
    assert inventory["inline_images"] == ["data:image/png;base64,iVBOR"]


def test_malformed_remote_url_never_raises() -> None:
    inventory, truncated = _inventory(
        "See ![x](http://h:badport/a.png) and ![y](https://example.com/b.png).",
        "SKILL.md",
        ["SKILL.md"],
    )
    assert inventory["remote_images"] == ["https://example.com"]
    assert truncated is False


def test_non_image_targets_are_excluded() -> None:
    inventory, truncated = _inventory(
        "Read [guide](docs/guide.md).",
        "SKILL.md",
        ["SKILL.md", "docs/guide.md"],
    )
    assert inventory == {"local_images": [], "remote_images": [], "inline_images": []}
    assert truncated is False


def test_expired_deadline_truncates_markdown_parsing() -> None:
    # Records apply cheaply even on an expired budget; the markdown
    # re-scan is what stops and reports truncation.
    records = [_record("assets/arch.png")]
    known = ["SKILL.md", "assets/arch.png"]
    live, live_truncated = collect_image_inventory(
        "See ![arch](assets/arch.png).",
        "SKILL.md",
        known,
        records,
        clock=time.monotonic,
        deadline=float("inf"),
    )
    assert live["local_images"] == ["assets/arch.png"]
    assert live_truncated is False
    expired, expired_truncated = collect_image_inventory(
        "See ![arch](assets/arch.png).",
        "SKILL.md",
        known,
        records,
        clock=lambda: 10.0,
        deadline=5.0,
    )
    assert expired["local_images"] == ["assets/arch.png"]
    assert expired_truncated is True


def test_unresolved_record_target_is_not_inventoried() -> None:
    # Reference resolution reports the missing target; the inventory
    # must not conjure a COMPLETED row for it.
    record = _record("assets/missing.png")
    record["status"] = "unresolved"
    inventory, _ = _inventory(
        "See ![missing](assets/missing.png).",
        "SKILL.md",
        ["SKILL.md"],
        [record],
    )
    assert inventory["local_images"] == []


def test_entry_cap_truncation_is_reported() -> None:
    known = ["SKILL.md", *(f"img/a{i}.png" for i in range(1100))]
    inventory, truncated = _inventory("No references.", "SKILL.md", known)
    assert len(inventory["local_images"]) == 1024
    assert truncated is True


def test_image_inventory_ledger_events_carry_no_verdicts() -> None:
    inventory = {
        "local_images": ["assets/a.png"],
        "remote_images": ["https://example.com"],
        "inline_images": [],
    }
    events = image_inventory_ledger_events(inventory, "SKILL.md")
    assert events
    assert {event["path"] for event in events} == {"assets/a.png"}
    for event in events:
        assert event["outcome"] is LedgerOutcome.COMPLETED
        assert event["phase"] == "image_inventory"
        assert "reason_code" not in event
        assert event["input_finding_ids"] == []
        assert event["emitted_finding_ids"] == []


def test_truncated_inventory_emits_partial_row() -> None:
    inventory = {"local_images": [], "remote_images": [], "inline_images": []}
    events = image_inventory_ledger_events(inventory, "SKILL.md", truncated=True)
    assert len(events) == 1
    assert events[0]["outcome"] is LedgerOutcome.PARTIAL
    assert events[0]["reason_code"] is LedgerReason.OUTPUT_LIMIT


def test_public_inventory_summarizes_inline_uris() -> None:
    raw = {
        "local_images": ["a.png"],
        "remote_images": ["https://example.com"],
        "inline_images": ["data:image/png;base64," + "A" * 100],
    }
    view = image_inventory_public_view(raw)
    assert view["local_images"] == ["a.png"]
    assert view["remote_images"] == ["https://example.com"]
    assert view["inline_images"] == ["data:image/png;base64(100 bytes)"]
    assert "AAAA" not in str(view)
