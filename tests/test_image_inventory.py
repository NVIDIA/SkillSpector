# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-skill local image inventory.

Local markdown image targets plus loose image files are inventoried;
remote URLs are cited-but-unfetched (never fetched over the network).
"""

from __future__ import annotations

import socket
import time

from skillspector.image_text import IMAGE_SUFFIXES
from skillspector.inspection_ledger import LedgerOutcome
from skillspector.references import collect_image_inventory, image_inventory_ledger_events


def _inventory(source: str, path: str, known: list[str]) -> dict[str, list[str]]:
    return collect_image_inventory(source, path, known, clock=time.monotonic, deadline=float("inf"))


def test_local_markdown_image_targets_are_inventoried() -> None:
    inventory = _inventory(
        "See ![arch](assets/arch.png) and ![logo](logo.jpg).",
        "SKILL.md",
        ["SKILL.md", "assets/arch.png", "logo.jpg"],
    )
    assert inventory["local_images"] == ["assets/arch.png", "logo.jpg"]
    assert inventory["remote_images"] == []


def test_loose_image_files_are_inventoried_without_references() -> None:
    inventory = _inventory(
        "No images referenced here.",
        "SKILL.md",
        ["SKILL.md", "assets/loose.webp", "notes.md"],
    )
    assert inventory["local_images"] == ["assets/loose.webp"]
    assert inventory["remote_images"] == []


def test_supported_suffixes_are_all_inventoried() -> None:
    known = ["SKILL.md", *(f"img/a{i}{suffix}" for i, suffix in enumerate(sorted(IMAGE_SUFFIXES)))]
    inventory = _inventory("No references.", "SKILL.md", known)
    assert inventory["local_images"] == sorted(known[1:])


def test_remote_image_urls_are_cited_but_unfetched(monkeypatch) -> None:
    def _no_network(*args: object, **kwargs: object) -> object:
        raise AssertionError("network fetch attempted during image inventory")

    monkeypatch.setattr(socket, "socket", _no_network)
    source = "![a](https://example.com/a.png) ![b](data:image/png;base64,iVBOR)"
    inventory = _inventory(source, "SKILL.md", ["SKILL.md"])
    assert inventory["local_images"] == []
    assert "https://example.com/a.png" in inventory["remote_images"]
    assert any(target.startswith("data:") for target in inventory["remote_images"])


def test_non_image_targets_are_excluded() -> None:
    inventory = _inventory(
        "Read [guide](docs/guide.md).",
        "SKILL.md",
        ["SKILL.md", "docs/guide.md"],
    )
    assert inventory == {"local_images": [], "remote_images": []}


def test_expired_deadline_truncates_markdown_parsing() -> None:
    # The target lives only in markdown (not in known paths): a live
    # deadline finds it, an expired one skips parsing yet still walks
    # the cheap known-paths list.
    source = "See ![arch](assets/arch.png)."
    live = collect_image_inventory(
        source,
        "SKILL.md",
        ["SKILL.md"],
        clock=time.monotonic,
        deadline=float("inf"),
    )
    assert live["local_images"] == ["assets/arch.png"]
    expired = collect_image_inventory(
        source,
        "SKILL.md",
        ["SKILL.md", "assets/arch.png"],
        clock=lambda: 10.0,
        deadline=5.0,
    )
    assert expired["local_images"] == ["assets/arch.png"]
    assert expired["remote_images"] == []


def test_image_inventory_ledger_events_carry_no_verdicts() -> None:
    inventory = {
        "local_images": ["assets/a.png"],
        "remote_images": ["https://example.com/b.png"],
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
