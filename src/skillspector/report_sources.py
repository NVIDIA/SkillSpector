# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded source identities and non-destructive whole-report-plan validation."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from time import monotonic
from typing import Never

from skillspector.file_output import open_output_parent, preflight_output
from skillspector.input_handler import _normalize_root_owned_alias
from skillspector.report_outputs import ReportOutput

# Match the existing multi-skill discovery entry/depth/runtime ceilings. Unlike
# analyzer inventories, protection has no excluded directories or partial mode.
MAX_SOURCE_PATHS = 10_000
MAX_SOURCE_ENTRIES = 10_000
MAX_SOURCE_DEPTH = 64
MAX_SOURCE_PATH_BYTES = 4 * 1024 * 1024
MAX_SOURCE_DISCOVERY_SECONDS = 2.0
_SOURCE_DISCOVERY_SUPPORTED = os.scandir in os.supports_fd

FileIdentity = tuple[int, int]


def _identity(info: os.stat_result) -> FileIdentity:
    return info.st_dev, info.st_ino


class SourceProtectionError(ValueError):
    """Required source protection could not be established; emission is unsafe."""


class SourceProtectionInventory:
    """Keep source names and historical identities outside retained report data.

    Discovery is metadata-only, includes *all* entries under the supplied local
    root, and never follows links. Symlinks, special files, inaccessible paths,
    and exhausted bounds fail closed; caught errors cannot make a partial
    inventory eligible for output. A completed root's file list is reusable via
    ``discover_local``/``discovered_files`` without another filesystem walk.

    Callers must supply known local roots before analysis, plus any standalone
    inputs (for example registry comparisons) via ``add_path``. This helper
    deliberately does not infer local paths from remote URLs.
    """

    def __init__(
        self,
        *,
        max_paths: int = MAX_SOURCE_PATHS,
        max_entries: int = MAX_SOURCE_ENTRIES,
        max_depth: int = MAX_SOURCE_DEPTH,
        max_path_bytes: int = MAX_SOURCE_PATH_BYTES,
        max_seconds: float = MAX_SOURCE_DISCOVERY_SECONDS,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        for value in (max_paths, max_entries, max_depth, max_path_bytes):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Source protection bounds must be positive integers.")
        if isinstance(max_seconds, bool) or not 0 < max_seconds < float("inf"):
            raise ValueError("Source protection runtime bound must be positive and finite.")
        self.max_paths = max_paths
        self.max_entries = max_entries
        self.max_depth = max_depth
        self.max_path_bytes = max_path_bytes
        self.max_seconds = max_seconds
        self._clock = clock
        self._paths: set[Path] = set()
        self._files: set[Path] = set()
        self._identities: set[FileIdentity] = set()
        self._complete_roots: set[Path] = set()
        self._path_bytes = 0
        self._entries = 0
        self._discovery_seconds = 0.0
        self._failure: str | None = None

    @property
    def paths(self) -> frozenset[Path]:
        return frozenset(self._paths)

    @property
    def identities(self) -> frozenset[FileIdentity]:
        return frozenset(self._identities)

    @property
    def complete(self) -> bool:
        return self._failure is None

    def require_complete(self) -> None:
        """Reject a poisoned inventory even if a caller caught its first error."""
        if self._failure is not None:
            raise SourceProtectionError(self._failure)

    def _fail(self, detail: str) -> Never:
        self._failure = f"Could not establish report source protection: {detail}."
        raise SourceProtectionError(self._failure)

    @contextmanager
    def _operation(self) -> Iterator[None]:
        self.require_complete()
        try:
            yield
        except SourceProtectionError:
            raise
        except (OSError, ValueError, NotImplementedError) as exc:
            self._fail(str(exc))
        except BaseException:
            # Preserve interrupts/cancellation, but never leave a partial
            # inventory eligible for output if an outer caller catches them.
            self._failure = "Report source protection was interrupted or failed."
            raise

    def _consume_entry(self) -> None:
        self._entries += 1
        if self._entries > self.max_entries:
            self._fail("source entry bound exhausted")

    def _remember(self, path: Path, info: os.stat_result) -> None:
        if not (S_ISREG(info.st_mode) or S_ISDIR(info.st_mode)):
            self._fail(f"non-regular or symlinked local source {str(path)!r}")
        if path not in self._paths:
            size = len(os.fsencode(path))
            if len(self._paths) >= self.max_paths:
                self._fail("source path bound exhausted")
            if self._path_bytes + size > self.max_path_bytes:
                self._fail("source path-byte bound exhausted")
            self._paths.add(path)
            self._path_bytes += size
        identity = _identity(info)
        if identity not in self._identities and len(self._identities) >= self.max_paths:
            self._fail("source identity bound exhausted")
        self._identities.add(identity)
        if S_ISREG(info.st_mode):
            self._files.add(path)

    def _stat_path(self, path: str | Path) -> tuple[Path, os.stat_result]:
        with open_output_parent(path) as (absolute, directory_fd):
            return absolute, os.stat(
                absolute.name or ".", dir_fd=directory_fd, follow_symlinks=False
            )

    def add_path(self, path: str | Path) -> Path:
        """Protect an existing standalone file without reading its contents.

        Directories are discovered fully. Re-adding a file keeps both previous
        and current identities, so cleanup/replacement cannot erase protection.
        """
        with self._operation():
            absolute, info = self._stat_path(path)
            if S_ISDIR(info.st_mode):
                self.discover_local(absolute)
            else:
                self._remember(absolute, info)
            return absolute

    def discovered_files(self, root: str | Path) -> tuple[Path, ...]:
        """Return a bounded cached file inventory for a completely walked root."""
        self.require_complete()
        absolute = _normalize_root_owned_alias(Path(root))
        if not any(absolute.is_relative_to(known) for known in self._complete_roots):
            raise SourceProtectionError("No complete source discovery inventory for this root.")
        return tuple(sorted(path for path in self._files if path.is_relative_to(absolute)))

    def discover_local(self, root: str | Path) -> tuple[Path, ...]:
        """Discover every local file, independently of analysis/retention limits.

        A file root protects that file only, not its unrelated siblings. A
        directory root includes hidden files, VCS files and every child tree.
        The cache is a point-in-time inventory, not a filesystem reservation.
        """
        with self._operation():
            if not _SOURCE_DISCOVERY_SUPPORTED:
                self._fail("descriptor-relative source discovery is unsupported on this platform")
            absolute = _normalize_root_owned_alias(Path(root))
            if any(absolute.is_relative_to(known) for known in self._complete_roots):
                return self.discovered_files(absolute)
            started = self._clock()
            deadline = started + max(0.0, self.max_seconds - self._discovery_seconds)

            def check_time() -> None:
                if self._clock() >= deadline:
                    self._fail("source discovery runtime bound exhausted")

            def walk(path: Path, fd: int, depth: int) -> None:
                check_time()
                with os.scandir(fd) as entries:
                    for entry in entries:
                        check_time()
                        self._consume_entry()
                        child = path / entry.name
                        info = os.stat(entry.name, dir_fd=fd, follow_symlinks=False)
                        self._remember(child, info)
                        if S_ISDIR(info.st_mode):
                            if depth >= self.max_depth:
                                self._fail("source discovery depth bound exhausted")
                            child_fd = os.open(
                                entry.name,
                                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=fd,
                            )
                            try:
                                if _identity(os.fstat(child_fd)) != _identity(info):
                                    self._fail("source directory changed during discovery")
                                walk(child, child_fd, depth + 1)
                            finally:
                                os.close(child_fd)

            try:
                check_time()
                with open_output_parent(absolute) as (absolute, parent_fd):
                    info = os.stat(absolute.name or ".", dir_fd=parent_fd, follow_symlinks=False)
                    self._remember(absolute, info)
                    if S_ISREG(info.st_mode):
                        check_time()
                        return (absolute,)
                    fd = os.open(
                        absolute.name or ".",
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=parent_fd,
                    )
                    try:
                        if _identity(os.fstat(fd)) != _identity(info):
                            self._fail("source directory changed during discovery")
                        walk(absolute, fd, 0)
                        check_time()
                    finally:
                        os.close(fd)
                self._complete_roots.add(absolute)
                return self.discovered_files(absolute)
            finally:
                self._discovery_seconds += max(0.0, self._clock() - started)

    def protects(self, path: Path, info: os.stat_result | None) -> bool:
        self.require_complete()
        return path in self._paths or info is not None and _identity(info) in self._identities

    def refresh(self) -> None:
        """Recheck bounded known names without rewalking any directory.

        Preserve historical identities when files change or are cleaned up.
        New identities are also protected; inaccessible or symlinked names
        still fail closed. New names must come from discovery/result tracking.
        """
        with self._operation():
            for path in tuple(self._paths):
                try:
                    absolute, info = self._stat_path(path)
                except FileNotFoundError:
                    continue
                except ValueError as exc:
                    # open_output_parent wraps a removed temporary parent.
                    if isinstance(exc.__cause__, FileNotFoundError):
                        continue
                    raise
                self._remember(absolute, info)

    def add_result(self, result: Mapping[str, object]) -> None:
        """Record an unmerged root/child graph result before truncation/cleanup.

        Concrete ``skill_path``/``root_path`` roots are fully discovered, reusing
        cached discovery where available. Components and every artifact row
        (including excluded/failed rows) add identities even if omitted later.
        Archive member names protect their physical outer container; scoped
        transitive names must be recorded *before* decoration/merging.

        Malformed, missing or ambiguous source metadata fails closed. No raw
        content, prepared snapshot, graph state or cleanup resource is retained.
        """
        with self._operation():
            records = 0

            def consume_record() -> None:
                nonlocal records
                records += 1
                if records > self.max_entries:
                    self._fail("source metadata entry bound exhausted")

            bases: list[Path] = []
            for key in ("skill_path", "root_path"):
                value = result.get(key)
                if value is None or value == "":
                    continue
                if not isinstance(value, str | Path):
                    self._fail(f"invalid {key} source metadata")
                consume_record()
                root = _normalize_root_owned_alias(Path(value))
                self.discover_local(root)
                base = root.parent if root in self._files else root
                if base not in bases:
                    bases.append(base)

            def source_path(value: object) -> None:
                consume_record()
                if not isinstance(value, str | Path) or not str(value):
                    self._fail("invalid component source path")
                name = str(value)
                if "://" in name or "::" in name:
                    self._fail("ambiguous source identity; record unmerged graph results")
                # Nested artifacts use container.zip!/member/path. The outer
                # file is the writable local source; members are virtual.
                physical = Path(name.split("!/", 1)[0])
                if ".." in physical.parts:
                    self._fail("parent traversal in component source metadata")
                if physical.is_absolute():
                    self.add_path(physical)
                elif bases:
                    # Raw graph-relative components belong to skill_path,
                    # falling back to root_path when that is the only root.
                    self.add_path(bases[0] / physical)
                else:
                    self._fail("relative component source has no concrete local root")

            def path_sequence(value: object) -> None:
                if value is None:
                    return
                if not isinstance(value, Sequence) or isinstance(value, str | bytes):
                    self._fail("invalid source path collection")
                for item in value:
                    source_path(item)

            path_sequence(result.get("components"))
            rows = result.get("artifact_inventory")
            if rows is not None:
                if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
                    self._fail("invalid artifact source inventory")
                for row in rows:
                    if not isinstance(row, Mapping):
                        self._fail("invalid artifact source record")
                    source_path(row.get("path"))
                    path_sequence(row.get("inspected_members"))


_SOURCE_PROTECTION: ContextVar[SourceProtectionInventory | None] = ContextVar(
    "report_source_protection_inventory", default=None
)


@contextmanager
def track_report_sources(
    inventory: SourceProtectionInventory,
) -> Iterator[SourceProtectionInventory]:
    """Activate result tracking for one scan, resetting on errors/interrupts.

    Main must call ``record_scan_sources(result)`` for every returned root and
    child in ``_run_graph_scan``, before transitive truncation and before cleanup.
    The same inventory then supplies the whole-plan pre-emission barrier.
    """
    inventory.require_complete()
    token = _SOURCE_PROTECTION.set(inventory)
    try:
        yield inventory
    finally:
        _SOURCE_PROTECTION.reset(token)


def current_source_inventory() -> SourceProtectionInventory | None:
    """Expose the bounded discovery cache to analysis without another walk."""
    return _SOURCE_PROTECTION.get()


def record_scan_sources[T: Mapping[str, object]](result: T) -> T:
    """Track a raw graph result if enabled, returning the exact same object.

    With no tracking context the legacy path performs no filesystem checks.
    """
    inventory = current_source_inventory()
    if inventory is not None:
        inventory.add_result(result)
    return result


@dataclass(frozen=True, slots=True)
class ValidatedReportOutput:
    """One non-destructively checked entry; no handles or reservations are held."""

    output: ReportOutput
    normalized_path: Path | None
    existing_stat: os.stat_result | None


def validate_report_outputs(
    entries: Sequence[ReportOutput], inventory: SourceProtectionInventory
) -> list[ValidatedReportOutput]:
    """Validate the whole plan before analysis or before *any* report emission.

    Reuse this same barrier after recording all root/child results, before
    primary stdout as well as file writes. This performs no output operations.
    Write-time protection still belongs to ``write_text_no_follow``.
    """
    inventory.require_complete()
    inventory.refresh()
    names: set[Path] = set()
    identities: set[FileIdentity] = set()
    validated: list[ValidatedReportOutput] = []
    for entry in entries:
        if entry.path is None:
            if not entry.is_primary:
                raise ValueError("Only the primary report may use stdout.")
            validated.append(ValidatedReportOutput(entry, None, None))
            continue
        absolute, info = preflight_output(entry.path)
        identity = _identity(info) if info is not None else None
        if absolute in names or identity is not None and identity in identities:
            raise ValueError(f"Duplicate or aliased report destination {entry.path!r}.")
        if inventory.protects(absolute, info):
            raise ValueError(f"Report destination {entry.path!r} aliases a protected local source.")
        names.add(absolute)
        if identity is not None:
            identities.add(identity)
        validated.append(ValidatedReportOutput(entry, absolute, info))
    return validated
