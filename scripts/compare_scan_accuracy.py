# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare two SkillSpector revisions on a human-adjudicated local corpus.

The corpus and generated report are intentionally external inputs. This keeps
private or disclosure-controlled fixtures out of the repository while making
the accuracy gate deterministic and reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
REQUIRED_CLASSIFICATIONS = frozenset({"maintained_benign", "approved_real_world"})
_SOURCE_SCAN_MODE = "--_source-bound-scan"
_GIT_TIMEOUT_SECONDS = 30.0
_IDENTITY_TIMEOUT_SECONDS = 120.0
_SCAN_TIMEOUT_SECONDS = 300.0
_GIT_STDOUT_LIMIT_BYTES = 64 * 1024 * 1024
_IDENTITY_STDOUT_LIMIT_BYTES = 16 * 1024 * 1024
_SCAN_STDOUT_LIMIT_BYTES = 32 * 1024 * 1024
_STDERR_LIMIT_BYTES = 1024 * 1024
_SOURCE_ARCHIVE_LIMIT_BYTES = 64 * 1024 * 1024
_SOURCE_ARCHIVE_MEMBER_LIMIT = 20_000
_ISOLATED_HOME_MARKER = "<fresh-owned-empty-home>"
_CONSOLE_ENTRYPOINT_BODY = b"""# -*- coding: utf-8 -*-
import sys
from skillspector.cli import app
if __name__ == "__main__":
    if sys.argv[0].endswith("-script.pyw"):
        sys.argv[0] = sys.argv[0][:-11]
    elif sys.argv[0].endswith(".exe"):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(app())
"""
_RUNTIME_IDENTITY_PROBE = r"""
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import urllib.parse
from pathlib import Path

MAX_DEPENDENCY_FILES = 200_000
MAX_DEPENDENCY_BYTES = 4 * 1024 * 1024 * 1024
EDITABLE_IGNORED_PARTS = {
    ".git", ".hg", ".svn", ".venv", "venv", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".nox",
}

total_files = 0
total_bytes = 0

def hash_file(digest, label, path):
    global total_files, total_bytes
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"dependency identity path is not a regular file: {label}")
    before = path.stat()
    if total_files >= MAX_DEPENDENCY_FILES:
        raise RuntimeError("installed dependency identity exceeds the file limit")
    if before.st_size < 0 or total_bytes + before.st_size > MAX_DEPENDENCY_BYTES:
        raise RuntimeError("installed dependency identity exceeds the byte limit")
    encoded_label = label.encode("utf-8")
    digest.update(len(encoded_label).to_bytes(8, "big"))
    digest.update(encoded_label)
    digest.update(before.st_size.to_bytes(8, "big"))
    with path.open("rb") as stream:
        opened_before = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino, before.st_size) != (
            opened_before.st_dev, opened_before.st_ino, opened_before.st_size
        ):
            raise RuntimeError(f"dependency was swapped before hashing: {label}")
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        opened_after = os.fstat(stream.fileno())
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        opened_before.st_dev,
        opened_before.st_ino,
        opened_before.st_size,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        opened_after.st_dev,
        opened_after.st_ino,
        opened_after.st_size,
    ):
        raise RuntimeError(f"dependency changed while hashing: {label}")
    total_files += 1
    total_bytes += before.st_size

dependencies = []
seen_names = set()
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name")
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError("installed distribution has no name")
    normalized_name = name.strip().lower().replace("_", "-")
    if normalized_name == "skillspector":
        continue
    if normalized_name in seen_names:
        raise RuntimeError(f"duplicate installed distribution: {normalized_name}")
    seen_names.add(normalized_name)
    files = distribution.files
    record = distribution.read_text("RECORD")
    metadata = distribution.read_text("METADATA")
    if files is None or record is None or metadata is None:
        raise RuntimeError(f"installed distribution has incomplete identity metadata: {normalized_name}")
    direct_url_text = distribution.read_text("direct_url.json") or ""
    metadata_digest = hashlib.sha256()
    for label, value in (("METADATA", metadata), ("RECORD", record), ("direct_url.json", direct_url_text)):
        encoded_label = label.encode("utf-8")
        encoded_value = value.encode("utf-8")
        metadata_digest.update(len(encoded_label).to_bytes(8, "big"))
        metadata_digest.update(encoded_label)
        metadata_digest.update(len(encoded_value).to_bytes(8, "big"))
        metadata_digest.update(encoded_value)
    contents_digest = hashlib.sha256()
    distribution_file_count = 0
    distribution_bytes = 0
    seen_paths = set()
    for package_path in sorted(files, key=str):
        located = Path(distribution.locate_file(package_path))
        resolved = located.resolve(strict=True)
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        before_files = total_files
        before_bytes = total_bytes
        hash_file(contents_digest, f"installed:{package_path}", located)
        distribution_file_count += total_files - before_files
        distribution_bytes += total_bytes - before_bytes

    editable = False
    editable_file_count = 0
    editable_bytes = 0
    if direct_url_text:
        direct_url = json.loads(direct_url_text)
        if not isinstance(direct_url, dict):
            raise RuntimeError(f"invalid direct_url.json for {normalized_name}")
        directory_info = direct_url.get("dir_info")
        editable = isinstance(directory_info, dict) and directory_info.get("editable") is True
        if editable:
            raw_url = direct_url.get("url")
            if not isinstance(raw_url, str):
                raise RuntimeError(f"editable dependency has no URL: {normalized_name}")
            parsed = urllib.parse.urlsplit(raw_url)
            if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
                raise RuntimeError(f"editable dependency is not a local file target: {normalized_name}")
            editable_root = Path(urllib.parse.unquote(parsed.path)).resolve(strict=True)
            if not editable_root.is_dir():
                raise RuntimeError(f"editable dependency target is not a directory: {normalized_name}")
            for editable_path in sorted(editable_root.rglob("*")):
                relative = editable_path.relative_to(editable_root)
                if any(part in EDITABLE_IGNORED_PARTS for part in relative.parts):
                    continue
                if editable_path.is_symlink():
                    raise RuntimeError(f"editable dependency contains a symlink: {normalized_name}")
                if not editable_path.is_file():
                    continue
                resolved = editable_path.resolve(strict=True)
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                before_files = total_files
                before_bytes = total_bytes
                hash_file(contents_digest, f"editable:{relative.as_posix()}", editable_path)
                editable_file_count += total_files - before_files
                editable_bytes += total_bytes - before_bytes
    dependencies.append(
        {
            "name": normalized_name,
            "version": distribution.version,
            "recorded_file_count": len(files),
            "distribution_metadata_sha256": f"sha256:{metadata_digest.hexdigest()}",
            "installed_file_count": distribution_file_count,
            "installed_bytes": distribution_bytes,
            "installed_contents_sha256": f"sha256:{contents_digest.hexdigest()}",
            "editable": editable,
            "editable_file_count": editable_file_count,
            "editable_bytes": editable_bytes,
        }
    )
payload = {
    "python_version": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "python_cache_tag": sys.implementation.cache_tag,
    "python_hexversion": sys.hexversion,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "byteorder": sys.byteorder,
    "dependency_file_count": total_files,
    "dependency_bytes": total_bytes,
    "dependencies": sorted(dependencies, key=lambda item: item["name"]),
}
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
""".strip()
POLICY_FIELDS = (
    "max_candidate_false_positives",
    "max_candidate_false_negatives",
    "max_false_positive_increase",
    "max_false_negative_increase",
    "max_per_rule_false_positive_increase",
    "max_per_rule_false_negative_increase",
    "max_per_cohort_false_positive_increase",
    "max_per_cohort_false_negative_increase",
    "max_per_case_false_positive_increase",
    "max_per_case_false_negative_increase",
)
MANIFEST_FIELDS = frozenset({"schema_version", "material_regression_policy", "cases"})
CASE_FIELDS = frozenset({"id", "path", "classification", "expected_rules"})
APPROVAL_FIELDS = frozenset(
    {
        "schema_version",
        "reviewer",
        "rationale",
        "corpus_identity",
        "manifest_sha256",
        "baseline_identity",
        "candidate_identity",
        "policy_sha256",
        "violations",
        "violations_sha256",
    }
)
APPROVAL_IDENTITY_FIELDS = (
    "revision",
    "source_tree_git_oid",
    "executable_sha256",
    "python_executable_sha256",
    "runtime_identity_sha256",
    "dependency_identity_sha256",
    "pyproject_sha256",
    "lockfile_sha256",
    "source_runner_sha256",
)


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _load_json_bytes(raw: bytes, path: Path) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _load_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    return _load_json_bytes(raw, path), raw


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _hash_record(digest: Any, label: bytes, value: bytes) -> None:
    """Add one unambiguous, length-delimited value to an evidence digest."""
    digest.update(len(label).to_bytes(8, "big"))
    digest.update(label)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _resolve_case_path(corpus_root: Path, relative_path: str) -> Path:
    root = corpus_root.resolve(strict=True)
    target = (root / relative_path).resolve(strict=True)
    if not target.is_relative_to(root) or not target.is_dir():
        raise ValueError(f"Corpus case must be a directory below the corpus root: {relative_path}")
    return target


def _corpus_identity(
    corpus_root: Path,
    cases: list[dict[str, Any]],
    manifest_bytes: bytes,
) -> str:
    """Hash exact adjudication bytes plus every selected corpus path and byte."""
    digest = hashlib.sha256()
    _hash_record(digest, b"domain", b"skillspector-accuracy-corpus-v2")
    _hash_record(digest, b"manifest", manifest_bytes)
    seen: set[Path] = set()
    root = corpus_root.resolve(strict=True)
    for case in sorted(cases, key=lambda item: str(item["id"])):
        target = _resolve_case_path(corpus_root, str(case["path"]))
        for path in sorted(target.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Corpus snapshot does not follow symlinks: {path}")
            if not path.is_file():
                continue
            resolved = path.resolve(strict=True)
            if resolved in seen:
                continue
            seen.add(resolved)
            relative = resolved.relative_to(root).as_posix().encode("utf-8")
            _hash_record(digest, b"path", relative)
            _hash_record(digest, b"contents", resolved.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Terminate a bounded subprocess and any children it placed in our session."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:  # pragma: no cover - exercised by Windows CI
            process.kill()
    except ProcessLookupError:
        return


def _run_bounded(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout_seconds: float,
    stdout_limit_bytes: int,
    stderr_limit_bytes: int,
) -> subprocess.CompletedProcess[bytes]:
    """Run a subprocess with wall-clock and captured-output limits.

    Dedicated drain threads prevent a producer from filling a pipe. Each
    thread retains at most its declared limit; the whole process group is
    killed as soon as either stream crosses its bound.
    """
    if timeout_seconds <= 0 or stdout_limit_bytes < 0 or stderr_limit_bytes < 0:
        raise ValueError("Subprocess bounds must be positive")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=(os.name == "posix"),
    )
    assert process.stdout is not None
    assert process.stderr is not None
    streams: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    over_limit = threading.Event()

    def drain(name: str, stream: Any, limit: int) -> None:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                return
            if len(streams[name]) + len(chunk) > limit:
                remaining = max(0, limit - len(streams[name]))
                streams[name].extend(chunk[:remaining])
                over_limit.set()
                return
            streams[name].extend(chunk)

    stdout_thread = threading.Thread(
        target=drain,
        args=("stdout", process.stdout, stdout_limit_bytes),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=drain,
        args=("stderr", process.stderr, stderr_limit_bytes),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    deadline = time.monotonic() + timeout_seconds
    failure: str | None = None
    try:
        while process.poll() is None:
            if over_limit.is_set():
                failure = "output exceeded its byte limit"
                _terminate_process_tree(process)
                break
            if time.monotonic() >= deadline:
                failure = f"timed out after {timeout_seconds:g} seconds"
                _terminate_process_tree(process)
                break
            time.sleep(0.01)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(process)
        process.wait()
        failure = failure or "could not be terminated within its runtime limit"
    finally:
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            failure = failure or "left an output stream open after exit"
            _terminate_process_tree(process)
        process.stdout.close()
        process.stderr.close()
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
    if over_limit.is_set():
        failure = "output exceeded its byte limit"
    if failure:
        raise RuntimeError(f"Bounded subprocess {failure}: {command[0]}")
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        bytes(streams["stdout"]),
        bytes(streams["stderr"]),
    )


@contextmanager
def _fresh_owned_home() -> Iterator[Path]:
    """Yield a new private empty HOME and remove it without following symlinks."""
    home = Path(tempfile.mkdtemp(prefix="skillspector-accuracy-home-"))
    try:
        home.chmod(0o700)
        before = home.lstat()
        if home.is_symlink() or not home.is_dir() or before.st_uid != os.geteuid():
            raise RuntimeError("Could not create a private owned accuracy-gate HOME")
        if any(home.iterdir()):
            raise RuntimeError("Accuracy-gate HOME was not empty at creation")
        yield home
    finally:
        if home.is_symlink():
            home.unlink(missing_ok=True)
        elif home.exists():
            shutil.rmtree(home)


def _git_output(worktree: Path, *args: str) -> str:
    try:
        completed = _run_bounded(
            ["git", "-C", str(worktree), *args],
            cwd=worktree,
            timeout_seconds=_GIT_TIMEOUT_SECONDS,
            stdout_limit_bytes=_GIT_STDOUT_LIMIT_BYTES,
            stderr_limit_bytes=_STDERR_LIMIT_BYTES,
        )
    except RuntimeError as error:
        raise ValueError(f"Cannot verify scanner worktree {worktree}: {error}") from error
    if completed.returncode != 0:
        detail = (
            completed.stderr.decode("utf-8", errors="replace").strip()
            or completed.stdout.decode("utf-8", errors="replace").strip()
            or "git command failed"
        )
        raise ValueError(f"Cannot verify scanner worktree {worktree}: {detail}")
    try:
        return completed.stdout.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise ValueError(f"Git returned non-UTF-8 identity output for {worktree}") from error


def _scan_environment(home: Path, *, disclose_home: bool = False) -> dict[str, str]:
    """Return the fixed, non-secret environment used by every accuracy scan."""
    environment = {
        "HOME": str(home) if disclose_home else _ISOLATED_HOME_MARKER,
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.defpath,
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "SKILLSPECTOR_LOG_LEVEL": "WARNING",
        "TZ": "UTC",
    }
    for name in ("SYSTEMROOT", "WINDIR"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def _runtime_identity(interpreter: Path, worktree: Path) -> dict[str, Any]:
    with _fresh_owned_home() as home:
        execution_environment = _scan_environment(home, disclose_home=True)
        completed = _run_bounded(
            [str(interpreter), "-I", "-B", "-c", _RUNTIME_IDENTITY_PROBE],
            cwd=worktree,
            env=execution_environment,
            timeout_seconds=_IDENTITY_TIMEOUT_SECONDS,
            stdout_limit_bytes=_IDENTITY_STDOUT_LIMIT_BYTES,
            stderr_limit_bytes=_STDERR_LIMIT_BYTES,
        )
    if completed.returncode != 0:
        detail = (
            completed.stderr.decode("utf-8", errors="replace").strip()
            or completed.stdout.decode("utf-8", errors="replace").strip()
            or "runtime probe failed"
        )
        raise ValueError(f"Cannot identify scanner dependency runtime: {detail}")
    try:
        identity = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Scanner dependency runtime returned invalid identity JSON") from error
    if not isinstance(identity, dict) or not isinstance(identity.get("dependencies"), list):
        raise ValueError("Scanner dependency runtime returned an invalid identity object")
    identity["probe_sha256"] = (
        f"sha256:{hashlib.sha256(_RUNTIME_IDENTITY_PROBE.encode()).hexdigest()}"
    )
    evidence_environment = _scan_environment(Path(_ISOLATED_HOME_MARKER))
    identity["environment"] = evidence_environment
    identity["environment_sha256"] = _json_sha256(evidence_environment)
    identity["dependency_identity_sha256"] = _json_sha256(identity["dependencies"])
    identity["runtime_identity_sha256"] = _json_sha256(
        {key: value for key, value in identity.items() if key != "runtime_identity_sha256"}
    )
    return identity


def _actual_source_files(source_root: Path) -> set[str]:
    """Inventory every regular source-tree file, including ignored files."""
    actual: set[str] = set()
    for directory, directory_names, file_names in os.walk(source_root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            child = directory_path / name
            if child.is_symlink():
                raise ValueError(f"Scanner source tree contains a symlink: {child}")
        for name in file_names:
            child = directory_path / name
            if child.is_symlink() or not child.is_file():
                raise ValueError(f"Scanner source tree contains a non-regular file: {child}")
            actual.add(child.relative_to(source_root.parent).as_posix())
    return actual


def _git_source_archive(worktree: Path, revision: str) -> bytes:
    try:
        completed = _run_bounded(
            ["git", "-C", str(worktree), "archive", "--format=tar", revision, "src"],
            cwd=worktree,
            timeout_seconds=_GIT_TIMEOUT_SECONDS,
            stdout_limit_bytes=_SOURCE_ARCHIVE_LIMIT_BYTES,
            stderr_limit_bytes=_STDERR_LIMIT_BYTES,
        )
    except RuntimeError as error:
        raise ValueError(f"Cannot snapshot scanner revision {revision}: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Cannot snapshot scanner revision {revision}: {detail}")
    return completed.stdout


def _extract_source_archive(archive_bytes: bytes, destination: Path) -> Path:
    """Extract the Git-produced source archive without archive traversal semantics."""
    destination.mkdir(mode=0o700, parents=True)
    total_bytes = 0
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        members = archive.getmembers()
        if len(members) > _SOURCE_ARCHIVE_MEMBER_LIMIT:
            raise ValueError("Scanner source snapshot exceeds the member limit")
        for member in members:
            pure_name = Path(member.name)
            if pure_name.is_absolute() or ".." in pure_name.parts:
                raise ValueError("Scanner source archive contains an unsafe path")
            output = destination / pure_name
            if not output.resolve(strict=False).is_relative_to(destination.resolve(strict=True)):
                raise ValueError("Scanner source archive escapes its snapshot root")
            if member.isdir():
                output.mkdir(mode=0o700, parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("Scanner source archive contains a non-regular member")
            total_bytes += member.size
            if total_bytes > _SOURCE_ARCHIVE_LIMIT_BYTES:
                raise ValueError("Scanner source snapshot exceeds the byte limit")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Scanner source archive member could not be read")
            output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with output.open("xb") as target:
                shutil.copyfileobj(stream, target, length=1024 * 1024)
            output.chmod(0o600)
    source_root = (destination / "src").resolve(strict=True)
    if not (source_root / "skillspector" / "__init__.py").is_file():
        raise ValueError("Scanner source snapshot is missing the package entrypoint")
    return source_root


def _copy_corpus_snapshot(
    corpus_root: Path,
    cases: list[dict[str, Any]],
    destination: Path,
) -> None:
    root = corpus_root.resolve(strict=True)
    destination.mkdir(mode=0o700, parents=True)
    copied: set[Path] = set()
    for case in sorted(cases, key=lambda item: str(item["id"])):
        source_case = _resolve_case_path(root, str(case["path"]))
        relative_case = source_case.relative_to(root)
        (destination / relative_case).mkdir(mode=0o700, parents=True, exist_ok=True)
        for source in sorted(source_case.rglob("*")):
            if source.is_symlink():
                raise ValueError(f"Corpus snapshot does not follow symlinks: {source}")
            relative = source.relative_to(root)
            target = destination / relative
            if source.is_dir():
                target.mkdir(mode=0o700, parents=True, exist_ok=True)
                continue
            if not source.is_file():
                raise ValueError(f"Corpus contains a non-regular path: {source}")
            resolved = source.resolve(strict=True)
            if resolved in copied:
                continue
            copied.add(resolved)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with source.open("rb") as source_stream, target.open("xb") as target_stream:
                shutil.copyfileobj(source_stream, target_stream, length=1024 * 1024)
            target.chmod(0o600)


@contextmanager
def _accuracy_snapshots(
    *,
    corpus_root: Path,
    cases: list[dict[str, Any]],
    manifest_bytes: bytes,
    corpus_identity: str,
    baseline_identity: dict[str, Any],
    candidate_identity: dict[str, Any],
) -> Iterator[dict[str, Path]]:
    """Yield private immutable-by-construction inputs used by both scans."""
    with tempfile.TemporaryDirectory(prefix="skillspector-accuracy-snapshot-") as temporary:
        snapshot_root = Path(temporary)
        snapshot_root.chmod(0o700)
        if snapshot_root.lstat().st_uid != os.geteuid() or any(snapshot_root.iterdir()):
            raise RuntimeError("Could not create a private empty accuracy snapshot root")
        corpus_snapshot = snapshot_root / "corpus"
        _copy_corpus_snapshot(corpus_root, cases, corpus_snapshot)
        if _corpus_identity(corpus_snapshot, cases, manifest_bytes) != corpus_identity:
            raise ValueError("Accuracy corpus changed while creating its private snapshot")

        baseline_source = _extract_source_archive(
            _git_source_archive(
                Path(baseline_identity["worktree"]),
                str(baseline_identity["revision"]),
            ),
            snapshot_root / "baseline",
        )
        candidate_source = _extract_source_archive(
            _git_source_archive(
                Path(candidate_identity["worktree"]),
                str(candidate_identity["revision"]),
            ),
            snapshot_root / "candidate",
        )
        runner = snapshot_root / "compare_scan_accuracy.py"
        runner.write_bytes(Path(__file__).resolve(strict=True).read_bytes())
        runner.chmod(0o500)
        expected_runner_sha256 = baseline_identity["source_runner_sha256"]
        if (
            expected_runner_sha256 != candidate_identity["source_runner_sha256"]
            or _file_sha256(runner) != expected_runner_sha256
        ):
            raise ValueError("Accuracy source runner changed while creating its private snapshot")
        yield {
            "corpus_root": corpus_snapshot,
            "baseline_source": baseline_source,
            "candidate_source": candidate_source,
            "runner": runner,
        }


def _console_python(executable: Path, worktree: Path) -> Path:
    """Validate the generated entrypoint and return its worktree-local Python."""
    raw = executable.read_bytes()
    shebang, separator, body = raw.partition(b"\n")
    if not separator or body != _CONSOLE_ENTRYPOINT_BODY:
        raise ValueError(
            f"Scanner executable is not the expected immutable SkillSpector entrypoint: {executable}"
        )
    try:
        shebang_text = shebang.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"Scanner executable has an invalid shebang: {executable}") from error
    if not shebang_text.startswith("#!"):
        raise ValueError(f"Scanner executable has no Python shebang: {executable}")
    interpreter_text = shebang_text[2:]
    if not interpreter_text or any(character.isspace() for character in interpreter_text):
        raise ValueError(f"Scanner executable has an ambiguous Python shebang: {executable}")
    interpreter = Path(interpreter_text)
    if not interpreter.is_absolute() or interpreter.parent != executable.parent:
        raise ValueError(
            f"Scanner executable must use a Python interpreter beside the entrypoint: {executable}"
        )
    # Check the lexical path before resolving a normal virtualenv interpreter
    # symlink to its shared base runtime.
    if not interpreter.is_relative_to(worktree):
        raise ValueError(f"Scanner Python interpreter must be inside its worktree: {interpreter}")
    resolved = interpreter.resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"Scanner Python interpreter is not executable: {resolved}")
    return interpreter


def _source_bound_command(
    *,
    executable: Path,
    target: Path | str,
    worktree: Path,
    source_root: Path | None = None,
    runner: Path | None = None,
) -> list[str]:
    """Build the exact isolated command that imports the declared worktree source."""
    interpreter = _console_python(executable, worktree)
    return [
        str(interpreter),
        "-I",
        "-B",
        str((runner or Path(__file__)).resolve(strict=True)),
        _SOURCE_SCAN_MODE,
        str((source_root or (worktree / "src")).resolve(strict=True)),
        str(target),
    ]


def _run_source_bound_scan(arguments: list[str]) -> int:
    """Internal subprocess mode: import only SkillSpector from the requested source tree."""
    if len(arguments) != 2:
        print("accuracy gate source runner received invalid arguments", file=sys.stderr)
        return 2
    source_root = Path(arguments[0]).resolve(strict=True)
    expected_package = (source_root / "skillspector").resolve(strict=True)
    target = Path(arguments[1]).resolve(strict=True)

    # -I prevents cwd/PYTHONPATH/user-site shadowing. The explicit first path and
    # fresh module namespace ensure an ignored installed package cannot win.
    sys.path.insert(0, str(source_root))
    for module_name in tuple(sys.modules):
        if module_name == "skillspector" or module_name.startswith("skillspector."):
            del sys.modules[module_name]
    try:
        skillspector = importlib.import_module("skillspector")
        package_location = getattr(skillspector, "__file__", None)
        if not isinstance(package_location, str):
            raise RuntimeError("scanner package import has no source file")
        package_file = Path(package_location).resolve(strict=True)
        expected_package_file = (expected_package / "__init__.py").resolve(strict=True)
        if package_file != expected_package_file:
            raise RuntimeError(f"scanner package import is not revision source: {package_file}")
        skillspector_cli = importlib.import_module("skillspector.cli")
        cli_location = getattr(skillspector_cli, "__file__", None)
        if not isinstance(cli_location, str):
            raise RuntimeError("scanner CLI import has no source file")
        cli_file = Path(cli_location).resolve(strict=True)
        expected_cli_file = (expected_package / "cli.py").resolve(strict=True)
        if cli_file != expected_cli_file:
            raise RuntimeError(f"scanner CLI import is not revision source: {cli_file}")
        app = skillspector_cli.app
    except Exception as error:
        print(f"accuracy gate source binding error: {error}", file=sys.stderr)
        return 2

    sys.argv = ["skillspector", "scan", str(target), "--format", "json", "--no-llm"]
    result = app()
    return result if isinstance(result, int) else 0


def _resolve_scanner_identity(
    *,
    executable: Path,
    worktree: Path,
    revision: str,
) -> dict[str, Any]:
    """Bind a revision label to one clean worktree and executable byte identity."""
    if not revision or revision != revision.strip():
        raise ValueError("Scanner revision must be a non-empty exact commit identity")
    root = worktree.resolve(strict=True)
    executable_path = executable.resolve(strict=True)
    if not executable_path.is_file():
        raise ValueError(f"Scanner executable is not a file: {executable_path}")
    if not os.access(executable_path, os.X_OK):
        raise ValueError(f"Scanner executable is not executable: {executable_path}")
    if not executable_path.is_relative_to(root):
        raise ValueError(f"Scanner executable must be inside its worktree: {executable_path}")

    actual_root = Path(_git_output(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if actual_root != root:
        raise ValueError(f"Scanner worktree must be the Git root: {root}")
    actual_revision = _git_output(root, "rev-parse", "HEAD").lower()
    if revision.strip().lower() != actual_revision:
        raise ValueError(
            f"Scanner revision mismatch for {root}: declared {revision}, actual {actual_revision}"
        )
    dirty = _git_output(root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValueError(f"Scanner worktree has changes and cannot be identified: {root}")

    source_root = (root / "src").resolve(strict=True)
    package_root = (source_root / "skillspector").resolve(strict=True)
    if not package_root.is_dir():
        raise ValueError(f"Scanner worktree has no SkillSpector source package: {package_root}")
    source_tree_oid = _git_output(root, "rev-parse", f"{actual_revision}:src/skillspector")
    tracked_source_files = set(
        _git_output(
            root,
            "ls-tree",
            "-r",
            "--name-only",
            actual_revision,
            "--",
            "src",
        ).splitlines()
    )
    required_source_files = {"src/skillspector/__init__.py", "src/skillspector/cli.py"}
    if not required_source_files.issubset(tracked_source_files):
        raise ValueError("Scanner revision is missing its tracked SkillSpector package entrypoints")
    actual_source_files = _actual_source_files(source_root)
    if actual_source_files != tracked_source_files:
        unexpected = sorted(actual_source_files - tracked_source_files)
        missing = sorted(tracked_source_files - actual_source_files)
        detail_parts = []
        if unexpected:
            detail_parts.append("unexpected: " + ", ".join(unexpected[:5]))
        if missing:
            detail_parts.append("missing: " + ", ".join(missing[:5]))
        raise ValueError(
            "Scanner source inventory differs from the committed revision ("
            + "; ".join(detail_parts)
            + ")"
        )
    interpreter = _console_python(executable_path, root)
    runner = Path(__file__).resolve(strict=True)
    runtime_identity = _runtime_identity(interpreter, root)
    pyproject = (root / "pyproject.toml").resolve(strict=True)
    lockfile = (root / "uv.lock").resolve(strict=True)

    return {
        "declared_revision": revision,
        "resolved_revision": actual_revision,
        "revision": actual_revision,
        "worktree": str(root),
        "executable": str(executable_path),
        "executable_relative_path": executable_path.relative_to(root).as_posix(),
        "executable_sha256": _file_sha256(executable_path),
        "python_executable": str(interpreter),
        "python_executable_resolved": str(interpreter.resolve(strict=True)),
        "python_executable_sha256": _file_sha256(interpreter),
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": runtime_identity["runtime_identity_sha256"],
        "dependency_identity_sha256": runtime_identity["dependency_identity_sha256"],
        "environment_sha256": runtime_identity["environment_sha256"],
        "pyproject_sha256": _file_sha256(pyproject),
        "lockfile": lockfile.name,
        "lockfile_sha256": _file_sha256(lockfile),
        "source_root": str(source_root),
        "source_tree_git_oid": source_tree_oid,
        "source_tree_revision": actual_revision,
        "source_binding": "isolated-worktree-source-import",
        "source_runner": str(runner),
        "source_runner_sha256": _file_sha256(runner),
        "worktree_clean": True,
        # Compatibility field retained from the initial identity contract.
        "tracked_worktree_clean": True,
    }


def _run_scan(
    executable: Path,
    target: Path,
    worktree: Path,
    *,
    source_root: Path | None = None,
    runner: Path | None = None,
) -> dict[str, Any]:
    command = _source_bound_command(
        executable=executable,
        target=target,
        worktree=worktree,
        source_root=source_root,
        runner=runner,
    )
    with _fresh_owned_home() as home:
        completed = _run_bounded(
            command,
            cwd=worktree,
            env=_scan_environment(home, disclose_home=True),
            timeout_seconds=_SCAN_TIMEOUT_SECONDS,
            stdout_limit_bytes=_SCAN_STDOUT_LIMIT_BYTES,
            stderr_limit_bytes=_STDERR_LIMIT_BYTES,
        )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(
            f"Scanner exited {completed.returncode} for {target.name}: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    try:
        report = json.loads(
            completed.stdout.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Scanner returned invalid JSON for {target.name}") from error
    if not isinstance(report, dict):
        raise RuntimeError(f"Scanner returned a non-object report for {target.name}")
    return report


def _nonnegative_integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Scanner JSON report has an invalid {field} field")
    return value


def _validate_complete_report(report: dict[str, Any]) -> None:
    """Reject any failed, partial, or structurally ambiguous accuracy input."""
    if report.get("execution_successful") is not True:
        raise ValueError("Scanner JSON report is not execution-successful")
    completeness = report.get("analysis_completeness")
    if not isinstance(completeness, dict):
        raise ValueError("Scanner JSON report has no analysis_completeness object")
    if completeness.get("execution_successful") is not True:
        raise ValueError("Scanner analysis completeness is not execution-successful")
    if completeness.get("is_complete") is not True or completeness.get("status") != "complete":
        raise ValueError("Scanner JSON report is not analysis-complete")
    total_components = _nonnegative_integer(
        completeness.get("total_components"),
        "analysis_completeness.total_components",
    )
    scanned_components = _nonnegative_integer(
        completeness.get("scanned_components"),
        "analysis_completeness.scanned_components",
    )
    fully_inspected = _nonnegative_integer(
        completeness.get("fully_inspected_files"),
        "analysis_completeness.fully_inspected_files",
    )
    partially_inspected = _nonnegative_integer(
        completeness.get("partially_inspected_files"),
        "analysis_completeness.partially_inspected_files",
    )
    entirely_uninspected = _nonnegative_integer(
        completeness.get("entirely_uninspected_files"),
        "analysis_completeness.entirely_uninspected_files",
    )
    if scanned_components != total_components or partially_inspected or entirely_uninspected:
        raise ValueError("Scanner JSON report completeness counters describe incomplete coverage")
    if fully_inspected != total_components:
        raise ValueError("Scanner JSON report has inconsistent fully-inspected coverage")
    coverage = completeness.get("coverage_percent")
    if not isinstance(coverage, (int, float)) or isinstance(coverage, bool) or coverage != 100:
        raise ValueError("Scanner JSON report coverage_percent must be exactly 100")
    for field in ("ledger_exceptions", "scope_exclusions", "limitations"):
        value = completeness.get(field)
        if not isinstance(value, list) or value:
            raise ValueError(f"Scanner JSON report has non-empty or invalid {field}")
    analyzer_statuses = completeness.get("analyzer_statuses")
    if not isinstance(analyzer_statuses, list):
        raise ValueError("Scanner JSON report has invalid analyzer_statuses")
    for index, status in enumerate(analyzer_statuses):
        if (
            not isinstance(status, dict)
            or not isinstance(status.get("analyzer_id"), str)
            or not status["analyzer_id"].strip()
            or status.get("status") not in {"completed", "not_applicable", "disabled"}
        ):
            raise ValueError(f"Scanner JSON report has an invalid analyzer status at index {index}")


def _rule_counts(report: dict[str, Any], selected_rules: frozenset[str]) -> Counter[str]:
    _validate_complete_report(report)
    if "issues" not in report:
        raise ValueError("Scanner JSON report is missing the 'issues' field")
    issues = report["issues"]
    if not isinstance(issues, list):
        raise ValueError("Scanner JSON report has a non-list 'issues' field")
    counts: Counter[str] = Counter()
    for index, issue in enumerate(issues):
        if (
            not isinstance(issue, dict)
            or not isinstance(issue.get("id"), str)
            or not issue["id"].strip()
        ):
            raise ValueError(f"Scanner JSON report has an invalid issue at index {index}")
        rule_id = issue["id"]
        raw_occurrences = issue.get("occurrences")
        occurrence_count = 1
        if raw_occurrences is not None:
            if not isinstance(raw_occurrences, list):
                raise ValueError(
                    f"Scanner JSON report has non-list occurrences at issue index {index}"
                )
            if any(not isinstance(occurrence, dict) for occurrence in raw_occurrences):
                raise ValueError(
                    f"Scanner JSON report has an invalid occurrence at issue index {index}"
                )
            occurrence_count = max(1, len(raw_occurrences))
        if not selected_rules or rule_id in selected_rules:
            counts[rule_id] += occurrence_count
    return counts


def _expected_range(value: object) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value, value
    if isinstance(value, dict):
        unknown_fields = set(value) - {"min", "max"}
        if unknown_fields:
            raise ValueError(
                "Expected rule count range has unknown field(s): "
                + ", ".join(sorted(str(field) for field in unknown_fields))
            )
        minimum = value.get("min", 0)
        maximum = value.get("max", minimum)
        if (
            isinstance(minimum, int)
            and not isinstance(minimum, bool)
            and isinstance(maximum, int)
            and not isinstance(maximum, bool)
            and 0 <= minimum <= maximum
        ):
            return minimum, maximum
    raise ValueError("Expected rule counts must be a non-negative integer or {min, max} object")


def _adjudicate_counts(
    case: dict[str, Any],
    counts: Counter[str],
    selected_rules: frozenset[str],
) -> dict[str, Any]:
    raw_expected = case.get("expected_rules", {})
    if not isinstance(raw_expected, dict):
        raise ValueError(f"Case {case['id']} has a non-object expected_rules field")
    expected: dict[str, tuple[int, int]] = {}
    for rule_id, value in raw_expected.items():
        if not isinstance(rule_id, str) or not rule_id.strip() or rule_id != rule_id.strip():
            raise ValueError(f"Case {case['id']} has an empty expected rule id")
        expected_range = _expected_range(value)
        if not selected_rules or rule_id in selected_rules:
            expected[rule_id] = expected_range
    errors: list[str] = []
    by_rule: dict[str, dict[str, int]] = {}
    for rule_id in sorted(set(expected) | set(counts)):
        minimum, maximum = expected.get(rule_id, (0, 0))
        actual = counts.get(rule_id, 0)
        false_positives = max(0, actual - maximum)
        false_negatives = max(0, minimum - actual)
        by_rule[rule_id] = {
            "expected_min": minimum,
            "expected_max": maximum,
            "observed": actual,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }
        if false_positives or false_negatives:
            errors.append(f"{rule_id}: expected {minimum}..{maximum}, observed {actual}")
    return {
        "false_positives": sum(item["false_positives"] for item in by_rule.values()),
        "false_negatives": sum(item["false_negatives"] for item in by_rule.values()),
        "by_rule": by_rule,
        "errors": errors,
    }


def _validate_policy(manifest: dict[str, Any]) -> dict[str, int]:
    raw_policy = manifest.get("material_regression_policy")
    if not isinstance(raw_policy, dict):
        raise ValueError("Accuracy manifest needs a material_regression_policy object")
    unknown_fields = sorted(set(raw_policy) - set(POLICY_FIELDS))
    if unknown_fields:
        raise ValueError(
            "Unknown material_regression_policy field(s): " + ", ".join(unknown_fields)
        )
    policy: dict[str, int] = {}
    for field in POLICY_FIELDS:
        value = raw_policy.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"material_regression_policy.{field} must be a non-negative integer")
        policy[field] = value
    return policy


def _aggregate_adjudication(
    case_results: list[dict[str, Any]],
    scanner: str,
    classification: str | None = None,
) -> dict[str, Any]:
    false_positives: Counter[str] = Counter()
    false_negatives: Counter[str] = Counter()
    for case in case_results:
        if classification is not None and case["classification"] != classification:
            continue
        for rule_id, values in case["adjudication"][scanner]["by_rule"].items():
            false_positives[rule_id] += values["false_positives"]
            false_negatives[rule_id] += values["false_negatives"]
    rules = sorted(set(false_positives) | set(false_negatives))
    return {
        "false_positives": sum(false_positives.values()),
        "false_negatives": sum(false_negatives.values()),
        "by_rule": {
            rule_id: {
                "false_positives": false_positives[rule_id],
                "false_negatives": false_negatives[rule_id],
            }
            for rule_id in rules
        },
    }


def _adjudication_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    rules = sorted(set(baseline["by_rule"]) | set(candidate["by_rule"]))
    return {
        "false_positives": candidate["false_positives"] - baseline["false_positives"],
        "false_negatives": candidate["false_negatives"] - baseline["false_negatives"],
        "by_rule": {
            rule_id: {
                "false_positives": candidate["by_rule"].get(rule_id, {}).get("false_positives", 0)
                - baseline["by_rule"].get(rule_id, {}).get("false_positives", 0),
                "false_negatives": candidate["by_rule"].get(rule_id, {}).get("false_negatives", 0)
                - baseline["by_rule"].get(rule_id, {}).get("false_negatives", 0),
            }
            for rule_id in rules
        },
    }


def _material_regressions(
    *,
    policy: dict[str, int],
    candidate: dict[str, Any],
    delta: dict[str, Any],
    by_classification: dict[str, dict[str, Any]],
    case_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    regressions: list[dict[str, Any]] = []

    def check(
        metric: str,
        observed: int,
        limit_field: str,
        *,
        scope: str,
        rule_id: str | None = None,
        classification: str | None = None,
        case_id: str | None = None,
    ) -> None:
        limit = policy[limit_field]
        if observed > limit:
            record: dict[str, Any] = {
                "metric": metric,
                "observed": observed,
                "limit": limit,
                "scope": scope,
            }
            if rule_id is not None:
                record["rule_id"] = rule_id
            if classification is not None:
                record["classification"] = classification
            if case_id is not None:
                record["case_id"] = case_id
            regressions.append(record)

    check(
        "candidate_false_positives",
        candidate["false_positives"],
        "max_candidate_false_positives",
        scope="global",
    )
    check(
        "candidate_false_negatives",
        candidate["false_negatives"],
        "max_candidate_false_negatives",
        scope="global",
    )
    check(
        "false_positive_increase",
        delta["false_positives"],
        "max_false_positive_increase",
        scope="global",
    )
    check(
        "false_negative_increase",
        delta["false_negatives"],
        "max_false_negative_increase",
        scope="global",
    )
    for rule_id, values in delta["by_rule"].items():
        check(
            "per_rule_false_positive_increase",
            values["false_positives"],
            "max_per_rule_false_positive_increase",
            scope="global",
            rule_id=rule_id,
        )
        check(
            "per_rule_false_negative_increase",
            values["false_negatives"],
            "max_per_rule_false_negative_increase",
            scope="global",
            rule_id=rule_id,
        )
    for classification, adjudication in sorted(by_classification.items()):
        cohort_delta = adjudication["delta"]
        check(
            "cohort_false_positive_increase",
            cohort_delta["false_positives"],
            "max_per_cohort_false_positive_increase",
            scope="cohort",
            classification=classification,
        )
        check(
            "cohort_false_negative_increase",
            cohort_delta["false_negatives"],
            "max_per_cohort_false_negative_increase",
            scope="cohort",
            classification=classification,
        )
        for rule_id, values in cohort_delta["by_rule"].items():
            check(
                "per_rule_false_positive_increase",
                values["false_positives"],
                "max_per_rule_false_positive_increase",
                scope="cohort",
                classification=classification,
                rule_id=rule_id,
            )
            check(
                "per_rule_false_negative_increase",
                values["false_negatives"],
                "max_per_rule_false_negative_increase",
                scope="cohort",
                classification=classification,
                rule_id=rule_id,
            )
    for case in sorted(case_results, key=lambda item: str(item["id"])):
        case_delta = case["adjudication"]["delta"]
        check(
            "case_false_positive_increase",
            case_delta["false_positives"],
            "max_per_case_false_positive_increase",
            scope="case",
            classification=case["classification"],
            case_id=case["id"],
        )
        check(
            "case_false_negative_increase",
            case_delta["false_negatives"],
            "max_per_case_false_negative_increase",
            scope="case",
            classification=case["classification"],
            case_id=case["id"],
        )
        for rule_id, values in case_delta["by_rule"].items():
            check(
                "per_rule_false_positive_increase",
                values["false_positives"],
                "max_per_rule_false_positive_increase",
                scope="case",
                classification=case["classification"],
                case_id=case["id"],
                rule_id=rule_id,
            )
            check(
                "per_rule_false_negative_increase",
                values["false_negatives"],
                "max_per_rule_false_negative_increase",
                scope="case",
                classification=case["classification"],
                case_id=case["id"],
                rule_id=rule_id,
            )
    return regressions


def _approval_metadata(
    artifact: Path | None,
    reviewer: str | None,
    *,
    corpus_identity: str,
    manifest_sha256: str,
    baseline_identity: dict[str, Any],
    candidate_identity: dict[str, Any],
    policy: dict[str, int],
    violations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    has_reviewer = isinstance(reviewer, str) and bool(reviewer.strip())
    if (artifact is None) != (not has_reviewer):
        raise ValueError("Material-regression approval requires both artifact and reviewer")
    if artifact is None:
        return None
    assert reviewer is not None
    artifact_path = artifact.resolve(strict=True)
    if not artifact_path.is_file():
        raise ValueError(f"Approval artifact is not a file: {artifact_path}")
    raw = artifact_path.read_bytes()
    document = _load_json_bytes(raw, artifact_path)
    unknown_fields = sorted(set(document) - APPROVAL_FIELDS)
    if unknown_fields:
        raise ValueError("Unknown approval artifact field(s): " + ", ".join(unknown_fields))
    if document.get("schema_version") != 1:
        raise ValueError("Material-regression approval schema_version must be 1")
    document_reviewer = document.get("reviewer")
    if (
        not isinstance(document_reviewer, str)
        or not document_reviewer.strip()
        or document_reviewer != document_reviewer.strip()
        or document_reviewer != reviewer.strip()
    ):
        raise ValueError("Material-regression approval reviewer does not match")
    rationale = document.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Material-regression approval needs a non-empty rationale")

    expected_baseline = {field: baseline_identity[field] for field in APPROVAL_IDENTITY_FIELDS}
    expected_candidate = {field: candidate_identity[field] for field in APPROVAL_IDENTITY_FIELDS}
    expected = {
        "corpus_identity": corpus_identity,
        "manifest_sha256": manifest_sha256,
        "baseline_identity": expected_baseline,
        "candidate_identity": expected_candidate,
        "policy_sha256": _json_sha256(policy),
        "violations": violations,
        "violations_sha256": _json_sha256(violations),
    }
    for field, expected_value in expected.items():
        if document.get(field) != expected_value:
            raise ValueError(f"Material-regression approval is not bound to the exact {field}")
    return {
        "reviewer": document_reviewer,
        "artifact": str(artifact_path),
        "artifact_sha256": f"sha256:{hashlib.sha256(raw).hexdigest()}",
        "binding_sha256": _json_sha256(expected),
        "authorization": "evidence-only-untrusted-local-artifact",
    }


def _snapshot_command_evidence(
    command: list[str],
    *,
    scanner: str,
    case_path: str,
) -> list[str]:
    """Replace random private snapshot paths with deterministic evidence markers."""
    if len(command) != 7 or command[4] != _SOURCE_SCAN_MODE:
        raise ValueError("Accuracy scanner command does not match the source-bound contract")
    evidence = list(command)
    evidence[3] = "<private-source-runner-snapshot>"
    evidence[5] = f"<private-{scanner}-source-snapshot>"
    evidence[6] = f"<private-corpus-snapshot>/{case_path}"
    return evidence


def _scan_accuracy_cases(
    *,
    cases: list[dict[str, Any]],
    snapshots: dict[str, Path],
    baseline_identity: dict[str, Any],
    candidate_identity: dict[str, Any],
    baseline_scan_executable: Path,
    candidate_scan_executable: Path,
    selected_rules: frozenset[str],
) -> tuple[Counter[str], Counter[str], list[dict[str, Any]]]:
    aggregate_baseline: Counter[str] = Counter()
    aggregate_candidate: Counter[str] = Counter()
    case_results: list[dict[str, Any]] = []
    for case in sorted(cases, key=lambda item: str(item["id"])):
        target = _resolve_case_path(snapshots["corpus_root"], str(case["path"]))
        baseline_report = _run_scan(
            baseline_scan_executable,
            target,
            Path(baseline_identity["worktree"]),
            source_root=snapshots["baseline_source"],
            runner=snapshots["runner"],
        )
        candidate_report = _run_scan(
            candidate_scan_executable,
            target,
            Path(candidate_identity["worktree"]),
            source_root=snapshots["candidate_source"],
            runner=snapshots["runner"],
        )
        baseline_counts = _rule_counts(baseline_report, selected_rules)
        candidate_counts = _rule_counts(candidate_report, selected_rules)
        baseline_adjudication = _adjudicate_counts(case, baseline_counts, selected_rules)
        candidate_adjudication = _adjudicate_counts(case, candidate_counts, selected_rules)
        adjudication_delta = _adjudication_delta(
            baseline_adjudication,
            candidate_adjudication,
        )
        aggregate_baseline.update(baseline_counts)
        aggregate_candidate.update(candidate_counts)
        all_rules = sorted(set(baseline_counts) | set(candidate_counts))
        case_results.append(
            {
                "id": case["id"],
                "path": case["path"],
                "classification": case["classification"],
                "scan_execution": {
                    "private_input_snapshot": True,
                    "baseline": {
                        "command": _snapshot_command_evidence(
                            _source_bound_command(
                                executable=baseline_scan_executable,
                                target=target,
                                worktree=Path(baseline_identity["worktree"]),
                                source_root=snapshots["baseline_source"],
                                runner=snapshots["runner"],
                            ),
                            scanner="baseline",
                            case_path=str(case["path"]),
                        ),
                        "working_directory": baseline_identity["worktree"],
                        "source_root": "<private-baseline-source-snapshot>",
                    },
                    "candidate": {
                        "command": _snapshot_command_evidence(
                            _source_bound_command(
                                executable=candidate_scan_executable,
                                target=target,
                                worktree=Path(candidate_identity["worktree"]),
                                source_root=snapshots["candidate_source"],
                                runner=snapshots["runner"],
                            ),
                            scanner="candidate",
                            case_path=str(case["path"]),
                        ),
                        "working_directory": candidate_identity["worktree"],
                        "source_root": "<private-candidate-source-snapshot>",
                    },
                },
                "baseline": dict(sorted(baseline_counts.items())),
                "candidate": dict(sorted(candidate_counts.items())),
                "delta": {
                    rule_id: candidate_counts[rule_id] - baseline_counts[rule_id]
                    for rule_id in all_rules
                },
                "adjudication": {
                    "baseline": baseline_adjudication,
                    "candidate": candidate_adjudication,
                    "delta": adjudication_delta,
                },
                # Compatibility field retained as an explicit candidate-adjudication view.
                "adjudication_errors": candidate_adjudication["errors"],
            }
        )
    return aggregate_baseline, aggregate_candidate, case_results


def compare_scanners(
    *,
    manifest_path: Path,
    corpus_root: Path,
    baseline_executable: Path,
    candidate_executable: Path,
    baseline_worktree: Path,
    candidate_worktree: Path,
    baseline_revision: str,
    candidate_revision: str,
    invocation: list[str],
    selected_rules: frozenset[str] = frozenset(),
    approval_artifact: Path | None = None,
    approval_reviewer: str | None = None,
) -> dict[str, Any]:
    """Run both scanners and return deterministic accuracy evidence."""
    if not invocation or any(
        not isinstance(argument, str) or not argument for argument in invocation
    ):
        raise ValueError("Accuracy evidence requires the exact non-empty invocation")
    if selected_rules:
        raise ValueError("Partial rule selection is not allowed in the accuracy gate")
    has_reviewer = isinstance(approval_reviewer, str) and bool(approval_reviewer.strip())
    if (approval_artifact is None) != (not has_reviewer):
        raise ValueError("Material-regression approval requires both artifact and reviewer")
    manifest, manifest_bytes = _load_manifest(manifest_path)
    unknown_manifest_fields = sorted(set(manifest) - MANIFEST_FIELDS)
    if unknown_manifest_fields:
        raise ValueError(
            "Unknown accuracy manifest field(s): " + ", ".join(unknown_manifest_fields)
        )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Accuracy manifest schema_version must be {SCHEMA_VERSION}")
    policy = _validate_policy(manifest)
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Accuracy manifest must contain a non-empty cases list")
    cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    case_paths: set[Path] = set()
    classifications: set[str] = set()
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("Each accuracy case must be a JSON object")
        unknown_case_fields = sorted(set(raw_case) - CASE_FIELDS)
        if unknown_case_fields:
            raise ValueError("Unknown accuracy case field(s): " + ", ".join(unknown_case_fields))
        case_id = raw_case.get("id")
        path = raw_case.get("path")
        classification = raw_case.get("classification")
        if (
            not isinstance(case_id, str)
            or not case_id.strip()
            or case_id != case_id.strip()
            or case_id in case_ids
        ):
            raise ValueError("Each accuracy case needs a unique non-empty id")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"Case {case_id} needs a non-empty path")
        if (
            not isinstance(classification, str)
            or not classification.strip()
            or classification != classification.strip()
            or classification not in REQUIRED_CLASSIFICATIONS
        ):
            raise ValueError(
                f"Case {case_id} classification must be one of: "
                + ", ".join(sorted(REQUIRED_CLASSIFICATIONS))
            )
        if not isinstance(raw_case.get("expected_rules"), dict):
            raise ValueError(f"Case {case_id} needs an explicit expected_rules object")
        # Validate every adjudication, including rules outside an optional CLI filter,
        # before either scanner executes.
        _adjudicate_counts(raw_case, Counter(), frozenset())
        resolved_case_path = _resolve_case_path(corpus_root, path)
        if resolved_case_path in case_paths:
            raise ValueError(f"Accuracy cases must reference unique corpus paths: {path}")
        overlapping_path = next(
            (
                existing
                for existing in case_paths
                if resolved_case_path.is_relative_to(existing)
                or existing.is_relative_to(resolved_case_path)
            ),
            None,
        )
        if overlapping_path is not None:
            raise ValueError(
                "Accuracy case roots must not overlap as ancestors or descendants: "
                f"{path} and {overlapping_path.relative_to(corpus_root.resolve(strict=True))}"
            )
        case_ids.add(case_id)
        case_paths.add(resolved_case_path)
        classifications.add(classification)
        cases.append(raw_case)
    missing_classifications = sorted(REQUIRED_CLASSIFICATIONS - classifications)
    if missing_classifications:
        raise ValueError(
            "Accuracy manifest is missing required classifications: "
            + ", ".join(missing_classifications)
        )

    corpus_identity = _corpus_identity(corpus_root, cases, manifest_bytes)
    manifest_sha256 = f"sha256:{hashlib.sha256(manifest_bytes).hexdigest()}"

    baseline_identity = _resolve_scanner_identity(
        executable=baseline_executable,
        worktree=baseline_worktree,
        revision=baseline_revision,
    )
    candidate_identity = _resolve_scanner_identity(
        executable=candidate_executable,
        worktree=candidate_worktree,
        revision=candidate_revision,
    )
    baseline_scan_executable = Path(baseline_identity["executable"])
    candidate_scan_executable = Path(candidate_identity["executable"])

    with _accuracy_snapshots(
        corpus_root=corpus_root,
        cases=cases,
        manifest_bytes=manifest_bytes,
        corpus_identity=corpus_identity,
        baseline_identity=baseline_identity,
        candidate_identity=candidate_identity,
    ) as snapshots:
        aggregate_baseline, aggregate_candidate, case_results = _scan_accuracy_cases(
            cases=cases,
            snapshots=snapshots,
            baseline_identity=baseline_identity,
            candidate_identity=candidate_identity,
            baseline_scan_executable=baseline_scan_executable,
            candidate_scan_executable=candidate_scan_executable,
            selected_rules=selected_rules,
        )

    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("Accuracy manifest changed during comparison")
    if _corpus_identity(corpus_root, cases, manifest_bytes) != corpus_identity:
        raise ValueError("Accuracy corpus changed during comparison")
    final_baseline_identity = _resolve_scanner_identity(
        executable=baseline_scan_executable,
        worktree=baseline_worktree,
        revision=baseline_revision,
    )
    final_candidate_identity = _resolve_scanner_identity(
        executable=candidate_scan_executable,
        worktree=candidate_worktree,
        revision=candidate_revision,
    )
    if final_baseline_identity != baseline_identity:
        raise ValueError("Baseline scanner identity changed during comparison")
    if final_candidate_identity != candidate_identity:
        raise ValueError("Candidate scanner identity changed during comparison")
    baseline_adjudication = _aggregate_adjudication(case_results, "baseline")
    candidate_adjudication = _aggregate_adjudication(case_results, "candidate")
    adjudication_delta = _adjudication_delta(baseline_adjudication, candidate_adjudication)
    adjudication_by_classification: dict[str, dict[str, Any]] = {}
    for classification in sorted(classifications):
        classification_baseline = _aggregate_adjudication(
            case_results,
            "baseline",
            classification,
        )
        classification_candidate = _aggregate_adjudication(
            case_results,
            "candidate",
            classification,
        )
        adjudication_by_classification[classification] = {
            "baseline": classification_baseline,
            "candidate": classification_candidate,
            "delta": _adjudication_delta(
                classification_baseline,
                classification_candidate,
            ),
        }
    regressions = _material_regressions(
        policy=policy,
        candidate=candidate_adjudication,
        delta=adjudication_delta,
        by_classification=adjudication_by_classification,
        case_results=case_results,
    )
    approval = _approval_metadata(
        approval_artifact,
        approval_reviewer,
        corpus_identity=corpus_identity,
        manifest_sha256=manifest_sha256,
        baseline_identity=baseline_identity,
        candidate_identity=candidate_identity,
        policy=policy,
        violations=regressions,
    )
    # A local JSON document is useful review evidence, but it has no
    # authentication root and therefore can never authorize a policy bypass.
    approved = False
    all_rules = sorted(
        set(aggregate_baseline)
        | set(aggregate_candidate)
        | set(baseline_adjudication["by_rule"])
        | set(candidate_adjudication["by_rule"])
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "corpus_identity": corpus_identity,
        # Compatibility alias; unlike v1, this includes manifest/adjudication bytes.
        "corpus_snapshot": corpus_identity,
        "manifest": manifest_path.name,
        "manifest_sha256": manifest_sha256,
        "baseline": {
            **baseline_identity,
            "working_directory": baseline_identity["worktree"],
            "command": _source_bound_command(
                executable=baseline_scan_executable,
                target="<case>",
                worktree=Path(baseline_identity["worktree"]),
            ),
        },
        "candidate": {
            **candidate_identity,
            "working_directory": candidate_identity["worktree"],
            "command": _source_bound_command(
                executable=candidate_scan_executable,
                target="<case>",
                worktree=Path(candidate_identity["worktree"]),
            ),
        },
        "selected_rules": sorted(selected_rules),
        "count_unit": "occurrence",
        "required_classifications": sorted(REQUIRED_CLASSIFICATIONS),
        "observed_classifications": sorted(classifications),
        "execution": {
            "invocation": list(invocation),
            "configuration": {
                "manifest": str(manifest_path.resolve(strict=True)),
                "corpus_root": str(corpus_root.resolve(strict=True)),
                "baseline_executable": baseline_identity["executable"],
                "candidate_executable": candidate_identity["executable"],
                "baseline_worktree": baseline_identity["worktree"],
                "candidate_worktree": candidate_identity["worktree"],
                "baseline_revision": baseline_identity["revision"],
                "candidate_revision": candidate_identity["revision"],
                "selected_rules": sorted(selected_rules),
                "scan_arguments": ["scan", "<case>", "--format", "json", "--no-llm"],
                "source_binding": "private-git-object-and-corpus-snapshot",
                "source_runner": baseline_identity["source_runner"],
                "source_runner_sha256": baseline_identity["source_runner_sha256"],
                "baseline_environment": baseline_identity["runtime_identity"]["environment"],
                "baseline_environment_sha256": baseline_identity["environment_sha256"],
                "candidate_environment": candidate_identity["runtime_identity"]["environment"],
                "candidate_environment_sha256": candidate_identity["environment_sha256"],
                "approval_artifact": approval["artifact"] if approval else None,
                "approval_reviewer": approval["reviewer"] if approval else None,
            },
            "inputs_verified_unchanged": True,
        },
        "per_rule": {
            rule_id: {
                "baseline": aggregate_baseline[rule_id],
                "candidate": aggregate_candidate[rule_id],
                "delta": aggregate_candidate[rule_id] - aggregate_baseline[rule_id],
                "baseline_false_positives": baseline_adjudication["by_rule"]
                .get(rule_id, {})
                .get("false_positives", 0),
                "candidate_false_positives": candidate_adjudication["by_rule"]
                .get(rule_id, {})
                .get("false_positives", 0),
                "false_positive_delta": adjudication_delta["by_rule"]
                .get(rule_id, {})
                .get("false_positives", 0),
                "baseline_false_negatives": baseline_adjudication["by_rule"]
                .get(rule_id, {})
                .get("false_negatives", 0),
                "candidate_false_negatives": candidate_adjudication["by_rule"]
                .get(rule_id, {})
                .get("false_negatives", 0),
                "false_negative_delta": adjudication_delta["by_rule"]
                .get(rule_id, {})
                .get("false_negatives", 0),
            }
            for rule_id in all_rules
        },
        "cases": case_results,
        "adjudication": {
            "baseline": baseline_adjudication,
            "candidate": candidate_adjudication,
            "delta": adjudication_delta,
            "by_classification": adjudication_by_classification,
        },
        "material_regression": {
            "policy": policy,
            "violations": regressions,
            "approval": approval,
            "approved": approved,
        },
    }
    result["passed"] = not regressions
    if approval and _file_sha256(Path(approval["artifact"])) != approval["artifact_sha256"]:
        raise ValueError("Material-regression approval artifact changed during comparison")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-executable",
        type=Path,
        required=True,
        help="executable file inside the baseline worktree",
    )
    parser.add_argument(
        "--candidate-executable",
        type=Path,
        required=True,
        help="executable file inside the candidate worktree",
    )
    parser.add_argument(
        "--baseline-worktree",
        type=Path,
        required=True,
        help="clean baseline Git worktree root used as the scanner working directory",
    )
    parser.add_argument(
        "--candidate-worktree",
        type=Path,
        required=True,
        help="clean candidate Git worktree root used as the scanner working directory",
    )
    parser.add_argument(
        "--baseline-revision",
        required=True,
        help="full commit identity that must exactly match baseline-worktree HEAD",
    )
    parser.add_argument(
        "--candidate-revision",
        required=True,
        help="full commit identity that must exactly match candidate-worktree HEAD",
    )
    parser.add_argument(
        "--approval-artifact",
        type=Path,
        help=(
            "schema-v1 JSON review record bound to this exact evidence set; "
            "recorded for audit only and never waives regressions"
        ),
    )
    parser.add_argument(
        "--approval-reviewer",
        help="named reviewer required with --approval-artifact for audit attribution",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if effective_argv and effective_argv[0] == _SOURCE_SCAN_MODE:
        return _run_source_bound_scan(effective_argv[1:])
    args = _parser().parse_args(effective_argv)
    invocation = [str(Path(sys.argv[0]).resolve()), *effective_argv]
    try:
        result = compare_scanners(
            manifest_path=args.manifest,
            corpus_root=args.corpus_root,
            baseline_executable=args.baseline_executable,
            candidate_executable=args.candidate_executable,
            baseline_worktree=args.baseline_worktree,
            candidate_worktree=args.candidate_worktree,
            baseline_revision=args.baseline_revision,
            candidate_revision=args.candidate_revision,
            invocation=invocation,
            selected_rules=frozenset(),
            approval_artifact=args.approval_artifact,
            approval_reviewer=args.approval_reviewer,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"accuracy gate error: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
