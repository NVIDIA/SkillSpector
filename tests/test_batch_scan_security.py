# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch enhancements must reuse the core scanner's provider-eligible snapshot."""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from contrib.batch_scan import batch_scan, runner
from contrib.batch_scan.gap_fill import run_gap_fill
from skillspector import llm_analyzer_base
from skillspector.nodes.build_context import build_context

_SECRET = "DUMMY_EXTERNAL_SECRET_NEVER_SEND"
_SAFE_TEXT = "# 安全助手\n这是一个帮助用户整理资料的安全技能。\n"


@pytest.fixture
def batch_skill(tmp_path: Path) -> tuple[Path, Path]:
    skill = tmp_path / "safe-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(_SAFE_TEXT, encoding="utf-8")
    secret = tmp_path / "outside.txt"
    secret.write_text(_SECRET, encoding="utf-8")
    try:
        (skill / "notes_zh.txt").symlink_to(secret)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")
    return skill, secret


def _mock_scan(monkeypatch: pytest.MonkeyPatch, mutate=None) -> dict:
    observed: dict = {"calls": []}

    def invoke(state):
        context = build_context({"skill_path": state["input_path"]})
        observed["context"] = context
        if mutate is not None:
            mutate(context)
        return context

    def gap_fill(file_cache, language, **kwargs):
        observed["calls"].append((file_cache, language))
        return []

    monkeypatch.setattr(runner.graph, "invoke", invoke)
    # Patch both locations so the same regression also exercises the old reader.
    monkeypatch.setattr(runner, "run_gap_fill", gap_fill, raising=False)
    monkeypatch.setattr(batch_scan, "run_gap_fill", gap_fill, raising=False)
    return observed


@pytest.mark.parametrize("language", ["auto", "zh"])
@pytest.mark.parametrize("replace_after_snapshot", [False, True])
def test_gap_fill_reuses_safe_snapshot(
    batch_skill, monkeypatch: pytest.MonkeyPatch, language, replace_after_snapshot
) -> None:
    skill, secret = batch_skill
    notes = skill / "notes_zh.txt"
    if replace_after_snapshot:
        notes.unlink()
        notes.write_text(_SAFE_TEXT, encoding="utf-8")

    def replace_file(context):
        if replace_after_snapshot:
            notes.unlink()
            notes.symlink_to(secret)

    observed = _mock_scan(monkeypatch, replace_file)
    entry, error, name = batch_scan._scan_skill(
        skill, skill.parent, use_llm=True, lang=language, require_llm=True
    )

    assert error is None, error
    assert name == skill.name
    assert len(observed["calls"]) == 1
    sent_cache, sent_language = observed["calls"][0]
    assert _SECRET not in "\n".join(sent_cache.values())
    assert sent_cache["SKILL.md"] == _SAFE_TEXT
    assert sent_cache is observed["context"]["llm_file_cache"]
    assert ("notes_zh.txt" in sent_cache) is replace_after_snapshot
    assert sent_language == entry["skill"]["language"] == "zh"
    assert entry["enhancements"]["gap_fill_applied"] is True


def test_gap_fill_provider_prompt_excludes_symlink_target(
    batch_skill, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill, _ = batch_skill
    _mock_scan(monkeypatch)
    monkeypatch.setattr(runner, "run_gap_fill", run_gap_fill)
    monkeypatch.setattr(batch_scan, "run_gap_fill", run_gap_fill)
    prompts = []

    def invoke(prompt):
        prompts.append(prompt)
        return AIMessage(content='{"findings": []}')

    monkeypatch.setattr(
        llm_analyzer_base,
        "get_chat_model",
        lambda **kwargs: SimpleNamespace(invoke=invoke),
    )
    monkeypatch.setattr(llm_analyzer_base, "get_max_input_tokens", lambda model: 100_000)

    entry, error, _ = batch_scan._scan_skill(
        skill, skill.parent, use_llm=True, lang="zh", require_llm=True
    )

    assert error is None, error
    assert len(prompts) == 1
    assert _SAFE_TEXT.splitlines()[1] in prompts[0]
    assert _SECRET not in prompts[0]
    assert entry["issues"] == []


@pytest.mark.parametrize("cache_state", ["empty", "missing"])
def test_gap_fill_never_falls_back_to_local_content(
    batch_skill, monkeypatch: pytest.MonkeyPatch, cache_state
) -> None:
    skill, _ = batch_skill

    def remove_provider_content(context):
        context["llm_file_cache"] = {}
        if cache_state == "missing":
            context.pop("llm_file_cache")
        for key in ("file_cache", "raw_file_cache", "local_file_cache"):
            context[key] = {"local-only.txt": _SECRET}

    observed = _mock_scan(monkeypatch, remove_provider_content)
    entry, error, _ = batch_scan._scan_skill(
        skill, skill.parent, use_llm=True, lang="zh", require_llm=True
    )

    assert error is None, error
    assert observed["calls"] == [({}, "zh")]
    assert entry["skill"]["language"] == "zh"


@pytest.mark.parametrize(
    ("language", "use_llm", "expected_language"),
    [("en", True, "en"), ("auto", False, "zh")],
)
def test_gap_fill_respects_language_and_no_llm(
    batch_skill, monkeypatch: pytest.MonkeyPatch, language, use_llm, expected_language
) -> None:
    skill, _ = batch_skill
    observed = _mock_scan(monkeypatch)

    entry, error, _ = batch_scan._scan_skill(
        skill, skill.parent, use_llm=use_llm, lang=language, require_llm=True
    )

    assert error is None, error
    assert observed["calls"] == []
    assert entry["skill"]["language"] == expected_language
    assert entry["enhancements"]["gap_fill_applied"] is False


@pytest.mark.parametrize("apply_gap_fill", [False, True])
def test_runner_cleans_up_with_optional_gap_fill(
    batch_skill, monkeypatch: pytest.MonkeyPatch, apply_gap_fill
) -> None:
    skill, _ = batch_skill
    cleanup_dir = skill.parent / "graph-temp"
    cleanup_dir.mkdir()
    pool = object()
    calls = []
    _mock_scan(
        monkeypatch,
        lambda context: context.update(temp_dir_for_cleanup=str(cleanup_dir)),
    )

    def fail_gap_fill(file_cache, language, **kwargs):
        calls.append(kwargs["api_pool"])
        raise ValueError("gap-fill failed")

    monkeypatch.setattr(runner, "run_gap_fill", fail_gap_fill)
    options = {"apply_gap_fill": True} if apply_gap_fill else {}
    entry, error = runner.run_one(
        skill,
        skill.parent,
        use_llm=True,
        detected_language="zh",
        api_pool=pool,
        **options,
    )

    assert not cleanup_dir.exists()
    assert calls == ([pool] if apply_gap_fill else [])
    assert error == ("gap-fill failed" if apply_gap_fill else None)
    if apply_gap_fill:
        assert entry["risk_assessment"]["severity"] == "ERROR"


@pytest.mark.parametrize("rich_available", [True, False])
@pytest.mark.parametrize("skill_name", ["safe-skill", "x\\"])
def test_cli_warns_using_detected_language(
    batch_skill,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rich_available: bool,
    skill_name: str,
) -> None:
    skill, _ = batch_skill
    if skill.name != skill_name:
        skill = skill.rename(skill.with_name(skill_name))
    if not rich_available:
        monkeypatch.setitem(sys.modules, "rich.console", None)
    observed = _mock_scan(monkeypatch)
    # Keep this language-formatting test's in-process graph double.
    monkeypatch.setattr(batch_scan, "_scan_skill_bounded", batch_scan._scan_skill)
    monkeypatch.setattr(batch_scan, "create_api_key_pool_from_env", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["batch_scan", str(skill.parent), "--no-llm", "--workers", "1", "-f", "json"],
    )

    batch_scan._main_impl()

    output = capsys.readouterr()
    output_text = " ".join((output.out + output.err).split())
    assert "WARNING:" in output_text
    assert f"skill '{skill_name}' (zh) scanned with --no-llm." in output_text
    assert observed["calls"] == []


@pytest.mark.parametrize("rich_available", [True, False])
@pytest.mark.parametrize("has_error", [True, False])
def test_cli_prints_literal_skill_names_and_errors(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rich_available,
    has_error,
) -> None:
    import builtins

    if not rich_available:
        original_import = builtins.__import__

        def without_rich(name, *args, **kwargs):
            if name == "rich.console":
                raise ImportError("Rich unavailable")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", without_rich)
    name = "[/bad]\n\x1b[31mchild:smile:"
    entry = {
        "skill": {"name": name, "language": "en"},
        "risk_assessment": {"score": 90, "severity": "CRITICAL"},
        "issues": [],
    }
    error = "[/bad]\n\x1b[31mfailed" if has_error else None
    if error:
        entry["error"] = error
    monkeypatch.setattr(batch_scan, "discover_skills", lambda root: [tmp_path / name])
    monkeypatch.setattr(batch_scan, "run_one", lambda *args, **kwargs: (entry, error))
    monkeypatch.setattr(batch_scan, "_scan_skill_bounded", batch_scan._scan_skill)
    monkeypatch.setattr(batch_scan, "create_api_key_pool_from_env", lambda: None)
    monkeypatch.setattr(sys, "argv", ["batch_scan", str(tmp_path), "--no-llm", "--workers", "1"])

    with pytest.raises(SystemExit) as exited:
        batch_scan._main_impl()

    assert exited.value.code == (2 if has_error else 1)
    rendered = capsys.readouterr().out
    assert "[/bad] [31mchild:smile:" in rendered
    assert "\x1b" not in rendered
    if has_error:
        assert "[/bad] [31mfailed" in rendered
    else:
        assert "90/100 CRITICAL" in rendered


def _stalled_scan_process(skill_dir, root, result_path, options, started):
    # Spawn imports a fresh module, so use the real worker setup around the stall.
    batch_scan._scan_skill = _stalled_scan
    batch_scan._scan_skill_process(skill_dir, root, result_path, options, started)


def _stalled_scan(skill_dir, root, **options):
    pool = options.get("api_pool")
    if pool is not None:
        pool.acquire()
    temporary = tempfile.mkdtemp(prefix="scan-owned-")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    (skill_dir / "started.json").write_text(json.dumps([os.getpid(), child.pid, temporary]))
    time.sleep(120)


def _successful_scan_process(skill_dir, root, result_path, options, started):
    if os.name == "posix":
        os.setsid()
    started.set()
    result_path.write_text(
        json.dumps([{"skill": {"name": skill_dir.name, "language": "en"}}, None, skill_dir.name])
    )


def test_worker_timeout_kills_scan_and_releases_pool_capacity(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILLSPECTOR_API_KEYS", "synthetic-a;synthetic-b")
    monkeypatch.setattr(batch_scan, "_scan_skill_process", _stalled_scan_process)
    with batch_scan._PoolManager(ctx=multiprocessing.get_context("spawn")) as manager:
        pool = manager.create_pool(1)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="scan timed out"):
            batch_scan._scan_skill_bounded(
                tmp_path, tmp_path, api_pool=pool, timeout=5, startup_timeout=20
            )
        assert time.monotonic() - started < 35
        assert pool.snapshot()["active_requests"] == 0
        assert pool.snapshot()["total_requests_served"] == 1
        worker_pid, child_pid, temporary = json.loads((tmp_path / "started.json").read_text())
        assert not Path(temporary).exists()
        assert worker_pid not in {child.pid for child in multiprocessing.active_children()}
        if os.name == "posix":
            _assert_process_stopped(child_pid)
        monkeypatch.setattr(batch_scan, "_scan_skill_process", _successful_scan_process)
        group_signals = []
        if os.name == "posix":
            monkeypatch.setattr(batch_scan.os, "killpg", lambda *args: group_signals.append(args))
        entry, error, name = batch_scan._scan_skill_bounded(
            tmp_path, tmp_path, api_pool=pool, timeout=10
        )
        assert entry["skill"]["name"] == name == tmp_path.name
        assert error is None
        assert group_signals == []  # Never signal a PID after its worker was reaped.


def test_process_pool_shares_slots_and_cancels_waiting_owner(monkeypatch):
    monkeypatch.setenv("SKILLSPECTOR_API_KEYS", "synthetic-a;synthetic-b")
    with batch_scan._PoolManager(ctx=multiprocessing.get_context("spawn")) as manager:
        pool = manager.create_pool(1)
        first = pool.acquire(owner="running")
        pool.acquire(owner="running")
        assert pool.try_acquire(owner="another") is None
        pool.release(first, owner="running")
        assert pool.snapshot()["active_requests"] == 1
        pool.acquire(owner="running")
        with ThreadPoolExecutor(max_workers=1) as executor:
            waiting = executor.submit(pool.acquire, owner="waiting")
            pool.release_owner("waiting")
            with pytest.raises(RuntimeError, match="stopped"):
                waiting.result(timeout=5)
        pool.release_owner("running")
        assert pool.snapshot()["active_requests"] == 0
        with pytest.raises(RuntimeError, match="stopped"):
            pool.try_acquire(owner="running")


def _assert_process_stopped(pid):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        if sys.platform == "linux":
            # Container PID 1 may leave a killed orphan awaiting reaping.
            status = Path(f"/proc/{pid}/stat")
            if not status.exists() or status.read_text().split()[2] == "Z":
                return
        time.sleep(0.05)
    pytest.fail(f"process {pid} remained alive")


def _slow_start_process(skill_dir, root, result_path, options, started):
    time.sleep(1.5)
    _successful_scan_process(skill_dir, root, result_path, options, started)


def _never_started_process(skill_dir, root, result_path, options, started):
    time.sleep(120)


def test_worker_startup_has_a_separate_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_scan, "_scan_skill_process", _slow_start_process)
    entry, error, name = batch_scan._scan_skill_bounded(
        tmp_path, tmp_path, timeout=1, startup_timeout=20
    )
    assert entry["skill"]["name"] == name == tmp_path.name
    assert error is None
    monkeypatch.setattr(batch_scan, "_scan_skill_process", _never_started_process)
    with pytest.raises(TimeoutError, match="startup timed out"):
        batch_scan._scan_skill_bounded(tmp_path, tmp_path, timeout=10, startup_timeout=0.2)


def _mixed_scan_process(skill_dir, root, result_path, options, started):
    if skill_dir.name == "stalled":
        batch_scan._scan_skill = _stalled_scan
    elif skill_dir.name == "crashed":
        os._exit(7)
    else:
        return _successful_scan_process(skill_dir, root, result_path, options, started)
    batch_scan._scan_skill_process(skill_dir, root, result_path, options, started)


@pytest.mark.parametrize("failed_name", ["stalled", "crashed"])
def test_cli_reports_failed_workers_with_completed_skills(tmp_path, monkeypatch, failed_name):
    failed = tmp_path / failed_name
    normal = tmp_path / "normal"
    for skill in (failed, normal):
        skill.mkdir()
        (skill / "SKILL.md").write_text(f"---\nname: {skill.name}\n---\nA helpful skill.\n")
    output = tmp_path / "report.json"
    monkeypatch.setattr(batch_scan, "_scan_skill_process", _mixed_scan_process)
    monkeypatch.setattr(
        batch_scan,
        "_scan_skill_bounded",
        partial(batch_scan._scan_skill_bounded, timeout=3, startup_timeout=20),
    )
    monkeypatch.setattr(batch_scan, "create_api_key_pool_from_env", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_scan",
            str(tmp_path),
            "--no-llm",
            "--workers",
            "2",
            "-f",
            "json",
            "-o",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exited:
        batch_scan._main_impl()
    assert exited.value.code == 2
    report = json.loads(output.read_text())
    assert report["batch"]["total_skills"] == 2
    entries = {entry["skill"]["name"]: entry for entry in report["skills"]}
    assert set(entries) == {failed_name, "normal"}
    assert entries[failed_name]["risk_assessment"]["severity"] == "ERROR"
    expected = (
        "scan timed out after 3s" if failed_name == "stalled" else "worker exited with code 7"
    )
    assert expected in entries[failed_name]["error"]
    assert entries[failed_name]["execution_successful"] is False
    assert entries[failed_name]["skill"]["language"] == "unknown"
    assert report["batch"]["enhancements"]["languages_detected"] == {}
    assert not entries["normal"].get("error")


def _abandoned_supervisor(skill_dir):
    batch_scan._scan_skill_process = _stalled_scan_process
    batch_scan._scan_skill_bounded(skill_dir, skill_dir, timeout=120)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup")
def test_worker_stops_descendants_when_supervisor_dies(tmp_path):
    supervisor = multiprocessing.get_context("spawn").Process(
        target=_abandoned_supervisor, args=(tmp_path,)
    )
    supervisor.start()
    try:
        marker = tmp_path / "started.json"
        # Two spawned interpreters may each need the production startup budget.
        deadline = time.monotonic() + 180
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists(), "worker did not start"
        worker_pid, child_pid, temporary = json.loads(marker.read_text())
        supervisor.kill()
        supervisor.join(5)
        assert not supervisor.is_alive()
        _assert_process_stopped(worker_pid)
        _assert_process_stopped(child_pid)
        assert not Path(temporary).parent.exists()
    finally:
        if supervisor.is_alive():
            supervisor.kill()
        supervisor.join(5)
        supervisor.close()


def _abandoned_pool_supervisor(skill_dir):
    manager = batch_scan._PoolManager(ctx=multiprocessing.get_context("spawn"))
    manager.start(batch_scan._start_parent_watch)
    pool = manager.create_pool()
    assert pool.snapshot()["keys_configured"] == 2
    (skill_dir / "manager.json").write_text(json.dumps(manager._process.pid))
    time.sleep(120)


@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor kill")
def test_api_pool_manager_stops_when_supervisor_dies(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILLSPECTOR_API_KEYS", "synthetic-a;synthetic-b")
    supervisor = multiprocessing.get_context("spawn").Process(
        target=_abandoned_pool_supervisor, args=(tmp_path,)
    )
    supervisor.start()
    manager_pid = None
    try:
        marker = tmp_path / "manager.json"
        deadline = time.monotonic() + 180
        while not marker.exists() and time.monotonic() < deadline and supervisor.is_alive():
            time.sleep(0.05)
        assert marker.exists(), "API pool manager did not start"
        manager_pid = json.loads(marker.read_text())
        supervisor.kill()
        supervisor.join(5)
        assert not supervisor.is_alive()
        _assert_process_stopped(manager_pid)
        manager_pid = None
    finally:
        if supervisor.is_alive():
            supervisor.kill()
        supervisor.join(5)
        supervisor.close()
        if manager_pid is not None:
            try:
                os.kill(manager_pid, 9)
            except ProcessLookupError:
                pass


def test_scan_forwards_verbose_logging(batch_skill, monkeypatch):
    skill, _ = batch_skill
    _mock_scan(monkeypatch)
    levels = []
    monkeypatch.setattr(batch_scan, "set_level", levels.append)
    batch_scan._scan_skill(
        skill, skill.parent, use_llm=False, lang="en", require_llm=False, verbose=True
    )
    assert levels == ["DEBUG"]
