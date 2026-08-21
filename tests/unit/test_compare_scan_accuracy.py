# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import io
import json
import subprocess
import sys
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

import pytest

from scripts import compare_scan_accuracy

ZERO_TOLERANCE_POLICY = {
    "max_candidate_false_positives": 0,
    "max_candidate_false_negatives": 0,
    "max_false_positive_increase": 0,
    "max_false_negative_increase": 0,
    "max_per_rule_false_positive_increase": 0,
    "max_per_rule_false_negative_increase": 0,
    "max_per_cohort_false_positive_increase": 0,
    "max_per_cohort_false_negative_increase": 0,
    "max_per_case_false_positive_increase": 0,
    "max_per_case_false_negative_increase": 0,
}
BASELINE_REVISION = "a" * 40
CANDIDATE_REVISION = "b" * 40


def _complete_report(issues: list[dict[str, object]]) -> dict[str, object]:
    return {
        "issues": issues,
        "execution_successful": True,
        "analysis_completeness": {
            "total_components": 1,
            "scanned_components": 1,
            "coverage_percent": 100.0,
            "is_complete": True,
            "status": "complete",
            "execution_successful": True,
            "fully_inspected_files": 1,
            "partially_inspected_files": 0,
            "entirely_uninspected_files": 0,
            "ledger_exceptions": [],
            "scope_exclusions": [],
            "analyzer_statuses": [],
            "limitations": [],
        },
    }


def _write_case(root: Path, name: str) -> None:
    target = root / name
    target.mkdir()
    (target / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")


def _cases() -> list[dict[str, object]]:
    return [
        {
            "id": "benign",
            "path": "benign",
            "classification": "maintained_benign",
            "expected_rules": {},
        },
        {
            "id": "real-world",
            "path": "real-world",
            "classification": "approved_real_world",
            "expected_rules": {"R1": 1},
        },
    ]


def _write_manifest(
    path: Path,
    *,
    cases: list[dict[str, object]] | None = None,
    policy: dict[str, int] | None = None,
    schema_version: int = 2,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "material_regression_policy": policy or ZERO_TOLERANCE_POLICY,
                "cases": _cases() if cases is None else cases,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _mock_scanner_identities(monkeypatch) -> None:
    def fake_identity(*, executable: Path, worktree: Path, revision: str) -> dict[str, object]:
        environment = {"HOME": compare_scan_accuracy._ISOLATED_HOME_MARKER}
        return {
            "declared_revision": revision,
            "resolved_revision": revision,
            "revision": revision,
            "worktree": str(worktree),
            "executable": str(executable),
            "executable_relative_path": f".venv/bin/{executable.name}",
            "executable_sha256": f"sha256:{executable.name}",
            "python_executable": str(worktree / ".venv/bin/python3"),
            "python_executable_resolved": str(worktree / ".venv/bin/python3"),
            "python_executable_sha256": "sha256:python",
            "runtime_identity": {"environment": environment},
            "runtime_identity_sha256": f"sha256:runtime-{revision}",
            "dependency_identity_sha256": f"sha256:dependencies-{revision}",
            "environment_sha256": f"sha256:environment-{revision}",
            "pyproject_sha256": f"sha256:pyproject-{revision}",
            "lockfile": "uv.lock",
            "lockfile_sha256": f"sha256:lock-{revision}",
            "source_root": str(worktree / "src"),
            "source_tree_git_oid": f"tree-{revision}",
            "source_tree_revision": revision,
            "source_binding": "isolated-worktree-source-import",
            "source_runner": str(Path(compare_scan_accuracy.__file__).resolve()),
            "source_runner_sha256": "sha256:runner",
            "worktree_clean": True,
            "tracked_worktree_clean": True,
        }

    monkeypatch.setattr(compare_scan_accuracy, "_resolve_scanner_identity", fake_identity)
    monkeypatch.setattr(
        compare_scan_accuracy,
        "_source_bound_command",
        lambda *, executable, target, worktree, source_root=None, runner=None: [
            str(worktree / ".venv/bin/python3"),
            "-I",
            "-B",
            str(runner or Path(compare_scan_accuracy.__file__).resolve()),
            "--_source-bound-scan",
            str(source_root or worktree / "src"),
            str(target),
        ],
    )

    @contextmanager
    def fake_snapshots(**arguments: object) -> Iterator[dict[str, Path]]:
        baseline = arguments["baseline_identity"]
        candidate = arguments["candidate_identity"]
        assert isinstance(baseline, dict)
        assert isinstance(candidate, dict)
        yield {
            "corpus_root": arguments["corpus_root"],
            "baseline_source": Path(str(baseline["source_root"])),
            "candidate_source": Path(str(candidate["source_root"])),
            "runner": Path(compare_scan_accuracy.__file__).resolve(),
        }

    monkeypatch.setattr(compare_scan_accuracy, "_accuracy_snapshots", fake_snapshots)


def _compare(
    tmp_path: Path,
    manifest: Path,
    corpus: Path,
    **overrides: object,
) -> dict[str, object]:
    arguments: dict[str, object] = {
        "manifest_path": manifest,
        "corpus_root": corpus,
        "baseline_executable": tmp_path / "baseline",
        "candidate_executable": tmp_path / "candidate",
        "baseline_worktree": tmp_path / "baseline-worktree",
        "candidate_worktree": tmp_path / "candidate-worktree",
        "baseline_revision": BASELINE_REVISION,
        "candidate_revision": CANDIDATE_REVISION,
        "invocation": ["compare_scan_accuracy.py", "--manifest", str(manifest)],
    }
    arguments.update(overrides)
    return compare_scan_accuracy.compare_scanners(**arguments)  # type: ignore[arg-type]


def _write_bound_approval(
    path: Path,
    result: dict[str, object],
    reviewer: str = "Security Reviewer",
) -> None:
    baseline = result["baseline"]
    candidate = result["candidate"]
    material = result["material_regression"]
    assert isinstance(baseline, dict)
    assert isinstance(candidate, dict)
    assert isinstance(material, dict)
    policy = material["policy"]
    violations = material["violations"]
    document = {
        "schema_version": 1,
        "reviewer": reviewer,
        "rationale": "Reviewed and explicitly approved for this exact evidence set.",
        "corpus_identity": result["corpus_identity"],
        "manifest_sha256": result["manifest_sha256"],
        "baseline_identity": {
            field: baseline[field] for field in compare_scan_accuracy.APPROVAL_IDENTITY_FIELDS
        },
        "candidate_identity": {
            field: candidate[field] for field in compare_scan_accuracy.APPROVAL_IDENTITY_FIELDS
        },
        "policy_sha256": compare_scan_accuracy._json_sha256(policy),
        "violations": violations,
        "violations_sha256": compare_scan_accuracy._json_sha256(violations),
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_compare_scanners_reports_bound_identity_and_explicit_adjudication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_case(corpus, "benign")
    _write_case(corpus, "real-world")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)

    def fake_scan(
        executable: Path,
        target: Path,
        worktree: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        del kwargs
        assert worktree.name in {"baseline-worktree", "candidate-worktree"}
        if target.name == "benign" or executable.name == "baseline":
            return _complete_report([])
        return _complete_report([{"id": "R1"}])

    _mock_scanner_identities(monkeypatch)
    monkeypatch.setattr(compare_scan_accuracy, "_run_scan", fake_scan)
    result = _compare(tmp_path, manifest, corpus)

    assert result["passed"] is True
    assert result["schema_version"] == 2
    assert result["corpus_identity"] == result["corpus_snapshot"]
    assert str(result["corpus_identity"]).startswith("sha256:")
    assert str(result["manifest_sha256"]).startswith("sha256:")
    assert result["baseline"]["revision"] == BASELINE_REVISION
    assert result["candidate"]["revision"] == CANDIDATE_REVISION
    assert result["execution"]["invocation"] == [
        "compare_scan_accuracy.py",
        "--manifest",
        str(manifest),
    ]
    assert result["execution"]["inputs_verified_unchanged"] is True
    assert result["execution"]["configuration"] == {
        "manifest": str(manifest.resolve()),
        "corpus_root": str(corpus.resolve()),
        "baseline_executable": str(tmp_path / "baseline"),
        "candidate_executable": str(tmp_path / "candidate"),
        "baseline_worktree": str(tmp_path / "baseline-worktree"),
        "candidate_worktree": str(tmp_path / "candidate-worktree"),
        "baseline_revision": BASELINE_REVISION,
        "candidate_revision": CANDIDATE_REVISION,
        "selected_rules": [],
        "scan_arguments": ["scan", "<case>", "--format", "json", "--no-llm"],
        "source_binding": "private-git-object-and-corpus-snapshot",
        "source_runner": str(Path(compare_scan_accuracy.__file__).resolve()),
        "source_runner_sha256": "sha256:runner",
        "baseline_environment": {"HOME": compare_scan_accuracy._ISOLATED_HOME_MARKER},
        "baseline_environment_sha256": f"sha256:environment-{BASELINE_REVISION}",
        "candidate_environment": {"HOME": compare_scan_accuracy._ISOLATED_HOME_MARKER},
        "candidate_environment_sha256": f"sha256:environment-{CANDIDATE_REVISION}",
        "approval_artifact": None,
        "approval_reviewer": None,
    }
    assert result["observed_classifications"] == [
        "approved_real_world",
        "maintained_benign",
    ]
    real_world = next(case for case in result["cases"] if case["id"] == "real-world")
    assert real_world["scan_execution"]["candidate"] == {
        "command": [
            str(tmp_path / "candidate-worktree/.venv/bin/python3"),
            "-I",
            "-B",
            "<private-source-runner-snapshot>",
            "--_source-bound-scan",
            "<private-candidate-source-snapshot>",
            "<private-corpus-snapshot>/real-world",
        ],
        "working_directory": str(tmp_path / "candidate-worktree"),
        "source_root": "<private-candidate-source-snapshot>",
    }
    assert result["per_rule"]["R1"] == {
        "baseline": 0,
        "candidate": 1,
        "delta": 1,
        "baseline_false_positives": 0,
        "candidate_false_positives": 0,
        "false_positive_delta": 0,
        "baseline_false_negatives": 1,
        "candidate_false_negatives": 0,
        "false_negative_delta": -1,
    }
    assert result["adjudication"]["delta"] == {
        "false_positives": 0,
        "false_negatives": -1,
        "by_rule": {"R1": {"false_positives": 0, "false_negatives": -1}},
    }
    assert result["material_regression"] == {
        "policy": ZERO_TOLERANCE_POLICY,
        "violations": [],
        "approval": None,
        "approved": False,
    }


def test_corpus_identity_changes_when_only_adjudication_manifest_changes(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_case(corpus, "benign")
    _write_case(corpus, "real-world")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    document, original_bytes = compare_scan_accuracy._load_manifest(manifest)
    original = compare_scan_accuracy._corpus_identity(
        corpus,
        document["cases"],
        original_bytes,
    )

    cases = _cases()
    cases[1]["expected_rules"] = {"R1": {"min": 1, "max": 2}}
    _write_manifest(manifest, cases=cases)
    document, changed_bytes = compare_scan_accuracy._load_manifest(manifest)
    changed = compare_scan_accuracy._corpus_identity(
        corpus,
        document["cases"],
        changed_bytes,
    )

    assert original != changed


def test_compare_scanners_rejects_corpus_changes_during_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_case(corpus, "benign")
    _write_case(corpus, "real-world")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    scan_count = 0

    def mutating_scan(
        executable: Path,
        target: Path,
        worktree: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        nonlocal scan_count
        del executable, worktree, kwargs
        scan_count += 1
        if scan_count == 4:
            (target / "SKILL.md").write_text("# changed\n", encoding="utf-8")
        return _complete_report([])

    _mock_scanner_identities(monkeypatch)
    monkeypatch.setattr(compare_scan_accuracy, "_run_scan", mutating_scan)

    with pytest.raises(ValueError, match="corpus changed during comparison"):
        _compare(tmp_path, manifest, corpus)


def test_compare_scanners_requires_both_review_classifications(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_case(corpus, "benign")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, cases=[_cases()[0]])
    _mock_scanner_identities(monkeypatch)

    with pytest.raises(ValueError, match="approved_real_world"):
        _compare(tmp_path, manifest, corpus)


def test_material_false_positive_regression_requires_named_approval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_case(corpus, "benign")
    _write_case(corpus, "real-world")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)

    def fake_scan(
        executable: Path,
        target: Path,
        worktree: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        del worktree, kwargs
        if executable.name == "baseline":
            return _complete_report([] if target.name == "benign" else [{"id": "R1"}])
        issues = [{"id": "R2"}] if target.name == "benign" else [{"id": "R1"}]
        return _complete_report(issues)

    _mock_scanner_identities(monkeypatch)
    monkeypatch.setattr(compare_scan_accuracy, "_run_scan", fake_scan)
    rejected = _compare(tmp_path, manifest, corpus)

    assert rejected["passed"] is False
    assert rejected["adjudication"]["candidate"]["false_positives"] == 1
    assert rejected["adjudication"]["delta"]["false_positives"] == 1
    assert rejected["per_rule"]["R2"]["false_positive_delta"] == 1
    assert rejected["material_regression"]["approval"] is None
    assert {violation["metric"] for violation in rejected["material_regression"]["violations"]} >= {
        "candidate_false_positives",
        "false_positive_increase",
        "per_rule_false_positive_increase",
    }

    approval = tmp_path / "SECURITY-APPROVAL.txt"
    _write_bound_approval(approval, rejected)
    approved = _compare(
        tmp_path,
        manifest,
        corpus,
        approval_artifact=approval,
        approval_reviewer="Security Reviewer",
    )

    assert approved["passed"] is False
    assert approved["material_regression"]["approved"] is False
    approval_metadata = approved["material_regression"]["approval"]
    assert approval_metadata["reviewer"] == "Security Reviewer"
    assert approval_metadata["artifact"] == str(approval.resolve())
    assert approval_metadata["artifact_sha256"] == compare_scan_accuracy._file_sha256(approval)
    assert str(approval_metadata["binding_sha256"]).startswith("sha256:")
    assert approval_metadata["authorization"] == "evidence-only-untrusted-local-artifact"


def test_false_negative_regression_is_explicit_and_fails_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_case(corpus, "benign")
    _write_case(corpus, "real-world")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)

    def fake_scan(
        executable: Path,
        target: Path,
        worktree: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        del worktree, kwargs
        if target.name == "benign" or executable.name == "candidate":
            return _complete_report([])
        return _complete_report([{"id": "R1"}])

    _mock_scanner_identities(monkeypatch)
    monkeypatch.setattr(compare_scan_accuracy, "_run_scan", fake_scan)
    result = _compare(tmp_path, manifest, corpus)

    assert result["passed"] is False
    assert result["adjudication"]["candidate"]["false_negatives"] == 1
    assert result["adjudication"]["delta"]["false_negatives"] == 1
    assert result["per_rule"]["R1"]["false_negative_delta"] == 1
    assert any(
        violation["metric"] == "false_negative_increase"
        for violation in result["material_regression"]["violations"]
    )


def test_scanner_identity_binds_revision_clean_worktree_and_executable_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    worktree = tmp_path / "scanner"
    executable = worktree / ".venv" / "bin" / "skillspector"
    executable.parent.mkdir(parents=True)
    interpreter = executable.parent / "python3"
    interpreter.write_text("python runtime\n", encoding="utf-8")
    interpreter.chmod(0o755)
    executable.write_bytes(
        f"#!{interpreter}\n".encode() + compare_scan_accuracy._CONSOLE_ENTRYPOINT_BODY
    )
    executable.chmod(0o755)
    package = worktree / "src" / "skillspector"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text("", encoding="utf-8")
    (worktree / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (worktree / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    actual_revision = "c" * 40

    def fake_git(root: Path, *args: str) -> str:
        assert root == worktree.resolve()
        if args == ("rev-parse", "--show-toplevel"):
            return str(worktree.resolve())
        if args == ("rev-parse", "HEAD"):
            return actual_revision
        if args == ("rev-parse", f"{actual_revision}:src/skillspector"):
            return "e" * 40
        if args == (
            "ls-tree",
            "-r",
            "--name-only",
            actual_revision,
            "--",
            "src",
        ):
            return "src/skillspector/__init__.py\nsrc/skillspector/cli.py"
        if args == ("status", "--porcelain", "--untracked-files=all"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(compare_scan_accuracy, "_git_output", fake_git)
    monkeypatch.setattr(
        compare_scan_accuracy,
        "_runtime_identity",
        lambda interpreter, root: {
            "environment": {"HOME": str(root)},
            "environment_sha256": "sha256:environment",
            "dependency_identity_sha256": "sha256:dependencies",
            "runtime_identity_sha256": "sha256:runtime",
        },
    )
    identity = compare_scan_accuracy._resolve_scanner_identity(
        executable=executable,
        worktree=worktree,
        revision=actual_revision,
    )

    assert identity["revision"] == actual_revision
    assert identity["declared_revision"] == actual_revision
    assert identity["resolved_revision"] == actual_revision
    assert identity["executable_relative_path"] == ".venv/bin/skillspector"
    assert identity["executable_sha256"] == compare_scan_accuracy._file_sha256(executable)
    assert identity["python_executable"] == str(interpreter)
    assert identity["python_executable_resolved"] == str(interpreter.resolve())
    assert identity["source_root"] == str((worktree / "src").resolve())
    assert identity["source_tree_git_oid"] == "e" * 40
    assert identity["source_tree_revision"] == actual_revision
    assert identity["source_binding"] == "isolated-worktree-source-import"
    assert identity["tracked_worktree_clean"] is True

    with pytest.raises(ValueError, match="revision mismatch"):
        compare_scan_accuracy._resolve_scanner_identity(
            executable=executable,
            worktree=worktree,
            revision="d" * 40,
        )

    ignored_import = package / "ignored_import.py"
    ignored_import.write_text("MUTATED = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source inventory differs"):
        compare_scan_accuracy._resolve_scanner_identity(
            executable=executable,
            worktree=worktree,
            revision=actual_revision,
        )
    ignored_import.unlink()

    def dirty_git(root: Path, *args: str) -> str:
        if args == ("status", "--porcelain", "--untracked-files=all"):
            return " M src/scanner.py"
        return fake_git(root, *args)

    monkeypatch.setattr(compare_scan_accuracy, "_git_output", dirty_git)
    with pytest.raises(ValueError, match="worktree has changes"):
        compare_scan_accuracy._resolve_scanner_identity(
            executable=executable,
            worktree=worktree,
            revision=actual_revision,
        )


def test_scanner_identity_rejects_ignored_modified_entrypoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    worktree = tmp_path / "scanner"
    executable = worktree / ".venv" / "bin" / "skillspector"
    executable.parent.mkdir(parents=True)
    interpreter = executable.parent / "python3"
    interpreter.write_text("python runtime\n", encoding="utf-8")
    interpreter.chmod(0o755)
    executable.write_bytes(
        f"#!{interpreter}\n".encode()
        + compare_scan_accuracy._CONSOLE_ENTRYPOINT_BODY
        + b"# ignored mutation\n"
    )
    executable.chmod(0o755)
    package = worktree / "src" / "skillspector"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text("", encoding="utf-8")
    revision = "c" * 40

    monkeypatch.setattr(
        compare_scan_accuracy,
        "_git_output",
        lambda root, *args: (
            str(worktree.resolve())
            if args == ("rev-parse", "--show-toplevel")
            else revision
            if args == ("rev-parse", "HEAD")
            else "e" * 40
            if args == ("rev-parse", f"{revision}:src/skillspector")
            else "src/skillspector/__init__.py\nsrc/skillspector/cli.py"
            if args
            == (
                "ls-tree",
                "-r",
                "--name-only",
                revision,
                "--",
                "src",
            )
            else ""
        ),
    )

    with pytest.raises(ValueError, match="immutable SkillSpector entrypoint"):
        compare_scan_accuracy._resolve_scanner_identity(
            executable=executable,
            worktree=worktree,
            revision=revision,
        )


def test_source_bound_runner_ignores_preloaded_installed_package(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    source_root = tmp_path / "revision-source"
    installed_root = tmp_path / "ignored-site-packages"
    for root, marker in ((source_root, "SOURCE"), (installed_root, "IGNORED")):
        package = root / "skillspector"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "cli.py").write_text(
            f'def app():\n    print(\'{{"issues": [{{"id": "{marker}"}}]}}\')\n    return 0\n',
            encoding="utf-8",
        )
    target = tmp_path / "case"
    target.mkdir()
    original_sys_path = list(sys.path)
    original_modules = {
        module_name: module
        for module_name, module in sys.modules.items()
        if module_name == "skillspector" or module_name.startswith("skillspector.")
    }
    for module_name in original_modules:
        del sys.modules[module_name]
    monkeypatch.syspath_prepend(str(installed_root))
    importlib.import_module("skillspector.cli")
    assert Path(sys.modules["skillspector"].__file__).is_relative_to(installed_root)

    try:
        result = compare_scan_accuracy._run_source_bound_scan([str(source_root), str(target)])
        rendered = capsys.readouterr().out
    finally:
        sys.path[:] = original_sys_path
        for module_name in tuple(sys.modules):
            if module_name == "skillspector" or module_name.startswith("skillspector."):
                del sys.modules[module_name]
        sys.modules.update(original_modules)

    assert result == 0
    assert json.loads(rendered)["issues"] == [{"id": "SOURCE"}]


def test_compare_scanners_rejects_scanner_identity_change_during_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_case(corpus, "benign")
    _write_case(corpus, "real-world")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    candidate_resolutions = 0
    _mock_scanner_identities(monkeypatch)

    def changing_identity(*, executable: Path, worktree: Path, revision: str) -> dict[str, object]:
        nonlocal candidate_resolutions
        if revision == CANDIDATE_REVISION:
            candidate_resolutions += 1
        suffix = "-changed" if revision == CANDIDATE_REVISION and candidate_resolutions == 2 else ""
        environment = {"HOME": str(worktree / ".git/accuracy-gate-empty-home")}
        return {
            "declared_revision": revision,
            "resolved_revision": revision,
            "revision": revision,
            "worktree": str(worktree),
            "executable": str(executable),
            "executable_relative_path": f".venv/bin/{executable.name}",
            "executable_sha256": f"sha256:{executable.name}{suffix}",
            "python_executable": str(worktree / ".venv/bin/python3"),
            "python_executable_resolved": str(worktree / ".venv/bin/python3"),
            "python_executable_sha256": "sha256:python",
            "runtime_identity": {"environment": environment},
            "runtime_identity_sha256": f"sha256:runtime-{revision}",
            "dependency_identity_sha256": f"sha256:dependencies-{revision}",
            "environment_sha256": f"sha256:environment-{revision}",
            "pyproject_sha256": f"sha256:pyproject-{revision}",
            "lockfile": "uv.lock",
            "lockfile_sha256": f"sha256:lock-{revision}",
            "source_root": str(worktree / "src"),
            "source_tree_git_oid": f"tree-{revision}",
            "source_tree_revision": revision,
            "source_binding": "isolated-worktree-source-import",
            "source_runner": str(Path(compare_scan_accuracy.__file__).resolve()),
            "source_runner_sha256": "sha256:runner",
            "worktree_clean": True,
            "tracked_worktree_clean": True,
        }

    monkeypatch.setattr(compare_scan_accuracy, "_resolve_scanner_identity", changing_identity)
    monkeypatch.setattr(
        compare_scan_accuracy,
        "_run_scan",
        lambda executable, target, worktree, **kwargs: _complete_report([]),
    )

    with pytest.raises(ValueError, match="Candidate scanner identity changed"):
        _compare(tmp_path, manifest, corpus)


def test_manifest_rejects_unknown_policy_fields_and_implicit_adjudication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_case(corpus, "benign")
    _write_case(corpus, "real-world")
    manifest = tmp_path / "manifest.json"
    policy = {**ZERO_TOLERANCE_POLICY, "max_false_positive_increse": 0}
    _write_manifest(manifest, policy=policy)
    _mock_scanner_identities(monkeypatch)

    with pytest.raises(ValueError, match="Unknown material_regression_policy"):
        _compare(tmp_path, manifest, corpus)

    cases = _cases()
    cases[0].pop("expected_rules")
    _write_manifest(manifest, cases=cases)
    with pytest.raises(ValueError, match="explicit expected_rules"):
        _compare(tmp_path, manifest, corpus)


def test_approval_requires_both_artifact_and_named_reviewer(tmp_path: Path) -> None:
    approval = tmp_path / "approval.txt"
    approval.write_text("approved\n", encoding="utf-8")
    expected = {
        "corpus_identity": "sha256:corpus",
        "manifest_sha256": "sha256:manifest",
        "baseline_identity": {},
        "candidate_identity": {},
        "policy": ZERO_TOLERANCE_POLICY,
        "violations": [],
    }

    with pytest.raises(ValueError, match="both artifact and reviewer"):
        compare_scan_accuracy._approval_metadata(approval, None, **expected)
    with pytest.raises(ValueError, match="both artifact and reviewer"):
        compare_scan_accuracy._approval_metadata(None, "Reviewer", **expected)


def test_compare_scanners_rejects_case_outside_corpus(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _write_case(corpus, "real-world")
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        cases=[
            {
                "id": "outside",
                "path": "../outside",
                "classification": "maintained_benign",
                "expected_rules": {},
            },
            _cases()[1],
        ],
    )
    _mock_scanner_identities(monkeypatch)

    with pytest.raises(ValueError, match="below the corpus root"):
        _compare(tmp_path, manifest, corpus)


def test_schema_v1_manifest_is_rejected_as_insufficient_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, schema_version=1)
    _mock_scanner_identities(monkeypatch)

    with pytest.raises(ValueError, match="schema_version must be 2"):
        _compare(tmp_path, manifest, corpus)


def test_partial_rule_selection_is_rejected_before_scanning(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Partial rule selection"):
        compare_scan_accuracy.compare_scanners(
            manifest_path=tmp_path / "missing.json",
            corpus_root=tmp_path,
            baseline_executable=tmp_path / "baseline",
            candidate_executable=tmp_path / "candidate",
            baseline_worktree=tmp_path,
            candidate_worktree=tmp_path,
            baseline_revision=BASELINE_REVISION,
            candidate_revision=CANDIDATE_REVISION,
            invocation=["compare_scan_accuracy.py"],
            selected_rules=frozenset({"R1"}),
        )


def test_rule_counts_use_occurrences_and_reject_malformed_occurrences() -> None:
    report = _complete_report(
        [
            {"id": "R1", "occurrences": [{"file": "a"}, {"file": "b"}, {"file": "c"}]},
            {"id": "R2"},
        ]
    )
    assert compare_scan_accuracy._rule_counts(report, frozenset()) == {"R1": 3, "R2": 1}

    with pytest.raises(ValueError, match="non-list occurrences"):
        compare_scan_accuracy._rule_counts(
            _complete_report([{"id": "R1", "occurrences": "three"}]),
            frozenset(),
        )


def test_cohort_and_case_gates_prevent_cross_cohort_cancellation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_case(corpus, "benign")
    _write_case(corpus, "real-world")
    manifest = tmp_path / "manifest.json"
    policy = {**ZERO_TOLERANCE_POLICY, "max_candidate_false_positives": 1}
    _write_manifest(manifest, policy=policy)

    def fake_scan(
        executable: Path,
        target: Path,
        worktree: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        del worktree, kwargs
        if target.name == "benign":
            return _complete_report([{"id": "R1"}] if executable.name == "baseline" else [])
        return _complete_report(
            [{"id": "R1"}] if executable.name == "baseline" else [{"id": "R1"}, {"id": "R1"}]
        )

    _mock_scanner_identities(monkeypatch)
    monkeypatch.setattr(compare_scan_accuracy, "_run_scan", fake_scan)
    result = _compare(tmp_path, manifest, corpus)

    assert result["adjudication"]["delta"]["false_positives"] == 0
    assert (
        result["adjudication"]["by_classification"]["approved_real_world"]["delta"][
            "false_positives"
        ]
        == 1
    )
    assert result["passed"] is False
    violations = result["material_regression"]["violations"]
    assert any(
        violation.get("scope") == "cohort"
        and violation.get("classification") == "approved_real_world"
        for violation in violations
    )
    assert any(
        violation.get("scope") == "case" and violation.get("case_id") == "real-world"
        for violation in violations
    )


def test_bound_approval_cannot_be_reused_for_different_violations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_case(corpus, "benign")
    _write_case(corpus, "real-world")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)

    def fake_scan(
        executable: Path,
        target: Path,
        worktree: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        del worktree, kwargs
        if executable.name == "baseline":
            return _complete_report([] if target.name == "benign" else [{"id": "R1"}])
        return _complete_report([{"id": "R2"}] if target.name == "benign" else [{"id": "R1"}])

    _mock_scanner_identities(monkeypatch)
    monkeypatch.setattr(compare_scan_accuracy, "_run_scan", fake_scan)
    rejected = _compare(tmp_path, manifest, corpus)
    approval = tmp_path / "approval.json"
    _write_bound_approval(approval, rejected)
    document = json.loads(approval.read_text(encoding="utf-8"))
    document["violations"] = []
    document["violations_sha256"] = compare_scan_accuracy._json_sha256([])
    approval.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="exact violations"):
        _compare(
            tmp_path,
            manifest,
            corpus,
            approval_artifact=approval,
            approval_reviewer="Security Reviewer",
        )

    _write_bound_approval(approval, rejected)
    document = json.loads(approval.read_text(encoding="utf-8"))
    document["candidate_identity"]["revision"] = "c" * 40
    approval.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="exact candidate_identity"):
        _compare(
            tmp_path,
            manifest,
            corpus,
            approval_artifact=approval,
            approval_reviewer="Security Reviewer",
        )


def test_manifest_rejects_unknown_duplicate_and_unknown_classification_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_case(corpus, "benign")
    _write_case(corpus, "real-world")
    manifest = tmp_path / "manifest.json"
    _mock_scanner_identities(monkeypatch)

    document = {
        "schema_version": 2,
        "material_regression_policy": ZERO_TOLERANCE_POLICY,
        "cases": _cases(),
        "casez": [],
    }
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown accuracy manifest field"):
        _compare(tmp_path, manifest, corpus)

    cases = _cases()
    cases[0]["clasification"] = cases[0]["classification"]
    _write_manifest(manifest, cases=cases)
    with pytest.raises(ValueError, match="Unknown accuracy case field"):
        _compare(tmp_path, manifest, corpus)

    cases = _cases()
    cases[0]["classification"] = "maintained-benign"
    _write_manifest(manifest, cases=cases)
    with pytest.raises(ValueError, match="classification must be one of"):
        _compare(tmp_path, manifest, corpus)

    manifest.write_text('{"schema_version":2,"schema_version":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        _compare(tmp_path, manifest, corpus)


def test_runtime_identity_hashes_fixed_environment_and_dependency_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    interpreter = tmp_path / "python3"
    interpreter.write_text("runtime", encoding="utf-8")
    monkeypatch.setenv("SKILLSPECTOR_MAX_FILES", "attacker-controlled")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai-token")
    monkeypatch.setenv("PIP_INDEX_URL", "https://user:private-token@example.invalid/simple")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-aws-token")
    payload = {
        "python_version": "3.13.1",
        "python_implementation": "CPython",
        "python_cache_tag": "cpython-313",
        "python_hexversion": 51183856,
        "platform": "test-platform",
        "machine": "test-machine",
        "byteorder": "little",
        "dependencies": [
            {
                "name": "pyyaml",
                "version": "6.0.2",
                "recorded_file_count": 10,
                "distribution_metadata_sha256": "sha256:dependency",
            }
        ],
    }

    def fake_run(command, **kwargs):
        assert command[:4] == [str(interpreter), "-I", "-B", "-c"]
        assert "SKILLSPECTOR_MAX_FILES" not in kwargs["env"]
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert "PIP_INDEX_URL" not in kwargs["env"]
        assert "AWS_SECRET_ACCESS_KEY" not in kwargs["env"]
        assert Path(kwargs["env"]["HOME"]).is_dir()
        assert not any(Path(kwargs["env"]["HOME"]).iterdir())
        return subprocess.CompletedProcess(command, 0, json.dumps(payload).encode(), b"")

    monkeypatch.setattr(compare_scan_accuracy, "_run_bounded", fake_run)
    identity = compare_scan_accuracy._runtime_identity(interpreter, tmp_path)

    assert identity["dependencies"] == payload["dependencies"]
    assert str(identity["dependency_identity_sha256"]).startswith("sha256:")
    assert str(identity["runtime_identity_sha256"]).startswith("sha256:")
    assert identity["environment"]["PYTHONHASHSEED"] == "0"
    assert identity["environment"]["HOME"] == compare_scan_accuracy._ISOLATED_HOME_MARKER
    assert identity["environment_sha256"] == compare_scan_accuracy._json_sha256(
        identity["environment"]
    )
    rendered = json.dumps(identity)
    assert "private-token" not in rendered
    assert "secret-openai-token" not in rendered
    assert "secret-aws-token" not in rendered


def test_runtime_probe_hashes_installed_and_editable_dependency_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    installed_root = tmp_path / "site-packages"
    installed_root.mkdir()
    installed_file = installed_root / "dependency.py"
    installed_file.write_text("VALUE = 'installed-v1'\n", encoding="utf-8")
    editable_root = tmp_path / "editable-dependency"
    editable_root.mkdir()
    editable_file = editable_root / "source.py"
    editable_file.write_text("VALUE = 'editable-v1'\n", encoding="utf-8")

    class FakeDistribution:
        metadata = {"Name": "example-dependency"}
        version = "1.0"
        files = ["dependency.py"]

        def read_text(self, name: str) -> str | None:
            if name == "RECORD":
                return "dependency.py,,\n"
            if name == "METADATA":
                return "Name: example-dependency\nVersion: 1.0\n"
            if name == "direct_url.json":
                return json.dumps({"url": editable_root.as_uri(), "dir_info": {"editable": True}})
            return None

        def locate_file(self, package_path: object) -> Path:
            return installed_root / str(package_path)

    monkeypatch.setattr(
        importlib.metadata,
        "distributions",
        lambda: [FakeDistribution()],
    )

    def probe() -> dict[str, object]:
        rendered = io.StringIO()
        with redirect_stdout(rendered):
            exec(compare_scan_accuracy._RUNTIME_IDENTITY_PROBE, {})
        value = json.loads(rendered.getvalue())
        assert isinstance(value, dict)
        return value

    original = probe()
    dependency = original["dependencies"][0]
    assert dependency["installed_file_count"] == 1
    assert dependency["editable"] is True
    assert dependency["editable_file_count"] == 1
    assert str(installed_root) not in json.dumps(original)
    assert str(editable_root) not in json.dumps(original)

    installed_file.write_text("VALUE = 'installed-v2'\n", encoding="utf-8")
    installed_changed = probe()
    assert installed_changed["dependencies"] != original["dependencies"]

    installed_file.write_text("VALUE = 'installed-v1'\n", encoding="utf-8")
    editable_file.write_text("VALUE = 'editable-v2'\n", encoding="utf-8")
    editable_changed = probe()
    assert editable_changed["dependencies"] != original["dependencies"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"execution_successful": False}, "not execution-successful"),
        (
            {"analysis_completeness": {"is_complete": False, "status": "partial"}},
            "not analysis-complete",
        ),
        (
            {"analysis_completeness": {"partially_inspected_files": 1}},
            "incomplete coverage",
        ),
        (
            {"analysis_completeness": {"ledger_exceptions": [{"fatal": False}]}},
            "ledger_exceptions",
        ),
    ],
)
def test_accuracy_counts_reject_failed_or_incomplete_reports(
    mutation: dict[str, object],
    message: str,
) -> None:
    report = _complete_report([])
    for field, value in mutation.items():
        if field == "analysis_completeness":
            assert isinstance(value, dict)
            completeness = report["analysis_completeness"]
            assert isinstance(completeness, dict)
            completeness.update(value)
        else:
            report[field] = value
    with pytest.raises(ValueError, match=message):
        compare_scan_accuracy._rule_counts(report, frozenset())


def test_accuracy_snapshots_execute_against_private_immutable_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_case(corpus, "benign")
    _write_case(corpus, "real-world")
    cases = _cases()
    manifest_bytes = b"manifest"
    corpus_identity = compare_scan_accuracy._corpus_identity(corpus, cases, manifest_bytes)

    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
        for name, contents in (
            ("src/skillspector/__init__.py", b""),
            ("src/skillspector/cli.py", b"app = None\n"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))
    monkeypatch.setattr(
        compare_scan_accuracy,
        "_git_source_archive",
        lambda worktree, revision: archive_buffer.getvalue(),
    )
    runner_sha256 = compare_scan_accuracy._file_sha256(
        Path(compare_scan_accuracy.__file__).resolve()
    )
    baseline_identity = {
        "worktree": str(tmp_path / "baseline"),
        "revision": BASELINE_REVISION,
        "source_runner_sha256": runner_sha256,
    }
    candidate_identity = {
        "worktree": str(tmp_path / "candidate"),
        "revision": CANDIDATE_REVISION,
        "source_runner_sha256": runner_sha256,
    }

    snapshot_parent: Path | None = None
    with compare_scan_accuracy._accuracy_snapshots(
        corpus_root=corpus,
        cases=cases,
        manifest_bytes=manifest_bytes,
        corpus_identity=corpus_identity,
        baseline_identity=baseline_identity,
        candidate_identity=candidate_identity,
    ) as snapshots:
        snapshot_parent = snapshots["corpus_root"].parent
        assert snapshot_parent.lstat().st_uid == compare_scan_accuracy.os.geteuid()
        assert snapshots["corpus_root"] != corpus
        original_snapshot = (snapshots["corpus_root"] / "benign" / "SKILL.md").read_bytes()
        (corpus / "benign" / "SKILL.md").write_text("# attacker swap\n", encoding="utf-8")
        assert (snapshots["corpus_root"] / "benign" / "SKILL.md").read_bytes() == original_snapshot
        assert snapshots["baseline_source"] != tmp_path / "baseline" / "src"
        assert snapshots["runner"] != Path(compare_scan_accuracy.__file__).resolve()
    assert snapshot_parent is not None
    assert not snapshot_parent.exists()


def test_fresh_home_is_owned_empty_worktree_independent_and_cleaned(tmp_path: Path) -> None:
    worktree_git_file = tmp_path / ".git"
    worktree_git_file.write_text("gitdir: elsewhere\n", encoding="utf-8")
    home_path: Path | None = None
    with compare_scan_accuracy._fresh_owned_home() as home:
        home_path = home
        assert home.is_dir()
        assert home.lstat().st_uid == compare_scan_accuracy.os.geteuid()
        assert not any(home.iterdir())
        assert not home.is_relative_to(tmp_path)
        (home / "scanner-created-state").write_text("state", encoding="utf-8")
    assert home_path is not None
    assert not home_path.exists()


def test_bounded_subprocess_fails_closed_on_output_and_runtime_limits(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="output exceeded"):
        compare_scan_accuracy._run_bounded(
            [sys.executable, "-c", "import sys; sys.stdout.write('x' * 4096)"],
            cwd=tmp_path,
            timeout_seconds=5,
            stdout_limit_bytes=128,
            stderr_limit_bytes=128,
        )
    with pytest.raises(RuntimeError, match="timed out"):
        compare_scan_accuracy._run_bounded(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            cwd=tmp_path,
            timeout_seconds=0.05,
            stdout_limit_bytes=128,
            stderr_limit_bytes=128,
        )


def test_compare_scanners_rejects_ancestor_descendant_case_roots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_case(corpus, "nested")
    _write_case(corpus, "nested/child")
    manifest = tmp_path / "manifest.json"
    cases = _cases()
    cases[0]["path"] = "nested"
    cases[1]["path"] = "nested/child"
    _write_manifest(manifest, cases=cases)
    _mock_scanner_identities(monkeypatch)

    with pytest.raises(ValueError, match="must not overlap"):
        _compare(tmp_path, manifest, corpus)
