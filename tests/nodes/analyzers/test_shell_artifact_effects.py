# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native effect callbacks retain bytes only after command identity is proved."""

from dataclasses import replace

import pytest

from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.shell_artifact_effects import ShellArtifactAudit
from skillspector.shell_artifacts import ShellArtifact
from skillspector.shell_source_effects import ShellEffectCommand, source_effects_preserve
from skillspector.shell_sources import SAFE_PATH_MARKER, ShellSourceContext, ShellSourceResolver, ShellSourceValue


def _audit():
    return ShellArtifactAudit(ShellSourceContext("tests/workflow.sh", {"tests/workflow.sh": "echo fixture\n", "scripts/setup.sh": "sample() { echo inspected; }\n"}), tm._next_shell_invocation_word, lambda: None)


def _command(argv, **kwargs):
    return ShellEffectCommand(argv, {}, ".", "tests/workflow.sh", 0, **kwargs)


def test_actual_native_path_is_required_before_publishing_helper_bytes() -> None:
    audit = _audit()
    assert audit(_command(("mktemp", "-d", "/tmp/native.XXXXXX"))) is not None
    directory = "/tmp/native." + SAFE_PATH_MARKER
    assert audit(_command(("cat",), stdin="#!/bin/bash\necho known\n", redirects=(("1>", directory + "/helper"),))) is not None
    assert audit.read_source(directory + "/helper") == "#!/bin/bash\necho known\n"
    assert audit(_command(("/tmp/cat",), stdin="different", redirects=(("1>", directory + "/helper"),))) is None
    assert audit.read_source(directory + "/helper") == "#!/bin/bash\necho known\n"


def test_path_lookup_distinguishes_generated_helper_from_native_command() -> None:
    audit = _audit()
    assert audit.store.create_empty_directory("/tmp/helpers")
    assert audit.store.write("/tmp/helpers/bash", ShellArtifact(ShellSourceValue("echo replaced\n"), executable=True))
    command = replace(_command(("bash", "-c", "echo safe"), identity_only=True), environment={"PATH": "/tmp/helpers:" + audit.path_tail})
    result = audit(command)
    assert result is not None and not result.native_identity and result.script_path == "/tmp/helpers/bash"
    result = audit(replace(command, argv=("/bin/bash", "-c", "echo safe")))
    assert result is not None and result.native_identity and result.script_path is None
    assert audit.steps == 0


def test_unknown_or_unexecutable_path_entry_never_proves_native_identity() -> None:
    audit = _audit()
    command = replace(_command(("cat",), identity_only=True), environment={"PATH": "/tmp/unknown:" + audit.path_tail})
    assert audit(command) is None
    assert audit.store.create_empty_directory("/tmp/unknown")
    assert audit.store.write("/tmp/unknown/cat", ShellArtifact(ShellSourceValue("echo replaced\n")))
    assert audit(command) is None


def test_unknown_conditional_write_forgets_previous_generated_source() -> None:
    audit = _audit()
    assert audit.store.write("/tmp/generated.sh", ShellArtifact(ShellSourceValue("echo old\n")))
    result = audit(_command(("printf", "%s", "echo branch\n"), uncertain=True, redirects=(("1>", "/tmp/generated.sh"),)))
    assert result is not None and result.writes == ("/tmp/generated.sh",)
    assert audit.read_source("/tmp/generated.sh") is None


def test_source_provenance_and_executable_mode_survive_archive_roundtrip() -> None:
    audit = _audit()
    assert audit.store.create_empty_directory("/tmp/work")
    assert audit.store.create_empty_directory("/tmp/extracted")
    assert audit(_command(("mkdir", "-p", "/tmp/work/package"))) is not None
    assert audit(_command(("cp", "scripts/setup.sh", "/tmp/work/package/setup.sh"))) is not None
    assert audit(_command(("chmod", "+x", "/tmp/work/package/setup.sh"))) is not None
    assert audit(_command(("tar", "-I", "pigz", "-cf", "/tmp/package.tgz", "-C", "/tmp/work", "package"))) is not None
    assert audit(_command(("tar", "-I", "pigz", "-xf", "/tmp/package.tgz", "-C", "/tmp/extracted"))) is not None
    copied = audit.store.read("/tmp/extracted/package/setup.sh")
    assert copied is not None and copied.executable is True
    assert copied.value == audit.store.read("scripts/setup.sh").value


@pytest.mark.parametrize("redirects", [
    (("1>", "/tmp/first"), ("1>", "/tmp/second")),
    (("1>", "/tmp/first"), ("2>&", "1")),
])
def test_unsupported_stream_routing_cannot_publish_exact_generated_bytes(redirects) -> None:
    audit = _audit()
    result = audit(_command(("printf", "%s", "known"), redirects=redirects))
    assert result is not None and "/tmp/first" in result.writes
    assert audit.read_source("/tmp/first") is None


def test_known_script_execution_is_not_treated_as_an_inert_cached_file() -> None:
    audit = _audit()
    result = audit(_command(("scripts/setup.sh",)))
    assert result is not None and result.script_path == "scripts/setup.sh"
    assert result.status is None


def test_post_child_callback_publishes_only_the_inspected_child_stdout() -> None:
    audit = _audit()
    result = audit(_command(("scripts/setup.sh",), child_inspected=True, stdout="echo child\n", redirects=(("1>", "/tmp/child.sh"),)))
    assert result is not None and result.script_path is None
    assert audit.read_source("/tmp/child.sh") == "echo child\n"


def test_native_unknown_command_is_not_accepted_just_because_path_is_standard() -> None:
    assert _audit()(_command(("python3", "-c", "arbitrary effects"))) is None


@pytest.mark.parametrize("argv", [
    ("mv", "--target-directory=scripts", "input"),
    ("mv", "-tscripts", "input"),
    ("ln", "--target-directory=scripts", "input"),
    ("ln", "-tscripts", "input"),
    ("install", "--target-directory=scripts", "input"),
    ("truncate", "--size=0", "scripts/setup.sh"),
])
def test_destination_bearing_native_flags_are_not_discarded(argv) -> None:
    audit = _audit()
    assert audit(_command(argv)) is None


def test_unknown_copy_success_cannot_skip_a_failure_branch() -> None:
    audit = _audit()
    result = audit(_command(("cp", "unknown-input", "/tmp/output")))
    assert result is not None and result.status is None
    assert audit.store.read("/tmp/output") is None


@pytest.mark.parametrize("operation", [
    "mv --target-directory=scripts input",
    "ln --target-directory=scripts input",
    "if cp missing /tmp/output; then :; else printf changed > scripts/setup.sh; fi",
    "cd() { :; }; cd /tmp/other; printf changed > scripts/setup.sh",
])
def test_effect_callback_cannot_prove_cached_source_unchanged_after_unbounded_writer(operation) -> None:
    caller = operation + "\n"
    context = ShellSourceContext("tests/caller.sh", {"tests/caller.sh": caller, "scripts/setup.sh": "echo inspected\n"})
    resolver = ShellSourceResolver(context, tm._next_shell_invocation_word, lambda text, start, end: tm._skip_command_substitution(text, start, end, lambda: None), tm._static_heredoc_header, tm._skip_static_heredoc_bodies, lambda: None)
    audit = ShellArtifactAudit(context, tm._next_shell_invocation_word, lambda: None)
    assert not source_effects_preserve(resolver, "scripts/setup.sh", len(caller), read_source=audit.read_source, native_effect=audit)


def test_inherited_native_identity_does_not_invent_command_availability() -> None:
    audit = _audit()
    result = audit(_command(("command", "-v", "missing-tool")))
    assert result is not None and result.status is None and result.stdout is None


def test_cd_distinguishes_known_input_root_from_unknown_runtime_directory() -> None:
    audit = _audit()
    result = audit(_command(("cd", "tests/..")))
    assert result is not None and result.status is True
    result = audit(_command(("cd", "/runtime/unknown")))
    assert result is not None and result.status is None


def test_native_archive_copy_retains_tree_names_modes_and_readonly_listing_uncertainty() -> None:
    audit = _audit()
    assert audit.store.create_empty_directory("/tmp/source")
    assert audit.store.create_empty_directory("/tmp/destination")
    artifact = ShellArtifact(ShellSourceValue("echo copied\n"), executable=True)
    assert audit.store.write("/tmp/source/package/runner.sh", artifact)
    result = audit(_command(("cp", "-a", "/tmp/source/package", "/tmp/destination/")))
    assert result is not None and result.status is True
    assert audit.store.read("/tmp/destination/package/runner.sh") == artifact
    assert audit(_command(("tar", "-cf", "/tmp/package.tar", "-C", "/tmp/destination", "package"))) is not None
    listing = audit(_command(("tar", "-tf", "/tmp/package.tar")))
    assert listing is not None and listing.status is True and listing.stdout is None and not listing.writes
