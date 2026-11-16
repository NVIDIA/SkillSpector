# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native effect callbacks retain bytes only after command identity is proved."""

from dataclasses import replace

import pytest

from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.shell_artifact_effects import ShellArtifactAudit
from skillspector.shell_artifacts import ShellArtifact
from skillspector.shell_source_effects import (
    SHELL_PID_MARKER,
    ShellEffectCommand,
    source_effects_preserve,
)
from skillspector.shell_sources import (
    SAFE_PATH_MARKER,
    ShellSourceContext,
    ShellSourceResolver,
    ShellSourceValue,
)


def _audit():
    return ShellArtifactAudit(
        ShellSourceContext(
            "tests/workflow.sh",
            {
                "tests/workflow.sh": "echo fixture\n",
                "scripts/setup.sh": "sample() { echo inspected; }\n",
            },
        ),
        tm._next_shell_invocation_word,
        lambda: None,
    )


def _command(argv, **kwargs):
    kwargs.setdefault("fail_stop", True)
    return ShellEffectCommand(argv, {}, ".", "tests/workflow.sh", 0, **kwargs)


def test_preserve_callback_keeps_unknown_printf_output_out_of_generated_sources() -> None:
    audit = _audit()
    assert audit.store.create_empty_directory("/tmp/logs")
    request = _command(("printf", "%s\\n", None), redirects=(("1>", "/tmp/logs/output"),))
    assert audit(request) is None
    result = audit.preserve(request)
    assert result is not None and result.stdout is None and result.status is None
    assert result.writes == ("/tmp/logs/output",)
    assert audit.read_source("/tmp/logs/output") is None
    assert audit.read_source("scripts/setup.sh") is not None


@pytest.mark.parametrize(
    "argv",
    [
        ("printf", None, "data"),
        ("printf", "%n", None),
        ("printf", "-v", None),
        ("printf", "%b", None),
        ("/tmp/printf", "%s", None),
    ],
)
def test_preserve_callback_does_not_expand_printf_code_or_identity_contract(argv) -> None:
    assert _audit().preserve(_command(argv)) is None


def test_preserve_callback_still_accounts_for_unknown_printf_destination() -> None:
    audit = _audit()
    assert audit.preserve(_command(("printf", "%s", None), redirects=(("1>", None),))) is None
    result = audit.preserve(
        _command(("printf", "%s", None), redirects=(("1>", "scripts/setup.sh"),))
    )
    assert result is not None and result.writes == ("scripts/setup.sh",)
    assert audit.read_source("scripts/setup.sh") is None


@pytest.mark.parametrize("program", ["1p", "1d;$d", "/^sample() {/,/^}/p", "s/ready/done/g"])
def test_native_readonly_sed_keeps_its_output_unknown(program: str) -> None:
    audit = _audit()
    result = audit(_command(("sed", "-n", program, "scripts/setup.sh")))
    assert (
        result is not None
        and result.writes == ()
        and result.stdout is None
        and result.status is None
    )
    result = audit(
        _command(
            ("sed", "-n", program, "scripts/setup.sh"), redirects=(("1>", "scripts/setup.sh"),)
        )
    )
    assert result is not None and result.writes == ("scripts/setup.sh",)
    assert audit.read_source("scripts/setup.sh") is None


@pytest.mark.parametrize(
    "arguments, preserved",
    [
        (("scripts/setup.sh",), True),
        (("-bi", "--mime-type", "scripts/setup.sh"), True),
        (("--", "-C"), True),
        (("-m", "-C", "scripts/setup.sh"), True),
        (("--magic-file=-C", "scripts/setup.sh"), True),
        (("-F", "--compile", "scripts/setup.sh"), True),
        (("-C", "-m", "definitions"), False),
        (("--compile", "--magic-file=definitions"), False),
        (("scripts/setup.sh", "-C"), False),
        (("-bC", "scripts/setup.sh"), False),
        (("--comp", "scripts/setup.sh"), False),
        (("-m", "definitions", "-C"), False),
        (("--magic-file",), False),
        ((None,), False),
    ],
)
def test_native_file_requires_readonly_arguments(arguments, preserved) -> None:
    audit = _audit()
    result = audit(_command(("file", *arguments)))
    assert (result is not None) is preserved
    if preserved:
        assert result.writes == () and result.stdout is None and result.status is None


def test_native_file_readonly_mode_still_accounts_for_redirects() -> None:
    audit = _audit()
    result = audit(_command(("file", "scripts/setup.sh"), redirects=(("1>", "scripts/setup.sh"),)))
    assert result is not None and result.writes == ("scripts/setup.sh",)
    assert audit.read_source("scripts/setup.sh") is None


@pytest.mark.parametrize(
    "argv",
    [
        ("sed", "-n", "e", "scripts/setup.sh"),
        ("sed", "-n", "w scripts/setup.sh", "input"),
        ("sed", "-n", "s/a/b/e", "input"),
        ("sed", "-n", "s/a/b/w scripts/setup.sh", "input"),
        ("sed", "-n", "p; e", "input"),
        ("sed", "-n", "p", "-i"),
        ("sed", "-n", "p", "input", "-e", "e"),
        ("sed", "-n", None, "input"),
        ("sed", "-n", "p", None),
        ("/tmp/sed", "-n", "p", "input"),
    ],
)
def test_readonly_sed_never_exempts_code_writes_unknown_argv_or_identity(argv) -> None:
    assert _audit()(_command(argv)) is None


def test_mktemp_regular_file_retains_exact_ownership_without_a_directory_claim() -> None:
    audit = _audit()
    request = _command(("mktemp", "/tmp/artifactory.XXXXXX.netrc"))
    result = audit(request)
    assert result is not None and result.status is True
    path = result.stdout.removesuffix("\n")
    artifact = audit.store.read(path)
    assert artifact is not None and artifact.value.text == ""
    assert artifact.value.origins == audit._origin(request)
    assert artifact.executable is False
    assert result.writes == (path,) and audit.store.owns(path)
    assert not audit.store.owns("/tmp") and not audit.store.owns(path + "/child")
    assert audit.store.inventory(path) is None
    assert audit(_command(("cat",), stdin="credential", redirects=(("1>", path),))) is not None
    assert audit(_command(("cat",), stdin="-suffix", redirects=(("1>>", path),))) is not None
    assert audit.read_source(path) == "credential-suffix"
    second = audit(request)
    assert second is not None and second.stdout != result.stdout
    assert audit.read_source(path) == "credential-suffix"


@pytest.mark.parametrize("directory", [False, True])
def test_mktemp_replaces_only_the_final_x_run(directory: bool) -> None:
    audit = _audit()
    args = ("mktemp", *(("-d",) if directory else ()), "/tmp/literalXXX.generatedXXXXXX.netrc")
    result = audit(_command(args))
    assert result is not None
    path = result.stdout.removesuffix("\n")
    assert path.startswith("/tmp/literalXXX.generated" + SAFE_PATH_MARKER)
    assert path.endswith(".netrc")
    assert audit.store.owns(path)
    assert (audit.store.read(path) is None) is directory


@pytest.mark.parametrize(
    "args",
    [
        ("mktemp",),
        ("mktemp", "-u", "/tmp/file.XXXXXX"),
        ("mktemp", "--suffix=.netrc", "/tmp/file.XXXXXX"),
        ("mktemp", "/tmp/file.XX"),
        ("mktemp", "/tmp/file.XXXXXX.suffixX"),
        ("mktemp", "/tmp/../file.XXXXXX"),
        ("mktemp", "relative.XXXXXX"),
        ("mktemp", "/tmp/file.XXXXXX", "extra"),
    ],
)
def test_mktemp_file_refuses_unsupported_templates_and_modes(args: tuple[str, ...]) -> None:
    assert _audit()(_command(args)) is None


@pytest.mark.parametrize("changes", [{"uncertain": True}, {"fail_stop": False}])
def test_mktemp_file_requires_ordered_success(changes: dict) -> None:
    audit = _audit()
    assert audit(_command(("mktemp", "/tmp/file.XXXXXX"), **changes)) is None


def test_mktemp_file_cannot_bypass_native_identity_or_invalidated_paths() -> None:
    audit = _audit()
    request = _command(("mktemp", "/tmp/file.XXXXXX"))
    assert audit(replace(request, environment={"PATH": "/tmp/unknown:" + audit.path_tail})) is None
    assert audit(replace(request, argv=("/tmp/mktemp", "/tmp/file.XXXXXX"))) is None
    audit.store.invalidate("/tmp")
    assert audit(request) is None


def test_actual_native_path_is_required_before_publishing_helper_bytes() -> None:
    audit = _audit()
    assert audit(_command(("mktemp", "-d", "/tmp/native.XXXXXX"))) is not None
    directory = "/tmp/native." + SAFE_PATH_MARKER
    assert (
        audit(
            _command(
                ("cat",),
                stdin="#!/bin/bash\necho known\n",
                redirects=(("1>", directory + "/helper"),),
            )
        )
        is not None
    )
    assert audit.read_source(directory + "/helper") == "#!/bin/bash\necho known\n"
    assert (
        audit(
            _command(("/tmp/cat",), stdin="different", redirects=(("1>", directory + "/helper"),))
        )
        is None
    )
    assert audit.read_source(directory + "/helper") == "#!/bin/bash\necho known\n"


def test_path_lookup_distinguishes_generated_helper_from_native_command() -> None:
    audit = _audit()
    assert audit.store.create_empty_directory("/tmp/helpers")
    assert audit.store.write(
        "/tmp/helpers/bash", ShellArtifact(ShellSourceValue("echo replaced\n"), executable=True)
    )
    command = replace(
        _command(("bash", "-c", "echo safe"), identity_only=True),
        environment={"PATH": "/tmp/helpers:" + audit.path_tail},
    )
    result = audit(command)
    assert (
        result is not None
        and not result.native_identity
        and result.script_path == "/tmp/helpers/bash"
    )
    result = audit(replace(command, argv=("/bin/bash", "-c", "echo safe")))
    assert result is not None and result.native_identity and result.script_path is None
    assert audit.steps == 0


def test_unknown_or_unexecutable_path_entry_never_proves_native_identity() -> None:
    audit = _audit()
    command = replace(
        _command(("cat",), identity_only=True),
        environment={"PATH": "/tmp/unknown:" + audit.path_tail},
    )
    assert audit(command) is None
    assert audit.store.create_empty_directory("/tmp/unknown")
    assert audit.store.write("/tmp/unknown/cat", ShellArtifact(ShellSourceValue("echo replaced\n")))
    assert audit(command) is None


@pytest.mark.parametrize("interpreter", ["bash", "/bin/bash", "/usr/bin/bash"])
def test_explicit_bash_reads_known_nonexecutable_generated_script(interpreter: str) -> None:
    audit = _audit()
    assert audit.store.create_empty_directory("/tmp/scripts")
    assert audit.store.write(
        "/tmp/scripts/child.sh", ShellArtifact(ShellSourceValue("printf ready\n"), executable=False)
    )
    request = _command(("/tmp/scripts/child.sh", "argument"), interpreter=interpreter)
    result = audit(request)
    assert result is not None and result.script_path == "/tmp/scripts/child.sh"
    assert not result.native_identity
    assert audit(replace(request, interpreter=None)) is None
    result = audit(replace(request, child_inspected=True, stdout="ready", child_status=True))
    assert result is not None and result.stdout == "ready" and not result.native_identity


@pytest.mark.parametrize(
    "interpreter, script",
    [("sh", "scripts/setup.sh"), ("/tmp/bash", "scripts/setup.sh"), ("bash", "scripts/missing.sh")],
)
def test_explicit_interpreter_requires_native_bash_and_known_script(
    interpreter: str, script: str
) -> None:
    assert _audit()(_command((script,), interpreter=interpreter)) is None


def test_explicit_bash_cannot_hide_path_shadow_or_changed_source() -> None:
    audit = _audit()
    assert audit.store.create_empty_directory("/tmp/helpers")
    assert audit.store.write(
        "/tmp/helpers/bash", ShellArtifact(ShellSourceValue("printf replaced\n"), executable=True)
    )
    request = _command(("scripts/setup.sh",), interpreter="bash")
    assert audit(replace(request, environment={"PATH": "/tmp/helpers:" + audit.path_tail})) is None
    audit.store.invalidate("scripts/setup.sh")
    assert audit(request) is None


def test_unknown_conditional_write_forgets_previous_generated_source() -> None:
    audit = _audit()
    assert audit.store.write("/tmp/generated.sh", ShellArtifact(ShellSourceValue("echo old\n")))
    result = audit(
        _command(
            ("printf", "%s", "echo branch\n"),
            uncertain=True,
            redirects=(("1>", "/tmp/generated.sh"),),
        )
    )
    assert result is not None and result.writes == ("/tmp/generated.sh",)
    assert audit.read_source("/tmp/generated.sh") is None


def test_native_echo_diagnostic_has_known_effects_without_proven_output() -> None:
    result = _audit()(_command(("echo", "--platform is required"), redirects=(("1>&", "2"),)))
    assert result is not None
    assert result.stdout is None and result.status is None and result.writes == ()


@pytest.mark.parametrize(
    "args, expected",
    [
        (("known",), "known\n"),
        (("-n", "known"), None),
        ((r"one\ntwo",), None),
        ((None,), None),
    ],
)
def test_native_echo_preserves_only_proven_generated_bytes(args, expected) -> None:
    audit = _audit()
    assert audit.store.create_empty_directory("/tmp/work")
    assert audit.store.write("/tmp/work/output.sh", ShellArtifact(ShellSourceValue("old\n")))
    result = audit(_command(("echo", *args), redirects=(("1>", "/tmp/work/output.sh"),)))
    assert result is not None and result.writes == ("/tmp/work/output.sh",)
    assert audit.read_source("/tmp/work/output.sh") == expected


@pytest.mark.parametrize(
    "argv",
    [
        ("df", "-B1", "--output=avail", "/tmp"),
        ("awk", "NR == 2 {print $1}"),
        ("awk", "NR==2{ print $1; }"),
    ],
)
def test_disk_space_query_pipeline_keeps_output_and_status_unknown(argv) -> None:
    audit = _audit()
    result = audit(_command(argv))
    assert result is not None and result.stdout is None and result.status is None
    assert result.writes == () and result.native_identity
    assert audit.store.create_empty_directory("/tmp/work")
    assert audit.store.write("/tmp/work/value", ShellArtifact(ShellSourceValue("old\n")))
    result = audit(_command(argv, redirects=(("1>", "/tmp/work/value"),)))
    assert result is not None and result.writes == ("/tmp/work/value",)
    assert audit.read_source("/tmp/work/value") is None


@pytest.mark.parametrize(
    "args",
    [
        ('NR == 2 {print $1 > "scripts/setup.sh"}',),
        ('NR == 2 {print $1; system("rm -rf /")}',),
        ('NR == 2 {print $1} END {print "changed" > "scripts/setup.sh"}',),
        ("-f", "scripts/setup.sh"),
        (None,),
    ],
)
def test_awk_query_allowance_cannot_hide_extra_program_effects(args) -> None:
    assert _audit()(_command(("awk", *args))) is None


def test_source_provenance_and_executable_mode_survive_archive_roundtrip() -> None:
    audit = _audit()
    assert audit.store.create_empty_directory("/tmp/work")
    assert audit.store.create_empty_directory("/tmp/extracted")
    assert audit(_command(("mkdir", "-p", "/tmp/work/package"))) is not None
    assert audit(_command(("cp", "scripts/setup.sh", "/tmp/work/package/setup.sh"))) is not None
    assert audit(_command(("chmod", "+x", "/tmp/work/package/setup.sh"))) is not None
    assert (
        audit(
            _command(
                ("tar", "-I", "pigz", "-cf", "/tmp/work/package.tgz", "-C", "/tmp/work", "package")
            )
        )
        is not None
    )
    assert (
        audit(
            _command(("tar", "-I", "pigz", "-xf", "/tmp/work/package.tgz", "-C", "/tmp/extracted"))
        )
        is not None
    )
    copied = audit.store.read("/tmp/extracted/package/setup.sh")
    assert copied is not None and copied.executable is True
    assert copied.value == audit.store.read("scripts/setup.sh").value


@pytest.mark.parametrize("operation", ["archive", "copy"])
@pytest.mark.parametrize("empty_present", [False, True])
def test_transferred_empty_directory_cannot_skip_a_cached_source_overwrite(
    operation: str, empty_present: bool
) -> None:
    caller = (
        "set -e\nwork=$(mktemp -d /tmp/artifact-source.XXXXXX)\n"
        "out=$(mktemp -d /tmp/artifact-target.XXXXXX)\n"
        'mkdir -p "$work/tree"\n'
        + ('mkdir -p "$work/tree/empty"\n' if empty_present else "")
        + 'cp scripts/marker.sh "$work/tree/known"\n'
        + (
            'tar -cf "$work/package.tar" -C "$work" tree\ntar -xf "$work/package.tar" -C "$out"\n'
            if operation == "archive"
            else 'cp -a "$work/tree" "$out"\n'
        )
        + 'if test -d "$out/tree/empty"; then printf changed > scripts/setup.sh; fi\n'
    )
    context = ShellSourceContext(
        "tests/caller.sh",
        {
            "tests/caller.sh": caller,
            "scripts/setup.sh": "echo inspected\n",
            "scripts/marker.sh": "echo marker\n",
        },
    )
    resolver = ShellSourceResolver(
        context,
        tm._next_shell_invocation_word,
        lambda text, start, end: tm._skip_command_substitution(text, start, end, lambda: None),
        tm._static_heredoc_header,
        tm._skip_static_heredoc_bodies,
        lambda: None,
    )
    audit = ShellArtifactAudit(context, tm._next_shell_invocation_word, lambda: None)
    assert source_effects_preserve(
        resolver,
        "scripts/setup.sh",
        len(caller),
        read_source=audit.read_source,
        native_effect=audit,
    ) is (not empty_present)


@pytest.mark.parametrize("operator", ["1>", "1>>"])
@pytest.mark.parametrize("executable", [False, True, None])
@pytest.mark.parametrize("open_first", [False, True])
def test_redirect_preserves_destination_mode_and_helper_identity(
    operator: str, executable: bool | None, open_first: bool
) -> None:
    audit = _audit()
    path = "/tmp/helpers/runner"
    assert audit.store.create_empty_directory("/tmp/helpers")
    assert audit.store.write(
        path, ShellArtifact(ShellSourceValue("echo old\n"), executable=executable)
    )
    command = _command(("printf", "%s", "echo new\n"), redirects=((operator, path),))
    if open_first:
        assert audit(replace(command, redirect_open=True)) is not None
        assert audit.store.read(path).executable is executable
    assert audit(command) is not None
    artifact = audit.store.read(path)
    assert artifact.executable is executable
    assert artifact.value.text == ("echo old\necho new\n" if operator == "1>>" else "echo new\n")
    identity = audit(_command((path,), identity_only=True))
    if executable is True:
        assert identity is not None and identity.script_path == path
    else:
        assert identity is None


def test_redirected_executable_source_does_not_make_a_new_file_executable() -> None:
    audit = _audit()
    assert audit.store.create_empty_directory("/tmp/helpers")
    assert audit.store.write(
        "/tmp/helpers/source", ShellArtifact(ShellSourceValue("echo helper\n"), executable=True)
    )
    assert (
        audit(_command(("cat", "/tmp/helpers/source"), redirects=(("1>", "/tmp/helpers/runner"),)))
        is not None
    )
    assert audit.store.read("/tmp/helpers/runner").executable is False
    assert audit(_command(("/tmp/helpers/runner",), identity_only=True)) is None


@pytest.mark.parametrize(
    "redirects",
    [
        (("1>", "/tmp/first"), ("1>", "/tmp/second")),
        (("1>", "/tmp/first"), ("2>&", "1")),
    ],
)
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
    assert audit.store.create_empty_directory("/tmp/work")
    result = audit(
        _command(
            ("scripts/setup.sh",),
            child_inspected=True,
            stdout="echo child\n",
            redirects=(("1>", "/tmp/work/child.sh"),),
        )
    )
    assert result is not None and result.script_path is None
    assert audit.read_source("/tmp/work/child.sh") == "echo child\n"


def test_native_unknown_command_is_not_accepted_just_because_path_is_standard() -> None:
    assert _audit()(_command(("python3", "-c", "arbitrary effects"))) is None


@pytest.mark.parametrize(
    "assignment, expected", [("", "echo one\\n"), ("MATCH_MODE=two\n", "echo two\\n")]
)
def test_generated_helper_uses_current_inherited_exported_setting(
    assignment: str, expected: str
) -> None:
    producer = "set -euo pipefail\n" + assignment + 'p4 print > "$1"\n'
    helper = """case "$MATCH_MODE" in
one) printf '%s' 'echo one\\n' ;;
two) printf '%s' 'echo two\\n' ;;
esac
"""
    context = ShellSourceContext("producer.sh", {"producer.sh": producer})
    audit = ShellArtifactAudit(context, tm._next_shell_invocation_word, lambda: None)
    assert audit.store.create_empty_directory("/tmp/helpers")
    assert audit.store.create_empty_directory("/tmp/result")
    assert audit.store.write(
        "/tmp/helpers/p4", ShellArtifact(ShellSourceValue(helper), executable=True)
    )
    command = ShellEffectCommand(
        ("./producer.sh", "/tmp/result/output.sh"),
        {"MATCH_MODE": "one", "PATH": "/tmp/helpers:" + audit.path_tail},
        ".",
        "producer.sh",
        0,
        fail_stop=True,
    )
    result = audit(command)
    assert result is not None and result.script_path is None
    assert audit.read_source("/tmp/result/output.sh") == expected


@pytest.mark.parametrize(
    "argv",
    [
        ("mv", "--target-directory=scripts", "input"),
        ("mv", "-tscripts", "input"),
        ("ln", "--target-directory=scripts", "input"),
        ("ln", "-tscripts", "input"),
        ("install", "--target-directory=scripts", "input"),
        ("truncate", "--size=0", "scripts/setup.sh"),
    ],
)
def test_destination_bearing_native_flags_are_not_discarded(argv) -> None:
    audit = _audit()
    assert audit(_command(argv)) is None


def test_unknown_copy_success_cannot_skip_a_failure_branch() -> None:
    audit = _audit()
    result = audit(_command(("cp", "unknown-input", "/tmp/output")))
    assert result is not None and result.status is None
    assert audit.store.read("/tmp/output") is None


@pytest.mark.parametrize(
    "operation",
    [
        "mv --target-directory=scripts input",
        "ln --target-directory=scripts input",
        "if cp missing /tmp/output; then :; else printf changed > scripts/setup.sh; fi",
        "cd() { :; }; cd /tmp/other; printf changed > scripts/setup.sh",
    ],
)
def test_effect_callback_cannot_prove_cached_source_unchanged_after_unbounded_writer(
    operation,
) -> None:
    caller = operation + "\n"
    context = ShellSourceContext(
        "tests/caller.sh", {"tests/caller.sh": caller, "scripts/setup.sh": "echo inspected\n"}
    )
    resolver = ShellSourceResolver(
        context,
        tm._next_shell_invocation_word,
        lambda text, start, end: tm._skip_command_substitution(text, start, end, lambda: None),
        tm._static_heredoc_header,
        tm._skip_static_heredoc_bodies,
        lambda: None,
    )
    audit = ShellArtifactAudit(context, tm._next_shell_invocation_word, lambda: None)
    assert not source_effects_preserve(
        resolver,
        "scripts/setup.sh",
        len(caller),
        read_source=audit.read_source,
        native_effect=audit,
    )


def test_inherited_native_identity_does_not_invent_command_availability() -> None:
    audit = _audit()
    result = audit(_command(("command", "-v", "missing-tool")))
    assert result is not None and result.status is None and result.stdout is None


def test_dirname_option_does_not_invent_successful_path_output() -> None:
    audit = _audit()
    assert audit(_command(("dirname", "-z"))) is None
    result = audit(_command(("dirname", "scripts/setup.sh")))
    assert result is not None and result.stdout == "scripts\n" and result.status is True


def test_cd_distinguishes_known_input_root_from_unknown_runtime_directory() -> None:
    audit = _audit()
    result = audit(_command(("cd", "tests/..")))
    assert result is not None and result.status is True
    result = audit(_command(("cd", "/runtime/unknown")))
    assert result is not None and result.status is None
    result = audit(_command(("cd", "./..")))
    assert result is not None and result.status is None and result.native_identity
    result = audit(_command(("cd", "tests/.."), fail_stop=False))
    assert result is not None and result.status is None


@pytest.mark.parametrize("source_mode", [False, True, None])
@pytest.mark.parametrize("destination_mode", [False, True, None])
def test_plain_copy_overwrite_preserves_destination_mode_and_source_provenance(
    source_mode: bool | None, destination_mode: bool | None
) -> None:
    audit = _audit()
    assert audit.store.create_empty_directory("/tmp/helpers")
    source = replace(audit.store.read("scripts/setup.sh"), executable=source_mode)
    assert audit.store.write("/tmp/helpers/source", source)
    assert audit.store.write(
        "/tmp/helpers/runner",
        ShellArtifact(ShellSourceValue("printf old\n"), executable=destination_mode),
    )
    result = audit(_command(("cp", "/tmp/helpers/source", "/tmp/helpers/runner")))
    assert result is not None and result.status is True
    assert result.writes == ("/tmp/helpers/runner",)
    assert audit.store.read("/tmp/helpers/runner") == replace(source, executable=destination_mode)
    identity = audit(_command(("/tmp/helpers/runner",), identity_only=True))
    if destination_mode is True:
        assert identity is not None and identity.script_path == "/tmp/helpers/runner"
        assert not identity.native_identity
    else:
        assert identity is None


@pytest.mark.parametrize("source_mode", [False, True, None])
def test_plain_copy_to_absent_destination_retains_source_mode(source_mode: bool | None) -> None:
    audit = _audit()
    assert audit.store.create_empty_directory("/tmp/helpers")
    source = replace(audit.store.read("scripts/setup.sh"), executable=source_mode)
    assert audit.store.write("/tmp/helpers/source", source)
    result = audit(_command(("cp", "/tmp/helpers/source", "/tmp/helpers/runner")))
    assert result is not None and result.status is True
    assert audit.store.read("/tmp/helpers/runner") == source


def test_native_archive_copy_retains_tree_names_modes_and_readonly_listing_uncertainty() -> None:
    audit = _audit()
    assert audit.store.create_empty_directory("/tmp/source")
    assert audit.store.create_empty_directory("/tmp/destination")
    artifact = ShellArtifact(ShellSourceValue("echo copied\n"), executable=True)
    assert audit.store.write("/tmp/source/package/runner.sh", artifact)
    result = audit(_command(("cp", "-a", "/tmp/source/package", "/tmp/destination/")))
    assert result is not None and result.status is True
    assert audit.store.read("/tmp/destination/package/runner.sh") == artifact
    assert (
        audit(
            _command(
                ("tar", "-cf", "/tmp/destination/package.tar", "-C", "/tmp/destination", "package")
            )
        )
        is not None
    )
    listing = audit(_command(("tar", "-tf", "/tmp/destination/package.tar")))
    assert (
        listing is not None
        and listing.status is True
        and listing.stdout is None
        and not listing.writes
    )


def test_unchecked_write_or_read_failure_cannot_publish_generated_source_bytes() -> None:
    audit = _audit()
    assert audit.store.write("/tmp/generated.sh", ShellArtifact(ShellSourceValue("echo old\n")))
    result = audit(
        _command(
            ("printf", "%s", "echo new\n"),
            fail_stop=False,
            redirects=(("1>", "/tmp/generated.sh"),),
        )
    )
    assert result is not None and audit.read_source("/tmp/generated.sh") is None
    read = audit(_command(("cat", "scripts/setup.sh"), fail_stop=False))
    assert read is not None and read.stdout is None and read.status is None


@pytest.mark.parametrize("writer", ["printf", "cp", "chmod", "tar"])
def test_unowned_external_destinations_never_acquire_independent_exact_bytes(writer: str) -> None:
    audit = _audit()
    assert audit.store.create_empty_directory("/tmp/fresh")
    assert audit.store.write("/tmp/fresh/package/setup.sh", audit.store.read("scripts/setup.sh"))
    assert audit.store.write("/tmp/existing-a", ShellArtifact(ShellSourceValue("echo old\n")))
    # Existing external paths can be aliases. A successful write alone cannot
    # prove either name is an independent regular file in a fresh namespace.
    if writer == "printf":
        command = _command(("printf", "%s", "echo new\n"), redirects=(("1>", "/tmp/existing-a"),))
    elif writer == "cp":
        command = _command(("cp", "scripts/setup.sh", "/tmp/existing-a"))
    elif writer == "chmod":
        command = _command(("chmod", "+x", "/tmp/existing-a"))
    else:
        command = _command(("tar", "-cf", "/tmp/existing-a", "-C", "/tmp/fresh", "package"))
    result = audit(command)
    assert result is not None and "/tmp/existing-a" in result.writes
    assert audit.read_source("/tmp/existing-a") is None
    assert audit.store.read("/tmp/existing-a") is None


def test_pid_log_write_invalidates_numeric_alias_without_creating_exact_source() -> None:
    audit = _audit()
    assert audit.store.create_empty_directory("/tmp/work")
    assert audit.store.write("/tmp/work/log.123", ShellArtifact(ShellSourceValue("echo old\n")))
    assert audit.store.write("/tmp/work/known.sh", ShellArtifact(ShellSourceValue("echo known\n")))
    path = "/tmp/work/log." + SHELL_PID_MARKER
    result = audit(_command(("printf", "%s", "echo new\n"), redirects=(("1>", path),)))
    assert result is not None and result.writes == (path,)
    assert audit.read_source(path) is None
    assert audit.read_source("/tmp/work/log.123") is None
    assert audit.read_source("/tmp/work/known.sh") == "echo known\n"
    assert audit.store.inventory("/tmp/work") is None


def test_pid_output_and_identity_never_become_literal_generated_facts() -> None:
    audit = _audit()
    assert audit.store.create_empty_directory("/tmp/work")
    result = audit(
        _command(("printf", "%s", SHELL_PID_MARKER), redirects=(("1>", "/tmp/work/generated.sh"),))
    )
    assert result is not None and result.stdout is None
    assert audit.read_source("/tmp/work/generated.sh") is None
    assert audit(_command(("/tmp/helper." + SHELL_PID_MARKER,))) is None
    directory = audit(_command(("cd", "/tmp/work/" + SHELL_PID_MARKER)))
    assert directory is not None and directory.status is None


def test_external_pid_logs_do_not_erase_an_unrelated_fresh_workspace() -> None:
    audit = _audit()
    assert audit.store.create_empty_directory("/tmp/work")
    assert audit.store.write("/tmp/work/known.sh", ShellArtifact(ShellSourceValue("echo known\n")))
    result = audit(
        _command(("printf", "%s", "log"), redirects=(("1>", "/tmp/log." + SHELL_PID_MARKER),))
    )
    assert result is not None
    assert audit.read_source("/tmp/work/known.sh") == "echo known\n"
    assert audit.store.inventory("/tmp/work") is not None
