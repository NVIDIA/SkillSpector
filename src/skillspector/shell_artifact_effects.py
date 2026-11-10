# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join a bounded shell effect audit to its virtual generated-file inventory.

Every fact comes from uploaded bytes or an explicitly modeled operation. Native
identity is checked before interpreting argv; uncertain branches invalidate their
destinations. This module never opens a sample path or executes a sample command.
"""

from __future__ import annotations

import hashlib
import posixpath
import re
import shlex
from collections.abc import Callable
from dataclasses import replace

from skillspector.shell_artifact_flow import (
    ShellArtifactEffects,
    ShellProducer,
    helper_stdout_artifacts,
)
from skillspector.shell_artifacts import ShellArtifact, ShellArtifactStore
from skillspector.shell_helper_argv import is_managed_shell_variable, reconstruct_helper_output
from skillspector.shell_source_effects import (
    SHELL_PID_MARKER,
    ShellEffectCommand,
    ShellEffectResult,
)
from skillspector.shell_sources import (
    SAFE_PATH_MARKER,
    ReadWord,
    ShellSourceContext,
    ShellSourceOrigin,
    ShellSourceValue,
    readonly_file_arguments,
    readonly_sed_program,
)

_BUILTINS = frozenset(
    {":", "true", "false", "set", "printf", "echo", "pwd", "cd", "command", "test", "umask", "wait"}
)
_READERS = frozenset(
    {
        "head",
        "tail",
        "grep",
        "egrep",
        "fgrep",
        "cut",
        "wc",
        "stat",
        "sha256sum",
        "sha1sum",
        "md5sum",
        "cmp",
        "diff",
        "readlink",
        "ls",
        "sleep",
        "id",
        "date",
        "kill",
        "df",
    }
)
_COMPRESSORS = frozenset({"gzip", "pigz", "bzip2", "lbzip2", "xz"})
_WRITER_FLAGS = {
    "rm": {"-f", "-r", "-R", "-rf", "-fr", "-v", "--"},
    "unlink": {"--"},
    "rmdir": {"-p", "--"},
    "touch": {"-a", "-m", "-c", "--"},
    "mv": {"-f", "-n", "-i", "-v", "-T", "--"},
    "ln": {"-s", "-f", "-n", "-sf", "-snf", "-sfn", "--"},
    "install": {"-d", "--"},
    "tee": {"-a", "-i", "--"},
    "truncate": {"--"},
    "mkfifo": {"--"},
    "chown": {"-R", "-h", "-f", "--"},
    "chgrp": {"-R", "-h", "-f", "--"},
}


class ShellArtifactAudit:
    """Stateful native-effect callback for one ordered source-preservation proof."""

    def __init__(
        self, context: ShellSourceContext, read_word: ReadWord, check_runtime: Callable[[], None]
    ) -> None:
        self.context = context
        self.read_word = read_word
        self.check_runtime = check_runtime
        self.store = ShellArtifactStore(check_runtime)
        self.effects = ShellArtifactEffects(self.store, check_runtime)
        self.temporary = 0
        self.steps = 0
        self.path_tail = SAFE_PATH_MARKER + "_PATH_TAIL"
        for path, source in context.sources.items():
            check_runtime()
            if (
                SAFE_PATH_MARKER in source
                or SHELL_PID_MARKER in source
                or SHELL_PID_MARKER in path
                or SHELL_PID_MARKER in context.path
            ):
                self.store.invalidate(None)
                break
            origin = ShellSourceOrigin(
                path, 0, len(source), hashlib.sha256(source.encode()).hexdigest()
            )
            if not self.store.write(
                path, ShellArtifact(ShellSourceValue(source, "code", (origin,)))
            ):
                break

    def path(self, value: str | None, cwd: str | None) -> str | None:
        if (
            value is None
            or not value
            or value.count(SHELL_PID_MARKER) > 1
            or any(item in value for item in ("\x00", "\\", ":"))
        ):
            return None
        if value.startswith("/"):
            result = posixpath.normpath(value)
        elif cwd is not None:
            result = posixpath.normpath(posixpath.join(cwd, value))
        else:
            return None
        return None if result in {"..", "/"} or result.startswith("../") else result

    def read_source(self, path: str) -> str | None:
        if SHELL_PID_MARKER in path:
            return None
        artifact = self.store.read(path)
        return artifact.value.text if artifact is not None and artifact.value is not None else None

    def preservation_state(self) -> tuple[object, ...]:
        """Include fresh-name allocation alongside current artifact facts."""
        return self.store.effect_snapshot(), self.temporary

    def _invalidate_path(self, path: str) -> bool:
        if SHELL_PID_MARKER in path:
            if path.count(SHELL_PID_MARKER) != 1:
                return False
            return self.store.invalidate_decimal_path(*path.split(SHELL_PID_MARKER))
        self.store.invalidate(path)
        return self.store.complete

    def identity(self, command: ShellEffectCommand, name: str) -> tuple[bool, str | None]:
        """Return native identity or an exact executable script selected by PATH."""
        if SHELL_PID_MARKER in name:
            return False, None
        if name in _BUILTINS:
            return True, None
        if "/" in name:
            basename = posixpath.basename(name)
            if name in {"/bin/" + basename, "/usr/bin/" + basename}:
                return True, None
            path = self.path(name, command.cwd)
            artifact = self.store.read(path) if path is not None else None
            if (
                artifact is not None
                and artifact.value is not None
                and (path in self.context.sources or artifact.executable is True)
            ):
                return False, path
            return False, None
        path_value = command.environment.get("PATH", self.path_tail)
        if path_value is None or SHELL_PID_MARKER in path_value:
            return False, None
        components = path_value.split(":")
        if not components or components[-1] != self.path_tail:
            return False, None
        for directory in components[:-1]:
            directory = self.path(directory, command.cwd)
            if directory is None or not self.store.owns(directory):
                return False, None
            candidate = directory + "/" + name
            artifact = self.store.read(candidate)
            if artifact is not None:
                if artifact.value is not None and artifact.executable is True:
                    return False, candidate
                # An unknown mode/type cannot be assumed absent from PATH.
                return False, None
            if self.store.exists(candidate) is not False:
                return False, None
        return True, None

    def _origin(self, command: ShellEffectCommand) -> tuple[ShellSourceOrigin, ...]:
        source = self.context.sources.get(command.source_path)
        if source is None:
            artifact = self.store.read(command.source_path)
            return (
                artifact.value.origins
                if artifact is not None and artifact.value is not None
                else ()
            )
        # A reached child can share its owner's source path while its local
        # offsets differ. Retain the whole uploaded owner rather than inventing
        # a byte-precise mapping for a nested command string.
        return (
            ShellSourceOrigin(
                command.source_path, 0, len(source), hashlib.sha256(source.encode()).hexdigest()
            ),
        )

    def _redirect(
        self, command: ShellEffectCommand, stdout: tuple[ShellArtifact, ...] | None
    ) -> tuple[str, ...] | None:
        writes = []
        used_stdout = False
        file_outputs = [
            (operator, value)
            for operator, value in command.redirects
            if operator.startswith("1>") and operator != "1>&" and value != "/dev/null"
        ]
        if (
            len(file_outputs) > 1
            or file_outputs
            and any(operator in {"1>&", "2>&"} for operator, _ in command.redirects)
        ):
            # Ordered descriptor duplication needs a separate stream model.
            # Account for every write while withholding exact generated bytes.
            stdout = None
        for operator, value in command.redirects:
            self.check_runtime()
            if operator in {"1>&", "2>&", "0<&"} and value in {"0", "1", "2", "-"}:
                continue
            if operator == "0<":
                continue
            if operator not in {"1>", "1>>", "2>", "2>>"}:
                return None
            if value == "/dev/null":
                continue
            destination = self.path(value, command.cwd)
            if destination is None:
                return None
            writes.append(destination)
            # Preexisting external paths may alias one another. Only a fresh
            # namespace can give a newly written file independent exact bytes.
            if (
                operator.startswith("1")
                and not used_stdout
                and command.fail_stop
                and not command.uncertain
                and stdout is not None
                and SHELL_PID_MARKER not in destination
                and self.store.owns(destination)
            ):
                used_stdout = True
                if not self.effects.redirect_stdout(
                    destination, stdout, append=operator.endswith(">>")
                ):
                    return None
            else:
                if not self._invalidate_path(destination):
                    return None
        return tuple(writes)

    def _helper_output(
        self,
        command: ShellEffectCommand,
        path: str,
        argv: tuple[str, ...],
        store: ShellArtifactStore,
        writes: set[str],
        writable: frozenset[str],
    ) -> tuple[ShellArtifact, ...] | None:
        artifact = store.read(path)
        if artifact is None or artifact.value is None:
            return None
        # The finite output interpreter only has these native operations. Prove
        # their external identity under the exact helper PATH before reusing it.
        if not self.identity(command, "cat")[0]:
            return None
        environment = {
            name: value
            for name, value in command.environment.items()
            if not is_managed_shell_variable(name)
        }
        written: set[str] = set()
        variants = reconstruct_helper_output(
            artifact.value.text,
            argv[1:],
            environment,
            writable,
            self.read_word,
            self.check_runtime,
            written_paths=written,
        )
        if variants is None:
            return None
        for destination in written:
            store.invalidate(destination)
        writes.update(written)
        return helper_stdout_artifacts(variants, store, artifact.value.origins, self.check_runtime)

    def _producer(self, command: ShellEffectCommand, path: str) -> ShellEffectResult | None:
        # A producer is an optional proof for one explicit output, never an
        # exemption for a file name. Other known scripts use the caller's audit.
        if (
            command.uncertain
            or not command.fail_stop
            or command.redirects
            or len(command.argv) != 2
            or any(
                SHELL_PID_MARKER in value
                for value in (*command.argv, *command.environment.values())
                if isinstance(value, str)
            )
        ):
            return None
        destination = self.path(command.argv[1], command.cwd)
        if destination is None or not self.store.owns(destination):
            return None
        source = self.read_source(path)
        if source is None:
            return None
        trial = self.store.fork()
        helper_writes: set[str] = set()
        # These locations were proved before entering the finite producer. Its
        # permitted helper writes create ordinary log bytes, never path aliases.
        # Repeated appends remain permitted after their bytes become unknown.
        writable = frozenset(
            value
            for value in command.environment.values()
            if isinstance(value, str) and value.startswith("/") and self.store.owns(value)
        )

        def output(
            argv: tuple[str, ...], environment: dict[str, str | None]
        ) -> tuple[ShellArtifact, ...] | None:
            # The finite producer cannot mutate PATH, but inherited exported
            # data variables may change before it launches a helper.
            helper_environment = {**environment}
            if "PATH" in command.environment:
                helper_environment["PATH"] = command.environment["PATH"]
            helper_command = replace(command, environment=helper_environment)
            _, helper = self.identity(helper_command, argv[0])
            return (
                self._helper_output(helper_command, helper, argv, trial, helper_writes, writable)
                if helper is not None
                else None
            )

        # Producer syntax also contains native names inside specialized forms.
        for name in (
            "dirname",
            "awk",
            "find",
            "sort",
            "xargs",
            "sha256sum",
            "cp",
            "tar",
            "mkdir",
            "chmod",
            "mktemp",
            "rm",
            *_COMPRESSORS,
        ):
            if not self.identity(command, name)[0]:
                return None
        environment = {
            name: value
            for name, value in command.environment.items()
            if not is_managed_shell_variable(name)
        }
        context = ShellSourceContext(path, {**self.context.sources, path: source})
        result = ShellProducer(trial, context, self.read_word, self.check_runtime, output).inspect(
            (destination,), environment, (destination,), initial_cwd=command.cwd
        )
        if result is None or not self.store.adopt(trial):
            return None
        return ShellEffectResult(
            status=True, writes=tuple(sorted(set(result.writes) | helper_writes))
        )

    def preserve(self, command: ShellEffectCommand) -> ShellEffectResult | None:
        """Audit unknown output as data without publishing generated bytes."""
        if (
            command.argv[:1] == ("printf",)
            and len(command.argv) >= 3
            and any(value is None for value in command.argv[2:])
            and not command.child_inspected
            and not command.identity_only
            and not command.redirect_open
            and command.interpreter is None
        ):
            self.check_runtime()
            self.steps += 1
            format_string = command.argv[1]
            if (
                not self.store.complete
                or self.steps > 65_536
                or len(command.argv) > 128
                or any(
                    is_managed_shell_variable(name) and name != "PATH"
                    for name in command.environment
                )
                or format_string is None
                or format_string.startswith("-")
                or re.fullmatch(r"(?:[^%\\]|%%|%[sq]|\\[nrt\\])*", format_string) is None
            ):
                return None
            # The walker resolved functions before this native builtin call.
            # Unknown printf arguments cannot supply a format, path, or code.
            writes = self._redirect(command, None)
            return (
                ShellEffectResult(writes=writes, native_identity=True)
                if writes is not None
                else None
            )
        return self(command)

    def __call__(self, command: ShellEffectCommand) -> ShellEffectResult | None:
        self.check_runtime()
        if (
            not self.store.complete
            or not command.argv
            or len(command.argv) > 128
            or command.argv[0] is None
        ):
            return None
        if any(is_managed_shell_variable(name) and name != "PATH" for name in command.environment):
            return None
        if command.interpreter is not None:
            if command.interpreter not in {"bash", "/bin/bash", "/usr/bin/bash"}:
                return None
            native_shell, shadow = self.identity(command, command.interpreter)
            if not native_shell or shadow is not None:
                return None
            script = self.path(command.argv[0], command.cwd)
            if script is None or self.read_source(script) is None:
                return None
            # An explicit interpreter reads a script regardless of its mode.
            # The script itself must never acquire native-command identity.
            native = False
        else:
            native, script = self.identity(command, command.argv[0])
        if command.identity_only:
            return (
                ShellEffectResult(native_identity=native, script_path=script)
                if native or script is not None
                else None
            )
        self.steps += 1
        if self.steps > 65_536:
            return None
        if command.redirect_open:
            redirects = self._redirect(
                command, (ShellArtifact(ShellSourceValue("", "code", self._origin(command))),)
            )
            return ShellEffectResult("", True, redirects) if redirects is not None else None
        if script is not None and not command.child_inspected:
            produced = self._producer(command, script)
            if produced is not None:
                return produced
            # A child requires its stdout/effects before committing redirects.
            # The walker sends a child_inspected completion call afterward.
            return ShellEffectResult(script_path=script)
        if not native and not (script is not None and command.child_inspected):
            return None
        name = posixpath.basename(command.argv[0])
        args = command.argv[1:]
        output: str | None = command.stdout if command.child_inspected else None
        artifacts: tuple[ShellArtifact, ...] | None = None
        writes: list[str] = []
        status: bool | None = True
        if command.child_inspected:
            status = command.child_status
        elif name in {":", "true", "false"} and not args:
            output, status = "", name != "false"
        elif name == "set" and args in {
            ("-euo", "pipefail"),
            ("-Eeuo", "pipefail"),
            ("-e",),
            ("+e",),
            ("-a",),
            ("+a",),
        }:
            output = ""
        elif name == "umask" and len(args) == 1 and args[0] in {"077", "022"}:
            output = ""
        elif name == "pwd" and not args:
            output = None if command.cwd is None else command.cwd + "\n"
        elif name == "cd" and len(args) == 1:
            path = self.path(args[0], command.cwd)
            if path is not None and SHELL_PID_MARKER in path:
                path = None
            known_directory = path is not None and (
                path == "." or any(source.startswith(path + "/") for source in self.context.sources)
            )
            exists = self.store.exists(path) if path is not None else None
            # Leaving the uploaded tree cannot establish a host-directory fact.
            # Preserve an unknown cwd in the walker so later relative effects
            # still refuse, while an unrelated path computation can continue.
            artifact = self.store.read(path) if path is not None else None
            if artifact is not None or exists is False:
                status = False
            elif command.fail_stop and (known_directory or exists is True):
                status = True
            else:
                status = None
            output = ""
        elif (
            name == "dirname"
            and len(args) == 1
            and args[0] is not None
            and not args[0].startswith("-")
        ):
            output = posixpath.dirname(args[0]) + "\n" if posixpath.dirname(args[0]) else ".\n"
        elif name == "echo" and all(
            value is not None and not value.startswith("-") and "\\" not in value for value in args
        ):
            output = " ".join(args) + "\n"
        elif name == "echo":
            # Native echo only emits output. Unsupported options and escapes
            # cannot establish exact bytes, but still have bounded effects.
            output, status = None, None
        elif name == "printf" and all(value is not None for value in args):
            variants = reconstruct_helper_output(
                shlex.join(("printf", *args)),
                (),
                {},
                frozenset(),
                self.read_word,
                self.check_runtime,
            )
            artifacts = (
                helper_stdout_artifacts(
                    variants, self.store, self._origin(command), self.check_runtime
                )
                if variants is not None
                else None
            )
            if artifacts is None or len(artifacts) != 1 or artifacts[0].value is None:
                return None
            output = artifacts[0].value.text
        elif name == "cat" and not args:
            output = command.stdin
        elif name == "cat" and all(
            value is not None and not value.startswith("-") for value in args
        ):
            values = [
                self.store.read(self.path(value, command.cwd) or "")
                if SHELL_PID_MARKER not in value
                else None
                for value in args
            ]
            artifacts = tuple(values) if len(values) == 1 and values[0] is not None else None
            output = (
                "".join(item.value.text for item in values)
                if all(item is not None and item.value is not None for item in values)
                else None
            )
            if output is not None:
                origins = tuple(
                    dict.fromkeys(origin for item in values for origin in item.value.origins)
                )
                artifacts = (ShellArtifact(ShellSourceValue(output, "code", origins)),)
            if not command.fail_stop:
                output, artifacts, status = None, None, None
        elif (
            name == "mktemp"
            and command.fail_stop
            and not command.uncertain
            and (len(args) == 1 or len(args) == 2 and args[0] == "-d")
        ):
            template = args[-1]
            if template is None or re.fullmatch(r"/tmp/[-A-Za-z0-9_.]+", template) is None:
                return None
            runs = tuple(re.finditer(r"X+", template))
            if not runs or len(runs[-1][0]) < 3:
                return None
            # Native mktemp replaces only the final X run. A regular file
            # owns its exact fresh identity, never its surrounding directory.
            directory = len(args) == 2
            self.temporary += 1
            marker = (
                SAFE_PATH_MARKER if directory else SAFE_PATH_MARKER + "_FILE_" + str(self.temporary)
            )
            path = template[: runs[-1].start()] + marker + template[runs[-1].end() :]
            created = (
                self.store.create_empty_directory(path)
                if directory
                else self.store.create_empty_file(
                    path,
                    ShellSourceValue("", "code", self._origin(command)),
                )
            )
            if not created:
                return None
            writes.append(path)
            output = path + "\n"
        elif name == "mkdir" and args[:1] == ("-p",) and len(args) > 1:
            for value in args[1:]:
                path = self.path(value, command.cwd)
                if path is None:
                    return None
                writes.append(path)
                if command.uncertain or not command.fail_stop or SHELL_PID_MARKER in path:
                    if not self._invalidate_path(path):
                        return None
                    status = None
                elif (
                    self.store.inventory(path) is None
                    or self.store.exists(path) is False
                    and not self.store.create_empty_directory(path)
                ):
                    return None
            output = ""
        elif (
            name == "chmod"
            and len(args) >= 2
            and args[0] in {"+x", "-x", "000", "600", "644", "700", "755"}
        ):
            for value in args[1:]:
                path = self.path(value, command.cwd)
                artifact = self.store.read(path) if path is not None else None
                if path is None:
                    return None
                writes.append(path)
                if (
                    command.uncertain
                    or not command.fail_stop
                    or artifact is None
                    or SHELL_PID_MARKER in path
                    or not self.store.owns(path)
                ):
                    if not self._invalidate_path(path):
                        return None
                    status = None
                elif not self.store.write(
                    path, replace(artifact, executable=args[0] in {"+x", "700", "755"})
                ):
                    return None
            output = ""
        elif (
            name == "cp"
            and len(args) == 2
            and all(value is not None and not value.startswith("-") for value in args)
        ):
            source, destination = (self.path(value, command.cwd) for value in args)
            if source is None or destination is None:
                return None
            writes.append(destination)
            if (
                command.uncertain
                or not command.fail_stop
                or SHELL_PID_MARKER in destination
                or not self.store.owns(destination)
            ):
                if not self._invalidate_path(destination):
                    return None
                status = None
            else:
                artifact = self.store.read(source)
                previous = self.store.read(destination)
                if artifact is not None and previous is not None:
                    # Plain cp replaces bytes but retains an existing file's
                    # permissions, including an unknown executable state.
                    artifact = replace(artifact, executable=previous.executable)
                if (
                    SHELL_PID_MARKER in source
                    or artifact is None
                    or not self.store.write(destination, artifact)
                ):
                    # A known disjoint destination may still receive unknown data.
                    self.store.invalidate(destination)
                    status = None
            output = ""
        elif (
            name == "cp"
            and len(args) == 3
            and args[0] == "-a"
            and all(value is not None and SHELL_PID_MARKER not in value for value in args)
            and not args[1].endswith(("/", "/."))
        ):
            source, destination = self.path(args[1], command.cwd), self.path(args[2], command.cwd)
            if source is None or destination is None:
                return None
            inventory = self.store.inventory(source)
            directories = self.store.directory_inventory(source)
            if (
                inventory is None
                or directories is None
                or self.store.exists(destination) is not True
                or self.store.read(destination) is not None
            ):
                return None
            target = destination + "/" + posixpath.basename(source)
            writes.append(target)
            if command.uncertain or not command.fail_stop or not self.store.owns(destination):
                self.store.invalidate(target)
                status = None
            elif not self.store.create_empty_directory(target):
                return None
            else:
                for relative in directories:
                    if not self.store.create_empty_directory(target + "/" + relative):
                        self.store.invalidate(target)
                        return None
                for relative, artifact in inventory:
                    self.check_runtime()
                    if not self.store.write(target + "/" + relative, artifact):
                        return None
            output = ""
        elif name == "tar" and all(
            value is not None and SHELL_PID_MARKER not in value for value in args
        ):
            if args[:1] == ("-I",) and (
                len(args) < 2
                or args[1] not in _COMPRESSORS
                or not self.identity(command, args[1])[0]
            ):
                return None
            operands = args[2:] if args[:1] == ("-I",) else args
            if len(operands) == 2 and operands[0] == "-tf":
                artifact = self.store.read(self.path(operands[1], command.cwd) or "")
                status = True if artifact is not None and artifact.is_archive else None
                # The inventory does not establish header order or native
                # listing details. Do not invent stdout from those facts.
                output = None
            elif command.uncertain or not command.fail_stop:
                return None
            else:
                if (
                    len(operands) not in {4, 5}
                    or operands[0] not in {"-cf", "-xf"}
                    or operands[2] != "-C"
                ):
                    return None
                archive, directory = (
                    self.path(operands[1], command.cwd),
                    self.path(operands[3], command.cwd),
                )
                if archive is None or directory is None:
                    return None
                destination = archive if operands[0] == "-cf" else directory
                if not self.store.owns(destination):
                    self.store.invalidate(destination)
                    status = None
                else:
                    normalized = (
                        (*args[:2], operands[0], archive, "-C", directory, *operands[4:])
                        if args[:1] == ("-I",)
                        else (operands[0], archive, "-C", directory, *operands[4:])
                    )
                    if not self.effects.tar(("tar", *normalized)):
                        return None
                writes.append(destination)
                output = ""
        elif name == "test" and len(args) == 2 and args[0] in {"-e", "-f", "-s", "-x", "-d"}:
            path = self.path(args[1], command.cwd)
            if path is not None and SHELL_PID_MARKER in path:
                path = None
            artifact = self.store.read(path) if path is not None else None
            exists = self.store.exists(path) if path is not None else None
            if args[0] == "-e":
                status = exists
            elif args[0] == "-x":
                status = (
                    artifact.executable
                    if artifact is not None
                    else False
                    if exists is False
                    else None
                )
            elif args[0] == "-f":
                status = True if artifact is not None else False if exists is False else None
            elif args[0] == "-d":
                status = False if artifact is not None or exists is False else exists
            else:
                status = (
                    bool(artifact.value.text)
                    if artifact is not None and artifact.value is not None
                    else False
                    if exists is False
                    else None
                )
            output = ""
        elif name == "command" and len(args) == 2 and args[0] == "-v" and args[1] is not None:
            found, selected = self.identity(command, args[1])
            # The inherited PATH tail establishes which native semantics may
            # apply; it does not prove a particular executable is installed.
            status = True if selected is not None or found and args[1] in _BUILTINS else None
            output = (selected or args[1]) + "\n" if status else None
        elif name in _WRITER_FLAGS:
            if any(value is None for value in args):
                return None
            # Flags carrying paths (including attached/long option values)
            # cannot be dropped while collecting possible write destinations.
            if any(value.startswith("-") and value not in _WRITER_FLAGS[name] for value in args):
                return None
            paths = [self.path(value, command.cwd) for value in args if not value.startswith("-")]
            if not paths or any(path is None for path in paths):
                return None
            for path in paths:
                if not self._invalidate_path(path):
                    return None
                writes.append(path)
            status = None
        elif name == "file":
            if not readonly_file_arguments(args, self.check_runtime):
                return None
            status = None
        elif name in _READERS or name == "wait":
            status = None
        elif (
            name == "awk"
            and len(args) == 1
            and args[0] is not None
            and re.fullmatch(
                r"[ \t]*NR[ \t]*==[ \t]*2[ \t]*\{[ \t]*print[ \t]+\$1[ \t]*;?[ \t]*\}[ \t]*",
                args[0],
            )
        ):
            # This bounded stdin filter has no file or command effects. Its
            # input is unknown; do not invent a disk-space value or success.
            status = None
        elif (
            name == "sed"
            and len(args) == 3
            and args[0] == "-n"
            and args[1] is not None
            and args[2] is not None
            and not args[2].startswith("-")
            and readonly_sed_program(args[1])
        ):
            # The shared grammar excludes writes and execution, including
            # substitution flags. Output stays unknown in the effects audit.
            status = None
        elif (
            name in {"bash", "sh"}
            and len(args) == 2
            and args[0] == "-n"
            and self.read_source(self.path(args[1], command.cwd) or "") is not None
        ):
            status = None
        else:
            return None
        if output is not None and SHELL_PID_MARKER in output:
            output, artifacts = None, None
        if artifacts is None and output is not None:
            artifacts = (ShellArtifact(ShellSourceValue(output, "code", self._origin(command))),)
        redirects = self._redirect(command, artifacts)
        if redirects is None:
            return None
        return ShellEffectResult(
            output,
            status,
            tuple(dict.fromkeys((*writes, *redirects))),
            native_identity=not command.child_inspected,
        )
