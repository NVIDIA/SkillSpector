# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight console entry point; load scan dependencies only when needed.

This only defers command loading. Static scans still use the normal workflow;
they are not a separate scanner or a provider-free execution path.
"""

from __future__ import annotations

from typing import Annotated

import click
import typer
from rich.console import Console
from typer.core import TyperGroup
from typer.main import get_group

from skillspector import __version__


class _LazyCommandGroup(TyperGroup):
    """Let eager root options finish before importing the command implementation."""

    def _load_commands(self) -> None:
        if not self.commands:
            from skillspector.cli import app as command_app

            # Reuse the real command definitions, including help, completion and
            # typo suggestions, rather than maintaining a second option schema.
            self.commands.update(get_group(command_app).commands)

    def list_commands(self, ctx: click.Context) -> list[str]:
        self._load_commands()
        return super().list_commands(ctx)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        self._load_commands()
        return super().get_command(ctx, cmd_name)


def version_callback(value: bool) -> None:
    """Print version and exit without importing the scanner or its providers."""
    if value:
        Console().print(f"SkillSpector v{__version__}")
        raise typer.Exit()


def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """
    SkillSpector - Security scanner for AI agent skills (LangGraph).

    Analyze skill bundles to detect vulnerabilities and security risks.
    Supports: Git URL, file URL, .zip file, .md file, or directory.
    """


def create_app(*, cls: type[TyperGroup] | None = None) -> typer.Typer:
    """Share root options and help between the console and implementation apps."""
    application = typer.Typer(
        name="skillspector",
        help="Security scanner for AI agent skills (LangGraph). Detect vulnerabilities before installation.",
        add_completion=False,
        no_args_is_help=True,
        cls=cls,
    )
    application.callback()(main)
    return application


app = create_app(cls=_LazyCommandGroup)
