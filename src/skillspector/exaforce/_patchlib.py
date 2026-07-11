# SPDX-License-Identifier: Apache-2.0
"""Guarded monkeypatch primitives for the ExaForce fork.

Each primitive asserts its upstream target still exists before mutating it, so
an upstream rename/rewrite fails loudly (``PatchDriftError``) at import time
instead of silently leaving the fork's runtime behavior stale.
"""

from __future__ import annotations

from types import ModuleType

from pydantic import BaseModel


class PatchDriftError(RuntimeError):
    """An upstream target a fork patch depends on has changed or is missing."""


def remove_model_fields(model: type[BaseModel], field_names: list[str]) -> None:
    """Delete ``field_names`` from ``model.model_fields``.

    The caller must run ``model.model_rebuild(force=True)`` afterward (and on any
    container model that nests ``model``) for the change to reach the emitted
    JSON schema. Raises ``PatchDriftError`` if a field is absent.
    """
    for name in field_names:
        if name not in model.model_fields:
            raise PatchDriftError(
                f"{model.__module__}.{model.__qualname__} has no field {name!r}; "
                "upstream changed — update the exaforce patch."
            )
        del model.model_fields[name]


def pop_field_validator(model: type[BaseModel], validator_name: str) -> None:
    """Remove a ``field_validator`` decorator by its function name (no-op if absent)."""
    model.__pydantic_decorators__.field_validators.pop(validator_name, None)


def replace_module_str(module: ModuleType, attr: str, old: str, new: str) -> None:
    """Replace substring ``old`` with ``new`` in module-global ``attr``.

    Raises ``PatchDriftError`` if ``old`` is not present in the current value.
    """
    current = getattr(module, attr)
    if old not in current:
        raise PatchDriftError(
            f"{module.__name__}.{attr} does not contain the expected text; "
            "upstream changed — update the exaforce patch."
        )
    setattr(module, attr, current.replace(old, new))
