from types import ModuleType

import pytest
from pydantic import BaseModel

from skillspector.exaforce._patchlib import (
    PatchDriftError,
    pop_field_validator,
    remove_model_fields,
    replace_module_str,
)


def test_remove_model_fields_removes_then_rebuild_drops_key():
    class M(BaseModel):
        a: str
        b: str = ""

    remove_model_fields(M, ["b"])
    M.model_rebuild(force=True)
    assert "b" not in M.model_json_schema()["properties"]
    assert "a" in M.model_json_schema()["properties"]


def test_remove_model_fields_raises_on_missing_field():
    class M(BaseModel):
        a: str

    with pytest.raises(PatchDriftError):
        remove_model_fields(M, ["nope"])


def test_pop_field_validator_is_noop_when_absent():
    class M(BaseModel):
        a: str

    pop_field_validator(M, "nonexistent")  # must not raise


def test_replace_module_str_replaces_substring():
    mod = ModuleType("dummy")
    mod.P = "hello world"
    replace_module_str(mod, "P", "world", "there")
    assert mod.P == "hello there"


def test_replace_module_str_raises_on_missing_substring():
    mod = ModuleType("dummy")
    mod.P = "hello world"
    with pytest.raises(PatchDriftError):
        replace_module_str(mod, "P", "absent", "x")
