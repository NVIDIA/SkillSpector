# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove matching-data roles in bounded, supported baseline documents."""

from __future__ import annotations

from collections.abc import Callable

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from skillspector.suppression import BASELINE_VERSION, baseline_from_dict
from skillspector.yaml_tokens import yaml_source_spans

_STRING = "tag:yaml.org,2002:str"
_MATCHERS = {"id", "rule_id", "path", "file", "message"}
_FINGERPRINT_FIELDS = {"hash", "rule_id", "file", "source_identity", "source_digest"}


def _mapping(node: Node | None) -> dict[str, Node] | None:
    if not isinstance(node, MappingNode):
        return None
    result: dict[str, Node] = {}
    for key, value in node.value:
        if not isinstance(key, ScalarNode) or key.tag != _STRING or key.value in result:
            return None
        result[key.value] = value
    return result


def _text(node: Node) -> bool:
    return isinstance(node, ScalarNode) and node.tag == _STRING and bool(node.value.strip())


def baseline_rule_scalar_roles(content: str, check_runtime: Callable[[], None]) -> dict[int, str]:
    """Own glob/fingerprint fields without applying any suppression rule.

    The full bounded YAML grammar rejects aliases and explicit tags first.
    Only the supported versioned schema grants roles; explanatory reasons and
    every raw/decoded security finding remain subject to ordinary inspection.
    """
    spans = yaml_source_spans(content, check_runtime)
    if spans is None or not any(scalar.value == "version" for scalar in spans.scalars):
        return {}
    check_runtime()
    loader = yaml.SafeLoader(content)
    try:
        node = loader.get_single_node()
        fields = _mapping(node)
        if (
            fields is None
            or "version" not in fields
            or not {"rules", "fingerprints"} & fields.keys()
        ):
            return {}
        if not fields.keys() <= {"version", "rules", "fingerprints", "scanner_version"}:
            return {}
        version = fields["version"]
        if not (
            isinstance(version, ScalarNode)
            and version.tag == "tag:yaml.org,2002:int"
            and version.value == str(BASELINE_VERSION)
        ):
            return {}
        roles: dict[int, str] = {}
        for name, allowed in (("rules", _MATCHERS), ("fingerprints", _FINGERPRINT_FIELDS)):
            if name not in fields:
                continue
            records = fields[name]
            if not isinstance(records, SequenceNode):
                return {}
            for record in records.value:
                check_runtime()
                values = _mapping(record)
                if values is None or not values.keys() <= allowed | {"reason"}:
                    return {}
                if not all(_text(value) for value in values.values()):
                    return {}
                if name == "rules" and (
                    {"id", "rule_id"} <= values.keys() or {"path", "file"} <= values.keys()
                ):
                    return {}
                for key in allowed & values.keys():
                    roles[values[key].start_mark.index] = "pattern"
        if "scanner_version" in fields and not _text(fields["scanner_version"]):
            return {}
        check_runtime()
        # Reuse the canonical field/value constraints, including required
        # reasons, record limits, fingerprint syntax and scanner version.
        baseline_from_dict(loader.construct_document(node))
        check_runtime()
        return roles
    except (yaml.YAMLError, ValueError, TypeError, RecursionError):
        return {}
    finally:
        loader.dispose()
